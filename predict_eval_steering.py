"""Main entrypoint script to train HyperDAS models and derivatives."""

import gc
import os
import random
from enum import Enum
from pathlib import Path
from typing import Dict, List

import httpx
import hydra
import pandas as pd
import torch
import torch.distributed as dist
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig
from openai import AsyncOpenAI
from tqdm import tqdm

from axbench.models.sae import load_metadata_flatten
from axbench.scripts.evaluate import eval_steering
from axbench.scripts.inference import (
    METADATA_FILE,
    create_data_steering,
    load_state,
    partition_concept_ids,
    save,
    save_state,
)
from axbench.utils.constants import CHAT_MODELS
from axbench.utils.dataset import SteeringDatasetFactory
from axbench.utils.model_utils import get_prefix_length
from logger import get_logger
from src.common.utils import setup_tokenizer
from src.hyperdas.data.axbench import get_axbench_collate_fn
from src.hyperdas.llama3.model import (
    SteeringInterpretorHypernetwork,
)
from src.hyperdas.utils import calculate_perplexity

logger = get_logger(__name__)


class InferenceModes(str, Enum):
    PROMPT_STEERING = "PromptSteering"
    HYPERREFT = "HyperReFT"

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)


@torch.no_grad()
def predict_steering(
    model,
    target_model_tokenizer,
    hypernet_tokenizer,
    concept_df: pd.DataFrame,
    device: str,
    mode: InferenceModes,
    batch_size: int = 16,
    eval_output_length: int = 100,
    temperature: float = 0.7,
    do_sample: bool = True,
    intervention_positions: str = None,
    intervention_layers: List[int] = None,
    **kwargs,
) -> Dict[str, float]:
    """
    Generate completions with steering or baseline mode.
    Returns dict with generations and metrics for the specified mode.
        interpretor: The model interpretor for generation
        target_model_tokenizer: Tokenizer for the target model
        hypernet_tokenizer: Tokenizer for the hypernetwork arch
        test_dataloader: DataLoader containing test data
        device: Device to run inference on
        mode: an InferenceModes enum value
        max_eval_steps: Maximum number of eval steps (-1 for no limit)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling for generation
        steering_prompt: Optional prompt to guide the steering (only used in steering mode)
        intervention_positions: Intervention positions
        intervention_layers: Intervention layers
    """
    all_generations = []
    all_perplexities = []

    for step in tqdm(range(0, len(concept_df), batch_size), desc="eval steps"):
        raw_batch = concept_df.iloc[step : step + batch_size]
        converted_batch = list(map(lambda x: x._asdict(), raw_batch.itertuples()))

        # compute batch from raw batch
        batch = get_axbench_collate_fn(
            hypernet_tokenizer,
            target_model_tokenizer,
            mode="steering_eval",
            intervention_layers=intervention_layers,
            intervention_positions=intervention_positions,
        )(converted_batch)

        # Move batch to GPU
        batch = {k: v.to(device) for k, v in batch.items()}

        # Generate outputs based on mode
        if mode == InferenceModes.PROMPT_STEERING:
            outputs = model.generate(
                input_ids=batch["editor_input_ids"],
                base_input_ids=batch.get("base_input_ids"),
                base_attention_mask=batch.get("base_attention_mask"),
                base_intervention_mask=batch.get("base_intervention_mask"),
                target_input_ids=batch.get("target_input_ids"),
                target_attention_mask=batch.get("target_attention_mask"),
                target_position_ids=batch.get("target_position_ids"),
                intervention_layers=batch.get("intervention_layers"),
                intervention_positions=batch.get("intervention_positions"),
                max_new_tokens=eval_output_length,
                do_sample=do_sample,
                temperature=temperature,
                do_intervention=True,
            )
        elif mode == InferenceModes.HYPERREFT:
            outputs = model.generate(
                input_ids=batch["editor_input_ids"],
                target_input_ids=batch.get("target_input_ids"),
                target_attention_mask=batch.get("target_attention_mask"),
                target_position_ids=batch.get("target_position_ids"),
                intervention_layers=batch.get("intervention_layers"),
                intervention_positions=batch.get("intervention_positions"),
                max_new_tokens=eval_output_length,
                do_sample=do_sample,
                temperature=temperature,
                do_intervention=False,
            )

        # Get input lengths to truncate prompts
        input_lengths = [len(ids) for ids in batch["target_input_ids"]]

        # Extract only the generated portions
        generations = [
            target_model_tokenizer.decode(output[length:], skip_special_tokens=True)
            for output, length in zip(outputs, input_lengths)
        ]

        # Calculate perplexity
        gen_ids = target_model_tokenizer(
            generations, return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(device)
        gen_mask = (gen_ids != target_model_tokenizer.pad_token_id).float()

        model_outputs = model.interpretor.target_model(
            input_ids=gen_ids, attention_mask=gen_mask
        )

        perplexities = calculate_perplexity(model_outputs.logits, gen_ids, gen_mask)

        # Store results
        all_generations.extend(generations)
        all_perplexities.extend(perplexities.tolist())

    return {
        f"{mode}_steered_generation": all_generations,
        f"{mode}_perplexity": all_perplexities,
    }


def infer_steering(config, rank, world_size, device, logger):
    logger.info(f"Starting inference for rank {rank} out of {world_size}")

    data_dir = config.axbench.inference.data_dir
    dump_dir = config.axbench.inference.dump_dir
    num_of_examples = config.axbench.inference.steering_num_of_examples
    metadata = load_metadata_flatten(Path(data_dir) / METADATA_FILE)
    steering_factors = config.axbench.inference.steering_factors
    steering_datasets = config.axbench.inference.steering_datasets

    logger.debug(f"Loaded metadata with {len(metadata)} examples")

    state = load_state(config.axbench.inference.dump_dir, "steering", rank)
    last_concept_id_processed = state.get("last_concept_id", None) if state else None
    logger.warning(
        f"Rank {rank} last concept_id processed: {last_concept_id_processed}"
    )

    # Get list of all concept_ids
    concept_ids = [metadata[i]["concept_id"] for i in range(len(metadata))]

    # Partition concept_ids among ranks sequentially
    concept_ids_per_rank = partition_concept_ids(concept_ids, world_size)
    my_concept_ids = concept_ids_per_rank[rank]

    if last_concept_id_processed is not None:
        if last_concept_id_processed in my_concept_ids:
            idx = my_concept_ids.index(last_concept_id_processed)
            my_concept_ids = my_concept_ids[idx + 1 :]
        else:
            # If last_concept_id_processed is not in my_concept_ids, process all
            pass

    if len(my_concept_ids) == 0:
        logger.warning(f"Rank {rank} has no concepts to process. Exiting.")
        return

    logger.info(f"Rank {rank} will process {len(my_concept_ids)} concepts")

    # Create a new OpenAI client.
    logger.debug("Initializing OpenAI client")
    lm_client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=1000),
            headers={"Connection": "close"},
        ),
        max_retries=3,
    )

    # Initialize the dataset factory with the tokenizer.
    logger.debug("Setting up tokenizers")
    tokenizer = setup_tokenizer(
        config.model.target_model_name_or_path, max_length=config.model.max_length
    )
    # Hypernet tokenizer
    hypernet_tokenizer = setup_tokenizer(
        model_name_or_path=config.model.name_or_path, max_length=config.model.max_length
    )

    if "PromptSteering" in config.axbench.evaluate.models:
        has_prompt_steering = True

    logger.debug("Initializing dataset factory")
    dataset_factory = SteeringDatasetFactory(
        tokenizer,
        dump_dir,
        master_data_dir=config.axbench.inference.master_data_dir,
        lm_client=lm_client,
        lm_model=config.axbench.inference.lm_model,
        has_prompt_steering=has_prompt_steering,
    )
    is_chat_model = (
        True if config.axbench.inference.model_name in CHAT_MODELS else False
    )
    prefix_length = 1  # prefix is default to 1 for all models due to the BOS token.
    if is_chat_model:
        prefix_length = get_prefix_length(tokenizer)
        logger.warning(f"Chat model prefix length: {prefix_length}")

    # Load model instance onto device
    logger.info("Loading model instance")
    model_instance = SteeringInterpretorHypernetwork(config, device)

    if config.training.load_trained_from is not None:
        logger.info(f"Loading model from {config.training.load_trained_from}")
        model_instance.load_model(config.training.load_trained_from)

    model_instance.eval()
    logger.info("Model loaded and set to eval mode")

    # Prepare data per concept
    logger.debug("Preparing data per concept")
    data_per_concept = {}
    for concept_id in my_concept_ids:
        current_df, (_, sae_link, sae_id) = create_data_steering(
            dataset_factory,
            metadata,
            concept_id,
            num_of_examples,
            steering_factors,
            steering_datasets,
            None,
        )
        data_per_concept[concept_id] = (current_df, sae_link, sae_id)
    logger.info(f"Prepared data for {len(data_per_concept)} concepts")

    # Now loop over concept_ids and use preloaded models
    for concept_id in my_concept_ids:
        logger.info(f"Processing concept {concept_id}")
        current_df, sae_link, sae_id = data_per_concept[concept_id]
        for model_name in config.axbench.inference.models:
            logger.debug(f"Running prediction for {model_name} mode")
            # Run prediction
            results = predict_steering(
                model_instance,
                tokenizer,
                hypernet_tokenizer,
                current_df,
                device,
                concept_id=concept_id,
                mode=model_name,
                batch_size=config.axbench.inference.steering_batch_size,
                eval_output_length=config.axbench.inference.steering_output_length,
                temperature=config.axbench.inference.temperature,
                prefix_length=prefix_length,
                intervention_positions=config.model.intervention_positions,
                intervention_layers=config.model.intervention_layer,
            )
            # Store the results in current_df
            for k, v in results.items():
                current_df[k] = v

        save(dump_dir, "steering", current_df, rank)
        logger.warning(
            f"Saved inference results for concept {concept_id} to rank_{rank}_steering_data.parquet"
        )
        # After processing, save state
        current_state = {"last_concept_id": concept_id}
        save_state(config.axbench.inference.dump_dir, current_state, "generate", rank)

    logger.info(f"Rank {rank} completed all concepts, cleaning up")
    del model_instance
    torch.cuda.empty_cache()

    # Synchronize all processes
    logger.debug("Waiting for all processes to complete")
    if dist.is_initialized():
        dist.barrier()

    # Rank 0 merges results
    if rank == 0:
        logger.warning("Rank 0 is merging results.")
        # Merge per-rank results
        all_parquet_files = list(
            (Path(dump_dir) / "inference").glob("rank_*_steering_data.parquet")
        )
        # Parse filenames to extract rank
        import re

        pattern = re.compile(r"rank_(\d+)_steering_data\.parquet")

        file_info_list = []
        for parquet_file in all_parquet_files:
            match = pattern.match(parquet_file.name)
            if match:
                rank_str = match.group(1)
                rank_int = int(rank_str)
                file_info_list.append({"rank": rank_int, "file": parquet_file})
            else:
                logger.warning(
                    f"Filename {parquet_file.name} does not match the expected pattern."
                )

        # Sort the file_info_list by rank
        file_info_list.sort(key=lambda x: x["rank"])
        logger.info(f"Found {len(file_info_list)} rank files to merge")

        # Read and concatenate dataframes
        dfs = []
        for info in file_info_list:
            logger.debug(f"Reading file from rank {info['rank']}")
            df = pd.read_parquet(info["file"])
            dfs.append(df)
        if len(dfs) > 0:
            logger.info("Merging and saving combined results")
            combined_df = pd.concat(dfs, ignore_index=True)
            # Optionally sort combined_df by 'concept_id' if needed
            combined_df = combined_df.sort_values(
                by=["concept_id", "input_id", "factor"]
            ).reset_index(drop=True)
            combined_df.to_parquet(
                Path(dump_dir) / "inference" / "steering_data.parquet", engine="pyarrow"
            )
            logger.warning(
                f"Saved combined steering inference results to {Path(dump_dir) / 'inference' / 'steering_data.parquet'}"
            )
        else:
            logger.warning("No results to merge.")

        # Optionally, delete per-rank files
        logger.debug("Cleaning up rank-specific files")
        for info in file_info_list:
            os.remove(info["file"])
            logger.warning(f"Deleted {info['file']}")


def run_inference(cfg: DictConfig, device: str | torch.DeviceObjType = "cuda"):
    infer_steering(
        cfg,
        0,
        1,
        device,
        get_logger("inference"),
    )


def run_eval(cfg: DictConfig):
    args = cfg.axbench.evaluate
    eval_steering(args)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    try:
        logger.info(f"Working directory : {os.getcwd()}")
        logger.info(f"Original working directory    : {get_original_cwd()}")
        logger.info(f"to_absolute_path('foo')       : {to_absolute_path('foo')}")
        logger.info(f"to_absolute_path('/foo')      : {to_absolute_path('/foo')}")

        # Check if we're running in serial mode
        is_serial = os.environ.get("LAUNCH_MODE", "parallel") == "serial"

        if cfg.device_mode is None:
            cfg.device_mode = (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        if cfg.device_mode == "mps":
            device = "mps"
        else:
            if is_serial:
                # Use single GPU for serial mode
                device = "cuda:0"
            elif cfg.device_mode == "cuda":
                # Use distributed GPUs for parallel mode
                num_gpus = torch.cuda.device_count()
                try:
                    job_num = getattr(HydraConfig.get().job, "num", 0)
                except Exception:
                    job_num = 0
                gpu_id = job_num % num_gpus
                device = f"cuda:{gpu_id}"
            else:
                device = "cpu"

        logger.info(
            f"Running in {'serial' if is_serial else 'parallel'} mode on device {device}"
        )

        if not is_serial and cfg.device_mode == "cuda":
            torch.cuda.set_device(device)

        # Set seed
        torch.manual_seed(cfg.training.seed)
        random.seed(cfg.training.seed)

        if cfg.axbench.mode == "inference":
            run_inference(cfg, device)
        elif cfg.axbench.mode == "eval":
            run_eval(cfg)
        elif cfg.axbench.mode == "all":
            run_inference(cfg, device)
            run_eval(cfg)

        if cfg.device_mode == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
    except Exception as e:
        logger.error(f"An error occurred in hydra_main: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
