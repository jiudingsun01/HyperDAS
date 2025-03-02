"""Main entrypoint script to evaluate experimental steering methods based on HyperDAS architectures/codebase."""

import atexit
import gc
import hashlib
import json
import os
import pickle
import random
from enum import Enum
from pathlib import Path
from typing import Dict, List

import httpx
import hydra
import pandas as pd
import torch
import torch.distributed as dist
import wandb
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from openai import AsyncOpenAI
from tqdm import tqdm

from axbench.models.sae import load_metadata_flatten
from axbench.scripts.evaluate import eval_steering
from axbench.scripts.inference import (
    METADATA_FILE,
    create_data_latent,
    create_data_steering,
    load_state,
    partition_concept_ids,
    prepare_df,
    save,
    save_state,
)
from axbench.utils.constants import CHAT_MODELS
from axbench.utils.dataset import DatasetFactory, SteeringDatasetFactory
from axbench.utils.model_utils import get_prefix_length
from logger import get_logger
from src.common.utils import setup_tokenizer
from src.hyperdas.data.axbench import get_axbench_collate_fn
from src.hyperdas.llama3.model import (
    SteeringInterpretor,
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


def get_steering_cache_key(config, concept_ids, metadata):
    """Create a hash key based on relevant inputs that would affect the steering data."""
    cache_key_dict = {
        "num_examples": config.axbench.inference.steering_num_of_examples,
        "steering_factors": OmegaConf.to_container(
            config.axbench.inference.steering_factors
        ),
        "steering_datasets": OmegaConf.to_container(
            config.axbench.inference.steering_datasets
        ),
        "concept_ids": sorted(concept_ids),  # Sort for consistency
        "metadata_hash": hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(cache_key_dict, sort_keys=True).encode()
    ).hexdigest()


@torch.no_grad()
def predict_steering(
    model,
    target_model_tokenizer,
    hypernet_tokenizer,
    concept_df: pd.DataFrame,
    device: str,
    mode: InferenceModes,
    reconstruction_data=None,
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
        reconstruction_data: reconstruction data
        max_test_steps: Maximum number of eval steps (-1 for no limit)
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
            modes="eval" if mode == InferenceModes.HYPERREFT else "prompt",
            intervention_layers=intervention_layers,
            intervention_positions=intervention_positions,
        )(converted_batch)

        # Move batch to GPU
        batch = {k: v.to(device) for k, v in batch.items()}

        # Generate outputs based on mode
        if mode == InferenceModes.HYPERREFT:
            outputs = model.generate(
                editor_input_ids=batch["editor_input_ids"],
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
        elif mode == InferenceModes.PROMPT_STEERING:
            outputs = model.generate(
                target_input_ids=batch.get("target_input_ids"),
                target_attention_mask=batch.get("target_attention_mask"),
                target_position_ids=batch.get("target_position_ids"),
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

    torch.cuda.empty_cache()
    gc.collect()

    return {
        f"{mode}_steered_generation": all_generations,
        f"{mode}_perplexity": all_perplexities,
    }


@torch.no_grad()
def predict_latent():
    pass


def infer_steering(config, rank, world_size, device, logger):
    logger.info(f"Starting inference for rank {rank} out of {world_size}")

    data_dir = Path(config.axbench.inference.data_dir)
    dump_dir = Path(config.training.load_trained_from)

    num_of_examples = config.axbench.inference.steering_num_of_examples
    metadata = load_metadata_flatten(data_dir / METADATA_FILE)
    steering_factors = config.axbench.inference.steering_factors
    steering_datasets = config.axbench.inference.steering_datasets

    logger.debug(f"Loaded metadata with {len(metadata)} examples")

    state = load_state(dump_dir, "steering", rank)
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
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    is_chat_model = (
        True if config.axbench.inference.model_name in CHAT_MODELS else False
    )
    prefix_length = 1  # prefix is default to 1 for all models due to the BOS token.
    if is_chat_model:
        prefix_length = get_prefix_length(tokenizer)
        logger.warning(f"Chat model prefix length: {prefix_length}")

    # Load model instance onto device
    logger.info("Loading model instance")
    model_instance = SteeringInterpretor(config, device)

    if config.training.load_trained_from is not None:
        logger.info(f"Loading model from {config.training.load_trained_from}")
        model_instance.load_model(config.training.load_trained_from)

    model_instance.eval()
    logger.info("Model loaded and set to eval mode")

    # Prepare data per concept
    logger.debug("Preparing data per concept")
    cache_key = get_steering_cache_key(config, my_concept_ids, metadata)
    cache_file = os.path.join(dump_dir, f"steering_data_cache_{cache_key}.parquet")

    # Try to load from cache first
    if os.path.exists(cache_file):
        logger.info("Loading steering data from cache")
        cached_df = pd.read_parquet(cache_file)
        data_per_concept = {}
        # Reconstruct the data_per_concept dictionary from the cached DataFrame
        for concept_id in my_concept_ids:
            concept_data = cached_df[cached_df["concept_id"] == concept_id]
            if not concept_data.empty:
                sae_link = concept_data["sae_link"].iloc[0]
                sae_id = concept_data["sae_id"].iloc[0]
                data_per_concept[concept_id] = (
                    concept_data.drop(["sae_link", "sae_id"], axis=1),
                    sae_link,
                    sae_id,
                )
    else:
        logger.info("Generating steering data and caching results")
        data_per_concept = {}
        all_dfs = []

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
            # Add sae info to DataFrame for caching
            current_df["sae_link"] = sae_link
            current_df["sae_id"] = sae_id
            all_dfs.append(current_df)

            # Store in memory format
            data_per_concept[concept_id] = (
                current_df.drop(["sae_link", "sae_id"], axis=1),
                sae_link,
                sae_id,
            )

        # Cache the results
        if all_dfs:
            pd.concat(all_dfs, ignore_index=True).to_parquet(cache_file)
            logger.info(f"Cached steering data to {cache_file}")

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
        save_state(dump_dir, current_state, "generate", rank)

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


def infer_latent(
    config: DictConfig, rank, world_size, device, logger, training_args, generate_args
):
    data_dir = Path(config.axbench.inference.data_dir)
    dump_dir = Path(config.training.load_trained_from)
    num_of_examples = config.axbench.inference.steering_num_of_examples
    metadata = load_metadata_flatten(data_dir / METADATA_FILE)

    state = load_state(dump_dir, "latent", rank)
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

    # Create a new OpenAI client.
    client = AsyncOpenAI(
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

    is_chat_model = (
        True if config.axbench.inference.model_name in CHAT_MODELS else False
    )
    prefix_length = 1  # prefix is default to 1 for all models due to the BOS token.
    if is_chat_model:
        prefix_length = get_prefix_length(tokenizer)
        logger.warning(f"Chat model prefix length: {prefix_length}")

    # Load dataset factory for evals.
    dataset_factory = DatasetFactory(
        None,
        client,
        tokenizer,
        generate_args.dataset_category,
        None,
        None,
        dump_dir,
        use_cache=False,
        master_data_dir=config.axbench.inference.master_data_dir,
        lm_model=config.axbench.inference.lm_model,
        logger=logger,
        is_inference=True,
        overwrite_inference_data_dir=training_args.overwrite_inference_data_dir,
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    # Now loop over concept_ids and use preloaded models
    cache_df = {}
    for concept_id in my_concept_ids:
        dataset_category = generate_args.dataset_category
        if (concept_id, dataset_category) not in cache_df:
            current_df = create_data_latent(
                dataset_factory,
                metadata,
                concept_id,
                num_of_examples,
                config.axbench.inference,
            )
            logger.warning(
                f"Inference latent with {config.axbench.inference.model_name} on {device} for concept {concept_id}."
            )
            current_df = prepare_df(
                current_df,
                tokenizer,
                is_chat_model,
                config.axbench.inference.model_name,
            )
            cache_df[(concept_id, dataset_category)] = current_df
        else:
            current_df = cache_df[(concept_id, dataset_category)]

        # TODO(sid): call our own predict_latent function
        results = predict_latent(
            current_df,
            batch_size=config.axbench.inference.latent_batch_size,
            prefix_length=prefix_length,
        )

        # Store the results in current_df
        for k, v in results.items():
            if k == "tokens":
                if "tokens" not in current_df:
                    current_df["tokens"] = v  # for tokens, they are global
                else:
                    continue
            else:
                current_df[f"{config.axbench.inference.model_name}_{k}"] = v

        save(dump_dir, "latent", current_df, rank)
        logger.warning(
            f"Saved inference results for concept {concept_id} to rank_{rank}_latent_data.parquet"
        )
        # After processing, save state
        current_state = {"last_concept_id": concept_id}
        save_state(dump_dir, current_state, "latent", rank)

    # Synchronize all processes
    dist.barrier()

    # Rank 0 merges results
    if rank == 0:
        logger.warning("Rank 0 is merging results.")
        # Merge per-rank results
        all_parquet_files = list(
            (Path(dump_dir) / "inference").glob("rank_*_latent_data.parquet")
        )
        # Parse filenames to extract rank
        import re

        pattern = re.compile(r"rank_(\d+)_latent_data\.parquet")

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

        # Read and concatenate dataframes
        dfs = []
        for info in file_info_list:
            df = pd.read_parquet(info["file"])
            dfs.append(df)
        if len(dfs) > 0:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df.to_parquet(
                Path(dump_dir) / "inference" / "latent_data.parquet", engine="pyarrow"
            )
            logger.warning(
                f"Saved combined latent inference results to {Path(dump_dir) / 'inference' / 'latent_data.parquet'}"
            )
        else:
            logger.warning("No results to merge.")

        # Optionally, delete per-rank files
        for info in file_info_list:
            os.remove(info["file"])
            logger.warning(f"Deleted {info['file']}")

        # Save top logits (optional)
        logger.warning("Saving top logits...")

        # TODO(sid): refactor to use our impl
        if "LsReFT" in args.models:
            model_name = "LsReFT"
            model_class = getattr(axbench, model_name)
            benchmark_model = model_class(
                model_instance,
                tokenizer,
                layer=layer,
                low_rank_dimension=len(metadata),
                device=device,
            )
            benchmark_model.load(dump_dir=train_dir, sae_path=metadata[0]["ref"])
            if hasattr(benchmark_model, "ax") and args.use_bf16:
                benchmark_model.ax.eval()
                benchmark_model.ax.to(torch.bfloat16)
            benchmark_model.to(device)
            for concept_id in concept_ids:
                top_logits, neg_logits = benchmark_model.get_logits(concept_id, k=10)
                top_logits_entry = {
                    "concept_id": int(concept_id),
                    "results": {
                        model_name: {"top_logits": top_logits, "neg_logits": neg_logits}
                    },
                }
                with open(Path(dump_dir) / "inference" / "top_logits.jsonl", "a") as f:
                    f.write(json.dumps(top_logits_entry) + "\n")


def infer_latent_imbalance(
    args, rank, world_size, device, logger, training_args, generate_args
):
    data_dir = args.data_dir
    train_dir = args.train_dir
    dump_dir = args.dump_dir
    num_of_examples = args.latent_num_of_examples
    config = load_config(train_dir)
    metadata = load_metadata_flatten(data_dir)
    layer = config["layer"] if config else 0  # default layer for prompt baselines

    # Get list of all concept_ids
    concept_ids = [metadata[i]["concept_id"] for i in range(len(metadata))]

    # Create a new OpenAI client.
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=1000),
            headers={"Connection": "close"},
        ),
        max_retries=3,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=1024)
    tokenizer.padding_side = "right"

    # Load model instance onto device
    if args.use_bf16:
        logger.warning(f"Using bfloat16 for model {args.model_name}")
    model_instance = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.use_bf16 else None,
        device_map="auto",
    )
    is_chat_model = True if args.model_name in CHAT_MODELS else False
    model_instance = model_instance.eval()

    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        need_resize = True
    else:
        need_resize = False
    if need_resize:
        model_instance.resize_token_embeddings(len(tokenizer))

    prefix_length = 1  # prefix is default to 1 for all models due to the BOS token.
    if is_chat_model:
        prefix_length = get_prefix_length(tokenizer)
        logger.warning(f"Chat model prefix length: {prefix_length}")

    # Load dataset factory for evals.
    dataset_factory = DatasetFactory(
        None,
        client,
        tokenizer,
        generate_args.dataset_category,
        None,
        None,
        dump_dir,
        use_cache=False,
        master_data_dir=args.master_data_dir,
        lm_model=args.lm_model,
        logger=logger,
        is_inference=True,
        overwrite_inference_data_dir=training_args.overwrite_inference_data_dir,
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    has_latent_model = False
    for model_name in args.models:
        # load model on the fly to save memory
        if model_name not in LATENT_EXCLUDE_MODELS:
            has_latent_model = True
            break

    if not has_latent_model:
        logger.warning("No latent model to infer. Exiting.")
        return

    logger.warning(
        f"We are inferencing imbalanced latent once for all concepts with factor {args.imbalance_factor}."
    )
    all_negative_df = dataset_factory.create_imbalance_eval_df(
        num_of_examples, factor=args.imbalance_factor
    )
    all_negative_df = prepare_df(
        all_negative_df, tokenizer, is_chat_model, args.model_name
    )

    # save all_negative_df to disk
    dump_dir = Path(dump_dir) / "inference_imbalance"
    dump_dir.mkdir(parents=True, exist_ok=True)
    all_negative_df.to_parquet(
        Path(dump_dir) / "all_negative_df.parquet", engine="pyarrow"
    )

    for model_name in args.models:
        # load model on the fly to save memory
        if model_name in LATENT_EXCLUDE_MODELS:
            continue
        model_class = getattr(axbench, model_name)
        logger.warning(f"Loading {model_class} on {device}.")
        benchmark_model = model_class(
            model_instance,
            tokenizer,
            layer=layer,
            low_rank_dimension=len(metadata),
            device=device,
        )
        if model_name in {"PromptDetection", "BoW"}:
            for concept_id in concept_ids:
                benchmark_model.load(
                    dump_dir=train_dir,
                    sae_path=metadata[0]["ref"],
                    mode="latent",
                    concept_id=concept_id,
                )
                benchmark_model.to(device)
                if hasattr(benchmark_model, "ax") and args.use_bf16:
                    benchmark_model.ax.eval()
                    benchmark_model.ax.to(torch.bfloat16)
                results = benchmark_model.predict_latent(
                    all_negative_df,
                    batch_size=args.latent_batch_size,
                    prefix_length=prefix_length,
                    concept=metadata[concept_id]["concept"],
                )
                # save results to disk
                with open(
                    dump_dir / f"{model_name}_concept_{concept_id}_latent_results.pkl",
                    "wb",
                ) as f:
                    pickle.dump(results, f)
        else:
            benchmark_model.load(
                dump_dir=train_dir, sae_path=metadata[0]["ref"], mode="latent"
            )
            benchmark_model.to(device)
            if hasattr(benchmark_model, "ax") and args.use_bf16:
                benchmark_model.ax.eval()
                benchmark_model.ax.to(torch.bfloat16)
            # we only save the max act for each concept to save disk space, otherwise each file will be ~3GB.
            # if you wish to save the raw acts, you can go into predict_latents and modify the output.
            results = benchmark_model.predict_latents(
                all_negative_df,
                batch_size=args.latent_batch_size,
                prefix_length=prefix_length,
            )
            # save results to disk
            with open(dump_dir / f"{model_name}_latent_results.pkl", "wb") as f:
                pickle.dump(results, f)


def find_run_by_name(run_name, entity, project):
    """Find a wandb run by name across all projects."""
    api = wandb.Api()
    # Search across all projects in the entity
    runs = api.runs(f"{entity}/{project}", {"display_name": run_name})
    for run in runs:
        if run.name == run_name:
            return run
    return None


def run_eval(cfg: DictConfig):
    args = cfg.axbench.evaluate

    checkpoint_path = Path(cfg.training.load_trained_from)
    # load config from checkpoint
    trained_cfg = OmegaConf.load(checkpoint_path.parent / "config.yaml")

    args.data_dir = os.path.join(cfg.training.load_trained_from, "inference")
    args.dump_dir = cfg.training.load_trained_from

    # Initialize wandb if needed
    if args.report_to is not None and "wandb" in args.report_to:
        if checkpoint_path.exists():
            # Extract run name from checkpoint dir - adjust pattern as needed
            run_name = checkpoint_path.parent.name

            # Try to find existing run
            existing_run = find_run_by_name(
                run_name,
                trained_cfg.wandb_config.entity,
                trained_cfg.wandb_config.project,
            )
            if existing_run:
                # Resume the existing run
                wandb.init(
                    project=trained_cfg.wandb_config.project,
                    entity=trained_cfg.wandb_config.entity,
                    id=existing_run.id,
                    resume="must",
                )
            else:
                # Create new run with standard naming
                wandb_name = f"{args.dump_dir.split('/')[-1]}"
                wandb.init(
                    project=trained_cfg.wandb_config.project,
                    entity=trained_cfg.wandb_config.entity,
                    name=f"{wandb_name}_{args.mode}"
                    if args.run_name is None
                    else f"{args.run_name}_{wandb_name}_{args.mode}",
                )
        else:
            # Standard wandb initialization if no checkpoint dir
            wandb_name = f"{args.dump_dir.split('/')[-1]}"
            wandb.init(
                project=trained_cfg.wandb_config.project,
                entity=trained_cfg.wandb_config.entity,
                name=f"{wandb_name}_{args.mode}"
                if args.run_name is None
                else f"{args.run_name}_{wandb_name}_{args.mode}",
            )

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

        exp_cfg = cfg.experiment if hasattr(cfg, "experiment") else cfg

        if exp_cfg.axbench.mode == "inference":
            run_inference(exp_cfg, device)
        elif exp_cfg.axbench.mode == "eval":
            run_eval(exp_cfg)
        elif exp_cfg.axbench.mode == "all":
            run_inference(exp_cfg, device)
            run_eval(exp_cfg)

        if cfg.device_mode == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
    except Exception as e:
        logger.error(f"An error occurred in hydra_main: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
