"""Main entrypoint script to evaluate experimental steering methods based on HyperDAS architectures/codebase."""

import atexit
import gc
import hashlib
import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List

import einops
import httpx
import hydra
import pandas as pd
import torch
import torch.distributed as dist
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from openai import AsyncOpenAI
from tqdm import tqdm

import wandb
from axbench.models.sae import load_metadata_flatten
from axbench.scripts.evaluate import eval_latent, eval_steering
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
from src.hyperdas.utils import AxbenchMode, calculate_perplexity

logger = get_logger(__name__)


def _prepare_config_for_cache(config_dict):
    """Helper function to prepare config dict for caching by removing/handling non-serializable values."""
    processed_dict = {}
    for key, value in config_dict.items():
        # Skip private keys (starting with _) and known non-serializable types
        if key.startswith("_") or callable(value):
            continue

        try:
            # Test JSON serialization
            json.dumps(value)
            processed_dict[key] = value
        except (TypeError, OverflowError):
            # If value isn't JSON serializable, convert to string representation
            processed_dict[key] = str(value)

    return processed_dict


def get_cache_key(config, concept_ids, metadata):
    """Create a hash key based on relevant inputs that would affect the steering data."""
    # Extract only the relevant keys from the inference config
    relevant_keys = [
        "input_length",
        "latent_num_of_examples",
        "latent_batch_size",
        "steering_output_length",
        "steering_num_of_examples",
        "steering_factors",
        "steering_datasets",
        "seed",
    ]

    # Convert OmegaConf to dict and extract only relevant keys
    config_dict = OmegaConf.to_container(config.axbench.inference, resolve=True)
    processed_config = {k: config_dict[k] for k in relevant_keys if k in config_dict}

    cache_key_dict = {
        "config": processed_config,
        "concept_ids": sorted(concept_ids),  # Sort for consistency
        "metadata_hash": hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(cache_key_dict, sort_keys=True).encode()
    ).hexdigest()


def pre_compute_mean_activations(model_name, dump_dir, **kwargs):
    dump_dir = Path(dump_dir) / "inference"

    if not os.path.exists(dump_dir):
        raise ValueError("Run latent inference first before steering.")

    max_activations = {}  # sae_id to max_activation
    # Loop over saved latent files in dump_dir.
    for file in os.listdir(dump_dir):
        if file.startswith("latent_") and file.endswith(".parquet"):
            latent_path = os.path.join(dump_dir, file)
            latent = pd.read_parquet(latent_path)
            # loop through unique sorted concept_id
            for concept_id in sorted(latent["concept_id"].unique()):
                concept_latent = latent[latent["concept_id"] == concept_id]
                max_act = concept_latent[f"{model_name}_max_act"].max()
                max_activations[concept_id] = max_act if max_act > 0 else 50
    return max_activations


@torch.no_grad()
def predict_steering(
    model,
    target_model_tokenizer,
    hypernet_tokenizer,
    concept_df: pd.DataFrame,
    device: str,
    mode: AxbenchMode,
    ground_truth_intervention: Dict[str, torch.Tensor] = None,
    batch_size: int = 16,
    eval_output_length: int = 100,
    temperature: float = 0.7,
    do_sample: bool = True,
    intervention_positions: str = None,
    intervention_layers: List[int] = None,
    max_acts_map: Dict[int, float] = None,
    intervene_during_generation: bool = True,
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
        reconstruction_data: reconstruction data (alternative to hypernetwork)
        max_test_steps: Maximum number of eval steps (-1 for no limit)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling for generation
        steering_prompt: Optional prompt to guide the steering (only used in steering mode)
        intervention_positions: Intervention positions
        intervention_layers: Intervention layers
        max_acts_map: max activations for each concept
        inteerve_during_generation: Whether to intervene during generation
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
            modes=mode,
            intervention_layers=intervention_layers,
            intervention_positions=intervention_positions,
        )(converted_batch)

        # Overrides hypernetwork weight generation with ground truth trained weights
        if ground_truth_intervention is not None:
            # Get concept IDs from the batch
            concept_ids = [row.concept_id for row in raw_batch.itertuples()]
            # Select tensors for these concept IDs
            batch_weights = torch.stack(
                [ground_truth_intervention[concept_id] for concept_id in concept_ids]
            )
            batch["target_weights"] = batch_weights

        # Move batch to GPU
        batch = {k: v.to(device) for k, v in batch.items()}

        max_acts = torch.tensor(
            [max_acts_map.get(id, 1.0) for id in raw_batch["concept_id"].tolist()]
        ).to(device)
        factors = torch.tensor(raw_batch["factor"].tolist()).to(device)

        # Generate outputs based on mode
        if mode == AxbenchMode.EVAL_STEERING:
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
                target_weights=batch.get("target_weights"),
                max_new_tokens=eval_output_length,
                do_sample=do_sample,
                temperature=temperature,
                do_intervention=True,
                max_acts=max_acts,
                factors=factors,
                intervene_during_generation=intervene_during_generation,
            )
        elif mode == AxbenchMode.PROMPT_STEERING:
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

        # Print the non-masked portion of each target input
        for i, (input_ids, attention_mask) in enumerate(zip(batch["target_input_ids"], batch["target_attention_mask"])):
            print(f"\nInput {i}:")
            print(input_ids[attention_mask.bool()]) # Print raw token ids
            print("Decoded input:")
            print(target_model_tokenizer.decode(input_ids[attention_mask.bool()], skip_special_tokens=True))
            print("-" * 40)
        breakpoint()
        # Print generations alongside inputs and concepts
        # for gen, batch_input, concept in zip(
        #     generations, raw_batch["input"], raw_batch["input_concept"]
        # ):
        #     print(f"\nGENERATION: {gen}")
        #     print(f"INPUT: {batch_input}")
        #     print(f"CONCEPT: {concept}")
        #     print("-" * 80)

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
        "steered_generation": all_generations,
        "perplexity": all_perplexities,
    }


@torch.no_grad()
def predict_latent(
    model,
    target_model_tokenizer,
    hypernet_tokenizer,
    concept_df: pd.DataFrame,
    device: str,
    ground_truth_intervention: Dict[str, torch.Tensor] = None,
    batch_size: int = 16,
    intervention_positions: str = None,
    intervention_layers: List[int] = None,
    **kwargs,
) -> Dict[str, List]:
    """
    Extract latent activations for concept detection evaluation.
    Returns dict with activations and detection scores.

    Args:
        model: The model interpretor
        target_model_tokenizer: Tokenizer for the target model
        hypernet_tokenizer: Tokenizer for the hypernetwork arch
        concept_df: DataFrame containing concept examples
        device: Device to run inference on
        reconstruction_data: reconstruction data (alternative to hypernetwork)
        batch_size: Batch size for processing
        intervention_positions: Intervention positions
        intervention_layers: Intervention layers
    """
    all_max_activations = []
    all_detection_scores = []
    all_tokens = []

    for step in tqdm(range(0, len(concept_df), batch_size), desc="latent eval steps"):
        raw_batch = concept_df.iloc[step : step + batch_size]
        converted_batch = list(map(lambda x: x._asdict(), raw_batch.itertuples()))

        # compute batch from raw batch
        batch = get_axbench_collate_fn(
            hypernet_tokenizer,
            target_model_tokenizer,
            modes="eval_concept",
            intervention_layers=intervention_layers,
            intervention_positions=intervention_positions,
        )(converted_batch)

        if ground_truth_intervention is not None:
            # Get concept IDs from the batch
            concept_ids = [row.concept_id for row in raw_batch.itertuples()]
            # Select tensors for these concept IDs
            batch_weights = torch.stack(
                [ground_truth_intervention[concept_id] for concept_id in concept_ids]
            )
            batch["target_weights"] = batch_weights

        # Move batch to GPU
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass to get activations
        outputs = model.forward(
            editor_input_ids=batch.get("editor_input_ids", None),
            target_input_ids=batch.get("target_input_ids", None),
            target_attention_mask=batch.get("target_attention_mask", None),
            target_position_ids=batch.get("target_position_ids", None),
            intervention_layers=batch.get("intervention_layers", None),
            intervention_positions=batch.get("intervention_positions", None),
            target_weights=batch.get("target_weights", None),
            is_causal=batch.get("is_causal", None),
            return_intervened_states=True,
        )

        # Process each sequence in the batch
        attn_mask = batch["target_attention_mask"]
        for seq_idx, act in enumerate(outputs.extra_outputs["detect_latent"]):
            # Get valid token positions
            valid_positions = attn_mask[seq_idx] > 0

            # Get activations for valid tokens
            acts = act[valid_positions].float().cpu().numpy()
            acts = [round(float(x), 3) for x in acts]

            # Get max activation
            max_act = max(acts) if acts else 0.0

            # Get token ids for valid positions
            token_ids = (
                batch["target_input_ids"][seq_idx][valid_positions].cpu().tolist()
            )
            tokens = target_model_tokenizer.convert_ids_to_tokens(token_ids)

            # Store results
            all_max_activations.append(max_act)
            all_detection_scores.append(acts)
            all_tokens.append(tokens)

    torch.cuda.empty_cache()
    gc.collect()

    return {
        "max_act": all_max_activations,
        "detection_scores": all_detection_scores,
        "tokens": all_tokens,
    }


def infer_steering(config, rank, world_size, device, logger):
    logger.info(f"Starting inference for rank {rank} out of {world_size}")

    data_dir = Path(config.axbench.inference.data_dir)
    dump_dir = Path(config.training.load_trained_from or "assets/aux")
    cache_dir = Path(config.axbench.cache_dir or "assets/data/axbench/cache")

    num_of_examples = config.axbench.inference.steering_num_of_examples
    metadata = load_metadata_flatten(data_dir / METADATA_FILE)
    steering_factors = config.axbench.inference.steering_factors
    steering_datasets = config.axbench.inference.steering_datasets

    logger.debug(f"Loaded metadata with {len(metadata)} examples")

    state = load_state(dump_dir, "steering", rank)
    last_concept_id_processed = (
        state.get("last_concept_id", None)
        if state and not config.axbench.inference.ignore_state
        else None
    )
    logger.warning(
        f"Rank {rank} last concept_id processed: {last_concept_id_processed}"
    )

    # Get list of all concept_ids
    concept_ids = [metadata[i]["concept_id"] for i in range(len(metadata))]

    # Partition concept_ids among ranks sequentially
    concept_ids_per_rank = partition_concept_ids(concept_ids, world_size)
    my_concept_ids = concept_ids_per_rank[rank]

    # Load reconstruction data
    if config.axbench.inference.ground_truth_intervention_dir is not None:
        ground_truth_intervention = torch.load(
            Path(config.axbench.inference.ground_truth_intervention_dir)
            / "LsReFT_weight.pt",
            map_location="cpu",
            mmap=True,
        )
        # Load metadata to get concept indices
        metadata = load_metadata_flatten(
            Path(config.axbench.inference.ground_truth_intervention_dir)
            / "metadata.jsonl"
        )
        concept_to_idx = {m["concept_id"]: i for i, m in enumerate(metadata)}

        # Map from concept_id -> subspace using metadata indices
        ground_truth_intervention = {
            concept_id: ground_truth_intervention[concept_to_idx[concept_id]]
            for concept_id in my_concept_ids
        }
    else:
        ground_truth_intervention = None

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
    if config.model.use_lm_hypernetwork:
        # Hypernet tokenizer
        hypernet_tokenizer = setup_tokenizer(
            model_name_or_path=config.model.name_or_path,
            max_length=config.model.max_length,
        )
    else:
        hypernet_tokenizer = None

    has_prompt_steering = "PromptSteering" in config.axbench.evaluate.models

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
    model_instance = SteeringInterpretor(config, device)

    if config.training.load_trained_from is not None:
        logger.info(f"Loading model from {config.training.load_trained_from}")
        try:
            model_instance.load_model(config.training.load_trained_from)
        except Exception as e:
            logger.error(
                f"Failed to load model from {config.training.load_trained_from}: {e}"
            )

    model_instance.eval()
    logger.info("Model loaded and set to eval mode")

    # Prepare data per concept
    logger.debug("Preparing data per concept")
    cache_key = get_cache_key(config, my_concept_ids, metadata)
    cache_file = os.path.join(cache_dir, f"steering_data_cache_{cache_key}.parquet")

    # Try to load from cache first
    if os.path.exists(cache_file):
        logger.info(f"Loading steering data from cache {cache_file}")
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
                config.axbench.inference,
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
            cached_df = pd.concat(all_dfs, ignore_index=True)
            cached_df.to_parquet(cache_file)
            logger.info(f"Cached steering data to {cache_file}")

    axbench_mode_map = {
        model_name: (
            AxbenchMode.PROMPT_STEERING
            if model_name == "PromptSteering"
            else AxbenchMode.EVAL_STEERING
        )
        for model_name in config.axbench.inference.models
    }

    # Process all concepts in batches across concept boundaries
    for model_name in config.axbench.inference.models:
        logger.info(f"Running prediction for {model_name} mode")

        max_acts_map = pre_compute_mean_activations(model_name, dump_dir)

        # Run prediction on all data at once
        results = predict_steering(
            model_instance,
            tokenizer,
            hypernet_tokenizer,
            cached_df,
            device,
            ground_truth_intervention=ground_truth_intervention,
            mode=axbench_mode_map[model_name],
            batch_size=config.axbench.inference.steering_batch_size,
            eval_output_length=config.axbench.inference.steering_output_length,
            temperature=config.axbench.inference.temperature,
            prefix_length=prefix_length,
            intervention_positions=config.model.intervention_positions,
            intervention_layers=config.model.intervention_layer,
            max_acts_map=max_acts_map,
            intervene_during_generation=config.axbench.inference.intervene_during_generation,
        )

        # Store the results in cached_df
        for k, v in results.items():
            cached_df[f"{model_name}_{k}"] = v

    # Save all results at once
    save(dump_dir, "steering", cached_df, rank)
    logger.warning(
        f"Saved inference results for all concepts to rank_{rank}_steering_data.parquet"
    )

    # After processing, save state with the last concept_id
    if my_concept_ids:
        current_state = {"last_concept_id": my_concept_ids[-1]}
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
    assert cfg.training.load_trained_from is not None, (
        "Please specify a checkpoint to load from"
    )

    if cfg.axbench.type == "steering":
        infer_steering(
            cfg,
            0,
            1,
            device,
            get_logger("infer_steering"),
        )
    elif cfg.axbench.type == "latent":
        infer_latent(
            cfg,
            0,
            1,
            device,
            get_logger("infer_latent"),
            cfg.axbench.train,
            cfg.axbench.generate,
        )
    elif cfg.axbench.type == "all":
        # NOTE: latent inference must be run first before steering
        # to compute max acts
        infer_latent(
            cfg,
            0,
            1,
            device,
            get_logger("infer_latent"),
            cfg.axbench.train,
            cfg.axbench.generate,
        )
        infer_steering(
            cfg,
            0,
            1,
            device,
            get_logger("infer_steering"),
        )
    else:
        raise ValueError(f"Invalid inference type: {cfg.axbench.type}")


def infer_latent(
    config: DictConfig, rank, world_size, device, logger, training_args, generate_args
):
    data_dir = Path(config.axbench.inference.data_dir)
    dump_dir = Path(config.training.load_trained_from or "assets/aux")
    cache_dir = Path(config.axbench.cache_dir or "assets/data/axbench/cache")

    num_of_examples = config.axbench.inference.steering_num_of_examples
    metadata = load_metadata_flatten(data_dir / METADATA_FILE)

    state = load_state(dump_dir, "latent", rank)
    last_concept_id_processed = (
        state.get("last_concept_id", None)
        if state and not config.axbench.inference.ignore_state
        else None
    )
    logger.warning(
        f"Rank {rank} last concept_id processed: {last_concept_id_processed}"
    )

    # Get list of all concept_ids
    concept_ids = [metadata[i]["concept_id"] for i in range(len(metadata))]

    # Partition concept_ids among ranks sequentially
    concept_ids_per_rank = partition_concept_ids(concept_ids, world_size)
    my_concept_ids = concept_ids_per_rank[rank]

    # Load reconstruction data
    if config.axbench.inference.ground_truth_intervention_dir is not None:
        ground_truth_intervention = torch.load(
            Path(config.axbench.inference.ground_truth_intervention_dir)
            / "LsReFT_weight.pt",
            map_location="cpu",
            mmap=True,
        )
        # Load metadata to get concept indices
        metadata = load_metadata_flatten(
            Path(config.axbench.inference.ground_truth_intervention_dir)
            / "metadata.jsonl"
        )
        concept_to_idx = {m["concept_id"]: i for i, m in enumerate(metadata)}

        # Map from concept_id -> subspace using metadata indices
        ground_truth_intervention = {
            concept_id: ground_truth_intervention[concept_to_idx[concept_id]]
            for concept_id in my_concept_ids
        }
    else:
        ground_truth_intervention = None

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
    if config.model.use_lm_hypernetwork:
        hypernet_tokenizer = setup_tokenizer(
            model_name_or_path=config.model.name_or_path,
            max_length=config.model.max_length,
        )
    else:
        hypernet_tokenizer = None

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
        try:
            model_instance.load_model(config.training.load_trained_from)
        except Exception as e:
            logger.error(
                f"Failed to load model from {config.training.load_trained_from}: {e}"
            )

    model_instance.eval()
    logger.info("Model loaded and set to eval mode")

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
        overwrite_inference_data_dir=config.axbench.inference.overwrite_inference_data_dir,
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    cache_key = get_cache_key(config, my_concept_ids, metadata)
    cache_file = os.path.join(cache_dir, f"latent_data_cache_{cache_key}.parquet")

    # Try to load from cache first
    if os.path.exists(cache_file):
        logger.info("Loading latent data from cache")
        cached_df = pd.read_parquet(cache_file)
        data_per_concept = {}
        # Reconstruct the data_per_concept dictionary from the cached DataFrame
        for concept_id in my_concept_ids:
            concept_data = cached_df[cached_df["concept_id"] == concept_id]
            if not concept_data.empty:
                data_per_concept[concept_id] = concept_data
    else:
        logger.info("Generating latent data and caching results")
        data_per_concept = {}
        all_dfs = []

        # Replace the existing loop with this
        for concept_id in my_concept_ids:
            current_df = create_data_latent(
                dataset_factory,
                metadata,
                concept_id,
                num_of_examples,
                config.axbench.inference,
            )
            current_df = prepare_df(
                current_df,
                tokenizer,
                is_chat_model,
                config.axbench.inference.model_name,
            )
            current_df["concept_id"] = concept_id  # Add concept_id for caching
            all_dfs.append(current_df)
            data_per_concept[concept_id] = current_df

        # Cache the results
        if all_dfs:
            cached_df = pd.concat(all_dfs, ignore_index=True)
            cached_df.to_parquet(cache_file)
            logger.info(f"Cached latent data to {cache_file}")

    # Process all concepts in one go
    # NOTE: we only support one model for now (WIP)
    for model_name in config.axbench.inference.models:
        results = predict_latent(
            model=model_instance,
            target_model_tokenizer=tokenizer,
            hypernet_tokenizer=hypernet_tokenizer,
            concept_df=cached_df,
            ground_truth_intervention=ground_truth_intervention,
            device=device,
            model_name=model_name,
            batch_size=config.axbench.inference.latent_batch_size,
            intervention_positions=config.model.intervention_positions,
            intervention_layers=config.model.intervention_layer,
        )

        # Store the results in cached_df
        for k, v in results.items():
            if k == "tokens":
                if "tokens" not in cached_df:
                    cached_df.loc[:, "tokens"] = v  # for tokens, they are global
                else:
                    continue
            else:
                cached_df.loc[:, f"{model_name}_{k}"] = v

    save(dump_dir, "latent", cached_df, rank)
    logger.warning(
        f"Saved inference results from concept {my_concept_ids[0]} to {my_concept_ids[-1]} to rank_{rank}_latent_data.parquet"
    )
    # After processing, save state
    current_state = {"last_concept_id": my_concept_ids[-1]}
    save_state(dump_dir, current_state, "latent", rank)

    # Synchronize all processes
    if dist.is_initialized():
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

        # Save top logits
        logger.warning("Saving top logits...")

        # Create directory if it doesn't exist
        (Path(dump_dir) / "inference").mkdir(parents=True, exist_ok=True)

        # Process each concept to get top logits
        for concept_id in concept_ids:
            concept_text = next(
                item["concept"] for item in metadata if item["concept_id"] == concept_id
            )
            # Tokenize the concept text
            editor_input_ids = tokenizer(
                concept_text, return_tensors="pt", padding=True, truncation=True
            ).input_ids.to(device)
            # Get top logits using the method in SteeringInterpretor
            top_logits, neg_logits = model_instance.get_logits(
                editor_input_ids, tokenizer=tokenizer, k=10
            )

            # Save results
            # NOTE: we only support one model for now (WIP)
            top_logits_entry = {
                "concept_id": int(concept_id),
                "results": {
                    config.axbench.inference.models[0]: {
                        "top_logits": top_logits,
                        "neg_logits": neg_logits,
                    }
                },
            }
            with open(Path(dump_dir) / "inference" / "top_logits.jsonl", "a") as f:
                f.write(json.dumps(top_logits_entry) + "\n")


def infer_latent_imbalance(
    config: DictConfig, rank, world_size, device, logger, training_args, generate_args
):
    data_dir = Path(config.axbench.inference.data_dir)
    dump_dir = Path(config.training.load_trained_from)
    num_of_examples = config.axbench.inference.steering_num_of_examples
    metadata = load_metadata_flatten(data_dir / METADATA_FILE)

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

    # Load model instance onto device
    logger.info("Loading model instance")
    model_instance = SteeringInterpretor(config, device)

    if config.training.load_trained_from is not None:
        logger.info(f"Loading model from {config.training.load_trained_from}")
        model_instance.load_model(config.training.load_trained_from)

    model_instance.eval()
    logger.info("Model loaded and set to eval mode")

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

    logger.warning(
        f"We are inferencing imbalanced latent once for all concepts with factor {config.axbench.inference.imbalance_factor}."
    )

    # Create imbalanced evaluation dataset
    all_negative_df = dataset_factory.create_imbalance_eval_df(
        num_of_examples, factor=config.axbench.inference.imbalance_factor
    )
    all_negative_df = prepare_df(
        all_negative_df, tokenizer, is_chat_model, config.axbench.inference.model_name
    )

    # Save all_negative_df to disk
    imbalance_dump_dir = Path(dump_dir) / "inference_imbalance"
    imbalance_dump_dir.mkdir(parents=True, exist_ok=True)
    all_negative_df.to_parquet(
        Path(imbalance_dump_dir) / "all_negative_df.parquet", engine="pyarrow"
    )

    # Process the imbalanced dataset using our model
    results = predict_latent(
        model=model_instance,
        target_model_tokenizer=tokenizer,
        hypernet_tokenizer=hypernet_tokenizer,
        concept_df=all_negative_df,
        device=device,
        batch_size=config.axbench.inference.latent_batch_size,
        intervention_positions=config.model.intervention_positions,
        intervention_layers=config.model.intervention_layer,
    )

    # Save results to disk
    with open(imbalance_dump_dir / "HyperReFT_latent_results.pkl", "wb") as f:
        pickle.dump(results, f)

    logger.warning(
        f"Saved imbalanced latent results to {imbalance_dump_dir}/HyperReFT_latent_results.pkl"
    )

    # Synchronize all processes
    if dist.is_initialized():
        dist.barrier()

    # Rank 0 computes top logits for each concept
    if rank == 0:
        logger.warning("Rank 0 is computing top logits for imbalanced dataset.")

        # Create directory if it doesn't exist
        imbalance_dump_dir.mkdir(parents=True, exist_ok=True)

        # Save top logits
        logger.warning("Saving top logits...")

        # Create directory if it doesn't exist
        (Path(dump_dir) / "inference").mkdir(parents=True, exist_ok=True)

        # Process each concept to get top logits
        for concept_id in concept_ids:
            concept_text = next(
                item["concept"] for item in metadata if item["concept_id"] == concept_id
            )
            # Tokenize the concept text
            editor_input_ids = tokenizer(
                concept_text, return_tensors="pt", padding=True, truncation=True
            ).input_ids.to(device)
            # Get top logits using the method in SteeringInterpretor
            top_logits, neg_logits = model_instance.get_logits(
                editor_input_ids, tokenizer=tokenizer, k=10
            )

            # Save results
            top_logits_entry = {
                "concept_id": int(concept_id),
                "results": {
                    "HyperReFT": {
                        "top_logits": top_logits,
                        "neg_logits": neg_logits,
                    }
                },
            }
            with open(Path(dump_dir) / "inference" / "top_logits.jsonl", "a") as f:
                f.write(json.dumps(top_logits_entry) + "\n")


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

    assert cfg.training.load_trained_from is not None, (
        "Please specify a checkpoint to load from"
    )

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

    if cfg.axbench.type == "steering":
        args.mode = "steering"
        eval_steering(args)
    elif cfg.axbench.type == "latent":
        eval_latent(args)
    elif cfg.axbench.type == "all":
        eval_steering(args)
        eval_latent(args)
    else:
        raise ValueError(f"Invalid inference type: {cfg.axbench.type}")


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
        else:
            raise ValueError(f"Invalid mode: {exp_cfg.axbench.mode}")

        if cfg.device_mode == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
    except Exception as e:
        logger.error(f"An error occurred in hydra_main: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
