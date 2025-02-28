import os
import random
from typing import Literal

import datasets
import numpy as np
import pandas as pd
import torch

from axbench.scripts.evaluate import load_jsonl


def split_axbench_train_test(axbench_train_set, split_by="concept", train_ratio=0.8):
    axbench_train_set = [d for d in axbench_train_set if d["category"] == "positive"]

    if split_by == "concept":
        all_concept_ids = list(set([d["concept_id"] for d in axbench_train_set]))

        train_concept_ids = set(
            random.sample(all_concept_ids, int(len(all_concept_ids) * train_ratio))
        )
        train_set = [
            d for d in axbench_train_set if d["concept_id"] in train_concept_ids
        ]
        test_set = [
            d for d in axbench_train_set if d["concept_id"] not in train_concept_ids
        ]

    elif split_by == "example":
        raise NotImplementedError

    elif split_by == "random":
        train_set = random.sample(
            axbench_train_set, int(len(axbench_train_set) * train_ratio)
        )
        test_set = [d for d in axbench_train_set if d not in train_set]

    return train_set, test_set


def split_axbench_train_test_fast(
    dataset: datasets.Dataset,
    split_by="concept",
    train_ratio=0.8,
    seed=42,
    num_proc=None,
    cache_dir="assets/data/axbench",
):
    """Split an axbench dataset into train and test sets using HF Dataset operations.

    Args:
        dataset: HuggingFace Dataset object containing axbench data
        split_by: How to split the dataset - "concept", "random", or "example"
        train_ratio: Fraction of data to use for training
        seed: Random seed for reproducibility
        num_proc: Number of processes to use for parallel processing
        cache_dir: Directory to cache the processed datasets. If None, uses default HF cache
    """
    # Create a unique cache key based on the parameters
    cache_key = f"split_{split_by}_ratio{train_ratio}_seed{seed}"
    os.makedirs(cache_dir, exist_ok=True)

    # Try to load from cache first
    if cache_dir:
        try:
            train_dataset = datasets.load_from_disk(f"{cache_dir}/train_{cache_key}")
            test_dataset = datasets.load_from_disk(f"{cache_dir}/test_{cache_key}")
            return train_dataset, test_dataset
        except Exception:
            pass  # If loading fails, proceed with computing the split

    # Filter for positive examples using multiprocessing
    dataset = dataset.filter(lambda x: x["category"] == "positive", num_proc=num_proc)

    if split_by == "concept":
        concept_ids = dataset.unique("concept_id")

        # Select train concepts
        random.seed(seed)
        train_concept_ids = set(
            random.sample(concept_ids, int(len(concept_ids) * train_ratio))
        )

        # Create filter functions
        def is_train(x):
            return x["concept_id"] in train_concept_ids

        def is_test(x):
            return x["concept_id"] not in train_concept_ids

        # Split based on concepts using multiprocessing
        train_dataset = dataset.filter(is_train, num_proc=num_proc)
        test_dataset = dataset.filter(is_test, num_proc=num_proc)

    elif split_by == "example":
        raise NotImplementedError("Example-based splitting not implemented")

    elif split_by == "random":
        # Shuffle and split
        dataset = dataset.shuffle(seed=seed)
        split_dict = dataset.train_test_split(
            train_size=train_ratio,
            seed=seed,
        )
        train_dataset = split_dict["train"]
        test_dataset = split_dict["test"]

    if cache_dir:
        train_dataset.save_to_disk(f"{cache_dir}/train_{cache_key}")
        test_dataset.save_to_disk(f"{cache_dir}/test_{cache_key}")

    return train_dataset, test_dataset


def split_axbench_reconstruction_train_test(
    reconstruction_data_path: str,
    split_by="concept",
    train_ratio=0.8,
    seed=42,
):
    """Split an axbench reconstruction dataset into train and test sets using HF Dataset operations."""
    lsreft = torch.load(f"{reconstruction_data_path}/LsReFT_weight.pt")
    lsreft_metadata = load_jsonl(f"{reconstruction_data_path}/metadata.jsonl")
    assert lsreft.shape[0] == len(lsreft_metadata), "Length mismatch"
    # Create a mapping from concept_id to subspace tensor
    concept_to_subspace = {}
    for idx, metadata_entry in enumerate(lsreft_metadata):
        concept_id = metadata_entry["concept_id"]
        concept_to_subspace[concept_id] = lsreft[idx]

    for metadata_entry in lsreft_metadata:
        metadata_entry["subspace"] = concept_to_subspace[metadata_entry["concept_id"]]

    if split_by == "concept":
        # Get unique concept IDs
        concept_ids = set(entry["concept_id"] for entry in lsreft_metadata)

        random.seed(seed)
        train_concept_ids = set(
            random.sample(list(concept_ids), int(len(concept_ids) * train_ratio))
        )

        train_dataset = [
            entry
            for entry in lsreft_metadata
            if entry["concept_id"] in train_concept_ids
        ]
        test_dataset = [
            entry
            for entry in lsreft_metadata
            if entry["concept_id"] not in train_concept_ids
        ]

    elif split_by == "example":
        raise NotImplementedError("Example-based splitting not implemented")

    elif split_by == "random":
        random.seed(seed)
        shuffled_data = lsreft_metadata.copy()
        random.shuffle(shuffled_data)

        split_idx = int(len(shuffled_data) * train_ratio)
        train_dataset = shuffled_data[:split_idx]
        test_dataset = shuffled_data[split_idx:]

    # Convert to HF datasets, ensuring tensors are preserved
    features = datasets.Features(
        {
            "concept_id": datasets.Value("string"),
            "concept": datasets.Value("string"),
            "ref": datasets.Value("string"),
            "concept_genres_map": datasets.Value("string"),
            "subspace": datasets.Sequence(datasets.Value("float32")),
        }
    )

    train_dataset = datasets.Dataset.from_list(train_dataset, features=features)
    test_dataset = datasets.Dataset.from_list(test_dataset, features=features)

    return train_dataset, test_dataset


def combine_concept_reconstruction_datasets(
    concept_train_dataset: datasets.Dataset,
    concept_test_dataset: datasets.Dataset | None,
    reconstruction_data_path: str,
    split_by="concept",
    train_ratio=0.8,
    seed=42,
    num_proc=None,
    cache_dir="assets/data/axbench",
):
    """Combine concept dataset with reconstruction data by matching concept IDs."""
    # Split reconstruction data
    train_lsreft, test_lsreft = split_axbench_reconstruction_train_test(
        reconstruction_data_path=reconstruction_data_path,
        split_by=split_by,
        train_ratio=train_ratio,
        seed=seed,
    )

    if concept_test_dataset is None:
        concept_train_dataset, concept_test_dataset = split_axbench_train_test(
            train_set=concept_train_dataset,
            split_by=split_by,
            train_ratio=train_ratio,
            seed=seed,
        )

    train_lsreft_df = pd.DataFrame(
        {"concept_id": train_lsreft["concept_id"], "subspace": train_lsreft["subspace"]}
    )
    test_lsreft_df = pd.DataFrame(
        {"concept_id": test_lsreft["concept_id"], "subspace": test_lsreft["subspace"]}
    )

    def _add_subspaces_to_dataset(concept_dataset, lsreft_df, cache_file_name=None):
        concept_df = concept_dataset.to_pandas()
        merged_df = pd.merge(concept_df, lsreft_df, on="concept_id", how="left")
        result_dataset = datasets.Dataset.from_pandas(merged_df)
        if cache_file_name:
            result_dataset.save_to_disk(cache_file_name)
        return result_dataset

    train_dataset = _add_subspaces_to_dataset(
        concept_train_dataset,
        train_lsreft_df,
        cache_file_name=f"{cache_dir}/train_with_subspaces.arrow"
        if cache_dir
        else None,
    )
    test_dataset = _add_subspaces_to_dataset(
        concept_test_dataset,
        test_lsreft_df,
        cache_file_name=f"{cache_dir}/test_with_subspaces.arrow" if cache_dir else None,
    )

    return train_dataset, test_dataset


def tokenize_text_inputs(
    tokenizer, inputs, targets, add_space_before_target=True, prefix="base"
):
    # TODO(sid): we should add padding to multiple of 8, for example, to improve training efficiency.
    if add_space_before_target:
        input_texts = []
        for ipt, targ in zip(inputs, targets):
            if (
                ipt.endswith(" ")
                or ipt.endswith('"')
                or ipt.endswith("'")
                or ipt.endswith("(")
            ):
                input_texts.append(ipt + targ)
            else:
                input_texts.append(ipt + " " + targ)
    else:
        input_texts = [ipt + targ for ipt, targ in zip(inputs, targets)]

    base_intervention_visibility_masks = []

    tokenized = tokenizer(
        input_texts, return_tensors="pt", padding=True, truncation=True
    )
    tokenized_labels = []

    for i, input_ids in enumerate(tokenized["input_ids"]):
        input_prompt = inputs[i]
        prompt_length = tokenizer(input_prompt, return_tensors="pt", padding=False)[
            "input_ids"
        ].shape[-1]
        if tokenizer.padding_side == "left":
            prompt_length += torch.sum(input_ids == tokenizer.pad_token_id)

        label = torch.full_like(input_ids, -100)
        label[prompt_length:] = input_ids[prompt_length:]
        label[input_ids == tokenizer.pad_token_id] = -100
        tokenized_labels.append(label)

        base_visibility_mask = tokenized["attention_mask"][i].clone()

        label_length = torch.sum(label != -100)
        base_visibility_mask[-label_length:] = 0

        base_intervention_visibility_masks.append(base_visibility_mask)

        if base_visibility_mask.sum(dim=-1) == 0:
            if base_visibility_mask.sum() == 0:
                print("Base Text: ", input_texts[i])
                print("Base Input: ", tokenized["input_ids"][i])
                print("Base Mask: ", base_visibility_mask)
                print("Base Attention Mask: ", tokenized["attention_mask"][i])

            print("Prompt Length: ", prompt_length)
            print("Label: ", label)
            print("Label Text: ", targets[i])
            print("Tokenized Label: ", tokenizer(targets[i]))
            print("Label Length: ", label_length)
            raise ValueError("Attention Mask is all 0")

    base_intervention_mask = torch.stack(base_intervention_visibility_masks)
    tokenized_labels = torch.stack(tokenized_labels)

    return_dict = {
        f"{prefix}_input_ids": tokenized["input_ids"],
        f"{prefix}_attention_mask": tokenized["attention_mask"],
        f"{prefix}_intervention_mask": base_intervention_mask,
        "labels": tokenized_labels,
    }

    return return_dict


def parse_positions_varlen(positions: str, seq_len: int):
    if positions == "all":
        return np.arange(seq_len).tolist()
    if "+" in positions:
        first_n = int(positions.split("+")[0].strip("f"))
        last_n = int(positions.split("+")[1].strip("l"))
        return np.concatenate(
            [np.arange(first_n), np.arange(seq_len - last_n, seq_len)]
        ).tolist()
    elif "f" in positions:
        return np.arange(int(positions.strip("f"))).tolist()
    elif "l" in positions:
        return np.arange(seq_len - int(positions.strip("l")), seq_len).tolist()
    return []


def get_axbench_collate_fn(
    hypernet_tokenizer,
    target_tokenizer=None,
    mode: Literal[
        "steering_train", "steering_eval", "steering_prompt", "concept"
    ] = "steering_train",
    intervention_layers=None,
    intervention_positions=None,
    objective="sft",
):
    # TODO need to implement grabbing reft parameters for supervised training/regression objective
    def sft_collate_fn(batch):
        inputs, edit_instructions, targets = [], [], []

        for b in batch:
            inputs.append(
                b["input"] if mode != "steering_prompt" else b["steered_input"]
            )
            edit_instructions.append(
                b["input_concept"]
                if mode in ["steering_eval", "steering_prompt"]
                else b["output_concept"]
            )
            if mode not in ["steering_eval", "steering_prompt"]:
                targets.append(b["output"])
            else:
                # Dummy since we are not using labels
                targets.append("")

        is_causal = torch.tensor([1 for _ in batch])
        returned_dict = {
            "is_causal": is_causal,
        }

        if hypernet_tokenizer:
            editor_input_ids = hypernet_tokenizer(
                edit_instructions, return_tensors="pt", padding=True, truncation=True
            )["input_ids"]
            returned_dict["editor_input_ids"] = editor_input_ids
            returned_dict.update(
                tokenize_text_inputs(hypernet_tokenizer, inputs, targets, prefix="base")
            )

        if target_tokenizer:
            target_inputs = tokenize_text_inputs(
                target_tokenizer, inputs, targets, prefix="target"
            )

            if "editor_input_ids" not in returned_dict:
                returned_dict["editor_input_ids"] = target_tokenizer(
                    edit_instructions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )["input_ids"]

            returned_dict.pop("labels", None)
            target_inputs.pop("target_intervention_mask", None)
            returned_dict.update(target_inputs)

        # create intervention layer and positions
        if "steering" in mode:
            # TODO(sid): we can eventually add padding to support variable position interventions within batch
            intervention_pos = torch.tensor(
                [
                    parse_positions_varlen(
                        intervention_positions or "f7+l7", len(targ_ids)
                    )
                    for targ_ids in returned_dict["target_input_ids"]
                ]
            )
            # NOTE: this is non-batched for ease of use, maybe we'll fix later?
            intervention_l = torch.tensor(intervention_layers or [7])
            returned_dict["intervention_layers"] = intervention_l
            returned_dict["intervention_positions"] = intervention_pos

        return returned_dict

    def reconstruction_collate_fn(batch):
        edit_instructions, reconstruction_targets = [], []
        for b in batch:
            if "sft" in objective:
                edit_instructions.append(
                    b["input_concept"]
                    if mode in ["steering_eval", "steering_prompt"]
                    else b["output_concept"]
                )
            else:
                edit_instructions.append(b["concept"])
            if mode == "steering_train":
                # We only have subspace labels for training,
                # inference is done as steering with the generated subspaces
                reconstruction_targets.append(torch.tensor(b["subspace"]))

        returned_dict = {
            "editor_input_ids": target_tokenizer(
                edit_instructions,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )["input_ids"],
        }

        if reconstruction_targets:
            returned_dict["reconstruction_targets"] = torch.stack(
                reconstruction_targets
            )

        return returned_dict

    if objective == "sft":
        collate_fn = sft_collate_fn
    elif objective == "reconstruction":
        collate_fn = reconstruction_collate_fn
    elif objective == "sft+reconstruction":
        # Merge collate function output dicts using a custom lambda
        def merged_collate_fn(batch):
            return {
                **sft_collate_fn(batch),
                **reconstruction_collate_fn(batch),
            }

        collate_fn = merged_collate_fn

    return collate_fn
