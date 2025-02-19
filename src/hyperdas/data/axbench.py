import os
import random
from typing import Literal

import datasets
import numpy as np
import torch


def split_axbench16k_train_test(axbench_train_set, split_by="concept", train_ratio=0.8):
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


def split_axbench16k_train_test_fast(
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
        # Get unique concept IDs using multiprocessing
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
):
    def collate_fn(batch):
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

        editor_input_ids = hypernet_tokenizer(
            edit_instructions, return_tensors="pt", padding=True, truncation=True
        )["input_ids"]
        is_causal = torch.tensor([1 for _ in batch])

        returned_dict = {
            "editor_input_ids": editor_input_ids,
            "is_causal": is_causal,
            **tokenize_text_inputs(hypernet_tokenizer, inputs, targets, prefix="base"),
        }

        if target_tokenizer:
            target_inputs = tokenize_text_inputs(
                target_tokenizer, inputs, targets, prefix="target"
            )
            returned_dict.pop("labels")
            target_inputs.pop("target_intervention_mask")
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

    return collate_fn
