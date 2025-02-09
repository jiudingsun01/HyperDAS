import torch
import random


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


def tokenize_text_inputs(tokenizer, inputs, targets, add_space_before_target=True):
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
        "base_input_ids": tokenized["input_ids"],
        "base_attention_mask": tokenized["attention_mask"],
        "base_intervention_mask": base_intervention_mask,
        "labels": tokenized_labels,
    }

    return return_dict


def parse_positions(positions: str):
    # parse position
    first_n, last_n = 0, 0
    if "+" in positions:
        first_n = int(positions.split("+")[0].strip("f"))
        last_n = int(positions.split("+")[1].strip("l"))
    else:
        if "f" in positions:
            first_n = int(positions.strip("f"))
        elif "l" in positions:
            last_n = int(positions.strip("l"))
    return first_n, last_n


def get_axbench_collate_fn(
    tokenizer,
    mode="steering",
    intervention_layers=None,
    intervention_positions=None,
):
    def collate_fn(batch):
        inputs, edit_instructions, targets = [], [], []

        for b in batch:
            inputs.append(b["input"])
            edit_instructions.append(b["output_concept"])
            targets.append(b["output"])

        editor_input_ids = tokenizer(
            edit_instructions, return_tensors="pt", padding=True, truncation=True
        )["input_ids"]
        is_causal = torch.tensor([1 for _ in batch])

        returned_dict = {
            "editor_input_ids": editor_input_ids,
            "is_causal": is_causal,
            **tokenize_text_inputs(tokenizer, inputs, targets),
        }

        # create intervention layer and positions
        if mode == "steering":
            intervention_pos = [
                parse_positions(intervention_positions or "f7+l7") for _ in batch
            ]
            intervention_l = [intervention_layers or [7] for _ in batch]
            returned_dict["intervention_layers"] = intervention_l
            returned_dict["intervention_positions"] = intervention_pos

        return returned_dict

    return collate_fn
