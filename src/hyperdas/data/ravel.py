import json
import os
import random

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_ravel_collate_fn(
    tokenizer,
    contain_entity_position=False,
    dataset_filtering=False,
    source_suffix_visibility=False,
    base_suffix_visibility=False,
    add_space_before_target=True,
):
    """
    Find the position of the entity text in the input_ids by comparing the entity text with the decoded input_ids.
    The entity text could contain multiple tokens.
    Return: a list of positions of the entity text in the input_ids.
    """

    def _find_entity_positions(decoded_input_tokens, entity_text, token_num=1):
        combined_decoded_input_ids = []

        if token_num > len(decoded_input_tokens):
            print(decoded_input_tokens)
            print(entity_text)
            raise ValueError("Entity text not found in input_ids, which is weird!")

        for i in range(len(decoded_input_tokens) - token_num + 1):
            combined_token = "".join(decoded_input_tokens[i : i + token_num])
            combined_decoded_input_ids.append(combined_token)

        for i in range(len(combined_decoded_input_ids)):
            if entity_text in combined_decoded_input_ids[i]:
                return [i + j for j in range(token_num)]

        return _find_entity_positions(decoded_input_tokens, entity_text, token_num + 1)

    def tokenize_text_inputs(
        prefixes,
        suffixes,
        counterfactual_prefixes,
        counterfactual_suffixes,
        target_texts,
        entities=None,
        counterfactual_entities=None,
    ):
        if add_space_before_target:
            input_texts = []
            for prefix, suffix, target in zip(prefixes, suffixes, target_texts):
                if (
                    suffix.endswith(" ")
                    or suffix.endswith('"')
                    or suffix.endswith("'")
                    or suffix.endswith("(")
                ):
                    input_texts.append(tokenizer.bos_token + prefix + suffix + target)
                else:
                    input_texts.append(
                        tokenizer.bos_token + prefix + suffix + " " + target
                    )
        else:
            input_texts = [
                tokenizer.bos_token + prefix + suffix + target
                for prefix, suffix, target in zip(prefixes, suffixes, target_texts)
            ]

        counterfactual_texts = [
            tokenizer.bos_token + prefix + suffix
            for prefix, suffix in zip(counterfactual_prefixes, counterfactual_suffixes)
        ]

        source_intervention_visibility_masks, base_intervention_visibility_masks = (
            [],
            [],
        )

        if entities is not None and counterfactual_entities is not None:
            source_entity_position_ids = []
            base_entity_position_ids = []

        tokenized = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            max_length=50,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_counterfactual = tokenizer(
            counterfactual_texts,
            return_tensors="pt",
            padding=True,
            max_length=50,
            truncation=True,
            add_special_tokens=False,
        )

        tokenized_labels = []

        for i, input_ids in enumerate(tokenized["input_ids"]):
            input_prompt = tokenizer.bos_token + prefixes[i] + suffixes[i]
            prompt_length = tokenizer(
                input_prompt,
                return_tensors="pt",
                padding=False,
                add_special_tokens=False,
            )["input_ids"].shape[-1]
            if tokenizer.padding_side == "left":
                prompt_length += torch.sum(input_ids == tokenizer.pad_token_id)

            label = torch.full_like(input_ids, -100)
            label[prompt_length:] = input_ids[prompt_length:]
            label[input_ids == tokenizer.pad_token_id] = -100
            tokenized_labels.append(label)

            if entities is not None and counterfactual_entities is not None:
                entity_token = entities[i]
                counterfactual_entity_token = counterfactual_entities[i]

                base_entity_position_ids.append(
                    _find_entity_positions(
                        [tokenizer.decode(ids) for ids in input_ids], entity_token
                    )[-1]
                )
                source_entity_position_ids.append(
                    _find_entity_positions(
                        [
                            tokenizer.decode(ids)
                            for ids in tokenized_counterfactual["input_ids"][i]
                        ],
                        counterfactual_entity_token,
                    )[-1]
                )

            source_visibility_mask = tokenized_counterfactual["attention_mask"][
                i
            ].clone()
            base_visibility_mask = tokenized["attention_mask"][i].clone()

            label_length = torch.sum(label != -100)
            base_visibility_mask[-label_length:] = 0

            if not source_suffix_visibility:
                source_suffix_length = tokenizer(
                    counterfactual_suffixes[i],
                    return_tensors="pt",
                    padding=False,
                    add_special_tokens=False,
                )["input_ids"].shape[-1]
                source_visibility_mask[-source_suffix_length:] = 0

            if not base_suffix_visibility:
                base_suffix_length = tokenizer(
                    suffixes[i],
                    return_tensors="pt",
                    padding=False,
                    add_special_tokens=False,
                )["input_ids"].shape[-1]
                base_visibility_mask[prompt_length - base_suffix_length :] = 0

            source_intervention_visibility_masks.append(source_visibility_mask)
            base_intervention_visibility_masks.append(base_visibility_mask)

        base_intervention_mask = torch.stack(base_intervention_visibility_masks)
        source_intervention_mask = torch.stack(source_intervention_visibility_masks)
        tokenized_labels = torch.stack(tokenized_labels)

        return_dict = {
            "base_input_ids": tokenized["input_ids"],
            "base_attention_mask": tokenized["attention_mask"],
            "base_intervention_mask": base_intervention_mask,
            "source_input_ids": tokenized_counterfactual["input_ids"],
            "source_attention_mask": tokenized_counterfactual["attention_mask"],
            "source_intervention_mask": source_intervention_mask,
            "labels": tokenized_labels,
        }

        if entities is not None and counterfactual_entities is not None:
            return_dict["source_entity_position_ids"] = torch.tensor(
                source_entity_position_ids
            )
            return_dict["base_entity_position_ids"] = torch.tensor(
                base_entity_position_ids
            )

        return return_dict

    def collate_fn(batch):
        (
            prefixes,
            suffixes,
            edit_instructions,
            targets,
            counterfactual_prefixes,
            counterfactual_suffixes,
        ) = [], [], [], [], [], []

        if contain_entity_position:
            assert (
                "entity" in batch[0].keys()
                and "counterfactual_entity" in batch[0].keys()
            )
            entities, counterfactual_entities = [], []
        else:
            entities, counterfactual_entities = None, None

        for b in batch:
            prefixes.append(b["input_prefix"])
            suffixes.append(b["input_suffix"])
            edit_instructions.append(tokenizer.bos_token + b["edit_instruction"])
            counterfactual_prefixes.append(b["counterfactual_input_prefix"])
            counterfactual_suffixes.append(b["counterfactual_input_suffix"])

            targets.append(
                b["counterfactual_target"]
                if b["attribute_type"] == "causal"
                else b["target"]
            )

            if contain_entity_position:
                entities.append(b["entity"])
                counterfactual_entities.append(b["counterfactual_entity"])

        # NOTE: truncate=true was here before (sidnb13 removed)
        editor_input_ids = tokenizer(
            edit_instructions,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )["input_ids"]

        is_causal = torch.tensor([b["attribute_type"] == "causal" for b in batch])
        returned_dict = {
            "editor_input_ids": editor_input_ids,
            "is_causal": is_causal,
            **tokenize_text_inputs(
                prefixes,
                suffixes,
                counterfactual_prefixes,
                counterfactual_suffixes,
                targets,
                entities=entities,
                counterfactual_entities=counterfactual_entities,
            ),
        }

        return returned_dict

    def filtering_collate_fn(batch):
        inputs, targets = [], []

        for b in batch:
            inputs.append(b["verify_text"])
            targets.append(
                b["counterfactual_target"]
                if b["attribute_type"] == "causal"
                else b["target"]
            )

        if add_space_before_target:
            input_texts = []
            for input_text, target in zip(inputs, targets):
                if (
                    input_text.endswith(" ")
                    or input_text.endswith('"')
                    or input_text.endswith("'")
                    or input_text.endswith("(")
                ):
                    input_texts.append(tokenizer.bos_token + input_text + target)
                else:
                    input_texts.append(tokenizer.bos_token + input_text + " " + target)
        else:
            input_texts = [
                tokenizer.bos_token + input_text + target
                for input_text, target in zip(inputs, targets)
            ]

        tokenized = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            max_length=50,
            truncation=True,
        )
        tokenized_labels = []

        for i, input_ids in enumerate(tokenized["input_ids"]):
            input_prompt = tokenizer.bos_token + inputs[i]
            prompt_length = tokenizer(input_prompt, return_tensors="pt", padding=False)[
                "input_ids"
            ].shape[-1]
            if tokenizer.padding_side == "left":
                prompt_length += torch.sum(input_ids == tokenizer.pad_token_id)

            label = torch.full_like(input_ids, -100)
            label[prompt_length:] = input_ids[prompt_length:]
            label[input_ids == tokenizer.pad_token_id] = -100
            tokenized_labels.append(label)

        tokenized_labels = torch.stack(tokenized_labels)

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized_labels,
        }

    return collate_fn if not dataset_filtering else filtering_collate_fn


def generate_ravel_dataset(
    n_samples=None,
    root_path="./data/ravel/ravel_clean",
    domain="city",
    isolate_attributes=["Continent"],
    seed=42,
    target_attributes=["Country"],
    template_split="train",
    entity_split="train",
    use_wikipedia_template=True,
    edit_instruction_template="{base_entity} ; {source_entity} - {random_target_attribute}",
):
    def split_into_prefix_suffix(text, entity):
        splits = text.split(entity)
        if len(splits) == 2:
            return splits[0] + entity, splits[1]
        elif len(splits) > 2:
            return entity.join(splits[:-1]) + entity, splits[-1]
        else:
            raise ValueError("Text does not contain entity")

    random.seed(seed)
    np.random.seed(seed)
    dataset = []

    if len(isolate_attributes) != 0:
        sample_per_target_attributes = n_samples // (2 * len(target_attributes))
        sample_per_isolate_attributes = n_samples // (2 * len(isolate_attributes))
    else:
        sample_per_target_attributes = n_samples // len(target_attributes)
        sample_per_isolate_attributes = 0

    entities_split = json.load(
        open(os.path.join(root_path, f"ravel_{domain}_entity_to_split.json"), "r")
    )
    entities = json.load(
        open(os.path.join(root_path, f"ravel_{domain}_entity_attributes.json"), "r")
    )
    templates_split = json.load(
        open(os.path.join(root_path, f"ravel_{domain}_prompt_to_split.json"), "r")
    )
    templates = json.load(
        open(os.path.join(root_path, f"ravel_{domain}_attribute_to_prompts.json"), "r")
    )

    all_attributes = [k for k in templates.keys()]

    wikipedia_templates = json.load(
        open(os.path.join(root_path, f"wikipedia_{domain}_entity_prompts.json"), "r")
    )

    name_all = list(entities.keys())

    entities_train = {k: v for k, v in entities.items() if entities_split[k] == "train"}
    name_train = list(entities_train.keys())

    entities_test = {k: v for k, v in entities.items() if entities_split[k] != "train"}
    name_test = list(entities_test.keys())

    template_train = {
        k: [v for v in vs if templates_split[v] == "train"]
        for k, vs in templates.items()
    }
    template_test = {
        k: [v for v in vs if templates_split[v] != "train"]
        for k, vs in templates.items()
    }

    wikipedia_templates_dict = dict()

    if template_split == "both":
        template_dict = templates

        for k, v_dict in wikipedia_templates.items():
            if v_dict["entity"] is not None:
                if v_dict["entity"] not in wikipedia_templates_dict.keys():
                    wikipedia_templates_dict[v_dict["entity"]] = []

                wikipedia_templates_dict[v_dict["entity"]].append(k)

    else:
        template_dict = template_train if template_split == "train" else template_test

        for k, v_dict in wikipedia_templates.items():
            if v_dict["entity"] is not None:
                if v_dict["entity"] not in wikipedia_templates_dict.keys():
                    wikipedia_templates_dict[v_dict["entity"]] = []

                if (template_split == "train") == (v_dict["split"] == "train"):
                    wikipedia_templates_dict[v_dict["entity"]].append(k)

    if entity_split == "both":
        entity_dict = entities
        entity_name = name_all
    elif entity_split == "train":
        entity_dict = entities_train
        entity_name = name_train
    else:
        entity_dict = entities_test
        entity_name = name_test

    for target_attribute in target_attributes:
        for _ in range(sample_per_target_attributes):
            base_entity, source_entity = (
                random.choice(entity_name),
                random.choice(entity_name),
            )
            while (
                entity_dict[base_entity][target_attribute]
                == entity_dict[source_entity][target_attribute]
            ):
                base_entity, source_entity = (
                    random.choice(entity_name),
                    random.choice(entity_name),
                )

            if source_entity in wikipedia_templates_dict.keys():
                source_template_attribute = (
                    random.choice(all_attributes)
                    if len(wikipedia_templates_dict[source_entity]) == 0
                    or not use_wikipedia_template
                    else random.choice(all_attributes + ["wikipedia"])
                )
            else:
                source_template_attribute = random.choice(all_attributes)

            base_template = random.choice(template_dict[target_attribute])

            if source_template_attribute == "wikipedia":
                source_template = random.choice(wikipedia_templates_dict[source_entity])
            else:
                source_template = random.choice(
                    template_dict[source_template_attribute]
                )

            input_text = base_template % base_entity
            source_text = source_template % source_entity
            verify_text = base_template % source_entity

            input_prefix, input_suffix = split_into_prefix_suffix(
                input_text, base_entity
            )
            counterfactual_input_prefix, counterfactual_input_suffix = (
                split_into_prefix_suffix(source_text, source_entity)
            )

            data = {
                "input_prefix": input_prefix,
                "input_suffix": input_suffix,
                "counterfactual_input_prefix": counterfactual_input_prefix,
                "counterfactual_input_suffix": counterfactual_input_suffix,
                # "edit_instruction": f"{base_entity} ; {source_entity} - {target_attribute}",
                "edit_instruction": edit_instruction_template.format(
                    base_entity=base_entity,
                    source_entity=source_entity,
                    random_target_attribute=random.choice(target_attributes),
                ),
                "entity": base_entity,
                "counterfactual_entity": source_entity,
                "target": entity_dict[base_entity][target_attribute],
                "counterfactual_target": entity_dict[source_entity][target_attribute],
                "attribute_type": "causal",
                "domain": domain,
                "attribute": target_attribute,
                "verify_text": verify_text,
            }

            dataset.append(data)

    for isolate_attribute in isolate_attributes:
        for _ in range(sample_per_isolate_attributes):
            base_entity, source_entity = (
                random.choice(entity_name),
                random.choice(entity_name),
            )
            while (
                entity_dict[base_entity][isolate_attribute]
                == entity_dict[source_entity][isolate_attribute]
            ):
                base_entity, source_entity = (
                    random.choice(entity_name),
                    random.choice(entity_name),
                )

            if source_entity in wikipedia_templates_dict.keys():
                source_template_attribute = (
                    random.choice(all_attributes)
                    if len(wikipedia_templates_dict[source_entity]) == 0
                    or not use_wikipedia_template
                    else random.choice(all_attributes + ["wikipedia"])
                )
            else:
                source_template_attribute = random.choice(all_attributes)

            base_template = random.choice(template_dict[isolate_attribute])

            if source_template_attribute == "wikipedia":
                try:
                    source_template = random.choice(
                        wikipedia_templates_dict[source_entity]
                    )
                except IndexError:
                    print(wikipedia_templates_dict[source_entity])
                    raise
            else:
                source_template = random.choice(
                    template_dict[source_template_attribute]
                )

            input_text = base_template % base_entity
            source_text = source_template % source_entity
            verify_text = base_template % base_entity

            input_prefix, input_suffix = split_into_prefix_suffix(
                input_text, base_entity
            )
            counterfactual_input_prefix, counterfactual_input_suffix = (
                split_into_prefix_suffix(source_text, source_entity)
            )

            data = {
                "input_prefix": input_prefix,
                "input_suffix": input_suffix,
                "counterfactual_input_prefix": counterfactual_input_prefix,
                "counterfactual_input_suffix": counterfactual_input_suffix,
                "edit_instruction": f"{base_entity} ; {source_entity} - {random.choice(target_attributes)}",
                "entity": base_entity,
                "counterfactual_entity": source_entity,
                "target": entity_dict[base_entity][isolate_attribute],
                "counterfactual_target": entity_dict[source_entity][isolate_attribute],
                "attribute_type": "isolate",
                "domain": domain,
                "attribute": isolate_attribute,
                "verify_text": verify_text,
            }

            dataset.append(data)

    dataset = Dataset.from_list(dataset)
    return dataset


def generate_ravel_dataset_from_filtered(
    n_samples,
    root_path="./data/ravel/ravel_raw",
    filtered_dataset_path="./data/ravel/llama3-8b_city_train_10k_per_attr.json",
    split="train",
    domain="city",
    isolate_attributes=["Continent"],
    seed=42,
    target_attributes=["Country"],
    generalization="entity",
):
    assert generalization in ["entity", "attribute"]
    # Seed

    random.seed(seed)
    np.random.seed(seed)
    dataset = []

    if len(isolate_attributes) != 0:
        sample_per_target_attributes = n_samples // (2 * len(target_attributes))
        sample_per_isolate_attributes = n_samples // (2 * len(isolate_attributes))
    else:
        sample_per_target_attributes = n_samples // len(target_attributes)
        sample_per_isolate_attributes = 0

    entities_split = json.load(
        open(os.path.join(root_path, f"ravel_{domain}_entity_to_split.json"), "r")
    )
    entities = json.load(
        open(os.path.join(root_path, f"ravel_{domain}_entity_attributes.json"), "r")
    )

    entities_train = {k: v for k, v in entities.items() if entities_split[k] == "train"}
    name_train = list(entities_train.keys())

    entities_test = {k: v for k, v in entities.items() if entities_split[k] != "train"}

    entity_dict = entities_train if split == "train" else entities_test

    for j, target_attribute in enumerate(target_attributes):
        filtered_dict_key = f"{target_attribute}-train"
        filtered_dict = json.load(open(filtered_dataset_path, "r"))[filtered_dict_key]
        sample_idxs = np.random.choice(
            len(filtered_dict), sample_per_target_attributes, replace=False
        )

        for idx in tqdm(sample_idxs):
            data = {}
            data_sample = filtered_dict[idx]

            input_text = data_sample["input"]
            source_text = data_sample["source_input"]
            verify_text = data_sample["split"] % data_sample["source_entity"]

            entity = data_sample["entity"]
            source_entity = data_sample["source_entity"]

            splits = input_text.split(entity)

            if len(splits) == 2:
                data["input_prefix"], data["input_suffix"] = splits
            elif len(splits) > 2:
                data["input_suffix"] = splits[-1]
                data["input_prefix"] = entity.join(splits[:-1])
            else:
                raise ValueError("Input Text does not contain entity")

            data["input_prefix"] += entity
            data["entity"] = entity

            source_splits = source_text.split(source_entity)

            if len(source_splits) == 2:
                (
                    data["counterfactual_input_prefix"],
                    data["counterfactual_input_suffix"],
                ) = source_splits
            elif len(source_splits) > 2:
                data["counterfactual_input_suffix"] = source_splits[-1]
                data["counterfactual_input_prefix"] = source_entity.join(
                    source_splits[:-1]
                )
            else:
                raise ValueError("Source Text does not contain source entity")

            data["counterfactual_input_prefix"] += source_entity
            data["counterfactual_entity"] = source_entity

            data["edit_instruction"] = (
                f"{entity} ; {source_entity} - {target_attribute}"
            )

            base_entity_dict = entity_dict[entity]
            source_entity_dict = entity_dict[source_entity]

            data["target"] = base_entity_dict[target_attribute]
            data["counterfactual_target"] = source_entity_dict[target_attribute]
            data["attribute_type"] = "causal"
            data["domain"] = domain
            data["attribute"] = target_attribute
            data["verify_text"] = verify_text
            dataset.append(data)

    for j, isolate_attribute in enumerate(isolate_attributes):
        filtered_dict_key = f"{isolate_attribute}-train"
        filtered_dict = json.load(open(filtered_dataset_path, "r"))[filtered_dict_key]

        # Adding Split Information
        for n in range(len(filtered_dict)):
            entity = filtered_dict[n]["entity"]
            filtered_dict[n]["train_test_split"] = (
                "train" if entity in name_train else "test"
            )

        filtered_dict_train = [
            d for d in filtered_dict if d["train_test_split"] == "train"
        ]
        filtered_dict_test = [
            d for d in filtered_dict if d["train_test_split"] == "test"
        ]
        filtered_dict = filtered_dict_train if split == "train" else filtered_dict_test

        sample_idxs = np.random.choice(
            len(filtered_dict), sample_per_isolate_attributes, replace=False
        )

        for _ in tqdm(sample_idxs):
            data = {}
            data_sample = filtered_dict[idx]

            input_text = data_sample["input"]
            source_text = data_sample["source_input"]
            verify_text = data_sample["input"]

            entity = data_sample["entity"]
            source_entity = data_sample["source_entity"]

            splits = input_text.split(entity)

            if len(splits) == 2:
                data["input_prefix"], data["input_suffix"] = splits
            elif len(splits) > 2:
                data["input_suffix"] = splits[-1]
                data["input_prefix"] = entity.join(splits[:-1])
            else:
                raise ValueError("Input Text does not contain entity")

            data["input_prefix"] += entity
            data["entity"] = entity

            source_splits = source_text.split(source_entity)

            if len(source_splits) == 2:
                (
                    data["counterfactual_input_prefix"],
                    data["counterfactual_input_suffix"],
                ) = source_splits
            elif len(source_splits) > 2:
                data["counterfactual_input_suffix"] = source_splits[-1]
                data["counterfactual_input_prefix"] = source_entity.join(
                    source_splits[:-1]
                )
            else:
                raise ValueError("Source Text does not contain source entity")
            data["counterfactual_input_prefix"] += source_entity
            data["counterfactual_entity"] = source_entity

            data["edit_instruction"] = (
                f"{entity} ; {source_entity} - {random.choice(target_attributes)}"
            )

            base_entity_dict = entity_dict[entity]
            source_entity_dict = entity_dict[source_entity]

            data["target"] = base_entity_dict[isolate_attribute]
            data["counterfactual_target"] = source_entity_dict[isolate_attribute]
            data["attribute_type"] = "isolate"
            data["domain"] = domain
            data["attribute"] = isolate_attribute
            data["verify_text"] = verify_text
            dataset.append(data)

    dataset = Dataset.from_list(dataset)
    return dataset


def filter_dataset(
    model,
    tokenizer,
    dataset,
    batch_size=16,
    eval_n_label_tokens=None,
    add_space_before_target=True,
    relative_position_ids=False,
):
    model.eval()
    correct_idxs = set()

    collate_fn = get_ravel_collate_fn(
        tokenizer,
        dataset_filtering=True,
        add_space_before_target=add_space_before_target,
    )

    data_loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
    )

    with torch.no_grad():
        for batch_id, batch in tqdm(enumerate(data_loader)):
            batch = {k: v.to("cuda") for k, v in batch.items()}

            if relative_position_ids:
                batch["position_ids"] = (
                    torch.cumsum(batch["attention_mask"], dim=1)
                    * batch["attention_mask"]
                    - 1
                )

            prediction = model(**batch)
            batch_pred_ids = torch.argmax(prediction["logits"], dim=-1)

            for i, (label, pred_ids) in enumerate(zip(batch["labels"], batch_pred_ids)):
                label_idx = label != -100
                output_idx = torch.zeros_like(label_idx)
                output_idx[:-1] = label_idx[1:]

                label = label[label_idx]
                pred_ids = pred_ids[output_idx]

                if eval_n_label_tokens is not None and len(label) > eval_n_label_tokens:
                    label = label[:eval_n_label_tokens]
                    pred_ids = pred_ids[:eval_n_label_tokens]

                is_correct = (torch.sum(label == pred_ids) == torch.numel(label)).item()

                if is_correct:
                    correct_idxs.add(batch_id * len(batch["labels"]) + i)

    filtered_dataset = dataset.select(list(correct_idxs))
    print(
        f"Accuracy: {len(filtered_dataset) / len(dataset)}; filtered out {len(dataset) - len(filtered_dataset)} examples"
    )
    return filtered_dataset
