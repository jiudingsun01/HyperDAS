import sys

sys.path.append("../..")


from torch.utils.data import DataLoader
from hyperdas.data.axbench import get_ravel_collate_fn

import torch


def get_run_data(model, tokenizer, dataset, inference_mode=None, eval_n_label_tokens=3):
    assert inference_mode in [
        None,
        "column_argmax",
        "global_argmax",
        "groundtruth",
        "bidding_argmax",
    ]
    model.interpretor.eval()

    dataset = [d for d in dataset if d["attribute_type"] == "causal"]

    collate_fn = get_ravel_collate_fn(
        tokenizer,
        add_space_before_target=True,
        contain_entity_position=True,
        source_suffix_visibility=False,
        base_suffix_visibility=False,
    )

    all_is_correct = []
    indices = []
    intervention_weights = []

    dataloader = DataLoader(
        dataset, batch_size=16, collate_fn=collate_fn, shuffle=False
    )

    curr_idx = 0
    with torch.no_grad():
        for batch_id, batch in enumerate(dataloader):
            batch_example_ids = range(
                curr_idx, curr_idx + len(batch["editor_input_ids"])
            )
            indices.extend(batch_example_ids)
            curr_idx += len(batch["editor_input_ids"])

            batch_is_correct = []

            if inference_mode == "groundtruth":
                intervention_weight = torch.zeros(
                    len(batch["editor_input_ids"]),
                    batch["source_input_ids"].shape[1] + 1,
                    batch["base_input_ids"].shape[1],
                ).to("cuda")
                intervention_weight[:, -1, :] = 1.0

                for i in range(len(batch["base_entity_position_ids"])):
                    intervention_weight[i, -1, batch["base_entity_position_ids"][i]] = (
                        0.0
                    )
                    intervention_weight[
                        i,
                        batch["source_entity_position_ids"][i],
                        batch["base_entity_position_ids"][i],
                    ] = 1.0
            else:
                intervention_weight = None

            predictions = model.forward(
                editor_input_ids=batch["editor_input_ids"].to("cuda"),
                base_input_ids=batch["base_input_ids"].to("cuda"),
                base_attention_mask=batch["base_attention_mask"].to("cuda"),
                base_intervention_mask=batch["base_intervention_mask"].to("cuda"),
                source_input_ids=batch["source_input_ids"].to("cuda"),
                source_attention_mask=batch["source_attention_mask"].to("cuda"),
                source_intervention_mask=batch["source_intervention_mask"].to("cuda"),
                labels=batch["labels"].to("cuda"),
                inference_mode=inference_mode,
                intervention_weight=intervention_weight,
                output_intervention_weight=True,
            )

            for i, weight in enumerate(predictions.intervention_weight):
                source_padding_idx = (
                    batch["source_input_ids"][i] == tokenizer.pad_token_id
                )
                base_padding_idx = batch["base_input_ids"][i] == tokenizer.pad_token_id
                weight_src_padding_idx = torch.cat(
                    [source_padding_idx, torch.tensor([False])]
                )
                weight = weight[~weight_src_padding_idx, :]
                weight = weight[:, ~base_padding_idx]
                intervention_weights.append(weight)

            batch_pred_ids = torch.argmax(predictions["logits"], dim=-1)

            for i, (label, pred_ids) in enumerate(
                zip(batch["labels"].to("cuda"), batch_pred_ids)
            ):
                label_idx = label != -100
                output_idx = torch.zeros_like(label_idx)
                output_idx[:-1] = label_idx[1:]

                label = label[label_idx]
                pred_ids = pred_ids[output_idx]

                if eval_n_label_tokens is not None and len(label) > eval_n_label_tokens:
                    label = label[:eval_n_label_tokens]
                    pred_ids = pred_ids[:eval_n_label_tokens]

                is_correct = (torch.sum(label == pred_ids) == torch.numel(label)).item()
                batch_is_correct.append(is_correct)

            all_is_correct.extend(batch_is_correct)

    intervention_weights = intervention_weights[: len(dataset)]
    all_is_correct = all_is_correct[: len(dataset)]

    for i, (weight, correct) in enumerate(zip(intervention_weights, all_is_correct)):
        dataset[i]["intervention_weight"] = weight
        dataset[i]["is_correct"] = correct

    return dataset


def get_example_max_weight_coord(example, tokenizer):
    base_input_prompt, source_input_prompt = assemble_input_sentences(
        example, tokenizer
    )

    base_input_ids = tokenizer(
        base_input_prompt, return_tensors="pt", padding=False, truncation=False
    ).input_ids[0]
    source_input_ids = tokenizer(
        source_input_prompt, return_tensors="pt", padding=False, truncation=False
    ).input_ids[0]

    intervention_weight = example["intervention_weight"]
    assert intervention_weight.shape == (len(source_input_ids) + 1, len(base_input_ids))

    intervention_weight = intervention_weight[:-1, :]

    # Find the coordinate of the maximum weight in the 2D weight matrix
    max_weight_coordinate = intervention_weight.argmax()
    max_weight_coordinate = (
        max_weight_coordinate // intervention_weight.shape[1],
        max_weight_coordinate % intervention_weight.shape[1],
    )
    return max_weight_coordinate[0].item(), max_weight_coordinate[1].item()


def assemble_input_sentences(example, tokenizer):
    base_input_prompt = None
    if (
        example["input_suffix"].endswith(" ")
        or example["input_suffix"].endswith('"')
        or example["input_suffix"].endswith("'")
        or example["input_suffix"].endswith("(")
    ):
        base_input_prompt = (
            tokenizer.bos_token
            + example["input_prefix"]
            + example["input_suffix"]
            + example["counterfactual_target"]
        )
    else:
        base_input_prompt = (
            tokenizer.bos_token
            + example["input_prefix"]
            + example["input_suffix"]
            + " "
            + example["counterfactual_target"]
        )

    source_input_prompt = (
        tokenizer.bos_token
        + example["counterfactual_input_prefix"]
        + example["counterfactual_input_suffix"]
    )

    return base_input_prompt, source_input_prompt


def get_entity_positions(example, tokenizer):
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

    base_input_prompt, source_input_prompt = assemble_input_sentences(
        example, tokenizer
    )

    base_input_ids = tokenizer(
        base_input_prompt, return_tensors="pt", padding=False, truncation=False
    ).input_ids[0]
    source_input_ids = tokenizer(
        source_input_prompt, return_tensors="pt", padding=False, truncation=False
    ).input_ids[0]

    entity_token = example["entity"]
    counterfactual_entity_token = example["counterfactual_entity"]

    base_entity_position_ids = _find_entity_positions(
        [tokenizer.decode(ids) for ids in base_input_ids], entity_token
    )
    source_entity_position_ids = _find_entity_positions(
        [tokenizer.decode(ids) for ids in source_input_ids], counterfactual_entity_token
    )

    source_entity_token = source_input_ids[source_entity_position_ids]
    base_entity_token = base_input_ids[base_entity_position_ids]

    return (source_entity_position_ids, source_entity_token), (
        base_entity_position_ids,
        base_entity_token,
    )


def get_sentence_components(text, tokenizer, entity, label):
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

    json_syntax = ["{", "}", "[", "]", ":", ",", '"', "'", "},", ",{", "{'", "'}"]

    components_breakdown_idx = {
        "BOS Token": [],
        "Subject Tokens": [],
        "Sentence Last Token": [],
        "JSON Syntax": [],
        "Country": [],
        "Others": [],
        "Label": [],
    }

    components_breakdown = {
        "BOS Token": [],
        "Subject Tokens": [],
        "Sentence Last Token": [],
        "JSON Syntax": [],
        "Country": [],
        "Others": [],
        "Label": [],
    }

    tokens = tokenizer(text)["input_ids"]
    textual_tokens = [tokenizer.decode([t]) for t in tokens]

    entity_token_index = _find_entity_positions(textual_tokens, entity)
    if label is not None:
        label_token_index = _find_entity_positions(textual_tokens, label)
    else:
        label_token_index = []

    last_token_index = (
        label_token_index[-1] - 1 if label is not None else len(textual_tokens) - 1
    )
    # last_token_index = entity_token_index[-1]

    for i, token in enumerate(textual_tokens):
        if i == 0:
            components_breakdown_idx["BOS Token"].append(i)
            components_breakdown["BOS Token"].append(token)
        elif i == last_token_index:
            components_breakdown_idx["Sentence Last Token"].append(i)
            components_breakdown["Sentence Last Token"].append(token)
        elif token.strip() in json_syntax:
            components_breakdown_idx["JSON Syntax"].append(i)
            components_breakdown["JSON Syntax"].append(token)
        elif i in entity_token_index:
            components_breakdown_idx["Subject Tokens"].append(i)
            components_breakdown["Subject Tokens"].append(token)
        elif token.strip() == "country":
            components_breakdown_idx["Country"].append(i)
            components_breakdown["Country"].append(token)
        else:
            components_breakdown_idx["Others"].append(i)
            components_breakdown["Others"].append(token)

    return components_breakdown, components_breakdown_idx


def get_max_weight_type(example, tokenizer):
    base_input_prompt, source_input_prompt = assemble_input_sentences(
        example, tokenizer
    )

    source_coord, base_coord = get_example_max_weight_coord(example, tokenizer)

    base_tokens = [
        tokenizer.decode([t]) for t in tokenizer(base_input_prompt)["input_ids"]
    ]
    source_tokens = [
        tokenizer.decode([t]) for t in tokenizer(source_input_prompt)["input_ids"]
    ]

    base_intervened_token = base_tokens[base_coord]
    source_intervened_token = source_tokens[source_coord]

    _, source_sentence_components_idx = get_sentence_components(
        source_input_prompt, tokenizer, example["counterfactual_entity"], None
    )
    _, base_sentence_components_idx = get_sentence_components(
        base_input_prompt,
        tokenizer,
        example["entity"],
        example["counterfactual_target"],
    )

    source_intervention_token_type = None
    base_intervention_token_type = None

    for key in source_sentence_components_idx:
        if source_coord in source_sentence_components_idx[key]:
            source_intervention_token_type = key
            break

    for key in base_sentence_components_idx:
        if base_coord in base_sentence_components_idx[key]:
            base_intervention_token_type = key
            break

    return (
        source_intervention_token_type,
        source_intervened_token,
        base_intervention_token_type,
        base_intervened_token,
    )
