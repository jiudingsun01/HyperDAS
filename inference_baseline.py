import argparse

import torch
from torch import compile
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import os
import time
import sys
import wandb
import random
import numpy as np
import json
from tqdm import tqdm
from datasets import Dataset, load_from_disk
from src.hyperdas.data_utils import get_ravel_collate_fn, generate_ravel_dataset_from_filtered
from src.hyperdas.utils import add_fwd_hooks
import argparse
from pyvene import IntervenableConfig, RepresentationConfig, LowRankRotatedSpaceIntervention, IntervenableModel, count_parameters

from torch import optim
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import get_scheduler


def run_experiment(
    intervention_layer=15,
    model_name_or_path="./models/llama3-8b",
    checkpoint_path=None,
    batch_size=16,
    save_dir=None,
    das_dimension=128,
    test_path=None,
    intervention_location="last_entity_token",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    test_set = load_from_disk(test_path)
                
    collate_fn = get_ravel_collate_fn(
        tokenizer, 
        source_suffix_visibility=True, 
        base_suffix_visibility=True, 
        add_space_before_target=True,
        contain_entity_position=True,
    )

    test_data_loader = DataLoader(
        test_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )

    model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
    model = model.to("cuda")

    intervention_config = IntervenableConfig(
        model_type=type(model),
        representations=[
        RepresentationConfig(
                intervention_layer,  # layer
                'block_output',  # intervention repr
                "pos",  # intervention unit
                1,  # max number of unit
                das_dimension)
        ],
        intervention_types=LowRankRotatedSpaceIntervention,
    )

    intervenable = IntervenableModel(intervention_config, model)
    intervenable.set_device(model.device)
    intervenable.disable_model_gradients()
    
    intervention_key = list(intervenable.interventions.keys())[0]
    intervenable.interventions[intervention_key][0].load_state_dict(torch.load(checkpoint_path))
    
    
    def forward(
        base_input_ids: torch.Tensor = None,
        base_attention_mask: torch.Tensor = None,
        base_intervention_position: torch.Tensor = None,
        base_position_ids: torch.Tensor = None,
        source_input_ids: torch.Tensor = None,
        source_attention_mask: torch.Tensor = None,
        source_intervention_position: torch.Tensor = None,
        source_position_ids: torch.Tensor = None,
        intervention_layer: int = None,
    ):
        if intervention_layer is None:
            raise ValueError("intervention_layer must be specified")
        
        if base_position_ids is None:
            # 0 for all the padding tokens and start from 1 for the rest
            base_position_ids = torch.cumsum(base_attention_mask, dim=1) * base_attention_mask
        
        if source_position_ids is None:
            source_position_ids = torch.cumsum(source_attention_mask, dim=1) * source_attention_mask
        
        # print(source_intervention_position.unsqueeze(0).shape, base_intervention_position.unsqueeze(0).shape)
        b_s = base_input_ids.shape[0]
        intervention_locations = {
            "sources->base": (
                source_intervention_position.unsqueeze(0).unsqueeze(-1),
                base_intervention_position.unsqueeze(0).unsqueeze(-1)
            )
        }
        
        _, counterfactual_outputs = intervenable(
            {
                "input_ids": base_input_ids,
                'attention_mask': base_attention_mask,
                'position_ids': base_position_ids
            }, [
                {
                    "input_ids": source_input_ids,
                    'attention_mask': source_attention_mask,
                    'position_ids': source_position_ids
                }
            ] , intervention_locations
        )
        
        return counterfactual_outputs
    
            
    def eval_accuracy(test_loader, eval_n_label_tokens=3):
        
        intervenable.eval()
        correct_idxs = []
        is_causal = []
        
        with torch.no_grad():
            for batch_id, batch in enumerate(test_loader):
                
                if intervention_location == "last_entity_token":
                    base_intervention_position = batch["base_entity_position_ids"].to("cuda") 
                    source_intervention_position = batch["source_entity_position_ids"].to("cuda")
                else:
                    base_intervention_position = batch["base_input_ids"].shape[1] - 1
                    source_intervention_position = batch["source_input_ids"].shape[1] - 1
                    
                    base_intervention_position = torch.tensor([base_intervention_position] * batch["base_input_ids"].shape[0]).to("cuda")
                    source_intervention_position = torch.tensor([source_intervention_position] * batch["source_input_ids"].shape[0]).to("cuda")
                
                output = forward(
                    base_input_ids=batch["base_input_ids"].to("cuda"),
                    base_attention_mask=batch["base_attention_mask"].to("cuda"),
                    base_intervention_position=base_intervention_position,
                    source_input_ids=batch["source_input_ids"].to("cuda"),
                    source_attention_mask=batch["source_attention_mask"].to("cuda"),
                    source_intervention_position=source_intervention_position,
                    intervention_layer=intervention_layer,
                )
                
                logits = output.logits
                                
                batch_pred_ids = torch.argmax(logits, dim=-1)
                is_causal.extend(batch["is_causal"].cpu().numpy().tolist())
                
                for i, (label, pred_ids) in enumerate(zip(batch["labels"].to("cuda"), batch_pred_ids)):
                    label_idx = label != -100
                    output_idx = torch.zeros_like(label_idx)
                    output_idx[:-1] = label_idx[1:]
                    
                    label = label[label_idx]
                    pred_ids = pred_ids[output_idx]
                    
                    if eval_n_label_tokens is not None and len(label) > eval_n_label_tokens:
                        label = label[:eval_n_label_tokens]
                        pred_ids = pred_ids[:eval_n_label_tokens]
                    
                    is_correct = (torch.sum (label == pred_ids) == torch.numel(label)).item()    
                    if is_correct:
                        correct_idxs.append(batch_id * len(batch["labels"]) + i)
                
                
        total_causal = sum(is_causal)
        total_isolate = len(is_causal) - total_causal
        
        correct_causal = sum([is_causal[i] for i in correct_idxs])
        correct_isolate = len(correct_idxs) - correct_causal
        
        causal_acc = correct_causal / total_causal if total_causal > 0 else 0.0
        isolate_acc = correct_isolate / total_isolate if total_isolate > 0 else 0.0
        
        disentangle_acc = 0.5 * (causal_acc + isolate_acc) if total_isolate > 0 else causal_acc
        
        accuracies = {
            "causal": causal_acc,
            "isolate": isolate_acc,
            "disentangle": disentangle_acc
        }
                    
        return accuracies
    
    acc = eval_accuracy(test_data_loader)
    for key, value in acc.items():
        print(f"{key}: {value}")
        
    if save_dir is not None:
        with open(os.path.join(save_dir), "w") as f:
            json.dump(acc, f)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervention_layer", type=int, default=15)
    parser.add_argument("--checkpoint_path", type=str, default="/nlp/scr/sjd24/HyperDAS/MDAS/MDAS_All/ravel_hyperdas_verb_singular/final_das_module.pt")
    parser.add_argument("--model_name_or_path", type=str, default="/nlp/scr/sjd24/llama3-8b")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default="/nlp/scr/sjd24/HyperDAS/")
    parser.add_argument("--test_path", type=str, default="./experiments/RAVEL/data/ICLR/verb_definition")
    
    # if None, use Boundless DAS
    parser.add_argument("--intervention_location", type=str, choices=["last_token", "last_entity_token"], default="last_entity_token")
    parser.add_argument("--das_dimension", type=int, default=128)
    
    args = parser.parse_args()
    args = dict(args.__dict__)
    run_experiment(**args)