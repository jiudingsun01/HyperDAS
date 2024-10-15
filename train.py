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
import argparse


from transformers import AutoTokenizer


def run_experiment(
    log_wandb=True,
    wandb_project="hypernetworks-interpretor",
    wandb_run_name=None,
    inference_modes=["default", "bidding_argmax"],
    intervention_layer=15,
    subspace_module="ReflectSelect",
    model_name_or_path="./models/llama3-8b",
    load_trained_from=None,
    batch_size=8,
    source_suffix_visibility=False,
    base_suffix_visibility=False,
    source_selection_sparsity_loss=True,
    sparsity_loss_weight_start=0.5,
    sparsity_loss_weight_end=1,
    sparsity_loss_warm_up_ratio=0.1,
    save_dir=None,
    n_epochs=1,
    das_dimension=None,
    lr=3e-5,
    weight_decay=0.01,
    eval_per_steps=100,
    checkpoint_per_steps=500,
    test_path=None,
    train_path=None,
    causal_loss_weight=1,
    iso_loss_weight=1,
    num_decoders=8,
    initialize_from_scratch=False,
    ablate_base_token_attention=False,
    ablate_source_token_attention=False,
    save_model=False,
    break_asymmetric=False,
    target_intervention_num=None,
):
    
    """if save_dir is not None:
        save_dir = os.path.join("./models", save_dir)"""
        
    
        
    if log_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "targetmodel": model_name_or_path, 
                "editormodel": model_name_or_path, 
                "dataset": "ravel",
                "intervention_layer": intervention_layer,
                "subspace_module": subspace_module,
                "source_suffix_visibility": source_suffix_visibility,
                "base_suffix_visibility": base_suffix_visibility,
                "das_dimension": das_dimension,
            },
        )

    if "default" in inference_modes:
        inference_modes.remove("default")
        inference_modes.append(None)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    train_set = load_from_disk(train_path)
    test_set = load_from_disk(test_path)
    
    # train_set = Dataset.from_list([d for d in train_set if d["attribute_type"] == "causal"])
    # test_set = Dataset.from_list([d for d in test_set if d["attribute_type"] == "causal"])

    collate_fn = get_ravel_collate_fn(
        tokenizer, 
        source_suffix_visibility=source_suffix_visibility, 
        base_suffix_visibility=base_suffix_visibility, 
        add_space_before_target=True,
        contain_entity_position="groundtruth" in inference_modes
    )
 
    data_loader = DataLoader(
        train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    
    test_data_loader = DataLoader(
        test_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )

    from src.hyperdas.llama3.model import RavelInterpretorHypernetwork

    hypernetwork = RavelInterpretorHypernetwork(
        model_name_or_path=model_name_or_path,
        num_editing_heads=32,
        intervention_layer=intervention_layer,
        subspace_module=subspace_module,
        das_dimension=das_dimension,
        chop_editor_at_layer=num_decoders,
        initialize_from_scratch=initialize_from_scratch,
        ablate_base_token_attention=ablate_base_token_attention,
        ablate_source_token_attention=ablate_source_token_attention,
        break_asymmetric=break_asymmetric
    )
    hypernetwork = hypernetwork.to("cuda")
    
    if load_trained_from is not None:
        hypernetwork.load_model(load_trained_from)

    # current problem: 1728 / 30864
    hypernetwork.run_train(
        train_loader=data_loader,
        test_loader=test_data_loader,
        inference_modes=inference_modes,
        epochs=n_epochs,
        checkpoint_per_steps = checkpoint_per_steps,
        eval_per_steps = eval_per_steps,
        save_dir=save_dir,
        apply_source_selection_sparsity_loss=source_selection_sparsity_loss,
        sparsity_loss_weight_start=sparsity_loss_weight_start,
        sparsity_loss_weight_end=sparsity_loss_weight_end,
        sparsity_loss_warm_up_ratio=sparsity_loss_warm_up_ratio,
        causal_loss_weight=causal_loss_weight,
        iso_loss_weight=iso_loss_weight,
        weight_decay=weight_decay, 
        lr=lr,
        save_model=save_model,
        target_intervention_num=target_intervention_num
    )

    if log_wandb:
        wandb.finish()
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_wandb", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="HyperDAS")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--intervention_layer", type=int, default=15)
    
    parser.add_argument("--load_trained_from", type=str, default=None)
    
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--model_name_or_path", type=str, default="/scr-ssd/sjd24/llama3-8b")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--source_suffix_visibility", default=False, action="store_true")
    parser.add_argument("--base_suffix_visibility", default=False, action="store_true")
 
    parser.add_argument("--test_path", type=str, default="./experiments/RAVEL/data/city_test")
    parser.add_argument("--train_path", type=str, default="./experiments/RAVEL/data/city_train")
    
    parser.add_argument("--source_selection_sparsity_loss", type=bool, default=True)
    parser.add_argument("--sparsity_loss_warm_up_ratio", type=float, default=0.25)
    parser.add_argument("--sparsity_loss_weight_start", type=float, default=0.0)
    parser.add_argument("--sparsity_loss_weight_end", type=float, default=1.0)
    
    parser.add_argument("--target_intervention_num", type=int, default=None)    
    parser.add_argument("--causal_loss_weight", type=float, default=3.5)
    parser.add_argument("--iso_loss_weight", type=float, default=1)
    
    parser.add_argument("--save_dir", type=str, default="/scr-ssd/sjd24/city_entropy_1014")
    parser.add_argument("--save_model", default=False, action="store_true")
        
    parser.add_argument('--inference_modes', nargs='+', default=["default", "bidding_argmax"])
    
    # Ablation
    parser.add_argument("--num_decoders", type=int, default=8)
    parser.add_argument("--initialize_from_scratch", default=False, action="store_true")
    parser.add_argument("--ablate_base_token_attention", default=False, action="store_true")
    parser.add_argument("--ablate_source_token_attention", default=False, action="store_true")
    parser.add_argument("--break_asymmetric", default=False, action="store_true")
    
    # if None, use Boundless DAS
    parser.add_argument('--subspace_module', default="ReflectSelect", choices=[None, "DAS", "BoundlessDAS", "MaskSelect", "ReflectSelect", "QuasiProjective"])
    parser.add_argument("--das_dimension", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_per_steps", type=int, default=2000)
    parser.add_argument("--checkpoint_per_steps", type=int, default=2000)
    
    args = parser.parse_args()
    args = dict(args.__dict__)
    run_experiment(**args)
