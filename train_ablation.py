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
from datasets import Dataset, load_from_disk, concatenate_datasets
from src.hyperdas.llama3.ablation import *
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
    save_dir=None,
    n_epochs=1,
    das_dimension=None,
    lr=3e-5,
    weight_decay=0.01,
    test_per_steps=100,
    checkpoint_per_steps=500,
    test_path=None,
    train_path=None,
    causal_loss_weight=1,
):
    if save_dir is not None:
        save_dir = os.path.join("./models", save_dir)
        
    
        
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
        
    entity_dictionary = generate_ravel_dictionary(concatenate_datasets([train_set, test_set]))
                    
    collate_fn = get_ravel_dictionary_collate_fn(
        tokenizer, 
        entity_dictionary,
        source_suffix_visibility=source_suffix_visibility, 
        base_suffix_visibility=base_suffix_visibility, 
        add_space_before_target=True
    )
    
    data_loader = DataLoader(
        train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    
    test_data_loader = DataLoader(
        test_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )

    hypernetwork = RavelAblatedInterpretorHypernetwork(
        model_name_or_path=model_name_or_path,
        num_concept=len(entity_dictionary),
        num_editing_heads=32,
        intervention_layer=intervention_layer,
        subspace_module=subspace_module,
        das_dimension=das_dimension,
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
        test_per_steps = test_per_steps,
        save_dir=save_dir,
        apply_source_selection_sparsity_loss=source_selection_sparsity_loss,
        causal_loss_weight=causal_loss_weight,
        weight_decay=weight_decay, 
        lr=lr
    )

    if log_wandb:
        wandb.finish()
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_wandb", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="HyperDAS")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--intervention_layer", type=int, default=12)
    
    parser.add_argument("--load_trained_from", type=str, default=None)
    
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--model_name_or_path", type=str, default="./models/llama3-8b")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--source_suffix_visibility", default=False, action="store_true")
    parser.add_argument("--base_suffix_visibility", default=False, action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--test_path", type=str, default="./experiments/ravel/data/city_country_test")
    parser.add_argument("--train_path", type=str, default="./experiments/ravel/data/city_country_train")
    parser.add_argument("--source_selection_sparsity_loss", type=bool, default=True)
    parser.add_argument("--causal_loss_weight", type=float, default=1)
        
    parser.add_argument('--inference_modes', nargs='+', default=["default", "column_argmax"])
    
    # if None, use Boundless DAS
    parser.add_argument('--subspace_module', default="DAS", choices=[None, "DAS", "BoundlessDAS", "MaskSelect", "ReflectSelect"])
    parser.add_argument("--das_dimension", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--test_per_steps", type=int, default=100)
    parser.add_argument("--checkpoint_per_steps", type=int, default=None)
    
    args = parser.parse_args()
    args = dict(args.__dict__)
    run_experiment(**args)