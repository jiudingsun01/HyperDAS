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
from src.hyperdas.data_utils import get_ravel_collate_fn
from src.mdas.llama3.model import RavelMDASNetwork
import argparse



from torch import optim
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import get_scheduler


def run_experiment(
    log_wandb=True,
    wandb_project="hypernetworks-interpretor",
    wandb_run_name=None,
    intervention_layer=15,
    model_name_or_path="./models/llama3-8b",
    load_trained_from=None,
    batch_size=8,
    save_dir=None,
    n_epochs=1,
    das_dimension=None,
    lr=3e-5,
    eval_per_steps=100,
    checkpoint_per_steps=500,
    test_path=None,
    train_path=None,
    causal_loss_weight=1,
    intervention_location="last_entity_token",
):        
    if log_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "targetmodel": model_name_or_path, 
                "dataset": "ravel",
                "intervention_layer": intervention_layer,
                "das_dimension": das_dimension,
            },
        )
        
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    train_set = load_from_disk(train_path)
    test_set = load_from_disk(test_path)
                
    collate_fn = get_ravel_collate_fn(
        tokenizer, 
        source_suffix_visibility=True, 
        base_suffix_visibility=True, 
        add_space_before_target=True,
        contain_entity_position=True,
    )
    
    data_loader = DataLoader(
        train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    
    test_data_loader = DataLoader(
        test_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    
    model = RavelMDASNetwork(
        model_name_or_path=model_name_or_path,
        intervention_layer=intervention_layer,
        das_dimension=das_dimension,
        intervention_location=intervention_location,
    )
    
    model = model.cuda()
    
    model.run_train(
        train_loader=data_loader,
        test_loader=test_data_loader,
        epochs=n_epochs,
        checkpoint_per_steps = checkpoint_per_steps,
        eval_per_steps = eval_per_steps,
        save_dir=save_dir,
        causal_loss_weight=causal_loss_weight,
        lr=lr
    )

    if log_wandb:
        wandb.finish()
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_wandb", default=False, action="store_true")
    parser.add_argument("--wandb_project", type=str, default="HyperDAS")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--intervention_layer", type=int, default=15)
    
    parser.add_argument("--load_trained_from", type=str, default=None)
    
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--model_name_or_path", type=str, default="/nlp/scr/sjd24/llama3-8b")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--test_path", type=str, default="./experiments/RAVEL/data/nobel_prize_winner_field_test")
    parser.add_argument("--train_path", type=str, default="./experiments/RAVEL/data/nobel_prize_winner_field_train")
    parser.add_argument("--causal_loss_weight", type=float, default=1)
    
    parser.add_argument("--intervention_location", type=str, choices=["last_token", "last_entity_token"], default="last_entity_token")
        
    # if None, use Boundless DAS
    parser.add_argument("--das_dimension", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_per_steps", type=int, default=100)
    parser.add_argument("--checkpoint_per_steps", type=int, default=None)
    
    args = parser.parse_args()
    args = dict(args.__dict__)
    run_experiment(**args)