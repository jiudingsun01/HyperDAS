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
from src.hyperdas.data_utils import get_collate_fn
import argparse


from transformers import AutoTokenizer


def run_experiment(
    log_wandb=True,
    wandb_project="hypernetworks-interpretor",
    wandb_run_name=None,
    debug_model=False,
    intervention_layer=15,
    subspace_module="ReflectSelect",
    model_name_or_path="./models/llama3-8b",
    load_trained_from=None,
    batch_size=8,
    save_dir=None,
    n_epochs=1,
    n_steps=1000,
    das_dimension=None,
    lr=3e-5,
    weight_decay=0.01,
    eval_per_steps=100,
    checkpoint_per_steps=500,
    test_path=None,
    train_path=None,
    causal_loss_weight=1,
    num_decoders=8,
    initialize_from_scratch=False,
    save_model=False,
    seed=None,
    sparsity_loss=True,
    sparsity_loss_weight=1.0,
    debug_mode=None,
):
    
    """if save_dir is not None:
        save_dir = os.path.join("./models", save_dir)"""
    
    if seed is not None:
        random.seed(args["seed"])
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        torch.cuda.manual_seed(args["seed"])
        torch.cuda.manual_seed_all(args["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
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
                "das_dimension": das_dimension,
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    if "llama" in model_name_or_path:
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif "gemma" in model_name_or_path:
        tokenizer.padding_side = "right"

    train_set = load_from_disk(train_path)
    test_set = load_from_disk(test_path)
    
    collate_fn = get_collate_fn(tokenizer)
 
    data_loader = DataLoader(
        train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    
    test_data_loader = DataLoader(
        test_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )

    from src.hyperdas.llama3.model import RavelInterpretorHypernetwork

    hypernetwork = RavelInterpretorHypernetwork(
        target_model_name_or_path=model_name_or_path,
        num_editing_heads=32,
        intervention_layer=intervention_layer,
        subspace_module=subspace_module,
        das_dimension=das_dimension,
        chop_editor_at_layer=num_decoders,
        initialize_from_scratch=initialize_from_scratch,
    )
    hypernetwork = hypernetwork.to("cuda")
    
    if load_trained_from is not None:
        hypernetwork.load_model(load_trained_from)

    hypernetwork.run_train(
        train_loader=data_loader,
        test_loader=test_data_loader,
        epochs=n_epochs,
        n_steps=n_steps,
        checkpoint_per_steps = checkpoint_per_steps,
        eval_per_steps = eval_per_steps,
        save_dir=save_dir,
        causal_loss_weight=causal_loss_weight,
        weight_decay=weight_decay, 
        lr=lr,
        save_model=save_model,
        sparsity_loss=sparsity_loss,
        sparsity_loss_weight=sparsity_loss_weight,
        debug_mode=debug_mode,
    )

    if log_wandb:
        wandb.finish()
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_wandb", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="HyperDAS-Symmetry-Real")
    parser.add_argument("--wandb_run_name", type=str, default="City-HouseHolder")
    parser.add_argument("--intervention_layer", type=int, default=15)
    
    parser.add_argument("--load_trained_from", type=str, default=None)
    
    parser.add_argument("--n_epochs", type=int, default=15)
    parser.add_argument("--n_steps", type=int, default=5000)
    
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--batch_size", type=int, default=16)
    
    parser.add_argument("--test_path", type=str, default="./experiments/Axbench/data/axbench16k_test")
    parser.add_argument("--train_path", type=str, default="./experiments/Axbench/data/axbench16k_train")
     
    parser.add_argument("--causal_loss_weight", type=float, default=3.5)
    
    parser.add_argument("--save_dir", type=str, default="./test_axbench")
    parser.add_argument("--save_model", default=False, action="store_true")
    
    parser.add_argument("--num_decoders", type=int, default=2)
    parser.add_argument("--initialize_from_scratch", default=False, action="store_true")
    
    # Sparsity Loss
    parser.add_argument("--sparsity_loss", default=True)
    parser.add_argument("--sparsity_loss_weight", type=float, default=1)
        
    # if None, use Boundless DAS
    parser.add_argument('--subspace_module', default="SelectiveLoReFT", choices=["LoReFT", "SelectiveLoReFT", "SteeringVec"])
    parser.add_argument("--das_dimension", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    parser.add_argument("--eval_per_steps", type=int, default=None)
    parser.add_argument("--checkpoint_per_steps", type=int, default=None)
    
    parser.add_argument('--debug_mode', default=None, choices=[None, "last_entity_token"])
    
    args = parser.parse_args()
    args = dict(args.__dict__)
    
    if not os.path.exists(args["save_dir"]):
        os.makedirs(args["save_dir"])
    
    if args["save_dir"] is not None:
        json.dump(args, open(os.path.join(args["save_dir"], "args.json"), "w"))
    
    run_experiment(**args)
