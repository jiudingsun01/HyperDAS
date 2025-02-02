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

from src.hyperdas.asymmetric.configs import (
    HouseholderConfig, 
    AsymmetricHyperDASConfig, 
    QuasiProjectiveConfig, 
    SymmetricHyperDASConfig,
    HyperDASConfig
)


from transformers import AutoTokenizer, AutoConfig


def run_experiment(
    log_wandb=True,
    wandb_project="hypernetworks-interpretor",
    wandb_run_name=None,
    inference_modes=["default", "bidding_argmax"],
    intervention_layer=21,
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
    save_model=False,
    hyperdas_config: HyperDASConfig = None,
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
    
    if "llama" in model_name_or_path:
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif "gemma" in model_name_or_path:
        tokenizer.padding_side = "right"

    train_set = load_from_disk(train_path)
    test_set = load_from_disk(test_path)
                
    collate_fn = get_ravel_collate_fn(
        tokenizer, 
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

    from src.hyperdas.asymmetric.hypernet import RavelInterpretorHypernetwork

    hypernetwork = RavelInterpretorHypernetwork(
        tokenizer=tokenizer,
        config=hyperdas_config
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
        save_model=save_model
    )

    if log_wandb:
        wandb.finish()
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Wadb Config
    parser.add_argument("--log_wandb", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="HyperDAS")
    parser.add_argument("--wandb_run_name", type=str, default="Sym-Full")
    parser.add_argument("--intervention_layer", type=int, default=15)
    
    # HyperDAS Config
    parser.add_argument('--token_selection_module', default="asymmetric", choices=["asymmetric", "symmetric"])
    parser.add_argument('--subspace_module', default="ReflectSelect", choices=[None, "DAS", "BoundlessDAS", "MaskSelect", "ReflectSelect", "QuasiProjective"])
    parser.add_argument("--das_dimension", type=int, default=128)
    parser.add_argument("--load_trained_from", type=str, default=None)
    # parser.add_argument("--model_name_or_path", type=str, default="/nlp/scr/sjd24/cache/hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819")
    parser.add_argument("--model_name_or_path", type=str, default="/nlp/scr/sjd24/llama3-8b")
    parser.add_argument("--num_decoders", type=int, default=4)
    parser.add_argument("--num_editing_heads", type=int, default=32)
    parser.add_argument("--initialize_from_pretrained", default=False, action="store_true")
    
    # Training Config
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_path", type=str, default="./experiments/RAVEL/data/city_test")
    parser.add_argument("--train_path", type=str, default="./experiments/RAVEL/data/city_train")
    
    parser.add_argument("--causal_loss_weight", type=float, default=3.5)
    parser.add_argument("--iso_loss_weight", type=float, default=0.5)
    parser.add_argument("--save_model", default=False, action="store_true")
    parser.add_argument("--save_dir", type=str, default="/scr-ssd/sjd24/HyperDAS/RAVEL-Sym")
    
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_per_steps", type=int, default=500)
    parser.add_argument("--checkpoint_per_steps", type=int, default=None)
    
    parser.add_argument("--source_selection_sparsity_loss", type=bool, default=True)
    parser.add_argument("--sparsity_loss_warm_up_ratio", type=float, default=0.5)
    parser.add_argument("--sparsity_loss_weight_start", type=float, default=0.15)
    parser.add_argument("--sparsity_loss_weight_end", type=float, default=1.5)
    
    # RAVEL Config
    parser.add_argument("--source_suffix_visibility", default=False, action="store_true")
    parser.add_argument("--base_suffix_visibility", default=False, action="store_true")
    
    # Asymmetric Implementation
    parser.add_argument('--inference_modes', nargs='+', default=["bidding_argmax"])
    parser.add_argument("--ablate_base_token_attention", default=False, action="store_true")
    parser.add_argument("--ablate_source_token_attention", default=False, action="store_true")
    parser.add_argument("--break_asymmetric", default=False, action="store_true")
    
    # QuasiProjective Config
    parser.add_argument('--selection_mechanism', default="dynamic", choices=["dynamic", "topk", "full"])
    parser.add_argument('--ridge_parameterization', default=None, choices=[None, "topk_ste"])
    parser.add_argument("--dict_size", type=int, default=32)
    parser.add_argument("--scoring_dimension", type=int, default=32)
    parser.add_argument("--lambda_parameter", type=float, default=0.001)
    parser.add_argument("--importance_power", type=int, default=-2)
    parser.add_argument("--epsilon", type=float, default=0.000001)
    parser.add_argument("--return_penalty", type=bool, default=True)
    parser.add_argument("--compute_metrics", type=bool, default=True)
    parser.add_argument("--orthogonal_init", type=bool, default=True)
    parser.add_argument("--hat_matrix", type=bool, default=True)
    
    args = parser.parse_args()
    args = dict(args.__dict__)
    
    
    if args["subspace_module"] == 'QuasiProjective':
        subspace_config = QuasiProjectiveConfig()
        
        subspace_config.dict_size = args.pop("dict_size")
        subspace_config.scoring_dimension = args.pop("scoring_dimension")
        subspace_config.lambda_parameter = args.pop("lambda_parameter")
        subspace_config.importance_power = args.pop("importance_power")
        subspace_config.epsilon = args.pop("epsilon")
        subspace_config.return_penalty = args.pop("return_penalty")
        subspace_config.ridge_parameterization = args.pop("ridge_parameterization")
        subspace_config.compute_metrics = args.pop("compute_metrics")
        subspace_config.orthogonal_init = args.pop("orthogonal_init")
        subspace_config.selection_mechanism = args.pop("selection_mechanism")
        subspace_config.hat_matrix = args.pop("hat_matrix")
        
    elif args["subspace_module"] == 'ReflectSelect':
        subspace_config = HouseholderConfig()
        args.pop("dict_size")
        args.pop("scoring_dimension")
        args.pop("lambda_parameter")
        args.pop("importance_power")
        args.pop("epsilon")
        args.pop("return_penalty")
        args.pop("ridge_parameterization")
        args.pop("compute_metrics")
        args.pop("orthogonal_init")
        args.pop("selection_mechanism")
        args.pop("hat_matrix")
    else:
        raise NotImplementedError(f"Subspace module {args['subspace_module']} not implemented.")
    
    subspace_config.subspace_dimension = args.pop("das_dimension")
    
    token_selection_module = args.pop("token_selection_module")
                                                  
    
    if token_selection_module == "asymmetric":
        hyperdas_config = AsymmetricHyperDASConfig.from_pretrained("/nlp/scr/sjd24/llama3-8b")
        hyperdas_config.break_asymmetric = args.pop("break_asymmetric")
        hyperdas_config.ablate_base_token_attention = args.pop("ablate_base_token_attention")
        hyperdas_config.ablate_source_token_attention = args.pop("ablate_source_token_attention")
        hyperdas_config.inference_modes = args.pop("inference_modes")
        
    elif token_selection_module == "symmetric":
        hyperdas_config = SymmetricHyperDASConfig.from_pretrained("/nlp/scr/sjd24/llama3-8b")
        
    target_model_config = AutoConfig.from_pretrained(args["model_name_or_path"])
    
    hyperdas_config.num_decoders = args.pop("num_decoders")
    hyperdas_config.num_editing_heads = args.pop("num_editing_heads")
    hyperdas_config.intervention_layer = args.pop("intervention_layer")
    hyperdas_config.initialize_from_pretrained = args.pop("initialize_from_pretrained")
    
    hyperdas_config.hidden_size = target_model_config.hidden_size
    hyperdas_config.intermediate_size = target_model_config.intermediate_size
    hyperdas_config.vocab_size = target_model_config.vocab_size
    hyperdas_config.target_model_num_hidden_layers = target_model_config.num_hidden_layers
    hyperdas_config.target_model_name_or_path = args["model_name_or_path"]
    
    hyperdas_config._attn_implementation = 'eager'
    hyperdas_config.subspace_config = subspace_config
    
        
    if not os.path.exists(args["save_dir"]):
        os.makedirs(args["save_dir"])
        
    json.dump(args, open(os.path.join(args["save_dir"], "config.json"), "w"))
    run_experiment(**args, hyperdas_config=hyperdas_config)