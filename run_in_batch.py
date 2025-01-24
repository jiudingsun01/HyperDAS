import sys

sys.path.append('../..')

from torch.utils.data import DataLoader
from datasets import load_from_disk
import torch
from src.hyperdas.data_utils import generate_ravel_dataset, get_ravel_collate_fn, filter_dataset
from train import run_experiment

from transformers import AutoTokenizer, LlamaForCausalLM
import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_layer", type=int)
    parser.add_argument("--end_layer", type=int)
    
    args = parser.parse_args()
    
    start_layer = args.start_layer
    end_layer = args.end_layer
    
    
    
    results = {}
    
    for layer in range(start_layer, end_layer, 2):
        run_experiment(
            log_wandb=True,
            wandb_project="ravel_country_layer_vs_accuracy_new_new_new",
            wandb_run_name=f"L{layer}",
            inference_modes=["default", "bidding_argmax"],
            intervention_layer=layer,
            subspace_module="ReflectSelect",
            model_name_or_path="/nlp/scr/sjd24/llama3-8b",
            load_trained_from=None,
            batch_size=16,
            source_suffix_visibility=False,
            base_suffix_visibility=False,
            source_selection_sparsity_loss=True,
            save_dir=f"/nlp/scr/sjd24/L{layer}",
            das_dimension=128,
            n_epochs=3,
            lr=4e-5,
            weight_decay=0.01,
            eval_per_steps=None,
            checkpoint_per_steps=None,
            test_path="./experiments/RAVEL/data/city_test",
            train_path="./experiments/RAVEL/data/city_train",
            causal_loss_weight=3.5
        )
        
        