import sys

sys.path.append('../..')

from torch.utils.data import DataLoader
from datasets import load_from_disk
import torch
from src.hyperdas.data_utils import generate_ravel_dataset, get_ravel_collate_fn, filter_dataset
from train import run_experiment

from transformers import AutoTokenizer, LlamaForCausalLM


if __name__ == "__main__":
    
    results = {}
    
    for layer in range(1, 32, 2):
        run_experiment(
            log_wandb=True,
            wandb_project="ravel_country_layer_vs_accuracy_new_new",
            wandb_run_name=f"L{layer}",
            inference_modes=["default", "bidding_argmax"],
            intervention_layer=layer,
            subspace_module="ReflectSelect",
            model_name_or_path="../../models/llama3-8b",
            load_trained_from=None,
            batch_size=16,
            source_suffix_visibility=False,
            base_suffix_visibility=False,
            source_selection_sparsity_loss=True,
            save_dir=None,
            das_dimension=128,
            n_epochs=3,
            lr=4e-5,
            weight_decay=0.01,
            eval_per_steps=None,
            checkpoint_per_steps=None,
            test_path="../../experiments/ravel/data/city_test",
            train_path="../../experiments/ravel/data/city_train",
            causal_loss_weight=3.5
        )
        
        