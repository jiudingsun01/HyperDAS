import torch
from.modules import LlamaInterpretorConfig, LlamaInterpretor
from ..utils import InterpretorModelOutput
from transformers import AutoTokenizer, AutoConfig
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import os
from tqdm import tqdm
import time
import json
import numpy as np
from ..das_utils import QuasiProjectiveIntervention


class RavelInterpretorHypernetwork(nn.Module):
    # Separating the editor config file, from its base model's configurations
    def __init__(
        self,
        model_name_or_path="meta-llama/Meta-Llama-3-8B",
        target_model_name_or_path="meta-llama/Meta-Llama-3-8B",
        num_editing_heads=32,
        chop_editor_at_layer=8,
        intervention_layer=0,
        subspace_module="ReflectSelect",
        torch_dtype=torch.bfloat16,
        das_dimension=None,
        initialize_from_scratch=False,
    ):
        super().__init__()

        self.interpretor_config = LlamaInterpretorConfig.from_pretrained(model_name_or_path)
        self.interpretor_config.name_or_path = model_name_or_path
        self.interpretor_config.torch_dtype = torch_dtype
        self.interpretor_config.num_editing_heads = num_editing_heads
        self.interpretor_config.chop_editor_at_layer = chop_editor_at_layer
        self.interpretor_config.intervention_layer = intervention_layer
        self.interpretor_config._attn_implementation = 'eager'
        self.interpretor_config.initialize_from_scratch = initialize_from_scratch
        
        target_model_config = AutoConfig.from_pretrained(target_model_name_or_path)
        self.interpretor_config.num_target_model_layers = target_model_config.num_hidden_layers
        
        if self.interpretor_config.hidden_size != target_model_config.hidden_size:
            print(f"Changing the hidden size ({self.interpretor_config.hidden_size}) of the editor to match the target model's hidden size ({target_model_config.hidden_size})")
            self.interpretor_config.hidden_size = target_model_config.hidden_size
        
        if self.interpretor_config.intermediate_size != target_model_config.intermediate_size:
            print(f"Changing the intermediate size ({self.interpretor_config.intermediate_size}) of the editor to match the target model's intermediate size ({target_model_config.intermediate_size})")
            self.interpretor_config.intermediate_size = target_model_config.intermediate_size
            
        if self.interpretor_config.num_attention_heads != target_model_config.num_attention_heads:
            print(f"Changing the number of attention heads ({self.interpretor_config.num_attention_heads}) of the editor to match the target model's number of attention heads ({target_model_config.num_attention_heads})")
            self.interpretor_config.num_attention_heads = target_model_config.num_attention_heads
        
        self.interpretor = LlamaInterpretor(
            self.interpretor_config, 
            target_model_name_or_path=target_model_name_or_path,
            subspace_module=subspace_module, 
            das_dimension=das_dimension,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name_or_path)
        
        if "llama" in model_name_or_path:
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        elif "gemma" in model_name_or_path:
            self.tokenizer.padding_side = "right"

        self.use_das_intervention = subspace_module != None
        self.das_dim = das_dimension
        self.residual_cache = None
        self.opt = None
        # self.training_loss = None
        
        # DAS Training Hyperparameters
        self.rotate_lr = 1e-3
        self.boundary_lr = 1e-2
        self.das_temperature_start = 50.0
        self.das_temperature_end = 0.1
        
    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        torch.save(self.interpretor.hypernetwork.state_dict(), os.path.join(save_dir, "hypernetwork.pt"))
        if self.use_das_intervention:
            torch.save(self.interpretor.das_module.state_dict(), os.path.join(save_dir, "das.pt"))
        
    def load_model(self, load_dir):
        self.interpretor.hypernetwork.load_state_dict(torch.load(os.path.join(load_dir, "hypernetwork.pt")))
        if self.use_das_intervention:
            self.interpretor.das_module.load_state_dict(torch.load(os.path.join(load_dir, "das.pt")))
            
    def set_intervention_layer(self, intervention_layer):
        self.interpretor.config.intervention_layer = intervention_layer
        
    def forward(
        self,
        editor_input_ids: torch.Tensor = None,
        base_input_ids: torch.Tensor = None,
        base_attention_mask: torch.Tensor = None,
        base_intervention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        is_causal: torch.Tensor = None,
        causal_loss_weight: float = 1.0,
        base_intervention_weight: torch.Tensor = None,
    ):
        
        _pred: InterpretorModelOutput = self.interpretor(
            editor_input_ids=editor_input_ids,
            editor_attention_mask=editor_input_ids != self.interpretor_config.eos_token_id,
            base_input_ids=base_input_ids,
            base_attention_mask=base_attention_mask,
            base_intervention_mask=base_intervention_mask,
            base_intervention_weight=base_intervention_weight,
        )
        
        if labels is not None:
            log_prob_predictions = torch.nn.functional.log_softmax(
                _pred.logits.reshape(-1, _pred.logits.shape[-1]),
                dim=1,
            )
            
            if is_causal is not None:
                loss_weight = torch.ones_like(labels, dtype=log_prob_predictions.dtype)
                loss_weight[is_causal, :] = causal_loss_weight
                loss_weight[~is_causal, :] = 1
            
            labels = labels.reshape(-1)
            
            if is_causal is not None:
                loss_weight = loss_weight.reshape(-1)

            assert labels.shape == log_prob_predictions.shape[:-1]
            
            # Only consider the tokens that are not -100 in target_labels
            label_indices = labels != -100
            output_idices = torch.zeros_like(label_indices)
            output_idices[:-1] = label_indices[1:]
            
            log_prob_predictions = log_prob_predictions[output_idices, :]
        
            labels = labels[label_indices]
            
            # Compute the cross-entropy loss with masking
            
            if is_causal is None:
                criterion = torch.nn.CrossEntropyLoss(reduction="mean")
                loss = criterion(log_prob_predictions, labels.long())
            else:
                loss_weight = loss_weight[label_indices]
                criterion = torch.nn.CrossEntropyLoss(reduction="none")
                loss = criterion(log_prob_predictions, labels.long())
                                
                loss = (loss * loss_weight).mean()
            
            
            _pred["loss"] = loss
        
        return _pred
        
    # Generate text using the target model, with a new edit application at every step.
    # This is a very slow way to generate text.
    # If you only want to edit first k tokens, use the forward pass instead with stop_editing_index = k
    def inspect_batch_prediction_ouptuts(self, batch, eval_n_label_tokens=None):
        
        self.interpretor.eval()
        
        correct_idxs = []
        
        with torch.no_grad():
            
            predictions = self.forward(
                editor_input_ids=batch["editor_input_ids"].to("cuda"),
                base_input_ids=batch["base_input_ids"].to("cuda"),
                base_attention_mask=batch["base_attention_mask"].to("cuda"),
                base_intervention_mask=batch["base_intervention_mask"].to("cuda"),
                labels=batch["labels"].to("cuda"),
            )    
            
            batch_pred_ids = torch.argmax(predictions["logits"], dim=-1)
            batch_full_output = self.tokenizer.batch_decode(batch_pred_ids, skip_special_tokens=True)
            
            batch_output = []
            correct = 0
            
            for i, (label, pred_ids) in enumerate(zip(batch["labels"].to("cuda"), batch_pred_ids)):
                
                label_idx = label != -100
                output_idx = torch.zeros_like(label_idx)
                output_idx[:-1] = label_idx[1:]
                
                label = label[label_idx]
                pred_ids = pred_ids[output_idx]
                
                if eval_n_label_tokens is not None and len(label) > eval_n_label_tokens:
                    label = label[:eval_n_label_tokens]
                    pred_ids = pred_ids[:eval_n_label_tokens]
                
                batch_output.append(
                    self.tokenizer.decode(pred_ids, skip_special_tokens=True)
                )
                
                is_correct = torch.sum(label == pred_ids) == torch.numel(label)
                
                if is_correct:
                    correct_idxs.append(i)
                correct += is_correct
                
        base_intervention_weight = predictions.base_intervention_weight
                
        return_dict = {
            "batch_output": batch_output,
            "batch_full_output": batch_full_output,
            "batch_base_intervention_weight": base_intervention_weight,
            "n_correct": correct,
            "correct_idxs": correct_idxs
        }   
        return return_dict
        
    
    def plot_heatmap(
        self, 
        data_loader, 
        idxs, 
        batch_size=4, 
        annot=True,
        indicate_masked_tokens=False,
        digits=2,
        font_scale=1.0,
        contain_title=True,
        simplified_annot=True,
        axes=None
    ):
        
        fig = None
        batch_id = idxs // batch_size
        example_id = idxs % batch_size            
        
        for i, batch in enumerate(data_loader):
            if i == batch_id:
                break
            
        editor_input_ids = batch["editor_input_ids"][example_id]
        base_input_ids = batch["base_input_ids"][example_id]
        label = batch["labels"][example_id]
        
        base_padding_idx = base_input_ids == self.tokenizer.pad_token_id
        
        base_input_ids = base_input_ids[~base_padding_idx]
        
        if indicate_masked_tokens:
            base_intervention_mask = batch["base_intervention_mask"][example_id]
            base_intervention_mask = base_intervention_mask[~base_padding_idx]
                    
        # Add a False value to the end of the source_padding_idx to account for the [SELF] token
        
        base_axis = [self.tokenizer.decode([i]) for i in base_input_ids]
        
        for axis in [base_axis]:
            for i, token in enumerate(axis):
                if token == self.tokenizer.bos_token:
                    axis[i] = "[BOS]"
        
        editor_text = self.tokenizer.decode(editor_input_ids, skip_special_tokens=True)
        label = label[label != -100]
        label = self.tokenizer.decode(label)
    
        results = self.inspect_batch_prediction_ouptuts(batch, eval_n_label_tokens=3)
        predictions = results['batch_output'][example_id]
        base_intervention_weight = results["batch_base_intervention_weight"][example_id]
        
        # set background color to be grey
    
        style = sns.axes_style("dark")
        style["axes.facecolor"] = "#100a17"
            
        sns.set(style=style, font_scale=font_scale)
        if axes is None:
            fig, ax = plt.subplots(1, 1, figsize=(15, 4))
            
        base_intervention_weight = base_intervention_weight[~base_padding_idx].unsqueeze(0)
        base_intervention_mask = (base_intervention_mask == 0).unsqueeze(0)
        
        sns.heatmap(base_intervention_weight.float().cpu().numpy(), xticklabels=base_axis, ax=ax, annot=annot, fmt=f".{digits}f", mask=base_intervention_mask.float().cpu().numpy())
        
        axes[0].set_yticklabels([""])
        axes[1].set_yticklabels([""])
        
        axes[0].set_ylabel("Source Sentence")
        axes[1].set_ylabel("Base Sentence")
        
        axes[0].set_title(f"Instruction: {editor_text}     Label: {label}    Pred: {predictions}")
        
        plt.subplots_adjust(hspace=1) 
        
        return fig, axes
        
        
    def eval_accuracy(self, test_loader, eval_n_label_tokens=None, debug_mode=None):
        
        self.interpretor.eval()
        test_loss = []
        correct_idxs = []
        is_causal = []
        
        with torch.no_grad():
            for batch_id, batch in tqdm(
                enumerate(test_loader), desc="Evaluating", total=len(test_loader)
            ):         
                
                base_intervention_weight = None
                
                if debug_mode is not None:
                    if debug_mode == "last_token":                    
                        base_last_token_idxs = torch.sum(batch["labels"] == -100, dim=-1) - 1
                        base_intervention_weight = torch.functional.F.one_hot(base_last_token_idxs, num_classes=batch["base_intervention_mask"].shape[-1])

                    elif debug_mode == "last_entity_token":
                        base_intervention_weight = torch.functional.F.one_hot(batch["base_entity_position_ids"], num_classes=batch["base_intervention_mask"].shape[-1])
                    
                    base_intervention_weight = base_intervention_weight.float().to("cuda")
                
                predictions = self.forward(
                    editor_input_ids=batch["editor_input_ids"].to("cuda"),
                    base_input_ids=batch["base_input_ids"].to("cuda"),
                    base_attention_mask=batch["base_attention_mask"].to("cuda"),
                    base_intervention_mask=batch["base_intervention_mask"].to("cuda"),
                    labels=batch["labels"].to("cuda"),
                    base_intervention_weight=base_intervention_weight,
                )
                
                test_loss.append(predictions["loss"].item())
                """if isinstance(self.interpretor.das_module, QuasiProjectiveIntervention):
                    self.interpretor.zero_penalty()"""
                
                batch_pred_ids = torch.argmax(predictions["logits"], dim=-1)
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
                    
        return accuracies, sum(test_loss) / len(test_loss), correct_idxs
    
    
    def _entropy(self, x, mean=True):
                
        if mean:
            return -torch.sum(x * torch.log(x + 1e-12), dim=-1).mean()
        
        return -torch.sum(x * torch.log(x + 1e-12), dim=-1)
    

    def run_train(
        self,
        train_loader,
        test_loader=None,
        epochs=1,
        n_steps=1000,
        eval_per_steps: int = None,
        checkpoint_per_steps: int = None,
        causal_loss_weight=1.0,
        iso_loss_weight=1.0,
        lr=3e-4,
        weight_decay=0.01,
        save_dir=None,
        save_model=False,
        sparsity_loss=True,
        sparsity_loss_weight=1.0,
        debug_mode=None,
    ):
        
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        if debug_mode is not None:
            assert debug_mode in ["last_token", "last_entity_token"]
            
        trainable_parameters = []
        for name, param in self.named_parameters():
            
            if "target_model" not in name:
                if "das_module" in name:
                    if "rotate_layer" in name:
                        trainable_parameters += [{"params": param, "lr": self.rotate_lr, "weight_decay": 0.0}]
                    elif "mask_projection" in name:
                        trainable_parameters += [{"params": param, "lr": self.boundary_lr}]
                    else:
                        trainable_parameters += [{"params": param}]
                else:
                    trainable_parameters += [{"params": param}]
        
        self.opt = optim.AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)  # usually: lr = 5e-5. 1e-3 worked well!
        
        total_steps = len(train_loader) * epochs if n_steps is None else n_steps
        cur_steps = 0
            
        for epoch in range(epochs):
            # Create a tqdm progress bar
            with tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{epochs}",
                unit="batch",
                disable=True,
            ) as pbar:
                num_datapoints_in_epoch = 0
                epoch_train_loss = 0
                # Train loop
                for step, batch in enumerate(
                    train_loader
                ):  
                    
                    if cur_steps >= total_steps:
                        break
                    
                    if eval_per_steps is not None:
                        if cur_steps % eval_per_steps == 0:
                            # Evaluate the model
                            
                            accuracies, test_loss, _ = self.eval_accuracy(
                                test_loader, eval_n_label_tokens=3, debug_mode=debug_mode
                            )
                                
                            causal_acc = accuracies["causal"]
                            isolate_acc = accuracies["isolate"]
                            disentangle_acc = accuracies["disentangle"]
                                                        
                            if wandb.run:
                                wandb.log(
                                    {
                                        f"test_average_loss": test_loss,
                                        f"causal_accuracy": causal_acc,
                                        f"isolate_accuracy": isolate_acc,
                                        f"disentangle_accuracy": disentangle_acc,
                                    }
                                )
                                
                            print(f"Disentangle Acc: {disentangle_acc}, Causal Acc: {causal_acc}, Isolate Acc: {isolate_acc}, Test Loss: {test_loss}")
                        
                    if checkpoint_per_steps is not None:
                        if cur_steps % checkpoint_per_steps == 0 and save_dir is not None and save_model:
                            print("Saving model to {}".format(os.path.join(save_dir, f"model_epoch_{epoch}_step_{step}")))
                            self.save_model(os.path.join(save_dir, f"model_epoch_{epoch}_step_{step}"))
                            
                    self.batch = batch
                    current_batch_size = len(batch["editor_input_ids"])
                    num_datapoints_in_epoch += current_batch_size
                    
                    base_intervention_weight = None
                    
                    if debug_mode is not None:
                        if debug_mode == "last_token":
                                         
                            base_last_token_idxs = torch.sum(batch["labels"] == -100, dim=-1) - 1
                            base_intervention_weight = torch.functional.F.one_hot(base_last_token_idxs, num_classes=batch["base_intervention_mask"].shape[-1])
                                                
                        base_intervention_weight = base_intervention_weight.float().to("cuda")
                        
                    
                    prediction = self.forward(
                        editor_input_ids=batch["editor_input_ids"].to("cuda"),
                        base_input_ids=batch["base_input_ids"].to("cuda"),
                        base_attention_mask=batch["base_attention_mask"].to("cuda"),
                        base_intervention_mask=batch["base_intervention_mask"].to("cuda"),
                        labels=batch["labels"].to("cuda"),
                        is_causal=batch["is_causal"].to("cuda"),
                        causal_loss_weight=causal_loss_weight,
                        base_intervention_weight=base_intervention_weight,
                    )
                    
                    training_loss = 0
                    
                    prediction_loss = prediction["loss"]                            
                    training_loss += prediction_loss
                    
                    if sparsity_loss:
                        
                        base_intervention_weight = prediction["base_intervention_weight"]
                        
                        base_entropy = self._entropy(base_intervention_weight, mean=True)
                        
                        sparsity_loss = sparsity_loss_weight * base_entropy
                        training_loss += sparsity_loss
                        
                                            
                    """if isinstance(self.interpretor.das_module, QuasiProjectiveIntervention):
                        penalty = self.interpretor.get_penalty()
                        training_loss += penalty
                        self.interpretor.zero_penalty()"""

                    training_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.parameters(), 4.0
                    )
                    # print(prediction.logits)
                    # print(self.interpretor.hypernetwork.model.layers[1].self_attn.q_proj.weight.grad)
                    self.opt.step()
                    # metrics
                    epoch_train_loss += training_loss.item() * current_batch_size
                    self.opt.zero_grad()
                    
                    # TEST: orthogonalize the rotation matrix every step
                    """if self.use_das_intervention:
                        self.interpretor.das_module.orthogonalize_rotation_matrix()"""

                    metrics = {
                        "step": cur_steps,
                        "train_batch_prediction_loss": prediction_loss.item(),
                    }
                    
                    if sparsity_loss:
                        metrics["train_batch_base_sparsity_loss"] = base_entropy.item()
                        
                    """if isinstance(self.interpretor.das_module, QuasiProjectiveIntervention):
                        metrics["train_batch_penalty"] = penalty.item()"""

                    if wandb.run:
                        wandb.log(metrics)
                        
                    if torch.isnan(prediction_loss):
                        raise
                        
                    if cur_steps % 5 == 0:
                        print(metrics)

                    # Update progress bar
                    pbar.update(1)  # note: this was incorrectly displaying before!
                    cur_steps += 1
                    
                if wandb.run:
                    wandb.log(
                        {
                            "epoch_train_total_loss": epoch_train_loss
                            / num_datapoints_in_epoch,
                        }
                    )
                
                if cur_steps >= total_steps:
                    break
                    
        result_dict = {}
        accs, test_loss, correct_indices = self.eval_accuracy(test_loader, eval_n_label_tokens=3, debug_mode=debug_mode)
        result_dict = {
            "accs": accs,
            "test_loss": test_loss,
            "correct_indices": correct_indices,
        }
        
        for k, v in accs.items():
            print(f"{k}: {v}")
                    
        # Save the final model
        if save_dir is not None:
            if save_model:
                self.save_model(os.path.join(save_dir, "final_model"))
            json.dump(result_dict, open(os.path.join(save_dir, "final_result.json"), "w"))
                    
