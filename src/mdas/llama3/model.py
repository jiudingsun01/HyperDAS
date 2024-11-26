import json
import os
from deepspeed.profiling.flops_profiler import get_model_profile
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import LlamaForCausalLM

import wandb
from pyvene import (
    IntervenableConfig,
    IntervenableModel,
    LowRankRotatedSpaceIntervention,
    RepresentationConfig,
    count_parameters,
)


class RavelMDASNetwork(nn.Module):
    
    def __init__(
        self,
        model_name_or_path="/home/ubuntu/llama3-8b",
        intervention_layer=0,
        torch_dtype=torch.bfloat16,
        das_dimension=128,
        intervention_location="last_entity_token"
    ):
        super().__init__()
        self.intervention_location = intervention_location
        self.intervention_layer = intervention_layer
        self.das_dimension=das_dimension
        
        model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype).to("cuda")
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
        
        self.intervenable = IntervenableModel(intervention_config, model)
        self.intervenable.set_device(model.device)
        self.intervenable.disable_model_gradients()
        
    
    def forward(
        self,
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
        
        _, counterfactual_outputs = self.intervenable(
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
    
    def eval_accuracy(self, test_loader, eval_n_label_tokens=3):
        
        self.intervenable.eval()
        correct_idxs = []
        is_causal = []
        
        with torch.no_grad():
            for batch_id, batch in enumerate(test_loader):
                
                if self.intervention_location == "last_entity_token":
                    base_intervention_position = batch["base_entity_position_ids"].to("cuda") 
                    source_intervention_position = batch["source_entity_position_ids"].to("cuda")
                else:
                    base_intervention_position = batch["base_input_ids"].shape[1] - 1
                    source_intervention_position = batch["source_input_ids"].shape[1] - 1
                    
                    base_intervention_position = torch.tensor([base_intervention_position] * batch["base_input_ids"].shape[0]).to("cuda")
                    source_intervention_position = torch.tensor([source_intervention_position] * batch["source_input_ids"].shape[0]).to("cuda")
                
                output = self.forward(
                    base_input_ids=batch["base_input_ids"].to("cuda"),
                    base_attention_mask=batch["base_attention_mask"].to("cuda"),
                    base_intervention_position=base_intervention_position,
                    source_input_ids=batch["source_input_ids"].to("cuda"),
                    source_attention_mask=batch["source_attention_mask"].to("cuda"),
                    source_intervention_position=source_intervention_position,
                    intervention_layer=self.intervention_layer,
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
    
    def run_train(
        self,
        train_loader,
        test_loader=None,
        epochs=1,
        eval_per_steps: int = None,
        checkpoint_per_steps: int = None,
        causal_loss_weight=1.0,
        lr=3e-4,
        save_dir=None,
    ):
        
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
        inv_keys = list(self.intervenable.interventions.keys())[0]
    
        optimizer_params = []
        for k, v in self.intervenable.interventions.items():
            optimizer_params += [{'params': v[0].rotate_layer.parameters()}]
                
        optimizer = torch.optim.AdamW(optimizer_params,
                                        lr=lr,
                                        weight_decay=0)

        total_steps = len(train_loader) * epochs
        print("Model trainable parameters: ", count_parameters(self.intervenable.model))
        print("Intervention trainable parameters: ", self.intervenable.count_parameters())
        
        cur_steps = 0
        
        total_tflops = 0
        pure_train_time = 0
        profile_frequency = 100
        last_profiled_flops = None
        
        
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
                    if eval_per_steps is not None:
                        if cur_steps % eval_per_steps == 0:
                            accuracies = self.eval_accuracy(
                                test_loader, eval_n_label_tokens=3
                            )
                            
                            causal_acc = accuracies["causal"]
                            isolate_acc = accuracies["isolate"]
                            disentangle_acc = accuracies["disentangle"]
                                                    
                            if wandb.run:
                                wandb.log(
                                    {
                                        "causal_accuracy": causal_acc,
                                        "isolate_accuracy": isolate_acc,
                                        "disentangle_accuracy": disentangle_acc,
                                    }
                                )
                            
                            print(f"Disentangle Acc: {disentangle_acc}, Causal Acc: {causal_acc}, Isolate Acc: {isolate_acc}")
                        
                    if checkpoint_per_steps is not None:
                        if cur_steps % checkpoint_per_steps == 0 and save_dir is not None:
                            print("Saving model to {}".format(os.path.join(save_dir, f"model_epoch_{epoch}_step_{step}")))
                            torch.save(self.intervenable.interventions[inv_keys][0].state_dict(), os.path.join(save_dir, "das_epoch_{epoch}_step_{step}.pt"))
                    
                    current_batch_size = len(batch["labels"])
                    num_datapoints_in_epoch += current_batch_size
                    
                    training_loss = 0
                    
                    if self.intervention_location == "last_entity_token":
                        base_intervention_position = batch["base_entity_position_ids"].to("cuda") 
                        source_intervention_position = batch["source_entity_position_ids"].to("cuda")
                    else:
                        base_intervention_position = batch["base_input_ids"].shape[1] - 1
                        source_intervention_position = batch["source_input_ids"].shape[1] - 1
                        
                        base_intervention_position = torch.tensor([base_intervention_position] * batch["base_input_ids"].shape[0]).to("cuda")
                        source_intervention_position = torch.tensor([source_intervention_position] * batch["source_input_ids"].shape[0]).to("cuda")
                        
                    train_start = time.time()
                    
                    output = self.forward(
                        base_input_ids=batch["base_input_ids"].to("cuda"),
                        base_attention_mask=batch["base_attention_mask"].to("cuda"),
                        base_intervention_position=base_intervention_position,
                        source_input_ids=batch["source_input_ids"].to("cuda"),
                        source_attention_mask=batch["source_attention_mask"].to("cuda"),
                        source_intervention_position=source_intervention_position,
                        intervention_layer=self.intervention_layer,
                    )
                                    
                    logits = output.logits
                    labels = batch["labels"].to("cuda")
                    
                    log_prob_predictions = torch.nn.functional.log_softmax(
                        logits.reshape(-1, logits.shape[-1]),
                        dim=1,
                    )
                    
                    loss_weight = torch.ones_like(labels, dtype=log_prob_predictions.dtype)
                    loss_weight[batch["is_causal"].to("cuda"), :] = causal_loss_weight
                    
                    labels = labels.reshape(-1)
                    
                    loss_weight = loss_weight.reshape(-1)

                    assert labels.shape == log_prob_predictions.shape[:-1]
                    
                    # Only consider the tokens that are not -100 in target_labels
                    label_indices = labels != -100
                    output_idices = torch.zeros_like(label_indices)
                    output_idices[:-1] = label_indices[1:]
                    
                    log_prob_predictions = log_prob_predictions[output_idices, :]
                
                    labels = labels[label_indices]
                    
                    # Compute the cross-entropy loss with masking
                    
                    loss_weight = loss_weight[label_indices]
                    criterion = torch.nn.CrossEntropyLoss(reduction="none")
                    loss = criterion(log_prob_predictions, labels.long())
                    loss = (loss * loss_weight).mean()

                    prediction_loss = loss                          
                    training_loss += prediction_loss
                                        
                    training_loss.backward()
                                
                    # metrics
                    epoch_train_loss += training_loss.item() * current_batch_size
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    train_end = time.time()

                    
                    if cur_steps % profile_frequency == 0:
                        profile_inputs = (
                            batch["base_input_ids"].to("cuda"),
                            batch["base_attention_mask"].to("cuda"), 
                            base_intervention_position,
                            None,
                            batch["source_input_ids"].to("cuda"),
                            batch["source_attention_mask"].to("cuda"),
                            source_intervention_position,
                            None,
                            self.intervention_layer,
                        )
    
                        # Count FLOPs using deepspeed
                        with torch.no_grad():
                            try:
                                flops, _, _ = get_model_profile(
                                    model=self,
                                    args=profile_inputs,
                                    print_profile=False,
                                    detailed=False
                                )
                                # DeepSpeed returns flops as a string like "1.23 TFLOPS"
                                if isinstance(flops, str):
                                    step_tflops = float(flops.split()[0])
                                else:
                                    step_tflops = flops / 10**12  # Convert to TFLOPS
                            except Exception as e:
                                print(f"Warning: DeepSpeed profiling failed: {e}")
                                step_tflops = last_profiled_flops if last_profiled_flops else 0

                        last_profiled_flops = step_tflops
                    else:
                        step_tflops = last_profiled_flops
 
                    # Estimate backward pass FLOPs (typically 2-3x forward pass)
                    # Using 2x as a conservative estimate
                    backward_tflops = 2 * step_tflops
    
                    total_tflops += step_tflops + backward_tflops
                    pure_train_time += train_end - train_start
                    
                    # TEST: orthogonalize the rotation matrix every step
                    """if self.use_das_intervention:
                        self.interpretor.das_module.orthogonalize_rotation_matrix()"""

                    metrics = {
                        "step": cur_steps,
                        "train_batch_total_loss": training_loss.item(),
                        "train_batch_prediction_loss": prediction_loss.item(),
                        "overhead/train_batch_tflops": backward_tflops + step_tflops,
                        "overhead/train_batch_time": train_end - train_start,
                        "overhead/train_cumulative_tflops": total_tflops,
                        "overhead/train_cumulative_time": pure_train_time,
                    }

                    if wandb.run:
                        wandb.log(metrics)
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
        
        accuracies = self.eval_accuracy(
            test_loader, eval_n_label_tokens=3
        )
        for k, v in accuracies.items():
            print(f"{k}: {v}")
        
        if save_dir is not None:
            torch.save(self.intervenable.interventions[inv_keys][0].state_dict(), os.path.join(save_dir, "final_das_module.pt"))
            json.dump(accuracies, open(os.path.join(save_dir, "final_accuracies.json"), "w"))
