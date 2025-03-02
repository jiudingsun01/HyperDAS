import torch
from.modules import LlamaInterpretorConfig
from ..utils import InterpretorModelOutput
from transformers import AutoTokenizer
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import os
from tqdm import tqdm
import time
from typing import Any, List, Mapping, Optional, Tuple, TypeVar, Union
import json

from .layers import InterpretorUnembedCrossAttention, LlamaDecoderLayerWithDoubleCrossAttention
from ..das_utils import BoundlessRotatedSpaceIntervention, RotatedSpaceIntervention, LowRankRotatedSpaceIntervention, SelectiveLowRankRotatedSpaceIntervention,ReflectiveLowRankRotatedSpaceIntervention


from ..utils import (
    InterpretorModelOutput,
    add_fwd_hooks,
    assign_layer_indices,
)

from transformers import AutoModelForCausalLM

T = TypeVar("T", bound="LlamaAblatedInterpretor")



def generate_ravel_dictionary(dataset):
    
    all_keys = set()
    for d in dataset:
        all_keys.add(d["entity"] + "_" + d["attribute"])
    
    # map each key to an index
    return {key: i for i, key in enumerate(all_keys)}
    


def get_ravel_dictionary_collate_fn(
    tokenizer, 
    mapping_dict,
    contain_entity_position=False, 
    dataset_filtering=False,
    source_suffix_visibility=False,
    base_suffix_visibility=False,
    add_space_before_target=True,
):
    
    """
    Find the position of the entity text in the input_ids by comparing the entity text with the decoded input_ids.
    The entity text could contain multiple tokens.
    Return: a list of positions of the entity text in the input_ids.
    """
    def _find_entity_positions(decoded_input_tokens, entity_text, token_num=1):
        
        combined_decoded_input_ids = []
        
        if token_num > len(decoded_input_tokens):
            raise ValueError("Entity text not found in input_ids, which is weird!")
        
        for i in range(len(decoded_input_tokens) - token_num + 1):
            combined_token = "".join(decoded_input_tokens[i:i+token_num])
            combined_decoded_input_ids.append(combined_token)
            
        for i in range(len(combined_decoded_input_ids)):
            if entity_text in combined_decoded_input_ids[i]:
                return [i + j for j in range(token_num)]
            
        return _find_entity_positions(decoded_input_tokens, entity_text, token_num + 1)
    
    def tokenize_text_inputs(prefixes, suffixes, counterfactual_prefixes, counterfactual_suffixes, target_texts, entities=None, counterfactual_entities=None):
        
        if add_space_before_target:
            input_texts = []
            for prefix, suffix, target in zip(prefixes, suffixes, target_texts):
                if suffix.endswith(" ") or suffix.endswith("\"") or suffix.endswith("\'") or suffix.endswith("("):
                    input_texts.append(tokenizer.bos_token + prefix + suffix + target)
                else:
                    input_texts.append(tokenizer.bos_token + prefix + suffix + " " + target)
        else:
            input_texts = [tokenizer.bos_token + prefix + suffix + target for prefix, suffix, target in zip(prefixes, suffixes, target_texts)]
            
        counterfactual_texts = [tokenizer.bos_token + prefix + suffix for prefix, suffix in zip(counterfactual_prefixes, counterfactual_suffixes)]
        
        source_intervention_visibility_masks, base_intervention_visibility_masks = [], []
        
        if entities is not None and counterfactual_entities is not None:
            source_entity_position_ids = []
            base_entity_position_ids = []
        
        tokenized = tokenizer(input_texts, return_tensors="pt", padding=True, max_length=50, truncation=True)
        tokenized_counterfactual = tokenizer(counterfactual_texts, return_tensors="pt", padding=True, max_length=50, truncation=True)
        tokenized_labels = []
        
        for i, input_ids in enumerate(tokenized["input_ids"]):
            input_prompt = tokenizer.bos_token + prefixes[i] + suffixes[i]
            prompt_length = tokenizer(input_prompt, return_tensors="pt", padding=False)["input_ids"].shape[-1]
            if tokenizer.padding_side == "left":
                prompt_length += torch.sum(input_ids == tokenizer.pad_token_id)
            
            label = torch.full_like(input_ids, -100)
            label[prompt_length:] = input_ids[prompt_length:]
            label[input_ids == tokenizer.pad_token_id] = -100
            tokenized_labels.append(label)
            
            if entities is not None and counterfactual_entities is not None:
                entity_token = entities[i]
                counterfactual_entity_token = counterfactual_entities[i]
                
                base_entity_position_ids.append(_find_entity_positions([tokenizer.decode(ids) for ids in input_ids], entity_token)[-1])
                source_entity_position_ids.append(_find_entity_positions([tokenizer.decode(ids).strip() for ids in tokenized_counterfactual["input_ids"][i]], counterfactual_entity_token)[-1])
            
            source_visibility_mask = tokenized_counterfactual["attention_mask"][i].clone()
            base_visibility_mask = tokenized["attention_mask"][i].clone()
            
            label_length = torch.sum(label != -100)
            base_visibility_mask[-label_length:] = 0
            
            if not source_suffix_visibility:
                source_suffix_length = tokenizer(counterfactual_suffixes[i], return_tensors="pt", padding=False)["input_ids"].shape[-1]
                source_visibility_mask[-source_suffix_length:] = 0
                
            if not base_suffix_visibility:
                base_suffix_length = tokenizer(suffixes[i], return_tensors="pt", padding=False)["input_ids"].shape[-1]
                base_visibility_mask[prompt_length - base_suffix_length:] = 0
                
            source_intervention_visibility_masks.append(source_visibility_mask)
            base_intervention_visibility_masks.append(base_visibility_mask)
                
        base_intervention_mask = torch.stack(base_intervention_visibility_masks)
        source_intervention_mask = torch.stack(source_intervention_visibility_masks)
        tokenized_labels = torch.stack(tokenized_labels)
        
        return_dict = {
            "base_input_ids": tokenized["input_ids"],
            "base_attention_mask": tokenized["attention_mask"],
            "base_intervention_mask": base_intervention_mask,
            "source_input_ids": tokenized_counterfactual["input_ids"],
            "source_attention_mask": tokenized_counterfactual["attention_mask"],
            "source_intervention_mask": source_intervention_mask,
            "labels": tokenized_labels
        }
        
        if entities is not None and counterfactual_entities is not None:
            return_dict["source_entity_position_ids"] = torch.tensor(source_entity_position_ids)
            return_dict["base_entity_position_ids"] = torch.tensor(base_entity_position_ids)
        
        return return_dict
    
    def collate_fn(batch):
        
        prefixes, suffixes, edit_instructions, targets, counterfactual_prefixes, counterfactual_suffixes, vector_keys = [], [], [], [], [], [], []
        
        if contain_entity_position:
            assert "entity" in batch[0].keys() and "counterfactual_entity" in batch[0].keys()
            entities, counterfactual_entities = [], []
        else:
            entities, counterfactual_entities = None, None
            
        for b in batch:
            
            vector_keys.append(b["entity"] + "_" + b["attribute"])
            prefixes.append(b["input_prefix"])
            suffixes.append(b["input_suffix"])
            edit_instructions.append(tokenizer.bos_token + b["edit_instruction"])
            counterfactual_prefixes.append(b["counterfactual_input_prefix"])
            counterfactual_suffixes.append(b["counterfactual_input_suffix"])
            
            targets.append(b["counterfactual_target"] if b["attribute_type"] == "causal" else b["target"])
            
            if contain_entity_position:
                entities.append(b["entity"])
                counterfactual_entities.append(b["counterfactual_entity"])
                
        vector_keys = [mapping_dict[key] for key in vector_keys]
        vector_keys = torch.tensor(vector_keys)
            
        editor_input_ids = tokenizer(edit_instructions, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        is_causal = torch.tensor([b["attribute_type"] == "causal" for b in batch])
        returned_dict = {
            "editor_input_ids": editor_input_ids,
            "is_causal": is_causal,
            "vector_ids": vector_keys,
            **tokenize_text_inputs(prefixes, suffixes, counterfactual_prefixes, counterfactual_suffixes, targets, entities=entities, counterfactual_entities=counterfactual_entities),
        }
        
        return returned_dict
    
    def filtering_collate_fn(batch):
        inputs, targets = [], []
        
        for b in batch:
            inputs.append(b["verify_text"])
            targets.append(b["counterfactual_target"] if b["attribute_type"] == "causal" else b["target"])
            
        if add_space_before_target:
            input_texts = []
            for input_text, target in zip(inputs, targets):
                if input_text.endswith(" ") or input_text.endswith("\"") or input_text.endswith("\'") or input_text.endswith("("):
                    input_texts.append(tokenizer.bos_token + input_text + target)
                else:
                    input_texts.append(tokenizer.bos_token + input_text + " " + target)
        else:
            input_texts = [tokenizer.bos_token + input_text + target for input_text, target in zip(inputs, targets)]
        
        
        tokenized = tokenizer(input_texts, return_tensors="pt", padding=True, max_length=50, truncation=True)
        tokenized_labels = []
        
        for i, input_ids in enumerate(tokenized["input_ids"]):
            input_prompt = tokenizer.bos_token + inputs[i]
            prompt_length = tokenizer(input_prompt, return_tensors="pt", padding=False)["input_ids"].shape[-1]
            if tokenizer.padding_side == "left":
                prompt_length += torch.sum(input_ids == tokenizer.pad_token_id)
            
            label = torch.full_like(input_ids, -100)
            label[prompt_length:] = input_ids[prompt_length:]
            label[input_ids == tokenizer.pad_token_id] = -100
            tokenized_labels.append(label)
            
        tokenized_labels = torch.stack(tokenized_labels)
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized_labels
        }
        
    
    return collate_fn if not dataset_filtering else filtering_collate_fn



class RavelAblatedInterpretorHypernetwork(nn.Module):
    # Separating the editor config file, from its base model's configurations
    def __init__(
        self,
        model_name_or_path="/home/ubuntu/llama3-8b",
        num_concept=None,
        num_editing_heads=32,
        intervention_layer=12,
        subspace_module="ReflectSelect",
        torch_dtype=torch.bfloat16,
        das_dimension=None,
    ):
        super().__init__()

        self.interpretor_config = LlamaInterpretorConfig.from_pretrained(model_name_or_path)
        self.interpretor_config.name_or_path = model_name_or_path
        self.interpretor_config.torch_dtype = torch_dtype
        self.interpretor_config.num_editing_heads = num_editing_heads
        self.interpretor_config.intervention_layer = intervention_layer
        self.interpretor_config._attn_implementation = 'eager'
                
        self.interpretor = LlamaAblatedInterpretor(
            self.interpretor_config, 
            subspace_module=subspace_module, 
            das_dimension=das_dimension,
            num_concept=num_concept,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

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
        self.source_sparsity_loss_weight= 0.25
        
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
        
    def forward(
        self,
        editor_input_ids: torch.Tensor = None,
        base_input_ids: torch.Tensor = None,
        base_attention_mask: torch.Tensor = None,
        base_intervention_mask: torch.Tensor = None,
        source_input_ids: torch.Tensor = None,
        source_attention_mask: torch.Tensor = None,
        source_intervention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        output_intervention_weight: bool = True,
        is_causal: torch.Tensor = None,
        causal_loss_weight: float = 1.0,
        intervention_weight: torch.Tensor = None,
        vector_ids: torch.Tensor = None,
        inference_mode = None,
    ):
        _pred: InterpretorModelOutput = self.interpretor(
            editor_input_ids=editor_input_ids,
            editor_attention_mask=editor_input_ids != self.interpretor_config.eos_token_id,
            base_input_ids=base_input_ids,
            base_attention_mask=base_attention_mask,
            base_intervention_mask=base_intervention_mask,
            source_input_ids=source_input_ids,
            source_attention_mask=source_attention_mask,
            source_intervention_mask=source_intervention_mask,
            output_intervention_weight=output_intervention_weight,
            intervention_weight=intervention_weight,
            vector_ids=vector_ids,
            inference_mode=inference_mode
        )
        
        if labels is not None:
            log_prob_predictions = torch.nn.functional.log_softmax(
                _pred.logits.reshape(-1, _pred.logits.shape[-1]),
                dim=1,
            )
            
            if is_causal is not None:
                loss_weight = torch.ones_like(labels, dtype=log_prob_predictions.dtype)
                loss_weight[is_causal, :] = causal_loss_weight
            
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
    def inspect_batch_prediction_ouptuts(self, batch, inference_mode=None, eval_n_label_tokens=None):
        assert inference_mode in [None, "column_argmax", "global_argmax", "groundtruth", "bidding_argmax"]
        self.interpretor.eval()
        
        correct_idxs = []
        
        if inference_mode == "groundtruth":
            intervention_weight = torch.zeros(len(batch["editor_input_ids"]), batch["source_input_ids"].shape[1] + 1, batch["base_input_ids"].shape[1]).to("cuda")
            intervention_weight[:, -1, :] = 1.0
            
            for i in range(len(batch["base_entity_position_ids"])):
                intervention_weight[i, -1, batch["base_entity_position_ids"][i]] = 0.0
                intervention_weight[i, batch["source_entity_position_ids"][i], batch["base_entity_position_ids"][i]] = 1.0
            
        else:
            intervention_weight=None
        
        with torch.no_grad():
            
            predictions = self.forward(
                editor_input_ids=batch["editor_input_ids"].to("cuda"),
                base_input_ids=batch["base_input_ids"].to("cuda"),
                base_attention_mask=batch["base_attention_mask"].to("cuda"),
                base_intervention_mask=batch["base_intervention_mask"].to("cuda"),
                source_input_ids=batch["source_input_ids"].to("cuda"),
                source_attention_mask=batch["source_attention_mask"].to("cuda"),
                source_intervention_mask=batch["source_intervention_mask"].to("cuda"),
                labels=batch["labels"].to("cuda"),
                output_intervention_weight=True,
                inference_mode=inference_mode,
                intervention_weight=intervention_weight
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
                
        return_dict = {
            "batch_output": batch_output,
            "batch_full_output": batch_full_output,
            "batch_intervention_weight": predictions.intervention_weight,
            "n_correct": correct,
            "correct_idxs": correct_idxs
        }   
        return return_dict
        
    
    def plot_heatmap(
        self, 
        data_loader, 
        idxs, 
        batch_size=4, 
        inference_mode=None, 
        annot=True,
        indicate_masked_tokens=False
    ):
        batch_id = idxs // batch_size
        example_id = idxs % batch_size

        for i, batch in enumerate(data_loader):
            if i == batch_id:
                break
            
        results = self.inspect_batch_prediction_ouptuts(batch, inference_mode=inference_mode, eval_n_label_tokens=3)

        editor_input_ids = batch["editor_input_ids"][example_id]
        base_input_ids = batch["base_input_ids"][example_id]
        source_input_ids = batch["source_input_ids"][example_id]
        intervention_weight = results["batch_intervention_weight"][example_id]
        label = batch["labels"][example_id]

        assert intervention_weight.size() == (len(source_input_ids) + 1, len(base_input_ids))

        source_axis = [self.tokenizer.decode([i]) for i in source_input_ids] + ["[SELF]"]
        base_axis = [self.tokenizer.decode([i]) for i in base_input_ids]
        editor_text = self.tokenizer.decode(editor_input_ids, skip_special_tokens=True)
        label = label[label != -100]
        label = self.tokenizer.decode(label)

        _, ax = plt.subplots(figsize=(15, 15))
       
        plot = sns.heatmap(intervention_weight.float().cpu().numpy(), xticklabels=base_axis, yticklabels=source_axis, ax=ax, annot=annot, fmt=".2f")

        ax.set_title(f"Instruction: {editor_text}     Label: {label}    Pred: {results['batch_output'][example_id]}")
        ax.set_xlabel("Base Sentence Tokens")
        ax.set_ylabel("Source Sentence Tokens")
        
        return plot, ax
        
        
    def eval_accuracy(self, test_loader, inference_mode=None, eval_n_label_tokens=None):
        assert inference_mode in [None, "column_argmax", "global_argmax", "groundtruth", "bidding_argmax"]
        
        self.interpretor.eval()
        test_loss = []
        correct_idxs = []
        is_causal = []
        
        with torch.no_grad():
            for batch_id, batch in enumerate(test_loader):
                
                if inference_mode == "groundtruth":
                    intervention_weight = torch.zeros(len(batch["editor_input_ids"]), batch["source_input_ids"].shape[1] + 1, batch["base_input_ids"].shape[1]).to("cuda")
                    intervention_weight[:, -1, :] = 1.0
                    
                    for i in range(len(batch["base_entity_position_ids"])):
                        intervention_weight[i, -1, batch["base_entity_position_ids"][i]] = 0.0
                        intervention_weight[i, batch["source_entity_position_ids"][i], batch["base_entity_position_ids"][i]] = 1.0
                else:
                    intervention_weight=None
                                        
                predictions = self.forward(
                    editor_input_ids=batch["editor_input_ids"].to("cuda"),
                    base_input_ids=batch["base_input_ids"].to("cuda"),
                    base_attention_mask=batch["base_attention_mask"].to("cuda"),
                    base_intervention_mask=batch["base_intervention_mask"].to("cuda"),
                    source_input_ids=batch["source_input_ids"].to("cuda"),
                    source_attention_mask=batch["source_attention_mask"].to("cuda"),
                    source_intervention_mask=batch["source_intervention_mask"].to("cuda"),
                    labels=batch["labels"].to("cuda"),
                    inference_mode=inference_mode,
                    intervention_weight=intervention_weight,
                    vector_ids=batch["vector_ids"].to("cuda")
                )
                test_loss.append(predictions["loss"].item())
                
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
             

    def run_train(
        self,
        train_loader,
        test_loader=None,
        inference_modes=[None],
        epochs=1,
        test_per_steps: int = None,
        checkpoint_per_steps: int = None,
        apply_source_selection_sparsity_loss=False,
        causal_loss_weight=1.0,
        lr=3e-4,
        weight_decay=0.01,
        save_dir=None,
    ):
        
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
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
        
        total_steps = len(train_loader) * epochs
        cur_steps = 0
        
        if self.use_das_intervention:
            das_temperature_schedule = torch.linspace(
                self.das_temperature_start, self.das_temperature_end, total_steps + 1
            ).to(self.interpretor_config.torch_dtype).to("cuda")
            self.interpretor.das_module.set_temperature(das_temperature_schedule[cur_steps])

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
                epoch_gradient_norm = 0
                # Train loop
                for step, batch in enumerate(
                    train_loader
                ):  
                    if test_per_steps is not None:
                        if cur_steps % test_per_steps == 0:
                            # Evaluate the model
                            
                            for mode in inference_modes:
                                accuracies, test_loss, _ = self.eval_accuracy(
                                    test_loader, inference_mode=mode, eval_n_label_tokens=3
                                )
                                
                                text_mode = "default" if mode is None else mode
                                
                                causal_acc = accuracies["causal"]
                                isolate_acc = accuracies["isolate"]
                                disentangle_acc = accuracies["disentangle"]
                                                        
                                if wandb.run:
                                    wandb.log(
                                        {
                                            f"{text_mode}_test_average_loss": test_loss,
                                            f"{text_mode}_causal_accuracy": causal_acc,
                                            f"{text_mode}_isolate_accuracy": isolate_acc,
                                            f"{text_mode}_disentangle_accuracy": disentangle_acc,
                                        }
                                    )
                                
                                print("Under Inference Mode: ", text_mode)
                                print(f"Disentangle Acc: {disentangle_acc}, Causal Acc: {causal_acc}, Isolate Acc: {isolate_acc}, Test Loss: {test_loss}")
                        
                    if checkpoint_per_steps is not None:
                        if cur_steps % checkpoint_per_steps == 0 and save_dir is not None:
                            print("Saving model to {}".format(os.path.join(save_dir, f"model_epoch_{epoch}_step_{step}")))
                            self.save_model(os.path.join(save_dir, f"model_epoch_{epoch}_step_{step}"))
                            
                    self.batch = batch
                    current_batch_size = len(batch["editor_input_ids"])
                    num_datapoints_in_epoch += current_batch_size
                    
                    prediction = self.forward(
                        editor_input_ids=batch["editor_input_ids"].to("cuda"),
                        base_input_ids=batch["base_input_ids"].to("cuda"),
                        base_attention_mask=batch["base_attention_mask"].to("cuda"),
                        base_intervention_mask=batch["base_intervention_mask"].to("cuda"),
                        source_input_ids=batch["source_input_ids"].to("cuda"),
                        source_attention_mask=batch["source_attention_mask"].to("cuda"),
                        source_intervention_mask=batch["source_intervention_mask"].to("cuda"),
                        labels=batch["labels"].to("cuda"),
                        is_causal=batch["is_causal"].to("cuda"),
                        causal_loss_weight=causal_loss_weight,
                        output_intervention_weight=True,
                        vector_ids=batch["vector_ids"].to("cuda"),
                        inference_mode=None
                    )
                    
                    training_loss = 0
                    
                    prediction_loss = prediction["loss"]                            
                    training_loss += prediction_loss
                                        
                    if apply_source_selection_sparsity_loss:
                        
                        source_selection_sum = prediction.intervention_weight[:, :-1, :].sum(dim=-1)
                        source_selection_loss = torch.where(
                            source_selection_sum > 1.0,
                            source_selection_sum,
                            torch.zeros_like(source_selection_sum)
                        ).sum(dim=-1)
                        batch_source_selection_loss = source_selection_loss.mean()
                        
                        training_loss += self.source_sparsity_loss_weight * batch_source_selection_loss
                    
                    training_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.parameters(), 4.0
                    )
                    
                    before = self.interpretor.hypernetwork.weight[batch["vector_ids"].to("cuda")].clone()
                    
                    self.opt.step()
                    
                    after = self.interpretor.hypernetwork.weight[batch["vector_ids"].to("cuda")].clone()
                    
                    # check if the weight has changed
                    if torch.allclose(before, after):
                        print("Weight has not changed")
                        
                    # metrics
                    epoch_train_loss += training_loss.item() * current_batch_size
                    self.opt.zero_grad()
                    

                    metrics = {
                        "step": cur_steps,
                        "train_batch_total_loss": training_loss.item(),
                        "train_batch_prediction_loss": prediction_loss.item(),
                    }
                    
                    if self.use_das_intervention:
                        metrics["das_sparsity"] = self.interpretor.das_module.get_boundary_sparsity().item()

                    if wandb.run:
                        wandb.log(metrics)
                    if cur_steps % 5 == 0:
                        print(metrics)

                    
                    # Update progress bar
                    pbar.update(1)  # note: this was incorrectly displaying before!
                    cur_steps += 1
                    if self.use_das_intervention:
                        self.interpretor.das_module.set_temperature(das_temperature_schedule[cur_steps])
                    
                if wandb.run:
                    wandb.log(
                        {
                            "epoch_train_total_loss": epoch_train_loss
                            / num_datapoints_in_epoch,
                        }
                    )
        
        result_dict = {}
        for inference_mode in inference_modes:
            accs, test_loss, correct_indices = self.eval_accuracy(test_loader, inference_mode=inference_mode, eval_n_label_tokens=3)
            if inference_mode is None:
                inference_mode = "default"
            result_dict[inference_mode] = {
                "accs": accs,
                "test_loss": test_loss,
                "correct_indices": correct_indices,
            }
            
            for k, v in accs.items():
                print(f"{inference_mode} {k}: {v}")
                    
        # Save the final model
        if save_dir is not None:
            self.save_model(os.path.join(save_dir, "final_model"))
            json.dump(result_dict, open(os.path.join(save_dir, "final_result.json"), "w"))





class LlamaAblatedInterpretor(nn.Module):
    def __init__(self, config: LlamaInterpretorConfig, num_concept, subspace_module=None, das_dimension=None):
        super().__init__()

        self.config = config
        assert num_concept is not None
        self.target_model = AutoModelForCausalLM.from_pretrained(
            config.name_or_path, torch_dtype = config.torch_dtype
        )
        
        self.hypernetwork = nn.Embedding(num_concept, self.target_model.config.hidden_size, dtype=config.torch_dtype)
        self.unembedding_head = InterpretorUnembedCrossAttention(config, layer_idx=1).to(dtype=config.torch_dtype)
        
        self.bidding_threshold = 0.1
        
        self.use_das_intervention = subspace_module != None
        self.das_selective_subspace = subspace_module in ["ReflectSelect", "MaskSelect"]
                
        if self.use_das_intervention:
            
            if subspace_module == "BoundlessDAS":
                self.das_module = BoundlessRotatedSpaceIntervention(
                    embed_dim=self.target_model.config.hidden_size, torch_dtype=config.torch_dtype
                )
            elif subspace_module == "DAS":
                self.das_module = LowRankRotatedSpaceIntervention(
                    embed_dim=self.target_model.config.hidden_size, low_rank_dimension=das_dimension, torch_dtype=config.torch_dtype
                )
            elif subspace_module == "MaskSelect":
                self.das_module = SelectiveLowRankRotatedSpaceIntervention(
                    embed_dim=self.target_model.config.hidden_size, low_rank_dimension=das_dimension, torch_dtype=config.torch_dtype
                )
            elif subspace_module == "ReflectSelect":
                self.das_module = ReflectiveLowRankRotatedSpaceIntervention(
                    embed_dim=self.target_model.config.hidden_size, low_rank_dimension=das_dimension, torch_dtype=config.torch_dtype
                )
            else:
                raise ValueError("Invalid subspace module")
                    
        # freeze target model
        for param in self.target_model.parameters():
            param.requires_grad = False

        assign_layer_indices(self.target_model)

        """if config.use_layerwise_embeddings:
            # extra layer is cross-attn in the lm_head
            self.layerwise_embeddings = nn.Parameter(
                torch.zeros(config.n_layer + 1, config.n_embd), requires_grad=True
            )
            self.layerwise_embeddings.data.normal_(
                mean=0.0, std=self.target_model.config.initializer_range
            )
        else:"""
        
        self.layerwise_embeddings = None

    def train(self: T, mode: bool = True) -> T:
        return self.hypernetwork.train(mode)

    def eval(self: T) -> T:
        return self.hypernetwork.eval()

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        """Only load weights for the trainable hypernetwork."""
        self.hypernetwork.load_state_dict(state_dict, strict=strict, assign=assign)

    @torch.no_grad()
    def _run_target_model_for_encoded_hidden_states(
        self,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        position_ids: torch.Tensor = None,
    ):
        """Gets the hidden states from the target model, if necessary"""

        if position_ids is not None:
            outputs = self.target_model(
                input_ids=target_input_ids,
                attention_mask=target_attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            )

        else:
            outputs = self.target_model(
                input_ids=target_input_ids,
                attention_mask=target_attention_mask,
                output_hidden_states=True,
            )

        return outputs.hidden_states

    def forward(
        self,
        editor_input_ids: torch.Tensor = None,
        editor_attention_mask: torch.Tensor = None,
        base_input_ids: torch.Tensor = None,
        base_attention_mask: torch.Tensor = None,
        base_intervention_mask: torch.Tensor = None,
        source_input_ids: torch.Tensor = None,
        source_attention_mask: torch.Tensor = None,
        source_intervention_mask: torch.Tensor = None,
        base_hidden_states: torch.Tensor = None,
        base_position_ids: torch.Tensor = None,
        source_hidden_states: torch.Tensor = None,
        source_position_ids: torch.Tensor = None,
        intervention_layer: int = None,
        output_vanilla_hidden_states: bool = True,
        output_edited_hidden_states: bool = False,
        output_intervention_weight: bool = True,
        intervention_weight: torch.Tensor = None,
        inference_mode: str = None,
        vector_ids: torch.Tensor = None,
    ) -> InterpretorModelOutput:
        
        assert inference_mode in [None, "column_argmax", "global_argmax", "groundtruth", "bidding_argmax"]
        
                
        if intervention_layer is None:
            intervention_layer = self.config.intervention_layer     
            
        if base_position_ids is None:
            # 0 for all the padding tokens and start from 1 for the rest
            base_position_ids = torch.cumsum(base_attention_mask, dim=1) * base_attention_mask
        
        if source_position_ids is None:
            source_position_ids = torch.cumsum(source_attention_mask, dim=1) * source_attention_mask
        
        # Run target model for encoded hidden states
        if base_hidden_states is None:
            base_hidden_states = torch.stack(
                self._run_target_model_for_encoded_hidden_states(
                    base_input_ids, base_attention_mask, base_position_ids
                ),  # seems to break while we are passing thru batch_size=1; the last (12th =) has different dimensions
                dim=2,
            )
        
        if source_hidden_states is None:
            source_hidden_states = torch.stack(
                self._run_target_model_for_encoded_hidden_states(
                    source_input_ids, source_attention_mask, source_position_ids
                ),
                dim=2,
            )
            
        if base_intervention_mask is None:
            if base_attention_mask is not None:
                base_intervention_mask = base_attention_mask.clone()
            else:
                base_intervention_mask = torch.ones_like(base_input_ids)
                
        if source_intervention_mask is None:
            if source_attention_mask is not None:
                source_intervention_mask = source_attention_mask.clone()
            else:
                source_intervention_mask = torch.ones_like(source_input_ids)
            
        # dimensions of target_hidden_states:
        # batch_size, token_sequence_length, num_layers = 13, resid_width = 768
        # Normalize along the last dimension
        base_normalization_factors = base_hidden_states.norm(dim=-1, keepdim=True)
        base_hidden_states = base_hidden_states / base_normalization_factors
        
        source_normalization_factors = source_hidden_states.norm(dim=-1, keepdim=True)
        source_hidden_states = source_hidden_states / source_normalization_factors

        if intervention_weight is None or inference_mode == "groundtruth":
            
            n_layer = base_hidden_states.shape[2]
                        
            collapsed_base_hidden_states = base_hidden_states.reshape(
                base_hidden_states.shape[0],
                base_hidden_states.shape[1] * base_hidden_states.shape[2],
                base_hidden_states.shape[3],
            )
            
            collapsed_base_attention_mask = base_intervention_mask.unsqueeze(-1).repeat(1, 1, n_layer)
            collapsed_base_attention_mask = collapsed_base_attention_mask.reshape(
                base_intervention_mask.shape[0],
                base_intervention_mask.shape[1] * n_layer,
            )
            
            collapsed_source_hidden_states = source_hidden_states.reshape(
                source_hidden_states.shape[0],
                source_hidden_states.shape[1] * source_hidden_states.shape[2],
                source_hidden_states.shape[3],
            )
            
            collapsed_source_attention_mask = source_intervention_mask.unsqueeze(-1).repeat(1, 1, n_layer)
            collapsed_source_attention_mask = collapsed_source_attention_mask.reshape(
                source_intervention_mask.shape[0],
                source_intervention_mask.shape[1] * n_layer,
            )
            
            hypernet_hidden_states = self.hypernetwork(vector_ids)
            hypernet_hidden_states = hypernet_hidden_states.unsqueeze(1)

            if inference_mode == "groundtruth":
                intervention_weight = intervention_weight.to(dtype=hypernet_hidden_states.dtype)
            else:
                intervention_weight = self.unembedding_head(
                    hypernet_hidden_states,
                    attention_mask=editor_attention_mask,
                    base_encoder_hidden_states=collapsed_base_hidden_states,
                    base_encoder_attention_mask=collapsed_base_attention_mask,
                    source_encoder_hidden_states=collapsed_source_hidden_states,
                    source_encoder_attention_mask=collapsed_source_attention_mask,
                    output_attentions=True,
                )
                intervention_weight = intervention_weight.squeeze()
                        
        if inference_mode == "global_argmax":
            batch_size, _, num_base_pos = intervention_weight.shape
            source_base_intervention_flatten = intervention_weight[:, :-1, :].view(batch_size, -1)
            max_intervention_position = torch.argmax(source_base_intervention_flatten, dim=1)
            intervention_weight = torch.zeros_like(intervention_weight)
            intervention_weight[:, -1, :] = 1.0
            for i in range(batch_size):
                source_token_idx = max_intervention_position[i] // num_base_pos
                base_token_idx = max_intervention_position[i] % num_base_pos
                intervention_weight[i, source_token_idx, base_token_idx] = 1.0
                intervention_weight[i, -1, base_token_idx] = 0.0
        elif inference_mode == "column_argmax":
            batch_size, num_src_pos, num_base_pos = intervention_weight.shape
            intervention_weight = torch.argmax(intervention_weight, dim=1)
            intervention_weight = torch.nn.functional.one_hot(intervention_weight, num_classes=num_src_pos).to(dtype=intervention_weight.dtype).permute(0, 2, 1)            
        elif inference_mode == "bidding_argmax":
            batch_size, num_src_pos, num_base_pos = intervention_weight.shape
            bidding_weight = torch.argmax(intervention_weight[:, :-1, :], dim=-1)
            bidding_weight = torch.nn.functional.one_hot(bidding_weight, num_classes=num_base_pos).float()
            bidding_weight = torch.cat([bidding_weight, torch.ones(batch_size, 1, num_base_pos).to(bidding_weight.device)], dim=1)
            intervention_weight = torch.where(bidding_weight == 1, intervention_weight, torch.zeros_like(intervention_weight))
            if self.bidding_threshold is not None:
                threshold = torch.Tensor([self.bidding_threshold]).to(intervention_weight.device)
                threshold = threshold.repeat(batch_size, num_base_pos)
                intervention_weight[:, -1, :] = torch.where(intervention_weight[:, -1, :] > self.bidding_threshold, intervention_weight[:, -1, :], threshold)
            intervention_weight = torch.argmax(intervention_weight, dim=1)
            intervention_weight = torch.nn.functional.one_hot(intervention_weight, num_classes=num_src_pos).to(dtype=intervention_weight.dtype).permute(0, 2, 1)
    

        if len(intervention_weight.shape) == 2:
            intervention_weight = intervention_weight.unsqueeze(0) # Unsqueeze first dim if batch size = 1
                
        source_output = self.target_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            output_hidden_states=True,
        )
        
        source_hidden_states = source_output.hidden_states[intervention_layer]
        
        intervention_matrix = torch.einsum("bij,bid->bijd", intervention_weight[:, :-1, :], source_hidden_states) # TODO: Fix it to help the new implement 
        intervention_matrix = intervention_matrix.sum(dim=1) 
        
        # Run target model with edit vectors.
        # This adds the edit vectors to the given hidden state at the specified batch index, position, and layer
        def representation_swap(module, input, output):
            base_hidden_states = output[0].clone()
            batch_size = base_hidden_states.shape[0]
            base_intervention_weight = intervention_weight[:, -1, :]
            
            if self.use_das_intervention:
                
                source_intervention_hidden_states = intervention_matrix + torch.einsum("bid,bi->bid", base_hidden_states, base_intervention_weight)
                
                if self.das_selective_subspace:
                    mixed_output = self.das_module(base_hidden_states, source_intervention_hidden_states, hypernet_hidden_states)
                else:
                    mixed_output = self.das_module(base_hidden_states, source_intervention_hidden_states, batch_size)
                
                output[0][:] += (mixed_output - base_hidden_states)
            else:
                res_diff = torch.einsum("bid,bi->bid", base_hidden_states, (1 - base_intervention_weight))
                output[0][:] += (intervention_matrix - res_diff)
            
        def embedding_representation_swap(module, input, output):
            if self.use_das_intervention:
                raise NotImplementedError("DAS intervention is not supported for token embeddings")
            
            base_hidden_states = output.clone()
            base_intervention_weight = intervention_weight[:, -1, :]
            res_diff = torch.einsum("bid,bi->bid", base_hidden_states, (1 - base_intervention_weight))
            output += (intervention_matrix - res_diff)    
        
        # Now editing the target model
        if intervention_layer == 0:
            hooks = [(self.target_model.model.embed_tokens, embedding_representation_swap)]
        else:
            hooks = [(self.target_model.model.layers[intervention_layer - 1], representation_swap)]

        with add_fwd_hooks(hooks):
            # THIS IS THE LINE WHERE THE MODEL IS CALLED (AND THE EDITOR IS CALLED AT
            # THE END OF `layer` AS A SIDE EFFECT)
            target_result = self.target_model(
                input_ids=base_input_ids,
                attention_mask=base_attention_mask,
                position_ids=base_position_ids,
                output_hidden_states=output_edited_hidden_states,
            )
    
        logits = target_result.logits
        
        output = InterpretorModelOutput(logits=logits)
        if output_edited_hidden_states:
            output.edited_hidden_states = target_result.hidden_states
            
        if output_intervention_weight:
            output.intervention_weight = intervention_weight
            
        if output_vanilla_hidden_states:
            output.vanilla_base_hidden_states = base_hidden_states
            output.vanilla_source_hidden_states = source_hidden_states

        return output
    
    
