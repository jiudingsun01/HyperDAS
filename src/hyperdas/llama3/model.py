import json
import os
from abc import ABC, abstractmethod
from math import ceil
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers.modeling_outputs import ModelOutput

import wandb
from logger import get_logger
from src.hyperdas.data.axbench import parse_positions_varlen

from ..utils import InterpretorModelOutput, NamedDataLoader, init_on_device
from .modules import (
    LlamaInterpretor,
    LlamaInterpretorConfig,
    LlamaInterpretorForReFTGeneration,
    LlamaInterpretorWithLearnedSource,
)

logger = get_logger(__name__)

INTERPRETOR_CLS = {
    "regular": LlamaInterpretor,
    "learned_source": LlamaInterpretorWithLearnedSource,
    "reft": LlamaInterpretorForReFTGeneration,
}


class BaseInterpretorHypernetwork(nn.Module, ABC):
    def __init__(self, config: DictConfig, device):
        super().__init__()
        self.config = config
        self.device = device

        self.interpretor_config: LlamaInterpretorConfig = (
            LlamaInterpretorConfig.from_pretrained(config.model.name_or_path)
        )

        # Basic model config
        self.interpretor_config.name_or_path = config.model.name_or_path
        self.interpretor_config.torch_dtype = torch.bfloat16
        self.interpretor_config.intervention_layer = config.model.intervention_layer
        self.interpretor_config._attn_implementation = "eager"

        # Model architecture config
        self.interpretor_config.num_editing_heads = config.model.num_editing_heads
        self.interpretor_config.chop_editor_at_layer = config.model.num_decoders
        self.interpretor_config.initialize_from_scratch = (
            config.model.initialize_from_scratch
        )

        # Tokenizer setup
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.target_model_tokenizer = AutoTokenizer.from_pretrained(
            config.model.target_model_name_or_path
        )
        self.target_model_tokenizer.padding_side = "left"
        self.target_model_tokenizer.pad_token = self.target_model_tokenizer.eos_token
        self.target_model_tokenizer.pad_token_id = (
            self.target_model_tokenizer.eos_token_id
        )

        self.residual_cache = None
        self.opt = None
        self.max_eval_steps = config.training.max_eval_steps

        # Dummy initialization for typing
        self.interpretor: LlamaInterpretor = None

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(
            self.interpretor.hypernetwork.state_dict(),
            os.path.join(save_dir, "hypernetwork.pt"),
        )
        torch.save(
            self.interpretor.intervention_module.state_dict(),
            os.path.join(save_dir, "intervenable.pt"),
        )

    def load_model(self, load_dir):
        self.interpretor.hypernetwork.load_state_dict(
            torch.load(os.path.join(load_dir, "hypernetwork.pt"), weights_only=True)
        )
        self.interpretor.intervention_module.load_state_dict(
            torch.load(os.path.join(load_dir, "intervenable.pt"), weights_only=True)
        )

    @abstractmethod
    def forward(self, **kwargs) -> InterpretorModelOutput:
        pass

    @abstractmethod
    def predict(self, test_loaders, **kwargs):
        pass

    @abstractmethod
    def run_train(self, train_loader, test_loader=None):
        pass


class RavelInterpretorHypernetwork(BaseInterpretorHypernetwork):
    def __init__(self, config: DictConfig, device):
        super().__init__(config, device)
        self.interpretor_config.das_dimension = config.model.das_dimension

        # Target model configs
        self.target_model_config = AutoConfig.from_pretrained(
            config.model.target_model_name_or_path
        )
        self.interpretor_config.target_hidden_size = (
            self.target_model_config.hidden_size
        )

        # Ablation configs
        self.interpretor_config.ablate_base_token_attention = (
            config.model.ablate_base_token_attention
        )
        self.interpretor_config.ablate_source_token_attention = (
            config.model.ablate_source_token_attention
        )
        self.interpretor_config.break_asymmetric = config.model.break_asymmetric
        self.interpretor_config.freeze_das_module = config.model.freeze_das_module

        # Ridge/projective configs
        self.interpretor_config.lambda_parameter = config.model.lambda_parameter
        self.interpretor_config.importance_power = config.model.importance_power
        self.interpretor_config.epsilon = config.model.epsilon
        self.interpretor_config.ridge_parameterization = (
            config.model.ridge_parameterization
        )
        self.interpretor_config.scoring_dimension = config.model.scoring_dimension
        self.interpretor_config.return_penalty = config.model.return_penalty
        self.interpretor_config.dict_size = config.model.dict_size
        self.interpretor_config.orthogonal_init = config.model.orthogonal_init
        self.interpretor_config.selection_mechanism = config.model.selection_mechanism
        self.interpretor_config.hat_matrix = config.model.hat_matrix

        # Initialize model components
        interpretor_cls = INTERPRETOR_CLS.get(config.model.intepretor_type, None)
        if interpretor_cls is None:
            raise ValueError("Invalid interpretor type")

        with init_on_device(self.device):
            self.interpretor = interpretor_cls(
                self.interpretor_config,
                target_model_name_or_path=config.model.target_model_name_or_path,
                subspace_module=config.model.subspace_module,
                compute_metrics=config.training.compute_metrics,
                device=self.device,
            )

        # DAS configs
        self.use_das_intervention = config.model.subspace_module is not None
        self.das_dim = config.model.das_dimension

        # DAS Training Hyperparameters
        self.rotate_lr = config.training.get("rotate_lr", 1e-3)
        self.boundary_lr = config.training.get("boundary_lr", 1e-2)
        self.das_temperature_start = config.training.get("das_temperature_start", 50.0)
        self.das_temperature_end = config.training.get("das_temperature_end", 0.1)

    def save_model(self, save_dir):
        super().save_model(save_dir)
        if self.use_das_intervention:
            torch.save(
                self.interpretor.intervention_module.state_dict(),
                os.path.join(save_dir, "das.pt"),
            )

    def load_model(self, load_dir):
        super().load_model(load_dir)
        if self.use_das_intervention:
            self.interpretor.intervention_module.load_state_dict(
                torch.load(os.path.join(load_dir, "das.pt"), weights_only=True)
            )

    def set_intervention_layer(self, intervention_layer):
        self.interpretor.config.intervention_layer = intervention_layer

    def forward(
        self,
        editor_input_ids: Optional[torch.LongTensor] = None,
        base_input_ids: Optional[torch.LongTensor] = None,
        base_attention_mask: Optional[torch.FloatTensor] = None,
        base_intervention_mask: Optional[torch.FloatTensor] = None,
        source_input_ids: Optional[torch.LongTensor] = None,
        source_attention_mask: Optional[torch.FloatTensor] = None,
        source_intervention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_intervention_weight: bool = True,
        is_causal: Optional[torch.BoolTensor] = None,
        causal_loss_weight: float = 1.0,
        iso_loss_weight: float = 1.0,
        source_intervention_weight: Optional[torch.FloatTensor] = None,
        base_intervention_weight: Optional[torch.FloatTensor] = None,
    ) -> InterpretorModelOutput:
        _pred: InterpretorModelOutput = self.interpretor(
            editor_input_ids=editor_input_ids,
            editor_attention_mask=editor_input_ids
            != self.interpretor_config.eos_token_id,
            base_input_ids=base_input_ids,
            base_attention_mask=base_attention_mask,
            base_intervention_mask=base_intervention_mask,
            source_input_ids=source_input_ids,
            source_attention_mask=source_attention_mask,
            source_intervention_mask=source_intervention_mask,
            output_intervention_weight=output_intervention_weight,
            source_intervention_weight=source_intervention_weight,
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
                loss_weight[~is_causal, :] = iso_loss_weight

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
                editor_input_ids=batch["editor_input_ids"].to(self.device),
                base_input_ids=batch["base_input_ids"].to(self.device),
                base_attention_mask=batch["base_attention_mask"].to(self.device),
                base_intervention_mask=batch["base_intervention_mask"].to(self.device),
                source_input_ids=batch["source_input_ids"].to(self.device),
                source_attention_mask=batch["source_attention_mask"].to(self.device),
                source_intervention_mask=batch["source_intervention_mask"].to(
                    self.device
                ),
                labels=batch["labels"].to(self.device),
                output_intervention_weight=True,
            )

            batch_pred_ids = torch.argmax(predictions["logits"], dim=-1)
            batch_full_output = self.tokenizer.batch_decode(
                batch_pred_ids, skip_special_tokens=True
            )

            batch_output = []
            correct = 0

            for i, (label, pred_ids) in enumerate(
                zip(batch["labels"].to("cuda"), batch_pred_ids)
            ):
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

        source_intervention_weight = predictions.source_intervention_weight
        base_intervention_weight = predictions.base_intervention_weight

        return_dict = {
            "batch_output": batch_output,
            "batch_full_output": batch_full_output,
            "batch_source_intervention_weight": source_intervention_weight,
            "batch_base_intervention_weight": base_intervention_weight,
            "n_correct": correct,
            "correct_idxs": correct_idxs,
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
        axes=None,
    ):
        fig = None
        batch_id = idxs // batch_size
        example_id = idxs % batch_size

        for i, batch in enumerate(data_loader):
            if i == batch_id:
                break

        editor_input_ids = batch["editor_input_ids"][example_id]
        base_input_ids = batch["base_input_ids"][example_id]
        source_input_ids = batch["source_input_ids"][example_id]
        label = batch["labels"][example_id]

        source_padding_idx = source_input_ids == self.tokenizer.pad_token_id
        base_padding_idx = base_input_ids == self.tokenizer.pad_token_id

        source_input_ids = source_input_ids[~source_padding_idx]
        base_input_ids = base_input_ids[~base_padding_idx]

        if indicate_masked_tokens:
            base_intervention_mask = batch["base_intervention_mask"][example_id]
            source_intervention_mask = batch["source_intervention_mask"][example_id]
            base_intervention_mask = base_intervention_mask[~base_padding_idx]
            source_intervention_mask = source_intervention_mask[~source_padding_idx]

        # Add a False value to the end of the source_padding_idx to account for the [SELF] token

        source_axis = [self.tokenizer.decode([i]) for i in source_input_ids]
        base_axis = [self.tokenizer.decode([i]) for i in base_input_ids]

        for axis in [source_axis, base_axis]:
            for i, token in enumerate(axis):
                if token == self.tokenizer.bos_token:
                    axis[i] = "[BOS]"

        editor_text = self.tokenizer.decode(editor_input_ids, skip_special_tokens=True)
        label = label[label != -100]
        label = self.tokenizer.decode(label)

        def plot_inference_model(ax, intervention_weight, prediction):
            if indicate_masked_tokens:
                masks = torch.ones_like(intervention_weight)

                for i, source_mask in enumerate(source_intervention_mask):
                    for j, base_mask in enumerate(base_intervention_mask):
                        if source_mask and base_mask:
                            masks[i, j] = False
                # masks[:, base_intervention_mask] = 0.0
                masks = masks.float().cpu().numpy()
            else:
                masks = None
            sns.heatmap(
                intervention_weight.float().cpu().numpy(),
                xticklabels=base_axis,
                yticklabels=source_axis,
                ax=ax,
                annot=annot,
                fmt=f".{digits}f",
                mask=masks,
            )

            if simplified_annot:
                for child in ax.get_children():
                    if isinstance(child, plt.Text):
                        # If child
                        if child.get_text().startswith("0."):
                            if (
                                child.get_text()
                                .replace("0.", "")
                                .replace("0", "")
                                .strip()
                                == ""
                            ):
                                child.set_text("0")
                            else:
                                child.set_text(child.get_text().replace("0.", "."))
                        elif child.get_text().startswith("1"):
                            child.set_text("1")

            # Render the cell at (0, 0) with black background
            if contain_title:
                ax.set_title(
                    f"Instruction: {editor_text}     Label: {label}    Pred: {prediction}"
                )
            else:
                print(
                    f"Instruction: {editor_text}     Label: {label}    Pred: {prediction}"
                )

            ax.set_xlabel("Base Sentence Tokens")
            ax.set_ylabel("Counterfactual Sentence Tokens")

        results = self.inspect_batch_prediction_ouptuts(batch, eval_n_label_tokens=3)
        predictions = results["batch_output"][example_id]
        source_intervention_weight = results["batch_source_intervention_weight"][
            example_id
        ]
        base_intervention_weight = results["batch_base_intervention_weight"][example_id]

        # set background color to be grey

        style = sns.axes_style("dark")
        style["axes.facecolor"] = "#100a17"

        sns.set(style=style, font_scale=font_scale)
        if axes is None:
            fig, axes = plt.subplots(2, 1, figsize=(15, 4))

        source_intervention_weight = source_intervention_weight[
            ~source_padding_idx
        ].unsqueeze(0)

        base_intervention_weight = base_intervention_weight[
            ~base_padding_idx
        ].unsqueeze(0)
        source_intervention_mask = (source_intervention_mask == 0).unsqueeze(0)
        base_intervention_mask = (base_intervention_mask == 0).unsqueeze(0)

        sns.heatmap(
            source_intervention_weight.float().cpu().numpy(),
            xticklabels=source_axis,
            ax=axes[0],
            annot=annot,
            fmt=f".{digits}f",
            mask=source_intervention_mask.float().cpu().numpy(),
        )
        sns.heatmap(
            base_intervention_weight.float().cpu().numpy(),
            xticklabels=base_axis,
            ax=axes[1],
            annot=annot,
            fmt=f".{digits}f",
            mask=base_intervention_mask.float().cpu().numpy(),
        )

        axes[0].set_yticklabels([""])
        axes[1].set_yticklabels([""])

        axes[0].set_ylabel("Source Sentence")
        axes[1].set_ylabel("Base Sentence")

        axes[0].set_title(
            f"Instruction: {editor_text}     Label: {label}    Pred: {predictions}"
        )

        plt.subplots_adjust(hspace=1)

        return fig, axes

    @torch.no_grad()
    def predict(
        self,
        test_loaders: List[NamedDataLoader],
        inference_mode=None,
        eval_n_label_tokens=None,
    ):
        assert inference_mode in [
            None,
            "column_argmax",
            "global_argmax",
            "groundtruth",
            "bidding_argmax",
        ]

        per_dataset_accuracies = {}
        per_dataset_test_loss = {}
        per_dataset_correct_idxs = {}

        self.interpretor.eval()

        for test_loader in test_loaders:
            test_loss = []
            correct_idxs = []
            is_causal = []

            for batch_id, batch in tqdm(
                enumerate(test_loader.data_loader),
                desc="Evaluating accuracy",
                total=self.max_eval_steps
                if self.max_eval_steps > 0
                else len(test_loader.data_loader),
            ):
                # Move entire batch to GPU once
                batch = {k: v.to(self.device) for k, v in batch.items()}

                predictions = self.forward(
                    editor_input_ids=batch["editor_input_ids"],
                    base_input_ids=batch["base_input_ids"],
                    base_attention_mask=batch["base_attention_mask"],
                    base_intervention_mask=batch["base_intervention_mask"],
                    source_input_ids=batch["source_input_ids"],
                    source_attention_mask=batch["source_attention_mask"],
                    source_intervention_mask=batch["source_intervention_mask"],
                    labels=batch["labels"],
                )
                test_loss.append(predictions["loss"].item())

                batch_pred_ids = torch.argmax(predictions["logits"], dim=-1)
                is_causal.extend(batch["is_causal"].cpu().numpy().tolist())

                for i, (label, pred_ids) in enumerate(
                    zip(batch["labels"].to(self.device), batch_pred_ids)
                ):
                    label_idx = label != -100
                    output_idx = torch.zeros_like(label_idx)
                    output_idx[:-1] = label_idx[1:]

                    label = label[label_idx]
                    pred_ids = pred_ids[output_idx]

                    if (
                        eval_n_label_tokens is not None
                        and len(label) > eval_n_label_tokens
                    ):
                        label = label[:eval_n_label_tokens]
                        pred_ids = pred_ids[:eval_n_label_tokens]

                    is_correct = (
                        torch.sum(label == pred_ids) == torch.numel(label)
                    ).item()
                    if is_correct:
                        correct_idxs.append(batch_id * len(batch["labels"]) + i)

                if self.max_eval_steps > 0 and batch_id + 1 > self.max_eval_steps:
                    break

            total_causal = sum(is_causal)
            total_isolate = len(is_causal) - total_causal

            correct_causal = sum([is_causal[i] for i in correct_idxs])
            correct_isolate = len(correct_idxs) - correct_causal

            causal_acc = correct_causal / total_causal if total_causal > 0 else 0.0
            isolate_acc = correct_isolate / total_isolate if total_isolate > 0 else 0.0

            disentangle_acc = (
                0.5 * (causal_acc + isolate_acc) if total_isolate > 0 else causal_acc
            )

            accuracies = {
                "causal": causal_acc,
                "isolate": isolate_acc,
                "disentangle": disentangle_acc,
            }

            per_dataset_accuracies[test_loader.name] = accuracies
            per_dataset_test_loss[test_loader.name] = sum(test_loss) / len(test_loss)
            per_dataset_correct_idxs[test_loader.name] = correct_idxs

        return per_dataset_accuracies, per_dataset_test_loss, per_dataset_correct_idxs

    def _entropy(self, x, mean=True):
        if mean:
            return -torch.sum(x * torch.log(x + 1e-12), dim=-1).mean()

        return -torch.sum(x * torch.log(x + 1e-12), dim=-1)

    def run_train(
        self,
        train_loader,
        test_loader: Optional[NamedDataLoader | List[NamedDataLoader]] = None,
    ):
        inference_modes = self.config.model.inference_modes
        epochs = self.config.training.n_epochs
        steps = self.config.training.n_steps
        eval_per_steps = self.config.training.eval_per_steps
        checkpoint_per_steps = self.config.training.checkpoint_per_steps
        save_dir = self.config.training.save_dir
        save_model = self.config.training.save_model
        run_name = self.config.wandb_config.run_name
        causal_loss_weight = self.config.loss.causal_loss_weight
        iso_loss_weight = self.config.loss.iso_loss_weight
        target_intervention_num = self.config.loss.get("target_intervention_num", None)
        lr = self.config.training.lr
        weight_decay = self.config.training.weight_decay
        max_grad_norm = self.config.training.max_grad_norm
        sparsity_loss = self.config.loss.sparsity.apply
        sparsity_loss_weight = self.config.loss.sparsity.weight

        os.makedirs(os.path.join(save_dir, run_name), exist_ok=True)
        OmegaConf.save(self.config, os.path.join(save_dir, run_name, "config.yaml"))

        trainable_parameters = []
        for name, param in self.named_parameters():
            if "target_model" not in name:
                if "intervention_module" in name:
                    if "rotate_layer" in name:
                        trainable_parameters += [
                            {"params": param, "lr": self.rotate_lr, "weight_decay": 0.0}
                        ]
                    elif "mask_projection" in name:
                        trainable_parameters += [
                            {"params": param, "lr": self.boundary_lr}
                        ]
                    else:
                        trainable_parameters += [{"params": param}]
                else:
                    trainable_parameters += [{"params": param}]

        self.opt = optim.AdamW(
            trainable_parameters, lr=lr, weight_decay=weight_decay
        )  # usually: lr = 5e-5. 1e-3 worked well!

        # We can either specify total steps or epochs
        total_steps = len(train_loader) * epochs if steps <= 0 else steps
        if steps > 0:
            epochs = ceil(total_steps / len(train_loader))
        cur_steps = 0

        if self.use_das_intervention and hasattr(
            self.interpretor.intervention_module, "set_temperature"
        ):
            das_temperature_schedule = (
                torch.linspace(
                    self.das_temperature_start,
                    self.das_temperature_end,
                    total_steps + 1,
                )
                .to(self.interpretor_config.torch_dtype)
                .to(self.device)
            )
            self.interpretor.intervention_module.set_temperature(
                das_temperature_schedule[cur_steps]
            )

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
                for batch in train_loader:
                    if cur_steps >= total_steps:
                        logger.info("Training stopped. Reached max steps.")
                        break
                    if eval_per_steps is not None:
                        if cur_steps % eval_per_steps == 0:
                            # Evaluate the model
                            for mode in inference_modes:
                                accuracies, test_loss, _ = self.predict(
                                    test_loader,
                                    inference_mode=mode,
                                    eval_n_label_tokens=3,
                                )

                                text_mode = "default" if mode is None else mode

                                for loader in test_loader:
                                    causal_acc = accuracies[loader.name]["causal"]
                                    isolate_acc = accuracies[loader.name]["isolate"]
                                    disentangle_acc = accuracies[loader.name][
                                        "disentangle"
                                    ]

                                    if wandb.run:
                                        wandb.log(
                                            {
                                                f"{loader.name}/{text_mode}_test_average_loss": test_loss,
                                                f"{loader.name}/{text_mode}_causal_accuracy": causal_acc,
                                                f"{loader.name}/{text_mode}_isolate_accuracy": isolate_acc,
                                                f"{loader.name}/{text_mode}_disentangle_accuracy": disentangle_acc,
                                            }
                                        )

                                    logger.info(
                                        "[%s] Under Inference Mode: %s",
                                        loader.name,
                                        text_mode,
                                    )
                                    logger.info(
                                        "[%s] Disentangle Acc: %.4f, Causal Acc: %.4f, Isolate Acc: %.4f, Test Loss: %.4f",
                                        loader.name,
                                        disentangle_acc,
                                        causal_acc,
                                        isolate_acc,
                                        test_loss[loader.name],
                                    )

                    if checkpoint_per_steps is not None:
                        if (
                            cur_steps % checkpoint_per_steps == 0
                            and save_dir is not None
                            and save_model
                        ):
                            logger.info(
                                "Saving model to {}".format(
                                    os.path.join(
                                        save_dir,
                                        run_name,
                                        f"model_epoch_{epoch}_step_{cur_steps}",
                                    )
                                )
                            )
                            self.save_model(
                                os.path.join(
                                    save_dir,
                                    run_name,
                                    f"model_epoch_{epoch}_step_{cur_steps}",
                                )
                            )

                    self.batch = batch
                    current_batch_size = len(batch["editor_input_ids"])
                    num_datapoints_in_epoch += current_batch_size

                    # Move all batch items to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    prediction = self.forward(
                        editor_input_ids=batch["editor_input_ids"],
                        base_input_ids=batch["base_input_ids"],
                        base_attention_mask=batch["base_attention_mask"],
                        base_intervention_mask=batch["base_intervention_mask"],
                        source_input_ids=batch["source_input_ids"],
                        source_attention_mask=batch["source_attention_mask"],
                        source_intervention_mask=batch["source_intervention_mask"],
                        labels=batch["labels"],
                        is_causal=batch["is_causal"],
                        causal_loss_weight=causal_loss_weight,
                        iso_loss_weight=iso_loss_weight,
                        output_intervention_weight=True,
                    )

                    training_loss = 0

                    prediction_loss = prediction["loss"]
                    training_loss += prediction_loss

                    if sparsity_loss:
                        source_intervention_weight = prediction[
                            "source_intervention_weight"
                        ]
                        base_intervention_weight = prediction[
                            "base_intervention_weight"
                        ]

                        source_entropy = self._entropy(
                            source_intervention_weight, mean=True
                        )
                        base_entropy = self._entropy(
                            base_intervention_weight, mean=True
                        )

                        sparsity_loss = sparsity_loss_weight * (
                            source_entropy + base_entropy
                        )
                        training_loss += sparsity_loss

                    if target_intervention_num is not None:
                        assert isinstance(target_intervention_num, int)

                        source_target_intervention_num = self._intervention_number(
                            prediction.intervention_weight, mean=True
                        )
                        intervention_number_loss = (
                            source_target_intervention_num - target_intervention_num
                        ) ** 2
                        training_loss += intervention_number_loss

                    training_loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.parameters(), max_grad_norm
                    )

                    # Log more gradient norms
                    if hasattr(self.interpretor.intervention_module, "gradient_norms"):
                        grad_norm_metrics = (
                            self.interpretor.intervention_module.gradient_norms()
                        )
                    else:
                        grad_norm_metrics = {}

                    self.opt.step()
                    # metrics
                    epoch_train_loss += training_loss.item() * current_batch_size
                    self.opt.zero_grad()

                    metrics = {
                        "counters/step": cur_steps,
                        "counters/epoch": cur_steps / len(train_loader),
                        "train_batch_prediction_loss": prediction_loss.item(),
                        "grad_norm": grad_norm.item(),
                        **grad_norm_metrics,
                    }
                    if (
                        hasattr(prediction, "metrics")
                        and prediction.metrics is not None
                    ):
                        metrics.update(prediction.metrics)
                    if target_intervention_num is not None:
                        metrics["train_batch_#intervention_loss"] = (
                            intervention_number_loss.item()
                        )

                    if wandb.run:
                        wandb.log(metrics)
                    if cur_steps % 5 == 0:
                        output_metrics = {**metrics}

                        logger.info(output_metrics)

                    # Update progress bar
                    pbar.update(1)  # note: this was incorrectly displaying before!
                    cur_steps += 1
                    if self.use_das_intervention and hasattr(
                        self.interpretor.intervention_module, "set_temperature"
                    ):
                        self.interpretor.intervention_module.set_temperature(
                            das_temperature_schedule[cur_steps]
                        )

                if wandb.run:
                    wandb.log(
                        {
                            "epoch_train_total_loss": epoch_train_loss
                            / num_datapoints_in_epoch,
                        }
                    )

        result_dict = {}
        for inference_mode in inference_modes:
            accs, test_loss, correct_indices = self.predict(
                test_loader, inference_mode=inference_mode, eval_n_label_tokens=3
            )
            if inference_mode is None:
                inference_mode = "default"
            result_dict[inference_mode] = {
                "accs": accs,
                "test_loss": test_loss,
                "correct_indices": correct_indices,
            }

            for k, v in accs.items():
                logger.info(f"{inference_mode} {k}: {v}")

        # Save the final model
        if save_model and save_dir:
            self.save_model(os.path.join(save_dir, run_name, "final_model"))
        if save_dir:
            json.dump(
                result_dict,
                open(os.path.join(save_dir, run_name, "final_result.json"), "w"),
            )


class SteeringInterpretorHypernetwork(BaseInterpretorHypernetwork):
    def __init__(self, config: DictConfig, device):
        super().__init__(config, device)
        # Define sizes of embedding tables for layers and positions
        self.interpretor_config.num_target_layers = (
            max(
                config.model.intervention_layer
                if isinstance(config.model.intervention_layer, ListConfig)
                else [config.model.intervention_layer]
            )
            + 1
        )
        self.interpretor_config.num_target_positions = (
            max(
                parse_positions_varlen(
                    config.model.intervention_positions, seq_len=config.model.max_length
                )
            )
            + 1
        )
        self.interpretor_config.target_hidden_size = config.model.target_hidden_size
        self.rotate_lr = config.training.get("rotate_lr", 1e-3)

        # Initialize model components
        interpretor_cls = INTERPRETOR_CLS.get(config.model.intepretor_type, None)
        if interpretor_cls is None:
            raise ValueError("Invalid interpretor type")

        with init_on_device(self.device):
            self.interpretor = interpretor_cls(
                self.interpretor_config,
                target_model_name_or_path=config.model.target_model_name_or_path,
                subspace_module=config.model.subspace_module,
                compute_metrics=config.training.compute_metrics,
                device=self.device,
            )

        self.lm_judge_evaluator = None
        self.winrate_evaluator = None

    def forward(
        self,
        editor_input_ids: Optional[torch.LongTensor] = None,
        base_input_ids: Optional[torch.LongTensor] = None,
        base_attention_mask: Optional[torch.FloatTensor] = None,
        base_intervention_mask: Optional[torch.FloatTensor] = None,
        target_input_ids: Optional[torch.LongTensor] = None,
        target_attention_mask: Optional[torch.FloatTensor] = None,
        target_position_ids: Optional[torch.LongTensor] = None,
        intervention_layers: Optional[torch.Tensor] = None,
        intervention_positions: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        is_causal: Optional[torch.BoolTensor] = None,
        causal_loss_weight: float = 1.0,
        iso_loss_weight: float = 1.0,
    ) -> InterpretorModelOutput:
        _pred: InterpretorModelOutput = self.interpretor(
            editor_input_ids=editor_input_ids,
            editor_attention_mask=editor_input_ids
            != self.interpretor_config.eos_token_id,
            base_input_ids=base_input_ids,
            base_attention_mask=base_attention_mask,
            base_intervention_mask=base_intervention_mask,
            target_input_ids=target_input_ids,
            target_attention_mask=target_attention_mask,
            target_position_ids=target_position_ids,
            intervention_layers=intervention_layers,
            intervention_positions=intervention_positions,
        )

        if labels is not None:
            log_prob_predictions = torch.nn.functional.log_softmax(
                _pred.logits.reshape(-1, _pred.logits.shape[-1]),
                dim=1,
            )

            if is_causal is not None:
                loss_weight = torch.ones_like(labels, dtype=log_prob_predictions.dtype)
                loss_weight[is_causal, :] = causal_loss_weight
                loss_weight[~is_causal, :] = iso_loss_weight

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

    @torch.no_grad()
    def generate(
        self,
        *args,
        **kwargs,
    ) -> ModelOutput:
        """Wrapper for generate method."""
        return self.interpretor.generate(*args, **kwargs)

    @torch.no_grad()
    def predict(
        self,
        test_dataloader: NamedDataLoader,
    ) -> Dict[str, float]:
        """
        Compute average loss and perplexity metrics for test data.
        Returns dict with metrics.
        """
        total_loss = 0
        total_perplexity = 0
        num_batches = 0

        for step, batch in enumerate(
            tqdm(test_dataloader.data_loader, desc="eval steps")
        ):
            if self.max_eval_steps >= 0 and step >= self.max_eval_steps:
                break

            # Move batch to GPU
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass through model
            outputs = self.forward(
                editor_input_ids=batch["editor_input_ids"],
                base_input_ids=batch["base_input_ids"],
                base_attention_mask=batch["base_attention_mask"],
                base_intervention_mask=batch["base_intervention_mask"],
                target_input_ids=batch.get("target_input_ids"),
                target_attention_mask=batch.get("target_attention_mask"),
                target_position_ids=batch.get("target_position_ids"),
                intervention_layers=batch["intervention_layers"],
                intervention_positions=batch["intervention_positions"],
                labels=batch["labels"],
                is_causal=batch["is_causal"],
            )

            # Get loss and perplexity
            loss = outputs["loss"]
            perplexity = torch.exp(loss)

            total_loss += loss.item()
            total_perplexity += perplexity.item()
            num_batches += 1

        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_perplexity = total_perplexity / num_batches

        return {"loss": avg_loss, "perplexity": avg_perplexity}

    def run_train(
        self,
        train_loader,
        test_loader: Optional[NamedDataLoader | List[NamedDataLoader]] = None,
    ):
        epochs = self.config.training.n_epochs
        steps = self.config.training.n_steps
        eval_per_steps = self.config.training.eval_per_steps
        checkpoint_per_steps = self.config.training.checkpoint_per_steps
        save_dir = self.config.training.save_dir
        save_model = self.config.training.save_model
        run_name = self.config.wandb_config.run_name
        lr = self.config.training.lr
        weight_decay = self.config.training.weight_decay
        max_grad_norm = self.config.training.max_grad_norm

        os.makedirs(os.path.join(save_dir, run_name), exist_ok=True)
        OmegaConf.save(self.config, os.path.join(save_dir, run_name, "config.yaml"))

        trainable_parameters = []
        for name, param in self.named_parameters():
            if "target_model" not in name:
                if "das_module" in name:
                    if "rotate_layer" in name:
                        trainable_parameters += [
                            {"params": param, "lr": self.rotate_lr, "weight_decay": 0.0}
                        ]
                    else:
                        trainable_parameters += [{"params": param}]
                else:
                    trainable_parameters += [{"params": param}]

        self.opt = optim.AdamW(
            trainable_parameters, lr=lr, weight_decay=weight_decay
        )  # usually: lr = 5e-5. 1e-3 worked well!

        # We can either specify total steps or epochs
        total_steps = len(train_loader) * epochs if steps <= 0 else steps
        if steps > 0:
            epochs = ceil(total_steps / len(train_loader))
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
                for batch in train_loader:
                    if cur_steps >= total_steps:
                        logger.info("Training stopped. Reached max steps.")
                        break
                    if eval_per_steps is not None:
                        if cur_steps % eval_per_steps == 0:
                            metrics = {}
                            for loader in test_loader:
                                metrics[loader.name] = self.predict(loader)

                            # Dump to json file
                            if save_dir is not None:
                                metrics_path = os.path.join(
                                    save_dir,
                                    run_name,
                                    f"metrics_epoch_{epoch}_step_{cur_steps}.json",
                                )
                                os.makedirs(
                                    os.path.dirname(metrics_path), exist_ok=True
                                )
                                with open(metrics_path, "w") as f:
                                    json.dump(metrics, f, indent=2)

                    if checkpoint_per_steps is not None:
                        if (
                            cur_steps % checkpoint_per_steps == 0
                            and save_dir is not None
                            and save_model
                        ):
                            logger.info(
                                "Saving model to {}".format(
                                    os.path.join(
                                        save_dir,
                                        run_name,
                                        f"model_epoch_{epoch}_step_{cur_steps}",
                                    )
                                )
                            )
                            self.save_model(
                                os.path.join(
                                    save_dir,
                                    run_name,
                                    f"model_epoch_{epoch}_step_{cur_steps}",
                                )
                            )

                    self.batch = batch
                    current_batch_size = len(batch["editor_input_ids"])
                    num_datapoints_in_epoch += current_batch_size

                    # Move all batch items to device
                    batch = {
                        k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                        for k, v in batch.items()
                    }

                    prediction = self.forward(
                        editor_input_ids=batch["editor_input_ids"],
                        base_input_ids=batch["base_input_ids"],
                        base_attention_mask=batch["base_attention_mask"],
                        base_intervention_mask=batch["base_intervention_mask"],
                        target_input_ids=batch.get("target_input_ids"),
                        target_attention_mask=batch.get("target_attention_mask"),
                        target_position_ids=batch.get("target_position_ids"),
                        intervention_layers=batch["intervention_layers"],
                        intervention_positions=batch["intervention_positions"],
                        labels=batch["labels"],
                        is_causal=batch["is_causal"],
                    )

                    training_loss = 0

                    prediction_loss = prediction["loss"]
                    training_loss += prediction_loss

                    training_loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.parameters(), max_grad_norm
                    )

                    # Log more gradient norms
                    if hasattr(self.interpretor.intervention_module, "gradient_norms"):
                        grad_norm_metrics = (
                            self.interpretor.intervention_module.gradient_norms()
                        )
                    else:
                        grad_norm_metrics = {}

                    self.opt.step()
                    # metrics
                    epoch_train_loss += training_loss.item() * current_batch_size
                    self.opt.zero_grad()

                    metrics = {
                        "counters/step": cur_steps,
                        "counters/epoch": cur_steps / len(train_loader),
                        "train_batch_prediction_loss": prediction_loss.item(),
                        "grad_norm": grad_norm.item(),
                        "learning_rate": self.opt.param_groups[0]["lr"],
                        **prediction.metrics,
                        **grad_norm_metrics,
                    }

                    if wandb.run:
                        wandb.log(metrics)
                    if cur_steps % 5 == 0:
                        output_metrics = {**metrics}

                        logger.info(output_metrics)

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

        if self.config.training.max_eval_steps != 0:
            result_dict = {}
            for tl in test_loader:
                result_dict[tl.name] = self.predict(tl)

        # Save the final model
        if save_model and save_dir:
            self.save_model(os.path.join(save_dir, run_name, "final_model"))
        if save_dir:
            json.dump(
                result_dict,
                open(os.path.join(save_dir, run_name, "final_result.json"), "w"),
            )
