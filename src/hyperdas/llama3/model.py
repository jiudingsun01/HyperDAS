import json
import os
from math import ceil

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer

import wandb
from logger import get_logger

from ..das_utils import QuasiProjectiveIntervention
from ..utils import InterpretorModelOutput
from .modules import LlamaInterpretor, LlamaInterpretorConfig

logger = get_logger(__name__)


class RavelInterpretorHypernetwork(nn.Module):
    # Separating the editor config file, from its base model's configurations
    def __init__(
        self,
        model_name_or_path="/home/ubuntu/llama3-8b",
        num_editing_heads=32,
        chop_editor_at_layer=8,
        intervention_layer=0,
        subspace_module="ReflectSelect",
        torch_dtype=torch.bfloat16,
        das_dimension=None,
        initialize_from_scratch=False,
        ablate_base_token_attention=False,
        ablate_source_token_attention=False,
        break_asymmetric=False,
        device="cuda",
        compute_metrics=False,
    ):
        super().__init__()

        self.interpretor_config = LlamaInterpretorConfig.from_pretrained(
            model_name_or_path
        )
        self.interpretor_config.name_or_path = model_name_or_path
        self.interpretor_config.torch_dtype = torch_dtype
        self.interpretor_config.num_editing_heads = num_editing_heads
        self.interpretor_config.chop_editor_at_layer = chop_editor_at_layer
        self.interpretor_config.intervention_layer = intervention_layer
        self.interpretor_config._attn_implementation = "eager"
        self.interpretor_config.initialize_from_scratch = initialize_from_scratch
        self.interpretor_config.ablate_base_token_attention = (
            ablate_base_token_attention
        )
        self.interpretor_config.ablate_source_token_attention = (
            ablate_source_token_attention
        )
        self.interpretor_config.break_asymmetric = break_asymmetric

        self.interpretor = LlamaInterpretor(
            self.interpretor_config,
            subspace_module=subspace_module,
            das_dimension=das_dimension,
            device=device,
            compute_metrics=compute_metrics,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

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

        torch.save(
            self.interpretor.hypernetwork.state_dict(),
            os.path.join(save_dir, "hypernetwork.pt"),
        )
        if self.use_das_intervention:
            torch.save(
                self.interpretor.das_module.state_dict(),
                os.path.join(save_dir, "das.pt"),
            )

    def load_model(self, load_dir):
        self.interpretor.hypernetwork.load_state_dict(
            torch.load(os.path.join(load_dir, "hypernetwork.pt"))
        )
        if self.use_das_intervention:
            self.interpretor.das_module.load_state_dict(
                torch.load(os.path.join(load_dir, "das.pt"))
            )

    def set_intervention_layer(self, intervention_layer):
        self.interpretor.config.intervention_layer = intervention_layer

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
        iso_loss_weight: float = 1.0,
        intervention_weight: torch.Tensor = None,
        inference_mode=None,
    ):
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
            intervention_weight=intervention_weight,
            inference_mode=inference_mode,
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
    def inspect_batch_prediction_ouptuts(
        self, batch, inference_mode=None, eval_n_label_tokens=None
    ):
        assert inference_mode in [
            None,
            "column_argmax",
            "global_argmax",
            "groundtruth",
            "bidding_argmax",
        ]
        self.interpretor.eval()

        correct_idxs = []

        if inference_mode == "groundtruth":
            intervention_weight = torch.zeros(
                len(batch["editor_input_ids"]),
                batch["source_input_ids"].shape[1] + 1,
                batch["base_input_ids"].shape[1],
            ).to("cuda")
            intervention_weight[:, -1, :] = 1.0

            for i in range(len(batch["base_entity_position_ids"])):
                intervention_weight[i, -1, batch["base_entity_position_ids"][i]] = 0.0
                intervention_weight[
                    i,
                    batch["source_entity_position_ids"][i],
                    batch["base_entity_position_ids"][i],
                ] = 1.0

        else:
            intervention_weight = None

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
                intervention_weight=intervention_weight,
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

        intervention_weight = predictions.intervention_weight
        batch_intervention_entropy = self._intervention_entropy(
            intervention_weight, mean=False
        )

        return_dict = {
            "batch_output": batch_output,
            "batch_full_output": batch_full_output,
            "batch_intervention_weight": intervention_weight,
            "batch_intervention_entropy": batch_intervention_entropy,
            "n_correct": correct,
            "correct_idxs": correct_idxs,
        }
        return return_dict

    def plot_heatmap(
        self,
        data_loader,
        idxs,
        batch_size=4,
        inference_mode=None,
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

        plot_multiple_modes = type(inference_mode) == list and len(inference_mode) > 1
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
            source_intervention_mask = torch.cat(
                [source_intervention_mask, torch.tensor([True])]
            )

        # Add a False value to the end of the source_padding_idx to account for the [SELF] token
        intervention_weight_source_padding_idx = torch.cat(
            [source_padding_idx, torch.tensor([False])]
        )

        source_axis = [self.tokenizer.decode([i]) for i in source_input_ids] + [
            "[SELF]"
        ]
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

        def process_intervention_weight(intervention_weight):
            intervention_weight = intervention_weight[
                ~intervention_weight_source_padding_idx, :
            ]
            intervention_weight = intervention_weight[:, ~base_padding_idx]
            return intervention_weight

        if not plot_multiple_modes:
            inference_mode = (
                inference_mode if type(inference_mode) != list else inference_mode[0]
            )
            results = self.inspect_batch_prediction_ouptuts(
                batch, inference_mode=inference_mode, eval_n_label_tokens=3
            )
            predictions = results["batch_output"][example_id]
            intervention_weight = results["batch_intervention_weight"][example_id]
        else:
            results = []
            for mode in inference_mode:
                result = self.inspect_batch_prediction_ouptuts(
                    batch, inference_mode=mode, eval_n_label_tokens=3
                )
                results.append(result)

            predictions = [r["batch_output"][example_id] for r in results]
            intervention_weight = [
                r["batch_intervention_weight"][example_id] for r in results
            ]

        # set background color to be grey
        if plot_multiple_modes:
            style = sns.axes_style("dark")
            style["axes.facecolor"] = "#100a17"

            sns.set(style=style, font_scale=3)
            intervention_weight = [
                process_intervention_weight(iw) for iw in intervention_weight
            ]

            if axes is not None:
                assert len(axes) == len(inference_mode)
            else:
                fig, axes = plt.subplots(
                    1, len(inference_mode), figsize=(10 + 15 * len(inference_mode), 15)
                )

            for i, ax in enumerate(axes):
                plot_inference_model(
                    ax=ax,
                    intervention_weight=intervention_weight.pop(0),
                    prediction=predictions.pop(0),
                )

                if i != 0:
                    ax.set_ylabel("")  # remove y-axis label for all but the first plot

                if i != len(axes) - 1:
                    # remove the heatmap colorbar for all but the last plot
                    ax.collections[0].colorbar.remove()

        else:
            style = sns.axes_style("dark")
            style["axes.facecolor"] = "#100a17"

            sns.set(style=style, font_scale=font_scale)
            intervention_weight = process_intervention_weight(intervention_weight)
            if axes is None:
                fig, axes = plt.subplots(figsize=(15, 15))
            plot_inference_model(
                ax=axes, intervention_weight=intervention_weight, prediction=predictions
            )

        return fig, axes

    def eval_accuracy(self, test_loader, inference_mode=None, eval_n_label_tokens=None):
        assert inference_mode in [
            None,
            "column_argmax",
            "global_argmax",
            "groundtruth",
            "bidding_argmax",
        ]

        self.interpretor.eval()
        test_loss = []
        correct_idxs = []
        is_causal = []

        with torch.no_grad():
            for batch_id, batch in tqdm(
                enumerate(test_loader),
                desc="Evaluating accuracy",
                total=len(test_loader),
            ):
                if inference_mode == "groundtruth":
                    intervention_weight = torch.zeros(
                        len(batch["editor_input_ids"]),
                        batch["source_input_ids"].shape[1] + 1,
                        batch["base_input_ids"].shape[1],
                    ).to("cuda")
                    intervention_weight[:, -1, :] = 1.0

                    for i in range(len(batch["base_entity_position_ids"])):
                        intervention_weight[
                            i, -1, batch["base_entity_position_ids"][i]
                        ] = 0.0
                        intervention_weight[
                            i,
                            batch["source_entity_position_ids"][i],
                            batch["base_entity_position_ids"][i],
                        ] = 1.0
                else:
                    intervention_weight = None

                predictions = self.forward(
                    editor_input_ids=batch["editor_input_ids"].to("cuda"),
                    base_input_ids=batch["base_input_ids"].to("cuda"),
                    base_attention_mask=batch["base_attention_mask"].to("cuda"),
                    base_intervention_mask=batch["base_intervention_mask"].to("cuda"),
                    source_input_ids=batch["source_input_ids"].to("cuda"),
                    source_attention_mask=batch["source_attention_mask"].to("cuda"),
                    source_intervention_mask=batch["source_intervention_mask"].to(
                        "cuda"
                    ),
                    labels=batch["labels"].to("cuda"),
                    inference_mode=inference_mode,
                    intervention_weight=intervention_weight,
                )
                test_loss.append(predictions["loss"].item())
                if isinstance(self.interpretor.das_module, QuasiProjectiveIntervention):
                    self.interpretor.zero_penalty()

                batch_pred_ids = torch.argmax(predictions["logits"], dim=-1)
                is_causal.extend(batch["is_causal"].cpu().numpy().tolist())

                for i, (label, pred_ids) in enumerate(
                    zip(batch["labels"].to("cuda"), batch_pred_ids)
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

        return accuracies, sum(test_loss) / len(test_loss), correct_idxs

    def _intervention_number(self, intervention_weight, mean=True):
        p = intervention_weight[:, :-1, :]
        p = torch.sum(p, dim=-1)
        p = torch.sum(p, dim=-1)

        if mean:
            p = torch.mean(p)

        return p

    def _intervention_entropy(
        self,
        intervention_weight,
        mean=True,
        normalize=False,
        balanced=False,
        atticus=False,
    ):
        """
        Normalize the weight along dim=-1; if weight is zero across all tokens, then return itself
        """

        def __normalize(weight):
            weight_sum = torch.sum(weight, dim=-1, keepdim=True)
            weight_sum[weight_sum == 0.0] = 1.0
            return weight / weight_sum

        def __balance(weight):
            weight_sum = torch.sum(weight, dim=-1, keepdim=False)
            weight_sum[weight_sum == 0.0] = 1.0
            return weight_sum

        def __append_column(weight):
            weight_sum = torch.sum(weight, dim=-1, keepdim=False)
            new_column = torch.max(
                1.0 - weight_sum, torch.tensor(0.0).to(weight.device)
            )
            return torch.cat([weight, new_column.unsqueeze(-1)], dim=-1)

        p = intervention_weight[:, :-1, :]

        if atticus:
            p = __append_column(p)

        if normalize:
            p = __normalize(p)

        logp = torch.log(p + 1e-8)
        entropy = -torch.sum(p * logp, dim=-1)

        if balanced:
            balanced_weight = __balance(intervention_weight[:, :-1, :])

            entropy = entropy * torch.pow(balanced_weight, 0.5)

        if mean:
            entropy = torch.mean(entropy, dim=0)
            entropy = torch.mean(entropy)

        return entropy

    def run_train(
        self,
        train_loader,
        test_loader=None,
        inference_modes=[None],
        epochs=1,
        steps=-1,
        eval_per_steps: int = None,
        checkpoint_per_steps: int = None,
        apply_source_selection_sparsity_loss=False,
        sparsity_loss_weight_start=0.5,
        sparsity_loss_weight_end=1.0,
        sparsity_loss_warm_up_ratio=0.1,
        causal_loss_weight=1.0,
        iso_loss_weight=1.0,
        lr=3e-4,
        weight_decay=0.01,
        save_dir=None,
        save_model=False,
        schedule_sparsity_loss=True,
        target_intervention_num=None,
    ):
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        trainable_parameters = []
        for name, param in self.named_parameters():
            if "target_model" not in name:
                if "das_module" in name:
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

        if self.use_das_intervention:
            das_temperature_schedule = (
                torch.linspace(
                    self.das_temperature_start,
                    self.das_temperature_end,
                    total_steps + 1,
                )
                .to(self.interpretor_config.torch_dtype)
                .to("cuda")
            )
            self.interpretor.das_module.set_temperature(
                das_temperature_schedule[cur_steps]
            )

        # Create a scheduler for the sparsity loss that is very small at the beginning from sparsity_loss_weight_start and increases to the sparsity_loss_weight_end
        # over the course of the training. Before sparsity_loss_warm_up_ratio * total_steps steps, the sparsity loss is not applied.
        if schedule_sparsity_loss:
            warm_up_steps = int(sparsity_loss_warm_up_ratio * total_steps)
            sparsity_loss_schedule = (
                torch.linspace(
                    sparsity_loss_weight_start,
                    sparsity_loss_weight_end,
                    total_steps + 1,
                )
                .to(self.interpretor_config.torch_dtype)
                .to("cuda")
            )
            sparsity_loss_schedule[:warm_up_steps] = 0.0

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
                for step, batch in enumerate(train_loader):
                    if step >= total_steps:
                        logger.info("Training stopped. Reached max steps.")
                        break
                    if eval_per_steps is not None:
                        if cur_steps % eval_per_steps == 0:
                            # Evaluate the model

                            for mode in inference_modes:
                                accuracies, test_loss, _ = self.eval_accuracy(
                                    test_loader,
                                    inference_mode=mode,
                                    eval_n_label_tokens=3,
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

                                logger.info("Under Inference Mode: %s", text_mode)
                                logger.info(
                                    "Disentangle Acc: %.4f, Causal Acc: %.4f, Isolate Acc: %.4f, Test Loss: %.4f",
                                    disentangle_acc,
                                    causal_acc,
                                    isolate_acc,
                                    test_loss,
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
                                        save_dir, f"model_epoch_{epoch}_step_{step}"
                                    )
                                )
                            )
                            self.save_model(
                                os.path.join(
                                    save_dir, f"model_epoch_{epoch}_step_{step}"
                                )
                            )

                    self.batch = batch
                    current_batch_size = len(batch["editor_input_ids"])
                    num_datapoints_in_epoch += current_batch_size

                    prediction = self.forward(
                        editor_input_ids=batch["editor_input_ids"].to("cuda"),
                        base_input_ids=batch["base_input_ids"].to("cuda"),
                        base_attention_mask=batch["base_attention_mask"].to("cuda"),
                        base_intervention_mask=batch["base_intervention_mask"].to(
                            "cuda"
                        ),
                        source_input_ids=batch["source_input_ids"].to("cuda"),
                        source_attention_mask=batch["source_attention_mask"].to("cuda"),
                        source_intervention_mask=batch["source_intervention_mask"].to(
                            "cuda"
                        ),
                        labels=batch["labels"].to("cuda"),
                        is_causal=batch["is_causal"].to("cuda"),
                        causal_loss_weight=causal_loss_weight,
                        iso_loss_weight=iso_loss_weight,
                        output_intervention_weight=True,
                        inference_mode=None,
                    )

                    training_loss = 0

                    prediction_loss = prediction["loss"]
                    training_loss += prediction_loss

                    if apply_source_selection_sparsity_loss:
                        source_selection_sparsity_loss = self._intervention_entropy(
                            prediction.intervention_weight
                        )

                        step_sparsity_loss_weight = sparsity_loss_schedule[cur_steps]
                        training_loss += (
                            step_sparsity_loss_weight * source_selection_sparsity_loss
                        )

                    if isinstance(
                        self.interpretor.das_module, QuasiProjectiveIntervention
                    ):
                        penalty = self.interpretor.get_penalty()
                        training_loss += penalty
                        self.interpretor.zero_penalty()

                    if target_intervention_num is not None:
                        assert type(target_intervention_num) == int

                        source_target_intervention_num = self._intervention_number(
                            prediction.intervention_weight, mean=True
                        )
                        intervention_number_loss = (
                            source_target_intervention_num - target_intervention_num
                        ) ** 2
                        training_loss += intervention_number_loss

                    training_loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 4.0)
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
                        "train_batch_sparsity_loss": source_selection_sparsity_loss.item(),
                        **prediction.metrics,
                    }
                    if target_intervention_num is not None:
                        metrics["train_batch_#intervention_loss"] = (
                            intervention_number_loss.item()
                        )

                    if isinstance(
                        self.interpretor.das_module, QuasiProjectiveIntervention
                    ):
                        metrics["train_batch_penalty"] = penalty.item()

                    if wandb.run:
                        wandb.log(metrics)
                    if cur_steps % 5 == 0:
                        output_metrics = {**metrics}
                        if apply_source_selection_sparsity_loss:
                            output_metrics["sparsity_weight"] = (
                                step_sparsity_loss_weight.item()
                            )

                        print(output_metrics)

                    # Update progress bar
                    pbar.update(1)  # note: this was incorrectly displaying before!
                    cur_steps += 1
                    if self.use_das_intervention:
                        self.interpretor.das_module.set_temperature(
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
            accs, test_loss, correct_indices = self.eval_accuracy(
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
                print(f"{inference_mode} {k}: {v}")

        # Save the final model
        if save_dir is not None:
            if save_model:
                self.save_model(os.path.join(save_dir, "final_model"))
            json.dump(
                result_dict, open(os.path.join(save_dir, "final_result.json"), "w")
            )
