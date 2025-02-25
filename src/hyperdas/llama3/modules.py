from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, List, Literal, Mapping, Optional, Tuple, TypeVar, Union
import warnings
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    PreTrainedModel,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from transformers.utils import ModelOutput

from logger import get_logger

from ..das_utils import (
    BoundlessRotatedSpaceIntervention,
    InterventionModuleOutput,
    LowRankRotatedSpaceIntervention,
    QuasiProjectiveIntervention,
    ReflectiveLowRankRotatedSpaceIntervention,
    SelectiveLowRankRotatedSpaceIntervention,
)
from ..reft_utils import (
    LoreftIntervention,
    ReFTHypernetwork,
    SelectiveLoreftIntervention,
    SteeringVecLoreftIntervention,
)
from ..utils import (
    InterpretorModelOutput,
    InterpretorModelOutputWithLearnedSource,
    SimpleTensorCache,
    add_fwd_hooks,
    assign_layer_indices,
)
from .layers import (
    InterpretorUnembedCrossAttention,
    LlamaDecoderLayerWithDoubleCrossAttention,
)

T_Learned = TypeVar("T_Learned", bound="LlamaInterpretorWithLearnedSource")
T_Regular = TypeVar("T_Regular", bound="LlamaInterpretorWithBaseSource")
T_Reft = TypeVar("T_Reft", bound="LlamaInterpretorForReFTGeneration")

logger = get_logger(__name__)


@dataclass
class HyperDASModelOutputWithCrossAttentions(BaseModelOutputWithPast):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    last_source_hidden_state: torch.FloatTensor = None
    last_base_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    source_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    base_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class LlamaInterpretorConfig(LlamaConfig):
    boundless_das: bool = False
    torch_dtype = torch.bfloat16
    chop_editor_at_layer: int = -1
    num_editing_heads: int = 32
    compute_position_ids: bool = True
    intervention_layer: int = 24
    initialize_from_scratch: bool = False
    num_target_model_layers: int = 32
    das_dimension: int = 128
    # For projective ridge regression
    return_penalty: bool = True
    lambda_parameter: int = 10
    importance_power: int = -2
    epsilon: float = 1e-6
    ridge_parameterization = "inv_alpha"
    selection_mechanism: Literal["full", "topk", "dynamic"] = "full"
    dict_size: int = None
    orthogonal_init: bool = False
    scoring_dimension: int = 8
    hat_matrix: bool = False
    # Other
    freeze_das_module: List[str] = None
    # For reft generation
    target_hidden_size: int = None
    num_target_layers: int = None
    num_target_positions: int = None
    dropout: float = 0.05
    hypernetwork_type: Literal["lm", "mlp"] = "lm"


class LlamaModelWithCrossAttention(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        source_encoder_hidden_states: Optional[torch.Tensor] = None,
        source_encoder_attention_mask: Optional[torch.FloatTensor] = None,
        base_encoder_hidden_states: Optional[torch.Tensor] = None,
        base_encoder_attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError(
                    "cache_position is a required argument when using StaticCache."
                )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not use_cache:
            past_seen_tokens = None

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_seen_tokens,
            output_attentions=True,
        )

        # embed positions
        hidden_states = inputs_embeds
        base_hidden_states = inputs_embeds
        source_hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_base_hidden_states = () if output_hidden_states else None
        all_source_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                all_base_hidden_states += (base_hidden_states,)
                all_source_hidden_states += (source_hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    source_hidden_states,
                    base_hidden_states,
                    causal_mask,
                    source_encoder_hidden_states,
                    source_encoder_attention_mask,
                    base_encoder_hidden_states,
                    base_encoder_attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    source_hidden_states=source_hidden_states,
                    base_hidden_states=base_hidden_states,
                    attention_mask=causal_mask,
                    source_encoder_hidden_states=source_encoder_hidden_states,
                    source_encoder_attention_mask=source_encoder_attention_mask,
                    base_encoder_hidden_states=base_encoder_hidden_states,
                    base_encoder_attention_mask=base_encoder_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states, base_hidden_states, source_hidden_states = (
                layer_outputs[0],
                layer_outputs[1],
                layer_outputs[2],
            )

            if use_cache:
                next_decoder_cache = layer_outputs[4 if output_attentions else 3]

            if output_attentions:
                all_self_attns += (layer_outputs[3],)

        hidden_states = self.norm(hidden_states)
        source_hidden_states = self.norm(source_hidden_states)
        base_hidden_states = self.norm(base_hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            all_source_hidden_states += (source_hidden_states,)
            all_base_hidden_states += (base_hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if isinstance(next_decoder_cache, Cache)
                else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )

        return HyperDASModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            last_source_hidden_state=source_hidden_states,
            last_base_hidden_state=base_hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            source_hidden_states=all_source_hidden_states,
            base_hidden_states=all_base_hidden_states,
            attentions=all_self_attns,
        )


class LlamaInterpretorHypernetwork(LlamaForCausalLM):
    _tied_weights_keys = []

    def __init__(self, config: LlamaInterpretorConfig):
        # This is key to avoid initializing a useless LlamaModel in the superclass.
        PreTrainedModel.__init__(self, config)

        self.lm_head = InterpretorUnembedCrossAttention(
            config=config,
            layer_idx=config.chop_editor_at_layer,
        ).to(dtype=config.torch_dtype)

        if config.target_hidden_size != config.hidden_size:
            self.target_adapter = nn.Linear(
                config.target_hidden_size,
                config.hidden_size,
                dtype=config.torch_dtype,
            )
        else:
            self.target_adapter = None

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        with torch.no_grad():
            if not config.initialize_from_scratch:
                # Initialize weights and apply final processing
                logger.debug("Calling post_init")
                self.post_init()
                logger.debug("Finished post_init")

                logger.debug("Initializing from pretrained model")

                self.model = LlamaModelWithCrossAttention.from_pretrained(
                    config.name_or_path, torch_dtype=config.torch_dtype
                )

                # prune layers and add cross attn heads
                self.model.layers = self.model.layers[
                    : self.config.chop_editor_at_layer
                ]
                cross_attn_layers = list(range(self.config.chop_editor_at_layer))

                for i, layer in enumerate(self.model.layers):
                    if i not in cross_attn_layers:
                        continue

                    # Create new layer with double cross attention
                    self.model.layers[i] = LlamaDecoderLayerWithDoubleCrossAttention(
                        self.config, i
                    ).to(dtype=self.config.torch_dtype)

                    # Copy original weights
                    def copy_weights(src_module, dst_module):
                        for name, param in src_module.named_parameters():
                            if hasattr(dst_module, name):
                                setattr(dst_module, name, nn.Parameter(param.data))

                    # Copy attention weights
                    copy_weights(layer.self_attn, self.model.layers[i].self_attn)
                    copy_weights(layer.self_attn, self.model.layers[i].cross_attn)

                    # Copy MLP weights
                    copy_weights(layer.mlp, self.model.layers[i].mlp)
                    copy_weights(layer.mlp, self.model.layers[i].cross_attn_mlp)

                    # Copy layer norms
                    copy_weights(
                        layer.input_layernorm, self.model.layers[i].input_layernorm
                    )
                    copy_weights(
                        layer.post_attention_layernorm,
                        self.model.layers[i].post_attention_layernorm,
                    )
                    self.model.layers[
                        i
                    ].cross_attn_input_layernorm.weight = nn.Parameter(
                        layer.input_layernorm.weight.data
                    )
                    self.model.layers[
                        i
                    ].post_cross_attn_layernorm.weight = nn.Parameter(
                        layer.post_attention_layernorm.weight.data
                    )

                logger.debug("Finished modifying layers with cross attention.")
            else:
                logger.debug("Initializing from scratch")
                self.model = LlamaModelWithCrossAttention._from_config(
                    config, torch_dtype=config.torch_dtype
                )

                logger.debug("Calling post_init")
                self.post_init()
                logger.debug("Finished post_init")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        base_encoder_hidden_states: Optional[torch.Tensor] = None,
        base_encoder_attention_mask: Optional[torch.FloatTensor] = None,
        source_encoder_hidden_states: Optional[torch.Tensor] = None,
        source_encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # set device for input_ids to cuda ?
        # input_ids = input_ids.to(self.lm_head.weight.device)
        if (
            attention_mask is not None
            and position_ids is None
            and self.config.compute_position_ids
        ):
            position_ids = attention_mask.cumsum(-1)

        if self.target_adapter is not None:
            inputs_embeds = self.target_adapter(inputs_embeds)

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            base_encoder_hidden_states=base_encoder_hidden_states,
            base_encoder_attention_mask=base_encoder_attention_mask,
            source_encoder_hidden_states=source_encoder_hidden_states,
            source_encoder_attention_mask=source_encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states, base_hidden_states, source_hidden_states = (
            transformer_outputs[0],
            transformer_outputs[1],
            transformer_outputs[2],
        )

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
            base_hidden_states = base_hidden_states.to(self.lm_head.weight.device)
            source_hidden_states = source_hidden_states.to(self.lm_head.weight.device)

        if (
            source_hidden_states is not None
            and source_encoder_hidden_states is not None
            and source_encoder_attention_mask is not None
        ):
            source_attn_weight = self.lm_head(
                source_hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=source_encoder_hidden_states,
                encoder_attention_mask=source_encoder_attention_mask,
                output_attentions=output_attentions,
            )
        else:
            source_attn_weight = None

        base_attn_weight = self.lm_head(
            base_hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=base_encoder_hidden_states,
            encoder_attention_mask=base_encoder_attention_mask,
            output_attentions=output_attentions,
        )

        return hidden_states, source_attn_weight, base_attn_weight


class SimpleMLPHypernetwork(nn.Module):
    def __init__(self, config: LlamaInterpretorConfig):
        super().__init__()
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(
                config.hidden_size, config.intermediate_size, dtype=config.torch_dtype
            ),
            nn.LayerNorm(config.intermediate_size, eps=1e-5),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            # nn.Linear(
            #     config.intermediate_size, config.hidden_size, dtype=config.torch_dtype
            # ),
            # nn.SiLU(),
            # nn.Dropout(config.dropout),
        )

    def forward(self, inputs_embeds: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.mlp(inputs_embeds)


class LlamaInterpretorForReFTGeneration(nn.Module):
    def __init__(
        self,
        config: LlamaInterpretorConfig,
        target_model_name_or_path,
        subspace_module=None,
        compute_metrics=False,
        device="cuda",
    ):
        super().__init__()

        self.config = config
        target_model_cfg = AutoConfig.from_pretrained(target_model_name_or_path)
        self.config.num_target_model_layers = target_model_cfg.num_hidden_layers

        if config.hypernetwork_type == "lm":
            self.hypernetwork = LlamaInterpretorHypernetwork(config)
        else:
            self.hypernetwork = SimpleMLPHypernetwork(config)

        logger.debug("Initializing ReFT generator")
        self.reft_generator = ReFTHypernetwork(
            hidden_size=config.hidden_size,
            target_hidden_size=config.target_hidden_size,
            num_target_layers=config.num_target_layers,
            num_target_positions=config.num_target_positions,
            intermediate_size=config.intermediate_size,
            rank=config.das_dimension,
            rms_norm_eps=config.rms_norm_eps,
            dropout=config.attention_dropout,
            dtype=config.torch_dtype,
        )
        self.hypernetwork_adapter = nn.Linear(
            config.target_hidden_size,
            config.hidden_size,
            dtype=config.torch_dtype,
        )

        logger.debug("Finished initializing ReFT generator")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name_or_path,
            torch_dtype=config.torch_dtype,
            device_map={"": device},
        )
        logger.debug("Finished loading target model")

        self.use_das_intervention = subspace_module is not None
        self.das_selective_subspace = subspace_module in [
            "SelectiveLoReFT",
            "SteeringVec",
        ]

        if self.use_das_intervention:
            if subspace_module == "LoReFT":
                self.intervention_module = LoreftIntervention(
                    embed_dim=self.target_model.config.hidden_size,
                    low_rank_dimension=config.das_dimension,
                    dtype=config.torch_dtype,
                    compute_metrics=compute_metrics,
                )
            elif subspace_module == "SelectiveLoReFT":
                self.intervention_module = SelectiveLoreftIntervention(
                    embed_dim=self.target_model.config.hidden_size,
                    low_rank_dimension=config.das_dimension,
                    dtype=config.torch_dtype,
                    compute_metrics=compute_metrics,
                )
            elif subspace_module == "SteeringVec":
                self.intervention_module = SteeringVecLoreftIntervention(
                    embed_dim=self.target_model.config.hidden_size,
                    low_rank_dimension=config.das_dimension,
                    dtype=config.torch_dtype,
                    compute_metrics=compute_metrics,
                )
            else:
                raise ValueError("Invalid subspace module")

        logger.debug("Finished initializing DAS module")

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
        self.original_layer_forward = {}

        self._temp_global_cache = SimpleTensorCache()

    def train(self: T_Reft, mode: bool = True) -> T_Reft:
        return self.hypernetwork.train(mode)

    def eval(self: T_Reft) -> T_Reft:
        return self.hypernetwork.eval()

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        """Only load weights for the trainable hypernetwork."""
        self.hypernetwork.load_state_dict(state_dict, strict=strict, assign=assign)

    @torch.no_grad()
    def _run_target_model_for_encoded_hidden_states(
        self,
        target_input_ids: torch.LongTensor,
        target_attention_mask: torch.BoolTensor,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, ...]:
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

    def generate(self, *args, **kwargs) -> ModelOutput | torch.LongTensor:
        do_intervention = kwargs.pop("do_intervention", True)

        # Prompt length is used later to determine if we are in generation mode
        self._temp_global_cache["prompt_lengths"] = [
            len(x) for x in kwargs["target_input_ids"]
        ]
        # prune kwargs for generation
        target_kwargs = {}
        for kw in kwargs.keys():
            if kw.startswith("target_"):
                target_kwargs[kw.replace("target_", "")] = kwargs[kw]
            elif all(not kw.startswith(x) for x in ["base", "editor", "intervention"]):
                target_kwargs[kw] = kwargs[kw]

        if do_intervention:
            try:
                self.prepare_for_generation(**kwargs)
                outputs = self.target_model.generate(*args, **target_kwargs)
            finally:
                self.cleanup_after_generation()
        else:
            outputs = self.target_model.generate(*args, **target_kwargs)
        return outputs

    def _create_generate_intervention_hook(self, layer_idx, **kwargs):
        def intervention_forward(module, input, target_kwargs, output):
            # Only apply intervention during prefill phase
            if (
                self._temp_global_cache["prompt_lengths"][0]
                == target_kwargs.get("past_key_value")._seen_tokens
            ):
                hidden_states = output[0]
                kwargs["generation"] = True
                kwargs["editor_attention_mask"] = (
                    kwargs["editor_input_ids"] != self.config.eos_token_id
                )
                _ = self.forward(**kwargs)
                self._temp_global_cache["current_layer_rel_idx"] = layer_idx
                modified_states, _ = self._apply_intervention(
                    hidden_states,
                    intervention_positions=kwargs.get("intervention_positions"),
                )
                # Manually handle cache clearing
                self._temp_global_cache.clear()
                output = (modified_states,) + output[1:]

            return output

        return intervention_forward

    def prepare_for_generation(self, **kwargs):
        """Setup hooks for generation"""
        self.intervention_hooks = []
        for layer_idx, layer in enumerate(self.target_model.model.layers):
            intervention_layers = (
                [self.config.intervention_layer]
                if isinstance(self.config.intervention_layer, int)
                else self.config.intervention_layer
            )
            if layer_idx in intervention_layers:
                # NOTE: important to allow kwargs else we can't check if in generation mode all kwargs ignored!
                current_layer_rel_idx = intervention_layers.index(layer_idx)
                # Use rel idx because of cache positioning
                hook = layer.register_forward_hook(
                    self._create_generate_intervention_hook(
                        current_layer_rel_idx, **kwargs
                    ),
                    with_kwargs=True,
                )
                self.intervention_hooks.append(hook)

    def cleanup_after_generation(self):
        """Remove hooks after generation"""
        for hook in self.intervention_hooks:
            hook.remove()
        self.intervention_hooks.clear()

    def _apply_intervention(
        self,
        hidden_states,
        intervention_positions,
        metrics=None,
    ):
        """Common intervention logic used by both forward and generation"""
        if not hasattr(self, "intervention_module"):
            raise ValueError("intervention module not initialized")

        current_layer_rel_idx = self._temp_global_cache["current_layer_rel_idx"]

        # During generation, we'll reuse the cached hypernet states
        rotation_matrix = self._temp_global_cache["rotation_matrix"][
            :, current_layer_rel_idx
        ]
        weight_matrix = self._temp_global_cache["weight_matrix"][
            :, current_layer_rel_idx
        ]

        # Apply intervention, assuming LoReFT impl
        intervention_output = self.intervention_module(
            hidden_states,
            intervention_positions=intervention_positions,
            batch_rotation=rotation_matrix,
            batch_weights=weight_matrix,
        )
        mixed_output = intervention_output.mixed_output
        metrics = intervention_output.metrics

        return mixed_output, metrics

    def forward(
        self,
        editor_input_ids: Optional[torch.Tensor] = None,
        editor_attention_mask: Optional[torch.Tensor] = None,
        base_input_ids: Optional[torch.Tensor] = None,
        base_attention_mask: Optional[torch.Tensor] = None,
        base_intervention_mask: Optional[torch.Tensor] = None,
        base_hidden_states: Optional[torch.Tensor] = None,
        base_position_ids: Optional[torch.Tensor] = None,
        target_input_ids: Optional[torch.Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        target_position_ids: Optional[torch.Tensor] = None,
        intervention_layers: Optional[List[int]] = None,
        intervention_positions: Optional[torch.LongTensor] = None,
        run_weight_generation: bool = True,
        generation: bool = False,
        **kwargs,
    ) -> InterpretorModelOutput:
        if intervention_layers is None:
            intervention_layers = self.config.intervention_layer

        if intervention_positions is None:
            intervention_positions = (
                torch.arange(
                    self.config.num_target_positions, device=editor_input_ids.device
                )
                .unsqueeze(0)
                .expand(editor_input_ids.shape[0], -1)
            )

        if isinstance(intervention_layers, int):
            intervention_layers = (
                torch.tensor(
                    [intervention_layers],
                    dtype=torch.long,
                    device=editor_input_ids.device,
                )
                .unsqueeze(0)
                .expand(editor_input_ids.shape[0], -1)
            )
        elif isinstance(intervention_layers, list):
            intervention_layers = (
                torch.tensor(
                    intervention_layers,
                    dtype=torch.long,
                    device=editor_input_ids.device,
                )
                .unsqueeze(0)
                .expand(editor_input_ids.shape[0], -1)
            )

        if base_position_ids is None:
            # -1 for all the padding tokens and start from 0 for the rest
            base_position_ids = (
                torch.cumsum(base_attention_mask, dim=1) * base_attention_mask - 1
            )

        if target_position_ids is None:
            # -1 for all the padding tokens and start from 0 for the rest
            target_position_ids = (
                torch.cumsum(target_attention_mask, dim=1) * target_attention_mask - 1
            )

        # Run target model for encoded hidden states
        if base_hidden_states is None and self.config.hypernetwork_type == "lm":
            base_hidden_states = torch.stack(
                self._run_target_model_for_encoded_hidden_states(
                    base_input_ids, base_attention_mask, base_position_ids
                ),  # seems to break while we are passing thru batch_size=1; the last (12th =) has different dimensions
                dim=2,
            )

        if base_intervention_mask is None:
            if base_attention_mask is not None:
                base_intervention_mask = base_attention_mask.clone()
            else:
                base_intervention_mask = torch.ones_like(base_input_ids)

        # dimensions of target_hidden_states:
        # batch_size, token_sequence_length, num_layers = 13, resid_width = 768
        # Normalize along the last dimension
        if base_hidden_states is not None:
            base_normalization_factors = base_hidden_states.norm(dim=-1, keepdim=True)
            base_hidden_states = base_hidden_states / base_normalization_factors

        if (
            base_intervention_mask is not None
            and (base_intervention_mask.sum(dim=-1) == 0).any()
        ):
            tokenizer = AutoTokenizer.from_pretrained("/nlp/scr/sjd24/llama3-8b")
            print("base_intervention_mask has zero row")
            zero_row_idx = (base_intervention_mask.sum(dim=-1) == 0).nonzero(
                as_tuple=True
            )[0]
            print(base_input_ids[zero_row_idx], zero_row_idx)
            print("Full sentence: ", tokenizer.decode(base_input_ids[zero_row_idx][0]))
            print("Attention mask: ", base_attention_mask[zero_row_idx][0])
            print(base_intervention_mask[zero_row_idx][0])
            raise

        if base_hidden_states is not None:
            n_layer = base_hidden_states.shape[2]
            collapsed_base_hidden_states = base_hidden_states.reshape(
                base_hidden_states.shape[0],
                base_hidden_states.shape[1] * base_hidden_states.shape[2],
                base_hidden_states.shape[3],
            )
            collapsed_base_attention_mask = base_intervention_mask.unsqueeze(-1).repeat(
                1, 1, n_layer
            )
            collapsed_base_attention_mask = collapsed_base_attention_mask.reshape(
                base_intervention_mask.shape[0],
                base_intervention_mask.shape[1] * n_layer,
            )

        inputs_embeds = self.target_model.model.embed_tokens(editor_input_ids)
        editor_input_ids = None

        # Project target states to compatibility with hypernet hidden size
        if (
            self.config.hypernetwork_type == "lm"
            and self.config.target_hidden_size != self.config.hidden_size
        ):
            collapsed_base_hidden_states = self.hypernetwork_adapter(
                collapsed_base_hidden_states
            )

        layer_hidden_states = []
        intervention_layer_list = intervention_layers.cpu().tolist()
        for layer_idx in intervention_layer_list:
            # Patch intervention layer each time
            if self.config.hypernetwork_type == "lm":
                self.hypernetwork.lm_head.intervention_layer = layer_idx
                hypernet_hidden_states = self.hypernetwork(
                    input_ids=editor_input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=editor_attention_mask,
                    base_encoder_hidden_states=collapsed_base_hidden_states,
                    base_encoder_attention_mask=collapsed_base_attention_mask,
                    use_cache=False,
                )[0]
            else:
                hypernet_hidden_states = self.hypernetwork(inputs_embeds=inputs_embeds)

            layer_hidden_states.append(hypernet_hidden_states)

        # Stack along layer axis -> (B, L, P, H)
        hypernet_hidden_states = torch.stack(layer_hidden_states, dim=1)

        if run_weight_generation:
            # run hypernetwork to generate ReFT weights
            # each of shape (B, L, P, H, R)
            rotation_matrix, weight_matrix = self.reft_generator(
                hypernet_hidden_states, intervention_layers, intervention_positions
            )
            self._temp_global_cache["rotation_matrix"] = rotation_matrix
            self._temp_global_cache["weight_matrix"] = weight_matrix
            self._temp_global_cache["hypernet_hidden_states"] = hypernet_hidden_states

        metrics = {}

        # Flag preventing redundant prefill
        if not generation:
            # Run target model with edit vectors.
            # This adds the edit vectors to the given hidden state at the specified batch index, position, and layer
            def representation_swap(module, input, output, current_layer_rel_idx=0):
                nonlocal metrics

                self._temp_global_cache["current_layer_rel_idx"] = current_layer_rel_idx
                mixed_output, metrics = self._apply_intervention(
                    output[0], intervention_positions=intervention_positions
                )

                output[0][:] += mixed_output

            # Now editing the target model
            if intervention_layers is None:
                raise ValueError("Cannot intervene at layer 0")
            else:
                hooks = [
                    (
                        self.target_model.model.layers[layer_idx - 1],
                        partial(representation_swap, current_layer_rel_idx=rel_idx),
                    )
                    for rel_idx, layer_idx in enumerate(intervention_layer_list)
                ]

            with add_fwd_hooks(hooks):
                # THIS IS THE LINE WHERE THE MODEL IS CALLED (AND THE EDITOR IS CALLED AT
                # THE END OF `layer` AS A SIDE EFFECT)
                target_result = self.target_model(
                    input_ids=target_input_ids,
                    attention_mask=target_attention_mask,
                    position_ids=target_position_ids,
                    output_hidden_states=True,
                )

            logits = target_result.logits

            # Clear cache of outdated entries in next forward pass to save memory
            # only during training in inference we manually handle
            self._temp_global_cache.clear()

            output = InterpretorModelOutput(
                logits=logits,
                target_hidden_states=target_result.hidden_states,
            )
            output.metrics = metrics
        else:
            output = None

        return output


class LlamaInterpretorWithLearnedSource(nn.Module):
    def __init__(
        self,
        config: LlamaInterpretorConfig,
        target_model_name_or_path,
        subspace_module=None,
        compute_metrics=False,
        device="cuda",
    ):
        # TODO(sid) implement target tokenizer functionality and inputs to forward pass
        super().__init__()

        self.config = config
        target_model_cfg = AutoConfig.from_pretrained(target_model_name_or_path)
        self.config.num_target_model_layers = target_model_cfg.num_hidden_layers
        self.hypernetwork = LlamaInterpretorHypernetwork(config)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name_or_path,
            torch_dtype=config.torch_dtype,
            device_map={"": device},
        )

        self.bidding_threshold = 0.1

        self.use_das_intervention = subspace_module is not None
        self.das_selective_subspace = subspace_module in [
            "SelectiveLoReFT",
            "SteeringVec",
        ]

        if self.use_das_intervention:
            if subspace_module == "LoReFT":
                self.intervention_module = LoreftIntervention(
                    embed_dim=self.target_model.config.hidden_size,
                    low_rank_dimension=config.das_dimension,
                    dtype=config.torch_dtype,
                    compute_metrics=compute_metrics,
                )
            elif subspace_module == "SelectiveLoReFT":
                self.intervention_module = SelectiveLoreftIntervention(
                    embed_dim=self.target_model.config.hidden_size,
                    low_rank_dimension=config.das_dimension,
                    dtype=config.torch_dtype,
                    compute_metrics=compute_metrics,
                )
            elif subspace_module == "SteeringVec":
                self.intervention_module = SteeringVecLoreftIntervention(
                    embed_dim=self.target_model.config.hidden_size,
                    low_rank_dimension=config.das_dimension,
                    dtype=config.torch_dtype,
                    compute_metrics=compute_metrics,
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

    def train(self: T_Learned, mode: bool = True) -> T_Learned:
        return self.hypernetwork.train(mode)

    def eval(self: T_Learned) -> T_Learned:
        return self.hypernetwork.eval()

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        """Only load weights for the trainable hypernetwork."""
        self.hypernetwork.load_state_dict(state_dict, strict=strict, assign=assign)

    @torch.no_grad()
    def _run_target_model_for_encoded_hidden_states(
        self,
        target_input_ids: torch.LongTensor,
        target_attention_mask: torch.BoolTensor,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, ...]:
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
        editor_input_ids: Optional[torch.Tensor] = None,
        editor_attention_mask: Optional[torch.Tensor] = None,
        base_input_ids: Optional[torch.Tensor] = None,
        base_attention_mask: Optional[torch.Tensor] = None,
        base_intervention_mask: Optional[torch.Tensor] = None,
        base_hidden_states: Optional[torch.Tensor] = None,
        base_position_ids: Optional[torch.Tensor] = None,
        intervention_layer: Optional[int] = None,
        base_intervention_weight: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> InterpretorModelOutput:
        if intervention_layer is None:
            intervention_layer = self.config.intervention_layer

        if base_position_ids is None:
            # -1 for all the padding tokens and start from 0 for the rest
            base_position_ids = (
                torch.cumsum(base_attention_mask, dim=1) * base_attention_mask - 1
            )

        # Run target model for encoded hidden states
        if base_hidden_states is None:
            base_hidden_states = torch.stack(
                self._run_target_model_for_encoded_hidden_states(
                    base_input_ids, base_attention_mask, base_position_ids
                ),  # seems to break while we are passing thru batch_size=1; the last (12th =) has different dimensions
                dim=2,
            )

        if base_intervention_mask is None:
            if base_attention_mask is not None:
                base_intervention_mask = base_attention_mask.clone()
            else:
                base_intervention_mask = torch.ones_like(base_input_ids)

        # dimensions of target_hidden_states:
        # batch_size, token_sequence_length, num_layers = 13, resid_width = 768
        # Normalize along the last dimension
        base_normalization_factors = base_hidden_states.norm(dim=-1, keepdim=True)
        base_hidden_states = base_hidden_states / base_normalization_factors

        if (base_intervention_mask.sum(dim=-1) == 0).any():
            tokenizer = AutoTokenizer.from_pretrained("/nlp/scr/sjd24/llama3-8b")
            print("base_intervention_mask has zero row")
            zero_row_idx = (base_intervention_mask.sum(dim=-1) == 0).nonzero(
                as_tuple=True
            )[0]
            print(base_input_ids[zero_row_idx], zero_row_idx)
            print("Full sentence: ", tokenizer.decode(base_input_ids[zero_row_idx][0]))
            print("Attention mask: ", base_attention_mask[zero_row_idx][0])
            print(base_intervention_mask[zero_row_idx][0])
            raise

        n_layer = base_hidden_states.shape[2]

        collapsed_base_hidden_states = base_hidden_states.reshape(
            base_hidden_states.shape[0],
            base_hidden_states.shape[1] * base_hidden_states.shape[2],
            base_hidden_states.shape[3],
        )

        collapsed_base_attention_mask = base_intervention_mask.unsqueeze(-1).repeat(
            1, 1, n_layer
        )
        collapsed_base_attention_mask = collapsed_base_attention_mask.reshape(
            base_intervention_mask.shape[0],
            base_intervention_mask.shape[1] * n_layer,
        )

        inputs_embeds = self.target_model.model.embed_tokens(editor_input_ids)
        editor_input_ids = None

        interpretor_output = self.hypernetwork(
            input_ids=editor_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=editor_attention_mask,
            base_encoder_hidden_states=collapsed_base_hidden_states,
            base_encoder_attention_mask=collapsed_base_attention_mask,
            use_cache=False,
        )

        if base_intervention_weight is None:
            base_intervention_weight = interpretor_output[1]
        else:
            base_intervention_weight = base_intervention_weight.to(
                interpretor_output[1].dtype
            )

        hypernet_hidden_states = interpretor_output[0]

        # print(source_intervention[0])
        metrics = {}

        # Run target model with edit vectors.
        # This adds the edit vectors to the given hidden state at the specified batch index, position, and layer
        def representation_swap(module, input, output):
            base_hidden_states = output[0].clone()

            # print(base_hidden_states[0, 18, :])
            # print(source_hidden_states[0, 6, :])

            nonlocal metrics

            if self.use_das_intervention:
                hidden_dim = base_hidden_states.shape[-1]
                base_hidden_states_weight = base_intervention_weight.unsqueeze(
                    -1
                ).repeat(1, 1, hidden_dim)

                if self.das_selective_subspace:
                    intervention_output = self.intervention_module(
                        base_hidden_states, hidden_states=hypernet_hidden_states
                    )
                    mixed_output = intervention_output.mixed_output
                    mixed_output = mixed_output * base_hidden_states_weight
                else:
                    intervention_output = self.intervention_module(base_hidden_states)
                    mixed_output = intervention_output.mixed_output
                    mixed_output = mixed_output * base_hidden_states_weight

                if isinstance(mixed_output, InterventionModuleOutput):
                    metrics = intervention_output.metrics

                output[0][:] += mixed_output
            else:
                raise NotImplementedError("Only DAS is implemented")

        # Now editing the target model
        if intervention_layer == 0:
            raise ValueError("Cannot intervene at layer 0")
        else:
            hooks = [
                (
                    self.target_model.model.layers[intervention_layer - 1],
                    representation_swap,
                )
            ]

        with add_fwd_hooks(hooks):
            # THIS IS THE LINE WHERE THE MODEL IS CALLED (AND THE EDITOR IS CALLED AT
            # THE END OF `layer` AS A SIDE EFFECT)
            target_result = self.target_model(
                input_ids=base_input_ids,
                attention_mask=base_attention_mask,
                position_ids=base_position_ids,
                output_hidden_states=True,
            )

        logits = target_result.logits

        output = InterpretorModelOutputWithLearnedSource(
            logits=logits,
            base_intervention_weight=base_intervention_weight,
            target_hidden_states=target_result.hidden_states,
        )
        output.metrics = metrics

        return output


class LlamaInterpretorWithBaseSource(nn.Module):
    def __init__(
        self,
        config: LlamaInterpretorConfig,
        target_model_name_or_path,
        subspace_module=None,
        compute_metrics=False,
        device="cuda",
    ):
        super().__init__()

        self.config = config
        self.hypernetwork = LlamaInterpretorHypernetwork(config)
        target_model_cfg = AutoConfig.from_pretrained(target_model_name_or_path)
        self.config.num_target_model_layers = target_model_cfg.num_hidden_layers
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name_or_path,
            torch_dtype=config.torch_dtype,
            device_map={"": device},
        )

        self.bidding_threshold = 0.1

        self.use_das_intervention = subspace_module is not None
        self.das_selective_subspace = subspace_module in [
            "ReflectSelect",
            "MaskSelect",
            "QuasiProjective",
        ]

        if self.use_das_intervention:
            if subspace_module == "BoundlessDAS":
                self.intervention_module = BoundlessRotatedSpaceIntervention(
                    embed_dim=self.target_model.config.hidden_size,
                    torch_dtype=config.torch_dtype,
                    compute_metrics=compute_metrics,
                )
            elif subspace_module == "DAS":
                self.intervention_module = LowRankRotatedSpaceIntervention(
                    embed_dim=self.target_model.config.hidden_size,
                    low_rank_dimension=config.das_dimension,
                    torch_dtype=config.torch_dtype,
                    compute_metrics=compute_metrics,
                )
            elif subspace_module == "MaskSelect":
                self.intervention_module = SelectiveLowRankRotatedSpaceIntervention(
                    embed_dim=self.target_model.config.hidden_size,
                    low_rank_dimension=config.das_dimension,
                    torch_dtype=config.torch_dtype,
                    compute_metrics=compute_metrics,
                )
            elif subspace_module == "ReflectSelect":
                self.intervention_module = ReflectiveLowRankRotatedSpaceIntervention(
                    embed_dim=self.target_model.config.hidden_size,
                    low_rank_dimension=config.das_dimension,
                    torch_dtype=config.torch_dtype,
                    compute_metrics=compute_metrics,
                )
            elif subspace_module == "QuasiProjective":
                self.intervention_module = QuasiProjectiveIntervention(
                    embed_dim=self.target_model.config.hidden_size,
                    dict_size=config.dict_size,
                    scoring_dimension=config.scoring_dimension,
                    top_k_parameter=config.das_dimension,
                    lambda_parameter=config.lambda_parameter,
                    importance_power=config.importance_power,
                    epsilon=config.epsilon,
                    return_penalty=config.return_penalty,
                    ridge_parameterization=config.ridge_parameterization,
                    torch_dtype=config.torch_dtype,
                    compute_metrics=compute_metrics,
                    orthogonal_init=config.orthogonal_init,
                    selection_mechanism=config.selection_mechanism,
                    hat_matrix=config.hat_matrix,
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

    def train(self: T_Regular, mode: bool = True) -> T_Regular:
        return self.hypernetwork.train(mode)

    def eval(self: T_Regular) -> T_Regular:
        return self.hypernetwork.eval()

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        """Only load weights for the trainable hypernetwork."""
        self.hypernetwork.load_state_dict(state_dict, strict=strict, assign=assign)

    @torch.no_grad()
    def _run_target_model_for_encoded_hidden_states(
        self,
        target_input_ids: torch.LongTensor,
        target_attention_mask: torch.FloatTensor,
        position_ids: Optional[torch.LongTensor] = None,
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
        editor_input_ids: Optional[torch.LongTensor] = None,
        editor_attention_mask: Optional[torch.FloatTensor] = None,
        base_input_ids: Optional[torch.LongTensor] = None,
        base_attention_mask: Optional[torch.FloatTensor] = None,
        base_intervention_mask: Optional[torch.FloatTensor] = None,
        source_input_ids: Optional[torch.LongTensor] = None,
        source_attention_mask: Optional[torch.FloatTensor] = None,
        source_intervention_mask: Optional[torch.FloatTensor] = None,
        base_hidden_states: Optional[torch.FloatTensor] = None,
        base_position_ids: Optional[torch.LongTensor] = None,
        source_hidden_states: Optional[torch.FloatTensor] = None,
        source_position_ids: Optional[torch.LongTensor] = None,
        intervention_layer: Optional[int] = None,
        output_intervention_weight: bool = True,
        source_intervention_weight: Optional[torch.FloatTensor] = None,
        base_intervention_weight: Optional[torch.FloatTensor] = None,
    ) -> InterpretorModelOutput:
        if intervention_layer is None:
            intervention_layer = self.config.intervention_layer

        if base_position_ids is None:
            # -1 for all the padding tokens and start from 0 for the rest
            base_position_ids = (
                torch.cumsum(base_attention_mask, dim=1) * base_attention_mask - 1
            )

        if source_position_ids is None:
            # -1 for all the padding tokens and start from 0 for the rest
            source_position_ids = (
                torch.cumsum(source_attention_mask, dim=1) * source_attention_mask - 1
            )

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

        if (source_intervention_mask.sum(dim=-1) == 0).any():
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
            zero_row_idx = (source_intervention_mask.sum(dim=-1) == 0).nonzero(
                as_tuple=True
            )[0]
            warning_msg = (
                "source_intervention_mask has zero row\n"
                f"IDs: {source_input_ids[zero_row_idx]}, Indices: {zero_row_idx}\n"
                f"Full sentence: {tokenizer.decode(source_input_ids[zero_row_idx][0])}\n"
                f"Attention mask: {source_attention_mask[zero_row_idx][0]}\n"
                f"Intervention mask: {source_intervention_mask[zero_row_idx][0]}"
            )
            warnings.warn(warning_msg)

        if (base_intervention_mask.sum(dim=-1) == 0).any():
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
            zero_row_idx = (base_intervention_mask.sum(dim=-1) == 0).nonzero(
                as_tuple=True
            )[0]
            warning_msg = (
                "base_intervention_mask has zero row\n"
                f"IDs: {base_input_ids[zero_row_idx]}, Indices: {zero_row_idx}\n"
                f"Full sentence: {tokenizer.decode(base_input_ids[zero_row_idx][0])}\n"
                f"Attention mask: {base_attention_mask[zero_row_idx][0]}\n"
                f"Intervention mask: {base_intervention_mask[zero_row_idx][0]}"
            )
            warnings.warn(warning_msg)

        n_layer = base_hidden_states.shape[2]

        collapsed_base_hidden_states = base_hidden_states.reshape(
            base_hidden_states.shape[0],
            base_hidden_states.shape[1] * base_hidden_states.shape[2],
            base_hidden_states.shape[3],
        )

        collapsed_base_attention_mask = base_intervention_mask.unsqueeze(-1).repeat(
            1, 1, n_layer
        )
        collapsed_base_attention_mask = collapsed_base_attention_mask.reshape(
            base_intervention_mask.shape[0],
            base_intervention_mask.shape[1] * n_layer,
        )

        collapsed_source_hidden_states = source_hidden_states.reshape(
            source_hidden_states.shape[0],
            source_hidden_states.shape[1] * source_hidden_states.shape[2],
            source_hidden_states.shape[3],
        )

        collapsed_source_attention_mask = source_intervention_mask.unsqueeze(-1).repeat(
            1, 1, n_layer
        )
        collapsed_source_attention_mask = collapsed_source_attention_mask.reshape(
            source_intervention_mask.shape[0],
            source_intervention_mask.shape[1] * n_layer,
        )

        interpretor_output = self.hypernetwork(
            input_ids=editor_input_ids,
            attention_mask=editor_attention_mask,
            base_encoder_hidden_states=collapsed_base_hidden_states,
            base_encoder_attention_mask=collapsed_base_attention_mask,
            source_encoder_hidden_states=collapsed_source_hidden_states,
            source_encoder_attention_mask=collapsed_source_attention_mask,
            use_cache=False,
        )

        # Multiply the outputs by normalization factors
        hypernet_hidden_states, source_intervention_weight, base_intervention_weight = (
            interpretor_output
        )

        source_output = self.target_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            output_hidden_states=True,
        )

        source_hidden_states = source_output.hidden_states[intervention_layer]

        hidden_dim = source_hidden_states.shape[-1]

        source_intervention = (
            source_intervention_weight.unsqueeze(-1).repeat(1, 1, hidden_dim)
            * source_hidden_states
        )

        source_intervention = source_intervention.sum(dim=1)

        # Run target model with edit vectors.
        # This adds the edit vectors to the given hidden state at the specified batch index, position, and layer
        def representation_swap(module, input, output):
            base_hidden_states = output[0].clone()
            batch_size = base_hidden_states.shape[0]

            if self.use_das_intervention:
                base_remaining_weight = 1 - base_intervention_weight
                source_intervention_hidden_states = source_intervention.unsqueeze(
                    1
                ).repeat(1, base_hidden_states.shape[1], 1)
                source_intervention_hidden_states = (
                    base_intervention_weight.unsqueeze(-1).repeat(1, 1, hidden_dim)
                    * source_intervention_hidden_states
                    + base_remaining_weight.unsqueeze(-1).repeat(1, 1, hidden_dim)
                    * base_hidden_states
                )

                if self.das_selective_subspace:
                    mixed_output = self.intervention_module(
                        base_hidden_states,
                        source_intervention_hidden_states,
                        hypernet_hidden_states,
                    )
                else:
                    mixed_output = self.intervention_module(
                        base_hidden_states,
                        source_intervention_hidden_states,
                        batch_size,
                    )

                if isinstance(mixed_output, InterventionModuleOutput):
                    mixed_output = mixed_output.mixed_output

                output[0][:] += mixed_output - base_hidden_states
            else:
                output[0][:] += source_intervention_hidden_states - base_hidden_states

        # Now editing the target model
        if intervention_layer == 0:
            raise ValueError("Cannot intervene at layer 0")
        else:
            hooks = [
                (
                    self.target_model.model.layers[intervention_layer - 1],
                    representation_swap,
                )
            ]

        with add_fwd_hooks(hooks):
            # THIS IS THE LINE WHERE THE MODEL IS CALLED (AND THE EDITOR IS CALLED AT
            # THE END OF `layer` AS A SIDE EFFECT)
            target_result = self.target_model(
                input_ids=base_input_ids,
                attention_mask=base_attention_mask,
                position_ids=base_position_ids,
                output_hidden_states=True,
            )

        logits = target_result.logits

        output = InterpretorModelOutput(
            logits=logits,
            source_intervention_weight=source_intervention_weight,
            base_intervention_weight=base_intervention_weight,
            target_hidden_states=target_result.hidden_states,
        )

        return output
