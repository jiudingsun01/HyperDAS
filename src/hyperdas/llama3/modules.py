from __future__ import annotations

import os
from typing import Any, List, Mapping, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from logger import get_logger

from ..das_utils import (
    BoundlessRotatedSpaceIntervention,
    LowRankRotatedSpaceIntervention,
    QuasiProjectiveIntervention,
    ReflectiveLowRankRotatedSpaceIntervention,
    SelectiveLowRankRotatedSpaceIntervention,
)
from ..utils import (
    InterpretorModelOutput,
    add_fwd_hooks,
    assign_layer_indices,
)
from .layers import (
    InterpretorUnembedCrossAttention,
    LlamaDecoderLayerWithDoubleCrossAttention,
)

logger = get_logger(__name__)


T = TypeVar("T", bound="LlamaInterpretor")


class LlamaInterpretorConfig(LlamaConfig):
    boundless_das: bool = False
    torch_dtype = torch.bfloat16
    chop_editor_at_layer: int = -1
    num_editing_heads: int = 32
    compute_position_ids: bool = True
    intervention_layer: int = 24
    initialize_from_scratch: bool = False
    ablate_base_token_attention: bool = False
    ablate_source_token_attention: bool = False
    break_asymmetric: bool = False
    # For projective ridge regression
    top_k_parameter: int = 128
    lambda_parameter: int = 10
    importance_power: int = -2
    epsilon: float = 1e-6


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
        source_hidden_states: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.FloatTensor] = None,
        base_hidden_states: Optional[torch.Tensor] = None,
        base_attention_mask: Optional[torch.FloatTensor] = None,
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

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_seen_tokens
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if isinstance(decoder_layer, LlamaDecoderLayerWithDoubleCrossAttention):
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        source_hidden_states,
                        source_attention_mask,
                        base_hidden_states,
                        base_attention_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                    )
                else:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                    )
            else:
                if isinstance(decoder_layer, LlamaDecoderLayerWithDoubleCrossAttention):
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        source_hidden_states=source_hidden_states,
                        source_attention_mask=source_attention_mask,
                        base_hidden_states=base_hidden_states,
                        base_attention_mask=base_attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        source_hidden_states=source_hidden_states,
                        source_attention_mask=source_attention_mask,
                        base_hidden_states=base_hidden_states,
                        base_attention_mask=base_attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                    )
                    """
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                    )
                    """
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

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
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaInterpretorHypernetwork(LlamaForCausalLM):
    _tied_weights_keys = []

    def __init__(self, config: LlamaInterpretorConfig):
        """Key change do not call post_init to prevent expensive weight initialization every time."""
        super(LlamaPreTrainedModel, self).__init__(config)
        self.vocab_size = config.vocab_size
        self.model = None
        self.lm_head = None
        self.cache_path = None

    def initialize(self):
        logger.debug("Initializing LlamaInterpretorHypernetwork")
        self._initialize_from_scratch()

    def _initialize_from_scratch(self):
        # Define a path for the cached model
        cache_dir = os.environ.get("MODEL_CACHE_DIR", "./assets/models")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(
            cache_dir,
            f"llama_interpretor_hypernetwork_{self.config.name_or_path.replace('/', '_')}.pt",
        )

        logger.debug("Loading pretrained model for LlamaModelWithCrossAttention")
        self.model = LlamaModelWithCrossAttention.from_pretrained(
            self.config.name_or_path, torch_dtype=self.config.torch_dtype
        )
        self.lm_head = InterpretorUnembedCrossAttention(
            config=self.config, layer_idx=self.config.chop_editor_at_layer
        ).to(dtype=self.config.torch_dtype)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        if not os.path.exists(self.cache_path):
            # Initialize weights and apply final processing
            logger.debug("Initializing LlamaInterpretorHypernetwork weights...")
            self.post_init()
            # Save the initialized model to cache
            logger.debug(f"Saving initialized model to {self.cache_path}")
            torch.save(self.state_dict(), self.cache_path)
        else:
            logger.debug("Loading cached LlamaInterpretorHypernetwork...")
            state_dict = torch.load(
                self.cache_path, weights_only=True, map_location=self.device
            )
            self.load_state_dict(state_dict, assign=True)

        # prune layers and add cross attn heads
        self.model.layers = self.model.layers[: self.config.chop_editor_at_layer]
        cross_attn_layers = list(range(self.config.chop_editor_at_layer))

        for i, layer in enumerate(self.model.layers):
            if i not in cross_attn_layers:
                continue

            self.model.layers[i] = LlamaDecoderLayerWithDoubleCrossAttention(
                self.config, i
            ).to(dtype=self.config.torch_dtype)

            if not self.config.initialize_from_scratch:
                original_q_weights = layer.self_attn.q_proj.weight
                original_k_weights = layer.self_attn.k_proj.weight
                original_v_weights = layer.self_attn.v_proj.weight
                original_o_weights = layer.self_attn.o_proj.weight

                original_mlp_gate_proj_weights = layer.mlp.gate_proj.weight
                original_mlp_up_proj_weights = layer.mlp.up_proj.weight
                original_mlp_down_proj_weights = layer.mlp.down_proj.weight

                original_input_layernorm_weights = layer.input_layernorm.weight
                original_post_attention_layernorm = (
                    layer.post_attention_layernorm.weight
                )

                # with torch.no_grad():
                # Initialize the new layer with these parameters
                self.model.layers[i].self_attn.q_proj.weight = nn.Parameter(
                    original_q_weights
                )
                self.model.layers[i].self_attn.k_proj.weight = nn.Parameter(
                    original_k_weights
                )
                self.model.layers[i].self_attn.v_proj.weight = nn.Parameter(
                    original_v_weights
                )
                self.model.layers[i].self_attn.o_proj.weight = nn.Parameter(
                    original_o_weights
                )

                if not self.config.ablate_source_token_attention:
                    self.model.layers[i].source_cross_attn.q_proj.weight = nn.Parameter(
                        original_q_weights
                    )
                    self.model.layers[i].source_cross_attn.k_proj.weight = nn.Parameter(
                        original_k_weights
                    )
                    self.model.layers[i].source_cross_attn.v_proj.weight = nn.Parameter(
                        original_v_weights
                    )
                    self.model.layers[i].source_cross_attn.o_proj.weight = nn.Parameter(
                        original_o_weights
                    )
                    self.model.layers[
                        i
                    ].source_cross_attn_input_layernorm.weight = nn.Parameter(
                        original_input_layernorm_weights
                    )

                if not self.config.ablate_base_token_attention:
                    self.model.layers[i].base_cross_attn.q_proj.weight = nn.Parameter(
                        original_q_weights
                    )
                    self.model.layers[i].base_cross_attn.k_proj.weight = nn.Parameter(
                        original_k_weights
                    )
                    self.model.layers[i].base_cross_attn.v_proj.weight = nn.Parameter(
                        original_v_weights
                    )
                    self.model.layers[i].base_cross_attn.o_proj.weight = nn.Parameter(
                        original_o_weights
                    )
                    self.model.layers[
                        i
                    ].base_cross_attn_input_layernorm.weight = nn.Parameter(
                        original_input_layernorm_weights
                    )

                if self.config.attention_bias:
                    original_q_bias = layer.self_attn.q_proj.bias
                    original_k_bias = layer.self_attn.k_proj.bias
                    original_v_bias = layer.self_attn.v_proj.bias
                    original_o_bias = layer.self_attn.o_proj.bias

                    if not self.config.ablate_source_token_attention:
                        self.model.layers[
                            i
                        ].source_cross_attn.q_proj.bias = nn.Parameter(original_q_bias)
                        self.model.layers[
                            i
                        ].source_cross_attn.k_proj.bias = nn.Parameter(original_k_bias)
                        self.model.layers[
                            i
                        ].source_cross_attn.v_proj.bias = nn.Parameter(original_v_bias)
                        self.model.layers[
                            i
                        ].source_cross_attn.o_proj.bias = nn.Parameter(original_o_bias)

                    if not self.config.ablate_base_token_attention:
                        self.model.layers[i].base_cross_attn.q_proj.bias = nn.Parameter(
                            original_q_bias
                        )
                        self.model.layers[i].base_cross_attn.k_proj.bias = nn.Parameter(
                            original_k_bias
                        )
                        self.model.layers[i].base_cross_attn.v_proj.bias = nn.Parameter(
                            original_v_bias
                        )
                        self.model.layers[i].base_cross_attn.o_proj.bias = nn.Parameter(
                            original_o_bias
                        )

                    self.model.layers[i].self_attn.q_proj.bias = nn.Parameter(
                        original_q_bias
                    )
                    self.model.layers[i].self_attn.k_proj.bias = nn.Parameter(
                        original_k_bias
                    )
                    self.model.layers[i].self_attn.v_proj.bias = nn.Parameter(
                        original_v_bias
                    )
                    self.model.layers[i].self_attn.o_proj.bias = nn.Parameter(
                        original_o_bias
                    )

                self.model.layers[i].mlp.gate_proj.weight = nn.Parameter(
                    original_mlp_gate_proj_weights
                )
                self.model.layers[i].mlp.up_proj.weight = nn.Parameter(
                    original_mlp_up_proj_weights
                )
                self.model.layers[i].mlp.down_proj.weight = nn.Parameter(
                    original_mlp_down_proj_weights
                )

                self.model.layers[i].input_layernorm.weight = nn.Parameter(
                    original_input_layernorm_weights
                )
                self.model.layers[i].post_attention_layernorm.weight = nn.Parameter(
                    original_post_attention_layernorm
                )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        base_hidden_states: Optional[torch.Tensor] = None,
        base_attention_mask: Optional[torch.FloatTensor] = None,
        source_hidden_states: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.FloatTensor] = None,
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

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            base_hidden_states=base_hidden_states,
            base_attention_mask=base_attention_mask,
            source_hidden_states=source_hidden_states,
            source_attention_mask=source_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        attn_weight = self.lm_head(
            hidden_states,
            attention_mask=attention_mask,
            base_encoder_hidden_states=base_hidden_states,
            base_encoder_attention_mask=base_attention_mask,
            source_encoder_hidden_states=source_hidden_states,
            source_encoder_attention_mask=source_attention_mask,
            output_attentions=output_attentions,
        )

        # (output, present[,attentions])
        return hidden_states, attn_weight


class LlamaInterpretor(nn.Module):
    def __init__(
        self,
        config: LlamaInterpretorConfig,
        subspace_module=None,
        das_dimension=None,
        device="cuda",
        compute_metrics=False,
    ):
        super().__init__()

        self.config = config
        self.hypernetwork = LlamaInterpretorHypernetwork(config)
        self.hypernetwork.initialize()
        self.hypernetwork = self.hypernetwork.to(device)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            config.name_or_path, torch_dtype=config.torch_dtype, device_map={"": device}
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
                self.das_module = BoundlessRotatedSpaceIntervention(
                    embed_dim=self.target_model.config.hidden_size,
                    torch_dtype=config.torch_dtype,
                    compute_metrics=compute_metrics,
                )
            elif subspace_module == "DAS":
                self.das_module = LowRankRotatedSpaceIntervention(
                    embed_dim=self.target_model.config.hidden_size,
                    low_rank_dimension=das_dimension,
                    torch_dtype=config.torch_dtype,
                    compute_metrics=compute_metrics,
                )
            elif subspace_module == "MaskSelect":
                self.das_module = SelectiveLowRankRotatedSpaceIntervention(
                    embed_dim=self.target_model.config.hidden_size,
                    low_rank_dimension=das_dimension,
                    torch_dtype=config.torch_dtype,
                    compute_metrics=compute_metrics,
                )
            elif subspace_module == "ReflectSelect":
                self.das_module = ReflectiveLowRankRotatedSpaceIntervention(
                    embed_dim=self.target_model.config.hidden_size,
                    low_rank_dimension=das_dimension,
                    torch_dtype=config.torch_dtype,
                    compute_metrics=compute_metrics,
                )
            elif subspace_module == "QuasiProjective":
                self.das_module = QuasiProjectiveIntervention(
                    embed_dim=self.target_model.config.hidden_size,
                    dict_size=self.target_model.config.hidden_size,
                    top_k_parameter=config.top_k_parameter,
                    lambda_parameter=config.lambda_parameter,
                    importance_power=config.importance_power,
                    epsilon=config.epsilon,
                    return_penalty=True,
                    torch_dtype=config.torch_dtype,
                    compute_metrics=compute_metrics,
                )
            else:
                raise ValueError("Invalid subspace module")

        if self.use_das_intervention:
            self.das_module = self.das_module.to(device)

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

    def get_penalty(self):
        if isinstance(self.das_module, QuasiProjectiveIntervention):
            return self.das_module.get_penalty()

        return None

    def zero_penalty(self):
        if isinstance(self.das_module, QuasiProjectiveIntervention):
            self.das_module.zero_penalty()

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
        compute_metrics: bool = False,
    ) -> InterpretorModelOutput:
        assert inference_mode in [
            None,
            "column_argmax",
            "global_argmax",
            "groundtruth",
            "bidding_argmax",
        ]

        metrics = {}

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

        if intervention_weight is None or inference_mode == "groundtruth":
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

            collapsed_source_attention_mask = source_intervention_mask.unsqueeze(
                -1
            ).repeat(1, 1, n_layer)
            collapsed_source_attention_mask = collapsed_source_attention_mask.reshape(
                source_intervention_mask.shape[0],
                source_intervention_mask.shape[1] * n_layer,
            )

            interpretor_output = self.hypernetwork(
                input_ids=editor_input_ids,
                attention_mask=editor_attention_mask,
                base_hidden_states=collapsed_base_hidden_states,
                base_attention_mask=collapsed_base_attention_mask,
                source_hidden_states=collapsed_source_hidden_states,
                source_attention_mask=collapsed_source_attention_mask,
                use_cache=False,
            )

            if inference_mode == "groundtruth":
                hypernet_hidden_states, _ = interpretor_output
                intervention_weight = intervention_weight.to(
                    dtype=hypernet_hidden_states.dtype
                )
            else:
                # Multiply the outputs by normalization factors
                hypernet_hidden_states, intervention_weight = interpretor_output
                intervention_weight = intervention_weight.squeeze()

        if inference_mode == "global_argmax":
            batch_size, _, num_base_pos = intervention_weight.shape
            source_base_intervention_flatten = intervention_weight[:, :-1, :].view(
                batch_size, -1
            )
            max_intervention_position = torch.argmax(
                source_base_intervention_flatten, dim=1
            )
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
            intervention_weight = (
                torch.nn.functional.one_hot(
                    intervention_weight, num_classes=num_src_pos
                )
                .to(dtype=intervention_weight.dtype)
                .permute(0, 2, 1)
            )
        elif inference_mode == "bidding_argmax":
            batch_size, num_src_pos, num_base_pos = intervention_weight.shape
            bidding_weight = torch.argmax(intervention_weight[:, :-1, :], dim=-1)
            bidding_weight = torch.nn.functional.one_hot(
                bidding_weight, num_classes=num_base_pos
            ).float()
            bidding_weight = torch.cat(
                [
                    bidding_weight,
                    torch.ones(batch_size, 1, num_base_pos).to(bidding_weight.device),
                ],
                dim=1,
            )
            intervention_weight = torch.where(
                bidding_weight == 1,
                intervention_weight,
                torch.zeros_like(intervention_weight),
            )
            if self.bidding_threshold is not None:
                threshold = torch.Tensor([self.bidding_threshold]).to(
                    intervention_weight.device
                )
                threshold = threshold.repeat(batch_size, num_base_pos)
                intervention_weight[:, -1, :] = torch.where(
                    intervention_weight[:, -1, :] > self.bidding_threshold,
                    intervention_weight[:, -1, :],
                    threshold,
                )
            intervention_weight = torch.argmax(intervention_weight, dim=1)
            intervention_weight = (
                torch.nn.functional.one_hot(
                    intervention_weight, num_classes=num_src_pos
                )
                .to(dtype=intervention_weight.dtype)
                .permute(0, 2, 1)
            )

        if len(intervention_weight.shape) == 2:
            intervention_weight = intervention_weight.unsqueeze(
                0
            )  # Unsqueeze first dim if batch size = 1

        source_output = self.target_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            output_hidden_states=True,
        )

        source_hidden_states = source_output.hidden_states[intervention_layer]

        intervention_matrix = torch.einsum(
            "bij,bid->bijd", intervention_weight[:, :-1, :], source_hidden_states
        )  # TODO: Fix it to help the new implement
        intervention_matrix = intervention_matrix.sum(dim=1)

        das_metrics = {}

        # Run target model with edit vectors.
        # This adds the edit vectors to the given hidden state at the specified batch index, position, and layer
        def representation_swap(module, input, output):
            nonlocal das_metrics
            base_hidden_states = output[0].clone()
            batch_size = base_hidden_states.shape[0]
            base_intervention_weight = intervention_weight[:, -1, :]

            if self.use_das_intervention:
                # print(intervention_matrix.shape)
                # print(intervention_matrix[0, 1])
                # print(base_hidden_states.shape)
                # print(base_intervention_weight[0])

                # print(torch.einsum("bid,bi->bid", base_hidden_states, - base_intervention_weight)[0, 1])
                # print(torch.einsum("bid,bi->bid", base_hidden_states, base_intervention_weight)[0, 1])
                # source_intervention_hidden_states = intervention_matrix + torch.einsum("bid,bi->bid", base_hidden_states, - base_intervention_weight)

                source_intervention_hidden_states = intervention_matrix + torch.einsum(
                    "bid,bi->bid", base_hidden_states, base_intervention_weight
                )

                if self.das_selective_subspace:
                    mixed_output, module_das_metrics = self.das_module(
                        base_hidden_states,
                        source_intervention_hidden_states,
                        hypernet_hidden_states,
                    )
                else:
                    mixed_output, module_das_metrics = self.das_module(
                        base_hidden_states,
                        source_intervention_hidden_states,
                        batch_size,
                    )

                # Find the module name in the state dict
                for k, v in module_das_metrics.items():
                    module_name = next(
                        (
                            name
                            for name, mod in self.target_model.named_modules()
                            if mod is module
                        ),
                        None,
                    )
                    if module_name:
                        das_metrics[f"{module_name}/{k}"] = v
                    else:
                        das_metrics[f"unknown_module/{k}"] = v

                output[0][:] += mixed_output - base_hidden_states
            else:
                res_diff = torch.einsum(
                    "bid,bi->bid", base_hidden_states, (1 - base_intervention_weight)
                )
                output[0][:] += intervention_matrix - res_diff

        def embedding_representation_swap(module, input, output):
            if self.use_das_intervention:
                raise NotImplementedError(
                    "DAS intervention is not supported for token embeddings"
                )

            base_hidden_states = output.clone()
            base_intervention_weight = intervention_weight[:, -1, :]
            res_diff = torch.einsum(
                "bid,bi->bid", base_hidden_states, (1 - base_intervention_weight)
            )
            output += intervention_matrix - res_diff

        # Now editing the target model
        if intervention_layer == 0:
            hooks = [
                (self.target_model.model.embed_tokens, embedding_representation_swap)
            ]
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
                output_hidden_states=output_edited_hidden_states,
            )

        logits = target_result.logits

        # collate metrics from das module, etc. to output
        output = InterpretorModelOutput(logits=logits)

        metrics.update(das_metrics)
        output.metrics = metrics

        if output_edited_hidden_states:
            output.edited_hidden_states = target_result.hidden_states

        if output_intervention_weight:
            output.intervention_weight = intervention_weight

        if output_vanilla_hidden_states:
            output.vanilla_base_hidden_states = base_hidden_states
            output.vanilla_source_hidden_states = source_hidden_states

        return output
