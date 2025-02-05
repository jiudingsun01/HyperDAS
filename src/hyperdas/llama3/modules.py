from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple, TypeVar, Union
import wandb
import os
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
    AutoTokenizer
)


from transformers.cache_utils import StaticCache, DynamicCache, Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithCrossAttentions

from ..utils import (
    InterpretorModelOutput,
    add_fwd_hooks,
    assign_layer_indices,
)
from .layers import InterpretorUnembedCrossAttention, LlamaDecoderLayerWithDoubleCrossAttention

from ..reft_utils import LoreftIntervention, SelectiveLoreftIntervention, SteeringVecLoreftIntervention
from ..utils import InterventionModuleOutput

from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
import seaborn as sns
import time
from dataclasses import dataclass


T = TypeVar("T", bound="LlamaInterpretor")


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
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
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
    num_target_model_layers : int = 32
    

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
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
            
        if not use_cache:
            past_seen_tokens = None

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens, output_attentions=True)

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
                
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
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
                    attention_mask=causal_mask,
                    base_encoder_hidden_states=base_encoder_hidden_states,
                    base_encoder_attention_mask=base_encoder_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
                
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache = layer_outputs[4 if output_attentions else 3]

            if output_attentions:
                all_self_attns += (layer_outputs[3],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return HyperDASModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaInterpretorHypernetwork(LlamaForCausalLM):
    _tied_weights_keys = []

    def __init__(self, config: LlamaInterpretorConfig):
        super().__init__(config)
        
        if not config.initialize_from_scratch:
            self.model = LlamaModelWithCrossAttention.from_pretrained(
                config.name_or_path, torch_dtype = config.torch_dtype
            )
        else:
            print("Initializing from scratch...")
            self.model = LlamaModelWithCrossAttention._from_config(config)
            self.model = self.model.to(dtype=config.torch_dtype)
        
        self.lm_head = InterpretorUnembedCrossAttention(
            config=config, layer_idx=config.chop_editor_at_layer
        ).to(dtype=config.torch_dtype)
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # Initialize weights and apply final processing
        self.post_init()

        # prune layers and add cross attn heads
        self.model.layers = self.model.layers[: config.chop_editor_at_layer]
        cross_attn_layers = list(range(config.chop_editor_at_layer))
        
        for i, layer in enumerate(self.model.layers):
            if i not in cross_attn_layers:
                continue
            
            self.model.layers[i] = LlamaDecoderLayerWithDoubleCrossAttention(config, i).to(dtype=config.torch_dtype)
            
            if not config.initialize_from_scratch:
                original_q_weights = layer.self_attn.q_proj.weight
                original_k_weights = layer.self_attn.k_proj.weight
                original_v_weights = layer.self_attn.v_proj.weight
                original_o_weights = layer.self_attn.o_proj.weight
                
                
                original_mlp_gate_proj_weights = layer.mlp.gate_proj.weight
                original_mlp_up_proj_weights = layer.mlp.up_proj.weight
                original_mlp_down_proj_weights = layer.mlp.down_proj.weight
                
                original_input_layernorm_weights = layer.input_layernorm.weight
                original_post_attention_layernorm = layer.post_attention_layernorm.weight
                
                self.model.layers[i].self_attn.q_proj.weight = nn.Parameter(original_q_weights)
                self.model.layers[i].self_attn.k_proj.weight = nn.Parameter(original_k_weights)
                self.model.layers[i].self_attn.v_proj.weight = nn.Parameter(original_v_weights)
                self.model.layers[i].self_attn.o_proj.weight = nn.Parameter(original_o_weights)

                self.model.layers[i].cross_attn.q_proj.weight = nn.Parameter(original_q_weights)
                self.model.layers[i].cross_attn.k_proj.weight = nn.Parameter(original_k_weights)
                self.model.layers[i].cross_attn.v_proj.weight = nn.Parameter(original_v_weights)
                self.model.layers[i].cross_attn.o_proj.weight = nn.Parameter(original_o_weights)
                
                self.model.layers[i].cross_attn_input_layernorm.weight = nn.Parameter(original_input_layernorm_weights)
                self.model.layers[i].post_cross_attn_layernorm.weight = nn.Parameter(original_post_attention_layernorm)
                
                if config.attention_bias:
                    original_q_bias = layer.self_attn.q_proj.bias
                    original_k_bias = layer.self_attn.k_proj.bias
                    original_v_bias = layer.self_attn.v_proj.bias
                    original_o_bias = layer.self_attn.o_proj.bias
                    
                    self.model.layers[i].cross_attn.q_proj.bias = nn.Parameter(original_q_bias)
                    self.model.layers[i].cross_attn.k_proj.bias = nn.Parameter(original_k_bias)
                    self.model.layers[i].cross_attn.v_proj.bias = nn.Parameter(original_v_bias)
                    self.model.layers[i].cross_attn.o_proj.bias = nn.Parameter(original_o_bias)
                        
                    self.model.layers[i].self_attn.q_proj.bias = nn.Parameter(original_q_bias)
                    self.model.layers[i].self_attn.k_proj.bias = nn.Parameter(original_k_bias)
                    self.model.layers[i].self_attn.v_proj.bias = nn.Parameter(original_v_bias)
                    self.model.layers[i].self_attn.o_proj.bias = nn.Parameter(original_o_bias)
                
                self.model.layers[i].mlp.gate_proj.weight = nn.Parameter(original_mlp_gate_proj_weights)
                self.model.layers[i].mlp.up_proj.weight = nn.Parameter(original_mlp_up_proj_weights)
                self.model.layers[i].mlp.down_proj.weight = nn.Parameter(original_mlp_down_proj_weights)
                
                self.model.layers[i].cross_attn_mlp.gate_proj.weight = nn.Parameter(original_mlp_gate_proj_weights)
                self.model.layers[i].cross_attn_mlp.up_proj.weight = nn.Parameter(original_mlp_up_proj_weights)
                self.model.layers[i].cross_attn_mlp.down_proj.weight = nn.Parameter(original_mlp_down_proj_weights)
                
                self.model.layers[i].input_layernorm.weight = nn.Parameter(original_input_layernorm_weights)
                self.model.layers[i].post_attention_layernorm.weight = nn.Parameter(original_post_attention_layernorm)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        base_encoder_hidden_states: Optional[torch.Tensor] = None,
        base_encoder_attention_mask: Optional[torch.FloatTensor] = None,
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
            base_encoder_hidden_states=base_encoder_hidden_states,
            base_encoder_attention_mask=base_encoder_attention_mask,
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

        base_attn_weight = self.lm_head(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=base_encoder_hidden_states,
            encoder_attention_mask=base_encoder_attention_mask,
            output_attentions=output_attentions,
        )
                
        # (output, present[,attentions])
        return hidden_states, base_attn_weight
        # return base_hidden_states, source_attn_weight, base_attn_weight
        


class LlamaInterpretor(nn.Module):
    def __init__(self, config: LlamaInterpretorConfig, target_model_name_or_path, subspace_module=None, das_dimension=None):
        super().__init__()

        self.config = config
        self.hypernetwork = LlamaInterpretorHypernetwork(config)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name_or_path, torch_dtype = config.torch_dtype
        )
        
        self.bidding_threshold = 0.1
        
        self.use_das_intervention = subspace_module != None
        self.das_selective_subspace = subspace_module in ["SelectiveLoReFT", "SteeringVec"]
                
        if self.use_das_intervention:
            
            if subspace_module == "LoReFT":
                self.das_module = LoreftIntervention(
                    embed_dim=self.target_model.config.hidden_size, low_rank_dimension=das_dimension, dtype=config.torch_dtype
                )
            elif subspace_module == "SelectiveLoReFT":
                self.das_module = SelectiveLoreftIntervention(
                    embed_dim=self.target_model.config.hidden_size, low_rank_dimension=das_dimension, dtype=config.torch_dtype
                )
            elif subspace_module == "SteeringVec":
                self.das_module = SteeringVecLoreftIntervention(
                    embed_dim=self.target_model.config.hidden_size, low_rank_dimension=das_dimension, dtype=config.torch_dtype
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
        base_hidden_states: torch.Tensor = None,
        base_position_ids: torch.Tensor = None,
        intervention_layer: int = None,
        base_intervention_weight: torch.Tensor = None,
    ) -> InterpretorModelOutput:        
                
        if intervention_layer is None:
            intervention_layer = self.config.intervention_layer 
            
        if base_position_ids is None:
            # -1 for all the padding tokens and start from 0 for the rest
            base_position_ids = torch.cumsum(base_attention_mask, dim=1) * base_attention_mask - 1
        
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
            zero_row_idx = (base_intervention_mask.sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
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
        
        collapsed_base_attention_mask = base_intervention_mask.unsqueeze(-1).repeat(1, 1, n_layer)
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
            use_cache=False
        )
            
        
        if base_intervention_weight is None:
            base_intervention_weight = interpretor_output[1]
        else:
            base_intervention_weight = base_intervention_weight.to(interpretor_output[1].dtype)
            
        hypernet_hidden_states = interpretor_output[0]
                                        
        # print(source_intervention[0])            
        
        # Run target model with edit vectors.
        # This adds the edit vectors to the given hidden state at the specified batch index, position, and layer
        def representation_swap(module, input, output):
            base_hidden_states = output[0].clone()
            
            # print(base_hidden_states[0, 18, :])
            # print(source_hidden_states[0, 6, :])
                                    
            if self.use_das_intervention:
                
                hidden_dim = base_hidden_states.shape[-1]
                base_hidden_states_weight = base_intervention_weight.unsqueeze(-1).repeat(1, 1, hidden_dim) 
                
                if self.das_selective_subspace:
                    mixed_output = self.das_module(base_hidden_states, hidden_states=hypernet_hidden_states) * base_hidden_states_weight
                else:
                    mixed_output = self.das_module(base_hidden_states) * base_hidden_states_weight
                
                output[0][:] += (mixed_output - base_hidden_states)    
            else:
                raise NotImplementedError("Only DAS is implemented")

        # Now editing the target model
        if intervention_layer == 0:
            raise ValueError("Cannot intervene at layer 0")
        else:
            hooks = [(self.target_model.model.layers[intervention_layer - 1], representation_swap)]

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
            base_intervention_weight=base_intervention_weight,
            target_hidden_states=target_result.hidden_states,
        )

        return output
    
