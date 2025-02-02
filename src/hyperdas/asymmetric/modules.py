from typing import Any, List, Mapping, Optional, Tuple, TypeVar, Union
import warnings

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaRMSNorm,
    LlamaDecoderLayer,
    apply_rotary_pos_emb,
    repeat_kv
)


T = TypeVar("T", bound="HyperDAS")

from ..utils import (
    InterpretorModelOutput,
    InterventionModuleOutput,
    add_fwd_hooks,
    assign_layer_indices,
)
from .layers import HyperDASDecoderLayerWithDoubleCrossAttention, HyperDASCrossAttention
from .configs import *

from ..das_utils import (
    BoundlessRotatedSpaceIntervention, 
    QuasiProjectiveIntervention,
    LowRankRotatedSpaceIntervention, 
    SelectiveLowRankRotatedSpaceIntervention,
    ReflectiveLowRankRotatedSpaceIntervention,
)


from transformers.cache_utils import StaticCache, DynamicCache, Cache

from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
    AutoTokenizer
)

from transformers.modeling_outputs import BaseModelOutputWithPast
import math

import torch.nn.functional as F
from transformers.cache_utils import Cache

from ..utils import apply_rotary_pos_emb_single_attn
    

class HyperDASModel(LlamaModel):
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
                
                if isinstance(decoder_layer, HyperDASDecoderLayerWithDoubleCrossAttention):
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
                if isinstance(decoder_layer, HyperDASDecoderLayerWithDoubleCrossAttention):
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
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class HyperDASForTokenSelection(LlamaForCausalLM):
    _tied_weights_keys = []

    def __init__(self, config: HyperDASConfig):
        super().__init__(config)
        
        if config.initialize_from_pretrained:
            self.model = HyperDASModel.from_pretrained(
                config.name_or_path, torch_dtype = config.torch_dtype
            )
        else:
            self.model = HyperDASModel(config=config)
            self.model = self.model.to(dtype=config.torch_dtype)
        
        self.lm_head = TokenSelectionHead(
            config=config, layer_idx=config.num_decoders
        ).to(dtype=config.torch_dtype)
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # Initialize weights and apply final processing
        self.post_init()

        # prune layers and add cross attn heads
        self.model.layers = self.model.layers[: config.num_decoders]
        cross_attn_layers = list(range(config.num_decoders))
        
        for i, layer in enumerate(self.model.layers):
            if i not in cross_attn_layers:
                continue
            
            self.model.layers[i] = HyperDASDecoderLayerWithDoubleCrossAttention(config, i).to(dtype=config.torch_dtype)
            
            if config.initialize_from_pretrained:
                original_q_weights = layer.self_attn.q_proj.weight
                original_k_weights = layer.self_attn.k_proj.weight
                original_v_weights = layer.self_attn.v_proj.weight
                original_o_weights = layer.self_attn.o_proj.weight
                
                original_mlp_gate_proj_weights = layer.mlp.gate_proj.weight
                original_mlp_up_proj_weights = layer.mlp.up_proj.weight
                original_mlp_down_proj_weights = layer.mlp.down_proj.weight
                
                original_input_layernorm_weights = layer.input_layernorm.weight
                original_post_attention_layernorm = layer.post_attention_layernorm.weight
                
                # with torch.no_grad():
                # Initialize the new layer with these parameters
                self.model.layers[i].self_attn.q_proj.weight = nn.Parameter(original_q_weights)
                self.model.layers[i].self_attn.k_proj.weight = nn.Parameter(original_k_weights)
                self.model.layers[i].self_attn.v_proj.weight = nn.Parameter(original_v_weights)
                self.model.layers[i].self_attn.o_proj.weight = nn.Parameter(original_o_weights)
                
                if not config.ablate_source_token_attention:
                    self.model.layers[i].source_cross_attn.q_proj.weight = nn.Parameter(original_q_weights)
                    self.model.layers[i].source_cross_attn.k_proj.weight = nn.Parameter(original_k_weights)
                    self.model.layers[i].source_cross_attn.v_proj.weight = nn.Parameter(original_v_weights)
                    self.model.layers[i].source_cross_attn.o_proj.weight = nn.Parameter(original_o_weights)
                    self.model.layers[i].source_cross_attn_input_layernorm.weight = nn.Parameter(original_input_layernorm_weights)
                
                if not config.ablate_base_token_attention:
                    self.model.layers[i].base_cross_attn.q_proj.weight = nn.Parameter(original_q_weights)
                    self.model.layers[i].base_cross_attn.k_proj.weight = nn.Parameter(original_k_weights)
                    self.model.layers[i].base_cross_attn.v_proj.weight = nn.Parameter(original_v_weights)
                    self.model.layers[i].base_cross_attn.o_proj.weight = nn.Parameter(original_o_weights)
                    self.model.layers[i].base_cross_attn_input_layernorm.weight = nn.Parameter(original_input_layernorm_weights)
                
                if config.attention_bias:
                    original_q_bias = layer.self_attn.q_proj.bias
                    original_k_bias = layer.self_attn.k_proj.bias
                    original_v_bias = layer.self_attn.v_proj.bias
                    original_o_bias = layer.self_attn.o_proj.bias
                    
                    if not config.ablate_source_token_attention:
                        self.model.layers[i].source_cross_attn.q_proj.bias = nn.Parameter(original_q_bias)
                        self.model.layers[i].source_cross_attn.k_proj.bias = nn.Parameter(original_k_bias)
                        self.model.layers[i].source_cross_attn.v_proj.bias = nn.Parameter(original_v_bias)
                        self.model.layers[i].source_cross_attn.o_proj.bias = nn.Parameter(original_o_bias)
                    
                    if not config.ablate_base_token_attention:
                        self.model.layers[i].base_cross_attn.q_proj.bias = nn.Parameter(original_q_bias)
                        self.model.layers[i].base_cross_attn.k_proj.bias = nn.Parameter(original_k_bias)
                        self.model.layers[i].base_cross_attn.v_proj.bias = nn.Parameter(original_v_bias)
                        self.model.layers[i].base_cross_attn.o_proj.bias = nn.Parameter(original_o_bias)
                        
                    self.model.layers[i].self_attn.q_proj.bias = nn.Parameter(original_q_bias)
                    self.model.layers[i].self_attn.k_proj.bias = nn.Parameter(original_k_bias)
                    self.model.layers[i].self_attn.v_proj.bias = nn.Parameter(original_v_bias)
                    self.model.layers[i].self_attn.o_proj.bias = nn.Parameter(original_o_bias)
                
                
                self.model.layers[i].mlp.gate_proj.weight = nn.Parameter(original_mlp_gate_proj_weights)
                self.model.layers[i].mlp.up_proj.weight = nn.Parameter(original_mlp_up_proj_weights)
                self.model.layers[i].mlp.down_proj.weight = nn.Parameter(original_mlp_down_proj_weights)
                
                self.model.layers[i].input_layernorm.weight = nn.Parameter(original_input_layernorm_weights)
                self.model.layers[i].post_attention_layernorm.weight = nn.Parameter(original_post_attention_layernorm)


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

class TokenSelectionHead(HyperDASCrossAttention):
    
    _flash_attn_enabled = False
    
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx, True)
        self.intervention_layer = config.intervention_layer
        self.q_combine = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        nn.init.uniform_(self.q_combine.weight)
        
        self.break_asymmetric = config.break_asymmetric
        self.encoder_hidden_states_num_layers = config.target_model_num_hidden_layers + 1
        
    def _update_encoder_attention_mask(self, attention_mask, attn_weights):
        attention_mask = attention_mask.unsqueeze(1)
        dtype = attn_weights.dtype
        dtype_min = torch.finfo(attn_weights.dtype).min
        attention_mask = attention_mask.to(dtype)
        attention_mask = attention_mask.masked_fill_(attention_mask == 0, dtype_min)
        attention_mask = attention_mask.masked_fill_(attention_mask == 1, 0.0)
        return attention_mask
        
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        base_encoder_hidden_states: Optional[torch.Tensor] = None,
        base_encoder_attention_mask: Optional[torch.FloatTensor] = None,
        source_encoder_hidden_states: Optional[torch.Tensor] = None,
        source_encoder_attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if base_encoder_hidden_states is not None or source_encoder_hidden_states is not None:
            if not hasattr(self, "q_combine"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
        else:
            raise ValueError("This class is only meant to be used as cross attention")
            # query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        
        n_layers = self.encoder_hidden_states_num_layers
        base_n_tokens = base_encoder_attention_mask.shape[-1] // n_layers
        source_n_tokens = source_encoder_attention_mask.shape[-1] // n_layers
        
        base_start, base_end = (
            self.intervention_layer * base_n_tokens,
            (self.intervention_layer + 1) * base_n_tokens   
        )
        
        source_start, source_end = (
            self.intervention_layer * source_n_tokens,
            (self.intervention_layer + 1) * source_n_tokens
        )
        
        batch_size = base_encoder_attention_mask.shape[0]
        base_encoder_attention_mask = base_encoder_attention_mask.reshape(batch_size, base_n_tokens, n_layers)[:, :, self.intervention_layer]
        base_encoder_hidden_states = base_encoder_hidden_states.reshape(batch_size, base_n_tokens, n_layers, -1)[:, :, self.intervention_layer, :]
        source_encoder_attention_mask = source_encoder_attention_mask.reshape(batch_size, source_n_tokens, n_layers)[:, :, self.intervention_layer]
        source_encoder_hidden_states = source_encoder_hidden_states.reshape(batch_size, source_n_tokens, n_layers, -1)[:, :, self.intervention_layer, :]
        
        expanded_base_encoder_hidden_states = base_encoder_hidden_states.unsqueeze(1).expand(-1, source_n_tokens, -1, -1)
        expanded_source_encoder_hidden_states = source_encoder_hidden_states.unsqueeze(2).expand(-1, -1, base_n_tokens, -1)
        
        if self.break_asymmetric:
            # Flip a coin to decide base_encoder_hidden_states goes first or source_encoder_hidden_states goes first
            
            coin_flip = torch.randint(0, 2, (1,)).item()
            if coin_flip:
                encoder_hidden_states = torch.concat([expanded_source_encoder_hidden_states, expanded_base_encoder_hidden_states], dim=-1)
            else:
                encoder_hidden_states = torch.concat([expanded_base_encoder_hidden_states, expanded_source_encoder_hidden_states], dim=-1)
        else:
            encoder_hidden_states = torch.concat([expanded_base_encoder_hidden_states, expanded_source_encoder_hidden_states], dim=-1)
        encoder_hidden_states = self.q_combine(encoder_hidden_states)
        encoder_hidden_states = torch.concat([encoder_hidden_states, base_encoder_hidden_states.unsqueeze(1)], dim=1)

        encoder_attention_mask = torch.einsum("bq,bs->bsq", base_encoder_attention_mask, source_encoder_attention_mask)
        encoder_attention_mask = torch.concat([encoder_attention_mask, torch.ones_like(base_encoder_attention_mask).unsqueeze(1)], dim=1)
        
        batch_size, _, _ = encoder_attention_mask.shape
        
        encoder_hidden_states = encoder_hidden_states.reshape(
            batch_size, 
            (source_n_tokens + 1) * base_n_tokens,
            -1
        )
        
        encoder_attention_mask = encoder_attention_mask.reshape(
            batch_size,
            (source_n_tokens + 1) * base_n_tokens
        )
        
        bsz, q_len, _ = hidden_states.size()
        _, kv_len, _ = encoder_hidden_states.size()
        
        if position_ids is None:
            if attention_mask is None:
                position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
            else:
                position_ids = torch.cumsum(attention_mask, dim=1) * attention_mask
            
        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            
            assert encoder_hidden_states is not None, "Cross attention requires encoder_hidden_states"
            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)
            
            key_position_ids = torch.cumsum(encoder_attention_mask, dim=1) * encoder_attention_mask
            key_states = [F.linear(encoder_hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)
        else:
            assert encoder_hidden_states is not None, "Cross attention requires encoder_hidden_states"
            query_states = self.q_proj(hidden_states)    
            key_states = self.k_proj(encoder_hidden_states)
            key_position_ids = torch.cumsum(encoder_attention_mask, dim=1) * encoder_attention_mask

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        past_key_value = getattr(self, "past_key_value", past_key_value)
        
        q_cos, q_sin = self.rotary_emb(query_states, position_ids)
        query_states = apply_rotary_pos_emb_single_attn(query_states, q_cos, q_sin)
        kv_cos, kv_sin = self.rotary_emb(key_states, key_position_ids)
        key_states = apply_rotary_pos_emb_single_attn(key_states, kv_cos, kv_sin)
            
        if past_key_value is not None:
            raise NotImplementedError
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        
        query_states = query_states[:, :, -1, :].unsqueeze(2) # Take the last token from the editor instruction
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        """
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        """
        
        # convert encoder attention mask from 1->0, 0->-inf to mask attention weights before softmax
        encoder_attention_mask = self._update_encoder_attention_mask(encoder_attention_mask, attn_weights)
        
        attn_weights = attn_weights + encoder_attention_mask.unsqueeze(1)
        attn_weights = torch.mean(attn_weights, dim=1, keepdim=False)
        attn_weights = attn_weights.view(bsz, 1, source_n_tokens + 1, base_n_tokens).squeeze()
        attn_weights = nn.functional.softmax(attn_weights, dim=1, dtype=torch.float32).to(query_states.dtype)
        
        # return None, attn_weights, past_key_value
        return attn_weights
    
class HyperDAS(nn.Module):
    def __init__(self, config: HyperDASConfig):
        super().__init__()

        self.config = config
        self.hypernetwork = HyperDASForTokenSelection(config)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            config.target_model_name_or_path, torch_dtype=config.torch_dtype
        )
        
        self.bidding_threshold = 0.1
        
        self.subspace_intervention = config.subspace_config.is_subspace_intervention
        self.selective_subspace = config.subspace_config.subspace_module in ["ReflectSelect", "QuasiProjective"]
                
        if self.subspace_intervention:
            
            if isinstance(config.subspace_config, BoundlessDASConfig):
                self.das_module = BoundlessRotatedSpaceIntervention(
                    embed_dim=self.target_model.config.hidden_size, torch_dtype=config.torch_dtype
                )
            elif isinstance(config.subspace_config, MDASConfig):
                self.das_module = LowRankRotatedSpaceIntervention(
                    embed_dim=self.target_model.config.hidden_size, low_rank_dimension=config.subspace_config.subspace_dimension, torch_dtype=config.torch_dtype
                )
            elif isinstance(config.subspace_config, MaskSelectConfig):
                self.das_module = SelectiveLowRankRotatedSpaceIntervention(
                    embed_dim=self.target_model.config.hidden_size, low_rank_dimension=config.subspace_config.subspace_dimension, torch_dtype=config.torch_dtype
                )
            elif isinstance(config.subspace_config, HouseholderConfig):
                self.das_module = ReflectiveLowRankRotatedSpaceIntervention(
                    embed_dim=self.target_model.config.hidden_size, low_rank_dimension=config.subspace_config.subspace_dimension, torch_dtype=config.torch_dtype
                )
            elif isinstance(config.subspace_config, QuasiProjectiveConfig):
                self.das_module = QuasiProjectiveIntervention(
                    embed_dim=self.target_model.config.hidden_size,
                    dict_size=config.subspace_config.dict_size,
                    scoring_dimension=config.subspace_config.scoring_dimension,
                    top_k_parameter=config.subspace_config.subspace_dimension,
                    lambda_parameter=config.subspace_config.lambda_parameter,
                    importance_power=config.subspace_config.importance_power,
                    epsilon=config.subspace_config.epsilon,
                    return_penalty=config.subspace_config.return_penalty,
                    ridge_parameterization=config.subspace_config.ridge_parameterization,
                    torch_dtype=config.torch_dtype,
                    compute_metrics=config.subspace_config.compute_metrics,
                    orthogonal_init=config.subspace_config.orthogonal_init,
                    selection_mechanism=config.subspace_config.selection_mechanism,
                    hat_matrix=config.subspace_config.hat_matrix,
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
        use_target_model_embedding: bool = True,
    ) -> InterpretorModelOutput:
        
        assert inference_mode in [None, "column_argmax", "global_argmax", "groundtruth", "bidding_argmax"]
        
                
        if intervention_layer is None:
            intervention_layer = self.config.intervention_layer 
            
        if base_position_ids is None:
            # -1 for all the padding tokens and start from 0 for the rest
            base_position_ids = torch.cumsum(base_attention_mask, dim=1) * base_attention_mask - 1
            
        if source_position_ids is None:
            # -1 for all the padding tokens and start from 0 for the rest
            source_position_ids = torch.cumsum(source_attention_mask, dim=1) * source_attention_mask - 1
        
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
            
            if use_target_model_embedding:
                """inputs_embeds = self._run_target_model_for_encoded_hidden_states(
                    editor_input_ids, target_attention_mask=editor_attention_mask
                )"""
                inputs_embeds = self.target_model.model.embed_tokens(editor_input_ids)
                editor_input_ids = None
            else:
                inputs_embeds = None

            interpretor_output = self.hypernetwork(
                input_ids=editor_input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=editor_attention_mask,
                base_hidden_states=collapsed_base_hidden_states,
                base_attention_mask=collapsed_base_attention_mask,
                source_hidden_states=collapsed_source_hidden_states,
                source_attention_mask=collapsed_source_attention_mask,
                use_cache=False
            )
            
            if inference_mode == "groundtruth":
                hypernet_hidden_states, _ = interpretor_output
                intervention_weight = intervention_weight.to(dtype=hypernet_hidden_states.dtype)
            else:
                # Multiply the outputs by normalization factors
                hypernet_hidden_states, intervention_weight = interpretor_output
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
            
            if self.subspace_intervention:
                
                # print(intervention_matrix.shape)
                # print(intervention_matrix[0, 1])
                # print(base_hidden_states.shape)
                # print(base_intervention_weight[0])
                
                # print(torch.einsum("bid,bi->bid", base_hidden_states, - base_intervention_weight)[0, 1])
                # print(torch.einsum("bid,bi->bid", base_hidden_states, base_intervention_weight)[0, 1])
                # source_intervention_hidden_states = intervention_matrix + torch.einsum("bid,bi->bid", base_hidden_states, - base_intervention_weight)
                
                source_intervention_hidden_states = intervention_matrix + torch.einsum("bid,bi->bid", base_hidden_states, base_intervention_weight)
                
                if self.selective_subspace:
                    mixed_output = self.das_module(base_hidden_states, source_intervention_hidden_states, hypernet_hidden_states)
                else:
                    mixed_output = self.das_module(base_hidden_states, source_intervention_hidden_states, batch_size)
                
                if isinstance(mixed_output, InterventionModuleOutput):
                    mixed_output = mixed_output.mixed_output
                
                output[0][:] += (mixed_output - base_hidden_states)
            else:
                res_diff = torch.einsum("bid,bi->bid", base_hidden_states, (1 - base_intervention_weight))
                output[0][:] += (intervention_matrix - res_diff)
            
        def embedding_representation_swap(module, input, output):
            if self.subspace_intervention:
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
    