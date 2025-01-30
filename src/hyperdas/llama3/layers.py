from typing import Optional, Tuple, Union
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

import math

import torch.nn.functional as F
from transformers.cache_utils import Cache

from ..utils import apply_rotary_pos_emb_single_attn


class LlamaAttentionWithCrossAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None, is_cross_attention=False):
        super().__init__(config=config, layer_idx=layer_idx)
        self.is_cross_attention = is_cross_attention

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
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()    
        _, kv_len, _ = encoder_hidden_states.size()
        
        
        
        # encoder_hidden_states (bsz, base_sentence_length * layers_num, hs)

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
            
            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)
            
            if self.is_cross_attention:
                assert encoder_hidden_states is not None, "Cross attention requires encoder_hidden_states"
                key_states = [F.linear(encoder_hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
                value_states = [F.linear(encoder_hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
                kv_position_ids = torch.arange(kv_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
            else:
                key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
                value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
                
            key_states = torch.cat(key_states, dim=-1)
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            if self.is_cross_attention:
                assert encoder_hidden_states is not None, "Cross attention requires encoder_hidden_states"
                key_states = self.k_proj(encoder_hidden_states)
                value_states = self.v_proj(encoder_hidden_states)
                # give 0 to the padding position and start from 1 for the rest
                kv_position_ids = torch.cumsum(encoder_attention_mask, dim=1) * encoder_attention_mask
            else:
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        
        if self.is_cross_attention:
            q_cos, q_sin = self.rotary_emb(query_states, position_ids)
            query_states = apply_rotary_pos_emb_single_attn(query_states, q_cos, q_sin)
            kv_cos, kv_sin = self.rotary_emb(value_states, kv_position_ids)
            key_states = apply_rotary_pos_emb_single_attn(key_states, kv_cos, kv_sin)
            
            kv_position_ids.view(33, -1)
        else:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            
        if past_key_value is not None:
            
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # no matter the length, we just slice it
        if attention_mask is not None:
            if not self.is_cross_attention:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
            else:
                # Take the attention mask of the last token to recover the oroginal (not causal) attention
                attention_mask = attention_mask[:, :, -1, :].unsqueeze(-2).repeat(1, 1, kv_len, 1)
                attention_mask = torch.transpose(attention_mask, -1, -2)
                attn_weights = attn_weights + attention_mask
        
        if encoder_attention_mask is not None:
            encoder_attention_mask = self._update_encoder_attention_mask(encoder_attention_mask, attn_weights)
            encoder_attention_mask = encoder_attention_mask.repeat(1, q_len, 1)
            attn_weights = attn_weights + encoder_attention_mask.unsqueeze(1)
            attn_weights = torch.clamp(
                attn_weights, 
                min=torch.finfo(attn_weights.dtype).min, 
                max=torch.finfo(attn_weights.dtype).max
            ) # clamp the position where attention_mask and encoder_attention_mask are both applied
        
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class LlamaDecoderLayerWithDoubleCrossAttention(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        
        self.ablate_base_token_attention = config.ablate_base_token_attention
        self.ablate_source_token_attention = config.ablate_source_token_attention
        
        if not self.ablate_source_token_attention:
            self.source_cross_attn = LlamaAttentionWithCrossAttention(config=config, layer_idx=layer_idx, is_cross_attention=True)
            self.source_cross_attn_input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if not self.ablate_base_token_attention:
            self.base_cross_attn = LlamaAttentionWithCrossAttention(config=config, layer_idx=layer_idx, is_cross_attention=True)
            self.base_cross_attn_input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        source_hidden_states: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.FloatTensor] = None,
        base_hidden_states: Optional[torch.Tensor] = None,
        base_attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        
        if source_hidden_states is not None and not self.ablate_source_token_attention:
            residual = hidden_states
            hidden_states = self.source_cross_attn_input_layernorm(hidden_states)
            source_cross_attn_outputs, _, _ = self.source_cross_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=source_hidden_states,
                encoder_attention_mask=source_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = residual + source_cross_attn_outputs

        if base_hidden_states is not None and not self.ablate_base_token_attention:
            residual = hidden_states
            hidden_states = self.base_cross_attn_input_layernorm(hidden_states)
            base_cross_attn_outputs, _, _ = self.base_cross_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=base_hidden_states,
                encoder_attention_mask=base_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = residual + base_cross_attn_outputs
           
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
    
class LlamaDecoderLayerWithCrossAttention(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int, add_cross_attention=False):
        super().__init__(config, layer_idx)
        if add_cross_attention:
            self.cross_attn = LlamaAttentionWithCrossAttention(config=config, layer_idx=layer_idx, is_cross_attention=True)
            self.cross_attn_input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        
        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "cross_attn") or not hasattr(self, "cross_attn_input_layernorm"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.cross_attn_input_layernorm(hidden_states)
            cross_attn_outputs, _, _ = self.cross_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = residual + cross_attn_outputs
            
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
    
class InterpretorUnembedCrossAttention(LlamaAttentionWithCrossAttention):
    
    _flash_attn_enabled = False
    
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx, True)
        self.intervention_layer = config.intervention_layer
        self.q_combine = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        nn.init.uniform_(self.q_combine.weight)
        
        self.break_asymmetric = config.break_asymmetric
        
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
        
        n_layers = self.config.num_hidden_layers + 1
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