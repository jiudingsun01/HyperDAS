from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_single_attn(q_or_k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_or_k_embed = (q_or_k * cos) + (rotate_half(q_or_k) * sin)
    return q_or_k_embed



class EditorConfig(PretrainedConfig):
    name_or_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v0.1"
    chop_editor_at_layer: int = -1
    num_editing_heads: int = 32
    use_layerwise_embeddings: bool = False
    kill_token_zero: bool = False
    cross_attn_layers: Optional[List[int]] = []


@dataclass
class EditorModelOutput(BaseModelOutput):
    logits: Optional[torch.Tensor] = None
    target_hidden_states: Optional[torch.Tensor] = None
    edited_hidden_states: Optional[torch.Tensor] = None
    edit_vectors: Optional[torch.Tensor] = None
    editor_attention: Optional[torch.Tensor] = None
    

@dataclass
class InterpretorModelOutput(BaseModelOutput):
    logits: Optional[torch.Tensor] = None
    target_hidden_states: Optional[torch.Tensor] = None
    edited_hidden_states: Optional[torch.Tensor] = None
    intervention_weight: Optional[torch.Tensor] = None
    vanilla_base_hidden_states: Optional[torch.Tensor] = None
    vanilla_source_hidden_states: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


@contextlib.contextmanager
def add_fwd_hooks(module_hooks):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for mod, hk in module_hooks:
            handles.append(mod.register_forward_hook(hk))
        yield
    finally:
        for h in handles:
            h.remove()


def assign_layer_indices(model):
    """
    Assigns a custom attribute 'layer_index' to each transformer layer in the GPT-2 model.
    This function iterates over the transformer blocks and assigns an index to each.
    """
    if "gpt2" in model.config.name_or_path:
        model.transformer.wte.layer_index = 0
        for i, layer in enumerate(model.transformer.h):
            layer.layer_index = i + 1
    
