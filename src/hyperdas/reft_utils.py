
from pyvene import (
    ConstantSourceIntervention,
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)

from transformers.models.llama.modeling_llama import LlamaRMSNorm

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from collections import OrderedDict


class HiddenStatesProjectionMLP(nn.Module):
    def __init__(self, in_size, out_size, intermediate_size=14336, torch_dtype=torch.float32):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.intermediate_size = intermediate_size
        
        self.gate_proj = nn.Linear(self.in_size, self.intermediate_size, bias=False, dtype=torch_dtype)
        self.up_proj = nn.Linear(self.in_size, self.intermediate_size, bias=False, dtype=torch_dtype)
        self.down_proj = nn.Linear(self.intermediate_size, self.out_size, bias=False, dtype=torch_dtype)
        self.act_fn = nn.SiLU()
        
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    

class LowRankRotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, m, init_orth=True):
        super().__init__()
        # n > m
        self.weight = torch.nn.Parameter(torch.empty(n, m), requires_grad=True)
        if init_orth:
            torch.nn.init.orthogonal_(self.weight)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)
    


class LoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.load_state_dict(state_dict, strict=False)

        # Caveat: without creating a new layer, it might not work (still not sure why)
        # We have to recreate a layer, and load back the columns.
        overload_w = state_dict["rotate_layer"].to(
            self.learned_source.weight.device)
        overload_w_width = overload_w.shape[-1]
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, overload_w_width, init_orth=True).to(
            self.learned_source.weight.device)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.rotate_layer.parametrizations.weight[0].base[:,:overload_w_width] = overload_w
        assert torch.allclose(self.rotate_layer.weight.data, overload_w.data) == True # we must match!
        
        return
    
    

class SelectiveLoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
        
        self.rv_proj = HiddenStatesProjectionMLP(
            in_size=self.embed_dim, 
            out_size=self.embed_dim, 
            torch_dtype=kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16
        )
        
        self.input_layernorm = LlamaRMSNorm(
            hidden_size=self.embed_dim, eps=1e-5
        ).to(dtype=kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        
    def forward(
        self, base, hidden_states, source=None, subspaces=None
    ):
        
        normalized_hidden_state = self.input_layernorm(hidden_states[:, -1, :])
        rv = self.rv_proj(normalized_hidden_state)
        rv = rv / torch.norm(rv, dim=-1, keepdim=True)
        
        householder = torch.eye(self.embed_dim, device=rv.device, dtype=rv.dtype).unsqueeze(0) - 2 * torch.bmm(rv.unsqueeze(2), rv.unsqueeze(1))
        reflected_weight = torch.matmul(householder, self.rotate_layer.weight.to(rv.dtype))
        
        rotated_base = torch.bmm(base, reflected_weight)
    
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base), torch.transpose(reflected_weight, 1, 2)
        )
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.load_state_dict(state_dict, strict=False)

        # Caveat: without creating a new layer, it might not work (still not sure why)
        # We have to recreate a layer, and load back the columns.
        overload_w = state_dict["rotate_layer"].to(
            self.learned_source.weight.device)
        overload_w_width = overload_w.shape[-1]
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, overload_w_width, init_orth=True).to(
            self.learned_source.weight.device)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.rotate_layer.parametrizations.weight[0].base[:,:overload_w_width] = overload_w
        assert torch.allclose(self.rotate_layer.weight.data, overload_w.data) == True # we must match!
        
        return
    
    
class SteeringVecLoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, hidden_states, source=None, subspaces=None
    ):
        output = base + hidden_states[:, -1, :].unsqueeze(1).repeat(1, base.shape[1], 1)
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        
        return
