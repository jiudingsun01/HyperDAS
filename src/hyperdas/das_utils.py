import torch.nn as nn
from abc import ABC, abstractmethod
from transformers.models.llama.modeling_llama import LlamaRMSNorm
import torch


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


class Intervention(nn.Module):

    """Intervention the original representations."""

    def __init__(self, **kwargs):
        super().__init__()
        self.trainable = False
        self.is_source_constant = False

        self.keep_last_dim = kwargs["keep_last_dim"] if "keep_last_dim" in kwargs else False
        self.use_fast = kwargs["use_fast"] if "use_fast" in kwargs else False
        self.subspace_partition = (
            kwargs["subspace_partition"] if "subspace_partition" in kwargs else None
        )
        # we turn the partition into list indices
        if self.subspace_partition is not None:
            expanded_subspace_partition = []
            for subspace in self.subspace_partition:
                if len(subspace) == 2 and isinstance(subspace[0], int):
                    expanded_subspace_partition.append([i for i in range(subspace[0],subspace[1])])
                else:
                    # it could be discrete indices.
                    expanded_subspace_partition.append(subspace)
            self.subspace_partition = expanded_subspace_partition
            
        if "embed_dim" in kwargs and kwargs["embed_dim"] is not None:
            self.register_buffer('embed_dim', torch.tensor(kwargs["embed_dim"]))
            self.register_buffer('interchange_dim', torch.tensor(kwargs["embed_dim"]))
        else:
            self.embed_dim = None
            self.interchange_dim = None
            
        if "source_representation" in kwargs and kwargs["source_representation"] is not None:
            self.is_source_constant = True
            self.register_buffer('source_representation', kwargs["source_representation"])
        else:
            if "hidden_source_representation" in kwargs and \
                kwargs["hidden_source_representation"] is not None:
                self.is_source_constant = True
            else:
                self.source_representation = None
                
    def set_source_representation(self, source_representation):
        self.is_source_constant = True
        self.register_buffer('source_representation', source_representation)
                
    def set_interchange_dim(self, interchange_dim):
        if not isinstance(interchange_dim, torch.Tensor):
            # Convert integer or list into torch.Tensor.
            self.interchange_dim = torch.tensor(interchange_dim)
        else:
            self.interchange_dim = interchange_dim
            
    @abstractmethod
    def forward(self, base, source, subspaces=None):
        pass
    
    
class DistributedRepresentationIntervention(nn.Module):

    """Distributed representation."""

    def __init__(self, **kwargs):
        super().__init__()
        self.is_repr_distributed = True
        
        
class TrainableIntervention(Intervention):

    """Intervention the original representations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainable = True
        self.is_source_constant = False
        
    def tie_weight(self, linked_intervention):
        pass
    

def _can_use_fast(
    subspaces
):
    tensorfiable = True
    row_same_val = False
    try:
        subspaces = torch.tensor(subspaces)
        row_same_val = torch.all(subspaces == subspaces[0], axis=1).all()
    except:
        tensorfiable = False
        
    return row_same_val and tensorfiable


class RotateLayer(nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, init_orth=True):
        super().__init__()
        weight = torch.empty(n, n)
        # we don't need init if the saved checkpoint has a nice
        # starting point already.
        # you can also study this if you want, but it is our focus.
        if init_orth:
            torch.nn.init.orthogonal_(weight)
        self.weight = torch.nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)


class LowRankRotateLayer(nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, m, init_orth=True):
        super().__init__()
        # n > m
        self.weight = torch.nn.Parameter(torch.empty(n, m), requires_grad=True)
        if init_orth:
            torch.nn.init.orthogonal_(self.weight)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)


def sigmoid_boundary(_input, boundary_x, boundary_y, temperature):
    """Generate sigmoid mask"""
    return torch.sigmoid((_input - boundary_x) / temperature) * torch.sigmoid(
        (boundary_y - _input) / temperature
    )

def get_rotation_mask(subspace_proj):
    """
    """
    intervention_boundaries = torch.clamp(subspace_proj.intervention_boundaries, 1e-3, 1)
    boundary_mask = sigmoid_boundary(
        subspace_proj.intervention_population.repeat(1, 1),
        0.0,
        intervention_boundaries[0] * int(subspace_proj.embed_dim),
        subspace_proj.temperature
    )
    return boundary_mask

def compute_rotation_mask_sparsity(subspace_proj):
    """
    """
    rotation_mask = get_rotation_mask(subspace_proj)
    return (rotation_mask.sum() / rotation_mask.numel()).item()

# from pyvene https://github.com/stanfordnlp/pyvene/blob/main/pyvene/models/interventions.py#L298
class BoundlessRotatedSpaceIntervention(TrainableIntervention, DistributedRepresentationIntervention):

    """Intervention in the rotated space with boundary mask."""

    def __init__(self, embed_dim, torch_dtype, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.rotate_layer = RotateLayer(self.embed_dim, torch_dtype=torch_dtype, init_orth=True)
        # self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.intervention_boundaries = torch.nn.Parameter(
            torch.tensor([0.5], dtype=torch_dtype), requires_grad=True
        )
        self.temperature = torch.nn.Parameter(torch.tensor(50.0, dtype=torch_dtype))
        self.intervention_population = torch.nn.Parameter(
            torch.arange(0, self.embed_dim, dtype=torch_dtype), requires_grad=False
        )

    def get_boundary_parameters(self):
        return self.intervention_boundaries
    
    def get_boundary_sparsity(self):
        intervention_boundaries = torch.clamp(self.intervention_boundaries, 1e-3, 1)
        boundary_mask = sigmoid_boundary(
            self.intervention_population.repeat(1, 1),
            0.0,
            intervention_boundaries[0] * int(self.embed_dim),
            self.temperature,
        )
        
        return boundary_mask.sum() / boundary_mask.numel()
        
        

    def get_temperature(self):
        return self.temperature

    def set_temperature(self, temp: torch.Tensor):
        self.temperature.data = temp

    def set_intervention_boundaries(self, intervention_boundaries):
        self.intervention_boundaries = torch.nn.Parameter(
            torch.tensor([intervention_boundaries]), requires_grad=True
        )
        
    def forward(self, base, source, batch_size):
        # batch_size = base.shape[0]
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        # get boundary
        intervention_boundaries = torch.clamp(self.intervention_boundaries, 1e-3, 1)
        boundary_mask = sigmoid_boundary(
            self.intervention_population.repeat(batch_size, 1),
            0.0,
            intervention_boundaries[0] * int(self.embed_dim),
            self.temperature,
        )

        boundary_mask = (
            torch.ones_like(base)[:,0].to(base.device) * boundary_mask
            # torch.ones(batch_size, device=base.device).unsqueeze(dim=-1) * boundary_mask
        ).unsqueeze(dim=1).expand(base.shape)
        boundary_mask = boundary_mask.to(rotated_base.dtype)
        # interchange
        rotated_output = (
            1.0 - boundary_mask
        ) * rotated_base + boundary_mask * rotated_source
        # inverse output
        output = torch.matmul(rotated_output, self.rotate_layer.weight.T)
        return output.to(base.dtype)

    def __str__(self):
        return f"BoundlessRotatedSpaceIntervention()"
    
    
class RotatedSpaceIntervention(TrainableIntervention, DistributedRepresentationIntervention):

    """Intervention in the rotated space with boundary mask."""

    def __init__(self, embed_dim, intervention_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.intervention_dim = intervention_dim
        self.rotate_layer = RotateLayer(self.embed_dim, init_orth=True)
        # self.rotate_layer = nn.utils.parametrizations.orthogonal(rotate_layer)
        intervention_mask = torch.zeros(self.embed_dim)
        intervention_mask[: self.intervention_dim] = 1
        self.intervention_mask = nn.Parameter(
            intervention_mask.unsqueeze(0), requires_grad=False
        )

    def get_boundary_parameters(self):
        return self.intervention_mask
    
    def get_boundary_sparsity(self):
        return self.intervention_mask.sum() / self.intervention_mask.numel()
    
    def get_temperature(self):
        pass

    def set_temperature(self, temp: torch.Tensor):
        pass

    def set_intervention_boundaries(self, intervention_boundaries):
        self.intervention_boundaries = torch.nn.Parameter(
            torch.tensor([intervention_boundaries]), requires_grad=False
        )
        
    def forward(self, base, source, batch_size):
        # batch_size = base.shape[0]
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        # get boundary
        boundary_mask = self.intervention_mask.repeat(batch_size, 1)

        boundary_mask = (
            torch.ones_like(base)[:,0].to(base.device) * boundary_mask
        ).unsqueeze(dim=1).expand(base.shape)
        boundary_mask = boundary_mask.to(rotated_base.dtype)
        # interchange
        rotated_output = (
            1.0 - boundary_mask
        ) * rotated_base + boundary_mask * rotated_source
        # inverse output
        output = torch.matmul(rotated_output, self.rotate_layer.weight.T)
        return output.to(base.dtype)

    def __str__(self):
        return f"RotatedSpaceIntervention()"
    
    
class LowRankRotatedSpaceIntervention(TrainableIntervention, DistributedRepresentationIntervention):

    """Intervention in the rotated space."""

    def __init__(self, embed_dim, low_rank_dimension, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.rotate_layer = LowRankRotateLayer(self.embed_dim, low_rank_dimension, init_orth=True)
        # self.rotate_layer = nn.utils.parametrizations.orthogonal(rotate_layer)
        self.sparsity = low_rank_dimension / self.embed_dim
        
    def get_boundary_parameters(self):
        return None
    
    def get_boundary_sparsity(self):
        return torch.Tensor([self.sparsity])
    
    def get_temperature(self):
        pass

    def set_temperature(self, temp: torch.Tensor):
        pass

    def set_intervention_boundaries(self, intervention_boundaries):
        pass

    def forward(self, base, source, batch_size=None):
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        
        output = base + torch.matmul(
            (rotated_source - rotated_base), self.rotate_layer.weight.T
        )
        
        return output.to(base.dtype)
    
    def __str__(self):
        return f"LowRankRotatedSpaceIntervention()"
    

class SelectiveLowRankRotatedSpaceIntervention(TrainableIntervention, DistributedRepresentationIntervention):

    """Intervention in the rotated space."""

    def __init__(self, embed_dim, low_rank_dimension, torch_dtype=torch.float32, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.rotate_layer = LowRankRotateLayer(self.embed_dim, low_rank_dimension, init_orth=True)
        # self.rotate_layer = nn.utils.parametrizations.orthogonal(rotate_layer)
        self.mask_projection = HiddenStatesProjectionMLP(
            in_size=self.embed_dim, 
            out_size=low_rank_dimension, 
            torch_dtype=torch_dtype
        )
        
        self.input_layernorm = LlamaRMSNorm(
            hidden_size=self.embed_dim, eps=1e-5
        ).to(dtype=torch_dtype)
        
        # Initialize bias with a large value to make the mask close to 1 initially
        # self.mask_projection.bias.data.fill_(500.0)
        
        self.sparsity = low_rank_dimension / self.embed_dim
        self.temperature = nn.Parameter(torch.tensor(50.0, dtype=torch_dtype), requires_grad=False)
        
    def get_boundary_parameters(self):
        return None
    
    def get_boundary_sparsity(self):
        return torch.Tensor([self.sparsity])
    
    def get_temperature(self):
        return self.temperature

    def set_temperature(self, temp: torch.Tensor):
        self.temperature.data = temp

    def set_intervention_boundaries(self, intervention_boundaries):
        return None

    def forward(self, base, source, hidden_states):
        
        normalized_hidden_state = self.input_layernorm(hidden_states[:, -1, :])
        mask = self.mask_projection(normalized_hidden_state)
        
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        
        mask = torch.sigmoid(mask / self.temperature)        
        mask = mask.unsqueeze(1)
        output = base + torch.matmul(
            mask * (rotated_source - rotated_base), self.rotate_layer.weight.T
        )
        return output.to(base.dtype)
    
    def __str__(self):
        return f"SelectiveLowRankRotatedSpaceIntervention()"
    
    

class ReflectiveLowRankRotatedSpaceIntervention(TrainableIntervention, DistributedRepresentationIntervention):

    """Intervention in the rotated space."""

    def __init__(self, embed_dim, low_rank_dimension, torch_dtype=torch.bfloat16, save_vector=False, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.rotate_layer = LowRankRotateLayer(self.embed_dim, low_rank_dimension, init_orth=True)
        # self.rotate_layer = nn.utils.parametrizations.orthogonal(rotate_layer)
        
        self.rv_proj = HiddenStatesProjectionMLP(
            in_size=self.embed_dim, 
            out_size=self.embed_dim, 
            torch_dtype=torch_dtype
        )
        
        self.input_layernorm = LlamaRMSNorm(
            hidden_size=self.embed_dim, eps=1e-5
        ).to(dtype=torch_dtype)

        self.sparsity = low_rank_dimension / self.embed_dim
        
        self.rvs = []
        self.save_rv = save_vector
        
    def get_boundary_parameters(self):
        return None
    
    def get_boundary_sparsity(self):
        return torch.Tensor([self.sparsity])
    
    def get_temperature(self):
        return None

    def set_temperature(self, temp: torch.Tensor):
        pass

    def set_intervention_boundaries(self, intervention_boundaries):
        return None

    def forward(self, base, source, hidden_states):
        
        normalized_hidden_state = self.input_layernorm(hidden_states[:, -1, :])
        rv = self.rv_proj(normalized_hidden_state)
        rv = rv / torch.norm(rv, dim=-1, keepdim=True)
        if self.save_rv:
            self.rvs.append(rv)
            
        householder = torch.eye(self.embed_dim, device=rv.device, dtype=rv.dtype).unsqueeze(0) - 2 * torch.bmm(rv.unsqueeze(2), rv.unsqueeze(1))
        reflected_weight = torch.matmul(householder, self.rotate_layer.weight.to(rv.dtype))
        
        
        rotated_base = torch.bmm(base, reflected_weight)
        rotated_source = torch.bmm(source, reflected_weight)
        
        output = base + torch.matmul(
            (rotated_source - rotated_base), torch.transpose(reflected_weight, 1, 2)
        )
        return output.to(base.dtype)
    
    def __str__(self):
        return f"ReflectiveLowRankRotatedSpaceIntervention()"