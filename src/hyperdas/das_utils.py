from abc import abstractmethod
from typing import Any, Dict, Literal, Mapping, Tuple

from src.hyperdas.utils import InterventionModuleOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRMSNorm


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
        self.compute_metrics = kwargs.get("compute_metrics", False)

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


class ChunkedHypernetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        num_chunks,
        intermediate_dim,
        output_dim,
        num_heads=4,
        orthogonal_init=False,
    ):
        super().__init__()
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.proj = nn.ModuleList(
            [
                nn.Linear(
                    input_dim, intermediate_dim * output_dim // num_chunks, bias=True
                )
                for _ in range(num_chunks)
            ]
        )
        self.gate = nn.Linear(input_dim, num_chunks)

        if orthogonal_init:
            for proj in self.proj:
                torch.nn.init.orthogonal_(proj.weight)

    def forward(self, x):
        projections = [proj(x) for proj in self.proj]
        # shape (batch_size, num_chunks, output_dim * intermediate_dim // num_chunks)
        out = torch.stack(projections, dim=1)
        out = out * F.sigmoid(self.gate(x)).unsqueeze(-1)
        return out
    

# from pyvene https://github.com/stanfordnlp/pyvene/blob/main/pyvene/models/interventions.py#L298
class BoundlessRotatedSpaceIntervention(TrainableIntervention, DistributedRepresentationIntervention):

    """Intervention in the rotated space with boundary mask."""

    def __init__(self, embed_dim, torch_dtype, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        rotate_layer = RotateLayer(self.embed_dim, torch_dtype=torch_dtype)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
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
        rotate_layer = RotateLayer(self.embed_dim)
        self.rotate_layer = nn.utils.parametrizations.orthogonal(rotate_layer)
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
        rotate_layer = LowRankRotateLayer(self.embed_dim, low_rank_dimension, init_orth=False)
        self.rotate_layer = nn.utils.parametrizations.orthogonal(rotate_layer)
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
        rotate_layer = LowRankRotateLayer(self.embed_dim, low_rank_dimension, init_orth=False)
        self.rotate_layer = nn.utils.parametrizations.orthogonal(rotate_layer)
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
        rotate_layer = LowRankRotateLayer(self.embed_dim, low_rank_dimension, init_orth=False)
        self.rotate_layer = nn.utils.parametrizations.orthogonal(rotate_layer)
        
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



class TopKSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        vals, indices = torch.topk(input, k, dim=1)
        ctx.save_for_backward(indices, torch.tensor(input.shape))
        return vals, indices

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        indices, input_shape = ctx.saved_tensors
        grad_input = torch.zeros(
            tuple(input_shape), dtype=grad_output.dtype, device=grad_output.device
        )
        grad_input.scatter_(1, indices, grad_output)

        # Compute and store gradient norm
        TopKSTE.last_grad_norm = grad_input.detach().norm().item()

        return grad_input, None

    @staticmethod
    def get_last_grad_norm():
        return getattr(TopKSTE, "last_grad_norm", None)


class QuasiProjectiveIntervention(
    TrainableIntervention, DistributedRepresentationIntervention
):
    """Intervention via (ridge) quasi-projection onto the trained space."""

    # Order of operations:
    # (1) Editor activations @ encoder matrix -> we get an encoder score per each element of the dictonary. This is nn.Linear with bias
    # (2) encoder scores -> topk dictionary elements
    # (3) topk -> select and mulitply, yielding [dictionary_element * encoder_score] for only the selected columns
    # (4) perform quasi-projection (i.e, ridge regression) interchange on the selected and scaled columns

    # side note: instead of multiplying those columns by i:
    # There is an equivalent form where we can keep columns X fixed, pre-compute X^T X elements, and then replace lambda* I with:  diag(score^2)

    def __init__(
        self,
        embed_dim,
        dict_size,
        top_k_parameter,
        lambda_parameter,
        epsilon=1e-6,
        importance_power=-2,
        torch_dtype=torch.bfloat16,
        return_penalty=True,
        ridge_parameterization: Literal[
            "inv_alpha", "ste", "sigmoid", "softmax"
        ] = "inv_alpha",
        selection_mechanism: Literal[
            "full", "topk", "dynamic", "dynamic_attn"
        ] = "full",
        scoring_dimension: int = 1,
        orthogonal_init=False,
        hat_matrix=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dict_size = dict_size
        self.top_k_parameter = top_k_parameter
        self.scoring_dimension = scoring_dimension
        self.hat_matrix = hat_matrix

        # note: you could technically set top_k_parameter equal to full dictionary, it'd just be more expensive computationally
        # and you might want more fall-off
        self.lambda_parameter = lambda_parameter
        self.importance_power = importance_power
        self.epsilon = epsilon
        self.ridge_parameterization = ridge_parameterization
        self.selection_mechanism = selection_mechanism
        self.return_penalty = return_penalty and selection_mechanism != "dynamic"

        assert ridge_parameterization in [
            "inv_alpha",
            "topk_ste",
            "sigmoid",
            "softmax",
            None,
        ], "Invalid ridge_parameterization"

        self.feature_dim = embed_dim

        self.edit_instruction_encodings = nn.Sequential(
            nn.Linear(
                in_features=embed_dim,
                out_features=scoring_dimension
                if "dynamic" in selection_mechanism
                else dict_size,
                bias=True,
            ).to(dtype=torch_dtype),
            nn.ReLU(),  # NOTE: can we use softplus instead of eps down below?
        )

        if selection_mechanism == "topk" or selection_mechanism == "full":
            # Create a dict_size * embed_dim embedding matrix
            self.dictionary = nn.Embedding(
                num_embeddings=dict_size, embedding_dim=embed_dim
            )
        elif selection_mechanism == "dynamic_attn":
            self.dictionary = ChunkedHypernetwork(
                scoring_dimension,
                dict_size,
                dict_size,
                embed_dim,
                orthogonal_init=orthogonal_init,
            )
        elif selection_mechanism == "dynamic":
            self.dictionary = nn.Linear(
                scoring_dimension, dict_size * embed_dim, bias=False
            )
            if orthogonal_init:
                torch.nn.init.orthogonal_(self.dictionary.weight)

        self.dictionary = self.dictionary.to(dtype=torch_dtype)

        self.penalty = None

        self.input_layernorm = LlamaRMSNorm(hidden_size=self.embed_dim, eps=1e-5).to(
            dtype=torch_dtype
        )
        self.base_layernorm = LlamaRMSNorm(hidden_size=self.embed_dim, eps=1e-5).to(
            dtype=torch_dtype
        )
        self.source_layernorm = LlamaRMSNorm(hidden_size=self.embed_dim, eps=1e-5).to(
            dtype=torch_dtype
        )

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        """Load a state dictionary with compatibility checks.

        Args:
            state_dict: The state dictionary to load
            strict: If True, raises error on missing keys
            assign: If True, directly assigns tensor values instead of copying
        """
        # First try normal loading
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except Exception as e:
            # Check if this is a ReflectDAS checkpoint
            if "rotate_layer.parametrizations.weight.original" in state_dict:
                # Get the rotation matrix dimensions
                reflect_weight = state_dict[
                    "rotate_layer.parametrizations.weight.original"
                ]
                reflect_dim = reflect_weight.shape[1]

                assert self.dict_size == reflect_dim

                # Initialize new dictionary weights (must be float32)
                new_dict_weights = torch.empty_like(
                    self.dictionary.weight,
                    dtype=self.dictionary.weight.dtype,
                )

                # Copy over the reflection weights to first reflect_dim columns
                if self.selection_mechanism in ["full", "topk"]:
                    new_dict_weights[:reflect_dim] = reflect_weight.T
                elif self.selection_mechanism == "dynamic":
                    new_dict_weights[:] = (
                        reflect_weight.flatten()
                        .unsqueeze(-1)
                        .expand(-1, self.scoring_dimension)
                    )
                else:
                    raise ValueError("Dictionary size and loaded weight mismatch")

                self.dictionary.weight.data.copy_(new_dict_weights)
            else:
                raise e

    def gradient_norms(self) -> Dict[str, float]:
        metrics = {}
        # Compute mean grad norm for edit_instruction_encodings
        edit_instruction_grad_norms = []
        for param in self.edit_instruction_encodings[0].parameters():
            if param.grad is not None:
                edit_instruction_grad_norms.append(param.grad.detach().norm().item())
        if edit_instruction_grad_norms:
            metrics["grad_norm/edit_instruction_encodings"] = sum(
                edit_instruction_grad_norms
            ) / len(edit_instruction_grad_norms)

        # Compute mean grad norm for basis_dictionary
        basis_dictionary_grad_norms = []
        for param in self.dictionary.parameters():
            if param.grad is not None:
                basis_dictionary_grad_norms.append(param.grad.detach().norm().item())
        if basis_dictionary_grad_norms:
            metrics["grad_norm/basis_dictionary"] = sum(
                basis_dictionary_grad_norms
            ) / len(basis_dictionary_grad_norms)
        if self.ridge_parameterization == "topk_ste":
            metrics["grad_norm/topk_ste"] = TopKSTE.get_last_grad_norm()
        return metrics

    def get_penalty(self):
        if self.penalty is None:
            return 0.0
        return self.penalty

    def zero_penalty(self):
        self.penalty = None

    def get_boundary_parameters(self):
        return None

    def get_boundary_sparsity(self):
        return torch.Tensor([self.dict_size / self.embed_dim])

    def get_temperature(self):
        return None

    def set_temperature(self, temp: torch.Tensor):
        pass

    def set_intervention_boundaries(self, intervention_boundaries):
        return None

    def compute_closeform_ridge(self, X, Y, importance_scores, importance_power=-2):
        # X: batch x k x d_embed
        # Y: batch x seq x d_embed
        # importance_scores: batch x k

        metrics = {}

        if (
            self.ridge_parameterization
            and self.ridge_parameterization != "topk_ste"
            and self.selection_mechanism
            in [
                "full",
                "topk",
            ]
        ):
            if self.ridge_parameterization == "inv_alpha":
                # We add an epsilon for instability prevention
                # denominator scores will be a component inside the matrix inversion
                # Note that alpha < 0 implies that denominator_scores_i is low for the most important features
                denominator_scores = torch.pow(
                    importance_scores + self.epsilon, importance_power
                )  # batch, num_active_features
            elif self.ridge_parameterization == "sigmoid":
                denominator_scores = torch.sigmoid(importance_scores)
            elif self.ridge_parameterization == "softmax":
                denominator_scores = importance_scores.softmax(-1)
        else:
            denominator_scores = None

        # Compute the ridge regression solution
        XTX = torch.matmul(
            X, X.transpose(-2, -1)
        )  # XTX: (batch x num_active_features x d_embed) * (batch x d_embed x num_active_features)
        if not self.hat_matrix:
            XTY = torch.matmul(X, Y.transpose(-2, -1))  # XTY: batch x d_embed x seq
        # diag_denominator_scores = torch.diag_embed(denominator_scores)  #diag_denominator_scores: batch x num_active_features x num_active_features
        if (
            "dynamic" in self.selection_mechanism
            or self.ridge_parameterization == "topk_ste"
        ):
            # Unmodified ridge formulation
            regularized_XTX = (
                XTX
                + self.lambda_parameter
                * torch.eye(XTX.shape[1], device=XTX.device)[None, :, :]
            )  # regularized_XTX:
        else:
            regularized_XTX = XTX + torch.diag_embed(
                denominator_scores
            )  # regularized_XTX:

        # Cast regularized_XTX and XTY to float32
        regularized_XTX = regularized_XTX.to(torch.float32)
        if not self.hat_matrix:
            XTY = XTY.to(torch.float32)

        def solve_single(A, b):
            # Compute Cholesky decomposition
            L = torch.linalg.cholesky(A)
            # Solve L @ y = b for y (forward substitution)
            # Note: solve_triangular takes (A, B) order
            y = torch.linalg.solve_triangular(L, b, upper=False)
            # Solve L.T @ x = y for x (back substitution)
            x = torch.linalg.solve_triangular(L.transpose(-1, -2), y, upper=True)

            return x

        if self.hat_matrix:
            # Solve for beta' instead of beta to compute df
            ridge_coeffs = torch.vmap(solve_single, in_dims=(0, 0))(regularized_XTX, X)
        else:
            ridge_coeffs = torch.vmap(solve_single, in_dims=(0, 0))(
                regularized_XTX, XTY
            )

        ridge_coeffs = ridge_coeffs.to(X.dtype)

        if self.compute_metrics and self.training:
            if denominator_scores is not None:
                # Compute mean, min, max of the denominator score vector
                metrics["denominator_scores_mean"] = denominator_scores.mean().item()
                metrics["denominator_scores_min"] = denominator_scores.min().item()
                metrics["denominator_scores_max"] = denominator_scores.max().item()
            metrics["importance_scores_norms"] = (
                importance_scores.norm(dim=-1).mean().item()
            )

            # effective dimensionality from sum(trace(eig(H))^2 / (trace(eig(H))^2 + lambda))
            # for regression hat matrix H
            if self.hat_matrix:
                hat_matrix = torch.bmm(X, ridge_coeffs.transpose(-2, -1))
                trace_matrix = hat_matrix.diagonal(offset=0, dim1=-1, dim2=-2)
                with torch.no_grad():
                    metrics["effective_dim"] = trace_matrix.sum(-1).mean().item()

        if self.hat_matrix:
            # beta = beta' @ Y
            ridge_coeffs = torch.bmm(ridge_coeffs, Y.transpose(-2, -1))

        # Multiply thru by X
        predictions = torch.matmul(ridge_coeffs.transpose(-2, -1), X)

        return predictions, metrics

    def forward(self, base, source, hidden_states, return_basis=False):
        metrics = {}
        # Base:     batch x seq x d_embed
        # Source:   batch x seq x d_embed
        # Hidden:   batch x instruction_seq x d_embed
        normalized_hidden_state = self.input_layernorm(
            hidden_states[:, -1, :]
        )  # normalized_hidden_state: batch x d_embed
        dictionary_encodings = self.edit_instruction_encodings(
            normalized_hidden_state
        )  # dictionary_encodings: batch x (d_embed or scoring_dimension)

        # Normalize base and source prior to regression
        normalized_base = self.base_layernorm(base)
        normalized_source = self.source_layernorm(source)

        # Perform top-k index selection
        # top_k_indices: batch x k; top_k_values: batch x k
        if self.selection_mechanism == "topk":
            if self.ridge_parameterization == "topk_ste":
                top_k_values, top_k_indices = TopKSTE.apply(
                    dictionary_encodings, self.top_k_parameter
                )
            else:
                top_k_values, top_k_indices = torch.topk(
                    dictionary_encodings, self.top_k_parameter, dim=-1
                )
        elif (
            self.selection_mechanism == "full" or "dynamic" in self.selection_mechanism
        ):
            top_k_values = dictionary_encodings

        # Remove indices where the value is less than zero
        # positive_mask = top_k_values > 0
        # top_k_values = top_k_values[positive_mask]
        # top_k_indices = top_k_indices[positive_mask]

        # Select rows of the dictionary according to top_k_indices
        if self.selection_mechanism == "topk":
            selected_dictionary = self.dictionary(top_k_indices)
        elif self.selection_mechanism == "full":
            selected_dictionary = self.dictionary(
                torch.arange(0, self.dict_size, device=top_k_values.device)
                .unsqueeze(0)
                .repeat(base.shape[0], 1)
            )
        elif "dynamic" in self.selection_mechanism:
            selected_dictionary = self.dictionary(top_k_values).reshape(
                -1, self.dict_size, self.embed_dim
            )

        base_interchange, base_metrics = self.compute_closeform_ridge(
            selected_dictionary,
            normalized_base,
            top_k_values,
            importance_power=self.importance_power,
        )
        source_interchange, source_metrics = self.compute_closeform_ridge(
            selected_dictionary,
            normalized_source,
            top_k_values,
            importance_power=self.importance_power,
        )

        output = base + (source_interchange - base_interchange)

        if self.compute_metrics and self.training:
            metrics.update({f"source_{k}": v for k, v in source_metrics.items()})
            metrics.update({f"base_{k}": v for k, v in base_metrics.items()})

            with torch.no_grad():
                metrics["source_interchange_norm"] = source_interchange.norm().item()
                metrics["base_interchange_norm"] = base_interchange.norm().item()
                metrics["intervention_norm"] = (
                    (source_interchange - base_interchange).norm().item()
                )
                metrics["dictionary_norm"] = top_k_values.norm().item()

                # Plot rank of generated basis
                metrics["basis_rank"] = (
                    torch.linalg.matrix_rank(selected_dictionary.float())
                    .float()
                    .mean()
                    .item()
                )

                # Intervention Directional Change
                metrics["angular_change"] = (
                    torch.acos(
                        F.cosine_similarity(base, output).clamp(-1 + 1e-6, 1 - 1e-6)
                    )
                    .mean()
                    .item()
                )

        if self.return_penalty:
            if self.ridge_parameterization == "inv_alpha":
                penalty = torch.mean(
                    self.lambda_parameter
                    / (self.lambda_parameter + top_k_values**self.importance_power)
                )
            elif self.ridge_parameterization == "sigmoid":
                penalty = torch.mean(
                    self.lambda_parameter
                    / (self.lambda_parameter + top_k_values.sigmoid())
                )
            elif self.ridge_parameterization == "softmax":
                penalty = torch.mean(
                    self.lambda_parameter
                    / (self.lambda_parameter + top_k_values.softmax(-1))
                )
            else:
                penalty = None

            if penalty is not None:
                metrics["lambda_penalty"] = (
                    penalty.item() if isinstance(penalty, torch.Tensor) else penalty
                )

            self.penalty = penalty

        # penalty is sensitive to lambda_parameter, and it controls how much the solutions are influenced by each dimension
        # ...in one of the limits, as you tune up lambda_parameter really big or small, you should get negligible interchange
        # (check this! as a sanity-check!)
        out = InterventionModuleOutput(
            mixed_output=output.to(base.dtype), metrics=metrics
        )
        if return_basis:
            out.basis = selected_dictionary.to(base.dtype)
        return out

    def __str__(self):
        return f"QuasiProjectedIntervention(top_k={self.top_k_parameter}, importance_power={self.importance_power}, lambda_parameter={self.lambda_parameter}, return_penalty={self.return_penalty})"