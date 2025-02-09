from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from pyvene import (
    DistributedRepresentationIntervention,
    SourcelessIntervention,
    TrainableIntervention,
)

from .utils import InterventionModuleOutput


class ReFTHypernetwork(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        target_hidden_size: int,
        num_target_layers: int,
        num_target_positions: int,
        intermediate_size: int,
        rank: int,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.05,
    ) -> None:
        """
        ReFT Hypernetwork that generates weight and rotation matrices for target model intervention.

        Args:
            hidden_size: Hidden dimension size of the source model
            target_hidden_size: Hidden dimension size of the target model
            num_target_layers: Number of layers in the target model
            num_target_positions: Number of positions to intervene on in the target model
            intermediate_size: Size of intermediate MLP layers
            rank: Rank of the factorized weight matrices
            rms_norm_eps: Epsilon value for layer normalization (default: 1e-6)
            dropout: Dropout probability (default: 0.05)
        """
        super().__init__()
        self.embed_dim = hidden_size
        self.target_embed_dim = target_hidden_size
        self.num_target_layers = num_target_layers
        self.rank = rank
        self.task_encoder = nn.Sequential(
            nn.Embedding(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2, eps=rms_norm_eps),
        )
        self.position_encoder = nn.Sequential(
            nn.Embedding(num_target_positions, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4, eps=rms_norm_eps),
        )
        self.layer_encoder = nn.Sequential(
            nn.Embedding(num_target_layers, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4, eps=rms_norm_eps),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.weight_head = nn.Linear(hidden_size, target_hidden_size * rank)
        self.rotate_head = nn.Linear(hidden_size, target_hidden_size * rank)

    def forward(
        self,
        task_encoding: torch.Tensor,
        layer_indices: torch.LongTensor,
        position_indices: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            task_encoding: Tensor of shape (batch_size, hidden_size) containing task embeddings
            layer_indices: LongTensor of shape (batch_size,) containing target layer indices
            position_indices: LongTensor of shape (batch_size,) containing position indices

        Returns:
            Tuple of (weight_matrix, rotation_matrix) of shape (batch_size, target_hidden_size, num_target_layers)
        """
        _, n_layers = layer_indices.shape
        _, n_positions = position_indices.shape
        # (B, 1, H // 2)
        task_encoding = self.task_encoder(task_encoding).unsqueeze(1)
        # (B, P, H // 4)
        pos_emb = self.position_encoder(position_indices)
        # (B, L, H // 4)
        layer_emb = self.layer_encoder(layer_indices)

        # Expand shapes to match
        task_encoding = task_encoding.expand(-1, n_layers, -1)
        pos_emb = pos_emb.unsqueeze(1).expand(-1, n_layers, -1, -1)
        layer_emb = layer_emb.unsqueeze(2)  # (B, L, 1, H//4)

        # (B, L, P, H)
        repr = torch.cat(
            [
                task_encoding.unsqueeze(2).expand(-1, -1, n_positions, -1),
                layer_emb.expand(-1, -1, n_positions, -1),
                pos_emb,
            ],
            dim=-1,
        )

        out = self.mlp(repr)
        rotation_matrix_unorth = self.rotate_head(out).reshape(
            -1, n_layers, n_positions, self.target_embed_dim, self.rank
        )
        rotation_matrix, _ = torch.linalg.qr(rotation_matrix_unorth, mode="reduced")
        weight_matrix = self.weight_head(out).reshape(
            -1, n_layers, n_positions, self.target_embed_dim, self.rank
        )
        return weight_matrix, rotation_matrix


class HiddenStatesProjectionMLP(nn.Module):
    def __init__(
        self, in_size, out_size, intermediate_size=14336, torch_dtype=torch.float32
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(
            self.in_size, self.intermediate_size, bias=False, dtype=torch_dtype
        )
        self.up_proj = nn.Linear(
            self.in_size, self.intermediate_size, bias=False, dtype=torch_dtype
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.out_size, bias=False, dtype=torch_dtype
        )
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
    SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention
):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)

    Supports batch-specific rotation and weight matrices.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.embed_dim = kwargs["embed_dim"]
        self.low_rank_dimension = kwargs["low_rank_dimension"]

        # Initialize with default weights that can be overridden per batch
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, self.low_rank_dimension, init_orth=True
        )
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, self.low_rank_dimension
        ).to(kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)

        self.dropout = torch.nn.Dropout(
            kwargs["dropout"] if "dropout" in kwargs else 0.0
        )
        self.act_fn = (
            ACT2FN["linear"]
            if "act_fn" not in kwargs or kwargs["act_fn"] is None
            else ACT2FN[kwargs["act_fn"]]
        )

        # Store batch-specific parameters
        self.batch_rotation = None
        self.batch_weights = None

    def set_batch_parameters(
        self, rotation_matrix: torch.Tensor, weight_matrix: torch.Tensor
    ):
        """Set batch-specific rotation and weight matrices.

        Args:
            rotation_matrix: Tensor of shape (batch_size, target_hidden_size, low_rank_dimension)
            weight_matrix: Tensor of shape (batch_size, target_hidden_size, low_rank_dimension)
        """
        self.batch_rotation = rotation_matrix
        self.batch_weights = weight_matrix

    def forward(
        self, base, source=None, subspaces=None, intervention_positions=None
    ) -> InterventionModuleOutput:
        metrics = {}

        if self.batch_rotation is not None and self.batch_weights is not None:
            # self.batch_rotation: [B, P, H, R]
            # self.batch_weights: [B, P, H, R]
            # base: [B, seq_len, H]

            batch_size, seq_len, hidden_dim = base.shape
            position_mask = torch.zeros(
                (batch_size, seq_len), device=base.device, dtype=torch.bool
            )
            batch_indices = torch.arange(batch_size, device=base.device).unsqueeze(-1)
            position_mask[batch_indices, intervention_positions] = True  # [B, seq_len]

            # Apply LoReFT transformation where mask is True
            rotated_states = torch.matmul(base, self.batch_rotation)  # [B, seq_len, R]
            learned_states = torch.matmul(base, self.batch_weights)  # [B, seq_len, R]

            mixed_states = torch.matmul(
                (self.act_fn(learned_states) - rotated_states),  # [B, seq_len, R]
                self.batch_rotation.transpose(-2, -1),  # [B, P, R, H]
            )  # [B, seq_len, H]

            # Blend based on position mask
            mixed_output = torch.where(
                position_mask.unsqueeze(-1),  # [B, seq_len, H]
                mixed_states,
                base,
            )

            mixed_output = self.dropout(mixed_output.to(base.dtype))
        else:
            # Fallback to global parameters
            rotated_base = self.rotate_layer(base)
            mixed_output = torch.matmul(
                (self.act_fn(self.learned_source(base)) - rotated_base),
                self.rotate_layer.weight.T,
            )

        mixed_output = self.dropout(mixed_output.to(base.dtype))
        return InterventionModuleOutput(mixed_output=mixed_output, metrics=metrics)

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
        overload_w = state_dict["rotate_layer"].to(self.learned_source.weight.device)
        overload_w_width = overload_w.shape[-1]
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, overload_w_width, init_orth=True
        ).to(self.learned_source.weight.device)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.rotate_layer.parametrizations.weight[0].base[:, :overload_w_width] = (
            overload_w
        )
        assert (
            torch.allclose(self.rotate_layer.weight.data, overload_w.data) is True
        )  # we must match!

        return


class SelectiveLoreftIntervention(
    SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention
):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, kwargs["low_rank_dimension"], init_orth=True
        )
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]
        ).to(kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(
            kwargs["dropout"] if "dropout" in kwargs else 0.0
        )
        self.act_fn = (
            ACT2FN["linear"]
            if "act_fn" not in kwargs or kwargs["act_fn"] is None
            else ACT2FN[kwargs["act_fn"]]
        )

        self.rv_proj = HiddenStatesProjectionMLP(
            in_size=self.embed_dim,
            out_size=self.embed_dim,
            torch_dtype=kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16,
        )

        self.input_layernorm = LlamaRMSNorm(hidden_size=self.embed_dim, eps=1e-5).to(
            dtype=kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16
        )

    def forward(
        self, base, hidden_states, source=None, subspaces=None
    ) -> InterventionModuleOutput:
        metrics = {}
        normalized_hidden_state = self.input_layernorm(hidden_states[:, -1, :])
        rv = self.rv_proj(normalized_hidden_state)
        rv = rv / torch.norm(rv, dim=-1, keepdim=True)

        householder = torch.eye(
            self.embed_dim, device=rv.device, dtype=rv.dtype
        ).unsqueeze(0) - 2 * torch.bmm(rv.unsqueeze(2), rv.unsqueeze(1))
        reflected_weight = torch.matmul(
            householder, self.rotate_layer.weight.to(rv.dtype)
        )

        rotated_base = torch.bmm(base, reflected_weight)

        output = torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base),
            torch.transpose(reflected_weight, 1, 2),
        )
        return InterventionModuleOutput(
            mixed_output=self.dropout(output.to(base.dtype)), metrics=metrics
        )

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
        overload_w = state_dict["rotate_layer"].to(self.learned_source.weight.device)
        overload_w_width = overload_w.shape[-1]
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, overload_w_width, init_orth=True
        ).to(self.learned_source.weight.device)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.rotate_layer.parametrizations.weight[0].base[:, :overload_w_width] = (
            overload_w
        )
        assert (
            torch.allclose(self.rotate_layer.weight.data, overload_w.data) is True
        )  # we must match!

        return


class SteeringVecLoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.dropout = torch.nn.Dropout(
            kwargs["dropout"] if "dropout" in kwargs else 0.0
        )
        self.act_fn = (
            ACT2FN["linear"]
            if "act_fn" not in kwargs or kwargs["act_fn"] is None
            else ACT2FN[kwargs["act_fn"]]
        )

    def forward(
        self, base, hidden_states, source=None, subspaces=None
    ) -> InterventionModuleOutput:
        metrics = {}
        output = hidden_states[:, -1, :].unsqueeze(1).repeat(1, base.shape[1], 1)
        return InterventionModuleOutput(
            mixed_output=self.dropout(output.to(base.dtype)), metrics=metrics
        )

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        return
