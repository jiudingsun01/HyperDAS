from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from pyvene import (
    DistributedRepresentationIntervention,
    SourcelessIntervention,
    TrainableIntervention,
)
from pyvene.models.intervention_utils import _do_intervention_by_swap

from .utils import InterventionModuleOutput


class LoReFTHypernetwork(nn.Module):
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
        dtype: torch.dtype = torch.bfloat16,
        weight_sharing: bool = False,
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
            weight_sharing: Whether to share weights on position axis
        """
        super().__init__()
        self.dtype = dtype
        self.weight_sharing = weight_sharing
        self.embed_dim = hidden_size
        self.target_embed_dim = target_hidden_size
        self.num_target_layers = num_target_layers
        self.rank = rank
        self.task_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2, dtype=dtype),
            nn.LayerNorm(hidden_size // 2, eps=rms_norm_eps, dtype=dtype),
        )
        # TODO(sid): add support for padding to position encoder to allow mixing in batch
        self.position_encoder = nn.Sequential(
            nn.Embedding(num_target_positions, hidden_size // 4, dtype=dtype),
            nn.LayerNorm(hidden_size // 4, eps=rms_norm_eps, dtype=dtype),
        )
        # TODO(sid): do embeddings need to be in float32 or bf16 and we cast? Not sure what the convention is
        self.layer_encoder = nn.Sequential(
            nn.Embedding(num_target_layers, hidden_size // 4, dtype=dtype),
            nn.LayerNorm(hidden_size // 4, eps=rms_norm_eps, dtype=dtype),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, dtype=dtype),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size, dtype=dtype),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.weight_head = nn.Linear(
            hidden_size, target_hidden_size * rank, dtype=dtype
        )
        self.rotate_head = nn.Linear(
            hidden_size, target_hidden_size * rank, dtype=dtype
        )

    def forward(
        self,
        task_encoding: torch.Tensor,
        layer_indices: torch.LongTensor,
        position_indices: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            task_encoding: Tensor of shape (batch_size, reft_target_layers, hidden_size) containing task embeddings
            layer_indices: LongTensor of shape (batch_size,) containing target layer indices
            position_indices: LongTensor of shape (batch_size,) containing position indices

        Returns:
            Tuple of (weight_matrix, rotation_matrix) of shape (batch_size, target_hidden_size, num_target_layers)
        """
        n_layers = layer_indices.shape[0]
        bsz, n_positions = position_indices.shape

        if self.weight_sharing:
            # We treat everything as one position
            n_positions = 1
            position_indices = position_indices[:, :1]

        # (B, L, H // 2)
        task_encoding = self.task_encoder(task_encoding[:, :, -1, :])
        # (B, P, H // 4)
        pos_emb = self.position_encoder(position_indices)
        # (B, L, H // 4)
        layer_emb = self.layer_encoder(layer_indices)

        # Expand shapes to match
        pos_emb = pos_emb.unsqueeze(1).expand(-1, n_layers, -1, -1)  # (B, L, P, H // 4)
        layer_emb = (
            layer_emb.unsqueeze(0).unsqueeze(2).expand(bsz, -1, n_positions, -1)
        )  # (B, L, 1, H//4)
        task_encoding = task_encoding.unsqueeze(2).expand(
            -1, -1, n_positions, -1
        )  # (B, L, P, H // 2)

        # (B, L, P, H)
        repr = torch.cat(
            [
                task_encoding,
                layer_emb,
                pos_emb,
            ],
            dim=-1,
        )

        out = self.mlp(repr)
        rotation_matrix = self.rotate_head(out).reshape(
            -1, n_layers, n_positions, self.target_embed_dim, self.rank
        )
        # rotation_matrix, _ = torch.linalg.qr(rotation_matrix, mode="reduced")
        weight_matrix = self.weight_head(out).reshape(
            -1, n_layers, n_positions, self.target_embed_dim, self.rank
        )
        return weight_matrix, rotation_matrix


class LsReFTHypernetwork(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        target_hidden_size: int,
        num_target_layers: int,
        num_target_positions: int,
        rms_norm_eps: float = 1e-6,
        dtype: torch.dtype = torch.bfloat16,
        weight_sharing: bool = False,
    ) -> None:
        """
        ReFT Hypernetwork that generates weight matrices for target model intervention.

        Args:
            hidden_size: Hidden dimension size of the source model
            target_hidden_size: Hidden dimension size of the target model
            num_target_layers: Number of layers in the target model
            num_target_positions: Number of positions to intervene on in the target model
            rank: Rank of the factorized weight matrices
            rms_norm_eps: Epsilon value for layer normalization (default: 1e-6)
            dropout: Dropout probability (default: 0.05)
            weight_sharing: Whether to share weights on position axis
        """
        super().__init__()
        self.dtype = dtype
        self.weight_sharing = weight_sharing
        self.embed_dim = hidden_size
        self.target_embed_dim = target_hidden_size
        self.num_target_layers = num_target_layers
        self.task_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2, dtype=dtype),
            nn.LayerNorm(hidden_size // 2, eps=rms_norm_eps, dtype=dtype),
        )
        # TODO(sid): add support for padding to position encoder to allow mixing in batch
        self.position_encoder = nn.Sequential(
            nn.Embedding(num_target_positions, hidden_size // 4, dtype=dtype),
            nn.LayerNorm(hidden_size // 4, eps=rms_norm_eps, dtype=dtype),
        )
        # TODO(sid): do embeddings need to be in float32 or bf16 and we cast? Not sure what the convention is
        self.layer_encoder = nn.Sequential(
            nn.Embedding(num_target_layers, hidden_size // 4, dtype=dtype),
            nn.LayerNorm(hidden_size // 4, eps=rms_norm_eps, dtype=dtype),
        )
        self.weight_head = nn.Linear(hidden_size, target_hidden_size, dtype=dtype)
        # TODO(sid): does this matter?
        self.act_fn = nn.SiLU()

    def forward(
        self,
        task_encoding: torch.Tensor,
        layer_indices: torch.LongTensor,
        position_indices: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            task_encoding: Tensor of shape (batch_size, reft_target_layers, hidden_size) containing task embeddings
            layer_indices: LongTensor of shape (batch_size,) containing target layer indices
            position_indices: LongTensor of shape (batch_size,) containing position indices

        Returns:
            ReFT-R1 weights of shape (batch_size, target_hidden_size, num_target_layers)
        """
        n_layers = layer_indices.shape[0]
        bsz, n_positions = position_indices.shape

        if self.weight_sharing:
            # We treat everything as one position
            n_positions = 1
            position_indices = position_indices[:, :1]

        # (B, L, H // 2)
        task_encoding = self.task_encoder(task_encoding[:, :, -1, :])
        # (B, P, H // 4)
        pos_emb = self.position_encoder(position_indices)
        # (B, L, H // 4)
        layer_emb = self.layer_encoder(layer_indices)

        # Expand shapes to match
        pos_emb = pos_emb.unsqueeze(1).expand(-1, n_layers, -1, -1)  # (B, L, P, H // 4)
        layer_emb = (
            layer_emb.unsqueeze(0).unsqueeze(2).expand(bsz, -1, n_positions, -1)
        )  # (B, L, 1, H//4)
        task_encoding = task_encoding.unsqueeze(2).expand(
            -1, -1, n_positions, -1
        )  # (B, L, P, H // 2)

        # (B, L, P, H)
        combined_embedding = torch.cat(
            [
                task_encoding,
                layer_emb,
                pos_emb,
            ],
            dim=-1,
        )
        # Run through nonlinearity
        combined_embedding = self.act_fn(combined_embedding)
        # (B, L, P, H)
        weight_matrix = self.weight_head(combined_embedding).reshape(
            -1, n_layers, n_positions, self.target_embed_dim
        )
        # unit norm
        return F.normalize(weight_matrix, dim=-1)


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


class BatchLsReftIntervention(nn.Module):
    """
    ReFT-R1 (LsReFT). Computes \psi(h) = ReLU(Wh) for concept detection
    and \Phi{h} = h + (1/k)*W||TopK{\psi(h)}||_1
    """

    def __init__(self, hidden_dim, top_k, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.top_k = top_k

    def forward(
        self,
        base,
        intervention_positions,
        batch_weights,
        factors=None,
        max_acts=None,
        steering=False,
    ) -> InterventionModuleOutput:
        """
        Performed batched-LsReFT intervention during training.
        base: [B, S, H]
        intervention_positions: [B, P]
        batch_weights: [B, 1, H]
        factor: [B] is a fixed scaling for inference time steering
        steering: whether or not we use training or inference time intervention
        """
        assert intervention_positions.shape[1] == base.shape[1], (
            "only sequence-level interventions supported for now"
        )
        bsz = base.shape[0]
        batch_indices = torch.arange(bsz, device=base.device).unsqueeze(-1)
        # (B, P, H)
        intervention_states = base[batch_indices, intervention_positions]

        if steering:
            # We are in steering mode, so we just need to apply the steering vec
            print(
                "SCALE",
                (20 * max_acts[:, None, None] * factors[:, None, None]).mean().item(),
            )
            batch_steering_vec = (
                20 * max_acts[:, None, None] * factors[:, None, None] * batch_weights
            )
            extra_outputs = {}
        else:
            # (B, P, H) * (B, H, 1)
            detect_latent = torch.relu(
                torch.bmm(intervention_states, batch_weights.transpose(1, 2))
            ).squeeze(-1)
            # (B, K) Sequence level topk
            topk_values, topk_indices = detect_latent.topk(self.top_k, dim=-1)
            non_topk_latents = detect_latent.clone()
            non_topk_latents.scatter_(-1, topk_indices, 0)
            # (B, 1) * (B, 1, H)
            batch_steering_vec = (
                topk_values.mean(dim=-1, keepdim=True).unsqueeze(-1) * batch_weights
            )

            extra_outputs = {
                "detect_latent": detect_latent,
                "non_topk_latents": non_topk_latents,
            }

        # Additive intervention
        intervened_output = base.clone()
        intervened_output[batch_indices, intervention_positions] += batch_steering_vec

        return InterventionModuleOutput(
            mixed_output=intervened_output.to(base.dtype),
            base=base,
            metrics={},
            extra_outputs=extra_outputs,
        )


class LoreftIntervention(
    SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention
):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.hidden_dim = kwargs["embed_dim"]
        self.low_rank_dimension = kwargs["low_rank_dimension"]

        # Initialize with global weights
        rotate_layer = LowRankRotateLayer(
            self.hidden_dim, self.low_rank_dimension, init_orth=True
        )
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.hidden_dim, self.low_rank_dimension
        ).to(kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)

        self.dropout = torch.nn.Dropout(
            kwargs["dropout"] if "dropout" in kwargs else 0.0
        )
        self.act_fn = (
            ACT2FN["linear"]
            if "act_fn" not in kwargs or kwargs["act_fn"] is None
            else ACT2FN[kwargs["act_fn"]]
        )

    def forward(
        self,
        base,
        source=None,
        subspaces=None,
    ) -> InterventionModuleOutput:
        metrics = {}

        # Apply global parameters
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
            self.hidden_dim, overload_w_width, init_orth=True
        ).to(self.learned_source.weight.device)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.rotate_layer.parametrizations.weight[0].base[:, :overload_w_width] = (
            overload_w
        )
        assert (
            torch.allclose(self.rotate_layer.weight.data, overload_w.data) is True
        )  # we must match!

        return


class BatchLoreftIntervention(nn.Module):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)

    Supports batch-specific rotation and weight matrices.
    No meaningful module state - weights are provided at runtime.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.hidden_dim = kwargs["embed_dim"]
        self.act_fn = (
            ACT2FN["linear"]
            if "act_fn" not in kwargs or kwargs["act_fn"] is None
            else ACT2FN[kwargs["act_fn"]]
        )

    def forward(
        self,
        base,
        intervention_positions=None,
        batch_rotation=None,
        batch_weights=None,
        **kwargs,
    ) -> InterventionModuleOutput:
        metrics = {}

        if batch_rotation is None or batch_weights is None:
            raise ValueError(
                "BatchLoreftIntervention requires batch_rotation and batch_weights"
            )

        # batch_rotation: [B, P, H, R]
        # batch_weights: [B, P, H, R]
        # base: [B, S, H]
        batch_size, seq_len, hidden_dim = base.shape
        batch_indices = torch.arange(batch_size, device=base.device).unsqueeze(-1)
        intervention_states = base[batch_indices, intervention_positions].unsqueeze(
            2
        )  # [B, P, 1, H]

        # Apply LoReFT transformation where mask is True
        rotated_states = torch.matmul(intervention_states, batch_rotation)  # [B, P, R]
        learned_states = torch.matmul(intervention_states, batch_weights)  # [B, P, R]

        mixed_states = torch.matmul(
            (self.act_fn(learned_states) - rotated_states),  # [B, P, R]
            batch_rotation.transpose(-2, -1),  # [B, P, R, H]
        )  # [B, P, H]

        # Blend based on intervention position mask
        mixed_output = base.clone()
        mixed_output[batch_indices, intervention_positions] = mixed_states.squeeze()

        return InterventionModuleOutput(
            mixed_output=mixed_output.to(base.dtype), base=base, metrics=metrics
        )


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
            mixed_output=self.dropout(output.to(base.dtype)), base=base, metrics=metrics
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
