import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CoordNorm(nn.Module):
    """Normalizes 3D coordinates while preserving equivariance."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(coords, dim=-1, keepdim=True).clamp(min=self.eps)
        return coords / norms


class GaussianSmearing(nn.Module):
    """Expands interatomic distances using Gaussian basis functions."""

    def __init__(self, start: float = 0.0, stop: float = 10.0, n_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        self.register_buffer("offset", offset)
        self.coeff = -0.5 / ((stop - start) / n_gaussians) ** 2

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * dist ** 2)


class InvariantMessagePassing(nn.Module):
    """SE(3)-invariant message passing using distance-based features."""

    def __init__(
        self,
        hidden_dim: int,
        n_gaussians: int = 50,
        cutoff: float = 10.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff

        self.smearing = GaussianSmearing(0.0, cutoff, n_gaussians)

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + n_gaussians, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index

        diff = coords[dst] - coords[src]
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8)
        dist_feat = self.smearing(dist.squeeze(-1))
        unit_vec = diff / dist

        hi, hj = h[src], h[dst]
        msg_input = torch.cat([hi, hj, dist_feat], dim=-1)
        msg = self.edge_mlp(msg_input)

        coord_update = self.coord_mlp(msg) * unit_vec

        try:
            from torch_scatter import scatter_sum
            agg_msg = scatter_sum(msg, dst, dim=0, dim_size=h.shape[0])
            agg_coord = scatter_sum(coord_update, dst, dim=0, dim_size=coords.shape[0])
        except ImportError:
            n = h.shape[0]
            agg_msg = torch.zeros(n, self.hidden_dim, device=h.device)
            agg_coord = torch.zeros_like(coords)
            agg_msg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
            agg_coord.scatter_add_(0, dst.unsqueeze(-1).expand_as(coord_update), coord_update)

        h_new = self.norm(h + self.dropout(self.node_mlp(torch.cat([h, agg_msg], dim=-1))))
        coords_new = coords + agg_coord

        return h_new, coords_new


class SE3EquivariantEncoder(nn.Module):
    """Full SE(3)-equivariant encoder for 3D molecular structures."""

    def __init__(
        self,
        in_node_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 6,
        n_gaussians: int = 50,
        cutoff: float = 10.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList([
            InvariantMessagePassing(hidden_dim, n_gaussians, cutoff, dropout)
            for _ in range(n_layers)
        ])
        self.coord_norm = CoordNorm()
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(node_features)

        for layer in self.layers:
            h, coords = layer(h, coords, edge_index)

        h = self.output_norm(h)
        return h, coords

    def encode_with_pooling(
        self,
        node_features: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h, _ = self.forward(node_features, coords, edge_index)

        if batch is None:
            return h.mean(dim=0, keepdim=True)

        try:
            from torch_geometric.nn import global_mean_pool
            return global_mean_pool(h, batch)
        except ImportError:
            n_graphs = batch.max().item() + 1
            pooled = torch.zeros(n_graphs, h.shape[-1], device=h.device)
            counts = torch.zeros(n_graphs, 1, device=h.device)
            pooled.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)
            counts.scatter_add_(0, batch.unsqueeze(-1), torch.ones(h.shape[0], 1, device=h.device))
            return pooled / counts.clamp(min=1)
