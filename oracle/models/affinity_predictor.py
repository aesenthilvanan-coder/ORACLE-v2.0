import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AffinityPredictor(nn.Module):
    """Predicts binding affinity (pKi) between a small molecule and a protein pocket.

    Architecture:
        - Molecule encoder: MPNN over molecular graph
        - Pocket encoder: MLP over pocket descriptor vector
        - Fusion: concatenate + MLP → pKi prediction
    """

    def __init__(
        self,
        mol_node_dim: int = 9,
        mol_edge_dim: int = 4,
        pocket_dim: int = 128,
        hidden_dim: int = 256,
        n_mpnn_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mol_node_dim = mol_node_dim
        self.hidden_dim = hidden_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(mol_node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(mol_edge_dim, hidden_dim // 4),
            nn.GELU(),
        )

        self.mpnn_layers = nn.ModuleList([
            _MPNNLayer(hidden_dim, hidden_dim // 4, dropout)
            for _ in range(n_mpnn_layers)
        ])

        self.pocket_encoder = nn.Sequential(
            nn.Linear(pocket_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        batch: torch.Tensor,
        pocket_feats: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_encoder(node_feats)
        e = self.edge_encoder(edge_feats)

        for layer in self.mpnn_layers:
            h = layer(h, edge_index, e)

        try:
            from torch_geometric.nn import global_mean_pool
            mol_repr = global_mean_pool(h, batch)
        except ImportError:
            n_graphs = int(batch.max().item()) + 1
            mol_repr = torch.zeros(n_graphs, self.hidden_dim, device=h.device)
            counts = torch.zeros(n_graphs, 1, device=h.device)
            mol_repr.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)
            counts.scatter_add_(0, batch.unsqueeze(-1), torch.ones(h.shape[0], 1, device=h.device))
            mol_repr = mol_repr / counts.clamp(min=1)

        pocket_repr = self.pocket_encoder(pocket_feats)
        fused = torch.cat([mol_repr, pocket_repr], dim=-1)
        return self.fusion(fused).squeeze(-1)

    def predict_pki(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        batch: torch.Tensor,
        pocket_feats: torch.Tensor,
    ) -> np.ndarray:
        with torch.no_grad():
            pki = self.forward(node_feats, edge_index, edge_feats, batch, pocket_feats)
        return pki.cpu().numpy()


class _MPNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.GELU(),
        )
        self.upd = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        msg = self.msg(torch.cat([h[src], h[dst], edge_feats], dim=-1))

        agg = torch.zeros_like(h)
        try:
            from torch_scatter import scatter_sum
            agg = scatter_sum(msg, dst, dim=0, dim_size=h.shape[0])
        except ImportError:
            agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)

        h_new = self.upd(torch.cat([h, agg], dim=-1))
        return self.norm(h + self.dropout(h_new))
