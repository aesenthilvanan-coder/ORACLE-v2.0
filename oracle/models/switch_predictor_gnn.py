import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
import logging

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import global_mean_pool, global_add_pool
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    Data = None  # type: ignore

logger = logging.getLogger(__name__)

NODE_DIM = 7
EDGE_DIM = 3


class GATConvWithEdgeFeatures(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=EDGE_DIM,
            concat=True,
        )
        self.out_dim = out_channels * heads

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.gat(h, edge_index, edge_attr=edge_attr)


class SwitchPredictorGNN(nn.Module):
    """8-layer Graph Attention Network for perturbation prediction."""

    def __init__(self, hidden_dim: int = 256, n_layers: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(NODE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(EDGE_DIM, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, EDGE_DIM),
        )

        heads = 4
        gat_out = hidden_dim // heads
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        in_dim = hidden_dim
        for _ in range(n_layers):
            self.gat_layers.append(GATConvWithEdgeFeatures(in_dim, gat_out, heads=heads, dropout=dropout))
            in_dim = gat_out * heads
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        pool_dim = hidden_dim * 2
        self.score_head = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
        self.importance_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        h = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)

        for gat, norm in zip(self.gat_layers, self.layer_norms):
            h_new = gat(h, data.edge_index, edge_attr=e)
            h = norm(h + h_new)

        importance = self.importance_head(h).squeeze(-1)
        h_mean = global_mean_pool(h, data.batch)
        h_add = global_add_pool(h, data.batch)
        h_global = torch.cat([h_mean, h_add], dim=-1)
        logits = self.score_head(h_global)

        return {
            "cancer_score": torch.sigmoid(logits[:, 0]),
            "reversion_prob": torch.sigmoid(logits[:, 1]),
            "gene_importance": importance,
        }


def build_grn_graph_data(
    grn,
    genes: list,
    cancer_attractor: torch.Tensor,
    normal_attractor: torch.Tensor,
    perturbation_activate: list,
    perturbation_repress: list,
) -> Data:
    """Build PyG Data object for the GNN."""
    import networkx as nx

    n_genes = len(genes)
    gene_idx = {g: i for i, g in enumerate(genes)}

    betweenness = nx.betweenness_centrality(grn, weight="weight")
    max_bc = max(betweenness.values()) if betweenness else 1.0
    in_degree = dict(grn.in_degree())
    out_degree = dict(grn.out_degree())
    max_in = max(in_degree.values()) if in_degree else 1
    max_out = max(out_degree.values()) if out_degree else 1

    act_set = set(perturbation_activate)
    rep_set = set(perturbation_repress)

    node_features = []
    for i, gene in enumerate(genes):
        feat = [
            cancer_attractor[i].item(),
            normal_attractor[i].item(),
            1.0 if i in act_set else 0.0,
            1.0 if i in rep_set else 0.0,
            in_degree.get(gene, 0) / max(max_in, 1),
            out_degree.get(gene, 0) / max(max_out, 1),
            betweenness.get(gene, 0) / max(max_bc, 1e-8),
        ]
        node_features.append(feat)

    x = torch.tensor(node_features, dtype=torch.float32)

    edge_src, edge_dst, edge_feats = [], [], []
    for u, v, data in grn.edges(data=True):
        if u in gene_idx and v in gene_idx:
            src_i = gene_idx[u]
            dst_i = gene_idx[v]
            sign = data.get("sign", 1)
            weight = data.get("weight", 1.0)
            cancer_specific = 1.0 if data.get("source", "") == "integrated" else 0.0
            edge_src.append(src_i)
            edge_dst.append(dst_i)
            edge_feats.append([(sign + 1) / 2.0, float(weight), cancer_specific])

    if len(edge_src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float32)
    else:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
