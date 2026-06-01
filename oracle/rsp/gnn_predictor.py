"""
gnn_predictor.py
----------------
GNN-based predictor for switch candidates in the Reversion Switch Predictor
module.

The GNNSwitchPredictor wraps a SwitchPredictorGNN model and provides a
high-level ``predict`` interface that:

1.  Converts a NetworkX DiGraph (the GRN) and current cancer attractor state
    to a PyTorch Geometric ``Data`` object.
2.  Enriches each node with biologically meaningful features (expression,
    perturbation flags, topological centrality).
3.  Passes the graph through the GNN.
4.  Returns per-gene importance scores alongside global cancer score and
    reversion probability predictions.

Node features (6-D per gene)
-----------------------------
[expression, activate_flag, repress_flag, in_degree, out_degree, betweenness]

Edge features (2-D per regulatory interaction)
-----------------------------------------------
[sign, weight]
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from oracle.rsp.cancer_score import RSPConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal GNN model (defined here to keep the module self-contained)
# ---------------------------------------------------------------------------

try:
    from torch_geometric.data import Data  # type: ignore
    from torch_geometric.nn import (  # type: ignore
        GATv2Conv,
        global_mean_pool,
    )

    _PYG_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYG_AVAILABLE = False
    logger.warning(
        "torch_geometric not found.  GNNSwitchPredictor will run in "
        "degraded (heuristic-only) mode."
    )
    Data = None  # type: ignore


class _GATBlock(nn.Module):
    """Single GATv2 message-passing block with residual + LayerNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        edge_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = GATv2Conv(
            in_channels,
            out_channels // heads,
            heads=heads,
            edge_dim=edge_dim,
            concat=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.skip = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, edge_index, edge_attr):
        h = self.conv(x, edge_index, edge_attr)
        return self.norm(h + self.skip(x))


class SwitchPredictorGNN(nn.Module):
    """Multi-layer GAT network for predicting reversion switch candidates.

    Architecture
    ------------
    * ``n_layers`` stacked GATv2 blocks, each producing ``hidden_dim``-dim
      node embeddings.
    * A global mean-pooling readout head predicts the *graph-level*
      (cancer_score, reversion_prob) pair.
    * A per-node MLP head produces *gene_importance* scores.

    Parameters
    ----------
    node_feat_dim : int
        Dimensionality of input node features (6 by default).
    edge_feat_dim : int
        Dimensionality of input edge features (2 by default).
    hidden_dim : int
        Width of all internal layers.
    n_layers : int
        Number of GATv2 message-passing layers.
    n_heads : int
        Number of attention heads per GATv2 layer.
    dropout : float
        Dropout rate applied inside GATv2Conv.
    """

    def __init__(
        self,
        node_feat_dim: int = 6,
        edge_feat_dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if not _PYG_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for SwitchPredictorGNN."
            )

        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        self.gat_layers = nn.ModuleList(
            [
                _GATBlock(
                    hidden_dim,
                    hidden_dim,
                    heads=n_heads,
                    edge_dim=edge_feat_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # Graph-level head: predicts [cancer_score, reversion_prob]
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )

        # Node-level head: gene importance scores
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, data: "Data") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the full GNN forward pass.

        Returns
        -------
        cancer_score : Tensor, shape (batch,)
        reversion_prob : Tensor, shape (batch,)
        gene_importance : Tensor, shape (n_nodes,)
        """
        x = self.input_proj(data.x)

        for layer in self.gat_layers:
            x = layer(x, data.edge_index, data.edge_attr)
            x = F.gelu(x)

        # Graph-level readout
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        g = global_mean_pool(x, batch)
        graph_out = self.graph_head(g)          # (batch, 2)
        cancer_score = graph_out[:, 0]
        reversion_prob = graph_out[:, 1]

        # Node-level importance
        gene_importance = self.node_head(x).squeeze(-1)  # (n_nodes,)

        return cancer_score, reversion_prob, gene_importance


# ---------------------------------------------------------------------------
# High-level predictor wrapper
# ---------------------------------------------------------------------------


class GNNSwitchPredictor:
    """High-level wrapper around SwitchPredictorGNN.

    Handles graph construction, model loading, and result formatting.

    Parameters
    ----------
    config : RSPConfig
        Shared RSP configuration.
    """

    # Feature dimensionalities (must match SwitchPredictorGNN defaults)
    NODE_FEAT_DIM: int = 6
    EDGE_FEAT_DIM: int = 2

    def __init__(self, config: RSPConfig) -> None:
        self.config = config
        self.model: Optional[SwitchPredictorGNN] = None
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, checkpoint_path: str) -> None:
        """Load a pre-trained SwitchPredictorGNN from a checkpoint file.

        Parameters
        ----------
        checkpoint_path : str
            Path to a ``.pt`` / ``.pth`` checkpoint saved with
            ``torch.save({'model_state_dict': ...})``.
        """
        if not _PYG_AVAILABLE:
            raise ImportError(
                "torch_geometric is required to load GNNSwitchPredictor."
            )

        self.model = SwitchPredictorGNN(
            node_feat_dim=self.NODE_FEAT_DIM,
            edge_feat_dim=self.EDGE_FEAT_DIM,
            hidden_dim=self.config.hidden_dim,
            n_layers=self.config.n_gnn_layers,
            n_heads=self.config.n_attention_heads,
        ).to(self._device)

        if not os.path.isfile(checkpoint_path):
            logger.warning(
                "Checkpoint not found at '%s'.  Running with random weights.",
                checkpoint_path,
            )
            return

        ckpt = torch.load(checkpoint_path, map_location=self._device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        logger.info("Loaded GNNSwitchPredictor from '%s'.", checkpoint_path)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        grn: nx.DiGraph,
        cancer_attractor: np.ndarray,
        perturbation_set: List[Tuple],
    ) -> Dict:
        """Predict cancer score, reversion probability, and gene importances.

        Parameters
        ----------
        grn : nx.DiGraph
            Gene regulatory network. Nodes should be gene names (str); edges
            should carry optional ``sign`` (+1/-1) and ``weight`` attributes.
        cancer_attractor : np.ndarray
            Current cancer attractor expression state, shape ``(n_genes,)``.
        perturbation_set : List[tuple]
            List of ``(gene_name, 'activate'|'repress')`` pairs describing
            the candidate perturbation.

        Returns
        -------
        dict
            Keys: ``'cancer_score'`` (float), ``'reversion_prob'`` (float),
            ``'gene_importance'`` (dict mapping gene name → score).
        """
        if self.model is None:
            # Auto-load from configured checkpoint path
            self.load_model(self.config.checkpoint_path)

        data = self._grn_to_pyg(grn, cancer_attractor, perturbation_set)
        data = data.to(self._device)

        self.model.eval()
        with torch.no_grad():
            cancer_score_t, reversion_prob_t, importance_t = self.model(data)

        cancer_score = float(cancer_score_t.squeeze().item())
        reversion_prob = float(reversion_prob_t.squeeze().item())

        node_list = list(grn.nodes())
        gene_importance = {
            node_list[i]: float(importance_t[i].item())
            for i in range(len(node_list))
        }

        return {
            "cancer_score": cancer_score,
            "reversion_prob": reversion_prob,
            "gene_importance": gene_importance,
        }

    def predict_switches(
        self,
        grn: nx.DiGraph,
        cancer_expression: Dict[str, float],
    ) -> Dict[str, float]:
        """Return per-gene switch importance scores.

        Parameters
        ----------
        grn : nx.DiGraph
            Gene regulatory network.
        cancer_expression : dict
            Gene → expression level in the cancer state.

        Returns
        -------
        dict
            Gene name → importance score (float).
        """
        nodes = list(grn.nodes())
        cancer_attractor = np.array(
            [cancer_expression.get(g, 0.5) for g in nodes], dtype=np.float32
        )
        try:
            result = self.predict(grn, cancer_attractor, [])
            return {g: float(result["gene_importance"].get(g, 0.0)) for g in nodes}
        except Exception:
            # Fallback: return uniform importance scores
            return {g: 1.0 / max(len(nodes), 1) for g in nodes}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _grn_to_pyg(
        self,
        grn: nx.DiGraph,
        cancer_attractor: np.ndarray,
        perturbation_set: List[Tuple],
    ) -> "Data":
        """Convert a GRN DiGraph to a PyTorch Geometric Data object.

        Node features (6-D)
        -------------------
        0. expression       – cancer attractor value for this gene
        1. activate_flag    – 1 if gene is in the activate set
        2. repress_flag     – 1 if gene is in the repress set
        3. in_degree        – normalised in-degree
        4. out_degree       – normalised out-degree
        5. betweenness      – betweenness centrality (normalised by N-1)

        Edge features (2-D)
        -------------------
        0. sign     – regulatory sign (+1 activation, -1 repression), scaled
                      to [0,1]: (sign + 1) / 2
        1. weight   – absolute regulatory weight (clipped to [0,1])
        """
        if not _PYG_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for _grn_to_pyg."
            )

        nodes = list(grn.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        # --- Perturbation flags ------------------------------------------
        activate_genes = {
            gene for gene, ptype in perturbation_set if ptype == "activate"
        }
        repress_genes = {
            gene for gene, ptype in perturbation_set if ptype == "repress"
        }

        # --- Topological features ----------------------------------------
        in_deg = dict(grn.in_degree())
        out_deg = dict(grn.out_degree())
        max_deg = max(max(in_deg.values(), default=1), max(out_deg.values(), default=1), 1)

        try:
            betweenness = nx.betweenness_centrality(grn)
        except Exception:
            betweenness = {node: 0.0 for node in nodes}

        # --- Build node feature matrix -----------------------------------
        node_feats = np.zeros((n, self.NODE_FEAT_DIM), dtype=np.float32)
        for i, node in enumerate(nodes):
            expr = (
                float(cancer_attractor[i])
                if i < len(cancer_attractor)
                else 0.5
            )
            node_feats[i, 0] = expr
            node_feats[i, 1] = 1.0 if node in activate_genes else 0.0
            node_feats[i, 2] = 1.0 if node in repress_genes else 0.0
            node_feats[i, 3] = in_deg.get(node, 0) / max_deg
            node_feats[i, 4] = out_deg.get(node, 0) / max_deg
            node_feats[i, 5] = float(betweenness.get(node, 0.0))

        # --- Build edge index and edge features --------------------------
        edge_src, edge_dst = [], []
        edge_feats = []

        for src, dst, attr in grn.edges(data=True):
            if src not in node_to_idx or dst not in node_to_idx:
                continue
            sign = float(attr.get("sign", 1))  # default: activating
            weight = float(attr.get("weight", 1.0))

            edge_src.append(node_to_idx[src])
            edge_dst.append(node_to_idx[dst])
            edge_feats.append([
                (sign + 1.0) / 2.0,            # map {-1,+1} → {0,1}
                float(np.clip(abs(weight), 0.0, 1.0)),
            ])

        if len(edge_src) == 0:
            # No edges: add self-loops to avoid empty edge_index
            edge_src = list(range(n))
            edge_dst = list(range(n))
            edge_feats = [[0.5, 0.0]] * n

        edge_index = torch.tensor(
            [edge_src, edge_dst], dtype=torch.long
        )
        x = torch.tensor(node_feats, dtype=torch.float32)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        """Return True if a model has been loaded."""
        return self.model is not None

    def __repr__(self) -> str:  # noqa: D105
        status = "loaded" if self.is_loaded() else "not loaded"
        return (
            f"GNNSwitchPredictor(hidden_dim={self.config.hidden_dim}, "
            f"n_layers={self.config.n_gnn_layers}, "
            f"n_heads={self.config.n_attention_heads}, "
            f"model={status})"
        )
