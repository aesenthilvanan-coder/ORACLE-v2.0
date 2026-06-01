"""
ternary_complex_predictor.py
-----------------------------
Predicts the stability score of a ternary complex formed by:
  TF (transcription factor) + TCIP (bifunctional molecule) + Recruiter (epigenetic writer/eraser)

The stability score (in [0, 1]) reflects the likelihood that the three
components form a stable, productive ternary complex suitable for epigenetic
reprogramming.

Architecture
------------
1. SE(3)-equivariant encoder: encodes the 3-D complex graph while respecting
   rotational and translational symmetry of the input coordinates.
2. Global mean pooling: collapses per-atom representations to a single
   complex-level embedding.
3. MLP prediction head: maps the complex embedding to a stability score.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import global_mean_pool
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    Data = None  # type: ignore

from oracle.models.shared.se3_equivariant import SE3EquivariantEncoder


# ---------------------------------------------------------------------------
# TernaryComplexPredictor
# ---------------------------------------------------------------------------


class TernaryComplexPredictor(nn.Module):
    """Predicts ternary complex stability from a 3-D structural graph.

    The input ``complex_graph`` represents the assembled TF + TCIP + Recruiter
    complex as a single molecular graph with 3-D atomic coordinates.

    Parameters
    ----------
    hidden_dim : int
        Feature dimensionality throughout the network (default 256).
    n_egnn_layers : int
        Number of EGNN layers in the equivariant encoder (default 4).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_egnn_layers: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Atom-type input projection (10 atom types -> hidden_dim)
        # The SE3EquivariantEncoder expects data.x already in hidden_dim
        # so we embed here first and set in_channels = hidden_dim.
        self.atom_type_embed = nn.Embedding(10, hidden_dim)

        # SE(3)-equivariant encoder
        self.encoder = SE3EquivariantEncoder(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
            n_layers=n_egnn_layers,
        )

        # MLP prediction head: complex embedding -> stability score in [0, 1]
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        nn.init.normal_(self.atom_type_embed.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    def forward(self, complex_graph: Data) -> torch.Tensor:
        """Predict ternary complex stability.

        Parameters
        ----------
        complex_graph : torch_geometric.data.Data
            3-D molecular graph of the assembled complex.  Required fields:

            * ``x``          – atom features.  Accepts:
                               - integer atom-type indices ``(n_atoms,)``
                               - float feature matrix ``(n_atoms, hidden_dim)``
                               - integer matrix ``(n_atoms, 1)``
            * ``pos``        – 3-D coordinates ``(n_atoms, 3)``
            * ``edge_index`` – ``(2, n_edges)``
            * ``batch``      – (optional) batch vector ``(n_atoms,)``

        Returns
        -------
        torch.Tensor
            Stability score in ``[0, 1]``.
            Shape ``(batch_size,)`` or ``()`` for a single un-batched graph.
        """
        device = complex_graph.pos.device

        # --- Embed atom features -----------------------------------------
        raw_x = complex_graph.x
        if raw_x is None:
            # Fallback: all atoms default to Carbon (index 1)
            h = self.atom_type_embed(
                torch.ones(complex_graph.pos.size(0), dtype=torch.long, device=device)
            )
        elif raw_x.dtype in (torch.int32, torch.int64) or (
            raw_x.dim() == 2 and raw_x.size(1) == 1
        ):
            # Integer indices
            h = self.atom_type_embed(raw_x.view(-1).long().clamp(0, 9))
        elif raw_x.dim() == 1:
            h = self.atom_type_embed(raw_x.long().clamp(0, 9))
        else:
            # Float feature matrix already in hidden_dim or larger
            if raw_x.size(1) == self.hidden_dim:
                h = raw_x.float()
            else:
                # Take first column as atom type index
                h = self.atom_type_embed(raw_x[:, 0].long().clamp(0, 9))

        # Temporarily override data.x with embedded features for the encoder
        original_x = complex_graph.x
        complex_graph.x = h

        # --- SE(3)-equivariant encoding -----------------------------------
        node_emb = self.encoder(complex_graph)              # (n_atoms, hidden_dim)

        # Restore original data
        complex_graph.x = original_x

        # --- Global pooling -----------------------------------------------
        batch = complex_graph.batch if complex_graph.batch is not None else \
            torch.zeros(node_emb.size(0), dtype=torch.long, device=device)

        complex_emb = global_mean_pool(node_emb, batch)    # (B, hidden_dim)

        # --- Prediction ---------------------------------------------------
        score = self.prediction_head(complex_emb)          # (B, 1)
        score = score.squeeze(-1)                          # (B,)

        # Squeeze batch dim for single un-batched graph
        if score.numel() == 1 and complex_graph.batch is None:
            score = score.squeeze(0)                       # scalar

        return score

    # ------------------------------------------------------------------

    def predict(self, complex_graph: Data) -> float:
        """Return stability score as a Python float (no-grad)."""
        with torch.no_grad():
            s = self.forward(complex_graph)
            return float(s.item() if s.dim() == 0 else s.mean().item())
