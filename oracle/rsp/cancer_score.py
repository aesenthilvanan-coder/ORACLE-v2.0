"""
cancer_score.py
---------------
Differentiable cancer score function and RSP configuration dataclass.

The CancerScoreFunction maps a gene-expression state x in [0,1]^n to a
scalar cancer score in [0,1].  Higher scores indicate a more cancer-like
state; lower scores indicate normal / reverted phenotype.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RSPConfig:
    """Hyper-parameters and runtime settings for the Reversion Switch Predictor."""

    # Gene space (required; set from GRN node count)
    n_genes: int = 0

    # Model architecture
    hidden_dim: int = 256
    n_hidden: int = 256        # alias kept for compatibility
    n_layers: int = 3          # MLP layers in CancerScoreFunction
    n_gnn_layers: int = 6
    n_attention_heads: int = 8

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 100
    batch_size: int = 64

    # Optimisation / search
    max_perturbations: int = 5
    target_cancer_score: float = 0.3

    # Classification threshold – states with cancer score < this value are
    # considered "normal / reverted".
    normal_threshold: float = 0.3

    # ODE integration
    validation_trajectories: int = 100
    integration_time: float = 100.0
    n_steps: int = 500

    # Checkpoint
    checkpoint_path: str = "./checkpoints/rsp_gnn.pt"


# ---------------------------------------------------------------------------
# Cancer score network
# ---------------------------------------------------------------------------


class CancerScoreFunction(nn.Module):
    """Differentiable 3-layer MLP with residual connections.

    Maps a gene-expression state vector x ∈ [0,1]^n to a scalar cancer
    score in [0,1].

    Architecture
    ------------
    encoder : Linear(n_genes → hidden) → LayerNorm → GELU
              → Linear(hidden → hidden) → LayerNorm → GELU
    residual : Linear(n_genes → hidden)          # skip connection
    head     : Linear(hidden → 64) → GELU → Linear(64 → 1) → Sigmoid

    Forward pass
    ------------
    h = encoder(x) + residual(x)
    score = head(h).squeeze(-1)
    """

    def __init__(self, n_genes_or_config, hidden_dim: int = 256) -> None:
        super().__init__()
        # Accept either RSPConfig or a plain int
        if isinstance(n_genes_or_config, int):
            n_genes = n_genes_or_config
        else:
            cfg = n_genes_or_config
            n_genes = getattr(cfg, "n_genes", 0)
            hidden_dim = getattr(cfg, "hidden_dim", hidden_dim)
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim

        # --- two-block encoder -------------------------------------------
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # --- skip / residual projection -----------------------------------
        self.residual = nn.Linear(n_genes, hidden_dim)

        # --- scoring head -------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Weight initialisation
        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Apply Kaiming normal initialisation to all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute cancer score(s) for a batch of gene-expression states.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, n_genes)`` or ``(n_genes,)``.  Values should
            lie in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch,)`` or scalar – cancer scores in ``[0, 1]``.
        """
        h = self.encoder(x) + self.residual(x)
        return self.head(h).squeeze(-1)

    # ------------------------------------------------------------------
    # Gradient utility
    # ------------------------------------------------------------------

    def gradient_wrt_input(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ∂(cancer_score) / ∂x.

        Positive gradient entries indicate genes whose *increased* expression
        raises the cancer score (candidate repression targets).
        Negative gradient entries identify genes whose *increased* expression
        lowers the cancer score (candidate activation targets).

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(n_genes,)`` or ``(batch, n_genes)``.  Does **not** need
            ``requires_grad`` set beforehand; this method handles that.

        Returns
        -------
        torch.Tensor
            Gradient tensor with the same shape as *x*.
        """
        x_in = x.detach().clone().requires_grad_(True)
        score = self.forward(x_in)

        # Sum over batch if needed so that backward gives per-gene grad
        if score.dim() > 0:
            score = score.sum()

        score.backward()
        return x_in.grad.detach()  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def score_numpy(self, x_np) -> float:
        """Score a single numpy state vector, returning a Python float."""
        import numpy as np

        x_t = torch.tensor(x_np, dtype=torch.float32)
        with torch.no_grad():
            return float(self.forward(x_t).item())

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"CancerScoreFunction(n_genes={self.n_genes}, "
            f"hidden_dim={self.hidden_dim})"
        )
