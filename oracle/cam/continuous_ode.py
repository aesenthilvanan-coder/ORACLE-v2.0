"""
Cancer Attraction Mapper - Continuous ODE GRN Dynamics

Implements a continuous-time GRN as a neural ODE:

    dx_i/dt = (-x_i + sigmoid(sum_j W_ij * x_j - theta_i)) / tau_i

where:
    W_ij  = signed weight from gene j to gene i (from GRN)
    theta_i = learned bias / activation threshold
    tau_i   = learned time constant

Fixed points correspond to attractor states of the biological system.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from oracle.cam.preprocessing import CAMConfig

logger = logging.getLogger(__name__)

# Attempt to import torchdiffeq for ODE integration; fall back to Euler method
try:
    from torchdiffeq import odeint as torchdiffeq_odeint

    _TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    _TORCHDIFFEQ_AVAILABLE = False
    logger.warning(
        "torchdiffeq not available; using fixed-step RK4 integrator fallback."
    )


class ContinuousGRNDynamics(nn.Module):
    """
    Continuous ODE model of GRN dynamics.

    Parameters
    ----------
    grn : nx.DiGraph
        Signed, weighted GRN.
    config : CAMConfig
        Pipeline configuration.
    """

    def __init__(self, grn: nx.DiGraph, config: CAMConfig):
        super().__init__()
        self.config = config
        self.grn = grn
        self.genes: List[str] = list(grn.nodes())
        self.n_genes: int = len(self.genes)
        self.gene_idx = {g: i for i, g in enumerate(self.genes)}

        # Build weight matrix W from GRN signs and weights
        W_init = self._build_weight_matrix(grn)

        # Register W as a non-learnable buffer (structural connectivity)
        self.register_buffer("W", torch.tensor(W_init, dtype=torch.float32))

        # Learnable parameters
        self.theta = nn.Parameter(torch.zeros(self.n_genes, dtype=torch.float32))
        self.log_tau = nn.Parameter(
            torch.zeros(self.n_genes, dtype=torch.float32)
        )  # tau = exp(log_tau) ensures positivity

        logger.info(
            "ContinuousGRNDynamics initialized: %d genes.", self.n_genes
        )

    # ------------------------------------------------------------------
    # Weight matrix
    # ------------------------------------------------------------------

    def _build_weight_matrix(self, grn: nx.DiGraph) -> np.ndarray:
        """
        Build the interaction weight matrix W from GRN edge attributes.

        W[i, j] = sign * weight for edge j -> i (j regulates i)

        Parameters
        ----------
        grn : nx.DiGraph

        Returns
        -------
        np.ndarray of shape (n_genes, n_genes)
        """
        n = self.n_genes
        W = np.zeros((n, n), dtype=np.float32)
        for src, tgt, data in grn.edges(data=True):
            if src in self.gene_idx and tgt in self.gene_idx:
                j = self.gene_idx[src]
                i = self.gene_idx[tgt]
                sign = data.get("sign", 1)
                weight = data.get("weight", 1.0)
                W[i, j] = sign * weight
        return W

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the ODE right-hand side.

        dx_i/dt = (-x_i + sigmoid(sum_j W_ij * x_j - theta_i)) / tau_i

        Parameters
        ----------
        t : torch.Tensor
            Current time (scalar, unused except for ODE interface).
        x : torch.Tensor
            Gene expression state, shape (batch, n_genes) or (n_genes,).

        Returns
        -------
        torch.Tensor
            dx/dt, same shape as x.
        """
        tau = torch.exp(self.log_tau)  # (n_genes,)

        # Handle batched and unbatched inputs
        if x.dim() == 1:
            # x: (n_genes,)
            # W: (n_genes, n_genes), theta: (n_genes,)
            input_current = torch.mv(self.W, x) - self.theta
            dx_dt = (-x + torch.sigmoid(input_current)) / tau
        else:
            # x: (batch, n_genes)
            # W: (n_genes, n_genes) -> (n_genes, n_genes)
            input_current = x @ self.W.t() - self.theta.unsqueeze(0)
            dx_dt = (-x + torch.sigmoid(input_current)) / tau.unsqueeze(0)

        return dx_dt

    # ------------------------------------------------------------------
    # ODE integration
    # ------------------------------------------------------------------

    def integrate(
        self,
        x0: torch.Tensor,
        t_span: Tuple[float, float],
        n_steps: int = 100,
    ) -> torch.Tensor:
        """
        Integrate the ODE from x0 over t_span.

        Uses torchdiffeq `odeint` with RK4 method if available,
        otherwise falls back to a fixed-step RK4 implementation.

        Parameters
        ----------
        x0 : torch.Tensor
            Initial state, shape (batch, n_genes) or (n_genes,).
        t_span : (float, float)
            (t_start, t_end).
        n_steps : int
            Number of time steps.

        Returns
        -------
        torch.Tensor
            Trajectory of shape (n_steps+1, batch, n_genes) or
            (n_steps+1, n_genes).
        """
        t = torch.linspace(t_span[0], t_span[1], n_steps + 1, dtype=torch.float32)

        if _TORCHDIFFEQ_AVAILABLE:
            trajectory = torchdiffeq_odeint(
                self,
                x0,
                t,
                method="rk4",
                options={"step_size": (t_span[1] - t_span[0]) / n_steps},
            )
        else:
            trajectory = self._rk4_integrate(x0, t)

        return trajectory

    def _rk4_integrate(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Fixed-step RK4 integration fallback.

        Parameters
        ----------
        x0 : torch.Tensor
            Initial state.
        t : torch.Tensor
            Time points of shape (n_steps+1,).

        Returns
        -------
        torch.Tensor
            Trajectory of shape (n_steps+1, ...).
        """
        states = [x0]
        x = x0.clone()
        for i in range(len(t) - 1):
            dt = t[i + 1] - t[i]
            k1 = self.forward(t[i], x)
            k2 = self.forward(t[i] + dt / 2, x + dt / 2 * k1)
            k3 = self.forward(t[i] + dt / 2, x + dt / 2 * k2)
            k4 = self.forward(t[i + 1], x + dt * k3)
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            states.append(x.clone())
        return torch.stack(states, dim=0)

    # ------------------------------------------------------------------
    # Fixed point finding
    # ------------------------------------------------------------------

    def find_fixed_points(
        self,
        n_init: int = 1000,
        convergence_threshold: float = 1e-6,
        n_optim_steps: int = 500,
        lr: float = 0.01,
    ) -> List[np.ndarray]:
        """
        Find fixed points of the ODE by minimizing ||dx/dt||^2.

        Uses random restarts with Adam optimizer.  Fixed points are
        detected when ||dx/dt||^2 < `convergence_threshold`.

        Parameters
        ----------
        n_init : int
            Number of random initial conditions.
        convergence_threshold : float
            Max ||dx/dt||^2 to declare a fixed point.
        n_optim_steps : int
            Gradient descent steps per restart.
        lr : float
            Adam learning rate.

        Returns
        -------
        List[np.ndarray]
            List of fixed point state vectors (shape: (n_genes,)).
        """
        logger.info("Finding ODE fixed points with %d random restarts.", n_init)

        device = next(self.parameters()).device
        fixed_points_raw: List[np.ndarray] = []
        rng = np.random.default_rng(seed=0)

        self.eval()
        for restart in range(n_init):
            # Random initial condition in [0, 1]^n_genes
            x0_np = rng.uniform(0, 1, size=self.n_genes).astype(np.float32)
            x = nn.Parameter(
                torch.tensor(x0_np, dtype=torch.float32, device=device)
            )
            optimizer = torch.optim.Adam([x], lr=lr)

            for step in range(n_optim_steps):
                optimizer.zero_grad()
                # Clamp x to [0, 1] (biologically meaningful range)
                x_clamped = torch.clamp(x, 0.0, 1.0)
                dxdt = self.forward(torch.tensor(0.0), x_clamped)
                loss = (dxdt ** 2).sum()
                loss.backward()
                optimizer.step()

                if loss.item() < convergence_threshold:
                    break

            with torch.no_grad():
                x_clamped = torch.clamp(x, 0.0, 1.0)
                dxdt = self.forward(torch.tensor(0.0), x_clamped)
                final_loss = (dxdt ** 2).sum().item()

            if final_loss < convergence_threshold:
                fp = x_clamped.detach().cpu().numpy()
                fixed_points_raw.append(fp)

        fixed_points = self._deduplicate_fixed_points(fixed_points_raw)
        logger.info(
            "Found %d fixed points (%d before deduplication).",
            len(fixed_points),
            len(fixed_points_raw),
        )
        return fixed_points

    def _deduplicate_fixed_points(
        self,
        fixed_points: List[np.ndarray],
        epsilon: float = 0.05,
    ) -> List[np.ndarray]:
        """
        Merge fixed points within `epsilon` L2 distance of each other.

        Parameters
        ----------
        fixed_points : list of np.ndarray
        epsilon : float
            Merge radius.

        Returns
        -------
        List[np.ndarray]
            Deduplicated list of fixed points (representatives).
        """
        if len(fixed_points) == 0:
            return []

        unique: List[np.ndarray] = []
        for fp in fixed_points:
            is_duplicate = False
            for ufp in unique:
                dist = np.linalg.norm(fp - ufp)
                if dist < epsilon:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(fp)
        return unique

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def weight_matrix(self) -> np.ndarray:
        """Return the weight matrix W as a numpy array."""
        return self.W.cpu().numpy()
