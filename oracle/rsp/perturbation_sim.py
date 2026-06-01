"""
perturbation_sim.py
-------------------
Simulates the effect of transcription factor (TF) perturbations on the
attractor landscape of a continuous GRN ODE model.

Perturbation types
------------------
Activation  – force a gene towards expression level 1.0 (≈ 0.9 in practice).
Repression  – force a gene towards expression level 0.0 (≈ 0.1 in practice).
Partial     – shift expression by a signed delta (used internally).

Workflow
--------
1.  Start N trajectories from the cancer attractor + small Gaussian noise.
2.  Apply perturbations as soft boundary conditions during integration.
3.  Integrate the ODE to find where the perturbed system settles.
4.  Score each final state with the CancerScoreFunction.
5.  Track the fraction of trajectories that reach the "normal" basin.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from oracle.rsp.cancer_score import RSPConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PerturbationResult:
    """Summary statistics for a single perturbation experiment.

    Attributes
    ----------
    mean_cancer_score : float
        Mean cancer score of all final states across trajectories.
    reversion_fraction : float
        Fraction of trajectories whose final cancer score fell below
        ``config.normal_threshold`` (i.e. reverted to normal).
    delta_score : float
        Change in cancer score relative to the baseline (unperturbed) run;
        negative values indicate improvement (reversion).
    genes_activated : List[int]
        Indices of genes that were activated (forced high).
    genes_repressed : List[int]
        Indices of genes that were repressed (forced low).
    final_states : np.ndarray or None
        Optional array of final expression states, shape
        ``(n_trajectories, n_genes)``.
    cancer_scores : List[float]
        Per-trajectory cancer scores at the final time-point.
    """

    mean_cancer_score: float
    reversion_fraction: float
    delta_score: float
    genes_activated: List[int]
    genes_repressed: List[int]
    final_states: Optional[np.ndarray] = field(default=None, repr=False)
    cancer_scores: List[float] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class PerturbationSimulator:
    """Simulate transcription-factor perturbations on an attractor landscape.

    Parameters
    ----------
    ode_model :
        A callable (or ``nn.Module``) with signature
        ``ode_model(t, x) -> dx/dt`` representing the continuous GRN
        dynamics.  Must accept torch Tensors.
    cancer_score_fn : CancerScoreFunction
        Trained differentiable cancer score function.
    cancer_attractor : torch.Tensor
        Reference cancer attractor state, shape ``(n_genes,)``.
    config : RSPConfig
        Shared runtime configuration.
    """

    # Expression levels used for soft-clamped perturbations
    ACTIVATION_LEVEL: float = 0.9
    REPRESSION_LEVEL: float = 0.1
    NOISE_STD: float = 0.05

    def __init__(
        self,
        config_or_ode_model=None,
        cancer_score_fn=None,
        cancer_attractor=None,
        config: Optional[RSPConfig] = None,
    ) -> None:
        # Accept (config) or (ode_model, cancer_score_fn, cancer_attractor, config)
        if isinstance(config_or_ode_model, RSPConfig):
            self.config = config_or_ode_model
            self.ode_model = None
            self.cancer_score_fn = None
            self.cancer_attractor = None
            self.n_genes: int = getattr(self.config, "n_genes", 0)
            self._baseline_score: float = 1.0
        else:
            self.ode_model = config_or_ode_model
            self.cancer_score_fn = cancer_score_fn
            if cancer_attractor is not None:
                self.cancer_attractor = torch.as_tensor(cancer_attractor).float()
            else:
                self.cancer_attractor = None
            self.config = config or RSPConfig()
            self.n_genes = int(self.cancer_attractor.shape[-1]) if self.cancer_attractor is not None else 0
            if self.cancer_score_fn is not None and self.cancer_attractor is not None:
                with torch.no_grad():
                    self._baseline_score = float(
                        self.cancer_score_fn(self.cancer_attractor.unsqueeze(0)).item()
                    )
            else:
                self._baseline_score = 1.0
        logger.debug("Baseline cancer score: %.4f", self._baseline_score)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(
        self,
        initial_state,
        perturbations: Dict[str, str],
        ode_model=None,
        cancer_score_fn=None,
        n_trajectories: int = 100,
        genes: Optional[List[str]] = None,
    ) -> "PerturbationResult":
        """High-level simulate API accepting gene-name perturbation dicts.

        Parameters
        ----------
        initial_state : array-like
            Starting state (cancer attractor).
        perturbations : dict
            {gene_name: 'Activation'|'Repression'} mapping.
        ode_model : optional
            ODE model; uses self.ode_model if not provided.
        cancer_score_fn : optional
            Score function; uses self.cancer_score_fn if not provided.
        n_trajectories : int
            Number of stochastic simulation runs.
        genes : list of str, optional
            Gene list for index lookup; inferred from initial_state length if omitted.
        """
        import numpy as np

        if ode_model is not None:
            self.ode_model = ode_model
        if cancer_score_fn is not None:
            self.cancer_score_fn = cancer_score_fn

        state_arr = np.array(initial_state, dtype=np.float32)
        self.cancer_attractor = torch.tensor(state_arr)
        self.n_genes = len(state_arr)

        if self.cancer_score_fn is not None:
            with torch.no_grad():
                self._baseline_score = float(
                    self.cancer_score_fn(self.cancer_attractor.unsqueeze(0)).item()
                )

        # Build gene-name → index mapping
        gene_names = genes or [str(i) for i in range(self.n_genes)]
        gene_idx = {g: i for i, g in enumerate(gene_names)}

        act_indices = [gene_idx[g] for g, t in perturbations.items()
                       if t.lower().startswith("act") and g in gene_idx]
        rep_indices = [gene_idx[g] for g, t in perturbations.items()
                       if t.lower().startswith("rep") and g in gene_idx]

        return self.simulate_perturbation(act_indices, rep_indices, n_trajectories)

    def simulate_perturbation(
        self,
        genes_to_activate: List[int],
        genes_to_repress: List[int],
        n_trajectories: int = 100,
    ) -> PerturbationResult:
        """Run a perturbation experiment and return summary statistics.

        Parameters
        ----------
        genes_to_activate : List[int]
            Gene indices to force towards high expression.
        genes_to_repress : List[int]
            Gene indices to force towards low expression.
        n_trajectories : int
            Number of stochastic trajectories to simulate.

        Returns
        -------
        PerturbationResult
        """
        final_states: List[np.ndarray] = []
        cancer_scores: List[float] = []
        n_reverted: int = 0

        for traj_idx in range(n_trajectories):
            # 1. Initial condition: cancer attractor + Gaussian noise
            x0 = self._perturbed_init(genes_to_activate, genes_to_repress)

            # 2. Integrate ODE under perturbation
            x_final = self._integrate(x0, genes_to_activate, genes_to_repress)

            # 3. Score final state
            with torch.no_grad():
                score = float(
                    self.cancer_score_fn(
                        torch.tensor(x_final, dtype=torch.float32).unsqueeze(0)
                    ).item()
                )

            final_states.append(x_final)
            cancer_scores.append(score)

            if score < self.config.normal_threshold:
                n_reverted += 1

        mean_score = float(np.mean(cancer_scores))
        reversion_fraction = n_reverted / max(n_trajectories, 1)
        delta_score = mean_score - self._baseline_score

        logger.info(
            "Perturbation act=%s rep=%s | mean_score=%.4f rev_frac=%.3f delta=%.4f",
            genes_to_activate,
            genes_to_repress,
            mean_score,
            reversion_fraction,
            delta_score,
        )

        return PerturbationResult(
            mean_cancer_score=mean_score,
            reversion_fraction=reversion_fraction,
            delta_score=delta_score,
            genes_activated=list(genes_to_activate),
            genes_repressed=list(genes_to_repress),
            final_states=np.array(final_states),
            cancer_scores=cancer_scores,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _perturbed_init(
        self,
        genes_to_activate: List[int],
        genes_to_repress: List[int],
    ) -> np.ndarray:
        """Create a noisy initial condition with perturbations applied."""
        x = self.cancer_attractor.cpu().numpy().copy()
        noise = np.random.normal(0.0, self.NOISE_STD, size=x.shape)
        x = np.clip(x + noise, 0.0, 1.0)

        for idx in genes_to_activate:
            x[idx] = self.ACTIVATION_LEVEL

        for idx in genes_to_repress:
            x[idx] = self.REPRESSION_LEVEL

        return x.astype(np.float32)

    def _apply_perturbation_clamp(
        self,
        x: np.ndarray,
        genes_to_activate: List[int],
        genes_to_repress: List[int],
    ) -> np.ndarray:
        """Re-apply perturbation boundary conditions after each ODE step."""
        for idx in genes_to_activate:
            x[idx] = self.ACTIVATION_LEVEL
        for idx in genes_to_repress:
            x[idx] = self.REPRESSION_LEVEL
        return x

    def _integrate(
        self,
        x0: np.ndarray,
        genes_to_activate: List[int],
        genes_to_repress: List[int],
    ) -> np.ndarray:
        """Integrate the ODE from x0 for config.integration_time steps.

        Uses simple Euler integration with periodic re-clamping of
        perturbed genes.  Falls back to a torchdiffeq-based solver when
        the ode_model exposes a ``use_torchdiffeq`` flag.

        Returns
        -------
        np.ndarray
            Final expression state, shape ``(n_genes,)``.
        """
        use_tde = getattr(self.ode_model, "use_torchdiffeq", False)

        if use_tde:
            return self._integrate_torchdiffeq(
                x0, genes_to_activate, genes_to_repress
            )

        return self._integrate_euler(x0, genes_to_activate, genes_to_repress)

    def _integrate_euler(
        self,
        x0: np.ndarray,
        genes_to_activate: List[int],
        genes_to_repress: List[int],
    ) -> np.ndarray:
        """Simple forward-Euler integration with perturbation clamping."""
        dt = self.config.integration_time / self.config.n_steps
        x = x0.copy()

        for step in range(self.config.n_steps):
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            t = torch.tensor(step * dt, dtype=torch.float32)

            with torch.no_grad():
                dxdt = self.ode_model(t, x_t)

            if isinstance(dxdt, torch.Tensor):
                dxdt_np = dxdt.squeeze(0).cpu().numpy()
            else:
                dxdt_np = np.array(dxdt, dtype=np.float32).ravel()

            x = x + dt * dxdt_np
            x = np.clip(x, 0.0, 1.0)

            # Re-clamp perturbed genes every step to maintain boundary cond.
            x = self._apply_perturbation_clamp(
                x, genes_to_activate, genes_to_repress
            )

        return x.astype(np.float32)

    def _integrate_torchdiffeq(
        self,
        x0: np.ndarray,
        genes_to_activate: List[int],
        genes_to_repress: List[int],
    ) -> np.ndarray:
        """ODE integration using torchdiffeq with perturbation callback."""
        try:
            from torchdiffeq import odeint  # type: ignore
        except ImportError:
            logger.warning(
                "torchdiffeq not available, falling back to Euler integration."
            )
            return self._integrate_euler(
                x0, genes_to_activate, genes_to_repress
            )

        t_span = torch.linspace(
            0.0, self.config.integration_time, self.config.n_steps + 1
        )
        x0_t = torch.tensor(x0, dtype=torch.float32)

        # Build a wrapper that re-clamps perturbed genes inside the ODE func
        activate_t = torch.tensor(genes_to_activate, dtype=torch.long)
        repress_t = torch.tensor(genes_to_repress, dtype=torch.long)

        def ode_fn(t, x):
            x_c = x.clone()
            if len(activate_t):
                x_c[activate_t] = self.ACTIVATION_LEVEL
            if len(repress_t):
                x_c[repress_t] = self.REPRESSION_LEVEL
            dx = self.ode_model(t, x_c.unsqueeze(0)).squeeze(0)
            # Zero out gradients for clamped genes
            if len(activate_t):
                dx[activate_t] = 0.0
            if len(repress_t):
                dx[repress_t] = 0.0
            return dx

        with torch.no_grad():
            trajectory = odeint(ode_fn, x0_t, t_span, method="euler")

        x_final = trajectory[-1].cpu().numpy()
        return np.clip(x_final, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Partial perturbation (shift by delta)
    # ------------------------------------------------------------------

    def simulate_partial_perturbation(
        self,
        gene_deltas: dict,
        n_trajectories: int = 100,
    ) -> PerturbationResult:
        """Apply partial (continuous delta) perturbations.

        Parameters
        ----------
        gene_deltas : dict
            Mapping ``{gene_index: delta}`` where delta shifts expression.
        n_trajectories : int

        Returns
        -------
        PerturbationResult
        """
        activated = [g for g, d in gene_deltas.items() if d > 0]
        repressed = [g for g, d in gene_deltas.items() if d < 0]

        final_states, cancer_scores, n_reverted = [], [], 0

        for _ in range(n_trajectories):
            x0 = self.cancer_attractor.cpu().numpy().copy()
            noise = np.random.normal(0.0, self.NOISE_STD, size=x0.shape)
            x0 = np.clip(x0 + noise, 0.0, 1.0)

            for g_idx, delta in gene_deltas.items():
                x0[g_idx] = float(np.clip(x0[g_idx] + delta, 0.0, 1.0))

            x_final = self._integrate_euler(x0, [], [])

            with torch.no_grad():
                score = float(
                    self.cancer_score_fn(
                        torch.tensor(x_final, dtype=torch.float32).unsqueeze(0)
                    ).item()
                )

            final_states.append(x_final)
            cancer_scores.append(score)
            if score < self.config.normal_threshold:
                n_reverted += 1

        mean_score = float(np.mean(cancer_scores))
        return PerturbationResult(
            mean_cancer_score=mean_score,
            reversion_fraction=n_reverted / max(n_trajectories, 1),
            delta_score=mean_score - self._baseline_score,
            genes_activated=activated,
            genes_repressed=repressed,
            final_states=np.array(final_states),
            cancer_scores=cancer_scores,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def baseline_score(self) -> float:
        """Unperturbed cancer attractor score."""
        return self._baseline_score
