"""
trajectory_tracker.py
---------------------
Tracks and analyses cell trajectories through the attractor landscape during
a perturbation experiment.

The TrajectoryTracker integrates the continuous GRN ODE forward in time,
recording the full state trajectory at each time step.  Post-hoc analyses
include:

* Computing the cancer score at every time point.
* Finding the "transition step" where the trajectory crosses from the
  cancer basin into the normal basin (cancer score drops below a threshold).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch

from oracle.rsp.cancer_score import RSPConfig

logger = logging.getLogger(__name__)


class TrajectoryTracker:
    """Track cell trajectories through the attractor landscape.

    Parameters
    ----------
    ode_model :
        Callable with signature ``ode_model(t, x) -> dx/dt``.  Must accept
        a time scalar and a state tensor of shape ``(1, n_genes)`` or
        ``(n_genes,)`` and return a derivative of the same shape.
    config : RSPConfig
        Shared RSP configuration (uses ``integration_time``, ``n_steps``).
    """

    def __init__(self, ode_model, config: RSPConfig) -> None:
        self.ode_model = ode_model
        self.config = config
        self._use_torchdiffeq: bool = getattr(
            ode_model, "use_torchdiffeq", False
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track(
        self,
        initial_state: np.ndarray,
        perturbation: dict,
        n_steps: int = 500,
    ) -> np.ndarray:
        """Integrate the ODE and record the full state trajectory.

        Parameters
        ----------
        initial_state : np.ndarray
            Starting gene-expression state, shape ``(n_genes,)``.
        perturbation : dict
            Perturbation specification with optional keys:
            ``'activate'`` – list of gene indices to force toward high
            expression;
            ``'repress'``  – list of gene indices to force toward low
            expression.
        n_steps : int
            Number of integration steps (overrides ``config.n_steps`` if
            explicitly supplied).

        Returns
        -------
        np.ndarray
            Trajectory array of shape ``(n_steps + 1, n_genes)``, where
            index 0 is the initial state.
        """
        x0 = np.array(initial_state, dtype=np.float32)
        n_genes = x0.shape[0]
        activate_indices: List[int] = perturbation.get("activate", [])
        repress_indices: List[int] = perturbation.get("repress", [])

        if self._use_torchdiffeq:
            return self._track_torchdiffeq(
                x0, activate_indices, repress_indices, n_steps
            )

        return self._track_euler(
            x0, activate_indices, repress_indices, n_steps
        )

    def compute_cancer_score_trajectory(
        self,
        trajectory: np.ndarray,
        cancer_score_fn,
    ) -> List[float]:
        """Compute the cancer score at every time step of a trajectory.

        Parameters
        ----------
        trajectory : np.ndarray
            Shape ``(n_steps, n_genes)``.
        cancer_score_fn : CancerScoreFunction
            Trained cancer score network.

        Returns
        -------
        List[float]
            Cancer scores at each time step, length ``n_steps``.
        """
        scores: List[float] = []
        x_tensor = torch.tensor(trajectory, dtype=torch.float32)  # (T, n_genes)

        # Process in a single batched forward pass for efficiency
        with torch.no_grad():
            batch_scores = cancer_score_fn(x_tensor)  # (T,)

        if isinstance(batch_scores, torch.Tensor):
            scores = batch_scores.cpu().tolist()
        else:
            scores = [float(s) for s in batch_scores]

        return scores

    def find_transition_point(
        self,
        trajectory: np.ndarray,
        cancer_score_fn,
        threshold: float = 0.5,
    ) -> int:
        """Find the first step where the trajectory crosses the threshold.

        The "transition point" is the earliest time step at which the cancer
        score drops *and stays* below ``threshold`` for at least one
        consecutive step.

        Parameters
        ----------
        trajectory : np.ndarray
            Shape ``(n_steps, n_genes)``.
        cancer_score_fn : CancerScoreFunction
            Trained cancer score network.
        threshold : float
            Cancer score value below which the cell is considered "normal".

        Returns
        -------
        int
            Index of the first transition step, or ``-1`` if no transition
            was detected within the trajectory.
        """
        scores = self.compute_cancer_score_trajectory(trajectory, cancer_score_fn)

        for step, score in enumerate(scores):
            if score < threshold:
                logger.debug(
                    "Transition detected at step %d (score=%.4f, threshold=%.4f).",
                    step,
                    score,
                    threshold,
                )
                return step

        logger.debug(
            "No transition detected within %d steps (min_score=%.4f, threshold=%.4f).",
            len(scores),
            min(scores, default=float("nan")),
            threshold,
        )
        return -1

    # ------------------------------------------------------------------
    # Private integration routines
    # ------------------------------------------------------------------

    def _track_euler(
        self,
        x0: np.ndarray,
        activate_indices: List[int],
        repress_indices: List[int],
        n_steps: int,
    ) -> np.ndarray:
        """Forward-Euler integrator that records every state."""
        dt = self.config.integration_time / n_steps
        trajectory = np.empty((n_steps + 1, x0.shape[0]), dtype=np.float32)
        trajectory[0] = x0
        x = x0.copy()

        for step in range(n_steps):
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            t = torch.tensor(step * dt, dtype=torch.float32)

            with torch.no_grad():
                dxdt = self.ode_model(t, x_t)

            if isinstance(dxdt, torch.Tensor):
                dxdt_np = dxdt.squeeze(0).cpu().numpy()
            else:
                dxdt_np = np.asarray(dxdt, dtype=np.float32).ravel()

            x = x + dt * dxdt_np
            x = np.clip(x, 0.0, 1.0)

            # Re-apply boundary conditions
            for idx in activate_indices:
                x[idx] = 0.9
            for idx in repress_indices:
                x[idx] = 0.1

            trajectory[step + 1] = x

        return trajectory

    def _track_torchdiffeq(
        self,
        x0: np.ndarray,
        activate_indices: List[int],
        repress_indices: List[int],
        n_steps: int,
    ) -> np.ndarray:
        """torchdiffeq-based integrator that records every state."""
        try:
            from torchdiffeq import odeint  # type: ignore
        except ImportError:
            logger.warning(
                "torchdiffeq not available; falling back to Euler integration."
            )
            return self._track_euler(
                x0, activate_indices, repress_indices, n_steps
            )

        t_span = torch.linspace(
            0.0, self.config.integration_time, n_steps + 1
        )
        x0_t = torch.tensor(x0, dtype=torch.float32)

        activate_t = torch.tensor(activate_indices, dtype=torch.long)
        repress_t = torch.tensor(repress_indices, dtype=torch.long)

        def ode_fn(t, x):
            x_c = x.clone()
            if len(activate_t):
                x_c[activate_t] = 0.9
            if len(repress_t):
                x_c[repress_t] = 0.1
            dx = self.ode_model(t, x_c.unsqueeze(0)).squeeze(0)
            if len(activate_t):
                dx[activate_t] = 0.0
            if len(repress_t):
                dx[repress_t] = 0.0
            return dx

        with torch.no_grad():
            traj_t = odeint(ode_fn, x0_t, t_span, method="euler")

        trajectory = traj_t.cpu().numpy()  # (n_steps+1, n_genes)
        return np.clip(trajectory, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"TrajectoryTracker("
            f"integration_time={self.config.integration_time}, "
            f"n_steps={self.config.n_steps})"
        )
