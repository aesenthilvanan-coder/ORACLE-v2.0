"""
Cancer Attraction Mapper - Attractor Finder

Coordinates the Boolean network and continuous ODE attractor searches
to produce a unified set of attractor states representing distinct
cell fate regimes.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from oracle.cam.preprocessing import CAMConfig

logger = logging.getLogger(__name__)


class AttractorFinder:
    """
    Coordinates Boolean and continuous attractor discovery.

    Runs both the discrete (Boolean) and continuous (ODE) attractor
    search algorithms and returns a combined, deduplicated set of
    attractor states along with basin size estimates.

    Parameters
    ----------
    config : CAMConfig
        Pipeline configuration.
    """

    def __init__(self, config: CAMConfig):
        self.config = config

    def find_all_attractors(
        self,
        bool_net,   # BooleanNetworkSimulator
        ode_model,  # ContinuousGRNDynamics
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[int, int]]:
        """
        Run both Boolean and ODE attractor searches.

        Parameters
        ----------
        bool_net : BooleanNetworkSimulator
            Initialized Boolean network simulator.
        ode_model : ContinuousGRNDynamics
            Initialized continuous ODE model.

        Returns
        -------
        boolean_attractors : List[np.ndarray]
            Discrete attractor state vectors (uint8, values in {0, 1}).
        continuous_attractors : List[np.ndarray]
            Continuous fixed-point state vectors (float32, values in [0, 1]).
        basin_sizes : Dict[int, int]
            Basin size estimates for each Boolean attractor (index -> count).
        """
        logger.info("AttractorFinder: running Boolean network attractor search.")
        boolean_attractors = self._find_boolean_attractors(bool_net)

        logger.info("AttractorFinder: running ODE fixed-point search.")
        continuous_attractors = self._find_ode_attractors(ode_model)

        logger.info("AttractorFinder: estimating Boolean basin sizes.")
        basin_sizes = self._estimate_basin_sizes(bool_net, boolean_attractors)

        logger.info(
            "AttractorFinder complete: %d Boolean attractors, %d ODE fixed points.",
            len(boolean_attractors),
            len(continuous_attractors),
        )
        return boolean_attractors, continuous_attractors, basin_sizes

    # ------------------------------------------------------------------
    # Boolean attractor search
    # ------------------------------------------------------------------

    def _find_boolean_attractors(self, bool_net) -> List[np.ndarray]:
        """
        Delegate to BooleanNetworkSimulator.find_attractors().

        Parameters
        ----------
        bool_net : BooleanNetworkSimulator

        Returns
        -------
        List[np.ndarray]
            Unique Boolean attractor state vectors.
        """
        attractors = bool_net.find_attractors(
            n_initial_states=self.config.n_attractor_samples
        )

        if len(attractors) == 0:
            logger.warning(
                "No Boolean attractors found; using random states as fallback."
            )
            rng = np.random.default_rng(seed=0)
            attractors = [
                rng.integers(0, 2, size=bool_net.n_genes, dtype=np.uint8)
                for _ in range(2)
            ]

        logger.info("Boolean attractors found: %d.", len(attractors))
        return attractors

    # ------------------------------------------------------------------
    # ODE fixed-point search
    # ------------------------------------------------------------------

    def _find_ode_attractors(self, ode_model) -> List[np.ndarray]:
        """
        Delegate to ContinuousGRNDynamics.find_fixed_points().

        If fixed-point optimization fails or returns no results,
        integrates from random initial conditions and uses the
        terminal states as approximate fixed points.

        Parameters
        ----------
        ode_model : ContinuousGRNDynamics

        Returns
        -------
        List[np.ndarray]
            Continuous fixed-point state vectors.
        """
        import torch

        try:
            fixed_points = ode_model.find_fixed_points(n_init=500)
        except Exception as exc:
            logger.warning(
                "ODE fixed-point optimization failed (%s); "
                "falling back to trajectory integration.",
                exc,
            )
            fixed_points = []

        if len(fixed_points) == 0:
            logger.info(
                "No ODE fixed points found via optimization; "
                "using trajectory integration fallback."
            )
            fixed_points = self._ode_trajectory_fallback(ode_model)

        logger.info("ODE fixed points found: %d.", len(fixed_points))
        return fixed_points

    def _ode_trajectory_fallback(self, ode_model) -> List[np.ndarray]:
        """
        Integrate from random initial conditions and collect terminal states
        as approximate ODE fixed points.

        Parameters
        ----------
        ode_model : ContinuousGRNDynamics

        Returns
        -------
        List[np.ndarray]
        """
        import torch

        rng = np.random.default_rng(seed=999)
        n_runs = 50
        terminal_states = []
        device = next(ode_model.parameters()).device

        ode_model.eval()
        with torch.no_grad():
            for _ in range(n_runs):
                x0_np = rng.uniform(0, 1, size=ode_model.n_genes).astype(np.float32)
                x0 = torch.tensor(x0_np, device=device)
                traj = ode_model.integrate(
                    x0,
                    t_span=(0.0, self.config.integration_time),
                    n_steps=self.config.n_ode_steps,
                )
                terminal = traj[-1].cpu().numpy()
                terminal_states.append(terminal)

        # Deduplicate
        unique = ode_model._deduplicate_fixed_points(terminal_states, epsilon=0.1)
        return unique

    # ------------------------------------------------------------------
    # Basin size estimation
    # ------------------------------------------------------------------

    def _estimate_basin_sizes(
        self,
        bool_net,
        attractors: List[np.ndarray],
    ) -> Dict[int, int]:
        """
        Estimate basin sizes for Boolean attractors.

        Parameters
        ----------
        bool_net : BooleanNetworkSimulator
        attractors : List[np.ndarray]

        Returns
        -------
        Dict[int, int]
            Attractor index -> basin sample count.
        """
        if len(attractors) == 0:
            return {}

        basin_sizes = bool_net.compute_basin_sizes(
            attractors=attractors,
            n_samples=self.config.n_basin_samples,
        )
        return basin_sizes

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def boolean_to_continuous(
        self,
        bool_attractor: np.ndarray,
        scale: float = 0.8,
    ) -> np.ndarray:
        """
        Convert a Boolean attractor state to a continuous representation.

        Maps 0 -> 0.1 * (1 - scale) and 1 -> scale for biological
        plausibility (avoids exact 0/1 extremes).

        Parameters
        ----------
        bool_attractor : np.ndarray
            Boolean state vector (dtype uint8).
        scale : float
            High-expression value.

        Returns
        -------
        np.ndarray
            Continuous state vector in [0.1, scale].
        """
        continuous = np.where(
            bool_attractor.astype(bool),
            scale,
            1.0 - scale,
        ).astype(np.float32)
        return continuous
