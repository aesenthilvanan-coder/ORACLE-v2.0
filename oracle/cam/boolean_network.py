"""
Cancer Attraction Mapper - Boolean Network Simulator

Implements asynchronous Boolean network dynamics on the inferred GRN.

Gene states s_i in {0, 1}.  Update rule:
    s_i(t+1) = f_i(s_j : j -> i)

where f_i uses threshold-based majority logic:
    - Activating inputs (sign=+1) push state toward 1
    - Repressing inputs (sign=-1) push state toward 0
    - If act_sum > rep_sum  -> 1
    - If rep_sum > act_sum  -> 0
    - If tied               -> keep current state (asynchronous update)

Attractors are identified by sampling random initial states and running
trajectories until a fixed point or limit cycle is detected.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from oracle.cam.preprocessing import CAMConfig

logger = logging.getLogger(__name__)


class BooleanNetworkSimulator:
    """
    Asynchronous Boolean network dynamics on a signed GRN.

    Parameters
    ----------
    grn : nx.DiGraph
        Signed GRN.  Each edge must have a 'sign' attribute (+1 / -1).
    config : CAMConfig
        Pipeline configuration.
    """

    def __init__(self, grn: nx.DiGraph, config: CAMConfig):
        self.grn = grn
        self.config = config

        # Ordered list of genes (node names)
        self.genes: List[str] = list(grn.nodes())
        self.n_genes: int = len(self.genes)

        # Map gene name -> integer index
        self.gene_idx: Dict[str, int] = {g: i for i, g in enumerate(self.genes)}

        # Precompute update functions for each gene
        self.update_funcs: List[Callable] = self._build_update_functions()

        logger.info(
            "BooleanNetworkSimulator initialized: %d genes, %d edges.",
            self.n_genes,
            grn.number_of_edges(),
        )

    # ------------------------------------------------------------------
    # Build update functions
    # ------------------------------------------------------------------

    def _build_update_functions(self) -> List[Callable]:
        """
        Pre-build a Boolean threshold update function for each gene.

        Returns
        -------
        List of callables, one per gene in self.genes order.
        Each callable signature: f(state: np.ndarray) -> int
        """
        funcs = []
        for gene in self.genes:
            # Collect all predecessors (regulators) of this gene
            regulators = []
            for pred in self.grn.predecessors(gene):
                sign = self.grn[pred][gene].get("sign", 1)
                weight = self.grn[pred][gene].get("weight", 1.0)
                pred_idx = self.gene_idx[pred]
                regulators.append((pred_idx, sign, weight))

            func = self._make_threshold_func(regulators)
            funcs.append(func)
        return funcs

    def _make_threshold_func(
        self,
        regulators: List[Tuple[int, int, float]],
    ) -> Callable:
        """
        Return a closure that implements the Boolean threshold update.

        For a gene with no regulators, the function returns the current
        state (self-sustaining / no external drive).

        Parameters
        ----------
        regulators : list of (regulator_index, sign, weight)

        Returns
        -------
        Callable : f(state: np.ndarray) -> int
            state[i] is the current Boolean state of gene i.
        """
        if len(regulators) == 0:
            # No regulators: keep current state
            def no_reg_func(state: np.ndarray, gene_idx: int) -> int:
                return int(state[gene_idx])

            return no_reg_func

        # Capture regulators in closure
        _regs = regulators

        def threshold_func(state: np.ndarray, gene_idx: int) -> int:
            """
            act_sum = sum of weights of active activating regulators
            rep_sum = sum of weights of active repressing regulators
            result  = 1 if act_sum > rep_sum else 0 if rep_sum > act_sum else current
            """
            act_sum = 0.0
            rep_sum = 0.0
            for (pred_idx, sign, weight) in _regs:
                if state[pred_idx] == 1:
                    if sign > 0:
                        act_sum += weight
                    else:
                        rep_sum += weight
            if act_sum > rep_sum:
                return 1
            elif rep_sum > act_sum:
                return 0
            else:
                return int(state[gene_idx])  # tie: maintain current state

        return threshold_func

    # ------------------------------------------------------------------
    # Attractor finding
    # ------------------------------------------------------------------

    def find_attractors(
        self,
        n_initial_states: int = 10000,
    ) -> List[np.ndarray]:
        """
        Find Boolean attractors by random trajectory sampling.

        Runs `n_initial_states` random initial conditions and collects
        states where trajectories converge.  Duplicate attractors
        (identical state vectors) are de-duplicated.

        Parameters
        ----------
        n_initial_states : int
            Number of random initial states to sample.

        Returns
        -------
        List[np.ndarray]
            List of unique attractor state vectors (dtype uint8).
        """
        logger.info(
            "Finding Boolean attractors: %d random initial states.", n_initial_states
        )
        n_samples = min(n_initial_states, self.config.n_attractor_samples)
        rng = np.random.default_rng(seed=42)

        attractor_set: Set[bytes] = set()
        attractors: List[np.ndarray] = []

        def _try_state(state: np.ndarray) -> None:
            terminal, _ = self._run_trajectory(state, max_steps=self.config.max_trajectory_steps)
            if terminal is not None and self._is_fixed_point(terminal):
                key = terminal.tobytes()
                if key not in attractor_set:
                    attractor_set.add(key)
                    attractors.append(terminal.copy())

        # Always test all-zeros and all-ones first; these are often valid fixed points
        # and ensure at least one attractor is found when n_initial_states is small.
        _try_state(np.zeros(self.n_genes, dtype=np.uint8))
        _try_state(np.ones(self.n_genes, dtype=np.uint8))

        for _ in range(n_samples):
            initial_state = rng.integers(0, 2, size=self.n_genes, dtype=np.uint8)
            _try_state(initial_state)

        logger.info("Found %d unique Boolean attractors.", len(attractors))
        return attractors

    def _is_fixed_point(self, state: np.ndarray) -> bool:
        """Return True if every gene's update function maps back to the current state."""
        for i in range(self.n_genes):
            if self.update_funcs[i](state, i) != state[i]:
                return False
        return True

    def _run_trajectory(
        self,
        initial_state: np.ndarray,
        max_steps: int = 1000,
        rng=None,
    ) -> tuple:
        """
        Run an asynchronous Boolean trajectory until convergence.

        Returns
        -------
        tuple (np.ndarray, int)
            The terminal attractor state and the number of steps taken.
        """
        state = initial_state.copy().astype(np.uint8)
        if rng is None:
            # Derive a deterministic seed from the initial state to guarantee
            # reproducibility without requiring callers to manage RNG state.
            seed = int(np.sum(state.astype(np.uint64) * np.arange(1, len(state) + 1, dtype=np.uint64)) % (2**31))
            rng = np.random.default_rng(seed)
        n = self.n_genes
        no_change_count = 0
        threshold = n * 3

        for step in range(max_steps):
            gene_to_update = int(rng.integers(0, n))
            new_val = self.update_funcs[gene_to_update](state, gene_to_update)
            if new_val != state[gene_to_update]:
                state[gene_to_update] = new_val
                no_change_count = 0
            else:
                no_change_count += 1
                if no_change_count >= threshold:
                    return state, step + 1

        return state, max_steps

    # ------------------------------------------------------------------
    # Basin size estimation
    # ------------------------------------------------------------------

    def compute_basin_sizes(
        self,
        attractors: List[np.ndarray],
        n_samples: int = 50000,
    ) -> Dict[int, int]:
        """
        Estimate basin sizes by random sampling.

        For each sampled initial condition, run a short trajectory and
        assign it to the nearest attractor by Hamming distance.

        Parameters
        ----------
        attractors : List[np.ndarray]
            List of attractor state vectors.
        n_samples : int
            Number of random initial states to sample.

        Returns
        -------
        Dict[int, int]
            Mapping from attractor index (into `attractors` list) to
            estimated number of states in its basin.
        """
        if len(attractors) == 0:
            return {}

        logger.info(
            "Estimating basin sizes: %d samples, %d attractors.",
            n_samples,
            len(attractors),
        )
        n_samples = min(n_samples, self.config.n_basin_samples)
        rng = np.random.default_rng(seed=123)

        # Attractors as 2D array for vectorized Hamming
        attractor_matrix = np.vstack(attractors).astype(np.float32)  # (n_attr, n_genes)
        basin_counts: Dict[int, int] = {i: 0 for i in range(len(attractors))}

        for _ in range(n_samples):
            init = rng.integers(0, 2, size=self.n_genes, dtype=np.uint8)
            terminal, _ = self._run_trajectory(
                init,
                max_steps=self.config.max_trajectory_steps // 2,
                rng=rng,
            )
            if terminal is None:
                terminal = init

            # Find nearest attractor by Hamming distance
            dists = np.sum(attractor_matrix != terminal.astype(np.float32), axis=1)
            nearest = int(np.argmin(dists))
            basin_counts[nearest] += 1

        total = sum(basin_counts.values())
        if total > 0:
            basin_fractions = {k: v / total for k, v in basin_counts.items()}
        else:
            basin_fractions = {k: 0.0 for k in basin_counts}

        logger.info(
            "Basin size estimates (fractions): %s",
            {k: f"{v:.3f}" for k, v in basin_fractions.items()},
        )
        return basin_fractions

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def state_to_dict(self, state: np.ndarray) -> Dict[str, int]:
        """Convert a Boolean state vector to a {gene: state} dict."""
        return {g: int(state[i]) for i, g in enumerate(self.genes)}

    def dict_to_state(self, state_dict: Dict[str, int]) -> np.ndarray:
        """Convert a {gene: state} dict to a Boolean state vector."""
        state = np.zeros(self.n_genes, dtype=np.uint8)
        for gene, val in state_dict.items():
            if gene in self.gene_idx:
                state[self.gene_idx[gene]] = int(val)
        return state
