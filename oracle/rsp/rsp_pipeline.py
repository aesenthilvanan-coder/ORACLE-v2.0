import numpy as np
import logging
import time
from typing import Optional

from oracle.interfaces import CAMOutput, RSPOutput, SwitchSet

logger = logging.getLogger(__name__)


class RSPPipeline:
    """Reversion Switch Predictor pipeline orchestrator."""

    def __init__(self, config=None):
        self.config = config or {}

    def run(self, cam_output: CAMOutput) -> RSPOutput:
        import torch
        from oracle.rsp.perturbation_sim import PerturbationSimulator
        from oracle.rsp.switch_optimizer import MinimalSwitchOptimizer
        from oracle.rsp.druggability_filter import DruggabilityFilter
        from oracle.models.switch_predictor_gnn import SwitchPredictorGNN

        t0 = time.time()
        logger.info("[RSP] Step 1: Druggability filtering")
        drugg_filter = DruggabilityFilter()
        druggable_indices = drugg_filter.filter_genes(cam_output.genes)
        logger.info(f"[RSP] {len(druggable_indices)}/{cam_output.n_genes} genes are druggable")

        logger.info("[RSP] Step 2: Setting up perturbation simulator")
        device = self._get_device()
        simulator = PerturbationSimulator(
            ode_model=cam_output.ode_model,
            cancer_attractor=cam_output.cancer_attractor,
            normal_attractor=cam_output.normal_attractor,
            cancer_score_fn=cam_output.cancer_score_func,
            genes=cam_output.genes,
            n_jobs=self.config.get("n_jobs", 4),
        )

        logger.info("[RSP] Step 3: Switch set optimization")
        cancer_attractor_t = torch.tensor(cam_output.cancer_attractor, dtype=torch.float32, device=device)
        normal_attractor_t = torch.tensor(cam_output.normal_attractor, dtype=torch.float32, device=device)

        gnn = SwitchPredictorGNN().to(device)

        optimizer = MinimalSwitchOptimizer(
            gnn=gnn,
            simulator=simulator,
            grn=cam_output.grn,
            genes=cam_output.genes,
            cancer_attractor=cancer_attractor_t,
            normal_attractor=normal_attractor_t,
            druggable_genes=druggable_indices,
            max_perturbations=self.config.get("max_perturbations", 5),
            beam_width=self.config.get("beam_width", 10),
            target_cancer_score=self.config.get("target_cancer_score", 0.25),
            device=device,
        )

        switch_set = optimizer.optimize(cam_output.cancer_score_func)
        logger.info(f"[RSP] Switch set: activate={switch_set.genes_to_activate}, repress={switch_set.genes_to_repress}")

        logger.info("[RSP] Step 4: Perturbation trajectory simulation")
        result = simulator.simulate_perturbation(
            genes_to_activate=[cam_output.genes.index(g) for g in switch_set.genes_to_activate if g in cam_output.genes],
            genes_to_repress=[cam_output.genes.index(g) for g in switch_set.genes_to_repress if g in cam_output.genes],
            n_trajectories=self.config.get("validation_trajectories", 200),
        )

        importance = switch_set.gene_importance_scores
        perturbation_type = switch_set.perturbation_types

        predicted_before = float(cam_output.cancer_score_func(cam_output.cancer_attractor, cam_output.genes))

        rsp_output = RSPOutput(
            switch_set=switch_set,
            genes_to_activate=switch_set.genes_to_activate,
            genes_to_repress=switch_set.genes_to_repress,
            n_perturbations=len(switch_set.genes_to_activate) + len(switch_set.genes_to_repress),
            predicted_cancer_score_before=predicted_before,
            predicted_cancer_score_after=switch_set.predicted_cancer_score_after,
            predicted_reversion_probability=switch_set.predicted_reversion_probability,
            validated_reversion_fraction=result.reversion_fraction,
            perturbation_trajectories=list(result.final_states) if result.final_states is not None else [],
            cancer_score_trajectory=[float(result.mean_cancer_score)],
            gene_importance=importance,
            perturbation_type=perturbation_type,
            cam_output=cam_output,
        )

        logger.info(f"[RSP] Complete in {time.time() - t0:.1f}s. Reversion fraction: {result.reversion_fraction:.1%}")
        return rsp_output

    def _get_device(self):
        import torch
        return torch.device("mps" if torch.backends.mps.is_available() else
                            "cuda" if torch.cuda.is_available() else "cpu")
