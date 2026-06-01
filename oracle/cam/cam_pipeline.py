import numpy as np
import logging
import time
from typing import Optional

import anndata as ad

from oracle.interfaces import CAMOutput

logger = logging.getLogger(__name__)


class CAMPipeline:
    """Cancer Attractor Mapper pipeline orchestrator."""

    def __init__(self, config=None):
        self.config = config or {}

    def run(self, adata: ad.AnnData, sample_id: str = "sample") -> CAMOutput:
        import torch
        from oracle.cam.grn_inference import GRNInferenceEngine
        from oracle.cam.boolean_network import BooleanNetworkSimulator
        from oracle.cam.attractor_classifier import AttractorClassifier
        from oracle.cam.continuous_ode import ContinuousGRNDynamics

        t0 = time.time()
        cancer_type = adata.uns.get("cancer_type", self.config.get("cancer_type", "colorectal"))
        tissue_type = adata.uns.get("tissue_type", self.config.get("tissue_type", "colon"))

        logger.info(f"[CAM] Step 1: GRN inference ({adata.n_obs} cells)")
        grn_engine = GRNInferenceEngine(
            cancer_type=cancer_type,
            n_top_regulators=self.config.get("n_top_regulators", 50),
        )
        grn = grn_engine.infer(adata)
        genes = list(grn.nodes())
        n_genes = len(genes)
        logger.info(f"[CAM] GRN: {n_genes} genes, {grn.number_of_edges()} edges")

        logger.info("[CAM] Step 2: Boolean network + attractor finding")
        bool_net = BooleanNetworkSimulator(
            grn,
            n_jobs=self.config.get("n_jobs", 4),
            max_steps=self.config.get("max_steps", 2000),
        )
        attractors = bool_net.find_attractors(
            n_initial_states=self.config.get("n_initial_states", 10000)
        )
        basin_sizes = bool_net.compute_basin_sizes(attractors)

        logger.info("[CAM] Step 3: Attractor classification")
        classifier = AttractorClassifier(cancer_type=cancer_type)
        labels = [classifier.classify(a, genes) for a in attractors]
        cancer_attractor, normal_attractor = classifier.get_cancer_normal_pair(
            attractors, labels, genes
        )

        logger.info("[CAM] Step 4: Continuous ODE model fitting")
        device = self._get_device()
        ode_model = ContinuousGRNDynamics(grn, genes).to(device)
        try:
            ode_model.fit_to_data(adata, genes, n_epochs=self.config.get("ode_epochs", 100))
        except Exception as e:
            logger.warning(f"ODE fitting failed: {e}. Using untrained model.")

        logger.info("[CAM] Step 5: Cancer score function + landscape")
        cancer_score_func = classifier.get_cancer_score_function(
            attractors, labels, genes, adata
        )

        landscape_embedding = self._compute_landscape_embedding(adata, genes)
        pseudotime = adata.obs.get("pseudotime", np.zeros(adata.n_obs)).values.astype(float)
        trajectory_cells = self._get_trajectory_cells(adata)

        cancer_attractor_np = np.array(cancer_attractor, dtype=float) if cancer_attractor is not None else np.zeros(n_genes)
        normal_attractor_np = np.array(normal_attractor, dtype=float) if normal_attractor is not None else np.zeros(n_genes)

        basin_sizes_dict = {tuple(int(x) for x in k): float(v) for k, v in basin_sizes.items()}

        logger.info(f"[CAM] Complete in {time.time() - t0:.1f}s")
        return CAMOutput(
            adata=adata,
            grn=grn,
            genes=genes,
            n_genes=n_genes,
            bool_network=bool_net,
            ode_model=ode_model,
            all_attractors=attractors,
            attractor_labels=labels,
            basin_sizes=basin_sizes_dict,
            cancer_attractor=cancer_attractor_np,
            normal_attractor=normal_attractor_np,
            cancer_score_func=cancer_score_func,
            landscape_embedding=landscape_embedding,
            pseudotime=pseudotime,
            trajectory_cells=trajectory_cells,
            cancer_type=cancer_type,
            tissue_type=tissue_type,
            sample_id=sample_id,
            metadata={"runtime_s": time.time() - t0, "n_attractors": len(attractors)},
        )

    def _compute_landscape_embedding(self, adata: ad.AnnData, genes: list) -> np.ndarray:
        if "X_umap" in adata.obsm:
            return adata.obsm["X_umap"]
        if "X_pca" in adata.obsm:
            return adata.obsm["X_pca"][:, :2]
        return np.random.randn(adata.n_obs, 2)

    def _get_trajectory_cells(self, adata: ad.AnnData) -> ad.AnnData:
        if "pseudotime" not in adata.obs.columns:
            return adata
        pt = adata.obs["pseudotime"].values
        n_traj = min(500, adata.n_obs)
        indices = np.argsort(pt)[-n_traj:]
        return adata[indices].copy()

    def _get_device(self):
        import torch
        return torch.device("mps" if torch.backends.mps.is_available() else
                            "cuda" if torch.cuda.is_available() else "cpu")
