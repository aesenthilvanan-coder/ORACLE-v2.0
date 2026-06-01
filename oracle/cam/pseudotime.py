"""
Cancer Attraction Mapper - Pseudotime Computer

Computes diffusion pseudotime (DPT) along the normal-to-cancer
transition axis. The root cell is the cell whose expression profile
is closest to the inferred normal attractor.

Also defines CAMOutput, the main output dataclass for Module 1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import scanpy as sc
from anndata import AnnData

from oracle.cam.preprocessing import CAMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CAMOutput dataclass
# ---------------------------------------------------------------------------

@dataclass
class CAMOutput:
    """
    Structured output of the Cancer Attractor Mapper (Module 1).

    Attributes
    ----------
    adata : AnnData
        Fully preprocessed and annotated single-cell dataset.
    grn : nx.DiGraph
        Inferred signed gene regulatory network.
    genes : List[str]
        Ordered gene list corresponding to attractor state vectors.
    n_genes : int
        Number of genes in the core GRN.
    bool_network : BooleanNetworkSimulator
        Discrete Boolean network model.
    ode_model : ContinuousGRNDynamics
        Continuous ODE GRN model (nn.Module).
    all_attractors : List[np.ndarray]
        All identified attractor state vectors.
    attractor_labels : List[str]
        Classification label per attractor ('normal'/'cancer'/'transitional').
    basin_sizes : Dict
        Basin size estimates (attractor index -> count or fraction).
    cancer_attractor : np.ndarray
        Best cancer attractor state vector.
    normal_attractor : np.ndarray
        Best normal attractor state vector.
    cancer_score_func : Callable
        Function f(attractor, genes) -> float returning a cancer score.
    landscape_embedding : np.ndarray
        UMAP projection of attractor states, shape (n_attractors, 2).
    trajectory_cells : AnnData
        Subset of adata containing transitional / trajectory cells.
    cancer_type : str
        Cancer type identifier.
    tissue_type : str
        Tissue type identifier.
    sample_id : str
        Sample identifier.
    metadata : Dict
        Arbitrary additional metadata.
    """

    adata: Any                          # AnnData
    grn: Any                            # nx.DiGraph
    genes: List[str]
    n_genes: int
    bool_network: Any                   # BooleanNetworkSimulator
    ode_model: Any                      # ContinuousGRNDynamics
    all_attractors: List[Any]           # List[np.ndarray]
    attractor_labels: List[str]
    basin_sizes: Dict
    cancer_attractor: Any               # np.ndarray
    normal_attractor: Any               # np.ndarray
    cancer_score_func: Any              # Callable
    landscape_embedding: Any            # np.ndarray
    trajectory_cells: Any               # AnnData
    cancer_type: str
    tissue_type: str
    sample_id: str
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PseudotimeComputer
# ---------------------------------------------------------------------------

class PseudotimeComputer:
    """
    Compute diffusion pseudotime (DPT) along the normal-to-cancer axis.

    The root cell is identified as the cell whose gene expression in PCA
    space is closest to the normal attractor state, giving a biologically
    meaningful pseudotime ordering from normal to cancer.

    Parameters
    ----------
    config : CAMConfig
        Pipeline configuration.
    """

    def __init__(self, config: CAMConfig):
        self.config = config

    def compute(
        self,
        adata: AnnData,
        cancer_attractor: np.ndarray,
        normal_attractor: np.ndarray,
        genes: Optional[List[str]] = None,
    ) -> AnnData:
        """
        Compute diffusion pseudotime and add it to adata.obs.

        Steps:
        1. Identify the root cell as closest to `normal_attractor`.
        2. Set adata.uns['iroot'] to the root cell index.
        3. Compute diffusion map (sc.tl.diffmap).
        4. Compute DPT (sc.tl.dpt).
        5. Store result in adata.obs['pseudotime'].

        Parameters
        ----------
        adata : AnnData
            Preprocessed AnnData with UMAP and neighbors graph.
        cancer_attractor : np.ndarray
            Cancer attractor state vector.
        normal_attractor : np.ndarray
            Normal attractor state vector.
        genes : List[str], optional
            Gene names corresponding to attractor dimensions.
            Defaults to adata.var_names.

        Returns
        -------
        AnnData
            AnnData with 'pseudotime' added to .obs.
        """
        logger.info("Computing diffusion pseudotime.")

        if normal_attractor is None:
            logger.warning(
                "No normal attractor available; setting pseudotime to zeros."
            )
            adata.obs["pseudotime"] = 0.0
            return adata

        # Resolve gene list
        attractor_genes = genes if genes is not None else list(adata.var_names)

        # Step 1: Find root cell (closest to normal attractor)
        root_idx = self._find_root_cell(adata, normal_attractor, attractor_genes)
        logger.info("Root cell (closest to normal attractor): cell index %d.", root_idx)
        adata.uns["iroot"] = int(root_idx)

        # Step 2: Ensure neighbors graph is computed
        if "neighbors" not in adata.uns:
            logger.info("Computing neighborhood graph for DPT.")
            sc.pp.neighbors(
                adata,
                n_neighbors=self.config.n_neighbors,
                n_pcs=self.config.n_pcs,
            )

        # Step 3: Diffusion map
        logger.info("Computing diffusion map.")
        try:
            sc.tl.diffmap(adata, n_comps=15)
        except Exception as exc:
            logger.warning("Diffusion map failed (%s); using PCA fallback.", exc)
            # Fallback: use PCA-based pseudotime
            adata.obs["pseudotime"] = self._pca_pseudotime(
                adata, cancer_attractor, normal_attractor, attractor_genes
            )
            return adata

        # Step 4: Diffusion pseudotime
        logger.info("Computing diffusion pseudotime (DPT).")
        try:
            sc.tl.dpt(adata, n_dcs=10)
            # Rename to 'pseudotime'
            if "dpt_pseudotime" in adata.obs.columns:
                adata.obs["pseudotime"] = adata.obs["dpt_pseudotime"].copy()
                logger.info("Pseudotime computed successfully.")
            else:
                logger.warning("dpt_pseudotime not found; using PCA fallback.")
                adata.obs["pseudotime"] = self._pca_pseudotime(
                    adata, cancer_attractor, normal_attractor, attractor_genes
                )
        except Exception as exc:
            logger.warning("DPT computation failed (%s); using PCA fallback.", exc)
            adata.obs["pseudotime"] = self._pca_pseudotime(
                adata, cancer_attractor, normal_attractor, attractor_genes
            )

        return adata

    # ------------------------------------------------------------------
    # Root cell identification
    # ------------------------------------------------------------------

    def _find_root_cell(
        self,
        adata: AnnData,
        normal_attractor: np.ndarray,
        attractor_genes: List[str],
    ) -> int:
        """
        Find the cell index closest to the normal attractor.

        Matching is performed in gene expression space (shared genes
        between attractor_genes and adata.var_names), falling back to
        PCA space if available.

        Parameters
        ----------
        adata : AnnData
        normal_attractor : np.ndarray
        attractor_genes : List[str]

        Returns
        -------
        int
            Index of root cell in adata.
        """
        import scipy.sparse as sp

        var_names = list(adata.var_names)

        # Build attractor expression vector in the data gene space
        attr_expr = np.zeros(len(var_names), dtype=np.float32)
        for gi, gene in enumerate(attractor_genes):
            if gene in var_names:
                vidx = var_names.index(gene)
                attr_expr[vidx] = float(normal_attractor[gi])

        if "X_pca" in adata.obsm and "PCs" in adata.varm:
            # Project attractor to PCA space and match
            pca_components = adata.varm["PCs"]  # (n_genes, n_pcs)
            attr_pca = attr_expr @ pca_components  # (n_pcs,)
            cell_pca = adata.obsm["X_pca"]        # (n_cells, n_pcs)
            dists = np.sum((cell_pca - attr_pca) ** 2, axis=1)
        else:
            # Match in expression space
            if sp.issparse(adata.X):
                cell_expr = adata.X.toarray()
            else:
                cell_expr = np.array(adata.X)
            dists = np.sum((cell_expr - attr_expr) ** 2, axis=1)

        return int(np.argmin(dists))

    # ------------------------------------------------------------------
    # PCA pseudotime fallback
    # ------------------------------------------------------------------

    def _pca_pseudotime(
        self,
        adata: AnnData,
        cancer_attractor: np.ndarray,
        normal_attractor: np.ndarray,
        attractor_genes: List[str],
    ) -> np.ndarray:
        """
        Compute pseudotime as projection onto the normal->cancer axis in PCA space.

        Pseudotime = dot(cell_pca - normal_pca, direction) / ||direction||^2
        clipped to [0, 1].

        Parameters
        ----------
        adata : AnnData
        cancer_attractor : np.ndarray
        normal_attractor : np.ndarray
        attractor_genes : List[str]

        Returns
        -------
        np.ndarray
            Pseudotime values, shape (n_cells,).
        """
        var_names = list(adata.var_names)

        def attractor_to_gene_expr(attractor: np.ndarray) -> np.ndarray:
            expr = np.zeros(len(var_names), dtype=np.float32)
            for gi, gene in enumerate(attractor_genes):
                if gene in var_names:
                    vidx = var_names.index(gene)
                    expr[vidx] = float(attractor[gi])
            return expr

        if "X_pca" in adata.obsm and "PCs" in adata.varm:
            pca_components = adata.varm["PCs"]

            if normal_attractor is not None:
                normal_expr = attractor_to_gene_expr(normal_attractor)
                normal_pca = normal_expr @ pca_components
            else:
                normal_pca = np.zeros(adata.obsm["X_pca"].shape[1])

            if cancer_attractor is not None:
                cancer_expr = attractor_to_gene_expr(cancer_attractor)
                cancer_pca = cancer_expr @ pca_components
            else:
                cancer_pca = np.ones(adata.obsm["X_pca"].shape[1])

            cell_pca = adata.obsm["X_pca"]
            direction = cancer_pca - normal_pca
            norm_sq = np.dot(direction, direction)
            if norm_sq < 1e-10:
                return np.zeros(adata.n_obs, dtype=np.float32)

            projections = (cell_pca - normal_pca) @ direction / norm_sq
        else:
            # Fallback: use cell state column if available
            if "cell_state" in adata.obs.columns:
                state_map = {"normal": 0.0, "transitional": 0.5, "cancer": 1.0}
                projections = adata.obs["cell_state"].map(state_map).fillna(0.5).values
            else:
                projections = np.random.default_rng(42).uniform(0, 1, adata.n_obs)

        # Clip to [0, 1]
        projections = np.clip(projections, 0.0, 1.0).astype(np.float32)
        return projections
