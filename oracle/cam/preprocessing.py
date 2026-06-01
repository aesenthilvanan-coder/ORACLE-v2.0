"""
Cancer Attraction Mapper - Preprocessing Module

Handles quality control, normalization, dimensionality reduction,
clustering, and cell state annotation for single-cell RNA-seq data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import scanpy as sc
from anndata import AnnData

logger = logging.getLogger(__name__)


@dataclass
class CAMConfig:
    """Configuration dataclass for the Cancer Attractor Mapper pipeline."""

    # Quality control
    min_genes: int = 200
    min_cells: int = 3
    max_pct_mt: float = 20.0

    # Feature selection
    n_top_genes: int = 3000

    # Dimensionality reduction
    n_pcs: int = 50
    n_neighbors: int = 15

    # GRN inference
    grn_size: int = 100
    data_weight: float = 0.6
    prior_weight: float = 0.4
    min_confidence: float = 0.3

    # Attractor sampling
    n_attractor_samples: int = 10000
    n_basin_samples: int = 50000
    max_trajectory_steps: int = 1000

    # ODE integration
    integration_time: float = 50.0
    n_ode_steps: int = 200

    # MD simulation
    md_frames: int = 100

    # Compute
    n_jobs: int = 8

    # Biology
    cancer_type: str = "colorectal"
    tissue: str = "colon"


class CellStateAnnotator:
    """
    Annotates cells as 'normal', 'cancer', or 'transitional' based on
    marker gene expression and known cancer driver signatures.
    """

    # Tissue-specific normal marker genes
    NORMAL_MARKERS = {
        "colon": [
            "CDX2", "MUC2", "FABP1", "CA1", "CA2", "CEACAM5", "EPCAM",
            "KRT8", "KRT18", "KRT19", "VIL1", "SLC26A3", "AQP8",
        ],
        "breast": [
            "ESR1", "PGR", "FOXA1", "GATA3", "TFF1", "TFF3", "KRT8",
            "KRT18", "KRT19", "EPCAM", "CDH1", "MUC1",
        ],
        "lung": [
            "SFTPA1", "SFTPB", "SFTPC", "SFTPD", "NKX2-1", "FOXA2",
            "KRT5", "TP63", "HOPX", "AGER", "PDPN",
        ],
        "brain": [
            "GFAP", "S100B", "VIM", "ALDH1A1", "SLC1A2", "AQP4",
            "OLIG2", "MBP", "PLP1", "MAP2", "SYP", "RBFOX3",
        ],
        "blood": [
            "HBB", "HBA1", "HBA2", "GYPA", "TFRC", "CD34", "CD38",
            "MPO", "ELANE", "AZU1", "LYZ", "S100A8", "S100A9",
        ],
        "skin": [
            "KRT1", "KRT10", "KRT14", "KRT5", "IVL", "FLG", "LOR",
            "TYRP1", "DCT", "MITF", "S100B",
        ],
    }

    # Cancer-type-specific oncogene / driver markers
    CANCER_MARKERS = {
        "colorectal": [
            "KRAS", "BRAF", "APC", "TP53", "SMAD4", "PIK3CA",
            "MYC", "CCND1", "CDH3", "CEACAM6", "MMP7", "S100A4",
            "SNAI1", "SNAI2", "VIM", "FN1", "CD44", "ALDH1A1",
        ],
        "leukemia_aml": [
            "FLT3", "NPM1", "DNMT3A", "IDH1", "IDH2", "RUNX1",
            "CEBPA", "MYC", "BCL2", "CD34", "CD117", "CD33",
            "HOXA9", "HOXA10", "MEIS1", "EVI1",
        ],
        "breast": [
            "ERBB2", "ESR1", "PGR", "MYC", "CCND1", "CDH3",
            "VIM", "SNAI1", "SNAI2", "ZEB1", "ZEB2", "TWIST1",
            "CD44", "CD24", "ALDH1A1", "MMP9",
        ],
        "lung": [
            "EGFR", "KRAS", "ALK", "ROS1", "MET", "BRAF",
            "NKX2-1", "TTF1", "CK7", "CK20", "NAPSA", "SP-A",
            "MYC", "BCL2", "CCND1",
        ],
        "glioblastoma": [
            "EGFR", "PTEN", "TP53", "IDH1", "TERT", "CDKN2A",
            "MYC", "PDGFRA", "MDM2", "RB1", "NESTIN", "SOX2",
            "CD44", "CD133", "OLIG2", "GFAP",
        ],
        "melanoma": [
            "BRAF", "NRAS", "KIT", "MITF", "SOX10", "PAX3",
            "TYRP1", "DCT", "MLANA", "S100B", "MET", "MYC",
            "SNAI2", "VIM", "ZEB2", "CD44",
        ],
    }

    # Cancer driver transcription factors per cancer type
    CANCER_DRIVER_TFS = {
        "colorectal": ["MYC", "TP53", "CDX2", "SNAI1", "SNAI2", "ZEB1", "HIF1A", "STAT3"],
        "leukemia_aml": ["RUNX1", "CEBPA", "SPI1", "IRF8", "MYC", "TP53", "HOXA9", "MEIS1"],
        "breast": ["ESR1", "FOXA1", "GATA3", "MYC", "SNAI1", "TWIST1", "ZEB1", "TP53"],
        "lung": ["NKX2-1", "TP63", "FOXA2", "MYC", "EGFR", "KRAS", "BRAF"],
        "glioblastoma": ["SOX2", "OLIG2", "MYC", "EGFR", "TP53", "PTEN", "IDH1"],
        "melanoma": ["MITF", "SOX10", "PAX3", "BRAF", "NRAS", "MYC", "SNAI2"],
    }

    def __init__(self, cancer_type: str = "colorectal", tissue: str = "colon"):
        self.cancer_type = cancer_type
        self.tissue = tissue
        self.normal_markers = self._load_normal_markers()
        self.cancer_markers = self._load_cancer_markers()
        self.driver_tfs = self._load_driver_tfs()

    def _load_normal_markers(self) -> List[str]:
        return self.NORMAL_MARKERS.get(self.tissue, self.NORMAL_MARKERS["colon"])

    def _load_cancer_markers(self) -> List[str]:
        return self.CANCER_MARKERS.get(self.cancer_type, self.CANCER_MARKERS["colorectal"])

    def _load_driver_tfs(self) -> List[str]:
        return self.CANCER_DRIVER_TFS.get(self.cancer_type, self.CANCER_DRIVER_TFS["colorectal"])

    def annotate(self, adata: AnnData) -> AnnData:
        """
        Annotate cells as 'normal', 'cancer', or 'transitional'.

        Scoring is based on fraction of marker genes expressed above
        the per-gene median. Cells with high cancer score and low normal
        score are labeled cancer; vice versa normal; mixed are transitional.

        Parameters
        ----------
        adata : AnnData
            Preprocessed AnnData (log-normalized).

        Returns
        -------
        AnnData with 'cell_state', 'cancer_score', 'normal_score' added
        to .obs.
        """
        import scipy.sparse as sp

        logger.info("Annotating cell states (normal / cancer / transitional).")

        var_names = list(adata.var_names)

        normal_genes = [g for g in self.normal_markers if g in var_names]
        cancer_genes = [g for g in self.cancer_markers if g in var_names]

        if len(normal_genes) == 0:
            logger.warning("No normal marker genes found in dataset; using fallback scoring.")
            adata.obs["normal_score"] = 0.0
        else:
            normal_idx = [var_names.index(g) for g in normal_genes]
            if sp.issparse(adata.X):
                normal_expr = np.asarray(adata.X[:, normal_idx].todense())
            else:
                normal_expr = adata.X[:, normal_idx]
            gene_medians = np.median(normal_expr, axis=0)
            expressed = (normal_expr > gene_medians).astype(float)
            adata.obs["normal_score"] = expressed.mean(axis=1)

        if len(cancer_genes) == 0:
            logger.warning("No cancer marker genes found in dataset; using fallback scoring.")
            adata.obs["cancer_score"] = 0.0
        else:
            cancer_idx = [var_names.index(g) for g in cancer_genes]
            if sp.issparse(adata.X):
                cancer_expr = np.asarray(adata.X[:, cancer_idx].todense())
            else:
                cancer_expr = adata.X[:, cancer_idx]
            gene_medians = np.median(cancer_expr, axis=0)
            expressed = (cancer_expr > gene_medians).astype(float)
            adata.obs["cancer_score"] = expressed.mean(axis=1)

        # Classification thresholds
        cancer_scores = adata.obs["cancer_score"].values
        normal_scores = adata.obs["normal_score"].values

        high_cancer_thresh = np.percentile(cancer_scores, 60)
        high_normal_thresh = np.percentile(normal_scores, 60)

        labels = np.full(adata.n_obs, "transitional", dtype=object)
        labels[
            (cancer_scores >= high_cancer_thresh) & (normal_scores < high_normal_thresh)
        ] = "cancer"
        labels[
            (normal_scores >= high_normal_thresh) & (cancer_scores < high_cancer_thresh)
        ] = "normal"

        adata.obs["cell_state"] = labels
        adata.obs["cell_state"] = adata.obs["cell_state"].astype("category")

        n_cancer = (labels == "cancer").sum()
        n_normal = (labels == "normal").sum()
        n_trans = (labels == "transitional").sum()
        logger.info(
            "Cell state annotation: %d cancer, %d normal, %d transitional.",
            n_cancer, n_normal, n_trans,
        )
        return adata


class CancerAttractionPreprocessor:
    """
    Full preprocessing pipeline for Cancer Attractor Mapper.

    Steps
    -----
    1. Quality control (filter cells/genes, mitochondrial content)
    2. Normalization (library-size to 10k, log1p)
    3. Highly variable gene selection (Seurat v3)
    4. Scaling (z-score, clipped at 10)
    5. Dimensionality reduction (PCA, neighbors, UMAP)
    6. Clustering (Leiden at multiple resolutions)
    7. Cell state annotation (normal / cancer / transitional)
    """

    def __init__(self, config: CAMConfig):
        self.config = config
        self.annotator = CellStateAnnotator(
            cancer_type=config.cancer_type,
            tissue=config.tissue,
        )

    def run(self, adata: AnnData) -> AnnData:
        """
        Execute the full preprocessing pipeline.

        Parameters
        ----------
        adata : AnnData
            Raw count matrix (cells x genes).

        Returns
        -------
        AnnData
            Processed AnnData with QC metrics, embeddings, clusters,
            and cell state annotations.
        """
        logger.info("Starting CAM preprocessing pipeline.")
        adata = self._quality_control(adata)
        adata = self._normalize(adata)
        adata = self._select_hvgs(adata)
        adata = self._scale(adata)
        adata = self._reduce_dims(adata)
        adata = self._cluster(adata)
        adata = self._annotate_normal_cancer(adata)
        logger.info("Preprocessing complete. Final shape: %s.", adata.shape)
        return adata

    # ------------------------------------------------------------------
    # Step 1: Quality control
    # ------------------------------------------------------------------

    def _quality_control(self, adata: AnnData) -> AnnData:
        """
        Filter low-quality cells and lowly expressed genes.

        - Remove cells expressing fewer than `min_genes` genes.
        - Remove genes detected in fewer than `min_cells` cells.
        - Compute mitochondrial gene fraction and filter cells
          with > `max_pct_mt` % mitochondrial reads.
        """
        logger.info("Running quality control.")

        # Basic cell/gene filtering
        sc.pp.filter_cells(adata, min_genes=self.config.min_genes)
        sc.pp.filter_genes(adata, min_cells=self.config.min_cells)

        # Mitochondrial gene percentage
        mito_genes = adata.var_names.str.upper().str.startswith("MT-")
        adata.obs["pct_counts_mt"] = (
            np.sum(adata[:, mito_genes].X, axis=1).A1
            if hasattr(np.sum(adata[:, mito_genes].X, axis=1), "A1")
            else np.sum(adata[:, mito_genes].X, axis=1)
        )
        # Normalize to percentage
        total_counts = np.asarray(adata.X.sum(axis=1)).flatten()
        adata.obs["total_counts"] = total_counts
        mt_counts = np.asarray(adata[:, mito_genes].X.sum(axis=1)).flatten()
        adata.obs["pct_counts_mt"] = np.where(
            total_counts > 0, mt_counts / total_counts * 100.0, 0.0
        )

        n_before = adata.n_obs
        adata = adata[adata.obs["pct_counts_mt"] < self.config.max_pct_mt].copy()
        n_after = adata.n_obs
        logger.info(
            "QC: removed %d cells with >%.1f%% mitochondrial reads. Retained %d cells.",
            n_before - n_after,
            self.config.max_pct_mt,
            n_after,
        )
        return adata

    # ------------------------------------------------------------------
    # Step 2: Normalization
    # ------------------------------------------------------------------

    def _normalize(self, adata: AnnData) -> AnnData:
        """
        Library-size normalize to 10,000 counts per cell, then log1p transform.
        Store the raw (pre-normalization) counts in adata.raw.
        """
        logger.info("Normalizing: library size -> 10k, log1p.")
        adata.raw = adata  # store raw counts
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        return adata

    # ------------------------------------------------------------------
    # Step 3: Highly variable gene selection
    # ------------------------------------------------------------------

    def _select_hvgs(self, adata: AnnData) -> AnnData:
        """
        Select highly variable genes using the Seurat v3 method.
        Falls back to 'seurat' flavor if seurat_v3 requires more cells than available.
        Restricts to top `n_top_genes` HVGs.
        """
        n_top = min(self.config.n_top_genes, adata.n_vars)

        # HVG selection requires enough cells and genes to be meaningful.
        # Skip it for tiny datasets (e.g. unit-test fixtures).
        if adata.n_obs < 30 or adata.n_vars <= n_top:
            logger.warning(
                "Dataset too small (%d cells × %d genes) for HVG selection; using all genes.",
                adata.n_obs, adata.n_vars,
            )
            adata.var["highly_variable"] = True
            return adata

        logger.info("Selecting %d highly variable genes (Seurat v3).", n_top)
        try:
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top)
        except (ValueError, ImportError) as exc:
            logger.warning(
                "seurat_v3 HVG selection failed (%s); falling back to 'seurat' flavor.", exc
            )
            try:
                sc.pp.highly_variable_genes(adata, n_top_genes=n_top)
            except Exception as exc2:
                logger.warning("HVG selection failed entirely (%s); using all genes.", exc2)
                adata.var["highly_variable"] = True
                return adata
        adata = adata[:, adata.var["highly_variable"]].copy()
        logger.info("HVG selection complete. Shape: %s.", adata.shape)
        return adata

    # ------------------------------------------------------------------
    # Step 4: Scaling
    # ------------------------------------------------------------------

    def _scale(self, adata: AnnData) -> AnnData:
        """
        Z-score scale expression values, clipping at max_value=10.
        """
        logger.info("Scaling data (z-score, max_value=10).")
        sc.pp.scale(adata, max_value=10)
        return adata

    # ------------------------------------------------------------------
    # Step 5: Dimensionality reduction
    # ------------------------------------------------------------------

    def _reduce_dims(self, adata: AnnData) -> AnnData:
        """
        PCA (svd_solver='arpack'), neighborhood graph, and UMAP.
        """
        # Cap PCA components to dataset dimensions (PCA requires n_comps < min(n_obs, n_vars))
        max_comps = min(adata.n_obs, adata.n_vars) - 1
        n_pcs = max(1, min(self.config.n_pcs, max_comps))
        n_neighbors = max(2, min(self.config.n_neighbors, adata.n_obs - 1))

        logger.info(
            "Running PCA (%d PCs), neighborhood graph (%d neighbors), UMAP.",
            n_pcs,
            n_neighbors,
        )
        sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.umap(adata)
        return adata

    # ------------------------------------------------------------------
    # Step 6: Clustering
    # ------------------------------------------------------------------

    def _cluster(self, adata: AnnData) -> AnnData:
        """
        Leiden clustering at resolutions [0.3, 0.5, 0.8, 1.0].
        Results stored in adata.obs as 'leiden_<resolution>'.
        """
        resolutions = [0.3, 0.5, 0.8, 1.0]
        logger.info("Running Leiden clustering at resolutions: %s.", resolutions)
        for res in resolutions:
            key = f"leiden_{res}"
            sc.tl.leiden(adata, resolution=res, key_added=key)
            n_clusters = adata.obs[key].nunique()
            logger.info("  Resolution %.1f -> %d clusters.", res, n_clusters)
        # Default clustering key at resolution 0.5
        adata.obs["leiden"] = adata.obs["leiden_0.5"]
        return adata

    # ------------------------------------------------------------------
    # Step 7: Cell state annotation
    # ------------------------------------------------------------------

    def _annotate_normal_cancer(self, adata: AnnData) -> AnnData:
        """
        Annotate cells as 'normal', 'cancer', or 'transitional' using
        the CellStateAnnotator based on marker gene expression.
        """
        logger.info("Annotating cell states.")
        adata = self.annotator.annotate(adata)
        return adata
