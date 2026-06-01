import numpy as np
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class ScRNAPreprocessor:
    """End-to-end scRNA-seq preprocessing for ORACLE.

    Runs 13 sequential steps: QC → doublet removal → normalization →
    HVG selection → scaling → PCA → batch correction → neighbor graph →
    UMAP → clustering → CNV scoring → cell state annotation → pseudotime.
    """

    def __init__(
        self,
        min_genes: int = 200,
        min_cells: int = 3,
        max_pct_mt: float = 20.0,
        max_pct_rb: float = 50.0,
        min_counts: int = 500,
        max_counts: int = 50000,
        n_top_genes: int = 3000,
        n_pcs: int = 50,
        n_neighbors: int = 15,
        target_sum: float = 1e4,
        batch_key: Optional[str] = None,
        cancer_type: str = "colorectal",
        tissue: str = "colon",
        random_state: int = 42,
    ):
        self.min_genes = min_genes
        self.min_cells = min_cells
        self.max_pct_mt = max_pct_mt
        self.max_pct_rb = max_pct_rb
        self.min_counts = min_counts
        self.max_counts = max_counts
        self.n_top_genes = n_top_genes
        self.n_pcs = n_pcs
        self.n_neighbors = n_neighbors
        self.target_sum = target_sum
        self.batch_key = batch_key
        self.cancer_type = cancer_type
        self.tissue = tissue
        self.random_state = random_state

    def run(self, adata):
        import anndata as ad
        import scanpy as sc

        logger.info(f"Starting preprocessing: {adata.n_obs} cells x {adata.n_vars} genes")

        adata = self._quality_control(adata)
        logger.info(f"After QC: {adata.n_obs} cells")

        adata = self._detect_doublets(adata)
        adata = self._normalize(adata)
        adata = self._select_hvgs(adata)
        adata = self._scale(adata)
        adata = self._reduce_dims(adata)
        adata = self._correct_batch(adata)
        adata = self._build_graph(adata)
        adata = self._embed_umap(adata)
        adata = self._cluster(adata)
        adata = self._compute_cnv_score(adata)
        adata = self._annotate_cell_states(adata)
        adata = self._compute_pseudotime(adata)

        logger.info(f"Preprocessing complete: {adata.n_obs} cells, {adata.n_vars} genes")
        return adata

    def _quality_control(self, adata):
        import scanpy as sc
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        adata.var["rb"] = adata.var_names.str.startswith(("RPS", "RPL"))
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mt", "rb"], percent_top=None, log1p=False, inplace=True
        )
        sc.pp.filter_cells(adata, min_genes=self.min_genes)
        sc.pp.filter_cells(adata, min_counts=self.min_counts)
        sc.pp.filter_cells(adata, max_counts=self.max_counts)
        sc.pp.filter_genes(adata, min_cells=self.min_cells)
        adata = adata[adata.obs["pct_counts_mt"] < self.max_pct_mt].copy()
        adata = adata[adata.obs["pct_counts_rb"] < self.max_pct_rb].copy()
        return adata

    def _detect_doublets(self, adata):
        try:
            import scrublet as scr
            scrub = scr.Scrublet(adata.X, random_state=self.random_state)
            doublet_scores, predicted_doublets = scrub.scrub_doublets(verbose=False)
            adata.obs["doublet_score"] = doublet_scores
            adata.obs["predicted_doublet"] = predicted_doublets
            adata = adata[~predicted_doublets].copy()
            logger.info(f"Doublet removal: {predicted_doublets.sum()} doublets removed")
        except Exception as e:
            logger.warning(f"Doublet detection failed (scrublet): {e}. Skipping.")
        return adata

    def _normalize(self, adata):
        import scanpy as sc
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=self.target_sum)
        sc.pp.log1p(adata)
        adata.raw = adata
        return adata

    def _select_hvgs(self, adata):
        import scanpy as sc
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=self.n_top_genes,
            subset=True,
            flavor="seurat_v3" if "counts" in adata.layers else "seurat",
            layer="counts" if "counts" in adata.layers else None,
        )
        return adata

    def _scale(self, adata):
        import scanpy as sc
        sc.pp.scale(adata, max_value=10)
        return adata

    def _reduce_dims(self, adata):
        import scanpy as sc
        sc.tl.pca(adata, n_comps=self.n_pcs, random_state=self.random_state)
        return adata

    def _correct_batch(self, adata):
        if self.batch_key is None or self.batch_key not in adata.obs.columns:
            return adata
        try:
            import harmonypy
            ho = harmonypy.run_harmony(
                adata.obsm["X_pca"], adata.obs, self.batch_key,
                random_state=self.random_state
            )
            adata.obsm["X_pca_harmony"] = ho.Z_corr.T
            adata.obsm["X_pca"] = adata.obsm["X_pca_harmony"]
            logger.info("Batch correction with Harmony applied")
        except ImportError:
            logger.warning("harmonypy not installed, skipping batch correction")
        except Exception as e:
            logger.warning(f"Harmony batch correction failed: {e}")
        return adata

    def _build_graph(self, adata):
        import scanpy as sc
        use_rep = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"
        sc.pp.neighbors(
            adata, n_neighbors=self.n_neighbors, n_pcs=self.n_pcs,
            use_rep=use_rep, random_state=self.random_state
        )
        return adata

    def _embed_umap(self, adata):
        import scanpy as sc
        sc.tl.umap(adata, random_state=self.random_state)
        return adata

    def _cluster(self, adata):
        import scanpy as sc
        for res in [0.3, 0.5, 0.8, 1.0]:
            sc.tl.leiden(adata, resolution=res, random_state=self.random_state,
                         key_added=f"leiden_{res}")
        adata.obs["leiden"] = adata.obs["leiden_0.5"]
        return adata

    def _compute_cnv_score(self, adata):
        from oracle.preprocessing.cnv_inference import SimpleCNVScorer
        try:
            scorer = SimpleCNVScorer()
            cnv_scores = scorer.compute(adata)
            adata.obs["cnv_score"] = cnv_scores
            logger.info(f"CNV scores: mean={cnv_scores.mean():.3f}, std={cnv_scores.std():.3f}")
        except Exception as e:
            logger.warning(f"CNV scoring failed: {e}")
            adata.obs["cnv_score"] = 0.0
        return adata

    def _annotate_cell_states(self, adata):
        from oracle.preprocessing.cell_annotator import CellStateAnnotator
        annotator = CellStateAnnotator(
            cancer_type=self.cancer_type,
            tissue_type=self.tissue,
        )
        annotator.annotate(adata)
        return adata

    def _compute_pseudotime(self, adata):
        import scanpy as sc
        try:
            import palantir
            import pandas as pd
            pca_df = pd.DataFrame(
                adata.obsm["X_pca"][:, :20],
                index=adata.obs_names
            )
            dm_res = palantir.utils.run_diffusion_maps(pca_df, n_components=5)
            ms_data = palantir.utils.determine_multiscale_space(dm_res)
            early_cell = adata.obs_names[adata.obs.get("cnv_score", pd.Series([0])).idxmin()]
            pr_res = palantir.core.run_palantir(
                ms_data, early_cell=early_cell, num_waypoints=500
            )
            adata.obs["pseudotime"] = pr_res.pseudotime.reindex(adata.obs_names).fillna(0).values
            logger.info("Pseudotime computed with Palantir")
        except Exception as e:
            logger.warning(f"Palantir failed ({e}), falling back to DPT")
            try:
                cancer_cells = adata.obs.get("cell_state", "").eq("cancer")
                if cancer_cells.any():
                    normal_cells = ~cancer_cells
                    root_idx = np.where(normal_cells)[0]
                    if len(root_idx) > 0:
                        adata.uns["iroot"] = root_idx[0]
                else:
                    adata.uns["iroot"] = 0
                sc.tl.dpt(adata)
                adata.obs["pseudotime"] = adata.obs["dpt_pseudotime"].values
            except Exception as e2:
                logger.warning(f"DPT failed ({e2}), setting pseudotime to 0")
                adata.obs["pseudotime"] = 0.0
        return adata
