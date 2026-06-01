import numpy as np
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CANCER_MARKERS: Dict[str, List[str]] = {
    "colorectal": ["EPCAM", "CDX2", "KRT20", "MUC2", "CEACAM5", "VIL1", "CLDN3"],
    "aml": ["CD34", "CD33", "CD13", "MPO", "FLT3", "NPM1", "DNMT3A"],
    "breast": ["EPCAM", "ESR1", "PGR", "ERBB2", "KRT8", "KRT18", "GATA3"],
    "lung": ["NKX2-1", "SFTPC", "SFTPB", "NAPSA", "KRT7", "TTF1", "EGFR"],
    "glioblastoma": ["EGFR", "PTEN", "IDH1", "TP53", "NESTIN", "SOX2", "OLIG2"],
    "melanoma": ["MITF", "SOX10", "MLANA", "TYR", "DCT", "S100B", "MBP"],
    "pancreatic": ["KRT19", "MUC1", "CEA", "CA19-9", "KRAS", "SMAD4", "CDKN2A"],
    "prostate": ["AR", "KLK3", "FOLH1", "NKX3-1", "TMPRSS2", "ERG", "EZH2"],
    "ovarian": ["CA125", "MUC16", "PAX8", "WT1", "BRCA1", "BRCA2", "TP53"],
    "hepatocellular": ["AFP", "EPCAM", "GPC3", "DLK1", "ALB", "TF", "CYP3A4"],
}

NORMAL_MARKERS: Dict[str, List[str]] = {
    "colon": ["CDX2", "MUC2", "VIL1", "FABP1", "SLC2A2"],
    "bone_marrow": ["HSP90B1", "SPN", "PTPRC", "CD3D", "CD14"],
    "breast_epithelium": ["ESR1", "PGR", "FOXA1", "GATA3", "KRT5"],
    "lung_epithelium": ["SFTPC", "HOPX", "AGER", "NKX2-1", "FOXJ1"],
    "brain": ["GFAP", "S100B", "RBFOX3", "SYP", "MBP"],
    "skin": ["KRT14", "KRT5", "DSG1", "FLG", "LOR"],
    "pancreas": ["PDX1", "NKX6-1", "INS", "GCG", "PRSS1"],
    "prostate_epithelium": ["NKX3-1", "KLK2", "KLK3", "FOLH1", "ACPP"],
    "ovarian_epithelium": ["PAX8", "WT1", "KRT7", "CALB2", "MSLN"],
    "liver": ["ALB", "CYP3A4", "APOB", "HNF4A", "TF"],
}


class CellStateAnnotator:
    """Annotates cells as cancer, normal, or transitional using CNV + marker expression."""

    def __init__(
        self,
        cancer_type: str = "colorectal",
        tissue_type: str = "colon",
        cancer_weight: float = 0.6,
        cnv_weight: float = 0.4,
        cancer_threshold: float = 0.6,
        normal_threshold: float = 0.4,
    ):
        self.cancer_type = cancer_type
        self.tissue_type = tissue_type
        self.cancer_weight = cancer_weight
        self.cnv_weight = cnv_weight
        self.cancer_threshold = cancer_threshold
        self.normal_threshold = normal_threshold

    def annotate(self, adata) -> None:
        cancer_markers = CANCER_MARKERS.get(self.cancer_type, [])
        normal_markers = NORMAL_MARKERS.get(self.tissue_type, [])

        cancer_score = self._compute_marker_score(adata, cancer_markers)
        normal_score = self._compute_marker_score(adata, normal_markers)

        if "cnv_score" in adata.obs.columns:
            cnv_score = adata.obs["cnv_score"].values.astype(float)
        else:
            cnv_score = np.zeros(adata.n_obs)

        combined = (
            self.cancer_weight * cancer_score
            + self.cnv_weight * cnv_score
            - 0.3 * normal_score
        )
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)

        labels = np.where(
            combined >= self.cancer_threshold,
            "cancer",
            np.where(combined <= self.normal_threshold, "normal", "transitional"),
        )

        adata.obs["cell_state"] = labels
        adata.obs["cancer_score_annotation"] = combined.astype(np.float32)
        adata.obs["cancer_marker_score"] = cancer_score.astype(np.float32)
        adata.obs["normal_marker_score"] = normal_score.astype(np.float32)

        n_cancer = (labels == "cancer").sum()
        n_normal = (labels == "normal").sum()
        n_trans = (labels == "transitional").sum()
        logger.info(f"Cell annotation: {n_cancer} cancer, {n_normal} normal, {n_trans} transitional")

    def _compute_marker_score(self, adata, markers: List[str]) -> np.ndarray:
        import scanpy as sc
        present = [m for m in markers if m in adata.var_names]
        if not present:
            return np.zeros(adata.n_obs)
        try:
            sc.tl.score_genes(adata, gene_list=present, score_name="_tmp_score_")
            score = adata.obs["_tmp_score_"].values.copy().astype(float)
            del adata.obs["_tmp_score_"]
            score = (score - score.min()) / (score.max() - score.min() + 1e-8)
            return score
        except Exception as e:
            logger.debug(f"Gene scoring failed: {e}")
            X = adata[:, present].X
            if hasattr(X, "toarray"):
                X = X.toarray()
            score = np.array(X, dtype=float).mean(axis=1)
            score = (score - score.min()) / (score.max() - score.min() + 1e-8)
            return score
