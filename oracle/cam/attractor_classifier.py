"""
Cancer Attraction Mapper - Attractor Classifier

Classifies Boolean/continuous attractor states as 'normal', 'cancer',
or 'transitional' based on marker gene expression patterns.

Cancer score = (fraction of cancer markers active - fraction of normal
               markers active + 1) / 2  -> normalized to [0, 1].

Thresholds:
    score >= 0.65 -> cancer
    score <= 0.35 -> normal
    else          -> transitional
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Curated marker gene dictionaries
# ---------------------------------------------------------------------------

CANCER_MARKERS: Dict[str, List[str]] = {
    "colorectal": [
        "KRAS", "BRAF", "APC", "TP53", "SMAD4", "PIK3CA",
        "MYC", "CCND1", "CDH3", "CEACAM6", "MMP7", "S100A4",
        "SNAI1", "SNAI2", "VIM", "FN1", "CD44", "ALDH1A1",
        "WNT5A", "LGR5", "ASCL2", "EPHB2",
    ],
    "leukemia_aml": [
        "FLT3", "NPM1", "DNMT3A", "IDH1", "IDH2", "RUNX1",
        "CEBPA", "MYC", "BCL2", "CD34", "CD117", "CD33",
        "HOXA9", "HOXA10", "MEIS1", "EVI1", "WT1", "MN1",
        "FLT3", "KIT", "MPO",
    ],
    "breast": [
        "ERBB2", "ESR1", "PGR", "MYC", "CCND1", "CDH3",
        "VIM", "SNAI1", "SNAI2", "ZEB1", "ZEB2", "TWIST1",
        "CD44", "CD24", "ALDH1A1", "MMP9", "VEGFA",
        "PIK3CA", "PTEN", "AKT1",
    ],
    "lung": [
        "EGFR", "KRAS", "ALK", "ROS1", "MET", "BRAF",
        "NKX2-1", "TTF1", "CK7", "CK20", "NAPSA",
        "MYC", "BCL2", "CCND1", "SOX2", "TP63",
        "FGFR1", "PDGFRA",
    ],
    "glioblastoma": [
        "EGFR", "PTEN", "TP53", "IDH1", "TERT", "CDKN2A",
        "MYC", "PDGFRA", "MDM2", "RB1", "NESTIN", "SOX2",
        "CD44", "CD133", "OLIG2", "GFAP", "CHI3L1",
        "MGMT", "VEGFA",
    ],
    "melanoma": [
        "BRAF", "NRAS", "KIT", "MITF", "SOX10", "PAX3",
        "TYRP1", "DCT", "MLANA", "S100B", "MET", "MYC",
        "SNAI2", "VIM", "ZEB2", "CD44", "CDKN2A",
        "PTEN", "AKT1",
    ],
}

NORMAL_MARKERS: Dict[str, List[str]] = {
    "colon": [
        "CDX2", "MUC2", "FABP1", "CA1", "CA2", "CEACAM5", "EPCAM",
        "KRT8", "KRT18", "KRT19", "VIL1", "SLC26A3", "AQP8",
        "SI", "LCT", "DEFA5", "DEFA6",
    ],
    "breast": [
        "ESR1", "PGR", "FOXA1", "GATA3", "TFF1", "TFF3", "KRT8",
        "KRT18", "KRT19", "EPCAM", "CDH1", "MUC1",
        "ACTA2", "TPM2", "MYH11",
    ],
    "lung": [
        "SFTPA1", "SFTPB", "SFTPC", "SFTPD", "NKX2-1", "FOXA2",
        "KRT5", "TP63", "HOPX", "AGER", "PDPN",
        "SCGB1A1", "CC10", "CCSP",
    ],
    "brain": [
        "GFAP", "S100B", "VIM", "ALDH1A1", "SLC1A2", "AQP4",
        "OLIG2", "MBP", "PLP1", "MAP2", "SYP", "RBFOX3",
        "TUBB3", "DCX",
    ],
    "blood": [
        "HBB", "HBA1", "HBA2", "GYPA", "TFRC", "CD34", "CD38",
        "MPO", "ELANE", "AZU1", "LYZ", "S100A8", "S100A9",
        "CEBPA", "SPI1",
    ],
    "skin": [
        "KRT1", "KRT10", "KRT14", "KRT5", "IVL", "FLG", "LOR",
        "TYRP1", "DCT", "MITF", "S100B",
        "KRT6A", "KRT16",
    ],
}

# Cancer driver transcription factors per cancer type
CANCER_DRIVER_TFS: Dict[str, List[str]] = {
    "colorectal": [
        "MYC", "TP53", "CDX2", "SNAI1", "SNAI2", "ZEB1",
        "HIF1A", "STAT3", "NFKB1", "CTNNB1", "TCF7L2",
    ],
    "leukemia_aml": [
        "RUNX1", "CEBPA", "SPI1", "IRF8", "MYC", "TP53",
        "HOXA9", "MEIS1", "FLT3", "TAL1", "LMO2",
    ],
    "breast": [
        "ESR1", "FOXA1", "GATA3", "MYC", "SNAI1", "TWIST1",
        "ZEB1", "TP53", "ERBB2",
    ],
    "lung": [
        "NKX2-1", "TP63", "FOXA2", "MYC", "EGFR", "KRAS",
        "BRAF", "SOX2",
    ],
    "glioblastoma": [
        "SOX2", "OLIG2", "MYC", "EGFR", "TP53", "PTEN",
        "IDH1", "NESTIN",
    ],
    "melanoma": [
        "MITF", "SOX10", "PAX3", "BRAF", "NRAS", "MYC",
        "SNAI2", "ZEB2",
    ],
}


class AttractorClassifier:
    """
    Classify attractor states as 'normal', 'cancer', or 'transitional'.

    Parameters
    ----------
    cancer_type : str
        Cancer type key (e.g. 'colorectal', 'leukemia_aml').
    tissue : str
        Tissue/organ key (e.g. 'colon', 'breast').
    """

    def __init__(self, cancer_type: str = "colorectal", tissue: str = "colon"):
        self.cancer_type = cancer_type
        self.tissue = tissue
        self.normal_markers = self._load_normal_markers()
        self.cancer_markers = self._load_cancer_markers()
        self.cancer_driver_tfs = self._load_cancer_drivers()

        logger.info(
            "AttractorClassifier: cancer_type=%s, tissue=%s, "
            "%d normal markers, %d cancer markers, %d driver TFs.",
            cancer_type,
            tissue,
            len(self.normal_markers),
            len(self.cancer_markers),
            len(self.cancer_driver_tfs),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        attractors: List[np.ndarray],
        genes: List[str],
    ) -> List[str]:
        """
        Classify each attractor as 'normal', 'cancer', or 'transitional'.

        Parameters
        ----------
        attractors : List[np.ndarray]
            List of attractor state vectors. For Boolean attractors,
            values in {0, 1}; for continuous, values in [0, 1].
        genes : List[str]
            Gene names corresponding to indices of the state vectors.

        Returns
        -------
        List[str]
            One label per attractor.
        """
        if len(attractors) == 0:
            return []

        labels = []
        for idx, attractor in enumerate(attractors):
            score = self._compute_cancer_score(attractor, genes)
            if score >= 0.65:
                label = "cancer"
            elif score <= 0.35:
                label = "normal"
            else:
                label = "transitional"
            labels.append(label)
            logger.debug(
                "Attractor %d: cancer_score=%.3f -> %s.", idx, score, label
            )

        n_cancer = labels.count("cancer")
        n_normal = labels.count("normal")
        n_trans = labels.count("transitional")
        logger.info(
            "Attractor classification: %d cancer, %d normal, %d transitional.",
            n_cancer,
            n_normal,
            n_trans,
        )
        return labels

    def _compute_cancer_score(
        self,
        attractor: np.ndarray,
        genes: List[str],
    ) -> float:
        """
        Compute a cancer score for an attractor state.

        Score = (cancer_marker_fraction - normal_marker_fraction + 1) / 2

        This maps the range [-1, 1] to [0, 1] where:
            1.0 = purely cancer-like
            0.0 = purely normal-like
            0.5 = neutral / mixed

        Parameters
        ----------
        attractor : np.ndarray
            Attractor state vector (values in {0,1} or [0,1]).
        genes : List[str]
            Gene names for each dimension.

        Returns
        -------
        float
            Cancer score in [0, 1].
        """
        gene_set = {g: i for i, g in enumerate(genes)}

        # Threshold continuous values at 0.5 to compute fractions
        binary = (attractor >= 0.5).astype(float)

        # Cancer marker fraction
        cancer_indices = [
            gene_set[g] for g in self.cancer_markers if g in gene_set
        ]
        if len(cancer_indices) > 0:
            cancer_frac = float(np.mean(binary[cancer_indices]))
        else:
            cancer_frac = 0.5  # no markers available, neutral

        # Normal marker fraction
        normal_indices = [
            gene_set[g] for g in self.normal_markers if g in gene_set
        ]
        if len(normal_indices) > 0:
            normal_frac = float(np.mean(binary[normal_indices]))
        else:
            normal_frac = 0.5  # no markers available, neutral

        score = (cancer_frac - normal_frac + 1.0) / 2.0
        return float(np.clip(score, 0.0, 1.0))

    def get_cancer_normal_pair(
        self,
        attractors: List[np.ndarray],
        labels: List[str],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Select the best cancer-normal attractor pair.

        'Best' cancer = highest cancer score.
        'Best' normal = highest normal score (lowest cancer score).

        Parameters
        ----------
        attractors : List[np.ndarray]
        labels : List[str]
            Classification labels parallel to `attractors`.

        Returns
        -------
        (cancer_attractor, normal_attractor)
            Either or both may be None if no suitable attractor exists.
        """
        cancer_candidates = [
            (i, a)
            for i, (a, l) in enumerate(zip(attractors, labels))
            if l == "cancer"
        ]
        normal_candidates = [
            (i, a)
            for i, (a, l) in enumerate(zip(attractors, labels))
            if l == "normal"
        ]

        # Fall back to transitional if no pure cancer/normal exists
        if len(cancer_candidates) == 0:
            logger.warning(
                "No cancer attractors found; using transitional attractor as cancer proxy."
            )
            cancer_candidates = [
                (i, a)
                for i, (a, l) in enumerate(zip(attractors, labels))
                if l == "transitional"
            ]

        if len(normal_candidates) == 0:
            logger.warning(
                "No normal attractors found; using transitional attractor as normal proxy."
            )
            normal_candidates = [
                (i, a)
                for i, (a, l) in enumerate(zip(attractors, labels))
                if l == "transitional"
            ]

        best_cancer = (
            max(cancer_candidates, key=lambda t: float(np.mean(t[1] >= 0.5)))[1]
            if cancer_candidates
            else None
        )
        best_normal = (
            min(normal_candidates, key=lambda t: float(np.mean(t[1] >= 0.5)))[1]
            if normal_candidates
            else None
        )

        return best_cancer, best_normal

    # ------------------------------------------------------------------
    # Knowledge loaders
    # ------------------------------------------------------------------

    def _load_normal_markers(self) -> List[str]:
        """Return tissue-specific normal cell marker genes."""
        return NORMAL_MARKERS.get(self.tissue, NORMAL_MARKERS["colon"])

    def _load_cancer_markers(self) -> List[str]:
        """Return cancer-type-specific oncogene markers."""
        return CANCER_MARKERS.get(self.cancer_type, CANCER_MARKERS["colorectal"])

    def _load_cancer_drivers(self) -> List[str]:
        """Return cancer driver transcription factors for the cancer type."""
        return CANCER_DRIVER_TFS.get(self.cancer_type, CANCER_DRIVER_TFS["colorectal"])
