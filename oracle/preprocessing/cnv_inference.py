import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_GENE_CHR_POSITIONS: Optional[pd.DataFrame] = None


def load_gene_chromosome_positions() -> pd.DataFrame:
    global _GENE_CHR_POSITIONS
    if _GENE_CHR_POSITIONS is not None:
        return _GENE_CHR_POSITIONS

    bundled = Path(__file__).parent.parent / "data" / "gene_chromosome_positions.csv"
    if bundled.exists():
        _GENE_CHR_POSITIONS = pd.read_csv(bundled)
        logger.info(f"Loaded {len(_GENE_CHR_POSITIONS)} gene positions from bundled CSV")
        return _GENE_CHR_POSITIONS

    try:
        import requests
        url = (
            "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz"
        )
        logger.info("Downloading gene chromosome positions from NCBI...")
        resp = requests.get(url, timeout=60, stream=True)
        import io, gzip
        with gzip.open(io.BytesIO(resp.content)) as f:
            df = pd.read_csv(f, sep="\t", usecols=["Symbol", "chromosome", "map_location"],
                             low_memory=False)
        df = df[df["#tax_id"] == 9606] if "#tax_id" in df.columns else df
        df = df.rename(columns={"Symbol": "gene", "chromosome": "chr"})
        df = df[["gene", "chr"]].dropna()
        _GENE_CHR_POSITIONS = df
        logger.info(f"Downloaded {len(df)} gene positions from NCBI")
        return df
    except Exception as e:
        logger.warning(f"Could not load gene chromosome positions: {e}")
        _GENE_CHR_POSITIONS = pd.DataFrame(columns=["gene", "chr"])
        return _GENE_CHR_POSITIONS


class SimpleCNVScorer:
    """Chromosomal smoothing-based CNV scorer from scRNA-seq data."""

    def __init__(self, window_size: int = 100, reference_group: str = "normal"):
        self.window_size = window_size
        self.reference_group = reference_group

    def compute(self, adata) -> np.ndarray:
        import anndata as ad
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.array(X, dtype=np.float32)

        smoothed = self._moving_average(X, self.window_size)

        ref_mask = np.zeros(adata.n_obs, dtype=bool)
        if self.reference_group in adata.obs.columns:
            ref_mask = adata.obs[self.reference_group].values.astype(bool)

        if ref_mask.sum() > 0:
            ref_mean = smoothed[ref_mask].mean(axis=0, keepdims=True)
        else:
            ref_mean = smoothed.mean(axis=0, keepdims=True)

        deviation = smoothed - ref_mean
        cnv_score = np.mean(deviation ** 2, axis=1)
        cnv_score = (cnv_score - cnv_score.min()) / (cnv_score.max() - cnv_score.min() + 1e-8)
        return cnv_score.astype(np.float32)

    def _moving_average(self, X: np.ndarray, window: int) -> np.ndarray:
        n_cells, n_genes = X.shape
        w = min(window, n_genes)
        cumsum = np.cumsum(X, axis=1)
        result = np.zeros_like(X)
        result[:, w - 1:] = (cumsum[:, w - 1:] - np.hstack([np.zeros((n_cells, 1)), cumsum[:, :-1]])[:, :n_genes - w + 1]) / w
        result[:, :w - 1] = X[:, :w - 1]
        return result
