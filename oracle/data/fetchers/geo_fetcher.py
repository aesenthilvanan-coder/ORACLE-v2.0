"""
GEOFetcher – downloads and caches scRNA-seq datasets from NCBI GEO.

Supports:
- Direct h5 / h5ad supplementary files
- 10X MTX format (matrix.mtx, barcodes.tsv, features.tsv)
"""

from __future__ import annotations

import ftplib
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated cancer-type panels
# ---------------------------------------------------------------------------

CANCER_GEO_PANELS: dict = {
    "colorectal": ["GSE132465", "GSE166555", "GSE200997"],
    "breast": ["GSE176078", "GSE161529"],
    "lung": ["GSE127465", "GSE149655"],
    "leukemia_aml": ["GSE116256", "GSE13159"],
    "glioblastoma": ["GSE84465", "GSE131928"],
    "melanoma": ["GSE72056", "GSE115978"],
}

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/"
GEO_FTP_HOST = "ftp.ncbi.nlm.nih.gov"


class GEOFetcher:
    """Downloads and caches scRNA-seq datasets from NCBI GEO.

    Parameters
    ----------
    cache_dir:
        Local directory for caching downloaded files.
    """

    def __init__(self, cache_dir: str = "./cache/geo") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, accession: str, cancer_type: str):
        """Fetch a GEO dataset by accession number.

        Checks the local cache first.  On a cache miss:
        1. Lists supplementary files on the GEO FTP.
        2. Tries to download an ``.h5ad`` or ``.h5`` file.
        3. Falls back to 10X MTX format.

        Parameters
        ----------
        accession:
            GEO series accession, e.g. ``"GSE132465"``.
        cancer_type:
            Human-readable cancer type label stored in the AnnData object.

        Returns
        -------
        anndata.AnnData
        """
        try:
            import anndata as ad  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "anndata is required for GEOFetcher. "
                "Install with: pip install anndata"
            ) from exc

        acc_dir = self.cache_dir / accession
        acc_dir.mkdir(parents=True, exist_ok=True)

        # Check for cached AnnData
        cached_path = acc_dir / f"{accession}.h5ad"
        if cached_path.exists():
            logger.info("Loading cached dataset: %s", cached_path)
            import scanpy as sc  # type: ignore
            adata = sc.read_h5ad(str(cached_path))
            return adata

        # Discover supplementary files
        logger.info("Listing supplementary files for %s …", accession)
        try:
            supp_files = self._list_supplementary(accession)
        except Exception as exc:
            logger.warning("Could not list supplementary files: %s", exc)
            supp_files = []

        adata = None

        # Attempt h5ad
        h5ad_files = [f for f in supp_files if f.lower().endswith(".h5ad")]
        if h5ad_files:
            url = self._build_supp_url(accession, h5ad_files[0])
            dest = acc_dir / h5ad_files[0]
            try:
                self._download(url, str(dest))
                import scanpy as sc  # type: ignore
                adata = sc.read_h5ad(str(dest))
            except Exception as exc:
                logger.warning("Failed to load h5ad from %s: %s", url, exc)

        # Attempt plain h5
        if adata is None:
            h5_files = [f for f in supp_files if f.lower().endswith(".h5")]
            if h5_files:
                url = self._build_supp_url(accession, h5_files[0])
                dest = acc_dir / h5_files[0]
                try:
                    self._download(url, str(dest))
                    import scanpy as sc  # type: ignore
                    adata = sc.read_10x_h5(str(dest))
                except Exception as exc:
                    logger.warning("Failed to load h5 from %s: %s", url, exc)

        # Attempt 10X MTX
        if adata is None:
            try:
                adata = self._load_10x(accession, supp_files)
            except Exception as exc:
                logger.warning("Failed to load 10X MTX format: %s", exc)

        if adata is None:
            raise RuntimeError(
                f"Could not load any recognised format for {accession}. "
                "Check GEO for available supplementary files."
            )

        # Annotate and cache
        adata.obs["cancer_type"] = cancer_type
        adata.obs["geo_accession"] = accession
        adata.write_h5ad(str(cached_path))
        logger.info("Cached AnnData to %s", cached_path)
        return adata

    def fetch_cancer_panel(self, cancer_type: str):
        """Fetch the curated panel of GEO datasets for a cancer type.

        Parameters
        ----------
        cancer_type:
            Must be a key in ``CANCER_GEO_PANELS``.

        Returns
        -------
        list of anndata.AnnData
        """
        if cancer_type not in CANCER_GEO_PANELS:
            raise ValueError(
                f"Unknown cancer type: '{cancer_type}'. "
                f"Available: {list(CANCER_GEO_PANELS.keys())}"
            )

        accessions = CANCER_GEO_PANELS[cancer_type]
        results = []
        for acc in accessions:
            try:
                adata = self.fetch(acc, cancer_type)
                results.append(adata)
            except Exception as exc:
                logger.error("Failed to fetch %s: %s", acc, exc)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _list_supplementary(self, accession: str) -> List[str]:
        """Query the GEO FTP server and return supplementary file names."""
        # GEO FTP path: /geo/series/GNNNnnn/GSEnnnnn/suppl/
        stub = accession[:-3] + "nnn"  # e.g. GSE132nnn
        ftp_path = f"/geo/series/{stub}/{accession}/suppl/"

        try:
            ftp = ftplib.FTP(GEO_FTP_HOST, timeout=30)
            ftp.login()
            file_list = ftp.nlst(ftp_path)
            ftp.quit()
        except Exception as exc:
            raise RuntimeError(
                f"FTP listing failed for {accession}: {exc}"
            ) from exc

        # Return bare filenames
        return [os.path.basename(f) for f in file_list]

    def _build_supp_url(self, accession: str, filename: str) -> str:
        stub = accession[:-3] + "nnn"
        return f"{BASE_URL}{stub}/{accession}/suppl/{filename}"

    def _download(self, url: str, dest: str) -> None:
        """Download *url* to *dest* with a tqdm progress bar."""
        try:
            from tqdm import tqdm  # type: ignore
        except ImportError:
            tqdm = None  # type: ignore

        logger.info("Downloading %s → %s", url, dest)

        if tqdm is not None:
            with tqdm(unit="B", unit_scale=True, desc=os.path.basename(dest)) as pbar:
                def _hook(count, block_size, total_size):
                    if total_size > 0 and pbar.total is None:
                        pbar.total = total_size
                    pbar.update(block_size)

                urllib.request.urlretrieve(url, dest, reporthook=_hook)
        else:
            urllib.request.urlretrieve(url, dest)

    def _load_10x(self, accession: str, files: List[str]):
        """Load 10X MTX format supplementary files.

        Looks for ``matrix.mtx``, ``barcodes.tsv``, and ``features.tsv``
        (or ``genes.tsv``) among the supplementary file list.

        Parameters
        ----------
        accession:
            GEO accession.
        files:
            List of supplementary file names.

        Returns
        -------
        anndata.AnnData
        """
        try:
            import scanpy as sc  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "scanpy is required for _load_10x. "
                "Install with: pip install scanpy"
            ) from exc

        acc_dir = self.cache_dir / accession

        # Find relevant files (case-insensitive, possibly gzipped)
        def _find(pattern: str) -> Optional[str]:
            for f in files:
                if re.search(pattern, f, re.IGNORECASE):
                    return f
            return None

        matrix_file = _find(r"matrix\.mtx")
        barcode_file = _find(r"barcodes\.tsv")
        feature_file = _find(r"(features|genes)\.tsv")

        if not all([matrix_file, barcode_file, feature_file]):
            raise RuntimeError(
                "Could not find all 10X MTX files in supplementary list."
            )

        # Download if not cached
        for fname in [matrix_file, barcode_file, feature_file]:
            dest = acc_dir / fname
            if not dest.exists():
                url = self._build_supp_url(accession, fname)
                self._download(url, str(dest))

        # scanpy expects an uncompressed directory; handle gz
        mtx_dir = acc_dir / "mtx"
        mtx_dir.mkdir(exist_ok=True)

        import shutil

        for fname in [matrix_file, barcode_file, feature_file]:
            src = acc_dir / fname
            target_name = fname
            if fname.endswith(".gz"):
                import gzip
                target_name = fname[:-3]
                target_path = mtx_dir / target_name
                if not target_path.exists():
                    with gzip.open(str(src), "rb") as f_in, open(
                        str(target_path), "wb"
                    ) as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                target_path = mtx_dir / target_name
                if not target_path.exists():
                    shutil.copy2(str(src), str(target_path))

        # Rename genes.tsv → features.tsv if needed
        genes_path = mtx_dir / "genes.tsv"
        feat_path = mtx_dir / "features.tsv"
        if genes_path.exists() and not feat_path.exists():
            genes_path.rename(feat_path)

        adata = sc.read_10x_mtx(str(mtx_dir), var_names="gene_symbols")
        return adata
