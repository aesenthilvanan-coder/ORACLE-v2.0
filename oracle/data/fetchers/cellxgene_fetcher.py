"""
CellxGeneFetcher – fetches scRNA-seq data from the CELLxGENE portal.

API reference: https://api.cellxgene.cziscience.com/curation/v1
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import List, Optional

import requests  # type: ignore

logger = logging.getLogger(__name__)

BASE_URL = "https://api.cellxgene.cziscience.com/curation/v1"


class CellxGeneFetcher:
    """Fetches scRNA-seq datasets from the CELLxGENE Discover portal.

    Parameters
    ----------
    cache_dir:
        Local directory for caching downloaded h5ad files.
    """

    def __init__(self, cache_dir: str = "./cache/cellxgene") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_dataset(self, dataset_id: str):
        """Fetch a dataset from CELLxGENE by its dataset ID.

        Checks the cache first; downloads the h5ad file on a cache miss.

        Parameters
        ----------
        dataset_id:
            CELLxGENE dataset UUID.

        Returns
        -------
        anndata.AnnData
        """
        try:
            import scanpy as sc  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "scanpy is required for fetch_dataset. "
                "Install with: pip install scanpy"
            ) from exc

        cached_path = self.cache_dir / f"{dataset_id}.h5ad"
        if cached_path.exists():
            logger.info("Loading cached dataset: %s", cached_path)
            return sc.read_h5ad(str(cached_path))

        # Download
        self.download_h5ad(dataset_id, str(cached_path))
        adata = sc.read_h5ad(str(cached_path))
        return adata

    def search_cancer(self, cancer_type: str) -> List[dict]:
        """Search CELLxGENE datasets by disease annotation.

        Queries the /datasets endpoint (returns a list) and filters by disease
        label using a case-insensitive substring match.

        Parameters
        ----------
        cancer_type:
            Human-readable cancer type string, e.g. ``"colorectal cancer"``.

        Returns
        -------
        list of dict
            Each dict contains ``id``, ``title``, ``disease``,
            ``organism``, and ``cell_count`` keys.
        """
        endpoint = f"{BASE_URL}/datasets"

        try:
            response = self._session.get(endpoint, timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logger.error("CELLxGENE search failed for '%s': %s", cancer_type, exc)
            return []

        # The curation API returns a list directly (not a dict wrapper)
        datasets = data if isinstance(data, list) else data.get("datasets", [])

        results = []
        for ds in datasets:
            diseases = ds.get("disease", [])
            if isinstance(diseases, str):
                diseases = [diseases]
            disease_labels = [
                d.get("label", d) if isinstance(d, dict) else str(d)
                for d in diseases
            ]
            if any(cancer_type.lower() in label.lower() for label in disease_labels):
                results.append(
                    {
                        "id": ds.get("dataset_id", ds.get("id", "")),
                        "title": ds.get("title", ""),
                        "disease": disease_labels,
                        "organism": ds.get("organism", []),
                        "cell_count": ds.get("cell_count", 0),
                    }
                )

        logger.info(
            "Found %d dataset(s) matching cancer type '%s'",
            len(results),
            cancer_type,
        )
        return results

    def download_h5ad(self, dataset_id: str, dest_path: str) -> None:
        """Download the h5ad file for a given dataset version ID.

        Queries the dataset_versions endpoint to resolve the asset download URL.
        ``dataset_id`` may be either a dataset UUID or a dataset_version UUID;
        both are tried.

        Parameters
        ----------
        dataset_id:
            CELLxGENE dataset (or dataset_version) UUID.
        dest_path:
            Local destination path for the h5ad file.
        """
        h5ad_url: Optional[str] = None

        # Try dataset_versions endpoint first (works for version IDs)
        for endpoint in [
            f"{BASE_URL}/dataset_versions/{dataset_id}",
            f"{BASE_URL}/datasets/{dataset_id}",
        ]:
            try:
                resp = self._session.get(endpoint, timeout=30)
                if resp.status_code != 200:
                    continue
                info = resp.json()
                assets = info.get("assets", [])
                for asset in assets:
                    if asset.get("filetype", "").lower() == "h5ad":
                        h5ad_url = asset.get("url") or asset.get("presigned_url")
                        break
                if h5ad_url:
                    break
            except requests.RequestException:
                continue

        if h5ad_url is None:
            # Direct CDN fallback: CZI hosts datasets at a predictable URL
            h5ad_url = f"https://datasets.cellxgene.cziscience.com/{dataset_id}.h5ad"
            logger.warning(
                "Could not resolve asset URL via API; trying CDN fallback: %s",
                h5ad_url,
            )

        logger.info("Downloading h5ad from %s → %s", h5ad_url, dest_path)
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            from tqdm import tqdm  # type: ignore

            with tqdm(unit="B", unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
                def _hook(count, block_size, total_size):
                    if total_size > 0 and pbar.total is None:
                        pbar.total = total_size
                    pbar.update(block_size)

                urllib.request.urlretrieve(h5ad_url, dest_path, reporthook=_hook)
        except ImportError:
            urllib.request.urlretrieve(h5ad_url, dest_path)

        logger.info("Download complete: %s", dest_path)
