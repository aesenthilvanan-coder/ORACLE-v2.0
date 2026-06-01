"""
Central data loading utilities for the ORACLE pipeline.

OracleDataLoader   – full-featured loader with caching, scanpy, networkx I/O.
MemoryEfficientDataLoader – streaming loader for large scRNA-seq datasets.
  M1 Macs have 8-64 GB unified memory; stream in batches to avoid OOM.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Optional

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# OracleDataLoader
# ---------------------------------------------------------------------------

class OracleDataLoader:
    """Central data loading utility for the ORACLE pipeline.

    Handles scRNA-seq (AnnData / h5ad), gene-regulatory networks (JSON /
    pickle), and CAMOutput serialisation.  A simple file-system cache avoids
    redundant computation across pipeline runs.
    """

    def __init__(self, config) -> None:
        self.config = config
        # Support omegaconf DictConfig, plain dicts, or objects with attributes
        if hasattr(config, "cache_dir"):
            cache_dir = config.cache_dir
        elif hasattr(config, "get"):
            cache_dir = config.get("cache_dir", "./cache/oracle")
        else:
            cache_dir = "./cache/oracle"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # scRNA-seq
    # ------------------------------------------------------------------

    def load_scrna(self, path: str):
        """Load an h5ad file via scanpy and return an AnnData object.

        Parameters
        ----------
        path:
            Absolute or relative path to a ``.h5ad`` file.

        Returns
        -------
        anndata.AnnData
        """
        try:
            import scanpy as sc  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "scanpy is required for load_scrna. "
                "Install with: pip install scanpy"
            ) from exc

        path = str(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"scRNA-seq file not found: {path}")

        adata = sc.read_h5ad(path)
        return adata

    # ------------------------------------------------------------------
    # Gene Regulatory Networks
    # ------------------------------------------------------------------

    def load_grn(self, path: str):
        """Load a GRN from a JSON or pickle file.

        JSON files should encode a node-link format (``nx.node_link_data``).
        Pickle files are loaded directly.

        Returns
        -------
        networkx.DiGraph
        """
        try:
            import networkx as nx  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "networkx is required for load_grn. "
                "Install with: pip install networkx"
            ) from exc

        path = str(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"GRN file not found: {path}")

        ext = Path(path).suffix.lower()
        if ext == ".json":
            with open(path, "r") as fh:
                data = json.load(fh)
            grn = nx.node_link_graph(data, directed=True, multigraph=False)
        elif ext in {".pkl", ".pickle"}:
            with open(path, "rb") as fh:
                grn = pickle.load(fh)
        else:
            raise ValueError(
                f"Unsupported GRN file format: {ext}. "
                "Use .json (node-link) or .pkl / .pickle."
            )

        if not isinstance(grn, nx.DiGraph):
            raise TypeError(
                f"Expected networkx.DiGraph, got {type(grn).__name__}"
            )

        return grn

    # ------------------------------------------------------------------
    # CAMOutput serialisation
    # ------------------------------------------------------------------

    def save_cam_output(self, cam_output, path: str) -> None:
        """Persist a CAMOutput object to disk using pickle.

        Parameters
        ----------
        cam_output:
            A ``CAMOutput`` (or any picklable) object produced by the CAM
            module.
        path:
            Destination file path.  Parent directories are created if they
            do not exist.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(cam_output, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load_cam_output(self, path: str):
        """Load a CAMOutput from disk.

        Returns
        -------
        CAMOutput
        """
        path = str(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"CAMOutput file not found: {path}")
        with open(path, "rb") as fh:
            return pickle.load(fh)

    # ------------------------------------------------------------------
    # Caching helper
    # ------------------------------------------------------------------

    def cache(self, key: str, fn: Callable[[], Any]) -> Any:
        """Cache-or-compute helper.

        If a pickle file for *key* exists in ``cache_dir`` the result is
        loaded from disk.  Otherwise *fn* is called and its return value is
        serialised for future calls.

        Parameters
        ----------
        key:
            Cache key string.  Should be a valid filename stem (no path
            separators).
        fn:
            Zero-argument callable that produces the value to cache.

        Returns
        -------
        Any
            The cached or freshly computed value.
        """
        # Sanitise key to be filesystem-safe
        safe_key = key.replace("/", "_").replace("\\", "_")
        cache_path = self.cache_dir / f"{safe_key}.pkl"

        if cache_path.exists():
            with open(cache_path, "rb") as fh:
                return pickle.load(fh)

        result = fn()
        with open(cache_path, "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)

        return result


# ---------------------------------------------------------------------------
# MemoryEfficientDataLoader
# ---------------------------------------------------------------------------

class MemoryEfficientDataLoader:
    """Streaming loader for large scRNA-seq h5ad datasets.

    Streams cells in fixed-size batches using h5py directly, avoiding the need
    to load the full expression matrix into memory.  Particularly important on
    M1/M2/M3 Macs with shared DRAM (8–64 GB).

    The loader expects the standard AnnData on-disk layout produced by
    ``adata.write_h5ad()``:

        /X           – expression matrix (dense float32 or sparse CSR)
        /obs         – cell metadata
        /var         – gene metadata

    For CSR sparse matrices the data is stored under ``/X/data``,
    ``/X/indices``, and ``/X/indptr``.

    Parameters
    ----------
    h5ad_path:
        Path to the ``.h5ad`` file.
    batch_size:
        Number of cells per batch.  Defaults to 512.
    """

    def __init__(self, h5ad_path: str, batch_size: int = 512) -> None:
        self.h5ad_path = str(h5ad_path)
        self.batch_size = batch_size

        if not os.path.exists(self.h5ad_path):
            raise FileNotFoundError(f"h5ad file not found: {self.h5ad_path}")

        # Probe the file to determine matrix shape and storage format
        with h5py.File(self.h5ad_path, "r") as f:
            x_node = f["X"]
            if isinstance(x_node, h5py.Dataset):
                # Dense storage
                self._sparse = False
                self._n_cells, self._n_genes = x_node.shape
            else:
                # Sparse CSR group
                self._sparse = True
                indptr = x_node["indptr"]
                # Number of rows = len(indptr) - 1
                self._n_cells = len(indptr) - 1
                # Infer n_genes from /var
                self._n_genes = len(f["var"]["_index"])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        """Total number of cells in the dataset."""
        return self._n_cells

    @property
    def n_genes(self) -> int:
        """Number of genes (features)."""
        return self._n_genes

    # ------------------------------------------------------------------
    # Streaming iterator
    # ------------------------------------------------------------------

    def iter_batches(self):
        """Yield expression matrix batches as dense numpy float32 arrays.

        Each yielded value is a 2-D array of shape
        ``(min(batch_size, remaining_cells), n_genes)`` with dtype
        ``numpy.float32``.

        Yields
        ------
        numpy.ndarray
            Dense expression matrix for the current batch of cells.
        """
        with h5py.File(self.h5ad_path, "r") as f:
            x_node = f["X"]

            for start in range(0, self._n_cells, self.batch_size):
                end = min(start + self.batch_size, self._n_cells)

                if not self._sparse:
                    # Dense: simple slice
                    batch = x_node[start:end].astype(np.float32)
                else:
                    # Sparse CSR: reconstruct dense batch
                    indptr = x_node["indptr"][start : end + 1]
                    data_start = int(indptr[0])
                    data_end = int(indptr[-1])

                    values = x_node["data"][data_start:data_end].astype(
                        np.float32
                    )
                    col_indices = x_node["indices"][data_start:data_end]

                    # Adjust indptr to be relative to data_start
                    local_indptr = (indptr - indptr[0]).astype(np.int32)

                    batch_rows = end - start
                    batch = np.zeros(
                        (batch_rows, self._n_genes), dtype=np.float32
                    )
                    for row_i in range(batch_rows):
                        row_start = local_indptr[row_i]
                        row_end = local_indptr[row_i + 1]
                        batch[row_i, col_indices[row_start:row_end]] = values[
                            row_start:row_end
                        ]

                yield batch

    def __iter__(self):
        return self.iter_batches()

    def __len__(self) -> int:
        """Number of batches."""
        return (self._n_cells + self.batch_size - 1) // self.batch_size

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MemoryEfficientDataLoader("
            f"cells={self._n_cells}, genes={self._n_genes}, "
            f"batch_size={self.batch_size})"
        )
