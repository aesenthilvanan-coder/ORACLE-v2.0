"""
Cancer Attraction Mapper - Landscape Computer

Computes the attractor landscape: projects attractor states onto the
cell UMAP embedding and estimates the epigenetic energy surface via
kernel density estimation over the cell manifold.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from anndata import AnnData

from oracle.cam.preprocessing import CAMConfig

logger = logging.getLogger(__name__)


class LandscapeComputer:
    """
    Compute and store the Waddington-style attractor landscape.

    The landscape combines:
      - Projection of Boolean/continuous attractor states onto the
        cell UMAP embedding (via nearest-neighbor matching).
      - Energy surface estimated as -log(cell density) over the UMAP.
      - Basin annotations derived from attractor labels.

    Parameters
    ----------
    config : CAMConfig
        Pipeline configuration.
    """

    def __init__(self, config: CAMConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        adata: AnnData,
        bool_net: Any,          # BooleanNetworkSimulator
        ode_model: Any,         # ContinuousGRNDynamics
        attractors: List[np.ndarray],
        labels: List[str],
    ) -> Dict[str, Any]:
        """
        Assemble the full landscape dictionary.

        Parameters
        ----------
        adata : AnnData
            Preprocessed AnnData with UMAP embedding in .obsm['X_umap'].
        bool_net : BooleanNetworkSimulator
            Boolean network (used for gene list).
        ode_model : ContinuousGRNDynamics
            ODE model (used for gene list).
        attractors : List[np.ndarray]
            Attractor state vectors (Boolean or continuous).
        labels : List[str]
            Classification label per attractor.

        Returns
        -------
        Dict[str, Any]
            Keys:
              - 'landscape_embedding': (n_attractors, 2) UMAP coords
              - 'basin_sizes': dict of attractor index -> basin fraction
              - 'energy_surface': (n_grid, n_grid) energy array
              - 'umap_grid_x', 'umap_grid_y': grid coordinates
              - 'attractor_labels': list of labels
              - 'cell_umap': (n_cells, 2) cell UMAP coordinates
        """
        logger.info("Computing attractor landscape.")

        if len(attractors) == 0:
            logger.warning("No attractors provided; returning empty landscape.")
            return {
                "landscape_embedding": np.zeros((0, 2), dtype=np.float32),
                "basin_sizes": {},
                "energy_surface": np.zeros((50, 50), dtype=np.float32),
                "umap_grid_x": np.zeros(50, dtype=np.float32),
                "umap_grid_y": np.zeros(50, dtype=np.float32),
                "attractor_labels": [],
                "cell_umap": self._get_cell_umap(adata),
            }

        # 1. Project attractors onto UMAP
        landscape_embedding = self._project_attractors_to_umap(attractors, adata, bool_net)

        # 2. Compute energy surface over UMAP grid
        energy_surface, grid_x, grid_y = self._compute_energy_surface(adata)

        # 3. Estimate basin sizes as fraction of cells in each basin
        basin_sizes = self._estimate_landscape_basins(
            adata, landscape_embedding, labels
        )

        cell_umap = self._get_cell_umap(adata)

        landscape = {
            "landscape_embedding": landscape_embedding,
            "basin_sizes": basin_sizes,
            "energy_surface": energy_surface,
            "umap_grid_x": grid_x,
            "umap_grid_y": grid_y,
            "attractor_labels": labels,
            "cell_umap": cell_umap,
        }

        logger.info(
            "Landscape computed: %d attractors projected onto UMAP.",
            len(attractors),
        )
        return landscape

    # ------------------------------------------------------------------
    # Attractor projection
    # ------------------------------------------------------------------

    def _project_attractors_to_umap(
        self,
        attractors: List[np.ndarray],
        adata: AnnData,
        bool_net: Any,
    ) -> np.ndarray:
        """
        Project attractor state vectors onto the cell UMAP embedding.

        Strategy: for each attractor, find the cell whose gene expression
        profile (in PCA space) is closest to the attractor (Euclidean
        distance), then return that cell's UMAP coordinates.

        Parameters
        ----------
        attractors : List[np.ndarray]
            Attractor state vectors.
        adata : AnnData
            AnnData with UMAP (.obsm['X_umap']) and PCA (.obsm['X_pca']).
        bool_net : BooleanNetworkSimulator
            Provides gene list for matching attractor dimensions to genes.

        Returns
        -------
        np.ndarray
            Shape (n_attractors, 2) - UMAP coordinates per attractor.
        """
        if "X_umap" not in adata.obsm:
            logger.warning(
                "UMAP not found in adata.obsm; returning zero coordinates."
            )
            return np.zeros((len(attractors), 2), dtype=np.float32)

        cell_umap = adata.obsm["X_umap"]  # (n_cells, 2)
        n_attractors = len(attractors)
        embedding = np.zeros((n_attractors, 2), dtype=np.float32)

        # Get gene names from attractor (try bool_net.genes, else adata.var_names)
        try:
            attractor_genes = bool_net.genes
        except AttributeError:
            attractor_genes = list(adata.var_names)

        if "X_pca" in adata.obsm:
            # Project in PCA space: map attractor genes -> PCA loadings
            cell_repr = adata.obsm["X_pca"]  # (n_cells, n_pcs)
            # Map attractor expression to gene-level and then project to PCA
            # via the PCA components matrix stored in adata.varm['PCs']
            if "PCs" in adata.varm:
                pca_components = adata.varm["PCs"]  # (n_genes, n_pcs)
                var_names = list(adata.var_names)

                for idx, attr in enumerate(attractors):
                    # Build full-gene expression vector for the attractor
                    attr_expr = np.zeros(len(var_names), dtype=np.float32)
                    for gi, gene in enumerate(attractor_genes):
                        if gene in var_names:
                            vidx = var_names.index(gene)
                            attr_expr[vidx] = float(attr[gi])

                    # Project to PCA space
                    attr_pca = attr_expr @ pca_components  # (n_pcs,)

                    # Find nearest cell in PCA space
                    dists = np.sum((cell_repr - attr_pca) ** 2, axis=1)
                    nearest_cell = int(np.argmin(dists))
                    embedding[idx] = cell_umap[nearest_cell]
                return embedding

        # Fallback: use raw expression space
        import scipy.sparse as sp

        if sp.issparse(adata.X):
            cell_expr = adata.X.toarray()
        else:
            cell_expr = np.array(adata.X)

        var_names = list(adata.var_names)

        for idx, attr in enumerate(attractors):
            attr_expr = np.zeros(len(var_names), dtype=np.float32)
            for gi, gene in enumerate(attractor_genes):
                if gene in var_names:
                    vidx = var_names.index(gene)
                    attr_expr[vidx] = float(attr[gi])

            dists = np.sum((cell_expr - attr_expr) ** 2, axis=1)
            nearest_cell = int(np.argmin(dists))
            embedding[idx] = cell_umap[nearest_cell]

        return embedding

    # ------------------------------------------------------------------
    # Energy surface
    # ------------------------------------------------------------------

    def _compute_energy_surface(
        self,
        adata: AnnData,
        n_grid: int = 50,
    ) -> tuple:
        """
        Compute the energy surface as -log(cell density) over a UMAP grid.

        Uses Gaussian kernel density estimation on cell UMAP coordinates.

        Parameters
        ----------
        adata : AnnData
        n_grid : int
            Grid resolution.

        Returns
        -------
        (energy_surface, grid_x, grid_y)
            energy_surface: (n_grid, n_grid)
            grid_x, grid_y: 1D coordinate arrays of length n_grid
        """
        if "X_umap" not in adata.obsm:
            dummy = np.zeros((n_grid, n_grid), dtype=np.float32)
            return dummy, np.zeros(n_grid), np.zeros(n_grid)

        umap_coords = adata.obsm["X_umap"]  # (n_cells, 2)
        x = umap_coords[:, 0]
        y = umap_coords[:, 1]

        x_min, x_max = x.min() - 0.5, x.max() + 0.5
        y_min, y_max = y.min() - 0.5, y.max() + 0.5

        grid_x = np.linspace(x_min, x_max, n_grid)
        grid_y = np.linspace(y_min, y_max, n_grid)
        xx, yy = np.meshgrid(grid_x, grid_y)

        # Gaussian KDE
        try:
            from scipy.stats import gaussian_kde

            coords = np.vstack([x, y])
            kde = gaussian_kde(coords, bw_method="silverman")
            positions = np.vstack([xx.ravel(), yy.ravel()])
            density = kde(positions).reshape(n_grid, n_grid)
        except Exception as e:
            logger.warning("KDE failed (%s); using histogram approximation.", e)
            density, _, _ = np.histogram2d(x, y, bins=n_grid, density=True)
            density = density.T

        # Energy = -log(density + epsilon)
        epsilon = density.max() * 1e-4 + 1e-10
        energy_surface = -np.log(density + epsilon)
        # Normalize to [0, 1]
        energy_surface = (energy_surface - energy_surface.min()) / (
            energy_surface.max() - energy_surface.min() + 1e-10
        )

        return energy_surface.astype(np.float32), grid_x.astype(np.float32), grid_y.astype(np.float32)

    # ------------------------------------------------------------------
    # Basin estimation
    # ------------------------------------------------------------------

    def _estimate_landscape_basins(
        self,
        adata: AnnData,
        attractor_umap: np.ndarray,
        labels: List[str],
    ) -> Dict[int, float]:
        """
        Estimate basin sizes as the fraction of cells closer to each
        attractor than to any other, in UMAP space.

        Parameters
        ----------
        adata : AnnData
        attractor_umap : np.ndarray
            (n_attractors, 2) UMAP positions of attractors.
        labels : List[str]

        Returns
        -------
        Dict[int, float]
            Attractor index -> fraction of cells in basin.
        """
        if "X_umap" not in adata.obsm or len(attractor_umap) == 0:
            return {}

        cell_umap = adata.obsm["X_umap"]  # (n_cells, 2)
        n_cells = cell_umap.shape[0]
        n_attractors = attractor_umap.shape[0]

        basin_counts = {i: 0 for i in range(n_attractors)}

        if n_attractors == 1:
            basin_counts[0] = n_cells
        else:
            # For each cell, find nearest attractor
            for ci in range(n_cells):
                cell_pos = cell_umap[ci]
                dists = np.sum((attractor_umap - cell_pos) ** 2, axis=1)
                nearest = int(np.argmin(dists))
                basin_counts[nearest] += 1

        # Convert to fractions
        basin_fractions = {k: v / n_cells for k, v in basin_counts.items()}
        return basin_fractions

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _get_cell_umap(self, adata: AnnData) -> np.ndarray:
        """Return cell UMAP coordinates or zeros fallback."""
        if "X_umap" in adata.obsm:
            return adata.obsm["X_umap"].astype(np.float32)
        return np.zeros((adata.n_obs, 2), dtype=np.float32)
