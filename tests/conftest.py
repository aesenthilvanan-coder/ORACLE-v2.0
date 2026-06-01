"""Shared pytest fixtures for the ORACLE test suite."""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Device fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Minimal GRN (NetworkX DiGraph)
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_grn():
    """A 6-node toy GRN for fast testing."""
    import networkx as nx

    G = nx.DiGraph()
    # nodes: A (TF), B (TF), C, D, E, F
    genes = ["GeneA", "GeneB", "GeneC", "GeneD", "GeneE", "GeneF"]
    G.add_nodes_from(genes)

    # edges: sign=+1 activation, sign=-1 repression
    edges = [
        ("GeneA", "GeneC", {"sign": 1, "weight": 0.8, "confidence": 0.9}),
        ("GeneA", "GeneD", {"sign": 1, "weight": 0.6, "confidence": 0.7}),
        ("GeneB", "GeneC", {"sign": -1, "weight": 0.5, "confidence": 0.6}),
        ("GeneB", "GeneE", {"sign": 1, "weight": 0.7, "confidence": 0.8}),
        ("GeneC", "GeneF", {"sign": 1, "weight": 0.4, "confidence": 0.5}),
        ("GeneD", "GeneF", {"sign": -1, "weight": 0.3, "confidence": 0.4}),
    ]
    G.add_edges_from(edges)
    return G


@pytest.fixture
def tiny_grn_genes(tiny_grn):
    return list(tiny_grn.nodes())


# ---------------------------------------------------------------------------
# CAM config / data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cam_config():
    from oracle.cam.preprocessing import CAMConfig
    return CAMConfig(
        min_genes=5,        # match the 10-gene minimal_adata fixture
        min_cells=1,
        n_attractor_samples=50,
        n_basin_samples=200,
        max_trajectory_steps=20,
        integration_time=5.0,
        n_ode_steps=10,
        n_jobs=1,
        cancer_type="colorectal",
        tissue="colon",
    )


@pytest.fixture
def minimal_adata():
    """Minimal AnnData with 20 cells x 10 genes for fast tests."""
    try:
        import anndata as ad
        import scanpy as sc

        np.random.seed(42)
        n_obs, n_vars = 20, 10
        X = np.random.poisson(5.0, (n_obs, n_vars)).astype(np.float32)
        obs_names = [f"cell_{i}" for i in range(n_obs)]
        var_names = [f"GeneA", f"GeneB", f"GeneC", f"GeneD", f"GeneE",
                     f"GeneF", f"GeneG", f"GeneH", f"GeneI", f"GeneJ"]
        adata = ad.AnnData(X=X)
        adata.obs_names = obs_names
        adata.var_names = var_names
        adata.obs["cell_state"] = ["cancer"] * 12 + ["normal"] * 8
        adata.obs["leiden"] = ["0"] * 10 + ["1"] * 10
        return adata
    except ImportError:
        pytest.skip("anndata not installed")


# ---------------------------------------------------------------------------
# RSP config / data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rsp_config(tiny_grn_genes):
    from oracle.rsp.cancer_score import RSPConfig
    return RSPConfig(n_genes=len(tiny_grn_genes))


@pytest.fixture
def cancer_expression(tiny_grn_genes):
    """Mock cancer-state expression for all genes."""
    rng = np.random.default_rng(42)
    return {g: float(rng.uniform(0.5, 1.0)) for g in tiny_grn_genes}


@pytest.fixture
def normal_expression(tiny_grn_genes):
    """Mock normal-state expression for all genes."""
    rng = np.random.default_rng(0)
    return {g: float(rng.uniform(0.0, 0.5)) for g in tiny_grn_genes}


@pytest.fixture
def cancer_attractor(tiny_grn_genes):
    rng = np.random.default_rng(42)
    return rng.choice([0, 1], size=len(tiny_grn_genes)).astype(np.float32)


@pytest.fixture
def normal_attractor(tiny_grn_genes):
    rng = np.random.default_rng(99)
    return rng.choice([0, 1], size=len(tiny_grn_genes)).astype(np.float32)


# ---------------------------------------------------------------------------
# TCD config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tcd_config():
    from oracle.tcd.tf_structurer import TCDConfig
    return TCDConfig(
        n_warhead_candidates=3,
        md_frames=2,
    )


# ---------------------------------------------------------------------------
# Molecule fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def example_smiles():
    return [
        "CCO",
        "c1ccccc1",
        "CC(=O)O",
        "C1CCCCC1",
        "c1ccc(cc1)C(=O)O",
    ]
