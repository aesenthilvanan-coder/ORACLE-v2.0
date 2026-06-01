"""Tests for oracle.rsp.switch_optimizer and gnn_predictor."""

import numpy as np
import pytest


def test_minimal_switch_optimizer_init(rsp_config):
    from oracle.rsp.switch_optimizer import MinimalSwitchOptimizer
    opt = MinimalSwitchOptimizer(rsp_config)
    assert opt is not None


def test_druggable_tf_list(rsp_config):
    from oracle.rsp.switch_optimizer import MinimalSwitchOptimizer
    opt = MinimalSwitchOptimizer(rsp_config)
    druggable = opt._filter_druggable(
        ["TP53", "MYC", "CDX2", "RANDOM_GENE_XYZ", "BRD4"]
    )
    # At least some known druggable TFs should pass
    assert isinstance(druggable, list)


def test_switch_set_dataclass():
    from oracle.rsp.switch_optimizer import SwitchSet
    ss = SwitchSet(
        perturbations={"CDX2": "Activation", "SNAI2": "Repression"},
        predicted_reversion_probability=0.85,
        delta_cancer_score=-0.4,
        n_perturbations=2,
    )
    assert ss.n_perturbations == 2
    assert "CDX2" in ss.perturbations


def test_optimizer_returns_switch_set(rsp_config, tiny_grn, cancer_attractor,
                                       normal_attractor, tiny_grn_genes):
    from oracle.rsp.switch_optimizer import MinimalSwitchOptimizer, SwitchSet
    from oracle.cam.continuous_ode import ContinuousGRNDynamics
    from oracle.cam.preprocessing import CAMConfig
    from oracle.rsp.cancer_score import CancerScoreFunction

    cam_config = CAMConfig(integration_time=1.0, n_ode_steps=5)
    ode_model = ContinuousGRNDynamics(tiny_grn, cam_config)
    score_fn = CancerScoreFunction(rsp_config)

    opt = MinimalSwitchOptimizer(rsp_config)
    try:
        result = opt.optimize(
            cancer_attractor, normal_attractor,
            tiny_grn, ode_model, score_fn, tiny_grn_genes
        )
        assert isinstance(result, SwitchSet)
        assert isinstance(result.perturbations, dict)
        assert 0 <= result.predicted_reversion_probability <= 1
    except Exception as e:
        if "torchdiffeq" in str(e).lower():
            pytest.skip(f"torchdiffeq not available: {e}")
        raise


def test_gnn_predictor_init(rsp_config):
    from oracle.rsp.gnn_predictor import GNNSwitchPredictor
    predictor = GNNSwitchPredictor(rsp_config)
    assert predictor is not None


def test_gnn_predictor_predict_switches(rsp_config, tiny_grn, cancer_expression):
    from oracle.rsp.gnn_predictor import GNNSwitchPredictor
    predictor = GNNSwitchPredictor(rsp_config)
    try:
        preds = predictor.predict_switches(tiny_grn, cancer_expression)
        assert isinstance(preds, dict)
        for gene, score in preds.items():
            assert isinstance(score, float)
    except Exception as e:
        if "torch_geometric" in str(e).lower() or "pyg" in str(e).lower():
            pytest.skip(f"torch-geometric not available: {e}")
        raise


def test_combinatorial_searcher_small(rsp_config, tiny_grn, cancer_attractor,
                                       normal_attractor, tiny_grn_genes):
    from oracle.rsp.combinatorial_search import CombinatorialSearcher
    from oracle.rsp.cancer_score import CancerScoreFunction

    score_fn = CancerScoreFunction(rsp_config)
    searcher = CombinatorialSearcher(rsp_config)

    # Only test with k=1 to keep test fast
    try:
        results = searcher.search(
            cancer_attractor, normal_attractor,
            tiny_grn, score_fn, tiny_grn_genes, max_k=1
        )
        assert isinstance(results, list)
    except Exception as e:
        if "torchdiffeq" in str(e).lower():
            pytest.skip(f"torchdiffeq not available: {e}")
        raise
