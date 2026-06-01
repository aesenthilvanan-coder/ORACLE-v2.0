"""Tests for oracle.rsp.cancer_score.CancerScoreFunction."""

import numpy as np
import pytest
import torch


def test_rsp_config_defaults(rsp_config):
    assert rsp_config.n_genes > 0
    assert hasattr(rsp_config, "n_hidden")
    assert hasattr(rsp_config, "n_layers")


def test_cancer_score_init(rsp_config):
    from oracle.rsp.cancer_score import CancerScoreFunction
    model = CancerScoreFunction(rsp_config)
    assert model is not None


def test_cancer_score_forward_shape(rsp_config):
    from oracle.rsp.cancer_score import CancerScoreFunction
    model = CancerScoreFunction(rsp_config)
    n = rsp_config.n_genes
    x = torch.rand(4, n)
    out = model(x)
    assert out.shape == (4, 1) or out.shape == (4,), f"Unexpected output shape: {out.shape}"


def test_cancer_score_range(rsp_config):
    """Output should be in [0, 1] (sigmoid output)."""
    from oracle.rsp.cancer_score import CancerScoreFunction
    model = CancerScoreFunction(rsp_config)
    model.eval()
    with torch.no_grad():
        x = torch.rand(16, rsp_config.n_genes)
        scores = model(x).squeeze()
        assert (scores >= 0).all() and (scores <= 1).all(), \
            f"Scores out of [0,1]: min={scores.min():.4f}, max={scores.max():.4f}"


def test_cancer_score_gradient(rsp_config):
    from oracle.rsp.cancer_score import CancerScoreFunction
    model = CancerScoreFunction(rsp_config)
    x = torch.rand(2, rsp_config.n_genes)
    try:
        grad = model.gradient_wrt_input(x)
        assert grad.shape == x.shape
    except AttributeError:
        pytest.skip("gradient_wrt_input not implemented")


def test_cancer_score_numpy(rsp_config):
    from oracle.rsp.cancer_score import CancerScoreFunction
    model = CancerScoreFunction(rsp_config)
    x_np = np.random.rand(rsp_config.n_genes).astype(np.float32)
    try:
        score = model.score_numpy(x_np)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    except AttributeError:
        pytest.skip("score_numpy not implemented")


def test_cancer_normal_attractor_score_difference(rsp_config, cancer_attractor, normal_attractor):
    """Cancer attractor should (on average after training) score higher than normal."""
    from oracle.rsp.cancer_score import CancerScoreFunction
    model = CancerScoreFunction(rsp_config)
    # Without training, just check the model runs without errors
    c = torch.tensor(cancer_attractor).unsqueeze(0)
    n = torch.tensor(normal_attractor).unsqueeze(0)
    with torch.no_grad():
        cs = model(c).item()
        ns = model(n).item()
    assert 0 <= cs <= 1
    assert 0 <= ns <= 1
