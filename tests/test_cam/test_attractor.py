"""Tests for oracle.cam.attractor_classifier and attractor_finder."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# AttractorClassifier
# ---------------------------------------------------------------------------

def test_classifier_init():
    from oracle.cam.attractor_classifier import AttractorClassifier
    clf = AttractorClassifier(cancer_type="colorectal", tissue="colon")
    assert clf.cancer_type == "colorectal"


def test_classifier_cancer_markers():
    from oracle.cam.attractor_classifier import AttractorClassifier, CANCER_MARKERS
    assert "colorectal" in CANCER_MARKERS
    assert "leukemia_aml" in CANCER_MARKERS
    assert len(CANCER_MARKERS["colorectal"]) > 0


def test_cancer_score_range(tiny_grn_genes):
    from oracle.cam.attractor_classifier import AttractorClassifier
    clf = AttractorClassifier(cancer_type="colorectal", tissue="colon")
    rng = np.random.default_rng(42)
    attractor = rng.choice([0, 1], size=len(tiny_grn_genes)).astype(float)
    score = clf._compute_cancer_score(attractor, tiny_grn_genes)
    assert 0.0 <= score <= 1.0, f"Cancer score {score} out of [0,1]"


def test_classify_returns_valid_labels(tiny_grn_genes):
    from oracle.cam.attractor_classifier import AttractorClassifier
    clf = AttractorClassifier(cancer_type="colorectal", tissue="colon")
    rng = np.random.default_rng(1)
    n_attractors = 3
    attractors = [rng.choice([0, 1], size=len(tiny_grn_genes)).astype(float)
                  for _ in range(n_attractors)]
    labels = clf.classify(attractors, tiny_grn_genes)
    assert len(labels) == n_attractors
    for label in labels:
        assert label in ("normal", "cancer", "transitional")


def test_get_cancer_normal_pair(tiny_grn_genes):
    from oracle.cam.attractor_classifier import AttractorClassifier
    clf = AttractorClassifier(cancer_type="colorectal", tissue="colon")
    rng = np.random.default_rng(2)
    attractors = [rng.choice([0, 1], size=len(tiny_grn_genes)).astype(float)
                  for _ in range(4)]
    labels = clf.classify(attractors, tiny_grn_genes)
    cancer_attr, normal_attr = clf.get_cancer_normal_pair(attractors, labels)
    # Should return None if no cancer/normal found, otherwise arrays
    if "cancer" in labels and "normal" in labels:
        assert cancer_attr is not None
        assert normal_attr is not None
        assert len(cancer_attr) == len(tiny_grn_genes)


# ---------------------------------------------------------------------------
# ContinuousGRNDynamics
# ---------------------------------------------------------------------------

def test_ode_model_init(tiny_grn, cam_config):
    from oracle.cam.continuous_ode import ContinuousGRNDynamics
    model = ContinuousGRNDynamics(tiny_grn, cam_config)
    assert model is not None
    n = tiny_grn.number_of_nodes()
    # Check weight matrix shape
    assert hasattr(model, "W") or hasattr(model, "weight_matrix")


def test_ode_forward_shape(tiny_grn, cam_config):
    import torch
    from oracle.cam.continuous_ode import ContinuousGRNDynamics
    model = ContinuousGRNDynamics(tiny_grn, cam_config)
    n = tiny_grn.number_of_nodes()
    x = torch.rand(2, n)
    t = torch.tensor(0.0)
    try:
        dx = model(t, x)
        assert dx.shape == x.shape
    except Exception as e:
        if "torchdiffeq" in str(e).lower():
            pytest.skip("torchdiffeq not available")
        raise


def test_ode_find_fixed_points_returns_list(tiny_grn, cam_config):
    from oracle.cam.continuous_ode import ContinuousGRNDynamics
    model = ContinuousGRNDynamics(tiny_grn, cam_config)
    try:
        fps = model.find_fixed_points(n_init=5)
        assert isinstance(fps, list)
    except Exception as e:
        if "torchdiffeq" in str(e).lower():
            pytest.skip("torchdiffeq not available")
        raise
