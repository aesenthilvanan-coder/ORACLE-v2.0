"""Tests for oracle.cam.grn_inference."""

import numpy as np
import pytest


def test_grn_inference_engine_init(cam_config):
    from oracle.cam.grn_inference import GRNInferenceEngine
    engine = GRNInferenceEngine(cam_config)
    assert engine is not None
    assert len(engine._load_tf_list()) > 0


def test_tf_list_contains_known_tfs(cam_config):
    from oracle.cam.grn_inference import GRNInferenceEngine
    engine = GRNInferenceEngine(cam_config)
    tf_list = engine._load_tf_list()
    for known_tf in ["TP53", "MYC", "CDX2", "SNAI2"]:
        assert known_tf in tf_list, f"{known_tf} missing from TF list"


def test_trrust_sign_dict(cam_config):
    from oracle.cam.grn_inference import GRNInferenceEngine
    engine = GRNInferenceEngine(cam_config)
    trrust = engine._load_trrust()
    assert isinstance(trrust, dict)
    # Signs should be +1 or -1
    for (tf, target), sign in trrust.items():
        assert sign in (1, -1), f"Invalid sign {sign} for ({tf}, {target})"


def test_prior_network_structure(cam_config):
    from oracle.cam.grn_inference import GRNInferenceEngine
    engine = GRNInferenceEngine(cam_config)
    prior = engine._build_prior_network()
    assert prior.number_of_nodes() > 0
    assert prior.number_of_edges() >= 0


def test_grn_infer_minimal(minimal_adata, cam_config):
    from oracle.cam.grn_inference import GRNInferenceEngine
    engine = GRNInferenceEngine(cam_config)
    try:
        grn = engine.infer(minimal_adata)
        assert grn is not None
        assert grn.number_of_nodes() > 0
    except Exception as e:
        if "arboreto" in str(e).lower() or "grnboost" in str(e).lower():
            pytest.skip(f"GRNBoost2 not available: {e}")
        raise
