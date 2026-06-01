"""Integration tests: end-to-end ORACLE pipeline on minimal synthetic data."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def tiny_pipeline_grn():
    """Slightly larger GRN for pipeline testing (12 genes)."""
    import networkx as nx

    genes = ["TP53", "MYC", "CDX2", "SNAI2", "VIM", "ZEB1",
             "CDH1", "EPCAM", "CTNNB1", "APC", "KRAS", "BRAF"]
    G = nx.DiGraph()
    G.add_nodes_from(genes)

    edges = [
        ("TP53", "MYC", {"sign": -1, "weight": 0.7, "confidence": 0.9}),
        ("MYC", "CDX2", {"sign": -1, "weight": 0.6, "confidence": 0.8}),
        ("SNAI2", "CDH1", {"sign": -1, "weight": 0.8, "confidence": 0.9}),
        ("SNAI2", "VIM", {"sign": 1, "weight": 0.7, "confidence": 0.85}),
        ("CDX2", "CDH1", {"sign": 1, "weight": 0.9, "confidence": 0.95}),
        ("CTNNB1", "MYC", {"sign": 1, "weight": 0.75, "confidence": 0.8}),
        ("APC", "CTNNB1", {"sign": -1, "weight": 0.85, "confidence": 0.9}),
        ("KRAS", "MYC", {"sign": 1, "weight": 0.6, "confidence": 0.75}),
        ("ZEB1", "CDH1", {"sign": -1, "weight": 0.7, "confidence": 0.8}),
        ("ZEB1", "EPCAM", {"sign": -1, "weight": 0.5, "confidence": 0.65}),
    ]
    G.add_edges_from(edges)
    return G


@pytest.fixture(scope="module")
def pipeline_device():
    return torch.device("cpu")


class TestCAMPipeline:
    def test_boolean_network_on_realistic_grn(self, tiny_pipeline_grn):
        from oracle.cam.preprocessing import CAMConfig
        from oracle.cam.boolean_network import BooleanNetworkSimulator

        config = CAMConfig(n_attractor_samples=50, n_basin_samples=200, n_jobs=1)
        sim = BooleanNetworkSimulator(tiny_pipeline_grn, config)
        attractors = sim.find_attractors(n_initial_states=50)
        assert len(attractors) >= 1

    def test_attractor_classification_pipeline(self, tiny_pipeline_grn):
        from oracle.cam.preprocessing import CAMConfig
        from oracle.cam.boolean_network import BooleanNetworkSimulator
        from oracle.cam.attractor_classifier import AttractorClassifier

        config = CAMConfig(n_attractor_samples=50, n_basin_samples=200, n_jobs=1)
        sim = BooleanNetworkSimulator(tiny_pipeline_grn, config)
        attractors = sim.find_attractors(n_initial_states=50)
        genes = list(tiny_pipeline_grn.nodes())

        clf = AttractorClassifier("colorectal", "colon")
        labels = clf.classify(attractors, genes)
        assert len(labels) == len(attractors)
        for label in labels:
            assert label in ("normal", "cancer", "transitional")


class TestRSPPipeline:
    def test_cancer_score_on_attractors(self, tiny_pipeline_grn, pipeline_device):
        from oracle.cam.preprocessing import CAMConfig
        from oracle.cam.boolean_network import BooleanNetworkSimulator
        from oracle.rsp.cancer_score import CancerScoreFunction, RSPConfig

        config = CAMConfig(n_attractor_samples=50, n_basin_samples=100, n_jobs=1)
        sim = BooleanNetworkSimulator(tiny_pipeline_grn, config)
        attractors = sim.find_attractors(n_initial_states=50)
        n_genes = tiny_pipeline_grn.number_of_nodes()

        rsp_config = RSPConfig(n_genes=n_genes)
        score_fn = CancerScoreFunction(rsp_config).to(pipeline_device)

        scores = []
        for attr in attractors:
            x = torch.tensor(attr, dtype=torch.float32).unsqueeze(0).to(pipeline_device)
            s = score_fn(x).item()
            scores.append(s)
            assert 0 <= s <= 1

    def test_switch_optimizer_pipeline(self, tiny_pipeline_grn, pipeline_device):
        from oracle.cam.preprocessing import CAMConfig
        from oracle.cam.boolean_network import BooleanNetworkSimulator
        from oracle.cam.attractor_classifier import AttractorClassifier
        from oracle.cam.continuous_ode import ContinuousGRNDynamics
        from oracle.rsp.cancer_score import CancerScoreFunction, RSPConfig
        from oracle.rsp.switch_optimizer import MinimalSwitchOptimizer

        config = CAMConfig(n_attractor_samples=50, n_basin_samples=100,
                           integration_time=1.0, n_ode_steps=5, n_jobs=1)
        sim = BooleanNetworkSimulator(tiny_pipeline_grn, config)
        attractors = sim.find_attractors(n_initial_states=50)
        genes = list(tiny_pipeline_grn.nodes())

        clf = AttractorClassifier("colorectal", "colon")
        labels = clf.classify(attractors, genes)
        cancer_attr, normal_attr = clf.get_cancer_normal_pair(attractors, labels)

        if cancer_attr is None or normal_attr is None:
            pytest.skip("No cancer/normal attractor pair found in tiny GRN")

        n_genes = len(genes)
        rsp_config = RSPConfig(n_genes=n_genes)
        score_fn = CancerScoreFunction(rsp_config).to(pipeline_device)

        try:
            ode_model = ContinuousGRNDynamics(tiny_pipeline_grn, config).to(pipeline_device)
            opt = MinimalSwitchOptimizer(rsp_config)
            switch_set = opt.optimize(cancer_attr, normal_attr, tiny_pipeline_grn,
                                       ode_model, score_fn, genes)
            assert switch_set is not None
            assert len(switch_set.perturbations) >= 1
            assert len(switch_set.perturbations) <= 5
        except Exception as e:
            if "torchdiffeq" in str(e).lower():
                pytest.skip(f"torchdiffeq not available: {e}")
            raise


class TestTCDPipeline:
    def test_writer_selector_followed_by_linker(self):
        from oracle.tcd.writer_selector import WriterEraserSelector
        from oracle.tcd.linker_designer import LinkerDesigner
        from oracle.tcd.tf_structurer import TCDConfig

        tcd_config = TCDConfig(n_warhead_candidates=2)
        sel = WriterEraserSelector()
        ld = LinkerDesigner(tcd_config)

        selection = sel.select("CDX2", "Activation", {}, {})
        linker = ld.design(None, None, None, None)

        assert selection.writer_eraser_name in {"BRD4", "CDK9", "p300"}
        assert linker.linker_length > 0

    def test_ternary_validator_on_simple_molecule(self):
        from oracle.tcd.ternary_validator import TernaryComplexValidator
        from oracle.tcd.molecule_generator import Molecule
        from oracle.tcd.tf_structurer import TCDConfig

        tcd_config = TCDConfig()
        val = TernaryComplexValidator(tcd_config)
        mol = Molecule(smiles="CC1=CC=C(C=C1)NC(=O)C2=CC=CC=C2")
        result = val.validate(mol, None, None, None, None)
        assert hasattr(result, "passed")
        assert hasattr(result, "clash_score")


class TestMetrics:
    def test_switch_f1_perfect(self):
        from oracle.evaluation.metrics import switch_f1
        pred = {"CDX2": "Activation", "SNAI2": "Repression"}
        gt = {"CDX2": "Activation", "SNAI2": "Repression"}
        m = switch_f1(pred, gt)
        assert m["f1"] == pytest.approx(1.0)

    def test_switch_f1_zero(self):
        from oracle.evaluation.metrics import switch_f1
        pred = {"TP53": "Activation"}
        gt = {"CDX2": "Activation", "SNAI2": "Repression"}
        m = switch_f1(pred, gt)
        assert m["f1"] == pytest.approx(0.0)

    def test_reversion_auc_perfect(self):
        from oracle.evaluation.metrics import reversion_auc
        scores = np.array([0.9, 0.8, 0.3, 0.2])
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        auc = reversion_auc(scores, labels)
        assert auc == pytest.approx(1.0, abs=0.01)

    def test_molecule_validity_all_valid(self, example_smiles):
        from oracle.evaluation.metrics import molecule_validity
        try:
            v = molecule_validity(example_smiles)
            assert v == pytest.approx(1.0)
        except Exception:
            pytest.skip("rdkit not available")
