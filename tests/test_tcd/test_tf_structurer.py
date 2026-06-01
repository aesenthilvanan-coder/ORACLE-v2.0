"""Tests for oracle.tcd.tf_structurer and writer_selector."""

import pytest


def test_tcd_config_defaults(tcd_config):
    assert tcd_config.n_diffusion_steps == 1000
    assert tcd_config.distance_cutoff == 10.0
    assert tcd_config.n_warhead_atoms == 25


def test_tf_structurer_init(tcd_config):
    from oracle.tcd.tf_structurer import TFStructurer
    structurer = TFStructurer(tcd_config)
    assert structurer is not None


def test_tf_structure_result_dataclass():
    from oracle.tcd.tf_structurer import TFStructureResult
    result = TFStructureResult(
        tf_name="CDX2",
        structure=None,
        domains=[{"name": "homeodomain", "start": 186, "end": 245}],
        binding_site=None,
        ensemble=[],
        perturbation_type="Activation",
    )
    assert result.tf_name == "CDX2"
    assert len(result.domains) == 1


def test_structurer_prepare_returns_result(tcd_config):
    from oracle.tcd.tf_structurer import TFStructurer, TFStructureResult
    structurer = TFStructurer(tcd_config)
    try:
        result = structurer.prepare("CDX2", "Activation")
        assert isinstance(result, TFStructureResult)
        assert result.tf_name == "CDX2"
        assert result.perturbation_type == "Activation"
    except Exception as e:
        # OpenMM / fpocket missing is acceptable in test env
        if any(x in str(e).lower() for x in ["openmm", "fpocket", "requests", "urllib"]):
            pytest.skip(f"Structural tools not available: {e}")
        raise


def test_writer_eraser_selector_init():
    from oracle.tcd.writer_selector import WriterEraserSelector
    selector = WriterEraserSelector()
    assert selector is not None
    assert hasattr(selector, "WRITERS")
    assert hasattr(selector, "ERASERS")


def test_writers_dict_keys():
    from oracle.tcd.writer_selector import WriterEraserSelector
    sel = WriterEraserSelector()
    expected_writers = {"BRD4", "CDK9", "p300"}
    expected_erasers = {"HDAC1", "EZH2"}
    assert set(sel.WRITERS.keys()) == expected_writers
    assert set(sel.ERASERS.keys()) == expected_erasers


def test_writer_selection_activation():
    from oracle.tcd.writer_selector import WriterEraserSelector, WriterEraserSelection
    sel = WriterEraserSelector()
    result = sel.select("CDX2", "Activation", {}, {})
    assert isinstance(result, WriterEraserSelection)
    assert result.perturbation_type == "Activation"
    assert result.writer_eraser_name in {"BRD4", "CDK9", "p300"}


def test_writer_selection_repression():
    from oracle.tcd.writer_selector import WriterEraserSelector, WriterEraserSelection
    sel = WriterEraserSelector()
    result = sel.select("SNAI2", "Repression", {}, {})
    assert result.perturbation_type == "Repression"
    assert result.writer_eraser_name in {"HDAC1", "EZH2"}


def test_recruiter_smiles_not_empty():
    from oracle.tcd.writer_selector import WriterEraserSelector
    sel = WriterEraserSelector()
    for name, props in {**sel.WRITERS, **sel.ERASERS}.items():
        smiles = props.get("smiles", "")
        assert len(smiles) > 5, f"{name} SMILES too short or missing"
