"""Tests for oracle.tcd.ternary_validator.TernaryComplexValidator."""

import pytest


def test_ternary_validator_init(tcd_config):
    from oracle.tcd.ternary_validator import TernaryComplexValidator
    val = TernaryComplexValidator(tcd_config)
    assert val is not None


def test_validate_returns_result(tcd_config):
    from oracle.tcd.ternary_validator import TernaryComplexValidator, ValidationResult
    from oracle.tcd.molecule_generator import Molecule

    val = TernaryComplexValidator(tcd_config)
    mol = Molecule(smiles="CC1=CC=C(C=C1)NC(=O)C2=CC=CC=C2")

    result = val.validate(
        tcip_molecule=mol,
        tf_structure=None,
        writer_structure=None,
        tf_warhead_pose=None,
        recruiter_pose=None,
    )
    assert isinstance(result, ValidationResult)
    assert isinstance(result.passed, bool)
    assert hasattr(result, "clash_score")
    assert hasattr(result, "interface_energy")
    assert hasattr(result, "sa_score")


def test_validation_result_fields(tcd_config):
    from oracle.tcd.ternary_validator import TernaryComplexValidator
    from oracle.tcd.molecule_generator import Molecule

    val = TernaryComplexValidator(tcd_config)
    mol = Molecule(smiles="c1ccccc1")
    result = val.validate(mol, None, None, None, None)

    # All required fields should be present
    assert result.clash_score >= 0
    assert isinstance(result.drug_like, bool)


def test_sa_score_range(tcd_config):
    from oracle.tcd.ternary_validator import TernaryComplexValidator
    from oracle.tcd.molecule_generator import Molecule

    val = TernaryComplexValidator(tcd_config)
    mol = Molecule(smiles="CCO")
    result = val.validate(mol, None, None, None, None)
    # SA score should be in [1, 10]
    assert 1 <= result.sa_score <= 10, f"SA score {result.sa_score} out of range"


def test_drug_likeness_check(tcd_config):
    from oracle.tcd.ternary_validator import TernaryComplexValidator
    from oracle.tcd.molecule_generator import Molecule

    val = TernaryComplexValidator(tcd_config)

    # Simple drug-like molecule
    drug_mol = Molecule(smiles="CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
    result = val.validate(drug_mol, None, None, None, None)
    assert isinstance(result.drug_like, bool)
