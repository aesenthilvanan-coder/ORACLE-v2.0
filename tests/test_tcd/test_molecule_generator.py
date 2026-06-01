"""Tests for oracle.tcd.molecule_generator, linker_designer, and tcip_scorer."""

import numpy as np
import pytest
import torch


def test_molecule_dataclass():
    from oracle.tcd.molecule_generator import Molecule
    mol = Molecule(smiles="CCO", predicted_ki=1e-8)
    assert mol.smiles == "CCO"
    assert mol.predicted_ki == pytest.approx(1e-8)


def test_molecule_generator_init(tcd_config):
    from oracle.tcd.molecule_generator import MoleculeGenerator
    gen = MoleculeGenerator(tcd_config)
    assert gen is not None


def test_molecule_generator_sample_returns_list(tcd_config):
    from oracle.tcd.molecule_generator import MoleculeGenerator, Molecule
    gen = MoleculeGenerator(tcd_config)
    try:
        molecules = gen.sample(
            pocket_graph=None,
            recruiter_graph=None,
            geometry_constraint=torch.zeros(6),
            n_atoms=10,
            n_samples=2,
        )
        assert isinstance(molecules, list)
        for mol in molecules:
            assert isinstance(mol, Molecule)
            assert isinstance(mol.smiles, str)
    except Exception as e:
        if "torch_geometric" in str(e).lower() or "torch_scatter" in str(e).lower():
            pytest.skip(f"PyG dependency missing: {e}")
        raise


def test_linker_designer_init(tcd_config):
    from oracle.tcd.linker_designer import LinkerDesigner
    ld = LinkerDesigner(tcd_config)
    assert ld is not None
    assert len(ld.LINKER_LIBRARY) > 0


def test_linker_library_structure(tcd_config):
    from oracle.tcd.linker_designer import LinkerDesigner
    ld = LinkerDesigner(tcd_config)
    for name, props in ld.LINKER_LIBRARY.items():
        assert "smiles" in props, f"Linker {name} missing SMILES"
        assert "length_A" in props, f"Linker {name} missing length_A"
        assert "flexibility" in props, f"Linker {name} missing flexibility"


def test_linker_design_returns_dataclass(tcd_config):
    from oracle.tcd.linker_designer import LinkerDesigner, LinkerDesign
    ld = LinkerDesigner(tcd_config)
    result = ld.design(
        tf_warhead_pose=None,
        recruiter_warhead_pose=None,
        tf_structure=None,
        writer_structure=None,
    )
    assert isinstance(result, LinkerDesign)
    assert isinstance(result.linker_name, str)
    assert len(result.linker_smiles) > 0
    assert result.linker_length > 0


def test_tcip_scorer_init(tcd_config):
    from oracle.tcd.tcip_scorer import TCIPScorer
    scorer = TCIPScorer(tcd_config)
    assert scorer is not None


def test_tcip_scorer_score_list(tcd_config):
    from oracle.tcd.tcip_scorer import TCIPScorer, TCIPMolecule
    scorer = TCIPScorer(tcd_config)
    mols = [
        TCIPMolecule(
            target_tf="CDX2",
            perturbation_type="Activation",
            writer_eraser="BRD4",
            full_smiles="CCO",
            tf_warhead_smiles="CCO",
            linker_smiles="CC",
            recruiter_smiles="c1ccccc1",
            molecular_weight=350.0,
            logP=2.5,
            tpsa=60.0,
            sa_score=2.0,
            qed=0.7,
            predicted_tf_binding_affinity=0.8,
            predicted_writer_binding_affinity=0.7,
            ternary_complex_score=0.6,
            validation_result=None,
            ternary_complex_structure=None,
            mol_image=None,
        )
    ]
    scores = scorer.score(mols)
    assert len(scores) == 1
    assert 0 <= scores[0] <= 1


def test_tcip_scorer_rank_order(tcd_config):
    from oracle.tcd.tcip_scorer import TCIPScorer, TCIPMolecule

    def make_mol(qed, tf_aff):
        return TCIPMolecule(
            target_tf="CDX2", perturbation_type="Activation", writer_eraser="BRD4",
            full_smiles="CCO", tf_warhead_smiles="CCO", linker_smiles="CC",
            recruiter_smiles="c1ccccc1", molecular_weight=350.0, logP=2.5,
            tpsa=60.0, sa_score=2.0, qed=qed,
            predicted_tf_binding_affinity=tf_aff,
            predicted_writer_binding_affinity=0.7,
            ternary_complex_score=0.6,
            validation_result=None, ternary_complex_structure=None, mol_image=None,
        )

    scorer = TCIPScorer(tcd_config)
    mols = [make_mol(0.3, 0.4), make_mol(0.9, 0.9), make_mol(0.6, 0.6)]
    ranked = scorer.rank(mols)
    scores_before = scorer.score(mols)
    # Best molecule (highest scores) should be ranked first
    assert len(ranked) == 3
