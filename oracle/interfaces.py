from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
import numpy as np
import torch
import networkx as nx
import anndata as ad
from rdkit import Chem


@dataclass(frozen=True)
class CAMOutput:
    adata: ad.AnnData
    grn: nx.DiGraph
    genes: List[str]
    n_genes: int
    bool_network: Any
    ode_model: Any
    all_attractors: List[np.ndarray]
    attractor_labels: List[str]
    basin_sizes: Dict[tuple, float]
    cancer_attractor: np.ndarray
    normal_attractor: np.ndarray
    cancer_score_func: Callable[[np.ndarray, List[str]], float]
    landscape_embedding: np.ndarray
    pseudotime: np.ndarray
    trajectory_cells: ad.AnnData
    cancer_type: str
    tissue_type: str
    sample_id: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class SwitchSet:
    genes_to_activate: List[str]
    genes_to_repress: List[str]
    predicted_reversion_probability: float
    validated_reversion_fraction: float
    predicted_cancer_score_after: float
    gene_importance_scores: Dict[str, float]
    perturbation_types: Dict[str, str]


@dataclass(frozen=True)
class RSPOutput:
    switch_set: SwitchSet
    genes_to_activate: List[str]
    genes_to_repress: List[str]
    n_perturbations: int
    predicted_cancer_score_before: float
    predicted_cancer_score_after: float
    predicted_reversion_probability: float
    validated_reversion_fraction: float
    perturbation_trajectories: List[np.ndarray]
    cancer_score_trajectory: List[float]
    gene_importance: Dict[str, float]
    perturbation_type: Dict[str, str]
    cam_output: CAMOutput


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    clash_score: float
    interface_energy: float
    writer_positioning_distance: float
    writer_positioning_productive: bool
    sa_score: float
    qed: float
    mw: float
    logP: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    passes_ro5: bool
    passes_veber: bool


@dataclass(frozen=True)
class TCIPMolecule:
    target_tf: str
    perturbation_type: str
    writer_eraser: str
    full_smiles: str
    tf_warhead_smiles: str
    linker_smiles: str
    recruiter_smiles: str
    molecular_weight: float
    logP: float
    tpsa: float
    sa_score: float
    qed: float
    hbd: int
    hba: int
    rotatable_bonds: int
    predicted_tf_binding_affinity_nM: float
    predicted_writer_binding_affinity_nM: float
    ternary_complex_score: float
    validation_result: ValidationResult
    mol_image_b64: str


@dataclass(frozen=True)
class TCDOutput:
    tcip_molecules: List[TCIPMolecule]
    n_molecules: int
    n_validated: int
    cancer_type: str
    patient_id: str
    predicted_reversion_probability: float
    rsp_output: RSPOutput
    cam_output: CAMOutput


@dataclass(frozen=True)
class OracleOutput:
    tcd_output: TCDOutput
    rsp_output: RSPOutput
    cam_output: CAMOutput
    patient_id: str
    cancer_type: str
    runtime_seconds: float
    pipeline_version: str
