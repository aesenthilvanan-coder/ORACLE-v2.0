"""
PyTorch Dataset classes for the three ORACLE training pipelines.

CancerScoreDataset  – trains CancerScoreFunction (CAM module)
GRNDataset          – trains GRNTransformer (RSP module)
TCIPDataset         – trains TCIPDiffusionModel (TCD module)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import networkx as nx  # type: ignore
except ImportError:
    nx = None  # type: ignore


# ---------------------------------------------------------------------------
# CancerScoreDataset
# ---------------------------------------------------------------------------

class CancerScoreDataset(Dataset):
    """Dataset for training the CancerScoreFunction.

    Combines cancer attractor states (label = 1) and normal attractor states
    (label = 0) into a single dataset with optional pseudotime ordering for
    the monotonicity loss.

    Parameters
    ----------
    cancer_states:
        Array of shape ``(N_cancer, n_genes)`` containing cancer cell states.
    normal_states:
        Array of shape ``(N_normal, n_genes)`` containing normal cell states.
    pseudotime_pairs:
        Optional array of shape ``(P, 2)`` where each row ``[i, j]`` means
        sample *i* should have a lower cancer score than sample *j* (early →
        late along a cancer trajectory).
    """

    def __init__(
        self,
        cancer_states: np.ndarray,
        normal_states: np.ndarray,
        pseudotime_pairs: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()

        cancer_states = np.asarray(cancer_states, dtype=np.float32)
        normal_states = np.asarray(normal_states, dtype=np.float32)

        if cancer_states.ndim != 2 or normal_states.ndim != 2:
            raise ValueError(
                "cancer_states and normal_states must be 2-D arrays "
                "(n_samples, n_genes)."
            )
        if cancer_states.shape[1] != normal_states.shape[1]:
            raise ValueError(
                "cancer_states and normal_states must have the same number of genes."
            )

        # Concatenate: cancer first, then normal
        self._states = np.vstack([cancer_states, normal_states])
        self._labels = np.concatenate(
            [
                np.ones(len(cancer_states), dtype=np.float32),
                np.zeros(len(normal_states), dtype=np.float32),
            ]
        )

        self._n_cancer = len(cancer_states)
        self._n_normal = len(normal_states)

        self.pseudotime_pairs: Optional[np.ndarray] = (
            np.asarray(pseudotime_pairs, dtype=np.int64)
            if pseudotime_pairs is not None
            else None
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._states)

    def __getitem__(self, idx: int) -> dict:
        return {
            "state": torch.tensor(self._states[idx], dtype=torch.float32),
            "label": torch.tensor(self._labels[idx], dtype=torch.float32),
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_genes(self) -> int:
        """Dimensionality of each cell state."""
        return self._states.shape[1]

    @property
    def n_cancer(self) -> int:
        """Number of cancer samples."""
        return self._n_cancer

    @property
    def n_normal(self) -> int:
        """Number of normal samples."""
        return self._n_normal

    # ------------------------------------------------------------------
    # Factory: build from CAMOutput
    # ------------------------------------------------------------------

    @classmethod
    def from_cam_output(cls, cam_output) -> "CancerScoreDataset":
        """Construct a CancerScoreDataset from a CAMOutput object.

        Expects ``cam_output`` to have:
        - ``cancer_states``: np.ndarray
        - ``normal_states``: np.ndarray
        - ``pseudotime_pairs`` (optional): np.ndarray or None
        """
        pseudotime_pairs = getattr(cam_output, "pseudotime_pairs", None)
        return cls(
            cancer_states=np.asarray(cam_output.cancer_states, dtype=np.float32),
            normal_states=np.asarray(cam_output.normal_states, dtype=np.float32),
            pseudotime_pairs=pseudotime_pairs,
        )


# ---------------------------------------------------------------------------
# GRNDataset
# ---------------------------------------------------------------------------

class GRNDataset(Dataset):
    """Dataset for training the GRNTransformer.

    Each sample corresponds to one gene-regulatory network with its associated
    expression matrix and binary edge labels.

    Parameters
    ----------
    grn_list:
        List of ``networkx.DiGraph`` objects.
    expr_matrices:
        Corresponding list of expression matrices, each of shape
        ``(n_cells, n_genes)``.
    edge_labels:
        List of binary label arrays, each of shape ``(n_candidate_edges,)``
        indicating true regulatory edges.
    """

    def __init__(
        self,
        grn_list: list,
        expr_matrices: List[np.ndarray],
        edge_labels: List[np.ndarray],
    ) -> None:
        super().__init__()

        if not (len(grn_list) == len(expr_matrices) == len(edge_labels)):
            raise ValueError(
                "grn_list, expr_matrices, and edge_labels must have the same length."
            )

        self._grns = grn_list
        self._expr_matrices = [
            np.asarray(m, dtype=np.float32) for m in expr_matrices
        ]
        self._edge_labels = [
            np.asarray(el, dtype=np.float32) for el in edge_labels
        ]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._grns)

    def __getitem__(self, idx: int) -> dict:
        grn = self._grns[idx]
        expr = self._expr_matrices[idx]
        labels = self._edge_labels[idx]

        # Build candidate edge index from the DiGraph
        if nx is not None and isinstance(grn, nx.DiGraph):
            edges = list(grn.edges())
            if edges:
                candidate_edges = np.array(edges, dtype=np.int64)
            else:
                candidate_edges = np.empty((0, 2), dtype=np.int64)
        else:
            # Fallback: treat grn as a plain edge array
            candidate_edges = np.asarray(grn, dtype=np.int64)

        return {
            "expr_matrix": torch.tensor(expr, dtype=torch.float32),
            "candidate_edges": torch.tensor(candidate_edges, dtype=torch.long),
            "edge_labels": torch.tensor(labels, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# TCIPDataset
# ---------------------------------------------------------------------------

class TCIPDataset(Dataset):
    """Dataset for training the TCIPDiffusionModel.

    Each sample is a dictionary describing one TCIP (ternary complex-inducing
    pharmacophore) molecule with:

    - ``coords``              – (N_atoms, 3) float32 3-D coordinates
    - ``atom_types``          – (N_atoms,) int64 atom-type indices
    - ``pocket_graph``        – adjacency / feature tensor for the target pocket
    - ``recruiter_graph``     – adjacency / feature tensor for the E3 ligase
    - ``geometry_constraint`` – scalar or vector geometric constraint value(s)

    Parameters
    ----------
    molecule_list:
        List of molecule dicts (see field descriptions above).
    """

    _REQUIRED_KEYS = {"coords", "atom_types", "pocket_graph", "recruiter_graph"}

    def __init__(self, molecule_list: List[dict]) -> None:
        super().__init__()
        for i, mol in enumerate(molecule_list):
            missing = self._REQUIRED_KEYS - set(mol.keys())
            if missing:
                raise ValueError(
                    f"Molecule at index {i} is missing required keys: {missing}"
                )

        self._molecules = molecule_list

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._molecules)

    def __getitem__(self, idx: int) -> dict:
        mol = self._molecules[idx]

        coords = torch.tensor(
            np.asarray(mol["coords"], dtype=np.float32), dtype=torch.float32
        )
        atom_types = torch.tensor(
            np.asarray(mol["atom_types"], dtype=np.int64), dtype=torch.long
        )

        # pocket_graph / recruiter_graph may already be tensors or arrays
        pocket_graph = _to_tensor(mol["pocket_graph"])
        recruiter_graph = _to_tensor(mol["recruiter_graph"])

        geometry_constraint = _to_tensor(
            mol.get("geometry_constraint", np.float32(0.0))
        )

        return {
            "coords": coords,
            "atom_types": atom_types,
            "pocket_graph": pocket_graph,
            "recruiter_graph": recruiter_graph,
            "geometry_constraint": geometry_constraint,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor(value) -> torch.Tensor:
    """Convert value to a torch.Tensor if it is not already one."""
    if isinstance(value, torch.Tensor):
        return value
    arr = np.asarray(value)
    if arr.dtype.kind in ("f",):
        return torch.tensor(arr, dtype=torch.float32)
    elif arr.dtype.kind in ("i", "u"):
        return torch.tensor(arr, dtype=torch.long)
    else:
        return torch.tensor(arr.astype(np.float32), dtype=torch.float32)
