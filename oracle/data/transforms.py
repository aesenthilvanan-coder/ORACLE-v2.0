"""
Data-transform utilities for the ORACLE pipeline.

GRNTransform     – converts networkx.DiGraph → PyTorch Geometric Data
MoleculeTransform – converts SMILES → PyTorch Geometric molecular graph
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import networkx as nx  # type: ignore
except ImportError:
    nx = None  # type: ignore

try:
    import torch
    from torch_geometric.data import Data  # type: ignore
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    Data = None  # type: ignore


# ---------------------------------------------------------------------------
# Atom-type vocabulary
# ---------------------------------------------------------------------------

ATOM_TYPES = [
    "C", "N", "O", "S", "F", "Cl", "Br", "I",
    "P", "Si", "B", "Se", "Te", "other",
]
_ATOM_TO_IDX = {sym: i for i, sym in enumerate(ATOM_TYPES)}

# Bond-type vocabulary
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
_BOND_TO_IDX = {b: i for i, b in enumerate(BOND_TYPES)}


# ---------------------------------------------------------------------------
# GRNTransform
# ---------------------------------------------------------------------------

class GRNTransform:
    """Transform a networkx.DiGraph into a PyTorch Geometric ``Data`` object.

    Node features
    -------------
    If *node_features* is provided it must have shape ``(n_nodes, F)``; those
    features are used verbatim.  Otherwise a one-hot degree feature is built.

    Edge attributes
    ---------------
    Two attributes per edge are stored:

    - **sign**   – +1.0 for activation, -1.0 for repression (from edge data
                   key ``"sign"`` / ``"interaction"``; defaults to +1.0).
    - **weight** – edge weight (from edge data key ``"weight"``; defaults to
                   1.0).
    """

    def __call__(
        self,
        grn,
        node_features: Optional[np.ndarray] = None,
    ):
        """Convert *grn* to a PyTorch Geometric ``Data`` object.

        Parameters
        ----------
        grn:
            ``networkx.DiGraph`` representing the gene-regulatory network.
        node_features:
            Optional pre-computed node features of shape ``(n_nodes, F)``.

        Returns
        -------
        torch_geometric.data.Data
        """
        if not _HAS_PYG:
            raise ImportError(
                "torch_geometric is required for GRNTransform. "
                "Install with: pip install torch-geometric"
            )
        if nx is None:
            raise ImportError(
                "networkx is required for GRNTransform. "
                "Install with: pip install networkx"
            )
        if not isinstance(grn, nx.DiGraph):
            raise TypeError(f"Expected nx.DiGraph, got {type(grn).__name__}")

        import torch

        # Map node labels to contiguous integer indices
        node_list = list(grn.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        n_nodes = len(node_list)

        # --- Node features ---
        if node_features is not None:
            node_features = np.asarray(node_features, dtype=np.float32)
            if node_features.shape[0] != n_nodes:
                raise ValueError(
                    f"node_features has {node_features.shape[0]} rows "
                    f"but GRN has {n_nodes} nodes."
                )
            x = torch.tensor(node_features, dtype=torch.float32)
        else:
            # One-hot in-degree as simple fallback feature
            in_deg = np.array(
                [grn.in_degree(n) for n in node_list], dtype=np.float32
            )
            out_deg = np.array(
                [grn.out_degree(n) for n in node_list], dtype=np.float32
            )
            x = torch.tensor(
                np.stack([in_deg, out_deg], axis=1), dtype=torch.float32
            )

        # --- Edges ---
        if grn.number_of_edges() == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.float32)
        else:
            src_list, dst_list, sign_list, weight_list = [], [], [], []
            for u, v, data in grn.edges(data=True):
                src_list.append(node_to_idx[u])
                dst_list.append(node_to_idx[v])

                # Determine edge sign
                sign = data.get("sign", None)
                if sign is None:
                    interaction = str(data.get("interaction", "+")).strip()
                    sign = -1.0 if interaction.startswith("-") else 1.0
                sign_list.append(float(sign))
                weight_list.append(float(data.get("weight", 1.0)))

            edge_index = torch.tensor(
                [src_list, dst_list], dtype=torch.long
            )
            edge_attr = torch.tensor(
                np.stack([sign_list, weight_list], axis=1),
                dtype=torch.float32,
            )

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = n_nodes
        # Preserve node label order for downstream use
        data.node_names = node_list
        return data


# ---------------------------------------------------------------------------
# MoleculeTransform
# ---------------------------------------------------------------------------

class MoleculeTransform:
    """Transform a SMILES string into a PyTorch Geometric molecular graph.

    Uses RDKit to parse the SMILES and extract:
    - **Node features**: atom type (one-hot), formal charge, is-in-ring
    - **Edge index**: bond adjacency (both directions)
    - **Edge attributes**: bond type (one-hot), is-conjugated, stereo flag

    Returns ``None`` for invalid SMILES so the caller can filter gracefully.
    """

    def __call__(self, smiles: str):
        """Convert a SMILES string to a ``Data`` object.

        Parameters
        ----------
        smiles:
            SMILES string of the molecule.

        Returns
        -------
        torch_geometric.data.Data or None
            Returns ``None`` if the SMILES is invalid.
        """
        if not _HAS_PYG:
            raise ImportError(
                "torch_geometric is required for MoleculeTransform. "
                "Install with: pip install torch-geometric"
            )

        try:
            from rdkit import Chem  # type: ignore
            from rdkit.Chem import rdmolops  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "rdkit is required for MoleculeTransform. "
                "Install with: pip install rdkit"
            ) from exc

        import torch

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # ------------------------------------------------------------------
        # Node features
        # ------------------------------------------------------------------
        atom_features = []
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            atom_idx = _ATOM_TO_IDX.get(sym, _ATOM_TO_IDX["other"])

            # One-hot atom type
            one_hot = [0.0] * len(ATOM_TYPES)
            one_hot[atom_idx] = 1.0

            features = one_hot + [
                float(atom.GetFormalCharge()),
                float(atom.IsInRing()),
                float(atom.GetIsAromatic()),
                float(atom.GetTotalNumHs()),
            ]
            atom_features.append(features)

        x = torch.tensor(atom_features, dtype=torch.float32)

        # ------------------------------------------------------------------
        # Edge index and edge attributes
        # ------------------------------------------------------------------
        bonds = mol.GetBonds()
        src_list, dst_list, bond_feats = [], [], []

        for bond in bonds:
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            btype = bond.GetBondTypeAsDouble()
            btype_str = bond.GetBondType().name  # e.g. "SINGLE"
            bt_idx = _BOND_TO_IDX.get(btype_str, 0)
            bt_one_hot = [0.0] * len(BOND_TYPES)
            bt_one_hot[bt_idx] = 1.0

            feat = bt_one_hot + [
                float(bond.GetIsConjugated()),
                float(bond.IsInRing()),
                float(bond.GetStereo().real),
            ]

            # Undirected: add both directions
            src_list += [i, j]
            dst_list += [j, i]
            bond_feats += [feat, feat]

        if src_list:
            edge_index = torch.tensor(
                [src_list, dst_list], dtype=torch.long
            )
            edge_attr = torch.tensor(bond_feats, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, len(BOND_TYPES) + 3), dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.smiles = smiles
        data.num_nodes = mol.GetNumAtoms()
        return data
