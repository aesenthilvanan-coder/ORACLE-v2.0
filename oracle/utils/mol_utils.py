"""
Molecular utility functions for the ORACLE pipeline.

All functions use RDKit internally.  Install with::

    pip install rdkit

Functions
---------
smiles_to_mol         – parse SMILES to RDKit Mol
mol_to_smiles         – convert RDKit Mol to canonical SMILES
compute_descriptors   – compute a standard drug-likeness descriptor set
draw_molecule_to_image – render 2-D depiction as numpy array
mol_to_b64            – base64-encoded PNG for HTML embedding
assemble_tcip         – concatenate warhead + linker + recruiter SMILES
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper: lazy RDKit import
# ---------------------------------------------------------------------------

def _rdkit():
    """Import and return (Chem, Descriptors, Draw, QED) from RDKit."""
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import Descriptors, Draw, QED  # type: ignore
        return Chem, Descriptors, Draw, QED
    except ImportError as exc:
        raise ImportError(
            "rdkit is required for molecular utilities. "
            "Install with: pip install rdkit"
        ) from exc


# ---------------------------------------------------------------------------
# smiles_to_mol
# ---------------------------------------------------------------------------

def smiles_to_mol(smiles: str):
    """Parse a SMILES string and return an RDKit Mol object.

    Parameters
    ----------
    smiles:
        SMILES string.

    Returns
    -------
    rdkit.Chem.Mol or None
        Returns ``None`` if the SMILES is invalid or empty.
    """
    Chem, _, _, _ = _rdkit()
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    return mol  # None if invalid


# ---------------------------------------------------------------------------
# mol_to_smiles
# ---------------------------------------------------------------------------

def mol_to_smiles(mol, canonical: bool = True) -> str:
    """Convert an RDKit Mol to a SMILES string.

    Parameters
    ----------
    mol:
        RDKit Mol object.
    canonical:
        If ``True`` (default), return the canonical SMILES.

    Returns
    -------
    str
        SMILES string, or ``""`` if *mol* is ``None``.
    """
    Chem, _, _, _ = _rdkit()
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=canonical) or ""


# ---------------------------------------------------------------------------
# compute_descriptors
# ---------------------------------------------------------------------------

def compute_descriptors(smiles: str) -> dict:
    """Compute a standard set of drug-likeness descriptors for a molecule.

    Parameters
    ----------
    smiles:
        SMILES string of the molecule.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``MW``       – molecular weight (Da)
        - ``LogP``     – Wildman-Crippen LogP
        - ``TPSA``     – topological polar surface area (Å²)
        - ``HBD``      – number of H-bond donors
        - ``HBA``      – number of H-bond acceptors
        - ``RotBonds`` – number of rotatable bonds
        - ``QED``      – quantitative estimate of drug-likeness [0, 1]
        - ``valid``    – ``True`` if SMILES was parsed successfully

    If the SMILES is invalid, all numeric values are ``None`` and
    ``valid`` is ``False``.
    """
    Chem, Descriptors, _, QED = _rdkit()

    mol = smiles_to_mol(smiles)
    if mol is None:
        return {
            "MW": None,
            "LogP": None,
            "TPSA": None,
            "HBD": None,
            "HBA": None,
            "RotBonds": None,
            "QED": None,
            "valid": False,
        }

    try:
        from rdkit.Chem import rdMolDescriptors  # type: ignore

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        qed_val = QED.qed(mol)

        return {
            "MW": round(mw, 3),
            "LogP": round(logp, 3),
            "TPSA": round(tpsa, 3),
            "HBD": hbd,
            "HBA": hba,
            "RotBonds": rot_bonds,
            "QED": round(qed_val, 4),
            "valid": True,
        }
    except Exception as exc:
        logger.warning("Descriptor computation failed for '%s': %s", smiles, exc)
        return {
            "MW": None,
            "LogP": None,
            "TPSA": None,
            "HBD": None,
            "HBA": None,
            "RotBonds": None,
            "QED": None,
            "valid": False,
        }


# ---------------------------------------------------------------------------
# draw_molecule_to_image
# ---------------------------------------------------------------------------

def draw_molecule_to_image(
    smiles: str,
    size: tuple = (300, 200),
) -> Optional[np.ndarray]:
    """Render a 2-D molecular depiction as a numpy RGBA array.

    Parameters
    ----------
    smiles:
        SMILES string.
    size:
        ``(width, height)`` of the output image in pixels.

    Returns
    -------
    numpy.ndarray or None
        Array of shape ``(height, width, 4)`` with dtype ``uint8`` (RGBA),
        or ``None`` if the SMILES is invalid.
    """
    Chem, _, Draw, _ = _rdkit()

    mol = smiles_to_mol(smiles)
    if mol is None:
        logger.warning("draw_molecule_to_image: invalid SMILES '%s'", smiles)
        return None

    try:
        from rdkit.Chem import rdDepictor  # type: ignore
        from rdkit.Chem.Draw import rdMolDraw2D  # type: ignore

        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()

        # Convert SVG → PNG → numpy via cairosvg if available
        try:
            import cairosvg  # type: ignore
            png_bytes = cairosvg.svg2png(
                bytestring=svg.encode(), output_width=size[0], output_height=size[1]
            )
            from PIL import Image  # type: ignore
            img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
            return np.array(img, dtype=np.uint8)
        except ImportError:
            pass

        # Fallback: use RDKit's PIL-based drawing
        pil_img = Draw.MolToImage(mol, size=size)
        return np.array(pil_img.convert("RGBA"), dtype=np.uint8)

    except Exception as exc:
        logger.warning(
            "draw_molecule_to_image failed for '%s': %s", smiles, exc
        )
        # Last resort: use MolToImage directly
        try:
            pil_img = Draw.MolToImage(mol, size=size)
            return np.array(pil_img.convert("RGBA"), dtype=np.uint8)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# mol_to_b64
# ---------------------------------------------------------------------------

def mol_to_b64(smiles: str, size: tuple = (300, 200)) -> str:
    """Render a molecule as a base64-encoded PNG for HTML embedding.

    Parameters
    ----------
    smiles:
        SMILES string.
    size:
        ``(width, height)`` in pixels.

    Returns
    -------
    str
        Base64 PNG string (without the ``data:image/png;base64,`` prefix),
        or an empty string if the SMILES is invalid.
    """
    Chem, _, Draw, _ = _rdkit()

    mol = smiles_to_mol(smiles)
    if mol is None:
        return ""

    try:
        from PIL import Image  # type: ignore

        pil_img = Draw.MolToImage(mol, size=size)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as exc:
        logger.warning("mol_to_b64 failed for '%s': %s", smiles, exc)
        return ""


# ---------------------------------------------------------------------------
# assemble_tcip
# ---------------------------------------------------------------------------

def assemble_tcip(
    warhead_smiles: str,
    linker_smiles: str,
    recruiter_smiles: str,
) -> str:
    """Assemble a TCIP bifunctional molecule by covalently connecting its three parts.

    The linker bridges the warhead and recruiter via explicit bond formation:
    warhead--linker--recruiter becomes one connected molecule.

    Linker SMILES must contain exactly two ``[*]`` attachment point atoms
    (as in LINKER_LIBRARY, e.g. ``"[*]OCCO[*]"``).  If a warhead or recruiter
    SMILES also contains a ``[*]``, that atom is used as the connection point;
    otherwise the most terminal aliphatic carbon (or lowest-degree atom) is
    selected automatically.

    Falls back to dot-separated mixture notation if RDKit is unavailable or
    covalent assembly fails for any reason.

    Parameters
    ----------
    warhead_smiles:
        SMILES of the TF-binding warhead fragment.
    linker_smiles:
        SMILES of the linker fragment (must have two ``[*]`` attachment points).
    recruiter_smiles:
        SMILES of the epigenetic recruiter fragment.

    Returns
    -------
    str
        Single covalently-connected SMILES, or dot-separated fallback.
    """
    def _find_attachment_atom(mol: Any) -> int:
        """Return best attachment atom index.

        An atom is only eligible if it has at least one implicit H
        (free valence slot).  Priority order:
        1. Explicit [*] dummy atom
        2. Terminal aliphatic C (degree 1, free valence)
        3. Any aliphatic C with free valence (lowest degree first)
        4. Any aromatic C with free valence (lowest degree first)
        5. Any atom with free valence
        """
        for a in mol.GetAtoms():
            if a.GetAtomicNum() == 0:
                return a.GetIdx()

        def free(a: Any) -> bool:
            return a.GetNumImplicitHs() > 0

        aliphatic_c = [a for a in mol.GetAtoms()
                       if a.GetAtomicNum() == 6 and not a.GetIsAromatic() and free(a)]
        if aliphatic_c:
            terminal = [a for a in aliphatic_c if a.GetDegree() == 1]
            if terminal:
                return terminal[0].GetIdx()
            return min(aliphatic_c, key=lambda a: a.GetDegree()).GetIdx()

        aromatic_c = [a for a in mol.GetAtoms()
                      if a.GetAtomicNum() == 6 and a.GetIsAromatic() and free(a)]
        if aromatic_c:
            return min(aromatic_c, key=lambda a: a.GetDegree()).GetIdx()

        free_atoms = [a for a in mol.GetAtoms() if free(a)]
        if free_atoms:
            return min(free_atoms, key=lambda a: a.GetDegree()).GetIdx()

        return 0

    def _attach(base_mol: Any, base_dummy_idx: int, frag_mol: Any) -> Any:
        """
        Connect base_mol at base_dummy_idx to frag_mol at its attachment point.
        Dummy ([*]) atoms at both connection sites are removed; a direct bond
        is formed between their respective neighbors.
        """
        from rdkit import Chem as _Chem

        frag_attach_idx = _find_attachment_atom(frag_mol)
        base_is_dummy = base_mol.GetAtomWithIdx(base_dummy_idx).GetAtomicNum() == 0
        frag_is_dummy = frag_mol.GetAtomWithIdx(frag_attach_idx).GetAtomicNum() == 0

        if base_is_dummy:
            nbrs = list(base_mol.GetAtomWithIdx(base_dummy_idx).GetNeighbors())
            if not nbrs:
                raise ValueError("linker [*] atom has no neighbors")
            base_bond_atom = nbrs[0].GetIdx()
        else:
            base_bond_atom = base_dummy_idx

        if frag_is_dummy:
            nbrs = list(frag_mol.GetAtomWithIdx(frag_attach_idx).GetNeighbors())
            if not nbrs:
                raise ValueError("fragment [*] atom has no neighbors")
            frag_bond_atom = nbrs[0].GetIdx()
        else:
            frag_bond_atom = frag_attach_idx

        n_base = base_mol.GetNumAtoms()
        combo = _Chem.RWMol(_Chem.CombineMols(base_mol, frag_mol))
        combo.AddBond(base_bond_atom, frag_bond_atom + n_base, _Chem.BondType.SINGLE)

        to_remove = []
        if base_is_dummy:
            to_remove.append(base_dummy_idx)
        if frag_is_dummy:
            to_remove.append(frag_attach_idx + n_base)
        for idx in sorted(to_remove, reverse=True):
            combo.RemoveAtom(idx)

        _Chem.SanitizeMol(combo)
        return combo.GetMol()

    try:
        from rdkit import Chem

        parts_raw = [warhead_smiles, linker_smiles, recruiter_smiles]
        mols = [Chem.MolFromSmiles(s.strip()) if s else None for s in parts_raw]
        if any(m is None for m in mols):
            raise ValueError("One or more component SMILES failed to parse")

        warhead_mol, linker_mol, recruiter_mol = mols

        linker_dummies = [a.GetIdx() for a in linker_mol.GetAtoms() if a.GetAtomicNum() == 0]
        if len(linker_dummies) < 2:
            raise ValueError(f"Linker has {len(linker_dummies)} [*] atoms; need 2")

        # Connect linker[*][0] → warhead
        intermediate = _attach(linker_mol, linker_dummies[0], warhead_mol)

        # Find surviving [*] in intermediate (originally linker's second dummy)
        remaining = [a.GetIdx() for a in intermediate.GetAtoms() if a.GetAtomicNum() == 0]
        if not remaining:
            raise ValueError("No [*] left in intermediate after first attachment")

        # Connect intermediate[*] → recruiter
        final_mol = _attach(intermediate, remaining[0], recruiter_mol)

        result = Chem.MolToSmiles(final_mol)
        if not result:
            raise ValueError("MolToSmiles returned empty string")

        logger.debug("assemble_tcip: connected SMILES length=%d", len(result))
        return result

    except ImportError:
        pass
    except Exception as exc:
        logger.warning("assemble_tcip covalent assembly failed (%s); using disconnected notation", exc)

    # Fallback: dot-separated (disconnected) notation
    parts = [s.strip() for s in [warhead_smiles, linker_smiles, recruiter_smiles] if s]
    return ".".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# smiles_to_graph  (PyTorch Geometric Data object)
# ---------------------------------------------------------------------------

def smiles_to_graph(smiles: str):
    """Convert a SMILES string to a PyTorch Geometric Data object.

    Returns None if the SMILES is invalid.

    Node features (graph.x): one-hot atom type + atomic number + charge +
        ring membership + aromaticity  →  shape (n_atoms, n_features)
    Edge index (graph.edge_index): shape (2, n_edges), undirected bonds
    Position (graph.pos): None (no 3-D coords unless pre-computed)
    """
    try:
        import torch
        from rdkit import Chem
        from torch_geometric.data import Data

        ATOM_TYPES = ["H", "C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
        atom_to_idx = {a: i for i, a in enumerate(ATOM_TYPES)}
        n_atom_types = len(ATOM_TYPES)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        n_atoms = mol.GetNumAtoms()
        if n_atoms == 0:
            return None

        # Node features
        x_list = []
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            one_hot = [0.0] * n_atom_types
            if sym in atom_to_idx:
                one_hot[atom_to_idx[sym]] = 1.0
            feats = one_hot + [
                atom.GetAtomicNum() / 100.0,
                float(atom.GetFormalCharge()),
                float(atom.IsInRing()),
                float(atom.GetIsAromatic()),
            ]
            x_list.append(feats)
        x = torch.tensor(x_list, dtype=torch.float32)

        # Edge index (undirected: add both directions)
        rows, cols = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            rows += [i, j]
            cols += [j, i]
        if rows:
            edge_index = torch.tensor([rows, cols], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, pos=None)

    except Exception as exc:
        logger.debug(f"smiles_to_graph failed for '{smiles}': {exc}")
        return None
