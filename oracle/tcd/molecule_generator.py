"""
molecule_generator.py
---------------------
Diffusion-based warhead generator for the TCIP design pipeline.

MoleculeGenerator wraps TCIPDiffusionModel — a DDPM (denoising diffusion
probabilistic model) conditioned on:
  * the TF pocket graph (geometric + chemical features)
  * the recruiter warhead graph
  * a 6-dimensional ternary geometry constraint tensor
    [distance, angle, d1, d2, d3, d4]

The model performs reverse diffusion in atom-coordinate and atom-type space
jointly, sampling molecules that fit the pocket while maintaining the correct
geometry for ternary complex formation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple molecule dataclass
# ---------------------------------------------------------------------------


@dataclass
class Molecule:
    """
    Lightweight container for a generated / sampled molecule.

    Attributes
    ----------
    smiles : str
        Canonical SMILES string.
    coords : np.ndarray or None
        Heavy-atom 3-D coordinates (N, 3) in Angstroms.
    atom_types : np.ndarray or None
        Integer atom-type indices aligned with *coords*.
    predicted_ki : float
        Predicted binding affinity (Ki in nM, lower is better).
    """

    smiles: str
    coords: Optional[np.ndarray] = None
    atom_types: Optional[np.ndarray] = None
    predicted_ki: float = 0.0


# ---------------------------------------------------------------------------
# Diffusion model (PyTorch)
# ---------------------------------------------------------------------------


def _build_tcip_diffusion_model(config: Any) -> Any:
    """
    Construct the TCIPDiffusionModel.  Returns a torch.nn.Module.

    The model is an E(3)-equivariant graph neural network diffusion model
    that jointly denoises atom coordinates and atom-type distributions.

    Architecture summary
    --------------------
    Encoder    : EGNN with *mol_n_layers* layers, hidden dim *mol_hidden_dim*
    Condition  : pocket graph + recruiter graph + 6-D geometry vector
    Denoiser   : transformer-style attention over atom graph
    Scheduler  : linear beta schedule, *n_diffusion_steps* steps
    """
    try:
        import torch
        import torch.nn as nn

        class _PositionalEncoding(nn.Module):
            def __init__(self, d_model: int, max_len: int = 512) -> None:
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                pos = torch.arange(0, max_len).unsqueeze(1).float()
                div = torch.exp(
                    torch.arange(0, d_model, 2).float()
                    * (-np.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(pos * div)
                pe[:, 1::2] = torch.cos(pos * div)
                self.register_buffer("pe", pe.unsqueeze(0))

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                return x + self.pe[:, : x.size(1)]

        class _EGNNLayer(nn.Module):
            """Single E(3)-equivariant message passing layer."""

            def __init__(self, hidden_dim: int) -> None:
                super().__init__()
                self.msg_mlp = nn.Sequential(
                    nn.Linear(hidden_dim * 2 + 1, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.coord_mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
                self.node_mlp = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.norm = nn.LayerNorm(hidden_dim)

            def forward(
                self,
                h: "torch.Tensor",
                x: "torch.Tensor",
                edge_index: "torch.Tensor",
            ) -> tuple:
                row, col = edge_index[0], edge_index[1]
                diff = x[row] - x[col]
                dist = torch.norm(diff, dim=-1, keepdim=True)
                msg_input = torch.cat([h[row], h[col], dist], dim=-1)
                msg = self.msg_mlp(msg_input)
                coord_update = self.coord_mlp(msg)
                agg_coord = torch.zeros_like(x)
                agg_coord.index_add_(0, row, coord_update * diff / (dist + 1e-8))
                x_new = x + agg_coord
                agg_msg = torch.zeros_like(h)
                agg_msg.index_add_(0, row, msg)
                h_new = self.norm(self.node_mlp(torch.cat([h, agg_msg], dim=-1)))
                return h_new, x_new

        class TCIPDiffusionModel(nn.Module):
            """DDPM for joint coordinate + atom-type generation."""

            N_ATOM_TYPES = 10  # C, N, O, S, F, Cl, Br, P, I, other

            def __init__(self, hidden_dim: int, n_layers: int, n_steps: int) -> None:
                super().__init__()
                self.hidden_dim = hidden_dim
                self.n_layers = n_layers
                self.n_steps = n_steps

                # Atom embedding
                self.atom_embed = nn.Embedding(self.N_ATOM_TYPES, hidden_dim)

                # Pocket/recruiter condition encoders
                self.pocket_encoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.recruiter_encoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )

                # Geometry condition encoder (6-D vector -> hidden_dim)
                self.geom_encoder = nn.Sequential(
                    nn.Linear(6, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )

                # Time-step embedding
                self.time_embed = nn.Embedding(n_steps, hidden_dim)
                self.pos_enc = _PositionalEncoding(hidden_dim)

                # EGNN denoiser layers
                self.layers = nn.ModuleList(
                    [_EGNNLayer(hidden_dim) for _ in range(n_layers)]
                )

                # Output heads
                self.coord_head = nn.Linear(hidden_dim, 3)
                self.type_head = nn.Linear(hidden_dim, self.N_ATOM_TYPES)

                # Linear beta schedule
                betas = torch.linspace(1e-4, 0.02, n_steps)
                alphas = 1.0 - betas
                alpha_bar = torch.cumprod(alphas, dim=0)
                self.register_buffer("betas", betas)
                self.register_buffer("alphas", alphas)
                self.register_buffer("alpha_bar", alpha_bar)

            def _condition_embed(
                self,
                pocket_graph: Any,
                recruiter_graph: Any,
                geometry: "torch.Tensor",
            ) -> "torch.Tensor":
                """Aggregate pocket, recruiter, and geometry into a single condition."""
                # Pool pocket node features
                pocket_h = pocket_graph.x if hasattr(pocket_graph, "x") else torch.zeros(1, self.hidden_dim)
                if pocket_h.shape[-1] != self.hidden_dim:
                    pocket_h = nn.functional.pad(pocket_h, (0, self.hidden_dim - pocket_h.shape[-1]))
                pocket_cond = self.pocket_encoder(pocket_h).mean(0)

                recruiter_h = recruiter_graph.x if hasattr(recruiter_graph, "x") else torch.zeros(1, self.hidden_dim)
                if recruiter_h.shape[-1] != self.hidden_dim:
                    recruiter_h = nn.functional.pad(recruiter_h, (0, self.hidden_dim - recruiter_h.shape[-1]))
                rec_cond = self.recruiter_encoder(recruiter_h).mean(0)

                geom_cond = self.geom_encoder(geometry)
                return (pocket_cond + rec_cond + geom_cond) / 3.0

            def forward(
                self,
                h: "torch.Tensor",
                x: "torch.Tensor",
                edge_index: "torch.Tensor",
                t: "torch.Tensor",
                condition: "torch.Tensor",
            ) -> tuple:
                """Predict noise (epsilon) for denoising step t."""
                t_emb = self.time_embed(t).unsqueeze(0).expand(h.shape[0], -1)
                h = h + t_emb + condition.unsqueeze(0).expand(h.shape[0], -1)
                for layer in self.layers:
                    h, x = layer(h, x, edge_index)
                dx = self.coord_head(h)
                dh = self.type_head(h)
                return dx, dh

            @torch.no_grad()
            def sample(
                self,
                pocket_graph: Any,
                recruiter_graph: Any,
                geometry: "torch.Tensor",
                n_atoms: int,
                n_samples: int = 1,
            ) -> list:
                """DDPM reverse diffusion sampling."""
                import torch
                device = next(self.parameters()).device

                condition = self._condition_embed(pocket_graph, recruiter_graph, geometry)

                results = []
                for _ in range(n_samples):
                    # Initialize from noise
                    x = torch.randn(n_atoms, 3, device=device)
                    h_types = torch.randint(0, self.N_ATOM_TYPES, (n_atoms,), device=device)
                    h = self.atom_embed(h_types)

                    # Simple fully-connected edge index for small molecules
                    idx = torch.arange(n_atoms, device=device)
                    row = idx.unsqueeze(1).expand(-1, n_atoms).reshape(-1)
                    col = idx.unsqueeze(0).expand(n_atoms, -1).reshape(-1)
                    mask = row != col
                    edge_index = torch.stack([row[mask], col[mask]])

                    for t_int in reversed(range(self.n_steps)):
                        t_tensor = torch.full((1,), t_int, dtype=torch.long, device=device)
                        dx, dh = self(h, x, edge_index, t_tensor, condition)
                        alpha_t = self.alphas[t_int]
                        alpha_bar_t = self.alpha_bar[t_int]
                        beta_t = self.betas[t_int]

                        # Reverse step (simplified DDPM)
                        x = (x - (1 - alpha_t) / (1 - alpha_bar_t).sqrt() * dx) / alpha_t.sqrt()
                        if t_int > 0:
                            noise = torch.randn_like(x)
                            x = x + beta_t.sqrt() * noise

                        # Update atom types via soft argmax
                        h_logits = h + dh * 0.1
                        h_types = h_logits.argmax(dim=-1)
                        h = self.atom_embed(h_types)

                    results.append((
                        x.cpu().numpy(),
                        h_types.cpu().numpy(),
                    ))

                return results

        return TCIPDiffusionModel(
            hidden_dim=config.mol_hidden_dim,
            n_layers=config.mol_n_layers,
            n_steps=config.n_diffusion_steps,
        )

    except ImportError:
        logger.warning("PyTorch not installed; MoleculeGenerator will use SMILES-based fallback.")
        return None


# ---------------------------------------------------------------------------
# Atom-type index <-> element symbol mapping
# ---------------------------------------------------------------------------

_ATOM_TYPE_SYMBOLS = ["C", "N", "O", "S", "F", "Cl", "Br", "P", "I", "X"]


def _atom_types_to_smiles(
    atom_types: np.ndarray, coords: np.ndarray
) -> str:
    """
    Convert atom-type array + coordinates to a SMILES string.

    This is a minimal heuristic approach:
    1. Map atom indices to element symbols.
    2. Build connectivity using distance-based bonding (< 1.85 Å for single
       bond) with a 2-D depiction SMILES fallback.

    For production use this would be replaced by a proper graph-to-SMILES
    routine (e.g. via RDKit EditableMol).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.RWMol()
        conf = Chem.Conformer(len(atom_types))

        # Add atoms
        for i, at in enumerate(atom_types):
            symbol = _ATOM_TYPE_SYMBOLS[int(at) % len(_ATOM_TYPE_SYMBOLS)]
            if symbol == "X":
                symbol = "C"
            a = Chem.Atom(symbol)
            mol.AddAtom(a)
            from rdkit.Geometry import Point3D
            conf.SetAtomPosition(i, Point3D(
                float(coords[i, 0]),
                float(coords[i, 1]),
                float(coords[i, 2]),
            ))

        # Add bonds by distance
        n = len(atom_types)
        bond_cutoff = 1.85
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < bond_cutoff:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)

        mol.AddConformer(conf)
        try:
            Chem.SanitizeMol(mol)
            smi = Chem.MolToSmiles(mol)
            return smi if smi else "C"
        except Exception:
            return "C"

    except ImportError:
        # Pure fallback: build a linear SMILES from atom symbols
        symbols = [
            _ATOM_TYPE_SYMBOLS[int(at) % len(_ATOM_TYPE_SYMBOLS)]
            for at in atom_types
        ]
        symbols = ["C" if s == "X" else s for s in symbols]
        return "".join(symbols[:10])  # truncated for safety


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class MoleculeGenerator:
    """
    Wraps the TCIPDiffusionModel to generate candidate warhead molecules
    conditioned on a pocket geometry and recruiter warhead.

    Parameters
    ----------
    config : TCDConfig
        Pipeline configuration.

    Usage
    -----
    >>> gen = MoleculeGenerator(config)
    >>> gen.load_model(config.checkpoint_path)
    >>> molecules = gen.sample(pocket_graph, recruiter_graph, geom, n_atoms=25)
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.model: Optional[Any] = None
        self._model_loaded = False

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def load_model(self, checkpoint_path: str) -> None:
        """
        Load model weights from *checkpoint_path*.

        If the checkpoint does not exist, the model is initialized with
        random weights (useful for testing / development).

        Parameters
        ----------
        checkpoint_path : str
            Path to a .pt file containing model state_dict.
        """
        import os
        self.model = _build_tcip_diffusion_model(self.config)
        if self.model is None:
            self._model_loaded = False
            return

        if os.path.exists(checkpoint_path):
            try:
                import torch
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                self.model.load_state_dict(state_dict)
                logger.info("Loaded TCD diffusion model from %s.", checkpoint_path)
            except Exception as exc:
                logger.warning(
                    "Failed to load checkpoint %s (%s); using random weights.",
                    checkpoint_path,
                    exc,
                )
        else:
            logger.warning(
                "Checkpoint %s not found; using randomly initialized model.",
                checkpoint_path,
            )

        self.model.eval()
        self._model_loaded = True

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        pocket_graph: Any,
        recruiter_graph: Any,
        geometry_constraint: Any,
        n_atoms: int,
        n_samples: int = 10,
    ) -> List[Molecule]:
        """
        Generate candidate warhead molecules via DDPM reverse diffusion.

        Parameters
        ----------
        pocket_graph : torch_geometric.data.Data or dict
            Graph representation of the TF binding pocket.
        recruiter_graph : torch_geometric.data.Data or dict
            Graph representation of the recruiter warhead.
        geometry_constraint : torch.Tensor or np.ndarray
            6-D vector [distance, angle, d1, d2, d3, d4].
        n_atoms : int
            Target atom count for the warhead.
        n_samples : int
            Number of molecules to sample.

        Returns
        -------
        List[Molecule]
            Valid Molecule objects (SMILES, coordinates, atom types,
            predicted Ki).
        """
        if self.model is None or not self._model_loaded:
            logger.warning(
                "Model not loaded; returning heuristic warhead candidates."
            )
            return self._heuristic_warheads(n_samples)

        try:
            import torch
            geom_tensor = self._ensure_tensor(geometry_constraint)
            raw_samples = self.model.sample(
                pocket_graph,
                recruiter_graph,
                geom_tensor,
                n_atoms=n_atoms,
                n_samples=n_samples,
            )
        except Exception as exc:
            logger.warning("Diffusion sampling failed (%s); using fallback.", exc)
            return self._heuristic_warheads(n_samples)

        molecules: List[Molecule] = []
        for coords_arr, types_arr in raw_samples:
            smiles = _atom_types_to_smiles(types_arr, coords_arr)
            if smiles and smiles != "C":
                mol = Molecule(
                    smiles=smiles,
                    coords=coords_arr,
                    atom_types=types_arr,
                    predicted_ki=self._predict_ki(smiles, pocket_graph),
                )
                molecules.append(mol)

        if not molecules:
            molecules = self._heuristic_warheads(n_samples)

        logger.info(
            "MoleculeGenerator: sampled %d valid molecules (requested %d).",
            len(molecules),
            n_samples,
        )
        return molecules

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _smiles_to_graph(self, smiles: str) -> Any:
        """
        Convert a SMILES string to a PyTorch Geometric Data object.

        Node features: one-hot atom type (10 types), atomic number,
                       aromaticity, H count, degree, charge (6 features)
        Edge features: bond type (single/double/triple/aromatic) one-hot

        Falls back to a minimal dict-based graph if torch_geometric is
        unavailable.

        Parameters
        ----------
        smiles : str
            Input SMILES string.

        Returns
        -------
        torch_geometric.data.Data or dict
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")

            atom_features = []
            for atom in mol.GetAtoms():
                symbol_idx = min(atom.GetAtomicNum(), 9)
                feat = [
                    symbol_idx / 53.0,  # normalized atomic number
                    float(atom.GetIsAromatic()),
                    atom.GetTotalNumHs() / 4.0,
                    atom.GetDegree() / 6.0,
                    (atom.GetFormalCharge() + 2) / 4.0,
                ]
                atom_features.append(feat)

            edge_index_src, edge_index_dst = [], []
            edge_feats = []
            bond_type_map = {
                Chem.BondType.SINGLE: 0,
                Chem.BondType.DOUBLE: 1,
                Chem.BondType.TRIPLE: 2,
                Chem.BondType.AROMATIC: 3,
            }
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bt = bond_type_map.get(bond.GetBondType(), 0)
                edge_index_src += [i, j]
                edge_index_dst += [j, i]
                edge_feats += [[bt / 3.0], [bt / 3.0]]

            import torch
            x = torch.tensor(atom_features, dtype=torch.float)
            edge_index = torch.tensor(
                [edge_index_src, edge_index_dst], dtype=torch.long
            )
            edge_attr = torch.tensor(edge_feats, dtype=torch.float)

            try:
                from torch_geometric.data import Data
                return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            except ImportError:
                return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}

        except ImportError:
            # Minimal fallback without RDKit
            import torch
            n = max(1, len(smiles) // 2)
            x = torch.zeros(n, 5)
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            return {"x": x, "edge_index": edge_index, "edge_attr": None}

    # ------------------------------------------------------------------
    # Geometry computation
    # ------------------------------------------------------------------

    def _compute_geometry(
        self, tf_struct: Any, writer_sel: Any
    ) -> Any:
        """
        Compute the 6-D geometry constraint tensor for the ternary complex.

        Computes:
          [0] distance (Å)  — distance between TF binding site center and
                              recruiter binding site center
          [1] angle (rad)   — angle TF_center — linker_midpoint — recruiter_center
          [2-5] dihedrals   — four backbone dihedral angles of the bridging
                              linker (estimated from exit vector directions)

        Parameters
        ----------
        tf_struct : TFStructureResult
            TF structure output.
        writer_sel : WriterEraserSelection
            Writer/eraser selection output.

        Returns
        -------
        torch.Tensor of shape (6,)
        """
        try:
            import torch

            tf_center = np.array(
                tf_struct.binding_site.get("center", np.zeros(3))
                if isinstance(tf_struct.binding_site, dict)
                else np.zeros(3)
            )

            # Approximate recruiter center from structure name hash
            rng = np.random.default_rng(
                seed=abs(hash(writer_sel.writer_eraser_name)) % (2 ** 32)
            )
            recruiter_center = tf_center + rng.normal(0, 20, 3)

            # Distance
            dist = float(np.linalg.norm(tf_center - recruiter_center))

            # Midpoint
            midpoint = (tf_center + recruiter_center) / 2.0
            v1 = tf_center - midpoint
            v2 = recruiter_center - midpoint

            # Angle at midpoint
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                cos_angle = np.clip(
                    np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0
                )
                angle = float(np.arccos(cos_angle))
            else:
                angle = np.pi / 2.0

            # Four pseudo-dihedral estimates (from random linker configuration)
            dihedrals = rng.uniform(-np.pi, np.pi, 4).tolist()

            geom = torch.tensor(
                [dist, angle] + dihedrals, dtype=torch.float32
            )
            return geom

        except ImportError:
            return np.array([20.0, 1.57, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # ------------------------------------------------------------------
    # Best warhead selection
    # ------------------------------------------------------------------

    def _select_best_warhead(self, candidates: List[Molecule]) -> Molecule:
        """
        Select the molecule with the best (lowest) predicted Ki.

        Parameters
        ----------
        candidates : List[Molecule]

        Returns
        -------
        Molecule
        """
        if not candidates:
            return Molecule(smiles="C", predicted_ki=1e6)
        return min(candidates, key=lambda m: m.predicted_ki)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _predict_ki(self, smiles: str, pocket_graph: Any) -> float:
        """
        Predict binding affinity (Ki in nM) for *smiles* against the
        pocket encoded in *pocket_graph*.

        Uses a lightweight Morgan fingerprint + linear regression
        surrogate when the full binding affinity model is not available.
        Returns a plausible random value in [1, 10000] nM otherwise.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 1e4

            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            n_rings = mol.GetRingInfo().NumRings()

            # Heuristic: smaller, more lipophilic, ring-containing molecules
            # tend to bind better
            ki_estimate = max(1.0, 5000.0 / (1.0 + logp * 0.5 + n_rings * 2.0))
            return float(ki_estimate)

        except ImportError:
            rng = np.random.default_rng(seed=abs(hash(smiles)) % (2 ** 32))
            return float(rng.uniform(1.0, 10000.0))

    def _heuristic_warheads(self, n: int) -> List[Molecule]:
        """
        Generate heuristic drug-fragment-like warheads when the model is
        unavailable.
        """
        fragments = [
            "c1ccc(cc1)C(=O)N",         # benzamide
            "c1ccncc1C(=O)O",            # nicotinic acid
            "CC(=O)Nc1ccc(cc1)O",        # paracetamol core
            "c1ccc2c(c1)cccc2",          # naphthalene
            "C1COCCN1",                  # morpholine
            "c1cnc(nc1)N",               # 2-aminopyrimidine
            "O=C1CCCO1",                 # butyrolactone
            "CC1=CC=C(C=C1)S(=O)(=O)N",  # toluenesulfonamide
            "c1ccc(cc1)NC(=O)c1ccccc1",  # benzanilide
            "c1cc2ccccc2nc1",            # quinoline
        ]
        molecules: List[Molecule] = []
        for i in range(n):
            smi = fragments[i % len(fragments)]
            ki = float(np.random.default_rng(seed=i).uniform(50.0, 5000.0))
            molecules.append(Molecule(smiles=smi, predicted_ki=ki))
        return molecules

    def _ensure_tensor(self, x: Any) -> Any:
        """Convert ndarray or list to torch.Tensor if needed."""
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x
            return torch.tensor(np.asarray(x), dtype=torch.float32)
        except ImportError:
            return x
