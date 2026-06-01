import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

try:
    from torch_geometric.data import Data, Batch
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    Data = None  # type: ignore
    Batch = None  # type: ignore

logger = logging.getLogger(__name__)

ATOM_TYPES = ["H", "C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
N_ATOM_TYPES = len(ATOM_TYPES)
ATOM_TYPE_TO_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class EGNNLayer(nn.Module):
    """Equivariant Graph Neural Network layer (Satorras et al. 2021 ICML)."""

    def __init__(self, hidden_dim: int, edge_dim: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + edge_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        n_nodes = h.size(0)
        coord_diff = coords[src] - coords[dst]
        sq_dist = (coord_diff ** 2).sum(-1, keepdim=True)

        if edge_attr is not None:
            edge_input = torch.cat([h[src], h[dst], sq_dist, edge_attr], dim=-1)
        else:
            edge_input = torch.cat([h[src], h[dst], sq_dist], dim=-1)

        m_ij = self.edge_mlp(edge_input)
        coord_weights = self.coord_mlp(m_ij)

        # Pure-PyTorch scatter — no torch_scatter dependency
        coord_update = torch.zeros(n_nodes, 3, dtype=coords.dtype, device=coords.device)
        coord_update.scatter_add_(0, dst.unsqueeze(-1).expand_as(coord_diff), coord_weights * coord_diff)
        coords_new = coords + coord_update

        agg = torch.zeros(n_nodes, m_ij.size(-1), dtype=m_ij.dtype, device=m_ij.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(m_ij), m_ij)

        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
        h_new = self.norm(h + h_new)

        return h_new, coords_new


class TCIPDiffusionModel(nn.Module):
    """SE(3)-equivariant DDPM for TCIP warhead molecule generation."""

    def __init__(
        self,
        hidden_dim: int = 768,
        n_egnn_layers: int = 12,
        n_pocket_layers: int = 5,
        n_recruiter_layers: int = 4,
        n_timesteps: int = 1000,
        cutoff_A: float = 5.0,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_timesteps = n_timesteps
        self.cutoff_A = cutoff_A
        self.gradient_checkpointing = gradient_checkpointing

        self.atom_emb = nn.Embedding(N_ATOM_TYPES + 1, hidden_dim)

        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.pocket_node_emb = nn.Linear(20 + 4, hidden_dim)
        self.pocket_encoder = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(n_pocket_layers)])
        self.pocket_pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.recruiter_node_emb = nn.Linear(N_ATOM_TYPES + 10, hidden_dim)
        self.recruiter_encoder = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(n_recruiter_layers)])

        self.geom_encoder = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.score_layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(n_egnn_layers)])
        self.context_injectors = nn.ModuleList([
            nn.Linear(hidden_dim * 3, hidden_dim)
            for _ in range(n_egnn_layers)
        ])

        self.atom_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, N_ATOM_TYPES),
        )
        self.coord_score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )
        self.bond_head = nn.Bilinear(hidden_dim, hidden_dim, 4)

        self.register_buffer("betas", torch.linspace(1e-4, 0.02, n_timesteps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))

    def encode_pocket(self, pocket_data: Data) -> torch.Tensor:
        h = self.pocket_node_emb(pocket_data.x)
        coords = pocket_data.pos
        for layer in self.pocket_encoder:
            h, coords = layer(h, coords, pocket_data.edge_index)
        if hasattr(pocket_data, "batch") and pocket_data.batch is not None:
            from torch_geometric.nn import global_mean_pool
            context = global_mean_pool(h, pocket_data.batch)
        else:
            context = h.mean(0, keepdim=True)
        return self.pocket_pooling(context)

    def encode_recruiter(self, recruiter_data: Data) -> torch.Tensor:
        h = self.recruiter_node_emb(recruiter_data.x)
        coords = (
            recruiter_data.pos
            if hasattr(recruiter_data, "pos") and recruiter_data.pos is not None
            else torch.zeros(h.size(0), 3, device=h.device)
        )
        for layer in self.recruiter_encoder:
            h, coords = layer(h, coords, recruiter_data.edge_index)
        if hasattr(recruiter_data, "batch") and recruiter_data.batch is not None:
            from torch_geometric.nn import global_mean_pool
            return global_mean_pool(h, recruiter_data.batch)
        return h.mean(0, keepdim=True)

    def forward(
        self,
        noisy_coords: torch.Tensor,
        noisy_atom_types_emb: torch.Tensor,
        edge_index: torch.Tensor,
        timestep: torch.Tensor,
        pocket_context: torch.Tensor,
        recruiter_context: torch.Tensor,
        geom_constraint: torch.Tensor,
        atom_batch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        t_emb = self.time_emb(timestep)
        geom_ctx = self.geom_encoder(geom_constraint)

        pocket_ctx_per_atom = pocket_context[atom_batch]
        recruiter_ctx_per_atom = recruiter_context[atom_batch]
        geom_ctx_per_atom = geom_ctx[atom_batch]
        t_emb_per_atom = t_emb[atom_batch]

        h = noisy_atom_types_emb + t_emb_per_atom
        coords = noisy_coords.clone()

        for layer, injector in zip(self.score_layers, self.context_injectors):
            context = injector(
                torch.cat([pocket_ctx_per_atom, recruiter_ctx_per_atom, geom_ctx_per_atom], dim=-1)
            )
            h = h + context
            if self.gradient_checkpointing and self.training:
                import torch.utils.checkpoint as ckpt
                h, coords = ckpt.checkpoint(layer, h, coords, edge_index, use_reentrant=False)
            else:
                h, coords = layer(h, coords, edge_index)

        coord_score = self.coord_score_head(h)
        atom_logits = self.atom_type_head(h)
        src, dst = edge_index
        bond_logits = self.bond_head(h[src], h[dst])

        return {
            "coord_score": coord_score,
            "atom_logits": atom_logits,
            "bond_logits": bond_logits,
            "node_repr": h,          # [N_atoms, hidden_dim] — used by affinity head
        }

    @torch.no_grad()
    def sample(
        self,
        pocket_context: torch.Tensor,
        recruiter_context: torch.Tensor,
        geom_constraint: torch.Tensor,
        n_atoms: int = 25,
        n_samples: int = 10,
    ) -> List[Optional[object]]:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        device = pocket_context.device
        molecules = []

        for _ in range(n_samples):
            coords = torch.randn(n_atoms, 3, device=device) * 5.0
            atom_type_emb = torch.randn(n_atoms, self.hidden_dim, device=device)
            edge_index = self._build_edge_index(n_atoms, device=device)
            atom_batch = torch.zeros(n_atoms, dtype=torch.long, device=device)

            for t_int in reversed(range(self.n_timesteps)):
                t = torch.tensor([t_int], device=device)
                output = self.forward(
                    coords, atom_type_emb, edge_index, t,
                    pocket_context, recruiter_context, geom_constraint, atom_batch,
                )

                beta_t = self.betas[t_int]
                alpha_t = self.alphas[t_int]
                alpha_bar_t = self.alpha_bars[t_int]

                if t_int > 0:
                    alpha_bar_prev = self.alpha_bars[t_int - 1]
                    sigma_t = torch.sqrt(beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t))
                    noise = torch.randn_like(coords)
                else:
                    sigma_t = 0.0
                    noise = torch.zeros_like(coords)

                coord_pred = (
                    coords - torch.sqrt(1.0 - alpha_bar_t) * output["coord_score"]
                ) / torch.sqrt(alpha_bar_t)

                coords = (
                    torch.sqrt(alpha_bar_prev if t_int > 0 else torch.tensor(1.0)) * coord_pred
                    + torch.sqrt(torch.clamp(1.0 - (alpha_bar_prev if t_int > 0 else torch.tensor(1.0)), min=0))
                    * output["coord_score"]
                    + sigma_t * noise
                )

                atom_types = torch.argmax(output["atom_logits"], dim=-1)
                atom_type_emb = self.atom_emb(atom_types)

                if t_int % 100 == 0:
                    edge_index = self._build_edge_index_from_coords(coords, self.cutoff_A)

            atom_types_final = torch.argmax(output["atom_logits"], dim=-1).cpu().numpy()
            coords_final = coords.cpu().numpy()
            mol = self._coords_to_mol(coords_final, atom_types_final)
            molecules.append(mol)

        valid = [m for m in molecules if m is not None]
        logger.info(f"Generated {len(valid)}/{n_samples} valid molecules")
        return molecules

    def _build_edge_index(self, n_atoms: int, device: torch.device) -> torch.Tensor:
        src, dst = [], []
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    src.append(i)
                    dst.append(j)
        return torch.tensor([src, dst], dtype=torch.long, device=device)

    def _build_edge_index_from_coords(self, coords: torch.Tensor, cutoff: float) -> torch.Tensor:
        n = coords.shape[0]
        device = coords.device
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)
        dist = diff.norm(dim=-1)
        mask = (dist < cutoff) & (dist > 0)
        src, dst = mask.nonzero(as_tuple=True)
        return torch.stack([src, dst], dim=0)

    def _coords_to_mol(self, coords: np.ndarray, atom_types: np.ndarray) -> Optional[object]:
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, rdDetermineBonds

            mol = Chem.RWMol()
            conf = Chem.Conformer(len(atom_types))
            for i, (at_idx, coord) in enumerate(zip(atom_types, coords)):
                atom_symbol = ATOM_TYPES[at_idx] if at_idx < len(ATOM_TYPES) else "C"
                atom = Chem.Atom(atom_symbol)
                mol.AddAtom(atom)
                conf.SetAtomPosition(i, coord.tolist())
            mol.AddConformer(conf)

            try:
                rdDetermineBonds.DetermineConnectivity(mol)
                rdDetermineBonds.DetermineBondOrders(mol, charge=0)
            except Exception:
                mol = self._distance_based_bonding(mol, coords)

            try:
                Chem.SanitizeMol(mol)
                return mol.GetMol()
            except Exception:
                return None
        except Exception as e:
            logger.debug(f"Mol conversion failed: {e}")
            return None

    def _distance_based_bonding(self, mol, coords: np.ndarray):
        from rdkit import Chem
        radii = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05,
                 "F": 0.57, "Cl": 1.02, "Br": 1.20, "I": 1.39, "P": 1.07}
        n_atoms = len(coords)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                sym_i = mol.GetAtomWithIdx(i).GetSymbol()
                sym_j = mol.GetAtomWithIdx(j).GetSymbol()
                r_i = radii.get(sym_i, 0.77)
                r_j = radii.get(sym_j, 0.77)
                if dist < (r_i + r_j) * 1.3:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
        return mol
