"""
molecule_diffusion.py
---------------------
Equivariant diffusion model for TCIP molecule generation.

Generates TF-binding warhead + linker conditioned on:
  - TF binding pocket (3-D structural graph)
  - Recruiter warhead (molecular graph)
  - Geometry constraint (distance, angle, 4 dihedrals)

Based on the DiffSBDD architecture with SE(3)-equivariant score network.

Denoising diffusion probabilistic model (DDPM) with:
  - Forward process: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t)*x_0, (1-alpha_bar_t)*I)
  - Reverse process: learned p_theta(x_{t-1} | x_t)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    Data = None  # type: ignore

from oracle.models.shared.se3_equivariant import SE3EquivariantEncoder, EGNNLayer
from oracle.models.shared.graph_layers import MolecularGraphEncoder


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TCDConfig:
    """Hyper-parameters for the TCIP diffusion model."""

    mol_hidden_dim: int = 256
    mol_n_layers: int = 8
    n_diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02


# ---------------------------------------------------------------------------
# Placeholder Molecule result type
# ---------------------------------------------------------------------------


class Molecule:
    """Lightweight container for a generated molecule."""

    def __init__(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        atom_logits: torch.Tensor,
    ) -> None:
        self.coords = coords          # (n_atoms, 3)
        self.atom_types = atom_types  # (n_atoms,)   int
        self.atom_logits = atom_logits  # (n_atoms, 10)

    def __repr__(self) -> str:
        return (
            f"Molecule(n_atoms={self.coords.size(0)}, "
            f"atom_types={self.atom_types.tolist()})"
        )


# ---------------------------------------------------------------------------
# SE(3)-equivariant score network
# ---------------------------------------------------------------------------


class SE3EquivariantScoreNetwork(nn.Module):
    """Score network for DDPM reverse process.

    Takes noisy coordinates, atom embeddings, timestep embedding, and
    conditioning context, then predicts the coordinate score (noise estimate)
    and per-atom-type logits via a stack of EGNN layers.

    Parameters
    ----------
    hidden_dim : int
    n_layers   : int
    cutoff     : float  (unused in current soft implementation, kept for API)
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        cutoff: float = 10.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff

        # Timestep sinusoidal embedding -> hidden_dim
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Fuse: atom_emb + time_emb + context
        self.input_proj = nn.Linear(hidden_dim * 3, hidden_dim)

        # EGNN layers
        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim) for _ in range(n_layers)]
        )

        # Coordinate score head: per-atom 3-D vector (equivariant by construction)
        self.coord_score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),  # scalar weight applied to relative pos
        )

        # Atom logit head
        self.atom_logit_head = nn.Linear(hidden_dim, 10)

    @staticmethod
    def _sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """Standard sinusoidal time-step embedding (Vaswani et al.)."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32,
                                             device=timesteps.device) / half
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # (batch_or_atoms, dim)

    def forward(
        self,
        noisy_coords: torch.Tensor,
        atom_emb: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict coordinate score and atom logits.

        Parameters
        ----------
        noisy_coords : (n_atoms, 3)
        atom_emb     : (n_atoms, hidden_dim)   atom type embeddings
        timestep     : (,) or (1,)             current diffusion step
        context      : (n_atoms, hidden_dim)   conditioning context per atom
        edge_index   : (2, n_edges)

        Returns
        -------
        dict with ``coord_score`` (n_atoms, 3) and ``atom_logits`` (n_atoms, 10)
        """
        n_atoms = atom_emb.size(0)

        # Timestep embedding broadcast to all atoms
        t_scalar = timestep.view(1).expand(n_atoms)              # (n_atoms,)
        t_emb = self._sinusoidal_embedding(t_scalar, self.hidden_dim)  # (n_atoms, D)
        t_emb = self.time_embed(t_emb)                           # (n_atoms, D)

        # Fuse features
        h = self.input_proj(
            torch.cat([atom_emb, t_emb, context], dim=-1)
        )                                                        # (n_atoms, D)
        pos = noisy_coords.clone()

        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)

        # Coordinate score: we parameterise it via per-edge direction weighting
        # then scatter to per-atom score via a residual from noisy_coords
        coord_score = pos - noisy_coords                         # (n_atoms, 3)

        atom_logits = self.atom_logit_head(h)                   # (n_atoms, 10)

        return {"coord_score": coord_score, "atom_logits": atom_logits,
                "node_repr": h}


# ---------------------------------------------------------------------------
# TCIPDiffusionModel
# ---------------------------------------------------------------------------


class TCIPDiffusionModel(nn.Module):
    """Equivariant DDPM for TCIP molecule generation.

    Conditions the reverse diffusion on:
    1. TF binding pocket (SE3-equivariant encoding)
    2. Recruiter/epigenetic warhead (molecular graph encoding)
    3. Geometry constraint: [distance, angle, d1, d2, d3, d4] (6 floats)

    Parameters
    ----------
    config : TCDConfig or compatible object
    """

    def __init__(self, config) -> None:
        super().__init__()

        hidden_dim = getattr(config, "mol_hidden_dim", 256)
        n_layers = getattr(config, "mol_n_layers", 8)
        n_timesteps = getattr(config, "n_diffusion_steps", 1000)
        beta_start = getattr(config, "beta_start", 1e-4)
        beta_end = getattr(config, "beta_end", 0.02)

        self.hidden_dim = hidden_dim
        self.n_timesteps = n_timesteps

        # --- Diffusion schedule -------------------------------------------
        betas = torch.linspace(beta_start, beta_end, n_timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        # Register as buffers so they move with .to(device)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

        # --- Embeddings ---------------------------------------------------
        # Atom type embedding: H=0 .. P=9
        self.atom_embedder = nn.Embedding(10, hidden_dim)

        # --- Conditioning encoders ----------------------------------------
        # Pocket: 3-D structural graph -> per-node embeddings -> mean pool
        self.pocket_encoder = SE3EquivariantEncoder(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
            n_layers=4,
        )

        # Recruiter warhead: molecular graph -> graph-level embedding
        self.recruiter_encoder = MolecularGraphEncoder(hidden_dim=hidden_dim)

        # Geometry constraint: [dist, angle, d1, d2, d3, d4] -> hidden_dim
        self.geometry_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- Score network ------------------------------------------------
        self.score_network = SE3EquivariantScoreNetwork(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            cutoff=10.0,
        )

        # --- Output heads -------------------------------------------------
        self.atom_type_head = nn.Linear(hidden_dim, 10)
        # Bond prediction: bilinear between pairs of node representations
        self.bond_head = nn.Bilinear(hidden_dim, hidden_dim, 4)

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Bilinear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Conditioning context builder
    # ------------------------------------------------------------------

    def _build_context(
        self,
        n_atoms: int,
        pocket_graph: Data,
        recruiter_graph: Data,
        geometry_constraint: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode conditioning signals and broadcast to per-atom context.

        Parameters
        ----------
        n_atoms              : number of atoms in the generated molecule
        pocket_graph         : Data with x (atom features), pos, edge_index
        recruiter_graph      : Data with x, edge_index, edge_attr, batch
        geometry_constraint  : (6,) or (1, 6) float tensor
        device               : target device

        Returns
        -------
        torch.Tensor  (n_atoms, hidden_dim)
        """
        # Encode pocket: SE3-equivariant, then mean-pool over pocket atoms
        pocket_node_emb = self.pocket_encoder(pocket_graph)   # (n_pocket, D)
        pocket_repr = pocket_node_emb.mean(dim=0, keepdim=True)  # (1, D)

        # Encode recruiter: molecular graph -> (1, D)
        recruiter_repr = self.recruiter_encoder(recruiter_graph)  # (B_r, D)
        if recruiter_repr.dim() == 2 and recruiter_repr.size(0) > 1:
            recruiter_repr = recruiter_repr.mean(0, keepdim=True)
        recruiter_repr = recruiter_repr[:1]                    # (1, D)

        # Encode geometry constraint
        geom = geometry_constraint.view(1, 6).float().to(device)
        geom_repr = self.geometry_encoder(geom)                # (1, D)

        # Sum conditioning signals, then broadcast to n_atoms
        context = pocket_repr + recruiter_repr + geom_repr    # (1, D)
        context = context.expand(n_atoms, -1)                 # (n_atoms, D)
        return context

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        noisy_coords: torch.Tensor,
        noisy_atom_types: torch.Tensor,
        timestep: torch.Tensor,
        pocket_graph: Data,
        recruiter_graph: Data,
        geometry_constraint: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict denoised coordinates and atom types.

        Parameters
        ----------
        noisy_coords        : (n_atoms, 3)   noisy 3-D coordinates
        noisy_atom_types    : (n_atoms,)     noisy atom type indices
        timestep            : scalar or (1,) current diffusion timestep
        pocket_graph        : PyG Data of TF binding pocket
        recruiter_graph     : PyG Data of recruiter warhead
        geometry_constraint : (6,)           [dist, angle, 4 dihedrals]

        Returns
        -------
        dict:
            ``coord_score``  : (n_atoms, 3)   predicted noise on coordinates
            ``atom_logits``  : (n_atoms, 10)  predicted atom-type logits
            ``node_repr``    : (n_atoms, D)   node representations
        """
        device = noisy_coords.device
        n_atoms = noisy_coords.size(0)

        # Build a fully-connected edge_index for the generated molecule
        # (in practice one would use a distance cutoff; here: all-pairs)
        edge_index = self._build_edge_index(n_atoms, device)

        # Embed noisy atom types
        atom_emb = self.atom_embedder(noisy_atom_types.long().clamp(0, 9))

        # Build conditioning context
        context = self._build_context(
            n_atoms, pocket_graph, recruiter_graph,
            geometry_constraint, device
        )

        # Score network forward pass
        out = self.score_network(
            noisy_coords, atom_emb, timestep, context, edge_index
        )

        return {
            "coord_score": out["coord_score"],
            "atom_logits": out["atom_logits"],
            "node_repr": out["node_repr"],
        }

    # ------------------------------------------------------------------
    # Sampling (reverse diffusion)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        pocket_graph: Data,
        recruiter_graph: Data,
        geometry_constraint: torch.Tensor,
        n_atoms: int,
        n_samples: int = 10,
    ) -> List[Optional[Molecule]]:
        """Generate molecules via DDPM reverse diffusion.

        Parameters
        ----------
        pocket_graph         : Data
        recruiter_graph      : Data
        geometry_constraint  : (6,)
        n_atoms              : number of heavy atoms to generate
        n_samples            : number of independent samples

        Returns
        -------
        List of Molecule objects (None entries where generation failed).
        """
        device = next(self.parameters()).device
        results: List[Optional[Molecule]] = []

        for _ in range(n_samples):
            # Initialise from Gaussian noise
            coords = torch.randn(n_atoms, 3, device=device)
            atom_types_cont = torch.randn(n_atoms, 10, device=device)

            # Reverse diffusion: T -> 0
            for t_int in reversed(range(self.n_timesteps)):
                t = torch.tensor([t_int], dtype=torch.long, device=device)

                # Current discrete atom types (argmax of continuous logits)
                atom_types_disc = atom_types_cont.argmax(-1)

                out = self.forward(
                    coords, atom_types_disc, t,
                    pocket_graph, recruiter_graph, geometry_constraint,
                )

                # DDPM coordinate denoising step
                coords = self._ddpm_step(coords, out["coord_score"], t)

                # Atom type update: blend with predicted logits
                atom_types_cont = (
                    0.9 * atom_types_cont + 0.1 * out["atom_logits"]
                )

            # Convert to Molecule
            final_types = atom_types_cont.argmax(-1)
            mol = self._coords_to_molecule(coords, final_types,
                                           atom_types_cont)
            results.append(mol)

        return results

    # ------------------------------------------------------------------
    # DDPM denoising step
    # ------------------------------------------------------------------

    def _ddpm_step(
        self,
        coords: torch.Tensor,
        coord_score: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Standard DDPM reverse step: x_{t-1} = mu_theta(x_t, t) + sigma_t * z.

        Parameters
        ----------
        coords       : (n_atoms, 3)  x_t
        coord_score  : (n_atoms, 3)  predicted noise (epsilon_theta)
        t            : (1,)          current timestep index

        Returns
        -------
        torch.Tensor  (n_atoms, 3)  x_{t-1}
        """
        t_int = t.item()
        beta_t = self.betas[t_int]
        alpha_t = self.alphas[t_int]
        alpha_bar_t = self.alpha_bar[t_int]

        # Predicted mean: mu_theta
        coeff = beta_t / (1.0 - alpha_bar_t).sqrt()
        mu = (coords - coeff * coord_score) / alpha_t.sqrt()

        if t_int == 0:
            return mu

        # Posterior variance: beta_t (simplified, as in Ho et al.)
        sigma_t = beta_t.sqrt()
        noise = torch.randn_like(coords)
        return mu + sigma_t * noise

    # ------------------------------------------------------------------
    # Coordinate -> Molecule converter
    # ------------------------------------------------------------------

    def _coords_to_molecule(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        atom_logits: torch.Tensor,
    ) -> Optional[Molecule]:
        """Convert raw denoised tensors to a Molecule object.

        This is a lightweight conversion; in a full pipeline this would
        invoke RDKit to sanitise and validate the structure.

        Returns ``None`` if the structure appears degenerate (e.g. all atoms
        at the same position or invalid atom type distribution).
        """
        # Basic validity check: atoms should not all collapse to one point
        spread = coords.std(dim=0).mean().item()
        if spread < 1e-3:
            return None

        return Molecule(
            coords=coords.detach().cpu(),
            atom_types=atom_types.detach().cpu(),
            atom_logits=atom_logits.detach().cpu(),
        )

    # ------------------------------------------------------------------
    # Helper: build fully-connected edge_index
    # ------------------------------------------------------------------

    @staticmethod
    def _build_edge_index(n: int, device: torch.device) -> torch.Tensor:
        """Build a fully-connected (excluding self-loops) edge_index."""
        rows, cols = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        if not rows:
            # Single atom: add a self-loop to avoid empty edge_index
            return torch.zeros(2, 1, dtype=torch.long, device=device)
        return torch.tensor([rows, cols], dtype=torch.long, device=device)
