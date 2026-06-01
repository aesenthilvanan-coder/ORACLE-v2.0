from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_pdb_coords(pdb_path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load Cα coordinates, residue names, and chain IDs from a PDB file."""
    coords, res_names, chains = [], [], []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                res = line[17:20].strip()
                chain = line[21].strip()
                coords.append([x, y, z])
                res_names.append(res)
                chains.append(chain)
    return np.array(coords, dtype=np.float32), res_names, chains


def compute_pocket_center(coords: np.ndarray) -> np.ndarray:
    return coords.mean(axis=0)


def compute_pocket_volume(coords: np.ndarray, probe_radius: float = 1.4) -> float:
    if len(coords) == 0:
        return 0.0
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(coords)
        return float(hull.volume)
    except Exception:
        span = coords.max(axis=0) - coords.min(axis=0)
        return float(np.prod(span + 2 * probe_radius))


def build_contact_map(coords: np.ndarray, cutoff_A: float = 8.0) -> np.ndarray:
    diff = coords[:, None] - coords[None, :]
    dist = np.linalg.norm(diff, axis=-1)
    return dist < cutoff_A


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None] - coords[None, :]
    return np.linalg.norm(diff, axis=-1)


def find_binding_residues(
    pocket_coords: np.ndarray,
    protein_coords: np.ndarray,
    cutoff_A: float = 5.0,
) -> np.ndarray:
    diff = protein_coords[:, None] - pocket_coords[None, :]
    min_dist = np.linalg.norm(diff, axis=-1).min(axis=-1)
    return np.where(min_dist < cutoff_A)[0]


def align_structures(
    mobile_coords: np.ndarray,
    reference_coords: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Kabsch algorithm for optimal rigid-body alignment."""
    mob_center = mobile_coords.mean(axis=0)
    ref_center = reference_coords.mean(axis=0)
    mob_c = mobile_coords - mob_center
    ref_c = reference_coords - ref_center

    n = min(len(mob_c), len(ref_c))
    H = mob_c[:n].T @ ref_c[:n]
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T

    aligned = (mob_c[:n] @ R.T) + ref_center
    rmsd = float(np.sqrt(((aligned - ref_c[:n]) ** 2).sum(axis=-1).mean()))
    full_aligned = (mobile_coords - mob_center) @ R.T + ref_center
    return full_aligned, rmsd
