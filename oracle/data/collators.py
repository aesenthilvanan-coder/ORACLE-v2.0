from typing import Dict, List, Optional
import torch
import numpy as np


def cancer_score_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for CancerScoreDataset batches."""
    expressions = torch.stack([torch.tensor(b["expression"], dtype=torch.float32) for b in batch])
    labels = torch.tensor([b["cancer_label"] for b in batch], dtype=torch.float32)
    out = {"expression": expressions, "cancer_label": labels}

    if "pseudotime" in batch[0]:
        out["pseudotime"] = torch.tensor([b["pseudotime"] for b in batch], dtype=torch.float32)

    return out


def grn_graph_collate(batch: List[Dict]):
    """Collate a list of PyG Data objects with associated labels."""
    try:
        from torch_geometric.data import Batch
        graphs = [b["graph"] for b in batch]
        cancer_scores = torch.tensor([b["cancer_score"] for b in batch], dtype=torch.float32)
        reversion_labels = torch.tensor([b["reversion_label"] for b in batch], dtype=torch.float32)
        return {
            "graph": Batch.from_data_list(graphs),
            "cancer_score": cancer_scores,
            "reversion_label": reversion_labels,
        }
    except ImportError:
        raise ImportError("torch_geometric required for GRN graph collation")


def tcip_diffusion_collate(batch: List[Dict]) -> Dict:
    """Collate TCIP molecule batches for diffusion model training."""
    max_atoms = max(b["coords"].shape[0] for b in batch)
    B = len(batch)

    coords = torch.zeros(B, max_atoms, 3, dtype=torch.float32)
    atom_types = torch.zeros(B, max_atoms, dtype=torch.long)
    mask = torch.zeros(B, max_atoms, dtype=torch.bool)

    for i, b in enumerate(batch):
        n = b["coords"].shape[0]
        coords[i, :n] = torch.tensor(b["coords"], dtype=torch.float32)
        atom_types[i, :n] = torch.tensor(b["atom_types"], dtype=torch.long)
        mask[i, :n] = True

    out: Dict = {"coords": coords, "atom_types": atom_types, "mask": mask}

    if "geometry" in batch[0]:
        out["geometry"] = torch.stack([torch.tensor(b["geometry"], dtype=torch.float32) for b in batch])

    return out


def variable_length_collate(batch: List[Dict], pad_keys: Optional[List[str]] = None) -> Dict:
    """Generic collate that pads specified keys along dim 0."""
    if pad_keys is None:
        pad_keys = []
    out = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if key in pad_keys and isinstance(vals[0], (torch.Tensor, np.ndarray)):
            tensors = [torch.as_tensor(v) for v in vals]
            max_len = max(t.shape[0] for t in tensors)
            padded = torch.zeros(len(tensors), max_len, *tensors[0].shape[1:], dtype=tensors[0].dtype)
            for i, t in enumerate(tensors):
                padded[i, :t.shape[0]] = t
            out[key] = padded
        elif isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)
        elif isinstance(vals[0], (int, float)):
            out[key] = torch.tensor(vals)
        else:
            out[key] = vals
    return out
