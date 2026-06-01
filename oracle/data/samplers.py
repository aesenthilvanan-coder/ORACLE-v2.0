from typing import Iterator, List, Optional
import torch
from torch.utils.data import Sampler
import numpy as np


class StratifiedSampler(Sampler):
    """Stratified sampler that ensures balanced cancer/normal label representation per batch."""

    def __init__(
        self,
        labels: List[int],
        n_samples: Optional[int] = None,
        replacement: bool = True,
    ):
        self.labels = np.array(labels)
        classes, counts = np.unique(self.labels, return_counts=True)
        self.classes = classes
        self.class_indices = {c: np.where(self.labels == c)[0] for c in classes}
        n_per_class = counts.min()
        self.n_samples = n_samples or (n_per_class * len(classes))
        self.replacement = replacement

    def __iter__(self) -> Iterator[int]:
        n_per_class = self.n_samples // len(self.classes)
        indices = []
        for c in self.classes:
            idxs = self.class_indices[c]
            if self.replacement:
                chosen = np.random.choice(idxs, n_per_class, replace=True)
            else:
                chosen = np.random.choice(idxs, min(n_per_class, len(idxs)), replace=False)
            indices.extend(chosen.tolist())
        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self.n_samples


class WeightedImportanceSampler(Sampler):
    """Sample proportional to gene importance weights."""

    def __init__(self, weights: List[float], n_samples: int, replacement: bool = True):
        self.weights = torch.tensor(weights, dtype=torch.float64)
        self.n_samples = n_samples
        self.replacement = replacement

    def __iter__(self) -> Iterator[int]:
        return iter(torch.multinomial(self.weights, self.n_samples, self.replacement).tolist())

    def __len__(self) -> int:
        return self.n_samples


class CancerTypeGroupSampler(Sampler):
    """Groups samples by cancer type to avoid information leakage across batches."""

    def __init__(self, cancer_types: List[str], batch_size: int):
        from collections import defaultdict
        self.batch_size = batch_size
        groups = defaultdict(list)
        for i, ct in enumerate(cancer_types):
            groups[ct].append(i)
        self.groups = list(groups.values())

    def __iter__(self) -> Iterator[int]:
        indices = []
        for group in self.groups:
            shuffled = list(group)
            np.random.shuffle(shuffled)
            indices.extend(shuffled)
        return iter(indices)

    def __len__(self) -> int:
        return sum(len(g) for g in self.groups)
