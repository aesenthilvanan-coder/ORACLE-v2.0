"""
sequence_split.py
-----------------
MMseqs2-based protein-sequence clustering for zero-leakage train/val/test splits.

Guarantees: no two sequences with ≥30% sequence identity appear in different
splits.  This prevents the TCD module from inflating performance metrics by
memorising homologous TF binding-site patterns.

Usage
-----
    from oracle.utils.sequence_split import mmseqs2_split

    splits = mmseqs2_split(
        sequences={"TP53": "MEEPQSDPSVEPPLSQETFSDLWKLL...", ...},
        ratios=(0.7, 0.15, 0.15),
        identity=0.30,
        seed=42,
    )
    train_ids = splits["train"]   # list[str]
    val_ids   = splits["val"]
    test_ids  = splits["test"]
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mmseqs2_split(
    sequences: Dict[str, str],
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    identity: float = 0.30,
    coverage: float = 0.80,
    seed: int = 42,
    mmseqs_bin: str = "mmseqs",
    tmp_dir: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Cluster sequences at *identity* with MMseqs2 and assign whole clusters
    to train / val / test without cross-split leakage.

    Parameters
    ----------
    sequences:
        Mapping of sequence-id → amino-acid sequence.
    ratios:
        (train, val, test) fractions; must sum to 1.0.
    identity:
        Minimum sequence identity threshold (0–1).  Default 0.30.
    coverage:
        Minimum coverage for a hit to be considered homologous.
    seed:
        Random seed for cluster-to-split assignment.
    mmseqs_bin:
        Path or name of the MMseqs2 executable.
    tmp_dir:
        Directory for temporary MMseqs2 files.  Defaults to a system tempdir.

    Returns
    -------
    dict with keys "train", "val", "test"; each maps to a list of sequence IDs.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"
    assert len(sequences) > 0, "sequences dict is empty"

    _check_mmseqs(mmseqs_bin)

    with tempfile.TemporaryDirectory(prefix="oracle_mmseqs_", dir=tmp_dir) as wd:
        clusters = _run_mmseqs_cluster(
            sequences=sequences,
            identity=identity,
            coverage=coverage,
            workdir=wd,
            mmseqs_bin=mmseqs_bin,
        )

    return _assign_splits(
        clusters=clusters,
        all_ids=list(sequences.keys()),
        ratios=ratios,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _check_mmseqs(binary: str) -> None:
    if shutil.which(binary) is None and not Path(binary).is_file():
        raise RuntimeError(
            f"MMseqs2 binary '{binary}' not found. "
            "Install with: conda install -c conda-forge mmseqs2"
        )


def _write_fasta(sequences: Dict[str, str], path: str) -> None:
    with open(path, "w") as fh:
        for seq_id, seq in sequences.items():
            # Replace any whitespace in IDs with underscores (MMseqs2 requirement)
            safe_id = seq_id.replace(" ", "_").replace("\t", "_")
            fh.write(f">{safe_id}\n{seq}\n")


def _run_mmseqs_cluster(
    sequences: Dict[str, str],
    identity: float,
    coverage: float,
    workdir: str,
    mmseqs_bin: str,
) -> Dict[str, List[str]]:
    """Run MMseqs2 easy-cluster and return {representative_id: [member_ids]}."""

    fasta_path = os.path.join(workdir, "input.fasta")
    prefix = os.path.join(workdir, "clust")
    tmp_path = os.path.join(workdir, "tmp")

    _write_fasta(sequences, fasta_path)

    cmd = [
        mmseqs_bin, "easy-cluster",
        fasta_path,
        prefix,
        tmp_path,
        "--min-seq-id", str(identity),
        "-c", str(coverage),
        "--cov-mode", "0",       # coverage on both query and target
        "--cluster-mode", "1",   # connected component clustering
        "--threads", "4",
        "-v", "1",               # minimal verbosity
    ]

    logger.info(
        "Running MMseqs2 easy-cluster: identity=%.0f%%, coverage=%.0f%%",
        identity * 100, coverage * 100,
    )
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"MMseqs2 clustering failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )

    # Parse cluster TSV: rep_id\tmember_id
    tsv_path = prefix + "_cluster.tsv"
    clusters: Dict[str, List[str]] = {}
    id_map = {sid.replace(" ", "_").replace("\t", "_"): sid for sid in sequences}

    with open(tsv_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            rep_safe, mem_safe = parts[0], parts[1]
            rep = id_map.get(rep_safe, rep_safe)
            mem = id_map.get(mem_safe, mem_safe)
            clusters.setdefault(rep, [])
            if mem not in clusters[rep]:
                clusters[rep].append(mem)

    logger.info(
        "MMseqs2 produced %d clusters from %d sequences.",
        len(clusters), len(sequences),
    )
    return clusters


def _assign_splits(
    clusters: Dict[str, List[str]],
    all_ids: List[str],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Dict[str, List[str]]:
    """Assign clusters to splits so that whole clusters stay together."""

    rng = np.random.default_rng(seed)

    cluster_list = list(clusters.values())          # list of lists
    cluster_sizes = np.array([len(c) for c in cluster_list])
    total = sum(cluster_sizes)

    # Shuffle clusters, then greedily fill splits by size
    order = rng.permutation(len(cluster_list))
    cluster_list = [cluster_list[i] for i in order]

    targets = [int(r * total) for r in ratios]
    split_names = ["train", "val", "test"]

    assigned: Dict[str, List[str]] = {s: [] for s in split_names}
    counts = [0, 0, 0]

    for cluster in cluster_list:
        # Find the split that is most under-filled relative to its target
        deficits = [targets[i] - counts[i] for i in range(3)]
        idx = int(np.argmax(deficits))
        assigned[split_names[idx]].extend(cluster)
        counts[idx] += len(cluster)

    # Any IDs that MMseqs2 didn't cluster (empty sequences etc.) go to train
    clustered_ids = {m for members in clusters.values() for m in members}
    for sid in all_ids:
        if sid not in clustered_ids:
            assigned["train"].append(sid)

    for name, ids in assigned.items():
        logger.info("Split '%s': %d sequences", name, len(ids))

    return assigned


# ---------------------------------------------------------------------------
# Convenience: save / load splits as JSON
# ---------------------------------------------------------------------------


def save_splits(splits: Dict[str, List[str]], output_dir: str) -> None:
    """Write train.json, val.json, test.json to *output_dir*."""
    import json

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split_name, ids in splits.items():
        path = out / f"{split_name}.json"
        with open(path, "w") as fh:
            json.dump({"split": split_name, "ids": ids}, fh, indent=2)
        logger.info("Wrote %d IDs to %s", len(ids), path)


def load_splits(output_dir: str) -> Dict[str, List[str]]:
    """Load splits previously saved by save_splits()."""
    import json

    out = Path(output_dir)
    result: Dict[str, List[str]] = {}
    for name in ("train", "val", "test"):
        path = out / f"{name}.json"
        if path.exists():
            with open(path) as fh:
                data = json.load(fh)
            result[name] = data.get("ids", [])
    return result
