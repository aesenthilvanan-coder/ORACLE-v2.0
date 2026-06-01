"""Tests for oracle.utils.sequence_split — MMseqs2 sequence-identity splitting."""

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EXAMPLE_SEQUENCES = {
    "TP53":  "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPG",
    "MYC":   "MPLNVSFTNRNYDLDYDSVQPYFYCDEEENFYQQQQQSELQPPAPEDLPQGKPQGSGSN",
    "RUNX1": "MASNSLFALSLTDDEFDPQTSRRNLLKASEPEDMSELYEAQMRHRPTLKAELQNLEREAG",
    "CEBPA": "MSSSSSSPAAPAPAPASWAAPAPAPASWAAPAPAPASWAAPSPGGSRAASSSPNMPYVSPP",
    "SNAI1": "MPRSFLVRKPSDPHIKAELESYIESQLRQQREQLLKEKEALQKELEQLRKSQDQLEQELQ",
    "CDX2":  "MQFVHPSPPMFRGPVYGAPYGLSPGYSPNLKRTKTAQKRAANEGSSTSGLPHHPQHPHSP",
}


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_write_and_parse_fasta(tmp_path):
    from oracle.utils.sequence_split import _write_fasta

    fasta = tmp_path / "seqs.fa"
    _write_fasta(EXAMPLE_SEQUENCES, str(fasta))
    content = fasta.read_text()
    for seq_id in EXAMPLE_SEQUENCES:
        assert f">{seq_id}" in content


def test_assign_splits_basic():
    from oracle.utils.sequence_split import _assign_splits

    clusters = {
        "A": ["A", "B"],
        "C": ["C"],
        "D": ["D", "E"],
        "F": ["F"],
    }
    all_ids = ["A", "B", "C", "D", "E", "F"]
    splits = _assign_splits(clusters, all_ids, ratios=(0.5, 0.25, 0.25), seed=0)

    assert set(splits.keys()) == {"train", "val", "test"}
    all_assigned = splits["train"] + splits["val"] + splits["test"]
    assert sorted(all_assigned) == sorted(all_ids), "all IDs must be assigned"


def test_assign_splits_no_cross_cluster_leakage():
    """Cluster members must never be split across different splits."""
    from oracle.utils.sequence_split import _assign_splits

    clusters = {
        "A": ["A", "B", "C"],
        "D": ["D", "E"],
        "F": ["F", "G", "H", "I"],
        "J": ["J"],
    }
    all_ids = [m for members in clusters.values() for m in members]
    splits = _assign_splits(clusters, all_ids, ratios=(0.6, 0.2, 0.2), seed=42)

    # Build reverse map: id → split
    id_to_split = {}
    for split_name, ids in splits.items():
        for sid in ids:
            id_to_split[sid] = split_name

    for rep, members in clusters.items():
        assigned = {id_to_split.get(m) for m in members}
        assert len(assigned) == 1, (
            f"Cluster {rep} members span multiple splits: {assigned}"
        )


def test_assign_splits_respects_ratios():
    from oracle.utils.sequence_split import _assign_splits

    clusters = {str(i): [str(i)] for i in range(100)}
    all_ids = [str(i) for i in range(100)]
    splits = _assign_splits(clusters, all_ids, ratios=(0.7, 0.15, 0.15), seed=0)

    train_f = len(splits["train"]) / 100
    assert 0.60 <= train_f <= 0.80, f"train fraction {train_f:.2f} out of range"


def test_mmseqs2_split_integration():
    """Full integration test: runs MMseqs2 on real sequences."""
    import shutil

    if not shutil.which("mmseqs"):
        pytest.skip("MMseqs2 not available in PATH")

    from oracle.utils.sequence_split import mmseqs2_split

    splits = mmseqs2_split(
        sequences=EXAMPLE_SEQUENCES,
        ratios=(0.60, 0.20, 0.20),
        identity=0.30,
        seed=42,
    )

    all_assigned = splits["train"] + splits["val"] + splits["test"]
    assert sorted(all_assigned) == sorted(EXAMPLE_SEQUENCES.keys()), \
        "all sequence IDs must be present in exactly one split"

    # No duplicate IDs across splits
    assert len(all_assigned) == len(set(all_assigned)), "duplicate IDs detected"


def test_save_and_load_splits(tmp_path):
    from oracle.utils.sequence_split import save_splits, load_splits

    splits = {
        "train": ["TP53", "MYC"],
        "val": ["RUNX1"],
        "test": ["CEBPA"],
    }
    save_splits(splits, str(tmp_path))
    loaded = load_splits(str(tmp_path))

    for name in ("train", "val", "test"):
        assert sorted(loaded[name]) == sorted(splits[name])
