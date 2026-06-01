"""
Mathematical utility functions for the ORACLE pipeline.

All functions operate on numpy arrays unless otherwise noted.
"""

from __future__ import annotations

import numpy as np


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the pairwise cosine-similarity matrix between two sets of vectors.

    Parameters
    ----------
    a:
        Array of shape ``(M, D)``.
    b:
        Array of shape ``(N, D)``.

    Returns
    -------
    numpy.ndarray
        Similarity matrix of shape ``(M, N)`` with values in ``[-1, 1]``.

    Raises
    ------
    ValueError
        If *a* and *b* do not have the same second dimension.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.ndim == 1:
        a = a[np.newaxis, :]
    if b.ndim == 1:
        b = b[np.newaxis, :]

    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Dimension mismatch: a has {a.shape[1]} features, "
            f"b has {b.shape[1]} features."
        )

    # L2-normalise each row
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)

    return a_norm @ b_norm.T


def entropy(probs: np.ndarray) -> float:
    """Compute the Shannon entropy of a probability distribution.

    Parameters
    ----------
    probs:
        1-D array of probabilities that should sum to 1.  Zero entries are
        handled gracefully (``0 * log(0) = 0``).

    Returns
    -------
    float
        Entropy in nats (natural logarithm base).
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    # Mask zero entries to avoid log(0)
    nonzero = probs > 0
    return float(-np.sum(probs[nonzero] * np.log(probs[nonzero])))


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute the softmax of an array with an optional temperature scaling.

    Parameters
    ----------
    x:
        Input array (any shape).  Softmax is applied element-wise over the
        entire array (flattened view).
    temperature:
        Temperature parameter τ.  ``softmax(x/τ)``; values < 1 sharpen the
        distribution, values > 1 flatten it.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as *x*, with values summing to 1.0.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    x = np.asarray(x, dtype=np.float64)
    scaled = x / temperature
    # Numerically stable: subtract max before exp
    shifted = scaled - np.max(scaled)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum()


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Compute the Hamming distance between two Boolean state vectors.

    Parameters
    ----------
    a:
        Boolean array of shape ``(N,)``.
    b:
        Boolean array of shape ``(N,)``.

    Returns
    -------
    int
        Number of positions at which the two arrays differ.

    Raises
    ------
    ValueError
        If *a* and *b* have different lengths.
    """
    a = np.asarray(a, dtype=bool).ravel()
    b = np.asarray(b, dtype=bool).ravel()

    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: a={a.shape}, b={b.shape}.  "
            "Arrays must have the same length."
        )

    return int(np.sum(a != b))


def basin_overlap(basin1: np.ndarray, basin2: np.ndarray) -> float:
    """Compute the fractional overlap between two attractor basins.

    Each basin is represented as a set of Boolean state vectors (2-D array of
    shape ``(n_states, n_genes)``).

    The overlap is defined as::

        |basin1 ∩ basin2| / max(|basin1|, |basin2|)

    Parameters
    ----------
    basin1:
        Array of shape ``(N1, n_genes)`` of Boolean cell states.
    basin2:
        Array of shape ``(N2, n_genes)`` of Boolean cell states.

    Returns
    -------
    float
        Overlap score in ``[0, 1]``.
    """
    basin1 = np.asarray(basin1, dtype=np.uint8)
    basin2 = np.asarray(basin2, dtype=np.uint8)

    if basin1.ndim == 1:
        basin1 = basin1[np.newaxis, :]
    if basin2.ndim == 1:
        basin2 = basin2[np.newaxis, :]

    # Convert each state row to a hashable tuple for set intersection
    set1 = {tuple(row) for row in basin1}
    set2 = {tuple(row) for row in basin2}

    intersection = len(set1 & set2)
    denom = max(len(set1), len(set2))

    if denom == 0:
        return 0.0

    return intersection / denom


def normalize_0_1(x: np.ndarray) -> np.ndarray:
    """Linearly scale an array to the range ``[0, 1]``.

    Parameters
    ----------
    x:
        Input array of any shape.

    Returns
    -------
    numpy.ndarray
        Array of the same shape with values in ``[0, 1]``.  Returns a
        zero-filled array if ``max(x) == min(x)``.
    """
    x = np.asarray(x, dtype=np.float64)
    x_min = x.min()
    x_max = x.max()
    rng = x_max - x_min

    if rng == 0:
        return np.zeros_like(x)

    return (x - x_min) / rng
