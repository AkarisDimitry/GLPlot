"""N-dimensional point-cloud operations for Math Lab's Multivariate category.

Scoped to an arbitrary number of numeric columns -- unlike mathops2d.py's clustering/
density functions, which are exactly-two-column by design (a plane has a natural (x, y)
preview), dimensionality reduction only makes sense starting from three or more input
dimensions, and its whole point is collapsing them back down to two or three.

Every function here works on plain numpy arrays: which dataset columns feed it is a
panel/UI concern, not this module's.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np

__all__ = [
    "SCALE_MODES",
    "scale_columns",
    "pca",
    "UMAP_UNAVAILABLE_MESSAGE",
    "UmapUnavailableError",
    "umap_available",
    "umap_embed",
]


#: Per-column preprocessing pca()/umap_embed() apply before the algorithm itself runs.
#: Both are scale-sensitive -- a column in meters would dominate one in kilometers
#: purely from unit choice -- so this is offered as an option, not silently baked in.
SCALE_MODES: Tuple[str, ...] = ("none", "zscore", "minmax")


def _as_matrix(columns: Sequence[Any]) -> np.ndarray:
    """Stack 1-D columns of equal length into an ``(n_samples, n_features)`` float64 matrix."""
    arrays = [np.asarray(c, dtype=np.float64).ravel() for c in columns]
    if len(arrays) < 2:
        raise ValueError(f"need at least 2 columns, got {len(arrays)}")
    n = arrays[0].size
    if any(a.size != n for a in arrays):
        raise ValueError(
            f"all columns must have the same length, got {[a.size for a in arrays]}"
        )
    if n < 1:
        raise ValueError("columns must not be empty")
    return np.column_stack(arrays)


def scale_columns(matrix: Any, *, mode: str = "zscore") -> np.ndarray:
    """Per-column rescaling of a 2-D matrix.

    Args:
        matrix: ``(n_samples, n_features)`` array-like.
        mode: One of :data:`SCALE_MODES`:

            * ``"none"`` -- pass through unchanged.
            * ``"zscore"`` (default) -- each column: subtract its mean, divide by its
              standard deviation.
            * ``"minmax"`` -- each column affine-mapped onto ``[0, 1]``.

    Returns:
        float64 array, same shape as ``matrix``. A column with zero spread (constant,
        or all-nan) is left at 0 after scaling rather than dividing by zero, matching
        :func:`mathops.normalize`'s degenerate-input convention.

    Raises:
        ValueError: On an unknown ``mode``.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if mode not in SCALE_MODES:
        raise ValueError(f"unknown scale mode {mode!r}; choose one of {', '.join(SCALE_MODES)}")
    if mode == "none" or m.size == 0:
        return m.copy()

    out = np.zeros_like(m)
    for j in range(m.shape[1]):
        col = m[:, j]
        if mode == "zscore":
            std = float(np.nanstd(col))
            if std > 0.0:
                out[:, j] = (col - float(np.nanmean(col))) / std
        else:  # "minmax"
            lo, hi = float(np.nanmin(col)), float(np.nanmax(col))
            span = hi - lo
            if span > 0.0:
                out[:, j] = (col - lo) / span
    return out


def pca(columns: Sequence[Any], *, n_components: int = 2, scale: str = "zscore") -> Dict[str, Any]:
    """Principal component analysis via SVD -- pure numpy, always available.

    Args:
        columns: 2+ equal-length 1-D arrays, one per input dimension. Rows with any
            non-finite value in any column are dropped before fitting.
        n_components: Number of principal components to return, ``1 <= n_components <=
            min(n_samples, n_features)``.
        scale: One of :data:`SCALE_MODES`, applied per-column before the decomposition
            (see :func:`scale_columns`) -- PCA finds directions of maximum VARIANCE, so
            an unscaled column in different units would dominate purely from that
            choice, not from anything about the data's actual structure.

    Returns:
        Dict with:

        * ``scores``: ``(n_samples, n_components)`` -- the projected points.
        * ``components``: ``(n_components, n_features)`` -- the principal axes, as
          unit vectors in the (scaled) input's own coordinate system.
        * ``explained_variance_ratio``: ``(n_components,)`` -- fraction of total
          variance each component captures.
        * ``n_samples``, ``n_features``: finite-row / column counts actually used.

    Raises:
        ValueError: Fewer than 2 columns, mismatched column lengths, fewer than 2
            finite rows after dropping non-finite ones, or an out-of-range
            ``n_components``.
    """
    matrix = _as_matrix(columns)
    good = np.all(np.isfinite(matrix), axis=1)
    m = matrix[good]
    n_samples, n_features = m.shape
    if n_samples < 2:
        raise ValueError(f"pca needs at least 2 finite rows, got {n_samples}")
    max_components = min(n_samples, n_features)
    if not (1 <= int(n_components) <= max_components):
        raise ValueError(
            f"n_components must be between 1 and {max_components} "
            f"(min(n_samples, n_features)), got {n_components}"
        )
    n_components = int(n_components)

    scaled = scale_columns(m, mode=scale)
    centered = scaled - np.mean(scaled, axis=0)

    # Economy SVD: X = U @ diag(S) @ Vt. Rows of Vt are the principal axes; U @
    # diag(S) are the scores. Equivalent to eigendecomposing the covariance matrix but
    # numerically better-conditioned and needs no separate covariance step.
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    scores = u[:, :n_components] * s[:n_components]

    total_variance = float(np.sum(s**2))
    if total_variance > 0.0:
        explained = (s[:n_components] ** 2) / total_variance
    else:
        explained = np.zeros(n_components, dtype=np.float64)

    return {
        "scores": scores,
        "components": components,
        "explained_variance_ratio": explained,
        "n_samples": n_samples,
        "n_features": n_features,
    }


UMAP_UNAVAILABLE_MESSAGE = (
    "UMAP needs the umap-learn package, which could not be imported. Install it "
    "(pip install umap-learn) to use this embedding, or try PCA instead -- pure numpy, "
    "always available."
)


class UmapUnavailableError(ValueError):
    """umap-learn could not be imported, so no UMAP embedding is possible.

    A :class:`ValueError` subclass, for the same reason as every other
    ``*UnavailableError`` in this codebase: a caller that already handles ValueError as
    the documented failure mode keeps doing so instead of crashing.
    """


def umap_available() -> bool:
    """True if :func:`umap_embed` can run, i.e. the umap-learn package imports.

    Ask this before offering the feature. :func:`umap_embed` raises
    :class:`UmapUnavailableError` when this is False.
    """
    try:
        import umap  # noqa: F401
    except Exception:  # pragma: no cover - only when umap-learn is absent/broken
        return False
    return True


def umap_embed(
    columns: Sequence[Any],
    *,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    scale: str = "zscore",
    seed: int = 0,
) -> Dict[str, Any]:
    """Nonlinear dimensionality reduction via UMAP.

    Args:
        columns: 2+ equal-length 1-D arrays, one per input dimension. Rows with any
            non-finite value in any column are dropped before fitting.
        n_components: Output dimensionality, usually 2.
        n_neighbors: Local neighborhood size -- small values preserve fine local
            structure, large values preserve more of the global layout. Clamped to
            ``[2, n_samples - 1]``, UMAP's own requirement, rather than raising on a
            small dataset.
        min_dist: Minimum distance apart points are allowed in the embedding -- small
            values pack same-neighborhood points tighter (reads more like clustering),
            large values spread them out (reads more like an overall-shape summary).
        scale: One of :data:`SCALE_MODES`, applied per-column first -- see
            :func:`pca`'s docstring for why; UMAP is likewise scale-sensitive.
        seed: Random seed. UMAP's optimization is stochastic, so this is what makes a
            run reproducible.

    Returns:
        Dict with ``embedding`` (``(n_samples, n_components)``), ``n_samples``,
        ``n_features``, ``n_neighbors_used`` (after the clamp above).

    Raises:
        UmapUnavailableError: umap-learn cannot be imported.
        ValueError: Fewer than 2 columns, mismatched column lengths, fewer than 3
            finite rows, or ``n_components`` < 1.
    """
    matrix = _as_matrix(columns)
    good = np.all(np.isfinite(matrix), axis=1)
    m = matrix[good]
    n_samples, n_features = m.shape
    if n_samples < 3:
        raise ValueError(f"umap_embed needs at least 3 finite rows, got {n_samples}")
    if int(n_components) < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")

    try:
        import umap
    except Exception as exc:  # pragma: no cover - only when umap-learn is absent/broken
        raise UmapUnavailableError(UMAP_UNAVAILABLE_MESSAGE) from exc

    effective_neighbors = max(2, min(int(n_neighbors), n_samples - 1))
    scaled = scale_columns(m, mode=scale)
    reducer = umap.UMAP(
        n_components=int(n_components),
        n_neighbors=effective_neighbors,
        min_dist=float(min_dist),
        random_state=int(seed),
    )
    embedding = reducer.fit_transform(scaled)

    return {
        "embedding": np.asarray(embedding, dtype=np.float64),
        "n_samples": n_samples,
        "n_features": n_features,
        "n_neighbors_used": effective_neighbors,
    }
