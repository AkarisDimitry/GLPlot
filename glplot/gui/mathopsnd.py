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
    "scale_columns_stats",
    "apply_scale",
    "pca",
    "pca_transform",
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
        raise ValueError(f"all columns must have the same length, got {[a.size for a in arrays]}")
    if n < 1:
        raise ValueError("columns must not be empty")
    return np.column_stack(arrays)


def scale_columns_stats(matrix: Any, *, mode: str = "zscore") -> Dict[str, np.ndarray]:
    """The per-column statistics :func:`scale_columns` computes internally, returned
    instead of discarded -- the "fit" half of a fit/transform split, so a later call
    (new data, not this call's own ``matrix``) can be scaled the exact same way via
    :func:`apply_scale`. This is what makes an out-of-sample projection (PCA/UMAP
    scoring a held-out val/test split, or transferring a fitted model onto a different
    dataset in a later phase) use the FITTED scale, not a fresh one re-derived from
    whatever new data happens to be in front of it.

    Args:
        matrix: ``(n_samples, n_features)`` array-like -- the data to compute
            statistics FROM (typically the training data).
        mode: One of :data:`SCALE_MODES`.

    Returns:
        ``mode="zscore"``: ``{"mean": (n_features,), "std": (n_features,)}``.
        ``mode="minmax"``: ``{"min": (n_features,), "span": (n_features,)}`` (``span``
        is ``max - min``). ``mode="none"``, or an empty ``matrix``: ``{}`` -- there is
        nothing to fit, matching :func:`scale_columns`'s own pass-through in both cases.

    Raises:
        ValueError: On an unknown ``mode``.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if mode not in SCALE_MODES:
        raise ValueError(f"unknown scale mode {mode!r}; choose one of {', '.join(SCALE_MODES)}")
    if mode == "none" or m.size == 0:
        return {}

    if mode == "zscore":
        mean = np.array([float(np.nanmean(m[:, j])) for j in range(m.shape[1])], dtype=np.float64)
        std = np.array([float(np.nanstd(m[:, j])) for j in range(m.shape[1])], dtype=np.float64)
        return {"mean": mean, "std": std}

    # "minmax"
    lo = np.array([float(np.nanmin(m[:, j])) for j in range(m.shape[1])], dtype=np.float64)
    hi = np.array([float(np.nanmax(m[:, j])) for j in range(m.shape[1])], dtype=np.float64)
    return {"min": lo, "span": hi - lo}


def apply_scale(matrix: Any, stats: Dict[str, np.ndarray], *, mode: str = "zscore") -> np.ndarray:
    """Apply previously fitted :func:`scale_columns_stats` statistics to ``matrix``.

    The "transform" half of the fit/transform split: unlike :func:`scale_columns`,
    which derives its own statistics from whatever it is given, this always uses the
    passed-in ``stats`` -- so new data (a held-out split, a different dataset) is
    scaled onto the SAME axes the original fit used, not its own.

    Args:
        matrix: ``(n_samples, n_features)`` array-like to scale.
        stats: A dict from :func:`scale_columns_stats`, fit against (usually) different
            data than ``matrix`` itself.
        mode: One of :data:`SCALE_MODES`; must match whatever ``stats`` was computed
            with -- this is not re-checked against ``stats``' own shape.

    Returns:
        float64 array, same shape as ``matrix``. Reproduces :func:`scale_columns`
        bit-for-bit when ``stats`` came from ``scale_columns_stats(matrix, mode=mode)``
        (the same ``matrix``, not new data) -- same degenerate-column convention too
        (zero spread scales to 0.0, not a divide-by-zero).

    Raises:
        ValueError: On an unknown ``mode``.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if mode not in SCALE_MODES:
        raise ValueError(f"unknown scale mode {mode!r}; choose one of {', '.join(SCALE_MODES)}")
    if mode == "none" or not stats or m.size == 0:
        return m.copy()

    out = np.zeros_like(m)
    if mode == "zscore":
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.asarray(stats["std"], dtype=np.float64)
        for j in range(m.shape[1]):
            if std[j] > 0.0:
                out[:, j] = (m[:, j] - mean[j]) / std[j]
    else:  # "minmax"
        lo = np.asarray(stats["min"], dtype=np.float64)
        span = np.asarray(stats["span"], dtype=np.float64)
        for j in range(m.shape[1]):
            if span[j] > 0.0:
                out[:, j] = (m[:, j] - lo[j]) / span[j]
    return out


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

    # Thin wrapper: same math as scale_columns_stats() + apply_scale() below, split
    # apart so pca()/umap_embed() (and a later phase's out-of-sample transforms) can
    # keep the fitted statistics instead of this function's own discarding of them.
    stats = scale_columns_stats(m, mode=mode)
    return apply_scale(m, stats, mode=mode)


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
        * ``explained_variance_ratio_full``: ``(min(n_samples, n_features),)`` -- the
          SAME ratio, for every singular value the SVD below already computed, not just
          the ``n_components`` kept -- a scree plot's data. Its first ``n_components``
          entries equal ``explained_variance_ratio`` exactly (same numerator, same
          ``total_variance`` denominator).
        * ``n_samples``, ``n_features``: finite-row / column counts actually used.
        * ``mean_``: ``(n_features,)`` -- the (scaled-space) centering mean, kept so
          :func:`pca_transform` can project NEW data through this exact fit instead of
          re-fitting on it.
        * ``scale_stats_``: this fit's :func:`scale_columns_stats` -- likewise kept for
          :func:`pca_transform`.

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

    scale_stats = scale_columns_stats(m, mode=scale)
    scaled = scale_columns(m, mode=scale)
    mean_ = np.mean(scaled, axis=0)
    centered = scaled - mean_

    # Economy SVD: X = U @ diag(S) @ Vt. Rows of Vt are the principal axes; U @
    # diag(S) are the scores. Equivalent to eigendecomposing the covariance matrix but
    # numerically better-conditioned and needs no separate covariance step.
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    scores = u[:, :n_components] * s[:n_components]

    total_variance = float(np.sum(s**2))
    if total_variance > 0.0:
        explained = (s[:n_components] ** 2) / total_variance
        explained_full = (s**2) / total_variance
    else:
        explained = np.zeros(n_components, dtype=np.float64)
        explained_full = np.zeros(s.shape[0], dtype=np.float64)

    return {
        "scores": scores,
        "components": components,
        "explained_variance_ratio": explained,
        "explained_variance_ratio_full": explained_full,
        "n_samples": n_samples,
        "n_features": n_features,
        "mean_": mean_,
        "scale_stats_": scale_stats,
    }


def pca_transform(
    columns: Sequence[Any],
    *,
    mean_: Any,
    components_: Any,
    scale_stats: Dict[str, np.ndarray],
    scale: str = "zscore",
) -> np.ndarray:
    """Project NEW data through an already-fitted :func:`pca` -- its out-of-sample half.

    Builds the matrix from ``columns`` the same way :func:`pca` itself does (via the
    shared ``_as_matrix``, dropping any row with a non-finite value in any column),
    then applies the FITTED ``scale_stats``/``scale`` (:func:`apply_scale`, never a
    fresh fit on ``columns``) and the FITTED ``mean_``/``components_`` -- so a held-out
    val/test split, or a different dataset entirely in a later phase, lands in exactly
    the coordinate system the original fit defined, not one re-derived from itself.

    Args:
        columns: 2+ equal-length 1-D arrays, one per input dimension -- the same
            columns (by position) :func:`pca` was originally called with, though not
            necessarily the same rows.
        mean_: ``pca()``'s returned ``"mean_"``.
        components_: ``pca()``'s returned ``"components"``.
        scale_stats: ``pca()``'s returned ``"scale_stats_"``.
        scale: The ``scale`` mode ``pca()`` was fit with -- must match, since
            ``scale_stats`` alone does not say which formula it belongs to.

    Returns:
        ``(n_finite_rows, n_components)`` scores. Calling this with the SAME call's own
        ``columns``/``mean_``/``components_``/``scale_stats_`` that produced a
        :func:`pca` result reproduces that result's own ``"scores"`` (up to ordinary
        floating-point round-off from taking a different arithmetic path -- a matrix
        product here, an SVD there -- not a different formula).

    Raises:
        ValueError: Fewer than 2 columns, mismatched column lengths, or no finite rows.
    """
    matrix = _as_matrix(columns)
    good = np.all(np.isfinite(matrix), axis=1)
    m = matrix[good]
    if m.shape[0] == 0:
        raise ValueError("pca_transform needs at least one finite row")

    scaled = apply_scale(m, scale_stats, mode=scale)
    centered = scaled - np.asarray(mean_, dtype=np.float64)
    components = np.asarray(components_, dtype=np.float64)
    return centered @ components.T


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
        ``n_features``, ``n_neighbors_used`` (after the clamp above), ``reducer`` (the
        fitted ``umap.UMAP`` object -- its ``.transform()`` projects NEW data through
        this same fit, the out-of-sample half a later phase's transfer/held-out-scoring
        needs), and ``scale_stats_`` (this fit's :func:`scale_columns_stats`, so new
        data can be scaled onto the same axes via :func:`apply_scale` before being
        handed to ``reducer.transform()``).

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
    scale_stats = scale_columns_stats(m, mode=scale)
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
        "reducer": reducer,
        "scale_stats_": scale_stats,
    }
