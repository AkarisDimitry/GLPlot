"""2-D multivariate operations for Math Lab: clustering and 2-D density estimation.

Kept apart from :mod:`glplot.gui.mathops` (already large, and entirely 1-D-signal-oriented:
one ``(x, y)`` curve in, one curve or scalar out) because these functions take **two
independent arrays as a point cloud**, not a signal, and return per-point or per-grid
results rather than a transformed curve. Same contract as ``mathops.py`` otherwise: pure
numpy/scipy, no imgui, no GL -- every function here is unit-testable fully headless.

**Scoped to exactly two columns by design, not as a v1 shortcut.** Math Lab's whole spine
is "resolved ``(x, y)`` pair -> 2-D preview", and every existing tab already lives inside
that limit. N-dimensional clustering needs a different source model (arbitrary column
selection from a dataset) and a projection to even draw a 2-D preview of it (e.g. PCA with
an explained-variance readout) -- a structurally different feature, not a bigger version of
this one. A Cluster tab built on this module clusters the two columns already plotted
against each other; reducing more variables to two first is the user's job.

**Accelerator pattern, with documented exceptions.** ``scipy.cluster`` is used when
importable (k-means++ seeding via ``scipy.cluster.vq.kmeans2``, several random restarts
scored by inertia) with a pure-numpy Lloyd's-algorithm fallback that is honest, not
degraded -- 2-D k-means is cheap enough to hand-roll exactly, so there is no accuracy
tradeoff in the fallback, only a speed one. Three functions are **hard** scipy
dependencies instead, each because there is no honest from-scratch numpy substitute for
what they compute -- exactly the same reasoning as ``mathops.fit_model``/
``mathops.baseline_asls``, and no real cost given scipy is a required dependency of this
package (``pyproject.toml``, not merely an optional accelerator someone might be missing):

* :func:`density2d`'s ``mode="kde"`` (kernel density estimation) -- ``mode="histogram"``
  (plain numpy) is always available as the fallback path a caller can offer instead.
* :func:`hierarchical_cluster` (agglomerative clustering via
  ``scipy.cluster.hierarchy``) -- :func:`kmeans_cluster` is always available instead.
* :func:`spatial_stats` (nearest-neighbour distances and a convex hull via
  ``scipy.spatial``) -- has no partial/degraded fallback to offer at all.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "kmeans_cluster",
    "hierarchical_cluster",
    "hierarchical_available",
    "HierarchicalUnavailableError",
    "HIERARCHICAL_UNAVAILABLE_MESSAGE",
    "HIERARCHICAL_METHODS",
    "nearest_centroid",
    "cluster_inertia",
    "silhouette_score",
    "density2d",
    "kde2d_available",
    "KdeUnavailableError",
    "KDE_UNAVAILABLE_MESSAGE",
    "spatial_stats",
    "spatial_available",
    "SpatialUnavailableError",
    "SPATIAL_UNAVAILABLE_MESSAGE",
]


def _as_1d(a: Any, name: str) -> np.ndarray:
    """Coerce ``a`` to a non-empty 1-D float64 array or raise ValueError.

    Deliberately not imported from :mod:`mathops` -- ``_as_1d``/``_as_xy`` are that
    module's private helpers, and each module here owns its own (mirrors
    ``mathops._as_1d`` exactly, the same convention ``panels/mathlab.py`` already follows
    for ``mathops._sorted_xy``).
    """
    try:
        arr = np.asarray(a, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric array-like ({exc})") from exc
    if arr.ndim > 1:
        arr = np.squeeze(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got array with shape {np.shape(a)}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def _as_xy(x: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Validate an (x, y) pair: both 1-D, non-empty, and equal length."""
    xa = _as_1d(x, "x")
    ya = _as_1d(y, "y")
    if xa.size != ya.size:
        raise ValueError(f"x and y must have the same length (got {xa.size} and {ya.size})")
    return xa, ya


def kmeans_cluster(
    x: Any,
    y: Any,
    k: int,
    *,
    seed: int = 0,
    n_init: int = 10,
    max_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K-means clustering of the 2-D point cloud ``(x, y)`` into ``k`` clusters.

    Args:
        x: First coordinate, 1-D.
        y: Second coordinate, 1-D, same length as ``x``.
        k: Number of clusters. Must be at least 1 and at most the number of finite points.
        seed: Seeds every random restart, for reproducible results across runs.
        n_init: Number of random restarts; the lowest-inertia (tightest) clustering wins.
            Guards against k-means landing in a bad local minimum from an unlucky start.
        max_iter: Iteration cap per restart, used only by the pure-numpy fallback (scipy's
            ``kmeans2`` manages its own internal convergence and ignores this).

    Returns:
        ``(labels, centroid_x, centroid_y)``. ``labels`` has length ``len(x)``: a
        non-finite ``(x, y)`` row gets label ``-1`` ("unclustered") rather than corrupting
        a real cluster's centroid. ``centroid_x``/``centroid_y`` normally have length
        ``k``, but fewer if an unlucky seed left a cluster with zero members across every
        restart -- labels are always dense and contiguous over whatever clusters survive
        (``0 .. len(centroid_x) - 1``, plus ``-1`` for unclustered rows).

    Raises:
        ValueError: On empty/mismatched input, ``k < 1``, or ``k`` greater than the number
            of finite points.
    """
    xa, ya = _as_xy(x, y)
    good = np.isfinite(xa) & np.isfinite(ya)
    n_good = int(np.count_nonzero(good))

    try:
        kk = int(k)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"k must be an integer (got {k!r})") from exc
    if kk < 1:
        raise ValueError(f"k must be >= 1 (got {kk})")
    if n_good == 0:
        raise ValueError("kmeans_cluster needs at least one finite (x, y) pair")
    if kk > n_good:
        raise ValueError(f"k ({kk}) cannot exceed the number of finite points ({n_good})")

    points = np.column_stack([xa[good], ya[good]]).astype(np.float64)
    centroids, assignments = _kmeans(points, kk, seed=seed, n_init=n_init, max_iter=max_iter)

    labels = np.full(xa.shape, -1.0, dtype=np.float64)
    labels[good] = assignments.astype(np.float64)
    return labels, centroids[:, 0].copy(), centroids[:, 1].copy()


def _inertia(points: np.ndarray, centroids: np.ndarray, assignments: np.ndarray) -> float:
    """Sum of squared distances from each point to its assigned centroid."""
    return float(np.sum((points - centroids[assignments]) ** 2))


def _drop_empty_clusters(
    centroids: np.ndarray, assignments: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Relabel densely after dropping any cluster with zero members."""
    used = np.unique(assignments)
    if used.size == centroids.shape[0]:
        return centroids, assignments
    remap = {int(old): new for new, old in enumerate(used.tolist())}
    new_assignments = np.array([remap[int(a)] for a in assignments.tolist()], dtype=np.int64)
    return centroids[used], new_assignments


def nearest_centroid(x: Any, y: Any, centroid_x: Any, centroid_y: Any) -> np.ndarray:
    """Assign each ``(x, y)`` row to its nearest centroid by Euclidean distance.

    The out-of-sample scoring rule Math Lab's Cluster tab uses to place held-out
    train/val/test rows against centroids fit on the train partition only, and (a
    later phase) the cross-dataset transfer mechanism for both k-means and
    hierarchical clustering -- hierarchical has no native out-of-sample rule at all
    (its labels come from cutting a merge tree, not from a distance to a centroid), so
    nearest-centroid against its post-hoc centroids (each cluster's point mean, already
    what :func:`hierarchical_cluster` returns) is the accepted approximation for it too.

    At convergence, this reproduces :func:`kmeans_cluster`'s own training-time
    assignment exactly: Lloyd's algorithm (and scipy's ``kmeans2``) only stops once
    every point's assignment already equals its nearest returned centroid, which is
    precisely what this function (re-)computes.

    Args:
        x: First coordinate, 1-D.
        y: Second coordinate, 1-D, same length as ``x``.
        centroid_x: Centroid first coordinates, 1-D, non-empty.
        centroid_y: Centroid second coordinates, 1-D, same length as ``centroid_x``.

    Returns:
        ``labels``, length ``len(x)``: the 0-based index into ``centroid_x``/
        ``centroid_y`` of each row's nearest centroid, or ``-1`` for a non-finite
        ``(x, y)`` row -- the same convention :func:`kmeans_cluster` and
        :func:`hierarchical_cluster` already use.

    Raises:
        ValueError: On empty/mismatched input, or no centroids given.
    """
    xa, ya = _as_xy(x, y)
    cx, cy = _as_xy(centroid_x, centroid_y)
    if cx.size == 0:
        raise ValueError("nearest_centroid needs at least one centroid")

    labels = np.full(xa.shape, -1.0, dtype=np.float64)
    good = np.isfinite(xa) & np.isfinite(ya)
    if np.any(good):
        points = np.column_stack([xa[good], ya[good]])
        centroids = np.column_stack([cx, cy])
        deltas = points[:, None, :] - centroids[None, :, :]
        dists = np.sum(deltas * deltas, axis=2)
        labels[good] = np.argmin(dists, axis=1).astype(np.float64)
    return labels


def cluster_inertia(x: Any, y: Any, centroid_x: Any, centroid_y: Any, labels: Any) -> float:
    """Sum of squared distances from each assigned point to its own centroid.

    The public counterpart of the private :func:`_inertia` helper above (which scores
    a restart during fitting, on a bare points array): this one takes ``labels``
    explicitly, so a caller can score points that were never part of the fit at all --
    a held-out val/test split assigned via :func:`nearest_centroid`, for instance.

    Args:
        x: First coordinate, 1-D.
        y: Second coordinate, 1-D, same length as ``x``.
        centroid_x: Centroid first coordinates, 1-D.
        centroid_y: Centroid second coordinates, 1-D, same length as ``centroid_x``.
        labels: Per-row centroid index (0-based into ``centroid_x``/``centroid_y``), or
            ``-1`` for a row to exclude -- the convention :func:`kmeans_cluster`,
            :func:`hierarchical_cluster` and :func:`nearest_centroid` all share.

    Returns:
        Sum of squared point-to-centroid distances, over rows with a non-negative
        label only (a non-finite/unassigned row contributes nothing, the same way an
        empty sum is 0.0 rather than nan).

    Raises:
        ValueError: On empty/mismatched input, or a label referencing a centroid index
            outside ``centroid_x``'s range.
    """
    xa, ya = _as_xy(x, y)
    cx, cy = _as_xy(centroid_x, centroid_y)
    lab = np.asarray(labels, dtype=np.float64)
    if lab.shape != xa.shape:
        raise ValueError(
            f"labels must have the same length as x and y (got {lab.shape} vs {xa.shape})"
        )
    good = np.isfinite(xa) & np.isfinite(ya) & np.isfinite(lab) & (lab >= 0.0)
    if not np.any(good):
        return 0.0
    idx = lab[good].astype(np.int64)
    if int(idx.min()) < 0 or int(idx.max()) >= cx.size:
        raise ValueError(
            f"labels reference a centroid index outside [0, {cx.size}) "
            f"(got range [{int(idx.min())}, {int(idx.max())}])"
        )
    points = np.column_stack([xa[good], ya[good]])
    centroids = np.column_stack([cx, cy])
    return float(np.sum((points - centroids[idx]) ** 2))


def silhouette_score(x: Any, y: Any, labels: Any) -> float:
    """Mean silhouette coefficient of a labelled 2-D point cloud.

    For each point ``i`` in cluster ``C``: ``a(i)`` is its mean distance to the other
    members of ``C``; ``b(i)`` is the smallest, over every OTHER cluster, of its mean
    distance to that cluster's members; the silhouette coefficient is
    ``(b(i) - a(i)) / max(a(i), b(i))`` (in ``[-1, 1]``, higher meaning better
    separated). The return value is the mean over every scored point.

    Pure numpy plus ``scipy.spatial.distance.cdist`` (this project's own scipy is
    already a hard dependency -- see the module docstring; there is no sklearn here to
    reach for).

    Args:
        x: First coordinate, 1-D.
        y: Second coordinate, 1-D, same length as ``x``.
        labels: Per-row cluster label. A negative label (the ``kmeans_cluster``/
            ``nearest_centroid`` "unclustered/non-finite" convention) is excluded, same
            as a non-finite ``(x, y)`` row.

    Returns:
        The mean silhouette coefficient, or ``nan`` when fewer than 2 distinct labels
        have at least 2 members each -- ``a(i)`` needs a same-cluster neighbour to
        average over, and ``b(i)`` needs a second cluster to compare against, so the
        metric is simply undefined with less than that, not a degenerate 0 or 1.

    Raises:
        ValueError: On empty/mismatched input.
    """
    from scipy.spatial.distance import cdist

    xa, ya = _as_xy(x, y)
    lab = np.asarray(labels, dtype=np.float64)
    if lab.shape != xa.shape:
        raise ValueError(
            f"labels must have the same length as x and y (got {lab.shape} vs {xa.shape})"
        )
    good = np.isfinite(xa) & np.isfinite(ya) & np.isfinite(lab) & (lab >= 0.0)
    if not np.any(good):
        return float("nan")

    points_all = np.column_stack([xa[good], ya[good]])
    labels_all = lab[good].astype(np.int64)
    unique, counts = np.unique(labels_all, return_counts=True)
    valid = unique[counts >= 2]
    if valid.size < 2:
        return float("nan")

    keep = np.isin(labels_all, valid)
    points = points_all[keep]
    cluster_of = labels_all[keep]
    dist = cdist(points, points)
    n = points.shape[0]
    a = np.zeros(n, dtype=np.float64)
    b = np.full(n, np.inf, dtype=np.float64)

    for lbl in valid:
        members = cluster_of == lbl
        idx = np.where(members)[0]
        # a(i): mean distance to every OTHER point in the same cluster (exclude self,
        # whose distance is 0 -- dividing by idx.size - 1, safe since valid clusters
        # have at least 2 members).
        a[idx] = dist[np.ix_(idx, idx)].sum(axis=1) / (idx.size - 1)
        others = np.where(~members)[0]
        if others.size:
            mean_to_this_cluster = dist[np.ix_(others, idx)].mean(axis=1)
            b[others] = np.minimum(b[others], mean_to_this_cluster)

    denom = np.maximum(a, b)
    silhouette = np.where(denom > 0.0, (b - a) / denom, 0.0)
    return float(np.mean(silhouette))


def _kmeans(
    points: np.ndarray, k: int, *, seed: int, n_init: int, max_iter: int
) -> Tuple[np.ndarray, np.ndarray]:
    """``(centroids, assignments)`` from the best of ``n_init`` restarts, by inertia."""
    try:
        from scipy.cluster.vq import kmeans2
    except Exception:  # pragma: no cover - only on a broken/absent scipy
        return _kmeans_numpy(points, k, seed=seed, n_init=n_init, max_iter=max_iter)

    rng = np.random.default_rng(seed)
    best_centroids = None
    best_assignments = None
    best_inertia = np.inf
    with warnings.catch_warnings():
        # kmeans2's default missing="warn" reports an emptied cluster on stderr; a bad
        # restart is simply scored worse and discarded below, so the warning is noise.
        warnings.simplefilter("ignore")
        for _ in range(max(1, n_init)):
            restart_seed = int(rng.integers(0, 2**31 - 1))
            try:
                centroids, assignments = kmeans2(points, k, minit="++", seed=restart_seed)
            except Exception:
                continue
            assignments = np.asarray(assignments, dtype=np.int64)
            if np.unique(assignments).size == 0:
                continue
            inertia = _inertia(points, np.asarray(centroids, dtype=np.float64), assignments)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = np.asarray(centroids, dtype=np.float64)
                best_assignments = assignments

    if best_centroids is None:
        return _kmeans_numpy(points, k, seed=seed, n_init=n_init, max_iter=max_iter)
    return _drop_empty_clusters(best_centroids, best_assignments)


def _kmeans_plusplus_init(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: spread initial centroids with probability ~ squared distance."""
    n = points.shape[0]
    centroids = np.empty((k, points.shape[1]), dtype=np.float64)
    centroids[0] = points[int(rng.integers(0, n))]
    closest_sq = np.sum((points - centroids[0]) ** 2, axis=1)
    for i in range(1, k):
        total = float(closest_sq.sum())
        if total <= 0.0:
            # Every remaining point coincides with an already-chosen centroid.
            idx = int(rng.integers(0, n))
        else:
            probs = closest_sq / total
            idx = int(rng.choice(n, p=probs))
        centroids[i] = points[idx]
        dist_to_new = np.sum((points - centroids[i]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, dist_to_new)
    return centroids


def _kmeans_numpy(
    points: np.ndarray, k: int, *, seed: int, n_init: int, max_iter: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm, k-means++ seeded, best of ``n_init`` restarts by inertia.

    The honest fallback for :func:`kmeans_cluster` when scipy is unavailable: 2-D k-means
    on a modest point count is cheap enough that hand-rolling it loses nothing but speed.
    """
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    best_centroids = None
    best_assignments = None
    best_inertia = np.inf

    for _ in range(max(1, n_init)):
        centroids = _kmeans_plusplus_init(points, k, rng)
        assignments = np.zeros(n, dtype=np.int64)
        for iteration in range(max(1, max_iter)):
            deltas = points[:, None, :] - centroids[None, :, :]
            dists = np.sum(deltas * deltas, axis=2)
            new_assignments = np.argmin(dists, axis=1)
            converged = iteration > 0 and np.array_equal(new_assignments, assignments)
            assignments = new_assignments
            if converged:
                break
            for c in range(k):
                mask = assignments == c
                if np.any(mask):
                    centroids[c] = points[mask].mean(axis=0)

        inertia = _inertia(points, centroids, assignments)
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids.copy()
            best_assignments = assignments.copy()

    return _drop_empty_clusters(best_centroids, best_assignments)


KDE_UNAVAILABLE_MESSAGE = (
    "2-D kernel density estimation needs scipy.stats.gaussian_kde, which could not be "
    "imported. The histogram mode still works without scipy; install or repair scipy "
    "(pip install -U scipy) to use a smooth KDE surface instead."
)


class KdeUnavailableError(ValueError):
    """scipy.stats.gaussian_kde could not be imported, so no 2-D KDE is possible.

    A :class:`ValueError` subclass, for the same reason as ``mathops.FitUnavailableError``:
    every entry point in this module documents ValueError as its failure mode, so a caller
    that already renders that as a message keeps doing so instead of crashing.
    """


def kde2d_available() -> bool:
    """True if a 2-D KDE is possible, i.e. ``scipy.stats.gaussian_kde`` imports.

    Ask this before offering the feature. :func:`density2d` with ``mode="kde"`` raises
    :class:`KdeUnavailableError` when this is False.
    """
    try:
        from scipy.stats import gaussian_kde  # noqa: F401
    except Exception:  # pragma: no cover - only on a broken/absent scipy
        return False
    return True


def _padded_range(values: np.ndarray, frac: float = 0.05) -> Tuple[float, float]:
    """(lo, hi) padded by ``frac`` of the span, or a fixed pad around a single value."""
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        pad = 0.5 if lo == 0.0 else abs(lo) * 0.05
        return lo - pad, hi + pad
    pad = (hi - lo) * frac
    return lo - pad, hi + pad


def density2d(
    x: Any,
    y: Any,
    *,
    mode: str = "histogram",
    bins: int = 40,
    grid: int = 60,
    bw_method: Any = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2-D density of the point cloud ``(x, y)``: a binned histogram, or a KDE surface.

    Args:
        x: First coordinate, 1-D.
        y: Second coordinate, 1-D, same length as ``x``.
        mode: ``"histogram"`` (``np.histogram2d``, pure numpy, always available) or
            ``"kde"`` (``scipy.stats.gaussian_kde``, a hard dependency -- see
            :func:`kde2d_available`).
        bins: Bin count per axis, ``mode="histogram"`` only.
        grid: Grid resolution per axis, ``mode="kde"`` only.
        bw_method: Bandwidth control, ``mode="kde"`` only. ``None`` (default) uses
            ``gaussian_kde``'s own automatic choice (Scott's rule). A float is a
            **multiplier of that automatic bandwidth** -- 1.0 reproduces it exactly, less
            than 1 is sharper/more detailed, greater than 1 is smoother -- which is *not*
            ``gaussian_kde``'s own ``bw_method`` convention (there, a scalar is the raw
            factor, so ``1.0`` would mean something arbitrary depending on the data's own
            scale). This module's convention is deliberately relative to automatic instead,
            since that is what a "make it smoother/sharper than the default" slider needs.

    Returns:
        ``(x_edges, y_edges, density)``. ``x_edges``/``y_edges`` are 1-D and ascending,
        length ``bins + 1`` (histogram) or ``grid + 1`` (kde) -- the same "bin edges"
        convention ``np.histogram2d`` itself returns, so a preview grid and an eventual
        "New layer" (the existing ``hist2d`` PlotKind, which is built the same way) agree
        on what each cell covers. ``density`` has shape
        ``(x_edges.size - 1, y_edges.size - 1)``: ``mode="histogram"`` is a raw point
        count per cell (not normalised, matching ``np.histogram2d``'s own default);
        ``mode="kde"`` is the estimated probability density evaluated at each cell's
        center.

    Raises:
        KdeUnavailableError: ``mode="kde"`` and ``scipy.stats.gaussian_kde`` cannot be
            imported.
        ValueError: On unusable input, an unknown ``mode``, ``bins``/``grid`` < 1, or (for
            ``mode="kde"``) fewer than 2 finite points (a KDE bandwidth needs a spread to
            estimate).
    """
    xa, ya = _as_xy(x, y)
    good = np.isfinite(xa) & np.isfinite(ya)
    xg, yg = xa[good], ya[good]
    if xg.size == 0:
        raise ValueError("density2d needs at least one finite (x, y) pair")

    if mode == "histogram":
        n_bins = int(bins)
        if n_bins < 1:
            raise ValueError(f"bins must be >= 1 (got {n_bins})")
        counts, x_edges, y_edges = np.histogram2d(xg, yg, bins=n_bins)
        return (
            np.asarray(x_edges, dtype=np.float64),
            np.asarray(y_edges, dtype=np.float64),
            np.asarray(counts, dtype=np.float64),
        )

    if mode == "kde":
        try:
            from scipy.stats import gaussian_kde
        except Exception as exc:  # pragma: no cover - only on a broken/absent scipy
            raise KdeUnavailableError(KDE_UNAVAILABLE_MESSAGE) from exc

        n_grid = int(grid)
        if n_grid < 1:
            raise ValueError(f"grid must be >= 1 (got {n_grid})")
        if xg.size < 2:
            raise ValueError("density2d(mode='kde') needs at least 2 finite (x, y) pairs")

        x_lo, x_hi = _padded_range(xg)
        y_lo, y_hi = _padded_range(yg)
        x_edges = np.linspace(x_lo, x_hi, n_grid + 1)
        y_edges = np.linspace(y_lo, y_hi, n_grid + 1)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

        points = np.vstack([xg, yg])
        try:
            kde = gaussian_kde(points)
            if bw_method is not None:
                # gaussian_kde's own bw_method is an ABSOLUTE factor when scalar, not a
                # multiplier of the automatic choice -- re-fit with the automatic factor
                # (already computed above) scaled by the caller's multiplier instead, so
                # 1.0 means "the automatic bandwidth" as documented.
                kde = gaussian_kde(points, bw_method=float(bw_method) * kde.factor)
        except Exception as exc:
            raise ValueError(f"KDE could not be fit: {exc}") from exc

        mesh_x, mesh_y = np.meshgrid(x_centers, y_centers, indexing="ij")
        with np.errstate(all="ignore"):
            density = kde(np.vstack([mesh_x.ravel(), mesh_y.ravel()])).reshape(mesh_x.shape)
        return x_edges, y_edges, np.asarray(density, dtype=np.float64)

    raise ValueError(f"unknown mode {mode!r}; expected 'histogram' or 'kde'")


#: Linkage methods hierarchical_cluster() offers, forwarded verbatim to
#: scipy.cluster.hierarchy.linkage. "ward" (minimises within-cluster variance) is the
#: default: it tends to produce the same kind of compact, roughly equal-sized clusters
#: k-means does, making it the natural "try a different algorithm on the same data" option.
HIERARCHICAL_METHODS: Tuple[str, ...] = ("ward", "single", "complete", "average")

HIERARCHICAL_UNAVAILABLE_MESSAGE = (
    "Hierarchical clustering needs scipy.cluster.hierarchy, which could not be "
    "imported. K-means clustering still works without scipy; install or repair scipy "
    "(pip install -U scipy) to use hierarchical clustering."
)


class HierarchicalUnavailableError(ValueError):
    """scipy.cluster.hierarchy could not be imported, so no hierarchical clustering is
    possible. A :class:`ValueError` subclass, for the same reason as every other
    ``*UnavailableError`` in this codebase."""


def hierarchical_available() -> bool:
    """True if hierarchical clustering is possible, i.e. ``scipy.cluster.hierarchy``
    imports. Ask this before offering the feature. :func:`hierarchical_cluster` raises
    :class:`HierarchicalUnavailableError` when this is False."""
    try:
        from scipy.cluster.hierarchy import linkage  # noqa: F401
    except Exception:  # pragma: no cover - only on a broken/absent scipy
        return False
    return True


def hierarchical_cluster(
    x: Any, y: Any, k: int, *, method: str = "ward"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Agglomerative hierarchical clustering of the 2-D point cloud ``(x, y)``.

    Builds the full linkage tree bottom-up (each point starts as its own cluster,
    repeatedly merging the closest pair by ``method``) and cuts it to produce exactly
    ``k`` flat clusters -- an entirely different algorithm from :func:`kmeans_cluster`
    (no random restarts, no local optimum to land in differently across seeds), useful as
    a second opinion or when clusters are not the roughly-spherical, similarly-sized
    shape k-means assumes.

    Args:
        x: First coordinate, 1-D.
        y: Second coordinate, 1-D, same length as ``x``.
        k: Number of clusters to cut the tree into. Must be at least 1 and at most the
            number of finite points.
        method: Linkage criterion, one of :data:`HIERARCHICAL_METHODS`.

    Returns:
        ``(labels, centroid_x, centroid_y)`` -- the exact same shape and convention
        :func:`kmeans_cluster` returns (including label ``-1`` for non-finite rows), so a
        caller can treat the two clustering methods interchangeably. Hierarchical
        clustering has no notion of a centroid while it runs (unlike k-means); the
        centroids returned here are each cluster's point mean, computed after the cut.

    Raises:
        HierarchicalUnavailableError: ``scipy.cluster.hierarchy`` cannot be imported.
        ValueError: On empty/mismatched input, an unknown ``method``, ``k < 1``, or ``k``
            greater than the number of finite points.
    """
    xa, ya = _as_xy(x, y)
    good = np.isfinite(xa) & np.isfinite(ya)
    n_good = int(np.count_nonzero(good))

    try:
        kk = int(k)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"k must be an integer (got {k!r})") from exc
    if kk < 1:
        raise ValueError(f"k must be >= 1 (got {kk})")
    if n_good == 0:
        raise ValueError("hierarchical_cluster needs at least one finite (x, y) pair")
    if kk > n_good:
        raise ValueError(f"k ({kk}) cannot exceed the number of finite points ({n_good})")
    if method not in HIERARCHICAL_METHODS:
        raise ValueError(
            f"unknown method {method!r}; choose one of {', '.join(HIERARCHICAL_METHODS)}"
        )

    try:
        from scipy.cluster.hierarchy import fcluster, linkage
    except Exception as exc:  # pragma: no cover - only on a broken/absent scipy
        raise HierarchicalUnavailableError(HIERARCHICAL_UNAVAILABLE_MESSAGE) from exc

    points = np.column_stack([xa[good], ya[good]]).astype(np.float64)
    if points.shape[0] == 1:
        # A single point cannot be linked to anything; it is trivially its own cluster.
        assignments = np.zeros(1, dtype=np.int64)
    else:
        tree = linkage(points, method=method)
        # fcluster's labels are 1-based and, under "maxclust", already dense (1..m, no
        # gaps, m <= k) -- _drop_empty_clusters below is then a defensive no-op, not load
        # bearing, but keeps this in lockstep with kmeans_cluster if that ever changes.
        assignments = fcluster(tree, t=kk, criterion="maxclust").astype(np.int64) - 1

    centroids = np.array([points[assignments == c].mean(axis=0) for c in np.unique(assignments)])
    centroids, assignments = _drop_empty_clusters(centroids, assignments)

    labels = np.full(xa.shape, -1.0, dtype=np.float64)
    labels[good] = assignments.astype(np.float64)
    return labels, centroids[:, 0].copy(), centroids[:, 1].copy()


SPATIAL_UNAVAILABLE_MESSAGE = (
    "Spatial statistics need scipy.spatial, which could not be imported. Install or "
    "repair scipy (pip install -U scipy) to compute them."
)


class SpatialUnavailableError(ValueError):
    """scipy.spatial could not be imported, so no spatial statistics are possible. A
    :class:`ValueError` subclass, for the same reason as every other
    ``*UnavailableError`` in this codebase."""


def spatial_available() -> bool:
    """True if spatial statistics are possible, i.e. ``scipy.spatial`` imports. Ask this
    before offering the feature. :func:`spatial_stats` raises
    :class:`SpatialUnavailableError` when this is False."""
    try:
        from scipy.spatial import cKDTree  # noqa: F401
    except Exception:  # pragma: no cover - only on a broken/absent scipy
        return False
    return True


def spatial_stats(
    x: Any, y: Any
) -> Tuple[Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
    """Spatial statistics of a 2-D point cloud: spacing, extent, and its convex hull.

    Args:
        x: First coordinate, 1-D.
        y: Second coordinate, 1-D, same length as ``x``.

    Returns:
        ``(stats, hull_x, hull_y)``.

        ``stats`` has keys ``n`` (finite-pair count), ``bbox_width``/``bbox_height`` (the
        axis-aligned bounding box), and -- when there are at least 2 finite points --
        ``mean_nn_distance``/``median_nn_distance``/``min_nn_distance``/
        ``max_nn_distance`` (each point's distance to its single nearest *other* point,
        reduced across all points: a small mean means points are packed tightly, a small
        min relative to the mean flags near-duplicates or a locally dense clump), and --
        when at least 3 non-collinear points exist -- ``hull_area``, ``hull_perimeter``,
        ``hull_vertex_count``.

        ``hull_x``/``hull_y`` are the convex hull's vertices in order and **closed** (the
        first vertex repeated at the end, ready to draw directly as a closed polygon), or
        both ``None`` when no hull exists (fewer than 3 points, or all points collinear).

    Note:
        Nearest-neighbour queries use ``scipy.spatial.cKDTree`` (O(n log n)); the hull
        uses ``scipy.spatial.ConvexHull`` (Qhull). Both are hard scipy dependencies --
        there is no reasonable from-scratch numpy substitute for either, unlike
        :func:`kmeans_cluster`'s Lloyd's-algorithm fallback.

    Raises:
        SpatialUnavailableError: ``scipy.spatial`` cannot be imported.
        ValueError: On empty/mismatched input, or no finite points at all.
    """
    xa, ya = _as_xy(x, y)
    good = np.isfinite(xa) & np.isfinite(ya)
    xg, yg = xa[good], ya[good]
    n = int(xg.size)
    if n == 0:
        raise ValueError("spatial_stats needs at least one finite (x, y) pair")

    try:
        from scipy.spatial import cKDTree
    except Exception as exc:  # pragma: no cover - only on a broken/absent scipy
        raise SpatialUnavailableError(SPATIAL_UNAVAILABLE_MESSAGE) from exc

    stats: Dict[str, float] = {
        "n": float(n),
        "bbox_width": float(np.max(xg) - np.min(xg)),
        "bbox_height": float(np.max(yg) - np.min(yg)),
    }

    points = np.column_stack([xg, yg])

    if n >= 2:
        tree = cKDTree(points)
        dists, _idx = tree.query(points, k=2)
        nn = dists[:, 1]
        stats["mean_nn_distance"] = float(np.mean(nn))
        stats["median_nn_distance"] = float(np.median(nn))
        stats["min_nn_distance"] = float(np.min(nn))
        stats["max_nn_distance"] = float(np.max(nn))

    hull_x: Optional[np.ndarray] = None
    hull_y: Optional[np.ndarray] = None
    if n >= 3:
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(points)
        except Exception:
            # Degenerate input (all points collinear, or coincident) has no hull; not an
            # error condition, just nothing to report.
            hull = None
        if hull is not None:
            # A 2-D ConvexHull's "volume" is its area and its "area" is its perimeter --
            # scipy's own naming generalises to N dimensions, where those are the right
            # words; in 2-D they read backwards, so this is not a typo.
            stats["hull_area"] = float(hull.volume)
            stats["hull_perimeter"] = float(hull.area)
            stats["hull_vertex_count"] = float(hull.vertices.size)
            verts = points[hull.vertices]
            hull_x = np.append(verts[:, 0], verts[0, 0])
            hull_y = np.append(verts[:, 1], verts[0, 1])

    return stats, hull_x, hull_y
