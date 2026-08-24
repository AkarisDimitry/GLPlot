"""The Math Lab panel: integrate, smooth, overlay and add noise to a signal.

This is the "plotear la integral de una funcion o de datos, suavizar y sobreponer,
agregar ruido" surface. Every operation tab renders a **live preview** in
:func:`widgets.mini_plot` with the source curve and the result drawn on the same axes
before anything is committed -- that overlay is the "sobreponer".

Structure
---------
A source (a :class:`DataSet` column pair, or the geometry of an existing plotted layer)
feeds a tab bar of pure :mod:`glplot.gui.mathops` operations. Nothing is applied until
the user picks an apply mode and presses Apply, at which point a :class:`Command` is
submitted to the :class:`CommandQueue` (CONTRACT 1.1: a draw callback may never touch
``scene.layers``) and recorded on the :class:`UndoStack`.

Preview caching
---------------
``draw`` runs every frame; ``mathops`` on 1e6 samples does not fit in a 16ms budget
(CONTRACT 1.10). Results are therefore memoised against a key of
``(epoch, source, fingerprint, params)``. The fingerprint is deliberately cheap -- length,
endpoints and the sum of ~64 strided samples -- so it is a *heuristic*: an edit that
leaves all of those invariant will not invalidate the preview. The Recompute button in
the source header exists precisely for that case, and Apply always commits the array the
preview last showed, so what you see is what you get either way.

Background compute
-------------------
Memoisation alone does not save a computation that is genuinely slow the FIRST time
(UMAP, hierarchical clustering, a robust/deconvolution fit, a distribution fit on a
large sample) -- that call still has to run once, and CONTRACT 1.10's 16ms budget has
no slack for it. Against a real, live plot (not the headless test harness --
see :meth:`_async_enabled`), :meth:`_cached_result` therefore runs ``_compute`` on a
:class:`~glplot.gui.background.BackgroundJob` (a daemon thread; see that module for
why a thread and not the ``utils/mpl_process.py`` subprocess pattern) instead of
inline, so a slow operation never blocks a frame. While a job is in flight the tab
shows a spinner and elapsed time instead of its usual preview (:meth:`_draw_pending_row`),
with a Cancel button; multiple tabs may have independent jobs running concurrently,
since switching tabs must not abandon one the user is still waiting on. A job that
finishes while its tab is not the one on screen still raises a toast
(``glplot.gui.notifications``) once it has run long enough to matter, so leaving a
slow tab to go do something else does not mean missing the result.

Ordering
--------
``integrate``/``derivative``/``resample`` return their results in ascending-x order
regardless of the input order. This module reproduces that sort locally (:func:`_sorted_by_x`
mirrors ``mathops._sorted_xy``) so the overlaid original lines up with the result sample
for sample, and so Apply plots the result against the x it was actually computed on.

Text is ASCII only: the default ImGui font is ProggyClean, whose glyph range stops at
U+00FF, so no integral sign and no Greek here (CONTRACT 3).
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from glplot.gui import (
    clipboard,
    dynamics,
    icons,
    layerops,
    mathadvise,
    mathops,
    mathops2d,
    mathopsnd,
    notifications,
    styles,
    theme,
    widgets,
)
from glplot.gui.background import BackgroundJob
from glplot.gui.datasets import Column, DataSet
from glplot.gui.history import Command, snapshot
from glplot.gui.models import TrainedModel
from glplot.gui.panels.base import Panel

try:
    from imgui_bundle import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - exercised only on GL-less systems
    IMGUI_AVAILABLE = False
    imgui = None

logger = logging.getLogger(__name__)

__all__ = ["MathLabPanel"]


# Height of every preview plot, in pixels.
_PREVIEW_HEIGHT = 150.0

# Floor on the scrolling body's height. Pinning Apply to the bottom must never squeeze
# the body to nothing: in a panel too short to hold both, the body keeps this much and
# the panel's own scrollbar takes over.
_MIN_BODY_HEIGHT = 80.0

# Width of the trained-model rail (Phase 3) and the height of each row's compact
# thumbnail. The thumbnail is deliberately smaller than _PREVIEW_HEIGHT: it is a
# list-row glance, not the main preview.
_RAIL_WIDTH = 200.0
_RAIL_THUMB_HEIGHT = 70.0

# The narrowest the main content column may be squeezed to before a side-by-side rail
# stops being worth it. Below this, _draw_source's category/operation tab bars start
# scrolling and controls like the Fit tab's Degree field get clipped to invisibility --
# verified against a live-window screenshot at the panel's real default width (roughly
# 450px before the rail exists at all), which is itself below _RAIL_WIDTH + this floor.
# See draw(): under the floor, the rail draws as a full-width section BELOW the body
# instead of a side column, so the body never loses width it had before the rail existed.
_RAIL_MIN_LEFT_WIDTH = 420.0

# Samples taken by _fingerprint. Small enough to be free at 1e6 rows, large enough that a
# realistic edit moves the sum.
_FINGERPRINT_SAMPLES = 64

#: Sentinel offered alongside a dataset's real column names in the X/Y source pickers:
#: maps the axis to the dataset's row number (``0..N-1``) rather than any column, for
#: tables with no natural "against what" axis (a bare 2-column x,y series with no time
#: or index column). Mirrors ``data_editor.DataEditorPanel._INDEX_OPTION`` -- same
#: concept, same bracketed-sentinel spelling, kept as its own constant here rather than
#: importing the other module's private class attribute across panels.
_INDEX_OPTION = "(index)"

# Geometry attributes a layer may expose, and the default name of each of its columns.
# Order matters: ``ab`` is probed before ``pts`` for the same reason datasets.py does it,
# so a line_family is never mistaken for a scatter.
_LAYER_GEOMETRY: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("ab", ("a", "b")),
    ("pts", ("x", "y")),
    ("vertices", ("x", "y", "z")),
)

# Layer types with no editable tabular form. ``patch`` is named explicitly because it also
# carries a ``vertices`` attribute and would otherwise be caught by the duck-typed probe.
_LAYER_UNSUPPORTED = frozenset({"text", "patch", "semantic_line", "semantic_span"})

# Layer types ``layerops.update_layer_xy`` can rewrite in place. 3D geometry is absent by
# design: its rows are (x, y, z) and a two-column swap would silently drop z.
_REPLACEABLE_LAYER_TYPES = frozenset({"scatter", "polyline", "line_family"})

_INTEGRAL_METHODS = ("trapezoid", "simpson", "rectangle")
_DERIVATIVE_METHODS = ("central", "forward", "savgol")
_SMOOTH_METHODS = ("moving_average", "gaussian", "savgol", "median", "ema")
_NOISE_KINDS = ("gaussian", "uniform", "poisson", "salt_pepper")
_RESAMPLE_KINDS = ("linear", "cubic")
_NORMALIZE_MODES = ("minmax", "zscore", "max_abs", "area")
#: Read from mathops so the tab and the operation can never disagree on the vocabulary.
_FILTER_TYPES = mathops.FILTER_TYPES
#: Baseline-removal modes: the two straight-line detrends plus the curved AsLS estimate.
_BASELINE_KINDS = mathops.DETREND_KINDS + ("asls",)
#: "none" (histogram only) plus every distribution mathops can fit.
_HIST_DIST_OPTIONS = ("none",) + mathops.DISTRIBUTIONS

#: The Dynamics tab's analyses. The signal is read as a dynamical system's time series and
#: reconstructed/characterised by :mod:`glplot.gui.dynamics`.
_DYNAMICS_MODES = ("phase", "return_map", "lyapunov")

#: The kinds the "new layer" apply mode may produce. Read from the registry that builds
#: them rather than copied: a pair hardcoded here is a result the user can compute and
#: then cannot plot as anything but a line or a scatter.
_LAYER_KINDS = layerops.KIND_KEYS

# Fit tab models. "polynomial" is linear least squares on the coefficients and keeps the
# old degree spinner (it is correct, it has callers, and it needs no initial guess); the
# rest are mathops.MODELS driven through scipy curve_fit; "custom" compiles a user-typed
# f(x). Driven from mathops.MODEL_NAMES rather than hardcoded so the registry stays the
# single source of truth.
_FIT_MODELS: Tuple[str, ...] = (
    ("polynomial",) + mathops.MODEL_NAMES + ("gaussians", "lorentzians", "custom")
)

#: Fit models that deconvolve a sum of N peaks, and the shape each one uses. Kept apart
#: from the single-shape MODELS registry because they take a peak count, not fixed params.
_MULTIPEAK_MODELS: Dict[str, str] = {"gaussians": "gaussian", "lorentzians": "lorentzian"}

#: Ceiling on the deconvolution peak count, mirroring mathops.multipeak_model's own cap.
_MAX_FIT_PEAKS = 10

# Default custom model. Chosen because its flat 1.0 start actually converges, so the
# first thing a user sees on the custom tab is a fit and not a failure.
_DEFAULT_FIT_EXPR = "a*exp(-x/b) + c"

#: Confidence-band levels the Fit tab offers, and their numeric value. Kept as a small
#: fixed set (rather than a free slider) because mathops.confidence_band's no-scipy
#: fallback table only covers these; a slider would silently be exact for scipy and
#: approximate for the four values that table happens to include.
_CONFIDENCE_LEVELS: Tuple[str, ...] = ("68%", "90%", "95%", "99%")
_CONFIDENCE_LEVEL_VALUES: Dict[str, float] = {"68%": 0.68, "90%": 0.90, "95%": 0.95, "99%": 0.99}

_APPLY_MODES: Tuple[Tuple[str, str], ...] = (
    ("new_layer", "New layer"),
    ("new_dataset", "New dataset"),
    ("add_column", "Add column"),
    ("replace", "Replace source"),
)

# (state key, tab title, tab body method name)
_TABS: Tuple[Tuple[str, str, str], ...] = (
    ("integral", "Integral", "_tab_integral"),
    ("derivative", "Derivative", "_tab_derivative"),
    ("smooth", "Smooth", "_tab_smooth"),
    ("noise", "Noise", "_tab_noise"),
    ("resample", "Resample", "_tab_resample"),
    ("normalize", "Normalize", "_tab_normalize"),
    ("filter", "Filter", "_tab_filter"),
    ("detrend", "Baseline", "_tab_detrend"),
    ("fit", "Fit", "_tab_fit"),
    ("fft", "FFT", "_tab_fft"),
    ("autocorr", "Autocorr", "_tab_autocorr"),
    ("peaks", "Peaks", "_tab_peaks"),
    ("histogram", "Histogram", "_tab_histogram"),
    ("stats", "Statistics", "_tab_stats"),
    ("correlate", "Correlate", "_tab_correlate"),
    ("dynamics", "Dynamics", "_tab_dynamics"),
    ("cluster", "Cluster", "_tab_cluster"),
    ("density2d", "Density 2D", "_tab_density2d"),
    ("envelope", "Envelope", "_tab_envelope"),
    ("rolling", "Rolling stats", "_tab_rolling"),
    ("spatial", "Spatial", "_tab_spatial"),
    ("compare", "Compare", "_tab_compare"),
    ("pca", "PCA", "_tab_pca"),
    ("umap", "UMAP", "_tab_umap"),
)

#: The tabs, grouped for the top-level category bar. Twenty operations overflow the
#: panel's tab strip at its default width (they scroll out of sight, so a user cannot even
#: see that they exist); grouping them under five headings keeps every operation one
#: glance away. Each key appears in exactly one category (asserted below), and the order
#: here is the presentation order both bars use.
#:
#: "Multivariate" is deliberately its own category, not folded into Statistics: every tab
#: everywhere else in this panel treats the source as one (x, y) *signal* -- one ordered
#: curve in, a transformed curve or a scalar summary out. Cluster, Density 2D and Spatial
#: instead treat it as an unordered 2-D *point cloud*, with a different preview (a
#: scatter or a heatmap, not a line) and a different Apply story (a data-driven colour or
#: the existing hist2d layer kind, not a transformed y). That is a different kind of tab,
#: not a different flavour of statistics.
_CATEGORIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Calculus", ("integral", "derivative", "resample", "normalize")),
    (
        "Signal",
        ("smooth", "filter", "detrend", "noise", "fft", "autocorr", "peaks", "envelope", "rolling"),
    ),
    ("Statistics", ("fit", "histogram", "stats", "correlate", "compare")),
    ("Dynamics", ("dynamics",)),
    ("Multivariate", ("cluster", "density2d", "spatial", "pca", "umap")),
)

#: key -> (title, body method name), so a category can render a tab from its key alone.
_TAB_BY_KEY: Dict[str, Tuple[str, str]] = {key: (title, method) for key, title, method in _TABS}

#: key -> category title, for :meth:`MathLabPanel.show_operation` to jump to the right
#: category before selecting the operation.
_CATEGORY_OF: Dict[str, str] = {key: title for title, keys in _CATEGORIES for key in keys}

#: Tabs whose computation is a "trainable" technique -- one that fits/projects/embeds
#: against however many rows the source has, rather than a pointwise transform whose
#: cost scales the same way regardless. These are the tabs :meth:`MathLabPanel._sampling_controls`
#: applies to: the ones where sub-sampling the input before the technique runs is a
#: meaningful lever (a slow UMAP or hierarchical clustering) rather than a no-op.
_TRAINABLE_TABS = frozenset({"fit", "cluster", "pca", "umap"})

# Every operation must live in exactly one category, or a tab would be unreachable (in no
# category) or drawn twice (in two). Cheap to check once at import; a loud failure here
# beats a silently missing tab.
assert set(_CATEGORY_OF) == {key for key, _t, _m in _TABS}, "every tab needs one category"

#: How long a tab's params must sit unchanged before a background compute REPLACING an
#: already-running one is (re)dispatched. Only applies to supersession (see
#: _cached_result_async): the very first computation on a tab always starts immediately,
#: so this never adds latency to the common "open a tab, see a result" case -- it exists
#: purely to stop a continuous slider drag from piling up abandoned slow jobs (UMAP,
#: hierarchical clustering, ...) that can never be hard-cancelled (see background.py).
_ASYNC_DEBOUNCE_SECONDS = 0.15

#: A completed/failed background job only raises a toast if it ran at least this long --
#: below it, the inline result/error the tab already shows is plenty; a toast for every
#: sub-second computation would be noise, not a notification.
_ASYNC_NOTIFY_THRESHOLD_SECONDS = 1.5
assert sum(len(keys) for _t, keys in _CATEGORIES) == len(_TABS), "a tab is double-categorised"

_DESCRIBE_ROWS: Tuple[Tuple[str, str], ...] = (
    ("n", "Finite samples"),
    ("min", "Minimum"),
    ("max", "Maximum"),
    ("mean", "Mean"),
    ("median", "Median"),
    ("std", "Std deviation"),
    ("var", "Variance"),
    ("sum", "Sum"),
    ("rms", "RMS"),
)

#: The extra rows "Show more" adds to the Statistics tab -- everything
#: mathops.describe_extended() has beyond describe(). Percentile keys must match its
#: default ``percentiles=(5.0, 25.0, 75.0, 95.0)`` exactly, since the key format is
#: ``f"percentile_{p:g}"``.
_DESCRIBE_EXTENDED_ROWS: Tuple[Tuple[str, str], ...] = (
    ("percentile_5", "5th percentile"),
    ("percentile_25", "25th percentile (Q1)"),
    ("percentile_75", "75th percentile (Q3)"),
    ("percentile_95", "95th percentile"),
    ("iqr", "Interquartile range"),
    ("mad", "Median abs deviation"),
    ("skewness", "Skewness"),
    ("kurtosis", "Excess kurtosis"),
    ("nan_count", "Non-finite samples"),
    ("outlier_count", "Outliers (Tukey fence)"),
)


# ----------------------------------------------------------------------------------
# Pure helpers (no imgui)
# ----------------------------------------------------------------------------------


def _fmt(value: float) -> str:
    """Format a float for a stat row: compact, but never lying about magnitude."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0.0 else "-inf"
    magnitude = abs(v)
    if magnitude != 0.0 and (magnitude < 1e-4 or magnitude >= 1e6):
        return f"{v:.4e}"
    return f"{v:.6g}"


def _sorted_by_x(x: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(x, y)`` as float64 ordered by ascending x.

    This deliberately mirrors ``mathops._sorted_xy`` (stable argsort, already-ascending
    input passed through unchanged) so that the abscissae this panel plots and overlays
    are exactly the ones ``mathops.integrate``/``derivative`` computed against. Using a
    different tie-break would misalign the overlay on duplicate x values.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size < 2 or bool(np.all(np.diff(xa) >= 0.0)):
        return xa, ya
    order = np.argsort(xa, kind="stable")
    return xa[order], ya[order]


def _histogram_silhouette(edges: np.ndarray, heights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The staircase outline of a histogram, as an ``(x, y)`` polyline for mini_plot.

    A line renderer cannot draw bars, but the classic histogram *silhouette* -- a riser at
    every bin edge and a flat top across each bar, dropping to zero at both ends -- reads
    as a histogram and, crucially, shares one x grid with a smooth PDF sampled at the same
    points, so the fitted density can be overlaid directly.
    """
    n = int(heights.size)
    x = np.empty(2 * n + 2, dtype=np.float64)
    y = np.empty(2 * n + 2, dtype=np.float64)
    x[0], y[0] = edges[0], 0.0
    x[-1], y[-1] = edges[-1], 0.0
    inner_x = x[1:-1]
    inner_y = y[1:-1]
    inner_x[0::2] = edges[:-1]
    inner_x[1::2] = edges[1:]
    inner_y[0::2] = heights
    inner_y[1::2] = heights
    return x, y


def _fingerprint(*arrays: Optional[np.ndarray]) -> Tuple[Any, ...]:
    """A cheap change-detection key for the preview cache.

    Samples at most ``_FINGERPRINT_SAMPLES`` strided elements per array plus its length
    and endpoints. This is O(1) in the data size, which is the whole point -- hashing a
    1e6-row column every frame would cost more than the operation being previewed. It is
    a heuristic and can miss an edit; see the module docstring.
    """
    parts: List[Any] = []
    for arr in arrays:
        if arr is None:
            parts.append(None)
            continue
        a = np.asarray(arr)
        n = int(a.size)
        parts.append(n)
        if n == 0:
            continue
        step = max(1, n // _FINGERPRINT_SAMPLES)
        with np.errstate(all="ignore"):
            parts.append(float(np.nansum(a[::step])))
        parts.append(float(a[0]))
        parts.append(float(a[-1]))
    return tuple(parts)


def _subsample_indices(n: int, max_rows: int, *, seed: int) -> Optional[np.ndarray]:
    """Ascending-sorted indices sub-sampling ``n`` rows down to at most ``max_rows``.

    None means "use everything, no-op": either ``n`` is already within the cap, or
    ``max_rows`` is non-positive (no cap requested at all). Otherwise ``max_rows``
    indices are drawn uniformly at random without replacement and returned in
    ascending order, so a caller that subsets a *sorted-by-x* array with the result
    does not scramble it back out of order.

    Uses its own ``Generator`` seeded by ``seed`` rather than numpy's global random
    state: the Noise tab already reaches for a seeded draw the same way, and this must
    not disturb it (or be disturbed by it) just because both happened to run this frame.
    """
    if max_rows <= 0 or n <= max_rows:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_rows, replace=False)
    idx.sort()
    return idx


def _split_indices(
    n: int, val_frac: float, test_frac: float, *, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A random (seeded) partition of ``range(n)`` into ``(train_idx, val_idx, test_idx)``.

    ``len(val_idx)`` is ``round(n * val_frac)`` and ``len(test_idx)`` is
    ``round(n * test_frac)`` (each clamped so neither exceeds what remains); every
    other index goes to train. ``val_frac + test_frac`` is clamped to at most 0.9
    (scaling both down proportionally if it would exceed that) so train never goes
    empty on a reasonable input -- a split that starves the very partition the model
    is fit on would defeat the point of having one.

    Returns:
        Three disjoint, ascending-sorted, together-exhaustive index arrays.
    """
    vf = max(0.0, float(val_frac))
    tf = max(0.0, float(test_frac))
    total = vf + tf
    if total > 0.9:
        scale = 0.9 / total
        vf *= scale
        tf *= scale

    rng = np.random.default_rng(seed)
    perm = rng.permutation(int(n))
    n_val = min(int(round(n * vf)), perm.size)
    n_test = min(int(round(n * tf)), perm.size - n_val)

    val_idx = np.sort(perm[:n_val])
    test_idx = np.sort(perm[n_val : n_val + n_test])
    train_idx = np.sort(perm[n_val + n_test :])
    return train_idx, val_idx, test_idx


def _held_out_fit_stats(
    fn: Callable[..., Any],
    fit_params: np.ndarray,
    xa: np.ndarray,
    ya: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    n_params: int,
) -> List[Tuple[str, str]]:
    """ "Train"/"Val"/"Test" R squared + RMSE stat rows for the Fit tab's split.

    Shared by the polynomial and nonlinear branches of ``_compute``'s "fit" kind, so
    both report a split in exactly the same shape. Reuses ``mathops.fit_statistics``
    (the nonlinear branch's own goodness-of-fit helper already) uniformly, evaluating
    ``fn(x, *fit_params)`` at each split's own x. A split with fewer than 2 points
    reports the metric as unavailable text -- R squared and RMSE need at least 2 points
    to mean anything, and a silent nan reads as "computed and undefined" rather than
    "there was nothing to compute this on".
    """
    rows: List[Tuple[str, str]] = []
    for label, idx in (("Train", train_idx), ("Val", val_idx), ("Test", test_idx)):
        if idx.size < 2:
            rows.append((f"{label} R squared", "not enough held-out points"))
            rows.append((f"{label} RMSE", "not enough held-out points"))
            continue
        xv, yv = xa[idx], ya[idx]
        with np.errstate(all="ignore"):
            yv_fit = np.asarray(fn(xv, *fit_params), dtype=np.float64)
        summary = mathops.fit_statistics(yv, yv_fit, n_params)
        rows.append((f"{label} R squared", _fmt(summary["r_squared"])))
        rows.append((f"{label} RMSE", _fmt(summary["rmse"])))
    return rows


def _split_metrics(
    stats: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Bucket a trainable tab's ``result.stats`` rows into ``(train, val, test)``.

    Every split-aware branch of ``_compute`` (see ``_held_out_fit_stats`` and the
    Cluster/PCA/UMAP split blocks) labels a held-out row with a "Train "/"Val "/"Test "
    prefix -- this is the one place that convention is read back, not just written. The
    prefix is stripped in the bucketed label (redundant once the row is already sorted
    into its own bucket). Rows with no such prefix -- the common, split-off case -- all
    describe the one model that was fit, so they go to ``train``.
    """
    train: List[Tuple[str, str]] = []
    val: List[Tuple[str, str]] = []
    test: List[Tuple[str, str]] = []
    for label, value in stats:
        if label.startswith("Train "):
            train.append((label[len("Train ") :], value))
        elif label.startswith("Val "):
            val.append((label[len("Val ") :], value))
        elif label.startswith("Test "):
            test.append((label[len("Test ") :], value))
        else:
            train.append((label, value))
    return train, val, test


def _finite_column_matrix(arrays: List[Any]) -> np.ndarray:
    """Stack equal-length 1-D ``arrays`` into a float64 matrix, keeping finite rows only.

    A tiny local mirror of ``mathopsnd._as_matrix`` + its finite-row filter (that
    helper is private to ``mathopsnd``, and this is the one call site -- the PCA
    branch's inline reconstruction-error computation -- that needs exactly this and
    nothing more).
    """
    m = np.column_stack([np.asarray(a, dtype=np.float64).ravel() for a in arrays])
    good = np.all(np.isfinite(m), axis=1)
    return m[good]


def _elbow_sweep(
    xa: np.ndarray, ya: np.ndarray, *, seed: int, max_k: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """k-means inertia and silhouette at every ``k`` from 2 to ``min(max_k, n - 1)``.

    ``n`` is ``xa.size``. Pure and side-effect free -- run on a :class:`BackgroundJob`
    thread by :meth:`MathLabPanel._draw_elbow_section`, never called from the main
    thread against live data (that is exactly what the k-fold-more-expensive cost this
    diagnostic exists for makes a bad idea).

    Returns:
        ``(k_values, inertias, silhouettes)``, all the same length -- possibly empty
        (all three ``size == 0``) when there are fewer than 3 points, i.e. not even
        ``k=2`` has a meaningful held-out silhouette (``k=2`` on 2 points assigns one
        point per cluster, which ``silhouette_score`` reports as ``nan`` for lack of a
        same-cluster neighbour; 3 points is the smallest input that can produce a real
        number at ``k=2``).
    """
    n = int(np.asarray(xa).size)
    k_max = min(int(max_k), n - 1)
    if k_max < 2:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty
    k_values: List[float] = []
    inertias: List[float] = []
    silhouettes: List[float] = []
    for k in range(2, k_max + 1):
        labels, cx, cy = mathops2d.kmeans_cluster(xa, ya, k, seed=seed)
        inertias.append(mathops2d.cluster_inertia(xa, ya, cx, cy, labels))
        silhouettes.append(mathops2d.silhouette_score(xa, ya, labels))
        k_values.append(float(k))
    return (
        np.asarray(k_values, dtype=np.float64),
        np.asarray(inertias, dtype=np.float64),
        np.asarray(silhouettes, dtype=np.float64),
    )


def _fit_state_key(model: str, spec: mathops.FitModel) -> str:
    """Identity of a fit model's parameter vector: the model name plus its parameter names.

    The Fit tab remembers an automatic and a manual guess per model. Keying those on the
    model name alone would carry one custom expression's values onto a different
    expression with the same number of parameters, which is a silently wrong start rather
    than a visible one.
    """
    return f"{model}|{','.join(spec.param_names)}"


def _r_squared(y: np.ndarray, y_fit: np.ndarray) -> float:
    """Coefficient of determination, computed over the finite pairs only."""
    good = np.isfinite(y) & np.isfinite(y_fit)
    if not bool(np.any(good)):
        return float("nan")
    observed = y[good]
    predicted = y_fit[good]
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - float(np.mean(observed))) ** 2))
    if ss_tot <= 0.0:
        # A constant signal has no variance to explain: a fit that reproduces it exactly
        # is perfect, anything else explains none of it. 0/0 would be nan, which is worse.
        return 1.0 if ss_res <= 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def _layer_table(layer: Any) -> Optional[Tuple[str, List[str], np.ndarray]]:
    """Resolve ``layer`` to ``(attr, column_names, array)``, or None if not tabular.

    Read-only and copy-free: the returned array is the layer's live geometry. Callers take
    column *views* of it and only cast to float64 inside the memoised compute step.
    """
    layer_type = getattr(layer, "layer_type", "") or ""
    if layer_type in _LAYER_UNSUPPORTED:
        return None
    for attr, default_names in _LAYER_GEOMETRY:
        array = getattr(layer, attr, None)
        if array is None:
            continue
        arr = np.asarray(array)
        if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 1:
            continue
        names = list(default_names[: arr.shape[1]])
        names.extend(f"c{i + 1}" for i in range(len(names), arr.shape[1]))
        return attr, names, arr
    return None


def _layer_display(layer: Any) -> str:
    """A stable, unique-per-layer combo entry. Labels alone are not unique."""
    label = getattr(layer, "label", None) or getattr(layer, "layer_type", "layer")
    return f"{label} [{getattr(layer, 'layer_id', '?')}]"


# ----------------------------------------------------------------------------------
# Value objects
# ----------------------------------------------------------------------------------


@dataclass(eq=False)
class _Source:
    """The resolved input of an operation.

    ``eq=False``: a generated ``__eq__`` would compare ndarrays and raise.

    ``x_raw``/``y_raw`` may be float32 strided views into live layer geometry. They are
    never mutated here and are cast to float64 by ``mathops`` at compute time.
    """

    key: Tuple[Any, ...]
    label: str
    x_name: str
    y_name: str
    x_raw: np.ndarray
    y_raw: np.ndarray
    dataset: Optional[DataSet] = None
    x_col: str = ""
    y_col: str = ""
    layer: Optional[Any] = None
    layer_attr: str = ""


@dataclass(eq=False)
class _Result:
    """The output of an operation, plus everything the preview and Apply need."""

    x: np.ndarray
    y: np.ndarray
    x_name: str
    y_name: str
    suffix: str
    plot_label: str
    overlay: Optional[np.ndarray] = None
    overlay_label: str = "source"
    x_changed: bool = False
    stats: List[Tuple[str, str]] = field(default_factory=list)
    allow_apply: bool = True
    allow_replace: bool = True
    #: When set, the preview draws THIS curve as the main line instead of ``x``/``y``.
    #: Peaks needs it: Apply commits the peak coordinates (``x``/``y``), but the preview
    #: must show the whole signal with the peaks marked on it.
    base_x: Optional[np.ndarray] = None
    base_y: Optional[np.ndarray] = None
    #: Points drawn as dots on top of the preview line (the detected peaks, by default).
    markers: Optional[Tuple[np.ndarray, np.ndarray]] = None
    #: Legend/count label for ``markers`` -- "peaks" fits every existing caller (Peaks,
    #: Autocorr, Dynamics); Fit's output-grid mode overrides it to "data" since its
    #: markers are the original samples, not detected peaks.
    marker_label: str = "peaks"
    #: Named columns for a multi-column "new dataset" Apply (the full peak table). When
    #: set, it -- not ``x``/``y`` -- is what New dataset writes; None keeps the 2-column
    #: (x, y) default every other operation uses.
    table: Optional[List[Tuple[str, np.ndarray]]] = None
    #: (lower, upper) confidence band, same x as ``y``/``base_y``, drawn as a translucent
    #: fill behind the line. Only the Fit tab sets this; every other operation leaves it
    #: None and the preview simply omits the band.
    band: Optional[Tuple[np.ndarray, np.ndarray]] = None
    #: Per-point category labels (e.g. a cluster id), same length as ``x``/``y``. When
    #: set, the preview draws an unordered scatter coloured by category
    #: (``widgets.mini_scatter``) instead of ``mini_plot``'s connected line, "New layer"
    #: makes the points a data-driven colour via GLPlot's existing per-point ``c=``
    #: encoding, and "Add column" writes the labels rather than ``y``. Only the Cluster
    #: tab sets this.
    color_values: Optional[np.ndarray] = None
    #: Colormap the ``color_values`` should be rendered with on a new layer (matches
    #: ``pyplot.scatter(cmap=...)``'s convention). Irrelevant when ``color_values`` is None.
    color_cmap: str = "viridis"
    #: A 2-D density grid (histogram or KDE) for the preview only -- ``x``/``y`` stay the
    #: raw point pass-through so New layer/Add column/Replace keep operating on real data,
    #: not the grid. ``grid_x``/``grid_y`` are bin edges (length ``nx+1``/``ny+1``,
    #: matching ``np.histogram2d``); ``grid_z`` has shape ``(nx, ny)``. Only the
    #: Density 2D tab sets these.
    grid_x: Optional[np.ndarray] = None
    grid_y: Optional[np.ndarray] = None
    grid_z: Optional[np.ndarray] = None
    #: True for an unordered point-cloud result with no per-point category (Spatial):
    #: the preview must still use ``mini_scatter``, not ``mini_plot``'s connected line,
    #: but there is no ``color_values`` to distinguish it by (that condition alone
    #: already routes Cluster through the scatter preview). ``markers`` here are
    #: whatever points the tab wants called out (Spatial's convex hull vertices).
    force_scatter: bool = False
    #: Which of the 4 Apply modes this result offers, in ``_APPLY_MODES`` order. None
    #: (the default) means all 4, as every existing tab already behaves. A result that
    #: has no natural (x, y) curve to plot or column to attach -- one row of summary
    #: statistics -- restricts this to ``("new_dataset",)`` rather than disabling Apply
    #: outright, so its numbers are still one click from becoming data.
    apply_modes: Optional[Tuple[str, ...]] = None
    #: The raw fitted state of a "trainable" technique (see ``_TRAINABLE_TABS``), for
    #: held-out scoring now and Save-as-model/transfer/diagnostics in a later phase.
    #: None for every tab except fit/cluster/pca/umap, which populate it in ``_compute``
    #: with a technique-specific dict:
    #:   Fit: {"popt": ndarray, "pcov": ndarray, "model_key": str, "expr": str,
    #:         "n_peaks": int}
    #:   Cluster: {"method": "kmeans" | "hierarchical", "centroid_x": ndarray,
    #:             "centroid_y": ndarray}
    #:   PCA: {"mean_": ndarray, "components_": ndarray, "scale_stats_": dict,
    #:         "scale": str, "columns": Tuple[str, ...]}
    #:   UMAP: {"reducer": Any, "scale_stats_": dict, "scale": str,
    #:          "columns": Tuple[str, ...]}
    #: When a train/val/test split is on (``training.split``), this reflects the model
    #: fit on the TRAIN partition only, not the full (sub-sampled) working set -- a
    #: later phase reading this to save/transfer the model must get the model that was
    #: actually validated, not one that saw the held-out rows during fitting.
    model_state: Optional[Dict[str, Any]] = None
    #: A generic carrier for a SEPARATE small diagnostic mini-plot (Phase 5), drawn
    #: alongside the main preview/stats rather than through ``_draw_result_preview``'s
    #: own x/y/color_values/grid_z dispatch -- a diagnostic's shape has nothing in
    #: common with any of those. None for every tab except fit/pca, which populate it
    #: in ``_compute`` with a technique-specific dict:
    #:   Fit: {"kind": "residuals", "x": ndarray, "y": ndarray} -- actual minus fitted
    #:        (``ya - evaluated``), evaluated fresh via the fitted popt at ``xa``/``ya``
    #:        (the post-subsample source, NOT ``xa_fit``/``ya_fit`` -- a split still
    #:        scores every row -- and independent of whether the main result used an
    #:        output grid).
    #:   PCA: {"kind": "scree", "ratios": ndarray, "chosen": int} -- the FULL explained-
    #:        variance spectrum (every singular value up to
    #:        ``min(n_samples, n_features)``, not just the ``n_components`` kept) and
    #:        the component count actually requested, for marking on the plot.
    #: Cluster's k-sweep (elbow/silhouette) and UMAP's "color by column" are both
    #: presentation-only features driven from their own dedicated state instead (see
    #: ``_elbow_jobs``/``_umap_color_column``) -- neither is a property of one
    #: particular ``_compute()`` call the way these two are, so neither belongs here.
    diagnostic: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class _SamplingParams:
    """Sub-sampling + train/val/test split controls for a 'trainable' tab (fit/cluster/pca/umap).
    Bundled into one object (not several trailing scalars) so a later phase adding split fields
    never changes the ARITY of the params tuple _compute()'s branches unpack -- only this object's
    own shape, which every existing direct _compute(source, (...)) test call passes as one value."""

    subsample: bool = False
    max_rows: int = 5000
    subsample_seed: int = 0
    # Inert in this phase; a later phase wires these into _compute. Present now so the params-tuple
    # arity for every trainable tab only changes ONCE across both phases.
    split: bool = False
    val_frac: float = 0.15
    test_frac: float = 0.15
    split_seed: int = 0


#: The no-op default: no sub-sampling, no split. Every early-return/error path in the
#: 4 trainable tabs' body methods passes this for their params tuple's trailing slot,
#: since no real source data exists yet at that point to sample from.
_NO_SAMPLING = _SamplingParams()


# ----------------------------------------------------------------------------------
# The panel
# ----------------------------------------------------------------------------------


class MathLabPanel(Panel):
    """Integrate / differentiate / smooth / resample / noise, with a live overlay."""

    title = "Math Lab"
    icon = "integral"
    default_open = False

    def __init__(self, ws: Any) -> None:
        super().__init__(ws)

        # -- source selection --
        self._source_kind = "dataset"
        self._ds_name = ""
        self._ds_x_col = ""
        self._ds_y_col = ""
        #: Remembered ``(x_col, y_col)`` pick per dataset name. Without this, switching
        #: dataset and switching back silently resets the columns to the first two, and
        #: a deliberate pick is indistinguishable from a stale one.
        self._ds_columns: Dict[str, Tuple[str, str]] = {}
        #: The dataset names ``_ds_columns`` was last pruned against.
        self._ds_known: List[str] = []
        self._layer_id: Optional[int] = None
        self._layer_x_col = ""
        self._layer_y_col = ""

        # -- operation parameters --
        self._integral_method = "trapezoid"
        self._integral_initial = 0.0

        self._derivative_order = 1
        self._derivative_method = "central"
        self._derivative_window = 11
        self._derivative_polyorder = 3
        #: dy/dx of a parametric curve (x(t), y(t)) via the chain rule, instead of assuming
        #: y is a function of x. Lets a looping/self-intersecting X-Y curve be differentiated.
        self._derivative_parametric = False

        self._smooth_method = "moving_average"
        self._smooth_window = 11
        self._smooth_sigma = 0.0  # 0 => let mathops default it to window / 6
        self._smooth_polyorder = 3
        self._smooth_alpha = 0.0  # 0 => let mathops default it to 2 / (window + 1)

        self._rolling_stat = "mean"
        self._rolling_window = 11
        self._rolling_center = True

        self._noise_kind = "gaussian"
        self._noise_amplitude = 0.1
        self._noise_relative = True
        self._noise_use_seed = True
        self._noise_seed = 0
        self._noise_nonce = 0

        self._resample_n = 512
        self._resample_kind = "linear"

        self._normalize_mode = "minmax"

        # Cutoffs are stored as a fraction of the Nyquist rate (0, 1), not an absolute
        # frequency: that is unit-free, so the slider needs no per-dataset rescaling, and
        # _compute reports the real cutoff and Nyquist frequency in the stats so the user
        # still sees the value in the data's own units.
        self._filter_type = "lowpass"
        self._filter_order = 4
        self._filter_cutoff = 0.25
        self._filter_cutoff_hi = 0.40
        #: IIR filter family: butter (default, no ripple) | cheby1 | cheby2 | bessel.
        self._filter_family = "butter"
        self._filter_ripple = 1.0  # cheby1 passband ripple, dB
        self._filter_attenuation = 40.0  # cheby2 stopband attenuation, dB

        self._detrend_kind = "linear"  # constant | linear | asls
        # AsLS controls: smoothness as log10(lambda) so a slider spans decades, and the
        # asymmetry p. 1e6 keeps the baseline below moderately sharp peaks by default.
        self._baseline_lam_log10 = 6.0
        self._baseline_p = 0.01

        # Prominence as a fraction of the signal's peak-to-peak range, for the same reason:
        # 0.05 means "at least 5% of the full swing", meaningful at any scale.
        self._peak_mode = "peaks"  # peaks | valleys
        self._peak_prominence = 0.05
        self._peak_distance_on = False
        self._peak_distance = 1

        # Max lag as a fraction of the signal length: the ACF is noisiest at long lags
        # (fewer overlapping samples), so half the record is a sensible default ceiling.
        self._autocorr_maxlag = 0.5
        self._autocorr_mode = "auto"  # auto | cross
        self._autocorr_cross_col = ""
        #: Stashed by _draw_tab_body every frame so a tab body (called with no
        #: arguments) can read the current source's dataset for a column picker.
        self._current_source: Optional[_Source] = None

        self._hist_bins = 50
        self._hist_density = False
        self._hist_dist = "normal"

        #: Statistics tab: percentiles/IQR/MAD/skewness/kurtosis/outliers, off by default
        #: so the common case (the 9-row describe() summary) stays uncluttered.
        self._stats_show_more = False
        #: Statistics tab: confidence interval for the mean, off by default (extra rows
        #: only when asked for, matching "Show more" above).
        self._stats_ci = False
        self._stats_ci_level = "95%"

        # Correlate tab: confidence interval for Pearson r (Fisher z).
        self._correlate_ci = False
        self._correlate_ci_level = "95%"

        # Compare tab: two-sample hypothesis test against another column.
        self._compare_col = ""
        self._compare_method = "welch"
        #: Compare tab: confidence interval for the mean difference. Restricted to
        #: welch/student (see mathops.DIFFERENCE_CI_METHODS): mannwhitney/ks test the
        #: full distribution, not the mean, and have no matching closed-form interval.
        self._compare_ci = False
        self._compare_ci_level = "95%"

        # Dynamics tab. Delay is in samples (the panel cannot know the signal's period, so
        # a sample count is the honest unit); dim is the embedding dimension used for the
        # Lyapunov reconstruction. Defaults draw a legible phase portrait on a finely
        # sampled ODE trajectory without further tuning.
        self._dyn_mode = "phase"
        self._dyn_delay = 10
        self._dyn_dim = 3

        # Cluster tab (Multivariate). k=3 is a reasonable "show me something" default on
        # data with no obvious cluster count; seed is fixed so results are reproducible
        # across frames and reruns until the user asks for a fresh reroll.
        self._cluster_method = "kmeans"  # kmeans | hierarchical
        self._cluster_k = 3
        self._cluster_seed = 0
        self._cluster_linkage = "ward"

        # Cluster tab elbow/silhouette diagnostic (Phase 5, kmeans only -- see
        # _draw_elbow_section). An independent BackgroundJob flow, deliberately NOT
        # routed through _cached_result_async/_async_jobs above: a k-sweep is several
        # fits, not one _Result, and must never auto-run just because the tab's own
        # params look stable (see _draw_elbow_section's own docstring). Keyed by tab
        # key for the same reason every other per-tab dict here is, even though only
        # one trainable tab can be "cluster" at a time.
        self._elbow_jobs: Dict[str, BackgroundJob] = {}
        self._elbow_results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        # Density 2D tab (Multivariate).
        self._density_mode = "histogram"
        self._density_bins = 40
        self._density_grid = 40
        self._density_custom_bw = False
        self._density_bw_factor = 1.0

        # PCA tab (Multivariate). Columns start empty and default to "every column of
        # the dataset" the first time the tab is shown against a real dataset (see
        # _tab_pca) -- an empty tuple here is "not yet initialised", not "none picked".
        self._pca_columns: Tuple[str, ...] = ()
        self._pca_n_components = 2
        self._pca_scale = "zscore"

        # UMAP tab (Multivariate). Same column-selection convention as PCA above.
        self._umap_columns: Tuple[str, ...] = ()
        self._umap_n_components = 2
        self._umap_n_neighbors = 15
        self._umap_min_dist = 0.1
        self._umap_scale = "zscore"
        self._umap_seed = 0
        #: "Color by" column pick for the UMAP preview (Phase 5), per tab key -- "" is
        #: the default "(none)"/uncolored state. Presentation only: never read by
        #: _compute, never part of model_state/Save-as-model. See _draw_umap_preview.
        self._umap_color_column: Dict[str, str] = {}

        self._fit_model = "polynomial"
        self._fit_degree = 3
        self._fit_npeaks = 2  # peaks to deconvolve when the model is a sum of peaks
        self._fit_expr = _DEFAULT_FIT_EXPR
        self._fit_manual = False
        #: Manual initial guess per _fit_state_key. Only consulted while _fit_manual.
        self._fit_p0: Dict[str, List[float]] = {}
        #: Heuristic initial guess per _fit_state_key, stashed by _compute. The
        #: heuristics read the source data, which a tab body never sees, so this is the
        #: only way the tab can show the automatic start or seed "Manual" from it.
        self._fit_auto: Dict[str, List[float]] = {}
        #: Outlier-robust fit (nonlinear models only): scipy.optimize's robust loss kinds.
        self._fit_robust = False
        self._fit_loss = "soft_l1"
        #: Confidence band: shown on any model (polynomial included), off by default so
        #: the common case -- just the fit line -- stays uncluttered.
        self._fit_band = False
        self._fit_band_level = "95%"
        #: Output grid: off evaluates the fit at the source's own x samples (today's
        #: behaviour, and what Replace source needs); on evaluates it at N evenly spaced
        #: points across the x range instead -- a smooth curve independent of how many
        #: samples the input has, which is the whole point for a sparse or unevenly
        #: sampled fit. This is the "how many points to add" control Apply needed.
        self._fit_output_grid = False
        self._fit_output_points = 200

        # -- sampling (fit/cluster/pca/umap only; see _sampling_controls/_TRAINABLE_TABS) --
        # Per-tab dicts, same idiom as _apply_modes/_apply_names below: Fit's "Sub-sample"
        # state is about the fit, carrying it onto Cluster (a single panel-global flag
        # would) makes an unrelated tab silently start dropping rows.
        self._sampling_enabled: Dict[str, bool] = {}
        self._sampling_max_rows: Dict[str, int] = {}
        self._sampling_seed: Dict[str, int] = {}
        #: Train/val/test split, same per-tab-dict idiom -- drawn inside the "Sampling"
        #: section by _sampling_controls, independent of the "Sub-sample" checkbox above
        #: (a split is meaningful whether or not the source was also sub-sampled first).
        self._split_enabled: Dict[str, bool] = {}
        self._split_val_frac: Dict[str, float] = {}
        self._split_test_frac: Dict[str, float] = {}
        self._split_seed: Dict[str, int] = {}

        # -- apply --
        # Per-tab, keyed by the _TABS key. A name typed on Smooth is about the smoothed
        # curve; carrying it onto Fit (as a single panel-global field did) silently
        # mislabels the output. Same for the mode: "Replace source" makes sense on
        # Smooth and rarely on FFT.
        self._apply_modes: Dict[str, str] = {}
        self._apply_names: Dict[str, str] = {}
        #: "Save as model" name field, per trainable tab (see ``_TRAINABLE_TABS``) --
        #: same per-tab-dict idiom as ``_apply_names``, so a name typed on Fit never
        #: leaks onto Cluster.
        self._model_names: Dict[str, str] = {}
        self._layer_kind = "line"
        #: The selected kind's parameters, re-defaulted when the kind changes: a bar's
        #: width means nothing to a histogram, which never declared that option.
        self._layer_opts: Dict[str, Any] = layerops.default_kind_options(self._layer_kind)
        #: Last measured height of each tab's Apply block, keyed by tab. Used to reserve
        #: the space the block needs at the bottom of the panel *before* the scrolling
        #: body is opened; see :meth:`_draw_tab_body`.
        self._apply_heights: Dict[str, float] = {}
        self._notice: Optional[str] = None

        # -- preview cache --
        self._cache_epoch = 0
        self._cache_key: Optional[Tuple[Any, ...]] = None
        self._cache_result: Optional[_Result] = None
        self._cache_error: Optional[str] = None

        # -- background compute (see _cached_result_async) -- every dict below is keyed
        # by tab key ("fit", "umap", ...), so multiple tabs' computations can be running
        # concurrently and surviving a tab switch, independent of each other.
        #: In-flight or settled-but-not-yet-collected job per tab.
        self._async_jobs: Dict[str, BackgroundJob] = {}
        #: The cache-key tuple each entry of _async_jobs was started for.
        self._async_job_keys: Dict[str, Tuple[Any, ...]] = {}
        #: monotonic() a tab's params last became stable, for the debounce in
        #: _cached_result_async. Only consulted while superseding an already-running job.
        self._async_stable_since: Dict[str, float] = {}
        #: (result, error, cache_key) of the most recently SETTLED computation per tab --
        #: this, not the single-slot _cache_* trio above, is what survives a tab switch.
        #: _cache_key/_cache_result/_cache_error still get updated to mirror whichever
        #: tab was drawn most recently, so every existing Apply/export/report/test code
        #: path that reads them keeps working unchanged in both sync and async mode.
        self._async_results: Dict[str, Tuple[Optional[_Result], Optional[str], Tuple[Any, ...]]] = (
            {}
        )
        #: Test-only escape hatch: force async mode on even under the headless harness's
        #: fake plot (which has no ._is_test_mode to read). See _async_enabled().
        self._force_async = False

        #: Operation key requested by :meth:`show_operation`; consumed by the next frame's
        #: tab bar. A pending value cannot be applied immediately -- the tab bar is only
        #: submitted from ``draw``.
        self._pending_tab: Optional[str] = None
        #: Category the pending tab lives in; the top-level bar must switch to it first,
        #: or the operation tab would not be among those drawn and could never be selected.
        self._pending_category: Optional[str] = None

        #: Frames left to show the "Copied" confirmation, and the tab it belongs to, so a
        #: copy on one tab does not flash a confirmation on another.
        self._copied_frames = 0
        self._copied_key = ""

        # -- CSV export -- path field shared by every tab, same convention as the Data
        # Editor's Import/Export section (no native file dialog exists anywhere in this
        # GUI -- CONTRACT-adjacent: pyimgui has none to offer).
        self._csv_path = os.path.join(os.getcwd(), "mathlab_export.csv")
        self._exported_frames = 0
        self._exported_key = ""

        # -- "Suggested next step" -- cached on the source's own key, not the per-frame
        # fingerprint _cached_result uses: a suggestion should react to a new source, not
        # to every keystroke of a parameter on whatever tab happens to be open.
        self._advise_key: Optional[Tuple[Any, ...]] = None
        self._advise_cache: List[mathadvise.Recommendation] = []

        # -- trained-model rail (Phase 3) -- closed by default; see draw()'s
        # rail_visible computation for why it still shows up once a model is saved.
        self._rail_open: bool = False
        #: Inline-rename state for a rail row, the scene.py idiom keyed by
        #: TrainedModel.name rather than an integer id: a model has no other stable
        #: identity, and ModelStore.add's rename-on-collision guarantees names are
        #: unique at any instant. The name being edited cannot change out from under
        #: this key while a rename is in flight -- the mutation is deferred (submit),
        #: so model.name only actually changes on the next queue drain, after
        #: _commit_rail_rename has already cleared this state.
        self._rail_rename_target: Optional[str] = None
        self._rail_rename_buf: str = ""
        self._rail_focus_rename: bool = False

        # -- "Apply to..." on a rail row (Phase 4) -- every dict below is keyed by
        # model.name, the same per-model idiom the rail's own state above uses, so
        # expanding one model's apply section never disturbs another's in-progress pick.
        #: Whether this model's "Apply to..." section is expanded.
        self._apply_model_open: Dict[str, bool] = {}
        #: The target dataset name picked for this model (independent of the Source
        #: tab's own ``self._ds_name`` -- this is a TRANSFER target, not the tab's source).
        self._apply_model_target_ds: Dict[str, str] = {}
        #: This model's column_map: {role/trained-column-name -> target dataset column}.
        self._apply_model_columns: Dict[str, Dict[str, str]] = {}
        #: The typed output name (column name, or new dataset name).
        self._apply_model_output_name: Dict[str, str] = {}
        #: "add_column" | "new_dataset" -- only these two modes make sense on a TRANSFER
        #: target (Replace/New layer only make sense on the dataset a model was trained on).
        self._apply_model_mode: Dict[str, str] = {}

    # -- public API (the workspace's Math actions call this) ------------------------

    def show_operation(self, key: str) -> None:
        """Select the operation tab ``key`` (an id from ``_TABS``) on the next frame.

        The hook the registry's quick verbs -- Integrate, Derivative, Smooth -- need: an
        action can open Math Lab *on the operation the user asked for* instead of on
        whichever tab was last used. Safe to call from a draw callback and from a headless
        process: it only sets state the next frame reads. An unknown key is ignored rather
        than raising, so a stale caller cannot take the frame down.
        """
        if key in _TAB_BY_KEY:
            self._pending_tab = key
            # Jump the top-level bar to the operation's category too, else the operation
            # tab is not among those drawn this frame and the request is silently lost.
            self._pending_category = _CATEGORY_OF[key]

    # -- lifecycle -----------------------------------------------------------------

    def draw(self) -> None:
        """Panel body. The Workspace owns begin/end.

        Adds a right-hand rail (Phase 3) listing every saved TrainedModel, but only
        when there is something to show: ``rail_visible`` is False for anyone who has
        never opened it AND never saved a model, in which case the body below draws
        exactly as it did before the rail existed -- no child-region wrapping at all.

        Even when the rail is visible, it only becomes a side COLUMN when the panel is
        wide enough to spare ``_RAIL_WIDTH`` without squeezing the body below
        ``_RAIL_MIN_LEFT_WIDTH`` -- a live-window screenshot at the panel's real (much
        narrower) default size showed a fixed side rail clipping the Fit tab's Degree
        field to invisibility and forcing the category/operation tab bars into
        scroll-arrow mode. Below the floor, the rail instead draws as a full-width
        collapsible section BELOW the body, so the body's own width is never worse than
        it was before the rail existed, at any panel size.
        """
        if not IMGUI_AVAILABLE:
            return

        rail_visible = self._rail_open or bool(self.models.models)
        if not rail_visible:
            self._draw_body()
            return

        avail = imgui.get_content_region_avail()
        # No other panel in this codebase reserves a fixed-width side region; item_spacing.x
        # (the same gap ImGui already inserts between two same_line() widgets) is the
        # natural amount to hold back for the same_line() gap below, matching how
        # dynamics.py/functions.py size a reserved region off the current style.
        spacing = imgui.get_style().item_spacing.x
        left_w = avail.x - _RAIL_WIDTH - spacing

        if left_w < _RAIL_MIN_LEFT_WIDTH:
            self._draw_body()
            imgui.separator()
            if widgets.section("Trained models", default_open=True):
                self._draw_model_rail()
            return

        # CONTRACT 2.6: end_child is unconditional -- begin_child returning
        # not-visible still opens a scope that must be closed.
        left_visible = imgui.begin_child("##mathlab_body", (left_w, 0.0))
        try:
            if left_visible:
                self._draw_body()
        finally:
            imgui.end_child()

        imgui.same_line()

        rail_child_visible = imgui.begin_child(
            "##mathlab_rail", (0.0, 0.0), imgui.ChildFlags_.borders
        )
        try:
            if rail_child_visible:
                self._draw_model_rail()
        finally:
            imgui.end_child()

    def _draw_body(self) -> None:
        """Everything the panel drew before the model rail (Phase 3) existed.

        Called identically whether or not the rail is visible this frame -- see
        draw(): only whether this is wrapped in a reserved-width child region differs.
        """
        self._draw_notice()
        self._draw_rail_toggle()
        source, source_error = self._draw_source()
        imgui.separator()

        if source is None:
            widgets.error_box(source_error or "No source selected.")
            return

        self._draw_recommendations(source)
        self._draw_tabs(source)

    def _draw_rail_toggle(self) -> None:
        """Header toggle for the trained-model rail.

        Always drawn, even with zero saved models: this is what lets a user open the
        rail for the first time (discoverability) without changing rail_visible's
        default-closed behaviour for someone who never touches it.
        """
        if icons.icon_button(
            "##toggle_rail",
            "layers",
            size=18.0,
            tooltip="Trained models",
            active=self._rail_open,
        ):
            self._rail_open = not self._rail_open

    def _draw_recommendations(self, source: _Source) -> None:
        """A collapsible "Suggested next step" strip, driven by :mod:`mathadvise`.

        Heuristic, explicitly labeled as such -- this is a nudge toward a tab worth
        trying, not a verdict about the data. A failing estimator is swallowed rather
        than surfaced: a broken suggestion is not worth interrupting the panel over.
        """
        if source.key != self._advise_key:
            self._advise_key = source.key
            try:
                n_columns = len(source.dataset.column_names()) if source.dataset is not None else 0
                self._advise_cache = mathadvise.recommend(
                    source.x_raw,
                    source.y_raw,
                    has_dataset=source.dataset is not None,
                    n_columns=n_columns,
                )
            except Exception:
                self._advise_cache = []

        if not widgets.section("Suggested next step", default_open=False):
            return

        if not self._advise_cache:
            imgui.text_disabled("No strong signal in this data -- browse the tabs below.")
            return

        imgui.text_wrapped(
            "Heuristic hints from a quick look at the data, not a verdict -- a nudge "
            "toward a tab worth trying."
        )
        for rec in self._advise_cache:
            imgui.bullet_text(rec.title)
            imgui.same_line()
            widgets.help_marker(rec.reason)
            imgui.same_line()
            if imgui.small_button(f"Try it##advise_{rec.tab_key}"):
                self.show_operation(rec.tab_key)
        imgui.separator()

    def _draw_notice(self) -> None:
        """Show the last deferred message (e.g. "not undoable"), with a dismiss button."""
        if self._notice is None:
            return
        widgets.error_box(self._notice)
        if icons.icon_button("##dismiss_notice", "close", size=18.0, tooltip="Dismiss"):
            self._notice = None

    # -- source --------------------------------------------------------------------

    def _draw_source(self) -> Tuple[Optional[_Source], Optional[str]]:
        """Draw the source selector and resolve it. Returns ``(source, error)``."""
        imgui.push_id("mathlab_source")
        try:
            imgui.text("Source")
            imgui.same_line()
            widgets.help_marker(
                "Operate on two columns of a dataset, or on the geometry of a layer that "
                "is already plotted. Every operation is previewed over the source before "
                "it is applied."
            )
            imgui.same_line()
            if icons.icon_button(
                "##recompute",
                "refresh",
                size=18.0,
                tooltip="Recompute the preview now (the change detector samples the data, "
                "so it can miss an edit)",
            ):
                self._cache_epoch += 1

            # The radios MUST have their own id scope. ImGui derives a widget id from
            # hash(label) ^ id_stack, so radio_button("Dataset") and the enum_combo
            # labelled "Dataset" below resolve to the SAME id, and the combo's popup --
            # which is keyed on that id -- can never open. That made the source selector
            # completely dead: the first control in the panel did nothing. push_id here
            # is what keeps the two apart; do not remove it, and do not rely on the
            # labels staying distinct.
            imgui.push_id("kind")
            try:
                if imgui.radio_button("Dataset", self._source_kind == "dataset"):
                    self._source_kind = "dataset"
                imgui.same_line()
                if imgui.radio_button("Plotted layer", self._source_kind == "layer"):
                    self._source_kind = "layer"
            finally:
                imgui.pop_id()

            imgui.push_item_width(-140.0)
            try:
                if self._source_kind == "dataset":
                    return self._draw_dataset_source()
                return self._draw_layer_source()
            finally:
                imgui.pop_item_width()
        finally:
            imgui.pop_id()

    def _draw_dataset_source(self) -> Tuple[Optional[_Source], Optional[str]]:
        """Dataset selector plus x/y column combos."""
        names = self.store.names()
        if not names:
            return None, "No datasets yet. Paste data or create one in the Data panel."
        if self._ds_name not in names:
            self._ds_name = names[0]
        _changed, self._ds_name = widgets.enum_combo("Dataset", self._ds_name, names)

        dataset = self.store.get(self._ds_name)
        if dataset is None:
            return None, "Select a dataset."
        columns = dataset.column_names()
        if not columns:
            return None, f"Dataset {dataset.name!r} has no columns."
        if dataset.n_rows() == 0:
            return None, f"Dataset {dataset.name!r} has no rows."

        # Per-dataset column memory: recall this dataset's own pick rather than whatever
        # the previously selected dataset happened to be showing.
        options = [_INDEX_OPTION] + list(columns)
        x_col, y_col = self._ds_columns.get(self._ds_name, ("", ""))
        if x_col not in options:
            x_col = columns[0]
        if y_col not in options:
            y_col = columns[1] if len(columns) > 1 else columns[0]
        _cx, x_col = widgets.enum_combo("X column", x_col, options)
        _cy, y_col = widgets.enum_combo("Y column", y_col, options)
        self._ds_columns[self._ds_name] = (x_col, y_col)
        if self._ds_known != names:
            # The set of datasets changed: forget the ones that are gone, so the map
            # cannot grow without bound across a session of create/delete. Comparing two
            # small string lists is cheaper than rebuilding the dict every frame, and a
            # count-based guard would miss a stale key whenever datasets outnumber it.
            self._ds_columns = {k: v for k, v in self._ds_columns.items() if k in names}
            self._ds_known = list(names)
        self._ds_x_col, self._ds_y_col = x_col, y_col

        x = (
            np.arange(dataset.n_rows(), dtype=np.float64)
            if self._ds_x_col == _INDEX_OPTION
            else dataset.get(self._ds_x_col)
        )
        y = (
            np.arange(dataset.n_rows(), dtype=np.float64)
            if self._ds_y_col == _INDEX_OPTION
            else dataset.get(self._ds_y_col)
        )
        if x is None or y is None:
            return None, "Pick an x and a y column."

        source = _Source(
            key=("dataset", dataset.name, self._ds_x_col, self._ds_y_col),
            label=f"{dataset.name}.{self._ds_y_col}",
            x_name=self._ds_x_col,
            y_name=self._ds_y_col,
            x_raw=x,
            y_raw=y,
            dataset=dataset,
            x_col=self._ds_x_col,
            y_col=self._ds_y_col,
        )
        return source, None

    def _draw_layer_source(self) -> Tuple[Optional[_Source], Optional[str]]:
        """Layer selector plus x/y column combos over the layer's geometry.

        Reading ``plot.scene.layers`` from a draw callback is safe: only *mutation* is
        forbidden (CONTRACT 1.1).
        """
        candidates: List[Tuple[Any, Tuple[str, List[str], np.ndarray]]] = []
        for layer in self.plot.scene.layers:
            table = _layer_table(layer)
            if table is not None:
                candidates.append((layer, table))
        if not candidates:
            return None, "No layer in the scene has tabular x/y data to operate on."

        entries = [_layer_display(layer) for layer, _table in candidates]
        current = ""
        for (layer, _table), entry in zip(candidates, entries):
            if getattr(layer, "layer_id", None) == self._layer_id:
                current = entry
                break
        if not current:
            current = entries[0]
            self._layer_id = getattr(candidates[0][0], "layer_id", None)
        _changed, chosen = widgets.enum_combo("Layer", current, entries)
        index = entries.index(chosen) if chosen in entries else 0
        layer, (attr, columns, array) = candidates[index]
        self._layer_id = getattr(layer, "layer_id", None)

        if self._layer_x_col not in columns:
            self._layer_x_col = columns[0]
        if self._layer_y_col not in columns:
            self._layer_y_col = columns[1]
        _cx, self._layer_x_col = widgets.enum_combo("X column", self._layer_x_col, columns)
        _cy, self._layer_y_col = widgets.enum_combo("Y column", self._layer_y_col, columns)

        # Strided views, not copies: the cast to float64 happens once, inside the cache.
        x_raw = array[:, columns.index(self._layer_x_col)]
        y_raw = array[:, columns.index(self._layer_y_col)]

        source = _Source(
            key=("layer", self._layer_id, attr, self._layer_x_col, self._layer_y_col),
            label=str(getattr(layer, "label", None) or "layer"),
            x_name=self._layer_x_col,
            y_name=self._layer_y_col,
            x_raw=x_raw,
            y_raw=y_raw,
            layer=layer,
            x_col=self._layer_x_col,
            y_col=self._layer_y_col,
            layer_attr=attr,
        )
        return source, None

    # -- tabs ----------------------------------------------------------------------

    def _draw_tabs(self, source: _Source) -> None:
        """Two-level tab bar: a category strip, then that category's operation tabs.

        The categories keep all thirteen operations visible without the tab strip
        overflowing (see ``_CATEGORIES``). The inner operation bar for a category is drawn
        inside that category's tab item, so only one operation bar exists per frame.
        """
        cat_bar_open = imgui.begin_tab_bar("##mathlab_categories")
        if not cat_bar_open:
            return
        try:
            for title, keys in _CATEGORIES:
                pending_here = self._pending_category == title
                flags = imgui.TabItemFlags_.set_selected if pending_here else 0
                selected, _ = imgui.begin_tab_item(title, flags=flags)
                if not selected:
                    continue
                if pending_here:
                    self._pending_category = None
                try:
                    self._draw_operation_tabs(source, keys)
                finally:
                    imgui.end_tab_item()
        finally:
            imgui.end_tab_bar()

    def _draw_operation_tabs(self, source: _Source, keys: Tuple[str, ...]) -> None:
        """The operation tab bar for one category. Each tab draws params, preview, Apply."""
        tab_bar_open = imgui.begin_tab_bar("##mathlab_tabs")
        if not tab_bar_open:
            return
        try:
            for key in keys:
                title, method_name = _TAB_BY_KEY[key]
                flags = imgui.TabItemFlags_.set_selected if self._pending_tab == key else 0
                selected, _ = imgui.begin_tab_item(title, flags=flags)
                if not selected:
                    continue
                if self._pending_tab == key:
                    self._pending_tab = None
                try:
                    imgui.push_id(key)
                    try:
                        self._draw_tab_body(source, key, method_name)
                    finally:
                        imgui.pop_id()
                finally:
                    imgui.end_tab_item()
        finally:
            imgui.end_tab_bar()

    def _draw_tab_body(self, source: _Source, key: str, method_name: str) -> None:
        """Draw one tab: parameters, preview and stats in a scrolling body, then Apply.

        Apply is **pinned to the bottom of the panel** rather than laid out after the
        stats. At the panel's default size it otherwise lands below the fold on 8 of the
        9 tabs (measured: Integral at y=456px in a 423px panel; Fit at y=501px, and it
        moves further down with every coefficient row), so a user could plausibly never
        discover that Apply exists. Everything above it scrolls inside a child region;
        the commit control does not move.

        The reserve is last frame's measured height for *this* tab (tabs differ: the
        Kind combo only exists in "New layer" mode). On the very first frame there is no
        measurement, so an estimate from the current style is used and the layout
        self-corrects on the next frame.
        """
        reserve = self._apply_reserve(key)
        avail = imgui.get_content_region_avail()[1]
        body_height = max(avail - reserve, _MIN_BODY_HEIGHT)

        child_visible = imgui.begin_child("##body", (0.0, body_height))
        result: Optional[_Result] = None
        try:
            if child_visible:
                imgui.spacing()
                # A handful of tabs (Autocorr's cross-correlate mode, Statistics'
                # Compare) need a second column from the *current* dataset to draw
                # their own picker, but tab bodies are called with no arguments (they
                # read only self). Stashing the source here -- freshest every frame --
                # is cheaper than threading it through every tab method's signature.
                self._current_source = source
                body: Callable[[], Tuple[Any, ...]] = getattr(self, method_name)
                imgui.push_item_width(-140.0)
                try:
                    params = body()
                finally:
                    imgui.pop_item_width()

                result, error, status = self._cached_result(source, params, key)
                imgui.spacing()
                if status == "pending":
                    self._draw_pending_row(key)
                elif status == "cancelled":
                    self._draw_cancelled_row(key)
                else:
                    if key == "umap" and result is not None and error is None:
                        # UMAP's own preview (Phase 5): the same scatter, plus a
                        # presentation-only "Color by column" pick -- see
                        # _draw_umap_preview's own docstring.
                        self._draw_umap_preview(source, result, key)
                    else:
                        self._draw_preview(source, result, error)
                    if result is not None:
                        imgui.spacing()
                        self._draw_export_row(result, key)
                    if result is not None and result.stats:
                        imgui.spacing()
                        self._draw_report_header(source, result, key)
                        widgets.stats_table(f"##stats_{key}", ("Statistic", "Value"), result.stats)
                    if result is not None and result.diagnostic is not None:
                        imgui.spacing()
                        self._draw_diagnostic(result)
        finally:
            # CONTRACT 2.2: end_child is unconditional.
            imgui.end_child()

        if result is not None and result.allow_apply:
            imgui.separator()
            before = imgui.get_cursor_pos_y()
            self._draw_apply(source, result, key)
            self._draw_save_model_row(source, result, key)
            self._apply_heights[key] = max(0.0, imgui.get_cursor_pos_y() - before)
        elif child_visible:
            # The body drew and has nothing to commit -- Statistics, or an op that
            # errored -- so give it the whole panel back. Only when the body actually
            # drew: a culled child must not clobber a good measurement with zero.
            self._apply_heights[key] = 0.0

    def _draw_pending_row(self, tab_key: str) -> None:
        """A background job is running (or still debouncing) for this tab: spinner,
        elapsed time and a Cancel button, in place of the normal preview/stats/Apply."""
        job = self._async_jobs.get(tab_key)
        elapsed = job.poll().elapsed if job is not None else 0.0
        title = _TAB_BY_KEY.get(tab_key, (tab_key, ""))[0]
        if widgets.busy_row(f"Computing {title}", elapsed, id_str=f"pending_{tab_key}"):
            self._cancel_async(tab_key)
        imgui.same_line()
        widgets.help_marker(
            "Running in the background so the interface stays responsive. Cancel "
            "stops waiting on it immediately -- the computation itself may keep "
            "running briefly, unobserved, since it cannot be forced to stop mid-call."
        )

    def _draw_cancelled_row(self, tab_key: str) -> None:
        """The user cancelled this tab's in-flight computation -- offer to retry it."""
        imgui.text_disabled("Cancelled.")
        imgui.same_line()
        if imgui.button(f"Retry##cancelled_{tab_key}", (90.0, 0.0)):
            self._retry_async(tab_key)

    def _apply_reserve(self, key: str) -> float:
        """Vertical space to keep free at the panel bottom for ``key``'s Apply block."""
        measured = self._apply_heights.get(key)
        if measured is None:
            # First frame for this tab. Four rows is the "New layer" default: the mode
            # radios, Kind, Name and the Apply button.
            measured = 4.0 * imgui.get_frame_height_with_spacing()
        if measured <= 0.0:
            return 0.0
        # Plus the separator that divides the body from the Apply block.
        return measured + imgui.get_style().item_spacing[1] * 2.0

    def _tab_integral(self) -> Tuple[Any, ...]:
        """Cumulative integral parameters."""
        _c, self._integral_method = widgets.enum_combo(
            "Method",
            self._integral_method,
            _INTEGRAL_METHODS,
            help="trapezoid: exact for straight lines, the safe default. simpson: exact "
            "for parabolas. rectangle: left endpoint, first order, offered for "
            "comparison rather than accuracy.",
        )
        _c, self._integral_initial = widgets.labeled_drag_float(
            "Constant of integration",
            self._integral_initial,
            speed=0.01,
            help="Added to every sample, so the curve starts here instead of at 0.",
        )
        return ("integral", self._integral_method, float(self._integral_initial))

    def _tab_derivative(self) -> Tuple[Any, ...]:
        """Derivative parameters."""
        changed, order = imgui.input_int("Order", int(self._derivative_order))
        if changed:
            self._derivative_order = max(1, min(int(order), 8))
        widgets.help_marker("1 = slope, 2 = curvature. Each order amplifies noise.")

        _c, self._derivative_parametric = imgui.checkbox(
            "Parametric (dy/dx via t)", self._derivative_parametric
        )
        widgets.help_marker(
            "Treat X and Y as a parametric curve (x(t), y(t)) and take dy/dx = (dy/dt)/(dx/dt) "
            "along the sample order. Use this when the curve loops back or crosses itself, so "
            "Y is not a single-valued function of X. Savgol is not available in this mode."
        )

        # Savgol assumes y = f(x) on a uniform x grid; it has no meaning for a parametric
        # curve, so only central/forward are offered there.
        methods = ("central", "forward") if self._derivative_parametric else _DERIVATIVE_METHODS
        if self._derivative_method not in methods:
            self._derivative_method = "central"
        _c, self._derivative_method = widgets.enum_combo(
            "Method",
            self._derivative_method,
            methods,
            help="central: np.gradient, second order and correct on a non-uniform grid. "
            "forward: one-sided. savgol: fits a local polynomial (needs scipy and an "
            "approximately uniform grid; falls back to central otherwise).",
        )
        if self._derivative_method == "savgol":
            changed, window = imgui.input_int("Window", int(self._derivative_window))
            if changed:
                self._derivative_window = max(1, int(window))
            changed, polyorder = imgui.input_int("Poly order", int(self._derivative_polyorder))
            if changed:
                self._derivative_polyorder = max(1, min(int(polyorder), 10))
        return (
            "derivative",
            int(self._derivative_order),
            self._derivative_method,
            int(self._derivative_window),
            int(self._derivative_polyorder),
            bool(self._derivative_parametric),
        )

    def _tab_smooth(self) -> Tuple[Any, ...]:
        """Smoothing parameters."""
        _c, self._smooth_method = widgets.enum_combo(
            "Method",
            self._smooth_method,
            _SMOOTH_METHODS,
            help="moving_average: boxcar, edge-corrected. gaussian: soft kernel. savgol: "
            "preserves peak height and width (needs scipy). median: kills salt-and-pepper "
            "spikes without rounding off edges. ema: exponential moving average -- "
            "causal (only past/current samples), what a live/streaming computation "
            "could actually produce, unlike the other four which are centered.",
        )
        changed, window = imgui.input_int("Window", int(self._smooth_window))
        if changed:
            self._smooth_window = max(1, int(window))
        widgets.help_marker(
            "Clamped to the data length and forced odd. 1 leaves the signal untouched."
        )
        if self._smooth_method == "gaussian":
            _c, self._smooth_sigma = widgets.labeled_drag_float(
                "Sigma",
                self._smooth_sigma,
                speed=0.05,
                vmin=0.0,
                vmax=1e6,
                help="0 means window / 6, i.e. the window spans about +/-3 sigma.",
            )
        if self._smooth_method == "savgol":
            changed, polyorder = imgui.input_int("Poly order", int(self._smooth_polyorder))
            if changed:
                self._smooth_polyorder = max(0, min(int(polyorder), 10))
        if self._smooth_method == "ema":
            _c, self._smooth_alpha = widgets.labeled_slider_float(
                "Alpha",
                self._smooth_alpha,
                vmin=0.0,
                vmax=1.0,
                help="0 means 2 / (window + 1), the standard 'N-period EMA' convention. "
                "Larger alpha weights recent samples more (less smoothing, faster to "
                "react); smaller weights history more (more smoothing, slower to react).",
            )
        return (
            "smooth",
            self._smooth_method,
            int(self._smooth_window),
            float(self._smooth_sigma),
            int(self._smooth_polyorder),
            float(self._smooth_alpha),
        )

    def _tab_noise(self) -> Tuple[Any, ...]:
        """Noise parameters."""
        _c, self._noise_kind = widgets.enum_combo(
            "Kind",
            self._noise_kind,
            _NOISE_KINDS,
            help="gaussian: normal, amplitude is the standard deviation. uniform: flat in "
            "+/-amplitude. poisson: shot noise, amplitude 1 is the true thing. "
            "salt_pepper: amplitude is the fraction of samples replaced by the min/max.",
        )
        _c, self._noise_amplitude = widgets.labeled_drag_float(
            "Amplitude",
            self._noise_amplitude,
            speed=0.005,
            vmin=0.0,
            vmax=1e9,
        )
        if self._noise_kind in ("gaussian", "uniform"):
            _c, self._noise_relative = imgui.checkbox(
                "Relative to data spread", self._noise_relative
            )
            imgui.same_line()
            widgets.help_marker(
                "Scale the amplitude by the signal's own std (gaussian) or peak-to-peak "
                "range (uniform), so 0.1 means 10% whatever the units are. Poisson and "
                "salt_pepper are already data-scaled and ignore this."
            )
        _c, self._noise_use_seed = imgui.checkbox("Fixed seed", self._noise_use_seed)
        if self._noise_use_seed:
            imgui.same_line()
            changed, seed = imgui.input_int("Seed", int(self._noise_seed))
            if changed:
                self._noise_seed = max(0, int(seed))
        imgui.same_line()
        if icons.icon_button("##reroll", "refresh", size=18.0, tooltip="Draw new noise"):
            self._noise_nonce += 1
            if self._noise_use_seed:
                self._noise_seed += 1
        return (
            "noise",
            self._noise_kind,
            float(self._noise_amplitude),
            bool(self._noise_relative),
            int(self._noise_seed) if self._noise_use_seed else None,
            self._noise_nonce,
        )

    def _tab_resample(self) -> Tuple[Any, ...]:
        """Resampling parameters."""
        changed, n = imgui.input_int("Samples", int(self._resample_n))
        if changed:
            self._resample_n = max(2, min(int(n), 10_000_000))
        widgets.help_marker("Evenly spaced over the source's x range; endpoints are exact.")
        _c, self._resample_kind = widgets.enum_combo(
            "Kind",
            self._resample_kind,
            _RESAMPLE_KINDS,
            help="linear: np.interp. cubic: a spline through every knot (needs scipy and "
            "strictly increasing x; falls back to linear otherwise).",
        )
        return ("resample", int(self._resample_n), self._resample_kind)

    def _tab_normalize(self) -> Tuple[Any, ...]:
        """Normalisation parameters."""
        _c, self._normalize_mode = widgets.enum_combo(
            "Mode",
            self._normalize_mode,
            _NORMALIZE_MODES,
            help="minmax: onto [0, 1]. zscore: mean 0, std 1. max_abs: onto [-1, 1], "
            "keeping the sign and any zero crossing. area: unit total absolute area.",
        )
        return ("normalize", self._normalize_mode)

    def _tab_filter(self) -> Tuple[Any, ...]:
        """Butterworth filter parameters. Cutoffs are fractions of the Nyquist rate."""
        if not mathops.filter_available():
            # No pure-numpy filtfilt exists, so there is no fallback to offer: say so and
            # let _compute raise the same message when it runs. Smooth is the alternative.
            widgets.error_box(mathops.FILTER_UNAVAILABLE_MESSAGE)
        _c, self._filter_type = widgets.enum_combo(
            "Type",
            self._filter_type,
            _FILTER_TYPES,
            help="lowpass: keep slow variation, drop fast wiggles. highpass: the reverse. "
            "bandpass: keep a band. bandstop: notch a band out (e.g. mains hum).",
        )
        _c, self._filter_family = widgets.enum_combo(
            "Family",
            self._filter_family,
            mathops.IIR_FAMILIES,
            help="butter (default): maximally flat passband, no ripple anywhere. "
            "cheby1: sharper roll-off than butter at the same order, at the cost of "
            "ripple in the PASSBAND (set below). cheby2: sharper roll-off with a flat "
            "passband like butter, but ripple in the STOPBAND instead (set below). "
            "bessel: gentler roll-off but near-linear phase and minimal ringing -- "
            "best when a pulse/waveform's SHAPE matters more than a sharp cutoff.",
        )
        changed, order = imgui.input_int("Order", int(self._filter_order))
        if changed:
            self._filter_order = max(1, min(int(order), 8))
        widgets.help_marker(
            "Sharper cutoff and more ringing as it rises. The filter is applied forwards "
            "and backwards (zero phase), so a feature is not shifted and the effective "
            "order is doubled."
        )
        if self._filter_family == "cheby1":
            _c, self._filter_ripple = widgets.labeled_slider_float(
                "Passband ripple (dB)",
                self._filter_ripple,
                vmin=0.01,
                vmax=10.0,
                help="How much gain is allowed to wobble inside the passband. Smaller "
                "is flatter (closer to butter) but needs a higher order for the same "
                "roll-off sharpness; larger buys a sharper transition at the same order.",
            )
        elif self._filter_family == "cheby2":
            _c, self._filter_attenuation = widgets.labeled_slider_float(
                "Stopband attenuation (dB)",
                self._filter_attenuation,
                vmin=10.0,
                vmax=100.0,
                help="How strongly the stopband must be suppressed. Larger pushes the "
                "rejected frequencies down further, at the cost of a slower transition "
                "for the same order.",
            )
        is_band = self._filter_type in ("bandpass", "bandstop")
        if is_band:
            _c, self._filter_cutoff = widgets.labeled_slider_float(
                "Low cutoff",
                self._filter_cutoff,
                vmin=0.001,
                vmax=0.999,
                help="Fraction of the Nyquist frequency (half the sample rate). The real "
                "frequency it maps to is shown below the preview.",
            )
            _c, self._filter_cutoff_hi = widgets.labeled_slider_float(
                "High cutoff", self._filter_cutoff_hi, vmin=0.001, vmax=0.999
            )
        else:
            _c, self._filter_cutoff = widgets.labeled_slider_float(
                "Cutoff",
                self._filter_cutoff,
                vmin=0.001,
                vmax=0.999,
                help="Fraction of the Nyquist frequency (half the sample rate). The real "
                "cutoff frequency is shown below the preview.",
            )
        return (
            "filter",
            self._filter_type,
            int(self._filter_order),
            float(self._filter_cutoff),
            float(self._filter_cutoff_hi),
            self._filter_family,
            float(self._filter_ripple),
            float(self._filter_attenuation),
        )

    def _tab_detrend(self) -> Tuple[Any, ...]:
        """Baseline removal: subtract a constant, a line, or a curved AsLS estimate."""
        _c, self._detrend_kind = widgets.enum_combo(
            "Baseline",
            self._detrend_kind,
            _BASELINE_KINDS,
            help="constant: subtract the mean. linear: subtract the least-squares line, "
            "flattening a sloped baseline. asls: a curved baseline that slides under the "
            "peaks -- for a fluorescent or scattering background a straight line cannot "
            "follow. The estimated baseline is drawn over the signal.",
        )
        if self._detrend_kind == "asls":
            if not mathops.baseline_available():
                widgets.error_box(mathops.BASELINE_UNAVAILABLE_MESSAGE)
            _c, self._baseline_lam_log10 = widgets.labeled_slider_float(
                "Smoothness",
                self._baseline_lam_log10,
                vmin=2.0,
                vmax=9.0,
                fmt="1e%.1f",
                help="log10 of lambda. Higher is a stiffer, flatter baseline; lower lets "
                "it bend into the signal. Raise it if the baseline eats into the peaks.",
            )
            _c, self._baseline_p = widgets.labeled_slider_float(
                "Asymmetry",
                self._baseline_p,
                vmin=0.001,
                vmax=0.5,
                fmt="%.3f",
                help="Weight of points above the baseline. Small (~0.01) keeps it below "
                "the peaks; raise it for a signal with dips instead of peaks.",
            )
        return (
            "detrend",
            self._detrend_kind,
            float(self._baseline_lam_log10),
            float(self._baseline_p),
        )

    def _tab_peaks(self) -> Tuple[Any, ...]:
        """Peak/valley-detection parameters. Prominence is a fraction of the signal's range."""
        _c, self._peak_mode = widgets.enum_combo(
            "Mode",
            self._peak_mode,
            ("peaks", "valleys"),
            help="peaks: local maxima. valleys: local minima -- exactly the same "
            "detector run on the negated signal, so every knob below means the same "
            "thing either way.",
        )
        noun = "valleys" if self._peak_mode == "valleys" else "peaks"
        extrema_word = "minima" if self._peak_mode == "valleys" else "maxima"
        imgui.text_disabled(f"Local {extrema_word}, marked on the signal below.")
        imgui.same_line()
        widgets.help_marker(
            f"Prominence is how far a {noun[:-1]} stands above the surrounding baseline, "
            "as a fraction of the signal's full peak-to-peak swing. Raise it to reject "
            "noise wiggles; lower it to catch small real ones. This is the one knob that "
            "matters most."
        )
        _c, self._peak_prominence = widgets.labeled_slider_float(
            "Min prominence",
            self._peak_prominence,
            vmin=0.0,
            vmax=1.0,
            help="Fraction of the peak-to-peak range. 0 accepts every local maximum.",
        )
        _c, self._peak_distance_on = imgui.checkbox("Min separation", self._peak_distance_on)
        if self._peak_distance_on:
            imgui.same_line()
            changed, distance = imgui.input_int("##peak_distance", int(self._peak_distance))
            if changed:
                self._peak_distance = max(1, int(distance))
            imgui.same_line()
            widgets.help_marker(
                "Minimum gap between accepted peaks, in samples. When two are closer, the "
                "shorter is dropped."
            )
        imgui.text_disabled("Apply as a new dataset for the full peak table.")
        imgui.same_line()
        widgets.help_marker(
            "New dataset writes one row per peak: position, height, prominence, width and "
            "area. The area is integrated between each peak's bounding valleys, accurate "
            "for baseline-separated peaks. Overlapping peaks that never return to baseline "
            "need deconvolution instead -- fit them on the Fit tab."
        )
        return (
            "peaks",
            self._peak_mode,
            float(self._peak_prominence),
            int(self._peak_distance) if self._peak_distance_on else None,
        )

    def _tab_histogram(self) -> Tuple[Any, ...]:
        """Histogram parameters, with an optional maximum-likelihood distribution fit."""
        changed, bins = imgui.input_int("Bins", int(self._hist_bins))
        if changed:
            self._hist_bins = max(1, min(int(bins), 10000))
        widgets.help_marker("Equal-width bins over the range of the y values.")
        _c, self._hist_density = imgui.checkbox("Density", self._hist_density)
        imgui.same_line()
        widgets.help_marker(
            "Scale the bars so they integrate to 1 -- a probability density, directly "
            "comparable to the fitted curve. Off counts samples per bin."
        )
        if not mathops.distributions_available():
            widgets.error_box(mathops.DIST_UNAVAILABLE_MESSAGE)
        _c, self._hist_dist = widgets.enum_combo(
            "Fit distribution",
            self._hist_dist,
            _HIST_DIST_OPTIONS,
            help="Overlay a maximum-likelihood fit of this distribution and score it with "
            "a Kolmogorov-Smirnov test (a p-value above ~0.05 means the data is consistent "
            "with it). 'none' shows the histogram alone.",
        )
        return ("histogram", int(self._hist_bins), self._hist_dist, bool(self._hist_density))

    def _fit_spec(self, model: str, expr: str, n_peaks: Optional[int] = None) -> mathops.FitModel:
        """The FitModel for a Fit-tab selection.

        Shared by ``_tab_fit`` and ``_compute`` so the parameter table and the fit can
        never disagree about what is being fitted. ``n_peaks`` defaults to the tab's own
        count; ``_compute`` passes the count captured in the cache key so a recompute uses
        exactly the number the preview was showing.

        Raises:
            ValueError: With a message meant for the user (bad expression, unknown model).
        """
        if model in _MULTIPEAK_MODELS:
            count = self._fit_npeaks if n_peaks is None else int(n_peaks)
            return mathops.multipeak_model(int(count), _MULTIPEAK_MODELS[model])
        if model == "custom":
            return mathops.custom_model(expr)
        return mathops.resolve_model(model)

    def _fit_band_controls(self) -> Tuple[bool, float]:
        """The confidence-band checkbox + level picker. Available on every model."""
        _c, self._fit_band = imgui.checkbox("Show confidence band", self._fit_band)
        imgui.same_line()
        widgets.help_marker(
            "A shaded region around the fit showing where the true curve likely lies, "
            "propagated from the fit's own parameter uncertainty. Wider where the data "
            "is sparser or noisier; not meaningful when a parameter's own standard error "
            "is infinite (an underdetermined fit)."
        )
        if self._fit_band:
            imgui.same_line()
            _c, self._fit_band_level = widgets.enum_combo(
                "##fit_band_level", self._fit_band_level, _CONFIDENCE_LEVELS
            )
        return self._fit_band, _CONFIDENCE_LEVEL_VALUES.get(self._fit_band_level, 0.95)

    def _fit_output_controls(self) -> Tuple[bool, int]:
        """The output-grid checkbox + point-count field. Available on every model.

        This is the "how many points should Apply add" control: off (default) evaluates
        the fit at the source's own x samples, matching the input point-for-point. On
        evaluates it at N evenly spaced points across the x range instead, independent
        of the input's own sample count -- a smooth curve for a sparse or unevenly
        sampled fit, not a same-density echo of it.
        """
        _c, self._fit_output_grid = imgui.checkbox(
            "Evaluate on a smooth grid", self._fit_output_grid
        )
        imgui.same_line()
        widgets.help_marker(
            "Off: the fit is evaluated at the source's own x samples. On: evaluated at "
            "N evenly spaced points across the x range instead -- pick N below. Useful "
            "when the input is sparse or unevenly sampled and a smooth curve, not a "
            "same-density echo of the input, is what 'New layer'/'New dataset' should "
            "commit. 'Replace source' is unavailable while this is on, since the point "
            "count no longer matches the input."
        )
        n_points = int(self._fit_output_points)
        if self._fit_output_grid:
            imgui.same_line()
            changed, n = imgui.input_int("Points##fit_output_points", n_points)
            if changed:
                self._fit_output_points = max(2, min(int(n), 100_000))
                n_points = self._fit_output_points
        return bool(self._fit_output_grid), n_points

    def _ci_controls(self, flag_attr: str, level_attr: str, *, help: str) -> Tuple[bool, float]:
        """A "Show confidence interval" checkbox + level picker, generic over which
        tab's state it reads/writes (``flag_attr``/``level_attr`` name the two ``self``
        attributes). Point 2 of the user's request: a confidence interval option,
        available wherever a point estimate is reported, not just on the Fit tab.
        """
        show = bool(getattr(self, flag_attr))
        _c, show = imgui.checkbox("Show confidence interval", show)
        setattr(self, flag_attr, show)
        imgui.same_line()
        widgets.help_marker(help)
        if show:
            imgui.same_line()
            level_label = getattr(self, level_attr)
            _c, level_label = widgets.enum_combo(f"##{flag_attr}", level_label, _CONFIDENCE_LEVELS)
            setattr(self, level_attr, level_label)
        return show, _CONFIDENCE_LEVEL_VALUES.get(getattr(self, level_attr), 0.95)

    def _fit_robust_controls(self) -> Tuple[bool, str]:
        """The outlier-robust-fit checkbox + loss picker. Nonlinear models only."""
        _c, self._fit_robust = imgui.checkbox("Robust fit (outliers)", self._fit_robust)
        imgui.same_line()
        widgets.help_marker(
            "Down-weights points that do not fit the model instead of forcing the curve "
            "toward them. Use this when a few bad points are dragging an ordinary fit off "
            "course. Reported parameter errors are approximate under a robust loss -- "
            "treat them as a rough sense of scale, not an exact standard error."
        )
        if self._fit_robust:
            imgui.same_line()
            _c, self._fit_loss = widgets.enum_combo(
                "##fit_loss",
                self._fit_loss,
                mathops.ROBUST_LOSS_KINDS,
                help="soft_l1/huber/cauchy/arctan down-weight large residuals "
                "increasingly aggressively. linear is ordinary least squares, kept for "
                "comparison.",
            )
        return self._fit_robust, self._fit_loss

    def _fit_guess_controls(
        self, model: str, spec: mathops.FitModel
    ) -> Optional[Tuple[float, ...]]:
        """The initial-guess block. Returns the p0 to fit with, or None for automatic.

        ``curve_fit`` is a *local* optimiser: the starting point decides whether it finds
        the right basin at all, so the guess is shown rather than hidden, and can be
        taken over. The automatic values come from ``_compute`` (the heuristics need the
        source data, which a tab body does not receive), so on the very first draw of a
        model there is nothing to show yet and the row says so; from the next frame on it
        is the real start.
        """
        key = _fit_state_key(model, spec)
        names = spec.param_names
        auto = self._fit_auto.get(key)
        seed: Optional[List[float]] = list(auto) if auto and len(auto) == len(names) else None

        _changed, self._fit_manual = imgui.checkbox("Manual initial guess", self._fit_manual)
        widgets.help_marker(
            "Off: each parameter starts from a heuristic read off the data -- peak "
            "position and half width for the peak models, a log-linear rate for the "
            "exponential, log-log for the power law. That is what makes these converge "
            "on real data. Turn it on when the fit lands somewhere obviously wrong: a "
            "frequency, for one, has a local minimum every half period and no heuristic "
            "can guess it for you."
        )

        if not self._fit_manual:
            if seed is None:
                imgui.text_disabled("Start values are chosen from the data.")
            else:
                for name, value in zip(names, seed):
                    widgets.stat_row(f"{name} (auto start)", _fmt(value))
            return None

        values = self._fit_p0.get(key)
        if values is None or len(values) != len(names):
            # Seed a manual guess from the automatic one: editing a good start is the
            # workflow; typing four numbers from nothing is not.
            values = seed if seed is not None else [1.0] * len(names)
            self._fit_p0[key] = values

        for i, name in enumerate(names):
            changed, value = imgui.input_float(
                f"{name} (start)", float(values[i]), 0.0, 0.0, "%.6g"
            )
            if changed and math.isfinite(float(value)):
                values[i] = float(value)
        if imgui.small_button("Reset to auto") and seed is not None:
            values[:] = seed
        return tuple(values)

    def _sampling_controls(self, key: str, n_total: int) -> _SamplingParams:
        """A collapsible "Sampling" block for a "trainable" tab (see ``_TRAINABLE_TABS``
        -- a fit, a clustering, a projection -- whose cost actually scales with the row
        count, unlike a pointwise transform): run the technique on a random subset of
        the rows instead of all of them, and/or hold out a random val/test slice to
        score separately from what it was fit on.

        Both controls live behind the same collapsible section (drawn only while it is
        open) but are otherwise independent: a split does not require sub-sampling
        first, and sub-sampling does not require a split.
        """
        assert key in _TRAINABLE_TABS, f"{key!r} is not a trainable tab"
        imgui.push_id(key)
        try:
            if not widgets.section("Sampling", default_open=False):
                return _NO_SAMPLING

            enabled = bool(self._sampling_enabled.get(key, False))
            _c, enabled = imgui.checkbox("Sub-sample", enabled)
            self._sampling_enabled[key] = enabled
            imgui.same_line()
            widgets.help_marker(
                "Run this operation on a random subset of the rows instead of all of "
                "them -- useful when the source is too large to fit the 16ms preview "
                "budget (UMAP and hierarchical clustering are the usual culprits). "
                "The subset is chosen once per seed, not reshuffled every frame."
            )

            max_rows = int(_NO_SAMPLING.max_rows)
            subsample_seed = int(_NO_SAMPLING.subsample_seed)
            if enabled:
                max_rows = max(100, int(self._sampling_max_rows.get(key, 5000)))
                changed, new_max_rows = imgui.input_int("Max rows", max_rows)
                if changed:
                    max_rows = max(100, int(new_max_rows))
                self._sampling_max_rows[key] = max_rows

                subsample_seed = int(self._sampling_seed.get(key, 0))
                _c, subsample_seed = imgui.input_int("Seed", subsample_seed)
                self._sampling_seed[key] = int(subsample_seed)
                imgui.same_line()
                widgets.help_marker(
                    "The subset is reproducible for a given seed; change it to try a "
                    "different random slice of the same size."
                )

                used = min(max_rows, n_total) if n_total > 0 else max_rows
                imgui.text_disabled(f"{used:,} of {n_total:,} rows")

            split_enabled = bool(self._split_enabled.get(key, False))
            _c, split_enabled = imgui.checkbox("Train/val/test split", split_enabled)
            self._split_enabled[key] = split_enabled
            imgui.same_line()
            widgets.help_marker(
                "Fit only on a random 'train' slice of the rows (after sub-sampling "
                "above, if that is also on) and score a held-out 'val'/'test' slice "
                "separately -- catches a fit/clustering/projection that only looks "
                "good on the exact rows it was trained on."
            )

            val_frac = float(_NO_SAMPLING.val_frac)
            test_frac = float(_NO_SAMPLING.test_frac)
            split_seed = int(_NO_SAMPLING.split_seed)
            if split_enabled:
                val_frac = float(self._split_val_frac.get(key, _NO_SAMPLING.val_frac))
                _c, val_frac = widgets.labeled_slider_float(
                    "Val fraction",
                    val_frac,
                    vmin=0.05,
                    vmax=0.45,
                    fmt="%.2f",
                    help="Fraction of the (sub-sampled) rows held out for validation.",
                )
                self._split_val_frac[key] = float(val_frac)

                test_frac = float(self._split_test_frac.get(key, _NO_SAMPLING.test_frac))
                _c, test_frac = widgets.labeled_slider_float(
                    "Test fraction",
                    test_frac,
                    vmin=0.05,
                    vmax=0.45,
                    fmt="%.2f",
                    help="Fraction held out for a final test score, separate from val. "
                    "Val + test are capped at 90% combined so train never goes empty.",
                )
                self._split_test_frac[key] = float(test_frac)

                split_seed = int(self._split_seed.get(key, 0))
                _c, split_seed = imgui.input_int("Split seed", split_seed)
                self._split_seed[key] = int(split_seed)
                imgui.same_line()
                widgets.help_marker(
                    "Which rows land in train/val/test is reproducible for a given "
                    "seed; change it to try a different random partition."
                )

            return _SamplingParams(
                subsample=enabled,
                max_rows=int(max_rows),
                subsample_seed=int(subsample_seed),
                split=split_enabled,
                val_frac=float(val_frac),
                test_frac=float(test_frac),
                split_seed=int(split_seed),
            )
        finally:
            imgui.pop_id()

    def _tab_fit(self) -> Tuple[Any, ...]:
        """Fit parameters: polynomial regression, a named nonlinear model, or custom f(x).

        Polynomial keeps its degree spinner and its closed-form path. Everything else is
        nonlinear least squares, which is what "fitting" means to the scientist this
        panel is for: free parameters, an initial guess, and a standard error on every
        fitted value.
        """
        n_total = int(len(self._current_source.x_raw)) if self._current_source is not None else 0
        changed, self._fit_model = widgets.enum_combo(
            "Model",
            self._fit_model,
            _FIT_MODELS,
            help="polynomial: linear least squares on the coefficients, no starting "
            "guess needed. Everything else is nonlinear least squares and reports a "
            "standard error per parameter. custom: type your own f(x).",
        )
        model = self._fit_model
        if changed:
            # A guess is a vector of this model's parameters and means nothing for
            # another one; the per-model _fit_p0/_fit_auto keys already keep the values
            # apart, so only the mode has to be reset.
            self._fit_manual = False

        if model == "polynomial":
            degree_changed, degree = imgui.input_int("Degree", int(self._fit_degree))
            if degree_changed:
                self._fit_degree = max(0, min(int(degree), 20))
            widgets.help_marker(
                "Least squares. Clamped to 20 and to the sample count; a high degree on "
                "few points oscillates wildly between them."
            )
            band, level = self._fit_band_controls()
            use_grid, n_points = self._fit_output_controls()
            training = self._sampling_controls("fit", n_total)
            return (
                "fit",
                "polynomial",
                int(self._fit_degree),
                "",
                None,
                False,
                "soft_l1",
                band,
                level,
                use_grid,
                n_points,
                training,
            )

        if not mathops.fit_available():
            # No numpy fallback exists for curve_fit, so say so plainly instead of
            # offering a control that cannot work. Polynomial is still right there.
            widgets.error_box(mathops.FIT_UNAVAILABLE_MESSAGE)
            return (
                "fit",
                model,
                0,
                self._fit_expr,
                None,
                False,
                "soft_l1",
                False,
                0.95,
                False,
                200,
                _NO_SAMPLING,
            )

        if model in _MULTIPEAK_MODELS:
            npeaks_changed, npeaks = imgui.input_int("Peaks", int(self._fit_npeaks))
            if npeaks_changed:
                self._fit_npeaks = max(1, min(int(npeaks), _MAX_FIT_PEAKS))
            widgets.help_marker(
                "How many overlapping peaks to separate. Their centres and widths are "
                "seeded from the data and fitted jointly, so this resolves peaks that run "
                "into each other -- the case valley integration on the Peaks tab cannot. "
                "Each peak's area is reported exactly from its fitted amplitude and width."
            )

        if model == "custom":
            _expr_changed, self._fit_expr = imgui.input_text("f(x)", self._fit_expr)
            widgets.help_marker(
                "Every name other than x becomes a fitted parameter: 'a*exp(-x/b) + c' "
                "fits a, b and c. Same functions as the Functions panel (sin, exp, "
                "gauss, ...)."
            )

        # The peak count rides in the tuple's int slot (unused by non-polynomial models) so
        # the preview cache invalidates when it changes.
        int_slot = int(self._fit_npeaks) if model in _MULTIPEAK_MODELS else 0
        try:
            spec = self._fit_spec(model, self._fit_expr, int_slot)
        except ValueError as exc:
            widgets.error_box(str(exc))
            return (
                "fit",
                model,
                int_slot,
                self._fit_expr,
                None,
                False,
                "soft_l1",
                False,
                0.95,
                False,
                200,
                _NO_SAMPLING,
            )

        imgui.text_disabled(spec.formula)
        p0 = self._fit_guess_controls(model, spec)
        robust, loss = self._fit_robust_controls()
        band, level = self._fit_band_controls()
        use_grid, n_points = self._fit_output_controls()
        training = self._sampling_controls("fit", n_total)
        return (
            "fit",
            model,
            int_slot,
            self._fit_expr,
            p0,
            robust,
            loss,
            band,
            level,
            use_grid,
            n_points,
            training,
        )

    def _tab_fft(self) -> Tuple[Any, ...]:
        """FFT parameters (there are none: the transform is fully determined)."""
        imgui.text_disabled("One-sided amplitude spectrum.")
        imgui.same_line()
        widgets.help_marker(
            "Frequencies are in cycles per unit of x, and a pure sinusoid of amplitude A "
            "peaks at A. A non-uniform x grid is linearly resampled onto a uniform one "
            "first, which smears the spectrum -- that is an approximation, not a detail."
        )
        return ("fft",)

    def _tab_envelope(self) -> Tuple[Any, ...]:
        """Envelope parameters (there are none: the transform is fully determined)."""
        imgui.text_disabled("The amplitude envelope of an oscillating signal.")
        imgui.same_line()
        widgets.help_marker(
            "Traces the top of the signal's swing -- the curve you would draw by eye "
            "connecting each cycle's peaks. Useful for amplitude-modulated or decaying "
            "oscillations, where the envelope itself (not the oscillation) is the "
            "interesting signal. Computed via the Hilbert transform; like FFT, a "
            "non-uniform x grid is linearly resampled onto a uniform one first."
        )
        return ("envelope",)

    def _tab_rolling(self) -> Tuple[Any, ...]:
        """Moving-window statistic parameters."""
        _c, self._rolling_stat = widgets.enum_combo(
            "Statistic",
            self._rolling_stat,
            mathops.ROLLING_STATS,
            help="mean/median: a smoothed trend, like Smooth. std: a rolling "
            "volatility/noise-level trace -- where the signal gets noisier, not just "
            "what it averages to. min/max: a moving envelope of the signal's own "
            "extremes.",
        )
        changed, window = imgui.input_int("Window", int(self._rolling_window))
        if changed:
            self._rolling_window = max(1, int(window))
        widgets.help_marker("Width of the moving window, in samples.")
        _c, self._rolling_center = imgui.checkbox("Center", self._rolling_center)
        imgui.same_line()
        widgets.help_marker(
            "On: each point uses the samples around it -- best for exploring data you "
            "already have in full. Off: trailing/causal, using only samples up to and "
            "including each point -- what a live computation could actually produce "
            "without seeing the future."
        )
        return (
            "rolling",
            self._rolling_stat,
            int(self._rolling_window),
            bool(self._rolling_center),
        )

    def _column_picker(
        self, label: str, current: str, *, help: Optional[str] = None, allow_none: bool = False
    ) -> Optional[str]:
        """A combo over the *current* source's dataset columns (see ``_current_source``).

        Returns the selected column name, or None with an explanatory disabled line when
        the source has no dataset (a plotted layer has no table to pick a second column
        from) or the dataset has no columns.

        ``allow_none``: prepend a "(none)" entry ahead of the dataset's own columns and
        return None when it is the current selection -- the UMAP "Color by" picker's
        default uncolored state (Phase 5). Every existing caller leaves this False and
        is unaffected: the combo always shows a real column, defaulting to the first.
        """
        source = self._current_source
        dataset = source.dataset if source is not None else None
        if dataset is None:
            imgui.text_disabled(f"{label}: needs a dataset source (a plotted layer has no table).")
            return None
        names = dataset.column_names()
        if not names:
            imgui.text_disabled(f"{label}: the dataset has no columns.")
            return None
        if allow_none:
            options = ("(none)",) + tuple(names)
            value = current if current in options else "(none)"
            _c, picked = widgets.enum_combo(label, value, options, help=help)
            return None if picked == "(none)" else picked
        value = current if current in names else names[0]
        _c, picked = widgets.enum_combo(label, value, tuple(names), help=help)
        return picked

    def _target_column_picker(
        self, label: str, dataset: DataSet, current: str, *, help: Optional[str] = None
    ) -> Optional[str]:
        """A combo over an EXPLICIT ``dataset``'s columns -- ``_column_picker``'s sibling
        for Phase 4's "Apply saved model to..." flow, which maps a model's own trained
        column names onto a TARGET dataset that is (deliberately) not whatever the
        Source tab currently has open.

        Same disabled-line behaviour as ``_column_picker`` when ``dataset`` has no
        columns; otherwise identical ``widgets.enum_combo`` mechanics.
        """
        names = dataset.column_names()
        if not names:
            imgui.text_disabled(f"{label}: the dataset has no columns.")
            return None
        value = current if current in names else names[0]
        _c, picked = widgets.enum_combo(label, value, tuple(names), help=help)
        return picked

    def _multi_column_picker(
        self, label: str, selected: Tuple[str, ...], *, help: Optional[str] = None
    ) -> Tuple[str, ...]:
        """A checkbox per column of the *current* source's dataset (see
        ``_current_source``) -- PCA/UMAP's input, unlike every other tab's single (x, y)
        pair, is however many columns the user picks.

        Returns the selected column names in dataset order, or ``()`` with an
        explanatory disabled line when the source has no dataset (a plotted layer has
        no table) or the dataset has no columns.
        """
        source = self._current_source
        dataset = source.dataset if source is not None else None
        if dataset is None:
            imgui.text_disabled(f"{label}: needs a dataset source (a plotted layer has no table).")
            return ()
        names = dataset.column_names()
        if not names:
            imgui.text_disabled(f"{label}: the dataset has no columns.")
            return ()
        imgui.text(label)
        if help:
            imgui.same_line()
            widgets.help_marker(help)
        chosen = set(selected)
        for name in names:
            checked = name in chosen
            changed, checked = imgui.checkbox(f"{name}##{label}_{name}", checked)
            if changed:
                if checked:
                    chosen.add(name)
                else:
                    chosen.discard(name)
        return tuple(n for n in names if n in chosen)

    def _tab_autocorr(self) -> Tuple[Any, ...]:
        """Autocorrelation (self vs. a delayed copy) or cross-correlation (vs. another
        column) parameters: how far out in lag to compute."""
        _c, self._autocorr_mode = widgets.enum_combo(
            "Mode",
            self._autocorr_mode,
            ("auto", "cross"),
            help="auto: how the signal correlates with a delayed copy of ITSELF -- a "
            "peak marks a repeating period. cross: how it correlates with a delayed "
            "copy of ANOTHER column -- a peak marks a lag/delay between the two.",
        )
        cross_col = ""
        if self._autocorr_mode == "cross":
            picked = self._column_picker(
                "Compare to",
                self._autocorr_cross_col,
                help="The second column, correlated against the source's y at every lag.",
            )
            if picked is not None:
                self._autocorr_cross_col = picked
                cross_col = picked

        if self._autocorr_mode == "auto":
            imgui.text_disabled("Self-similarity vs lag; a peak marks a repeating period.")
            imgui.same_line()
            widgets.help_marker(
                "The autocorrelation is 1 at lag 0 and shows a peak at the lag of any "
                "period the signal repeats over -- often clearer than an FFT on a short "
                "record. The first peak (marked) is reported as the dominant period, in "
                "x units."
            )
        else:
            imgui.text_disabled("Similarity vs lag against another column; a peak marks a delay.")
            imgui.same_line()
            widgets.help_marker(
                "A peak at a positive lag means the compared column lags the source's y "
                "by that many samples; a peak at a negative lag means it leads. Unlike "
                "autocorrelation this is not symmetric and is not 1 at lag 0."
            )
        _c, self._autocorr_maxlag = widgets.labeled_slider_float(
            "Max lag",
            self._autocorr_maxlag,
            vmin=0.05,
            vmax=1.0,
            help="Largest lag to compute, as a fraction of the record length. The estimate "
            "is noisier at long lags, so half is a good default.",
        )
        return ("autocorr", self._autocorr_mode, cross_col, float(self._autocorr_maxlag))

    def _tab_stats(self) -> Tuple[Any, ...]:
        """Statistics: the source's y column, described. "Show more" adds shape/robust
        statistics (percentiles, IQR, MAD, skewness, kurtosis, outlier count)."""
        imgui.text_disabled("Summary of the source's y column. Every statistic is nan-safe.")
        _c, self._stats_show_more = imgui.checkbox("Show more", self._stats_show_more)
        imgui.same_line()
        widgets.help_marker(
            "Percentiles, interquartile range, median absolute deviation, skewness, "
            "excess kurtosis, and a count of samples outside the standard Tukey fence "
            "(1.5x the IQR beyond Q1/Q3) -- the shape and robust-spread statistics "
            "describe() alone does not report."
        )
        show_ci, ci_level = self._ci_controls(
            "_stats_ci",
            "_stats_ci_level",
            help="A range likely to contain the TRUE population mean, via Student's t "
            "-- not the spread of the data itself (that is the Std deviation row).",
        )
        return ("stats", bool(self._stats_show_more), show_ci, ci_level)

    def _tab_correlate(self) -> Tuple[Any, ...]:
        """Correlate has no parameters: it reads the source's own (x, y) pair."""
        imgui.text_disabled("How the source's y column tracks its x column.")
        imgui.same_line()
        widgets.help_marker(
            "Pearson r measures a LINEAR relationship; Spearman rho measures any "
            "consistently increasing or decreasing one (robust to outliers and curved-"
            "but-monotonic relationships Pearson misses). Both range -1 to 1, with the "
            "p-value the chance of seeing a correlation this strong from independent "
            "data. The overlay is the ordinary least-squares line through the points."
        )
        show_ci, ci_level = self._ci_controls(
            "_correlate_ci",
            "_correlate_ci_level",
            help="A range likely to contain the TRUE Pearson r, via Fisher's z-"
            "transform -- narrower with more points or a stronger relationship.",
        )
        return ("correlate", show_ci, ci_level)

    def _tab_compare(self) -> Tuple[Any, ...]:
        """Two-sample hypothesis test: is the source's y column drawn from the same
        population as another column?"""
        picked = self._column_picker(
            "Compare to",
            self._compare_col,
            help="The second, independent sample -- not a paired (x, y) relationship "
            "the way Correlate reads its input, so it need not be the same length.",
        )
        if picked is not None:
            self._compare_col = picked
        _c, self._compare_method = widgets.enum_combo(
            "Method",
            self._compare_method,
            mathops.TWO_SAMPLE_METHODS,
            help="welch: tests whether the two means differ, without assuming equal "
            "variances (the safe default). student: the same, assuming equal "
            "variances. mannwhitney: a rank-based test for which sample tends larger, "
            "no normality assumed -- robust to outliers and skew. ks: tests whether "
            "the full distributions differ in shape, not just location.",
        )
        show_ci, ci_level = self._ci_controls(
            "_compare_ci",
            "_compare_ci_level",
            help="A range likely to contain the TRUE difference of the two population "
            "means (welch/student only -- mannwhitney/ks test the full distribution, "
            "not the mean, so no matching interval exists for them).",
        )
        return ("compare", self._compare_col, self._compare_method, show_ci, ci_level)

    def _cluster_training_xy(self, source: _Source, key: str) -> Tuple[np.ndarray, np.ndarray]:
        """The (x, y) pair the main Cluster computation for tab ``key`` actually fits
        on: the source's own (x, y), sub-sampled and (if a split is on) reduced to the
        train partition only -- mirroring ``_compute``'s own "cluster" branch exactly.

        Used by the elbow/silhouette diagnostic so its k-sweep runs against the SAME
        rows the tab's own result was fit on, read fresh from the panel's current
        sampling/split state rather than threaded through ``_Result`` (nothing else
        needs it).
        """
        xa = np.asarray(source.x_raw, dtype=np.float64)
        ya = np.asarray(source.y_raw, dtype=np.float64)
        if self._sampling_enabled.get(key, False):
            max_rows = int(self._sampling_max_rows.get(key, _NO_SAMPLING.max_rows))
            seed = int(self._sampling_seed.get(key, _NO_SAMPLING.subsample_seed))
            idx = _subsample_indices(int(xa.size), max_rows, seed=seed)
            if idx is not None:
                xa, ya = xa[idx], ya[idx]
        if self._split_enabled.get(key, False):
            val_frac = float(self._split_val_frac.get(key, _NO_SAMPLING.val_frac))
            test_frac = float(self._split_test_frac.get(key, _NO_SAMPLING.test_frac))
            split_seed = int(self._split_seed.get(key, _NO_SAMPLING.split_seed))
            train_idx, _val_idx, _test_idx = _split_indices(
                int(xa.size), val_frac, test_frac, seed=split_seed
            )
            xa, ya = xa[train_idx], ya[train_idx]
        return xa, ya

    def _draw_elbow_section(self, source: _Source, key: str) -> None:
        """The k-means elbow/silhouette diagnostic (Phase 5): a k-fold-more-expensive
        k-sweep, offered ONLY as an explicit button click -- never dispatched through
        ``_cached_result_async``/``_compute``, since a k-sweep is several fits, not one
        ``_Result``, and must not restart just because the tab's own params look stable
        (the debounce/memoisation machinery those build on assumes a single dispatch-by-
        kind-string result, which this is not).

        Collapsible, same convention as ``_sampling_controls``, so a tab that never
        clicks it stays uncluttered.
        """
        if not widgets.section("Elbow / silhouette (k-means)", default_open=False):
            return
        imgui.text_disabled(
            "Fits k-means at every k from 2 up to 10 (capped below the row count) and "
            "reports inertia and silhouette for each -- helps pick k, at the cost of "
            "several fits instead of one."
        )
        imgui.same_line()
        widgets.help_marker(
            "Runs on the same (sub-sampled/train) rows the Cluster computation above "
            "uses. Never runs automatically -- only when this button is clicked."
        )

        job = self._elbow_jobs.get(key)
        if job is not None:
            status = job.poll()
            if status.state == "running":
                if widgets.busy_row(
                    "Computing elbow/silhouette", status.elapsed, id_str=f"elbow_{key}"
                ):
                    job.cancel()
                    self._elbow_jobs.pop(key, None)
                return
            self._elbow_jobs.pop(key, None)
            if status.state == "error":
                widgets.error_box(status.error or "elbow/silhouette failed")
            elif status.state == "done" and status.result is not None:
                self._elbow_results[key] = status.result

        if imgui.button(f"Compute elbow/silhouette (k=2..N)##elbow_{key}"):
            xa, ya = self._cluster_training_xy(source, key)
            seed = int(self._cluster_seed)
            self._elbow_jobs[key] = BackgroundJob(lambda: _elbow_sweep(xa, ya, seed=seed))

        results = self._elbow_results.get(key)
        if results is None:
            return
        k_values, inertias, silhouettes = results
        if k_values.size == 0:
            imgui.text_disabled("Not enough rows for a k-sweep.")
            return
        chosen = float(self._cluster_k)
        nearest = int(np.argmin(np.abs(k_values - chosen)))
        background, grid, palette = self._preview_style()
        line_color = palette[0 % len(palette)] if palette else None
        widgets.mini_plot(
            "elbow_inertia",
            inertias,
            x=k_values,
            markers=([float(k_values[nearest])], [float(inertias[nearest])]),
            marker_color=theme.get_color("warn"),
            height=90.0,
            label="Inertia vs k",
            color=line_color,
            background_color=background,
            grid_color=grid,
        )
        widgets.mini_plot(
            "elbow_silhouette",
            silhouettes,
            x=k_values,
            markers=([float(k_values[nearest])], [float(silhouettes[nearest])]),
            marker_color=theme.get_color("warn"),
            height=90.0,
            label="Silhouette vs k",
            color=line_color,
            background_color=background,
            grid_color=grid,
        )

    def _tab_cluster(self) -> Tuple[Any, ...]:
        """K-means or hierarchical clustering of the source's (x, y) points into k groups."""
        n_total = int(len(self._current_source.x_raw)) if self._current_source is not None else 0
        imgui.text_disabled("Groups the plotted (x, y) points into k clusters.")
        imgui.same_line()
        widgets.help_marker(
            "Clusters exactly the two columns already plotted -- this tab does not "
            "select more variables. If your data has more than two dimensions worth "
            "clustering on, reduce it to two first (e.g. with Correlate or a Data Editor "
            "transform), then cluster the result."
        )
        _c, self._cluster_method = widgets.enum_combo(
            "Method",
            self._cluster_method,
            ("kmeans", "hierarchical"),
            help="kmeans: fast, assumes roughly spherical/similarly-sized clusters, "
            "restarted from several random seeds to avoid a bad local minimum. "
            "hierarchical: builds a merge tree bottom-up and cuts it to k groups -- no "
            "randomness, and copes better with elongated or unevenly sized clusters.",
        )
        changed, k = imgui.input_int("Clusters (k)", int(self._cluster_k))
        if changed:
            self._cluster_k = max(2, min(int(k), 10))
        widgets.help_marker(
            "Number of groups to find. There is no single 'correct' k -- try a few and "
            "see which one splits the data into groups that actually look separated in "
            "the preview."
        )

        if self._cluster_method == "hierarchical":
            if not mathops2d.hierarchical_available():
                widgets.error_box(mathops2d.HIERARCHICAL_UNAVAILABLE_MESSAGE)
            _c, self._cluster_linkage = widgets.enum_combo(
                "Linkage",
                self._cluster_linkage,
                mathops2d.HIERARCHICAL_METHODS,
                help="ward: minimises within-cluster variance, usually the best "
                "default. single/complete/average: link clusters by their nearest, "
                "farthest, or average pairwise distance -- single is prone to "
                "'chaining' long strands together; complete and average stay compact.",
            )
            training = self._sampling_controls("cluster", n_total)
            return (
                "cluster",
                "hierarchical",
                int(self._cluster_k),
                self._cluster_linkage,
                training,
            )

        _c, self._cluster_seed = imgui.input_int("Seed", int(self._cluster_seed))
        imgui.same_line()
        widgets.help_marker(
            "K-means starts from a random guess and can land in different (equally "
            "valid) groupings depending on it. Changing the seed reruns with a different "
            "start; the same seed always reproduces the same result."
        )
        training = self._sampling_controls("cluster", n_total)
        if self._current_source is not None:
            # kmeans only: hierarchical re-cuts the SAME merge tree per k rather than
            # re-fitting, so there is no comparable "try several k cheaply" story for
            # it -- see _draw_elbow_section's own docstring.
            self._draw_elbow_section(self._current_source, "cluster")
        return (
            "cluster",
            "kmeans",
            int(self._cluster_k),
            int(self._cluster_seed),
            training,
        )

    def _tab_density2d(self) -> Tuple[Any, ...]:
        """2-D density of the source's (x, y) points: a binned histogram, or a KDE surface."""
        imgui.text_disabled("Density of the plotted (x, y) points, not their trend over x.")
        imgui.same_line()
        widgets.help_marker(
            "Like Cluster, this reads exactly the two columns already plotted -- no "
            "further column selection. For 'New layer', pick 'heat map (hist2d)' from "
            "the Kind dropdown below to commit this as a real layer."
        )
        _c, self._density_mode = widgets.enum_combo(
            "Mode",
            self._density_mode,
            ("histogram", "kde"),
            help="histogram: a plain 2-D bin count, pure numpy, always available. kde: a "
            "smooth probability-density surface (scipy.stats.gaussian_kde) -- reads "
            "better with fewer samples, at the cost of needing scipy.",
        )

        if self._density_mode == "histogram":
            changed, bins = imgui.input_int("Bins", int(self._density_bins))
            if changed:
                self._density_bins = max(5, min(int(bins), 100))
            widgets.help_marker(
                "Bin count per axis. More bins resolve finer structure but need more "
                "points per bin to stay meaningful -- with few samples, fewer, bigger "
                "bins read more honestly."
            )
            return ("density2d", "histogram", int(self._density_bins), 0, False, 1.0)

        if not mathops2d.kde2d_available():
            widgets.error_box(mathops2d.KDE_UNAVAILABLE_MESSAGE)
            return ("density2d", "kde", self._density_bins, int(self._density_grid), False, 1.0)

        changed, grid = imgui.input_int("Grid resolution", int(self._density_grid))
        if changed:
            self._density_grid = max(5, min(int(grid), 100))
        widgets.help_marker(
            "Grid points per axis the KDE surface is evaluated on -- how detailed the "
            "drawn grid is, not how smooth the density estimate itself is (that is the "
            "bandwidth, below)."
        )
        _c, self._density_custom_bw = imgui.checkbox("Custom bandwidth", self._density_custom_bw)
        imgui.same_line()
        widgets.help_marker(
            "Off: scipy picks a bandwidth automatically (Scott's rule) from the point "
            "count and spread. On: scale it manually -- below 1 is peakier/more "
            "detailed, above 1 is smoother -- useful when the automatic choice over- or "
            "under-smooths."
        )
        bw_factor = 1.0
        if self._density_custom_bw:
            imgui.same_line()
            _c, self._density_bw_factor = widgets.labeled_slider_float(
                "##bw_factor", self._density_bw_factor, vmin=0.1, vmax=3.0, fmt="%.2f"
            )
            bw_factor = float(self._density_bw_factor)
        return (
            "density2d",
            "kde",
            int(self._density_bins),
            int(self._density_grid),
            bool(self._density_custom_bw),
            bw_factor,
        )

    def _tab_spatial(self) -> Tuple[Any, ...]:
        """Spatial statistics have no parameters: they read the source's own (x, y) pair."""
        imgui.text_disabled("Spacing, extent and the convex hull of the plotted points.")
        imgui.same_line()
        widgets.help_marker(
            "Nearest-neighbour distance describes how tightly the points are packed -- "
            "a minimum far below the mean flags near-duplicates or a locally dense "
            "clump. The convex hull is the smallest polygon containing every point; its "
            "vertices are marked on the preview. Needs at least 3 non-collinear points "
            "to have a hull at all."
        )
        return ("spatial",)

    def _default_nd_columns(self, current: Tuple[str, ...]) -> Tuple[str, ...]:
        """ "Every column of the dataset" the first time an N-D tab meets a real source.

        Shared by PCA and UMAP: both start with ``current == ()`` (see the fields'
        own comments in __init__) and want the same "just try it" default once a
        dataset with columns actually exists, without overriding a deliberate pick
        the user later empties back out to zero.
        """
        if current:
            return current
        source = self._current_source
        dataset = source.dataset if source is not None else None
        if dataset is None:
            return current
        return tuple(dataset.column_names())

    def _tab_pca(self) -> Tuple[Any, ...]:
        """Principal component analysis: project 2+ columns onto their top directions
        of variance."""
        imgui.text_disabled("Reduce several columns to their top directions of variance.")
        imgui.same_line()
        widgets.help_marker(
            "Pick 2 or more columns below. The preview always shows PC1 vs PC2 -- the "
            "two directions capturing the most spread in the data -- as a scatter; "
            "'New dataset' commits every requested component as its own column."
        )
        self._pca_columns = self._default_nd_columns(self._pca_columns)
        self._pca_columns = self._multi_column_picker(
            "Columns",
            self._pca_columns,
            help="Every column selected here becomes one input dimension. Unrelated "
            "columns dilute the result -- pick the ones you actually think vary "
            "together.",
        )
        changed, n = imgui.input_int("Components", int(self._pca_n_components))
        if changed:
            self._pca_n_components = max(2, min(int(n), 20))
        widgets.help_marker(
            "How many components to compute and commit to 'New dataset'. The preview "
            "only ever plots the first two, regardless of this count."
        )
        _c, self._pca_scale = widgets.enum_combo(
            "Scale",
            self._pca_scale,
            mathopsnd.SCALE_MODES,
            help="none: use the columns as-is. zscore (default): each column is "
            "centered and divided by its own standard deviation first, so PCA finds "
            "structure, not just whichever column happens to have the biggest raw "
            "numbers. minmax: each column mapped onto [0, 1] first instead.",
        )
        dataset = self._current_source.dataset if self._current_source is not None else None
        n_total = int(dataset.n_rows()) if dataset is not None else 0
        training = self._sampling_controls("pca", n_total)
        return (
            "pca",
            self._pca_columns,
            int(self._pca_n_components),
            self._pca_scale,
            training,
        )

    def _tab_umap(self) -> Tuple[Any, ...]:
        """UMAP: a nonlinear embedding of 2+ columns onto 2 (or more) dimensions."""
        imgui.text_disabled("Nonlinear embedding of several columns onto a 2-D layout.")
        imgui.same_line()
        widgets.help_marker(
            "Unlike PCA, UMAP has no linear formula and no explained-variance readout "
            "-- it optimises a layout where points close in the original columns stay "
            "close in the embedding. Better at revealing clusters/manifolds a linear "
            "projection would flatten together; the trade-off is that distances and "
            "axis directions in the result are not otherwise meaningful."
        )
        if not mathopsnd.umap_available():
            widgets.error_box(mathopsnd.UMAP_UNAVAILABLE_MESSAGE)
        self._umap_columns = self._default_nd_columns(self._umap_columns)
        self._umap_columns = self._multi_column_picker(
            "Columns",
            self._umap_columns,
            help="Every column selected here becomes one input dimension.",
        )
        changed, n = imgui.input_int("Components", int(self._umap_n_components))
        if changed:
            self._umap_n_components = max(2, min(int(n), 10))
        widgets.help_marker("Output dimensionality. The preview only ever plots the first two.")
        changed, neighbors = imgui.input_int("Neighbors", int(self._umap_n_neighbors))
        if changed:
            self._umap_n_neighbors = max(2, min(int(neighbors), 200))
        widgets.help_marker(
            "Local neighborhood size. Small: preserves fine local structure (tight, "
            "many small clusters). Large: preserves more of the overall/global "
            "layout. Automatically capped below the point count."
        )
        changed, min_dist = imgui.slider_float("Min distance", float(self._umap_min_dist), 0.0, 1.0)
        if changed:
            self._umap_min_dist = float(min_dist)
        widgets.help_marker(
            "How tightly points are allowed to pack in the embedding. Small: "
            "same-neighborhood points can sit almost on top of each other (reads "
            "more like discrete clusters). Large: spread out more evenly (reads "
            "more like a continuous shape)."
        )
        _c, self._umap_scale = widgets.enum_combo(
            "Scale",
            self._umap_scale,
            mathopsnd.SCALE_MODES,
            help="Same per-column preprocessing as PCA's Scale option, applied before "
            "UMAP runs -- see there for what each mode does.",
        )
        _c, self._umap_seed = imgui.input_int("Seed", int(self._umap_seed))
        imgui.same_line()
        widgets.help_marker(
            "UMAP's layout optimisation is stochastic; the same seed always "
            "reproduces the same embedding, a different seed gives an equally valid "
            "but differently arranged one."
        )
        dataset = self._current_source.dataset if self._current_source is not None else None
        n_total = int(dataset.n_rows()) if dataset is not None else 0
        training = self._sampling_controls("umap", n_total)
        return (
            "umap",
            self._umap_columns,
            int(self._umap_n_components),
            int(self._umap_n_neighbors),
            float(self._umap_min_dist),
            self._umap_scale,
            int(self._umap_seed),
            training,
        )

    def _tab_dynamics(self) -> Tuple[Any, ...]:
        """Dynamical-systems analysis: reconstruct and characterise the flow behind y(t)."""
        imgui.text_disabled("Read the y column as a dynamical system's time series.")
        imgui.same_line()
        widgets.help_marker(
            "These analyses treat the source's y column as one observable of a dynamical "
            "system sampled over x (time). Feed them a trajectory column from the Dynamics "
            "panel, or any time series.\n\n"
            "phase: reconstruct the attractor from the single signal by plotting y(t) "
            "against a time-delayed copy y(t+delay) -- a Takens embedding.\n"
            "return map: plot each local maximum of y against the next; a chaotic flow "
            "like Lorenz collapses onto a sharp curve here.\n"
            "lyapunov: estimate the largest Lyapunov exponent (Rosenstein). Positive means "
            "chaos, ~0 a limit cycle, negative a fixed point."
        )
        _c, self._dyn_mode = widgets.enum_combo("Analysis", self._dyn_mode, _DYNAMICS_MODES)

        if self._dyn_mode in ("phase", "lyapunov"):
            changed, delay = imgui.input_int("Delay (samples)", int(self._dyn_delay))
            if changed:
                self._dyn_delay = max(1, int(delay))
            widgets.help_marker(
                "Time shift for the delay coordinate, in samples. Too small and the "
                "embedding is a thin diagonal line; open the loop by raising it toward a "
                "quarter of the signal's period."
            )
        if self._dyn_mode == "lyapunov":
            changed, dim = imgui.input_int("Embedding dim", int(self._dyn_dim))
            if changed:
                self._dyn_dim = max(2, min(int(dim), 10))
            widgets.help_marker(
                "Dimensions of the reconstructed phase space. 3 suits most 3-D flows; raise "
                "it if the exponent looks unstable."
            )
        return ("dynamics", self._dyn_mode, int(self._dyn_delay), int(self._dyn_dim))

    # -- compute -------------------------------------------------------------------

    def _async_enabled(self) -> bool:
        """True when compute should run on a background thread instead of inline.

        Defaults to True for a real, live plot (``GPULinePlot._is_test_mode`` is False
        unless explicitly set) and to False for anything else -- including every
        headless test's ``_FakePlot`` stand-in, which has no ``_is_test_mode``
        attribute at all. That default (missing attribute => synchronous) is what
        keeps the whole pre-existing test suite deterministic: it drives the panel
        through a fixed, small number of ``_draw_frame()`` calls and asserts on
        ``self._cache_result``/``_cache_error`` immediately afterward, which is only
        valid if compute is synchronous. ``_force_async`` is the escape hatch for
        tests that specifically want to exercise the background path.
        """
        if self._force_async:
            return True
        return not getattr(self.plot, "_is_test_mode", True)

    def _async_compute_fn(self, source: _Source, params: Tuple[Any, ...]) -> Callable[[], _Result]:
        """Wrap ``_compute`` so a background failure carries the exact same message the
        synchronous path would show. ``BackgroundJob`` only stringifies whatever it
        catches, so the ValueError-vs-anything-else classification has to happen here,
        not there.
        """

        def run() -> _Result:
            try:
                return self._compute(source, params)
            except ValueError as exc:
                # The documented failure mode of every mathops entry point.
                raise RuntimeError(str(exc)) from exc
            except Exception as exc:  # pragma: no cover - defensive, mirrors the sync path
                logger.exception("Math Lab background compute failed for params %r", params)
                raise RuntimeError(f"Unexpected error: {exc}") from exc

        return run

    def _cached_result(
        self, source: _Source, params: Tuple[Any, ...], tab_key: str
    ) -> Tuple[Optional[_Result], Optional[str], str]:
        """Memoised ``_compute``. See the module docstring on the fingerprint heuristic.

        Returns ``(result, error, status)``. ``status`` is one of:

        * ``"ready"`` -- ``result``/``error`` reflect a settled computation (sync or
          async); this is the only status possible in synchronous mode, unchanged
          from before background compute existed.
        * ``"pending"`` -- still computing, or debouncing before starting (see
          ``_cached_result_async``); ``result``/``error`` are both None. Async only.
        * ``"cancelled"`` -- the user cancelled this exact computation; both None.
          Async only.
        """
        key = (
            self._cache_epoch,
            source.key,
            _fingerprint(source.x_raw, source.y_raw),
            params,
        )
        if key == self._cache_key:
            return self._cache_result, self._cache_error, "ready"

        if not self._async_enabled():
            self._cache_key = key
            try:
                self._cache_result = self._compute(source, params)
                self._cache_error = None
            except ValueError as exc:
                # The documented failure mode of every mathops entry point.
                self._cache_result = None
                self._cache_error = str(exc)
            except Exception as exc:  # pragma: no cover - defensive
                # An exception escaping here would kill the imgui frame, taking the whole
                # window with it. Surface it and keep drawing.
                logger.exception("Math Lab preview failed for params %r", params)
                self._cache_result = None
                self._cache_error = f"Unexpected error: {exc}"
            return self._cache_result, self._cache_error, "ready"

        return self._cached_result_async(source, params, tab_key, key)

    def _cached_result_async(
        self,
        source: _Source,
        params: Tuple[Any, ...],
        tab_key: str,
        key: Tuple[Any, ...],
    ) -> Tuple[Optional[_Result], Optional[str], str]:
        """The background half of :meth:`_cached_result`.

        One :class:`~glplot.gui.background.BackgroundJob` at a time per tab, tracked
        in ``self._async_jobs``/``_async_job_keys``/``_async_stable_since`` (all keyed
        by ``tab_key``, never globally): switching tabs must not cancel a slow
        computation the user is still waiting on elsewhere, so each tab's in-flight/
        settled state is independent of every other tab's.

        Debounce applies ONLY when superseding a job that was still running --
        stopping a continuous parameter drag from piling up several abandoned slow
        jobs it can never hard-cancel (see background.py). The very first dispatch for
        a tab, or a dispatch after a settled/cancelled state, starts immediately: no
        latency added to the common "open a tab, see a result" case.
        """
        settled = self._async_results.get(tab_key)
        if settled is not None and settled[2] == key:
            result, error, _settled_key = settled
            self._cache_key, self._cache_result, self._cache_error = key, result, error
            return result, error, "ready"

        now = time.monotonic()
        job = self._async_jobs.get(tab_key)

        if self._async_job_keys.get(tab_key) != key:
            # "Churning" covers both a job actually running right now AND an earlier
            # supersession still debouncing with no job dispatched yet -- either way
            # this key change extends the SAME wait, it does not start a fresh one.
            # Using only "is a job running" here is the bug this guards against: once
            # one supersession pops the job (below) without dispatching a replacement,
            # a second rapid change would see no job at all and wrongly conclude
            # nothing was in flight, defeating the debounce for every change after
            # the first.
            was_churning = job is not None or tab_key in self._async_stable_since
            if job is not None:
                job.cancel()
            self._async_job_keys[tab_key] = key
            self._async_jobs.pop(tab_key, None)
            job = None
            if was_churning:
                self._async_stable_since[tab_key] = now
            else:
                self._async_stable_since.pop(tab_key, None)

        if job is None:
            stable_since = self._async_stable_since.get(tab_key)
            if stable_since is not None and now - stable_since < _ASYNC_DEBOUNCE_SECONDS:
                return None, None, "pending"
            self._async_stable_since.pop(tab_key, None)
            self._async_jobs[tab_key] = BackgroundJob(self._async_compute_fn(source, params))
            return None, None, "pending"

        status = job.poll()
        if status.state == "running":
            return None, None, "pending"
        if status.state == "cancelled":
            return None, None, "cancelled"

        # Settled ("done" or "error") -- becomes this tab's result until its params
        # next change.
        del self._async_jobs[tab_key]
        title = _TAB_BY_KEY.get(tab_key, (tab_key, ""))[0]
        if status.state == "done":
            result, error = status.result, None
            if status.elapsed >= _ASYNC_NOTIFY_THRESHOLD_SECONDS:
                notifications.push(f"{title} finished ({status.elapsed:.1f}s)", kind="success")
        else:
            result, error = None, status.error
            if status.elapsed >= _ASYNC_NOTIFY_THRESHOLD_SECONDS:
                notifications.push(f"{title} failed: {error}", kind="error")
        self._async_results[tab_key] = (result, error, key)
        self._cache_key, self._cache_result, self._cache_error = key, result, error
        return result, error, "ready"

    def _cancel_async(self, tab_key: str) -> None:
        """Abandon ``tab_key``'s in-flight job. See background.py: cooperative, not a
        hard kill -- poll() reports "cancelled" from the next call on regardless of
        whether the underlying thread has actually stopped."""
        job = self._async_jobs.get(tab_key)
        if job is not None:
            job.cancel()

    def _retry_async(self, tab_key: str) -> None:
        """Re-run a cancelled computation with its same params, right away -- no
        debounce, the user just explicitly asked for this one; it is not drag churn."""
        self._async_jobs.pop(tab_key, None)
        self._async_stable_since.pop(tab_key, None)

    def _compute(self, source: _Source, params: Tuple[Any, ...]) -> _Result:
        """Run one operation. Pure: reads the source, returns a new _Result."""
        kind = params[0]
        x_raw, y_raw = source.x_raw, source.y_raw
        y_name = source.y_name

        if kind == "integral":
            _, method, initial = params
            xs, ys = _sorted_by_x(x_raw, y_raw)
            out = mathops.integrate(x_raw, y_raw, method=method, initial=initial)
            total = mathops.definite_integral(x_raw, y_raw, method=method)
            return _Result(
                x=xs,
                y=out,
                x_name=source.x_name,
                y_name=f"{y_name} integral",
                suffix="integral",
                plot_label=f"cumulative integral ({method})",
                overlay=ys,
                x_changed=True,
                stats=[
                    ("Definite integral", _fmt(total)),
                    ("Result at x max", _fmt(float(out[-1]))),
                    ("Span of x", _fmt(float(xs[-1] - xs[0]))),
                ],
            )

        if kind == "derivative":
            _, order, method, window, polyorder, parametric = params
            if parametric:
                # A parametric curve keeps its own sample order (it may loop back on itself),
                # so nothing is sorted and the original x is the abscissa of the result.
                par_method = method if method in ("central", "forward") else "central"
                out = mathops.parametric_derivative(x_raw, y_raw, order=order, method=par_method)
                xs, ys = np.asarray(x_raw, dtype=np.float64), np.asarray(y_raw, dtype=np.float64)
                label = f"parametric derivative order {order} ({par_method})"
                suffix = f"d^{order}y/dx^{order}" if order > 1 else "dy/dx"
                x_changed = False
            else:
                xs, ys = _sorted_by_x(x_raw, y_raw)
                out = mathops.derivative(
                    x_raw,
                    y_raw,
                    order=order,
                    method=method,
                    window=window,
                    polyorder=polyorder,
                )
                label = f"derivative order {order} ({method})"
                suffix = f"derivative {order}"
                x_changed = True
            with np.errstate(all="ignore"):
                peak = float(np.nanmax(np.abs(out))) if out.size else float("nan")
            return _Result(
                x=xs,
                y=out,
                x_name=source.x_name,
                y_name=f"{y_name} {suffix}",
                suffix=suffix,
                plot_label=label,
                overlay=ys,
                x_changed=x_changed,
                stats=[
                    ("Peak absolute slope", _fmt(peak)),
                    ("Mean slope", _fmt(float(np.nanmean(out)) if out.size else float("nan"))),
                ],
            )

        if kind == "smooth":
            _, method, window, sigma, polyorder, alpha = params
            xa = np.asarray(x_raw, dtype=np.float64)
            ya = np.asarray(y_raw, dtype=np.float64)
            kwargs: Dict[str, Any] = {}
            if method == "gaussian" and sigma > 0.0:
                kwargs["sigma"] = sigma
            if method == "savgol":
                kwargs["polyorder"] = polyorder
            if method == "ema" and alpha > 0.0:
                kwargs["alpha"] = alpha
            out = mathops.smooth(y_raw, method=method, window=window, **kwargs)
            residual = ya - out
            with np.errstate(all="ignore"):
                rms = float(np.sqrt(np.nanmean(residual**2))) if residual.size else float("nan")
            return _Result(
                x=xa,
                y=out,
                x_name=source.x_name,
                y_name=f"{y_name} smoothed",
                suffix="smoothed",
                plot_label=f"smoothed ({method}, window {window})",
                overlay=ya,
                stats=[
                    ("Removed (RMS)", _fmt(rms)),
                    ("Peak residual", _fmt(_nan_peak(residual))),
                ],
            )

        if kind == "noise":
            _, noise_kind, amplitude, relative, seed, _nonce = params
            xa = np.asarray(x_raw, dtype=np.float64)
            ya = np.asarray(y_raw, dtype=np.float64)
            out = mathops.add_noise(
                y_raw,
                kind=noise_kind,
                amplitude=amplitude,
                relative=relative,
                seed=seed,
            )
            added = out - ya
            with np.errstate(all="ignore"):
                rms = float(np.sqrt(np.nanmean(added**2))) if added.size else float("nan")
                signal_rms = float(np.sqrt(np.nanmean(ya**2))) if ya.size else float("nan")
            stats = [("Added noise (RMS)", _fmt(rms))]
            if math.isfinite(rms) and rms > 0.0 and math.isfinite(signal_rms):
                stats.append(("Signal / noise (dB)", _fmt(20.0 * math.log10(signal_rms / rms))))
            return _Result(
                x=xa,
                y=out,
                x_name=source.x_name,
                y_name=f"{y_name} + noise",
                suffix="+ noise",
                plot_label=f"with {noise_kind} noise",
                overlay=ya,
                stats=stats,
            )

        if kind == "resample":
            _, n, resample_kind = params
            x_new, y_new = mathops.resample(x_raw, y_raw, n, kind=resample_kind)
            xs, ys = _sorted_by_x(x_raw, y_raw)
            # The overlay must live on the RESULT's grid, since mini_plot draws both
            # series against a single x. Linear interpolation of the source onto x_new is
            # the honest way to say "here is where the original went".
            overlay = np.interp(x_new, xs, ys)
            spacing = float(x_new[1] - x_new[0]) if x_new.size > 1 else float("nan")
            return _Result(
                x=x_new,
                y=y_new,
                x_name=source.x_name,
                y_name=f"{y_name} resampled",
                suffix="resampled",
                plot_label=f"resampled to {len(y_new)} ({resample_kind})",
                overlay=overlay,
                overlay_label="source (interpolated)",
                x_changed=True,
                stats=[
                    ("Samples in", _fmt(float(np.asarray(y_raw).size))),
                    ("Samples out", _fmt(float(y_new.size))),
                    ("New spacing", _fmt(spacing)),
                ],
            )

        if kind == "normalize":
            _, mode = params
            xa = np.asarray(x_raw, dtype=np.float64)
            ya = np.asarray(y_raw, dtype=np.float64)
            out = mathops.normalize(y_raw, mode=mode)
            return _Result(
                x=xa,
                y=out,
                x_name=source.x_name,
                y_name=f"{y_name} normalized",
                suffix="normalized",
                plot_label=f"normalized ({mode})",
                overlay=ya,
                stats=[
                    ("Result minimum", _fmt(_nan_min(out))),
                    ("Result maximum", _fmt(_nan_max(out))),
                ],
            )

        if kind == "filter":
            _, btype, order, frac_lo, frac_hi, family, ripple, attenuation = params
            xs, ys = _sorted_by_x(x_raw, y_raw)
            if xs.size < 2 or float(xs[-1] - xs[0]) <= 0.0:
                raise ValueError("filtering needs at least 2 samples over a non-zero x span")
            nyquist = 0.5 * (xs.size - 1) / float(xs[-1] - xs[0])
            is_band = btype in ("bandpass", "bandstop")
            lo = min(max(frac_lo, 1e-6), 1.0 - 1e-6) * nyquist
            hi = min(max(frac_hi, 1e-6), 1.0 - 1e-6) * nyquist
            cutoff: Any = (lo, hi) if is_band else lo
            out = mathops.iir_filter(
                x_raw,
                y_raw,
                cutoff,
                btype=btype,
                family=family,
                order=int(order),
                ripple=ripple,
                attenuation=attenuation,
            )
            ya = np.asarray(ys, dtype=np.float64)
            residual = ya - out
            with np.errstate(all="ignore"):
                rms = float(np.sqrt(np.nanmean(residual**2))) if residual.size else float("nan")
            stats = [("Family", family), ("Nyquist frequency", _fmt(nyquist))]
            if is_band:
                stats.append(("Low cutoff", _fmt(lo)))
                stats.append(("High cutoff", _fmt(hi)))
            else:
                stats.append(("Cutoff frequency", _fmt(lo)))
            if family == "cheby1":
                stats.append(("Passband ripple", f"{ripple:g} dB"))
            elif family == "cheby2":
                stats.append(("Stopband attenuation", f"{attenuation:g} dB"))
            stats.append(("Removed (RMS)", _fmt(rms)))
            return _Result(
                x=xs,
                y=out,
                x_name=source.x_name,
                y_name=f"{y_name} filtered",
                suffix="filtered",
                plot_label=f"{btype} {family} (order {order})",
                overlay=ya,
                stats=stats,
            )

        if kind == "detrend":
            _, detrend_kind, lam_log10, asym = params
            xa = np.asarray(x_raw, dtype=np.float64)
            ya = np.asarray(y_raw, dtype=np.float64)
            if detrend_kind == "asls":
                baseline = mathops.baseline_asls(y_raw, lam=10.0**lam_log10, p=asym)
                out = ya - baseline
                label = f"baseline-corrected (asls, 1e{lam_log10:.1f})"
            else:
                out = mathops.detrend(x_raw, y_raw, kind=detrend_kind)
                baseline = ya - out
                label = f"baseline-corrected ({detrend_kind})"
            return _Result(
                x=xa,
                y=out,
                x_name=source.x_name,
                y_name=f"{y_name} corrected",
                suffix="baseline-corrected",
                plot_label=label,
                # Show the original signal with the estimated baseline drawn through it, so
                # the user can judge whether the baseline tracks the background; Apply
                # commits the corrected signal (x/y).
                base_x=xa,
                base_y=ya,
                overlay=baseline,
                overlay_label="baseline",
                stats=[
                    ("Result mean", _fmt(_nan_reduce(out, np.nanmean))),
                    (
                        "Baseline at x min",
                        _fmt(float(baseline[0]) if baseline.size else float("nan")),
                    ),
                    (
                        "Baseline at x max",
                        _fmt(float(baseline[-1]) if baseline.size else float("nan")),
                    ),
                ],
            )

        if kind == "peaks":
            _, peak_mode, prominence_frac, distance = params
            is_valleys = peak_mode == "valleys"
            noun = "Valley" if is_valleys else "Peak"
            xs, ys = _sorted_by_x(x_raw, y_raw)
            with np.errstate(all="ignore"):
                span = float(np.nanmax(ys) - np.nanmin(ys)) if ys.size else 0.0
            prominence = prominence_frac * span if span > 0.0 else None
            # Valleys are exactly the same detector run on the negated signal -- no
            # separate algorithm, so the two modes can never disagree on what counts.
            search_y = (-np.asarray(y_raw, dtype=np.float64)) if is_valleys else y_raw
            peak_x, search_peak_y, props = mathops.find_peaks(
                x_raw, search_y, prominence=prominence, distance=distance
            )
            peak_y = -search_peak_y if is_valleys else search_peak_y
            prominences = np.asarray(props.get("prominences", np.empty(0)), dtype=np.float64)
            widths = np.asarray(props.get("widths", np.empty(0)), dtype=np.float64)
            areas = np.asarray(props.get("areas", np.empty(0)), dtype=np.float64)
            stats = [(f"{noun}s found", str(int(peak_x.size)))]
            table: Optional[List[Tuple[str, np.ndarray]]] = None
            if peak_x.size:
                extreme = int(np.argmin(peak_y)) if is_valleys else int(np.argmax(peak_y))
                extreme_label = (
                    f"Deepest {noun.lower()}" if is_valleys else f"Tallest {noun.lower()}"
                )
                value_label = "value" if is_valleys else "height"
                stats.append((f"{extreme_label} at x", _fmt(float(peak_x[extreme]))))
                stats.append((f"{extreme_label} {value_label}", _fmt(float(peak_y[extreme]))))
                if peak_x.size > 1:
                    stats.append(("Mean spacing", _fmt(float(np.mean(np.diff(peak_x))))))
                if widths.size and bool(np.any(np.isfinite(widths))):
                    stats.append(("Mean width", _fmt(_nan_reduce(widths, np.nanmean))))
                if areas.size and bool(np.any(np.isfinite(areas))):
                    stats.append(("Total area", _fmt(float(np.nansum(areas)))))
                    stats.append((f"Largest {noun.lower()} area", _fmt(float(areas[extreme]))))
                # The deliverable of peak analysis: a full table, applied as a new dataset.
                table = [
                    ("position", peak_x),
                    ("height", peak_y),
                    ("prominence", prominences),
                    ("width", widths),
                    ("area", areas),
                ]
            return _Result(
                # x/y are the peak coordinates: that is what a "new layer" Apply plots.
                # base_x/base_y carry the full signal so the preview shows it with the peaks
                # marked; ``table`` is the full per-peak table a "new dataset" Apply writes.
                x=peak_x,
                y=peak_y,
                x_name=source.x_name,
                y_name=f"{y_name} {noun.lower()}s",
                suffix=f"{noun.lower()}s",
                plot_label="signal",
                base_x=np.asarray(xs, dtype=np.float64),
                base_y=np.asarray(ys, dtype=np.float64),
                markers=(peak_x, peak_y),
                table=table,
                # A peak table cannot overwrite the signal it was measured from.
                allow_replace=False,
                # With no peaks there is nothing to plot or store; guard Apply.
                allow_apply=bool(peak_x.size),
                stats=stats,
            )

        if kind == "histogram":
            _, bins, dist_name, density = params
            ya = np.asarray(y_raw, dtype=np.float64)
            edges, heights, centers = mathops.histogram(y_raw, bins=bins, density=density)
            sil_x, sil_y = _histogram_silhouette(edges, heights)
            n_finite = int(np.count_nonzero(np.isfinite(ya)))
            height_name = "density" if density else "count"
            stats = [("Finite samples", str(n_finite)), ("Bins", str(int(heights.size)))]

            overlay: Optional[np.ndarray] = None
            overlay_label = "source"
            if dist_name != "none":
                try:
                    fit = mathops.fit_distribution(y_raw, dist_name, sil_x)
                    pdf = np.asarray(fit["pdf"], dtype=np.float64)
                    if not density:
                        # The PDF integrates to 1; scale it onto the count axis so it sits
                        # on the bars instead of hugging zero.
                        width = float(edges[1] - edges[0]) if edges.size > 1 else 1.0
                        pdf = pdf * n_finite * width
                    overlay = pdf
                    overlay_label = f"{dist_name} fit"
                    for label, value in zip(fit["labels"], fit["params"]):
                        stats.append((label, _fmt(value)))
                    stats.append(("KS statistic", _fmt(fit["ks_statistic"])))
                    stats.append(("KS p-value", _fmt(fit["ks_pvalue"])))
                    stats.append(("Log likelihood", _fmt(fit["log_likelihood"])))
                except ValueError as exc:
                    # A failed or unavailable fit must not take the histogram down with it.
                    stats.append(("Distribution fit", str(exc)))

            return _Result(
                # Apply commits the histogram table (bin centre, height); the preview draws
                # the silhouette (base_*) with the fitted PDF overlaid on the same grid.
                x=centers,
                y=heights,
                x_name=f"{y_name} value",
                y_name=height_name,
                suffix="histogram",
                plot_label="histogram",
                base_x=sil_x,
                base_y=sil_y,
                overlay=overlay,
                overlay_label=overlay_label,
                x_changed=True,
                # A histogram lives in value/count space, not over the signal's own x.
                allow_replace=False,
                stats=stats,
            )

        if kind == "fit":
            (
                _,
                model_key,
                degree,
                expr,
                p0,
                robust,
                loss,
                want_band,
                level,
                use_grid,
                n_points,
                training,
            ) = params
            ya = np.asarray(y_raw, dtype=np.float64)
            xa = np.asarray(x_raw, dtype=np.float64)
            total_rows = int(xa.size)
            used_rows = total_rows
            if training.subsample:
                idx = _subsample_indices(
                    total_rows, training.max_rows, seed=training.subsample_seed
                )
                if idx is not None:
                    xa = xa[idx]
                    ya = ya[idx]
                    used_rows = int(idx.size)
            rows_used_stat: Optional[Tuple[str, str]] = None
            if training.subsample:
                suffix = " (subsampled)" if used_rows < total_rows else ""
                rows_used_stat = ("Rows used", f"{used_rows:,} of {total_rows:,}{suffix}")

            # Train/val/test split (post-subsample): fitting below uses xa_fit/ya_fit,
            # which equal xa/ya themselves (same objects, not copies) when the split is
            # off -- every existing computation that follows is then byte-for-byte
            # unchanged from before this feature existed.
            if training.split:
                train_idx, val_idx, test_idx = _split_indices(
                    int(xa.size), training.val_frac, training.test_frac, seed=training.split_seed
                )
                xa_fit, ya_fit = xa[train_idx], ya[train_idx]
            else:
                train_idx = val_idx = test_idx = np.empty(0, dtype=np.int64)
                xa_fit, ya_fit = xa, ya

            def _output_grid(fn: Callable[..., Any], fit_params: np.ndarray) -> np.ndarray:
                """N evenly spaced points across the x range, evaluated through ``fn``.

                The "how many points should Apply add" control: a fitted model is
                defined everywhere, not just at the input's own samples, so committing
                it at N points of the caller's choosing -- not one point per original
                sample -- is what makes a sparse or unevenly sampled fit usable as a
                smooth curve.
                """
                x_min = float(np.nanmin(xa)) if xa.size else 0.0
                x_max = float(np.nanmax(xa)) if xa.size else 0.0
                grid = np.linspace(x_min, x_max, max(2, int(n_points))) if x_max > x_min else xa
                with np.errstate(all="ignore"):
                    values = np.asarray(fn(grid, *fit_params), dtype=np.float64)
                return grid, np.broadcast_to(values, grid.shape).astype(np.float64, copy=True)

            if model_key == "polynomial":
                coeffs, cov, y_fit = mathops.fit_polynomial_covariance(xa_fit, ya_fit, degree)
                poly_fn = lambda xx, *c: np.polyval(c, xx)  # noqa: E731
                stats: List[Tuple[str, str]] = []
                if rows_used_stat is not None:
                    stats.append(rows_used_stat)
                stats.extend(
                    [
                        ("R squared", _fmt(_r_squared(ya_fit, y_fit))),
                        ("Effective degree", _fmt(float(len(coeffs) - 1))),
                    ]
                )
                # np.polyfit returns highest order first.
                for i, coeff in enumerate(coeffs):
                    power = len(coeffs) - 1 - i
                    name = "constant" if power == 0 else ("x" if power == 1 else f"x^{power}")
                    stats.append((f"Coefficient of {name}", _fmt(float(coeff))))
                if training.split:
                    stats.extend(
                        _held_out_fit_stats(
                            poly_fn, coeffs, xa, ya, train_idx, val_idx, test_idx, len(coeffs)
                        )
                    )

                if use_grid:
                    x_out, y_out = _output_grid(poly_fn, coeffs)
                    overlay_out, markers_out = None, (xa_fit, ya_fit)
                    stats.append(("Output points", str(int(x_out.size))))
                else:
                    x_out, y_out, overlay_out, markers_out = xa_fit, y_fit, ya_fit, None
                band = None
                if want_band:
                    band = mathops.confidence_band(x_out, coeffs, cov, poly_fn, level=level)
                model_state = {
                    "popt": coeffs,
                    "pcov": cov,
                    "model_key": "polynomial",
                    "expr": "",
                    "n_peaks": 0,
                }
                # Residuals diagnostic (Phase 5): actual minus fitted at the source's OWN
                # x -- xa/ya (post-subsample, every row -- not xa_fit/ya_fit, which is
                # train-only when a split is on) -- independent of x_out, which may be a
                # synthetic output grid instead of the source's own samples.
                with np.errstate(all="ignore"):
                    residual_fit = np.polyval(coeffs, xa)
                diagnostic = {"kind": "residuals", "x": xa, "y": ya - residual_fit}
                return _Result(
                    x=x_out,
                    y=y_out,
                    x_name=source.x_name,
                    y_name=f"{y_name} fit",
                    suffix="fit",
                    plot_label=f"polynomial fit (degree {len(coeffs) - 1})",
                    overlay=overlay_out,
                    markers=markers_out,
                    marker_label="data",
                    x_changed=use_grid,
                    stats=stats,
                    band=band,
                    model_state=model_state,
                    diagnostic=diagnostic,
                )

            if not mathops.fit_available():
                raise ValueError(mathops.FIT_UNAVAILABLE_MESSAGE)
            # ``degree`` carries the peak count for the deconvolution models (0 otherwise).
            spec = self._fit_spec(model_key, expr, degree)

            # Stash the heuristic start before fitting, not after: when the fit fails
            # this is exactly the moment the user needs to see what it started from and
            # take it over.
            key = _fit_state_key(model_key, spec)
            try:
                self._fit_auto[key] = [
                    float(v) for v in mathops.initial_guess(xa_fit, ya_fit, spec)
                ]
            except ValueError:
                self._fit_auto.pop(key, None)

            start = np.asarray(p0, dtype=np.float64) if p0 is not None else None
            if robust:
                popt, pcov, y_fit, names = mathops.fit_model_robust(
                    xa_fit, ya_fit, spec, p0=start, loss=loss
                )
            else:
                popt, pcov, y_fit, names = mathops.fit_model(xa_fit, ya_fit, spec, p0=start)
            errors = mathops.fit_errors(pcov)
            # Goodness-of-fit always compares the model to the ACTUAL data it was fit on
            # -- independent of what the output grid below commits, and (with a split
            # on) that is the train partition, not the full working set.
            summary = mathops.fit_statistics(ya_fit, y_fit, len(names))
            stats = []
            if rows_used_stat is not None:
                stats.append(rows_used_stat)
            stats.extend(
                [
                    ("R squared", _fmt(summary["r_squared"])),
                    # No per-point errors are available here, so chi squared carries unit
                    # weights and this is the residual variance, in y units squared -- not
                    # the dimensionless "should be near 1" statistic. Labelled, not implied.
                    ("Reduced chi sq (unit weights)", _fmt(summary["reduced_chi_squared"])),
                    ("Degrees of freedom", _fmt(summary["dof"])),
                    ("RMS residual", _fmt(summary["rmse"])),
                ]
            )
            # The whole point of the tab: a value AND its standard error, per parameter.
            for name, value, error in zip(names, popt, errors):
                stats.append((name, f"{_fmt(float(value))} +/- {_fmt(float(error))}"))
            if model_key in _MULTIPEAK_MODELS:
                # The deconvolution payoff: each fitted peak's area, exact even where the
                # peaks overlap, computed analytically from its amplitude and width.
                shape = _MULTIPEAK_MODELS[model_key]
                peak_areas = [
                    mathops.peak_area(shape, popt[3 * i], popt[3 * i + 2])
                    for i in range((len(names) - 1) // 3)
                ]
                stats.append(("Total peak area", _fmt(float(sum(peak_areas)))))
                for i, area in enumerate(peak_areas, start=1):
                    stats.append((f"Peak {i} area", _fmt(float(area))))
            if training.split:
                stats.extend(
                    _held_out_fit_stats(
                        spec.fn, popt, xa, ya, train_idx, val_idx, test_idx, len(names)
                    )
                )

            if use_grid:
                x_out, y_out = _output_grid(spec.fn, popt)
                overlay_out, markers_out = None, (xa_fit, ya_fit)
                stats.append(("Output points", str(int(x_out.size))))
            else:
                x_out, y_out, overlay_out, markers_out = xa_fit, y_fit, ya_fit, None
            band = None
            if want_band:
                band = mathops.confidence_band(x_out, popt, pcov, spec.fn, level=level)
            plot_label = f"{spec.label} fit" + (" (robust)" if robust else "")
            model_state = {
                "popt": popt,
                "pcov": pcov,
                "model_key": model_key,
                "expr": expr,
                "n_peaks": int(degree) if model_key in _MULTIPEAK_MODELS else 0,
            }
            # Residuals diagnostic (Phase 5): same convention as the polynomial branch
            # above -- actual minus fitted at xa/ya (post-subsample, every row),
            # mirroring _evaluate_model's own named/multipeak/custom reconstruction
            # (spec.fn(x, *popt)) rather than the polynomial special case.
            with np.errstate(all="ignore"):
                residual_fit = np.asarray(spec.fn(xa, *popt), dtype=np.float64)
            diagnostic = {"kind": "residuals", "x": xa, "y": ya - residual_fit}
            return _Result(
                x=x_out,
                y=y_out,
                x_name=source.x_name,
                y_name=f"{y_name} fit",
                suffix="fit",
                plot_label=plot_label,
                overlay=overlay_out,
                markers=markers_out,
                marker_label="data",
                x_changed=use_grid,
                stats=stats,
                band=band,
                model_state=model_state,
                diagnostic=diagnostic,
            )

        if kind == "fft":
            freqs, magnitude = mathops.fft_spectrum(x_raw, y_raw)
            stats = [("Bins", _fmt(float(freqs.size)))]
            # Ignore DC: it is the mean, it is almost always the tallest bin, and calling
            # it "the dominant frequency" would be wrong.
            if magnitude.size > 1:
                ac = magnitude[1:]
                if bool(np.any(np.isfinite(ac))):
                    peak_index = int(np.nanargmax(ac)) + 1
                    stats.append(("Peak frequency", _fmt(float(freqs[peak_index]))))
                    stats.append(("Peak amplitude", _fmt(float(magnitude[peak_index]))))
            if freqs.size:
                stats.append(("Nyquist frequency", _fmt(float(freqs[-1]))))
            return _Result(
                x=freqs,
                y=magnitude,
                x_name="frequency",
                y_name="magnitude",
                suffix="spectrum",
                plot_label="amplitude spectrum",
                overlay=None,
                x_changed=True,
                # The spectrum lives in frequency, not in x: writing it back over the
                # signal it came from would be meaningless.
                allow_replace=False,
                stats=stats,
            )

        if kind == "envelope":
            x_out, env = mathops.envelope(x_raw, y_raw)
            # The original signal, shown as context under the traced envelope. If x had
            # to be resampled onto a uniform grid, interpolate the original onto that
            # same grid so the overlay still lines up sample-for-sample.
            xs, ys = _sorted_by_x(x_raw, y_raw)
            if x_out.shape == xs.shape and np.allclose(x_out, xs):
                overlay_y = ys
            else:
                overlay_y = np.interp(x_out, xs, ys)
            stats = [("Samples", str(int(env.size)))]
            if env.size:
                stats.append(("Peak envelope", _fmt(float(np.max(env)))))
                stats.append(("Mean envelope", _fmt(float(np.mean(env)))))
            return _Result(
                x=x_out,
                y=env,
                x_name=source.x_name,
                y_name=f"{y_name} envelope",
                suffix="envelope",
                plot_label="envelope",
                overlay=overlay_y,
                overlay_label="signal",
                # x may have been resampled onto a uniform grid.
                x_changed=True,
                # A derived analysis curve, not a correction of the signal it traces.
                allow_replace=False,
                stats=stats,
            )

        if kind == "rolling":
            _, roll_stat, window, center = params
            xs, ys = _sorted_by_x(x_raw, y_raw)
            out = mathops.rolling_stat(ys, window, stat=roll_stat, center=center)
            valid = np.isfinite(out)
            n_valid = int(np.count_nonzero(valid))
            stats = [
                ("Samples", str(int(out.size))),
                ("Full-window samples", str(n_valid)),
            ]
            if n_valid:
                stats.append((f"Mean rolling {roll_stat}", _fmt(float(np.mean(out[valid])))))
            return _Result(
                x=xs,
                y=out,
                x_name=source.x_name,
                y_name=f"{y_name} rolling {roll_stat}",
                suffix=f"rolling_{roll_stat}",
                plot_label=f"rolling {roll_stat} (window={window})",
                overlay=ys,
                overlay_label="signal",
                stats=stats,
            )

        if kind == "autocorr":
            _, corr_mode, cross_col, maxlag_frac = params
            xa = np.asarray(x_raw, dtype=np.float64)
            ya = np.asarray(y_raw, dtype=np.float64)
            if xa.size >= 2 and not bool(np.all(np.diff(xa) >= 0.0)):
                order = np.argsort(xa, kind="stable")
            else:
                order = np.arange(xa.size)
            xs = xa[order]
            ys = ya[order]
            n = int(xs.size)
            max_lag = max(1, int(maxlag_frac * (n - 1))) if n > 1 else 0
            dx = float(np.median(np.diff(xs))) if n > 1 else 1.0
            if dx <= 0.0:
                dx = 1.0

            is_cross = corr_mode == "cross" and bool(cross_col) and source.dataset is not None
            if is_cross:
                # Same dataset, same row order as x_raw/y_raw -- the *same* permutation
                # keeps the two columns paired sample-for-sample after sorting.
                cross_raw = np.asarray(source.dataset.get(cross_col), dtype=np.float64)
                cross_sorted = cross_raw[order] if cross_raw.size == xa.size else cross_raw
                lags, corr = mathops.cross_correlation(ys, cross_sorted, max_lag=max_lag)
                lag_x = lags * dx
                stats = [("Lags computed", str(int(lags.size)))]
                markers: Optional[Tuple[np.ndarray, np.ndarray]] = None
                if corr.size:
                    # A delay can lead or lag, unlike a repeating period: report whichever
                    # lag correlates strongest in either direction, not the first peak.
                    extreme = int(np.argmax(np.abs(corr)))
                    markers = (np.array([lag_x[extreme]]), np.array([corr[extreme]]))
                    stats.append(("Peak lag", _fmt(float(lag_x[extreme]))))
                    stats.append(("Peak correlation", _fmt(float(corr[extreme]))))
                y_name_result = f"{source.y_name} x {cross_col}"
                plot_label = f"cross-correlation with {cross_col}"
                suffix = "cross-correlation"
            else:
                lags, corr = mathops.autocorrelation(ys, max_lag=max_lag)
                lag_x = lags * dx
                # The dominant period is the first ACF peak past lag 0; a modest
                # prominence floor keeps noise ripple from being called a period.
                peak_x, peak_y, _props = mathops.find_peaks(lag_x, corr, prominence=0.1)
                stats = [("Lags computed", str(int(lags.size)))]
                markers = None
                if peak_x.size:
                    markers = (peak_x, peak_y)
                    stats.append(("Dominant period", _fmt(float(peak_x[0]))))
                    stats.append(("Correlation at period", _fmt(float(peak_y[0]))))
                else:
                    stats.append(("Dominant period", "no clear period"))
                y_name_result = "autocorrelation"
                plot_label = "autocorrelation"
                suffix = "autocorrelation"

            return _Result(
                x=lag_x,
                y=corr,
                x_name="lag",
                y_name=y_name_result,
                suffix=suffix,
                plot_label=plot_label,
                overlay=None,
                markers=markers,
                x_changed=True,
                # An ACF/XCF lives in lag space, not over the signal's own x.
                allow_replace=False,
                stats=stats,
            )

        if kind == "stats":
            _, show_more, show_ci, ci_level = params
            xa = np.asarray(x_raw, dtype=np.float64)
            ya = np.asarray(y_raw, dtype=np.float64)
            summary = mathops.describe_extended(y_raw) if show_more else mathops.describe(y_raw)
            rows = _DESCRIBE_ROWS + _DESCRIBE_EXTENDED_ROWS if show_more else _DESCRIBE_ROWS
            int_keys = {"n", "nan_count", "outlier_count"}
            stats = [
                (label, str(int(summary[key])) if key in int_keys else _fmt(summary[key]))
                for key, label in rows
            ]
            stats.insert(1, ("Total samples", str(int(ya.size))))
            # No natural (x, y) curve exists for a summary -- one row, one column per
            # statistic, is what New dataset commits instead (point 6: always a way to
            # turn an analysis into data, even when there is nothing to plot).
            table: List[Tuple[str, np.ndarray]] = [
                (label, np.array([float(summary[key])], dtype=np.float64)) for key, label in rows
            ]
            table.insert(1, ("Total samples", np.array([float(ya.size)], dtype=np.float64)))
            if show_ci:
                ci = mathops.confidence_interval_mean(ya, level=ci_level)
                pct = f"{ci_level * 100:.0f}%"
                stats.append((f"{pct} CI for mean", f"[{_fmt(ci['lower'])}, {_fmt(ci['upper'])}]"))
                table.append(("CI lower (mean)", np.array([ci["lower"]], dtype=np.float64)))
                table.append(("CI upper (mean)", np.array([ci["upper"]], dtype=np.float64)))
            return _Result(
                x=xa,
                y=ya,
                x_name=source.x_name,
                y_name=source.y_name,
                suffix="stats",
                plot_label=source.label,
                overlay=None,
                allow_replace=False,
                apply_modes=("new_dataset",),
                stats=stats,
                table=table,
            )

        if kind == "correlate":
            _, show_ci, ci_level = params
            xs, ys = _sorted_by_x(x_raw, y_raw)
            corr = mathops.correlate(xs, ys)
            fit_line = corr["slope"] * xs + corr["intercept"]
            stats = [
                ("n", str(int(corr["n"]))),
                ("Pearson r", _fmt(corr["pearson_r"])),
                ("Pearson p-value", _fmt(corr["pearson_p"])),
                ("Spearman rho", _fmt(corr["spearman_rho"])),
                ("Spearman p-value", _fmt(corr["spearman_p"])),
                ("Covariance", _fmt(corr["covariance"])),
                ("R squared (linear)", _fmt(corr["r_squared"])),
                ("Slope", _fmt(corr["slope"])),
                ("Intercept", _fmt(corr["intercept"])),
            ]
            if show_ci:
                ci = mathops.confidence_interval_correlation(xs, ys, level=ci_level)
                pct = f"{ci_level * 100:.0f}%"
                stats.append(
                    (f"{pct} CI for Pearson r", f"[{_fmt(ci['lower'])}, {_fmt(ci['upper'])}]")
                )
            return _Result(
                x=xs,
                y=ys,
                x_name=source.x_name,
                y_name=source.y_name,
                suffix="correlation",
                plot_label=source.label,
                overlay=fit_line,
                overlay_label="linear fit",
                allow_replace=False,
                stats=stats,
            )

        if kind == "compare":
            _, compare_col, compare_method, show_ci, ci_level = params
            if not compare_col or source.dataset is None:
                raise ValueError(
                    "Compare needs a second column from a dataset source (a plotted "
                    "layer has no table, and no column has been picked yet)."
                )
            a = np.asarray(y_raw, dtype=np.float64)
            b = np.asarray(source.dataset.get(compare_col), dtype=np.float64)
            outcome = mathops.two_sample_test(a, b, method=compare_method)

            a_label, b_label = source.y_name, compare_col
            stats = [
                (f"n ({a_label})", str(int(outcome["n_a"]))),
                (f"n ({b_label})", str(int(outcome["n_b"]))),
                (f"Mean ({a_label})", _fmt(outcome["mean_a"])),
                (f"Mean ({b_label})", _fmt(outcome["mean_b"])),
                (f"Std ({a_label})", _fmt(outcome["std_a"])),
                (f"Std ({b_label})", _fmt(outcome["std_b"])),
                ("Statistic", _fmt(outcome["statistic"])),
                ("p-value", _fmt(outcome["p_value"])),
            ]
            # Always Welch's CI for the mean gap, independent of the hypothesis-test
            # method picked above: mannwhitney/ks test the full distribution, not the
            # mean, so there is no matching closed-form interval for them to switch to.
            diff_ci = (
                mathops.confidence_interval_difference(a, b, level=ci_level) if show_ci else None
            )
            if diff_ci is not None:
                pct = f"{ci_level * 100:.0f}%"
                stats.append(
                    (
                        f"{pct} CI for mean diff ({a_label} - {b_label})",
                        f"[{_fmt(diff_ci['lower'])}, {_fmt(diff_ci['upper'])}]",
                    )
                )

            # No natural (x, y) curve exists for two independent samples -- overlay
            # each sample's own histogram (density, shared bin edges so the two share
            # one x axis) so the comparison is something to actually look at, not just
            # a table of numbers.
            n_bins = 30
            finite_a = a[np.isfinite(a)]
            finite_b = b[np.isfinite(b)]
            lo = float(min(np.min(finite_a), np.min(finite_b)))
            hi = float(max(np.max(finite_a), np.max(finite_b)))
            if hi <= lo:
                hi = lo + 1.0
            edges = np.linspace(lo, hi, n_bins + 1)
            heights_a, _e = np.histogram(finite_a, bins=edges, density=True)
            heights_b, _e = np.histogram(finite_b, bins=edges, density=True)
            sil_x, sil_a = _histogram_silhouette(edges, heights_a)
            _sil_x2, sil_b = _histogram_silhouette(edges, heights_b)

            # Same reasoning as Stats: two independent samples have no (x, y) curve to
            # commit, but the test's own numbers -- means, spread, the statistic, the
            # p-value -- are exactly what New dataset should turn into one row.
            table: List[Tuple[str, np.ndarray]] = [
                (f"n ({a_label})", np.array([float(outcome["n_a"])], dtype=np.float64)),
                (f"n ({b_label})", np.array([float(outcome["n_b"])], dtype=np.float64)),
                (f"Mean ({a_label})", np.array([float(outcome["mean_a"])], dtype=np.float64)),
                (f"Mean ({b_label})", np.array([float(outcome["mean_b"])], dtype=np.float64)),
                (f"Std ({a_label})", np.array([float(outcome["std_a"])], dtype=np.float64)),
                (f"Std ({b_label})", np.array([float(outcome["std_b"])], dtype=np.float64)),
                ("Statistic", np.array([float(outcome["statistic"])], dtype=np.float64)),
                ("p-value", np.array([float(outcome["p_value"])], dtype=np.float64)),
            ]
            if diff_ci is not None:
                table.append(
                    ("CI lower (mean diff)", np.array([diff_ci["lower"]], dtype=np.float64))
                )
                table.append(
                    ("CI upper (mean diff)", np.array([diff_ci["upper"]], dtype=np.float64))
                )
            return _Result(
                x=sil_x,
                y=sil_a,
                x_name="value",
                y_name=f"{a_label} density",
                suffix="comparison",
                plot_label=a_label,
                overlay=sil_b,
                overlay_label=b_label,
                x_changed=True,
                allow_replace=False,
                apply_modes=("new_dataset",),
                stats=stats,
                table=table,
            )

        if kind == "cluster":
            _, cluster_method, k, extra, training = params
            xa = np.asarray(x_raw, dtype=np.float64)
            ya = np.asarray(y_raw, dtype=np.float64)
            total_rows = int(xa.size)
            used_rows = total_rows
            if training.subsample:
                idx = _subsample_indices(
                    total_rows, training.max_rows, seed=training.subsample_seed
                )
                if idx is not None:
                    xa = xa[idx]
                    ya = ya[idx]
                    used_rows = int(idx.size)
            if training.split:
                train_idx, val_idx, test_idx = _split_indices(
                    int(xa.size), training.val_frac, training.test_frac, seed=training.split_seed
                )
                xa_fit, ya_fit = xa[train_idx], ya[train_idx]
                if cluster_method == "hierarchical":
                    _labels_fit, centroid_x, centroid_y = mathops2d.hierarchical_cluster(
                        xa_fit, ya_fit, k, method=extra
                    )
                    method_label = f"hierarchical ({extra})"
                else:
                    _labels_fit, centroid_x, centroid_y = mathops2d.kmeans_cluster(
                        xa_fit, ya_fit, k, seed=extra
                    )
                    method_label = "k-means"
                # Every post-subsample point (train + held-out) is coloured/tabled here,
                # assigned to the TRAIN-fitted centroids via nearest-centroid -- k-means's
                # own assignments reduce to exactly this rule at convergence (see
                # mathops2d.nearest_centroid), and it is the accepted approximation for
                # hierarchical, which has no native out-of-sample rule at all.
                labels = mathops2d.nearest_centroid(xa, ya, centroid_x, centroid_y)
            else:
                # Unchanged path: fit directly on every post-subsample point, exactly as
                # before this feature existed.
                if cluster_method == "hierarchical":
                    labels, centroid_x, centroid_y = mathops2d.hierarchical_cluster(
                        xa, ya, k, method=extra
                    )
                    method_label = f"hierarchical ({extra})"
                else:
                    labels, centroid_x, centroid_y = mathops2d.kmeans_cluster(xa, ya, k, seed=extra)
                    method_label = "k-means"
            n_clusters = int(centroid_x.size)
            stats = [("Method", method_label)]
            if training.subsample:
                suffix = " (subsampled)" if used_rows < total_rows else ""
                stats.append(("Rows used", f"{used_rows:,} of {total_rows:,}{suffix}"))
            stats.append(("Clusters found", str(n_clusters)))
            unclustered = int(np.count_nonzero(labels < 0.0))
            if unclustered:
                stats.append(("Unclustered (non-finite) points", str(unclustered)))
            for i in range(n_clusters):
                stats.append((f"Cluster {i} size", str(int(np.count_nonzero(labels == float(i))))))
            if training.split:
                for label, idx in (("Train", train_idx), ("Val", val_idx), ("Test", test_idx)):
                    if idx.size < 2:
                        stats.append((f"{label} inertia", "not enough held-out points"))
                        stats.append((f"{label} silhouette", "not enough held-out points"))
                        continue
                    xs_split, ys_split = xa[idx], ya[idx]
                    labels_split = mathops2d.nearest_centroid(
                        xs_split, ys_split, centroid_x, centroid_y
                    )
                    inertia = mathops2d.cluster_inertia(
                        xs_split, ys_split, centroid_x, centroid_y, labels_split
                    )
                    stats.append((f"{label} inertia", _fmt(inertia)))
                    sil = mathops2d.silhouette_score(xs_split, ys_split, labels_split)
                    stats.append((f"{label} silhouette", _fmt(sil)))
            model_state = {
                "method": cluster_method,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
            }
            return _Result(
                x=xa,
                y=ya,
                x_name=source.x_name,
                y_name=source.y_name,
                suffix="cluster",
                plot_label=source.label,
                overlay=None,
                allow_replace=False,
                stats=stats,
                color_values=labels,
                markers=(centroid_x, centroid_y),
                table=[(source.x_name, xa), (source.y_name, ya), ("cluster", labels)],
                model_state=model_state,
            )

        if kind == "density2d":
            _, mode, bins, grid, custom_bw, bw_factor = params
            xa = np.asarray(x_raw, dtype=np.float64)
            ya = np.asarray(y_raw, dtype=np.float64)
            if mode == "kde":
                bw_method = float(bw_factor) if custom_bw else None
                gx, gy, gz = mathops2d.density2d(xa, ya, mode="kde", grid=grid, bw_method=bw_method)
            else:
                gx, gy, gz = mathops2d.density2d(xa, ya, mode="histogram", bins=bins)
            n_finite = int(np.count_nonzero(np.isfinite(xa) & np.isfinite(ya)))
            peak = float(np.max(gz)) if gz.size else 0.0
            stats = [
                ("Mode", mode),
                ("Points", str(n_finite)),
                ("Grid cells", f"{gz.shape[0]} x {gz.shape[1]}"),
                ("Peak density" if mode == "kde" else "Peak count", _fmt(peak)),
            ]
            return _Result(
                x=xa,
                y=ya,
                x_name=source.x_name,
                y_name=source.y_name,
                suffix="density",
                plot_label=source.label,
                overlay=None,
                allow_replace=False,
                stats=stats,
                grid_x=gx,
                grid_y=gy,
                grid_z=gz,
            )

        if kind == "spatial":
            xa = np.asarray(x_raw, dtype=np.float64)
            ya = np.asarray(y_raw, dtype=np.float64)
            spatial, hull_x, hull_y = mathops2d.spatial_stats(xa, ya)
            stats = [
                ("Points", str(int(spatial["n"]))),
                ("Bounding box width", _fmt(spatial["bbox_width"])),
                ("Bounding box height", _fmt(spatial["bbox_height"])),
            ]
            if "mean_nn_distance" in spatial:
                stats.append(("Mean nearest-neighbor distance", _fmt(spatial["mean_nn_distance"])))
                stats.append(
                    ("Median nearest-neighbor distance", _fmt(spatial["median_nn_distance"]))
                )
                stats.append(("Min nearest-neighbor distance", _fmt(spatial["min_nn_distance"])))
                stats.append(("Max nearest-neighbor distance", _fmt(spatial["max_nn_distance"])))
            table: Optional[List[Tuple[str, np.ndarray]]] = None
            if "hull_area" in spatial:
                stats.append(("Convex hull area", _fmt(spatial["hull_area"])))
                stats.append(("Convex hull perimeter", _fmt(spatial["hull_perimeter"])))
                stats.append(("Convex hull vertices", str(int(spatial["hull_vertex_count"]))))
                table = [("hull_x", hull_x), ("hull_y", hull_y)]
            return _Result(
                x=xa,
                y=ya,
                x_name=source.x_name,
                y_name=source.y_name,
                suffix="spatial",
                plot_label=source.label,
                overlay=None,
                allow_replace=False,
                force_scatter=True,
                markers=(hull_x, hull_y) if hull_x is not None else None,
                table=table,
                stats=stats,
            )

        if kind == "dynamics":
            _, mode, delay, dim = params
            xs, ys = _sorted_by_x(x_raw, y_raw)

            if mode == "phase":
                emb = dynamics.delay_embedding(ys, delay, dim=2)
                a, b = emb[:, 0], emb[:, 1]
                return _Result(
                    x=a,
                    y=b,
                    x_name=f"{y_name}(t)",
                    y_name=f"{y_name}(t+{delay})",
                    suffix="phase",
                    plot_label=f"delay embedding (delay {delay})",
                    overlay=None,
                    x_changed=True,
                    # A reconstructed phase portrait lives in delay space, not over x.
                    allow_replace=False,
                    stats=[
                        ("Embedded points", str(int(a.size))),
                        ("Delay (samples)", str(int(delay))),
                    ],
                )

            if mode == "return_map":
                m_n, m_next = dynamics.return_map(ys)
                stats = [("Maxima found", str(int(m_n.size) + (1 if m_n.size else 0)))]
                diag_x = diag_y = np.empty(0, dtype=np.float64)
                if m_n.size:
                    both = np.concatenate([m_n, m_next])
                    lo, hi = float(np.min(both)), float(np.max(both))
                    # The y = x reference line: on a return map, points above it are growing
                    # maxima and points below are shrinking ones.
                    diag_x = np.array([lo, hi], dtype=np.float64)
                    diag_y = diag_x.copy()
                    stats.append(("Map points", str(int(m_n.size))))
                return _Result(
                    # x/y are the map points (what New layer/dataset commits); the preview
                    # draws the y = x reference with the points marked on it.
                    x=m_n,
                    y=m_next,
                    x_name=f"{y_name} max (n)",
                    y_name=f"{y_name} max (n+1)",
                    suffix="return map",
                    plot_label="y = x",
                    base_x=diag_x,
                    base_y=diag_y,
                    markers=(m_n, m_next) if m_n.size else None,
                    table=[("max_n", m_n), ("max_n_plus_1", m_next)] if m_n.size else None,
                    x_changed=True,
                    allow_replace=False,
                    allow_apply=bool(m_n.size),
                    stats=stats,
                )

            if mode == "lyapunov":
                dx = float(np.median(np.diff(xs))) if xs.size > 1 else 1.0
                if not math.isfinite(dx) or dx <= 0.0:
                    dx = 1.0
                res = dynamics.lyapunov_rosenstein(ys, dt=dx, delay=delay, dim=dim)
                times = np.asarray(res["times"], dtype=np.float64)
                divergence = np.asarray(res["divergence"], dtype=np.float64)
                fit = np.asarray(res["fit"], dtype=np.float64)
                lam = float(res["lyapunov"])
                verdict = (
                    "chaotic (sensitive)"
                    if lam > 0.01
                    else "regular (cycle / fixed point)" if math.isfinite(lam) else "undetermined"
                )
                stats = [
                    ("Largest Lyapunov exponent", _fmt(lam)),
                    ("Verdict", verdict),
                    ("Embedding dim", str(int(dim))),
                    ("Delay (samples)", str(int(delay))),
                    ("Embedded points", str(int(res["n_points"]))),
                ]
                if int(res["stride"]) > 1:
                    stats.append(("Decimation stride", str(int(res["stride"]))))
                return _Result(
                    x=times,
                    y=divergence,
                    x_name="divergence time",
                    y_name="mean log divergence",
                    suffix="lyapunov divergence",
                    plot_label="divergence curve",
                    # The fitted line whose slope is the exponent, over the linear region.
                    overlay=fit,
                    overlay_label="linear fit",
                    x_changed=True,
                    allow_replace=False,
                    stats=stats,
                )

            raise ValueError(f"unknown dynamics analysis {mode!r}")

        if kind == "pca":
            _, columns, n_components, scale, training = params
            if source.dataset is None:
                raise ValueError("PCA needs a dataset source (a plotted layer has no table).")
            if len(columns) < 2:
                raise ValueError("PCA needs at least 2 columns selected.")
            arrays = [source.dataset.get(name) for name in columns]
            total_rows = int(len(arrays[0])) if arrays else 0
            if training.subsample:
                idx = _subsample_indices(
                    total_rows, training.max_rows, seed=training.subsample_seed
                )
                if idx is not None:
                    arrays = [np.asarray(a)[idx] for a in arrays]

            if training.split:
                n_post_subsample = int(len(arrays[0])) if arrays else 0
                train_idx, val_idx, test_idx = _split_indices(
                    n_post_subsample,
                    training.val_frac,
                    training.test_frac,
                    seed=training.split_seed,
                )
                train_arrays = [np.asarray(a)[train_idx] for a in arrays]
                # Fit on train only; project every post-subsample row (train + held-out)
                # through that fit via pca_transform -- the out-of-sample half -- rather
                # than a fresh pca() over everything, so the preview/table cover every
                # row without leaking val/test into the fit itself.
                out = mathopsnd.pca(train_arrays, n_components=n_components, scale=scale)
                scores = mathopsnd.pca_transform(
                    arrays,
                    mean_=out["mean_"],
                    components_=out["components"],
                    scale_stats=out["scale_stats_"],
                    scale=scale,
                )
            else:
                # Unchanged path: fit directly on every post-subsample row, exactly as
                # before this feature existed.
                out = mathopsnd.pca(arrays, n_components=n_components, scale=scale)
                scores = out["scores"]
            explained = out["explained_variance_ratio"]
            # "Rows used" is the technique's own finite-row count (out["n_samples"]) --
            # after sub-sampling too, since that ran before pca() saw the data. Only the
            # *format* changes when sub-sampling is on: "of <total>" and the annotation
            # are meaningless noise on the common (off) path, so that path's value stays
            # byte-identical to before this feature existed.
            used_rows = int(out["n_samples"])
            if training.subsample:
                suffix = " (subsampled)" if used_rows < total_rows else ""
                rows_used_value = f"{used_rows:,} of {total_rows:,}{suffix}"
            else:
                rows_used_value = str(used_rows)
            stats = [
                ("Method", "PCA (SVD)"),
                ("Columns used", ", ".join(columns)),
                ("Rows used", rows_used_value),
                ("Scale", scale),
            ]
            for i, ratio in enumerate(explained, start=1):
                stats.append((f"PC{i} explained variance", f"{float(ratio) * 100:.1f}%"))
            stats.append(
                ("Cumulative explained variance", f"{float(np.sum(explained)) * 100:.1f}%")
            )
            if training.split:
                # Reconstruction error per split: scale (with the FITTED stats) and
                # center each split's own rows, project through the train-fitted
                # components, reconstruct back (scores @ components + mean_), and report
                # the mean squared error between the scaled input and that
                # reconstruction -- the same metric, computed the same way, for every
                # split including train, so the three numbers are directly comparable.
                components_arr = np.asarray(out["components"], dtype=np.float64)
                mean_arr = np.asarray(out["mean_"], dtype=np.float64)
                for label, idx in (("Train", train_idx), ("Val", val_idx), ("Test", test_idx)):
                    if idx.size < 1:
                        stats.append((f"{label} reconstruction error", "not enough points"))
                        continue
                    split_arrays = [np.asarray(a)[idx] for a in arrays]
                    m_split = _finite_column_matrix(split_arrays)
                    if m_split.shape[0] < 1:
                        stats.append((f"{label} reconstruction error", "not enough points"))
                        continue
                    scaled_split = mathopsnd.apply_scale(m_split, out["scale_stats_"], mode=scale)
                    centered_split = scaled_split - mean_arr
                    scores_split = centered_split @ components_arr.T
                    recon_split = scores_split @ components_arr + mean_arr
                    mse = float(np.mean((scaled_split - recon_split) ** 2))
                    stats.append((f"{label} reconstruction error", _fmt(mse)))
            table = [(f"PC{i + 1}", scores[:, i]) for i in range(scores.shape[1])]
            model_state = {
                "mean_": out["mean_"],
                "components_": out["components"],
                "scale_stats_": out["scale_stats_"],
                "scale": scale,
                "columns": tuple(columns),
            }
            # Scree diagnostic (Phase 5): the FULL explained-variance spectrum (every
            # singular value, not just the n_components kept) plus which count was
            # actually requested, for marking on the plot.
            diagnostic = {
                "kind": "scree",
                "ratios": out["explained_variance_ratio_full"],
                "chosen": int(n_components),
            }
            return _Result(
                x=scores[:, 0],
                y=scores[:, 1],
                x_name="PC1",
                y_name="PC2",
                suffix="pca",
                plot_label=f"PCA ({', '.join(columns)})",
                overlay=None,
                allow_replace=False,
                apply_modes=("new_layer", "new_dataset"),
                force_scatter=True,
                table=table,
                stats=stats,
                model_state=model_state,
                diagnostic=diagnostic,
            )

        if kind == "umap":
            _, columns, n_components, n_neighbors, min_dist, scale, seed, training = params
            if source.dataset is None:
                raise ValueError("UMAP needs a dataset source (a plotted layer has no table).")
            if len(columns) < 2:
                raise ValueError("UMAP needs at least 2 columns selected.")
            arrays = [source.dataset.get(name) for name in columns]
            total_rows = int(len(arrays[0])) if arrays else 0
            if training.subsample:
                idx = _subsample_indices(
                    total_rows, training.max_rows, seed=training.subsample_seed
                )
                if idx is not None:
                    arrays = [np.asarray(a)[idx] for a in arrays]

            train_idx = val_idx = test_idx = np.empty(0, dtype=np.int64)
            fit_arrays = arrays
            if training.split:
                n_post_subsample = int(len(arrays[0])) if arrays else 0
                train_idx, val_idx, test_idx = _split_indices(
                    n_post_subsample,
                    training.val_frac,
                    training.test_frac,
                    seed=training.split_seed,
                )
                # Fit (and embed, for the preview) on train only -- UMAP has no closed
                # form to project new points through cheaply the way PCA does, so
                # phase 1 does not embed the held-out rows at all; a later phase adds a
                # visual (embedding-coloured-by-split) read using the reducer's own
                # .transform().
                fit_arrays = [np.asarray(a)[train_idx] for a in arrays]

            out = mathopsnd.umap_embed(
                fit_arrays,
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                scale=scale,
                seed=seed,
            )
            emb = out["embedding"]
            # See the PCA branch above: same "Rows used" convention.
            used_rows = int(out["n_samples"])
            if training.subsample:
                suffix = " (subsampled)" if used_rows < total_rows else ""
                rows_used_value = f"{used_rows:,} of {total_rows:,}{suffix}"
            else:
                rows_used_value = str(used_rows)
            stats = [
                ("Method", "UMAP"),
                ("Columns used", ", ".join(columns)),
                ("Rows used", rows_used_value),
                ("Neighbors used", str(int(out["n_neighbors_used"]))),
                ("Min distance", _fmt(min_dist)),
                ("Scale", scale),
                ("Seed", str(int(seed))),
            ]
            if training.split:
                # No fabricated quality score -- UMAP has no reconstruction error or
                # out-of-sample metric this project is willing to present as one. Row
                # counts are the honest, purely descriptive thing to report here.
                stats.append(("Train rows", str(int(train_idx.size))))
                stats.append(("Val rows", str(int(val_idx.size))))
                stats.append(("Test rows", str(int(test_idx.size))))
            table = [(f"UMAP{i + 1}", emb[:, i]) for i in range(emb.shape[1])]
            model_state = {
                "reducer": out["reducer"],
                "scale_stats_": out["scale_stats_"],
                "scale": scale,
                "columns": tuple(columns),
            }
            return _Result(
                x=emb[:, 0],
                y=emb[:, 1],
                x_name="UMAP1",
                y_name="UMAP2",
                suffix="umap",
                plot_label=f"UMAP ({', '.join(columns)})",
                overlay=None,
                allow_replace=False,
                apply_modes=("new_layer", "new_dataset"),
                force_scatter=True,
                table=table,
                stats=stats,
                model_state=model_state,
            )

        raise ValueError(f"unknown operation {kind!r}")

    # -- preview -------------------------------------------------------------------

    def _preview_style(self) -> Tuple[Any, Any, Tuple[Any, ...]]:
        """(background, grid, palette) read from the live plot -- not the ImGui theme.

        The preview used to borrow generic chrome colors from ``theme.py``, which look
        nothing like the actual figure. This reads ground truth off ``plot.options``
        (always correct, however the current look was reached) for background/grid, and
        the current :class:`~glplot.gui.styles.PlotStyle`'s palette (keyed by
        ``plot._style_key``, which :func:`styles.apply_style` now keeps in sync for both
        the ``gplt.plot_style()`` API and a GUI preset click) for series colors.

        Every lookup is defensive: headless tests and stub plots may not carry
        ``.options``/``.visual`` at all, and this must never raise from a draw callback.
        """
        plot = self.plot
        options = getattr(plot, "options", None)
        visual = getattr(options, "visual", None)
        background = tuple(getattr(visual, "background_color", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))

        grid_raw = getattr(options, "axis_grid_color", None)
        if grid_raw is None or tuple(grid_raw) == styles.AUTO_GRID_COLOR:
            grid = styles.auto_grid_color(background)
        else:
            grid = tuple(grid_raw)

        style_key = str(getattr(plot, "_style_key", "") or "") or styles.DEFAULT_STYLE_KEY
        try:
            palette = styles.get_style(style_key).palette
        except KeyError:
            palette = styles.get_style(styles.DEFAULT_STYLE_KEY).palette
        return background, grid, palette

    def _draw_preview(
        self, source: _Source, result: Optional[_Result], error: Optional[str]
    ) -> None:
        """The live overlay: source and result on one set of axes ("sobreponer")."""
        if error:
            widgets.error_box(error)

        if result is None:
            background, grid, _palette = self._preview_style()
            widgets.mini_plot(
                "preview",
                source.y_raw,
                x=source.x_raw,
                height=_PREVIEW_HEIGHT,
                label=source.label,
                color=theme.get_color("muted"),
                background_color=background,
                grid_color=grid,
            )
            return

        if result.allow_apply:
            # Nothing on screen said the curve below is uncommitted. ASCII only: the
            # default font is ProggyClean and stops at U+00FF, so no em dash (CONTRACT 3).
            imgui.text_disabled("Preview -- not applied")
            imgui.same_line()
            widgets.help_marker(
                "This curve is computed but not committed. Nothing is added to the scene "
                "or to your data until you press Apply at the bottom of the panel."
            )

        self._draw_result_preview(result, height=_PREVIEW_HEIGHT, id_prefix="preview")

    def _draw_result_preview(
        self, result: _Result, *, height: float, id_prefix: str = "preview"
    ) -> None:
        """Dispatch ``result`` to the widget its shape calls for -- scatter, heatmap or
        line -- plus its matching legend. Extracted from :meth:`_draw_preview` so a
        trained-model rail row (Phase 3) can render the exact ``_Result`` a "Save as
        model" click captured, at a different (smaller) height, without recomputing it.

        ``id_prefix`` becomes the widget's own ``id_str``, so a rail thumbnail's ImGui
        ids never collide with the live preview's own ``"preview"``-prefixed ones when
        both are on screen in the same frame.
        """
        background, grid, palette = self._preview_style()
        line_color = palette[0 % len(palette)]
        overlay_color = palette[1 % len(palette)]

        if result.color_values is not None or result.force_scatter:
            # An unordered point cloud (Cluster, coloured by category; Spatial, plain),
            # not a signal: a scatter, not a connected line -- mini_plot's polyline
            # would draw meaningless zig-zags between points with no natural x order.
            widgets.mini_scatter(
                id_prefix,
                result.x,
                result.y,
                point_colors=result.color_values,
                palette=palette,
                color=None if result.color_values is not None else line_color,
                markers=result.markers,
                marker_color=theme.get_color("warn"),
                height=height,
                label=result.plot_label,
                background_color=background,
                grid_color=grid,
            )
            if result.color_values is not None:
                self._draw_cluster_legend(result, palette)
            return

        if result.grid_z is not None:
            # A density grid (Density 2D): filled cells, not a connected line or a
            # scatter -- mini_heatmap is the one widget that draws a 2-D grid.
            widgets.mini_heatmap(
                id_prefix,
                result.grid_z,
                result.grid_x,
                result.grid_y,
                height=height,
                background_color=background,
                grid_color=grid,
                overlay_points=(result.x, result.y),
                label=result.plot_label,
            )
            return

        # base_x/base_y override the drawn line (Peaks shows the signal, not its result).
        line_y = result.base_y if result.base_y is not None else result.y
        line_x = result.base_x if result.base_x is not None else result.x
        widgets.mini_plot(
            id_prefix,
            line_y,
            x=line_x,
            overlay=result.overlay,
            markers=result.markers,
            marker_color=theme.get_color("warn"),
            height=height,
            label=result.plot_label,
            color=line_color,
            overlay_color=overlay_color,
            background_color=background,
            grid_color=grid,
            band=result.band,
            band_color=overlay_color,
        )
        self._draw_legend(result, line_color, overlay_color)

    def _draw_legend(
        self, result: _Result, line_color: Any = None, overlay_color: Any = None
    ) -> None:
        """A colour key for the preview series, matching the colors mini_plot just drew."""
        imgui.text_colored(line_color or theme.get_color("accent"), "--")
        imgui.same_line()
        imgui.text(result.plot_label)
        if result.overlay is not None:
            imgui.same_line()
            imgui.text_colored(overlay_color or theme.get_color("warn"), "--")
            imgui.same_line()
            imgui.text(result.overlay_label)
        if result.markers is not None:
            imgui.same_line()
            imgui.text_colored(theme.get_color("warn"), "o")
            imgui.same_line()
            imgui.text(f"{result.marker_label} ({int(np.asarray(result.markers[0]).size)})")

    def _draw_cluster_legend(self, result: _Result, palette: Tuple[Any, ...]) -> None:
        """A colour key for the cluster scatter: one swatch and point count per cluster."""
        labels = np.asarray(result.color_values)
        unique = sorted({int(v) for v in labels.tolist() if v >= 0.0})
        for idx in unique:
            color = palette[idx % len(palette)] if palette else theme.get_color("accent")
            imgui.text_colored(color, "o")
            imgui.same_line()
            count = int(np.count_nonzero(labels == float(idx)))
            imgui.text(f"cluster {idx} ({count} pts)")
        if result.markers is not None and np.asarray(result.markers[0]).size:
            imgui.text_colored(theme.get_color("warn"), "o")
            imgui.same_line()
            imgui.text("centroids")

    def _draw_diagnostic(self, result: _Result) -> None:
        """A separate, clearly-labeled mini-plot for ``result.diagnostic`` (Phase 5),
        drawn below the main preview/stats. Collapsible, same convention as
        ``_sampling_controls``, so a tab with a diagnostic to show does not force it on
        someone who never opens the section.
        """
        diag = result.diagnostic
        if diag is None:
            return
        kind = diag.get("kind")
        background, grid, palette = self._preview_style()
        line_color = palette[0 % len(palette)] if palette else None

        if kind == "residuals":
            if not widgets.section("Residuals", default_open=False):
                return
            imgui.text_disabled("Actual minus fitted, at every (post-subsample) source point.")
            widgets.mini_plot(
                "fit_residuals",
                diag["y"],
                x=diag["x"],
                height=90.0,
                label="Residuals",
                color=line_color,
                background_color=background,
                grid_color=grid,
            )
            return

        if kind == "scree":
            if not widgets.section("Scree plot", default_open=False):
                return
            ratios = np.asarray(diag["ratios"], dtype=np.float64)
            chosen = int(diag["chosen"])
            xs = np.arange(1, ratios.size + 1, dtype=np.float64)
            markers = None
            if 1 <= chosen <= ratios.size:
                markers = ([float(chosen)], [float(ratios[chosen - 1])])
            widgets.mini_plot(
                "pca_scree",
                ratios,
                x=xs,
                markers=markers,
                marker_color=theme.get_color("warn"),
                height=90.0,
                label="Explained variance by component",
                color=line_color,
                background_color=background,
                grid_color=grid,
            )
            return

    # -- UMAP "color by column" preview (Phase 5) ------------------------------------
    # Presentation only: none of this touches _compute, model_state, or Save-as-model
    # -- it only changes how the ALREADY-COMPUTED embedding in `result` is drawn.

    def _umap_row_indices(
        self, source: _Source, key: str, embed_columns: Tuple[str, ...]
    ) -> np.ndarray:
        """Which of ``source.dataset``'s row indices feed the UMAP embedding currently
        shown for tab ``key``, recomputed deterministically from the panel's own
        sampling/split state -- no new data threaded through ``_Result`` for this.

        Mirrors ``_compute``'s own "umap" branch's row selection exactly: sub-sample,
        then reduce to the train partition only if a split is on, then drop any row
        non-finite in any of ``embed_columns`` -- the same finite-row filter
        ``mathopsnd.umap_embed`` applies internally (``_as_matrix`` + ``np.isfinite``),
        reproduced here so the returned indices line up with the embedding's own rows
        one for one, including when that filter dropped some.
        """
        dataset = source.dataset
        n_total = int(dataset.n_rows()) if dataset is not None else 0
        base = np.arange(n_total)
        if self._sampling_enabled.get(key, False):
            max_rows = int(self._sampling_max_rows.get(key, _NO_SAMPLING.max_rows))
            seed = int(self._sampling_seed.get(key, _NO_SAMPLING.subsample_seed))
            idx = _subsample_indices(n_total, max_rows, seed=seed)
            if idx is not None:
                base = idx

        if self._split_enabled.get(key, False):
            val_frac = float(self._split_val_frac.get(key, _NO_SAMPLING.val_frac))
            test_frac = float(self._split_test_frac.get(key, _NO_SAMPLING.test_frac))
            split_seed = int(self._split_seed.get(key, _NO_SAMPLING.split_seed))
            train_idx, _val_idx, _test_idx = _split_indices(
                int(base.size), val_frac, test_frac, seed=split_seed
            )
            base = base[train_idx]

        if not embed_columns or dataset is None or base.size == 0:
            return base
        arrays = [np.asarray(dataset.get(name), dtype=np.float64)[base] for name in embed_columns]
        matrix = np.column_stack(arrays)
        good = np.all(np.isfinite(matrix), axis=1)
        return base[good]

    def _umap_color_values(
        self, source: _Source, key: str, result: _Result, column: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Bucket ``column``'s values, via ``np.unique(..., return_inverse=True)``, for
        exactly the rows ``result``'s embedding used.

        Returns ``(categories, bucketed)`` -- ``categories`` the unique values (for a
        legend), ``bucketed`` an int array the same length as ``result.x``/``result.y``
        -- or None if the source has no dataset, ``result`` has no ``model_state``
        (should not happen for a real UMAP result), or (a defensive fallback) the
        recomputed row count does not match the embedding's own, in which case coloring
        is skipped rather than raising or silently misaligning the two arrays.
        """
        dataset = source.dataset
        if dataset is None or result.model_state is None:
            return None
        values = dataset.get(column)
        if values is None:
            return None
        embed_columns = tuple(result.model_state.get("columns", ()))
        row_idx = self._umap_row_indices(source, key, embed_columns)
        n_embedding = int(np.asarray(result.x).size)
        if row_idx.size != n_embedding:
            return None
        picked = np.asarray(values)[row_idx]
        categories, bucketed = np.unique(picked, return_inverse=True)
        return categories, bucketed.astype(np.int64)

    def _draw_umap_color_legend(self, categories: np.ndarray, palette: Tuple[Any, ...]) -> None:
        """A colour key for the UMAP "color by column" scatter: one swatch per category
        VALUE (unlike :meth:`_draw_cluster_legend`, whose categories are anonymous
        cluster indices, these are the picked column's own values)."""
        for i, value in enumerate(categories.tolist()):
            color = palette[i % len(palette)] if palette else theme.get_color("accent")
            imgui.text_colored(color, "o")
            imgui.same_line()
            imgui.text(str(value))

    def _draw_umap_preview(self, source: _Source, result: _Result, key: str) -> None:
        """The UMAP tab's own preview: the normal embedding scatter, plus a "Color by"
        column pick (Phase 5) that recolors it in place. Replaces the generic
        :meth:`_draw_preview` call for this one tab; every other tab is unaffected.
        """
        if result.allow_apply:
            imgui.text_disabled("Preview -- not applied")
            imgui.same_line()
            widgets.help_marker(
                "This curve is computed but not committed. Nothing is added to the scene "
                "or to your data until you press Apply at the bottom of the panel."
            )

        current = self._umap_color_column.get(key, "")
        picked = self._column_picker(
            "Color by",
            current,
            allow_none=True,
            help="Colour the embedding by a chosen column's value instead of one flat "
            "colour. Presentation only -- does not change what Apply commits.",
        )
        self._umap_color_column[key] = picked or ""

        if picked:
            bucket = self._umap_color_values(source, key, result, picked)
            if bucket is not None:
                categories, bucketed = bucket
                background, grid, palette = self._preview_style()
                widgets.mini_scatter(
                    "preview",
                    result.x,
                    result.y,
                    point_colors=bucketed,
                    palette=palette,
                    markers=result.markers,
                    marker_color=theme.get_color("warn"),
                    height=_PREVIEW_HEIGHT,
                    label=result.plot_label,
                    background_color=background,
                    grid_color=grid,
                )
                self._draw_umap_color_legend(categories, palette)
                return
            imgui.text_disabled(f"Cannot color by {picked!r} for this embedding.")

        self._draw_result_preview(result, height=_PREVIEW_HEIGHT, id_prefix="preview")

    # -- trained-model rail (Phase 3) ------------------------------------------------

    def _draw_model_rail(self) -> None:
        """The right-hand rail: every saved TrainedModel, newest first.

        ImGui child windows scroll their own overflow by default, and this rail is
        already drawn inside one (see draw()) -- a second, nested child here would add
        nothing but a redundant scrollbar.
        """
        if not self.models.models:
            imgui.text_disabled("No saved models yet -- train something")
            imgui.text_disabled("and press Save as model.")
            return

        for model in reversed(self.models.models):
            self._draw_model_row(model)
            imgui.separator()

    def _draw_model_row(self, model: TrainedModel) -> None:
        """One rail row: a technique badge, a renameable name, a thumbnail, delete.

        Scoped by ``model.name`` rather than ``id(model)``: it reads more meaningfully
        in a debugger and ``ModelStore.add`` already guarantees names are unique at any
        instant, which is all a per-frame id scope needs.
        """
        imgui.push_id(model.name)
        try:
            imgui.text_colored(theme.get_color("accent"), model.technique.upper())

            if self._rail_rename_target == model.name:
                self._draw_rail_rename_field(model)
            else:
                avail = imgui.get_content_region_avail().x
                name_w = max(40.0, avail - 18.0 - 8.0)
                clicked, _ = imgui.selectable(
                    f"{model.name}##name",
                    False,
                    imgui.SelectableFlags_.allow_double_click,
                    (name_w, 0.0),
                )
                if clicked and imgui.is_mouse_double_clicked(0):
                    self._begin_rail_rename(model)
                imgui.same_line()
                if icons.icon_button(
                    f"##del_{model.name}",
                    "trash",
                    size=18.0,
                    tooltip=f"Delete '{model.name}'",
                ):
                    self.submit(lambda m=model: self.models.remove(m))

            try:
                self._draw_result_preview(
                    model.preview, height=_RAIL_THUMB_HEIGHT, id_prefix=f"rail_{model.name}"
                )
            except Exception:
                # Defensive only: model.preview is meant to be an immutable, already-
                # settled _Result, but a row that ever failed to render must not take
                # the whole panel down with it.
                imgui.text_disabled("(preview unavailable)")

            self._draw_apply_model_section(model)
        finally:
            imgui.pop_id()

    def _draw_apply_model_section(self, model: TrainedModel) -> None:
        """The "Apply to..." affordance on a rail row: transfer ``model`` onto a
        different (or the same) dataset elsewhere in the workspace.

        Collapsed by default (a button that expands an inline section) since the rail
        is only ``_RAIL_WIDTH`` wide and every row already carries a thumbnail. Expanded,
        the flow is: pick a target dataset -> map the model's own trained column
        name(s) onto that dataset's columns -> name the output -> pick Add column/New
        dataset -> Apply.
        """
        key = model.name
        open_now = self._apply_model_open.get(key, False)
        label = "Apply to..." if not open_now else "Apply to... (close)"
        if imgui.button(label, (-1.0, 0.0)):
            open_now = not open_now
            self._apply_model_open[key] = open_now
        if not open_now:
            return

        imgui.indent()
        try:
            if model.technique == "umap" and model.umap_reducer is None:
                widgets.help_marker(
                    "Unavailable: this model's UMAP reducer was not available when it "
                    "was trained."
                )
                return

            names = self.store.names()
            if not names:
                imgui.text_disabled("No datasets to apply to yet.")
                return
            target_name = self._apply_model_target_ds.get(key, "")
            if target_name not in names:
                target_name = names[0]
            _c, target_name = widgets.enum_combo("Target dataset", target_name, names)
            self._apply_model_target_ds[key] = target_name
            dataset = self.store.get(target_name)
            if dataset is None:
                return

            column_map = dict(self._apply_model_columns.get(key, {}))
            if model.technique == "fit":
                picked = self._target_column_picker(
                    "x",
                    dataset,
                    column_map.get("x", ""),
                    help="The dataset column to evaluate the fit against.",
                )
                if picked is not None:
                    column_map["x"] = picked
            elif model.technique == "cluster":
                picked_x = self._target_column_picker(
                    "x",
                    dataset,
                    column_map.get("x", ""),
                    help="The dataset column mapped to the trained x axis.",
                )
                if picked_x is not None:
                    column_map["x"] = picked_x
                picked_y = self._target_column_picker(
                    "y",
                    dataset,
                    column_map.get("y", ""),
                    help="The dataset column mapped to the trained y axis.",
                )
                if picked_y is not None:
                    column_map["y"] = picked_y
                if model.cluster_method == "hierarchical":
                    widgets.help_marker(
                        "No native out-of-sample rule for hierarchical clustering -- "
                        "points are assigned to the nearest saved centroid (approximate)."
                    )
            else:  # pca / umap
                trained_columns = (
                    model.pca_columns if model.technique == "pca" else model.umap_columns
                )
                for trained_col in trained_columns:
                    picked = self._target_column_picker(
                        f"'{trained_col}' maps to",
                        dataset,
                        column_map.get(trained_col, ""),
                    )
                    if picked is not None:
                        column_map[trained_col] = picked
            self._apply_model_columns[key] = column_map

            default_output = f"{model.name} applied"
            output_name = self._apply_model_output_name.get(key, "") or default_output
            _c, output_name = imgui.input_text("Output name", output_name)
            self._apply_model_output_name[key] = output_name
            output_name = output_name.strip() or default_output

            mode = self._apply_model_mode.get(key, "add_column")
            imgui.text("Apply as")
            imgui.same_line()
            if imgui.radio_button("Add column", mode == "add_column"):
                mode = "add_column"
            imgui.same_line()
            if imgui.radio_button("New dataset", mode == "new_dataset"):
                mode = "new_dataset"
            self._apply_model_mode[key] = mode

            theme.push_accent()
            clicked = imgui.button("Apply", (-1.0, 0.0))
            theme.pop_accent()
            if clicked:
                self._submit_apply_model(model, dataset, dict(column_map), output_name, mode)
        finally:
            imgui.unindent()

    def _submit_apply_model(
        self,
        model: TrainedModel,
        dataset: DataSet,
        column_map: Dict[str, str],
        output_name: str,
        mode: str,
    ) -> None:
        """Build the right Command for ``mode`` and queue it -- ``_submit_apply``'s
        sibling for a transferred model instead of a tab's own settled result."""
        if mode == "new_dataset":
            cmd = self._command_apply_model_new_dataset(model, dataset, column_map, output_name)
        else:
            cmd = self._command_apply_model_add_column(model, dataset, column_map, output_name)
        if cmd is None:
            return
        self.submit(lambda: self._push(cmd))

    def _begin_rail_rename(self, model: TrainedModel) -> None:
        """Enter inline rename on a rail row."""
        self._rail_rename_target = model.name
        self._rail_rename_buf = model.name
        self._rail_focus_rename = True

    def _cancel_rail_rename(self) -> None:
        """Leave rail rename without touching the model."""
        self._rail_rename_target = None
        self._rail_rename_buf = ""
        self._rail_focus_rename = False

    def _commit_rail_rename(self, model: TrainedModel, text: str) -> None:
        """Commit a rail rename via :meth:`submit`, deliberately NOT via
        :meth:`push_command`: trained models are outside the undo stack, same as
        Phase 2's "Save as model" (see :meth:`_draw_save_model_row`). An empty or
        unchanged name is a no-op; a name colliding with a DIFFERENT model is deduped
        via ``ModelStore.unique_name``, the same rename-on-collision logic
        ``ModelStore.add`` already uses.
        """
        new = str(text).strip()
        old = model.name
        self._cancel_rail_rename()
        if not new or new == old:
            return
        unique = self.models.unique_name(new)
        self.submit(lambda: setattr(model, "name", unique))

    def _draw_rail_rename_field(self, model: TrainedModel) -> None:
        """Inline rename field. Enter commits, Escape cancels, click-away commits.

        The exact idiom scene.py's ``_draw_rename_field`` uses for a layer row.
        """
        if self._rail_focus_rename:
            imgui.set_keyboard_focus_here()
            self._rail_focus_rename = False

        imgui.push_item_width(max(60.0, imgui.get_content_region_avail().x - 8.0))
        # buffer_length is left at its -1 default: an explicit length shorter than the
        # text silently truncates it (CONTRACT §2.4).
        entered, text = imgui.input_text(
            "##rail_rename",
            self._rail_rename_buf,
            flags=imgui.InputTextFlags_.enter_returns_true | imgui.InputTextFlags_.auto_select_all,
        )
        imgui.pop_item_width()
        self._rail_rename_buf = text

        if entered:
            self._commit_rail_rename(model, text)
        elif imgui.is_item_deactivated():
            # Escape reverts imgui's own buffer, so the returned text is already the
            # original; check the key rather than trusting the value.
            if imgui.is_key_pressed(imgui.Key.escape):
                self._cancel_rail_rename()
            else:
                self._commit_rail_rename(model, text)

    # -- report --------------------------------------------------------------------

    def _report_text(self, source: _Source, result: _Result) -> str:
        """The operation's result as tab-separated text, ready to paste into a spreadsheet.

        A header line names the operation and its source; every stat row follows as
        ``label<TAB>value``. This is the escape hatch the panel was missing: a fit gives
        ``mu = 2.00 +/- 0.03`` on screen, and this is how that number leaves the tool.
        """
        lines = [f"# {result.plot_label}\t(source: {source.label})"]
        lines.extend(f"{label}\t{value}" for label, value in result.stats)
        return "\n".join(lines)

    def _draw_export_row(self, result: _Result, key: str) -> None:
        """A path field + an export button, writing this tab's result to CSV.

        Same convention as the Data Editor's Import/Export section (``clipboard.
        format_table`` + a direct synchronous write) -- there is no native file dialog
        anywhere in this GUI, so a text field is the established, not the improvised,
        answer. Not a :class:`Command`: writing a file touches no scene state and needs
        no undo entry.
        """
        imgui.set_next_item_width(-70.0)
        _c, self._csv_path = imgui.input_text("##mathlab_csv", self._csv_path)
        imgui.same_line()
        if icons.icon_button("##mathlab_export", "download", tooltip="Export result as CSV"):
            self._export_result_csv(result)
            self._exported_frames = 90
            self._exported_key = key
        imgui.same_line()
        widgets.help_marker(
            "Write this operation's result to the CSV file at the path above -- the full "
            "table for Peaks/Histogram-like operations, otherwise the (x, y) curve shown "
            "in the preview."
        )
        if self._exported_key == key and self._exported_frames > 0:
            self._exported_frames -= 1
            imgui.same_line()
            imgui.text_disabled("Exported")

    def _export_result_csv(self, result: _Result) -> None:
        """Write ``result`` (its table, or its (x, y) pair) to ``self._csv_path``."""
        if result.table is not None:
            headers = [name for name, _ in result.table]
            columns = [np.asarray(values, dtype=np.float64) for _, values in result.table]
        else:
            x_name = result.x_name
            y_name = result.y_name if result.y_name != x_name else f"{result.y_name} (2)"
            headers = [x_name, y_name]
            columns = [
                np.asarray(result.x, dtype=np.float64),
                np.asarray(result.y, dtype=np.float64),
            ]

        path = self._csv_path.strip()
        try:
            data = np.column_stack(columns)
            text = clipboard.format_table(headers, data, delimiter=",")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
        except (OSError, ValueError) as exc:
            self._notice = f"Could not write {path!r}: {exc}"

    def _draw_report_header(self, source: _Source, result: _Result, key: str) -> None:
        """A "Copy report" button above the stat rows, with a transient confirmation."""
        if imgui.small_button("Copy report"):
            imgui.set_clipboard_text(self._report_text(source, result))
            # ~1.5 s of confirmation at 60 fps, scoped to this tab.
            self._copied_frames = 90
            self._copied_key = key
        imgui.same_line()
        widgets.help_marker(
            "Copy this operation's statistics as tab-separated text (label and value per "
            "row), ready to paste into a spreadsheet or your notes."
        )
        if self._copied_key == key and self._copied_frames > 0:
            self._copied_frames -= 1
            imgui.same_line()
            imgui.text_disabled("Copied")

    # -- apply ---------------------------------------------------------------------

    def _replace_reason(self, source: _Source, result: _Result) -> Optional[str]:
        """None if "Replace source" is possible, else a human explanation of why not."""
        if not result.allow_replace:
            return "This operation's output is not a replacement for its own input."
        # The synthetic row-index axis (_INDEX_OPTION) has no dataset column to write
        # back into. ``_command_replace_dataset`` always writes y, and writes x only when
        # the result changed it -- so only an index Y axis, or an index X axis this
        # result actually changed, makes Replace impossible; an index X axis the result
        # leaves alone is fine (only y gets written).
        if source.y_col == _INDEX_OPTION or (result.x_changed and source.x_col == _INDEX_OPTION):
            axis = "Y" if source.y_col == _INDEX_OPTION else "X"
            return (
                f"The {axis} axis is the row index, not a real dataset column, so there is "
                "nothing to write the result back into. Use New dataset or New layer instead."
            )
        if source.dataset is not None:
            rows = source.dataset.n_rows()
            if int(result.y.size) != rows:
                return (
                    f"Replacing in place needs the source's row count: this result has "
                    f"{result.y.size} rows, the dataset has {rows}. Use New dataset or "
                    f"New layer instead."
                )
            return None
        layer_type = getattr(source.layer, "layer_type", "")
        if layer_type not in _REPLACEABLE_LAYER_TYPES:
            return f"A {layer_type!r} layer has no in-place x/y update path."
        return None

    def _apply_mode_for(self, key: str) -> str:
        """This tab's apply mode. Defaults to "New layer", the non-destructive choice."""
        return self._apply_modes.get(key, "new_layer")

    def _effective_apply_mode(self, key: str, result: _Result) -> str:
        """``_apply_mode_for(key)``, clamped to what ``result.apply_modes`` allows.

        A stored preference from another tab (or from before this result restricted its
        modes) can name a mode ``result`` does not offer -- fall back to the first
        allowed one rather than showing radios with none selected.
        """
        allowed = result.apply_modes or tuple(candidate for candidate, _label in _APPLY_MODES)
        mode = self._apply_mode_for(key)
        return mode if mode in allowed else allowed[0]

    def _apply_name_for(self, key: str) -> str:
        """This tab's user-typed output name ("" means "use the default")."""
        return self._apply_names.get(key, "")

    def _draw_apply(self, source: _Source, result: _Result, key: str) -> None:
        """The apply-mode radios, the output name, and the Apply button.

        All state here is per-tab (``key``): the mode and the name belong to the
        operation, not to the panel.
        """
        reason = self._replace_reason(source, result)
        mode = self._effective_apply_mode(key, result)
        offered = result.apply_modes or tuple(candidate for candidate, _label in _APPLY_MODES)

        imgui.text("Apply as")
        imgui.same_line()
        first = True
        for candidate, label in _APPLY_MODES:
            if candidate not in offered:
                continue
            if not first:
                imgui.same_line()
            first = False
            if imgui.radio_button(label, mode == candidate):
                self._apply_modes[key] = candidate
                mode = candidate
            if candidate == "replace" and reason is not None:
                imgui.same_line()
                widgets.help_marker(f"Unavailable: {reason}")
            if candidate == "add_column" and source.dataset is None:
                imgui.same_line()
                widgets.help_marker("Unavailable: a plotted layer has no table to add a column to.")
        if len(offered) < len(_APPLY_MODES):
            imgui.same_line()
            widgets.help_marker(
                "This result has no natural (x, y) curve or matching-length column, so "
                "only the modes that make sense for it are offered."
            )

        imgui.push_item_width(-140.0)
        try:
            if mode == "new_layer":
                kind_changed, self._layer_kind = widgets.enum_combo(
                    "Kind", self._layer_kind, _LAYER_KINDS
                )
                if kind_changed:
                    self._layer_opts = layerops.default_kind_options(self._layer_kind)
                widgets.kind_options_editor(self._layer_opts)
            if mode in ("new_layer", "new_dataset", "add_column"):
                caption = "Column name" if mode == "add_column" else "Name"
                _c, name = imgui.input_text(caption, self._apply_name_for(key))
                self._apply_names[key] = name
                imgui.same_line()
                widgets.help_marker(
                    f"Leave empty for the default: {self._default_name(source, result)!r}"
                )
            if mode == "add_column" and source.dataset is not None:
                n_rows = source.dataset.n_rows()
                n_values = int(np.asarray(result.y).size)
                if n_rows and n_values != n_rows:
                    verb = "truncated to" if n_values > n_rows else "padded with NaN to"
                    imgui.text_colored(
                        f"This tab's {n_values} points will be {verb} the dataset's "
                        f"{n_rows} rows -- the Samples count above is not preserved.",
                        1.0,
                        0.7,
                        0.2,
                        1.0,
                    )
        finally:
            imgui.pop_item_width()

        if mode == "replace" and reason is not None:
            imgui.text_disabled("Apply")
            imgui.same_line()
            widgets.help_marker(reason)
            return
        if mode == "add_column" and source.dataset is None:
            imgui.text_disabled("Apply")
            imgui.same_line()
            widgets.help_marker("Switch the source to a dataset to add a column.")
            return

        theme.push_accent()
        clicked = imgui.button("Apply", (130.0, 0.0))
        theme.pop_accent()
        imgui.same_line()
        widgets.help_marker(
            "Queued for the next frame (a draw callback may not touch the scene) and "
            "recorded on the undo stack."
        )
        if clicked:
            self._submit_apply(source, result, key)

    def _build_trained_model(
        self, source: _Source, result: _Result, key: str
    ) -> Optional[TrainedModel]:
        """Build a :class:`TrainedModel` by reading ``result.model_state``/``result.stats``.

        A pure read of the already-settled ``result`` -- this never calls ``_compute``
        again. Returns None when ``key`` is not one of ``_TRAINABLE_TABS`` or ``result``
        carries no ``model_state`` (an operation that has not successfully trained yet).
        """
        if key not in _TRAINABLE_TABS or result.model_state is None:
            return None
        state = result.model_state
        train_metrics, val_metrics, test_metrics = _split_metrics(result.stats)
        title = _TAB_BY_KEY.get(key, (key, ""))[0]

        if key == "fit":
            input_columns: Tuple[str, ...] = (source.x_name,)
        elif key == "cluster":
            # A cluster assignment compares a new point against the centroids in the
            # same (x, y) space it was fit on, so reuse needs both axes, not just one.
            input_columns = (source.x_name, source.y_name)
        else:
            input_columns = tuple(state.get("columns", ()))

        if key in ("pca", "umap"):
            base = source.dataset.name if source.dataset is not None else source.label
            source_label = f"{base} [{', '.join(state.get('columns', ()))}]"
        else:
            source_label = f"{source.label} ({source.x_name}, {source.y_name})"

        model = TrainedModel(
            name=f"{title} model",
            technique=key,
            created=time.time(),
            source_label=source_label,
            input_columns=input_columns,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            preview=result,
        )
        if key == "fit":
            model.fit_model_key = str(state["model_key"])
            model.fit_expr = str(state["expr"])
            model.fit_n_peaks = int(state["n_peaks"])
            model.fit_popt = state["popt"]
            model.fit_pcov = state["pcov"]
        elif key == "cluster":
            model.cluster_method = str(state["method"])
            model.cluster_centroid_x = state["centroid_x"]
            model.cluster_centroid_y = state["centroid_y"]
        elif key == "pca":
            model.pca_columns = tuple(state.get("columns", ()))
            model.pca_scale = str(state.get("scale", "zscore"))
            model.pca_mean_ = state.get("mean_")
            model.pca_components_ = state.get("components_")
            model.pca_scale_stats = state.get("scale_stats_")
        else:  # umap
            model.umap_columns = tuple(state.get("columns", ()))
            model.umap_scale = str(state.get("scale", "zscore"))
            model.umap_scale_stats = state.get("scale_stats_")
            model.umap_reducer = state.get("reducer")
        return model

    def _draw_save_model_row(self, source: _Source, result: _Result, key: str) -> None:
        """ "Save as model" name field and button, drawn right after Apply.

        Only for the four trainable tabs (``_TRAINABLE_TABS``), and only once this
        frame's result has actually trained something (``result.model_state`` is not
        None -- an errored or not-yet-successful fit has nothing to save). Registering
        the model uses :meth:`submit`, not :meth:`push_command`: trained models are
        deliberately outside the undo stack.
        """
        if key not in _TRAINABLE_TABS or result.model_state is None:
            return
        imgui.separator()
        title = _TAB_BY_KEY.get(key, (key, ""))[0]
        default_name = self.models.unique_name(f"{title} model")
        current = self._model_names.get(key, "")
        imgui.push_item_width(-140.0)
        try:
            _c, typed = imgui.input_text("Model name", current)
        finally:
            imgui.pop_item_width()
        self._model_names[key] = typed
        imgui.same_line()
        widgets.help_marker(f"Leave empty for the default: {default_name!r}")
        clicked = imgui.button("Save as model", (160.0, 0.0))
        imgui.same_line()
        widgets.help_marker(
            "Saves the trained model (parameters, not a live link) for reuse elsewhere. "
            "Not recorded on the undo stack."
        )
        if clicked:
            model = self._build_trained_model(source, result, key)
            if model is not None:
                model.name = typed.strip() or default_name
                self.submit(lambda m=model: self.models.add(m))

    def _default_name(self, source: _Source, result: _Result) -> str:
        """The auto-generated name for a new layer or dataset."""
        return f"{source.label} {result.suffix}".strip()

    def _output_name(self, source: _Source, result: _Result, key: str) -> str:
        """The name typed on *this* tab if there is one, else the default."""
        return self._apply_name_for(key).strip() or self._default_name(source, result)

    def _push(self, cmd: Command) -> None:
        """Record and run ``cmd``. Runs inside the drain, never from a draw callback.

        Deliberately not ``Panel.push_command``: that discards ``UndoStack.push``'s return
        value, and a caller is required to surface "operation not undoable" when it comes
        back False. Everything else about the two is identical.
        """
        if not self.undo.push(cmd):
            self._notice = (
                f"{cmd.label}: applied, but the data is too large to snapshot for undo. "
                f"The undo history was cleared."
            )

    def _submit_apply(self, source: _Source, result: _Result, key: str) -> None:
        """Build the Command for this tab's apply mode and queue it."""
        mode = self._effective_apply_mode(key, result)
        if mode == "new_layer":
            cmd = self._command_new_layer(source, result, key)
        elif mode == "new_dataset":
            cmd = self._command_new_dataset(source, result, key)
        elif mode == "add_column":
            cmd = self._command_add_column(source, result, key)
        elif source.dataset is not None:
            cmd = self._command_replace_dataset(source, result)
        else:
            cmd = self._command_replace_layer(source, result)
        if cmd is None:
            return
        self.submit(lambda: self._push(cmd))

    def _command_new_layer(self, source: _Source, result: _Result, key: str) -> Command:
        """Add the result as a fresh layer; undo removes it with the full teardown."""
        plot = self.plot
        hud = self.hud
        # Detach from the live columns now: the source may be edited between this frame
        # and the drain, and the command must apply what the preview showed.
        x = np.array(result.x, dtype=np.float64)
        y = np.array(result.y, dtype=np.float64)
        kind = self._layer_kind
        # Copied for the same reason as x/y: the command drains later, and the options
        # it applies must be the ones the preview was showing.
        options = dict(self._layer_opts)
        label = self._output_name(source, result, key)
        # A cluster (or any operation carrying per-point category labels) closes the loop
        # with GLPlot's own data-driven color encoding: the new layer's points are
        # coloured by cluster from the moment they exist, not left for a second trip
        # through the encodings picker.
        color_values = (
            np.array(result.color_values, dtype=np.float64)
            if result.color_values is not None
            else None
        )
        color_cmap = result.color_cmap
        holder: Dict[str, Any] = {}

        def do() -> None:
            holder["layer"] = layerops.add_xy_layer(
                plot, x, y, kind=kind, label=label, options=options, c=color_values, cmap=color_cmap
            )

        def undo() -> None:
            layer = holder.pop("layer", None)
            if layer is not None:
                layerops.remove_layer(plot, hud, layer)

        return Command(label=f"Plot {label!r}", do=do, undo=undo)

    def _command_new_dataset(self, source: _Source, result: _Result, key: str) -> Command:
        """Register the result as a new dataset; undo unregisters it.

        A result carrying a ``table`` (the Peaks tab does) becomes a multi-column dataset
        of that table; everything else is the two-column ``(x, y)`` the operation produced.
        """
        store = self.store
        name = self._output_name(source, result, key)
        if result.table is not None:
            columns = [
                Column(col_name, np.array(values, dtype=np.float64))
                for col_name, values in result.table
            ]
        else:
            x_name = result.x_name
            y_name = result.y_name if result.y_name != x_name else f"{result.y_name} (2)"
            columns = [
                Column(x_name, np.array(result.x, dtype=np.float64)),
                Column(y_name, np.array(result.y, dtype=np.float64)),
            ]
        dataset = DataSet(name, columns)
        dataset.source = "derived"

        def do() -> None:
            store.add(dataset)

        def undo() -> None:
            store.remove(dataset)

        return Command(label=f"Create dataset {name!r}", do=do, undo=undo)

    def _command_add_column(self, source: _Source, result: _Result, key: str) -> Optional[Command]:
        """Append the computed variable as a NEW column of the source dataset.

        This is what turns a derivative / integral / smoothed signal into a column the user can
        then map to colour or size, closing the loop with the encodings picker. The dataset
        keeps every column the same length, so the result is padded with NaN when it is shorter
        (it usually matches: derivative, integral and smooth are length-preserving) and
        truncated when longer (resample, peaks). Undo drops the column again.
        """
        dataset = source.dataset
        if dataset is None:
            self._notice = "Add column needs a dataset source (a plotted layer has no table)."
            return None

        name = self._output_name(source, result, key)
        # A cluster result's payload is the label, not y -- add_column writes the category
        # a user would actually want a new column for.
        source_values = result.color_values if result.color_values is not None else result.y
        values = np.asarray(source_values, dtype=np.float64)
        n = dataset.n_rows()
        if n and values.size != n:
            fitted = np.full(n, np.nan, dtype=np.float64)
            fitted[: min(values.size, n)] = values[: min(values.size, n)]
            values = fitted

        added: Dict[str, Any] = {}

        def do() -> None:
            column = dataset.add_column(name, values)
            added["name"] = column.name

        def undo() -> None:
            col_name = added.pop("name", None)
            if col_name is not None:
                dataset.remove_column(col_name)

        return Command(label=f"Add column {name!r} to {dataset.name!r}", do=do, undo=undo)

    # -- apply a saved model to a dataset (Phase 4) -----------------------------------

    def _evaluate_model(
        self, model: TrainedModel, dataset: DataSet, column_map: Dict[str, str]
    ) -> Any:
        """Evaluate a saved :class:`TrainedModel` against ``dataset``'s own columns.

        ``column_map`` maps a role the model needs onto the ``dataset`` column that
        supplies it: ``{"x": ...}`` for fit, ``{"x": ..., "y": ...}`` for cluster, and
        for pca/umap one entry per ``model.pca_columns``/``model.umap_columns`` name
        (the model's OWN trained column name as key, the ``dataset`` column to read as
        value).

        Returns a 1-D array (fit/cluster, length ``dataset.n_rows()``) or a 2-D
        ``(n_rows, n_components)`` array (pca/umap).

        Raises:
            ValueError: A required column is missing from ``column_map``/``dataset``,
                the model has no technique this function knows, or (umap only) the
                model was saved with no reducer.
        """
        technique = model.technique

        if technique == "fit":
            x_col = column_map.get("x")
            x_values = dataset.get(x_col) if x_col else None
            if x_col is None or x_values is None:
                raise ValueError(f"'{dataset.name}' has no column {x_col!r}")
            x_values = np.asarray(x_values, dtype=np.float64)
            if model.fit_model_key == "polynomial":
                # See the module-level note (and _fit_spec's own docstring): a
                # polynomial fit's expr is "" and n_peaks is 0 -- _fit_spec only knows
                # named/multipeak/custom models, so this one is evaluated directly.
                return np.polyval(model.fit_popt, x_values)
            spec = self._fit_spec(model.fit_model_key, model.fit_expr, model.fit_n_peaks)
            return spec.fn(x_values, *model.fit_popt)

        if technique == "cluster":
            x_col = column_map.get("x")
            y_col = column_map.get("y")
            x_values = dataset.get(x_col) if x_col else None
            y_values = dataset.get(y_col) if y_col else None
            if x_col is None or x_values is None:
                raise ValueError(f"'{dataset.name}' has no column {x_col!r}")
            if y_col is None or y_values is None:
                raise ValueError(f"'{dataset.name}' has no column {y_col!r}")
            return mathops2d.nearest_centroid(
                x_values, y_values, model.cluster_centroid_x, model.cluster_centroid_y
            )

        if technique in ("pca", "umap"):
            trained_columns = model.pca_columns if technique == "pca" else model.umap_columns
            arrays: List[np.ndarray] = []
            for trained_col in trained_columns:
                target_col = column_map.get(trained_col)
                values = dataset.get(target_col) if target_col else None
                if target_col is None or values is None:
                    raise ValueError(
                        f"'{dataset.name}' has no column {target_col!r} "
                        f"(mapped from {trained_col!r})"
                    )
                arrays.append(values)

            n_rows = dataset.n_rows()
            if technique == "pca":
                projected = mathopsnd.pca_transform(
                    arrays,
                    mean_=model.pca_mean_,
                    components_=model.pca_components_,
                    scale_stats=model.pca_scale_stats,
                    scale=model.pca_scale,
                )
            else:
                if model.umap_reducer is None:
                    # Defensive backstop only: the UI checks model.umap_reducer is None
                    # BEFORE ever calling this function and shows a disabled reason
                    # instead, so this path should not normally be reached.
                    raise ValueError(
                        f"Model {model.name!r} has no saved UMAP reducer; transfer is "
                        f"unavailable for it."
                    )
                # The same tiny local mirror of mathopsnd._as_matrix's finite-row-
                # filtered stacking that the PCA reconstruction-error branch above
                # already uses -- _as_matrix is private to mathopsnd and this file
                # already declined to reach into it once, so this follows the same
                # call rather than a second, inconsistent way in.
                matrix = _finite_column_matrix(arrays)
                scaled = mathopsnd.apply_scale(
                    matrix, model.umap_scale_stats, mode=model.umap_scale
                )
                projected = model.umap_reducer.transform(scaled)

            if int(np.asarray(projected).shape[0]) != n_rows:
                # Both pca_transform and the manual UMAP path above drop rows with a
                # non-finite value in any mapped column (mathopsnd's own, established
                # convention -- see pca_transform's docstring). That is the one case
                # where this technique's output is not exactly dataset.n_rows() long
                # "by construction"; surfaced as a clear error rather than silently
                # padding with NaN (see _command_apply_model_add_column's docstring).
                raise ValueError(
                    f"'{dataset.name}' has non-finite values in the mapped column(s); "
                    f"{technique.upper()} transfer needs a finite value in every row."
                )
            return projected

        raise ValueError(f"unknown technique {technique!r}")

    def _command_apply_model_add_column(
        self,
        model: TrainedModel,
        dataset: DataSet,
        column_map: Dict[str, str],
        output_name: str,
    ) -> Optional[Command]:
        """Evaluate ``model`` against ``dataset`` and append the result as new column(s).

        Mirrors ``_command_add_column``'s do/undo/Command shape, but never needs that
        command's NaN-padding branch: every technique's output is exactly
        ``dataset.n_rows()`` long by construction (fit/cluster evaluate pointwise;
        pca/umap read the dataset's own chosen columns), except the pca/umap
        non-finite-input edge case ``_evaluate_model`` already turns into a clear
        ValueError instead of a silently short array.

        A 1-D result (fit/cluster) becomes one column named ``output_name``. A 2-D
        result (pca/umap, ``n_components`` columns) becomes one column per component,
        named ``f"{output_name} {i + 1}"`` (1-based) -- the same convention
        ``_command_apply_model_new_dataset`` uses for its own output columns.
        """
        try:
            values = self._evaluate_model(model, dataset, column_map)
        except ValueError as exc:
            self._notice = str(exc)
            return None

        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            columns_to_add: List[Tuple[str, np.ndarray]] = [(output_name, arr)]
        else:
            columns_to_add = [(f"{output_name} {i + 1}", arr[:, i]) for i in range(arr.shape[1])]

        added: Dict[str, Any] = {}

        def do() -> None:
            names: List[str] = []
            for name, col_values in columns_to_add:
                column = dataset.add_column(name, col_values)
                names.append(column.name)
            added["names"] = names

        def undo() -> None:
            for col_name in added.pop("names", []):
                dataset.remove_column(col_name)

        return Command(label=f"Apply {model.name!r} to {dataset.name!r}", do=do, undo=undo)

    def _command_apply_model_new_dataset(
        self,
        model: TrainedModel,
        dataset: DataSet,
        column_map: Dict[str, str],
        output_name: str,
    ) -> Optional[Command]:
        """Evaluate ``model`` against ``dataset`` and register the result as a fresh
        dataset, undo unregisters it -- ``_command_new_dataset``'s shape.

        Fit's new dataset gets the x column plus one output column; Cluster's gets x,
        y, and the cluster-label column, mirroring the ORIGINAL cluster tab's own
        ``table=[(x_name, xa), (y_name, ya), ("cluster", labels))]`` convention (see
        the Cluster branch of ``_compute``) so a transferred cluster assignment is
        immediately plottable the same way a freshly trained one is. PCA/UMAP's new
        dataset carries only the ``n_components`` output columns (named the same
        ``f"{output_name} {i + 1}"`` way ``_command_apply_model_add_column`` does) --
        the input columns are not included there, since which of potentially many
        mapped columns would even be "the" input is not obvious the way it is for
        fit/cluster's single (x)/(x, y).
        """
        try:
            values = self._evaluate_model(model, dataset, column_map)
        except ValueError as exc:
            self._notice = str(exc)
            return None

        arr = np.asarray(values, dtype=np.float64)
        store = self.store
        name = output_name
        new_dataset = DataSet(name)
        new_dataset.source = "derived"

        if model.technique == "fit":
            x_col = column_map["x"]
            new_dataset.add_column(x_col, np.array(dataset.get(x_col), dtype=np.float64))
            new_dataset.add_column(output_name, arr)
        elif model.technique == "cluster":
            x_col = column_map["x"]
            y_col = column_map["y"]
            new_dataset.add_column(x_col, np.array(dataset.get(x_col), dtype=np.float64))
            new_dataset.add_column(y_col, np.array(dataset.get(y_col), dtype=np.float64))
            new_dataset.add_column("cluster", arr)
        else:  # pca / umap
            for i in range(arr.shape[1]):
                new_dataset.add_column(f"{output_name} {i + 1}", arr[:, i])

        def do() -> None:
            store.add(new_dataset)

        def undo() -> None:
            store.remove(new_dataset)

        return Command(label=f"Create dataset {name!r}", do=do, undo=undo)

    def _command_replace_dataset(self, source: _Source, result: _Result) -> Optional[Command]:
        """Overwrite the source columns in place, honouring any live layer link."""
        dataset = source.dataset
        if dataset is None:
            return None

        writes: List[Tuple[str, np.ndarray]] = []
        if result.x_changed and source.x_col != source.y_col:
            writes.append((source.x_col, np.array(result.x, dtype=np.float64)))
        writes.append((source.y_col, np.array(result.y, dtype=np.float64)))

        targets: List[Tuple[Column, Optional[np.ndarray], np.ndarray]] = []
        for col_name, values in writes:
            column = next((c for c in dataset.columns if c.name == col_name), None)
            if column is None:
                self._notice = f"Column {col_name!r} is gone from {dataset.name!r}."
                return None
            if column.values.size != values.size:
                # Re-checked here, not just at the radio: the dataset may have been
                # edited between the frame that enabled Replace and this click.
                self._notice = (
                    f"Column {col_name!r} now has {column.values.size} rows but the result "
                    f"has {values.size}. Nothing was changed."
                )
                return None
            targets.append((column, snapshot(column.values), values))

        label = f"Replace {dataset.name}.{source.y_col}"
        sync = self._make_sync(dataset)

        def do() -> None:
            for column, _old, new in targets:
                column.values[...] = new
                column.invalidate()
            sync()

        if any(old is None for _column, old, _new in targets):
            # snapshot() refused: the column is too large to copy (history.MAX_SNAPSHOT_
            # ELEMENTS) and there is no inverse for an overwrite.
            return Command.not_undoable(label, do)

        def undo() -> None:
            for column, old, _new in targets:
                column.values[...] = old
                column.invalidate()
            sync()

        return Command(label=label, do=do, undo=undo)

    def _make_sync(self, dataset: DataSet) -> Callable[[], None]:
        """Return a callback that pushes ``dataset`` back onto its layer, if it has one.

        This is what keeps the "edicion en vivo" link alive: a dataset built from a layer
        writes back to it, and ``write_back`` sets ``gpu_dirty`` per CONTRACT 1.4.
        Queued commands only.
        """
        plot = self.plot

        def sync() -> None:
            if not dataset.is_bound() or dataset.layer_id is None:
                return
            layer = layerops.find_layer(plot, dataset.layer_id)
            if layer is None:
                return
            dataset.write_back(layer)
            layerops.mark_scene_dirty(plot)

        return sync

    def _command_replace_layer(self, source: _Source, result: _Result) -> Optional[Command]:
        """Swap the layer's geometry for the result; undo restores the old arrays.

        The snapshot is taken from the geometry's own columns 0 and 1, not from the
        user's x/y picks -- those may be swapped, and undo must restore what was there.
        """
        plot = self.plot
        layer = source.layer
        if layer is None:
            return None
        geometry = getattr(layer, source.layer_attr, None)
        if geometry is None:
            self._notice = "The source layer no longer has geometry to replace."
            return None
        geo = np.asarray(geometry)

        x = np.array(result.x, dtype=np.float64)
        y = np.array(result.y, dtype=np.float64)
        old_a = snapshot(np.asarray(geo[:, 0], dtype=np.float64))
        old_b = snapshot(np.asarray(geo[:, 1], dtype=np.float64))
        old_colors = snapshot(getattr(layer, "colors", None))
        had_colors = getattr(layer, "colors", None) is not None
        label = f"Replace layer {getattr(layer, 'label', '?')!r}"

        def do() -> None:
            layerops.update_layer_xy(plot, layer, x, y)

        if old_a is None or old_b is None or (had_colors and old_colors is None):
            return Command.not_undoable(label, do)

        def undo() -> None:
            layerops.update_layer_xy(plot, layer, old_a, old_b)
            if old_colors is not None:
                # update_layer_xy refits colours to the new length; on the way back that
                # refit cannot recover per-vertex colours a shrink discarded, so restore
                # the originals verbatim.
                layer.colors = np.ascontiguousarray(old_colors)
                layer.dirty.gpu_dirty = True
                layerops.mark_scene_dirty(plot)

        return Command(label=label, do=do, undo=undo)


# ----------------------------------------------------------------------------------
# nan-safe reductions used by the stat rows
# ----------------------------------------------------------------------------------


def _nan_reduce(values: np.ndarray, fn: Callable[[np.ndarray], Any]) -> float:
    """Apply a nan-ignoring reduction, returning nan for an all-nan or empty input."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not bool(np.any(np.isfinite(arr))):
        return float("nan")
    with np.errstate(all="ignore"):
        return float(fn(arr))


def _nan_min(values: np.ndarray) -> float:
    """Smallest finite value, or nan."""
    return _nan_reduce(values, np.nanmin)


def _nan_max(values: np.ndarray) -> float:
    """Largest finite value, or nan."""
    return _nan_reduce(values, np.nanmax)


def _nan_peak(values: np.ndarray) -> float:
    """Largest finite absolute value, or nan."""
    return _nan_reduce(values, lambda arr: np.nanmax(np.abs(arr)))
