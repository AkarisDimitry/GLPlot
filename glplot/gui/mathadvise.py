"""Heuristic "what to try next" suggestions for Math Lab.

Pure numpy / :mod:`mathops` estimators over a signal's shape -- noise level, periodicity,
peak count, trend strength -- each scored independently and turned into a pointer at an
existing Math Lab tab. Deliberately heuristic, not statistical inference or ML: every
number here is a rough, fast estimate meant to nudge a user toward a tab worth trying, not
a claim about the data.

Depends only on :mod:`mathops` and numpy -- **never** on ``panels.mathlab`` -- so a
``tab_key`` here is a bare string the caller resolves/validates against its own
``_TAB_BY_KEY``. Importing the panel module here would make the dependency circular
(``panels.mathlab`` already imports this module to draw the suggestion strip).

No function in this module raises on data that simply does not fit a heuristic's
assumptions well -- it returns no recommendation instead, and :func:`recommend` additionally
isolates every heuristic behind its own ``try/except`` so one failing estimator can never
blank the whole list.
"""

from __future__ import annotations

import math
from typing import Any, List, NamedTuple, Tuple

import numpy as np

from . import mathops

__all__ = ["Recommendation", "recommend"]


class Recommendation(NamedTuple):
    """One suggested tab: what it is, why it was suggested, and how strongly."""

    tab_key: str
    title: str
    reason: str
    score: float


def _clean(x: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Finite, equal-length, ascending-x (x, y) -- the shape every heuristic assumes."""
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    n = min(xa.size, ya.size)
    xa, ya = xa[:n], ya[:n]
    finite = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[finite], ya[finite]
    if xa.size >= 2 and not bool(np.all(np.diff(xa) >= 0.0)):
        order = np.argsort(xa, kind="stable")
        xa, ya = xa[order], ya[order]
    return xa, ya


def _noise_recommendations(x: np.ndarray, y: np.ndarray) -> List[Recommendation]:
    """High-frequency wiggle vs. the signal's total spread -> Smooth / Filter.

    Noise is estimated from the second finite difference rather than a fixed-window
    smoothing residual: for white noise of standard deviation sigma, ``var(diff(y, 2)) ==
    6 * sigma**2``, so ``std(diff(y, 2)) / sqrt(6)`` recovers sigma directly regardless of
    the record length. A window-based residual was tried first and produced false
    positives on cleanly sampled, fast-oscillating signals (a smoothing window spanning
    several periods "removes" real structure and reports it as noise); the second-
    difference estimator is insensitive to smooth structure at any timescale coarser than
    the sample spacing, which is what actually distinguishes noise from signal here.
    """
    if y.size < 8:
        return []
    d2 = y[2:] - 2.0 * y[1:-1] + y[:-2]
    sigma_noise = float(np.nanstd(d2)) / math.sqrt(6.0)

    spread = float(np.nanstd(y))
    if spread <= 0.0 or not np.isfinite(spread):
        return []
    ratio = sigma_noise / spread
    if not np.isfinite(ratio) or ratio < 0.12:
        return []

    score = min(1.0, ratio)
    reason = (
        f"About {ratio * 100:.0f}% of the signal's spread looks like sample-to-sample "
        "noise rather than shape."
    )
    return [
        Recommendation("smooth", "Smooth", reason, score),
        Recommendation("filter", "Filter", reason, score * 0.9),
    ]


def _periodicity_recommendations(x: np.ndarray, y: np.ndarray) -> List[Recommendation]:
    """A strong autocorrelation echo away from lag 0 -> FFT / Autocorr."""
    if y.size < 16:
        return []
    lags, corr = mathops.autocorrelation(y, max_lag=max(2, y.size // 2))
    if corr.size < 3:
        return []

    interior = corr[1:-1]
    is_local_max = (interior > corr[:-2]) & (interior > corr[2:])
    if not np.any(is_local_max):
        return []
    candidates = np.where(is_local_max, interior, -np.inf)
    idx = int(np.argmax(candidates)) + 1
    peak_val = float(corr[idx])
    if not np.isfinite(peak_val) or peak_val < 0.3:
        return []

    score = min(1.0, peak_val)
    reason = (
        f"Autocorrelation has a strong echo at lag {float(lags[idx]):.3g} "
        f"(r={peak_val:.2f}) -- the signal may repeat."
    )
    return [
        Recommendation("fft", "FFT", reason, score),
        Recommendation("autocorr", "Autocorr", reason, score * 0.95),
    ]


def _peaks_recommendations(x: np.ndarray, y: np.ndarray) -> List[Recommendation]:
    """Several prominent local maxima -> Peaks, and Fit when there are only a few."""
    if y.size < 8:
        return []
    span = float(np.nanmax(y) - np.nanmin(y))
    if span <= 0.0 or not np.isfinite(span):
        return []

    peak_x, _peak_y, _props = mathops.find_peaks(x, y, prominence=0.1 * span)
    count = int(peak_x.size)
    if count < 2:
        return []

    score = min(1.0, count / 8.0)
    reason = f"{count} distinct peaks stand out above 10% of the signal's range."
    out = [Recommendation("peaks", "Peaks", reason, score)]
    if count <= 6:
        out.append(
            Recommendation(
                "fit",
                "Fit (multi-peak)",
                reason + " A sum-of-peaks fit may separate them.",
                score * 0.85,
            )
        )
    return out


def _trend_recommendations(x: np.ndarray, y: np.ndarray) -> List[Recommendation]:
    """A straight line already explains most of the variance -> Baseline (detrend)."""
    if x.size < 8:
        return []
    ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
    if ss_tot <= 0.0:
        return []

    try:
        slope, intercept = np.polyfit(x, y, 1)
    except (np.linalg.LinAlgError, ValueError):
        return []
    if abs(float(slope)) < 1e-12:
        return []

    fitted = slope * x + intercept
    ss_res = float(np.nansum((y - fitted) ** 2))
    r_squared = 1.0 - ss_res / ss_tot
    if r_squared < 0.4:
        return []

    score = min(1.0, r_squared)
    reason = (
        f"A straight line already explains {r_squared * 100:.0f}% of the variance -- "
        "removing it as a baseline may reveal the rest."
    )
    return [Recommendation("detrend", "Baseline", reason, score)]


def _cluster_recommendation(x: np.ndarray, y: np.ndarray) -> List[Recommendation]:
    """A quiet, constant-score nudge toward Cluster once there are enough points.

    Unlike every other heuristic here, this is not a claim that the data *shows*
    clusters -- there is no cheap, reliable way to detect cluster structure without
    running a clustering algorithm, which is exactly what the tab itself is for. It is
    scored low and flat so it only surfaces when nothing more specific (noise,
    periodicity, peaks, trend) has a stronger signal, functioning as a background "you
    could also explore this" rather than a diagnosis.
    """
    if x.size < 20:
        return []
    return [
        Recommendation(
            "cluster",
            "Cluster",
            "Enough points to try grouping them with k-means -- worth a look if the "
            "plotted (x, y) pairs might fall into natural groups.",
            0.3,
        )
    ]


def _density2d_recommendation(x: np.ndarray, y: np.ndarray) -> List[Recommendation]:
    """A quiet, constant-score nudge toward Density 2D once there are enough points to
    bin usefully. Same reasoning as :func:`_cluster_recommendation`: no cheap, reliable
    way to detect "this would make a good heatmap" without building the heatmap."""
    if x.size < 50:
        return []
    return [
        Recommendation(
            "density2d",
            "Density 2D",
            "Enough points to bin into a 2-D histogram or KDE -- useful if the plotted "
            "(x, y) pairs overlap heavily and a scatter alone hides where they pile up.",
            0.25,
        )
    ]


def _spatial_recommendation(x: np.ndarray, y: np.ndarray) -> List[Recommendation]:
    """A quiet, constant-score nudge toward Spatial once there are enough points to say
    anything about spacing. Same reasoning as :func:`_cluster_recommendation`."""
    if x.size < 10:
        return []
    return [
        Recommendation(
            "spatial",
            "Spatial",
            "Enough points to compute nearest-neighbour spacing and a convex hull -- "
            "worth a look at how the plotted points are distributed in the plane.",
            0.2,
        )
    ]


def _envelope_recommendations(x: np.ndarray, y: np.ndarray) -> List[Recommendation]:
    """A real, varying amplitude envelope suggests amplitude modulation or decay --
    Envelope traces exactly that. A near-flat envelope (a steady, unmodulated
    oscillation, or a non-oscillating signal) is not flagged."""
    if y.size < 40:
        return []
    _x_out, env = mathops.envelope(x, y)
    if env.size == 0:
        return []
    mean_env = float(np.mean(env))
    if mean_env <= 0.0:
        return []
    coeff_var = float(np.std(env)) / mean_env
    if coeff_var < 0.15:
        return []
    score = min(1.0, coeff_var)
    reason = (
        f"The signal's amplitude envelope varies by about {coeff_var * 100:.0f}% of "
        "its own mean -- looks amplitude-modulated or decaying, not a steady oscillation."
    )
    return [Recommendation("envelope", "Envelope", reason, score)]


def _rolling_recommendations(x: np.ndarray, y: np.ndarray) -> List[Recommendation]:
    """A signal whose local spread changes noticeably across the record (louder in one
    stretch than another) is a rolling-std story a single describe() summary cannot
    tell. Splits the record into 5 chunks and compares their standard deviations."""
    n_chunks = 5
    chunk_size = y.size // n_chunks
    if chunk_size < 5:
        return []
    stds = []
    for i in range(n_chunks):
        chunk = y[i * chunk_size : (i + 1) * chunk_size]
        s = float(np.nanstd(chunk))
        if math.isfinite(s):
            stds.append(s)
    if len(stds) < 3:
        return []
    lo, hi = min(stds), max(stds)
    if hi <= 0.0:
        return []
    ratio = hi / max(lo, 1e-12)
    if ratio < 2.5:
        return []
    score = min(1.0, 0.3 + (ratio - 2.5) / 5.0)
    reason = (
        f"The signal's local spread varies by roughly {ratio:.1f}x across the record -- "
        "a rolling std would show where it gets noisier or quieter."
    )
    return [Recommendation("rolling", "Rolling stats", reason, score)]


def _compare_recommendation(x: np.ndarray, y: np.ndarray) -> List[Recommendation]:
    """A quiet, constant-score nudge toward Compare -- only offered when the source has
    a dataset with other columns to compare against (see ``recommend``'s ``has_dataset``)."""
    if x.size < 20:
        return []
    return [
        Recommendation(
            "compare",
            "Compare",
            "Another column is available in this dataset -- a two-sample test can say "
            "whether it and this one plausibly come from the same distribution.",
            0.25,
        )
    ]


def _multivariate_recommendations(
    x: np.ndarray, y: np.ndarray, n_columns: int
) -> List[Recommendation]:
    """A quiet, constant-score nudge toward PCA/UMAP -- only offered when the source has
    a dataset with enough OTHER columns to make dimensionality reduction meaningful (see
    ``recommend``'s ``n_columns``). Two columns is just the plotted (x, y) itself; three
    or more is where "reduce several columns to two" actually does something."""
    if n_columns < 3 or x.size < 20:
        return []
    return [
        Recommendation(
            "pca",
            "PCA",
            f"This dataset has {n_columns} columns -- a principal component analysis "
            "can summarise how they vary together in just 2.",
            0.2,
        ),
        Recommendation(
            "umap",
            "UMAP",
            f"With {n_columns} columns available, a UMAP embedding can reveal cluster/"
            "manifold structure a linear projection like PCA would flatten together.",
            0.15,
        ),
    ]


#: Every heuristic, run independently and merged. Each takes the same cleaned (x, y) and
#: returns zero or more recommendations; order here has no effect on the final ranking,
#: :func:`recommend` sorts by score. ``_compare_recommendation``/
#: ``_multivariate_recommendations`` are not in this tuple: they only make sense when
#: the source has a dataset with other columns available, so ``recommend`` adds them
#: conditionally rather than unconditionally like every heuristic here.
_HEURISTICS = (
    _noise_recommendations,
    _periodicity_recommendations,
    _peaks_recommendations,
    _trend_recommendations,
    _cluster_recommendation,
    _density2d_recommendation,
    _spatial_recommendation,
    _envelope_recommendations,
    _rolling_recommendations,
)


def recommend(
    x: Any,
    y: Any,
    *,
    has_dataset: bool = False,
    n_columns: int = 0,
    max_results: int = 5,
) -> List[Recommendation]:
    """Rank a handful of existing Math Lab tabs worth trying on ``(x, y)``.

    Every heuristic is isolated behind its own ``try/except``: a degenerate input that
    breaks one estimator (e.g. too few points for autocorrelation) simply contributes no
    recommendation from that heuristic rather than raising out of the whole call -- this
    runs from a draw callback, where an exception would kill the frame.

    Args:
        x: First coordinate, 1-D.
        y: Second coordinate, 1-D, same length as x.
        has_dataset: Whether the source is backed by a dataset with other columns --
            gates the one heuristic (Compare) that needs a second column to mean
            anything at all. A plotted layer with no table cannot offer it.
        n_columns: Total column count of the source's dataset (0 for a plotted layer)
            -- gates PCA/UMAP, which only add value with three or more columns to
            reduce. Independent of ``has_dataset``: a dataset with exactly the two
            plotted columns has ``has_dataset=True`` but ``n_columns=2``, too few.
        max_results: Cap on the number of recommendations returned.

    Returns the top ``max_results`` recommendations, highest score first. An empty list
    means no heuristic found anything worth flagging, not that Math Lab has nothing to
    offer -- every tab is still reachable directly from the tab bar.
    """
    xa, ya = _clean(x, y)
    if xa.size < 4:
        return []

    heuristics = _HEURISTICS + ((_compare_recommendation,) if has_dataset else ())

    found: List[Recommendation] = []
    for heuristic in heuristics:
        try:
            found.extend(heuristic(xa, ya))
        except Exception:
            continue

    try:
        found.extend(_multivariate_recommendations(xa, ya, n_columns))
    except Exception:
        pass

    found.sort(key=lambda rec: rec.score, reverse=True)
    return found[:max_results]
