"""Forward/inverse transforms for axis scale modes.

"linear" (identity), "log" (base-10), "symlog", "asinh", and "logit" are real. This is
deliberately a tiny, dependency-free module: it is called from both the renderers
(per-axis, at VBO-upload time) and the tick/pick/autoscale code in ``engine.py`` and
``managers/axis.py``, none of which may import ``pyplot`` (that would be circular),
so there is nowhere higher to put it.

Matplotlib itself does not warn when individual data points fall outside a scale's
domain (non-positive on a log axis, outside ``(0, 1)`` on a logit axis) -- it silently
masks them (a `Line2D`/`PathCollection` just draws a gap where the mask is set).
``forward()`` matches that for "log"/"logit": no warning, just NaN, which every
renderer already treats as "don't draw this vertex" (the same convention `plot()`
uses for a manual NaN gap). "symlog" and "asinh" are total functions over the reals
(no domain restriction), so neither masks anything -- see
``matplotlib.scale.SymmetricalLogTransform``/``AsinhTransform``/``LogitTransform``,
which the formulas below are copied from verbatim so each stays numerically
identical (checked directly against matplotlib in tests, not just visually similar).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def forward(values: np.ndarray, mode: str, params: Optional[dict] = None) -> np.ndarray:
    """World data -> the space the camera/ticks/GPU buffers operate in."""
    if mode == "log":
        arr = np.asarray(values, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(arr > 0, np.log10(arr), np.nan)
    if mode == "symlog":
        base, linthresh, linscale = _symlog_params(params)
        linscale_adj = linscale / (1.0 - base**-1)
        log_base = np.log(base)
        arr = np.asarray(values, dtype=np.float64)
        abs_arr = np.abs(arr)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.sign(arr) * linthresh * (linscale_adj + np.log(abs_arr / linthresh) / log_base)
        inside = abs_arr <= linthresh
        out = np.where(inside, arr * linscale_adj, out)
        return out
    if mode == "asinh":
        linear_width = _asinh_params(params)
        return linear_width * np.arcsinh(np.asarray(values, dtype=np.float64) / linear_width)
    if mode == "logit":
        arr = np.asarray(values, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.log10(arr / (1.0 - arr))
        return np.where((arr > 0) & (arr < 1), out, np.nan)
    return values


def inverse(values: np.ndarray, mode: str, params: Optional[dict] = None) -> np.ndarray:
    """The space the camera/ticks/GPU buffers operate in -> world data."""
    if mode == "log":
        return np.power(10.0, values)
    if mode == "symlog":
        base, linthresh, linscale = _symlog_params(params)
        linscale_adj = linscale / (1.0 - base**-1)
        invlinthresh = linthresh * linscale_adj
        arr = np.asarray(values, dtype=np.float64)
        abs_arr = np.abs(arr)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.sign(arr) * linthresh * np.power(base, abs_arr / linthresh - linscale_adj)
        inside = abs_arr <= invlinthresh
        out = np.where(inside, arr / linscale_adj, out)
        return out
    if mode == "asinh":
        linear_width = _asinh_params(params)
        return linear_width * np.sinh(np.asarray(values, dtype=np.float64) / linear_width)
    if mode == "logit":
        return 1.0 / (1.0 + np.power(10.0, -np.asarray(values, dtype=np.float64)))
    return values


def clamp_to_domain(coords: np.ndarray, mode: str, params: Optional[dict] = None) -> np.ndarray:
    """Floor/ceiling ``coords`` into a scale's valid domain before ``forward()``.

    For geometry that is a triangle soup rather than independent points/segments (a
    bar, a patch, an image quad) an out-of-domain vertex is not a harmless dropped
    primitive the way it is for a line gap or a missing scatter point -- one bad
    corner corrupts the whole triangle. This gives every such vertex a tiny push back
    inside the domain instead, relative to the *other* valid coordinates already in
    the same piece of geometry, so the shape still renders (typically running off the
    edge of whatever view it's in) rather than vanishing or corrupting its neighbours.

    A no-op for "linear"/"symlog"/"asinh" (every real number is already valid).
    """
    if mode == "log":
        positive = coords[coords > 0]
        floor = float(positive.min()) / 1e6 if positive.size else 1e-6
        return np.where(coords > 0, coords, floor)
    if mode == "logit":
        valid = coords[(coords > 0) & (coords < 1)]
        lo = float(valid.min()) if valid.size else 0.5
        hi = float(valid.max()) if valid.size else 0.5
        floor = lo / 1e6 if lo > 0 else 1e-6
        ceil = 1.0 - (1.0 - hi) / 1e6 if hi < 1 else 1.0 - 1e-6
        out = np.where(coords <= 0, floor, coords)
        return np.where(out >= 1, ceil, out)
    return coords


def _symlog_params(params: Optional[dict]) -> tuple:
    """``(base, linthresh, linscale)``, matplotlib's own defaults for any unset key."""
    p = params or {}
    return float(p.get("base", 10)), float(p.get("linthresh", 2)), float(p.get("linscale", 1))


def _asinh_params(params: Optional[dict]) -> float:
    """``linear_width``, matplotlib's own default (1.0) if unset."""
    return float((params or {}).get("linear_width", 1.0))
