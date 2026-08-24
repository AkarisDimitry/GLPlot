"""Reusable imgui presentation primitives for the GLPlot workstation.

Every public function re-checks ``IMGUI_AVAILABLE`` at the call site, so this module
imports cleanly on a GL-less machine (see CONTRACT 5.1). The centrepieces are the two
custom ``draw_list`` renderers: :func:`mini_plot`, the 2-D live preview used by the
function generator and the math lab, and :func:`mini_scene3d`, its 3-D counterpart used
by the Objects 3D catalogue.

The helpers below them (``_minmax_decimate``, ``_nice_ticks``, :func:`project_points_3d`,
:func:`bounds3d`) are pure numpy and carry no imgui dependency, so they are unit-testable
headless.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from imgui_bundle import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - exercised only on GL-less systems
    IMGUI_AVAILABLE = False
    imgui = None

try:
    from imgui_bundle import imspinner

    IMSPINNER_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - exercised only on GL-less systems
    IMSPINNER_AVAILABLE = False
    imspinner = None


# Maximum number of vertices a mini_plot series may emit after decimation.
MAX_PLOT_POINTS = 2000

# Inner margins of a mini_plot: room for the y tick labels on the left and the
# x tick labels along the bottom.
_MARGIN_LEFT = 40.0
_MARGIN_RIGHT = 6.0
_MARGIN_TOP = 4.0
_MARGIN_BOTTOM = 15.0

# Hard ceilings on what one mini_scene3d frame may emit. Callers are expected to decimate
# to their own budget first -- these exist so a caller that forgets cannot stall the UI.
# The numbers come from measurement, not taste: on this pyimgui build 2000 add_line calls
# cost ~0.34 ms, 2000 add_circle_filled ~0.46 ms and 1800 add_triangle_filled ~0.35 ms, so
# a fully loaded thumbnail stays around a millisecond -- affordable at 60 fps, and the
# reason a draw_list preview is viable at all.
MAX_SCENE3D_POINTS = 2500
MAX_SCENE3D_SEGMENTS = 2500
MAX_SCENE3D_TRIANGLES = 2200

# Degrees of orbit per pixel of drag inside a mini_scene3d. 0.45 puts a full turn at
# about 800 px, which is a comfortable flick across a docked panel.
SCENE3D_ORBIT_DEG_PER_PX = 0.45

# Alpha multiplier applied to the primitive furthest from the eye, ramping to 1.0 at the
# nearest. A draw list has no depth buffer, so this fade plus a painter's sort is the only
# depth cue a thumbnail gets -- without it a wireframe sphere reads as a flat disc.
_SCENE3D_FAR_ALPHA = 0.34

# Inner margin of a mini_scene3d. Small: there are no tick labels to leave room for.
_SCENE3D_MARGIN = 3.0

# Used when theme.py is unavailable (e.g. a test importing widgets in isolation).
_FALLBACK_COLORS: Dict[str, Tuple[float, float, float, float]] = {
    "accent": (0.26, 0.59, 0.98, 1.00),
    "ok": (0.35, 0.78, 0.45, 1.00),
    "warn": (0.95, 0.68, 0.25, 1.00),
    "err": (0.92, 0.35, 0.35, 1.00),
    "muted": (0.60, 0.62, 0.66, 1.00),
    "panel_bg": (0.12, 0.13, 0.15, 1.00),
    "grid": (0.30, 0.32, 0.36, 1.00),
}

#: Default qualitative palette for mini_scatter's point_colors -- distinguishable by hue
#: alone (not a gradient), the way a categorical label (a cluster id) should read. Callers
#: that want the live plot's own series colors instead pass the resolved
#: ``PlotStyle.palette`` (see Math Lab's ``_preview_style``); this is only the fallback.
_DEFAULT_QUALITATIVE_PALETTE: Tuple[Tuple[float, float, float, float], ...] = (
    (0.26, 0.59, 0.98, 1.00),  # blue
    (0.95, 0.55, 0.25, 1.00),  # orange
    (0.35, 0.78, 0.45, 1.00),  # green
    (0.92, 0.35, 0.35, 1.00),  # red
    (0.65, 0.45, 0.90, 1.00),  # purple
    (0.80, 0.65, 0.30, 1.00),  # gold
    (0.35, 0.75, 0.80, 1.00),  # teal
    (0.90, 0.45, 0.70, 1.00),  # pink
)

_theme_module: Any = None
_theme_probed = False


def _theme_colors() -> Dict[str, Tuple[float, float, float, float]]:
    """Return the theme's semantic color tokens, falling back to a local palette.

    ``theme.py`` is imported lazily so that this module stays importable on its own
    and so that a partially-installed package cannot break a draw callback.
    """
    global _theme_module, _theme_probed
    if not _theme_probed:
        _theme_probed = True
        try:
            from . import theme as _theme

            _theme_module = _theme
        except Exception:
            _theme_module = None
    colors = getattr(_theme_module, "COLORS", None)
    if isinstance(colors, dict) and colors:
        return colors
    return _FALLBACK_COLORS


def _token(name: str) -> Tuple[float, float, float, float]:
    """Look up a semantic color token, tolerating a theme that lacks it."""
    colors = _theme_colors()
    value = colors.get(name)
    if value is None:
        value = _FALLBACK_COLORS.get(name, (1.0, 1.0, 1.0, 1.0))
    return _as_rgba(value)


def _as_rgba(color: Any) -> Tuple[float, float, float, float]:
    """Normalise a 3- or 4-component color to an (r, g, b, a) float tuple."""
    try:
        vals = [float(c) for c in color]
    except (TypeError, ValueError):
        return (1.0, 1.0, 1.0, 1.0)
    if len(vals) == 3:
        vals.append(1.0)
    elif len(vals) < 3:
        return (1.0, 1.0, 1.0, 1.0)
    return (vals[0], vals[1], vals[2], vals[3])


def _u32(color: Any, alpha_scale: float = 1.0) -> int:
    """Pack a color tuple into an ImU32, optionally scaling its alpha."""
    r, g, b, a = _as_rgba(color)
    return imgui.get_color_u32((r, g, b, max(0.0, min(1.0, a * alpha_scale))))


def _icon(draw_list: Any, shape: str, cx: float, cy: float, radius: float, col: int) -> bool:
    """Draw an icon from ``icons.py`` if that module is importable.

    Returns True when the icon was drawn, so callers can fall back to plain text.
    """
    try:
        from . import icons
    except Exception:
        return False
    draw = getattr(icons, "draw_icon", None)
    if draw is None:
        return False
    try:
        draw(draw_list, shape, cx, cy, radius, col)
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Pure-numpy helpers (no imgui)
# ---------------------------------------------------------------------------


def _as_1d(values: Any) -> np.ndarray:
    """Coerce input to a contiguous 1-D float64 array."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


def _finite_range(*arrays: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
    """Return (min, max) over the finite samples of every array, or None."""
    lo = math.inf
    hi = -math.inf
    found = False
    for arr in arrays:
        if arr is None or arr.size == 0:
            continue
        mask = np.isfinite(arr)
        if not mask.any():
            continue
        sub = arr[mask]
        lo = min(lo, float(sub.min()))
        hi = max(hi, float(sub.max()))
        found = True
    if not found:
        return None
    return lo, hi


def _pad_range(lo: float, hi: float, frac: float = 0.05) -> Tuple[float, float]:
    """Expand a range by ``frac`` on each side, fixing degenerate spans."""
    if not math.isfinite(lo) or not math.isfinite(hi):
        return -1.0, 1.0
    if hi < lo:
        lo, hi = hi, lo
    span = hi - lo
    if span <= 0.0:
        pad = abs(hi) * 0.5 if hi != 0.0 else 0.5
        return lo - pad, hi + pad
    pad = span * frac
    return lo - pad, hi + pad


def _nice_ticks(lo: float, hi: float, count: int = 4) -> List[float]:
    """Return ~``count`` human-readable tick values inside [lo, hi].

    Uses the classic 1/2/2.5/5/10 mantissa ladder so labels land on round numbers.
    """
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo or count < 1:
        return []
    raw = (hi - lo) / float(count)
    if raw <= 0.0 or not math.isfinite(raw):
        return []
    magnitude = 10.0 ** math.floor(math.log10(raw))
    norm = raw / magnitude
    # Snap to the NEAREST rung of the 1/2/2.5/5/10 ladder. Always rounding up would
    # overshoot the step and yield far fewer ticks than asked for (e.g. 2 instead of 4).
    candidate = min((1.0, 2.0, 2.5, 5.0, 10.0), key=lambda c: abs(c - norm))
    step = candidate * magnitude
    if step <= 0.0 or not math.isfinite(step):
        return []
    first = math.ceil(lo / step)
    ticks: List[float] = []
    # Build from an integer index rather than accumulating, to avoid float drift.
    for i in range(0, count * 4 + 4):
        value = (first + i) * step
        if value > hi + step * 1e-9:
            break
        if value >= lo - step * 1e-9:
            # Snap values that are a hair off zero (e.g. -0.0 or 1e-17).
            if abs(value) < step * 1e-9:
                value = 0.0
            ticks.append(value)
    return ticks


def _minmax_decimate(x: np.ndarray, y: np.ndarray, columns: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce a series to <= ``2 * columns`` points via min/max decimation.

    The index range is split into ``columns`` buckets; each bucket contributes its
    minimum and its maximum sample, emitted in original index order. This preserves
    the visual envelope of the signal, unlike naive striding which drops extrema and
    aliases high-frequency content into a plausible-looking lie.

    Non-finite samples (nan and +/-inf) never win a bucket. A bucket containing no
    finite sample at all emits nan, so the caller's polyline splitter breaks the line
    there rather than bridging the gap.

    Returns ``(x, y)`` unchanged when the series is already small enough to draw.
    """
    n = int(y.size)
    if n == 0:
        return x, y
    cols = max(1, int(columns))
    if n <= 2 * cols:
        return x, y

    per_bucket = int(math.ceil(n / cols))
    n_buckets = int(math.ceil(n / per_bucket))
    padding = n_buckets * per_bucket - n
    if padding:
        y_padded = np.concatenate([y, np.full(padding, np.nan, dtype=np.float64)])
    else:
        y_padded = y
    grid = y_padded.reshape(n_buckets, per_bucket)

    finite = np.isfinite(grid)
    finite_count = finite.sum(axis=1)
    # +inf/-inf sentinels let us use argmin/argmax, which (unlike nanargmin) cannot
    # raise on an all-nan bucket.
    i_min = np.where(finite, grid, np.inf).argmin(axis=1)
    i_max = np.where(finite, grid, -np.inf).argmax(axis=1)

    base = np.arange(n_buckets, dtype=np.int64) * per_bucket
    first = base + np.minimum(i_min, i_max)
    second = base + np.maximum(i_min, i_max)

    indices = np.empty(n_buckets * 2, dtype=np.int64)
    indices[0::2] = first
    indices[1::2] = second
    np.clip(indices, 0, n - 1, out=indices)

    out_x = x[indices]
    out_y = y[indices]
    dead = finite_count == 0
    if dead.any():
        out_y[np.repeat(dead, 2)] = np.nan
    return out_x, out_y


def _fmt_tick(value: float) -> str:
    """Format a tick label compactly."""
    if value == 0.0:
        return "0"
    return "%.4g" % value


def _fmt_value(value: float) -> str:
    """Format a hover readout value, nan included."""
    if not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    return "%.6g" % value


def _nearest_index(xs: np.ndarray, target: float) -> int:
    """Index of the sample whose x is closest to ``target``.

    Makes no assumption that x is sorted, and never selects a non-finite x.
    """
    if xs.size == 0:
        return 0
    distance = np.abs(xs - target)
    distance = np.where(np.isfinite(distance), distance, np.inf)
    return int(distance.argmin())


def as_points3d(points: Any) -> np.ndarray:
    """Coerce anything to a contiguous ``(N, 3)`` float64 array, or ``(0, 3)`` on failure.

    Never raises. :func:`mini_scene3d` runs inside a draw callback, where an exception
    kills the whole frame and takes every other panel with it, so a caller that hands over
    a ragged list gets an empty scene and a "no data" box rather than a dead window.
    """
    try:
        arr = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError):
        return np.zeros((0, 3), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float64)
    return arr


def bounds3d(points: np.ndarray) -> Optional[Tuple[float, float, float, float, float, float]]:
    """``(xmin, xmax, ymin, ymax, zmin, zmax)`` over the finite rows, or None if there are none.

    The tuple order is :data:`glplot.core.camera3d.Bounds3D`'s, so the result feeds
    :meth:`~glplot.core.camera3d.Camera3D.mvp` directly.
    """
    if points.size == 0:
        return None
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        return None
    sub = points[finite]
    lo = sub.min(axis=0)
    hi = sub.max(axis=0)
    return (
        float(lo[0]),
        float(hi[0]),
        float(lo[1]),
        float(hi[1]),
        float(lo[2]),
        float(hi[2]),
    )


def project_points_3d(
    points: np.ndarray,
    mvp: np.ndarray,
    rect: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points to screen pixels through a row-major 4x4 MVP.

    ``mvp`` is exactly what :meth:`glplot.core.camera3d.Camera3D.mvp` returns -- the same
    matrix the GL renderer uploads -- so a thumbnail drawn from it and the real scene agree
    on where everything is, and a camera preset that looks right in one looks right in the
    other. ``rect`` is ``(x0, y0, x1, y1)`` in screen pixels, y growing downward.

    Returns ``(sx, sy, depth)``. ``depth`` is normalised device z: -1 at the near plane,
    +1 at the far one, so sorting descending is a painter's ordering.

    Vertices behind the eye (clip ``w <= 0``) come back as nan in all three components
    rather than being mirrored to the far side of the screen by the divide. That is the
    same non-finite convention :func:`_draw_series` already uses, so a polyline breaks
    there instead of folding across the widget -- which is what a naive divide produces
    and what makes a hand-rolled projection look broken the first time the camera enters
    the data.
    """
    n = int(points.shape[0]) if points.ndim == 2 else 0
    if n == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty.copy(), empty.copy()
    m = np.asarray(mvp, dtype=np.float64)
    if m.shape != (4, 4) or not np.all(np.isfinite(m)):
        nan = np.full(n, np.nan, dtype=np.float64)
        return nan, nan.copy(), nan.copy()

    clip = points @ m[:3, :3].T + m[:3, 3]
    w = points @ m[3, :3] + m[3, 3]
    bad = ~np.isfinite(w) | (w <= 1e-9) | ~np.isfinite(clip).all(axis=1)
    ndc = clip / np.where(bad, 1.0, w)[:, None]

    x0, y0, x1, y1 = rect
    sx = x0 + (ndc[:, 0] * 0.5 + 0.5) * (x1 - x0)
    # Screen y grows downward and NDC y grows upward, so the vertical axis flips here.
    sy = y1 - (ndc[:, 1] * 0.5 + 0.5) * (y1 - y0)
    depth = ndc[:, 2]
    if bad.any():
        sx = np.where(bad, np.nan, sx)
        sy = np.where(bad, np.nan, sy)
        depth = np.where(bad, np.nan, depth)
    return sx, sy, depth


def _pack_u32(rgba: np.ndarray) -> np.ndarray:
    """Pack an ``(N, 4)`` float RGBA array into ImU32s, as imgui's own ``IM_COL32`` does.

    Vectorised rather than looping over :func:`imgui.get_color_u32` because a
    thumbnail packs a few thousand colours per frame and that call is not free. Verified
    bit-for-bit identical to the imgui helper for the default style; the one thing it does
    not reproduce is ``style.Alpha``, the global fade imgui applies to disabled content,
    which no preview here is drawn under.
    """
    values = np.clip(np.asarray(rgba, dtype=np.float64), 0.0, 1.0)
    if values.ndim != 2 or values.shape[1] != 4:
        return np.zeros(0, dtype=np.uint32)
    bytes_ = np.rint(values * 255.0).astype(np.uint32)
    return (
        bytes_[:, 0] | (bytes_[:, 1] << 8) | (bytes_[:, 2] << 16) | (bytes_[:, 3] << 24)
    ).astype(np.uint32)


def _depth_fade(depth: np.ndarray) -> np.ndarray:
    """Alpha multipliers from :data:`_SCENE3D_FAR_ALPHA` (far) to 1.0 (near)."""
    finite = np.isfinite(depth)
    if not finite.any():
        return np.ones(depth.shape, dtype=np.float64)
    lo = float(depth[finite].min())
    hi = float(depth[finite].max())
    if hi - lo < 1e-9:
        return np.ones(depth.shape, dtype=np.float64)
    t = np.clip((np.where(finite, depth, hi) - lo) / (hi - lo), 0.0, 1.0)
    return 1.0 + t * (_SCENE3D_FAR_ALPHA - 1.0)


def _thin_indices(count: int, limit: int) -> Optional[np.ndarray]:
    """Evenly spaced positions keeping at most ``limit`` of ``count``, or None if it fits.

    Even striding, not min/max decimation: there is no "envelope" to preserve in a bag of
    3D primitives, and a stride keeps a surface's rows and columns evenly spread, which is
    what makes a thinned wireframe still read as the same shape.
    """
    if count <= limit:
        return None
    return np.unique(np.linspace(0, count - 1, limit).astype(np.int64))


def _split_runs(mask: np.ndarray) -> List[np.ndarray]:
    """Split a boolean mask into arrays of consecutive True indices."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [idx.size]))
    return [idx[s:e] for s, e in zip(starts.tolist(), ends.tolist())]


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


def section(label: str, *, default_open: bool = True) -> bool:
    """A styled collapsing header. Returns True when the body should be drawn."""
    if not IMGUI_AVAILABLE:
        return False
    flags = imgui.TreeNodeFlags_.default_open if default_open else 0
    imgui.push_style_var(imgui.StyleVar_.frame_padding, (6.0, 4.0))
    expanded = imgui.collapsing_header(label, flags=flags)
    imgui.pop_style_var(1)
    return bool(expanded)


def help_marker(text: str) -> None:
    """Draw a dimmed "(?)" that reveals ``text`` in a wrapped tooltip on hover."""
    if not IMGUI_AVAILABLE:
        return
    imgui.text_disabled("(?)")
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.push_text_wrap_pos(imgui.get_font_size() * 28.0)
        imgui.text_unformatted(text)
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()


def _trailing_help(help: Optional[str]) -> None:
    """Append a help marker on the same line, if help text was supplied."""
    if help:
        imgui.same_line()
        help_marker(help)


def labeled_slider_float(
    label: str,
    value: float,
    vmin: float,
    vmax: float,
    *,
    fmt: str = "%.3f",
    help: Optional[str] = None,
) -> Tuple[bool, float]:
    """A slider that leaves room for a trailing help marker.

    Returns ``(changed, value)``.
    """
    if not IMGUI_AVAILABLE:
        return False, float(value)
    changed, new_value = imgui.slider_float(
        label, float(value), float(vmin), float(vmax), format=fmt
    )
    _trailing_help(help)
    return bool(changed), float(new_value)


def labeled_drag_float(
    label: str,
    value: float,
    *,
    speed: float = 0.1,
    vmin: float = 0.0,
    vmax: float = 0.0,
    fmt: str = "%.3f",
    help: Optional[str] = None,
) -> Tuple[bool, float]:
    """A drag-float that leaves room for a trailing help marker.

    ``vmin == vmax == 0`` means unbounded, matching imgui's own convention.
    Returns ``(changed, value)``.
    """
    if not IMGUI_AVAILABLE:
        return False, float(value)
    changed, new_value = imgui.drag_float(
        label,
        float(value),
        v_speed=float(speed),
        v_min=float(vmin),
        v_max=float(vmax),
        format=fmt,
    )
    _trailing_help(help)
    return bool(changed), float(new_value)


def enum_combo(
    label: str,
    current: str,
    options: Sequence[str],
    *,
    help: Optional[str] = None,
) -> Tuple[bool, str]:
    """A string-valued combo box.

    Uses ``begin_combo`` rather than ``combo`` so that a ``current`` value which is
    not in ``options`` still previews correctly instead of rendering blank.
    Returns ``(changed, value)``.
    """
    if not IMGUI_AVAILABLE:
        return False, current
    items = [str(o) for o in options]
    changed = False
    selected = current
    combo_opened = imgui.begin_combo(label, str(current))
    if combo_opened:
        for item in items:
            is_selected = item == current
            clicked, _ = imgui.selectable(item, is_selected)
            if clicked and item != current:
                selected = item
                changed = True
            if is_selected:
                imgui.set_item_default_focus()
        imgui.end_combo()
    _trailing_help(help)
    return changed, selected


def kind_options_editor(opts: Dict[str, Any], *, width: float = 120.0) -> bool:
    """Edit a plot kind's parameters in place. Returns True if any of them changed.

    Driven by which keys ``opts`` *has*, not by the kind's name: ``layerops`` defaults a
    kind's options from its own registry, so a kind that grows a parameter gets its
    control here for free, and one that has none (``line``, ``scatter``) draws nothing.

    Every panel that builds a layer shares this, because the alternative is what shipped:
    the parameters existed and only the Data panel drew them, so a bar plotted from
    anywhere else was stuck at its default width with no control in sight.
    """
    if not IMGUI_AVAILABLE:
        return False

    # Imported here, not at module scope: `layerops` reaches OpenGL through
    # `core.layers`, and this module promises to import cleanly on a GL-less machine
    # (CONTRACT 5.1). Inside the function the import only runs once imgui is live, which
    # it cannot be without GL anyway.
    from . import layerops

    changed = False
    if "where" in opts:
        hit, opts["where"] = enum_combo(
            "Step at", str(opts["where"]), list(layerops.STEP_WHERE_OPTIONS)
        )
        changed |= hit
    if "bins" in opts:
        imgui.set_next_item_width(width)
        hit, bins = imgui.drag_int(
            "Bins (0 = auto)", int(opts["bins"]), 1.0, 0, layerops.MAX_HIST_BINS
        )
        opts["bins"] = max(0, min(int(bins), layerops.MAX_HIST_BINS))
        changed |= hit
    if "gridsize" in opts:
        imgui.set_next_item_width(width)
        hit, gridsize = imgui.drag_int(
            "Gridsize (0 = auto)", int(opts["gridsize"]), 1.0, 0, layerops.MAX_HIST_BINS
        )
        opts["gridsize"] = max(0, min(int(gridsize), layerops.MAX_HIST_BINS))
        changed |= hit
    if "mincnt" in opts:
        imgui.set_next_item_width(width)
        hit, mincnt = imgui.drag_int(
            "Min count (0 = show all)", int(opts["mincnt"]), 1.0, 0, 1_000_000
        )
        opts["mincnt"] = max(0, int(mincnt))
        changed |= hit
    if "pad" in opts:
        imgui.set_next_item_width(width)
        hit, pad = imgui.drag_float(
            "Cell size (0% = exact fit)", float(opts["pad"]), 0.5, -90.0, 200.0, "%.1f%%"
        )
        opts["pad"] = max(-90.0, float(pad))
        changed |= hit
    if "bar_width" in opts:
        imgui.set_next_item_width(width)
        hit, value = imgui.drag_float("Bar width (0 = auto)", float(opts["bar_width"]), 0.05)
        opts["bar_width"] = max(0.0, float(value))
        changed |= hit
    if "align" in opts:
        hit, opts["align"] = enum_combo(
            "Align", str(opts["align"]), list(layerops.BAR_ALIGN_OPTIONS)
        )
        changed |= hit
    if "density" in opts:
        hit, opts["density"] = imgui.checkbox("Density (area = 1)", bool(opts["density"]))
        changed |= hit
    if "cumulative" in opts:
        hit, opts["cumulative"] = imgui.checkbox("Cumulative (CDF)", bool(opts["cumulative"]))
        changed |= hit
    if "baseline" in opts:
        imgui.set_next_item_width(width)
        hit, opts["baseline"] = imgui.drag_float("Baseline", float(opts["baseline"]), 0.1)
        changed |= hit
    if "whis" in opts:
        imgui.set_next_item_width(width)
        hit, value = imgui.drag_float("Whisker (x IQR)", float(opts["whis"]), 0.05, 0.0, 20.0)
        opts["whis"] = max(0.0, float(value))
        changed |= hit
    if "cmap" in opts:
        hit, opts["cmap"] = enum_combo(
            "Colormap", str(opts["cmap"]), list(layerops.CPU_COLORMAP_NAMES)
        )
        changed |= hit
    return changed


def toolbar_separator() -> None:
    """A thin vertical rule between groups of toolbar buttons."""
    if not IMGUI_AVAILABLE:
        return
    imgui.same_line()
    draw_list = imgui.get_window_draw_list()
    pos = imgui.get_cursor_screen_pos()
    height = imgui.get_frame_height()
    col = _u32(_token("grid"))
    draw_list.add_line((pos.x + 3.0, pos.y + 2.0), (pos.x + 3.0, pos.y + height - 2.0), col, 1.0)
    imgui.dummy((7.0, height))
    imgui.same_line()


def stat_row(label: str, value: str) -> None:
    """One line of a statistics table: a muted label and a value column."""
    if not IMGUI_AVAILABLE:
        return
    imgui.text_disabled(str(label))
    imgui.same_line()
    # Align values into a column at ~45% of the window width, but never let the
    # column overlap a label that is longer than that.
    target = imgui.get_window_width() * 0.45
    if target > imgui.get_cursor_pos_x():
        imgui.set_cursor_pos_x(target)
    imgui.text(str(value))


def stats_table(id_str: str, headers: Tuple[str, str], rows: Sequence[Tuple[str, str]]) -> None:
    """A real two-column, bordered table of (label, value) rows.

    ``stat_row`` (above) fakes a column with manual cursor placement, which is fine for a
    handful of rows but reads as a list, not a table, once several tabs (Fit, Statistics,
    Correlate, Cluster) all need the same "label -> value" readout. This is that, done with
    a real ``imgui.begin_table`` (CONTRACT 2.3: ``end_table`` only when ``begin_table``
    returned True -- it is False when the table is clipped or out of space).
    """
    if not IMGUI_AVAILABLE:
        return
    flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h
    if not imgui.begin_table(id_str, 2, flags=flags):
        return
    try:
        imgui.table_setup_column(headers[0], imgui.TableColumnFlags_.width_stretch)
        imgui.table_setup_column(headers[1], imgui.TableColumnFlags_.width_stretch)
        imgui.table_headers_row()
        for label, value in rows:
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled(str(label))
            imgui.table_next_column()
            imgui.text(str(value))
    finally:
        imgui.end_table()


def error_box(msg: str) -> None:
    """A red-tinted, wrapped error panel with a warning glyph."""
    if not IMGUI_AVAILABLE:
        return
    err = _token("err")
    draw_list = imgui.get_window_draw_list()
    origin = imgui.get_cursor_screen_pos()
    width = max(32.0, imgui.get_content_region_avail().x)

    text = str(msg)
    wrap_width = max(16.0, width - 36.0)
    text_size = imgui.calc_text_size(text, hide_text_after_double_hash=False, wrap_width=wrap_width)
    height = max(24.0, text_size.y + 10.0)

    # Panel body + border, drawn behind the text.
    draw_list.add_rect_filled(
        (origin.x, origin.y), (origin.x + width, origin.y + height), _u32(err, 0.12), 4.0
    )
    draw_list.add_rect(
        (origin.x, origin.y),
        (origin.x + width, origin.y + height),
        _u32(err, 0.55),
        rounding=4.0,
        thickness=1.0,
    )
    icon_col = _u32(err)
    cy = origin.y + height * 0.5
    if not _icon(draw_list, "warning", origin.x + 15.0, cy, 6.0, icon_col):
        draw_list.add_circle((origin.x + 15.0, cy), 6.0, icon_col, 12, 1.5)

    # Lay the wrapped message inside the panel, then advance past the whole box.
    imgui.set_cursor_screen_pos((origin.x + 28.0, origin.y + 5.0))
    imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + wrap_width)
    imgui.push_style_color(imgui.Col_.text, (err[0], err[1], err[2], 1.0))
    imgui.text_unformatted(text)
    imgui.pop_style_color(1)
    imgui.pop_text_wrap_pos()
    imgui.set_cursor_screen_pos((origin.x, origin.y + height + 4.0))
    imgui.dummy((0.0, 0.0))


def confirm_popup(id_str: str, message: str) -> Optional[bool]:
    """Draw a modal confirmation popup previously opened with ``imgui.open_popup``.

    Returns None while the popup is open or closed without a decision, True on OK
    and False on Cancel. The caller must call ``imgui.open_popup(id_str)`` with the
    same id to raise it.
    """
    if not IMGUI_AVAILABLE:
        return None
    result: Optional[bool] = None
    opened, _visible = imgui.begin_popup_modal(id_str, flags=imgui.WindowFlags_.always_auto_resize)
    if opened:
        imgui.push_text_wrap_pos(imgui.get_font_size() * 24.0)
        imgui.text_unformatted(str(message))
        imgui.pop_text_wrap_pos()
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        if imgui.button("OK", (110.0, 0.0)):
            result = True
            imgui.close_current_popup()
        imgui.same_line()
        if imgui.button("Cancel", (110.0, 0.0)):
            result = False
            imgui.close_current_popup()
        imgui.end_popup()
    return result


def _draw_axes_frame(
    draw_list: Any,
    x0: float,
    px0: float,
    py0: float,
    px1: float,
    py1: float,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    scale_x: float,
    scale_y: float,
    grid_col: int,
    zero_col: int,
    frame_col: int,
    muted_col: int,
) -> None:
    """Grid lines, ~4 nice-numbered ticks per axis, a y=0 line, and the frame rect.

    The shared "what a mini chart's axes look like" block -- factored out of
    :func:`mini_plot` so :func:`mini_scatter` and :func:`mini_heatmap` draw an identical
    frame instead of a hand-copied approximation of one. ``x0`` is the widget's own
    screen-space left edge (not ``px0``, which is already inset by the tick-label margin),
    needed only to clamp a y-tick label from running off the left of the widget.
    """
    for tick in _nice_ticks(y_lo, y_hi, 4):
        sy = py1 - (tick - y_lo) * scale_y
        if sy < py0 - 0.5 or sy > py1 + 0.5:
            continue
        draw_list.add_line((px0, sy), (px1, sy), grid_col, 1.0)
        text = _fmt_tick(tick)
        size = imgui.calc_text_size(text)
        draw_list.add_text((max(x0 + 1.0, px0 - 5.0 - size.x), sy - size.y * 0.5), muted_col, text)

    for tick in _nice_ticks(x_lo, x_hi, 4):
        sx = px0 + (tick - x_lo) * scale_x
        if sx < px0 - 0.5 or sx > px1 + 0.5:
            continue
        draw_list.add_line((sx, py0), (sx, py1), grid_col, 1.0)
        text = _fmt_tick(tick)
        size = imgui.calc_text_size(text)
        tx = min(max(sx - size.x * 0.5, px0 - 4.0), px1 - size.x)
        draw_list.add_text((tx, py1 + 2.0), muted_col, text)

    if y_lo < 0.0 < y_hi:
        sy = py1 + y_lo * scale_y
        draw_list.add_line((px0, sy), (px1, sy), zero_col, 1.5)

    draw_list.add_rect((px0, py0), (px1, py1), frame_col, rounding=0.0, thickness=1.0)


def mini_plot(
    id_str: str,
    y: np.ndarray,
    *,
    x: Optional[np.ndarray] = None,
    height: float = 70.0,
    overlay: Optional[np.ndarray] = None,
    label: str = "",
    color: Optional[Any] = None,
    overlay_color: Optional[Any] = None,
    markers: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    marker_color: Optional[Any] = None,
    background_color: Optional[Any] = None,
    grid_color: Optional[Any] = None,
    band: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    band_color: Optional[Any] = None,
) -> None:
    """An interactive inline plot rendered with ``draw_list`` primitives.

    Deliberately not ``imgui.plot_lines``: that cannot overlay a second series, cannot
    show a real x axis, and cannot break at nan.

    Features:
      * Autoscales to the union of ``y`` and ``overlay`` so the two are comparable.
      * Axes frame, ~4 nice-numbered ticks per axis, and a highlighted y=0 line.
      * min/max decimation per pixel column above ``MAX_PLOT_POINTS`` samples, which
        preserves the signal envelope (naive striding does not).
      * nan-safe: the polyline breaks at every non-finite sample, never interpolating
        across a gap. Isolated finite samples are drawn as dots.
      * Hover: a vertical crosshair snapped to the nearest sample plus an (x, y)
        tooltip readout taken from the ORIGINAL, undecimated data.
      * Optional ``markers`` -- an ``(xs, ys)`` pair drawn as filled dots on top of the
        series, for calling out detected peaks or other points of interest. Markers are
        NOT decimated (there are few of them) and do not affect autoscaling; one outside
        the data's own range is simply clipped.
      * ``background_color``/``grid_color`` default to the ImGui chrome theme's tokens
        (unchanged behaviour for every existing caller) but can be overridden -- e.g. by
        Math Lab's preview, which passes the *real* plot's background/grid so the preview
        reads as "this plot, zoomed into one operation" rather than a generic chart.
      * Optional ``band`` -- a ``(lower, upper)`` pair, same x as ``y`` -- fills a
        translucent confidence/prediction band behind the series. Stride-subsampled
        (not min/max decimated: a band is smooth, not noisy, so an envelope-preserving
        decimation buys nothing) and broken at every non-finite sample the same way the
        series lines are, so a band that is only defined over part of the domain does not
        bridge across the gap.

    Mismatched lengths are truncated to the shortest series rather than raising --
    this runs inside a draw callback where an exception would kill the frame.
    """
    if not IMGUI_AVAILABLE:
        return

    y_data = _as_1d(y)
    x_data = _as_1d(x) if x is not None else np.arange(y_data.size, dtype=np.float64)
    if x_data.size != y_data.size:
        n = min(x_data.size, y_data.size)
        x_data = x_data[:n]
        y_data = y_data[:n]
    ov_data: Optional[np.ndarray] = None
    if overlay is not None:
        ov_data = _as_1d(overlay)
        if ov_data.size != y_data.size:
            n = min(ov_data.size, y_data.size)
            ov_data = ov_data[:n]

    band_lo_data: Optional[np.ndarray] = None
    band_hi_data: Optional[np.ndarray] = None
    if band is not None:
        band_lo_data = _as_1d(band[0])
        band_hi_data = _as_1d(band[1])
        n = min(band_lo_data.size, band_hi_data.size, x_data.size)
        band_lo_data = band_lo_data[:n]
        band_hi_data = band_hi_data[:n]

    marker_x: Optional[np.ndarray] = None
    marker_y: Optional[np.ndarray] = None
    if markers is not None:
        marker_x = _as_1d(markers[0])
        marker_y = _as_1d(markers[1])
        if marker_x.size != marker_y.size:
            m = min(marker_x.size, marker_y.size)
            marker_x, marker_y = marker_x[:m], marker_y[:m]

    series_color = _as_rgba(color) if color is not None else _token("accent")
    second_color = _as_rgba(overlay_color) if overlay_color is not None else _token("warn")
    dot_color = _as_rgba(marker_color) if marker_color is not None else _token("accent")
    band_rgba = _as_rgba(band_color) if band_color is not None else series_color
    grid_rgba = _as_rgba(grid_color) if grid_color is not None else _token("grid")
    bg_rgba = _as_rgba(background_color) if background_color is not None else _token("panel_bg")
    grid_col = _u32(grid_rgba, 0.55)
    zero_col = _u32(grid_rgba, 1.0)
    frame_col = _u32(grid_rgba, 0.9)
    muted_col = _u32(_token("muted"))
    bg_col = _u32(bg_rgba)

    # Reserve the widget rect. invisible_button (not dummy) so hover honours window
    # stacking and clipping.
    avail = imgui.get_content_region_avail()
    width = max(64.0, float(avail.x))
    box_height = max(32.0, float(height))
    origin = imgui.get_cursor_screen_pos()
    x0, y0 = float(origin.x), float(origin.y)

    imgui.push_id(id_str)
    imgui.invisible_button("##miniplot", (width, box_height))
    hovered = imgui.is_item_hovered()
    imgui.pop_id()

    x1, y1 = x0 + width, y0 + box_height
    draw_list = imgui.get_window_draw_list()
    draw_list.add_rect_filled((x0, y0), (x1, y1), bg_col, 3.0)

    px0 = x0 + _MARGIN_LEFT
    py0 = y0 + _MARGIN_TOP
    px1 = x1 - _MARGIN_RIGHT
    py1 = y1 - _MARGIN_BOTTOM
    inner_w = px1 - px0
    inner_h = py1 - py0
    if inner_w < 8.0 or inner_h < 8.0:
        draw_list.add_rect((x0, y0), (x1, y1), frame_col, rounding=3.0, thickness=1.0)
        return

    if y_data.size == 0:
        draw_list.add_rect((px0, py0), (px1, py1), frame_col, rounding=0.0, thickness=1.0)
        text = "no data"
        size = imgui.calc_text_size(text)
        draw_list.add_text(
            (px0 + (inner_w - size.x) * 0.5, py0 + (inner_h - size.y) * 0.5), muted_col, text
        )
        return

    x_range = _finite_range(x_data)
    y_range = _finite_range(y_data, ov_data, band_lo_data, band_hi_data)
    if x_range is None or y_range is None:
        draw_list.add_rect((px0, py0), (px1, py1), frame_col, rounding=0.0, thickness=1.0)
        text = "no finite data"
        size = imgui.calc_text_size(text)
        draw_list.add_text(
            (px0 + (inner_w - size.x) * 0.5, py0 + (inner_h - size.y) * 0.5), muted_col, text
        )
        return

    x_lo, x_hi = _pad_range(x_range[0], x_range[1], 0.0)
    if x_hi <= x_lo:
        x_lo, x_hi = _pad_range(x_range[0], x_range[1], 0.05)
    y_lo, y_hi = _pad_range(y_range[0], y_range[1], 0.05)

    scale_x = inner_w / (x_hi - x_lo)
    scale_y = inner_h / (y_hi - y_lo)

    def to_screen_x(values: np.ndarray) -> np.ndarray:
        return px0 + (values - x_lo) * scale_x

    def to_screen_y(values: np.ndarray) -> np.ndarray:
        return py1 - (values - y_lo) * scale_y

    _draw_axes_frame(
        draw_list,
        x0,
        px0,
        py0,
        px1,
        py1,
        x_lo,
        x_hi,
        y_lo,
        y_hi,
        scale_x,
        scale_y,
        grid_col,
        zero_col,
        frame_col,
        muted_col,
    )

    # --- series -----------------------------------------------------------------
    # One bucket per pixel column, capped so each series stays under MAX_PLOT_POINTS.
    columns = min(int(inner_w), MAX_PLOT_POINTS // 2)
    draw_list.push_clip_rect((px0, py0), (px1, py1), True)
    try:
        if band_lo_data is not None and band_hi_data is not None and band_lo_data.size >= 2:
            bx, blo, bhi = x_data[: band_lo_data.size], band_lo_data, band_hi_data
            if bx.size > MAX_PLOT_POINTS:
                stride = max(1, bx.size // MAX_PLOT_POINTS)
                bx, blo, bhi = bx[::stride], blo[::stride], bhi[::stride]
            _draw_band(
                draw_list,
                to_screen_x(bx),
                to_screen_y(blo),
                to_screen_y(bhi),
                _u32(band_rgba, 0.25),
            )
        if ov_data is not None:
            ox, oy = _minmax_decimate(x_data[: ov_data.size], ov_data, columns)
            _draw_series(draw_list, to_screen_x(ox), to_screen_y(oy), _u32(second_color), 1.5)
        dx, dy = _minmax_decimate(x_data, y_data, columns)
        _draw_series(draw_list, to_screen_x(dx), to_screen_y(dy), _u32(series_color), 1.5)

        if marker_x is not None and marker_x.size:
            mfinite = np.isfinite(marker_x) & np.isfinite(marker_y)
            msx = to_screen_x(marker_x[mfinite])
            msy = to_screen_y(marker_y[mfinite])
            dot_u32 = _u32(dot_color)
            # A thin panel-coloured ring keeps a dot legible where it sits on the line.
            ring_u32 = _u32(_token("panel_bg"))
            for cx, cy in zip(msx.tolist(), msy.tolist()):
                draw_list.add_circle_filled((cx, cy), 3.5, dot_u32)
                draw_list.add_circle((cx, cy), 3.5, ring_u32, 0, 1.0)

        # --- hover crosshair + readout ------------------------------------------
        if hovered:
            mouse = imgui.get_mouse_pos()
            if px0 <= mouse.x <= px1 and py0 <= mouse.y <= py1:
                target = x_lo + (float(mouse.x) - px0) / scale_x
                index = _nearest_index(x_data, target)
                sample_x = float(x_data[index])
                sample_y = float(y_data[index])
                if math.isfinite(sample_x):
                    sx = px0 + (sample_x - x_lo) * scale_x
                    draw_list.add_line((sx, py0), (sx, py1), muted_col, 1.0)
                    if math.isfinite(sample_y):
                        sy = py1 - (sample_y - y_lo) * scale_y
                        draw_list.add_circle_filled((sx, sy), 3.0, _u32(series_color))
                    _hover_tooltip(label, index, sample_x, sample_y, ov_data, y_data.size)
    finally:
        draw_list.pop_clip_rect()

    if label:
        draw_list.add_text((px0 + 4.0, py0 + 2.0), muted_col, str(label))


def mini_scatter(
    id_str: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    height: float = 150.0,
    point_colors: Optional[np.ndarray] = None,
    palette: Optional[Sequence[Any]] = None,
    color: Optional[Any] = None,
    markers: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    marker_color: Optional[Any] = None,
    background_color: Optional[Any] = None,
    grid_color: Optional[Any] = None,
    label: str = "",
) -> None:
    """An interactive scatter of unconnected points, optionally colored by category.

    The unordered-point-cloud counterpart to :func:`mini_plot`: unlike a signal, a
    cluster's x is not a function of a monotonic index, so there is no polyline and no
    min/max envelope decimation (which assumes an ordering that does not exist here) --
    points above ``MAX_PLOT_POINTS`` are uniform-stride subsampled instead. Apply always
    commits the full, non-decimated result; this only affects what the preview draws.

    Args:
        point_colors: Optional per-point category labels (e.g. from
            :func:`~glplot.gui.mathops2d.kmeans_cluster`). A non-negative label ``i``
            indexes ``palette[i % len(palette)]``; a negative label (``-1``,
            "unclustered") draws in the muted theme color instead. ``None`` draws every
            point in ``color`` (or the accent token).
        palette: The colors ``point_colors`` indexes into.
            Defaults to :data:`_DEFAULT_QUALITATIVE_PALETTE`.
        markers: An ``(xs, ys)`` pair drawn as larger, ringed dots on top -- centroids.
        background_color/grid_color: Same override convention as :func:`mini_plot`.

    Mismatched lengths are truncated to the shortest rather than raising -- this runs
    inside a draw callback where an exception would kill the frame.
    """
    if not IMGUI_AVAILABLE:
        return

    x_data = _as_1d(x)
    y_data = _as_1d(y)
    if x_data.size != y_data.size:
        n = min(x_data.size, y_data.size)
        x_data, y_data = x_data[:n], y_data[:n]

    colors_data: Optional[np.ndarray] = None
    if point_colors is not None:
        colors_data = _as_1d(point_colors)
        if colors_data.size != x_data.size:
            n = min(colors_data.size, x_data.size)
            colors_data = colors_data[:n]
            x_data, y_data = x_data[:n], y_data[:n]

    marker_x: Optional[np.ndarray] = None
    marker_y: Optional[np.ndarray] = None
    if markers is not None:
        marker_x = _as_1d(markers[0])
        marker_y = _as_1d(markers[1])
        if marker_x.size != marker_y.size:
            m = min(marker_x.size, marker_y.size)
            marker_x, marker_y = marker_x[:m], marker_y[:m]

    active_palette = tuple(palette) if palette else _DEFAULT_QUALITATIVE_PALETTE
    series_color = _as_rgba(color) if color is not None else _token("accent")
    dot_color = _as_rgba(marker_color) if marker_color is not None else _token("warn")
    grid_rgba = _as_rgba(grid_color) if grid_color is not None else _token("grid")
    bg_rgba = _as_rgba(background_color) if background_color is not None else _token("panel_bg")
    grid_col = _u32(grid_rgba, 0.55)
    zero_col = _u32(grid_rgba, 1.0)
    frame_col = _u32(grid_rgba, 0.9)
    muted_col = _u32(_token("muted"))
    bg_col = _u32(bg_rgba)

    avail = imgui.get_content_region_avail()
    width = max(64.0, float(avail.x))
    box_height = max(32.0, float(height))
    origin = imgui.get_cursor_screen_pos()
    x0, y0 = float(origin.x), float(origin.y)

    imgui.push_id(id_str)
    imgui.invisible_button("##miniscatter", (width, box_height))
    hovered = imgui.is_item_hovered()
    imgui.pop_id()

    x1, y1 = x0 + width, y0 + box_height
    draw_list = imgui.get_window_draw_list()
    draw_list.add_rect_filled((x0, y0), (x1, y1), bg_col, 3.0)

    px0 = x0 + _MARGIN_LEFT
    py0 = y0 + _MARGIN_TOP
    px1 = x1 - _MARGIN_RIGHT
    py1 = y1 - _MARGIN_BOTTOM
    inner_w = px1 - px0
    inner_h = py1 - py0
    if inner_w < 8.0 or inner_h < 8.0:
        draw_list.add_rect((x0, y0), (x1, y1), frame_col, rounding=3.0, thickness=1.0)
        return

    if x_data.size == 0:
        draw_list.add_rect((px0, py0), (px1, py1), frame_col, rounding=0.0, thickness=1.0)
        text = "no data"
        size = imgui.calc_text_size(text)
        draw_list.add_text(
            (px0 + (inner_w - size.x) * 0.5, py0 + (inner_h - size.y) * 0.5), muted_col, text
        )
        return

    x_range = _finite_range(x_data)
    y_range = _finite_range(y_data)
    if x_range is None or y_range is None:
        draw_list.add_rect((px0, py0), (px1, py1), frame_col, rounding=0.0, thickness=1.0)
        text = "no finite data"
        size = imgui.calc_text_size(text)
        draw_list.add_text(
            (px0 + (inner_w - size.x) * 0.5, py0 + (inner_h - size.y) * 0.5), muted_col, text
        )
        return

    x_lo, x_hi = _pad_range(x_range[0], x_range[1], 0.05)
    y_lo, y_hi = _pad_range(y_range[0], y_range[1], 0.05)
    scale_x = inner_w / (x_hi - x_lo)
    scale_y = inner_h / (y_hi - y_lo)

    def to_screen_x(values: np.ndarray) -> np.ndarray:
        return px0 + (values - x_lo) * scale_x

    def to_screen_y(values: np.ndarray) -> np.ndarray:
        return py1 - (values - y_lo) * scale_y

    _draw_axes_frame(
        draw_list,
        x0,
        px0,
        py0,
        px1,
        py1,
        x_lo,
        x_hi,
        y_lo,
        y_hi,
        scale_x,
        scale_y,
        grid_col,
        zero_col,
        frame_col,
        muted_col,
    )

    # --- points -------------------------------------------------------------
    plot_x, plot_y, plot_colors = x_data, y_data, colors_data
    if plot_x.size > MAX_PLOT_POINTS:
        stride = max(1, plot_x.size // MAX_PLOT_POINTS)
        plot_x, plot_y = plot_x[::stride], plot_y[::stride]
        if plot_colors is not None:
            plot_colors = plot_colors[::stride]

    draw_list.push_clip_rect((px0, py0), (px1, py1), True)
    try:
        finite = np.isfinite(plot_x) & np.isfinite(plot_y)
        sx = to_screen_x(plot_x[finite])
        sy = to_screen_y(plot_y[finite])
        if plot_colors is not None:
            labels = plot_colors[finite]
            unclustered_u32 = _u32(_token("muted"), 0.6)
            for px, py, lbl in zip(sx.tolist(), sy.tolist(), labels.tolist()):
                idx = int(round(lbl))
                col = (
                    unclustered_u32 if idx < 0 else _u32(active_palette[idx % len(active_palette)])
                )
                draw_list.add_circle_filled((px, py), 2.5, col)
        else:
            col = _u32(series_color)
            for px, py in zip(sx.tolist(), sy.tolist()):
                draw_list.add_circle_filled((px, py), 2.5, col)

        if marker_x is not None and marker_x.size:
            mfinite = np.isfinite(marker_x) & np.isfinite(marker_y)
            msx = to_screen_x(marker_x[mfinite])
            msy = to_screen_y(marker_y[mfinite])
            dot_u32 = _u32(dot_color)
            ring_u32 = _u32(_token("panel_bg"))
            for cx, cy in zip(msx.tolist(), msy.tolist()):
                draw_list.add_circle_filled((cx, cy), 5.0, dot_u32)
                draw_list.add_circle((cx, cy), 5.0, ring_u32, 0, 1.5)

        if hovered:
            mouse = imgui.get_mouse_pos()
            if px0 <= mouse.x <= px1 and py0 <= mouse.y <= py1 and sx.size:
                dist2 = (sx - float(mouse.x)) ** 2 + (sy - float(mouse.y)) ** 2
                nearest = int(np.argmin(dist2))
                if float(dist2[nearest]) <= 144.0:  # within 12px
                    raw_index = np.flatnonzero(finite)[nearest]
                    imgui.begin_tooltip()
                    if label:
                        imgui.text_disabled(str(label))
                    imgui.text("x = %s" % _fmt_value(float(plot_x[raw_index])))
                    imgui.text("y = %s" % _fmt_value(float(plot_y[raw_index])))
                    if plot_colors is not None:
                        lbl = int(round(float(plot_colors[raw_index])))
                        imgui.text_disabled(
                            "cluster = unclustered" if lbl < 0 else f"cluster = {lbl}"
                        )
                    imgui.end_tooltip()
    finally:
        draw_list.pop_clip_rect()

    if label:
        draw_list.add_text((px0 + 4.0, py0 + 2.0), muted_col, str(label))


def mini_heatmap(
    id_str: str,
    counts: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    height: float = 150.0,
    cmap: str = "magma",
    background_color: Optional[Any] = None,
    grid_color: Optional[Any] = None,
    overlay_points: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    label: str = "",
) -> None:
    """A 2-D density grid (a histogram or KDE surface) rendered as filled cells.

    Args:
        counts: ``(nx, ny)`` grid of density/count values, e.g. from
            :func:`~glplot.gui.mathops2d.density2d`.
        x_edges: Bin edges along x, length ``nx + 1`` -- the ``np.histogram2d`` convention.
        y_edges: Bin edges along y, length ``ny + 1``.
        cmap: A matplotlib colormap name, looked up through
            :func:`~glplot.gui.layerops.colormap_colors` -- the same normalise-then-look-up
            ``pyplot.scatter(c=...)`` performs, so a heat map built here and one built from
            code read as the same picture. ``layerops`` is imported locally (not at module
            scope) so this module keeps importing on a machine with no OpenGL.
        overlay_points: An optional ``(xs, ys)`` pair drawn as small dots on top -- the raw
            samples the grid was built from, so a user sees the density AND what produced
            it. Uniform-stride subsampled above ``MAX_PLOT_POINTS``, like every other
            point-cloud overlay in this module.

    A cell whose value cannot be colour-mapped (an unknown ``cmap`` name, or matplotlib
    unavailable) falls back to a flat frame-coloured fill rather than raising -- this runs
    inside a draw callback where an exception would kill the frame.
    """
    if not IMGUI_AVAILABLE:
        return

    grid = np.asarray(counts, dtype=np.float64)
    x_edges_arr = _as_1d(x_edges)
    y_edges_arr = _as_1d(y_edges)
    nx = x_edges_arr.size - 1
    ny = y_edges_arr.size - 1

    grid_rgba = _as_rgba(grid_color) if grid_color is not None else _token("grid")
    bg_rgba = _as_rgba(background_color) if background_color is not None else _token("panel_bg")
    grid_col = _u32(grid_rgba, 0.55)
    frame_col = _u32(grid_rgba, 0.9)
    muted_col = _u32(_token("muted"))
    bg_col = _u32(bg_rgba)

    avail = imgui.get_content_region_avail()
    width = max(64.0, float(avail.x))
    box_height = max(32.0, float(height))
    origin = imgui.get_cursor_screen_pos()
    x0, y0 = float(origin.x), float(origin.y)

    imgui.push_id(id_str)
    imgui.invisible_button("##miniheatmap", (width, box_height))
    imgui.pop_id()

    x1, y1 = x0 + width, y0 + box_height
    draw_list = imgui.get_window_draw_list()
    draw_list.add_rect_filled((x0, y0), (x1, y1), bg_col, 3.0)

    px0 = x0 + _MARGIN_LEFT
    py0 = y0 + _MARGIN_TOP
    px1 = x1 - _MARGIN_RIGHT
    py1 = y1 - _MARGIN_BOTTOM
    inner_w = px1 - px0
    inner_h = py1 - py0
    if inner_w < 8.0 or inner_h < 8.0:
        draw_list.add_rect((x0, y0), (x1, y1), frame_col, rounding=3.0, thickness=1.0)
        return

    x_lo, x_hi = float(x_edges_arr[0]), float(x_edges_arr[-1])
    y_lo, y_hi = float(y_edges_arr[0]), float(y_edges_arr[-1])
    valid_grid = (
        nx >= 1
        and ny >= 1
        and grid.shape == (nx, ny)
        and math.isfinite(x_lo)
        and math.isfinite(x_hi)
        and x_hi > x_lo
        and math.isfinite(y_lo)
        and math.isfinite(y_hi)
        and y_hi > y_lo
    )
    if not valid_grid:
        draw_list.add_rect((px0, py0), (px1, py1), frame_col, rounding=0.0, thickness=1.0)
        text = "no data"
        size = imgui.calc_text_size(text)
        draw_list.add_text(
            (px0 + (inner_w - size.x) * 0.5, py0 + (inner_h - size.y) * 0.5), muted_col, text
        )
        return

    scale_x = inner_w / (x_hi - x_lo)
    scale_y = inner_h / (y_hi - y_lo)

    def to_screen_x(values: np.ndarray) -> np.ndarray:
        return px0 + (values - x_lo) * scale_x

    def to_screen_y(values: np.ndarray) -> np.ndarray:
        return py1 - (values - y_lo) * scale_y

    try:
        from . import layerops

        cell_colors = layerops.colormap_colors(grid.ravel(), cmap).reshape(nx, ny, 4)
    except Exception:
        cell_colors = None

    draw_list.push_clip_rect((px0, py0), (px1, py1), True)
    try:
        sx_edges = to_screen_x(x_edges_arr).tolist()
        sy_edges = to_screen_y(y_edges_arr).tolist()
        for i in range(nx):
            left, right = sorted((sx_edges[i], sx_edges[i + 1]))
            for j in range(ny):
                top, bottom = sorted((sy_edges[j], sy_edges[j + 1]))
                col = frame_col if cell_colors is None else _u32(tuple(cell_colors[i, j]))
                draw_list.add_rect_filled((left, top), (right, bottom), col)

        _draw_axes_frame(
            draw_list,
            x0,
            px0,
            py0,
            px1,
            py1,
            x_lo,
            x_hi,
            y_lo,
            y_hi,
            scale_x,
            scale_y,
            grid_col,
            grid_col,
            frame_col,
            muted_col,
        )

        if overlay_points is not None:
            ox = _as_1d(overlay_points[0])
            oy = _as_1d(overlay_points[1])
            n = min(ox.size, oy.size)
            ox, oy = ox[:n], oy[:n]
            if ox.size > MAX_PLOT_POINTS:
                stride = max(1, ox.size // MAX_PLOT_POINTS)
                ox, oy = ox[::stride], oy[::stride]
            finite = np.isfinite(ox) & np.isfinite(oy)
            if np.any(finite):
                psx = to_screen_x(ox[finite]).tolist()
                psy = to_screen_y(oy[finite]).tolist()
                dot_col = _u32(_token("accent"), 0.5)
                for px_, py_ in zip(psx, psy):
                    draw_list.add_circle_filled((px_, py_), 1.5, dot_col)
    finally:
        draw_list.pop_clip_rect()

    if label:
        draw_list.add_text((px0 + 4.0, py0 + 2.0), muted_col, str(label))


def _draw_series(
    draw_list: Any, sx: np.ndarray, sy: np.ndarray, col: int, thickness: float
) -> None:
    """Stroke a screen-space series, breaking the polyline at every non-finite sample."""
    finite = np.isfinite(sx) & np.isfinite(sy)
    runs = _split_runs(finite)
    if not runs:
        return
    xs = sx.tolist()
    ys = sy.tolist()
    for run in runs:
        if run.size >= 2:
            points = [(xs[i], ys[i]) for i in run.tolist()]
            draw_list.add_polyline(points, col, thickness=thickness, flags=0)
        else:
            i = int(run[0])
            draw_list.add_circle_filled((xs[i], ys[i]), thickness * 0.9, col)


def _draw_band(
    draw_list: Any, sx: np.ndarray, s_lo: np.ndarray, s_hi: np.ndarray, col: int
) -> None:
    """Fill a screen-space (lower, upper) band, breaking at non-finite samples like
    :func:`_draw_series` -- one quad per consecutive pair within a finite run, so a band
    that is only defined over part of the domain never bridges across the gap."""
    finite = np.isfinite(sx) & np.isfinite(s_lo) & np.isfinite(s_hi)
    runs = _split_runs(finite)
    if not runs:
        return
    xs, los, his = sx.tolist(), s_lo.tolist(), s_hi.tolist()
    for run in runs:
        if run.size < 2:
            continue
        idx = run.tolist()
        for a, b in zip(idx[:-1], idx[1:]):
            draw_list.add_quad_filled(
                (xs[a], los[a]), (xs[b], los[b]), (xs[b], his[b]), (xs[a], his[a]), col
            )


def _hover_tooltip(
    label: str,
    index: int,
    sample_x: float,
    sample_y: float,
    overlay: Optional[np.ndarray],
    total: int,
) -> None:
    """Tooltip readout for the sample under the crosshair."""
    imgui.begin_tooltip()
    if label:
        imgui.text_disabled(str(label))
    imgui.text("x = %s" % _fmt_value(sample_x))
    imgui.text("y = %s" % _fmt_value(sample_y))
    if overlay is not None and index < overlay.size:
        imgui.text("overlay = %s" % _fmt_value(float(overlay[index])))
    imgui.text_disabled("i = %d / %d" % (index, total))
    imgui.end_tooltip()


# ---------------------------------------------------------------------------
# 3D thumbnail
# ---------------------------------------------------------------------------

#: The eight corners of a unit box, as (x, y, z) selectors into (min, max) per axis, and
#: the twelve edges joining them. Module-level so the reference box costs no allocation
#: per frame.
_BOX_CORNERS: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
)
_BOX_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _box_points(bounds: Tuple[float, float, float, float, float, float]) -> np.ndarray:
    """The eight corners of ``bounds`` as an ``(8, 3)`` array."""
    axes = ((bounds[0], bounds[1]), (bounds[2], bounds[3]), (bounds[4], bounds[5]))
    return np.array(
        [[axes[0][cx], axes[1][cy], axes[2][cz]] for cx, cy, cz in _BOX_CORNERS],
        dtype=np.float64,
    )


def _scaled_bounds(
    bounds: Tuple[float, float, float, float, float, float], factor: float
) -> Tuple[float, float, float, float, float, float]:
    """``bounds`` scaled about its own centre.

    Used to magnify the thumbnail: the camera frames the *bounding sphere* of what it is
    given, which is conservative for a box -- a unit sphere handed to a default
    :class:`~glplot.core.camera3d.Camera3D` fills barely half the height, because the
    radius it fits is the box's half-diagonal. Handing it a proportionally smaller box
    crops in without touching the camera's own state, so ``distance = None`` still means
    "fit the data" and a user who dollies is still in charge.
    """
    out = []
    for lo, hi in ((bounds[0], bounds[1]), (bounds[2], bounds[3]), (bounds[4], bounds[5])):
        centre = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * float(factor)
        out.extend((centre - half, centre + half))
    return (out[0], out[1], out[2], out[3], out[4], out[5])


def mini_scene3d(
    id_str: str,
    points: Any,
    *,
    camera: Any,
    colors: Optional[Any] = None,
    segments: Optional[Any] = None,
    triangles: Optional[Any] = None,
    dots: bool = False,
    height: float = 190.0,
    label: str = "",
    hint: str = "",
    point_radius: float = 1.6,
    thickness: float = 1.2,
    fill: float = 1.35,
    show_box: bool = True,
    orbit: bool = True,
) -> bool:
    """An orbitable 3D thumbnail rendered with ``draw_list`` primitives.

    The 3D counterpart of :func:`mini_plot`, and it exists for the same reason: a panel
    that builds geometry should be able to *show* the geometry before the user commits to
    it. What it cannot do is render with GL. An imgui widget has no framebuffer of its own
    -- the workstation's GL context is mid-frame drawing the real scene when a panel body
    runs -- so a preview here is a CPU projection painted into the window's draw list, not
    a texture. Rejected alternatives: an offscreen FBO (a second GL context and a
    per-frame readback for a 200 px thumbnail), and ``imgui.image`` over a
    software-rasterised buffer (a full rasteriser in Python, at which point the projection
    below is the cheap half of the work anyway).

    Geometry arrives as an index soup over one shared vertex array, which is exactly the
    shape :mod:`glplot.gui.layerops3d`'s ``geom`` functions already produce:

    ``points``
        ``(N, 3)`` world-space vertices.
    ``colors``
        ``(N, 4)`` float RGBA, one per vertex. A line takes the colour of its first
        endpoint and a triangle the mean of its three, which is flat shading -- a draw
        list cannot interpolate a vertex colour across a primitive. On a colour-mapped
        surface at thumbnail scale the difference is invisible.
    ``segments``
        ``(M, 2)`` integer index pairs, drawn as lines.
    ``triangles``
        ``(T, 3)`` integer index triples, drawn filled.
    ``dots``
        Draw every vertex as a disc. Set for the point kinds.

    All three primitive sets may be given at once; they are drawn triangles, then lines,
    then dots, each independently sorted back-to-front.

    Depth is faked, and deliberately so. There is no depth buffer, so ordering is a
    painter's sort on the projected z and occlusion is approximated by fading the far end
    of the scene towards :data:`_SCENE3D_FAR_ALPHA`. Interpenetrating surfaces will show
    the seam. For "what am I about to add", that is the right trade.

    Drag inside the widget to orbit ``camera`` **in place** -- the object is mutated, the
    way :func:`kind_options_editor` edits its dict -- and the return value is True on the
    frames where that happened, so the caller can react (stop an auto-spin, remember that
    the user chose this angle). ``camera`` needs :meth:`~glplot.core.camera3d.Camera3D.mvp`
    and :meth:`~glplot.core.camera3d.Camera3D.orbit` and nothing else.

    ``fill`` magnifies the framing over the camera's own (see :func:`_scaled_bounds`); the
    reference box is always drawn at the true bounds, so raising it crops rather than
    lying about where the data ends.

    Every count is capped (:data:`MAX_SCENE3D_POINTS`, :data:`MAX_SCENE3D_SEGMENTS`,
    :data:`MAX_SCENE3D_TRIANGLES`) by even striding. Callers should decimate to their own
    budget first -- the cap is a floor under the frame time, not a substitute for knowing
    what your geometry is.
    """
    if not IMGUI_AVAILABLE:
        return False

    verts = as_points3d(points)
    rgba: Optional[np.ndarray] = None
    if colors is not None:
        try:
            candidate = np.asarray(colors, dtype=np.float64)
        except (TypeError, ValueError):
            candidate = np.zeros((0, 4))
        if candidate.ndim == 2 and candidate.shape[1] == 4 and len(candidate) == len(verts):
            rgba = candidate

    grid_col = _u32(_token("grid"), 0.75)
    frame_col = _u32(_token("grid"), 0.9)
    muted_col = _u32(_token("muted"))
    bg_col = _u32(_token("panel_bg"))
    default_rgba = _token("accent")

    avail = imgui.get_content_region_avail()
    width = max(64.0, float(avail.x))
    box_height = max(48.0, float(height))
    origin = imgui.get_cursor_screen_pos()
    x0, y0 = float(origin.x), float(origin.y)

    imgui.push_id(id_str)
    imgui.invisible_button("##scene3d", (width, box_height))
    active = imgui.is_item_active()
    imgui.pop_id()

    orbited = False
    if orbit and active and hasattr(camera, "orbit"):
        delta = imgui.get_io().mouse_delta
        if float(delta.x) or float(delta.y):
            # Matplotlib's sign convention for a 3D drag: right drags the scene right (so
            # the azimuth of the *eye* decreases), down tips the top towards the viewer.
            camera.orbit(
                -float(delta.x) * SCENE3D_ORBIT_DEG_PER_PX,
                float(delta.y) * SCENE3D_ORBIT_DEG_PER_PX,
            )
            orbited = True

    x1, y1 = x0 + width, y0 + box_height
    draw_list = imgui.get_window_draw_list()
    draw_list.add_rect_filled((x0, y0), (x1, y1), bg_col, 3.0)

    px0 = x0 + _SCENE3D_MARGIN
    py0 = y0 + _SCENE3D_MARGIN
    px1 = x1 - _SCENE3D_MARGIN
    py1 = y1 - _SCENE3D_MARGIN
    inner_w, inner_h = px1 - px0, py1 - py0
    if inner_w < 8.0 or inner_h < 8.0:
        draw_list.add_rect((x0, y0), (x1, y1), frame_col, rounding=3.0, thickness=1.0)
        return orbited

    bounds = bounds3d(verts)
    if bounds is None:
        draw_list.add_rect((x0, y0), (x1, y1), frame_col, rounding=3.0, thickness=1.0)
        text = "no data" if len(verts) == 0 else "no finite data"
        size = imgui.calc_text_size(text)
        draw_list.add_text(
            (px0 + (inner_w - size.x) * 0.5, py0 + (inner_h - size.y) * 0.5), muted_col, text
        )
        return orbited

    framed = bounds if fill == 1.0 else _scaled_bounds(bounds, 1.0 / max(float(fill), 1e-3))
    try:
        mvp = camera.mvp(inner_w / max(inner_h, 1e-6), framed)
    except Exception:  # pragma: no cover - a camera that cannot build a matrix
        draw_list.add_rect((x0, y0), (x1, y1), frame_col, rounding=3.0, thickness=1.0)
        return orbited

    rect = (px0, py0, px1, py1)
    sx, sy, depth = project_points_3d(verts, mvp, rect)
    finite = np.isfinite(sx) & np.isfinite(sy)
    fade = _depth_fade(depth)

    if rgba is None:
        rgba = np.tile(np.asarray(default_rgba, dtype=np.float64), (len(verts), 1))

    draw_list.push_clip_rect((px0, py0), (px1, py1), True)
    try:
        if show_box:
            _draw_scene3d_box(draw_list, _box_points(bounds), mvp, rect, grid_col)
        if triangles is not None:
            _draw_scene3d_triangles(draw_list, triangles, sx, sy, depth, finite, rgba, fade)
        if segments is not None:
            _draw_scene3d_segments(
                draw_list, segments, sx, sy, depth, finite, rgba, fade, thickness
            )
        if dots:
            _draw_scene3d_dots(draw_list, sx, sy, depth, finite, rgba, fade, point_radius)
    finally:
        draw_list.pop_clip_rect()

    draw_list.add_rect((x0, y0), (x1, y1), frame_col, rounding=3.0, thickness=1.0)
    if label:
        draw_list.add_text((px0 + 4.0, py0 + 2.0), muted_col, str(label))
    if hint:
        size = imgui.calc_text_size(hint)
        draw_list.add_text((px0 + 4.0, py1 - size.y - 2.0), muted_col, str(hint))
    return orbited


def _draw_scene3d_box(
    draw_list: Any,
    corners: np.ndarray,
    mvp: np.ndarray,
    rect: Tuple[float, float, float, float],
    col: int,
) -> None:
    """The data's bounding box, twelve faint edges.

    Not decoration: with no box a rotating point cloud has nothing to rotate *against* and
    the motion reads as the points swimming rather than the view turning.
    """
    bx, by, _bd = project_points_3d(corners, mvp, rect)
    for i, j in _BOX_EDGES:
        if not (np.isfinite(bx[i]) and np.isfinite(bx[j])):
            continue
        draw_list.add_line((float(bx[i]), float(by[i])), (float(bx[j]), float(by[j])), col, 1.0)


def _scene3d_order(
    index: Any,
    arity: int,
    depth: np.ndarray,
    finite: np.ndarray,
    limit: int,
) -> Optional[np.ndarray]:
    """``index`` thinned to ``limit`` primitives and sorted back to front.

    Returns None when nothing is drawable. ``index`` must be ``(M, arity)`` vertex
    indices; a primitive survives only if every one of its vertices projected. A wrong
    shape or an out-of-range index drops the whole set rather than raising -- this is
    called from a draw callback, where an exception costs the entire frame, and a caller
    who mixed up "segments" and "triangles" should see nothing rather than a dead window.
    """
    try:
        idx = np.asarray(index, dtype=np.int64)
    except (TypeError, ValueError):
        return None
    if idx.ndim != 2 or idx.shape[0] == 0 or idx.shape[1] != arity:
        return None
    if int(idx.max()) >= depth.size or int(idx.min()) < 0:
        return None
    keep = finite[idx].all(axis=1)
    if not keep.any():
        return None
    idx = idx[keep]
    thin = _thin_indices(len(idx), limit)
    if thin is not None:
        idx = idx[thin]
    return idx[np.argsort(-depth[idx].mean(axis=1), kind="stable")]


def _faded_u32(rgba: np.ndarray, fade: np.ndarray) -> List[int]:
    """Per-primitive packed colours with the depth fade folded into the alpha."""
    shaded = rgba.copy()
    shaded[:, 3] = shaded[:, 3] * fade
    return _pack_u32(shaded).tolist()


def _draw_scene3d_triangles(
    draw_list: Any,
    triangles: Any,
    sx: np.ndarray,
    sy: np.ndarray,
    depth: np.ndarray,
    finite: np.ndarray,
    rgba: np.ndarray,
    fade: np.ndarray,
) -> None:
    """Flat-filled triangles, painted back to front."""
    idx = _scene3d_order(triangles, 3, depth, finite, MAX_SCENE3D_TRIANGLES)
    if idx is None:
        return
    a, b, c = idx[:, 0], idx[:, 1], idx[:, 2]
    # Flat shading from the mean of the three vertex colours: a draw list fills a triangle
    # with one colour, so the alternative is three sub-triangles per triangle for a
    # gradient nobody can see at 190 px.
    face = (rgba[a] + rgba[b] + rgba[c]) / 3.0
    cols = _faded_u32(face, (fade[a] + fade[b] + fade[c]) / 3.0)
    ax, ay = sx[a].tolist(), sy[a].tolist()
    bx, by = sx[b].tolist(), sy[b].tolist()
    cx, cy = sx[c].tolist(), sy[c].tolist()
    for i in range(len(cols)):
        draw_list.add_triangle_filled((ax[i], ay[i]), (bx[i], by[i]), (cx[i], cy[i]), cols[i])


def _draw_scene3d_segments(
    draw_list: Any,
    segments: Any,
    sx: np.ndarray,
    sy: np.ndarray,
    depth: np.ndarray,
    finite: np.ndarray,
    rgba: np.ndarray,
    fade: np.ndarray,
    thickness: float,
) -> None:
    """Line segments, painted back to front, one colour per segment.

    One ``add_line`` per segment rather than an ``add_polyline`` per run: a polyline takes
    a single colour, and the whole point of the preview is that the colormap is visible
    along the object. At the segment cap this costs well under a millisecond.
    """
    idx = _scene3d_order(segments, 2, depth, finite, MAX_SCENE3D_SEGMENTS)
    if idx is None:
        return
    a, b = idx[:, 0], idx[:, 1]
    cols = _faded_u32(rgba[a], (fade[a] + fade[b]) * 0.5)
    ax, ay = sx[a].tolist(), sy[a].tolist()
    bx, by = sx[b].tolist(), sy[b].tolist()
    for i in range(len(cols)):
        draw_list.add_line((ax[i], ay[i]), (bx[i], by[i]), cols[i], thickness)


def _draw_scene3d_dots(
    draw_list: Any,
    sx: np.ndarray,
    sy: np.ndarray,
    depth: np.ndarray,
    finite: np.ndarray,
    rgba: np.ndarray,
    fade: np.ndarray,
    radius: float,
) -> None:
    """One disc per vertex, painted back to front."""
    keep = np.flatnonzero(finite)
    if keep.size == 0:
        return
    thin = _thin_indices(keep.size, MAX_SCENE3D_POINTS)
    if thin is not None:
        keep = keep[thin]
    keep = keep[np.argsort(-depth[keep], kind="stable")]
    cols = _faded_u32(rgba[keep], fade[keep])
    xs, ys = sx[keep].tolist(), sy[keep].tolist()
    r = max(0.75, float(radius))
    for i in range(len(cols)):
        draw_list.add_circle_filled((xs[i], ys[i]), r, cols[i])


def spinner(id_str: str, *, radius: float = 7.0, thickness: float = 2.5) -> None:
    """A small indeterminate busy indicator, themed with the accent color.

    Backed by imgui_bundle's ImSpinner bindings (``imspinner.spinner_ang``). Draws
    dimmed static text instead if that submodule is unavailable -- degrade, don't
    disappear, matching every other optional-dependency widget in this module.
    """
    if not IMGUI_AVAILABLE:
        return
    if not IMSPINNER_AVAILABLE:
        imgui.text_disabled("...")
        return
    accent = _token("accent")
    imspinner.spinner_ang(id_str, radius, thickness, imgui.ImColor(*accent))


def busy_row(label: str, elapsed: float, *, id_str: str, cancellable: bool = True) -> bool:
    """A "<spinner> {label}... {elapsed}s   [Cancel]" row for a pending background job.

    Returns True the frame Cancel is clicked. This widget only reports the click --
    the caller owns actually cancelling the job (BackgroundJob.cancel()).
    """
    if not IMGUI_AVAILABLE:
        return False
    spinner(f"##spinner_{id_str}")
    imgui.same_line()
    imgui.text(f"{label}... {elapsed:.1f}s")
    if not cancellable:
        return False
    imgui.same_line()
    return imgui.button(f"Cancel##{id_str}", (90.0, 0.0))


def draw_toasts() -> None:
    """Render the global notification stack (``notifications.py``) as a column of
    small cards stacked bottom-right, on top of whatever panel is active.

    Call exactly once per frame, after every panel has drawn -- imgui's own draw
    list is composited last, so anything drawn after ``workspace.draw()`` sits on top
    of the 3D scene and every panel, which is what makes a toast visible regardless of
    which panel (if any) currently has focus.
    """
    if not IMGUI_AVAILABLE:
        return
    try:
        from . import notifications
    except Exception:  # pragma: no cover - defensive, mirrors _icon()'s own guard
        return

    toasts = notifications.active()
    if not toasts:
        return

    from . import icons

    viewport = imgui.get_main_viewport()
    margin = 16.0
    width = 320.0
    y = viewport.work_pos.y + viewport.work_size.y - margin
    kind_token = {"info": "accent", "success": "ok", "error": "err"}
    flags = (
        imgui.WindowFlags_.no_decoration
        | imgui.WindowFlags_.no_move
        | imgui.WindowFlags_.no_saved_settings
        | imgui.WindowFlags_.no_focus_on_appearing
        | imgui.WindowFlags_.no_nav
        | imgui.WindowFlags_.always_auto_resize
    )

    # Newest toast at the bottom (the stack grows upward from there), so a fresh
    # notification never shoves an older one somewhere the user wasn't looking.
    for toast in reversed(toasts):
        text = str(toast.message)
        wrap_width = width - 44.0
        text_h = imgui.calc_text_size(
            text, hide_text_after_double_hash=False, wrap_width=wrap_width
        ).y
        height = max(32.0, text_h + 16.0)
        y -= height + 8.0

        imgui.set_next_window_pos((viewport.work_pos.x + viewport.work_size.x - margin - width, y))
        imgui.set_next_window_size((width, height))
        border = _token(kind_token.get(toast.kind, "accent"))
        imgui.push_style_color(imgui.Col_.border, border)
        imgui.push_style_var(imgui.StyleVar_.window_border_size, 1.5)
        expanded, _p_open = imgui.begin(f"##toast_{toast.id}", flags=flags)
        try:
            if expanded:
                imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + wrap_width)
                imgui.text_unformatted(text)
                imgui.pop_text_wrap_pos()
                imgui.same_line(width - 34.0)
                if icons.icon_button(f"##toast_close_{toast.id}", "close", size=16.0):
                    notifications.dismiss(toast.id)
        finally:
            # CONTRACT 2.2: end() is unconditional, exactly like end_child()/end().
            imgui.end()
            imgui.pop_style_var(1)
            imgui.pop_style_color(1)
