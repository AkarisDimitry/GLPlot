"""Inline contour labels for the live GL window: the value, seated on its own level line.

matplotlib's ``clabel`` splits the contour path and drops a rotated ``Text`` into the gap.
GLPlot has no path-splitting machinery live -- but it does have the geometry, which is the
part that matters: :func:`glplot.pyplot.contour` draws one real polyline layer per level,
each tagged ``artist="contour_line"`` and carrying its own ``level``. That is enough to put
the number *on* the line, and to clear a small patch of background behind it so the line
appears to break for it. Only the reconstruction is missing, not the data -- which is why
this exists rather than the live window silently dropping the labels the export draws.

Deliberately simpler than matplotlib's placement, and the differences are visible:
  * one label per level, not one per disconnected run of it;
  * centred on the middle of the level's *currently visible* run, so it follows pan/zoom
    instead of drifting off-screen or pinning to a vertex that has scrolled away;
  * horizontal, never rotated to the local tangent.
The export still goes through real matplotlib (``utils/preview.py``), so a saved figure
keeps matplotlib's own placement -- these two are not pixel-identical and are not meant
to be.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from .legend import _background_is_light

#: Default label size in px when ``clabel(fontsize=...)`` names none. Matches the engine's
#: own chrome rather than the export's much larger point sizes -- a contour label sits
#: *inside* the data area, where the export's 24pt would swamp the lines it annotates.
_DEFAULT_TEXT_PX = 14.0

#: Padding around the cleared patch behind a label, in px. This is what stands in for
#: matplotlib's broken path: enough to read as a gap in the line, not so much that it
#: punches a visible hole in a dense contour field.
_CLEAR_PAD_X_PX = 4.0
_CLEAR_PAD_Y_PX = 1.0


def draw_contour_labels(plot: Any, ctx: Any) -> int:
    """Draw every requested contour label for the active panel. Returns how many it drew.

    Called from the engine's per-panel overlay pass, beside the axis labels and the
    legend, so it inherits that pass's already-swapped per-panel camera and pixel size.
    Returns the count so a test can assert placement happened without reading pixels.
    """
    layers = _labelled_layers(plot)
    if not layers:
        return 0
    try:
        from imgui_bundle import imgui
    except (ImportError, Exception):  # noqa: B014 - imgui raises non-ImportError GL-less
        return 0

    options = getattr(plot, "options", None)
    is_light = _background_is_light(options)
    ink = imgui.get_color_u32((0.15, 0.15, 0.15, 1.0) if is_light else (0.9, 0.9, 0.9, 1.0))
    clear = _background_u32(imgui, options, is_light)
    draw_list = imgui.get_background_draw_list()

    project = _projector(ctx)
    view = _view_rect(ctx)
    placed: List[Tuple[float, float, float, float]] = []
    drawn = 0
    for layer in layers:
        request = layer.metadata.get("clabel") or {}
        text = _format_level(layer.metadata.get("level"), request.get("fmt"))
        if not text:
            continue
        px = float(request.get("fontsize") or _DEFAULT_TEXT_PX)
        w, h = _measure(imgui, text, px)
        for anchor in _candidate_anchors(layer, project, view):
            rect = _label_rect(anchor, w, h)
            if _overlaps(rect, placed):
                continue
            _draw_one(imgui, draw_list, anchor, text, px, w, h, ink, clear)
            placed.append(rect)
            drawn += 1
            break
        # Falling through means every candidate collided: this level goes unlabelled
        # rather than stacking a number on top of a neighbour's. Zooming in separates
        # the lines and the label reappears.
    return drawn


def _labelled_layers(plot: Any) -> List[Any]:
    """The active panel's contour-line layers that ``clabel()`` marked."""
    panel = getattr(plot, "active_panel", None)
    scene = getattr(panel, "scene", None)
    out = []
    for layer in getattr(scene, "layers", None) or []:
        meta = getattr(layer, "metadata", None)
        if not isinstance(meta, dict):
            continue
        if meta.get("artist") == "contour_line" and meta.get("clabel") and layer.pts is not None:
            out.append(layer)
    return out


def _projector(ctx: Any):
    """World -> window pixels, the same transform ``AxisRenderer._draw_labels`` builds.

    ``px_offset`` shifts panel-local pixels into window space, so a split figure's labels
    land over their own panel rather than over the window's corner.
    """
    off_x, off_y = ctx.px_offset
    mvp = ctx.mvp

    def project(points: np.ndarray) -> np.ndarray:
        n = len(points)
        homogeneous = np.empty((n, 4), dtype=np.float64)
        homogeneous[:, 0] = points[:, 0]
        homogeneous[:, 1] = points[:, 1]
        homogeneous[:, 2] = 0.0
        homogeneous[:, 3] = 1.0
        # One matmul for the whole polyline: this runs per contour level, every frame.
        ndc = homogeneous @ np.asarray(mvp, dtype=np.float64).T
        w = ndc[:, 3]
        w = np.where(np.abs(w) < 1e-12, 1.0, w)
        ndc = ndc / w[:, None]
        screen = np.empty((n, 2), dtype=np.float64)
        screen[:, 0] = (ndc[:, 0] + 1.0) * 0.5 * ctx.width_px + off_x
        screen[:, 1] = (1.0 - ndc[:, 1]) * 0.5 * ctx.height_px + off_y
        return screen

    return project


def _view_rect(ctx: Any) -> Tuple[float, float, float, float]:
    """The panel's own window-pixel rect, as ``(left, top, right, bottom)``."""
    off_x, off_y = ctx.px_offset
    return (off_x, off_y, off_x + ctx.width_px, off_y + ctx.height_px)


#: Where along a level's visible run to try seating its label, as fractions of that run.
#: Several, because nested contours (a dipole's rings, a peak's shells) all have their
#: midpoint at nearly the same screen angle -- placing every one at 0.5 stacks the whole
#: family into one unreadable column. The first candidate that does not collide wins.
_ANCHOR_FRACTIONS = (0.5, 0.25, 0.75, 0.12, 0.88, 0.38, 0.62)


def _label_anchor(layer: Any, project, view) -> Optional[Tuple[float, float]]:
    """The preferred anchor: the middle of this level's longest on-screen run.

    Kept as its own function because it is the placement rule worth pinning in a test;
    :func:`_candidate_anchors` generalises it to the fallbacks used to dodge collisions.
    """
    candidates = _candidate_anchors(layer, project, view)
    return candidates[0] if candidates else None


def _candidate_anchors(layer: Any, project, view) -> List[Tuple[float, float]]:
    """Anchors to try for this level, best first, all on its longest visible run.

    Using the *visible* run rather than a fixed vertex is what keeps the label on the line
    while the user pans and zooms -- a fixed index scrolls off with the geometry and takes
    the annotation with it.
    """
    pts = np.asarray(getattr(layer, "pts", None), dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 2:
        return []
    screen = project(pts)
    left, top, right, bottom = view
    inside = (
        (screen[:, 0] >= left)
        & (screen[:, 0] <= right)
        & (screen[:, 1] >= top)
        & (screen[:, 1] <= bottom)
        & np.isfinite(screen).all(axis=1)
    )
    if not inside.any():
        return []
    start, length = _longest_run(inside)
    if length < 2:
        return []

    out = []
    for frac in _ANCHOR_FRACTIONS:
        idx = start + min(int(length * frac), length - 1)
        point = screen[idx]
        out.append((float(point[0]), float(point[1])))
    return out


def _overlaps(rect, placed) -> bool:
    """True when ``rect`` intersects any already-placed label rect."""
    ax0, ay0, ax1, ay1 = rect
    for bx0, by0, bx1, by1 in placed:
        if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
            return True
    return False


def _longest_run(mask: np.ndarray) -> Tuple[int, int]:
    """``(start, length)`` of the longest run of True in ``mask``."""
    best_start = best_len = 0
    run_start = -1
    for i, flag in enumerate(mask):
        if flag:
            if run_start < 0:
                run_start = i
        elif run_start >= 0:
            if i - run_start > best_len:
                best_start, best_len = run_start, i - run_start
            run_start = -1
    if run_start >= 0 and len(mask) - run_start > best_len:
        best_start, best_len = run_start, len(mask) - run_start
    return best_start, best_len


def _format_level(level: Any, fmt: Any) -> str:
    """The label text for ``level``, honouring ``clabel(fmt=...)``."""
    if level is None:
        return ""
    value = float(level)
    if fmt is not None:
        try:
            if callable(fmt):
                return str(fmt(value))
            spec = str(fmt)
            return spec % value if "%" in spec else spec.format(value)
        except Exception:
            pass  # fall through to the default rather than drawing nothing
    return f"{value:g}"


def _background_u32(imgui, options: Any, is_light: bool) -> int:
    """The colour to clear behind a label: the scene's own background, fully opaque.

    Reading the real background (rather than assuming white) is what lets this stand in
    for a broken path on a styled figure -- on "dark"/"neon"/"chalk" a white patch would
    be a bright hole rather than a gap.
    """
    visual = getattr(options, "visual", None)
    gradient = getattr(visual, "gradient_background", None)
    if gradient is not None and getattr(gradient, "enabled", False):
        top, bottom = gradient.top_color, gradient.bottom_color
        rgb = tuple(0.5 * (top[i] + bottom[i]) for i in range(3))
    else:
        rgb = tuple(getattr(visual, "background_color", (1.0, 1.0, 1.0))[:3])
    if rgb == (0.0, 0.0, 0.0) and is_light:  # defensive: a mis-set background
        rgb = (1.0, 1.0, 1.0)
    return imgui.get_color_u32((rgb[0], rgb[1], rgb[2], 1.0))


def _measure(imgui, text: str, px: float) -> Tuple[float, float]:
    """``(width, height)`` of ``text`` at ``px``.

    ``calc_text_size`` measures at the *current* font size, so laying sized text out by it
    directly would centre every label against the wrong extent.
    """
    w, h = imgui.calc_text_size(text)
    scale = px / (imgui.get_font_size() or 1.0)
    return w * scale, h * scale


def _label_rect(anchor, w: float, h: float) -> Tuple[float, float, float, float]:
    """The cleared patch a label centred on ``anchor`` would occupy."""
    cx, cy = anchor
    x, y = cx - w * 0.5, cy - h * 0.5
    return (
        x - _CLEAR_PAD_X_PX,
        y - _CLEAR_PAD_Y_PX,
        x + w + _CLEAR_PAD_X_PX,
        y + h + _CLEAR_PAD_Y_PX,
    )


def _draw_one(imgui, draw_list, anchor, text, px, w, h, ink, clear) -> None:
    """Clear a patch of background, then seat the number in it, centred on ``anchor``."""
    x0, y0, x1, y1 = _label_rect(anchor, w, h)
    draw_list.add_rect_filled((x0, y0), (x1, y1), clear)
    draw_list.add_text(imgui.get_font(), px, (x0 + _CLEAR_PAD_X_PX, y0 + _CLEAR_PAD_Y_PX), ink, text)
