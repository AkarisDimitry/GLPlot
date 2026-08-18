"""Vector icons drawn with ``draw_list`` primitives. No font, no bundled asset.

Rationale is CONTRACT section 3: the default ProggyClean atlas covers U+0020-U+00FF only,
so every arrow/gear/glyph codepoint renders as ``?`` -- and ``calc_text_size`` cannot
detect it, because ProggyClean measures absent glyphs at the same fixed width as present
ones. A bundled TTF is excluded by constraint and a per-OS font path table degrades
silently. Hand-built geometry costs a few lines per icon, is identical on all three
platforms, and scales to any size.

**Geometry convention.** Each shape is authored in a normalised space where ``(0, 0)`` is
the icon centre and ``+-1`` is the icon radius ``r``, with **y pointing down** (screen
space). The ``_line``/``_poly``/``_rect``/``_circle`` helpers do the mapping, so shape
bodies read as pure geometry. Keep shapes inside ``[-1, 1]`` on both axes; ``icon_button``
sizes ``r`` so that a small overshoot still fits the button's padding.

**The placement rule is load-bearing** (CONTRACT section 3): ``get_cursor_screen_pos()``
must be called *before* ``invisible_button``. It reports the screen-space top-left of the
*next* item; after the button the cursor has advanced and the icon draws in the wrong
place -- typically one slot to the right, which looks like an off-by-one in the toolbar.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .theme import COLORS

try:
    from imgui_bundle import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - imgui is a hard dependency in CI
    IMGUI_AVAILABLE = False
    imgui = None

logger = logging.getLogger(__name__)

__all__ = ["ICON_SHAPES", "IMGUI_AVAILABLE", "draw_icon", "icon_button", "icon_inline"]

Point = Tuple[float, float]

_MISSING_WARNED = set()


# --- normalised drawing helpers ---------------------------------------------


def _stroke(r: float) -> float:
    """Stroke width for an icon of radius ``r``. ~2px at the default 22px button."""
    return max(1.4, r * 0.28)


def _line(dl, cx: float, cy: float, r: float, col: int, t: float, x0, y0, x1, y1) -> None:
    """Line in normalised icon space."""
    dl.add_line((cx + x0 * r, cy + y0 * r), (cx + x1 * r, cy + y1 * r), col, t)


def _poly(dl, cx, cy, r, col, t, pts: Sequence[Point], closed: bool = False) -> None:
    """Polyline in normalised icon space."""
    flags = imgui.ImDrawFlags_.closed if closed else 0
    dl.add_polyline([(cx + x * r, cy + y * r) for x, y in pts], col, flags=flags, thickness=t)


def _rect(dl, cx, cy, r, col, t, x0, y0, x1, y1, rounding: float = 0.0) -> None:
    """Outlined rect in normalised icon space (``rounding`` is also normalised)."""
    dl.add_rect(
        (cx + x0 * r, cy + y0 * r),
        (cx + x1 * r, cy + y1 * r),
        col,
        rounding=rounding * r,
        thickness=t,
    )


def _rect_filled(dl, cx, cy, r, col, x0, y0, x1, y1, rounding: float = 0.0) -> None:
    """Filled rect in normalised icon space."""
    dl.add_rect_filled((cx + x0 * r, cy + y0 * r), (cx + x1 * r, cy + y1 * r), col, rounding * r, 0)


def _circle(dl, cx, cy, r, col, t, x, y, rad, segments: int = 20) -> None:
    """Outlined circle in normalised icon space."""
    dl.add_circle((cx + x * r, cy + y * r), rad * r, col, segments, t)


def _dot(dl, cx, cy, r, col, x, y, rad) -> None:
    """Filled circle in normalised icon space."""
    dl.add_circle_filled((cx + x * r, cy + y * r), rad * r, col, 14)


def _tri_filled(dl, cx, cy, r, col, p0: Point, p1: Point, p2: Point) -> None:
    """Filled triangle in normalised icon space."""
    dl.add_triangle_filled(
        (cx + p0[0] * r, cy + p0[1] * r),
        (cx + p1[0] * r, cy + p1[1] * r),
        (cx + p2[0] * r, cy + p2[1] * r),
        col,
    )


def _bezier(dl, cx, cy, r, col, t, p0: Point, p1: Point, p2: Point, p3: Point) -> None:
    """Cubic bezier in normalised icon space. ``thickness`` is required in pyimgui 2.0."""
    dl.add_bezier_cubic(
        (cx + p0[0] * r, cy + p0[1] * r),
        (cx + p1[0] * r, cy + p1[1] * r),
        (cx + p2[0] * r, cy + p2[1] * r),
        (cx + p3[0] * r, cy + p3[1] * r),
        col,
        t,
    )


def _arc(dl, cx, cy, r, col, t, x, y, rad, a_min, a_max, segments: int = 24) -> None:
    """Stroked arc in normalised icon space. Angles are radians, clockwise (y is down)."""
    dl.path_arc_to((cx + x * r, cy + y * r), rad * r, a_min, a_max, segments)
    dl.path_stroke(col, thickness=t)


# --- shapes: edit ------------------------------------------------------------


def _plus(dl, cx, cy, r, col, t) -> None:
    _line(dl, cx, cy, r, col, t, -0.85, 0.0, 0.85, 0.0)
    _line(dl, cx, cy, r, col, t, 0.0, -0.85, 0.0, 0.85)


def _minus(dl, cx, cy, r, col, t) -> None:
    _line(dl, cx, cy, r, col, t, -0.85, 0.0, 0.85, 0.0)


def _trash(dl, cx, cy, r, col, t) -> None:
    _line(dl, cx, cy, r, col, t, -1.0, -0.5, 1.0, -0.5)  # lid
    _poly(dl, cx, cy, r, col, t, [(-0.3, -0.5), (-0.3, -0.85), (0.3, -0.85), (0.3, -0.5)])
    _poly(dl, cx, cy, r, col, t, [(-0.72, -0.32), (-0.58, 0.92), (0.58, 0.92), (0.72, -0.32)])
    _line(dl, cx, cy, r, col, t * 0.8, -0.24, -0.08, -0.2, 0.62)  # ribs
    _line(dl, cx, cy, r, col, t * 0.8, 0.24, -0.08, 0.2, 0.62)


def _copy(dl, cx, cy, r, col, t) -> None:
    # Back sheet drawn as the L-shaped portion the front sheet does not occlude, so the
    # two rects read as stacked paper without needing an opaque fill.
    _poly(
        dl,
        cx,
        cy,
        r,
        col,
        t,
        [(0.25, -0.25), (0.25, -0.95), (-0.95, -0.95), (-0.95, 0.25), (-0.25, 0.25)],
    )
    _rect(dl, cx, cy, r, col, t, -0.25, -0.25, 0.95, 0.95, rounding=0.15)


def _paste(dl, cx, cy, r, col, t) -> None:
    _rect(dl, cx, cy, r, col, t, -0.8, -0.7, 0.8, 0.95, rounding=0.18)  # board
    _rect_filled(dl, cx, cy, r, col, -0.38, -0.95, 0.38, -0.45, rounding=0.12)  # clip
    _line(dl, cx, cy, r, col, t * 0.8, -0.42, 0.1, 0.42, 0.1)  # content
    _line(dl, cx, cy, r, col, t * 0.8, -0.42, 0.5, 0.42, 0.5)


def _cut(dl, cx, cy, r, col, t) -> None:
    _line(dl, cx, cy, r, col, t, -0.42, 0.5, 0.58, -0.95)  # blades cross near the middle
    _line(dl, cx, cy, r, col, t, 0.42, 0.5, -0.58, -0.95)
    _circle(dl, cx, cy, r, col, t, -0.5, 0.72, 0.28, 14)  # handles
    _circle(dl, cx, cy, r, col, t, 0.5, 0.72, 0.28, 14)


def _rotate_arrow(dl, cx, cy, r, col, t, sign: float) -> None:
    """Curl arrow. ``sign=+1`` puts the head on the left (undo), ``-1`` on the right (redo)."""
    _arc(dl, cx, cy, r, col, t, 0.0, 0.25, 0.65, math.pi, math.tau, 22)
    hx, tx = -0.65 * sign, 0.65 * sign
    _line(dl, cx, cy, r, col, t, tx, 0.25, tx, 0.92)  # tail leg
    _line(dl, cx, cy, r, col, t, hx, 0.25, hx - 0.3, -0.07)  # head, pointing down
    _line(dl, cx, cy, r, col, t, hx, 0.25, hx + 0.3, -0.07)


def _undo(dl, cx, cy, r, col, t) -> None:
    _rotate_arrow(dl, cx, cy, r, col, t, 1.0)


def _redo(dl, cx, cy, r, col, t) -> None:
    _rotate_arrow(dl, cx, cy, r, col, t, -1.0)


# --- shapes: transport / system ---------------------------------------------


def _play(dl, cx, cy, r, col, t) -> None:
    _tri_filled(dl, cx, cy, r, col, (-0.55, -0.9), (0.85, 0.0), (-0.55, 0.9))


def _pause(dl, cx, cy, r, col, t) -> None:
    _rect_filled(dl, cx, cy, r, col, -0.62, -0.85, -0.18, 0.85, rounding=0.08)
    _rect_filled(dl, cx, cy, r, col, 0.18, -0.85, 0.62, 0.85, rounding=0.08)


def _refresh(dl, cx, cy, r, col, t) -> None:
    # ~333 degrees of arc; the gap sits top-right and the head points into it.
    _arc(dl, cx, cy, r, col, t, 0.0, 0.0, 0.68, -0.35 * math.pi, 1.5 * math.pi, 30)
    _line(dl, cx, cy, r, col, t, 0.0, -0.68, -0.3, -0.98)
    _line(dl, cx, cy, r, col, t, 0.0, -0.68, -0.3, -0.38)


def _gear(dl, cx, cy, r, col, t, teeth: int = 8) -> None:
    # Trapezoidal teeth, not alternating radii: simply alternating outer/inner points
    # renders as a spiky star, because each tooth collapses to a single vertex. Each
    # tooth needs a flat top (two points at the outer radius) and a flat gap floor.
    step = math.tau / teeth
    w_out, w_in, r_in = step * 0.16, step * 0.30, 0.68
    pts = []
    for i in range(teeth):
        a = i * step
        for da, rr in ((-w_out, 1.0), (w_out, 1.0), (w_in, r_in), (step - w_in, r_in)):
            pts.append((math.cos(a + da) * rr, math.sin(a + da) * rr))
    _poly(dl, cx, cy, r, col, t, pts, closed=True)
    _circle(dl, cx, cy, r, col, t, 0.0, 0.0, 0.32, 14)


# --- shapes: charts / data ---------------------------------------------------


def _axes(dl, cx, cy, r, col, t) -> None:
    """Shared L-shaped axis frame for the chart_* icons."""
    _poly(dl, cx, cy, r, col, t, [(-0.9, -0.9), (-0.9, 0.9), (0.9, 0.9)])


def _chart_line(dl, cx, cy, r, col, t) -> None:
    _axes(dl, cx, cy, r, col, t)
    _poly(dl, cx, cy, r, col, t, [(-0.6, 0.35), (-0.15, -0.3), (0.25, 0.1), (0.75, -0.65)])


def _chart_scatter(dl, cx, cy, r, col, t) -> None:
    _axes(dl, cx, cy, r, col, t)
    for x, y in ((-0.45, 0.4), (-0.05, -0.15), (0.35, 0.25), (0.7, -0.55)):
        _dot(dl, cx, cy, r, col, x, y, 0.18)


def _chart_bar(dl, cx, cy, r, col, t) -> None:
    _axes(dl, cx, cy, r, col, t)
    for x0, y0 in ((-0.6, 0.05), (-0.1, -0.5), (0.4, -0.2)):
        _rect_filled(dl, cx, cy, r, col, x0, y0, x0 + 0.35, 0.9)


def _table(dl, cx, cy, r, col, t) -> None:
    _rect(dl, cx, cy, r, col, t, -0.95, -0.8, 0.95, 0.8, rounding=0.1)
    _rect_filled(dl, cx, cy, r, col, -0.95, -0.8, 0.95, -0.35, rounding=0.1)  # header band
    _line(dl, cx, cy, r, col, t * 0.8, -0.95, 0.15, 0.95, 0.15)
    _line(dl, cx, cy, r, col, t * 0.8, -0.28, -0.35, -0.28, 0.8)
    _line(dl, cx, cy, r, col, t * 0.8, 0.35, -0.35, 0.35, 0.8)


def _grid(dl, cx, cy, r, col, t) -> None:
    _rect(dl, cx, cy, r, col, t, -0.9, -0.9, 0.9, 0.9, rounding=0.08)
    _line(dl, cx, cy, r, col, t * 0.8, -0.3, -0.9, -0.3, 0.9)
    _line(dl, cx, cy, r, col, t * 0.8, 0.3, -0.9, 0.3, 0.9)
    _line(dl, cx, cy, r, col, t * 0.8, -0.9, -0.3, 0.9, -0.3)
    _line(dl, cx, cy, r, col, t * 0.8, -0.9, 0.3, 0.9, 0.3)


# --- shapes: math ------------------------------------------------------------


def _function(dl, cx, cy, r, col, t) -> None:
    # An "f(x)" glyph is impossible without a font (CONTRACT section 3), so: a curve in a box.
    _rect(dl, cx, cy, r, col, t, -0.95, -0.95, 0.95, 0.95, rounding=0.22)
    _bezier(dl, cx, cy, r, col, t, (-0.55, 0.55), (-0.1, 0.6), (0.05, -0.6), (0.55, -0.55))


def _integral(dl, cx, cy, r, col, t) -> None:
    # A single cubic gives the whole glyph: top hook right, sweep down, bottom hook left.
    _bezier(dl, cx, cy, r, col, t * 1.05, (0.5, -0.95), (-0.55, -0.75), (0.55, 0.75), (-0.5, 0.95))


def _derivative(dl, cx, cy, r, col, t) -> None:
    # Tangent at a minimum, not on a flank. A tangent anywhere on a monotone flank runs
    # along the curve for most of its length and the two strokes merge into one thick
    # diagonal at icon sizes; at the extremum the curve peels away from the horizontal
    # fast enough to stay legible at 22px. Bezier bottom B(0.5).y = 0.7625 by symmetry.
    _bezier(dl, cx, cy, r, col, t, (-0.85, -0.85), (-0.15, 1.3), (0.15, 1.3), (0.85, -0.85))
    _line(dl, cx, cy, r, col, t * 0.8, -0.92, 0.7625, 0.92, 0.7625)
    _dot(dl, cx, cy, r, col, 0.0, 0.7625, 0.13)


def _wave(dl, cx, cy, r, col, t) -> None:
    n = 17
    pts = []
    for i in range(n):
        fx = -0.95 + 1.9 * i / (n - 1)
        pts.append((fx, -0.68 * math.sin(math.tau * (fx + 0.95) / 1.9)))
    _poly(dl, cx, cy, r, col, t, pts)


# Three zigzag cycles. Eleven points looked convincingly random at 96px but aliased into
# an unreadable scribble at 22px -- the toolbar size is the one that has to work.
_NOISE_Y = (0.3, -0.55, 0.05, -0.85, 0.5, -0.35, 0.75)


def _noise(dl, cx, cy, r, col, t) -> None:
    n = len(_NOISE_Y)
    pts = [(-0.95 + 1.9 * i / (n - 1), y) for i, y in enumerate(_NOISE_Y)]
    _poly(dl, cx, cy, r, col, t, pts)


# Four zigzag cycles, no more: at 22px a denser or steeper wobble stops reading as a
# jagged line and fills in as a solid band.
_WOBBLE = (0.0, 0.3, -0.26, 0.28, -0.24, 0.3, -0.28, 0.26, 0.0)


def _base_trend(fx: float) -> float:
    """The smooth icon's underlying signal: a monotone S-ramp (y is down, so it rises)."""
    return 0.5 * math.cos(math.pi * (fx + 0.95) / 1.9)


def _smooth(dl, cx, cy, r, col, t) -> None:
    # Noisy input wobbling *about* the clean output, both drawn: the icon must say
    # "smoothed, overlaid on the original" -- exactly what the mathlab panel does.
    # The trend is an S-ramp rather than a sine on purpose: with a sine underneath, this
    # icon collapses into `wave` once the wobble is small enough to read as noise.
    n = len(_WOBBLE)
    jag = [
        (-0.95 + 1.9 * i / (n - 1), _base_trend(-0.95 + 1.9 * i / (n - 1)) + w)
        for i, w in enumerate(_WOBBLE)
    ]
    _poly(dl, cx, cy, r, col, t * 0.45, jag)  # the noisy input, hairline
    clean = [(-0.95 + 1.9 * i / 16.0, _base_trend(-0.95 + 1.9 * i / 16.0)) for i in range(17)]
    _poly(dl, cx, cy, r, col, t * 1.1, clean)  # the smoothed output, full weight


def _filter(dl, cx, cy, r, col, t) -> None:
    _poly(
        dl,
        cx,
        cy,
        r,
        col,
        t,
        [(-0.95, -0.8), (0.95, -0.8), (0.22, 0.0), (0.22, 0.9), (-0.22, 0.6), (-0.22, 0.0)],
        closed=True,
    )


# --- shapes: visibility / files ---------------------------------------------


def _eye(dl, cx, cy, r, col, t) -> None:
    dl.add_bezier_quadratic((cx - 0.95 * r, cy), (cx, cy - 1.3 * r), (cx + 0.95 * r, cy), col, t)
    dl.add_bezier_quadratic((cx - 0.95 * r, cy), (cx, cy + 1.3 * r), (cx + 0.95 * r, cy), col, t)
    _circle(dl, cx, cy, r, col, t, 0.0, 0.0, 0.32, 14)
    _dot(dl, cx, cy, r, col, 0.0, 0.0, 0.12)


def _eye_off(dl, cx, cy, r, col, t) -> None:
    _eye(dl, cx, cy, r, col, t)
    _line(dl, cx, cy, r, col, t, -0.85, 0.85, 0.85, -0.85)


def _lock(dl, cx, cy, r, col, t) -> None:
    _arc(dl, cx, cy, r, col, t, 0.0, -0.18, 0.45, math.pi, math.tau, 16)  # shackle
    _line(dl, cx, cy, r, col, t, -0.45, -0.18, -0.45, 0.05)
    _line(dl, cx, cy, r, col, t, 0.45, -0.18, 0.45, 0.05)
    _rect(dl, cx, cy, r, col, t, -0.75, 0.05, 0.75, 0.9, rounding=0.15)  # body
    _dot(dl, cx, cy, r, col, 0.0, 0.38, 0.13)  # keyhole
    _line(dl, cx, cy, r, col, t * 0.8, 0.0, 0.42, 0.0, 0.68)


def _save(dl, cx, cy, r, col, t) -> None:
    _poly(
        dl,
        cx,
        cy,
        r,
        col,
        t,
        [(-0.9, -0.9), (0.6, -0.9), (0.9, -0.6), (0.9, 0.9), (-0.9, 0.9)],
        closed=True,
    )
    _rect_filled(dl, cx, cy, r, col, -0.45, -0.9, 0.4, -0.3)  # shutter
    _rect(dl, cx, cy, r, col, t * 0.8, -0.55, 0.2, 0.55, 0.9)  # label


def _folder(dl, cx, cy, r, col, t) -> None:
    _poly(
        dl,
        cx,
        cy,
        r,
        col,
        t,
        [(-0.95, 0.75), (-0.95, -0.7), (-0.25, -0.7), (-0.02, -0.4), (0.95, -0.4), (0.95, 0.75)],
        closed=True,
    )


def _tray(dl, cx, cy, r, col, t) -> None:
    """Shared open-tray base for download/upload."""
    _poly(dl, cx, cy, r, col, t, [(-0.85, 0.4), (-0.85, 0.9), (0.85, 0.9), (0.85, 0.4)])


def _download(dl, cx, cy, r, col, t) -> None:
    _tray(dl, cx, cy, r, col, t)
    _line(dl, cx, cy, r, col, t, 0.0, -0.9, 0.0, 0.3)
    _line(dl, cx, cy, r, col, t, 0.0, 0.3, -0.42, -0.15)
    _line(dl, cx, cy, r, col, t, 0.0, 0.3, 0.42, -0.15)


def _upload(dl, cx, cy, r, col, t) -> None:
    _tray(dl, cx, cy, r, col, t)
    _line(dl, cx, cy, r, col, t, 0.0, 0.3, 0.0, -0.9)
    _line(dl, cx, cy, r, col, t, 0.0, -0.9, -0.42, -0.45)
    _line(dl, cx, cy, r, col, t, 0.0, -0.9, 0.42, -0.45)


# --- shapes: navigation ------------------------------------------------------


def _search(dl, cx, cy, r, col, t) -> None:
    _circle(dl, cx, cy, r, col, t, -0.15, -0.15, 0.6, 18)
    d = 0.6 * 0.707
    _line(dl, cx, cy, r, col, t, -0.15 + d, -0.15 + d, 0.85, 0.85)


def _zoom_in(dl, cx, cy, r, col, t) -> None:
    _search(dl, cx, cy, r, col, t)
    _line(dl, cx, cy, r, col, t * 0.85, -0.45, -0.15, 0.15, -0.15)
    _line(dl, cx, cy, r, col, t * 0.85, -0.15, -0.45, -0.15, 0.15)


def _zoom_out(dl, cx, cy, r, col, t) -> None:
    _search(dl, cx, cy, r, col, t)
    _line(dl, cx, cy, r, col, t * 0.85, -0.45, -0.15, 0.15, -0.15)


def _home(dl, cx, cy, r, col, t) -> None:
    _poly(dl, cx, cy, r, col, t, [(-0.95, 0.0), (0.0, -0.85), (0.95, 0.0)])  # roof
    _poly(dl, cx, cy, r, col, t, [(-0.7, -0.06), (-0.7, 0.9), (0.7, 0.9), (0.7, -0.06)])
    _rect(dl, cx, cy, r, col, t * 0.8, -0.22, 0.35, 0.22, 0.9)  # door


def _target(dl, cx, cy, r, col, t) -> None:
    _circle(dl, cx, cy, r, col, t, 0.0, 0.0, 0.82, 24)
    _circle(dl, cx, cy, r, col, t * 0.85, 0.0, 0.0, 0.4, 16)
    _dot(dl, cx, cy, r, col, 0.0, 0.0, 0.13)
    for x0, y0, x1, y1 in (
        (-1.0, 0.0, -0.6, 0.0),
        (0.6, 0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0, -0.6),
        (0.0, 0.6, 0.0, 1.0),
    ):
        _line(dl, cx, cy, r, col, t, x0, y0, x1, y1)


def _chevron(dl, cx, cy, r, col, t, direction: str) -> None:
    pts = {
        "right": [(-0.35, -0.8), (0.4, 0.0), (-0.35, 0.8)],
        "left": [(0.35, -0.8), (-0.4, 0.0), (0.35, 0.8)],
        "down": [(-0.8, -0.35), (0.0, 0.4), (0.8, -0.35)],
        "up": [(-0.8, 0.35), (0.0, -0.4), (0.8, 0.35)],
    }[direction]
    _poly(dl, cx, cy, r, col, t, pts)


# --- shapes: scene / style ---------------------------------------------------


def _layers(dl, cx, cy, r, col, t) -> None:
    _poly(
        dl,
        cx,
        cy,
        r,
        col,
        t,
        [(0.0, -0.9), (0.95, -0.45), (0.0, 0.0), (-0.95, -0.45)],
        closed=True,
    )
    _poly(dl, cx, cy, r, col, t, [(-0.95, 0.1), (0.0, 0.55), (0.95, 0.1)])
    _poly(dl, cx, cy, r, col, t, [(-0.95, 0.5), (0.0, 0.95), (0.95, 0.5)])


def _palette(dl, cx, cy, r, col, t) -> None:
    _circle(dl, cx, cy, r, col, t, 0.0, 0.0, 0.9, 24)
    for x, y in ((-0.44, -0.32), (0.08, -0.56), (0.52, -0.08)):
        _dot(dl, cx, cy, r, col, x, y, 0.19)  # ~1.3px at 22px; smaller vanishes
    _circle(dl, cx, cy, r, col, t * 0.85, 0.05, 0.46, 0.27, 12)  # thumb hole


def _sliders(dl, cx, cy, r, col, t) -> None:
    for y, knob_x in ((-0.55, -0.25), (0.0, 0.35), (0.55, -0.05)):
        _line(dl, cx, cy, r, col, t * 0.8, -0.95, y, 0.95, y)
        _dot(dl, cx, cy, r, col, knob_x, y, 0.22)


def _link(dl, cx, cy, r, col, t) -> None:
    # Two three-quarter arcs, each opening toward the other, joined by a bar.
    _arc(dl, cx, cy, r, col, t, 0.45, -0.45, 0.45, 1.1 * math.pi, 2.4 * math.pi, 20)
    _arc(dl, cx, cy, r, col, t, -0.45, 0.45, 0.45, 0.1 * math.pi, 1.4 * math.pi, 20)
    _line(dl, cx, cy, r, col, t, -0.3, 0.3, 0.3, -0.3)


# --- shapes: status ----------------------------------------------------------


def _info(dl, cx, cy, r, col, t) -> None:
    _circle(dl, cx, cy, r, col, t, 0.0, 0.0, 0.88, 24)
    # The tittle needs a real gap above the stem or the two blur into one bar at 22px.
    _dot(dl, cx, cy, r, col, 0.0, -0.5, 0.15)
    _line(dl, cx, cy, r, col, t, 0.0, -0.12, 0.0, 0.54)


def _warning(dl, cx, cy, r, col, t) -> None:
    _poly(dl, cx, cy, r, col, t, [(0.0, -0.9), (0.95, 0.75), (-0.95, 0.75)], closed=True)
    _line(dl, cx, cy, r, col, t, 0.0, -0.3, 0.0, 0.28)
    _dot(dl, cx, cy, r, col, 0.0, 0.53, 0.11)


def _check(dl, cx, cy, r, col, t) -> None:
    _poly(dl, cx, cy, r, col, t, [(-0.85, 0.05), (-0.3, 0.65), (0.85, -0.65)])


def _close(dl, cx, cy, r, col, t) -> None:
    _line(dl, cx, cy, r, col, t, -0.75, -0.75, 0.75, 0.75)
    _line(dl, cx, cy, r, col, t, 0.75, -0.75, -0.75, 0.75)


# --- shapes: 3D --------------------------------------------------------------
#
# Every ``chart_*3d`` icon is a mark drawn inside a shared isometric tripod, the way every
# ``chart_*`` icon is a mark inside the shared 2D L-frame (:func:`_axes`). The tripod does
# the whole job of saying "this is the 3D version of that": at 22px a mark alone cannot
# carry the difference between a bar chart and a bar chart in space, but three axes
# meeting at a corner can, and the eye reads the pair as a family rather than as ten
# unrelated glyphs.

#: Where the tripod's three axes meet, in normalised icon space (y is down).
_TRIPOD = (-0.88, 0.72)


def _axes3d(dl, cx, cy, r, col, t) -> None:
    """Shared 3D corner: the 2D L-frame plus a short receding depth axis.

    Deliberately the same L as :func:`_axes` with one stroke added, rather than a
    free-standing tripod: the 2D and 3D icons for the same plot type then differ by
    exactly that stroke, which is the difference the icons are there to communicate. The
    depth axis is kept short so the mark has the whole upper-right quadrant to itself —
    a full-length third axis crosses every mark and turns the glyph into a scribble.
    """
    ox, oy = _TRIPOD
    _poly(dl, cx, cy, r, col, t, [(ox, -0.9), (ox, oy), (0.9, oy)])
    _line(dl, cx, cy, r, col, t * 0.9, ox, oy, -0.3, 0.26)


def _cube(dl, cx, cy, r, col, t) -> None:
    """A wireframe box: the 3D scene itself, and the axis-box toggle."""
    # Front face, then the back face offset up-right, then the four connectors.
    f = [(-0.78, -0.08), (0.26, -0.08), (0.26, 0.88), (-0.78, 0.88)]
    b = [(x + 0.52, y - 0.52) for x, y in f]
    _poly(dl, cx, cy, r, col, t, f, closed=True)
    _poly(dl, cx, cy, r, col, t, b, closed=True)
    for (fx, fy), (bx, by) in zip(f, b):
        _line(dl, cx, cy, r, col, t * 0.85, fx, fy, bx, by)


def _chart_scatter3d(dl, cx, cy, r, col, t) -> None:
    _axes3d(dl, cx, cy, r, col, t)
    for x, y, rad in (
        (-0.5, 0.18, 0.15),
        (-0.06, -0.3, 0.16),
        (0.34, 0.2, 0.15),
        (0.66, -0.5, 0.13),
    ):
        _dot(dl, cx, cy, r, col, x, y, rad)


def _chart_line3d(dl, cx, cy, r, col, t) -> None:
    _axes3d(dl, cx, cy, r, col, t)
    _poly(dl, cx, cy, r, col, t, [(-0.58, 0.3), (-0.16, -0.4), (0.24, 0.2), (0.72, -0.6)])


def _chart_stem3d(dl, cx, cy, r, col, t) -> None:
    _axes3d(dl, cx, cy, r, col, t)
    # Stems standing on the floor edge, each capped with its sample point.
    for x, top in ((-0.5, 0.06), (-0.1, -0.42), (0.32, -0.1), (0.7, -0.56)):
        _line(dl, cx, cy, r, col, t * 0.85, x, 0.72, x, top)
        _dot(dl, cx, cy, r, col, x, top, 0.14)


def _chart_ribbon3d(dl, cx, cy, r, col, t) -> None:
    _axes3d(dl, cx, cy, r, col, t)
    # A single filled band under a profile: one shape, so it stays legible at 22px where
    # the earlier four-triangle version fused into a blob.
    top = [(-0.58, 0.14), (-0.18, -0.42), (0.24, -0.02), (0.72, -0.52)]
    band = top + [(0.72, 0.62), (-0.58, 0.62)]
    _poly(dl, cx, cy, r, col, t * 0.85, band, closed=True)
    _tri_filled(dl, cx, cy, r, col, top[0], top[1], (-0.18, 0.62))
    _tri_filled(dl, cx, cy, r, col, top[0], (-0.18, 0.62), (-0.58, 0.62))
    _tri_filled(dl, cx, cy, r, col, top[2], top[3], (0.72, 0.62))
    _tri_filled(dl, cx, cy, r, col, top[2], (0.72, 0.62), (0.24, 0.62))


def _chart_bar3d(dl, cx, cy, r, col, t) -> None:
    _axes3d(dl, cx, cy, r, col, t)
    # Three boxes, each a filled front face plus an outlined lid so it reads as solid.
    for x0, top in ((-0.6, 0.02), (-0.14, -0.46), (0.32, -0.18)):
        x1 = x0 + 0.34
        d = 0.18
        _rect_filled(dl, cx, cy, r, col, x0, top, x1, 0.7)
        _poly(
            dl,
            cx,
            cy,
            r,
            col,
            t * 0.75,
            [(x0, top), (x0 + d, top - d), (x1 + d, top - d), (x1, top)],
            closed=True,
        )


def _chart_surface3d(dl, cx, cy, r, col, t) -> None:
    _axes3d(dl, cx, cy, r, col, t)
    # Three depth-stacked profiles plus two cross ribs: the minimum that says "continuous
    # sheet" rather than "three separate curves".
    rows = [
        [(-0.56, -0.16), (-0.16, -0.5), (0.24, -0.24), (0.7, -0.56)],
        [(-0.48, 0.14), (-0.08, -0.2), (0.32, 0.06), (0.78, -0.26)],
        [(-0.4, 0.44), (0.0, 0.1), (0.4, 0.36), (0.86, 0.04)],
    ]
    for row in rows:
        _poly(dl, cx, cy, r, col, t * 0.85, row)
    for i in (1, 2):
        _poly(dl, cx, cy, r, col, t * 0.7, [rows[0][i], rows[1][i], rows[2][i]])


def _chart_wireframe3d(dl, cx, cy, r, col, t) -> None:
    _axes3d(dl, cx, cy, r, col, t)

    # A skewed 4x4 lattice: parallel rulings in two directions, which is what a wireframe is.
    def node(u, v):
        return (-0.56 + u * 1.34 + v * 0.26, 0.42 - u * 0.2 - v * 0.72)

    for i in range(4):
        f = i / 3.0
        _poly(dl, cx, cy, r, col, t * 0.7, [node(f, v / 3.0) for v in range(4)])
        _poly(dl, cx, cy, r, col, t * 0.7, [node(u / 3.0, f) for u in range(4)])


def _chart_volume3d(dl, cx, cy, r, col, t) -> None:
    _axes3d(dl, cx, cy, r, col, t)
    # A cloud: many small dots at varied radii, dense in the middle.
    pts = (
        (-0.3, 0.1, 0.11),
        (-0.02, -0.24, 0.13),
        (0.28, 0.02, 0.12),
        (0.54, -0.34, 0.1),
        (0.1, 0.4, 0.1),
        (0.62, 0.3, 0.09),
        (-0.16, -0.56, 0.08),
        (0.4, 0.5, 0.08),
        (0.84, -0.08, 0.07),
        (-0.48, 0.44, 0.07),
        (0.2, -0.62, 0.07),
    )
    for x, y, rad in pts:
        _dot(dl, cx, cy, r, col, x, y, rad)


def _chart_quiver3d(dl, cx, cy, r, col, t) -> None:
    _axes3d(dl, cx, cy, r, col, t)
    for x0, y0, x1, y1 in (
        (-0.58, 0.4, -0.1, -0.02),
        (-0.1, 0.5, 0.42, 0.06),
        (0.26, -0.02, 0.8, -0.44),
    ):
        _arrow2d(dl, cx, cy, r, col, t, x0, y0, x1, y1, head=0.3)


def _arrow2d(dl, cx, cy, r, col, t, x0, y0, x1, y1, head: float = 0.28) -> None:
    """A line with a two-barb arrowhead at ``(x1, y1)``, in normalised icon space.

    Shared by every arrow glyph so the heads are identical across the set; the barbs are
    built from the shaft's own direction, which is what makes it work at any angle.
    """
    _line(dl, cx, cy, r, col, t, x0, y0, x1, y1)
    dx, dy = x1 - x0, y1 - y0
    norm = max(math.hypot(dx, dy), 1e-6)
    ux, uy = dx / norm, dy / norm
    for sign in (1.0, -1.0):
        bx = x1 - head * (ux * 0.87 - sign * uy * 0.5)
        by = y1 - head * (uy * 0.87 + sign * ux * 0.5)
        _line(dl, cx, cy, r, col, t * 0.9, x1, y1, bx, by)


def _orbit(dl, cx, cy, r, col, t) -> None:
    """Rotate-the-view: a filled body with a tilted ring around it.

    The ring is *tilted*, and the body is filled rather than outlined. Both details are
    load-bearing: an unfilled circle inside a horizontal ellipse is the ``eye`` icon, and
    the two sat next to each other in the rail.
    """
    _dot(dl, cx, cy, r, col, 0.0, 0.0, 0.32)
    tilt = math.radians(-24.0)
    ct, st = math.cos(tilt), math.sin(tilt)
    pts = []
    for i in range(29):
        a = i / 28.0 * math.tau
        ex, ey = 0.94 * math.cos(a), 0.36 * math.sin(a)
        pts.append((ex * ct - ey * st, ex * st + ey * ct))
    _poly(dl, cx, cy, r, col, t * 0.9, pts, closed=True)


def _pan3d(dl, cx, cy, r, col, t) -> None:
    """Move-the-target: a four-way arrow cross."""
    _line(dl, cx, cy, r, col, t, -0.88, 0.0, 0.88, 0.0)
    _line(dl, cx, cy, r, col, t, 0.0, -0.88, 0.0, 0.88)
    for tipx, tipy in ((-0.88, 0.0), (0.88, 0.0), (0.0, -0.88), (0.0, 0.88)):
        _arrow2d(dl, cx, cy, r, col, t, tipx * 0.55, tipy * 0.55, tipx, tipy, head=0.3)


def _perspective(dl, cx, cy, r, col, t) -> None:
    """A frustum: near face small, far face large, with the four rays between them."""
    near = [(-0.3, -0.22), (0.3, -0.22), (0.3, 0.28), (-0.3, 0.28)]
    far = [(-0.9, -0.72), (0.9, -0.72), (0.9, 0.86), (-0.9, 0.86)]
    _poly(dl, cx, cy, r, col, t, near, closed=True)
    _poly(dl, cx, cy, r, col, t * 0.8, far, closed=True)
    for (nx, ny), (fx, fy) in zip(near, far):
        _line(dl, cx, cy, r, col, t * 0.7, nx, ny, fx, fy)


def _orthographic(dl, cx, cy, r, col, t) -> None:
    """A prism: two identical faces joined by parallel edges — no convergence."""
    front = [(-0.75, -0.15), (0.25, -0.15), (0.25, 0.85), (-0.75, 0.85)]
    back = [(x + 0.5, y - 0.5) for x, y in front]
    _poly(dl, cx, cy, r, col, t, front, closed=True)
    _poly(dl, cx, cy, r, col, t * 0.8, back, closed=True)
    for (fx, fy), (bx, by) in zip(front, back):
        _line(dl, cx, cy, r, col, t * 0.7, fx, fy, bx, by)


def _view_box(dl, cx, cy, r, col, t, face: str) -> None:
    """A cube with one face shaded: the standard-view presets.

    ``face`` picks which of top/front/right is filled, so the three icons differ by the
    highlighted plane rather than by a letter — which no font-less icon could draw.
    """
    f = [(-0.7, 0.0), (0.24, 0.0), (0.24, 0.86), (-0.7, 0.86)]
    b = [(x + 0.48, y - 0.48) for x, y in f]
    if face == "front":
        _rect_filled(dl, cx, cy, r, col, f[0][0], f[0][1], f[2][0], f[2][1])
    elif face == "top":
        # The lid is the quad f3-b3-b2-f2 (the two top edges of the front and back faces).
        _tri_filled(dl, cx, cy, r, col, f[3], b[3], b[2])
        _tri_filled(dl, cx, cy, r, col, f[3], b[2], f[2])
    elif face == "right":
        _tri_filled(dl, cx, cy, r, col, f[1], b[1], b[2])
        _tri_filled(dl, cx, cy, r, col, f[1], b[2], f[2])
    _poly(dl, cx, cy, r, col, t, f, closed=True)
    _poly(dl, cx, cy, r, col, t * 0.8, b, closed=True)
    for (fx, fy), (bx, by) in zip(f, b):
        _line(dl, cx, cy, r, col, t * 0.7, fx, fy, bx, by)


def _view_iso(dl, cx, cy, r, col, t) -> None:
    """The isometric preset: a hexagon split by a three-spoke Y — the canonical iso cube.

    Distinct from :func:`_cube` on purpose. ``cube`` is a *toggle* ("show the axis box")
    and ``view_iso`` is an *action* ("look from the corner"), and two identical glyphs in
    the same panel doing different things is worse than either being slightly less literal.
    """
    pts = [
        (math.cos(a), math.sin(a)) for a in (math.tau * i / 6.0 + math.pi / 6.0 for i in range(6))
    ]
    pts = [(0.92 * x, 0.92 * y) for x, y in pts]
    _poly(dl, cx, cy, r, col, t, pts, closed=True)
    # The three interior edges meeting at the centre — the near corner of the cube.
    for i in (0, 2, 4):
        _line(dl, cx, cy, r, col, t * 0.85, 0.0, 0.0, pts[i][0], pts[i][1])


def _box_aspect(dl, cx, cy, r, col, t) -> None:
    """Axis normalisation: a tall thin box beside a cube, with an arrow between them."""
    _rect(dl, cx, cy, r, col, t * 0.9, -0.92, -0.8, -0.5, 0.8)
    _rect(dl, cx, cy, r, col, t, 0.24, -0.34, 0.92, 0.34)
    _line(dl, cx, cy, r, col, t * 0.85, -0.3, 0.0, 0.05, 0.0)
    _line(dl, cx, cy, r, col, t * 0.85, 0.05, 0.0, -0.1, -0.16)
    _line(dl, cx, cy, r, col, t * 0.85, 0.05, 0.0, -0.1, 0.16)


def _light(dl, cx, cy, r, col, t) -> None:
    """Shading / ambient occlusion: a sun with rays."""
    _circle(dl, cx, cy, r, col, t, 0.0, 0.0, 0.36, 16)
    for i in range(8):
        a = i / 8.0 * 6.283185
        ca, sa = math.cos(a), math.sin(a)
        _line(dl, cx, cy, r, col, t * 0.85, ca * 0.58, sa * 0.58, ca * 0.92, sa * 0.92)


def _spin(dl, cx, cy, r, col, t) -> None:
    """Auto-rotate: a circular arrow."""
    _arc(dl, cx, cy, r, col, t, 0.0, 0.0, 0.72, 0.5, 5.6, 26)
    _line(dl, cx, cy, r, col, t, 0.63, -0.35, 0.78, -0.02)
    _line(dl, cx, cy, r, col, t, 0.63, -0.35, 0.3, -0.4)


# --- shapes: animation / timeline --------------------------------------------


def _stop(dl, cx, cy, r, col, t) -> None:
    _rect_filled(dl, cx, cy, r, col, -0.55, -0.55, 0.55, 0.55, rounding=0.08)


def _skip_start(dl, cx, cy, r, col, t) -> None:
    _rect_filled(dl, cx, cy, r, col, -0.75, -0.6, -0.5, 0.6)
    _tri_filled(dl, cx, cy, r, col, (0.7, -0.65), (0.7, 0.65), (-0.35, 0.0))


def _skip_end(dl, cx, cy, r, col, t) -> None:
    _tri_filled(dl, cx, cy, r, col, (-0.7, -0.65), (-0.7, 0.65), (0.35, 0.0))
    _rect_filled(dl, cx, cy, r, col, 0.5, -0.6, 0.75, 0.6)


def _step_back(dl, cx, cy, r, col, t) -> None:
    _tri_filled(dl, cx, cy, r, col, (0.3, -0.6), (0.3, 0.6), (-0.6, 0.0))
    _rect_filled(dl, cx, cy, r, col, 0.5, -0.6, 0.72, 0.6)


def _step_forward(dl, cx, cy, r, col, t) -> None:
    _rect_filled(dl, cx, cy, r, col, -0.72, -0.6, -0.5, 0.6)
    _tri_filled(dl, cx, cy, r, col, (-0.3, -0.6), (-0.3, 0.6), (0.6, 0.0))


def _loop(dl, cx, cy, r, col, t) -> None:
    """Playback looping: a rounded rectangle circuit with an arrowhead on it."""
    _rect(dl, cx, cy, r, col, t, -0.8, -0.5, 0.8, 0.5, rounding=0.45)
    _line(dl, cx, cy, r, col, t, 0.1, -0.5, -0.12, -0.72)
    _line(dl, cx, cy, r, col, t, 0.1, -0.5, -0.12, -0.28)


def _keyframe(dl, cx, cy, r, col, t) -> None:
    """One keyframe: the diamond every animation tool uses for a keyed value."""
    _poly(dl, cx, cy, r, col, t, [(0.0, -0.7), (0.62, 0.0), (0.0, 0.7), (-0.62, 0.0)], closed=True)
    _tri_filled(dl, cx, cy, r, col, (0.0, -0.36), (0.32, 0.0), (0.0, 0.36))
    _tri_filled(dl, cx, cy, r, col, (0.0, -0.36), (0.0, 0.36), (-0.32, 0.0))


def _timeline(dl, cx, cy, r, col, t) -> None:
    """The timeline itself: a keyed track crossed by a playhead.

    A *track with keys on it* rather than a ruler with tick marks — the tick version read
    as a garden rake at 22px, and the diamonds tie the glyph to :func:`_keyframe`.
    """
    _rect(dl, cx, cy, r, col, t * 0.9, -0.95, -0.12, 0.95, 0.5, rounding=0.16)
    for x in (-0.56, 0.06, 0.62):
        _tri_filled(dl, cx, cy, r, col, (x, 0.02), (x + 0.2, 0.19), (x, 0.36))
        _tri_filled(dl, cx, cy, r, col, (x, 0.02), (x, 0.36), (x - 0.2, 0.19))
    # Playhead: a stem crossing the whole track, headed at the top.
    _line(dl, cx, cy, r, col, t, -0.22, -0.5, -0.22, 0.82)
    _tri_filled(dl, cx, cy, r, col, (-0.5, -0.88), (0.06, -0.88), (-0.22, -0.44))


def _missing(dl, cx, cy, r, col, t) -> None:
    """Placeholder for an unknown shape name: visibly wrong, but never fatal mid-frame."""
    _rect(dl, cx, cy, r, col, t, -0.8, -0.8, 0.8, 0.8, rounding=0.1)
    _line(dl, cx, cy, r, col, t, -0.8, -0.8, 0.8, 0.8)


_SHAPES: Dict[str, Callable[..., None]] = {
    "plus": _plus,
    "minus": _minus,
    "trash": _trash,
    "copy": _copy,
    "paste": _paste,
    "cut": _cut,
    "undo": _undo,
    "redo": _redo,
    "play": _play,
    "pause": _pause,
    "refresh": _refresh,
    "gear": _gear,
    "chart_line": _chart_line,
    "chart_scatter": _chart_scatter,
    "chart_bar": _chart_bar,
    "table": _table,
    "function": _function,
    "integral": _integral,
    "derivative": _derivative,
    "wave": _wave,
    "noise": _noise,
    "smooth": _smooth,
    "filter": _filter,
    "eye": _eye,
    "eye_off": _eye_off,
    "lock": _lock,
    "save": _save,
    "folder": _folder,
    "download": _download,
    "upload": _upload,
    "search": _search,
    "zoom_in": _zoom_in,
    "zoom_out": _zoom_out,
    "home": _home,
    "target": _target,
    "layers": _layers,
    "palette": _palette,
    "sliders": _sliders,
    "info": _info,
    "warning": _warning,
    "check": _check,
    "close": _close,
    "chevron_right": lambda dl, cx, cy, r, col, t: _chevron(dl, cx, cy, r, col, t, "right"),
    "chevron_down": lambda dl, cx, cy, r, col, t: _chevron(dl, cx, cy, r, col, t, "down"),
    "chevron_left": lambda dl, cx, cy, r, col, t: _chevron(dl, cx, cy, r, col, t, "left"),
    "chevron_up": lambda dl, cx, cy, r, col, t: _chevron(dl, cx, cy, r, col, t, "up"),
    "grid": _grid,
    "link": _link,
    # -- 3D plot types (each is the shared tripod plus its mark; see `_axes3d`) --
    "axes3d": _axes3d,
    "cube": _cube,
    "chart_scatter3d": _chart_scatter3d,
    "chart_line3d": _chart_line3d,
    "chart_stem3d": _chart_stem3d,
    "chart_ribbon3d": _chart_ribbon3d,
    "chart_bar3d": _chart_bar3d,
    "chart_surface3d": _chart_surface3d,
    "chart_wireframe3d": _chart_wireframe3d,
    "chart_volume3d": _chart_volume3d,
    "chart_quiver3d": _chart_quiver3d,
    # -- 3D camera and view --
    "orbit": _orbit,
    "pan3d": _pan3d,
    "perspective": _perspective,
    "orthographic": _orthographic,
    "view_top": lambda dl, cx, cy, r, col, t: _view_box(dl, cx, cy, r, col, t, "top"),
    "view_front": lambda dl, cx, cy, r, col, t: _view_box(dl, cx, cy, r, col, t, "front"),
    "view_right": lambda dl, cx, cy, r, col, t: _view_box(dl, cx, cy, r, col, t, "right"),
    "view_iso": _view_iso,
    "box_aspect": _box_aspect,
    "light": _light,
    "spin": _spin,
    # -- animation / timeline transport --
    "stop": _stop,
    "skip_start": _skip_start,
    "skip_end": _skip_end,
    "step_back": _step_back,
    "step_forward": _step_forward,
    "loop": _loop,
    "keyframe": _keyframe,
    "timeline": _timeline,
}

#: Every drawable shape name, in a stable order (the SPEC list, then the two extra
#: chevrons). Panels may iterate this for a gallery/debug view.
ICON_SHAPES: List[str] = list(_SHAPES.keys())


def draw_icon(draw_list, shape: str, cx: float, cy: float, r: float, col: int) -> None:
    """Draw ``shape`` centred at ``(cx, cy)`` with radius ``r`` in packed colour ``col``.

    ``r`` is the half-extent of the glyph, not the button size: the shape occupies
    roughly ``[cx - r, cx + r] x [cy - r, cy + r]``.

    An unknown ``shape`` logs once and draws a crossed-box placeholder. It never raises:
    this runs inside a draw callback, where an exception would unbalance the caller's
    imgui begin/end stack and take down the frame.
    """
    if not IMGUI_AVAILABLE:
        return
    fn = _SHAPES.get(shape)
    if fn is None:
        if shape not in _MISSING_WARNED:
            _MISSING_WARNED.add(shape)
            logger.warning("Unknown icon shape %r; drawing placeholder.", shape)
        fn = _missing
    fn(draw_list, cx, cy, r, col, _stroke(r))


def _icon_color(name: str, alpha: Optional[float] = None) -> int:
    """Pack a theme token for draw_list use, tolerating a theme that never got applied."""
    c = COLORS.get(name, (0.85, 0.86, 0.88, 1.0))
    a = c[3] if alpha is None else alpha
    return imgui.get_color_u32((c[0], c[1], c[2], a))


def icon_button(
    id_str: str,
    shape: str,
    *,
    size: float = 22.0,
    tooltip: Optional[str] = None,
    active: bool = False,
    enabled: bool = True,
) -> bool:
    """A clickable vector icon. Returns True on click (never when ``enabled=False``).

    ``active`` renders the toggled-on state (accent fill + accent glyph) and is for
    latching buttons -- a visible panel, an armed tool. It is presentation only; the
    caller owns the state.

    ``id_str`` is a full imgui id: pass ``"##undo"`` or ``"undo##toolbar"`` to keep it
    out of the visible label space. Ids must be unique within the window, as usual.
    """
    if not IMGUI_AVAILABLE:
        return False

    dl = imgui.get_window_draw_list()
    # MANDATORY ORDERING (CONTRACT section 3): the cursor is the *next* item's top-left.
    # Read it before invisible_button advances it, or every icon lands one slot late.
    ox, oy = imgui.get_cursor_screen_pos()

    clicked = imgui.invisible_button(id_str, (size, size))
    hovered = imgui.is_item_hovered()
    held = imgui.is_item_active()

    if not enabled:
        clicked = False
        held = False

    rounding = imgui.get_style().frame_rounding

    if not enabled:
        col = _icon_color("muted", 0.45)
    elif held:
        _rect_bg(dl, ox, oy, size, _icon_color("accent_active", 0.85), rounding)
        col = _icon_color("on_accent")
    elif active:
        _rect_bg(dl, ox, oy, size, _icon_color("accent", 0.22 if not hovered else 0.34), rounding)
        col = _icon_color("accent_hover")
    elif hovered:
        _rect_bg(dl, ox, oy, size, _icon_color("text", 0.10), rounding)
        col = _icon_color("text")
    else:
        col = _icon_color("text", 0.82)

    draw_icon(dl, shape, ox + size * 0.5, oy + size * 0.5, size * 0.32, col)

    if tooltip and hovered:
        imgui.set_tooltip(tooltip)
    return bool(clicked)


def _rect_bg(dl, ox: float, oy: float, size: float, col: int, rounding: float) -> None:
    """Fill the button's hit rect. Kept separate so the state ladder above stays readable."""
    dl.add_rect_filled((ox, oy), (ox + size, oy + size), col, rounding, 0)


def icon_inline(shape: str, *, size: float = 16.0, color: Optional[str] = None) -> None:
    """Draw a non-interactive icon and reserve its layout space.

    For decorating panel titles and rows (``Panel.icon``). ``color`` is a
    :data:`glplot.gui.theme.COLORS` token name; defaults to dimmed body text.
    """
    if not IMGUI_AVAILABLE:
        return
    dl = imgui.get_window_draw_list()
    ox, oy = imgui.get_cursor_screen_pos()
    col = _icon_color(color) if color else _icon_color("text", 0.82)
    draw_icon(dl, shape, ox + size * 0.5, oy + size * 0.5, size * 0.34, col)
    imgui.dummy((size, size))
