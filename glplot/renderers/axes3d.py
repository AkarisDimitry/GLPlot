"""Geometry for a decorated 3D axis box: walls, grid, tick marks and label anchors.

Pure numpy. No GL, no imgui, no engine import — everything here is a function from
``(bounds, camera)`` to arrays, so the whole module is testable headless and the engine
can call it from anywhere.

What this replaces
------------------
``engine.ensure_3d_axes`` used to emit two things: a 24-vertex wireframe box and a shaded
floor quad. That is a *bounding box*, not an axis. It says where the data ends and nothing
about what the numbers are — a 3D plot drawn with it cannot be read, only admired.

A readable 3D axis needs four more pieces, and all four depend on where the camera is:

* **Grid walls.** Ruled lines on three of the box's six faces. Which three is a camera
  question: the grid belongs on the *far* walls, so the data is seen against it rather
  than through it. :func:`back_wall_axes` picks them from the view direction.
* **Tick marks.** Short segments stepping out of the box at each tick value, on the edge
  the viewer can actually see. :func:`tick_edges` chooses those edges by projecting the
  candidates and taking the one that lands lowest (x, y) or leftmost (z) on screen, which
  is the rule that keeps ticks out from under the data at any orientation.
* **Numeric labels.** Text cannot be a GL primitive here — GLPlot draws all of its text
  through imgui — so this module returns *anchors*: world positions plus strings. The
  caller projects them with the same MVP the geometry was drawn with and hands them to a
  draw list. :func:`label_anchors` produces both the numbers and the axis titles.
* **Nice numbers.** The tick values come from :func:`glplot.managers.axis._nice_step` and
  are formatted by :func:`glplot.managers.axis._auto_format` — the 2D axis renderer's own
  helpers, imported rather than reimplemented so a value never prints one way on a 2D plot
  and another way on a 3D one.

Everything is generated in **world units on the raw (unscaled) bounds**. The box-aspect
model matrix is applied by the shader along with the rest of the scene, so the decoration
stretches with the data instead of drifting off it.

The one exception to "pure numpy" is :func:`draw_labels` at the bottom, which paints the
anchors into an imgui draw list. It imports imgui lazily inside the function and returns
quietly when there is none, exactly as :meth:`glplot.renderers.axis.AxisRenderer._draw_labels`
does — the text has to be drawn somewhere, and putting it beside the geometry that placed
it beats a second module that knows the same conventions.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ..core.camera3d import Camera3D, bounds_centre_radius
from ..managers.axis import _auto_format, _nice_step

Bounds3D = Tuple[float, float, float, float, float, float]

# Every offset below is a fraction of the extent of **the axis it is measured along**, not
# of the box's longest edge.
#
# The longest-edge version was wrong the moment the data was anisotropic, which is most
# real data. With x spanning 8 and y spanning 24, an axis title at 0.135 of the longest
# edge landed 3.2 units outside an axis whose whole half-width is 4 — the labels drifted
# off into empty space, nowhere near the box they annotate. Scaling each offset by its own
# axis makes the layout scale-free: the picture is identical whether y spans 24 or 24
# million.

#: Length of a tick mark, as a fraction of the extent of the axis it points along.
TICK_LEN_FRAC = 0.030

#: How far beyond the box a numeric tick label sits, along the axis it is offset on.
TICK_LABEL_FRAC = 0.075

#: How far beyond the box an axis *title* sits. Further out than the numbers, which it has
#: to clear.
AXIS_LABEL_FRAC = 0.190

#: The diagonal offsets a z label takes (it steps out along both x and y) are scaled by
#: this so the total distance matches a single-axis offset rather than exceeding it by √2.
_DIAGONAL = 0.70710678

#: Default tick count per axis when the caller does not ask for one.
DEFAULT_TICK_COUNT = 5

#: Hard cap on ticks per axis. A grid wall costs ``2 x n`` vertices per direction, and past
#: this the wall reads as a solid sheet anyway.
MAX_TICKS = 40


# ----------------------------------------------------------------------------------
# Ticks
# ----------------------------------------------------------------------------------


def axis_ticks(
    lo: float, hi: float, target: int = DEFAULT_TICK_COUNT
) -> Tuple[np.ndarray, List[str]]:
    """Nice tick values and their labels across ``[lo, hi]``.

    The same 1-2-5 ladder and the same formatter the 2D axes use, so "0.25" is spelled
    identically on both. A degenerate or non-finite range yields no ticks rather than
    raising — a 3D scene with one point in it must still draw.

    ``target`` is a request. The ladder rounds it, so a target of 3 over a span of 9 comes
    back as 5 ticks on a step of 2 — asking for a count and getting a *round* count is the
    trade the ladder exists to make.
    """
    lo, hi = float(lo), float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(0, dtype=np.float64), []
    target = max(2, min(int(target), MAX_TICKS))
    step = _nice_step((hi - lo) / target)
    if step <= 0 or not np.isfinite(step):
        return np.zeros(0, dtype=np.float64), []

    def on_step(step: float) -> np.ndarray:
        start = np.ceil(lo / step) * step
        end = np.floor(hi / step) * step
        if start > end:
            return np.zeros(0, dtype=np.float64)
        return np.arange(start, end + step / 2, step)

    values = on_step(step)
    # Over the cap, double the step until it fits. The ladder can overshoot the target --
    # a target of 30 on this span may land on a step giving 47 -- and the cap has to be
    # honoured by making the ticks *coarser*, not by keeping the first MAX_TICKS of them:
    # slicing labelled the bottom of the axis and left the top bare, which reads as a plot
    # whose axis stops halfway. Doubling stays on the ladder (1-2-5 -> 2-4-10 -> ...), and
    # the span is finite so it terminates.
    while values.size > MAX_TICKS and np.isfinite(step) and step > 0:
        step *= 2.0
        values = on_step(step)
    if values.size == 0:
        return np.zeros(0, dtype=np.float64), []
    # -0.0 prints as "-0", which looks like a bug in an axis that straddles the origin.
    values = values + 0.0
    return values, _auto_format(values, step)


def ticks_for_bounds(
    bounds: Bounds3D,
    target: Union[int, Sequence[int]] = DEFAULT_TICK_COUNT,
    /,
) -> Tuple[Tuple[np.ndarray, List[str]], ...]:
    """``((values, labels), ...)`` for the x, y and z axes of ``bounds``.

    ``target`` is either one count for all three axes or three counts, one per axis.
    Per-axis is what an adaptive density needs: the three axes of a box are three different
    lengths on screen, and giving a 40-pixel edge the same number of labels as a 900-pixel
    one is the whole problem. See :func:`adaptive_tick_counts`.
    """
    if isinstance(target, (int, np.integer)):
        tx = ty = tz = int(target)
    else:
        tx, ty, tz = (int(t) for t in target)
    return (
        axis_ticks(bounds[0], bounds[1], tx),
        axis_ticks(bounds[2], bounds[3], ty),
        axis_ticks(bounds[4], bounds[5], tz),
    )


# ----------------------------------------------------------------------------------
# Box geometry
# ----------------------------------------------------------------------------------


def _corners(bounds: Bounds3D) -> np.ndarray:
    """The box's eight corners as ``(8, 3)``, indexed by ``(x_hi, y_hi, z_hi)`` bits."""
    xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in bounds)
    return np.array(
        [
            [xmin, ymin, zmin],
            [xmin, ymin, zmax],
            [xmin, ymax, zmin],
            [xmin, ymax, zmax],
            [xmax, ymin, zmin],
            [xmax, ymin, zmax],
            [xmax, ymax, zmin],
            [xmax, ymax, zmax],
        ],
        dtype=np.float32,
    )


def box_edges(bounds: Bounds3D) -> np.ndarray:
    """The twelve edges of the box as ``(24, 3)`` ``GL_LINES`` vertices.

    Byte-for-byte the array ``ensure_3d_axes`` has always built (same edges, same order),
    so replacing the old inline literal with this call changes no pixels.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in bounds)
    return np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmin, ymin, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmax, ymin, zmin],
            [xmax, ymin, zmax],
            [xmin, ymax, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymax, zmax],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmin, ymin, zmax],
            [xmin, ymax, zmax],
            [xmax, ymax, zmin],
            [xmax, ymax, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
            [xmax, ymax, zmax],
        ],
        dtype=np.float32,
    )


def floor_quad(bounds: Bounds3D) -> np.ndarray:
    """The ``z = zmin`` plane as ``(6, 3)`` ``GL_TRIANGLES`` vertices."""
    xmin, xmax, ymin, ymax, zmin, _ = (float(v) for v in bounds)
    return np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
        ],
        dtype=np.float32,
    )


def wall_panes(bounds: Bounds3D, camera: Camera3D) -> np.ndarray:
    """The three *back* walls as shaded quads — ``(18, 3)`` ``GL_TRIANGLES`` vertices.

    This is the single largest thing separating a finished-looking 3D plot from a bare
    wireframe box. matplotlib calls them panes, and every 3D tool draws something like
    them, because a grid line floating in empty space gives the eye no surface to sit the
    data against — depth reads from the shaded planes, not from the lines on them.

    Which three walls is a camera question, answered by :func:`back_wall_axes`: always the
    ones facing away, so the data is seen *against* the panes rather than *through* them.
    Replacing the old fixed ``z = zmin`` floor with all three also fixes the case where the
    camera drops below the data and the only pane was above it, out of sight.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in bounds)
    x_at_max, y_at_max, z_at_max = back_wall_axes(camera)
    wall_x = xmax if x_at_max else xmin
    wall_y = ymax if y_at_max else ymin
    wall_z = zmax if z_at_max else zmin

    def quad(a, b, c, d):
        return [a, b, c, a, c, d]

    verts: List[Sequence[float]] = []
    # Constant-z pane (the floor, or the ceiling when looking from below).
    verts += quad(
        (xmin, ymin, wall_z), (xmax, ymin, wall_z), (xmax, ymax, wall_z), (xmin, ymax, wall_z)
    )
    # Constant-x pane.
    verts += quad(
        (wall_x, ymin, zmin), (wall_x, ymax, zmin), (wall_x, ymax, zmax), (wall_x, ymin, zmax)
    )
    # Constant-y pane.
    verts += quad(
        (xmin, wall_y, zmin), (xmax, wall_y, zmin), (xmax, wall_y, zmax), (xmin, wall_y, zmax)
    )
    return np.ascontiguousarray(np.array(verts, dtype=np.float32))


def box_diagonal(bounds: Bounds3D) -> float:
    """The longest edge of the box — the length every offset here is a fraction of.

    The *longest edge* rather than the diagonal, so a flat scene (z span ≈ 0) still gets
    tick marks scaled to the axes that do have extent.
    """
    return max(
        float(bounds[1]) - float(bounds[0]),
        float(bounds[3]) - float(bounds[2]),
        float(bounds[5]) - float(bounds[4]),
        1e-6,
    )


# ----------------------------------------------------------------------------------
# Camera-dependent placement
# ----------------------------------------------------------------------------------


def back_wall_axes(camera: Camera3D) -> Tuple[bool, bool, bool]:
    """Which side each of the three wall pairs should be drawn on.

    Returns ``(x_at_max, y_at_max, z_at_max)``: True means the wall sits at that axis'
    maximum. The chosen wall is always the one *facing away* from the eye, so the grid
    stays behind the data. With the eye at ``+x`` the x wall belongs at ``xmin``, hence
    the negation.
    """
    direction = camera.direction()  # target -> eye
    return (
        bool(direction[0] < 0.0),
        bool(direction[1] < 0.0),
        bool(direction[2] < 0.0),
    )


def grid_lines(
    bounds: Bounds3D,
    ticks: Sequence[Tuple[np.ndarray, List[str]]],
    camera: Camera3D,
) -> np.ndarray:
    """Ruled grid lines on the three back walls, as ``(N, 2*3)``-shaped ``GL_LINES``.

    Each wall carries the ticks of the two axes that span it, so a value can be traced from
    its label, across a wall, to the data — which is the whole reason a grid exists.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in bounds)
    (xt, _), (yt, _), (zt, _) = ticks
    x_at_max, y_at_max, z_at_max = back_wall_axes(camera)
    wall_x = xmax if x_at_max else xmin
    wall_y = ymax if y_at_max else ymin
    wall_z = zmax if z_at_max else zmin

    segments: List[np.ndarray] = []

    def add(p0: Sequence[float], p1: Sequence[float]) -> None:
        segments.append(np.array([p0, p1], dtype=np.float32))

    # Floor/ceiling wall (constant z): ruled by the x and y ticks.
    for x in xt:
        add((x, ymin, wall_z), (x, ymax, wall_z))
    for y in yt:
        add((xmin, y, wall_z), (xmax, y, wall_z))

    # Constant-x wall: ruled by the y and z ticks.
    for y in yt:
        add((wall_x, y, zmin), (wall_x, y, zmax))
    for z in zt:
        add((wall_x, ymin, z), (wall_x, ymax, z))

    # Constant-y wall: ruled by the x and z ticks.
    for x in xt:
        add((x, wall_y, zmin), (x, wall_y, zmax))
    for z in zt:
        add((xmin, wall_y, z), (xmax, wall_y, z))

    if not segments:
        return np.zeros((0, 3), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(segments, axis=0), dtype=np.float32)


def _project_ndc(
    camera: Camera3D,
    bounds: Bounds3D,
    points: np.ndarray,
    aspect: float,
    mvp: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Project world ``(N, 3)`` to normalised device coordinates ``(N, 3)``.

    Used only to *choose* between candidate edges, so a point behind the eye (w <= 0) does
    not need to be exact — it needs to lose. Clamping ``w`` keeps the comparison finite
    instead of producing an inf that would win every ``min``.

    ``mvp`` may be supplied by a caller that already has the matrix, which is not a
    micro-optimisation: :meth:`Camera3D.mvp` rebuilds a model, a view and a projection
    matrix from scratch, and this used to be called three times per :func:`tick_edges` for
    three sets of probes that share one camera.
    """
    if mvp is None:
        mvp = camera.mvp(aspect, bounds)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homo = np.column_stack([pts, np.ones(len(pts))])
    clip = homo @ np.asarray(mvp, dtype=np.float64).T
    w = clip[:, 3:4]
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    return clip[:, :3] / w


#: What :func:`tick_edges` returns: the two fixed coordinates of the x, y and z tick edges.
TickEdges = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]


def tick_edges(bounds: Bounds3D, camera: Camera3D, aspect: float = 1.0) -> TickEdges:
    """Where each axis' ticks are drawn, as the *other two* coordinates.

    Returns ``((x_edge_y, x_edge_z), (y_edge_x, y_edge_z), (z_edge_x, z_edge_y))``.

    The x ticks run along an edge at fixed ``(y, z)``, and there are four candidates; the
    same for y and z. The choice is made in screen space, because "which edge is in front"
    has no answer in world space:

    * x and y ticks take the edge that projects **lowest** on screen, putting them under
      the data where an axis belongs.
    * z ticks take the vertical edge that projects **leftmost**, matching every 3D plotting
      tool's convention and keeping the numbers clear of the other two axes' labels.

    All twelve candidates go through one matrix and one 12x4 multiply. They share a camera,
    so building three identical MVPs for three groups of four points was pure repetition.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in bounds)
    xmid, ymid, zmid = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0

    # Candidates are the four edges parallel to each axis, sampled at their midpoint.
    x_candidates = [(y, z) for y in (ymin, ymax) for z in (zmin, zmax)]
    y_candidates = [(x, z) for x in (xmin, xmax) for z in (zmin, zmax)]
    z_candidates = [(x, y) for x in (xmin, xmax) for y in (ymin, ymax)]
    probes = np.array(
        [[xmid, y, z] for y, z in x_candidates]
        + [[x, ymid, z] for x, z in y_candidates]
        + [[x, y, zmid] for x, y in z_candidates],
        dtype=np.float64,
    )
    ndc = _project_ndc(camera, bounds, probes, aspect)

    return (
        x_candidates[int(np.argmin(ndc[0:4, 1]))],
        y_candidates[int(np.argmin(ndc[4:8, 1]))],
        z_candidates[int(np.argmin(ndc[8:12, 0]))],
    )


# ----------------------------------------------------------------------------------
# Adaptive tick density
# ----------------------------------------------------------------------------------
#
# A 3D axis has to choose its own tick count, and the only honest input is how long the
# axis is **on screen**. The 2D renderer already works this way -- ``managers/axis.update``
# asks for one tick per 160 px across and 120 px down, and re-asks every frame, which is
# why zooming a 2D plot keeps re-labelling it at a readable density.
#
# The 3D axes did not: they took a fixed ``tick_count`` (5) from the data bounds, which are
# the same at every camera distance. Dollying in stretched five numbers across the whole
# window with nothing between them; dollying out squeezed the same five into a corner where
# they collided with each other and with the box. The numbers never changed either, so the
# axis carried no more information at any zoom than it did at the first frame.
#
# Measuring the *projected* edge fixes both at once, and it comes free of the nice-number
# ladder: as the pixel length grows the target count grows, ``_nice_step`` walks down the
# 1-2-5 rungs, and the labels stay round (5 -> 2.5 -> 2 -> 1 -> 0.5). The ladder is also
# what keeps it from flickering -- a target drifting from 6.2 to 6.8 lands on the same rung.

#: Pixels of projected axis per tick. Larger than the 2D renderer's 160 px *per axis* is not
#: needed: a 3D box shows three axes at once and they meet at the corners, so a slightly
#: looser density leaves room for the numbers to not run into each other there.
TICK_DENSITY_PX = 130.0

#: Fewest ticks an adaptive axis may ask for. Three, not two: a *target* of two puts
#: ``_nice_step`` at half the span, which on most ranges rounds up to a step larger than the
#: span and leaves the axis with a single number on it. Three reliably yields two or more.
MIN_ADAPTIVE_TICKS = 3

#: Most an adaptive axis may ask for. Deliberately generous, and below
#: :data:`MAX_TICKS` so :func:`axis_ticks` is never the thing that truncates.
#:
#: A tighter cap looks safer and is wrong: when the camera is close enough that the box
#: overflows the window, the projected edge is thousands of pixels long and only a slice of
#: it is visible. The count that gives *that slice* one label per :data:`TICK_DENSITY_PX` is
#: the large one; capping it low spaces the visible numbers hundreds of pixels apart, which
#: is the sparse-at-high-zoom behaviour this is here to remove. The labels that fall outside
#: the window cost a projection each and are dropped by ``draw_labels``.
MAX_ADAPTIVE_TICKS = 30


def axis_screen_lengths(
    bounds: Bounds3D,
    camera: Camera3D,
    aspect: float,
    width_px: float,
    height_px: float,
    edges: Optional[TickEdges] = None,
) -> Tuple[float, float, float]:
    """Length in window pixels of the three box edges the ticks are drawn on.

    The *tick* edge and not an arbitrary parallel one, so the measurement is of the line the
    numbers will actually be strung along.

    Two cases do not have a finite projected length, and they are opposites:

    * The axis points **at** the camera and collapses to a few pixels. That length is real,
      and the small count that follows from it is the right answer — there is nowhere to put
      more numbers.
    * An end of the axis is **behind** the eye, because the camera has been dollied into the
      data. The projection is meaningless (that is what the ``ok`` mask says), but the edge
      certainly crosses the whole window, so the window's diagonal is used as a lower bound.
      Returning 0 here instead — the obvious reading of "could not measure" — stripped the
      labels off exactly the axis the viewer had zoomed in to read.
    """
    if edges is None:
        edges = tick_edges(bounds, camera, aspect)
    (x_ey, x_ez), (y_ex, y_ez), (z_ex, z_ey) = edges
    xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in bounds)
    ends = [
        (xmin, x_ey, x_ez),
        (xmax, x_ey, x_ez),
        (y_ex, ymin, y_ez),
        (y_ex, ymax, y_ez),
        (z_ex, z_ey, zmin),
        (z_ex, z_ey, zmax),
    ]
    pixels, ok = project_points(ends, camera, bounds, width_px, height_px)
    diagonal = float(np.hypot(max(float(width_px), 1.0), max(float(height_px), 1.0)))
    lengths = []
    for i in range(3):
        a, b = 2 * i, 2 * i + 1
        if not (ok[a] and ok[b]):
            lengths.append(diagonal)
            continue
        length = float(np.hypot(*(pixels[b] - pixels[a])))
        lengths.append(length if np.isfinite(length) else diagonal)
    return (lengths[0], lengths[1], lengths[2])


def adaptive_tick_counts(
    bounds: Bounds3D,
    camera: Camera3D,
    aspect: float,
    width_px: float,
    height_px: float,
    *,
    edges: Optional[TickEdges] = None,
    forced: int = 0,
    density_px: float = TICK_DENSITY_PX,
) -> Tuple[int, int, int]:
    """Ticks to aim for on each axis, from how long that axis is on screen.

    ``forced > 0`` pins every axis to that count and skips the projection entirely — the
    escape hatch for a caller that wants a fixed axis, and the same convention the 2D
    renderer uses for ``axis_tick_count_x`` (0 means auto).

    A *target*, not a promise: :func:`axis_ticks` rounds it onto the 1-2-5 ladder, so the
    count that comes back is whatever that ladder allows near it.
    """
    if forced > 0:
        return (forced, forced, forced)
    density = max(float(density_px), 1.0)
    lengths = axis_screen_lengths(bounds, camera, aspect, width_px, height_px, edges)
    return tuple(  # type: ignore[return-value]
        int(np.clip(round(length / density) + 1, MIN_ADAPTIVE_TICKS, MAX_ADAPTIVE_TICKS))
        for length in lengths
    )


def _outward(value: float, lo: float, hi: float) -> float:
    """+1 when ``value`` is at the high end of ``[lo, hi]``, -1 at the low end.

    The direction a tick mark and its label step *away* from the box on that axis.
    """
    return 1.0 if abs(value - hi) < abs(value - lo) else -1.0


def axis_extents(bounds: Bounds3D) -> Tuple[float, float, float]:
    """The three axis spans, floored so a flat axis still yields a usable offset.

    A degenerate axis (every sample at the same z, say) has no extent of its own to
    measure an offset against, so it borrows the largest span that does exist. Without
    that floor its labels would be placed exactly on the box edge and overlap it.
    """
    spans = [
        float(bounds[1]) - float(bounds[0]),
        float(bounds[3]) - float(bounds[2]),
        float(bounds[5]) - float(bounds[4]),
    ]
    largest = max(max(spans), 1e-6)
    result = tuple(span if span > largest * 1e-3 else largest for span in spans)
    return result  # type: ignore[return-value]


def tick_marks(
    bounds: Bounds3D,
    ticks: Sequence[Tuple[np.ndarray, List[str]]],
    camera: Camera3D,
    aspect: float = 1.0,
    length_frac: float = TICK_LEN_FRAC,
    *,
    edges: Optional[TickEdges] = None,
) -> np.ndarray:
    """Short outward segments at every tick value, as ``GL_LINES`` vertices ``(N, 3)``.

    Each mark's length is a fraction of the axis it points *along*, so on anisotropic data
    the marks stay the same visual length on all three axes instead of the narrow ones
    being overwhelmed by a length derived from the widest.

    ``edges`` lets a caller that is also calling :func:`label_anchors` — which every real
    caller is — pass one :func:`tick_edges` result to both instead of solving the same
    camera question twice per frame.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in bounds)
    (xt, _), (yt, _), (zt, _) = ticks
    x_edge, y_edge, z_edge = edges if edges is not None else tick_edges(bounds, camera, aspect)
    span_x, span_y, span_z = axis_extents(bounds)
    frac = float(length_frac)

    segments: List[List[float]] = []

    # x ticks step outward along y, in the floor plane of the edge they sit on.
    ey, ez = x_edge
    dy = _outward(ey, ymin, ymax) * span_y * frac
    for x in xt:
        segments.append([x, ey, ez])
        segments.append([x, ey + dy, ez])

    # y ticks step outward along x.
    ex, ez = y_edge
    dx = _outward(ex, xmin, xmax) * span_x * frac
    for y in yt:
        segments.append([ex, y, ez])
        segments.append([ex + dx, y, ez])

    # z ticks step outward along both x and y: a vertical edge has two outward directions
    # and either alone can send the mark across the front face at some orientations.
    ex, ey = z_edge
    dx = _outward(ex, xmin, xmax) * span_x * frac * _DIAGONAL
    dy = _outward(ey, ymin, ymax) * span_y * frac * _DIAGONAL
    for z in zt:
        segments.append([ex, ey, z])
        segments.append([ex + dx, ey + dy, z])

    if not segments:
        return np.zeros((0, 3), dtype=np.float32)
    return np.ascontiguousarray(np.array(segments, dtype=np.float32))


# ----------------------------------------------------------------------------------
# Label anchors
# ----------------------------------------------------------------------------------


class LabelAnchor:
    """One piece of 3D axis text: where it lives in world space and what it says.

    Not a dataclass so ``__slots__`` keeps it cheap — a five-tick scene builds eighteen of
    these every time the camera moves.

    Attributes
    ----------
    position
        World-space ``(x, y, z)``. The caller projects it with the scene's MVP.
    text
        The string to draw.
    axis
        ``"x"``, ``"y"`` or ``"z"``.
    kind
        ``"tick"`` for a number, ``"title"`` for an axis name.
    pivot
        The world point this label is laid out *from*: the tip of its tick mark, or — for a
        title — the same offset taken at the middle of the axis. ``None`` for a hand-built
        anchor.

        The mark's tip rather than its root, so the pixel gap below is measured from the
        thing the number must clear. A mark is :data:`TICK_LEN_FRAC` of its axis long and
        so is short at one camera angle and long at another; starting the gap at the root
        would let a number land on top of its own mark whenever the mark projected long.

        A title takes the *same* offset even though it has no mark, so that the room its
        numbers reserved and the distance it is pushed out are measured from one surface.
        Measuring the numbers from the mark tips and the title from the box edge put the
        two at the same place on screen whenever the marks projected long — the title ended
        up level with its own numbers instead of outside them.

        :func:`draw_labels` lays the text out from the *pivot*, using ``position`` only for
        the direction ``position - pivot`` points in. That split is what makes the layout
        camera-independent: the outward step is then measured in screen pixels, so a number
        sits the same distance off the box whether its axis faces the viewer or is seen
        almost edge-on. Placing the text at the projected ``position`` instead — which is
        what this used to do — offset it by a *world* distance, and a world distance
        projects to anything between zero and half the screen depending on where the
        camera is: the numbers crowded onto the box at one angle and drifted far off it at
        the next.

    Not a dataclass so ``__slots__`` keeps it cheap — a five-tick scene builds eighteen of
    these every time the camera moves.
    """

    __slots__ = ("position", "text", "axis", "kind", "pivot")

    def __init__(
        self,
        position: Tuple[float, float, float],
        text: str,
        axis: str,
        kind: str,
        pivot: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        self.position = position
        self.text = text
        self.axis = axis
        self.kind = kind
        self.pivot = pivot

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<LabelAnchor {self.kind} {self.axis}={self.text!r} at {self.position}>"

    def __eq__(self, other: object) -> bool:
        # ``pivot`` is deliberately not compared: it is derived from the same edge choice
        # that produced ``position``, so two anchors agreeing on position, text, axis and
        # kind are the same label.
        if not isinstance(other, LabelAnchor):
            return NotImplemented
        return (
            np.allclose(self.position, other.position)
            and self.text == other.text
            and self.axis == other.axis
            and self.kind == other.kind
        )


def label_anchors(
    bounds: Bounds3D,
    ticks: Sequence[Tuple[np.ndarray, List[str]]],
    camera: Camera3D,
    aspect: float = 1.0,
    *,
    tick_labels: bool = True,
    axis_labels: Tuple[str, str, str] = ("", "", ""),
    edges: Optional[TickEdges] = None,
) -> List[LabelAnchor]:
    """Every piece of text the 3D axes want drawn, anchored in world space.

    Numbers sit just past their tick marks; titles sit past the numbers at the middle of
    their axis. Both follow the same edges :func:`tick_marks` chose, so text and marks can
    never end up on opposite sides of the box.

    Every offset is scaled by the extent of **the axis it is measured along** (see
    :func:`axis_extents`). The earlier version used one global scale — the box's longest
    edge — which on anisotropic data threw the labels far outside the box: with x spanning
    8 and y spanning 24, the y title landed 3.2 units past an x axis only 4 units wide.

    ``edges`` accepts a :func:`tick_edges` result the caller has already computed, so the
    text lands on the same edges as the marks without solving for them a second time.

    Every anchor also carries a ``pivot``: the tip of its tick mark (the box edge, for a
    title). ``position`` and ``pivot`` together give :func:`draw_labels` an outward
    *direction*; the outward *distance* is that function's business and is spent in pixels,
    not world units. The world offsets below therefore no longer decide how far off the box
    the text lands — they only have to be long enough to point somewhere unambiguous, and
    to keep a title outside its own numbers for any caller reading the anchors directly.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in bounds)
    (xt, xl), (yt, yl), (zt, zl) = ticks
    x_edge, y_edge, z_edge = edges if edges is not None else tick_edges(bounds, camera, aspect)
    span_x, span_y, span_z = axis_extents(bounds)

    anchors: List[LabelAnchor] = []

    # x ticks and title are offset along y.
    ey, ez = x_edge
    sy = _outward(ey, ymin, ymax) * span_y
    if tick_labels:
        for value, text in zip(xt, xl):
            anchors.append(
                LabelAnchor(
                    (float(value), ey + sy * TICK_LABEL_FRAC, ez),
                    text,
                    "x",
                    "tick",
                    (float(value), ey + sy * TICK_LEN_FRAC, ez),
                )
            )
    if axis_labels[0]:
        xmid = (xmin + xmax) / 2.0
        anchors.append(
            LabelAnchor(
                (xmid, ey + sy * AXIS_LABEL_FRAC, ez),
                axis_labels[0],
                "x",
                "title",
                (xmid, ey + sy * TICK_LEN_FRAC, ez),
            )
        )

    # y ticks and title are offset along x.
    ex, ez = y_edge
    sx = _outward(ex, xmin, xmax) * span_x
    if tick_labels:
        for value, text in zip(yt, yl):
            anchors.append(
                LabelAnchor(
                    (ex + sx * TICK_LABEL_FRAC, float(value), ez),
                    text,
                    "y",
                    "tick",
                    (ex + sx * TICK_LEN_FRAC, float(value), ez),
                )
            )
    if axis_labels[1]:
        ymid = (ymin + ymax) / 2.0
        anchors.append(
            LabelAnchor(
                (ex + sx * AXIS_LABEL_FRAC, ymid, ez),
                axis_labels[1],
                "y",
                "title",
                (ex + sx * TICK_LEN_FRAC, ymid, ez),
            )
        )

    # z ticks and title step out along both x and y, each scaled by its own axis.
    ex, ey = z_edge
    sx = _outward(ex, xmin, xmax) * span_x * _DIAGONAL
    sy = _outward(ey, ymin, ymax) * span_y * _DIAGONAL
    if tick_labels:
        for value, text in zip(zt, zl):
            anchors.append(
                LabelAnchor(
                    (
                        ex + sx * TICK_LABEL_FRAC,
                        ey + sy * TICK_LABEL_FRAC,
                        float(value),
                    ),
                    text,
                    "z",
                    "tick",
                    (ex + sx * TICK_LEN_FRAC, ey + sy * TICK_LEN_FRAC, float(value)),
                )
            )
    if axis_labels[2]:
        zmid = (zmin + zmax) / 2.0
        anchors.append(
            LabelAnchor(
                (
                    ex + sx * AXIS_LABEL_FRAC,
                    ey + sy * AXIS_LABEL_FRAC,
                    zmid,
                ),
                axis_labels[2],
                "z",
                "title",
                (ex + sx * TICK_LEN_FRAC, ey + sy * TICK_LEN_FRAC, zmid),
            )
        )
    return anchors


def project_points(
    points: Sequence[Sequence[float]],
    camera: Camera3D,
    bounds: Bounds3D,
    width_px: float,
    height_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project world ``(N, 3)`` to top-left-origin window pixels.

    Returns ``(pixels, ok)`` — an ``(N, 2)`` array and an ``(N,)`` boolean mask. ``ok`` is
    False for a point behind the eye or one whose projection is not finite; its row in
    ``pixels`` is meaningless and must not be read. Dropping those rather than dividing by
    a negative ``w`` is what keeps a label from being mirrored to the wrong side of the
    screen.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=bool)
    aspect = max(float(width_px), 1.0) / max(float(height_px), 1.0)
    mvp = np.asarray(camera.mvp(aspect, bounds), dtype=np.float64)
    clip = np.column_stack([pts, np.ones(len(pts))]) @ mvp.T
    w = clip[:, 3]
    ok = w > 1e-9
    safe_w = np.where(ok, w, 1.0)
    ndc = clip[:, :2] / safe_w[:, None]
    pixels = np.column_stack(
        [
            (ndc[:, 0] * 0.5 + 0.5) * float(width_px),
            (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * float(height_px),
        ]
    )
    ok &= np.all(np.isfinite(pixels), axis=1)
    return pixels, ok


def project_anchors(
    anchors: Sequence[LabelAnchor],
    camera: Camera3D,
    bounds: Bounds3D,
    width_px: float,
    height_px: float,
) -> List[Tuple[LabelAnchor, Tuple[float, float], float]]:
    """Project ``anchors`` to top-left-origin window pixels.

    Returns ``(anchor, (x_px, y_px), depth)`` for each anchor that lands in front of the
    eye; anchors behind it are dropped rather than mirrored to the wrong side of the
    screen, which is what a naive divide by a negative ``w`` would do.
    """
    if not anchors:
        return []
    aspect = max(float(width_px), 1.0) / max(float(height_px), 1.0)
    mvp = np.asarray(camera.mvp(aspect, bounds), dtype=np.float64)
    pts = np.array([a.position for a in anchors], dtype=np.float64)
    homo = np.column_stack([pts, np.ones(len(pts))])
    clip = homo @ mvp.T

    out: List[Tuple[LabelAnchor, Tuple[float, float], float]] = []
    for anchor, row in zip(anchors, clip):
        w = float(row[3])
        if w <= 1e-9:
            continue
        ndc_x, ndc_y, ndc_z = row[0] / w, row[1] / w, row[2] / w
        x_px = (ndc_x * 0.5 + 0.5) * float(width_px)
        y_px = (1.0 - (ndc_y * 0.5 + 0.5)) * float(height_px)
        if not (np.isfinite(x_px) and np.isfinite(y_px)):
            continue
        out.append((anchor, (float(x_px), float(y_px)), float(ndc_z)))
    return out


def scene_radius(bounds: Optional[Bounds3D]) -> float:
    """The bounding radius of ``bounds`` — re-exported so callers need one import."""
    return bounds_centre_radius(bounds)[1]


# ----------------------------------------------------------------------------------
# Screen-space text
# ----------------------------------------------------------------------------------

#: Axis titles are drawn this much larger than the tick numbers.
TITLE_SCALE = 1.25

# The outward layout below is spent in **pixels**, measured from the box edge outward along
# the projected outward direction, and a label's own text box is pushed clear of that
# distance rather than centred on it. Both halves matter:
#
# * Pixels, because the previous world-space offset projected to a different screen gap at
#   every camera angle -- numbers touching the box when an axis was foreshortened and
#   floating far off it when the same axis faced the viewer.
# * Clear of, because a text box centred on its anchor overlaps whatever the anchor sits
#   on. A number centred a few pixels off the box edge is drawn *through* the edge.

#: Gap between the tip of a tick mark and the nearest corner of its number, in window
#: pixels. Measured from the mark's tip, not from the box: the mark is the thing the number
#: must not touch, and how long it looks on screen is a camera question.
TICK_GAP_PX = 6.0

#: Gap between the outermost tick number of an axis and the nearest corner of its title.
#: Measured from the numbers rather than from the box so a title clears them whatever they
#: say -- "0.5" and "-1.25e+06" do not reserve the same amount of room.
TITLE_GAP_PX = 10.0

#: Where a title goes on an axis whose numbers are all hidden or off-screen.
TITLE_FALLBACK_GAP_PX = 16.0

#: Overlap, in pixels, that two tick numbers may share before the later one is dropped.
#: Not zero: boxes that merely touch read as correctly spaced, and culling on contact
#: throws away every other number on a densely ticked axis for no gain.
LABEL_OVERLAP_SLACK_PX = 2.0

#: A projected outward direction shorter than this is no direction at all (the axis is
#: pointing at the camera). The label then steps away from the centre of the box instead.
_MIN_DIR_PX = 1e-3


def _outward_screen_dirs(
    anchor_px: np.ndarray, pivot_px: np.ndarray, ok: np.ndarray, centre_px: np.ndarray
) -> np.ndarray:
    """Unit screen directions pointing out of the box, one per label.

    ``anchor_px - pivot_px`` is the outward direction the world offset was built to
    express. It degenerates exactly when that offset points at the camera, and there the
    fallback is ``anchor - box centre``: a label near the middle of the picture has no
    meaningful "outward" of its own, but the box always does. If even that is degenerate
    the label steps straight down, which is where a reader looks for an axis number.
    """
    delta = anchor_px - pivot_px
    length = np.hypot(delta[:, 0], delta[:, 1])
    bad = (~ok) | (length < _MIN_DIR_PX)
    if np.any(bad):
        fallback = anchor_px[bad] - centre_px
        fallback_len = np.hypot(fallback[:, 0], fallback[:, 1])
        # Down, for a label sitting on the projected centre of the box.
        fallback = np.where(fallback_len[:, None] < _MIN_DIR_PX, np.array([[0.0, 1.0]]), fallback)
        delta = delta.copy()
        delta[bad] = fallback
        length = np.hypot(delta[:, 0], delta[:, 1])
    return delta / np.maximum(length, _MIN_DIR_PX)[:, None]


def _place_box(
    origin: Tuple[float, float],
    direction: Tuple[float, float],
    size: Tuple[float, float],
    distance: float,
) -> Tuple[Tuple[float, float], float]:
    """Top-left corner of a ``size`` text box set ``distance`` px out from ``origin``.

    Returns ``(top_left, reach)``, where ``reach`` is how far past ``origin`` the box's far
    side ends up — what the next thing out along the same direction (an axis title, past
    its own numbers) has to clear.

    The box is placed so its *nearest* side is ``distance`` from the origin, whatever the
    direction: its centre goes out by ``distance`` plus the box's own half-extent measured
    along that direction. Centring the box on the offset point instead is what let long
    numbers overhang back onto the axis they label.
    """
    ux, uy = direction
    half_w, half_h = size[0] * 0.5, size[1] * 0.5
    half_extent = abs(ux) * half_w + abs(uy) * half_h
    centre_dist = distance + half_extent
    cx = origin[0] + ux * centre_dist
    cy = origin[1] + uy * centre_dist
    return (cx - half_w, cy - half_h), centre_dist + half_extent


def _overlaps(box: Tuple[float, float, float, float], others: Sequence[Tuple[float, ...]]) -> bool:
    """Whether ``(x0, y0, x1, y1)`` overlaps any of ``others`` by more than the slack."""
    slack = LABEL_OVERLAP_SLACK_PX
    for other in others:
        if (
            box[0] < other[2] - slack
            and box[2] > other[0] + slack
            and box[1] < other[3] - slack
            and box[3] > other[1] + slack
        ):
            return True
    return False


def draw_labels(engine: object) -> int:
    """Paint the engine's 3D tick numbers and axis titles. Returns how many were drawn.

    Reads ``engine._axes3d_labels`` and ``engine._axes3d_bounds``, which
    ``ensure_3d_axes`` refreshes whenever the camera or the data moves. The *world* half of
    the placement is settled there; the screen half — how far off the box a number sits and
    which way it steps — is settled here, because it is the only place the text has been
    measured. See :data:`TICK_GAP_PX`.

    Numbers are laid out first and titles second, per axis, so a title can be pushed past
    however much room its own numbers turned out to need instead of past a guess.

    Silently does nothing when imgui is missing, when there are no anchors, or when the
    engine is mid-teardown. It runs once per frame from inside the imgui frame, so it must
    not be able to take the frame down.
    """
    try:
        import imgui
    except (ImportError, Exception):  # pragma: no cover - GL-less import guard
        return 0

    anchors = getattr(engine, "_axes3d_labels", None)
    bounds = getattr(engine, "_axes3d_bounds", None)
    if not anchors or bounds is None:
        return 0
    # ``_axes3d_camera`` is the *resolved* camera the anchors were placed with — the panel
    # camera with the auto-fit distance and pan filled in. It has to be that one and not
    # ``camera3d``: an auto-framed camera keeps ``distance is None``, so projecting through
    # it lands the numbers somewhere the geometry never went (~90 px off at the stock
    # framing, which is what tore the numbers away from their tick marks). The fallback is
    # for a caller that has anchors but never ran ``ensure_3d_axes``.
    camera = getattr(engine, "_axes3d_camera", None) or getattr(engine, "camera3d", None)
    if camera is None:
        return 0

    width = float(getattr(engine, "width", 0) or 0)
    height = float(getattr(engine, "height", 0) or 0)
    if width < 1.0 or height < 1.0:
        return 0

    projected = project_anchors(anchors, camera, bounds, width, height)
    if not projected:
        return 0

    # Same contrast rule as the 2D tick labels: ink chosen off the background luminance,
    # so the numbers stay readable when the user flips to the light theme.
    options = getattr(engine, "options", None)
    background = (0.0, 0.0, 0.0)
    visual = getattr(options, "visual", None)
    if visual is not None:
        background = tuple(getattr(visual, "background_color", (0.0, 0.0, 0.0)))[:3]
    luminance = 0.299 * background[0] + 0.587 * background[1] + 0.114 * background[2]
    if luminance > 0.5:
        tick_color = imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 1.0)
        title_color = imgui.get_color_u32_rgba(0.05, 0.05, 0.05, 1.0)
    else:
        tick_color = imgui.get_color_u32_rgba(0.82, 0.86, 0.92, 1.0)
        title_color = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0)

    # pyimgui 2.0's ``add_text`` takes ``(x, y, col, text)`` and nothing else — the
    # font/size overload is unwrapped and there is no ``imgui.get_font()`` to feed it.
    # Scaling therefore goes through the vertex-buffer transform the 2D axis renderer
    # already uses for its rotated y-label, which degrades to unscaled text rather than
    # raising if anything about the buffer is unexpected.
    from .axis import draw_text_scaled

    draw_list = imgui.get_background_draw_list()
    off_x, off_y = getattr(engine, "_panel_offset_px", (0.0, 0.0))

    # Everything the layout needs in screen space: where each label's root on the box is,
    # and which way is "out" from there. A hand-built anchor with no pivot falls back to
    # its own position, which makes its direction degenerate and sends it to the
    # box-centre rule in ``_outward_screen_dirs`` — the sane reading of "no root given".
    anchor_px = np.array([p for _, p, _ in projected], dtype=np.float64)
    pivot_world = [a.pivot if a.pivot is not None else a.position for a, _, _ in projected]
    pivot_px, pivot_ok = project_points(pivot_world, camera, bounds, width, height)
    centre, _ = bounds_centre_radius(bounds)
    centre_px, centre_ok = project_points([centre], camera, bounds, width, height)
    origin_px = np.where(pivot_ok[:, None], pivot_px, anchor_px)
    dirs = _outward_screen_dirs(
        anchor_px,
        pivot_px,
        pivot_ok,
        centre_px[0] if bool(centre_ok[0]) else np.array([width * 0.5, height * 0.5]),
    )

    def place(index: int, size: Tuple[float, float], distance: float):
        return _place_box(
            (origin_px[index, 0], origin_px[index, 1]),
            (dirs[index, 0], dirs[index, 1]),
            size,
            distance,
        )

    def on_screen(px: float, py: float, size: Tuple[float, float]) -> bool:
        # A label whose whole box is off-screen is skipped rather than clamped to the edge,
        # where it would pile up with the others into an unreadable stack.
        return not (px < -size[0] or px > width + off_x or py < -size[1] or py > height + off_y)

    # --- Numbers first ---------------------------------------------------------------
    #
    # An axis seen almost end-on packs its ticks into a few pixels, and every number then
    # lands on the one before it: a smear that reads as a rendering fault rather than as an
    # axis. Dropping the ones that would collide leaves a sparser but legible axis, which is
    # the same bargain the 2D renderer makes.
    #
    # Per axis, not across all three. Two axes only ever meet at the corner they share, and
    # there both numbers are correct and each belongs to a different axis -- deleting one
    # silently truncates an axis' range to keep a couple of pixels tidy. Within an axis the
    # opposite holds: the numbers are interchangeable, so thinning them costs nothing but
    # resolution. Titles are exempt from both -- an axis with no name is worse than a title
    # that grazes a number.
    painted: dict = {"x": [], "y": [], "z": []}
    to_draw: List[Tuple[bool, str, Tuple[float, float]]] = []
    axis_reach = {"x": 0.0, "y": 0.0, "z": 0.0}

    for i, (anchor, _, _) in enumerate(projected):
        if anchor.kind == "title" or not anchor.text:
            continue
        measured = imgui.calc_text_size(anchor.text)
        size = (measured.x, measured.y)
        (px, py), reach = place(i, size, TICK_GAP_PX)
        # Reserved whether or not the number survives the cull: the title clears the room
        # the axis' numbers asked for, so it does not creep inward when one is dropped.
        axis_reach[anchor.axis] = max(axis_reach.get(anchor.axis, 0.0), reach)
        px += off_x
        py += off_y
        if not on_screen(px, py, size):
            continue
        neighbours = painted.setdefault(anchor.axis, [])
        box = (px, py, px + size[0], py + size[1])
        if _overlaps(box, neighbours):
            continue
        neighbours.append(box)
        to_draw.append((False, anchor.text, (px, py)))

    # --- Then the titles, past whatever room the numbers took ------------------------
    for i, (anchor, _, _) in enumerate(projected):
        if anchor.kind != "title" or not anchor.text:
            continue
        measured = imgui.calc_text_size(anchor.text)
        size = (measured.x * TITLE_SCALE, measured.y * TITLE_SCALE)
        room = axis_reach.get(anchor.axis, 0.0)
        distance = (room + TITLE_GAP_PX) if room > 0.0 else TITLE_FALLBACK_GAP_PX
        (px, py), _ = place(i, size, distance)
        px += off_x
        py += off_y
        if not on_screen(px, py, size):
            continue
        to_draw.append((True, anchor.text, (px, py)))

    for is_title, text, (px, py) in to_draw:
        if is_title:
            draw_text_scaled(imgui, draw_list, text, title_color, (px, py), TITLE_SCALE)
        else:
            draw_list.add_text(px, py, tick_color, text)
    return len(to_draw)
