"""The live legend overlay, with smart grouping.

Until this module existed, ``gplt.legend()`` drew **nothing** in the live window: it
assigned ``plot.legend_labels`` and no renderer read it. (The matplotlib savefig
fallback builds its own legend from matplotlib's artist handles, so it never read that
attribute either -- see the note on agreement below.) This is the same defect
``xlabel``/``ylabel``/``title`` had, and it is fixed the same way: draw the overlay with
``imgui.get_background_draw_list()``, exactly as :mod:`glplot.renderers.axis` draws the
tick and annotation text, so the legend composites over the scene and survives the
interaction cache (the overlay is re-drawn every frame, the cached impostor is not).

**Smart grouping** is the reason this module is more than a for-loop. Plotting a family
in a loop -- the normal way to get 100 curves on a plot -- used to mean 100 legend rows,
which is not a legend, it is a wall. Three collapses run, in order:

1. exact duplicate labels fold into one entry with a count;
2. labels differing only by a trailing index (``Curve 0`` ... ``Curve 99``) fold into one
   entry on the common stem, ``Curve (x100)``;
3. whatever is still over ``legend_max_items`` folds into a ``+N more layers`` row.

Step 3 reuses the wording of the existing ``+N more layers`` entry that
``pyplot.legend()`` and ``utils.preview`` both already emit, so the live window and the
exported PNG say the same thing rather than inventing two dialects.

Everything above :func:`draw_legend` is pure -- no imgui, no OpenGL, numpy only for the
colour sampling -- so it is unit-testable with zero guards, and so the matplotlib export
path can import :func:`group_legend_entries` to grow the same grouping without taking a
GL dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..core.layers import BaseLayer

RGBA = Tuple[float, float, float, float]

#: The overflow row's wording. Shared verbatim with ``pyplot.legend`` and
#: ``utils.preview.render_preview`` -- if the live legend and the exported PNG disagree
#: on this string, one of them is lying about the same plot. Formatted with the count.
MORE_LABEL = "+{n} more layers"

#: Grouped-entry suffix. ``x`` here is U+00D7 MULTIPLICATION SIGN, which is inside
#: ProggyClean's U+0020-U+00FF range and therefore actually renders -- unlike the arrows
#: and symbols outside Latin-1, which the default font silently draws as ``?``
#: (CONTRACT section 3). Do not "upgrade" this to a nicer glyph without a font.
COUNT_SUFFIX = " (×{n})"

#: Legal ``legend_loc`` values. Matplotlib's spelling, because ``utils.preview`` already
#: passes ``loc="upper right"`` to ``ax.legend`` and the two should be talking about the
#: same thing.
LEGEND_LOCATIONS = ("best", "upper right", "upper left", "lower left", "lower right")

#: A label ends in an index when it ends in digits reachable across a run of separator
#: characters. ``stem`` is lazy so it stops at the *first* position from which the tail
#: parses -- "Curve 100" yields ("Curve", " ", "100"), not ("Curve 1", "", "00").
_INDEXED_RE = re.compile(r"^(?P<stem>.*?)(?P<sep>[ _\-#:]*)(?P<idx>\d+)$")

#: Colour arrays are sampled, not scanned: a scatter can hold millions of rows and this
#: runs in the per-frame overlay pass. Evenly-spaced samples decide "uniform vs varied"
#: and supply the gradient's endpoints.
_COLOR_SAMPLES = 256

#: How much better a rival corner must score before ``legend_loc="best"`` will actually
#: move an already-placed legend, as a fraction of the legend box's own area. Zero would
#: make the box twitch between corners on sub-pixel scoring noise during a pan.
_BEST_HYSTERESIS = 0.10

#: Geometry, in pixels at font scale 1.0.
_SWATCH_W = 22.0
_SWATCH_H = 11.0
_PAD = 6.0
_GAP = 6.0
_ROW_GAP = 3.0
_MARGIN = 10.0


@dataclass
class LegendEntry:
    """One drawn row: display text, swatch colour(s), and how many layers it stands for.

    ``colors`` holds one colour for a flat swatch or two for a gradient. Two is not
    decoration -- it is the entry refusing to claim a single colour it does not have,
    either because the grouped layers disagree or because one layer is itself
    per-element coloured.
    """

    label: str
    colors: Tuple[RGBA, ...]
    count: int = 1


def split_indexed_label(label: str) -> Optional[Tuple[str, int]]:
    """Split ``"Curve 12"`` into ``("Curve", 12)``; return None when there is no index.

    Rejections are as load-bearing as the matches, because a false positive silently
    merges two unrelated series into one row:

    * an empty stem (``"0"``, ``"1"``) -- collapsing those yields a nameless ``(x2)``;
    * a stem ending in ``.``, which is how ``"Data 3.5"`` and ``"Data 3.6"`` present
      themselves to a trailing-digit regex. They are decimals, not an indexed family.
    """
    match = _INDEXED_RE.match(label)
    if match is None:
        return None
    stem = match.group("stem")
    if not stem or stem.endswith("."):
        return None
    return stem, int(match.group("idx"))


def _sample_colors(colors: Any) -> Optional[Tuple[RGBA, ...]]:
    """Reduce a per-element colour array to a flat colour or a gradient pair.

    Returns one colour when the sampled rows agree, two (first and last sampled) when
    they do not, and None when there is nothing usable. Sampling is a deliberate
    approximation: an array whose variation hides entirely between two samples will read
    as flat. The alternative is a full scan of a million-row array every frame, which is
    a worse bug than an occasionally under-described swatch.
    """
    if colors is None:
        return None
    arr = np.asarray(colors, dtype=np.float32)
    if arr.ndim == 1:
        return (_as_rgba(arr),) if arr.size >= 3 else None
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
        return None

    n = arr.shape[0]
    if n <= _COLOR_SAMPLES:
        sample = arr
    else:
        sample = arr[np.linspace(0, n - 1, _COLOR_SAMPLES).astype(np.intp)]

    first = _as_rgba(sample[0])
    if bool(np.all(sample[:, :3] == sample[0, :3])):
        return (first,)
    return (first, _as_rgba(sample[-1]))


def _as_rgba(row: Any) -> RGBA:
    """Coerce a 3- or 4-component colour to RGBA floats, defaulting alpha to opaque."""
    vals = [float(v) for v in np.asarray(row).ravel()[:4]]
    while len(vals) < 4:
        vals.append(1.0)
    return (vals[0], vals[1], vals[2], vals[3])


def layer_colors(layer: BaseLayer) -> Tuple[RGBA, ...]:
    """The colour(s) a layer actually draws with, per its own renderer's resolution.

    Each branch mirrors the renderer that owns the layer type, including that renderer's
    fallback, so the swatch cannot disagree with the pixels:

    * ``polyline`` -> ``style.color``, falling back to opaque black (``polyline.py:173``);
    * ``patch`` -> ``style.face_color`` (``patch.py:112``), the only colour it fills with;
    * ``scatter`` / ``line_family`` / 3D -> the baked per-element array, which has no
      single colour. The **representative is the first element**, and when the sampled
      rows disagree a second (the last sampled) is returned so the swatch draws as a
      gradient instead of claiming the first colour speaks for all of them.
    """
    style = getattr(layer, "style", None)
    layer_type = getattr(layer, "layer_type", "")

    if layer_type == "patch":
        face = getattr(style, "face_color", None)
        if face is not None:
            return (_as_rgba(face),)

    sampled = _sample_colors(getattr(layer, "colors", None))
    if sampled is not None:
        return sampled

    color = getattr(style, "color", None)
    if color is not None:
        return (_as_rgba(color),)
    if layer_type == "polyline":
        return ((0.0, 0.0, 0.0, 1.0),)
    return ((0.7, 0.7, 0.72, 1.0),)


def collect_legend_items(layers: Sequence[BaseLayer]) -> List[Tuple[str, Tuple[RGBA, ...]]]:
    """Gather ``(label, colors)`` for every visible, labelled, non-system layer.

    Invisible layers are skipped because a legend describing something the user cannot
    see is noise, and the 3D axis scaffolding is skipped because it is chrome the engine
    added, not data the user plotted. That exclusion reads
    :data:`~glplot.core.camera3d.SYSTEM_3D_ARTISTS` rather than naming the artists here:
    the local ``("axis3d", "floor3d")`` literal it replaced went stale as soon as the grid
    and tick artists were added, and a legend listing "3D grid" is exactly the noise this
    filter exists to prevent.
    """
    from ..core.camera3d import SYSTEM_3D_ARTISTS

    items: List[Tuple[str, Tuple[RGBA, ...]]] = []
    for layer in layers:
        label = str(getattr(layer, "label", "") or "")
        if not label:
            continue
        style = getattr(layer, "style", None)
        if not getattr(style, "visible", True):
            continue
        metadata = getattr(layer, "metadata", None) or {}
        if metadata.get("artist") in SYSTEM_3D_ARTISTS:
            continue
        items.append((label, layer_colors(layer)))
    return items


def _merge_colors(groups: Sequence[Tuple[RGBA, ...]]) -> Tuple[RGBA, ...]:
    """Collapse several members' colours into one flat colour or one gradient pair."""
    flat: List[RGBA] = []
    for colors in groups:
        flat.extend(colors)
    if not flat:
        return ((0.7, 0.7, 0.72, 1.0),)
    if all(c[:3] == flat[0][:3] for c in flat):
        return (flat[0],)
    return (flat[0], flat[-1])


def group_legend_entries(
    items: Sequence[Tuple[str, Tuple[RGBA, ...]]],
    max_items: Optional[int] = None,
) -> List[LegendEntry]:
    """Fold raw ``(label, colors)`` pairs into the rows a legend should actually show.

    The three collapses run in order -- exact duplicates, then indexed families, then the
    ``max_items`` overflow -- because each one feeds the next: deduplicating first means
    a family's count is a layer count rather than a distinct-label count, and grouping
    before truncating is what lets 100 curves fit under a limit of 5 instead of being
    guillotined into "Curve 0..4, +95 more".

    An indexed family needs **two or more distinct labels** to form. A lone ``Curve 0``
    stays ``Curve 0``: rewriting it to ``Curve (x1)`` would be pure loss.

    Order is first-appearance throughout, which is layer order, which is the order the
    user plotted in.
    """
    # 1. Exact duplicates.
    order: List[str] = []
    counts: dict = {}
    palettes: dict = {}
    for label, colors in items:
        if label not in counts:
            order.append(label)
            counts[label] = 0
            palettes[label] = []
        counts[label] += 1
        palettes[label].append(colors)

    # 2. Indexed families, keyed on the common stem.
    stem_order: List[str] = []
    stem_members: dict = {}
    for label in order:
        split = split_indexed_label(label)
        stem = split[0] if split is not None else None
        key = ("stem", stem) if stem is not None else ("exact", label)
        if key not in stem_members:
            stem_order.append(key)
            stem_members[key] = []
        stem_members[key].append(label)

    entries: List[LegendEntry] = []
    for key in stem_order:
        members = stem_members[key]
        total = sum(counts[m] for m in members)
        colors = _merge_colors([c for m in members for c in palettes[m]])
        if key[0] == "stem" and len(members) > 1:
            text = key[1] + COUNT_SUFFIX.format(n=total)
        elif total > 1:
            text = members[0] + COUNT_SUFFIX.format(n=total)
        else:
            text = members[0]
        entries.append(LegendEntry(label=text, colors=colors, count=total))

    # 3. Overflow.
    if max_items is not None and int(max_items) >= 0 and len(entries) > int(max_items):
        limit = int(max_items)
        hidden = sum(e.count for e in entries[limit:])
        entries = entries[:limit]
        entries.append(
            LegendEntry(label=MORE_LABEL.format(n=hidden), colors=(), count=hidden),
        )
    return entries


def resolve_legend_show(options: Any, plot: Any) -> bool:
    """Decide whether the legend draws, from its two independent channels.

    ``options.legend_show`` is the GUI's channel and ``plot.legend_show`` -- set by
    ``pyplot.legend()`` -- is the public API's, the same two-channel split
    ``AxisRenderer._resolve_annotation`` already uses for the axis labels. The option is
    tri-state (``None`` means "not configured") specifically so that the GUI can turn the
    legend **off** after ``legend()`` turned it on; a plain bool defaulting to False
    could never express the difference between "unset" and "hidden", and the GUI's
    checkbox would be unable to win. The option wins when set, for the axis renderer's
    reason: it is the one the user can watch themselves click.
    """
    configured = getattr(options, "legend_show", None)
    if configured is not None:
        return bool(configured)
    return bool(getattr(plot, "legend_show", False))


def resolve_max_items(options: Any, plot: Any) -> Optional[int]:
    """``legend(max_items=...)`` wins over the option, being the more specific request."""
    from_call = getattr(plot, "legend_max_items", None)
    if from_call is not None:
        return int(from_call)
    value = getattr(options, "legend_max_items", None)
    return None if value is None else int(value)


def build_entries(plot: Any) -> List[LegendEntry]:
    """The full grouping pipeline for a plot, as the renderer and the tests both see it."""
    layers = getattr(getattr(plot, "scene", None), "layers", None) or []
    return group_legend_entries(
        collect_legend_items(layers),
        resolve_max_items(getattr(plot, "options", None), plot),
    )


def _background_is_light(options: Any) -> bool:
    """Perceived luminance of the backdrop, by the same test the axis renderer applies."""
    visual = getattr(options, "visual", None)
    gradient = getattr(visual, "gradient_background", None)
    if gradient is not None and getattr(gradient, "enabled", False):
        top, bottom = gradient.top_color, gradient.bottom_color
        bg = (
            0.5 * (top[0] + bottom[0]),
            0.5 * (top[1] + bottom[1]),
            0.5 * (top[2] + bottom[2]),
        )
    else:
        bg = getattr(visual, "background_color", (0.0, 0.0, 0.0))
    return (0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]) > 0.5


def _project_bounds(plot: Any, mvp: Any) -> np.ndarray:
    """Screen-space AABBs of the visible data layers, as an ``(N, 4)`` array of x0,y0,x1,y1.

    For the ``best`` placement search only. Every layer's four corners go through **one**
    matmul rather than one per corner: this runs in the per-frame overlay pass, and the
    obvious per-layer loop measured 1.15 ms at 100 layers -- 7% of a 60 Hz frame spent
    deciding where to put a box. Batching halves that to ~0.55 ms.

    The rest is ``get_intrinsic_bounds``, which recomputes from the numpy arrays on every
    call by design (there is no bounds cache to invalidate -- see the layer docs), so
    0.55 ms is the floor for an honest answer. That is why :func:`_choose_location` caches
    the *decision* rather than trying to make this cheaper.
    """
    corners: List[Tuple[float, float]] = []
    for layer in getattr(plot.scene, "layers", []):
        if not getattr(getattr(layer, "style", None), "visible", True):
            continue
        try:
            bounds = layer.get_intrinsic_bounds()
        except Exception:  # pragma: no cover - a layer mid-edit must not kill the frame
            bounds = None
        if not bounds:
            continue
        x0, x1, y0, y1 = bounds
        dx, dy = getattr(layer, "translation", (0.0, 0.0))
        corners.extend(
            (
                (x0 + dx, y0 + dy),
                (x1 + dx, y0 + dy),
                (x0 + dx, y1 + dy),
                (x1 + dx, y1 + dy),
            )
        )
    if not corners:
        return np.empty((0, 4), dtype=np.float32)

    pts = np.ones((len(corners), 4), dtype=np.float32)
    pts[:, :2] = np.asarray(corners, dtype=np.float32)
    ndc = pts @ np.asarray(mvp, dtype=np.float32).T
    w = ndc[:, 3:4]
    ndc = np.divide(ndc, w, out=ndc, where=w != 0.0)

    sx = ((ndc[:, 0] + 1.0) * 0.5 * float(plot.width)).reshape(-1, 4)
    sy = ((1.0 - ndc[:, 1]) * 0.5 * float(plot.height)).reshape(-1, 4)
    return np.column_stack((sx.min(1), sy.min(1), sx.max(1), sy.max(1)))


def _corner_origin(loc: str, width: float, height: float, box_w: float, box_h: float):
    """Top-left pixel of a legend box of ``box_w`` x ``box_h`` parked at ``loc``."""
    left = _MARGIN
    right = max(width - box_w - _MARGIN, _MARGIN)
    top = _MARGIN
    bottom = max(height - box_h - _MARGIN, _MARGIN)
    return {
        "upper left": (left, top),
        "upper right": (right, top),
        "lower left": (left, bottom),
        "lower right": (right, bottom),
    }[loc]


def _choose_location(plot: Any, box_w: float, box_h: float) -> Tuple[float, float]:
    """Place the box, resolving ``best`` to the corner the data leaves emptiest.

    ``best`` scores the four corners by how much projected data area each would cover and
    takes the lowest, ties going to ``upper right`` -- matplotlib's default, and the
    corner ``utils.preview`` hardcodes, so an unobstructed plot lands the legend in the
    same corner on screen as in its exported PNG.

    Two things stop this from being a naive per-frame search, and they share one fix.
    The search costs ~0.55 ms at 100 layers -- ``get_intrinsic_bounds`` recomputes from
    the numpy arrays on every call by design, so there is no cheaper honest answer -- and
    re-deciding every frame would let the legend **hop between corners mid-pan**, which
    is a worse defect than the cost. So the decision is cached against the camera window
    and the layer count, and re-decided only when one of those actually changes; and when
    it is re-decided, the incumbent corner is kept unless a rival beats it by
    :data:`_BEST_HYSTERESIS` of the box's own area. A legend that twitches as you drag is
    not "best", whatever it scores.

    A pure-3D scene has no 2D world bounds worth projecting through this matrix, so it
    skips the search rather than scoring garbage.
    """
    options = plot.options
    loc = str(getattr(options, "legend_loc", "best") or "best")
    if loc not in LEGEND_LOCATIONS:
        loc = "best"
    width, height = float(plot.width), float(plot.height)

    if loc != "best":
        return _corner_origin(loc, width, height, box_w, box_h)

    try:
        window = tuple(plot.camera_controller.world_window(plot.width, plot.height))
    except Exception:  # pragma: no cover - defensive
        window = None
    key = (window, len(getattr(plot.scene, "layers", ())), round(box_w, 1), round(box_h, 1))
    cached = getattr(plot, "_legend_best_cache", None)
    if cached is not None and cached[0] == key:
        return _corner_origin(cached[1], width, height, box_w, box_h)

    corners = ("upper right", "upper left", "lower left", "lower right")
    boxes = np.empty((0, 4), dtype=np.float32)
    if not plot._is_pure_3d_scene():
        try:
            mvp = plot.camera_controller.mvp(plot.width, plot.height)
            boxes = _project_bounds(plot, mvp)
        except Exception:  # pragma: no cover - defensive; fall through to upper right
            boxes = np.empty((0, 4), dtype=np.float32)
    if boxes.shape[0] == 0:
        return _corner_origin("upper right", width, height, box_w, box_h)

    # Score every layer against a corner in one pass. `upper right` is first, so a plot
    # whose data leaves all four corners equally clear (score 0) keeps matplotlib's --
    # and the savefig path's -- default corner rather than an arbitrary one.
    scores = {}
    for corner in corners:
        ox, oy = _corner_origin(corner, width, height, box_w, box_h)
        ow = np.minimum(ox + box_w, boxes[:, 2]) - np.maximum(ox, boxes[:, 0])
        oh = np.minimum(oy + box_h, boxes[:, 3]) - np.maximum(oy, boxes[:, 1])
        scores[corner] = float(np.sum(np.clip(ow, 0.0, None) * np.clip(oh, 0.0, None)))

    best_loc = min(corners, key=lambda c: scores[c])
    incumbent = cached[1] if cached is not None else None
    if incumbent in scores:
        margin = _BEST_HYSTERESIS * box_w * box_h
        if scores[incumbent] - scores[best_loc] < margin:
            best_loc = incumbent

    plot._legend_best_cache = (key, best_loc)
    return _corner_origin(best_loc, width, height, box_w, box_h)


def _add_scaled_text(imgui, draw_list, x: float, y: float, color: int, text: str, scale: float):
    """Draw ``text`` at ``scale``, working around pyimgui 2.0 having no sized ``add_text``.

    ``ImDrawList::AddText`` has a ``(font, font_size, ...)`` overload in C++; pyimgui 2.0
    wraps only ``add_text(x, y, col, text)``, so the font size cannot be passed. What
    that overload does internally is scale the glyph quads it emits -- the atlas UVs are
    size-independent -- so emitting at the default size and scaling the quads in place is
    the *same operation*, not an approximation of it. It is also the idiom this repo
    already ships: ``axis.py::_draw_text_rotated`` rotates its glyphs through the same
    vertex buffer, and this borrows its ``_ImDrawVert`` layout rather than re-declaring a
    struct whose correctness is measured in scribbled-over memory.

    At ``scale == 1.0`` the ctypes path never runs at all, so the default legend cannot
    be broken by it. When it does run, the emitted quads are bounds-checked against
    ``calc_text_size`` before a byte is written -- a wrong struct layout or CPU-culled
    glyphs leave the text unscaled instead of corrupting the frame.
    """
    if abs(scale - 1.0) < 1e-3:
        draw_list.add_text(x, y, color, text)
        return

    from .axis import _ImDrawVert

    width, height = imgui.calc_text_size(text)
    first = draw_list.vtx_buffer_size
    draw_list.add_text(x, y, color, text)
    count = draw_list.vtx_buffer_size - first
    if count <= 0 or count % 4 != 0:
        return
    try:
        verts = (_ImDrawVert * (first + count)).from_address(int(draw_list.vtx_buffer_data))
        xs = [verts[i].x for i in range(first, first + count)]
        ys = [verts[i].y for i in range(first, first + count)]
    except (ValueError, TypeError, OSError):  # pragma: no cover - defensive
        return
    slack = 2.0
    if not (
        min(xs) >= x - slack
        and max(xs) <= x + width + slack
        and min(ys) >= y - slack
        and max(ys) <= y + height + slack
    ):
        return
    for i in range(first, first + count):
        verts[i].x = x + (verts[i].x - x) * scale
        verts[i].y = y + (verts[i].y - y) * scale


def _draw_swatch(imgui, draw_list, x: float, y: float, w: float, h: float, colors) -> None:
    """A filled rounded chip, or a left-to-right gradient when the entry has a range."""
    if not colors:
        return
    if len(colors) == 1:
        r, g, b, a = colors[0]
        draw_list.add_rect_filled(
            x, y, x + w, y + h, imgui.get_color_u32_rgba(r, g, b, max(a, 0.25)), 2.0
        )
        return
    c0 = imgui.get_color_u32_rgba(colors[0][0], colors[0][1], colors[0][2], max(colors[0][3], 0.25))
    c1 = imgui.get_color_u32_rgba(colors[1][0], colors[1][1], colors[1][2], max(colors[1][3], 0.25))
    # upper-left, upper-right, lower-right, lower-left
    draw_list.add_rect_filled_multicolor(x, y, x + w, y + h, c0, c1, c1, c0)


def draw_legend(plot: Any) -> Optional[Tuple[float, float, float, float]]:
    """Draw the legend overlay for ``plot``; return its screen rect, or None if it drew nothing.

    Called once per frame from the engine's overlay pass, after the scene and the axis
    labels. It early-outs before touching imgui when the legend is off, which is the
    default -- an unasked-for legend appearing on every existing plot would be a
    restyling, not a feature. ``gplt.legend()`` is what turns it on.

    Returning the rect is not decoration: it is the only way a test can assert *where*
    the box landed, and it lets a caller reason about the occupied region.
    """
    if not resolve_legend_show(getattr(plot, "options", None), plot):
        return None
    try:
        import imgui
    except (ImportError, Exception):  # noqa: B014 - imgui raises non-ImportError GL-less
        return None

    entries = build_entries(plot)
    if not entries:
        return None

    options = plot.options
    scale = float(getattr(options, "legend_font_scale", 1.0) or 1.0)
    scale = min(max(scale, 0.5), 4.0)
    line_h = imgui.get_text_line_height() * scale
    swatch_w, swatch_h = _SWATCH_W * scale, _SWATCH_H * scale
    pad, gap, row_gap = _PAD * scale, _GAP * scale, _ROW_GAP * scale

    text_w = max(imgui.calc_text_size(e.label)[0] * scale for e in entries)
    box_w = pad * 2 + swatch_w + gap + text_w
    box_h = pad * 2 + len(entries) * line_h + (len(entries) - 1) * row_gap

    ox, oy = _choose_location(plot, box_w, box_h)
    # Shift into the active panel's window rectangle: _choose_location works in panel-local
    # pixels, but the imgui draw list is window space. (0, 0) for a single full-window panel.
    panel_off = getattr(plot, "_panel_offset_px", (0.0, 0.0))
    ox += panel_off[0]
    oy += panel_off[1]

    is_light = _background_is_light(options)
    draw_list = imgui.get_background_draw_list()

    if getattr(options, "legend_background", True):
        fill = (
            imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.78)
            if is_light
            else imgui.get_color_u32_rgba(0.09, 0.09, 0.11, 0.78)
        )
        draw_list.add_rect_filled(ox, oy, ox + box_w, oy + box_h, fill, 4.0)
    if getattr(options, "legend_border", True):
        edge = (
            imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 0.55)
            if is_light
            else imgui.get_color_u32_rgba(0.85, 0.85, 0.85, 0.45)
        )
        draw_list.add_rect(ox, oy, ox + box_w, oy + box_h, edge, 4.0)

    text_color = (
        imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 1.0)
        if is_light
        else imgui.get_color_u32_rgba(0.85, 0.85, 0.85, 1.0)
    )

    y = oy + pad
    for entry in entries:
        if entry.colors:
            _draw_swatch(
                imgui,
                draw_list,
                ox + pad,
                y + (line_h - swatch_h) * 0.5,
                swatch_w,
                swatch_h,
                entry.colors,
            )
        _add_scaled_text(
            imgui, draw_list, ox + pad + swatch_w + gap, y, text_color, entry.label, scale
        )
        y += line_h + row_gap

    return (ox, oy, ox + box_w, oy + box_h)
