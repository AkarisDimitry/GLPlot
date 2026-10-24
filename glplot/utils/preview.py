from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

MAX_LEGEND_ITEMS = 5

#: Gallery/export chrome reads small at the default matplotlib rcParams sizes,
#: especially once a figure is downscaled for a README thumbnail. These apply to
#: every headless export uniformly rather than each caller guessing its own sizes.
#: Doubled from the original 20/15/12/12 pass per explicit user feedback that the
#: first bump still read small.
TITLE_FONTSIZE = 40
AXIS_LABEL_FONTSIZE = 30
TICK_LABEL_FONTSIZE = 24
LEGEND_FONTSIZE = 24

#: Area, in pt^2, of a scatter marker whose size nobody chose -- matplotlib's own default
#: with the same doubling the font sizes above get, and for the same reason: `savefig()`
#: renders at `figsize * scale` (scale=2) but sizes markers and text in *points*, so an
#: unscaled default covers a quarter of the relative area it would in a plain matplotlib
#: figure. Measured before this existed: 0.32x matplotlib's relative diameter, ~10% of its
#: area, which is what "the default points are tiny" was.
DEFAULT_SCATTER_SIZE_PT2 = (2.0 * 6.0) ** 2  # 6.0 == matplotlib's rcParams lines.markersize


def _point_diameter_px_to_area_pt2(size_px, engine: object = None):
    """Convert GLPlot's marker diameter in pixels to matplotlib's marker area in pt^2.

    The live 3D renderer reads ``scatter3d(s=...)`` as a screen-space pixel diameter,
    whereas matplotlib's ``Axes3D.scatter`` reads the same number as an area in points
    squared. Passing the number through unchanged therefore makes the two views diverge
    quadratically: a value such as 72 is a huge 72 px disc live but only an 8.5 pt marker
    in an exported GIF. Convert the units before squaring so both paths represent the same
    physical marker diameter at the figure's configured DPI.
    """
    figure_dpi = float(getattr(engine, "_figure_dpi", 100.0) or 100.0)
    diameter_pt = np.maximum(np.asarray(size_px, dtype=float), 0.5) * 72.0 / figure_dpi
    area_pt2 = diameter_pt * diameter_pt
    return float(area_pt2) if area_pt2.ndim == 0 else area_pt2


#: A colorbar's own chrome. Smaller than the panel-scale sizes above (the bar is a
#: narrow strip beside a full axis, not an axis itself) but nowhere near matplotlib's
#: unstyled ~10pt default, which is what these used to fall through to -- a bar's
#: numbers were rendering at a third the size of the tick numbers right next to them.
COLORBAR_LABEL_FONTSIZE = 22
COLORBAR_TICK_FONTSIZE = 18

#: An `inset=True` colorbar sits inside its host panel, over whatever was already
#: plotted there, in a strip a fraction of that panel's own size -- a notch smaller
#: again so the tick row and the axis label do not crowd each other inside it.
COLORBAR_INSET_LABEL_FONTSIZE = 19
COLORBAR_INSET_TICK_FONTSIZE = 16


def _legend_label(layer: object) -> Optional[str]:
    """The label to hand a matplotlib artist for this layer, or ``None`` for no legend entry.

    ``layer.label`` is not "no label" just because the caller never passed one: the engine
    stamps every unlabeled scatter/polyline/patch with a bookkeeping name ("Scatter 7") for
    the Scene panel's layer list, and marks that on ``metadata["auto_label"]`` at creation
    time. Without this check, `ax.get_legend_handles_labels()` below cannot tell that name
    apart from a real `label=` a caller asked to see in the legend, and every multi-layer
    plot exports with its legend flooded by "Patch 3", "Polyline 12", ... capped at
    ``MAX_LEGEND_ITEMS`` and hiding the labels that were actually requested.
    """
    if getattr(layer, "metadata", None) and layer.metadata.get("auto_label"):
        return None
    return layer.label or None


def _rgba(
    color: Optional[Union[Tuple[float, float, float, float], Sequence[float], np.ndarray]],
    default: Tuple[float, float, float, float] = (0, 0, 0, 1),
) -> Tuple[float, float, float, float]:
    if color is None:
        return default
    arr = np.asarray(color, dtype=float)
    if arr.shape == (4,):
        return tuple(arr)  # type: ignore
    if arr.shape == (3,):
        return tuple(np.r_[arr, 1.0])  # type: ignore
    return default


def _collapse_uniform_colors(colors: object) -> object:
    """One RGBA row when every row of ``colors`` is identical, otherwise ``colors`` untouched.

    A layer stores an ``(N, 4)`` colour array even when every point is the same colour, and
    handing that to ``ax.scatter(c=...)`` sends Agg down its per-element colour path: at
    500k points the export took ~1.8 s against ~0.08 s for the same figure with a single
    colour. The picture is the same either way, so an all-equal array is reduced to its
    first row, which is exactly the call matplotlib makes for ``scatter(x, y)``.
    """
    arr = np.asarray(colors) if colors is not None else None
    if arr is None or arr.ndim != 2 or len(arr) < 2:
        return colors
    if np.array_equal(arr, np.broadcast_to(arr[0], arr.shape)):
        return arr[:1]
    return colors


def _outline_rgba(style: object) -> Optional[Tuple[float, float, float, float]]:
    """``outline_color`` with ``outline_alpha`` folded in, or None when there is no outline.

    None is the answer for every layer that never touched the four ``outline_*`` fields,
    which is what keeps an untouched export byte-identical: the callers below then pass no
    outline keyword at all, rather than passing a transparent one.

    ``style.alpha`` is deliberately *not* folded in, matching the GL renderers and
    :attr:`glplot.core.layers.LayerStyle.outline_alpha`: an outline drawn on a translucent
    layer is there to keep it readable, so it must be able to stay opaque.
    """
    if not bool(getattr(style, "outline_enabled", False)):
        return None
    width = float(getattr(style, "outline_width", 0.0) or 0.0)
    if not np.isfinite(width) or width <= 0.0:
        return None
    r, g, b, a = _rgba(getattr(style, "outline_color", None), (0.0, 0.0, 0.0, 1.0))
    alpha = float(getattr(style, "outline_alpha", 1.0))
    return (r, g, b, float(np.clip(a * alpha, 0.0, 1.0)))


def _outline_path_effects(style: object, base_linewidth: float = 0.0) -> Optional[list]:
    """``[withStroke(...)]`` reproducing the layer's outline, or None when it has none.

    ``matplotlib.patheffects.withStroke`` draws a fattened copy of the artist's own path
    *behind* it and then the artist on top -- which is exactly the casing the GL renderers
    draw for a line, and exactly the silhouette they dilate for a filled patch. It is the
    one path effect that needs no knowledge of what the artist is, so a Line2D, a Polygon,
    a PolyCollection and a Text all take the same one.

    The stroke straddles the path, so ``base_linewidth + 2 * outline_width`` is what leaves
    ``outline_width`` points showing on the outside of a stroke of ``base_linewidth``.
    Points, not pixels: matplotlib has no device-pixel ratio, and the preview is a figure
    at its own DPI, so the number is carried across as-is rather than pretending to a pixel
    accuracy the two renderers cannot share.
    """
    color = _outline_rgba(style)
    if color is None:
        return None
    from matplotlib import patheffects

    width = float(getattr(style, "outline_width", 0.0) or 0.0)
    return [
        patheffects.withStroke(
            linewidth=float(base_linewidth) + 2.0 * width,
            foreground=color,
            alpha=color[3],
        )
    ]


def _outline_edge_kwargs(style: object) -> dict:
    """``edgecolors``/``linewidths`` for a marker collection; ``{}`` when there is no outline.

    A scatter's ring is a marker *edge* in matplotlib, not a path effect: ``ax.scatter``
    already draws one and only needs telling what colour and how thick. The empty dict for
    an untouched layer is what makes the historical call site unchanged -- ``**{}`` adds no
    keyword, so matplotlib's own default edge (``'face'``) is not overridden.

    The edge straddles the marker's path, so ``2 * outline_width`` leaves ``outline_width``
    showing outside it -- the closest matplotlib gets to the GL ring, which is drawn wholly
    outside the marker in a sprite grown for it. The marker's *fill* therefore ends up
    slightly smaller here than on screen; matplotlib has no way to grow a marker for its
    edge the way a point sprite can be grown.
    """
    color = _outline_rgba(style)
    if color is None:
        return {}
    width = float(getattr(style, "outline_width", 0.0) or 0.0)
    return {"edgecolors": [color], "linewidths": 2.0 * width}


def _apply_outline(artist: object, style: object, base_linewidth: float = 0.0) -> None:
    """Give ``artist`` the layer's outline, if it has one. A no-op otherwise.

    Set after construction rather than passed as a keyword so that an artist which does not
    accept ``path_effects`` in its constructor (``ax.quiver``'s ``Quiver`` is the awkward
    one) is handled the same way as the rest.
    """
    effects = _outline_path_effects(style, base_linewidth)
    if effects is not None and hasattr(artist, "set_path_effects"):
        artist.set_path_effects(effects)


def _apply_preview_ssao(colors: np.ndarray, z_values: np.ndarray, strength: float) -> np.ndarray:
    cols = np.asarray(colors, dtype=float).copy()
    if cols.ndim == 1:
        cols = np.tile(cols, (len(z_values), 1))
    z = np.asarray(z_values, dtype=float)
    zn = (z - np.nanmin(z)) / max(float(np.nanmax(z) - np.nanmin(z)), 1e-9)
    cavity = 1.0 - np.clip(zn, 0.0, 1.0)
    ao = np.clip(1.0 - float(strength) * 0.62 * cavity, 0.58, 1.0)
    cols[:, :3] *= ao[:, None]
    return cols


def _has_3d_layers(layers) -> bool:
    return any(
        layer.layer_type in {"scatter3d", "mesh3d", "wireframe3d", "bars3d", "volume3d"}
        for layer in layers
    )


#: zorder for 3D overlay lines (``style.depth_test=False``). mplot3d numbers its sorted
#: collections upward from just above the axis artists, so this only has to beat the
#: collection count of any plausible figure.
_OVERLAY_ZORDER = 1_000_000


#: Figure fraction a twin axis needs for its tick labels and title: they sit outside the axes
#: (right for ``twinx``, top for ``twiny``), where matplotlib's default margins leave ~10%.
_TWIN_ROOM_X = 0.07
_TWIN_ROOM_Y = 0.06


def _reserve_for_twins(panel: object, rect: Tuple[float, float, float, float]):
    """``rect`` (an axes box in figure fractions) pulled in to leave room for ``panel``'s twins.

    Capped at 30% of the box so a narrow cell of a dense grid keeps most of its width.
    """
    right, top = _twin_room(panel)
    x0, y0, w, h = rect
    if right:
        w -= min(_TWIN_ROOM_X, 0.3 * w)
    if top:
        h -= min(_TWIN_ROOM_Y, 0.3 * h)
    return (x0, y0, w, h)


def _twin_room(panel: object) -> Tuple[bool, bool]:
    """``(right, top)``: does ``panel`` have an axis whose labels sit outside on that side?

    A ``twinx`` and a ``secondary_yaxis("right")`` use the right; a ``twiny`` and a
    ``secondary_xaxis("top")`` the top. (A secondary on the bottom or left overlaps the main
    axis' own ticks in matplotlib too, so it asks for nothing.)
    """
    twins = getattr(panel, "twins", None) or {}
    secondaries = getattr(panel, "secondaries", None) or []
    right = "y" in twins or any(s.axis == "y" and s.location == "right" for s in secondaries)
    top = "x" in twins or any(s.axis == "x" and s.location == "top" for s in secondaries)
    return right, top


def _boundary_segments(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """The outline of a triangulated shape: ``(M, 2, 2)`` segments of its boundary edges.

    An edge shared by two triangles is interior; one that belongs to a single triangle is on
    the boundary. A fan-triangulated polygon or a ``fill_between`` strip stroked triangle by
    triangle shows every internal seam, so a shape's *edge* is drawn from these instead.
    """
    tri = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    pairs = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    undirected = np.sort(pairs, axis=1)
    _, first, counts = np.unique(undirected, axis=0, return_index=True, return_counts=True)
    boundary = pairs[first[counts == 1]]
    return np.asarray(vertices, dtype=float)[boundary]


def _hatch_style(layer: object, face: tuple, edge: tuple) -> dict:
    """matplotlib keywords that draw ``layer``'s hatch pattern, or ``{}`` for an unhatched one.

    matplotlib hatches in the patch's *edge* colour. GLPlot patches default to an edge equal to
    the face, which would hatch in the fill colour and show nothing, so an edge the caller did
    not choose becomes black -- matplotlib's own hatch colour -- and gets no outline of its own.
    """
    hatch = layer.metadata.get("hatch")
    if not hatch:
        return {}
    chosen = tuple(edge) != tuple(face)
    return {
        "hatch": hatch,
        "edgecolor": edge if chosen else (0.0, 0.0, 0.0, 1.0),
        "linewidth": layer.style.line_width if chosen else 0.0,
    }


def _raw_twin_layer(layer: object) -> object:
    """A copy of a twin-axis ``layer`` that carries the values the user plotted.

    The layer itself holds a copy mapped into the host viewport (``glplot.core.twin``) so the
    GL renderers need no idea twins exist. A real matplotlib twin axes has its own scale, so
    it is drawn from the originals.
    """
    import copy

    clone = copy.copy(layer)
    raw = layer.metadata.get("twin_raw")
    if raw is None:
        return clone
    if layer.layer_type in ("polyline", "scatter"):
        clone.pts = np.asarray(raw, dtype=np.float32)
    elif layer.layer_type == "patch":
        clone.vertices = np.asarray(raw, dtype=np.float32)
    elif layer.layer_type == "text":
        clone.x, clone.y = float(raw[0][0]), float(raw[0][1])
    return clone


def _draw_panel_layers(ax: object, panel: object, layers, has_3d: bool, engine: object) -> None:
    """Draw one panel: its own layers on ``ax``, and each twin's on a real twin axes.

    ``ax.twinx()`` / ``ax.twiny()`` share the other axis with ``ax``, exactly as in the
    script that built the figure. The twin axes are kept on ``ax._glplot_twins`` for the
    legend, which spans both.
    """
    twins = getattr(panel, "twins", None)
    secondaries = getattr(panel, "secondaries", None)
    if has_3d or not (twins or secondaries):
        _draw_layers(ax, layers, has_3d, engine)
        return
    from ..core import twin as _twin

    _draw_layers(ax, _twin.host_layers(layers), has_3d, engine)
    twin_axes = []
    for letter, state in twins.items():
        ax2 = ax.twinx() if letter == "y" else ax.twiny()
        mine = [_raw_twin_layer(layer) for layer in _twin.layers_of(layers, letter)]
        _draw_layers(ax2, mine, False, engine)
        _finish_twin_axes(ax2, letter, state)
        ax2._glplot_twin_state = state
        twin_axes.append(ax2)
    ax._glplot_twins = twin_axes
    ax._glplot_secondaries = []
    for state in secondaries or []:
        make = ax.secondary_xaxis if state.axis == "x" else ax.secondary_yaxis
        strip = make(state.location, functions=state.functions)
        _finish_secondary_axis(strip, state)
        strip._glplot_secondary_state = state
        ax._glplot_secondaries.append(strip)


def _finish_secondary_axis(strip: object, state: object) -> None:
    """A ``secondary_*axis``' pinned ticks, tick colour and title."""
    letter = state.axis
    if state.ticks is not None:
        (strip.set_xticks if letter == "x" else strip.set_yticks)(list(state.ticks))
        if state.tick_labels is not None:
            (strip.set_xticklabels if letter == "x" else strip.set_yticklabels)(
                list(state.tick_labels)
            )
    style = {"axis": letter, "labelsize": TICK_LABEL_FONTSIZE}
    if state.tick_color is not None:
        style["colors"] = state.tick_color
    strip.tick_params(**style)
    if state.label:
        label_style = {"fontsize": state.label_fontsize or AXIS_LABEL_FONTSIZE}
        if state.label_color is not None:
            label_style["color"] = state.label_color
        (strip.set_xlabel if letter == "x" else strip.set_ylabel)(state.label, **label_style)


def _finish_twin_axes(ax2: object, letter: str, state: object) -> None:
    """The twin axes' own scale, limits, ticks, tick style and title."""
    import copy

    set_scale = ax2.set_yscale if letter == "y" else ax2.set_xscale
    set_lim = ax2.set_ylim if letter == "y" else ax2.set_xlim
    set_ticks = ax2.set_yticks if letter == "y" else ax2.set_xticks
    set_ticklabels = ax2.set_yticklabels if letter == "y" else ax2.set_xticklabels
    axis = ax2.yaxis if letter == "y" else ax2.xaxis
    if state.scale != "linear":
        set_scale(state.scale, **(state.scale_params or {}))
    if state.lim is not None:
        lo, hi = state.lim
        set_lim((hi, lo) if state.inverted else (lo, hi))
    elif state.inverted:
        (ax2.invert_yaxis if letter == "y" else ax2.invert_xaxis)()
    if state.ticks is not None:
        set_ticks(list(state.ticks))
        if state.tick_labels is not None:
            set_ticklabels(list(state.tick_labels))
    if state.locator is not None:
        axis.set_major_locator(copy.copy(state.locator))
    if state.formatter is not None:
        axis.set_major_formatter(copy.copy(state.formatter))
    tick_style = {"axis": letter, "labelsize": TICK_LABEL_FONTSIZE}
    if state.tick_color is not None:
        tick_style["colors"] = state.tick_color
    ax2.tick_params(**tick_style)
    if state.label:
        label_style = {"fontsize": state.label_fontsize or AXIS_LABEL_FONTSIZE}
        if state.label_color is not None:
            label_style["color"] = state.label_color
        (ax2.set_ylabel if letter == "y" else ax2.set_xlabel)(state.label, **label_style)


def _draw_layers(ax: object, layers, has_3d: bool, engine: object = None) -> None:
    """Draw one panel's layers into ``ax``.

    A free function rather than inline in :func:`render_preview` so the multi-panel branch
    there can call it once per panel -- ``subplots(2, 2)`` and ``inset_axes()`` both add
    panels beyond the active one, and a saved file used to only ever contain that one.

    ``engine`` supplies the scene-wide SSAO defaults for the mesh3d/bars3d branch, the same
    way :func:`_finish_axes` reads the scene-wide axis labels off it. It is optional so a
    caller that only has layers still works: the lookups below degrade to the per-layer
    ``metadata`` values, which is what a layer carries when SSAO was set on the layer itself.
    """
    from matplotlib.patches import Polygon

    from ..core.camera3d import SYSTEM_3D_ARTISTS

    for layer in layers:
        if not layer.style.visible:
            continue
        # GLPlot's own 3D box, floor, grid and tick marks are chrome, and matplotlib's 3D
        # axes already draw all four natively. Exporting them would double the decoration,
        # list "3D grid" in the legend, and cost one ``ax.plot`` call per grid segment --
        # a 5-tick scene is ~90 of them, drawn on top of the axes matplotlib just made.
        if layer.metadata.get("artist") in SYSTEM_3D_ARTISTS:
            continue
        # contour()/contourf() draw a real, visible line/patch per level for the live GL
        # view (see their own docstrings) *in addition to* the invisible imshow placeholder
        # below, which is what this preview's own "artist == contour/contourf" branch
        # reconstructs headless through a fresh matplotlib call. Falling through to the
        # generic polyline/patch handling for the live layers too would draw every level
        # twice in the exported PNG.
        if layer.metadata.get("artist") in ("contour_line", "contourf_fill"):
            continue
        if layer.layer_type in {"scatter3d", "volume3d"} and layer.vertices is not None:
            verts = layer.vertices
            size_px = (
                layer.sizes
                if layer.sizes is not None and len(layer.sizes) == len(verts)
                else layer.style.point_size
            )
            # A marker's outline is its edge, which matplotlib takes as a keyword pair
            # rather than as a path effect. `_outline_edge_kwargs` is empty for a layer with
            # no outline, so the call is character-for-character the historical one.
            ax.scatter(
                verts[:, 0],
                verts[:, 1],
                verts[:, 2],
                c=_collapse_uniform_colors(layer.colors),
                s=_point_diameter_px_to_area_pt2(size_px, engine),
                depthshade=False,
                label=_legend_label(layer),
                **_outline_edge_kwargs(layer.style),
            )
            continue
        if layer.layer_type == "wireframe3d" and layer.vertices is not None:
            verts = layer.vertices.reshape(-1, 2, 3)
            color = _rgba(
                layer.colors[0]
                if layer.colors is not None and len(layer.colors)
                else layer.style.color
            )
            line_width = max(layer.style.line_width, 0.4)
            # style.depth_test=False is an overlay in the GL renderer. mplot3d re-numbers
            # every collection's zorder (surfaces, scatters) above the lines' default, so
            # a plain Line3D always lands under a surface; a zorder above anything it
            # assigns is what puts the line on top here too.
            overlay = not getattr(layer.style, "depth_test", True)
            for idx, seg in enumerate(verts):
                (line,) = ax.plot(
                    seg[:, 0],
                    seg[:, 1],
                    seg[:, 2],
                    color=color,
                    linewidth=line_width,
                    label=_legend_label(layer) if idx == 0 else None,
                )
                _apply_outline(line, layer.style, line_width)
                if overlay:
                    line.set_zorder(_OVERLAY_ZORDER)
            continue
        if layer.layer_type in {"mesh3d", "bars3d"} and layer.vertices is not None:
            verts = layer.vertices
            color = (
                layer.colors[:, :4]
                if layer.colors is not None and len(layer.colors) == len(verts)
                else None
            )
            ssao_opts = getattr(getattr(engine, "options", None), "visual", None)
            ssao_opts = getattr(ssao_opts, "ssao", None)
            ssao_on = bool(
                getattr(ssao_opts, "enabled", False) or layer.metadata.get("ssao", False)
            )
            ssao_strength = float(
                layer.metadata.get("ssao_strength", getattr(ssao_opts, "strength", 0.45))
            )
            if layer.indices is not None and len(layer.indices) >= 3:
                tris = np.asarray(layer.indices, dtype=int).reshape(-1, 3)
                face_color = color[tris[:, 0]] if color is not None else None
                if face_color is not None and ssao_on:
                    face_color = _apply_preview_ssao(
                        face_color, verts[tris[:, 0], 2], ssao_strength
                    )
                surf = ax.plot_trisurf(
                    verts[:, 0],
                    verts[:, 1],
                    verts[:, 2],
                    triangles=tris,
                    linewidth=0.0 if layer.layer_type == "bars3d" else 0.08,
                    antialiased=True,
                    shade=True,
                    color=(
                        None
                        if face_color is not None
                        else _rgba(layer.style.color, (0.1, 0.45, 1.0, 0.8))
                    ),
                    alpha=0.92,
                )
                if face_color is not None:
                    # A collection-wide alpha overrides every face's own, which silently
                    # threw away the caller's transparency: plot_surface(alpha=0.5) bakes
                    # its alpha into the vertex colours, and the 0.92 above turned it back
                    # into a near-opaque sheet hiding every point behind it. Keep 0.92 as a
                    # ceiling so an opaque mesh previews exactly as before, and let anything
                    # more transparent through, times style.alpha like the GL shader does.
                    face_color = np.array(face_color, dtype=float, copy=True)
                    face_color[:, 3] = np.minimum(face_color[:, 3] * float(layer.style.alpha), 0.92)
                    surf.set_alpha(None)
                    surf.set_facecolors(face_color)
                if layer.layer_type == "bars3d":
                    surf.set_edgecolor((0.0, 0.0, 0.0, 0.0))
                    surf.set_linewidth(0.0)
            else:
                ax.scatter(
                    verts[:, 0],
                    verts[:, 1],
                    verts[:, 2],
                    c=_collapse_uniform_colors(layer.colors),
                    s=layer.style.point_size,
                    depthshade=False,
                    label=_legend_label(layer),
                )
            continue
        # ``pcolormesh()`` itself now draws a per-vertex-coloured *patch* (``_pcolor_impl``) and
        # tags it with this same artist name but none of the X/Y/C arrays this branch was
        # written for, so it died here with ``KeyError: 'X'`` on every ``savefig``. Only a
        # layer that actually carries the arrays takes this path; the patch falls through to
        # the generic patch drawing below, which already honours per-vertex colours.
        if layer.metadata.get("artist") == "pcolormesh" and "X" in layer.metadata:
            ax.pcolormesh(
                layer.metadata["X"],
                layer.metadata["Y"],
                layer.metadata["C"],
                cmap=layer.metadata.get("cmap", "viridis"),
                shading=layer.metadata.get("shading", "auto"),
                vmin=layer.metadata.get("vmin"),
                vmax=layer.metadata.get("vmax"),
            )
            continue
        if layer.metadata.get("artist") == "contourf":
            cf_norm = layer.metadata.get("norm")
            ax.contourf(
                layer.metadata["X"],
                layer.metadata["Y"],
                layer.metadata["Z"],
                levels=layer.metadata.get("levels", 10),
                cmap=layer.metadata.get("cmap", "viridis"),
                norm=cf_norm,
                # A real Normalize instance and vmin/vmax are mutually exclusive
                # (matplotlib raises rather than picking a winner) -- only fall back to
                # vmin/vmax when there is no norm to conflict with.
                vmin=layer.metadata.get("vmin") if cf_norm is None else None,
                vmax=layer.metadata.get("vmax") if cf_norm is None else None,
                alpha=layer.metadata.get("alpha"),
            )
            continue
        if layer.metadata.get("artist") == "contour":
            contour_kwargs = {
                "levels": layer.metadata.get("levels", 10),
                "linewidths": layer.metadata.get("linewidths", 1.0),
            }
            if layer.metadata.get("colors") is not None:
                contour_kwargs["colors"] = layer.metadata.get("colors")
            else:
                contour_kwargs["cmap"] = layer.metadata.get("cmap", "viridis")
                c_norm = layer.metadata.get("norm")
                contour_kwargs["norm"] = c_norm
                if c_norm is None:
                    contour_kwargs["vmin"] = layer.metadata.get("vmin")
                    contour_kwargs["vmax"] = layer.metadata.get("vmax")
            contour_set = ax.contour(
                layer.metadata["X"], layer.metadata["Y"], layer.metadata["Z"], **contour_kwargs
            )
            # gplt.clabel(cs, ...) stashes its request on the layer itself (see that
            # function's docstring): GLPlot has no live contour geometry to break a line
            # against, so labelling only ever happens here, against the real ContourSet
            # this export just built.
            clabel_kwargs = layer.metadata.get("clabel")
            if clabel_kwargs is not None:
                ax.clabel(contour_set, **clabel_kwargs)
            continue
        if layer.metadata.get("artist") == "imshow":
            xmin, xmax, ymin, ymax = layer.metadata["extent"]
            im_matrix = layer.metadata["matrix"]
            # A true-color RGB(A) image carries its own per-pixel colour and has no
            # scalar norm/cmap to reconstruct -- matplotlib's own imshow() raises if
            # `norm=` is passed alongside (M, N, 3/4) data, same as pyplot.imshow()
            # silently ignores cmap/vmin/vmax/norm for it (see that docstring).
            im_is_rgb = np.asarray(im_matrix).ndim == 3
            im_norm = None if im_is_rgb else layer.metadata.get("norm")
            ax.imshow(
                im_matrix,
                extent=(xmin, xmax, ymin, ymax),
                origin=layer.metadata.get("origin", "upper"),
                cmap=layer.metadata.get("cmap", "viridis"),
                norm=im_norm,
                vmin=layer.metadata.get("vmin") if im_norm is None else None,
                vmax=layer.metadata.get("vmax") if im_norm is None else None,
                # Whatever `imshow(aspect=...)` actually resolved to (`pyplot.imshow`
                # defaults this to 'equal', matplotlib's own default) -- hardcoding
                # 'auto' here made every headless-exported image stretch to the
                # viewport regardless of what was requested or drawn live.
                aspect=layer.metadata.get("aspect", "equal"),
                interpolation="nearest",
            )
            continue
        if layer.layer_type == "polyline" and layer.pts is not None:
            if layer.metadata.get("artist_group") == "quiver":
                # Reconstruct (x, y, u, v) from the NaN-separated shaft polyline and
                # render with matplotlib's quiver for correct shaft + head appearance.
                # Layout: [start0, end0, NaN, start1, end1, NaN, ..., startN, endN]
                pts = layer.pts
                starts = pts[0::3]
                ends = pts[1::3]
                valid = ~np.any(np.isnan(starts), axis=1) & ~np.any(np.isnan(ends), axis=1)
                if valid.any():
                    xs, ys = starts[valid, 0], starts[valid, 1]
                    us = ends[valid, 0] - xs
                    vs = ends[valid, 1] - ys
                    col = _rgba(layer.style.color, (0, 0, 0, 1))
                    quiver = ax.quiver(
                        xs,
                        ys,
                        us,
                        vs,
                        angles="xy",
                        scale_units="xy",
                        scale=1,
                        color=[col],
                        width=max(layer.style.line_width * 0.002, 0.002),
                        headwidth=4,
                        headlength=4,
                        headaxislength=3.5,
                        label=_legend_label(layer),
                    )
                    # An arrow is a filled polygon here, not a stroke, so the casing has no
                    # line width of its own to add to.
                    _apply_outline(quiver, layer.style)
            else:
                pts = layer.pts
                (line,) = ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    color=_rgba(layer.style.color),
                    linewidth=layer.style.line_width,
                    label=_legend_label(layer),
                )
                _apply_outline(line, layer.style, layer.style.line_width)
        elif layer.layer_type == "patch" and layer.metadata.get("artist_group") == "quiver":
            pass  # arrowheads handled by ax.quiver() above — skip raw triangles
        elif layer.layer_type == "scatter" and layer.pts is not None:
            pts = layer.pts
            # A size nobody chose falls back to `DEFAULT_SCATTER_SIZE_PT2` rather than to
            # GLPlot's own default, which is a *pixel diameter*: handing that 10 straight
            # to `s` -- a pt^2 AREA -- drew a 3.2 pt marker where matplotlib draws 6 pt.
            # A size the caller *did* choose is still passed through untouched, so no
            # existing figure that tuned `s=` moves.
            explicit = layer.metadata.get("size_is_explicit", True)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=_collapse_uniform_colors(layer.colors),
                s=layer.style.point_size if explicit else DEFAULT_SCATTER_SIZE_PT2,
                # `marker=` was accepted by scatter() and stored on the layer, but never
                # actually reached this reconstruction -- every point exported as a
                # circle regardless of what was asked for. `None` here is the same "use
                # matplotlib's own default" as omitting the keyword entirely.
                marker=layer.metadata.get("marker") or None,
                label=_legend_label(layer),
                **_outline_edge_kwargs(layer.style),
            )
        elif layer.layer_type == "patch" and layer.vertices is not None:
            fc = _rgba(layer.style.face_color, (0.2, 0.4, 0.8, 0.35))
            ec = _rgba(layer.style.edge_color, (0.2, 0.4, 0.8, 1.0))
            if layer.metadata.get("artist") in ("bar", "barh") and len(layer.vertices) == 4:
                # A bar is one rectangle the live renderer happens to cut into two
                # triangles. Exported as those triangles, a bar whose edge differs from its
                # face shows the seam as a diagonal stroke (and a translucent one shows it
                # as a darker line), so it goes out as the single polygon it is.
                polygon_style = dict(edgecolor=ec, linewidth=layer.style.line_width)
                polygon_style.update(_hatch_style(layer, fc, ec))
                patch = Polygon(
                    np.asarray(layer.vertices, dtype=float),
                    closed=True,
                    facecolor=fc,
                    label=_legend_label(layer),
                    **polygon_style,
                )
                if not layer.metadata.get("edge_as_outline"):
                    _apply_outline(patch, layer.style, layer.style.line_width)
                ax.add_patch(patch)
            elif layer.indices is not None and getattr(layer, "mode", "strip") == "triangles":
                from matplotlib.collections import PolyCollection

                tris = np.asarray(layer.indices, dtype=int).reshape(-1, 3)
                polys = layer.vertices[tris]
                # A patch that carries per-vertex colours (a hexbin's hexagons, a
                # pcolor's cells, a tripcolor's faces) must export with those colours,
                # not one `face_color`. Without this the PNG comes out a single flat
                # blue while the live view is right -- the colormap silently dropped on
                # savefig. One facecolour per triangle, taken from its first vertex:
                # exact for the flat-shaded cases, and a faithful flat stand-in for a
                # gouraud tripcolor, which a PolyCollection cannot interpolate anyway.
                vertex_colors = getattr(layer, "colors", None)
                if vertex_colors is not None and len(vertex_colors) == len(layer.vertices):
                    face_colors = np.asarray(vertex_colors, dtype=float)[tris[:, 0]]
                    edge_colors = face_colors
                else:
                    face_colors = [fc]
                    edge_colors = [ec]
                collection_style = dict(edgecolors=edge_colors, linewidths=layer.style.line_width)
                hatched = _hatch_style(layer, fc, ec)
                if hatched:
                    collection_style = dict(
                        edgecolors=[hatched["edgecolor"]],
                        linewidths=hatched["linewidth"],
                        hatch=hatched["hatch"],
                    )
                # An edge the caller chose is the outline of the *shape*. Stroked per triangle
                # it draws every seam of the triangulation; the triangles are left unstroked
                # and the boundary edges drawn once, below.
                outline_segments = None
                # Only for an edge asked for through the API (``edgecolor=``/``linewidth=`` mark
                # the layer): a bare ``add_patch`` keeps stroking its triangles, as it always did.
                chosen_edge = (
                    bool(layer.metadata.get("edge_as_outline"))
                    and edge_colors[0] is ec
                    and layer.style.line_width > 0
                )
                if chosen_edge:
                    outline_segments = _boundary_segments(layer.vertices, tris)
                    collection_style["linewidths"] = 0.0
                    collection_style.setdefault("edgecolors", [ec])
                coll = PolyCollection(
                    polys,
                    facecolors=face_colors,
                    label=_legend_label(layer),
                    **collection_style,
                )
                if outline_segments is not None and len(outline_segments):
                    from matplotlib.collections import LineCollection

                    ax.add_collection(
                        LineCollection(
                            outline_segments,
                            colors=[ec],
                            linewidths=layer.style.line_width,
                            capstyle="round",
                            joinstyle="round",
                        )
                    )
                # A silhouette around each triangle, which is what the GL dilation gives a
                # triangle-mode patch too -- the seams between adjacent triangles are the
                # one place the two disagree, and only when the outline is on.
                if not layer.metadata.get("edge_as_outline"):
                    _apply_outline(coll, layer.style, layer.style.line_width)
                ax.add_collection(coll)
            else:
                outline = np.asarray(layer.vertices)
                if layer.indices is None and getattr(layer, "mode", "strip") == "strip":
                    # A strip stores its two sides interleaved (v0 v1 v2 ...), which the GL
                    # renderer draws as a ribbon. Handed to Polygon as-is that order is the
                    # boundary of a zigzag -- a violin came out as a comb. The boundary of
                    # a strip is one side forward and the other back.
                    outline = np.concatenate([outline[0::2], outline[1::2][::-1]])
                polygon_style = dict(edgecolor=ec, linewidth=layer.style.line_width)
                polygon_style.update(_hatch_style(layer, fc, ec))
                patch = Polygon(
                    outline,
                    closed=True,
                    facecolor=fc,
                    label=_legend_label(layer),
                    **polygon_style,
                )
                _apply_outline(patch, layer.style, layer.style.line_width)
                ax.add_patch(patch)
        elif layer.layer_type == "line_family" and layer.ab is not None:
            # No outline here on purpose: a family is previewed as a *density image*, not as
            # lines, so there is no path to case. The GL renderer refuses the outline above
            # OUTLINE_MAX_INSTANCES for related reasons; below it, the live view shows a
            # casing this preview cannot reproduce without drawing every line individually,
            # which is the very thing the density preview exists to avoid.
            x0, x1 = layer.x_range
            ab = np.asarray(layer.ab, dtype=np.float32)
            xs = np.linspace(float(x0), float(x1), 320, dtype=np.float32)
            y0 = ab[:, 0] * float(x0) + ab[:, 1]
            y1 = ab[:, 0] * float(x1) + ab[:, 1]
            y_bounds = np.concatenate([y0, y1])
            ymin_v, ymax_v = np.percentile(y_bounds, [0.2, 99.8])
            pad = max((float(ymax_v) - float(ymin_v)) * 0.04, 1e-3)
            ymin_v -= pad
            ymax_v += pad
            density = np.zeros((280, len(xs)), dtype=np.float32)
            y_edges = np.linspace(
                float(ymin_v), float(ymax_v), density.shape[0] + 1, dtype=np.float32
            )
            chunk_size = 8192
            for start in range(0, len(ab), chunk_size):
                chunk = ab[start : start + chunk_size]
                ys = chunk[:, 0, None] * xs[None, :] + chunk[:, 1, None]
                for col in range(len(xs)):
                    density[:, col] += np.histogram(ys[:, col], bins=y_edges)[0]
            ax.imshow(
                np.log1p(density),
                extent=(float(x0), float(x1), float(ymin_v), float(ymax_v)),
                origin="lower",
                aspect="auto",
                cmap=layer.metadata.get("cmap", "magma"),
                interpolation="bilinear",
            )
            ax.set_xlim(float(x0), float(x1))
            ax.set_ylim(float(ymin_v), float(ymax_v))
        elif layer.layer_type == "text":
            meta = layer.metadata
            # Alignment/rotation are set on the call, not read back from matplotlib's own
            # defaults: GLPlot's live text is left/baseline-aligned, and a script that said
            # `ha="center"` is asking for what only this export can draw.
            extras = {
                key: meta[name]
                for key, name in (("ha", "ha"), ("va", "va"), ("rotation", "rotation"))
                if meta.get(name) is not None
            }
            fraction = meta.get("axes_fraction")
            if fraction is not None:
                extras["transform"] = ax.transAxes
            text = ax.text(
                fraction[0] if fraction is not None else layer.x,
                fraction[1] if fraction is not None else layer.y,
                layer.text,
                fontsize=layer.style.text_size_px,
                color=_rgba(layer.style.color),
                bbox=layer.metadata.get("bbox"),
                **extras,
            )
            # The classic use of withStroke: a halo that keeps a label legible over data.
            # Glyphs have no line width, so the stroke is the outline width alone.
            _apply_outline(text, layer.style)


def _resolve_axes_title(panel: object, engine: object, allow_caption: bool = True) -> str:
    """The axes title for ``panel``: its own, else the window caption if a real one.

    ``engine.title`` doubles as the GLFW window caption and defaults to "GLPlot", so
    falling straight through to it stamped that literal string as the axes title of
    every untitled figure -- which is exactly what the live renderer already refuses to
    do (``renderers/axis.py::_resolve_title``, same ``STOCK_WINDOW_TITLES`` test). The
    export disagreeing with the live window about this was the whole bug.

    ``allow_caption`` is the *other* half of it. ``set_title`` writes both the caption
    and the active panel's own title, so on a split figure the caption holds whichever
    panel was titled last -- letting every panel fall back to it stamps that one title
    across all of them. Only the primary panel may fall back (that is the single-panel
    case, where the caption genuinely is the figure's one title); the rest show their
    own title or none.
    """
    from ..options import STOCK_WINDOW_TITLES

    candidate = str(getattr(panel, "title", "") or "")
    if not candidate:
        if not allow_caption:
            return ""
        candidate = str(getattr(engine, "title", "") or "")
    # Filter whichever source won, not just the caption: `set_title` copies its argument
    # onto the active panel as well, so testing only `engine.title` let a stock caption
    # reach the plot through the panel instead. Same rule as the live renderer's
    # `renderers/axis.py::_resolve_title`.
    return "" if candidate in STOCK_WINDOW_TITLES else candidate


def _text_style_kwargs(
    opts: object, fontsize_attr: str, color_attr: str, default_fontsize: float
) -> dict:
    """``fontsize=``/``color=`` kwargs for one text element, matching the live-window default.

    ``color`` is only included when set: matplotlib's ``Text.set_color(None)`` raises, so
    an unset option must leave the keyword out entirely and fall back to matplotlib's own
    default (black), which already matches the live renderer's default light-on-dark ink on
    the near-universal white savefig background.
    """
    kwargs = {"fontsize": getattr(opts, fontsize_attr, None) or default_fontsize}
    color = getattr(opts, color_attr, None)
    if color is not None:
        kwargs["color"] = color
    return kwargs


def _view_is_pinned(engine: object, panel: object) -> bool:
    """Did the caller set ``panel``'s view on purpose (``set_xlim``/``set_ylim``/``autoscale``)?

    A pinned view is *deliberate*: do not re-fit it to whichever data this particular
    frame happens to hold. ``Panel.view_pinned`` answers per panel. The engine-wide
    ``_needs_initial_autoscale`` flag is only consulted when there is one panel (or none to
    ask), because with several it cannot say whose view it was -- trusting it made
    ``axs[0].set_ylim(...)`` pin whichever panel happened to be active at save time and
    leave the one that was actually configured to autoscale.
    """
    if panel is not None and getattr(panel, "view_pinned", False):
        return True
    if panel is not None and len(getattr(engine, "panels", None) or ()) > 1:
        return False
    return not getattr(engine, "_needs_initial_autoscale", True)


@contextmanager
def _panel_active(engine: object, panel: object):
    """Make ``panel`` the engine's active panel for the block, then put the old one back."""
    panels = getattr(engine, "panels", None)
    if panel is None or not panels or panel not in panels:
        yield
        return
    previous = engine.active_panel_index
    engine.active_panel_index = panels.index(panel)
    try:
        yield
    finally:
        engine.active_panel_index = previous


def _apply_view(ax: object, engine: object, panel: object = None) -> bool:
    """Give ``ax`` the view ``panel`` has: its pinned limits, else matplotlib's autoscale.

    Returns whether the view was pinned. Direction comes along either way: pinned limits
    arrive as ``(hi, lo)`` when the axis is inverted, which matplotlib reads as an inverted
    axis by itself; an autoscaled axis is inverted explicitly from the camera's flip flags.
    """
    pinned = False
    if _view_is_pinned(engine, panel):
        get_xlim = getattr(engine, "get_xlim", None)
        get_ylim = getattr(engine, "get_ylim", None)
        if callable(get_xlim) and callable(get_ylim):
            with _panel_active(engine, panel):
                xlim, ylim = get_xlim(), get_ylim()
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            pinned = True
    if not pinned:
        ax.autoscale(enable=True)
        camera = getattr(panel, "camera", None)
        if getattr(camera, "flip_x", False) and not ax.xaxis_inverted():
            ax.invert_xaxis()
        if getattr(camera, "flip_y", False) and not ax.yaxis_inverted():
            ax.invert_yaxis()
    if getattr(panel, "aspect_equal", False):
        # `datalim` widens whichever axis is short -- the same thing GLPlot's own equal
        # aspect does to the camera -- so nothing the caller had in view is cropped. The
        # live view equalised for the window's pixel shape; this saved figure has its own.
        ax.set_aspect("equal", adjustable="datalim")
    return pinned


def _spaced_rect(panel: object, panels: Sequence[object], wspace: float, hspace: float):
    """``panel.rect_frac`` pulled in along every edge it shares with another grid panel.

    A panel's rect carries its tick labels and title *inside* it, as a gutter (see the
    note in :func:`_build_preview_figure`), so grid cells can sit flush. A matplotlib Axes
    draws that chrome *outside* its box, so flush cells left the right panel's tick labels
    printed over the left panel's. matplotlib's own answer is ``wspace``/``hspace``: a gap
    of that fraction of the (average) axes size. For a cell of width ``c`` the box is
    ``c / (1 + wspace)`` wide, so each side that touches a neighbour gives up half the gap,
    ``c * wspace / (2 * (1 + wspace))``, and the neighbour gives up the other half.

    Only panels cut from a grid take part: an ``inset_axes`` or an ``ax.set_position()``
    rect (``Panel.explicit_rect``) is exactly where it was asked to be. An edge with no
    neighbour -- the figure border, or a colorbar strip -- is left alone.
    """
    x0, y0, w, h = panel.rect_frac
    if getattr(panel, "explicit_rect", False):
        return (x0, y0, w, h)
    gap_x = 0.5 * wspace / (1.0 + wspace) * w
    gap_y = 0.5 * hspace / (1.0 + hspace) * h
    eps = 1e-6
    left, right, bottom, top = x0, x0 + w, y0, y0 + h
    for other in panels:
        if other is panel or getattr(other, "explicit_rect", False):
            continue
        ox0, oy0, ow, oh = other.rect_frac
        overlaps_y = min(top, oy0 + oh) - max(bottom, oy0) > eps
        overlaps_x = min(right, ox0 + ow) - max(left, ox0) > eps
        if overlaps_y and abs(ox0 - (x0 + w)) < eps:
            right -= gap_x
        if overlaps_y and abs((ox0 + ow) - x0) < eps:
            left += gap_x
        if overlaps_x and abs(oy0 - (y0 + h)) < eps:
            top -= gap_y
        if overlaps_x and abs((oy0 + oh) - y0) < eps:
            bottom += gap_y
    return (left, bottom, right - left, top - bottom)


def _apply_axis_ticker(ax: object, opts: object) -> None:
    """Install the matplotlib ``Locator``/``Formatter`` set through ``ax.xaxis``/``ax.yaxis``.

    Copies, not the objects themselves: matplotlib binds a ticker to the one axis it is
    given, and one set of options feeds every panel of the export.
    """
    import copy

    for axis in ("x", "y"):
        target = ax.xaxis if axis == "x" else ax.yaxis
        locator = getattr(opts, f"axis_tick_locator_{axis}", None)
        formatter = getattr(opts, f"axis_tick_formatter_{axis}", None)
        if locator is not None:
            target.set_major_locator(copy.copy(locator))
        if formatter is not None:
            target.set_major_formatter(copy.copy(formatter))


def _apply_tick_rotation(ax: object, opts: object) -> None:
    """Rotate ``ax``'s tick labels as ``xticks(rotation=...)`` / ``tick_params`` asked.

    A rotated label is anchored by its *end* in matplotlib's own usage (``ha="right"``);
    centred, a 45-degree label sits under the wrong tick. Right-aligning matches what
    nearly every script that rotates ticks then does by hand.
    """
    for axis in ("x", "y"):
        degrees = float(getattr(opts, f"axis_tick_rotation_{axis}", 0.0) or 0.0)
        if degrees == 0.0:
            continue
        ax.tick_params(axis=axis, labelrotation=degrees)
        if axis == "x" and 0.0 < abs(degrees) < 90.0:
            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right" if degrees > 0 else "left")
                label.set_rotation_mode("anchor")


def _apply_panel_labels(ax: object, panel: object, engine: object) -> None:
    """Apply ``panel``'s own axis names and title -- the per-panel half of the chrome.

    Split out of :func:`_finish_axes` (which stays primary-only, because the legend and
    the axis *scales* it also applies are genuinely figure-global) so that every panel
    of a split figure gets named, not just whichever one happened to be active when
    ``savefig()`` ran. See :attr:`glplot.core.panel.Panel.xlabel`.
    """
    opts = getattr(engine, "options", None)
    if getattr(panel, "xlabel", ""):
        ax.set_xlabel(
            panel.xlabel,
            **_text_style_kwargs(
                opts, "axis_xlabel_fontsize", "axis_xlabel_color", AXIS_LABEL_FONTSIZE
            ),
        )
    if getattr(panel, "ylabel", ""):
        ax.set_ylabel(
            panel.ylabel,
            **_text_style_kwargs(
                opts, "axis_ylabel_fontsize", "axis_ylabel_color", AXIS_LABEL_FONTSIZE
            ),
        )
    title = _resolve_axes_title(panel, engine, allow_caption=False)
    if title:
        ax.set_title(
            title,
            **_text_style_kwargs(opts, "axis_title_fontsize", "axis_title_color", TITLE_FONTSIZE),
        )


def _finish_axes(ax: object, has_3d: bool, engine: object, panel: object = None) -> None:
    """Apply the figure-global chrome (title, labels, legend, scale, ticks) to ``ax``.

    These read off ``engine`` directly (``engine.title``, ``engine.xlabel``, ...) rather
    than off any one panel: GLPlot's per-panel state is the scene/camera/interaction, not
    the chrome, so a multi-panel figure still has exactly one shared title/label/legend.
    Called once, for whichever panel is ``engine.active_panel`` -- the historical
    single-scene behaviour, preserved rather than guessed at for every extra panel.
    """
    # `title()`/`xlabel()`/`ylabel()` all accept an explicit `fontsize=` and store it on
    # `engine.options.axis_*_fontsize` (see `_set_title`/`xlabel()`/`ylabel()` in
    # pyplot.py) -- honoured here when given, rather than always overwritten by the
    # panel-scale constant below. A caller's explicit size being silently dropped in the
    # one place it actually renders (the headless export) is exactly the "keyword
    # accepted, quietly ignored" failure this codebase's own compat tests exist to catch.
    opts = getattr(engine, "options", None)
    title_kwargs = _text_style_kwargs(
        opts, "axis_title_fontsize", "axis_title_color", TITLE_FONTSIZE
    )
    xlabel_kwargs = _text_style_kwargs(
        opts, "axis_xlabel_fontsize", "axis_xlabel_color", AXIS_LABEL_FONTSIZE
    )
    ylabel_kwargs = _text_style_kwargs(
        opts, "axis_ylabel_fontsize", "axis_ylabel_color", AXIS_LABEL_FONTSIZE
    )
    tick_kwargs = {"labelsize": getattr(opts, "axis_tick_fontsize", None) or TICK_LABEL_FONTSIZE}
    tick_color = getattr(opts, "axis_tick_color", None)
    if tick_color is not None:
        tick_kwargs["labelcolor"] = tick_color

    if getattr(engine, "grid_visible", False):
        ax.grid(True, alpha=0.25)
    if hasattr(engine, "xlabel"):
        ax.set_xlabel(engine.xlabel, **xlabel_kwargs)
    if hasattr(engine, "ylabel"):
        ax.set_ylabel(engine.ylabel, **ylabel_kwargs)
    if has_3d:
        # The axis titles set through the 3D panel take precedence over the engine's 2D
        # ones: a 3D scene's labels live on ``axes3d``, and falling straight through to
        # ``engine.zlabel`` used to export a literal "z" over a named axis.
        axes3d_opts = getattr(engine, "axes3d", None)
        show_axes3d = getattr(axes3d_opts, "show_axes", True)
        if show_axes3d:
            if getattr(axes3d_opts, "xlabel", ""):
                ax.set_xlabel(axes3d_opts.xlabel, **xlabel_kwargs)
            if getattr(axes3d_opts, "ylabel", ""):
                ax.set_ylabel(axes3d_opts.ylabel, **ylabel_kwargs)
            ax.set_zlabel(
                getattr(axes3d_opts, "zlabel", "") or getattr(engine, "zlabel", "z") or "z",
                fontsize=AXIS_LABEL_FONTSIZE,
            )
        view3d = getattr(engine, "view3d", {})
        ax.view_init(elev=float(view3d.get("elev", 28.0)), azim=float(view3d.get("azim", -45.0)))
        camera = getattr(engine, "camera3d", None)
        if camera is not None and camera.projection == "orthographic":
            ax.set_proj_type("ortho")
        if show_axes3d:
            ax.tick_params(**tick_kwargs)
        else:
            # set_axis3d_off()/axis('off') on a 3D scene: matplotlib's Axes3D.set_axis_off()
            # drops the box, floor panes, grid and tick labels together -- the GL engine's
            # own axes3d.show_axes flag was only ever wired into the live renderer, so a
            # headless savefig() silently ignored it and always drew the full 3D chrome.
            ax.set_axis_off()
    else:
        ax.tick_params(axis="both", **tick_kwargs)
    resolved_title = _resolve_axes_title(panel, engine)
    if resolved_title:
        ax.set_title(resolved_title, **title_kwargs)
    _apply_legend(ax)
    if has_3d:
        ax.autoscale(enable=True)
        # An explicit set_xlim3d()/set_ylim3d()/set_zlim3d() call (stored on
        # axes3d.xlim/ylim/zlim -- see glplot/pyplot.py's _set_axis3d_limit) used to be
        # silently discarded here: ax.autoscale(enable=True) ran unconditionally on
        # every export, so those calls had no effect on the saved file. Most visible in
        # an animation, where the box visibly resized every frame as autoscale re-fit to
        # whatever that frame's data happened to span -- which is why 3D animation
        # scripts have had to fake a pinned view with invisible box-corner anchor points
        # instead of just calling the real, documented API. Applied per-axis, after
        # autoscale, so only the axes the caller actually pinned override the auto-fit
        # ones -- an unset axis still autoscales normally.
        axes3d_opts = getattr(engine, "axes3d", None)
        xlim3d = getattr(axes3d_opts, "xlim", None)
        if xlim3d is not None:
            ax.set_xlim3d(*xlim3d)
        ylim3d = getattr(axes3d_opts, "ylim", None)
        if ylim3d is not None:
            ax.set_ylim3d(*ylim3d)
        zlim3d = getattr(axes3d_opts, "zlim", None)
        if zlim3d is not None:
            ax.set_zlim3d(*zlim3d)
    else:
        _apply_axis_state(ax, engine, panel)


def _apply_legend(ax: object) -> None:
    """Draw ``ax``'s legend from its labelled artists: deduplicated, capped, top right.

    A panel with twin axes gets one legend over all of them -- the combined-handles legend
    that matplotlib scripts otherwise build by hand.
    """
    handles, labels = ax.get_legend_handles_labels()
    for twin_ax in getattr(ax, "_glplot_twins", ()):
        twin_handles, twin_labels = twin_ax.get_legend_handles_labels()
        handles, labels = handles + twin_handles, labels + twin_labels
    if not labels:
        return
    unique = []
    seen = set()
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        unique.append((handle, label))
    hidden = max(0, len(unique) - MAX_LEGEND_ITEMS)
    visible = unique[:MAX_LEGEND_ITEMS]
    legend_handles = [item[0] for item in visible]
    legend_labels = [item[1] for item in visible]
    if hidden:
        from matplotlib.lines import Line2D

        legend_handles.append(Line2D([], [], color="none"))
        legend_labels.append(f"+{hidden} more layers")
    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        framealpha=0.78,
        borderpad=0.4,
        labelspacing=0.4,
        handlelength=1.6,
    )


def _apply_axis_state(ax: object, engine: object, panel: object = None) -> None:
    """Give a 2D ``ax`` the scale, view, ticks and tick styling ``panel`` has.

    Everything here is *per axes* in matplotlib, and reads the per-axes settings of whichever
    panel is active on ``engine`` (see ``GPULinePlot.active_panel_index``), so callers run it
    under :func:`_panel_active`. It used to be written inline in :func:`_finish_axes`, which
    only ever ran for the primary panel -- so ``axs[1].set_yscale("log")`` exported as linear.
    """
    opts = getattr(engine, "options", None)
    scale_x = getattr(opts, "axis_scale_x", "linear")
    if scale_x and scale_x != "linear":
        ax.set_xscale(scale_x, **(getattr(opts, "axis_scale_params_x", None) or {}))
    scale_y = getattr(opts, "axis_scale_y", "linear")
    if scale_y and scale_y != "linear":
        ax.set_yscale(scale_y, **(getattr(opts, "axis_scale_params_y", None) or {}))
    _apply_view(ax, engine, panel)
    # A pinned `xticks()`/`yticks()` call (explicit positions, or -- via `_coerce_axis_values`
    # -- a categorical axis's string labels) is stored on the options, not on any layer, so
    # nothing else reconstructs it. `set_xticks`/`set_yticks` run after `autoscale()` because
    # they install a `FixedLocator`, which -- unlike the default locator -- survives it.
    x_values = getattr(opts, "axis_tick_values_x", None)
    if x_values:
        ax.set_xticks(x_values)
        x_labels = getattr(opts, "axis_tick_labels_x", None)
        if x_labels:
            ax.set_xticklabels(x_labels)
    y_values = getattr(opts, "axis_tick_values_y", None)
    if y_values:
        ax.set_yticks(y_values)
        y_labels = getattr(opts, "axis_tick_labels_y", None)
        if y_labels:
            ax.set_yticklabels(y_labels)
    # Last, so a formatter/locator the caller installed wins over the pinned ticks above (as
    # it does in matplotlib, where the last call wins) and the rotation lands on the final
    # tick labels rather than ones about to be replaced.
    _apply_axis_ticker(ax, opts)
    _apply_tick_rotation(ax, opts)
    # ``axis("off")``: matplotlib hides the frame, ticks, tick labels, grid and axis titles
    # together; the title and the artists stay. Used to be dropped by every 2D export.
    if getattr(panel, "axis_off", False):
        ax.set_axis_off()


def _apply_style_chrome(fig: object, ax: object, has_3d: bool, engine: object) -> None:
    """Paint the export with the current style's background, plus contrast-matched ink.

    Every layer is reconstructed through matplotlib here, but until now the figure was
    always plain white no matter what background a style preset (or a bare
    ``plt.plot_style(...)`` call) had set on the live scene -- "dark"/"neon"/"chalk"/
    "blueprint" looked identical to "clean" in every exported PNG/GIF, because nothing read
    ``VisualOptions.background_color``, the single source of truth the live GL renderer
    paints with. This reads that same value and derives readable text/spine/legend colour
    via simple relative-luminance thresholding -- the headless counterpart to the
    ``AUTO_GRID_COLOR`` sentinel ``renderers/axis.py`` already resolves against background
    luminance live.
    """
    opts = getattr(engine, "options", None)
    visual = getattr(opts, "visual", None)
    bg = getattr(visual, "background_color", None)
    if bg is None:
        return
    bg = tuple(float(c) for c in bg[:3])
    if bg == (1.0, 1.0, 1.0):
        return  # the default -- leave the historical white export untouched
    luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    ink = (0.95, 0.96, 0.98) if luminance < 0.5 else (0.05, 0.05, 0.07)

    fig.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.tick_params(colors=ink)
    ax.xaxis.label.set_color(ink)
    ax.yaxis.label.set_color(ink)
    ax.title.set_color(ink)
    for spine in ax.spines.values():
        spine.set_color(ink)
    if has_3d:
        ax.zaxis.label.set_color(ink)
        for axis3d in (ax.xaxis, ax.yaxis, ax.zaxis):
            set_pane_color = getattr(axis3d, "set_pane_color", None)
            if set_pane_color is not None:
                set_pane_color((*bg, 1.0))
    for strip in getattr(ax, "_glplot_secondaries", ()):
        state = getattr(strip, "_glplot_secondary_state", None)
        if getattr(state, "tick_color", None) is None:
            strip.tick_params(colors=ink)
        if getattr(state, "label_color", None) is None:
            strip.xaxis.label.set_color(ink)
            strip.yaxis.label.set_color(ink)
    for twin_ax in getattr(ax, "_glplot_twins", ()):
        # A colour the caller gave the twin (``tick_params(colors=...)``) stays theirs.
        state = getattr(twin_ax, "_glplot_twin_state", None)
        if getattr(state, "tick_color", None) is None:
            twin_ax.tick_params(colors=ink)
        if getattr(state, "label_color", None) is None:
            twin_ax.yaxis.label.set_color(ink)
            twin_ax.xaxis.label.set_color(ink)
    legend = ax.get_legend()
    if legend is not None:
        frame = legend.get_frame()
        frame.set_facecolor(bg)
        frame.set_edgecolor(ink)
        for text in legend.get_texts():
            text.set_color(ink)


#: An absolute floor, not tied to AXIS_LABEL_FONTSIZE -- a title that's still too wide
#: at 30pt (a long sentence on a modest figsize) needs to keep shrinking past it. This is
#: still comfortably above the ~12pt matplotlib's own unstyled default title used to be.
TITLE_MIN_FONTSIZE = 14


def _autosize_title(fig: object, ax: object, min_fontsize: float = TITLE_MIN_FONTSIZE) -> None:
    """Shrink an overlong title just enough to stop it clipping the figure's own edges.

    ``TITLE_FONTSIZE`` is deliberately large (readable in a README thumbnail), but a
    descriptive one-sentence title at that size can be wider than a modest ``figsize``
    figure -- it was clipping clean off both edges before this existed. Measuring against
    the real rendered glyph width (not a character-count guess) and backing off in small
    steps keeps every title as large as it can be while still fitting, rather than either
    picking one fixed smaller size for everyone or leaving the largest titles cut off.
    """
    title = ax.title
    if not title.get_text():
        return
    canvas = getattr(fig, "canvas", None)
    if canvas is None or not hasattr(canvas, "get_renderer"):
        return
    fig.canvas.draw()
    fig_width_px = fig.get_size_inches()[0] * fig.dpi
    margin_px = fig_width_px * 0.015
    # The title is centered on the *axes* (matplotlib's default title x=0.5 is in axes,
    # not figure, coordinates), and the axes itself is usually shifted right of the
    # figure's own center to make room for the y-axis tick labels/ylabel -- so a title
    # only slightly narrower than the full figure can still clip the right edge even
    # though its total width alone looked like it fit. Comparing the actual left/right
    # edges to the figure's own bounds (not just total width against a budget) is what
    # catches that case.
    for _ in range(20):
        renderer = fig.canvas.get_renderer()
        bbox = title.get_window_extent(renderer=renderer)
        if (
            bbox.x0 >= margin_px and bbox.x1 <= fig_width_px - margin_px
        ) or title.get_fontsize() <= min_fontsize:
            break
        title.set_fontsize(max(title.get_fontsize() * 0.9, min_fontsize))
        fig.canvas.draw()


def _reapply_pinned_limits(ax: object, has_3d: bool, engine: object) -> None:
    """Re-apply an explicit xlim/ylim/3D-limit pin, if one is in effect, one last time.

    Called after :func:`_autosize_title`'s own internal redraws, which can silently
    regrow a 3D axes' pinned range back toward the live data's extent (see the call
    site's comment) -- cheap enough to always call, and a no-op when nothing is pinned.
    """
    if has_3d:
        axes3d_opts = getattr(engine, "axes3d", None)
        xlim3d = getattr(axes3d_opts, "xlim", None)
        if xlim3d is not None:
            ax.set_xlim3d(*xlim3d)
        ylim3d = getattr(axes3d_opts, "ylim", None)
        if ylim3d is not None:
            ax.set_ylim3d(*ylim3d)
        zlim3d = getattr(axes3d_opts, "zlim", None)
        if zlim3d is not None:
            ax.set_zlim3d(*zlim3d)
    elif _view_is_pinned(engine, getattr(engine, "active_panel", None)):
        get_xlim = getattr(engine, "get_xlim", None)
        get_ylim = getattr(engine, "get_ylim", None)
        if callable(get_xlim) and callable(get_ylim):
            ax.set_xlim(*get_xlim())
            ax.set_ylim(*get_ylim())


def _build_preview_figure(engine: object, scale: float = 1.0):
    """Build (but do not save) the matplotlib ``Figure`` a preview export draws into.

    Shared by :func:`render_preview` (writes it to a file) and :func:`render_preview_array`
    (reads its pixels back directly) so the two only differ in how a finished figure turns
    into output, not in how the figure itself is built -- one panel loop, one set of
    layer/label/colorbar/title rules, instead of two copies that could drift apart.

    A single panel renders with matplotlib's own auto margins, unchanged from before panels
    existed. Two or more panels -- a ``subplots()`` grid, ``subplot2grid()``, or
    ``inset_axes()`` -- each get their own :class:`~matplotlib.axes.Axes` at their own
    ``rect_frac`` (already the exact bottom-left-origin figure-fraction format
    ``Figure.add_axes`` expects), so the saved file matches what the live GL window shows
    instead of only whichever panel happened to be active when ``savefig()`` was called.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.cm as mcm
    import matplotlib.pyplot as mpl

    # `engine.width`/`engine.height` are pixels; recovering the figure's physical size in
    # inches means dividing by whatever dpi actually produced them. `figure()` stamps that
    # onto `_figure_dpi` -- a hardcoded /100.0 here (the historical version) silently gave
    # the wrong physical size whenever a caller passed `figure(figsize=..., dpi=200)`, since
    # `width` had already been computed as `figsize * 200`.
    figure_dpi = float(getattr(engine, "_figure_dpi", 100.0)) or 100.0
    width = max(engine.width / figure_dpi, 4.0)
    height = max(engine.height / figure_dpi, 3.0)

    panels = list(getattr(engine, "panels", None) or [])
    primary = getattr(engine, "active_panel", None)
    single_3d_ax = None
    title_ax = None

    def _colorbar_mappable(cb):
        # `norm`/`cmap` are the same objects `colorbar()` resolved live (see
        # `ColorbarSpec`), so the headless bar cannot independently drift from what the
        # live GL window shows.
        return mcm.ScalarMappable(norm=cb.norm, cmap=cb.cmap)

    # An inset bar has no room outside the panel for its ticks (that is the whole reason
    # to inset it) -- point them inward instead. Same mapping the live renderer uses
    # (`renderers/colorbar.py`), kept as a local copy rather than an import so this module
    # stays decoupled from the live-renderer internals it otherwise never touches.
    _opposite_side = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}

    def _colorbar_location(cb):
        return _opposite_side[cb.location] if cb.inset else cb.location

    def _finalize_colorbar(cb_obj, cb):
        """Size (and, for an inset bar, outline) a finished colorbar's own chrome.

        Applied after the real ``Colorbar`` exists rather than through ``fig.colorbar``'s
        own ``label=``, which offers no size control and left every bar's numbers at
        matplotlib's unstyled default next to panel ticks four times their size.

        An inset bar sits *over* the plotted content, so its numbers get a thin
        contrasting stroke instead of the opaque backing panel they used to be given --
        the box hid a rectangle of the very image the bar is describing, which is a
        strange thing for a colorbar to do. ``withStroke`` is the same path effect
        ``_apply_outline`` already uses to keep a data label legible over data.
        """
        tick_size = COLORBAR_INSET_TICK_FONTSIZE if cb.inset else COLORBAR_TICK_FONTSIZE
        label_size = COLORBAR_INSET_LABEL_FONTSIZE if cb.inset else COLORBAR_LABEL_FONTSIZE
        cb_obj.ax.tick_params(labelsize=tick_size)
        if cb.label:
            cb_obj.set_label(cb.label, fontsize=label_size)
        if not cb.inset:
            return

        import matplotlib.patheffects as path_effects

        stroke = [path_effects.withStroke(linewidth=2.4, foreground="white")]
        for text in list(cb_obj.ax.get_xticklabels()) + list(cb_obj.ax.get_yticklabels()):
            text.set_path_effects(stroke)
        for axis_label in (cb_obj.ax.xaxis.label, cb_obj.ax.yaxis.label):
            if axis_label.get_text():
                axis_label.set_path_effects(stroke)

    if len(panels) > 1:
        fig = mpl.figure(figsize=(width * scale, height * scale), dpi=120)
        # Every panel's `rect_frac` reserves zero *outer* margin by design (see
        # glplot/core/layout.py's `DEFAULT_OUTER_MARGIN`): the live GL renderer draws each
        # panel's ticks/labels/title *inside* its own rect as an internal gutter, so panels
        # are free to sit flush against each other and the window edge. A real matplotlib
        # Axes instead draws that chrome *outside* its rect -- reusing rect_frac verbatim
        # here pins every panel edge-to-edge against the figure border, leaving nowhere for
        # a title/tick/label to go (they land outside the canvas and simply do not appear).
        # That is worst on an `ax.inset_axes()` figure: the only other panels are the
        # insets, so the "primary" panel's own rect is the default full-figure (0,0,1,1)
        # and its entire title/tick/label chrome renders off-canvas. Map every rect through
        # matplotlib's own default axes margins -- the same ones the single-panel branch
        # below gets for free via `mpl.subplots()` -- so that chrome has somewhere to go.
        # This only changes where each panel's *preview* axes lands on the saved figure:
        # `panel.rect_frac` itself is left untouched, so the live window, inset placement
        # math (which reads a parent's `rect_frac` at inset-creation time) and picking are
        # unaffected, and a nested inset's rect maps consistently since the transform below
        # is affine and independent per axis, matching how `inset_axes` composes rects.
        rc = matplotlib.rcParams
        margin_l, margin_r = rc["figure.subplot.left"], 1.0 - rc["figure.subplot.right"]
        margin_b, margin_t = rc["figure.subplot.bottom"], 1.0 - rc["figure.subplot.top"]
        span_x, span_y = 1.0 - margin_l - margin_r, 1.0 - margin_b - margin_t

        def _margined_rect(rect):
            x0, y0, w, h = rect
            return (margin_l + x0 * span_x, margin_b + y0 * span_y, w * span_x, h * span_y)

        for panel in panels:
            if not getattr(panel, "visible", True):
                continue
            panel_3d = panel.is_3d()
            rect = _reserve_for_twins(
                panel,
                _margined_rect(
                    _spaced_rect(
                        panel, panels, rc["figure.subplot.wspace"], rc["figure.subplot.hspace"]
                    )
                ),
            )
            ax = fig.add_axes(rect, projection="3d") if panel_3d else fig.add_axes(rect)
            # Under the panel's own settings: scale, ticks and legend are per axes, and the
            # engine holds one panel's at a time (see `GPULinePlot.active_panel_index`).
            with _panel_active(engine, panel):
                _draw_panel_layers(ax, panel, panel.scene.layers, panel_3d, engine)
                if panel is primary:
                    _finish_axes(ax, panel_3d, engine, panel=panel)
                    title_ax = ax
                else:
                    if panel_3d:
                        ax.autoscale(enable=True)
                    else:
                        _apply_axis_state(ax, engine, panel)
                    _apply_panel_labels(ax, panel, engine)
                    # `_finish_axes` (title/xlabel/ylabel) stays primary-only -- those read
                    # one shared value off `engine`, not a per-panel one, and applying them
                    # here would repeat whichever panel happens to be primary onto every
                    # other panel. Tick label size is different: it is a genuine per-axes
                    # matplotlib property with no such shared-value constraint, so leaving
                    # it off every non-primary panel had no reason behind it and was just
                    # leaving 3 panels of a 4-panel figure at matplotlib's tiny unstyled
                    # default while the 4th got the figure's real (much larger) size.
                    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
                    # The primary panel's legend is drawn whenever it has labelled layers
                    # (its historical behaviour); another panel gets one only if its own
                    # `legend()` was called, or a figure of unlabelled-by-intent panels
                    # would sprout legends it never had.
                    if not panel_3d and getattr(engine, "legend_show", False):
                        _apply_legend(ax)
            _apply_style_chrome(fig, ax, panel_3d, engine)
            for cb in getattr(panel, "colorbars", None) or []:
                # `cax=`, not `ax=`: `panel.rect_frac` was already shrunk by `colorbar()`
                # itself at call time (unless `inset=True`, which never shrinks it), so
                # letting matplotlib auto-shrink `ax` again here would double the gap.
                cax = fig.add_axes(_margined_rect(tuple(cb.rect_frac)))
                cb_obj = fig.colorbar(
                    _colorbar_mappable(cb),
                    cax=cax,
                    orientation=cb.orientation,
                    # `location=`, not just `orientation=`: with an explicit `cax=`,
                    # matplotlib has no way to infer which side of the bar the caller
                    # actually put ticks-outward on, and silently defaults to the right
                    # (for vertical) / bottom (for horizontal) regardless of where the
                    # bar itself landed -- wrong for `location="left"`/`"top"`, and for
                    # any `inset=True` bar, whose ticks point inward instead.
                    location=_colorbar_location(cb),
                    ticks=cb.ticks,
                    format=cb.format,
                    label=None,
                )
                _finalize_colorbar(cb_obj, cb)
    else:
        # No panel model (defensive, for a test double), or exactly one panel: matplotlib's
        # own auto margins for ticks, labels and title, exactly as before panels existed.
        layers = primary.scene.layers if primary is not None else engine.scene.layers
        has_3d = primary.is_3d() if primary is not None else _has_3d_layers(layers)
        if has_3d:
            fig = mpl.figure(figsize=(width * scale, height * scale), dpi=120)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = mpl.subplots(figsize=(width * scale, height * scale), dpi=120)
        _draw_panel_layers(ax, primary, layers, has_3d, engine)
        _finish_axes(ax, has_3d, engine, panel=primary)
        _apply_style_chrome(fig, ax, has_3d, engine)
        title_ax = ax
        if has_3d:
            single_3d_ax = ax
        # No `panel.rect_frac` placement to reuse here (`mpl.subplots()` owns its own
        # default margins), so this is matplotlib's own auto-shrink form instead of the
        # `cax=` one the multi-panel branch above uses.
        for cb in getattr(primary, "colorbars", None) or []:
            if cb.inset:
                # `ax.inset_axes` bounds are fractions of `ax`'s own box, not the
                # figure's -- recomputed here rather than reusing `cb.rect_frac` (which
                # is a figure-fraction rect meant for the live renderer's own geometry
                # and the multi-panel branch's `cax=` placement above, a different
                # coordinate system from this single-ax branch's).
                from ..pyplot import inset_colorbar_bounds

                cax = ax.inset_axes(inset_colorbar_bounds(cb), transform=ax.transAxes)
                cb_obj = fig.colorbar(
                    _colorbar_mappable(cb),
                    cax=cax,
                    orientation=cb.orientation,
                    location=_colorbar_location(cb),
                    ticks=cb.ticks,
                    format=cb.format,
                    label=None,
                )
                _finalize_colorbar(cb_obj, cb)
                continue
            cb_obj = fig.colorbar(
                _colorbar_mappable(cb),
                ax=ax,
                location=cb.location,
                fraction=cb.fraction,
                pad=cb.pad,
                shrink=cb.shrink,
                aspect=cb.aspect,
                ticks=cb.ticks,
                format=cb.format,
                label=None,
            )
            _finalize_colorbar(cb_obj, cb)

    if len(panels) <= 1:
        # Fixed, content-independent margins -- not fig.tight_layout(). tight_layout()
        # recomputes margins from whatever tick-label text currently exists every time
        # render_preview() runs, which is harmless for a one-off static image but not
        # for an animation: FuncAnimation.save() (glplot/animation.py) calls
        # render_preview() fresh, independently, for every single frame, and a
        # zooming/panning animation's tick labels change width frame to frame (e.g.
        # "-1.0" vs "-0.7500") -- so tight_layout() picked a different left margin each
        # frame, visibly shifting the whole axes box's position and size across the
        # animation ("frame wobble", reported against examples/gallery/animations/
        # 07_fractal_zoom.py, confirmed by comparing extracted frames). Fixed margins
        # guarantee the box never moves, at the cost of not hugging unusually short
        # labels as tightly as tight_layout would -- generous enough for this
        # project's typical label lengths at the current (doubled) font sizes without
        # clipping.
        if single_3d_ax is not None:
            # Axes3D's own box never fills its allocated rect at most view angles (and
            # has no y-axis tick-label gutter to reserve room for), so this pushes the
            # rect near the figure edges rather than reusing the 2D margins below.
            fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.93)
        else:
            # A twin axis puts tick labels and a title outside the axes, on the right
            # (twinx) or top (twiny); leave them room.
            room_right, room_top = _twin_room(primary)
            fig.subplots_adjust(
                left=0.15,
                right=0.96 - (_TWIN_ROOM_X if room_right else 0.0),
                bottom=0.13,
                top=0.90 - (_TWIN_ROOM_Y if room_top else 0.0),
            )
    suptitle = getattr(engine, "_suptitle", None)
    if suptitle is not None and len(panels) > 1:
        text, style = suptitle
        extras = {k: v for k, v in (style or {}).items() if v is not None}
        extras.setdefault("fontsize", TITLE_FONTSIZE * 1.1)
        fig.suptitle(text, **extras)
    for axis, (text, kwargs) in (getattr(engine, "_suplabels", None) or {}).items():
        (fig.supxlabel if axis == "x" else fig.supylabel)(text, **kwargs)
    if title_ax is not None:
        # Must run after tight_layout()/subplots_adjust() above, not before: both can
        # shift the axes horizontally (room for y-tick labels, the 3D margin fix), and a
        # title is centered on its *axes*, not the figure -- sizing against the pre-layout
        # position measured a fit that layout then invalidated.
        _autosize_title(fig, title_ax)
        # _autosize_title() redraws the canvas (fig.canvas.draw()) up to 20 times while it
        # measures the title -- and matplotlib's Axes3D re-runs its own content-fitting
        # autoscale on some of those internal draws even after set_xlim3d()/set_ylim3d()/
        # set_zlim3d() already pinned a range, silently regrowing the box back toward
        # the live data's own extent. Reproduced: a short title (few or zero shrink
        # iterations) left a pinned 3D range alone; a long one (many iterations) visibly
        # regrew it every time. Re-applying the same pin after every draw the title
        # autosizer might have triggered -- rather than only once, before it -- is what
        # actually survives to the saved file.
        _reapply_pinned_limits(title_ax, single_3d_ax is not None, engine)
    return fig


def render_preview(engine: object, filename: str, scale: float = 1.0, **savefig_kwargs) -> None:
    """Export ``engine``'s figure to a static image file.

    See :func:`_build_preview_figure` for how the figure itself is built; this just writes
    it out. ``dpi``/``bbox_inches``/``transparent``/``facecolor``/``pad_inches``/``format``
    (and anything else a caller passes through ``savefig()``) are real ``Figure.savefig()``
    keywords here -- this *is* a real matplotlib figure -- so they are forwarded rather than
    dropped.
    """
    import matplotlib.pyplot as mpl

    fig = _build_preview_figure(engine, scale)
    # A file-like target (``io.BytesIO()``, an open file) is a legitimate ``savefig``
    # destination and has no parent directory to create.
    if isinstance(filename, (str, os.PathLike)):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(filename, **savefig_kwargs)
    finally:
        mpl.close(fig)


def render_preview_array(engine: object, scale: float = 1.0) -> np.ndarray:
    """Render ``engine``'s figure and return its pixels directly, with no file I/O.

    Same figure as :func:`render_preview` (see :func:`_build_preview_figure`), but skips
    the PNG encode and the disk write/read a file-based caller needs. Profiled at ~35% of
    one frame's total cost in a headless animation loop (``PIL``'s PNG encoder alone was
    as expensive as matplotlib's own draw) -- pure encode/decode overhead a caller that only
    wants an ``(H, W, 3)`` uint8 array never needed. Used by
    :func:`glplot.animation.figure_to_rgb`'s headless path, where an animation's hundreds of
    frames each paid that cost once per frame.
    """
    import matplotlib.pyplot as mpl

    fig = _build_preview_figure(engine, scale)
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    mpl.close(fig)
    return rgb


def render_preview_bytes(
    engine: object,
    scale: float = 1.0,
    *,
    format: str = "png",
    inspect: Optional[Callable[[object], Optional[Mapping[str, object]]]] = None,
    **savefig_kwargs,
) -> bytes:
    """Render ``engine``'s figure and return the encoded image, with no file I/O.

    Same figure as :func:`render_preview` (see :func:`_build_preview_figure`), written to
    memory instead of disk -- what a caller that hands the picture to someone else (the
    MCP server returning it to a language model) needs, since a temp file would only be
    read straight back.

    ``inspect`` is called with the finished matplotlib ``Figure`` *before* it is closed, so
    a caller can read what the picture actually resolved to (axis limits, tick labels,
    legend entries) from the one render that produced the bytes, rather than building the
    figure a second time to ask. It may return a mapping of extra ``savefig`` keywords
    (a ``dpi`` chosen from the figure's size, say), which win over ``savefig_kwargs``; setting
    ``fig.dpi`` instead has no effect on what ``savefig`` writes. An exception from it
    propagates and the figure is still closed.
    """
    import io

    import matplotlib.pyplot as mpl

    fig = _build_preview_figure(engine, scale)
    try:
        overrides = inspect(fig) if inspect is not None else None
        buffer = io.BytesIO()
        fig.savefig(buffer, format=format, **{**savefig_kwargs, **(overrides or {})})
    finally:
        mpl.close(fig)
    return buffer.getvalue()
