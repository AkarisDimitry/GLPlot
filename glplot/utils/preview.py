from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

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
            # A marker's outline is its edge, which matplotlib takes as a keyword pair
            # rather than as a path effect. `_outline_edge_kwargs` is empty for a layer with
            # no outline, so the call is character-for-character the historical one.
            ax.scatter(
                verts[:, 0],
                verts[:, 1],
                verts[:, 2],
                c=layer.colors,
                s=max(layer.style.point_size, 0.5),
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
            for idx, seg in enumerate(verts):
                (line,) = ax.plot(
                    seg[:, 0],
                    seg[:, 1],
                    seg[:, 2],
                    color=color,
                    linewidth=line_width,
                    label=layer.label if idx == 0 and layer.label else None,
                )
                _apply_outline(line, layer.style, line_width)
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
                    surf.set_facecolors(face_color)
                if layer.layer_type == "bars3d":
                    surf.set_edgecolor((0.0, 0.0, 0.0, 0.0))
                    surf.set_linewidth(0.0)
            else:
                ax.scatter(
                    verts[:, 0],
                    verts[:, 1],
                    verts[:, 2],
                    c=layer.colors,
                    s=layer.style.point_size,
                    depthshade=False,
                    label=_legend_label(layer),
                )
            continue
        if layer.metadata.get("artist") == "pcolormesh":
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
                c=layer.colors,
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
            if layer.indices is not None and getattr(layer, "mode", "strip") == "triangles":
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
                coll = PolyCollection(
                    polys,
                    facecolors=face_colors,
                    edgecolors=edge_colors,
                    linewidths=layer.style.line_width,
                    label=_legend_label(layer),
                )
                # A silhouette around each triangle, which is what the GL dilation gives a
                # triangle-mode patch too -- the seams between adjacent triangles are the
                # one place the two disagree, and only when the outline is on.
                _apply_outline(coll, layer.style, layer.style.line_width)
                ax.add_collection(coll)
            else:
                patch = Polygon(
                    layer.vertices,
                    closed=True,
                    facecolor=fc,
                    edgecolor=ec,
                    linewidth=layer.style.line_width,
                    label=_legend_label(layer),
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
            text = ax.text(
                layer.x,
                layer.y,
                layer.text,
                fontsize=layer.style.text_size_px,
                color=_rgba(layer.style.color),
                bbox=layer.metadata.get("bbox"),
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
        ax.tick_params(**tick_kwargs)
    else:
        ax.tick_params(axis="both", **tick_kwargs)
    resolved_title = _resolve_axes_title(panel, engine)
    if resolved_title:
        ax.set_title(resolved_title, **title_kwargs)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
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
        opts = getattr(engine, "options", None)
        scale_x = getattr(opts, "axis_scale_x", "linear")
        if scale_x and scale_x != "linear":
            ax.set_xscale(scale_x, **getattr(opts, "axis_scale_params_x", {}))
        scale_y = getattr(opts, "axis_scale_y", "linear")
        if scale_y and scale_y != "linear":
            ax.set_yscale(scale_y, **getattr(opts, "axis_scale_params_y", {}))
        # An explicit plt.xlim()/plt.ylim() call used to be silently discarded the same
        # way: engine._needs_initial_autoscale is False once the caller has set an
        # explicit view (engine.py's set_view(), which xlim()/ylim() call), meaning "this
        # view is deliberate, do not re-fit it to whichever data this particular frame
        # happens to hold." This is the engine-level flag, not Panel.needs_initial_
        # autoscale (a separate, per-panel flag the live run loop's one-time auto-fit
        # uses) -- the two are not kept in sync outside that loop, so a headless script
        # that never calls .run() only ever updates the engine-level one.
        pinned = False
        if not getattr(engine, "_needs_initial_autoscale", True):
            get_xlim = getattr(engine, "get_xlim", None)
            get_ylim = getattr(engine, "get_ylim", None)
            if callable(get_xlim) and callable(get_ylim):
                ax.set_xlim(*get_xlim())
                ax.set_ylim(*get_ylim())
                pinned = True
        if not pinned:
            ax.autoscale(enable=True)
    # A pinned `xticks()`/`yticks()` call (explicit positions, or -- via `_coerce_axis_values`
    # -- a categorical axis's string labels) is stored on `engine.options`, not on any layer,
    # so nothing above ever reconstructs it: this export used to always fall back to
    # matplotlib's own autoscaled tick locator, silently dropping any custom ticks the
    # script had set. `set_xticks`/`set_yticks` run after `autoscale()` specifically because
    # they install a `FixedLocator`, which -- unlike the default locator -- survives it.
    if not has_3d:
        opts = getattr(engine, "options", None)
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
    elif not getattr(engine, "_needs_initial_autoscale", True):
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
            panel_3d = panel.is_3d()
            rect = _margined_rect(tuple(panel.rect_frac))
            ax = fig.add_axes(rect, projection="3d") if panel_3d else fig.add_axes(rect)
            _draw_layers(ax, panel.scene.layers, panel_3d, engine)
            if panel is primary:
                _finish_axes(ax, panel_3d, engine, panel=panel)
                title_ax = ax
            else:
                ax.autoscale(enable=True)
                _apply_panel_labels(ax, panel, engine)
                # `_finish_axes` (title/xlabel/ylabel/legend) stays primary-only -- those
                # read one shared value off `engine`, not a per-panel one, and applying
                # them here would just repeat whichever panel happens to be primary onto
                # every other panel. Tick label size is different: it is a genuine
                # per-axes matplotlib property with no such shared-value constraint, so
                # leaving it off every non-primary panel had no reason behind it and was
                # just leaving 3 panels of a 4-panel figure at matplotlib's tiny unstyled
                # default while the 4th got the figure's real (much larger) size.
                ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
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
        _draw_layers(ax, layers, has_3d, engine)
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
            fig.subplots_adjust(left=0.15, right=0.96, bottom=0.13, top=0.90)
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
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, **savefig_kwargs)
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
