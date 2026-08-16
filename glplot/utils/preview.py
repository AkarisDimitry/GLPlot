from __future__ import annotations

import warnings
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
            ax.contourf(
                layer.metadata["X"],
                layer.metadata["Y"],
                layer.metadata["Z"],
                levels=layer.metadata.get("levels", 10),
                cmap=layer.metadata.get("cmap", "viridis"),
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
            ax.contour(
                layer.metadata["X"], layer.metadata["Y"], layer.metadata["Z"], **contour_kwargs
            )
            continue
        if layer.metadata.get("artist") == "imshow":
            xmin, xmax, ymin, ymax = layer.metadata["extent"]
            ax.imshow(
                layer.metadata["matrix"],
                extent=(xmin, xmax, ymin, ymax),
                origin=layer.metadata.get("origin", "upper"),
                cmap=layer.metadata.get("cmap", "viridis"),
                vmin=layer.metadata.get("vmin"),
                vmax=layer.metadata.get("vmax"),
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
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=layer.colors,
                s=layer.style.point_size,
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


def _finish_axes(ax: object, has_3d: bool, engine: object) -> None:
    """Apply the figure-global chrome (title, labels, legend, scale, ticks) to ``ax``.

    These read off ``engine`` directly (``engine.title``, ``engine.xlabel``, ...) rather
    than off any one panel: GLPlot's per-panel state is the scene/camera/interaction, not
    the chrome, so a multi-panel figure still has exactly one shared title/label/legend.
    Called once, for whichever panel is ``engine.active_panel`` -- the historical
    single-scene behaviour, preserved rather than guessed at for every extra panel.
    """
    if getattr(engine, "grid_visible", False):
        ax.grid(True, alpha=0.25)
    if hasattr(engine, "xlabel"):
        ax.set_xlabel(engine.xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    if hasattr(engine, "ylabel"):
        ax.set_ylabel(engine.ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    if has_3d:
        # The axis titles set through the 3D panel take precedence over the engine's 2D
        # ones: a 3D scene's labels live on ``axes3d``, and falling straight through to
        # ``engine.zlabel`` used to export a literal "z" over a named axis.
        axes3d_opts = getattr(engine, "axes3d", None)
        if getattr(axes3d_opts, "xlabel", ""):
            ax.set_xlabel(axes3d_opts.xlabel, fontsize=AXIS_LABEL_FONTSIZE)
        if getattr(axes3d_opts, "ylabel", ""):
            ax.set_ylabel(axes3d_opts.ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_zlabel(
            getattr(axes3d_opts, "zlabel", "") or getattr(engine, "zlabel", "z") or "z",
            fontsize=AXIS_LABEL_FONTSIZE,
        )
        view3d = getattr(engine, "view3d", {})
        ax.view_init(elev=float(view3d.get("elev", 28.0)), azim=float(view3d.get("azim", -45.0)))
        camera = getattr(engine, "camera3d", None)
        if camera is not None and camera.projection == "orthographic":
            ax.set_proj_type("ortho")
        ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    else:
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    if getattr(engine, "title", ""):
        ax.set_title(engine.title, fontsize=TITLE_FONTSIZE)
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
    if not has_3d:
        opts = getattr(engine, "options", None)
        scale_x = getattr(opts, "axis_scale_x", "linear")
        if scale_x and scale_x != "linear":
            ax.set_xscale(scale_x, **getattr(opts, "axis_scale_params_x", {}))
        scale_y = getattr(opts, "axis_scale_y", "linear")
        if scale_y and scale_y != "linear":
            ax.set_yscale(scale_y, **getattr(opts, "axis_scale_params_y", {}))
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


def render_preview(engine: object, filename: str, scale: float = 1.0, **savefig_kwargs) -> None:
    """Export ``engine``'s figure to a static image file.

    A single panel renders with matplotlib's own auto margins, unchanged from before panels
    existed. Two or more panels -- a ``subplots()`` grid, ``subplot2grid()``, or
    ``inset_axes()`` -- each get their own :class:`~matplotlib.axes.Axes` at their own
    ``rect_frac`` (already the exact bottom-left-origin figure-fraction format
    ``Figure.add_axes`` expects), so the saved file matches what the live GL window shows
    instead of only whichever panel happened to be active when ``savefig()`` was called.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
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
                _finish_axes(ax, panel_3d, engine)
                title_ax = ax
            else:
                ax.autoscale(enable=True)
            _apply_style_chrome(fig, ax, panel_3d, engine)
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
        _finish_axes(ax, has_3d, engine)
        _apply_style_chrome(fig, ax, has_3d, engine)
        title_ax = ax
        if has_3d:
            single_3d_ax = ax

    if len(panels) <= 1:
        # Meaningless (and noisy) for the multi-panel branch: `tight_layout` only adjusts
        # the margins of Gridspec-managed axes, and every panel there was placed explicitly
        # via `add_axes(rect_frac)` instead, which it leaves alone but warns about anyway.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Tight layout not applied.*")
            fig.tight_layout()
        if single_3d_ax is not None:
            # `tight_layout()` above is a documented no-op for 3D axes (that's the exact
            # warning it just suppressed), so a 3D export otherwise keeps matplotlib's
            # default *2D* subplot margins -- sized for a legend/label gutter a cubic
            # Axes3D box never needed, which is exactly the wide blank band down each
            # side of every 3D gallery preview. Axes3D's own box also never fills its
            # allocated rect at most view angles, so this only pushes that rect near the
            # figure edges rather than trying to compute an exact fit.
            fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.93)
    if title_ax is not None:
        # Must run after tight_layout()/subplots_adjust() above, not before: both can
        # shift the axes horizontally (room for y-tick labels, the 3D margin fix), and a
        # title is centered on its *axes*, not the figure -- sizing against the pre-layout
        # position measured a fit that layout then invalidated.
        _autosize_title(fig, title_ax)
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    # `dpi`/`bbox_inches`/`transparent`/`facecolor`/`pad_inches`/`format` (and anything else
    # a caller passes through `savefig()`) are real `Figure.savefig()` keywords here -- this
    # *is* a real matplotlib figure -- so they are forwarded rather than dropped.
    fig.savefig(filename, **savefig_kwargs)
    mpl.close(fig)
