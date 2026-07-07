from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

MAX_LEGEND_ITEMS = 5


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


def _apply_preview_ssao(
    colors: np.ndarray, z_values: np.ndarray, strength: float
) -> np.ndarray:
    cols = np.asarray(colors, dtype=float).copy()
    if cols.ndim == 1:
        cols = np.tile(cols, (len(z_values), 1))
    z = np.asarray(z_values, dtype=float)
    zn = (z - np.nanmin(z)) / max(float(np.nanmax(z) - np.nanmin(z)), 1e-9)
    cavity = 1.0 - np.clip(zn, 0.0, 1.0)
    ao = np.clip(1.0 - float(strength) * 0.62 * cavity, 0.58, 1.0)
    cols[:, :3] *= ao[:, None]
    return cols


def render_preview(engine: object, filename: str, scale: float = 1.0) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as mpl
    from matplotlib.patches import Polygon

    width = max(engine.width / 100.0, 4.0)
    height = max(engine.height / 100.0, 3.0)
    has_3d = any(
        layer.layer_type in {"scatter3d", "mesh3d", "wireframe3d", "bars3d", "volume3d"}
        for layer in engine.scene.layers
    )
    if has_3d:
        fig = mpl.figure(figsize=(width * scale, height * scale), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = mpl.subplots(figsize=(width * scale, height * scale), dpi=120)

    for layer in engine.scene.layers:
        if not layer.style.visible:
            continue
        if layer.layer_type in {"scatter3d", "volume3d"} and layer.vertices is not None:
            verts = layer.vertices
            ax.scatter(
                verts[:, 0],
                verts[:, 1],
                verts[:, 2],
                c=layer.colors,
                s=max(layer.style.point_size, 0.5),
                depthshade=False,
                label=layer.label or None,
            )
            continue
        if layer.layer_type == "wireframe3d" and layer.vertices is not None:
            verts = layer.vertices.reshape(-1, 2, 3)
            color = _rgba(
                layer.colors[0]
                if layer.colors is not None and len(layer.colors)
                else layer.style.color
            )
            for idx, seg in enumerate(verts):
                ax.plot(
                    seg[:, 0],
                    seg[:, 1],
                    seg[:, 2],
                    color=color,
                    linewidth=max(layer.style.line_width, 0.4),
                    label=layer.label if idx == 0 and layer.label else None,
                )
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
                    label=layer.label or None,
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
                aspect="auto",
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
                    ax.quiver(
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
                        label=layer.label or None,
                    )
            else:
                pts = layer.pts
                ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    color=_rgba(layer.style.color),
                    linewidth=layer.style.line_width,
                    label=layer.label or None,
                )
        elif layer.layer_type == "patch" and layer.metadata.get("artist_group") == "quiver":
            pass  # arrowheads handled by ax.quiver() above — skip raw triangles
        elif layer.layer_type == "scatter" and layer.pts is not None:
            pts = layer.pts
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=layer.colors,
                s=layer.style.point_size,
                label=layer.label or None,
            )
        elif layer.layer_type == "patch" and layer.vertices is not None:
            fc = _rgba(layer.style.face_color, (0.2, 0.4, 0.8, 0.35))
            ec = _rgba(layer.style.edge_color, (0.2, 0.4, 0.8, 1.0))
            if layer.indices is not None and getattr(layer, "mode", "strip") == "triangles":
                from matplotlib.collections import PolyCollection

                tris = np.asarray(layer.indices, dtype=int).reshape(-1, 3)
                polys = layer.vertices[tris]
                coll = PolyCollection(
                    polys,
                    facecolors=[fc],
                    edgecolors=[ec],
                    linewidths=layer.style.line_width,
                    label=layer.label or None,
                )
                ax.add_collection(coll)
            else:
                patch = Polygon(
                    layer.vertices,
                    closed=True,
                    facecolor=fc,
                    edgecolor=ec,
                    linewidth=layer.style.line_width,
                    label=layer.label or None,
                )
                ax.add_patch(patch)
        elif layer.layer_type == "line_family" and layer.ab is not None:
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
                cmap="magma",
                interpolation="bilinear",
            )
            ax.set_xlim(float(x0), float(x1))
            ax.set_ylim(float(ymin_v), float(ymax_v))
        elif layer.layer_type == "text":
            ax.text(
                layer.x,
                layer.y,
                layer.text,
                fontsize=layer.style.text_size_px,
                color=_rgba(layer.style.color),
            )

    if getattr(engine, "grid_visible", False):
        ax.grid(True, alpha=0.25)
    if hasattr(engine, "xlabel"):
        ax.set_xlabel(engine.xlabel)
    if hasattr(engine, "ylabel"):
        ax.set_ylabel(engine.ylabel)
    if has_3d:
        ax.set_zlabel(getattr(engine, "zlabel", "z"))
        view3d = getattr(engine, "view3d", {})
        ax.view_init(elev=float(view3d.get("elev", 28.0)), azim=float(view3d.get("azim", -45.0)))
    if getattr(engine, "title", ""):
        ax.set_title(engine.title)
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
            fontsize=8,
            framealpha=0.78,
            borderpad=0.35,
            labelspacing=0.3,
            handlelength=1.4,
        )
    ax.autoscale(enable=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Tight layout not applied.*")
        fig.tight_layout()
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename)
    mpl.close(fig)
