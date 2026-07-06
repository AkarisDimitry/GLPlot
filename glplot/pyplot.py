from __future__ import annotations

import atexit
import time
from typing import Optional, Tuple, Sequence, Union, Literal, Iterable, Any

import numpy as np

from .engine import GPULinePlot
from .core.layers import Layer3D

ColorLike = Union[
    Tuple[float, float, float, float],
    Sequence[float],
    np.ndarray,
]

BlendMode = Literal["auto", "on", "off"]


# ------------------------------------------------------------------
# Global pyplot-like state
# ------------------------------------------------------------------

_CURRENT_PLOT: Optional[GPULinePlot] = None
_ALL_PLOTS: list[GPULinePlot] = []


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _as_float_array(x, ndim: Optional[int] = None, name: str = "array") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {arr.ndim}")
    return np.ascontiguousarray(arr)


def _normalize_rgba(
    color: Optional[ColorLike],
    n: Optional[int] = None,
    default=(0.0, 0.0, 0.0, 1.0),
) -> np.ndarray:
    """
    Returns:
        - shape (4,) if n is None
        - shape (n,4) if n is given
    """
    COLOR_MAP = {
        "white": (1.0, 1.0, 1.0, 1.0),
        "black": (0.0, 0.0, 0.0, 1.0),
        "red": (1.0, 0.0, 0.0, 1.0),
        "green": (0.0, 1.0, 0.0, 1.0),
        "blue": (0.0, 0.0, 1.0, 1.0),
        "cyan": (0.0, 1.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0, 1.0),
        "yellow": (1.0, 1.0, 0.0, 1.0),
        "k": (0.0, 0.0, 0.0, 1.0),
        "w": (1.0, 1.0, 1.0, 1.0),
        "r": (1.0, 0.0, 0.0, 1.0),
        "g": (0.0, 1.0, 0.0, 1.0),
        "b": (0.0, 0.0, 1.0, 1.0),
    }

    if color is None:
        base = np.asarray(default, dtype=np.float32)
    elif isinstance(color, str):
        key = color.lower()
        if key in COLOR_MAP:
            base = np.asarray(COLOR_MAP[key], dtype=np.float32)
        else:
            try:
                from matplotlib.colors import to_rgba

                base = np.asarray(to_rgba(color), dtype=np.float32)
            except ValueError as exc:
                raise ValueError(f"unknown color: {color}") from exc
    else:
        try:
            base = np.asarray(color, dtype=np.float32)
        except (ValueError, TypeError):
            base = np.asarray(default, dtype=np.float32)

    if n is None:
        if base.ndim == 0:  # single value broadcast
            base = np.array([base, base, base, 1.0], dtype=np.float32)
        if base.shape != (4,):
            # Fallback if it's RGB
            if base.shape == (3,):
                base = np.array([base[0], base[1], base[2], 1.0], dtype=np.float32)
            else:
                raise ValueError(
                    f"color must be a single RGBA tuple with shape (4,), got {base.shape}"
                )
        return np.ascontiguousarray(np.clip(base, 0.0, 1.0))

    # Per-object/per-point color
    if base.ndim == 1:
        if base.shape == (3,):
            base = np.array([base[0], base[1], base[2], 1.0], dtype=np.float32)
        if base.shape != (4,):
            raise ValueError("single color must have shape (4,)")
        out = np.tile(base, (n, 1))
        return np.ascontiguousarray(np.clip(out, 0.0, 1.0))

    if base.ndim == 2:
        if base.shape != (n, 4):
            # Handle (n, 3)
            if base.shape == (n, 3):
                new_base = np.ones((n, 4), dtype=np.float32)
                new_base[:, :3] = base
                base = new_base
            else:
                raise ValueError(f"color array must have shape ({n},4), got {base.shape}")
        return np.ascontiguousarray(np.clip(base, 0.0, 1.0))

    raise ValueError("invalid color format")


def _get_or_create_plot() -> GPULinePlot:
    global _CURRENT_PLOT
    if _CURRENT_PLOT is None:
        _CURRENT_PLOT = GPULinePlot()
        _ALL_PLOTS.append(_CURRENT_PLOT)
    return _CURRENT_PLOT


def _parse_plot_format(fmt: Optional[str]) -> dict[str, Any]:
    if not fmt:
        return {}
    colors = {
        "b": "b",
        "g": "g",
        "r": "r",
        "c": "cyan",
        "m": "magenta",
        "y": "yellow",
        "k": "k",
        "w": "w",
    }
    markers = {
        ".",
        ",",
        "o",
        "v",
        "^",
        "<",
        ">",
        "1",
        "2",
        "3",
        "4",
        "s",
        "p",
        "*",
        "h",
        "H",
        "+",
        "x",
        "D",
        "d",
    }
    linestyles = {"--": "--", "-.": "-.", ":": ":", "-": "-"}
    out: dict[str, Any] = {}
    rest = fmt
    for idx in range(10):
        token = f"C{idx}"
        if token in rest:
            out["color"] = token
            rest = rest.replace(token, "", 1)
            break
    for token, style in sorted(linestyles.items(), key=lambda kv: -len(kv[0])):
        if token in rest:
            out["linestyle"] = style
            rest = rest.replace(token, "", 1)
            break
    for ch in rest:
        if ch in colors:
            out["color"] = colors[ch]
        elif ch in markers:
            out["marker"] = ch
        elif ch:
            raise ValueError(f"unsupported format character: {ch!r}")
    return out


def _plot_single(x, y=None, fmt: Optional[str] = None, **kwargs):
    style = _parse_plot_format(fmt)
    style.update({k: v for k, v in kwargs.items() if v is not None})
    if y is None:
        y_arr = _as_float_array(x, ndim=1, name="y")
        x_arr = np.arange(len(y_arr), dtype=np.float32)
    else:
        x_arr = _as_float_array(x, ndim=1, name="x")
        y_arr = _as_float_array(y, ndim=1, name="y")
    return _add_plot_primitive(x_arr, y_arr, **style)


def _project_3d(
    x,
    y,
    z,
    *,
    elev: float = 30.0,
    azim: float = -60.0,
    scale_z: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_arr = _as_float_array(x, ndim=1, name="x")
    y_arr = _as_float_array(y, ndim=1, name="y")
    z_arr = _as_float_array(z, ndim=1, name="z")
    if not (len(x_arr) == len(y_arr) == len(z_arr)):
        raise ValueError("x, y, and z must have the same length")

    az = np.deg2rad(float(azim))
    el = np.deg2rad(float(elev))
    xp = x_arr * np.cos(az) - y_arr * np.sin(az)
    depth = x_arr * np.sin(az) + y_arr * np.cos(az)
    yp = depth * np.sin(el) + z_arr * float(scale_z) * np.cos(el)
    return (
        np.ascontiguousarray(xp, dtype=np.float32),
        np.ascontiguousarray(yp, dtype=np.float32),
        z_arr,
    )


def _colormap_values(
    values, cmap: Optional[str] = None, vmin: Optional[float] = None, vmax: Optional[float] = None
) -> np.ndarray:
    from matplotlib import colormaps

    arr = np.asarray(values, dtype=np.float32)
    lo = float(np.nanmin(arr)) if vmin is None else float(vmin)
    hi = float(np.nanmax(arr)) if vmax is None else float(vmax)
    denom = max(hi - lo, 1e-12)
    normed = np.clip((arr - lo) / denom, 0.0, 1.0)
    return np.asarray(colormaps.get_cmap(cmap or "viridis")(normed), dtype=np.float32)


def _add_3d_layer(
    vertices,
    *,
    colors=None,
    indices=None,
    primitive: str = "points",
    layer_type: str = "scatter3d",
    label: Optional[str] = None,
    elev: float = 28.0,
    azim: float = -45.0,
    point_size: float = 3.0,
    color: Optional[ColorLike] = None,
    alpha: Optional[float] = None,
    metadata: Optional[dict[str, Any]] = None,
):
    plot_obj = _get_or_create_plot()
    verts = _as_float_array(vertices, ndim=2, name="vertices")
    if verts.shape[1] != 3:
        raise ValueError("3D vertices must have shape (N, 3)")
    if colors is None:
        rgba = list(_normalize_rgba(color or (0.1, 0.45, 1.0, 1.0), n=None))
        if alpha is not None:
            rgba[3] *= float(alpha)
        cols = np.tile(np.asarray(rgba, dtype=np.float32), (len(verts), 1))
    else:
        cols = _as_float_array(colors, ndim=2, name="colors")
        if cols.shape != (len(verts), 4):
            raise ValueError("3D colors must have shape (N, 4)")
        if alpha is not None:
            cols = cols.copy()
            cols[:, 3] *= float(alpha)
    idx = None if indices is None else np.ascontiguousarray(indices, dtype=np.uint32)
    layer = Layer3D(
        verts,
        colors=np.ascontiguousarray(cols, dtype=np.float32),
        indices=idx,
        primitive=primitive,
        label=label or "",
        layer_type=layer_type,
    )
    layer.style.point_size = float(point_size)
    layer.metadata.update(metadata or {})
    global_camera = getattr(plot_obj, "view3d", {})
    camera = {
        "elev": float(global_camera.get("elev", elev)),
        "azim": float(global_camera.get("azim", azim)),
        "fov": float(global_camera.get("fov", 42.0)),
    }
    if global_camera.get("distance") is not None:
        camera["distance"] = float(global_camera["distance"])
    layer.metadata["camera"] = camera
    plot_obj.scene.layers.append(layer)
    if getattr(plot_obj, "view3d", {}).get("show_axes", True):
        plot_obj.ensure_3d_axes()
    _set_dirty(plot_obj)
    return layer


def _add_plot_primitive(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    *,
    color: ColorLike = (0.0, 0.0, 0.0, 1.0),
    linestyle: Optional[str] = "-",
    marker: Optional[str] = None,
    linewidth: Optional[float] = None,
    lw: Optional[float] = None,
    width: Optional[float] = None,
    alpha: Optional[float] = None,
    label: Optional[str] = None,
    markersize: Optional[float] = None,
    ms: Optional[float] = None,
    **kwargs,
):
    plot_obj = _get_or_create_plot()
    if len(x_arr) != len(y_arr):
        raise ValueError("x and y must have the same length")

    rgba = list(_normalize_rgba(color, n=None))
    if alpha is not None:
        rgba[3] *= float(alpha)

    line_width = width if width is not None else (linewidth if linewidth is not None else lw)
    line_width = 1.0 if line_width is None else float(line_width)
    artists = []

    if linestyle not in (None, "", "None", "none", " "):
        plot_obj.add_line_strip(x_arr, y_arr, tuple(rgba), width=line_width, label=label)
        layer = plot_obj.scene.layers[-1]
        layer.metadata.update({"linestyle": linestyle, "artist": "line"})
        artists.append(layer)

    if marker not in (None, "", "None", "none", " "):
        size = ms if ms is not None else markersize
        plot_obj.add_scatter(
            x_arr,
            y_arr,
            _normalize_rgba(tuple(rgba), n=len(x_arr)),
            float(size or 6.0),
            label=label,
        )
        layer = plot_obj.scene.layers[-1]
        layer.metadata.update({"marker": marker, "artist": "marker"})
        artists.append(layer)

    _set_dirty(plot_obj)
    return artists or [plot_obj]


def get_engine() -> GPULinePlot:
    """Returns the current active GPULinePlot engine, or creates one if it doesn't exist."""
    return _get_or_create_plot()


def _set_dirty(plot: GPULinePlot) -> None:
    if hasattr(plot, "view") and hasattr(plot.view, "dirty"):
        plot.view.dirty = True
    elif hasattr(plot, "frame") and hasattr(plot.frame, "dirty_scene"):
        plot.frame.dirty_scene = True


def _call_if_exists(plot: GPULinePlot, method_names: Sequence[str], *args, **kwargs):
    for name in method_names:
        fn = getattr(plot, name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    return None


def _set_density(plot: GPULinePlot, enabled: bool) -> None:
    if _call_if_exists(plot, ("set_density_enabled", "set_density_mode"), enabled) is not None:
        return
    if hasattr(plot, "view") and hasattr(plot.view, "show_density"):
        plot.view.show_density = bool(enabled)
    elif hasattr(plot, "show_density"):
        plot.show_density = bool(enabled)
    _set_dirty(plot)


def _set_hud(plot: GPULinePlot, enabled: bool) -> None:
    if _call_if_exists(plot, ("set_hud_enabled",), enabled) is not None:
        return
    if hasattr(plot, "view") and hasattr(plot.view, "hud_visible"):
        plot.view.hud_visible = bool(enabled)
    _set_dirty(plot)


def _set_blending(plot: GPULinePlot, mode: BlendMode) -> None:
    # Preferred backend API
    if _call_if_exists(plot, ("set_blending_mode",), mode) is not None:
        return

    # Fallback attributes if backend stores policy directly
    if hasattr(plot, "blending_mode"):
        plot.blending_mode = mode
    elif hasattr(plot, "policy") and hasattr(plot.policy, "runtime"):
        # do not mutate runtime every frame if backend owns policy;
        # this is just a fallback
        plot.blending_mode = mode
    _set_dirty(plot)


def _set_title(plot: GPULinePlot, title: str) -> None:
    if _call_if_exists(plot, ("set_title",), title) is not None:
        return
    if hasattr(plot, "title"):
        plot.title = str(title)
    _set_dirty(plot)


def _set_view_limits(
    plot: GPULinePlot,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    if _call_if_exists(plot, ("set_view",), xlim=xlim, ylim=ylim) is not None:
        return

    # Fallback only if backend exposes camera-like state
    if hasattr(plot, "view"):
        if xlim is not None and ylim is not None:
            xmin, xmax = float(xlim[0]), float(xlim[1])
            ymin, ymax = float(ylim[0]), float(ylim[1])
            if xmax <= xmin or ymax <= ymin:
                raise ValueError("invalid limits")
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            half_h = 0.5 * (ymax - ymin)
            if hasattr(plot, "width") and hasattr(plot, "height"):
                aspect = max(plot.width, 1) / max(plot.height, 1)
                if aspect <= 0:
                    aspect = 1.0
                # backend world_window uses half_h = padding / zoom
                zoom = 1.0 / max(half_h, 1e-12)
                plot.view.cx = cx
                plot.view.cy = cy
                plot.view.zoom = zoom
                _set_dirty(plot)
                return

    raise AttributeError("Backend does not expose a compatible set_view/xlim/ylim API")


# ------------------------------------------------------------------
# Figure management
# ------------------------------------------------------------------


def figure(
    title: str = "GLPlot",
    width: int = 1280,
    height: int = 800,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 100,
    *,
    hud: bool = False,
    density: bool = False,
    blending: BlendMode = "auto",
    lod: bool = True,
    budget: int = 8,
    multisample: bool = False,
    cache: bool = True,
    clipping: bool = True,
    ssao: bool = False,
) -> GPULinePlot:
    """
    Create a new figure and make it current.
    """
    global _CURRENT_PLOT
    if figsize is not None:
        width = int(float(figsize[0]) * int(dpi))
        height = int(float(figsize[1]) * int(dpi))
    plot = GPULinePlot(width=width, height=height, title=title)

    # Apply optimization settings
    plot.options.lod_enabled = bool(lod)
    plot.options.lod_target_coverage = float(budget) / 8.0
    plot.options.enable_hud = bool(hud)
    plot.options.enable_multisample = bool(multisample)
    plot.options.enable_cache_interaction_path = bool(cache)
    plot.options.enable_clipping_optimization = bool(clipping)
    plot.options.visual.ssao.enabled = bool(ssao)

    _set_hud(plot, hud)
    _set_density(plot, density)
    _set_blending(plot, blending)

    _CURRENT_PLOT = plot
    _ALL_PLOTS.append(plot)
    _set_dirty(plot)
    return plot


def gcf() -> GPULinePlot:
    """Get current figure."""
    return _get_or_create_plot()


def options(**kwargs):
    """
    Update EngineOptions for the current figure.
    Example: gplt.options(density_resolution_scale=0.5, cache_refresh_hz=60)
    """
    plot = _get_or_create_plot()
    for k, v in kwargs.items():
        if hasattr(plot.options, k):
            setattr(plot.options, k, v)
        else:
            raise AttributeError(f"EngineOptions has no attribute '{k}'")
    _set_dirty(plot)


def subplots(
    title: str = "GLPlot",
    width: int = 1280,
    height: int = 800,
    **kwargs,
):
    """
    Matplotlib-like convenience.
    For now this backend manages a single interactive axes/view.
    Returns (fig, ax_like), both pointing to the same GPULinePlot object.
    """
    fig = figure(title=title, width=width, height=height, **kwargs)
    return fig, fig


def close(fig: Optional[GPULinePlot] = None) -> None:
    """
    Close a figure reference from pyplot state.
    Note: actual window destruction depends on backend lifecycle.
    """
    global _CURRENT_PLOT

    if fig is None:
        fig = _CURRENT_PLOT

    if fig is None:
        return

    try:
        _ALL_PLOTS.remove(fig)
    except ValueError:
        pass

    if fig is _CURRENT_PLOT:
        _CURRENT_PLOT = _ALL_PLOTS[-1] if _ALL_PLOTS else None

    # Optional backend hook
    _call_if_exists(fig, ("close", "shutdown"))


def clf() -> None:
    """Clear current figure."""
    plot = _get_or_create_plot()
    for name in ("clear", "clf", "reset_scene"):
        fn = getattr(plot, name, None)
        if callable(fn):
            fn()
            _set_dirty(plot)
            return

    # Conservative fallback for older engines.
    if hasattr(plot, "scene"):
        from .core.legacy import SceneData

        plot.scene = SceneData()
        _set_dirty(plot)
        return


def cla() -> None:
    """Alias for clf() in this single-axes backend."""
    clf()


# ------------------------------------------------------------------
# Plotting primitives
# ------------------------------------------------------------------


def lines(
    a: Sequence[float],
    b: Sequence[float],
    x_range: Tuple[float, float],
    color: Optional[ColorLike] = None,
    width: float = 1.0,
    alpha: Optional[float] = None,
    label: Optional[str] = None,
):
    """
    Plot many lines in the form y = a*x + b.
    This is the main high-performance primitive.
    """
    plot = _get_or_create_plot()

    a_arr = _as_float_array(a, ndim=1, name="a")
    b_arr = _as_float_array(b, ndim=1, name="b")
    if len(a_arr) != len(b_arr):
        raise ValueError("a and b must have the same length")

    ab = np.column_stack([a_arr, b_arr]).astype(np.float32, copy=False)

    # Resolve color and alpha
    cols = _normalize_rgba(color, n=len(ab)) if color is not None else None
    if alpha is not None:
        if cols is None:
            # Default to black with alpha
            cols = np.zeros((len(ab), 4), dtype=np.float32)
            cols[:, 3] = float(alpha)
        else:
            cols[:, 3] *= float(alpha)

    plot.set_lines_ab(ab, x_range=x_range, colors=cols)

    if hasattr(plot.scene.lines, "style"):
        plot.scene.lines.style.line_width = float(width)
        if alpha is not None:
            plot.scene.lines.style.alpha = float(alpha)
        plot.scene.lines.label = label or "Lines"

    _set_dirty(plot)
    return plot


def plot_lines(
    a: Sequence[float],
    b: Sequence[float],
    x_range: Tuple[float, float],
    colors: Optional[np.ndarray] = None,
):
    """
    Backward-compatible alias for line family plotting.
    """
    plot = _get_or_create_plot()

    a_arr = _as_float_array(a, ndim=1, name="a")
    b_arr = _as_float_array(b, ndim=1, name="b")
    if len(a_arr) != len(b_arr):
        raise ValueError("a and b must have the same length")

    ab = np.column_stack([a_arr, b_arr]).astype(np.float32, copy=False)
    cols = None if colors is None else _as_float_array(colors, ndim=2, name="colors")
    plot.set_lines_ab(ab, x_range=x_range, colors=cols)
    _set_dirty(plot)
    return plot


def plot(*args, **kwargs):
    """
    Plot one or more traditional connected polylines.
    Supports Matplotlib-style forms: plot(y), plot(x, y), plot(x, y, "ro-"),
    and repeated groups such as plot(x1, y1, "r-", x2, y2, "bo").
    """
    if not args:
        raise TypeError("plot() missing data")
    artists = []
    i = 0
    while i < len(args):
        if i + 1 < len(args) and not isinstance(args[i + 1], str):
            x, y = args[i], args[i + 1]
            i += 2
        else:
            x, y = args[i], None
            i += 1
        fmt = None
        if i < len(args) and isinstance(args[i], str):
            fmt = args[i]
            i += 1
        artists.extend(_plot_single(x, y, fmt, **kwargs))
    return artists


def plot3d(x, y, z, *args, elev: float = 30.0, azim: float = -60.0, scale_z: float = 0.7, **kwargs):
    """Plot a 3D line using the native 3D geometry shader path."""
    style = _parse_plot_format(args[0] if args and isinstance(args[0], str) else None)
    style.update(kwargs)
    verts = np.column_stack(
        [
            _as_float_array(x, ndim=1, name="x"),
            _as_float_array(y, ndim=1, name="y"),
            _as_float_array(z, ndim=1, name="z") * float(scale_z),
        ]
    ).astype(np.float32)
    if len(verts) < 2:
        raise ValueError("plot3d requires at least two points")
    segs = np.empty(((len(verts) - 1) * 2, 3), dtype=np.float32)
    segs[0::2] = verts[:-1]
    segs[1::2] = verts[1:]
    layer = _add_3d_layer(
        segs,
        primitive="lines",
        layer_type="wireframe3d",
        label=style.get("label"),
        elev=elev,
        azim=azim,
        color=style.get("color", "C0"),
        alpha=style.get("alpha"),
        metadata={"artist": "plot3d", "zdata": verts[:, 2], "scale_z": scale_z},
    )
    layer.style.line_width = float(style.get("linewidth", style.get("lw", 1.0)))
    return [layer]


def scatter(
    x: Sequence[float],
    y: Sequence[float],
    color: Optional[ColorLike] = None,
    size: float = 10.0,
    c: Optional[ColorLike] = None,
    s: Optional[float] = None,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: Optional[float] = None,
    label: Optional[str] = None,
    marker: Optional[str] = None,
):
    """
    Scatter plot.
    """
    plot_obj = _get_or_create_plot()
    x_arr = _as_float_array(x, ndim=1, name="x")
    y_arr = _as_float_array(y, ndim=1, name="y")

    if len(x_arr) != len(y_arr):
        raise ValueError("x and y must have the same length")

    resolved_color = c if c is not None else color
    if resolved_color is not None:
        maybe_values = np.asarray(resolved_color)
    else:
        maybe_values = None

    if (
        maybe_values is not None
        and maybe_values.ndim == 1
        and len(maybe_values) == len(x_arr)
        and not isinstance(resolved_color, str)
        and np.issubdtype(maybe_values.dtype, np.number)
    ):
        from matplotlib import colormaps

        lo = float(np.nanmin(maybe_values)) if vmin is None else float(vmin)
        hi = float(np.nanmax(maybe_values)) if vmax is None else float(vmax)
        denom = max(hi - lo, 1e-12)
        normed = np.clip((maybe_values.astype(np.float32) - lo) / denom, 0.0, 1.0)
        cols = np.asarray(colormaps.get_cmap(cmap or "viridis")(normed), dtype=np.float32)
    else:
        cols = _normalize_rgba(resolved_color or (0.0, 0.0, 0.0, 1.0), n=len(x_arr))
    if alpha is not None:
        cols[:, 3] *= float(alpha)
    plot_obj.add_scatter(x_arr, y_arr, cols, float(s if s is not None else size), label=label)
    plot_obj.scene.layers[-1].metadata.update(
        {"marker": marker, "artist": "scatter", "cmap": cmap, "vmin": vmin, "vmax": vmax}
    )
    _set_dirty(plot_obj)
    return plot_obj.scene.layers[-1]


def scatter3d(
    x,
    y,
    z,
    *args,
    elev: float = 30.0,
    azim: float = -60.0,
    scale_z: float = 0.7,
    c: Optional[ColorLike] = None,
    cmap: Optional[str] = None,
    **kwargs,
):
    """Scatter a 3D point cloud using the native 3D geometry shader path."""
    x_arr = _as_float_array(x, ndim=1, name="x")
    y_arr = _as_float_array(y, ndim=1, name="y")
    z_arr = _as_float_array(z, ndim=1, name="z")
    if not (len(x_arr) == len(y_arr) == len(z_arr)):
        raise ValueError("x, y, and z must have the same length")
    verts = np.column_stack([x_arr, y_arr, z_arr * float(scale_z)]).astype(np.float32)
    values = z_arr if c is None else c
    if (
        values is not None
        and not isinstance(values, str)
        and np.asarray(values).ndim == 1
        and len(np.asarray(values)) == len(verts)
    ):
        cols = _colormap_values(values, cmap=cmap, vmin=kwargs.get("vmin"), vmax=kwargs.get("vmax"))
    else:
        cols = None
    layer = _add_3d_layer(
        verts,
        colors=cols,
        primitive="points",
        layer_type="scatter3d",
        label=kwargs.get("label"),
        elev=elev,
        azim=azim,
        point_size=float(kwargs.get("s", kwargs.get("size", 3.0))),
        color=values if isinstance(values, str) else kwargs.get("color"),
        alpha=kwargs.get("alpha"),
        metadata={"artist": "scatter3d", "zdata": z_arr, "scale_z": scale_z, "cmap": cmap},
    )
    return layer


def fill_between(
    x: Sequence[float],
    y1: Sequence[float],
    y2: Union[float, Sequence[float]] = 0,
    *,
    color: ColorLike = (0.2, 0.4, 0.8, 0.35),
    alpha: Optional[float] = None,
    label: Optional[str] = None,
):
    x_arr = _as_float_array(x, ndim=1, name="x")
    y1_arr = _as_float_array(y1, ndim=1, name="y1")
    y2_arr = (
        np.full_like(y1_arr, float(y2))
        if np.isscalar(y2)
        else _as_float_array(y2, ndim=1, name="y2")
    )
    if not (len(x_arr) == len(y1_arr) == len(y2_arr)):
        raise ValueError("x, y1, and y2 must have the same length")
    # Interleave top/bottom for GL_TRIANGLE_STRIP: top[0], bot[0], top[1], bot[1], ...
    # Each pair of consecutive quads forms two triangles that correctly fill the band.
    verts = np.empty((2 * len(x_arr), 2), dtype=np.float32)
    verts[0::2, 0] = x_arr
    verts[0::2, 1] = y1_arr
    verts[1::2, 0] = x_arr
    verts[1::2, 1] = y2_arr
    rgba = list(_normalize_rgba(color, n=None))
    if alpha is not None:
        rgba[3] *= float(alpha)
    return add_patch(
        verts, mode="strip", face_color=tuple(rgba), edge_color=tuple(rgba), label=label
    )


def bar(
    x: Sequence[float],
    height: Sequence[float],
    width: float = 0.8,
    bottom: Union[float, Sequence[float]] = 0,
    *,
    color: ColorLike = (0.2, 0.4, 0.8, 1.0),
    alpha: Optional[float] = None,
    label: Optional[str] = None,
):
    x_arr = _as_float_array(x, ndim=1, name="x")
    h_arr = _as_float_array(height, ndim=1, name="height")
    b_arr = (
        np.full_like(h_arr, float(bottom))
        if np.isscalar(bottom)
        else _as_float_array(bottom, ndim=1, name="bottom")
    )
    if not (len(x_arr) == len(h_arr) == len(b_arr)):
        raise ValueError("x, height, and bottom must have the same length")
    half = float(width) / 2.0
    patches = []
    rgba = list(_normalize_rgba(color, n=None))
    if alpha is not None:
        rgba[3] *= float(alpha)
    _QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    for idx, (xc, h, btm) in enumerate(zip(x_arr, h_arr, b_arr)):
        verts = np.array(
            [[xc - half, btm], [xc - half, btm + h], [xc + half, btm + h], [xc + half, btm]],
            dtype=np.float32,
        )
        patches.append(
            add_patch(
                verts,
                indices=_QUAD_INDICES,
                mode="triangles",
                face_color=tuple(rgba),
                edge_color=tuple(rgba),
                label=label if idx == 0 else None,
            )
        )
    return patches


def hist(
    x: Sequence[float],
    bins: Union[int, Sequence[float]] = 10,
    *,
    density: bool = False,
    color: ColorLike = (0.2, 0.4, 0.8, 1.0),
    alpha: Optional[float] = None,
    label: Optional[str] = None,
):
    values = _as_float_array(x, ndim=1, name="x")
    counts, edges = np.histogram(values, bins=bins, density=density)
    centers = 0.5 * (edges[:-1] + edges[1:])
    artists = bar(
        centers, counts, width=float(np.min(np.diff(edges))), color=color, alpha=alpha, label=label
    )
    return counts, edges, artists


def hist2d(
    x,
    y,
    bins=100,
    range=None,
    density: bool = False,
    cmap: str = "magma",
    s: Optional[float] = None,
    label: Optional[str] = None,
):
    x_arr = _as_float_array(x, ndim=1, name="x")
    y_arr = _as_float_array(y, ndim=1, name="y")
    if len(x_arr) != len(y_arr):
        raise ValueError("x and y must have the same length")
    counts, xedges, yedges = np.histogram2d(x_arr, y_arr, bins=bins, range=range, density=density)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    xx, yy = np.meshgrid(xc, yc, indexing="ij")
    values = counts.ravel()
    mask = values > 0
    layer = scatter(
        xx.ravel()[mask],
        yy.ravel()[mask],
        c=values[mask],
        cmap=cmap,
        s=s or max(2.0, 9000.0 / max(len(values), 1)),
        marker="s",
        label=label,
    )
    layer.metadata.update(
        {"artist": "hist2d", "counts": counts, "xedges": xedges, "yedges": yedges}
    )
    return counts, xedges, yedges, layer


def imshow(
    X,
    cmap: str = "viridis",
    origin: str = "upper",
    extent: Optional[Tuple[float, float, float, float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: Optional[float] = None,
    label: Optional[str] = None,
    **kwargs,
):
    matrix = _as_float_array(X, ndim=2, name="X")
    rows, cols = matrix.shape
    if extent is None:
        xmin, xmax, ymin, ymax = -0.5, cols - 0.5, -0.5, rows - 0.5
    else:
        xmin, xmax, ymin, ymax = map(float, extent)
    xs = np.linspace(xmin, xmax, cols, dtype=np.float32)
    ys = (
        np.linspace(ymax, ymin, rows, dtype=np.float32)
        if origin == "upper"
        else np.linspace(ymin, ymax, rows, dtype=np.float32)
    )
    xx, yy = np.meshgrid(xs, ys)
    colors = _colormap_values(matrix.ravel(), cmap=cmap, vmin=vmin, vmax=vmax)
    if alpha is not None:
        colors[:, 3] *= float(alpha)
    size = kwargs.pop("s", max(1.0, 650.0 / max(rows, cols)))
    plot_obj = _get_or_create_plot()
    plot_obj.add_scatter(xx.ravel(), yy.ravel(), colors, float(size), label=label)
    layer = plot_obj.scene.layers[-1]
    layer.metadata.update(
        {
            "artist": "imshow",
            "matrix": matrix,
            "extent": (xmin, xmax, ymin, ymax),
            "origin": origin,
            "cmap": cmap,
            "vmin": vmin,
            "vmax": vmax,
        }
    )
    _set_dirty(plot_obj)
    return layer


def matshow(A, **kwargs):
    return imshow(A, origin=kwargs.pop("origin", "upper"), **kwargs)


def pcolormesh(
    X,
    Y=None,
    C=None,
    *,
    cmap: str = "viridis",
    shading: str = "auto",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: Optional[float] = None,
    label: Optional[str] = None,
    **kwargs,
):
    if C is None:
        matrix = _as_float_array(X, ndim=2, name="C")
        yy, xx = np.indices(matrix.shape, dtype=np.float32)
    else:
        xx = _as_float_array(X, name="X")
        yy = _as_float_array(Y, name="Y")
        matrix = _as_float_array(C, ndim=2, name="C")
        if xx.shape != matrix.shape or yy.shape != matrix.shape:
            raise ValueError("X, Y, and C must have matching shapes")

    colors = _colormap_values(matrix.ravel(), cmap=cmap, vmin=vmin, vmax=vmax)
    if alpha is not None:
        colors[:, 3] *= float(alpha)
    plot_obj = _get_or_create_plot()
    size = kwargs.pop("s", max(1.0, min(7.0, 700.0 / max(matrix.shape))))
    plot_obj.add_scatter(xx.ravel(), yy.ravel(), colors, float(size), label=label)
    layer = plot_obj.scene.layers[-1]
    layer.metadata.update(
        {
            "artist": "pcolormesh",
            "X": xx,
            "Y": yy,
            "C": matrix,
            "cmap": cmap,
            "shading": shading,
            "vmin": vmin,
            "vmax": vmax,
        }
    )
    _set_dirty(plot_obj)
    return layer


def contour(
    X,
    Y=None,
    Z=None,
    levels=10,
    *,
    colors: Optional[ColorLike] = None,
    cmap: str = "viridis",
    linewidths: float = 1.0,
    label: Optional[str] = None,
    **kwargs,
):
    if Z is None:
        matrix = _as_float_array(X, ndim=2, name="Z")
        yy, xx = np.indices(matrix.shape, dtype=np.float32)
    else:
        xx = _as_float_array(X, name="X")
        yy = _as_float_array(Y, name="Y")
        matrix = _as_float_array(Z, ndim=2, name="Z")
    extent = (float(np.min(xx)), float(np.max(xx)), float(np.min(yy)), float(np.max(yy)))
    layer = imshow(np.zeros((2, 2), dtype=np.float32), extent=extent, alpha=0.0, label=label)
    layer.metadata.update(
        {
            "artist": "contour",
            "X": xx,
            "Y": yy,
            "Z": matrix,
            "levels": levels,
            "colors": colors,
            "cmap": cmap,
            "linewidths": linewidths,
        }
    )
    return layer


def contourf(
    X,
    Y=None,
    Z=None,
    levels=10,
    *,
    cmap: str = "viridis",
    alpha: Optional[float] = None,
    label: Optional[str] = None,
    **kwargs,
):
    if Z is None:
        matrix = _as_float_array(X, ndim=2, name="Z")
        yy, xx = np.indices(matrix.shape, dtype=np.float32)
    else:
        xx = _as_float_array(X, name="X")
        yy = _as_float_array(Y, name="Y")
        matrix = _as_float_array(Z, ndim=2, name="Z")
    extent = (float(np.min(xx)), float(np.max(xx)), float(np.min(yy)), float(np.max(yy)))
    layer = imshow(matrix, extent=extent, cmap=cmap, alpha=alpha, label=label)
    layer.metadata.update(
        {
            "artist": "contourf",
            "X": xx,
            "Y": yy,
            "Z": matrix,
            "levels": levels,
            "cmap": cmap,
            "alpha": alpha,
        }
    )
    return layer


def plot_surface(
    X,
    Y,
    Z,
    *,
    cmap: str = "viridis",
    elev: float = 30.0,
    azim: float = -60.0,
    scale_z: float = 0.7,
    rstride: int = 1,
    cstride: int = 1,
    alpha: Optional[float] = None,
    label: Optional[str] = None,
    **kwargs,
):
    xx = _as_float_array(X, name="X")
    yy = _as_float_array(Y, name="Y")
    zz = _as_float_array(Z, name="Z")
    if not (xx.shape == yy.shape == zz.shape):
        raise ValueError("X, Y, and Z must have matching shapes")
    xs, ys, zs = xx[::rstride, ::cstride], yy[::rstride, ::cstride], zz[::rstride, ::cstride]
    rows, cols = xs.shape
    verts = np.column_stack([xs.ravel(), ys.ravel(), (zs * float(scale_z)).ravel()]).astype(
        np.float32
    )
    colors = _colormap_values(zs.ravel(), cmap=cmap)
    if alpha is not None:
        colors[:, 3] *= float(alpha)
    indices = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            a = r * cols + c
            b = a + 1
            d = (r + 1) * cols + c
            e = d + 1
            indices.extend([a, d, b, b, d, e])
    layer = _add_3d_layer(
        verts,
        colors=colors,
        indices=np.asarray(indices, dtype=np.uint32),
        primitive="triangles",
        layer_type="mesh3d",
        label=label,
        elev=elev,
        azim=azim,
        metadata={
            "artist": "plot_surface",
            "X": xx,
            "Y": yy,
            "Z": zz,
            "scale_z": scale_z,
            "cmap": cmap,
            "rstride": rstride,
            "cstride": cstride,
        },
    )
    return layer


def mesh3d(
    vertices,
    faces=None,
    *,
    color: ColorLike = "tab:blue",
    c: Optional[Sequence[float]] = None,
    cmap: str = "viridis",
    alpha: Optional[float] = None,
    elev: float = 30.0,
    azim: float = -60.0,
    label: Optional[str] = None,
):
    verts = _as_float_array(vertices, ndim=2, name="vertices")
    if verts.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    colors = _colormap_values(c, cmap=cmap) if c is not None else None
    layer = _add_3d_layer(
        verts,
        colors=colors,
        indices=None if faces is None else np.asarray(faces, dtype=np.uint32).ravel(),
        primitive="triangles" if faces is not None else "points",
        layer_type="mesh3d",
        label=label,
        elev=elev,
        azim=azim,
        color=color,
        alpha=alpha,
        metadata={"artist": "mesh3d", "faces": faces, "cmap": cmap},
    )
    return layer


def volume3d(
    x,
    y,
    z,
    values=None,
    *,
    threshold: Optional[float] = None,
    cmap: str = "magma",
    alpha: float = 0.45,
    elev: float = 30.0,
    azim: float = -60.0,
    s: float = 2.0,
    label: Optional[str] = None,
):
    x_arr = _as_float_array(x, name="x").ravel()
    y_arr = _as_float_array(y, name="y").ravel()
    z_arr = _as_float_array(z, name="z").ravel()
    if values is None:
        val_arr = z_arr
    else:
        val_arr = _as_float_array(values, name="values").ravel()
    if not (len(x_arr) == len(y_arr) == len(z_arr) == len(val_arr)):
        raise ValueError("x, y, z, and values must have compatible sizes")
    mask = np.ones(len(x_arr), dtype=bool)
    if threshold is not None:
        mask = val_arr >= float(threshold)
    verts = np.column_stack([x_arr[mask], y_arr[mask], z_arr[mask]]).astype(np.float32)
    colors = _colormap_values(val_arr[mask], cmap=cmap)
    colors[:, 3] *= float(alpha)
    layer = _add_3d_layer(
        verts,
        colors=colors,
        primitive="points",
        layer_type="volume3d",
        label=label,
        elev=elev,
        azim=azim,
        point_size=s,
        metadata={
            "artist": "volume3d",
            "threshold": threshold,
            "cmap": cmap,
            "values": val_arr[mask],
        },
    )
    return layer


voxels = volume3d


def plot_wireframe(
    X,
    Y,
    Z,
    *,
    elev: float = 30.0,
    azim: float = -60.0,
    scale_z: float = 0.7,
    rstride: int = 4,
    cstride: int = 4,
    color: ColorLike = "k",
    linewidth: float = 0.7,
    label: Optional[str] = None,
    **kwargs,
):
    xx = _as_float_array(X, name="X")
    yy = _as_float_array(Y, name="Y")
    zz = _as_float_array(Z, name="Z")
    if not (xx.shape == yy.shape == zz.shape):
        raise ValueError("X, Y, and Z must have matching shapes")
    segments = []
    for row in range(0, xx.shape[0], max(1, int(rstride))):
        pts = np.column_stack([xx[row, :], yy[row, :], zz[row, :] * float(scale_z)]).astype(
            np.float32
        )
        segments.extend(np.column_stack([pts[:-1], pts[1:]]).reshape(-1, 3))
    for col in range(0, xx.shape[1], max(1, int(cstride))):
        pts = np.column_stack([xx[:, col], yy[:, col], zz[:, col] * float(scale_z)]).astype(
            np.float32
        )
        segments.extend(np.column_stack([pts[:-1], pts[1:]]).reshape(-1, 3))
    layer = _add_3d_layer(
        np.asarray(segments, dtype=np.float32),
        primitive="lines",
        layer_type="wireframe3d",
        label=label,
        elev=elev,
        azim=azim,
        color=color,
        metadata={
            "artist": "plot_wireframe",
            "X": xx,
            "Y": yy,
            "Z": zz,
            "scale_z": scale_z,
            "rstride": rstride,
            "cstride": cstride,
        },
    )
    layer.style.line_width = float(linewidth)
    return [layer]


def _bar3d_box_geometry(x_arr, y_arr, z_arr, dx_arr, dy_arr, dz_arr, bar_colors, gap_value):
    N = len(x_arr)
    shrink_x = dx_arr * gap_value
    shrink_y = dy_arr * gap_value
    xx = x_arr + shrink_x * 0.5
    yy = y_arr + shrink_y * 0.5
    ddx = dx_arr - shrink_x
    ddy = dy_arr - shrink_y

    verts = np.empty((N, 8, 3), dtype=np.float32)
    verts[:, 0] = np.stack([xx, yy, z_arr], axis=1)
    verts[:, 1] = np.stack([xx + ddx, yy, z_arr], axis=1)
    verts[:, 2] = np.stack([xx + ddx, yy + ddy, z_arr], axis=1)
    verts[:, 3] = np.stack([xx, yy + ddy, z_arr], axis=1)
    verts[:, 4] = np.stack([xx, yy, z_arr + dz_arr], axis=1)
    verts[:, 5] = np.stack([xx + ddx, yy, z_arr + dz_arr], axis=1)
    verts[:, 6] = np.stack([xx + ddx, yy + ddy, z_arr + dz_arr], axis=1)
    verts[:, 7] = np.stack([xx, yy + ddy, z_arr + dz_arr], axis=1)

    rgba_bottom = bar_colors.copy()
    rgba_bottom[:, :3] *= 0.48
    cols = np.empty((N, 8, 4), dtype=np.float32)
    cols[:, :4] = rgba_bottom[:, np.newaxis, :]
    cols[:, 4:] = bar_colors[:, np.newaxis, :]

    box_tris = np.array(
        [
            0,
            1,
            2,
            0,
            2,
            3,
            4,
            6,
            5,
            4,
            7,
            6,
            0,
            4,
            5,
            0,
            5,
            1,
            1,
            5,
            6,
            1,
            6,
            2,
            2,
            6,
            7,
            2,
            7,
            3,
            3,
            7,
            4,
            3,
            4,
            0,
        ],
        dtype=np.uint32,
    )
    base = np.arange(N, dtype=np.uint32)[:, np.newaxis] * 8
    indices = (box_tris[np.newaxis, :] + base).ravel()

    ea = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3], dtype=np.intp)
    eb = np.array([1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7], dtype=np.intp)
    edge_verts = np.stack([verts[:, ea, :], verts[:, eb, :]], axis=2).reshape(-1, 3)

    return verts.reshape(-1, 3), cols.reshape(-1, 4), indices, edge_verts


def _bar3d_hex_geometry(x_arr, y_arr, z_arr, dx_arr, dy_arr, dz_arr, bar_colors, gap_value):
    N = len(x_arr)
    shrink_x = dx_arr * gap_value
    shrink_y = dy_arr * gap_value
    xx = x_arr + shrink_x * 0.5
    yy = y_arr + shrink_y * 0.5
    ddx = dx_arr - shrink_x
    ddy = dy_arr - shrink_y

    rx = (ddx * 0.5).astype(np.float32)
    ry = (ddy * 0.5).astype(np.float32)
    cx = (xx + rx).astype(np.float32)
    cy = (yy + ry).astype(np.float32)

    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False).astype(np.float32)
    bx = cx[:, np.newaxis] + rx[:, np.newaxis] * np.cos(angles)  # (N, 6)
    by = cy[:, np.newaxis] + ry[:, np.newaxis] * np.sin(angles)  # (N, 6)
    bz = np.broadcast_to(z_arr[:, np.newaxis].astype(np.float32), (N, 6))
    tz = np.broadcast_to((z_arr + dz_arr)[:, np.newaxis].astype(np.float32), (N, 6))

    bottom = np.stack([bx, by, np.array(bz)], axis=2)  # (N, 6, 3)
    top = np.stack([bx, by, np.array(tz)], axis=2)  # (N, 6, 3)
    verts = np.concatenate([bottom, top], axis=1)  # (N, 12, 3)

    rgba_bottom = bar_colors.copy()
    rgba_bottom[:, :3] *= 0.48
    cols = np.empty((N, 12, 4), dtype=np.float32)
    cols[:, :6] = rgba_bottom[:, np.newaxis, :]
    cols[:, 6:] = bar_colors[:, np.newaxis, :]

    j6 = np.arange(6, dtype=np.uint32)
    k6 = (j6 + 1) % 6
    side_rel = np.stack([j6, k6, j6 + 6, k6, k6 + 6, j6 + 6], axis=1).ravel()  # (36,)
    bot_fan = np.array([0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5], dtype=np.uint32)
    top_fan = np.array([6, 8, 7, 6, 9, 8, 6, 10, 9, 6, 11, 10], dtype=np.uint32)
    all_rel = np.concatenate([side_rel, bot_fan, top_fan])  # (60,)
    base = np.arange(N, dtype=np.uint32)[:, np.newaxis] * 12
    indices = (all_rel[np.newaxis, :] + base).ravel()

    ji = j6.astype(np.intp)
    ki = k6.astype(np.intp)
    edge_verts = np.stack(
        [bottom[:, ji], bottom[:, ki], top[:, ji], top[:, ki], bottom[:, ji], top[:, ji]],
        axis=2,
    ).reshape(-1, 3)

    return verts.reshape(-1, 3), cols.reshape(-1, 4), indices, edge_verts


def bar3d(
    x,
    y,
    z,
    dx,
    dy,
    dz,
    *,
    color: ColorLike = "tab:blue",
    alpha: Optional[float] = None,
    elev: float = 30.0,
    azim: float = -60.0,
    scale_z: float = 0.7,
    label: Optional[str] = None,
    shape: str = "box",
    c: Optional[ColorLike] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    gap: float = 0.0,
    edge_color: Optional[ColorLike] = "k",
    edge_width: float = 0.8,
    ssao: Optional[bool] = None,
    ssao_strength: Optional[float] = None,
    **kwargs,
):
    x_arr = _as_float_array(x, name="x").ravel()
    y_arr = _as_float_array(y, name="y").ravel()
    z_arr = _as_float_array(z, name="z").ravel()
    dx_arr = np.broadcast_to(np.asarray(dx, dtype=np.float32), x_arr.shape)
    dy_arr = np.broadcast_to(np.asarray(dy, dtype=np.float32), x_arr.shape)
    dz_arr = np.broadcast_to(np.asarray(dz, dtype=np.float32), x_arr.shape)
    if not (len(x_arr) == len(y_arr) == len(z_arr) == len(dx_arr) == len(dy_arr) == len(dz_arr)):
        raise ValueError("x, y, z, dx, dy, and dz must have compatible sizes")
    gap_value = float(np.clip(gap, 0.0, 0.95))
    maybe_color = np.asarray(c if c is not None else color)
    if (
        c is not None
        and maybe_color.ndim == 1
        and len(maybe_color) == len(x_arr)
        and not isinstance(c, str)
        and np.issubdtype(maybe_color.dtype, np.number)
    ):
        bar_colors = _colormap_values(maybe_color, cmap=cmap, vmin=vmin, vmax=vmax)
    elif c is not None and maybe_color.ndim == 2 and maybe_color.shape == (len(x_arr), 4):
        bar_colors = maybe_color.astype(np.float32, copy=False)
    else:
        rgba = list(_normalize_rgba(c if c is not None else color, n=None))
        bar_colors = np.tile(np.asarray(rgba, dtype=np.float32), (len(x_arr), 1))
    if alpha is not None:
        bar_colors = bar_colors.copy()
        bar_colors[:, 3] *= float(alpha)

    if shape == "hex":
        verts, bar_color_arr, idx_arr, edge_verts_arr = _bar3d_hex_geometry(
            x_arr, y_arr, z_arr, dx_arr, dy_arr, dz_arr, bar_colors, gap_value
        )
    else:
        verts, bar_color_arr, idx_arr, edge_verts_arr = _bar3d_box_geometry(
            x_arr, y_arr, z_arr, dx_arr, dy_arr, dz_arr, bar_colors, gap_value
        )
    verts = verts.copy()
    verts[:, 2] *= float(scale_z)
    layer = _add_3d_layer(
        verts,
        colors=bar_color_arr,
        indices=idx_arr,
        primitive="triangles",
        layer_type="bars3d",
        label=label,
        elev=elev,
        azim=azim,
        metadata={
            "artist": "bar3d",
            "shape": shape,
            "scale_z": scale_z,
            "gap": gap_value,
            "cmap": cmap if c is not None else None,
            "ssao": bool(ssao) if ssao is not None else False,
            "ssao_strength": 0.45 if ssao_strength is None else float(ssao_strength),
        },
    )
    artists = [layer]
    if edge_color is not None and len(edge_verts_arr):
        edge_verts = edge_verts_arr.copy()
        edge_verts[:, 2] *= float(scale_z)
        edge_layer = _add_3d_layer(
            edge_verts,
            primitive="lines",
            layer_type="wireframe3d",
            label=f"{label} edges" if label else "",
            elev=elev,
            azim=azim,
            color=edge_color,
            metadata={"artist": "bar3d_edges", "shape": shape, "parent": layer.layer_id},
        )
        edge_layer.style.line_width = float(edge_width)
        artists.append(edge_layer)
    return artists


def arrow(
    x,
    y,
    dx,
    dy,
    *,
    width: float = 1.0,
    head_width: float = 0.12,
    head_length: float = 0.18,
    color: ColorLike = "k",
    alpha: Optional[float] = None,
    label: Optional[str] = None,
    **kwargs,
):
    rgba = list(_normalize_rgba(color, n=None))
    if alpha is not None:
        rgba[3] *= float(alpha)
    artists = plot([x, x + dx], [y, y + dy], color=tuple(rgba), linewidth=width, label=label)
    length = float(np.hypot(dx, dy))
    if length > 1e-12:
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        tip = np.array([x + dx, y + dy], dtype=np.float32)
        base = tip - head_length * np.array([ux, uy], dtype=np.float32)
        verts = np.array(
            [
                tip,
                base + 0.5 * head_width * np.array([px, py], dtype=np.float32),
                base - 0.5 * head_width * np.array([px, py], dtype=np.float32),
            ],
            dtype=np.float32,
        )
        add_patch(verts, mode="triangles", face_color=tuple(rgba), edge_color=tuple(rgba))
        artists.append(_get_or_create_plot().scene.layers[-1])
    for artist in artists:
        artist.metadata["artist"] = "arrow"
    return artists


def quiver(
    x,
    y,
    u,
    v,
    *,
    color: ColorLike = "k",
    scale: float = 1.0,
    width: float = 1.0,
    head_width: float = 0.08,
    head_length: float = 0.12,
    label: Optional[str] = None,
    **kwargs,
):
    x_arr = _as_float_array(x, name="x").ravel()
    y_arr = _as_float_array(y, name="y").ravel()
    u_arr = _as_float_array(u, name="u").ravel()
    v_arr = _as_float_array(v, name="v").ravel()
    if not (len(x_arr) == len(y_arr) == len(u_arr) == len(v_arr)):
        raise ValueError("x, y, u, and v must have the same size")
    rgba = tuple(_normalize_rgba(color, n=None))
    N = len(x_arr)
    artists = []
    plot_obj = _get_or_create_plot()

    if N == 0:
        return artists

    dx = u_arr * scale
    dy = v_arr * scale

    # --- batch all shafts into one polyline with NaN separators ---
    # Layout: [start0, end0, NaN, start1, end1, NaN, ..., startN, endN]  (3N-1 rows)
    shaft_pts = np.empty((3 * N - 1, 2), dtype=np.float32)
    shaft_pts[0::3, 0] = x_arr
    shaft_pts[0::3, 1] = y_arr
    shaft_pts[1::3, 0] = x_arr + dx
    shaft_pts[1::3, 1] = y_arr + dy
    if N > 1:
        shaft_pts[2::3] = np.nan  # N-1 separators; NaN segments are invisible on GPU
    plot_obj.add_line_strip(shaft_pts[:, 0], shaft_pts[:, 1], rgba, width=width, label=label)
    shaft_layer = plot_obj.scene.layers[-1]
    shaft_layer.metadata["artist_group"] = "quiver"
    artists.append(shaft_layer)

    # --- batch all arrowheads into one triangles patch ---
    lengths = np.hypot(dx, dy)
    valid = lengths > 1e-12
    if valid.any():
        dx_v, dy_v, ln_v = dx[valid], dy[valid], lengths[valid]
        xv, yv = x_arr[valid], y_arr[valid]
        ux, uy = dx_v / ln_v, dy_v / ln_v
        px, py = -uy, ux
        tip_x = xv + dx_v
        tip_y = yv + dy_v
        base_x = tip_x - head_length * ux
        base_y = tip_y - head_length * uy
        lx = base_x + 0.5 * head_width * px
        ly = base_y + 0.5 * head_width * py
        rx = base_x - 0.5 * head_width * px
        ry = base_y - 0.5 * head_width * py
        M = int(valid.sum())
        head_verts = np.empty((M * 3, 2), dtype=np.float32)
        head_verts[0::3, 0] = tip_x
        head_verts[0::3, 1] = tip_y
        head_verts[1::3, 0] = lx
        head_verts[1::3, 1] = ly
        head_verts[2::3, 0] = rx
        head_verts[2::3, 1] = ry
        head_indices = np.arange(M * 3, dtype=np.uint32)
        add_patch(
            head_verts, indices=head_indices, mode="triangles", face_color=rgba, edge_color=rgba
        )
        head_layer = plot_obj.scene.layers[-1]
        head_layer.metadata["artist_group"] = "quiver"
        artists.append(head_layer)

    _set_dirty(plot_obj)
    return artists


def quiver3d(
    x,
    y,
    z,
    u,
    v,
    w,
    *,
    color: ColorLike = "C0",
    scale: float = 1.0,
    linewidth: float = 0.8,
    head_length: float = 0.18,
    head_width: float = 0.09,
    normalize: bool = False,
    elev: float = 30.0,
    azim: float = -60.0,
    label: Optional[str] = None,
    **kwargs,
):
    x_arr = _as_float_array(x, name="x").ravel()
    y_arr = _as_float_array(y, name="y").ravel()
    z_arr = _as_float_array(z, name="z").ravel()
    u_arr = _as_float_array(u, name="u").ravel()
    v_arr = _as_float_array(v, name="v").ravel()
    w_arr = _as_float_array(w, name="w").ravel()
    if not (len(x_arr) == len(y_arr) == len(z_arr) == len(u_arr) == len(v_arr) == len(w_arr)):
        raise ValueError("x, y, z, u, v, and w must have the same size")

    origins = np.column_stack([x_arr, y_arr, z_arr]).astype(np.float32)
    vecs = np.column_stack([u_arr, v_arr, w_arr]).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1)
    safe_norms = np.maximum(norms, 1e-12)
    dirs = vecs / safe_norms[:, None]
    lengths = np.ones_like(norms) if normalize else norms
    tips = origins + dirs * (lengths * float(scale))[:, None]

    segments: list[np.ndarray] = []
    for origin, tip, direction, length in zip(origins, tips, dirs, lengths):
        if length <= 1e-12:
            continue
        segments.append(origin)
        segments.append(tip)

        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(direction, ref))) > 0.92:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        side = np.cross(direction, ref)
        side /= max(float(np.linalg.norm(side)), 1e-12)
        head_base = tip - direction * float(head_length) * float(scale)
        left = head_base + side * float(head_width) * float(scale)
        right = head_base - side * float(head_width) * float(scale)
        segments.extend([tip, left, tip, right])

    if not segments:
        segments = [np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)]

    layer = _add_3d_layer(
        np.asarray(segments, dtype=np.float32),
        primitive="lines",
        layer_type="wireframe3d",
        label=label,
        elev=elev,
        azim=azim,
        color=color,
        metadata={
            "artist": "quiver3d",
            "vector_count": int(len(origins)),
            "normalize": bool(normalize),
        },
    )
    layer.style.line_width = float(linewidth)
    return [layer]


def annotate(
    text_value: str,
    xy,
    xytext=None,
    arrowprops: Optional[dict[str, Any]] = None,
    fontsize: int = 12,
    color: ColorLike = "k",
    **kwargs,
):
    tx, ty = xy if xytext is None else xytext
    text_layer = text(tx, ty, text_value, fontsize=fontsize, color=color)
    text_layer.scene.layers[-1].metadata["artist"] = "annotate_text"
    artists = [text_layer.scene.layers[-1]]
    if xytext is not None and arrowprops is not None:
        props = dict(arrowprops)
        arr_color = props.pop("color", color)
        artists.extend(
            arrow(
                tx, ty, float(xy[0]) - float(tx), float(xy[1]) - float(ty), color=arr_color, **props
            )
        )
    return artists


def hlines(
    y,
    xmin,
    xmax,
    colors: ColorLike = "k",
    linestyles: str = "-",
    label: Optional[str] = None,
    linewidth: float = 1.0,
    **kwargs,
):
    y_arr = np.atleast_1d(_as_float_array(y, name="y"))
    xmin_arr = np.broadcast_to(np.asarray(xmin, dtype=np.float32), y_arr.shape)
    xmax_arr = np.broadcast_to(np.asarray(xmax, dtype=np.float32), y_arr.shape)
    artists = []
    for idx, (yy, x0, x1) in enumerate(zip(y_arr, xmin_arr, xmax_arr)):
        artists.extend(
            plot(
                [x0, x1],
                [yy, yy],
                color=colors,
                linestyle=linestyles,
                linewidth=linewidth,
                label=label if idx == 0 else None,
            )
        )
        artists[-1].metadata["artist"] = "hline"
    return artists


def vlines(
    x,
    ymin,
    ymax,
    colors: ColorLike = "k",
    linestyles: str = "-",
    label: Optional[str] = None,
    linewidth: float = 1.0,
    **kwargs,
):
    x_arr = np.atleast_1d(_as_float_array(x, name="x"))
    ymin_arr = np.broadcast_to(np.asarray(ymin, dtype=np.float32), x_arr.shape)
    ymax_arr = np.broadcast_to(np.asarray(ymax, dtype=np.float32), x_arr.shape)
    artists = []
    for idx, (xx, y0, y1) in enumerate(zip(x_arr, ymin_arr, ymax_arr)):
        artists.extend(
            plot(
                [xx, xx],
                [y0, y1],
                color=colors,
                linestyle=linestyles,
                linewidth=linewidth,
                label=label if idx == 0 else None,
            )
        )
        artists[-1].metadata["artist"] = "vline"
    return artists


def axhline(
    y: float = 0.0,
    xmin: float = 0.0,
    xmax: float = 1.0,
    color: ColorLike = "k",
    linestyle: str = "-",
    linewidth: float = 1.0,
    label: Optional[str] = None,
    **kwargs,
):
    left, right = _get_or_create_plot().get_xlim()
    x0 = left + (right - left) * float(xmin)
    x1 = left + (right - left) * float(xmax)
    artists = hlines(
        float(y), x0, x1, colors=color, linestyles=linestyle, linewidth=linewidth, label=label
    )
    for artist in artists:
        artist.metadata.update({"guide": True, "artist": "axhline"})
    return artists[0] if artists else None


def axvline(
    x: float = 0.0,
    ymin: float = 0.0,
    ymax: float = 1.0,
    color: ColorLike = "k",
    linestyle: str = "-",
    linewidth: float = 1.0,
    label: Optional[str] = None,
    **kwargs,
):
    bottom, top = _get_or_create_plot().get_ylim()
    y0 = bottom + (top - bottom) * float(ymin)
    y1 = bottom + (top - bottom) * float(ymax)
    artists = vlines(
        float(x), y0, y1, colors=color, linestyles=linestyle, linewidth=linewidth, label=label
    )
    for artist in artists:
        artist.metadata.update({"guide": True, "artist": "axvline"})
    return artists[0] if artists else None


def axline(
    xy1,
    xy2=None,
    *,
    slope: Optional[float] = None,
    color: ColorLike = "k",
    linestyle: str = "-",
    linewidth: float = 1.0,
    label: Optional[str] = None,
    **kwargs,
):
    x0, y0 = map(float, xy1)
    if xy2 is not None:
        x1, y1 = map(float, xy2)
        if abs(x1 - x0) < 1e-12:
            return axvline(x0, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
        slope = (y1 - y0) / (x1 - x0)
    if slope is None:
        raise ValueError("axline requires xy2 or slope")
    left, right = _get_or_create_plot().get_xlim()
    y_left = y0 + float(slope) * (left - x0)
    y_right = y0 + float(slope) * (right - x0)
    artists = plot(
        [left, right],
        [y_left, y_right],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )
    artists[-1].metadata.update({"guide": True, "artist": "axline", "slope": float(slope)})
    return artists[-1]


def step(x, y, *args, where: str = "pre", **kwargs):
    x_arr = _as_float_array(x, ndim=1, name="x")
    y_arr = _as_float_array(y, ndim=1, name="y")
    if len(x_arr) != len(y_arr):
        raise ValueError("x and y must have the same length")
    if where not in {"pre", "post", "mid"}:
        raise ValueError("where must be 'pre', 'post', or 'mid'")
    if len(x_arr) < 2:
        return plot(x_arr, y_arr, *args, **kwargs)

    if where == "post":
        xs = np.repeat(x_arr, 2)[1:]
        ys = np.repeat(y_arr, 2)[:-1]
    elif where == "pre":
        xs = np.repeat(x_arr, 2)[:-1]
        ys = np.repeat(y_arr, 2)[1:]
    else:
        mid = 0.5 * (x_arr[:-1] + x_arr[1:])
        xs = np.r_[x_arr[0], np.repeat(mid, 2), x_arr[-1]]
        ys = np.repeat(y_arr, 2)
    artists = plot(xs, ys, *args, **kwargs)
    for artist in artists:
        artist.metadata["artist"] = "step"
        artist.metadata["where"] = where
    return artists


def errorbar(
    x,
    y,
    yerr=None,
    xerr=None,
    fmt: str = "",
    ecolor: Optional[ColorLike] = None,
    elinewidth: float = 1.0,
    capsize: float = 0.0,
    label: Optional[str] = None,
    **kwargs,
):
    x_arr = _as_float_array(x, ndim=1, name="x")
    y_arr = _as_float_array(y, ndim=1, name="y")
    if len(x_arr) != len(y_arr):
        raise ValueError("x and y must have the same length")
    artists = []
    err_color = ecolor if ecolor is not None else kwargs.get("color", "k")
    if yerr is not None:
        yerr_arr = np.broadcast_to(np.asarray(yerr, dtype=np.float32), y_arr.shape)
        for xx, yy, ee in zip(x_arr, y_arr, yerr_arr):
            artists.extend(vlines(xx, yy - ee, yy + ee, colors=err_color, linewidth=elinewidth))
            if capsize:
                artists.extend(
                    hlines(
                        [yy - ee, yy + ee],
                        xx - capsize,
                        xx + capsize,
                        colors=err_color,
                        linewidth=elinewidth,
                    )
                )
    if xerr is not None:
        xerr_arr = np.broadcast_to(np.asarray(xerr, dtype=np.float32), x_arr.shape)
        for xx, yy, ee in zip(x_arr, y_arr, xerr_arr):
            artists.extend(hlines(yy, xx - ee, xx + ee, colors=err_color, linewidth=elinewidth))
            if capsize:
                artists.extend(
                    vlines(
                        [xx - ee, xx + ee],
                        yy - capsize,
                        yy + capsize,
                        colors=err_color,
                        linewidth=elinewidth,
                    )
                )
    if fmt not in (None, "", "none", "None"):
        artists.extend(plot(x_arr, y_arr, fmt, label=label, **kwargs))
    else:
        artists.append(scatter(x_arr, y_arr, label=label, **kwargs))
    for artist in artists:
        artist.metadata["artist_group"] = "errorbar"
    return artists


def stem(
    x,
    y=None,
    linefmt: str = "C0-",
    markerfmt: str = "C0o",
    basefmt: str = "k-",
    bottom: float = 0.0,
    label: Optional[str] = None,
):
    if y is None:
        y_arr = _as_float_array(x, ndim=1, name="y")
        x_arr = np.arange(len(y_arr), dtype=np.float32)
    else:
        x_arr = _as_float_array(x, ndim=1, name="x")
        y_arr = _as_float_array(y, ndim=1, name="y")
    artists = []
    color = _parse_plot_format(linefmt).get("color", "b")
    artists.extend(vlines(x_arr, bottom, y_arr, colors=color, linewidth=1.0, label=label))
    marker_style = _parse_plot_format(markerfmt)
    artists.append(
        scatter(
            x_arr,
            y_arr,
            color=marker_style.get("color", color),
            marker=marker_style.get("marker", "o"),
        )
    )
    artists.extend(plot([float(np.min(x_arr)), float(np.max(x_arr))], [bottom, bottom], basefmt))
    for artist in artists:
        artist.metadata["artist_group"] = "stem"
    return artists


def text(
    x: float,
    y: float,
    s: str,
    fontsize: int = 12,
    color: ColorLike = (0.0, 0.0, 0.0, 1.0),
    label: Optional[str] = None,
):
    """
    Add text annotation.
    """
    plot_obj = _get_or_create_plot()

    rgba = _normalize_rgba(color, n=None)
    # backend may ignore fontsize/color for now, but keep API stable
    plot_obj.add_text(float(x), float(y), str(s), fontsize=int(fontsize), color=rgba, label=label)
    _set_dirty(plot_obj)
    return plot_obj


def add_patch(
    vertices: Union[np.ndarray, Sequence],
    indices: Optional[np.ndarray] = None,
    mode: str = "strip",
    face_color: Optional[ColorLike] = None,
    edge_color: Optional[ColorLike] = None,
    label: Optional[str] = None,
):
    """
    Add a geometric patch (polygon, strip, etc.) to the plot.
    """
    plot_obj = _get_or_create_plot()

    verts = _as_float_array(vertices, ndim=2, name="vertices")
    f_col = _normalize_rgba(face_color, n=None) if face_color is not None else None
    e_col = _normalize_rgba(edge_color, n=None) if edge_color is not None else None

    plot_obj.add_patch(
        verts,
        indices=indices,
        mode=mode,
        face_color=tuple(f_col) if f_col is not None else None,
        edge_color=tuple(e_col) if e_col is not None else None,
        label=label,
    )
    _set_dirty(plot_obj)
    return plot_obj


# ------------------------------------------------------------------
# View / styling / policies
# ------------------------------------------------------------------


def title(s: str) -> None:
    plot = _get_or_create_plot()
    _set_title(plot, s)


def xlabel(s: str) -> None:
    plot = _get_or_create_plot()
    plot.xlabel = str(s)
    _set_dirty(plot)


def ylabel(s: str) -> None:
    plot = _get_or_create_plot()
    plot.ylabel = str(s)
    _set_dirty(plot)


def zlabel(s: str) -> None:
    plot = _get_or_create_plot()
    plot.zlabel = str(s)
    _set_dirty(plot)


def ssao(enabled: bool = True, strength: float = 0.45, radius: float = 1.0) -> None:
    """Enable/disable lightweight 3D ambient occlusion shading."""
    plot = _get_or_create_plot()
    plot.options.visual.ssao.enabled = bool(enabled)
    plot.options.visual.ssao.strength = float(strength)
    plot.options.visual.ssao.radius = float(radius)
    _set_dirty(plot)


def grid(visible: bool = True, **kwargs) -> None:
    plot = _get_or_create_plot()
    plot.grid_visible = bool(visible)
    if hasattr(plot.options, "show_grid"):
        plot.options.show_grid = bool(visible)
    _set_dirty(plot)


def legend(*args, max_items: Optional[int] = None, **kwargs):
    plot = _get_or_create_plot()
    labels = []
    seen = set()
    for layer in plot.scene.layers:
        label = getattr(layer, "label", "")
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    if max_items is not None and len(labels) > int(max_items):
        shown = labels[: int(max_items)]
        shown.append(f"+{len(labels) - int(max_items)} more layers")
        labels = shown
    plot.legend_labels = labels
    return labels


def xlim(
    left: Optional[float] = None, right: Optional[float] = None
) -> Optional[Tuple[float, float]]:
    """
    Get or set the x-limits of the current axes.
    """
    plot = _get_or_create_plot()
    if left is None and right is None:
        return plot.get_xlim()

    # Handle single tuple argument like matplotlib
    if left is not None and right is None and isinstance(left, (tuple, list, np.ndarray)):
        left, right = left[0], left[1]

    plot.set_view(xlim=(left, right))
    _set_dirty(plot)
    return (left, right)


def ylim(
    bottom: Optional[float] = None, top: Optional[float] = None
) -> Optional[Tuple[float, float]]:
    """
    Get or set the y-limits of the current axes.
    """
    plot = _get_or_create_plot()
    if bottom is None and top is None:
        return plot.get_ylim()

    # Handle single tuple argument like matplotlib
    if bottom is not None and top is None and isinstance(bottom, (tuple, list, np.ndarray)):
        bottom, top = bottom[0], bottom[1]

    plot.set_view(ylim=(bottom, top))
    _set_dirty(plot)
    return (bottom, top)


def axis(
    mode: Union[str, Tuple[float, float, float, float]] = "auto",
) -> Optional[Tuple[float, float, float, float]]:
    """
    Supported:
        axis("auto")
        axis("tight")
        axis("reset")
        axis((xmin, xmax, ymin, ymax))
    """
    plot = _get_or_create_plot()

    if isinstance(mode, str):
        m = mode.lower()
        if m in ("auto", "tight"):
            if _call_if_exists(plot, ("autoscale", "auto_view", "fit_view")) is None:
                raise AttributeError("Backend does not expose autoscale()/fit_view()")
            _set_dirty(plot)
            return None
        if m in ("reset", "home"):
            if _call_if_exists(plot, ("reset_view", "home_view")) is None:
                plot.set_view(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))  # Absolute reset fallback
            _set_dirty(plot)
            return None
        raise ValueError(f"unsupported axis mode: {mode}")

    if len(mode) != 4:
        raise ValueError("axis tuple must be (xmin, xmax, ymin, ymax)")

    xmin, xmax, ymin, ymax = map(float, mode)
    plot.set_view(xlim=(xmin, xmax), ylim=(ymin, ymax))
    _set_dirty(plot)
    return (xmin, xmax, ymin, ymax)


def autoscale(enable: bool = True, axis: str = "both", tight: Optional[bool] = None) -> None:
    """
    Autoscale the view to fit data.
    """
    plot = _get_or_create_plot()

    # Handle tight parameter
    padding = 0.0 if tight else 0.05

    if enable:
        plot.autoscale(axes=axis, padding=padding)
        _set_dirty(plot)


def reset_view() -> None:
    axis("reset")


def home() -> None:
    """Home view (alias for reset_view)"""
    reset_view()


def set_global_alpha(alpha: float) -> None:
    plot = _get_or_create_plot()
    if hasattr(plot, "set_global_alpha"):
        plot.set_global_alpha(float(alpha))
    else:
        if hasattr(plot, "global_alpha"):
            plot.global_alpha = float(alpha)
        _set_dirty(plot)


def alpha(value: float) -> None:
    set_global_alpha(value)


def set_lod(enabled: bool = True, max_lines_per_px: int = 8) -> None:
    plot = _get_or_create_plot()

    if hasattr(plot, "set_lod"):
        plot.set_lod(enabled=enabled, max_lines_per_px=max_lines_per_px)
        return
    plot.enable_subsample = bool(enabled)
    plot.max_lines_per_px = max(1, int(max_lines_per_px))
    _set_dirty(plot)


def lod(enabled: bool = True, max_lines_per_px: int = 8) -> None:
    set_lod(enabled=enabled, max_lines_per_px=max_lines_per_px)


def blending(mode: BlendMode = "auto") -> None:
    plot = _get_or_create_plot()
    _set_blending(plot, mode)


def density(enabled: bool = True) -> None:
    plot = _get_or_create_plot()
    _set_density(plot, enabled)


def density_gain(value: float) -> None:
    """Set the gain/factor for density plots."""
    plot = _get_or_create_plot()
    if hasattr(plot, "set_density_gain"):
        plot.set_density_gain(value)
    _set_dirty(plot)


def hud(enabled: bool = True) -> None:
    plot = _get_or_create_plot()
    _set_hud(plot, enabled)


# ------------------------------------------------------------------
# Analysis / export / execution
# ------------------------------------------------------------------


def stats(scope: str = "visible"):
    plot = _get_or_create_plot()
    if not hasattr(plot, "get_summary_stats"):
        raise AttributeError("Backend does not expose get_summary_stats()")

    s = plot.get_summary_stats(scope)
    print(f"\n--- Statistics ({scope}) ---")
    for k, v in s.items():
        if isinstance(v, float):
            print(f"{k:12}: {v:.6f}")
        else:
            print(f"{k:12}: {v}")
    return s


def profile(name: str) -> None:
    """
    Apply a performance profile: 'extreme', 'performance', 'balanced', 'quality'.
    """
    plot = _get_or_create_plot()
    if hasattr(plot, "set_profile"):
        plot.set_profile(name)
    _set_dirty(plot)


def export(filename: Optional[str] = None, scale: float = 2.0):
    plot = _get_or_create_plot()
    fname = filename or f"plot_{int(time.time())}.png"
    if hasattr(plot, "savefig"):
        plot.savefig(fname, scale=scale)
    else:
        # Fallback
        if (
            _call_if_exists(plot, ("save_current_view", "export_high_res"), fname, scale=scale)
            is None
        ):
            raise AttributeError("Backend does not expose export functions")


def savefig(filename: str, density: Optional[bool] = None, scale: float = 2.0):
    plot = _get_or_create_plot()

    if density is not None:
        _set_density(plot, density)

    # If no GL window exists (plt.show() hasn't been called), fall back to the
    # matplotlib-based preview renderer — same path the gallery runner uses.
    # Attempting to create a headless GLFW context here crashes on macOS when
    # Python is not running as a proper app bundle.
    if getattr(plot, "window", None) is None:
        from glplot.utils.preview import render_preview as _render_preview

        _render_preview(plot, filename, scale)
        return

    if hasattr(plot, "savefig"):
        plot.savefig(filename, scale=scale)
        return

    _call_if_exists(plot, ("save_current_view",), filename, scale=scale)


def show(
    density: Optional[bool] = None,
    *,
    test_mode: bool = False,
) -> None:
    plot = _get_or_create_plot()

    if density is not None:
        _set_density(plot, density)

    if hasattr(plot, "_is_test_mode"):
        plot._is_test_mode = bool(test_mode)

    plot.run()


# ------------------------------------------------------------------
# Convenience aliases
# ------------------------------------------------------------------

lineplot = lines
points = scatter


# ------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------


@atexit.register
def _cleanup_pyplot_state():
    global _CURRENT_PLOT
    _CURRENT_PLOT = None
    _ALL_PLOTS.clear()
