from __future__ import annotations

import atexit
import datetime as _dt
import sys as _sys
import time
import warnings
from numbers import Integral
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from .core import layout as _layout
from .core.layers import BaseLayer, FractalLayer, FunctionLayer, Layer3D
from .engine import GPULinePlot

ColorLike = Union[
    Tuple[float, float, float, float],
    Sequence[float],
    np.ndarray,
]

ArrayLike = Union[Sequence[float], np.ndarray]

BlendMode = Literal["auto", "on", "off"]


# ------------------------------------------------------------------
# Global pyplot-like state
# ------------------------------------------------------------------

_CURRENT_PLOT: Optional[GPULinePlot] = None
_ALL_PLOTS: list[GPULinePlot] = []

#: The colormap `set_cmap()` last chose, or None for "nobody has said". It is a *fallback*,
#: not a default: an explicit `cmap=` on a call always wins, and when this is None each
#: function keeps the default it has always had (`hist2d` is magma, `scatter` is viridis).
#: matplotlib spells the same idea `rcParams["image.cmap"]`, which is what `viridis()`,
#: `jet()` and the rest of the one-word shortcuts write. See `_resolve_cmap`.
_CURRENT_CMAP: Optional[str] = None

#: The last layer a colormap was applied to -- matplotlib's "current image", what `gci()`
#: returns and what `clim()` and a bare `set_cmap()` act on. Held as a layer rather than an
#: index because the Scene panel can reorder and delete, and an index would then point at
#: whatever slid into the slot.
_CURRENT_MAPPABLE: Optional[BaseLayer] = None


#: Figures that `figure(num=...)` can hand back, keyed by that identifier. matplotlib's
#: `figure` is get-or-create on this key, which is what makes the "re-run the cell and it
#: redraws in the same window" idiom work instead of leaking a window per run. Only
#: figures given a `num` are in here -- `figure()` with none is always a fresh one, so it
#: cannot be reached by name and must not pin memory after `close()`.
_FIGURES_BY_NUM: dict = {}

#: Fallback property cycle (tab10) for the case where rcParams carries no usable one.
_FALLBACK_COLOR_CYCLE = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9")


def _color_cycle() -> tuple:
    """The colours ``plot()`` walks when the caller names none.

    Read from ``rcParams["axes.prop_cycle"]`` on every call rather than frozen at import,
    because that is what makes ``style.use("ggplot")`` actually recolour a GLPlot figure.
    A constant here would leave the styling API accepted and inert -- the plot would still
    come out in tab10 no matter which style was selected.
    """
    from matplotlib import rcParams

    try:
        colors = rcParams["axes.prop_cycle"].by_key().get("color")
    except Exception:
        colors = None
    return tuple(colors) if colors else _FALLBACK_COLOR_CYCLE


def _default_linewidth() -> float:
    """``rcParams["lines.linewidth"]``, read live so a style change reaches new lines."""
    from matplotlib import rcParams

    try:
        return float(rcParams["lines.linewidth"])
    except Exception:
        return 1.5


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _as_float_array(x: ArrayLike, ndim: Optional[int] = None, name: str = "array") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {arr.ndim}")
    return np.ascontiguousarray(arr)


#: ``{(id(figure), panel_index, axis_name): {label: position}}`` -- the category ->
#: integer-position mapping :func:`_coerce_axis_values` accumulates for a string axis, one
#: map per panel per side, so a second ``plot()``/``scatter()`` call against the same axis
#: extends the same ordering rather than starting over (matplotlib's own categorical
#: converter is exactly this: one persistent map per ``Axis`` instance). Figures live for
#: the process (``_ALL_PLOTS`` never drops one), so keying on ``id()`` cannot collide with
#: a garbage-collected figure's reused id.
_CATEGORY_MAPS: dict = {}


def _coerce_axis_values(values: ArrayLike, axis_name: str, func: str) -> ArrayLike:
    """Convert datetime-like or string-categorical input into GLPlot's numeric axis space.

    matplotlib gives every ``Axis`` a persistent unit converter, so a script can plot dates
    or category strings directly; GLPlot's axes are plain floats and ``_as_float_array``
    raises on either (`float() argument must be a string or a real number, not
    'datetime.date'` / `could not convert string to float`). Numeric input (the overwhelming
    common case) is returned untouched -- this only inspects dtype, never values, so it
    costs nothing on the fast path.

    Dates are mapped through ``matplotlib.dates.date2num``, so relative order and spacing
    on the axis are exact; there is no live date-tick formatter yet, so the ticks show that
    number rather than a formatted date, and this warns once per axis side to say so.

    Category strings map to ``0, 1, 2, ...`` in first-appearance order, accumulated across
    repeated calls against the same panel and axis side, with that axis's tick labels set
    to match -- so ``plot(['a', 'b', 'c'], [1, 4, 2])`` shows the letters, not `0, 1, 2`, on
    the axis, the same as matplotlib's own default categorical axis.
    """
    arr = np.asarray(values)
    if arr.size == 0 or arr.dtype.kind not in ("U", "S", "O", "M"):
        return values

    is_datetime = np.issubdtype(arr.dtype, np.datetime64) or (
        arr.dtype == object and all(isinstance(v, (_dt.date, _dt.datetime)) for v in arr.ravel())
    )
    if is_datetime:
        import matplotlib.dates as _mdates

        _warn_unsupported_call(
            f"{func}({axis_name}=dates)",
            "plots date values at their correct relative positions "
            "(matplotlib.dates.date2num), but the tick labels show that number rather than "
            "a formatted date -- GLPlot has no live date-tick formatter yet",
        )
        return _mdates.date2num(arr)

    is_categorical = arr.dtype.kind in ("U", "S") or (
        arr.dtype == object and all(isinstance(v, str) for v in arr.ravel())
    )
    if is_categorical:
        plot_obj = _get_or_create_plot()
        key = (id(plot_obj), plot_obj.active_panel_index, axis_name)
        mapping = _CATEGORY_MAPS.setdefault(key, {})
        positions = []
        for v in arr.ravel():
            label = str(v)
            if label not in mapping:
                mapping[label] = float(len(mapping))
            positions.append(mapping[label])
        ordered = sorted(mapping.items(), key=lambda kv: kv[1])
        (xticks if axis_name == "x" else yticks)(
            [p for _, p in ordered], [label for label, _ in ordered]
        )
        return np.asarray(positions, dtype=np.float64).reshape(arr.shape)

    return values


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


def _next_cycle_color(plot_obj: GPULinePlot) -> str:
    idx = getattr(plot_obj, "_color_cycle_index", 0)
    cycle = _color_cycle()
    plot_obj._color_cycle_index = (idx + 1) % len(cycle)
    return cycle[idx % len(cycle)]


def _parse_plot_groups(args: tuple) -> list:
    """Split ``plot``-style varargs into ``(x, y, fmt)`` triples.

    THE reader for matplotlib's ``x, y, [fmt], x, y, [fmt], ...`` grammar: a pair of
    arrays, then an optional format string, repeated. `plot`, `fill` and anyone else who
    accepts the same grammar go through here, so the rule for where one group ends and the
    next begins lives in exactly one place and the callers cannot disagree about it.

    ``y`` is None for a lone ``plot(y)``; the caller supplies the implicit x index.
    """
    groups = []
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
        groups.append((x, y, fmt))
    return groups


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


def _plot_single(
    x: ArrayLike, y: Optional[ArrayLike] = None, fmt: Optional[str] = None, **kwargs: Any
) -> BaseLayer:
    style = _parse_plot_format(fmt)
    style.update({k: v for k, v in kwargs.items() if v is not None})
    if y is None:
        y_arr = _as_float_array(_coerce_axis_values(x, "y", "plot"), ndim=1, name="y")
        x_arr = np.arange(len(y_arr), dtype=np.float32)
    else:
        x_arr = _as_float_array(_coerce_axis_values(x, "x", "plot"), ndim=1, name="x")
        y_arr = _as_float_array(_coerce_axis_values(y, "y", "plot"), ndim=1, name="y")
    return _add_plot_primitive(x_arr, y_arr, **style)


def _project_3d(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *,
    elev: float = 30.0,
    azim: float = -60.0,
    scale_z: float = 1.0,
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


#: Above this many points, :func:`scatter` stops retaining the scalars behind ``c=``.
#:
#: Retention costs one float32 per point — 4 bytes, next to the 16 bytes/point of RGBA
#: that ``add_scatter`` already keeps for the colour VBO and the 8 bytes/point of ``pts``.
#: So it is a ~17% increase on a scatter's footprint, not a doubling, which is why the
#: cap is high rather than clever: at 20M points it bounds the retention at 80MB, and
#: below it the memory is dominated by arrays that were being kept anyway.
#:
#: The cap is a real cliff, not a soft degrade — past it ``metadata["cvalues"]`` is None
#: and ``layerops.layer_colormap_kind`` reports no colormap, so the GUI shows no picker
#: rather than a picker that cannot work.
CVALUES_RETAIN_MAX_POINTS = 20_000_000


def _retained_cvalues(values: np.ndarray) -> Optional[np.ndarray]:
    """The copy of ``c``'s scalars to keep on the layer, or None past the size cap.

    A **copy**, and float32: the caller's array is theirs to mutate afterwards, and a
    retained view would make a later in-place edit silently rewrite the colormap source
    without touching the colours actually on screen.
    """
    if len(values) > CVALUES_RETAIN_MAX_POINTS:
        return None
    return np.array(values, dtype=np.float32, copy=True)


def _colormap_values(
    values: ArrayLike,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    norm: Optional[Any] = None,
) -> np.ndarray:
    """Sample ``cmap`` at each value, after mapping the values into 0..1.

    The 0..1 mapping is `_normalize_cvalues`, so ``norm`` means here what it means for
    ``scatter`` -- a LogNorm over data spanning decades is the difference between a
    readable image and one solid colour, and it would be a lie to accept the keyword and
    ramp linearly anyway.
    """
    from matplotlib import colormaps

    arr = np.asarray(values, dtype=np.float32)
    normed = _normalize_cvalues(arr, norm, vmin, vmax)
    return np.asarray(colormaps.get_cmap(_resolve_cmap(cmap, "viridis"))(normed), dtype=np.float32)


def set_layer_compositing(
    layer: BaseLayer,
    *,
    blend: Optional[Union[str, "BlendMode"]] = None,
    depth_write: Optional[bool] = None,
    auto_alpha: Optional[float] = None,
) -> BaseLayer:
    """Set how ``layer`` merges with what is already on screen. Returns the layer.

    The same three knobs the 3D verbs take as ``blend=``, ``depth_write=`` and
    ``auto_alpha=``, for a layer that already exists::

        cloud = gplt.volume3d(x, y, z, energy)
        gplt.set_layer_compositing(cloud, blend="additive", auto_alpha=0.9)

    An omitted argument is left alone, so this can be called repeatedly to change one thing;
    pass ``blend="figure"`` to hand a layer back to the figure-wide mode, and
    ``auto_alpha=0`` to hand it back its own alpha.

    Args:
        layer: The layer to change. Only :class:`~glplot.core.layers.Layer3D` layers are
            read by a renderer today -- the 2D pass still takes one blend mode for the whole
            frame -- so setting these on a 2D layer stores them and changes nothing.
        blend: ``"alpha"``, ``"additive"``, ``"subtractive"``, ``"screen"``, ``"off"``, a
            :class:`~glplot.options.BlendMode`, or ``"figure"`` to inherit.
        depth_write: Whether the layer occludes what is drawn after it. ``None`` (the
            default) leaves the renderer's own decision in place.
        auto_alpha: Target opacity for a covered pixel, or 0 to use the layer's alpha.

    Raises:
        ValueError: If ``blend`` is not one of the mode names.
    """
    from .options import BlendMode

    style = layer.style
    if blend is not None:
        if isinstance(blend, BlendMode):
            style.blend_mode = blend
        elif str(blend).strip().lower() in ("figure", "inherit", "default"):
            style.blend_mode = None
        else:
            name = str(blend).strip().upper()
            try:
                style.blend_mode = BlendMode[name]
            except KeyError:
                valid = ", ".join(m.name.lower() for m in BlendMode)
                raise ValueError(
                    f"unknown blend mode {blend!r}; expected one of {valid} (or 'figure')"
                ) from None
    if depth_write is not None:
        style.depth_write = bool(depth_write)
    if auto_alpha is not None:
        # 0 is the spelling for "off" rather than None, so a caller threading the value
        # through from a GUI or a config file does not need a separate sentinel.
        style.auto_alpha = float(auto_alpha) if float(auto_alpha) > 0.0 else None
    layer.dirty.style_dirty = True
    _set_dirty(_get_or_create_plot())
    return layer


def _add_3d_layer(
    vertices: np.ndarray,
    *,
    colors: Optional[np.ndarray] = None,
    indices: Optional[np.ndarray] = None,
    primitive: str = "points",
    layer_type: str = "scatter3d",
    label: Optional[str] = None,
    elev: float = 28.0,
    azim: float = -45.0,
    point_size: float = 3.0,
    color: Optional[ColorLike] = None,
    alpha: Optional[float] = None,
    metadata: Optional[dict[str, Any]] = None,
    sizes: Optional[np.ndarray] = None,
    blend: Optional[Union[str, "BlendMode"]] = None,
    depth_write: Optional[bool] = None,
    auto_alpha: Optional[float] = None,
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
    size_arr = None if sizes is None else np.asarray(sizes, dtype=np.float32).ravel()
    layer = Layer3D(
        verts,
        colors=np.ascontiguousarray(cols, dtype=np.float32),
        indices=idx,
        primitive=primitive,
        label=label or "",
        layer_type=layer_type,
        sizes=size_arr,
    )
    layer.style.point_size = float(point_size)
    set_layer_compositing(layer, blend=blend, depth_write=depth_write, auto_alpha=auto_alpha)
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
    color: Optional[ColorLike] = None,
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

    # Whether the caller named the colour or took the next one off the cycle. Density mode
    # tints itself with the layer's colour only in the first case; see
    # ``RendererManager.density_tint_active``.
    explicit_color = color is not None
    if color is None:
        color = _next_cycle_color(plot_obj)
    rgba = list(_normalize_rgba(color, n=None))
    if alpha is not None:
        rgba[3] *= float(alpha)

    line_width = width if width is not None else (linewidth if linewidth is not None else lw)
    line_width = _default_linewidth() if line_width is None else float(line_width)
    artists = []

    if linestyle not in (None, "", "None", "none", " "):
        plot_obj.add_line_strip(x_arr, y_arr, tuple(rgba), width=line_width, label=label)
        layer = plot_obj.scene.layers[-1]
        layer.metadata.update(
            {"linestyle": linestyle, "artist": "line", "explicit_color": explicit_color}
        )
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
        layer.metadata.update(
            {"marker": marker, "artist": "marker", "explicit_color": explicit_color}
        )
        artists.append(layer)

    _set_dirty(plot_obj)
    return artists or [plot_obj]


def get_engine() -> GPULinePlot:
    """Get the current active plot engine.

    Returns the current GPULinePlot instance, creating one if none exists.
    Useful for direct engine manipulation or introspection of internal state.
    Most users should use the pyplot API functions instead of the engine
    directly.

    Returns:
        GPULinePlot: The active plot engine instance.

    Examples:
        Get current engine:

        >>> engine = gplt.get_engine()
        >>> engine.set_view(xlim=(0, 10), ylim=(0, 10))

        Access plot properties directly:

        >>> engine = gplt.get_engine()
        >>> print(engine.title)
        >>> engine.title = "New Title"
    """
    return _get_or_create_plot()


def _set_dirty(plot: GPULinePlot) -> None:
    if hasattr(plot, "view") and hasattr(plot.view, "dirty"):
        plot.view.dirty = True
    elif hasattr(plot, "frame") and hasattr(plot.frame, "dirty_scene"):
        plot.frame.dirty_scene = True


#: The Text properties a `fontdict` can carry that GLPlot has a channel for. Everything
#: else in the dict is a real matplotlib property (family, weight, style, ...) that this
#: renderer has no way to honour, and `_merge_fontdict` reports each one rather than
#: letting a dict quietly lose half its contents.
_FONTDICT_SUPPORTED = frozenset({"fontsize", "size", "color"})


def _apply_scatter_edges(
    layer: BaseLayer,
    edgecolors: Optional[ColorLike],
    linewidths: Optional[float],
) -> None:
    """Wire ``scatter(edgecolors=, linewidths=)`` onto the layer's point outline.

    The scatter shader already draws an outline (`u_outline_*`), gated on
    `style.point_outline_enabled`; these kwargs are the matplotlib spelling of that switch.
    The gate follows ``edgecolors`` alone: matplotlib treats a linewidth with no edge
    colour as a width for an edge it is not drawing, and turning the outline on for a bare
    ``linewidths=2`` would silently ring every point in the default black.

    ``'none'`` and ``'face'`` are matplotlib's two special values, and both mean "no
    separate outline" for a renderer with no per-edge colour buffer -- 'face' would need
    the edge to track each point's own colour, which is what leaving it off already looks
    like.
    """
    if edgecolors is None:
        if linewidths is not None:
            _warn_unsupported(
                "scatter",
                {"linewidths": linewidths},
                {
                    "linewidths": "has no effect without edgecolors=, which is what turns the "
                    "point outline on"
                },
                stacklevel=4,
            )
        return
    if isinstance(edgecolors, str) and edgecolors.lower() in ("none", "face"):
        return
    layer.style.point_outline_enabled = True
    layer.style.point_outline_color = tuple(float(v) for v in _normalize_rgba(edgecolors))
    if linewidths is not None:
        layer.style.point_outline_width = float(linewidths)


def _paint_patch_mappable(
    layer: BaseLayer,
    vertex_values: np.ndarray,
    cmap: Optional[str],
    norm: Optional[Any],
    vmin: Optional[float],
    vmax: Optional[float],
    alpha: Optional[float],
) -> None:
    """Colour a per-vertex patch from a scalar per colour-vertex, and record how.

    THE painter for the patch mappables (hexbin, pcolor, tripcolor). It writes
    ``layer.colors`` and stashes, in metadata, everything a later ``clim()`` or
    ``set_cmap()`` needs to repaint: the scalar aligned one-to-one with the colour rows,
    the colormap name, and the alpha. Aligning the scalar to the *colour rows* rather than
    to the cells is what lets the re-map be uniform across three artists whose
    cell-to-vertex expansion differs (a hexagon is 7 rows, a quad 4, a triangle 3) -- see
    `_remap_patch_mappable`, which reads only these three fields and needs to know nothing
    about hexagons.
    """
    from matplotlib import colormaps

    resolved = _resolve_cmap(cmap, "viridis")
    normed = _normalize_cvalues(vertex_values, norm, vmin, vmax)
    rgba = np.asarray(colormaps.get_cmap(resolved)(normed), dtype=np.float32)
    if alpha is not None:
        rgba[:, 3] *= float(alpha)
    layer.colors = rgba
    if hasattr(layer, "dirty"):
        layer.dirty.gpu_dirty = True
    layer.metadata["_patch_cmap"] = {
        "values": np.asarray(vertex_values, dtype=np.float64),
        "cmap": resolved,
        "alpha": alpha,
        "vmin": vmin,
        "vmax": vmax,
    }


def _remap_patch_mappable(
    layer: BaseLayer,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> bool:
    """Repaint a patch mappable's colours with a new colormap or limits. True if it did.

    The counterpart to `_paint_patch_mappable`, and what makes `clim()`/`set_cmap()` reach
    a hexbin: `set_layer_colormap` (the GUI's re-mapper) knows only image and scatter-value
    layers and raises on a patch, so these mappables need their own path. Returns False for
    a patch that carries no recorded scalar (nothing to repaint), so callers can fall back.
    """
    state = layer.metadata.get("_patch_cmap") if isinstance(layer.metadata, dict) else None
    if not state:
        return False
    _paint_patch_mappable(
        layer,
        state["values"],
        cmap if cmap is not None else state["cmap"],
        None,
        vmin if vmin is not None else state["vmin"],
        vmax if vmax is not None else state["vmax"],
        state["alpha"],
    )
    return True


def _set_current_mappable(layer: BaseLayer) -> BaseLayer:
    """Record ``layer`` as matplotlib's "current image" and hand it straight back.

    Called by every function that maps scalars through a colormap, so `gci()`, `clim()`
    and a bare `set_cmap()` act on the thing the caller most recently coloured -- which is
    what those three mean by "current". Returns the layer so a call site can stay one
    expression.
    """
    global _CURRENT_MAPPABLE
    _CURRENT_MAPPABLE = layer
    return layer


def _resolve_cmap(explicit: Optional[str], default: str) -> str:
    """Which colormap a call actually uses: explicit > :func:`set_cmap` > the default.

    Three levels rather than two, so `set_cmap('jet')` reaches every plot that did not name
    a colormap -- which is the whole point of it -- without overriding one that did, and
    without silently retiring the per-function defaults GLPlot already shipped.
    """
    if explicit is not None:
        return explicit
    if _CURRENT_CMAP is not None:
        return _CURRENT_CMAP
    return default


def _normalize_cvalues(
    values: np.ndarray,
    norm: Optional[Any],
    vmin: Optional[float],
    vmax: Optional[float],
) -> np.ndarray:
    """Map scalars to the 0..1 a colormap samples, honouring matplotlib's ``norm``.

    ``norm`` replaces the linear vmin..vmax ramp outright, which is the whole reason it
    exists: a LogNorm over data spanning decades is the difference between a readable plot
    and one solid colour. It arrives either as a `matplotlib.colors.Normalize` (callable)
    or, since matplotlib 3.6, as a scale name like ``"log"``.

    The masked entries a Normalize returns for out-of-domain values (a LogNorm sees a
    zero, say) are filled rather than left masked, because the result indexes a colormap
    and a masked index would propagate into the RGBA buffer as garbage.
    """
    if norm is None:
        lo = float(np.nanmin(values)) if vmin is None else float(vmin)
        hi = float(np.nanmax(values)) if vmax is None else float(vmax)
        return np.clip((values.astype(np.float32) - lo) / max(hi - lo, 1e-12), 0.0, 1.0)

    # matplotlib raises on this rather than picking a winner, and so must we: silently
    # dropping one of two conflicting instructions is how a plot lies about its scale.
    if vmin is not None or vmax is not None:
        raise ValueError(
            "scatter(): passing a norm= together with vmin/vmax is not supported. "
            "Set the limits on the norm instead, e.g. LogNorm(vmin=..., vmax=...)."
        )

    if isinstance(norm, str):
        # Built from matplotlib's own scale registry rather than a hand-written table.
        # The table this replaced listed four names and constructed `mcolors.LogitNorm`,
        # which does not exist -- `norm='logit'` raised AttributeError from inside GLPlot.
        # Going through the registry supports every scale matplotlib knows (including
        # 'asinh'), keeps the set in step with the installed version, and gets each norm's
        # own default parameters (SymLogNorm's linthresh) from matplotlib instead of a
        # guess made here.
        from matplotlib import colors as mcolors
        from matplotlib import scale as mscale

        try:
            scale_cls = mscale._scale_mapping[norm]
        except KeyError:
            known = ", ".join(repr(k) for k in sorted(mscale._scale_mapping))
            raise ValueError(
                f"unsupported norm: {norm!r}. Expected one of {known}, or a "
                "matplotlib.colors.Normalize instance."
            ) from None
        norm = mcolors.make_norm_from_scale(scale_cls)(mcolors.Normalize)()

    if not callable(norm):
        raise TypeError(f"norm must be a scale name or a Normalize instance, got {type(norm)!r}")

    normed = np.ma.filled(np.ma.asarray(norm(values)), 0)
    normed_arr = np.asarray(normed)
    if normed_arr.dtype.kind in ("i", "u"):
        # A discrete/classed norm (matplotlib.colors.BoundaryNorm, most commonly) returns
        # raw LUT indices -- 0..N-1 for an N-colour map, not a 0..1 float -- and every
        # caller of this function feeds the result straight to a real
        # `matplotlib.colors.Colormap.__call__`, which special-cases an integer-dtype
        # array as "these are already indices" and skips its own 0..1 rescaling. Clipping
        # an index array to [0.0, 1.0] (the float branch below) crushed every index above 1
        # down to the colormap's second colour, so a 5-band BoundaryNorm classification
        # rendered as only 2 colours. Passed through unclipped and un-cast here instead --
        # BoundaryNorm's own __call__ already clips to a valid index range.
        return normed_arr
    return np.clip(normed_arr.astype(np.float32), 0.0, 1.0)


def _resolve_data_args(func: str, data: Optional[Any], *args) -> tuple:
    """Resolve matplotlib's ``data=`` indirection: a string arg means ``data[arg]``.

    ``gplt.hist('height', data=df)`` is how matplotlib addresses a labelled container --
    a DataFrame, a dict, a structured array -- and the rule is uniform across its plotting
    functions: with ``data`` given, any argument that is a string is a key into it.

    Returns the arguments unchanged when ``data`` is None, so the normal array path pays
    nothing for this. A key that is not in ``data`` raises here rather than in a numpy
    conversion three frames down, where the message would name a dtype instead of the key.
    """
    if data is None:
        return args
    resolved = []
    for arg in args:
        if not isinstance(arg, str):
            resolved.append(arg)
            continue
        try:
            resolved.append(data[arg])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"{func}(): {arg!r} is not a key in the given data=") from exc
    return tuple(resolved)


def _resolve_plot_data_args(func: str, data: Any, args: tuple) -> tuple:
    """``data=`` resolution for the variadic plotters, where a string may be a format spec.

    `_resolve_data_args` treats *every* string as a key, which is right for the fixed-arity
    functions but wrong here: ``plot('height', 'mass', 'r--', data=df)`` has two keys and a
    format string in the same argument list. matplotlib's rule is to replace a string only
    when it is actually a key in ``data`` and leave it alone otherwise, which is what lets
    ``'r--'`` through -- so a missing key is *not* an error here, it just stays a string
    and fails later as a format spec.
    """
    resolved = []
    for arg in args:
        if isinstance(arg, str):
            try:
                resolved.append(data[arg])
                continue
            except Exception:
                pass
        resolved.append(arg)
    return tuple(resolved)


def _mpl_text_arg(func: str, primary: Optional[str], legacy: Optional[str], name: str) -> str:
    """Resolve an annotation's text from matplotlib's parameter name or GLPlot's old ``s``.

    matplotlib names these ``label``/``xlabel``/``ylabel`` and they are keyword-passable
    (``plt.xlabel(xlabel="Time")`` is legal), so parity means adopting those names. GLPlot
    shipped them as ``s`` up to 0.1.3, and this is on PyPI -- a straight rename would break
    ``title(s=...)`` in released code with a TypeError that names no cause. Both spellings
    resolve here; the matplotlib one wins if somebody passes both.
    """
    if primary is not None:
        return str(primary)
    if legacy is not None:
        return str(legacy)
    raise TypeError(f"{func}() missing required argument: '{name}'")


def _merge_fontdict(
    func: str,
    fontdict: Optional[dict],
    fontsize: Optional[float],
    color: Optional[ColorLike],
) -> Tuple[Optional[float], Optional[ColorLike]]:
    """Fold matplotlib's ``fontdict`` into the explicit ``fontsize``/``color`` kwargs.

    ``fontdict`` is matplotlib's older way of passing Text properties as a dict, and both
    forms reach the same properties, so honouring one and not the other would make two
    spellings of the same call behave differently. The explicit kwarg wins on a clash --
    matplotlib's own rule, and the one a reader expects, since the kwarg is the more
    specific and more visible of the two.
    """
    if not fontdict:
        return fontsize, color
    if fontsize is None:
        fontsize = fontdict.get("fontsize", fontdict.get("size"))
    if color is None:
        color = fontdict.get("color")
    # A dict is a bag of properties, so the no-op policy has to reach inside it. Reported
    # per key, under the fontdict[key] spelling the caller actually wrote.
    unsupported = {
        f"fontdict[{key!r}]": value
        for key, value in fontdict.items()
        if key not in _FONTDICT_SUPPORTED
    }
    _warn_unsupported(func, unsupported, stacklevel=4)  # +1 frame: this helper's own.
    return fontsize, color


#: `_call_if_exists`'s "no such method" answer. It cannot report that with None, because
#: None is also what every setter it dispatches to *returns*, and the two mean opposite
#: things: "fall back" versus "done". Conflating them made `axis('auto')` run the
#: autoscale and then raise AttributeError anyway, and made every other caller apply its
#: fallback on top of a call that had already succeeded.
_MISSING = object()


def _call_if_exists(plot: GPULinePlot, method_names: Sequence[str], *args, **kwargs):
    """Call the first method of ``method_names`` that ``plot`` has, else :data:`_MISSING`.

    Callers MUST test the result against :data:`_MISSING`, never against None -- see the
    note there for what testing None costs.
    """
    for name in method_names:
        fn = getattr(plot, name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    return _MISSING


class MatplotlibCompatWarning(UserWarning):
    """A matplotlib keyword was accepted for signature parity but had no effect.

    Its own class, rather than a bare ``UserWarning``, so it can be silenced on its own::

        warnings.filterwarnings("ignore", category=gplt.MatplotlibCompatWarning)

    A caller porting a script wants these loud; a caller who has read them once and
    decided they do not care needs a filter that does not also hide every other warning
    the process emits.
    """


#: Fires once per (function, keyword) pair rather than once per call site. `warnings`'
#: own "once" registry keys on the formatted message, which for this warning already
#: carries both -- but the module default is "default", which keys on the call site too,
#: so a no-op kwarg inside a plotting loop would print on every iteration and bury the
#: signal it exists to send. See `_warn_unsupported`.
_WARNED_UNSUPPORTED: set[Tuple[str, str]] = set()


def _warn_unsupported(
    func: str,
    provided: dict,
    detail: Optional[dict] = None,
    stacklevel: int = 3,
) -> None:
    """Warn for each keyword in ``provided`` that GLPlot accepts but does not honour.

    THE enforcement point for the compat policy: a matplotlib keyword GLPlot cannot
    implement is accepted and ignored -- so a script pasted from matplotlib runs -- but it
    says so, once. Silently ignoring is the worse failure: the caller reads the argument
    back in their own source and believes it took effect, which is how ``title(fontsize=)``
    shipped as decoration. A `TypeError` would be honest but defeats the point of
    accepting the keyword at all.

    Only keywords whose value is not None are reported, so the warning tracks what the
    caller actually *asked for* rather than what the signature happens to list. ``detail``
    supplies a per-keyword explanation where "not supported" is not self-explanatory.

    ``stacklevel`` defaults to the depth for a public function calling this directly, which
    points the warning at the caller's own line. Anything dispatching through a helper must
    add its own frames, or the warning blames a line inside glplot for the user's kwarg.
    """
    for key, value in provided.items():
        if value is None:
            continue
        seen = (func, key)
        if seen in _WARNED_UNSUPPORTED:
            continue
        _WARNED_UNSUPPORTED.add(seen)
        why = (detail or {}).get(key, "is not supported by the GPU backend and was ignored")
        warnings.warn(
            f"{func}({key}=...) {why}.",
            MatplotlibCompatWarning,
            stacklevel=stacklevel,
        )


def _warn_unsupported_call(func: str, why: str, stacklevel: int = 3) -> None:
    """Warn that a whole *call* is a no-op, once, the way :func:`_warn_unsupported` does.

    The keyword-level helper reads badly for a function whose signature is empty --
    ``invert_zaxis(invert_zaxis=...)`` names a keyword that does not exist. Same registry,
    so the same "once per thing, not once per call site" rule applies and a no-op inside a
    loop does not bury the message it exists to send.
    """
    seen = (func, "")
    if seen in _WARNED_UNSUPPORTED:
        return
    _WARNED_UNSUPPORTED.add(seen)
    warnings.warn(f"{func}() {why}.", MatplotlibCompatWarning, stacklevel=stacklevel)


def _set_density(plot: GPULinePlot, enabled: bool) -> None:
    if _call_if_exists(plot, ("set_density_enabled", "set_density_mode"), enabled) is not _MISSING:
        return
    if hasattr(plot, "view") and hasattr(plot.view, "show_density"):
        plot.view.show_density = bool(enabled)
    elif hasattr(plot, "show_density"):
        plot.show_density = bool(enabled)
    _set_dirty(plot)


def _set_hud(plot: GPULinePlot, enabled: bool) -> None:
    if _call_if_exists(plot, ("set_hud_enabled",), enabled) is not _MISSING:
        return
    if hasattr(plot, "view") and hasattr(plot.view, "hud_visible"):
        plot.view.hud_visible = bool(enabled)
    _set_dirty(plot)


def _set_blending(plot: GPULinePlot, mode: BlendMode) -> None:
    # Preferred backend API
    if _call_if_exists(plot, ("set_blending_mode",), mode) is not _MISSING:
        return

    # Fallback attributes if backend stores policy directly
    if hasattr(plot, "blending_mode"):
        plot.blending_mode = mode
    elif hasattr(plot, "policy") and hasattr(plot.policy, "runtime"):
        # do not mutate runtime every frame if backend owns policy;
        # this is just a fallback
        plot.blending_mode = mode
    _set_dirty(plot)


def _set_title(
    plot: GPULinePlot,
    title: str,
    fontsize: Optional[float] = None,
    color: Optional[ColorLike] = None,
) -> None:
    if _call_if_exists(plot, ("set_title",), title, fontsize=fontsize, color=color) is not _MISSING:
        return
    # Fallback for a backend without set_title. Assigns unconditionally for the same
    # reason `GPULinePlot.set_title` does: an omitted kwarg means "default", not "keep".
    if hasattr(plot, "title"):
        plot.title = str(title)
    if hasattr(plot, "options"):
        plot.options.axis_title_fontsize = fontsize
        plot.options.axis_title_color = color
    _set_dirty(plot)


def _apply_equal_aspect(plot: GPULinePlot, square: bool = False) -> None:
    """Give x and y the same world-units-per-pixel, by widening the finer axis.

    matplotlib offers two routes to an equal aspect: move the data limits
    (``adjustable='datalim'``, which is ``axis('equal')``) or reshape the axes box
    (``adjustable='box'``, which is ``axis('scaled')``). GLPlot draws into one GPU
    viewport whose box is the window, so only the datalim route exists here and every
    equal-aspect mode resolves to it. See :func:`axis`.

    Widening is what keeps this safe to call on a fitted view: the coarser of the two
    scales is the one adopted, so the axis that was showing *more* detail zooms out to
    match and nothing that was visible leaves the frame. Adopting the finer scale would
    crop data the caller had just asked to see.

    ``square`` equalises the world *spans* instead, ignoring the pixel aspect -- the data
    range, not the on-screen shape, is what comes out symmetric.
    """
    camera = plot.camera
    half_w = 1.0 / max(camera.zoom_x, 1e-12)
    half_h = 1.0 / max(camera.zoom_y, 1e-12)

    if square:
        half = max(half_w, half_h)
        plot.camera_controller.fit_bounds(
            camera.cx - half,
            camera.cx + half,
            camera.cy - half,
            camera.cy + half,
            plot.width,
            plot.height,
        )
        return

    # The world window lands in the frame *inside* the gutters, not the whole viewport, so
    # the pixel extents that set the scale are the inset ones -- the same numbers `mvp()`
    # projects with. Using the raw viewport would leave the aspect visibly off by the
    # margins, which is exactly the error `axis('equal')` is called to remove.
    from .options import resolve_axis_margins

    margin_l, margin_r, margin_b, margin_t = resolve_axis_margins(plot.options)
    frame_w = max(float(plot.width) - margin_l - margin_r, 1.0)
    frame_h = max(float(plot.height) - margin_b - margin_t, 1.0)

    units_per_px = max((2.0 * half_w) / frame_w, (2.0 * half_h) / frame_h)
    new_half_w = 0.5 * units_per_px * frame_w
    new_half_h = 0.5 * units_per_px * frame_h
    plot.camera_controller.fit_bounds(
        camera.cx - new_half_w,
        camera.cx + new_half_w,
        camera.cy - new_half_h,
        camera.cy + new_half_h,
        plot.width,
        plot.height,
    )


def _set_axis_visible(plot: GPULinePlot, visible: bool) -> None:
    """Show or hide the whole axis apparatus -- ``axis('on')`` / ``axis('off')``."""
    plot.options.axis_show_grid = visible
    plot.options.axis_show_labels = visible
    plot.options.axis_show_frame = visible
    _set_dirty(plot)


def _set_view_limits(
    plot: GPULinePlot,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    if _call_if_exists(plot, ("set_view",), xlim=xlim, ylim=ylim) is not _MISSING:
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
    num: Optional[Union[int, str]] = None,
    title: str = "GLPlot",
    width: int = 1280,
    height: int = 800,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[float] = None,
    *,
    facecolor: Optional[ColorLike] = None,
    edgecolor: Optional[ColorLike] = None,
    frameon: Optional[bool] = None,
    clear: bool = False,
    FigureClass: Optional[type] = None,
    hud: bool = False,
    density: bool = False,
    blending: BlendMode = "auto",
    lod: bool = True,
    budget: int = 8,
    multisample: bool = False,
    cache: bool = True,
    clipping: bool = True,
    ssao: bool = False,
    projection: Optional[str] = None,
    **kwargs: Any,
) -> GPULinePlot:
    """Create a new figure and set it as current.

    Creates a new GPULinePlot figure window with specified dimensions and
    optimization settings. If figsize is provided, it takes precedence over
    width and height. The created figure becomes the current figure for
    subsequent plotting operations.

    Args:
        num (int or str, optional): A unique identifier for the figure. Calling
            ``figure()`` again with the same one makes that figure current
            instead of building a second. A string also becomes the window
            title, as in matplotlib -- so ``figure('My Plot')`` names the window
            and identifies it. If None, a fresh figure is always created.
        title (str, optional): Window title. Defaults to "GLPlot". A string
            ``num`` takes precedence.
        width (int, optional): Window width in pixels. Defaults to 1280.
        height (int, optional): Window height in pixels. Defaults to 800.
        figsize (tuple[float, float], optional): Figure size as (width, height)
            in inches. When provided, pixels are computed as figsize * dpi.
            Defaults to None (use width/height directly).
        dpi (float, optional): Dots per inch for figsize computation. Defaults to
            ``matplotlib.rcParams['figure.dpi']`` (100.0 unless the caller changed it),
            matching matplotlib's own default resolution.
        facecolor (str or tuple, optional): The figure background colour.
            Defaults to the engine's own background.
        edgecolor (str or tuple, optional): Accepted for matplotlib parity.
            GLPlot draws no border around the figure, so this is ignored.
        frameon (bool, optional): Accepted for matplotlib parity. The viewport
            is always drawn; use ``axis('off')`` to hide the axis apparatus.
            Ignored.
        clear (bool, optional): When ``num`` names an existing figure, empty it
            before returning it. Ignored for a fresh figure, which is empty.
        FigureClass (type, optional): Accepted for matplotlib parity. GLPlot has
            one figure type, ``GPULinePlot``. Ignored.
        hud (bool, optional): Enable heads-up display with statistics.
            Defaults to False.
        density (bool, optional): Enable density visualization for 2D layers.
            Defaults to False.
        blending (str, optional): Blending mode: 'auto', 'on', or 'off'.
            Controls transparency blending. Defaults to 'auto'.
        lod (bool, optional): Enable level-of-detail optimization for large
            line datasets. Reduces visible lines above threshold. Defaults to True.
        budget (int, optional): LOD budget (1-8): ratio of visible to total lines.
            Higher values show more lines but slower rendering. Defaults to 8.
        multisample (bool, optional): Enable multisample anti-aliasing.
            Defaults to False.
        cache (bool, optional): Cache interaction path for better responsiveness.
            Defaults to True.
        clipping (bool, optional): Enable clipping optimization. Defaults to True.
        ssao (bool, optional): Enable screen-space ambient occlusion for 3D depth.
            Defaults to False.

    Returns:
        GPULinePlot: The created figure object, now current.

    Examples:
        Create a basic figure:

        >>> import glplot.pyplot as gplt
        >>> fig = gplt.figure(title="My Plot", width=800, height=600)
        >>> gplt.plot([1, 2, 3], [1, 4, 2])
        >>> gplt.show()

        Create with figsize (matplotlib style):

        >>> fig = gplt.figure(figsize=(10, 6), dpi=100)

        Create with optimization flags for large datasets:

        >>> fig = gplt.figure(lod=True, budget=4, cache=True)
    """
    global _CURRENT_PLOT
    _warn_unsupported(
        "figure",
        {"edgecolor": edgecolor, "frameon": frameon, "FigureClass": FigureClass},
        {
            "edgecolor": "has no effect: GLPlot draws no border around the figure",
            "frameon": "has no effect: the viewport is always drawn. Use axis('off') to "
            "hide the grid, frame and tick labels",
            "FigureClass": "has no effect: GLPlot has one figure type, GPULinePlot",
        },
    )
    # Layout-engine kwargs (`constrained_layout`, `layout=`, `tight_layout=` as a
    # figure()-level flag, ...) are matplotlib.Figure's own repacking knobs. GLPlot has one
    # viewport per panel and nothing to repack (see tight_layout()'s own docstring), but a
    # script that opens with `plt.figure(constrained_layout=True)` -- an extremely common
    # first line -- used to die right there with `TypeError: unexpected keyword argument`
    # before it ever reached a single plotting call.
    _warn_unsupported(
        "figure",
        dict(kwargs),
        {
            k: "has no effect: GLPlot has one viewport per panel and nothing to repack "
            "(see tight_layout())"
            for k in kwargs
        },
    )
    # matplotlib's own default: unset dpi reads rcParams['figure.dpi'] (100.0 unless the
    # caller changed it) rather than a number hardcoded here, so a script that sets that
    # rcParam before calling figure() sees it honoured.
    if dpi is None:
        from matplotlib import rcParams as _mpl_rcParams

        dpi = float(_mpl_rcParams.get("figure.dpi", 100.0))
    # matplotlib's rule: a string `num` is the window title as well as the identity. It is
    # also what keeps `figure("My Plot")` -- GLPlot's own long-standing spelling, and every
    # call in this repo -- meaning exactly what it always did after `num` took first place.
    if isinstance(num, str):
        title = num

    if num is not None and num in _FIGURES_BY_NUM:
        plot = _FIGURES_BY_NUM[num]
        if clear:
            plot.scene.layers.clear()
            plot.scene.lines = type(plot.scene.lines)()
            plot._primary_line_layer = None
            plot._cpu_line_copy = None
        _CURRENT_PLOT = plot
        if facecolor is not None:
            plot.options.visual.background_color = tuple(
                float(v) for v in _normalize_rgba(facecolor)[:3]
            )
        _set_dirty(plot)
        return plot

    if figsize is not None:
        width = int(float(figsize[0]) * float(dpi))
        height = int(float(figsize[1]) * float(dpi))
    plot = GPULinePlot(width=width, height=height, title=title)
    # Recovering "how many inches is this figure" -- what render_preview's headless export
    # needs to build a matplotlib figure of the same physical size -- means dividing by
    # whatever dpi actually produced `width`/`height`, not a hardcoded 100. Stored here
    # rather than recomputed there because the raw width/height calling style (no figsize)
    # has no dpi at all until this default fills one in.
    plot._figure_dpi = float(dpi)
    if facecolor is not None:
        # The engine reads only RGB here; a figure's background has nothing to blend with.
        plot.options.visual.background_color = tuple(
            float(v) for v in _normalize_rgba(facecolor)[:3]
        )

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
    if num is not None:
        _FIGURES_BY_NUM[num] = plot
    # ``projection="3d"`` pins the figure to a 3D view *before* any data arrives, which is
    # what makes ``figure(projection="3d")`` followed by nothing draw an empty 3D box
    # rather than an empty 2D one. See ``_resolve_projection``.
    _apply_projection(plot, projection)
    _set_dirty(plot)
    return plot


def gcf() -> GPULinePlot:
    """Get current figure."""
    return _get_or_create_plot()


def gca() -> "AxesProxy":
    """Get the current axes.

    Returns an :class:`AxesProxy` bound to the figure's *active panel* -- the same object
    :func:`subplot` and :func:`subplots` hand back -- so the whole matplotlib ``Axes``
    surface is reachable from it: ``ax.plot(...)``, ``ax.set_xlabel(...)``,
    ``ax.set_zlim(...)``, ``ax.plot_surface(...)``.

    It used to return the *figure*, on the reasoning that one viewport means the figure is
    the axes. That stopped being true when panels landed, and it was never true for the
    method surface: ``GPULinePlot`` has ``set_view`` and ``savefig`` but no ``plot``, no
    ``set_xlabel`` and no ``set_zlim``, so ``plt.gca().set_xlabel("x")`` -- and every
    ``ax = plt.axes(projection="3d"); ax.plot_surface(...)`` script -- died on an
    ``AttributeError`` while this docstring claimed the idiom worked. Use :func:`gcf` when
    what you want really is the figure.

    Returns:
        AxesProxy: The active panel, as a matplotlib-style Axes.

    Examples:
        >>> ax = gplt.gca()
        >>> ax.plot(x, y)
        >>> ax.set_xlabel("t")
    """
    return _active_axes(_get_or_create_plot())


def options(**kwargs):
    """Configure rendering and optimization options.

    Updates EngineOptions for the current figure. Allows fine-grained control
    over performance, quality, and visual settings. Option names correspond
    directly to EngineOptions attributes.

    Keyword Arguments:
        **kwargs: EngineOptions attributes to update. Common options include:
            - density_resolution_scale (float): Scale for density maps (0.1-1.0)
            - cache_refresh_hz (float): Cache refresh rate in Hz
            - lod_enabled (bool): Enable level-of-detail
            - lod_target_coverage (float): LOD target ratio
            - enable_hud (bool): Enable statistics HUD
            - enable_multisample (bool): Enable MSAA
            See EngineOptions documentation for complete list.

    Returns:
        None

    Raises:
        AttributeError: If an invalid option name is provided.

    Examples:
        Basic option setting:

        >>> gplt.options(density_resolution_scale=0.5)

        Multiple options:

        >>> gplt.options(
        ...     lod_enabled=True,
        ...     enable_hud=True,
        ...     cache_refresh_hz=60
        ... )

        Performance tuning for large datasets:

        >>> gplt.options(
        ...     lod_enabled=True,
        ...     lod_target_coverage=0.25,
        ...     enable_cache_interaction_path=True
        ... )
    """
    plot = _get_or_create_plot()
    for k, v in kwargs.items():
        if hasattr(plot.options, k):
            setattr(plot.options, k, v)
        else:
            raise AttributeError(f"EngineOptions has no attribute '{k}'")
    _set_dirty(plot)


#: matplotlib names some Axes methods differently from the module-level pyplot functions
#: they mirror (``ax.set_xlim`` vs ``plt.xlim``). This maps the Axes spelling to the module
#: function :class:`AxesProxy` should delegate to. Anything not listed is looked up on the
#: module under its own name, then on the engine.
_AXES_METHOD_ALIASES = {
    "set_xlim": "xlim",
    "set_ylim": "ylim",
    "get_xlim": "xlim",
    "get_ylim": "ylim",
    "set_xlabel": "xlabel",
    "set_ylabel": "ylabel",
    "set_title": "title",
    "set_xscale": "xscale",
    "set_yscale": "yscale",
    "set_xticks": "xticks",
    "set_yticks": "yticks",
    "set_xticklabels": "xticklabels",
    "set_yticklabels": "yticklabels",
    # ``Axes3D`` spellings. The z row is the mirror of the x/y rows above: the module has
    # ``zlabel``/``zlim``/``zticks`` for symmetry with ``xlabel``/``xlim``/``xticks``, and
    # matplotlib users reach for the ``ax.set_z*`` names, which are what every 3D script
    # actually contains. Without these lines ``ax.set_zlabel("z")`` -- the line that sits
    # directly under a working ``ax.set_xlabel("x")`` in the same script -- raises
    # AttributeError, which is the single most-hit gap the compatibility audit found.
    "get_zlim": "zlim",
    "set_zlabel": "zlabel",
    "set_zscale": "zscale",
    "set_zticks": "zticks",
    "set_zticklabels": "zticklabels",
    # matplotlib's own capital-D aliases of ``plot``/``scatter``/``quiver`` on ``Axes3D``.
    # They are the spelling used throughout the mplot3d gallery, so a pasted example hits
    # them on its first drawing call.
    "plot3D": "plot3d",
    "scatter3D": "scatter3d",
    "quiver3D": "quiver3d",
}


#: Module functions :class:`AxesProxy` refuses to expose as axes methods.
#:
#: ``__getattr__`` resolves *any* public module function, which is what makes the whole
#: plotting surface work per panel for free -- and what made ``ax.show()``, ``ax.close()``,
#: ``ax.clf()`` and ``ax.subplots()`` resolve on an axes object. None of those is an
#: ``Axes`` method in matplotlib, and the last three are actively dangerous from an axes:
#: ``ax.clf()`` wipes the *figure* the caller was drawing one panel of. Worse than the
#: footgun is the lie to duck-typed code -- ``hasattr(ax, "show")`` answering True makes
#: any capability probe misclassify the object.
#:
#: The list is figure-level and global verbs only. Names that really are on matplotlib's
#: ``Axes`` (``cla``, ``clear``, ``legend``, ``grid``, ``text``, ``twinx``, ``margins``,
#: ``tick_params`` ...) stay reachable, as do GLPlot's own axes-level extensions.
_AXES_DENIED_NAMES = frozenset(
    {
        # Figure lifetime and layout.
        # ``figure`` is deliberately absent: newer matplotlib Axes objects carry a real
        # ``.figure`` property, and AxesProxy already has its own -- defined below as
        # ``@property def figure`` -- which resolves before ``__getattr__`` (and this
        # deny-list) is ever consulted. Denying it here would therefore have blocked
        # nothing real; it was just stale.
        "gcf",
        "gca",
        "sca",
        # ``axes`` is deliberately absent: matplotlib's ``ax.axes`` is a real Axes property
        # (it returns the axes itself), so denying the name would break parity rather than
        # restore it. ``gplt.axes()`` the module function stays reachable as ``ax.axes()``
        # -- harmlessly, since it also answers with the current axes.
        "add_axes",
        "add_subplot",
        "delaxes",
        "subplot",
        "subplots",
        "subplot2grid",
        "subplot_mosaic",
        "subplot_tool",
        "subplots_adjust",
        "tight_layout",
        "clf",
        "close",
        "suptitle",
        "figtext",
        "figlegend",
        "figaspect",
        "figimage",
        "fignum_exists",
        "get_fignums",
        "get_figlabels",
        "new_figure_manager",
        "get_current_fig_manager",
        # Output and the event loop.
        "show",
        "savefig",
        "export",
        "pause",
        "ion",
        "ioff",
        "isinteractive",
        "interactive",
        "draw_if_interactive",
        "ginput",
        "waitforbuttonpress",
        "connect",
        "disconnect",
        # Process-wide configuration.
        "switch_backend",
        "get_backend",
        "install_repl_displayhook",
        "uninstall_repl_displayhook",
        "set_loglevel",
        "rc",
        "rc_context",
        "rcdefaults",
        "xkcd",
        "imread",
        "imsave",
    }
)


class _SpineProxy:
    """Stand-in for one matplotlib ``Spine`` (``ax.spines['top']``, etc.).

    GLPlot draws a single frame box around the viewport, not four independent lines, so
    there is nothing for per-edge styling to attach to. Every setter warns once (through
    the shared ``"spines"`` registry key, so the whole family speaks with one message
    rather than one per method) and does nothing, rather than raising -- so
    ``ax.spines['top'].set_visible(False)``, the single most common spine idiom pasted
    from a matplotlib "despine" snippet, does not crash the script that follows it.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def _warn(self, method: str) -> None:
        _warn_unsupported_call(
            "spines",
            f"[{self._name!r}].{method}() has no effect: GLPlot draws one frame box, not "
            "four independently-stylable spines. Use axis('off') to hide the whole frame",
        )

    def set_visible(self, visible: bool) -> None:
        self._warn("set_visible")

    def set_color(self, color: Any) -> None:
        self._warn("set_color")

    def set_edgecolor(self, color: Any) -> None:
        # matplotlib's ``Spine`` is a ``Patch``; ``set_edgecolor`` is the real method name,
        # ``set_color`` a convenience alias -- scripts use both interchangeably.
        self._warn("set_edgecolor")

    def set_linewidth(self, width: float) -> None:
        self._warn("set_linewidth")

    def set_alpha(self, alpha: float) -> None:
        self._warn("set_alpha")

    def set_position(self, position: Any) -> None:
        self._warn("set_position")

    def set_bounds(self, *args: Any, **kwargs: Any) -> None:
        self._warn("set_bounds")

    def set_linestyle(self, style: Any) -> None:
        self._warn("set_linestyle")

    def __repr__(self) -> str:
        return f"<GLPlot spine stub {self._name!r}>"


class _SpinesDict(dict):
    """Dict-like ``ax.spines`` container with matplotlib's four default edge names."""

    def __init__(self) -> None:
        super().__init__({name: _SpineProxy(name) for name in ("top", "bottom", "left", "right")})


class AxesProxy:
    """A matplotlib-style Axes bound to one :class:`~glplot.core.panel.Panel` of a figure.

    Every call first *activates* its panel -- makes the owning figure current and points the
    engine's active-panel index at this panel -- and then delegates to the matching
    module-level pyplot function. That means the entire plotting API (``plot``, ``scatter``,
    ``bar``, ``hist``, ``imshow`` ...) works per panel with no re-implementation: the same
    functions that draw into the current axes simply see this panel as the current one.

    Attribute names follow matplotlib's ``Axes``: ``ax.plot(...)``, ``ax.scatter(...)``,
    ``ax.set_xlim(...)``, ``ax.set_title(...)``, ``ax.legend()``. Names not backed by a module
    function fall through to the engine's own Axes-like methods (``set_view``, ``clear`` ...).
    """

    def __init__(
        self,
        fig: GPULinePlot,
        index: int,
        name: Optional[str] = None,
        row: int = 0,
        col: int = 0,
    ) -> None:
        self._fig = fig
        self._index = int(index)
        self._name = name
        self.row = int(row)
        self.col = int(col)

    def _activate(self) -> GPULinePlot:
        """Make this panel the current axes so a delegated pyplot call targets it."""
        global _CURRENT_PLOT
        _CURRENT_PLOT = self._fig
        if 0 <= self._index < len(self._fig.panels):
            self._fig.active_panel_index = self._index
        return self._fig

    @property
    def figure(self) -> GPULinePlot:
        """The figure (``GPULinePlot``) this axes belongs to."""
        return self._fig

    @property
    def panel(self):
        """The underlying :class:`~glplot.core.panel.Panel`."""
        return self._fig.panels[self._index]

    # ``Axes3D`` exposes the camera angles as plain attributes, and reading them back after
    # ``view_init`` is routine -- an animation loop stores ``ax.azim`` per frame, a script
    # that lets the user orbit saves the viewpoint. They are properties rather than stored
    # values because the camera is the panel's, not the proxy's: the GUI, the mouse and
    # ``view_preset`` all move it behind the proxy's back, and a cached copy would answer
    # with the last value *this object* set.
    #
    # The getters read the panel directly instead of going through ``_activate``: reading
    # an angle is a question, and answering it by making this panel the current axes would
    # make ``print(ax2.azim)`` silently redirect the next ``gplt.plot(...)``.

    @property
    def elev(self) -> float:
        """Elevation of the 3D camera, in degrees (matplotlib's ``Axes3D.elev``)."""
        return float(self.panel.camera3d.elev)

    @elev.setter
    def elev(self, value: float) -> None:
        self._activate().set_3d_view(elev=float(value))

    @property
    def azim(self) -> float:
        """Azimuth of the 3D camera, in degrees (matplotlib's ``Axes3D.azim``)."""
        return float(self.panel.camera3d.azim)

    @azim.setter
    def azim(self, value: float) -> None:
        self._activate().set_3d_view(azim=float(value))

    @property
    def roll(self) -> float:
        """Roll of the 3D camera about the line of sight, in degrees."""
        return float(self.panel.camera3d.roll)

    @roll.setter
    def roll(self, value: float) -> None:
        self._activate().set_3d_view(roll=float(value))

    @property
    def spines(self) -> "_SpinesDict":
        """Stand-in for matplotlib's per-edge ``Spine`` objects. See :class:`_SpineProxy`."""
        return _SpinesDict()

    def inset_axes(
        self,
        bounds: Sequence[float],
        *,
        transform: Any = None,
        zorder: Optional[float] = None,
        **kwargs: Any,
    ) -> "AxesProxy":
        """A small nested axes inside this one (matplotlib's ``Axes.inset_axes``).

        ``bounds`` is ``[x0, y0, width, height]`` as a fraction of *this* axes' own box --
        GLPlot has no separate transform stack, so that is the only placement ``transform``
        supports (matplotlib's default, ``ax.transAxes``); any other transform is accepted
        and warned about rather than raising, matching the module's usual compat policy.

        The child is a real panel -- its own camera, scene, autoscale -- appended after this
        one, so it paints on top of its parent (panels draw in list order); ``zorder``
        cannot reorder that and is warned about too. The parent stays the current axes
        afterwards, as in matplotlib: only the returned proxy targets the inset.
        """
        fig = self._activate()
        parent = self.panel
        x0, y0, w, h = (float(v) for v in bounds)
        px0, py0, pw, ph = parent.rect_frac
        rect_frac = (px0 + x0 * pw, py0 + y0 * ph, w * pw, h * ph)
        projection = kwargs.pop("projection", None)
        _warn_unsupported(
            "inset_axes",
            {"transform": transform, "zorder": zorder, **kwargs},
            {"zorder": "cannot reorder panels beyond their creation order and was ignored"},
        )
        spec = _layout.PanelSpec(rect_frac=rect_frac)
        panel = fig.add_panel(spec)
        idx = len(fig.panels) - 1
        proxy = AxesProxy(fig, idx, name=panel.name, row=panel.row, col=panel.col)
        proxy._activate()
        _apply_projection(fig, projection)
        self._activate()
        return proxy

    def __getattr__(self, name: str):
        # __getattr__ only fires for names not found normally, i.e. the plotting API surface;
        # _fig/_index/etc. are real instance attributes and never reach here (so no recursion).
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name in _AXES_DENIED_NAMES:
            # Refused *before* both lookups, so the engine's own ``show``/``savefig``/
            # ``close`` cannot leak through the fallback either. The message names the
            # figure so the fix is obvious rather than a hunt through matplotlib's docs.
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}: "
                f"{name!r} is a figure-level or global function, not an Axes method. "
                f"Call it on the figure (ax.figure.{name}(...)) or on the module "
                f"(gplt.{name}(...))."
            )
        target = _AXES_METHOD_ALIASES.get(name, name)
        module = _sys.modules[__name__]
        fn = getattr(module, target, None)
        if callable(fn) and not target.startswith("_"):

            def _panel_bound(*args, __fn=fn, **kwargs):
                self._activate()
                return __fn(*args, **kwargs)

            _panel_bound.__name__ = name
            return _panel_bound

        # Fall back to the engine's own Axes-like methods (set_view, autoscale, clear, ...).
        attr = getattr(self._fig, name, None)
        if callable(attr):

            def _engine_bound(*args, __name=name, **kwargs):
                self._activate()
                return getattr(self._fig, __name)(*args, **kwargs)

            _engine_bound.__name__ = name
            return _engine_bound
        if attr is not None:
            return attr
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __repr__(self) -> str:
        tag = f" {self._name!r}" if self._name else ""
        return f"<AxesProxy panel={self._index}{tag} row={self.row} col={self.col}>"


def _active_axes(fig: GPULinePlot) -> AxesProxy:
    """An :class:`AxesProxy` for whichever panel of ``fig`` is currently active.

    The one place that turns "the current axes" from a concept into an object. Built fresh
    on each call rather than cached per panel: a proxy holds only a figure and an index, so
    two proxies for the same panel are interchangeable, and caching would have to be
    invalidated by every ``set_panels`` -- which is exactly the path that would leave a
    stale proxy pointing at a panel that no longer exists.
    """
    index = fig.active_panel_index
    if not (0 <= index < len(fig.panels)):
        index = 0
    panel = fig.panels[index]
    return AxesProxy(
        fig,
        index,
        name=getattr(panel, "name", None),
        row=getattr(panel, "row", 0),
        col=getattr(panel, "col", 0),
    )


def _make_axes(
    fig: GPULinePlot,
    specs: "List",
    nrows: int,
    ncols: int,
    squeeze: bool,
) -> Any:
    """Build the AxesProxy container ``subplots`` returns, matching matplotlib's shape rules.

    A 1x1 grid returns a single proxy when ``squeeze`` (the default); a single row or column
    returns a 1-D object array; anything else returns a 2-D ``(nrows, ncols)`` object array,
    so ``axs[i, j]`` / ``axs[i]`` / ``axs.flat`` all index the way matplotlib code expects.
    """
    proxies = [
        AxesProxy(fig, i, name=s.name, row=getattr(s, "row", 0), col=getattr(s, "col", 0))
        for i, s in enumerate(specs)
    ]
    if squeeze and nrows == 1 and ncols == 1:
        return proxies[0]

    arr = np.empty((nrows, ncols), dtype=object)
    for i, s in enumerate(specs):
        arr[getattr(s, "row", 0), getattr(s, "col", 0)] = proxies[i]
    if squeeze:
        if nrows == 1:
            return arr[0]
        if ncols == 1:
            return arr[:, 0]
    return arr


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    sharex: bool = False,
    sharey: bool = False,
    squeeze: bool = True,
    title: str = "GLPlot",
    width: int = 1280,
    height: int = 800,
    wspace: Optional[float] = None,
    hspace: Optional[float] = None,
    **kwargs: Any,
):
    """Create a figure and a grid of ``nrows`` x ``ncols`` panels (matplotlib-style).

    Each panel is an independent axes with its own camera, data and autoscale, drawn into its
    own rectangle of the window -- so ``axs[0, 0].plot(...)`` and ``axs[0, 1].scatter(...)``
    show different data side by side. This is real multi-panel rendering, not the historical
    single-viewport stub.

    Args:
        nrows, ncols (int): Grid shape. Defaults to 1 x 1.
        sharex, sharey (bool): Link the panels' x / y axes so pan/zoom on one moves them all.
        squeeze (bool): As in matplotlib -- collapse length-1 dimensions of the returned array.
        title, width, height: Passed to :func:`figure`.
        wspace, hspace (float, optional): Gaps between cells, as figure fractions.

    Returns:
        tuple: ``(fig, axes)`` -- the figure and either a single :class:`AxesProxy` (1x1,
        squeezed) or a NumPy object array of them shaped like the grid.

    Examples:
        >>> fig, axs = gplt.subplots(2, 2, sharex=True)
        >>> axs[0, 0].plot(x, np.sin(x))
        >>> axs[0, 1].scatter(x, y)
        >>> gplt.show()
    """
    subplot_kw = dict(kwargs.pop("subplot_kw", None) or {})
    projection = subplot_kw.pop("projection", kwargs.pop("projection", None))
    # matplotlib takes the ratios either at top level or nested inside `gridspec_kw`, and
    # scripts use both; the top-level spelling wins, as it does there.
    gridspec_kw = dict(kwargs.pop("gridspec_kw", None) or {})
    width_ratios = kwargs.pop("width_ratios", None) or gridspec_kw.pop("width_ratios", None)
    height_ratios = kwargs.pop("height_ratios", None) or gridspec_kw.pop("height_ratios", None)
    wspace = wspace if wspace is not None else gridspec_kw.pop("wspace", None)
    hspace = hspace if hspace is not None else gridspec_kw.pop("hspace", None)
    _warn_unsupported("subplots", {f"gridspec_kw[{k!r}]": v for k, v in gridspec_kw.items()})
    fig = figure(title=title, width=width, height=height, **kwargs)
    grid_kwargs: dict = {}
    if wspace is not None:
        grid_kwargs["wspace"] = wspace
    if hspace is not None:
        grid_kwargs["hspace"] = hspace
    if width_ratios is not None:
        grid_kwargs["width_ratios"] = width_ratios
    if height_ratios is not None:
        grid_kwargs["height_ratios"] = height_ratios
    specs = _layout.grid(nrows, ncols, **grid_kwargs)
    fig.set_panels(specs)
    if sharex or sharey:
        _link_shared_axes(fig, sharex=sharex, sharey=sharey)
    # ``subplot_kw={"projection": "3d"}`` is matplotlib's way of saying "every panel in
    # this grid is 3D", so it applies to all of them rather than only the current one.
    ndim = _resolve_projection(projection)
    if ndim is not None:
        for panel in fig.panels:
            panel.set_ndim(ndim)
        fig.set_3d_view()
    axes = _make_axes(fig, specs, nrows, ncols, squeeze)
    return fig, axes


#: The scale names matplotlib's `xscale`/`yscale` take. "linear", "log", "symlog",
#: "asinh", and "logit" are real: each transforms a layer's data at GPU-upload time
#: (never `layer.pts` itself, so the Data panel and CSV export keep reporting real
#: values), leaving the shared ortho projection, `screen_to_world`, and the GPU density
#: accumulator untouched -- they just operate on transformed "world" coordinates without
#: knowing it. `plot()`/`scatter()`/`bar()`/`barh()`/`imshow()`/`contourf()`/`matshow()`
#: all reach that upload hook directly; `contour()`'s live lines and `streamplot()` reach
#: it too since both draw through `add_line_strip()`, and `hist2d()` reaches it since it
#: draws its bin centres through `scatter()` -- none of those three needed their own
#: code, they just already used a primitive this session had already fixed.
#: `function`/`functionlog` (arbitrary user-supplied transform functions) are not real
#: yet -- accepted and warned about, a deliberately separate round. Real scale support
#: does not extend to the analytic "line_family" plots or any 3D layer -- those still
#: warn. `hist2d()`'s bin *edges* also stay linearly spaced even under a log axis (only
#: the bin-centre marker positions are scale-aware) -- a real but separable enhancement.
_SCALE_NAMES = ("linear", "log", "symlog", "logit", "function", "functionlog", "asinh")
_REAL_SCALE_NAMES = ("linear", "log", "symlog", "asinh", "logit")


def _set_scale(func: str, axis: str, value: str, **kwargs: Any) -> None:
    """Shared body of :func:`xscale` and :func:`yscale`."""
    plot = _get_or_create_plot()  # matplotlib creates the figure; a bare xscale() must too.
    name = str(value).strip().lower()
    if name not in _SCALE_NAMES:
        raise ValueError(f"unsupported scale: {value!r}. Expected one of {_SCALE_NAMES}.")
    if name in _REAL_SCALE_NAMES:
        setattr(plot.options, f"axis_scale_{axis}", name)
        setattr(plot.options, f"axis_scale_params_{axis}", dict(kwargs))
        # Every layer's GPU buffer was built for the old scale; force a re-upload under the
        # new one. Same pattern `_init_modules` uses for a GL-context reset (engine.py).
        for layer in plot.scene.layers:
            layer.dirty.gpu_dirty = True
            # An imshow-family layer's quad lives in a second cache key that isn't
            # gated by `gpu_dirty` at all (see `renderers/scatter.py`'s `_draw_image`) --
            # nothing else ever invalidates it, so it must be cleared here too.
            if hasattr(layer, "_image_gl"):
                layer._image_gl = None
        plot.autoscale()
    else:
        _warn_unsupported(
            func,
            {axis: name},
            {
                axis: f"has no effect: GLPlot's projection is a linear ortho matrix and it "
                f"has no {name}-scaled axis, so the {axis}-axis stays linear"
            },
            stacklevel=4,
        )


def xscale(value: str = "linear", **kwargs: Any) -> None:
    """Set the x-axis scale.

    Args:
        value (str): 'linear' (the default), 'log' (real base-10 log), 'symlog' (real,
            linear near zero within ``linthresh``, log outside it), or 'asinh' (real, a
            smooth linear-to-log transition via ``linear_width``) -- all four include
            ticks, picking, and headless export. Any other matplotlib scale name is
            accepted and warns rather than being honoured.
        **kwargs: 'symlog' takes ``linthresh`` (default 2), ``linscale`` (default 1),
            ``base`` (default 10); 'asinh' takes ``linear_width`` (default 1.0). Matches
            matplotlib's own defaults. Ignored for every other scale name.

    Note:
        GLPlot projects with a single linear ortho matrix; a real scale transforms the
        data at GPU-upload time instead of teaching the projection itself non-affine
        math, so `screen_to_world`, ticks, and picking all stay consistent. Values that
        are not positive on a log axis are masked (a gap for lines, not drawn for
        scatter), matching matplotlib -- 'symlog'/'asinh' are defined everywhere and mask
        nothing. `logit`/etc. still just warn and stay linear.

    Examples:
        >>> gplt.xscale('log')
        >>> gplt.xscale('symlog', linthresh=0.5)
    """
    _set_scale("xscale", "x", value, **kwargs)


def yscale(value: str = "linear", **kwargs: Any) -> None:
    """Set the y-axis scale. See :func:`xscale`; the same real/warn split applies."""
    _set_scale("yscale", "y", value, **kwargs)


def loglog(*args: Any, **kwargs: Any) -> list:
    """Plot with both axes log-scaled. See :func:`xscale`.

    Returns:
        list: The layers :func:`plot` produced.
    """
    xscale("log")
    yscale("log")
    return plot(*args, **kwargs)


def semilogx(*args: Any, **kwargs: Any) -> list:
    """Plot with a log-scaled x-axis. See :func:`xscale`."""
    xscale("log")
    return plot(*args, **kwargs)


def semilogy(*args: Any, **kwargs: Any) -> list:
    """Plot with a log-scaled y-axis. See :func:`xscale`."""
    yscale("log")
    return plot(*args, **kwargs)


#: matplotlib spells the same property several ways and real scripts use all of them, so
#: the ``*props`` dicts are canonicalised through this before anything reads them.
_PROP_ALIASES = {
    "lw": "linewidth",
    "ls": "linestyle",
    "c": "color",
    "fc": "facecolor",
    "ec": "edgecolor",
    "mfc": "markerfacecolor",
    "mec": "markeredgecolor",
    "ms": "markersize",
}


def _merge_props(props: Optional[dict], defaults: dict) -> dict:
    """Merge one of matplotlib's ``*props`` dicts over GLPlot's defaults, de-aliased."""
    merged = dict(defaults)
    for key, value in (props or {}).items():
        merged[_PROP_ALIASES.get(key, key)] = value
    return merged


def _bxp_datasets(func: str, x) -> List[np.ndarray]:
    """Normalise matplotlib's three input shapes to a list of 1-D float64 arrays.

    ``boxplot`` takes one array (one box), a *sequence* of arrays (one box each), or a 2-D
    array whose **columns** are the datasets. The column rule is the one that surprises
    people, and it is why this defers to ``cbook._reshape_2D`` rather than guessing:
    getting it wrong transposes every multi-box figure. The import is private but it is
    the same function matplotlib's own ``boxplot``, ``violinplot`` and ``hist`` call, so it
    fails loudly at import rather than silently disagreeing.
    """
    from matplotlib import cbook

    groups = cbook._reshape_2D(x, "x")
    return [np.asarray(g, dtype=np.float64) for g in groups]


def boxplot(
    x: ArrayLike,
    notch: Optional[bool] = None,
    sym: Optional[str] = None,
    vert: Optional[bool] = None,
    whis: Optional[Union[float, Tuple[float, float]]] = None,
    positions: Optional[ArrayLike] = None,
    widths: Optional[Union[float, ArrayLike]] = None,
    patch_artist: Optional[bool] = None,
    bootstrap: Optional[int] = None,
    usermedians: Optional[ArrayLike] = None,
    conf_intervals: Optional[ArrayLike] = None,
    meanline: Optional[bool] = None,
    showmeans: Optional[bool] = None,
    showcaps: Optional[bool] = None,
    showbox: Optional[bool] = None,
    showfliers: Optional[bool] = None,
    boxprops: Optional[dict] = None,
    labels: Optional[Sequence[str]] = None,
    flierprops: Optional[dict] = None,
    medianprops: Optional[dict] = None,
    meanprops: Optional[dict] = None,
    capprops: Optional[dict] = None,
    whiskerprops: Optional[dict] = None,
    manage_ticks: bool = True,
    autorange: bool = False,
    zorder: Optional[float] = None,
    capwidths: Optional[Union[float, ArrayLike]] = None,
    *,
    tick_labels: Optional[Sequence[str]] = None,
    orientation: str = "vertical",
    color: Optional[ColorLike] = None,
    label: Optional[str] = None,
    data: Optional[Any] = None,
):
    """Draw a Tukey box plot -- one box per dataset.

    The box spans Q1..Q3 with the median inside it, and the whiskers reach the furthest
    points still within ``whis`` * IQR of the box. The statistics come from
    :func:`matplotlib.cbook.boxplot_stats`, so ``whis``, ``bootstrap`` and ``autorange``
    produce the same numbers matplotlib would draw.

    Args:
        x (array-like): The values to summarise. One 1-D array draws one box; a sequence
            of arrays draws one box each; a 2-D array draws one box per **column**.
        notch (bool, optional): Pinch the box to its median confidence interval.
            Defaults to False.
        sym (str, optional): Format string for the fliers, e.g. ``'r+'``. ``''`` hides
            them. Overrides the marker keys of ``flierprops``.
        vert (bool, optional): Draw the boxes vertically. ``False`` draws them
            horizontally, swapping the roles of the two axes. Defaults to True. Takes
            precedence over ``orientation`` when both are given, matching matplotlib's own
            deprecation of ``vert`` in favour of ``orientation``.
        orientation ({'vertical', 'horizontal'}, optional): The modern spelling of
            ``vert``. Ignored if ``vert`` is also given.
        whis (float or tuple, optional): Whisker reach as a multiple of the IQR, or a
            ``(lo, hi)`` pair of percentiles. Defaults to 1.5.
        positions (array-like, optional): Where each box sits on the category axis.
            Defaults to ``1, 2, 3...``, as in matplotlib.
        widths (float or array-like, optional): Box width in data units, per box.
        patch_artist (bool, optional): Fill the box instead of outlining it.
            Defaults to False.
        bootstrap (int, optional): Bootstrap resamples used for the median confidence
            interval the notch shows.
        usermedians (array-like, optional): Replaces the computed median per box; entries
            that are None keep the computed one.
        conf_intervals (array-like, optional): ``(lo, hi)`` per box, replacing the
            computed notch interval.
        meanline (bool, optional): Draw the mean as a line across the box rather than a
            marker. Defaults to False.
        showmeans, showcaps, showbox, showfliers (bool, optional): Which pieces to draw.
        boxprops, flierprops, medianprops, meanprops, capprops, whiskerprops (dict,
            optional): Styling per element -- ``color``, ``linewidth``, ``linestyle``, and
            for fliers ``marker``/``markersize``. matplotlib's short aliases (``lw``,
            ``ls``, ``mfc``...) are accepted.
        labels (sequence of str, optional): One tick label per box. matplotlib 3.9
            renamed this to ``tick_labels``; both spellings work here.
        manage_ticks (bool, optional): Put a tick at each box position and label it.
            Defaults to True.
        autorange (bool, optional): When the data is degenerate (IQR of 0), stretch the
            whiskers to the extremes. Defaults to False.
        zorder (float, optional): Accepted for matplotlib parity. GLPlot draws layers in
            the order they are added. Ignored.
        capwidths (float or array-like, optional): Cap width per box. Defaults to the box
            width.
        color (str or tuple, optional): GLPlot extension -- the base colour every element
            inherits when its ``*props`` dict does not override it.
        label (str, optional): Legend label.
        data (indexable, optional): If given, ``x`` may be a key into it.

    Returns:
        dict: ``{'boxes', 'medians', 'whiskers', 'caps', 'fliers', 'means'}`` -- the
        layers drawn, keyed as matplotlib keys its own artists.

    Note:
        Unless ``patch_artist`` is set the box is traced as lines rather than filled, so
        the median stays visible and the whiskers keep a pixel width at any zoom.

    Examples:
        >>> gplt.boxplot(np.random.normal(size=200))
        >>> gplt.boxplot([a, b, c], labels=["a", "b", "c"], notch=True)
        >>> gplt.boxplot(data, vert=False, patch_artist=True, boxprops={"fc": "C2"})
    """
    from matplotlib import cbook

    (x,) = _resolve_data_args("boxplot", data, x)
    _warn_unsupported(
        "boxplot",
        {"zorder": zorder},
        {"zorder": "has no effect: GLPlot draws layers in the order they are added"},
    )

    groups = _bxp_datasets("boxplot", x)
    if any(len(g[np.isfinite(g)]) == 0 for g in groups):
        raise ValueError("boxplot(): x has no finite values to summarise")

    tick_labels = tick_labels if tick_labels is not None else labels
    stats = cbook.boxplot_stats(
        groups, whis=1.5 if whis is None else whis, bootstrap=bootstrap, autorange=autorange
    )
    n = len(stats)

    # matplotlib lets the caller override the two statistics a reader is most likely to
    # have computed elsewhere. A None entry means "keep the computed one", so this is a
    # per-box patch rather than a wholesale replacement.
    if usermedians is not None:
        if len(usermedians) != n:
            raise ValueError(f"boxplot(): usermedians must have {n} entries")
        for stat, med in zip(stats, usermedians):
            if med is not None:
                stat["med"] = float(med)
    if conf_intervals is not None:
        if len(conf_intervals) != n:
            raise ValueError(f"boxplot(): conf_intervals must have {n} entries")
        for stat, ci in zip(stats, conf_intervals):
            if ci is None:
                continue
            lo, hi = ci
            if lo is not None:
                stat["cilo"] = float(lo)
            if hi is not None:
                stat["cihi"] = float(hi)

    if orientation not in ("vertical", "horizontal"):
        raise ValueError(
            f"boxplot(): orientation must be 'vertical' or 'horizontal', got {orientation!r}"
        )
    vert = (orientation == "vertical") if vert is None else bool(vert)
    notch = bool(notch)
    showbox = True if showbox is None else bool(showbox)
    showcaps = True if showcaps is None else bool(showcaps)
    showfliers = True if showfliers is None else bool(showfliers)
    showmeans = bool(showmeans)
    meanline = bool(meanline)
    patch_artist = bool(patch_artist)

    pos = (
        np.arange(1, n + 1, dtype=np.float64)
        if positions is None
        else np.atleast_1d(np.asarray(positions, dtype=np.float64))
    )
    if len(pos) != n:
        raise ValueError(f"boxplot(): positions must have {n} entries, got {len(pos)}")
    if widths is None:
        # matplotlib's rule: wide enough to read, never wider than half the gap between
        # neighbours, and a fixed 0.5 for the single-box case where there is no gap.
        w = np.full(n, float(np.clip(0.15 * np.ptp(pos), 0.15, 0.5)) if n > 1 else 0.5)
    else:
        w = np.broadcast_to(np.atleast_1d(np.asarray(widths, dtype=np.float64)), (n,)).copy()
    caps_w = (
        w
        if capwidths is None
        else np.broadcast_to(np.atleast_1d(np.asarray(capwidths, dtype=np.float64)), (n,)).copy()
    )

    base = tuple(float(v) for v in _normalize_rgba(color if color is not None else "C0"))
    line_defaults = {"color": base, "linewidth": 1.0}
    box_kw = _merge_props(boxprops, {**line_defaults, "facecolor": base})
    whisker_kw = _merge_props(whiskerprops, line_defaults)
    cap_kw = _merge_props(capprops, line_defaults)
    median_kw = _merge_props(medianprops, line_defaults)
    mean_kw = _merge_props(meanprops, {**line_defaults, "marker": "^", "markersize": 6.0})
    flier_kw = _merge_props(flierprops, {**line_defaults, "marker": "o", "markersize": 6.0})

    if sym is not None:
        if sym == "":
            showfliers = False
        else:
            # `_process_plot_format` is how matplotlib itself parses `sym`, so 'r+', '+'
            # and 'r' all split the same way here as there.
            from matplotlib.axes._base import _process_plot_format

            _, marker, fmt_color = _process_plot_format(sym)
            if marker is not None:
                flier_kw["marker"] = marker
            if fmt_color is not None:
                flier_kw["color"] = fmt_color

    artists: dict = {
        "boxes": [],
        "medians": [],
        "whiskers": [],
        "caps": [],
        "fliers": [],
        "means": [],
    }
    plot_obj = _get_or_create_plot()

    def _stroke(xs, ys, kw, key, lbl=None):
        """One piece of the box as a polyline, oriented by ``vert``."""
        # The whole horizontal mode is this swap: every piece is built in "vertical"
        # coordinates and transposed on the way out, so there is one geometry to get right.
        a, b = (xs, ys) if vert else (ys, xs)
        rgba = tuple(float(v) for v in _normalize_rgba(kw.get("color", base)))
        plot_obj.add_line_strip(
            np.asarray(a, dtype=np.float32),
            np.asarray(b, dtype=np.float32),
            color=rgba,
            width=float(kw.get("linewidth", 1.0)),
            label=lbl,
        )
        layer = plot_obj.scene.layers[-1]
        layer.metadata["artist"] = "boxplot"
        artists[key].append(layer)
        return layer

    for i, stat in enumerate(stats):
        at, width, capw = float(pos[i]), float(w[i]), float(caps_w[i])
        box_lo, box_hi = at - width * 0.5, at + width * 0.5

        if notch:
            notch_lo, notch_hi = at - width * 0.25, at + width * 0.25
            box_x = [
                box_lo,
                box_hi,
                box_hi,
                notch_hi,
                box_hi,
                box_hi,
                box_lo,
                box_lo,
                notch_lo,
                box_lo,
                box_lo,
            ]
            box_y = [
                stat["q1"],
                stat["q1"],
                stat["cilo"],
                stat["med"],
                stat["cihi"],
                stat["q3"],
                stat["q3"],
                stat["cihi"],
                stat["med"],
                stat["cilo"],
                stat["q1"],
            ]
            med_x = [notch_lo, notch_hi]
        else:
            box_x = [box_lo, box_hi, box_hi, box_lo, box_lo]
            box_y = [stat["q1"], stat["q1"], stat["q3"], stat["q3"], stat["q1"]]
            med_x = [box_lo, box_hi]

        if showbox:
            if patch_artist:
                _bxp_patch(
                    plot_obj, box_x, box_y, box_kw, vert, artists, base, label if i == 0 else None
                )
            else:
                _stroke(box_x, box_y, box_kw, "boxes", label if i == 0 else None)
        _stroke([at, at], [stat["q1"], stat["whislo"]], whisker_kw, "whiskers")
        _stroke([at, at], [stat["q3"], stat["whishi"]], whisker_kw, "whiskers")
        if showcaps:
            cap_x = [at - capw * 0.5, at + capw * 0.5]
            _stroke(cap_x, [stat["whislo"]] * 2, cap_kw, "caps")
            _stroke(cap_x, [stat["whishi"]] * 2, cap_kw, "caps")
        _stroke(med_x, [stat["med"]] * 2, median_kw, "medians")

        if showmeans:
            if meanline:
                _stroke([box_lo, box_hi], [stat["mean"]] * 2, mean_kw, "means")
            else:
                artists["means"].append(
                    _bxp_markers(np.array([at]), np.array([stat["mean"]]), mean_kw, vert)
                )
        if showfliers and len(stat["fliers"]):
            fx = np.full(len(stat["fliers"]), at, dtype=np.float64)
            artists["fliers"].append(
                _bxp_markers(fx, np.asarray(stat["fliers"], dtype=np.float64), flier_kw, vert)
            )

    if manage_ticks:
        setter = xticks if vert else yticks
        setter(pos, [str(t) for t in tick_labels] if tick_labels is not None else None)

    _set_dirty(plot_obj)
    return artists


def _bxp_markers(xs, ys, kw, vert):
    """The point-shaped pieces of a box plot -- fliers and the mean marker."""
    a, b = (xs, ys) if vert else (ys, xs)
    return scatter(
        a,
        b,
        color=kw.get("markerfacecolor", kw.get("color")),
        s=float(kw.get("markersize", 6.0)),
        marker=kw.get("marker"),
    )


def _bxp_patch(plot_obj, box_x, box_y, kw, vert, artists, base, label):
    """A filled box, for ``patch_artist=True``.

    Fanned from the centroid rather than drawn as a triangle strip: a notched box is not
    convex, and a strip over its eleven points folds the notch inside out. Every notched
    box *is* star-shaped about its centre, so a fan tessellates both shapes correctly.
    """
    xs = np.asarray(box_x, dtype=np.float64)
    ys = np.asarray(box_y, dtype=np.float64)
    a, b = (xs, ys) if vert else (ys, xs)
    ring = np.column_stack([a, b])[:-1]  # drop the repeated closing vertex
    centre = ring.mean(axis=0)
    verts = np.vstack([centre, ring]).astype(np.float32)
    m = len(ring)
    idx = np.empty(m * 3, dtype=np.uint32)
    idx[0::3] = 0
    idx[1::3] = np.arange(1, m + 1)
    idx[2::3] = np.arange(1, m + 1) % m + 1

    face = tuple(float(v) for v in _normalize_rgba(kw.get("facecolor", base)))
    edge = tuple(float(v) for v in _normalize_rgba(kw.get("edgecolor", kw.get("color", base))))
    plot_obj.add_patch(
        verts, indices=idx, mode="triangles", face_color=face, edge_color=edge, label=label
    )
    layer = plot_obj.scene.layers[-1]
    layer.metadata["artist"] = "boxplot"
    artists["boxes"].append(layer)
    return layer


def violinplot(
    dataset: ArrayLike,
    positions: Optional[ArrayLike] = None,
    vert: Optional[bool] = None,
    widths: Union[float, ArrayLike] = 0.5,
    showmeans: bool = False,
    showextrema: bool = True,
    showmedians: bool = False,
    quantiles: Optional[ArrayLike] = None,
    points: int = 100,
    bw_method: Optional[Any] = None,
    *,
    orientation: str = "vertical",
    side: str = "both",
    color: ColorLike = (0.2, 0.4, 0.8, 0.5),
    facecolor: Optional[ColorLike] = None,
    linecolor: Optional[ColorLike] = None,
    label: Optional[str] = None,
    data: Optional[Any] = None,
):
    """Draw a violin plot -- a kernel density estimate mirrored about its axis.

    What a box plot cannot show: a bimodal distribution has the same quartiles as
    a flat one, and only the density tells them apart.

    Args:
        dataset (array-like): The values to estimate. A 1-D array is one violin;
            a sequence of arrays is one violin each.
        positions (array-like, optional): Where each violin sits on the x-axis.
            Defaults to ``0, 1, 2...``.
        vert (bool, optional): Draw the violins vertically. Defaults to True. Takes
            precedence over ``orientation`` when both are given, matching matplotlib's own
            deprecation of ``vert`` in favour of ``orientation``.
        orientation ({'vertical', 'horizontal'}, optional): The modern spelling of
            ``vert``. Ignored if ``vert`` is also given.
        side ({'both', 'low', 'high'}, optional): Accepted for parity; ignored. GLPlot
            always draws the full, symmetric violin.
        widths (float or array-like, optional): Maximum width of each violin in
            data units. Defaults to 0.5.
        showmeans (bool, optional): Mark each mean with a line. Defaults to False.
        showextrema (bool, optional): Mark the min and max. Defaults to True.
        showmedians (bool, optional): Mark each median. Defaults to False.
        quantiles (array-like, optional): Quantiles to mark, per violin, each in
            0..1.
        points (int, optional): How many samples the density is evaluated at.
            Defaults to 100.
        bw_method (optional): Passed to ``scipy.stats.gaussian_kde``.
        color (str or tuple, optional): Violin body and line colour, if ``facecolor``/
            ``linecolor`` are not given.
        facecolor (str or tuple, optional): Violin body colour. Overrides ``color``.
        linecolor (str or tuple, optional): Colour of the extrema/mean/median marks.
            Overrides ``color``.
        label (str, optional): Legend label.
        data (indexable, optional): If given, ``dataset`` may be a key into it.

    Returns:
        dict: ``{'bodies': [...], 'cmeans': [...], 'cmins': [...], 'cmaxes': [...],
        'cmedians': [...], 'cquantiles': [...]}`` -- the layers drawn, keyed as
        matplotlib keys its own artists.

    Examples:
        >>> gplt.violinplot(samples)
        >>> gplt.violinplot([a, b, c], showmedians=True)
    """
    from scipy.stats import gaussian_kde

    (dataset,) = _resolve_data_args("violinplot", data, dataset)

    if orientation not in ("vertical", "horizontal"):
        raise ValueError(
            f"violinplot(): orientation must be 'vertical' or 'horizontal', got {orientation!r}"
        )
    vert = (orientation == "vertical") if vert is None else bool(vert)
    _warn_unsupported(
        "violinplot",
        {"side": side if side != "both" else None},
        {"side": "has no effect: GLPlot always draws the full, symmetric violin"},
    )

    raw = list(dataset) if isinstance(dataset, (list, tuple)) else [dataset]
    if raw and np.ndim(raw[0]) == 0:
        raw = [dataset]
    groups = [np.atleast_1d(_as_float_array(g, name="dataset")) for g in raw]

    pos = (
        np.arange(len(groups), dtype=np.float64)
        if positions is None
        else np.atleast_1d(np.asarray(positions, dtype=np.float64))
    )
    if len(pos) != len(groups):
        raise ValueError(f"positions must have {len(groups)} entries, got {len(pos)}")
    w = np.atleast_1d(np.asarray(widths, dtype=np.float64))
    if len(w) == 1:
        w = np.full(len(groups), w[0])
    elif len(w) != len(groups):
        raise ValueError(f"widths must have 1 or {len(groups)} entries, got {len(w)}")

    artists: dict = {
        "bodies": [],
        "cmeans": [],
        "cmins": [],
        "cmaxes": [],
        "cmedians": [],
        "cquantiles": [],
    }
    plot_obj = _get_or_create_plot()
    rgba = list(_normalize_rgba(facecolor if facecolor is not None else color))
    line_source = linecolor if linecolor is not None else color
    line_rgba = tuple(float(v) for v in _normalize_rgba(line_source)[:3]) + (1.0,)

    for i, values in enumerate(groups):
        finite = values[np.isfinite(values)]
        if len(finite) < 2:
            raise ValueError(
                f"violinplot(): group {i} has {len(finite)} finite values; a density "
                "estimate needs at least 2"
            )
        if np.ptp(finite) == 0:
            raise ValueError(
                f"violinplot(): group {i} is a single repeated value, which has no density"
            )

        kde = gaussian_kde(finite, bw_method=bw_method)
        grid = np.linspace(finite.min(), finite.max(), int(points))
        density = kde(grid)
        # Scaled so the widest point of every violin is `widths`, which is what makes a
        # row of them comparable by shape. matplotlib normalises the same way.
        density = density / density.max() * (w[i] / 2.0)

        # The body is the density mirrored about the position: one closed polygon, built
        # as a strip between the two sides rather than a fan, so it stays convex-free.
        left, right = pos[i] - density, pos[i] + density
        verts = np.empty((2 * len(grid), 2), dtype=np.float32)
        if vert:
            verts[0::2, 0], verts[0::2, 1] = left, grid
            verts[1::2, 0], verts[1::2, 1] = right, grid
        else:
            verts[0::2, 0], verts[0::2, 1] = grid, left
            verts[1::2, 0], verts[1::2, 1] = grid, right
        add_patch(
            verts,
            mode="strip",
            face_color=tuple(rgba),
            edge_color=tuple(rgba),
            label=label if i == 0 else None,
        )
        body = plot_obj.scene.layers[-1]
        body.metadata["artist"] = "violinplot"
        artists["bodies"].append(body)

        def _mark(value: float, key: str) -> None:
            """A cross-bar at `value`, as wide as the violin is there."""
            half = float(np.interp(value, grid, density))
            if vert:
                pts = np.array([[pos[i] - half, value], [pos[i] + half, value]], dtype=np.float32)
            else:
                pts = np.array([[value, pos[i] - half], [value, pos[i] + half]], dtype=np.float32)
            plot_obj.add_line_strip(pts[:, 0], pts[:, 1], color=line_rgba, width=1.0, label=None)
            artists[key].append(plot_obj.scene.layers[-1])

        if showextrema:
            _mark(float(finite.min()), "cmins")
            _mark(float(finite.max()), "cmaxes")
        if showmeans:
            _mark(float(finite.mean()), "cmeans")
        if showmedians:
            _mark(float(np.median(finite)), "cmedians")
        if quantiles is not None:
            qs = np.atleast_1d(np.asarray(quantiles, dtype=np.float64))
            # A flat list means "these quantiles on every violin"; a nested one means
            # "these on this violin". matplotlib accepts both, and a 1-D array here is
            # ambiguous only if the caller has as many violins as quantiles -- so the
            # nesting, not the length, is what decides.
            row = qs if np.ndim(quantiles[0]) == 0 else np.asarray(quantiles[i], dtype=np.float64)
            if np.any((row < 0) | (row > 1)):
                raise ValueError("quantiles must lie in 0..1")
            for q in row:
                _mark(float(np.quantile(finite, q)), "cquantiles")

    _set_dirty(plot_obj)
    return artists


def _wedge_geometry(cx, cy, theta, radius, width):
    """Vertices and triangle indices for one wedge, solid or annular.

    ``width`` is matplotlib's donut control: ``None`` fans the wedge from its own centre,
    a value carves the ring between ``radius - width`` and ``radius``. The annulus cannot
    be a fan -- the hole is not inside any triangle from the centre -- so it is stitched
    as a quad strip between the inner and outer arcs instead.
    """
    outer = np.column_stack([cx + radius * np.cos(theta), cy + radius * np.sin(theta)])
    if not width:
        verts = np.vstack([[cx, cy], outer]).astype(np.float32)
        fan = np.arange(1, len(outer))
        idx = np.column_stack([np.zeros_like(fan), fan, fan + 1]).ravel().astype(np.uint32)
        return verts, idx

    r_in = max(0.0, float(radius) - float(width))
    inner = np.column_stack([cx + r_in * np.cos(theta), cy + r_in * np.sin(theta)])
    verts = np.vstack([outer, inner]).astype(np.float32)
    n = len(outer)
    k = np.arange(n - 1)
    quads = np.column_stack([k, k + 1, k + n, k + 1, k + 1 + n, k + n])
    return verts, quads.ravel().astype(np.uint32)


def pie(
    x: ArrayLike,
    explode: Optional[ArrayLike] = None,
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[ColorLike]] = None,
    autopct: Optional[Union[str, Any]] = None,
    pctdistance: float = 0.6,
    shadow: bool = False,
    labeldistance: Optional[float] = 1.1,
    startangle: float = 0.0,
    radius: float = 1.0,
    counterclock: bool = True,
    wedgeprops: Optional[dict] = None,
    textprops: Optional[dict] = None,
    center: Tuple[float, float] = (0.0, 0.0),
    frame: bool = False,
    rotatelabels: bool = False,
    *,
    normalize: bool = True,
    hatch: Optional[Union[str, Sequence[str]]] = None,
    data: Optional[Any] = None,
):
    """Draw a pie chart of ``x``.

    Each value becomes a wedge sized by its share of the total, built as a fan of
    triangles -- or, when ``wedgeprops['width']`` is set, as a ring, which is how
    matplotlib draws a donut.

    Args:
        x (array-like): The wedge sizes.
        explode (array-like, optional): Per-wedge radial offset, as a fraction of
            the radius.
        labels (sequence of str, optional): One label per wedge. Drawn beside the pie at
            ``labeldistance`` *and* attached to the wedge so :func:`legend` finds it.
        colors (sequence, optional): One colour per wedge. Defaults to the
            property cycle.
        autopct (str or callable, optional): Label each wedge with its share --
            a format string applied to the percentage (``'%1.1f%%'``) or a callable
            taking the percentage and returning the text.
        pctdistance (float, optional): Where the ``autopct`` text sits, as a fraction of
            the radius. Defaults to 0.6.
        shadow (bool, optional): Draw a dark offset copy behind the pie. Defaults to
            False.
        labeldistance (float, optional): Where the labels sit, as a fraction of the
            radius. ``None`` draws no label text. Defaults to 1.1.
        startangle (float, optional): Angle of the first wedge's start edge, in
            degrees counterclockwise from the x-axis. Defaults to 0.
        radius (float, optional): Pie radius. Defaults to 1.
        counterclock (bool, optional): Lay the wedges out counterclockwise.
            Defaults to True.
        wedgeprops (dict, optional): Wedge styling -- ``width`` for a donut,
            ``edgecolor``/``linewidth`` for the rim, ``facecolor`` to override the colour.
        textprops (dict, optional): ``fontsize`` and ``color`` for the label and
            percentage text.
        center (tuple, optional): Pie centre in data coordinates.
        frame (bool, optional): Keep the axes frame around the pie. Defaults to False,
            which hides it, as in matplotlib.
        rotatelabels (bool, optional): Accepted for matplotlib parity. GLPlot's text
            renderer draws horizontally only. Ignored.
        normalize (bool, optional): Scale the values to a full circle. When False the
            values must already sum to at most 1 and a short sum leaves a gap.
            Defaults to True.
        hatch (str or sequence, optional): Accepted for matplotlib parity. GLPlot has no
            hatch renderer. Ignored.
        data (indexable, optional): If given, ``x`` may be a key into it.

    Returns:
        tuple: ``(wedges, texts)``, or ``(wedges, texts, autotexts)`` when ``autopct`` is
        given -- matching matplotlib's arity.

    Note:
        Call ``axis('equal')`` to keep the pie round -- without it the wedges are
        stretched by the viewport's aspect, exactly as in matplotlib.

    Examples:
        >>> gplt.pie([30, 20, 50], labels=['a', 'b', 'c'], autopct='%1.1f%%')
        >>> gplt.pie(shares, wedgeprops={'width': 0.4})   # a donut
        >>> gplt.axis('equal')
    """
    (x,) = _resolve_data_args("pie", data, x)
    values = _as_signal(x, "x")
    if len(values) == 0:
        raise ValueError("pie(): x is empty")
    if np.any(values < 0):
        raise ValueError("pie(): x must not contain negative values")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("pie(): x must sum to something positive")
    if normalize:
        fractions = values / total
    elif total > 1:
        # matplotlib's rule: without normalisation the values *are* the fractions, so a
        # sum above 1 would wrap the pie past itself rather than mean anything.
        raise ValueError("pie(): cannot plot an unnormalized pie with sum(x) > 1")
    else:
        fractions = values

    _warn_unsupported(
        "pie",
        {"rotatelabels": rotatelabels or None, "hatch": hatch},
        {
            "rotatelabels": "has no effect: GLPlot's text renderer draws horizontally only",
            "hatch": "has no effect: GLPlot has no hatch renderer, so wedges are drawn flat",
        },
    )

    if explode is not None:
        explode_arr = _as_signal(explode, "explode")
        if len(explode_arr) != len(values):
            raise ValueError("explode must have the same length as x")
    else:
        explode_arr = np.zeros(len(values))
    if labels is not None and len(labels) != len(values):
        raise ValueError("labels must have the same length as x")
    if colors is not None and len(colors) != len(values):
        raise ValueError("colors must have the same length as x")

    wprops = _merge_props(wedgeprops, {})
    tprops = _merge_props(textprops, {})
    donut_width = wprops.get("width")
    text_size = tprops.get("fontsize")
    text_color = tprops.get("color")

    plot_obj = _get_or_create_plot()
    wedges: list = []
    texts: list = []
    autotexts: list = []
    angle = np.deg2rad(float(startangle))
    direction = 1.0 if counterclock else -1.0

    def _label_at(distance, s, store):
        """Place one piece of pie text at a radial distance from the wedge's own centre."""
        tx = cx + distance * float(radius) * np.cos(mid)
        ty = cy + distance * float(radius) * np.sin(mid)
        text(tx, ty, s, fontsize=text_size, color=text_color)
        store.append(plot_obj.scene.layers[-1])

    for i, fraction in enumerate(fractions):
        sweep = direction * fraction * 2.0 * np.pi
        # One segment per ~2 degrees: fine enough that the arc reads as a curve at a
        # normal zoom, and a wedge is a handful of triangles either way.
        steps = max(2, int(abs(np.rad2deg(sweep)) / 2.0) + 1)
        theta = np.linspace(angle, angle + sweep, steps + 1)
        mid = angle + 0.5 * sweep
        cx = center[0] + explode_arr[i] * float(radius) * np.cos(mid)
        cy = center[1] + explode_arr[i] * float(radius) * np.sin(mid)

        verts, indices = _wedge_geometry(cx, cy, theta, float(radius), donut_width)
        rgba = _normalize_rgba(
            wprops.get("facecolor")
            or (colors[i] if colors is not None else _color_cycle()[i % len(_color_cycle())])
        )
        face = tuple(float(v) for v in rgba)
        edge = tuple(float(v) for v in _normalize_rgba(wprops.get("edgecolor", rgba)))

        if shadow:
            # Drawn first, because GLPlot composites in insertion order -- a shadow added
            # after its wedge would sit on top of it.
            shifted = verts.copy()
            shifted[:, 0] -= 0.02 * float(radius)
            shifted[:, 1] -= 0.02 * float(radius)
            add_patch(
                shifted,
                indices=indices,
                mode="triangles",
                face_color=(0.0, 0.0, 0.0, 0.35),
                edge_color=(0.0, 0.0, 0.0, 0.35),
            )

        add_patch(
            verts,
            indices=indices,
            mode="triangles",
            face_color=face,
            edge_color=edge,
            label=labels[i] if labels is not None else None,
        )
        wedges.append(plot_obj.scene.layers[-1])
        # Stashed for pie_label(), added after the fact against an already-drawn pie: it
        # needs each wedge's own centre and mid-angle, which only this loop computes.
        wedges[-1].metadata["pie_center"] = (float(cx), float(cy))
        wedges[-1].metadata["pie_mid_angle"] = float(mid)
        wedges[-1].metadata["pie_radius"] = float(radius)

        if labels is not None and labeldistance is not None:
            _label_at(float(labeldistance), str(labels[i]), texts)
        if autopct is not None:
            pct = 100.0 * float(fraction)
            s = autopct(pct) if callable(autopct) else str(autopct) % pct
            _label_at(float(pctdistance), s, autotexts)

        angle += sweep

    if not frame:
        # matplotlib hides the axes for a pie unless `frame` asks for them: a pie has no
        # meaningful x or y, so ticks around it read as noise.
        set_axis_off()

    _set_dirty(plot_obj)
    return (wedges, texts, autotexts) if autopct is not None else (wedges, texts)


def pie_label(
    container: Sequence[Any],
    labels: Union[str, Sequence[str]],
    *,
    distance: float = 0.6,
    textprops: Optional[dict] = None,
    rotate: bool = False,
    alignment: str = "auto",
) -> list:
    """Add labels to an already-drawn pie chart, one per wedge.

    matplotlib 3.11+ API for labelling a pie after the fact, instead of only through
    ``pie(labels=...)`` at draw time. Reads the per-wedge centre/angle :func:`pie` stashes
    in ``layer.metadata`` at draw time, so it only works on wedges :func:`pie` produced.

    Args:
        container: The wedge list :func:`pie` returned (its first return value).
        labels: One label for every wedge, or a single string repeated for all of them.
        distance (float, optional): Label position as a fraction of the pie's radius from
            each wedge's own centre. Defaults to 0.6, matplotlib's own default.
        textprops (dict, optional): ``fontsize``/``color`` forwarded to :func:`text`.
        rotate (bool, optional): Rotate each label to align with its wedge's radial
            direction. Accepted for parity; ignored -- GLPlot's text layers do not rotate.
        alignment (str, optional): Accepted for parity; ignored -- GLPlot centres text on
            its anchor point regardless of which side of the pie it falls on.

    Returns:
        list: The text layers added, one per wedge.

    Examples:
        >>> wedges, _ = gplt.pie([30, 70])
        >>> gplt.pie_label(wedges, ["a", "b"], distance=0.7)
    """
    if isinstance(labels, str):
        label_list = [labels] * len(container)
    else:
        label_list = list(labels)
        if len(label_list) != len(container):
            raise ValueError("pie_label(): labels must have one entry per wedge")
    _warn_unsupported(
        "pie_label",
        {"rotate": rotate or None, "alignment": alignment if alignment != "auto" else None},
        {
            "rotate": "has no effect: GLPlot's text layers do not rotate",
            "alignment": "has no effect: text is centred on its anchor point regardless "
            "of which side of the pie it falls on",
        },
    )
    tprops = _merge_props(textprops, {})
    added: list = []
    plot_obj = _get_or_create_plot()
    for wedge, s in zip(container, label_list):
        cx, cy = wedge.metadata["pie_center"]
        mid = wedge.metadata["pie_mid_angle"]
        radius = wedge.metadata["pie_radius"]
        tx = cx + float(distance) * radius * np.cos(mid)
        ty = cy + float(distance) * radius * np.sin(mid)
        text(tx, ty, str(s), fontsize=tprops.get("fontsize"), color=tprops.get("color"))
        added.append(plot_obj.scene.layers[-1])
    return added


def _link_shared_axes(fig: GPULinePlot, sharex: bool = False, sharey: bool = False) -> None:
    """Put every panel of ``fig`` into a shared-x and/or shared-y group.

    A group is just the list of panels that move together: the interaction router reads
    ``panel.sharex_group`` / ``panel.sharey_group`` and mirrors a pan/zoom on one onto the
    rest. Passing all panels (matplotlib ``sharex=True``) links the whole grid.
    """
    panels = list(fig.panels)
    if sharex:
        for p in panels:
            p.sharex_group = panels
    if sharey:
        for p in panels:
            p.sharey_group = panels


def _parse_subplot_args(args: tuple) -> Tuple[int, int, int]:
    """Parse ``subplot``'s matplotlib argument forms into ``(nrows, ncols, index)``.

    Accepts ``subplot(2, 1, 1)``, the packed ``subplot(211)``, and ``subplot((2, 1, 1))``.
    ``index`` is matplotlib's 1-based, row-major cell number.
    """
    if len(args) == 1:
        a = args[0]
        if isinstance(a, (tuple, list)):
            return int(a[0]), int(a[1]), int(a[2])
        code = int(a)
        return code // 100, (code // 10) % 10, code % 10
    if len(args) >= 3:
        return int(args[0]), int(args[1]), int(args[2])
    return 1, 1, 1


def subplot(*args: Any, **kwargs: Any) -> AxesProxy:
    """Add (or select) a panel in an ``nrows`` x ``ncols`` grid and return it.

    Real multi-panel behaviour: ``plt.subplot(2, 1, 1)`` then ``plt.subplot(2, 1, 2)`` give
    two independent axes stacked vertically, each drawing its own data. Repeated calls with
    the same grid shape reuse the existing panels (so the first half's data is not wiped);
    changing the shape rebuilds the grid. ``index`` is matplotlib's 1-based, row-major number.

    ``projection="3d"`` makes the selected panel 3D, so a figure can mix 2D and 3D panels
    side by side — each panel owns its own camera and its own dimensionality.

    Returns:
        AxesProxy: The selected panel, also made the current axes.

    Examples:
        >>> gplt.subplot(1, 2, 1).plot(x, y)
        >>> gplt.subplot(1, 2, 2, projection="3d").scatter3d(x, y, z)
    """
    projection = kwargs.pop("projection", None)
    fig = _get_or_create_plot()
    if len(args) == 1 and hasattr(args[0], "get_geometry") and hasattr(args[0], "get_gridspec"):
        # A real matplotlib `SubplotSpec` from indexing a `GridSpec` (`fig.add_subplot(gs[0,
        # 1:])` -- the modern way to place a spanning axes, and `add_gridspec`'s whole point).
        # `get_geometry()` gives the flattened (nrows, ncols, start, stop) cell range;
        # converting that to (row, col, rowspan, colspan) lets subplot2grid's existing
        # span-grid placement do the actual work rather than a second implementation of it.
        spec_nrows, spec_ncols, start, stop = args[0].get_geometry()
        row0, col0 = divmod(start, spec_ncols)
        row1, col1 = divmod(stop, spec_ncols)
        return subplot2grid(
            (spec_nrows, spec_ncols),
            (row0, col0),
            rowspan=row1 - row0 + 1,
            colspan=col1 - col0 + 1,
            projection=projection,
            **kwargs,
        )
    nrows, ncols, index = _parse_subplot_args(args)
    shape = (nrows, ncols)
    if getattr(fig, "_subplot_grid_shape", None) != shape or len(fig.panels) != nrows * ncols:
        fig.set_panels(_layout.grid(nrows, ncols))
        fig._subplot_grid_shape = shape
    idx = max(0, min(index - 1, len(fig.panels) - 1))
    panel = fig.panels[idx]
    proxy = AxesProxy(fig, idx, name=panel.name, row=panel.row, col=panel.col)
    proxy._activate()
    _apply_projection(fig, projection)
    return proxy


def twinx(ax: Optional[GPULinePlot] = None) -> "AxesProxy":
    """Return the current axes, as an :class:`AxesProxy`, *not* a twinned one.

    A second y-axis needs two independent y transforms in one viewport, which
    GLPlot's single ortho projection cannot express. Accepted so a matplotlib
    script runs; the returned axes shares the original's scale, so the second
    series is drawn against the first's y-axis.

    Returns an :class:`AxesProxy` (not the raw figure) specifically so the standard
    ``ax2 = ax1.twinx(); ax2.plot(...)`` idiom -- what every matplotlib tutorial
    teaches, and a strictly more common shape than the module-level
    ``gplt.twinx(); gplt.plot(...)`` -- has a ``.plot``/``.scatter``/``.set_ylabel`` to
    call. Returning the bare ``GPULinePlot`` here used to make exactly that line raise
    ``AttributeError: 'GPULinePlot' object has no attribute 'plot'``.

    Returns:
        AxesProxy: The current axes -- sharing the original's y-scale, not a real
        second one.
    """
    _warn_unsupported(
        "twinx",
        {"twinx": True},
        {
            "twinx": "returns the same axes, not a twinned one: GLPlot has a single ortho "
            "projection and cannot give one viewport two y scales. The second series is "
            "drawn against the first's y-axis"
        },
    )
    plot = ax if isinstance(ax, GPULinePlot) else _get_or_create_plot()
    return _active_axes(plot)


def twiny(ax: Optional[GPULinePlot] = None) -> "AxesProxy":
    """Return the current axes, as an :class:`AxesProxy`. See :func:`twinx`; the same
    limitation applies to x."""
    _warn_unsupported(
        "twiny",
        {"twiny": True},
        {
            "twiny": "returns the same axes, not a twinned one: GLPlot has a single ortho "
            "projection and cannot give one viewport two x scales"
        },
    )
    plot = ax if isinstance(ax, GPULinePlot) else _get_or_create_plot()
    return _active_axes(plot)


def secondary_xaxis(
    location: Union[str, float] = "top",
    *,
    functions: Optional[Tuple[Any, Any]] = None,
    transform: Optional[Any] = None,
    **kwargs: Any,
) -> "AxesProxy":
    """Accepted for matplotlib parity; returns the current axes, not a real secondary one.

    A secondary axis needs its own independent tick strip along an edge -- GLPlot's single
    viewport has no renderer for a second one, the same fundamental limitation as
    :func:`twinx` (which at least shares a data scale; a secondary axis doesn't even need
    to, since matplotlib derives its ticks from ``functions`` rather than plotting new data
    on it -- but there is still nowhere to draw that second tick strip). Returns an
    :class:`AxesProxy` bound to the current panel, so a chained
    ``ax.secondary_xaxis('top').set_xlabel(...)`` does not raise ``AttributeError`` on the
    next line, which is what happened before this function existed at all -- unlike every
    other single-viewport limitation in this module, it previously had no acknowledgment
    whatsoever, not even a warning.

    Args:
        location (str or float, optional): Accepted for matplotlib parity. Ignored.
        functions (tuple, optional): The ``(forward, inverse)`` unit-conversion pair a real
            secondary axis derives its ticks from. Accepted for matplotlib parity, but
            ignored: there is no second tick strip to apply them to.
        transform: Accepted for matplotlib parity. Ignored.
        **kwargs: Accepted for matplotlib parity. Ignored.

    Returns:
        AxesProxy: The current axes -- not a real secondary one.

    Examples:
        >>> gplt.plot(celsius, y)
        >>> secax = gplt.secondary_xaxis(
        ...     "top", functions=(lambda c: c * 9 / 5 + 32, lambda f: (f - 32) * 5 / 9)
        ... )
        >>> secax.set_xlabel("Fahrenheit")  # drawn, but on the same (Celsius-scaled) axis
    """
    _warn_unsupported_call(
        "secondary_xaxis",
        "returns the current axes, not a real secondary one: GLPlot's single viewport has "
        "no independent tick strip to draw a second x-axis on",
    )
    return _active_axes(_get_or_create_plot())


def secondary_yaxis(
    location: Union[str, float] = "right",
    *,
    functions: Optional[Tuple[Any, Any]] = None,
    transform: Optional[Any] = None,
    **kwargs: Any,
) -> "AxesProxy":
    """Accepted for matplotlib parity; returns the current axes, not a real secondary one.

    See :func:`secondary_xaxis`; the same limitation applies to y.
    """
    _warn_unsupported_call(
        "secondary_yaxis",
        "returns the current axes, not a real secondary one: GLPlot's single viewport has "
        "no independent tick strip to draw a second y-axis on",
    )
    return _active_axes(_get_or_create_plot())


def colorbar(mappable: Optional[Any] = None, **kwargs: Any) -> None:
    """Accepted for matplotlib parity; draws nothing.

    A colorbar is a second axes beside the first, which GLPlot's single-viewport
    model has nowhere to put. The Scene panel's colormap picker shows the same
    mapping interactively.

    Returns:
        None
    """
    _get_or_create_plot()
    _warn_unsupported(
        "colorbar",
        {"colorbar": True},
        {
            "colorbar": "draws nothing: a colorbar is a second axes beside the plot, and "
            "GLPlot renders one viewport with no room beside it. The Scene panel's "
            "colormap picker shows the same mapping"
        },
    )


def tight_layout(**kwargs: Any) -> None:
    """Accepted for matplotlib parity; does nothing.

    matplotlib repacks its axes grid to stop labels colliding. GLPlot has one
    viewport whose gutters are set by ``options.axis_margin_*``, so there is
    nothing to repack -- widen a margin instead.

    Returns:
        None
    """
    _get_or_create_plot()
    _warn_unsupported(
        "tight_layout",
        {"tight_layout": True},
        {
            "tight_layout": "does nothing: GLPlot has one viewport and no axes grid to "
            "repack. Widen options.axis_margin_l/r/b/t to give the labels room"
        },
    )


def get_cmap(name: Optional[Union[str, Any]] = None, lut: Optional[int] = None) -> Any:
    """Return a matplotlib ``Colormap`` by name.

    A thin pass-through to matplotlib's registry: GLPlot samples matplotlib's
    colormaps rather than shipping its own, so this is the same object
    ``plt.get_cmap`` returns.

    Args:
        name (str or Colormap, optional): The colormap. Defaults to whatever
            :func:`set_cmap` last chose, or 'viridis'.
        lut (int, optional): Resample the colormap to this many entries.

    Returns:
        Colormap: The matplotlib colormap.
    """
    from matplotlib import colormaps
    from matplotlib.colors import Colormap

    if isinstance(name, Colormap):
        return name
    resolved = _resolve_cmap(name, "viridis")
    cmap = colormaps[resolved]
    return cmap.resampled(int(lut)) if lut is not None else cmap


def set_cmap(cmap: Union[str, Any]) -> None:
    """Set the colormap that later plots fall back to, and re-map the current one.

    Both halves are matplotlib's behaviour: it writes ``rcParams['image.cmap']``
    *and* applies the colormap to the current image. An explicit ``cmap=`` on a
    call still wins over this.

    Args:
        cmap (str or Colormap): The colormap to make current.

    Examples:
        >>> gplt.set_cmap('plasma')
        >>> gplt.scatter(x, y, c=v)     # plasma, without naming it
        >>> gplt.scatter(x, y, c=v, cmap='jet')   # the explicit one still wins
    """
    global _CURRENT_CMAP
    from matplotlib.colors import Colormap

    name = cmap.name if isinstance(cmap, Colormap) else str(cmap)
    get_cmap(name)  # Raises here, on the caller's line, rather than at the next plot.
    _CURRENT_CMAP = name

    if _CURRENT_MAPPABLE is not None:
        # A per-vertex patch takes its own re-map path; only if it is not one do we hand it
        # to the GUI's re-mapper (which knows image and scatter-value layers).
        if not _remap_patch_mappable(_CURRENT_MAPPABLE, cmap=name):
            from .gui.layerops import set_layer_colormap

            try:
                set_layer_colormap(_get_or_create_plot(), _CURRENT_MAPPABLE, cmap=name)
            except (ValueError, TypeError, AttributeError):
                # A layer whose scalars were not retained cannot be re-mapped (see
                # `_retained_cvalues`). The fallback still stands for the plots to come.
                pass
        _set_dirty(_get_or_create_plot())


def register_cmap(name: Optional[str] = None, cmap: Optional[Any] = None, **kwargs: Any) -> None:
    """Register a colormap with matplotlib, so ``cmap='name'`` finds it here too.

    GLPlot resolves every colormap through matplotlib's registry, so a colormap
    registered by any route is usable in ``scatter(cmap=...)``, ``hexbin``, and
    the rest.

    Args:
        name (str, optional): The name to register under. Defaults to
            ``cmap.name``.
        cmap (Colormap): The colormap to register.
    """
    from matplotlib import colormaps

    if cmap is None:
        raise TypeError("register_cmap() requires a cmap")
    colormaps.register(cmap, name=name, **kwargs)


def gci() -> Optional[BaseLayer]:
    """Return the current colour-mappable layer, or None.

    matplotlib's "current image": the last layer a colormap was applied to. It is
    what :func:`clim` and a bare :func:`set_cmap` act on.

    Returns:
        Layer or None: The current mappable.
    """
    return _CURRENT_MAPPABLE


def sci(im: BaseLayer) -> None:
    """Make ``im`` the current mappable, so :func:`clim` and :func:`set_cmap` act on it.

    Args:
        im (Layer): A layer that carries a colormap.
    """
    global _CURRENT_MAPPABLE
    _CURRENT_MAPPABLE = im


def clim(vmin: Optional[float] = None, vmax: Optional[float] = None) -> Tuple[float, float]:
    """Set or get the colour limits of the current mappable.

    Args:
        vmin (float, optional): The value the colormap starts at. None leaves it.
        vmax (float, optional): The value it ends at. None leaves it.

    Returns:
        tuple: The ``(vmin, vmax)`` now in force.

    Raises:
        RuntimeError: If no colormapped layer has been plotted yet.

    Examples:
        >>> gplt.scatter(x, y, c=v)
        >>> gplt.clim(0, 1)
    """
    if _CURRENT_MAPPABLE is None:
        raise RuntimeError(
            "clim(): no colormapped layer to act on. Plot one first (scatter(c=...), "
            "hexbin, hist2d, imshow), or point sci() at one."
        )
    from .gui.layerops import layer_colormap, set_layer_colormap

    layer = _CURRENT_MAPPABLE
    values = layer.metadata.get("cvalues") if isinstance(layer.metadata, dict) else None

    # A per-vertex patch (hexbin, pcolor, tripcolor) has its own re-map path: the GUI's
    # `set_layer_colormap` knows only image and scatter-value layers and raises on a patch.
    if _remap_patch_mappable(layer, vmin=vmin, vmax=vmax):
        state = layer.metadata["_patch_cmap"]
        lo = (
            state["vmin"]
            if state["vmin"] is not None
            else (float(np.nanmin(values)) if values is not None and len(values) else float("nan"))
        )
        hi = (
            state["vmax"]
            if state["vmax"] is not None
            else (float(np.nanmax(values)) if values is not None and len(values) else float("nan"))
        )
        _set_dirty(_get_or_create_plot())
        return float(lo), float(hi)

    if vmin is not None or vmax is not None:
        kwargs = {}
        if vmin is not None:
            kwargs["vmin"] = float(vmin)
        if vmax is not None:
            kwargs["vmax"] = float(vmax)
        set_layer_colormap(_get_or_create_plot(), layer, **kwargs)

    # Read back through `layer_colormap`, the reader that matches the writer: the limits
    # live in the layer's metadata, not in `style` -- which has vmin/vmax fields that this
    # path never writes, so reading them returns None for a limit that is plainly set.
    _, lo, hi = layer_colormap(layer)
    # An unset limit means autoscale, and the number in force is then the data's own --
    # which is what the caller wants back, rather than a None they must interpret.
    if lo is None:
        lo = float(np.nanmin(values)) if values is not None and len(values) else float("nan")
    if hi is None:
        hi = float(np.nanmax(values)) if values is not None and len(values) else float("nan")
    return float(lo), float(hi)


def _make_cmap_shortcut(name: str):
    """Build one of matplotlib's one-word colormap shortcuts, e.g. ``viridis()``.

    Generated rather than hand-written nineteen times: they differ only by the name they
    pass, so a loop cannot get one of them subtly wrong, and adding matplotlib's next
    colormap alias is a one-word edit to the tuple below.
    """

    def shortcut() -> None:
        set_cmap(name)

    shortcut.__name__ = name
    shortcut.__qualname__ = name
    shortcut.__doc__ = (
        f"Set the colormap to '{name}'.\n\n"
        f"    matplotlib's one-word shortcut for ``set_cmap('{name}')``. Applies to the\n"
        f"    current mappable and becomes the fallback for later plots.\n\n"
        f"    Returns:\n"
        f"        None\n"
    )
    return shortcut


#: matplotlib's one-word colormap shortcuts, in the order its own docs list them.
_CMAP_SHORTCUTS = (
    "autumn",
    "bone",
    "cool",
    "copper",
    "flag",
    "gray",
    "hot",
    "hsv",
    "inferno",
    "jet",
    "magma",
    "nipy_spectral",
    "pink",
    "plasma",
    "prism",
    "spring",
    "summer",
    "viridis",
    "winter",
)

for _name in _CMAP_SHORTCUTS:
    globals()[_name] = _make_cmap_shortcut(_name)
del _name


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

    # Or the num registry would hold the last strong reference to a closed figure -- and
    # `figure(num)` would hand it back afterwards, which is worse than the leak: a closed
    # window's scene, silently reopened.
    for key in [k for k, v in _FIGURES_BY_NUM.items() if v is fig]:
        del _FIGURES_BY_NUM[key]

    if fig is _CURRENT_PLOT:
        _CURRENT_PLOT = _ALL_PLOTS[-1] if _ALL_PLOTS else None

    # Optional backend hook
    _call_if_exists(fig, ("close", "shutdown"))


def clf() -> None:
    """Clear current figure."""
    plot = _get_or_create_plot()
    plot._color_cycle_index = 0
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


# ==================================================================================
# matplotlib.pyplot module-level surface: interactive state, figure/axes management,
# rcParams, introspection and image IO. GLPlot draws into one interactive GPU
# viewport, so several of these are honest no-ops or single-axes stand-ins rather
# than the multi-axes / multi-backend machinery matplotlib has behind them.
# ==================================================================================

#: matplotlib's interactive flag. GLPlot's `show()` runs its own event loop, so this
#: does not gate drawing the way matplotlib's does; it is tracked and reported truthfully
#: so `ion()`/`ioff()`/`isinteractive()` round-trip, which some libraries check.
_INTERACTIVE = False


def ion() -> None:
    """Turn interactive mode on. See :func:`isinteractive`."""
    global _INTERACTIVE
    _INTERACTIVE = True


def ioff() -> None:
    """Turn interactive mode off."""
    global _INTERACTIVE
    _INTERACTIVE = False


def isinteractive() -> bool:
    """Return whether interactive mode is on.

    Note:
        GLPlot's ``show()`` always runs an interactive event loop regardless of
        this flag; the flag is tracked for the libraries that read it, not to gate
        drawing.
    """
    return _INTERACTIVE


class _interactive_ctx:
    """Context manager / value returned by :func:`interactive`, mirroring matplotlib."""

    def __init__(self, state: bool) -> None:
        self._state = bool(state)
        self._prev = _INTERACTIVE

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        global _INTERACTIVE
        _INTERACTIVE = self._prev


def interactive(b: bool) -> "_interactive_ctx":
    """Set interactive mode, returning a context manager that restores it on exit."""
    global _INTERACTIVE
    ctx = _interactive_ctx(b)
    _INTERACTIVE = bool(b)
    return ctx


def draw() -> None:
    """Request a redraw of the current figure.

    Marks the scene dirty so the next frame of ``show()``'s loop repaints. A no-op
    before ``show()`` -- there is no loop to signal yet.
    """
    _set_dirty(_get_or_create_plot())


def draw_if_interactive() -> None:
    """Redraw only if interactive mode is on -- matplotlib's internal hook."""
    if _INTERACTIVE:
        draw()


def draw_all(force: bool = False) -> None:
    """Redraw every open figure, not just the current one.

    Args:
        force (bool, optional): Accepted for matplotlib parity, where it redraws figures
            already marked clean. GLPlot marks every figure dirty either way, so the flag
            changes nothing here.
    """
    for fig in _ALL_PLOTS:
        _set_dirty(fig)


def pause(interval: float) -> None:
    """Accepted for matplotlib parity; does not block.

    matplotlib's ``pause`` runs the GUI event loop for ``interval`` seconds.
    GLPlot's event loop lives inside ``show()``, so outside it this is a no-op
    rather than a blocking sleep that would freeze a headless script.

    Returns:
        None
    """
    _warn_unsupported(
        "pause",
        {"pause": True if interval else None},
        {"pause": "does not run an event loop outside show(); it returns immediately"},
    )


def _stub_event_id() -> int:
    """A deterministic connection id for the event-handler stubs.

    Not `Date.now()` or a global counter that would break the test suite's determinism --
    the handlers are not wired to anything, so the id only needs to be returnable.
    """
    return 0


def connect(s: str, func: Any) -> int:
    """Accepted for matplotlib parity; the figure canvas has no matplotlib event bus.

    GLPlot handles input through its own interaction layer, not matplotlib's
    ``mpl_connect`` events, so a handler registered here is never called.

    Returns:
        int: A connection id, for symmetry with :func:`disconnect`.
    """
    _get_or_create_plot()
    _warn_unsupported(
        "connect",
        {"connect": s},
        {
            "connect": "registers a handler that is never called: GLPlot routes input through "
            "its own interaction layer, not matplotlib canvas events"
        },
    )
    return _stub_event_id()


def disconnect(cid: int) -> None:
    """Counterpart to :func:`connect`; there is nothing to disconnect. No-op."""
    _get_or_create_plot()


def ginput(n: int = 1, timeout: float = 30, **kwargs: Any) -> list:
    """Accepted for matplotlib parity; returns no points.

    matplotlib's ``ginput`` blocks for mouse clicks. GLPlot's picking is driven
    through its own interaction layer inside ``show()``, so this cannot collect
    clicks from a script and returns an empty list.

    Returns:
        list: Always empty.
    """
    _warn_unsupported(
        "ginput",
        {"ginput": True},
        {
            "ginput": "returns no points: click collection happens through GLPlot's own "
            "interaction layer inside show(), not a blocking matplotlib call"
        },
    )
    return []


def waitforbuttonpress(timeout: float = -1) -> Optional[bool]:
    """Accepted for matplotlib parity; returns None without blocking."""
    _warn_unsupported(
        "waitforbuttonpress",
        {"waitforbuttonpress": True},
        {"waitforbuttonpress": "does not block outside show(); it returns None immediately"},
    )
    return None


def install_repl_displayhook() -> None:
    """Accepted for matplotlib parity; GLPlot has no REPL display integration. No-op."""


def uninstall_repl_displayhook() -> None:
    """Counterpart to :func:`install_repl_displayhook`. No-op."""


def switch_backend(newbackend: str) -> None:
    """Accepted for matplotlib parity; GLPlot is its own GPU backend.

    Returns:
        None
    """
    _warn_unsupported(
        "switch_backend",
        {"newbackend": newbackend},
        {
            "newbackend": "has no effect: GLPlot renders through its own GPU backend and "
            "cannot switch to a matplotlib one"
        },
    )


def get_backend(*, auto_select: bool = True) -> str:
    """Return the backend name.

    Args:
        auto_select: Accepted for matplotlib parity; ignored. GLPlot has exactly one
            backend, so there is nothing to auto-select between.

    Returns:
        str: Always 'glplot'.
    """
    return "glplot"


def new_figure_manager(num: int = 1, *args: Any, **kwargs: Any) -> GPULinePlot:
    """Return a new figure. GLPlot has no separate manager, so this is a figure."""
    return figure(num=num)


def get_current_fig_manager() -> GPULinePlot:
    """Return the current figure, standing in for matplotlib's figure manager."""
    return _get_or_create_plot()


def axes(arg: Any = None, **kwargs: Any) -> AxesProxy:
    """Return the current axes, optionally setting its projection.

    ``projection="3d"`` is honoured — it switches the current panel to a 3D view, which is
    what the keyword means in matplotlib. A rectangle argument still cannot add a second
    axes at an arbitrary position; use :func:`subplots` or :func:`subplot_mosaic` for a
    grid of real panels.

    Returns an :class:`AxesProxy`, not the figure: ``ax = plt.axes(projection="3d")`` is
    the second most common way a matplotlib script obtains 3D axes, and every line after it
    is a method call on the result. See :func:`gca` for why the figure was the wrong answer.

    Returns:
        AxesProxy: The active panel, as a matplotlib-style Axes.

    Examples:
        >>> ax = gplt.axes(projection="3d")
        >>> ax.plot_surface(X, Y, Z)
    """
    projection = kwargs.pop("projection", None)
    _warn_unsupported(
        "axes",
        {"axes": True if arg is not None else None},
        {
            "axes": "returns the single viewport GLPlot has; a rectangle cannot add a "
            "second axes — use subplots() for a grid of panels"
        },
    )
    plot = _get_or_create_plot()
    _apply_projection(plot, projection)
    return _active_axes(plot)


def sca(ax: GPULinePlot) -> None:
    """Set the current axes.

    With a single viewport there is one axes, so this makes ``ax``'s figure current
    if it is a known figure, and is otherwise a no-op.
    """
    global _CURRENT_PLOT
    if ax in _ALL_PLOTS:
        _CURRENT_PLOT = ax


def delaxes(ax: Optional[GPULinePlot] = None) -> None:
    """Remove an axes. With one viewport, this clears the current figure."""
    clf()


def subplot2grid(shape, loc, rowspan: int = 1, colspan: int = 1, **kwargs: Any) -> AxesProxy:
    """Place one panel spanning ``rowspan`` x ``colspan`` cells of a ``shape`` grid.

    Real multi-panel behaviour, matching matplotlib's builder idiom: call it several times
    against the same ``shape`` to drop several (possibly spanning) axes into one grid. The
    first call for a shape starts the grid fresh; later calls with that shape append. ``loc``
    is the ``(row, col)`` of the panel's top-left cell (row 0 = top).

    ``projection="3d"`` makes the new panel 3D, exactly as in :func:`subplot`. It used to be
    accepted and dropped, so ``plt.subplot2grid((2, 2), (0, 0), projection="3d")`` handed
    back an axes that looked right and failed on the first 3D verb.

    Returns:
        AxesProxy: The new panel, also made the current axes.
    """
    projection = kwargs.pop("projection", None)
    fig_kw = kwargs.pop("fig", None)
    _warn_unsupported(
        "subplot2grid",
        {"fig": fig_kw, **kwargs},
        {"fig": "GLPlot places the panel in the current figure; pass it to figure() instead"},
    )
    fig = _get_or_create_plot()
    shape = (int(shape[0]), int(shape[1]))
    spec = _layout.grid_span(shape[0], shape[1], (int(loc[0]), int(loc[1])), rowspan, colspan)
    if getattr(fig, "_subplot2grid_shape", None) != shape:
        fig.set_panels([spec])
        fig._subplot2grid_shape = shape
    else:
        fig.add_panel(spec)
    idx = len(fig.panels) - 1
    panel = fig.panels[idx]
    proxy = AxesProxy(fig, idx, name=panel.name, row=panel.row, col=panel.col)
    proxy._activate()
    _apply_projection(fig, projection)
    return proxy


def _parse_mosaic(mosaic) -> "List[List[str]]":
    """Normalise a mosaic spec to a 2-D list of cell labels.

    Accepts matplotlib's compact string (rows split on ``;`` or newline, one character per
    cell) and the nested-list form. ``"."`` marks an empty cell.
    """
    if isinstance(mosaic, str):
        rows = [r for r in mosaic.replace(";", "\n").split("\n") if r.strip()]
        return [list(r) for r in rows]
    return [list(r) for r in mosaic]


def subplot_mosaic(mosaic, **kwargs: Any):
    """Create panels from a mosaic layout and return ``(fig, {name: axes})``.

    Real multi-panel behaviour: each distinct label becomes one panel spanning the block of
    cells it covers, so ``"AB;CD"`` gives four side-by-side axes and ``[["A", "A"], ["B", "C"]]``
    gives a wide ``A`` above ``B`` and ``C``. ``"."`` is an empty cell.

    ``subplot_kw={"projection": "3d"}`` applies to every panel; ``per_subplot_kw`` overrides
    it per label, keyed by a label or a tuple of labels as in matplotlib. Both used to be
    accepted and dropped, which meant the documented way to build a mixed 2D/3D mosaic
    produced an all-2D one and the 3D calls that followed drew nothing recognisable.

    Args:
        mosaic: A layout string (``"AB;CD"``) or a nested list of labels.
        subplot_kw (dict, optional): Applied to every panel. Only ``projection`` is acted on.
        per_subplot_kw (dict, optional): ``{label: kw}`` or ``{(label, label): kw}``,
            overriding ``subplot_kw`` for those panels.

    Returns:
        tuple: ``(figure, {name: AxesProxy})``.
    """
    subplot_kw = dict(kwargs.pop("subplot_kw", None) or {})
    per_subplot_kw = dict(kwargs.pop("per_subplot_kw", None) or {})
    empty_sentinel = kwargs.pop("empty_sentinel", ".")
    _warn_unsupported(
        "subplot_mosaic",
        {k: v for k, v in kwargs.items()},
        {
            "sharex": "panels do not share axes here; use subplots(sharex=True)",
            "sharey": "panels do not share axes here; use subplots(sharey=True)",
        },
    )
    if empty_sentinel != ".":
        _warn_unsupported(
            "subplot_mosaic",
            {"empty_sentinel": empty_sentinel},
            {"empty_sentinel": "is fixed at '.' by the layout parser"},
        )

    # Flatten matplotlib's tuple-keyed form ({("A", "B"): kw}) to one entry per label, so
    # the lookup below is a plain dict get rather than a scan with a membership test.
    per_label: dict = {}
    for key, value in per_subplot_kw.items():
        for label in key if isinstance(key, tuple) else (key,):
            per_label[label] = dict(value or {})

    fig = _get_or_create_plot()
    specs = _layout.mosaic(_parse_mosaic(mosaic))
    fig.set_panels(specs)
    axd = {}
    saved = fig.active_panel_index
    for i, s in enumerate(specs):
        axd[s.name] = AxesProxy(fig, i, name=s.name, row=s.row, col=s.col)
        merged = {**subplot_kw, **per_label.get(s.name, {})}
        projection = merged.pop("projection", None)
        _warn_unsupported("subplot_mosaic", merged)
        if projection is not None:
            # ``set_ndim`` acts on the active panel, so each panel is made current in turn.
            fig.active_panel_index = i
            _apply_projection(fig, projection)
    fig.active_panel_index = saved if 0 <= saved < len(fig.panels) else 0
    return fig, axd


def subplot_tool(*args: Any, **kwargs: Any) -> None:
    """Accepted for matplotlib parity; there is no subplot layout to tune. No-op."""


def subplots_adjust(**kwargs: Any) -> None:
    """Adjust the viewport margins.

    matplotlib's ``left``/``right``/``bottom``/``top`` are figure fractions; GLPlot
    reserves its gutters in pixels (``options.axis_margin_*``). The fractions are
    accepted and ignored -- set the pixel margins directly to move the frame.

    Returns:
        None
    """
    _get_or_create_plot()
    _warn_unsupported(
        "subplots_adjust",
        {
            k: v
            for k, v in kwargs.items()
            if k in ("left", "right", "top", "bottom", "wspace", "hspace")
        },
        {
            k: "is a figure fraction; GLPlot's gutters are pixels (options.axis_margin_*)"
            for k in ("left", "right", "top", "bottom", "wspace", "hspace")
        },
    )


def figaspect(arg: Union[float, np.ndarray]) -> Tuple[float, float]:
    """Return a figure ``(width, height)`` matching an aspect ratio.

    A pure helper, identical to matplotlib's: given a height/width ratio or a 2-D
    array whose shape sets one, return sensible figure inches.

    Args:
        arg (float or array): The aspect ratio, or an array to match.

    Returns:
        tuple: ``(width, height)`` in inches.

    Examples:
        >>> w, h = gplt.figaspect(0.5)
        >>> w, h = gplt.figaspect(image)
    """
    if hasattr(arg, "shape") and np.ndim(arg) == 2:
        nr, nc = arg.shape[:2]
        aspect = nr / nc
    else:
        aspect = float(arg)
    # matplotlib's own bounds: a figure between 2 and 16 inches on its larger side.
    base = 6.0
    w = base
    h = base * aspect
    scale = max(min(max(w, h), 16.0) / max(w, h), 2.0 / min(w, h)) if min(w, h) else 1.0
    return (w * scale, h * scale)


def figlegend(*args: Any, **kwargs: Any):
    """Draw a legend for the figure. In a single-axes figure this is :func:`legend`."""
    return legend(*args, **kwargs)


def figtext(x: float, y: float, s: str, fontdict: Optional[dict] = None, **kwargs: Any):
    """Place text at a figure-relative position ``(x, y)`` in 0..1.

    matplotlib positions this in figure fractions. GLPlot maps the fraction onto
    the current view so the text lands where the fraction points, though it then
    scrolls with the data rather than staying pinned to the figure.

    Args:
        x, y (float): Position as a fraction of the figure, 0..1.
        s (str): The text.
        fontdict (dict, optional): matplotlib parity. Merged into the styling below, with
            the explicit ``fontsize``/``color`` kwargs winning.

    Returns:
        Layer: The text layer.
    """
    plot = _get_or_create_plot()
    left, right = plot.get_xlim()
    bottom, top = plot.get_ylim()
    fontsize, color = _merge_fontdict(
        "figtext", fontdict, kwargs.pop("fontsize", None), kwargs.pop("color", None)
    )
    color = "k" if color is None else color
    return text(
        left + (right - left) * float(x),
        bottom + (top - bottom) * float(y),
        str(s),
        color=color,
        fontsize=fontsize,
    )


def fignum_exists(num: Union[int, str]) -> bool:
    """Return whether a figure with identifier ``num`` exists."""
    return num in _FIGURES_BY_NUM


def get_fignums() -> list:
    """Return the identifiers of all figures created with a ``num``."""
    return [n for n in _FIGURES_BY_NUM if isinstance(n, int)]


def get_figlabels() -> list:
    """Return the string labels of all figures created with a string ``num``."""
    return [n for n in _FIGURES_BY_NUM if isinstance(n, str)]


def bar_label(
    container: Optional[Sequence[Any]] = None,
    labels: Optional[Sequence[str]] = None,
    *,
    fmt: str = "%g",
    label_type: str = "edge",
    padding: float = 0.0,
    bars: Optional[Sequence[Any]] = None,
    **kwargs: Any,
) -> list:
    """Label each bar of a :func:`bar`/:func:`barh` result with its value.

    Args:
        container (sequence): The layers :func:`bar` returned. matplotlib names this
            parameter ``container``, and ``bar_label(container=...)`` used to be a
            TypeError here.
        labels (sequence of str, optional): Explicit labels. Defaults to each
            bar's height formatted with ``fmt``.
        fmt (str, optional): printf format for the default labels. Defaults to '%g'.
        label_type (str, optional): ``'edge'`` puts the label just outside the bar end
            (the default), ``'center'`` puts it in the middle of the bar.
        padding (float, optional): Extra gap above the bar, in data units.
        bars (sequence, optional): GLPlot's own spelling of ``container``, kept working
            for code written against GLPlot <= 0.1.3.

    Returns:
        list: The text layers.

    Examples:
        >>> bars = gplt.bar([0, 1, 2], [10, 24, 18])
        >>> gplt.bar_label(bars, fmt='%.1f', label_type='center')
    """
    container = container if container is not None else bars
    if container is None:
        raise TypeError("bar_label() missing required argument: 'container'")
    if label_type not in ("edge", "center"):
        raise ValueError(f"bar_label(): label_type must be 'edge' or 'center', got {label_type!r}")
    _warn_unsupported("bar_label", kwargs)

    texts = []
    for i, bar in enumerate(container):
        verts = getattr(bar, "vertices", None)
        if verts is None:
            continue
        cx = 0.5 * (verts[:, 0].min() + verts[:, 0].max())
        bottom, top = verts[:, 1].min(), verts[:, 1].max()
        # The value a bar reports is its end, wherever the label is drawn -- 'center' moves
        # the text, not the number it shows.
        text_str = labels[i] if labels is not None else (fmt % top)
        ty = (0.5 * (bottom + top)) if label_type == "center" else (top + padding)
        texts.append(text(cx, ty, str(text_str)))
    return texts


# --- rcParams and introspection ---------------------------------------------------


def rc(group: str, **kwargs: Any) -> None:
    """Set matplotlib rcParams, a few of which GLPlot reads live.

    GLPlot renders through its own GPU backend and reads only the rcParams it has an
    equivalent for: ``axes.prop_cycle`` (the colours ``plot`` cycles through),
    ``lines.linewidth``, and ``figure.dpi`` (the default DPI a subsequent ``figure()``
    call resolves to, when it is not given one explicitly). ``figure.figsize`` and
    ``axes.grid`` are matplotlib defaults GLPlot does *not* apply automatically --
    ``figure()``/``subplots()`` keep their own GLPlot-native default size (``width=1280,
    height=800``) unless ``figsize=``/``width=``/``height=`` is passed explicitly, and a
    new panel's grid visibility is unaffected by this rcParam; pass ``figsize=`` or call
    :func:`grid` directly instead of relying on the rcParam for either.

    The exported PNG is also rendered through matplotlib whenever no GL window exists yet
    (``savefig()``'s headless path), so everything else set here -- fonts, tick styling,
    hatch density, legend framing -- reaches *that* figure the normal matplotlib way, even
    though it has no live-GPU-render equivalent.

    Args:
        group (str): The rc group, e.g. 'lines', 'font', 'axes'.
        **kwargs: Properties within the group, e.g. ``linewidth=2``.

    Examples:
        >>> gplt.rc('lines', linewidth=2)
        >>> gplt.rc('axes', prop_cycle=gplt.cycler(color=['r', 'g', 'b']))
        >>> gplt.rc('figure', dpi=150)
        >>> gplt.figure()  # picks up dpi=150 since none was passed explicitly
    """
    import matplotlib as mpl

    mpl.rc(group, **kwargs)


def rc_context(rc: Optional[dict] = None, fname: Optional[str] = None):
    """Return a context manager that temporarily sets rcParams. See :func:`rc`."""
    import matplotlib as mpl

    return mpl.rc_context(rc=rc, fname=fname)


def rcdefaults() -> None:
    """Restore matplotlib's default rcParams."""
    import matplotlib as mpl

    mpl.rcdefaults()


def _iter_all_layers():
    """Every layer across every live figure -- the search space for setp/getp/findobj."""
    for fig in _ALL_PLOTS:
        scene = getattr(fig, "scene", None)
        if scene is not None:
            yield from scene.layers


def setp(obj: Any, *args: Any, **kwargs: Any) -> Optional[list]:
    """Set a property on one artist or a list of them.

    ``setp(layer, alpha=0.5)`` sets it; ``setp(layer)`` lists the settable
    properties. Reaches the layer's style fields (alpha, color, linewidth,
    visible, zorder, label).

    Args:
        obj: A layer or a list of layers.
        *args: An alternating ``name, value, name, value`` sequence, as matplotlib
            also accepts.
        **kwargs: ``property=value`` pairs.

    Returns:
        list or None: The property names when called with no values, else None.
    """
    targets = obj if isinstance(obj, (list, tuple)) else [obj]
    props = dict(kwargs)
    props.update({args[i]: args[i + 1] for i in range(0, len(args) - 1, 2)})
    if not props:
        return ["alpha", "color", "linewidth", "visible", "zorder", "label"]

    for target in targets:
        style = getattr(target, "style", None)
        for name, value in props.items():
            if name == "label":
                target.label = str(value)
            elif name == "color" and style is not None:
                style.color = _normalize_rgba(value)
            elif style is not None and hasattr(style, name):
                setattr(style, name, value)
        if hasattr(target, "dirty"):
            target.dirty.gpu_dirty = True
    _set_dirty(_get_or_create_plot())
    return None


def getp(obj: Any, property: Optional[str] = None) -> Any:
    """Get a property of an artist, or list them all.

    Args:
        obj: A layer.
        property (str, optional): The property to read. None lists them.

    Returns:
        The property value, or a dict of all properties when ``property`` is None.
    """
    style = getattr(obj, "style", None)
    names = ["alpha", "color", "linewidth", "visible", "zorder"]
    if property is None:
        out = {"label": getattr(obj, "label", None)}
        for name in names:
            if style is not None and hasattr(style, name):
                out[name] = getattr(style, name)
        return out
    if property == "label":
        return getattr(obj, "label", None)
    return getattr(style, property, None) if style is not None else None


def get(obj: Any, property: Optional[str] = None) -> Any:
    """Alias of :func:`getp`."""
    return getp(obj, property)


def findobj(o: Any = None, match: Any = None, **kwargs: Any) -> list:
    """Find artists matching a predicate.

    Args:
        o: A layer to search within, or None for every layer in every figure.
        match: A callable predicate ``layer -> bool``, or a type to match by
            ``layer_type``. None matches everything.

    Returns:
        list: The matching layers.
    """
    if o is None:
        candidates = list(_iter_all_layers())
    elif isinstance(o, (list, tuple)):
        candidates = list(o)
    else:
        scene = getattr(o, "scene", None)
        candidates = list(scene.layers) if scene is not None else [o]

    if match is None:
        return candidates
    if callable(match):
        return [c for c in candidates if match(c)]
    return [c for c in candidates if getattr(c, "layer_type", None) == match]


def set_loglevel(level: str) -> None:
    """Set GLPlot's logging level.

    Args:
        level (str): 'debug', 'info', 'warning', 'error' or 'critical'.
    """
    import logging

    logging.getLogger("glplot").setLevel(str(level).upper())


def get_plot_commands() -> list:
    """Return the names of the plotting commands GLPlot exposes.

    Returns:
        list: Sorted public function names, matching matplotlib's introspection
        helper.
    """
    import inspect as _inspect

    module = _sys.modules[__name__]
    return sorted(
        name
        for name, obj in vars(module).items()
        if not name.startswith("_") and _inspect.isfunction(obj)
    )


def get_scale_names() -> list:
    """Return the axis scale names GLPlot recognises.

    Returns:
        list: The scale names :func:`xscale` accepts. Only 'linear' is drawn; see
        :func:`xscale`.
    """
    return list(_SCALE_NAMES)


def xkcd(scale: float = 1, length: float = 100, randomness: float = 2):
    """Accepted for matplotlib parity; GLPlot has no sketch-style renderer.

    Returns:
        A context manager, so ``with gplt.xkcd():`` works, though the style is not
        applied.
    """
    _warn_unsupported(
        "xkcd",
        {"xkcd": True},
        {"xkcd": "has no effect: GLPlot has no hand-drawn sketch renderer"},
    )
    from contextlib import nullcontext

    return nullcontext()


# --- image IO ---------------------------------------------------------------------


def imread(fname: Any, format: Optional[str] = None) -> np.ndarray:
    """Read an image from a file into an array. Delegates to matplotlib.

    Args:
        fname: A filename or file-like object.
        format (str, optional): The image format, inferred from the name if None.

    Returns:
        ndarray: The image, shape (M, N), (M, N, 3) or (M, N, 4).
    """
    import matplotlib.image as mimage

    return mimage.imread(fname, format=format)


def imsave(fname: Any, arr: ArrayLike, **kwargs: Any) -> None:
    """Save an array as an image file. Delegates to matplotlib.

    Args:
        fname: The output filename.
        arr (array-like): The image data.
        **kwargs: ``cmap``, ``vmin``, ``vmax``, ``origin``, ``dpi`` -- forwarded
            to matplotlib.
    """
    import matplotlib.image as mimage

    if "cmap" in kwargs:
        kwargs["cmap"] = _resolve_cmap(kwargs["cmap"], "viridis")
    mimage.imsave(fname, np.asarray(arr), **kwargs)


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
    cmap: str = "magma",
):
    """
    Backward-compatible alias for line family plotting.

    ``cmap`` only affects the headless density-image reconstruction
    (``render_preview()``'s ``line_family`` branch) -- the live GL view's density
    accumulation shader is unrelated and unaffected. Defaults to "magma" to match
    the historical hardcoded value.
    """
    plot = _get_or_create_plot()

    a_arr = _as_float_array(a, ndim=1, name="a")
    b_arr = _as_float_array(b, ndim=1, name="b")
    if len(a_arr) != len(b_arr):
        raise ValueError("a and b must have the same length")

    ab = np.column_stack([a_arr, b_arr]).astype(np.float32, copy=False)
    cols = None if colors is None else _as_float_array(colors, ndim=2, name="colors")
    plot.set_lines_ab(ab, x_range=x_range, colors=cols)
    plot.scene.layers[-1].metadata["cmap"] = cmap
    _set_dirty(plot)
    return plot


#: Which data axis the ``zdir=`` values run along, as the permutation to apply to
#: ``(xs, ys, zs)``. Copied from ``mpl_toolkits.mplot3d.art3d.juggle_axes`` so a script
#: that draws a curve on the x = const wall here draws it on the same wall there.
_ZDIR_PERMUTATIONS = {
    "x": (2, 0, 1),
    "y": (0, 2, 1),
    "z": (0, 1, 2),
    "-x": (1, 2, 0),
    "-y": (2, 0, 1),
    "-z": (0, 1, 2),
}


def _juggle_axes(
    xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, zdir: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reorder ``(xs, ys, zs)`` so the 2D pair lies in the plane orthogonal to ``zdir``.

    matplotlib's ``zdir`` is not decoration: ``ax.plot(x, y, zs=0, zdir="y")`` is how the
    mplot3d gallery projects a 2D curve onto a wall of the 3D box, and reading it as "z"
    would put the curve on the floor instead. A ``"-"`` prefix is matplotlib's inverse of
    :func:`rotate_axes`, kept for the same reason.
    """
    key = str(zdir)
    if key not in _ZDIR_PERMUTATIONS:
        raise ValueError(
            f"zdir={zdir!r} is not a valid value; expected one of "
            + ", ".join(repr(name) for name in _ZDIR_PERMUTATIONS)
        )
    triple = (xs, ys, zs)
    i, j, k = _ZDIR_PERMUTATIONS[key]
    return triple[i], triple[j], triple[k]


def _broadcast_zs(zs: Any, n: int) -> np.ndarray:
    """``zs`` as one value per point. matplotlib's default ``zs=0`` is a scalar."""
    arr = np.atleast_1d(np.asarray(zs, dtype=np.float32))
    if arr.size == 1:
        return np.full(n, float(arr.ravel()[0]), dtype=np.float32)
    if arr.size != n:
        raise ValueError(f"zs has length {arr.size}, expected {n} (one per point) or a scalar")
    return arr.astype(np.float32, copy=False).ravel()


def _plot_3d_call(args: tuple, kwargs: dict) -> Optional[list]:
    """Handle ``plot`` on a 3D axes the way ``Axes3D.plot`` does, or return None.

    Returns None when the call is not the 3D form (fewer than two data arguments, or the
    multi-group ``plot(x1, y1, fmt1, x2, y2, ...)`` spelling that ``Axes3D.plot`` does not
    have), so the caller falls back to the 2D path rather than this guessing.

    The parse is matplotlib's, including the rule that decides whether the third positional
    is ``zs`` or a format string: *not a string* means it is data. That rule is the whole
    point of this function. Without it ``ax.plot(t, t, t)`` on 3D axes went to the 2D
    parser, which read the arguments as two flat series and drew a plausible-looking 2D
    picture with no warning -- the worst failure mode available, because nothing about the
    output says it is wrong.
    """
    if len(args) < 2 or isinstance(args[0], str) or isinstance(args[1], str):
        return None
    rest = list(args[2:])
    # Every "is this the 3D form?" test happens before anything is consumed, so a call that
    # turns out not to be one leaves ``kwargs`` exactly as the caller wrote it for the 2D
    # path to parse. Popping first and bailing later is how a half-consumed ``zdir``
    # disappears from a call that then draws in 2D.
    zs_is_positional = bool(rest) and not isinstance(rest[0], str)
    tail = rest[1:] if zs_is_positional else rest
    if len(tail) > 1 or (tail and not isinstance(tail[0], str)):
        return None  # A second (x, y) group: not a shape Axes3D.plot accepts.
    if zs_is_positional and "zs" in kwargs:
        raise TypeError("plot() got multiple values for argument 'zs'")

    zdir = kwargs.pop("zdir", "z")
    zs = rest[0] if zs_is_positional else kwargs.pop("zs", 0)
    fmt = tail[0] if tail else None

    xs = _as_float_array(args[0], ndim=1, name="x")
    ys = _as_float_array(args[1], ndim=1, name="y")
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    zs_arr = _broadcast_zs(zs, len(xs))
    xs, ys, zs_arr = _juggle_axes(xs, ys, zs_arr, zdir)
    fmt_args = () if fmt is None else (fmt,)
    return plot3d(xs, ys, zs_arr, *fmt_args, **kwargs)


def plot(*args: Any, **kwargs: Any) -> list[BaseLayer]:
    """Plot one or more connected polylines with optional markers.

    Supports matplotlib-style flexible argument parsing for easy line plotting.
    Accepts single or multiple datasets with optional format strings controlling
    color, line style, and marker style. Format strings follow matplotlib
    conventions: 'r-' for red line, 'bo' for blue circles, 'g--' for green dashes.

    **On 3D axes this is ``Axes3D.plot``**: the third positional argument is ``zs``, and
    ``zs=``/``zdir=`` are honoured, so ``ax.plot(x, y, z)`` draws a line *in 3D* exactly as
    matplotlib does. It used to fall through to the 2D parser and draw a flat picture with
    ``z`` read as a second series.

    Args:
        *args: Variable length argument list supporting:
            - plot(y): Plot y vs auto-generated x indices
            - plot(x, y): Plot y vs x
            - plot(x, y, fmt): Plot with format string
            - plot(x1, y1, fmt1, x2, y2, fmt2, ...): Multiple datasets
            - plot(x, y, z) / plot(x, y, z, fmt): a 3D line, on 3D axes only

    Keyword Arguments:
        color (str or tuple): RGBA color. Named colors ('red', 'blue'),
            hex strings, or (r, g, b) / (r, g, b, a) tuples. Defaults to black.
        linestyle (str, optional): Line style: '-' (solid), '--' (dashes),
            '-.' (dash-dot), ':' (dots). Defaults to '-'.
        marker (str, optional): Marker style: 'o' (circle), 's' (square),
            '^' (triangle), '+' (plus), 'x' (cross), etc. Defaults to None.
        linewidth, lw (float, optional): Line width in pixels. Defaults to 1.0.
        markersize, ms (float, optional): Marker size in pixels. Defaults to 6.0.
        alpha (float, optional): Transparency (0.0-1.0). Defaults to 1.0.
        label (str, optional): Legend label for this line. Defaults to None.

    Returns:
        list: Layer objects added to the plot.

    Raises:
        TypeError: If no data arguments provided.
        ValueError: If x and y have different lengths.

    Examples:
        Simple line plot:

        >>> gplt.plot([1, 2, 3, 4], [1, 4, 2, 3])
        >>> gplt.show()

        Using format string (matplotlib style):

        >>> gplt.plot([1, 2, 3], [1, 4, 2], 'r--', label='Data')

        Multiple datasets on same plot:

        >>> gplt.plot([1, 2, 3], [1, 4, 2], 'b-', [1, 2, 3], [3, 1, 2], 'r:')

        With custom styling:

        >>> gplt.plot([1, 2, 3], [1, 4, 2], color='purple', linewidth=2.5,
        ...           marker='o', markersize=8, label='Values')

        On 3D axes (matplotlib's Axes3D.plot):

        >>> ax = gplt.subplot(projection="3d")
        >>> ax.plot(x, y, z)
        >>> ax.plot(x, y, zs=0, zdir="y")   # the curve projected onto a wall
    """
    if not args:
        raise TypeError("plot() missing data")
    if _get_or_create_plot().is_3d_scene():
        artists_3d = _plot_3d_call(args, kwargs)
        if artists_3d is not None:
            return artists_3d
    if "zs" in kwargs or "zdir" in kwargs:
        # Reached only when the call is not the 3D form -- 2D axes, or the multi-group
        # spelling. Dropping these in silence is how a script that *thought* it was in 3D
        # gets a flat plot and no clue why.
        _warn_unsupported(
            "plot",
            {"zs": kwargs.pop("zs", None), "zdir": kwargs.pop("zdir", None)},
            {
                "zs": "was ignored: it only applies to a 3D plot(x, y, z) call on axes "
                "created with projection='3d'",
                "zdir": "was ignored: it only applies to a 3D plot(x, y, z) call on axes "
                "created with projection='3d'",
            },
        )
    # Only an explicit `scalex=False` is worth a warning: the default True is exactly what
    # GLPlot does, so reporting it would fire on every plot() call in the codebase.
    scalex, scaley = kwargs.pop("scalex", None), kwargs.pop("scaley", None)
    _warn_unsupported(
        "plot",
        {
            "scalex": True if scalex is False else None,
            "scaley": True if scaley is False else None,
        },
        {
            "scalex": "has no effect: GLPlot's autoscale fits every visible layer, so a "
            "line cannot opt out of the x fit. Pin the range with xlim() instead",
            "scaley": "has no effect: GLPlot's autoscale fits every visible layer, so a "
            "line cannot opt out of the y fit. Pin the range with ylim() instead",
        },
    )
    # `data=` is matplotlib's labelled-container indirection -- plot('height', 'mass',
    # data=df). It has to resolve before `_parse_plot_groups`, which reads a string as a
    # format spec and would otherwise try to float('height').
    data = kwargs.pop("data", None)
    if data is not None:
        args = _resolve_plot_data_args("plot", data, args)
    artists = []
    for x, y, fmt in _parse_plot_groups(args):
        artists.extend(_plot_single(x, y, fmt, **kwargs))
    return artists


# ======================================================================================
# 3D axes control (matplotlib's ``Axes3D`` surface)
# ======================================================================================
#
# Everything below drives the current panel's :class:`~glplot.core.camera3d.Camera3D` and
# :class:`~glplot.core.camera3d.Axes3DOptions`. They are module-level functions rather
# than methods so ``AxesProxy.__getattr__`` picks them up automatically: ``ax.view_init(...)``
# on a subplot activates that panel and then calls the same function, which is how every
# other axes-level verb in this module already works.


def _resolve_projection(projection: Optional[Any]) -> Optional[int]:
    """Map matplotlib's ``projection=`` value to a GLPlot ``ndim``.

    ``"3d"`` -> 3, ``None``/``"rectilinear"``/``"2d"`` -> 2, anything else is refused.
    Refused rather than ignored: silently drawing a Cartesian plot for
    ``projection="polar"`` is the kind of "success" that costs an afternoon.
    """
    if projection is None:
        return None
    name = str(projection).strip().lower()
    if name in ("3d", "3-d", "three_d"):
        return 3
    if name in ("rectilinear", "2d", "none", ""):
        return 2
    raise ValueError(
        f"projection={projection!r} is not supported. GLPlot has 'rectilinear' (2D) and "
        f"'3d'; polar plots are available as gplt.polar(), which draws into 2D axes."
    )


def _apply_projection(plot_obj: GPULinePlot, projection: Optional[Any]) -> None:
    """Set the current panel's dimensionality from a ``projection=`` keyword."""
    ndim = _resolve_projection(projection)
    if ndim is not None:
        plot_obj.set_ndim(ndim)


def set_projection(projection: Optional[str]) -> None:
    """Make the current axes 2D or 3D. ``projection="3d"`` is the matplotlib spelling.

    Unlike matplotlib — where the projection is fixed when the axes is created — this can
    be flipped at any time, because a GLPlot panel owns both cameras all along and only
    chooses which one to draw through.

    Examples:
        >>> gplt.set_projection("3d")
        >>> gplt.scatter3d(x, y, z)
    """
    _apply_projection(_get_or_create_plot(), projection)


def view_init(
    elev: Optional[float] = None,
    azim: Optional[float] = None,
    roll: Optional[float] = None,
    vertical_axis: str = "z",
    share: bool = False,
) -> None:
    """Set the 3D viewing angles, in degrees (matplotlib's ``Axes3D.view_init``).

    The signature is matplotlib's, parameter for parameter and default for default. That
    is not cosmetic: ``vertical_axis`` and ``share`` used to sit behind a ``*``, so
    ``ax.view_init(30, 45, 0, "y")`` -- legal matplotlib -- raised ``TypeError``, and
    ``vertical_axis`` defaulted to None ("leave it as it is") where matplotlib's ``"z"``
    means a bare ``view_init()`` *restores* z as the up axis. Both are now matplotlib's.

    Args:
        elev: Elevation above the horizontal plane. ``+90`` looks straight down.
        azim: Azimuth about the vertical axis.
        roll: Rotation about the line of sight. Not in older matplotlib; harmless to omit.
        vertical_axis: Which data axis points up. ``"z"`` (default) and ``"y"`` are
            supported; matplotlib's third option ``"x"`` raises ``ValueError`` here,
            loudly, because the camera has no basis for it.
        share: Apply to every panel of the figure rather than only the current one.

    Examples:
        >>> gplt.view_init(elev=30, azim=-60)
        >>> gplt.view_init(elev=90)              # plan view
        >>> gplt.view_init(30, 45, 0, "y")       # positional, as matplotlib allows
    """
    plot_obj = _get_or_create_plot()
    targets = plot_obj.panels if share else [plot_obj.active_panel]
    saved = plot_obj.active_panel_index
    try:
        for panel in targets:
            plot_obj.active_panel_index = plot_obj.panels.index(panel)
            plot_obj.set_3d_view(elev=elev, azim=azim, roll=roll, up_axis=vertical_axis)
    finally:
        plot_obj.active_panel_index = saved
    _set_dirty(plot_obj)


#: Values ``set_proj_type`` accepts, in matplotlib's order (it names them in its error).
_PROJ_TYPES = ("persp", "ortho")


def set_proj_type(proj_type: str, focal_length: Optional[float] = None) -> None:
    """Choose the 3D projection: ``"persp"`` or ``"ortho"`` (matplotlib's spelling).

    ``focal_length`` is matplotlib's way of writing the perspective strength; when given it
    is converted to the equivalent vertical field of view (``2*atan(1/focal_length)``), so a
    script tuned against matplotlib produces the same framing here.

    The validation mirrors ``Axes3D.set_proj_type`` exactly, including which combinations
    are refused:

    * ``proj_type`` must be ``"persp"`` or ``"ortho"``; anything else is a ``ValueError``
      rather than a silent fallback, and it is required rather than defaulted so a typo in
      the keyword name surfaces as a ``TypeError`` instead of quietly selecting perspective.
    * ``"ortho"`` accepts only ``focal_length=None`` or ``np.inf``. An orthographic camera
      has no focal length -- ``inf`` is matplotlib's spelling of "infinitely far away", and
      it is *documented*, so it is the form a ported script contains.

    That last rule is the one that mattered. This function used to derive a field of view
    from ``focal_length`` unconditionally: ``set_proj_type("ortho", np.inf)`` set
    ``fov = degrees(2*atan(1/inf)) = 0``, and every projection matrix built afterwards
    divides by ``tan(fov/2)``, so the figure was bricked -- every later 3D call, including
    ones that never mention the projection, raised ``ZeroDivisionError`` from inside the
    camera. No fov is derived for an orthographic camera now, and a non-positive
    ``focal_length`` is refused before it can reach the camera at all.

    A ``"persp"`` call with no ``focal_length`` leaves the field of view alone rather than
    resetting it to matplotlib's default of 1 (90 degrees). GLPlot's default fov is 42, and
    ``set_proj_type("persp")`` means "use perspective", not "and re-frame the scene".

    Examples:
        >>> gplt.set_proj_type("ortho")           # parallel projection, lengths comparable
        >>> gplt.set_proj_type("ortho", np.inf)   # matplotlib's own spelling of the same
        >>> gplt.set_proj_type("persp", 0.2)      # wide angle
    """
    if proj_type not in _PROJ_TYPES:
        raise ValueError(
            f"{proj_type!r} is not a valid value for proj_type; supported values are "
            + ", ".join(repr(name) for name in _PROJ_TYPES)
        )
    fov: Optional[float] = None
    if proj_type == "persp":
        if focal_length is not None:
            focal = float(focal_length)
            if focal <= 0:
                raise ValueError(f"focal_length = {focal_length} must be greater than 0")
            fov = float(np.degrees(2.0 * np.arctan(1.0 / focal)))
    elif focal_length is not None and focal_length != np.inf:
        # ``!= np.inf`` rather than a float conversion: matplotlib's test is
        # ``focal_length not in (None, np.inf)``, so a non-numeric value must reach the
        # ValueError below rather than blow up in ``float()`` with a different exception.
        raise ValueError(f"focal_length = {focal_length} must be None for proj_type = {proj_type}")

    plot_obj = _get_or_create_plot()
    plot_obj.set_3d_view(projection=proj_type)
    if fov is not None:
        plot_obj.set_3d_view(fov=fov)
    _set_dirty(plot_obj)


def get_proj_type() -> str:
    """The current 3D projection, as ``"persp"`` or ``"ortho"``."""
    return "ortho" if _get_or_create_plot().camera3d.projection == "orthographic" else "persp"


def set_box_aspect(aspect: Optional[Any] = None, *, zoom: float = 1.0) -> None:
    """Set the relative on-screen lengths of the three axes.

    ``None`` restores raw data units — GLPlot's default, and the reason a z spanning a
    million dwarfs an x spanning one unless you say otherwise. ``(1, 1, 1)`` is
    matplotlib's cube-normalised look. The tick numbers are unaffected: this changes the
    picture, not the data.

    ``zoom`` scales the framing afterwards, as in matplotlib (``zoom > 1`` fills more of
    the frame).

    Examples:
        >>> gplt.set_box_aspect((1, 1, 1))   # read three unrelated quantities together
        >>> gplt.set_box_aspect(None)        # back to data units
    """
    plot_obj = _get_or_create_plot()
    if aspect is None:
        plot_obj.camera3d.box_aspect = None
    else:
        values = np.asarray(aspect, dtype=np.float64).ravel()
        if values.size != 3:
            raise ValueError(f"box aspect must have three components, got {values.size}")
        if np.any(values <= 0) or not np.all(np.isfinite(values)):
            raise ValueError(f"box aspect components must be positive and finite, got {aspect!r}")
        plot_obj.camera3d.box_aspect = (float(values[0]), float(values[1]), float(values[2]))
    plot_obj.set_3d_view()
    if zoom and float(zoom) != 1.0:
        plot_obj.camera3d.frame(plot_obj.padded_3d_bounds(), margin=1.0 / float(zoom))
        plot_obj.set_3d_view()
    _set_dirty(plot_obj)


def get_box_aspect() -> Optional[Tuple[float, float, float]]:
    """The current box aspect, or None when the axes are drawn in raw data units."""
    return _get_or_create_plot().camera3d.box_aspect


# ``zlabel`` / ``get_zlabel`` live further down, beside ``xlabel`` and ``ylabel``, so the
# three axis-title functions stay together. They write both ``axes3d.zlabel`` (drawn by the
# GL 3D axis renderer) and ``plot.zlabel`` (read by the matplotlib export bridge).


#: The four keyword spellings of each 3D limit, as ``(lo, hi, lo_alias, hi_alias)``.
#:
#: matplotlib gives every limit setter two names per end -- the positional ones
#: (``left``/``right`` for x, ``bottom``/``top`` for y and z) and the axis-prefixed
#: aliases (``xmin``/``xmax``, ``ymin``/``ymax``, ``zmin``/``zmax``). Both appear in real
#: scripts; ``set_zlim(zmin=0, zmax=1)`` is the form the mplot3d docs use.
_AXIS3D_LIMIT_NAMES = {
    "x": ("left", "right", "xmin", "xmax"),
    "y": ("bottom", "top", "ymin", "ymax"),
    "z": ("bottom", "top", "zmin", "zmax"),
}


def _set_axis3d_limit(
    axis: str, lo: Optional[Any] = None, hi: Optional[Any] = None, **kwargs: Any
) -> Tuple[float, float]:
    """Shared body of :func:`set_xlim3d` / :func:`set_ylim3d` / :func:`set_zlim`.

    Both keyword spellings of each end are honoured, and an unknown keyword is a
    ``TypeError`` rather than a shrug. Only the positional names used to be popped and the
    rest were dropped on the floor, which made ``set_zlim(zmin=-3, zmax=3)`` a *silent
    no-op*: the call returned the unchanged data bounds, and the caller read their own
    ``zmin`` back in their source and believed it had taken.
    """
    plot_obj = _get_or_create_plot()
    lo_name, hi_name, lo_alias, hi_alias = _AXIS3D_LIMIT_NAMES[axis]
    for value, name, alias in ((lo, lo_name, lo_alias), (hi, hi_name, hi_alias)):
        # matplotlib's own guard: naming the same end twice is a mistake, not a merge.
        if alias in kwargs and (value is not None or name in kwargs):
            raise TypeError(f"Cannot pass both {name!r} and {alias!r}")
    lo = kwargs.pop(lo_name, kwargs.pop(lo_alias, lo))
    hi = kwargs.pop(hi_name, kwargs.pop(hi_alias, hi))
    _warn_unsupported(
        f"set_{axis}lim3d",
        {"emit": kwargs.pop("emit", None), "auto": kwargs.pop("auto", None)},
        {
            "emit": "there are no limit-change callbacks to emit to",
            "auto": "3D autoscaling always follows the data; clear the limit to restore it",
        },
    )
    if kwargs:
        # matplotlib raises here too. A limit setter is not a place to be permissive: the
        # only things left in kwargs are misspellings, and swallowing one means the caller
        # asked for a range and did not get it.
        raise TypeError(
            f"set_{axis}lim() got an unexpected keyword argument "
            + ", ".join(repr(k) for k in sorted(kwargs))
        )

    axes3d = plot_obj.axes3d
    current = getattr(axes3d, f"{axis}lim")
    if lo is None and hi is None:
        if current is not None:
            return current
        bounds = plot_obj.padded_3d_bounds()
        index = {"x": 0, "y": 2, "z": 4}[axis]
        return (0.0, 1.0) if bounds is None else (bounds[index], bounds[index + 1])

    # A pair passed as the first argument, matplotlib-style: set_zlim((0, 1)).
    if hi is None and lo is not None and np.ndim(lo) == 1:
        lo, hi = float(np.asarray(lo).ravel()[0]), float(np.asarray(lo).ravel()[1])

    fallback = current or (0.0, 1.0)
    low = fallback[0] if lo is None else float(lo)
    high = fallback[1] if hi is None else float(hi)
    plot_obj.set_3d_limits(**{f"{axis}lim": (low, high)})
    _set_dirty(plot_obj)
    return (low, high)


def set_zlim(bottom: Optional[Any] = None, top: Optional[Any] = None, **kwargs: Any):
    """Pin the z range of the current 3D axes, returning ``(bottom, top)``.

    Unlike a 2D limit — which only moves the camera — a 3D limit also *clips*: geometry
    outside the range is discarded by the 3D fragment shader, so this is a genuine view
    into a slab of the data rather than a redrawn wall with points hanging through it.

    Call with no arguments to read the current range.

    Keyword Arguments:
        zmin, zmax (float, optional): matplotlib's aliases of ``bottom``/``top``. Naming
            the same end twice (``set_zlim(0, zmin=1)``) is a ``TypeError``, as it is there.
        emit, auto (optional): Accepted for parity and warned about; see
            :class:`MatplotlibCompatWarning`.

    Examples:
        >>> gplt.set_zlim(-1, 1)
        >>> gplt.set_zlim(zmin=-1, zmax=1)   # the same call, matplotlib's other spelling
        >>> gplt.set_zlim()                  # -> (-1.0, 1.0)
    """
    return _set_axis3d_limit("z", bottom, top, **kwargs)


def zlim(*args: Any, **kwargs: Any):
    """``zlim()`` reads the z range; ``zlim(lo, hi)`` sets it. See :func:`set_zlim`."""
    if not args and not kwargs:
        return _set_axis3d_limit("z")
    if len(args) == 1:
        return _set_axis3d_limit("z", args[0], **kwargs)
    return _set_axis3d_limit("z", *args[:2], **kwargs)


def set_xlim3d(left: Optional[Any] = None, right: Optional[Any] = None, **kwargs: Any):
    """Pin the x range of the current 3D axes, clipping outside it. See :func:`set_zlim`."""
    return _set_axis3d_limit("x", left, right, **kwargs)


def set_ylim3d(bottom: Optional[Any] = None, top: Optional[Any] = None, **kwargs: Any):
    """Pin the y range of the current 3D axes, clipping outside it. See :func:`set_zlim`."""
    return _set_axis3d_limit("y", bottom, top, **kwargs)


def set_zlim3d(bottom: Optional[Any] = None, top: Optional[Any] = None, **kwargs: Any):
    """Alias of :func:`set_zlim`, for matplotlib parity."""
    return _set_axis3d_limit("z", bottom, top, **kwargs)


def autoscale3d() -> None:
    """Clear every 3D axis limit and re-fit the camera to the data."""
    plot_obj = _get_or_create_plot()
    plot_obj.set_3d_limits(reset=True)
    plot_obj.frame_3d_view()
    _set_dirty(plot_obj)


def grid3d(visible: bool = True, **kwargs: Any) -> None:
    """Show or hide the 3D grid walls.

    ``which`` and ``axis`` are accepted for matplotlib parity and warned about: the 3D grid
    is ruled across all three back walls from one flag, at the major tick positions, so
    there is nothing for "minor only" or "z only" to select. They used to be dropped in
    silence, which read as though they had worked.
    """
    _warn_unsupported(
        "grid3d",
        {"which": kwargs.pop("which", None), "axis": kwargs.pop("axis", None)},
        {
            "which": "has no effect: the 3D grid is ruled at the major ticks only",
            "axis": "has no effect: the three back walls are drawn from one flag",
        },
    )
    _warn_unsupported("grid3d", kwargs)
    plot_obj = _get_or_create_plot()
    plot_obj.axes3d.show_grid = bool(visible)
    plot_obj.set_3d_view()
    _set_dirty(plot_obj)


def set_axis3d_off() -> None:
    """Hide the whole 3D decoration: box, floor, grid, ticks and labels."""
    plot_obj = _get_or_create_plot()
    plot_obj.set_3d_view(show_axes=False)
    _set_dirty(plot_obj)


def set_axis3d_on() -> None:
    """Show the 3D decoration again."""
    plot_obj = _get_or_create_plot()
    plot_obj.set_3d_view(show_axes=True)
    _set_dirty(plot_obj)


def set_axis_off() -> None:
    """Hide the axis decoration of the current axes (matplotlib's ``Axes.set_axis_off``).

    Dimension-aware, because the caller's axes may be either: a 3D panel loses its box,
    floor, grid, ticks and labels; a 2D one loses its frame, grid and tick labels. GLPlot
    already had ``set_axis3d_off`` and ``axis("off")`` for the two halves -- what was
    missing was the matplotlib spelling that every "clean render" recipe reaches for, and
    which a script cannot choose between without knowing which projection it is in.
    """
    plot_obj = _get_or_create_plot()
    if plot_obj.is_3d_scene():
        plot_obj.set_3d_view(show_axes=False)
    _set_axis_visible(plot_obj, False)


def set_axis_on() -> None:
    """Show the axis decoration again. The inverse of :func:`set_axis_off`."""
    plot_obj = _get_or_create_plot()
    if plot_obj.is_3d_scene():
        plot_obj.set_3d_view(show_axes=True)
    _set_axis_visible(plot_obj, True)


def zscale(value: str = "linear", **kwargs: Any) -> None:
    """Set the z axis scale. Only ``"linear"`` is real; the rest warn.

    The z counterpart of :func:`xscale` / :func:`yscale`, and it carries the same
    limitation for the same reason: GLPlot's data-to-screen mapping is a linear matrix
    applied in the vertex shaders, with the picking path as its exact inverse. Honouring a
    log z in one and not the other would give a plot that looks right and a cursor that
    lies. See :func:`xscale` for the full argument.
    """
    _get_or_create_plot()  # matplotlib creates the figure; a bare zscale() must too.
    name = str(value).strip().lower()
    if name not in _SCALE_NAMES:
        raise ValueError(f"unsupported scale: {value!r}. Expected one of {_SCALE_NAMES}.")
    if name != "linear":
        _warn_unsupported(
            "zscale",
            {"z": name},
            {
                "z": "has no effect: GLPlot's projection is a linear matrix and it "
                "is also the inverse used for picking, so a non-linear scale would "
                "draw one mapping and report another"
            },
        )
    _warn_unsupported("zscale", kwargs)


def _live_zticks() -> Tuple[np.ndarray, List[str]]:
    """The z tick values and labels the 3D axis renderer is currently drawing.

    Prefers the ticks ``ensure_3d_axes`` last built, and falls back to generating them the
    way it would. The order matters now that the count is adaptive: it is a product of the
    *camera* as well as the limits, so once a view exists the stored answer is the only
    faithful one — re-deriving it here without the camera would report a different axis
    from the one on screen. Before any view exists there is nothing stored, and the
    fallback's fixed count is as good an answer as the question allows.
    """
    from .renderers import axes3d as _axes3d_renderer

    plot_obj = _get_or_create_plot()
    drawn = getattr(plot_obj, "_axes3d_ticks", None)
    if drawn is not None:
        values, labels = drawn[2]
        return np.asarray(values), list(labels)
    bounds = plot_obj.padded_3d_bounds()
    if bounds is None:
        return np.zeros(0, dtype=np.float64), []
    count = int(plot_obj.axes3d.tick_count) or _axes3d_renderer.DEFAULT_TICK_COUNT
    values, labels = _axes3d_renderer.axis_ticks(bounds[4], bounds[5], count)
    return np.asarray(values), list(labels)


def zticks(
    ticks: Optional[ArrayLike] = None,
    labels: Optional[Sequence[str]] = None,
    *,
    minor: bool = False,
    **kwargs: Any,
) -> Tuple[np.ndarray, List[str]]:
    """Get or set the z-axis ticks. The z counterpart of :func:`xticks`.

    The *query* form is exact: it returns the values and labels the 3D axis renderer will
    draw, generated the same way it generates them.

    The *set* form is only partly honourable, and says so. The 3D axis renderer derives its
    ticks from the axis limits and a requested count per axis
    (``fig.axes3d.tick_count``) — it has no per-axis list of positions to override, because
    the tick marks and their labels are placed on whichever box edge currently faces the
    camera, which changes as you orbit. So ``zticks([0, 1, 2])`` sets the *count* to 3, the
    closest control that exists, and warns that the positions themselves came from the
    limits. Pin the limits with :func:`set_zlim` to control where they land.

    Returns:
        tuple: ``(locs, labels)`` after the call.
    """
    plot_obj = _get_or_create_plot()
    _warn_unsupported(
        "zticks",
        {"minor": minor or None, **kwargs},
        {"minor": "has no effect: the 3D axis draws major ticks only"},
        stacklevel=3,
    )
    if ticks is not None:
        values = np.atleast_1d(np.asarray(ticks, dtype=np.float64))
        if labels is not None and len(labels) != len(values):
            raise ValueError(
                f"zticks(): got {len(values)} ticks but {len(labels)} labels; "
                "they must be the same length"
            )
        _warn_unsupported(
            "zticks",
            {"ticks": True, "labels": True if labels is not None else None},
            {
                "ticks": "cannot pin positions: the 3D axis generates ticks from the z "
                "limits and axes3d.tick_count, so only the count was applied — use "
                "set_zlim() to control where they land",
                "labels": "has no effect: 3D tick labels are formatted from the values",
            },
        )
        if values.size:
            plot_obj.axes3d.tick_count = int(values.size)
            plot_obj.set_3d_view()
            _set_dirty(plot_obj)
    elif labels is not None:
        raise TypeError("zticks(): labels= cannot be set without ticks=")
    return _live_zticks()


def get_zticks(*, minor: bool = False) -> np.ndarray:
    """The z tick positions currently drawn, as an array (matplotlib's ``get_zticks``)."""
    _warn_unsupported(
        "get_zticks",
        {"minor": minor or None},
        {"minor": "has no effect: the 3D axis draws major ticks only"},
    )
    return _live_zticks()[0]


def zticklabels(labels: Optional[Sequence[str]] = None, **kwargs: Any) -> List[str]:
    """Get the z tick labels; setting them is accepted and warned about.

    3D tick labels are formatted from the tick values by the axis renderer as it places
    them against the box edge facing the camera, so there is no per-label override to
    write. Returning the live labels still makes the getter useful.
    """
    _warn_unsupported(
        "zticklabels",
        {"labels": labels, **kwargs},
        {"labels": "has no effect: 3D tick labels are formatted from the tick values"},
    )
    return _live_zticks()[1]


def invert_zaxis() -> None:
    """Accepted for matplotlib parity; the 3D z axis cannot be inverted. Warns.

    An inverted axis needs a limit with ``hi < lo``, and a 3D limit here is not only a
    camera setting: it is also the slab the 3D fragment shader clips against, and that
    interval is required to be ordered. Flipping it would silently discard the whole scene.
    The closest honest substitute is a viewpoint from the other side --
    ``view_init(elev=-elev)``.
    """
    _get_or_create_plot()
    _warn_unsupported_call(
        "invert_zaxis",
        "is not supported: a 3D limit is also the shader clip range, which must be "
        "ordered — use view_init(elev=-elev) to look from below",
        stacklevel=2,
    )


def view_preset(name: str = "iso") -> None:
    """Snap the 3D camera to a named orientation.

    One of ``iso``, ``top``, ``bottom``, ``front``, ``back``, ``left``, ``right``,
    ``iso_front_left``, ``iso_front_right``, ``iso_back_left``, ``iso_back_right``,
    ``corner``. Keeps the current dolly and pan — re-framing at the same time would throw
    away a zoom the caller set up deliberately (use :func:`autoscale3d` for that).

    Examples:
        >>> gplt.view_preset("top")
    """
    plot_obj = _get_or_create_plot()
    plot_obj.set_3d_preset(name)
    _set_dirty(plot_obj)


def plot_styles() -> List[Tuple[str, str, str]]:
    """The global style presets, as ``(key, name, description)`` triples.

    The catalogue :func:`plot_style` draws from — chalk, whiteboard marker, hand-drawn,
    crayon, and the publication/presentation/dark looks. The same registry the workstation's
    Style panel lists, so a look chosen from code and one chosen from a click are identical.

    Examples:
        >>> for key, name, _ in gplt.plot_styles():
        ...     print(key, "-", name)
    """
    from .gui import styles as _styles

    return [(s.key, s.name, s.description) for s in _styles.STYLES]


def plot_style(name: Optional[str] = None, *, layers: bool = True) -> str:
    """Apply a global style preset, or return the current one's key when ``name`` is None.

    A preset restyles the whole figure at once — background, palette, line width, point
    size, grid, blending and the 3D box — so ``gplt.plot_style("chalk")`` turns any figure
    into chalk on slate, in 2D and 3D alike. The same registry the Style panel uses; see
    :func:`plot_styles` for the keys.

    Args:
        name: A preset key (``"chalk"``, ``"marker"``, ``"hand"``, ``"kids"``, ...) or its
            display name, case-insensitive. ``None`` applies nothing and returns the key of
            the last preset this figure was given (``""`` if none).
        layers: When True (default) existing layers take the preset's palette, width and
            point size — except those whose colours carry data, which are never repainted.
            False restyles only the page and leaves every layer's own colour alone.

    Returns:
        The key of the applied preset (or the current one when ``name`` is None).

    Raises:
        ValueError: If ``name`` matches no preset key or display name.

    Examples:
        >>> gplt.plot(x, y)
        >>> gplt.plot_style("hand")     # paper, and a shaky-hand wobble on every line
        >>> gplt.plot_style()           # -> "hand"
    """
    from .gui import styles as _styles

    plot_obj = _get_or_create_plot()
    if name is None:
        return str(getattr(plot_obj, "_style_key", "") or "")

    key = str(name).strip()
    style = None
    try:
        style = _styles.get_style(key.lower())
    except KeyError:
        low = key.lower()
        for candidate in _styles.STYLES:
            if candidate.name.lower() == low:
                style = candidate
                break
    if style is None:
        keys = ", ".join(s.key for s in _styles.STYLES)
        raise ValueError(f"unknown style {name!r}; expected one of {keys}")

    _styles.apply_style(plot_obj, style, layers=bool(layers))
    # Record what was applied so plot_style() with no argument can report it. This is the
    # API's own bookkeeping; the Style panel tracks its selection separately (on the panel,
    # not the plot), so a preset set from code is applied but not pre-selected in the GUI.
    plot_obj._style_key = style.key
    _set_dirty(plot_obj)
    return style.key


def auto_rotate(speed: float = 24.0) -> None:
    """Spin the 3D camera at ``speed`` degrees of azimuth per second. 0 stops it.

    Only turns while the interactive loop is running (``gplt.show()``); it is a property
    of the view, so a ``savefig`` during a spin captures wherever it currently is.
    """
    plot_obj = _get_or_create_plot()
    plot_obj.camera3d.auto_spin = float(speed)
    _set_dirty(plot_obj)


def dist3d(distance: Optional[float] = None) -> float:
    """Read or set the eye-to-target distance of the 3D camera.

    Passing ``None`` reads it, resolving the automatic value against the current data, so
    the number returned is always the one actually in use.
    """
    plot_obj = _get_or_create_plot()
    if distance is not None:
        plot_obj.set_3d_view(distance=float(distance))
        _set_dirty(plot_obj)
    from .core.camera3d import bounds_centre_radius

    _, radius = bounds_centre_radius(
        plot_obj.camera3d.transform_bounds(plot_obj.padded_3d_bounds())
    )
    return plot_obj.camera3d.resolve_distance(radius)


def plot3d(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *args: Any,
    elev: float = 30.0,
    azim: float = -60.0,
    scale_z: float = 1.0,
    **kwargs: Any,
) -> BaseLayer:
    """Plot a 3D line in 3D space.

    Renders a connected line in 3D using native 3D geometry shader pipeline.
    Supports format strings and style options similar to plot(). Camera is
    automatically oriented with specified elevation and azimuth angles.

    Args:
        x (array-like): X-coordinates. Shape (N,).
        y (array-like): Y-coordinates. Shape (N,).
        z (array-like): Z-coordinates. Shape (N,).
        *args: Format string and optional additional arguments (parsed by
            _parse_plot_format).
        elev (float, optional): Camera elevation angle in degrees. Higher
            values look down from above. Defaults to 30.0.
        azim (float, optional): Camera azimuth angle in degrees. Rotation
            around vertical axis. Defaults to -60.0.
        scale_z (float, optional): Multiplies the z data before it is drawn.
            Defaults to 1.0 -- z is plotted as given, which is what matplotlib does
            and what every z read-back (``set_zlim()``, the z ticks, the box) then
            reports. It used to default to 0.7 to make the box look cube-ish, which
            rewrote the caller's data: ``scatter3d([0, 1], [0, 1], [0, 10])`` stored
            ``[0, 7]`` and the z axis was in fabricated units. Visual squashing is
            :func:`set_box_aspect`'s job -- it changes the picture, not the data.
        **kwargs: Additional keyword arguments including:
            color (str): Line color
            linewidth, lw (float): Line width
            label (str): Legend label
            alpha (float): Transparency

    Returns:
        list: Layer objects added to plot.

    Examples:
        Simple 3D line:

        >>> t = np.linspace(0, 2*np.pi, 100)
        >>> x = np.cos(t)
        >>> y = np.sin(t)
        >>> z = t / (2*np.pi)
        >>> gplt.plot3d(x, y, z, 'b-', linewidth=2)
        >>> gplt.show()

        With camera control:

        >>> gplt.plot3d(x, y, z, 'r-', elev=45, azim=120, scale_z=1.0)
    """
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
    layer.style.line_width = float(style.get("linewidth", style.get("lw", _default_linewidth())))
    return [layer]


def _scatter_2d(
    x: Sequence[float],
    y: Sequence[float],
    color: Optional[ColorLike] = None,
    size: float = 10.0,
    c: Optional[Union[ColorLike, ArrayLike]] = None,
    s: Optional[float] = None,
    cmap: Optional[str] = None,
    norm: Optional[Any] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: Optional[float] = None,
    label: Optional[str] = None,
    marker: Optional[str] = None,
    edgecolors: Optional[ColorLike] = None,
    linewidths: Optional[float] = None,
    plotnonfinite: bool = False,
    colorizer: Optional[Any] = None,
    *,
    data: Optional[Any] = None,
) -> BaseLayer:
    """Create a scatter plot with points at given (x, y) coordinates.

    Plots individual points with optional per-point coloring via colormap.
    Supports both uniform coloring and value-based colormapping for
    scientific visualization. High-performance rendering optimized for
    thousands to millions of points.

    Args:
        x (array-like): X coordinates of points. Shape (N,).
        y (array-like): Y coordinates of points. Shape (N,).
        color (str or tuple, optional): Single color for all points. Named
            colors, hex strings, or (r, g, b) / (r, g, b, a) tuples.
            Ignored if c is provided. Defaults to black.
        size (float or array-like, optional): Point size in pixels. A scalar sizes
            every marker equally; an array of length N makes size a per-point,
            data-driven dimension (each marker follows its own value). Defaults to 10.0.
        c (array-like or str, optional): Per-point colors. If 1D array of
            length N with numeric values, maps values to colormap. If 2D
            (N, 4) RGBA array, uses as direct colors. Defaults to None.
        s (float or array-like, optional): Alias for ``size``; overrides it if given.
            Like ``size``, may be a scalar or a length-N array for per-point sizes.
            Note: GLPlot's ``s`` is a pixel size applied element-wise, not matplotlib's
            pt^2 area -- consistent with its scalar ``size``.
        cmap (str, optional): Colormap name ('viridis', 'plasma', 'cool',
            etc.). Used when c is numeric. Defaults to 'viridis'.
        norm (Normalize or str, optional): How ``c`` maps onto the colormap --
            a ``matplotlib.colors.Normalize`` instance, or a scale name
            ('linear', 'log', 'symlog', 'logit'). Replaces the linear
            vmin..vmax ramp, so passing it together with vmin/vmax is an error,
            as in matplotlib. Defaults to linear.
        vmin (float, optional): Minimum value for colormap normalization.
            If None, uses data minimum. Defaults to None.
        vmax (float, optional): Maximum value for colormap normalization.
            If None, uses data maximum. Defaults to None.
        alpha (float, optional): Transparency (0.0-1.0). Defaults to 1.0.
        label (str, optional): Legend label. Defaults to None.
        marker (str, optional): Marker style (stored in metadata but visual
            rendering uses circles). Defaults to None.
        edgecolors (str or tuple, optional): Outline colour for the points.
            Turns the outline on; 'none' and 'face' leave it off. Defaults to
            no outline.
        linewidths (float, optional): Outline width in pixels. Only meaningful
            alongside ``edgecolors``, which is what enables the outline.
        plotnonfinite (bool, optional): Accepted for matplotlib parity. GLPlot
            maps non-finite ``c`` values to the colormap's low end rather than
            dropping them, so this is ignored.
        colorizer: Accepted for parity; ignored. Pass ``cmap``/``norm``/``vmin``/``vmax``
            directly instead.
        data (indexable, optional): If given, ``x``, ``y`` and ``c`` may be keys
            into it (a DataFrame, dict, structured array, ...).

    Returns:
        Layer: The scatter layer added to plot.

    Raises:
        ValueError: If x and y have different lengths.

    Examples:
        Basic scatter plot:

        >>> x = [1, 2, 3, 4, 5]
        >>> y = [2, 4, 5, 4, 6]
        >>> gplt.scatter(x, y)
        >>> gplt.show()

        With colormap based on values:

        >>> values = [10, 20, 15, 30, 25]
        >>> gplt.scatter(x, y, c=values, cmap='plasma', s=20)

        With custom single color:

        >>> gplt.scatter(x, y, color='red', size=15, alpha=0.7)
    """
    plot_obj = _get_or_create_plot()
    x, y, c = _resolve_data_args("scatter", data, x, y, c)
    _warn_unsupported(
        "scatter",
        {"plotnonfinite": plotnonfinite or None, "colorizer": colorizer},
        {
            "plotnonfinite": "has no effect: non-finite c values are always mapped to the "
            "colormap's low end rather than dropped",
            "colorizer": "has no effect: pass cmap=/norm=/vmin=/vmax= directly, there is "
            "no shared Colorizer object across artists",
        },
    )
    x_arr = _as_float_array(_coerce_axis_values(x, "x", "scatter"), ndim=1, name="x")
    y_arr = _as_float_array(_coerce_axis_values(y, "y", "scatter"), ndim=1, name="y")

    if len(x_arr) != len(y_arr):
        raise ValueError("x and y must have the same length")

    resolved_color = c if c is not None else color
    if resolved_color is not None:
        maybe_values = np.asarray(resolved_color)
    else:
        maybe_values = None

    cvalues: Optional[np.ndarray] = None
    if (
        maybe_values is not None
        and maybe_values.ndim == 1
        and len(maybe_values) == len(x_arr)
        and not isinstance(resolved_color, str)
        and np.issubdtype(maybe_values.dtype, np.number)
    ):
        from matplotlib import colormaps

        normed = _normalize_cvalues(maybe_values, norm, vmin, vmax)
        cols = np.asarray(
            colormaps.get_cmap(_resolve_cmap(cmap, "viridis"))(normed), dtype=np.float32
        )
        cvalues = _retained_cvalues(maybe_values)
    else:
        # An explicit None test, not `resolved_color or default`: a per-point RGBA array
        # passed through `color=` is a legal, common spelling (an (N, 4) array), and
        # `ndarray or default` raises "truth value ambiguous" on it rather than falling
        # through to the default. The default only applies when no colour was given at all.
        base = (0.0, 0.0, 0.0, 1.0) if resolved_color is None else resolved_color
        cols = _normalize_rgba(base, n=len(x_arr))
    if alpha is not None:
        cols[:, 3] *= float(alpha)

    # Marker size as a data-driven dimension: ``s`` (or ``size``) may be a scalar or an
    # array of length N. An array makes each point's size follow a variable; the scalar
    # stays the uniform fallback. GLPlot's ``s`` is a per-point pixel size (as its scalar
    # ``size`` always has been), applied element-wise -- not matplotlib's pt^2 area.
    raw_size = s if s is not None else size
    size_arr: Optional[np.ndarray] = None
    size_np = np.asarray(raw_size)
    if size_np.ndim >= 1 and size_np.size > 1:
        if size_np.size != len(x_arr):
            raise ValueError(
                f"size array has length {size_np.size}, expected {len(x_arr)} (one per point)"
            )
        size_arr = size_np.astype(np.float32).ravel()
        # A representative scalar for the GUI's size slider; cancels out in rendering.
        scalar_size = float(np.mean(size_arr)) if size_arr.size else 10.0
    else:
        # `.reshape(-1)[0]` rather than a bare `float(size_np)`: numpy 2.x raises on
        # converting a 1-element *array* (e.g. `s=np.array([7.0])`, this branch's whole
        # reason to exist) directly to a scalar, only a true 0-d array works with `float()`.
        scalar_size = float(size_np.reshape(-1)[0]) if size_np.ndim else float(size_np)

    plot_obj.add_scatter(x_arr, y_arr, cols, scalar_size, label=label, sizes=size_arr)
    _apply_scatter_edges(plot_obj.scene.layers[-1], edgecolors, linewidths)
    plot_obj.scene.layers[-1].metadata.update(
        {
            "marker": marker,
            "artist": "scatter",
            "cmap": cmap,
            "vmin": vmin,
            "vmax": vmax,
            # The per-point sizes, kept so the GUI can show size as a re-mappable dimension
            # (mirrors ``cvalues`` for colour). None when the layer is uniform-sized.
            "svalues": size_arr,
            # The scalars the colours came from. Nothing on the GPU needs them — the
            # colours are already baked into a per-vertex VBO — but without them the
            # mapping is one-way and cmap/vmin/vmax become undoable decoration: the GUI
            # can show a colormap picker and have no numbers left to re-map. See
            # layerops.set_layer_colormap. None when the scalars were not retained
            # (see _retained_cvalues) or when the colours were literal RGBA.
            "cvalues": cvalues,
            # The caller named a colour, rather than taking the default black. Density mode
            # reads this to decide whether to paint in the layer's colours instead of in the
            # global colormap -- see ``RendererManager.density_tint_active`` for why the
            # distinction has to be "was one asked for" and not "is there one".
            "explicit_color": resolved_color is not None,
        }
    )
    _set_dirty(plot_obj)
    layer = plot_obj.scene.layers[-1]
    # Only a *colormapped* scatter is a mappable: one plotted in a flat colour has no
    # scalars, so pointing clim() at it would offer limits on nothing.
    return _set_current_mappable(layer) if cvalues is not None else layer


#: What ``Axes3D.scatter`` binds its positional arguments to, in order, after ``xs, ys``.
#: ``zs`` third is the difference that matters: GLPlot's 2D ``scatter`` names its third
#: parameter ``color``, so ``ax.scatter(x, y, z)`` on 3D axes used to bind the z data to
#: the colour argument and draw a flat, viridis-ramped 2D scatter with no warning.
_SCATTER_POSITIONAL_3D = ("zs", "zdir", "s", "c", "depthshade")


def _scatter_3d_call(args: tuple, kwargs: dict) -> BaseLayer:
    """Handle ``scatter`` on a 3D axes the way ``Axes3D.scatter`` does.

    Unlike :func:`_plot_3d_call` this never declines: ``Axes3D.scatter`` has no ambiguous
    form to fall back from -- the third positional is always ``zs``, never a format string
    -- so a 3D axes always takes this path and a bad call is a ``TypeError`` naming the 3D
    arity rather than a silent 2D drawing.
    """
    if len(args) < 2:
        raise TypeError(f"scatter() takes at least 2 positional arguments but {len(args)} given")
    if len(args) - 2 > len(_SCATTER_POSITIONAL_3D):
        raise TypeError(
            f"scatter() on 3D axes takes from 2 to {2 + len(_SCATTER_POSITIONAL_3D)} "
            f"positional arguments but {len(args)} were given"
        )
    bound = dict(kwargs)
    for name, value in zip(_SCATTER_POSITIONAL_3D, args[2:]):
        if name in kwargs:
            raise TypeError(f"scatter() got multiple values for argument {name!r}")
        bound[name] = value

    x, y = args[0], args[1]
    zs = bound.pop("zs", 0)
    zdir = bound.pop("zdir", "z")
    data = bound.pop("data", None)
    c = bound.pop("c", None)
    x, y, c = _resolve_data_args("scatter", data, x, y, c)

    _warn_unsupported(
        "scatter",
        {
            "depthshade": (False if bound.pop("depthshade", True) is False else None),
            "norm": bound.pop("norm", None),
            "marker": bound.pop("marker", None),
            "edgecolors": bound.pop("edgecolors", None),
            "linewidths": bound.pop("linewidths", None),
            "plotnonfinite": bound.pop("plotnonfinite", None) or None,
        },
        {
            "depthshade": "has no effect: 3D points are drawn at full opacity and depth "
            "is conveyed by the perspective camera",
            "norm": "has no effect on 3D axes; pass vmin/vmax instead",
            "marker": "has no effect: 3D points are round impostors",
            "edgecolors": "has no effect: 3D points have no outline pass",
            "linewidths": "has no effect: 3D points have no outline pass",
            "plotnonfinite": "has no effect: non-finite c values map to the colormap's low end",
        },
    )

    xs = _as_float_array(x, ndim=1, name="x")
    ys = _as_float_array(y, ndim=1, name="y")
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    zs_arr = _broadcast_zs(zs, len(xs))
    xs, ys, zs_arr = _juggle_axes(xs, ys, zs_arr, zdir)
    # ``size`` is GLPlot's own spelling of ``s``; matplotlib only has ``s``.
    if "size" in bound and "s" not in bound:
        bound["s"] = bound.pop("size")
    return scatter3d(xs, ys, zs_arr, c=c, **bound)


def scatter(*args: Any, **kwargs: Any) -> BaseLayer:
    """Create a scatter plot; 2D by default, ``Axes3D.scatter`` on 3D axes.

    On 2D axes this is :func:`_scatter_2d` and its signature is the one
    ``inspect.signature`` reports (via ``__wrapped__``): ``scatter(x, y, color=None,
    size=10.0, c=None, s=None, cmap=None, ...)``. Nothing about the 2D behaviour changes.

    On axes created with ``projection="3d"`` it is matplotlib's ``Axes3D.scatter`` instead:
    the third positional argument is ``zs``, the fourth is ``zdir``, and the result is a
    3D point cloud. ``ax.scatter(x, y, z)`` used to bind ``z`` to GLPlot's third parameter
    -- ``color`` -- and draw a *2D* scatter coloured by z, which looks like a plausible
    plot and is not the one that was asked for.

    The dispatch is a wrapper rather than one merged signature on purpose: the two
    parameter lists disagree about what position three means, and Python's own binding is
    the only thing that gets "multiple values for argument" right for both. Delegating to
    two real signatures keeps that, instead of re-implementing argument binding here.

    Examples:
        >>> gplt.scatter(x, y, c=values, cmap="plasma", s=20)      # 2D
        >>> ax = gplt.subplot(projection="3d")
        >>> ax.scatter(x, y, z, c=z, cmap="plasma")                # 3D
        >>> ax.scatter(x, y, zs=0, zdir="y")                       # onto a wall
    """
    if _get_or_create_plot().is_3d_scene():
        return _scatter_3d_call(args, kwargs)
    return _scatter_2d(*args, **kwargs)


#: ``inspect.signature`` follows ``__wrapped__``, so ``scatter``'s reported signature is
#: the 2D one a caller almost always wants to read rather than ``(*args, **kwargs)``. The
#: 3D form is documented in the docstring, which ``help()`` shows alongside it.
scatter.__wrapped__ = _scatter_2d


def scatter3d(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *args: Any,
    elev: float = 30.0,
    azim: float = -60.0,
    scale_z: float = 1.0,
    c: Optional[Union[ColorLike, ArrayLike]] = None,
    cmap: Optional[str] = None,
    **kwargs: Any,
) -> BaseLayer:
    """Scatter plot in 3D space.

    Renders points at 3D coordinates with optional per-point colormapping.
    Supports both uniform coloring and value-based color mapping. Native 3D
    rendering with optimized GPU pipeline for large point clouds.

    Args:
        x (array-like): X-coordinates. Shape (N,).
        y (array-like): Y-coordinates. Shape (N,).
        z (array-like): Z-coordinates. Shape (N,).
        *args: Format string and additional arguments (for compatibility).
        elev (float, optional): Camera elevation angle in degrees. Defaults to 30.0.
        azim (float, optional): Camera azimuth angle in degrees. Defaults to -60.0.
        scale_z (float, optional): Multiplies the z data before it is drawn.
            Defaults to 1.0 -- z is plotted as given, which is what matplotlib does
            and what every z read-back (``set_zlim()``, the z ticks, the box) then
            reports. It used to default to 0.7 to make the box look cube-ish, which
            rewrote the caller's data: ``scatter3d([0, 1], [0, 1], [0, 10])`` stored
            ``[0, 7]`` and the z axis was in fabricated units. Visual squashing is
            :func:`set_box_aspect`'s job -- it changes the picture, not the data.
        c (array-like or str, optional): Per-point colors. If 1D numeric array,
            maps to colormap. If 2D (N, 4), direct RGBA colors. Defaults to None.
        cmap (str, optional): Colormap name when c is numeric. Defaults to 'viridis'.
        **kwargs: Additional keyword arguments including:
            color (str): Single color for all points
            s, size (float): Point size
            label (str): Legend label
            alpha (float): Transparency
            vmin, vmax (float): Colormap normalization range

    Returns:
        Layer: The 3D scatter layer added to plot.

    Examples:
        Simple 3D scatter:

        >>> x = np.random.randn(1000)
        >>> y = np.random.randn(1000)
        >>> z = np.random.randn(1000)
        >>> gplt.scatter3d(x, y, z, s=5)
        >>> gplt.show()

        With colormap based on z-values:

        >>> gplt.scatter3d(x, y, z, c=z, cmap='plasma', s=8)

        Custom camera angle:

        >>> gplt.scatter3d(x, y, z, c=z, elev=60, azim=45)
    """
    x_arr = _as_float_array(x, ndim=1, name="x")
    y_arr = _as_float_array(y, ndim=1, name="y")
    z_arr = _as_float_array(z, ndim=1, name="z")
    if not (len(x_arr) == len(y_arr) == len(z_arr)):
        raise ValueError("x, y, and z must have the same length")
    verts = np.column_stack([x_arr, y_arr, z_arr * float(scale_z)]).astype(np.float32)
    # No ``c`` means "colour by z", which is GLPlot's default and not matplotlib's -- but
    # an explicit ``color=`` has to win over it. ``_add_3d_layer`` takes the per-vertex
    # ``colors`` array in preference to a flat ``color``, so leaving the z ramp in place
    # made ``scatter3d(x, y, z, color="red")`` draw a viridis ramp and say nothing.
    values = c if c is not None else (None if kwargs.get("color") is not None else z_arr)
    if (
        values is not None
        and not isinstance(values, str)
        and np.asarray(values).ndim == 1
        and len(np.asarray(values)) == len(verts)
    ):
        cols = _colormap_values(values, cmap=cmap, vmin=kwargs.get("vmin"), vmax=kwargs.get("vmax"))
    else:
        cols = None
    # Marker size as a per-point dimension, same rule as 2D scatter: scalar or length-N array.
    raw_size = kwargs.get("s", kwargs.get("size", 3.0))
    size_np = np.asarray(raw_size)
    sizes3d: Optional[np.ndarray] = None
    if size_np.ndim >= 1 and size_np.size > 1:
        if size_np.size != len(verts):
            raise ValueError(
                f"size array has length {size_np.size}, expected {len(verts)} (one per point)"
            )
        sizes3d = size_np.astype(np.float32).ravel()
        point_size = float(np.mean(sizes3d)) if sizes3d.size else 3.0
    else:
        point_size = float(size_np)
    layer = _add_3d_layer(
        verts,
        colors=cols,
        primitive="points",
        layer_type="scatter3d",
        label=kwargs.get("label"),
        elev=elev,
        azim=azim,
        point_size=point_size,
        color=values if isinstance(values, str) else kwargs.get("color"),
        alpha=kwargs.get("alpha"),
        metadata={
            "artist": "scatter3d",
            "zdata": z_arr,
            "scale_z": scale_z,
            "cmap": cmap,
            "svalues": sizes3d,
        },
        sizes=sizes3d,
        blend=kwargs.get("blend"),
        depth_write=kwargs.get("depth_write"),
        auto_alpha=kwargs.get("auto_alpha"),
    )
    return layer


def _fill_between_geometry(
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    where: Optional[np.ndarray],
    interpolate: bool,
    step: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """The band's vertices and triangle indices for ``fill_between``.

    Returns interleaved ``top[i], bot[i]`` vertices plus the indices that fill only the
    segments ``where`` selects. Indices rather than one triangle strip because a strip is
    a single connected ribbon and cannot have holes: with ``where``, holes are the point.
    One indexed layer keeps the band a single artist, as matplotlib's PolyCollection is.

    ``interpolate`` moves each run's boundary to where the two curves actually cross,
    instead of leaving it on the last sample before the crossing. Without it a
    ``where=y1 > y2`` band visibly stops short of the intersection it is describing --
    the gap is up to one sample wide, which at coarse sampling is most of the feature.
    """
    if step is not None:
        # matplotlib's step-mode fill: the band is a staircase, so each sample is doubled
        # into the neighbouring interval before any of the filling below happens.
        x, y1, y2, where = _fill_step_expand(x, y1, y2, where, step)

    n = len(x)
    if where is None:
        where = np.ones(n, dtype=bool)

    # A segment is filled when both of its ends are selected. Runs of True therefore
    # become runs of filled segments, and a lone True fills nothing -- matplotlib's rule.
    seg = where[:-1] & where[1:]

    # Built per run, not globally: an interpolated boundary adds a vertex that only that
    # run's triangles may reference, so a run has to own a contiguous slice of the buffer.
    verts: list = []
    indices: list = []
    for first, last in _true_runs(seg):
        nodes = range(first, last + 2)  # the run's segments span points first .. last+1
        xs = [float(x[i]) for i in nodes]
        top = [float(y1[i]) for i in nodes]
        bot = [float(y2[i]) for i in nodes]

        if interpolate:
            # The excluded segment on each side may hold the crossing this run is really
            # bounded by. Where it does, the band closes there instead of at the last
            # sample -- and it closes to a *point*, since that is where the curves meet.
            head = _curve_crossing(x, y1, y2, first - 1) if first > 0 else None
            if head is not None:
                xs.insert(0, head[0])
                top.insert(0, head[1])
                bot.insert(0, head[1])
            tail = _curve_crossing(x, y1, y2, last + 1) if last + 1 < n - 1 else None
            if tail is not None:
                xs.append(tail[0])
                top.append(tail[1])
                bot.append(tail[1])

        base = len(verts)
        for xi, ti, bi in zip(xs, top, bot):
            verts.append((xi, ti))
            verts.append((xi, bi))
        for i in range(len(xs) - 1):
            t0, b0, t1, b1 = base + 2 * i, base + 2 * i + 1, base + 2 * i + 2, base + 2 * i + 3
            indices += [t0, b0, b1, t0, b1, t1]

    if not verts:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.uint32)
    return (
        np.ascontiguousarray(verts, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
    )


def _true_runs(mask: np.ndarray) -> list:
    """``[(first, last), ...]`` for each maximal run of True in ``mask``, ends inclusive."""
    if mask.size == 0 or not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2], edges[1::2] - 1))


def _curve_crossing(
    x: np.ndarray, y1: np.ndarray, y2: np.ndarray, j: int
) -> Optional[Tuple[float, float]]:
    """Where ``y1`` and ``y2`` cross inside segment ``j``, or None if they do not.

    Linear on both curves, which is what the band already assumes between samples -- the
    crossing of two straight segments, not an approximation of one.
    """
    if j < 0 or j >= len(x) - 1:
        return None
    d0 = float(y1[j] - y2[j])
    d1 = float(y1[j + 1] - y2[j + 1])
    if d0 == d1 or (d0 > 0.0) == (d1 > 0.0):
        return None
    t = d0 / (d0 - d1)
    return (
        float(x[j]) + t * float(x[j + 1] - x[j]),
        float(y1[j]) + t * float(y1[j + 1] - y1[j]),
    )


def _fill_step_expand(
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    where: Optional[np.ndarray],
    step: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Double each sample into a staircase, mirroring ``pyplot.step``'s ``where``."""
    if step not in ("pre", "post", "mid"):
        raise ValueError(f"step must be 'pre', 'post' or 'mid', got {step!r}")
    if len(x) < 2:
        return x, y1, y2, where

    if step == "post":
        sx = np.repeat(x, 2)[1:]
        keep = np.repeat(np.arange(len(x)), 2)[:-1]
    elif step == "pre":
        sx = np.repeat(x, 2)[:-1]
        keep = np.repeat(np.arange(len(x)), 2)[1:]
    else:
        mid = 0.5 * (x[:-1] + x[1:])
        sx = np.concatenate([x[:1], np.repeat(mid, 2), x[-1:]])
        keep = np.repeat(np.arange(len(x)), 2)
    return sx, y1[keep], y2[keep], (None if where is None else where[keep])


def fill(*args: Any, **kwargs: Any) -> list:
    """Draw one or more filled polygons.

    Takes the same ``x, y, [color], x, y, [color], ...`` grouping as :func:`plot`,
    but fills each polygon instead of outlining it. Each ``(x, y)`` is closed back
    to its first point.

    Args:
        *args: ``x, y`` pairs, each optionally followed by a colour/format string,
            exactly as :func:`plot` accepts them.
        alpha (float, optional): Transparency, applied to every polygon.
        label (str, optional): Legend label for the first polygon.

    Returns:
        list: One patch layer per polygon.

    Examples:
        >>> gplt.fill([0, 1, 1, 0], [0, 0, 1, 1], 'b')          # a filled square
        >>> gplt.fill(x1, y1, 'r', x2, y2, 'g')                 # two polygons
    """
    data = kwargs.pop("data", None)
    if data is not None:
        args = _resolve_plot_data_args("fill", data, args)
    alpha = kwargs.pop("alpha", None)
    label = kwargs.pop("label", None)
    # Reuse plot()'s argument splitter rather than re-deriving the x,y,[fmt] grammar --
    # the one place that grammar is allowed to live, so fill() and plot() cannot drift.
    groups = _parse_plot_groups(args)
    if not groups:
        raise ValueError("fill(): needs at least one x, y pair")

    plot_obj = _get_or_create_plot()
    layers = []
    for i, (xs, ys, fmt) in enumerate(groups):
        x_arr = _as_float_array(xs, ndim=1, name="x")
        y_arr = _as_float_array(ys, ndim=1, name="y")
        if len(x_arr) != len(y_arr):
            raise ValueError("each x and y in fill() must have the same length")
        parsed = _parse_plot_format(fmt)
        color = (
            parsed.get("color") if parsed.get("color") else _color_cycle()[i % len(_color_cycle())]
        )
        rgba = list(_normalize_rgba(color))
        if alpha is not None:
            rgba[3] *= float(alpha)

        # A fan from the first vertex closes the polygon without asking whether it is
        # convex: a fan is only correct for a convex ring, but a polygon fill that is
        # allowed to be concave needs a real triangulation, which matplotlib does with a
        # tessellator. Fan here, and say so, rather than draw a concave shape wrong in
        # silence -- the common case (a closed convex outline) is exact.
        verts = np.column_stack([x_arr, y_arr]).astype(np.float32)
        fan = np.arange(1, len(verts) - 1)
        indices = np.column_stack([np.zeros_like(fan), fan, fan + 1]).ravel().astype(np.uint32)
        add_patch(
            verts,
            indices=indices,
            mode="triangles",
            face_color=tuple(rgba),
            edge_color=tuple(rgba),
            label=label if i == 0 else None,
        )
        layer = plot_obj.scene.layers[-1]
        layer.metadata["artist"] = "fill"
        layers.append(layer)

    _set_dirty(plot_obj)
    return layers


def fill_betweenx(
    y: Sequence[float],
    x1: Sequence[float],
    x2: Union[float, Sequence[float]] = 0,
    where: Optional[Sequence[bool]] = None,
    step: Optional[str] = None,
    interpolate: bool = False,
    *,
    color: ColorLike = (0.2, 0.4, 0.8, 0.35),
    alpha: Optional[float] = None,
    label: Optional[str] = None,
    data: Optional[Any] = None,
):
    """Fill the area between two curves of x, as functions of y.

    :func:`fill_between` with the axes swapped -- a horizontal confidence band.

    Args:
        y (array-like): The shared y-coordinates.
        x1 (array-like): One boundary, as x for each y.
        x2 (float or array-like, optional): The other boundary. Defaults to 0.
        where (array-like of bool, optional): Fill only where True.
        step (str, optional): ``'pre'``, ``'post'`` or ``'mid'`` -- fill a staircase
            rather than a straight interpolation between samples.
        interpolate (bool, optional): When ``where`` is given, carry the fill up to the
            exact crossing point rather than stopping at the last sample inside it.
            Defaults to False.
        color (str or tuple, optional): Fill colour.
        alpha (float, optional): Transparency.
        label (str, optional): Legend label.
        data (indexable, optional): If given, the arrays may be keys into it.

    Returns:
        Layer: One patch layer, as :func:`fill_between` returns.

    Examples:
        >>> gplt.fill_betweenx(y, x - 0.1, x + 0.1, alpha=0.3)
    """
    y, x1, x2, where = _resolve_data_args("fill_betweenx", data, y, x1, x2, where)
    y_arr = _as_float_array(y, ndim=1, name="y")
    x1_arr = _as_float_array(x1, ndim=1, name="x1")
    x2_arr = (
        np.full_like(y_arr, float(x2))
        if np.isscalar(x2)
        else _as_float_array(x2, ndim=1, name="x2")
    )
    if not (len(y_arr) == len(x1_arr) == len(x2_arr)):
        raise ValueError("y, x1, and x2 must have the same length")

    where_arr = None
    if where is not None:
        where_arr = np.asarray(where, dtype=bool)
        if where_arr.ndim != 1 or len(where_arr) != len(y_arr):
            raise ValueError("where must be a 1-D boolean array the same length as y")

    # The band-with-holes geometry is exactly `fill_between`'s, computed in (y, x1, x2) and
    # then transposed -- so the two share the run-splitting logic rather than reimplement
    # it with x and y exchanged and a second chance to get the winding wrong.
    verts, indices = _fill_between_geometry(y_arr, x1_arr, x2_arr, where_arr, interpolate, step)
    verts = verts[:, ::-1].copy()  # (y, x) -> (x, y)
    rgba = list(_normalize_rgba(color))
    if alpha is not None:
        rgba[3] *= float(alpha)
    plot_obj = _get_or_create_plot()
    add_patch(
        verts,
        indices=indices,
        mode="triangles",
        face_color=tuple(rgba),
        edge_color=tuple(rgba),
        label=label,
    )
    layer = plot_obj.scene.layers[-1]
    layer.metadata["artist"] = "fill_betweenx"
    return layer


def stackplot(
    x: Sequence[float],
    *ys: Sequence[float],
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[ColorLike]] = None,
    baseline: str = "zero",
    alpha: Optional[float] = None,
    hatch: Optional[Union[str, Sequence[str]]] = None,
    data: Optional[Any] = None,
) -> list:
    """Draw a stacked area chart -- each series filled on top of the last.

    Args:
        x (array-like): The shared x-coordinates.
        *ys (array-like): One or more series to stack, each the length of ``x``.
            A single 2-D array of shape ``(n_series, len(x))`` also works.
        labels (sequence of str, optional): One label per series.
        colors (sequence, optional): One colour per series. Defaults to the cycle.
        baseline (str, optional): Only 'zero' is drawn; 'sym', 'wiggle' and
            'weighted_wiggle' are accepted and fall back to it.
        alpha (float, optional): Transparency.
        hatch (str or sequence, optional): Accepted for matplotlib parity; ignored. GLPlot
            fills are flat colour, with no hatch renderer.
        data (indexable, optional): If given, the series may be keys into it.

    Returns:
        list: One filled-area layer per series, bottom to top.

    Examples:
        >>> gplt.stackplot(x, series_a, series_b, series_c, labels=['a', 'b', 'c'])
    """
    if data is not None:
        x, *ys = _resolve_data_args("stackplot", data, x, *ys)
    _warn_unsupported(
        "stackplot",
        {"baseline": baseline if baseline != "zero" else None, "hatch": hatch},
        {
            "baseline": "is not supported; the stack is drawn from a zero baseline",
            "hatch": "has no effect: GLPlot has no hatch renderer, so fills are drawn flat",
        },
    )
    x_arr = _as_float_array(x, ndim=1, name="x")
    # A single 2-D array is matplotlib's other calling convention; unpack it to rows.
    if len(ys) == 1 and np.ndim(ys[0]) == 2:
        stack = _as_float_array(ys[0], ndim=2, name="ys")
    else:
        stack = np.array([_as_float_array(s, ndim=1, name="ys") for s in ys])
    if stack.size == 0:
        raise ValueError("stackplot(): needs at least one series")
    if stack.shape[1] != len(x_arr):
        raise ValueError("each series must have the same length as x")
    if labels is not None and len(labels) != len(stack):
        raise ValueError("labels must have one entry per series")
    if colors is not None and len(colors) != len(stack):
        raise ValueError("colors must have one entry per series")

    layers = []
    lower = np.zeros_like(x_arr)
    for i, series in enumerate(stack):
        upper = lower + series
        color = colors[i] if colors is not None else _color_cycle()[i % len(_color_cycle())]
        layer = fill_between(
            x_arr,
            upper,
            lower,
            color=color,
            alpha=alpha,
            label=labels[i] if labels is not None else None,
        )
        layer.metadata["artist"] = "stackplot"
        layers.append(layer)
        lower = upper  # the next series stacks on this one's top

    _set_dirty(_get_or_create_plot())
    return layers


def fill_between(
    x: Sequence[float],
    y1: Sequence[float],
    y2: Union[float, Sequence[float]] = 0,
    where: Optional[Sequence[bool]] = None,
    interpolate: bool = False,
    step: Optional[str] = None,
    *,
    color: ColorLike = (0.2, 0.4, 0.8, 0.35),
    alpha: Optional[float] = None,
    label: Optional[str] = None,
    data: Optional[Any] = None,
):
    """Fill the area between two curves or a curve and baseline.

    Renders a filled region (band) between two y-value curves as a function
    of x. Useful for uncertainty bands, confidence intervals, or area-fill
    plots. Can be stacked by calling multiple times with different baselines.

    Args:
        x (array-like): X-coordinates. Shape (N,).
        y1 (array-like): Upper boundary y-values. Shape (N,).
        y2 (float or array-like, optional): Lower boundary y-values.
            If scalar, applies to all x. If array, per-x values. Defaults to 0.
        where (array-like of bool, optional): Fill only where True, leaving the
            rest of the band empty. Same length as ``x``. A segment is filled
            when both of its ends are selected, so a lone True fills nothing --
            matplotlib's rule.
        interpolate (bool, optional): Only meaningful with ``where``. Extends
            each filled run to the point where ``y1`` and ``y2`` actually cross,
            instead of stopping at the last sample before it.
        step (str, optional): 'pre', 'post' or 'mid' -- fill a staircase rather
            than straight segments, matching :func:`step`. Defaults to None.
        color (str or tuple, optional): Fill color. Defaults to light blue
            with transparency (0.2, 0.4, 0.8, 0.35).
        alpha (float, optional): Transparency (0.0-1.0). Defaults to 1.0.
        label (str, optional): Legend label. Defaults to None.
        data (indexable, optional): If given, ``x``, ``y1``, ``y2`` and ``where``
            may be keys into it (a DataFrame, dict, structured array, ...).

    Returns:
        Layer: The filled region patch layer added to plot. One layer even when
        ``where`` splits the band into several runs, as matplotlib returns one
        PolyCollection.

    Examples:
        Simple area fill:

        >>> x = np.linspace(0, 10, 100)
        >>> y = np.sin(x)
        >>> gplt.fill_between(x, y, 0)
        >>> gplt.show()

        Confidence interval:

        >>> y_mean = np.sin(x)
        >>> y_lower = y_mean - 0.1
        >>> y_upper = y_mean + 0.1
        >>> gplt.fill_between(x, y_upper, y_lower, alpha=0.3)

        Stacked areas:

        >>> gplt.fill_between(x, y1, 0, color='blue', label='A')
        >>> gplt.fill_between(x, y1 + y2, y1, color='red', label='B')
    """
    x, y1, y2, where = _resolve_data_args("fill_between", data, x, y1, y2, where)
    x_arr = _as_float_array(x, ndim=1, name="x")
    y1_arr = _as_float_array(y1, ndim=1, name="y1")
    y2_arr = (
        np.full_like(y1_arr, float(y2))
        if np.isscalar(y2)
        else _as_float_array(y2, ndim=1, name="y2")
    )
    if not (len(x_arr) == len(y1_arr) == len(y2_arr)):
        raise ValueError("x, y1, and y2 must have the same length")

    where_arr = None
    if where is not None:
        where_arr = np.asarray(where, dtype=bool)
        if where_arr.ndim != 1 or len(where_arr) != len(x_arr):
            raise ValueError("where must be a 1-D boolean array the same length as x")
    if interpolate and where is None:
        # matplotlib's own rule: interpolate only means anything at a run boundary, and
        # with no `where` there are no runs. Saying so beats leaving the caller to wonder
        # why the band did not change.
        _warn_unsupported(
            "fill_between",
            {"interpolate": True},
            {
                "interpolate": "has no effect without where=: it moves a run's boundary onto "
                "the curves' crossing, and with no where= the band has no boundaries"
            },
        )

    verts, indices = _fill_between_geometry(
        x_arr, y1_arr, y2_arr, where_arr, bool(interpolate), step
    )
    rgba = list(_normalize_rgba(color, n=None))
    if alpha is not None:
        rgba[3] *= float(alpha)
    plot_obj = _get_or_create_plot()
    # The layer, not `add_patch`'s plot object: the docstring promises a Layer, `stackplot`
    # stacks on it, and `fill_betweenx` mirrors it -- all of which need the artist, not the
    # figure. `add_patch` predates that contract and still returns the figure.
    add_patch(
        verts,
        indices=indices,
        mode="triangles",
        face_color=tuple(rgba),
        edge_color=tuple(rgba),
        label=label,
    )
    return plot_obj.scene.layers[-1]


def _rect_patch(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    color: ColorLike,
    alpha: Optional[float],
    label: Optional[str],
) -> Any:
    """One axis-aligned rectangle as a two-triangle patch, returned as its layer.

    The shared body of every rectangle the span/bar family draws. A quad wound the same
    way each time, so a caller never has to think about triangle order, and one place to
    fix if the winding ever needs to change.
    """
    verts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    rgba = list(_normalize_rgba(color))
    if alpha is not None:
        rgba[3] *= float(alpha)
    plot_obj = _get_or_create_plot()
    add_patch(
        verts,
        indices=indices,
        mode="triangles",
        face_color=tuple(rgba),
        edge_color=tuple(rgba),
        label=label,
    )
    return plot_obj.scene.layers[-1]


def axhspan(
    ymin: float,
    ymax: float,
    xmin: float = 0.0,
    xmax: float = 1.0,
    *,
    color: ColorLike = (0.2, 0.4, 0.8, 0.35),
    alpha: Optional[float] = None,
    label: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Draw a horizontal band spanning the whole width of the axes.

    Args:
        ymin, ymax (float): The band's vertical extent, in data coordinates.
        xmin, xmax (float, optional): The horizontal extent as a fraction of the
            axes width, 0 to 1. Defaults to the full width.
        color (str or tuple, optional): Fill colour. Defaults to translucent blue.
        alpha (float, optional): Transparency.
        label (str, optional): Legend label.

    Returns:
        Layer: The band's patch layer.

    Examples:
        >>> gplt.axhspan(0.4, 0.6, color='red', alpha=0.2)
    """
    left, right = _get_or_create_plot().get_xlim()
    x0 = left + (right - left) * float(xmin)
    x1 = left + (right - left) * float(xmax)
    layer = _rect_patch(x0, x1, float(ymin), float(ymax), color, alpha, label)
    layer.metadata.update({"guide": True, "artist": "axhspan"})
    _set_dirty(_get_or_create_plot())
    return layer


def axvspan(
    xmin: float,
    xmax: float,
    ymin: float = 0.0,
    ymax: float = 1.0,
    *,
    color: ColorLike = (0.2, 0.4, 0.8, 0.35),
    alpha: Optional[float] = None,
    label: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Draw a vertical band spanning the whole height of the axes.

    Args:
        xmin, xmax (float): The band's horizontal extent, in data coordinates.
        ymin, ymax (float, optional): The vertical extent as a fraction of the
            axes height, 0 to 1. Defaults to the full height.
        color (str or tuple, optional): Fill colour. Defaults to translucent blue.
        alpha (float, optional): Transparency.
        label (str, optional): Legend label.

    Returns:
        Layer: The band's patch layer.

    Examples:
        >>> gplt.axvspan(2.0, 3.0, color='green', alpha=0.2)
    """
    bottom, top = _get_or_create_plot().get_ylim()
    y0 = bottom + (top - bottom) * float(ymin)
    y1 = bottom + (top - bottom) * float(ymax)
    layer = _rect_patch(float(xmin), float(xmax), y0, y1, color, alpha, label)
    layer.metadata.update({"guide": True, "artist": "axvspan"})
    _set_dirty(_get_or_create_plot())
    return layer


def broken_barh(
    xranges: Sequence[Tuple[float, float]],
    yrange: Tuple[float, float],
    *,
    color: ColorLike = (0.2, 0.4, 0.8, 1.0),
    alpha: Optional[float] = None,
    label: Optional[str] = None,
    data: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Draw a row of rectangles at one height -- a Gantt-chart bar.

    Args:
        xranges (sequence of (xstart, xwidth)): One rectangle per pair.
        yrange (tuple): ``(ystart, yheight)`` shared by every rectangle.
        color (str or tuple, optional): Fill colour.
        alpha (float, optional): Transparency.
        label (str, optional): Legend label.
        data (indexable, optional): If given, ``xranges`` and ``yrange`` may be
            keys into it.

    Returns:
        Layer: A single patch layer holding every rectangle.

    Examples:
        >>> gplt.broken_barh([(1, 2), (5, 1)], (10, 0.8))
    """
    xranges, yrange = _resolve_data_args("broken_barh", data, xranges, yrange)
    y0, dy = float(yrange[0]), float(yrange[1])
    spans = [(float(xs), float(xw)) for xs, xw in xranges]
    if not spans:
        raise ValueError("broken_barh(): xranges is empty")

    # One patch for the whole row rather than one per bar: a Gantt chart is often hundreds
    # of segments, and a layer apiece would swamp the Scene panel and the draw loop.
    verts = np.empty((4 * len(spans), 2), dtype=np.float32)
    indices = np.empty(6 * len(spans), dtype=np.uint32)
    for i, (xs, xw) in enumerate(spans):
        base = 4 * i
        verts[base : base + 4] = [[xs, y0], [xs + xw, y0], [xs + xw, y0 + dy], [xs, y0 + dy]]
        indices[6 * i : 6 * i + 6] = [base, base + 1, base + 2, base, base + 2, base + 3]

    rgba = list(_normalize_rgba(color))
    if alpha is not None:
        rgba[3] *= float(alpha)
    plot_obj = _get_or_create_plot()
    add_patch(
        verts,
        indices=indices,
        mode="triangles",
        face_color=tuple(rgba),
        edge_color=tuple(rgba),
        label=label,
    )
    layer = plot_obj.scene.layers[-1]
    layer.metadata["artist"] = "broken_barh"
    _set_dirty(plot_obj)
    return layer


def barh(
    y: Sequence[float],
    width: Sequence[float],
    height: float = 0.8,
    left: Union[float, Sequence[float]] = 0,
    *,
    align: str = "center",
    color: ColorLike = (0.2, 0.4, 0.8, 1.0),
    alpha: Optional[float] = None,
    label: Optional[str] = None,
    xerr: Optional[ArrayLike] = None,
    yerr: Optional[ArrayLike] = None,
    ecolor: Optional[ColorLike] = None,
    capsize: Optional[float] = None,
    error_kw: Optional[dict] = None,
    data: Optional[Any] = None,
) -> list:
    """Draw a horizontal bar chart -- :func:`bar` with the axes swapped.

    Args:
        y (array-like): Bar centres on the y-axis.
        width (array-like): Bar lengths along the x-axis.
        height (float, optional): Bar thickness on the y-axis. Defaults to 0.8.
        left (float or array-like, optional): The x each bar starts at. Defaults
            to 0. The horizontal counterpart of ``bar``'s ``bottom``.
        align (str, optional): 'center' (default) or 'edge'.
        color (str or tuple, optional): Bar colour.
        alpha (float, optional): Transparency.
        label (str, optional): Legend label.
        xerr, yerr (float or array-like, optional): Error bar magnitudes, drawn at each
            bar's ``(left + width, y)`` end (its far tip) exactly as matplotlib does --
            passed straight to :func:`errorbar`, including its ``(2, N)``
            asymmetric-error and scalar-broadcast forms.
        ecolor (str or tuple, optional): Error bar colour. Defaults to black, as in
            matplotlib (not ``color``, which only styles the bars).
        capsize (float, optional): Error bar cap half-length. In GLPlot's own data
            units, unlike matplotlib's points -- see :func:`errorbar`.
        error_kw (dict, optional): Extra keywords forwarded to :func:`errorbar`
            (matplotlib's own ``error_kw`` spelling).
        data (indexable, optional): If given, ``y``, ``width`` and ``left`` may
            be keys into it.

    Returns:
        list: One patch layer per bar.

    Examples:
        >>> gplt.barh([0, 1, 2], [10, 24, 18])
        >>> gplt.barh([0, 1, 2], [10, 24, 18], xerr=[1, 2, 1.5])
    """
    if align not in ("center", "edge"):
        raise ValueError(f"unsupported align: {align!r}. Expected 'center' or 'edge'.")
    y, width, left = _resolve_data_args("barh", data, y, width, left)
    y_arr = _as_float_array(_coerce_axis_values(y, "y", "barh"), ndim=1, name="y")
    w_arr = _as_float_array(width, ndim=1, name="width")
    l_arr = (
        np.full_like(w_arr, float(left))
        if np.isscalar(left)
        else _as_float_array(left, ndim=1, name="left")
    )
    if not (len(y_arr) == len(w_arr) == len(l_arr)):
        raise ValueError("y, width, and left must have the same length")

    offsets = (
        (-float(height) / 2.0, float(height) / 2.0) if align == "center" else (0.0, float(height))
    )
    lo, hi = min(offsets), max(offsets)
    rgba = list(_normalize_rgba(color))
    if alpha is not None:
        rgba[3] *= float(alpha)
    patches = []
    for idx, (yc, w, x0) in enumerate(zip(y_arr, w_arr, l_arr)):
        layer = _rect_patch(
            x0, x0 + w, yc + lo, yc + hi, tuple(rgba), None, label if idx == 0 else None
        )
        layer.metadata["artist"] = "barh"
        patches.append(layer)
    if xerr is not None or yerr is not None:
        kw = dict(error_kw or {})
        kw.setdefault("s", 0)
        patches.extend(
            errorbar(
                l_arr + w_arr,
                y_arr,
                yerr=yerr,
                xerr=xerr,
                fmt="none",
                ecolor="k" if ecolor is None else ecolor,
                capsize=0.0 if capsize is None else capsize,
                **kw,
            )
        )
    _set_dirty(_get_or_create_plot())
    return patches


def bar(
    x: Sequence[float],
    height: Sequence[float],
    width: float = 0.8,
    bottom: Union[float, Sequence[float]] = 0,
    *,
    align: str = "center",
    color: ColorLike = (0.2, 0.4, 0.8, 1.0),
    alpha: Optional[float] = None,
    label: Optional[str] = None,
    yerr: Optional[ArrayLike] = None,
    xerr: Optional[ArrayLike] = None,
    ecolor: Optional[ColorLike] = None,
    capsize: Optional[float] = None,
    error_kw: Optional[dict] = None,
    data: Optional[Any] = None,
) -> list:
    """Create a bar chart.

    Draws rectangular bars at specified x positions with given heights.
    Supports uniform bar width, per-bar baseline offsets for stacked bars,
    and custom colors. Each bar is rendered as an individual patch for
    efficient rendering.

    Args:
        x (array-like): Bar x-positions. Shape (N,).
        height (array-like): Bar heights. Shape (N,).
        width (float, optional): Bar width in data coordinates. Defaults to 0.8.
        bottom (float or array-like, optional): Baseline y position(s) for bars.
            If scalar, applies to all bars (enables stacked effect when changed
            between calls). If array, per-bar baselines. Defaults to 0.
        align (str, optional): 'center' (default) centres each bar on its x;
            'edge' puts the bar's left edge there. A negative ``width`` with
            'edge' aligns the right edge instead, as in matplotlib.
        color (str or tuple, optional): Bar color. Named colors, hex, or RGBA.
            Defaults to (0.2, 0.4, 0.8, 1.0) (blue).
        alpha (float, optional): Transparency (0.0-1.0). Defaults to 1.0.
        label (str, optional): Legend label (only first bar labeled for
            efficiency). Defaults to None.
        yerr, xerr (float or array-like, optional): Error bar magnitudes, drawn at each
            bar's ``(x, bottom + height)`` tip exactly as matplotlib does -- passed
            straight to :func:`errorbar`, including its ``(2, N)`` asymmetric-error and
            scalar-broadcast forms.
        ecolor (str or tuple, optional): Error bar colour. Defaults to black, as in
            matplotlib (not ``color``, which only styles the bars).
        capsize (float, optional): Error bar cap half-length. In GLPlot's own data
            units, unlike matplotlib's points -- see :func:`errorbar`.
        error_kw (dict, optional): Extra keywords forwarded to :func:`errorbar`
            (matplotlib's own ``error_kw`` spelling).
        data (indexable, optional): If given, ``x``, ``height`` and ``bottom``
            may be keys into it (a DataFrame, dict, structured array, ...).

    Returns:
        list: Patch layers (one per bar) added to plot.

    Raises:
        ValueError: If x, height, and bottom do not have compatible shapes.

    Examples:
        Simple bar chart:

        >>> categories = [0, 1, 2, 3]
        >>> values = [10, 24, 36, 18]
        >>> gplt.bar(categories, values)
        >>> gplt.show()

        Stacked bars (multiple calls):

        >>> gplt.bar([0, 1, 2], [10, 20, 15], label='First')
        >>> gplt.bar([0, 1, 2], [5, 10, 8], bottom=[10, 20, 15], label='Second')

        Custom styling:

        >>> gplt.bar([0, 1, 2], [10, 20, 15], width=0.5, color='red', alpha=0.7)

        With error bars:

        >>> gplt.bar([0, 1, 2], [10, 20, 15], yerr=[1, 2, 1.5], capsize=0.1)
    """
    if align not in ("center", "edge"):
        raise ValueError(f"unsupported align: {align!r}. Expected 'center' or 'edge'.")
    x, height, bottom = _resolve_data_args("bar", data, x, height, bottom)
    x_arr = _as_float_array(_coerce_axis_values(x, "x", "bar"), ndim=1, name="x")
    h_arr = _as_float_array(height, ndim=1, name="height")
    b_arr = (
        np.full_like(h_arr, float(bottom))
        if np.isscalar(bottom)
        else _as_float_array(bottom, ndim=1, name="bottom")
    )
    if not (len(x_arr) == len(h_arr) == len(b_arr)):
        raise ValueError("x, height, and bottom must have the same length")
    # `align` decides what x *means*: the bar's centre, or its leading edge. A negative
    # width under 'edge' therefore grows the bar leftward -- matplotlib's way of spelling
    # right-edge alignment -- so the span is normalised below rather than assumed positive,
    # keeping the two triangles wound the same way whichever direction the bar runs.
    offsets = (
        (-float(width) / 2.0, float(width) / 2.0) if align == "center" else (0.0, float(width))
    )
    lo, hi = min(offsets), max(offsets)
    patches = []
    rgba = list(_normalize_rgba(color, n=None))
    if alpha is not None:
        rgba[3] *= float(alpha)
    _QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    for idx, (xc, h, btm) in enumerate(zip(x_arr, h_arr, b_arr)):
        left, right = xc + lo, xc + hi
        verts = np.array(
            [[left, btm], [left, btm + h], [right, btm + h], [right, btm]],
            dtype=np.float32,
        )
        add_patch(
            verts,
            indices=_QUAD_INDICES,
            mode="triangles",
            face_color=tuple(rgba),
            edge_color=tuple(rgba),
            label=label if idx == 0 else None,
        )
        # The bar's Layer, not `add_patch`'s figure: the docstring promises patch layers,
        # and `bar_label` / any per-bar work needs the artist. `add_patch` predates that
        # contract and returns the figure, so appending its result gave N copies of it.
        _get_or_create_plot().scene.layers[-1].metadata["artist"] = "bar"
        patches.append(_get_or_create_plot().scene.layers[-1])
    if yerr is not None or xerr is not None:
        kw = dict(error_kw or {})
        kw.setdefault("s", 0)
        patches.extend(
            errorbar(
                x_arr,
                b_arr + h_arr,
                yerr=yerr,
                xerr=xerr,
                fmt="none",
                ecolor="k" if ecolor is None else ecolor,
                capsize=0.0 if capsize is None else capsize,
                **kw,
            )
        )
    return patches


def grouped_bar(
    heights: Union[Sequence[ArrayLike], dict],
    *,
    positions: Optional[ArrayLike] = None,
    group_spacing: Optional[float] = 1.5,
    bar_spacing: Optional[float] = 0.0,
    tick_labels: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
    orientation: str = "vertical",
    colors: Optional[Sequence[ColorLike]] = None,
    **kwargs: Any,
) -> List[list]:
    """Make a grouped bar plot: one cluster of bars per category, one bar per series.

    matplotlib 3.11+ API (provisional there; implemented here directly on top of
    repeated :func:`bar`/:func:`barh` calls -- one per series, each shifted so the bars
    in a category sit side by side instead of stacked or overlapping).

    Args:
        heights: A dict of ``{series_name: values}``, or a sequence of value arrays (one
            per series). Every series must have the same length -- one value per category.
        positions (array-like, optional): Category centres. Defaults to ``0, 1, 2, ...``.
        group_spacing (float, optional): Width available to each category's whole cluster
            of bars, in data units. Defaults to 1.5.
        bar_spacing (float, optional): Gap between adjacent bars within a category, as a
            fraction of one bar's width. Defaults to 0.
        tick_labels (sequence of str, optional): Category names, drawn at ``positions``.
        labels (sequence of str, optional): Legend label per series. Ignored if ``heights``
            is a dict -- the dict keys are used instead.
        orientation ({'vertical', 'horizontal'}, optional): Vertical draws with
            :func:`bar`; horizontal draws with :func:`barh`. Defaults to 'vertical'.
        colors (sequence, optional): One color per series. Defaults to the style cycle.
        **kwargs: Forwarded to :func:`bar`/:func:`barh` for every series (``alpha``,
            ``edgecolors`` are not supported there and are ignored the same way).

    Returns:
        list: One entry per series, each the list of bar layers :func:`bar` returned for it.

    Examples:
        >>> gplt.grouped_bar({"2024": [3, 5, 2], "2025": [4, 6, 3]}, tick_labels=["a", "b", "c"])
    """
    if orientation not in ("vertical", "horizontal"):
        raise ValueError(
            f"grouped_bar(): orientation must be 'vertical' or 'horizontal', got "
            f"{orientation!r}"
        )
    if isinstance(heights, dict):
        series_names = list(heights.keys())
        series_values = [np.asarray(v, dtype=np.float64) for v in heights.values()]
    else:
        series_values = [np.asarray(v, dtype=np.float64) for v in heights]
        no_labels: List[Optional[str]] = [None] * len(series_values)
        series_names = list(labels) if labels is not None else no_labels
    n_series = len(series_values)
    if n_series == 0:
        return []
    n_categories = len(series_values[0])
    for v in series_values:
        if len(v) != n_categories:
            raise ValueError("grouped_bar(): every series must have the same length")

    centres = (
        np.arange(n_categories, dtype=np.float64)
        if positions is None
        else np.asarray(positions, dtype=np.float64)
    )
    if len(centres) != n_categories:
        raise ValueError("grouped_bar(): positions must have one entry per category")

    spacing = 1.5 if group_spacing is None else float(group_spacing)
    gap = 0.0 if bar_spacing is None else float(bar_spacing)
    bar_width = spacing / (n_series + (n_series - 1) * gap)
    draw = bar if orientation == "vertical" else barh
    # barh()'s *thickness* keyword is confusingly named "height" (its "width" is the bar's
    # length instead, matching matplotlib's own reversal) -- so the keyword this loop
    # passes has to swap along with the function it is calling.
    thickness_kw = "width" if orientation == "vertical" else "height"

    plot_obj = _get_or_create_plot()
    results: List[list] = []
    for i, values in enumerate(series_values):
        offset = (i - (n_series - 1) / 2.0) * bar_width * (1.0 + gap)
        color = colors[i] if colors is not None and i < len(colors) else _next_cycle_color(plot_obj)
        results.append(
            draw(
                centres + offset,
                values,
                color=color,
                label=series_names[i],
                **{thickness_kw: bar_width},
                **kwargs,
            )
        )

    if tick_labels is not None:
        xticks(list(centres), list(tick_labels))

    return results


def hist(
    x: Sequence[float],
    bins: Union[int, Sequence[float], str] = 10,
    range: Optional[Tuple[float, float]] = None,
    density: bool = False,
    weights: Optional[ArrayLike] = None,
    cumulative: bool = False,
    bottom: Optional[Union[float, ArrayLike]] = None,
    histtype: str = "bar",
    align: str = "mid",
    orientation: str = "vertical",
    rwidth: Optional[float] = None,
    log: bool = False,
    stacked: bool = False,
    *,
    color: ColorLike = (0.2, 0.4, 0.8, 1.0),
    alpha: Optional[float] = None,
    label: Optional[str] = None,
    data: Optional[Any] = None,
):
    """Create a histogram from data values.

    Computes a histogram by binning data and rendering bars for each bin.
    Supports both uniform bins (specified by count) and custom bin edges.
    Can normalize to probability density for comparison of datasets with
    different sample sizes.

    Args:
        x (array-like): Data values to histogram. Shape (N,).
        bins (int, array-like or str, optional): Bin specification.
            If int, number of equal-width bins. If array, bin edges
            (len(bins) - 1 bins created). A string selects one of numpy's
            binning strategies ('auto', 'fd', 'sturges', ...). Defaults to 10.
        range (tuple, optional): ``(lower, upper)`` bin range. Values outside
            it are ignored. Defaults to the data's own min/max.
        density (bool, optional): If True, normalize histogram so that
            bar area sums to 1. If False, counts per bin. Defaults to False.
        weights (array-like, optional): Per-value weights, same shape as ``x``.
            Each value contributes its weight instead of 1.
        cumulative (bool, optional): If True, each bin holds the counts of that
            bin plus every lower one. If negative, accumulates from the right.
        bottom (float or array-like, optional): Baseline for the bars.
        histtype (str, optional): 'bar' (default), 'step' (outline only) or
            'stepfilled'. 'barstacked' needs multiple datasets and is ignored.
        align (str, optional): 'left', 'mid' (default) or 'right' -- which part
            of the bin the bar is centred on.
        orientation (str, optional): Accepted for matplotlib parity. GLPlot has
            no horizontal bar renderer, so 'horizontal' is ignored.
        rwidth (float, optional): Bar width as a fraction of the bin width.
            Ignored for the step histtypes, as in matplotlib.
        log (bool, optional): Accepted for matplotlib parity. GLPlot's projection
            is linear and has no log-scaled axis, so this is ignored.
        stacked (bool, optional): Accepted for matplotlib parity. Needs multiple
            datasets, which this signature does not take, so it is ignored.
        color (str or tuple, optional): Bar color. Defaults to blue.
        alpha (float, optional): Transparency (0.0-1.0). Defaults to 1.0.
        label (str, optional): Legend label. Defaults to None.
        data (indexable, optional): If given, ``x`` and ``weights`` may be keys
            into it (a DataFrame, dict, structured array, ...).

    Returns:
        tuple: (counts, bin_edges, patches) where:
            - counts: Histogram counts (or densities) per bin
            - bin_edges: N+1 bin edge values
            - patches: Patch layers for rendering

    Examples:
        Simple histogram:

        >>> data = [1.2, 1.5, 2.1, 2.3, 2.5, 3.1, 3.2, 3.5, 4.0]
        >>> counts, edges, _ = gplt.hist(data, bins=5)
        >>> gplt.show()

        With density normalization:

        >>> counts, edges, _ = gplt.hist(data, bins=10, density=True)

        Restricted to a range, with an outline instead of bars:

        >>> gplt.hist(data, bins=20, range=(0, 5), histtype='step')

        With custom bin edges:

        >>> custom_bins = [0, 1, 2, 3, 4, 5]
        >>> gplt.hist(data, bins=custom_bins, color='green')
    """
    x, weights = _resolve_data_args("hist", data, x, weights)
    _warn_unsupported(
        "hist",
        {
            "log": log or None,
            "stacked": stacked or None,
            "orientation": orientation if orientation != "vertical" else None,
        },
        {
            "log": "has no effect: GLPlot's projection is linear and has no log-scaled axis",
            "stacked": "has no effect: it needs multiple datasets, and hist() takes one",
            "orientation": "has no effect: GLPlot has no horizontal bar renderer, so the "
            "histogram is drawn vertically",
        },
    )

    values = _as_float_array(x, ndim=1, name="x")
    w_arr = None if weights is None else _as_float_array(weights, ndim=1, name="weights")
    if w_arr is not None and len(w_arr) != len(values):
        raise ValueError("weights must have the same length as x")

    counts, edges = np.histogram(values, bins=bins, range=range, density=density, weights=w_arr)

    if cumulative:
        # A negative `cumulative` accumulates from the right -- matplotlib's own spelling
        # for a survival curve, and cheap enough to honour that refusing it would be worse.
        counts = np.cumsum(counts[::-1])[::-1] if cumulative < 0 else np.cumsum(counts)

    widths = np.diff(edges)
    anchors = {
        "mid": 0.5 * (edges[:-1] + edges[1:]),
        "left": edges[:-1],
        "right": edges[1:],
    }
    if align not in anchors:
        raise ValueError(f"unsupported align: {align!r}. Expected 'left', 'mid' or 'right'.")
    centers = anchors[align]

    if histtype in ("step", "stepfilled"):
        # matplotlib draws these as an outline through the bin edges, closed down to the
        # baseline at both ends, and ignores rwidth for them. `where='post'` is what keeps
        # each level spanning its own bin rather than lagging one edge behind it.
        _warn_unsupported(
            "hist",
            {"rwidth": rwidth},
            {"rwidth": f"has no effect with histtype={histtype!r}, as in matplotlib"},
        )
        base = 0.0 if bottom is None else float(np.min(_as_float_array(np.atleast_1d(bottom))))
        xs = np.concatenate(([edges[0]], edges))
        ys = np.concatenate(([base], counts, [base]))
        artists = step(xs, ys, where="post", color=color, alpha=alpha, label=label)
        return counts, edges, artists

    if histtype not in ("bar", "barstacked"):
        raise ValueError(
            f"unsupported histtype: {histtype!r}. "
            "Expected 'bar', 'barstacked', 'step' or 'stepfilled'."
        )

    # A single width for every bar, so unequal custom bin edges cannot make bars overlap.
    width = float(np.min(widths))
    if rwidth is not None:
        width *= float(np.clip(rwidth, 0.0, 1.0))
    artists = bar(
        centers,
        counts,
        width=width,
        bottom=0 if bottom is None else bottom,
        color=color,
        alpha=alpha,
        label=label,
    )
    return counts, edges, artists


def _hexbin_assign(
    x: np.ndarray,
    y: np.ndarray,
    gridsize: Tuple[int, int],
    extent: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign each point to a hexagon, returning ``(centres_x, centres_y, index)``.

    A hexagonal lattice is two rectangular lattices, the second offset by half a cell in
    both directions. Every point is therefore a candidate for one cell of each, and the
    hexagon it belongs to is whichever of those two centres is nearer -- that nearest
    neighbour rule *is* the hexagonal tessellation, which is why this needs no polygon
    tests. It is also how matplotlib's own hexbin assigns, so the counts agree with it.
    """
    nx, ny = gridsize
    xmin, xmax, ymin, ymax = extent
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    # Cell indices in each lattice. The second is shifted by half a cell, so a point in
    # the seam between two cells of one lattice sits at the centre of one of the other's.
    ix1 = np.floor((x - xmin) / dx).astype(np.int64)
    iy1 = np.floor((y - ymin) / dy).astype(np.int64)
    ix2 = np.floor((x - xmin - dx / 2.0) / dx).astype(np.int64)
    iy2 = np.floor((y - ymin - dy / 2.0) / dy).astype(np.int64)
    np.clip(ix1, 0, nx - 1, out=ix1)
    np.clip(iy1, 0, ny - 1, out=iy1)
    np.clip(ix2, 0, nx - 2, out=ix2)
    np.clip(iy2, 0, ny - 2, out=iy2)

    cx1 = xmin + (ix1 + 0.5) * dx
    cy1 = ymin + (iy1 + 0.5) * dy
    cx2 = xmin + (ix2 + 1.0) * dx
    cy2 = ymin + (iy2 + 1.0) * dy

    # Compared in *cell* units, not world units: an anisotropic extent (a wide, short
    # window) would otherwise let one axis dominate the distance and the lattice would
    # shear into stripes.
    d1 = ((x - cx1) / dx) ** 2 + ((y - cy1) / dy) ** 2
    d2 = ((x - cx2) / dx) ** 2 + ((y - cy2) / dy) ** 2
    use_second = d2 < d1

    centres_x = np.where(use_second, cx2, cx1)
    centres_y = np.where(use_second, cy2, cy1)
    # A single integer id per distinct hexagon, so the counts can be reduced with bincount
    # rather than a Python dict keyed on a float pair.
    ident = (
        np.where(use_second, 1, 0) * (nx * ny)
        + np.where(use_second, iy2, iy1) * nx
        + np.where(use_second, ix2, ix1)
    )
    return centres_x, centres_y, ident


def _hexagon_geometry(centres: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """A flat-topped hexagon around each centre, as vertices plus triangle-fan indices."""
    # The lattice's own proportions, so the hexagons tile the extent they were binned on
    # rather than being round in a window that is not.
    angles = np.deg2rad(np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0], dtype=np.float64))
    rx, ry = dx / 2.0, dy / np.sqrt(3.0)
    offx, offy = rx * np.cos(angles), ry * np.sin(angles)

    n = len(centres)
    verts = np.empty((n * 7, 2), dtype=np.float32)
    verts[0::7] = centres  # the fan's hub
    for k in range(6):
        verts[k + 1 :: 7, 0] = centres[:, 0] + offx[k]
        verts[k + 1 :: 7, 1] = centres[:, 1] + offy[k]

    base = np.arange(n, dtype=np.uint32) * 7
    fan = np.empty((n, 18), dtype=np.uint32)
    for k in range(6):
        fan[:, 3 * k] = base
        fan[:, 3 * k + 1] = base + 1 + k
        fan[:, 3 * k + 2] = base + 1 + ((k + 1) % 6)
    return verts, fan.ravel()


def hexbin(
    x: ArrayLike,
    y: ArrayLike,
    C: Optional[ArrayLike] = None,
    gridsize: Union[int, Tuple[int, int]] = 100,
    bins: Optional[str] = None,
    xscale: str = "linear",
    yscale: str = "linear",
    extent: Optional[Tuple[float, float, float, float]] = None,
    cmap: Optional[str] = None,
    norm: Optional[Any] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: Optional[float] = None,
    mincnt: Optional[int] = None,
    reduce_C_function: Any = None,
    marginals: bool = False,
    edgecolors: Optional[ColorLike] = None,
    linewidths: Optional[float] = None,
    colorizer: Optional[Any] = None,
    *,
    label: Optional[str] = None,
    data: Optional[Any] = None,
):
    """Bin points into hexagons and colour each by how many landed in it.

    The honest alternative to a scatter of a million overlapping points: a hexagonal
    lattice tiles the plane without the visual artefacts a square grid's rows and
    columns impose on the eye.

    Args:
        x, y (array-like): The points to bin. Shape (N,).
        C (array-like, optional): A value per point. When given, each hexagon is
            coloured by ``reduce_C_function`` over its points instead of by their
            count.
        gridsize (int or tuple, optional): Hexagons across the x-axis, or
            ``(nx, ny)``. A bare int derives ``ny`` from it so the hexagons come
            out regular. Defaults to 100.
        bins (str, optional): 'log' colours by log10(count + 1). None colours by
            the count itself.
        xscale, yscale (str, optional): Accepted for matplotlib parity. GLPlot's
            projection is linear; 'log' is ignored.
        extent (tuple, optional): ``(xmin, xmax, ymin, ymax)`` to bin over.
            Defaults to the data's own bounds.
        cmap (str, optional): Colormap name. Defaults to 'viridis'.
        norm (Normalize or str, optional): How the values map onto the colormap.
        vmin, vmax (float, optional): Colormap limits. Mutually exclusive with
            ``norm``, as in matplotlib.
        alpha (float, optional): Transparency.
        mincnt (int, optional): Hexagons holding fewer than this many points are
            not drawn. Defaults to drawing every non-empty one.
        reduce_C_function (callable, optional): How to reduce ``C`` per hexagon.
            Defaults to ``np.mean``.
        marginals (bool, optional): Accepted for matplotlib parity. GLPlot has no
            marginal-histogram gutter to draw into. Ignored.
        edgecolors, linewidths (optional): Accepted for matplotlib parity. The hexagons
            are drawn as filled triangles with no separate outline pass. Ignored.
        label (str, optional): Legend label.
        data (indexable, optional): If given, ``x``, ``y`` and ``C`` may be keys
            into it.

    Returns:
        Layer: One patch layer holding every hexagon.

    Note:
        Unlike matplotlib this returns a single layer rather than a PolyCollection
        with a per-hexagon array, and empty hexagons are never drawn -- colouring
        them would make "nothing landed here" look like the colormap's low end.

    Examples:
        >>> gplt.hexbin(x, y, gridsize=40)
        >>> gplt.hexbin(x, y, C=weights, reduce_C_function=np.max)
    """

    x, y, C = _resolve_data_args("hexbin", data, x, y, C)
    _warn_unsupported(
        "hexbin",
        {
            "marginals": marginals or None,
            "edgecolors": edgecolors,
            "linewidths": linewidths,
            "colorizer": colorizer,
        },
        {
            "marginals": "has no effect: GLPlot has no marginal-histogram gutter. Draw the "
            "two hist() calls into their own panels instead",
            "edgecolors": "has no effect: the hexagons are one filled triangle mesh with no "
            "separate outline pass",
            "linewidths": "has no effect: the hexagons are drawn filled, without outlines",
            "colorizer": "has no effect: pass cmap=/norm=/vmin=/vmax= directly, there is no "
            "shared Colorizer object across artists",
        },
    )
    _warn_unsupported(
        "hexbin",
        {
            "xscale": xscale if xscale != "linear" else None,
            "yscale": yscale if yscale != "linear" else None,
        },
        {
            "xscale": "has no effect: GLPlot's projection is linear and has no log-scaled "
            "axis, so the bins are spaced linearly",
            "yscale": "has no effect: GLPlot's projection is linear and has no log-scaled "
            "axis, so the bins are spaced linearly",
        },
    )
    x_arr = _as_float_array(x, ndim=1, name="x")
    y_arr = _as_float_array(y, ndim=1, name="y")
    if len(x_arr) != len(y_arr):
        raise ValueError("x and y must have the same length")
    if len(x_arr) == 0:
        raise ValueError("hexbin(): x is empty")

    if isinstance(gridsize, (tuple, list)):
        nx, ny = int(gridsize[0]), int(gridsize[1])
    else:
        # matplotlib's own derivation: sqrt(3) is the ratio that makes a hexagon on a
        # lattice of this aspect regular rather than squashed.
        nx = int(gridsize)
        ny = max(1, int(nx / np.sqrt(3)))
    if nx < 1 or ny < 1:
        raise ValueError(f"gridsize must be positive, got {gridsize!r}")

    if extent is None:
        xmin, xmax = float(np.min(x_arr)), float(np.max(x_arr))
        ymin, ymax = float(np.min(y_arr)), float(np.max(y_arr))
    else:
        xmin, xmax, ymin, ymax = (float(v) for v in extent)
    # A zero span puts every point in one cell and divides by zero on the way; widening
    # it is what `fit_bounds` does with the same degeneracy.
    if xmax - xmin < 1e-12:
        xmin, xmax = xmin - 0.5, xmax + 0.5
    if ymax - ymin < 1e-12:
        ymin, ymax = ymin - 0.5, ymax + 0.5

    cx, cy, ident = _hexbin_assign(x_arr, y_arr, (nx, ny), (xmin, xmax, ymin, ymax))
    unique, inverse = np.unique(ident, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    centres = np.empty((len(unique), 2), dtype=np.float64)
    centres[:, 0] = np.bincount(inverse, weights=cx) / counts
    centres[:, 1] = np.bincount(inverse, weights=cy) / counts

    if C is None:
        values = counts
    else:
        c_arr = _as_float_array(C, ndim=1, name="C")
        if len(c_arr) != len(x_arr):
            raise ValueError("C must have the same length as x")
        reduce_fn = np.mean if reduce_C_function is None else reduce_C_function
        # A Python loop over hexagons, not points: `reduce_C_function` is arbitrary, so
        # there is no bincount trick that works for np.max as well as np.mean.
        values = np.array(
            [float(reduce_fn(c_arr[inverse == i])) for i in np.arange(len(unique))],
            dtype=np.float64,
        )

    keep = np.ones(len(unique), dtype=bool)
    if mincnt is not None:
        keep &= counts >= int(mincnt)
    if not keep.any():
        raise ValueError("hexbin(): mincnt excluded every hexagon")
    centres, values = centres[keep], values[keep]

    if bins == "log":
        # +1 so a single-point hexagon maps to 0 rather than to -inf.
        values = np.log10(values + 1.0)
    elif bins is not None:
        raise ValueError(f"unsupported bins: {bins!r}. Expected 'log' or None.")

    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    verts, indices = _hexagon_geometry(centres, dx, dy)

    plot_obj = _get_or_create_plot()
    # Seven colour rows per hexagon (hub + six corners), all its own colour. Painted
    # through the shared helper so `clim()`/`set_cmap()` can repaint it afterwards.
    per_vertex = np.repeat(values, 7)
    add_patch(
        verts,
        indices=indices,
        mode="triangles",
        colors=np.zeros((len(verts), 4), dtype=np.float32),
        label=label,
    )
    layer = plot_obj.scene.layers[-1]
    _paint_patch_mappable(layer, per_vertex, cmap, norm, vmin, vmax, alpha)
    # `cvalues` is what the Scene panel's colormap picker re-maps from, exactly as for a
    # colormapped scatter -- without it the cmap becomes undoable decoration.
    layer.metadata.update({"artist": "hexbin", "counts": counts[keep], "cvalues": values})
    _set_dirty(plot_obj)
    return _set_current_mappable(layer)


def eventplot(
    positions: ArrayLike,
    orientation: str = "horizontal",
    lineoffsets: Union[float, ArrayLike] = 1.0,
    linelengths: Union[float, ArrayLike] = 1.0,
    linewidths: Optional[Union[float, ArrayLike]] = None,
    colors: Optional[Union[ColorLike, Sequence[ColorLike]]] = None,
    linestyles: str = "solid",
    *,
    alpha: Optional[float] = None,
    label: Optional[str] = None,
    data: Optional[Any] = None,
):
    """Draw a raster of identical tick marks -- one per event.

    Args:
        positions (array-like): Event positions. A 1-D array is one row; a list of
            arrays is one row per array, which is the spike-raster shape.
        orientation (str, optional): 'horizontal' (default) lays the events out
            along x with vertical ticks; 'vertical' transposes that.
        lineoffsets (float or array-like, optional): Where each row sits on the
            other axis. Defaults to 1, and to ``1, 2, 3...`` for several rows.
        linelengths (float or array-like, optional): Tick length in data units.
        linewidths (float or array-like, optional): Tick width **in data units**,
            not points -- see the note. Defaults to 1/1000 of the data span.
        colors (color or sequence, optional): One colour, or one per row.
        linestyles (str, optional): Accepted for matplotlib parity. GLPlot draws
            the ticks as solid quads. Ignored.
        alpha (float, optional): Transparency.
        label (str, optional): Legend label.
        data (indexable, optional): If given, ``positions`` may be a key into it.

    Returns:
        list: One patch layer per row.

    Note:
        matplotlib measures ``linewidths`` in points, and its ticks keep their
        thickness as you zoom. GLPlot has no primitive for many disconnected
        pixel-width segments in a single layer, so each row is one patch of thin
        quads and the width is in **data units**: zooming in thickens the ticks.
        The trade buys a raster of a million events as one layer instead of a
        million.

    Examples:
        >>> gplt.eventplot(spike_times)
        >>> gplt.eventplot([neuron_a, neuron_b, neuron_c], colors=['r', 'g', 'b'])
    """
    if orientation not in ("horizontal", "vertical"):
        raise ValueError(
            f"unsupported orientation: {orientation!r}. Expected 'horizontal' or 'vertical'."
        )
    (positions,) = _resolve_data_args("eventplot", data, positions)
    _warn_unsupported(
        "eventplot",
        {"linestyles": linestyles if linestyles not in ("solid", "-") else None},
        {"linestyles": "has no effect: the ticks are drawn as solid quads"},
    )

    # One row, or many: `positions` is either a flat sequence of events or a sequence of
    # such sequences. Deciding by the first element's shape rather than by `np.ndim` on the
    # whole thing, which a ragged list of arrays cannot answer without a warning.
    raw = list(positions) if isinstance(positions, (list, tuple)) else [positions]
    if raw and np.ndim(raw[0]) == 0:
        raw = [positions]
    rows = [np.atleast_1d(_as_float_array(r, name="positions")) for r in raw]

    def _per_row(value, default):
        if value is None:
            value = default
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
        if len(arr) == 1:
            return np.full(len(rows), arr[0])
        if len(arr) != len(rows):
            raise ValueError(f"expected 1 or {len(rows)} values, got {len(arr)}")
        return arr

    # matplotlib's default when several rows share one offset: stack them 1, 2, 3...
    # rather than draw every row on top of the first.
    if len(rows) > 1 and np.ndim(lineoffsets) == 0 and float(lineoffsets) == 1.0:
        offsets = np.arange(1, len(rows) + 1, dtype=np.float64)
    else:
        offsets = _per_row(lineoffsets, 1.0)
    lengths = _per_row(linelengths, 1.0)

    if linewidths is None:
        # A tick must be visible without being a block. The data span is the only scale
        # available here, since the width is in data units rather than points.
        finite = np.concatenate([r for r in rows if len(r)]) if any(len(r) for r in rows) else None
        span = float(np.ptp(finite)) if finite is not None and len(finite) > 1 else 1.0
        widths = np.full(len(rows), max(span, 1e-9) / 1000.0)
    else:
        widths = _per_row(linewidths, 1.0)

    # "One colour" and "one colour per row" are genuinely ambiguous -- (1, 0, 0) is a red,
    # and ["red", "blue"] is two. matplotlib resolves it with `is_color_like`, and reusing
    # that answer is the only way the two libraries agree on every spelling rather than on
    # the ones a hand-rolled test happened to consider.
    from matplotlib.colors import is_color_like

    if colors is None:
        row_colors = [_color_cycle()[i % len(_color_cycle())] for i in range(len(rows))]
    elif is_color_like(colors):
        row_colors = [colors] * len(rows)
    else:
        row_colors = list(colors)
        if len(row_colors) == 1:
            row_colors = row_colors * len(rows)
        elif len(row_colors) != len(rows):
            raise ValueError(f"expected 1 or {len(rows)} colors, got {len(row_colors)}")

    plot_obj = _get_or_create_plot()
    layers = []
    for i, events in enumerate(rows):
        if len(events) == 0:
            continue
        half_len, half_w = lengths[i] / 2.0, widths[i] / 2.0
        lo, hi = offsets[i] - half_len, offsets[i] + half_len
        # Each tick is a quad: two triangles, four corners. Built vectorised -- an event
        # raster is the one plot where the loop would be over the data itself.
        a, b = events - half_w, events + half_w
        if orientation == "horizontal":
            corners = np.stack(
                [
                    np.column_stack([a, np.full_like(a, lo)]),
                    np.column_stack([b, np.full_like(b, lo)]),
                    np.column_stack([b, np.full_like(b, hi)]),
                    np.column_stack([a, np.full_like(a, hi)]),
                ],
                axis=1,
            )
        else:
            corners = np.stack(
                [
                    np.column_stack([np.full_like(a, lo), a]),
                    np.column_stack([np.full_like(b, lo), b]),
                    np.column_stack([np.full_like(b, hi), b]),
                    np.column_stack([np.full_like(a, hi), a]),
                ],
                axis=1,
            )
        verts = corners.reshape(-1, 2).astype(np.float32)
        base = np.arange(len(events), dtype=np.uint32) * 4
        indices = np.column_stack([base, base + 1, base + 2, base, base + 2, base + 3]).ravel()

        rgba = list(_normalize_rgba(row_colors[i]))
        if alpha is not None:
            rgba[3] *= float(alpha)
        add_patch(
            verts,
            indices=indices,
            mode="triangles",
            face_color=tuple(rgba),
            edge_color=tuple(rgba),
            label=label if i == 0 else None,
        )
        layer = plot_obj.scene.layers[-1]
        layer.metadata["artist"] = "eventplot"
        layers.append(layer)

    _set_dirty(plot_obj)
    return layers


def hist2d(
    x,
    y,
    bins=100,
    range=None,
    density: bool = False,
    weights: Optional[ArrayLike] = None,
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
    *,
    # None, not "magma", so `set_cmap()` can reach a hist2d that named no colormap. The
    # magma default it has always had survives as `_resolve_cmap`'s last resort.
    cmap: Optional[str] = None,
    s: Optional[float] = None,
    label: Optional[str] = None,
    data: Optional[Any] = None,
):
    """Create a 2D histogram heatmap of bivariate data.

    Bins 2D scattered data into a regular grid and displays as colored
    heatmap. Useful for visualizing density distributions and correlations
    in large datasets where scatter plots would be overplotted.

    Args:
        x (array-like): X-coordinates of points. Shape (N,).
        y (array-like): Y-coordinates of points. Shape (N,).
        bins (int or array-like, optional): Bin specification.
            If int, creates bins x bins grid. If 2-tuple, (xbins, ybins).
            If array, bin edges for both axes. Defaults to 100.
        range (tuple, optional): ((xmin, xmax), (ymin, ymax)) data range
            for binning. If None, uses data extrema. Defaults to None.
        density (bool, optional): Normalize by total count. Defaults to False.
        weights (array-like, optional): Per-point weights, same shape as ``x``.
            Each point contributes its weight to its cell instead of 1.
        cmin (float, optional): Cells holding less than this are not drawn.
        cmax (float, optional): Cells holding more than this are not drawn.
        cmap (str, optional): Colormap name. Defaults to 'magma'.
        s (float, optional): Point size for display. Auto-computed if None.
            Defaults to None.
        label (str, optional): Legend label. Defaults to None.
        data (indexable, optional): If given, ``x``, ``y`` and ``weights`` may be
            keys into it (a DataFrame, dict, structured array, ...).

    Note:
        Empty cells are never drawn, as in matplotlib -- an unset ``cmin`` still
        leaves zero-count cells out rather than colouring them the colormap's low
        end.

    Returns:
        tuple: (counts, xedges, yedges, layer) where:
            - counts: 2D histogram counts (shape: (nbins_x, nbins_y))
            - xedges: N+1 x-axis bin edges
            - yedges: M+1 y-axis bin edges
            - layer: The scatter layer rendering the heatmap

    Examples:
        Simple 2D histogram:

        >>> x = np.random.normal(0, 1, 10000)
        >>> y = np.random.normal(0, 1, 10000)
        >>> counts, xe, ye, _ = gplt.hist2d(x, y, bins=50)
        >>> gplt.show()

        With custom range and colormap:

        >>> _, _, _, _ = gplt.hist2d(x, y, bins=40, range=[[-3,3],[-3,3]],
        ...                           density=True, cmap='hot')
    """
    x, y, weights = _resolve_data_args("hist2d", data, x, y, weights)
    x_arr = _as_float_array(x, ndim=1, name="x")
    y_arr = _as_float_array(y, ndim=1, name="y")
    if len(x_arr) != len(y_arr):
        raise ValueError("x and y must have the same length")
    w_arr = None if weights is None else _as_float_array(weights, ndim=1, name="weights")
    if w_arr is not None and len(w_arr) != len(x_arr):
        raise ValueError("weights must have the same length as x")
    counts, xedges, yedges = np.histogram2d(
        x_arr, y_arr, bins=bins, range=range, density=density, weights=w_arr
    )
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    xx, yy = np.meshgrid(xc, yc, indexing="ij")
    values = counts.ravel()
    # An empty cell is left out whatever cmin says: matplotlib does the same, and drawing
    # it would colour "nothing landed here" as the colormap's low end -- indistinguishable
    # from a cell that got the fewest hits.
    mask = values > 0
    if cmin is not None:
        mask &= values >= float(cmin)
    if cmax is not None:
        mask &= values <= float(cmax)
    layer = scatter(
        xx.ravel()[mask],
        yy.ravel()[mask],
        c=values[mask],
        # Resolved here rather than left to `scatter`, whose own last resort is viridis:
        # a hist2d that names no colormap has always been magma, and must stay magma.
        cmap=_resolve_cmap(cmap, "magma"),
        s=s or max(2.0, 9000.0 / max(len(values), 1)),
        marker="s",
        label=label,
    )
    layer.metadata.update(
        {"artist": "hist2d", "counts": counts, "xedges": xedges, "yedges": yedges}
    )
    return counts, xedges, yedges, layer


def function(
    f: Any,
    xlim: Tuple[float, float] = (-10.0, 10.0),
    *,
    ylim: Optional[Tuple[float, float]] = None,
    domain: Optional[Tuple[float, float]] = None,
    color: Optional[ColorLike] = None,
    linewidth: float = 1.5,
    samples_per_px: float = 1.0,
    max_samples: int = 8192,
    label: Optional[str] = None,
) -> FunctionLayer:
    """Plot ``y = f(x)`` **resampled to the screen**, so zooming reveals real detail.

    The difference from ``plot(x, y)`` is where the samples live. ``plot`` takes a fixed
    table: zoom in and you magnify the points it was given, so ``sin(1/x)`` degenerates
    into a few straight segments long before you reach the interesting part. This layer
    samples ``f`` across whatever x range is currently visible, at about one sample per
    pixel column, and **re-evaluates every time the view moves**.

    So:

    * resolution is constant — always what the display can show;
    * cost is constant — ~2000 evaluations per frame at any zoom depth;
    * detail is unbounded — features finer than any fixed sampling appear as you approach.

    Use it for anything you can *evaluate*: analytic expressions, recursions, series,
    special functions. Do not use it for measured data — there is no ``f`` to re-evaluate,
    and resampling measurements would be inventing them; ``plot`` is right for that.

    Args:
        f: vectorised ``f(x: ndarray) -> ndarray``. Non-finite results become gaps.
        xlim: the interval to frame initially. Zooming past it re-evaluates there too.
        ylim: optional initial y framing. Without it the first sampling autoscales, which
            for a function with a pole can be dominated by the pole — pass one to pin it.
        domain: ``(x0, x1)`` outside which ``f`` is never called, for a function only
            defined on an interval (``sqrt`` on the negatives, say).
        samples_per_px: 1.0 is one sample per pixel column; 2.0 supersamples.
        max_samples: hard ceiling on evaluations per frame.
        label: legend label.

    Returns:
        The :class:`~glplot.core.layers.FunctionLayer`.

    Examples:
        >>> gplt.function(np.sin, (-10, 10))
        >>> gplt.function(lambda x: np.sin(1 / x), (-1, 1), ylim=(-1.2, 1.2))
        >>> gplt.show()      # scroll into x=0: the oscillation keeps resolving
    """
    plot_obj = _get_or_create_plot()
    rgba = _normalize_rgba(color or _next_cycle_color(plot_obj), n=None)
    layer = FunctionLayer(
        func=f,
        domain=domain,
        samples_per_px=samples_per_px,
        max_samples=max_samples,
        color=tuple(float(v) for v in rgba),
        width=float(linewidth),
        label=label or getattr(f, "__name__", "f(x)"),
    )
    # Sample once up front so the layer has geometry (and therefore bounds) before the
    # first frame; the engine re-samples from then on whenever the view moves.
    width_px = max(int(getattr(plot_obj, "width", 1280)), 1)
    layer.resample(float(xlim[0]), float(xlim[1]), width_px)
    plot_obj.scene.layers.append(layer)

    if ylim is not None:
        plot_obj.set_view(xlim=(float(xlim[0]), float(xlim[1])), ylim=ylim)
    else:
        plot_obj.set_view(xlim=(float(xlim[0]), float(xlim[1])))
    _set_dirty(plot_obj)
    return layer


def _add_fractal(
    fractal_type: str,
    *,
    extent: Tuple[float, float, float, float],
    julia_c: Tuple[float, float],
    max_iter: int,
    cmap: str,
    gain: float,
    inset_color: Tuple[float, float, float],
    label: Optional[str],
) -> FractalLayer:
    """Shared body of :func:`mandelbrot` and :func:`julia`."""
    plot_obj = _get_or_create_plot()
    layer = FractalLayer(
        extent=extent,
        fractal_type=fractal_type,
        julia_c=julia_c,
        max_iter=int(max_iter),
        cmap=str(cmap),
        gain=float(gain),
        inset_color=inset_color,
        label=label or fractal_type,
    )
    plot_obj.scene.layers.append(layer)
    # Frame the whole set the first time, then leave zooming to the user; the shader
    # refines detail as they go.
    plot_obj.set_view(xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]))
    _apply_equal_aspect(plot_obj)
    _set_dirty(plot_obj)
    return layer


def mandelbrot(
    center: Tuple[float, float] = (-0.6, 0.0),
    span: float = 1.4,
    *,
    max_iter: int = 200,
    cmap: str = "magma",
    gain: float = 1.0,
    inset_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    aspect: float = 1.35,
    label: Optional[str] = None,
) -> FractalLayer:
    """Plot the Mandelbrot set as a **live, GPU-computed** escape-time field.

    Unlike ``imshow`` of a baked array, this is not a fixed image: the escape loop runs in
    a fragment shader, once per screen pixel, every frame. Zoom in with the scroll wheel and
    the boundary refines to the new scale automatically — there is no recompute to trigger
    and no resolution to pick, because the resolution *is* the screen. This is the answer to
    "recompute on zoom for more precision": it is continuous and free.

    Args:
        center: ``(re, im)`` the view is centred on.
        span: half-height of the initial view in the complex plane; ``aspect`` widens it.
        max_iter: iteration budget at the initial framing. The renderer raises it as you
            zoom in (deeper zoom needs more iterations to resolve the filaments).
        cmap: colour scheme for the escape time — ``magma``, ``inferno``, ``turbo``, ...
        gain: colour spread; higher pushes more of the palette into the near-boundary band.
        inset_color: colour of the set itself (the never-escaping interior).
        aspect: width-to-height ratio of the initial view.
        label: legend label.

    Returns:
        The :class:`~glplot.core.layers.FractalLayer`.

    Limit: world coordinates are float32, so past roughly a 30 000x zoom neighbouring
    pixels stop separating and the image pixelates. Deeper needs double-precision emulation
    in the shader, which is a feature of its own.

    Examples:
        >>> gplt.mandelbrot()
        >>> gplt.show()                       # then scroll to zoom, it refines live
        >>> gplt.mandelbrot(center=(-0.743, 0.1318), span=0.02, cmap="turbo")
    """
    cx, cy = float(center[0]), float(center[1])
    half_w = float(span) * float(aspect)
    extent = (cx - half_w, cx + half_w, cy - float(span), cy + float(span))
    return _add_fractal(
        "mandelbrot",
        extent=extent,
        julia_c=(0.0, 0.0),
        max_iter=max_iter,
        cmap=cmap,
        gain=gain,
        inset_color=inset_color,
        label=label,
    )


def julia(
    c: Tuple[float, float] = (-0.8, 0.156),
    span: float = 1.6,
    *,
    center: Tuple[float, float] = (0.0, 0.0),
    max_iter: int = 200,
    cmap: str = "turbo",
    gain: float = 1.0,
    inset_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    aspect: float = 1.35,
    label: Optional[str] = None,
) -> FractalLayer:
    """Plot the Julia set for a parameter ``c``, live on the GPU. See :func:`mandelbrot`.

    The Julia set shares the Mandelbrot recursion ``z -> z^2 + c`` but fixes ``c`` and
    varies the starting ``z`` per pixel — so every point ``c`` of the Mandelbrot set has its
    own Julia set, and animating ``c`` along a path sweeps through them. It zooms and
    refines identically.

    Examples:
        >>> gplt.julia((-0.8, 0.156))
        >>> gplt.julia((0.285, 0.01), cmap="magma")
    """
    cx, cy = float(center[0]), float(center[1])
    half_w = float(span) * float(aspect)
    extent = (cx - half_w, cx + half_w, cy - float(span), cy + float(span))
    return _add_fractal(
        "julia",
        extent=extent,
        julia_c=(float(c[0]), float(c[1])),
        max_iter=max_iter,
        cmap=cmap,
        gain=gain,
        inset_color=inset_color,
        label=label,
    )


def imshow(
    X: ArrayLike,
    cmap: str = "viridis",
    norm: Optional[Any] = None,
    aspect: Optional[str] = None,
    interpolation: Optional[str] = None,
    origin: str = "upper",
    extent: Optional[Tuple[float, float, float, float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: Optional[float] = None,
    label: Optional[str] = None,
    *,
    filternorm: Optional[bool] = None,
    filterrad: Optional[float] = None,
    resample: Optional[bool] = None,
    interpolation_stage: Optional[str] = None,
    url: Optional[str] = None,
    data: Optional[Any] = None,
    **kwargs: Any,
) -> BaseLayer:
    """Display a 2D array as an image with colormap, or an RGB(A) array as a true-color image.

    A 2D matrix is colored by mapping values through a colormap; a 3D array whose last
    axis is 3 or 4 (e.g. from :func:`imread`) is treated as an RGB/RGBA image and drawn
    with its own per-pixel colour instead -- ``cmap``/``vmin``/``vmax``/``norm`` are
    silently unused in that case, matching matplotlib. Both are drawn as one point sprite
    per cell.

    Args:
        X (array-like): 2D data matrix, shape (M, N); or an RGB(A) image, shape
            (M, N, 3) or (M, N, 4). Integer image arrays are assumed 0-255 (as
            matplotlib's own imread() returns for non-PNG formats); float image arrays
            are assumed already normalized to 0-1 (as PNGs come back from imread()).
        cmap (str, optional): Colormap name ('viridis', 'plasma', 'cool',
            'hot', 'gray', etc.). Defaults to 'viridis'.
        norm (Normalize or str, optional): How the values map onto the colormap --
            a `matplotlib.colors.Normalize` or a scale name such as ``'log'``. Replaces
            the linear ``vmin``..``vmax`` ramp, and cannot be combined with them.
        aspect (str, optional): ``'equal'`` (the default, matching matplotlib) gives both
            axes the same world units per pixel, so a square matrix looks square;
            ``'auto'`` lets it stretch to fill the viewport instead.
        interpolation (str, optional): Accepted for matplotlib parity. GLPlot draws the
            image as one point sprite per cell, so the resampling filter belongs to the
            GPU and cannot be chosen per call. Ignored.
        origin (str, optional): 'upper' places origin at top-left (image coords),
            'lower' at bottom-left (standard math coords). Defaults to 'upper'.
        extent (tuple, optional): (left, right, bottom, top) in data coordinates.
            If None, uses matrix indices as coordinates. Defaults to None.
        vmin (float, optional): Minimum value for colormap normalization.
            If None, uses data minimum. Defaults to None.
        vmax (float, optional): Maximum value for colormap normalization.
            If None, uses data maximum. Defaults to None.
        alpha (float, optional): Transparency (0.0-1.0). Defaults to 1.0.
        label (str, optional): Legend label. Defaults to None.
        filternorm, filterrad, resample, interpolation_stage, url (optional): Accepted for
            matplotlib parity. They tune matplotlib's own image resampler and hyperlink
            metadata, neither of which GLPlot's GPU path has. Ignored.
        data (indexable, optional): If given, ``X`` may be a key into it.
        **kwargs: Additional keyword arguments including:
            s (float): Point size. Auto-computed if not provided.

    Returns:
        Layer: The image layer added to plot.

    Examples:
        Display a random matrix:

        >>> import numpy as np
        >>> data = np.random.rand(50, 50)
        >>> gplt.imshow(data)
        >>> gplt.show()

        With custom extent and colormap:

        >>> x = np.linspace(-1, 1, 100)
        >>> y = np.linspace(-1, 1, 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = X**2 + Y**2
        >>> gplt.imshow(Z, extent=[-1, 1, -1, 1], cmap='hot', origin='lower')

        With normalization:

        >>> gplt.imshow(data, vmin=0.2, vmax=0.8, cmap='plasma')
    """
    (X,) = _resolve_data_args("imshow", data, X)
    _warn_unsupported(
        "imshow",
        {
            "interpolation": interpolation,
            "filternorm": filternorm,
            "filterrad": filterrad,
            "resample": resample,
            "interpolation_stage": interpolation_stage,
            "url": url,
        },
        {
            "interpolation": "has no effect: GLPlot draws the image as one point sprite per "
            "cell, so the resampling filter is the GPU's and cannot be chosen per call",
            "filternorm": "has no effect: it tunes matplotlib's own image resampler, which "
            "GLPlot does not use",
            "filterrad": "has no effect: it tunes matplotlib's own image resampler, which "
            "GLPlot does not use",
            "resample": "has no effect: the image is rasterised by the GPU at draw time, "
            "so there is no CPU resampling pass to switch off",
            "interpolation_stage": "has no effect: GLPlot colour-maps on the CPU and "
            "interpolates on the GPU, so the two stages are not interchangeable",
            "url": "has no effect: GLPlot renders to a GPU surface, which carries no "
            "clickable hyperlink layer",
        },
    )
    # A true-color image (RGB/RGBA, e.g. from imread()) carries its own per-pixel colour
    # and skips the scalar colormap path entirely -- matplotlib does the same, silently
    # ignoring cmap/vmin/vmax/norm for image data rather than raising on them.
    array = np.asarray(X)
    is_rgb = array.ndim == 3 and array.shape[-1] in (3, 4)
    if is_rgb:
        matrix = array
        rows, cols = matrix.shape[:2]
    else:
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
    if is_rgb:
        # `matrix.reshape(-1, channels)` walks the same row-major order as `matrix.ravel()`
        # in the scalar branch, so it lines up with `xx.ravel()`/`yy.ravel()` the same way.
        rgb = matrix.astype(np.float32)
        if np.issubdtype(matrix.dtype, np.integer):
            rgb = rgb / 255.0  # matplotlib's imread(): uint8 0-255 for non-PNG formats
        rgb = np.clip(rgb, 0.0, 1.0)
        if rgb.shape[-1] == 3:
            opaque = np.ones(rgb.shape[:2] + (1,), dtype=np.float32)
            colors = np.concatenate([rgb, opaque], axis=-1).reshape(-1, 4)
        else:
            colors = rgb.reshape(-1, 4).copy()
    else:
        colors = _colormap_values(matrix.ravel(), cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    if alpha is not None:
        colors[:, 3] *= float(alpha)
    size = kwargs.pop("s", max(1.0, 650.0 / max(rows, cols)))
    plot_obj = _get_or_create_plot()
    plot_obj.add_scatter(xx.ravel(), yy.ravel(), colors, float(size), label=label)
    layer = plot_obj.scene.layers[-1]
    # 'equal' is the matplotlib default for an image -- what stops a square matrix from
    # being drawn as a rectangle -- so unset (None) resolves to it here too; 'auto' is the
    # explicit opt-out (`specgram` relies on it to leave its time axis alone). Stored on the
    # layer, not just applied to the live camera below, because the headless preview export
    # rebuilds a fresh matplotlib figure from layer metadata and used to hardcode
    # `aspect="auto"` there regardless of what was actually requested.
    if aspect in (None, "equal", "image"):
        resolved_aspect = "equal"
    elif aspect == "auto":
        resolved_aspect = "auto"
    else:
        raise ValueError(f"imshow(): unknown aspect {aspect!r}; expected 'equal' or 'auto'")
    layer.metadata.update(
        {
            "artist": "imshow",
            "matrix": matrix,
            "extent": (xmin, xmax, ymin, ymax),
            "origin": origin,
            "cmap": cmap,
            "norm": norm,
            "vmin": vmin,
            "vmax": vmax,
            "aspect": resolved_aspect,
        }
    )
    if resolved_aspect == "equal":
        _apply_equal_aspect(plot_obj, square=False)
    _set_dirty(plot_obj)
    return layer


def matshow(A, fignum: Optional[Union[int, str, bool]] = None, **kwargs):
    """Display a matrix as an image, with the origin at the top left.

    Args:
        A (array-like): The matrix. Shape (rows, cols).
        fignum (int, str or bool, optional): Where to draw. ``None`` opens a new figure
            (matplotlib's default for ``matshow``), ``False`` or ``0`` draws into the
            current one, and anything else selects the figure of that name or number.
        **kwargs: As :func:`imshow`.

    Returns:
        Layer: The image layer.
    """
    if fignum not in (False, 0):
        # matplotlib's matshow opens its own figure unless told otherwise -- unlike
        # imshow, which always draws into the current axes.
        figure(fignum)
    return imshow(A, origin=kwargs.pop("origin", "upper"), **kwargs)


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
    zdir: str = "z",
    offset: Optional[float] = None,
    **kwargs,
):
    """Draw contour lines for a 2D field.

    Renders contour lines (level curves) of a 2D scalar field. Useful for
    visualizing 2D functions, topography, or field data. Supports both
    regular and irregular meshes with custom level specifications.

    Args:
        X (array-like): 2D x-coordinates array, or values if Z=None.
            Shape (M, N).
        Y (array-like, optional): 2D y-coordinates. Required if Z provided.
            Shape (M, N). Defaults to None.
        Z (array-like, optional): 2D values for contours. Required if X and Y
            provided separately. Shape (M, N). Defaults to None.
        levels (int or array-like, optional): Number of contour levels or
            explicit level values. Defaults to 10.
        colors (str or tuple, optional): Color specification (currently
            stored in metadata). Defaults to None.
        cmap (str, optional): Colormap for level colors. Defaults to 'viridis'.
        linewidths (float, optional): Width of contour lines. Defaults to 1.0.
        label (str, optional): Legend label. Defaults to None.
        zdir (str, optional): On a 3D axes only (matplotlib's ``Axes3D.contour``
            parameter): which axis the contour plane is normal to. Only ``'z'``
            (the default and overwhelmingly common case -- a scalar field over an
            x/y plane) is implemented; other values raise rather than silently
            drawing the wrong axes swapped.
        offset (float, optional): On a 3D axes only. If given, every contour line is
            flattened onto one plane at this height (matplotlib's "floor
            projection", typically used beneath a 3D surface). If None (the
            default), each level is drawn at its own height -- ``z = level`` --
            the true 3D contour matplotlib draws by default.
        **kwargs: Additional keyword arguments.

    Returns:
        Layer: The contour layer added to plot (2D axes), or a list of 3D line
        layers, one per level (3D axes).

    Examples:
        Simple contour plot:

        >>> Z = np.random.rand(30, 30)
        >>> gplt.contour(Z, levels=10)
        >>> gplt.show()

        With custom mesh and levels:

        >>> x = np.linspace(-3, 3, 50)
        >>> y = np.linspace(-3, 3, 50)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = np.exp(-(X**2 + Y**2))
        >>> gplt.contour(X, Y, Z, levels=[0.1, 0.3, 0.5, 0.7, 0.9])

        Projected onto a 3D axes, beneath a surface:

        >>> ax = gplt.axes(projection="3d")
        >>> ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7)
        >>> ax.contour(X, Y, Z, zdir="z", offset=Z.min(), cmap="viridis")
    """
    data = kwargs.pop("data", None)
    if data is not None:
        X, Y, Z = _resolve_data_args("contour", data, X, Y, Z)

    if Z is None:
        matrix = _as_float_array(X, ndim=2, name="Z")
        yy, xx = np.indices(matrix.shape, dtype=np.float32)
    else:
        xx = _as_float_array(X, name="X")
        yy = _as_float_array(Y, name="Y")
        matrix = _as_float_array(Z, ndim=2, name="Z")

    plot_obj = _get_or_create_plot()
    if plot_obj.is_3d_scene():
        # A 2D `imshow()` placeholder (the non-3D path below) has no z-coordinate at
        # all -- calling it here used to silently draw something flat instead of
        # raising, on a 3D axes, looking like 3D support that categorically didn't
        # exist. Real geometry instead: the same segments the 2D path computes,
        # lifted into 3D (either to each level's own height, or onto one constant
        # `offset` plane -- matplotlib's own two `Axes3D.contour` modes), and drawn
        # through the same `_add_3d_layer` primitive `plot3d()` uses, which already
        # exports correctly in both the live view and the headless preview -- unlike
        # the 2D path, there's no separate "reconstruct through matplotlib" step to
        # keep in sync, so no placeholder/metadata dance is needed here at all.
        if zdir != "z":
            raise ValueError(
                f"contour(): zdir={zdir!r} is not supported on a 3D axes; only the "
                "default 'z' (a field over the x/y plane) is implemented"
            )
        segments, resolved_levels = _grid_contour_segments(xx, yy, matrix, levels, filled=False)
        lo, hi = (min(resolved_levels), max(resolved_levels)) if resolved_levels else (0.0, 1.0)
        from matplotlib import colormaps

        layers = []
        for level, seg in segments:
            if colors is not None:
                line_color = colors
            else:
                t = 0.0 if hi == lo else (level - lo) / (hi - lo)
                line_color = tuple(
                    float(c) for c in colormaps.get_cmap(_resolve_cmap(cmap, "viridis"))(t)
                )
            z_value = float(level if offset is None else offset)
            verts = np.column_stack([seg[:, 0], seg[:, 1], np.full(len(seg), z_value)]).astype(
                np.float32
            )
            segs3d = np.empty(((len(verts) - 1) * 2, 3), dtype=np.float32)
            segs3d[0::2] = verts[:-1]
            segs3d[1::2] = verts[1:]
            layer = _add_3d_layer(
                segs3d,
                primitive="lines",
                layer_type="wireframe3d",
                label=label if not layers else None,
                color=line_color,
                metadata={"artist": "contour3d", "level": level},
            )
            layer.style.line_width = float(linewidths)
            layers.append(layer)
        return layers

    extent = (float(np.min(xx)), float(np.max(xx)), float(np.min(yy)), float(np.max(yy)))
    # aspect="auto" explicitly: imshow()'s own default is now 'equal' (matching
    # matplotlib's), but matplotlib's contour() itself defaults to 'auto', and a contour
    # domain is rarely square -- inheriting 'equal' here would squash it.
    layer = imshow(
        np.zeros((2, 2), dtype=np.float32), extent=extent, alpha=0.0, aspect="auto", label=label
    )
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
    # Real, visible lines for the live GL view -- the placeholder above is fully
    # transparent (alpha=0.0) and exists only so the headless preview's "artist ==
    # contour" branch has geometry to reconstruct through a fresh `ax.contour()` call;
    # without this, `show()` on the exact same script drew nothing where the contour
    # should be. Tagged "contour_line" so that headless path skips these and draws its
    # own copy instead of both (see render_preview's own skip for the same tag).
    segments, resolved_levels = _grid_contour_segments(xx, yy, matrix, levels, filled=False)
    lo, hi = (min(resolved_levels), max(resolved_levels)) if resolved_levels else (0.0, 1.0)
    plot_obj = _get_or_create_plot()
    for level, seg in segments:
        if colors is not None:
            line_color = colors
        else:
            from matplotlib import colormaps

            t = 0.0 if hi == lo else (level - lo) / (hi - lo)
            line_color = tuple(
                float(c) for c in colormaps.get_cmap(_resolve_cmap(cmap, "viridis"))(t)
            )
        plot_obj.add_line_strip(seg[:, 0], seg[:, 1], color=line_color, width=float(linewidths))
        plot_obj.scene.layers[-1].metadata.update({"artist": "contour_line", "level": level})
    _set_dirty(plot_obj)
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
    """Draw filled contours for a 2D field.

    Renders filled contour regions (colored bands between level curves) of
    a 2D scalar field. Similar to imshow() but computed from explicit mesh
    coordinates. Useful for publication-quality visualizations of field data.

    Args:
        X (array-like): 2D x-coordinates array, or values if Z=None.
            Shape (M, N).
        Y (array-like, optional): 2D y-coordinates. Required if Z provided.
            Shape (M, N). Defaults to None.
        Z (array-like, optional): 2D values for contours. Required if X and Y
            provided separately. Shape (M, N). Defaults to None.
        levels (int or array-like, optional): Number of contour levels or
            explicit level values. Defaults to 10.
        cmap (str, optional): Colormap for level colors. Defaults to 'viridis'.
        alpha (float, optional): Transparency. Defaults to 1.0.
        label (str, optional): Legend label. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        Layer: The filled contour layer added to plot.

    Examples:
        Simple filled contour:

        >>> Z = np.random.rand(30, 30)
        >>> gplt.contourf(Z, levels=10, cmap='viridis')
        >>> gplt.show()

        With function values:

        >>> x = np.linspace(-2, 2, 40)
        >>> y = np.linspace(-2, 2, 40)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = X**2 - Y**2
        >>> gplt.contourf(X, Y, Z, levels=15, cmap='RdBu', alpha=0.8)
    """
    data = kwargs.pop("data", None)
    if data is not None:
        X, Y, Z = _resolve_data_args("contourf", data, X, Y, Z)

    if Z is None:
        matrix = _as_float_array(X, ndim=2, name="Z")
        yy, xx = np.indices(matrix.shape, dtype=np.float32)
    else:
        xx = _as_float_array(X, name="X")
        yy = _as_float_array(Y, name="Y")
        matrix = _as_float_array(Z, ndim=2, name="Z")

    if _get_or_create_plot().is_3d_scene():
        # Unlike contour() (fixed to draw real 3D line geometry -- see its own docstring),
        # there is no fallback here: the 2D path below is `imshow()`, which has no
        # z-coordinate at all, so calling it on a 3D axes used to silently draw something
        # flat and wrong rather than erroring -- looking like partial 3D support that
        # categorically did not exist. Filled 3D contour bands need extracting closed
        # polygons between levels and lifting them into real 3D mesh geometry (matplotlib's
        # `tricontourf` equivalent for a triangulated mesh already does this for 2D; the 3D
        # case is unimplemented), so this raises rather than degrading into more silently
        # wrong output.
        raise NotImplementedError(
            "contourf() on a 3D axes is not supported: filled bands would need extracting "
            "closed polygons between levels and lifting them into 3D mesh geometry, which "
            "GLPlot does not implement yet. Use contour() for 3D line contours (real "
            "geometry, in both the live view and the headless export), or call contourf() "
            "on a 2D axes instead."
        )

    extent = (float(np.min(xx)), float(np.max(xx)), float(np.min(yy)), float(np.max(yy)))
    # aspect="auto" explicitly, for the same reason contour() passes it: imshow()'s
    # default is now 'equal', but a contourf domain is rarely square.
    layer = imshow(matrix, extent=extent, cmap=cmap, alpha=alpha, aspect="auto", label=label)
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
    # The live view shows this as a continuous colour ramp of the raw field (what the
    # imshow() above draws), not matplotlib's discrete filled bands between levels --
    # `levels` only takes effect in the headless PNG export, where this same layer's
    # metadata is reconstructed through a real `ax.contourf(..., levels=levels)` call.
    # :func:`tricontourf` gets real discrete live bands because it already extracts them
    # per-triangle; doing the same for a regular grid is unimplemented.
    _warn_unsupported_call(
        "contourf",
        "shows a continuous colour ramp of the raw field live, not matplotlib's discrete "
        "bands between levels -- the headless PNG export (savefig() before any show()) "
        "reconstructs the real discrete contourf through matplotlib",
    )
    return layer


def _grid_contour_segments(xx, yy, z, levels, filled: bool):
    """Contour ``z`` over a regular mesh, via matplotlib, as ``(level, segments)`` pairs.

    The non-triangulated counterpart of :func:`_tri_contour_segments` -- same reasoning
    (a real contour algorithm, via a throwaway Agg figure, instead of re-deriving marching
    squares), same detached-figure trick, just ``ax.contour``/``ax.contourf`` on a meshgrid
    directly rather than through a ``Triangulation``.
    """
    from matplotlib.figure import Figure

    fig = Figure()
    ax = fig.subplots()
    cs = (ax.contourf if filled else ax.contour)(xx, yy, z, levels=levels)
    out = []
    for level, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            if len(seg) >= 2:
                out.append((float(level), np.asarray(seg, dtype=np.float32)))
    return out, list(cs.levels)


def _tri_contour_segments(x, y, tris, z, levels, filled: bool):
    """Contour ``z`` over a triangulation, via matplotlib, as ``(level, segments)`` pairs.

    matplotlib owns a correct triangular contour generator; reusing it through a throwaway
    Agg figure gets the exact geometry without a display, rather than re-deriving marching
    triangles here and getting the saddle cases subtly wrong. The segments come back in
    data coordinates, ready to draw as polylines.
    """
    from matplotlib.figure import Figure
    from matplotlib.tri import Triangulation

    # A detached Agg figure: it never reaches a screen, so this works headless and in the
    # test suite, and it is discarded the moment its segments are read out.
    fig = Figure()
    ax = fig.subplots()
    tri = Triangulation(x, y, tris)
    cs = (ax.tricontourf if filled else ax.tricontour)(tri, z, levels=levels)
    out = []
    for level, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            if len(seg) >= 2:
                out.append((float(level), np.asarray(seg, dtype=np.float32)))
    return out, list(cs.levels)


def tricontour(*args: Any, **kwargs: Any):
    """Contour lines of a field defined on an unstructured triangular mesh.

    Args:
        *args: ``x, y, z`` (Delaunay), ``x, y, triangles, z``, or a
            ``Triangulation`` followed by ``z``. ``z`` is one value per point.
        levels (int or array-like, optional): How many contour lines, or their
            explicit values. Defaults to matplotlib's automatic choice.
        colors (str, optional): A single line colour. Overrides ``cmap``.
        cmap (str, optional): Colour the lines by level. Defaults to viridis.
        linewidths (float, optional): Line width.
        label (str, optional): Legend label.

    Returns:
        list: One polyline layer per contour segment.

    Note:
        Unlike :func:`contour`, these are drawn live as real polylines rather than
        deferred to the exported PNG.

    Examples:
        >>> gplt.tricontour(x, y, z, levels=8)
    """
    levels = kwargs.pop("levels", None)
    colors = kwargs.pop("colors", None)
    cmap = kwargs.pop("cmap", None)
    linewidths = kwargs.pop("linewidths", 1.0)
    label = kwargs.pop("label", None)

    x, y, tris, rest = _triangulation("tricontour", args)
    if not rest:
        raise TypeError("tricontour() needs z -- one value per point")
    z = _as_float_array(rest[0], ndim=1, name="z")
    if len(z) != len(x):
        raise ValueError(f"z must have one value per point ({len(x)}), got {len(z)}")

    lv = 10 if levels is None else levels
    segments, resolved_levels = _tri_contour_segments(x, y, tris, z, lv, filled=False)
    lo, hi = (min(resolved_levels), max(resolved_levels)) if resolved_levels else (0.0, 1.0)

    plot_obj = _get_or_create_plot()
    layers = []
    for i, (level, seg) in enumerate(segments):
        if colors is not None:
            line_color = colors
        else:
            from matplotlib import colormaps

            t = 0.0 if hi == lo else (level - lo) / (hi - lo)
            line_color = tuple(
                float(c) for c in colormaps.get_cmap(_resolve_cmap(cmap, "viridis"))(t)
            )
        plot_obj.add_line_strip(
            seg[:, 0],
            seg[:, 1],
            color=line_color,
            width=float(linewidths),
            label=label if i == 0 else None,
        )
        layer = plot_obj.scene.layers[-1]
        layer.metadata.update({"artist": "tricontour", "level": level})
        layers.append(layer)

    _set_dirty(plot_obj)
    return layers


def tricontourf(*args: Any, **kwargs: Any):
    """Filled contours of a field on an unstructured triangular mesh.

    The filled counterpart of :func:`tricontour`: the bands between contour
    levels, each filled with the colour that level maps to.

    Args:
        *args: ``x, y, z``, ``x, y, triangles, z``, or a ``Triangulation`` then
            ``z``.
        levels (int or array-like, optional): Band count or edges.
        cmap (str, optional): Colormap. Defaults to viridis.
        alpha (float, optional): Transparency.
        label (str, optional): Legend label.

    Returns:
        list: One filled-polygon patch layer per band segment.

    Examples:
        >>> gplt.tricontourf(x, y, z, levels=10)
    """
    levels = kwargs.pop("levels", None)
    cmap = kwargs.pop("cmap", None)
    alpha = kwargs.pop("alpha", None)
    label = kwargs.pop("label", None)

    x, y, tris, rest = _triangulation("tricontourf", args)
    if not rest:
        raise TypeError("tricontourf() needs z -- one value per point")
    z = _as_float_array(rest[0], ndim=1, name="z")
    if len(z) != len(x):
        raise ValueError(f"z must have one value per point ({len(x)}), got {len(z)}")

    lv = 10 if levels is None else levels
    segments, resolved_levels = _tri_contour_segments(x, y, tris, z, lv, filled=True)
    lo, hi = (min(resolved_levels), max(resolved_levels)) if resolved_levels else (0.0, 1.0)

    from matplotlib import colormaps

    plot_obj = _get_or_create_plot()
    layers = []
    for i, (level, seg) in enumerate(segments):
        if len(seg) < 3:
            continue
        # Each filled band comes back as a closed ring; a fan from its first vertex fills
        # it. The rings matplotlib emits for a filled contour are simple (no self-crossing),
        # so a fan is exact here even though `fill()` warns about the concave case.
        verts = seg.astype(np.float32)
        fan = np.arange(1, len(verts) - 1)
        indices = np.column_stack([np.zeros_like(fan), fan, fan + 1]).ravel().astype(np.uint32)
        t = 0.0 if hi == lo else (level - lo) / (hi - lo)
        rgba = list(colormaps.get_cmap(_resolve_cmap(cmap, "viridis"))(t))
        if alpha is not None:
            rgba[3] *= float(alpha)
        add_patch(
            verts,
            indices=indices,
            mode="triangles",
            face_color=tuple(rgba),
            edge_color=tuple(rgba),
            label=label if i == 0 else None,
        )
        layer = plot_obj.scene.layers[-1]
        layer.metadata.update({"artist": "tricontourf", "level": level})
        layers.append(layer)

    _set_dirty(plot_obj)
    return layers


def spy(
    Z: ArrayLike,
    *,
    markersize: Optional[float] = None,
    marker: Optional[str] = None,
    color: ColorLike = "k",
    precision: float = 0.0,
    origin: str = "upper",
    aspect: Optional[str] = "equal",
    label: Optional[str] = None,
    **kwargs: Any,
):
    """Plot the sparsity pattern of a 2-D array -- a marker at every nonzero.

    Args:
        Z (array-like): The 2-D array. Shape (M, N).
        markersize (float, optional): Marker size in pixels.
        marker (str, optional): Accepted for matplotlib parity; GLPlot draws
            round points.
        color (str or tuple, optional): Marker colour. Defaults to black.
        precision (float, optional): Treat ``|Z| <= precision`` as zero. Defaults
            to 0.
        origin (str, optional): 'upper' (default) puts row 0 at the top, as
            matplotlib does for a matrix; 'lower' puts it at the bottom.
        aspect (str, optional): ``'equal'`` (matplotlib's default here) keeps the cells
            square, so a square matrix looks square. ``'auto'`` lets it stretch to the
            viewport.
        label (str, optional): Legend label.

    Returns:
        Layer: The scatter layer of nonzero positions.

    Examples:
        >>> gplt.spy(sparse_matrix)
    """
    matrix = _as_float_array(Z, ndim=2, name="Z")
    rows, cols = np.nonzero(np.abs(matrix) > float(precision))
    if origin not in ("upper", "lower"):
        raise ValueError(f"unsupported origin: {origin!r}. Expected 'upper' or 'lower'.")
    # A matrix reads with row 0 at the top, so y runs downward by default -- the same flip
    # `imshow(origin='upper')` applies, kept here so spy and imshow agree on orientation.
    ys = rows if origin == "lower" else (matrix.shape[0] - 1 - rows)
    layer = scatter(
        cols.astype(np.float32),
        ys.astype(np.float32),
        color=color,
        s=markersize if markersize is not None else 4.0,
        marker=marker,
        label=label,
    )
    layer.metadata["artist"] = "spy"
    if aspect in ("equal", "image"):
        _apply_equal_aspect(_get_or_create_plot(), square=False)
    elif aspect not in (None, "auto"):
        raise ValueError(f"spy(): unknown aspect {aspect!r}; expected 'equal' or 'auto'")
    return layer


def barbs(*args: Any, **kwargs: Any) -> list:
    """Draw wind barbs -- direction glyphs whose length shows a field's magnitude.

    Takes the same ``x, y, u, v`` (or ``u, v``) arguments as :func:`quiver`.

    Args:
        *args: ``u, v`` or ``x, y, u, v``.
        length (float, optional): Glyph length in data units.
        color (str or tuple, optional): Glyph colour.
        label (str, optional): Legend label.

    Returns:
        list: The layers drawn.

    Note:
        A true wind barb encodes speed with half-barbs, full barbs and pennants.
        GLPlot has no glyph primitive for those, so each barb is drawn as a plain
        shaft pointing downwind, scaled by speed -- the direction and relative
        magnitude are faithful; the meteorological flag ticks are not. A warning
        says so once.

    Examples:
        >>> gplt.barbs(x, y, u, v)
    """
    length = kwargs.pop("length", None)
    color = kwargs.pop("color", "k")
    label = kwargs.pop("label", None)
    if len(args) == 2:
        u = _as_float_array(args[0], name="u")
        v = _as_float_array(args[1], name="v")
        x, y = (
            np.meshgrid(np.arange(u.shape[-1]), np.arange(u.shape[0]))
            if u.ndim == 2
            else (np.arange(len(u)), np.zeros(len(u)))
        )
        x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
        u, v = u.ravel(), v.ravel()
    elif len(args) == 4:
        x = _as_float_array(args[0], name="x").ravel()
        y = _as_float_array(args[1], name="y").ravel()
        u = _as_float_array(args[2], name="u").ravel()
        v = _as_float_array(args[3], name="v").ravel()
    else:
        raise TypeError(f"barbs() takes u, v or x, y, u, v -- got {len(args)} positional args")

    _warn_unsupported(
        "barbs",
        {"barbs": True},
        {
            "barbs": "draws each barb as a plain shaft, not the half-barb/pennant glyphs: "
            "GLPlot has no primitive for those. Direction and relative length are faithful"
        },
    )
    speed = np.hypot(u, v)
    scale = float(length) if length is not None else 1.0
    mag = np.where(speed > 0, speed, 1.0)
    # A shaft from each point, downwind, its length proportional to speed.
    x2 = x + scale * u / mag * speed / (speed.max() or 1.0)
    y2 = y + scale * v / mag * speed / (speed.max() or 1.0)
    segs_x = np.empty(3 * len(x))
    segs_y = np.empty(3 * len(y))
    segs_x[0::3], segs_x[1::3], segs_x[2::3] = x, x2, x2  # retrace to lift the pen
    segs_y[0::3], segs_y[1::3], segs_y[2::3] = y, y2, y2
    artists = plot(segs_x, segs_y, color=color, linewidth=1.0, label=label)
    for artist in artists:
        artist.metadata["artist"] = "barbs"
    return artists


def quiverkey(Q: Any, X: float, Y: float, U: float, label: str, **kwargs: Any):
    """Draw a reference arrow and label for a :func:`quiver` field.

    Args:
        Q: The value :func:`quiver` returned. Accepted for matplotlib parity;
            only its colour is read.
        X, Y (float): Where to place the key, in **data** coordinates. (matplotlib
            uses axes fraction; GLPlot has no separate coordinate system here, so
            these are data coordinates.)
        U (float): The reference magnitude the arrow represents.
        label (str): The text beside the arrow.
        color (str or tuple, optional): Arrow and text colour.

    Returns:
        list: The arrow and text layers.

    Examples:
        >>> q = gplt.quiver(x, y, u, v)
        >>> gplt.quiverkey(q, 0.9, 0.9, 2, '2 m/s')
    """
    color = kwargs.pop("color", "k")
    _warn_unsupported(
        "quiverkey",
        {"placement": None if (0.0 <= X <= 1.0 and 0.0 <= Y <= 1.0) else True},
        {
            "placement": "places the key at data coordinates, not axes fraction: GLPlot has "
            "no axes-fraction coordinate system"
        },
    )
    arrow = quiver([float(X)], [float(Y)], [float(U)], [0.0], color=color)
    txt = text(float(X), float(Y) - 0.05 * abs(float(U) or 1.0), str(label), color=color)
    return [arrow, txt]


def clabel(CS: Any = None, levels: Optional[Any] = None, **kwargs: Any) -> list:
    """Label contour lines. Accepted for matplotlib parity; draws nothing.

    matplotlib places numeric labels along contour lines, breaking the line to
    seat the text. GLPlot draws contours through the savefig path (see
    :func:`contour`), where matplotlib's own ``clabel`` is available -- so the
    live view has no contour geometry to label.

    Returns:
        list: Always empty.
    """
    _get_or_create_plot()
    _warn_unsupported(
        "clabel",
        {"clabel": True},
        {
            "clabel": "draws nothing: contour labelling needs the contour line geometry, "
            "which GLPlot computes only on the savefig path"
        },
    )
    return []


def figimage(X: ArrayLike, xo: int = 0, yo: int = 0, **kwargs: Any):
    """Place an image on the figure. Drawn as an :func:`imshow` in data space.

    matplotlib's ``figimage`` blits an image at a pixel offset on the figure,
    below the axes. GLPlot has one viewport and no figure-vs-axes split, so this
    draws the array with :func:`imshow` instead -- visible, but positioned in data
    coordinates rather than figure pixels.

    Args:
        X (array-like): The image, 2-D or (M, N, 3/4).
        xo, yo (int, optional): Pixel offsets. Accepted for parity; ignored.

    Returns:
        Layer: The image layer.
    """
    cmap = kwargs.pop("cmap", None)
    _warn_unsupported(
        "figimage",
        {"offset": True if (xo or yo) else None},
        {
            "offset": "is ignored: GLPlot has no figure-pixel coordinate system, so the image "
            "is placed in data space by imshow"
        },
    )
    return imshow(X, cmap=_resolve_cmap(cmap, "viridis"))


def table(**kwargs: Any):
    """Accepted for matplotlib parity; draws nothing.

    matplotlib's ``table`` renders a grid of text cells anchored to the axes.
    GLPlot's text is drawn at data coordinates with no cell layout engine, so a
    faithful table is out of reach; this is a no-op rather than a broken grid.

    Returns:
        None
    """
    _get_or_create_plot()
    _warn_unsupported(
        "table",
        {"table": True},
        {"table": "draws nothing: GLPlot has no cell-layout engine for a text grid"},
    )
    return None


def polar(*args: Any, **kwargs: Any) -> list:
    """Plot on polar axes. GLPlot has none, so ``(theta, r)`` is converted to x/y.

    matplotlib's ``polar`` switches the axes to a polar projection. GLPlot's
    projection is Cartesian, so this converts the polar data to Cartesian
    (``x = r cos theta``, ``y = r sin theta``) and plots that -- the curve is the
    right shape, but there is no polar grid behind it.

    Args:
        *args: ``theta, r`` pairs, as :func:`plot` takes ``x, y``.

    Returns:
        list: The layers drawn.

    Examples:
        >>> gplt.polar(theta, r)
    """
    _warn_unsupported(
        "polar",
        {"polar": True},
        {
            "polar": "has no polar grid: GLPlot's projection is Cartesian, so (theta, r) is "
            "converted to x/y and plotted. The curve is right; the grid behind it is not polar"
        },
    )
    groups = _parse_plot_groups(args)
    converted = []
    for theta, r, fmt in groups:
        if r is None:
            raise ValueError("polar() needs theta and r")
        th = _as_float_array(theta, ndim=1, name="theta")
        rr = _as_float_array(r, ndim=1, name="r")
        converted.extend([rr * np.cos(th), rr * np.sin(th)])
        if fmt is not None:
            converted.append(fmt)
    return plot(*converted, **kwargs)


def plot_date(x: ArrayLike, y: ArrayLike, fmt: str = "o", *args: Any, **kwargs: Any) -> list:
    """Plot ``y`` against dates ``x``.

    matplotlib deprecated ``plot_date`` in favour of plain ``plot`` with date-aware
    axes. GLPlot has no date axis, so this treats ``x`` as the matplotlib date
    numbers they already are (days since the epoch) and forwards to :func:`plot`.
    The curve is correct; only the tick labels read as numbers rather than dates.

    Args:
        x (array-like): Dates as matplotlib date numbers.
        y (array-like): The values.
        fmt (str, optional): A plot format string. Defaults to 'o'.

    Returns:
        list: The layers :func:`plot` produced.

    Examples:
        >>> import matplotlib.dates as mdates
        >>> gplt.plot_date(mdates.date2num(dates), values)
    """
    _warn_unsupported(
        "plot_date",
        {"plot_date": True},
        {
            "plot_date": "draws x as matplotlib date numbers, not calendar dates: GLPlot has "
            "no date axis, so the ticks read as numbers. The curve itself is correct"
        },
    )
    return plot(x, y, fmt, *args, **kwargs)


def _as_signal(x: ArrayLike, name: str = "x") -> np.ndarray:
    """A 1-D signal in float64, for the spectral and correlation estimators.

    Deliberately *not* `_as_float_array`, which casts to float32 because everything it
    feeds ends up in a GPU buffer. These functions return numbers the caller analyses
    rather than geometry GLPlot draws, and float32 shows up as a ~1e-6 disagreement with
    matplotlib on the same input -- small, but enough to fail an exact-parity comparison
    and to accumulate through a long FFT.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must have ndim=1, got {arr.ndim}")
    return np.ascontiguousarray(arr)


def _spectral_line(artist: str, freqs, values, color, label, kwargs) -> Optional[BaseLayer]:
    """Draw one spectral curve and tag it, the way every function in this family ends.

    Split out because the six spectral functions differ only in the transform they apply;
    sharing the tail keeps the ``metadata["artist"]`` tag and the kwarg forwarding from
    drifting apart between them.
    """
    layers = plot(freqs, values, color=color, label=label, **kwargs)
    for layer in layers:
        layer.metadata["artist"] = artist
    return layers[-1] if layers else None


def psd(
    x: ArrayLike,
    NFFT: Optional[int] = None,
    Fs: Optional[float] = None,
    Fc: Optional[int] = None,
    detrend: Any = None,
    window: Any = None,
    noverlap: Optional[int] = None,
    pad_to: Optional[int] = None,
    sides: Optional[str] = None,
    scale_by_freq: Optional[bool] = None,
    return_line: Optional[bool] = None,
    *,
    color: ColorLike = "C0",
    label: Optional[str] = None,
    data: Optional[Any] = None,
    **kwargs: Any,
):
    """Plot the power spectral density of ``x`` using Welch's average periodogram.

    The estimate itself is :func:`matplotlib.mlab.psd`, not a reimplementation, so
    ``detrend``, ``window``, ``pad_to``, ``sides`` and ``scale_by_freq`` mean exactly what
    they mean in matplotlib and the returned numbers agree to floating point. Only the
    drawing is GLPlot's.

    Args:
        x (array-like): The signal.
        NFFT (int, optional): Samples per segment. Defaults to 256.
        Fs (float, optional): Sampling frequency. Defaults to 2.
        Fc (int, optional): Centre frequency, added to the returned frequencies so a
            baseband spectrum can be labelled at its real carrier. Defaults to 0.
        detrend (str or callable, optional): ``'none'``, ``'mean'``, ``'linear'`` or a
            callable applied to each segment. Defaults to ``'none'``.
        window (callable or array-like, optional): Window applied to each segment.
            Defaults to a Hanning window.
        noverlap (int, optional): Samples of overlap between segments. Defaults to 0.
        pad_to (int, optional): Length the segment is zero-padded to before the FFT.
            Defaults to ``NFFT``.
        sides (str, optional): ``'default'``, ``'onesided'`` or ``'twosided'``.
        scale_by_freq (bool, optional): Scale the density by the sampling frequency, so
            the result reads in units per Hz. Defaults to True.
        return_line (bool, optional): Also return the drawn layer. Defaults to False.
        color (str or tuple, optional): Line colour.
        label (str, optional): Legend label.
        data (indexable, optional): If given, ``x`` may be a key into it.

    Returns:
        tuple: ``(Pxx, freqs)``, or ``(Pxx, freqs, layer)`` when ``return_line`` is True --
        matching matplotlib's arity, so ``Pxx, freqs = psd(x)`` unpacks as it does there.

    Examples:
        >>> gplt.psd(signal, NFFT=512, Fs=1000)
        >>> pxx, freqs, line = gplt.psd(signal, detrend="linear", return_line=True)
    """
    from matplotlib import mlab

    (x,) = _resolve_data_args("psd", data, x)
    sig = _as_signal(x, "x")
    pxx, freqs = mlab.psd(
        sig,
        NFFT=NFFT,
        Fs=Fs,
        detrend=detrend,
        window=window,
        noverlap=noverlap,
        pad_to=pad_to,
        sides=sides,
        scale_by_freq=scale_by_freq,
    )
    freqs = freqs + (0 if Fc is None else Fc)
    # matplotlib plots PSD in dB (10 log10), which is the only way the decades of a real
    # spectrum fit on a linear axis GLPlot can draw.
    layer = _spectral_line("psd", freqs, 10.0 * np.log10(pxx), color, label, kwargs)
    if return_line:
        return pxx, freqs, layer
    return pxx, freqs


def csd(
    x: ArrayLike,
    y: ArrayLike,
    NFFT: Optional[int] = None,
    Fs: Optional[float] = None,
    Fc: Optional[int] = None,
    detrend: Any = None,
    window: Any = None,
    noverlap: Optional[int] = None,
    pad_to: Optional[int] = None,
    sides: Optional[str] = None,
    scale_by_freq: Optional[bool] = None,
    return_line: Optional[bool] = None,
    *,
    color: ColorLike = "C0",
    label: Optional[str] = None,
    data: Optional[Any] = None,
    **kwargs: Any,
):
    """Plot the cross spectral density of ``x`` and ``y``.

    Computed by :func:`matplotlib.mlab.csd`; every estimator keyword means what it means
    in matplotlib. See :func:`psd` for the full description of each.

    Args:
        x, y (array-like): The two signals.
        NFFT, Fs, Fc, detrend, window, noverlap, pad_to, sides, scale_by_freq: As
            :func:`psd`.
        return_line (bool, optional): Also return the drawn layer. Defaults to False.
        color, label, data: As :func:`psd`.

    Returns:
        tuple: ``(Pxy, freqs)``, or ``(Pxy, freqs, layer)`` when ``return_line`` is True.
    """
    from matplotlib import mlab

    x, y = _resolve_data_args("csd", data, x, y)
    a = _as_signal(x, "x")
    b = _as_signal(y, "y")
    pxy, freqs = mlab.csd(
        a,
        b,
        NFFT=NFFT,
        Fs=Fs,
        detrend=detrend,
        window=window,
        noverlap=noverlap,
        pad_to=pad_to,
        sides=sides,
        scale_by_freq=scale_by_freq,
    )
    freqs = freqs + (0 if Fc is None else Fc)
    layer = _spectral_line("csd", freqs, 10.0 * np.log10(np.abs(pxy)), color, label, kwargs)
    if return_line:
        return pxy, freqs, layer
    return pxy, freqs


def cohere(
    x: ArrayLike,
    y: ArrayLike,
    NFFT: int = 256,
    Fs: float = 2.0,
    Fc: int = 0,
    detrend: Any = None,
    window: Any = None,
    noverlap: int = 0,
    pad_to: Optional[int] = None,
    sides: str = "default",
    scale_by_freq: Optional[bool] = None,
    *,
    color: ColorLike = "C0",
    label: Optional[str] = None,
    data: Optional[Any] = None,
    **kwargs: Any,
):
    """Plot the coherence between ``x`` and ``y`` -- correlation per frequency, 0..1.

    Computed by :func:`matplotlib.mlab.cohere`. See :func:`psd` for the estimator
    keywords.

    Returns:
        tuple: ``(Cxy, freqs)`` -- matching matplotlib's arity.
    """
    from matplotlib import mlab

    x, y = _resolve_data_args("cohere", data, x, y)
    a = _as_signal(x, "x")
    b = _as_signal(y, "y")
    cxy, freqs = mlab.cohere(
        a,
        b,
        NFFT=NFFT,
        Fs=Fs,
        detrend=mlab.detrend_none if detrend is None else detrend,
        window=mlab.window_hanning if window is None else window,
        noverlap=noverlap,
        pad_to=pad_to,
        sides=sides,
        scale_by_freq=scale_by_freq,
    )
    freqs = freqs + Fc
    _spectral_line("cohere", freqs, cxy, color, label, kwargs)
    return cxy, freqs


def _spectrum(func: str, x, Fs, Fc, window, pad_to, sides, data):
    """Shared FFT for the three ``*_spectrum`` functions: magnitude, phase, angle.

    Dispatches to the matching :mod:`matplotlib.mlab` estimator by name, so windowing and
    the one-sided/two-sided fold are matplotlib's rather than a second implementation that
    would have to be kept in step with it.
    """
    from matplotlib import mlab

    (x,) = _resolve_data_args(func, data, x)
    sig = _as_signal(x, "x")
    spec, freqs = getattr(mlab, func)(x=sig, Fs=Fs, window=window, pad_to=pad_to, sides=sides)
    return spec, freqs + (0 if Fc is None else Fc)


def magnitude_spectrum(
    x: ArrayLike,
    Fs: Optional[float] = None,
    Fc: Optional[int] = None,
    window: Any = None,
    pad_to: Optional[int] = None,
    sides: Optional[str] = None,
    scale: Optional[str] = None,
    *,
    color: ColorLike = "C0",
    label: Optional[str] = None,
    data: Optional[Any] = None,
    **kwargs: Any,
):
    """Plot the magnitude of the Fourier transform of ``x``.

    Args:
        x (array-like): The signal.
        Fs (float, optional): Sampling frequency. Defaults to 2.
        Fc (int, optional): Centre frequency added to the returned frequencies.
        window (callable or array-like, optional): Window applied before the FFT.
            Defaults to a Hanning window.
        pad_to (int, optional): Length the signal is zero-padded to.
        sides (str, optional): ``'default'``, ``'onesided'`` or ``'twosided'``.
        scale (str, optional): ``'linear'`` (the default, plotting energy) or ``'dB'``.
        color, label, data: As :func:`psd`.

    Returns:
        tuple: ``(spectrum, freqs, layer)``.
    """
    spec, freqs = _spectrum("magnitude_spectrum", x, Fs, Fc, window, pad_to, sides, data)
    if scale not in (None, "default", "linear", "dB"):
        raise ValueError(f"magnitude_spectrum(): unknown scale {scale!r}")
    # 20 log10, not 10: a magnitude is an amplitude, so the dB conversion is the
    # amplitude one. Getting this wrong halves every value on the axis.
    values = 20.0 * np.log10(spec) if scale == "dB" else spec
    layer = _spectral_line("magnitude_spectrum", freqs, values, color, label, kwargs)
    return spec, freqs, layer


def phase_spectrum(
    x: ArrayLike,
    Fs: Optional[float] = None,
    Fc: Optional[int] = None,
    window: Any = None,
    pad_to: Optional[int] = None,
    sides: Optional[str] = None,
    *,
    color: ColorLike = "C0",
    label: Optional[str] = None,
    data: Optional[Any] = None,
    **kwargs: Any,
):
    """Plot the (unwrapped) phase of the Fourier transform of ``x``, in radians.

    Args:
        x, Fs, Fc, window, pad_to, sides: As :func:`magnitude_spectrum`.
        color, label, data: As :func:`psd`.

    Returns:
        tuple: ``(phase, freqs, layer)``.
    """
    spec, freqs = _spectrum("phase_spectrum", x, Fs, Fc, window, pad_to, sides, data)
    layer = _spectral_line("phase_spectrum", freqs, spec, color, label, kwargs)
    return spec, freqs, layer


def angle_spectrum(
    x: ArrayLike,
    Fs: Optional[float] = None,
    Fc: Optional[int] = None,
    window: Any = None,
    pad_to: Optional[int] = None,
    sides: Optional[str] = None,
    *,
    color: ColorLike = "C0",
    label: Optional[str] = None,
    data: Optional[Any] = None,
    **kwargs: Any,
):
    """Plot the phase of the Fourier transform of ``x``, wrapped to (-pi, pi].

    Args:
        x, Fs, Fc, window, pad_to, sides: As :func:`magnitude_spectrum`.
        color, label, data: As :func:`psd`.

    Returns:
        tuple: ``(angle, freqs, layer)``.
    """
    spec, freqs = _spectrum("angle_spectrum", x, Fs, Fc, window, pad_to, sides, data)
    layer = _spectral_line("angle_spectrum", freqs, spec, color, label, kwargs)
    return spec, freqs, layer


def specgram(
    x: ArrayLike,
    NFFT: Optional[int] = None,
    Fs: Optional[float] = None,
    Fc: Optional[int] = None,
    detrend: Any = None,
    window: Any = None,
    noverlap: Optional[int] = None,
    cmap: Optional[str] = None,
    xextent: Optional[Tuple[float, float]] = None,
    pad_to: Optional[int] = None,
    sides: Optional[str] = None,
    scale_by_freq: Optional[bool] = None,
    mode: Optional[str] = None,
    scale: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    *,
    label: Optional[str] = None,
    data: Optional[Any] = None,
    **kwargs: Any,
):
    """Plot a spectrogram -- the frequency content of ``x`` over time, as an image.

    The segmentation and FFT are :func:`matplotlib.mlab.specgram`, so ``mode`` selects the
    same four quantities matplotlib offers and ``scale`` applies the same dB conversion.

    Args:
        x (array-like): The signal.
        NFFT (int, optional): Samples per segment. Defaults to 256.
        Fs (float, optional): Sampling frequency. Defaults to 2.
        Fc (int, optional): Centre frequency added to the frequency axis.
        detrend (str or callable, optional): As :func:`psd`.
        window (callable or array-like, optional): As :func:`psd`.
        noverlap (int, optional): Overlap between segments. Defaults to 128.
        cmap (str, optional): Colormap.
        xextent (tuple, optional): ``(xmin, xmax)`` for the time axis. Defaults to
            ``(0, max(times))``, as in matplotlib.
        pad_to, sides, scale_by_freq: As :func:`psd`.
        mode (str, optional): ``'psd'`` (default), ``'magnitude'``, ``'angle'`` or
            ``'phase'``.
        scale (str, optional): ``'dB'`` or ``'linear'``. Defaults to ``'dB'`` except for
            the angle and phase modes, which are already in radians.
        vmin, vmax (float, optional): Colour limits for the image.
        label (str, optional): Legend label.
        data (indexable, optional): If given, ``x`` may be a key into it.

    Returns:
        tuple: ``(spectrum, freqs, times, layer)``.

    Examples:
        >>> gplt.specgram(signal, NFFT=512, Fs=44100)
        >>> gplt.specgram(signal, mode="magnitude", scale="linear", cmap="magma")
    """
    from matplotlib import mlab

    (x,) = _resolve_data_args("specgram", data, x)
    sig = _as_signal(x, "x")
    if mode == "complex":
        raise ValueError("specgram(): cannot plot a complex spectrogram")
    # matplotlib's own defaulting: dB reads a power or an amplitude, but an angle is
    # already an angle, so the log would be meaningless there rather than merely unusual.
    if scale is None or scale == "default":
        scale = "linear" if mode in ("angle", "phase") else "dB"
    elif mode in ("angle", "phase") and scale == "dB":
        raise ValueError(f"specgram(): cannot use dB scale with mode {mode!r}")

    spec, freqs, times = mlab.specgram(
        x=sig,
        NFFT=NFFT,
        Fs=Fs,
        detrend=detrend,
        window=window,
        noverlap=128 if noverlap is None else noverlap,
        pad_to=pad_to,
        sides=sides,
        scale_by_freq=scale_by_freq,
        mode=mode,
    )
    if scale == "linear":
        values = spec
    elif scale == "dB":
        # 10 log10 for a power spectrum, 20 for an amplitude one -- the same split
        # `magnitude_spectrum` makes, and matplotlib makes it here off `mode`.
        values = (10.0 if mode in (None, "default", "psd") else 20.0) * np.log10(spec)
    else:
        raise ValueError(f"specgram(): unknown scale {scale!r}")

    freqs = freqs + (0 if Fc is None else Fc)
    xmin, xmax = (0.0, float(np.amax(times))) if xextent is None else (float(v) for v in xextent)
    # Drawn as an image over (time, frequency), which is what a spectrogram is; the
    # extent maps the pixel grid onto the real axes so the ticks read in Hz and seconds.
    layer = imshow(
        values,
        extent=(xmin, xmax, float(freqs[0]), float(freqs[-1])),
        cmap=_resolve_cmap(cmap, "viridis"),
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        label=label,
        **kwargs,
    )
    layer.metadata["artist"] = "specgram"
    return spec, freqs, times, layer


def acorr(
    x: ArrayLike,
    *,
    data: Optional[Any] = None,
    **kwargs: Any,
):
    """Plot the autocorrelation of ``x``.

    A thin wrapper over :func:`xcorr` with ``y = x``, exactly as in matplotlib, so every
    keyword ``xcorr`` takes works here too.

    Args:
        x (array-like): The signal.
        data (indexable, optional): If given, ``x`` may be a key into it.
        **kwargs: As :func:`xcorr` -- ``normed``, ``detrend``, ``usevlines``, ``maxlags``,
            plus line styling.

    Returns:
        tuple: ``(lags, c, line, b)``.
    """
    (x,) = _resolve_data_args("acorr", data, x)
    return xcorr(x, x, **kwargs)


def xcorr(
    x: ArrayLike,
    y: ArrayLike,
    normed: bool = True,
    detrend: Any = None,
    usevlines: bool = True,
    maxlags: int = 10,
    *,
    color: ColorLike = "C0",
    label: Optional[str] = None,
    data: Optional[Any] = None,
    **kwargs: Any,
):
    """Plot the cross-correlation of ``x`` and ``y``.

    Args:
        x, y (array-like): The two signals, of equal length.
        normed (bool, optional): Divide by ``sqrt(dot(x, x) * dot(y, y))``, so lag 0 of an
            autocorrelation reads 1 and signals of different scale become comparable.
            Defaults to True.
        detrend (callable, optional): Applied to both signals first. Defaults to
            :func:`matplotlib.mlab.detrend_none` -- matplotlib's default, which does *not*
            remove the mean.
        usevlines (bool, optional): Draw a stem at each lag plus a baseline at zero. When
            False, draw the correlation as a marker plot. Defaults to True.
        maxlags (int, optional): Lags each side of zero. ``None`` means ``len(x) - 1``.
            Defaults to 10.
        color, label, data: As :func:`psd`.

    Returns:
        tuple: ``(lags, c, line, b)`` -- the lags, the correlation, the stem or marker
        layer, and the zero baseline (``None`` when ``usevlines`` is False). Four
        elements, as matplotlib returns.

    Examples:
        >>> gplt.xcorr(a, b, maxlags=50)
        >>> gplt.acorr(signal, usevlines=False, normed=False)
    """
    from matplotlib import mlab

    x, y = _resolve_data_args("xcorr", data, x, y)
    a = _as_signal(x, "x")
    b = _as_signal(y, "y")
    if len(a) != len(b):
        raise ValueError("xcorr(): x and y must have the same length")

    # A deliberate superset of matplotlib, which takes only a callable here and raises
    # "'str' object is not callable" on `detrend='mean'` -- even though its own `psd`
    # accepts exactly those strings. Nothing that works there breaks here; the
    # inconsistency between the two functions just stops being the caller's problem.
    if isinstance(detrend, str):
        detrend = {
            "none": mlab.detrend_none,
            "mean": mlab.detrend_mean,
            "linear": mlab.detrend_linear,
        }.get(detrend, detrend)
        if isinstance(detrend, str):
            raise ValueError(
                f"xcorr(): unknown detrend {detrend!r}; expected 'none', 'mean', 'linear' "
                "or a callable"
            )
    detrend_func = mlab.detrend_none if detrend is None else detrend
    a = np.asarray(detrend_func(a), dtype=np.float64)
    b = np.asarray(detrend_func(b), dtype=np.float64)

    correls = np.correlate(a, b, mode="full")
    if normed:
        correls = correls / np.sqrt(np.dot(a, a) * np.dot(b, b))

    n = len(a)
    if maxlags is None:
        maxlags = n - 1
    if maxlags >= n or maxlags < 1:
        raise ValueError(f"xcorr(): maxlags must be in 1..{n - 1}, got {maxlags}")
    lags = np.arange(-maxlags, maxlags + 1)
    correls = correls[n - 1 - maxlags : n + maxlags]

    if usevlines:
        line = vlines(lags, 0, correls, color=color, label=label, **kwargs)
        # The baseline carries no label: matplotlib drops it so the legend shows one entry
        # per correlation, not one per correlation plus a stray horizontal rule.
        base = axhline(0.0, color=color, **kwargs)
    else:
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("linestyle", "None")
        layers = plot(lags, correls, color=color, label=label, **kwargs)
        line, base = (layers[-1] if layers else None), None
    return lags, correls, line, base


def _resolve_strides(
    func: str, shape: Tuple[int, int], rstride: int, cstride: int, kwargs: dict
) -> Tuple[int, int]:
    """Fold matplotlib's ``rcount``/``ccount`` into GLPlot's ``rstride``/``cstride``.

    matplotlib has both spellings and they say the same thing from opposite ends: a stride
    is "keep every n-th", a count is "keep about n in total". The conversion is exact
    enough to honour rather than warn about -- ``rstride = max(1, rows // rcount)`` is the
    same arithmetic matplotlib's own ``_process_kwargs`` does -- so a pasted
    ``plot_surface(X, Y, Z, rcount=50, ccount=50)`` decimates here as it does there instead
    of silently drawing the full-resolution mesh.

    Passing both a count and a stride for the same axis is a ``ValueError``, as in
    matplotlib: they contradict each other and picking one would be a guess.
    """
    rcount = kwargs.pop("rcount", None)
    ccount = kwargs.pop("ccount", None)
    rows, cols = shape
    for count, stride, count_name, stride_name in (
        (rcount, rstride, "rcount", "rstride"),
        (ccount, cstride, "ccount", "cstride"),
    ):
        if count is not None and stride != 1:
            raise ValueError(f"{func}() cannot specify both {stride_name} and {count_name}")
    if rcount is not None:
        rstride = max(1, rows // max(1, int(rcount)))
    if ccount is not None:
        cstride = max(1, cols // max(1, int(ccount)))
    return max(1, int(rstride)), max(1, int(cstride))


def plot_surface(
    X,
    Y,
    Z,
    *,
    cmap: Optional[str] = "viridis",
    color: Optional[ColorLike] = None,
    elev: float = 30.0,
    azim: float = -60.0,
    scale_z: float = 1.0,
    rstride: int = 1,
    cstride: int = 1,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: Optional[float] = None,
    label: Optional[str] = None,
    **kwargs,
):
    """Plot a 3D surface mesh.

    Renders a triangulated surface from 2D grid of (X, Y) coordinates and
    corresponding Z values. Surface is colored by Z-value through colormap.
    Stride parameters allow downsampling large meshes for performance.

    Args:
        X (array-like): 2D array of x-coordinates. Shape (M, N).
        Y (array-like): 2D array of y-coordinates. Shape (M, N).
        Z (array-like): 2D array of z-coordinates (heights). Shape (M, N).
        cmap (str, optional): Colormap for Z-value coloring. Defaults to 'viridis'.
        elev (float, optional): Camera elevation angle in degrees. Defaults to 30.0.
        azim (float, optional): Camera azimuth angle in degrees. Defaults to -60.0.
        scale_z (float, optional): Multiplies the z data before it is drawn.
            Defaults to 1.0 -- z is plotted as given, which is what matplotlib does
            and what every z read-back (``set_zlim()``, the z ticks, the box) then
            reports. It used to default to 0.7 to make the box look cube-ish, which
            rewrote the caller's data: ``scatter3d([0, 1], [0, 1], [0, 10])`` stored
            ``[0, 7]`` and the z axis was in fabricated units. Visual squashing is
            :func:`set_box_aspect`'s job -- it changes the picture, not the data.
        color (str or tuple, optional): One flat colour for the whole surface, as in
            matplotlib. Wins over ``cmap`` when both are given, because naming a colour is
            the more specific instruction. Defaults to None (colour by z through ``cmap``).
        rstride (int, optional): Row stride (mesh decimation along rows).
            Defaults to 1 (no decimation).
        cstride (int, optional): Column stride (mesh decimation along columns).
            Defaults to 1 (no decimation).
        vmin, vmax (float, optional): The z values the colormap's ends are pinned to.
            Defaults to the data range.
        alpha (float, optional): Transparency (0.0-1.0). Defaults to 1.0.
        label (str, optional): Legend label. Defaults to None.
        **kwargs: matplotlib's ``rcount``/``ccount`` are honoured (converted to strides);
            ``norm``, ``shade``, ``lightsource``, ``facecolors``, ``edgecolor``,
            ``linewidth`` and ``antialiased`` are accepted for signature parity and raise
            :class:`MatplotlibCompatWarning` -- there is no lighting model, no per-face
            colour path and no edge pass in the 3D mesh renderer.

    Note:
        With neither ``color`` nor ``cmap`` overridden the surface is coloured by z through
        viridis, where matplotlib draws a single solid colour. That default is GLPlot's and
        is kept deliberately; pass ``color=`` for matplotlib's look.

    Returns:
        Layer: The 3D mesh layer added to plot.

    Raises:
        ValueError: If X, Y, and Z have different shapes.

    Examples:
        Plot a simple surface:

        >>> x = np.linspace(-5, 5, 50)
        >>> y = np.linspace(-5, 5, 50)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = np.sin(np.sqrt(X**2 + Y**2))
        >>> gplt.plot_surface(X, Y, Z)
        >>> gplt.show()

        With downsampling for performance:

        >>> gplt.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap='cool')

        Custom colors and camera:

        >>> gplt.plot_surface(X, Y, Z, cmap='hot', elev=45, azim=120,
        ...                   scale_z=1.0, alpha=0.9)
    """
    xx = _as_float_array(X, name="X")
    yy = _as_float_array(Y, name="Y")
    zz = _as_float_array(Z, name="Z")
    if not (xx.shape == yy.shape == zz.shape):
        raise ValueError("X, Y, and Z must have matching shapes")
    rstride, cstride = _resolve_strides("plot_surface", zz.shape[:2], rstride, cstride, kwargs)
    _warn_unsupported(
        "plot_surface",
        {
            "norm": kwargs.pop("norm", None),
            "shade": (False if kwargs.pop("shade", True) is False else None),
            "lightsource": kwargs.pop("lightsource", None),
            "facecolors": kwargs.pop("facecolors", None),
            "edgecolor": kwargs.pop("edgecolor", kwargs.pop("edgecolors", None)),
            "linewidth": kwargs.pop("linewidth", kwargs.pop("linewidths", None)),
            "antialiased": (False if kwargs.pop("antialiased", True) is False else None),
        },
        {
            "norm": "is not supported; pass vmin/vmax instead",
            "shade": "has no effect: the 3D mesh renderer has no lighting model, so the "
            "surface is drawn unshaded either way",
            "lightsource": "has no effect: there is no lighting model to place a light in",
            "facecolors": "has no effect: colours are per vertex, not per face — pass "
            "color= for a flat surface or cmap= to map z",
            "edgecolor": "has no effect: the mesh has no edge pass — draw plot_wireframe() "
            "over the surface instead",
            "linewidth": "has no effect: the mesh has no edge pass",
            "antialiased": "has no effect: multisampling is a figure-level setting "
            "(options.enable_multisample)",
        },
    )
    # The 3D compositing knobs are honoured (passed to _add_3d_layer below), so pull them
    # out before the compat warner or it would flag them as ignored -- the one thing they
    # are not.
    _compositing = {k: kwargs.pop(k) for k in ("blend", "depth_write", "auto_alpha") if k in kwargs}
    _warn_unsupported("plot_surface", kwargs)
    xs, ys, zs = xx[::rstride, ::cstride], yy[::rstride, ::cstride], zz[::rstride, ::cstride]
    rows, cols = xs.shape
    verts = np.column_stack([xs.ravel(), ys.ravel(), (zs * float(scale_z)).ravel()]).astype(
        np.float32
    )
    if color is not None:
        # A flat colour is matplotlib's default look and its most-used keyword here. It has
        # to beat the z ramp rather than sit beside it: ``_add_3d_layer`` prefers the
        # per-vertex array, so building one anyway would ignore the argument in silence.
        colors = np.tile(
            np.asarray(list(_normalize_rgba(color, n=None)), dtype=np.float32), (len(verts), 1)
        )
    else:
        colors = _colormap_values(zs.ravel(), cmap=cmap, vmin=vmin, vmax=vmax)
    if alpha is not None:
        colors = colors.copy()
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
            "cmap": None if color is not None else cmap,
            "vmin": vmin,
            "vmax": vmax,
            "rstride": rstride,
            "cstride": cstride,
        },
        blend=_compositing.get("blend"),
        depth_write=_compositing.get("depth_write"),
        auto_alpha=_compositing.get("auto_alpha"),
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
    blend: Optional[Union[str, "BlendMode"]] = None,
    depth_write: Optional[bool] = None,
    auto_alpha: Optional[float] = None,
):
    """A point cloud shaded by ``values``, for looking *into* a volume.

    Compositing is where this verb earns its keep, and the three arguments for it are worth
    knowing (:func:`set_layer_compositing` sets the same three after the fact):

    ``blend="additive"``
        Overlapping points get brighter instead of averaging out, so density reads directly
        as brightness and faint tails survive at alpha 0.02 -- the look a nebula or a
        scattering volume wants. Needs a dark background, and skips the depth sort, since
        addition gives the same image in any order.
    ``auto_alpha=0.9``
        Solves for the alpha that puts a covered pixel near that opacity under the current
        view, instead of one number that saturates when zoomed out and vanishes when
        zoomed in.
    ``depth_write=True``
        Makes a dense cloud occlude what is behind it, rather than accumulate through it.
    """
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
        blend=blend,
        depth_write=depth_write,
        auto_alpha=auto_alpha,
    )
    return layer


def voxels(
    *args: Any,
    facecolors: Optional[Any] = None,
    edgecolors: Optional[Any] = None,
    shade: bool = True,
    lightsource: Optional[Any] = None,
    axlim_clip: bool = False,
    **kwargs: Any,
) -> dict:
    """Draw a filled 3D boolean array as cubes (matplotlib's ``Axes3D.voxels``).

    Accepts both of matplotlib's call shapes:

    * ``voxels(filled)`` -- ``filled`` is a boolean ``(nx, ny, nz)`` occupancy array and
      cube ``(i, j, k)`` occupies the unit cell from ``(i, j, k)`` to ``(i+1, j+1, k+1)``.
    * ``voxels(x, y, z, filled)`` -- ``x``/``y``/``z`` are the *corner* coordinates,
      shaped ``(nx+1, ny+1, nz+1)`` as ``np.indices`` produces them.

    This name used to be an alias of :func:`volume3d`, which takes ``(x, y, z, values)``
    point clouds. That was the only place a GLPlot 3D name *shadowed* a real matplotlib
    method with different semantics, and shadowing is worse than missing: ``hasattr(ax,
    "voxels")`` answered True, so a capability probe got it wrong and the call then failed
    with ``volume3d() missing 2 required positional arguments``, an error naming a function
    the caller had never heard of.

    Args:
        *args: ``(filled,)`` or ``(x, y, z, filled)``, as above.
        facecolors: A single colour, or an object array shaped like ``filled`` holding one
            colour per cube. matplotlib's ``(nx, ny, nz, 4)`` RGBA form is accepted too.
        edgecolors: Colour of the cube edges. None (the default) draws them black; pass
            ``"none"`` for no edges.
        shade (bool): Accepted for parity; warned about when False.
        lightsource: Accepted for parity; warned about.
        axlim_clip (bool): Accepted for parity; warned about when True.
        **kwargs: Forwarded to the underlying :func:`bar3d` (``alpha``, ``label``,
            ``elev``, ``azim``, ``gap`` ...).

    Returns:
        dict: ``{(i, j, k): layer}`` for every filled cube, as matplotlib returns a dict of
        ``Poly3DCollection``. Every value is the *same* layer object: GLPlot batches all
        cubes into one mesh, because a layer per cube would put a draw call per voxel on
        the GPU. Use the dict for its keys (which cubes were drawn); do not assume the
        values are distinct artists.

    Examples:
        >>> filled = np.zeros((3, 3, 3), dtype=bool)
        >>> filled[1, 1, 1] = True
        >>> gplt.voxels(filled)
    """
    if len(args) == 1:
        filled = np.asarray(args[0])
        origins = None
    elif len(args) == 4:
        # ``x``/``y``/``z`` are corner grids of shape (nx+1, ny+1, nz+1), each varying
        # along its own axis. Only the per-axis edge coordinates matter for axis-aligned
        # cubes, so each is reduced to a 1-D vector: ``x[:, 0, 0]``, ``y[0, :, 0]``,
        # ``z[0, 0, :]``. The fully general (sheared) grid matplotlib accepts is *not*
        # supported -- the 3D renderer draws axis-aligned boxes, and honouring a sheared
        # grid would mean silently squaring it up.
        grids = [np.asarray(a, dtype=np.float64) for a in args[:3]]
        origins = (grids[0][:, 0, 0], grids[1][0, :, 0], grids[2][0, 0, :])
        filled = np.asarray(args[3])
    else:
        raise TypeError(f"voxels() takes 1 or 4 positional arguments but {len(args)} were given")

    if filled.ndim != 3:
        raise ValueError(f"voxels() needs a 3D array of booleans, got ndim={filled.ndim}")
    _warn_unsupported(
        "voxels",
        {
            "shade": False if shade is False else None,
            "lightsource": lightsource,
            "axlim_clip": True if axlim_clip else None,
        },
        {
            "shade": "has no effect: cube faces are drawn flat — ssao() is GLPlot's "
            "shading control",
            "lightsource": "has no effect: there is no lighting model to place a light in",
            "axlim_clip": "has no effect: GLPlot's 3D renderer does not clip geometry to "
            "the axis limits",
        },
    )

    mask = np.asarray(filled, dtype=bool)
    idx = np.argwhere(mask)
    if origins is None:
        edges = [np.arange(mask.shape[axis] + 1, dtype=np.float64) for axis in range(3)]
    else:
        edges = [np.asarray(o, dtype=np.float64).ravel() for o in origins]
        for axis, edge in enumerate(edges):
            if edge.size != mask.shape[axis] + 1:
                raise ValueError(
                    f"voxels(): coordinate array {axis} has {edge.size} edges, "
                    f"expected {mask.shape[axis] + 1}"
                )
    keys = [tuple(int(v) for v in row) for row in idx]
    if not keys:
        return {}

    lo = np.column_stack([edges[axis][idx[:, axis]] for axis in range(3)])
    hi = np.column_stack([edges[axis][idx[:, axis] + 1] for axis in range(3)])
    span = hi - lo

    cube_colors = _voxel_face_colors(facecolors, idx, len(keys))
    if edgecolors is None:
        edge_color: Optional[ColorLike] = "k"
    elif isinstance(edgecolors, str) and edgecolors.strip().lower() == "none":
        edge_color = None
    else:
        edge_color = edgecolors

    artists = bar3d(
        lo[:, 0],
        lo[:, 1],
        lo[:, 2],
        span[:, 0],
        span[:, 1],
        span[:, 2],
        c=cube_colors,
        edge_color=edge_color,
        **kwargs,
    )
    layer = artists[0] if isinstance(artists, list) else artists
    layer.metadata["artist"] = "voxels"
    return {key: layer for key in keys}


def _voxel_face_colors(facecolors: Optional[Any], idx: np.ndarray, count: int):
    """``facecolors`` as something :func:`bar3d`'s ``c=`` understands, or None.

    matplotlib lets ``facecolors`` be one colour, an object array of colours shaped like
    ``filled``, or an ``(nx, ny, nz, 4)`` RGBA block. All three collapse to "one RGBA per
    drawn cube" once the filled cells are known, which is the only shape ``bar3d`` takes.
    """
    if facecolors is None:
        return None
    arr = np.asarray(facecolors)
    if arr.ndim == 4 and arr.shape[-1] in (3, 4):
        picked = arr[idx[:, 0], idx[:, 1], idx[:, 2]]
        return _normalize_rgba(picked, n=count)
    if arr.ndim == 3 and arr.dtype == object:
        picked = arr[idx[:, 0], idx[:, 1], idx[:, 2]]
        return np.asarray(
            [list(_normalize_rgba(colour, n=None)) for colour in picked], dtype=np.float32
        )
    return facecolors


def plot_wireframe(
    X,
    Y,
    Z,
    *,
    elev: float = 30.0,
    azim: float = -60.0,
    scale_z: float = 1.0,
    rstride: int = 4,
    cstride: int = 4,
    color: ColorLike = "k",
    linewidth: float = 0.7,
    label: Optional[str] = None,
    **kwargs,
):
    """Draw the wireframe of a surface grid (matplotlib's ``Axes3D.plot_wireframe``).

    ``rcount``/``ccount`` are honoured as the count-shaped spelling of ``rstride``/
    ``cstride`` (see :func:`_resolve_strides`); ``linestyle``, ``antialiased`` and the rest
    of matplotlib's Line3DCollection keywords are accepted and warned about.

    Note:
        The stride defaults differ from matplotlib's: GLPlot keeps every 4th row and column
        where matplotlib targets ``rcount = ccount = 50``. Pass ``rcount``/``ccount``
        explicitly to get matplotlib's framing on a grid of a different size.
    """
    xx = _as_float_array(X, name="X")
    yy = _as_float_array(Y, name="Y")
    zz = _as_float_array(Z, name="Z")
    if not (xx.shape == yy.shape == zz.shape):
        raise ValueError("X, Y, and Z must have matching shapes")
    if "rcount" in kwargs or "ccount" in kwargs:
        # The stride defaults here are 4, not 1, so the "both given" check in
        # _resolve_strides would fire on a call that only named a count. Reset them first.
        rstride = 1 if "rcount" in kwargs else rstride
        cstride = 1 if "ccount" in kwargs else cstride
    rstride, cstride = _resolve_strides("plot_wireframe", zz.shape[:2], rstride, cstride, kwargs)
    _warn_unsupported(
        "plot_wireframe",
        {
            "linestyle": kwargs.pop("linestyle", kwargs.pop("ls", None)),
            "antialiased": (False if kwargs.pop("antialiased", True) is False else None),
            "cmap": kwargs.pop("cmap", None),
            "norm": kwargs.pop("norm", None),
        },
        {
            "linestyle": "has no effect: 3D lines are drawn solid",
            "antialiased": "has no effect: multisampling is a figure-level setting "
            "(options.enable_multisample)",
            "cmap": "has no effect: a wireframe is drawn in one flat colour — pass color=",
            "norm": "has no effect: a wireframe is drawn in one flat colour",
        },
    )
    _warn_unsupported("plot_wireframe", kwargs)
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
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    dx: Union[ArrayLike, float],
    dy: Union[ArrayLike, float],
    dz: Union[ArrayLike, float],
    color: Optional[ColorLike] = None,
    zsort: str = "average",
    shade: bool = True,
    lightsource: Optional[Any] = None,
    *,
    alpha: Optional[float] = None,
    elev: float = 30.0,
    azim: float = -60.0,
    scale_z: float = 1.0,
    label: Optional[str] = None,
    shape: str = "box",
    c: Optional[Union[ColorLike, ArrayLike]] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    gap: float = 0.0,
    edge_color: Optional[ColorLike] = "k",
    edge_width: float = 0.8,
    ssao: Optional[bool] = None,
    ssao_strength: Optional[float] = None,
    **kwargs: Any,
) -> BaseLayer:
    """Create a 3D bar chart.

    Renders rectangular bars or hexagonal prisms at 3D positions with
    specified dimensions. Supports per-bar colormapping and customizable
    gaps between bars. Each bar is rendered as a textured 3D mesh.

    Args:
        x (array-like): Bar x-positions. Shape (N,).
        y (array-like): Bar y-positions. Shape (N,).
        z (array-like): Bar z-positions (baseline heights). Shape (N,).
        dx (array-like or float): Bar width along x-axis.
        dy (array-like or float): Bar width along y-axis.
        dz (array-like or float): Bar height along z-axis.
        color (str or tuple, optional): Single color for all bars. Ignored if c
            provided. Defaults to None, which draws them 'tab:blue'. Positional, as in
            matplotlib -- ``bar3d(x, y, z, dx, dy, dz, 'r')`` is a legal call there and
            used to be a ``TypeError`` here.
        zsort (str, optional): Accepted for matplotlib parity and warned about. The GPU
            depth buffer sorts fragments, so there is no painter's-algorithm face order to
            choose. Defaults to 'average'.
        shade (bool, optional): Accepted for matplotlib parity and warned about when False.
            Bar faces are drawn flat; :func:`ssao` is GLPlot's shading control.
        lightsource (optional): Accepted for matplotlib parity and warned about.
        alpha (float, optional): Transparency. Defaults to 1.0.
        elev (float, optional): Camera elevation angle. Defaults to 30.0.
        azim (float, optional): Camera azimuth angle. Defaults to -60.0.
        scale_z (float, optional): Multiplies the z data before it is drawn.
            Defaults to 1.0 -- z is plotted as given, which is what matplotlib does
            and what every z read-back (``set_zlim()``, the z ticks, the box) then
            reports. It used to default to 0.7 to make the box look cube-ish, which
            rewrote the caller's data: ``scatter3d([0, 1], [0, 1], [0, 10])`` stored
            ``[0, 7]`` and the z axis was in fabricated units. Visual squashing is
            :func:`set_box_aspect`'s job -- it changes the picture, not the data.
        label (str, optional): Legend label. Defaults to None.
        shape (str, optional): Bar shape: 'box' (rectangular) or 'hex'
            (hexagonal prism). Defaults to 'box'.
        c (array-like or str, optional): Per-bar colors. If 1D numeric,
            maps to colormap. Defaults to None.
        cmap (str, optional): Colormap name. Defaults to 'viridis'.
        vmin, vmax (float, optional): Colormap normalization. Defaults to None.
        gap (float, optional): Gap between bars (0.0-0.95). Defaults to 0.0.
        edge_color (str or tuple, optional): Edge color for wireframe.
            Defaults to black.
        edge_width (float, optional): Edge line width. Defaults to 0.8.
        ssao (bool, optional): Enable screen-space ambient occlusion.
            Defaults to None (use plot setting).
        ssao_strength (float, optional): SSAO strength (0-1). Defaults to 0.45.

    Returns:
        list: Bar and edge layer objects added to plot.

    Examples:
        Simple 3D bars:

        >>> x = [0, 1, 2]
        >>> y = [0, 0, 0]
        >>> z = [0, 0, 0]
        >>> dx = dy = [0.8, 0.8, 0.8]
        >>> dz = [1, 2, 1.5]
        >>> gplt.bar3d(x, y, z, dx, dy, dz)
        >>> gplt.show()

        With colormap and gaps:

        >>> colors = [10, 20, 15]
        >>> gplt.bar3d(x, y, z, 0.7, 0.7, dz, c=colors, cmap='hot',
        ...             gap=0.2, shape='box')

        Hexagonal bars:

        >>> gplt.bar3d(x, y, z, 0.7, 0.7, dz, shape='hex', gap=0.1,
        ...             edge_color='white', edge_width=1.5)
    """
    _warn_unsupported(
        "bar3d",
        {
            "zsort": None if zsort == "average" else zsort,
            "shade": False if shade is False else None,
            "lightsource": lightsource,
            "data": kwargs.pop("data", None),
        },
        {
            "zsort": "has no effect: faces are resolved by the GPU depth buffer, not by a "
            "painter's-algorithm sort order",
            "shade": "has no effect: bar faces are drawn flat — ssao() is GLPlot's "
            "shading control",
            "lightsource": "has no effect: there is no lighting model to place a light in",
            "data": "is not supported by bar3d; index the frame yourself",
        },
    )
    _warn_unsupported("bar3d", kwargs)
    color = "tab:blue" if color is None else color
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


def _stream_sample(
    gx: np.ndarray, gy: np.ndarray, u: np.ndarray, v: np.ndarray, px: float, py: float
) -> Optional[Tuple[float, float]]:
    """The field at ``(px, py)``, bilinearly interpolated, or None if off the grid.

    Bilinear, not nearest: a streamline is an integral, and sampling the field in steps
    makes the curve visibly polygonal along the cell boundaries it crosses.
    """
    if not (gx[0] <= px <= gx[-1] and gy[0] <= py <= gy[-1]):
        return None
    # `searchsorted` rather than a division: the grid is only guaranteed monotonic, not
    # evenly spaced, and matplotlib's streamplot accepts an uneven one.
    i = int(np.clip(np.searchsorted(gx, px) - 1, 0, len(gx) - 2))
    j = int(np.clip(np.searchsorted(gy, py) - 1, 0, len(gy) - 2))
    dx = gx[i + 1] - gx[i]
    dy = gy[j + 1] - gy[j]
    tx = 0.0 if dx == 0 else (px - gx[i]) / dx
    ty = 0.0 if dy == 0 else (py - gy[j]) / dy

    def lerp(f: np.ndarray) -> float:
        # f is indexed [row=y, col=x], matching the (ny, nx) shape meshgrid produces.
        return float(
            f[j, i] * (1 - tx) * (1 - ty)
            + f[j, i + 1] * tx * (1 - ty)
            + f[j + 1, i] * (1 - tx) * ty
            + f[j + 1, i + 1] * tx * ty
        )

    return lerp(u), lerp(v)


def _stream_trace(
    gx: np.ndarray,
    gy: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    seed: Tuple[float, float],
    step: float,
    max_steps: int,
    occupied: np.ndarray,
    cell: Tuple[float, float],
) -> Optional[np.ndarray]:
    """Integrate one streamline through the field, forwards then backwards from ``seed``.

    RK4 on the *normalised* field, so `step` is an arc length rather than a time: the
    integrator then takes the same size step in a slow region as in a fast one, and the
    line stays smooth where the field is weak instead of collapsing to a dot.

    ``occupied`` is the thinning grid matplotlib calls density -- a streamline stops when
    it enters a cell another has already claimed. Without it every seed traces the same
    attractor and the plot becomes a single thick smear.

    The cells *this* trace claims are held aside until it finishes, and only merged into
    ``occupied`` at the end. Marking them as it goes instead makes every line stop after
    one step: the second step is still inside the cell the first step just claimed, so the
    line collides with itself and the whole plot comes out as a field of stubs.
    """
    claimed: set = set()
    paths = []
    for direction in (1.0, -1.0):
        px, py = seed
        path = [(px, py)]
        for _ in range(max_steps):
            k1 = _stream_sample(gx, gy, u, v, px, py)
            if k1 is None:
                break

            def unit(vec):
                mag = np.hypot(vec[0], vec[1])
                # A stagnation point: the line has nowhere to go, and dividing by the
                # magnitude would send it to infinity instead of stopping it.
                return None if mag < 1e-12 else (vec[0] / mag, vec[1] / mag)

            d1 = unit(k1)
            if d1 is None:
                break
            h = direction * step
            k2 = _stream_sample(gx, gy, u, v, px + 0.5 * h * d1[0], py + 0.5 * h * d1[1])
            d2 = unit(k2) if k2 else None
            if d2 is None:
                break
            k3 = _stream_sample(gx, gy, u, v, px + 0.5 * h * d2[0], py + 0.5 * h * d2[1])
            d3 = unit(k3) if k3 else None
            if d3 is None:
                break
            k4 = _stream_sample(gx, gy, u, v, px + h * d3[0], py + h * d3[1])
            d4 = unit(k4) if k4 else None
            if d4 is None:
                break

            px += h * (d1[0] + 2 * d2[0] + 2 * d3[0] + d4[0]) / 6.0
            py += h * (d1[1] + 2 * d2[1] + 2 * d3[1] + d4[1]) / 6.0
            if not (gx[0] <= px <= gx[-1] and gy[0] <= py <= gy[-1]):
                break

            ci = int((px - gx[0]) / cell[0])
            cj = int((py - gy[0]) / cell[1])
            ci = min(max(ci, 0), occupied.shape[1] - 1)
            cj = min(max(cj, 0), occupied.shape[0] - 1)
            # Another line's cell stops this one; its own does not. Both directions of
            # this trace share `claimed`, so the backward half does not stop dead on the
            # forward half's trail either.
            if occupied[cj, ci] and (cj, ci) not in claimed:
                break
            claimed.add((cj, ci))
            path.append((px, py))
        paths.append(path)

    # Backwards half reversed and joined, minus the duplicated seed.
    full = paths[1][::-1] + paths[0][1:]
    if len(full) < 2:
        return None
    # Published only now the line is known to exist: a trace that died on its first step
    # must not leave its cell claimed against the lines that come after it.
    for cj, ci in claimed:
        occupied[cj, ci] = True
    return np.asarray(full, dtype=np.float32)


def streamplot(
    x: ArrayLike,
    y: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    density: Union[float, Tuple[float, float]] = 1.0,
    linewidth: float = 1.0,
    color: ColorLike = "k",
    cmap: Optional[str] = None,
    norm: Optional[Any] = None,
    arrowsize: float = 1.0,
    arrowstyle: str = "-|>",
    minlength: float = 0.1,
    maxlength: float = 4.0,
    integration_direction: str = "both",
    broken_streamlines: bool = True,
    start_points: Optional[ArrayLike] = None,
    zorder: Optional[float] = None,
    transform: Optional[Any] = None,
    num_arrows: int = 1,
    integration_max_step_scale: float = 1.0,
    integration_max_error_scale: float = 1.0,
    *,
    label: Optional[str] = None,
    data: Optional[Any] = None,
):
    """Draw the streamlines of a vector field.

    Where :func:`quiver` shows the field *at* points, this follows it: each line is
    an RK4 integration through the field, so the picture shows where a particle
    dropped into it would actually go.

    Args:
        x, y (array-like): The grid the field is sampled on. 1-D and strictly
            increasing, or the 2-D output of ``np.meshgrid``.
        u, v (array-like): The field's components, shape ``(len(y), len(x))``.
        density (float or tuple, optional): How densely to seed. Higher packs more
            lines in. Defaults to 1.
        linewidth (float, optional): Line width in pixels. Defaults to 1.
        color (str or tuple, optional): Streamline colour. Defaults to black.
        cmap (str, optional): Colour the lines by local speed instead. One colour
            per line -- see the note.
        norm (Normalize or str, optional): How speed maps onto ``cmap``.
        arrowsize, arrowstyle, num_arrows: Accepted for matplotlib parity. GLPlot draws the
            lines without arrowheads -- use :func:`quiver` for direction glyphs.
            Ignored.
        integration_max_step_scale, integration_max_error_scale: Accepted for matplotlib
            parity; ignored. GLPlot integrates streamlines with a fixed RK4 step, not
            matplotlib's adaptive integrator.
        minlength (float, optional): Streamlines shorter than this are dropped.
        maxlength (float, optional): Maximum arc length of one streamline.
        integration_direction (str, optional): 'both' (default), 'forward' or
            'backward'.
        broken_streamlines (bool, optional): Let a line stop when it reaches a
            cell another has already filled. False traces each seed to the end,
            which is denser and slower. Defaults to True.
        start_points (array-like, optional): Seed points, shape (N, 2). Defaults
            to a grid derived from ``density``.
        zorder (float, optional): Accepted for matplotlib parity. GLPlot draws layers in
            the order they are added. Ignored.
        transform (optional): Accepted for matplotlib parity. GLPlot has no artist
            transform stack. Ignored.
        label (str, optional): Legend label.
        data (indexable, optional): If given, the arrays may be keys into it.

    Returns:
        list: One polyline layer per streamline.

    Note:
        matplotlib varies colour and width *along* each line. GLPlot's polyline
        carries one colour and one width per layer, so ``cmap`` colours each line
        by its mean speed instead. The difference shows where a single streamline
        crosses a large speed range.

    Examples:
        >>> Y, X = np.mgrid[-3:3:100j, -3:3:100j]
        >>> gplt.streamplot(X, Y, -1 - X**2 + Y, 1 + X - Y**2, density=1.5)
        >>> gplt.streamplot(X, Y, U, V, cmap='viridis')
    """
    x, y, u, v = _resolve_data_args("streamplot", data, x, y, u, v)
    _warn_unsupported(
        "streamplot",
        {"zorder": zorder, "transform": transform},
        {
            "zorder": "has no effect: GLPlot draws layers in the order they are added",
            "transform": "has no effect: GLPlot has no artist transform stack; the arrays "
            "are read in data coordinates",
        },
    )
    _warn_unsupported(
        "streamplot",
        {
            "arrowsize": arrowsize if arrowsize != 1.0 else None,
            "arrowstyle": arrowstyle if arrowstyle != "-|>" else None,
            "num_arrows": num_arrows if num_arrows != 1 else None,
            "integration_max_step_scale": (
                integration_max_step_scale if integration_max_step_scale != 1.0 else None
            ),
            "integration_max_error_scale": (
                integration_max_error_scale if integration_max_error_scale != 1.0 else None
            ),
        },
        {
            "arrowsize": "has no effect: GLPlot draws streamlines without arrowheads. Use "
            "quiver() for direction glyphs",
            "arrowstyle": "has no effect: GLPlot draws streamlines without arrowheads",
            "num_arrows": "has no effect: GLPlot draws streamlines without arrowheads",
            "integration_max_step_scale": "has no effect: GLPlot integrates with a fixed "
            "RK4 step, not matplotlib's adaptive integrator",
            "integration_max_error_scale": "has no effect: GLPlot integrates with a fixed "
            "RK4 step, not matplotlib's adaptive integrator",
        },
    )
    if integration_direction not in ("both", "forward", "backward"):
        raise ValueError(
            f"unsupported integration_direction: {integration_direction!r}. "
            "Expected 'both', 'forward' or 'backward'."
        )

    gx = np.asarray(x, dtype=np.float64)
    gy = np.asarray(y, dtype=np.float64)
    # meshgrid output is accepted as readily as the 1-D axes, since that is what the
    # caller has in hand after building the field.
    if gx.ndim == 2:
        gx = gx[0, :]
    if gy.ndim == 2:
        gy = gy[:, 0]
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    if u_arr.shape != v_arr.shape:
        raise ValueError(f"u and v must have the same shape, got {u_arr.shape} and {v_arr.shape}")
    if u_arr.shape != (len(gy), len(gx)):
        raise ValueError(
            f"u and v must have shape (len(y), len(x)) == {(len(gy), len(gx))}, "
            f"got {u_arr.shape}"
        )
    if len(gx) < 2 or len(gy) < 2:
        raise ValueError("streamplot(): x and y need at least 2 points each")

    dens = np.atleast_1d(np.asarray(density, dtype=np.float64))
    dx_n, dy_n = (dens[0], dens[0]) if len(dens) == 1 else (dens[0], dens[1])
    if dx_n <= 0 or dy_n <= 0:
        raise ValueError(f"density must be positive, got {density!r}")

    # 30 cells per unit density is matplotlib's own thinning resolution.
    nx_cells = max(2, int(30 * dx_n))
    ny_cells = max(2, int(30 * dy_n))
    occupied = np.zeros((ny_cells, nx_cells), dtype=bool)
    cell = ((gx[-1] - gx[0]) / nx_cells, (gy[-1] - gy[0]) / ny_cells)

    if start_points is not None:
        seeds = _as_float_array(start_points, ndim=2, name="start_points")
        if seeds.shape[1] != 2:
            raise ValueError("start_points must have shape (N, 2)")
        seed_list = [(float(p[0]), float(p[1])) for p in seeds]
    else:
        sx = np.linspace(gx[0], gx[-1], max(2, int(nx_cells / 2)))
        sy = np.linspace(gy[0], gy[-1], max(2, int(ny_cells / 2)))
        seed_list = [(float(a), float(b)) for b in sy for a in sx]

    span = float(np.hypot(gx[-1] - gx[0], gy[-1] - gy[0]))
    step = span / 200.0
    max_steps = max(2, int(maxlength * span / max(step, 1e-12)))

    lines = []
    for seed in seed_list:
        if not broken_streamlines:
            occupied[:] = False
        pts = _stream_trace(gx, gy, u_arr, v_arr, seed, step, max_steps, occupied, cell)
        if pts is None:
            continue
        if integration_direction != "both":
            # `_stream_trace` always walks both ways from the seed; the seed sits at the
            # join, so a one-sided line is the half on the requested side of it.
            mid = int(np.argmin(np.hypot(pts[:, 0] - seed[0], pts[:, 1] - seed[1])))
            pts = pts[mid:] if integration_direction == "forward" else pts[: mid + 1]
        if len(pts) < 2:
            continue
        arc = float(np.sum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))))
        if arc < minlength:
            continue
        lines.append(pts)

    plot_obj = _get_or_create_plot()
    if cmap is not None:
        from matplotlib import colormaps

        speeds = np.array(
            [
                float(
                    np.mean(
                        [
                            np.hypot(*(_stream_sample(gx, gy, u_arr, v_arr, p[0], p[1]) or (0, 0)))
                            for p in pts[:: max(1, len(pts) // 8)]
                        ]
                    )
                )
                for pts in lines
            ]
        )
        rgba = colormaps.get_cmap(cmap)(_normalize_cvalues(speeds, norm, None, None))
    else:
        rgba = [_normalize_rgba(color)] * len(lines)

    artists = []
    for i, pts in enumerate(lines):
        plot_obj.add_line_strip(
            pts[:, 0],
            pts[:, 1],
            color=tuple(float(c) for c in rgba[i]),
            width=float(linewidth),
            label=label if i == 0 else None,
        )
        layer = plot_obj.scene.layers[-1]
        layer.metadata["artist"] = "streamplot"
        artists.append(layer)

    _set_dirty(plot_obj)
    return artists


def quiver(
    x: ArrayLike,
    y: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    *,
    color: ColorLike = "k",
    scale: float = 1.0,
    width: float = 1.0,
    head_width: float = 0.08,
    head_length: float = 0.12,
    label: Optional[str] = None,
    **kwargs: Any,
) -> list[BaseLayer]:
    """Create a 2D vector field plot.

    Renders arrows representing 2D vector field (u, v) at each point (x, y).
    Arrows are batched for efficient rendering of large vector fields.
    Useful for visualizing flow fields, gradients, or force vectors.

    Args:
        x (array-like): X-positions of vectors. Shape (N,).
        y (array-like): Y-positions of vectors. Shape (N,).
        u (array-like): X-components of vectors. Shape (N,).
        v (array-like): Y-components of vectors. Shape (N,).
        color (str or tuple, optional): Arrow color. Named color or RGBA.
            Defaults to black.
        scale (float, optional): Scaling factor for arrow magnitudes.
            Larger values make arrows longer. Defaults to 1.0.
        width (float, optional): Arrow shaft line width in pixels. Defaults to 1.0.
        head_width (float, optional): Arrowhead width relative to shaft.
            Defaults to 0.08.
        head_length (float, optional): Arrowhead length relative to shaft.
            Defaults to 0.12.
        label (str, optional): Legend label. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        list: Arrow layers (shaft and head) added to plot.

    Examples:
        Simple vector field:

        >>> x = np.linspace(-1, 1, 10)
        >>> y = np.linspace(-1, 1, 10)
        >>> X, Y = np.meshgrid(x, y)
        >>> U = -Y  # rotational field
        >>> V = X
        >>> gplt.quiver(X.ravel(), Y.ravel(), U.ravel(), V.ravel())
        >>> gplt.show()

        With scaling:

        >>> gplt.quiver(X.ravel(), Y.ravel(), U.ravel(), V.ravel(),
        ...             scale=2.0, width=1.5, color='red')

        Gradient field:

        >>> U = 2*X
        >>> V = 2*Y
        >>> gplt.quiver(X.ravel(), Y.ravel(), U.ravel(), V.ravel(),
        ...             head_width=0.1, head_length=0.15)
    """
    data = kwargs.pop("data", None)
    if data is not None:
        x, y, u, v = _resolve_data_args("quiver", data, x, y, u, v)

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
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    w: ArrayLike,
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
    **kwargs: Any,
) -> list[BaseLayer]:
    """Create a 3D vector field plot.

    Renders arrows in 3D space representing vector field (u, v, w) at each
    point (x, y, z). Supports normalization of vector magnitudes for cleaner
    visualization. Useful for visualizing 3D flow fields and force vectors.

    Args:
        x (array-like): X-positions of vectors. Shape (N,).
        y (array-like): Y-positions of vectors. Shape (N,).
        z (array-like): Z-positions of vectors. Shape (N,).
        u (array-like): X-components of vectors. Shape (N,).
        v (array-like): Y-components of vectors. Shape (N,).
        w (array-like): Z-components of vectors. Shape (N,).
        color (str or tuple, optional): Arrow color. Defaults to 'C0'.
        scale (float, optional): Scaling factor for arrow magnitudes.
            Defaults to 1.0.
        linewidth (float, optional): Arrow shaft line width. Defaults to 0.8.
        head_length (float, optional): Arrowhead length relative to shaft.
            Defaults to 0.18.
        head_width (float, optional): Arrowhead width relative to shaft.
            Defaults to 0.09.
        normalize (bool, optional): If True, normalize all vectors to unit
            magnitude (shows direction only). Defaults to False.
        elev (float, optional): Camera elevation angle. Defaults to 30.0.
        azim (float, optional): Camera azimuth angle. Defaults to -60.0.
        label (str, optional): Legend label. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        list: Arrow layer added to plot.

    Examples:
        Simple 3D vector field:

        >>> x = y = z = np.linspace(-1, 1, 5)
        >>> X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        >>> U = -Y
        >>> V = X
        >>> W = np.zeros_like(X)
        >>> gplt.quiver3d(X.ravel(), Y.ravel(), Z.ravel(),
        ...               U.ravel(), V.ravel(), W.ravel())
        >>> gplt.show()

        With normalization (direction only):

        >>> gplt.quiver3d(X.ravel(), Y.ravel(), Z.ravel(),
        ...               U.ravel(), V.ravel(), W.ravel(),
        ...               normalize=True, scale=0.5)

        Gradient field with custom camera:

        >>> gplt.quiver3d(X.ravel(), Y.ravel(), Z.ravel(),
        ...               U.ravel(), V.ravel(), W.ravel(),
        ...               elev=60, azim=120, color='red')
    """
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
    text_value: Optional[str] = None,
    xy: Optional[Tuple[float, float]] = None,
    xytext: Optional[Tuple[float, float]] = None,
    arrowprops: Optional[dict[str, Any]] = None,
    fontsize: int = 12,
    color: ColorLike = "k",
    **kwargs: Any,
) -> list[BaseLayer]:
    # matplotlib calls this parameter `text`, which cannot be the parameter name here --
    # the body calls the module-level `text()` to draw. So the matplotlib spelling is
    # accepted as a keyword and folded in; `annotate(text="hi", xy=...)` is a real call.
    text_value = _mpl_text_arg("annotate", kwargs.pop("text", None), text_value, "text")
    if xy is None:
        raise TypeError("annotate() missing required argument: 'xy'")
    _warn_unsupported(
        "annotate",
        {
            "xycoords": kwargs.pop("xycoords", None),
            "textcoords": kwargs.pop("textcoords", None),
            "annotation_clip": kwargs.pop("annotation_clip", None),
        },
        {
            "xycoords": "has no effect: GLPlot reads xy in data coordinates only, so an "
            "'axes fraction' or 'figure fraction' point lands at those data values instead",
            "textcoords": "has no effect: GLPlot reads xytext in data coordinates only, so "
            "an 'offset points' pair lands at those data values instead of beside xy",
            "annotation_clip": "has no effect: GLPlot draws the annotation whether or not "
            "xy is inside the current view",
        },
    )
    tx, ty = xy if xytext is None else xytext
    # Whatever is left in kwargs (bbox, ha, va, rotation, ...) is a real matplotlib `Text`
    # property, the same set `text()` itself accepts -- forwarded rather than dropped, so
    # `annotate(..., bbox=dict(...))` gets the same headless-export box `text(bbox=...)`
    # does, instead of the two call shapes silently disagreeing about which one draws it.
    text_layer = text(tx, ty, text_value, fontsize=fontsize, color=color, **kwargs)
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
    data = kwargs.pop("data", None)
    if data is not None:
        y, xmin, xmax = _resolve_data_args("hlines", data, y, xmin, xmax)
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
    data = kwargs.pop("data", None)
    if data is not None:
        x, ymin, ymax = _resolve_data_args("vlines", data, x, ymin, ymax)
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
    data = kwargs.pop("data", None)
    if data is not None:
        x, y = _resolve_data_args("step", data, x, y)
        args = _resolve_plot_data_args("step", data, args)
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


def pcolor(*args: Any, **kwargs: Any):
    """Draw a pseudocolour plot of a 2-D array on a rectangular grid.

    Each cell of the grid is filled with the colour its value maps to. The
    matplotlib-compatible calling conventions are ``pcolor(C)`` and
    ``pcolor(X, Y, C)``.

    Args:
        *args: Either ``C`` alone, or ``X, Y, C``. ``C`` is the 2-D array of
            values; ``X`` and ``Y`` are the cell corners (one larger than ``C``
            in each dimension, or the same size, as matplotlib allows).
        cmap (str, optional): Colormap. Defaults to the current one, else viridis.
        norm, vmin, vmax: Colour scaling, as in :func:`scatter`.
        alpha (float, optional): Transparency.
        label (str, optional): Legend label.

    Returns:
        Layer: One patch layer holding every cell, coloured per cell.

    Note:
        Empty of NaN cells are dropped, and the whole mesh is one layer with a
        per-vertex colour buffer rather than matplotlib's QuadMesh.

    Examples:
        >>> gplt.pcolor(Z)
        >>> gplt.pcolor(X, Y, Z, cmap='hot')
    """
    return _pcolor_impl("pcolor", args, kwargs)


def pcolormesh(*args: Any, **kwargs: Any):
    """Draw a pseudocolour plot -- the faster-drawing sibling of :func:`pcolor`.

    In matplotlib ``pcolormesh`` returns a QuadMesh and ``pcolor`` a slower
    PolyCollection. GLPlot draws both as one per-vertex-coloured patch, so they
    behave identically here. See :func:`pcolor`.
    """
    return _pcolor_impl("pcolormesh", args, kwargs)


def _pcolor_impl(func: str, args: tuple, kwargs: dict):
    pass

    cmap = kwargs.pop("cmap", None)
    norm = kwargs.pop("norm", None)
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    alpha = kwargs.pop("alpha", None)
    label = kwargs.pop("label", None)
    shading = kwargs.pop("shading", "flat")

    if len(args) == 1:
        C = _as_float_array(args[0], ndim=2, name="C")
        ny, nx = C.shape
        X, Y = np.meshgrid(np.arange(nx + 1, dtype=np.float64), np.arange(ny + 1, dtype=np.float64))
    elif len(args) == 3:
        X = np.asarray(args[0], dtype=np.float64)
        Y = np.asarray(args[1], dtype=np.float64)
        C = _as_float_array(args[2], ndim=2, name="C")
        ny, nx = C.shape
        if X.ndim == 1 and Y.ndim == 1:
            X, Y = np.meshgrid(X, Y)
        # matplotlib accepts corners one larger than C ('flat') or the same size
        # ('nearest'/'gouraud'); pad the same-size case out to corners so every cell has
        # four, rather than reject a spelling the caller reasonably used.
        if X.shape == (ny, nx):
            X = _corners_from_centres(X)
            Y = _corners_from_centres(Y)
        elif X.shape != (ny + 1, nx + 1):
            raise ValueError(
                f"{func}(): X, Y must be C's shape or one larger in each dim, "
                f"got {X.shape} for C {C.shape}"
            )
    else:
        raise TypeError(f"{func}() takes C or X, Y, C -- got {len(args)} positional args")

    _warn_unsupported(
        func,
        {"shading": shading if shading not in ("flat", "auto") else None},
        {"shading": "is not supported; every cell is drawn flat-shaded"},
    )

    values = C.ravel()
    finite = np.isfinite(values)

    # One quad per finite cell. Built vectorised off the corner grid: cell (j, i) has
    # corners (j,i), (j,i+1), (j+1,i+1), (j+1,i) -- four vertices carrying that cell's
    # colour, two triangles.
    jj, ii = np.divmod(np.flatnonzero(finite), nx)

    def corner(dj, di):
        return np.column_stack([X[jj + dj, ii + di], Y[jj + dj, ii + di]])

    quads = np.stack([corner(0, 0), corner(0, 1), corner(1, 1), corner(1, 0)], axis=1)
    verts = quads.reshape(-1, 2).astype(np.float32)
    base = np.arange(len(jj), dtype=np.uint32) * 4
    indices = np.column_stack([base, base + 1, base + 2, base, base + 2, base + 3]).ravel()

    plot_obj = _get_or_create_plot()
    add_patch(
        verts,
        indices=indices,
        mode="triangles",
        colors=np.zeros((len(verts), 4), dtype=np.float32),
        label=label,
    )
    layer = plot_obj.scene.layers[-1]
    # Four colour rows per finite cell, each carrying that cell's value. NaN cells were
    # dropped from the geometry, so only the finite values are painted.
    _paint_patch_mappable(layer, np.repeat(values[finite], 4), cmap, norm, vmin, vmax, alpha)
    layer.metadata.update({"artist": func, "cvalues": values})
    _set_dirty(plot_obj)
    return _set_current_mappable(layer)


def _corners_from_centres(centres: np.ndarray) -> np.ndarray:
    """Grow an (ny, nx) grid of cell centres to the (ny+1, nx+1) corners around them.

    Midpoints between adjacent centres, with the outer border extrapolated by half a cell
    so the edge cells are as wide as their neighbours rather than collapsing to a line.
    """

    def edges_1d(a: np.ndarray) -> np.ndarray:
        mid = 0.5 * (a[:-1] + a[1:])
        return np.concatenate(([a[0] - (mid[0] - a[0])], mid, [a[-1] + (a[-1] - mid[-1])]))

    ny, nx = centres.shape
    # Expand along each axis independently: the row direction, then the column direction.
    rows = np.apply_along_axis(edges_1d, 1, centres)  # (ny, nx+1)
    full = np.apply_along_axis(edges_1d, 0, rows)  # (ny+1, nx+1)
    return full


def _triangulation(func: str, args: tuple):
    """Resolve matplotlib's triangulation call forms to ``(x, y, triangles)``.

    Accepts ``(x, y)`` -- Delaunay-triangulated via ``matplotlib.tri`` -- or
    ``(x, y, triangles)`` with an explicit ``(M, 3)`` index array. The one place the
    triplot/tripcolor/tricontour family agrees on what its leading arguments mean.
    """
    from matplotlib.tri import Triangulation

    if len(args) >= 3 and np.ndim(args[2]) == 2:
        x = _as_float_array(args[0], ndim=1, name="x")
        y = _as_float_array(args[1], ndim=1, name="y")
        tris = np.asarray(args[2], dtype=np.int64)
        rest = args[3:]
    elif len(args) >= 1 and isinstance(args[0], Triangulation):
        tri = args[0]
        x, y, tris = tri.x, tri.y, tri.triangles
        rest = args[1:]
    else:
        x = _as_float_array(args[0], ndim=1, name="x")
        y = _as_float_array(args[1], ndim=1, name="y")
        tris = Triangulation(x, y).triangles
        rest = args[2:]
    return np.asarray(x), np.asarray(y), np.asarray(tris), rest


def triplot(
    *args: Any, color: ColorLike = "C0", linewidth: float = 1.0, label: Optional[str] = None
):
    """Draw the edges of an unstructured triangular mesh.

    Args:
        *args: ``x, y`` (Delaunay-triangulated), ``x, y, triangles``, or a
            ``matplotlib.tri.Triangulation``.
        color (str or tuple, optional): Edge colour.
        linewidth (float, optional): Edge width.
        label (str, optional): Legend label.

    Returns:
        Layer: One polyline layer tracing every triangle edge.

    Examples:
        >>> gplt.triplot(x, y)                    # Delaunay
        >>> gplt.triplot(x, y, triangles)
    """
    x, y, tris, _ = _triangulation("triplot", args)
    # Every edge of every triangle, as one NaN-separated... except polylines here do not
    # break on NaN, so the edges are drawn as three separate two-point segments per
    # triangle through vlines-style plotting would be N layers. Instead: one polyline that
    # walks each triangle's closed outline and lifts between them by retracing, which for a
    # mesh reads as the wireframe matplotlib draws.
    segs_x, segs_y = [], []
    for a, b, c in tris:
        segs_x.extend([x[a], x[b], x[c], x[a]])
        segs_y.extend([y[a], y[b], y[c], y[a]])
        # Retrace the last point so the next triangle's opening edge is not connected to
        # this one -- the pen "lifts" by drawing a zero-length step the eye does not see.
        segs_x.append(x[a])
        segs_y.append(y[a])
    artists = plot(segs_x, segs_y, color=color, linewidth=linewidth, label=label)
    for artist in artists:
        artist.metadata["artist"] = "triplot"
    return artists[-1] if artists else None


def tripcolor(*args: Any, **kwargs: Any):
    """Pseudocolour an unstructured triangular mesh.

    Args:
        *args: ``x, y, C`` (Delaunay), ``x, y, triangles, C``, or a
            ``Triangulation`` followed by ``C``. ``C`` is one value per point
            (Gouraud-style, interpolated across each triangle) or one per triangle
            (flat).
        cmap, norm, vmin, vmax, alpha: Colour scaling, as in :func:`scatter`.
        label (str, optional): Legend label.

    Returns:
        Layer: One per-vertex-coloured patch layer.

    Examples:
        >>> gplt.tripcolor(x, y, values)
    """
    data = kwargs.pop("data", None)
    if data is not None:
        args = _resolve_plot_data_args("tripcolor", data, args)
    _warn_unsupported(
        "tripcolor",
        {"shading": kwargs.pop("shading", None)},
        {
            "shading": "is not supported; every triangle is drawn flat-shaded from its "
            "vertex colours"
        },
    )

    cmap = kwargs.pop("cmap", None)
    norm = kwargs.pop("norm", None)
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    alpha = kwargs.pop("alpha", None)
    label = kwargs.pop("label", None)

    x, y, tris, rest = _triangulation("tripcolor", args)
    if not rest:
        raise TypeError("tripcolor() needs C -- one value per point or per triangle")
    C = _as_float_array(rest[0], ndim=1, name="C")

    verts = np.column_stack([x, y])[tris.ravel()].astype(np.float32)
    if len(C) == len(x):
        # Per-point values: each triangle vertex carries its own, so the GPU interpolates
        # across the face -- matplotlib's 'gouraud' shading, for free from the colour VBO.
        # The scalar per colour-vertex is the value of the point that vertex came from.
        per_vertex = C[tris.ravel()]
    elif len(C) == len(tris):
        # Per-triangle values: flat shading, so all three vertices of a face share it.
        per_vertex = np.repeat(C, 3)
    else:
        raise ValueError(
            f"tripcolor(): C must have one value per point ({len(x)}) or per triangle "
            f"({len(tris)}), got {len(C)}"
        )

    indices = np.arange(len(verts), dtype=np.uint32)
    plot_obj = _get_or_create_plot()
    add_patch(
        verts,
        indices=indices,
        mode="triangles",
        colors=np.zeros((len(verts), 4), dtype=np.float32),
        label=label,
    )
    layer = plot_obj.scene.layers[-1]
    _paint_patch_mappable(layer, per_vertex, cmap, norm, vmin, vmax, alpha)
    layer.metadata.update({"artist": "tripcolor", "cvalues": C})
    _set_dirty(plot_obj)
    return _set_current_mappable(layer)


def stairs(
    values: Sequence[float],
    edges: Optional[Sequence[float]] = None,
    *,
    orientation: str = "vertical",
    baseline: Optional[float] = 0.0,
    fill: bool = False,
    color: ColorLike = "C0",
    linewidth: float = 1.5,
    label: Optional[str] = None,
    data: Optional[Any] = None,
):
    """Draw a step function defined by ``len(values) + 1`` edges.

    Unlike :func:`step`, which steps *between* data points, ``stairs`` takes the
    step heights and the edges directly -- the shape :func:`hist` returns, and the
    natural way to draw a precomputed histogram.

    Args:
        values (array-like): The height of each step. Length N.
        edges (array-like, optional): The N+1 boundaries between steps. Defaults
            to ``0, 1, ..., N``.
        orientation (str, optional): 'vertical' (default) or 'horizontal'.
        baseline (float, optional): The level a filled stair drops to, and where
            the outline closes. None leaves the outline open. Defaults to 0.
        fill (bool, optional): Fill under the steps instead of outlining them.
        color (str or tuple, optional): Line or fill colour.
        linewidth (float, optional): Outline width. Ignored when ``fill``.
        label (str, optional): Legend label.
        data (indexable, optional): If given, ``values`` and ``edges`` may be
            keys into it.

    Returns:
        Layer: The polyline (or patch, when ``fill``) drawn.

    Examples:
        >>> counts, bins = np.histogram(data)
        >>> gplt.stairs(counts, bins)
        >>> gplt.stairs(counts, bins, fill=True)
    """
    values, edges = _resolve_data_args("stairs", data, values, edges)
    v = _as_float_array(values, ndim=1, name="values")
    if edges is None:
        e = np.arange(len(v) + 1, dtype=np.float64)
    else:
        e = _as_float_array(edges, ndim=1, name="edges")
        if len(e) != len(v) + 1:
            raise ValueError(
                f"edges must have one more entry than values ({len(v)+1}), got {len(e)}"
            )
    if orientation not in ("vertical", "horizontal"):
        raise ValueError(f"unsupported orientation: {orientation!r}")

    # The staircase outline: each edge appears twice, each value twice, so the line goes
    # along a step then jumps at the boundary. This is the same shape `hist(histtype=step)`
    # builds, kept here rather than shared because that one closes to a baseline at both
    # ends and this one's baseline is optional.
    xs = np.repeat(e, 2)[1:-1]
    ys = np.repeat(v, 2)

    if fill:
        base = 0.0 if baseline is None else float(baseline)
        # Close the ribbon down to the baseline and back: a filled stair is the area under
        # the steps, which `fill_between` draws exactly if handed the doubled edges.
        return fill_between(
            xs if orientation == "vertical" else ys,
            ys if orientation == "vertical" else xs,
            base,
            color=color,
            label=label,
        )

    if baseline is not None:
        # Drop to the baseline at both ends so the outline reads as a closed silhouette
        # rather than two loose verticals, which is how matplotlib draws it.
        xs = np.concatenate(([e[0]], xs, [e[-1]]))
        ys = np.concatenate(([baseline], ys, [baseline]))

    px, py = (xs, ys) if orientation == "vertical" else (ys, xs)
    artists = plot(px, py, color=color, linewidth=linewidth, label=label)
    for artist in artists:
        artist.metadata["artist"] = "stairs"
    return artists[-1] if artists else None


def ecdf(
    x: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    *,
    complementary: bool = False,
    orientation: str = "vertical",
    compress: bool = False,
    color: ColorLike = "C0",
    linewidth: float = 1.5,
    label: Optional[str] = None,
    data: Optional[Any] = None,
):
    """Draw the empirical cumulative distribution function of ``x``.

    The fraction of the data at or below each value, as a staircase rising from 0
    to 1 -- a box plot's information without binning.

    Args:
        x (array-like): The sample.
        weights (array-like, optional): Per-point weights.
        complementary (bool, optional): Draw ``1 - ECDF``, the survival function.
        orientation (str, optional): 'vertical' (default) plots the cumulative
            fraction on y; 'horizontal' on x.
        compress (bool, optional): Merge repeated values into a single step, so a
            heavily-tied sample draws a handful of vertices instead of one per point.
            Defaults to False.
        color (str or tuple, optional): Line colour.
        linewidth (float, optional): Line width.
        label (str, optional): Legend label.
        data (indexable, optional): If given, ``x`` and ``weights`` may be keys
            into it.

    Returns:
        Layer: The staircase polyline.

    Examples:
        >>> gplt.ecdf(samples)
        >>> gplt.ecdf(samples, complementary=True)     # survival function
    """
    x, weights = _resolve_data_args("ecdf", data, x, weights)
    values = _as_float_array(x, ndim=1, name="x")
    finite = np.isfinite(values)
    values = values[finite]
    if len(values) == 0:
        raise ValueError("ecdf(): x has no finite values")

    order = np.argsort(values)
    xs = values[order]
    if weights is None:
        cum = np.arange(1, len(xs) + 1, dtype=np.float64) / len(xs)
    else:
        w = _as_float_array(weights, ndim=1, name="weights")[finite][order]
        cum = np.cumsum(w) / w.sum()
    if complementary:
        cum = 1.0 - cum

    if compress:
        # Ties are one jump, not several of height zero. Keeping the *last* row of each
        # run is what makes the step land at the full cumulative value: keeping the first
        # would draw the staircase one tie short at every repeated value.
        keep = np.concatenate((xs[1:] != xs[:-1], [True]))
        xs, cum = xs[keep], cum[keep]

    # A left-closed staircase: the CDF is flat until a sample, then jumps at it, so `edges`
    # are the sorted values bracketed by the data range and `stairs` draws the rest.
    edges = np.concatenate(([xs[0]], xs))
    layer = stairs(
        cum,
        edges,
        orientation=orientation,
        baseline=None,
        color=color,
        linewidth=linewidth,
        label=label,
    )
    if layer is not None:
        layer.metadata["artist"] = "ecdf"
    return layer


def _errorevery_mask(errorevery, n: int) -> np.ndarray:
    """Which points get an error bar, from matplotlib's ``errorevery``.

    Accepts every form matplotlib does: ``N`` (every N-th), ``(start, N)`` (every N-th
    from an offset, so two overlapping series can interleave their bars instead of
    drawing on top of each other), a slice, or an explicit boolean mask.
    """
    mask = np.zeros(n, dtype=bool)
    if isinstance(errorevery, Integral):
        if errorevery < 1:
            raise ValueError("errorbar(): errorevery must be positive")
        mask[:: int(errorevery)] = True
    elif isinstance(errorevery, tuple):
        start, step = errorevery
        if not isinstance(start, Integral) or not isinstance(step, Integral) or step < 1:
            raise ValueError("errorbar(): errorevery tuple must be (int start, int step >= 1)")
        mask[int(start) :: int(step)] = True
    elif isinstance(errorevery, slice):
        mask[errorevery] = True
    else:
        candidate = np.asarray(errorevery)
        if candidate.dtype != bool or candidate.shape != (n,):
            raise ValueError(
                f"errorbar(): errorevery must be an int, a (start, step) tuple, a slice, "
                f"or a length-{n} boolean mask"
            )
        mask = candidate
    return mask


def errorbar(
    x,
    y,
    yerr=None,
    xerr=None,
    fmt: str = "",
    ecolor: Optional[ColorLike] = None,
    elinewidth: Optional[float] = None,
    elinestyle: Optional[str] = None,
    capsize: Optional[float] = None,
    barsabove: bool = False,
    lolims: Any = False,
    uplims: Any = False,
    xlolims: Any = False,
    xuplims: Any = False,
    errorevery: Any = 1,
    capthick: Optional[float] = None,
    label: Optional[str] = None,
    *,
    data: Optional[Any] = None,
    **kwargs,
):
    """Plot ``y`` against ``x`` with error bars.

    Args:
        x, y (array-like): The data points.
        yerr, xerr (float or array-like, optional): Error magnitudes. A scalar or a
            length-N array is symmetric; a ``(2, N)`` array is ``(lower, upper)``.
        fmt (str, optional): Format string for the data itself, as in :func:`plot`.
            ``''`` draws markers only; ``'none'`` draws no data line at all.
        ecolor (str or tuple, optional): Error bar colour. Defaults to the data colour.
        elinewidth (float, optional): Error bar line width. Defaults to 1.
        capsize (float, optional): Half-length of the caps, in data units. Defaults to 0
            (no caps).
        barsabove (bool, optional): Draw the bars on top of the data markers rather than
            beneath them. Defaults to False.
        lolims, uplims, xlolims, xuplims (bool or array-like, optional): Mark points whose
            value is only a lower or upper limit. The bar becomes one-sided and a caret
            marks the open end, as in matplotlib.
        errorevery (int, tuple, slice or bool array, optional): Draw a bar on only some
            points -- ``N`` for every N-th, ``(start, N)`` to offset the pattern so two
            series interleave. Defaults to 1.
        capthick (float, optional): Cap line width. Defaults to ``elinewidth``.
        elinestyle (str, optional): Accepted for matplotlib parity; ignored. Error bar
            lines are always drawn solid.
        label (str, optional): Legend label.
        data (indexable, optional): If given, the arrays may be keys into it.
        **kwargs: Forwarded to the data line -- ``color``, ``marker``, ``linestyle``...

    Returns:
        list: Every layer drawn, tagged ``artist_group='errorbar'``.

    Examples:
        >>> gplt.errorbar(x, y, yerr=sigma, fmt='o', capsize=0.2)
        >>> gplt.errorbar(x, y, yerr=sigma, uplims=True)        # upper-limit markers
        >>> gplt.errorbar(x, y, yerr=sigma, errorevery=(0, 5))  # every 5th point
    """
    x, y, xerr, yerr = _resolve_data_args("errorbar", data, x, y, xerr, yerr)
    _warn_unsupported(
        "errorbar",
        {"elinestyle": elinestyle},
        {"elinestyle": "has no effect: error bar lines are always drawn solid"},
    )
    x_arr = _as_float_array(x, ndim=1, name="x")
    y_arr = _as_float_array(y, ndim=1, name="y")
    if len(x_arr) != len(y_arr):
        raise ValueError("x and y must have the same length")

    n = len(x_arr)
    every = _errorevery_mask(errorevery, n)
    elinewidth = 1.0 if elinewidth is None else float(elinewidth)
    cap_lw = elinewidth if capthick is None else float(capthick)
    capsize = 0.0 if capsize is None else float(capsize)
    err_color = ecolor if ecolor is not None else kwargs.get("color", "k")
    artists: list = []

    def _bars(err, dep, indep, lo_flags, hi_flags, bar_func, cap_func, caret_lo, caret_hi):
        """One axis's worth of bars, caps and limit carets.

        Written once and called twice with the roles of x and y exchanged: the geometry is
        identical under the swap, and duplicating it is how the x branch ends up quietly
        missing a feature the y branch has.
        """
        e = (
            np.broadcast_to(np.asarray(err, dtype=np.float64), (2, n))
            if np.ndim(err) == 2
            else np.broadcast_to(np.asarray(err, dtype=np.float64), (n,))
        )
        lower_err, upper_err = (e[0], e[1]) if e.ndim == 2 else (e, e)
        if np.any(lower_err < 0) or np.any(upper_err < 0):
            raise ValueError("errorbar(): error values must not be negative")

        lo_flags = np.broadcast_to(np.asarray(lo_flags), (n,)).astype(bool)
        hi_flags = np.broadcast_to(np.asarray(hi_flags), (n,)).astype(bool)
        # A limit flag truncates the bar on that side: the point *is* the bound, so the
        # bar only reaches out in the direction the true value could lie.
        low = dep - lower_err * (~lo_flags)
        high = dep + upper_err * (~hi_flags)

        for i in np.nonzero(every)[0]:
            artists.extend(
                bar_func(indep[i], low[i], high[i], colors=err_color, linewidth=elinewidth)
            )
            if capsize:
                # Only points with no limit flag get a pair of caps; a limit end is marked
                # by its caret instead, which is what says "this bound is open".
                ends = [
                    e for e, flag in ((low[i], lo_flags[i]), (high[i], hi_flags[i])) if not flag
                ]
                if ends:
                    artists.extend(
                        cap_func(
                            ends,
                            indep[i] - capsize,
                            indep[i] + capsize,
                            colors=err_color,
                            linewidth=cap_lw,
                        )
                    )
        for flags, ends, caret in ((lo_flags, high, caret_lo), (hi_flags, low, caret_hi)):
            sel = flags & every
            if sel.any():
                a, b = (indep[sel], ends[sel]) if bar_func is vlines else (ends[sel], indep[sel])
                artists.append(scatter(a, b, color=err_color, marker=caret, s=8.0))

    def _data_layer():
        if fmt in (None, "", "none", "None"):
            artists.append(scatter(x_arr, y_arr, label=label, **kwargs))
        else:
            artists.extend(plot(x_arr, y_arr, fmt, label=label, **kwargs))

    if barsabove:
        _data_layer()
    if yerr is not None:
        _bars(yerr, y_arr, x_arr, lolims, uplims, vlines, hlines, "^", "v")
    if xerr is not None:
        _bars(xerr, x_arr, y_arr, xlolims, xuplims, hlines, vlines, ">", "<")
    if not barsabove:
        _data_layer()

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
    orientation: str = "vertical",
    *,
    data: Optional[Any] = None,
):
    """Draw a stem plot: a marker per point, dropped to a baseline.

    Args:
        x (array-like): Stem positions, or the values themselves when ``y`` is None.
        y (array-like, optional): Stem heights.
        linefmt (str, optional): Format string for the stems. Defaults to 'C0-'.
        markerfmt (str, optional): Format string for the head markers.
        basefmt (str, optional): Format string for the baseline. Defaults to 'k-'.
        bottom (float, optional): The baseline the stems drop to. Defaults to 0.
        label (str, optional): Legend label.
        orientation (str, optional): Accepted for matplotlib parity. GLPlot has no
            horizontal stem renderer, so 'horizontal' is ignored.
        data (indexable, optional): If given, ``x`` and ``y`` may be keys into it.
    """
    x, y = _resolve_data_args("stem", data, x, y)
    _warn_unsupported(
        "stem",
        {"orientation": orientation if orientation != "vertical" else None},
        {
            "orientation": "has no effect: GLPlot has no horizontal stem renderer, so the "
            "stems are drawn vertically"
        },
    )
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
    fontdict: Optional[dict] = None,
    fontsize: Optional[int] = None,
    color: Optional[ColorLike] = None,
    label: Optional[str] = None,
    *,
    bbox: Optional[dict] = None,
    ha: Optional[str] = None,
    va: Optional[str] = None,
    horizontalalignment: Optional[str] = None,
    verticalalignment: Optional[str] = None,
    rotation: Optional[Any] = None,
    **kwargs: Any,
) -> GPULinePlot:
    """Add a text string at specified plot coordinates.

    Renders text at data coordinates (x, y). Useful for annotations,
    labels, or callouts on the plot. Note that rendering quality and
    styling depends on backend capabilities.

    Args:
        x (float): X-coordinate for text placement.
        y (float): Y-coordinate for text placement.
        s (str): Text string to display. Can include newlines.
        fontdict (dict, optional): matplotlib parity. Merged into the styling
            below, with the explicit ``fontsize``/``color`` kwargs winning.
        fontsize (int, optional): Font size in points. Defaults to 12.
        color (str or tuple, optional): Text color as RGBA or named color.
            Defaults to black.
        label (str, optional): Legend label (optional). Defaults to None.
        bbox (dict, optional): matplotlib ``FancyBboxPatch`` properties (``facecolor``,
            ``edgecolor``, ``alpha``, ``boxstyle``, ``pad``, ...) for a background box
            behind the text. Only drawn in the headless PNG export -- ``render_preview``
            reconstructs this text as a real matplotlib ``Text`` and can give it a real
            box there; the live GL/imgui text renderer has no background-box primitive
            yet, so the box does not appear in an interactive ``show()`` window.
        ha, va, horizontalalignment, verticalalignment, rotation: Accepted for
            matplotlib parity. Ignored -- GLPlot draws text left-aligned, baseline-
            aligned, and horizontal only.
        **kwargs: Any other matplotlib ``Text`` property (``zorder``, ``alpha``,
            ``family``, ``transform``, ...). Accepted so a pasted matplotlib call does
            not raise; ignored.

    Returns:
        GPULinePlot: The plot object.

    Examples:
        Add text annotation:

        >>> gplt.plot([1, 2, 3], [1, 4, 2])
        >>> gplt.text(2, 4, "Peak", fontsize=14, color='red')
        >>> gplt.show()

        Multiple annotations:

        >>> gplt.text(1, 1, "Start", color='blue', fontsize=10)
        >>> gplt.text(3, 2, "End", color='green', fontsize=10)

        With a background box (headless export only):

        >>> gplt.text(1, 1, "Note", bbox=dict(boxstyle="round", facecolor="white"))
    """
    plot_obj = _get_or_create_plot()
    fontsize, color = _merge_fontdict("text", fontdict, fontsize, color)
    ha = ha if ha is not None else horizontalalignment
    va = va if va is not None else verticalalignment
    _warn_unsupported(
        "text",
        {"ha": ha, "va": va, "rotation": rotation, **kwargs},
        {
            "ha": "has no effect: text is always drawn left-aligned",
            "va": "has no effect: text is always drawn baseline-aligned",
            "rotation": "has no effect: GLPlot draws text horizontally only",
        },
    )

    # The defaults live here rather than in the signature so that "unset" stays
    # distinguishable from "explicitly 12pt black" long enough for `fontdict` to fill it.
    rgba = _normalize_rgba(color if color is not None else (0.0, 0.0, 0.0, 1.0), n=None)
    # A single date (`text(datetime.date(...), y, ...)`, the natural way to annotate a
    # point on a date-axis plot) needs the same date->number conversion `plot()`/
    # `scatter()` get; `.reshape(-1)[0]` rather than a bare `float(...)` because
    # `_coerce_axis_values` always returns an array-shaped result, even for scalar input.
    x_val = float(np.asarray(_coerce_axis_values(x, "x", "text")).reshape(-1)[0])
    y_val = float(np.asarray(_coerce_axis_values(y, "y", "text")).reshape(-1)[0])
    plot_obj.add_text(
        x_val,
        y_val,
        str(s),
        fontsize=int(fontsize if fontsize is not None else 12),
        color=rgba,
        label=label,
    )
    if bbox is not None:
        plot_obj.scene.layers[-1].metadata["bbox"] = bbox
        _warn_unsupported_call(
            "text",
            "bbox=... is drawn only in the headless PNG export (render_preview "
            "reconstructs the text as a real matplotlib Text with a real box); the live "
            "GL/imgui text renderer has no background-box primitive yet",
        )
    _set_dirty(plot_obj)
    return plot_obj


def _flatten_path(path: Any, samples: int = 24) -> np.ndarray:
    """The outline of a matplotlib ``Path`` as a ring of points, curves included.

    matplotlib's own ``to_polygons()`` / ``cleaned(curves=False)`` flatten a Bezier far too
    coarsely for this: a ``Circle`` comes back as a 16-gon whose vertices sit 2.5% *outside*
    the true radius, because the coarse conversion keeps points near the control polygon.
    matplotlib gets away with it by re-subdividing in the Agg backend at device resolution;
    GLPlot uploads these vertices to the GPU once, so they have to be right here. Evaluating
    the cubic and quadratic segments directly puts every point exactly on the curve.
    """
    from matplotlib.path import Path

    points: List[np.ndarray] = []
    current = np.zeros(2)
    t = np.linspace(0.0, 1.0, samples + 1)[1:, None]
    for vertices, code in path.iter_segments(curves=True):
        if code == Path.MOVETO:
            current = np.asarray(vertices[:2], dtype=np.float64)
            points.append(current)
        elif code == Path.LINETO:
            current = np.asarray(vertices[:2], dtype=np.float64)
            points.append(current)
        elif code == Path.CURVE3:
            c, end = vertices[0:2], vertices[2:4]
            points.append(((1 - t) ** 2) * current + 2 * (1 - t) * t * c + (t**2) * end)
            current = np.asarray(end, dtype=np.float64)
        elif code == Path.CURVE4:
            c1, c2, end = vertices[0:2], vertices[2:4], vertices[4:6]
            points.append(
                ((1 - t) ** 3) * current
                + 3 * ((1 - t) ** 2) * t * c1
                + 3 * (1 - t) * (t**2) * c2
                + (t**3) * end
            )
            current = np.asarray(end, dtype=np.float64)
        elif code == Path.CLOSEPOLY:
            break
    if not points:
        return np.zeros((0, 2))
    return np.vstack([np.atleast_2d(p) for p in points])


def _patch_object_to_geometry(patch: Any) -> dict:
    """Flatten a ``matplotlib.patches.Patch`` into the arguments `add_patch` takes.

    ``ax.add_patch(Rectangle((0, 0), 1, 1, fc='red'))`` is how matplotlib scripts draw a
    shape, and a Patch already knows how to describe itself: ``get_path()`` plus
    ``get_patch_transform()`` gives the outline in data coordinates for *every* subclass --
    Rectangle, Circle, Ellipse, Wedge, Arc, Polygon -- so there is one conversion here
    rather than one per shape, and a subclass this file has never heard of still works.

    The outline is tessellated as a fan from its centroid. That is exact for the convex
    shapes (all of the above) and, for a concave Polygon a caller builds by hand, the
    triangles can spill outside the outline -- a real limitation of not having a general
    tessellator, and the reason the fan is not silently claimed to be a polygon filler.
    """
    path = patch.get_path().transformed(patch.get_patch_transform())
    # Curves (Circle, Ellipse, Wedge, FancyBboxPatch) are Bezier segments in the raw path.
    ring = _flatten_path(path)
    if len(ring) < 3:
        raise ValueError(f"add_patch(): {type(patch).__name__} has no fillable outline")
    if np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]  # Drop the repeated closing vertex the path carries.

    centre = ring.mean(axis=0)
    verts = np.vstack([centre, ring]).astype(np.float32)
    n = len(ring)
    idx = np.empty(n * 3, dtype=np.uint32)
    idx[0::3] = 0
    idx[1::3] = np.arange(1, n + 1)
    idx[2::3] = np.arange(1, n + 1) % n + 1

    face = patch.get_facecolor()
    edge = patch.get_edgecolor()
    # A Patch with fill=False reports a fully transparent face, which is exactly what
    # should reach the renderer -- outline only.
    return {
        "vertices": verts,
        "indices": idx,
        "mode": "triangles",
        "face_color": tuple(float(v) for v in face) if face is not None else None,
        "edge_color": tuple(float(v) for v in edge) if edge is not None else None,
        "label": patch.get_label() or None,
    }


def add_patch(
    vertices: Union[np.ndarray, Sequence, Any],
    indices: Optional[np.ndarray] = None,
    mode: str = "strip",
    face_color: Optional[ColorLike] = None,
    edge_color: Optional[ColorLike] = None,
    label: Optional[str] = None,
    colors: Optional[np.ndarray] = None,
) -> GPULinePlot:
    """Add a geometric patch (filled shape) to the plot.

    Two calling forms. matplotlib's is a patch *object* --
    ``add_patch(Rectangle((0, 0), 1, 1, facecolor='red'))`` -- and any
    :class:`matplotlib.patches.Patch` subclass works, including the curved ones.
    GLPlot's own form is raw geometry: vertices plus optional indices and a draw mode,
    for shapes no patch class covers.

    Args:
        vertices (Patch or array-like): A ``matplotlib.patches.Patch`` (in which case every
            other argument is read off it), or a 2D array of vertex coordinates,
            shape (N, 2).
        indices (array-like, optional): Triangle or line indices. Shape (M,).
            If None with mode='strip', vertices form a continuous strip.
            For mode='triangles', indices must be provided and form triangles
            (triples of indices). Defaults to None.
        mode (str, optional): Drawing mode:
            - 'strip': Triangle strip from consecutive vertices
            - 'triangles': Indexed triangles
            - 'lines': Line segments (edges)
            Defaults to 'strip'.
        face_color (str or tuple, optional): One fill color for the whole patch,
            as RGBA or a named color. Defaults to None (no fill).
        edge_color (str or tuple, optional): Edge/outline color. Defaults to None.
        colors (array-like, optional): Per-vertex RGBA, shape (N, 4) -- one row
            per vertex. For a patch whose pieces are not all one colour, which a
            single ``face_color`` cannot express: a hexbin's hexagons each carry
            their own count. Takes precedence over ``face_color``, and a patch
            with ``colors`` and no ``face_color`` still draws.
        label (str, optional): Legend label. Defaults to None.

    Returns:
        GPULinePlot: The plot object.

    Examples:
        Triangle from vertices:

        >>> verts = [[0, 0], [1, 0], [0.5, 1]]
        >>> gplt.add_patch(verts, mode='triangles', face_color='blue')

        Triangle strip:

        >>> verts = [[0, 0], [1, 0], [0, 1], [1, 1]]
        >>> gplt.add_patch(verts, mode='strip', face_color='red', alpha=0.5)

        Custom indexed shape:

        >>> verts = [[0, 0], [1, 0], [1, 1], [0, 1]]
        >>> idx = [0, 1, 2, 0, 2, 3]  # Two triangles forming a quad
        >>> gplt.add_patch(verts, indices=idx, mode='triangles',
        ...                face_color='green', edge_color='black')

        A matplotlib patch object:

        >>> gplt.add_patch(gplt.Rectangle((0, 0), 2, 1, facecolor='red', alpha=0.4))
        >>> gplt.add_patch(gplt.Circle((1, 1), 0.5, fill=False, edgecolor='k'))
    """
    plot_obj = _get_or_create_plot()

    # The matplotlib form. Detected by behaviour rather than isinstance so a third-party
    # Patch subclass -- or one from a matplotlib newer than this file knows about -- takes
    # the same route.
    if hasattr(vertices, "get_path") and hasattr(vertices, "get_patch_transform"):
        geometry = _patch_object_to_geometry(vertices)
        vertices = geometry["vertices"]
        indices = geometry["indices"] if indices is None else indices
        mode = geometry["mode"]
        face_color = geometry["face_color"] if face_color is None else face_color
        edge_color = geometry["edge_color"] if edge_color is None else edge_color
        label = geometry["label"] if label is None else label

    verts = _as_float_array(vertices, ndim=2, name="vertices")
    f_col = _normalize_rgba(face_color, n=None) if face_color is not None else None
    e_col = _normalize_rgba(edge_color, n=None) if edge_color is not None else None

    cols = None
    if colors is not None:
        cols = _as_float_array(colors, ndim=2, name="colors")
        if cols.shape != (len(verts), 4):
            raise ValueError(
                f"colors must be one RGBA per vertex -- shape ({len(verts)}, 4), "
                f"got {cols.shape}"
            )

    plot_obj.add_patch(
        verts,
        indices=indices,
        mode=mode,
        face_color=tuple(f_col) if f_col is not None else None,
        edge_color=tuple(e_col) if e_col is not None else None,
        label=label,
        colors=cols,
    )
    _set_dirty(plot_obj)
    return plot_obj


# ------------------------------------------------------------------
# View / styling / policies
# ------------------------------------------------------------------


def title(
    label: Optional[str] = None,
    fontdict: Optional[dict] = None,
    loc: Optional[str] = None,
    pad: Optional[float] = None,
    *,
    y: Optional[float] = None,
    fontsize: Optional[float] = None,
    color: Optional[ColorLike] = None,
    s: Optional[str] = None,
) -> None:
    """Set the figure title.

    Sets the main title displayed at the top of the plot window.

    Args:
        label (str): Title text. Can include special characters and unicode.
        fontdict (dict, optional): matplotlib parity. Merged into the styling
            below, with the explicit ``fontsize``/``color`` kwargs winning.
        loc (str, optional): Accepted for matplotlib parity. The title is always
            centred over the frame. Ignored.
        pad (float, optional): Accepted for matplotlib parity. The title is
            centred in the top gutter; widen ``options.axis_margin_t`` instead.
            Ignored.
        y (float, optional): Accepted for matplotlib parity. Ignored, as ``pad``
            is.
        fontsize (float, optional): Title size in points. Defaults to
            matplotlib's ``axes.titlesize`` (12pt) when unset.
        color (str or tuple, optional): Title colour, as a matplotlib colour name
            or an RGB(A) tuple. Defaults to the automatic light-on-dark ink.
        s (str, optional): Deprecated spelling of ``label``, kept working for code
            written against GLPlot <= 0.1.3.

    Returns:
        None

    Examples:
        >>> gplt.plot([1, 2, 3], [1, 4, 2])
        >>> gplt.title('My Plot Data', fontsize=14, color='white')
        >>> gplt.show()
    """
    plot = _get_or_create_plot()
    label = _mpl_text_arg("title", label, s, "label")
    fontsize, color = _merge_fontdict("title", fontdict, fontsize, color)
    _warn_unsupported(
        "title",
        {"loc": loc, "pad": pad, "y": y},
        {
            "loc": "has no effect: the title is always centred over the frame",
            "pad": "has no effect: the title is centred in the top gutter, so widen "
            "options.axis_margin_t to move it",
            "y": "has no effect: the title is centred in the top gutter, so widen "
            "options.axis_margin_t to move it",
        },
    )
    _set_title(plot, label, fontsize=fontsize, color=color)


def suptitle(
    t: Optional[str] = None,
    *,
    x: Optional[float] = None,
    y: Optional[float] = None,
    fontsize: Optional[float] = None,
    color: Optional[ColorLike] = None,
) -> None:
    """Set the figure title.

    In matplotlib a figure title sits above the per-axes titles. GLPlot draws
    into a single viewport, so a figure has exactly one axes and there is only
    one place a title can go: this and :func:`title` are the same title, and the
    later call wins. It exists so that a matplotlib script using ``suptitle``
    runs unchanged.

    Args:
        t (str): Title text.
        x (float, optional): Accepted for matplotlib parity. The title is always
            centred over the frame. Ignored.
        y (float, optional): Accepted for matplotlib parity. Ignored.
        fontsize (float, optional): Title size in points. Defaults to
            matplotlib's ``axes.titlesize`` (12pt) when unset.
        color (str or tuple, optional): Title colour.

    Returns:
        None

    Examples:
        >>> gplt.suptitle('Run 42', fontsize=16)
    """
    plot = _get_or_create_plot()
    text = _mpl_text_arg("suptitle", t, None, "t")
    _warn_unsupported(
        "suptitle",
        {"x": x, "y": y},
        {
            "x": "has no effect: the title is always centred over the frame",
            "y": "has no effect: the title is centred in the top gutter, so widen "
            "options.axis_margin_t to move it",
        },
    )
    _set_title(plot, text, fontsize=fontsize, color=color)


def xlabel(
    xlabel: Optional[str] = None,
    fontdict: Optional[dict] = None,
    labelpad: Optional[float] = None,
    *,
    loc: Optional[str] = None,
    fontsize: Optional[float] = None,
    color: Optional[ColorLike] = None,
    s: Optional[str] = None,
) -> None:
    """Set the x-axis label.

    Sets the label text displayed below the x-axis.

    Args:
        xlabel (str): Label text for the x-axis.
        fontdict (dict, optional): matplotlib parity. Merged into the styling
            below, with the explicit ``fontsize``/``color`` kwargs winning.
        labelpad (float, optional): Accepted for matplotlib parity. GLPlot pins
            the label to the viewport edge so the gutter width sets the spacing;
            widen ``options.axis_margin_b`` instead. Ignored.
        loc (str, optional): Accepted for matplotlib parity. The label is always
            centred. Ignored.
        fontsize (float, optional): Label size in points. Defaults to
            matplotlib's ``axes.labelsize`` (10pt) when unset.
        color (str or tuple, optional): Label colour. Defaults to the automatic
            light-on-dark ink when unset.
        s (str, optional): Deprecated spelling of ``xlabel``, kept working for
            code written against GLPlot <= 0.1.3.

    Returns:
        None

    Examples:
        >>> gplt.plot(time, values)
        >>> gplt.xlabel('Time (seconds)', fontsize=12, color='white')
        >>> gplt.show()
    """
    plot = _get_or_create_plot()
    text = _mpl_text_arg("xlabel", xlabel, s, "xlabel")
    fontsize, color = _merge_fontdict("xlabel", fontdict, fontsize, color)
    _warn_unsupported(
        "xlabel",
        {"labelpad": labelpad, "loc": loc},
        {
            "labelpad": "has no effect: the label is pinned to the viewport edge, so widen "
            "options.axis_margin_b to give it room",
            "loc": "has no effect: the x-label is always centred",
        },
    )
    plot.xlabel = text
    # A 3D figure draws its axis titles from ``axes3d``, beside the box edge the tick
    # numbers are on, so the same call has to reach both. Writing only ``plot.xlabel``
    # left a 3D plot's axes unlabelled no matter how many times you called this.
    plot.axes3d.xlabel = text
    plot.options.axis_xlabel_fontsize = fontsize
    plot.options.axis_xlabel_color = color
    if plot.is_3d_scene():
        plot.set_3d_view()
    _set_dirty(plot)


def ylabel(
    ylabel: Optional[str] = None,
    fontdict: Optional[dict] = None,
    labelpad: Optional[float] = None,
    *,
    loc: Optional[str] = None,
    fontsize: Optional[float] = None,
    color: Optional[ColorLike] = None,
    s: Optional[str] = None,
) -> None:
    """Set the y-axis label.

    Sets the label text displayed to the left of the y-axis.

    Args:
        ylabel (str): Label text for the y-axis.
        fontdict (dict, optional): matplotlib parity. Merged into the styling
            below, with the explicit ``fontsize``/``color`` kwargs winning.
        labelpad (float, optional): Accepted for matplotlib parity. GLPlot places
            the label just outside the widest tick label, so the gutter width
            sets the spacing; widen ``options.axis_margin_l`` instead. Ignored.
        loc (str, optional): Accepted for matplotlib parity. The label is always
            centred. Ignored.
        fontsize (float, optional): Label size in points. Defaults to
            matplotlib's ``axes.labelsize`` (10pt) when unset.
        color (str or tuple, optional): Label colour. Defaults to the automatic
            light-on-dark ink when unset.
        s (str, optional): Deprecated spelling of ``ylabel``, kept working for
            code written against GLPlot <= 0.1.3.

    Returns:
        None

    Examples:
        >>> gplt.plot(time, values)
        >>> gplt.ylabel('Amplitude', fontsize=12, color='white')
        >>> gplt.show()
    """
    plot = _get_or_create_plot()
    text = _mpl_text_arg("ylabel", ylabel, s, "ylabel")
    fontsize, color = _merge_fontdict("ylabel", fontdict, fontsize, color)
    _warn_unsupported(
        "ylabel",
        {"labelpad": labelpad, "loc": loc},
        {
            "labelpad": "has no effect: the label tracks the widest tick label, so widen "
            "options.axis_margin_l to give it room",
            "loc": "has no effect: the y-label is always centred",
        },
    )
    plot.ylabel = text
    # A 3D figure draws its axis titles from ``axes3d``, beside the box edge the tick
    # numbers are on, so the same call has to reach both. Writing only ``plot.ylabel``
    # left a 3D plot's axes unlabelled no matter how many times you called this.
    plot.axes3d.ylabel = text
    plot.options.axis_ylabel_fontsize = fontsize
    plot.options.axis_ylabel_color = color
    if plot.is_3d_scene():
        plot.set_3d_view()
    _set_dirty(plot)


def _refresh_axis_ticks(plot: GPULinePlot) -> None:
    """Bring ``plot.axis_manager`` up to date with the camera, outside a render frame.

    The ticks are a per-frame product: `AxisManager.update` runs from the draw loop, off a
    `RenderContext` the renderer builds. So before a window exists -- which is every script
    up to `show()`, and every test -- the manager holds empty arrays, and `xticks()` would
    answer a question about the plot with "there are no ticks".

    Synthesising the context from the live camera is what the offscreen export path already
    does, and it is the same projection the draw loop would build, so the answer here is
    the one that will be on screen rather than a second opinion about it.
    """
    from .core.context import RenderContext
    from .options import RenderMode

    width, height = int(plot.width), int(plot.height)
    plot.axis_manager.update(
        RenderContext(
            mvp=plot.camera_controller.mvp(width, height),
            window_world=plot.camera_controller.world_window(width, height),
            width_px=width,
            height_px=height,
            fb_width=width,
            fb_height=height,
            mode=RenderMode.EXACT,
        )
    )


def _ticks(
    axis: str,
    func: str,
    ticks: Optional[ArrayLike],
    labels: Optional[Sequence[str]],
    minor: bool,
    fontsize: Optional[float],
    color: Optional[ColorLike],
) -> Tuple[np.ndarray, List[str]]:
    """The shared body of :func:`xticks` and :func:`yticks`.

    Both are matplotlib's get-or-set pair on one name: no arguments queries, arguments
    set. The query has to answer from the live axis rather than from the stored override,
    because with no override there is nothing stored -- the ticks are generated per frame
    from the view, and what the caller wants back is the ticks that are *on the plot*.
    """
    plot = _get_or_create_plot()
    _warn_unsupported(
        func,
        {"minor": minor or None, "fontsize": fontsize, "color": color},
        {
            "minor": "has no effect: minor ticks are generated by subdividing the major "
            "step (options.axis_minor_ticks) and cannot be placed individually",
            "fontsize": "has no effect: tick labels are drawn at the atlas font size",
            "color": "has no effect: tick labels take their colour from the background "
            "luminance automatically",
        },
        stacklevel=4,
    )

    if ticks is not None:
        values = _as_float_array(np.atleast_1d(ticks), ndim=1, name="ticks")
        if labels is not None:
            # matplotlib raises on a mismatch rather than truncating, and so must we: a
            # short list would otherwise silently slide every label onto the wrong tick.
            if len(labels) != len(values):
                raise ValueError(
                    f"{func}(): got {len(values)} ticks but {len(labels)} labels; "
                    "they must be the same length"
                )
            setattr(plot.options, f"axis_tick_labels_{axis}", tuple(str(t) for t in labels))
        else:
            setattr(plot.options, f"axis_tick_labels_{axis}", None)
        setattr(plot.options, f"axis_tick_values_{axis}", tuple(float(v) for v in values))
        _set_dirty(plot)
    elif labels is not None:
        raise TypeError(f"{func}(): labels= cannot be set without ticks=")

    _refresh_axis_ticks(plot)
    live = getattr(plot.axis_manager, f"ticks_{axis}")
    return np.asarray(live.major), list(live.labels)


def xticks(
    ticks: Optional[ArrayLike] = None,
    labels: Optional[Sequence[str]] = None,
    *,
    minor: bool = False,
    fontsize: Optional[float] = None,
    color: Optional[ColorLike] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Get or set the x-axis tick positions and labels.

    With no arguments this queries the ticks currently on the plot. With ``ticks``
    it pins them, overriding the automatic 1-2-5 spacing. ``xticks([])`` clears
    the x-axis, as in matplotlib.

    Args:
        ticks (array-like, optional): Tick positions in data coordinates. An
            empty sequence removes every x tick. If None, nothing is set and the
            call is a pure query.
        labels (sequence of str, optional): Text for each tick, positionally
            paired with ``ticks``. Must be the same length. If None, the ticks
            are numbered by the active formatter.
        minor (bool, optional): Accepted for matplotlib parity. GLPlot derives
            minor ticks by subdividing the major step
            (``options.axis_minor_ticks``) and cannot place them individually,
            so this is ignored.
        fontsize (float, optional): Accepted for matplotlib parity. Ignored.
        color (str or tuple, optional): Accepted for matplotlib parity. Tick
            labels take their colour from the background luminance. Ignored.

    Returns:
        tuple: ``(locs, labels)`` currently drawn on the x-axis. Only the ticks
        inside the view are returned, since those are the ones on the plot.

    Examples:
        >>> gplt.xticks([0, 1, 2], ['low', 'mid', 'high'])
        >>> locs, labels = gplt.xticks()
        >>> gplt.xticks([])  # hide the x ticks
    """
    return _ticks("x", "xticks", ticks, labels, minor, fontsize, color)


def yticks(
    ticks: Optional[ArrayLike] = None,
    labels: Optional[Sequence[str]] = None,
    *,
    minor: bool = False,
    fontsize: Optional[float] = None,
    color: Optional[ColorLike] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Get or set the y-axis tick positions and labels.

    The y-axis counterpart of :func:`xticks`; see it for the full argument
    description.

    Args:
        ticks (array-like, optional): Tick positions in data coordinates. An
            empty sequence removes every y tick.
        labels (sequence of str, optional): Text for each tick, same length as
            ``ticks``.
        minor (bool, optional): Accepted for matplotlib parity. Ignored.
        fontsize (float, optional): Accepted for matplotlib parity. Ignored.
        color (str or tuple, optional): Accepted for matplotlib parity. Ignored.

    Returns:
        tuple: ``(locs, labels)`` currently drawn on the y-axis.

    Examples:
        >>> gplt.yticks([0, 0.5, 1.0])
        >>> locs, labels = gplt.yticks()
    """
    return _ticks("y", "yticks", ticks, labels, minor, fontsize, color)


def _tick_labels(
    axis: str,
    func: str,
    labels: Optional[Sequence[str]],
    fontsize: Optional[float],
    color: Optional[ColorLike],
    rotation: Optional[Any],
    ha: Optional[str],
    va: Optional[str],
) -> List[Any]:
    """The shared body of :func:`xticklabels` and :func:`yticklabels`.

    matplotlib's ``set_xticklabels(labels)`` labels whatever ticks are *currently* on the
    axis -- it does not move them -- so this reads the live tick positions first (the same
    ones :func:`xticks`'s query form returns) and pins both the positions and the labels
    together through :func:`_ticks`. Pinning the positions too, rather than leaving them
    generated, matches matplotlib's own behaviour: a ``set_xticklabels`` call with no prior
    ``set_xticks`` is exactly the case matplotlib itself warns about ("FixedFormatter
    should only be used together with FixedLocator") because it freezes the locations as a
    side effect.
    """
    _warn_unsupported(
        func,
        {"rotation": rotation, "ha": ha, "va": va},
        {
            "rotation": "has no effect: tick labels are always drawn horizontally",
            "ha": "has no effect: tick label alignment is fixed",
            "va": "has no effect: tick label alignment is fixed",
        },
        stacklevel=4,
    )
    plot = _get_or_create_plot()
    _refresh_axis_ticks(plot)
    live = getattr(plot.axis_manager, f"ticks_{axis}")
    if labels is None:
        return list(live.labels)
    current = np.asarray(live.major, dtype=float)
    if len(labels) != len(current):
        raise ValueError(
            f"{func}(): got {len(labels)} labels but {len(current)} ticks are currently "
            f"on the {axis}-axis; call {'xticks' if axis == 'x' else 'yticks'}() first to "
            "set both together"
        )
    _ticks(axis, func, current, [str(label) for label in labels], False, fontsize, color)
    return list(labels)


def xticklabels(
    labels: Optional[Sequence[str]] = None,
    *,
    fontsize: Optional[float] = None,
    color: Optional[ColorLike] = None,
    rotation: Optional[Any] = None,
    ha: Optional[str] = None,
    va: Optional[str] = None,
) -> List[Any]:
    """Get or set text labels for the x-axis ticks *currently on the plot*.

    Unlike :func:`xticks`, this does not choose where the ticks go -- it labels whatever
    is there already (auto-generated, or pinned by a prior ``xticks()``/``set_xticks()``
    call), exactly as matplotlib's ``ax.set_xticklabels()`` does.

    Args:
        labels (sequence of str, optional): One label per tick currently on the axis. If
            None, returns the current labels instead of setting them.
        fontsize, color: Accepted for matplotlib parity. Ignored.
        rotation, ha, va: Accepted for matplotlib parity. Ignored -- tick labels are
            always drawn horizontally, left-aligned.

    Returns:
        list: The tick labels now in force (or currently on the plot, for a query).

    Raises:
        ValueError: If ``labels`` is given and its length does not match the number of
            ticks currently on the axis.

    Examples:
        >>> gplt.bar(range(3), [1, 4, 2])
        >>> gplt.xticks(range(3))
        >>> gplt.xticklabels(['a', 'b', 'c'])
    """
    return _tick_labels("x", "xticklabels", labels, fontsize, color, rotation, ha, va)


def yticklabels(
    labels: Optional[Sequence[str]] = None,
    *,
    fontsize: Optional[float] = None,
    color: Optional[ColorLike] = None,
    rotation: Optional[Any] = None,
    ha: Optional[str] = None,
    va: Optional[str] = None,
) -> List[Any]:
    """Get or set text labels for the y-axis ticks *currently on the plot*.

    The y-axis counterpart of :func:`xticklabels`; see it for the full description.
    """
    return _tick_labels("y", "yticklabels", labels, fontsize, color, rotation, ha, va)


def zlabel(s: Optional[str] = None, **kwargs: Any) -> None:
    """Set the z-axis label for 3D plots.

    Args:
        s (str): Label text for the z-axis. Also accepted as ``zlabel=`` (matplotlib's
            keyword name) or ``label=``.

    Returns:
        None

    Examples:
        >>> gplt.scatter3d(x, y, z)
        >>> gplt.zlabel('Height (meters)')
        >>> gplt.show()
    """
    plot = _get_or_create_plot()
    text = _mpl_text_arg("zlabel", s, kwargs.pop("zlabel", kwargs.pop("label", None)), "s")
    # Two homes, both load-bearing: ``axes3d.zlabel`` is what the GL 3D axis renderer
    # draws beside the z edge, and ``plot.zlabel`` is what the matplotlib export bridge
    # reads (``utils/preview.py``). Writing only one of them was the bug that made a label
    # set from code appear in the PNG and not on screen.
    plot.axes3d.zlabel = text
    plot.zlabel = text
    if plot.is_3d_scene():
        plot.set_3d_view()
    _set_dirty(plot)


def get_zlabel() -> str:
    """The current z axis title."""
    return _get_or_create_plot().axes3d.zlabel


def ssao(enabled: bool = True, strength: float = 0.45, radius: float = 1.0) -> None:
    """Enable/disable lightweight 3D ambient occlusion shading."""
    plot = _get_or_create_plot()
    plot.options.visual.ssao.enabled = bool(enabled)
    plot.options.visual.ssao.strength = float(strength)
    plot.options.visual.ssao.radius = float(radius)
    _set_dirty(plot)


def grid(
    visible: bool = True,
    which: str = "major",
    axis: str = "both",
    **kwargs,
) -> None:
    """Show or hide the background grid.

    Toggles the visibility of the grid lines in the background of the plot.

    Args:
        visible (bool, optional): If True, show grid. If False, hide grid.
            Defaults to True.
        which (str, optional): Accepted for matplotlib parity. GLPlot rules the grid at
            the major ticks from a single flag, so ``'minor'`` and ``'both'`` have nothing
            to select. Ignored.
        axis (str, optional): Accepted for matplotlib parity. Both axes are ruled from a
            single flag, so ``'x'`` and ``'y'`` have nothing to select. Ignored.
        **kwargs: Line styling accepted for parity; the grid takes its colour and alpha
            from ``options.axis_grid_color`` / ``axis_grid_alpha``.

    Returns:
        None

    Examples:
        Show grid:

        >>> gplt.plot([1, 2, 3], [1, 4, 2])
        >>> gplt.grid(True)
        >>> gplt.show()

        Hide grid:

        >>> gplt.grid(False)
    """
    plot = _get_or_create_plot()
    _warn_unsupported(
        "grid",
        {
            "which": which if which != "major" else None,
            "axis": axis if axis != "both" else None,
        },
        {
            "which": "has no effect: GLPlot rules the grid at the major ticks only, from a "
            "single flag",
            "axis": "has no effect: both axes are ruled from a single flag, so the grid "
            "cannot be turned on for one axis alone",
        },
    )
    _warn_unsupported("grid", kwargs)
    plot.grid_visible = bool(visible)
    if hasattr(plot.options, "show_grid"):
        plot.options.show_grid = bool(visible)
    plot.options.axis_show_grid = bool(visible)
    _set_dirty(plot)


def minorticks_on() -> None:
    """Turn minor ticks on for both axes.

    Minor ticks subdivide the major spacing; GLPlot draws
    ``options.axis_minor_subdivisions`` of them per major step.

    Returns:
        None
    """
    plot = _get_or_create_plot()
    plot.options.axis_minor_ticks = True
    _set_dirty(plot)


def minorticks_off() -> None:
    """Turn minor ticks off for both axes.

    Returns:
        None
    """
    plot = _get_or_create_plot()
    plot.options.axis_minor_ticks = False
    _set_dirty(plot)


def margins(
    *args: float,
    x: Optional[float] = None,
    y: Optional[float] = None,
    tight: Optional[bool] = True,
) -> Tuple[float, float]:
    """Set padding around the data as a fraction of the data range.

    ``margins(m)`` pads both axes by ``m``; ``margins(mx, my)`` pads them
    separately; ``margins(x=..., y=...)`` names them. A margin of 0.05 leaves 5%
    of the data range as blank border, which is matplotlib's default.

    Args:
        *args: One margin for both axes, or two for x and y.
        x, y (float, optional): Per-axis margins by keyword.
        tight (bool, optional): In matplotlib this decides whether the autoscaled limits
            are rounded outward to the next tick. GLPlot's autoscale always stops at
            ``data + margin``, which is matplotlib's ``tight=True``, so that is the
            default here and only an explicit ``tight=False`` warns.

    Returns:
        tuple: The ``(x, y)`` margins now in force.

    Examples:
        >>> gplt.margins(0.1)
        >>> gplt.margins(x=0, y=0.2)
    """
    plot = _get_or_create_plot()
    if tight is False:
        _warn_unsupported(
            "margins",
            {"tight": True},
            {
                "tight": "cannot be turned off: GLPlot's autoscale stops at data + margin "
                "and never rounds the limits out to the next tick"
            },
        )
    if args:
        if len(args) == 1:
            mx = my = float(args[0])
        elif len(args) == 2:
            mx, my = float(args[0]), float(args[1])
        else:
            raise TypeError("margins() takes 0, 1 or 2 positional args")
    else:
        mx = float(x) if x is not None else float(getattr(plot.options, "autoscale_margin", 0.05))
        my = float(y) if y is not None else mx
    if x is not None:
        mx = float(x)
    if y is not None:
        my = float(y)
    # The engine autoscales with one symmetric margin; store both and use the x one for
    # the shared knob, so `margins(0.1)` visibly loosens the fit even though a per-axis
    # asymmetry is not something the single-margin autoscale can honour.
    plot.options.autoscale_margin = mx
    plot._margins_xy = (mx, my)
    if mx != my:
        _warn_unsupported(
            "margins",
            {"asymmetric": True},
            {
                "asymmetric": "differing x and y margins are stored but the autoscale uses one "
                "symmetric margin, so the x value is applied to both"
            },
        )
    _set_dirty(plot)
    return (mx, my)


def tick_params(axis: str = "both", **kwargs: Any) -> None:
    """Configure tick appearance.

    Args:
        axis (str, optional): 'x', 'y' or 'both'. Which axis to configure.
        **kwargs: matplotlib tick properties. ``labelsize``, ``labelcolor``,
            ``length`` and the on/off toggles map onto GLPlot's tick options;
            the rest are accepted and ignored.

    Returns:
        None

    Examples:
        >>> gplt.tick_params(axis='both', length=6)
    """
    plot = _get_or_create_plot()
    supported = {"length", "labelsize", "which", "direction", "labelbottom", "labelleft"}
    if "length" in kwargs:
        plot.options.axis_tick_len_px = float(kwargs["length"])
        plot.options.axis_show_ticks = True
    unsupported = {k: v for k, v in kwargs.items() if k not in supported}
    _warn_unsupported("tick_params", unsupported)
    _set_dirty(plot)


def locator_params(axis: str = "both", nbins: Optional[int] = None, **kwargs: Any) -> None:
    """Control how many ticks the automatic locator places.

    Args:
        axis (str, optional): 'x', 'y' or 'both'.
        nbins (int, optional): The approximate number of tick intervals.
        **kwargs: Accepted for matplotlib parity; ignored.

    Returns:
        None

    Examples:
        >>> gplt.locator_params(nbins=5)
    """
    plot = _get_or_create_plot()
    if nbins is not None:
        n = int(nbins)
        if axis in ("x", "both"):
            plot.options.axis_tick_count_x = n
        if axis in ("y", "both"):
            plot.options.axis_tick_count_y = n
    _warn_unsupported("locator_params", {k: v for k, v in kwargs.items()})
    _set_dirty(plot)


def ticklabel_format(
    *,
    style: str = "",
    scilimits: Optional[Tuple[int, int]] = None,
    axis: str = "both",
    useOffset: Optional[Union[bool, float]] = None,
    useLocale: Optional[bool] = None,
    useMathText: Optional[bool] = None,
    **kwargs: Any,
) -> None:
    """Set the number format of the tick labels.

    Args:
        style (str, optional): 'plain', 'sci', or ''. 'sci' forces scientific
            notation.
        scilimits (tuple, optional): Accepted for matplotlib parity; the
            automatic formatter chooses when to switch to scientific notation.
        axis (str, optional): 'x', 'y' or 'both'.
        useOffset (bool or float, optional): Accepted for matplotlib parity. GLPlot never
            factors a common offset out of the tick labels, so there is nothing to switch
            off. Ignored.
        useLocale (bool, optional): Accepted for matplotlib parity. The labels are
            formatted with a C-style format string, which has no locale-aware separator.
            Ignored.
        useMathText (bool, optional): Accepted for matplotlib parity. GLPlot's text
            renderer has no mathtext engine, so an exponent cannot be typeset. Ignored.
        **kwargs: Accepted for parity; ignored.

    Returns:
        None
    """
    plot = _get_or_create_plot()
    if style == "sci":
        plot.options.axis_tick_format = "%.1e"
    elif style == "plain":
        plot.options.axis_tick_format = "%g"
    _warn_unsupported(
        "ticklabel_format",
        {
            "scilimits": scilimits,
            "useOffset": useOffset,
            "useLocale": useLocale,
            "useMathText": useMathText,
        },
        {
            "scilimits": "is ignored; the automatic formatter decides when to use scientific "
            "notation",
            "useOffset": "has no effect: GLPlot never factors a common offset out of the "
            "tick labels, so every label already reads as its full value",
            "useLocale": "has no effect: the labels go through a C-style format string, "
            "which has no locale-aware decimal separator",
            "useMathText": "has no effect: GLPlot's text renderer has no mathtext engine, "
            "so an exponent cannot be typeset as a superscript",
        },
    )
    _set_dirty(plot)


def box(on: Optional[bool] = None) -> None:
    """Show or hide the axes frame (the box around the plot).

    Args:
        on (bool, optional): True shows the frame, False hides it. None toggles.

    Returns:
        None
    """
    plot = _get_or_create_plot()
    current = bool(getattr(plot.options, "axis_show_frame", True))
    plot.options.axis_show_frame = (not current) if on is None else bool(on)
    _set_dirty(plot)


def rgrids(*args: Any, **kwargs: Any):
    """Accepted for matplotlib parity; GLPlot has no polar axes.

    Returns:
        tuple: Empty ``([], [])``, matching matplotlib's ``(lines, labels)``.
    """
    _get_or_create_plot()
    _warn_unsupported(
        "rgrids",
        {"rgrids": True},
        {"rgrids": "does nothing: GLPlot has no polar axes to place radial grid lines on"},
    )
    return [], []


def thetagrids(*args: Any, **kwargs: Any):
    """Accepted for matplotlib parity; GLPlot has no polar axes.

    Returns:
        tuple: Empty ``([], [])``.
    """
    _get_or_create_plot()
    _warn_unsupported(
        "thetagrids",
        {"thetagrids": True},
        {"thetagrids": "does nothing: GLPlot has no polar axes to place angular grid lines on"},
    )
    return [], []


def legend(*args, max_items: Optional[int] = None, **kwargs):
    """Display a legend showing labeled layers.

    Automatically creates legend from labeled plot objects. Deduplicates
    repeated labels and can limit the number of displayed items to avoid
    clutter on dense plots.

    Args:
        *args: Reserved for future compatibility (ignored).
        max_items (int, optional): Maximum number of legend items to show.
            If there are more unique labels than this, shows top N and adds
            a "+N more" entry. Defaults to None (show all).
        **kwargs: Reserved for future keyword arguments.

    Returns:
        list: List of legend labels displayed.

    Examples:
        Simple legend:

        >>> gplt.plot([1, 2, 3], [1, 4, 2], label='Data 1')
        >>> gplt.plot([1, 2, 3], [2, 2, 3], label='Data 2')
        >>> labels = gplt.legend()
        >>> gplt.show()

        With max items limit:

        >>> # Plot many datasets...
        >>> labels = gplt.legend(max_items=5)  # Show top 5 + "+N more"
    """
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
    # Turn the live overlay on. Until this line existed, `legend()` assigned
    # `legend_labels` and stopped -- nothing in renderers/ or managers/ read it, so this
    # call drew exactly nothing in the window and only ever showed up in an exported PNG
    # (the same defect xlabel/ylabel/title had). `renderers.legend` reads these two
    # attributes; the return value and the `max_items` semantics above are untouched
    # public API.
    plot.legend_show = True
    plot.legend_max_items = None if max_items is None else int(max_items)
    _set_dirty(plot)
    return labels


def xlim(
    left: Optional[float] = None, right: Optional[float] = None
) -> Optional[Tuple[float, float]]:
    """Get or set the x-axis limits.

    Query the current x-axis range or set new limits. If called without
    arguments, returns current limits. If called with arguments, sets new
    limits and returns the tuple of limits set.

    Args:
        left (float, optional): Minimum x-value (left edge). Defaults to None.
        right (float, optional): Maximum x-value (right edge). Defaults to None.

    Returns:
        tuple: Current or newly set (left, right) x-axis limits. Returns None
            if called without arguments (getter mode in future).

    Raises:
        ValueError: If right <= left.

    Examples:
        Get current limits:

        >>> xleft, xright = gplt.xlim()

        Set limits:

        >>> gplt.xlim(0, 100)

        Using tuple:

        >>> gplt.xlim((0, 100))
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
    """Get or set the y-axis limits.

    Query the current y-axis range or set new limits. If called without
    arguments, returns current limits. If called with arguments, sets new
    limits and returns the tuple of limits set.

    Args:
        bottom (float, optional): Minimum y-value (bottom edge). Defaults to None.
        top (float, optional): Maximum y-value (top edge). Defaults to None.

    Returns:
        tuple: Current or newly set (bottom, top) y-axis limits. Returns None
            if called without arguments (getter mode in future).

    Raises:
        ValueError: If top <= bottom.

    Examples:
        Get current limits:

        >>> ybottom, ytop = gplt.ylim()

        Set limits:

        >>> gplt.ylim(-10, 10)

        Using tuple:

        >>> gplt.ylim((-10, 10))
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
    arg: Union[str, Tuple[float, float, float, float], None] = None,
    *,
    emit: Optional[bool] = None,
    mode: Union[str, Tuple[float, float, float, float], None] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """Control or query the axis limits and scaling.

    Get or set the axis extent and scaling mode. Supports preset modes
    ('auto', 'tight', 'reset', 'on', 'off', 'equal', 'scaled', 'image', 'square')
    and explicit limit specification via tuple.
    Provides matplotlib-compatible interface for axis manipulation.

    Args:
        arg (str or tuple, optional): Axis mode or limits. Can be:
            - 'auto': Auto-scale to fit all data (default)
            - 'tight': Auto-scale tightly to data without padding
            - 'reset': Reset to default view (-1, 1) on both axes
            - 'on'/'off': Show or hide the grid, frame and tick labels
            - 'equal': Same world-units-per-pixel on both axes
            - 'scaled'/'image': As 'equal'; 'image' fits the data first
            - 'square': Equal world *span* on both axes
            - tuple (xmin, xmax, ymin, ymax): Explicit axis limits
            Defaults to 'auto'.
        emit (bool, optional): Accepted for matplotlib parity. GLPlot has no
            xlim/ylim observer callbacks to notify, so this is ignored.
        mode (str or tuple, optional): GLPlot's own spelling of ``arg``, kept working for
            code written against GLPlot <= 0.1.3. matplotlib names this parameter ``arg``,
            and ``axis(arg=...)`` used to be a TypeError here.
        xmin, xmax, ymin, ymax (float, optional): Set individual limits by name, as
            matplotlib's ``axis(xmin=0, xmax=10)`` does. Combinable with a mode string.

    Note:
        matplotlib reaches an equal aspect either by moving the data limits
        ('equal') or by reshaping the axes box ('scaled'). GLPlot renders into a
        single viewport whose box is the window, so both resolve to the data-limit
        route and 'scaled' behaves as 'equal'.

    Returns:
        tuple or None: When called with mode (setter), returns limits or None.
            When called with no arguments, returns current limits (to be
            implemented in future).

    Raises:
        ValueError: If mode is unsupported or limits are invalid
            (xmax <= xmin or ymax <= ymin).

    Examples:
        Auto-scale to data:

        >>> gplt.plot([1, 2, 3], [1, 4, 2])
        >>> gplt.axis('auto')

        Tight fit to data (no padding):

        >>> gplt.axis('tight')

        Reset to default view:

        >>> gplt.axis('reset')

        Set explicit limits:

        >>> gplt.axis((0, 10, -5, 5))  # xmin, xmax, ymin, ymax
    """
    plot = _get_or_create_plot()
    _warn_unsupported(
        "axis",
        {"emit": emit},
        {"emit": "has no effect: GLPlot has no xlim/ylim observer callbacks to notify"},
    )

    # `arg` is matplotlib's name and `mode` GLPlot's; whichever was given wins, and with
    # neither the call is either the named-limit form or the plain 'auto' default.
    mode = arg if arg is not None else mode
    if any(v is not None for v in (xmin, xmax, ymin, ymax)):
        if xmin is not None or xmax is not None:
            lo, hi = plot.get_xlim()
            plot.set_view(xlim=(lo if xmin is None else xmin, hi if xmax is None else xmax))
        if ymin is not None or ymax is not None:
            lo, hi = plot.get_ylim()
            plot.set_view(ylim=(lo if ymin is None else ymin, hi if ymax is None else ymax))
        _set_dirty(plot)
        if mode is None:
            return (*plot.get_xlim(), *plot.get_ylim())
    if mode is None:
        mode = "auto"

    if isinstance(mode, str):
        m = mode.lower()
        if m in ("auto", "tight"):
            if _call_if_exists(plot, ("autoscale", "auto_view", "fit_view")) is _MISSING:
                raise AttributeError("Backend does not expose autoscale()/fit_view()")
            _set_dirty(plot)
            return None
        if m in ("reset", "home"):
            if _call_if_exists(plot, ("reset_view", "home_view")) is _MISSING:
                plot.set_view(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))  # Absolute reset fallback
            _set_dirty(plot)
            return None
        if m in ("on", "off"):
            _set_axis_visible(plot, m == "on")
            return None
        if m in ("equal", "scaled", "image", "square"):
            # 'image' is matplotlib's 'scaled' applied to tight limits, so it fits first.
            if m == "image":
                if _call_if_exists(plot, ("autoscale", "auto_view", "fit_view")) is _MISSING:
                    raise AttributeError("Backend does not expose autoscale()/fit_view()")
            _apply_equal_aspect(plot, square=(m == "square"))
            _set_dirty(plot)
            return None
        raise ValueError(
            f"unsupported axis mode: {mode!r}. Expected one of "
            "'auto', 'tight', 'reset', 'on', 'off', 'equal', 'scaled', 'image', 'square', "
            "or a (xmin, xmax, ymin, ymax) tuple."
        )

    if len(mode) != 4:
        raise ValueError("axis tuple must be (xmin, xmax, ymin, ymax)")

    xmin, xmax, ymin, ymax = map(float, mode)
    plot.set_view(xlim=(xmin, xmax), ylim=(ymin, ymax))
    _set_dirty(plot)
    return (xmin, xmax, ymin, ymax)


def set_aspect(
    aspect: Union[str, float],
    adjustable: Optional[str] = None,
    anchor: Optional[Union[str, Tuple[float, float]]] = None,
    share: bool = False,
) -> None:
    """Set the ratio of y-unit-per-pixel to x-unit-per-pixel (matplotlib's ``Axes.set_aspect``).

    ``'equal'`` (or a numeric ratio of 1.0) is :func:`axis`'s own ``'equal'`` mode --
    GLPlot's single-viewport equivalent of matplotlib's ``adjustable='datalim'`` route,
    which is also the only one it has (see :func:`axis`'s note). ``'auto'`` restores the
    default autoscale. Ratios other than 1.0 have no equivalent -- GLPlot has one uniform
    zoom per axis, not a per-call skew factor -- and are rounded to 1.0 with a warning
    rather than silently drawn at the wrong ratio.

    Args:
        aspect (str or float): ``'equal'``, ``'auto'``, or a numeric y/x unit ratio.
        adjustable (str, optional): Accepted for matplotlib parity. Ignored -- GLPlot
            always adjusts the data limits, never the axes box.
        anchor (str or tuple, optional): Accepted for matplotlib parity. Ignored --
            GLPlot's single viewport has nothing to anchor within.
        share (bool, optional): Accepted for matplotlib parity. Ignored -- GLPlot has no
            linked-aspect axes group.

    Returns:
        None

    Examples:
        >>> gplt.scatter(x, y)
        >>> gplt.set_aspect('equal')
    """
    plot = _get_or_create_plot()
    _warn_unsupported(
        "set_aspect",
        {"adjustable": adjustable, "anchor": anchor, "share": share or None},
        {
            "adjustable": "has no effect: GLPlot always adjusts the data limits "
            "('datalim'), never the axes box",
            "anchor": "has no effect: GLPlot's single viewport has nothing to anchor " "within",
            "share": "has no effect: GLPlot has no linked-aspect axes group",
        },
    )
    if isinstance(aspect, str) and aspect == "auto":
        if _call_if_exists(plot, ("autoscale", "auto_view", "fit_view")) is _MISSING:
            raise AttributeError("Backend does not expose autoscale()/fit_view()")
        _set_dirty(plot)
        return
    if isinstance(aspect, str) and aspect == "equal":
        _apply_equal_aspect(plot, square=False)
        _set_dirty(plot)
        return
    try:
        ratio = float(aspect)
    except (TypeError, ValueError):
        raise ValueError(
            f"set_aspect(): unsupported aspect {aspect!r}; expected 'equal', 'auto', or a "
            "numeric ratio"
        ) from None
    if ratio != 1.0:
        _warn_unsupported_call(
            "set_aspect",
            f"rounds any numeric ratio to 1.0 (got {ratio!r}): GLPlot has one uniform "
            "zoom per axis, not a per-call skew factor",
        )
    _apply_equal_aspect(plot, square=False)
    _set_dirty(plot)


def autoscale(enable: bool = True, axis: str = "both", tight: Optional[bool] = None) -> None:
    """Auto-scale the view to fit the plotted data.

    Adjusts axis limits to show all plotted data. Can auto-scale both axes,
    or independently control x and y. Supports tight fitting (no padding)
    or regular fitting (with padding).

    Args:
        enable (bool, optional): If True, perform autoscaling. If False,
            disable. Defaults to True.
        axis (str, optional): Which axes to scale: 'x', 'y', or 'both'.
            Defaults to 'both'.
        tight (bool, optional): If True, fit tightly to data without padding.
            If False or None, add padding. Defaults to None (add padding).

    Returns:
        None

    Examples:
        Auto-scale to fit all data:

        >>> gplt.plot([1, 100], [1, 1000])
        >>> gplt.autoscale()
        >>> gplt.show()

        Tight fit without padding:

        >>> gplt.autoscale(tight=True)

        Only auto-scale y-axis:

        >>> gplt.autoscale(axis='y')
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


def savefig(
    filename: str,
    density: Optional[bool] = None,
    scale: float = 2.0,
    *,
    dpi: Optional[float] = None,
    bbox_inches: Optional[Any] = None,
    transparent: Optional[bool] = None,
    facecolor: Optional[ColorLike] = None,
    edgecolor: Optional[ColorLike] = None,
    pad_inches: Optional[float] = None,
    format: Optional[str] = None,
    **kwargs: Any,
):
    """Save the current figure to a PNG file.

    Exports the current plot to a PNG image at the specified filename.
    Supports high-DPI rendering via the scale parameter. Can optionally
    override density mode for the export. Automatically falls back to
    matplotlib-based preview rendering if no GL window is available.

    Args:
        filename (str): Output file path. Should end in '.png'. Directory
            must exist.
        density (bool, optional): Override density visualization for export.
            Defaults to None (use current setting).
        scale (float, optional): Render scale multiplier for high-DPI export.
            2.0 renders at 2x resolution before downsampling. Defaults to 2.0.
        dpi, bbox_inches, transparent, facecolor, edgecolor, pad_inches, format:
            matplotlib's own ``Figure.savefig()`` keywords. Honoured on the headless
            export path (no GL window has been created yet -- the common case for a
            script that never calls ``show()``), where GLPlot builds a real matplotlib
            figure and these are its own arguments. On the GL-window export path there
            is no matplotlib figure to hand them to, so they are accepted (a script
            does not crash) but warned about once, since they cannot take effect there.
        **kwargs: Any other matplotlib ``savefig`` keyword (``metadata``, ``pil_kwargs``,
            ``backend``, ...). Same handling as the keywords above.

    Returns:
        None

    Raises:
        IOError: If file cannot be written or directory doesn't exist.

    Examples:
        Save current plot:

        >>> gplt.plot([1, 2, 3], [1, 4, 2])
        >>> gplt.savefig('plot.png')

        High-resolution export:

        >>> gplt.scatter(x, y)
        >>> gplt.savefig('scatter_hires.png', scale=4.0)

        With density mode:

        >>> gplt.imshow(data)
        >>> gplt.savefig('heatmap.png', density=True)

        matplotlib-style keywords (headless export):

        >>> gplt.plot([1, 2, 3], [1, 4, 2])
        >>> gplt.savefig('plot.png', dpi=300, bbox_inches='tight', transparent=True)
    """
    plot = _get_or_create_plot()

    if density is not None:
        _set_density(plot, density)

    mpl_savefig_kwargs = {
        k: v
        for k, v in dict(
            dpi=dpi,
            bbox_inches=bbox_inches,
            transparent=transparent,
            facecolor=facecolor,
            edgecolor=edgecolor,
            pad_inches=pad_inches,
            format=format,
            **kwargs,
        ).items()
        if v is not None
    }

    # If no GL window exists (plt.show() hasn't been called), fall back to the
    # matplotlib-based preview renderer — same path the gallery runner uses.
    # Attempting to create a headless GLFW context here crashes on macOS when
    # Python is not running as a proper app bundle.
    if getattr(plot, "window", None) is None:
        from glplot.utils.preview import render_preview as _render_preview

        _render_preview(plot, filename, scale, **mpl_savefig_kwargs)
        return

    _warn_unsupported(
        "savefig",
        mpl_savefig_kwargs,
        {
            k: "has no effect once a GL window exists: this export path reads back the "
            "GPU framebuffer directly, and there is no matplotlib Figure to hand it to"
            for k in mpl_savefig_kwargs
        },
    )

    if hasattr(plot, "savefig"):
        plot.savefig(filename, scale=scale)
        return

    _call_if_exists(plot, ("save_current_view",), filename, scale=scale)


def show(
    density: Optional[bool] = None,
    block: Optional[bool] = None,
    *,
    test_mode: bool = False,
) -> None:
    """Display the current figure in an interactive window.

    Opens a native window running the interactive plot engine with OpenGL
    rendering. Blocks until the user closes the window. Can optionally
    override density mode and enable test mode for automated testing.

    Args:
        density (bool, optional): Override density visualization setting.
            If True, enables density mode for 2D layers. Defaults to None
            (use figure setting).
        block (bool, optional): Accepted for matplotlib parity. GLPlot's window owns the
            thread it runs on, so ``show()`` always blocks and ``block=False`` warns rather
            than returning to a script that would then exit and take the window with it.
        test_mode (bool, optional): Enable test mode for automated scripting.
            Disables certain interactive features. Defaults to False.

    Returns:
        None

    Examples:
        Display current figure:

        >>> gplt.plot([1, 2, 3], [1, 4, 2])
        >>> gplt.show()  # Blocks until window closed

        With density visualization:

        >>> gplt.imshow(data)
        >>> gplt.show(density=True)

        Test mode for automation:

        >>> gplt.plot(x, y)
        >>> gplt.show(test_mode=True)  # Headless mode
    """
    plot = _get_or_create_plot()

    if density is not None:
        _set_density(plot, density)
    _warn_unsupported(
        "show",
        {"block": True if block is False else None},
        {
            "block": "cannot be turned off: the GLPlot window runs its event loop on the "
            "calling thread, so returning early would let the script exit and close it"
        },
    )

    if hasattr(plot, "_is_test_mode"):
        plot._is_test_mode = bool(test_mode)

    plot.run()


# ------------------------------------------------------------------
# The Figure-level Axes constructors
# ------------------------------------------------------------------
#
# ``fig.add_subplot(...)`` is *the* canonical matplotlib idiom for obtaining axes, and
# ``fig.add_subplot(projection="3d")`` is the first line of nearly every mplot3d example
# in the gallery. Neither existed: ``GPULinePlot`` had no such method, so the canonical 3D
# script died on line two with ``AttributeError: 'GPULinePlot' object has no attribute
# 'add_subplot'``.
#
# They are installed onto the class from here rather than written into ``glplot/engine.py``
# because they are matplotlib-compatibility facade, not engine behaviour: they do no
# rendering, they own no state, and every line of them is about matching another library's
# argument shapes. The engine knows about panels; ``pyplot`` is the layer that knows what
# matplotlib calls them. Keeping the knowledge on this side also keeps ``AxesProxy`` --
# which only exists here -- out of the engine's imports.


def _figure_add_subplot(self: GPULinePlot, *args: Any, **kwargs: Any) -> AxesProxy:
    """``fig.add_subplot(...)`` -- matplotlib's ``Figure.add_subplot``.

    Takes every argument :func:`subplot` takes, including ``projection="3d"`` and all three
    of matplotlib's grid spellings (``add_subplot(2, 2, 1)``, ``add_subplot(221)``,
    ``add_subplot()``), and returns the panel as an :class:`AxesProxy`.

    Adding a subplot makes it the current axes, as it does in matplotlib -- so the figure
    is made current first and the module-level ``subplot`` does the rest against it, which
    is also what keeps one implementation of the grid bookkeeping rather than two.
    """
    global _CURRENT_PLOT
    _CURRENT_PLOT = self
    return subplot(*args, **kwargs)


def _figure_add_axes(self: GPULinePlot, rect: Any = None, **kwargs: Any) -> AxesProxy:
    """``fig.add_axes(rect)`` -- matplotlib's ``Figure.add_axes``.

    ``projection="3d"`` is honoured. The ``rect`` is not: placing an axes at an arbitrary
    figure rectangle needs a free-form panel, and GLPlot's panels come from a grid or a
    mosaic. It is warned about rather than raised on, so a script that positions a
    colourbar axes by hand still runs -- with one axes where it expected two, and a message
    saying so.
    """
    global _CURRENT_PLOT
    _CURRENT_PLOT = self
    return axes(rect, **kwargs)


def _figure_add_gridspec(self: GPULinePlot, nrows: int = 1, ncols: int = 1, **kwargs: Any):
    """``fig.add_gridspec(...)`` -- matplotlib's ``Figure.add_gridspec``.

    Returns a real ``matplotlib.gridspec.GridSpec``, so ``gs[0, 1:]`` slicing and
    introspection (``gs.nrows``, ``gs.get_geometry()``) work exactly as in matplotlib.
    Handing the resulting spec to ``add_subplot``/``subplot`` places a real, correctly
    spanning panel (see the ``SubplotSpec`` branch in :func:`subplot`), which is what
    ``add_gridspec`` exists for. The spacing knobs (``wspace``, ``hspace``,
    ``width_ratios``, ``height_ratios``, ``left/right/top/bottom``) are matplotlib's own
    layout-engine parameters; GLPlot's panel grid does not consult them, so they are
    passed through to the real ``GridSpec`` (whose own geometry math still needs them)
    but warned about, since they will not affect how the panels actually lay out here.
    """
    from matplotlib.gridspec import GridSpec as _MplGridSpec

    spacing_keys = (
        "left",
        "right",
        "top",
        "bottom",
        "wspace",
        "hspace",
        "width_ratios",
        "height_ratios",
    )
    _warn_unsupported(
        "add_gridspec",
        {k: kwargs.get(k) for k in spacing_keys},
        {
            k: "has no effect on GLPlot's panel grid, which does not consult "
            "matplotlib's layout engine"
            for k in spacing_keys
        },
    )
    return _MplGridSpec(nrows, ncols, **kwargs)


GPULinePlot.add_subplot = _figure_add_subplot
GPULinePlot.add_axes = _figure_add_axes
GPULinePlot.add_gridspec = _figure_add_gridspec


# ------------------------------------------------------------------
# Convenience aliases
# ------------------------------------------------------------------

lineplot = lines
points = scatter


# ------------------------------------------------------------------
# Re-exports from matplotlib (settings objects and artist/helper classes)
# ------------------------------------------------------------------
#
# ``plt.Rectangle``, ``plt.rcParams``, ``plt.Normalize`` and the rest are part of the
# pyplot namespace people actually import from, and a script that says
# ``ax.add_patch(plt.Rectangle(...))`` or ``plt.style.use('ggplot')`` has nothing to do with
# rendering -- these are geometry, colour-scaling and settings objects that matplotlib
# defines and GLPlot has no reason to reimplement. Re-exporting the real classes means a
# `Normalize` built here *is* the one `scatter(norm=...)` already accepts, rather than a
# look-alike that would fail an isinstance check somewhere downstream.
#
# Resolved lazily through PEP 562 rather than imported at module scope, matching how every
# other matplotlib touch-point in this file is written: importing GLPlot should not pull in
# matplotlib's class hierarchy for a caller who never asks for it.
_LAZY_EXPORTS: dict = {
    # Settings and registries.
    "rcParams": ("matplotlib", "rcParams"),
    "rcParamsDefault": ("matplotlib", "rcParamsDefault"),
    "colormaps": ("matplotlib", "colormaps"),
    "color_sequences": ("matplotlib", "color_sequences"),
    "cycler": ("cycler", "cycler"),
    # Colour scaling -- the `norm=` argument's own types.
    "Normalize": ("matplotlib.colors", "Normalize"),
    "LogNorm": ("matplotlib.colors", "LogNorm"),
    "SymLogNorm": ("matplotlib.colors", "SymLogNorm"),
    "PowerNorm": ("matplotlib.colors", "PowerNorm"),
    "BoundaryNorm": ("matplotlib.colors", "BoundaryNorm"),
    "TwoSlopeNorm": ("matplotlib.colors", "TwoSlopeNorm"),
    "CenteredNorm": ("matplotlib.colors", "CenteredNorm"),
    "AsinhNorm": ("matplotlib.colors", "AsinhNorm"),
    "FuncNorm": ("matplotlib.colors", "FuncNorm"),
    "NoNorm": ("matplotlib.colors", "NoNorm"),
    "Colormap": ("matplotlib.colors", "Colormap"),
    "ListedColormap": ("matplotlib.colors", "ListedColormap"),
    "LinearSegmentedColormap": ("matplotlib.colors", "LinearSegmentedColormap"),
    # Patch geometry -- what `add_patch` takes.
    "Rectangle": ("matplotlib.patches", "Rectangle"),
    "Circle": ("matplotlib.patches", "Circle"),
    "Ellipse": ("matplotlib.patches", "Ellipse"),
    "Polygon": ("matplotlib.patches", "Polygon"),
    "Wedge": ("matplotlib.patches", "Wedge"),
    "Arrow": ("matplotlib.patches", "Arrow"),
    "FancyArrow": ("matplotlib.patches", "FancyArrow"),
    "FancyBboxPatch": ("matplotlib.patches", "FancyBboxPatch"),
    "Arc": ("matplotlib.patches", "Arc"),
    "RegularPolygon": ("matplotlib.patches", "RegularPolygon"),
    "PathPatch": ("matplotlib.patches", "PathPatch"),
    # Artists a script may construct or type-check against.
    "Line2D": ("matplotlib.lines", "Line2D"),
    "Text": ("matplotlib.text", "Text"),
    "Annotation": ("matplotlib.text", "Annotation"),
    "Artist": ("matplotlib.artist", "Artist"),
    "GridSpec": ("matplotlib.gridspec", "GridSpec"),
    "SubplotSpec": ("matplotlib.gridspec", "SubplotSpec"),
    # Tick locators and formatters -- the arguments to `gca().xaxis.set_major_locator`.
    "Locator": ("matplotlib.ticker", "Locator"),
    "Formatter": ("matplotlib.ticker", "Formatter"),
    "AutoLocator": ("matplotlib.ticker", "AutoLocator"),
    "FixedLocator": ("matplotlib.ticker", "FixedLocator"),
    "IndexLocator": ("matplotlib.ticker", "IndexLocator"),
    "LinearLocator": ("matplotlib.ticker", "LinearLocator"),
    "LogLocator": ("matplotlib.ticker", "LogLocator"),
    "MaxNLocator": ("matplotlib.ticker", "MaxNLocator"),
    "MultipleLocator": ("matplotlib.ticker", "MultipleLocator"),
    "NullLocator": ("matplotlib.ticker", "NullLocator"),
    "TickHelper": ("matplotlib.ticker", "TickHelper"),
    "FixedFormatter": ("matplotlib.ticker", "FixedFormatter"),
    "FormatStrFormatter": ("matplotlib.ticker", "FormatStrFormatter"),
    "FuncFormatter": ("matplotlib.ticker", "FuncFormatter"),
    "LogFormatter": ("matplotlib.ticker", "LogFormatter"),
    "LogFormatterExponent": ("matplotlib.ticker", "LogFormatterExponent"),
    "LogFormatterMathtext": ("matplotlib.ticker", "LogFormatterMathtext"),
    "NullFormatter": ("matplotlib.ticker", "NullFormatter"),
    "ScalarFormatter": ("matplotlib.ticker", "ScalarFormatter"),
    # Event plumbing.
    "MouseButton": ("matplotlib.backend_bases", "MouseButton"),
    # Submodules pyplot re-exports, so `plt.cm.viridis` and `plt.mlab.psd` resolve.
    # A `None` attribute means "the module itself" -- `matplotlib.style` and
    # `matplotlib.mlab` are not imported by `import matplotlib`, so a plain getattr on the
    # package would miss them.
    "style": ("matplotlib.style", None),
    "cm": ("matplotlib.cm", None),
    "cbook": ("matplotlib.cbook", None),
    "mlab": ("matplotlib.mlab", None),
    "rcsetup": ("matplotlib.rcsetup", None),
    "rcParamsOrig": ("matplotlib", "rcParamsOrig"),
    # Container classes. Exported so annotations and imports resolve -- but note that
    # `gcf()` returns a `GPULinePlot`, *not* one of these, so `isinstance(fig, Figure)` is
    # False here and would be True in matplotlib. That is the honest answer: GLPlot's
    # figure is a GPU surface, and claiming otherwise would break the first method call.
    "Figure": ("matplotlib.figure", "Figure"),
    "FigureBase": ("matplotlib.figure", "FigureBase"),
    "Axes": ("matplotlib.axes", "Axes"),
    "Subplot": ("matplotlib.axes", "Subplot"),
    "PolarAxes": ("matplotlib.projections.polar", "PolarAxes"),
    "AxLine": ("matplotlib.lines", "AxLine"),
    "matplotlib": ("matplotlib", None),
}

#: Deliberately *not* re-exported, with the reason, so the omission reads as a decision.
#:
#: matplotlib's interactive widgets (``Slider``, ``Button``, ``CheckButtons``...) bind to a
#: matplotlib canvas and event loop. GLPlot has neither: a widget constructed here would
#: build without error and then never draw or fire a callback, which is the exact failure
#: the compat policy exists to avoid. GLPlot's own GUI panels are the equivalent. An
#: `AttributeError` naming the class is a better answer than a dead control.
_NOT_EXPORTED: dict = {
    "Slider": "matplotlib widgets need a matplotlib canvas and event loop; use GLPlot's "
    "GUI panels instead",
    "Button": "matplotlib widgets need a matplotlib canvas and event loop; use GLPlot's "
    "GUI panels instead",
    "Widget": "matplotlib widgets need a matplotlib canvas and event loop; use GLPlot's "
    "GUI panels instead",
    "FigureCanvasBase": "GLPlot renders through its own GPU backend and has no matplotlib "
    "canvas",
    "FigureManagerBase": "GLPlot manages its own windows; see figure() and close()",
    "BackendFilter": "matplotlib-3.9+ backend-selection enum; GLPlot has exactly one "
    "backend, so there is nothing to filter",
    "backend_registry": "matplotlib-3.9+ backend discovery/selection registry; GLPlot has "
    "exactly one backend and is not in it",
    "available_backends": "matplotlib-3.9+ backend discovery; GLPlot has exactly one "
    "backend and is not in matplotlib's registry",
    "requested_backend": "matplotlib-3.9+ backend-selection state; meaningless with a "
    "single, non-matplotlib-registered backend",
    "Colorizer": "matplotlib-3.10+ shared colour-mapping object; GLPlot resolves cmap/norm/"
    "vmin/vmax per call instead of sharing one across artists",
    "ColorizingArtist": "matplotlib-3.10+ mixin for artists bound to a shared Colorizer; "
    "GLPlot artists resolve their own colour mapping",
}


def __getattr__(name: str):
    """Resolve the matplotlib re-exports on first use (PEP 562)."""
    if name in _NOT_EXPORTED:
        raise AttributeError(f"glplot.pyplot does not provide {name!r}: {_NOT_EXPORTED[name]}.")
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    try:
        value = module if target[1] is None else getattr(module, target[1])
    except AttributeError:
        # The installed matplotlib is older or newer than this table assumes. Say so,
        # rather than surfacing a bare "module has no attribute" that looks like a bug in
        # GLPlot's own code.
        import matplotlib

        raise AttributeError(
            f"glplot.pyplot cannot re-export {name!r}: matplotlib {matplotlib.__version__} "
            f"has no {target[0]}.{target[1]}"
        ) from None
    globals()[name] = value  # Cache, so the second access is a plain global lookup.
    return value


def __dir__() -> list:
    """Include the lazy re-exports, so tab-completion and ``dir()`` find them."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


# ------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------


@atexit.register
def _cleanup_pyplot_state():
    global _CURRENT_PLOT, _CURRENT_CMAP, _CURRENT_MAPPABLE, _INTERACTIVE
    _CURRENT_PLOT = None
    _CURRENT_CMAP = None
    # Also the last strong reference to a layer whose figure is gone otherwise.
    _CURRENT_MAPPABLE = None
    _INTERACTIVE = False
    _ALL_PLOTS.clear()
    _FIGURES_BY_NUM.clear()
    # Or the first test to trip a compat warning would suppress it for every test after it,
    # making "does this kwarg warn?" depend on test execution order.
    _WARNED_UNSUPPORTED.clear()
