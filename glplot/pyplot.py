from __future__ import annotations

import atexit
from typing import Optional, Tuple, Sequence, Union, Literal, Iterable, Any

import numpy as np

from .engine import GPULinePlot


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
        'white': (1.0, 1.0, 1.0, 1.0),
        'black': (0.0, 0.0, 0.0, 1.0),
        'red':   (1.0, 0.0, 0.0, 1.0),
        'green': (0.0, 1.0, 0.0, 1.0),
        'blue':  (0.0, 0.0, 1.0, 1.0),
        'cyan':  (0.0, 1.0, 1.0, 1.0),
        'magenta': (1.0, 0.0, 1.0, 1.0),
        'yellow': (1.0, 1.0, 0.0, 1.0),
        'k': (0.0, 0.0, 0.0, 1.0),
        'w': (1.0, 1.0, 1.0, 1.0),
        'r': (1.0, 0.0, 0.0, 1.0),
        'g': (0.0, 1.0, 0.0, 1.0),
        'b': (0.0, 0.0, 1.0, 1.0),
    }

    if color is None:
        base = np.asarray(default, dtype=np.float32)
    elif isinstance(color, str):
        c_val = COLOR_MAP.get(color.lower(), (0, 0, 0, 1))
        base = np.asarray(c_val, dtype=np.float32)
    else:
        try:
            base = np.asarray(color, dtype=np.float32)
        except (ValueError, TypeError):
             base = np.asarray(default, dtype=np.float32)

    if n is None:
        if base.ndim == 0: # single value broadcast
             base = np.array([base, base, base, 1.0], dtype=np.float32)
        if base.shape != (4,):
             # Fallback if it's RGB
             if base.shape == (3,):
                 base = np.array([base[0], base[1], base[2], 1.0], dtype=np.float32)
             else:
                 raise ValueError(f"color must be a single RGBA tuple with shape (4,), got {base.shape}")
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
    *,
    hud: bool = False,
    density: bool = False,
    blending: BlendMode = "auto",
    lod: bool = True,
    budget: int = 8,
    multisample: bool = False,
    cache: bool = True,
    clipping: bool = True,
) -> GPULinePlot:
    """
    Create a new figure and make it current.
    """
    global _CURRENT_PLOT
    plot = GPULinePlot(width=width, height=height, title=title)

    # Apply optimization settings
    plot.options.lod_enabled = bool(lod)
    plot.options.lod_target_coverage = float(budget) / 8.0 
    plot.options.enable_hud = bool(hud)
    plot.options.enable_multisample = bool(multisample)
    plot.options.enable_cache_interaction_path = bool(cache)
    plot.options.enable_clipping_optimization = bool(clipping)
    
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
    if _call_if_exists(plot, ("clear", "clf", "reset_scene")) is not None:
        _set_dirty(plot)
        return

    # Conservative fallback
    if hasattr(plot, "_line_strips"):
        plot._line_strips.clear()
    if hasattr(plot, "_scatters"):
        plot._scatters.clear()
    if hasattr(plot, "_spatial_texts"):
        plot._spatial_texts.clear()
    if hasattr(plot, "N"):
        plot.N = 0
    if hasattr(plot, "_cpu_ab"):
        plot._cpu_ab = None
    _set_dirty(plot)


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


def plot(
    x: Sequence[float],
    y: Sequence[float],
    color: ColorLike = (0.0, 0.0, 0.0, 1.0),
    width: float = 1.0,
    alpha: Optional[float] = None,
    label: Optional[str] = None,
):
    """
    Plot a traditional connected polyline.
    """
    plot_obj = _get_or_create_plot()
    x_arr = _as_float_array(x, ndim=1, name="x")
    y_arr = _as_float_array(y, ndim=1, name="y")

    if len(x_arr) != len(y_arr):
        raise ValueError("x and y must have the same length")
    if len(x_arr) < 2:
        return plot_obj

    rgba = list(_normalize_rgba(color, n=None))
    if alpha is not None:
        rgba[3] *= float(alpha)

    plot_obj.add_line_strip(x_arr, y_arr, tuple(rgba), width=float(width), label=label)
    _set_dirty(plot_obj)
    return plot_obj


def scatter(
    x: Sequence[float],
    y: Sequence[float],
    color: ColorLike = (0.0, 0.0, 0.0, 1.0),
    size: float = 10.0,
):
    """
    Scatter plot.
    """
    plot_obj = _get_or_create_plot()
    x_arr = _as_float_array(x, ndim=1, name="x")
    y_arr = _as_float_array(y, ndim=1, name="y")

    if len(x_arr) != len(y_arr):
        raise ValueError("x and y must have the same length")

    cols = _normalize_rgba(color, n=len(x_arr))
    plot_obj.add_scatter(x_arr, y_arr, cols, float(size))
    _set_dirty(plot_obj)
    return plot_obj


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
        verts, indices=indices, mode=mode, 
        face_color=tuple(f_col) if f_col is not None else None,
        edge_color=tuple(e_col) if e_col is not None else None,
        label=label
    )
    _set_dirty(plot_obj)
    return plot_obj


# ------------------------------------------------------------------
# View / styling / policies
# ------------------------------------------------------------------

def title(s: str) -> None:
    plot = _get_or_create_plot()
    _set_title(plot, s)


def xlim(left: Optional[float] = None, right: Optional[float] = None) -> Optional[Tuple[float, float]]:
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


def ylim(bottom: Optional[float] = None, top: Optional[float] = None) -> Optional[Tuple[float, float]]:
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


def axis(mode: Union[str, Tuple[float, float, float, float]] = "auto") -> Optional[Tuple[float, float, float, float]]:
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
                plot.set_view(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0)) # Absolute reset fallback
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

    if hasattr(plot, "enable_subsample"):
        plot.enable_subsample = bool(enabled)
    if hasattr(plot, "max_lines_per_px"):
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
        if _call_if_exists(plot, ("save_current_view", "export_high_res"), fname, scale=scale) is None:
            raise AttributeError("Backend does not expose export functions")


def savefig(filename: str, density: Optional[bool] = None, scale: float = 2.0):
    plot = _get_or_create_plot()

    if density is not None:
        _set_density(plot, density)

    # Preferred path: direct headless/offscreen export
    if hasattr(plot, "savefig"):
        plot.savefig(filename, scale=scale)
        return

    # Fallback path: one-frame initialization then export
    if hasattr(plot, "_is_test_mode"):
        plot._is_test_mode = True
    plot.run()
    if hasattr(plot, "savefig"):
        plot.savefig(filename, scale=scale)
    elif _call_if_exists(plot, ("save_current_view",), filename, scale=scale) is None:
        raise AttributeError("Backend does not expose a compatible export function")


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
