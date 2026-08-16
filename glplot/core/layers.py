from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Protocol, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..options import BlendMode
    from .context import RenderContext


@dataclass
class LayerStyle:
    """Encapsulates all non-geometric visual properties of a layer."""

    visible: bool = True
    alpha: float = 1.0
    zorder: int = 0
    pickable: bool = False

    # Colors
    color: Optional[Tuple[float, float, float, float]] = None  # Primary (Lines, edges)
    edge_color: Optional[Tuple[float, float, float, float]] = None  # Edges for patches
    face_color: Optional[Tuple[float, float, float, float]] = None  # Fill for patches

    # Geometry
    line_width: float = 1.0
    point_size: float = 6.0

    # Scatter Polish
    point_outline_enabled: bool = False
    point_outline_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    point_outline_width: float = 1.0

    # Outline / silhouette — the general form of the three ``point_outline_*`` fields
    # above, which stay exactly what they are: the 2D scatter's own per-marker ring
    # (``renderers/scatter.py``). These are deliberately *not* the same switch. A layer can
    # legitimately want one and not the other, the 2D scatter has shipped with the narrow
    # pair for long enough that user scripts set them by name, and folding them together
    # would change what an existing ``edgecolors=`` call does.
    #
    #: Master switch for the general outline. Read by
    #: :class:`glplot.renderers.geometry3d.Geometry3DRenderer` for all three 3D primitives
    #: (points, lines, triangles) and by the Style panel, which owns the UI for it. False
    #: is load-bearing: every outline branch in ``GEOMETRY3D_VS``/``GEOMETRY3D_FS`` is
    #: gated on it, so an untouched layer renders exactly as it did before outlines existed.
    outline_enabled: bool = False
    #: Outline RGBA. Read by ``Geometry3DRenderer`` (uniform ``u_outline_color``). Black is
    #: the value the existing ``point_outline_color`` defaults to, so the two read alike
    #: when both are switched on.
    outline_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    #: Outline thickness in *logical* pixels — ``Geometry3DRenderer`` multiplies by
    #: ``ctx.dpr`` before it reaches the shader, exactly as ``point_size`` is handled, so
    #: an outline keeps its apparent width on a Retina display and in a scaled ``savefig``.
    #: 0 (or less) disables the outline as surely as the flag does; the renderer checks.
    outline_width: float = 1.5
    #: Opacity of the outline, multiplied into ``outline_color``'s own alpha. Read by
    #: ``Geometry3DRenderer`` (uniform ``u_outline_alpha``). Separate from :attr:`alpha` on
    #: purpose: an outline drawn on a translucent layer exists to keep that layer readable,
    #: so it must be able to stay opaque while the layer fades. The renderer does fold
    #: ``ctx.global_alpha`` in, so a whole-scene fade still fades the outlines with it.
    outline_alpha: float = 1.0

    # Colormapping
    use_colormap: bool = False
    cmap: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None

    # Text
    text_size_px: float = 12.0

    # -- Compositing -------------------------------------------------------------------
    #
    # The three levers over how a layer merges with what is already on screen. All three
    # default to "whatever the figure/renderer decides", so an untouched layer composites
    # exactly as it did before they existed, and all three are read by
    # :class:`~glplot.renderers.geometry3d.Geometry3DRenderer` -- the 2D renderers still
    # take the figure's blend mode for the whole pass (see the Style panel's
    # ``_draw_layer_blending``, which says so rather than offering a switch that does
    # nothing).

    #: Blend mode for this layer alone; ``None`` inherits ``options.blend_mode``.
    #:
    #: The reason this is worth having per layer is that one figure legitimately wants two:
    #: a volumetric cloud reads far better in ``ADDITIVE``, where overlapping points get
    #: *brighter* instead of averaging out and the faint tails survive at alpha 0.02, while
    #: the surfaces and axes in the same scene need ordinary ``ALPHA`` or they wash out.
    blend_mode: Optional["BlendMode"] = None

    #: Whether this layer writes the depth buffer -- i.e. whether it occludes what is drawn
    #: after it. ``None`` leaves the decision to the renderer, which turns writes off for
    #: translucent points and lines and keeps them on for meshes (see
    #: ``Geometry3DRenderer.draw`` for why the two differ). ``True``/``False`` force it.
    #:
    #: The case for forcing it off is a translucent surface you want to see *through*,
    #: including through its own far side; the case for forcing it on is a cloud dense
    #: enough to read as a solid body that should hide what is behind it.
    depth_write: Optional[bool] = None

    #: Target opacity for automatic alpha, or ``None`` for the alpha the caller set.
    #:
    #: A fixed alpha is wrong at every zoom but one: markers keep their pixel size, so the
    #: same cloud piles ten points on a pixel when zoomed out and none when zoomed in --
    #: which is the "sometimes it saturates, sometimes there is nothing there" swing. Set
    #: this to the opacity a *typical covered pixel* should reach (0.9 is a good starting
    #: point) and the renderer solves for the per-point alpha that gets there under the
    #: current view, re-doing it as the view changes.
    auto_alpha: Optional[float] = None


@dataclass
class LayerDirtyState:
    """Fine-grained invalidation flags to optimize GPU updates."""

    data_dirty: bool = True
    style_dirty: bool = True
    gpu_dirty: bool = True
    bounds_dirty: bool = True

    def clear(self) -> None:
        self.data_dirty = False
        self.style_dirty = False
        self.gpu_dirty = False
        self.bounds_dirty = False


class CompiledLayer:
    """GPU-ready geometry and cached bounds."""

    def __init__(self, layer_id: int) -> None:
        self.layer_id = layer_id
        self.bounds_world: Optional[Tuple[float, float, float, float]] = None
        self.gpu_initialized: bool = False


class BaseLayer:
    """Abstract base for all visual primitives."""

    def __init__(self, layer_type: str, label: str = "") -> None:
        self.layer_id = uuid.uuid4().int & (1 << 31) - 1
        self.layer_type = layer_type
        self.label = label
        self.style = LayerStyle()
        self.dirty = LayerDirtyState()
        self.bounds_world: Optional[Tuple[float, float, float, float]] = None
        self.translation: Tuple[float, float] = (0.0, 0.0)
        self.metadata: dict[str, Any] = {}

    def get_intrinsic_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        return None


#: Every concrete layer is ``@dataclass(eq=False)``, never a plain ``@dataclass``.
#:
#: A generated ``__eq__`` compares fields, and every one of these classes holds ndarrays --
#: so ``layer_a == layer_b`` raises "operands could not be broadcast together" and, worse,
#: ``layer in scene.layers`` raises the same thing from inside ``list.__contains__``. That
#: is why ``layerops`` carries its own ``_index_of`` / ``_remove_identity`` helpers: they
#: exist to avoid an ``in`` that cannot work.
#:
#: ``eq=False`` restores identity semantics, which is the correct comparison here anyway --
#: two layers holding equal arrays are still two different layers in the scene. The same
#: reasoning, and the same fix, as :class:`glplot.gui.datasets.Column`.


@dataclass(eq=False)
class LineFamilyLayer(BaseLayer):
    """High-performance layer for millions of lines y = ax + b."""

    ab: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    x_range: Tuple[float, float] = (-1.0, 1.0)

    def __init__(
        self,
        ab: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        label: str = "",
    ) -> None:
        super().__init__(layer_type="line_family", label=label)
        self.ab = ab
        self.colors = colors
        self.x_range = x_range

    def get_intrinsic_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        if self.ab is None or len(self.ab) == 0:
            return None
        x0, x1 = self.x_range
        y_at_x0 = self.ab[:, 0] * x0 + self.ab[:, 1]
        y_at_x1 = self.ab[:, 0] * x1 + self.ab[:, 1]
        return (
            x0,
            x1,
            float(min(np.min(y_at_x0), np.min(y_at_x1))),
            float(max(np.max(y_at_x0), np.max(y_at_x1))),
        )


@dataclass(eq=False)
class ScatterLayer(BaseLayer):
    """Layer for point clouds."""

    pts: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    #: Optional per-point marker size in the same units as ``style.point_size`` (pixels).
    #: ``None`` means "every point uses ``style.point_size``". Shape ``(N,)``. Set from a
    #: data variable to make marker size a data-driven dimension (matplotlib ``s`` array).
    sizes: Optional[np.ndarray] = None

    def __init__(
        self,
        pts: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        size: float = 6.0,
        label: str = "",
        sizes: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(layer_type="scatter", label=label)
        self.pts = pts
        self.colors = colors
        self.style.point_size = size
        self.sizes = None if sizes is None else np.asarray(sizes, dtype=np.float32).ravel()

    def get_intrinsic_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        if self.pts is None or len(self.pts) == 0:
            return None
        return (
            float(np.min(self.pts[:, 0])),
            float(np.max(self.pts[:, 0])),
            float(np.min(self.pts[:, 1])),
            float(np.max(self.pts[:, 1])),
        )


@dataclass(eq=False)
class PolylineLayer(BaseLayer):
    """Layer for connected line segments (Polyline)."""

    pts: Optional[np.ndarray] = None
    #: Optional per-vertex RGBA (shape (N, 4)) to colour the line by a data variable. Each
    #: segment interpolates between its two endpoints' colours. ``None`` = flat ``style.color``.
    colors: Optional[np.ndarray] = None

    def __init__(
        self,
        pts: Optional[np.ndarray] = None,
        color: Optional[Tuple[float, float, float, float]] = None,
        width: float = 1.0,
        label: str = "",
        colors: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(layer_type="polyline", label=label)
        self.pts = pts
        if color:
            self.style.color = color
        self.style.line_width = width
        self.colors = None if colors is None else np.asarray(colors, dtype=np.float32)

    def get_intrinsic_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        if self.pts is None or len(self.pts) == 0:
            return None
        return (
            float(np.min(self.pts[:, 0])),
            float(np.max(self.pts[:, 0])),
            float(np.min(self.pts[:, 1])),
            float(np.max(self.pts[:, 1])),
        )


@dataclass(eq=False)
class PatchLayer(BaseLayer):
    """Layer for filled areas (tri-strips, bars, rects)."""

    vertices: Optional[np.ndarray] = None  # (N, 2)
    indices: Optional[np.ndarray] = None  # (M,)
    mode: str = "strip"  # "strip", "triangles", "rects"
    #: Per-vertex RGBA, shape (N, 4). None means the whole patch is `style.face_color`,
    #: which is what a bar, a fill_between or a pie wedge wants and is the default the
    #: renderer's uniform path serves. It is not None for a patch whose pieces carry their
    #: own colours -- a hexbin's hexagons -- which one uniform cannot express. See
    #: `renderers.patch.PatchRenderer` and the `u_use_vertex_color` gate in `PATCH_VS`.
    colors: Optional[np.ndarray] = None

    def __init__(
        self,
        vertices: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
        mode: str = "strip",
        label: str = "",
        colors: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(layer_type="patch", label=label)
        self.vertices = vertices
        self.indices = indices
        self.mode = mode
        self.colors = colors

    def get_intrinsic_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        if self.vertices is None or len(self.vertices) == 0:
            return None
        return (
            float(np.min(self.vertices[:, 0])),
            float(np.max(self.vertices[:, 0])),
            float(np.min(self.vertices[:, 1])),
            float(np.max(self.vertices[:, 1])),
        )


@dataclass(eq=False)
class FunctionLayer(PolylineLayer):
    """A curve sampled in **screen space**, re-evaluated whenever the view changes.

    Every other line layer here stores points sampled once, in data space. Zooming such a
    line only magnifies the samples it was born with: zoom far enough into ``sin(1/x)`` and
    a smooth curve turns into a handful of straight segments, because the detail was never
    computed. Zoom *out* and you are paying to draw a million points into a thousand pixels.

    This layer instead fixes the sample count to the **screen**: about one sample per pixel
    column of the viewport, spread across whatever x range is currently visible. The
    consequences are the point of the design:

    * **Resolution is constant.** The curve is always sampled at the density the display
      can actually show — never coarser, never wastefully finer.
    * **Cost is constant.** ~2000 evaluations per frame regardless of zoom depth, so a
      1e-12 zoom costs exactly what the full view costs. (Contrast
      :class:`FractalLayer`, which must evaluate *per pixel of area* and therefore gets
      slower the deeper you go.)
    * **Detail is unbounded.** Zooming re-evaluates ``f`` on the new interval, so features
      finer than the original sampling appear as you approach them, the way a graphing
      calculator behaves.

    That makes it the right layer for an **analytic or recursive function** — anything you
    can evaluate at an arbitrary x rather than a fixed table of measurements. It is
    explicitly *not* for measured data: there is no ``f`` to re-evaluate, and resampling
    would be inventing values.

    matplotlib has no equivalent — ``plot(x, y)`` is a fixed table, and its zoom never
    re-samples. Desmos-style graphing tools and Datashader's per-view re-aggregation work
    on this principle, but within the scientific-Python plotting stack this is new.

    ``func`` must be vectorised: it receives an ``(n,)`` float64 array of x and returns an
    ``(n,)`` array of y. Non-finite results are kept as ``nan``, which the renderer draws
    as a gap — that is what makes a pole in ``1/x`` a break rather than a spike across the
    plot.

    Attributes
    ----------
    func
        ``f(x: ndarray) -> ndarray``.
    domain
        Optional ``(x0, x1)`` outside which the function is not evaluated at all, for a
        function that is only defined on an interval. ``None`` means "wherever the view is".
    samples_per_px
        Sample density relative to the viewport width. 1.0 is one per pixel column; 2.0
        supersamples, which smooths a steep curve at the cost of double the evaluations.
    min_samples, max_samples
        Bounds on the resulting count, so a tiny window still gets a usable curve and a
        4K display cannot ask for an unbounded number of evaluations.
    """

    func: Optional[Any] = None
    domain: Optional[Tuple[float, float]] = None
    samples_per_px: float = 1.0
    min_samples: int = 64
    max_samples: int = 8192

    def __init__(
        self,
        func: Optional[Any] = None,
        *,
        domain: Optional[Tuple[float, float]] = None,
        samples_per_px: float = 1.0,
        min_samples: int = 64,
        max_samples: int = 8192,
        color: Optional[Tuple[float, float, float, float]] = None,
        width: float = 1.5,
        label: str = "",
    ) -> None:
        super().__init__(pts=None, color=color, width=width, label=label)
        self.func = func
        self.domain = None if domain is None else (float(domain[0]), float(domain[1]))
        self.samples_per_px = float(samples_per_px)
        self.min_samples = int(min_samples)
        self.max_samples = int(max_samples)
        #: The x interval the current ``pts`` were sampled over. The engine compares the
        #: live view against this to decide whether a resample is owed, so an idle figure
        #: costs nothing.
        self.sampled_window: Optional[Tuple[float, float]] = None
        #: The sample count last used, so a window resize also triggers a resample.
        self.sampled_count: int = 0

    def sample_count(self, width_px: int) -> int:
        """How many samples this viewport width deserves, clamped to the layer's bounds."""
        wanted = int(round(max(int(width_px), 1) * max(self.samples_per_px, 1e-3)))
        return int(np.clip(wanted, max(self.min_samples, 2), max(self.max_samples, 2)))

    def needs_resample(self, x0: float, x1: float, width_px: int) -> bool:
        """Whether the visible interval has moved enough to be worth re-evaluating.

        A tolerance rather than an exact compare: a camera can jitter by a float epsilon
        while nothing is happening, and re-evaluating a function every idle frame would
        burn CPU for an identical curve. A thousandth of the span is far below one pixel.
        """
        if self.func is None or self.pts is None or self.sampled_window is None:
            return True
        if self.sample_count(width_px) != self.sampled_count:
            return True
        was0, was1 = self.sampled_window
        span = max(abs(was1 - was0), 1e-30)
        return abs(x0 - was0) > span * 1e-3 or abs(x1 - was1) > span * 1e-3

    def resample(self, x0: float, x1: float, width_px: int) -> bool:
        """Re-evaluate ``func`` across ``[x0, x1]``. True when the points changed.

        Never raises: a function that blows up on some input must not take the frame down,
        so an evaluation failure leaves the previous curve on screen. That is the same
        bargain the Functions panel makes with a half-typed expression.
        """
        if self.func is None:
            return False
        lo, hi = (float(x0), float(x1)) if x1 > x0 else (float(x1), float(x0))
        if self.domain is not None:
            lo = max(lo, self.domain[0])
            hi = min(hi, self.domain[1])
            if not hi > lo:
                # The view has left the function's domain entirely: draw nothing rather
                # than extrapolating into a region where it is not defined.
                self.pts = np.zeros((0, 2), dtype=np.float32)
                self.sampled_window = (float(x0), float(x1))
                self.sampled_count = 0
                self.dirty.gpu_dirty = True
                return True

        n = self.sample_count(width_px)
        xs = np.linspace(lo, hi, n, dtype=np.float64)
        try:
            # errstate, because this runs every time the view moves: a function with a pole
            # would otherwise emit a divide-by-zero warning on every frame of a drag. The
            # non-finite results it produces are wanted — they become the gaps below.
            with np.errstate(all="ignore"):
                ys = np.asarray(self.func(xs), dtype=np.float64)
        except Exception:  # pragma: no cover - a user function is arbitrary
            return False
        if ys.shape != xs.shape:
            # A non-vectorised function that returned a scalar still has a sane reading.
            try:
                ys = np.broadcast_to(ys, xs.shape).astype(np.float64)
            except Exception:  # pragma: no cover - defensive
                return False

        # inf would stretch the autoscale to infinity; nan is the renderer's gap marker.
        ys = np.where(np.isfinite(ys), ys, np.nan)
        self.pts = np.ascontiguousarray(np.column_stack([xs, ys]), dtype=np.float32)
        self.sampled_window = (float(x0), float(x1))
        self.sampled_count = n
        self.dirty.gpu_dirty = True
        self.dirty.bounds_dirty = True
        return True

    def get_intrinsic_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Bounds ignoring the gaps.

        Overridden because gaps are normal here — ``tan`` is nan at every pole — and the
        inherited ``np.min`` would return nan for the whole layer, which then propagates
        into autoscale and blanks the figure.
        """
        if self.pts is None or len(self.pts) == 0:
            return None
        ys = self.pts[:, 1]
        finite = np.isfinite(ys)
        if not finite.any():
            return None
        ys = ys[finite]
        return (
            float(np.min(self.pts[:, 0])),
            float(np.max(self.pts[:, 0])),
            float(np.min(ys)),
            float(np.max(ys)),
        )


@dataclass(eq=False)
class FractalLayer(BaseLayer):
    """A view-dependent escape-time field computed live on the GPU.

    Unlike every other layer here it carries **no sampled data** — only the *definition* of
    a field (``fractal_type``, the region it lives over, the iteration budget, the colour
    mapping). The pixels are produced by :class:`glplot.renderers.fractal.FractalRenderer`'s
    fragment shader at draw time, per screen fragment, so the same layer is infinitely
    re-detailed as the camera zooms into it. See :data:`glplot.utils.shaders.FRACTAL_FS`.

    This is the general shape of a "view-driven layer": its content is a pure function of
    the current view, recomputed when the view changes. Fractals are the case where that
    function is cheap enough to run per pixel every frame; a CPU-callback version of the
    same idea (any Python ``f(extent, size)``) is the counterpart for arbitrary fields.

    Attributes
    ----------
    extent
        ``(x0, x1, y0, y1)`` world region the field is defined over. Autoscale frames this,
        after which the user zooms freely and the shader recomputes at the new scale.
    fractal_type
        ``"mandelbrot"`` or ``"julia"``.
    julia_c
        The Julia parameter ``c`` (ignored for Mandelbrot).
    max_iter
        Base iteration budget. The renderer raises it as the view zooms in, because deeper
        zoom needs more iterations to resolve the boundary.
    cmap, gain
        Colour scheme name and spread. ``inset_color`` is the colour of the set itself.
    """

    extent: Tuple[float, float, float, float] = (-2.2, 0.8, -1.25, 1.25)
    fractal_type: str = "mandelbrot"
    julia_c: Tuple[float, float] = (-0.8, 0.156)
    max_iter: int = 200
    cmap: str = "magma"
    gain: float = 1.0
    inset_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __init__(
        self,
        extent: Tuple[float, float, float, float] = (-2.2, 0.8, -1.25, 1.25),
        fractal_type: str = "mandelbrot",
        julia_c: Tuple[float, float] = (-0.8, 0.156),
        max_iter: int = 200,
        cmap: str = "magma",
        gain: float = 1.0,
        inset_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        label: str = "",
    ) -> None:
        super().__init__(layer_type="fractal", label=label)
        self.extent = tuple(float(v) for v in extent)
        self.fractal_type = str(fractal_type)
        self.julia_c = (float(julia_c[0]), float(julia_c[1]))
        self.max_iter = int(max_iter)
        self.cmap = str(cmap)
        self.gain = float(gain)
        self.inset_color = tuple(float(v) for v in inset_color)

    def get_intrinsic_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """The extent, so autoscale frames the field the first time it is shown."""
        x0, x1, y0, y1 = self.extent
        return (float(x0), float(x1), float(y0), float(y1))


@dataclass(eq=False)
class TextLayer(BaseLayer):
    """Layer for text labels proyected from world coordinates."""

    x: float = 0.0
    y: float = 0.0
    text: str = ""

    def __init__(self, x: float = 0.0, y: float = 0.0, text: str = "", label: str = "") -> None:
        super().__init__(layer_type="text", label=label)
        self.x = x
        self.y = y
        self.text = text

    def get_intrinsic_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        # Text does not participate in autoscale by default
        return None


@dataclass(eq=False)
class Layer3D(BaseLayer):
    """Generic GPU 3D geometry layer."""

    vertices: Optional[np.ndarray] = None  # (N, 3)
    colors: Optional[np.ndarray] = None  # (N, 4)
    indices: Optional[np.ndarray] = None
    primitive: str = "points"  # points, lines, triangles
    #: Optional per-point marker size (pixels), like :attr:`ScatterLayer.sizes`. Only used by
    #: ``points`` primitives; ``None`` means every point uses ``style.point_size``.
    sizes: Optional[np.ndarray] = None

    def __init__(
        self,
        vertices: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
        primitive: str = "points",
        label: str = "",
        layer_type: str = "geometry3d",
        sizes: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(layer_type=layer_type, label=label)
        self.vertices = vertices
        self.colors = colors
        self.indices = indices
        self.primitive = primitive
        self.style.point_size = 3.0
        self.sizes = None if sizes is None else np.asarray(sizes, dtype=np.float32).ravel()
        # Memo for get_bounds_3d; see its docstring for the invalidation rule.
        self._bounds3d_key: Optional[Tuple[Any, ...]] = None
        self._bounds3d_value: Optional[Tuple[float, float, float, float, float, float]] = None

    def get_intrinsic_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """The x/y half of :meth:`get_bounds_3d`, and memoised with it."""
        bounds = self.get_bounds_3d()
        if bounds is None:
            return None
        return (bounds[0], bounds[1], bounds[2], bounds[3])

    def get_bounds_3d(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """The axis-aligned box around :attr:`vertices`, **memoised**.

        The cache is not an optimisation of a cold path -- it is what makes 3D interaction
        possible at all. One camera nudge runs ``get_3d_bounds`` four times (``sync_3d_camera``
        computes raw *and* padded bounds, then ``ensure_3d_axes`` computes both again), and
        the renderer asks every layer for its box once more per draw. On a 420k-point cloud
        that was ~22 full scans of a 5 MB array per frame: 3 ms in the steady state and
        60-80 ms whenever the array fell out of cache, which is exactly the periodic hitch
        that made a rotate drag jump.

        Invalidation follows the array rather than a flag, so the common case needs no
        cooperation from the caller: swapping ``vertices`` for a new array changes ``id``
        and the cache misses by itself. A caller that mutates the array *in place* must
        raise ``dirty.gpu_dirty`` or ``dirty.bounds_dirty`` -- which is already the
        contract for in-place edits (see ``gui/datasets.py`` CONTRACT 1.4), and both flags
        are part of the key, so honouring it invalidates the bounds too.

        The scan itself is ``min``/``max`` over ``axis=0`` rather than six reductions on
        strided columns: same numbers, one sequential pass over the buffer instead of six
        stride-3 ones.
        """
        vertices = self.vertices
        if vertices is None or len(vertices) == 0:
            return None
        key = (
            id(vertices),
            vertices.shape,
            bool(self.dirty.gpu_dirty),
            bool(self.dirty.bounds_dirty),
        )
        if key == self._bounds3d_key:
            return self._bounds3d_value
        lo = np.min(vertices, axis=0)
        hi = np.max(vertices, axis=0)
        value = (
            float(lo[0]),
            float(hi[0]),
            float(lo[1]),
            float(hi[1]),
            float(lo[2]),
            float(hi[2]),
        )
        self._bounds3d_key = key
        self._bounds3d_value = value
        return value
