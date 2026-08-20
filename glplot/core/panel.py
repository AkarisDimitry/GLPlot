"""Panel: one independent plotting region inside a figure.

A :class:`Panel` bundles the per-region state that the engine historically kept as
single top-level attributes -- the scene (layers), the camera, its controller, and the
interaction/cache state -- together with the rectangle it occupies inside the window.

The engine holds ``self.panels: list[Panel]`` and an ``active_panel_index``. It exposes
``scene`` / ``camera`` / ``camera_controller`` / ``interaction`` / ``cache`` as properties
that delegate to the *active* panel, so every code path that predates multi-panel support
(pyplot, the GUI, the tests) keeps working unchanged: with a single full-window panel the
behaviour is identical to the pre-panel engine.

Rectangles are stored in *figure fractions* ``(x0, y0, w, h)`` with the origin at the
bottom-left, matching matplotlib's figure-fraction convention. ``pixel_rect`` converts a
fraction to an OpenGL viewport ``(x, y, w, h)`` in framebuffer pixels (also bottom-left
origin, which is what ``glViewport`` wants).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from ..controllers import CameraController
from .camera3d import Axes3DOptions, Camera3D
from .legacy import CacheState, CameraState, InteractionState, SceneData
from .timeline import Timeline

if TYPE_CHECKING:
    from ..options import EngineOptions


@dataclass
class ColorbarSpec:
    """One colorbar attached to a :class:`Panel`, drawn as a screen-space overlay.

    Deliberately not a :class:`Panel` itself: any rect registered on ``fig.panels`` is
    click-to-activate and pan/zoom-able (see ``GPULinePlot._panel_index_at``), so a
    colorbar built that way would silently steal "current axes" the moment a script did
    ``im = plt.imshow(...); plt.colorbar(); plt.title(...)`` -- the title would land on
    the colorbar. A ``ColorbarSpec`` instead just describes a strip of the figure and is
    drawn by :func:`glplot.renderers.colorbar.draw_colorbars`, in the same whole-window,
    camera-free pixel space :meth:`GPULinePlot._draw_panel_borders` already draws in.

    ``norm`` is always a concrete, bounded ``matplotlib.colors.Normalize`` instance (never
    a bare scale name and never unbounded) -- the single source of truth the gradient fill,
    the tick locator, and the headless ``savefig()`` reconstruction all read alike, so the
    live bar and the exported PNG cannot independently drift from what was actually asked
    for.
    """

    #: (x0, y0, w, h) in figure fractions, bottom-left origin -- same convention as
    #: :attr:`Panel.rect_frac`, and the strip vacated from the host panel's own rect.
    rect_frac: Tuple[float, float, float, float]
    #: "vertical" for a right/left bar, "horizontal" for a top/bottom one.
    orientation: str
    #: Which edge of the host panel this bar sits against: "right", "left", "top" or
    #: "bottom". Distinct from ``orientation`` because a caller can ask for either
    #: independently, as matplotlib's own ``colorbar()`` does.
    location: str
    #: Colormap name, resolved (never ``None``) at ``colorbar()`` call time.
    cmap: str
    #: A concrete ``matplotlib.colors.Normalize`` (or subclass) instance with real
    #: ``vmin``/``vmax`` already resolved.
    norm: Any
    #: Explicit tick positions, or ``None`` to auto-locate from ``norm``'s type.
    ticks: Optional[Any] = None
    #: A tick label format (matplotlib ``Formatter``, format string, or ``None`` to
    #: auto-format from the locator).
    format: Optional[Any] = None
    #: The colorbar's own axis label (``Colorbar.set_label``), or ``None``.
    label: Optional[str] = None
    #: Matplotlib-parity placement knobs, stored for the headless reconstruction (which
    #: rebuilds a real matplotlib colorbar and can honour them precisely); the live
    #: renderer's own bar geometry does not re-derive a precise aspect-ratio width from
    #: these -- see ``colorbar()``'s docstring.
    fraction: float = 0.15
    pad: float = 0.05
    shrink: float = 1.0
    aspect: float = 20.0
    #: When True, the bar sits *inside* the host panel's own rect (over the plotted
    #: content, near ``location``'s edge) instead of shrinking the panel to make room
    #: outside it. Ticks/label flip to point inward, toward the panel's centre, since
    #: there is no room outside the panel for them to point into.
    inset: bool = False


class Panel:
    """One plotting region: its scene, camera, interaction state, and window rectangle."""

    def __init__(
        self,
        options: "EngineOptions",
        rect_frac: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        name: Optional[str] = None,
    ) -> None:
        self.options = options
        #: Optional label (mosaic key, or user-facing name). ``None`` for grid panels.
        self.name = name
        #: (x0, y0, w, h) in figure fractions, bottom-left origin.
        self.rect_frac: Tuple[float, float, float, float] = rect_frac

        self.scene = SceneData()
        self.camera = CameraState()
        self.camera_controller = CameraController(self.camera, options)
        self.interaction = InteractionState()
        self.cache = CacheState()

        #: The 3D point of view and the decoration drawn around it. Always present, even in
        #: a 2D panel: constructing them is free, and a panel that later gains a 3D layer
        #: must not have to invent a camera mid-frame.
        self.camera3d = Camera3D()
        self.axes3d = Axes3DOptions()

        #: This panel's animation: keyframes, scenes, frame rate and playhead.
        #:
        #: Per panel for the same reason the cameras are: a split figure must be able to
        #: animate one region while another holds still, and an animation that lived on the
        #: figure could only ever do all of them at once. Empty and paused by default, so a
        #: panel that is never animated costs one object and no frame time.
        self.timeline = Timeline()

        #: Colorbars drawn against this panel's edges. Metadata only -- never a
        #: :class:`Panel` of its own, never registered on ``fig.panels`` -- so a
        #: colorbar can never become the active/pannable axes. See
        #: :class:`ColorbarSpec` for why.
        self.colorbars: List["ColorbarSpec"] = []

        #: This panel's own axis names and axes title.
        #:
        #: Per panel for the same reason the cameras and scenes are: a split figure plots
        #: four different quantities and has to be able to *name* them four different
        #: ways. These were figure-global until they lived here, so a 2x2 grid could only
        #: ever carry one x-name, one y-name and one title between all four panels --
        #: whichever call ran last won, and the other three axes went unlabelled.
        #:
        #: The engine forwards ``xlabel``/``ylabel`` to the active panel's copy (the same
        #: delegation ``scene``/``camera``/``cache`` use), so ``gplt.xlabel(...)`` keeps
        #: writing "the current axes' x-name" and every existing single-panel caller is
        #: unaffected. ``title`` is *not* forwarded that way -- ``engine.title`` is also
        #: the GLFW window caption and is assigned in ``__init__`` before any panel
        #: exists -- so ``set_title`` writes both, and readers prefer this one.
        self.xlabel: str = ""
        self.ylabel: str = ""
        self.title: str = ""

        #: Explicit dimensionality: 2, 3, or ``None`` for "infer from the layers".
        #:
        #: None is the default and reproduces GLPlot's historical behaviour exactly — a
        #: scene is 3D when it holds a ``*3d`` layer and 2D otherwise. Setting it to 3
        #: (which ``projection="3d"`` does) makes an *empty* panel 3D, which is what lets
        #: the workstation offer 3D tools before any data exists. See
        #: :meth:`glplot.engine.GPULinePlot.is_3d_scene`.
        self.ndim: Optional[int] = None

        #: Axis-link groups (populated by sharex/sharey). ``None`` means "independent".
        self.sharex_group = None
        self.sharey_group = None

        #: Whether this panel still owes an initial autoscale-to-fit.
        self.needs_initial_autoscale: bool = True

    # -- dimensionality ------------------------------------------------

    def is_3d(self) -> bool:
        """True when this panel should be treated as 3D.

        An explicit :attr:`ndim` always wins; otherwise the scene's layers decide, which is
        the rule every pre-``ndim`` code path used.
        """
        if self.ndim is not None:
            return int(self.ndim) == 3
        return any(
            str(getattr(layer, "layer_type", "")).endswith("3d") for layer in self.scene.layers
        )

    def set_ndim(self, ndim: Optional[int]) -> None:
        """Pin this panel to 2D or 3D, or pass ``None`` to go back to inferring it."""
        if ndim is not None and int(ndim) not in (2, 3):
            raise ValueError(f"ndim must be 2, 3 or None, got {ndim!r}")
        self.ndim = None if ndim is None else int(ndim)

    # -- geometry ------------------------------------------------------

    def pixel_rect(self, fb_width: int, fb_height: int) -> Tuple[int, int, int, int]:
        """The panel's ``glViewport`` rectangle ``(x, y, w, h)`` in framebuffer pixels.

        Bottom-left origin, matching both the fractional convention and OpenGL. Widths and
        heights are clamped to at least 1px so a degenerate rect never produces a zero-size
        viewport (which some drivers treat as an error).
        """
        x0, y0, w, h = self.rect_frac
        px = int(round(x0 * fb_width))
        py = int(round(y0 * fb_height))
        pw = max(1, int(round(w * fb_width)))
        ph = max(1, int(round(h * fb_height)))
        return px, py, pw, ph

    def contains_window_px(self, mx: float, my: float, width: int, height: int) -> bool:
        """True when window-space cursor ``(mx, my)`` (top-left origin, GLFW style) is inside.

        GLFW reports the cursor with the origin at the *top-left*, while the panel rectangle
        uses a bottom-left origin, so the y axis is flipped here before the test.
        """
        x0, y0, w, h = self.rect_frac
        fx = mx / max(width, 1)
        fy = 1.0 - (my / max(height, 1))  # flip to bottom-left origin
        return (x0 <= fx <= x0 + w) and (y0 <= fy <= y0 + h)

    def local_cursor(self, mx: float, my: float, width: int, height: int) -> Tuple[float, float]:
        """Map a window-space cursor to this panel's local pixel space.

        Returns ``(lx, ly)`` with a top-left origin relative to the panel, so it can be fed
        straight to ``screen_to_world`` / picking with the panel's own pixel width/height.
        """
        x0, y0, w, h = self.rect_frac
        panel_left_px = x0 * width
        # rect_frac uses a bottom-left origin; the panel's top edge in top-left window space
        # is at (1 - (y0 + h)) of the window height.
        panel_top_px = (1.0 - (y0 + h)) * height
        return mx - panel_left_px, my - panel_top_px

    def pixel_size(self, width: int, height: int) -> Tuple[int, int]:
        """This panel's ``(width, height)`` in the given (window or framebuffer) pixel space."""
        _, _, w, h = self.rect_frac
        return max(1, int(round(w * width))), max(1, int(round(h * height)))

    def window_offset_px(self, width: int, height: int) -> Tuple[float, float]:
        """Top-left-origin ``(ox, oy)`` of this panel in window pixels, for imgui overlays.

        imgui draws in window space with a top-left origin, so a tick label or legend computed
        in the panel's local pixels must be shifted by this to land over the panel. The rect is
        stored bottom-left origin, so the top edge is ``1 - (y0 + h)`` of the window height.
        """
        x0, y0, w, h = self.rect_frac
        return x0 * width, (1.0 - (y0 + h)) * height
