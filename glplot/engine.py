from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import glfw
import numpy as np
from OpenGL.GL import *

from .controllers import CameraController
from .core.camera3d import (
    SYSTEM_3D_ARTISTS,
    Axes3DOptions,
    Camera3D,
    View3DProxy,
    bounds_centre_radius,
)
from .core.context import RenderContext
from .core.layers import (
    BaseLayer,
    Layer3D,
    LineFamilyLayer,
    PatchLayer,
    PolylineLayer,
    ScatterLayer,
    TextLayer,
)
from .core.legacy import (
    CacheState,
    CameraState,
    FrameState,
    InteractionState,
    LineDataset,
    ScatterDataset,
    SceneData,
    StripDataset,
)
from .core.panel import Panel
from .managers.axis import AxisManager
from .managers.effects import EffectManager
from .managers.hud import HudManager
from .managers.picking import PickingManager
from .managers.renderer_manager import RendererManager
from .options import (
    DEFAULT_AXIS_MARGINS,
    BlendMode,
    EngineOptions,
    RenderMode,
    override_axis_margins,
)
from .policy import RenderPolicyManager
from .renderers import axes3d
from .renderers.density import DensityRenderer
from .renderers.exact import ExactLineRenderer
from .renderers.interaction import InteractionRenderer
from .renderers.legend import draw_legend
from .utils.export import ExportManager
from .utils.scale import forward as _scale_forward
from .utils.scale import inverse as _scale_inverse
from .utils.shaders import DENSITY_SCHEMES

#: How far past the camera window a screen-sampled layer is evaluated, as a multiple of the
#: half-extent. ``world_window`` reports the extent *before* axis margins, and the MVP then
#: insets it, so a little world outside that window is still on screen at the frame edges.
#: 1.15 comfortably covers the default margins at any sane figure size.
_VIEW_SAMPLE_PAD = 1.15


def _finite_log_bounds(vmin: float, vmax: float) -> Tuple[float, float]:
    """A ``(vmin, vmax)`` pair already in log10 space, with NaN edges patched.

    ``_scale_forward`` maps a non-positive raw bound to NaN (there is no log of zero or a
    negative number). If only one edge is non-positive -- data straddling zero -- fall back
    to one decade below the finite edge, matplotlib's own rule of thumb for a partially
    invalid log range. If *both* are non-positive (no positive data at all, so there is
    nothing sensible to fit), fall back to a fixed one-decade window so the camera never
    receives NaN, matching matplotlib's own "cannot be log-scaled" degenerate case.
    """
    vmin, vmax = float(vmin), float(vmax)
    if math.isnan(vmin) and math.isnan(vmax):
        return -1.0, 0.0
    if math.isnan(vmin):
        return vmax - 1.0, vmax
    if math.isnan(vmax):
        return vmin, vmin + 1.0
    return vmin, vmax


class GPULinePlot:
    def __init__(
        self,
        width: int = 1280,
        height: int = 800,
        title: str = "GLPlot",
        options: Optional[EngineOptions] = None,
    ) -> None:
        self.options = options or EngineOptions(
            window_width=width, window_height=height, title=title
        )
        self.policy = RenderPolicyManager(self.options)

        # Multi-panel: the engine draws into one or more Panels, each carrying its own
        # scene/camera/interaction/cache and a rectangle inside the window. The default is a
        # single full-window panel, which reproduces the historical single-viewport engine
        # exactly. ``scene`` / ``camera`` / ``camera_controller`` / ``interaction`` / ``cache``
        # are properties (below) delegating to ``self.panels[self.active_panel_index]``, so
        # every pre-panel code path keeps operating on "the current axes" unchanged.
        self.panels: List[Panel] = [Panel(self.options, rect_frac=(0.0, 0.0, 1.0, 1.0))]
        self.active_panel_index: int = 0
        #: Window-pixel offset of the panel currently being drawn, for imgui overlays (tick
        #: labels, legend). Set by ``_draw_panels`` per panel; (0, 0) for a full-window panel.
        self._panel_offset_px: Tuple[float, float] = (0.0, 0.0)
        #: Framebuffer-pixel viewport (x, y, w, h) of the panel currently being drawn, so the
        #: density resolve lands in the right sub-rect. ``None`` = full framebuffer.
        self._panel_fb_rect: Optional[Tuple[int, int, int, int]] = None
        #: Tight axis gutters (l, r, b, t) applied to every panel while the window is split,
        #: so the frames pack closely instead of each reserving the rail-widened left gutter.
        #: ``None`` uses the live per-window gutters (full labels). Compact by default.
        from .options import PANEL_COMPACT_MARGINS

        self._panel_margins: Optional[Tuple[float, float, float, float]] = PANEL_COMPACT_MARGINS
        self.frame = FrameState()

        self.window = None
        self.width = self.options.window_width
        self.height = self.options.window_height
        self.fb_width = self.options.window_width
        self.fb_height = self.options.window_height
        self.title = title

        self.exact_renderer = ExactLineRenderer(self.options)
        self.interaction_renderer = InteractionRenderer(self)
        self.hud = HudManager(self)
        self.picking = PickingManager(self.options)
        self.export = ExportManager(self)
        self.renderer_manager = RendererManager(self)
        self.axis_manager = AxisManager(self)

        self._cpu_line_copy: Optional[np.ndarray] = None
        self._is_test_mode: bool = False
        self._needs_initial_autoscale: bool = True
        self.display_density: bool = False
        self.density_renderer = DensityRenderer(self)
        self.global_alpha = float(self.options.default_global_alpha)
        self.enable_subsample = bool(self.options.lod_enabled)
        self.max_lines_per_px = int(self.options.default_line_budget_per_px)
        # ``view3d`` is no longer a dict of five floats: the real state lives on the active
        # panel as a ``Camera3D`` plus an ``Axes3DOptions``, and this proxy presents it
        # through the same mapping interface every existing caller (the HUD, the preview
        # bridge, the Style panel, user scripts poking ``fig.view3d["azim"]``) already uses.
        # See ``core/camera3d.View3DProxy``.
        self._view3d_proxy = View3DProxy(self.active_panel.camera3d, self.active_panel.axes3d)
        #: World-space anchors for the 3D tick numbers and axis titles, refreshed by
        #: :meth:`ensure_3d_axes`. Text is not a GL primitive here — the HUD projects these
        #: and draws them into an imgui draw list, the same way the 2D tick labels work.
        #: Per-frame callbacks, run at the top of the main loop. See
        #: :meth:`add_frame_callback` — the hook that makes live animation possible.
        self._frame_callbacks: List[Callable[[float], None]] = []
        self._axes3d_labels: List[axes3d.LabelAnchor] = []
        #: The padded box those anchors were computed against, so the HUD projects them
        #: with exactly the matrix the geometry was drawn with.
        self._axes3d_bounds: Optional[Tuple[float, float, float, float, float, float]] = None
        #: The *resolved* camera the anchors were placed with — ``camera3d`` with the
        #: auto-fit distance and pan filled in, which is what the renderer draws through.
        #: ``camera3d`` itself stays in auto mode (``distance is None``) and projects to a
        #: visibly different place. See :meth:`ensure_3d_axes`.
        self._axes3d_camera: Optional[Camera3D] = None
        #: The ``((values, labels), ...)`` triple last drawn, x/y/z. The tick count adapts
        #: to the camera, so this is the only record of what is actually on the axes.
        self._axes3d_ticks: Optional[Tuple[Tuple[np.ndarray, List[str]], ...]] = None

        self.picked_info: Optional[dict] = None
        self.mouse_world: Optional[Tuple[float, float]] = None
        self._last_perf_t = time.perf_counter()

        self.effects = EffectManager(self)
        self._shim_cache: Dict[str, BaseLayer] = {}
        #: Last ``light_bg_mode`` value :meth:`_apply_background_mode` acted on. None means
        #: "never applied", so the constructor's call below always seeds the stock colours.
        self._background_mode_applied: Optional[bool] = None
        self._apply_background_mode()

    # --------------------------------------------------------
    # Panel delegation
    # --------------------------------------------------------
    # The active panel owns the scene/camera/interaction/cache the rest of the code
    # historically read straight off the engine. These properties forward to it so a single
    # full-window panel is indistinguishable from the pre-panel engine, and switching the
    # active panel re-points "the current axes" at another region.

    @property
    def active_panel(self) -> Panel:
        idx = self.active_panel_index
        if idx < 0 or idx >= len(self.panels):
            idx = 0
            self.active_panel_index = 0
        return self.panels[idx]

    @property
    def scene(self) -> SceneData:
        return self.active_panel.scene

    @scene.setter
    def scene(self, value: SceneData) -> None:
        self.active_panel.scene = value

    @property
    def camera(self) -> CameraState:
        return self.active_panel.camera

    @camera.setter
    def camera(self, value: CameraState) -> None:
        self.active_panel.camera = value

    @property
    def camera_controller(self) -> CameraController:
        return self.active_panel.camera_controller

    @camera_controller.setter
    def camera_controller(self, value: CameraController) -> None:
        self.active_panel.camera_controller = value

    @property
    def interaction(self) -> InteractionState:
        return self.active_panel.interaction

    @interaction.setter
    def interaction(self, value: InteractionState) -> None:
        self.active_panel.interaction = value

    @property
    def cache(self) -> CacheState:
        return self.active_panel.cache

    @cache.setter
    def cache(self, value: CacheState) -> None:
        self.active_panel.cache = value

    @property
    def camera3d(self) -> Camera3D:
        """The active panel's 3D point of view."""
        return self.active_panel.camera3d

    @camera3d.setter
    def camera3d(self, value: Camera3D) -> None:
        self.active_panel.camera3d = value

    @property
    def axes3d(self) -> Axes3DOptions:
        """The active panel's 3D axis decoration (box, floor, grid, ticks, labels)."""
        return self.active_panel.axes3d

    @axes3d.setter
    def axes3d(self, value: Axes3DOptions) -> None:
        self.active_panel.axes3d = value

    @property
    def view3d(self) -> View3DProxy:
        """The 3D view as a plain mapping, backed by the active panel's camera.

        Re-pointed at the active panel on every access, so ``fig.view3d["azim"] = 30``
        moves *the current axes*' camera the way ``fig.camera`` addresses the current 2D
        one. Kept as a mapping purely for compatibility (see :class:`View3DProxy`); new
        code should reach for :attr:`camera3d` and :attr:`axes3d`.
        """
        panel = self.active_panel
        self._view3d_proxy._rebind(panel.camera3d, panel.axes3d)
        return self._view3d_proxy

    @view3d.setter
    def view3d(self, value: Any) -> None:
        """Assigning a mapping updates the camera key by key rather than replacing it."""
        proxy = self.view3d
        for key, item in dict(value).items():
            proxy[key] = item

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    @property
    def show_density(self) -> bool:
        return self.display_density

    @show_density.setter
    def show_density(self, enabled: bool) -> None:
        self.set_density_enabled(enabled)

    @property
    def N(self) -> int:
        return self.scene.lines.count

    @property
    def _xrange(self) -> Tuple[float, float]:
        return self.scene.lines.x_range

    @property
    def _scatters(self) -> list:
        return [layer for layer in self.scene.layers if layer.layer_type == "scatter"]

    @property
    def _line_strips(self) -> list:
        return [layer for layer in self.scene.layers if layer.layer_type == "polyline"]

    @property
    def _text_annotations(self) -> list:
        return [layer for layer in self.scene.layers if layer.layer_type == "text"]

    def set_title(
        self,
        title: str,
        fontsize: Optional[float] = None,
        color: Optional[object] = None,
    ) -> None:
        """Set the window caption and, with it, the on-plot title's styling.

        ``fontsize``/``color`` are assigned unconditionally, so omitting one restores the
        default rather than inheriting whatever the previous call left behind. That is
        matplotlib's rule -- an unspecified property resolves to the rcParam, it does not
        persist from an earlier ``title()`` -- and the alternative makes styling depend on
        call history, which is untraceable in a script that titles a figure per frame.
        """
        self.title = str(title)
        self.options.title = str(title)
        self.options.axis_title_fontsize = fontsize
        self.options.axis_title_color = color
        self.frame.dirty_ui = True

    def set_global_alpha(self, alpha: float) -> None:
        self.global_alpha = float(alpha)
        self.options.default_global_alpha = float(alpha)
        self.frame.dirty_scene = True

    def set_lod(self, enabled: bool = True, max_lines_per_px: int = 8) -> None:
        self.enable_subsample = bool(enabled)
        self.max_lines_per_px = max(1, int(max_lines_per_px))
        self.options.lod_enabled = bool(enabled)
        self.options.default_line_budget_per_px = float(self.max_lines_per_px)
        self.frame.dirty_scene = True

    def is_3d_scene(self) -> bool:
        """True when the active panel is 3D.

        An explicit ``projection="3d"`` (``panel.ndim``) wins; with none set the answer is
        the historical one — "does the scene hold a 3D layer" — so nothing that predates
        the mode flag changes behaviour.
        """
        return self.active_panel.is_3d()

    @property
    def ndim(self) -> int:
        """2 or 3: the active panel's dimensionality, resolved."""
        return 3 if self.is_3d_scene() else 2

    def set_ndim(self, ndim: Optional[int]) -> None:
        """Pin the active panel to 2D or 3D (``None`` = infer from the layers).

        Switching *to* 3D seeds the axis box so an empty 3D panel still reads as a 3D
        space rather than a blank window; switching to 2D strips the 3D system artists,
        which would otherwise linger as an empty wireframe over a 2D plot.
        """
        self.active_panel.set_ndim(ndim)
        if self.active_panel.is_3d():
            self.set_3d_view()
        else:
            self.scene.layers = [
                layer
                for layer in self.scene.layers
                if layer.metadata.get("artist") not in self._SYSTEM_3D_ARTISTS
            ]
        self.frame.dirty_scene = True
        self.frame.dirty_ui = True

    #: Alias of :data:`glplot.core.camera3d.SYSTEM_3D_ARTISTS`, kept as a class attribute
    #: because it has been part of the engine's surface since the 3D box existed.
    _SYSTEM_3D_ARTISTS = SYSTEM_3D_ARTISTS

    def _is_pure_3d_scene(self) -> bool:
        """True when nothing 2D should be drawn over this panel.

        An explicit ``projection="3d"`` answers True on its own, *including for an empty
        panel*. Without that clause a figure pinned to 3D but not yet given any data fell
        through to the 2D branch and drew a pair of screen-aligned spines with 2D tick
        labels — the one state where "is this 3D" has no layers to infer it from, and the
        state ``figure(projection="3d")`` puts you in before you plot anything.

        With no explicit mode the rule is the historical one: every data layer is 3D, and
        there is at least one. The 3D decoration artists are excluded because they are
        chrome the engine added, not data the user plotted.
        """
        if self.active_panel.ndim is not None:
            return int(self.active_panel.ndim) == 3
        data_layers = [
            layer
            for layer in self.scene.layers
            if layer.metadata.get("artist") not in self._SYSTEM_3D_ARTISTS
        ]
        return bool(data_layers) and all(
            getattr(layer, "layer_type", "").endswith("3d") for layer in data_layers
        )

    def get_3d_layers(self) -> list[Layer3D]:
        return [
            layer
            for layer in self.scene.layers
            if isinstance(layer, Layer3D)
            and layer.metadata.get("artist") not in self._SYSTEM_3D_ARTISTS
        ]

    def get_3d_bounds(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        bounds = [layer.get_bounds_3d() for layer in self.get_3d_layers()]
        bounds = [b for b in bounds if b is not None and all(np.isfinite(b))]
        if not bounds:
            return None
        arr = np.asarray(bounds, dtype=np.float32)
        return (
            float(np.min(arr[:, 0])),
            float(np.max(arr[:, 1])),
            float(np.min(arr[:, 2])),
            float(np.max(arr[:, 3])),
            float(np.min(arr[:, 4])),
            float(np.max(arr[:, 5])),
        )

    def padded_3d_bounds(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """The box the axis wireframe, floor, grid and ticks are drawn on.

        The data bounds grown by :attr:`Axes3DOptions.pad`, **except** on an axis the user
        gave an explicit limit: that range is used verbatim. Padding an explicit limit
        would put the wall somewhere other than the number the user typed, and the tick at
        that number would then sit inside the box rather than on its edge.

        One definition, used by the decoration, the framing and the clip box alike.
        """
        bounds = self.get_3d_bounds()
        axes = self.axes3d
        if bounds is None:
            # With no data there is still a box worth drawing if the user pinned limits.
            if not axes.has_limits():
                return None
            bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        pad = float(axes.pad)
        out = []
        for index, limit in enumerate(axes.limits()):
            lo, hi = float(bounds[index * 2]), float(bounds[index * 2 + 1])
            if limit is not None:
                out.extend((float(limit[0]), float(limit[1])))
                continue
            span = max(hi - lo, 1e-6)
            out.extend((lo - pad * span, hi + pad * span))
        return tuple(out)  # type: ignore[return-value]

    def clip_3d_bounds(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """The world box outside which 3D geometry is discarded, or None for no clipping.

        Only the axes with an explicit limit constrain anything; the rest get an inverted
        interval, which the shader reads as "unbounded". Returns None when no axis has a
        limit, so the default path sets the disable sentinel and never compares.
        """
        axes = self.axes3d
        if not axes.has_limits():
            return None
        out = []
        for limit in axes.limits():
            if limit is None:
                out.extend((1.0, 0.0))  # inverted == unbounded, matching the shader
            else:
                out.extend((float(limit[0]), float(limit[1])))
        return tuple(out)  # type: ignore[return-value]

    def set_3d_limits(
        self,
        *,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        zlim: Optional[Tuple[float, float]] = None,
        reset: bool = False,
    ) -> None:
        """Pin one or more 3D axis ranges. ``reset=True`` clears them all first.

        Geometry outside a pinned range is clipped, not merely left outside the box, so a
        limit is a genuine view into a slab of the data rather than a redrawn wall.
        """
        axes = self.axes3d
        if reset:
            axes.xlim = axes.ylim = axes.zlim = None
        for name, value in (("xlim", xlim), ("ylim", ylim), ("zlim", zlim)):
            if value is not None:
                lo, hi = float(value[0]), float(value[1])
                if not hi > lo:
                    raise ValueError(f"{name} needs hi > lo, got ({lo}, {hi})")
                setattr(axes, name, (lo, hi))
        self.set_3d_view()

    def _camera_metadata(
        self, bounds: Optional[Tuple[float, float, float, float, float, float]]
    ) -> dict:
        """The camera as the resolved dict every 3D layer carries in ``metadata["camera"]``.

        "Resolved" means ``distance`` is always a number: the renderer reads this dict with
        ``camera.get("distance", ...)``, so a present-but-None key would poison the matrix.
        An auto-fit camera (``distance is None``) therefore resolves against the scene here,
        which is also what makes the framing follow the data when a layer is replaced.
        """
        camera = self.camera3d.to_dict()
        if camera.get("distance") is not None or bounds is None:
            if camera.get("distance") is None:
                camera.pop("distance", None)
            return camera

        # Auto-framing. Fit the *projected box* to the viewport rather than the bounding
        # sphere: the old seed — three times the widest axis — is orientation-blind, so an
        # elongated dataset (which most real data is) was framed as though it were a ball
        # of its own diagonal and sat in the middle of a mostly empty canvas.
        #
        # The fit is written into the *metadata* and not onto the camera, so the camera
        # stays in auto mode: replacing a layer re-frames the plot, which is the behaviour
        # ``distance is None`` exists to express. Mutating the camera here would silently
        # pin it the first time anything drew.
        aspect = max(self.width, 1) / max(self.height, 1)
        fitted = self.camera3d.fit_distance(bounds, aspect)
        if fitted is None:  # pragma: no cover - bounds is not None here
            dx = max(bounds[1] - bounds[0], 1e-6)
            dy = max(bounds[3] - bounds[2], 1e-6)
            dz = max(bounds[5] - bounds[4], 1e-6)
            camera["distance"] = max(dx, dy, dz) * 3.0
            return camera
        camera["distance"] = fitted

        # Centre it too, but only while the user has not panned: a pan is an explicit
        # statement about where to look, and re-centring on top of it would undo the drag
        # on the next frame.
        if tuple(self.camera3d.pan) == (0.0, 0.0, 0.0):
            probe = self.camera3d.copy()
            probe.distance = fitted
            camera["pan"] = probe.centre_offset(bounds, aspect)
        return camera

    def sync_3d_camera(self) -> None:
        """Push the panel camera and the shared scene box onto every 3D layer.

        The renderer builds its matrix per layer from ``metadata``, so this is the step
        that makes one camera govern the whole scene instead of each layer drifting into
        its own view. Called by every 3D verb below, and by the GUI after it mutates the
        camera directly.
        """
        raw = self.get_3d_bounds()
        padded = self.padded_3d_bounds()
        camera = self._camera_metadata(padded or raw)
        clip = self.clip_3d_bounds()
        for layer in self.scene.layers:
            if isinstance(layer, Layer3D):
                merged = dict(layer.metadata.get("camera", {}))
                merged.update(camera)
                layer.metadata["camera"] = merged
                if padded is not None:
                    layer.metadata["scene_bounds"] = padded
                if clip is None:
                    layer.metadata.pop("clip_bounds", None)
                else:
                    layer.metadata["clip_bounds"] = clip
                layer.dirty.style_dirty = True

    def set_3d_view(
        self,
        *,
        elev: Optional[float] = None,
        azim: Optional[float] = None,
        roll: Optional[float] = None,
        fov: Optional[float] = None,
        distance: Optional[float] = None,
        show_axes: Optional[bool] = None,
        projection: Optional[str] = None,
        up_axis: Optional[str] = None,
        pan: Optional[Tuple[float, float, float]] = None,
        box_aspect: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Set any part of the 3D view and re-sync the scene.

        Every argument is optional and an omitted one is left untouched, so this doubles as
        "re-apply the current view" — which is what the interaction handlers and the GUI
        call after mutating :attr:`camera3d` directly.
        """
        camera = self.camera3d
        if elev is not None:
            camera.elev = float(elev)
        if azim is not None:
            camera.azim = float(azim)
        if roll is not None:
            camera.roll = float(roll)
        if fov is not None:
            camera.fov = float(fov)
        if distance is not None:
            camera.distance = float(distance)
        if projection is not None:
            camera.set_projection(projection)
        if up_axis is not None:
            camera.set_up_axis(up_axis)
        if pan is not None:
            camera.pan = (float(pan[0]), float(pan[1]), float(pan[2]))
        if box_aspect is not None:
            camera.box_aspect = (
                float(box_aspect[0]),
                float(box_aspect[1]),
                float(box_aspect[2]),
            )
        if show_axes is not None:
            self.axes3d.show_axes = bool(show_axes)

        self.sync_3d_camera()

        if self.axes3d.show_axes:
            self.ensure_3d_axes()
        else:
            self.scene.layers = [
                layer
                for layer in self.scene.layers
                if layer.metadata.get("artist") not in self._SYSTEM_3D_ARTISTS
            ]
        self.frame.dirty_scene = True
        self.cache.refresh_requested = True

    def set_3d_preset(self, name: str) -> None:
        """Snap the camera to a named orientation (see :data:`STANDARD_VIEWS`)."""
        self.camera3d.apply_view(name)
        self.set_3d_view()

    def reset_3d_view(self) -> None:
        """Back to the default isometric framing: no dolly, no pan, no roll."""
        self.camera3d.reset()
        for layer in self.get_3d_layers():
            cam = layer.metadata.get("camera", {})
            cam.pop("distance", None)
            layer.metadata["camera"] = cam
        self.set_3d_view()

    def frame_3d_view(self) -> None:
        """Re-fit the camera so the data fills the viewport, keeping the orientation."""
        aspect = max(self.width, 1) / max(self.height, 1)
        self.camera3d.frame_viewport(self.padded_3d_bounds(), aspect)
        self.set_3d_view()

    def orbit_3d(self, d_azim: float, d_elev: float) -> None:
        """Rotate the 3D camera around its target by a delta in degrees."""
        self.camera3d.orbit(d_azim, d_elev)
        self.set_3d_view()

    def pan_3d(self, dx_px: float, dy_px: float) -> None:
        """Slide the 3D camera's target by a screen-space drag, in pixels."""
        _, radius = bounds_centre_radius(self.camera3d.transform_bounds(self.padded_3d_bounds()))
        self.camera3d.pan_pixels(dx_px, dy_px, max(self.height, 1), radius)
        self.set_3d_view()

    def _advance_auto_spin(self, now: float) -> None:
        """Step every panel's turntable rotation to wall-clock time ``now``.

        Per panel, not just the active one: in a split figure a spinning 3D panel must
        keep spinning while the user works in another. Returns quietly when nothing is
        spinning, which is the normal case and must cost no more than the flag check.
        """
        spinning = [panel for panel in self.panels if panel.camera3d.auto_spin and panel.is_3d()]
        last = getattr(self, "_last_spin_time", None)
        self._last_spin_time = now
        if not spinning or last is None:
            return
        dt = float(now) - float(last)
        # A large gap means the loop was blocked (a modal, a slow export) or the clock
        # jumped; advancing by it would snap the view through a whole revolution.
        if not (0.0 < dt <= 0.25):
            return

        saved = self.active_panel_index
        try:
            for panel in spinning:
                panel.camera3d.advance_spin(dt)
                # ``set_3d_view`` works on the *active* panel, so each one is made current
                # for the length of its own re-sync.
                self.active_panel_index = self.panels.index(panel)
                self.set_3d_view()
        finally:
            self.active_panel_index = saved

    # --------------------------------------------------------
    # Animation
    # --------------------------------------------------------

    def _resample_view_layers(self, pixel_scale: float = 1.0) -> None:
        """Re-evaluate every screen-sampled layer whose visible interval has moved.

        A :class:`~glplot.core.layers.FunctionLayer` holds no fixed table of points: it
        samples its function across whatever x range is on screen, at roughly one sample
        per pixel column. So the *view* is an input to its geometry, and the geometry has
        to be rebuilt when the view changes — which is what this does, once per frame, at
        the top of the loop where mutating the scene is legal.

        Two properties make it cheap enough to run unconditionally. The count is tied to
        the viewport, not the zoom, so a 1e-12 zoom costs exactly what the full view costs.
        And ``needs_resample`` compares the interval against the one already sampled, so an
        idle figure does no work at all — the common case is a few float comparisons per
        layer per frame.

        Per panel, because each panel has its own camera and therefore its own visible
        interval; a layer in a panel nobody is looking at still keeps itself correct.

        ``pixel_scale`` multiplies the pixel width the sample count is derived from. The
        export path passes the ratio between the image it is about to write and the window,
        so a 4x ``savefig`` gets 4x the samples — otherwise a high-resolution export would
        magnify the *screen's* sampling and reintroduce exactly the faceting this layer
        exists to avoid. The count is part of ``needs_resample``, so the next on-screen
        frame samples back down by itself.
        """
        scale = max(float(pixel_scale), 1e-3)
        for panel in self.panels:
            candidates = [
                layer
                for layer in panel.scene.layers
                if callable(getattr(layer, "resample", None)) and layer.style.visible
            ]
            if not candidates:
                continue
            panel_w, height_px = panel.pixel_size(self.width, self.height)
            width_px = max(1, int(round(panel_w * scale)))
            # A little wider than the data area: ``world_window`` is the pre-margin extent,
            # and the MVP insets it, so the axes' left and right edges show world slightly
            # outside it. Without the pad the curve would stop short of the frame.
            x0, x1, _y0, _y1 = panel.camera_controller.world_window(
                panel_w, height_px, padding=_VIEW_SAMPLE_PAD
            )
            changed = False
            for layer in candidates:
                if layer.needs_resample(x0, x1, width_px) and layer.resample(x0, x1, width_px):
                    changed = True
            if changed:
                panel.cache.refresh_requested = True
                self.frame.dirty_scene = True

    def add_frame_callback(self, fn: Callable[[float], None]) -> Callable[[float], None]:
        """Register ``fn(t)`` to run once per frame inside the main loop. Returns ``fn``.

        The hook the engine was missing. ``run()`` owns its loop and exposed no way in, so
        anything that wanted to drive the scene over time — a ``FuncAnimation``, a
        simulation stepping itself, a recorder — could render frames offline but could not
        make an open window move. That is why ``glplot.animation`` can ``save()`` a movie
        but could not play one.

        ``fn`` receives the loop's wall-clock time in seconds. It runs on the GL thread at
        the top of the frame, in the same slot the command queue drains, so it may mutate
        the scene directly — this is *not* a draw callback and CONTRACT §1.1 does not apply
        to it.

        An exception in a callback removes that callback and is logged, rather than taking
        down the loop: a broken user animation must not make the window unclosable.
        """
        self._frame_callbacks.append(fn)
        return fn

    def remove_frame_callback(self, fn: Callable[[float], None]) -> bool:
        """Unregister a callback added by :meth:`add_frame_callback`."""
        for i, existing in enumerate(self._frame_callbacks):
            if existing is fn:
                del self._frame_callbacks[i]
                return True
        return False

    def _run_frame_callbacks(self, now: float) -> None:
        """Run every per-frame callback. Never raises out of the loop."""
        if not self._frame_callbacks:
            return
        for fn in list(self._frame_callbacks):
            try:
                fn(float(now))
            except Exception:  # pragma: no cover - defensive; a user callback is arbitrary
                logging.getLogger(__name__).exception(
                    "A frame callback raised and has been removed."
                )
                self.remove_frame_callback(fn)
        self.frame.dirty_scene = True

    def _advance_timelines(self, now: float) -> None:
        """Step every panel's timeline to wall-clock ``now`` and apply what it evaluates.

        Per panel, not per figure, for the same reason the cameras are: a split figure must
        be able to animate one region while another holds still.

        Wall-clock, not per frame, so an animation authored at 5 seconds lasts 5 seconds
        whether the window is managing 12fps or 144 — and so a dropped frame skips content
        rather than stretching time. ``Timeline.advance`` clamps an implausible delta
        itself, which is what keeps a blocked loop from jumping half the animation.

        Returns immediately when nothing is playing, which is the normal case and must cost
        no more than the flag check.
        """
        playing = [panel for panel in self.panels if panel.timeline.playing]
        last = getattr(self, "_last_timeline_time", None)
        self._last_timeline_time = now
        if not playing or last is None:
            return
        dt = float(now) - float(last)

        for panel in playing:
            if not panel.timeline.advance(dt):
                continue
            values = panel.timeline.evaluate()
            if values:
                self._apply_timeline_values(panel, values)
        self.frame.dirty_scene = True
        self.cache.refresh_requested = True

    def _apply_timeline_values(self, panel: Panel, values: Dict[Any, Any]) -> None:
        """Apply one evaluated frame to ``panel``.

        Delegates to :mod:`glplot.anim.applier`, imported lazily so the engine does not
        depend on the animation package at import time — a figure that never animates must
        not pay for it, and the package must stay optional to the core.
        """
        try:
            from .anim.applier import apply_values
        except Exception:  # pragma: no cover - the package may not be installed
            return
        saved = self.active_panel_index
        try:
            self.active_panel_index = self.panels.index(panel)
            apply_values(self, values)
        except Exception:  # pragma: no cover - defensive
            logging.getLogger(__name__).exception("Applying an animation frame failed.")
        finally:
            self.active_panel_index = saved

    def _zoom_3d(self, factor: float) -> None:
        """Dolly the 3D camera. ``factor > 1`` pulls back, ``< 1`` moves in."""
        if self.camera3d.distance is None:
            # The historical seed for a never-dollied camera: 3.2x the widest half-extent
            # of the *raw* bounds. Kept literally so the first wheel click feels the same.
            bounds = self.get_3d_bounds()
            if bounds is None:
                return
            half = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) / 2.0
            self.camera3d.distance = half * 3.2
        self.camera3d.dolly(factor)
        self.set_3d_view()

    #: Colour of each 3D system artist, as ``(dark_bg, light_bg)`` RGBA pairs. Two palettes
    #: because a 0.5-alpha near-white box is invisible on a white background — the 3D axes
    #: had no light-mode variant at all, which made ``light=True`` unusable for 3D.
    #: Each 3D system artist's ink as ``(on_dark, on_light)`` RGBA pairs.
    #:
    #: The weights are the whole visual design of a 3D plot, so they are worth stating:
    #: the panes are a barely-there wash that gives the eye a surface without competing
    #: with the data, the grid sits just above them, the box frames it, and the tick marks
    #: are the most opaque because they are what a number is read against.
    _AXIS3D_COLORS = {
        "axis3d": ((0.78, 0.84, 0.96, 0.68), (0.24, 0.28, 0.36, 0.72)),
        "floor3d": ((0.55, 0.62, 0.78, 0.13), (0.42, 0.48, 0.60, 0.10)),
        "grid3d": ((0.66, 0.74, 0.90, 0.26), (0.36, 0.42, 0.54, 0.24)),
        "ticks3d": ((0.88, 0.92, 1.00, 0.90), (0.14, 0.18, 0.26, 0.88)),
    }

    def _axis3d_color(self, artist: str) -> np.ndarray:
        """The RGBA for a 3D system artist, chosen against the background actually drawn.

        Derived from the background's **luminance**, not from the ``light_bg_mode`` flag.
        Those two disagree in a case that matters: ``ExportManager.savefig`` clears to
        white regardless of the mode, so a figure exported in the default dark mode used
        to paint a near-white box onto a white page and the whole 3D decoration vanished
        from the PNG. Reading the colour the renderer is about to clear with cannot drift
        from it.

        Same 0.5 luminance threshold and the same reasoning as
        :meth:`glplot.renderers.axis.AxisRenderer._draw_labels`, so the 2D tick labels and
        the 3D decoration flip together instead of at different moments.
        """
        dark_ink, light_ink = self._AXIS3D_COLORS[artist]
        background = (0.0, 0.0, 0.0)
        visual = getattr(self.options, "visual", None)
        if visual is not None:
            background = tuple(getattr(visual, "background_color", (0.0, 0.0, 0.0)))[:3]
        luminance = 0.299 * background[0] + 0.587 * background[1] + 0.114 * background[2]
        chosen = light_ink if luminance > 0.5 else dark_ink
        return np.asarray(chosen, dtype=np.float32)

    def _system_3d_layer(
        self,
        artist: str,
        vertices: np.ndarray,
        primitive: str,
        *,
        label: str,
        line_width: float = 1.0,
        on_top: bool = False,
    ) -> Optional[Layer3D]:
        """Create or update one of the 3D decoration layers, and return it.

        Reuses the existing layer when there is one so the GPU buffers survive a camera
        move; only the vertex array is re-uploaded. An empty ``vertices`` removes the layer
        outright rather than leaving a zero-length draw call in the scene, which is what
        makes the per-artist toggles in the View panel actually free.
        """
        existing = next(
            (layer for layer in self.scene.layers if layer.metadata.get("artist") == artist),
            None,
        )
        if vertices is None or len(vertices) == 0:
            if existing is not None:
                self.scene.layers = [layer for layer in self.scene.layers if layer is not existing]
            return None

        colors = np.tile(self._axis3d_color(artist), (len(vertices), 1))
        if existing is None:
            existing = Layer3D(
                vertices=vertices,
                colors=colors,
                primitive=primitive,
                label=label,
                layer_type="wireframe3d",
            )
            existing.metadata["artist"] = artist
            if on_top:
                self.scene.layers.append(existing)
            else:
                self.scene.layers.insert(0, existing)
        else:
            existing.vertices = vertices
            existing.colors = colors
            existing.dirty.gpu_dirty = True
            if on_top:
                # Re-append so it stays last in draw order even after new data arrived.
                self.scene.layers = [layer for layer in self.scene.layers if layer is not existing]
                self.scene.layers.append(existing)
        existing.style.line_width = float(line_width)
        return existing

    def ensure_3d_axes(self) -> Optional[Layer3D]:
        """Build (or refresh) the 3D axis decoration and return the bounding-box layer.

        Emits up to four system layers — floor, grid walls, tick marks and the box — each
        gated by its own :class:`~glplot.core.camera3d.Axes3DOptions` flag, plus the
        world-space label anchors the HUD projects into text. All of them are rebuilt
        rather than transformed when the camera moves, because which walls and which edges
        they belong on is itself a function of the camera (see
        :mod:`glplot.renderers.axes3d`).
        """
        bounds = self.padded_3d_bounds()
        if bounds is None:
            self._axes3d_labels = []
            self._axes3d_camera = None
            return None

        opts = self.axes3d
        camera_meta = self._camera_metadata(self.get_3d_bounds())
        aspect = max(self.width, 1) / max(self.height, 1)

        # **The camera the scene is actually drawn with**, which is not ``self.camera3d``.
        #
        # An auto-framed camera keeps ``distance = None`` on purpose (see
        # ``_camera_metadata``): the fitted distance and pan are written into each layer's
        # metadata instead, and the renderer builds its matrix from *those*. Anything here
        # that projects — choosing tick edges, measuring an axis in pixels, placing the
        # numbers — has to project through the same matrix or it is answering questions
        # about a camera nobody is looking through. It was not: with the stock auto framing
        # the two matrices put the same box corner ~90 px apart, which is why the numbers
        # floated away from the tick marks they belong to.
        camera = Camera3D.from_dict(camera_meta)

        # Which box edge each axis' ticks live on, solved once. The marks and the numbers
        # must agree on it — a mark on one edge and its label on another is not a rounding
        # difference, it is an unreadable axis — and solving it twice per frame projected
        # twelve probes through a freshly built MVP for the second time to get the same
        # answer. Passed to every consumer below.
        edges = axes3d.tick_edges(bounds, camera, aspect)

        # Tick *density* from the projected length of each axis, the way the 2D renderer
        # takes it from the viewport width (``managers/axis.update``: one tick per 160 px).
        # A fixed count per axis is what made the 3D ticks feel wrong under a dolly: the
        # values never changed, so zooming in spread five numbers across the whole screen
        # and zooming out packed them into a corner. ``opts.tick_count > 0`` still pins it.
        targets = axes3d.adaptive_tick_counts(
            bounds,
            camera,
            aspect,
            float(self.width),
            float(self.height),
            edges=edges,
            forced=int(opts.tick_count),
        )
        ticks = axes3d.ticks_for_bounds(bounds, targets)
        # Kept so ``pyplot.zticks()`` can report what is on screen rather than re-deriving
        # it: with an adaptive count there is no longer a number a caller could re-derive
        # it *from*, only the camera that produced it.
        self._axes3d_ticks = ticks

        # --- Back panes (drawn first, behind all data) ---
        #
        # Three shaded walls rather than the single ``z = zmin`` floor this used to draw.
        # It is the largest single step from "wireframe box" to "finished 3D plot": grid
        # lines floating in empty space give the eye nothing to judge depth against, and
        # the old lone floor disappeared entirely whenever the camera dropped below the
        # data. ``wall_panes`` picks the three walls facing away from the eye, so the data
        # is always seen *against* them.
        self._system_3d_layer(
            "floor3d",
            (
                axes3d.wall_panes(bounds, camera)
                if opts.show_floor
                else np.zeros((0, 3), dtype=np.float32)
            ),
            "triangles",
            label="3D panes",
        )

        # --- Grid walls (behind the data, in front of the floor) ---
        self._system_3d_layer(
            "grid3d",
            (
                axes3d.grid_lines(bounds, ticks, camera)
                if opts.show_grid
                else np.zeros((0, 3), dtype=np.float32)
            ),
            "lines",
            label="3D grid",
            line_width=1.0,
        )

        # --- Tick marks (on top, so they are never buried by the data) ---
        self._system_3d_layer(
            "ticks3d",
            (
                axes3d.tick_marks(bounds, ticks, camera, aspect, edges=edges)
                if opts.show_ticks
                else np.zeros((0, 3), dtype=np.float32)
            ),
            "lines",
            label="3D ticks",
            line_width=1.6,
            on_top=True,
        )

        # --- Bounding-box wireframe (drawn last, always on top) ---
        box = self._system_3d_layer(
            "axis3d",
            axes3d.box_edges(bounds) if opts.show_box else np.zeros((0, 3), dtype=np.float32),
            "lines",
            label="3D axes",
            line_width=2.2,
            on_top=True,
        )

        # --- Label anchors, for the HUD to project and draw as text ---
        self._axes3d_labels = axes3d.label_anchors(
            bounds,
            ticks,
            camera,
            aspect,
            tick_labels=bool(opts.show_tick_labels),
            axis_labels=(
                opts.xlabel if opts.show_axis_labels else "",
                opts.ylabel if opts.show_axis_labels else "",
                opts.zlabel if opts.show_axis_labels else "",
            ),
            edges=edges,
        )
        self._axes3d_bounds = bounds
        # Handed to ``axes3d.draw_labels``, which runs later in the frame and must project
        # the anchors through the matrix they were placed with, not through ``camera3d``.
        self._axes3d_camera = camera

        # Sync every 3D layer (data and decoration alike) onto the one camera and box.
        for layer in self.scene.layers:
            if isinstance(layer, Layer3D):
                layer.metadata["scene_bounds"] = bounds
                layer_camera = dict(layer.metadata.get("camera", {}))
                layer_camera.update(camera_meta)
                layer.metadata["camera"] = layer_camera
                layer.dirty.style_dirty = True

        return box

    def set_lines_ab(
        self,
        ab: np.ndarray,
        x_range=(-3.0, 3.0),
        colors: Optional[np.ndarray] = None,
        label: Optional[str] = None,
    ) -> None:
        ab = np.ascontiguousarray(ab, np.float32)
        cols = None if colors is None else np.ascontiguousarray(colors, np.float32)
        x_range = (float(x_range[0]), float(x_range[1]))

        # --- Legacy LineDataset (kept for exact_renderer compatibility) ---
        self.scene.lines = LineDataset(ab=ab, colors=cols, x_range=x_range)
        self.scene.lines.validate()
        self._cpu_line_copy = ab

        # --- Layer registration: make the line family visible in the HUD ---
        # Reuse the existing layer if one was already created by a previous call.
        existing = getattr(self, "_primary_line_layer", None)
        if existing is None:
            layer_label = label or "Lines"
            existing = LineFamilyLayer(ab=ab, colors=cols, x_range=x_range, label=layer_label)
            self._primary_line_layer = existing
            self.scene.layers.insert(0, existing)  # lines always render first
        else:
            # Update data in-place so the GPU buffers are refreshed next frame
            existing.ab = ab
            existing.colors = cols
            existing.x_range = x_range
            existing.dirty.gpu_dirty = True
            if label:
                existing.label = label

        self.frame.dirty_scene = True
        self.frame.dirty_pick = True
        if self.exact_renderer.buffers.vao:
            self.exact_renderer.upload(self.scene.lines)

    def add_text(
        self,
        x: float,
        y: float,
        text: str,
        fontsize: int = 12,
        color: Optional[Any] = None,
        label: Optional[str] = None,
    ) -> None:
        layer_label = label or f"Text: {text[:10]}"
        layer = TextLayer(x=x, y=y, text=text, label=layer_label)
        layer.style.text_size_px = fontsize
        if color is not None:
            layer.style.color = color
        self.scene.layers.append(layer)
        self.scene.texts.append({"x": x, "y": y, "str": text, "fontsize": fontsize, "color": color})
        self.frame.dirty_ui = True

    def add_scatter(
        self,
        x: np.ndarray,
        y: np.ndarray,
        colors: np.ndarray,
        size: float = 6.0,
        label: Optional[str] = None,
        sizes: Optional[np.ndarray] = None,
    ) -> None:
        pts = np.column_stack([x, y]).astype(np.float32)
        cols = np.ascontiguousarray(colors, np.float32)
        layer_label = label or f"Scatter {len(self.scene.layers)}"
        # ``sizes`` (optional, shape (N,)) makes marker size a per-point, data-driven
        # dimension; ``size`` remains the scalar fallback used where ``sizes`` is None.
        size_arr = None if sizes is None else np.asarray(sizes, dtype=np.float32).ravel()
        layer = ScatterLayer(pts=pts, colors=cols, size=size, label=layer_label, sizes=size_arr)
        if not label:
            # This name exists for the Scene panel's layer list, not because anyone asked
            # for a legend entry -- flagged so the headless preview's auto-legend (which
            # otherwise cannot tell "Scatter 7" apart from a real `label=`) can skip it.
            layer.metadata["auto_label"] = True
        self.scene.layers.append(layer)
        self.scene.scatters.append(ScatterDataset(pts=pts, colors=cols, size=size))
        self.frame.dirty_scene = True

    def add_geometry3d(
        self,
        vertices: np.ndarray,
        colors: Optional[np.ndarray] = None,
        *,
        indices: Optional[np.ndarray] = None,
        primitive: str = "points",
        layer_type: str = "scatter3d",
        label: Optional[str] = None,
        point_size: float = 3.0,
        line_width: float = 1.0,
        sizes: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Layer3D:
        """Append a 3D geometry layer and return it — the ``add_*`` verb 3D was missing.

        ``add_scatter``/``add_line_strip``/``add_patch`` have always been the engine's
        public way to put geometry in a scene, but there was no 3D counterpart:
        ``pyplot._add_3d_layer`` constructed the :class:`Layer3D` itself and appended it
        straight to ``scene.layers``. That left the GUI with nothing to call — CONTRACT
        §1.5 routes every queued command through the public API — which is a large part of
        why the Data panel could not build a 3D layer at all.

        The new layer is synced onto the panel's camera before it is returned, so it is
        framed by the same view as everything else in the scene rather than carrying the
        stale default until the next ``set_3d_view``.
        """
        verts = np.ascontiguousarray(vertices, dtype=np.float32)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError(f"3D vertices must have shape (N, 3), got {verts.shape}")
        if colors is None:
            cols = None
        else:
            cols = np.ascontiguousarray(colors, dtype=np.float32)
            if cols.shape != (len(verts), 4):
                raise ValueError(f"3D colors must have shape ({len(verts)}, 4), got {cols.shape}")
        layer = Layer3D(
            vertices=verts,
            colors=cols,
            indices=None if indices is None else np.ascontiguousarray(indices, dtype=np.uint32),
            primitive=primitive,
            label=label or f"3D {len(self.scene.layers)}",
            layer_type=layer_type,
            sizes=None if sizes is None else np.asarray(sizes, dtype=np.float32).ravel(),
        )
        layer.style.point_size = float(point_size)
        layer.style.line_width = float(line_width)
        layer.metadata.update(metadata or {})
        self.scene.layers.append(layer)
        # Frame it with the rest of the scene now rather than at the next camera change:
        # a layer added mid-session would otherwise draw under the default isometric view
        # while everything around it used the user's.
        self.set_3d_view()
        self.frame.dirty_scene = True
        return layer

    def add_line_strip(
        self,
        x: np.ndarray,
        y: np.ndarray,
        color: Tuple[float, float, float, float] = (0, 0, 0, 1),
        width: float = 1.0,
        label: Optional[str] = None,
        colors: Optional[np.ndarray] = None,
    ) -> None:
        pts = np.column_stack([x, y]).astype(np.float32)
        layer_label = label or f"Polyline {len(self.scene.layers)}"
        # ``colors`` (shape (N, 4)) colours the line by a data variable; ``color`` stays the
        # flat fallback used when it is None.
        col_arr = None if colors is None else np.ascontiguousarray(colors, dtype=np.float32)
        layer = PolylineLayer(pts=pts, color=color, width=width, label=layer_label, colors=col_arr)
        if not label:
            layer.metadata["auto_label"] = True
        self.scene.layers.append(layer)
        self.scene.strips.append(StripDataset(pts=pts, color=color))
        self.frame.dirty_scene = True

    def add_patch(
        self,
        vertices: np.ndarray,
        indices: Optional[np.ndarray] = None,
        mode: str = "strip",
        face_color: Optional[Tuple] = None,
        edge_color: Optional[Tuple] = None,
        label: Optional[str] = None,
        colors: Optional[np.ndarray] = None,
    ) -> None:
        layer_label = label or f"Patch {len(self.scene.layers)}"
        layer = PatchLayer(
            vertices=vertices, indices=indices, mode=mode, label=layer_label, colors=colors
        )
        if not label:
            layer.metadata["auto_label"] = True
        if face_color is not None:
            layer.style.face_color = face_color
        if edge_color is not None:
            layer.style.edge_color = edge_color
        self.scene.layers.append(layer)
        self.frame.dirty_scene = True

    def set_density_enabled(self, enabled: bool) -> None:
        self.display_density = bool(enabled)
        self.frame.dirty_scene = True
        self.cache.refresh_requested = True

    def _density_active(self) -> bool:
        """Whether the density accumulator should actually run this frame.

        Density is a **2D** technique: it accumulates overplotted line/scatter primitives
        into a screen-space heatmap. It has no meaning for 3D geometry — the density
        renderer does not even look at ``Layer3D`` vertices — so taking the density branch
        in a 3D scene replaced the whole draw with a pass that drew *nothing*, and the plot
        vanished. The 3D equivalent of "show me where the points pile up" is a
        ``volume3d`` layer, whose translucent alpha stacking is exactly a density cloud.

        So the flag can be on, but it only *acts* when there is a 2D scene to act on. The
        mixed case (a 3D scene that also holds 2D layers) also draws exact, because the 2D
        density pass would drop the 3D geometry.
        """
        return self.display_density and not self.is_3d_scene()

    def density_tint_active(self) -> bool:
        """Whether this frame's density pass paints in the layers' own colours.

        Asked by the background pass as well as by the density pass itself, because the two
        answers have to agree: the heatmap fills the window with the colormap's dark end and
        then replaces it, while a tinted resolve is *composited*, so it needs the plot's
        ordinary background under it -- the same one exact mode draws on. Getting this wrong
        does not misplace a pixel, it puts a red cloud on a slab of near-black.
        """
        if not self._density_active():
            return False
        from .managers.renderer_manager import LayerCapability

        layers = self.renderer_manager.filter_layers(
            self._get_all_layers(), LayerCapability.DENSITY
        )
        return self.renderer_manager.density_tint_active(layers)

    def set_density_gain(self, value: float) -> None:
        self.options.density_gain = float(value)
        self.frame.dirty_scene = True
        self.cache.refresh_requested = True

    def increase_density_gain(self) -> None:
        self.options.density_gain *= self.options.density_gain_step
        self.frame.dirty_scene = True
        self.cache.refresh_requested = True

    def decrease_density_gain(self) -> None:
        self.options.density_gain /= self.options.density_gain_step
        self.frame.dirty_scene = True
        self.cache.refresh_requested = True

    def next_density_scheme(self) -> None:
        self.options.density_scheme_index = (self.options.density_scheme_index + 1) % len(
            DENSITY_SCHEMES
        )
        self.frame.dirty_scene = True
        self.cache.refresh_requested = True

    def previous_density_scheme(self) -> None:
        self.options.density_scheme_index = (self.options.density_scheme_index - 1) % len(
            DENSITY_SCHEMES
        )
        self.frame.dirty_scene = True
        self.cache.refresh_requested = True

    def toggle_density(self) -> None:
        self.set_density_enabled(not self.display_density)

    def rebuild_density_renderer(self) -> None:
        """Trigger a resource reconstruction for the density engine when scale changes."""
        self.density_renderer.rebuild_target(self.fb_width, self.fb_height)
        self.frame.dirty_scene = True

    def set_view(
        self, xlim: Optional[Tuple[float, float]] = None, ylim: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Sets the world-space view limits, mimicking Matplotlib's xlim/ylim.
        Allows independent scaling of X and Y axes (unforcing 1:1 data aspect).
        """
        if xlim is None and ylim is None:
            return

        tx = xlim if xlim is not None else self.get_xlim()
        ty = ylim if ylim is not None else self.get_ylim()

        self.camera_controller.fit_bounds(tx[0], tx[1], ty[0], ty[1], self.width, self.height)

        self._needs_initial_autoscale = False
        # Flush interaction cache on manual view changes
        self.cache.active = False
        self.cache.capture_window = None
        self.frame.dirty_scene = True
        self.cache.refresh_requested = True

    def set_hud_enabled(self, enabled: bool) -> None:
        self.options.enable_hud = bool(enabled)
        self.frame.dirty_ui = True

    def set_blending_mode(self, mode: str | BlendMode) -> None:
        if isinstance(mode, str):
            mapping = {
                "auto": BlendMode.AUTO,
                "alpha": BlendMode.ALPHA,
                "on": BlendMode.ALPHA,  # Legacy shim
                "additive": BlendMode.ADDITIVE,
                "subtractive": BlendMode.SUBTRACTIVE,
                "screen": BlendMode.SCREEN,
                "off": BlendMode.OFF,
            }
            m = mode.lower()
            if m not in mapping:
                raise ValueError(
                    "blend mode must be 'auto', 'alpha', 'additive', 'subtractive', 'screen', "
                    "or 'off'"
                )
            mode = mapping[m]

        self.options.blend_mode = mode
        self.frame.dirty_scene = True

    def cycle_blending_mode(self) -> None:
        modes = [
            BlendMode.AUTO,
            BlendMode.ALPHA,
            BlendMode.ADDITIVE,
            BlendMode.SUBTRACTIVE,
            BlendMode.SCREEN,
            BlendMode.OFF,
        ]
        try:
            current_idx = modes.index(self.options.blend_mode)
        except ValueError:
            current_idx = 0

        idx = (current_idx + 1) % len(modes)
        self.options.blend_mode = modes[idx]
        self.frame.dirty_scene = True
        self.frame.dirty_ui = True

    def set_profile(self, name: str) -> None:
        """
        Applies a performance preset.
        Options: 'extreme', 'performance', 'balanced', 'quality'.
        """
        if name == "extreme":
            self.options.default_line_budget_per_px = 0.5
            self.options.interaction_budget_lines_per_screen_px = 1.0
            self.options.enable_cache_interaction_path = True
            self.options.cache_safe_margin = 0.4
        elif name == "performance":
            self.options.default_line_budget_per_px = 1.0
            self.options.interaction_budget_lines_per_screen_px = 2.0
            self.options.enable_cache_interaction_path = True
        elif name == "balanced":
            self.options.default_line_budget_per_px = 5.0
            self.options.interaction_budget_lines_per_screen_px = 5.0
            self.options.enable_cache_interaction_path = True
        elif name == "quality":
            self.options.default_line_budget_per_px = 20.0
            self.options.interaction_budget_lines_per_screen_px = 20.0
            self.options.enable_cache_interaction_path = False
        self.frame.dirty_scene = True

    def _get_all_layers(self) -> List[BaseLayer]:
        """
        Internal bridge: returns all active layers.
        The legacy LineDataset is now always mirrored into scene.layers as
        _primary_line_layer, so we just return scene.layers directly.
        """
        return list(self.scene.layers)

    def autoscale(self, axes: str = "both", padding: Optional[float] = None) -> None:
        """
        Autoscale view to fit all (legacy and layer) data.
        Supports axes="x", "y", or "both".

        ``padding`` is the fractional border left around the data. None means "read
        it from ``options.autoscale_margin``", which is what makes ``gplt.margins()``
        take effect on the next fit; an explicit number overrides that for this call.
        """
        if padding is None:
            padding = float(getattr(self.options, "autoscale_margin", 0.05))
        layers = self._get_all_layers()
        bounds = self.renderer_manager.get_bounds(layers)

        if bounds is None:
            if axes == "both":
                self.camera_controller.reset_view()
            return

        xmin, xmax, ymin, ymax = bounds

        # Every real scale is monotonic, so transforming the two bounds is equivalent to
        # transforming every point -- `get_bounds()` above stays reading raw layer data
        # either way. Only "log" can turn a bound into NaN (a raw min <= 0); symlog/asinh
        # are defined everywhere, so `_finite_log_bounds` is a no-op passthrough for them
        # and only does real work -- a sane fallback instead of a NaN camera bound -- when
        # the mode is "log".
        scale_x = getattr(self.options, "axis_scale_x", "linear")
        scale_y = getattr(self.options, "axis_scale_y", "linear")
        params_x = getattr(self.options, "axis_scale_params_x", None)
        params_y = getattr(self.options, "axis_scale_params_y", None)
        if scale_x != "linear":
            xmin, xmax = _finite_log_bounds(
                *_scale_forward(np.array([xmin, xmax]), scale_x, params_x)
            )
        if scale_y != "linear":
            ymin, ymax = _finite_log_bounds(
                *_scale_forward(np.array([ymin, ymax]), scale_y, params_y)
            )

        # Apply fractional padding
        dx = (xmax - xmin) * padding
        dy = (ymax - ymin) * padding

        # Sane defaults: only apply a fixed buffer if the original data span is zero
        # to prevent division by zero in camera projection.
        if (xmax - xmin) < 1e-9:
            dx = 0.5
        if (ymax - ymin) < 1e-9:
            dy = 0.5

        self.camera_controller.fit_bounds(
            xmin - dx, xmax + dx, ymin - dy, ymax + dy, self.width, self.height, axes=axes
        )

        self._needs_initial_autoscale = False
        self.cache.active = False
        self.frame.dirty_scene = True

    def reset_view(self) -> None:
        self.camera_controller.reset_view()
        self.frame.dirty_scene = True

    # --------------------------------------------------------
    # Panel layout
    # --------------------------------------------------------

    def set_panels(self, specs: "List") -> "List[Panel]":
        """Rebuild ``self.panels`` from a list of :class:`~glplot.core.layout.PanelSpec`.

        Every spec becomes a fresh, empty :class:`Panel` at the spec's rectangle -- matching
        ``plt.subplots``, which hands back empty axes. The active panel resets to the first
        one and a fresh autoscale is owed. An empty ``specs`` falls back to one full-window
        panel so the engine is never left with zero panels.
        """
        panels: List[Panel] = []
        for spec in specs:
            panel = Panel(self.options, rect_frac=tuple(spec.rect_frac), name=spec.name)
            # Grid coordinates, carried for matplotlib-style row/col indexing (Panel facade).
            panel.row = int(getattr(spec, "row", 0))
            panel.col = int(getattr(spec, "col", 0))
            panel.rowspan = int(getattr(spec, "rowspan", 1))
            panel.colspan = int(getattr(spec, "colspan", 1))
            panels.append(panel)
        if not panels:
            panels = [Panel(self.options, rect_frac=(0.0, 0.0, 1.0, 1.0))]
        self.panels = panels
        self.active_panel_index = 0
        self._needs_initial_autoscale = True
        self.frame.dirty_scene = True
        self.frame.dirty_ui = True
        self.frame.dirty_pick = True
        return panels

    def set_layout(self, nrows: int = 1, ncols: int = 1, **kwargs: Any) -> "List[Panel]":
        """Convenience: build a regular ``nrows`` x ``ncols`` grid of panels.

        See ``layout.grid``.
        """
        from .core import layout

        return self.set_panels(layout.grid(nrows, ncols, **kwargs))

    def _panel_index_at(self, mx: float, my: float) -> Optional[int]:
        """Index of the panel under window-space cursor ``(mx, my)``, or ``None``.

        Scans from the last-added panel back so a panel drawn on top of another (overlapping
        rects) wins the hit. With a single full-window panel this is always ``0``.
        """
        for i in range(len(self.panels) - 1, -1, -1):
            if self.panels[i].contains_window_px(mx, my, self.width, self.height):
                return i
        return None

    def _local_cursor(self, mx: float, my: float) -> Tuple[float, float, int, int]:
        """Map a window cursor to the ACTIVE panel: ``(local_x, local_y, panel_w, panel_h)``.

        The camera maths (``screen_to_world``, ``apply_zoom_at_cursor``) expect a cursor and a
        viewport size in the panel's own pixel space. For a full-window panel this returns the
        cursor and window size unchanged.
        """
        panel = self.active_panel
        lx, ly = panel.local_cursor(mx, my, self.width, self.height)
        pw, ph = panel.pixel_size(self.width, self.height)
        return lx, ly, pw, ph

    def _panel_screen_to_world(self, mx: float, my: float) -> Tuple[float, float]:
        """``screen_to_world`` for the active panel, taking a window-space cursor."""
        lx, ly, pw, ph = self._local_cursor(mx, my)
        return self.camera_controller.screen_to_world(lx, ly, pw, ph)

    def mouse_world_display(self) -> Optional[Tuple[float, float]]:
        """``self.mouse_world`` converted to real data values for user-facing display.

        ``mouse_world`` (and every other internal consumer of ``screen_to_world`` -- panning,
        box-zoom) stays in whatever space the camera actually works in, which is log10 space
        on a log axis. The HUD readout and a picked point's reported ``x``/``y`` are the two
        places that number reaches the user, so this is where -- and only where -- the
        inverse (``10**v``) belongs.
        """
        if self.mouse_world is None:
            return None
        wx, wy = self.mouse_world
        scale_x = getattr(self.options, "axis_scale_x", "linear")
        scale_y = getattr(self.options, "axis_scale_y", "linear")
        if scale_x == "linear" and scale_y == "linear":
            return wx, wy
        params_x = getattr(self.options, "axis_scale_params_x", None)
        params_y = getattr(self.options, "axis_scale_params_y", None)
        return (
            float(_scale_inverse(np.array(wx), scale_x, params_x)),
            float(_scale_inverse(np.array(wy), scale_y, params_y)),
        )

    def _activate_panel_under_cursor(self, mx: float, my: float) -> None:
        """Make the panel under the cursor the active one (no-op if the cursor is in a gutter)."""
        idx = self._panel_index_at(mx, my)
        if idx is not None:
            self.active_panel_index = idx

    def _sync_shared_axes(self) -> None:
        """Mirror the active panel's view onto its shared-x / shared-y group members.

        Called after any pan/zoom on the active panel. ``sharex`` copies the x centre and
        zoom; ``sharey`` copies the y ones. Linked panels have their interaction cache
        invalidated so the shared move repaints rather than showing a stale impostor.
        """
        src = self.active_panel
        if src.sharex_group:
            for p in src.sharex_group:
                if p is not src:
                    p.camera.cx = src.camera.cx
                    p.camera.zoom_x = src.camera.zoom_x
                    p.cache.active = False
                    p.cache.refresh_requested = True
        if src.sharey_group:
            for p in src.sharey_group:
                if p is not src:
                    p.camera.cy = src.camera.cy
                    p.camera.zoom_y = src.camera.zoom_y
                    p.cache.active = False
                    p.cache.refresh_requested = True

    def set_shared_axes(self, sharex: bool, sharey: bool) -> None:
        """Link (or unlink) every panel's x and/or y axis so pan/zoom on one moves them all.

        Linking points every panel's ``sharex_group`` / ``sharey_group`` at the whole panel
        list; unlinking clears it. With one panel there is nothing to share, so the groups
        stay ``None``. After (re)linking, the panels are brought into agreement immediately
        using the active panel as the reference.
        """
        panels = list(self.panels)
        group = panels if len(panels) > 1 else None
        for p in panels:
            p.sharex_group = group if sharex else None
            p.sharey_group = group if sharey else None
        self._sync_shared_axes()
        self.frame.dirty_scene = True

    def add_panel(self, spec: Any) -> Panel:
        """Append one panel from a :class:`~glplot.core.layout.PanelSpec` and return it.

        Used by ``subplot2grid``, which places spanning axes one call at a time into the same
        grid. The new panel becomes the active one and owes an initial autoscale.
        """
        panel = Panel(self.options, rect_frac=tuple(spec.rect_frac), name=spec.name)
        panel.row = int(getattr(spec, "row", 0))
        panel.col = int(getattr(spec, "col", 0))
        panel.rowspan = int(getattr(spec, "rowspan", 1))
        panel.colspan = int(getattr(spec, "colspan", 1))
        self.panels.append(panel)
        self.active_panel_index = len(self.panels) - 1
        self._needs_initial_autoscale = True
        self.frame.dirty_scene = True
        self.frame.dirty_ui = True
        self.frame.dirty_pick = True
        return panel

    def split_view(
        self,
        nrows: int,
        ncols: int,
        *,
        wspace: Optional[float] = None,
        hspace: Optional[float] = None,
        outer: Optional[float] = None,
    ) -> "List[Panel]":
        """Re-tile the window into an ``nrows`` x ``ncols`` grid, keeping current content.

        Unlike ``set_panels`` (which starts every panel empty, as ``plt.subplots`` does), this
        preserves the active panel's scene and camera in the top-left cell and fills the rest
        with fresh empty panels. It backs the GUI's interactive "split" so a user does not lose
        the plot they were looking at when they divide the view. ``wspace``/``hspace``/``outer``
        control how tightly the panels are packed (figure fractions); ``None`` keeps the grid
        defaults.
        """
        from .core import layout

        kw = {}
        if wspace is not None:
            kw["wspace"] = wspace
        if hspace is not None:
            kw["hspace"] = hspace
        if outer is not None:
            kw["outer"] = outer
        specs = layout.grid(nrows, ncols, **kw)
        kept = self.active_panel
        panels: List[Panel] = []
        for i, spec in enumerate(specs):
            if i == 0:
                kept.rect_frac = tuple(spec.rect_frac)
                kept.row, kept.col = spec.row, spec.col
                kept.rowspan, kept.colspan = spec.rowspan, spec.colspan
                kept.sharex_group = None
                kept.sharey_group = None
                panels.append(kept)
            else:
                p = Panel(self.options, rect_frac=tuple(spec.rect_frac), name=spec.name)
                p.row, p.col = spec.row, spec.col
                p.rowspan, p.colspan = spec.rowspan, spec.colspan
                panels.append(p)
        self.panels = panels
        self.active_panel_index = 0
        self.frame.dirty_scene = True
        self.frame.dirty_ui = True
        self.frame.dirty_pick = True
        return panels

    def retile_current(self, wspace: float, hspace: float, outer: float) -> bool:
        """Reflow the *existing* panels to new grid spacing, keeping every panel's content.

        The compactness control: it changes only the rectangles, not the panels, so no plot is
        lost. Works on a regular ``rows x cols`` grid (the shape ``split_view`` /
        ``plt.subplots`` make) -- it infers the shape from the panels' row/col and reassigns
        each panel's ``rect_frac`` from a freshly spaced grid. Returns False (a no-op) for a
        single panel or an irregular layout (mosaic / spanning cells) it cannot cleanly reflow.
        """
        from .core import layout

        panels = self.panels
        if len(panels) <= 1:
            return False
        nrows = max(p.row for p in panels) + 1
        ncols = max(p.col for p in panels) + 1
        if len(panels) != nrows * ncols or any(p.rowspan != 1 or p.colspan != 1 for p in panels):
            return False  # mosaic / spanning grid — not a plain grid, leave it alone
        specs = {
            (s.row, s.col): s
            for s in layout.grid(nrows, ncols, wspace=wspace, hspace=hspace, outer=outer)
        }
        for p in panels:
            spec = specs.get((p.row, p.col))
            if spec is not None:
                p.rect_frac = tuple(spec.rect_frac)
        self.frame.dirty_scene = True
        self.frame.dirty_ui = True
        self.frame.dirty_pick = True
        return True

    def merge_view(self) -> "List[Panel]":
        """Collapse back to a single full-window panel, keeping the active panel's content."""
        kept = self.active_panel
        kept.rect_frac = (0.0, 0.0, 1.0, 1.0)
        kept.row = kept.col = 0
        kept.rowspan = kept.colspan = 1
        kept.sharex_group = None
        kept.sharey_group = None
        self.panels = [kept]
        self.active_panel_index = 0
        self.frame.dirty_scene = True
        self.frame.dirty_ui = True
        self.frame.dirty_pick = True
        return self.panels

    def _autoscale_all_panels(self) -> None:
        """Autoscale every panel that still owes an initial fit, restoring the active one."""
        saved = self.active_panel_index
        try:
            for i in range(len(self.panels)):
                self.active_panel_index = i
                panel = self.panels[i]
                if panel.needs_initial_autoscale and not self._is_pure_3d_scene():
                    self.autoscale()
                    panel.needs_initial_autoscale = False
        finally:
            self.active_panel_index = saved

    def clear(self) -> None:
        self.scene = SceneData()
        # set_lines_ab() caches this to update the line-family layer in place on a
        # second call rather than re-inserting it -- without resetting it here, that
        # cache survives the fresh SceneData() above and points at a Layer object no
        # longer in scene.layers, so the very next set_lines_ab() call "updates" a
        # layer that isn't in the scene, leaving scene.layers empty and crashing the
        # pyplot-level plot_lines()'s scene.layers[-1] lookup with an IndexError. Hit
        # by any script that calls cla()/clf() then plot_lines() again, e.g. a
        # FuncAnimation whose update() does exactly that every frame.
        self._primary_line_layer = None
        self.frame.dirty_scene = True

    def close(self) -> None:
        if self.window:
            glfw.set_window_should_close(self.window, True)

    def run(self) -> None:
        self._init_window()
        self._init_gl()
        self._init_modules()
        if self._needs_initial_autoscale and not self._is_pure_3d_scene():
            self._autoscale_all_panels()
        if self._is_test_mode:
            self._update_runtime_policy()
            glViewport(0, 0, self.fb_width, self.fb_height)
            # 1. Clear Frame (Primary Surface)
            c = self.options.visual.background_color
            glClearColor(c[0], c[1], c[2], 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            self._apply_blending_policy()
            # Test mode is a single exact frame with no imgui context, so it skips the
            # overlay pass (axis labels / legend go through the imgui draw list, which needs
            # an active frame the main loop opens with hud.begin()).
            self._draw_panels(self._draw_exact_view)
            glfw.swap_buffers(self.window)
            return

        self._main_loop()

    def save_current_view(self, filename: Optional[str] = None, scale: float = 2.0) -> None:
        # Legacy shim
        fname = filename or f"plot_{int(time.time())}.png"
        self.savefig(fname, scale=scale)

    def get_density_array(self) -> np.ndarray:
        """
        Read back the accumulated density values from the GPU framebuffer texture
        and return them as a 2D numpy array of shape (height, width).
        """
        if self.window:
            glfw.make_context_current(self.window)
        return self.density_renderer.get_density_array()

    # --------------------------------------------------------
    # Init
    # --------------------------------------------------------

    def _init_window(self) -> None:
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # Destroy any existing window (e.g. a previous headless context created
        # by savefig()) so we don't leak GLFW resources.
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
        if self.options.enable_multisample:
            glfw.window_hint(glfw.SAMPLES, 4)

        if self._is_test_mode:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        self.window = glfw.create_window(self.width, self.height, self.options.title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        self.width, self.height = glfw.get_window_size(self.window)
        self.fb_width, self.fb_height = glfw.get_framebuffer_size(self.window)

        glfw.set_window_size_callback(self.window, self._on_resize)
        glfw.set_framebuffer_size_callback(self.window, self._on_fb_resize)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_char_callback(self.window, self._on_char)

    def _init_gl(self) -> None:
        glViewport(0, 0, self.fb_width, self.fb_height)
        glClearColor(1.0, 1.0, 1.0, 1.0)

        # Clipping Optimizations (Must be enabled for shaders to work correctly)
        if self.options.enable_clipping_optimization:
            for i in range(4):
                glEnable(GL_CLIP_DISTANCE0 + i)

        if self.options.enable_multisample:
            glEnable(GL_MULTISAMPLE)
        else:
            glDisable(GL_MULTISAMPLE)

    def _init_modules(self) -> None:
        # When a new GL context is created (e.g. show() after a headless savefig()),
        # any per-layer VAO/VBO handles from the previous context are invalid.
        # Reset them so renderers re-upload data into the new context on first draw.
        for layer in self.scene.layers:
            if hasattr(layer, "_gl"):
                layer._gl = None
            layer.dirty.gpu_dirty = True

        self.exact_renderer.initialize()
        self.interaction_renderer.initialize(self.fb_width, self.fb_height)
        self.density_renderer.initialize(self.fb_width, self.fb_height)
        self.hud.initialize(self.window)
        self.picking.initialize(self.fb_width, self.fb_height)
        # A new GL context was created (e.g. show() after a headless savefig()). The
        # EffectManager's FBO/program IDs from the old context are stale, so force it
        # to rebuild all resources in the current context.
        self.effects.initialized = False
        self.effects.ensure_resources()
        self.renderer_manager.initialize()

        if self.scene.lines.count > 0:
            self.exact_renderer.upload(self.scene.lines)

    # --------------------------------------------------------
    # Frame policies
    # --------------------------------------------------------

    def _update_runtime_policy(self) -> None:
        prev_mode = self.policy.runtime.current_mode
        self.policy.update(self.scene, self.interaction, self.cache)
        if (
            prev_mode == RenderMode.INTERACTIVE
            and self.policy.runtime.current_mode == RenderMode.EXACT
        ):
            self.frame.dirty_scene = True
        if self.hud.state.show_profiler:
            self.policy.runtime.hud_enabled_this_frame = True

    def set_background_mode(self, light: bool, *, apply_preset: bool = True) -> None:
        """Switch between the light and dark background presets.

        ``apply_preset=False`` adopts the mode *without* touching any colour: for a caller
        that is setting its own background and only wants ``light_bg_mode`` to describe it
        (the GUI's style presets do exactly this — see ``glplot/gui/styles.py``). Without
        it, the next :meth:`_apply_background_mode` would see the flag change and overwrite
        the caller's background with the stock one.
        """
        self.options.light_bg_mode = bool(light)
        if apply_preset:
            self._background_mode_applied = None  # force the latch below to re-apply
        else:
            self._background_mode_applied = bool(light)
        self._apply_background_mode()

    def _apply_background_mode(self) -> None:
        """Apply the stock background/grid colours for ``light_bg_mode``, when it changes.

        **The latch is load-bearing.** This is called once per frame from ``_main_loop``,
        and it used to run its body unconditionally: it overwrote ``background_color``, the
        entire ``gradient_background`` and ``axis_grid_color`` on *every frame*, so those
        four public options could not be set at all. Assigning
        ``plot.options.visual.background_color`` survived only until the loop reached this
        line, and the HUD's own gradient checkbox could never stay off. Re-applying only on
        a *change* of ``light_bg_mode`` keeps the intended behaviour — flipping the mode
        (HUD 'B', :meth:`set_background_mode`) still restyles the background — while
        treating these colours as what they always were: a default, not an invariant.
        """
        light = bool(getattr(self.options, "light_bg_mode", False))
        if light == self._background_mode_applied:
            return
        self._background_mode_applied = light

        if light:
            # Light BG mode
            self.options.visual.background_color = (1.0, 1.0, 1.0)
            self.options.visual.gradient_background.enabled = True
            self.options.visual.gradient_background.top_color = (1.0, 1.0, 1.0)
            self.options.visual.gradient_background.bottom_color = (0.95, 0.95, 0.95)
            self.options.axis_grid_color = (0.8, 0.8, 0.8)
        else:
            # Dark BG mode
            self.options.visual.background_color = (0.0, 0.0, 0.0)
            self.options.visual.gradient_background.enabled = True
            self.options.visual.gradient_background.top_color = (0.0, 0.0, 0.0)
            self.options.visual.gradient_background.bottom_color = (0.08, 0.08, 0.12)
            self.options.axis_grid_color = (0.2, 0.2, 0.2)

    def _get_adaptive_alpha(self, count: int) -> float:
        """
        Calculates a balanced alpha value based on object count and display density (DPR).
        Ensures visibility on High-DPI displays while preventing saturation on dense datasets.
        """
        base_alpha = self.options.default_global_alpha

        if self.options.enable_auto_alpha and count > 1000:
            scale_factor = math.sqrt(count / 1000.0)
            # Unified Floor at 0.15 to ensure visibility
            base_alpha = max(0.15, base_alpha / scale_factor)

        # High-DPI (Retina) compensation: single-pixel lines are physically thinner,
        # so we boost alpha to maintain perceived weight.
        dpr = self.fb_width / max(self.width, 1)
        if dpr > 1.1:
            base_alpha = min(1.0, base_alpha * 1.5)

        return float(base_alpha)

    def _compute_lod_keep_prob(self) -> float:
        """
        Calculates the fraction of objects to keep during interaction (LOD).
        Uses a width-aware policy that accounts for fill-rate.
        """
        if not self.options.lod_enabled:
            return 1.0

        window = self.camera_controller.world_window(self.width, self.height)
        ndc_scale, ndc_offset = self._get_ndc_transform(window)

        ctx = RenderContext(
            mvp=self.camera_controller.mvp(self.width, self.height),
            window_world=window,
            ndc_scale=ndc_scale,
            ndc_offset=ndc_offset,
            width_px=self.width,
            height_px=self.height,
            fb_width=self.fb_width,
            fb_height=self.fb_height,
            mode=self.policy.runtime.current_mode,
            global_alpha=self.options.default_global_alpha,
            lod_keep_prob=1.0,
            is_density=self.display_density,
            time=time.perf_counter(),
        )

        return self.policy.calculate_width_aware_lod(self.scene, ctx)

    def _get_ndc_transform(
        self, window: Tuple[float, float, float, float]
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Calculate scale and offset to transform world coordinates to NDC [-1, 1]."""
        l, r, b, t = window
        rl = r - l
        tb = t - b
        sx = 2.0 / max(rl, 1e-12)
        sy = 2.0 / max(tb, 1e-12)
        ox = -(r + l) / max(rl, 1e-12)
        oy = -(t + b) / max(tb, 1e-12)
        return (sx, sy), (ox, oy)

    def _apply_blending_policy(self, premultiplied_target: bool = False) -> None:
        """Set the blend function for ``options.blend_mode``, the figure-wide one.

        A layer that carries a ``style.blend_mode`` of its own overrides this for the length
        of its own draw and restores it afterwards, which is why the mode-to-GL-state
        mapping lives in :func:`glplot.utils.blending.apply_blend_mode` rather than here:
        two callers, one meaning. ``premultiplied_target`` is documented there too.
        """
        from .utils.blending import apply_blend_mode

        if not self.policy.runtime.blending_enabled:
            glDisable(GL_BLEND)
            return

        glEnable(GL_BLEND)
        apply_blend_mode(self.options.blend_mode, premultiplied_target=premultiplied_target)

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    def _draw_overlays(self) -> None:
        """Draw the axis scale labels and the legend for the active panel.

        Screen-aligned overlays that must repaint every frame (so they stay live and survive
        the cached impostor). Reads the active panel's camera and the engine's current pixel
        dimensions, which ``_draw_panels`` sets to the panel's rect when several panels exist.
        """
        # Draw Axis Labels (Scale) on every frame so they update dynamically and persist when
        # the scene is cached.
        if self.options.axis_show_labels:
            window_world = self.camera_controller.world_window(self.width, self.height)
            mvp = self.camera_controller.mvp(self.width, self.height)
            ndc_scale, ndc_offset = self._get_ndc_transform(window_world)
            ctx_labels = RenderContext(
                mvp=mvp,
                window_world=window_world,
                ndc_scale=ndc_scale,
                ndc_offset=ndc_offset,
                width_px=self.width,
                height_px=self.height,
                fb_width=self.fb_width,
                fb_height=self.fb_height,
                dpr=self.fb_width / max(self.width, 1),
                mode=self.policy.runtime.current_mode,
                global_alpha=1.0,
                lod_keep_prob=1.0,
                is_density=self.display_density,
                time=time.perf_counter(),
                px_offset=self._panel_offset_px,
            )
            if self._is_pure_3d_scene():
                # A 3D scene's numbers live on the box's edges, not on two screen-aligned
                # spines, so they come from the 3D anchors rather than the AxisManager.
                # Same option gate: "show axis labels" means the same thing in both modes.
                axes3d.draw_labels(self)
            else:
                self.axis_manager.update(ctx_labels)
                self.renderer_manager.renderers["axis"]._draw_labels(self.axis_manager, ctx_labels)

        # The legend, on the same background draw list as the axis labels and for the
        # same reason: it must composite over the cached impostor, which is not
        # redrawn during an interaction. Deliberately outside the `axis_show_labels`
        # gate (a legend is not an axis annotation) and outside the pure-3D gate (a 3D
        # scene has series worth naming too). Draws nothing unless `legend()` or the
        # `legend_show` option asked for it -- see `renderers.legend.draw_legend`.
        draw_legend(self)

    def _draw_panels(self, view_fn) -> None:
        """Run ``view_fn`` once per panel, each inside its own ``glViewport``/scissor rect.

        With a single full-window panel this is a no-op wrapper -- ``view_fn`` runs once
        against the full framebuffer exactly as before, so the pre-multi-panel path is
        untouched. With several panels, the active panel and the engine's pixel dimensions
        (``width``/``height``/``fb_width``/``fb_height``) are swapped to the panel's size for
        the duration of each call. Every downstream consumer -- the exact/interaction view,
        the axis labels, the legend -- reads those, so it draws into the panel's rectangle
        without any of them needing to know panels exist. State is restored in ``finally``.
        """
        if len(self.panels) <= 1:
            view_fn()
            return

        from contextlib import nullcontext

        from .options import override_axis_margins

        # Compact per-panel gutters (if set) so split frames pack closely instead of each
        # reserving the full/rail-widened left gutter. Wraps the whole loop so both the frame
        # projection (mvp) and the imgui labels see the same tight gutters.
        margins_ctx = (
            override_axis_margins(self.options, self._panel_margins)
            if self._panel_margins is not None
            else nullcontext()
        )

        saved_idx = self.active_panel_index
        saved_w, saved_h = self.width, self.height
        saved_fw, saved_fh = self.fb_width, self.fb_height
        try:
            with margins_ctx:
                for i, panel in enumerate(self.panels):
                    self.active_panel_index = i
                    vx, vy, vw, vh = panel.pixel_rect(saved_fw, saved_fh)
                    self.width, self.height = panel.pixel_size(saved_w, saved_h)
                    self.fb_width, self.fb_height = vw, vh
                    # Window-pixel offset for imgui overlays (tick labels, legend).
                    self._panel_offset_px = panel.window_offset_px(saved_w, saved_h)
                    # Framebuffer rect for the density resolve (which otherwise targets 0,0).
                    self._panel_fb_rect = (vx, vy, vw, vh)
                    glViewport(vx, vy, vw, vh)
                    # Scissor bounds any full-viewport fill (axis background, gradients) to
                    # this panel so it cannot paint over its neighbours.
                    glEnable(GL_SCISSOR_TEST)
                    glScissor(vx, vy, vw, vh)
                    view_fn()
        finally:
            glDisable(GL_SCISSOR_TEST)
            self.active_panel_index = saved_idx
            self.width, self.height = saved_w, saved_h
            self.fb_width, self.fb_height = saved_fw, saved_fh
            self._panel_offset_px = (0.0, 0.0)
            self._panel_fb_rect = None
            glViewport(0, 0, saved_fw, saved_fh)

    def _draw_exact_view(self) -> None:
        t_start = time.perf_counter()
        self._apply_blending_policy()

        # 1. Prepare RenderContext for this frame
        mvp = self.camera_controller.mvp(self.width, self.height)
        window = self.camera_controller.world_window(self.width, self.height)
        prob = self._compute_lod_keep_prob()
        base_alpha = self._get_adaptive_alpha(self.scene.lines.count)

        ndc_scale, ndc_offset = self._get_ndc_transform(window)

        ctx = RenderContext(
            mvp=mvp,
            window_world=window,
            ndc_scale=ndc_scale,
            ndc_offset=ndc_offset,
            width_px=self.width,
            height_px=self.height,
            fb_width=self.fb_width,
            fb_height=self.fb_height,
            dpr=self.fb_width / max(self.width, 1),
            mode=self.policy.runtime.current_mode,
            global_alpha=base_alpha,
            lod_keep_prob=prob,
            is_density=self.display_density,
            time=time.perf_counter(),
        )

        # 2. Draw using the new RendererManager (Modular Architecture)
        layers = self._get_all_layers()

        # Skip 2D axis overlay for pure 3D scenes — 3D bounding box from ensure_3d_axes() takes
        # its place
        if not self._is_pure_3d_scene():
            self.axis_manager.update(ctx)
            self.renderer_manager.draw_axes(self.axis_manager, ctx)

        if self._density_active():
            # Modular Density Pass (Lines, Scatters)
            current_fbo = int(glGetIntegerv(GL_FRAMEBUFFER_BINDING))
            self.renderer_manager.draw_density(
                layers, ctx, target_fbo=current_fbo, target_viewport=self._panel_fb_rect
            )
        else:
            # Standard Pass
            self.renderer_manager.draw_exact(layers, ctx)

        # Overlay Text pass (screen-aligned, always last)
        self.renderer_manager.renderers["text"].draw_all(layers, ctx)

        self.hud.state.gpu_timings["Exact Render"] = time.perf_counter() - t_start

    def _draw_interaction_view(self) -> None:
        t_start = time.perf_counter()
        self._apply_blending_policy()

        # Disable world clipping for screen-space impostor
        if self.options.enable_clipping_optimization:
            for i in range(4):
                glDisable(GL_CLIP_DISTANCE0 + i)

        current_window = self.camera_controller.world_window(self.width, self.height)
        if (
            self.options.enable_cache_interaction_path
            # cache.active is what _service_deferred_cache_refresh() gates refreshes
            # on, so it is also the only state in which the cache is being kept up to
            # date. Showing the impostor without it displays a texture nobody repaints:
            # between a press and the drag threshold drag_active alone puts us in
            # INTERACTIVE mode, and capture_window outlives both a view change made by
            # other means and the blank texture _on_fb_resize() reallocates.
            and self.cache.active
            and self.cache.capture_window is not None
            and not self.is_3d_scene()
            # The interaction impostor caches the full framebuffer; with several panels each
            # occupies a sub-rect, so fall back to an exact per-panel redraw during drags.
            and len(self.panels) == 1
        ):
            current_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
            self.interaction_renderer.draw_cached_impostor(
                self.cache.capture_window, current_window, target_fbo=current_fbo
            )
        else:
            self._draw_exact_view()

        # Re-enable if needed for next passes (exact view usually enables it anyway)
        if self.options.enable_clipping_optimization:
            for i in range(4):
                glEnable(GL_CLIP_DISTANCE0 + i)

        self.hud.state.gpu_timings["Interaction"] = time.perf_counter() - t_start

    def _capture_interaction_cache(self) -> None:
        capture_window = self.camera_controller.world_window(
            self.width,
            self.height,
            padding=self.options.cache_padding,
        )
        mvp = self.camera_controller.mvp(self.width, self.height, window=capture_window)
        target_fbo = self.interaction_renderer.cache_target.fbo
        target_size = (self.fb_width, self.fb_height)

        glBindFramebuffer(GL_FRAMEBUFFER, target_fbo)
        glViewport(0, 0, self.fb_width, self.fb_height)
        # Transparent BLACK — premultiplied alpha so that blending src*A + 0*(1-A)
        # stores src.RGB*A in the cache instead of mixing with white.
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)

        prob = self._compute_lod_keep_prob()
        base_alpha = self._get_adaptive_alpha(self.scene.lines.count)

        if prob < 1.0:
            base_alpha = 1.0

        ndc_scale, ndc_offset = self._get_ndc_transform(capture_window)

        # How much wider this capture is than the view it stands in for. The impostor
        # magnifies the texture back by exactly this factor, so a renderer whose primitives
        # have a fixed *pixel* size has to account for it (see RenderContext.capture_scale).
        cur_l, cur_r, _, _ = self.camera_controller.world_window(self.width, self.height)
        capture_scale = (capture_window[1] - capture_window[0]) / max(cur_r - cur_l, 1e-12)

        ctx = RenderContext(
            mvp=mvp,
            window_world=capture_window,
            ndc_scale=ndc_scale,
            ndc_offset=ndc_offset,
            width_px=self.width,
            height_px=self.height,
            fb_width=self.fb_width,
            fb_height=self.fb_height,
            dpr=self.fb_width / max(self.width, 1),
            mode=RenderMode.INTERACTIVE,
            global_alpha=base_alpha,
            lod_keep_prob=prob,
            is_density=self.display_density,
            time=time.perf_counter(),
            capture_scale=float(max(capture_scale, 1.0)),
        )

        layers = self._get_all_layers()

        if self._density_active():
            self.renderer_manager.draw_density(
                layers, ctx, target_fbo=target_fbo, target_size=target_size
            )
        else:
            self._apply_blending_policy(premultiplied_target=True)
            # Only draw primal geometry into the interaction cache
            # HUD/Axes/Labels are overlays drawn in the main view pass
            self.renderer_manager.draw_exact(layers, ctx)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self.cache.capture_window = capture_window
        self.cache.last_capture_time = glfw.get_time()
        self.cache.refresh_requested = False

    def _cache_needs_refresh(self) -> bool:
        if not self.cache.capture_window:
            return True

        cl, cr, cb, ct = self.cache.capture_window
        l, r, b, t = self.camera_controller.world_window(self.width, self.height)
        margin = self.options.cache_safe_margin
        cw, ch = (cr - cl), (ct - cb)
        return (
            l < cl + cw * margin
            or r > cr - cw * margin
            or b < cb + ch * margin
            or t > ct - ch * margin
        )

    def _service_deferred_cache_refresh(self) -> None:
        if not self.cache.active:
            return
        if self.is_3d_scene():
            return
        if not self.cache.refresh_requested:
            return
        now = glfw.get_time()
        min_dt = 1.0 / max(self.options.cache_refresh_hz, 1e-6)
        if now - self.cache.last_capture_time >= min_dt:
            self._capture_interaction_cache()

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    def _main_loop(self) -> None:
        glfw.swap_interval(1)

        while not glfw.window_should_close(self.window):
            if self.options.reactive_rendering:
                glfw.wait_events_timeout(0.02)
            else:
                glfw.poll_events()

            # 0. Drain GUI workspace commands.
            # Ordering is load-bearing: panels may not mutate the scene from their draw
            # callbacks, so they queue closures instead, and the queue must run HERE --
            # at the top of the frame, before the need_render gate below. The dirty flags
            # a queued mutation raises then survive to gate this frame's render and are
            # only cleared after it (at the bottom of the loop). Draining any later would
            # latch-lose them and a GUI-created layer would not appear until an unrelated
            # event happened to wake the loop.
            gui_queue = getattr(getattr(self.hud, "workspace", None), "queue", None)
            if gui_queue is not None:
                gui_queue.drain(self)

            # 1. Update Input and State
            self.hud.process_inputs()
            self._update_runtime_policy()
            self._apply_background_mode()

            # Check cache release deadline
            t_now = glfw.get_time()

            # Auto-rotate. Advanced from wall-clock rather than per frame so the turntable
            # runs at the speed the user asked for regardless of frame rate, and gated on
            # the flag so a still scene stays still (and, under reactive rendering, keeps
            # the loop asleep).
            self._advance_auto_spin(t_now)

            # View-driven layers: anything sampled against the *screen* rather than stored
            # in data space is re-evaluated here when the view has moved.
            self._resample_view_layers()

            # Animation. Advances every panel's timeline and applies it, then runs any
            # registered per-frame callbacks.
            self._advance_timelines(t_now)
            self._run_frame_callbacks(t_now)
            if (
                self.cache.active
                and not self.interaction.drag_active
                and not self.interaction.right_drag_active
                and t_now >= self.cache.release_deadline
            ):
                self.cache.active = False
                self.frame.dirty_scene = True

            need_render = (
                not self.options.reactive_rendering
                or self.frame.dirty_scene
                or self.frame.dirty_ui
                or self.frame.dirty_pick
                or self.interaction.drag_active
                or self.interaction.right_drag_active
                or self.hud.state.show_profiler
                # A command queued after the drain above (or re-queued by one) must wake a
                # sleeping reactive loop rather than wait for the next incidental event.
                or (gui_queue is not None and not gui_queue.is_empty())
            )

            if not need_render:
                continue

            # 2. Start ImGui frame before ANY rendering/processing happens
            self.hud.begin()

            self._service_deferred_cache_refresh()

            # Picking Pass (Deferred).
            # dirty_pick is set on explicit Shift+Click → always honour it.
            # The extra gate only applies to continuous hover-picking when shift is held.
            if self.frame.dirty_pick:
                run_pick = (
                    (not self.options.shift_required_for_picking)
                    or self.interaction.shift_down
                    or self.interaction.explicit_pick_requested
                )
                if run_pick:
                    # Only a completed Shift+Click may rewrite the selection; a
                    # hover/scroll/resize re-pick must leave it alone.
                    self._run_picking_pass(
                        update_selection=self.interaction.selection_pick_requested
                    )
                self.frame.dirty_pick = False
                self.interaction.explicit_pick_requested = False
                self.interaction.selection_pick_requested = False

            t0 = glfw.get_time()

            t_scene_start = time.perf_counter()
            if self.frame.dirty_scene or not self.effects.any_post_enabled():
                self.effects.begin_scene()

                self.effects.draw_background()
                self._apply_blending_policy()

                if self.policy.runtime.current_mode == RenderMode.INTERACTIVE:
                    self._draw_panels(self._draw_interaction_view)
                else:
                    self._draw_panels(self._draw_exact_view)

                # Draw zoom box if active
                if self.interaction.right_drag_active:
                    if self.options.enable_clipping_optimization:
                        for i in range(4):
                            glDisable(GL_CLIP_DISTANCE0 + i)
                    self._draw_zoom_box()

                self.effects.end_scene()
            else:
                self.effects.resolve()
            self.hud.state.gpu_timings["Render Scene"] = time.perf_counter() - t_scene_start

            # Marquee rubber band. Outside the dirty_scene branch above on
            # purpose: it goes to the ImGui draw list (composited at hud.end()),
            # and a still mouse leaves dirty_scene False while the drag is live,
            # which would blink the band out on those frames.
            if (
                self.interaction.drag_active
                and self.interaction.drag_confirmed
                and self.interaction.drag_mode == "marquee"
            ):
                self._draw_marquee_box()

            # Panel frames + active-panel highlight (only when the window is split).
            self._draw_panel_borders()

            # Update HUD metrics and Draw
            self._service_hud_metrics(t0)

            # Disable world clipping for HUD
            if self.options.enable_clipping_optimization:
                for i in range(4):
                    glDisable(GL_CLIP_DISTANCE0 + i)

            t_hud_start = time.perf_counter()
            # Axis labels + legend, once per panel (each reads the active panel's camera and
            # pixel size via _draw_panels, which swaps them for the panel's rect).
            self._draw_panels(self._draw_overlays)

            # HUD panels are only updated if HUD is enabled, but begin/end must wrap all
            if self.policy.runtime.hud_enabled_this_frame:
                self.hud.update()

            self.hud.end()
            self.hud.state.gpu_timings["UI Panels"] = time.perf_counter() - t_hud_start

            # Note: GL state is cleaned up/reset at start of next frame or specific renderers

            t_swap_start = time.perf_counter()
            glfw.swap_buffers(self.window)
            self.hud.state.gpu_timings["Buffer Swap"] = time.perf_counter() - t_swap_start
            t1 = glfw.get_time()

            dt = max(t1 - t0, 1e-6)
            self.frame.fps_estimate = 1.0 / dt
            self.frame.last_frame_time = t1
            self.frame.dirty_scene = False
            self.frame.dirty_ui = False

            # Progressive refinement. A renderer that drew a cheap preview because the view
            # was moving asks here for one more frame at full quality. It has to come after
            # the flags are cleared, or the clear above would swallow it and a reactive loop
            # would go to sleep on the low-quality image. See renderers/fractal.py.
            if self._refinement_pending():
                self.frame.dirty_scene = True

        self.effects.shutdown()

    def _refinement_pending(self) -> bool:
        """Whether any renderer drew a reduced-quality frame and wants a full one.

        Duck-typed rather than hard-wired to the fractal: ``consume_refine_request`` is the
        contract, and any renderer that trades quality for latency during motion can opt in
        by growing one. Draining every renderer (rather than stopping at the first True)
        matters — a scene with two of them must not leave the second's request latched.
        """
        renderers = getattr(getattr(self, "renderer_manager", None), "renderers", None)
        if not renderers:
            return False
        pending = False
        for renderer in renderers.values():
            consume = getattr(renderer, "consume_refine_request", None)
            if callable(consume) and consume():
                pending = True
        return pending

    def _service_hud_metrics(self, t0: float) -> None:
        now = glfw.get_time()

        # Fast bucket (Every frame)
        self.hud.state.cpu_frame_times.append(time.perf_counter() - self._last_perf_t)
        self._last_perf_t = time.perf_counter()
        self.hud.state.selected_object = self.picked_info

        # Medium bucket (4 Hz)
        if now - self.hud.state.last_medium_update > 0.25:
            self.hud.state.last_medium_update = now
            # Profiler stats
            self.hud.state.fps_history.append(self.frame.fps_estimate)

        # Slow bucket (2 Hz or Idle)
        if now - self.hud.state.last_slow_update > 0.5:
            self.hud.state.last_slow_update = now
            self._update_slow_analysis()

    def _update_slow_analysis(self) -> None:
        # Sampled histograms for performance
        if self.scene.lines.ab is not None:
            n = self.scene.lines.count
            sample_size = min(n, 10000)
            indices = np.random.choice(n, sample_size, replace=False)
            sample = self.scene.lines.ab[indices]

            # Simple histogram calculation
            hist_a, _ = np.histogram(sample[:, 0], bins=50)
            hist_b, _ = np.histogram(sample[:, 1], bins=50)
            self.hud.state.sampled_histogram_a = hist_a.astype(np.float32)
            self.hud.state.sampled_histogram_b = hist_b.astype(np.float32)

    def _draw_zoom_box(self) -> None:
        # Modern replacement for immediate mode glBegin
        px, py = self.interaction.right_press_mouse
        mx, my = self.interaction.last_mouse

        # We reuse the TextRenderer's unit quad or similar to avoid defining a new VAO just for
        # this.
        # However, for robustness, we'll just use AxisRenderer's logic or simple GL lines.
        # Actually, let's just use the TextRenderer's draw_list approach if available,
        # but since we are in the engine, we'll do a quick VAO-less draw if possible,
        # or just use a simple 4-vertex local buffer.

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # For V1 optimization, we'll use a simple attribute-less draw or just keep it simple.
        # Since this is a UI element, using the ImGui draw list is the best path.
        draw_list = self.hud.get_draw_list()
        if draw_list:
            color = 0x4C3366CC  # Abgr: (0.3, 0.4, 0.8, 1.0) approx
            draw_list.add_rect_filled(px, py, mx, my, color)
            draw_list.add_rect(px, py, mx, my, 0xCC3366CC)

    def _draw_panel_borders(self) -> None:
        """Outline each panel and highlight the active one, so the current axes is obvious.

        Only draws when the window is actually split (more than one panel). Uses the ImGui
        draw list in logical window pixels (top-left origin), converting from each panel's
        bottom-left figure-fraction rect. The active panel gets a bright, thicker frame; the
        rest a faint one.
        """
        if len(self.panels) <= 1:
            return
        draw_list = self.hud.get_draw_list()
        if not draw_list:
            return
        for i, panel in enumerate(self.panels):
            x0f, y0f, wf, hf = panel.rect_frac
            left = x0f * self.width
            right = (x0f + wf) * self.width
            top = (1.0 - (y0f + hf)) * self.height  # bottom-left origin -> top-left px
            bottom = (1.0 - y0f) * self.height
            if i == self.active_panel_index:
                draw_list.add_rect(left, top, right, bottom, 0xFFDD9933, 0.0, 0, 2.0)
            else:
                draw_list.add_rect(left, top, right, bottom, 0x55888888, 0.0, 0, 1.0)

    def _draw_marquee_box(self) -> None:
        """Rubber band for the Shift+Drag marquee selection.

        Both corners are in logical window units, which is what the ImGui draw
        list expects. The rect pick DPR-scales the same two corners; if these two
        ever disagree the marquee selects a different region than it drew.
        """
        px, py = self.interaction.press_mouse
        mx, my = self.interaction.last_mouse

        draw_list = self.hud.get_draw_list()
        if draw_list:
            # Packed ABGR literals, as in _draw_zoom_box. Cyan, to read as
            # distinct from the right-drag box-zoom rectangle.
            draw_list.add_rect_filled(px, py, mx, my, 0x40CCAA33)
            draw_list.add_rect(px, py, mx, my, 0xFFCCAA33)

    def _run_marquee_pick(self, mx: float, my: float) -> None:
        """Resolve a Shift+Drag marquee into the element selection.

        (mx, my) is the release position in logical window units, matching
        `interaction.press_mouse`. The picking target is in framebuffer pixels,
        so both corners are DPR-scaled here; `pick_rect_readback` owns the y-flip.
        """
        px, py = self.interaction.press_mouse
        mvp = self.camera_controller.mvp(self.width, self.height)
        window = self.camera_controller.world_window(self.width, self.height)

        dpr_x = self.fb_width / max(self.width, 1)
        dpr_y = self.fb_height / max(self.height, 1)

        # The pick target must hold the CURRENT view before it is read back --
        # the deferred pass may never have run for this view, or may hold an
        # older one. Same order as _run_picking_pass: draw, then read.
        self.picking.draw_pick_scene(self.scene, self.exact_renderer.buffers, mvp, window)
        hits = self.picking.pick_rect_readback(
            px * dpr_x, py * dpr_y, mx * dpr_x, my * dpr_y, self.scene
        )

        if self.interaction.selection_additive:
            self.interaction.selection.update(hits)
        else:
            self.interaction.selection.set_all(hits)

        # Only adopt a marquee's layer as "the" selected layer when it is
        # unambiguous; a multi-layer marquee must not pick one arbitrarily.
        if len(hits) == 1:
            self.interaction.selected_layer_id = next(iter(hits))

        self.frame.dirty_scene = True
        self.frame.dirty_ui = True

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    def _on_resize(self, window: Any, w: int, h: int) -> None:
        self.width = max(1, int(w))
        self.height = max(1, int(h))
        self.frame.dirty_scene = True
        self.frame.dirty_pick = True

    def _on_fb_resize(self, window: Any, w: int, h: int) -> None:
        self.fb_width = max(1, int(w))
        self.fb_height = max(1, int(h))
        glViewport(0, 0, self.fb_width, self.fb_height)
        self.interaction_renderer.rebuild_cache_target(self.fb_width, self.fb_height)
        # rebuild_cache_target() hands back a freshly allocated, unpainted texture, so
        # the window the old one was captured through no longer describes anything.
        # Leaving it set would let the next interaction reproject blank pixels.
        self.cache.capture_window = None
        self.cache.refresh_requested = True
        self.density_renderer.rebuild_target(self.fb_width, self.fb_height)
        self.picking.rebuild_target(self.fb_width, self.fb_height)
        self.effects.on_resize()
        self.frame.dirty_scene = True
        self.frame.dirty_pick = True

    def _on_scroll(self, window: Any, dx: float, dy: float) -> None:
        self.hud.on_scroll(window, dx, dy)
        if self.hud.wants_mouse():
            return

        # Zoom targets whichever panel the cursor is over (it becomes the current axes).
        mx, my = glfw.get_cursor_pos(self.window)
        self._activate_panel_under_cursor(mx, my)

        if not self.cache.active:
            self.cache.active = True
        self.cache.refresh_requested = True
        self.cache.release_deadline = glfw.get_time() + 0.20

        if self.is_3d_scene():
            factor = 0.9 if dy > 0 else 1.0 / 0.9
            self._zoom_3d(factor)
        else:
            factor = (
                self.options.zoom_scroll_factor if dy > 0 else 1.0 / self.options.zoom_scroll_factor
            )
            lx, ly, pw, ph = self._local_cursor(mx, my)
            self.camera_controller.apply_zoom_at_cursor(factor, lx, ly, pw, ph)
            self._sync_shared_axes()

        self.frame.dirty_scene = True
        self.frame.dirty_pick = True

    def _on_mouse_button(self, window: Any, button: int, action: int, mods: int) -> None:
        self.frame.dirty_ui = True
        self.hud.on_mouse_button(window, button, action, mods)
        if self.hud.wants_mouse():
            return

        mx, my = glfw.get_cursor_pos(self.window)

        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                # The gesture belongs to the panel it starts in: make that the active panel so
                # every self.interaction.* / self.camera_controller.* below reads its state. It
                # stays active for the whole drag (cursor-move never re-activates), so the drag
                # keeps operating on this panel even if the cursor wanders into a neighbour.
                self._activate_panel_under_cursor(mx, my)

                # 1. Picking Pass (Shift + Click)
                if (mods & glfw.MOD_SHIFT) or self.interaction.shift_down:
                    self.interaction.last_mouse = (mx, my)
                    # Alt = add to / toggle within the selection instead of
                    # replacing it. Read once here so the click and the marquee
                    # that may follow agree on the modifier.
                    self.interaction.selection_additive = bool(mods & glfw.MOD_ALT)
                    self.frame.dirty_scene = True
                    if mods & glfw.MOD_CONTROL:
                        # Ctrl owns the gesture (move/ratio) and never edits the
                        # selection, so no marquee can follow: pick at press,
                        # exactly as before.
                        self.frame.dirty_pick = True
                        self.interaction.explicit_pick_requested = True
                    else:
                        # This may still turn into a marquee. Resolve on RELEASE
                        # so a Shift+Drag does not first apply a stray single
                        # pick -- which an additive marquee would then keep.
                        self.interaction.pick_press_requested = True

                # 2. Start Drag State
                self.interaction.drag_active = True
                self.interaction.drag_confirmed = False
                self.interaction.drag_start_translation = None
                self.interaction.press_mouse = (mx, my)
                self.interaction.last_mouse = (mx, my)
                self.interaction.drag_start_world = self._panel_screen_to_world(mx, my)

                # 3. Determine Drag Mode
                #
                # 3D takes the modifiers first. Ctrl+Drag's 2D meanings (move a layer,
                # scale the axis ratio) have no 3D counterpart -- a 3D layer has no
                # 2-vector translation and there is no per-axis zoom to skew -- so in a 3D
                # scene the modifier is free for the gesture every 3D tool binds to it.
                # Shift is left alone so Shift+Click picking keeps behaving the same way
                # in both modes.
                if self.is_3d_scene() and not (mods & glfw.MOD_SHIFT):
                    self.interaction.drag_start_azim = self.camera3d.azim
                    self.interaction.drag_start_elev = self.camera3d.elev
                    self.interaction.drag_start_roll = self.camera3d.roll
                    if mods & glfw.MOD_CONTROL:
                        self.interaction.drag_mode = "pan3d"
                    elif mods & glfw.MOD_ALT:
                        self.interaction.drag_mode = "roll3d"
                    else:
                        self.interaction.drag_mode = "rotate3d"
                elif mods & glfw.MOD_CONTROL:
                    # For professional UX, resolve the mode ONCE at the start of the drag
                    self._run_picking_pass()

                    if self.interaction.selected_layer_id is not None:
                        self.interaction.drag_mode = "move"
                        layer = next(
                            (
                                candidate
                                for candidate in self.scene.layers
                                if candidate.layer_id == self.interaction.selected_layer_id
                            ),
                            None,
                        )
                        if layer:
                            self.interaction.drag_start_translation = layer.translation
                    else:
                        # Scientific ratio scaling mode
                        self.interaction.drag_mode = "ratio"
                        self.interaction.drag_start_zoom_x = self.camera.zoom_x
                        self.interaction.drag_start_zoom_y = self.camera.zoom_y
                elif mods & glfw.MOD_SHIFT:
                    # BEHAVIOUR CHANGE: Shift+Drag used to translate the selected
                    # layer, duplicating Ctrl+Drag above. It is now the marquee:
                    # Shift+Click picks one element, Shift+Drag picks many.
                    # Ctrl+Drag remains the way to move a layer.
                    self.interaction.drag_mode = "marquee"
                else:
                    self.interaction.drag_mode = "pan"

            elif action == glfw.RELEASE:
                # A confirmed marquee drag resolves on release. An unconfirmed one
                # was a Shift+Click, already handled by the deferred single pick.
                was_marquee = (
                    self.interaction.drag_active
                    and self.interaction.drag_mode == "marquee"
                    and self.interaction.drag_confirmed
                )
                self.interaction.drag_active = False
                if was_marquee:
                    self._run_marquee_pick(mx, my)
                elif self.interaction.pick_press_requested:
                    # A Shift+Click that never became a drag: pick one element now.
                    self.interaction.last_mouse = (mx, my)
                    self.frame.dirty_pick = True
                    self.interaction.explicit_pick_requested = True
                    self.interaction.selection_pick_requested = True
                self.interaction.pick_press_requested = False
                if self.cache.active:
                    self.cache.release_deadline = glfw.get_time() + 0.05
                self.frame.dirty_scene = True

        elif button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                self._activate_panel_under_cursor(mx, my)
                self.interaction.right_drag_active = True
                self.interaction.right_press_mouse = (mx, my)
                self.interaction.last_mouse = (mx, my)
            elif action == glfw.RELEASE:
                # Zoom-to-rectangle is a 2D verb: a screen rectangle names a world
                # rectangle only when the projection is axis-aligned. In 3D the right drag
                # has already been panning the camera frame by frame (see ``_on_cursor``),
                # so there is nothing to resolve on release.
                if self.interaction.right_drag_active and not self.is_3d_scene():
                    px, py = self.interaction.right_press_mouse
                    if abs(mx - px) > 5 and abs(my - py) > 5:
                        w0, h0 = self._panel_screen_to_world(px, py)
                        w1, h1 = self._panel_screen_to_world(mx, my)
                        self.set_view(
                            xlim=(min(w0, w1), max(w0, w1)), ylim=(min(h0, h1), max(h0, h1))
                        )
                        self._sync_shared_axes()
                self.interaction.right_drag_active = False
                self.frame.dirty_scene = True

        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            # Middle-drag pans in 3D — the binding every DCC and CAD tool uses, and the one
            # that needs no modifier. Nothing in 2D: the left button already pans there,
            # and inventing a second way would only add a gesture to document.
            if action == glfw.PRESS:
                self._activate_panel_under_cursor(mx, my)
                self.interaction.middle_drag_active = True
                self.interaction.last_mouse = (mx, my)
            elif action == glfw.RELEASE:
                self.interaction.middle_drag_active = False
                self.frame.dirty_scene = True

    def _on_cursor(self, window: Any, x: float, y: float) -> None:
        if (x, y) == self.interaction.last_mouse:
            return

        # Coordinate read-out: while dragging it tracks the drag's panel (the active one);
        # otherwise the panel the cursor is hovering, so the HUD shows the right axes' units.
        if self.interaction.drag_active:
            self.mouse_world = self._panel_screen_to_world(x, y)
        else:
            hover = self._panel_index_at(x, y)
            if hover is not None:
                p = self.panels[hover]
                hlx, hly = p.local_cursor(x, y, self.width, self.height)
                hpw, hph = p.pixel_size(self.width, self.height)
                self.mouse_world = p.camera_controller.screen_to_world(hlx, hly, hpw, hph)
            else:
                self.mouse_world = self._panel_screen_to_world(x, y)
        self.frame.dirty_ui = True

        if self.hud.wants_mouse():
            self.interaction.last_mouse = (x, y)
            return

        if self.interaction.drag_active:
            px, py = self.interaction.press_mouse
            dist2 = (x - px) ** 2 + (y - py) ** 2
            if not self.interaction.drag_confirmed and dist2 > self.options.drag_threshold_px**2:
                self.interaction.drag_confirmed = True
                self.cache.active = True
                self.cache.refresh_requested = True
                self.cache.release_deadline = glfw.get_time() + 0.20

            if (
                self.interaction.drag_mode == "move"
                and self.interaction.selected_layer_id is not None
            ):
                # MOVE MODE: Translate the layer
                layer = next(
                    (
                        candidate
                        for candidate in self.scene.layers
                        if candidate.layer_id == self.interaction.selected_layer_id
                    ),
                    None,
                )
                if layer:
                    if self.interaction.drag_start_translation is None:
                        self.interaction.drag_start_translation = layer.translation
                        self.interaction.drag_start_world = self._panel_screen_to_world(x, y)

                    curr_world = self._panel_screen_to_world(x, y)
                    start_world = self.interaction.drag_start_world
                    start_trans = self.interaction.drag_start_translation

                    dx = curr_world[0] - start_world[0]
                    dy = curr_world[1] - start_world[1]
                    layer.translation = (start_trans[0] + dx, start_trans[1] + dy)
                    self.cache.refresh_requested = True
            elif self.interaction.drag_mode == "ratio":
                # RATIO MODE: Exponential Anisotropic Scaling
                dx = x - px
                dy = y - py

                # Base-2 Exponential Law (100px = factor of 2.0 change)
                sensitivity = 0.01
                self.camera.zoom_x = self.interaction.drag_start_zoom_x * (
                    2.0 ** (dx * sensitivity)
                )
                self.camera.zoom_y = self.interaction.drag_start_zoom_y * (
                    2.0 ** (-dy * sensitivity)
                )

                # Clamp to safe camera limits
                self.camera.zoom_x = float(
                    np.clip(self.camera.zoom_x, self.camera.zoom_min, self.camera.zoom_max)
                )
                self.camera.zoom_y = float(
                    np.clip(self.camera.zoom_y, self.camera.zoom_min, self.camera.zoom_max)
                )
                self._sync_shared_axes()
                self.cache.refresh_requested = True
            elif self.interaction.drag_mode == "marquee":
                # MARQUEE MODE: the rubber band tracks last_mouse (updated below)
                # and the rect pick runs on release. Deliberately no camera or
                # layer mutation -- the view must not move under the band.
                pass
            elif self.interaction.drag_mode == "rotate3d":
                # ROTATE3D MODE: horizontal drag → azimuth, vertical drag → elevation
                dx = x - px
                dy = y - py
                # Normalize by viewport size so rotation feels similar on laptop
                # and external displays, with a gentle cubic easing near zero.
                span = max(float(min(self.width, self.height)), 1.0)
                nx = float(np.clip(dx / span, -1.0, 1.0))
                ny = float(np.clip(dy / span, -1.0, 1.0))
                eased_x = nx * (0.55 + 0.45 * abs(nx))
                eased_y = ny * (0.55 + 0.45 * abs(ny))
                new_azim = self.interaction.drag_start_azim - eased_x * 180.0
                # ±90 rather than the old ±89: the pole is a legal orientation now
                # (`Camera3D.basis` derives an up vector there), so a drag can reach a
                # true top-down view instead of stopping one degree short of it.
                new_elev = float(
                    np.clip(self.interaction.drag_start_elev + eased_y * 120.0, -90.0, 90.0)
                )
                self.set_3d_view(azim=new_azim, elev=new_elev)
                self.cache.refresh_requested = True
            elif self.interaction.drag_mode == "pan3d":
                # PAN3D MODE: slide the camera's target in its own screen plane. The delta
                # is per-frame (last_mouse, not press_mouse) because the pan accumulates on
                # the camera rather than being recomputed from a stored start value.
                lx, ly = self.interaction.last_mouse
                self.pan_3d(x - lx, y - ly)
                self.cache.refresh_requested = True
            elif self.interaction.drag_mode == "roll3d":
                # ROLL3D MODE: horizontal drag spins the view about the line of sight.
                span = max(float(self.width), 1.0)
                self.set_3d_view(roll=self.interaction.drag_start_roll + (x - px) / span * 360.0)
                self.cache.refresh_requested = True
            else:
                # PAN MODE: Translate the camera
                lx, ly = self.interaction.last_mouse
                wx0, wy0 = self._panel_screen_to_world(lx, ly)
                wx1, wy1 = self._panel_screen_to_world(x, y)
                self.camera.cx -= wx1 - wx0
                self.camera.cy -= wy1 - wy0
                self._sync_shared_axes()

            self.interaction.last_mouse = (x, y)
            self.frame.dirty_scene = True
            if self.cache.active and self._cache_needs_refresh():
                self.cache.refresh_requested = True
        elif self.interaction.right_drag_active or self.interaction.middle_drag_active:
            # In 3D both of these pan the camera; in 2D the right drag is still drawing a
            # zoom rectangle that resolves on release, and the middle button is unbound.
            if self.is_3d_scene():
                lx, ly = self.interaction.last_mouse
                self.pan_3d(x - lx, y - ly)
                self.frame.dirty_scene = True
                self.cache.refresh_requested = True
            self.interaction.last_mouse = (x, y)
            self.frame.dirty_ui = True
        else:
            self.interaction.last_mouse = (x, y)

    def _run_picking_pass(self, update_selection: bool = False) -> None:
        """Pick the single element under the cursor.

        ``update_selection`` is opt-in and only true for an explicit Shift+Click,
        so that hover/scroll/resize re-picks and the Ctrl+Drag mode resolution do
        not silently rewrite the user's multi-selection.
        """
        if not self.interaction.last_mouse:
            return

        mx, my = self.interaction.last_mouse
        mvp = self.camera_controller.mvp(self.width, self.height)
        window = self.camera_controller.world_window(self.width, self.height)

        # Scale to framebuffer (pixel) coordinates for Retina / High-DPI displays.
        # GLFW cursor positions are in logical window units; the picking FBO is in pixels.
        dpr_x = self.fb_width / max(self.width, 1)
        dpr_y = self.fb_height / max(self.height, 1)
        px = mx * dpr_x
        py = my * dpr_y

        # 1. Render scene to picking buffer
        self.picking.draw_pick_scene(self.scene, self.exact_renderer.buffers, mvp, window)

        # 2. Read back hit result at cursor (in pixel coords)
        hit = self.picking.pick_readback(px, py, self.scene)

        if hit:
            display_world = self.mouse_world_display()
            self.picked_info = {
                "type": hit["type"],
                "layer_id": hit["layer_id"],
                "element_idx": hit["element_idx"],
                "layer": hit["layer"],
                "x": display_world[0] if display_world else 0.0,
                "y": display_world[1] if display_world else 0.0,
            }
            # Update interaction selection
            self.interaction.selected_layer_id = hit["layer_id"]

            if update_selection:
                # Shift+Click = pick one. Shift+Alt+Click toggles it into the
                # existing multi-selection instead of replacing it.
                if self.interaction.selection_additive:
                    self.interaction.selection.toggle(hit["layer_id"], hit["element_idx"])
                else:
                    self.interaction.selection.set(hit["layer_id"], hit["element_idx"])

            # Specific logic for lines to get exact Y
            if hit["type"] == "line_family" and hit["layer"].ab is not None:
                ei = hit["element_idx"]
                layer = hit["layer"]
                tx, ty = layer.translation
                wx = self.picked_info["x"]
                # Line Eq is local: y_local = a * (wx - tx) + b
                # Then y_global = y_local + ty
                y_local = layer.ab[ei, 0] * (wx - tx) + layer.ab[ei, 1]
                self.picked_info["y"] = y_local + ty
        else:
            self.picked_info = None
            # We don't clear selected_layer_id on "miss" to allow dragging it
            # after selection even if the cursor moves off.
            if update_selection and not self.interaction.selection_additive:
                # An explicit Shift+Click on empty space deselects.
                self.interaction.selection.clear()

    def get_xlim(self) -> Tuple[float, float]:
        l, r, _, _ = self.camera_controller.world_window(self.width, self.height)
        return l, r

    def get_ylim(self) -> Tuple[float, float]:
        _, _, b, t = self.camera_controller.world_window(self.width, self.height)
        return b, t

    def savefig(self, filename: str, scale: float = 2.0) -> None:
        """
        Public API for saving high-resolution figures.
        """
        self.export.savefig(filename, scale=scale)

    def _create_rgba_fbo(self, width: int, height: int) -> Tuple[int, int, int]:
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)

        # The 3D geometry pass turns GL_DEPTH_TEST on (`renderers/geometry3d.py`), so
        # without a depth attachment every 3D scene resolves by draw order instead of by
        # distance: far bars paint over near ones and the render collapses inside-out.
        rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glDeleteFramebuffers(1, [fbo])
            glDeleteTextures(1, [tex])
            glDeleteRenderbuffers(1, [rbo])
            raise RuntimeError("Failed to create RGBA export framebuffer")
        return fbo, tex, rbo

    def capture_snapshot(
        self,
        scale: float = 1.0,
        transparent: bool = True,
        include_axes: bool = False,
        include_postfx: bool = True,
        preserve_screen_space_styles: bool = True,
    ) -> "GLPlotSnapshot":
        """
        Level 1 API: Capture the current viewport as a raster image + extent.
        Ensures perfect GL state restoration.

        The image maps exactly onto ``extent``: with ``include_axes`` off the projection
        runs on zero gutters, so the raster is pure plot content and the receiving library
        can frame it with its own axes. With ``include_axes`` on, GLPlot draws its own axes
        into the stock gutters and ``extent`` widens to stay true to the pixels.
        """
        from .utils.mpl_bridge import GLPlotSnapshot

        target_w = max(1, int(round(self.fb_width * scale)))
        target_h = max(1, int(round(self.fb_height * scale)))

        # Capture state to restore
        prev_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
        prev_viewport = glGetIntegerv(GL_VIEWPORT)
        prev_clear_col = glGetFloatv(GL_COLOR_CLEAR_VALUE)

        fbo, tex, rbo = self._create_rgba_fbo(target_w, target_h)

        try:
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            glViewport(0, 0, target_w, target_h)

            if transparent:
                glClearColor(0.0, 0.0, 0.0, 0.0)
            else:
                c = self.options.visual.background_color
                glClearColor(c[0], c[1], c[2], 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Style scaling for high-res
            style_scale = scale if preserve_screen_space_styles else 1.0

            # A snapshot is handed to a library that draws its own axes, so the gutters the
            # live window reserves for GLPlot's tick labels would only land as empty bands
            # inside the receiver's frame. Zeroing them makes the raster span the world
            # window exactly, which is what `extent` promises. The whole projection-and-draw
            # span runs inside this, so `mvp()` and the density resolve pass's UV clip box
            # (`renderers/density.py`) read the same gutters -- they must, or the density
            # image would clip against a box the geometry was never projected into.
            gutters = DEFAULT_AXIS_MARGINS if include_axes else (0.0, 0.0, 0.0, 0.0)
            with override_axis_margins(self.options, gutters):
                # At target resolution: the gutters are pixel amounts, so measuring them
                # against the logical window would misplace them in a scaled-up raster.
                window = self.camera_controller.world_window(target_w, target_h)
                mvp = self.camera_controller.mvp(target_w, target_h)

                # The world coords of the image's own corners. `screen_to_world` is the
                # exact inverse of `mvp`, so this stays true to the pixels whatever the
                # gutters are -- and collapses to `window` when they are zero.
                xmin, ymin = self.camera_controller.screen_to_world(
                    0.0, float(target_h), target_w, target_h
                )
                xmax, ymax = self.camera_controller.screen_to_world(
                    float(target_w), 0.0, target_w, target_h
                )
                extent = (float(xmin), float(xmax), float(ymin), float(ymax))

                ndc_scale, ndc_offset = self._get_ndc_transform(window)
                prob = self._compute_lod_keep_prob()
                alpha = self._get_adaptive_alpha(self.scene.lines.count)

                ctx = RenderContext(
                    mvp=mvp,
                    window_world=window,
                    ndc_scale=ndc_scale,
                    ndc_offset=ndc_offset,
                    width_px=target_w,
                    height_px=target_h,
                    fb_width=target_w,
                    fb_height=target_h,
                    dpr=style_scale * (self.fb_width / max(self.width, 1)),
                    mode=self.policy.runtime.current_mode,
                    global_alpha=alpha,
                    lod_keep_prob=prob,
                    is_density=self.display_density,
                    time=time.perf_counter(),
                )

                self._apply_blending_policy()
                layers = self._get_all_layers()

                if include_axes:
                    self.axis_manager.update(ctx)
                    self.renderer_manager.draw_axes(self.axis_manager, ctx)

                # Pass to modular managers
                if self._density_active():
                    self.renderer_manager.draw_density(
                        layers, ctx, target_fbo=fbo, target_size=(target_w, target_h)
                    )
                else:
                    self.renderer_manager.draw_exact(layers, ctx)

                # Text overlay
                self.renderer_manager.renderers["text"].draw_all(layers, ctx)

            glFinish()
            glReadBuffer(GL_COLOR_ATTACHMENT0)
            glPixelStorei(GL_PACK_ALIGNMENT, 1)
            raw = glReadPixels(0, 0, target_w, target_h, GL_RGBA, GL_UNSIGNED_BYTE)
            rgba = np.frombuffer(raw, dtype=np.uint8).reshape((target_h, target_w, 4))
            # glReadPixels hands back rows bottom-up; flip to the top-row-first order every
            # image library expects (`plt.imsave`, `imshow`'s default origin='upper').
            rgba = np.ascontiguousarray(np.flipud(rgba))

        finally:
            glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
            glViewport(*prev_viewport)
            glClearColor(*prev_clear_col)
            glDeleteFramebuffers(1, [fbo])
            glDeleteTextures(1, [tex])
            glDeleteRenderbuffers(1, [rbo])

        return GLPlotSnapshot(
            rgba=rgba,
            extent=extent,
            xlim=(extent[0], extent[1]),
            ylim=(extent[2], extent[3]),
            width_px=target_w,
            height_px=target_h,
            transparent=transparent,
            projected_3d=self._is_pure_3d_scene(),
        )

    def to_matplotlib(
        self, ax: Optional[Any] = None, mpl_kwargs: dict = {}, **kwargs: Any
    ) -> tuple[Any, Any, Any]:
        """Level 2 API: Render and embed directly into Matplotlib."""
        from .utils.mpl_bridge import snapshot_to_matplotlib

        snap = self.capture_snapshot(**kwargs)
        return snapshot_to_matplotlib(snap, ax=ax, **mpl_kwargs)

    def set_matplotlib_transfer_target(
        self, ax: Optional[Any] = None, callback: Optional[Any] = None
    ) -> None:
        """Level 3 API Setup: Redirect 'M' key transfers."""
        self._mpl_transfer_ax = ax
        self._mpl_transfer_callback = callback

    def transfer_to_matplotlib_default(self) -> None:
        """Default action for Key 'M'.

        Programmatic targets registered through ``set_matplotlib_transfer_target``
        keep the in-process behaviour: the caller owns the figure and drives
        matplotlib itself.

        With no target registered, the snapshot is handed to a separate viewer
        process instead of being drawn in-process. An in-process figure cannot
        survive the GL loop on macOS: ``glfw.poll_events`` drains the shared
        NSAutoreleasePool that owns matplotlib's window, so the figure is
        destroyed about a second after it opens. A child process owns its own
        event loop, stays interactive, and outlives this one. It is spawned
        non-blocking, so the GL loop is never stalled.
        """
        if hasattr(self, "_mpl_transfer_callback") and self._mpl_transfer_callback:
            snap = self.capture_snapshot(scale=2.0)
            self._mpl_transfer_callback(snap)
            return

        ax = getattr(self, "_mpl_transfer_ax", None)
        if ax is not None:
            import matplotlib.pyplot as plt

            fig, ax, artist = self.to_matplotlib(ax=ax, scale=2.0)
            plt.show(block=False)
            fig.canvas.draw_idle()
            return

        from .utils.mpl_process import launch_snapshot_viewer

        snap = self.capture_snapshot(scale=2.0)
        launch_snapshot_viewer(
            snap,
            xlabel=getattr(self, "xlabel", None),
            ylabel=getattr(self, "ylabel", None),
            title=getattr(self, "title", None),
        )

    def toggle_line_colormap(self) -> None:
        self.options.line_colormap_enabled = not self.options.line_colormap_enabled
        self.frame.dirty_scene = True

    def _on_key(self, window: Any, key: int, sc: int, action: int, mods: int) -> None:
        self.frame.dirty_ui = True
        self.hud.on_key(window, key, sc, action, mods)
        if self.hud.wants_keyboard():
            return

        if action in (glfw.PRESS, glfw.REPEAT):
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(self.window, True)

            elif key in (glfw.KEY_R, glfw.KEY_HOME):
                self.reset_view()

            elif key == glfw.KEY_D and action == glfw.PRESS:
                self.toggle_density()

            elif key == glfw.KEY_C and action == glfw.PRESS:
                self.toggle_line_colormap()

            elif key == glfw.KEY_F3 and action == glfw.PRESS:
                self.hud.state.show_profiler = not self.hud.state.show_profiler
                if self.hud.state.show_profiler:
                    self.options.enable_hud = True
                self.frame.dirty_scene = True
                self.frame.dirty_ui = True

            # --- Visual Parameters (Arrows) ---
            if key == glfw.KEY_UP:
                if self.display_density:
                    self.options.density_gain *= 1.2
                else:
                    self.options.default_global_alpha = min(
                        1.0, self.options.default_global_alpha * 1.2
                    )
                self.frame.dirty_scene = True
                self.frame.dirty_ui = True

            elif key == glfw.KEY_DOWN:
                if self.display_density:
                    self.options.density_gain /= 1.2
                else:
                    self.options.default_global_alpha = max(
                        0.001, self.options.default_global_alpha / 1.2
                    )
                self.frame.dirty_scene = True
                self.frame.dirty_ui = True

            elif key == glfw.KEY_LEFT:
                self.previous_density_scheme()

            elif key == glfw.KEY_RIGHT:
                self.next_density_scheme()

            # --- Global Density / Style Controls (PgUp/PgDn and Brackets) ---

            # --- Zoom ---
            elif key == glfw.KEY_EQUAL or key == glfw.KEY_KP_ADD:
                self.camera_controller.apply_zoom_at_cursor(
                    self.options.zoom_scroll_factor,
                    self.width * 0.5,
                    self.height * 0.5,
                    self.width,
                    self.height,
                )

            elif key == glfw.KEY_MINUS or key == glfw.KEY_KP_SUBTRACT:
                self.camera_controller.apply_zoom_at_cursor(
                    1.0 / self.options.zoom_scroll_factor,
                    self.width * 0.5,
                    self.height * 0.5,
                    self.width,
                    self.height,
                )

            elif key == glfw.KEY_B and action == glfw.PRESS:
                self.cycle_blending_mode()

            elif key == glfw.KEY_BACKSLASH and action == glfw.PRESS:
                self.options.enable_auto_alpha = not self.options.enable_auto_alpha
                self.frame.dirty_scene = True

            elif key == glfw.KEY_LEFT_BRACKET and action in (glfw.PRESS, glfw.REPEAT):
                self.decrease_density_gain()
                self.frame.dirty_scene = True

            elif key == glfw.KEY_RIGHT_BRACKET and action in (glfw.PRESS, glfw.REPEAT):
                self.increase_density_gain()
                self.frame.dirty_scene = True

            elif key == glfw.KEY_H and action == glfw.PRESS:
                self.set_hud_enabled(not self.options.enable_hud)

            elif key == glfw.KEY_S and action == glfw.PRESS:
                self.savefig(f"plot_{int(time.time())}.png", scale=self.options.export_scale)

            elif key == glfw.KEY_M and action == glfw.PRESS:
                self.transfer_to_matplotlib_default()

            self.frame.dirty_scene = True

        if action == glfw.PRESS:
            if key in (glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT):
                self.interaction.shift_down = True
        elif action == glfw.RELEASE:
            if key in (glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT):
                self.interaction.shift_down = False

    def _on_char(self, window: Any, char: int) -> None:
        self.frame.dirty_ui = True
        self.hud.on_char(window, char)
