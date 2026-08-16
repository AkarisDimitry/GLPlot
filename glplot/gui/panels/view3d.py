"""The 3D view panel: dimensionality, camera, projection, axes and lighting.

Every control here writes to the *panel's* :class:`~glplot.core.camera3d.Camera3D` and
:class:`~glplot.core.camera3d.Axes3DOptions` through a queued command, never directly —
CONTRACT §1.1 applies to the camera exactly as it does to the scene, because
``engine.set_3d_view`` rebuilds the axis-box geometry as a side effect and that is scene
mutation.

Why a panel and not more rows in Style
--------------------------------------
The Style panel already had a "3D camera" section, and it was three sliders (``elev``,
``azim``, ``fov``) hidden under a per-layer disclosure, because that is what the camera
*was*: three floats copied onto each layer's metadata. With a real camera there are now
five distinct groups of decisions — where you are, how you project, what the box looks
like, how the axes are scaled, and how the scene is lit — and none of them is per-layer.
Folding a dozen more rows into the layer editor would bury the layer controls and still
mislabel the camera as a property of a layer.

The mode switch is here for the same reason: "is this figure 2D or 3D" is the first
question this panel answers, and every other control in it is conditional on the answer.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from ...core.camera3d import (
    MAX_ELEV,
    MAX_FOV,
    MIN_ELEV,
    MIN_FOV,
    STANDARD_VIEW_LABELS,
    STANDARD_VIEWS,
)
from .. import widgets
from ..icons import icon_button
from .base import Panel

try:
    import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - exercised only on GL-less systems
    IMGUI_AVAILABLE = False
    imgui = None


#: The presets given a dedicated icon button, in toolbar order, as
#: ``(view key, icon, tooltip)``. The remaining entries of :data:`STANDARD_VIEWS` are
#: reachable from the combo below the row — six buttons is a toolbar, twelve is a wall.
_VIEW_BUTTONS: Tuple[Tuple[str, str, str], ...] = (
    ("iso", "view_iso", "Isometric (default)"),
    ("top", "view_top", "Top — looking down +Z"),
    ("front", "view_front", "Front — looking along +Y"),
    ("right", "view_right", "Right — looking along −X"),
)

#: Box-aspect presets, as ``(label, value)``. ``None`` means raw data units.
_ASPECT_PRESETS: Tuple[Tuple[str, Optional[Tuple[float, float, float]]], ...] = (
    ("Data units", None),
    ("Cube (1:1:1)", (1.0, 1.0, 1.0)),
    ("Wide (2:2:1)", (2.0, 2.0, 1.0)),
    ("Tall (1:1:2)", (1.0, 1.0, 2.0)),
)

_HELP_MODE = (
    "3D pins this figure to a 3D view even while it is empty, so the 3D tools are "
    "available before there is any data. 'Auto' is the historical behaviour: the figure "
    "is 3D exactly when it holds a 3D layer."
)

_HELP_PROJECTION = (
    "Perspective converges — near things draw larger, which reads as depth. Orthographic "
    "keeps parallel lines parallel, so lengths stay comparable anywhere in the box; it is "
    "what you want for measuring, and for a top view that should read as a plan. The "
    "field of view still sets the framing in both, so switching between them does not "
    "jump."
)

_HELP_BOX_ASPECT = (
    "How the three axes are scaled against each other. 'Data units' draws the raw "
    "numbers, so an axis spanning a million dwarfs one spanning one. 'Cube' rescales each "
    "axis to the same on-screen length, which is what makes a plot of unrelated "
    "quantities readable. It changes the picture, not the data — the tick numbers stay "
    "true."
)

_HELP_SPIN = (
    "Degrees of azimuth added per second. Useful for reading a point cloud's structure, "
    "and for recording a turntable. 0 stops it."
)

_HELP_DISTANCE = (
    "Eye-to-target distance. 'Fit' recomputes it from the data, and it goes back to "
    "following the data automatically after 'Reset view'."
)

_HELP_GESTURES = (
    "Drag: orbit.  Ctrl+drag, right-drag or middle-drag: pan.  Alt+drag: roll.  "
    "Scroll: dolly in and out."
)


class View3DPanel(Panel):
    """Dimensionality, camera, projection, axis decoration and lighting for a 3D figure."""

    title = "View 3D"
    icon = "cube"
    default_open = False

    def __init__(self, ws: Any) -> None:
        super().__init__(ws)
        #: Last custom box-aspect triple, remembered so switching to a preset and back
        #: restores what the user dialled in rather than resetting to 1:1:1.
        self._custom_aspect: List[float] = [1.0, 1.0, 1.0]
        #: Whether the custom aspect drags are shown. Sticky: a custom triple that happens
        #: to equal a preset (1:1:1 *is* "Cube") must not make the drags disappear
        #: mid-edit, which is what deriving this from the value alone would do.
        self._show_custom_aspect: bool = False

    # -- state helpers ---------------------------------------------------------

    @property
    def camera(self) -> Any:
        """The active panel's 3D camera."""
        return self.plot.camera3d

    @property
    def axes(self) -> Any:
        """The active panel's 3D axis options."""
        return self.plot.axes3d

    def _apply_view(self) -> None:
        """Re-sync the scene after a direct camera write. Queued, never immediate."""
        plot = self.plot
        self.submit(plot.set_3d_view)

    def _set_camera(self, **fields: Any) -> None:
        """Queue a camera change through ``set_3d_view`` so the axis box follows it."""
        plot = self.plot

        def apply() -> None:
            plot.set_3d_view(**fields)

        self.submit(apply)

    def _set_axes(self, name: str, value: Any) -> None:
        """Queue an :class:`Axes3DOptions` change and rebuild the decoration."""
        plot = self.plot

        def apply() -> None:
            setattr(plot.axes3d, name, value)
            plot.set_3d_view()

        self.submit(apply)

    def _set_box_aspect(self, aspect: Optional[Tuple[float, float, float]]) -> None:
        """Queue a box-aspect change. ``None`` restores raw data units."""
        plot = self.plot

        def apply() -> None:
            plot.camera3d.box_aspect = None if aspect is None else tuple(float(v) for v in aspect)
            plot.set_3d_view()

        self.submit(apply)

    # -- frame -----------------------------------------------------------------

    def draw(self) -> None:
        if not IMGUI_AVAILABLE:
            return

        self._draw_mode()
        if not self.plot.is_3d_scene():
            imgui.spacing()
            imgui.text_disabled("This figure is 2D.")
            imgui.text_wrapped(
                "Switch the mode to 3D above, or plot a 3D layer from the Data or "
                "Objects panel, to use the controls below."
            )
            return

        imgui.spacing()
        if widgets.section("Camera", default_open=True):
            self._draw_camera()
        if widgets.section("Projection", default_open=True):
            self._draw_projection()
        if widgets.section("Axes & grid", default_open=False):
            self._draw_axes_options()
        if widgets.section("Axis scaling", default_open=False):
            self._draw_box_aspect()
        if widgets.section("Lighting", default_open=False):
            self._draw_lighting()
        if widgets.section("Navigation", default_open=False):
            self._draw_gestures()

    # -- sections --------------------------------------------------------------

    def _draw_mode(self) -> None:
        """The 2D / 3D / Auto switch. The first question, so it sits above everything."""
        panel = self.plot.active_panel
        current = panel.ndim
        imgui.text("Mode")
        imgui.same_line()
        widgets.help_marker(_HELP_MODE)

        options = (("Auto", None), ("2D", 2), ("3D", 3))
        for index, (label, value) in enumerate(options):
            if index:
                imgui.same_line()
            if imgui.radio_button(f"{label}##ndim", current == value):
                self._set_ndim(value)

    def _set_ndim(self, ndim: Optional[int]) -> None:
        plot = self.plot

        def apply() -> None:
            plot.set_ndim(ndim)

        self.submit(apply)

    def _draw_camera(self) -> None:
        camera = self.camera

        # Preset row: the four orientations worth a button, then home and fit.
        for key, icon, tooltip in _VIEW_BUTTONS:
            if icon_button(f"##view_{key}", icon, tooltip=tooltip):
                self._set_preset(key)
            imgui.same_line()
        if icon_button("##view_home", "home", tooltip="Reset view (orientation, dolly and pan)"):
            self.submit(self.plot.reset_3d_view)
        imgui.same_line()
        if icon_button("##view_fit", "target", tooltip="Fit the data, keeping the orientation"):
            self.submit(self.plot.frame_3d_view)
        imgui.same_line()
        spinning = bool(camera.auto_spin)
        if icon_button("##view_spin", "spin", tooltip="Auto-rotate", active=spinning):
            self._toggle_spin()

        # The full preset list, for the eight orientations without a button.
        labels = [STANDARD_VIEW_LABELS.get(k, k) for k in STANDARD_VIEWS]
        changed, picked = widgets.enum_combo("Preset", self._current_view_label(), labels)
        if changed:
            for key in STANDARD_VIEWS:
                if STANDARD_VIEW_LABELS.get(key, key) == picked:
                    self._set_preset(key)
                    break

        imgui.spacing()
        changed, value = widgets.labeled_slider_float(
            "Elevation", camera.elev, MIN_ELEV, MAX_ELEV, fmt="%.1f deg"
        )
        if changed:
            self._set_camera(elev=value)
        changed, value = widgets.labeled_slider_float(
            "Azimuth", camera.azim, -180.0, 180.0, fmt="%.1f deg"
        )
        if changed:
            self._set_camera(azim=value)
        changed, value = widgets.labeled_slider_float(
            "Roll",
            camera.roll,
            -180.0,
            180.0,
            fmt="%.1f deg",
            help="Spin about the line of sight. Alt+drag does the same thing.",
        )
        if changed:
            self._set_camera(roll=value)

        # Distance: a drag rather than a slider because there is no meaningful upper
        # bound, and None ("fit the data") has to be representable as a state.
        distance = camera.distance
        if distance is None:
            imgui.text_disabled("Distance: auto (fits the data)")
            imgui.same_line()
            widgets.help_marker(_HELP_DISTANCE)
        else:
            changed, value = widgets.labeled_drag_float(
                "Distance",
                float(distance),
                speed=max(abs(distance) * 0.01, 1e-3),
                vmin=1e-3,
                vmax=1e6,
                help=_HELP_DISTANCE,
            )
            if changed:
                self._set_camera(distance=value)

        changed, value = widgets.labeled_slider_float(
            "Auto-rotate",
            float(camera.auto_spin),
            -180.0,
            180.0,
            fmt="%.0f deg/s",
            help=_HELP_SPIN,
        )
        if changed:
            self._set_spin(value)

    def _toggle_spin(self) -> None:
        """Turn auto-rotation on at a readable default speed, or off."""
        camera = self.camera
        self._set_spin(0.0 if camera.auto_spin else 24.0)

    def _set_spin(self, value: float) -> None:
        plot = self.plot

        def apply() -> None:
            plot.camera3d.auto_spin = float(value)
            # No `set_3d_view` here: spin is advanced by the frame loop, and re-syncing
            # every layer's metadata on a slider drag would rebuild the axis box 60x a
            # second for a value that changes nothing about the current frame.
            plot.frame.dirty_scene = True

        self.submit(apply)

    def _current_view_label(self) -> str:
        """The preset whose angles match the camera, or "Custom"."""
        camera = self.camera
        for key, (elev, azim, roll) in STANDARD_VIEWS.items():
            if (
                abs(camera.elev - elev) < 0.5
                and abs(((camera.azim - azim + 180.0) % 360.0) - 180.0) < 0.5
                and abs(camera.roll - roll) < 0.5
            ):
                return STANDARD_VIEW_LABELS.get(key, key)
        return "Custom"

    def _set_preset(self, key: str) -> None:
        plot = self.plot

        def apply() -> None:
            plot.set_3d_preset(key)

        self.submit(apply)

    def _draw_projection(self) -> None:
        camera = self.camera
        is_persp = camera.projection == "perspective"

        if icon_button("##proj_persp", "perspective", tooltip="Perspective", active=is_persp):
            self._set_camera(projection="perspective")
        imgui.same_line()
        if icon_button("##proj_ortho", "orthographic", tooltip="Orthographic", active=not is_persp):
            self._set_camera(projection="orthographic")
        imgui.same_line()
        imgui.text("Perspective" if is_persp else "Orthographic")
        imgui.same_line()
        widgets.help_marker(_HELP_PROJECTION)

        changed, value = widgets.labeled_slider_float(
            "Field of view",
            camera.fov,
            MIN_FOV,
            MAX_FOV,
            fmt="%.0f deg",
            help="Narrow is a telephoto look with little distortion; wide exaggerates "
            "depth. It sets the orthographic box height too, so the framing "
            "matches when you switch projection.",
        )
        if changed:
            self._set_camera(fov=value)

        changed, value = widgets.enum_combo(
            "Up axis",
            camera.up_axis.upper(),
            ("Z", "Y"),
            help="Z up is the convention in engineering and in matplotlib; Y up is the "
            "graphics one. It changes what 'elevation' and 'azimuth' rotate about.",
        )
        if changed:
            self._set_camera(up_axis=value.lower())

    def _draw_axes_options(self) -> None:
        axes = self.axes

        changed, value = imgui.checkbox("Show axes", bool(axes.show_axes))
        if changed:
            self._set_axes("show_axes", bool(value))
        if not axes.show_axes:
            imgui.text_disabled("The box, floor, grid, ticks and labels are all hidden.")
            return

        for label, name in (
            ("Bounding box", "show_box"),
            ("Floor", "show_floor"),
            ("Grid walls", "show_grid"),
            ("Tick marks", "show_ticks"),
            ("Tick numbers", "show_tick_labels"),
            ("Axis titles", "show_axis_labels"),
        ):
            changed, value = imgui.checkbox(label, bool(getattr(axes, name)))
            if changed:
                self._set_axes(name, bool(value))

        # 0 is a real setting here, not an empty one: it hands the count back to the
        # per-axis screen-length rule, which is the default and what most scenes want.
        changed, value = imgui.slider_int(
            "Ticks per axis", int(axes.tick_count), 0, 20, "%d" if axes.tick_count else "auto"
        )
        if changed:
            self._set_axes("tick_count", max(0, int(value)))

        changed, value = widgets.labeled_slider_float(
            "Box padding",
            float(axes.pad),
            0.0,
            0.5,
            fmt="%.2f",
            help="How far the box is grown beyond the data, as a fraction of each axis' "
            "extent. 0 puts the box exactly on the data.",
        )
        if changed:
            self._set_axes("pad", float(value))

        imgui.spacing()
        imgui.text("Axis titles")
        for label, name in (("X", "xlabel"), ("Y", "ylabel"), ("Z", "zlabel")):
            changed, value = imgui.input_text(f"{label}##axis3dlabel", str(getattr(axes, name)))
            if changed:
                self._set_axes(name, str(value))

    def _draw_box_aspect(self) -> None:
        camera = self.camera
        imgui.text("Axis scaling")
        imgui.same_line()
        widgets.help_marker(_HELP_BOX_ASPECT)

        labels = [label for label, _ in _ASPECT_PRESETS] + ["Custom"]
        current = "Custom" if self._show_custom_aspect else self._aspect_label(camera.box_aspect)
        changed, picked = widgets.enum_combo("Scaling", current, labels)
        if changed:
            self._show_custom_aspect = picked == "Custom"
            if picked == "Custom":
                self._set_box_aspect(tuple(self._custom_aspect))
            else:
                for label, value in _ASPECT_PRESETS:
                    if label == picked:
                        self._set_box_aspect(value)
                        break

        if self._show_custom_aspect:
            aspect = list(camera.box_aspect or self._custom_aspect)
            for index, name in enumerate("XYZ"):
                changed, value = widgets.labeled_drag_float(
                    f"{name}##aspect",
                    float(aspect[index]),
                    speed=0.02,
                    vmin=0.01,
                    vmax=100.0,
                    fmt="%.2f",
                )
                if changed:
                    aspect[index] = float(value)
                    self._custom_aspect = list(aspect)
                    self._set_box_aspect(tuple(aspect))

    def _aspect_label(self, aspect: Optional[Tuple[float, float, float]]) -> str:
        """Which preset ``aspect`` corresponds to, or "Custom"."""
        for label, value in _ASPECT_PRESETS:
            if value is None and aspect is None:
                return label
            if (
                value is not None
                and aspect is not None
                and all(abs(float(a) - float(b)) < 1e-6 for a, b in zip(aspect, value))
            ):
                return label
        return "Custom"

    def _draw_lighting(self) -> None:
        """Screen-space ambient occlusion — the scene's one shading control."""
        visual = getattr(self.plot.options, "visual", None)
        ssao = getattr(visual, "ssao", None)
        if ssao is None:
            imgui.text_disabled("This build has no SSAO options.")
            return

        changed, value = imgui.checkbox("Ambient occlusion", bool(ssao.enabled))
        if changed:
            self._set_ssao("enabled", bool(value))
        imgui.same_line()
        widgets.help_marker(
            "Darkens crevices and the far side of dense geometry, which is what gives a "
            "point cloud or a mesh readable depth. Costs nothing per point — it is a "
            "depth-based shading term in the 3D shader, not a separate pass."
        )
        changed, value = widgets.labeled_slider_float(
            "Strength", float(getattr(ssao, "strength", 0.45)), 0.0, 1.0, fmt="%.2f"
        )
        if changed:
            self._set_ssao("strength", float(value))

    def _set_ssao(self, name: str, value: Any) -> None:
        plot = self.plot

        def apply() -> None:
            ssao = plot.options.visual.ssao
            setattr(ssao, name, value)
            # SSAO uniforms are read at draw time, so no GPU re-upload is needed — but the
            # cached impostor has to be redrawn or the change is invisible until the next
            # interaction.
            plot.frame.dirty_scene = True
            plot.cache.refresh_requested = True

        self.submit(apply)

    def _draw_gestures(self) -> None:
        imgui.text_wrapped(_HELP_GESTURES)
        imgui.spacing()
        for gesture, meaning in (
            ("Drag", "Orbit around the target"),
            ("Ctrl + drag", "Pan (slide the target)"),
            ("Right / middle drag", "Pan"),
            ("Alt + drag", "Roll about the line of sight"),
            ("Scroll", "Dolly in / out"),
        ):
            imgui.text_disabled(f"{gesture:>20s}")
            imgui.same_line()
            imgui.text(meaning)
