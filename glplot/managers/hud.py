from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from .hud_state import HudController, HudState

if TYPE_CHECKING:
    from ..engine import GPULinePlot

try:
    from imgui_bundle import imgui
    from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

    IMGUI_AVAILABLE = True
except (ImportError, Exception):
    IMGUI_AVAILABLE = False
    imgui = None
    GlfwRenderer = None

logger = logging.getLogger(__name__)

#: Cached in ``_workspace`` when construction raised, so a broken panel import is not
#: retried (and re-logged) on every single frame. Distinct from None, which means
#: "not built yet".
_WORKSPACE_FAILED = object()


class HudManager:
    def __init__(self, plot: GPULinePlot) -> None:
        self.plot = plot
        self.options = plot.options
        self.state = HudState()
        self.controller = HudController(plot)
        self.imgui_ctx = None
        self.imgui_impl = None
        self._workspace: Optional[Any] = None

    @property
    def workspace(self) -> Optional[Any]:
        """The mathematical workstation, built on first access; None when unavailable.

        Lazy for two reasons: importing the panels pulls in imgui and every op module,
        which a HUD-less or GL-less engine must not pay for; and ``engine._main_loop``
        reads ``hud.workspace`` through ``getattr`` every frame to drain its command
        queue, so the attribute has to exist even when there is nothing behind it.

        ``enable_hud`` gates construction precisely because of that per-frame read: an
        engine with the HUD off must stay unaffected, and a property that built the whole
        panel set on first access would make the drain's ``getattr`` do exactly what it
        was written to avoid. Turning the HUD on later still builds it on demand.

        A failure to build is logged and cached as None — the engine keeps rendering with
        the classic HUD rather than dying on an import error in a panel.
        """
        if self._workspace is None and IMGUI_AVAILABLE and self.options.enable_hud:
            try:
                from ..gui.workspace import Workspace

                self._workspace = Workspace(self.plot, hud=self)
            except Exception:
                logger.exception("Workspace unavailable; falling back to the classic HUD.")
                self._workspace = _WORKSPACE_FAILED
        if self._workspace is _WORKSPACE_FAILED:
            return None
        return self._workspace

    def _scene_3d_stats(self) -> dict[str, Any]:
        layers = [layer for layer in self.plot.scene.layers if self._is_3d_layer(layer)]
        from ..core.camera3d import SYSTEM_3D_ARTISTS

        data_layers = [
            layer for layer in layers if layer.metadata.get("artist") not in SYSTEM_3D_ARTISTS
        ]
        vertices = sum(
            0 if getattr(layer, "vertices", None) is None else len(layer.vertices)
            for layer in data_layers
        )
        triangles = 0
        for layer in data_layers:
            if getattr(layer, "primitive", None) == "triangles":
                if getattr(layer, "indices", None) is not None:
                    triangles += len(layer.indices) // 3
                elif getattr(layer, "vertices", None) is not None:
                    triangles += len(layer.vertices) // 3
        return {
            "layers": len(data_layers),
            "vertices": int(vertices),
            "triangles": int(triangles),
            "has_3d": bool(data_layers),
        }

    @staticmethod
    def _is_3d_layer(layer: Any) -> bool:
        return getattr(layer, "layer_type", "").endswith("3d")

    @staticmethod
    def _layer_size_text(layer: Any) -> str:
        if hasattr(layer, "vertices") and layer.vertices is not None:
            return f"{len(layer.vertices):,} verts"
        if hasattr(layer, "pts") and layer.pts is not None:
            return f"{len(layer.pts):,} pts"
        if hasattr(layer, "ab") and layer.ab is not None:
            return f"{len(layer.ab):,} lines"
        return ""

    def initialize(self, window: Any) -> None:
        # opaque GLFW window handle (glfw._GLFWwindow*, no public stub type)
        if not IMGUI_AVAILABLE:
            return
        self.imgui_ctx = imgui.create_context()
        self.imgui_impl = GlfwRenderer(window, attach_callbacks=False)
        # Use a more professional dark theme
        style = imgui.get_style()
        imgui.style_colors_dark(style)
        style.window_rounding = 4.0
        style.frame_rounding = 3.0

    def process_inputs(self) -> None:
        if self.imgui_impl:
            self.imgui_impl.process_inputs()

    def on_scroll(self, window: Any, dx: float, dy: float) -> None:
        if self.imgui_impl:
            self.imgui_impl.scroll_callback(window, dx, dy)

    def on_cursor(self, window: Any, x: float, y: float) -> None:
        # GlfwRenderer.mouse_callback ignores its args and re-reads the live cursor
        # position itself (see its source) -- calling it on every cursor-move event is
        # the imgui-bundle idiom for feeding continuous mouse position, replacing the
        # per-frame glfw.get_cursor_pos() poll that pyimgui's process_inputs() used to do.
        if self.imgui_impl:
            self.imgui_impl.mouse_callback(window, x, y)

    def on_mouse_button(self, window: Any, button: int, action: int, mods: int) -> None:
        # imgui-bundle's GlfwRenderer no longer feeds button state from process_inputs()
        # (pyimgui's backend polled glfw.get_mouse_button() every frame; this one does
        # not) and its mouse_callback ignores button/action/mods entirely -- it now only
        # pushes cursor position. Button state must be pushed explicitly.
        if imgui is not None:
            imgui.get_io().add_mouse_button_event(button, action != 0)

    def on_key(self, window: Any, key: int, sc: int, action: int, mods: int) -> None:
        if self.imgui_impl:
            self.imgui_impl.keyboard_callback(window, key, sc, action, mods)

    def on_char(self, window: Any, char: int) -> None:
        if self.imgui_impl:
            self.imgui_impl.char_callback(window, char)

    def wants_mouse(self) -> bool:
        if not IMGUI_AVAILABLE or not self.imgui_impl:
            return False
        return imgui.get_io().want_capture_mouse

    def wants_keyboard(self) -> bool:
        if not IMGUI_AVAILABLE or not self.imgui_impl:
            return False
        return imgui.get_io().want_capture_keyboard

    def begin(self) -> None:
        if self.imgui_impl:
            imgui.new_frame()

    def update(self) -> None:
        """Main draw orchestration for all HUD components."""
        if not self.imgui_impl or not self.options.enable_hud:
            return

        # SYNC: selected_layer_id between engine and HUD
        # We detect which one changed since last frame and propagate it
        engine_sel = self.plot.interaction.selected_layer_id
        hud_sel = self.state.selected_layer_id

        if engine_sel != self.state._last_engine_selection:
            # Picking changed it (Engine -> HUD)
            self.state.selected_layer_id = engine_sel
            self.state._last_engine_selection = engine_sel
            self.state._last_hud_selection = engine_sel
        elif hud_sel != self.state._last_hud_selection:
            # UI changed it (HUD -> Engine)
            self.plot.interaction.selected_layer_id = hud_sel
            self.state._last_hud_selection = hud_sel
            self.state._last_engine_selection = hud_sel

        # The workspace owns the main menu bar when it is present: Dear ImGui allows only
        # one per frame, and the workspace's menu already reproduces the View/Actions
        # items below (driving this same state and controller).
        workspace = self.workspace
        if workspace is None:
            self._draw_main_menu()

        if self.state.show_status_overlay:
            self._draw_status_overlay()

        if self.state.show_profiler:
            self._draw_profiler_panel()

        if self.state.show_selection and self.state.selected_object:
            self._draw_selection_panel()

        if self.state.show_analysis:
            self._draw_analysis_panel()

        # The layer tree, render controls and layer inspector now live in the workspace's
        # Scene and Style panels, which supersede them (richer, and correct about which
        # fields the renderers actually read). Drawing both would give the user two
        # disagreeing editors for the same fields, so the legacy panels are only used as
        # the fallback for when the workspace could not be built.
        if workspace is not None:
            workspace.draw()
        else:
            if self.state.show_layers:
                self._draw_layers_panel()
            if self.state.show_render_controls:
                self._draw_render_panel()
            self._draw_layer_inspector()

        # Drawn last and unconditionally (whichever branch above ran): imgui's own
        # draw list composites in call order, so a toast raised by any panel -- or by
        # background work started while that panel wasn't even open -- always ends up
        # on top, regardless of which panel (if any) currently has focus.
        from ..gui import widgets as _widgets

        _widgets.draw_toasts()

    def _draw_main_menu(self) -> None:
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("View"):
                _, self.state.show_status_overlay = imgui.menu_item(
                    "Status Overlay", "", self.state.show_status_overlay
                )
                _, self.state.show_layers = imgui.menu_item("Layers", "", self.state.show_layers)
                _, self.state.show_render_controls = imgui.menu_item(
                    "Render Controls", "", self.state.show_render_controls
                )
                _, self.state.show_profiler = imgui.menu_item(
                    "Profiler", "", self.state.show_profiler
                )
                _, self.state.show_selection = imgui.menu_item(
                    "Selection Info", "", self.state.show_selection
                )
                _, self.state.show_analysis = imgui.menu_item(
                    "Analysis", "", self.state.show_analysis
                )
                imgui.end_menu()

            if imgui.begin_menu("Actions"):
                if imgui.menu_item("Reset View", "", False)[0]:
                    self.controller.reset_view()
                if imgui.menu_item("Autoscale", "", False)[0]:
                    self.controller.autoscale()
                imgui.separator()
                if imgui.menu_item("Toggle Density", "", False)[0]:
                    self.controller.toggle_density()
                if imgui.menu_item("Export PNG", "", False)[0]:
                    self.controller.export()
                imgui.end_menu()

            imgui.end_main_menu_bar()

    def _draw_status_overlay(self) -> None:
        # Transparent overlay strip at the top (below menu)
        imgui.set_next_window_pos((0, 20))
        imgui.set_next_window_size((self.plot.width, 30))
        flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_background
        )

        imgui.begin("StatusOverlay", flags=flags)

        fps = (
            1.0 / max(0.0001, self.state.cpu_frame_times[-1]) if self.state.cpu_frame_times else 0.0
        )
        n_lines = self.plot.scene.lines.count
        n_3d_layers = len(self.plot.get_3d_layers()) if hasattr(self.plot, "get_3d_layers") else 0
        n_3d_vertices = (
            sum(
                len(layer.vertices)
                for layer in self.plot.get_3d_layers()
                if getattr(layer, "vertices", None) is not None
            )
            if n_3d_layers
            else 0
        )

        # Density is a 2D-only accumulator (see engine._density_active). When it is switched
        # on in a 3D scene it does nothing, so the readout says so rather than claiming a
        # "Density" mode that the frame is not actually in — and points at the 3D
        # equivalent, a volume3d layer.
        if self.plot.display_density and self.plot.is_3d_scene():
            mode = "Exact (density is 2D — use volume3d in 3D)"
        elif self.plot.display_density:
            mode = "Density"
        else:
            mode = "Exact"
        imgui.text(
            f"FPS: {fps:4.1f} | Lines: {n_lines:,} | 3D Layers: {n_3d_layers:,} "
            f"({n_3d_vertices:,} verts) | Mode: {mode} | "
        )
        imgui.same_line()
        display_world = self.plot.mouse_world_display()
        if display_world:
            imgui.text(f"Mouse: ({display_world[0]:.4f}, {display_world[1]:.4f}) | ")
            imgui.same_line()

        imgui.text(f"Alpha: {self.options.default_global_alpha:.3f} | ")
        imgui.same_line()

        mode = self.plot.interaction.drag_mode
        imgui.text_colored((1.0, 1.0, 0.4, 1.0), f"Drag: {mode.upper()}")

        imgui.end()

    def _draw_layers_panel(self) -> None:
        imgui.set_next_window_size((300, 400), imgui.Cond_.first_use_ever)
        imgui.begin("Layers & Stacking", True)

        imgui.text_disabled("Drag to reorder (List order = Render order)")
        imgui.separator()

        layers = self.plot.scene.layers
        to_move = None

        for i, layer in enumerate(layers):
            imgui.push_id(str(layer.layer_id))

            # Visibility checkbox
            changed, visible = imgui.checkbox("##vis", layer.style.visible)
            if changed:
                layer.style.visible = visible
                layer.dirty.style_dirty = True
                self.plot.frame.dirty_scene = True
                self.plot.cache.refresh_requested = True

            imgui.same_line()

            # Layer Selectable (Drag Source/Target)
            is_selected = self.state.selected_layer_id == layer.layer_id
            layer_name = layer.label or layer.layer_type
            size_text = self._layer_size_text(layer)
            suffix = f" [{layer.layer_type}"
            if size_text:
                suffix += f", {size_text}"
            suffix += "]"
            _, selected = imgui.selectable(f"{layer_name}{suffix}##{i}", is_selected)
            if selected:
                self.state.selected_layer_id = layer.layer_id

            # Drag and Drop Reordering
            if imgui.is_item_active() and not imgui.is_item_hovered():
                # Potential drag start
                pass

            if imgui.begin_drag_drop_source():
                imgui.set_drag_drop_payload_py_id("LAYER_ORDER", i)
                imgui.text(f"Moving {layer.label}...")
                imgui.end_drag_drop_source()

            if imgui.begin_drag_drop_target():
                payload = imgui.accept_drag_drop_payload_py_id("LAYER_ORDER")
                if payload:
                    src_idx = payload.data_id
                    to_move = (src_idx, i)
                imgui.end_drag_drop_target()

            imgui.pop_id()

        if to_move:
            src, dst = to_move
            layer = layers.pop(src)
            layers.insert(dst, layer)
            self.plot.frame.dirty_scene = True
            self.plot.cache.refresh_requested = True

        imgui.end()

    def _draw_layer_inspector(self) -> None:
        if self.state.selected_layer_id is None:
            return

        layer = next(
            (lyr for lyr in self.plot.scene.layers if lyr.layer_id == self.state.selected_layer_id),
            None,
        )
        if not layer:
            return

        def _mark_dirty():
            layer.dirty.style_dirty = True
            self.plot.frame.dirty_scene = True
            self.plot.cache.refresh_requested = True

        imgui.set_next_window_size((300, 300), imgui.Cond_.first_use_ever)
        imgui.begin("Layer Inspector", True)

        imgui.text(f"ID: {layer.layer_id}")
        imgui.text(f"Type: {layer.layer_type.upper()}")

        changed, label = imgui.input_text("Label", layer.label, 64)
        if changed:
            layer.label = label

        imgui.separator()

        style = layer.style
        if imgui.collapsing_header("Style Properties", imgui.TreeNodeFlags_.default_open):
            # Alpha
            changed, alpha = imgui.slider_float("Alpha", style.alpha, 0.0, 1.0)
            if changed:
                style.alpha = alpha
                _mark_dirty()

            # Color (if applicable)
            if style.color is not None:
                changed, color = imgui.color_edit4("Primary Color", style.color)
                if changed:
                    style.color = color
                    _mark_dirty()

            # Line Width
            if layer.layer_type in ["polyline", "wireframe3d"]:
                changed, lw = imgui.slider_float("Line Width", style.line_width, 0.1, 10.0)
                if changed:
                    style.line_width = lw
                    _mark_dirty()

            # Point Size
            if layer.layer_type in ["scatter", "scatter3d", "volume3d"]:
                changed, ps = imgui.slider_float("Point Size", style.point_size, 1.0, 100.0)
                if changed:
                    style.point_size = ps
                    _mark_dirty()

                imgui.separator()
                imgui.text("Outline")
                changed, out_en = imgui.checkbox("Enable Outline", style.point_outline_enabled)
                if changed:
                    style.point_outline_enabled = out_en
                    _mark_dirty()

                changed, out_col = imgui.color_edit4("Outline Color", style.point_outline_color)
                if changed:
                    style.point_outline_color = out_col
                    _mark_dirty()

                changed, out_w = imgui.slider_float(
                    "Outline Width", style.point_outline_width, 0.1, 5.0
                )
                if changed:
                    style.point_outline_width = out_w
                    _mark_dirty()

            if self._is_3d_layer(layer):
                imgui.separator()
                imgui.text("3D Camera")
                camera = dict(layer.metadata.get("camera", {}))
                changed, elev = imgui.slider_float(
                    "Elev",
                    float(camera.get("elev", self.plot.view3d.get("elev", 28.0))),
                    -90.0,
                    90.0,
                )
                if changed:
                    camera["elev"] = elev
                    layer.metadata["camera"] = camera
                    _mark_dirty()
                changed, azim = imgui.slider_float(
                    "Azim",
                    float(camera.get("azim", self.plot.view3d.get("azim", -45.0))),
                    -180.0,
                    180.0,
                )
                if changed:
                    camera["azim"] = azim
                    layer.metadata["camera"] = camera
                    _mark_dirty()
                changed, fov = imgui.slider_float(
                    "FOV", float(camera.get("fov", self.plot.view3d.get("fov", 42.0))), 15.0, 90.0
                )
                if changed:
                    camera["fov"] = fov
                    layer.metadata["camera"] = camera
                    _mark_dirty()

        if imgui.collapsing_header("Transformation", imgui.TreeNodeFlags_.default_open):
            imgui.text("World space offset")
            tx, ty = layer.translation
            changed, n_trans = imgui.drag_float2("Translation", (tx, ty), 0.05)
            if changed:
                layer.translation = (n_trans[0], n_trans[1])
                self.plot.frame.dirty_scene = True
                self.plot.cache.refresh_requested = True

            if imgui.button("Reset Position"):
                layer.translation = (0.0, 0.0)
                self.plot.frame.dirty_scene = True
                self.plot.cache.refresh_requested = True

        imgui.end()

    def _draw_render_panel(self) -> None:
        imgui.set_next_window_size((300, 450), imgui.Cond_.first_use_ever)
        imgui.begin("Render & Style", True)

        if imgui.collapsing_header("Global Style", imgui.TreeNodeFlags_.default_open):
            ov = self.options.visual.overrides
            vis = self.options.visual

            _, vis.background_color = imgui.color_edit3("Background Color", vis.background_color)
            if imgui.is_item_deactivated_after_edit():
                self.plot.frame.dirty_scene = True

            changed, alpha = imgui.slider_float("Alpha Mult", ov.alpha_multiplier, 0.0, 2.0)
            if changed:
                ov.alpha_multiplier = alpha
                self.plot.frame.dirty_scene = True

            changed, lw = imgui.slider_float("Line Width Mult", ov.line_width_multiplier, 0.1, 5.0)
            if changed:
                ov.line_width_multiplier = lw
                self.plot.frame.dirty_scene = True

            changed, ps = imgui.slider_float("Point Size Mult", ov.point_size_multiplier, 0.1, 5.0)
            if changed:
                ov.point_size_multiplier = ps
                self.plot.frame.dirty_scene = True

            # Blending Mode Dropdown
            from ..options import BlendMode

            items = [m.name for m in BlendMode]
            try:
                current = items.index(self.options.blend_mode.name)
            except Exception:
                current = 0

            imgui.push_item_width(120)
            changed, clicked = imgui.combo("Blending", current, items)
            if changed:
                self.options.blend_mode = list(BlendMode)[clicked]
                self.plot.frame.dirty_scene = True
            imgui.pop_item_width()

            # Antialiasing Toggle
            changed, aa_en = imgui.checkbox("Antialiasing (SDF)", self.options.enable_antialiasing)
            if changed:
                self.options.enable_antialiasing = aa_en
                self.plot.frame.dirty_scene = True

        if imgui.collapsing_header("Density Engine", imgui.TreeNodeFlags_.default_open):
            from ..utils.shaders import DENSITY_SCHEMES

            # Mode Toggle
            changed, den_en = imgui.checkbox(
                "Enable Density Mode (Heatmap)", self.plot.display_density
            )
            if changed:
                self.plot.set_density_enabled(den_en)

            # Weighted Density Toggle
            changed, weighted = imgui.checkbox(
                "Weighted Accumulation", self.options.density_weighted
            )
            if changed:
                self.options.density_weighted = weighted
                self.plot.frame.dirty_scene = True

            # Scheme Selection
            imgui.push_item_width(150)
            changed, idx = imgui.combo("Scheme", self.options.density_scheme_index, DENSITY_SCHEMES)
            if changed:
                self.options.density_scheme_index = idx
                self.plot.frame.dirty_scene = True
            imgui.pop_item_width()

            # Light BG Mode
            light_mode = getattr(self.options, "light_bg_mode", False)
            changed_bg, val_bg = imgui.checkbox("Light BG Mode", light_mode)
            if changed_bg:
                self.options.light_bg_mode = val_bg
                self.options.density_invert = val_bg
                self.plot.frame.dirty_scene = True
                self.plot.frame.dirty_ui = True

            # Invert Colors
            invert_mode = getattr(self.options, "density_invert", False)
            changed_inv, val_inv = imgui.checkbox("Invert Colors", invert_mode)
            if changed_inv:
                self.options.density_invert = val_inv
                self.plot.frame.dirty_scene = True
                self.plot.frame.dirty_ui = True

            # Light to Color Only option
            is_inverted = getattr(self.options, "density_invert", False)
            label = "Light to Color Only" if is_inverted else "Dark to Color Only"
            light_to_color = getattr(self.options, "density_light_to_color", True)
            changed_ltc, val_ltc = imgui.checkbox(label, light_to_color)
            if changed_ltc:
                self.options.density_light_to_color = val_ltc
                self.plot.frame.dirty_scene = True
                self.plot.frame.dirty_ui = True

            # Gain and Scale
            imgui.push_item_width(180)
            changed, gain = imgui.drag_float(
                "Gain (Intensity)", self.options.density_gain, 1.0, 0.1, 10000.0, "%.1f"
            )
            if changed:
                self.options.density_gain = gain
                self.plot.frame.dirty_scene = True

            is_log = getattr(self.options, "density_is_log", True)
            changed_log, val_log = imgui.checkbox("Logarithmic Scale", is_log)
            if changed_log:
                self.options.density_is_log = val_log
                self.plot.frame.dirty_scene = True

            changed, res = imgui.slider_float(
                "Inner Resolution", self.options.density_resolution_scale, 0.1, 1.0, "%.2f"
            )
            if changed:
                self.controller.set_density_resolution(res)
            imgui.text_disabled("0.5x is 4x faster, 1.0x is sharpest")
            imgui.pop_item_width()

            # Color Bar Preview
            imgui.text("Colormap Preview:")
            draw_list = imgui.get_window_draw_list()
            pos = imgui.get_cursor_screen_pos()
            w, h = 220, 18

            from ..utils.shaders import eval_colormap

            is_inverted = getattr(self.options, "density_invert", False)
            ltc = getattr(self.options, "density_light_to_color", True)
            scheme_idx = self.options.density_scheme_index

            for i in range(20):
                v = i / 19.0
                if is_inverted:
                    if ltc:
                        norm = 1.0 - 0.75 * v
                    else:
                        norm = 1.0 - v
                else:
                    if ltc:
                        norm = 0.75 * v
                    else:
                        norm = v

                c = eval_colormap(scheme_idx, norm)
                color = imgui.get_color_u32(
                    (max(0, min(1, c[0])), max(0, min(1, c[1])), max(0, min(1, c[2])), 1.0)
                )
                draw_list.add_rect_filled(
                    (pos.x + i * (w / 20), pos.y), (pos.x + (i + 1) * (w / 20), pos.y + h), color
                )

            imgui.dummy((w, h + 5))

        if hasattr(self.plot, "is_3d_scene") and self.plot.is_3d_scene():
            if imgui.collapsing_header("3D Camera", imgui.TreeNodeFlags_.default_open):
                imgui.text_disabled("Drag: rotate   Scroll: zoom   Reset: button below")
                imgui.separator()

                view = self.plot.view3d
                changed, elev = imgui.slider_float(
                    "Elevation", float(view.get("elev", 28.0)), -90.0, 90.0
                )
                if changed:
                    self.controller.set_3d_view(elev=elev)
                changed, azim = imgui.slider_float(
                    "Azimuth", float(view.get("azim", -45.0)), -180.0, 180.0
                )
                if changed:
                    self.controller.set_3d_view(azim=azim)
                changed, fov = imgui.slider_float("FOV", float(view.get("fov", 42.0)), 15.0, 90.0)
                if changed:
                    self.controller.set_3d_view(fov=fov)

                dist = view.get("distance")
                if dist is None:
                    imgui.text_disabled("Distance: auto  (scroll to control)")
                else:
                    dist_max = max(float(dist) * 8.0, 10.0)
                    changed, dist_val = imgui.slider_float(
                        "Distance##3d", float(dist), 0.001, dist_max
                    )
                    if changed:
                        self.controller.set_3d_view(distance=max(0.001, dist_val))
                    imgui.same_line()
                    if imgui.button("Auto##3ddist"):
                        self.plot.view3d["distance"] = None
                        for layer in self.plot.get_3d_layers():
                            cam = layer.metadata.get("camera", {})
                            cam.pop("distance", None)
                            layer.metadata["camera"] = cam
                            layer.dirty.style_dirty = True
                        self.plot.frame.dirty_scene = True

                changed, show_axes = imgui.checkbox(
                    "Show 3D Axes", bool(view.get("show_axes", True))
                )
                if changed:
                    self.controller.set_3d_view(show_axes=show_axes)
                if imgui.button("Reset 3D View"):
                    self.controller.reset_3d_view()
                imgui.same_line()
                if imgui.button("Refresh 3D Axes"):
                    self.plot.ensure_3d_axes()
                    self.plot.frame.dirty_scene = True

                imgui.separator()
                ssao = self.options.visual.ssao
                changed, ssao_enabled = imgui.checkbox("SSAO / Ambient Occlusion", ssao.enabled)
                if changed:
                    ssao.enabled = ssao_enabled
                    self.plot.frame.dirty_scene = True
                changed, strength = imgui.slider_float("SSAO Strength", ssao.strength, 0.0, 1.5)
                if changed:
                    ssao.strength = strength
                    self.plot.frame.dirty_scene = True
                changed, radius = imgui.slider_float("SSAO Radius", ssao.radius, 0.1, 5.0)
                if changed:
                    ssao.radius = radius
                    self.plot.frame.dirty_scene = True

        if imgui.collapsing_header("Plot Framework", imgui.TreeNodeFlags_.default_open):
            _, self.options.axis_show_grid = imgui.checkbox(
                "Show Grid", self.options.axis_show_grid
            )
            if self.options.axis_show_grid:
                imgui.same_line()
                imgui.push_item_width(100)
                _, self.options.axis_grid_alpha = imgui.slider_float(
                    "Alpha##Grid", self.options.axis_grid_alpha, 0.0, 1.0
                )
                imgui.pop_item_width()

                imgui.indent(10)
                _, self.options.axis_grid_color = imgui.color_edit3(
                    "Grid Color", self.options.axis_grid_color
                )
                imgui.unindent(10)

            _, self.options.axis_show_labels = imgui.checkbox(
                "Show Scale (Labels)", self.options.axis_show_labels
            )
            _, self.options.axis_show_frame = imgui.checkbox(
                "Show Framework", self.options.axis_show_frame
            )

        if imgui.collapsing_header("LOD Configuration", imgui.TreeNodeFlags_.default_open):
            changed, lod_en = imgui.checkbox("LOD Enabled (Sub-sampling)", self.options.lod_enabled)
            if changed:
                self.controller.set_lod_enabled(lod_en)

            if self.options.lod_enabled:
                imgui.push_item_width(180)
                # Range 0.1 to 500.0 covers from very sparse to extreme fidelity (overkill)
                changed, budget = imgui.slider_float(
                    "Complexity Budget", self.options.lod_target_coverage, 0.1, 500.0, "%.2f"
                )
                if changed:
                    self.controller.set_lod_budget(budget)
                imgui.pop_item_width()
                imgui.text_disabled("Higher = More detail, lower FPS")

        if imgui.collapsing_header("Visual Effects", imgui.TreeNodeFlags_.default_open):
            v = self.options.visual

            # --- Background ---
            if imgui.tree_node_ex("Gradient Background", imgui.TreeNodeFlags_.default_open):
                _, v.gradient_background.enabled = imgui.checkbox(
                    "Enabled##BG", v.gradient_background.enabled
                )
                _, v.gradient_background.top_color = imgui.color_edit3(
                    "Top Color", v.gradient_background.top_color
                )
                _, v.gradient_background.bottom_color = imgui.color_edit3(
                    "Bottom Color", v.gradient_background.bottom_color
                )
                imgui.tree_pop()

            # --- Bloom ---
            if imgui.tree_node_ex("Bloom / Glow", imgui.TreeNodeFlags_.default_open):
                _, v.glow.enabled = imgui.checkbox("Enabled##Bloom", v.glow.enabled)
                imgui.push_item_width(120)
                _, v.glow.intensity = imgui.slider_float("Intensity", v.glow.intensity, 0.0, 5.0)
                _, v.glow.threshold = imgui.slider_float("Threshold", v.glow.threshold, 0.0, 1.0)
                _, v.glow.radius_px = imgui.slider_float("Radius", v.glow.radius_px, 1.0, 20.0)
                imgui.pop_item_width()
                imgui.tree_pop()

        imgui.end()

    def _draw_profiler_panel(self) -> None:
        imgui.set_next_window_size((350, 420), imgui.Cond_.first_use_ever)
        expanded, opened = imgui.begin("Profiler", True)
        if not opened:
            self.state.show_profiler = False
            imgui.end()
            return

        if expanded:
            # 1. Performance Overview
            imgui.text_colored((0.3, 0.8, 1.0, 1.0), "PERFORMANCE OVERVIEW")
            if self.state.cpu_frame_times:
                avg_ms = sum(self.state.cpu_frame_times) / len(self.state.cpu_frame_times) * 1000.0
                imgui.text(f"Avg CPU Frame: {avg_ms:5.2f} ms ({1000.0/avg_ms:.1f} FPS)")

                # Sparkline
                import numpy as np

                history = np.array(self.state.cpu_frame_times, dtype=np.float32)
                imgui.plot_lines(
                    "##FPSGraph",
                    history,
                    overlay_text="",
                    scale_min=0,
                    scale_max=0.033,
                    graph_size=(0, 50),
                )

            imgui.separator()

            # 2. Rendering Pipelines Detailed Timings
            imgui.text_colored((0.3, 0.8, 1.0, 1.0), "SUB-OPERATION TIMINGS")
            for k, v in sorted(self.state.gpu_timings.items()):
                imgui.text(f" - {k}: {v*1000.0:5.2f} ms")

            imgui.separator()

            # 3. Engine State & Statistics
            imgui.text_colored((0.3, 0.8, 1.0, 1.0), "ENGINE STATISTICS")
            mode_str = self.plot.policy.runtime.current_mode.name
            imgui.text(f"Render Mode: {mode_str}")

            sleep_str = (
                "SLEEPING (Reactive)"
                if self.plot.options.reactive_rendering and not self.plot.hud.state.show_profiler
                else "ACTIVE Redrawing"
            )
            imgui.text(f"Gating Loop: {sleep_str}")

            imgui.text(f"Total Lines: {self.plot.scene.lines.count:,}")

            win = self.plot.camera_controller.world_window(self.plot.width, self.plot.height)
            imgui.text(f"X range: [{win[0]:.2f}, {win[1]:.2f}]")
            imgui.text(f"Y range: [{win[2]:.2f}, {win[3]:.2f}]")

        imgui.end()

    def _draw_selection_panel(self) -> None:
        imgui.set_next_window_size((250, 180), imgui.Cond_.first_use_ever)
        imgui.begin("Selection Info", True)

        info = self.state.selected_object
        imgui.text_colored((1.0, 0.8, 0.4, 1.0), f"Type: {info.get('type', 'N/A').upper()}")
        imgui.text(f"Dataset: {info.get('dataset_idx', 0)}")
        imgui.text(f"Index: {info.get('element_idx', 0)}")
        imgui.separator()
        imgui.text(f"X: {info.get('x', 0.0):.6f}")
        imgui.text(f"Y: {info.get('y', 0.0):.6f}")

        if imgui.button("Isolate"):
            # Logic later
            pass
        imgui.same_line()
        if imgui.button("Reset"):
            self.state.selected_object = None

        imgui.end()

    def _draw_analysis_panel(self) -> None:
        imgui.set_next_window_size((400, 300), imgui.Cond_.first_use_ever)
        imgui.begin("Analysis (Sampled)", True)

        if self.state.sampled_histogram_a is not None:
            imgui.text("Slope (a) Distribution")
            imgui.plot_histogram("##HistA", self.state.sampled_histogram_a, graph_size=(0, 80))

        if self.state.sampled_histogram_b is not None:
            imgui.text("Intercept (b) Distribution")
            imgui.plot_histogram("##HistB", self.state.sampled_histogram_b, graph_size=(0, 80))

        imgui.end()

    def end(self) -> None:
        if self.imgui_impl:
            imgui.render()
            self.imgui_impl.render(imgui.get_draw_data())

    def get_draw_list(self) -> Any:
        if not IMGUI_AVAILABLE:
            return None
        return imgui.get_background_draw_list()
