from __future__ import annotations
import typing
from typing import TYPE_CHECKING, Optional, Tuple, Any

from .hud_state import HudState, HudController

if TYPE_CHECKING:
    from ..options import EngineOptions, BlendMode
    from ..core import CameraState, FrameState
    from ..engine import GPULinePlot

try:
    import imgui
    from imgui.integrations.glfw import GlfwRenderer
    IMGUI_AVAILABLE = True
except (ImportError, Exception):
    IMGUI_AVAILABLE = False
    imgui = None
    GlfwRenderer = None

class HudManager:
    def __init__(self, plot: GPULinePlot):
        self.plot = plot
        self.options = plot.options
        self.state = HudState()
        self.controller = HudController(plot)
        self.imgui_ctx = None
        self.imgui_impl = None

    def initialize(self, window) -> None:
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

    def on_scroll(self, window, dx, dy) -> None:
        if self.imgui_impl:
            self.imgui_impl.scroll_callback(window, dx, dy)

    def on_mouse_button(self, window, button, action, mods) -> None:
        if self.imgui_impl:
            self.imgui_impl.mouse_callback(window, button, action, mods)

    def on_key(self, window, key, sc, action, mods) -> None:
        if self.imgui_impl:
            self.imgui_impl.keyboard_callback(window, key, sc, action, mods)

    def on_char(self, window, char) -> None:
        if self.imgui_impl:
            self.imgui_impl.char_callback(window, char)

    def wants_mouse(self) -> bool:
        if not IMGUI_AVAILABLE or not self.imgui_impl:
            return False
        return imgui.get_io().want_capture_mouse

    def get_draw_list(self):
        """Returns the ImGui background draw list if available."""
        if not IMGUI_AVAILABLE:
            return None
        return imgui.get_background_draw_list()

    def wants_keyboard(self) -> bool:
        if not IMGUI_AVAILABLE or not self.imgui_impl:
            return False
        return imgui.get_io().want_capture_keyboard

    def begin(self) -> None:
        if self.imgui_impl:
            imgui.new_frame()

    def update(self):
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

        self._draw_main_menu()
        
        if self.state.show_status_overlay:
            self._draw_status_overlay()
            
        if self.state.show_layers:
            self._draw_layers_panel()
            
        if self.state.show_render_controls:
            self._draw_render_panel()
            
        if self.state.show_profiler:
            self._draw_profiler_panel()
            
        if self.state.show_selection and self.state.selected_object:
            self._draw_selection_panel()

        if self.state.show_analysis:
            self._draw_analysis_panel()
            
        self._draw_layer_inspector()

    def _draw_main_menu(self):
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("View"):
                _, self.state.show_status_overlay = imgui.menu_item("Status Overlay", None, self.state.show_status_overlay)
                _, self.state.show_layers = imgui.menu_item("Layers", None, self.state.show_layers)
                _, self.state.show_render_controls = imgui.menu_item("Render Controls", None, self.state.show_render_controls)
                _, self.state.show_profiler = imgui.menu_item("Profiler", None, self.state.show_profiler)
                _, self.state.show_selection = imgui.menu_item("Selection Info", None, self.state.show_selection)
                _, self.state.show_analysis = imgui.menu_item("Analysis", None, self.state.show_analysis)
                imgui.end_menu()
            
            if imgui.begin_menu("Actions"):
                if imgui.menu_item("Reset View")[0]: self.controller.reset_view()
                if imgui.menu_item("Autoscale")[0]: self.controller.autoscale()
                imgui.separator()
                if imgui.menu_item("Toggle Density")[0]: self.controller.toggle_density()
                if imgui.menu_item("Export PNG")[0]: self.controller.export()
                imgui.end_menu()
                
            imgui.end_main_menu_bar()

    def _draw_status_overlay(self):
        # Transparent overlay strip at the top (below menu)
        imgui.set_next_window_position(0, 20)
        imgui.set_next_window_size(self.plot.width, 30)
        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | 
                 imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR | 
                 imgui.WINDOW_NO_BACKGROUND)
        
        imgui.begin("StatusOverlay", flags=flags)
        
        fps = 1.0 / max(0.0001, self.state.cpu_frame_times[-1]) if self.state.cpu_frame_times else 0.0
        n_lines = self.plot.scene.lines.count
        
        imgui.text(f"FPS: {fps:4.1f} | Lines: {n_lines:,} | Mode: {'Density' if self.plot.display_density else 'Exact'} | ")
        imgui.same_line()
        if self.plot.mouse_world:
            imgui.text(f"Mouse: ({self.plot.mouse_world[0]:.4f}, {self.plot.mouse_world[1]:.4f}) | ")
            imgui.same_line()
        
        imgui.text(f"Alpha: {self.options.default_global_alpha:.3f} | ")
        imgui.same_line()
        
        mode = self.plot.interaction.drag_mode
        imgui.text_colored(f"Drag: {mode.upper()}", 1.0, 1.0, 0.4)
        
        imgui.end()

    def _draw_layers_panel(self):
        imgui.set_next_window_size(300, 400, imgui.FIRST_USE_EVER)
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
            is_selected = (self.state.selected_layer_id == layer.layer_id)
            _, selected = imgui.selectable(f"{layer.label}##{i}", is_selected)
            if selected:
                self.state.selected_layer_id = layer.layer_id
            
            # Drag and Drop Reordering
            if imgui.is_item_active() and not imgui.is_item_hovered():
                # Potential drag start
                pass
            
            if imgui.begin_drag_drop_source():
                imgui.set_drag_drop_payload("LAYER_ORDER", str(i).encode())
                imgui.text(f"Moving {layer.label}...")
                imgui.end_drag_drop_source()
                
            if imgui.begin_drag_drop_target():
                payload = imgui.accept_drag_drop_payload("LAYER_ORDER")
                if payload:
                    src_idx = int(payload.decode())
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

    def _draw_layer_inspector(self):
        if self.state.selected_layer_id is None: return
        
        layer = next((l for l in self.plot.scene.layers if l.layer_id == self.state.selected_layer_id), None)
        if not layer: return

        def _mark_dirty():
            layer.dirty.style_dirty = True
            self.plot.frame.dirty_scene = True
            self.plot.cache.refresh_requested = True
        
        imgui.set_next_window_size(300, 300, imgui.FIRST_USE_EVER)
        imgui.begin("Layer Inspector", True)
        
        imgui.text(f"ID: {layer.layer_id}")
        imgui.text(f"Type: {layer.layer_type.upper()}")
        
        changed, label = imgui.input_text("Label", layer.label, 64)
        if changed: layer.label = label
        
        imgui.separator()
        
        style = layer.style
        if imgui.collapsing_header("Style Properties", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            # Alpha
            changed, alpha = imgui.slider_float("Alpha", style.alpha, 0.0, 1.0)
            if changed: style.alpha = alpha; _mark_dirty()
            
            # Color (if applicable)
            if style.color is not None:
                changed, color = imgui.color_edit4("Primary Color", *style.color)
                if changed: style.color = color; _mark_dirty()
                
            # Line Width
            if layer.layer_type in ["polyline"]:
                changed, lw = imgui.slider_float("Line Width", style.line_width, 0.1, 10.0)
                if changed: style.line_width = lw; _mark_dirty()
                
            # Point Size
            if layer.layer_type == "scatter":
                changed, ps = imgui.slider_float("Point Size", style.point_size, 1.0, 100.0)
                if changed: style.point_size = ps; _mark_dirty()
                
                imgui.separator()
                imgui.text("Outline")
                changed, out_en = imgui.checkbox("Enable Outline", style.point_outline_enabled)
                if changed: style.point_outline_enabled = out_en; _mark_dirty()
                
                changed, out_col = imgui.color_edit4("Outline Color", *style.point_outline_color)
                if changed: style.point_outline_color = out_col; _mark_dirty()
                
                changed, out_w = imgui.slider_float("Outline Width", style.point_outline_width, 0.1, 5.0)
                if changed: style.point_outline_width = out_w; _mark_dirty()

        if imgui.collapsing_header("Transformation", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            imgui.text("World space offset")
            tx, ty = layer.translation
            changed, n_trans = imgui.drag_float2("Translation", tx, ty, 0.05)
            if changed:
                layer.translation = (n_trans[0], n_trans[1])
                self.plot.frame.dirty_scene = True
                self.plot.cache.refresh_requested = True
            
            if imgui.button("Reset Position"):
                layer.translation = (0.0, 0.0)
                self.plot.frame.dirty_scene = True
                self.plot.cache.refresh_requested = True

        imgui.end()

    def _draw_render_panel(self):
        imgui.set_next_window_size(300, 450, imgui.FIRST_USE_EVER)
        imgui.begin("Render & Style", True)
        
        if imgui.collapsing_header("Global Style", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            ov = self.options.visual.overrides
            vis = self.options.visual
            
            _, vis.background_color = imgui.color_edit3("Background Color", *vis.background_color)
            if imgui.is_item_deactivated_after_edit():
                 self.plot.frame.dirty_scene = True
            
            changed, alpha = imgui.slider_float("Alpha Mult", ov.alpha_multiplier, 0.0, 2.0)
            if changed: ov.alpha_multiplier = alpha; self.plot.frame.dirty_scene = True
            
            changed, lw = imgui.slider_float("Line Width Mult", ov.line_width_multiplier, 0.1, 5.0)
            if changed: ov.line_width_multiplier = lw; self.plot.frame.dirty_scene = True
            
            changed, ps = imgui.slider_float("Point Size Mult", ov.point_size_multiplier, 0.1, 5.0)
            if changed: ov.point_size_multiplier = ps; self.plot.frame.dirty_scene = True
            
            # Blending Mode Dropdown
            from ..options import BlendMode
            items = [m.name for m in BlendMode]
            try:
                current = items.index(self.options.blend_mode.name)
            except:
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
        
        if imgui.collapsing_header("Density Engine", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            from ..utils.shaders import DENSITY_SCHEMES
            
            # Mode Toggle
            changed, den_en = imgui.checkbox("Enable Density Mode (Heatmap)", self.plot.display_density)
            if changed: self.plot.set_density_enabled(den_en)
            
            # Weighted Density Toggle
            changed, weighted = imgui.checkbox("Weighted Accumulation", self.options.density_weighted)
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
            
            # Gain and Scale
            imgui.push_item_width(180)
            changed, gain = imgui.drag_float("Gain (Intensity)", self.options.density_gain, 1.0, 0.1, 10000.0, "%.1f")
            if changed: self.options.density_gain = gain; self.plot.frame.dirty_scene = True
            
            changed, lscale = imgui.drag_float("Log Scale (Contrast)", self.options.density_log_scale, 0.05, 0.1, 20.0, "%.2f")
            if changed: self.options.density_log_scale = lscale; self.plot.frame.dirty_scene = True

            changed, res = imgui.slider_float("Inner Resolution", self.options.density_resolution_scale, 0.1, 1.0, "%.2f")
            if changed: self.controller.set_density_resolution(res)
            imgui.text_disabled("0.5x is 4x faster, 1.0x is sharpest")
            imgui.pop_item_width()
            
            # Color Bar Preview
            imgui.text("Colormap Preview:")
            draw_list = imgui.get_window_draw_list()
            pos = imgui.get_cursor_screen_pos()
            w, h = 220, 18
            
            def get_scheme_col(v):
                scheme = self.options.density_scheme_index
                if scheme == 1: # Viridis
                    if v < 0.5: return (0.2+v*0.1, 0.1+v*0.8, 0.4+v*0.2)
                    return (0.3+(v-0.5)*1.4, 0.5+(v-0.5)*0.8, 0.5-(v-0.5)*0.8)
                if scheme == 2: # Plasma
                    return (0.1+v*1.8, 0.0, 0.5+v*0.5) if v < 0.5 else (1.0, (v-0.5)*2.0, 1.0-v)
                if scheme == 3: # Inferno
                    return (v*0.5, 0.0, v*0.2) if v < 0.5 else (v, (v-0.5)*1.5, (v-0.5)*2.0)
                if scheme == 4: # Turbo
                    if v < 0.33: return (0.2, v*2.5, 0.8)
                    if v < 0.66: return (v*1.5, 0.8, 0.2)
                    return (0.9, 0.2, 0.1)
                if scheme == 5: # Ink Fire (White BG)
                    return (1.0, 1.0-(v*0.5), 1.0-(v*1.5)) if v < 0.5 else (1.0-(v-0.5)*2.0, 0.25-(v-0.5)*0.5, 0.0)
                if scheme == 6: # Magma
                    return (v*0.4, 0.1, 0.5) if v < 0.5 else (v, 0.5+(v-0.5), 0.8+(v-0.5)*0.4)
                return (v, v, v)
                
            for i in range(20):
                v = i / 19.0
                c = get_scheme_col(v)
                color = imgui.get_color_u32_rgba(max(0,min(1,c[0])), max(0,min(1,c[1])), max(0,min(1,c[2])), 1.0)
                draw_list.add_rect_filled(pos.x + i*(w/20), pos.y, pos.x + (i+1)*(w/20), pos.y + h, color)
            
            imgui.dummy(w, h + 5)

        if imgui.collapsing_header("Plot Framework", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            _, self.options.axis_show_grid = imgui.checkbox("Show Grid", self.options.axis_show_grid)
            if self.options.axis_show_grid:
                imgui.same_line()
                imgui.push_item_width(100)
                _, self.options.axis_grid_alpha = imgui.slider_float("Alpha##Grid", self.options.axis_grid_alpha, 0.0, 1.0)
                imgui.pop_item_width()
                
                imgui.indent(10)
                _, self.options.axis_grid_color = imgui.color_edit3("Grid Color", *self.options.axis_grid_color)
                imgui.unindent(10)
                
            _, self.options.axis_show_labels = imgui.checkbox("Show Scale (Labels)", self.options.axis_show_labels)
            _, self.options.axis_show_frame = imgui.checkbox("Show Framework", self.options.axis_show_frame)

        if imgui.collapsing_header("LOD Configuration", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            changed, lod_en = imgui.checkbox("LOD Enabled (Sub-sampling)", self.options.lod_enabled)
            if changed: self.controller.set_lod_enabled(lod_en)

            if self.options.lod_enabled:
                imgui.push_item_width(180)
                # Range 0.1 to 500.0 covers from very sparse to extreme fidelity (overkill)
                changed, budget = imgui.slider_float("Complexity Budget", self.options.lod_target_coverage, 0.1, 500.0, "%.2f")
                if changed: self.controller.set_lod_budget(budget)
                imgui.pop_item_width()
                imgui.text_disabled("Higher = More detail, lower FPS")
                

        if imgui.collapsing_header("Visual Effects", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            v = self.options.visual
            
            # --- Background ---
            if imgui.tree_node("Gradient Background", imgui.TREE_NODE_DEFAULT_OPEN):
                _, v.gradient_background.enabled = imgui.checkbox("Enabled##BG", v.gradient_background.enabled)
                _, v.gradient_background.top_color = imgui.color_edit3("Top Color", *v.gradient_background.top_color)
                _, v.gradient_background.bottom_color = imgui.color_edit3("Bottom Color", *v.gradient_background.bottom_color)
                imgui.tree_pop()
            
            # --- Bloom ---
            if imgui.tree_node("Bloom / Glow", imgui.TREE_NODE_DEFAULT_OPEN):
                _, v.glow.enabled = imgui.checkbox("Enabled##Bloom", v.glow.enabled)
                imgui.push_item_width(120)
                _, v.glow.intensity = imgui.slider_float("Intensity", v.glow.intensity, 0.0, 5.0)
                _, v.glow.threshold = imgui.slider_float("Threshold", v.glow.threshold, 0.0, 1.0)
                _, v.glow.radius_px = imgui.slider_float("Radius", v.glow.radius_px, 1.0, 20.0)
                imgui.pop_item_width()
                imgui.tree_pop()

        imgui.end()

    def _draw_profiler_panel(self):
        imgui.set_next_window_size(300, 200, imgui.FIRST_USE_EVER)
        imgui.begin("Profiler", True)
        
        if self.state.cpu_frame_times:
            avg_ms = sum(self.state.cpu_frame_times) / len(self.state.cpu_frame_times) * 1000.0
            imgui.text(f"Avg CPU Frame: {avg_ms:5.2f} ms")
            
            # Sparkline
            import numpy as np
            history = np.array(self.state.cpu_frame_times, dtype=np.float32)
            imgui.plot_lines("##FPSGraph", history, overlay_text=f"{1000.0/avg_ms:.1f} FPS", 
                             scale_min=0, scale_max=0.033, graph_size=(0, 60))
        
        imgui.separator()
        for k, v in self.state.gpu_timings.items():
            imgui.text(f"{k}: {v*1000.0:5.2f} ms")
            
        imgui.end()

    def _draw_selection_panel(self):
        imgui.set_next_window_size(250, 180, imgui.FIRST_USE_EVER)
        imgui.begin("Selection Info", True)
        
        info = self.state.selected_object
        imgui.text_colored(f"Type: {info.get('type', 'N/A').upper()}", 1.0, 0.8, 0.4)
        imgui.text(f"Dataset: {info.get('dataset_idx', 0)}")
        imgui.text(f"Index: {info.get('element_idx', 0)}")
        imgui.separator()
        imgui.text(f"X: {info.get('x', 0.0):.6f}")
        imgui.text(f"Y: {info.get('y', 0.0):.6f}")
        
        if imgui.button("Isolate"): 
            # Logic later
            pass
        imgui.same_line()
        if imgui.button("Reset"): self.state.selected_object = None

        imgui.end()

    def _draw_analysis_panel(self):
        imgui.set_next_window_size(400, 300, imgui.FIRST_USE_EVER)
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
