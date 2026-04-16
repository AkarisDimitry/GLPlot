from __future__ import annotations
import math
import time
from typing import Optional, Tuple, Any
import numpy as np
import glfw
from OpenGL.GL import *

from .options import EngineOptions, RenderMode, BlendMode
from .policy import RenderPolicyManager
from .core.legacy import (
    SceneData, CameraState, InteractionState, 
    CacheState, FrameState, LineDataset, 
    ScatterDataset, StripDataset
)
from .core.layers import BaseLayer, LineFamilyLayer, ScatterLayer, PolylineLayer, PatchLayer, TextLayer
from .core.context import RenderContext
from .controllers import CameraController
from .renderers.exact import ExactLineRenderer
from .renderers.interaction import InteractionRenderer
from .renderers.density import DensityRenderer
from .managers.hud import HudManager
from .managers.picking import PickingManager
from .utils.export import ExportManager
from .utils.shaders import DENSITY_SCHEMES
from .managers.effects import EffectManager
from .managers.renderer_manager import RendererManager
from .managers.axis import AxisManager


class GPULinePlot:
    def __init__(self, width: int = 1280, height: int = 800, title: str = "GLPlot", options: Optional[EngineOptions] = None):
        self.options = options or EngineOptions(window_width=width, window_height=height, title=title)
        self.policy = RenderPolicyManager(self.options)
        self.scene = SceneData()
        self.camera = CameraState()
        self.interaction = InteractionState()
        self.cache = CacheState()
        self.frame = FrameState()

        self.window = None
        self.width = self.options.window_width
        self.height = self.options.window_height
        self.fb_width = self.options.window_width
        self.fb_height = self.options.window_height

        self.camera_controller = CameraController(self.camera, self.options)
        self.exact_renderer = ExactLineRenderer(self.options)
        self.interaction_renderer = InteractionRenderer(self)
        self.hud = HudManager(self)
        self.picking = PickingManager(self.options)
        self.export = ExportManager(self)
        self.renderer_manager = RendererManager(self)
        self.axis_manager = AxisManager(self)

        self._cpu_line_copy: Optional[np.ndarray] = None
        self._is_test_mode: bool = False
        self.display_density: bool = False
        self.density_renderer = DensityRenderer(self)
        
        self.picked_info: Optional[dict] = None
        self.mouse_world: Optional[Tuple[float, float]] = None
        self._last_perf_t = time.perf_counter()
        
        self.effects = EffectManager(self)
        self._shim_cache: Dict[str, BaseLayer] = {}

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def set_lines_ab(self, ab: np.ndarray, x_range=(-3.0, 3.0), colors: Optional[np.ndarray] = None, label: Optional[str] = None) -> None:
        ab   = np.ascontiguousarray(ab, np.float32)
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
            existing = LineFamilyLayer(
                ab=ab, colors=cols, x_range=x_range, label=layer_label
            )
            self._primary_line_layer = existing
            self.scene.layers.insert(0, existing)   # lines always render first
        else:
            # Update data in-place so the GPU buffers are refreshed next frame
            existing.ab      = ab
            existing.colors  = cols
            existing.x_range = x_range
            existing.dirty.gpu_dirty = True
            if label:
                existing.label = label

        self.frame.dirty_scene = True
        self.frame.dirty_pick  = True
        if self.exact_renderer.buffers.vao:
            self.exact_renderer.upload(self.scene.lines)

    def add_text(self, x: float, y: float, text: str, fontsize: int = 12, color: Optional[Any] = None, label: Optional[str] = None) -> None:
        layer_label = label or f"Text: {text[:10]}"
        layer = TextLayer(x=x, y=y, text=text, label=layer_label)
        layer.style.text_size_px = fontsize
        if color is not None: layer.style.color = color
        self.scene.layers.append(layer)
        self.frame.dirty_ui = True

    def add_scatter(self, x: np.ndarray, y: np.ndarray, colors: np.ndarray, size: float = 6.0, label: Optional[str] = None) -> None:
        pts = np.column_stack([x, y]).astype(np.float32)
        cols = np.ascontiguousarray(colors, np.float32)
        layer_label = label or f"Scatter {len(self.scene.layers)}"
        layer = ScatterLayer(pts=pts, colors=cols, size=size, label=layer_label)
        self.scene.layers.append(layer)
        self.frame.dirty_scene = True

    def add_line_strip(self, x: np.ndarray, y: np.ndarray, color: Tuple[float, float, float, float] = (0,0,0,1), width: float = 1.0, label: Optional[str] = None) -> None:
        pts = np.column_stack([x, y]).astype(np.float32)
        layer_label = label or f"Polyline {len(self.scene.layers)}"
        layer = PolylineLayer(pts=pts, color=color, width=width, label=layer_label)
        self.scene.layers.append(layer)
        self.frame.dirty_scene = True

    def add_patch(self, vertices: np.ndarray, indices: Optional[np.ndarray] = None, mode: str = "strip", face_color: Optional[Tuple] = None, edge_color: Optional[Tuple] = None, label: Optional[str] = None) -> None:
        layer_label = label or f"Patch {len(self.scene.layers)}"
        layer = PatchLayer(vertices=vertices, indices=indices, mode=mode, label=layer_label)
        if face_color is not None: layer.style.face_color = face_color
        if edge_color is not None: layer.style.edge_color = edge_color
        self.scene.layers.append(layer)
        self.frame.dirty_scene = True

    def set_density_enabled(self, enabled: bool) -> None:
        self.display_density = bool(enabled)
        self.frame.dirty_scene = True
        self.cache.refresh_requested = True

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
        self.options.density_scheme_index = (self.options.density_scheme_index + 1) % len(DENSITY_SCHEMES)
        self.frame.dirty_scene = True
        self.cache.refresh_requested = True

    def previous_density_scheme(self) -> None:
        self.options.density_scheme_index = (self.options.density_scheme_index - 1) % len(DENSITY_SCHEMES)
        self.frame.dirty_scene = True
        self.cache.refresh_requested = True

    def toggle_density(self) -> None:
        self.set_density_enabled(not self.display_density)

    def rebuild_density_renderer(self) -> None:
        """Trigger a resource reconstruction for the density engine when scale changes."""
        self.density_renderer.rebuild_target(self.fb_width, self.fb_height)
        self.frame.dirty_scene = True

    def set_view(self, xlim: Optional[Tuple[float, float]] = None, ylim: Optional[Tuple[float, float]] = None) -> None:
        """
        Sets the world-space view limits, mimicking Matplotlib's xlim/ylim.
        Allows independent scaling of X and Y axes (unforcing 1:1 data aspect).
        """
        if xlim is None and ylim is None:
            return

        tx = xlim if xlim is not None else self.get_xlim()
        ty = ylim if ylim is not None else self.get_ylim()
        
        self.camera_controller.fit_bounds(
            tx[0], tx[1], 
            ty[0], ty[1], 
            self.width, self.height
        )
        
        # Flush interaction cache on manual view changes
        self.cache.active = False
        self.cache.capture_window = None
        self.frame.dirty_scene = True
        self.cache.refresh_requested = True

    def get_xlim(self) -> Tuple[float, float]:
        l, r, b, t = self.camera_controller.world_window(self.width, self.height)
        return (float(l), float(r))

    def get_ylim(self) -> Tuple[float, float]:
        l, r, b, t = self.camera_controller.world_window(self.width, self.height)
        return (float(b), float(t))

    def set_hud_enabled(self, enabled: bool) -> None:
        self.options.enable_hud = bool(enabled)
        self.frame.dirty_ui = True

    def set_blending_mode(self, mode: str | BlendMode) -> None:
        if isinstance(mode, str):
            mapping = {
                "auto": BlendMode.AUTO,
                "alpha": BlendMode.ALPHA,
                "on": BlendMode.ALPHA, # Legacy shim
                "additive": BlendMode.ADDITIVE,
                "subtractive": BlendMode.SUBTRACTIVE,
                "screen": BlendMode.SCREEN,
                "off": BlendMode.OFF
            }
            m = mode.lower()
            if m not in mapping:
                raise ValueError("blend mode must be 'auto', 'alpha', 'additive', 'subtractive', 'screen', or 'off'")
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
            BlendMode.OFF
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
        if name == 'extreme':
            self.options.default_line_budget_per_px = 0.5
            self.options.interaction_budget_lines_per_screen_px = 1.0
            self.options.enable_cache_interaction_path = True
            self.options.cache_safe_margin = 0.4
        elif name == 'performance':
            self.options.default_line_budget_per_px = 1.0
            self.options.interaction_budget_lines_per_screen_px = 2.0
            self.options.enable_cache_interaction_path = True
        elif name == 'balanced':
            self.options.default_line_budget_per_px = 5.0
            self.options.interaction_budget_lines_per_screen_px = 5.0
            self.options.enable_cache_interaction_path = True
        elif name == 'quality':
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

    def autoscale(self, axes: str = "both", padding: float = 0.05) -> None:
        """
        Autoscale view to fit all (legacy and layer) data.
        Supports axes="x", "y", or "both".
        """
        layers = self._get_all_layers()
        bounds = self.renderer_manager.get_bounds(layers)
        
        if bounds is None:
             if axes == "both":
                 self.camera_controller.reset_view()
             return

        xmin, xmax, ymin, ymax = bounds
        
        # Apply fractional padding
        dx = (xmax - xmin) * padding
        dy = (ymax - ymin) * padding
        
        # Sane defaults: only apply a fixed buffer if the original data span is zero
        # to prevent division by zero in camera projection.
        if (xmax - xmin) < 1e-9: dx = 0.5
        if (ymax - ymin) < 1e-9: dy = 0.5
        
        self.camera_controller.fit_bounds(
            xmin - dx, xmax + dx, 
            ymin - dy, ymax + dy, 
            self.width, self.height,
            axes=axes
        )
        
        self.cache.active = False
        self.frame.dirty_scene = True

    def reset_view(self) -> None:
        self.camera_controller.reset_view()
        self.frame.dirty_scene = True

    def clear(self) -> None:
        self.scene = SceneData()
        self.frame.dirty_scene = True

    def close(self) -> None:
        if self.window:
            glfw.set_window_should_close(self.window, True)

    def run(self) -> None:
        self._init_window()
        self._init_gl()
        self._init_modules()
        if self._is_test_mode:
            self._update_runtime_policy()
            glViewport(0, 0, self.fb_width, self.fb_height)
            # 1. Clear Frame (Primary Surface)
            c = self.options.visual.background_color
            glClearColor(c[0], c[1], c[2], 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            self._apply_blending_policy()
            self._draw_exact_view()
            glfw.swap_buffers(self.window)
            return

        self._main_loop()

    def savefig(self, filename: str, scale: float = 1.0) -> None:
        """
        Public API for saving high-resolution figures.
        """
        self.export.savefig(filename, scale=scale)

    def save_current_view(self, filename: Optional[str] = None, scale: float = 2.0) -> None:
        # Legacy shim
        fname = filename or f"plot_{int(time.time())}.png"
        self.savefig(fname, scale=scale)

    # --------------------------------------------------------
    # Init
    # --------------------------------------------------------

    def _init_window(self) -> None:
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

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
        self.exact_renderer.initialize()
        self.interaction_renderer.initialize(self.fb_width, self.fb_height)
        self.density_renderer.initialize(self.fb_width, self.fb_height)
        self.hud.initialize(self.window)
        self.picking.initialize(self.fb_width, self.fb_height)
        self.effects.ensure_resources()
        self.renderer_manager.initialize()

        if self.scene.lines.count > 0:
            self.exact_renderer.upload(self.scene.lines)

    # --------------------------------------------------------
    # Frame policies
    # --------------------------------------------------------

    def _update_runtime_policy(self) -> None:
        self.policy.update(self.scene, self.interaction, self.cache)

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
            time=time.perf_counter()
        )
        
        return self.policy.calculate_width_aware_lod(self.scene, ctx)

    def _get_ndc_transform(self, window: Tuple[float, float, float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Calculate scale and offset to transform world coordinates to NDC [-1, 1]."""
        l, r, b, t = window
        rl = r - l
        tb = t - b
        sx = 2.0 / max(rl, 1e-12)
        sy = 2.0 / max(tb, 1e-12)
        ox = -(r + l) / max(rl, 1e-12)
        oy = -(t + b) / max(tb, 1e-12)
        return (sx, sy), (ox, oy)

    def _apply_blending_policy(self) -> None:
        if not self.policy.runtime.blending_enabled:
            glDisable(GL_BLEND)
            return

        glEnable(GL_BLEND)
        glBlendEquation(GL_FUNC_ADD) # Default reset

        from .options import BlendMode
        m = self.options.blend_mode
        if m == BlendMode.ALPHA or m == BlendMode.AUTO:
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        elif m == BlendMode.ADDITIVE:
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        elif m == BlendMode.SUBTRACTIVE:
            glBlendEquation(GL_FUNC_REVERSE_SUBTRACT)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        elif m == BlendMode.SCREEN:
            glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR)

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

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
            time=time.perf_counter()
        )

        # 2. Draw using the new RendererManager (Modular Architecture)
        layers = self._get_all_layers()
        self.axis_manager.update(ctx)
        
        # Always draw Axes/Framework first (unless hidden via HUD)
        self.renderer_manager.draw_axes(self.axis_manager, ctx)
        
        if self.display_density:
             # Modular Density Pass (Lines, Scatters)
             self.renderer_manager.draw_density(layers, ctx)
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
            for i in range(4): glDisable(GL_CLIP_DISTANCE0 + i)
            
        current_window = self.camera_controller.world_window(self.width, self.height)
        if self.options.enable_cache_interaction_path and self.cache.capture_window is not None:
            self.interaction_renderer.draw_cached_impostor(self.cache.capture_window, current_window)
        else:
            self._draw_exact_view()
            
        # Re-enable if needed for next passes (exact view usually enables it anyway)
        if self.options.enable_clipping_optimization:
            for i in range(4): glEnable(GL_CLIP_DISTANCE0 + i)
            
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
        # Transparent background for the cache to allow blending during interaction
        glClearColor(1.0, 1.0, 1.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)

        prob = self._compute_lod_keep_prob()
        base_alpha = self._get_adaptive_alpha(self.scene.lines.count)

        if prob < 1.0:
            base_alpha = 1.0

        ndc_scale, ndc_offset = self._get_ndc_transform(capture_window)

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
            time=time.perf_counter()
        )

        layers = self._get_all_layers()

        if self.display_density:
            self.renderer_manager.draw_density(layers, ctx, target_fbo=target_fbo, target_size=target_size)
        else:
            self._apply_blending_policy()
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
            l < cl + cw * margin or
            r > cr - cw * margin or
            b < cb + ch * margin or
            t > ct - ch * margin
        )

    def _service_deferred_cache_refresh(self) -> None:
        if not self.cache.active:
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
            glfw.poll_events()
            
            # 1. Update Input and State
            self.hud.process_inputs()
            self._update_runtime_policy()

            # 2. Start ImGui frame before ANY rendering/processing happens
            self.hud.begin()

            self._service_deferred_cache_refresh()
            
            # Picking Pass (Deferred).
            # dirty_pick is set on explicit Shift+Click → always honour it.
            # The extra gate only applies to continuous hover-picking when shift is held.
            if self.frame.dirty_pick:
                run_pick = (not self.options.shift_required_for_picking) or \
                           self.interaction.shift_down or \
                           self.interaction.explicit_pick_requested
                if run_pick:
                    self._run_picking_pass()
                self.frame.dirty_pick = False
                self.interaction.explicit_pick_requested = False

            t0 = glfw.get_time()

            self.effects.begin_scene()

            self.effects.draw_background()
            self._apply_blending_policy()

            if self.policy.runtime.current_mode == RenderMode.INTERACTIVE:
                self._draw_interaction_view()
            else:
                self._draw_exact_view()

            # Draw zoom box if active
            if self.interaction.right_drag_active:
                if self.options.enable_clipping_optimization:
                    for i in range(4): glDisable(GL_CLIP_DISTANCE0 + i)
                self._draw_zoom_box()
                
            self.effects.end_scene()

            # Update HUD metrics and Draw
            self._service_hud_metrics(t0)
            
            # Disable world clipping for HUD
            if self.options.enable_clipping_optimization:
                for i in range(4): glDisable(GL_CLIP_DISTANCE0 + i)

            # HUD panels are only updated if HUD is enabled, but begin/end must wrap all
            if self.policy.runtime.hud_enabled_this_frame:
                self.hud.update()
            
            self.hud.end()
            
            # Note: GL state is cleaned up/reset at start of next frame or specific renderers

            glfw.swap_buffers(self.window)
            t1 = glfw.get_time()

            dt = max(t1 - t0, 1e-6)
            self.frame.fps_estimate = 1.0 / dt
            self.frame.last_frame_time = t1
            self.frame.dirty_scene = False
            self.frame.dirty_ui = False

            if self.cache.active and not self.interaction.drag_active and not self.interaction.right_drag_active and t1 >= self.cache.release_deadline:
                self.cache.active = False
                self.frame.dirty_scene = True

        self.effects.shutdown()

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

    def _update_slow_analysis(self):
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
        
        # Screen to NDC [-1, 1]
        x0, y0 = 2.0 * px / self.width - 1.0, 1.0 - 2.0 * py / self.height
        x1, y1 = 2.0 * mx / self.width - 1.0, 1.0 - 2.0 * my / self.height
        
        # We reuse the TextRenderer's unit quad or similar to avoid defining a new VAO just for this.
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
             color = 0x4C3366CC # Abgr: (0.3, 0.4, 0.8, 1.0) approx
             draw_list.add_rect_filled(px, py, mx, my, color)
             draw_list.add_rect(px, py, mx, my, 0xCC3366CC)

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    def _on_resize(self, window, w, h) -> None:
        self.width = max(1, int(w))
        self.height = max(1, int(h))
        self.frame.dirty_scene = True
        self.frame.dirty_pick = True

    def _on_fb_resize(self, window, w, h) -> None:
        self.fb_width = max(1, int(w))
        self.fb_height = max(1, int(h))
        glViewport(0, 0, self.fb_width, self.fb_height)
        self.interaction_renderer.rebuild_cache_target(self.fb_width, self.fb_height)
        self.density_renderer.rebuild_target(self.fb_width, self.fb_height)
        self.picking.rebuild_target(self.fb_width, self.fb_height)
        self.effects.on_resize()
        self.frame.dirty_scene = True
        self.frame.dirty_pick = True

    def _on_scroll(self, window, dx, dy) -> None:
        self.hud.on_scroll(window, dx, dy)
        if self.hud.wants_mouse():
            return

        if not self.cache.active:
            self.cache.active = True
        self.cache.refresh_requested = True
        self.cache.release_deadline = glfw.get_time() + 0.20

        factor = self.options.zoom_scroll_factor if dy > 0 else 1.0 / self.options.zoom_scroll_factor
        mx, my = glfw.get_cursor_pos(self.window)
        self.camera_controller.apply_zoom_at_cursor(factor, mx, my, self.width, self.height)

        self.frame.dirty_scene = True
        self.frame.dirty_pick = True

    def _on_mouse_button(self, window, button, action, mods) -> None:
        self.hud.on_mouse_button(window, button, action, mods)
        if self.hud.wants_mouse():
            return

        mx, my = glfw.get_cursor_pos(self.window)
        
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                # 1. Picking Pass (Shift + Click)
                if (mods & glfw.MOD_SHIFT) or self.interaction.shift_down:
                    self.interaction.last_mouse = (mx, my)
                    self.frame.dirty_pick = True
                    self.interaction.explicit_pick_requested = True
                    self.frame.dirty_scene = True
                
                # 2. Start Drag State
                self.interaction.drag_active = True
                self.interaction.drag_confirmed = False
                self.interaction.drag_start_translation = None
                self.interaction.press_mouse = (mx, my)
                self.interaction.last_mouse = (mx, my)
                self.interaction.drag_start_world = self.camera_controller.screen_to_world(mx, my, self.width, self.height)
                
                # 3. Determine Drag Mode
                if (mods & glfw.MOD_CONTROL):
                    # For professional UX, resolve the mode ONCE at the start of the drag
                    self._run_picking_pass()
                    
                    if self.interaction.selected_layer_id is not None:
                        self.interaction.drag_mode = "move"
                        layer = next((l for l in self.scene.layers if l.layer_id == self.interaction.selected_layer_id), None)
                        if layer:
                            self.interaction.drag_start_translation = layer.translation
                    else:
                        # Scientific ratio scaling mode
                        self.interaction.drag_mode = "ratio"
                        self.interaction.drag_start_zoom_x = self.camera.zoom_x
                        self.interaction.drag_start_zoom_y = self.camera.zoom_y
                elif (mods & glfw.MOD_SHIFT):
                    self.interaction.drag_mode = "move"
                    if self.interaction.selected_layer_id is not None:
                        layer = next((l for l in self.scene.layers if l.layer_id == self.interaction.selected_layer_id), None)
                        if layer:
                            self.interaction.drag_start_translation = layer.translation
                else:
                    self.interaction.drag_mode = "pan"

            elif action == glfw.RELEASE:
                self.interaction.drag_active = False
                if self.cache.active:
                    self.cache.release_deadline = glfw.get_time() + 0.05
                self.frame.dirty_scene = True

        elif button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                self.interaction.right_drag_active = True
                self.interaction.right_press_mouse = (mx, my)
                self.interaction.last_mouse = (mx, my)
            elif action == glfw.RELEASE:
                if self.interaction.right_drag_active:
                    px, py = self.interaction.right_press_mouse
                    if abs(mx - px) > 5 and abs(my - py) > 5:
                        w0, h0 = self.camera_controller.screen_to_world(px, py, self.width, self.height)
                        w1, h1 = self.camera_controller.screen_to_world(mx, my, self.width, self.height)
                        self.set_view(xlim=(min(w0, w1), max(w0, w1)), ylim=(min(h0, h1), max(h0, h1)))
                self.interaction.right_drag_active = False
                self.frame.dirty_scene = True

    def _on_cursor(self, window, x, y) -> None:
        if self.hud.wants_mouse():
            # Still update world coords for status panel
            self.mouse_world = self.camera_controller.screen_to_world(x, y, self.width, self.height)
            return

        self.mouse_world = self.camera_controller.screen_to_world(x, y, self.width, self.height)
        self.frame.dirty_ui = True

        if self.interaction.drag_active:
            px, py = self.interaction.press_mouse
            dist2 = (x - px) ** 2 + (y - py) ** 2
            if not self.interaction.drag_confirmed and dist2 > self.options.drag_threshold_px ** 2:
                self.interaction.drag_confirmed = True
                self.cache.active = True
                self.cache.refresh_requested = True
                self.cache.release_deadline = glfw.get_time() + 0.20

            if self.interaction.drag_mode == "move" and self.interaction.selected_layer_id is not None:
                # MOVE MODE: Translate the layer
                layer = next((l for l in self.scene.layers if l.layer_id == self.interaction.selected_layer_id), None)
                if layer:
                    if self.interaction.drag_start_translation is None:
                         self.interaction.drag_start_translation = layer.translation
                         self.interaction.drag_start_world = self.camera_controller.screen_to_world(x, y, self.width, self.height)

                    curr_world = self.camera_controller.screen_to_world(x, y, self.width, self.height)
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
                self.camera.zoom_x = self.interaction.drag_start_zoom_x * (2.0 ** (dx * sensitivity))
                self.camera.zoom_y = self.interaction.drag_start_zoom_y * (2.0 ** (-dy * sensitivity)) 
                
                # Clamp to safe camera limits
                self.camera.zoom_x = float(np.clip(self.camera.zoom_x, self.camera.zoom_min, self.camera.zoom_max))
                self.camera.zoom_y = float(np.clip(self.camera.zoom_y, self.camera.zoom_min, self.camera.zoom_max))
                self.cache.refresh_requested = True
            else:
                # PAN MODE: Translate the camera
                lx, ly = self.interaction.last_mouse
                wx0, wy0 = self.camera_controller.screen_to_world(lx, ly, self.width, self.height)
                wx1, wy1 = self.camera_controller.screen_to_world(x, y, self.width, self.height)
                self.camera.cx -= (wx1 - wx0)
                self.camera.cy -= (wy1 - wy0)
                
            self.interaction.last_mouse = (x, y)
            self.frame.dirty_scene = True
            if self.cache.active and self._cache_needs_refresh():
                self.cache.refresh_requested = True
        elif self.interaction.right_drag_active:
            self.interaction.last_mouse = (x, y)
            self.frame.dirty_ui = True

    def _run_picking_pass(self) -> None:
        if not self.interaction.last_mouse:
            return
        
        mx, my = self.interaction.last_mouse
        mvp = self.camera_controller.mvp(self.width, self.height)
        window = self.camera_controller.world_window(self.width, self.height)
        
        # Scale to framebuffer (pixel) coordinates for Retina / High-DPI displays.
        # GLFW cursor positions are in logical window units; the picking FBO is in pixels.
        dpr_x = self.fb_width  / max(self.width,  1)
        dpr_y = self.fb_height / max(self.height, 1)
        px = mx * dpr_x
        py = my * dpr_y
        
        # 1. Render scene to picking buffer
        self.picking.draw_pick_scene(self.scene, self.exact_renderer.buffers, mvp, window)
        
        # 2. Read back hit result at cursor (in pixel coords)
        hit = self.picking.pick_readback(px, py, self.scene)
        
        if hit:
            self.picked_info = {
                "type": hit["type"],
                "layer_id": hit["layer_id"],
                "element_idx": hit["element_idx"],
                "layer": hit["layer"],
                "x": self.mouse_world[0] if self.mouse_world else 0.0,
                "y": self.mouse_world[1] if self.mouse_world else 0.0
            }
            # Update interaction selection
            self.interaction.selected_layer_id = hit["layer_id"]
            
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
    def get_xlim(self) -> Tuple[float, float]:
        l, r, _, _ = self.camera_controller.world_window(self.width, self.height)
        return l, r

    def get_ylim(self) -> Tuple[float, float]:
        _, _, b, t = self.camera_controller.world_window(self.width, self.height)
        return b, t

    def savefig(self, filename: str, scale: float = 2.0) -> None:
        self.export.savefig(filename, scale=scale)

    def _create_rgba_fbo(self, width: int, height: int) -> Tuple[int, int]:
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
        
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glDeleteFramebuffers(1, [fbo])
            glDeleteTextures(1, [tex])
            raise RuntimeError("Failed to create RGBA export framebuffer")
        return fbo, tex

    def capture_snapshot(
        self,
        scale: float = 1.0,
        transparent: bool = True,
        include_axes: bool = False,
        include_postfx: bool = True,
        preserve_screen_space_styles: bool = True
    ) -> "GLPlotSnapshot":
        """
        Level 1 API: Capture the current viewport as a raster image + extent.
        Ensures perfect GL state restoration.
        """
        from .utils.mpl_bridge import GLPlotSnapshot
        
        target_w = max(1, int(round(self.fb_width * scale)))
        target_h = max(1, int(round(self.fb_height * scale)))

        # Capture state to restore
        prev_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
        prev_viewport = glGetIntegerv(GL_VIEWPORT)
        prev_clear_col = glGetFloatv(GL_COLOR_CLEAR_VALUE)

        fbo, tex = self._create_rgba_fbo(target_w, target_h)
        
        xmin, xmax = self.get_xlim()
        ymin, ymax = self.get_ylim()
        window = (xmin, xmax, ymin, ymax)

        try:
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            glViewport(0, 0, target_w, target_h)
            
            if transparent:
                glClearColor(0.0, 0.0, 0.0, 0.0)
            else:
                c = self.options.visual.background_color
                glClearColor(c[0], c[1], c[2], 1.0)
            glClear(GL_COLOR_BUFFER_BIT)

            # Style scaling for high-res
            style_scale = scale if preserve_screen_space_styles else 1.0
            
            mvp = self.camera_controller.mvp(self.width, self.height)
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
                time=time.perf_counter()
            )

            self._apply_blending_policy()
            layers = self._get_all_layers()

            if include_axes:
                self.axis_manager.update(ctx)
                self.renderer_manager.draw_axes(self.axis_manager, ctx)

            # Pass to modular managers
            if self.display_density:
                self.renderer_manager.draw_density(layers, ctx, target_fbo=fbo, target_size=(target_w, target_h))
            else:
                self.renderer_manager.draw_exact(layers, ctx)

            # Text overlay
            self.renderer_manager.renderers["text"].draw_all(layers, ctx)

            glReadBuffer(GL_COLOR_ATTACHMENT0)
            glPixelStorei(GL_PACK_ALIGNMENT, 1)
            raw = glReadPixels(0, 0, target_w, target_h, GL_RGBA, GL_UNSIGNED_BYTE)
            rgba = np.frombuffer(raw, dtype=np.uint8).reshape((target_h, target_w, 4))
            rgba = np.flipud(rgba)

        finally:
            glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
            glViewport(*prev_viewport)
            glClearColor(*prev_clear_col)
            glDeleteFramebuffers(1, [fbo])
            glDeleteTextures(1, [tex])

        return GLPlotSnapshot(
            rgba=rgba,
            extent=window,
            xlim=(xmin, xmax),
            ylim=(ymin, ymax),
            width_px=target_w,
            height_px=target_h,
            transparent=transparent
        )

    def to_matplotlib(self, ax: Optional[Axes] = None, mpl_kwargs: dict = {}, **kwargs):
        """Level 2 API: Render and embed directly into Matplotlib."""
        from .utils.mpl_bridge import snapshot_to_matplotlib
        snap = self.capture_snapshot(**kwargs)
        return snapshot_to_matplotlib(snap, ax=ax, **mpl_kwargs)

    def set_matplotlib_transfer_target(self, ax=None, callback=None):
        """Level 3 API Setup: Redirect 'M' key transfers."""
        self._mpl_transfer_ax = ax
        self._mpl_transfer_callback = callback

    def transfer_to_matplotlib_default(self):
        """Default action for Key 'M'."""
        if hasattr(self, "_mpl_transfer_callback") and self._mpl_transfer_callback:
            snap = self.capture_snapshot(scale=2.0)
            self._mpl_transfer_callback(snap)
            return

        import matplotlib.pyplot as plt
        ax = getattr(self, "_mpl_transfer_ax", None)
        fig, ax, artist = self.to_matplotlib(ax=ax, scale=2.0)
        plt.show(block=False)
        fig.canvas.draw_idle()

    def toggle_line_colormap(self) -> None:
        self.options.line_colormap_enabled = not self.options.line_colormap_enabled
        self.frame.dirty_scene = True

    def _on_key(self, window, key, sc, action, mods) -> None:
        self.hud.on_key(window, key, sc, action, mods)

        if action in (glfw.PRESS, glfw.REPEAT):
            shift = (mods & glfw.MOD_SHIFT)
            
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(self.window, True)

            elif key in (glfw.KEY_R, glfw.KEY_HOME):
                self.reset_view()

            elif key == glfw.KEY_D and action == glfw.PRESS:
                self.toggle_density()

            elif key == glfw.KEY_C and action == glfw.PRESS:
                self.toggle_line_colormap()

            # --- Visual Parameters (Arrows) ---
            if key == glfw.KEY_UP:
                if self.display_density:
                    self.options.density_gain *= 1.2
                else:
                    self.options.default_global_alpha = min(1.0, self.options.default_global_alpha * 1.2)
                self.frame.dirty_scene = True
                self.frame.dirty_ui = True
                
            elif key == glfw.KEY_DOWN:
                if self.display_density:
                    self.options.density_gain /= 1.2
                else:
                    self.options.default_global_alpha = max(0.001, self.options.default_global_alpha / 1.2)
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
                    self.height
                )

            elif key == glfw.KEY_MINUS or key == glfw.KEY_KP_SUBTRACT:
                self.camera_controller.apply_zoom_at_cursor(
                    1.0 / self.options.zoom_scroll_factor,
                    self.width * 0.5,
                    self.height * 0.5,
                    self.width,
                    self.height
                )

            elif key == glfw.KEY_B and action == glfw.PRESS:
                self.cycle_blending_mode()

            elif key == glfw.KEY_BACKSLASH and action == glfw.PRESS:
                self.options.enable_auto_alpha = not self.options.enable_auto_alpha
                self.frame.dirty_scene = True

            elif key == glfw.KEY_LEFT_BRACKET and action in (glfw.PRESS, glfw.REPEAT):
                self.options.density_log_scale = max(0.1, self.options.density_log_scale - 0.2)
                self.frame.dirty_scene = True

            elif key == glfw.KEY_RIGHT_BRACKET and action in (glfw.PRESS, glfw.REPEAT):
                self.options.density_log_scale += 0.2
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

    def _on_char(self, window, char) -> None:
        self.hud.on_char(window, char)
