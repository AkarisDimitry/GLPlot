from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Tuple
from .options import RenderMode

if TYPE_CHECKING:
    from .options import EngineOptions
    from .core.legacy import SceneData, InteractionState, CacheState
    from .core.context import RenderContext

class RenderPolicyManager:
    """
    Orchestrates rendering policies including LOD, blending, and density fallbacks.
    Implements 'Width-Aware' scaling to protect against fill-rate bottlenecks.
    """
    def __init__(self, options: EngineOptions):
        from .options import RuntimePolicy
        self.options = options
        self.runtime = RuntimePolicy()

    def update(self, scene: SceneData, interaction: InteractionState, cache: CacheState) -> None:
        n = scene.lines.count

        is_hud_interacting = False
        try:
            import imgui
            if imgui.get_current_context() is not None:
                is_hud_interacting = imgui.is_any_item_active()
        except Exception:
            pass

        if interaction.drag_active or cache.active or is_hud_interacting:
            self.runtime.current_mode = RenderMode.INTERACTIVE
        else:
            self.runtime.current_mode = RenderMode.EXACT

        if self.options.shift_required_for_picking:
            self.runtime.picking_enabled_this_frame = interaction.shift_down and not interaction.drag_active
        else:
            self.runtime.picking_enabled_this_frame = not interaction.drag_active

        if self.options.enable_hud and (self.runtime.current_mode == RenderMode.EXACT or is_hud_interacting):
            self.runtime.hud_enabled_this_frame = True
        else:
            self.runtime.hud_enabled_this_frame = False

        # Blending policy
        from .options import BlendMode
        m = self.options.blend_mode
        if m == BlendMode.OFF:
            self.runtime.blending_enabled = False
        elif m == BlendMode.AUTO:
            self.runtime.blending_enabled = (n <= self.options.auto_disable_blending_threshold)
        else:
            self.runtime.blending_enabled = True

    def estimate_polyline_screen_length_px(self, pts: np.ndarray, ctx: RenderContext, max_samples: int = 4096) -> float:
        """Estimate the total length of a polyline in screen pixels (cheaply)."""
        if pts is None or len(pts) < 2:
            return 0.0

        # Sample long polylines to keep policy update fast
        stride = max(1, (len(pts) - 1) // max_samples)
        sample = pts[::stride]
        if len(sample) < 2:
            return 0.0

        l, r, b, t = ctx.window_world
        # Screen units per world unit
        sx = ctx.fb_width / max(r - l, 1e-12)
        sy = ctx.fb_height / max(t - b, 1e-12)

        diffs = np.diff(sample, axis=0)
        # Transform diffs to px and get length
        seg_px = np.linalg.norm(diffs * np.array([sx, sy], dtype=np.float32)[None, :], axis=1)

        return float(seg_px.sum() * stride)

    def calculate_width_aware_lod(self, scene: SceneData, ctx: RenderContext) -> float:
        """
        Calculates keep_prob based on 'Fill-Rate' budget rather than just primitive count.
        Avoids performance degradation when using thick lines.
        """
        target_coverage = self.options.lod_target_coverage
        total_est_px2 = 0.0
        target_px2 = target_coverage * ctx.fb_width * ctx.fb_height
        
        overrides = self.options.visual.overrides

        # 1. Line Families (Approx coverage: Count * ViewportWidth * Width)
        if scene.lines.ab is not None:
            from .options import RenderMode
            # Relax budget for density mode as overdraw is the goal
            mode_multiplier = 100.0 if ctx.is_density else 1.0
            
            width = self.options.global_line_width * overrides.line_width_multiplier
            est = float(len(scene.lines.ab)) * ctx.fb_width * max(1.0, width)
            total_est_px2 += est / mode_multiplier

        # 2. Polylines (Approx coverage: Length * Width)
        for layer in scene.layers:
            from .core.layers import PolylineLayer
            if isinstance(layer, PolylineLayer):
                width = layer.style.line_width * overrides.line_width_multiplier
                length = self.estimate_polyline_screen_length_px(layer.pts, ctx)
                total_est_px2 += length * max(1.0, width)

        if total_est_px2 <= target_px2:
            return 1.0
            
        return max(0.001, min(1.0, target_px2 / total_est_px2))

    def should_force_density_mode(self, scene: SceneData, ctx: RenderContext, factor: float = 3.0) -> bool:
        """Determines if the scene is so complex that density mode should be forced during interaction."""
        target_px2 = ctx.fb_width * ctx.fb_height * factor
        
        # Reuse LOD calc logic but for a higher threshold
        # (This is a simplified version for V1)
        prob = self.calculate_width_aware_lod(scene, ctx, target_coverage=factor)
        return prob < 1.0
