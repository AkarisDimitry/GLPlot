from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..engine import GPULinePlot
    from ..options import EngineOptions


@dataclass
class HudState:
    # Visibility
    show_hud: bool = False
    show_status_overlay: bool = True
    show_layers: bool = False
    show_render_controls: bool = False
    show_profiler: bool = False
    show_selection: bool = True
    show_analysis: bool = False

    # Selection info
    selected_object: Optional[dict] = None
    selected_layer_id: Optional[int] = None

    # Internal state for bidirectional sync
    _last_engine_selection: Optional[int] = None
    _last_hud_selection: Optional[int] = None

    # Profiler metrics (Rolling buffers)
    fps_history: collections.deque = field(default_factory=lambda: collections.deque(maxlen=120))
    cpu_frame_times: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=120)
    )
    gpu_timings: Dict[str, float] = field(default_factory=dict)

    # Analysis Cache
    cached_stats: Dict[str, Any] = field(default_factory=dict)
    sampled_histogram_a: Optional[Any] = None
    sampled_histogram_b: Optional[Any] = None

    # Timestamps
    last_medium_update: float = 0.0
    last_slow_update: float = 0.0


class HudController:
    def __init__(self, plot: GPULinePlot) -> None:
        self.plot = plot

    @property
    def options(self) -> EngineOptions:
        return self.plot.options

    def toggle_hud(self) -> None:
        self.plot.set_hud_enabled(not self.options.enable_hud)

    def reset_view(self) -> None:
        self.plot.reset_view()

    def autoscale(self) -> None:
        self.plot.autoscale()

    def toggle_density(self) -> None:
        self.plot.toggle_density()

    def cycle_scheme(self, direction: int = 1) -> None:
        if direction > 0:
            self.plot.next_density_scheme()
        else:
            self.plot.previous_density_scheme()

    def cycle_blending(self) -> None:
        self.plot.cycle_blending_mode()

    def export(self) -> None:
        # Trigger default export
        import time

        self.plot.savefig(f"plot_{int(time.time())}.png")

    def toggle_layer(self, layer_id: str, visible: bool) -> None:
        # Placeholder for real layer management
        # For now, we only have one main set of lines
        pass

    def set_lod_enabled(self, val: bool) -> None:
        self.options.lod_enabled = val
        self.plot.frame.dirty_scene = True
        self.plot.cache.refresh_requested = True

    def set_lod_budget(self, val: float) -> None:
        self.options.lod_target_coverage = val
        self.plot.frame.dirty_scene = True
        self.plot.cache.refresh_requested = True

    def set_density_resolution(self, val: float) -> None:
        self.options.density_resolution_scale = val
        self.plot.rebuild_density_renderer()

    def set_3d_view(self, **kwargs: Any) -> None:
        if hasattr(self.plot, "set_3d_view"):
            self.plot.set_3d_view(**kwargs)

    def reset_3d_view(self) -> None:
        if hasattr(self.plot, "reset_3d_view"):
            self.plot.reset_3d_view()
