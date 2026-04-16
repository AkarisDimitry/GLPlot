from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING, Optional
import numpy as np

if TYPE_CHECKING:
    from ..options import EngineOptions, RenderMode

@dataclass
class RenderContext:
    """
    Stable rendering state provided to compilers and renderers.
    Ensures consistent projection and policy across a single frame.
    """
    mvp: np.ndarray # 4x4 Ortho projection matrix
    window_world: Tuple[float, float, float, float] # (l, r, b, t) in world units
    
    width_px: int # Screen width
    height_px: int # Screen height
    fb_width: int # Framebuffer width (for multisampling/HighDPI)
    fb_height: int # Framebuffer height
    
    mode: RenderMode # EXACT, INTERACTIVE, or PICKING
    is_density: bool = False # Flag for density/heatmap accumulation mode
    
    # Passing global settings
    global_alpha: float = 1.0
    lod_keep_prob: float = 1.0
    
    # Picking context
    id_offset: int = 0
    
    # Time for animated effects (grain, shimmer, etc)
    time: float = 0.0
    
    # Device Pixel Ratio for HighDPI / Retina consistency
    dpr: float = 1.0

    # Orthographic specialization (skip 4x4 matrix multiplication in hot paths)
    ndc_scale: Tuple[float, float] = (1.0, 1.0)
    ndc_offset: Tuple[float, float] = (0.0, 0.0)

    @property
    def aspect(self) -> float:
        return max(self.width_px, 1) / max(self.height_px, 1)

    def screen_to_world(self, sx: float, sy: float) -> Tuple[float, float]:
        l, r, b, t = self.window_world
        x = l + (sx / self.width_px) * (r - l)
        y = b + ((self.height_px - sy) / self.height_px) * (t - b)
        return x, y
