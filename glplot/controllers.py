from __future__ import annotations
from typing import Tuple, Optional, TYPE_CHECKING
import numpy as np
from .utils.gl_utils import ortho

if TYPE_CHECKING:
    from .core.legacy import CameraState
    from .options import EngineOptions

class CameraController:
    def __init__(self, camera: CameraState, options: EngineOptions):
        self.camera = camera
        self.options = options

    def world_window(self, width: int, height: int, padding: float = 1.0) -> Tuple[float, float, float, float]:
        """
        Returns the world-space bounds (l, r, b, t) of the view, 
        potentially expanded by padding for cache utility.
        """
        half_w = padding / max(self.camera.zoom_x, 1e-12)
        half_h = padding / max(self.camera.zoom_y, 1e-12)
        
        l = self.camera.cx - half_w
        r = self.camera.cx + half_w
        b = self.camera.cy - half_h
        t = self.camera.cy + half_h
        return l, r, b, t

    def mvp(self, width: int, height: int, window: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
        l, r, b, t = window if window is not None else self.world_window(width, height)
        return ortho(l, r, b, t)

    def screen_to_world(self, sx: float, sy: float, width: int, height: int) -> Tuple[float, float]:
        l, r, b, t = self.world_window(width, height)
        x = l + (sx / width) * (r - l)
        y = b + ((height - sy) / height) * (t - b)
        return x, y

    def apply_zoom_at_cursor(self, factor: float, mx: float, my: float, width: int, height: int) -> None:
        """Isotropic zoom centered at cursor position."""
        wx0, wy0 = self.screen_to_world(mx, my, width, height)
        
        self.camera.zoom_x = float(np.clip(self.camera.zoom_x * factor, self.camera.zoom_min, self.camera.zoom_max))
        self.camera.zoom_y = float(np.clip(self.camera.zoom_y * factor, self.camera.zoom_min, self.camera.zoom_max))
        
        wx1, wy1 = self.screen_to_world(mx, my, width, height)
        self.camera.cx += (wx0 - wx1)
        self.camera.cy += (wy0 - wy1)

    def fit_bounds(
        self, 
        xmin: float, xmax: float, 
        ymin: float, ymax: float, 
        width: int, height: int,
        axes: str = "both"
    ) -> None:
        """
        Calculates independent zoom_x and zoom_y to fit the requested bounds.
        Handles zero-span degenerate cases by providing a sane default unit span.
        """
        # Handle degenerate spans
        if abs(xmax - xmin) < 1e-9:
            xmin -= 0.5
            xmax += 0.5
        if abs(ymax - ymin) < 1e-9:
            ymin -= 0.5
            ymax += 0.5

        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        span_x = xmax - xmin
        span_y = ymax - ymin
        
        if "x" in axes or axes == "both":
            self.camera.cx = float(cx)
            self.camera.zoom_x = float(np.clip(2.0 / span_x, self.camera.zoom_min, self.camera.zoom_max))
            
        if "y" in axes or axes == "both":
            self.camera.cy = float(cy)
            self.camera.zoom_y = float(np.clip(2.0 / span_y, self.camera.zoom_min, self.camera.zoom_max))

    def reset_view(self) -> None:
        self.camera.cx = 0.0
        self.camera.cy = 0.0
        self.camera.zoom_x = 1.0
        self.camera.zoom_y = 1.0
