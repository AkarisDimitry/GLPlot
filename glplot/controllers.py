from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from .utils.gl_utils import ortho

if TYPE_CHECKING:
    from .core.legacy import CameraState
    from .options import EngineOptions


class CameraController:
    def __init__(self, camera: CameraState, options: EngineOptions) -> None:
        self.camera = camera
        self.options = options

    def world_window(
        self, width: int, height: int, padding: float = 1.0
    ) -> Tuple[float, float, float, float]:
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

    def mvp(
        self, width: int, height: int, window: Optional[Tuple[float, float, float, float]] = None
    ) -> np.ndarray:
        l, r, b, t = window if window is not None else self.world_window(width, height)

        # Apply inset margins for axes and labels visibility
        margin_l = 60.0
        margin_r = 20.0
        margin_b = 40.0
        margin_t = 20.0

        rl, tb = (r - l), (t - b)

        # Calculate NDC scale and translation for margins
        w_px = max(width, 1e-12)
        h_px = max(height, 1e-12)
        sx = (width - margin_l - margin_r) / w_px
        tx = (margin_l - margin_r) / w_px
        sy = (height - margin_t - margin_b) / h_px
        ty = (margin_b - margin_t) / h_px

        # Standard ortho values
        m00 = 2.0 / max(rl, 1e-12)
        m03 = -(r + l) / max(rl, 1e-12)
        m11 = 2.0 / max(tb, 1e-12)
        m13 = -(t + b) / max(tb, 1e-12)

        # Apply transformation
        r00 = sx * m00
        r03 = sx * m03 + tx
        r11 = sy * m11
        r13 = sy * m13 + ty

        return np.array(
            [
                [r00, 0.0, 0.0, r03],
                [0.0, r11, 0.0, r13],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def screen_to_world(self, sx: float, sy: float, width: int, height: int) -> Tuple[float, float]:
        l, r, b, t = self.world_window(width, height)

        # Apply inset margins for axes and labels visibility
        margin_l = 60.0
        margin_r = 20.0
        margin_b = 40.0
        margin_t = 20.0

        w_px = max(width, 1e-12)
        h_px = max(height, 1e-12)

        # Scale and translation in NDC space
        scale_x = (width - margin_l - margin_r) / w_px
        trans_x = (margin_l - margin_r) / w_px
        scale_y = (height - margin_t - margin_b) / h_px
        trans_y = (margin_b - margin_t) / h_px

        # Screen to NDC inset
        ndc_x_inset = (sx / w_px) * 2.0 - 1.0
        ndc_y_inset = ((height - sy) / h_px) * 2.0 - 1.0

        # NDC inset to standard NDC
        ndc_x = (ndc_x_inset - trans_x) / max(scale_x, 1e-12)
        ndc_y = (ndc_y_inset - trans_y) / max(scale_y, 1e-12)

        # Standard NDC to world
        x = l + (ndc_x + 1.0) * 0.5 * (r - l)
        y = b + (ndc_y + 1.0) * 0.5 * (t - b)
        return x, y

    def apply_zoom_at_cursor(
        self, factor: float, mx: float, my: float, width: int, height: int
    ) -> None:
        """Isotropic zoom centered at cursor position."""
        wx0, wy0 = self.screen_to_world(mx, my, width, height)

        self.camera.zoom_x = float(
            np.clip(self.camera.zoom_x * factor, self.camera.zoom_min, self.camera.zoom_max)
        )
        self.camera.zoom_y = float(
            np.clip(self.camera.zoom_y * factor, self.camera.zoom_min, self.camera.zoom_max)
        )

        wx1, wy1 = self.screen_to_world(mx, my, width, height)
        self.camera.cx += wx0 - wx1
        self.camera.cy += wy0 - wy1

    def fit_bounds(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        width: int,
        height: int,
        axes: str = "both",
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
            self.camera.zoom_x = float(
                np.clip(2.0 / span_x, self.camera.zoom_min, self.camera.zoom_max)
            )

        if "y" in axes or axes == "both":
            self.camera.cy = float(cy)
            self.camera.zoom_y = float(
                np.clip(2.0 / span_y, self.camera.zoom_min, self.camera.zoom_max)
            )

    def reset_view(self) -> None:
        self.camera.cx = 0.0
        self.camera.cy = 0.0
        self.camera.zoom_x = 1.0
        self.camera.zoom_y = 1.0
