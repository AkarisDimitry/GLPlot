from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

try:
    import imgui

    IMGUI_AVAILABLE = True
except ImportError:
    IMGUI_AVAILABLE = False

if TYPE_CHECKING:
    from ..core.context import RenderContext
    from ..core.layers import BaseLayer, TextLayer
    from ..options import EngineOptions


class TextRenderer:
    """
    Primitive renderer for TextLayer.
    Uses ImGui's background draw list to render proyected labels.
    """

    def __init__(self, options: EngineOptions) -> None:
        self.options = options

    def initialize(self) -> None:
        pass

    def draw(self, layer: TextLayer, ctx: RenderContext) -> None:
        if not IMGUI_AVAILABLE or not layer.style.visible:
            return

        # 1. Project world coordinate to NDC
        # Use the MVP matrix provided in the context
        tx, ty = layer.translation
        pos_world = np.array([layer.x + tx, layer.y + ty, 0.0, 1.0], dtype=np.float32)
        pos_ndc = ctx.mvp @ pos_world

        # Perspective divide
        if pos_ndc[3] != 0:
            pos_ndc /= pos_ndc[3]

        # Clipping: if outside NDC [-1, 1], don't draw
        if abs(pos_ndc[0]) > 1.1 or abs(pos_ndc[1]) > 1.1:
            return

        # 2. Convert NDC to screen-space (pixels)
        # NDC (-1, 1) -> Screen (0, Width)
        # Note: y is flipped in screen-space
        screen_x = (pos_ndc[0] + 1.0) * 0.5 * ctx.width_px
        screen_y = (1.0 - pos_ndc[1]) * 0.5 * ctx.height_px

        # 3. Draw using ImGui
        draw_list = imgui.get_background_draw_list()

        color = layer.style.color if layer.style.color is not None else (0.0, 0.0, 0.0, 1.0)
        # ImGui expects packed color or separate components depending on the call
        # imgui.get_color_u32_rgba takes (r, g, b, a) in 0..1
        u32_color = imgui.get_color_u32_rgba(*color)

        # In V1, we just use the default font.
        # Future enhancement: custom font sizes via pusher/pop
        draw_list.add_text(screen_x, screen_y, u32_color, layer.text)

    def draw_all(self, layers: List[BaseLayer], ctx: RenderContext) -> None:
        """Helper to draw all text layers in the scene."""
        if not IMGUI_AVAILABLE:
            return
        for l in layers:
            if l.layer_type == "text":
                self.draw(l, ctx)

    def draw_density(self, layer: TextLayer, ctx: RenderContext) -> None:
        # Text does not participate in density
        pass
