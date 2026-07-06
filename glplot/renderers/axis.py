from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from OpenGL.GL import *

from ..utils.gl_utils import link_program
from ..utils.shaders import STRIP_FS, STRIP_VS

if TYPE_CHECKING:
    from ..core.context import RenderContext
    from ..managers.axis import AxisManager
    from ..options import EngineOptions


class AxisRenderer:
    """
    Specialized renderer for the plot framework: grid, spines, and ticks.
    """

    def __init__(self, options: EngineOptions):
        self.options = options
        self.prog = 0
        self.u_mvp = -1
        self.u_color = -1
        self.u_alpha = -1

        # Temp buffer for line drawing
        self.vbo = 0
        self.vao = 0

    def initialize(self) -> None:
        self.prog = link_program(STRIP_VS, STRIP_FS)
        self.u_mvp = glGetUniformLocation(self.prog, "u_mvp")
        self.u_color = glGetUniformLocation(self.prog, "u_color")
        self.u_alpha = glGetUniformLocation(self.prog, "u_alpha")

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

    def draw(self, axis: AxisManager, ctx: RenderContext) -> None:
        # Check overall visibility
        if not any(
            [
                self.options.axis_show_grid,
                self.options.axis_show_frame,
                self.options.axis_show_labels,
            ]
        ):
            return

        glUseProgram(self.prog)
        glUniformMatrix4fv(self.u_mvp, 1, GL_TRUE, ctx.mvp)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        win = ctx.window_world

        # Get background color to calculate contrast
        if ctx.is_density:
            from ..utils.shaders import get_colormap_min_color

            scheme_idx = getattr(self.options, "density_scheme_index", 0)
            invert = getattr(self.options, "density_invert", False)
            ltc = getattr(self.options, "density_light_to_color", True)
            bg_color = get_colormap_min_color(scheme_idx, invert, ltc)
        else:
            v = self.options.visual.gradient_background
            if v.enabled:
                bg_color = (
                    0.5 * (v.top_color[0] + v.bottom_color[0]),
                    0.5 * (v.top_color[1] + v.bottom_color[1]),
                    0.5 * (v.top_color[2] + v.bottom_color[2]),
                )
            else:
                bg_color = self.options.visual.background_color

        lum = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
        is_light = lum > 0.5

        # 1. Draw Grid
        if self.options.axis_show_grid:
            c = (0.2, 0.2, 0.2) if is_light else (0.8, 0.8, 0.8)
            glUniform4f(self.u_color, c[0], c[1], c[2], self.options.axis_grid_alpha)
            glUniform1f(self.u_alpha, 1.0)

            grid_lines = []
            # Vertical lines (X-ticks)
            for x in axis.ticks_x.major:
                grid_lines.extend([(x, win[2]), (x, win[3])])
            # Horizontal lines (Y-ticks)
            for y in axis.ticks_y.major:
                grid_lines.extend([(win[0], y), (win[1], y)])

            if grid_lines:
                data = np.array(grid_lines, dtype=np.float32)
                glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STREAM_DRAW)
                glDrawArrays(GL_LINES, 0, len(grid_lines))

        # 2. Draw Spines (Frame)
        if self.options.axis_show_frame:
            c_spine = (0.15, 0.15, 0.15, 1.0) if is_light else (0.85, 0.85, 0.85, 1.0)
            glUniform4f(self.u_color, c_spine[0], c_spine[1], c_spine[2], c_spine[3])
            frame = [
                (win[0], win[2]),
                (win[1], win[2]),
                (win[1], win[2]),
                (win[1], win[3]),
                (win[1], win[3]),
                (win[0], win[3]),
                (win[0], win[3]),
                (win[0], win[2]),
            ]
            data_frame = np.array(frame, dtype=np.float32)
            glBufferData(GL_ARRAY_BUFFER, data_frame.nbytes, data_frame, GL_STREAM_DRAW)
            glDrawArrays(GL_LINES, 0, 8)

        glBindVertexArray(0)
        glUseProgram(0)

    def _draw_labels(self, axis: AxisManager, ctx: RenderContext) -> None:
        """Draw numeric labels along the axes."""
        try:
            import imgui
        except ImportError:
            return

        draw_list = imgui.get_background_draw_list()

        # Get background color to calculate contrast
        if ctx.is_density:
            from ..utils.shaders import get_colormap_min_color

            scheme_idx = getattr(self.options, "density_scheme_index", 0)
            invert = getattr(self.options, "density_invert", False)
            ltc = getattr(self.options, "density_light_to_color", True)
            bg_color = get_colormap_min_color(scheme_idx, invert, ltc)
        else:
            v = self.options.visual.gradient_background
            if v.enabled:
                bg_color = (
                    0.5 * (v.top_color[0] + v.bottom_color[0]),
                    0.5 * (v.top_color[1] + v.bottom_color[1]),
                    0.5 * (v.top_color[2] + v.bottom_color[2]),
                )
            else:
                bg_color = self.options.visual.background_color

        lum = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
        is_light = lum > 0.5

        if is_light:
            color = imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 1.0)
        else:
            color = imgui.get_color_u32_rgba(0.85, 0.85, 0.85, 1.0)

        win = ctx.window_world

        # Helper to project world to screen
        def project(wx, wy):
            pos_world = np.array([wx, wy, 0.0, 1.0], dtype=np.float32)
            pos_ndc = ctx.mvp @ pos_world
            if pos_ndc[3] != 0:
                pos_ndc /= pos_ndc[3]
            screen_x = (pos_ndc[0] + 1.0) * 0.5 * ctx.width_px
            screen_y = (1.0 - pos_ndc[1]) * 0.5 * ctx.height_px
            return screen_x, screen_y

        # X-Axis Labels (along bottom)
        for val, label in zip(axis.ticks_x.major, axis.ticks_x.labels):
            sx, sy = project(val, win[2])
            # Offset labels slightly below the spine
            draw_list.add_text(sx - 15, sy + 5, color, label)

        # Y-Axis Labels (along left)
        for val, label in zip(axis.ticks_y.major, axis.ticks_y.labels):
            sx, sy = project(win[0], val)
            # Offset labels slightly to the left of the spine
            draw_list.add_text(sx - 45, sy - 7, color, label)
