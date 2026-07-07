from __future__ import annotations

import ctypes as C
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from OpenGL.GL import *

from ..utils.gl_utils import link_program
from ..utils.shaders import DENSITY_ACCUM_FS, EXACT_LINES_FS, WIDE_LINES_INSTANCED_VS
from .base import GLLineBuffers

if TYPE_CHECKING:
    from ..core.context import RenderContext
    from ..core.layers import LineFamilyLayer
    from ..options import EngineOptions


class LineFamilyRenderer:
    """
    Primitive renderer for LineFamilyLayer.
    Specialized for high-performance instanced rendering of lines via quad expansion.
    Uses optimized orthographic shaders for maximum throughput.
    """

    def __init__(self, options: EngineOptions) -> None:
        self.options = options
        self.prog = 0

        # Uniform locations (Exact)
        self.u_ndc_scale = -1
        self.u_ndc_offset = -1
        self.u_xrange = -1
        self.u_window = -1
        self.u_use_color = -1
        self.u_alpha = -1
        self.u_width = -1
        self.u_viewport = -1
        self.u_keep_prob = -1
        self.u_total_count = -1
        self.u_offset = -1
        self.u_use_colormap = -1
        self.u_scheme = -1
        self.u_antialiasing = -1

        # Accumulation uniforms
        self.accum_prog = 0
        self.u_accum_ndc_scale = -1
        self.u_accum_ndc_offset = -1
        self.u_accum_xrange = -1
        self.u_accum_window = -1
        self.u_accum_use_color = -1
        self.u_accum_alpha = -1
        self.u_accum_width = -1
        self.u_accum_viewport = -1
        self.u_accum_keep_prob = -1
        self.u_accum_offset = -1

        # Configuration
        self.use_fp16_ab = True
        self.use_packed_color = True

    def initialize(self) -> None:
        """Link wide-line instanced shaders and setup uniform locations."""
        self.prog = link_program(WIDE_LINES_INSTANCED_VS, EXACT_LINES_FS)
        self.u_ndc_scale = glGetUniformLocation(self.prog, "u_ndc_scale")
        self.u_ndc_offset = glGetUniformLocation(self.prog, "u_ndc_offset")
        self.u_xrange = glGetUniformLocation(self.prog, "u_xrange")
        self.u_window = glGetUniformLocation(self.prog, "u_window")
        self.u_use_color = glGetUniformLocation(self.prog, "u_use_color")
        self.u_alpha = glGetUniformLocation(self.prog, "u_alpha")
        self.u_width = glGetUniformLocation(self.prog, "u_width")
        self.u_viewport = glGetUniformLocation(self.prog, "u_viewport_size")
        self.u_keep_prob = glGetUniformLocation(self.prog, "u_keep_prob")
        self.u_total_count = glGetUniformLocation(self.prog, "u_total_count")
        self.u_offset = glGetUniformLocation(self.prog, "u_layer_offset")
        self.u_use_colormap = glGetUniformLocation(self.prog, "u_use_colormap")
        self.u_scheme = glGetUniformLocation(self.prog, "u_scheme")
        self.u_antialiasing = glGetUniformLocation(self.prog, "u_antialiasing")

        # Density Accumulation Program
        self.accum_prog = link_program(WIDE_LINES_INSTANCED_VS, DENSITY_ACCUM_FS)
        self.u_accum_ndc_scale = glGetUniformLocation(self.accum_prog, "u_ndc_scale")
        self.u_accum_ndc_offset = glGetUniformLocation(self.accum_prog, "u_ndc_offset")
        self.u_accum_xrange = glGetUniformLocation(self.accum_prog, "u_xrange")
        self.u_accum_window = glGetUniformLocation(self.accum_prog, "u_window")
        self.u_accum_use_color = glGetUniformLocation(self.accum_prog, "u_use_color")
        self.u_accum_alpha = glGetUniformLocation(self.accum_prog, "u_alpha")
        self.u_accum_width = glGetUniformLocation(self.accum_prog, "u_width")
        self.u_accum_viewport = glGetUniformLocation(self.accum_prog, "u_viewport_size")
        self.u_accum_keep_prob = glGetUniformLocation(self.accum_prog, "u_keep_prob")
        self.u_accum_offset = glGetUniformLocation(self.accum_prog, "u_layer_offset")

    def _create_buffers(self, layer: LineFamilyLayer) -> GLLineBuffers:
        """Create and initialize GPU buffers for a quad-expanded line family."""
        bufs = GLLineBuffers()
        bufs.vao = glGenVertexArrays(1)
        glBindVertexArray(bufs.vao)

        # 1. Base vertex buffer (a_corner: [t, side] x 4)
        # t in {0,1}, side in {-0.5, 0.5}
        bufs.vbo_base = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_base)
        quad_data = np.array([0.0, -0.5, 0.0, 0.5, 1.0, -0.5, 1.0, 0.5], dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, quad_data.nbytes, quad_data, GL_STATIC_DRAW)

        # a_corner (loc 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, C.c_void_p(0))

        # 2. Instanced line parameters (a, b) (loc 1)
        bufs.vbo_ab = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_ab)
        glBufferData(GL_ARRAY_BUFFER, 16, None, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        if self.use_fp16_ab:
            glVertexAttribPointer(1, 2, GL_HALF_FLOAT, GL_FALSE, 0, C.c_void_p(0))
        else:
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
        glVertexAttribDivisor(1, 1)

        # 3. Instanced colors (loc 2)
        bufs.vbo_col = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_col)
        glBufferData(GL_ARRAY_BUFFER, 16, None, GL_STATIC_DRAW)
        glEnableVertexAttribArray(2)
        if self.use_packed_color:
            glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, C.c_void_p(0))
        else:
            glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
        glVertexAttribDivisor(2, 1)

        glBindVertexArray(0)
        return bufs

    def update_gpu_data(self, layer: LineFamilyLayer, bufs: GLLineBuffers) -> None:
        """Upload semantic data to GPU buffers."""
        if layer.ab is None:
            return

        glBindVertexArray(bufs.vao)

        # Upload AB
        ab_u = layer.ab.astype(np.float16) if self.use_fp16_ab else layer.ab
        glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_ab)
        glBufferData(GL_ARRAY_BUFFER, ab_u.nbytes, ab_u, GL_STATIC_DRAW)

        # Upload Colors
        bufs.has_color = layer.colors is not None
        if bufs.has_color:
            if self.use_packed_color:
                cols_u8 = np.clip(layer.colors * 255.0, 0, 255).astype(np.uint8, copy=False)
                glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_col)
                glBufferData(GL_ARRAY_BUFFER, cols_u8.nbytes, cols_u8, GL_STATIC_DRAW)
            else:
                glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_col)
                glBufferData(GL_ARRAY_BUFFER, layer.colors.nbytes, layer.colors, GL_STATIC_DRAW)

        glBindVertexArray(0)
        layer.dirty.gpu_dirty = False

    def draw(self, layer: LineFamilyLayer, ctx: RenderContext) -> None:
        """Draw the layer using current context."""
        if layer.ab is None or len(layer.ab) == 0:
            return

        # 1. Resource Management
        if not hasattr(layer, "_gl") or layer._gl is None:
            layer._gl = self._create_buffers(layer)
            layer.dirty.gpu_dirty = True

        if layer.dirty.gpu_dirty:
            self.update_gpu_data(layer, layer._gl)

        # 2. Setup Shaders & Uniforms
        glUseProgram(self.prog)
        glUniform2f(self.u_ndc_scale, *ctx.ndc_scale)
        glUniform2f(self.u_ndc_offset, *ctx.ndc_offset)
        glUniform2f(self.u_viewport, float(ctx.fb_width), float(ctx.fb_height))
        glUniform2f(self.u_xrange, float(layer.x_range[0]), float(layer.x_range[1]))
        glUniform4f(self.u_window, *ctx.window_world)
        glUniform1i(self.u_use_color, 1 if layer._gl.has_color else 0)

        # Style Application
        overrides = self.options.visual.overrides
        alpha = ctx.global_alpha * layer.style.alpha * overrides.alpha_multiplier
        glUniform1f(self.u_alpha, float(alpha))

        width = layer.style.line_width * overrides.line_width_multiplier
        glUniform1f(self.u_width, float(width))

        # LOD
        prob = ctx.lod_keep_prob
        glUniform1f(self.u_keep_prob, float(prob))
        glUniform1i(self.u_total_count, len(layer.ab))
        glUniform2f(self.u_offset, *layer.translation)

        # Colormap
        glUniform1i(self.u_use_colormap, 1 if self.options.line_colormap_enabled else 0)
        glUniform1i(self.u_scheme, self.options.density_scheme_index)
        # Antialiasing
        glUniform1i(self.u_antialiasing, 1 if self.options.enable_antialiasing else 0)
        # 3. Draw call (Instanced Quads)
        glBindVertexArray(layer._gl.vao)
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, len(layer.ab))

        # Cleanup
        glBindVertexArray(0)
        glUseProgram(0)

    def draw_density(self, layer: LineFamilyLayer, ctx: RenderContext) -> None:
        """Accumulate line density into the current R32F target."""
        if layer.ab is None or len(layer.ab) == 0:
            return

        # 1. Resource Management
        if not hasattr(layer, "_gl") or layer._gl is None:
            layer._gl = self._create_buffers(layer)
            layer.dirty.gpu_dirty = True

        if layer.dirty.gpu_dirty:
            self.update_gpu_data(layer, layer._gl)

        # 2. Setup Shaders
        glUseProgram(self.accum_prog)
        glUniform2f(self.u_accum_ndc_scale, *ctx.ndc_scale)
        glUniform2f(self.u_accum_ndc_offset, *ctx.ndc_offset)
        glUniform2f(self.u_accum_viewport, float(ctx.fb_width), float(ctx.fb_height))
        glUniform2f(self.u_accum_xrange, float(layer.x_range[0]), float(layer.x_range[1]))
        glUniform4f(self.u_accum_window, *ctx.window_world)

        overrides = self.options.visual.overrides
        width = max(1.0, layer.style.line_width * overrides.line_width_multiplier)
        glUniform1f(self.u_accum_width, float(width))

        if self.options.density_weighted:
            glUniform1i(self.u_accum_use_color, 1 if layer._gl.has_color else 0)
            alpha = ctx.global_alpha * layer.style.alpha * overrides.alpha_multiplier
            glUniform1f(self.u_accum_alpha, float(alpha))
        else:
            glUniform1i(self.u_accum_use_color, 0)
            glUniform1f(self.u_accum_alpha, 1.0)

        # LOD
        prob = ctx.lod_keep_prob
        glUniform1f(self.u_accum_keep_prob, float(prob))
        glUniform2f(self.u_accum_offset, *layer.translation)

        # 3. Draw call
        glBindVertexArray(layer._gl.vao)
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, len(layer.ab))

        # Cleanup
        glBindVertexArray(0)
        glUseProgram(0)
