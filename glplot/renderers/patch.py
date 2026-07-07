from __future__ import annotations

import ctypes as C
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from OpenGL.GL import *

from ..utils.gl_utils import link_program
from ..utils.shaders import DENSITY_ACCUM_FS, PATCH_FS, PATCH_VS

if TYPE_CHECKING:
    from ..core.context import RenderContext
    from ..core.layers import PatchLayer
    from ..options import EngineOptions


class GLPatchBuffers:
    def __init__(self, vao: int, vbo: int, ebo: Optional[int] = None) -> None:
        self.vao = vao
        self.vbo = vbo
        self.ebo = ebo
        self.count = 0


class PatchRenderer:
    """
    Primitive renderer for PatchLayer.
    Specialized for area fills, bars, and bands using GL_TRIANGLE_STRIP/TRIANGLES.
    """

    def __init__(self, options: EngineOptions) -> None:
        self.options = options
        self.prog = 0
        self.u_mvp = -1
        self.u_color = -1
        self.u_alpha = -1
        self.u_offset = -1
        self.u_window = -1
        self.u_accum_window = -1

    def initialize(self) -> None:
        self.prog = link_program(PATCH_VS, PATCH_FS)
        self.u_mvp = glGetUniformLocation(self.prog, "u_mvp")
        self.u_color = glGetUniformLocation(self.prog, "u_color")
        self.u_alpha = glGetUniformLocation(self.prog, "u_alpha")
        self.u_offset = glGetUniformLocation(self.prog, "u_layer_offset")
        self.u_window = glGetUniformLocation(self.prog, "u_window")

        # Density accumulation (Triangles into R32F)
        self.accum_prog = link_program(PATCH_VS, DENSITY_ACCUM_FS)
        self.u_accum_mvp = glGetUniformLocation(self.accum_prog, "u_mvp")
        self.u_accum_alpha = glGetUniformLocation(self.accum_prog, "u_alpha")
        self.u_accum_weighted = glGetUniformLocation(self.accum_prog, "u_density_weighted")
        self.u_accum_window = glGetUniformLocation(self.accum_prog, "u_window")

    def _create_buffers(self, layer: PatchLayer) -> GLPatchBuffers:
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))

        ebo = None
        if layer.indices is not None:
            ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)

        glBindVertexArray(0)
        return GLPatchBuffers(vao, vbo, ebo)

    def update_gpu_data(self, layer: PatchLayer, bufs: GLPatchBuffers) -> None:
        if layer.vertices is None or len(layer.vertices) == 0:
            return

        glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo)
        glBufferData(GL_ARRAY_BUFFER, layer.vertices.nbytes, layer.vertices, GL_STATIC_DRAW)

        if layer.indices is not None and bufs.ebo is not None:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufs.ebo)
            glBufferData(
                GL_ELEMENT_ARRAY_BUFFER, layer.indices.nbytes, layer.indices, GL_STATIC_DRAW
            )
            bufs.count = len(layer.indices)
        else:
            bufs.count = len(layer.vertices)

        layer.dirty.gpu_dirty = False

    def draw(self, layer: PatchLayer, ctx: RenderContext) -> None:
        if layer.vertices is None or len(layer.vertices) == 0:
            return

        if not hasattr(layer, "_gl") or layer._gl is None:
            layer._gl = self._create_buffers(layer)
            layer.dirty.gpu_dirty = True

        if layer.dirty.gpu_dirty:
            self.update_gpu_data(layer, layer._gl)

        glUseProgram(self.prog)
        glUniformMatrix4fv(self.u_mvp, 1, GL_TRUE, ctx.mvp)
        glUniform4f(self.u_window, *ctx.window_world)
        overrides = self.options.visual.overrides
        alpha = ctx.global_alpha * layer.style.alpha * overrides.alpha_multiplier
        glUniform1f(self.u_alpha, float(alpha))
        glUniform2f(self.u_offset, *layer.translation)

        # Draw face
        if layer.style.face_color is not None:
            glUniform4f(self.u_color, *layer.style.face_color)
            glBindVertexArray(layer._gl.vao)

            mode = GL_TRIANGLE_STRIP
            if layer.mode == "triangles":
                mode = GL_TRIANGLES

            if layer._gl.ebo is not None:
                glDrawElements(mode, layer._gl.count, GL_UNSIGNED_INT, None)
            else:
                glDrawArrays(mode, 0, layer._gl.count)

        glBindVertexArray(0)
        glUseProgram(0)

    def draw_density(self, layer: PatchLayer, ctx: RenderContext) -> None:
        """Accumulate patch area into density heatmap."""
        if layer.vertices is None or len(layer.vertices) == 0:
            return

        # 1. Resource Management
        if not hasattr(layer, "_gl") or layer._gl is None:
            layer._gl = self._create_buffers(layer)
            layer.dirty.gpu_dirty = True
        if layer.dirty.gpu_dirty:
            self.update_gpu_data(layer, layer._gl)

        # 2. Setup Shaders
        glUseProgram(self.accum_prog)
        glUniformMatrix4fv(self.u_accum_mvp, 1, GL_TRUE, ctx.mvp)
        glUniform4f(self.u_accum_window, *ctx.window_world)
        overrides = self.options.visual.overrides
        if self.options.density_weighted:
            glUniform1i(self.u_accum_weighted, 1)
            alpha = ctx.global_alpha * layer.style.alpha * overrides.alpha_multiplier
            glUniform1f(self.u_accum_alpha, float(alpha))
        else:
            glUniform1i(self.u_accum_weighted, 0)
            glUniform1f(self.u_accum_alpha, 1.0)

        # 3. Draw call
        glBindVertexArray(layer._gl.vao)
        mode = GL_TRIANGLE_STRIP
        if layer.mode == "triangles":
            mode = GL_TRIANGLES

        if layer._gl.ebo is not None:
            glDrawElements(mode, layer._gl.count, GL_UNSIGNED_INT, None)
        else:
            glDrawArrays(mode, 0, layer._gl.count)

        glBindVertexArray(0)
        glUseProgram(0)
