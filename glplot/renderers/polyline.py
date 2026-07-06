from __future__ import annotations
import ctypes as C
import numpy as np
from OpenGL.GL import *
from typing import TYPE_CHECKING
from dataclasses import dataclass

from ..utils.gl_utils import link_program
from ..utils.shaders import (
    WIDE_SEGMENT_INSTANCED_VS,
    WIDE_SEGMENT_INSTANCED_FS,
    WIDE_SEGMENT_DENSITY_FS,
)

if TYPE_CHECKING:
    from ..core.layers import PolylineLayer
    from ..core.context import RenderContext
    from ..options import EngineOptions

@dataclass
class GLWideSegmentBuffers:
    vao: int = 0
    vbo_quad: int = 0
    vbo_inst: int = 0
    ebo: int = 0
    instance_count: int = 0

class PolylineRenderer:
    """
    High-performance renderer for PolylineLayer using GPU Instancing.
    Replaces the older CPU-side expansion logic.
    """
    def __init__(self, options: EngineOptions):
        self.options = options

        self.prog = 0
        self.u_mvp = -1
        self.u_viewport = -1
        self.u_width = -1
        self.u_color = -1
        self.u_alpha = -1
        self.u_offset = -1
        self.u_use_colormap = -1
        self.u_scheme = -1
        self.u_id_norm = -1
        self.u_window = -1

        self.accum_prog = 0
        self.u_acc_mvp = -1
        self.u_acc_viewport = -1
        self.u_acc_width = -1
        self.u_acc_alpha = -1
        self.u_acc_weighted = -1
        self.u_acc_offset = -1
        self.u_acc_window = -1

    def initialize(self) -> None:
        self.prog = link_program(WIDE_SEGMENT_INSTANCED_VS, WIDE_SEGMENT_INSTANCED_FS)
        self.u_mvp = glGetUniformLocation(self.prog, "u_mvp")
        self.u_viewport = glGetUniformLocation(self.prog, "u_viewport_size")
        self.u_width = glGetUniformLocation(self.prog, "u_width")
        self.u_color = glGetUniformLocation(self.prog, "u_color")
        self.u_alpha = glGetUniformLocation(self.prog, "u_alpha")
        self.u_offset = glGetUniformLocation(self.prog, "u_layer_offset")
        self.u_use_colormap = glGetUniformLocation(self.prog, "u_use_colormap")
        self.u_scheme = glGetUniformLocation(self.prog, "u_scheme")
        self.u_id_norm = glGetUniformLocation(self.prog, "u_id_norm")
        self.u_window = glGetUniformLocation(self.prog, "u_window")

        self.accum_prog = link_program(WIDE_SEGMENT_INSTANCED_VS, WIDE_SEGMENT_DENSITY_FS)
        self.u_acc_mvp = glGetUniformLocation(self.accum_prog, "u_mvp")
        self.u_acc_viewport = glGetUniformLocation(self.accum_prog, "u_viewport_size")
        self.u_acc_width = glGetUniformLocation(self.accum_prog, "u_width")
        self.u_acc_alpha = glGetUniformLocation(self.accum_prog, "u_alpha")
        self.u_acc_weighted = glGetUniformLocation(self.accum_prog, "u_density_weighted")
        self.u_acc_offset = glGetUniformLocation(self.accum_prog, "u_layer_offset")
        self.u_acc_window = glGetUniformLocation(self.accum_prog, "u_window")

    def _create_buffers(self, layer: PolylineLayer) -> GLWideSegmentBuffers:
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        # Static quad corners: (t, side)
        # t in [0, 1] goes from start to end of segment
        # side in [-0.5, 0.5] handles expansion around centerline
        quad = np.array([
            [0.0, -0.5],
            [0.0,  0.5],
            [1.0, -0.5],
            [1.0,  0.5],
        ], dtype=np.float32)

        indices = np.array([0, 1, 2,  2, 1, 3], dtype=np.uint16)

        vbo_quad = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_quad)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, C.c_void_p(0))

        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Per-instance segment buffer: [p0.x, p0.y, p1.x, p1.y]
        vbo_inst = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_inst)
        glBufferData(GL_ARRAY_BUFFER, 16, None, GL_STREAM_DRAW) # Initial tiny allocation

        stride = 4 * 4
        # i_p0 (location 1)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, C.c_void_p(0))
        glVertexAttribDivisor(1, 1)

        # i_p1 (location 2)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, C.c_void_p(8))
        glVertexAttribDivisor(2, 1)

        glBindVertexArray(0)
        return GLWideSegmentBuffers(
            vao=vao,
            vbo_quad=vbo_quad,
            vbo_inst=vbo_inst,
            ebo=ebo,
            instance_count=0,
        )

    def update_gpu_data(self, layer: PolylineLayer, bufs: GLWideSegmentBuffers) -> None:
        if layer.pts is None or len(layer.pts) < 2:
            bufs.instance_count = 0
            layer.dirty.gpu_dirty = False
            return

        # Prepare segment data [p0x, p0y, p1x, p1y]
        pts = np.ascontiguousarray(layer.pts, dtype=np.float32)
        segs = np.empty((len(pts) - 1, 4), dtype=np.float32)
        segs[:, 0:2] = pts[:-1]
        segs[:, 2:4] = pts[1:]

        glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_inst)
        
        # Buffer Orphaning for performance
        glBufferData(GL_ARRAY_BUFFER, segs.nbytes, None, GL_STREAM_DRAW)
        glBufferSubData(GL_ARRAY_BUFFER, 0, segs.nbytes, segs)

        bufs.instance_count = len(segs)
        layer.dirty.gpu_dirty = False

    def draw(self, layer: PolylineLayer, ctx: RenderContext, id_norm: float = 0.0) -> None:
        if layer.pts is None or len(layer.pts) < 2:
            return

        if not hasattr(layer, "_gl") or layer._gl is None:
            layer._gl = self._create_buffers(layer)
            layer.dirty.gpu_dirty = True

        if layer.dirty.gpu_dirty:
            self.update_gpu_data(layer, layer._gl)

        overrides = self.options.visual.overrides
        width_px = max(1.0, layer.style.line_width * overrides.line_width_multiplier)
        alpha = ctx.global_alpha * layer.style.alpha * overrides.alpha_multiplier
        color = layer.style.color if layer.style.color is not None else (0.0, 0.0, 0.0, 1.0)

        glUseProgram(self.prog)
        glUniformMatrix4fv(self.u_mvp, 1, GL_TRUE, ctx.mvp)
        glUniform2f(self.u_viewport, float(ctx.fb_width), float(ctx.fb_height))
        glUniform1f(self.u_width, float(width_px))
        glUniform4f(self.u_color, *color)
        glUniform1f(self.u_alpha, float(alpha))
        glUniform2f(self.u_offset, *layer.translation)
        
        glUniform1i(self.u_use_colormap, 1 if self.options.line_colormap_enabled else 0)
        glUniform1i(self.u_scheme, self.options.density_scheme_index)
        glUniform1f(self.u_id_norm, float(id_norm))
        l, r, b, t = ctx.window_world
        glUniform4f(self.u_window, l, r, b, t)

        glBindVertexArray(layer._gl.vao)
        glDrawElementsInstanced(
            GL_TRIANGLES,
            6,
            GL_UNSIGNED_SHORT,
            C.c_void_p(0),
            layer._gl.instance_count
        )
        glBindVertexArray(0)
        glUseProgram(0)

    def draw_density(self, layer: PolylineLayer, ctx: RenderContext) -> None:
        if layer.pts is None or len(layer.pts) < 2:
            return

        if not hasattr(layer, "_gl") or layer._gl is None:
            layer._gl = self._create_buffers(layer)
            layer.dirty.gpu_dirty = True

        if layer.dirty.gpu_dirty:
            self.update_gpu_data(layer, layer._gl)

        overrides = self.options.visual.overrides
        width_px = max(1.0, layer.style.line_width * overrides.line_width_multiplier)

        if self.options.density_weighted:
            alpha = ctx.global_alpha * layer.style.alpha * overrides.alpha_multiplier
            weighted = 1
        else:
            alpha = 1.0
            weighted = 0

        glUseProgram(self.accum_prog)
        glUniformMatrix4fv(self.u_acc_mvp, 1, GL_TRUE, ctx.mvp)
        glUniform2f(self.u_acc_viewport, float(ctx.fb_width), float(ctx.fb_height))
        glUniform1f(self.u_acc_width, float(width_px))
        glUniform1f(self.u_acc_alpha, float(alpha))
        glUniform1i(self.u_acc_weighted, weighted)
        glUniform2f(self.u_acc_offset, *layer.translation)
        l, r, b, t = ctx.window_world
        glUniform4f(self.u_acc_window, l, r, b, t)

        glBindVertexArray(layer._gl.vao)
        glDrawElementsInstanced(
            GL_TRIANGLES,
            6,
            GL_UNSIGNED_SHORT,
            C.c_void_p(0),
            layer._gl.instance_count
        )
        glBindVertexArray(0)
        glUseProgram(0)
