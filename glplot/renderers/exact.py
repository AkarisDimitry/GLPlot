from __future__ import annotations
import ctypes as C
import numpy as np
from OpenGL.GL import *
from typing import Tuple, TYPE_CHECKING
from ..utils.shaders import EXACT_LINES_VS, EXACT_LINES_FS, SCATTER_VS, SCATTER_FS, STRIP_VS, STRIP_FS
from ..utils.gl_utils import link_program
from ..utils.gl_utils import link_program
from .base import GLLineBuffers, GLScatterBuffers, GLStripBuffers

if TYPE_CHECKING:
    from ..options import EngineOptions
    from ..core.legacy import SceneData, LineDataset, ScatterDataset, StripDataset

class ExactLineRenderer:
    def __init__(self, options: EngineOptions):
        self.options = options
        self.prog = 0
        self.u_mvp = -1
        self.u_xrange = -1
        self.u_window = -1
        self.u_use_color = -1
        self.u_alpha = -1
        self.u_enable_sub = -1
        self.u_total_count = -1
        self.u_antialiasing = -1
        self.buffers = GLLineBuffers()

        self.scatter_prog = 0
        self.u_scat_mvp = -1
        self.u_scat_size = -1
        self.u_scat_alpha = -1

        self.strip_prog = 0
        self.u_strip_mvp = -1
        self.u_strip_color = -1
        self.u_strip_alpha = -1

        self.use_fp16_ab = True
        self.use_packed_color = True

    def initialize(self) -> None:
        self.prog = link_program(EXACT_LINES_VS, EXACT_LINES_FS)
        self.u_mvp = glGetUniformLocation(self.prog, "u_mvp")
        self.u_xrange = glGetUniformLocation(self.prog, "u_xrange")
        self.u_window = glGetUniformLocation(self.prog, "u_window")
        self.u_use_color = glGetUniformLocation(self.prog, "u_use_color")
        self.u_alpha = glGetUniformLocation(self.prog, "u_alpha")
        self.u_enable_sub = glGetUniformLocation(self.prog, "u_enable_subsample")
        self.u_keep_prob = glGetUniformLocation(self.prog, "u_keep_prob")
        self.u_total_count = glGetUniformLocation(self.prog, "u_total_count")
        self.u_use_colormap = glGetUniformLocation(self.prog, "u_use_colormap")
        self.u_scheme = glGetUniformLocation(self.prog, "u_scheme")
        self.u_antialiasing = glGetUniformLocation(self.prog, "u_antialiasing")

        self.scatter_prog = link_program(SCATTER_VS, SCATTER_FS)
        self.u_scat_mvp = glGetUniformLocation(self.scatter_prog, "u_mvp")
        self.u_scat_size = glGetUniformLocation(self.scatter_prog, "u_size")
        self.u_scat_alpha = glGetUniformLocation(self.scatter_prog, "u_alpha")

        self.strip_prog = link_program(STRIP_VS, STRIP_FS)
        self.u_strip_mvp = glGetUniformLocation(self.strip_prog, "u_mvp")
        self.u_strip_color = glGetUniformLocation(self.strip_prog, "u_color")
        self.u_strip_alpha = glGetUniformLocation(self.strip_prog, "u_alpha")

        self.buffers.vao = glGenVertexArrays(1)
        glBindVertexArray(self.buffers.vao)

        self.buffers.vbo_base = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers.vbo_base)
        t = np.array([0.0, 1.0], dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, t.nbytes, t, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))

        self.buffers.vbo_ab = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers.vbo_ab)
        glBufferData(GL_ARRAY_BUFFER, 16, None, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        if self.use_fp16_ab:
            glVertexAttribPointer(1, 2, GL_HALF_FLOAT, GL_FALSE, 0, C.c_void_p(0))
        else:
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
        glVertexAttribDivisor(1, 1)

        self.buffers.vbo_col = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers.vbo_col)
        glBufferData(GL_ARRAY_BUFFER, 16, None, GL_STATIC_DRAW)
        glEnableVertexAttribArray(2)
        if self.use_packed_color:
            glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, C.c_void_p(0))
        else:
            glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
        glVertexAttribDivisor(2, 1)

        glBindVertexArray(0)

    def upload(self, dataset: LineDataset) -> None:
        if dataset.ab is None:
            return
        dataset.validate()
        glBindVertexArray(self.buffers.vao)

        ab_u = dataset.ab.astype(np.float16) if self.use_fp16_ab else dataset.ab
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers.vbo_ab)
        glBufferData(GL_ARRAY_BUFFER, ab_u.nbytes, ab_u, GL_STATIC_DRAW)

        self.buffers.has_color = dataset.colors is not None
        if self.buffers.has_color:
            if self.use_packed_color:
                cols_u8 = np.clip(dataset.colors * 255.0, 0, 255).astype(np.uint8, copy=False)
                glBindBuffer(GL_ARRAY_BUFFER, self.buffers.vbo_col)
                glBufferData(GL_ARRAY_BUFFER, cols_u8.nbytes, cols_u8, GL_STATIC_DRAW)
            else:
                glBindBuffer(GL_ARRAY_BUFFER, self.buffers.vbo_col)
                glBufferData(GL_ARRAY_BUFFER, dataset.colors.nbytes, dataset.colors, GL_STATIC_DRAW)

        glBindVertexArray(0)

    def draw(
        self,
        scene: SceneData,
        mvp: np.ndarray,
        window: Tuple[float, float, float, float],
        global_alpha: float,
        lod_keep_prob: float,
        clipping_enabled: bool,
    ) -> None:
        # 1. Draw main lines
        dataset = scene.lines
        if dataset.ab is not None and dataset.count > 0:
            l, r, b, t = window
            glUseProgram(self.prog)
            glUniformMatrix4fv(self.u_mvp, 1, GL_TRUE, mvp)
            glUniform2f(self.u_xrange, float(dataset.x_range[0]), float(dataset.x_range[1]))
            glUniform4f(self.u_window, l, r, b, t)
            glUniform1i(self.u_use_color, 1 if self.buffers.has_color else 0)
            glUniform1f(self.u_alpha, float(global_alpha))
            glUniform1i(self.u_enable_sub, 1 if lod_keep_prob < 1.0 else 0)
            glUniform1f(self.u_keep_prob, float(lod_keep_prob))
            glUniform1i(self.u_total_count, dataset.count)
            glUniform1i(self.u_use_colormap, 1 if self.options.line_colormap_enabled else 0)
            glUniform1i(self.u_scheme, self.options.density_scheme_index)
            glUniform1i(self.u_antialiasing, 1 if self.options.enable_antialiasing else 0)

            if clipping_enabled:
                glEnable(GL_CLIP_DISTANCE0)
                glEnable(GL_CLIP_DISTANCE1)
                glEnable(GL_CLIP_DISTANCE2)
                glEnable(GL_CLIP_DISTANCE3)
            else:
                glDisable(GL_CLIP_DISTANCE0)
                glDisable(GL_CLIP_DISTANCE1)
                glDisable(GL_CLIP_DISTANCE2)
                glDisable(GL_CLIP_DISTANCE3)

            glBindVertexArray(self.buffers.vao)
            glDrawArraysInstanced(GL_LINES, 0, 2, dataset.count)
            glBindVertexArray(0)
            glUseProgram(0)

        # 2. Draw scatters
        if scene.scatters:
            glUseProgram(self.scatter_prog)
            glEnable(GL_PROGRAM_POINT_SIZE)
            glUniformMatrix4fv(self.u_scat_mvp, 1, GL_TRUE, mvp)
            glUniform1f(self.u_scat_alpha, float(global_alpha))
            for scat in scene.scatters:
                if not hasattr(scat, '_gl'):
                    scat._gl = self._create_scatter_buffers(scat)
                glUniform1f(self.u_scat_size, float(scat.size))
                glBindVertexArray(scat._gl.vao)
                glDrawArrays(GL_POINTS, 0, scat._gl.count)
            glBindVertexArray(0)
            glUseProgram(0)

        # 3. Draw line strips
        if scene.strips:
            glUseProgram(self.strip_prog)
            glUniformMatrix4fv(self.u_strip_mvp, 1, GL_TRUE, mvp)
            glUniform1f(self.u_strip_alpha, float(global_alpha))
            for strip in scene.strips:
                if not hasattr(strip, '_gl'):
                    strip._gl = self._create_strip_buffers(strip)
                glUniform4f(self.u_strip_color, *strip.color)
                glBindVertexArray(strip._gl.vao)
                glDrawArrays(GL_LINE_STRIP, 0, strip._gl.count)
            glBindVertexArray(0)
            glUseProgram(0)

    def _create_scatter_buffers(self, scat: ScatterDataset) -> GLScatterBuffers:
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        vbo_pts = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pts)
        glBufferData(GL_ARRAY_BUFFER, scat.pts.nbytes, scat.pts, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))

        vbo_col = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_col)
        glBufferData(GL_ARRAY_BUFFER, scat.colors.nbytes, scat.colors, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))

        glBindVertexArray(0)
        return GLScatterBuffers(vao=vao, vbo_pts=vbo_pts, vbo_col=vbo_col, count=len(scat.pts), size=scat.size)

    def _create_strip_buffers(self, strip: StripDataset) -> GLStripBuffers:
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        vbo_pts = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pts)
        glBufferData(GL_ARRAY_BUFFER, strip.pts.nbytes, strip.pts, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
        glBindVertexArray(0)
        return GLStripBuffers(vao=vao, vbo_pts=vbo_pts, count=len(strip.pts), color=strip.color)
