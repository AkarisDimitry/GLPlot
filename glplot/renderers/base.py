from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class GLLineBuffers:
    vao: int = 0
    vbo_base: int = 0
    vbo_ab: int = 0
    vbo_col: int = 0
    has_color: bool = False

    def render(self, count: int) -> None:
        from OpenGL.GL import GL_LINES, glBindVertexArray, glDrawArraysInstanced

        if self.vao and count > 0:
            glBindVertexArray(self.vao)
            glDrawArraysInstanced(GL_LINES, 0, 2, count)
            glBindVertexArray(0)


@dataclass
class GLScatterBuffers:
    vao: int = 0
    vbo_pts: int = 0
    vbo_col: int = 0
    vbo_size: int = 0  # per-point size multipliers (attribute 2); 1.0 when uniform-sized
    vbo_lod: int = 0  # element buffer of decimated indices for zoom-out LOD; 0 = draw all
    lod_stride: int = 1  # decimation stride the LOD index buffer was built at
    lod_count: int = 0  # number of indices in the LOD buffer
    count: int = 0
    size: float = 5.0


@dataclass
class GLStripBuffers:
    vao: int = 0
    vbo_pts: int = 0
    count: int = 0
    color: Tuple[float, float, float, float] = (0, 0, 0, 1)


@dataclass
class GLOffscreenTarget:
    fbo: int = 0
    tex: int = 0
    width: int = 0
    height: int = 0
    depth_rbo: int = 0
