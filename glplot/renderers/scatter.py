from __future__ import annotations

import ctypes as C
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from OpenGL.GL import *

from ..utils.gl_utils import link_program
from ..utils.shaders import (
    DENSITY_POINTS_FS,
    DENSITY_POINTS_VS,
    IMAGE_FS,
    IMAGE_VS,
    SCATTER_FS,
    SCATTER_VS,
)
from .base import GLScatterBuffers

if TYPE_CHECKING:
    from ..core.context import RenderContext
    from ..core.layers import ScatterLayer
    from ..options import EngineOptions


class ScatterRenderer:
    """
    Primitive renderer for ScatterLayer.
    Specialized for point clouds (GL_POINTS).
    """

    def __init__(self, options: EngineOptions) -> None:
        self.options = options
        self.prog = 0

        # Uniform locations
        self.u_mvp = -1
        self.u_size = -1
        self.u_alpha = -1
        self.u_offset = -1

        # Accumulation uniforms
        self.accum_prog = 0
        self.u_accum_mvp = -1
        self.u_accum_size = -1
        self.u_accum_alpha = -1
        self.u_accum_offset = -1

        # Textured-quad program (imshow layers)
        self.image_prog = 0
        self.u_image_mvp = -1
        self.u_image_alpha = -1
        self.u_image_tex = -1

    def initialize(self) -> None:
        """Link shaders and setup uniform locations."""
        self.prog = link_program(SCATTER_VS, SCATTER_FS)
        self.u_mvp = glGetUniformLocation(self.prog, "u_mvp")
        self.u_size = glGetUniformLocation(self.prog, "u_size")
        self.u_alpha = glGetUniformLocation(self.prog, "u_alpha")
        self.u_offset = glGetUniformLocation(self.prog, "u_layer_offset")
        self.u_point_size_px = glGetUniformLocation(self.prog, "u_point_size_px")
        self.u_outline_enabled = glGetUniformLocation(self.prog, "u_outline_enabled")
        self.u_outline_color = glGetUniformLocation(self.prog, "u_outline_color")
        self.u_outline_width_px = glGetUniformLocation(self.prog, "u_outline_width_px")

        # Textured-quad program for imshow layers
        self.image_prog = link_program(IMAGE_VS, IMAGE_FS)
        self.u_image_mvp = glGetUniformLocation(self.image_prog, "u_mvp")
        self.u_image_alpha = glGetUniformLocation(self.image_prog, "u_alpha")
        self.u_image_tex = glGetUniformLocation(self.image_prog, "u_tex")

        # Density Accumulation Program
        self.accum_prog = link_program(DENSITY_POINTS_VS, DENSITY_POINTS_FS)
        self.u_accum_mvp = glGetUniformLocation(self.accum_prog, "u_mvp")
        self.u_accum_size = glGetUniformLocation(self.accum_prog, "u_size")
        self.u_accum_alpha = glGetUniformLocation(self.accum_prog, "u_alpha")
        self.u_accum_offset = glGetUniformLocation(self.accum_prog, "u_layer_offset")

    def _create_buffers(self, layer: ScatterLayer) -> GLScatterBuffers:
        """Create and initialize GPU buffers for a scatter layer."""
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        # 1. Point Positions
        vbo_pts = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pts)
        glBufferData(GL_ARRAY_BUFFER, 16, None, GL_STATIC_DRAW)  # Pre-allocate to avoid segfault
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))

        # 2. Point Colors
        vbo_col = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_col)
        glBufferData(GL_ARRAY_BUFFER, 16, None, GL_STATIC_DRAW)  # Pre-allocate to avoid segfault
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))

        glBindVertexArray(0)
        return GLScatterBuffers(vao=vao, vbo_pts=vbo_pts, vbo_col=vbo_col)

    def update_gpu_data(self, layer: ScatterLayer, bufs: GLScatterBuffers) -> None:
        """Upload semantic points and colors to GPU buffers."""
        if layer.pts is None or len(layer.pts) == 0:
            return

        # Upload Positions
        glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_pts)
        glBufferData(GL_ARRAY_BUFFER, layer.pts.nbytes, layer.pts, GL_STATIC_DRAW)

        # Upload Colors
        if layer.colors is not None:
            # If colors are provided as a single color, broadcast it
            if layer.colors.ndim == 1 and len(layer.colors) == 4:
                cols = np.tile(layer.colors, (len(layer.pts), 1)).astype(np.float32)
            else:
                cols = layer.colors.astype(np.float32)

            glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_col)
            glBufferData(GL_ARRAY_BUFFER, cols.nbytes, cols, GL_STATIC_DRAW)

        bufs.count = len(layer.pts)
        layer.dirty.gpu_dirty = False

    def draw(self, layer: ScatterLayer, ctx: RenderContext) -> None:
        """Draw the scatter layer using current context."""
        if layer.metadata.get("artist") == "imshow":
            self._draw_image(layer, ctx)
            return

        if layer.pts is None or len(layer.pts) == 0:
            return

        # 1. Resource Management
        if not hasattr(layer, "_gl") or layer._gl is None:
            layer._gl = self._create_buffers(layer)
            layer.dirty.gpu_dirty = True

        if layer.dirty.gpu_dirty:
            self.update_gpu_data(layer, layer._gl)

        # 2. Setup OpenGL State & Shaders
        glUseProgram(self.prog)
        glEnable(GL_PROGRAM_POINT_SIZE)

        glUniformMatrix4fv(self.u_mvp, 1, GL_TRUE, ctx.mvp)

        # Style Resolution: Base * Multipliers
        overrides = self.options.visual.overrides
        effective_size = float(layer.style.point_size) * ctx.dpr * overrides.point_size_multiplier
        effective_alpha = ctx.global_alpha * layer.style.alpha * overrides.alpha_multiplier

        glUniform1f(self.u_size, effective_size)
        glUniform1f(self.u_alpha, float(effective_alpha))
        glUniform1f(self.u_point_size_px, effective_size)

        # Outline logic
        glUniform1i(self.u_outline_enabled, 1 if layer.style.point_outline_enabled else 0)
        if layer.style.point_outline_enabled:
            glUniform4f(self.u_outline_color, *layer.style.point_outline_color)
            glUniform1f(self.u_outline_width_px, float(layer.style.point_outline_width) * ctx.dpr)

        glUniform2f(self.u_offset, *layer.translation)

        # 3. Draw call
        glBindVertexArray(layer._gl.vao)
        glDrawArrays(GL_POINTS, 0, layer._gl.count)

        # Cleanup
        glBindVertexArray(0)
        glUseProgram(0)
        glDisable(GL_PROGRAM_POINT_SIZE)

    # ------------------------------------------------------------------
    # Textured-quad path for imshow layers
    # ------------------------------------------------------------------

    def _make_image_texture(self, layer: ScatterLayer) -> int:
        """Upload the imshow matrix as a GL_RGBA texture and return the texture ID."""
        import matplotlib.cm as mcm

        meta = layer.metadata
        matrix = np.asarray(meta["matrix"], dtype=float)
        lo = meta.get("vmin")
        hi = meta.get("vmax")
        lo = float(np.nanmin(matrix)) if lo is None else float(lo)
        hi = float(np.nanmax(matrix)) if hi is None else float(hi)
        norm = np.clip((matrix - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        cmap_fn = mcm.get_cmap(meta.get("cmap", "viridis"))
        rgba = cmap_fn(norm).astype(np.float32)  # (rows, cols, 4)
        nan_mask = np.isnan(matrix)
        rgba[nan_mask] = 0.0
        # OpenGL origin is bottom-left; "upper" means row-0 = image top → flip
        if meta.get("origin", "upper") == "upper":
            rgba = rgba[::-1, :, :]
        img = (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)
        h, w = img.shape[:2]
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, img.tobytes())
        glBindTexture(GL_TEXTURE_2D, 0)
        return tex

    def _make_image_quad(self, layer: ScatterLayer) -> dict:
        """Create VAO/VBO for a textured quad covering the imshow extent."""
        xmin, xmax, ymin, ymax = layer.metadata["extent"]
        # interleaved: [pos_x, pos_y, uv_x, uv_y]
        verts = np.array(
            [
                [xmin, ymin, 0.0, 0.0],
                [xmax, ymin, 1.0, 0.0],
                [xmax, ymax, 1.0, 1.0],
                [xmin, ymax, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        stride = 4 * 4  # 4 floats × 4 bytes
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, C.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, C.c_void_p(8))
        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        glBindVertexArray(0)
        return {"vao": vao, "vbo": vbo, "ebo": ebo, "tex": self._make_image_texture(layer)}

    def _draw_image(self, layer: ScatterLayer, ctx: RenderContext) -> None:
        """Render an imshow layer as a smooth textured quad."""
        if not hasattr(layer, "_image_gl") or layer._image_gl is None:
            layer._image_gl = self._make_image_quad(layer)
            layer.dirty.gpu_dirty = False
        elif layer.dirty.gpu_dirty:
            old_tex = layer._image_gl["tex"]
            glDeleteTextures(1, [old_tex])
            layer._image_gl["tex"] = self._make_image_texture(layer)
            layer.dirty.gpu_dirty = False

        gl = layer._image_gl
        overrides = self.options.visual.overrides
        alpha = ctx.global_alpha * layer.style.alpha * overrides.alpha_multiplier

        glUseProgram(self.image_prog)
        glUniformMatrix4fv(self.u_image_mvp, 1, GL_TRUE, ctx.mvp)
        glUniform1f(self.u_image_alpha, float(alpha))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, gl["tex"])
        glUniform1i(self.u_image_tex, 0)
        glBindVertexArray(gl["vao"])
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)

    def draw_density(self, layer: ScatterLayer, ctx: RenderContext) -> None:
        """Accumulate point density into the current R32F target."""
        if layer.metadata.get("artist") == "imshow":
            return
        if layer.pts is None or len(layer.pts) == 0:
            return

        # 1. Resource Management
        if not hasattr(layer, "_gl") or layer._gl is None:
            layer._gl = self._create_buffers(layer)
            layer.dirty.gpu_dirty = True

        if layer.dirty.gpu_dirty:
            self.update_gpu_data(layer, layer._gl)

        # 2. Setup Shaders
        glUseProgram(self.accum_prog)
        glEnable(GL_PROGRAM_POINT_SIZE)

        glUniformMatrix4fv(self.u_accum_mvp, 1, GL_TRUE, ctx.mvp)

        overrides = self.options.visual.overrides
        # Use a slightly smaller size for density to prevent over-blurring
        # unless user explicitly requested massive points.
        effective_size = max(
            1.0, float(layer.style.point_size) * ctx.dpr * 0.5 * overrides.point_size_multiplier
        )
        glUniform1f(self.u_accum_size, effective_size)

        # Weighted Accumulation
        if self.options.density_weighted:
            alpha = ctx.global_alpha * layer.style.alpha * overrides.alpha_multiplier
            glUniform1f(self.u_accum_alpha, float(alpha))
        else:
            glUniform1f(self.u_accum_alpha, 1.0)  # Simple counting mode

        glUniform2f(self.u_accum_offset, *layer.translation)

        # 3. Draw call
        glBindVertexArray(layer._gl.vao)
        glDrawArrays(GL_POINTS, 0, layer._gl.count)

        # Cleanup
        glBindVertexArray(0)
        glUseProgram(0)
        glDisable(GL_PROGRAM_POINT_SIZE)
