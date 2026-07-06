from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from OpenGL.GL import *

from ..renderers.base import GLOffscreenTarget
from ..utils.gl_utils import link_program
from ..utils.shaders import (
    PICKING_LINES_FS,
    PICKING_LINES_VS,
    PICKING_PATCH_FS,
    PICKING_PATCH_VS,
    PICKING_SCATTER_FS,
    PICKING_SCATTER_VS,
    PICKING_STRIP_FS,
    PICKING_STRIP_VS,
)

if TYPE_CHECKING:
    from ..core import SceneData
    from ..options import EngineOptions
    from ..renderers.exact import GLLineBuffers


class PickingManager:
    def __init__(self, options: EngineOptions):
        self.options = options
        self.target = GLOffscreenTarget()
        self.pid_lines = -1
        self.pid_scatter = -1
        self.pid_strip = -1
        self.pid_patch = -1

    def initialize(self, fb_width: int, fb_height: int) -> None:
        self.pid_lines = link_program(PICKING_LINES_VS, PICKING_LINES_FS)
        self.pid_scatter = link_program(PICKING_SCATTER_VS, PICKING_SCATTER_FS)
        self.pid_strip = link_program(PICKING_STRIP_VS, PICKING_STRIP_FS)
        self.pid_patch = link_program(PICKING_PATCH_VS, PICKING_PATCH_FS)
        self.rebuild_target(fb_width, fb_height)

    def rebuild_target(self, fb_width: int, fb_height: int) -> None:
        if self.target.fbo:
            glDeleteFramebuffers(1, [self.target.fbo])
            glDeleteTextures(1, [self.target.tex])

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        # We store 32-bit Integer IDs
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_R32I, fb_width, fb_height, 0, GL_RED_INTEGER, GL_INT, None
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print(f"Picking Framebuffer incomplete: {status}")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self.target = GLOffscreenTarget(fbo=fbo, tex=tex, width=fb_width, height=fb_height)

    def draw_pick_scene(
        self,
        scene: SceneData,
        buffers: Any,
        mvp: np.ndarray,
        window: Tuple[float, float, float, float],
    ) -> None:
        glBindFramebuffer(GL_FRAMEBUFFER, self.target.fbo)
        glViewport(0, 0, self.target.width, self.target.height)

        # 1. State for picking: no blending, no multisampling
        glDisable(GL_BLEND)
        glDisable(GL_MULTISAMPLE)

        # Clear with 0 (no object) using integer clear
        glClearBufferiv(GL_COLOR, 0, np.array([0], dtype=np.int32))

        current_offset = 0
        for i, layer in enumerate(scene.layers):
            if not layer.style.visible:
                continue

            # IDs are 1-based, 0 is "nothing"
            layer_id_start = current_offset + 1

            if layer.layer_type == "line_family":
                if layer.ab is not None and hasattr(layer, "_gl"):
                    glUseProgram(self.pid_lines)
                    glUniformMatrix4fv(
                        glGetUniformLocation(self.pid_lines, "u_mvp"), 1, GL_FALSE, mvp
                    )
                    glUniform2f(glGetUniformLocation(self.pid_lines, "u_xrange"), *layer.x_range)
                    glUniform4f(glGetUniformLocation(self.pid_lines, "u_window"), *window)
                    glUniform2f(
                        glGetUniformLocation(self.pid_lines, "u_layer_offset"), *layer.translation
                    )
                    glUniform1i(glGetUniformLocation(self.pid_lines, "u_id_offset"), current_offset)
                    layer._gl.render(len(layer.ab))
                    current_offset += len(layer.ab)

            elif layer.layer_type == "scatter":
                if hasattr(layer, "_gl"):
                    glUseProgram(self.pid_scatter)
                    glEnable(GL_PROGRAM_POINT_SIZE)
                    glUniformMatrix4fv(
                        glGetUniformLocation(self.pid_scatter, "u_mvp"), 1, GL_FALSE, mvp
                    )
                    glUniform1f(
                        glGetUniformLocation(self.pid_scatter, "u_size"),
                        float(layer.style.point_size),
                    )
                    glUniform1i(
                        glGetUniformLocation(self.pid_scatter, "u_id_offset"), current_offset
                    )
                    glUniform2f(
                        glGetUniformLocation(self.pid_scatter, "u_layer_offset"), *layer.translation
                    )
                    glBindVertexArray(layer._gl.vao)
                    glDrawArrays(GL_POINTS, 0, layer._gl.count)
                    current_offset += layer._gl.count

            elif layer.layer_type == "polyline":
                if hasattr(layer, "_gl"):
                    # Polyline is treated as 1 object for now
                    glUseProgram(self.pid_strip)
                    glUniformMatrix4fv(
                        glGetUniformLocation(self.pid_strip, "u_mvp"), 1, GL_FALSE, mvp
                    )
                    glUniform1i(glGetUniformLocation(self.pid_strip, "u_id"), current_offset + 1)
                    glUniform2f(
                        glGetUniformLocation(self.pid_strip, "u_layer_offset"), *layer.translation
                    )
                    glBindVertexArray(layer._gl.vao)
                    # Polyline uses instanced quads (6 verts per segment)
                    glDrawElements(
                        GL_TRIANGLES, layer._gl.instance_count * 6, GL_UNSIGNED_SHORT, None
                    )
                    current_offset += 1

            elif layer.layer_type == "patch":
                if hasattr(layer, "_gl"):
                    glUseProgram(self.pid_patch)
                    glUniformMatrix4fv(
                        glGetUniformLocation(self.pid_patch, "u_mvp"), 1, GL_FALSE, mvp
                    )
                    glUniform1i(glGetUniformLocation(self.pid_patch, "u_id"), current_offset + 1)
                    glUniform2f(
                        glGetUniformLocation(self.pid_patch, "u_layer_offset"), *layer.translation
                    )
                    glBindVertexArray(layer._gl.vao)
                    mode = GL_TRIANGLE_STRIP
                    if getattr(layer, "mode", "") == "triangles":
                        mode = GL_TRIANGLES
                    if layer._gl.ebo:
                        glDrawElements(mode, layer._gl.count, GL_UNSIGNED_INT, None)
                    else:
                        glDrawArrays(mode, 0, layer._gl.count)
                    current_offset += 1

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)

    def pick_readback(self, sx: float, sy: float, scene: SceneData) -> Optional[dict]:
        """sx, sy must be in framebuffer/pixel coordinates (already DPR-scaled)."""
        gx = int(sx)
        gy = int(self.target.height - sy)

        if gx < 0 or gx >= self.target.width or gy < 0 or gy >= self.target.height:
            return None

        glBindFramebuffer(GL_FRAMEBUFFER, self.target.fbo)
        pixels = glReadPixels(gx, gy, 1, 1, GL_RED_INTEGER, GL_INT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        val = int(np.frombuffer(pixels, dtype=np.int32)[0])
        if val <= 0:
            return None

        # Decode based on global offsets
        current_offset = 0
        for i, layer in enumerate(scene.layers):
            if not layer.style.visible:
                continue

            count = 0
            if layer.layer_type == "line_family":
                count = len(layer.ab) if layer.ab is not None else 0
            elif layer.layer_type == "scatter":
                count = layer._gl.count if hasattr(layer, "_gl") else 0
            elif layer.layer_type in ["polyline", "patch"]:
                count = 1

            if current_offset < val <= current_offset + count:
                return {
                    "type": layer.layer_type,
                    "layer_id": layer.layer_id,  # Use stable ID
                    "element_idx": val - current_offset - 1,
                    "layer": layer,
                }
            current_offset += count

        return None
