from __future__ import annotations
from OpenGL.GL import *
from typing import Tuple, TYPE_CHECKING
from ..utils.shaders import INTERACTION_FULLSCREEN_VS, CACHE_IMPOSTOR_FS
from ..utils.gl_utils import link_program
from .base import GLOffscreenTarget

if TYPE_CHECKING:
    from ..engine import GPULinePlot

class InteractionRenderer:
    """
    Interaction rendering should never compromise frame pacing.

    This renderer supports two lightweight interaction paths:
    1. cached impostor reprojection
    2. density/aggregate interaction image (placeholder for expansion)
    """

    def __init__(self, plot: "GPULinePlot"):
        self.plot = plot
        self.options = plot.options
        self.cache_prog = 0
        self.u_cache_tex = -1
        self.u_cache_window = -1
        self.u_cur_window = -1
        self.cache_vao = 0
        self.cache_target = GLOffscreenTarget()

    def initialize(self, fb_width: int, fb_height: int) -> None:
        self.cache_prog = link_program(INTERACTION_FULLSCREEN_VS, CACHE_IMPOSTOR_FS)
        self.u_cache_tex = glGetUniformLocation(self.cache_prog, "u_tex")
        self.u_cache_window = glGetUniformLocation(self.cache_prog, "u_cache_window")
        self.u_cur_window = glGetUniformLocation(self.cache_prog, "u_cur_window")
        self.cache_vao = glGenVertexArrays(1)
        self.rebuild_cache_target(fb_width, fb_height)

    def rebuild_cache_target(self, fb_width: int, fb_height: int) -> None:
        if self.cache_target.fbo:
            glDeleteFramebuffers(1, [self.cache_target.fbo])
            glDeleteTextures(1, [self.cache_target.tex])

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, fb_width, fb_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        self.cache_target = GLOffscreenTarget(fbo=fbo, tex=tex, width=fb_width, height=fb_height)

    def draw_cached_impostor(
        self,
        capture_window: Tuple[float, float, float, float],
        current_window: Tuple[float, float, float, float],
        target_fbo: int = 0,
    ) -> None:
        glBindFramebuffer(GL_FRAMEBUFFER, target_fbo)
        glViewport(0, 0, self.plot.fb_width, self.plot.fb_height)
        glEnable(GL_BLEND)
        # Cache stores premultiplied RGB (cleared to transparent black); use
        # GL_ONE so the already-multiplied RGB reaches the screen unchanged.
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(self.cache_prog)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.cache_target.tex)
        glUniform1i(self.u_cache_tex, 0)
        glUniform4f(self.u_cache_window, *capture_window)
        glUniform4f(self.u_cur_window, *current_window)
        glBindVertexArray(self.cache_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
