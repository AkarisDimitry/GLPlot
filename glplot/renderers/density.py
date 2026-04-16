from __future__ import annotations
import numpy as np
from OpenGL.GL import *
from typing import Tuple, Optional, TYPE_CHECKING

from ..utils.shaders import INTERACTION_FULLSCREEN_VS, DENSITY_RESOLVE_FS
from ..utils.gl_utils import link_program
from .base import GLOffscreenTarget

if TYPE_CHECKING:
    from ..options import EngineOptions
    from ..core.context import RenderContext

class DensityRenderer:
    """
    Modular Density Manager for Phase 5.
    
    Coordinates the accumulation of density data from multiple primitive
    renderers into a shared R32F texture, then resolves it into a heatmap.
    """

    def __init__(self, plot: "GPULinePlot"):
        self.plot = plot
        self.options = plot.options

        # Resolve pass
        self.resolve_prog = 0
        self.u_resolve_tex = -1
        self.u_resolve_gain = -1
        self.u_resolve_scheme = -1
        self.resolve_vao = 0

        self.accum_target = GLOffscreenTarget()
        self._clear_zero = np.array([0.0], dtype=np.float32)

    def initialize(self, fb_width: int, fb_height: int) -> None:
        """Initialize shaders and framebuffer targets."""
        self.resolve_prog = link_program(INTERACTION_FULLSCREEN_VS, DENSITY_RESOLVE_FS)
        self.u_resolve_tex = glGetUniformLocation(self.resolve_prog, "u_tex")
        self.u_resolve_gain = glGetUniformLocation(self.resolve_prog, "u_gain")
        self.u_resolve_log_scale = glGetUniformLocation(self.resolve_prog, "u_log_scale")
        self.u_resolve_scheme = glGetUniformLocation(self.resolve_prog, "u_scheme")
        self.resolve_vao = glGenVertexArrays(1)

        self.rebuild_target(fb_width, fb_height)

    def rebuild_target(self, fb_width: int, fb_height: int) -> None:
        """Create/Resize the R32F accumulation texture."""
        if self.accum_target.fbo:
            glDeleteFramebuffers(1, [self.accum_target.fbo])
            glDeleteTextures(1, [self.accum_target.tex])

        scale = max(0.05, float(self.options.density_resolution_scale))
        w = max(1, int(round(fb_width * scale)))
        h = max(1, int(round(fb_height * scale)))

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Density accumulation framebuffer is incomplete")

        self.accum_target = GLOffscreenTarget(fbo=fbo, tex=tex, width=w, height=h)

    def begin_accum(self) -> None:
        """Prepare the accumulation target for a new frame."""
        glBindFramebuffer(GL_FRAMEBUFFER, self.accum_target.fbo)
        glViewport(0, 0, self.accum_target.width, self.accum_target.height)
        glClearBufferfv(GL_COLOR, 0, self._clear_zero)

        # DENSITY ALWAYS NEEDS ADDITIVE BLENDING for accumulation
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE)
        glDisable(GL_DEPTH_TEST)

        # Handle clipping state if enabled globally
        if self.options.enable_clipping_optimization:
            glEnable(GL_CLIP_DISTANCE0)
            glEnable(GL_CLIP_DISTANCE1)
            glEnable(GL_CLIP_DISTANCE2)
            glEnable(GL_CLIP_DISTANCE3)
        else:
            glDisable(GL_CLIP_DISTANCE0)
            glDisable(GL_CLIP_DISTANCE1)
            glDisable(GL_CLIP_DISTANCE2)
            glDisable(GL_CLIP_DISTANCE3)

    def resolve(self, target_fbo: int = 0, target_size: Optional[Tuple[int, int]] = None) -> None:
        """Resolve the accumulated density into a color heatmap in the target FBO."""
        glBindFramebuffer(GL_FRAMEBUFFER, target_fbo)
        
        # Robust Viewport Management (Fix for 1/4 size rendering on HighDPI)
        if target_size is not None:
            glViewport(0, 0, int(target_size[0]), int(target_size[1]))
        else:
            glViewport(0, 0, self.plot.fb_width, self.plot.fb_height)

        glDisable(GL_BLEND)
        glUseProgram(self.resolve_prog)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.accum_target.tex)
        glUniform1i(self.u_resolve_tex, 0)
        glUniform1f(self.u_resolve_gain, float(self.options.density_gain))
        glUniform1f(self.u_resolve_log_scale, float(self.options.density_log_scale))
        glUniform1i(self.u_resolve_scheme, self.options.density_scheme_index)

        glBindVertexArray(self.resolve_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
