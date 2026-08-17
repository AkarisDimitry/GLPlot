from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from OpenGL.GL import *

from ..options import resolve_axis_margins
from ..utils.gl_utils import link_program
from ..utils.shaders import DENSITY_RESOLVE_FS, INTERACTION_FULLSCREEN_VS
from .base import GLOffscreenTarget

if TYPE_CHECKING:
    pass


class DensityRenderer:
    """
    Modular Density Manager for Phase 5.

    Coordinates the accumulation of density data from multiple primitive
    renderers into a shared RGBA32F texture, then resolves it into a heatmap.

    The accumulation buffer holds ``(sum(w), sum(w*rgb))`` per pixel: red is the density the
    whole mode is built on, and the other three channels ride alongside it so the resolve can
    recover the weight-averaged colour of what landed there. That is what lets a layer's own
    colour survive the pass -- the buffer used to be single-channel, so ``scatter(color=...)``
    contributed a *weight* and nothing else, and every density image came out in the global
    colormap no matter what the caller asked for. See :meth:`resolve` for the two modes.
    """

    def __init__(self, plot: "GPULinePlot") -> None:
        self.plot = plot
        self.options = plot.options

        # Resolve pass
        self.resolve_prog = 0
        self.u_resolve_tex = -1
        self.u_resolve_gain = -1
        self.u_resolve_max_val = -1
        self.u_resolve_is_log = -1
        self.u_resolve_scheme = -1
        self.u_resolve_invert = -1
        self.u_resolve_light_to_color = -1
        self.u_resolve_uv_min = -1
        self.u_resolve_uv_max = -1
        self.u_resolve_tint = -1
        self.resolve_vao = 0

        self.accum_target = GLOffscreenTarget()
        # glClearBufferfv(GL_COLOR, ...) requires exactly 4 floats per the OpenGL spec.
        self._clear_zero = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        #: The max-density readback in ``resolve`` is a full-buffer GPU->CPU transfer AND a
        #: pipeline sync (the CPU waits for the GPU). Doing it every frame stalls the pipeline;
        #: the colormap normalisation it feeds tolerates a slightly stale value, so it is
        #: throttled to every _MAX_READ_INTERVAL frames and cached in between.
        self._cached_max_val = 1.0
        self._frames_since_read = 10_000  # force a read on the first resolve

    def initialize(self, fb_width: int, fb_height: int) -> None:
        """Initialize shaders and framebuffer targets."""
        self.resolve_prog = link_program(INTERACTION_FULLSCREEN_VS, DENSITY_RESOLVE_FS)
        self.u_resolve_tex = glGetUniformLocation(self.resolve_prog, "u_tex")
        self.u_resolve_gain = glGetUniformLocation(self.resolve_prog, "u_gain")
        self.u_resolve_max_val = glGetUniformLocation(self.resolve_prog, "u_max_val")
        self.u_resolve_is_log = glGetUniformLocation(self.resolve_prog, "u_is_log")
        self.u_resolve_scheme = glGetUniformLocation(self.resolve_prog, "u_scheme")
        self.u_resolve_invert = glGetUniformLocation(self.resolve_prog, "u_invert")
        self.u_resolve_light_to_color = glGetUniformLocation(self.resolve_prog, "u_light_to_color")
        self.u_resolve_uv_min = glGetUniformLocation(self.resolve_prog, "u_uv_min")
        self.u_resolve_uv_max = glGetUniformLocation(self.resolve_prog, "u_uv_max")
        self.u_resolve_tint = glGetUniformLocation(self.resolve_prog, "u_tint")
        self.resolve_vao = glGenVertexArrays(1)

        self.rebuild_target(fb_width, fb_height)

    def rebuild_target(self, fb_width: int, fb_height: int) -> None:
        """Create/Resize the RGBA32F accumulation texture."""
        if self.accum_target.fbo:
            glDeleteFramebuffers(1, [self.accum_target.fbo])
            glDeleteTextures(1, [self.accum_target.tex])

        scale = max(0.05, float(self.options.density_resolution_scale))
        w = max(1, int(round(fb_width * scale)))
        h = max(1, int(round(fb_height * scale)))

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, None)
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
        # The accumulation buffer is filled edge to edge (a panel's data via its own mvp fills
        # NDC [-1,1]); a scissor left enabled by the multi-panel loop would clip it to the
        # panel's sub-rect and corrupt the accumulation, so it is turned off here.
        glDisable(GL_SCISSOR_TEST)
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

    def resolve(
        self,
        target_fbo: int = 0,
        target_size: Optional[Tuple[int, int]] = None,
        target_viewport: Optional[Tuple[int, int, int, int]] = None,
        tint: bool = False,
    ) -> None:
        """Resolve the accumulated density into a colour image in the target FBO.

        ``target_viewport`` (x, y, w, h in framebuffer pixels) places the heatmap in a panel's
        sub-rectangle when the window is split; without it the whole framebuffer is used.

        Two modes, and which one runs is the caller's call (see
        ``RendererManager.density_tint_active``):

        * **Heatmap** (``tint=False``, the default and what every density plot has always
          been): the accumulated weight goes through ``density_scheme_index`` and the whole
          image is one colormap. Opaque, so it replaces whatever is under it.
        * **Tinted** (``tint=True``): the pixel is painted in the weight-averaged colour the
          layers themselves contributed, at an opacity given by the same normalised density.
          Composited over the background rather than replacing it -- premultiplied, because
          that is the form the resolve shader emits and the only one that survives a
          gradient background underneath.
        """
        glBindFramebuffer(GL_FRAMEBUFFER, target_fbo)

        # Robust Viewport Management (Fix for 1/4 size rendering on HighDPI)
        if target_viewport is not None:
            glViewport(*(int(v) for v in target_viewport))
        elif target_size is not None:
            glViewport(0, 0, int(target_size[0]), int(target_size[1]))
        else:
            glViewport(0, 0, self.plot.fb_width, self.plot.fb_height)

        # Read back the maximum density (for colormap normalisation) only every few frames:
        # the readback is a full-buffer transfer + a hard GPU sync, and a slightly stale max
        # is invisible in the colour mapping. Between reads the cached value is reused, so the
        # per-frame pipeline stall (which dominates when zoomed out and the accumulation is
        # heavy) goes away.
        interval = int(getattr(self.options, "density_max_read_interval", 12) or 1)
        if self._frames_since_read >= max(1, interval):
            w, h = self.accum_target.width, self.accum_target.height
            glBindFramebuffer(GL_FRAMEBUFFER, self.accum_target.fbo)
            # GL_RED is the weight channel; the other three are a premultiplied colour sum
            # whose magnitude says nothing about density (a black layer accumulates weight
            # and no colour at all). Asking for the one channel also keeps this transfer the
            # size it was before the buffer grew to RGBA.
            data = glReadPixels(0, 0, w, h, GL_RED, GL_FLOAT)
            buf = np.frombuffer(data, dtype=np.float32)
            self._cached_max_val = max(1.0, float(buf.max()) if buf.size else 1.0)
            self._frames_since_read = 0
        else:
            self._frames_since_read += 1
        max_val = self._cached_max_val

        glBindFramebuffer(GL_FRAMEBUFFER, target_fbo)

        # Pass UV bounds of the inset region to the shader to clip margins cleanly
        w_px = max(self.plot.width, 1e-12)
        h_px = max(self.plot.height, 1e-12)
        margin_l, margin_r, margin_b, margin_t = resolve_axis_margins(self.options)

        uv_min_x = margin_l / w_px
        uv_min_y = margin_b / h_px
        uv_max_x = 1.0 - margin_r / w_px
        uv_max_y = 1.0 - margin_t / h_px

        if tint:
            # Premultiplied source over the background that is already in the target: the
            # sparse edges of the cloud fade into it instead of being cut out against a flat
            # fill of the ramp's dark end.
            glEnable(GL_BLEND)
            glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
        else:
            glDisable(GL_BLEND)
        glUseProgram(self.resolve_prog)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.accum_target.tex)
        glUniform1i(self.u_resolve_tex, 0)
        glUniform1f(self.u_resolve_gain, float(self.options.density_gain))
        glUniform1f(self.u_resolve_max_val, max_val)
        glUniform1i(
            self.u_resolve_is_log, 1 if getattr(self.options, "density_is_log", True) else 0
        )
        glUniform1i(self.u_resolve_scheme, self.options.density_scheme_index)
        glUniform1i(
            self.u_resolve_invert, 1 if getattr(self.options, "density_invert", False) else 0
        )
        glUniform1i(
            self.u_resolve_light_to_color,
            1 if getattr(self.options, "density_light_to_color", True) else 0,
        )
        glUniform2f(self.u_resolve_uv_min, uv_min_x, uv_min_y)
        glUniform2f(self.u_resolve_uv_max, uv_max_x, uv_max_y)
        glUniform1i(self.u_resolve_tint, 1 if tint else 0)

        glBindVertexArray(self.resolve_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        if tint:
            glDisable(GL_BLEND)

    def get_density_array(self) -> np.ndarray:
        """
        Read back the accumulated float32 densities from the accumulation target
        and return them as a 2D float32 numpy array.

        The weight channel only -- the same numbers this returned when the buffer was
        single-channel. The colour the buffer now also carries is a rendering detail and
        would not be a density if it were summed into these.
        """
        if not self.accum_target or not self.accum_target.fbo:
            return np.zeros((0, 0), dtype=np.float32)

        import sys

        gl = sys.modules.get("OpenGL.GL")
        if gl is None:
            import OpenGL.GL as gl

        w, h = self.accum_target.width, self.accum_target.height

        # Save current bound FBO
        prev_fbo = gl.glGetIntegerv(gl.GL_FRAMEBUFFER_BINDING)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.accum_target.fbo)
        data = gl.glReadPixels(0, 0, w, h, gl.GL_RED, gl.GL_FLOAT)

        # Restore previous bound FBO
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, prev_fbo)

        buf = np.frombuffer(data, dtype=np.float32)
        return buf.reshape((h, w))
