from __future__ import annotations

import glfw
from OpenGL.GL import *
from typing import TYPE_CHECKING

from ..utils.gl_utils import link_program
from ..utils.shaders import (
    POST_FX_VS,
    GRADIENT_BG_FS,
    BLOOM_EXTRACT_FS,
    GAUSSIAN_BLUR_FS,
    POST_COMPOSITE_FS,
)
from ..renderers.base import GLOffscreenTarget

if TYPE_CHECKING:
    from ..engine import GPULinePlot


class EffectManager:
    """
    Post-processing manager.

    Design goals:
    - Zero meaningful overhead when all effects are disabled.
    - Lazy initialization of shaders/FBOs.
    - Explicit scene render flow:
        begin_scene()
        draw_background()
        <draw scene here>
        end_scene()
    """

    def __init__(self, plot: "GPULinePlot"):
        self.plot = plot
        self.options = plot.options

        # Programs
        self.prog_bg = 0
        self.prog_extract = 0
        self.prog_blur = 0
        self.prog_composite = 0

        # Cached uniform locations
        self.u_bg_top_color = -1
        self.u_bg_bottom_color = -1

        self.u_extract_tex = -1
        self.u_extract_threshold = -1

        self.u_blur_tex = -1
        self.u_blur_horizontal = -1
        self.u_blur_radius = -1

        self.u_comp_scene_tex = -1
        self.u_comp_bloom_tex = -1
        self.u_comp_bloom_enabled = -1
        self.u_comp_bloom_intensity = -1

        # FBOs
        self.scene_fbo = GLOffscreenTarget()
        self.extract_fbo = GLOffscreenTarget()
        self.ping_fbo = GLOffscreenTarget()
        self.pong_fbo = GLOffscreenTarget()

        # Fullscreen quad
        self.quad_vao = 0

        self.initialized = False

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    def any_post_enabled(self) -> bool:
        if self.options.reactive_rendering:
            return True
        v = self.options.visual
        return v.glow.enabled

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ensure_resources(self) -> None:
        if self.initialized:
            return

        self.prog_bg = link_program(POST_FX_VS, GRADIENT_BG_FS)
        self.prog_extract = link_program(POST_FX_VS, BLOOM_EXTRACT_FS)
        self.prog_blur = link_program(POST_FX_VS, GAUSSIAN_BLUR_FS)
        self.prog_composite = link_program(POST_FX_VS, POST_COMPOSITE_FS)

        self.u_bg_top_color = glGetUniformLocation(self.prog_bg, "u_top_color")
        self.u_bg_bottom_color = glGetUniformLocation(self.prog_bg, "u_bottom_color")

        self.u_extract_tex = glGetUniformLocation(self.prog_extract, "u_tex")
        self.u_extract_threshold = glGetUniformLocation(self.prog_extract, "u_threshold")

        self.u_blur_tex = glGetUniformLocation(self.prog_blur, "u_tex")
        self.u_blur_horizontal = glGetUniformLocation(self.prog_blur, "u_horizontal")
        self.u_blur_radius = glGetUniformLocation(self.prog_blur, "u_radius")

        self.u_comp_scene_tex = glGetUniformLocation(self.prog_composite, "u_scene_tex")
        self.u_comp_bloom_tex = glGetUniformLocation(self.prog_composite, "u_bloom_tex")
        self.u_comp_bloom_enabled = glGetUniformLocation(self.prog_composite, "u_bloom_enabled")
        self.u_comp_bloom_intensity = glGetUniformLocation(self.prog_composite, "u_bloom_intensity")

        self.quad_vao = glGenVertexArrays(1)
        self._rebuild_targets()
        self.initialized = True

    def shutdown(self) -> None:
        self._destroy_target(self.scene_fbo)
        self._destroy_target(self.extract_fbo)
        self._destroy_target(self.ping_fbo)
        self._destroy_target(self.pong_fbo)

        if self.quad_vao:
            glDeleteVertexArrays(1, [self.quad_vao])
            self.quad_vao = 0

        for prog in (self.prog_bg, self.prog_extract, self.prog_blur, self.prog_composite):
            if prog:
                glDeleteProgram(prog)

        self.prog_bg = 0
        self.prog_extract = 0
        self.prog_blur = 0
        self.prog_composite = 0
        self.initialized = False

    def on_resize(self) -> None:
        if self.initialized:
            self._rebuild_targets()

    # ------------------------------------------------------------------
    # Scene flow
    # ------------------------------------------------------------------

    def begin_scene(self) -> None:
        """
        Bind the correct render target for the scene.
        """
        if not self.any_post_enabled():
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glViewport(0, 0, self.plot.fb_width, self.plot.fb_height)
            return

        self.ensure_resources()
        glBindFramebuffer(GL_FRAMEBUFFER, self.scene_fbo.fbo)
        glViewport(0, 0, self.scene_fbo.width, self.scene_fbo.height)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def end_scene(self) -> None:
        """
        Resolve post-processing if needed.
        """
        if self.any_post_enabled():
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            self.resolve()

    # ------------------------------------------------------------------
    # Background
    # ------------------------------------------------------------------

    def draw_background(self) -> None:
        """
        Draw the background into whichever framebuffer is currently bound.
        """
        # Overwrite background color with the colormap's minimum color in density mode
        if getattr(self.plot, "display_density", False):
            scheme_idx = getattr(self.options, "density_scheme_index", 0)
            invert = getattr(self.options, "density_invert", False)
            ltc = getattr(self.options, "density_light_to_color", True)

            from ..utils.shaders import get_colormap_min_color

            c = get_colormap_min_color(scheme_idx, invert, ltc)
            glClearColor(c[0], c[1], c[2], 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            return

        v = self.options.visual.gradient_background

        if not v.enabled:
            c = self.options.visual.background_color
            glClearColor(c[0], c[1], c[2], 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            return

        self.ensure_resources()
        glDisable(GL_BLEND)

        # Disable world clipping for background pass
        if self.options.enable_clipping_optimization:
            for i in range(4):
                glDisable(GL_CLIP_DISTANCE0 + i)

        glUseProgram(self.prog_bg)
        glUniform3f(self.u_bg_top_color, *v.top_color)
        glUniform3f(self.u_bg_bottom_color, *v.bottom_color)

        glBindVertexArray(self.quad_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def resolve(self) -> None:
        """
        Perform post-processing and draw to the default framebuffer.
        """
        v = self.options.visual
        if not self.any_post_enabled():
            return

        self.ensure_resources()

        glDisable(GL_BLEND)
        glBindVertexArray(self.quad_vao)

        # 1) Bright-pass extraction
        if v.glow.enabled:
            glBindFramebuffer(GL_FRAMEBUFFER, self.extract_fbo.fbo)
            glViewport(0, 0, self.extract_fbo.width, self.extract_fbo.height)
            glUseProgram(self.prog_extract)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.scene_fbo.tex)
            glUniform1i(self.u_extract_tex, 0)
            glUniform1f(self.u_extract_threshold, v.glow.threshold)

            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

            # 2) Horizontal blur
            glBindFramebuffer(GL_FRAMEBUFFER, self.ping_fbo.fbo)
            glViewport(0, 0, self.ping_fbo.width, self.ping_fbo.height)
            glUseProgram(self.prog_blur)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.extract_fbo.tex)
            glUniform1i(self.u_blur_tex, 0)
            glUniform1i(self.u_blur_horizontal, 1)
            glUniform1f(self.u_blur_radius, v.glow.radius_px)

            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

            # 3) Vertical blur
            glBindFramebuffer(GL_FRAMEBUFFER, self.pong_fbo.fbo)
            glViewport(0, 0, self.pong_fbo.width, self.pong_fbo.height)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.ping_fbo.tex)
            glUniform1i(self.u_blur_tex, 0)
            glUniform1i(self.u_blur_horizontal, 0)
            glUniform1f(self.u_blur_radius, v.glow.radius_px)

            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        # 4) Final composite to default framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.plot.fb_width, self.plot.fb_height)
        glUseProgram(self.prog_composite)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.scene_fbo.tex)
        glUniform1i(self.u_comp_scene_tex, 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.pong_fbo.tex if v.glow.enabled else self.scene_fbo.tex)
        glUniform1i(self.u_comp_bloom_tex, 1)

        glUniform1i(self.u_comp_bloom_enabled, 1 if v.glow.enabled else 0)
        glUniform1f(self.u_comp_bloom_intensity, v.glow.intensity)

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        if self.options.enable_clipping_optimization:
            # Re-enable for subsequent exact draws if they don't explicitly handle it
            for i in range(4):
                glEnable(GL_CLIP_DISTANCE0 + i)

        glBindVertexArray(0)
        glUseProgram(0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_targets(self) -> None:
        w, h = self.plot.fb_width, self.plot.fb_height
        if w <= 0 or h <= 0:
            return

        self._destroy_target(self.scene_fbo)
        self._destroy_target(self.extract_fbo)
        self._destroy_target(self.ping_fbo)
        self._destroy_target(self.pong_fbo)

        # Scene target: full res, float, needs depth buffer
        self.scene_fbo = self._create_target(w, h, GL_RGBA16F, with_depth=True)

        # Bloom chain: half res
        bw = max(1, int(round(w * 0.5)))
        bh = max(1, int(round(h * 0.5)))
        self.extract_fbo = self._create_target(bw, bh, GL_RGBA8)
        self.ping_fbo = self._create_target(bw, bh, GL_RGBA8)
        self.pong_fbo = self._create_target(bw, bh, GL_RGBA8)

    def _create_target(
        self, w: int, h: int, internal_format: int, with_depth: bool = False
    ) -> GLOffscreenTarget:
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)

        if internal_format == GL_RGBA16F:
            fmt = GL_RGBA
            dtype = GL_FLOAT
        elif internal_format == GL_RGBA8:
            fmt = GL_RGBA
            dtype = GL_UNSIGNED_BYTE
        elif internal_format == GL_R32F:
            fmt = GL_RED
            dtype = GL_FLOAT
        elif internal_format == GL_R32I:
            fmt = GL_RED_INTEGER
            dtype = GL_INT
        else:
            raise ValueError(f"Unsupported internal format: {internal_format}")

        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, w, h, 0, fmt, dtype, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)

        depth_rbo = 0
        if with_depth:
            depth_rbo = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, depth_rbo)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h)
            glFramebufferRenderbuffer(
                GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rbo
            )
            glBindRenderbuffer(GL_RENDERBUFFER, 0)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Framebuffer incomplete: {status}")

        return GLOffscreenTarget(fbo=fbo, tex=tex, width=w, height=h, depth_rbo=depth_rbo)

    def _destroy_target(self, target: GLOffscreenTarget) -> None:
        if target.fbo:
            glDeleteFramebuffers(1, [target.fbo])
        if target.tex:
            glDeleteTextures(1, [target.tex])
        if getattr(target, "depth_rbo", 0):
            glDeleteRenderbuffers(1, [target.depth_rbo])
        target.fbo = 0
        target.tex = 0
        target.depth_rbo = 0
        target.width = 0
        target.height = 0
