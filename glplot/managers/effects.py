from __future__ import annotations

import math
from typing import TYPE_CHECKING, List

from OpenGL.GL import *

from ..renderers.base import GLOffscreenTarget
from ..utils.gl_utils import link_program
from ..utils.shaders import (
    BLOOM_DOWNSAMPLE_FS,
    BLOOM_EXTRACT_FS,
    BLOOM_UPSAMPLE_FS,
    GRADIENT_BG_FS,
    POST_COMPOSITE_FS,
    POST_FX_VS,
)

if TYPE_CHECKING:
    from ..engine import GPULinePlot

#: Maximum number of dual-filter mip levels below the half-res bright pass.
#: 5 levels reaches an effective blur radius of ~64 screen px, which covers the
#: whole radius_px slider range; more would only cost bandwidth.
BLOOM_MAX_LEVELS = 5

#: Smallest allowed dimension for a bloom mip. Below this the dual filter's tap
#: offsets exceed the target and the blur degenerates.
BLOOM_MIN_DIM = 8

#: Soft-knee half-width relative to the threshold, used when ``GlowOptions`` has
#: no ``knee`` field. See the integrator note: adding ``knee: float = 0.5`` to
#: ``GlowOptions`` makes this user-tunable with no change here.
DEFAULT_GLOW_KNEE = 0.5

#: Tone-map operators understood by ``POST_COMPOSITE_FS``.
TONEMAP_NONE = 0
TONEMAP_REINHARD = 1
TONEMAP_ACES = 2


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

    def __init__(self, plot: "GPULinePlot") -> None:
        self.plot = plot
        self.options = plot.options

        # Programs
        self.prog_bg = 0
        self.prog_extract = 0
        self.prog_down = 0
        self.prog_up = 0
        self.prog_composite = 0

        # Cached uniform locations
        self.u_bg_top_color = -1
        self.u_bg_bottom_color = -1

        self.u_extract_tex = -1
        self.u_extract_threshold = -1
        self.u_extract_knee = -1

        self.u_down_tex = -1
        self.u_down_halfpixel = -1

        self.u_up_tex = -1
        self.u_up_halfpixel = -1

        self.u_comp_scene_tex = -1
        self.u_comp_bloom_tex = -1
        self.u_comp_bloom_enabled = -1
        self.u_comp_bloom_intensity = -1
        self.u_comp_tonemap = -1
        self.u_comp_grain = -1

        # FBOs
        self.scene_fbo = GLOffscreenTarget()

        # Bloom mip chain. Index 0 is the half-res bright pass; each subsequent
        # level is half the size of the one before it. The dual filter walks
        # down the chain and back up; level 0 is what the composite samples.
        self.bloom_chain: List[GLOffscreenTarget] = []

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
        return v.glow.enabled or self.grain_amount() > 0.0

    def tonemap_index(self) -> int:
        """Tone-map operator for the composite pass.

        Tone mapping is deliberately independent of glow. It used to be applied
        if and only if bloom was enabled, which meant ticking "Enable Glow"
        halved every white pixel before any glow appeared. Default is
        ``TONEMAP_NONE`` so the composite is a faithful pass-through and turning
        glow on can only ever *add* light.

        Reads the optional ``visual.tonemap_index`` field; absent or invalid
        values fall back to ``TONEMAP_NONE``.
        """
        raw = getattr(self.options.visual, "tonemap_index", TONEMAP_NONE)
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            return TONEMAP_NONE
        if idx not in (TONEMAP_NONE, TONEMAP_REINHARD, TONEMAP_ACES):
            return TONEMAP_NONE
        return idx

    def glow_knee(self) -> float:
        """Soft-knee half-width for the bright pass, relative to the threshold.

        Reads the optional ``visual.glow.knee`` field, clamped to ``[0, 1]``;
        absent or invalid values fall back to ``DEFAULT_GLOW_KNEE``.
        """
        raw = getattr(self.options.visual.glow, "knee", DEFAULT_GLOW_KNEE)
        try:
            knee = float(raw)
        except (TypeError, ValueError):
            return DEFAULT_GLOW_KNEE
        return min(max(knee, 0.0), 1.0)

    def grain_amount(self) -> float:
        """Static value-noise overlay strength for the composite pass, clamped to [0, 0.5].

        This is the "chalk" texture the plot-style presets ask for (``gui/styles.py``): a
        cheap per-pixel hash added *after* tone mapping, so it reads as tooth in the paper
        rather than as noise in the light. It is deliberately not animated — see
        ``POST_COMPOSITE_FS``.

        Reads the optional ``visual.grain_amount`` field; absent or invalid values fall
        back to ``0.0``, which makes the composite bit-identical to a build without it.
        """
        raw = getattr(self.options.visual, "grain_amount", 0.0)
        try:
            amount = float(raw)
        except (TypeError, ValueError):
            return 0.0
        if not amount == amount:  # NaN
            return 0.0
        return min(max(amount, 0.0), 0.5)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ensure_resources(self) -> None:
        if self.initialized:
            return

        self.prog_bg = link_program(POST_FX_VS, GRADIENT_BG_FS)
        self.prog_extract = link_program(POST_FX_VS, BLOOM_EXTRACT_FS)
        self.prog_down = link_program(POST_FX_VS, BLOOM_DOWNSAMPLE_FS)
        self.prog_up = link_program(POST_FX_VS, BLOOM_UPSAMPLE_FS)
        self.prog_composite = link_program(POST_FX_VS, POST_COMPOSITE_FS)

        self.u_bg_top_color = glGetUniformLocation(self.prog_bg, "u_top_color")
        self.u_bg_bottom_color = glGetUniformLocation(self.prog_bg, "u_bottom_color")

        self.u_extract_tex = glGetUniformLocation(self.prog_extract, "u_tex")
        self.u_extract_threshold = glGetUniformLocation(self.prog_extract, "u_threshold")
        self.u_extract_knee = glGetUniformLocation(self.prog_extract, "u_knee")

        self.u_down_tex = glGetUniformLocation(self.prog_down, "u_tex")
        self.u_down_halfpixel = glGetUniformLocation(self.prog_down, "u_halfpixel")

        self.u_up_tex = glGetUniformLocation(self.prog_up, "u_tex")
        self.u_up_halfpixel = glGetUniformLocation(self.prog_up, "u_halfpixel")

        self.u_comp_scene_tex = glGetUniformLocation(self.prog_composite, "u_scene_tex")
        self.u_comp_bloom_tex = glGetUniformLocation(self.prog_composite, "u_bloom_tex")
        self.u_comp_bloom_enabled = glGetUniformLocation(self.prog_composite, "u_bloom_enabled")
        self.u_comp_bloom_intensity = glGetUniformLocation(self.prog_composite, "u_bloom_intensity")
        self.u_comp_tonemap = glGetUniformLocation(self.prog_composite, "u_tonemap")
        self.u_comp_grain = glGetUniformLocation(self.prog_composite, "u_grain")

        self.quad_vao = glGenVertexArrays(1)
        self._rebuild_targets()
        self.initialized = True

    def shutdown(self) -> None:
        self._destroy_target(self.scene_fbo)
        self._destroy_bloom_chain()

        if self.quad_vao:
            glDeleteVertexArrays(1, [self.quad_vao])
            self.quad_vao = 0

        for prog in (
            self.prog_bg,
            self.prog_extract,
            self.prog_down,
            self.prog_up,
            self.prog_composite,
        ):
            if prog:
                glDeleteProgram(prog)

        self.prog_bg = 0
        self.prog_extract = 0
        self.prog_down = 0
        self.prog_up = 0
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

        Both paths clear the **depth** buffer; only the offscreen one clears colour, since
        on the default framebuffer ``draw_background`` owns the colour (it either clears to
        the background colour or paints a gradient quad over it).

        The depth clear on the default framebuffer is not housekeeping -- without it a 3D
        scene renders exactly once. Depth-tested geometry writes the depth buffer on frame
        1; on frame 2 the same geometry is at the same depth, fails ``GL_LESS``, and draws
        nothing. What is left is whatever does *not* depth-test, which is the ``axis3d``
        wireframe alone: the window shows an empty box with the data missing, and rotating
        the camera brings back only the fragments that happen to land nearer than a stale
        depth value from an earlier orientation. It went unnoticed for as long as it did
        because nothing else here uses depth -- the 2D renderers never touch it, and the
        post-processing path (SSAO, glow, DOF) clears depth with its colour on the line
        below, so any scene with an effect enabled was always correct.
        """
        if not self.any_post_enabled():
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glViewport(0, 0, self.plot.fb_width, self.plot.fb_height)
            # glClear obeys the depth mask, and a renderer that left it off would silently
            # turn this back into a no-op.
            glDepthMask(GL_TRUE)
            glClear(GL_DEPTH_BUFFER_BIT)
            return

        self.ensure_resources()
        glBindFramebuffer(GL_FRAMEBUFFER, self.scene_fbo.fbo)
        glViewport(0, 0, self.scene_fbo.width, self.scene_fbo.height)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glDepthMask(GL_TRUE)
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
        # Overwrite background color with the colormap's minimum color in density mode --
        # unless the density pass is going to paint in the layers' own colours, which is
        # composited over the ordinary background rather than over the ramp's dark end.
        tinted = bool(getattr(self.plot, "density_tint_active", lambda: False)())
        if getattr(self.plot, "display_density", False) and not tinted:
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

        bloom_ready = v.glow.enabled and len(self.bloom_chain) >= 2
        if bloom_ready:
            self._render_bloom(v.glow)

        # Final composite to the default framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.plot.fb_width, self.plot.fb_height)
        glUseProgram(self.prog_composite)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.scene_fbo.tex)
        glUniform1i(self.u_comp_scene_tex, 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.bloom_chain[0].tex if bloom_ready else self.scene_fbo.tex)
        glUniform1i(self.u_comp_bloom_tex, 1)

        glUniform1i(self.u_comp_bloom_enabled, 1 if bloom_ready else 0)
        glUniform1f(self.u_comp_bloom_intensity, v.glow.intensity)
        glUniform1i(self.u_comp_tonemap, self.tonemap_index())
        glUniform1f(self.u_comp_grain, self.grain_amount())

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        if self.options.enable_clipping_optimization:
            # Re-enable for subsequent exact draws if they don't explicitly handle it
            for i in range(4):
                glEnable(GL_CLIP_DISTANCE0 + i)

        glBindVertexArray(0)
        glUseProgram(0)

    # ------------------------------------------------------------------
    # Bloom
    # ------------------------------------------------------------------

    def _bloom_levels(self, radius_px: float) -> int:
        """Number of dual-filter mip steps below the bright pass for a radius.

        Each step doubles the effective blur width, so the level count is the
        integer part of ``log2(radius)``; the fractional part is carried by the
        tap offset (see :meth:`_bloom_offset`), which keeps the radius slider
        continuous instead of stepping in octaves.
        """
        available = max(0, len(self.bloom_chain) - 1)
        if available == 0:
            return 0
        lvl = int(math.floor(math.log2(max(radius_px, 1.0))))
        return max(1, min(lvl, available, BLOOM_MAX_LEVELS))

    def _bloom_offset(self, radius_px: float, levels: int) -> float:
        """Fractional tap scale between two mip levels, in ``[0.5, 2.0]``.

        Outside that range the dual filter's taps stop overlapping and the
        ringing the old blur suffered from would come back.
        """
        frac = math.log2(max(radius_px, 1.0)) - float(levels)
        return min(max(2.0**frac, 0.5), 2.0)

    def _render_bloom(self, glow) -> None:
        """Bright pass + dual-filter blur. Result lands in ``bloom_chain[0]``."""
        levels = self._bloom_levels(float(glow.radius_px))
        offset = self._bloom_offset(float(glow.radius_px), levels)

        # 1) Bright pass with soft knee: scene -> chain[0] (half res)
        dst = self.bloom_chain[0]
        glBindFramebuffer(GL_FRAMEBUFFER, dst.fbo)
        glViewport(0, 0, dst.width, dst.height)
        glUseProgram(self.prog_extract)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.scene_fbo.tex)
        glUniform1i(self.u_extract_tex, 0)
        glUniform1f(self.u_extract_threshold, float(glow.threshold))
        glUniform1f(self.u_extract_knee, self.glow_knee())

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        # 2) Walk down the chain
        glUseProgram(self.prog_down)
        glUniform1i(self.u_down_tex, 0)
        for i in range(1, levels + 1):
            self._dual_filter_pass(
                self.bloom_chain[i - 1], self.bloom_chain[i], self.u_down_halfpixel, offset
            )

        # 3) Walk back up, ending at chain[0] for the composite to sample
        glUseProgram(self.prog_up)
        glUniform1i(self.u_up_tex, 0)
        for i in range(levels, 0, -1):
            self._dual_filter_pass(
                self.bloom_chain[i], self.bloom_chain[i - 1], self.u_up_halfpixel, offset
            )

    def _dual_filter_pass(
        self,
        src: GLOffscreenTarget,
        dst: GLOffscreenTarget,
        halfpixel_loc: int,
        offset: float,
    ) -> None:
        """One dual-filter blit. ``u_halfpixel`` is half a DESTINATION texel."""
        glBindFramebuffer(GL_FRAMEBUFFER, dst.fbo)
        glViewport(0, 0, dst.width, dst.height)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex)
        glUniform2f(
            halfpixel_loc,
            0.5 * offset / float(dst.width),
            0.5 * offset / float(dst.height),
        )

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_targets(self) -> None:
        w, h = self.plot.fb_width, self.plot.fb_height
        if w <= 0 or h <= 0:
            return

        self._destroy_target(self.scene_fbo)
        self._destroy_bloom_chain()

        # Scene target: full res, float, needs depth buffer
        self.scene_fbo = self._create_target(w, h, GL_RGBA16F, with_depth=True)

        # Bloom mip chain: half res, then halving. RGBA16F to match the scene —
        # an RGBA8 chain clamped every bright pixel to 1.0 at the bright pass,
        # so bloom could never be brighter than its source and came out as a
        # flat grey wash with no HDR headroom at all.
        cw = max(1, int(round(w * 0.5)))
        ch = max(1, int(round(h * 0.5)))
        for _ in range(BLOOM_MAX_LEVELS + 1):
            self.bloom_chain.append(self._create_target(cw, ch, GL_RGBA16F))
            if cw // 2 < BLOOM_MIN_DIM or ch // 2 < BLOOM_MIN_DIM:
                break
            cw //= 2
            ch //= 2

    def _destroy_bloom_chain(self) -> None:
        for target in self.bloom_chain:
            self._destroy_target(target)
        self.bloom_chain = []

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
