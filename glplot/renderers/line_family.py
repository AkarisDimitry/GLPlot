from __future__ import annotations

import ctypes as C
import warnings
from typing import TYPE_CHECKING

import numpy as np
from OpenGL.GL import *

from ..utils.gl_utils import link_program
from ..utils.shaders import DENSITY_ACCUM_FS, EXACT_LINES_FS, WIDE_LINES_INSTANCED_VS
from .base import GLLineBuffers

if TYPE_CHECKING:
    from ..core.context import RenderContext
    from ..core.layers import LineFamilyLayer
    from ..options import EngineOptions


#: Largest family that may carry an outline. Above this the renderer refuses it and says so
#: once, rather than shipping a silent halving of the frame rate.
#:
#: The casing is one extra instanced draw of the whole layer -- the cheapest outline in this
#: codebase in *draw calls*, and still exactly 2x the cost of the layer, because this layer
#: has nothing else in it. A million-line family is the case this renderer exists for, and
#: it is fill-rate bound; doubling it is a frame-rate decision no style toggle should be
#: allowed to make silently.
#:
#: The cap is also where the outline stops meaning anything. Past a few hundred thousand
#: lines a family is a solid field of ink whose individual lines are sub-pixel and overlap
#: everywhere, so each line's casing lands on its neighbours' bodies: the picture that comes
#: back is uniformly outline-coloured, not outlined. Families small enough to read
#: individual lines -- a regression fan, a bundle of trajectories -- are exactly the ones
#: under this bound, and they get the outline.
OUTLINE_MAX_INSTANCES = 200_000


class LineFamilyRenderer:
    """Primitive renderer for LineFamilyLayer.

    Specialized for high-performance instanced rendering of lines via quad expansion.
    Uses optimized orthographic shaders for maximum throughput.

    Outlines
    --------
    ``style.outline_enabled`` puts the same **casing** on a family that
    :class:`~glplot.renderers.polyline.PolylineRenderer` puts on a single line, by the same
    means: ``WIDE_LINES_INSTANCED_VS`` builds each line's width from ``u_width``, so a
    casing is one re-draw of the layer with that width grown by ``2 * outline_width``,
    under the real geometry. **One extra draw call**, and the shader's own probabilistic
    LOD drops the same instances in both passes, so the casing never outlives its line.

    It is gated, and that gate is the honest part. See :data:`OUTLINE_MAX_INSTANCES`: one
    extra instanced draw is 2x the cost of a layer that *is* that draw, and this renderer's
    reason to exist is families of a million lines. Above the bound the outline is refused
    with a warning naming the count, not silently ignored and not silently applied.

    Limits, honestly:

    * The casing is drawn **under** the lines (2D renders with the depth test off, so
      painter's order is the only arbiter). A family drawn at a low per-line alpha -- the
      usual way a big family is drawn -- shows its casings through itself.
    * Lines within one family do not share a contour: each is cased independently, so where
      two lines cross, the later one's casing lies over the earlier one's body.
    * The casing is widened but not lengthened. Every line spans the layer's whole x range
      and is clipped to the window, so there are no free ends to cap.
    * ``u_use_colormap`` does not apply to the casing: it is flat ``outline_color``, by
      design. An outline that took the colormap would be a second line, not an edge.
    """

    def __init__(self, options: EngineOptions) -> None:
        self.options = options
        self.prog = 0
        #: Layer ids already warned about, so a refusal is reported once and not once per
        #: frame. Keyed by id, not by layer, so it cannot keep a deleted layer alive.
        self._outline_refused: set[int] = set()

        # Uniform locations (Exact)
        self.u_ndc_scale = -1
        self.u_ndc_offset = -1
        self.u_xrange = -1
        self.u_window = -1
        self.u_use_color = -1
        self.u_alpha = -1
        self.u_width = -1
        self.u_viewport = -1
        self.u_keep_prob = -1
        self.u_total_count = -1
        self.u_offset = -1
        self.u_use_colormap = -1
        self.u_scheme = -1
        self.u_antialiasing = -1
        # Outline / casing; see the class docstring and OUTLINE_MAX_INSTANCES.
        self.u_outline_pass = -1
        self.u_outline_width = -1
        self.u_outline_color = -1
        self.u_outline_alpha = -1

        # Accumulation uniforms
        self.accum_prog = 0
        self.u_accum_ndc_scale = -1
        self.u_accum_ndc_offset = -1
        self.u_accum_xrange = -1
        self.u_accum_window = -1
        self.u_accum_use_color = -1
        self.u_accum_alpha = -1
        self.u_accum_width = -1
        self.u_accum_viewport = -1
        self.u_accum_keep_prob = -1
        self.u_accum_offset = -1

        # Configuration
        self.use_fp16_ab = True
        self.use_packed_color = True

    def initialize(self) -> None:
        """Link wide-line instanced shaders and setup uniform locations."""
        self.prog = link_program(WIDE_LINES_INSTANCED_VS, EXACT_LINES_FS)
        self.u_ndc_scale = glGetUniformLocation(self.prog, "u_ndc_scale")
        self.u_ndc_offset = glGetUniformLocation(self.prog, "u_ndc_offset")
        self.u_xrange = glGetUniformLocation(self.prog, "u_xrange")
        self.u_window = glGetUniformLocation(self.prog, "u_window")
        self.u_use_color = glGetUniformLocation(self.prog, "u_use_color")
        self.u_alpha = glGetUniformLocation(self.prog, "u_alpha")
        self.u_width = glGetUniformLocation(self.prog, "u_width")
        self.u_viewport = glGetUniformLocation(self.prog, "u_viewport_size")
        self.u_keep_prob = glGetUniformLocation(self.prog, "u_keep_prob")
        self.u_total_count = glGetUniformLocation(self.prog, "u_total_count")
        self.u_offset = glGetUniformLocation(self.prog, "u_layer_offset")
        self.u_use_colormap = glGetUniformLocation(self.prog, "u_use_colormap")
        self.u_scheme = glGetUniformLocation(self.prog, "u_scheme")
        self.u_antialiasing = glGetUniformLocation(self.prog, "u_antialiasing")
        self.u_outline_pass = glGetUniformLocation(self.prog, "u_outline_pass")
        self.u_outline_width = glGetUniformLocation(self.prog, "u_outline_width")
        self.u_outline_color = glGetUniformLocation(self.prog, "u_outline_color")
        self.u_outline_alpha = glGetUniformLocation(self.prog, "u_outline_alpha")

        # Density Accumulation Program
        self.accum_prog = link_program(WIDE_LINES_INSTANCED_VS, DENSITY_ACCUM_FS)
        self.u_accum_ndc_scale = glGetUniformLocation(self.accum_prog, "u_ndc_scale")
        self.u_accum_ndc_offset = glGetUniformLocation(self.accum_prog, "u_ndc_offset")
        self.u_accum_xrange = glGetUniformLocation(self.accum_prog, "u_xrange")
        self.u_accum_window = glGetUniformLocation(self.accum_prog, "u_window")
        self.u_accum_use_color = glGetUniformLocation(self.accum_prog, "u_use_color")
        self.u_accum_alpha = glGetUniformLocation(self.accum_prog, "u_alpha")
        self.u_accum_width = glGetUniformLocation(self.accum_prog, "u_width")
        self.u_accum_viewport = glGetUniformLocation(self.accum_prog, "u_viewport_size")
        self.u_accum_keep_prob = glGetUniformLocation(self.accum_prog, "u_keep_prob")
        self.u_accum_offset = glGetUniformLocation(self.accum_prog, "u_layer_offset")

    def _create_buffers(self, layer: LineFamilyLayer) -> GLLineBuffers:
        """Create and initialize GPU buffers for a quad-expanded line family."""
        bufs = GLLineBuffers()
        bufs.vao = glGenVertexArrays(1)
        glBindVertexArray(bufs.vao)

        # 1. Base vertex buffer (a_corner: [t, side] x 4)
        # t in {0,1}, side in {-0.5, 0.5}
        bufs.vbo_base = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_base)
        quad_data = np.array([0.0, -0.5, 0.0, 0.5, 1.0, -0.5, 1.0, 0.5], dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, quad_data.nbytes, quad_data, GL_STATIC_DRAW)

        # a_corner (loc 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, C.c_void_p(0))

        # 2. Instanced line parameters (a, b) (loc 1)
        bufs.vbo_ab = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_ab)
        glBufferData(GL_ARRAY_BUFFER, 16, None, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        if self.use_fp16_ab:
            glVertexAttribPointer(1, 2, GL_HALF_FLOAT, GL_FALSE, 0, C.c_void_p(0))
        else:
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
        glVertexAttribDivisor(1, 1)

        # 3. Instanced colors (loc 2)
        bufs.vbo_col = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_col)
        glBufferData(GL_ARRAY_BUFFER, 16, None, GL_STATIC_DRAW)
        glEnableVertexAttribArray(2)
        if self.use_packed_color:
            glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, C.c_void_p(0))
        else:
            glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
        glVertexAttribDivisor(2, 1)

        glBindVertexArray(0)
        return bufs

    def update_gpu_data(self, layer: LineFamilyLayer, bufs: GLLineBuffers) -> None:
        """Upload semantic data to GPU buffers."""
        if layer.ab is None:
            return

        glBindVertexArray(bufs.vao)

        # Upload AB
        ab_u = layer.ab.astype(np.float16) if self.use_fp16_ab else layer.ab
        glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_ab)
        glBufferData(GL_ARRAY_BUFFER, ab_u.nbytes, ab_u, GL_STATIC_DRAW)

        # Upload Colors
        bufs.has_color = layer.colors is not None
        if bufs.has_color:
            if self.use_packed_color:
                cols_u8 = np.clip(layer.colors * 255.0, 0, 255).astype(np.uint8, copy=False)
                glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_col)
                glBufferData(GL_ARRAY_BUFFER, cols_u8.nbytes, cols_u8, GL_STATIC_DRAW)
            else:
                glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_col)
                glBufferData(GL_ARRAY_BUFFER, layer.colors.nbytes, layer.colors, GL_STATIC_DRAW)

        glBindVertexArray(0)
        layer.dirty.gpu_dirty = False

    def draw(self, layer: LineFamilyLayer, ctx: RenderContext) -> None:
        """Draw the layer using current context."""
        if layer.ab is None or len(layer.ab) == 0:
            return

        # 1. Resource Management
        if not hasattr(layer, "_gl") or layer._gl is None:
            layer._gl = self._create_buffers(layer)
            layer.dirty.gpu_dirty = True

        if layer.dirty.gpu_dirty:
            self.update_gpu_data(layer, layer._gl)

        # 2. Setup Shaders & Uniforms
        glUseProgram(self.prog)
        glUniform2f(self.u_ndc_scale, *ctx.ndc_scale)
        glUniform2f(self.u_ndc_offset, *ctx.ndc_offset)
        glUniform2f(self.u_viewport, float(ctx.fb_width), float(ctx.fb_height))
        glUniform2f(self.u_xrange, float(layer.x_range[0]), float(layer.x_range[1]))
        glUniform4f(self.u_window, *ctx.window_world)
        glUniform1i(self.u_use_color, 1 if layer._gl.has_color else 0)

        # Style Application
        overrides = self.options.visual.overrides
        alpha = ctx.global_alpha * layer.style.alpha * overrides.alpha_multiplier
        glUniform1f(self.u_alpha, float(alpha))

        width = layer.style.line_width * overrides.line_width_multiplier * ctx.dpr
        glUniform1f(self.u_width, float(width))

        # LOD
        prob = ctx.lod_keep_prob
        glUniform1f(self.u_keep_prob, float(prob))
        glUniform1i(self.u_total_count, len(layer.ab))
        glUniform2f(self.u_offset, *layer.translation)

        # Colormap — per-layer override, falling back to the scene-global options.
        # metadata is the source rather than layer.style: style.use_colormap/.cmap are
        # dead fields no renderer has ever read, and making them live here would change
        # the meaning of every existing LayerStyle. A missing key (the default for every
        # layer the engine builds) means "follow the scene", so behaviour is unchanged
        # unless something has explicitly opted out. See gui.layerops.set_layer_gl_colormap.
        use_cmap = layer.metadata.get("use_colormap")
        scheme = layer.metadata.get("cmap_index")
        if use_cmap is None:
            use_cmap = self.options.line_colormap_enabled
        glUniform1i(self.u_use_colormap, 1 if use_cmap else 0)
        glUniform1i(
            self.u_scheme,
            self.options.density_scheme_index if scheme is None else int(scheme),
        )
        # Antialiasing
        glUniform1i(self.u_antialiasing, 1 if self.options.enable_antialiasing else 0)

        outline_width = self._push_outline_uniforms(layer, ctx)

        # 3. Draw call (Instanced Quads)
        glBindVertexArray(layer._gl.vao)
        # Casing first, lines on top: 2D draws with the depth test off, so painter's order
        # is the only arbiter -- a casing laid down afterwards would paint over the family
        # in the outline colour instead of edging it.
        if outline_width > 0.0:
            glUniform1i(self.u_outline_pass, 1)
            glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, len(layer.ab))
            glUniform1i(self.u_outline_pass, 0)
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, len(layer.ab))

        # Cleanup
        glBindVertexArray(0)
        glUseProgram(0)

    def _push_outline_uniforms(self, layer: LineFamilyLayer, ctx: RenderContext) -> float:
        """Upload the casing uniforms; return its width per side, in framebuffer pixels.

        0.0 means "no casing" -- because the layer never asked for one, because the width
        rounds away to nothing, or because the family is larger than
        :data:`OUTLINE_MAX_INSTANCES`, in which case the refusal is warned about once per
        layer. ``u_outline_pass`` is left at 0 either way, so the vertex shader's casing
        block is skipped entirely and the geometry is float-for-float the pre-outline one.
        """
        style = layer.style
        enabled = bool(getattr(style, "outline_enabled", False))
        # Logical px -> framebuffer px, the same conversion line_width gets.
        width = float(getattr(style, "outline_width", 0.0)) * ctx.dpr
        if not enabled or not np.isfinite(width) or width <= 0.0:
            glUniform1i(self.u_outline_pass, 0)
            glUniform1f(self.u_outline_width, 0.0)
            return 0.0

        count = len(layer.ab)
        if count > OUTLINE_MAX_INSTANCES:
            glUniform1i(self.u_outline_pass, 0)
            glUniform1f(self.u_outline_width, 0.0)
            if layer.layer_id not in self._outline_refused:
                self._outline_refused.add(layer.layer_id)
                warnings.warn(
                    f"outline_enabled ignored on line family {layer.label or layer.layer_id!r}: "
                    f"{count:,} lines exceeds the {OUTLINE_MAX_INSTANCES:,}-line limit. The "
                    "casing is one extra full draw of the layer -- it would halve the frame "
                    "rate, and at this density every line's casing lands on its neighbours, "
                    "so the result would be a flat field of the outline colour rather than "
                    "an outline. Draw fewer lines, or use density mode.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return 0.0

        color = tuple(float(v) for v in getattr(style, "outline_color", (0.0, 0.0, 0.0, 1.0)))
        glUniform1i(self.u_outline_pass, 0)
        glUniform1f(self.u_outline_width, width)
        glUniform4f(self.u_outline_color, *color)
        # The layer's own alpha is deliberately not folded in (see LayerStyle.outline_alpha);
        # the scene-wide fade is, so a globally dimmed scene dims its casings too.
        glUniform1f(
            self.u_outline_alpha,
            float(getattr(style, "outline_alpha", 1.0)) * float(ctx.global_alpha),
        )
        return width

    def draw_density(self, layer: LineFamilyLayer, ctx: RenderContext) -> None:
        """Accumulate line density into the current R32F target."""
        if layer.ab is None or len(layer.ab) == 0:
            return

        # 1. Resource Management
        if not hasattr(layer, "_gl") or layer._gl is None:
            layer._gl = self._create_buffers(layer)
            layer.dirty.gpu_dirty = True

        if layer.dirty.gpu_dirty:
            self.update_gpu_data(layer, layer._gl)

        # 2. Setup Shaders
        glUseProgram(self.accum_prog)
        glUniform2f(self.u_accum_ndc_scale, *ctx.ndc_scale)
        glUniform2f(self.u_accum_ndc_offset, *ctx.ndc_offset)
        glUniform2f(self.u_accum_viewport, float(ctx.fb_width), float(ctx.fb_height))
        glUniform2f(self.u_accum_xrange, float(layer.x_range[0]), float(layer.x_range[1]))
        glUniform4f(self.u_accum_window, *ctx.window_world)

        overrides = self.options.visual.overrides
        width = max(1.0, layer.style.line_width * overrides.line_width_multiplier * ctx.dpr)
        glUniform1f(self.u_accum_width, float(width))

        # The per-line colours are read whenever the layer has them, not only in the
        # weighted branch: since the density buffer started carrying colour alongside weight
        # (``renderers/density.py``) they are also what a tinted resolve paints with. The
        # weight is untouched either way -- unweighted accumulation is a flat 1.0 per
        # fragment and never looks at the colour.
        glUniform1i(self.u_accum_use_color, 1 if layer._gl.has_color else 0)
        if self.options.density_weighted:
            alpha = ctx.global_alpha * layer.style.alpha * overrides.alpha_multiplier
            glUniform1f(self.u_accum_alpha, float(alpha))
        else:
            glUniform1f(self.u_accum_alpha, 1.0)

        # LOD
        prob = ctx.lod_keep_prob
        glUniform1f(self.u_accum_keep_prob, float(prob))
        glUniform2f(self.u_accum_offset, *layer.translation)

        # 3. Draw call
        glBindVertexArray(layer._gl.vao)
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, len(layer.ab))

        # Cleanup
        glBindVertexArray(0)
        glUseProgram(0)
