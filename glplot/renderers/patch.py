from __future__ import annotations

import ctypes as C
from typing import TYPE_CHECKING, Optional

import numpy as np
from OpenGL.GL import *

from ..utils.gl_utils import link_program
from ..utils.scale import clamp_to_domain as _clamp_to_domain
from ..utils.scale import forward as _scale_forward
from ..utils.shaders import DENSITY_ACCUM_FS, PATCH_FS, PATCH_VS

# The dilation ring is defined once, in the 3D renderer, and imported rather than
# re-derived: a filled 2D patch and a 3D triangle mesh want the *same* silhouette, and two
# copies of the sampling rule would be two things to keep in step. `tests/test_outline.py`
# already covers the geometry of it.
from .geometry3d import _outline_offsets

if TYPE_CHECKING:
    from ..core.context import RenderContext
    from ..core.layers import PatchLayer
    from ..options import EngineOptions


class GLPatchBuffers:
    def __init__(
        self, vao: int, vbo: int, ebo: Optional[int] = None, cbo: Optional[int] = None
    ) -> None:
        self.vao = vao
        self.vbo = vbo
        self.ebo = ebo
        #: Per-vertex colour buffer. None for the overwhelming majority of patches, which
        #: are one `face_color` and take the shader's uniform path instead.
        self.cbo = cbo
        self.count = 0


class PatchRenderer:
    """Primitive renderer for PatchLayer.

    Specialized for area fills, bars, and bands using GL_TRIANGLE_STRIP/TRIANGLES.

    Outlines
    --------
    ``style.outline_enabled`` puts a silhouette around the filled shape. A fill has no
    width uniform to widen -- unlike a stroke, which is why
    :class:`~glplot.renderers.polyline.PolylineRenderer` gets away with one extra draw --
    so this is the same **screen-space dilation** the 3D renderer uses on triangles: the
    layer is redrawn :data:`~glplot.renderers.geometry3d.OUTLINE_SOLID_STEPS` times, each
    copy displaced ``outline_width`` px in a different direction on a circle, and the union
    of the copies is the shape's screen footprint grown by a disc. **8 extra draws** of the
    layer, which is why the feature is per-layer and off by default.

    The copies go down **first**, before the fill, and that is the one place this differs
    from the 3D implementation. There, the geometry is drawn first so that the depth it
    writes rejects the copies over its own body -- the order that keeps a *translucent*
    mesh from coming out solid outline colour. 2D layers render with ``GL_DEPTH_TEST``
    off, so there is no depth buffer to arbitrate with and painter's order is all there is:
    copies first, fill on top of them, rim left showing around the edge. This is the same
    fallback the 3D renderer uses for its own depth-test-less layer, the axis3d overlay.

    Rejected alternatives:

    * *Enabling the depth test for this layer only*, to reproduce the 3D trick exactly.
      It would leak: the fill's depth values stay in the buffer and would occlude any 3D
      layer drawn after it in a mixed scene, and the interaction-cache framebuffer has no
      depth attachment at all -- so the outline would flood the shape while the user drags
      and behave while they do not. A mode-dependent picture is worse than a documented
      limitation.
    * *Pushing the vertices outward from the shape's centroid.* One extra draw, but the
      width of the rim then varies with each vertex's distance from the centroid, and for a
      concave shape it moves vertices the wrong way entirely.
    * *Walking the triangulation's boundary edges on the CPU and stroking it.* Exact, and
      the right answer for a static polygon -- but a PatchLayer is a triangle soup that is
      rebuilt whenever the data changes (hexbin, fill_between, bars), so the boundary would
      have to be re-extracted on every update, and a strip's boundary is not well defined
      when the strip self-intersects.

    Limits, honestly:

    * A **translucent** fill shows its own silhouette through itself, because the
      silhouette is underneath it and nothing can hide it. Expect a tint towards the
      outline colour on a ``face_color`` whose alpha is low.
    * The dilation outlines every boundary between "this layer drew here" and "it did
      not" -- the outer silhouette **and** any holes -- but not the seams between adjacent
      triangles inside the shape, which is what you want for a bar or a hexbin.
    * A patch with no ``face_color`` and no per-vertex colours draws nothing, and therefore
      gets no outline either. An outline is the edge of a fill, not a substitute for one.
    * The copies overlap one another, so a translucent *outline*
      (``outline_alpha < 1``) compounds where they do and the rim reads unevenly. There is
      no stencil buffer in this pipeline to suppress that with.
    """

    def __init__(self, options: EngineOptions) -> None:
        self.options = options
        self.prog = 0
        self.u_mvp = -1
        self.u_color = -1
        self.u_alpha = -1
        self.u_offset = -1
        self.u_window = -1
        self.u_vertex_color = -1
        self.u_accum_window = -1
        self.u_accum_color = -1
        self.u_accum_vertex_color = -1
        # Outline / silhouette; see the class docstring.
        self.u_outline_pass = -1
        self.u_outline_offset = -1
        self.u_outline_color = -1
        self.u_outline_alpha = -1
        self.u_viewport = -1

    def initialize(self) -> None:
        self.prog = link_program(PATCH_VS, PATCH_FS)
        self.u_mvp = glGetUniformLocation(self.prog, "u_mvp")
        self.u_color = glGetUniformLocation(self.prog, "u_color")
        self.u_alpha = glGetUniformLocation(self.prog, "u_alpha")
        self.u_offset = glGetUniformLocation(self.prog, "u_layer_offset")
        self.u_window = glGetUniformLocation(self.prog, "u_window")
        self.u_vertex_color = glGetUniformLocation(self.prog, "u_use_vertex_color")
        self.u_outline_pass = glGetUniformLocation(self.prog, "u_outline_pass")
        self.u_outline_offset = glGetUniformLocation(self.prog, "u_outline_offset")
        self.u_outline_color = glGetUniformLocation(self.prog, "u_outline_color")
        self.u_outline_alpha = glGetUniformLocation(self.prog, "u_outline_alpha")
        self.u_viewport = glGetUniformLocation(self.prog, "u_viewport")

        # Density accumulation (Triangles into R32F)
        self.accum_prog = link_program(PATCH_VS, DENSITY_ACCUM_FS)
        self.u_accum_mvp = glGetUniformLocation(self.accum_prog, "u_mvp")
        self.u_accum_alpha = glGetUniformLocation(self.accum_prog, "u_alpha")
        self.u_accum_weighted = glGetUniformLocation(self.accum_prog, "u_density_weighted")
        self.u_accum_window = glGetUniformLocation(self.accum_prog, "u_window")
        # The fill colour, needed since the density buffer started carrying colour as well
        # as weight (``renderers/density.py``). It went unset before, when nothing in the
        # accumulation pass read anything but the alpha.
        self.u_accum_color = glGetUniformLocation(self.accum_prog, "u_color")
        self.u_accum_vertex_color = glGetUniformLocation(self.accum_prog, "u_use_vertex_color")

    def _create_buffers(self, layer: PatchLayer) -> GLPatchBuffers:
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))

        # Allocated only when the layer actually carries per-vertex colours: attribute 1
        # stays disabled otherwise, and `u_use_vertex_color` keeps the shader off it, so a
        # one-colour patch costs neither the buffer nor a read of an undefined attribute.
        cbo = None
        if layer.colors is not None:
            cbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, cbo)
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))

        ebo = None
        if layer.indices is not None:
            ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)

        glBindVertexArray(0)
        return GLPatchBuffers(vao, vbo, ebo, cbo)

    def update_gpu_data(self, layer: PatchLayer, bufs: GLPatchBuffers) -> None:
        if layer.vertices is None or len(layer.vertices) == 0:
            return

        # `layer.vertices` itself is never touched -- it is the Data panel's and CSV
        # export's single source of raw truth -- so a real axis scale transforms a
        # throwaway copy here, at upload time, exactly like scatter.py/polyline.py.
        scale_x = getattr(self.options, "axis_scale_x", "linear")
        scale_y = getattr(self.options, "axis_scale_y", "linear")
        if scale_x == "linear" and scale_y == "linear":
            gpu_vertices = layer.vertices
        else:
            params_x = getattr(self.options, "axis_scale_params_x", None)
            params_y = getattr(self.options, "axis_scale_params_y", None)
            xs, ys = layer.vertices[:, 0], layer.vertices[:, 1]
            # A patch is a triangle soup, not points/segments: one NaN corner (what a
            # bar's baseline at 0 would produce under plain log) corrupts the whole
            # triangle, not just that one vertex the way it harmlessly drops a scatter
            # point or opens a line gap. symlog/asinh have no such corner (both are
            # defined at 0), so only "log"/"logit" need the clamp -- see
            # `glplot.utils.scale.clamp_to_domain` (a no-op for every other mode).
            xs = _clamp_to_domain(xs, scale_x, params_x)
            ys = _clamp_to_domain(ys, scale_y, params_y)
            gpu_vertices = np.column_stack(
                [_scale_forward(xs, scale_x, params_x), _scale_forward(ys, scale_y, params_y)]
            ).astype(np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo)
        glBufferData(GL_ARRAY_BUFFER, gpu_vertices.nbytes, gpu_vertices, GL_STATIC_DRAW)

        if layer.colors is not None and bufs.cbo is not None:
            glBindBuffer(GL_ARRAY_BUFFER, bufs.cbo)
            glBufferData(GL_ARRAY_BUFFER, layer.colors.nbytes, layer.colors, GL_STATIC_DRAW)

        if layer.indices is not None and bufs.ebo is not None:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufs.ebo)
            glBufferData(
                GL_ELEMENT_ARRAY_BUFFER, layer.indices.nbytes, layer.indices, GL_STATIC_DRAW
            )
            bufs.count = len(layer.indices)
        else:
            bufs.count = len(layer.vertices)

        layer.dirty.gpu_dirty = False

    def draw(self, layer: PatchLayer, ctx: RenderContext) -> None:
        if layer.vertices is None or len(layer.vertices) == 0:
            return

        if not hasattr(layer, "_gl") or layer._gl is None:
            layer._gl = self._create_buffers(layer)
            layer.dirty.gpu_dirty = True

        if layer.dirty.gpu_dirty:
            self.update_gpu_data(layer, layer._gl)

        glUseProgram(self.prog)
        glUniformMatrix4fv(self.u_mvp, 1, GL_TRUE, ctx.mvp)
        glUniform4f(self.u_window, *ctx.window_world)
        overrides = self.options.visual.overrides
        alpha = ctx.global_alpha * layer.style.alpha * overrides.alpha_multiplier
        glUniform1f(self.u_alpha, float(alpha))
        glUniform2f(self.u_offset, *layer.translation)
        outline_width = self._push_outline_uniforms(layer, ctx)

        # Draw face. Per-vertex colours are a face of their own: the layer has no single
        # `face_color` to gate on -- that is the whole reason it carries a colour buffer --
        # so gating on face_color alone would draw a hexbin as nothing at all.
        vertex_coloured = layer.colors is not None and layer._gl.cbo is not None
        if layer.style.face_color is not None or vertex_coloured:
            glUniform1i(self.u_vertex_color, 1 if vertex_coloured else 0)
            # Still uploaded when unused: the shader reads one or the other, and leaving a
            # stale u_color behind would surface the moment a layer flips the flag off.
            glUniform4f(self.u_color, *(layer.style.face_color or (0.0, 0.0, 0.0, 1.0)))
            glBindVertexArray(layer._gl.vao)

            mode = GL_TRIANGLE_STRIP
            if layer.mode == "triangles":
                mode = GL_TRIANGLES

            # Silhouette first, fill on top: see the class docstring for why 2D has to take
            # the order the 3D renderer reserves for its depth-test-less overlay.
            if outline_width > 0.0:
                self._draw_outline_pass(layer, mode, outline_width)
            self._issue_draw(layer, mode)

        glBindVertexArray(0)
        glUseProgram(0)

    # ------------------------------------------------------------------
    # Outline / silhouette
    # ------------------------------------------------------------------

    def _issue_draw(self, layer: PatchLayer, mode: int) -> None:
        """One draw of the layer's geometry. Assumes its VAO and the program are bound."""
        if layer._gl.ebo is not None:
            glDrawElements(mode, layer._gl.count, GL_UNSIGNED_INT, None)
        else:
            glDrawArrays(mode, 0, layer._gl.count)

    def _push_outline_uniforms(self, layer: PatchLayer, ctx: RenderContext) -> float:
        """Upload the outline uniforms; return the dilation radius in framebuffer pixels.

        0.0 means "no outline", and is what an untouched layer returns -- ``PATCH_VS`` then
        never touches ``gl_Position`` and ``PATCH_FS`` never takes its branch, so the
        picture is bit-for-bit the pre-outline one. Pushed unconditionally so no layer
        inherits the previous layer's outline state.
        """
        style = layer.style
        enabled = bool(getattr(style, "outline_enabled", False))
        # Logical px -> framebuffer px, the same conversion every other px style gets.
        width = float(getattr(style, "outline_width", 0.0)) * ctx.dpr
        if not enabled or not np.isfinite(width) or width <= 0.0:
            glUniform1i(self.u_outline_pass, 0)
            glUniform2f(self.u_outline_offset, 0.0, 0.0)
            return 0.0

        color = tuple(float(v) for v in getattr(style, "outline_color", (0.0, 0.0, 0.0, 1.0)))
        glUniform1i(self.u_outline_pass, 0)
        glUniform2f(self.u_outline_offset, 0.0, 0.0)
        glUniform4f(self.u_outline_color, *color)
        # The layer's own alpha is deliberately not folded in (see LayerStyle.outline_alpha);
        # the scene-wide fade is, so a globally dimmed scene dims its outlines too.
        glUniform1f(
            self.u_outline_alpha,
            float(getattr(style, "outline_alpha", 1.0)) * float(ctx.global_alpha),
        )
        # Viewport, not window: `_draw_panels` swaps fb_width/fb_height to this panel's
        # rectangle, and that is what the pixel-to-NDC conversion must divide by.
        glUniform2f(self.u_viewport, float(max(ctx.fb_width, 1)), float(max(ctx.fb_height, 1)))
        return width

    def _draw_outline_pass(self, layer: PatchLayer, mode: int, width_px: float) -> None:
        """Redraw the layer once per dilation direction, in the outline colour.

        Assumes the program and the layer's VAO are bound. Restores ``u_outline_pass`` and
        ``u_outline_offset`` afterwards, so the fill drawn straight after this -- and the
        next layer -- do not inherit them.
        """
        offsets = _outline_offsets(width_px, solid=True)
        if len(offsets) == 0:
            return
        glUniform1i(self.u_outline_pass, 1)
        for dx, dy in offsets:
            glUniform2f(self.u_outline_offset, float(dx), float(dy))
            self._issue_draw(layer, mode)
        glUniform1i(self.u_outline_pass, 0)
        glUniform2f(self.u_outline_offset, 0.0, 0.0)

    def draw_density(self, layer: PatchLayer, ctx: RenderContext) -> None:
        """Accumulate patch area into density heatmap."""
        if layer.vertices is None or len(layer.vertices) == 0:
            return

        # 1. Resource Management
        if not hasattr(layer, "_gl") or layer._gl is None:
            layer._gl = self._create_buffers(layer)
            layer.dirty.gpu_dirty = True
        if layer.dirty.gpu_dirty:
            self.update_gpu_data(layer, layer._gl)

        # 2. Setup Shaders
        glUseProgram(self.accum_prog)
        glUniformMatrix4fv(self.u_accum_mvp, 1, GL_TRUE, ctx.mvp)
        glUniform4f(self.u_accum_window, *ctx.window_world)
        overrides = self.options.visual.overrides
        if self.options.density_weighted:
            glUniform1i(self.u_accum_weighted, 1)
            alpha = ctx.global_alpha * layer.style.alpha * overrides.alpha_multiplier
            glUniform1f(self.u_accum_alpha, float(alpha))
        else:
            glUniform1i(self.u_accum_weighted, 0)
            glUniform1f(self.u_accum_alpha, 1.0)
        # The same fill the exact pass uses, for the tinted resolve to paint with. The
        # weight is untouched: unweighted accumulation is a flat 1.0 and never reads it.
        vertex_coloured = layer.colors is not None and layer._gl.cbo is not None
        glUniform1i(self.u_accum_vertex_color, 1 if vertex_coloured else 0)
        glUniform4f(self.u_accum_color, *(layer.style.face_color or (0.0, 0.0, 0.0, 1.0)))

        # 3. Draw call
        glBindVertexArray(layer._gl.vao)
        mode = GL_TRIANGLE_STRIP
        if layer.mode == "triangles":
            mode = GL_TRIANGLES

        if layer._gl.ebo is not None:
            glDrawElements(mode, layer._gl.count, GL_UNSIGNED_INT, None)
        else:
            glDrawArrays(mode, 0, layer._gl.count)

        glBindVertexArray(0)
        glUseProgram(0)
