from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from OpenGL.GL import *

from ..utils.gl_utils import link_program
from ..utils.scale import forward as _scale_forward
from ..utils.shaders import (
    WIDE_SEGMENT_DENSITY_FS,
    WIDE_SEGMENT_INSTANCED_FS,
    WIDE_SEGMENT_INSTANCED_VS,
)

if TYPE_CHECKING:
    from ..core.context import RenderContext
    from ..core.layers import PolylineLayer
    from ..options import EngineOptions


@dataclass
class GLWideSegmentBuffers:
    vao: int = 0
    vbo_quad: int = 0
    vbo_inst: int = 0
    vbo_colors: int = 0  # per-segment (col0, col1) for a data-driven line; 0 when flat-coloured
    ebo: int = 0
    instance_count: int = 0
    stride: int = 1  # LOD decimation stride the current buffers were built at (1 = full detail)


class PolylineRenderer:
    """High-performance renderer for PolylineLayer using GPU Instancing.

    Replaces the older CPU-side expansion logic.

    Outlines
    --------
    ``style.outline_enabled`` puts a **casing** around the stroke: the standard way a line
    is made readable over a busy background. The layer is drawn twice through the same
    instanced call -- once with ``u_outline_pass = 1``, which widens every segment quad by
    ``outline_width`` px on each side and lengthens it by the same at both ends, and then
    once normally on top. **One extra draw call**, whatever the point count.

    That is deliberately *not* the K-copy screen-space dilation
    :class:`~glplot.renderers.geometry3d.Geometry3DRenderer` uses. It does not have to be:
    a 3D line is a ``GL_LINES`` primitive the driver rasterises one pixel wide, so the only
    way to widen it is to redraw it displaced; a 2D polyline is already expanded into quads
    *by the vertex shader*, whose width is a uniform. Widening that uniform gives an exact,
    uniform-width casing with the same joins and the same caps as the stroke, for 1 extra
    draw instead of 8-24.

    Rejected alternatives:

    * ``glLineWidth``. Core-profile OpenGL guarantees only 1.0, and Apple's driver reports
      exactly ``[1, 1]`` for ``GL_ALIASED_LINE_WIDTH_RANGE`` -- a silent no-op here.
    * *Casing drawn on top, with the stroke punched out of it.* Needs a stencil buffer;
      this pipeline has none.
    * *Offsetting the path in world space.* The casing's width would then vary with the
      zoom and with the aspect ratio, which is exactly what a casing must not do.

    Limits, honestly:

    * The casing is drawn **under** the stroke, which is the only order painter's algorithm
      offers -- 2D layers render with ``GL_DEPTH_TEST`` off, so there is no depth buffer to
      reject the casing over the stroke's own body with (the trick that lets the 3D
      renderer draw its geometry first). A **translucent** stroke therefore shows its own
      casing through itself and comes out tinted towards the outline colour. An outline is
      for making an opaque line readable; on a translucent line, expect the tint.
    * Self-intersections are not merged: where the path crosses itself, the later segment's
      casing lies over the earlier segment's stroke. One casing pass cannot see the whole
      path at once.
    * The casing follows the same LOD decimation as the stroke, so both stay in step when
      the line is decimated on zoom-out.
    """

    def __init__(self, options: EngineOptions) -> None:
        self.options = options

        self.prog = 0
        self.u_mvp = -1
        self.u_viewport = -1
        self.u_width = -1
        self.u_color = -1
        self.u_alpha = -1
        self.u_offset = -1
        self.u_use_colormap = -1
        self.u_scheme = -1
        self.u_id_norm = -1
        self.u_window = -1
        self.u_vertex_color = -1
        # Outline / casing; see the class docstring.
        self.u_outline_pass = -1
        self.u_outline_width = -1
        self.u_outline_color = -1
        self.u_outline_alpha = -1

        self.accum_prog = 0
        self.u_acc_mvp = -1
        self.u_acc_viewport = -1
        self.u_acc_width = -1
        self.u_acc_alpha = -1
        self.u_acc_weighted = -1
        self.u_acc_offset = -1
        self.u_acc_window = -1
        self.u_acc_color = -1
        self.u_acc_vertex_color = -1

    def initialize(self) -> None:
        self.prog = link_program(WIDE_SEGMENT_INSTANCED_VS, WIDE_SEGMENT_INSTANCED_FS)
        self.u_mvp = glGetUniformLocation(self.prog, "u_mvp")
        self.u_viewport = glGetUniformLocation(self.prog, "u_viewport_size")
        self.u_width = glGetUniformLocation(self.prog, "u_width")
        self.u_color = glGetUniformLocation(self.prog, "u_color")
        self.u_alpha = glGetUniformLocation(self.prog, "u_alpha")
        self.u_offset = glGetUniformLocation(self.prog, "u_layer_offset")
        self.u_use_colormap = glGetUniformLocation(self.prog, "u_use_colormap")
        self.u_scheme = glGetUniformLocation(self.prog, "u_scheme")
        self.u_id_norm = glGetUniformLocation(self.prog, "u_id_norm")
        self.u_window = glGetUniformLocation(self.prog, "u_window")
        self.u_vertex_color = glGetUniformLocation(self.prog, "u_use_vertex_color")
        self.u_outline_pass = glGetUniformLocation(self.prog, "u_outline_pass")
        self.u_outline_width = glGetUniformLocation(self.prog, "u_outline_width")
        self.u_outline_color = glGetUniformLocation(self.prog, "u_outline_color")
        self.u_outline_alpha = glGetUniformLocation(self.prog, "u_outline_alpha")

        self.accum_prog = link_program(WIDE_SEGMENT_INSTANCED_VS, WIDE_SEGMENT_DENSITY_FS)
        self.u_acc_mvp = glGetUniformLocation(self.accum_prog, "u_mvp")
        self.u_acc_viewport = glGetUniformLocation(self.accum_prog, "u_viewport_size")
        self.u_acc_width = glGetUniformLocation(self.accum_prog, "u_width")
        self.u_acc_alpha = glGetUniformLocation(self.accum_prog, "u_alpha")
        self.u_acc_weighted = glGetUniformLocation(self.accum_prog, "u_density_weighted")
        self.u_acc_offset = glGetUniformLocation(self.accum_prog, "u_layer_offset")
        self.u_acc_window = glGetUniformLocation(self.accum_prog, "u_window")
        # The stroke colour, which the accumulation pass needs since the density buffer
        # started carrying colour as well as weight (``renderers/density.py``). Before that
        # it went unset, and ``u_color`` sat at its default zero -- harmless while the only
        # thing read off it was ``.a`` in the ``density_weighted`` branch, which is off by
        # default and accumulated nothing at all when it was on.
        self.u_acc_color = glGetUniformLocation(self.accum_prog, "u_color")
        self.u_acc_vertex_color = glGetUniformLocation(self.accum_prog, "u_use_vertex_color")

    def _create_buffers(self, layer: PolylineLayer) -> GLWideSegmentBuffers:
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        # Static quad corners: (t, side)
        # t in [0, 1] goes from start to end of segment
        # side in [-0.5, 0.5] handles expansion around centerline
        quad = np.array(
            [
                [0.0, -0.5],
                [0.0, 0.5],
                [1.0, -0.5],
                [1.0, 0.5],
            ],
            dtype=np.float32,
        )

        indices = np.array([0, 1, 2, 2, 1, 3], dtype=np.uint16)

        vbo_quad = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_quad)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, C.c_void_p(0))

        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Per-instance segment buffer: [p0.x, p0.y, p1.x, p1.y]
        vbo_inst = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_inst)
        glBufferData(GL_ARRAY_BUFFER, 16, None, GL_STREAM_DRAW)  # Initial tiny allocation

        stride = 4 * 4
        # i_p0 (location 1)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, C.c_void_p(0))
        glVertexAttribDivisor(1, 1)

        # i_p1 (location 2)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, C.c_void_p(8))
        glVertexAttribDivisor(2, 1)

        # Per-segment colour pair (i_col0 @3, i_col1 @4). Only bound when the line carries a
        # per-vertex colour; otherwise the attributes stay disabled and u_use_vertex_color
        # keeps the shader off them, exactly as the patch renderer does for its faces.
        vbo_colors = 0
        if getattr(layer, "colors", None) is not None:
            vbo_colors = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_colors)
            glBufferData(GL_ARRAY_BUFFER, 32, None, GL_STREAM_DRAW)  # (col0[4], col1[4])
            col_stride = 8 * 4
            glEnableVertexAttribArray(3)
            glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, col_stride, C.c_void_p(0))
            glVertexAttribDivisor(3, 1)
            glEnableVertexAttribArray(4)
            glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, col_stride, C.c_void_p(16))
            glVertexAttribDivisor(4, 1)

        glBindVertexArray(0)
        return GLWideSegmentBuffers(
            vao=vao,
            vbo_quad=vbo_quad,
            vbo_inst=vbo_inst,
            vbo_colors=vbo_colors,
            ebo=ebo,
            instance_count=0,
        )

    def _decimated_indices(self, n: int, stride: int) -> np.ndarray:
        """Indices kept at ``stride`` (every k-th point), always including the last point.

        Uniform stride preserves the path's connectivity -- the survivors stay in order, so
        the decimated line is the same path drawn coarser, not a broken set of pieces.
        """
        if stride <= 1:
            return np.arange(n)
        idx = np.arange(0, n, stride)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)
        return idx

    def update_gpu_data(
        self, layer: PolylineLayer, bufs: GLWideSegmentBuffers, stride: int = 1
    ) -> None:
        if layer.pts is None or len(layer.pts) < 2:
            bufs.instance_count = 0
            bufs.stride = stride
            layer.dirty.gpu_dirty = False
            return

        # LOD: keep every ``stride``-th point. A polyline zoomed far out has thousands of
        # points landing on a handful of pixels; drawing them all costs vertex + setup work
        # (and overdraw) that grows as you zoom out, for a picture the extra points do not
        # change. Decimating keeps the drawn count proportional to on-screen detail.
        # `layer.pts` itself is never touched -- it is the Data panel's and CSV export's
        # single source of raw truth -- so a log axis transforms a throwaway copy here, at
        # upload time. Non-positive values become NaN, which `nansum` above and the GPU both
        # already treat as a legitimate line gap (a pole in 1/x, a missing sample).
        scale_x = getattr(self.options, "axis_scale_x", "linear")
        scale_y = getattr(self.options, "axis_scale_y", "linear")
        if scale_x == "linear" and scale_y == "linear":
            pts_full = np.ascontiguousarray(layer.pts, dtype=np.float32)
        else:
            params_x = getattr(self.options, "axis_scale_params_x", None)
            params_y = getattr(self.options, "axis_scale_params_y", None)
            pts_full = np.column_stack(
                [
                    _scale_forward(layer.pts[:, 0], scale_x, params_x),
                    _scale_forward(layer.pts[:, 1], scale_y, params_y),
                ]
            ).astype(np.float32)
        n = len(pts_full)
        idx = self._decimated_indices(n, stride)
        pts = pts_full[idx]

        segs = np.empty((len(pts) - 1, 4), dtype=np.float32)
        segs[:, 0:2] = pts[:-1]
        segs[:, 2:4] = pts[1:]

        glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_inst)
        # Buffer Orphaning for performance
        glBufferData(GL_ARRAY_BUFFER, segs.nbytes, None, GL_STREAM_DRAW)
        glBufferSubData(GL_ARRAY_BUFFER, 0, segs.nbytes, segs)

        # Per-segment colours: each segment interpolates from its start vertex's colour to its
        # end vertex's, so segment i carries (colors[i], colors[i+1]). Uploaded only when the
        # line is data-coloured and the colour buffer exists (see _create_buffers).
        colors = getattr(layer, "colors", None)
        if colors is not None and bufs.vbo_colors:
            cols = np.ascontiguousarray(colors, dtype=np.float32)
            if cols.ndim == 2 and cols.shape[0] == n and cols.shape[1] == 4:
                cols = cols[idx]
                seg_cols = np.empty((len(pts) - 1, 8), dtype=np.float32)
                seg_cols[:, 0:4] = cols[:-1]
                seg_cols[:, 4:8] = cols[1:]
                glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_colors)
                glBufferData(GL_ARRAY_BUFFER, seg_cols.nbytes, None, GL_STREAM_DRAW)
                glBufferSubData(GL_ARRAY_BUFFER, 0, seg_cols.nbytes, seg_cols)

        bufs.instance_count = len(segs)
        bufs.stride = stride
        layer.dirty.gpu_dirty = False

    def _lod_stride(self, layer: PolylineLayer, ctx: RenderContext) -> int:
        """Power-of-two decimation stride for the line at the current zoom, or 1 for full detail.

        Keeps roughly ``default_line_budget_per_px`` points per pixel of the line's on-screen
        length. Snapped to a power of two so the stride -- and the buffer re-upload it triggers
        -- changes only at octave boundaries, not every frame of a continuous zoom.
        """
        n = len(layer.pts)
        if n < 8 or not getattr(self.options, "lod_enabled", True):
            return 1
        # On-screen length of the (sampled) path in pixels.
        max_samples = 4096
        s = max(1, (n - 1) // max_samples)
        sample = layer.pts[::s]
        if len(sample) < 2:
            return 1
        l, r, b, t = ctx.window_world
        sx = ctx.fb_width / max(r - l, 1e-12)
        sy = ctx.fb_height / max(t - b, 1e-12)
        diffs = np.diff(np.asarray(sample, dtype=np.float64), axis=0)
        seg_px = np.linalg.norm(diffs * np.array([sx, sy])[None, :], axis=1)
        # nansum, because a nan in the points is legitimate: it is how a line marks a gap
        # (a pole in 1/x, a missing sample). A plain sum would make length_px nan and the
        # int() below would raise, taking the frame down over a perfectly valid line.
        length_px = float(np.nansum(seg_px) * s)
        if not np.isfinite(length_px):
            return 1
        budget = max(1, int(getattr(self.options, "default_line_budget_per_px", 8)))
        max_pts = max(2, int(length_px * budget))
        if n <= max_pts:
            return 1
        k = n // max_pts
        stride = 1
        while stride < k:
            stride *= 2
        return stride

    def draw(self, layer: PolylineLayer, ctx: RenderContext, id_norm: float = 0.0) -> None:
        if layer.pts is None or len(layer.pts) < 2:
            return

        if not hasattr(layer, "_gl") or layer._gl is None:
            layer._gl = self._create_buffers(layer)
            layer.dirty.gpu_dirty = True

        # Zoom-adaptive LOD: rebuild the segment buffers at a coarser stride when the line is
        # oversampled for its on-screen size. Re-upload only when the stride actually changes
        # (octave-snapped) or the data changed, so a continuous zoom is not a per-frame upload.
        stride = self._lod_stride(layer, ctx)
        if layer.dirty.gpu_dirty or stride != layer._gl.stride:
            self.update_gpu_data(layer, layer._gl, stride=stride)

        overrides = self.options.visual.overrides
        width_px = max(1.0, layer.style.line_width * overrides.line_width_multiplier * ctx.dpr)
        alpha = ctx.global_alpha * layer.style.alpha * overrides.alpha_multiplier
        color = layer.style.color if layer.style.color is not None else (0.0, 0.0, 0.0, 1.0)

        glUseProgram(self.prog)
        glUniformMatrix4fv(self.u_mvp, 1, GL_TRUE, ctx.mvp)
        glUniform2f(self.u_viewport, float(ctx.fb_width), float(ctx.fb_height))
        glUniform1f(self.u_width, float(width_px))
        glUniform4f(self.u_color, *color)
        glUniform1f(self.u_alpha, float(alpha))
        glUniform2f(self.u_offset, *layer.translation)
        # Data-driven line colour: on only when the layer carries a per-vertex colour and its
        # buffer was actually built. Otherwise the flat u_color path is used, unchanged.
        vertex_coloured = getattr(layer, "colors", None) is not None and bool(
            getattr(layer._gl, "vbo_colors", 0)
        )
        glUniform1i(self.u_vertex_color, 1 if vertex_coloured else 0)

        # Per-layer colormap override, falling back to the scene-global options.
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
        glUniform1f(self.u_id_norm, float(id_norm))
        l, r, b, t = ctx.window_world
        glUniform4f(self.u_window, l, r, b, t)

        outline_width = self._push_outline_uniforms(layer, ctx)

        glBindVertexArray(layer._gl.vao)
        # Casing first, stroke second. 2D draws with the depth test off, so painter's order
        # is the only arbiter there is: a casing laid down after the stroke would simply
        # paint over it in the outline colour. This is the same order the 3D renderer falls
        # back to for its one depth-test-less layer, the axis3d overlay.
        if outline_width > 0.0:
            glUniform1i(self.u_outline_pass, 1)
            glDrawElementsInstanced(
                GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, C.c_void_p(0), layer._gl.instance_count
            )
            glUniform1i(self.u_outline_pass, 0)
        glDrawElementsInstanced(
            GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, C.c_void_p(0), layer._gl.instance_count
        )
        glBindVertexArray(0)
        glUseProgram(0)

    def _push_outline_uniforms(self, layer: PolylineLayer, ctx: RenderContext) -> float:
        """Upload the casing uniforms; return its width per side, in framebuffer pixels.

        0.0 means "no casing", and is what an untouched layer returns: ``u_outline_pass``
        stays 0, the whole casing block in ``WIDE_SEGMENT_INSTANCED_VS`` is skipped, and
        the stroke's geometry is the same float-for-float as before outlines existed. The
        uniforms are pushed unconditionally so no layer inherits the previous layer's.
        """
        style = layer.style
        enabled = bool(getattr(style, "outline_enabled", False))
        # Logical px -> framebuffer px, the same conversion line_width gets.
        width = float(getattr(style, "outline_width", 0.0)) * ctx.dpr
        if not enabled or not np.isfinite(width) or width <= 0.0:
            glUniform1i(self.u_outline_pass, 0)
            glUniform1f(self.u_outline_width, 0.0)
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

    def draw_density(self, layer: PolylineLayer, ctx: RenderContext) -> None:
        if layer.pts is None or len(layer.pts) < 2:
            return

        if not hasattr(layer, "_gl") or layer._gl is None:
            layer._gl = self._create_buffers(layer)
            layer.dirty.gpu_dirty = True

        stride = self._lod_stride(layer, ctx)
        if layer.dirty.gpu_dirty or stride != layer._gl.stride:
            self.update_gpu_data(layer, layer._gl, stride=stride)

        overrides = self.options.visual.overrides
        width_px = max(1.0, layer.style.line_width * overrides.line_width_multiplier * ctx.dpr)

        if self.options.density_weighted:
            alpha = ctx.global_alpha * layer.style.alpha * overrides.alpha_multiplier
            weighted = 1
        else:
            alpha = 1.0
            weighted = 0

        glUseProgram(self.accum_prog)
        glUniformMatrix4fv(self.u_acc_mvp, 1, GL_TRUE, ctx.mvp)
        glUniform2f(self.u_acc_viewport, float(ctx.fb_width), float(ctx.fb_height))
        glUniform1f(self.u_acc_width, float(width_px))
        glUniform1f(self.u_acc_alpha, float(alpha))
        glUniform1i(self.u_acc_weighted, weighted)
        glUniform2f(self.u_acc_offset, *layer.translation)
        # Same colour the exact pass draws with, so a tinted density is the same colour the
        # user sees when they turn density off.
        color = layer.style.color if layer.style.color is not None else (0.0, 0.0, 0.0, 1.0)
        glUniform4f(self.u_acc_color, *color)
        vertex_coloured = getattr(layer, "colors", None) is not None and bool(
            getattr(layer._gl, "vbo_colors", 0)
        )
        glUniform1i(self.u_acc_vertex_color, 1 if vertex_coloured else 0)
        l, r, b, t = ctx.window_world
        glUniform4f(self.u_acc_window, l, r, b, t)

        glBindVertexArray(layer._gl.vao)
        glDrawElementsInstanced(
            GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, C.c_void_p(0), layer._gl.instance_count
        )
        glBindVertexArray(0)
        glUseProgram(0)
