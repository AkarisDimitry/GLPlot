from __future__ import annotations

import ctypes as C
from typing import TYPE_CHECKING, Tuple

import numpy as np
from OpenGL.GL import *

from ..utils.gl_utils import link_program
from ..utils.scale import clamp_to_domain as _clamp_to_domain
from ..utils.scale import forward as _scale_forward
from ..utils.shaders import (
    DENSITY_POINTS_FS,
    DENSITY_POINTS_VS,
    IMAGE_FS,
    IMAGE_VS,
    SCATTER_FS,
    SCATTER_VS,
    marker_shape_index,
)
from .base import GLScatterBuffers

if TYPE_CHECKING:
    from ..core.context import RenderContext
    from ..core.layers import ScatterLayer
    from ..options import EngineOptions

#: Accent ring drawn around selected points. Warm amber reads as "selected" against
#: both the dark and the light background, and against every DENSITY_SCHEMES ramp.
SELECTION_OUTLINE_COLOR: Tuple[float, float, float, float] = (1.0, 0.78, 0.25, 1.0)

#: Selected points are drawn this much larger than the layer's point size, so a
#: selection is legible even where the cloud is dense enough to be solid.
SELECTION_SIZE_BOOST = 1.9

#: Ring thickness, in logical px (scaled by ctx.dpr like every other px style).
SELECTION_OUTLINE_WIDTH_PX = 2.0

#: A highlight must not inherit a nearly-transparent layer alpha: at alpha=0.05 the
#: selection would be invisible and the user would think the click missed.
SELECTION_MIN_ALPHA = 0.9


class ScatterRenderer:
    """Primitive renderer for ScatterLayer. Specialized for point clouds (GL_POINTS).

    Outlines
    --------
    Two independent rings can be drawn around a marker, and they are deliberately not the
    same switch (see :class:`~glplot.core.layers.LayerStyle`):

    * ``style.point_outline_*`` -- this renderer's original ring, drawn **inside** the
      sprite. The marker keeps its pixel size and the fill shrinks by the ring's width. It
      is what an existing script's ``edgecolors=`` sets, and it is left exactly as it was.
    * ``style.outline_enabled`` -- the general outline, drawn **outside** the marker. The
      sprite is grown by twice the outline width in ``SCATTER_VS``, a ``v_fill_frac``
      varying tells ``SCATTER_FS`` where the marker ends, and everything past that radius
      is painted in the outline colour. This is the same analytic ring
      ``Geometry3DRenderer`` puts on a 3D point cloud, for the same reason: it costs **no
      extra draw call at all**, one branch in each shader stage, and the marker keeps the
      size the caller asked for.

    Both can be on at once, in which case the general ring sits outside the per-point one.

    Rejected alternative: a second, larger draw of the whole cloud underneath the first.
    It doubles the draw cost of the layer for a worse result -- with alpha blending, the
    larger points show through translucent markers -- and a point sprite already has a
    fragment shader that knows its own radius, so there is nothing to gain by it.

    Limits, honestly:

    * ``GL_POINT_SIZE_RANGE`` caps a sprite (64 px on Apple's GL). A marker already near
      that cap cannot grow, so the ring eats into the fill instead of growing outside it.
    * The boundary between fill and ring is a hard step, like the per-point ring's own;
      only the sprite's outer rim is feathered.
    * With per-point sizes (``ScatterLayer.sizes``) the ring is a constant *pixel* width on
      every marker, which is what an outline should be -- it is not scaled per point.
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
        # The per-marker ring (style.point_outline_*), inside the sprite.
        self.u_point_outline_enabled = glGetUniformLocation(self.prog, "u_point_outline_enabled")
        self.u_point_outline_color = glGetUniformLocation(self.prog, "u_point_outline_color")
        self.u_point_outline_width_px = glGetUniformLocation(self.prog, "u_point_outline_width_px")
        # The general outline (style.outline_*), outside the marker. See the class docstring.
        self.u_marker_shape = glGetUniformLocation(self.prog, "u_marker_shape")

        self.u_outline_enabled = glGetUniformLocation(self.prog, "u_outline_enabled")
        self.u_outline_color = glGetUniformLocation(self.prog, "u_outline_color")
        self.u_outline_width = glGetUniformLocation(self.prog, "u_outline_width")
        self.u_outline_alpha = glGetUniformLocation(self.prog, "u_outline_alpha")

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

        # 3. Per-point size multiplier (attribute 2). Always enabled; carries 1.0 for every
        # point when the layer is uniform-sized, so the shader's `u_size * a_size` is a no-op.
        vbo_size = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_size)
        glBufferData(GL_ARRAY_BUFFER, 4, None, GL_STATIC_DRAW)  # Pre-allocate to avoid segfault
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))

        glBindVertexArray(0)
        return GLScatterBuffers(vao=vao, vbo_pts=vbo_pts, vbo_col=vbo_col, vbo_size=vbo_size)

    def update_gpu_data(self, layer: ScatterLayer, bufs: GLScatterBuffers) -> None:
        """Upload semantic points and colors to GPU buffers."""
        if layer.pts is None or len(layer.pts) == 0:
            return

        # Upload Positions. `layer.pts` itself is never touched -- it is the Data panel's and
        # CSV export's single source of raw truth -- so a log axis transforms a throwaway
        # copy here, at upload time, rather than the stored array. Non-positive values become
        # NaN, which the GPU naturally drops (the same convention a manual NaN gap uses).
        scale_x = getattr(self.options, "axis_scale_x", "linear")
        scale_y = getattr(self.options, "axis_scale_y", "linear")
        if scale_x == "linear" and scale_y == "linear":
            gpu_pts = layer.pts
        else:
            params_x = getattr(self.options, "axis_scale_params_x", None)
            params_y = getattr(self.options, "axis_scale_params_y", None)
            gpu_pts = np.column_stack(
                [
                    _scale_forward(layer.pts[:, 0], scale_x, params_x),
                    _scale_forward(layer.pts[:, 1], scale_y, params_y),
                ]
            ).astype(np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_pts)
        glBufferData(GL_ARRAY_BUFFER, gpu_pts.nbytes, gpu_pts, GL_STATIC_DRAW)

        # Upload Colors
        if layer.colors is not None:
            # If colors are provided as a single color, broadcast it
            if layer.colors.ndim == 1 and len(layer.colors) == 4:
                cols = np.tile(layer.colors, (len(layer.pts), 1)).astype(np.float32)
            else:
                cols = layer.colors.astype(np.float32)

            glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_col)
            glBufferData(GL_ARRAY_BUFFER, cols.nbytes, cols, GL_STATIC_DRAW)

        # Per-point size multipliers. `layer.sizes` holds absolute per-point sizes in the same
        # units as `style.point_size`; dividing by that base makes them a multiplier the shader
        # applies as `u_size * a_size` (the base cancels, so the on-screen size is absolute).
        # None -> a flat 1.0, i.e. every point takes the uniform `u_size`.
        n = len(layer.pts)
        sizes = getattr(layer, "sizes", None)
        if sizes is not None and len(sizes) == n:
            base = float(layer.style.point_size) or 1.0
            mult = (np.asarray(sizes, dtype=np.float32) / base).astype(np.float32)
        else:
            mult = np.ones(n, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_size)
        glBufferData(GL_ARRAY_BUFFER, mult.nbytes, mult, GL_STATIC_DRAW)

        bufs.count = len(layer.pts)
        layer.dirty.gpu_dirty = False

    def _lod_stride(self, layer: ScatterLayer, ctx: RenderContext) -> int:
        """Power-of-two decimation stride for a point cloud at the current zoom, or 1.

        A cloud zoomed far out packs all N points into a few pixels; with alpha blending that
        is heavy overdraw for a picture the extra points do not change. Keeps roughly a few
        points per pixel of the cloud's on-screen bounding box, snapped to a power of two so
        the stride (and its index-buffer rebuild) changes only at octave boundaries.

        A capture into the interaction cache owes a second, independent decimation on top of
        that one -- see :meth:`_capture_scale_stride`.
        """
        pts = layer.pts
        n = 0 if pts is None else len(pts)
        if n < 4096 or not getattr(self.options, "lod_enabled", True):
            return 1  # small clouds are never worth decimating
        xmin, ymin = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
        xmax, ymax = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
        l, r, b, t = ctx.window_world
        sx = ctx.fb_width / max(r - l, 1e-12)
        sy = ctx.fb_height / max(t - b, 1e-12)
        screen_w = (xmax - xmin) * sx
        screen_h = (ymax - ymin) * sy
        # Cannot usefully draw more points than the on-screen area (in px), capped at the
        # viewport; a few per pixel keeps a dense look under alpha.
        area_px = min(max(screen_w, 1.0) * max(screen_h, 1.0), float(ctx.fb_width * ctx.fb_height))
        budget = max(1, int(getattr(self.options, "scatter_budget_per_px", 4)))
        max_pts = max(1024, int(area_px * budget))
        stride = 1
        if n > max_pts:
            k = n // max_pts
            while stride < k:
                stride *= 2
        return max(stride, self._capture_scale_stride(layer, ctx, n, area_px))

    def _capture_scale_stride(
        self, layer: ScatterLayer, ctx: RenderContext, n: int, area_px: float
    ) -> int:
        """Extra decimation owed to rendering into the interaction cache; 1 outside it.

        The cache is captured through a ``ctx.capture_scale`` times wider world window than
        the view it stands in for, and the impostor magnifies it back by exactly that factor.
        A point sprite keeps its size in *pixels* at any zoom, so without this the capture
        packs the same markers into ``1/capture_scale**2`` of the area: the cached picture is
        that much more overdrawn than the exact view, and at the default 3x padding the plot
        visibly *darkened* the moment you grabbed it (measured on ``ex_scatter.py``: 3.3x the
        ink of the exact frame it replaces). Dropping ``capture_scale**2`` of the points
        restores the on-screen density -- and makes the capture that many times cheaper,
        which is most of the reason the impostor exists at all.

        The cap is what keeps this honest on a cloud that is *not* overdrawn. Decimation may
        only remove markers that were painting over other markers; once the survivors no
        longer cover the cached image, dropping more would delete points the user can see
        individually, and they would blink out for the length of the drag. So the stride
        never exceeds the number of points it takes to cover that image once over.
        """
        scale = float(getattr(ctx, "capture_scale", 1.0))
        if scale <= 1.0:
            return 1
        marker_px = max(float(layer.style.point_size) * ctx.dpr, 1.0)
        marker_area = 0.25 * np.pi * marker_px * marker_px
        coverage_limit = int((n * marker_area) / max(area_px, 1.0))
        return max(1, min(int(round(scale * scale)), coverage_limit))

    def _ensure_lod(self, layer: ScatterLayer, bufs: GLScatterBuffers, ctx: RenderContext) -> int:
        """Update the LOD index buffer for the current zoom; return the active stride.

        Keeps the full vertex buffers intact (so per-point picking/selection indices stay
        valid) and draws a strided subset through an index buffer. Rebuilt only when the
        stride changes.
        """
        stride = self._lod_stride(layer, ctx)
        if stride <= 1:
            return 1
        if bufs.lod_stride == stride and bufs.vbo_lod and bufs.lod_count > 0:
            return stride
        idx = np.arange(0, bufs.count, stride, dtype=np.uint32)
        if not bufs.vbo_lod:
            bufs.vbo_lod = int(glGenBuffers(1))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufs.vbo_lod)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        bufs.lod_stride = stride
        bufs.lod_count = int(idx.size)
        return stride

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

        # Set unconditionally, like the outline uniforms below: a shader program is shared
        # across every layer drawn in a frame, so a layer that names no marker must reset
        # this to the circle rather than inherit the previous layer's shape.
        glUniform1i(self.u_marker_shape, marker_shape_index(layer.metadata.get("marker")))

        # The per-marker ring, inside the sprite (style.point_outline_*).
        glUniform1i(self.u_point_outline_enabled, 1 if layer.style.point_outline_enabled else 0)
        if layer.style.point_outline_enabled:
            glUniform4f(self.u_point_outline_color, *layer.style.point_outline_color)
            glUniform1f(
                self.u_point_outline_width_px, float(layer.style.point_outline_width) * ctx.dpr
            )

        # The general silhouette, outside the marker (style.outline_*). Costs no draw call.
        self._push_outline_uniforms(layer, ctx)

        glUniform2f(self.u_offset, *layer.translation)

        # 3. Draw call. Zoom-out LOD: draw a strided subset through an index buffer when the
        # cloud is oversampled for its on-screen size; full draw otherwise.
        glBindVertexArray(layer._gl.vao)
        stride = self._ensure_lod(layer, layer._gl, ctx)
        if stride > 1 and layer._gl.vbo_lod and layer._gl.lod_count > 0:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, layer._gl.vbo_lod)
            glDrawElements(GL_POINTS, layer._gl.lod_count, GL_UNSIGNED_INT, None)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        else:
            glDrawArrays(GL_POINTS, 0, layer._gl.count)

        # 4. Selection highlight — re-draws ONLY the selected points, on top.
        self._draw_selection_highlight(layer, ctx, effective_size)

        # Cleanup
        glBindVertexArray(0)
        glUseProgram(0)
        glDisable(GL_PROGRAM_POINT_SIZE)

    # ------------------------------------------------------------------
    # Outline / silhouette
    # ------------------------------------------------------------------

    def _push_outline_uniforms(self, layer: ScatterLayer, ctx: RenderContext) -> float:
        """Upload the general-outline uniforms; return the ring width in framebuffer px.

        0.0 means "no outline", and is what an untouched layer returns -- ``SCATTER_VS``
        then adds a literal 0.0 to ``gl_PointSize`` and pins ``v_fill_frac`` to 1.0, so the
        picture is bit-for-bit the pre-outline one. Pushed unconditionally so a layer never
        inherits the previous layer's outline, exactly as ``Geometry3DRenderer`` does.
        """
        style = layer.style
        enabled = bool(getattr(style, "outline_enabled", False))
        # Logical px -> framebuffer px, the same conversion point_size gets, so the ring
        # keeps its apparent width on a Retina display and in a scaled savefig.
        width = float(getattr(style, "outline_width", 0.0)) * ctx.dpr
        if not enabled or not np.isfinite(width) or width <= 0.0:
            glUniform1i(self.u_outline_enabled, 0)
            glUniform1f(self.u_outline_width, 0.0)
            return 0.0

        color = tuple(float(v) for v in getattr(style, "outline_color", (0.0, 0.0, 0.0, 1.0)))
        glUniform1i(self.u_outline_enabled, 1)
        glUniform1f(self.u_outline_width, width)
        glUniform4f(self.u_outline_color, *color)
        # The layer's own alpha is deliberately not folded in (see LayerStyle.outline_alpha);
        # the scene-wide fade is, so a globally dimmed scene dims its outlines too.
        glUniform1f(
            self.u_outline_alpha,
            float(getattr(style, "outline_alpha", 1.0)) * float(ctx.global_alpha),
        )
        return width

    # ------------------------------------------------------------------
    # Selection highlight
    # ------------------------------------------------------------------

    def _sync_selection_buffer(self, layer: ScatterLayer, bufs: GLScatterBuffers) -> int:
        """Upload ``layer._selected_indices`` as an element buffer; return the draw count.

        The highlight is an index buffer over the layer's existing position/colour
        VBOs, so it costs O(selected) -- not O(points) -- and needs no per-point mask
        attribute and no second copy of the geometry.

        The buffer is named ``vbo_sel`` on purpose. ``gui.layerops.release_gl`` frees
        every handle whose name starts with ``vbo`` (``layerops._BUFFER_PREFIXES``), so
        this name gets the §1.6 GPU teardown for free; an ``ebo_sel`` would match
        nothing and leak on every layer removal. An element buffer is an ordinary
        buffer object, so ``glDeleteBuffers`` is the correct deleter either way.
        """
        raw = getattr(layer, "_selected_indices", None)
        if raw is None:
            return 0

        version = int(getattr(layer, "_selection_version", -1))
        cached = int(getattr(bufs, "sel_version", -2))
        if version >= 0 and cached == version:
            return int(getattr(bufs, "sel_count", 0))

        idx = np.asarray(raw).ravel()
        # A selection can outlive the array it indexes (delete rows, then re-select):
        # an out-of-range element index reads past the VBO, which is undefined
        # behaviour, not an exception. Clamp rather than trust.
        idx = idx[(idx >= 0) & (idx < int(bufs.count))].astype(np.uint32, copy=False)

        if idx.size == 0:
            bufs.sel_version = version
            bufs.sel_count = 0
            return 0

        ebo = int(getattr(bufs, "vbo_sel", 0))
        if not ebo:
            ebo = glGenBuffers(1)
            bufs.vbo_sel = int(ebo)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        bufs.sel_version = version
        bufs.sel_count = int(idx.size)
        return int(idx.size)

    def _draw_selection_highlight(
        self, layer: ScatterLayer, ctx: RenderContext, base_size: float
    ) -> None:
        """Re-draw the selected points larger, with an accent ring.

        Called from :meth:`draw` with ``self.prog`` bound and the layer's VAO bound.
        Every uniform it touches is re-pushed unconditionally at the top of the next
        ``draw``, so it does not restore them.

        **Mid-drag behaviour (CONTRACT §1.2's impostor path).** This draws inside the
        EXACT pass, and ``engine._capture_interaction_cache`` builds the interaction
        impostor by calling the very same ``renderer_manager.draw_exact``. The
        highlight is therefore baked into the impostor texture and pans/zooms with the
        data instead of blinking out -- it is exactly as fresh as the rest of the
        cached scene, and no more stale. A selection *changed* through the GUI queue
        rebuilds the impostor anyway, because the §1.3 drain epilogue clears
        ``cache.capture_window`` and sets ``cache.refresh_requested``.
        """
        count = self._sync_selection_buffer(layer, layer._gl)
        if count <= 0:
            return

        overrides = self.options.visual.overrides
        alpha = ctx.global_alpha * layer.style.alpha * overrides.alpha_multiplier
        size = base_size * SELECTION_SIZE_BOOST

        glUniform1f(self.u_size, float(size))
        glUniform1f(self.u_point_size_px, float(size))
        glUniform1f(self.u_alpha, float(max(alpha, SELECTION_MIN_ALPHA)))
        glUniform1i(self.u_point_outline_enabled, 1)
        glUniform4f(self.u_point_outline_color, *SELECTION_OUTLINE_COLOR)
        glUniform1f(self.u_point_outline_width_px, SELECTION_OUTLINE_WIDTH_PX * ctx.dpr)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, int(layer._gl.vbo_sel))
        glDrawElements(GL_POINTS, count, GL_UNSIGNED_INT, None)
        # Unbind while the VAO is still bound: the element binding is VAO state, and
        # the normal path draws this VAO with glDrawArrays. Leaving it bound would
        # hand a stale EBO to any future indexed draw on this VAO.
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    # ------------------------------------------------------------------
    # Textured-quad path for imshow layers
    # ------------------------------------------------------------------

    def _make_image_texture(self, layer: ScatterLayer) -> int:
        """Upload the imshow matrix as a GL_RGBA texture and return the texture ID."""
        # Deferred: pyplot imports this module (via the renderer manager), so importing
        # pyplot at module scope here would be circular. Same idiom as
        # renderers/axis.py's `from .legend import _add_scaled_text`.
        from ..pyplot import _colormap_values

        meta = layer.metadata
        matrix = np.asarray(meta["matrix"], dtype=float)
        norm = meta.get("norm")
        vmin = meta.get("vmin")
        vmax = meta.get("vmax")
        # A real Normalize (or scale name) replaces vmin/vmax outright -- same contract
        # `_normalize_cvalues` enforces for scatter/patch coloring. Falling back to a
        # hardcoded linear ramp here (as this used to) silently ignored `norm=` for
        # every imshow()/contourf() pixel actually painted live.
        rgba_flat = _colormap_values(
            matrix.ravel(),
            cmap=meta.get("cmap", "viridis"),
            vmin=None if norm is not None else vmin,
            vmax=None if norm is not None else vmax,
            norm=norm,
        )
        rgba = np.asarray(rgba_flat, dtype=np.float32).reshape(matrix.shape + (4,))
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
        """Create VAO/VBO for a textured quad covering the imshow extent.

        `layer.metadata["extent"]` itself is never touched -- same invariant as every
        other renderer's upload hook -- so a real axis scale transforms a throwaway
        copy of the 4 corners here. `imshow()`/`contourf()`/`matshow()`/`specgram()`/
        `figimage()` (and the GUI's own "add image layer" command) all route through
        this one method, so fixing it here covers all of them.
        """
        xmin, xmax, ymin, ymax = layer.metadata["extent"]
        scale_x = getattr(self.options, "axis_scale_x", "linear")
        scale_y = getattr(self.options, "axis_scale_y", "linear")
        if scale_x != "linear" or scale_y != "linear":
            params_x = getattr(self.options, "axis_scale_params_x", None)
            params_y = getattr(self.options, "axis_scale_params_y", None)
            xs = _clamp_to_domain(np.array([xmin, xmax]), scale_x, params_x)
            ys = _clamp_to_domain(np.array([ymin, ymax]), scale_y, params_y)
            xmin, xmax = _scale_forward(xs, scale_x, params_x)
            ymin, ymax = _scale_forward(ys, scale_y, params_y)
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

        # 3. Draw call, with the same zoom-out LOD as the exact pass (accumulating 1M+ points
        # every frame is what makes density heavy when zoomed out).
        glBindVertexArray(layer._gl.vao)
        stride = self._ensure_lod(layer, layer._gl, ctx)
        if stride > 1 and layer._gl.vbo_lod and layer._gl.lod_count > 0:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, layer._gl.vbo_lod)
            glDrawElements(GL_POINTS, layer._gl.lod_count, GL_UNSIGNED_INT, None)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        else:
            glDrawArrays(GL_POINTS, 0, layer._gl.count)

        # Cleanup
        glBindVertexArray(0)
        glUseProgram(0)
        glDisable(GL_PROGRAM_POINT_SIZE)
