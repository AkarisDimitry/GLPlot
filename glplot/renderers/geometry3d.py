from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from OpenGL.GL import *

from ..core.camera3d import Camera3D, bounds_centre_radius
from ..core.camera3d import look_at as _look_at_impl
from ..core.camera3d import normalize as _normalize_impl
from ..core.camera3d import perspective as _perspective_impl
from ..utils.blending import apply_blend_mode, hides_nothing, is_order_independent
from ..utils.gl_utils import link_program
from ..utils.shaders import GEOMETRY3D_FS, GEOMETRY3D_VS

if TYPE_CHECKING:
    from ..core.context import RenderContext
    from ..core.layers import Layer3D
    from ..options import EngineOptions


#: Alpha at or above which a layer is treated as opaque. One 8-bit colour step is 1/255, so
#: anything nearer to 1 than this cannot change a pixel and must keep writing depth.
OPAQUE_ALPHA_EPS = 1.0 / 512.0

#: Depth buckets a translucent point cloud is ordered into before it is drawn.
#:
#: Quantising to ``uint16`` is what makes the sort affordable at all: numpy's *stable* sort
#: takes a radix path on 2-byte integers, which puts 1.75M points in order in ~14 ms against
#: ~140 ms for an ``argsort`` of the float depths -- a tenfold difference, and the difference
#: between re-sorting on a camera nudge and stalling the drag. 65k buckets across the scene's
#: depth range is far finer than any ordering error the eye can find among overlapping
#: sprites, and ties inside a bucket keep their array order (the sort is stable).
SORT_BUCKETS = 65535

#: How far the view may rotate before a cached point order is thrown away, as a cosine
#: between the old and new depth axes. ~1.5 degrees: tight enough that no visibly wrong
#: ordering survives, loose enough that a slow drag re-sorts a handful of times instead of
#: once per frame.
SORT_DIRECTION_EPS = 0.9997

#: Largest cloud that may be re-sorted *during a drag*. Above it the last order is kept
#: until the camera settles and the frame goes back to :attr:`RenderMode.EXACT`.
#:
#: Sorting costs ~6 ms at 400k points and ~20 ms at 1.75M, which is fine once and ruinous
#: on every frame of a rotation -- 20 ms of CPU is most of a 60 Hz budget spent before the
#: GPU is asked for anything. The stale order is wrong in exactly the way the sort exists to
#: fix, so this is the usual bargain the INTERACTIVE mode already makes everywhere else
#: (dropped points, cached impostors): approximate while the mouse is down, exact the
#: instant it comes up.
SORT_INTERACTIVE_MAX_POINTS = 750_000


@dataclass
class GLGeometry3DBuffers:
    vao: int = 0
    vbo_pos: int = 0
    vbo_col: int = 0
    vbo_size: int = 0  # per-point size multipliers (attribute 2); only for point primitives
    ebo: int = 0
    count: int = 0
    #: Index buffer holding the back-to-front point order, and how many indices are in it.
    #: Only translucent point clouds allocate one; see :func:`depth_order`.
    ebo_order: int = 0
    order_count: int = 0


# The matrix primitives now live in ``core.camera3d`` so the camera can build a view
# without importing a GL module. These aliases keep the historical private names working —
# they are part of this module's tested surface (``tests/test_geometry3d.py`` imports all
# three) and re-exporting is cheaper than a rename that buys nothing.
_normalize = _normalize_impl
_perspective = _perspective_impl
_look_at = _look_at_impl


#: Directions sampled on the circle of radius ``outline_width`` for the silhouette pass,
#: as ``round(OUTLINE_STEPS_PER_PX * width)`` clamped to the two bounds below.
#:
#: The sampling only has to be dense enough that neighbouring copies of a *one-pixel* line
#: overlap: their perpendicular spacing is ``width * 2*pi/steps`` at the shallow angles, so
#: 6 steps per pixel keeps that under one pixel up to a ~4 px outline. The cap exists
#: because each step is another full draw of the layer, and 24 draws of a million-triangle
#: mesh is already a frame-rate decision the user should be making deliberately.
OUTLINE_STEPS_PER_PX = 6.0
OUTLINE_MIN_STEPS = 8
OUTLINE_MAX_STEPS = 24

#: Steps used for a *solid* primitive (triangles). Each copy covers the whole interior, so
#: the union is a regular polygon inscribed in the dilated shape -- no gaps at any step
#: count, and 8 already puts the thinnest part of the rim at ``cos(pi/8) = 0.92`` of the
#: requested width. Only thin geometry needs the dense sampling above.
OUTLINE_SOLID_STEPS = 8

#: NDC depth pushed onto every silhouette copy. Large enough to beat depth quantisation and
#: the depth slope across a few pixels of a steep surface, small enough that the outline
#: does not sink behind unrelated geometry that is genuinely in front of it. In window
#: depth this is 5e-4, i.e. ~8000 units of a 24-bit depth buffer.
OUTLINE_DEPTH_BIAS = 1e-3


def _outline_offsets(width_px: float, *, solid: bool) -> np.ndarray:
    """Screen-space displacements, in px, whose union dilates a silhouette by ``width_px``.

    An outline *is* the layer's screen footprint grown by ``width_px`` in every direction --
    a morphological dilation by a disc. Redrawing the geometry once per direction on a
    circle of that radius and taking the union is exactly that dilation, sampled at
    :data:`OUTLINE_SOLID_STEPS` or :data:`OUTLINE_STEPS_PER_PX` directions.

    Returns an ``(steps, 2)`` float32 array; empty when the width rounds away to nothing,
    which is the renderer's signal to skip the pass entirely.
    """
    if not np.isfinite(width_px) or width_px <= 0.0:
        return np.zeros((0, 2), dtype=np.float32)
    steps = (
        OUTLINE_SOLID_STEPS
        if solid
        else int(
            np.clip(round(OUTLINE_STEPS_PER_PX * width_px), OUTLINE_MIN_STEPS, OUTLINE_MAX_STEPS)
        )
    )
    ang = np.linspace(0.0, 2.0 * np.pi, steps, endpoint=False)
    return (np.stack([np.cos(ang), np.sin(ang)], axis=1) * float(width_px)).astype(np.float32)


def _layer_bounds(layer: Layer3D):
    """The box this layer is framed against: the shared scene box, or its own extent."""
    return layer.metadata.get("scene_bounds") or layer.get_bounds_3d()


def _bounds_3d(layer: Layer3D) -> tuple[np.ndarray, float]:
    """``(centre, radius)`` of the box this layer is framed against."""
    return bounds_centre_radius(_layer_bounds(layer))


def layer_min_alpha(layer: Layer3D) -> float:
    """The smallest per-vertex alpha in ``layer.colors``; 1.0 when it carries no alpha.

    Memoised on the same ``(id, shape, dirty)`` key as :meth:`Layer3D.get_bounds_3d`, and for
    the same reason: this is a full scan of the colour buffer and it is asked for once per
    layer per frame. Replacing the array changes ``id`` and misses the cache by itself; an
    in-place edit must raise ``dirty.gpu_dirty``, which is already the contract for one (and
    is what gets the new colours onto the GPU at all).
    """
    colors = layer.colors
    if colors is None or getattr(colors, "ndim", 0) != 2 or colors.shape[1] < 4:
        return 1.0
    key = (id(colors), colors.shape, bool(layer.dirty.gpu_dirty), bool(layer.dirty.bounds_dirty))
    if key == getattr(layer, "_min_alpha_key", None):
        return layer._min_alpha_value
    value = 1.0 if colors.shape[0] == 0 else float(np.min(colors[:, 3]))
    layer._min_alpha_key = key
    layer._min_alpha_value = value
    return value


def is_translucent(layer: Layer3D) -> bool:
    """Whether anything this layer draws can let what is behind it show through.

    Three places translucency can hide. ``style.alpha`` is the one the Style panel and
    ``alpha=`` on ``scatter3d``/``surface3d`` set; the **colour array** is where
    ``pyplot.volume3d`` puts it (``colors[:, 3] *= alpha``), so a volume with ``alpha=0.42``
    has ``style.alpha == 1.0`` and reads as fully opaque unless the colours are inspected;
    and the **blend mode**, because an accumulating one never replaces the destination at
    all -- an ADDITIVE layer at alpha 1.0 is still something you see through, so reading
    translucency off the alpha channel alone would have it write depth and punch a hole in
    everything behind it.

    ``style.auto_alpha`` implies it too: it exists to solve for an alpha below 1.

    ``ctx.global_alpha`` is deliberately not part of this. It is a scene-wide fade driven by
    the *2D* line count (:meth:`GPULinePlot._get_adaptive_alpha`), and folding it in would
    strip depth writes from an entirely opaque 3D scene the moment a busy 2D panel dimmed it
    -- turning a solid mesh into a draw-order lottery to fix a translucency it does not have.
    """
    style = layer.style
    if hides_nothing(style.blend_mode):
        return True
    if style.auto_alpha is not None:
        return True
    return float(style.alpha) * layer_min_alpha(layer) < 1.0 - OPAQUE_ALPHA_EPS


def _depth_row(mvp: np.ndarray) -> np.ndarray:
    """The row of ``mvp`` whose value increases with distance from the eye.

    Alpha compositing is an ordered operation, so "which point is further away" has to be
    answered in the same space the depth test answers it: clip space. A perspective matrix
    puts the view-axis distance straight into ``w`` (the fourth row), which is monotonic in
    depth and needs no division. An orthographic one has ``w = 1`` everywhere and carries the
    depth in ``z`` (the third row) instead. Picking the row rather than projecting properly
    costs one dot product per point instead of two and a divide, and the ordering is the same.
    """
    w_row = np.asarray(mvp, dtype=np.float32)[3]
    if abs(float(w_row[0])) + abs(float(w_row[1])) + abs(float(w_row[2])) < 1e-9:
        return np.asarray(mvp, dtype=np.float32)[2]  # orthographic: w is constant
    return w_row


def depth_order(
    layer: Layer3D, mvp: np.ndarray, *, may_resort: bool = True
) -> Optional[np.ndarray]:
    """Indices drawing ``layer``'s points **far to near** under ``mvp``, or None.

    Blending translucent points in array order is what makes a rotated cloud look wrong: a
    fragment behind another one still paints over it, so which point "wins" a pixel is
    decided by where it happens to sit in the buffer. Sorting back to front makes the
    composite the true one -- near points dominate, the cloud reads as having a front and a
    back, and rotating it no longer reshuffles which points show through.

    Returns None when there is nothing to order (no vertices, or the layer already draws
    through its own index buffer, where a second ordering would fight the first).

    Memoised on the view: re-sorting is ~6 ms on a 400k-point cloud and ~20 ms on 1.75M,
    which is affordable when the camera turns and not affordable once per frame. The cached
    order is kept until the geometry changes or the depth axis rotates past
    :data:`SORT_DIRECTION_EPS`. ``may_resort=False`` (a big cloud mid-drag) keeps whatever
    order is already cached however stale it is, and still sorts once when there is none --
    a first frame with no order at all is the artefact at its worst.
    """
    vertices = layer.vertices
    if vertices is None or len(vertices) == 0 or layer.indices is not None:
        return None

    row = _depth_row(mvp)
    axis = row[:3]
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm < 1e-12:
        return None
    axis = axis / norm

    cached = getattr(layer, "_depth_order", None)
    cached_key = getattr(layer, "_depth_order_key", None)
    # ``update_gpu_data`` clears this key, which covers the in-place edit that ``id`` cannot
    # see: it is the one place that knows the vertices were just re-uploaded.
    key = (id(vertices), vertices.shape)
    if cached is not None and cached_key == key:
        # Same geometry: the order survives until the view has turned far enough for the
        # ordering it encodes to be wrong.
        if not may_resort:
            return cached
        if float(np.dot(axis, getattr(layer, "_depth_order_axis"))) >= SORT_DIRECTION_EPS:
            return cached

    depth = np.asarray(vertices, dtype=np.float32) @ axis
    lo, hi = float(depth.min()), float(depth.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi - lo < 1e-12:
        # A flat cloud (every point at the same depth) has no order to get wrong.
        order = np.arange(len(vertices), dtype=np.uint32)
    else:
        # Far first: the largest depth must come out at index 0, so the buckets are filled
        # from the far end down. Quantising into uint16 is what buys the radix sort.
        buckets = ((hi - depth) * (SORT_BUCKETS / (hi - lo))).astype(np.uint16)
        order = np.argsort(buckets, kind="stable").astype(np.uint32)

    layer._depth_order = order
    layer._depth_order_key = key
    layer._depth_order_axis = axis
    return order


#: Points projected to estimate a cloud's screen footprint. A few thousand is plenty for the
#: percentile extent below -- the estimate feeds a slowly-varying alpha, not a measurement --
#: and it keeps the cost flat at ~0.1 ms whether the cloud has 50k points or 5M.
AUTO_ALPHA_SAMPLES = 4096

#: Percentile of the projected samples the footprint is measured across. The outer 5% at
#: each end are the sparse halo, and letting it set the area would say the cloud is spread
#: over a region most of it is not in -- which is the opposite of the reading wanted here.
AUTO_ALPHA_PERCENTILE = 5.0


def auto_alpha_for(layer: Layer3D, mvp: np.ndarray, ctx: RenderContext) -> float:
    """The per-point alpha that lands a typical covered pixel at ``style.auto_alpha``.

    A fixed alpha is only right at one zoom level. Markers keep their size in *pixels* while
    the cloud's screen footprint grows and shrinks with the view, so the number of points
    stacked on a pixel swings by orders of magnitude between "zoomed out, everything
    saturated" and "zoomed in, nothing visible" -- the complaint this exists to answer.

    Stacking ``k`` fragments of alpha ``a`` gives ``1 - (1-a)^k``, so the alpha that reaches
    a target opacity ``T`` is ``a = 1 - (1-T)^(1/k)``. All this has to supply is ``k``:

        k = N * (sprite area in px) / (cloud's screen footprint in px)

    with the footprint measured by projecting a sample of the points and taking their
    :data:`AUTO_ALPHA_PERCENTILE` extent, rather than the bounding box -- a box drawn round
    a spherical cloud is half empty, and the corners are exactly where there is nothing.
    """
    target = layer.style.auto_alpha
    vertices = layer.vertices
    if target is None or vertices is None or len(vertices) == 0:
        return -1.0
    target = float(np.clip(target, 1e-3, 1.0))

    n = len(vertices)
    step = max(1, n // AUTO_ALPHA_SAMPLES)
    sample = np.asarray(vertices[::step], dtype=np.float32)

    mvp = np.asarray(mvp, dtype=np.float32)
    clip = sample @ mvp[:3, :3].T + mvp[:3, 3]
    w = sample @ mvp[3, :3] + mvp[3, 3]
    visible = np.isfinite(w) & (np.abs(w) > 1e-9)
    if not np.any(visible):
        return -1.0
    ndc = clip[visible, :2] / w[visible, None]

    lo, hi = AUTO_ALPHA_PERCENTILE, 100.0 - AUTO_ALPHA_PERCENTILE
    x0, x1 = np.percentile(ndc[:, 0], [lo, hi])
    y0, y1 = np.percentile(ndc[:, 1], [lo, hi])
    # NDC spans 2 units across the viewport, so a span of `s` is `s/2 * pixels` wide.
    width_px = max(float(x1 - x0) * 0.5 * ctx.fb_width, 1.0)
    height_px = max(float(y1 - y0) * 0.5 * ctx.fb_height, 1.0)
    footprint = width_px * height_px

    radius = max(float(layer.style.point_size) * ctx.dpr * 0.5, 0.5)
    sprite_px = np.pi * radius * radius
    # Only the sampled fraction was measured, but every point is drawn.
    per_pixel = max(float(n) * sprite_px / footprint, 1.0)
    return float(np.clip(1.0 - (1.0 - target) ** (1.0 / per_pixel), 1.0 / 255.0, 1.0))


def _layer_camera(layer: Layer3D) -> Camera3D:
    """Rebuild the camera a layer was last synced with.

    The renderer has no access to the engine — a layer carries its view in
    ``metadata["camera"]``, and ``engine.sync_3d_camera`` keeps every layer's copy in step.
    A layer whose metadata predates the full camera (only elev/azim/fov) still rebuilds
    into a valid :class:`Camera3D`; the missing keys simply take their defaults, which are
    the values the old renderer hard-coded.
    """
    return Camera3D.from_dict(layer.metadata.get("camera", {}))


class Geometry3DRenderer:
    """GPU renderer for large 3D point clouds, meshes, wireframes, volumes, and bars.

    Compositing
    -----------
    How a 3D layer merges with what is already on screen has three levers, all on
    :class:`~glplot.core.layers.LayerStyle` and all defaulting to "the renderer decides",
    so a layer that sets none of them composites exactly as it always has:

    ``blend_mode``
        Overrides ``options.blend_mode`` for this layer alone, restoring the figure's after.
        This is the lever that makes translucency in 3D worth having: a volumetric cloud in
        ``ADDITIVE`` accumulates light instead of averaging it, so density reads as
        brightness and the tails survive at alpha 0.02, while the surfaces beside it keep
        ordinary alpha. An accumulating mode is also order-independent, so such a layer skips
        the depth sort entirely -- cheaper *and* better looking.
    ``depth_write``
        Overrides the occlusion heuristic below. Forcing it off makes a translucent surface
        see-through including its own far side; forcing it on makes a dense cloud hide what
        is behind it.
    ``auto_alpha``
        Solves for the alpha that puts a covered pixel at a target opacity under the current
        view (:func:`auto_alpha_for`), instead of holding one number that is only right at
        one zoom level.

    Occlusion, and what the default heuristic is for
    -----------------------------------------------
    Depth *writes* are what make a translucent layer opaque to itself: left on, the first
    fragment to land on a pixel claims it and everything behind is discarded before it can
    blend. Turning them off lets the layer accumulate, at the price of making draw order the
    only arbiter -- which is why points are also depth-sorted and triangles keep their
    writes. See :meth:`draw` for the per-primitive reasoning.

    Outlines
    --------
    ``style.outline_enabled`` puts a silhouette on any 3D layer. Two different techniques
    are used, because "an outline" means two different things depending on the primitive,
    and both are documented here rather than in a commit message because both have limits
    a caller can see on screen.

    **Points -- an analytic ring, one pass.** The sprite is grown by twice the outline
    width in ``GEOMETRY3D_VS`` and the fragment shader paints everything past the marker's
    own radius in the outline colour. Exact, costs one extra branch, and the marker keeps
    the size the user asked for. *Limit:* ``GL_POINT_SIZE_RANGE`` caps the sprite (64 px on
    Apple's GL, where the shader's own clamp of 512 never bites), so on a marker already
    near that cap the ring eats into the fill instead of growing outside it.

    **Lines and triangles -- a screen-space dilation, K passes.** The geometry is drawn
    normally, then redrawn :func:`_outline_offsets` times, each copy displaced by
    ``outline_width`` pixels in a different direction on a circle and pushed
    :data:`OUTLINE_DEPTH_BIAS` further away in NDC depth. The union of the copies is the
    layer's footprint dilated by a disc -- a true outline of uniform pixel width -- and the
    depth the layer wrote on its own pass rejects every copy over its own body, leaving
    only the rim. Doing it in that order (rather than laying the copies down first and
    painting over them) is what makes an outline work on a *translucent* layer, which
    cannot hide anything by painting over it but can still own the depth buffer. The
    ``axis3d`` overlay is the one exception: it draws with the depth test off, so nothing
    but draw order can arbitrate and its copies go down first.

    Rejected alternatives, and why:

    * *Back-face-expanded shell with front-face culling*, the textbook inverted hull. It
      needs vertex normals, which no :class:`~glplot.core.layers.Layer3D` carries, and it
      needs a closed mesh. GLPlot's commonest triangle layer is ``surface3d``: an open
      sheet, whose back faces do not exist from the front, so a front-face-culled shell
      would draw nothing at all for the layer that most wants an outline.
    * *Widening the line with* ``glLineWidth``. Core-profile OpenGL only guarantees a width
      of 1.0 and Apple's driver reports exactly ``[1, 1]`` for
      ``GL_ALIASED_LINE_WIDTH_RANGE``, so a casing pass would be a silent no-op on the
      platform this was written on.
    * *Expanding along the direction from the mesh centroid.* Cheap, but the outline's
      width then varies with how far each part of the shape is from the centroid, and it
      is plain wrong for anything concave.

    Limits of the dilation, honestly:

    * It outlines the layer's **outer silhouette and its holes** -- every boundary between
      "this layer drew here" and "it did not". It does **not** outline interior creases
      where the surface folds over itself; that needs a depth-discontinuity pass over a
      depth texture, and this renderer draws straight into the panel's framebuffer.
    * Each layer is dilated **independently**. Two outlined layers that overlap do not
      share one contour; the nearer one's outline is drawn under it, so its neighbour's
      geometry can cover it.
    * The copies overlap each other, so a **translucent** outline
      (``outline_alpha < 1``) can compound where they do and the rim read unevenly. The
      depth each copy writes suppresses most of it; there is no stencil buffer in this
      pipeline to suppress the rest with. Opaque outlines are unaffected.
    * The rim is separated from the body by :data:`OUTLINE_DEPTH_BIAS`, a *constant* in
      NDC. A surface seen nearly edge-on changes depth faster than that across the outline
      width, so a few interior fragments of such a surface can win the depth test and
      speckle it in the outline colour. A slope-scaled ``glPolygonOffset`` would fix it for
      triangles and does nothing for line primitives, so the constant is used for both.
    * The cost is K extra draws of the whole layer (8 for triangles, 8-24 for lines). This
      is why the feature is per-layer and off by default rather than a global toggle.
    """

    def __init__(self, options: EngineOptions) -> None:
        self.options = options
        self.prog = 0
        self.u_mvp = -1
        self.u_alpha = -1
        self.u_auto_alpha = -1
        self.u_point_size = -1
        self.u_z_range = -1
        self.u_ssao_enabled = -1
        self.u_ssao_strength = -1
        self.u_is_points = -1
        self.u_ref_w = -1
        self.u_clip_lo = -1
        self.u_clip_hi = -1
        # Outline uniforms; see the class docstring for what each technique does with them.
        self.u_outline_enabled = -1
        self.u_outline_pass = -1
        self.u_outline_color = -1
        self.u_outline_width = -1
        self.u_outline_alpha = -1
        self.u_outline_offset = -1
        self.u_outline_depth_bias = -1
        self.u_viewport = -1

    def initialize(self) -> None:
        self.prog = link_program(GEOMETRY3D_VS, GEOMETRY3D_FS)
        self.u_clip_lo = glGetUniformLocation(self.prog, "u_clip_lo")
        self.u_clip_hi = glGetUniformLocation(self.prog, "u_clip_hi")
        self.u_mvp = glGetUniformLocation(self.prog, "u_mvp")
        self.u_alpha = glGetUniformLocation(self.prog, "u_alpha")
        self.u_auto_alpha = glGetUniformLocation(self.prog, "u_auto_alpha")
        self.u_point_size = glGetUniformLocation(self.prog, "u_point_size")
        self.u_z_range = glGetUniformLocation(self.prog, "u_z_range")
        self.u_ssao_enabled = glGetUniformLocation(self.prog, "u_ssao_enabled")
        self.u_ssao_strength = glGetUniformLocation(self.prog, "u_ssao_strength")
        self.u_is_points = glGetUniformLocation(self.prog, "u_is_points")
        self.u_ref_w = glGetUniformLocation(self.prog, "u_ref_w")
        self.u_outline_enabled = glGetUniformLocation(self.prog, "u_outline_enabled")
        self.u_outline_pass = glGetUniformLocation(self.prog, "u_outline_pass")
        self.u_outline_color = glGetUniformLocation(self.prog, "u_outline_color")
        self.u_outline_width = glGetUniformLocation(self.prog, "u_outline_width")
        self.u_outline_alpha = glGetUniformLocation(self.prog, "u_outline_alpha")
        self.u_outline_offset = glGetUniformLocation(self.prog, "u_outline_offset")
        self.u_outline_depth_bias = glGetUniformLocation(self.prog, "u_outline_depth_bias")
        self.u_viewport = glGetUniformLocation(self.prog, "u_viewport")

    def _create_buffers(self, layer: Layer3D) -> GLGeometry3DBuffers:
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo_pos = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))

        vbo_col = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_col)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))

        # Per-point size multiplier (attribute 2). Only point clouds carry it; for line/
        # triangle layers the attribute stays disabled and gl_PointSize is ignored anyway, so
        # they pay no extra buffer. The shader multiplies u_point_size by a_size (1.0 = no-op).
        vbo_size = 0
        if layer.primitive == "points":
            vbo_size = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_size)
            glBufferData(GL_ARRAY_BUFFER, 4, None, GL_STATIC_DRAW)  # pre-allocate
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))

        ebo = 0
        if layer.indices is not None:
            ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)

        glBindVertexArray(0)
        return GLGeometry3DBuffers(
            vao=vao, vbo_pos=vbo_pos, vbo_col=vbo_col, vbo_size=vbo_size, ebo=ebo
        )

    def update_gpu_data(self, layer: Layer3D, bufs: GLGeometry3DBuffers) -> None:
        if layer.vertices is None or len(layer.vertices) == 0:
            bufs.count = 0
            layer.dirty.gpu_dirty = False
            return

        vertices = np.ascontiguousarray(layer.vertices, dtype=np.float32)
        colors = layer.colors
        if colors is None:
            base = layer.style.color or (0.1, 0.45, 1.0, 1.0)
            colors = np.tile(np.asarray(base, dtype=np.float32), (len(vertices), 1))
        colors = np.ascontiguousarray(colors, dtype=np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_col)
        glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)

        # Per-point size multipliers for point clouds (see _create_buffers). Absolute
        # per-point sizes divided by the base cancel to on-screen absolute sizes; None -> 1.0.
        if layer.primitive == "points" and bufs.vbo_size:
            n = len(vertices)
            sizes = getattr(layer, "sizes", None)
            if sizes is not None and len(sizes) == n:
                base = float(layer.style.point_size) or 1.0
                mult = (np.asarray(sizes, dtype=np.float32) / base).astype(np.float32)
            else:
                mult = np.ones(n, dtype=np.float32)
            glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_size)
            glBufferData(GL_ARRAY_BUFFER, mult.nbytes, mult, GL_STATIC_DRAW)

        if layer.indices is not None and bufs.ebo:
            indices = np.ascontiguousarray(layer.indices, dtype=np.uint32)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufs.ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
            bufs.count = len(indices)
        else:
            bufs.count = len(vertices)
        # A cloud's back-to-front order describes the vertices that were just replaced, and
        # an in-place edit keeps the array's ``id``, so this is the only honest place to
        # invalidate it. See :func:`depth_order`.
        layer._depth_order_key = None
        layer.dirty.gpu_dirty = False

    def _mvp(self, layer: Layer3D, ctx: RenderContext) -> np.ndarray:
        """The model-view-projection matrix for ``layer`` under its stored camera.

        One line of real work now: :meth:`Camera3D.mvp` owns the maths, which is what lets
        pan, roll, orthographic projection and box-aspect exist at all — each of them
        changes this matrix and none of them could be expressed in the three floats the
        renderer used to inline.

        The pre-camera fallbacks survive in :meth:`Camera3D.from_dict` and
        :meth:`Camera3D.resolve_distance`: an ``elev``/``azim``-only metadata dict with no
        distance still resolves to ``2.9 * radius``, the literal this method used to carry.
        """
        camera = _layer_camera(layer)
        # A layer written before ``metadata["elev"]`` moved into the camera dict.
        if "elev" not in layer.metadata.get("camera", {}) and "elev" in layer.metadata:
            camera.elev = float(layer.metadata["elev"])
        if "azim" not in layer.metadata.get("camera", {}) and "azim" in layer.metadata:
            camera.azim = float(layer.metadata["azim"])
        return camera.mvp(ctx.aspect, _layer_bounds(layer))

    def draw(self, layer: Layer3D, ctx: RenderContext) -> None:
        if layer.vertices is None or len(layer.vertices) == 0:
            return
        if not hasattr(layer, "_gl") or layer._gl is None:
            layer._gl = self._create_buffers(layer)
            layer.dirty.gpu_dirty = True
        if layer.dirty.gpu_dirty:
            self.update_gpu_data(layer, layer._gl)

        mode_map = {
            "points": GL_POINTS,
            "lines": GL_LINES,
            "triangles": GL_TRIANGLES,
        }
        mode = mode_map.get(layer.primitive, GL_POINTS)

        # axis3d wireframe is drawn last and must always be visible regardless of
        # what volumetric data lies in front of it — skip depth testing for it.
        is_axis_overlay = layer.metadata.get("artist") == "axis3d"
        if not is_axis_overlay:
            glEnable(GL_DEPTH_TEST)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glUseProgram(self.prog)
        mvp = self._mvp(layer, ctx)
        glUniformMatrix4fv(self.u_mvp, 1, GL_TRUE, mvp)
        glUniform1f(self.u_alpha, float(ctx.global_alpha * layer.style.alpha))
        # Negative unless the layer asked for automatic alpha, which is the shader's "use
        # the layer's own alpha" sentinel -- so this is pushed unconditionally and no draw
        # can inherit the previous layer's.
        glUniform1f(self.u_auto_alpha, auto_alpha_for(layer, mvp, ctx))
        glUniform1f(self.u_point_size, float(layer.style.point_size * ctx.dpr))
        b = layer.get_bounds_3d()
        zmin, zmax = (0.0, 1.0) if b is None else (float(b[4]), float(b[5]))
        glUniform2f(self.u_z_range, zmin, zmax)
        ssao = getattr(self.options.visual, "ssao", None)
        enabled = bool(getattr(ssao, "enabled", False) or layer.metadata.get("ssao", False))
        strength = float(layer.metadata.get("ssao_strength", getattr(ssao, "strength", 0.45)))
        glUniform1i(self.u_ssao_enabled, 1 if enabled else 0)
        glUniform1f(self.u_ssao_strength, strength)
        glUniform1i(self.u_is_points, 1 if mode == GL_POINTS else 0)

        # Perspective-correct point sizing: pass camera distance as the reference
        # clip-space w so that points closer than the scene centre look proportionally
        # larger and farther points look smaller — matching natural perspective depth cues.
        # Only for point primitives; lines/triangles keep a fixed screen-space width.
        # Orthographic projection has no foreshortening by construction, so a distance-based
        # size reference would be a lie: every point must draw at the same size regardless
        # of depth. ref_w = 0 is the shader's "disable perspective sizing" sentinel.
        camera_meta = layer.metadata.get("camera", {})
        if (
            mode == GL_POINTS
            and str(camera_meta.get("projection", "perspective")) != "orthographic"
        ):
            _, radius = _bounds_3d(layer)
            ref_w = float(camera_meta.get("distance", radius * 3.2))
        else:
            ref_w = 0.0
        glUniform1f(self.u_ref_w, ref_w)

        # Explicit axis limits, when the user set any. A layer with none gets an inverted
        # box (lo = 1, hi = 0), which the shader reads as "this axis is unbounded" — so
        # the common case never discards and needs no branch on the CPU side either.
        clip = layer.metadata.get("clip_bounds")
        if clip is None or len(clip) != 6:
            glUniform3f(self.u_clip_lo, 1.0, 1.0, 1.0)
            glUniform3f(self.u_clip_hi, 0.0, 0.0, 0.0)
        else:
            glUniform3f(self.u_clip_lo, float(clip[0]), float(clip[2]), float(clip[4]))
            glUniform3f(self.u_clip_hi, float(clip[1]), float(clip[3]), float(clip[5]))

        outline_width = self._push_outline_uniforms(layer, ctx)

        glBindVertexArray(layer._gl.vao)
        # Where the silhouette goes in the draw order is decided by the depth test, and both
        # orders answer the same requirement: the real geometry must win wherever the two
        # overlap, or the outline colour floods the shape instead of edging it. Points never
        # take this path at all -- their ring is analytic, in the uniforms pushed above.
        #
        #  * Depth test on (everything but the axis overlay): geometry first, so it owns the
        #    depth buffer, then the copies pushed OUTLINE_DEPTH_BIAS further away. They fail
        #    GL_LESS over every pixel the layer already drew and survive only on the dilated
        #    rim. Drawing them first instead would look identical on an opaque mesh and be
        #    unusable on a translucent one, which cannot hide them by painting over them --
        #    a 30%-alpha floor came out solid outline colour.
        #  * Depth test off (the axis3d overlay, which must stay visible through the data):
        #    there is no depth to lose, so draw order is the only arbiter -- copies first.
        outlined = outline_width > 0.0 and mode != GL_POINTS

        # Depth *writes* are what make a translucent layer opaque to itself. The depth test
        # keeps its ordering job either way; the mask decides whether a fragment that blends
        # also claims the pixel. Left on, the first fragment to land on a pixel wins it and
        # every fragment behind it is discarded before it can blend -- a 1.7M-point volume at
        # alpha=0.42 renders as a single 0.42 shell over the background, no accumulation
        # anywhere, which is the "blending does nothing in 3D" symptom. Off, the cloud
        # accumulates and still sits correctly behind opaque geometry that already wrote depth.
        #
        # Which primitives get the mask is decided by what *unsorted* blending would look
        # like, since dropping the mask lets every fragment through in draw order:
        #
        #  * Points -- yes, and they are sorted below, so the order is the real one. A point
        #    contributes a sprite's worth of colour and the whole purpose of a translucent
        #    cloud is that those contributions add up.
        #  * Lines -- yes, unsorted. A line is thin enough that which of two crossing
        #    segments blends first is not a picture anyone can read wrongly, and this is what
        #    the 3D box furniture (grid, ticks, floor) is made of.
        #  * Triangles -- no. A mesh fragment covers the pixel, so "unsorted" degrades to
        #    "whichever triangle is last in the buffer wins", which is worse than the nearest
        #    surface the depth test picks. Measured on ``plot_surface(..., alpha=0.92)``: the
        #    red peak vanished behind the far side of the trough, which was drawn later.
        #    A surface therefore still reads as a surface, and its transparency is against
        #    what is *behind* it -- which is the half `RendererManager._order_by_opacity`
        #    fixes, by putting every opaque layer down first.
        #
        # Outlined layers keep their depth writes too: the dilation pass relies on the
        # geometry having written depth to reject the copies over its own body (see above), so
        # a translucent outlined layer stays self-occluding rather than flooding with outline
        # colour. Points never take that path, so no point cloud is affected.
        #
        # ``style.depth_write`` overrides all of it. The heuristic is a default, not a
        # policy: a caller who wants to see through their own surface, or wants a dense cloud
        # to occlude, knows something about their data that no rule here can.
        depth_masked = (
            not is_axis_overlay and not outlined and mode != GL_TRIANGLES and is_translucent(layer)
        )
        if layer.style.depth_write is not None and not is_axis_overlay:
            depth_masked = not layer.style.depth_write
        if depth_masked:
            glDepthMask(GL_FALSE)

        # This layer's own blend mode, for as long as it is drawing. Restored below, so the
        # figure's mode governs everything that did not ask for something else.
        layer_blend = layer.style.blend_mode
        if layer_blend is not None:
            apply_blend_mode(layer_blend)

        # Back-to-front ordering for the one case where draw order is now the *only* arbiter
        # and the artefact is plainly visible: a translucent cloud, where an unsorted composite
        # lets points behind paint over points in front and re-shuffles which ones show through
        # every time the camera turns. See :func:`depth_order` for what it costs and when it
        # re-runs.
        #
        # An accumulating blend mode is exempt: addition and screen give the same image
        # whatever order the fragments arrive in, so sorting one would be milliseconds spent
        # to produce identical pixels.
        sorted_count = 0
        if depth_masked and mode == GL_POINTS and not is_order_independent(layer_blend):
            sorted_count = self._upload_depth_order(layer, mvp, mode=ctx.mode)

        if outlined and is_axis_overlay:
            self._draw_outline_pass(layer, mode, outline_width, solid=(mode == GL_TRIANGLES))
            self._issue_draw(layer, mode, sorted_count=sorted_count)
        else:
            self._issue_draw(layer, mode, sorted_count=sorted_count)
            if outlined:
                self._draw_outline_pass(layer, mode, outline_width, solid=(mode == GL_TRIANGLES))
        glBindVertexArray(0)
        glUseProgram(0)
        glDisable(GL_PROGRAM_POINT_SIZE)
        glDisable(GL_DEPTH_TEST)
        if depth_masked:
            # Unconditionally back to the default: the next layer, the effects manager's depth
            # clear (glClear obeys the mask) and the picking pass all assume writes are on.
            glDepthMask(GL_TRUE)
        if layer_blend is not None:
            # Hand the figure's mode back. Every other renderer in the pass was written when
            # blending was set once for the whole frame and still assumes it.
            apply_blend_mode(self.options.blend_mode)

    # ------------------------------------------------------------------
    # Draw helpers
    # ------------------------------------------------------------------

    def _issue_draw(self, layer: Layer3D, mode: int, *, sorted_count: int = 0) -> None:
        """One draw of the layer's geometry. Assumes its VAO and the program are bound.

        ``sorted_count`` draws through the back-to-front index buffer instead of the raw
        vertex order; 0 (the default, and every non-cloud layer) is the historical path.
        """
        if sorted_count > 0:
            glDrawElements(mode, sorted_count, GL_UNSIGNED_INT, None)
        elif layer.indices is not None and layer._gl.ebo:
            glDrawElements(mode, layer._gl.count, GL_UNSIGNED_INT, None)
        else:
            glDrawArrays(mode, 0, layer._gl.count)

    def _upload_depth_order(self, layer: Layer3D, mvp: np.ndarray, *, mode: Any = None) -> int:
        """Bind (and refresh, if the view moved) this cloud's back-to-front index buffer.

        Returns the number of indices to draw, or 0 when the layer is not orderable and the
        caller should fall back to the raw vertex order. Assumes the layer's VAO is bound --
        the element buffer binding is VAO state, which is exactly where it needs to live.
        """
        from ..options import RenderMode

        may_resort = mode != RenderMode.INTERACTIVE or (
            len(layer.vertices) <= SORT_INTERACTIVE_MAX_POINTS
        )
        order = depth_order(layer, mvp, may_resort=may_resort)
        if order is None or len(order) == 0:
            return 0

        bufs = layer._gl
        if not bufs.ebo_order:
            bufs.ebo_order = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufs.ebo_order)
        # ``_depth_order_uploaded`` is the array object the buffer currently holds. Identity
        # is the right test: :func:`depth_order` returns the *same* array while its memo is
        # valid and a new one whenever it re-sorts, so this uploads exactly on a re-sort
        # rather than pushing 7 MB of indices up the bus every frame.
        if getattr(layer, "_depth_order_uploaded", None) is not order:
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, order.nbytes, order, GL_DYNAMIC_DRAW)
            layer._depth_order_uploaded = order
        bufs.order_count = len(order)
        return bufs.order_count

    def _push_outline_uniforms(self, layer: Layer3D, ctx: RenderContext) -> float:
        """Upload the outline uniforms; return the effective width in framebuffer pixels.

        0.0 means "no outline", and is what an untouched layer returns -- every outline
        branch in the shader is then dead and the picture is bit-for-bit the pre-outline
        one. The uniforms are pushed unconditionally so no draw can inherit the previous
        layer's outline state.
        """
        style = layer.style
        enabled = bool(getattr(style, "outline_enabled", False))
        width = float(getattr(style, "outline_width", 0.0)) * ctx.dpr
        if not enabled or not np.isfinite(width) or width <= 0.0:
            glUniform1i(self.u_outline_enabled, 0)
            glUniform1i(self.u_outline_pass, 0)
            glUniform1f(self.u_outline_width, 0.0)
            glUniform2f(self.u_outline_offset, 0.0, 0.0)
            return 0.0

        color = tuple(float(v) for v in getattr(style, "outline_color", (0.0, 0.0, 0.0, 1.0)))
        glUniform1i(self.u_outline_enabled, 1)
        glUniform1i(self.u_outline_pass, 0)
        glUniform4f(self.u_outline_color, *color)
        glUniform1f(self.u_outline_width, width)
        # The layer's own alpha is deliberately not folded in (see LayerStyle.outline_alpha);
        # the scene-wide fade is, so a globally dimmed scene dims its outlines too.
        glUniform1f(
            self.u_outline_alpha,
            float(getattr(style, "outline_alpha", 1.0)) * float(ctx.global_alpha),
        )
        glUniform1f(self.u_outline_depth_bias, OUTLINE_DEPTH_BIAS)
        # Viewport, not window: ``_draw_panels`` swaps ``fb_width``/``fb_height`` to the
        # panel's rectangle, which is what the pixel-to-NDC conversion has to divide by.
        glUniform2f(self.u_viewport, float(max(ctx.fb_width, 1)), float(max(ctx.fb_height, 1)))
        glUniform2f(self.u_outline_offset, 0.0, 0.0)
        return width

    def _draw_outline_pass(
        self, layer: Layer3D, mode: int, width_px: float, *, solid: bool
    ) -> None:
        """Redraw the layer once per dilation direction, in the outline colour.

        Assumes the program and the layer's VAO are bound. Restores ``u_outline_pass`` and
        ``u_outline_offset`` afterwards, so neither this layer's own second draw (the
        axis-overlay order) nor the next layer's inherits them.
        """
        offsets = _outline_offsets(width_px, solid=solid)
        if len(offsets) == 0:
            return
        glUniform1i(self.u_outline_pass, 1)
        for dx, dy in offsets:
            glUniform2f(self.u_outline_offset, float(dx), float(dy))
            self._issue_draw(layer, mode)
        glUniform1i(self.u_outline_pass, 0)
        glUniform2f(self.u_outline_offset, 0.0, 0.0)
