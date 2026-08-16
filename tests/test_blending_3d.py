"""Alpha blending in 3D: depth *writes*, and the order translucent layers draw in.

The bug this pins down: every 3D layer used to draw with depth writes on, translucent or
not. The depth test then resolved each pixel to the single nearest fragment and discarded
the rest *before* they could blend, so a translucent cloud accumulated nothing -- a 1.7M
point ``volume3d`` at ``alpha=0.42`` came out as one 0.42-thick shell over the background,
and opaque geometry inside it was swallowed by whatever cloud fragment happened to be in
front. Measured on the real GL renderer, ten coincident points at ``alpha=0.3`` composited
to one point's worth of colour (0.30 of the way from white to red) instead of ten (0.97).

Dropping the depth writes then makes *draw order* the only arbiter inside a cloud, which is
its own visible bug -- points behind painting over points in front, re-shuffling every time
the camera turns. Measured the same way, on a layer whose near half is red and far half is
blue: unsorted the far blue half won the centre (R=0.08, B=0.93), sorted the near red half
does (R=0.91, B=0.09). So the third half of the fix is the back-to-front sort.

All three are checked here without a GL context, by swapping the entry points
``geometry3d`` star-imported for recorders (the technique ``test_outline.py`` documents):

* the **renderer** must drop depth writes for a translucent layer and restore them, and
  must leave an opaque layer's state sequence exactly as it was;
* the **sort** must order a cloud far-to-near, re-run when the view turns, and stay cached
  when it does not;
* the **manager** must draw opaque 3D layers before translucent ones, since a layer that
  no longer writes depth cannot occlude anything drawn after it.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.core.context import RenderContext
from glplot.core.layers import Layer3D, PolylineLayer
from glplot.managers.renderer_manager import RendererManager
from glplot.options import EngineOptions, RenderMode
from glplot.renderers import geometry3d as g3d

UNIFORM_NAMES = (
    "u_mvp",
    "u_alpha",
    "u_point_size",
    "u_z_range",
    "u_ssao_enabled",
    "u_ssao_strength",
    "u_is_points",
    "u_ref_w",
    "u_clip_lo",
    "u_clip_hi",
    "u_outline_enabled",
    "u_outline_pass",
    "u_outline_color",
    "u_outline_width",
    "u_outline_alpha",
    "u_outline_offset",
    "u_outline_depth_bias",
    "u_viewport",
)


class GLRecorder:
    """Records the state changes and draws ``Geometry3DRenderer.draw`` issues, in order."""

    def __init__(self) -> None:
        self.state: list = []
        self.draws: list = []
        #: Index-buffer uploads: (target, byte size). A sorted cloud must upload its order
        #: exactly when it re-sorts, so the count of these is the thing worth asserting on.
        self.uploads: list = []

    def enable(self, cap):
        self.state.append(("enable", int(cap)))

    def disable(self, cap):
        self.state.append(("disable", int(cap)))

    def depth_mask(self, flag):
        self.state.append(("depth_mask", bool(flag)))

    def draw_arrays(self, mode, first, count):
        self.state.append(("draw", int(mode)))
        self.draws.append(("arrays", int(mode), int(count)))

    def draw_elements(self, mode, count, gl_type, indices):
        self.state.append(("draw", int(mode)))
        self.draws.append(("elements", int(mode), int(count)))

    def buffer_data(self, target, size, data, usage):
        self.uploads.append((int(target), int(size)))


@pytest.fixture
def recorder(monkeypatch) -> GLRecorder:
    rec = GLRecorder()
    monkeypatch.setattr(g3d, "glEnable", rec.enable)
    monkeypatch.setattr(g3d, "glDisable", rec.disable)
    monkeypatch.setattr(g3d, "glDepthMask", rec.depth_mask)
    monkeypatch.setattr(g3d, "glUseProgram", lambda prog: None)
    monkeypatch.setattr(g3d, "glBindVertexArray", lambda vao: None)
    monkeypatch.setattr(g3d, "glUniformMatrix4fv", lambda *a: None)
    for n in (1, 2, 3, 4):
        for suffix in ("f", "i"):
            name = f"glUniform{n}{suffix}"
            if hasattr(g3d, name):
                monkeypatch.setattr(g3d, name, lambda *a: None)
    monkeypatch.setattr(g3d, "glDrawArrays", rec.draw_arrays)
    monkeypatch.setattr(g3d, "glDrawElements", rec.draw_elements)
    # The sorted-cloud path allocates and fills an index buffer.
    monkeypatch.setattr(g3d, "glGenBuffers", lambda n: 42)
    monkeypatch.setattr(g3d, "glBindBuffer", lambda target, buf: None)
    monkeypatch.setattr(g3d, "glBufferData", rec.buffer_data)
    return rec


def make_renderer() -> g3d.Geometry3DRenderer:
    renderer = g3d.Geometry3DRenderer(EngineOptions())
    renderer.prog = 1
    for i, name in enumerate(UNIFORM_NAMES):
        setattr(renderer, name, i)
    return renderer


def make_layer(primitive: str = "points", *, alpha: float | None = None) -> Layer3D:
    """A tiny 3D layer whose GPU buffers are already "uploaded".

    ``alpha`` goes into the **colour array**, which is where ``pyplot.volume3d`` puts it.
    """
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.5], [1.0, 1.0, 1.0], [0.0, 1.0, 0.25]],
        dtype=np.float32,
    )
    colors = np.tile(np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32), (len(vertices), 1))
    if alpha is not None:
        colors[:, 3] = alpha
    layer = Layer3D(vertices=vertices, colors=colors, primitive=primitive)
    layer._gl = g3d.GLGeometry3DBuffers(vao=7, vbo_pos=1, vbo_col=2, count=len(vertices))
    layer.dirty.gpu_dirty = False
    return layer


def make_ctx(**kwargs) -> RenderContext:
    params = dict(
        mvp=np.eye(4, dtype=np.float32),
        window_world=(-1.0, 1.0, -1.0, 1.0),
        width_px=400,
        height_px=300,
        fb_width=800,
        fb_height=600,
        mode=RenderMode.EXACT,
        dpr=2.0,
    )
    params.update(kwargs)
    return RenderContext(**params)


# ----------------------------------------------------------------------------------
# Reading translucency off a layer
# ----------------------------------------------------------------------------------


class TestLayerMinAlpha:
    def test_a_layer_without_colours_is_opaque(self):
        layer = Layer3D(vertices=np.zeros((3, 3), dtype=np.float32))
        assert g3d.layer_min_alpha(layer) == 1.0
        assert g3d.is_translucent(layer) is False

    def test_it_reads_the_smallest_per_vertex_alpha(self):
        layer = make_layer()
        layer.colors[2, 3] = 0.25
        assert g3d.layer_min_alpha(layer) == pytest.approx(0.25)

    def test_an_empty_colour_array_is_opaque(self):
        layer = Layer3D(
            vertices=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 4), dtype=np.float32),
        )
        assert g3d.layer_min_alpha(layer) == 1.0

    def test_rgb_colours_carry_no_alpha_to_read(self):
        layer = Layer3D(
            vertices=np.zeros((2, 3), dtype=np.float32),
            colors=np.zeros((2, 3), dtype=np.float32),
        )
        assert g3d.layer_min_alpha(layer) == 1.0

    def test_replacing_the_colour_array_invalidates_the_memo(self):
        layer = make_layer()
        assert g3d.layer_min_alpha(layer) == 1.0
        layer.colors = np.tile(np.array([0.0, 0.0, 1.0, 0.4], dtype=np.float32), (4, 1))
        assert g3d.layer_min_alpha(layer) == pytest.approx(0.4)

    def test_an_in_place_edit_invalidates_the_memo_through_the_dirty_flag(self):
        """The documented contract for editing an array in place, and the same one
        ``Layer3D.get_bounds_3d`` memoises against."""
        layer = make_layer()
        assert g3d.layer_min_alpha(layer) == 1.0
        layer.colors[:, 3] = 0.6
        layer.dirty.gpu_dirty = True
        assert g3d.layer_min_alpha(layer) == pytest.approx(0.6)


class TestIsTranslucent:
    def test_style_alpha_makes_a_layer_translucent(self):
        layer = make_layer()
        layer.style.alpha = 0.5
        assert g3d.is_translucent(layer) is True

    def test_colour_alpha_makes_a_layer_translucent(self):
        """``volume3d``'s spelling: ``colors[:, 3] *= alpha``, ``style.alpha`` left at 1."""
        layer = make_layer(alpha=0.42)
        assert layer.style.alpha == 1.0
        assert g3d.is_translucent(layer) is True

    def test_a_fully_opaque_layer_is_not_translucent(self):
        assert g3d.is_translucent(make_layer()) is False

    def test_alpha_within_a_colour_step_of_one_stays_opaque(self):
        """It cannot change a pixel, and losing depth writes over it would cost far more."""
        assert g3d.is_translucent(make_layer(alpha=1.0 - 1e-4)) is False


# ----------------------------------------------------------------------------------
# The draw path
# ----------------------------------------------------------------------------------


class TestDepthWrites:
    def test_an_opaque_layer_keeps_the_historical_state_sequence(self, recorder):
        """No depth-mask call at all: an opaque 3D layer must draw as it always did."""
        make_renderer().draw(make_layer("triangles"), make_ctx())
        assert not any(entry[0] == "depth_mask" for entry in recorder.state)

    @pytest.mark.parametrize("primitive", ["points", "lines"])
    def test_a_translucent_layer_draws_with_depth_writes_off(self, recorder, primitive):
        make_renderer().draw(make_layer(primitive, alpha=0.3), make_ctx())
        masks = [entry for entry in recorder.state if entry[0] == "depth_mask"]
        assert masks == [("depth_mask", False), ("depth_mask", True)]
        # Off *before* the draw and back on after it, or the layer occludes itself anyway
        # and the next layer inherits a framebuffer it cannot write depth into.
        kinds = [entry[0] for entry in recorder.state]
        assert kinds.index("depth_mask") < kinds.index("draw")
        assert len(kinds) - 1 - kinds[::-1].index("depth_mask") > kinds.index("draw")

    def test_the_depth_test_is_still_on_for_a_translucent_layer(self, recorder):
        """Writes go, the test stays: opaque geometry in front must still occlude."""
        make_renderer().draw(make_layer("points", alpha=0.3), make_ctx())
        assert ("enable", int(g3d.GL_DEPTH_TEST)) in recorder.state

    def test_the_axis_overlay_is_left_alone(self, recorder):
        """It draws with the depth test off, so it writes no depth to begin with."""
        layer = make_layer("lines", alpha=0.3)
        layer.metadata["artist"] = "axis3d"
        make_renderer().draw(layer, make_ctx())
        assert not any(entry[0] == "depth_mask" for entry in recorder.state)

    def test_a_translucent_mesh_keeps_its_depth_writes(self, recorder):
        """Unsorted blending of surface-covering fragments degrades to "last triangle in the
        buffer wins", which is a worse picture than the nearest surface the depth test picks:
        ``plot_surface(..., alpha=0.92)`` lost its peak behind the far side of the trough.
        A mesh's transparency is against what is *behind* it, and the manager's opaque-first
        ordering is what makes that work."""
        make_renderer().draw(make_layer("triangles", alpha=0.3), make_ctx())
        assert not any(entry[0] == "depth_mask" for entry in recorder.state)

    def test_an_outlined_translucent_line_layer_keeps_its_depth_writes(self, recorder):
        """Lines would otherwise qualify. The dilation pass rejects its copies with the depth
        the geometry wrote; without that, a translucent outlined layer floods with outline
        colour instead of being edged by it."""
        layer = make_layer("lines", alpha=0.3)
        layer.style.outline_enabled = True
        layer.style.outline_width = 2.0
        make_renderer().draw(layer, make_ctx())
        assert not any(entry[0] == "depth_mask" for entry in recorder.state)
        assert len(recorder.draws) > 1  # the outline pass did run

    def test_a_translucent_point_cloud_is_never_held_back_by_the_outline_rule(self, recorder):
        """Points get an analytic ring, not the dilation, so the carve-out cannot reach
        them -- and a point cloud is the layer that needs the accumulation most."""
        layer = make_layer("points", alpha=0.3)
        layer.style.outline_enabled = True
        layer.style.outline_width = 2.0
        make_renderer().draw(layer, make_ctx())
        assert ("depth_mask", False) in recorder.state


# ----------------------------------------------------------------------------------
# Back-to-front sorting
# ----------------------------------------------------------------------------------


def perspective_mvp(axis=(0.0, 0.0, 1.0)) -> np.ndarray:
    """An MVP whose ``w`` row measures depth along ``axis`` -- i.e. a perspective one."""
    mvp = np.eye(4, dtype=np.float32)
    mvp[3, :3] = np.asarray(axis, dtype=np.float32)
    mvp[3, 3] = 10.0
    return mvp


def cloud(depths) -> Layer3D:
    """A translucent point cloud whose z coordinates are ``depths``, in that array order."""
    verts = np.zeros((len(depths), 3), dtype=np.float32)
    verts[:, 2] = np.asarray(depths, dtype=np.float32)
    layer = Layer3D(vertices=verts, primitive="points")
    layer.colors = np.tile(np.array([1.0, 0.0, 0.0, 0.3], dtype=np.float32), (len(depths), 1))
    layer._gl = g3d.GLGeometry3DBuffers(vao=7, vbo_pos=1, vbo_col=2, count=len(depths))
    layer.dirty.gpu_dirty = False
    return layer


class TestDepthOrder:
    def test_points_come_out_far_to_near(self):
        layer = cloud([0.0, 5.0, -3.0, 2.0])
        order = g3d.depth_order(layer, perspective_mvp())
        # +z is away from the eye under this mvp, so 5.0 is the farthest and must be drawn
        # first -- it is the one everything else has to paint over.
        assert list(order) == [1, 3, 0, 2]

    def test_an_orthographic_view_orders_by_the_z_row_instead(self):
        """``w`` is constant under orthographic projection and says nothing about depth."""
        mvp = np.eye(4, dtype=np.float32)
        mvp[2, :3] = (0.0, 0.0, 1.0)  # depth lives in z
        layer = cloud([0.0, 5.0, -3.0, 2.0])
        assert list(g3d.depth_order(layer, mvp)) == [1, 3, 0, 2]

    def test_ties_keep_their_array_order(self):
        """The sort is stable, so a flat cloud is not reshuffled for no reason."""
        layer = cloud([1.0, 1.0, 1.0, 1.0])
        assert list(g3d.depth_order(layer, perspective_mvp())) == [0, 1, 2, 3]

    def test_reversing_the_view_reverses_the_order(self):
        layer = cloud([0.0, 5.0, -3.0, 2.0])
        forward = list(g3d.depth_order(layer, perspective_mvp((0.0, 0.0, 1.0))))
        backward = list(g3d.depth_order(layer, perspective_mvp((0.0, 0.0, -1.0))))
        assert backward == forward[::-1]

    def test_the_same_view_reuses_the_cached_order(self):
        layer = cloud([0.0, 5.0, -3.0, 2.0])
        first = g3d.depth_order(layer, perspective_mvp())
        assert g3d.depth_order(layer, perspective_mvp()) is first

    def test_a_nudge_below_the_threshold_does_not_re_sort(self):
        """Re-sorting is milliseconds; a drag must not pay it once per frame."""
        layer = cloud(np.linspace(-1.0, 1.0, 64))
        first = g3d.depth_order(layer, perspective_mvp((0.0, 0.0, 1.0)))
        assert g3d.depth_order(layer, perspective_mvp((0.0, 0.005, 1.0))) is first

    def test_turning_the_camera_re_sorts(self):
        layer = cloud(np.linspace(-1.0, 1.0, 64))
        first = g3d.depth_order(layer, perspective_mvp((0.0, 0.0, 1.0)))
        second = g3d.depth_order(layer, perspective_mvp((1.0, 0.0, 0.2)))
        assert second is not first

    def test_new_geometry_invalidates_the_order(self):
        layer = cloud([0.0, 5.0, -3.0, 2.0])
        g3d.depth_order(layer, perspective_mvp())
        layer.vertices = np.zeros((3, 3), dtype=np.float32)
        assert len(g3d.depth_order(layer, perspective_mvp())) == 3

    def test_an_in_place_edit_is_invalidated_by_the_upload(self, recorder):
        """``id(vertices)`` cannot see an in-place edit, so ``update_gpu_data`` clears the
        memo -- it is the one place that knows the vertices were just re-uploaded."""
        layer = cloud([0.0, 5.0, -3.0, 2.0])
        assert list(g3d.depth_order(layer, perspective_mvp())) == [1, 3, 0, 2]
        layer.vertices[:, 2] = [5.0, 0.0, 2.0, -3.0]
        make_renderer().update_gpu_data(layer, layer._gl)
        assert list(g3d.depth_order(layer, perspective_mvp())) == [0, 2, 1, 3]

    def test_an_indexed_layer_is_left_alone(self):
        """Its own index buffer already decides the order; a second one would fight it."""
        layer = cloud([0.0, 5.0, -3.0, 2.0])
        layer.indices = np.arange(4, dtype=np.uint32)
        assert g3d.depth_order(layer, perspective_mvp()) is None

    def test_a_big_cloud_keeps_its_stale_order_mid_drag(self):
        """``may_resort=False`` is what stops a 20 ms sort landing on every frame of a
        rotation; the order it keeps is stale, which is the INTERACTIVE bargain."""
        layer = cloud(np.linspace(-1.0, 1.0, 64))
        first = g3d.depth_order(layer, perspective_mvp((0.0, 0.0, 1.0)))
        kept = g3d.depth_order(layer, perspective_mvp((1.0, 0.0, 0.2)), may_resort=False)
        assert kept is first

    def test_a_cloud_with_no_order_yet_sorts_even_mid_drag(self):
        """No order at all is the artefact at its worst -- worse than a stale one."""
        layer = cloud(np.linspace(-1.0, 1.0, 64))
        assert g3d.depth_order(layer, perspective_mvp(), may_resort=False) is not None


class TestSortedDraw:
    def test_a_translucent_cloud_draws_through_its_index_buffer(self, recorder):
        make_renderer().draw(cloud(np.linspace(-1.0, 1.0, 64)), make_ctx())
        assert [d[0] for d in recorder.draws] == ["elements"]
        assert recorder.draws[0][2] == 64

    def test_an_opaque_cloud_still_draws_straight_from_the_vertex_buffer(self, recorder):
        layer = cloud(np.linspace(-1.0, 1.0, 64))
        layer.colors[:, 3] = 1.0
        make_renderer().draw(layer, make_ctx())
        assert [d[0] for d in recorder.draws] == ["arrays"]
        assert not recorder.uploads

    def test_the_order_is_uploaded_once_and_reused(self, recorder):
        """Re-uploading 7 MB of indices per frame would cost more than the sort it follows."""
        layer = cloud(np.linspace(-1.0, 1.0, 64))
        renderer, ctx = make_renderer(), make_ctx()
        renderer.draw(layer, ctx)
        renderer.draw(layer, ctx)
        renderer.draw(layer, ctx)
        assert len(recorder.uploads) == 1
        assert len(recorder.draws) == 3


# ----------------------------------------------------------------------------------
# Draw order
# ----------------------------------------------------------------------------------


class TestOpacityOrdering:
    def test_translucent_3d_layers_are_drawn_last(self):
        cloud = make_layer("points", alpha=0.42)
        probes = make_layer("points")
        assert RendererManager._order_by_opacity([cloud, probes]) == [probes, cloud]

    def test_order_is_stable_within_each_group(self):
        a, b = make_layer("points"), make_layer("lines")
        x, y = make_layer("points", alpha=0.2), make_layer("lines", alpha=0.5)
        assert RendererManager._order_by_opacity([x, a, y, b]) == [a, b, x, y]

    def test_a_batch_without_3d_layers_is_returned_untouched(self):
        """2D layers have no depth test to arbitrate anything -- ``zorder`` is the contract."""
        first, second = PolylineLayer(), PolylineLayer()
        first.style.alpha = 0.3
        batch = [first, second]
        assert RendererManager._order_by_opacity(batch) is batch

    def test_2d_layers_keep_their_place_ahead_of_a_translucent_cloud(self):
        flat = PolylineLayer()
        flat.style.alpha = 0.3
        cloud = make_layer("points", alpha=0.42)
        assert RendererManager._order_by_opacity([cloud, flat]) == [flat, cloud]
