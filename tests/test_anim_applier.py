"""Test the bridge from an evaluated timeline to a live scene.

These run against a **real** ``GPULinePlot``. It constructs without a window and without a
GL context (the same headless path ``tests/test_layerops3d.py`` and
``tests/test_gui_engine_integration.py`` rely on), and the layers are built through
``glplot.gui.layerops`` / ``layerops3d`` rather than by hand, so what is asserted here is
what the GUI would actually be looking at.

Three things get more attention than the rest, because they are the three ways this module
can be wrong without looking wrong:

* **Dirty flags.** A geometry swap that does not set ``gpu_dirty`` changes the CPU array
  and leaves the old buffer on the card, so the picture never moves. A style change that
  does not reach ``mark_scene_dirty`` never triggers a redraw.
* **The 3D colour trap.** ``geometry3d.py`` reads ``style.color`` at upload time, so a 3D
  recolour needs ``gpu_dirty`` too, where a 2D one does not.
* **Staleness.** A timeline outlives the layers it was keyed against. Playing an animation
  whose layer was deleted an hour ago must be a no-op, not an exception.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.anim import applier as ap
from glplot.anim import primitives as P
from glplot.core.timeline import Timeline
from glplot.engine import GPULinePlot
from glplot.gui import layerops as lo
from glplot.gui import layerops3d as l3


@pytest.fixture
def fig():
    """A 2D figure. ``GPULinePlot()`` opens no window and touches no GL."""
    return GPULinePlot()


@pytest.fixture
def fig3d():
    plot = GPULinePlot()
    plot.set_ndim(3)
    return plot


def _line(plot, n=10, label="Line"):
    xs = np.linspace(0.0, 1.0, n)
    return lo.add_xy_layer(plot, xs, xs**2, kind="line", label=label, color=(0, 0, 1, 1))


def _scatter(plot, n=10, label="Scatter", **kw):
    xs = np.linspace(0.0, 1.0, n)
    return lo.add_xy_layer(plot, xs, xs, kind="scatter", label=label, **kw)


def _points3d(plot, n=12, label="Cloud"):
    rng = np.random.default_rng(0)
    x, y, z = rng.random((3, n))
    return l3.add_xyz_layer(plot, x, y, z, kind="scatter3d", label=label)


def _surface3d(plot, label="Surface"):
    """A layer with an index buffer — the case a vertex-count change must refuse."""
    gx, gy = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 4))
    return l3.add_xyz_layer(
        plot, gx.ravel(), gy.ravel(), (gx * gy).ravel(), kind="surface3d", label=label
    )


def _clean(plot):
    """Clear every dirty flag, so the next assertion is about what we just did."""
    for layer in plot.scene.layers:
        layer.dirty.clear()
    plot.frame.dirty_scene = False
    plot.cache.refresh_requested = False


# ----------------------------------------------------------------------------------
# Layer style
# ----------------------------------------------------------------------------------


class TestLayerStyle:
    def test_alpha_reaches_the_layer(self, fig):
        layer = _line(fig)
        assert ap.apply_values(fig, {(layer.layer_id, "alpha"): 0.25}) == 1
        assert layer.style.alpha == pytest.approx(0.25)

    def test_an_overshooting_alpha_is_clamped_not_refused(self, fig):
        """`back`/`elastic` legitimately produce 1.03 mid-flight."""
        layer = _line(fig)
        ap.apply_values(fig, {(layer.layer_id, "alpha"): 1.08})
        assert layer.style.alpha == pytest.approx(1.0)

    def test_visible_is_coerced_to_a_bool(self, fig):
        layer = _line(fig)
        ap.apply_values(fig, {(layer.layer_id, "visible"): False})
        assert layer.style.visible is False

    def test_color_accepts_rgb_and_gains_alpha_one(self, fig):
        layer = _line(fig)
        ap.apply_values(fig, {(layer.layer_id, "color"): (1.0, 0.5, 0.0)})
        assert layer.style.color == pytest.approx((1.0, 0.5, 0.0, 1.0))

    def test_translation_lands_on_the_layer_not_the_style(self, fig):
        layer = _line(fig)
        ap.apply_values(fig, {(layer.layer_id, "translation"): [2.0, -3.0]})
        assert layer.translation == (2.0, -3.0)
        assert not hasattr(layer.style, "translation")

    def test_point_size_line_width_and_zorder(self, fig):
        layer = _scatter(fig)
        ap.apply_values(
            fig,
            {
                (layer.layer_id, "point_size"): 14.0,
                (layer.layer_id, "line_width"): 3.5,
                (layer.layer_id, "zorder"): 7.4,
            },
        )
        assert layer.style.point_size == pytest.approx(14.0)
        assert layer.style.line_width == pytest.approx(3.5)
        assert layer.style.zorder == 7  # rounded to an int

    def test_a_style_change_marks_the_scene_dirty(self, fig):
        layer = _line(fig)
        _clean(fig)
        ap.apply_values(fig, {(layer.layer_id, "alpha"): 0.5})
        assert fig.frame.dirty_scene is True
        assert fig.cache.refresh_requested is True
        assert fig.cache.capture_window is None

    def test_a_2d_recolour_needs_no_gpu_upload(self, fig):
        """2D uniforms re-push every draw; only the 3D path caches colour in a VBO."""
        layer = _line(fig)
        _clean(fig)
        ap.apply_values(fig, {(layer.layer_id, "color"): (1, 0, 0, 1)})
        assert layer.dirty.gpu_dirty is False

    def test_a_3d_recolour_sets_gpu_dirty(self, fig3d):
        """The 3D colour trap: geometry3d.py reads style.color at upload time."""
        layer = _points3d(fig3d)
        _clean(fig3d)
        ap.apply_values(fig3d, {(layer.layer_id, "color"): (1, 0, 0, 1)})
        assert layer.style.color == pytest.approx((1.0, 0.0, 0.0, 1.0))
        assert layer.dirty.gpu_dirty is True

    def test_a_malformed_colour_is_skipped_not_raised(self, fig):
        layer = _line(fig)
        before = layer.style.color
        assert ap.apply_values(fig, {(layer.layer_id, "color"): (1.0, 2.0)}) == 0
        assert layer.style.color == before

    def test_a_non_finite_value_is_skipped(self, fig):
        layer = _line(fig)
        assert ap.apply_values(fig, {(layer.layer_id, "point_size"): float("nan")}) == 0

    def test_an_unknown_property_is_skipped(self, fig):
        layer = _line(fig)
        assert ap.apply_values(fig, {(layer.layer_id, "wobble"): 1.0}) == 0

    def test_a_style_field_outside_the_allowlist_is_skipped(self, fig):
        """`use_colormap` has a second, authoritative home in metadata; cmap owns it."""
        layer = _line(fig)
        assert ap.apply_values(fig, {(layer.layer_id, "use_colormap"): True}) == 0


# ----------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------


class TestGeometry:
    def test_pts_is_swapped_and_gpu_dirty_is_set(self, fig):
        layer = _line(fig, 10)
        _clean(fig)
        target = np.zeros((10, 2), dtype=np.float32)
        assert ap.apply_values(fig, {(layer.layer_id, "pts"): target}) == 1
        assert np.allclose(layer.pts, 0.0)
        assert layer.dirty.gpu_dirty is True, "without this the old VBO stays on screen"
        assert layer.dirty.bounds_dirty is True

    def test_vertices_is_an_alias_that_resolves_against_the_layer(self, fig):
        """A verb should not have to know whether its target keeps geometry in pts."""
        layer = _line(fig, 10)
        assert ap.apply_values(fig, {(layer.layer_id, "vertices"): np.ones((10, 2))}) == 1
        assert np.allclose(layer.pts, 1.0)

    def test_a_3d_layer_takes_the_same_property_name(self, fig3d):
        layer = _points3d(fig3d, 12)
        assert ap.apply_values(fig3d, {(layer.layer_id, "pts"): np.zeros((12, 3))}) == 1
        assert np.allclose(layer.vertices, 0.0)

    def test_a_vertex_count_change_refits_the_colours(self, fig):
        layer = _scatter(fig, 10)
        assert layer.colors is not None and len(layer.colors) == 10
        ap.apply_values(fig, {(layer.layer_id, "pts"): np.zeros((4, 2), dtype=np.float32)})
        assert len(layer.pts) == 4
        assert len(layer.colors) == 4, "a short colour VBO is read past its end"

    def test_the_legacy_mirror_follows_the_new_array(self, fig):
        """An orphaned mirror pins the old array and survives remove_layer's purge."""
        layer = _scatter(fig, 10)
        old = layer.pts
        mirrors = [m for m in fig.scene.scatters if m.pts is old]
        assert mirrors, "the fixture should have produced a mirror to begin with"
        ap.apply_values(fig, {(layer.layer_id, "pts"): np.zeros((10, 2), dtype=np.float32)})
        assert mirrors[0].pts is layer.pts

    def test_mixing_2d_and_3d_geometry_is_refused(self, fig):
        layer = _line(fig, 10)
        before = layer.pts.copy()
        assert ap.apply_values(fig, {(layer.layer_id, "pts"): np.zeros((10, 3))}) == 0
        assert np.allclose(layer.pts, before)

    def test_an_indexed_layer_refuses_a_vertex_count_change(self, fig3d):
        """Keeping the indices across a resize makes the draw read past the VBO's end."""
        layer = _surface3d(fig3d)
        assert layer.indices is not None
        n = len(layer.vertices)
        assert ap.apply_values(fig3d, {(layer.layer_id, "vertices"): np.zeros((n - 1, 3))}) == 0
        assert len(layer.vertices) == n

    def test_an_indexed_layer_accepts_a_same_count_morph(self, fig3d):
        layer = _surface3d(fig3d)
        n = len(layer.vertices)
        assert ap.apply_values(fig3d, {(layer.layer_id, "vertices"): np.zeros((n, 3))}) == 1
        assert np.allclose(layer.vertices, 0.0)

    def test_empty_geometry_is_refused(self, fig):
        layer = _line(fig, 10)
        assert ap.apply_values(fig, {(layer.layer_id, "pts"): np.zeros((0, 2))}) == 0

    def test_a_3d_geometry_change_re_syncs_the_view(self, fig3d):
        """Bounds moved, so the axis box and the auto-fit distance have to follow."""
        layer = _points3d(fig3d, 12)
        moved = np.asarray(layer.vertices, dtype=np.float32) + 100.0
        ap.apply_values(fig3d, {(layer.layer_id, "vertices"): moved})
        assert layer.metadata.get("scene_bounds") is not None
        assert layer.metadata["scene_bounds"][1] > 50.0


class TestProgressiveReveal:
    def test_a_half_drawn_polyline_keeps_its_first_half(self, fig):
        layer = _line(fig, 10)
        full = layer.pts.copy()
        assert ap.apply_values(fig, {(layer.layer_id, "draw_fraction"): 0.5}) == 1
        assert len(layer.pts) == 5
        assert np.allclose(layer.pts, full[:5])

    def test_it_grows_back_to_the_original_array_object(self, fig):
        layer = _line(fig, 10)
        full = layer.pts
        ap.apply_values(fig, {(layer.layer_id, "draw_fraction"): 0.3})
        ap.apply_values(fig, {(layer.layer_id, "draw_fraction"): 1.0})
        assert layer.pts is full

    def test_fraction_zero_leaves_a_degenerate_stub_not_an_empty_array(self, fig):
        layer = _line(fig, 10)
        ap.apply_values(fig, {(layer.layer_id, "draw_fraction"): 0.0})
        assert len(layer.pts) == 2

    def test_the_full_geometry_is_parked_in_metadata(self, fig):
        layer = _line(fig, 10)
        ap.apply_values(fig, {(layer.layer_id, "draw_fraction"): 0.5})
        assert len(layer.metadata[ap.FULL_GEOMETRY_KEY]) == 10

    def test_a_reveal_sets_gpu_dirty(self, fig):
        layer = _line(fig, 10)
        _clean(fig)
        ap.apply_values(fig, {(layer.layer_id, "draw_fraction"): 0.5})
        assert layer.dirty.gpu_dirty is True

    def test_a_held_fraction_does_not_re_upload(self, fig):
        layer = _line(fig, 10)
        ap.apply_values(fig, {(layer.layer_id, "draw_fraction"): 0.5})
        _clean(fig)
        # 0.52 of 10 vertices still rounds to 5 -- nothing to do.
        assert ap.apply_values(fig, {(layer.layer_id, "draw_fraction"): 0.52}) == 1
        assert layer.dirty.gpu_dirty is False

    def test_an_indexed_layer_cannot_be_revealed(self, fig3d):
        layer = _surface3d(fig3d)
        n = len(layer.vertices)
        assert ap.apply_values(fig3d, {(layer.layer_id, "draw_fraction"): 0.5}) == 0
        assert len(layer.vertices) == n

    def test_reset_reveal_restores_and_clears_the_park(self, fig):
        layer = _line(fig, 10)
        ap.apply_values(fig, {(layer.layer_id, "draw_fraction"): 0.4})
        assert ap.reset_reveal(layer) is True
        assert len(layer.pts) == 10
        assert ap.FULL_GEOMETRY_KEY not in layer.metadata
        assert ap.reset_reveal(layer) is False

    def test_text_is_revealed_character_by_character(self, fig):
        fig.add_text(0.0, 0.0, "Hello world", label="T")
        layer = fig.scene.layers[-1]
        ap.apply_values(fig, {(layer.layer_id, "text_fraction"): 0.5})
        assert layer.text == "Hello "
        ap.apply_values(fig, {(layer.layer_id, "text_fraction"): 1.0})
        assert layer.text == "Hello world"

    def test_reset_reveal_restores_the_text(self, fig):
        fig.add_text(0.0, 0.0, "abcdef", label="T")
        layer = fig.scene.layers[-1]
        ap.apply_values(fig, {(layer.layer_id, "text_fraction"): 0.5})
        ap.reset_reveal(layer)
        assert layer.text == "abcdef"

    def test_text_fraction_on_a_layer_with_no_text_is_skipped(self, fig):
        layer = _line(fig)
        assert ap.apply_values(fig, {(layer.layer_id, "text_fraction"): 0.5}) == 0


class TestColormap:
    def test_a_values2d_scatter_takes_a_matplotlib_name(self, fig):
        layer = _scatter(fig, 10, c=np.arange(10.0), cmap="viridis")
        assert lo.layer_colormap_kind(layer) == "values2d"
        before = layer.colors.copy()
        assert ap.apply_values(fig, {(layer.layer_id, "cmap"): "magma"}) == 1
        assert layer.metadata["cmap"] == "magma"
        assert not np.allclose(layer.colors, before)

    def test_a_polyline_takes_a_shader_colormap_index(self, fig):
        layer = _line(fig)
        assert lo.layer_colormap_kind(layer) == "gl_line"
        assert ap.apply_values(fig, {(layer.layer_id, "cmap"): 2}) == 1
        assert lo.layer_gl_colormap(layer) == (True, 2)

    def test_a_polyline_also_takes_a_shader_colormap_name(self, fig):
        layer = _line(fig)
        ap.apply_values(fig, {(layer.layer_id, "cmap"): "Magma"})
        enabled, index = lo.layer_gl_colormap(layer)
        assert enabled is True
        assert lo.GL_COLORMAP_NAMES[index] == "Magma"

    def test_a_name_that_is_not_a_shader_colormap_is_refused(self, fig):
        """The two colormap namespaces overlap but are not the same list."""
        layer = _line(fig)
        assert "coolwarm" not in [n.lower() for n in lo.GL_COLORMAP_NAMES]
        assert ap.apply_values(fig, {(layer.layer_id, "cmap"): "coolwarm"}) == 0
        assert lo.layer_gl_colormap(layer) == (None, None), "follow the scene, untouched"

    def test_a_shared_name_is_matched_case_insensitively(self, fig):
        """`viridis` names a matplotlib colormap *and* a shader one; on a line it is the latter."""
        layer = _line(fig)
        assert ap.apply_values(fig, {(layer.layer_id, "cmap"): "viridis"}) == 1
        _, index = lo.layer_gl_colormap(layer)
        assert lo.GL_COLORMAP_NAMES[index] == "Viridis"

    def test_a_layer_with_no_colormap_is_skipped_rather_than_raising(self, fig):
        """set_layer_colormap raises for these; playback must not."""
        layer = _scatter(fig, 10)
        assert lo.layer_colormap_kind(layer) is None
        assert ap.apply_values(fig, {(layer.layer_id, "cmap"): "magma"}) == 0


# ----------------------------------------------------------------------------------
# Cameras, axes, timeline, options
# ----------------------------------------------------------------------------------


class TestCamera3D:
    def test_every_documented_property_lands(self, fig3d):
        _points3d(fig3d)
        applied = ap.apply_values(
            fig3d,
            {
                ("camera3d", "elev"): 61.0,
                ("camera3d", "azim"): 123.0,
                ("camera3d", "roll"): -12.0,
                ("camera3d", "distance"): 4.5,
                ("camera3d", "fov"): 30.0,
                ("camera3d", "pan"): (1.0, 2.0, 3.0),
                ("camera3d", "box_aspect"): (1.0, 1.0, 2.0),
            },
        )
        camera = fig3d.camera3d
        assert applied == 7
        assert (camera.elev, camera.azim, camera.roll) == (61.0, 123.0, -12.0)
        assert camera.distance == pytest.approx(4.5)
        assert camera.fov == pytest.approx(30.0)
        assert camera.pan == (1.0, 2.0, 3.0)
        assert camera.box_aspect == (1.0, 1.0, 2.0)

    def test_distance_none_is_applied_not_skipped(self, fig3d):
        fig3d.camera3d.distance = 9.0
        assert ap.apply_values(fig3d, {("camera3d", "distance"): None}) == 1
        assert fig3d.camera3d.distance is None

    def test_box_aspect_none_is_applied(self, fig3d):
        fig3d.camera3d.box_aspect = (1.0, 1.0, 1.0)
        ap.apply_values(fig3d, {("camera3d", "box_aspect"): None})
        assert fig3d.camera3d.box_aspect is None

    def test_the_camera_is_pushed_onto_every_3d_layer(self, fig3d):
        """The renderer builds its matrix from metadata; without the sync nothing moves."""
        layer = _points3d(fig3d)
        ap.apply_values(fig3d, {("camera3d", "azim"): 77.0})
        assert layer.metadata["camera"]["azim"] == pytest.approx(77.0)

    def test_a_string_property_goes_through_its_validator(self, fig3d):
        ap.apply_values(fig3d, {("camera3d", "projection"): "ortho"})
        assert fig3d.camera3d.projection == "orthographic"

    def test_an_invalid_string_is_skipped_not_raised(self, fig3d):
        before = fig3d.camera3d.projection
        assert ap.apply_values(fig3d, {("camera3d", "projection"): "isometric"}) == 0
        assert fig3d.camera3d.projection == before

    def test_a_bad_pan_shape_is_skipped(self, fig3d):
        before = fig3d.camera3d.pan
        assert ap.apply_values(fig3d, {("camera3d", "pan"): (1.0, 2.0)}) == 0
        assert fig3d.camera3d.pan == before


class TestCamera2D:
    def test_centre_and_zoom_land(self, fig):
        applied = ap.apply_values(
            fig,
            {
                ("camera", "cx"): 3.0,
                ("camera", "cy"): -1.0,
                ("camera", "zoom_x"): 2.0,
                ("camera", "zoom_y"): 4.0,
            },
        )
        assert applied == 4
        assert (fig.camera.cx, fig.camera.cy) == (3.0, -1.0)
        assert (fig.camera.zoom_x, fig.camera.zoom_y) == (2.0, 4.0)

    def test_camera2d_is_a_synonym(self, fig):
        ap.apply_values(fig, {("camera2d", "cx"): 5.0})
        assert fig.camera.cx == pytest.approx(5.0)

    def test_zoom_sets_both_axes_through_the_legacy_property(self, fig):
        ap.apply_values(fig, {("camera", "zoom"): 3.0})
        assert fig.camera.zoom_x == pytest.approx(3.0)
        assert fig.camera.zoom_y == pytest.approx(3.0)

    def test_an_overshooting_zoom_is_clamped_to_the_cameras_own_limits(self, fig):
        """`back`/`elastic` really can drive a zoom keyframe through zero."""
        ap.apply_values(fig, {("camera", "zoom_x"): -4.0})
        assert fig.camera.zoom_x == pytest.approx(fig.camera.zoom_min)


class TestAxes3DAndFriends:
    def test_axes3d_booleans_and_labels(self, fig3d):
        applied = ap.apply_values(
            fig3d,
            {
                ("axes3d", "show_grid"): False,
                ("axes3d", "show_ticks"): False,
                ("axes3d", "tick_count"): 9,
                ("axes3d", "xlabel"): "t",
                ("axes3d", "zlim"): (0.0, 5.0),
            },
        )
        assert applied == 5
        assert fig3d.axes3d.show_grid is False
        assert fig3d.axes3d.tick_count == 9
        assert fig3d.axes3d.xlabel == "t"
        assert fig3d.axes3d.zlim == (0.0, 5.0)

    def test_an_unknown_axes3d_field_is_skipped(self, fig3d):
        assert ap.apply_values(fig3d, {("axes3d", "show_moon"): True}) == 0

    def test_the_timeline_speed_can_be_animated(self, fig):
        assert ap.apply_values(fig, {("timeline", "speed"): 0.25}) == 1
        assert fig.active_panel.timeline.speed == pytest.approx(0.25)

    def test_the_playhead_is_deliberately_not_animatable(self, fig):
        """A track that moves the playhead is evaluated from the playhead: a feedback loop."""
        before = fig.active_panel.timeline.time
        assert ap.apply_values(fig, {("timeline", "time"): 3.0}) == 0
        assert fig.active_panel.timeline.time == before

    def test_global_alpha_lives_on_the_engine_not_on_options(self, fig):
        assert ap.apply_values(fig, {("options", "global_alpha"): 0.4}) == 1
        assert fig.global_alpha == pytest.approx(0.4)

    def test_an_option_is_coerced_to_the_type_it_already_holds(self, fig):
        assert ap.apply_values(fig, {("options", "lod_enabled"): 0.0}) == 1
        assert fig.options.lod_enabled is False
        assert ap.apply_values(fig, {("options", "axis_grid_alpha"): 0.75}) == 1
        assert fig.options.axis_grid_alpha == pytest.approx(0.75)

    def test_an_option_that_does_not_exist_is_skipped(self, fig):
        assert ap.apply_values(fig, {("options", "turbo_mode"): True}) == 0


# ----------------------------------------------------------------------------------
# Staleness, dispatch and the whole-frame contract
# ----------------------------------------------------------------------------------


class TestStalenessAndDispatch:
    def test_a_deleted_layers_track_is_skipped_without_raising(self, fig):
        """The case the whole design is arranged around: a timeline outlives its layers."""
        layer = _line(fig)
        layer_id = layer.layer_id
        timeline = fig.active_panel.timeline
        P.FadeIn().apply(timeline, layer_id, 0.0, 1.0)
        P.MoveTo((1.0, 1.0)).apply(timeline, layer_id, 0.0, 1.0)

        lo.remove_layer(fig, None, layer)
        assert lo.find_layer(fig, layer_id) is None

        # Two tracks, both stale. Nothing raises and nothing is applied.
        assert ap.apply_timeline(fig, time=0.5) == 0

    def test_playback_survives_a_partly_stale_timeline(self, fig):
        alive = _line(fig, label="Alive")
        doomed = _line(fig, label="Doomed")
        timeline = fig.active_panel.timeline
        P.FadeIn().apply(timeline, alive.layer_id, 0.0, 2.0)
        P.FadeIn().apply(timeline, doomed.layer_id, 0.0, 2.0)
        lo.remove_layer(fig, None, doomed)

        assert ap.apply_timeline(fig, time=2.0) == 1
        assert alive.style.alpha == pytest.approx(1.0)

    def test_an_unknown_target_is_skipped(self, fig):
        assert ap.apply_values(fig, {("spaceship", "warp"): 9}) == 0

    def test_a_malformed_key_is_skipped(self, fig):
        assert ap.apply_values(fig, {"not-a-pair": 1}) == 0

    def test_an_empty_dict_is_a_no_op(self, fig):
        _clean(fig)
        assert ap.apply_values(fig, {}) == 0
        assert fig.frame.dirty_scene is False

    def test_nothing_applied_means_nothing_marked_dirty(self, fig):
        _clean(fig)
        ap.apply_values(fig, {(123456, "alpha"): 0.5})
        assert fig.frame.dirty_scene is False

    def test_a_bool_target_is_not_mistaken_for_a_layer_id(self, fig):
        """bool is an int subclass; True must not resolve to layer 1."""
        assert ap.apply_values(fig, {(True, "alpha"): 0.5}) == 0

    def test_apply_timeline_defaults_to_the_panels_own_timeline_and_playhead(self, fig):
        layer = _line(fig)
        timeline = fig.active_panel.timeline
        P.FadeIn().apply(timeline, layer.layer_id, 0.0, 2.0)
        timeline.seek(1.0)
        assert ap.apply_timeline(fig) == 1
        assert layer.style.alpha == pytest.approx(0.5)

    def test_apply_timeline_accepts_a_foreign_timeline(self, fig):
        layer = _line(fig)
        other = Timeline(duration=4.0)
        P.FadeOut().apply(other, layer.layer_id, 0.0, 4.0)
        ap.apply_timeline(fig, timeline=other, time=4.0)
        assert layer.style.alpha == pytest.approx(0.0)

    def test_a_panel_with_no_timeline_is_a_no_op(self, fig):
        class _Bare:
            timeline = None

        assert ap.apply_timeline(fig, panel=_Bare()) == 0


class TestPanels:
    """A timeline is per panel, so the applier has to be too."""

    def test_a_layer_id_is_resolved_inside_the_named_panel(self, fig):
        from glplot.core.layout import PanelSpec

        first = fig.active_panel
        front = _line(fig, label="Front")
        fig.add_panel(PanelSpec(rect_frac=(0.5, 0.0, 0.5, 1.0), name="second"))
        second = fig.active_panel
        assert second is not first
        back = _line(fig, label="Back")

        # ``back`` lives in the active (second) panel; asking the first panel for it must
        # not fall through to the active one.
        assert ap.apply_values(fig, {(back.layer_id, "alpha"): 0.5}, panel=first) == 0
        assert ap.apply_values(fig, {(front.layer_id, "alpha"): 0.5}, panel=first) == 1
        assert front.style.alpha == pytest.approx(0.5)

    def test_a_background_panel_gets_its_own_cache_invalidated(self, fig):
        from glplot.core.layout import PanelSpec

        first = fig.active_panel
        front = _line(fig, label="Front")
        fig.add_panel(PanelSpec(rect_frac=(0.5, 0.0, 0.5, 1.0), name="second"))
        first.cache.refresh_requested = False
        first.cache.capture_window = "stale"

        ap.apply_values(fig, {(front.layer_id, "alpha"): 0.25}, panel=first)
        assert first.cache.refresh_requested is True
        assert first.cache.capture_window is None


# ----------------------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------------------


class TestEndToEnd:
    """A script written with verbs, played through the applier, asserted on the scene."""

    def test_a_fade_plays_over_its_span(self, fig):
        layer = _line(fig)
        timeline = fig.active_panel.timeline
        P.FadeIn(easing="linear").apply(timeline, layer.layer_id, 0.0, 2.0)

        for time, expected in ((0.0, 0.0), (1.0, 0.5), (2.0, 1.0)):
            ap.apply_timeline(fig, time=time)
            assert layer.style.alpha == pytest.approx(expected)

    def test_a_morph_between_two_layers_of_different_lengths(self, fig):
        source = _line(fig, 8, label="Source")
        dest = _line(fig, 30, label="Dest")
        timeline = fig.active_panel.timeline
        P.Transform.from_layers(source, dest, easing="linear").apply(
            timeline, source.layer_id, 0.0, 2.0
        )
        _clean(fig)

        ap.apply_timeline(fig, time=2.0)
        assert len(source.pts) == 30, "resampled to the longer side, as documented"
        assert np.allclose(source.pts, dest.pts, atol=1e-5)
        assert source.dirty.gpu_dirty is True

    def test_a_morph_refits_a_per_vertex_colour_array(self, fig):
        source = _scatter(fig, 8, label="Source", c=np.arange(8.0))
        timeline = fig.active_panel.timeline
        P.Transform(source.pts.copy(), np.zeros((25, 2), dtype=np.float32)).apply(
            timeline, source.layer_id, 0.0, 1.0
        )
        ap.apply_timeline(fig, time=1.0)
        assert len(source.pts) == 25
        assert len(source.colors) == 25, "a short colour VBO is read past its end"

    def test_a_rotation_really_rotates_on_screen(self, fig):
        layer = _line(fig, 12)
        original = layer.pts.copy()
        centre = original.mean(axis=0)
        radius = np.linalg.norm(original - centre, axis=1)
        timeline = fig.active_panel.timeline
        P.Rotate(original, 180.0, easing="linear").apply(timeline, layer.layer_id, 0.0, 2.0)

        ap.apply_timeline(fig, time=1.0)
        assert np.allclose(np.linalg.norm(layer.pts - centre, axis=1), radius, atol=0.05)

    def test_a_succession_reaches_the_scene_in_order(self, fig):
        layer = _line(fig)
        timeline = fig.active_panel.timeline
        P.succession(
            timeline,
            (P.Create(easing="linear"), layer.layer_id, 1.0),
            (P.ColorTo(to=(1, 0, 0), start_color=(0, 0, 1), easing="linear"), layer.layer_id, 1.0),
            (P.FadeOut(easing="linear"), layer.layer_id, 1.0),
        )

        ap.apply_timeline(fig, time=0.5)
        assert len(layer.pts) == 5
        assert layer.style.alpha == pytest.approx(1.0)

        ap.apply_timeline(fig, time=1.5)
        assert len(layer.pts) == 10
        assert layer.style.color == pytest.approx((0.5, 0.0, 0.5, 1.0))

        ap.apply_timeline(fig, time=2.5)
        assert layer.style.alpha == pytest.approx(0.5)

    def test_an_orbit_drives_the_3d_camera_past_360(self, fig3d):
        _points3d(fig3d)
        timeline = fig3d.active_panel.timeline
        P.Orbit(turns=1.0).apply(timeline, "camera3d", 0.0, 4.0)

        ap.apply_timeline(fig3d, time=2.0)
        assert fig3d.camera3d.azim == pytest.approx(180.0)
        ap.apply_timeline(fig3d, time=4.0)
        assert fig3d.camera3d.azim == pytest.approx(360.0)

    def test_a_baked_timeline_can_be_replayed_frame_by_frame(self, fig):
        """The export path: every frame is computed, not accumulated."""
        layer = _line(fig)
        timeline = Timeline(duration=1.0, fps=10.0)
        P.FadeIn(easing="linear").apply(timeline, layer.layer_id, 0.0, 1.0, fit=False)

        seen = []
        for frame in timeline.bake():
            ap.apply_values(fig, frame)
            seen.append(layer.style.alpha)
        assert seen == pytest.approx(list(np.linspace(0.0, 1.0, 11)))

    def test_scrubbing_backwards_gives_the_same_values(self, fig):
        layer = _line(fig)
        timeline = fig.active_panel.timeline
        P.FadeIn(easing="smooth").apply(timeline, layer.layer_id, 0.0, 2.0)

        ap.apply_timeline(fig, time=1.5)
        forward = layer.style.alpha
        ap.apply_timeline(fig, time=0.2)
        ap.apply_timeline(fig, time=1.5)
        assert layer.style.alpha == pytest.approx(forward)
