"""Test the 3D axis decoration: ticks, box, floor, grid walls, marks and label anchors.

Pure numpy, like the module under test. The geometry is a function of ``(bounds, camera)``,
so every placement rule the module makes — which walls the grid goes on, which edge the
ticks sit on, where a label is anchored — is an assertion rather than a screenshot.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.core.camera3d import Camera3D
from glplot.renderers import axes3d

BOX = (-2.0, 2.0, -1.0, 3.0, 0.0, 10.0)
UNIT = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)


class TestAxisTicks:
    """Nice numbers, shared with the 2D axis renderer so a value prints one way."""

    def test_ticks_land_inside_the_range(self):
        values, labels = axes3d.axis_ticks(-2.0, 7.0, 5)
        assert values.size == len(labels) > 0
        assert values.min() >= -2.0 and values.max() <= 7.0

    def test_ticks_are_evenly_spaced_on_a_nice_step(self):
        values, _ = axes3d.axis_ticks(0.0, 10.0, 5)
        steps = np.diff(values)
        assert np.allclose(steps, steps[0])
        assert steps[0] in (1.0, 2.0, 2.5, 5.0)

    @pytest.mark.parametrize("lo,hi", [(1.0, 1.0), (5.0, 1.0), (np.nan, 1.0), (0.0, np.inf)])
    def test_a_degenerate_range_yields_no_ticks_rather_than_raising(self, lo, hi):
        values, labels = axes3d.axis_ticks(lo, hi)
        assert values.size == 0 and labels == []

    def test_tick_count_is_capped(self):
        values, _ = axes3d.axis_ticks(0.0, 1.0, 10_000)
        assert values.size <= axes3d.MAX_TICKS

    def test_the_cap_coarsens_rather_than_truncating(self):
        """Slicing the array labelled the bottom of the axis and left the top bare."""
        values, _ = axes3d.axis_ticks(0.0, 1.0, 10_000)
        assert values.size > 1
        step = float(np.diff(values)[0])
        assert values.min() <= step  # reaches the bottom
        assert values.max() >= 1.0 - step  # and the top
        assert np.allclose(np.diff(values), step)  # still one even spacing

    def test_coarsening_stays_on_the_nice_ladder(self):
        values, _ = axes3d.axis_ticks(-500.0, 500.0, 10_000)
        step = float(np.diff(values)[0])
        mantissa = step / 10 ** np.floor(np.log10(step))
        assert mantissa == pytest.approx(round(mantissa, 6))
        assert any(abs(mantissa - m) < 1e-6 for m in (1.0, 2.0, 2.5, 4.0, 5.0))

    def test_no_negative_zero_label(self):
        """ "-0" in an axis that straddles the origin reads as a rendering bug."""
        _, labels = axes3d.axis_ticks(-1.0, 1.0, 5)
        assert "-0" not in labels
        assert not any(label.startswith("-0.0") and float(label) == 0.0 for label in labels)

    def test_ticks_for_bounds_covers_three_axes(self):
        ticks = axes3d.ticks_for_bounds(BOX, 5)
        assert len(ticks) == 3
        for values, labels in ticks:
            assert values.size == len(labels)

    def test_ticks_for_bounds_takes_one_target_per_axis(self):
        """Three axes of a box are three different lengths on screen."""
        sparse = axes3d.ticks_for_bounds(BOX, (2, 2, 2))
        dense = axes3d.ticks_for_bounds(BOX, (2, 2, 12))
        assert sparse[0][0].size == dense[0][0].size  # x untouched
        assert dense[2][0].size > sparse[2][0].size  # z alone got denser


class TestAdaptiveTickDensity:
    """The count comes from the axis' length **in pixels**, as the 2D renderer's does.

    A fixed count is what made a 3D axis feel wrong under a dolly: the data bounds do not
    change when the camera moves, so five numbers were stretched across the whole window on
    the way in and crushed into a corner on the way out.
    """

    def _lengths(self, camera, w=900.0, h=600.0):
        return axes3d.axis_screen_lengths(BOX, camera, w / h, w, h)

    def test_dollying_in_lengthens_the_axes_on_screen(self):
        near = self._lengths(Camera3D(distance=8.0))
        far = self._lengths(Camera3D(distance=40.0))
        assert all(n > f for n, f in zip(near, far))

    def test_a_longer_axis_asks_for_more_ticks(self):
        def counts(distance):
            camera = Camera3D(distance=distance)
            return axes3d.adaptive_tick_counts(BOX, camera, 1.5, 900.0, 600.0)

        assert sum(counts(8.0)) > sum(counts(40.0))

    def test_the_density_is_about_one_tick_per_target_pixels(self):
        camera = Camera3D(distance=14.0)
        lengths = self._lengths(camera)
        counts = axes3d.adaptive_tick_counts(BOX, camera, 1.5, 900.0, 600.0)
        for length, count in zip(lengths, counts):
            if count <= axes3d.MIN_ADAPTIVE_TICKS or count >= axes3d.MAX_ADAPTIVE_TICKS:
                continue  # clamped, so the density is not what decided it
            assert length / (count - 1) == pytest.approx(axes3d.TICK_DENSITY_PX, rel=0.75)

    def test_counts_stay_within_their_bounds(self):
        for distance in (0.2, 1.0, 5.0, 50.0, 5000.0):
            counts = axes3d.adaptive_tick_counts(
                BOX, Camera3D(distance=distance), 1.5, 900.0, 600.0
            )
            for count in counts:
                assert axes3d.MIN_ADAPTIVE_TICKS <= count <= axes3d.MAX_ADAPTIVE_TICKS

    def test_a_forced_count_pins_every_axis(self):
        counts = axes3d.adaptive_tick_counts(
            BOX, Camera3D(distance=8.0), 1.5, 900.0, 600.0, forced=7
        )
        assert counts == (7, 7, 7)

    def test_an_axis_crossing_behind_the_eye_is_not_measured_as_zero(self):
        """Dollying *into* the data must not strip the labels off the axis being read."""
        inside = Camera3D(distance=0.01)
        lengths = self._lengths(inside)
        diagonal = float(np.hypot(900.0, 600.0))
        assert any(length == pytest.approx(diagonal) for length in lengths)
        counts = axes3d.adaptive_tick_counts(BOX, inside, 1.5, 900.0, 600.0)
        assert all(count > axes3d.MIN_ADAPTIVE_TICKS for count in counts)

    def test_lengths_reuse_a_supplied_edge_choice(self):
        camera = Camera3D(azim=-45.0)
        edges = axes3d.tick_edges(BOX, camera, 1.5)
        assert self._lengths(camera) == axes3d.axis_screen_lengths(
            BOX, camera, 1.5, 900.0, 600.0, edges
        )


class TestBoxGeometry:
    """The static pieces: the wireframe, the floor and the scale every offset uses."""

    def test_box_edges_are_twelve_segments(self):
        edges = axes3d.box_edges(BOX)
        assert edges.shape == (24, 3)
        assert edges.dtype == np.float32

    def test_box_edges_span_exactly_the_bounds(self):
        edges = axes3d.box_edges(BOX)
        for axis, (lo, hi) in enumerate(((BOX[0], BOX[1]), (BOX[2], BOX[3]), (BOX[4], BOX[5]))):
            assert edges[:, axis].min() == pytest.approx(lo)
            assert edges[:, axis].max() == pytest.approx(hi)

    def test_every_box_edge_is_axis_aligned(self):
        """A box edge varies in exactly one coordinate; two would be a diagonal."""
        edges = axes3d.box_edges(BOX).reshape(-1, 2, 3)
        for segment in edges:
            varying = np.count_nonzero(np.abs(segment[1] - segment[0]) > 1e-6)
            assert varying == 1

    def test_floor_sits_at_zmin(self):
        floor = axes3d.floor_quad(BOX)
        assert floor.shape == (6, 3)
        assert np.allclose(floor[:, 2], BOX[4])

    def test_box_diagonal_uses_the_longest_edge(self):
        assert axes3d.box_diagonal(BOX) == pytest.approx(10.0)

    def test_box_diagonal_survives_a_flat_scene(self):
        """A z span of zero must still yield a usable tick scale from x and y."""
        assert axes3d.box_diagonal((0.0, 4.0, 0.0, 2.0, 1.0, 1.0)) == pytest.approx(4.0)


class TestBackWalls:
    """The grid belongs on the far walls, so the data is seen against it not through it."""

    @pytest.mark.parametrize(
        "azim,elev,expected",
        [
            (-45.0, 28.0, (False, True, False)),  # eye at +x, -y, +z
            (135.0, 28.0, (True, False, False)),  # eye at -x, +y, +z
            (-45.0, -28.0, (False, True, True)),  # looking up: z wall at the top
        ],
    )
    def test_wall_is_opposite_the_eye(self, azim, elev, expected):
        camera = Camera3D(elev=elev, azim=azim)
        assert axes3d.back_wall_axes(camera) == expected

    def test_grid_lines_lie_on_the_chosen_walls(self):
        camera = Camera3D(elev=28.0, azim=-45.0)
        ticks = axes3d.ticks_for_bounds(BOX, 5)
        lines = axes3d.grid_lines(BOX, ticks, camera)
        assert lines.ndim == 2 and lines.shape[1] == 3
        assert len(lines) % 2 == 0
        x_at_max, y_at_max, z_at_max = axes3d.back_wall_axes(camera)
        wall_x = BOX[1] if x_at_max else BOX[0]
        wall_y = BOX[3] if y_at_max else BOX[2]
        wall_z = BOX[5] if z_at_max else BOX[4]
        # Every segment must be flat against one of the three chosen walls.
        on_wall = (
            np.isclose(lines[:, 0], wall_x)
            | np.isclose(lines[:, 1], wall_y)
            | np.isclose(lines[:, 2], wall_z)
        )
        assert bool(on_wall.all())

    def test_grid_lines_stay_inside_the_box(self):
        camera = Camera3D()
        lines = axes3d.grid_lines(BOX, axes3d.ticks_for_bounds(BOX, 5), camera)
        for axis, (lo, hi) in enumerate(((BOX[0], BOX[1]), (BOX[2], BOX[3]), (BOX[4], BOX[5]))):
            assert lines[:, axis].min() >= lo - 1e-6
            assert lines[:, axis].max() <= hi + 1e-6

    def test_grid_on_a_tickless_axis_is_empty_not_broken(self):
        empty = (np.zeros(0), [])
        lines = axes3d.grid_lines(BOX, (empty, empty, empty), Camera3D())
        assert lines.shape == (0, 3)


class TestTickPlacement:
    """Which edge the ticks sit on is a screen-space question, answered per camera."""

    def test_tick_edges_are_box_corners(self):
        for azim in (-135.0, -45.0, 45.0, 135.0):
            camera = Camera3D(azim=azim)
            x_edge, y_edge, z_edge = axes3d.tick_edges(BOX, camera, 1.6)
            assert x_edge[0] in (BOX[2], BOX[3]) and x_edge[1] in (BOX[4], BOX[5])
            assert y_edge[0] in (BOX[0], BOX[1]) and y_edge[1] in (BOX[4], BOX[5])
            assert z_edge[0] in (BOX[0], BOX[1]) and z_edge[1] in (BOX[2], BOX[3])

    def test_tick_edges_follow_the_camera(self):
        """Orbiting to the opposite side must move the ticks to the other edge."""
        front = axes3d.tick_edges(BOX, Camera3D(azim=-45.0), 1.6)
        back = axes3d.tick_edges(BOX, Camera3D(azim=135.0), 1.6)
        assert front != back

    def test_tick_marks_step_outward(self):
        """A mark that pointed inward would run across the data it is annotating."""
        camera = Camera3D()
        ticks = axes3d.ticks_for_bounds(BOX, 5)
        marks = axes3d.tick_marks(BOX, ticks, camera, 1.6).reshape(-1, 2, 3)
        centre = np.array(
            [0.5 * (BOX[0] + BOX[1]), 0.5 * (BOX[2] + BOX[3]), 0.5 * (BOX[4] + BOX[5])]
        )
        for segment in marks:
            inner = np.linalg.norm(segment[0] - centre)
            outer = np.linalg.norm(segment[1] - centre)
            assert outer > inner

    def test_tick_marks_are_short(self):
        marks = axes3d.tick_marks(BOX, axes3d.ticks_for_bounds(BOX, 5), Camera3D(), 1.6)
        segments = marks.reshape(-1, 2, 3)
        lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
        assert float(lengths.max()) < axes3d.box_diagonal(BOX) * 0.05

    def test_no_ticks_gives_an_empty_array(self):
        empty = (np.zeros(0), [])
        marks = axes3d.tick_marks(BOX, (empty, empty, empty), Camera3D(), 1.6)
        assert marks.shape == (0, 3)


class TestLabelAnchors:
    """Text placement. The anchors are world points; the HUD projects them."""

    def _anchors(self, **kw):
        ticks = axes3d.ticks_for_bounds(BOX, 5)
        return axes3d.label_anchors(BOX, ticks, Camera3D(), 1.6, **kw)

    def test_one_anchor_per_tick_on_each_axis(self):
        ticks = axes3d.ticks_for_bounds(BOX, 5)
        anchors = self._anchors()
        expected = sum(values.size for values, _ in ticks)
        assert len([a for a in anchors if a.kind == "tick"]) == expected

    def test_anchor_text_matches_the_tick_label(self):
        ticks = axes3d.ticks_for_bounds(BOX, 5)
        anchors = [a for a in self._anchors() if a.kind == "tick" and a.axis == "x"]
        assert [a.text for a in anchors] == list(ticks[0][1])

    def test_titles_appear_only_when_named(self):
        assert not [a for a in self._anchors() if a.kind == "title"]
        titled = self._anchors(axis_labels=("X", "Y", "Z"))
        titles = {a.axis: a.text for a in titled if a.kind == "title"}
        assert titles == {"x": "X", "y": "Y", "z": "Z"}

    def test_tick_labels_can_be_suppressed_while_titles_stay(self):
        anchors = self._anchors(tick_labels=False, axis_labels=("X", "Y", "Z"))
        assert not [a for a in anchors if a.kind == "tick"]
        assert len([a for a in anchors if a.kind == "title"]) == 3

    def test_titles_sit_further_out_than_the_numbers(self):
        """A title must clear the numbers, or the two overlap on every orientation.

        Compared per coordinate rather than by distance: the anchors share the coordinate
        of the edge they sit on (a title and a tick are on the *same* box edge), so an
        aggregate over all three axes is dominated by that shared term and would compare
        equal no matter how far apart they were.
        """
        anchors = self._anchors(axis_labels=("X", "Y", "Z"))
        for axis in "xyz":
            index = "xyz".index(axis)
            ticks = [a for a in anchors if a.kind == "tick" and a.axis == axis]
            title = next(a for a in anchors if a.kind == "title" and a.axis == axis)
            # The axis a label runs *along* is the one it is centred on; the other two
            # carry the outward offset that has to grow.
            offset_axes = [i for i in range(3) if i != index]
            strictly_further = False
            for other in offset_axes:
                edge = ticks[0].position[other]
                spread = [abs(a.position[other] - edge) for a in ticks]
                assert max(spread) < 1e-9, "ticks share the edge coordinate"
                if abs(title.position[other]) > abs(edge) + 1e-9:
                    strictly_further = True
            assert strictly_further, f"the {axis} title does not clear its numbers"

    def test_projection_returns_window_pixels(self):
        anchors = self._anchors()
        projected = axes3d.project_anchors(anchors, Camera3D(), BOX, 800.0, 600.0)
        assert projected
        for anchor, (x_px, y_px), depth in projected:
            assert np.isfinite(x_px) and np.isfinite(y_px)
            assert np.isfinite(depth)

    def test_projection_drops_anchors_behind_the_eye(self):
        """A naive divide by a negative w mirrors a point to the wrong side of the screen."""
        camera = Camera3D(distance=0.05)  # inside the box: much of it is behind the eye
        anchors = self._anchors()
        projected = axes3d.project_anchors(anchors, camera, BOX, 800.0, 600.0)
        assert len(projected) < len(anchors)

    def test_projection_of_nothing_is_empty(self):
        assert axes3d.project_anchors([], Camera3D(), BOX, 800.0, 600.0) == []

    def test_anchor_equality_and_repr(self):
        one = axes3d.LabelAnchor((1.0, 2.0, 3.0), "5", "x", "tick")
        same = axes3d.LabelAnchor((1.0, 2.0, 3.0), "5", "x", "tick")
        other = axes3d.LabelAnchor((1.0, 2.0, 3.0), "6", "x", "tick")
        assert one == same and one != other
        assert "tick" in repr(one)


class TestSceneRadius:
    def test_matches_the_camera_helper(self):
        assert axes3d.scene_radius(UNIT) == pytest.approx(float(np.linalg.norm(np.ones(3))) * 0.5)

    def test_none_is_a_unit_radius(self):
        assert axes3d.scene_radius(None) == pytest.approx(1.0)


@pytest.fixture
def imgui_frame():
    """A headless imgui context with an open frame, torn down afterwards.

    imgui's context is global state, so the teardown keeps the suite order-independent
    (see ``tests/test_gui_3d_panels.py`` for the failure that taught us this).
    """
    imgui = pytest.importorskip("imgui")
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = 900, 700
    io.fonts.get_tex_data_as_rgba32()
    io.fonts.texture_id = 1
    io.delta_time = 1 / 60.0
    imgui.new_frame()
    yield imgui
    imgui.render()
    imgui.destroy_context(ctx)


class _FakeEngine:
    """The five attributes ``draw_labels`` reads off an engine."""

    def __init__(self, anchors, width=900, height=700):
        self._axes3d_labels = anchors
        self._axes3d_bounds = BOX
        self.camera3d = Camera3D()
        self.width = width
        self.height = height
        self._panel_offset_px = (0.0, 0.0)
        self.options = None


class TestDrawLabelsReally:
    """Actually paint the labels, inside a real imgui frame.

    This is the test that was missing. The early-return cases below never reached an
    ``add_text`` call, so a wrong pyimgui signature in the drawing path — the one thing
    that can only be checked by calling it — shipped and crashed the live window on the
    first 3D plot that had an axis title.
    """

    def test_it_draws_tick_numbers(self, imgui_frame):
        ticks = axes3d.ticks_for_bounds(BOX, 5)
        anchors = axes3d.label_anchors(BOX, ticks, Camera3D(), 1.3)
        assert axes3d.draw_labels(_FakeEngine(anchors)) > 0

    def test_it_draws_axis_titles(self, imgui_frame):
        """The crashing case: a title takes the scaled-text path, a number does not."""
        ticks = axes3d.ticks_for_bounds(BOX, 5)
        anchors = axes3d.label_anchors(
            BOX, ticks, Camera3D(), 1.3, tick_labels=False, axis_labels=("X", "Y", "Z")
        )
        assert [a.kind for a in anchors] == ["title"] * 3
        assert axes3d.draw_labels(_FakeEngine(anchors)) == 3

    def test_it_draws_both_together(self, imgui_frame):
        ticks = axes3d.ticks_for_bounds(BOX, 5)
        anchors = axes3d.label_anchors(BOX, ticks, Camera3D(), 1.3, axis_labels=("X", "Y", "Z"))
        assert axes3d.draw_labels(_FakeEngine(anchors)) == len(anchors)

    def test_an_empty_string_is_skipped_not_drawn(self, imgui_frame):
        anchors = [
            axes3d.LabelAnchor((0.0, 0.0, 5.0), "", "x", "tick"),
            axes3d.LabelAnchor((0.0, 0.0, 5.0), "7", "x", "tick"),
        ]
        assert axes3d.draw_labels(_FakeEngine(anchors)) == 1

    def test_offscreen_labels_are_skipped(self, imgui_frame):
        """Clamping them to the edge would pile them into an unreadable stack."""
        far_away = [axes3d.LabelAnchor((1e6, 1e6, 1e6), "far", "x", "tick")]
        assert axes3d.draw_labels(_FakeEngine(far_away)) == 0

    def test_scaled_text_reports_success(self, imgui_frame):
        """The title path's helper must apply the scale, not silently fall back."""
        from glplot.renderers.axis import draw_text_scaled

        draw_list = imgui_frame.get_background_draw_list()
        color = imgui_frame.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0)
        assert draw_text_scaled(imgui_frame, draw_list, "Z axis", color, (40.0, 40.0), 1.25)

    def test_scale_of_one_is_a_plain_draw(self, imgui_frame):
        from glplot.renderers.axis import draw_text_scaled

        draw_list = imgui_frame.get_background_draw_list()
        color = imgui_frame.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0)
        before = draw_list.vtx_buffer_size
        assert draw_text_scaled(imgui_frame, draw_list, "x", color, (10.0, 10.0), 1.0)
        assert draw_list.vtx_buffer_size > before

    def test_empty_text_scales_to_nothing(self, imgui_frame):
        from glplot.renderers.axis import draw_text_scaled

        draw_list = imgui_frame.get_background_draw_list()
        color = imgui_frame.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0)
        assert draw_text_scaled(imgui_frame, draw_list, "", color, (10.0, 10.0), 2.0)


class TestScreenSpacePlacement:
    """The layout arithmetic ``draw_labels`` runs once the text has been measured.

    These are the rules that replaced "project the world offset and centre the text on it",
    which spent a *world* distance and so put the numbers a different distance off the box
    at every camera angle.
    """

    def test_a_box_starts_exactly_the_asked_for_distance_from_the_origin(self):
        """Whatever the direction, the box's nearest side is ``distance`` away."""
        root_half = float(np.sqrt(0.5))
        for ux, uy in [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.6, -0.8), (-root_half, root_half)]:
            (px, py), _ = axes3d._place_box((100.0, 100.0), (ux, uy), (40.0, 14.0), 9.0)
            corners = np.array([[px, py], [px + 40.0, py], [px, py + 14.0], [px + 40.0, py + 14.0]])
            # Distance along the direction, which is what the gap is measured in.
            along = (corners - np.array([100.0, 100.0])) @ np.array([ux, uy])
            assert along.min() == pytest.approx(9.0, abs=1e-6)

    def test_reach_is_the_far_side_of_the_placed_box(self):
        """What the next label out — an axis title — has to clear."""
        (px, py), reach = axes3d._place_box((0.0, 0.0), (1.0, 0.0), (40.0, 14.0), 9.0)
        assert reach == pytest.approx(px + 40.0)

    def test_the_gap_does_not_depend_on_the_text(self):
        near = [
            axes3d._place_box((0.0, 0.0), (0.0, 1.0), (w, 14.0), 7.0)[0][1] for w in (10.0, 200.0)
        ]
        assert near[0] == pytest.approx(near[1])

    def test_direction_is_away_from_the_pivot(self):
        dirs = axes3d._outward_screen_dirs(
            np.array([[10.0, 0.0], [0.0, -5.0]]),
            np.array([[0.0, 0.0], [0.0, 0.0]]),
            np.array([True, True]),
            np.array([100.0, 100.0]),
        )
        assert np.allclose(dirs, [[1.0, 0.0], [0.0, -1.0]])

    def test_a_degenerate_offset_falls_back_to_the_box_centre(self):
        """An axis pointing at the camera projects its outward step to nothing."""
        dirs = axes3d._outward_screen_dirs(
            np.array([[50.0, 100.0]]),
            np.array([[50.0, 100.0]]),  # identical: no direction of its own
            np.array([True]),
            np.array([100.0, 100.0]),  # centre is to the right, so "out" is to the left
        )
        assert np.allclose(dirs, [[-1.0, 0.0]])

    def test_a_label_on_the_box_centre_steps_down(self):
        dirs = axes3d._outward_screen_dirs(
            np.array([[100.0, 100.0]]),
            np.array([[100.0, 100.0]]),
            np.array([True]),
            np.array([100.0, 100.0]),
        )
        assert np.allclose(dirs, [[0.0, 1.0]])

    def test_a_pivot_behind_the_eye_is_not_used_as_a_direction(self):
        dirs = axes3d._outward_screen_dirs(
            np.array([[50.0, 100.0]]),
            np.array([[9e9, 9e9]]),  # nonsense, and flagged as such
            np.array([False]),
            np.array([100.0, 100.0]),
        )
        assert np.allclose(dirs, [[-1.0, 0.0]])

    def test_overlap_ignores_a_graze(self):
        box = (0.0, 0.0, 20.0, 10.0)
        assert not axes3d._overlaps(box, [(19.0, 0.0, 40.0, 10.0)])
        assert axes3d._overlaps(box, [(5.0, 0.0, 40.0, 10.0)])

    def test_overlap_of_nothing_is_false(self):
        assert not axes3d._overlaps((0.0, 0.0, 1.0, 1.0), [])


class TestAnchorPivots:
    """Every anchor knows the point on the box it is laid out from."""

    def _anchors(self, camera=None):
        ticks = axes3d.ticks_for_bounds(BOX, 5)
        return axes3d.label_anchors(
            BOX, ticks, camera or Camera3D(), 1.6, axis_labels=("X", "Y", "Z")
        )

    def test_every_anchor_has_one(self):
        assert all(a.pivot is not None for a in self._anchors())

    def test_the_pivot_is_inside_the_position(self):
        """The label steps outward *from* the pivot, so the pivot is the nearer point."""
        centre = np.array([0.0, 1.0, 5.0])  # BOX's centre
        for anchor in self._anchors():
            pivot = np.linalg.norm(np.asarray(anchor.pivot) - centre)
            position = np.linalg.norm(np.asarray(anchor.position) - centre)
            assert pivot < position

    def test_a_title_shares_the_offset_surface_of_its_numbers(self):
        """Both are measured from the tick-mark tips, or a title lands among the numbers."""
        anchors = self._anchors()
        for axis in "xyz":
            index = "xyz".index(axis)
            ticks = [a for a in anchors if a.kind == "tick" and a.axis == axis]
            title = next(a for a in anchors if a.kind == "title" and a.axis == axis)
            for other in (i for i in range(3) if i != index):
                assert title.pivot[other] == pytest.approx(ticks[0].pivot[other])

    def test_the_pivot_clears_the_tick_mark(self):
        """It is the mark's *tip*: a number placed at the root would sit on its own mark."""
        ticks = axes3d.ticks_for_bounds(BOX, 5)
        camera = Camera3D()
        marks = axes3d.tick_marks(BOX, ticks, camera, 1.6).reshape(-1, 2, 3)
        tips = np.asarray([m[1] for m in marks], dtype=np.float64)
        anchors = [a for a in axes3d.label_anchors(BOX, ticks, camera, 1.6) if a.kind == "tick"]
        assert anchors
        for anchor in anchors:
            distance = np.linalg.norm(tips - np.asarray(anchor.pivot), axis=1)
            assert distance.min() < 1e-5, f"{anchor.text} is not on a mark tip"

    def test_equality_still_ignores_the_pivot(self):
        one = axes3d.LabelAnchor((1.0, 2.0, 3.0), "5", "x", "tick", (1.0, 2.0, 2.0))
        same = axes3d.LabelAnchor((1.0, 2.0, 3.0), "5", "x", "tick")
        assert one == same


class TestProjectPoints:
    def test_returns_pixels_and_a_validity_mask(self):
        pts = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
        px, ok = axes3d.project_points(pts, Camera3D(), BOX, 800.0, 600.0)
        assert px.shape == (2, 2) and ok.shape == (2,)
        assert ok.all() and np.isfinite(px).all()

    def test_nothing_projects_to_nothing(self):
        px, ok = axes3d.project_points([], Camera3D(), BOX, 800.0, 600.0)
        assert len(px) == 0 and len(ok) == 0

    def test_points_behind_the_eye_are_flagged_not_mirrored(self):
        camera = Camera3D(distance=0.05)  # inside the box
        px, ok = axes3d.project_points(_corners_of(BOX), camera, BOX, 800.0, 600.0)
        assert not ok.all()

    def test_agrees_with_project_anchors(self):
        anchors = [axes3d.LabelAnchor((0.0, 1.0, 5.0), "0", "x", "tick")]
        projected = axes3d.project_anchors(anchors, Camera3D(), BOX, 800.0, 600.0)
        px, ok = axes3d.project_points([anchors[0].position], Camera3D(), BOX, 800.0, 600.0)
        assert ok[0]
        assert projected[0][1] == pytest.approx(tuple(px[0]))


def _corners_of(bounds):
    return [(x, y, z) for x in bounds[0:2] for y in bounds[2:4] for z in bounds[4:6]]


class TestLabelLayoutInAFrame:
    """Placement rules that can only be checked once imgui has measured the text."""

    def _engine(self, camera, width=900, height=700, labels=("X", "Y", "Z")):
        ticks = axes3d.ticks_for_bounds(BOX, 5)
        anchors = axes3d.label_anchors(BOX, ticks, camera, width / height, axis_labels=labels)
        engine = _FakeEngine(anchors, width, height)
        engine.camera3d = camera
        return engine, anchors

    def test_titles_are_never_culled(self, imgui_frame):
        """An axis with no name is worse than a title that grazes a number."""
        engine, anchors = self._engine(Camera3D(elev=0.0, azim=0.0))
        drawn = axes3d.draw_labels(engine)
        titles = [a for a in anchors if a.kind == "title"]
        assert drawn >= len(titles)

    def test_numbers_that_land_on_each_other_are_dropped(self, imgui_frame):
        """An axis seen end-on packs its ticks into one point; drawing them all is a smear."""
        stacked = [
            axes3d.LabelAnchor((0.0, 1.0, 5.0), "0", "x", "tick", (0.0, 0.9, 5.0)) for _ in range(4)
        ]
        assert axes3d.draw_labels(_FakeEngine(stacked)) == 1

    def test_two_axes_may_share_a_corner(self, imgui_frame):
        """Culling is per axis: deleting one of these would truncate an axis' range."""
        position, pivot = (0.0, 1.0, 5.0), (0.0, 0.9, 5.0)
        pair = [
            axes3d.LabelAnchor(position, "0", "x", "tick", pivot),
            axes3d.LabelAnchor(position, "0", "z", "tick", pivot),
        ]
        assert axes3d.draw_labels(_FakeEngine(pair)) == 2

    def test_a_normal_view_keeps_every_number(self, imgui_frame):
        engine, anchors = self._engine(Camera3D())
        assert axes3d.draw_labels(engine) == len(anchors)


class TestDrawLabels:
    """The early-return paths. They must degrade, never raise."""

    def test_returns_zero_without_an_engine_state(self):
        class Bare:
            pass

        assert axes3d.draw_labels(Bare()) == 0

    def test_returns_zero_with_no_anchors(self):
        class Empty:
            _axes3d_labels = []
            _axes3d_bounds = BOX
            camera3d = Camera3D()
            width = 800
            height = 600

        assert axes3d.draw_labels(Empty()) == 0

    def test_returns_zero_on_a_zero_sized_viewport(self):
        class Collapsed:
            _axes3d_labels = [axes3d.LabelAnchor((0.0, 0.0, 0.0), "0", "x", "tick")]
            _axes3d_bounds = BOX
            camera3d = Camera3D()
            width = 0
            height = 0

        assert axes3d.draw_labels(Collapsed()) == 0
