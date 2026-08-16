"""Test the animation verbs: the keyframes they write, and what the timeline reads back.

Two assertions per verb, and they are not the same assertion:

1. **The keyframes** — times, values and easing. This is the verb's contract with a
   keyframe editor, an exporter and a serialiser, all of which read the track rather than
   evaluating it.
2. **``timeline.evaluate`` at the start, the middle and the end** — the verb's contract
   with the eye. A verb can write plausible keyframes and still animate wrongly (see
   :class:`Rotate`, whose whole reason for baking is that the obvious two keyframes
   evaluate to a shape collapsing through its own pivot).

Pure numpy: nothing here constructs an engine, opens a window or touches GL, exactly like
``tests/test_camera3d.py`` and for the same reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.anim.primitives import (
    DEFAULT_LAG_RATIO,
    INHERIT,
    UNSET,
    Animation,
    CameraMoveTo,
    ColorMapTo,
    ColorTo,
    Create,
    Delta,
    FadeIn,
    FadeOut,
    Hide,
    Morph,
    MoveTo,
    Orbit,
    Play,
    PropertySpan,
    Rotate,
    ScaleTo,
    Set,
    Shift,
    Show,
    Transform,
    Uncreate,
    Unwrite,
    Wait,
    Write,
    ZoomTo,
    baked_span,
    eased_fractions,
    group,
    lagged,
    match_vertex_counts,
    parallel,
    resample_vertices,
    rotate_vertices,
    scale_vertices,
    span,
    stagger,
    succession,
)
from glplot.core.timeline import STEP, Timeline


@pytest.fixture
def tl():
    """A timeline long enough that ``fit_duration`` never has to grow it mid-test."""
    return Timeline(duration=10.0)


def _times(track):
    return [k.time for k in track.keyframes]


def _values(track):
    return [k.value for k in track.keyframes]


def _easings(track):
    return [k.easing for k in track.keyframes]


def _square():
    """A unit square centred on the origin, as a closed (5, 2) path."""
    return np.array(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]], dtype=np.float32
    )


# ----------------------------------------------------------------------------------
# The shared machinery
# ----------------------------------------------------------------------------------


class TestApplyMechanics:
    """Everything every verb inherits: timing, easing, overrides, inheritance."""

    def test_keys_land_at_start_and_start_plus_duration(self, tl):
        (track,) = FadeIn().apply(tl, 1, 2.0, 3.0)
        assert _times(track) == [2.0, 5.0]

    def test_the_track_is_identified_by_target_and_prop(self, tl):
        (track,) = FadeIn().apply(tl, 42, 0.0, 1.0)
        assert track.key() == (42, "alpha")
        assert tl.track(42, "alpha", create=False) is track

    def test_the_easing_is_written_onto_the_keys(self, tl):
        (track,) = FadeIn(easing="bounce").apply(tl, 1, 0.0, 1.0)
        assert _easings(track) == ["bounce", "bounce"]

    def test_overrides_are_field_replacements_for_one_application(self, tl):
        verb = FadeIn()
        (track,) = verb.apply(tl, 1, 0.0, 1.0, easing="linear", to=0.25)
        assert _easings(track)[0] == "linear"
        assert _values(track)[-1] == 0.25
        # The record itself is untouched: it is a value, not a builder.
        assert verb.easing == "smooth" and verb.to == 1.0

    def test_an_unknown_override_raises_rather_than_being_ignored(self, tl):
        with pytest.raises(TypeError):
            FadeIn().apply(tl, 1, 0.0, 1.0, esaing="linear")

    def test_zero_duration_collapses_to_a_snap(self, tl):
        """Both keys land on one instant; Track.add replaces, so the destination wins."""
        (track,) = FadeIn().apply(tl, 1, 3.0, 0.0)
        assert _times(track) == [3.0]
        assert _values(track) == [1.0]

    def test_a_negative_duration_is_clamped_not_reversed(self, tl):
        (track,) = FadeIn().apply(tl, 1, 3.0, -2.0)
        assert _times(track) == [3.0]

    def test_apply_returns_the_tracks_it_touched(self, tl):
        tracks = CameraMoveTo(elev=10.0, azim=20.0).apply(tl, "camera3d", 0.0, 1.0)
        assert [t.prop for t in tracks] == ["elev", "azim"]

    def test_fit_grows_the_timeline_to_cover_the_keys(self):
        timeline = Timeline(duration=1.0)
        FadeIn().apply(timeline, 1, 0.0, 8.0)
        assert timeline.duration == pytest.approx(8.0)

    def test_fit_false_leaves_the_duration_alone(self):
        timeline = Timeline(duration=1.0)
        FadeIn().apply(timeline, 1, 0.0, 8.0, fit=False)
        assert timeline.duration == pytest.approx(1.0)

    def test_inherit_takes_the_start_from_the_existing_track(self, tl):
        FadeIn(to=0.4).apply(tl, 1, 0.0, 1.0)
        FadeOut(to=0.0, start_alpha=INHERIT).apply(tl, 1, 1.0, 1.0)
        track = tl.track(1, "alpha", create=False)
        assert _values(track) == [0.0, 0.4, 0.0]
        assert tl.evaluate(1.0)[(1, "alpha")] == pytest.approx(0.4)

    def test_inherit_against_an_empty_track_writes_only_the_destination(self, tl):
        """Documented degradation: nothing to inherit, so the property snaps and holds."""
        (track,) = ColorTo(to=(1, 0, 0, 1)).apply(tl, 1, 2.0, 2.0)
        assert _times(track) == [4.0]
        assert tl.evaluate(0.0)[(1, "color")] == (1.0, 0.0, 0.0, 1.0)

    def test_props_lists_what_a_verb_writes(self):
        assert CameraMoveTo(elev=1.0, distance=2.0).props() == ("elev", "distance")

    def test_the_base_class_refuses_to_be_used_directly(self, tl):
        with pytest.raises(NotImplementedError):
            Animation().apply(tl, 1, 0.0, 1.0)


class TestDelta:
    """Relative values, resolved against whatever the property already holds."""

    def test_scalars_add(self):
        assert Delta(5.0).resolve(2.0) == pytest.approx(7.0)

    def test_a_missing_base_falls_back_to_the_default(self):
        assert Delta(5.0, default=1.0).resolve(None) == pytest.approx(6.0)

    def test_sequences_add_elementwise_and_keep_their_type(self):
        assert Delta((1.0, 2.0)).resolve((10.0, 20.0)) == (11.0, 22.0)

    def test_a_scalar_amount_broadcasts_over_a_sequence_base(self):
        assert Delta(1.0).resolve((10.0, 20.0)) == (11.0, 21.0)

    def test_arrays_add(self):
        out = Delta(np.ones(3)).resolve(np.arange(3.0))
        assert np.allclose(out, [1.0, 2.0, 3.0])


class TestSpanHelpers:
    def test_span_is_two_keys_at_the_ends(self):
        s = span("alpha", 0.0, 1.0)
        assert s.keys == ((0.0, 0.0), (1.0, 1.0))

    def test_baked_span_spreads_uniformly_and_is_linear(self):
        s = baked_span("x", [0, 1, 2, 3])
        assert [f for f, _ in s.keys] == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
        assert s.easing == "linear"

    def test_baked_span_needs_two_values(self):
        with pytest.raises(ValueError, match="at least two"):
            baked_span("x", [1])

    def test_eased_fractions_start_at_zero_and_end_at_one(self):
        out = eased_fractions("smooth", 4)
        assert len(out) == 5
        assert out[0] == pytest.approx(0.0) and out[-1] == pytest.approx(1.0)

    def test_eased_fractions_are_uniform_for_linear(self):
        assert eased_fractions("linear", 4) == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])

    def test_eased_fractions_are_not_uniform_for_smooth(self):
        out = eased_fractions("smooth", 4)
        assert out[1] < 0.25 and out[3] > 0.75


# ----------------------------------------------------------------------------------
# Opacity and visibility
# ----------------------------------------------------------------------------------


class TestFades:
    def test_fade_in_goes_zero_to_one(self, tl):
        (track,) = FadeIn().apply(tl, 1, 0.0, 2.0)
        assert _values(track) == [0.0, 1.0]
        assert tl.evaluate(0.0)[(1, "alpha")] == pytest.approx(0.0)
        assert tl.evaluate(1.0)[(1, "alpha")] == pytest.approx(0.5)  # smooth is symmetric
        assert tl.evaluate(2.0)[(1, "alpha")] == pytest.approx(1.0)

    def test_fade_out_starts_at_the_layer_style_default(self, tl):
        (track,) = FadeOut().apply(tl, 1, 0.0, 2.0)
        assert _values(track) == [1.0, 0.0]
        assert tl.evaluate(2.0)[(1, "alpha")] == pytest.approx(0.0)

    def test_a_fade_can_be_pointed_at_another_alpha(self, tl):
        (track,) = FadeIn(prop="outline_alpha").apply(tl, 1, 0.0, 1.0)
        assert track.key() == (1, "outline_alpha")

    def test_partial_fades_respect_their_endpoints(self, tl):
        FadeIn(to=0.3, start_alpha=0.1).apply(tl, 1, 0.0, 1.0)
        assert tl.evaluate(0.5)[(1, "alpha")] == pytest.approx(0.2)


class TestVisibility:
    def test_show_writes_one_held_key_at_the_start(self, tl):
        (track,) = Show().apply(tl, 1, 4.0, 2.0)
        assert _times(track) == [4.0]
        assert _values(track) == [True]
        assert track.interpolate is False
        assert _easings(track) == [STEP]

    def test_hide_is_show_inverted(self, tl):
        (track,) = Hide().apply(tl, 1, 0.0, 1.0)
        assert _values(track) == [False]

    def test_a_boolean_never_blends(self, tl):
        Hide().apply(tl, 1, 0.0, 1.0)
        Show().apply(tl, 1, 2.0, 1.0)
        assert tl.evaluate(1.0)[(1, "visible")] is False
        assert tl.evaluate(2.0)[(1, "visible")] is True

    def test_set_is_the_escape_hatch_for_unblendable_values(self, tl):
        (track,) = Set("projection", "orthographic").apply(tl, "camera3d", 1.0, 2.0)
        assert _times(track) == [1.0]
        assert _values(track) == ["orthographic"]
        assert track.interpolate is False

    def test_set_needs_a_prop(self, tl):
        with pytest.raises(ValueError, match="prop"):
            Set(value=1).apply(tl, 1, 0.0, 1.0)


# ----------------------------------------------------------------------------------
# Progressive reveal
# ----------------------------------------------------------------------------------


class TestReveal:
    def test_create_keys_the_draw_fraction_zero_to_one(self, tl):
        (track,) = Create().apply(tl, 1, 0.0, 2.0)
        assert track.key() == (1, "draw_fraction")
        assert _values(track) == [0.0, 1.0]
        assert tl.evaluate(2.0)[(1, "draw_fraction")] == pytest.approx(1.0)

    def test_uncreate_runs_the_other_way(self, tl):
        (track,) = Uncreate().apply(tl, 1, 0.0, 2.0)
        assert _values(track) == [1.0, 0.0]

    def test_write_keys_the_text_fraction_and_is_linear(self, tl):
        (track,) = Write().apply(tl, 1, 0.0, 2.0)
        assert track.key() == (1, "text_fraction")
        assert _easings(track) == ["linear", "linear"]
        assert tl.evaluate(1.0)[(1, "text_fraction")] == pytest.approx(0.5)

    def test_unwrite_runs_the_other_way(self, tl):
        (track,) = Unwrite().apply(tl, 1, 0.0, 2.0)
        assert _values(track) == [1.0, 0.0]


# ----------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------


class TestResampling:
    def test_a_matching_count_is_returned_untouched(self):
        arr = _square()
        assert np.allclose(resample_vertices(arr, len(arr)), arr)

    def test_upsampling_hits_both_endpoints_exactly(self):
        arr = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32)
        out = resample_vertices(arr, 5)
        assert out.shape == (5, 2)
        assert np.allclose(out[0], arr[0]) and np.allclose(out[-1], arr[-1])
        assert np.allclose(out[2], [0.5, 1.0])

    def test_downsampling_also_hits_both_endpoints(self):
        arr = np.column_stack([np.linspace(0, 1, 101), np.zeros(101)])
        out = resample_vertices(arr, 3)
        assert np.allclose(out[:, 0], [0.0, 0.5, 1.0], atol=1e-6)

    def test_a_single_vertex_is_repeated(self):
        out = resample_vertices([[1.0, 2.0]], 4)
        assert out.shape == (4, 2) and np.allclose(out, [1.0, 2.0])

    def test_an_empty_array_raises_rather_than_inventing_the_origin(self):
        with pytest.raises(ValueError, match="empty"):
            resample_vertices(np.zeros((0, 2)), 3)

    def test_match_takes_the_longer_count_and_leaves_it_alone(self):
        a = np.zeros((4, 2), dtype=np.float32)
        b = np.ones((17, 2), dtype=np.float32)
        ra, rb = match_vertex_counts(a, b)
        assert len(ra) == len(rb) == 17
        assert np.allclose(rb, b)

    def test_match_refuses_to_mix_dimensions(self):
        with pytest.raises(ValueError, match="2-D and 3-D"):
            match_vertex_counts(np.zeros((4, 2)), np.zeros((4, 3)))


class TestRotateVertices:
    def test_a_quarter_turn_about_the_origin(self):
        out = rotate_vertices([[1.0, 0.0]], 90.0, pivot=(0.0, 0.0))
        assert np.allclose(out, [[0.0, 1.0]], atol=1e-6)

    def test_the_default_pivot_is_the_centroid_so_a_shape_spins_in_place(self):
        arr = _square()
        out = rotate_vertices(arr, 37.0)
        assert np.allclose(out.mean(axis=0), arr.mean(axis=0), atol=1e-5)

    def test_a_full_turn_is_the_identity(self):
        arr = _square()
        assert np.allclose(rotate_vertices(arr, 360.0), arr, atol=1e-5)

    def test_3d_rotation_about_z_leaves_z_alone(self):
        out = rotate_vertices([[1.0, 0.0, 5.0]], 90.0, pivot=(0.0, 0.0, 0.0), axis="z")
        assert np.allclose(out, [[0.0, 1.0, 5.0]], atol=1e-6)

    def test_3d_rotation_about_x_leaves_x_alone(self):
        out = rotate_vertices([[5.0, 1.0, 0.0]], 90.0, pivot=(0.0, 0.0, 0.0), axis="x")
        assert np.allclose(out, [[5.0, 0.0, 1.0]], atol=1e-6)

    def test_an_unknown_axis_raises(self):
        with pytest.raises(ValueError, match="axis"):
            rotate_vertices(np.zeros((3, 3)), 10.0, axis="w")

    def test_scale_about_the_centroid_keeps_the_centroid(self):
        arr = _square()
        out = scale_vertices(arr, 3.0)
        assert np.allclose(out.mean(axis=0), arr.mean(axis=0), atol=1e-5)
        assert np.allclose(out - arr.mean(axis=0), 3.0 * (arr - arr.mean(axis=0)), atol=1e-5)

    def test_per_axis_scale(self):
        out = scale_vertices([[1.0, 1.0]], (2.0, 5.0), pivot=(0.0, 0.0))
        assert np.allclose(out, [[2.0, 5.0]])


class TestTransform:
    def test_equal_counts_are_stored_verbatim(self, tl):
        a = _square()
        b = a * 2.0
        (track,) = Transform(a, b).apply(tl, 1, 0.0, 2.0)
        assert np.allclose(_values(track)[0], a)
        assert np.allclose(_values(track)[1], b)

    def test_the_midpoint_is_the_average_of_the_endpoints(self, tl):
        a = _square()
        b = a * 3.0
        Transform(a, b, easing="linear").apply(tl, 1, 0.0, 2.0)
        mid = tl.evaluate(1.0)[(1, "pts")]
        assert np.allclose(mid, 0.5 * (a + b), atol=1e-5)

    def test_a_vertex_count_mismatch_is_resampled_to_the_longer_side(self, tl):
        """The documented policy: max(len(a), len(b)), both sides put on that grid."""
        a = np.zeros((4, 2), dtype=np.float32)
        b = np.ones((11, 2), dtype=np.float32)
        (track,) = Transform(a, b).apply(tl, 1, 0.0, 1.0)
        assert _values(track)[0].shape == (11, 2)
        assert _values(track)[1].shape == (11, 2)
        assert np.allclose(_values(track)[1], b)

    def test_the_resampled_morph_actually_moves(self, tl):
        a = np.zeros((4, 2), dtype=np.float32)
        b = np.ones((11, 2), dtype=np.float32)
        Transform(a, b, easing="linear").apply(tl, 1, 0.0, 2.0)
        assert np.allclose(tl.evaluate(1.0)[(1, "pts")], 0.5, atol=1e-6)

    def test_resample_false_leaves_a_mismatch_to_be_held(self, tl):
        """The 'do not invent vertices' option. The timeline then cannot blend, so it holds."""
        a = np.zeros((4, 2), dtype=np.float32)
        b = np.ones((11, 2), dtype=np.float32)
        Transform(a, b, resample=False, easing="linear").apply(tl, 1, 0.0, 2.0)
        assert np.allclose(tl.evaluate(1.0)[(1, "pts")], a)
        assert np.allclose(tl.evaluate(2.0)[(1, "pts")], b)

    def test_a_missing_endpoint_raises_at_apply_time(self, tl):
        with pytest.raises(ValueError, match="source and a dest"):
            Transform(source=_square()).apply(tl, 1, 0.0, 1.0)

    def test_2d_and_3d_cannot_be_morphed_into_each_other(self, tl):
        with pytest.raises(ValueError, match="2-D and 3-D"):
            Transform(np.zeros((4, 2)), np.zeros((4, 3))).apply(tl, 1, 0.0, 1.0)

    def test_from_layers_snapshots_and_copies(self, tl):
        class _FakeLayer:
            def __init__(self, pts):
                self.pts = pts

        source = _FakeLayer(np.zeros((3, 2), dtype=np.float32))
        dest = _FakeLayer(np.ones((3, 2), dtype=np.float32))
        verb = Transform.from_layers(source, dest)
        source.pts[:] = 99.0  # the record must not follow the live layer
        assert np.allclose(verb.source, 0.0)

    def test_from_layers_reads_vertices_when_there_is_no_pts(self):
        class _Mesh:
            def __init__(self):
                self.vertices = np.zeros((3, 3), dtype=np.float32)

        verb = Transform.from_layers(_Mesh(), _Mesh())
        assert verb.source.shape == (3, 3)

    def test_from_layers_refuses_a_layer_with_no_geometry(self):
        class _Text:
            text = "hi"

        with pytest.raises(ValueError, match="no pts/vertices"):
            Transform.from_layers(_Text(), _Text())

    def test_morph_is_the_same_verb(self):
        assert Morph is Transform


class TestRotateVerb:
    def test_it_bakes_one_key_per_fifteen_degrees(self, tl):
        (track,) = Rotate(_square(), 90.0).apply(tl, 1, 0.0, 1.0)
        assert len(track.keyframes) == 7  # ceil(90/15) = 6 steps -> 7 keys

    def test_every_baked_key_is_linear(self, tl):
        (track,) = Rotate(_square(), 90.0).apply(tl, 1, 0.0, 1.0)
        assert set(_easings(track)) == {"linear"}

    def test_the_keys_are_uniformly_spaced_in_time(self, tl):
        (track,) = Rotate(_square(), 90.0, steps=4).apply(tl, 1, 0.0, 2.0)
        assert _times(track) == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])

    def test_the_midpoint_of_a_180_turn_is_rotated_not_collapsed(self, tl):
        """The whole reason Rotate bakes. Two keyframes would put the midpoint at the pivot."""
        arr = _square()
        Rotate(arr, 180.0, easing="linear").apply(tl, 1, 0.0, 2.0)
        mid = tl.evaluate(1.0)[(1, "pts")]
        centre = arr.mean(axis=0)
        radius = np.linalg.norm(arr - centre, axis=1)
        assert np.allclose(np.linalg.norm(mid - centre, axis=1), radius, atol=0.05)
        assert np.allclose(mid, rotate_vertices(arr, 90.0), atol=0.05)

    def test_a_two_key_rotation_would_have_collapsed(self, tl):
        """The counter-example, asserted so the design note cannot rot."""
        arr = _square()
        Rotate(arr, 180.0, steps=1, easing="linear").apply(tl, 1, 0.0, 2.0)
        mid = tl.evaluate(1.0)[(1, "pts")]
        centre = arr.mean(axis=0)
        assert np.allclose(mid, centre, atol=1e-5)

    def test_the_end_value_is_the_full_rotation(self, tl):
        arr = _square()
        Rotate(arr, 90.0).apply(tl, 1, 0.0, 1.0)
        assert np.allclose(tl.evaluate(1.0)[(1, "pts")], rotate_vertices(arr, 90.0), atol=1e-5)

    def test_3d_rotation_keeps_the_third_column(self, tl):
        arr = np.array([[1.0, 0.0, 7.0], [0.0, 1.0, 7.0]], dtype=np.float32)
        Rotate(arr, 90.0, axis="z").apply(tl, 1, 0.0, 1.0)
        assert np.allclose(tl.evaluate(1.0)[(1, "pts")][:, 2], 7.0, atol=1e-5)

    def test_no_vertices_raises(self, tl):
        with pytest.raises(ValueError, match="vertex array"):
            Rotate(angle=90.0).apply(tl, 1, 0.0, 1.0)

    def test_from_layer_snapshots(self):
        class _L:
            pts = np.zeros((4, 2), dtype=np.float32)

        assert Rotate.from_layer(_L(), 45.0).angle == 45.0


class TestScaleTo:
    def test_two_keys_are_enough_because_scaling_is_linear(self, tl):
        arr = _square()
        (track,) = ScaleTo(arr, 3.0).apply(tl, 1, 0.0, 2.0)
        assert len(track.keyframes) == 2

    def test_the_midpoint_is_the_halfway_scale(self, tl):
        arr = _square()
        ScaleTo(arr, 3.0, easing="linear").apply(tl, 1, 0.0, 2.0)
        assert np.allclose(tl.evaluate(1.0)[(1, "pts")], scale_vertices(arr, 2.0), atol=1e-5)

    def test_no_vertices_raises(self, tl):
        with pytest.raises(ValueError, match="vertex array"):
            ScaleTo(factor=2.0).apply(tl, 1, 0.0, 1.0)


# ----------------------------------------------------------------------------------
# Placement and colour
# ----------------------------------------------------------------------------------


class TestPlacement:
    def test_move_to_is_absolute_from_the_origin(self, tl):
        (track,) = MoveTo((3.0, 4.0)).apply(tl, 1, 0.0, 2.0)
        assert track.key() == (1, "translation")
        assert _values(track) == [(0.0, 0.0), (3.0, 4.0)]
        assert tl.evaluate(1.0)[(1, "translation")] == pytest.approx((1.5, 2.0))

    def test_shift_is_relative_to_its_start(self, tl):
        (track,) = Shift((1.0, 0.0), start_at=(10.0, 10.0)).apply(tl, 1, 0.0, 1.0)
        assert _values(track) == [(10.0, 10.0), (11.0, 10.0)]

    def test_shift_chains_off_a_previous_verb(self, tl):
        MoveTo((5.0, 0.0)).apply(tl, 1, 0.0, 1.0)
        Shift((2.0, 1.0), start_at=INHERIT).apply(tl, 1, 1.0, 1.0)
        assert tl.evaluate(2.0)[(1, "translation")] == pytest.approx((7.0, 1.0))

    def test_shift_on_an_empty_track_falls_back_to_the_origin(self, tl):
        (track,) = Shift((2.0, 1.0), start_at=INHERIT).apply(tl, 1, 0.0, 1.0)
        assert _values(track) == [(2.0, 1.0)]


class TestColour:
    def test_color_to_blends_rgba(self, tl):
        (track,) = ColorTo(to=(1, 0, 0), start_color=(0, 0, 1)).apply(tl, 1, 0.0, 2.0)
        assert _values(track) == [(0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 0.0, 1.0)]
        assert tl.evaluate(1.0)[(1, "color")] == pytest.approx((0.5, 0.0, 0.5, 1.0))

    def test_a_three_component_colour_gains_alpha_one(self, tl):
        (track,) = ColorTo(to=(0.2, 0.4, 0.6)).apply(tl, 1, 0.0, 1.0)
        assert _values(track)[-1] == pytest.approx((0.2, 0.4, 0.6, 1.0))

    def test_from_layer_captures_the_current_colour(self):
        class _L:
            class style:
                color = (0.1, 0.2, 0.3, 0.4)

        verb = ColorTo.from_layer(_L(), (1, 1, 1, 1))
        assert verb.start_color == pytest.approx((0.1, 0.2, 0.3, 0.4))

    def test_from_layer_falls_back_to_white_when_there_is_no_colour(self):
        class _L:
            class style:
                color = None

        assert ColorTo.from_layer(_L(), (0, 0, 0, 1)).start_color == (1.0, 1.0, 1.0, 1.0)

    def test_a_malformed_colour_raises(self, tl):
        with pytest.raises(ValueError, match="3 or 4"):
            ColorTo(to=(1.0, 2.0)).apply(tl, 1, 0.0, 1.0)

    def test_colormaps_are_held_never_blended(self, tl):
        (track,) = ColorMapTo("magma").apply(tl, 1, 2.0, 3.0)
        assert _times(track) == [2.0]
        assert track.interpolate is False
        assert tl.evaluate(4.0)[(1, "cmap")] == "magma"

    def test_two_colormap_switches_step_rather_than_mixing(self, tl):
        ColorMapTo("viridis").apply(tl, 1, 0.0, 1.0)
        ColorMapTo("magma").apply(tl, 1, 4.0, 1.0)
        assert tl.evaluate(2.0)[(1, "cmap")] == "viridis"
        assert tl.evaluate(4.0)[(1, "cmap")] == "magma"


# ----------------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------------


class TestCameraMoveTo:
    def test_only_the_named_props_are_keyed(self, tl):
        tracks = CameraMoveTo(elev=60.0).apply(tl, "camera3d", 0.0, 1.0)
        assert [t.prop for t in tracks] == ["elev"]
        assert tl.track("camera3d", "azim", create=False) is None

    def test_distance_none_is_a_real_pose_not_an_omission(self, tl):
        tracks = CameraMoveTo(distance=None).apply(tl, "camera3d", 0.0, 1.0)
        assert [t.prop for t in tracks] == ["distance"]
        assert _values(tracks[0]) == [None]

    def test_unset_really_omits(self):
        assert CameraMoveTo().spans() == ()
        assert CameraMoveTo(elev=UNSET).spans() == ()

    def test_from_camera_snapshots_only_the_animated_props(self, tl):
        class _Cam:
            elev, azim, roll, distance, fov = 28.0, -45.0, 0.0, None, 42.0
            pan = (0.0, 0.0, 0.0)
            box_aspect = None

        verb = CameraMoveTo.from_camera(_Cam(), azim=90.0)
        assert verb.start == {"azim": -45.0}
        (track,) = verb.apply(tl, "camera3d", 0.0, 2.0)
        assert _values(track) == [-45.0, 90.0]
        assert tl.evaluate(1.0)[("camera3d", "azim")] == pytest.approx(22.5)

    def test_from_camera_rejects_a_property_the_camera_does_not_have(self):
        class _Cam:
            elev = 1.0

        with pytest.raises(ValueError, match="unknown camera property"):
            CameraMoveTo.from_camera(_Cam(), zoom=2.0)

    def test_pan_is_a_three_vector_and_interpolates(self, tl):
        CameraMoveTo(pan=(2.0, 4.0, 6.0), start={"pan": (0.0, 0.0, 0.0)}, easing="linear").apply(
            tl, "camera3d", 0.0, 2.0
        )
        assert tl.evaluate(1.0)[("camera3d", "pan")] == pytest.approx((1.0, 2.0, 3.0))


class TestOrbit:
    def test_a_full_turn_ends_at_360_not_at_zero(self, tl):
        (track,) = Orbit(turns=1.0).apply(tl, "camera3d", 0.0, 4.0)
        assert _values(track)[0] == pytest.approx(0.0)
        assert _values(track)[-1] == pytest.approx(360.0)

    def test_two_turns_reach_720_so_the_sweep_never_unwinds(self, tl):
        (track,) = Orbit(turns=2.0).apply(tl, "camera3d", 0.0, 4.0)
        assert _values(track)[-1] == pytest.approx(720.0)
        assert _values(track) == sorted(_values(track))

    def test_it_is_linear_by_default(self, tl):
        (track,) = Orbit(turns=1.0).apply(tl, "camera3d", 0.0, 4.0)
        assert set(_easings(track)) == {"linear"}
        assert tl.evaluate(2.0)[("camera3d", "azim")] == pytest.approx(180.0)

    def test_it_is_relative_so_it_continues_from_the_track(self, tl):
        CameraMoveTo(azim=100.0, start={"azim": 100.0}).apply(tl, "camera3d", 0.0, 0.5)
        Orbit(turns=1.0, start_azim=INHERIT).apply(tl, "camera3d", 0.5, 4.0)
        assert tl.evaluate(4.5)[("camera3d", "azim")] == pytest.approx(460.0)

    def test_from_camera_starts_where_the_camera_is(self, tl):
        class _Cam:
            azim = 30.0

        (track,) = Orbit.from_camera(_Cam(), turns=1.0).apply(tl, "camera3d", 0.0, 1.0)
        assert _values(track)[0] == pytest.approx(30.0)
        assert _values(track)[-1] == pytest.approx(390.0)


class TestZoomTo:
    def test_distance_targets_the_3d_camera(self, tl):
        tracks = ZoomTo(distance=4.0).apply(tl, "camera3d", 0.0, 1.0)
        assert [t.prop for t in tracks] == ["distance"]

    def test_zoom_expands_to_both_axes(self, tl):
        tracks = ZoomTo(zoom=2.0).apply(tl, "camera", 0.0, 1.0)
        assert sorted(t.prop for t in tracks) == ["zoom_x", "zoom_y"]

    def test_an_explicit_axis_wins_over_the_shorthand(self, tl):
        spans = {s.prop: s.keys[-1][1] for s in ZoomTo(zoom=2.0, zoom_x=5.0).spans()}
        assert spans == {"zoom_x": 5.0, "zoom_y": 2.0}

    def test_centre_and_zoom_can_move_together(self, tl):
        tracks = ZoomTo(zoom=2.0, cx=1.0, cy=-1.0, start={"zoom_x": 1.0, "zoom_y": 1.0}).apply(
            tl, "camera", 0.0, 1.0
        )
        assert len(tracks) == 4

    def test_from_camera_snapshots_the_expanded_zoom(self):
        class _Cam:
            cx = cy = 0.0
            zoom_x = zoom_y = 3.0

        verb = ZoomTo.from_camera(_Cam(), zoom=6.0)
        assert verb.start == {"zoom_x": 3.0, "zoom_y": 3.0}


# ----------------------------------------------------------------------------------
# Sequencing
# ----------------------------------------------------------------------------------


class TestSequencing:
    """The lag-ratio rule, and the two names for its endpoints."""

    def test_succession_starts_each_play_when_the_last_one_ends(self, tl):
        end = succession(tl, (FadeIn(), 1, 2.0), (FadeIn(), 2, 3.0), (FadeIn(), 3, 1.0))
        assert _times(tl.track(1, "alpha", create=False)) == [0.0, 2.0]
        assert _times(tl.track(2, "alpha", create=False)) == [2.0, 5.0]
        assert _times(tl.track(3, "alpha", create=False)) == [5.0, 6.0]
        assert end == pytest.approx(6.0)

    def test_succession_honours_its_start(self, tl):
        succession(tl, (FadeIn(), 1, 1.0), (FadeIn(), 2, 1.0), start=4.0)
        assert _times(tl.track(2, "alpha", create=False)) == [5.0, 6.0]

    def test_parallel_starts_every_play_together(self, tl):
        end = parallel(tl, (FadeIn(), 1, 2.0), (FadeIn(), 2, 5.0), (FadeIn(), 3, 1.0))
        for target in (1, 2, 3):
            assert _times(tl.track(target, "alpha", create=False))[0] == pytest.approx(0.0)
        assert end == pytest.approx(5.0), "the group ends with its longest play"

    def test_stagger_overlaps_by_the_lag_ratio(self, tl):
        stagger(tl, (FadeIn(), 1, 4.0), (FadeIn(), 2, 4.0), (FadeIn(), 3, 4.0), lag_ratio=0.25)
        assert _times(tl.track(1, "alpha", create=False))[0] == pytest.approx(0.0)
        assert _times(tl.track(2, "alpha", create=False))[0] == pytest.approx(1.0)
        assert _times(tl.track(3, "alpha", create=False))[0] == pytest.approx(2.0)

    def test_the_default_lag_ratio_is_manims(self):
        assert DEFAULT_LAG_RATIO == 0.25

    def test_lag_ratio_one_is_succession(self, tl):
        group(tl, [(FadeIn(), 1, 2.0), (FadeIn(), 2, 2.0)], lag_ratio=1.0)
        assert _times(tl.track(2, "alpha", create=False))[0] == pytest.approx(2.0)

    def test_lag_ratio_zero_is_parallel(self, tl):
        group(tl, [(FadeIn(), 1, 2.0), (FadeIn(), 2, 2.0)], lag_ratio=0.0)
        assert _times(tl.track(2, "alpha", create=False))[0] == pytest.approx(0.0)

    def test_a_play_record_carries_its_own_duration(self, tl):
        succession(tl, Play(FadeIn(), 1, duration=7.0), Play(FadeIn(), 2, duration=1.0))
        assert _times(tl.track(2, "alpha", create=False)) == [7.0, 8.0]

    def test_a_delay_shifts_a_play_inside_its_slot(self, tl):
        parallel(tl, Play(FadeIn(), 1, 1.0), Play(FadeIn(), 2, 1.0, delay=0.5))
        assert _times(tl.track(2, "alpha", create=False))[0] == pytest.approx(0.5)

    def test_bare_tuples_take_the_groups_default_duration(self, tl):
        succession(tl, (FadeIn(), 1), (FadeIn(), 2), duration=2.5)
        assert _times(tl.track(2, "alpha", create=False)) == [2.5, 5.0]

    def test_a_wait_consumes_time_and_writes_nothing(self, tl):
        succession(tl, (FadeIn(), 1, 1.0), (Wait(), None, 2.0), (FadeIn(), 2, 1.0))
        assert _times(tl.track(2, "alpha", create=False)) == [3.0, 4.0]
        assert tl.track(None, "alpha", create=False) is None

    def test_lagged_applies_one_verb_to_many_targets(self, tl):
        end = lagged(tl, FadeIn(), [10, 11, 12], duration=2.0, lag_ratio=0.5)
        assert _times(tl.track(10, "alpha", create=False))[0] == pytest.approx(0.0)
        assert _times(tl.track(11, "alpha", create=False))[0] == pytest.approx(1.0)
        assert _times(tl.track(12, "alpha", create=False))[0] == pytest.approx(2.0)
        assert end == pytest.approx(4.0)

    def test_the_returned_end_time_chains(self, tl):
        end = succession(tl, (FadeIn(), 1, 1.5))
        succession(tl, (FadeOut(), 1, 1.5), start=end)
        assert _times(tl.track(1, "alpha", create=False)) == [0.0, 1.5, 3.0]

    def test_a_bad_play_shape_raises(self, tl):
        with pytest.raises(TypeError, match="Play"):
            succession(tl, FadeIn())

    def test_groups_compose_with_the_scene_model(self, tl):
        """A five-line 'script' really does produce a scrubbable, scened animation."""
        end = succession(
            tl,
            (Create(), 1, 1.0),
            (ColorTo(to=(1, 0, 0), start_color=(0, 0, 1)), 1, 1.0),
            (FadeOut(), 1, 1.0),
        )
        tl.set_duration(end)
        tl.auto_scenes(3)
        assert [s.name for s in tl.scenes] == ["Scene 1", "Scene 2", "Scene 3"]
        assert tl.evaluate(0.5)[(1, "draw_fraction")] == pytest.approx(0.5)
        assert tl.evaluate(2.5)[(1, "alpha")] == pytest.approx(0.5)


class TestRecordsAreData:
    """The property the whole design rests on: a verb is a value."""

    def test_verbs_compare_by_value(self):
        assert FadeIn(to=0.5) == FadeIn(to=0.5)
        assert FadeIn(to=0.5) != FadeIn(to=0.6)

    def test_verbs_are_immutable(self):
        with pytest.raises(Exception):
            FadeIn().to = 0.5

    def test_verbs_are_hashable_so_they_can_key_a_cache(self):
        assert len({FadeIn(), FadeIn(), FadeOut()}) == 2

    def test_a_span_is_a_record_too(self):
        s = PropertySpan("alpha", ((0.0, 0.0), (1.0, 1.0)))
        assert s == PropertySpan("alpha", ((0.0, 0.0), (1.0, 1.0)))
