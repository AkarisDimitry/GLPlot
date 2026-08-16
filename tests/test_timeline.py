"""Test the animation core: easing, interpolation, tracks, scenes and the playhead.

Pure numpy. No GL, no window, no imgui — which is the point of keeping the timeline as
plain data: an animation model you can only exercise by watching it play is one you cannot
test.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.core.timeline import (
    COMMON_FPS,
    DEFAULT_FPS,
    EASING_LABELS,
    EASINGS,
    LOOP_MODES,
    MAX_DURATION,
    MIN_DURATION,
    STEP,
    Keyframe,
    Scene,
    Timeline,
    TimelineError,
    Track,
    ease,
    easing,
    interpolate_values,
    is_blendable,
)


class TestEasing:
    """Every curve must start at 0 and end at 1, or segments would not meet their keys."""

    @pytest.mark.parametrize("name", sorted(EASINGS))
    def test_endpoints_are_exact(self, name):
        fn = EASINGS[name]
        assert fn(0.0) == pytest.approx(0.0, abs=1e-9)
        assert fn(1.0) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("name", sorted(EASINGS))
    def test_finite_across_the_unit_interval(self, name):
        values = [EASINGS[name](t) for t in np.linspace(0.0, 1.0, 51)]
        assert all(np.isfinite(v) for v in values)

    @pytest.mark.parametrize("name", sorted(EASINGS))
    def test_every_easing_is_labelled(self, name):
        """An unlabelled easing cannot appear in the picker, so it may as well not exist."""
        assert name in EASING_LABELS and EASING_LABELS[name]

    def test_step_holds_until_the_very_end(self):
        """The discrete-progression easing: no blending anywhere inside the segment."""
        assert ease(STEP, 0.0) == 0.0
        assert ease(STEP, 0.5) == 0.0
        assert ease(STEP, 0.999) == 0.0
        assert ease(STEP, 1.0) == 1.0

    def test_linear_is_the_identity(self):
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert ease("linear", t) == pytest.approx(t)

    def test_smooth_is_monotone_with_flat_ends(self):
        values = np.array([EASINGS["smooth"](t) for t in np.linspace(0, 1, 101)])
        assert np.all(np.diff(values) >= -1e-12)
        assert values[1] - values[0] < values[50] - values[49]

    def test_back_and_elastic_overshoot_on_purpose(self):
        """They leave [0, 1] mid-flight — that is the effect, not a bug."""
        assert max(EASINGS["back"](t) for t in np.linspace(0, 1, 101)) > 1.0
        assert min(EASINGS["elastic"](t) for t in np.linspace(0, 1, 101)) < 1.0

    def test_ease_clamps_out_of_range_input(self):
        assert ease("linear", -5.0) == pytest.approx(0.0)
        assert ease("linear", 5.0) == pytest.approx(1.0)

    def test_unknown_easing_raises(self):
        with pytest.raises(TimelineError, match="unknown easing"):
            easing("swoosh")


class TestInterpolation:
    """What can be blended, and what must be held."""

    def test_floats_blend(self):
        assert interpolate_values(0.0, 10.0, 0.25) == pytest.approx(2.5)

    def test_sequences_blend_elementwise_and_keep_their_type(self):
        out = interpolate_values((0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0), 0.5)
        assert isinstance(out, tuple)
        assert out == pytest.approx((0.5, 0.5, 0.5, 1.0))
        assert isinstance(interpolate_values([0.0], [2.0], 0.5), list)

    def test_arrays_blend_elementwise(self):
        a, b = np.zeros((3, 2)), np.ones((3, 2))
        assert np.allclose(interpolate_values(a, b, 0.25), 0.25)

    def test_mismatched_array_shapes_hold_rather_than_resample(self):
        """A silent resample would change what the user actually keyed."""
        a, b = np.zeros(4), np.ones(9)
        assert np.allclose(interpolate_values(a, b, 0.5), a)

    def test_mismatched_sequence_lengths_hold(self):
        assert interpolate_values((0.0, 0.0), (1.0, 1.0, 1.0), 0.5) == (0.0, 0.0)

    def test_booleans_are_never_blended(self):
        """ "Half visible" has no meaning a plot could draw."""
        assert is_blendable(True) is False
        assert interpolate_values(True, False, 0.5) is True

    def test_strings_are_held(self):
        assert is_blendable("magma") is False
        assert interpolate_values("viridis", "magma", 0.5) == "viridis"

    def test_endpoints_are_exact(self):
        assert interpolate_values(3.0, 9.0, 0.0) == 3.0
        assert interpolate_values(3.0, 9.0, 1.0) == 9.0

    def test_an_empty_sequence_is_not_blendable(self):
        assert is_blendable([]) is False


class TestKeyframe:
    def test_it_normalises_and_validates(self):
        key = Keyframe(time=2, value=1.0, easing="LINEAR")
        assert key.time == 2.0 and key.easing == "linear"

    def test_a_non_finite_time_is_refused(self):
        with pytest.raises(TimelineError, match="finite"):
            Keyframe(time=float("nan"), value=0.0)

    def test_an_unknown_easing_is_refused(self):
        with pytest.raises(TimelineError, match="unknown easing"):
            Keyframe(time=0.0, value=0.0, easing="swoosh")


class TestTrack:
    def _track(self):
        track = Track(target=1, prop="alpha")
        track.add(0.0, 0.0, "linear")
        track.add(2.0, 1.0, "linear")
        return track

    def test_keys_stay_sorted_however_they_are_added(self):
        track = Track(target=1, prop="alpha")
        for t in (3.0, 1.0, 2.0, 0.0):
            track.add(t, t)
        assert [k.time for k in track.keyframes] == [0.0, 1.0, 2.0, 3.0]

    def test_adding_at_an_existing_time_replaces(self):
        """Two keys at one instant have no defined order and one would be unreachable."""
        track = self._track()
        track.add(0.0, 0.5)
        assert len(track.keyframes) == 2
        assert track.keyframes[0].value == 0.5

    def test_value_interpolates_between_keys(self):
        assert self._track().value_at(1.0) == pytest.approx(0.5)

    def test_value_is_held_outside_the_keyed_range(self):
        """Extrapolating an eased curve invents values the user never specified."""
        track = self._track()
        assert track.value_at(-10.0) == 0.0
        assert track.value_at(99.0) == 1.0

    def test_an_empty_track_has_no_value(self):
        assert Track(target=1, prop="alpha").value_at(0.0) is None
        assert Track(target=1, prop="alpha").time_range() is None

    def test_interpolate_false_holds_every_segment(self):
        """For a value that cannot be blended, stated once instead of per keyframe."""
        track = self._track()
        track.interpolate = False
        assert track.value_at(1.9) == 0.0
        assert track.value_at(2.0) == 1.0

    def test_step_easing_holds_that_segment_only(self):
        track = Track(target=1, prop="n")
        track.add(0.0, 0.0, STEP)
        track.add(1.0, 10.0, "linear")
        track.add(2.0, 20.0)
        assert track.value_at(0.9) == 0.0
        assert track.value_at(1.5) == pytest.approx(15.0)

    def test_coincident_keys_do_not_divide_by_zero(self):
        track = Track(target=1, prop="a")
        track.keyframes = [Keyframe(1.0, 0.0), Keyframe(1.0, 5.0)]
        assert np.isfinite(track.value_at(1.0))

    def test_remove_and_move(self):
        track = self._track()
        assert track.remove_at(2.0) is True
        assert track.remove_at(2.0) is False
        track.add(5.0, 1.0)
        assert track.move(1, 0.5) is True
        assert [k.time for k in track.keyframes] == [0.0, 0.5]
        assert track.move(99, 0.0) is False

    def test_bake_evaluates_at_every_requested_time(self):
        assert self._track().bake([0.0, 1.0, 2.0]) == [0.0, pytest.approx(0.5), 1.0]

    def test_key_is_the_identity(self):
        assert self._track().key() == (1, "alpha")


class TestTimelineTransport:
    def test_defaults(self):
        tl = Timeline()
        assert tl.fps == DEFAULT_FPS
        assert tl.playing is False
        assert tl.time == 0.0
        assert tl.loop in LOOP_MODES

    def test_duration_is_clamped(self):
        assert Timeline(duration=-5.0).duration >= MIN_DURATION
        assert Timeline(duration=1e9).duration <= MAX_DURATION

    def test_frame_grid_covers_both_ends(self):
        tl = Timeline(duration=2.0, fps=10)
        times = tl.frame_times()
        assert tl.frame_count == 21
        assert times[0] == 0.0 and times[-1] == pytest.approx(2.0)

    def test_seek_clamps(self):
        tl = Timeline(duration=2.0)
        assert tl.seek(-1.0) == 0.0
        assert tl.seek(99.0) == pytest.approx(2.0)

    def test_frame_stepping(self):
        tl = Timeline(duration=1.0, fps=10)
        tl.seek_frame(0)
        assert tl.step_frames(3) == pytest.approx(0.3)
        assert tl.step_frames(-1) == pytest.approx(0.2)
        tl.step_frames(-99)
        assert tl.time == 0.0

    def test_toggle_and_stop(self):
        tl = Timeline()
        assert tl.toggle() is True
        assert tl.toggle() is False
        tl.play()
        tl.seek(1.0)
        tl.stop()
        assert tl.playing is False and tl.time == 0.0

    def test_advance_does_nothing_while_paused(self):
        tl = Timeline(duration=5.0)
        assert tl.advance(0.1) is False
        assert tl.time == 0.0

    def test_advance_ignores_an_implausible_delta(self):
        """A blocked loop or a clock jump must not skip half the animation."""
        tl = Timeline(duration=5.0)
        tl.play()
        assert tl.advance(10.0) is False
        assert tl.time == 0.0

    def test_loop_wraps(self):
        tl = Timeline(duration=1.0, loop="loop")
        tl.play()
        tl.seek(0.9)
        tl.advance(0.2)
        assert tl.time == pytest.approx(0.1, abs=1e-6)
        assert tl.playing is True

    def test_once_stops_at_the_end(self):
        tl = Timeline(duration=1.0, loop="once")
        tl.play()
        tl.seek(0.95)
        tl.advance(0.2)
        assert tl.time == pytest.approx(1.0)
        assert tl.playing is False

    def test_pingpong_reverses(self):
        tl = Timeline(duration=1.0, loop="pingpong")
        tl.play()
        tl.seek(0.95)
        tl.advance(0.2)
        assert tl.time < 1.0
        assert tl._direction == -1
        tl.seek(0.05)
        tl.advance(0.2)  # travelling backwards now, bounces off zero
        assert tl._direction == 1

    def test_negative_speed_plays_backwards(self):
        tl = Timeline(duration=2.0, loop="once")
        tl.play()
        tl.seek(1.0)
        tl.speed = -1.0
        tl.advance(0.1)
        assert tl.time < 1.0

    def test_an_unknown_loop_mode_is_refused(self):
        with pytest.raises(TimelineError, match="loop must be one of"):
            Timeline(loop="boomerang")

    def test_common_fps_are_all_usable(self):
        for fps in COMMON_FPS:
            tl = Timeline(duration=1.0, fps=fps)
            assert tl.frame_count > 1

    def test_a_bad_fps_falls_back_to_the_default(self):
        assert Timeline(fps=0).fps == DEFAULT_FPS
        assert Timeline(fps=-30).fps == DEFAULT_FPS


class TestTimelineTracks:
    def test_track_is_created_on_demand_and_reused(self):
        tl = Timeline()
        first = tl.track(1, "alpha")
        assert tl.track(1, "alpha") is first
        assert len(tl.tracks) == 1

    def test_create_false_does_not_create(self):
        tl = Timeline()
        assert tl.track(1, "alpha", create=False) is None
        assert tl.tracks == []

    def test_key_is_the_one_liner(self):
        tl = Timeline(duration=4.0)
        tl.seek(1.0)
        key = tl.key(7, "alpha", 0.5)
        assert key.time == pytest.approx(1.0)
        assert tl.evaluate(1.0)[(7, "alpha")] == pytest.approx(0.5)

    def test_evaluate_returns_only_live_tracks(self):
        tl = Timeline(duration=2.0)
        tl.key(1, "alpha", 0.0, time=0.0)
        tl.key(1, "alpha", 1.0, time=2.0)
        tl.track(2, "empty")  # no keyframes
        muted = tl.track(3, "muted")
        muted.add(0.0, 5.0)
        muted.enabled = False

        values = tl.evaluate(1.0)
        assert values[(1, "alpha")] == pytest.approx(0.5)
        assert (2, "empty") not in values
        assert (3, "muted") not in values

    def test_add_track_replaces_by_identity(self):
        tl = Timeline()
        tl.add_track(Track(target=1, prop="a", label="first"))
        tl.add_track(Track(target=1, prop="a", label="second"))
        assert len(tl.tracks) == 1 and tl.tracks[0].label == "second"

    def test_remove_and_discard(self):
        tl = Timeline()
        tl.key(1, "alpha", 0.0)
        tl.key(1, "color", (0.0, 0.0, 0.0, 1.0))
        tl.key(2, "alpha", 0.0)
        assert len(tl.tracks_for(1)) == 2
        assert tl.remove_track(1, "alpha") is True
        assert tl.remove_track(1, "alpha") is False
        assert tl.discard_target(1) == 1
        assert len(tl.tracks) == 1

    def test_is_empty_ignores_tracks_with_no_keys(self):
        tl = Timeline()
        tl.track(1, "alpha")
        assert tl.is_empty() is True
        tl.key(1, "alpha", 0.0)
        assert tl.is_empty() is False

    def test_bake_produces_one_dict_per_frame(self):
        tl = Timeline(duration=1.0, fps=10)
        tl.key(1, "alpha", 0.0, time=0.0)
        tl.key(1, "alpha", 1.0, time=1.0)
        baked = tl.bake()
        assert len(baked) == tl.frame_count
        assert baked[0][(1, "alpha")] == 0.0
        assert baked[-1][(1, "alpha")] == pytest.approx(1.0)

    def test_a_discrete_track_never_blends(self):
        """Half of one Game-of-Life generation and half of the next is not a board."""
        tl = Timeline(duration=2.0)
        track = tl.track(9, "grid")
        track.interpolate = False
        track.add(0.0, np.zeros((4, 4)))
        track.add(1.0, np.ones((4, 4)))
        mid = tl.evaluate(0.5)[(9, "grid")]
        assert set(np.unique(mid)) <= {0.0, 1.0}

    def test_fit_duration_grows_but_never_shrinks(self):
        tl = Timeline(duration=2.0)
        tl.key(1, "alpha", 0.0, time=7.0)
        assert tl.fit_duration() == pytest.approx(7.0)
        tl.remove_track(1, "alpha")
        assert tl.fit_duration() == pytest.approx(7.0)


class TestScenes:
    def test_a_scene_needs_a_positive_span(self):
        with pytest.raises(TimelineError, match="end > start"):
            Scene(name="s", start=1.0, end=1.0)

    def test_contains_is_half_open_so_scenes_can_abut(self):
        first = Scene(name="a", start=0.0, end=1.0)
        second = Scene(name="b", start=1.0, end=2.0)
        assert first.contains(0.0) and not first.contains(1.0)
        assert second.contains(1.0)

    def test_progress_is_clamped(self):
        scene = Scene(name="a", start=2.0, end=4.0)
        assert scene.progress(1.0) == 0.0
        assert scene.progress(3.0) == pytest.approx(0.5)
        assert scene.progress(9.0) == 1.0

    def test_scenes_stay_sorted_and_extend_the_duration(self):
        tl = Timeline(duration=1.0)
        tl.add_scene("second", 4.0, 6.0)
        tl.add_scene("first", 0.0, 2.0)
        assert [s.name for s in tl.scenes] == ["first", "second"]
        assert tl.duration >= 6.0

    def test_scene_at_and_the_final_instant(self):
        tl = Timeline(duration=4.0)
        tl.auto_scenes(2)
        assert tl.scene_at(0.0).name == "Scene 1"
        assert tl.scene_at(3.0).name == "Scene 2"
        # The last scene owns its own end, so the final frame is not "nowhere".
        assert tl.scene_at(4.0).name == "Scene 2"

    def test_navigation(self):
        tl = Timeline(duration=3.0)
        tl.auto_scenes(3)
        tl.seek(0.0)
        assert tl.next_scene().name == "Scene 2"
        assert tl.next_scene().name == "Scene 3"
        assert tl.next_scene().name == "Scene 3"  # clamped at the end
        assert tl.previous_scene().name == "Scene 2"

    def test_navigation_with_no_scenes_is_safe(self):
        tl = Timeline()
        assert tl.next_scene() is None
        assert tl.go_to_scene(0) is None
        assert tl.scene_index() == -1

    def test_auto_scenes_tile_the_whole_timeline(self):
        tl = Timeline(duration=6.0)
        scenes = tl.auto_scenes(3)
        assert len(scenes) == 3
        assert scenes[0].start == 0.0
        assert scenes[-1].end == pytest.approx(6.0)
        for a, b in zip(scenes, scenes[1:]):
            assert a.end == pytest.approx(b.start)

    def test_remove_scene(self):
        tl = Timeline(duration=2.0)
        tl.auto_scenes(2)
        assert tl.remove_scene("Scene 1") is True
        assert tl.remove_scene("Scene 1") is False


class TestPanelIntegration:
    """Each panel owns a timeline, so a split figure can animate one region."""

    def test_every_panel_has_its_own(self):
        from glplot.engine import GPULinePlot

        plot = GPULinePlot()
        assert isinstance(plot.active_panel.timeline, Timeline)

        from glplot.core import layout

        plot.set_panels(layout.grid(1, 2))
        assert plot.panels[0].timeline is not plot.panels[1].timeline

    def test_a_fresh_timeline_costs_nothing(self):
        from glplot.engine import GPULinePlot

        timeline = GPULinePlot().active_panel.timeline
        assert timeline.is_empty()
        assert timeline.playing is False
