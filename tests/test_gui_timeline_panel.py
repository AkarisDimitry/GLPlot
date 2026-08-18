"""Test the animation GUI: the Timeline panel, the Presentation panel, and their actions.

Panel *bodies* go through the sanctioned headless imgui harness (CONTRACT §2.10): create a
context, open a frame, call ``draw``, render, assert the frame closed. That is what catches
a wrong pyimgui signature, an unbalanced begin/end or a missing pop -- which is most of
what can go wrong in a panel, and none of which a state-only test would ever see. The
``harness`` fixture destroys its context in teardown: imgui's context is *global state*,
and a leaked one makes unrelated files fail depending on collection order.

The action half is asserted on state, and always in the same shape: the mutation must NOT
have happened before ``queue.drain`` and must have happened after (CONTRACT §1.1).
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.core.timeline import EASING_LABELS, Keyframe, Track
from glplot.engine import GPULinePlot
from glplot.gui.commands import CommandQueue
from glplot.gui.datasets import DataStore
from glplot.gui.history import UndoStack
from glplot.gui.panels import timeline as timeline_mod


class FakeWorkspace:
    """The four services a panel reads off a workspace.

    A real ``Workspace`` needs imgui to construct and builds every panel; a stand-in keeps
    these tests to the panel under test.
    """

    def __init__(self, plot):
        self.plot = plot
        self.queue = CommandQueue()
        self.undo = UndoStack()
        self.store = DataStore()
        self.hud = None

    def drain(self):
        """Run the queue against the plot, as ``_main_loop`` does at the frame top."""
        self.queue.drain(self.plot)


class FakeClock:
    """A monotonic clock the tests advance by hand, so playback is not wall-clock flaky."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = float(now)

    def perf_counter(self) -> float:
        return self.now


@pytest.fixture
def plot():
    return GPULinePlot()


@pytest.fixture
def ws(plot):
    return FakeWorkspace(plot)


@pytest.fixture
def harness():
    """A headless imgui context sized like a real window, destroyed afterwards.

    The teardown is not optional: a leaked current context is seen by every later file's
    own ``create_context`` fixture, and the failures land somewhere else entirely.
    """
    imgui = pytest.importorskip("imgui_bundle").imgui
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = 1400, 900
    # Font atlas is dynamic under imgui-bundle; get_tex_data_as_rgba32()/texture_id no
    # longer exist. Telling imgui a backend owns texture building is the headless
    # equivalent, since this harness never renders real pixels.
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    io.delta_time = 1 / 60.0
    yield imgui
    imgui.destroy_context(ctx)


@pytest.fixture
def all_sections(monkeypatch):
    """Force every collapsing header open, so one frame covers every widget branch."""
    from glplot.gui import widgets

    monkeypatch.setattr(widgets, "section", lambda label, **kw: True)


@pytest.fixture
def clock(monkeypatch):
    """Replace the panel module's clock so playback advances deterministically."""
    fake = FakeClock()
    monkeypatch.setattr(timeline_mod, "_time", fake)
    return fake


def _frame(imgui, panel, title="Panel"):
    imgui.new_frame()
    imgui.begin(title)
    panel.draw()
    imgui.end()
    imgui.render()
    assert imgui.get_draw_data() is not None


def _timeline_panel(ws):
    from glplot.gui.panels.timeline import TimelinePanel

    return TimelinePanel(ws)


def _presentation_panel(ws):
    from glplot.gui.panels.presentation import PresentationPanel

    return PresentationPanel(ws)


def _populate(plot, *, keys=12):
    """A timeline with three tracks of differing character and a run of keyframes."""
    timeline = plot.active_panel.timeline
    timeline.set_duration(4.0)
    timeline.key("camera", "azim", 0.0, time=0.0)
    timeline.key("camera", "azim", 180.0, time=3.0, easing_name="smoother")
    timeline.key("camera", "elev", 15.0, time=0.5)
    for i in range(keys):
        timeline.key("camera", "roll", float(i), time=i * (3.5 / max(keys - 1, 1)))
    # A track whose values cannot be blended, so the hold path is drawn too.
    grid = timeline.track("state", "grid")
    grid.interpolate = False
    grid.add(0.0, np.zeros((3, 3)))
    grid.add(1.0, np.ones((3, 3)), "step")
    return timeline


# ======================================================================================
# Pure helpers
# ======================================================================================


class TestHelpers:
    def test_snap_puts_the_playhead_on_a_real_frame(self, plot):
        timeline = plot.active_panel.timeline
        timeline.set_fps(10.0)
        assert timeline_mod.snap_time(timeline, 0.44) == pytest.approx(0.4)
        assert timeline_mod.snap_time(timeline, 0.44, snap=False) == pytest.approx(0.44)

    def test_snap_clamps_into_the_timeline(self, plot):
        timeline = plot.active_panel.timeline
        timeline.set_duration(2.0)
        assert timeline_mod.snap_time(timeline, -5.0) == pytest.approx(0.0)
        assert timeline_mod.snap_time(timeline, 99.0) == pytest.approx(2.0)

    def test_a_track_label_falls_back_to_target_and_prop(self):
        assert timeline_mod.track_label(Track(target="camera", prop="azim")) == "camera.azim"
        named = Track(target="camera", prop="azim", label="Turntable")
        assert timeline_mod.track_label(named) == "Turntable"

    def test_a_label_containing_the_id_separator_is_defanged(self):
        track = Track(target=1, prop="alpha", label="a##b")
        assert "##" not in timeline_mod.track_label(track)

    def test_value_descriptions_cover_every_kind(self):
        assert timeline_mod._describe_value(1.5) == "1.5"
        assert timeline_mod._describe_value(True) == "true"
        assert "array" in timeline_mod._describe_value(np.zeros((2, 3)))
        assert timeline_mod._describe_value((1.0, 2.0)).startswith("(")
        assert timeline_mod._describe_value(None) == "-"

    def test_only_a_plain_number_counts_as_scalar(self):
        assert timeline_mod._is_scalar(3) is True
        assert timeline_mod._is_scalar(np.float64(3)) is True
        assert timeline_mod._is_scalar(True) is False
        assert timeline_mod._is_scalar(np.zeros(3)) is False


# ======================================================================================
# The applier and playback
# ======================================================================================


class TestApplyTimeline:
    def test_it_drives_a_layer_style_property(self, plot):
        plot.add_scatter(np.arange(5.0), np.arange(5.0), None)
        layer = plot.scene.layers[0]
        timeline = plot.active_panel.timeline
        timeline.key(layer.layer_id, "alpha", 0.0, time=0.0)
        timeline.key(layer.layer_id, "alpha", 1.0, time=1.0)
        timeline.seek(1.0)
        assert timeline_mod.apply_timeline(plot, timeline) == 1
        assert layer.style.alpha == pytest.approx(1.0)

    def test_it_drives_the_3d_camera(self, plot):
        timeline = plot.active_panel.timeline
        timeline.key("camera", "azim", 42.0, time=0.0)
        assert timeline_mod.apply_timeline(plot, timeline) == 1
        assert plot.camera3d.azim == pytest.approx(42.0)

    def test_a_2d_figure_gains_no_3d_decoration_from_a_camera_track(self, plot):
        """`set_3d_view` builds the axis box; calling it in a 2D figure would grow one."""
        plot.add_scatter(np.arange(5.0), np.arange(5.0), None)
        before = len(plot.scene.layers)
        timeline = plot.active_panel.timeline
        timeline.key("camera", "elev", 30.0, time=0.0)
        timeline_mod.apply_timeline(plot, timeline)
        assert len(plot.scene.layers) == before
        assert plot.camera3d.elev == pytest.approx(30.0)

    def test_an_unknown_property_is_ignored_rather_than_raising(self, plot):
        plot.add_scatter(np.arange(5.0), np.arange(5.0), None)
        layer = plot.scene.layers[0]
        timeline = plot.active_panel.timeline
        timeline.key(layer.layer_id, "not_a_field", 1.0, time=0.0)
        timeline.key("camera", "not_a_field", 1.0, time=0.0)
        assert timeline_mod.apply_timeline(plot, timeline) == 0

    def test_a_muted_track_contributes_nothing(self, plot):
        plot.add_scatter(np.arange(5.0), np.arange(5.0), None)
        layer = plot.scene.layers[0]
        timeline = plot.active_panel.timeline
        track = timeline.track(layer.layer_id, "alpha")
        track.add(0.0, 0.25)
        track.enabled = False
        assert timeline_mod.apply_timeline(plot, timeline) == 0

    def test_the_new_camera_target_drives_the_3d_camera_too(self, plot):
        timeline = plot.active_panel.timeline
        timeline.key(timeline_mod.CAMERA_TARGET, "azim", 42.0, time=0.0)
        assert timeline_mod.apply_timeline(plot, timeline) == 1
        assert plot.camera3d.azim == pytest.approx(42.0)

    def test_it_drives_the_2d_camera(self, plot):
        timeline = plot.active_panel.timeline
        timeline.key(timeline_mod.CAMERA2D_TARGET, "cx", 7.0, time=0.0)
        timeline.key(timeline_mod.CAMERA2D_TARGET, "zoom_x", 3.0, time=0.0)
        assert timeline_mod.apply_timeline(plot, timeline) == 2
        assert plot.camera.cx == pytest.approx(7.0)
        assert plot.camera.zoom_x == pytest.approx(3.0)


class TestCameraTargetSeam:
    """The 3D camera target must be the one the *engine* resolves, not only this panel's.

    ``GPULinePlot._advance_timelines`` applies playback through ``glplot.anim.applier``,
    where ``"camera"`` means the 2D camera and ``"camera3d"`` the 3D one. A GUI-authored
    pose track spelled ``"camera"`` therefore scrubbed correctly and then did nothing at
    all on Play — visible only in a live window, which is exactly why it is pinned here.
    """

    def test_the_panel_authors_the_target_the_engine_applier_resolves(self, plot):
        from glplot.anim.applier import TARGET_CAMERA3D

        assert timeline_mod.CAMERA_TARGET == TARGET_CAMERA3D

    def test_a_captured_pose_survives_the_engines_own_applier(self, plot):
        from glplot.anim.applier import apply_values

        plot.set_ndim(3)
        timeline = plot.active_panel.timeline
        plot.camera3d.azim = 0.0
        timeline_mod.capture_keyframes(plot, timeline, time=0.0)
        plot.camera3d.azim = 90.0
        timeline_mod.capture_keyframes(plot, timeline, time=2.0)

        plot.camera3d.azim = -999.0
        assert apply_values(plot, timeline.evaluate(1.0)) > 0
        assert plot.camera3d.azim == pytest.approx(45.0)

    def test_the_legacy_camera_spelling_still_scrubs(self, plot):
        """Timelines authored before the rename keep working in this panel's applier."""
        timeline = plot.active_panel.timeline
        timeline.key("camera", "elev", 12.0, time=0.0)
        assert timeline_mod.apply_timeline(plot, timeline) == 1
        assert plot.camera3d.elev == pytest.approx(12.0)


class TestCapture:
    """Capture is the one authoring gesture that works on an empty timeline."""

    def test_it_reads_the_3d_pose_in_a_3d_figure(self, plot):
        plot.set_ndim(3)
        plot.camera3d.elev, plot.camera3d.azim = 12.0, -34.0
        state = timeline_mod.capture_state(plot, "view")
        values = {prop: value for _, prop, value in state}
        assert {t for t, _, _ in state} == {timeline_mod.CAMERA_TARGET}
        assert values["elev"] == pytest.approx(12.0)
        assert values["azim"] == pytest.approx(-34.0)

    def test_it_reads_pan_and_zoom_in_a_2d_figure(self, plot):
        plot.add_scatter(np.arange(5.0), np.arange(5.0), None)
        state = timeline_mod.capture_state(plot, "view")
        assert {t for t, _, _ in state} == {timeline_mod.CAMERA2D_TARGET}
        assert {prop for _, prop, _ in state} == {"cx", "cy", "zoom_x", "zoom_y"}

    def test_an_auto_distance_is_not_keyed(self, plot):
        """`distance=None` means "fit the data" — a real state, but not a keyframe value."""
        plot.set_ndim(3)
        plot.camera3d.distance = None
        assert "distance" not in {prop for _, prop, _ in timeline_mod.capture_state(plot, "view")}

    def test_the_layers_scope_reads_every_layer(self, plot):
        plot.add_scatter(np.arange(5.0), np.arange(5.0), None)
        plot.add_scatter(np.arange(5.0), np.arange(5.0) * 2, None)
        ids = {layer.layer_id for layer in plot.scene.layers}
        state = timeline_mod.capture_state(plot, "layers")
        assert {t for t, _, _ in state} == ids
        assert "alpha" in {prop for _, prop, _ in state}

    def test_everything_is_the_union_of_the_two(self, plot):
        plot.add_scatter(np.arange(5.0), np.arange(5.0), None)
        view = timeline_mod.capture_state(plot, "view")
        layers = timeline_mod.capture_state(plot, "layers")
        assert len(timeline_mod.capture_state(plot, "all")) == len(view) + len(layers)

    def test_the_tracks_scope_captures_only_what_exists(self, plot):
        timeline = plot.active_panel.timeline
        plot.camera3d.azim = 55.0
        timeline.track(timeline_mod.CAMERA_TARGET, "azim")
        state = timeline_mod.capture_state(plot, "tracks", timeline=timeline)
        assert state == [(timeline_mod.CAMERA_TARGET, "azim", pytest.approx(55.0))]

    def test_an_unknown_scope_raises(self, plot):
        with pytest.raises(ValueError, match="unknown capture scope"):
            timeline_mod.capture_state(plot, "everything-ish")

    def test_capturing_creates_the_missing_tracks(self, plot):
        plot.set_ndim(3)
        timeline = plot.active_panel.timeline
        assert timeline.tracks == []
        touched = timeline_mod.capture_keyframes(plot, timeline, time=0.0)
        assert touched and timeline.tracks
        assert all(len(track.keyframes) == 1 for track in touched)

    def test_capturing_twice_makes_a_move(self, plot):
        plot.set_ndim(3)
        timeline = plot.active_panel.timeline
        plot.camera3d.azim = 0.0
        timeline_mod.capture_keyframes(plot, timeline, time=0.0)
        plot.camera3d.azim = 80.0
        timeline_mod.capture_keyframes(plot, timeline, time=1.0)
        azim = timeline.track(timeline_mod.CAMERA_TARGET, "azim", create=False)
        assert len(azim.keyframes) == 2
        assert azim.value_at(0.0) == pytest.approx(0.0)
        assert azim.value_at(1.0) == pytest.approx(80.0)

    def test_capturing_grows_the_duration_to_cover_the_key(self, plot):
        plot.set_ndim(3)
        timeline = plot.active_panel.timeline
        timeline.set_duration(2.0)
        timeline_mod.capture_keyframes(plot, timeline, time=9.0)
        assert timeline.duration == pytest.approx(9.0)

    def test_a_boolean_is_held_rather_than_blended(self, plot):
        plot.add_scatter(np.arange(5.0), np.arange(5.0), None)
        timeline = plot.active_panel.timeline
        timeline_mod.capture_keyframes(plot, timeline, scope="layers", time=0.0)
        visible = next(t for t in timeline.tracks if t.prop == "visible")
        assert visible.interpolate is False
        assert visible.keyframes[0].easing == "step"

    def test_it_does_not_override_an_existing_tracks_interpolation(self, plot):
        plot.add_scatter(np.arange(5.0), np.arange(5.0), None)
        layer_id = plot.scene.layers[0].layer_id
        timeline = plot.active_panel.timeline
        track = timeline.track(layer_id, "visible")
        track.add(0.0, True)
        track.interpolate = True
        timeline_mod.capture_keyframes(plot, timeline, scope="layers", time=1.0)
        assert track.interpolate is True

    def test_live_value_degrades_to_the_fallback(self, plot):
        assert timeline_mod.live_value(plot, "camera3d", "not_a_field", 7.0) == 7.0
        assert timeline_mod.live_value(plot, 999999, "alpha") is None


class TestPresets:
    """A preset is a whole animation from one click — and ordinary keyframes afterwards."""

    def _prepare(self, plot, preset):
        """Put ``plot`` in the state ``preset`` needs, and return its timeline."""
        plot.add_scatter(np.arange(30.0), np.arange(30.0), None)
        if preset.requires == "3d":
            plot.set_ndim(3)
        return plot.active_panel.timeline

    @pytest.mark.parametrize("preset", timeline_mod.PRESETS, ids=lambda p: p.key)
    def test_every_preset_builds_real_keyframes(self, plot, preset):
        timeline = self._prepare(plot, preset)
        assert timeline_mod.preset_available(plot, preset) is True
        touched = timeline_mod.apply_preset(plot, timeline, preset.key)
        assert touched, preset.key
        # A preset that wrote one keyframe wrote a snap, not an animation.
        assert all(len(track.keyframes) >= 2 for track in touched), preset.key

    @pytest.mark.parametrize("preset", timeline_mod.PRESETS, ids=lambda p: p.key)
    def test_every_preset_is_documented_and_has_a_real_icon(self, preset):
        from glplot.gui.icons import ICON_SHAPES

        assert preset.label and preset.description
        assert preset.requires in ("3d", "2d", "layers", "any")
        assert preset.icon in ICON_SHAPES, preset.key

    @pytest.mark.parametrize("preset", timeline_mod.PRESETS, ids=lambda p: p.key)
    def test_every_preset_lands_inside_its_span(self, plot, preset):
        timeline = self._prepare(plot, preset)
        timeline.set_duration(4.0)
        touched = timeline_mod.apply_preset(plot, timeline, preset.key)
        times = [key.time for track in touched for key in track.keyframes]
        assert min(times) >= -1e-9, preset.key
        assert max(times) <= 4.0 + 1e-6, preset.key

    @pytest.mark.parametrize("preset", timeline_mod.PRESETS, ids=lambda p: p.key)
    def test_every_preset_applies_to_the_scene(self, plot, preset):
        """Built keyframes that no applier drives would be an animation of nothing."""
        timeline = self._prepare(plot, preset)
        timeline.set_duration(2.0)
        timeline_mod.apply_preset(plot, timeline, preset.key)
        timeline.seek(1.0)
        assert timeline_mod.apply_timeline(plot, timeline) > 0, preset.key

    def test_building_from_the_playhead_leaves_the_start_alone(self, plot):
        plot.set_ndim(3)
        timeline = plot.active_panel.timeline
        timeline.set_duration(4.0)
        touched = timeline_mod.apply_preset(plot, timeline, "turntable", start=2.0)
        assert min(k.time for t in touched for k in t.keyframes) == pytest.approx(2.0)

    def test_the_grand_tour_starts_where_the_camera_is(self, plot):
        """INHERIT has nothing to inherit on a fresh track, so leg one must be seeded."""
        plot.set_ndim(3)
        plot.camera3d.azim, plot.camera3d.elev = -10.0, 5.0
        timeline = plot.active_panel.timeline
        timeline.set_duration(4.0)
        timeline_mod.apply_preset(plot, timeline, "grand_tour")
        azim = timeline.track(timeline_mod.CAMERA_TARGET, "azim", create=False)
        assert azim.value_at(0.0) == pytest.approx(-10.0)
        # And it genuinely moves rather than snapping at the leg boundary.
        assert -90.0 < azim.value_at(0.5) < -10.0

    def test_the_grand_tour_visits_every_angle(self, plot):
        from glplot.core.camera3d import STANDARD_VIEWS

        plot.set_ndim(3)
        timeline = plot.active_panel.timeline
        timeline.set_duration(4.0)
        timeline_mod.apply_preset(plot, timeline, "grand_tour")
        azim = timeline.track(timeline_mod.CAMERA_TARGET, "azim", create=False)
        per = 4.0 / len(timeline_mod.GRAND_TOUR_VIEWS)
        for index, name in enumerate(timeline_mod.GRAND_TOUR_VIEWS):
            expected = STANDARD_VIEWS[name][1]
            assert azim.value_at((index + 1) * per) == pytest.approx(expected), name

    def test_rock_returns_to_where_it_started(self, plot):
        plot.set_ndim(3)
        plot.camera3d.azim = 15.0
        timeline = plot.active_panel.timeline
        timeline.set_duration(4.0)
        timeline_mod.apply_preset(plot, timeline, "rock")
        azim = timeline.track(timeline_mod.CAMERA_TARGET, "azim", create=False)
        assert azim.value_at(0.0) == pytest.approx(15.0)
        assert azim.value_at(4.0) == pytest.approx(15.0)
        assert azim.value_at(1.0) > 15.0 and azim.value_at(3.0) < 15.0

    def test_reveal_ends_at_each_layers_own_opacity(self, plot):
        """Fading everything to 1.0 would brighten a layer the user dimmed on purpose."""
        plot.add_scatter(np.arange(5.0), np.arange(5.0), None)
        layer = plot.scene.layers[0]
        layer.style.alpha = 0.4
        timeline = plot.active_panel.timeline
        timeline.set_duration(2.0)
        timeline_mod.apply_preset(plot, timeline, "reveal")
        alpha = timeline.track(layer.layer_id, "alpha", create=False)
        assert alpha.value_at(0.0) == pytest.approx(0.0)
        assert alpha.value_at(2.0) == pytest.approx(0.4)

    def test_a_stagger_fills_the_span_without_overrunning_it(self, plot):
        for i in range(4):
            plot.add_scatter(np.arange(5.0), np.arange(5.0) + i, None)
        timeline = plot.active_panel.timeline
        timeline.set_duration(4.0)
        touched = timeline_mod.apply_preset(plot, timeline, "reveal")
        ends = [max(k.time for k in track.keyframes) for track in touched]
        # The last layer finishes exactly at the end; the first finishes well before it.
        assert max(ends) == pytest.approx(4.0)
        assert min(ends) < 4.0

    def test_draw_on_traces_the_geometry_while_scrubbing(self, plot):
        """Not only under Play: a preset that is inert under the playhead reads as broken."""
        plot.add_scatter(np.arange(50.0), np.arange(50.0), None)
        layer = plot.scene.layers[0]
        timeline = plot.active_panel.timeline
        timeline.set_duration(2.0)
        timeline_mod.apply_preset(plot, timeline, "draw_on")

        seen = []
        for at in (0.0, 1.0, 2.0):
            timeline.seek(at)
            timeline_mod.apply_timeline(plot, timeline)
            seen.append(len(layer.pts))
        assert seen[0] < seen[1] < seen[2] == 50

    def test_an_unknown_preset_raises(self, plot):
        with pytest.raises(ValueError, match="unknown preset"):
            timeline_mod.apply_preset(plot, plot.active_panel.timeline, "moonwalk")

    def test_availability_follows_the_figure(self, plot):
        by_key = timeline_mod.PRESET_BY_KEY
        assert timeline_mod.preset_available(plot, by_key["turntable"]) is False
        assert timeline_mod.preset_available(plot, by_key["zoom_in_2d"]) is True
        assert timeline_mod.preset_available(plot, by_key["reveal"]) is False
        plot.set_ndim(3)
        plot.add_scatter(np.arange(5.0), np.arange(5.0), None)
        assert timeline_mod.preset_available(plot, by_key["turntable"]) is True
        assert timeline_mod.preset_available(plot, by_key["zoom_in_2d"]) is False
        assert timeline_mod.preset_available(plot, by_key["reveal"]) is True

    def test_presets_chain_from_the_playhead(self, plot):
        plot.set_ndim(3)
        plot.add_scatter(np.arange(5.0), np.arange(5.0), None)
        timeline = plot.active_panel.timeline
        timeline.set_duration(4.0)
        timeline_mod.apply_preset(plot, timeline, "reveal", start=0.0, duration=2.0)
        timeline_mod.apply_preset(plot, timeline, "turntable", start=2.0)
        azim = timeline.track(timeline_mod.CAMERA_TARGET, "azim", create=False)
        assert min(k.time for k in azim.keyframes) == pytest.approx(2.0)


class TestFitColorsSentinel:
    """``add_scatter(x, y, None)`` stores a one-element NaN array, not a colour."""

    def test_the_no_colours_sentinel_passes_through(self):
        from glplot.gui.layerops import _fit_colors

        sentinel = np.ascontiguousarray(None, np.float32)
        assert _fit_colors(sentinel, 50) is sentinel

    def test_a_genuinely_malformed_colour_array_still_raises(self):
        from glplot.gui.layerops import _fit_colors

        with pytest.raises(ValueError, match="colors must be"):
            _fit_colors(np.array([0.1, 0.2], dtype=np.float32), 4)


class TestTurntable:
    def test_it_sweeps_a_whole_turn_from_the_current_angle(self, plot):
        plot.set_ndim(3)
        plot.camera3d.azim = 30.0
        timeline = plot.active_panel.timeline
        timeline.set_duration(4.0)
        tracks = timeline_mod.turntable(plot, timeline)
        assert [t.prop for t in tracks] == ["azim"]
        track = tracks[0]
        assert track.target == timeline_mod.CAMERA_TARGET
        assert track.value_at(0.0) == pytest.approx(30.0)
        assert track.value_at(4.0) == pytest.approx(390.0)

    def test_it_is_readable_as_more_than_two_keys(self, plot):
        """Baked intermediates are what make the sweep legible in the track view."""
        plot.set_ndim(3)
        timeline = plot.active_panel.timeline
        assert len(timeline_mod.turntable(plot, timeline)[0].keyframes) > 2


class TestPlayback:
    """Who advances the playhead — and, since the engine hook landed, who must not.

    ``GPULinePlot._advance_timelines`` now steps every panel's timeline once per frame from
    wall clock and applies it through ``glplot.anim.applier``. ``queue_playback`` therefore
    stands down where that exists: two independent advances in one frame play the animation
    at double speed, and the second one is invisible from inside this panel's own tests.
    ``engine_drives_playback`` is the probe, so the panel still works against an engine
    that predates the hook — which is the state it was written against, and which the
    fallback tests below still cover.
    """

    def test_the_engine_owns_playback_on_a_current_engine(self, ws, plot):
        panel = _timeline_panel(ws)
        assert timeline_mod.engine_drives_playback(panel) is True

    def test_the_panel_stands_down_when_the_engine_drives(self, ws, plot, clock):
        """The regression this guards: drawing the panel must not double the rate."""
        panel = _timeline_panel(ws)
        timeline = plot.active_panel.timeline
        timeline.play()
        timeline_mod.queue_playback(panel)
        ws.drain()
        clock.now += 0.1
        timeline_mod.queue_playback(panel)
        ws.drain()
        assert timeline.time == pytest.approx(0.0)

    def test_the_engine_advances_it(self, ws, plot):
        """The engine hook is what moves the playhead now — end to end."""
        timeline = plot.active_panel.timeline
        timeline.set_duration(4.0)
        timeline.play()
        plot._advance_timelines(500.0)
        plot._advance_timelines(500.1)
        assert timeline.time == pytest.approx(0.1, abs=1e-6)

    def test_a_paused_timeline_does_not_move(self, ws, plot, clock):
        panel = _timeline_panel(ws)
        timeline = plot.active_panel.timeline
        timeline_mod.queue_playback(panel)
        ws.drain()
        clock.now += 0.1
        timeline_mod.queue_playback(panel)
        ws.drain()
        assert timeline.time == pytest.approx(0.0)

    def test_playing_advances_by_wall_clock(self, ws, plot, clock, monkeypatch):
        # The fallback path: an engine with no ``_advance_timelines`` hook.
        monkeypatch.setattr(timeline_mod, "engine_drives_playback", lambda p: False)
        panel = _timeline_panel(ws)
        timeline = plot.active_panel.timeline
        timeline.play()
        timeline_mod.queue_playback(panel)
        ws.drain()  # first call only stamps the clock
        clock.now += 0.1
        timeline_mod.queue_playback(panel)
        ws.drain()
        assert timeline.time == pytest.approx(0.1, abs=1e-6)

    def test_two_panels_in_one_frame_advance_it_once(self, ws, plot, clock, monkeypatch):
        # The fallback path: an engine with no ``_advance_timelines`` hook.
        monkeypatch.setattr(timeline_mod, "engine_drives_playback", lambda p: False)
        """The Presentation panel drives playback too; the two must not double it."""
        first, second = _timeline_panel(ws), _presentation_panel(ws)
        timeline = plot.active_panel.timeline
        timeline.play()
        timeline_mod.queue_playback(first)
        timeline_mod.queue_playback(second)
        ws.drain()
        clock.now += 0.1
        timeline_mod.queue_playback(first)
        timeline_mod.queue_playback(second)
        ws.drain()
        assert timeline.time == pytest.approx(0.1, abs=1e-6)

    def test_drawing_the_panel_drives_the_playhead(self, harness, ws, plot, clock, monkeypatch):
        """The fallback path: on a pre-hook engine, drawing the panel is what plays."""
        monkeypatch.setattr(timeline_mod, "engine_drives_playback", lambda p: False)
        panel = _timeline_panel(ws)
        timeline = plot.active_panel.timeline
        timeline.play()
        _frame(harness, panel, "Timeline")
        ws.drain()
        clock.now += 0.05
        _frame(harness, panel, "Timeline")
        ws.drain()
        assert timeline.time > 0.0


# ======================================================================================
# The Timeline panel
# ======================================================================================


class TestTimelinePanelDraws:
    def test_it_draws_an_empty_timeline(self, harness, ws, all_sections):
        _frame(harness, _timeline_panel(ws), "Timeline")

    def test_it_draws_many_tracks_and_keyframes(self, harness, ws, plot, all_sections):
        _populate(plot, keys=40)
        panel = _timeline_panel(ws)
        for _ in range(3):
            _frame(harness, panel, "Timeline")
            ws.drain()

    def test_it_draws_with_a_selected_track_and_keyframe(self, harness, ws, plot, all_sections):
        timeline = _populate(plot)
        panel = _timeline_panel(ws)
        panel._sel_track = timeline.tracks[0]
        panel._sel_key = timeline.tracks[0].keyframes[0]
        _frame(harness, panel, "Timeline")
        ws.drain()

    def test_it_draws_a_selected_keyframe_holding_an_array(self, harness, ws, plot, all_sections):
        """The read-only value branch — a different code path from the drag_float one."""
        timeline = _populate(plot)
        track = next(t for t in timeline.tracks if t.prop == "grid")
        panel = _timeline_panel(ws)
        panel._sel_track = track
        panel._sel_key = track.keyframes[0]
        _frame(harness, panel, "Timeline")
        ws.drain()

    def test_it_draws_while_playing(self, harness, ws, plot, all_sections):
        """The panel must render every playing state — the transport, the moving playhead
        and the highlighted frame — while the *engine* is what advances the clock.

        The playhead is stepped directly rather than by drawing, because on a current
        engine drawing the panel deliberately does not advance it (see ``TestPlayback``).
        """
        timeline = _populate(plot)
        timeline.set_duration(4.0)
        timeline.play()
        panel = _timeline_panel(ws)
        for i in range(3):
            plot._advance_timelines(500.0 + i * 0.05)
            _frame(harness, panel, "Timeline")
            ws.drain()
        assert timeline.time > 0.0
        assert timeline.playing is True

    def test_it_draws_a_muted_and_a_held_track(self, harness, ws, plot, all_sections):
        timeline = _populate(plot)
        timeline.tracks[0].enabled = False
        timeline.tracks[1].interpolate = False
        _frame(harness, _timeline_panel(ws), "Timeline")

    @pytest.mark.parametrize("loop", ["once", "loop", "pingpong"])
    def test_it_draws_in_every_loop_mode(self, harness, ws, plot, all_sections, loop):
        timeline = _populate(plot)
        timeline.set_loop(loop)
        _frame(harness, _timeline_panel(ws), "Timeline")

    def test_it_draws_with_layers_available_as_targets(self, harness, ws, plot, all_sections):
        plot.add_scatter(np.arange(5.0), np.arange(5.0), None)
        panel = _timeline_panel(ws)
        # Both cameras, then the one layer.
        layer_id = plot.scene.layers[0].layer_id
        targets = [target for _, target in panel._target_options()]
        assert targets == [timeline_mod.CAMERA_TARGET, timeline_mod.CAMERA2D_TARGET, layer_id]
        _frame(harness, panel, "Timeline")

    def test_it_draws_an_error(self, harness, ws, all_sections):
        panel = _timeline_panel(ws)
        panel._add_track("camera", "   ")
        assert panel._error
        _frame(harness, panel, "Timeline")

    def test_the_empty_state_draws_in_both_dimensionalities(self, harness, ws, plot, all_sections):
        """Empty is the state a first-time user sees, and it carries the two big buttons."""
        panel = _timeline_panel(ws)
        assert plot.active_panel.timeline.tracks == []
        _frame(harness, panel, "Timeline")
        plot.set_ndim(3)
        _frame(harness, panel, "Timeline")

    @pytest.mark.parametrize("scope", timeline_mod.CAPTURE_SCOPES)
    def test_it_draws_in_every_capture_scope(self, harness, ws, plot, all_sections, scope):
        plot.add_scatter(np.arange(5.0), np.arange(5.0), None)
        panel = _timeline_panel(ws)
        panel._capture_scope = scope
        _frame(harness, panel, "Timeline")

    def test_it_draws_the_property_picker_and_the_free_text_escape(
        self, harness, ws, plot, all_sections
    ):
        panel = _timeline_panel(ws)
        _frame(harness, panel, "Timeline")
        panel._custom_prop = True
        _frame(harness, panel, "Timeline")

    @pytest.mark.parametrize("preset", timeline_mod.PRESETS, ids=lambda p: p.key)
    def test_it_draws_with_every_preset_selected(self, harness, ws, plot, all_sections, preset):
        """Both branches of the picker: the Build button, and the "needs a ..." note."""
        plot.add_scatter(np.arange(5.0), np.arange(5.0), None)
        panel = _timeline_panel(ws)
        panel._preset = preset.key
        _frame(harness, panel, "Timeline")
        plot.set_ndim(3)
        _frame(harness, panel, "Timeline")

    def test_it_draws_the_preset_section_once_tracks_exist(self, harness, ws, plot, all_sections):
        _populate(plot)
        panel = _timeline_panel(ws)
        panel._preset_from_playhead = True
        _frame(harness, panel, "Timeline")

    def test_the_property_options_follow_the_target(self, ws, plot):
        assert "azim" in timeline_mod._prop_options(timeline_mod.CAMERA_TARGET)
        assert "zoom_x" in timeline_mod._prop_options(timeline_mod.CAMERA2D_TARGET)
        assert "alpha" in timeline_mod._prop_options(12345)


class TestTimelinePanelActions:
    """Every mutation must be queued: not before the drain, applied after it."""

    def test_seek_is_queued(self, ws, plot):
        timeline = plot.active_panel.timeline
        timeline.set_duration(4.0)
        panel = _timeline_panel(ws)
        panel.seek(1.5)
        assert timeline.time == pytest.approx(0.0), "mutation must not happen in draw"
        ws.drain()
        assert timeline.time == pytest.approx(1.5)

    def test_seek_snaps_to_a_frame_when_asked(self, ws, plot):
        timeline = plot.active_panel.timeline
        timeline.set_fps(10.0)
        panel = _timeline_panel(ws)
        panel.seek(1.04)
        ws.drain()
        assert timeline.time == pytest.approx(1.0)
        panel._snap = False
        panel.seek(1.04)
        ws.drain()
        assert timeline.time == pytest.approx(1.04)

    def test_frame_stepping_is_queued(self, ws, plot):
        timeline = plot.active_panel.timeline
        timeline.set_fps(10.0)
        panel = _timeline_panel(ws)
        panel.step_frames(3)
        assert timeline.frame_index() == 0
        ws.drain()
        assert timeline.frame_index() == 3
        panel.step_frames(-1)
        ws.drain()
        assert timeline.frame_index() == 2

    def test_play_toggle_and_stop_are_queued(self, ws, plot):
        timeline = plot.active_panel.timeline
        panel = _timeline_panel(ws)
        panel.toggle_play()
        assert timeline.playing is False
        ws.drain()
        assert timeline.playing is True

        timeline.seek(1.0)
        panel.stop()
        assert timeline.playing is True
        ws.drain()
        assert timeline.playing is False
        assert timeline.time == pytest.approx(0.0)

    def test_duration_fps_and_loop_go_through_their_setters(self, ws, plot):
        timeline = plot.active_panel.timeline
        panel = _timeline_panel(ws)
        panel._set_field("duration", 12.0)
        panel._set_field("fps", 24.0)
        panel._set_field("loop", "pingpong")
        panel._set_field("speed", 2.0)
        assert timeline.duration != 12.0
        ws.drain()
        assert timeline.duration == pytest.approx(12.0)
        assert timeline.fps == pytest.approx(24.0)
        assert timeline.loop == "pingpong"
        assert timeline.speed == pytest.approx(2.0)

    def test_fit_duration_is_queued(self, ws, plot):
        timeline = plot.active_panel.timeline
        timeline.key("camera", "azim", 0.0, time=17.0)
        panel = _timeline_panel(ws)
        panel._fit_duration()
        assert timeline.duration < 17.0
        ws.drain()
        assert timeline.duration == pytest.approx(17.0)

    def test_adding_and_removing_a_track_is_queued(self, ws, plot):
        timeline = plot.active_panel.timeline
        panel = _timeline_panel(ws)
        panel._add_track("camera", "azim")
        assert timeline.tracks == []
        ws.drain()
        assert len(timeline.tracks) == 1

        panel._remove_track(timeline.tracks[0])
        assert len(timeline.tracks) == 1
        ws.drain()
        assert timeline.tracks == []

    def test_a_track_needs_a_property_name(self, ws, plot):
        panel = _timeline_panel(ws)
        panel._add_track("camera", "")
        ws.drain()
        assert plot.active_panel.timeline.tracks == []
        assert "property" in panel._error

    def test_track_flags_are_queued(self, ws, plot):
        timeline = _populate(plot)
        track = timeline.tracks[0]
        panel = _timeline_panel(ws)
        panel._set_track_flag(track, "enabled", False)
        assert track.enabled is True
        ws.drain()
        assert track.enabled is False

    def test_keying_the_playhead_records_the_live_value(self, ws, plot):
        plot.camera3d.azim = 33.0
        timeline = plot.active_panel.timeline
        timeline.seek(0.0)
        panel = _timeline_panel(ws)
        panel._add_track("camera", "azim")
        ws.drain()
        panel._sel_track = timeline.tracks[0]
        panel.key_at_playhead()
        assert timeline.tracks[0].keyframes == []
        ws.drain()
        assert timeline.tracks[0].keyframes[0].value == pytest.approx(33.0)

    def test_keying_with_no_selection_keys_every_track(self, ws, plot):
        timeline = plot.active_panel.timeline
        panel = _timeline_panel(ws)
        panel._add_track("camera", "azim")
        panel._add_track("camera", "elev")
        ws.drain()
        assert panel.key_at_playhead() == 2
        ws.drain()
        assert all(len(t.keyframes) == 1 for t in timeline.tracks)

    def test_capture_is_queued(self, ws, plot):
        plot.set_ndim(3)
        timeline = plot.active_panel.timeline
        panel = _timeline_panel(ws)
        assert panel.capture() > 0
        assert timeline.tracks == []
        ws.drain()
        assert timeline.tracks

    def test_capture_returns_what_it_recorded(self, ws, plot):
        plot.set_ndim(3)
        panel = _timeline_panel(ws)
        assert panel.capture() == panel.capture_count()

    def test_capture_honours_the_panels_snap_setting(self, ws, plot):
        plot.set_ndim(3)
        timeline = plot.active_panel.timeline
        timeline.set_fps(10.0)
        timeline.seek(0.31)
        panel = _timeline_panel(ws)
        panel._snap = True
        panel.capture()
        ws.drain()
        assert timeline.tracks[0].keyframes[0].time == pytest.approx(0.3)

    def test_an_unknown_capture_scope_is_reported_not_raised(self, ws, plot):
        panel = _timeline_panel(ws)
        assert panel.capture("nonsense") == 0
        assert panel._error

    def test_the_capture_scope_is_remembered(self, ws, plot):
        plot.add_scatter(np.arange(5.0), np.arange(5.0), None)
        panel = _timeline_panel(ws)
        panel._capture_scope = "layers"
        panel.capture()
        ws.drain()
        assert {t.target for t in plot.active_panel.timeline.tracks} == {
            plot.scene.layers[0].layer_id
        }

    def test_the_turntable_is_queued(self, ws, plot):
        plot.set_ndim(3)
        timeline = plot.active_panel.timeline
        panel = _timeline_panel(ws)
        panel.add_turntable()
        assert timeline.tracks == []
        ws.drain()
        assert [t.prop for t in timeline.tracks] == ["azim"]

    def test_building_a_preset_is_queued(self, ws, plot):
        plot.set_ndim(3)
        timeline = plot.active_panel.timeline
        panel = _timeline_panel(ws)
        assert panel.build_preset("grand_tour") is True
        assert timeline.tracks == []
        ws.drain()
        assert {t.prop for t in timeline.tracks} == {"elev", "azim", "roll"}

    def test_an_inapplicable_preset_is_refused_with_a_reason(self, ws, plot):
        panel = _timeline_panel(ws)
        assert plot.is_3d_scene() is False
        assert panel.build_preset("turntable") is False
        ws.drain()
        assert plot.active_panel.timeline.tracks == []
        assert "3D" in panel._error

    def test_an_unknown_preset_is_refused_rather_than_raised(self, ws, plot):
        panel = _timeline_panel(ws)
        assert panel.build_preset("moonwalk") is False
        assert panel._error

    def test_the_playhead_option_moves_where_a_preset_starts(self, ws, plot):
        plot.set_ndim(3)
        timeline = plot.active_panel.timeline
        timeline.set_duration(4.0)
        timeline.seek(2.0)
        panel = _timeline_panel(ws)
        panel._preset_from_playhead = True
        panel.build_preset("turntable")
        ws.drain()
        track = timeline.track(timeline_mod.CAMERA_TARGET, "azim", create=False)
        assert min(k.time for k in track.keyframes) == pytest.approx(2.0)

    def test_keying_grows_the_duration_to_cover_the_key(self, ws, plot):
        timeline = plot.active_panel.timeline
        timeline.set_duration(2.0)
        panel = _timeline_panel(ws)
        panel._add_track("camera", "azim")
        ws.drain()
        timeline.seek(2.0)
        panel.key_at_playhead()
        ws.drain()
        assert timeline.duration >= 2.0

    def test_removing_a_keyframe_is_queued(self, ws, plot):
        timeline = _populate(plot)
        track = timeline.tracks[0]
        key = track.keyframes[0]
        panel = _timeline_panel(ws)
        panel._sel_track, panel._sel_key = track, key
        panel._remove_key(track, key)
        assert key in track.keyframes
        ws.drain()
        assert key not in track.keyframes

    def test_moving_a_keyframe_is_queued_and_resorts(self, ws, plot):
        timeline = _populate(plot)
        track = timeline.track("camera", "azim")
        first = track.keyframes[0]
        panel = _timeline_panel(ws)
        panel._move_key(track, first, 3.4)
        assert first.time == pytest.approx(0.0)
        ws.drain()
        assert first.time == pytest.approx(3.4)
        # It was the first key and is now the last: identity, not index, is what tracked it.
        assert track.keyframes[-1] is first

    def test_moving_a_deleted_keyframe_is_a_no_op(self, ws, plot):
        timeline = _populate(plot)
        track = timeline.track("camera", "azim")
        orphan = Keyframe(time=0.0, value=0.0)
        panel = _timeline_panel(ws)
        panel._move_key(track, orphan, 1.0)
        ws.drain()
        assert orphan.time == pytest.approx(0.0)

    def test_editing_a_keyframe_field_is_queued(self, ws, plot):
        timeline = _populate(plot)
        key = timeline.tracks[0].keyframes[0]
        panel = _timeline_panel(ws)
        panel._set_key_field(key, "easing", "bounce")
        assert key.easing != "bounce"
        ws.drain()
        assert key.easing == "bounce"

    def test_every_easing_label_maps_back_to_a_name(self):
        """The picker round-trips through the labels, so a duplicate label would hide one."""
        assert len(set(EASING_LABELS.values())) == len(EASING_LABELS)

    def test_the_hit_test_finds_the_nearest_diamond(self, ws, plot):
        timeline = plot.active_panel.timeline
        timeline.set_duration(10.0)
        track = timeline.track("camera", "azim")
        track.add(0.0, 0.0)
        track.add(5.0, 1.0)
        panel = _timeline_panel(ws)
        # A 100px axis over 10s: t=5 sits at x=50.
        assert panel._key_near(timeline, track, 0.0, 100.0, 50.0) is track.keyframes[1]
        assert panel._key_near(timeline, track, 0.0, 100.0, 2.0) is track.keyframes[0]
        assert panel._key_near(timeline, track, 0.0, 100.0, 30.0) is None

    def test_selection_is_dropped_when_its_track_goes_away(self, ws, plot):
        timeline = _populate(plot)
        panel = _timeline_panel(ws)
        panel._sel_track = timeline.tracks[0]
        panel._sel_key = timeline.tracks[0].keyframes[0]
        timeline.tracks.clear()
        panel._sync_selection()
        assert panel._sel_track is None and panel._sel_key is None

    def test_selection_is_dropped_when_its_keyframe_goes_away(self, ws, plot):
        timeline = _populate(plot)
        track = timeline.tracks[0]
        panel = _timeline_panel(ws)
        panel._sel_track, panel._sel_key = track, track.keyframes[0]
        track.keyframes.clear()
        panel._sync_selection()
        assert panel._sel_track is track
        assert panel._sel_key is None


class TestPointerInteraction:
    """Drive real mouse input through the harness: scrubbing, and the keyframe drag.

    These are the two gestures the panel exists for, and neither is reachable from a
    state-only test: a hit test off by the width of the label gutter would still pass
    every other test in this file, and the panel would be unusable.

    The strip's y is *searched for* rather than hardcoded, so a layout change moves the
    test with the panel instead of breaking it.
    """

    def _frame(self, imgui, panel, mx, my, down):
        io = imgui.get_io()
        io.mouse_pos = (mx, my)
        io.mouse_down[0] = bool(down)
        imgui.new_frame()
        imgui.set_next_window_pos((0.0, 0.0))
        imgui.set_next_window_size((1200.0, 700.0))
        imgui.begin("Timeline")
        panel.draw()
        imgui.end()
        imgui.render()

    def _settle(self, imgui, panel, ws, frames=2):
        for _ in range(frames):
            self._frame(imgui, panel, -1000.0, -1000.0, False)
            ws.drain()

    def _click(self, imgui, panel, ws, x, y):
        for _ in range(2):
            self._frame(imgui, panel, x, y, True)
            ws.drain()
        self._frame(imgui, panel, -1000.0, -1000.0, False)
        ws.drain()

    def _dense_track(self, plot, keys=101):
        """A track whose diamonds are closer together than the hit radius.

        So that a click anywhere on the row lands on one, which is what lets the search
        below find the row without knowing the panel's geometry.
        """
        timeline = plot.active_panel.timeline
        timeline.set_duration(10.0)
        timeline.set_fps(10.0)
        track = timeline.track("camera", "azim")
        for i in range(keys):
            track.add(i * timeline.duration / (keys - 1), float(i))
        return timeline, track

    def test_clicking_the_scrubber_seeks_the_playhead(self, harness, ws, plot):
        timeline = plot.active_panel.timeline
        timeline.set_duration(10.0)
        timeline.set_fps(10.0)
        panel = _timeline_panel(ws)
        self._settle(harness, panel, ws)

        strip_y = None
        for y in range(20, 400, 2):
            timeline.seek(0.0)
            self._click(harness, panel, ws, 900.0, float(y))
            if timeline.time > 0.0:
                strip_y = float(y)
                break
        assert strip_y is not None, "the scrubber was not hittable anywhere"

        timeline.seek(0.0)
        self._click(harness, panel, ws, 400.0, strip_y)
        near = timeline.time
        self._click(harness, panel, ws, 900.0, strip_y)
        far = timeline.time
        assert 0.0 < near < far <= timeline.duration

    def test_dragging_a_diamond_retimes_the_keyframe(self, harness, ws, plot):
        timeline, track = self._dense_track(plot)
        panel = _timeline_panel(ws)
        self._settle(harness, panel, ws)

        row_y = None
        for y in range(20, 460, 2):
            panel._sel_key = None
            self._click(harness, panel, ws, 700.0, float(y))
            if panel._sel_key is not None:
                row_y = float(y)
                break
        assert row_y is not None, "no keyframe row was hittable"

        key = panel._sel_key
        before = key.time
        self._frame(harness, panel, 700.0, row_y, True)
        ws.drain()
        for step in range(1, 6):
            pending = key.time
            self._frame(harness, panel, 700.0 + step * 20.0, row_y, True)
            # The drag is queued like everything else: nothing moves until the drain.
            assert key.time == pytest.approx(pending), "a drag frame mutated the keyframe"
            ws.drain()
            assert key.time > pending, "the queued move did not land"
        self._frame(harness, panel, -1000.0, -1000.0, False)
        ws.drain()

        assert key.time > before + 0.3, "the drag did not retime the keyframe"
        assert panel._drag_key is None, "the drag must end when the button is released"
        times = [k.time for k in track.keyframes]
        assert times == sorted(times), "the track must stay sorted after a drag"

    def test_clicking_an_empty_part_of_a_row_selects_the_track_and_scrubs(self, harness, ws, plot):
        timeline = plot.active_panel.timeline
        timeline.set_duration(10.0)
        timeline.set_fps(10.0)
        track = timeline.track("camera", "azim")
        track.add(0.0, 0.0)  # one key, at the far left: the rest of the row is empty
        panel = _timeline_panel(ws)
        self._settle(harness, panel, ws)

        for y in range(20, 460, 2):
            panel._sel_track = None
            timeline.seek(0.0)
            self._click(harness, panel, ws, 900.0, float(y))
            if panel._sel_track is track:
                assert panel._sel_key is None
                assert timeline.time > 0.0
                return
        pytest.fail("no keyframe row was hittable")


# ======================================================================================
# The Presentation panel
# ======================================================================================


class TestPresentationPanelDraws:
    def test_it_draws_with_no_scenes(self, harness, ws, all_sections):
        _frame(harness, _presentation_panel(ws), "Presentation")

    def test_it_draws_a_scene_list(self, harness, ws, plot, all_sections):
        timeline = _populate(plot)
        timeline.auto_scenes(4)
        panel = _presentation_panel(ws)
        for _ in range(3):
            _frame(harness, panel, "Presentation")
            ws.drain()

    def test_it_draws_while_presenting(self, harness, ws, plot, all_sections, clock):
        timeline = _populate(plot)
        timeline.auto_scenes(3)
        panel = _presentation_panel(ws)
        panel.enter_presentation()
        ws.drain()
        assert panel._presenting is True
        for _ in range(3):
            clock.now += 0.05
            _frame(harness, panel, "Presentation")
            ws.drain()

    def test_it_draws_a_scene_with_a_note(self, harness, ws, plot, all_sections):
        timeline = _populate(plot)
        timeline.add_scene("Intro", 0.0, 1.0, description="Why any of this matters")
        panel = _presentation_panel(ws)
        panel.enter_presentation()
        ws.drain()
        _frame(harness, panel, "Presentation")

    def test_it_draws_an_inline_rename(self, harness, ws, plot, all_sections):
        timeline = _populate(plot)
        timeline.auto_scenes(2)
        panel = _presentation_panel(ws)
        panel._begin_rename(0, timeline.scenes[0])
        _frame(harness, panel, "Presentation")
        _frame(harness, panel, "Presentation")

    def test_it_draws_an_error(self, harness, ws, plot, all_sections):
        panel = _presentation_panel(ws)
        panel.enter_presentation()
        assert panel._error
        _frame(harness, panel, "Presentation")

    def test_it_draws_a_held_and_an_unheld_scene(self, harness, ws, plot, all_sections):
        timeline = _populate(plot)
        timeline.auto_scenes(2)
        timeline.scenes[0].hold = False
        _frame(harness, _presentation_panel(ws), "Presentation")


class TestPresentationPanelActions:
    def test_presenting_needs_a_scene(self, ws, plot):
        panel = _presentation_panel(ws)
        panel.enter_presentation()
        ws.drain()
        assert panel._presenting is False
        assert "scenes" in panel._error

    def test_entering_presentation_is_queued(self, ws, plot):
        timeline = _populate(plot)
        timeline.auto_scenes(3)
        timeline.seek(2.5)
        panel = _presentation_panel(ws)
        panel.enter_presentation()
        assert timeline.playing is False, "mutation must not happen in draw"
        ws.drain()
        assert timeline.playing is True
        assert timeline.time == pytest.approx(timeline.scenes[panel._present_index].start)

    def test_exiting_presentation_pauses(self, ws, plot):
        timeline = _populate(plot)
        timeline.auto_scenes(2)
        panel = _presentation_panel(ws)
        panel.enter_presentation()
        ws.drain()
        panel.exit_presentation()
        assert timeline.playing is True
        ws.drain()
        assert timeline.playing is False
        assert panel._presenting is False

    def test_toggle_presentation_round_trips(self, ws, plot):
        timeline = _populate(plot)
        timeline.auto_scenes(2)
        panel = _presentation_panel(ws)
        panel.toggle_presentation()
        ws.drain()
        assert panel._presenting is True
        panel.toggle_presentation()
        ws.drain()
        assert panel._presenting is False

    def test_scene_stepping_is_queued_and_clamps(self, ws, plot):
        timeline = _populate(plot)
        timeline.auto_scenes(3)
        panel = _presentation_panel(ws)
        panel.next_scene()
        assert timeline.time == pytest.approx(0.0)
        ws.drain()
        assert timeline.time == pytest.approx(timeline.scenes[1].start)
        panel.next_scene()
        panel.next_scene()
        ws.drain()
        assert panel._present_index == 2
        panel.previous_scene()
        ws.drain()
        assert panel._present_index == 1

    def test_a_held_scene_stops_playback_at_its_end(self, harness, ws, plot, all_sections, clock):
        timeline = _populate(plot)
        timeline.auto_scenes(2)
        panel = _presentation_panel(ws)
        panel.enter_presentation()
        ws.drain()
        assert timeline.playing is True
        # Land the playhead exactly on the first scene's end and draw one frame.
        timeline.seek(timeline.scenes[0].end)
        _frame(harness, panel, "Presentation")
        ws.drain()
        assert timeline.playing is False
        assert timeline.time == pytest.approx(timeline.scenes[0].end)

    def test_an_unheld_scene_runs_on(self, harness, ws, plot, all_sections, clock):
        timeline = _populate(plot)
        timeline.auto_scenes(2)
        timeline.scenes[0].hold = False
        panel = _presentation_panel(ws)
        panel.enter_presentation()
        ws.drain()
        timeline.seek(timeline.scenes[0].end)
        _frame(harness, panel, "Presentation")
        ws.drain()
        assert timeline.playing is True

    def test_a_new_scene_starts_at_the_playhead(self, ws, plot):
        timeline = _populate(plot)
        timeline.seek(1.0)
        panel = _presentation_panel(ws)
        panel._new_scene()
        assert timeline.scenes == []
        ws.drain()
        assert len(timeline.scenes) == 1
        assert timeline.scenes[0].start == pytest.approx(1.0)

    def test_a_new_scene_butts_against_the_next_one(self, ws, plot):
        timeline = _populate(plot)
        timeline.add_scene("Later", 2.0, 3.0)
        timeline.seek(1.0)
        panel = _presentation_panel(ws)
        panel._new_scene()
        ws.drain()
        created = next(s for s in timeline.scenes if s.name != "Later")
        assert created.end == pytest.approx(2.0)

    def test_splitting_at_the_playhead_is_queued(self, ws, plot):
        timeline = _populate(plot)
        timeline.add_scene("Whole", 0.0, 2.0)
        timeline.seek(1.0)
        panel = _presentation_panel(ws)
        panel._split()
        assert len(timeline.scenes) == 1
        ws.drain()
        assert len(timeline.scenes) == 2
        assert timeline.scenes[0].end == pytest.approx(1.0)
        assert timeline.scenes[1].start == pytest.approx(1.0)

    def test_splitting_outside_a_scene_is_refused(self, ws, plot):
        timeline = _populate(plot)
        timeline.add_scene("Whole", 0.0, 1.0)
        timeline.seek(2.0)
        panel = _presentation_panel(ws)
        panel._split()
        ws.drain()
        assert len(timeline.scenes) == 1
        assert panel._error

    def test_auto_scenes_is_queued(self, ws, plot):
        timeline = _populate(plot)
        panel = _presentation_panel(ws)
        panel._auto_scenes(5)
        assert timeline.scenes == []
        ws.drain()
        assert len(timeline.scenes) == 5
        assert timeline.scenes[-1].end == pytest.approx(timeline.duration)

    def test_deleting_a_scene_is_queued_and_by_identity(self, ws, plot):
        timeline = _populate(plot)
        timeline.add_scene("Same", 0.0, 1.0)
        timeline.add_scene("Same", 1.0, 2.0)
        second = timeline.scenes[1]
        panel = _presentation_panel(ws)
        panel._delete(second)
        assert len(timeline.scenes) == 2
        ws.drain()
        assert len(timeline.scenes) == 1
        assert timeline.scenes[0] is not second

    def test_reordering_swaps_the_spans_and_keeps_the_durations(self, ws, plot):
        timeline = _populate(plot)
        timeline.add_scene("A", 0.0, 1.0)
        timeline.add_scene("B", 1.0, 3.0)
        panel = _presentation_panel(ws)
        panel._swap(1)
        assert [s.name for s in timeline.scenes] == ["A", "B"]
        ws.drain()
        assert [s.name for s in timeline.scenes] == ["B", "A"]
        assert timeline.scenes[0].duration == pytest.approx(2.0)
        assert timeline.scenes[1].duration == pytest.approx(1.0)

    def test_reordering_the_first_scene_is_a_no_op(self, ws, plot):
        timeline = _populate(plot)
        timeline.auto_scenes(2)
        names = [s.name for s in timeline.scenes]
        panel = _presentation_panel(ws)
        panel._swap(0)
        ws.drain()
        assert [s.name for s in timeline.scenes] == names

    def test_renaming_is_queued(self, ws, plot):
        timeline = _populate(plot)
        timeline.auto_scenes(2)
        scene = timeline.scenes[0]
        panel = _presentation_panel(ws)
        panel._begin_rename(0, scene)
        panel._commit_rename(scene, "  Opening  ")
        assert scene.name != "Opening"
        ws.drain()
        assert scene.name == "Opening"
        assert panel._rename_index is None

    def test_an_empty_rename_is_ignored(self, ws, plot):
        timeline = _populate(plot)
        timeline.auto_scenes(1)
        scene = timeline.scenes[0]
        before = scene.name
        panel = _presentation_panel(ws)
        panel._commit_rename(scene, "   ")
        ws.drain()
        assert scene.name == before

    def test_retiming_a_scene_cannot_invert_it(self, ws, plot):
        timeline = _populate(plot)
        timeline.add_scene("A", 0.0, 1.0)
        scene = timeline.scenes[0]
        panel = _presentation_panel(ws)
        panel._set_scene_span(scene, 2.0, 1.0)
        ws.drain()
        assert scene.end > scene.start

    def test_the_hold_flag_is_queued(self, ws, plot):
        timeline = _populate(plot)
        timeline.auto_scenes(1)
        scene = timeline.scenes[0]
        panel = _presentation_panel(ws)
        panel._set_scene_field(scene, "hold", False)
        assert scene.hold is True
        ws.drain()
        assert scene.hold is False

    def test_new_scene_names_do_not_collide(self, ws, plot):
        timeline = _populate(plot)
        panel = _presentation_panel(ws)
        for _ in range(3):
            panel._new_scene()
            ws.drain()
            timeline.seek(timeline.time + 0.5)
        assert len({s.name for s in timeline.scenes}) == len(timeline.scenes)


# ======================================================================================
# Workspace integration
# ======================================================================================


class TestWorkspaceIntegration:
    """The rail, the action table and the keymap, on a real workspace."""

    @pytest.fixture
    def workspace(self, harness):
        plot = GPULinePlot()
        plot.options.enable_hud = True
        ws = plot.hud.workspace
        assert ws is not None
        return ws

    def test_both_animation_panels_are_registered(self, workspace):
        assert "timeline" in workspace.panels
        assert "presentation" in workspace.panels

    def test_both_panels_have_a_first_run_layout_and_a_chord(self, workspace):
        from glplot.gui.workspace import _LAYOUT, _PANEL_CHORDS

        for key in ("timeline", "presentation"):
            assert key in _LAYOUT
            assert key in _PANEL_CHORDS
            assert workspace.registry.get(f"panel.{key}") is not None

    def test_the_animation_actions_exist_and_are_documented(self, workspace):
        actions = [a for a in workspace.registry._actions if a.category == "Animation"]
        assert len(actions) >= 10
        for action in actions:
            assert action.title and action.description
            assert action.icon

    def test_the_animation_group_covers_the_required_verbs(self, workspace):
        ids = {a.id for a in workspace.registry._actions if a.category == "Animation"}
        assert {
            "anim.timeline",
            "anim.presentation",
            "anim.play_pause",
            "anim.stop",
            "anim.next_frame",
            "anim.prev_frame",
            "anim.next_scene",
            "anim.prev_scene",
            "anim.add_key",
            "anim.capture",
        } <= ids

    def test_capture_is_bound_to_a_bare_k(self, workspace):
        from glplot.gui import keys

        action = workspace.registry.get("anim.capture")
        assert action.chord == keys.parse("K")
        # Bare, therefore suppressed while a text field has the keyboard: typing "k" into a
        # layer name must not silently key the camera.
        assert keys.is_text_safe(action.chord) is False

    def test_capture_works_on_an_empty_timeline(self, workspace):
        """The point of the whole gesture: no track has to exist first."""
        plot = workspace.plot
        plot.set_ndim(3)
        timeline = plot.active_panel.timeline
        assert timeline.tracks == []
        workspace.registry.get("anim.capture").run()
        workspace.queue.drain(plot)
        assert timeline.tracks
        assert "Captured" in workspace._status

    def test_capture_is_available_before_anything_is_animated(self, workspace):
        """Unlike `anim.add_key`, which needs a track to key."""
        assert workspace.registry.get("anim.add_key").is_enabled() is False
        assert workspace.registry.get("anim.capture").is_enabled() is True

    def test_every_preset_is_in_the_command_palette(self, workspace):
        """A preset only the panel knows about is a feature only its finders can use."""
        from glplot.gui.panels.timeline import PRESETS

        ids = {a.id for a in workspace.registry.all()}
        for preset in PRESETS:
            assert f"anim.preset_{preset.key}" in ids, preset.key

    def test_preset_actions_are_gated_on_the_figure(self, workspace):
        plot = workspace.plot
        assert workspace.registry.get("anim.preset_turntable").is_enabled() is False
        assert workspace.registry.get("anim.preset_reveal").is_enabled() is False
        plot.set_ndim(3)
        plot.add_scatter(np.arange(5.0), np.arange(5.0), None)
        assert workspace.registry.get("anim.preset_turntable").is_enabled() is True
        assert workspace.registry.get("anim.preset_reveal").is_enabled() is True

    def test_running_a_preset_action_builds_it_and_reports(self, workspace):
        plot = workspace.plot
        plot.set_ndim(3)
        timeline = plot.active_panel.timeline
        workspace.registry.get("anim.preset_turntable").run()
        workspace.queue.drain(plot)
        assert [t.prop for t in timeline.tracks] == ["azim"]
        assert "turntable" in workspace._status

    def test_an_inapplicable_preset_action_reports_instead_of_building(self, workspace):
        plot = workspace.plot
        assert plot.is_3d_scene() is False
        workspace.registry.get("anim.preset_turntable").run()
        workspace.queue.drain(plot)
        assert plot.active_panel.timeline.tracks == []
        assert "3D" in workspace._status

    def test_every_preset_action_runs_without_raising(self, workspace):
        """Including the ones that do not apply — a greyed action is still invocable."""
        plot = workspace.plot
        plot.add_scatter(np.arange(5.0), np.arange(5.0), None)
        for action in [a for a in workspace.registry.all() if a.id.startswith("anim.preset_")]:
            action.run()
            workspace.queue.drain(plot)

    def test_the_number_row_is_bound_to_the_standard_angles(self, workspace):
        from glplot.core.camera3d import STANDARD_VIEWS
        from glplot.gui import keys
        from glplot.gui.workspace import _VIEW_DIGITS

        assert len(_VIEW_DIGITS) == 9
        for index, (view_key, _, _) in enumerate(_VIEW_DIGITS):
            assert view_key in STANDARD_VIEWS, view_key
            action = workspace.registry.get(f"view3d.preset_{view_key}")
            assert action.chord == keys.parse(str(index + 1)), view_key
        assert workspace.registry.get("view3d.reset").chord == keys.parse("0")

    def test_the_angle_chords_are_suppressed_while_typing(self, workspace):
        from glplot.gui import keys
        from glplot.gui.workspace import _VIEW_DIGITS

        for view_key, _, _ in _VIEW_DIGITS:
            chord = workspace.registry.get(f"view3d.preset_{view_key}").chord
            assert keys.is_text_safe(chord) is False, view_key

    def test_the_angle_chords_are_dead_in_a_2d_figure(self, workspace):
        from glplot.gui.workspace import _VIEW_DIGITS

        assert workspace.plot.is_3d_scene() is False
        for view_key, _, _ in _VIEW_DIGITS:
            assert workspace.registry.get(f"view3d.preset_{view_key}").is_enabled() is False

    def test_pressing_a_number_actually_moves_the_camera(self, workspace):
        from glplot.core.camera3d import STANDARD_VIEWS
        from glplot.gui.workspace import _VIEW_DIGITS

        plot = workspace.plot
        plot.set_ndim(3)
        for view_key, _, _ in _VIEW_DIGITS:
            workspace.registry.get(f"view3d.preset_{view_key}").run()
            workspace.queue.drain(plot)
            elev, azim, roll = STANDARD_VIEWS[view_key]
            assert plot.camera3d.elev == pytest.approx(elev), view_key
            assert plot.camera3d.azim == pytest.approx(azim), view_key
            assert plot.camera3d.roll == pytest.approx(roll), view_key

    def test_the_angle_then_capture_workflow_builds_a_camera_move(self, workspace):
        """1, K, move the playhead, 5, K — the workflow the two bindings exist for."""
        plot = workspace.plot
        plot.set_ndim(3)
        timeline = plot.active_panel.timeline

        workspace.registry.get("view3d.preset_front").run()
        workspace.queue.drain(plot)
        workspace.registry.get("anim.capture").run()
        workspace.queue.drain(plot)

        timeline.seek(2.0)
        workspace.registry.get("view3d.preset_top").run()
        workspace.queue.drain(plot)
        workspace.registry.get("anim.capture").run()
        workspace.queue.drain(plot)

        from glplot.gui.panels.timeline import CAMERA_TARGET

        elev = timeline.track(CAMERA_TARGET, "elev", create=False)
        assert elev is not None and len(elev.keyframes) == 2
        # Front is elev 0, top is elev 90: halfway through, the camera is genuinely tilted.
        assert 0.0 < elev.value_at(1.0) < 90.0

    def test_no_two_actions_share_a_chord_except_the_known_one(self, workspace):
        """A duplicate chord means only the earlier action ever runs."""
        from collections import Counter

        counts = Counter(str(a.chord) for a in workspace.registry._actions if a.chord)
        duplicates = {chord for chord, n in counts.items() if n > 1}
        offenders = {
            chord: [a.id for a in workspace.registry._actions if str(a.chord) == chord]
            for chord in duplicates
        }
        # `panel.dynamics` vs `view.autoscale` on Mod+0 predates this work.
        assert all(
            set(ids) == {"panel.dynamics", "view.autoscale"} for ids in offenders.values()
        ), offenders

    def test_play_pause_is_suppressed_while_a_text_field_has_focus(self, workspace):
        """Space is the natural transport binding only because a bare key is text-unsafe."""
        from glplot.gui import keys

        action = workspace.registry.get("anim.play_pause")
        assert action.chord == keys.parse("Space")
        assert keys.is_text_safe(action.chord) is False

    def test_the_presentation_chord_survives_a_text_field(self, workspace):
        from glplot.gui import keys

        assert keys.is_text_safe(workspace.registry.get("anim.present").chord) is True

    def test_every_animation_action_runs_without_raising(self, workspace):
        plot = workspace.plot
        timeline = plot.active_panel.timeline
        timeline.key("camera", "azim", 0.0, time=0.0)
        timeline.key("camera", "azim", 90.0, time=1.0)
        timeline.auto_scenes(3)
        for action in [a for a in workspace.registry._actions if a.category == "Animation"]:
            action.run()
            workspace.queue.drain(plot)

    def test_the_scene_actions_are_disabled_without_scenes(self, workspace):
        gated = [
            workspace.registry.get(name)
            for name in ("anim.next_scene", "anim.prev_scene", "anim.present")
        ]
        assert all(a.is_enabled() is False for a in gated)
        workspace.plot.active_panel.timeline.auto_scenes(2)
        assert all(a.is_enabled() is True for a in gated)

    def test_adding_a_keyframe_is_disabled_without_a_track(self, workspace):
        action = workspace.registry.get("anim.add_key")
        assert action.is_enabled() is False
        workspace.plot.active_panel.timeline.track("camera", "azim")
        assert action.is_enabled() is True

    def test_the_workspace_draws_with_both_panels_open(self, workspace, harness):
        """The real shell: menu, rail, and both animation panels in their own windows."""
        plot = workspace.plot
        plot.add_scatter(np.arange(20.0), np.arange(20.0), None)
        layer_id = plot.scene.layers[0].layer_id
        timeline = plot.active_panel.timeline
        timeline.key(layer_id, "alpha", 1.0, time=0.0)
        timeline.key(layer_id, "alpha", 0.2, time=1.0)
        timeline.auto_scenes(2)
        workspace.open["timeline"] = True
        workspace.open["presentation"] = True
        for _ in range(3):
            harness.new_frame()
            workspace.draw()
            harness.render()
            assert harness.get_draw_data() is not None
            workspace.queue.drain(plot)

    def test_the_play_action_opens_the_timeline_panel(self, workspace):
        workspace.open["timeline"] = False
        workspace.registry.get("anim.play_pause").run()
        workspace.queue.drain(workspace.plot)
        assert workspace.open["timeline"] is True
        assert workspace.plot.active_panel.timeline.playing is True
