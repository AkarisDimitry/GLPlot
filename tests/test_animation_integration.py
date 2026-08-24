"""End-to-end animation: timeline → applier → engine → the actual scene.

Every layer of the animation stack has its own suite. None of them crosses the seams, and
the seams are where this can break: the engine's per-frame hook has to find the applier,
the applier has to resolve the timeline's opaque targets onto real layers and cameras, and
the 3D camera has to reach each layer's metadata copy or the picture does not move even
though every individual component passed its own tests.

No window and no GL — ``GPULinePlot()`` constructs headless, and the loop's per-frame step
is driven directly rather than by running ``run()``.
"""

from __future__ import annotations

import numpy as np
import pytest

import glplot.animation as animation
from glplot.engine import GPULinePlot
from glplot.gui import layerops, layerops3d

#: Several tests below build a ``FuncAnimation`` without binding it (they only care about
#: what it did to ``fig.active_panel.timeline``), so it warns about being deleted without
#: rendering at an arbitrary later GC point -- possibly attributed to an unrelated test.
#: Same call as ``test_animation_api.py``'s module-level one, and for the same reason.
pytestmark = pytest.mark.filterwarnings("ignore:Animation was deleted without rendering")


@pytest.fixture
def plot3d():
    """A 3D figure with one curve layer."""
    fig = GPULinePlot()
    fig.set_ndim(3)
    t = np.linspace(0.0, 8.0, 200)
    layer = layerops3d.add_xyz_layer(
        fig, np.cos(t), np.sin(t), t / 4.0, kind="line3d", label="curve"
    )
    return fig, layer


def _run(fig, seconds: float, step: float = 0.05, t0: float = 100.0) -> None:
    """Drive the loop's animation step over ``seconds`` of wall clock.

    The first call only seeds the clock (there is no previous timestamp to difference
    against), which is why the loop starts at 1.
    """
    fig._advance_timelines(t0)
    for i in range(1, int(seconds / step) + 1):
        fig._advance_timelines(t0 + i * step)


class TestFrameCallbacks:
    """The hook that makes live playback possible at all."""

    def test_a_callback_runs_every_frame(self):
        fig = GPULinePlot()
        seen = []
        fig.add_frame_callback(lambda t: seen.append(t))
        fig._run_frame_callbacks(1.0)
        fig._run_frame_callbacks(2.0)
        assert seen == [1.0, 2.0]

    def test_a_callback_can_be_removed(self):
        fig = GPULinePlot()
        fn = fig.add_frame_callback(lambda t: None)
        assert fig.remove_frame_callback(fn) is True
        assert fig.remove_frame_callback(fn) is False

    def test_a_raising_callback_is_dropped_not_propagated(self):
        """A broken user animation must not make the window unclosable."""
        fig = GPULinePlot()
        survivors = []

        def boom(t):
            raise RuntimeError("user bug")

        fig.add_frame_callback(boom)
        fig.add_frame_callback(lambda t: survivors.append(t))
        fig._run_frame_callbacks(1.0)  # must not raise
        fig._run_frame_callbacks(2.0)
        assert len(fig._frame_callbacks) == 1
        assert survivors == [1.0, 2.0]

    def test_callbacks_mark_the_scene_dirty(self):
        fig = GPULinePlot()
        fig.frame.dirty_scene = False
        fig.add_frame_callback(lambda t: None)
        fig._run_frame_callbacks(1.0)
        assert fig.frame.dirty_scene is True


def _engine_frame(fig, t: float) -> None:
    """Simulate one iteration of ``GPULinePlot.run()``'s loop body, headless.

    The real loop calls both in this order every frame (``engine.py``): the timeline moves
    first, the frame callbacks read wherever it landed second. A test that only drives
    ``_run_frame_callbacks`` never moves ``fig.active_panel.timeline.time`` at all, since
    that is ``_advance_timelines``'s job alone now that a bound ``FuncAnimation`` reads the
    timeline instead of keeping its own clock.
    """
    fig._advance_timelines(t)
    fig._run_frame_callbacks(t)


class TestFuncAnimationLivePlayback:
    """``glplot.animation.TimedAnimation`` bound to ``fig.active_panel.timeline``.

    A ``FuncAnimation`` built against a live GLPlot figure hands its frames to the same
    playhead the GUI's Timeline panel already has transport controls for (see
    :meth:`glplot.animation.TimedAnimation._start_live_playback`), rather than keeping an
    independent clock the panel cannot reach. Driven headless via :func:`_engine_frame` --
    no window, no GL needed.
    """

    def test_registers_exactly_one_frame_callback(self):
        fig = GPULinePlot()
        animation.FuncAnimation(fig, lambda i: None, frames=3, interval=10)
        assert len(fig._frame_callbacks) == 1

    def test_construction_configures_and_plays_the_panel_timeline(self):
        fig = GPULinePlot()
        animation.FuncAnimation(fig, lambda i: None, frames=5, interval=20, repeat=True)
        timeline = fig.active_panel.timeline
        assert timeline.playing is True
        assert timeline.loop == "loop"
        # 5 frames at 20ms apart span 80ms; an untouched fresh timeline may as well match
        # exactly rather than leave the scrub bar longer than the animation.
        assert timeline.duration == pytest.approx(0.08)

    def test_non_repeating_sets_loop_mode_once(self):
        fig = GPULinePlot()
        animation.FuncAnimation(fig, lambda i: None, frames=5, interval=20, repeat=False)
        assert fig.active_panel.timeline.loop == "once"

    def test_existing_keyframes_are_not_truncated(self):
        """A timeline with hand-authored content keeps its own (longer) length."""
        fig = GPULinePlot()
        timeline = fig.active_panel.timeline
        timeline.key("camera3d", "elev", 0.0, time=0.0)
        timeline.key("camera3d", "elev", 30.0, time=10.0)
        timeline.fit_duration()
        assert timeline.duration >= 10.0
        animation.FuncAnimation(fig, lambda i: None, frames=3, interval=20)
        assert timeline.duration >= 10.0

    def test_steps_forward_as_the_timeline_plays(self):
        fig = GPULinePlot()
        seen = []
        animation.FuncAnimation(
            fig, lambda i: seen.append(i), frames=100, interval=10, repeat=False
        )
        t = 0.0
        _engine_frame(fig, t)  # seeds the clock and draws frame 0
        for _ in range(30):
            t += 0.01  # exactly one animation frame of wall time per tick
            _engine_frame(fig, t)
        assert seen == sorted(seen), "frame index must never move backward while playing forward"
        assert seen[0] == 0
        assert seen[-1] >= 10

    def test_repeat_true_replays_frame_zero(self):
        fig = GPULinePlot()
        seen = []
        animation.FuncAnimation(fig, lambda i: seen.append(i), frames=3, interval=10, repeat=True)
        t = 0.0
        _engine_frame(fig, t)
        for _ in range(60):
            t += 0.007  # a stride that does not line up with the loop boundary
            _engine_frame(fig, t)
        assert seen[0] == 0
        assert seen.count(0) > 1, "a looping animation must replay frame 0"

    def test_repeat_false_stops_advancing_after_the_last_frame(self):
        fig = GPULinePlot()
        seen = []
        animation.FuncAnimation(fig, lambda i: seen.append(i), frames=2, interval=10, repeat=False)
        t = 0.0
        _engine_frame(fig, t)
        for _ in range(50):
            t += 0.01
            _engine_frame(fig, t)
        assert seen == [0, 1]
        assert fig.active_panel.timeline.playing is False, "'once' must stop at the last frame"

    def test_pause_freezes_playback_and_resume_continues_it(self):
        fig = GPULinePlot()
        seen = []
        ani = animation.FuncAnimation(
            fig, lambda i: seen.append(i), frames=200, interval=10, repeat=False
        )
        t = 0.0
        _engine_frame(fig, t)
        t += 0.03
        _engine_frame(fig, t)
        ani.pause()
        assert fig.active_panel.timeline.playing is False
        frozen_at = len(seen)
        assert 0 < frozen_at < 200, "test setup should leave frames both seen and remaining"
        for _ in range(5):
            t += 0.05
            _engine_frame(fig, t)
        assert len(seen) == frozen_at, "a paused animation must not advance"
        ani.resume()
        assert fig.active_panel.timeline.playing is True
        t += 0.05
        _engine_frame(fig, t)
        assert len(seen) > frozen_at, "resuming must let it advance again"

    def test_the_timeline_panels_own_transport_controls_it(self):
        """Play/Pause/Stop/seek as the GUI's Timeline panel actually invokes them.

        No independent knowledge of ``ani`` is used here on purpose: this is exactly what
        pressing the panel's buttons and dragging its scrub bar does (see
        ``gui/panels/timeline.py``'s ``_toggle_play``/``_stop``/``_seek``), proving the
        animation is genuinely driven by the shared playhead, not by anything private to it.
        """
        fig = GPULinePlot()
        seen = []
        animation.FuncAnimation(fig, lambda i: seen.append(i), frames=50, interval=10, repeat=True)
        timeline = fig.active_panel.timeline

        t = 0.0
        _engine_frame(fig, t)
        for _ in range(10):
            t += 0.01
            _engine_frame(fig, t)
        assert len(seen) > 1

        timeline.toggle()  # the panel's Play/Pause button
        assert timeline.playing is False
        frozen_at = len(seen)
        for _ in range(5):
            t += 0.01
            _engine_frame(fig, t)
        assert len(seen) == frozen_at

        timeline.seek(0.2)  # dragging the scrub bar while paused
        _engine_frame(fig, t)
        assert seen[-1] == timeline.frame_index()

        timeline.stop()  # the panel's Stop button: pause + rewind
        _engine_frame(fig, t)
        assert seen[-1] == 0
        assert timeline.playing is False

    def test_a_real_matplotlib_figure_gets_no_frame_callback(self):
        """The dual-path guarantee: a real canvas keeps using its own timer, untouched."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as mpl_plt

        fig, ax = mpl_plt.subplots()
        ani = animation.FuncAnimation(fig, lambda i: None, frames=3, interval=10)
        try:
            assert not hasattr(fig, "_frame_callbacks")
            assert ani.event_source is not None
        finally:
            # Suppress the unrelated "deleted without rendering" warning: this test never
            # drives the animation, real or otherwise, on purpose.
            ani._draw_was_started = True
            mpl_plt.close(fig)


class TestTimelinePlayback:
    """The engine stepping a timeline from wall-clock deltas."""

    def test_a_paused_timeline_does_not_move(self, plot3d):
        fig, _layer = plot3d
        fig.active_panel.timeline.key("camera3d", "elev", 0.0, time=0.0)
        _run(fig, 1.0)
        assert fig.active_panel.timeline.time == 0.0

    def test_playback_advances_the_playhead(self, plot3d):
        fig, _layer = plot3d
        timeline = fig.active_panel.timeline
        timeline.set_duration(4.0)
        timeline.key("camera3d", "elev", 0.0, time=0.0)
        timeline.key("camera3d", "elev", 60.0, time=4.0)
        timeline.play()
        _run(fig, 1.0)
        assert timeline.time == pytest.approx(1.0, abs=0.11)

    def test_an_implausible_delta_is_ignored(self, plot3d):
        """A blocked loop or a clock jump must not skip half the animation."""
        fig, _layer = plot3d
        timeline = fig.active_panel.timeline
        timeline.set_duration(10.0)
        timeline.key("camera3d", "elev", 0.0, time=0.0)
        timeline.play()
        fig._advance_timelines(100.0)
        fig._advance_timelines(105.0)  # five seconds in one frame
        assert timeline.time == 0.0


class TestCameraAnimation:
    """A keyed camera must move the *picture*, not just the state object."""

    def test_the_3d_camera_moves(self, plot3d):
        fig, _layer = plot3d
        timeline = fig.active_panel.timeline
        timeline.set_duration(2.0)
        timeline.key("camera3d", "elev", 0.0, time=0.0, easing_name="linear")
        timeline.key("camera3d", "elev", 80.0, time=2.0, easing_name="linear")
        timeline.play()
        _run(fig, 1.0)
        assert 20.0 < fig.camera3d.elev < 70.0

    def test_every_layer_metadata_camera_follows(self, plot3d):
        """The renderer builds its matrix per layer from metadata, not from the camera.

        This is the seam that a component test cannot see: the camera state can be
        perfectly animated while every layer still draws under the old view.
        """
        fig, layer = plot3d
        timeline = fig.active_panel.timeline
        timeline.set_duration(2.0)
        timeline.key("camera3d", "azim", -45.0, time=0.0, easing_name="linear")
        timeline.key("camera3d", "azim", 90.0, time=2.0, easing_name="linear")
        timeline.play()
        _run(fig, 1.0)
        assert layer.metadata["camera"]["azim"] == pytest.approx(fig.camera3d.azim)

    def test_the_2d_camera_moves(self):
        fig = GPULinePlot()
        layerops.add_xy_layer(fig, np.arange(10.0), np.arange(10.0), kind="line", label="l")
        timeline = fig.active_panel.timeline
        timeline.set_duration(2.0)
        timeline.key("camera", "cx", 0.0, time=0.0, easing_name="linear")
        timeline.key("camera", "cx", 10.0, time=2.0, easing_name="linear")
        timeline.play()
        _run(fig, 1.0)
        assert 2.0 < fig.camera.cx < 8.0


class TestLayerAnimation:
    """A keyed layer property must reach the layer."""

    def test_alpha_fades(self, plot3d):
        fig, layer = plot3d
        timeline = fig.active_panel.timeline
        timeline.set_duration(2.0)
        timeline.key(layer.layer_id, "alpha", 0.0, time=0.0, easing_name="linear")
        timeline.key(layer.layer_id, "alpha", 1.0, time=2.0, easing_name="linear")
        # "once", not the default "loop": a looping timeline restarts the fade at 0 as soon
        # as it reaches the end, which is correct playback but not a monotone ramp.
        timeline.set_loop("once")
        timeline.play()

        # 40 steps of 0.05s covers the whole 2s fade; half of them would only reach 0.5.
        seen = []
        fig._advance_timelines(100.0)
        for i in range(1, 41):
            fig._advance_timelines(100.0 + i * 0.05)
            seen.append(layer.style.alpha)
        assert min(seen) < 0.1 and max(seen) > 0.9
        assert seen == sorted(seen), "a linear fade must be monotone"

    def test_a_deleted_layer_does_not_break_playback(self, plot3d):
        """A timeline outlives the layer it was keyed against."""
        fig, layer = plot3d
        timeline = fig.active_panel.timeline
        timeline.set_duration(2.0)
        timeline.key(layer.layer_id, "alpha", 0.0, time=0.0)
        timeline.key(layer.layer_id, "alpha", 1.0, time=2.0)
        timeline.play()
        layerops.remove_layer(fig, None, layer)
        _run(fig, 1.0)  # must not raise
        assert timeline.time > 0.0


class TestManimVerbs:
    """The primitives, driven through the engine rather than inspected as keyframes."""

    def test_orbit_and_fade_compose(self, plot3d):
        from glplot.anim import primitives as anim

        fig, layer = plot3d
        timeline = fig.active_panel.timeline
        timeline.set_duration(4.0)
        anim.Orbit(turns=1.0).apply(timeline, "camera3d", start=0.0, duration=4.0)
        anim.FadeIn().apply(timeline, layer.layer_id, start=0.0, duration=2.0)
        timeline.play()

        start_azim = fig.camera3d.azim
        alphas = []
        fig._advance_timelines(100.0)
        for i in range(1, 21):
            fig._advance_timelines(100.0 + i * 0.05)
            alphas.append(layer.style.alpha)

        assert fig.camera3d.azim != pytest.approx(start_azim)
        assert max(alphas) - min(alphas) > 0.3

    def test_succession_runs_one_after_another(self, plot3d):
        from glplot.anim import primitives as anim

        fig, layer = plot3d
        timeline = fig.active_panel.timeline
        # `succession` takes the plays as varargs, not as one sequence.
        end = anim.succession(
            timeline,
            (anim.FadeOut(), layer.layer_id, 1.0),
            (anim.FadeIn(), layer.layer_id, 1.0),
        )
        assert end == pytest.approx(2.0)
        # Out then in: the midpoint of the whole run is the darkest moment.
        assert timeline.evaluate(1.0)[(layer.layer_id, "alpha")] == pytest.approx(0.0, abs=1e-6)
        assert timeline.evaluate(2.0)[(layer.layer_id, "alpha")] == pytest.approx(1.0, abs=1e-6)


class TestPerPanelIsolation:
    """A split figure must animate one region and leave the other alone."""

    def test_only_the_playing_panel_advances(self):
        from glplot.core import layout

        fig = GPULinePlot()
        fig.set_panels(layout.grid(1, 2))
        first, second = fig.panels
        for panel in (first, second):
            panel.timeline.set_duration(4.0)
            panel.timeline.key("camera", "cx", 0.0, time=0.0, easing_name="linear")
            panel.timeline.key("camera", "cx", 10.0, time=4.0, easing_name="linear")
        first.timeline.play()

        _run(fig, 1.0)
        assert first.timeline.time > 0.0
        assert second.timeline.time == 0.0
        assert first.camera.cx > 0.0
        assert second.camera.cx == 0.0

    def test_the_active_panel_is_restored_after_applying(self):
        from glplot.core import layout

        fig = GPULinePlot()
        fig.set_panels(layout.grid(1, 2))
        fig.active_panel_index = 1
        fig.panels[0].timeline.set_duration(2.0)
        fig.panels[0].timeline.key("camera", "cx", 0.0, time=0.0)
        fig.panels[0].timeline.key("camera", "cx", 5.0, time=2.0)
        fig.panels[0].timeline.play()
        _run(fig, 0.5)
        assert fig.active_panel_index == 1


class TestDiscreteProgression:
    """A stepped simulation played back frame by frame, never blended."""

    def test_a_game_of_life_run_plays_without_blending(self, plot3d):
        from glplot.anim import processes

        fig, _layer = plot3d
        life = processes.process("life")
        params = life.resolve({"width": 16, "height": 16, "toroidal": 1})
        state = life.initial_state(params)

        timeline = fig.active_panel.timeline
        track = timeline.track("options", "global_alpha")
        track.interpolate = False
        grids = []
        for frame in range(6):
            grids.append(np.array(state["grid"], copy=True))
            state = life.step(state)

        # Key the *populations* on a step track: a value that must never be averaged.
        pop_track = timeline.track(99, "population")
        pop_track.interpolate = False
        for frame, grid in enumerate(grids):
            pop_track.add(frame * 0.5, float(grid.sum()))

        # Halfway between two generations must give the earlier one, not the mean.
        populations = [float(g.sum()) for g in grids]
        assert pop_track.value_at(0.25) == populations[0]
        assert pop_track.value_at(0.5) == populations[1]
