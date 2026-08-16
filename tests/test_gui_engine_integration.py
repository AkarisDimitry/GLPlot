"""Test the GUI command queue's frame-safety contract against the real engine.

No OpenGL context and no window are created: ``GPULinePlot()`` builds its state objects
without a window (``plot.window`` stays None) and ``run()``/``show()`` are never called.
The engine's ``_main_loop`` is exercised the only way it can be exercised headless --
by replicating its drain/need_render expressions verbatim against a real plot, and by
reading its source to pin the ordering that makes those expressions correct.

The contract under test (``glplot/gui/commands.py`` + ``glplot/engine.py`` _main_loop):
panels may not mutate the scene from an imgui draw callback, because the dirty flags
raised there are cleared later in the same frame. They queue closures instead, and the
queue is drained at the TOP of the frame, BEFORE the need_render gate -- so the flags a
queued mutation raises survive to gate the very render they were raised for.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Dict

import numpy as np
import pytest

import glplot.engine as engine_module
from glplot.engine import GPULinePlot
from glplot.gui.commands import CommandQueue
from glplot.gui.layerops import add_xy_layer

#: The four fields ``CommandQueue.drain`` documents itself as setting, and no others.
DRAIN_EPILOGUE = {
    "frame.dirty_scene": True,
    "frame.dirty_ui": True,
    "cache.capture_window": None,
    "cache.refresh_requested": True,
}


def _state_snapshot(plot: GPULinePlot) -> Dict[str, Any]:
    """Flatten every FrameState and CacheState field into a ``"frame.x" -> value`` dict."""
    snap: Dict[str, Any] = {}
    for group in ("frame", "cache"):
        state = getattr(plot, group)
        for f in dataclasses.fields(state):
            snap[f"{group}.{f.name}"] = getattr(state, f.name)
    return snap


def _changed(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Return the ``after`` entries whose value differs from ``before``."""
    return {k: v for k, v in after.items() if before[k] != v}


def _neutral_state(plot: GPULinePlot) -> None:
    """Drive frame/cache to a state where every drain-set field would visibly change.

    Freshly constructed state has the dirty flags already True, which would mask a drain
    that set nothing at all, so the flags are cleared and ``capture_window`` is given a
    non-None sentinel first.
    """
    plot.frame.dirty_scene = False
    plot.frame.dirty_ui = False
    plot.frame.dirty_pick = False
    plot.cache.capture_window = (1.0, 2.0, 3.0, 4.0)
    plot.cache.refresh_requested = False
    plot.cache.active = True


def _need_render(plot: GPULinePlot, gui_queue: Any) -> bool:
    """Replicate the engine's ``need_render`` expression (engine.py:1139-1150) headless.

    ``reactive_rendering`` is forced on: with it off the gate is unconditionally True and
    the queue term could not be observed.
    """
    return bool(
        not plot.options.reactive_rendering
        or plot.frame.dirty_scene
        or plot.frame.dirty_ui
        or plot.frame.dirty_pick
        or plot.interaction.drag_active
        or plot.interaction.right_drag_active
        or plot.hud.state.show_profiler
        or (gui_queue is not None and not gui_queue.is_empty())
    )


@pytest.fixture
def plot():
    """A real headless GPULinePlot: no window, no GL context, no HUD."""
    p = GPULinePlot()
    p.options.reactive_rendering = True
    return p


class TestDrainFieldContract:
    """Test that drain touches exactly the four documented fields and nothing else."""

    def test_drain_sets_exactly_the_four_documented_fields(self, plot):
        """Test that a drain mutates the four epilogue fields and no other state."""
        _neutral_state(plot)
        q = CommandQueue()
        q.submit(lambda: None, wake=False)

        before = _state_snapshot(plot)
        q.drain(plot)
        after = _state_snapshot(plot)

        assert _changed(before, after) == DRAIN_EPILOGUE

    def test_drain_epilogue_values_are_exact(self, plot):
        """Test the epilogue's literal values on the real FrameState/CacheState."""
        _neutral_state(plot)
        q = CommandQueue()
        q.submit(lambda: None, wake=False)
        q.drain(plot)

        assert plot.frame.dirty_scene is True
        assert plot.frame.dirty_ui is True
        assert plot.cache.capture_window is None
        assert plot.cache.refresh_requested is True

    def test_drain_leaves_unrelated_state_fields_alone(self, plot):
        """Test that dirty_pick, cache.active and the timing fields are not touched."""
        _neutral_state(plot)
        plot.cache.last_capture_time = 12.5
        plot.cache.release_deadline = 99.0
        plot.frame.fps_estimate = 60.0
        q = CommandQueue()
        q.submit(lambda: None, wake=False)
        q.drain(plot)

        assert plot.frame.dirty_pick is False
        assert plot.cache.active is True
        assert np.allclose(plot.cache.last_capture_time, 12.5)
        assert np.allclose(plot.cache.release_deadline, 99.0)
        assert np.allclose(plot.frame.fps_estimate, 60.0)

    def test_drain_returns_true_when_commands_ran(self, plot):
        """Test that drain reports True when it ran a batch."""
        q = CommandQueue()
        q.submit(lambda: None, wake=False)
        assert q.drain(plot) is True

    def test_drain_empties_the_queue(self, plot):
        """Test that a drained queue is empty afterwards."""
        q = CommandQueue()
        q.submit(lambda: None, wake=False)
        q.drain(plot)
        assert q.is_empty()
        assert len(q) == 0


class TestEmptyDrain:
    """Test that draining an empty queue is a total no-op."""

    def test_empty_drain_returns_false(self, plot):
        """Test that drain on an empty queue reports False."""
        assert CommandQueue().drain(plot) is False

    def test_empty_drain_mutates_nothing(self, plot):
        """Test that an empty drain does not raise the dirty flags."""
        _neutral_state(plot)
        before = _state_snapshot(plot)
        CommandQueue().drain(plot)
        after = _state_snapshot(plot)

        assert _changed(before, after) == {}

    def test_empty_drain_does_not_touch_the_scene(self, plot):
        """Test that an empty drain leaves the layer list untouched."""
        n_before = len(plot.scene.layers)
        CommandQueue().drain(plot)
        assert len(plot.scene.layers) == n_before


class TestDrainExecution:
    """Test ordering, isolation and re-entrancy of the command batch."""

    def test_commands_run_fifo(self, plot):
        """Test that queued commands run in submission order."""
        seen = []
        q = CommandQueue()
        for i in range(5):
            q.submit(lambda i=i: seen.append(i), wake=False)
        q.drain(plot)

        assert seen == [0, 1, 2, 3, 4]

    def test_exception_does_not_abort_remaining_commands(self, plot):
        """Test that one failing command does not strand the commands behind it."""
        seen = []

        def boom():
            raise RuntimeError("panel action blew up")

        q = CommandQueue()
        q.submit(lambda: seen.append("first"), wake=False)
        q.submit(boom, wake=False)
        q.submit(lambda: seen.append("third"), wake=False)
        q.drain(plot)

        assert seen == ["first", "third"]

    def test_exception_does_not_propagate_to_the_loop(self, plot):
        """Test that drain swallows command exceptions rather than killing the frame."""
        q = CommandQueue()
        q.submit(lambda: 1 / 0, wake=False)
        assert q.drain(plot) is True  # would raise ZeroDivisionError if it propagated

    def test_exception_still_applies_the_dirty_epilogue(self, plot):
        """Test that the flags are raised even when every command failed."""
        _neutral_state(plot)
        q = CommandQueue()
        q.submit(lambda: 1 / 0, wake=False)
        q.drain(plot)

        assert plot.frame.dirty_scene is True
        assert plot.frame.dirty_ui is True

    def test_failing_command_is_logged(self, plot, caplog):
        """Test that a failing command is logged rather than silently dropped."""
        q = CommandQueue()
        q.submit(lambda: 1 / 0, wake=False)
        with caplog.at_level("ERROR", logger="glplot.gui.commands"):
            q.drain(plot)

        assert "GUI command failed" in caplog.text

    def test_command_submitted_during_drain_defers_to_next_drain(self, plot):
        """Test that a self-resubmitting command cannot spin the frame forever."""
        seen = []
        q = CommandQueue()

        def respawn():
            seen.append("ran")
            q.submit(lambda: seen.append("child"), wake=False)

        q.submit(respawn, wake=False)
        q.drain(plot)
        assert seen == ["ran"]
        assert len(q) == 1

        q.drain(plot)
        assert seen == ["ran", "child"]

    def test_reentrant_drain_is_a_noop(self, plot):
        """Test that a command calling drain() re-entrantly does not recurse."""
        results = []
        q = CommandQueue()

        def reenter():
            q.submit(lambda: results.append("inner"), wake=False)
            results.append(q.drain(plot))

        q.submit(reenter, wake=False)
        q.drain(plot)

        assert results == [False]  # the re-entrant drain refused; "inner" is still queued
        assert len(q) == 1

    def test_clear_drops_pending_commands_unrun(self, plot):
        """Test that clear() discards commands without executing them."""
        seen = []
        q = CommandQueue()
        q.submit(lambda: seen.append(1), wake=False)
        q.clear()

        assert q.is_empty()
        assert q.drain(plot) is False
        assert seen == []

    def test_len_and_is_empty_track_pending_commands(self, plot):
        """Test the queue's length/emptiness accessors."""
        q = CommandQueue()
        assert len(q) == 0 and q.is_empty()
        q.submit(lambda: None, wake=False)
        assert len(q) == 1 and not q.is_empty()

    def test_submit_with_wake_is_safe_headless(self):
        """Test that submit's default wake=True does not raise with no GLFW window."""
        q = CommandQueue()
        q.submit(lambda: None)  # wake=True: the glfw poke must be fully swallowed
        assert len(q) == 1


class TestQueuedLayerCreation:
    """Test that a queued layerops mutation actually lands in the live scene."""

    def test_queued_add_xy_layer_lands_in_the_scene(self, plot):
        """Test that a command adding a line layer produces a labelled scene layer."""
        x = np.linspace(0.0, 1.0, 10)
        y = x**2
        q = CommandQueue()
        q.submit(lambda: add_xy_layer(plot, x, y, kind="line", label="from GUI"), wake=False)

        assert len(plot.scene.layers) == 0  # nothing happens until the drain
        q.drain(plot)

        assert len(plot.scene.layers) == 1
        assert plot.scene.layers[0].label == "from GUI"

    def test_queued_layer_arrays_are_float32_and_c_contiguous(self, plot):
        """Test that the queued layer's GPU-bound array is float32 and C-contiguous."""
        x = np.linspace(0.0, 1.0, 10, dtype=np.float64)
        y = x**2
        q = CommandQueue()
        q.submit(lambda: add_xy_layer(plot, x, y, kind="line", label="L"), wake=False)
        q.drain(plot)

        pts = plot.scene.layers[0].pts
        assert pts.dtype == np.float32
        assert pts.flags["C_CONTIGUOUS"]
        assert pts.shape == (10, 2)
        assert np.allclose(pts[:, 0], x, atol=1e-6)
        assert np.allclose(pts[:, 1], y, atol=1e-6)

    def test_queued_scatter_layer_lands_in_the_scene(self, plot):
        """Test that kind='scatter' also routes through the queue into the scene."""
        x = np.linspace(0.0, 1.0, 8)
        q = CommandQueue()
        q.submit(lambda: add_xy_layer(plot, x, x, kind="scatter", label="pts"), wake=False)
        q.drain(plot)

        layer = plot.scene.layers[0]
        assert layer.label == "pts"
        assert layer.pts.dtype == np.float32
        assert layer.pts.flags["C_CONTIGUOUS"]

    def test_queued_layer_raises_the_dirty_flags(self, plot):
        """Test that adding a layer via the queue leaves the scene marked dirty."""
        _neutral_state(plot)
        x = np.linspace(0.0, 1.0, 4)
        q = CommandQueue()
        q.submit(lambda: add_xy_layer(plot, x, x, kind="line", label="L"), wake=False)
        q.drain(plot)

        assert plot.frame.dirty_scene is True
        assert plot.frame.dirty_ui is True


class TestNeedRenderGate:
    """Test that a pending queue wakes the reactive loop."""

    def test_engine_source_ors_the_queue_into_need_render(self):
        """Test that _main_loop's need_render expression includes the queue term."""
        source = inspect.getsource(GPULinePlot._main_loop)
        assert "need_render = (" in source
        assert "gui_queue is not None and not gui_queue.is_empty()" in source

    def test_pending_queue_makes_need_render_true(self, plot):
        """Test that a queued command alone flips need_render, with no dirty flags set."""
        _neutral_state(plot)
        q = CommandQueue()
        assert _need_render(plot, q) is False

        q.submit(lambda: None, wake=False)
        assert _need_render(plot, q) is True

    def test_need_render_false_when_queue_empty_and_scene_clean(self, plot):
        """Test that the gate stays shut on an idle frame with an empty queue."""
        _neutral_state(plot)
        assert _need_render(plot, CommandQueue()) is False

    def test_missing_queue_does_not_flip_need_render(self, plot):
        """Test the ``gui_queue is not None`` guard: a HUD-less engine renders as before."""
        _neutral_state(plot)
        assert _need_render(plot, None) is False

    def test_hudless_plot_exposes_no_workspace_queue(self, plot):
        """Test the engine's exact lookup: no HUD means no workspace, so no drain."""
        assert plot.options.enable_hud is False
        gui_queue = getattr(getattr(plot.hud, "workspace", None), "queue", None)
        assert gui_queue is None


class TestDrainOrdering:
    """Test the load-bearing ordering: drain at the top, before the need_render gate."""

    def test_drain_precedes_need_render_which_precedes_the_flag_clear(self):
        """Test the source ordering that keeps a queued mutation's flags from latch-loss."""
        source = inspect.getsource(GPULinePlot._main_loop)
        drain_at = source.index("gui_queue.drain(self)")
        gate_at = source.index("need_render = (")
        clear_at = source.index("self.frame.dirty_scene = False")

        assert drain_at < gate_at < clear_at

    def test_drain_is_guarded_and_runs_once_per_frame(self):
        """Test that the loop drains behind a None-check and calls drain exactly once."""
        source = inspect.getsource(GPULinePlot._main_loop)
        assert "if gui_queue is not None:" in source
        assert source.count("gui_queue.drain(self)") == 1

    def test_drained_flags_survive_to_gate_this_frames_render(self, plot):
        """Test that flags raised by the drain are still up when the gate is evaluated."""
        _neutral_state(plot)
        x = np.linspace(0.0, 1.0, 6)
        q = CommandQueue()
        q.submit(lambda: add_xy_layer(plot, x, x, kind="line", label="L"), wake=False)

        q.drain(plot)  # engine.py: step 0, top of the frame
        assert _need_render(plot, q) is True  # the gate the drained layer must pass

    def test_flags_raised_after_the_gate_would_be_latch_lost(self, plot):
        """Test the failure mode the ordering prevents: a late drain misses this frame."""
        _neutral_state(plot)
        q = CommandQueue()
        q.submit(lambda: None, wake=False)

        # Counterfactual: evaluate the gate against a queue that has not been drained yet
        # but whose pending term is ignored -- the frame would be skipped entirely.
        assert _need_render(plot, None) is False
        # The real loop drains first, so the same frame renders.
        q.drain(plot)
        assert _need_render(plot, None) is True

    def test_refresh_requested_survives_the_post_render_flag_clear(self, plot):
        """Test that the cache refresh latch outlives the loop's dirty-flag reset."""
        _neutral_state(plot)
        q = CommandQueue()
        q.submit(lambda: None, wake=False)
        q.drain(plot)

        plot.frame.dirty_scene = False  # engine.py:1253-1254, bottom of the frame
        plot.frame.dirty_ui = False

        assert plot.cache.refresh_requested is True
        assert plot.cache.capture_window is None

    def test_engine_module_exposes_the_main_loop_under_test(self):
        """Test that the source assertions above are pinned to the real engine module."""
        assert inspect.getmodule(GPULinePlot._main_loop) is engine_module
