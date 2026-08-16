"""Test the GUI deferred command queue in glplot.gui.commands.

Focus on FIFO drain ordering, per-command error isolation and the dirty-flag
epilogue, without requiring OpenGL or GPU. GPULinePlot constructs no window, so
the epilogue is asserted against the real engine state objects.
"""

from __future__ import annotations

import logging

import pytest

from glplot import GPULinePlot
from glplot.gui.commands import CommandQueue


class _FakeFrame:
    """Stand-in for engine FrameState carrying only the flags the epilogue touches."""

    def __init__(self) -> None:
        self.dirty_scene = False
        self.dirty_ui = False


class _FakeCache:
    """Stand-in for engine CacheState carrying only the fields the epilogue touches."""

    def __init__(self) -> None:
        self.capture_window = (0.0, 1.0, 0.0, 1.0)
        self.refresh_requested = False


class _FakePlot:
    """Minimal plot double: the queue only ever reaches for .frame and .cache."""

    def __init__(self) -> None:
        self.frame = _FakeFrame()
        self.cache = _FakeCache()


def _clean_plot() -> GPULinePlot:
    """Build a real GPULinePlot with every epilogue field pre-set to its 'wrong' value.

    A fresh engine already has dirty_scene/dirty_ui True and capture_window None, so
    without this the epilogue assertions would pass vacuously.
    """
    plot = GPULinePlot()
    plot.frame.dirty_scene = False
    plot.frame.dirty_ui = False
    plot.cache.capture_window = (0.0, 1.0, 0.0, 1.0)
    plot.cache.refresh_requested = False
    return plot


class TestSubmit:
    """Test CommandQueue submission and pending-state introspection."""

    def test_new_queue_is_empty(self):
        """A fresh queue reports empty and zero length."""
        queue = CommandQueue()
        assert queue.is_empty()
        assert len(queue) == 0

    def test_submit_appends_and_grows_length(self):
        """Each submit appends one pending command."""
        queue = CommandQueue()
        queue.submit(lambda: None, wake=False)
        queue.submit(lambda: None, wake=False)
        assert len(queue) == 2
        assert not queue.is_empty()

    def test_submit_with_wake_is_safe_headless(self):
        """Default wake=True must not raise with no GLFW window; the glfw call is guarded."""
        queue = CommandQueue()
        queue.submit(lambda: None)
        assert len(queue) == 1

    def test_submit_does_not_run_the_command(self):
        """Submission is deferred: nothing executes until drain."""
        ran = []
        queue = CommandQueue()
        queue.submit(lambda: ran.append(1), wake=False)
        assert ran == []

    def test_clear_drops_pending_without_running(self):
        """clear() discards queued commands and never executes them."""
        ran = []
        queue = CommandQueue()
        queue.submit(lambda: ran.append(1), wake=False)
        queue.clear()
        assert queue.is_empty()
        assert ran == []


class TestDrainOrdering:
    """Test that drain runs queued commands FIFO and reports whether it ran."""

    def test_drain_runs_commands_in_fifo_order(self):
        """Commands run oldest-first, in submission order."""
        order = []
        queue = CommandQueue()
        for i in range(5):
            queue.submit(lambda i=i: order.append(i), wake=False)
        assert queue.drain(_FakePlot()) is True
        assert order == [0, 1, 2, 3, 4]

    def test_drain_returns_true_when_anything_ran(self):
        """A non-empty drain reports True."""
        queue = CommandQueue()
        queue.submit(lambda: None, wake=False)
        assert queue.drain(_FakePlot()) is True

    def test_drain_empties_the_queue(self):
        """Every command queued at entry is consumed."""
        queue = CommandQueue()
        queue.submit(lambda: None, wake=False)
        queue.drain(_FakePlot())
        assert queue.is_empty()

    def test_drain_runs_each_command_exactly_once(self):
        """A drained command does not re-run on a later drain."""
        calls = []
        queue = CommandQueue()
        queue.submit(lambda: calls.append(1), wake=False)
        plot = _FakePlot()
        queue.drain(plot)
        queue.drain(plot)
        assert calls == [1]

    def test_command_submitted_during_drain_runs_on_the_next_drain(self):
        """The batch is snapshotted at entry, so a self-resubmitting closure cannot spin a frame."""
        queue = CommandQueue()
        calls = []

        def resubmit():
            calls.append(len(calls))
            if len(calls) < 3:
                queue.submit(resubmit, wake=False)

        queue.submit(resubmit, wake=False)
        plot = _FakePlot()
        assert queue.drain(plot) is True
        assert calls == [0]
        assert len(queue) == 1
        assert queue.drain(plot) is True
        assert calls == [0, 1]


class TestDrainErrorIsolation:
    """Test that one failing command never aborts the drain."""

    def test_exception_does_not_abort_remaining_commands(self):
        """A command that raises is skipped; later commands still run."""
        order = []
        queue = CommandQueue()

        def boom():
            raise RuntimeError("command failed on purpose")

        queue.submit(lambda: order.append(1), wake=False)
        queue.submit(boom, wake=False)
        queue.submit(lambda: order.append(3), wake=False)
        assert queue.drain(_FakePlot()) is True
        assert order == [1, 3]

    def test_multiple_failures_do_not_stop_the_drain(self):
        """Every good command runs even when it is surrounded by failures."""
        order = []
        queue = CommandQueue()

        def boom():
            raise ValueError("nope")

        queue.submit(boom, wake=False)
        queue.submit(lambda: order.append("a"), wake=False)
        queue.submit(boom, wake=False)
        queue.submit(lambda: order.append("b"), wake=False)
        queue.submit(boom, wake=False)
        queue.drain(_FakePlot())
        assert order == ["a", "b"]

    def test_failure_is_logged_not_swallowed_silently(self):
        """A failing command is reported through logging.exception, with a traceback."""
        queue = CommandQueue()

        def boom():
            raise RuntimeError("command failed on purpose")

        queue.submit(boom, wake=False)
        with caplog_at_error() as records:
            queue.drain(_FakePlot())
        assert any("command failed" in r.getMessage().lower() for r in records)
        assert any(r.exc_info is not None for r in records)

    def test_drain_returns_true_when_every_command_failed(self):
        """Commands ran (and failed); the drain still reports that work happened."""
        queue = CommandQueue()
        queue.submit(lambda: 1 / 0, wake=False)
        assert queue.drain(_FakePlot()) is True

    def test_epilogue_applies_even_when_every_command_failed(self):
        """A partial mutation may have landed before the raise, so the flags must still be set."""
        queue = CommandQueue()
        queue.submit(lambda: 1 / 0, wake=False)
        plot = _FakePlot()
        queue.drain(plot)
        assert plot.frame.dirty_scene is True
        assert plot.frame.dirty_ui is True
        assert plot.cache.capture_window is None
        assert plot.cache.refresh_requested is True

    def test_queue_is_reusable_after_a_failure(self):
        """A failed drain leaves the queue healthy for the next frame."""
        queue = CommandQueue()
        queue.submit(lambda: 1 / 0, wake=False)
        plot = _FakePlot()
        queue.drain(plot)
        ran = []
        queue.submit(lambda: ran.append(1), wake=False)
        assert queue.drain(plot) is True
        assert ran == [1]


class TestDirtyFlagEpilogue:
    """Test the CONTRACT 1.3 epilogue: exactly four fields, no more."""

    def test_epilogue_sets_the_four_documented_fields(self):
        """drain sets dirty_scene, dirty_ui, capture_window=None and refresh_requested."""
        queue = CommandQueue()
        queue.submit(lambda: None, wake=False)
        plot = _FakePlot()
        queue.drain(plot)
        assert plot.frame.dirty_scene is True
        assert plot.frame.dirty_ui is True
        assert plot.cache.capture_window is None
        assert plot.cache.refresh_requested is True

    def test_epilogue_touches_exactly_four_fields_on_real_engine_state(self):
        """Against real FrameState/CacheState, drain mutates those four fields and nothing else."""
        plot = _clean_plot()
        before_frame = dict(vars(plot.frame))
        before_cache = dict(vars(plot.cache))

        queue = CommandQueue()
        queue.submit(lambda: None, wake=False)
        queue.drain(plot)

        changed_frame = {k for k, v in vars(plot.frame).items() if before_frame[k] != v}
        changed_cache = {k for k, v in vars(plot.cache).items() if before_cache[k] != v}
        assert changed_frame == {"dirty_scene", "dirty_ui"}
        assert changed_cache == {"capture_window", "refresh_requested"}

    def test_epilogue_does_not_add_or_remove_state_attributes(self):
        """drain must not invent new attributes on the engine's state objects."""
        plot = _clean_plot()
        frame_keys = set(vars(plot.frame))
        cache_keys = set(vars(plot.cache))
        queue = CommandQueue()
        queue.submit(lambda: None, wake=False)
        queue.drain(plot)
        assert set(vars(plot.frame)) == frame_keys
        assert set(vars(plot.cache)) == cache_keys

    def test_epilogue_runs_after_the_commands(self):
        """A command that clears a flag itself cannot defeat the epilogue."""
        queue = CommandQueue()
        plot = _FakePlot()

        def sneaky():
            plot.frame.dirty_scene = False
            plot.cache.refresh_requested = False

        queue.submit(sneaky, wake=False)
        queue.drain(plot)
        assert plot.frame.dirty_scene is True
        assert plot.cache.refresh_requested is True

    def test_refresh_requested_survives_to_rebuild_the_impostor(self):
        """refresh_requested is the one flag the engine never clears; drain must raise it."""
        plot = _clean_plot()
        queue = CommandQueue()
        queue.submit(lambda: None, wake=False)
        queue.drain(plot)
        assert plot.cache.refresh_requested is True
        assert plot.cache.capture_window is None


class TestEmptyDrain:
    """Test that an empty drain is a true no-op."""

    def test_empty_drain_returns_false(self):
        """Draining an empty queue reports that nothing ran."""
        assert CommandQueue().drain(_FakePlot()) is False

    def test_empty_drain_sets_no_flags(self):
        """No commands ran, so no dirty flags may be raised: an idle frame must stay idle."""
        plot = _FakePlot()
        CommandQueue().drain(plot)
        assert plot.frame.dirty_scene is False
        assert plot.frame.dirty_ui is False
        assert plot.cache.capture_window == (0.0, 1.0, 0.0, 1.0)
        assert plot.cache.refresh_requested is False

    def test_empty_drain_mutates_nothing_on_real_engine_state(self):
        """Against real engine state, an empty drain changes no field at all."""
        plot = _clean_plot()
        before_frame = dict(vars(plot.frame))
        before_cache = dict(vars(plot.cache))
        assert CommandQueue().drain(plot) is False
        assert vars(plot.frame) == before_frame
        assert vars(plot.cache) == before_cache

    def test_drain_after_clear_returns_false(self):
        """Clearing pending work makes the next drain a no-op."""
        queue = CommandQueue()
        queue.submit(lambda: None, wake=False)
        queue.clear()
        plot = _FakePlot()
        assert queue.drain(plot) is False
        assert plot.frame.dirty_scene is False


class _CapturingHandler(logging.Handler):
    """Collect records emitted by the commands module during a drain."""

    def __init__(self) -> None:
        super().__init__()
        self.records = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _CaplogAtError:
    """Context manager attaching a capturing handler to the commands logger."""

    def __init__(self) -> None:
        self.handler = _CapturingHandler()
        self.logger = logging.getLogger("glplot.gui.commands")

    def __enter__(self):
        self.logger.addHandler(self.handler)
        self.previous = self.logger.level
        self.logger.setLevel(logging.ERROR)
        return self.handler.records

    def __exit__(self, *exc):
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self.previous)
        return False


def caplog_at_error() -> _CaplogAtError:
    """Capture ERROR records from glplot.gui.commands regardless of global logging config."""
    return _CaplogAtError()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q", "--no-cov"])
