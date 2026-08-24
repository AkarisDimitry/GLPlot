"""Test glplot.gui.background: BackgroundJob, the daemon-thread compute primitive
Math Lab (and, potentially, any future panel) uses to run slow work without blocking
the GL/imgui frame.

These tests exercise real threads with small, bounded real sleeps -- unavoidable for a
concurrency primitive -- but every wait is a poll-until-condition loop with a generous
timeout, never a fixed sleep-then-assert, so the suite stays robust under CI jitter
instead of flaky.
"""

from __future__ import annotations

import time

import pytest

from glplot.gui.background import BackgroundJob, JobStatus


def _wait_until(predicate, *, timeout=2.0, interval=0.002):
    """Poll ``predicate`` until it's truthy or ``timeout`` seconds have elapsed.

    Robust against scheduler jitter: passes as soon as the condition is true, only
    fails if it genuinely never becomes true within a generous margin.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s")


class TestBackgroundJobSuccess:
    def test_reports_running_immediately(self):
        job = BackgroundJob(lambda: (time.sleep(0.05), 42)[1])
        status = job.poll()
        assert status.state == "running"
        assert status.result is None
        assert status.error is None

    def test_settles_to_done_with_the_return_value(self):
        job = BackgroundJob(lambda: 21 * 2)
        _wait_until(lambda: job.poll().state != "running")
        status = job.poll()
        assert status.state == "done"
        assert status.result == 42
        assert status.error is None

    def test_result_can_be_any_object_not_just_a_scalar(self):
        payload = {"a": [1, 2, 3], "b": "text"}
        job = BackgroundJob(lambda: payload)
        _wait_until(lambda: job.poll().state != "running")
        assert job.poll().result is payload

    def test_elapsed_increases_monotonically_while_running(self):
        job = BackgroundJob(lambda: time.sleep(0.08))
        first = job.poll().elapsed
        time.sleep(0.02)
        second = job.poll().elapsed
        assert second > first >= 0.0

    def test_poll_is_idempotent_after_settling(self):
        job = BackgroundJob(lambda: "value")
        _wait_until(lambda: job.poll().state != "running")
        s1 = job.poll()
        s2 = job.poll()
        assert s1.state == s2.state == "done"
        assert s1.result == s2.result == "value"


class TestBackgroundJobFailure:
    def test_settles_to_error_with_the_exception_message(self):
        def boom():
            raise ValueError("bad cutoff frequency")

        job = BackgroundJob(boom)
        _wait_until(lambda: job.poll().state != "running")
        status = job.poll()
        assert status.state == "error"
        assert status.error == "bad cutoff frequency"
        assert status.result is None

    def test_a_slow_failure_still_reports_running_first(self):
        def slow_boom():
            time.sleep(0.05)
            raise RuntimeError("late failure")

        job = BackgroundJob(slow_boom)
        assert job.poll().state == "running"
        _wait_until(lambda: job.poll().state != "running")
        assert job.poll().state == "error"
        assert job.poll().error == "late failure"


class TestBackgroundJobCancellation:
    def test_cancel_reports_cancelled_immediately_even_while_running(self):
        job = BackgroundJob(lambda: time.sleep(0.2))
        assert job.poll().state == "running"
        job.cancel()
        assert job.poll().state == "cancelled"

    def test_cancelled_property_reflects_cancel(self):
        job = BackgroundJob(lambda: time.sleep(0.05))
        assert job.cancelled is False
        job.cancel()
        assert job.cancelled is True

    def test_cancel_after_completion_does_not_resurrect_the_result(self):
        """A late cancel() -- after the thread already finished -- must not somehow
        make poll() report "done"; cancellation always wins once called."""
        job = BackgroundJob(lambda: "value")
        _wait_until(lambda: job.poll().state != "running")
        assert job.poll().state == "done"
        job.cancel()
        assert job.poll().state == "cancelled"

    def test_cancel_before_the_thread_finishes_suppresses_the_result(self):
        """The thread keeps running after cancel() (cooperative, not a hard kill --
        see the module docstring), but its eventual result must never surface."""
        job = BackgroundJob(lambda: (time.sleep(0.05), "should never be seen")[1])
        job.cancel()
        # Wait out longer than the thread's own sleep to prove the result is
        # permanently suppressed, not just not-yet-visible.
        time.sleep(0.15)
        status = job.poll()
        assert status.state == "cancelled"
        assert status.result is None

    def test_cancel_before_a_failure_suppresses_the_error_too(self):
        def slow_boom():
            time.sleep(0.05)
            raise ValueError("should never surface")

        job = BackgroundJob(slow_boom)
        job.cancel()
        time.sleep(0.15)
        status = job.poll()
        assert status.state == "cancelled"
        assert status.error is None

    def test_cancel_is_idempotent(self):
        job = BackgroundJob(lambda: time.sleep(0.02))
        job.cancel()
        job.cancel()  # must not raise
        assert job.poll().state == "cancelled"


class TestBackgroundJobDoesNotBlockTheCaller:
    def test_construction_returns_immediately_for_a_slow_function(self):
        start = time.monotonic()
        job = BackgroundJob(lambda: time.sleep(0.3))
        construction_time = time.monotonic() - start
        assert construction_time < 0.1, "BackgroundJob() must not block on fn()"
        assert job.poll().state == "running"

    def test_the_underlying_thread_is_a_daemon(self):
        """A non-daemon thread would keep the process alive after the GUI window
        closes if a job were still running -- must never happen."""
        job = BackgroundJob(lambda: time.sleep(0.05))
        assert job._thread.daemon is True


class TestJobStatus:
    def test_is_a_plain_frozen_dataclass(self):
        status = JobStatus("done", result=1, error=None, elapsed=0.5)
        assert status.state == "done"
        assert status.result == 1
        with pytest.raises(Exception):
            status.state = "error"  # frozen -- must not be mutable
