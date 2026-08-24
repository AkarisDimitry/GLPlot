"""Background execution for potentially-slow GUI computations.

Why threads, not the subprocess pattern utils/mpl_process.py uses: that module spawns
a fresh interpreter because a live GL context is unsafe to inherit via ``fork()`` on
macOS -- Cocoa's autorelease pool and the MacOSX matplotlib backend both abort outside
the main thread. None of that applies here: the work this module backgrounds (calls
into ``mathops``/``mathops2d``/``mathopsnd`` -- numpy/scipy on plain arrays) has no
Cocoa, GL, or imgui entanglement, and spends most of its time inside C extensions that
release the GIL. A plain ``threading.Thread`` gives real responsiveness with none of a
subprocess's pickling/IPC overhead, and needs no change to the functions it runs.

Cancellation is cooperative, not a hard kill. CPython cannot forcibly terminate a
thread -- there is no SIGKILL-equivalent for one thread inside a live process (killing
the *computation* would mean killing the whole process, GL context and all).
:meth:`BackgroundJob.cancel` therefore only marks the job abandoned: :meth:`poll`
reports ``"cancelled"`` immediately and permanently from that call on, even though the
underlying call may keep running to completion, unobserved, until it returns. This is
the same model production tools use to "cancel" a library call with no internal
cancellation hook (a database client's "cancel query", a browser's "stop" on an
in-flight synchronous request) -- from the caller's perspective the job *is*
cancelled: nothing about it is waited on, shown, or acted on ever again.

A caller that needs GL-thread safety gets it for free: nothing in this module touches
``plot.scene``/``frame``/``cache`` or imgui, so CONTRACT 1.1 is simply not in play --
:meth:`poll` only ever hands back plain Python/numpy values for the CALLER to apply on
its own thread, exactly like any other panel-local state read from ``draw()``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

__all__ = ["BackgroundJob", "JobStatus"]

#: The states a JobStatus can report. "cancelled" is sticky: once cancel() is called,
#: every subsequent poll() reports it, regardless of what the thread does afterward.
JOB_STATES = ("running", "done", "error", "cancelled")


@dataclass(frozen=True)
class JobStatus:
    """A snapshot of a :class:`BackgroundJob` at the moment :meth:`BackgroundJob.poll`
    was called.

    ``result`` is only meaningful when ``state == "done"``; ``error`` only when
    ``state == "error"``. ``elapsed`` is wall-clock seconds since the job started.
    """

    state: str
    result: Any = None
    error: Optional[str] = None
    elapsed: float = 0.0


class BackgroundJob:
    """Runs ``fn`` on a daemon thread; :meth:`poll` for its result without blocking.

    ``fn`` takes no arguments -- wrap whatever inputs it needs in a closure (e.g.
    ``BackgroundJob(lambda: mathops.smooth(y, method="savgol"))``). Starts immediately
    on construction; there is no separate "start" step.
    """

    def __init__(self, fn: Callable[[], Any]) -> None:
        self._fn = fn
        self._lock = threading.Lock()
        self._done = False
        self._cancelled = False
        self._result: Any = None
        self._error: Optional[str] = None
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:  # noqa: BLE001 - marshaled to the poller, never raised here
            with self._lock:
                if not self._cancelled:
                    self._error = str(exc)
                    self._done = True
            return
        with self._lock:
            if not self._cancelled:
                self._result = result
                self._done = True

    def cancel(self) -> None:
        """Abandon this job. See the module docstring: cooperative, not a hard kill."""
        with self._lock:
            self._cancelled = True

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def poll(self) -> JobStatus:
        """Non-blocking status check. Cheap enough to call once every frame."""
        elapsed = time.monotonic() - self._started
        with self._lock:
            if self._cancelled:
                return JobStatus("cancelled", elapsed=elapsed)
            if not self._done:
                return JobStatus("running", elapsed=elapsed)
            if self._error is not None:
                return JobStatus("error", error=self._error, elapsed=elapsed)
            return JobStatus("done", result=self._result, elapsed=elapsed)
