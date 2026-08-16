"""Deferred mutation queue for the GUI (pure logic, no imgui, no GL).

Producers are imgui draw callbacks, which may never touch ``plot.scene`` directly
(the dirty-flag latch set inside ``hud.update()`` is erased later in the same frame
by ``engine.py:1238-1239``). The single consumer is ``_main_loop``, which drains the
queue at the top of the frame so the flags this module sets survive to gate the very
render they were raised for.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Callable, Deque, List

logger = logging.getLogger(__name__)


class CommandQueue:
    """Deferred scene mutations. Producers: imgui draw callbacks. Consumer: ``_main_loop`` top.

    Every mutation of the live scene (add/remove/reorder/restyle/retranslate, autoscale,
    ``gui_clear``) must be submitted here as a zero-argument closure instead of being run
    from inside a draw callback.
    """

    def __init__(self) -> None:
        self._q: Deque[Callable[[], None]] = deque()
        self._draining = False

    def submit(self, fn: Callable[[], None], *, wake: bool = True) -> None:
        """Queue ``fn`` to run at the top of the next frame.

        ``wake`` posts an empty GLFW event so a reactive loop sleeping inside
        ``wait_events_timeout`` returns immediately. The glfw import is function-local
        and fully guarded: this module must stay importable headless.
        """
        self._q.append(fn)
        if wake:
            self._wake()

    @staticmethod
    def _wake() -> None:
        """Break ``glfw.wait_events_timeout`` so a queued command is drained promptly.

        Headless there is no GLFW library to wake, and the call reports that as a *warning*
        rather than an exception, so warnings are suppressed alongside the try/except:
        submitting a command must be silent and safe with no window.
        """
        try:
            import warnings

            import glfw

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                glfw.post_empty_event()
        except Exception:  # pragma: no cover - headless / no glfw context
            pass

    def __len__(self) -> int:
        return len(self._q)

    def is_empty(self) -> bool:
        """True when nothing is pending. ``need_render`` must OR in ``not is_empty()``."""
        return not self._q

    def clear(self) -> None:
        """Drop every pending command without running it."""
        self._q.clear()

    def drain(self, plot: Any) -> bool:
        """Run every command queued at entry. Call ONCE per frame, at the TOP of ``_main_loop``.

        Exceptions from a command are caught, logged, and do not abort the drain: one bad
        panel action must never take down the render loop or strand the remaining commands.

        Commands submitted *by* a command run on the next drain, not this one: the batch is
        snapshotted at entry so a self-resubmitting closure cannot spin the frame forever.
        Re-entrant calls are a no-op.

        Returns True if anything ran (and therefore if the dirty-flag epilogue was applied).
        """
        if not self._q or self._draining:
            return False
        batch: List[Callable[[], None]] = []
        while self._q:
            batch.append(self._q.popleft())
        self._draining = True
        try:
            for fn in batch:
                try:
                    fn()
                except Exception:
                    logger.exception("GUI command failed: %r", fn)
        finally:
            self._draining = False
        plot.frame.dirty_scene = True
        plot.frame.dirty_ui = True
        plot.cache.capture_window = None
        plot.cache.refresh_requested = True
        return True
