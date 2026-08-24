"""A tiny global toast/notification queue, drawn by the HUD on top of every panel.

Deliberately a bare module-level registry, not a class threaded through
Workspace/HUD -- GLPlot runs one window per process (mirroring
``utils/mpl_process.py``'s own ``_VIEWERS`` list, the same convention for "there is
exactly one of these per running application"), so a singleton is the simplest honest
model here, not a shortcut.

Pure logic, no imgui import (CONTRACT 5.1): headless-testable on its own.
:func:`glplot.gui.widgets.draw_toasts` is what actually renders :func:`active`'s
result; nothing in this module draws.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import List

__all__ = ["Toast", "TOAST_KINDS", "push", "active", "dismiss", "clear"]

TOAST_KINDS = ("info", "success", "error")

_DEFAULT_DURATION = 5.0
_counter = itertools.count(1)


@dataclass
class Toast:
    """One notification. ``created`` is a ``time.monotonic()`` timestamp, not wall-clock."""

    id: int
    message: str
    kind: str
    created: float
    duration: float


_toasts: List[Toast] = []


def push(message: str, *, kind: str = "info", duration: float = _DEFAULT_DURATION) -> int:
    """Queue a toast. Returns its id, for an early :func:`dismiss`."""
    if kind not in TOAST_KINDS:
        raise ValueError(f"unknown toast kind {kind!r}; choose one of {', '.join(TOAST_KINDS)}")
    toast = Toast(
        id=next(_counter),
        message=str(message),
        kind=kind,
        created=time.monotonic(),
        duration=float(duration),
    )
    _toasts.append(toast)
    return toast.id


def active() -> List[Toast]:
    """Toasts not yet expired, oldest first. Prunes expired ones as a side effect --
    call this once per frame (that's what keeps the list from growing unbounded)."""
    now = time.monotonic()
    _toasts[:] = [t for t in _toasts if now - t.created < t.duration]
    return list(_toasts)


def dismiss(toast_id: int) -> None:
    """Remove one toast early (e.g. the user clicked its close button)."""
    _toasts[:] = [t for t in _toasts if t.id != toast_id]


def clear() -> None:
    """Drop every toast. Test-only -- nothing in the app itself calls this."""
    _toasts.clear()
