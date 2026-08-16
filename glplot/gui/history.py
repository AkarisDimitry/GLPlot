"""Undo/redo history for the GUI (pure logic, no imgui, no GL).

Snapshot-based undo is the preferred form for data edits, but it is memory-aware: an
operation whose inverse would require copying more than ``MAX_SNAPSHOT_ELEMENTS``
array elements is *not* recorded. See :class:`UndoStack.push` for the exact contract —
panels must surface "operation not undoable" when it returns ``False``.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: Never copy an array larger than this to build an undo snapshot (5e6 float64 ~ 40 MB).
MAX_SNAPSHOT_ELEMENTS = 5_000_000


def _noop() -> None:
    """Undo callback for a command that cannot be reversed."""


@dataclass
class Command:
    """A single reversible edit.

    ``do`` and ``undo`` take no arguments and return nothing; both must be idempotent
    with respect to repeated undo/redo cycling (i.e. ``do(); undo(); do()`` must land on
    the same state as ``do()``).

    ``undoable=False`` marks an operation whose inverse is unaffordable (see
    :func:`snapshot`). Such a command still runs, but :meth:`UndoStack.push` refuses to
    record it and wipes the history — replaying older undos across an unrecorded change
    would corrupt state.
    """

    label: str
    do: Callable[[], None]
    undo: Callable[[], None] = field(default=_noop)
    undoable: bool = True

    @classmethod
    def not_undoable(cls, label: str, do: Callable[[], None]) -> "Command":
        """Build a command that runs but can never be reversed."""
        return cls(label=label, do=do, undo=_noop, undoable=False)


def is_snapshot_safe(*arrays: Any) -> bool:
    """True when snapshotting every given array costs at most ``MAX_SNAPSHOT_ELEMENTS``."""
    total = 0
    for arr in arrays:
        if arr is None:
            continue
        total += int(np.asarray(arr).size)
        if total > MAX_SNAPSHOT_ELEMENTS:
            return False
    return True


def snapshot(arr: Any) -> Optional[np.ndarray]:
    """Return an independent copy of ``arr``, or ``None`` when it is too large to copy.

    ``None`` is the signal to build a :meth:`Command.not_undoable` (or an inverse-op based
    command) instead of a snapshot-based one. Never silently copies a 100M-element array.
    """
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.size > MAX_SNAPSHOT_ELEMENTS:
        return None
    return a.copy()


def array_edit_command(
    label: str,
    target: np.ndarray,
    new_values: Any,
    *,
    on_apply: Optional[Callable[[], None]] = None,
) -> Command:
    """Build a command that writes ``new_values`` into ``target`` in place, undoably.

    The in-place write is what keeps a live layer link working: ``target`` may be a view
    into a layer's array, so the layer sees the edit without a rebind. ``on_apply`` runs
    after both ``do`` and ``undo`` — that is where callers put ``layer.dirty.gpu_dirty =
    True`` plus the §1.4 incantation (from inside a queued command only).

    When ``target`` is larger than ``MAX_SNAPSHOT_ELEMENTS`` the returned command has
    ``undoable=False``: it still applies, but :meth:`UndoStack.push` will not record it.
    """
    new = np.array(new_values, dtype=target.dtype)
    old = snapshot(target)

    def do() -> None:
        target[...] = new
        if on_apply is not None:
            on_apply()

    if old is None:
        return Command.not_undoable(label, do)

    def undo() -> None:
        target[...] = old
        if on_apply is not None:
            on_apply()

    return Command(label=label, do=do, undo=undo)


class UndoStack:
    """Bounded undo/redo history.

    Ordering is strict LIFO. Any :meth:`push` clears the redo branch; when the undo stack
    exceeds ``limit`` the oldest entry is evicted (it becomes permanent history).
    """

    def __init__(self, limit: int = 200) -> None:
        self.limit = max(1, int(limit))
        self._undo: Deque[Command] = deque()
        self._redo: Deque[Command] = deque()

    def push(self, cmd: Command, *, execute: bool = True) -> bool:
        """Run ``cmd.do()`` (unless ``execute=False``) and record it.

        Returns ``True`` when the command was recorded and is undoable. Returns ``False``
        for a command with ``undoable=False``: the operation still runs, but it is not
        recorded and the whole history is cleared (undoing an *older* command across an
        unrecorded change would corrupt state). Callers must surface "operation not
        undoable" in the UI on ``False``.
        """
        if execute:
            cmd.do()
        if not cmd.undoable:
            self.clear()
            logger.debug("Not undoable, history cleared: %s", cmd.label)
            return False
        self._redo.clear()
        self._undo.append(cmd)
        while len(self._undo) > self.limit:
            self._undo.popleft()
        return True

    def undo(self) -> Optional[str]:
        """Reverse the most recent command; returns its label, or ``None`` if nothing to undo.

        If the undo callback raises, the failure is logged and the history is cleared: the
        state is no longer known to match the recorded chain, so replaying it is unsafe.
        """
        if not self._undo:
            return None
        cmd = self._undo.pop()
        try:
            cmd.undo()
        except Exception:
            logger.exception("Undo failed, clearing history: %s", cmd.label)
            self.clear()
            return None
        self._redo.append(cmd)
        return cmd.label

    def redo(self) -> Optional[str]:
        """Re-apply the most recently undone command; returns its label, or ``None``."""
        if not self._redo:
            return None
        cmd = self._redo.pop()
        try:
            cmd.do()
        except Exception:
            logger.exception("Redo failed, clearing history: %s", cmd.label)
            self.clear()
            return None
        self._undo.append(cmd)
        while len(self._undo) > self.limit:
            self._undo.popleft()
        return cmd.label

    def can_undo(self) -> bool:
        """True when :meth:`undo` would do something."""
        return bool(self._undo)

    def can_redo(self) -> bool:
        """True when :meth:`redo` would do something."""
        return bool(self._redo)

    def peek_undo(self) -> Optional[str]:
        """Label of the next command :meth:`undo` would reverse, for menu text."""
        return self._undo[-1].label if self._undo else None

    def peek_redo(self) -> Optional[str]:
        """Label of the next command :meth:`redo` would re-apply, for menu text."""
        return self._redo[-1].label if self._redo else None

    def clear(self) -> None:
        """Forget both branches."""
        self._undo.clear()
        self._redo.clear()

    def __len__(self) -> int:
        return len(self._undo)
