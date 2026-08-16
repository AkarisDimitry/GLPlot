"""Test the GUI undo/redo history in glplot.gui.history.

Pure logic: no OpenGL context, no window and no imgui are created here. Commands are
plain closures that record their calls into a list.
"""

from __future__ import annotations

import numpy as np

from glplot.gui.history import (
    MAX_SNAPSHOT_ELEMENTS,
    Command,
    UndoStack,
    array_edit_command,
    is_snapshot_safe,
    snapshot,
)


def _recording_command(label: str, log: list) -> Command:
    """Build a command that appends ``do:<label>`` / ``undo:<label>`` to ``log``."""
    return Command(
        label=label,
        do=lambda: log.append(f"do:{label}"),
        undo=lambda: log.append(f"undo:{label}"),
    )


class TestCommand:
    """Test the Command dataclass."""

    def test_defaults(self):
        """Test that a command is undoable by default with a no-op undo."""
        cmd = Command(label="edit", do=lambda: None)
        assert cmd.undoable is True
        assert cmd.undo() is None

    def test_not_undoable_factory(self):
        """Test that Command.not_undoable() marks the command unrecordable."""
        cmd = Command.not_undoable("huge", lambda: None)
        assert cmd.undoable is False
        assert cmd.label == "huge"


class TestSnapshot:
    """Test the memory-aware snapshot helpers."""

    def test_snapshot_copies(self):
        """Test that snapshot() returns an independent copy."""
        arr = np.arange(4.0)
        copy = snapshot(arr)
        copy[0] = 99.0
        assert np.allclose(arr[0], 0.0)

    def test_snapshot_of_none(self):
        """Test that snapshot(None) is None."""
        assert snapshot(None) is None

    def test_snapshot_refuses_oversized_array(self):
        """Test that an array over the element budget snapshots to None."""
        assert snapshot(np.empty(MAX_SNAPSHOT_ELEMENTS, dtype=np.int8)) is not None
        assert snapshot(np.empty(MAX_SNAPSHOT_ELEMENTS + 1, dtype=np.int8)) is None

    def test_is_snapshot_safe_small(self):
        """Test that small arrays are reported safe to snapshot."""
        assert is_snapshot_safe(np.zeros(10), np.zeros(20)) is True

    def test_is_snapshot_safe_ignores_none(self):
        """Test that None entries do not count toward the budget."""
        assert is_snapshot_safe(None, np.zeros(5)) is True

    def test_is_snapshot_safe_sums_across_arrays(self):
        """Test that the budget applies to the total, not to each array."""
        half = MAX_SNAPSHOT_ELEMENTS // 2 + 10
        a = np.empty(half, dtype=np.int8)
        assert is_snapshot_safe(a) is True
        assert is_snapshot_safe(a, a) is False


class TestArrayEditCommand:
    """Test the array_edit_command builder."""

    def test_do_and_undo_round_trip(self):
        """Test that the command writes in place and restores on undo."""
        target = np.array([1.0, 2.0, 3.0])
        cmd = array_edit_command("edit", target, [4.0, 5.0, 6.0])
        cmd.do()
        assert np.allclose(target, [4.0, 5.0, 6.0])
        cmd.undo()
        assert np.allclose(target, [1.0, 2.0, 3.0])

    def test_write_is_in_place(self):
        """Test that the target array object itself is mutated, not replaced."""
        target = np.zeros(3)
        view = target[:]
        array_edit_command("edit", target, [1.0, 1.0, 1.0]).do()
        assert np.allclose(view, 1.0)

    def test_on_apply_runs_for_do_and_undo(self):
        """Test that on_apply fires after both do() and undo()."""
        calls = []
        target = np.zeros(2)
        cmd = array_edit_command("edit", target, [1.0, 1.0], on_apply=lambda: calls.append(1))
        cmd.do()
        cmd.undo()
        assert len(calls) == 2

    def test_new_values_coerced_to_target_dtype(self):
        """Test that new values are cast to the target's dtype."""
        target = np.zeros(2, dtype=np.float32)
        array_edit_command("edit", target, [1.0, 2.0]).do()
        assert target.dtype == np.float32
        assert np.allclose(target, [1.0, 2.0])

    def test_oversized_target_is_not_undoable(self):
        """Test that an over-budget target yields a non-undoable command."""
        target = np.empty(MAX_SNAPSHOT_ELEMENTS + 1, dtype=np.int8)
        cmd = array_edit_command("edit", target, 0)
        assert cmd.undoable is False


class TestUndoStackOrdering:
    """Test undo/redo ordering across multiple commands."""

    def test_push_executes_by_default(self):
        """Test that push() runs the command's do()."""
        log = []
        UndoStack().push(_recording_command("a", log))
        assert log == ["do:a"]

    def test_push_without_execute(self):
        """Test that execute=False records without running do()."""
        log = []
        stack = UndoStack()
        stack.push(_recording_command("a", log), execute=False)
        assert log == []
        assert stack.can_undo() is True

    def test_undo_is_lifo(self):
        """Test that undo reverses commands in last-in-first-out order."""
        log = []
        stack = UndoStack()
        stack.push(_recording_command("a", log))
        stack.push(_recording_command("b", log))
        stack.push(_recording_command("c", log))
        assert stack.undo() == "c"
        assert stack.undo() == "b"
        assert stack.undo() == "a"
        assert log == ["do:a", "do:b", "do:c", "undo:c", "undo:b", "undo:a"]

    def test_redo_replays_in_original_order(self):
        """Test that redo re-applies undone commands in their original order."""
        log = []
        stack = UndoStack()
        stack.push(_recording_command("a", log))
        stack.push(_recording_command("b", log))
        stack.undo()
        stack.undo()
        log.clear()
        assert stack.redo() == "a"
        assert stack.redo() == "b"
        assert log == ["do:a", "do:b"]

    def test_undo_on_empty_stack(self):
        """Test that undo() on an empty stack returns None."""
        assert UndoStack().undo() is None

    def test_redo_with_empty_redo_branch(self):
        """Test that redo() returns None when nothing has been undone."""
        stack = UndoStack()
        stack.push(_recording_command("a", []))
        assert stack.redo() is None

    def test_state_round_trips_through_undo_redo(self):
        """Test a real array edit cycling through undo and redo."""
        target = np.zeros(3)
        stack = UndoStack()
        stack.push(array_edit_command("set", target, [1.0, 2.0, 3.0]))
        stack.undo()
        assert np.allclose(target, 0.0)
        stack.redo()
        assert np.allclose(target, [1.0, 2.0, 3.0])

    def test_len_reports_undo_depth(self):
        """Test that len(stack) is the number of undoable entries."""
        stack = UndoStack()
        stack.push(_recording_command("a", []))
        stack.push(_recording_command("b", []))
        assert len(stack) == 2
        stack.undo()
        assert len(stack) == 1


class TestUndoStackRedoBranch:
    """Test that a new push discards the redo branch."""

    def test_push_clears_redo(self):
        """Test that pushing after an undo drops the redoable command."""
        stack = UndoStack()
        stack.push(_recording_command("a", []))
        stack.undo()
        assert stack.can_redo() is True
        stack.push(_recording_command("b", []))
        assert stack.can_redo() is False
        assert stack.redo() is None

    def test_redo_branch_survives_until_pushed_over(self):
        """Test that consecutive undos keep everything redoable."""
        stack = UndoStack()
        stack.push(_recording_command("a", []))
        stack.push(_recording_command("b", []))
        stack.undo()
        stack.undo()
        assert stack.peek_redo() == "a"
        stack.push(_recording_command("c", []))
        assert stack.peek_redo() is None


class TestUndoStackQueries:
    """Test can_undo/can_redo/peek_undo/peek_redo and clear."""

    def test_queries_on_empty_stack(self):
        """Test that a fresh stack reports nothing to undo or redo."""
        stack = UndoStack()
        assert stack.can_undo() is False
        assert stack.can_redo() is False
        assert stack.peek_undo() is None
        assert stack.peek_redo() is None

    def test_peek_undo_names_the_next_target(self):
        """Test that peek_undo() names the most recent command."""
        stack = UndoStack()
        stack.push(_recording_command("a", []))
        stack.push(_recording_command("b", []))
        assert stack.peek_undo() == "b"
        assert stack.can_undo() is True

    def test_peek_redo_names_the_next_target(self):
        """Test that peek_redo() names the most recently undone command."""
        stack = UndoStack()
        stack.push(_recording_command("a", []))
        stack.undo()
        assert stack.peek_redo() == "a"
        assert stack.peek_undo() is None

    def test_peek_does_not_consume(self):
        """Test that peeking is non-destructive."""
        stack = UndoStack()
        stack.push(_recording_command("a", []))
        stack.peek_undo()
        stack.peek_undo()
        assert len(stack) == 1

    def test_clear_forgets_both_branches(self):
        """Test that clear() empties the undo and redo branches."""
        stack = UndoStack()
        stack.push(_recording_command("a", []))
        stack.push(_recording_command("b", []))
        stack.undo()
        stack.clear()
        assert stack.can_undo() is False
        assert stack.can_redo() is False
        assert len(stack) == 0

    def test_clear_does_not_revert_state(self):
        """Test that clear() only forgets history; it does not undo anything."""
        target = np.zeros(2)
        stack = UndoStack()
        stack.push(array_edit_command("set", target, [1.0, 1.0]))
        stack.clear()
        assert np.allclose(target, 1.0)


class TestUndoStackLimit:
    """Test bounded history and oldest-entry eviction."""

    def test_oldest_entry_evicted(self):
        """Test that pushing past the limit drops the oldest command."""
        stack = UndoStack(limit=2)
        stack.push(_recording_command("a", []))
        stack.push(_recording_command("b", []))
        stack.push(_recording_command("c", []))
        assert len(stack) == 2
        assert stack.undo() == "c"
        assert stack.undo() == "b"
        assert stack.undo() is None

    def test_limit_floor_is_one(self):
        """Test that a non-positive limit is clamped to 1."""
        assert UndoStack(limit=0).limit == 1
        assert UndoStack(limit=-5).limit == 1

    def test_default_limit(self):
        """Test the documented default history depth."""
        assert UndoStack().limit == 200

    def test_redo_also_respects_the_limit(self):
        """Test that redo does not push the undo branch past the limit."""
        stack = UndoStack(limit=1)
        stack.push(_recording_command("a", []))
        stack.undo()
        stack.redo()
        assert len(stack) == 1


class TestUndoStackNotUndoable:
    """Test the memory rule: non-undoable commands are run but never recorded."""

    def test_push_returns_true_for_normal_command(self):
        """Test that push() reports True for a recordable command."""
        assert UndoStack().push(_recording_command("a", [])) is True

    def test_push_returns_false_for_non_undoable(self):
        """Test that push() reports False so panels can surface 'not undoable'."""
        stack = UndoStack()
        assert stack.push(Command.not_undoable("huge", lambda: None)) is False

    def test_non_undoable_command_still_runs(self):
        """Test that a non-undoable command's do() is still executed."""
        log = []
        UndoStack().push(Command.not_undoable("huge", lambda: log.append("do")))
        assert log == ["do"]

    def test_non_undoable_command_is_not_recorded(self):
        """Test that a non-undoable command leaves nothing to undo."""
        stack = UndoStack()
        stack.push(Command.not_undoable("huge", lambda: None))
        assert stack.can_undo() is False
        assert len(stack) == 0

    def test_non_undoable_command_wipes_prior_history(self):
        """Test that an unrecorded change clears the whole history."""
        stack = UndoStack()
        stack.push(_recording_command("a", []))
        stack.push(_recording_command("b", []))
        stack.undo()
        assert stack.push(Command.not_undoable("huge", lambda: None)) is False
        assert stack.can_undo() is False
        assert stack.can_redo() is False

    def test_oversized_array_edit_is_not_recorded(self):
        """Test the end-to-end memory rule with a real over-budget array edit."""
        target = np.empty(MAX_SNAPSHOT_ELEMENTS + 1, dtype=np.int8)
        stack = UndoStack()
        stack.push(_recording_command("a", []))
        assert stack.push(array_edit_command("fill", target, 7)) is False
        assert int(target[0]) == 7
        assert stack.can_undo() is False


class TestUndoStackFailureRecovery:
    """Test that a raising callback clears the history rather than lying."""

    def test_failed_undo_clears_history(self):
        """Test that an exception from undo() wipes the stack and returns None."""

        def _boom() -> None:
            raise RuntimeError("nope")

        stack = UndoStack()
        stack.push(Command(label="a", do=lambda: None, undo=_boom))
        assert stack.undo() is None
        assert stack.can_undo() is False
        assert stack.can_redo() is False

    def test_failed_redo_clears_history(self):
        """Test that an exception from do() during redo wipes the stack."""
        calls = []

        def _do() -> None:
            calls.append(1)
            if len(calls) > 1:
                raise RuntimeError("nope")

        stack = UndoStack()
        stack.push(Command(label="a", do=_do, undo=lambda: None))
        stack.undo()
        assert stack.redo() is None
        assert stack.can_undo() is False
        assert stack.can_redo() is False
