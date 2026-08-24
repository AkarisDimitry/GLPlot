"""Test glplot.gui.notifications: the global toast queue drawn by widgets.draw_toasts().

Pure logic, no imgui -- runs fully headless. Every test clears the module-level
registry first (it's a deliberate singleton, see the module docstring) so tests never
see another test's leftover toasts.
"""

from __future__ import annotations

import time

import pytest

from glplot.gui import notifications


@pytest.fixture(autouse=True)
def _clean_registry():
    notifications.clear()
    yield
    notifications.clear()


class TestPush:
    def test_returns_a_distinct_id_per_call(self):
        a = notifications.push("first")
        b = notifications.push("second")
        assert a != b

    def test_defaults_to_info_kind(self):
        notifications.push("hello")
        toast = notifications.active()[0]
        assert toast.kind == "info"

    def test_accepts_success_and_error_kinds(self):
        notifications.push("ok", kind="success")
        notifications.push("bad", kind="error")
        kinds = {t.kind for t in notifications.active()}
        assert kinds == {"success", "error"}

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown toast kind"):
            notifications.push("x", kind="warning")

    def test_message_is_stringified(self):
        notifications.push(12345)
        assert notifications.active()[0].message == "12345"


class TestActive:
    def test_empty_registry_returns_empty_list(self):
        assert notifications.active() == []

    def test_returns_oldest_first(self):
        notifications.push("first")
        notifications.push("second")
        notifications.push("third")
        messages = [t.message for t in notifications.active()]
        assert messages == ["first", "second", "third"]

    def test_expired_toasts_are_pruned(self):
        notifications.push("short-lived", duration=0.02)
        assert len(notifications.active()) == 1
        time.sleep(0.05)
        assert notifications.active() == []

    def test_unexpired_toasts_survive_a_call_that_prunes_others(self):
        notifications.push("expires soon", duration=0.02)
        notifications.push("lives on", duration=10.0)
        time.sleep(0.05)
        remaining = notifications.active()
        assert len(remaining) == 1
        assert remaining[0].message == "lives on"

    def test_returns_a_copy_not_the_live_list(self):
        notifications.push("one")
        snapshot = notifications.active()
        snapshot.append("tampered")
        assert len(notifications.active()) == 1


class TestDismiss:
    def test_removes_the_matching_toast(self):
        keep = notifications.push("keep me")
        drop = notifications.push("drop me")
        notifications.dismiss(drop)
        remaining = notifications.active()
        assert len(remaining) == 1
        assert remaining[0].id == keep

    def test_dismissing_an_unknown_id_is_a_no_op(self):
        notifications.push("only one")
        notifications.dismiss(999999)
        assert len(notifications.active()) == 1


class TestClear:
    def test_drops_everything(self):
        notifications.push("a")
        notifications.push("b")
        notifications.clear()
        assert notifications.active() == []
