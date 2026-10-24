"""Test the Controls panel and its workspace integration.

Panel bodies go through the sanctioned headless imgui harness (create a context, open a
frame, draw, render): that is what catches a wrong imgui_bundle signature or an
unbalanced push/pop. Popups and context menus are *forced open* by monkeypatching their
``begin_*`` call, because a real one only opens on a click the harness cannot make --
and menu code is exactly where this build's missing ``menu_item`` defaults have crashed
the live app before while every panel test stayed green.
"""

from __future__ import annotations

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot import controls as ctl
from glplot.gui import notifications
from glplot.gui.commands import CommandQueue
from glplot.gui.datasets import DataStore
from glplot.gui.history import UndoStack

imgui = pytest.importorskip("imgui_bundle").imgui

from glplot.gui.panels import controls as panel_mod  # noqa: E402
from glplot.gui.panels.controls import ControlsPanel  # noqa: E402


class FakeWorkspace:
    def __init__(self, plot):
        self.plot = plot
        self.queue = CommandQueue()
        self.undo = UndoStack()
        self.store = DataStore()
        self.hud = None


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    notifications.clear()
    yield
    gplt._cleanup_pyplot_state()
    notifications.clear()


@pytest.fixture
def harness():
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = 1400, 900
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    io.delta_time = 1 / 60.0
    yield imgui
    imgui.destroy_context(ctx)


@pytest.fixture
def all_sections(monkeypatch):
    from glplot.gui import widgets

    monkeypatch.setattr(widgets, "section", lambda label, **kw: True)


@pytest.fixture
def fig():
    return gplt.figure()


def _frame(panel, title="Controls"):
    imgui.new_frame()
    imgui.begin(title)
    panel.draw()
    imgui.end()
    imgui.render()
    assert imgui.get_draw_data() is not None


def _populate():
    x = np.linspace(0, 1, 50)
    with gplt.control_group("Signal"):
        gplt.slider("Amp", 0, 5)
        gplt.slider("Freq", 0.1, 100, log=True, live=False, unit="Hz")
        gplt.slider("N", 1, 10, step=1, integer=True)
        gplt.range_slider("Band", 0, 100)
        gplt.range_slider("Bins", 0, 100, step=1, integer=True)
    gplt.checkbox("Noise")
    gplt.dropdown("Wave", {"Sine": np.sin, "Cosine": np.cos})
    gplt.radio("Mode", ["a", "b", "c"])
    gplt.textbox("Title", "hello", placeholder="title")
    gplt.textbox("Notes", "", multiline=True)
    gplt.textbox("Live", "", live=True)
    gplt.number("K", 3.0)
    gplt.number("I", 3, integer=True, min=0, max=10)
    gplt.color_picker("Colour")
    gplt.color_picker("Tint", "red", alpha=True)
    gplt.readout("RMS", 0.123, fmt="%.2f")
    gplt.button("Go", lambda: None, confirm="Sure?", help="runs it")
    gplt.button("Stop", lambda: None, enabled=False)
    gplt.slider("Hidden", 0, 1, visible=False)

    @gplt.interact(a=(0, 2))
    def stats(a=1.0):
        return {"peak": a, "arr": np.arange(3)}

    @gplt.interact(b=(0, 2))
    def curve(b=1.0):
        return x, b * x


class TestDrawing:
    def test_every_kind_draws(self, harness, fig, all_sections):
        _populate()
        fig.controls["amp"].value = 4  # a modified (accent) label
        panel = ControlsPanel(FakeWorkspace(fig))
        for _ in range(3):
            _frame(panel)

    def test_the_empty_state_draws(self, harness, fig):
        _frame(ControlsPanel(FakeWorkspace(fig)))

    def test_errors_draw(self, harness, fig, all_sections):
        s = gplt.slider("a", 0, 1)
        s.error = "ValueError: bad"

        @gplt.interact
        def f(a):
            return None

        f.error = "ZeroDivisionError: x"
        _frame(ControlsPanel(FakeWorkspace(fig)))

    def test_forced_popups_and_context_menus_draw(self, harness, fig, all_sections, monkeypatch):
        _populate()
        fig.controls.save_preset("first")
        monkeypatch.setattr(imgui, "begin_popup_context_item", lambda *a, **k: True)
        monkeypatch.setattr(imgui, "begin_popup", lambda *a, **k: True)
        monkeypatch.setattr(imgui, "end_popup", lambda: None)
        _frame(ControlsPanel(FakeWorkspace(fig)))

    def test_the_context_menu_resets_a_control(self, harness, fig, monkeypatch):
        s = gplt.slider("a", 0, 10, 2)
        s.value = 9
        monkeypatch.setattr(imgui, "begin_popup_context_item", lambda *a, **k: True)
        monkeypatch.setattr(imgui, "end_popup", lambda: None)
        monkeypatch.setattr(
            imgui,
            "menu_item",
            lambda label, shortcut, selected, *a: (label.startswith("Reset"), False),
        )
        _frame(ControlsPanel(FakeWorkspace(fig)))
        assert s.value == 2.0


class TestInteraction:
    def test_a_slider_edit_reaches_the_model(self, harness, fig, monkeypatch):
        s = gplt.slider("a", 0, 10, 1)
        monkeypatch.setattr(imgui, "slider_float", lambda *a, **k: (True, 7.0))
        _frame(ControlsPanel(FakeWorkspace(fig)))
        assert s.value == 7.0

    def test_a_non_live_slider_commits_only_on_release(self, harness, fig, monkeypatch):
        s = gplt.slider("a", 0, 10, 1, live=False)
        released = {"now": False}
        monkeypatch.setattr(imgui, "slider_float", lambda *a, **k: (True, 6.0))
        monkeypatch.setattr(imgui, "is_item_deactivated_after_edit", lambda: released["now"])
        panel = ControlsPanel(FakeWorkspace(fig))
        _frame(panel)
        assert s.value == 1.0  # dragging: draft only
        released["now"] = True
        _frame(panel)
        assert s.value == 6.0

    def test_a_confirmed_button_waits_for_the_answer(self, harness, fig, monkeypatch):
        hits = []
        b = gplt.button("Wipe", lambda: hits.append(1), confirm="Wipe everything?")
        answer = {"value": None}
        monkeypatch.setattr(panel_mod.widgets, "confirm_popup", lambda *a: answer["value"])
        panel = ControlsPanel(FakeWorkspace(fig))
        panel.request_click(b)
        _frame(panel)
        assert hits == []
        answer["value"] = True
        _frame(panel)
        assert hits == [1]

    def test_a_declined_confirmation_does_nothing(self, harness, fig, monkeypatch):
        hits = []
        b = gplt.button("Wipe", lambda: hits.append(1), confirm="Wipe?")
        monkeypatch.setattr(panel_mod.widgets, "confirm_popup", lambda *a: False)
        panel = ControlsPanel(FakeWorkspace(fig))
        panel.request_click(b)
        _frame(panel)
        assert hits == [] and panel._confirm_target is None

    def test_an_invalid_value_becomes_a_toast(self, fig):
        t = gplt.textbox("t")
        panel = ControlsPanel(FakeWorkspace(fig))
        panel._commit(t, "x" * (ctl.MAX_TEXT_LENGTH + 5))
        assert t.value == "" and [x.kind for x in notifications.active()] == ["error"]


class TestWorkspace:
    @pytest.fixture
    def ws(self, fig):
        from glplot.gui.workspace import Workspace

        return Workspace(fig)

    def test_the_panel_is_registered_with_a_chord(self, ws):
        assert "controls" in ws.panels
        assert ws.registry.get("panel.controls").chord is not None

    def test_the_panel_opens_once_when_controls_appear(self, fig, ws):
        ws._sync_controls()
        assert not ws.open["controls"]
        gplt.slider("a", 0, 1)
        ws._sync_controls()
        assert ws.open["controls"]
        ws.open["controls"] = False  # the user closes it
        gplt.slider("b", 0, 1)
        ws._sync_controls()
        assert not ws.open["controls"]

    def test_buttons_become_palette_actions(self, fig, ws):
        hits = []
        gplt.button("Randomize", lambda: hits.append(1), shortcut="Mod+Alt+R", help="shuffle")
        ws._sync_controls()
        action = ws.registry.get("controls.randomize")
        assert action.category == "Controls" and action.description == "shuffle"
        assert action.chord is not None
        action.run()
        assert hits == [1]
        assert any(a.id == "controls.randomize" for a, _, _ in ws.registry.search("random"))
        fig.controls.remove("randomize")
        ws._sync_controls()
        assert ws.registry.get("controls.randomize") is None

    def test_a_taken_chord_is_dropped_with_a_warning(self, fig, ws):
        gplt.button("Clash", lambda: None, shortcut="Mod+K")  # the palette's chord
        ws._sync_controls()
        assert ws.registry.get("controls.clash").chord is None
        assert ws.registry.get("global.palette").chord is not None
        assert any("already" in t.message for t in notifications.active())

    def test_palette_invocation_honours_confirm(self, fig, ws):
        hits = []
        gplt.button("Wipe", lambda: hits.append(1), confirm="Sure?")
        ws._sync_controls()
        ws.open["controls"] = False
        ws.registry.get("controls.wipe").run()
        assert hits == [] and ws.open["controls"]
        assert ws.panels["controls"]._confirm_target is fig.controls["wipe"]

    def test_focus_mode_opens_only_the_controls(self, fig):
        from glplot.gui.workspace import Workspace

        gplt.slider("a", 0, 1)
        fig._controls_focus_mode = True
        ws = Workspace(fig)
        assert [k for k, is_open in ws.open.items() if is_open] == ["controls"]
        ws.reset_layout()
        assert [k for k, is_open in ws.open.items() if is_open] == ["controls"]

    def test_registry_unregister(self, ws):
        assert ws.registry.unregister("global.palette")
        assert ws.registry.get("global.palette") is None
        assert not ws.registry.unregister("global.palette")
