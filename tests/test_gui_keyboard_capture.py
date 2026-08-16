"""Test that engine hotkeys yield to the HUD when it wants the keyboard.

Focus on GPULinePlot._on_key gating on hud.wants_keyboard() so that typing
into a GUI text field does not fire bare-letter hotkeys (D toggles density,
S writes a PNG). Headless: no OpenGL, no GPU, no window, no file output.

Key constants come from ``glplot.engine.glfw`` rather than a fresh ``import
glfw``: tests/test_camera_anisotropy.py installs a MagicMock into
sys.modules["glfw"] at import time, so a top-level ``import glfw`` here would
bind a mock whose constants never compare equal to the ones _on_key tests,
and every "hotkey did not fire" assertion would pass vacuously.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import pytest

import glplot.pyplot as gplt
from glplot import engine as _engine
from glplot.engine import GPULinePlot

glfw = _engine.glfw


class StubHud:
    """Minimal HUD stand-in recording forwarded keys with a settable capture flag."""

    def __init__(self, wants: bool) -> None:
        self.wants = wants
        self.forwarded: List[Tuple[int, int, int, int]] = []

    def on_key(self, window: Any, key: int, sc: int, action: int, mods: int) -> None:
        """Record the key so we can prove imgui still sees it."""
        self.forwarded.append((key, sc, action, mods))

    def wants_keyboard(self) -> bool:
        """Report whether imgui is capturing keyboard input this frame."""
        return self.wants


class Recorder:
    """Callable that counts invocations and stores their arguments."""

    def __init__(self) -> None:
        self.calls: List[Tuple[tuple, dict]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Record one invocation and do nothing else."""
        self.calls.append((args, kwargs))

    @property
    def called(self) -> bool:
        """True if this recorder was invoked at least once."""
        return bool(self.calls)


@pytest.fixture(autouse=True)
def clean_state():
    """Reset pyplot module state around every test."""
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


def _make_plot(wants_keyboard: bool):
    """Build a headless plot with a stub HUD and recorders on the hotkey targets."""
    plot = GPULinePlot()
    assert plot.window is None, "test must never create a GL window"
    hud = StubHud(wants_keyboard)
    plot.hud = hud
    density = Recorder()
    savefig = Recorder()
    plot.toggle_density = density
    plot.savefig = savefig
    return plot, hud, density, savefig


class TestHudWantsKeyboard:
    """Test the wants_keyboard() guard used by the engine key callback."""

    def test_real_hud_degrades_to_false_without_imgui_impl(self):
        """A HUD-less/GL-less engine must report False so hotkeys keep working."""
        plot = GPULinePlot()
        assert plot.hud.imgui_impl is None
        assert plot.hud.wants_keyboard() is False


class TestKeyboardCaptureBlocksHotkeys:
    """Test that _on_key returns early while the GUI is capturing the keyboard."""

    def test_d_does_not_toggle_density_while_typing(self):
        """Typing 'd' in a text field must not toggle density, but 'd' on canvas must."""
        plot, hud, density, _savefig = _make_plot(wants_keyboard=True)
        plot._on_key(None, glfw.KEY_D, 0, glfw.PRESS, 0)
        assert not density.called
        # Positive control: the same key on the same plot fires once capture ends.
        # Without this the assertion above would pass even if the key never matched.
        hud.wants = False
        plot._on_key(None, glfw.KEY_D, 0, glfw.PRESS, 0)
        assert len(density.calls) == 1

    def test_s_does_not_write_a_file_while_typing(self, tmp_path, monkeypatch):
        """Typing 's' (as in 'sin(x)') must not dump a PNG, but 's' on canvas must save."""
        monkeypatch.chdir(tmp_path)
        plot, hud, _density, savefig = _make_plot(wants_keyboard=True)
        plot._on_key(None, glfw.KEY_S, 0, glfw.PRESS, 0)
        assert not savefig.called
        assert list(tmp_path.iterdir()) == []
        hud.wants = False
        plot._on_key(None, glfw.KEY_S, 0, glfw.PRESS, 0)
        assert len(savefig.calls) == 1

    def test_typing_a_word_fires_no_hotkeys(self, tmp_path, monkeypatch):
        """Typing 'density' must fire nothing and leave the filesystem untouched."""
        monkeypatch.chdir(tmp_path)
        plot, _hud, density, savefig = _make_plot(wants_keyboard=True)
        word = (
            glfw.KEY_D,
            glfw.KEY_E,
            glfw.KEY_N,
            glfw.KEY_S,
            glfw.KEY_I,
            glfw.KEY_T,
            glfw.KEY_Y,
        )
        for key in word:
            plot._on_key(None, key, 0, glfw.PRESS, 0)
        assert not density.called
        assert not savefig.called
        assert list(tmp_path.iterdir()) == []

    def test_key_is_still_forwarded_to_the_hud(self):
        """imgui must still receive the key even though the engine ignores it."""
        plot, hud, _density, _savefig = _make_plot(wants_keyboard=True)
        plot._on_key(None, glfw.KEY_D, 0, glfw.PRESS, 0)
        assert hud.forwarded == [(glfw.KEY_D, 0, glfw.PRESS, 0)]

    def test_shift_state_is_not_tracked_while_captured(self):
        """Shift bookkeeping sits after the early return, so it is skipped when captured."""
        plot, _hud, _density, _savefig = _make_plot(wants_keyboard=True)
        plot.interaction.shift_down = False
        plot._on_key(None, glfw.KEY_LEFT_SHIFT, 0, glfw.PRESS, 0)
        assert plot.interaction.shift_down is False


class TestHotkeysFireWhenNotCaptured:
    """Test that hotkeys still work when no text field has focus."""

    def test_d_toggles_density(self):
        """'d' over the canvas must toggle density exactly once."""
        plot, _hud, density, _savefig = _make_plot(wants_keyboard=False)
        plot._on_key(None, glfw.KEY_D, 0, glfw.PRESS, 0)
        assert len(density.calls) == 1

    def test_s_saves_a_figure(self, tmp_path, monkeypatch):
        """'s' over the canvas must invoke savefig with the export scale."""
        monkeypatch.chdir(tmp_path)
        plot, _hud, _density, savefig = _make_plot(wants_keyboard=False)
        plot._on_key(None, glfw.KEY_S, 0, glfw.PRESS, 0)
        assert len(savefig.calls) == 1
        args, kwargs = savefig.calls[0]
        assert args[0].startswith("plot_") and args[0].endswith(".png")
        assert kwargs["scale"] == plot.options.export_scale
        assert list(tmp_path.iterdir()) == [], "savefig is stubbed; nothing may be written"

    def test_key_is_forwarded_before_dispatch(self):
        """The HUD sees the key first, then the engine acts on it."""
        plot, hud, density, _savefig = _make_plot(wants_keyboard=False)
        plot._on_key(None, glfw.KEY_D, 0, glfw.PRESS, 0)
        assert hud.forwarded == [(glfw.KEY_D, 0, glfw.PRESS, 0)]
        assert density.called

    def test_shift_state_is_tracked(self):
        """Shift press/release bookkeeping runs when the HUD is not capturing."""
        plot, _hud, _density, _savefig = _make_plot(wants_keyboard=False)
        plot._on_key(None, glfw.KEY_LEFT_SHIFT, 0, glfw.PRESS, 0)
        assert plot.interaction.shift_down is True
        plot._on_key(None, glfw.KEY_LEFT_SHIFT, 0, glfw.RELEASE, 0)
        assert plot.interaction.shift_down is False
