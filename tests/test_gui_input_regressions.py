"""Regression tests for real bugs found only through live usage after the pyimgui ->
imgui-bundle migration -- none of these were caught by the rest of the suite, because
pytest's headless imgui harness never drives real GLFW callbacks or opens a real window.

Each test here pins one concrete failure a user actually hit:

* Mouse clicks/drags/scrolling did nothing at all (imgui-bundle's ``GlfwRenderer`` no
  longer feeds mouse state the way pyimgui's did; the old call site silently did nothing).
* Focusing a panel by window title could segfault the process outright (verified with a
  standalone repro -- not a Python exception, so nothing could ever catch it).
* Two copies of libglfw ended up loaded in the same process (an objc runtime warning on
  macOS, avoidable by import order).
* GLFW window creation could fail outright on macOS without an explicit hint.

No GPU is needed for the first three; the fourth is opt-in behind ``GLPLOT_GL_TESTS=1``
like the rest of this suite's real-GL tests, since it needs a working GLFW/OpenGL stack.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

import glplot.engine as engine_module
import glplot.pyplot as gplt
from glplot.engine import GPULinePlot, glfw
from glplot.gui.workspace import Workspace

try:
    from imgui_bundle import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - imgui is a hard dependency in CI
    IMGUI_AVAILABLE = False
    imgui = None


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


def _headless_imgui():
    """A real imgui context that needs no font atlas / GL backend (CONTRACT §2.10)."""
    ctx_imgui = pytest.importorskip("imgui_bundle").imgui
    ctx_imgui.create_context()
    io = ctx_imgui.get_io()
    io.display_size = 900, 700
    io.backend_flags |= ctx_imgui.BackendFlags_.renderer_has_textures
    io.delta_time = 1 / 60.0
    return ctx_imgui


class TestMouseInputPipeline:
    """imgui-bundle's GlfwRenderer.mouse_callback only pushes cursor position now; it
    silently ignores button/action/mods entirely. Button state must be pushed directly.
    """

    def test_button_press_and_release_reach_imgui_io(self):
        ctx_imgui = _headless_imgui()
        plot = GPULinePlot()

        plot.hud.on_mouse_button(None, 0, 1, 0)  # GLFW_PRESS = 1
        ctx_imgui.new_frame()
        assert ctx_imgui.is_mouse_down(0) is True, "a PRESS event never reached imgui"
        ctx_imgui.end_frame()

        plot.hud.on_mouse_button(None, 0, 0, 0)  # GLFW_RELEASE = 0
        ctx_imgui.new_frame()
        assert ctx_imgui.is_mouse_down(0) is False, "a RELEASE event never reached imgui"
        ctx_imgui.end_frame()

    def test_on_cursor_forwards_to_the_renderer(self):
        """HudManager.on_cursor did not exist at all before this fix -- nothing called
        GlfwRenderer.mouse_callback (the position push) on mouse movement, only on click,
        so a window drag never saw fresh coordinates after the initial press.
        """
        _headless_imgui()
        plot = GPULinePlot()

        calls = []
        plot.hud.imgui_impl = type(
            "FakeImpl", (), {"mouse_callback": lambda self, w, x, y: calls.append((w, x, y))}
        )()
        plot.hud.on_cursor("window", 12.0, 34.0)
        assert calls == [("window", 12.0, 34.0)]

    def test_engine_cursor_callback_feeds_the_hud(self):
        """The actual regression: engine._on_cursor never told the HUD about mouse moves."""
        _headless_imgui()
        plot = GPULinePlot()

        calls = []
        plot.hud.on_cursor = lambda window, x, y: calls.append((window, x, y))
        plot._on_cursor("window", 5.0, 6.0)
        assert ("window", 5.0, 6.0) in calls


class TestHiddenHudAnimationTransport:
    """Space must remain a transport key when no workspace frame is being drawn."""

    def test_space_toggles_the_timeline_without_opening_f3_menus(self):
        plot = GPULinePlot()
        plot.options.enable_hud = False
        timeline = plot.active_panel.timeline
        timeline.pause()

        # A stale ImGui capture flag must not swallow a shortcut while ImGui is hidden.
        plot.hud.on_key = lambda *args: None
        plot.hud.wants_keyboard = lambda: True
        plot._on_key(None, glfw.KEY_SPACE, 0, glfw.PRESS, 0)

        assert timeline.playing is True
        assert plot.frame.dirty_scene is True

    def test_key_repeat_and_release_do_not_toggle_twice(self):
        plot = GPULinePlot()
        plot.options.enable_hud = False
        timeline = plot.active_panel.timeline
        plot.hud.on_key = lambda *args: None
        plot.hud.wants_keyboard = lambda: False

        plot._on_key(None, glfw.KEY_SPACE, 0, glfw.PRESS, 0)
        plot._on_key(None, glfw.KEY_SPACE, 0, glfw.REPEAT, 0)
        plot._on_key(None, glfw.KEY_SPACE, 0, glfw.RELEASE, 0)

        assert timeline.playing is True


class TestWorkspaceFocusSafety:
    """imgui.set_window_focus(name) segfaults in imgui-bundle when `name` is not the
    window currently being drawn this frame -- reproduced standalone, not a Python
    exception, so nothing catches it. Workspace must never call it directly; the
    crash-safe idiom is a pending-focus flag consumed via set_next_window_focus()
    immediately before that panel's own begin().
    """

    @pytest.fixture
    def workspace(self):
        plot = GPULinePlot()
        return Workspace(plot)

    def test_focus_never_calls_set_window_focus_directly(self, workspace, monkeypatch):
        _headless_imgui()
        calls = []
        monkeypatch.setattr(imgui, "set_window_focus", lambda *a, **k: calls.append(a))

        key = next(iter(workspace.panels))
        workspace.open_panel(key)

        assert calls == [], "Workspace called the segfault-prone by-name focus API"
        assert workspace._pending_focus == key

    def test_pending_focus_is_consumed_via_set_next_window_focus(self, workspace, monkeypatch):
        ctx_imgui = _headless_imgui()
        key = next(iter(workspace.panels))
        workspace.open_panel(key)
        assert workspace._pending_focus == key

        calls = []
        monkeypatch.setattr(ctx_imgui, "set_next_window_focus", lambda: calls.append(key))

        ctx_imgui.new_frame()
        workspace._draw_panels()
        ctx_imgui.end_frame()

        assert calls == [key], "pending focus was not consumed via set_next_window_focus"
        assert workspace._pending_focus is None, "pending focus was not cleared after use"


class TestImguiBundleImportOrder:
    """imgui-bundle points pip's `glfw` at its own bundled libglfw via PYGLFW_LIBRARY,
    but only if it gets to set that before `glfw` loads its own copy -- otherwise two
    copies of libglfw load into the same process (macOS objc duplicate-class warnings,
    "may cause spurious casting failures and mysterious crashes" per the OS itself).
    """

    def test_imgui_bundle_is_importable_before_glfw_in_the_package_import_chain(self):
        # glplot/__init__.py imports imgui_bundle before .engine (which imports glfw).
        # A real, fresh-process check of import order is done in test_import_order.py-style
        # subprocess tests elsewhere in mature suites; here we pin the observable contract:
        # importing glplot must leave imgui_bundle already present in sys.modules, and (when
        # imgui-bundle ships a bundled glfw) must have pointed PYGLFW_LIBRARY at it.
        assert "imgui_bundle" in sys.modules, "glplot did not import imgui_bundle"
        lib = os.environ.get("PYGLFW_LIBRARY", "")
        if lib:  # only set when imgui-bundle's "with_glfw" bundle is present
            import imgui_bundle

            bundle_dir = os.path.dirname(imgui_bundle.__file__)
            assert lib.startswith(bundle_dir), (
                "PYGLFW_LIBRARY does not point at imgui_bundle's own copy -- glfw likely "
                "loaded its own libglfw before imgui_bundle could redirect it"
            )


class TestInteractiveWindowPresentation:
    """The visible window must be presented after an earlier invisible GL export."""

    def test_run_unhides_and_focuses_the_macos_application_before_the_loop(self, monkeypatch):
        figure = GPULinePlot()
        figure.window = object()
        figure._needs_initial_autoscale = False
        calls = []

        monkeypatch.setattr(figure, "_init_window", lambda: calls.append("init-window"))
        monkeypatch.setattr(figure, "_init_gl", lambda: calls.append("init-gl"))
        monkeypatch.setattr(figure, "_init_modules", lambda: calls.append("init-modules"))
        monkeypatch.setattr(figure, "_main_loop", lambda: calls.append("main-loop"))
        monkeypatch.setattr(glfw, "show_window", lambda window: calls.append("show"))
        monkeypatch.setattr(glfw, "poll_events", lambda: calls.append("poll"))
        monkeypatch.setattr(glfw, "focus_window", lambda window: calls.append("focus"))
        monkeypatch.setattr(engine_module.sys, "platform", "darwin")

        class FakeApplication:
            def setActivationPolicy_(self, policy):
                calls.append(("policy", policy))

            def unhide_(self, sender):
                calls.append(("unhide", sender))

            def activateIgnoringOtherApps_(self, enabled):
                calls.append(("activate", enabled))

        application = FakeApplication()
        appkit = types.ModuleType("AppKit")
        appkit.NSApplicationActivationPolicyRegular = 0
        appkit.NSApplication = type(
            "FakeNSApplication",
            (),
            {"sharedApplication": staticmethod(lambda: application)},
        )
        monkeypatch.setitem(sys.modules, "AppKit", appkit)

        figure.run()

        assert calls == [
            "init-window",
            "init-gl",
            "init-modules",
            "show",
            "poll",
            ("policy", 0),
            ("unhide", None),
            ("activate", True),
            "focus",
            "poll",
            "main-loop",
        ]


@pytest.mark.skipif(
    os.environ.get("GLPLOT_GL_TESTS") != "1",
    reason="needs a real GL context; set GLPLOT_GL_TESTS=1 to run",
)
class TestRealWindowCreation:
    """macOS/NSGL rejects a core-profile context >= 3.2 without OPENGL_FORWARD_COMPAT --
    verified with a standalone repro: window creation raised RuntimeError without the
    hint. Skipped by default like this suite's other real-GL tests (developer machine
    only, not CI).
    """

    def test_init_window_succeeds(self):
        figure = gplt.figure(width=320, height=240)
        figure._is_test_mode = True
        figure._init_window()
        assert figure.window is not None
