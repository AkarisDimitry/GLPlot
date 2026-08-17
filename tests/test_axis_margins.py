"""Test the axis content inset (``EngineOptions.axis_margin_*``).

The gutters are read by three sites that must never disagree: the projection
(``CameraController.mvp``), its exact inverse (``CameraController.screen_to_world``) and
the density resolve pass's UV clip box. They used to be three hand-synced copies of the
literals ``60/20/40/20``.

Focus on the projection arithmetic and the export path's margin isolation, without
requiring OpenGL or GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

import glplot.utils.export as export_mod
from glplot.controllers import CameraController
from glplot.core.legacy import CameraState
from glplot.options import DEFAULT_AXIS_MARGINS, EngineOptions, resolve_axis_margins
from glplot.utils.export import ExportManager

W, H = 1280, 720

#: A gutter wider than stock, as the workstation's rail forces (RAIL_WIDTH + 60).
RAIL_MARGINS = (98.0, 20.0, 40.0, 20.0)


def _controller(margins=None) -> CameraController:
    """A controller on a deliberately off-centre, anisotropic camera."""
    options = EngineOptions()
    if margins is not None:
        options.set_axis_margins(*margins)
    camera = CameraState()
    camera.cx, camera.cy, camera.zoom_x, camera.zoom_y = 3.0, -7.0, 0.02, 0.05
    return CameraController(camera, options)


def _project(ctrl: CameraController, wx: float, wy: float):
    """World -> screen through the same mvp the renderers use."""
    ndc = ctrl.mvp(W, H) @ np.array([wx, wy, 0.0, 1.0], dtype=np.float32)
    if ndc[3] != 0:
        ndc = ndc / ndc[3]
    return (ndc[0] + 1.0) * 0.5 * W, (1.0 - ndc[1]) * 0.5 * H


class TestAxisMarginOptions:
    """Test the option fields and their accessors."""

    def test_defaults_are_the_historical_literals(self):
        """The stock gutters must stay 60/20/40/20 -- this is a published library."""
        options = EngineOptions()
        assert options.axis_margins() == (60.0, 20.0, 40.0, 20.0)
        assert DEFAULT_AXIS_MARGINS == (60.0, 20.0, 40.0, 20.0)

    def test_set_axis_margins_round_trips(self):
        """set_axis_margins is the inverse of axis_margins."""
        options = EngineOptions()
        options.set_axis_margins(98.0, 21.0, 41.0, 22.0)
        assert options.axis_margins() == (98.0, 21.0, 41.0, 22.0)

    def test_resolve_falls_back_for_option_objects_without_the_fields(self):
        """Callers pass None or hand-rolled stubs as options; they get the defaults."""

        class Stub:
            pass

        assert resolve_axis_margins(None) == DEFAULT_AXIS_MARGINS
        assert resolve_axis_margins(Stub()) == DEFAULT_AXIS_MARGINS


class TestMarginRefactorIsANoOp:
    """Test that promoting the literals to options did not move a single pixel."""

    def test_default_mvp_matches_the_preexisting_literals(self):
        """The default projection must be bitwise identical to the hardcoded version."""
        ctrl = _controller()
        left, right, bottom, top = ctrl.world_window(W, H)

        margin_l, margin_r, margin_b, margin_t = 60.0, 20.0, 40.0, 20.0
        rl, tb = (right - left), (top - bottom)
        sx = (W - margin_l - margin_r) / W
        tx = (margin_l - margin_r) / W
        sy = (H - margin_t - margin_b) / H
        ty = (margin_b - margin_t) / H
        m00, m03 = 2.0 / rl, -(right + left) / rl
        m11, m13 = 2.0 / tb, -(top + bottom) / tb
        expected = np.array(
            [
                [sx * m00, 0.0, 0.0, sx * m03 + tx],
                [0.0, sy * m11, 0.0, sy * m13 + ty],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        assert np.array_equal(ctrl.mvp(W, H), expected)

    def test_controller_tolerates_options_of_none(self):
        """test_camera_projections builds CameraController(cam, None); keep it working."""
        ctrl = CameraController(CameraState(), None)
        assert ctrl._margins() == DEFAULT_AXIS_MARGINS
        assert np.all(np.isfinite(ctrl.mvp(W, H)))


class TestMvpAndScreenToWorldAgree:
    """The one thing not to get wrong: the projection and its inverse must match.

    If they drift, the cursor silently stops matching the data underneath it.
    """

    @pytest.mark.parametrize(
        "margins",
        [(60.0, 20.0, 40.0, 20.0), RAIL_MARGINS, (140.0, 55.0, 70.0, 33.0), (0.0, 0.0, 0.0, 0.0)],
    )
    def test_world_screen_world_round_trip_is_identity(self, margins):
        """A world point projected to screen and back must land where it started."""
        ctrl = _controller(margins)
        left, right, bottom, top = ctrl.world_window(W, H)
        for wx in np.linspace(left, right, 7):
            for wy in np.linspace(bottom, top, 7):
                sx, sy = _project(ctrl, wx, wy)
                rx, ry = ctrl.screen_to_world(sx, sy, W, H)
                assert np.allclose([rx, ry], [wx, wy], rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("margins", [(60.0, 20.0, 40.0, 20.0), RAIL_MARGINS])
    def test_both_sites_read_the_same_option(self, margins):
        """Changing the option must move mvp and screen_to_world together, not one of them."""
        ctrl = _controller(margins)
        left, _, bottom, _ = ctrl.world_window(W, H)
        sx, sy = _project(ctrl, left, bottom)
        # The bottom-left world corner projects to exactly (margin_l, height - margin_b).
        assert np.allclose([sx, sy], [margins[0], H - margins[2]], atol=1e-3)
        assert np.allclose(ctrl.screen_to_world(sx, sy, W, H), [left, bottom], atol=1e-4)


class TestRailClearsTheYAxis:
    """The user-facing defect: the icon rail painted over every Y tick label."""

    def test_stock_gutter_puts_labels_under_the_rail(self):
        """Reproduces the original bug: the left spine sits at x=60, labels start at 15."""
        from glplot.gui.workspace import RAIL_WIDTH

        ctrl = _controller((60.0, 20.0, 40.0, 20.0))
        left, _, bottom, _ = ctrl.world_window(W, H)
        spine_x, _ = _project(ctrl, left, bottom)
        assert spine_x == pytest.approx(60.0, abs=1e-3)
        # A 4-glyph label right-aligned at spine-8 still starts left of the rail's edge.
        assert spine_x - 8.0 - 28.0 < RAIL_WIDTH

    def test_rail_gutter_clears_the_rail(self):
        """With the rail-aware gutter even a wide label starts clear of the rail."""
        from glplot.gui.workspace import RAIL_AXIS_MARGIN_L, RAIL_WIDTH

        ctrl = _controller((RAIL_AXIS_MARGIN_L, 20.0, 40.0, 20.0))
        left, _, bottom, _ = ctrl.world_window(W, H)
        spine_x, _ = _project(ctrl, left, bottom)
        assert spine_x == pytest.approx(RAIL_AXIS_MARGIN_L, abs=1e-3)
        # "-100000" measures ~49px in ProggyClean; it must still clear the rail.
        assert spine_x - 8.0 - 49.0 > RAIL_WIDTH


class TestExportUsesDefaultMargins:
    """An exported PNG has no rail, so it must never inherit the rail-sized gutter."""

    def test_context_manager_swaps_and_restores(self):
        """Inside: the defaults. Outside: exactly what the caller had."""
        options = EngineOptions()
        options.set_axis_margins(*RAIL_MARGINS)

        with export_mod._default_axis_margins(options):
            assert options.axis_margins() == DEFAULT_AXIS_MARGINS
        assert options.axis_margins() == RAIL_MARGINS

    def test_context_manager_restores_on_exception(self):
        """A failed save must not leave the live view mis-projected."""
        options = EngineOptions()
        options.set_axis_margins(*RAIL_MARGINS)

        with pytest.raises(RuntimeError):
            with export_mod._default_axis_margins(options):
                raise RuntimeError("save blew up")
        assert options.axis_margins() == RAIL_MARGINS

    def test_savefig_projects_with_default_margins_when_the_gui_raised_them(
        self, tmp_path, monkeypatch
    ):
        """THE regression test: drive the real savefig and capture the mvp it renders with.

        The GUI widens axis_margin_l to clear its rail. savefig builds its projection from
        the same camera_controller, so without isolation every exported image grows a
        mystery 98px empty band down its left edge.
        """
        width, height = 64, 48
        seen = {}

        class _RendererManager:
            def draw_exact(self, layers, ctx):
                # Capture what the export actually rendered with.
                seen["mvp"] = np.array(ctx.mvp, copy=True)
                seen["margin_l"] = engine.options.axis_margin_l

            def draw_axes(self, axis_manager, ctx):
                # The export draws the axis pass (frame, grid, ticks) after the data.
                # Recorded so the assertion below can prove it ran on the same stock
                # gutters as the data — an axis frame drawn with the rail's margin would
                # be the same 98px band this whole test exists to keep out of the file.
                seen["axes_margin_l"] = engine.options.axis_margin_l

        class _AxisManager:
            def update(self, ctx):
                pass

        class _Runtime:
            current_mode = None

        class _Policy:
            runtime = _Runtime()

        class _Lines:
            count = 0

        class _Scene:
            lines = _Lines()

        class _Engine:
            # Literals, not the closure vars: a class body cannot read enclosing locals.
            fb_width = 64
            fb_height = 48
            width = 64
            display_density = False
            _needs_initial_autoscale = False
            scene = _Scene()
            policy = _Policy()
            renderer_manager = _RendererManager()
            axis_manager = _AxisManager()

            def __init__(self):
                self.options = EngineOptions()
                camera = CameraState()
                camera.cx, camera.cy, camera.zoom_x, camera.zoom_y = 3.0, -7.0, 0.02, 0.05
                self.camera_controller = CameraController(camera, self.options)
                # savefig renders per panel; a single full-window panel is the default figure.
                from glplot.core.panel import Panel

                self.active_panel_index = 0
                self.panels = [Panel(self.options)]

            def _get_ndc_transform(self, window):
                return (1.0, 1.0), (0.0, 0.0)

            def _apply_blending_policy(self):
                pass

            def _get_all_layers(self):
                return []

            def _is_pure_3d_scene(self):
                # A 2D scene, so the export runs the 2D axis pass.
                return False

            def _density_active(self):
                # A 2D exact scene: no density accumulator.
                return False

        engine = _Engine()
        # The workstation is mounted: the gutter is rail-sized.
        engine.options.set_axis_margins(*RAIL_MARGINS)

        for name, value in [
            ("glGenFramebuffers", lambda n: 1),
            ("glGenTextures", lambda n: 2),
            ("glGenRenderbuffers", lambda n: 3),
            ("glBindFramebuffer", lambda *a: None),
            ("glBindTexture", lambda *a: None),
            ("glTexImage2D", lambda *a: None),
            ("glFramebufferTexture2D", lambda *a: None),
            ("glBindRenderbuffer", lambda *a: None),
            ("glRenderbufferStorage", lambda *a: None),
            ("glFramebufferRenderbuffer", lambda *a: None),
            ("glCheckFramebufferStatus", lambda *a: export_mod.GL_FRAMEBUFFER_COMPLETE),
            ("glGetIntegerv", lambda *a: (0, 0, width, height)),
            ("glViewport", lambda *a: None),
            ("glClearColor", lambda *a: None),
            ("glClear", lambda *a: None),
            ("glFinish", lambda *a: None),
            ("glReadPixels", lambda *a: bytes(width * height * 3)),
            ("glDeleteFramebuffers", lambda *a: None),
            ("glDeleteTextures", lambda *a: None),
            ("glDeleteRenderbuffers", lambda *a: None),
        ]:
            monkeypatch.setattr(export_mod, name, value)

        ExportManager(engine).savefig(str(tmp_path / "out.png"), scale=1.0)

        # 1. The render ran on the stock gutters, not the rail's.
        assert seen["margin_l"] == DEFAULT_AXIS_MARGINS[0]

        # 2. The captured mvp is the default-margin mvp, NOT the rail-margin one.
        engine.options.set_axis_margins(*DEFAULT_AXIS_MARGINS)
        expected = engine.camera_controller.mvp(width, height)
        engine.options.set_axis_margins(*RAIL_MARGINS)
        rail_mvp = engine.camera_controller.mvp(width, height)

        assert np.allclose(seen["mvp"], expected)
        assert not np.allclose(seen["mvp"], rail_mvp), "export inherited the rail gutter"

        # 3. The axis pass ran, and on the same stock gutters as the data.
        assert seen["axes_margin_l"] == DEFAULT_AXIS_MARGINS[0]

        # 4. The live view's gutter survived the export untouched.
        assert engine.options.axis_margins() == RAIL_MARGINS
