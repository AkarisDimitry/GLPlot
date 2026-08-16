"""Test the screen-sampled function layer.

``FunctionLayer`` inverts the usual bargain. Every other line layer stores a table of
points sampled once, in data space, so zooming magnifies the samples it was born with. This
one stores a *function* and samples it against the screen — about one sample per pixel
column of whatever x range is visible — so the view is an input to its geometry.

Three properties are the whole point of the design, and each has a test class here:

* **constant resolution** — the sample count follows the viewport, never the zoom;
* **constant cost** — a 1e-12 zoom evaluates exactly as many points as the full view;
* **unbounded detail** — zooming re-evaluates, so features finer than any fixed sampling
  resolve as you approach them.

The other half is the engine hook (``_resample_view_layers``) that drives it: it must fire
when the view moves, must *not* fire when nothing happened, and must never let a user
function take the frame down. Those are tested against a stub engine, because the real one
needs a GL context.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.core.layers import FunctionLayer, PolylineLayer
from glplot.engine import _VIEW_SAMPLE_PAD, GPULinePlot


class TestSampling:
    """Test that the curve is sampled where the screen is, at the screen's density."""

    def test_it_samples_across_the_requested_interval(self):
        layer = FunctionLayer(np.sin)
        layer.resample(-3.0, 5.0, 800)
        assert layer.pts is not None
        assert layer.pts[0, 0] == pytest.approx(-3.0, abs=1e-5)
        assert layer.pts[-1, 0] == pytest.approx(5.0, abs=1e-5)

    def test_one_sample_per_pixel_column_by_default(self):
        """The density that matters: what the display can actually resolve, no more."""
        layer = FunctionLayer(np.sin)
        layer.resample(0.0, 1.0, 640)
        assert len(layer.pts) == 640

    def test_samples_per_px_supersamples(self):
        layer = FunctionLayer(np.sin, samples_per_px=2.0)
        layer.resample(0.0, 1.0, 640)
        assert len(layer.pts) == 1280

    def test_a_reversed_interval_is_accepted(self):
        """Cameras can hand back l > r; the curve should still be ascending in x."""
        layer = FunctionLayer(np.sin)
        layer.resample(5.0, -3.0, 400)
        assert layer.pts[0, 0] < layer.pts[-1, 0]

    def test_the_values_are_the_function(self):
        layer = FunctionLayer(lambda x: x**2)
        layer.resample(-2.0, 2.0, 101)
        assert layer.pts[:, 1] == pytest.approx(layer.pts[:, 0] ** 2, abs=1e-4)

    def test_it_is_a_polyline_so_the_existing_renderer_draws_it(self):
        """No renderer changes: the layer type is the one the polyline renderer claims."""
        layer = FunctionLayer(np.sin)
        assert isinstance(layer, PolylineLayer)
        assert layer.layer_type == "polyline"

    def test_a_layer_with_no_function_samples_nothing(self):
        layer = FunctionLayer(None)
        assert layer.resample(0.0, 1.0, 100) is False
        assert layer.pts is None


class TestConstantResolution:
    """Test the headline claim: resolution does not degrade with zoom."""

    def test_the_sample_count_is_identical_at_every_zoom_depth(self):
        layer = FunctionLayer(np.sin)
        counts = []
        span = 10.0
        for _ in range(12):  # 10 -> 1e-11, twelve decades of zoom
            layer.resample(-span, span, 1000)
            counts.append(len(layer.pts))
            span *= 0.1
        assert counts == [1000] * 12

    def test_the_sample_spacing_shrinks_with_the_view(self):
        """Constant *count* over a shrinking window means growing precision."""
        layer = FunctionLayer(np.sin)
        layer.resample(-10.0, 10.0, 1000)
        wide = float(np.diff(layer.pts[:, 0]).mean())
        layer.resample(-1e-6, 1e-6, 1000)
        deep = float(np.diff(layer.pts[:, 0]).mean())
        assert deep < wide * 1e-6

    def test_a_wider_window_gets_more_samples(self):
        """Resolution follows the *viewport*, which is the only thing that should drive it."""
        layer = FunctionLayer(np.sin)
        layer.resample(0.0, 1.0, 400)
        narrow = len(layer.pts)
        layer.resample(0.0, 1.0, 1600)
        assert len(layer.pts) == 4 * narrow


class TestUnboundedDetail:
    """Test that zooming reveals structure a fixed sampling could never have held."""

    def test_zooming_into_sin_one_over_x_resolves_oscillations_a_table_cannot(self):
        """The canonical case, stated as the comparison it is.

        Zoom to x in (1e-3, 2e-3): ``sin(1/x)`` runs through ~80 full periods there. A
        1000-point table sampled once over (-1, 1) has *one* sample in that window, so a
        magnified table shows a straight line. Re-sampling the function shows the 80
        periods, because it computes them on arrival.
        """
        f = lambda x: np.sin(1.0 / x)
        window = (1e-3, 2e-3)

        def sign_changes(y):
            y = y[np.isfinite(y)]
            return int(np.count_nonzero(np.diff(np.signbit(y))))

        stored_x = np.linspace(-1.0, 1.0, 1000)
        inside = (stored_x >= window[0]) & (stored_x <= window[1])
        table_detail = sign_changes(f(stored_x)[inside])

        live = FunctionLayer(f)
        live.resample(*window, 1000)
        live_detail = sign_changes(live.pts[:, 1])

        assert table_detail == 0  # the fixed table has nothing to show here at all
        assert live_detail > 100  # ...the live one resolves every period

    def test_a_fixed_table_holds_one_sample_where_this_layer_computes_a_thousand(self):
        """The same contrast at the level of raw sample counts."""
        xs = np.linspace(-1.0, 1.0, 1000)
        fixed = PolylineLayer(pts=np.column_stack([xs, np.sin(1.0 / xs)]).astype(np.float32))
        window = (fixed.pts[:, 0] > -1e-3) & (fixed.pts[:, 0] < 1e-3)
        assert int(window.sum()) <= 2

        live = FunctionLayer(lambda x: np.sin(1.0 / x))
        live.resample(-1e-3, 1e-3, 1000)
        assert len(live.pts) == 1000


class TestConstantCost:
    """Test that deep zoom is not more expensive — the fractal layer's weakness."""

    def test_evaluation_count_does_not_grow_with_zoom(self):
        calls = {"n": 0}

        def counted(x):
            calls["n"] += len(x)
            return np.sin(x)

        layer = FunctionLayer(counted)
        layer.resample(-10.0, 10.0, 1000)
        shallow = calls["n"]
        calls["n"] = 0
        layer.resample(-1e-12, 1e-12, 1000)
        assert calls["n"] == shallow

    def test_max_samples_caps_the_per_frame_work(self):
        layer = FunctionLayer(np.sin, max_samples=2048)
        layer.resample(0.0, 1.0, 8000)
        assert len(layer.pts) == 2048

    def test_min_samples_keeps_a_sliver_of_a_panel_usable(self):
        layer = FunctionLayer(np.sin, min_samples=64)
        layer.resample(0.0, 1.0, 3)
        assert len(layer.pts) == 64


class TestNeedsResample:
    """Test the guard that keeps an idle figure free."""

    def test_a_fresh_layer_needs_sampling(self):
        assert FunctionLayer(np.sin).needs_resample(0.0, 1.0, 800) is True

    def test_an_unchanged_view_needs_nothing(self):
        layer = FunctionLayer(np.sin)
        layer.resample(0.0, 1.0, 800)
        assert layer.needs_resample(0.0, 1.0, 800) is False

    def test_sub_pixel_jitter_is_ignored(self):
        """A camera can wobble by a float epsilon while nothing is happening."""
        layer = FunctionLayer(np.sin)
        layer.resample(0.0, 1.0, 800)
        assert layer.needs_resample(1e-9, 1.0 + 1e-9, 800) is False

    def test_a_real_pan_triggers_a_resample(self):
        layer = FunctionLayer(np.sin)
        layer.resample(0.0, 1.0, 800)
        assert layer.needs_resample(0.5, 1.5, 800) is True

    def test_a_zoom_triggers_a_resample(self):
        layer = FunctionLayer(np.sin)
        layer.resample(0.0, 1.0, 800)
        assert layer.needs_resample(0.4, 0.6, 800) is True

    def test_a_window_resize_triggers_a_resample(self):
        """More pixels means more resolution is now affordable and wanted."""
        layer = FunctionLayer(np.sin)
        layer.resample(0.0, 1.0, 800)
        assert layer.needs_resample(0.0, 1.0, 1600) is True

    def test_the_tolerance_scales_with_the_window(self):
        """A deep zoom's 'tiny' is absolutely tiny; the guard must be relative, not absolute."""
        layer = FunctionLayer(np.sin)
        layer.resample(0.0, 1e-9, 800)
        assert layer.needs_resample(5e-10, 1.5e-9, 800) is True  # a half-window pan


class TestNonFiniteValues:
    """Test poles and undefined regions — gaps, not spikes and not crashes."""

    def test_inf_becomes_a_gap(self):
        layer = FunctionLayer(lambda x: 1.0 / x)
        layer.resample(-1.0, 1.0, 501)  # odd count puts a sample exactly on x = 0
        assert np.isnan(layer.pts[:, 1]).any()
        assert not np.isinf(layer.pts[:, 1]).any()

    def test_bounds_ignore_the_gaps(self):
        """np.min over a nan would poison autoscale and blank the whole figure."""
        layer = FunctionLayer(lambda x: np.where(np.abs(x) < 0.1, np.nan, x))
        layer.resample(-1.0, 1.0, 201)
        bounds = layer.get_intrinsic_bounds()
        assert bounds is not None
        assert all(np.isfinite(v) for v in bounds)

    def test_an_all_nan_curve_reports_no_bounds(self):
        layer = FunctionLayer(lambda x: np.full_like(x, np.nan))
        layer.resample(0.0, 1.0, 100)
        assert layer.get_intrinsic_bounds() is None

    def test_evaluation_does_not_warn_on_a_pole(self):
        """This runs every frame of a drag; a per-frame RuntimeWarning is unusable."""
        layer = FunctionLayer(lambda x: 1.0 / x)
        with warnings_as_errors():
            layer.resample(-1.0, 1.0, 501)

    def test_a_raising_function_leaves_the_previous_curve(self):
        """A half-typed expression must not take the frame down."""
        state = {"boom": False}

        def f(x):
            if state["boom"]:
                raise ValueError("bad expression")
            return np.sin(x)

        layer = FunctionLayer(f)
        layer.resample(0.0, 1.0, 200)
        good = layer.pts.copy()
        state["boom"] = True
        assert layer.resample(2.0, 3.0, 200) is False
        assert np.array_equal(layer.pts, good)

    def test_a_scalar_returning_function_is_broadcast(self):
        """``lambda x: 3`` is a constant function, not an error."""
        layer = FunctionLayer(lambda x: 3.0)
        layer.resample(0.0, 1.0, 128)
        assert layer.pts[:, 1] == pytest.approx(3.0)


class TestDomain:
    """Test the optional interval outside which the function is never called."""

    def test_the_view_is_clipped_to_the_domain(self):
        seen = {}

        def f(x):
            seen["lo"] = float(x.min())
            return np.sqrt(x)

        layer = FunctionLayer(f, domain=(0.0, np.inf))
        layer.resample(-5.0, 5.0, 400)
        assert seen["lo"] >= 0.0

    def test_a_view_entirely_outside_the_domain_draws_nothing(self):
        layer = FunctionLayer(np.sqrt, domain=(0.0, 10.0))
        layer.resample(-9.0, -1.0, 400)
        assert layer.pts.shape == (0, 2)
        assert layer.get_intrinsic_bounds() is None

    def test_leaving_and_re_entering_the_domain_restores_the_curve(self):
        layer = FunctionLayer(np.sqrt, domain=(0.0, 10.0))
        layer.resample(-9.0, -1.0, 400)
        assert len(layer.pts) == 0
        layer.resample(1.0, 4.0, 400)
        assert len(layer.pts) == 400


# --------------------------------------------------------------------------------------
# The engine hook
# --------------------------------------------------------------------------------------


class _StubCameraController:
    def __init__(self, window):
        self.window = window

    def world_window(self, width, height, padding=1.0):
        cx = 0.5 * (self.window[0] + self.window[1])
        cy = 0.5 * (self.window[2] + self.window[3])
        hw = 0.5 * (self.window[1] - self.window[0]) * padding
        hh = 0.5 * (self.window[3] - self.window[2]) * padding
        return cx - hw, cx + hw, cy - hh, cy + hh


class _StubScene:
    def __init__(self, layers):
        self.layers = list(layers)


class _StubCache:
    refresh_requested = False


class _StubPanel:
    """The slice of ``Panel`` that ``_resample_view_layers`` actually touches."""

    def __init__(self, layers, window=(-10.0, 10.0, -1.0, 1.0), size=(800, 600)):
        self.scene = _StubScene(layers)
        self.camera_controller = _StubCameraController(window)
        self.cache = _StubCache()
        self._size = size

    def pixel_size(self, width, height):
        return self._size


class _StubFrame:
    dirty_scene = False


class _StubEngine:
    """Enough engine to call the real ``_resample_view_layers`` on."""

    width, height = 1280, 720

    def __init__(self, panels):
        self.panels = list(panels)
        self.frame = _StubFrame()

    run = GPULinePlot._resample_view_layers


class TestEngineHook:
    """Test ``GPULinePlot._resample_view_layers`` — what actually drives the layer."""

    def test_it_samples_a_function_layer_on_the_first_frame(self):
        layer = FunctionLayer(np.sin)
        engine = _StubEngine([_StubPanel([layer])])
        engine.run()
        assert layer.pts is not None and len(layer.pts) == 800

    def test_it_samples_the_padded_window_so_the_curve_reaches_the_frame_edge(self):
        """``world_window`` is the pre-margin extent; the MVP insets it, so a curve stopping
        exactly at l and r would stop short of the drawn axes."""
        layer = FunctionLayer(np.sin)
        engine = _StubEngine([_StubPanel([layer], window=(-10.0, 10.0, -1.0, 1.0))])
        engine.run()
        assert layer.pts[0, 0] < -10.0
        assert layer.pts[-1, 0] > 10.0
        assert layer.pts[-1, 0] == pytest.approx(10.0 * _VIEW_SAMPLE_PAD, rel=1e-4)

    def test_an_idle_frame_does_no_work(self):
        calls = {"n": 0}

        def counted(x):
            calls["n"] += 1
            return np.sin(x)

        layer = FunctionLayer(counted)
        engine = _StubEngine([_StubPanel([layer])])
        engine.run()
        engine.run()
        engine.run()
        assert calls["n"] == 1

    def test_moving_the_camera_resamples(self):
        layer = FunctionLayer(np.sin)
        panel = _StubPanel([layer])
        engine = _StubEngine([panel])
        engine.run()
        before = layer.pts.copy()
        panel.camera_controller.window = (0.0, 0.1, -1.0, 1.0)
        engine.run()
        assert not np.array_equal(layer.pts, before)

    def test_it_marks_the_scene_dirty_only_when_something_changed(self):
        layer = FunctionLayer(np.sin)
        panel = _StubPanel([layer])
        engine = _StubEngine([panel])
        engine.run()
        assert engine.frame.dirty_scene is True
        assert panel.cache.refresh_requested is True

        engine.frame.dirty_scene = False
        panel.cache.refresh_requested = False
        engine.run()
        assert engine.frame.dirty_scene is False
        assert panel.cache.refresh_requested is False

    def test_a_hidden_layer_is_not_evaluated(self):
        layer = FunctionLayer(np.sin)
        layer.style.visible = False
        engine = _StubEngine([_StubPanel([layer])])
        engine.run()
        assert layer.pts is None

    def test_ordinary_layers_are_left_alone(self):
        """The walk must be free for the 99% of scenes that hold no function layer."""
        xs = np.linspace(0, 1, 10)
        plain = PolylineLayer(pts=np.column_stack([xs, xs]).astype(np.float32))
        before = plain.pts.copy()
        engine = _StubEngine([_StubPanel([plain])])
        engine.run()
        assert np.array_equal(plain.pts, before)

    def test_every_panel_keeps_its_own_layers_correct(self):
        """Each panel has its own camera, so each has its own visible interval."""
        a, b = FunctionLayer(np.sin), FunctionLayer(np.sin)
        engine = _StubEngine(
            [
                _StubPanel([a], window=(-10.0, 10.0, -1.0, 1.0), size=(800, 600)),
                _StubPanel([b], window=(0.0, 1.0, -1.0, 1.0), size=(400, 300)),
            ]
        )
        engine.run()
        assert len(a.pts) == 800 and len(b.pts) == 400
        assert a.pts[-1, 0] > 5.0 and b.pts[-1, 0] < 2.0

    def test_a_raising_function_does_not_break_the_frame(self):
        def boom(x):
            raise RuntimeError("no")

        good = FunctionLayer(np.sin)
        engine = _StubEngine([_StubPanel([FunctionLayer(boom), good])])
        engine.run()  # must not raise
        assert good.pts is not None


class TestPolylineLodIsNanSafe:
    """A gap in a line used to crash the LOD stride computation (``int(nan)``)."""

    def test_a_nan_gap_does_not_raise(self):
        from glplot.options import EngineOptions
        from glplot.renderers.polyline import PolylineRenderer

        renderer = PolylineRenderer.__new__(PolylineRenderer)
        renderer.options = EngineOptions()

        xs = np.linspace(0.0, 1.0, 256)
        ys = np.sin(xs)
        ys[100] = np.nan
        layer = PolylineLayer(pts=np.column_stack([xs, ys]).astype(np.float32))

        ctx = type(
            "Ctx", (), {"window_world": (0.0, 1.0, -1.0, 1.0), "fb_width": 800, "fb_height": 600}
        )()
        assert renderer._lod_stride(layer, ctx) >= 1


# --------------------------------------------------------------------------------------


class warnings_as_errors:
    """Context manager turning warnings into errors, for the per-frame-warning test."""

    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("error")
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


# --------------------------------------------------------------------------------------
# The expression -> live-curve path (the GUI's "Plot live")
# --------------------------------------------------------------------------------------


class TestCompile1d:
    """``expressions.compile_1d`` — validate once, evaluate many times.

    ``evaluate_1d`` re-parses on every call, which is right for "generate this curve once"
    and wrong for a layer that re-evaluates on every view change: the parse would run per
    frame of a drag to produce the identical code object.
    """

    def test_it_evaluates_the_expression(self):
        from glplot.gui.expressions import compile_1d

        f = compile_1d("sin(x)")
        xs = np.linspace(0.0, 3.0, 17)
        assert f(xs) == pytest.approx(np.sin(xs))

    def test_it_agrees_with_evaluate_1d(self):
        """The two must not drift: same expression, same numbers."""
        from glplot.gui.expressions import compile_1d, evaluate_1d

        xs = np.linspace(-2.0, 2.0, 31)
        variables = {"a": 2.0, "b": 3.0}
        compiled = compile_1d("a*sin(b*x)", variables=variables)(xs)
        assert compiled == pytest.approx(evaluate_1d("a*sin(b*x)", xs, variables=variables))

    def test_a_hostile_expression_is_rejected_at_build_time(self):
        """Not at evaluation time: a render loop has nowhere to report an error."""
        from glplot.gui.expressions import ExpressionError, compile_1d

        with pytest.raises(ExpressionError):
            compile_1d("__import__('os')")

    def test_variables_are_snapshotted(self):
        """A slider moved after plotting must not silently redefine the plotted curve."""
        from glplot.gui.expressions import compile_1d

        variables = {"a": 2.0}
        f = compile_1d("a*x", variables=variables)
        variables["a"] = 100.0
        assert f(np.array([1.0])) == pytest.approx([2.0])

    def test_a_scalar_expression_is_broadcast(self):
        from glplot.gui.expressions import compile_1d

        assert compile_1d("3")(np.linspace(0.0, 1.0, 5)) == pytest.approx(3.0)

    def test_division_by_zero_over_an_array_gives_inf_not_an_error(self):
        """Which the layer then turns into a gap. Matching ``evaluate``'s errstate."""
        from glplot.gui.expressions import compile_1d

        out = compile_1d("1/x")(np.array([-1.0, 0.0, 1.0]))
        assert np.isinf(out[1])

    def test_it_drives_a_function_layer(self):
        """The whole point of the function: an expression that resamples with the view."""
        from glplot.gui.expressions import compile_1d

        layer = FunctionLayer(compile_1d("sin(1/x)"))
        layer.resample(-1.0, 1.0, 500)
        wide = layer.pts[:, 1].copy()
        layer.resample(-1e-4, 1e-4, 500)
        assert not np.array_equal(wide, layer.pts[:, 1])


class TestAddFunctionLayer:
    """``layerops.add_function_layer`` — the GUI's constructor for a live curve."""

    def _plot(self):
        from glplot.engine import GPULinePlot

        return GPULinePlot()

    def test_it_appends_a_sampled_function_layer(self):
        from glplot.gui import layerops

        plot = self._plot()
        layer = layerops.add_function_layer(plot, np.sin, (-5.0, 5.0), label="live")
        assert plot.scene.layers[-1] is layer
        assert isinstance(layer, FunctionLayer)
        assert layer.pts is not None and len(layer.pts) > 1

    def test_it_is_tagged_as_a_line_for_the_scene_and_style_panels(self):
        from glplot.gui import layerops

        plot = self._plot()
        layer = layerops.add_function_layer(plot, np.sin, (-1.0, 1.0), label="live")
        assert layerops.layer_kind(layer) == "line"
        assert layer.metadata.get("live_function") is True

    def test_an_empty_label_is_refused(self):
        """§1.5: the engine's index-based default renames itself as the scene changes."""
        from glplot.gui import layerops

        with pytest.raises(ValueError):
            layerops.add_function_layer(self._plot(), np.sin, (-1.0, 1.0), label="  ")

    def test_the_colour_and_width_land_on_the_style(self):
        from glplot.gui import layerops

        layer = layerops.add_function_layer(
            self._plot(), np.sin, (-1.0, 1.0), label="live", color=(1.0, 0.0, 0.0), width=3.5
        )
        assert layer.style.color == pytest.approx((1.0, 0.0, 0.0, 1.0))
        assert layer.style.line_width == pytest.approx(3.5)


class TestFunctionsPanelPlotLive:
    """The panel action. Deferred like every other one (CONTRACT §1.1)."""

    def _panel(self):
        from glplot.engine import GPULinePlot
        from glplot.gui.panels.functions import FunctionsPanel
        from glplot.gui.workspace import Workspace

        plot = GPULinePlot()
        ws = Workspace(plot)
        return plot, ws, FunctionsPanel(ws)

    def _sync(self, panel, expr):
        panel.expr = expr
        panel._synced_expr = None
        panel._sync_params()

    def test_nothing_happens_before_the_queue_drains(self):
        plot, ws, panel = self._panel()
        self._sync(panel, "sin(1/x)")
        panel.x_min, panel.x_max = -1.0, 1.0
        before = len(plot.scene.layers)
        panel._action_plot_live()
        assert len(plot.scene.layers) == before
        ws.queue.drain(plot)
        assert len(plot.scene.layers) == before + 1

    def test_it_adds_a_live_layer_that_resamples(self):
        plot, ws, panel = self._panel()
        self._sync(panel, "sin(1/x)")
        panel.x_min, panel.x_max = -1.0, 1.0
        panel._action_plot_live()
        ws.queue.drain(plot)

        layer = plot.scene.layers[-1]
        assert isinstance(layer, FunctionLayer)
        wide = layer.pts[:, 1].copy()
        layer.resample(-1e-4, 1e-4, len(wide))
        assert not np.array_equal(wide, layer.pts[:, 1])

    def test_undo_removes_it(self):
        plot, ws, panel = self._panel()
        self._sync(panel, "sin(x)")
        panel.x_min, panel.x_max = -1.0, 1.0
        panel._action_plot_live()
        ws.queue.drain(plot)
        assert len(plot.scene.layers) == 1
        ws.undo.undo()
        ws.queue.drain(plot)
        assert len(plot.scene.layers) == 0

    def test_the_slider_values_are_baked_in(self):
        """Stated in the tooltip, so it had better be true."""
        plot, ws, panel = self._panel()
        self._sync(panel, "a*x")
        panel.params["a"].value = 2.0
        panel.x_min, panel.x_max = 0.0, 1.0
        panel._action_plot_live()
        ws.queue.drain(plot)
        layer = plot.scene.layers[-1]

        panel.params["a"].value = 100.0
        layer.resample(0.0, 1.0, 10)
        assert layer.pts[-1, 1] == pytest.approx(2.0, abs=1e-5)

    def test_a_rejected_expression_reports_in_the_panel_and_adds_nothing(self):
        plot, ws, panel = self._panel()
        panel.expr = "__import__('os')"
        panel.x_min, panel.x_max = -1.0, 1.0
        panel._action_plot_live()
        ws.queue.drain(plot)
        assert plot.scene.layers == []
        assert panel._error and panel._status_ok is False

    def test_a_degenerate_domain_is_refused(self):
        plot, ws, panel = self._panel()
        self._sync(panel, "sin(x)")
        panel.x_min = panel.x_max = 1.0
        panel._action_plot_live()
        ws.queue.drain(plot)
        assert plot.scene.layers == []
        assert panel._status_ok is False


class TestExportSamplesAtImageResolution:
    """A high-resolution ``savefig`` must resample, not magnify the screen's sampling.

    The point of the layer is that resolution follows the pixels that will show it. An
    export writes far more pixels than the window has, so sampling at the window's width
    would reintroduce exactly the faceting this layer exists to avoid.
    """

    def test_pixel_scale_multiplies_the_sample_count(self):
        layer = FunctionLayer(np.sin)
        engine = _StubEngine([_StubPanel([layer], size=(800, 600))])
        engine.run()
        assert len(layer.pts) == 800

        GPULinePlot._resample_view_layers(engine, 4.0)
        assert len(layer.pts) == 3200

    def test_the_next_screen_frame_samples_back_down(self):
        """The count is part of ``needs_resample``, so nothing has to undo the export."""
        layer = FunctionLayer(np.sin)
        engine = _StubEngine([_StubPanel([layer], size=(800, 600))])
        GPULinePlot._resample_view_layers(engine, 4.0)
        assert len(layer.pts) == 3200
        engine.run()
        assert len(layer.pts) == 800

    def test_the_sampled_interval_is_unchanged_by_the_scale(self):
        """Only the density scales; the window is still the window."""
        a, b = FunctionLayer(np.sin), FunctionLayer(np.sin)
        window = (-10.0, 10.0, -1.0, 1.0)
        GPULinePlot._resample_view_layers(_StubEngine([_StubPanel([a], window=window)]), 1.0)
        GPULinePlot._resample_view_layers(_StubEngine([_StubPanel([b], window=window)]), 4.0)
        assert a.pts[0, 0] == pytest.approx(b.pts[0, 0], abs=1e-4)
        assert a.pts[-1, 0] == pytest.approx(b.pts[-1, 0], abs=1e-4)
