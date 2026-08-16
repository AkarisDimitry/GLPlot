"""Test the live GPU fractal: the layer, the adaptive iteration policy, and the API.

The shader itself can only be exercised with a GL context (a subprocess probe, like
``tests/test_outline.py``), but the layer definition, the pyplot constructors and the
zoom-adaptive iteration count are all pure state and test headless. Those are where the
behaviour the user asked for — "recompute on zoom for more precision" — actually lives.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.core.layers import FractalLayer
from glplot.renderers.fractal import _MAX_ITER, _SCHEME_INDEX, FractalRenderer


class _Ctx:
    """The one field ``_adaptive_iter`` reads off a RenderContext."""

    def __init__(self, window_world):
        self.window_world = window_world


class TestFractalLayer:
    def test_defaults_are_a_framed_mandelbrot(self):
        layer = FractalLayer()
        assert layer.layer_type == "fractal"
        assert layer.fractal_type == "mandelbrot"
        assert layer.get_intrinsic_bounds() == layer.extent

    def test_the_extent_drives_autoscale(self):
        layer = FractalLayer(extent=(-1.0, 2.0, -3.0, 4.0))
        assert layer.get_intrinsic_bounds() == (-1.0, 2.0, -3.0, 4.0)

    def test_it_carries_no_sampled_data(self):
        """The whole point: the field is a definition, not a baked grid."""
        layer = FractalLayer()
        assert getattr(layer, "vertices", None) is None
        assert getattr(layer, "pts", None) is None

    def test_identity_equality_like_every_other_layer(self):
        a, b = FractalLayer(), FractalLayer()
        assert a == a and a != b
        assert a in [a, b]


class TestAdaptiveIteration:
    """The zoom-refinement policy: more iterations as the view shrinks."""

    def _renderer(self):
        return FractalRenderer(options=None)

    def test_iterations_rise_with_zoom(self):
        r = self._renderer()
        layer = FractalLayer(extent=(-2.0, 1.0, -1.5, 1.5), max_iter=200)
        full = r._adaptive_iter(layer, _Ctx((-2.0, 1.0, -1.5, 1.5)))
        zoomed = r._adaptive_iter(layer, _Ctx((-0.5, -0.4, -0.05, 0.05)))
        deep = r._adaptive_iter(layer, _Ctx((-0.501, -0.499, -0.001, 0.001)))
        assert full == 200  # at the framing, the base budget
        assert zoomed > full
        assert deep > zoomed

    def test_the_budget_is_capped(self):
        r = self._renderer()
        layer = FractalLayer(extent=(-2.0, 1.0, -1.5, 1.5))
        # An absurd zoom must not ask the shader for more than it will run.
        tiny = r._adaptive_iter(layer, _Ctx((-1e-9, 1e-9, -1e-9, 1e-9)))
        assert tiny <= _MAX_ITER

    def test_zooming_out_never_drops_below_the_base(self):
        r = self._renderer()
        layer = FractalLayer(extent=(-2.0, 1.0, -1.5, 1.5), max_iter=200)
        # A window wider than the fractal (zoomed out) keeps the base budget.
        out = r._adaptive_iter(layer, _Ctx((-10.0, 10.0, -8.0, 8.0)))
        assert out == 200


class TestSchemeMapping:
    def test_common_names_map_to_heatmap_indices(self):
        assert _SCHEME_INDEX["magma"] == 6
        assert _SCHEME_INDEX["viridis"] == 1
        assert _SCHEME_INDEX["turbo"] == 4

    def test_an_unknown_name_is_absent_so_the_renderer_defaults_it(self):
        assert "rainbow6" not in _SCHEME_INDEX


class TestPyplotAPI:
    @pytest.fixture(autouse=True)
    def _fresh(self):
        import glplot.pyplot as gplt

        gplt._CURRENT_PLOT = None
        gplt._ALL_PLOTS.clear()
        gplt._FIGURES_BY_NUM.clear()
        yield
        gplt._CURRENT_PLOT = None
        gplt._ALL_PLOTS.clear()
        gplt._FIGURES_BY_NUM.clear()

    def test_mandelbrot_adds_a_fractal_layer(self):
        import glplot.pyplot as gplt

        gplt.figure("m")
        layer = gplt.mandelbrot()
        assert isinstance(layer, FractalLayer)
        assert layer.fractal_type == "mandelbrot"
        assert layer in gplt.gcf().scene.layers

    def test_mandelbrot_frames_the_view_on_the_set(self):
        import glplot.pyplot as gplt

        gplt.figure("m")
        gplt.mandelbrot(center=(-0.6, 0.0), span=1.4, aspect=1.0)
        # The view was set to the extent, so the set is on screen from the first frame.
        cam = gplt.gcf().camera
        assert np.isfinite(cam.cx) and np.isfinite(cam.cy)

    def test_julia_carries_its_parameter(self):
        import glplot.pyplot as gplt

        gplt.figure("j")
        layer = gplt.julia((-0.8, 0.156))
        assert layer.fractal_type == "julia"
        assert layer.julia_c == pytest.approx((-0.8, 0.156))

    def test_a_deep_span_produces_a_small_extent(self):
        import glplot.pyplot as gplt

        gplt.figure("z")
        layer = gplt.mandelbrot(center=(-0.743, 0.1318), span=0.01, aspect=1.0)
        width = layer.extent[1] - layer.extent[0]
        assert width == pytest.approx(0.02, abs=1e-6)


class TestProgressiveRefinement:
    """The answer to "cuando hago zoom el plot se comienza a ver laggeado".

    The fractal is the one layer whose cost scales with screen *area*: ~700 iterations
    across ~2.5M fragments at a deep zoom. That is a fine still frame and far too slow to
    keep up with a scroll wheel, which is exactly what the lag was. So a moving view gets a
    cheap preview and the settled view gets the full budget, one frame later.

    The motion test reads the *window*, not a drag flag, because the wheel zooms without
    any drag being active.
    """

    def _renderer(self):
        return FractalRenderer(options=None)

    def test_the_first_frame_is_not_treated_as_motion(self):
        """A still figure that is merely being drawn for the first time is not moving."""
        r = self._renderer()
        assert r._is_moving(_Ctx((-2.0, 1.0, -1.5, 1.5))) is False

    def test_a_repeated_window_is_not_motion(self):
        r = self._renderer()
        window = (-2.0, 1.0, -1.5, 1.5)
        r._is_moving(_Ctx(window))
        assert r._is_moving(_Ctx(window)) is False

    def test_a_zoom_is_motion(self):
        r = self._renderer()
        r._is_moving(_Ctx((-2.0, 1.0, -1.5, 1.5)))
        assert r._is_moving(_Ctx((-1.0, 0.5, -0.75, 0.75))) is True

    def test_a_pan_is_motion(self):
        r = self._renderer()
        r._is_moving(_Ctx((-2.0, 1.0, -1.5, 1.5)))
        assert r._is_moving(_Ctx((-1.9, 1.1, -1.5, 1.5))) is True

    def test_float_noise_is_not_motion(self):
        """A camera can jitter by an epsilon; that must not halve the quality forever."""
        r = self._renderer()
        r._is_moving(_Ctx((-2.0, 1.0, -1.5, 1.5)))
        assert r._is_moving(_Ctx((-2.0 + 1e-12, 1.0, -1.5, 1.5))) is False

    def test_the_motion_threshold_is_relative_to_the_window(self):
        """At a 1e-9 window, 1e-10 is a huge pan; an absolute epsilon would miss it."""
        r = self._renderer()
        r._is_moving(_Ctx((0.0, 1e-9, 0.0, 1e-9)))
        assert r._is_moving(_Ctx((1e-10, 1.1e-9, 0.0, 1e-9))) is True

    def test_no_refinement_is_owed_before_anything_is_drawn(self):
        assert self._renderer().consume_refine_request() is False

    def test_the_request_is_consumed_not_peeked(self):
        """Consuming is what makes the refine sequence terminate rather than loop."""
        r = self._renderer()
        r._refine_pending = True
        assert r.consume_refine_request() is True
        assert r.consume_refine_request() is False

    def test_the_interactive_budget_is_a_fraction_of_the_full_one(self):
        from glplot.renderers.fractal import _INTERACTIVE_ITER_FRAC, _MIN_INTERACTIVE_ITER

        r = self._renderer()
        layer = FractalLayer(extent=(-2.0, 1.0, -1.5, 1.5), max_iter=300)
        deep = _Ctx((-0.5001, -0.4999, -0.0001, 0.0001))
        full = r._adaptive_iter(layer, deep)
        cheap = max(int(full * _INTERACTIVE_ITER_FRAC), _MIN_INTERACTIVE_ITER)
        assert cheap < full
        assert cheap >= _MIN_INTERACTIVE_ITER

    def test_the_preview_never_degenerates_into_a_blob(self):
        """Below ~60 iterations the set stops being recognisable as itself."""
        from glplot.renderers.fractal import _INTERACTIVE_ITER_FRAC, _MIN_INTERACTIVE_ITER

        layer = FractalLayer(max_iter=20)  # a deliberately stingy budget
        full = self._renderer()._adaptive_iter(layer, _Ctx((-2.0, 1.0, -1.5, 1.5)))
        assert max(int(full * _INTERACTIVE_ITER_FRAC), _MIN_INTERACTIVE_ITER) == 60


class TestEngineDrainsRefineRequests:
    """``GPULinePlot._refinement_pending`` — the hook that wakes a reactive loop."""

    def _engine(self, renderers):
        from glplot.engine import GPULinePlot

        engine = type(
            "E",
            (),
            {
                "renderer_manager": type("M", (), {"renderers": renderers})(),
                "_refinement_pending": GPULinePlot._refinement_pending,
            },
        )()
        return engine

    def test_no_renderers_means_nothing_pending(self):
        assert self._engine({})._refinement_pending() is False

    def test_a_renderer_without_the_hook_is_ignored(self):
        assert self._engine({"polyline": object()})._refinement_pending() is False

    def test_a_pending_fractal_asks_for_a_frame(self):
        r = FractalRenderer(options=None)
        r._refine_pending = True
        engine = self._engine({"fractal": r})
        assert engine._refinement_pending() is True
        assert engine._refinement_pending() is False  # drained, so the loop settles

    def test_every_renderer_is_drained_not_just_the_first(self):
        """Stopping at the first True would latch the second one's request forever."""
        a, b = FractalRenderer(options=None), FractalRenderer(options=None)
        a._refine_pending = b._refine_pending = True
        engine = self._engine({"a": a, "b": b})
        assert engine._refinement_pending() is True
        assert a._refine_pending is False and b._refine_pending is False
