"""Zoom-adaptive decimation for polylines (PolylineRenderer._lod_stride).

A polyline oversampled for its on-screen size (zoomed far out) is drawn at a coarser stride so
the segment count stays proportional to visible detail -- fixing the FPS falling off as you
zoom out. Pure numpy, no GL context needed for the stride decision.
"""

from __future__ import annotations

import numpy as np

from glplot.core.context import RenderContext
from glplot.core.layers import PolylineLayer, ScatterLayer
from glplot.options import EngineOptions, RenderMode
from glplot.renderers.polyline import PolylineRenderer
from glplot.renderers.scatter import ScatterRenderer


def _ctx(world):
    """A minimal RenderContext for a given (l, r, b, t) world window at 1000x800 px."""
    return RenderContext(
        mvp=np.eye(4, dtype=np.float32),
        window_world=world,
        width_px=1000,
        height_px=800,
        fb_width=1000,
        fb_height=800,
        mode=RenderMode.EXACT,
    )


def _line(n):
    x = np.linspace(0.0, 100.0, n)
    return PolylineLayer(pts=np.column_stack([x, np.sin(x)]).astype(np.float32))


class TestPolylineLOD:
    def test_no_decimation_when_line_fills_the_view(self):
        r = PolylineRenderer(EngineOptions())
        # World window matches the data extent: the line uses the whole viewport.
        assert r._lod_stride(_line(100_000), _ctx((0.0, 100.0, -1.5, 1.5))) == 1

    def test_decimates_when_zoomed_far_out(self):
        r = PolylineRenderer(EngineOptions())
        # The data (0..100) sits in a tiny corner of a huge window -> heavy decimation.
        stride = r._lod_stride(_line(100_000), _ctx((-1e5, 1e5, -1e5, 1e5)))
        assert stride > 1

    def test_stride_grows_as_you_zoom_further_out(self):
        r = PolylineRenderer(EngineOptions())
        line = _line(100_000)
        s_near = r._lod_stride(line, _ctx((-1e4, 1e4, -1e4, 1e4)))
        s_far = r._lod_stride(line, _ctx((-1e6, 1e6, -1e6, 1e6)))
        assert s_far >= s_near

    def test_stride_is_a_power_of_two(self):
        r = PolylineRenderer(EngineOptions())
        s = r._lod_stride(_line(100_000), _ctx((-1e5, 1e5, -1e5, 1e5)))
        assert s & (s - 1) == 0  # power of two

    def test_small_lines_are_never_decimated(self):
        r = PolylineRenderer(EngineOptions())
        assert r._lod_stride(_line(5), _ctx((-1e9, 1e9, -1e9, 1e9))) == 1

    def test_lod_disabled_keeps_full_detail(self):
        opts = EngineOptions()
        opts.lod_enabled = False
        r = PolylineRenderer(opts)
        assert r._lod_stride(_line(100_000), _ctx((-1e9, 1e9, -1e9, 1e9))) == 1

    def test_decimated_indices_keep_endpoints_and_order(self):
        r = PolylineRenderer(EngineOptions())
        idx = r._decimated_indices(1000, 8)
        assert idx[0] == 0 and idx[-1] == 999  # first and last always kept
        assert np.all(np.diff(idx) > 0)  # strictly increasing (connectivity)
        assert len(idx) < 1000  # actually decimated


def _cloud(n):
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(n, 2)).astype(np.float32)
    return ScatterLayer(pts=pts)


class TestScatterLOD:
    def test_no_decimation_when_cloud_fills_view(self):
        r = ScatterRenderer(EngineOptions())
        assert r._lod_stride(_cloud(1_000_000), _ctx((-4.0, 4.0, -4.0, 4.0))) == 1

    def test_decimates_when_zoomed_far_out(self):
        r = ScatterRenderer(EngineOptions())
        assert r._lod_stride(_cloud(1_000_000), _ctx((-1e4, 1e4, -1e4, 1e4))) > 1

    def test_stride_grows_as_you_zoom_further_out(self):
        r = ScatterRenderer(EngineOptions())
        cloud = _cloud(1_000_000)
        near = r._lod_stride(cloud, _ctx((-1e2, 1e2, -1e2, 1e2)))
        far = r._lod_stride(cloud, _ctx((-1e5, 1e5, -1e5, 1e5)))
        assert far >= near

    def test_small_clouds_never_decimated(self):
        r = ScatterRenderer(EngineOptions())
        assert r._lod_stride(_cloud(1000), _ctx((-1e9, 1e9, -1e9, 1e9))) == 1

    def test_lod_disabled_keeps_full_detail(self):
        opts = EngineOptions()
        opts.lod_enabled = False
        r = ScatterRenderer(opts)
        assert r._lod_stride(_cloud(1_000_000), _ctx((-1e9, 1e9, -1e9, 1e9))) == 1

    def test_stride_is_power_of_two(self):
        r = ScatterRenderer(EngineOptions())
        s = r._lod_stride(_cloud(1_000_000), _ctx((-1e4, 1e4, -1e4, 1e4)))
        assert s & (s - 1) == 0
