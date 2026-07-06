"""
Robustness and regression tests for fixes made in the rendering pipeline.

Covers:
- quiver() batching (2 layers for any N, NaN separators, arrowhead geometry)
- imshow() point size formula (no 8px cap, scales with resolution)
- savefig() fallback path (no GL window → render_preview)
- default_global_alpha == 1.0
- fill_between() vertex interleaving for GL_TRIANGLE_STRIP
- bar() quad indices and geometry
- WIDE_SEGMENT_INSTANCED_VS shader: u_window uniform + gl_ClipDistance writes
- PolylineLayer NaN passthrough (GPU data prep doesn't crash on NaN input)
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.options import EngineOptions
from glplot.utils.shaders import WIDE_SEGMENT_INSTANCED_VS

# ---------------------------------------------------------------------------
# Shared fixture: reset pyplot state around every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


# ---------------------------------------------------------------------------
# quiver() batching
# ---------------------------------------------------------------------------


class TestQuiverBatching:
    """quiver() must create exactly 2 layers (1 polyline shaft + 1 patch head)
    regardless of the number of arrows N, replacing the old per-arrow approach
    that created 2*N layers and was catastrophically slow for large grids."""

    def test_two_arrows_creates_two_layers(self):
        artists = gplt.quiver([0, 1], [0, 1], [1, 0], [0, 1], scale=0.2)
        assert len(artists) == 2
        types = [a.layer_type for a in artists]
        assert "polyline" in types
        assert "patch" in types

    def test_large_grid_creates_exactly_two_layers(self):
        N = 31 * 31  # 961 arrows like the gallery example
        xs = np.linspace(-1, 1, 31)
        ys = np.linspace(-1, 1, 31)
        xx, yy = np.meshgrid(xs, ys)
        us = -yy.ravel()
        vs = xx.ravel()
        artists = gplt.quiver(xx.ravel(), yy.ravel(), us, vs, scale=0.05)
        assert len(artists) == 2, f"Expected 2 layers, got {len(artists)}"
        total_scene_quiver = [
            a for a in gplt.gcf().scene.layers if a.metadata.get("artist_group") == "quiver"
        ]
        assert len(total_scene_quiver) == 2

    def test_all_artists_tagged_as_quiver(self):
        artists = gplt.quiver([0, 1, 2], [0, 0, 0], [1, 1, 1], [0, 0, 0], scale=0.1)
        assert all(a.metadata.get("artist_group") == "quiver" for a in artists)

    def test_shaft_layer_is_polyline_with_nan_separators(self):
        N = 4
        artists = gplt.quiver(
            [0, 1, 2, 3],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            scale=0.5,
        )
        shaft = next(a for a in artists if a.layer_type == "polyline")
        pts = shaft.pts
        # Layout: [start0, end0, NaN, start1, end1, NaN, ..., startN-1, endN-1]
        # => 3*N - 1 rows for N arrows
        assert pts.shape == (3 * N - 1, 2), f"Expected ({3 * N - 1}, 2) pts, got {pts.shape}"
        # Every 3rd row starting at index 2 must be NaN (separators)
        nan_rows = pts[2::3]
        assert np.all(np.isnan(nan_rows)), "Separator rows must be NaN"
        # Start and end rows must NOT be NaN
        start_rows = pts[0::3]
        end_rows = pts[1::3]
        assert not np.any(np.isnan(start_rows))
        assert not np.any(np.isnan(end_rows))

    def test_arrowhead_patch_has_three_vertices_per_arrow(self):
        N = 5
        xs = np.arange(N, dtype=float)
        artists = gplt.quiver(xs, np.zeros(N), np.ones(N), np.zeros(N), scale=0.2)
        head = next(a for a in artists if a.layer_type == "patch")
        # M valid arrows (all have non-zero length) → M*3 vertices
        assert head.vertices.shape == (N * 3, 2)

    def test_single_arrow_no_nan_separators(self):
        artists = gplt.quiver([0.0], [0.0], [1.0], [0.0], scale=1.0)
        shaft = next(a for a in artists if a.layer_type == "polyline")
        # N=1: 3*1-1 = 2 rows (just start + end, no separator)
        assert shaft.pts.shape == (2, 2)
        assert not np.any(np.isnan(shaft.pts))

    def test_zero_length_arrows_excluded_from_head_patch(self):
        # u=0, v=0 → zero length; no head patch should be created
        artists = gplt.quiver([0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], scale=1.0)
        patch_artists = [a for a in artists if a.layer_type == "patch"]
        assert len(patch_artists) == 0

    def test_mixed_zero_and_nonzero_arrows(self):
        # 3 arrows: 2 valid, 1 zero-length; head patch should have 2*3=6 vertices
        artists = gplt.quiver(
            [0, 1, 2],
            [0, 0, 0],
            [1, 0, 1],
            [0, 0, 0],  # middle arrow has zero length
            scale=0.3,
        )
        head = next((a for a in artists if a.layer_type == "patch"), None)
        assert head is not None
        assert head.vertices.shape == (2 * 3, 2)

    def test_arrowhead_tip_is_at_shaft_end(self):
        # Single arrow from (0,0) pointing right with scale=1 → tip at (1, 0)
        artists = gplt.quiver([0.0], [0.0], [1.0], [0.0], scale=1.0)
        head = next(a for a in artists if a.layer_type == "patch")
        tip = head.vertices[0]  # first vertex of head is the tip
        assert np.allclose(tip, [1.0, 0.0], atol=1e-5)


# ---------------------------------------------------------------------------
# imshow() point size
# ---------------------------------------------------------------------------


class TestImshowPointSize:
    """imshow() must use max(1.0, 650/max(rows,cols)) with NO upper cap.
    The old cap of min(8.0, ...) was removed so sparse images fill their pixels."""

    def test_small_matrix_gets_large_point_size(self):
        matrix = np.zeros((2, 2), dtype=np.float32)
        layer = gplt.imshow(matrix)
        expected = max(1.0, 650.0 / 2)  # 325.0
        assert math.isclose(
            layer.style.point_size, expected, rel_tol=1e-3
        ), f"Expected point_size≈{expected}, got {layer.style.point_size}"

    def test_medium_matrix_scales_correctly(self):
        matrix = np.zeros((50, 50), dtype=np.float32)
        layer = gplt.imshow(matrix)
        expected = max(1.0, 650.0 / 50)  # 13.0
        assert math.isclose(layer.style.point_size, expected, rel_tol=1e-3)

    def test_large_matrix_clamps_to_one(self):
        matrix = np.zeros((1000, 1000), dtype=np.float32)
        layer = gplt.imshow(matrix)
        expected = max(1.0, 650.0 / 1000)  # clamped to 1.0
        assert math.isclose(layer.style.point_size, expected, rel_tol=1e-3)

    def test_wide_matrix_uses_longest_dimension(self):
        matrix = np.zeros((10, 200), dtype=np.float32)
        layer = gplt.imshow(matrix)
        expected = max(1.0, 650.0 / 200)  # 3.25
        assert math.isclose(layer.style.point_size, expected, rel_tol=1e-3)

    def test_custom_point_size_kwarg_overrides_formula(self):
        matrix = np.zeros((4, 4), dtype=np.float32)
        layer = gplt.imshow(matrix, s=42.0)
        assert math.isclose(layer.style.point_size, 42.0, rel_tol=1e-3)


# ---------------------------------------------------------------------------
# savefig() fallback path
# ---------------------------------------------------------------------------


class TestSavefigFallback:
    """When plot.window is None (no GL context created yet), savefig() must
    call render_preview() instead of trying to create a headless GLFW window,
    which crashes on macOS when Python is not an app bundle."""

    def test_no_window_calls_render_preview(self, tmp_path):
        gplt.plot([0, 1], [0, 1])
        outfile = str(tmp_path / "out.png")
        with patch("glplot.utils.preview.render_preview") as mock_preview:
            gplt.savefig(outfile)
        mock_preview.assert_called_once()
        call_args = mock_preview.call_args
        assert call_args[0][1] == outfile

    def test_window_present_skips_render_preview(self, tmp_path):
        gplt.plot([0, 1], [0, 1])
        plot = gplt.gcf()
        plot.window = MagicMock()  # simulate an open GL window
        plot.savefig = MagicMock()
        outfile = str(tmp_path / "out.png")
        with patch("glplot.utils.preview.render_preview") as mock_preview:
            gplt.savefig(outfile)
        mock_preview.assert_not_called()
        plot.savefig.assert_called_once()


# ---------------------------------------------------------------------------
# default_global_alpha
# ---------------------------------------------------------------------------


class TestDefaultGlobalAlpha:
    """default_global_alpha must be 1.0.  The prior value of 0.20 caused lines
    with explicit alpha=0.17 to render at only 3.4% effective opacity."""

    def test_engine_options_default_is_one(self):
        opts = EngineOptions()
        assert opts.default_global_alpha == 1.0

    def test_new_figure_global_alpha_is_one(self):
        fig = gplt.figure()
        assert fig.options.default_global_alpha == 1.0
        assert fig.global_alpha == 1.0


# ---------------------------------------------------------------------------
# fill_between() vertex interleaving
# ---------------------------------------------------------------------------


class TestFillBetweenGeometry:
    """fill_between() must interleave top/bottom vertices for GL_TRIANGLE_STRIP.
    Columns [0::2] are y1 (top), columns [1::2] are y2 (bottom), and x values
    must match the input x array at both positions."""

    def test_vertices_shape(self):
        x = [0.0, 1.0, 2.0]
        layer = gplt.fill_between(x, [1, 2, 1], 0)
        verts = layer.scene.layers[-1].vertices
        assert verts.shape == (6, 2)

    def test_strip_interleaving_top_bottom(self):
        x = np.array([0.0, 1.0, 2.0])
        y1 = np.array([3.0, 4.0, 3.0])
        y2 = np.array([1.0, 2.0, 1.0])
        layer = gplt.fill_between(x, y1, y2)
        verts = layer.scene.layers[-1].vertices
        # Even rows: (x[i], y1[i]) — top boundary
        np.testing.assert_allclose(verts[0::2, 0], x)
        np.testing.assert_allclose(verts[0::2, 1], y1)
        # Odd rows: (x[i], y2[i]) — bottom boundary
        np.testing.assert_allclose(verts[1::2, 0], x)
        np.testing.assert_allclose(verts[1::2, 1], y2)

    def test_scalar_y2_broadcasts(self):
        x = np.array([0.0, 1.0])
        layer = gplt.fill_between(x, [5.0, 6.0], 0.0)
        verts = layer.scene.layers[-1].vertices
        np.testing.assert_allclose(verts[1::2, 1], [0.0, 0.0])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            gplt.fill_between([0, 1], [1, 2, 3], 0)


# ---------------------------------------------------------------------------
# bar() geometry
# ---------------------------------------------------------------------------


class TestBarGeometry:
    """Each bar must be a 4-vertex quad with indices [0,1,2,0,2,3] forming two
    counter-clockwise triangles (lower-left → upper-left → upper-right → lower-right).

    Note: gplt.bar() and add_patch() return the plot object (GPULinePlot), not
    the layer itself.  Layers are accessed via gcf().scene.layers."""

    def _bar_layer(self, index: int = -1):
        layers = [l for l in gplt.gcf().scene.layers if l.layer_type == "patch"]
        return layers[index]

    def test_single_bar_quad_corners(self):
        gplt.bar([0.0], [2.0], width=1.0)
        verts = self._bar_layer().vertices
        assert verts.shape == (4, 2)
        half = 0.5
        expected = np.array(
            [
                [-half, 0.0],  # lower-left
                [-half, 2.0],  # upper-left
                [half, 2.0],  # upper-right
                [half, 0.0],  # lower-right
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(verts, expected, atol=1e-5)

    def test_quad_indices_form_two_triangles(self):
        gplt.bar([0.0], [1.0])
        indices = self._bar_layer().indices
        np.testing.assert_array_equal(indices, [0, 1, 2, 0, 2, 3])

    def test_multiple_bars_create_separate_patches(self):
        gplt.bar([0, 1, 2], [1, 2, 3])
        patch_layers = [l for l in gplt.gcf().scene.layers if l.layer_type == "patch"]
        assert len(patch_layers) == 3
        for p in patch_layers:
            assert p.vertices.shape == (4, 2)

    def test_bar_with_bottom_offset(self):
        gplt.bar([0.0], [3.0], bottom=1.0)
        verts = self._bar_layer().vertices
        # y values should span [1.0, 4.0]
        assert np.isclose(verts[:, 1].min(), 1.0)
        assert np.isclose(verts[:, 1].max(), 4.0)

    def test_bar_with_custom_width(self):
        gplt.bar([0.0], [1.0], width=0.4)
        verts = self._bar_layer().vertices
        assert np.isclose(verts[:, 0].min(), -0.2, atol=1e-5)
        assert np.isclose(verts[:, 0].max(), 0.2, atol=1e-5)


# ---------------------------------------------------------------------------
# Shader: u_window + gl_ClipDistance
# ---------------------------------------------------------------------------


class TestShaderClipDistanceWrites:
    """WIDE_SEGMENT_INSTANCED_VS must declare u_window and unconditionally write
    all 4 gl_ClipDistance values.  Omitting these writes on macOS GL drivers causes
    undefined (often negative) clip distances → random vertex culling → dotted lines."""

    def test_u_window_uniform_declared(self):
        assert (
            "uniform vec4  u_window;" in WIDE_SEGMENT_INSTANCED_VS
        ), "u_window must be declared as a vec4 uniform"

    def test_all_four_clip_distances_written(self):
        for i in range(4):
            assert (
                f"gl_ClipDistance[{i}]" in WIDE_SEGMENT_INSTANCED_VS
            ), f"gl_ClipDistance[{i}] must be written in the vertex shader"

    def test_clip_distances_use_u_window_components(self):
        # Each clip distance should reference u_window (not a hardcoded value)
        lines_with_clip = [
            line
            for line in WIDE_SEGMENT_INSTANCED_VS.splitlines()
            if "gl_ClipDistance[" in line and "=" in line
        ]
        assert len(lines_with_clip) >= 4
        for line in lines_with_clip:
            assert (
                "u_window" in line or "-1.0" in line
            ), f"Clip distance assignment should use u_window or be a NaN-guard: {line}"


# ---------------------------------------------------------------------------
# PolylineLayer NaN segment passthrough
# ---------------------------------------------------------------------------


class TestPolylineNanSegment:
    """PolylineLayer.pts with NaN rows (from batched quiver shafts) must not
    crash when processed by update_gpu_data — NaN segments produce degenerate
    quads that the GPU discards silently."""

    def test_polyline_layer_stores_nan_pts(self):
        # Verify that add_line_strip accepts NaN-containing arrays without error
        plot = gplt.gcf()
        nan_pts_x = np.array([0.0, 1.0, np.nan, 2.0, 3.0], dtype=np.float32)
        nan_pts_y = np.array([0.0, 0.0, np.nan, 1.0, 1.0], dtype=np.float32)
        plot.add_line_strip(nan_pts_x, nan_pts_y, color=(1, 0, 0, 1), width=1)
        layer = plot.scene.layers[-1]
        assert layer.layer_type == "polyline"
        assert np.any(np.isnan(layer.pts))

    def test_update_gpu_data_does_not_crash_on_nan(self):
        import glplot.renderers.polyline as polyline_mod
        from glplot.core.layers import PolylineLayer
        from glplot.options import EngineOptions
        from glplot.renderers.polyline import GLWideSegmentBuffers, PolylineRenderer

        # Always replace GL calls with no-ops using create=True so that:
        # (a) when OpenGL.GL is mocked globally the symbols don't exist yet, and
        # (b) when OpenGL.GL is real they are replaced so no GL context is needed.
        # patch.object restores originals (or removes injected attrs) on exit.
        with (
            patch.object(polyline_mod, "glBindBuffer", MagicMock(), create=True),
            patch.object(polyline_mod, "glBufferData", MagicMock(), create=True),
            patch.object(polyline_mod, "glBufferSubData", MagicMock(), create=True),
            patch.object(polyline_mod, "GL_ARRAY_BUFFER", 0x8892, create=True),
            patch.object(polyline_mod, "GL_STREAM_DRAW", 0x88E0, create=True),
        ):

            renderer = PolylineRenderer(EngineOptions())
            layer = PolylineLayer(
                pts=np.array([[0, 0], [1, 0], [np.nan, np.nan], [2, 0], [3, 0]], dtype=np.float32),
                color=(1, 0, 0, 1),
                width=1.0,
            )
            bufs = GLWideSegmentBuffers()
            renderer.update_gpu_data(layer, bufs)

        # 5 pts → 4 segments; NaN rows are treated as degenerate segments
        assert bufs.instance_count == 4
        assert layer.dirty.gpu_dirty is False
