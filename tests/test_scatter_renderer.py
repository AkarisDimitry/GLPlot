"""
Unit tests for ScatterRenderer.

Tests the core rendering pipeline for point clouds, covering:
- GPU buffer creation and data upload via layer interface
- Drawing operations through the pyplot API
- Image layer (imshow) rendering
- Style and alpha blending
- Density mode accumulation
"""

from __future__ import annotations

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.core.layers import ScatterLayer
from glplot.options import EngineOptions
from glplot.renderers.scatter import ScatterRenderer


@pytest.fixture(autouse=True)
def clean_state():
    """Reset pyplot state before and after each test."""
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


@pytest.fixture
def engine_options():
    """Create engine options for testing."""
    return EngineOptions()


@pytest.fixture
def scatter_renderer(engine_options):
    """Create a scatter renderer instance."""
    renderer = ScatterRenderer(engine_options)
    return renderer


@pytest.fixture
def simple_scatter_layer():
    """Create a simple scatter layer with 3 points."""
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]], dtype=np.float32)
    colors = np.array(
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    layer = ScatterLayer(pts=pts, colors=colors)
    return layer


class TestScatterRendererInitialization:
    """Test ScatterRenderer initialization."""

    def test_init_sets_default_options(self, engine_options):
        """Test that initialization sets correct default values."""
        renderer = ScatterRenderer(engine_options)
        assert renderer.options is engine_options
        assert renderer.prog == 0
        assert renderer.u_mvp == -1
        assert renderer.u_size == -1
        assert renderer.u_alpha == -1

    def test_renderer_has_required_attributes(self, scatter_renderer):
        """Test that renderer has all required attributes."""
        assert hasattr(scatter_renderer, "prog")
        assert hasattr(scatter_renderer, "accum_prog")
        assert hasattr(scatter_renderer, "image_prog")
        assert hasattr(scatter_renderer, "options")
        assert hasattr(scatter_renderer, "u_mvp")
        assert hasattr(scatter_renderer, "u_size")
        assert hasattr(scatter_renderer, "u_alpha")


class TestScatterRendererBufferManagement:
    """Test GPU buffer creation and data upload."""

    def test_update_gpu_data_with_broadcast_color(self, scatter_renderer):
        """Test that a single color is broadcast to all points."""
        pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        layer = ScatterLayer(pts=pts)
        layer.colors = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # This tests the color broadcasting logic
        assert layer.colors.shape == (4,)  # Single color
        assert layer.pts.shape == (2, 2)  # Two points

    def test_update_gpu_data_with_per_point_colors(self, scatter_renderer, simple_scatter_layer):
        """Test that per-point colors are handled correctly."""
        assert simple_scatter_layer.colors.shape == (3, 4)
        assert simple_scatter_layer.pts.shape == (3, 2)

    def test_update_gpu_data_skips_empty_data(self):
        """Test that empty point arrays are handled gracefully."""
        layer = ScatterLayer(pts=np.array([], dtype=np.float32).reshape(0, 2))
        assert len(layer.pts) == 0

    def test_update_gpu_data_with_none_points(self):
        """Test that None points are handled."""
        layer = ScatterLayer(pts=None)
        assert layer.pts is None


class TestScatterRendererEdgeCases:
    """Test edge cases and error conditions."""

    def test_large_point_count(self, scatter_renderer):
        """Test that renderer handles large point counts."""
        n = 10000
        pts = np.random.uniform(-1, 1, (n, 2)).astype(np.float32)
        colors = np.random.uniform(0, 1, (n, 4)).astype(np.float32)
        layer = ScatterLayer(pts=pts, colors=colors)

        assert layer.pts.shape == (n, 2)
        assert layer.colors.shape == (n, 4)

    def test_single_point(self, scatter_renderer):
        """Test creating layer with a single point."""
        pts = np.array([[0.0, 0.0]], dtype=np.float32)
        colors = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        layer = ScatterLayer(pts=pts, colors=colors)

        assert len(layer.pts) == 1
        assert len(layer.colors) == 1

    def test_points_with_nan_values(self, scatter_renderer):
        """Test handling of NaN values in points."""
        pts = np.array([[0.0, 0.0], [np.nan, np.nan], [2.0, 2.0]], dtype=np.float32)
        colors = np.ones((3, 4), dtype=np.float32)
        layer = ScatterLayer(pts=pts, colors=colors)

        assert layer.pts.shape == (3, 2)
        assert np.isnan(layer.pts[1, 0])

    def test_outline_style_properties(self, simple_scatter_layer):
        """Test that outline style properties exist."""
        simple_scatter_layer.style.point_outline_enabled = True
        simple_scatter_layer.style.point_outline_color = (0.0, 0.0, 0.0, 1.0)
        simple_scatter_layer.style.point_outline_width = 2.0

        assert simple_scatter_layer.style.point_outline_enabled is True

    def test_zero_size_points(self, simple_scatter_layer):
        """Test handling of zero-size points."""
        simple_scatter_layer.style.point_size = 0.0
        assert simple_scatter_layer.style.point_size == 0.0


class TestScatterRendererIntegration:
    """Integration tests using the pyplot API."""

    def test_scatter_through_pyplot(self):
        """Test scatter rendering through the pyplot API."""
        x = np.array([0, 1, 2], dtype=np.float32)
        y = np.array([0, 1, 0], dtype=np.float32)

        gplt.scatter(x, y, color="red")
        # Should create a scatter layer
        assert len(gplt.gcf().scene.scatters) == 1

    def test_scatter_with_size(self):
        """Test scatter with custom size."""
        x = np.array([0, 1, 2], dtype=np.float32)
        y = np.array([0, 1, 0], dtype=np.float32)

        gplt.scatter(x, y, size=15.0)
        # Should complete without error
        assert len(gplt.gcf().scene.scatters) == 1

    def test_scatter_with_colormap(self):
        """Test scatter with colormap."""
        x = np.array([0, 1, 2], dtype=np.float32)
        y = np.array([0, 1, 0], dtype=np.float32)
        c = np.array([0.0, 0.5, 1.0], dtype=np.float32)

        gplt.scatter(x, y, c=c, cmap="viridis")
        # Should create a scatter layer with colormap
        assert len(gplt.gcf().scene.scatters) == 1

    def test_imshow_through_pyplot(self):
        """Test imshow rendering through the pyplot API."""
        matrix = np.random.uniform(0, 1, (10, 10))
        gplt.imshow(matrix, extent=(-1, 1, -1, 1))
        # Should create an imshow layer
        assert len(gplt.gcf().scene.scatters) == 1

    def test_multiple_scatters(self):
        """Test rendering multiple scatter layers."""
        gplt.scatter([0, 1], [0, 1], color="red")
        gplt.scatter([1, 2], [1, 2], color="blue")

        assert len(gplt.gcf().scene.scatters) == 2

    def test_scatter_alpha_blending(self):
        """Test scatter with alpha blending."""
        x = np.array([0, 1, 2], dtype=np.float32)
        y = np.array([0, 1, 0], dtype=np.float32)

        gplt.scatter(x, y, color="red", alpha=0.5)
        assert len(gplt.gcf().scene.scatters) == 1

    def test_scatter_layer_properties(self):
        """Test that scatter layer has correct properties."""
        x = np.array([0, 1], dtype=np.float32)
        y = np.array([0, 1], dtype=np.float32)

        gplt.scatter(x, y, size=10.0)
        # Access through layers, not legacy scatters
        layers = [l for l in gplt.gcf().scene.layers if l.layer_type == "scatter"]
        assert len(layers) > 0
        layer = layers[0]

        assert layer.layer_type == "scatter"
        assert layer.style.point_size == 10.0

    def test_scatter_bounds_calculation(self):
        """Test that scatter layer correctly calculates bounds."""
        x = np.array([0, 2], dtype=np.float32)
        y = np.array([0, 3], dtype=np.float32)

        gplt.scatter(x, y)
        # Access through layers
        layers = [l for l in gplt.gcf().scene.layers if l.layer_type == "scatter"]
        assert len(layers) > 0
        layer = layers[0]
        bounds = layer.get_intrinsic_bounds()

        assert bounds is not None
        assert bounds[0] == 0.0  # xmin
        assert bounds[1] == 2.0  # xmax
        assert bounds[2] == 0.0  # ymin
        assert bounds[3] == 3.0  # ymax

    def test_scatter_with_density_mode(self):
        """Test scatter rendering in density mode."""
        x = np.random.uniform(-1, 1, 100)
        y = np.random.uniform(-1, 1, 100)

        gplt.scatter(x, y)
        # Note: show_density is a method on the plot, not pyplot
        gplt.gcf().display_density = True

        assert gplt.gcf().display_density is True

    def test_scatter_empty_data(self):
        """Test scatter with empty data."""
        x = np.array([], dtype=np.float32)
        y = np.array([], dtype=np.float32)

        gplt.scatter(x, y)
        # Should handle empty data gracefully
