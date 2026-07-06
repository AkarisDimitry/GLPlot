"""Test edge cases and boundary conditions."""

import numpy as np
import pytest

import glplot.pyplot as gplt


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


class TestNumericalEdgeCases:
    """Test handling of extreme numerical values."""

    def test_very_large_values(self):
        """Test plotting very large numbers."""
        x = np.array([0, 1e10, 2e10])
        y = np.array([0, 1e10, 2e10])
        gplt.plot(x, y)
        assert len(gplt.gcf().scene.layers) > 0

    def test_very_small_values(self):
        """Test plotting very small numbers."""
        x = np.array([0, 1e-10, 2e-10])
        y = np.array([0, 1e-10, 2e-10])
        gplt.plot(x, y)
        assert len(gplt.gcf().scene.layers) > 0

    def test_mixed_magnitude_values(self):
        """Test data spanning many orders of magnitude."""
        x = np.array([1e-5, 1, 1e5])
        y = np.array([1e-5, 1, 1e5])
        gplt.plot(x, y)
        assert len(gplt.gcf().scene.layers) > 0

    def test_negative_values(self):
        """Test plotting negative values."""
        x = np.array([-10, -5, 0, 5, 10])
        y = np.array([-10, -5, 0, 5, 10])
        gplt.plot(x, y)
        assert len(gplt.gcf().scene.layers) > 0

    def test_mixed_positive_negative(self):
        """Test mixed positive and negative values."""
        x = np.linspace(-10, 10, 50)
        y = np.sin(x)
        gplt.plot(x, y)
        assert len(gplt.gcf().scene.layers) > 0

    def test_zero_axis_range(self):
        """Test data with no range (constant)."""
        x = np.zeros(5)
        y = np.ones(5)
        gplt.plot(x, y)
        # Should handle gracefully

    def test_nan_in_data(self):
        """Test data containing NaN values."""
        x = np.array([0, 1, np.nan, 3, 4])
        y = np.array([0, 1, 2, np.nan, 4])
        # Should not crash
        try:
            gplt.plot(x, y)
        except (ValueError, RuntimeError):
            pass  # Expected to fail or handle gracefully

    def test_inf_in_data(self):
        """Test data containing infinite values."""
        x = np.array([0, 1, np.inf, 3, 4])
        y = np.array([0, 1, 2, 3, np.inf])
        try:
            gplt.plot(x, y)
        except (ValueError, RuntimeError):
            pass  # Expected to fail or handle gracefully


class TestArrayShapesAndSizes:
    """Test various array shapes and sizes."""

    def test_single_point(self):
        """Test plotting single point."""
        gplt.plot([1], [1])
        assert len(gplt.gcf().scene.layers) > 0

    def test_two_points(self):
        """Test plotting two points."""
        gplt.plot([0, 1], [0, 1])
        assert len(gplt.gcf().scene.layers) > 0

    def test_large_dataset(self):
        """Test plotting large dataset."""
        n = 100_000
        x = np.linspace(0, 10, n)
        y = np.sin(x)
        gplt.plot(x, y)
        assert len(gplt.gcf().scene.layers) > 0

    def test_very_large_dataset(self):
        """Test plotting very large dataset (1M points)."""
        n = 1_000_000
        x = np.linspace(0, 100, n)
        y = np.sin(x)
        gplt.plot(x, y)
        assert len(gplt.gcf().scene.layers) > 0

    def test_scatter_dense_points(self):
        """Test scatter with many points."""
        n = 100_000
        x = np.random.randn(n)
        y = np.random.randn(n)
        gplt.scatter(x, y, s=1)
        assert len(gplt.gcf().scene.layers) > 0

    def test_histogram_many_bins(self):
        """Test histogram with many bins."""
        data = np.random.randn(10000)
        counts, edges, artists = gplt.hist(data, bins=500)
        assert len(artists) == 500

    def test_histogram_few_bins(self):
        """Test histogram with few bins."""
        data = np.random.randn(100)
        counts, edges, artists = gplt.hist(data, bins=2)
        assert len(artists) == 2


class TestColorHandling:
    """Test color parameter handling."""

    def test_named_colors(self):
        """Test various named color formats."""
        colors = ["red", "green", "blue", "black", "white", "cyan", "magenta", "yellow"]
        for i, color in enumerate(colors):
            gplt.scatter([i], [0], c=color)

    def test_hex_colors(self):
        """Test hex color codes."""
        colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFFFF", "#000000"]
        for i, color in enumerate(colors):
            gplt.scatter([i], [0], c=color)

    def test_rgb_tuples(self):
        """Test RGB tuple colors."""
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0.5)]
        for i, color in enumerate(colors):
            gplt.scatter([i], [0], c=color)

    def test_rgba_tuples(self):
        """Test RGBA tuple colors."""
        colors = [(1, 0, 0, 1), (0, 1, 0, 0.5), (0, 0, 1, 0.75)]
        for i, color in enumerate(colors):
            gplt.scatter([i], [0], c=color)

    def test_color_array_normalization(self):
        """Test numeric color array with automatic normalization."""
        gplt.scatter([0, 1, 2], [0, 1, 0], c=[0, 0.5, 1], cmap="viridis")
        assert gplt.gcf().scene.layers[-1].colors.shape[0] == 3

    def test_color_array_explicit_range(self):
        """Test numeric color array with explicit range."""
        layer = gplt.scatter([0, 1, 2], [0, 1, 0], c=[10, 50, 100], vmin=0, vmax=100)
        assert layer.colors.shape[0] == 3


class TestAlphaTransparency:
    """Test alpha/transparency parameter handling."""

    def test_plot_alpha_scalar(self):
        """Test plot with scalar alpha."""
        gplt.plot([0, 1], [0, 1], alpha=0.5)
        assert len(gplt.gcf().scene.layers) > 0

    def test_scatter_alpha_scalar(self):
        """Test scatter with scalar alpha."""
        gplt.scatter([0, 1], [0, 1], alpha=0.3)
        assert len(gplt.gcf().scene.layers) > 0

    def test_fill_alpha_scalar(self):
        """Test fill_between with alpha."""
        gplt.fill_between([0, 1, 2], [0, 1, 0], alpha=0.2)
        assert len(gplt.gcf().scene.layers) > 0

    def test_alpha_extreme_values(self):
        """Test extreme alpha values."""
        gplt.plot([0, 1], [0, 1], alpha=0)  # Fully transparent
        gplt.scatter([0, 1], [0, 1], alpha=1)  # Fully opaque
        assert len(gplt.gcf().scene.layers) > 0


class TestLabelsAndAnnotations:
    """Test label and annotation edge cases."""

    def test_empty_labels(self):
        """Test empty string labels."""
        gplt.plot([0, 1], [0, 1], label="")
        gplt.xlabel("")
        gplt.ylabel("")

    def test_very_long_labels(self):
        """Test very long label strings."""
        long_label = "X" * 1000
        gplt.xlabel(long_label)
        gplt.ylabel(long_label)

    def test_special_characters_in_labels(self):
        """Test special characters in labels."""
        labels = [
            "μ = 0.5",
            "α + β = γ",
            "σ²",
            "∂/∂x",
            "±√2",
            "∑ x_i"
        ]
        for label in labels:
            try:
                gplt.xlabel(label)
            except Exception:
                pass  # May not support all Unicode

    def test_multilevel_annotation(self):
        """Test multiple annotations."""
        gplt.plot([0, 1, 2], [0, 1, 0])
        gplt.annotate("Start", xy=(0, 0), xytext=(0, 0.5))
        gplt.annotate("Peak", xy=(1, 1), xytext=(1, 1.5))
        gplt.annotate("End", xy=(2, 0), xytext=(2, 0.5))
        assert len(gplt.gcf().scene.layers) > 0


class TestLayerInteraction:
    """Test interactions between multiple layers."""

    def test_many_layers(self):
        """Test figure with many layers."""
        for i in range(100):
            gplt.plot([i, i+1], [i, i+1], alpha=0.1)
        assert len(gplt.gcf().scene.layers) == 100

    def test_mixed_layer_types(self):
        """Test mixed layer types in one figure."""
        gplt.plot([0, 1, 2], [0, 1, 0], "r-")
        gplt.scatter([0, 1, 2], [0, 1, 0], c="b")
        gplt.bar([0, 1], [1, 2], color="g")
        gplt.fill_between([0, 1, 2], [0, 0.5, 0], color="y", alpha=0.3)
        assert len(gplt.gcf().scene.layers) > 4

    def test_overlapping_layers(self):
        """Test layers with overlapping data."""
        x = np.linspace(0, 10, 100)
        for i in range(5):
            gplt.plot(x, np.sin(x + i), alpha=0.5)
        assert len(gplt.gcf().scene.layers) >= 5

    def test_layer_zorder_independence(self):
        """Test that layers render with implicit z-ordering."""
        # Layers should render in order added
        gplt.scatter([5], [5], c="red", s=100)
        gplt.scatter([5], [5], c="blue", s=50)
        assert len(gplt.gcf().scene.layers) == 2


class TestBoundsCalculation:
    """Test automatic bounds calculation."""

    def test_bounds_single_point(self):
        """Test bounds with single point."""
        gplt.plot([1], [2])
        bounds = gplt.gcf().compute_bounds()
        assert bounds is not None

    def test_bounds_no_data(self):
        """Test bounds on empty figure."""
        gplt.figure()
        try:
            bounds = gplt.gcf().compute_bounds()
        except Exception:
            pass  # May raise on empty figure

    def test_bounds_include_3d(self):
        """Test bounds calculation with 3D data."""
        gplt.scatter3d([0, 1], [0, 1], [0, 1])
        bounds = gplt.gcf().compute_bounds()
        assert bounds is not None

    def test_bounds_with_3d_layers(self):
        """Test bounds with mixed 2D and 3D layers."""
        gplt.plot([0, 1], [0, 1])
        gplt.scatter3d([0, 1], [0, 1], [0, 1])
        bounds = gplt.gcf().compute_bounds()
        assert bounds is not None
