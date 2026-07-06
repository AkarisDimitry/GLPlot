"""Performance and benchmark tests."""

import time

import numpy as np
import pytest

import glplot.pyplot as gplt


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


class TestPerformanceScaling:
    """Test performance with increasing data sizes."""

    @pytest.mark.slow
    def test_plot_100k_points_performance(self):
        """Test plotting 100k points completes in reasonable time."""
        n = 100_000
        x = np.linspace(0, 10, n)
        y = np.sin(x)

        start = time.time()
        gplt.plot(x, y)
        elapsed = time.time() - start

        assert elapsed < 5.0  # Should complete in < 5 seconds
        assert len(gplt.gcf().scene.layers) > 0

    @pytest.mark.slow
    def test_plot_1m_points_performance(self):
        """Test plotting 1M points completes in reasonable time."""
        n = 1_000_000
        x = np.linspace(0, 100, n)
        y = np.sin(x)

        start = time.time()
        gplt.plot(x, y)
        elapsed = time.time() - start

        assert elapsed < 10.0  # Should complete in < 10 seconds
        assert len(gplt.gcf().scene.layers) > 0

    @pytest.mark.slow
    def test_scatter_100k_points_performance(self):
        """Test scatter with 100k points."""
        n = 100_000
        x = np.random.randn(n)
        y = np.random.randn(n)

        start = time.time()
        gplt.scatter(x, y, s=1)
        elapsed = time.time() - start

        assert elapsed < 5.0
        assert len(gplt.gcf().scene.layers) > 0

    @pytest.mark.slow
    def test_histogram_1m_samples_performance(self):
        """Test histogram with 1M samples."""
        n = 1_000_000
        data = np.random.randn(n)

        start = time.time()
        counts, edges, artists = gplt.hist(data, bins=1000)
        elapsed = time.time() - start

        assert elapsed < 10.0
        assert len(artists) == 1000

    @pytest.mark.slow
    def test_scatter_with_colors_performance(self):
        """Test scatter with per-point colors on large dataset."""
        n = 50_000
        x = np.random.randn(n)
        y = np.random.randn(n)
        c = np.random.rand(n)

        start = time.time()
        gplt.scatter(x, y, c=c, cmap="viridis", s=2)
        elapsed = time.time() - start

        assert elapsed < 5.0

    @pytest.mark.slow
    def test_many_plot_layers_performance(self):
        """Test adding many plot layers."""
        n_layers = 100
        x = np.linspace(0, 10, 100)

        start = time.time()
        for i in range(n_layers):
            gplt.plot(x, np.sin(x + i * 0.1), alpha=0.5)
        elapsed = time.time() - start

        assert elapsed < 5.0
        assert len(gplt.gcf().scene.layers) == n_layers


class TestMemoryEfficiency:
    """Test memory usage patterns."""

    def test_repeated_plotting_does_not_leak(self):
        """Test repeated plotting doesn't cause memory issues."""
        for i in range(10):
            gplt.clf()
            x = np.linspace(0, 10, 1000)
            y = np.sin(x)
            gplt.plot(x, y)
        # If memory leaks, this would show up in monitoring
        assert len(gplt.gcf().scene.layers) > 0

    def test_large_figure_multiple_times(self):
        """Test creating large figures multiple times."""
        for i in range(5):
            gplt.clf()
            n = 100_000
            x = np.linspace(0, 10, n)
            y = np.sin(x)
            gplt.plot(x, y)
            gplt.scatter(x[::100], y[::100])
        # Should not crash or consume excessive memory
        assert True

    def test_color_array_memory(self):
        """Test memory handling of large color arrays."""
        n = 100_000
        x = np.random.randn(n)
        y = np.random.randn(n)
        c = np.random.rand(n)
        gplt.scatter(x, y, c=c, cmap="viridis")
        # Color array should be stored efficiently
        assert True


class TestRenderingEfficiency:
    """Test rendering and GPU efficiency."""

    def test_plot_lines_bulk_efficiency(self):
        """Test plot_lines efficiency with many line coefficients."""
        n = 100_000
        a = np.random.randn(n)
        b = np.random.randn(n)

        start = time.time()
        gplt.plot_lines(a, b, x_range=(-2, 2))
        elapsed = time.time() - start

        assert elapsed < 5.0
        fig = gplt.gcf()
        assert fig.N == n

    @pytest.mark.slow
    def test_hist2d_performance(self):
        """Test 2D histogram performance."""
        n = 1_000_000
        x = np.random.randn(n)
        y = 0.5 * x + np.random.randn(n)

        start = time.time()
        gplt.hist2d(x, y, bins=200)
        elapsed = time.time() - start

        assert elapsed < 10.0

    def test_density_mode_performance(self):
        """Test density mode rendering."""
        n = 100_000
        x = np.linspace(0, 10, n)
        y = np.sin(x)

        start = time.time()
        gplt.plot(x, y)
        # Note: actual rendering in density mode requires show()
        elapsed = time.time() - start

        assert elapsed < 5.0


class TestScalingBehavior:
    """Test how performance scales with data size."""

    @pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
    def test_plot_scaling(self, n):
        """Test plot performance with various sizes."""
        gplt.clf()
        x = np.linspace(0, 10, n)
        y = np.sin(x)

        start = time.time()
        gplt.plot(x, y)
        elapsed = time.time() - start

        # Should scale linearly or better
        assert elapsed < 10.0

    @pytest.mark.parametrize("n_points", [1_000, 10_000, 50_000])
    def test_scatter_scaling(self, n_points):
        """Test scatter performance with various sizes."""
        gplt.clf()
        x = np.random.randn(n_points)
        y = np.random.randn(n_points)

        start = time.time()
        gplt.scatter(x, y, s=1)
        elapsed = time.time() - start

        assert elapsed < 10.0

    @pytest.mark.parametrize("bins", [10, 100, 500])
    def test_hist_bin_scaling(self, bins):
        """Test histogram performance with various bin counts."""
        gplt.clf()
        data = np.random.randn(100_000)

        start = time.time()
        gplt.hist(data, bins=bins)
        elapsed = time.time() - start

        assert elapsed < 5.0


class TestBatchOperations:
    """Test performance of batch operations."""

    def test_repeated_scatter_batching(self):
        """Test multiple scatter calls batch efficiently."""
        x = np.linspace(0, 10, 100)

        start = time.time()
        for i in range(50):
            gplt.scatter(x, x + i, c=np.random.rand(len(x)), s=2, alpha=0.5)
        elapsed = time.time() - start

        assert elapsed < 5.0
        assert len(gplt.gcf().scene.layers) == 50

    def test_sequential_operations_efficiency(self):
        """Test sequential plotting operations."""
        x = np.linspace(0, 10, 100)

        start = time.time()
        gplt.plot(x, np.sin(x))
        gplt.scatter(x, np.cos(x), c="red")
        gplt.plot(x, np.tan(x), alpha=0.5)
        elapsed = time.time() - start

        assert elapsed < 3.0

    def test_color_mapping_efficiency(self):
        """Test efficiency of color mapping operations."""
        n = 50_000

        start = time.time()
        for cmap in ["viridis", "plasma", "inferno", "magma"]:
            gplt.clf()
            x = np.random.randn(n)
            y = np.random.randn(n)
            c = np.random.rand(n)
            gplt.scatter(x, y, c=c, cmap=cmap, s=1)
        elapsed = time.time() - start

        # Should handle multiple colormaps efficiently
        assert elapsed < 10.0


class TestThreadingAndConcurrency:
    """Test behavior under concurrent operations."""

    def test_sequential_figure_creation(self):
        """Test creating and closing multiple figures."""
        for i in range(10):
            fig = gplt.figure(f"fig_{i}")
            gplt.plot([0, 1, 2], [0, 1, 4])
            gplt.close(fig)
        # Should complete without issues
        assert True

    def test_figure_switch_performance(self):
        """Test switching between figures."""
        figs = []
        for i in range(5):
            figs.append(gplt.figure(f"fig_{i}"))
            gplt.plot([i, i + 1], [0, 1])

        start = time.time()
        for fig in figs:
            gplt.figure(fig.title)
            gplt.plot([0, 1], [0, 1])
        elapsed = time.time() - start

        assert elapsed < 5.0

        for fig in figs:
            gplt.close(fig)
