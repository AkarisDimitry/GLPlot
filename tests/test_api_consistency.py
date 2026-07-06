"""Test API consistency and compatibility with Matplotlib conventions."""

import numpy as np
import pytest

import glplot.pyplot as gplt


@pytest.fixture(autouse=True)
def clean_state():
    """Ensure clean pyplot state before and after each test."""
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


class TestFigureCreation:
    """Test figure creation and configuration."""

    def test_figure_default_size(self):
        """Test default figure dimensions."""
        fig = gplt.figure()
        assert fig.width > 0
        assert fig.height > 0

    def test_figure_with_figsize(self):
        """Test figure sizing with figsize parameter."""
        fig = gplt.figure(figsize=(10, 8))
        assert fig.width == 1000
        assert fig.height == 800

    def test_figure_with_dpi(self):
        """Test figure sizing with custom DPI."""
        fig = gplt.figure(figsize=(5, 4), dpi=150)
        assert fig.width == 750
        assert fig.height == 600

    def test_figure_title_setting(self):
        """Test figure title parameter."""
        title = "Test Figure Title"
        fig = gplt.figure(title=title)
        assert fig.title == title

    def test_multiple_figures(self):
        """Test managing multiple figures."""
        fig1 = gplt.figure("fig1")
        fig2 = gplt.figure("fig2")
        assert gplt.gcf().title == "fig2"
        gplt.figure("fig1")
        assert gplt.gcf().title == "fig1"

    def test_figure_bounds(self):
        """Test figure bounds are reasonable."""
        fig = gplt.figure(figsize=(1, 1))
        assert fig.width >= 100
        assert fig.height >= 100


class TestAxisLabeling:
    """Test axis labeling API."""

    def test_xlabel_ylabel_zlabel(self):
        """Test axis label setting."""
        gplt.figure()
        gplt.plot([0, 1], [0, 1])
        gplt.xlabel("X Axis")
        gplt.ylabel("Y Axis")
        gplt.zlabel("Z Axis")

        hud = gplt.gcf()._hud_manager
        assert hud.axis.labels.get("xlabel") == "X Axis"
        assert hud.axis.labels.get("ylabel") == "Y Axis"
        assert hud.axis.labels.get("zlabel") == "Z Axis"

    def test_title_setting(self):
        """Test figure title via API."""
        gplt.figure()
        gplt.title("Main Title")
        assert gplt.gcf().title == "Main Title"

    def test_grid_toggle(self):
        """Test grid enable/disable."""
        gplt.figure()
        gplt.grid(True)
        assert gplt.gcf()._hud_manager.axis.grid_enabled is True
        gplt.grid(False)
        assert gplt.gcf()._hud_manager.axis.grid_enabled is False

    def test_axis_limits(self):
        """Test axis limit setting."""
        gplt.figure()
        gplt.plot([0, 10], [0, 10])
        gplt.xlim(0, 5)
        gplt.ylim(2, 8)

        assert gplt.gcf().xlim == (0, 5)
        assert gplt.gcf().ylim == (2, 8)


class TestLayerAPI:
    """Test layer creation and properties."""

    def test_plot_returns_artists(self):
        """Test that plot returns artist list."""
        artists = gplt.plot([0, 1, 2], [0, 1, 4])
        assert isinstance(artists, list)
        assert len(artists) > 0

    def test_plot_with_label(self):
        """Test plot with label parameter."""
        artists = gplt.plot([0, 1], [0, 1], label="Test Line")
        assert artists[0].label == "Test Line"

    def test_scatter_with_color_array(self):
        """Test scatter with color array."""
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 0])
        c = np.array([0, 0.5, 1])

        layer = gplt.scatter(x, y, c=c)
        assert layer.colors.shape[0] == len(x)

    def test_scatter_with_colormap(self):
        """Test scatter with colormap parameter."""
        layer = gplt.scatter([0, 1, 2], [0, 1, 0], c=[0, 0.5, 1], cmap="viridis")
        assert layer.metadata.get("cmap") == "viridis"

    def test_legend_creation(self):
        """Test legend functionality."""
        gplt.plot([0, 1], [0, 1], label="Line 1")
        gplt.plot([0, 1], [1, 0], label="Line 2")
        gplt.legend()

        fig = gplt.gcf()
        assert fig._hud_manager.legend.visible is True

    def test_empty_figure_bounds(self):
        """Test bounds calculation on empty figure."""
        gplt.figure()
        bounds = gplt.gcf().compute_bounds()
        assert bounds is not None


class TestMatplotlibSyntax:
    """Test Matplotlib-style syntax compatibility."""

    def test_color_string_formats(self):
        """Test various color string formats."""
        colors = ["r", "g", "b", "k", "w", "tab:blue", "#FF0000"]

        for color in colors:
            gplt.clf()
            gplt.figure()
            try:
                gplt.plot([0, 1], [0, 1], color=color)
            except Exception as e:
                pytest.fail(f"Failed to handle color {color}: {e}")

    def test_linestyle_formats(self):
        """Test various linestyle formats."""
        styles = ["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]

        for style in styles:
            gplt.clf()
            gplt.figure()
            try:
                gplt.plot([0, 1], [0, 1], linestyle=style)
            except Exception as e:
                pytest.fail(f"Failed to handle linestyle {style}: {e}")

    def test_marker_formats(self):
        """Test various marker formats."""
        markers = ["o", "s", "^", "v", "<", ">", "x", "+", "*", ".", ","]

        for marker in markers:
            gplt.clf()
            gplt.figure()
            try:
                gplt.scatter([0, 1], [0, 1], marker=marker)
            except Exception as e:
                pytest.fail(f"Failed to handle marker {marker}: {e}")

    def test_format_string_combinations(self):
        """Test format string parsing."""
        formats = ["r-", "b--", "g^", "ko-", "c--s", "m:+"]

        for fmt in formats:
            gplt.clf()
            gplt.figure()
            try:
                gplt.plot([0, 1], [0, 1], fmt)
            except Exception as e:
                pytest.fail(f"Failed to handle format {fmt}: {e}")


class TestDataValidation:
    """Test data validation and type handling."""

    def test_plot_accepts_lists(self):
        """Test plot with Python lists."""
        gplt.plot([0, 1, 2], [0, 1, 4])
        assert len(gplt.gcf().scene.layers) > 0

    def test_plot_accepts_numpy_arrays(self):
        """Test plot with NumPy arrays."""
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 4])
        gplt.plot(x, y)
        assert len(gplt.gcf().scene.layers) > 0

    def test_plot_accepts_tuples(self):
        """Test plot with tuple input."""
        gplt.plot((0, 1, 2), (0, 1, 4))
        assert len(gplt.gcf().scene.layers) > 0

    def test_plot_scalar_to_array(self):
        """Test plot converts scalars appropriately."""
        gplt.plot([0, 1, 2])
        assert len(gplt.gcf().scene.layers) > 0

    def test_scatter_size_parameter_types(self):
        """Test scatter with various size parameter types."""
        # Scalar size
        gplt.scatter([0, 1], [0, 1], s=10)

        gplt.clf()
        # Array of sizes
        gplt.scatter([0, 1], [0, 1], s=[5, 15])
        assert len(gplt.gcf().scene.layers) > 0

    def test_empty_data_handling(self):
        """Test handling of empty data arrays."""
        with pytest.raises((ValueError, IndexError)):
            gplt.plot([], [])

    def test_single_point(self):
        """Test plotting single point."""
        gplt.plot([0], [0])
        assert len(gplt.gcf().scene.layers) > 0


class TestClearAndReset:
    """Test clearing and resetting figure state."""

    def test_clf_clears_layers(self):
        """Test clf clears all layers."""
        gplt.figure()
        gplt.plot([0, 1], [0, 1])
        gplt.scatter([0, 1], [0, 1])

        assert len(gplt.gcf().scene.layers) > 0
        gplt.clf()
        assert len(gplt.gcf().scene.layers) == 0

    def test_cla_clears_current_axes(self):
        """Test cla clears current axes."""
        gplt.figure()
        gplt.plot([0, 1], [0, 1])
        gplt.cla()
        assert len(gplt.gcf().scene.layers) == 0

    def test_close_figure(self):
        """Test closing a figure."""
        fig = gplt.figure("test_close")
        gplt.plot([0, 1], [0, 1])
        gplt.close(fig)
        assert gplt.gcf() is not fig


class TestSubplots:
    """Test subplot functionality."""

    def test_subplots_creation(self):
        """Test creating subplots."""
        try:
            figs, axes = gplt.subplots(2, 2)
            assert figs is not None
        except NotImplementedError:
            pytest.skip("Subplots not yet implemented")

    def test_subplot_selection(self):
        """Test subplot selection."""
        try:
            gplt.figure()
            # Basic subplot test
            gplt.subplot(2, 2, 1)
        except NotImplementedError:
            pytest.skip("Subplots not yet implemented")
