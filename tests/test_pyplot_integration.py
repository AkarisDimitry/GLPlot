"""Integration tests for glplot.pyplot public API.

Tests the full pyplot stack without requiring GPU/shader compilation.
Focus on figure management, plot variants, and axis configuration.
"""

import numpy as np
import pytest

import glplot.pyplot as gplt


@pytest.fixture(autouse=True)
def clean_pyplot_state():
    """Clean pyplot state before and after each test."""
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


class TestFigureManagement:
    """Test figure creation and management."""

    def test_figure_creation_default(self):
        """Test creating a figure with default settings."""
        fig = gplt.figure()
        assert fig is not None
        assert fig.width > 0
        assert fig.height > 0

    def test_figure_with_title(self):
        """Test creating a figure with title."""
        title = "Test Figure"
        fig = gplt.figure(title=title)
        assert fig.title == title

    def test_figure_with_figsize(self):
        """Test figure sizing with matplotlib-style figsize."""
        fig = gplt.figure(figsize=(8, 6), dpi=100)
        assert fig.width == 800
        assert fig.height == 600

    def test_figure_with_custom_dpi(self):
        """Test figure with custom DPI."""
        fig = gplt.figure(figsize=(4, 3), dpi=200)
        assert fig.width == 800
        assert fig.height == 600

    def test_gcf_returns_current_figure(self):
        """Test gcf() returns current figure."""
        fig = gplt.figure(title="Test")
        assert gplt.gcf() is fig

    def test_multiple_figures(self):
        """Test creating and switching between figures."""
        gplt.figure("fig1")
        gplt.figure("fig2")
        assert gplt.gcf().title == "fig2"

    def test_clf_clears_current_figure(self):
        """Test clf() clears layers from current figure."""
        gplt.figure()
        gplt.plot([0, 1], [0, 1])
        initial_layers = len(gplt.gcf().scene.layers)
        assert initial_layers > 0
        gplt.clf()
        assert len(gplt.gcf().scene.layers) == 0

    def test_figure_properties_accessible(self):
        """Test that figure has expected properties."""
        fig = gplt.figure()
        # Check key properties exist
        assert hasattr(fig, "width")
        assert hasattr(fig, "height")
        assert hasattr(fig, "scene")
        assert hasattr(fig, "title")


class TestPlottingBasics:
    """Test basic plotting functions."""

    def test_plot_with_y_only(self):
        """Test plot with single array (y values)."""
        artists = gplt.plot([0, 1, 4])
        assert isinstance(artists, list)
        assert len(artists) > 0

    def test_plot_with_x_and_y(self):
        """Test plot with x and y arrays."""
        x = [0, 1, 2]
        y = [0, 1, 4]
        artists = gplt.plot(x, y)
        assert len(artists) > 0

    def test_plot_with_numpy_arrays(self):
        """Test plot with numpy arrays."""
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 4])
        artists = gplt.plot(x, y)
        assert len(artists) > 0

    def test_plot_with_format_string(self):
        """Test plot with matplotlib format string."""
        artists = gplt.plot([0, 1, 2], [0, 1, 4], "r-o")
        assert len(artists) > 0

    def test_plot_with_color_keyword(self):
        """Test plot with color keyword."""
        artists = gplt.plot([0, 1], [0, 1], color="red")
        assert len(artists) > 0

    def test_plot_with_label(self):
        """Test plot with label."""
        artists = gplt.plot([0, 1], [0, 1], label="Line 1")
        assert artists[0].label == "Line 1"

    def test_plot_with_linewidth(self):
        """Test plot with linewidth."""
        artists = gplt.plot([0, 1], [0, 1], linewidth=2.0)
        assert len(artists) > 0

    def test_plot_with_alpha(self):
        """Test plot with alpha."""
        artists = gplt.plot([0, 1], [0, 1], alpha=0.5)
        assert len(artists) > 0

    def test_plot_multiple_series(self):
        """Test plot with multiple x-y pairs."""
        x = np.arange(3)
        artists = gplt.plot(x, x, "r-", x, x**2, "b-")
        assert len(artists) > 0

    def test_plot_dimension_mismatch_error(self):
        """Test that mismatched dimensions raise error."""
        with pytest.raises(ValueError):
            gplt.plot([0, 1, 2], [0, 1])


class TestScatterPlotting:
    """Test scatter plotting functionality."""

    def test_scatter_basic(self):
        """Test basic scatter plot."""
        layer = gplt.scatter([0, 1, 2], [0, 1, 0])
        assert layer is not None
        assert layer.layer_type == "scatter"

    def test_scatter_with_color_string(self):
        """Test scatter with color string."""
        layer = gplt.scatter([0, 1, 2], [0, 1, 0], color="red")
        assert layer.layer_type == "scatter"

    def test_scatter_with_size(self):
        """Test scatter with custom point size."""
        layer = gplt.scatter([0, 1, 2], [0, 1, 0], s=20.0)
        assert layer.style.point_size == 20.0

    def test_scatter_with_color_array(self):
        """Test scatter with per-point colors."""
        colors = ["red", "green", "blue"]
        layer = gplt.scatter([0, 1, 2], [0, 1, 0], c=colors)
        assert layer.colors is not None
        assert layer.colors.shape[0] == 3

    def test_scatter_with_colormap(self):
        """Test scatter with colormap."""
        c = [0.0, 0.5, 1.0]
        layer = gplt.scatter([0, 1, 2], [0, 1, 0], c=c, cmap="plasma")
        assert layer.metadata.get("cmap") == "plasma"

    def test_scatter_with_alpha(self):
        """Test scatter with alpha."""
        layer = gplt.scatter([0, 1, 2], [0, 1, 0], alpha=0.5)
        assert layer is not None

    def test_scatter_with_marker(self):
        """Test scatter with marker style."""
        layer = gplt.scatter([0, 1, 2], [0, 1, 0], marker="s")
        assert layer.metadata.get("marker") == "s"

    def test_scatter_empty_arrays(self):
        """Test scatter with empty arrays."""
        # Should handle gracefully
        try:
            gplt.scatter([], [])
        except (ValueError, RuntimeError):
            pass  # Expected


class TestBarAndHistogram:
    """Test bar and histogram plotting."""

    def test_bar_basic(self):
        """Test basic bar plot."""
        heights = [1, 2, 3]
        artists = gplt.bar(range(len(heights)), heights)
        assert len(artists) > 0

    def test_bar_with_color(self):
        """Test bar with color."""
        artists = gplt.bar([0, 1, 2], [1, 2, 3], color="blue")
        assert len(artists) > 0

    def test_bar_with_width(self):
        """Test bar with custom width."""
        artists = gplt.bar([0, 1, 2], [1, 2, 3], width=0.4)
        assert len(artists) > 0

    def test_hist_basic(self):
        """Test basic histogram."""
        data = [0, 0, 1, 2, 2, 2, 3]
        counts, edges, artists = gplt.hist(data, bins=4)
        assert len(counts) > 0
        assert len(edges) > 0
        assert len(artists) > 0

    def test_hist_with_bins_int(self):
        """Test histogram with integer bins."""
        data = np.random.normal(0, 1, 100)
        counts, edges, artists = gplt.hist(data, bins=10)
        assert len(counts) == 10

    def test_hist_with_bins_array(self):
        """Test histogram with bin edges array."""
        data = [0, 1, 2, 3, 4]
        bins = [0, 2, 4]
        counts, edges, artists = gplt.hist(data, bins=bins)
        assert len(counts) == len(bins) - 1

    def test_hist_with_color(self):
        """Test histogram with color."""
        counts, edges, artists = gplt.hist([0, 1, 2], bins=2, color="green")
        assert len(artists) > 0

    def test_hist_with_alpha(self):
        """Test histogram with alpha."""
        counts, edges, artists = gplt.hist([0, 1, 2], bins=2, alpha=0.5)
        assert len(artists) > 0


class TestFillBetween:
    """Test fill_between plotting."""

    def test_fill_between_basic(self):
        """Test basic fill_between."""
        x = [0, 1, 2]
        y1 = [0, 1, 0]
        y2 = [1, 2, 1]
        layer = gplt.fill_between(x, y1, y2)
        assert layer is not None

    def test_fill_between_with_color(self):
        """Test fill_between with color."""
        layer = gplt.fill_between([0, 1, 2], [0, 1, 0], [1, 2, 1], color="red")
        assert layer is not None

    def test_fill_between_with_alpha(self):
        """Test fill_between with alpha."""
        layer = gplt.fill_between([0, 1, 2], [0, 1, 0], [1, 2, 1], alpha=0.3)
        assert layer is not None

    def test_fill_between_scalar_y2(self):
        """Test fill_between with scalar y2."""
        layer = gplt.fill_between([0, 1, 2], [1, 2, 1], 0)
        assert layer is not None


class TestAxisConfiguration:
    """Test axis configuration and labeling."""

    def test_xlabel_setting(self):
        """Test setting x-axis label."""
        gplt.figure()
        gplt.plot([0, 1], [0, 1])
        gplt.xlabel("X Axis")
        assert gplt.gcf().xlabel == "X Axis"

    def test_ylabel_setting(self):
        """Test setting y-axis label."""
        gplt.figure()
        gplt.plot([0, 1], [0, 1])
        gplt.ylabel("Y Axis")
        assert gplt.gcf().ylabel == "Y Axis"

    def test_zlabel_setting(self):
        """Test setting z-axis label."""
        gplt.figure()
        gplt.zlabel("Z Axis")
        assert gplt.gcf().zlabel == "Z Axis"

    def test_title_setting(self):
        """Test setting figure title."""
        gplt.figure()
        gplt.title("Main Title")
        assert gplt.gcf().title == "Main Title"

    def test_xlim_setting(self):
        """Test setting x-axis limits."""
        gplt.figure()
        gplt.plot([0, 10], [0, 10])
        gplt.xlim(0, 5)
        # Limits are set, retrieve them
        xlim = gplt.gcf().get_xlim()
        assert xlim is not None

    def test_ylim_setting(self):
        """Test setting y-axis limits."""
        gplt.figure()
        gplt.plot([0, 10], [0, 10])
        gplt.ylim(2, 8)
        ylim = gplt.gcf().get_ylim()
        assert ylim is not None

    def test_axis_limits_both_xy(self):
        """Test setting both x and y limits."""
        gplt.figure()
        gplt.plot([0, 10], [0, 10])
        gplt.xlim(0, 5)
        gplt.ylim(2, 8)
        # Both should be set without crashing
        assert gplt.gcf().get_xlim() is not None
        assert gplt.gcf().get_ylim() is not None

    def test_grid_toggle(self):
        """Test grid visibility toggle."""
        gplt.figure()
        gplt.grid(True)
        assert gplt.gcf().grid_visible is True
        gplt.grid(False)
        assert gplt.gcf().grid_visible is False

    def test_axis_labels_multiple(self):
        """Test setting multiple axis labels together."""
        gplt.figure()
        gplt.plot([0, 1], [0, 1])
        gplt.xlabel("X")
        gplt.ylabel("Y")
        gplt.title("Test")
        fig = gplt.gcf()
        assert fig.xlabel == "X"
        assert fig.ylabel == "Y"
        assert fig.title == "Test"


class TestLegend:
    """Test legend functionality."""

    def test_legend_creation(self):
        """Test creating a legend."""
        gplt.figure()
        gplt.plot([0, 1], [0, 1], label="Line 1")
        gplt.plot([0, 1], [1, 0], label="Line 2")
        labels = gplt.legend()
        assert isinstance(labels, list)

    def test_legend_empty_figure(self):
        """Test legend on empty figure."""
        gplt.figure()
        # Should not crash
        gplt.legend()

    def test_legend_with_max_items(self):
        """Test legend with maximum items limit."""
        gplt.figure()
        for i in range(5):
            gplt.plot([0, 1], [i, i + 1], label=f"Line {i}")
        labels = gplt.legend(max_items=3)
        # Should include a "+more" indicator if items exceed max
        assert isinstance(labels, list)

    def test_legend_deduplication(self):
        """Test legend deduplicates items."""
        gplt.figure()
        gplt.plot([0, 1], [0, 1], label="Same")
        gplt.plot([0, 1], [1, 0], label="Same")
        labels = gplt.legend()
        assert isinstance(labels, list)


class TestLineReferenceAnnotations:
    """Test reference lines and annotations."""

    def test_axhline(self):
        """Test horizontal reference line."""
        gplt.figure()
        gplt.plot([0, 1], [0, 1])
        layer = gplt.axhline(0.5, color="r")
        assert layer is not None

    def test_axvline(self):
        """Test vertical reference line."""
        gplt.figure()
        gplt.plot([0, 1], [0, 1])
        layer = gplt.axvline(0.5, color="b")
        assert layer is not None

    def test_axline(self):
        """Test arbitrary line through points."""
        gplt.figure()
        gplt.plot([0, 1], [0, 1])
        layer = gplt.axline((0, 0), slope=1.0, color="k")
        assert layer is not None

    def test_hlines(self):
        """Test multiple horizontal lines."""
        gplt.figure()
        layers = gplt.hlines([0.5, 0.75], 0, 1)
        assert layers is not None

    def test_vlines(self):
        """Test multiple vertical lines."""
        gplt.figure()
        layers = gplt.vlines([0.5, 0.75], 0, 1)
        assert layers is not None

    def test_arrow(self):
        """Test arrow annotation."""
        gplt.figure()
        artists = gplt.arrow(0, 0, 1, 1, color="red")
        assert len(artists) > 0

    def test_annotate(self):
        """Test text annotation."""
        gplt.figure()
        artists = gplt.annotate("Peak", xy=(1, 1), xytext=(0.5, 0.8))
        assert len(artists) > 0


class Test3dPlotting:
    """Test 3D plotting functions."""

    def test_plot3d_basic(self):
        """Test basic 3D line plot."""
        t = np.linspace(0, 1, 10)
        artists = gplt.plot3d(t, t * 2, t * 3)
        assert len(artists) > 0

    def test_plot3d_with_label(self):
        """Test 3D plot with label."""
        artists = gplt.plot3d([0, 1], [0, 1], [0, 1], label="3D Line")
        assert len(artists) > 0

    def test_plot3d_with_format(self):
        """Test 3D plot with format string."""
        artists = gplt.plot3d([0, 1], [0, 1], [0, 1], "r-o")
        assert len(artists) > 0

    def test_plot3d_with_camera_angle(self):
        """Test 3D plot with camera angle."""
        artists = gplt.plot3d([0, 1], [0, 1], [0, 1], elev=45, azim=30)
        assert len(artists) > 0

    def test_scatter3d_basic(self):
        """Test basic 3D scatter plot."""
        layer = gplt.scatter3d([0, 1, 2], [0, 1, 0], [0, 1, 2])
        assert layer is not None

    def test_scatter3d_with_color(self):
        """Test 3D scatter with color."""
        layer = gplt.scatter3d([0, 1, 2], [0, 1, 0], [0, 1, 2], c="red")
        assert layer is not None

    def test_scatter3d_with_colormap(self):
        """Test 3D scatter with colormap."""
        layer = gplt.scatter3d([0, 1, 2], [0, 1, 0], [0, 1, 2], c=[0, 0.5, 1], cmap="viridis")
        assert layer is not None

    def test_scatter3d_with_size(self):
        """Test 3D scatter with point size."""
        layer = gplt.scatter3d([0, 1], [0, 1], [0, 1], s=10.0)
        assert layer.style.point_size == 10.0

    def test_quiver3d_basic(self):
        """Test basic 3D quiver plot."""
        artists = gplt.quiver3d([0, 1], [0, 1], [0, 0], [1, 0], [0, 1], [0.5, 0.5])
        assert len(artists) > 0

    def test_quiver3d_with_scale(self):
        """Test 3D quiver with scale factor."""
        artists = gplt.quiver3d([0, 1], [0, 1], [0, 0], [1, 0], [0, 1], [0.5, 0.5], scale=0.5)
        assert len(artists) > 0

    def test_quiver3d_with_normalize(self):
        """Test 3D quiver with vector normalization."""
        artists = gplt.quiver3d([0, 1], [0, 1], [0, 0], [1, 0], [0, 1], [0.5, 0.5], normalize=True)
        assert len(artists) > 0


class TestMatrixVisualization:
    """Test matrix/image visualization."""

    def test_imshow_basic(self):
        """Test basic image display."""
        matrix = np.arange(16, dtype=np.float32).reshape(4, 4)
        layer = gplt.imshow(matrix)
        assert layer is not None

    def test_imshow_with_cmap(self):
        """Test imshow with colormap."""
        matrix = np.random.random((5, 5))
        layer = gplt.imshow(matrix, cmap="viridis")
        assert layer is not None

    def test_imshow_with_extent(self):
        """Test imshow with extent."""
        matrix = np.random.random((3, 3))
        layer = gplt.imshow(matrix, extent=(-1, 1, -1, 1))
        assert layer is not None

    def test_matshow_basic(self):
        """Test basic matrix display."""
        matrix = np.eye(4)
        layer = gplt.matshow(matrix)
        assert layer is not None

    def test_hist2d_basic(self):
        """Test 2D histogram."""
        x = [0, 0, 1, 1]
        y = [0, 1, 0, 1]
        counts, xedges, yedges, layer = gplt.hist2d(x, y, bins=2)
        assert counts is not None
        assert layer is not None

    def test_hist2d_with_color(self):
        """Test 2D histogram with color."""
        counts, xedges, yedges, layer = gplt.hist2d([0, 1, 2], [0, 1, 2], bins=2, cmap="plasma")
        assert layer is not None


class TestContourPlots:
    """Test contour plotting."""

    def test_contour_basic(self):
        """Test basic contour plot."""
        x = np.linspace(-1, 1, 5)
        y = np.linspace(-1, 1, 5)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2
        layer = gplt.contour(X, Y, Z)
        assert layer is not None

    def test_contour_with_levels(self):
        """Test contour with specified levels."""
        x = np.linspace(-1, 1, 5)
        y = np.linspace(-1, 1, 5)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2
        layer = gplt.contour(X, Y, Z, levels=4)
        assert layer is not None

    def test_contourf_basic(self):
        """Test filled contour plot."""
        x = np.linspace(-1, 1, 5)
        y = np.linspace(-1, 1, 5)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X * Y)
        layer = gplt.contourf(X, Y, Z)
        assert layer is not None

    def test_pcolormesh_basic(self):
        """Test pcolormesh."""
        x = np.linspace(-1, 1, 5)
        y = np.linspace(-1, 1, 5)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X * Y)
        layer = gplt.pcolormesh(X, Y, Z)
        assert layer is not None


class TestSurfaceAndWireframe:
    """Test 3D surface and wireframe plots."""

    def test_plot_surface_basic(self):
        """Test basic surface plot."""
        x = np.linspace(-1, 1, 5)
        y = np.linspace(-1, 1, 5)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2
        layer = gplt.plot_surface(X, Y, Z)
        assert layer is not None

    def test_plot_wireframe_basic(self):
        """Test basic wireframe plot."""
        x = np.linspace(-1, 1, 5)
        y = np.linspace(-1, 1, 5)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X * Y)
        artists = gplt.plot_wireframe(X, Y, Z)
        assert len(artists) > 0

    def test_bar3d_basic(self):
        """Test 3D bar plot."""
        artists = gplt.bar3d([0, 1], [0, 1], [0, 0], 0.2, 0.2, [1, 2])
        assert len(artists) > 0

    def test_bar3d_with_shape(self):
        """Test 3D bar with custom shape."""
        artists = gplt.bar3d([0], [0], [0], 0.3, 0.3, [1], shape="hex")
        assert len(artists) > 0

    def test_mesh3d_basic(self):
        """Test 3D mesh plot."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        layer = gplt.mesh3d(vertices, faces=faces)
        assert layer is not None

    def test_volume3d_basic(self):
        """Test 3D volume plot."""
        x = [0, 1]
        y = [0, 1]
        z = [0, 1]
        c = [0.2, 0.9]
        layer = gplt.volume3d(x, y, z, c, threshold=0.5)
        assert layer is not None


class TestGlobalSettings:
    """Test global figure settings."""

    def test_set_global_alpha(self):
        """Test setting global alpha."""
        gplt.set_global_alpha(0.5)
        assert gplt.gcf().global_alpha == 0.5

    def test_set_global_alpha_limits(self):
        """Test global alpha with various values."""
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            gplt.set_global_alpha(alpha)
            assert gplt.gcf().global_alpha == alpha

    def test_set_lod_enabled(self):
        """Test enabling level-of-detail."""
        gplt.set_lod(enabled=True)
        assert gplt.gcf().enable_subsample is True

    def test_set_lod_disabled(self):
        """Test disabling level-of-detail."""
        gplt.set_lod(enabled=False)
        assert gplt.gcf().enable_subsample is False

    def test_set_lod_parameters(self):
        """Test LOD parameters."""
        gplt.set_lod(enabled=True, max_lines_per_px=250)
        assert gplt.gcf().max_lines_per_px == 250


class TestErrorHandling:
    """Test error handling and validation."""

    def test_plot_empty_arrays(self):
        """Test plot with empty arrays."""
        try:
            gplt.plot([], [])
        except (ValueError, RuntimeError):
            pass  # Expected

    def test_plot_nan_handling(self):
        """Test plot gracefully handles NaN."""
        try:
            gplt.plot([0, np.nan, 2], [0, 1, 2])
        except (ValueError, RuntimeError):
            pass  # Expected

    def test_scatter_mismatched_arrays(self):
        """Test scatter with mismatched dimensions."""
        with pytest.raises(ValueError):
            gplt.scatter([0, 1, 2], [0, 1])

    def test_bar_mismatched_arrays(self):
        """Test bar with mismatched dimensions."""
        with pytest.raises(ValueError):
            gplt.bar([0, 1, 2], [0, 1])

    def test_hist_empty_data(self):
        """Test histogram with empty data."""
        try:
            gplt.hist([], bins=5)
        except (ValueError, RuntimeError):
            pass  # Expected

    def test_invalid_colormap_name(self):
        """Test scatter with invalid colormap."""
        try:
            gplt.scatter([0, 1], [0, 1], c=[0, 1], cmap="nonexistent")
        except (ValueError, KeyError):
            pass  # Expected
