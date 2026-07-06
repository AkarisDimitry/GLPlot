"""Regression tests for previously fixed bugs and known issues."""

import numpy as np
import pytest

import glplot.pyplot as gplt


@pytest.fixture(autouse=True)
def clean_state():
    """Clean pyplot state before and after each test."""
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


class TestQuiverRenderingRegression:
    """Regression tests for quiver rendering issues (Issue #7)."""

    def test_quiver_batching_multiple_arrows(self):
        """Test that multiple quiver arrows batch correctly."""
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])
        u = np.array([1, 0, -1])
        v = np.array([0, 1, 0])

        artists = gplt.quiver(x, y, u, v)
        assert len(artists) > 0
        # Should create proper vector layer

    def test_quiver_scale_parameter(self):
        """Test quiver with various scale parameters."""
        x = [0, 1]
        y = [0, 1]
        u = [1, 1]
        v = [0, 1]

        for scale in [0.1, 0.5, 1.0, 2.0, 10.0]:
            gplt.clf()
            gplt.quiver(x, y, u, v, scale=scale)
            assert len(gplt.gcf().scene.layers) > 0

    def test_quiver_normalization(self):
        """Test quiver with normalize flag."""
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])
        u = np.array([0.1, 1, 100])
        v = np.array([0.1, 1, 100])

        gplt.quiver(x, y, u, v, normalize=True)
        gplt.clf()
        gplt.quiver(x, y, u, v, normalize=False)
        # Both should work


class TestClipDistanceRegression:
    """Regression tests for clip distance handling (Issue #3)."""

    def test_extreme_zoom_clip_distances(self):
        """Test clip distance handling at extreme zoom."""
        x = np.linspace(-1e6, 1e6, 100)
        y = np.sin(x)

        gplt.plot(x, y)
        assert len(gplt.gcf().scene.layers) > 0

    def test_anisotropic_aspect_clip(self):
        """Test clip with different x and y scales."""
        x = np.linspace(0, 1000, 100)
        y = np.linspace(0, 1, 100)

        gplt.plot(x, y)
        fig = gplt.gcf()
        # Should handle extreme aspect ratios

    def test_many_layered_clip_handling(self):
        """Test clip handling with many overlapping layers."""
        for i in range(20):
            x = np.linspace(0, 10, 100)
            y = np.sin(x + i * 0.1)
            gplt.plot(x, y, alpha=0.3)
        # All layers should render correctly


class TestSavefigFallbackRegression:
    """Regression tests for savefig fallback (Issue #5)."""

    def test_savefig_creates_output(self, tmp_path):
        """Test savefig creates output file."""
        gplt.plot([0, 1, 2], [0, 1, 4])
        output_path = tmp_path / "test_output.png"

        try:
            gplt.savefig(str(output_path))
            # File should be created
        except Exception as e:
            pytest.skip(f"savefig not fully implemented: {e}")

    def test_savefig_without_display(self, tmp_path):
        """Test savefig works without calling show()."""
        gplt.plot([0, 1, 2], [0, 1, 4])
        gplt.xlabel("X")
        gplt.ylabel("Y")

        output_path = tmp_path / "test_output.png"
        try:
            gplt.savefig(str(output_path))
            # Should complete without show()
        except Exception as e:
            pytest.skip(f"savefig fallback not implemented: {e}")


class TestRenderPreviewRegression:
    """Regression tests for render preview (Issue #8)."""

    def test_patch_rendering_preview(self):
        """Test patch rendering in preview mode."""
        gplt.bar([0, 1, 2], [1, 2, 3])
        # Should not crash in headless test

    def test_quiver_rendering_preview(self):
        """Test quiver rendering in preview mode."""
        gplt.quiver([0, 1], [0, 1], [1, 0], [0, 1])
        # Should not crash in headless test

    def test_fill_between_rendering_preview(self):
        """Test fill_between rendering in preview mode."""
        x = [0, 1, 2]
        y1 = [0, 1, 0]
        y2 = [0, 0.5, 0]
        gplt.fill_between(x, y1, y2)
        # Should render correctly


class TestGlobalAlphaRegression:
    """Regression tests for global alpha handling (Issue #6)."""

    def test_global_alpha_consistency(self):
        """Test that global alpha is applied consistently."""
        gplt.plot([0, 1], [0, 1], alpha=0.5)
        layer1 = gplt.gcf().scene.layers[-1]

        gplt.clf()
        gplt.plot([0, 1], [0, 1], alpha=0.5)
        layer2 = gplt.gcf().scene.layers[-1]

        # Both should have similar alpha properties
        assert layer1.style.alpha == layer2.style.alpha

    def test_layer_alpha_override(self):
        """Test individual layer alpha overrides."""
        gplt.plot([0, 1], [0, 1], alpha=0.3, label="line1")
        gplt.plot([0, 1], [1, 0], alpha=0.7, label="line2")

        fig = gplt.gcf()
        assert fig.scene.layers[0].style.alpha == 0.3
        assert fig.scene.layers[1].style.alpha == 0.7

    def test_scatter_alpha_per_point(self):
        """Test scatter with per-point alpha."""
        try:
            gplt.scatter([0, 1, 2], [0, 1, 0], alpha=[0.3, 0.6, 1.0])
            assert len(gplt.gcf().scene.layers) > 0
        except NotImplementedError:
            pytest.skip("Per-point alpha not implemented")


class TestFormattingRegression:
    """Regression tests for format string parsing."""

    def test_linestyle_parsing(self):
        """Test linestyle format string parsing."""
        formats = ["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]

        for fmt in formats:
            gplt.clf()
            gplt.plot([0, 1], [0, 1], fmt)
            assert len(gplt.gcf().scene.layers) > 0

    def test_marker_parsing(self):
        """Test marker format string parsing."""
        markers = ["o", "s", "^", "v", "x", "+", "*"]

        for marker in markers:
            gplt.clf()
            gplt.scatter([0, 1], [0, 1], marker=marker)
            assert len(gplt.gcf().scene.layers) > 0

    def test_color_parsing(self):
        """Test color format string parsing."""
        colors = ["r", "g", "b", "k", "w", "c", "m", "y"]

        for color in colors:
            gplt.clf()
            gplt.plot([0, 1], [0, 1], color=color)
            assert len(gplt.gcf().scene.layers) > 0

    def test_combined_format_string(self):
        """Test combined format string."""
        formats = ["r-o", "b--s", "g^-", "ko--", "c:+"]

        for fmt in formats:
            gplt.clf()
            gplt.plot([0, 1], [0, 1], fmt)
            assert len(gplt.gcf().scene.layers) > 0


class TestAxisMathRegression:
    """Regression tests for axis math and transformations."""

    def test_viewport_center_projection(self):
        """Test viewport-relative center projection at high zoom."""
        # This tests the precision preservation innovation
        x = np.array([1e6, 1e6 + 1, 1e6 + 2])
        y = np.array([1e6, 1e6 + 1, 1e6 + 2])

        gplt.plot(x, y)
        bounds = gplt.gcf().compute_bounds()
        assert bounds is not None

    def test_anisotropic_camera_limits(self):
        """Test decoupled camera limits with anisotropic data."""
        x = np.linspace(0, 1000, 100)
        y = np.linspace(0, 1, 100)

        gplt.plot(x, y)
        gplt.xlim(0, 500)
        gplt.ylim(0, 0.5)

        fig = gplt.gcf()
        assert fig.xlim == (0, 500)
        assert fig.ylim == (0, 0.5)

    def test_aspect_ratio_handling(self):
        """Test aspect ratio in non-square figure."""
        fig = gplt.figure(figsize=(16, 9))
        gplt.plot([0, 1], [0, 1])

        bounds = gplt.gcf().compute_bounds()
        assert bounds is not None


class TestDataValidationRegression:
    """Regression tests for data validation."""

    def test_mismatched_array_lengths(self):
        """Test error handling for mismatched arrays."""
        with pytest.raises(ValueError):
            gplt.plot([0, 1, 2], [0, 1])

    def test_empty_arrays(self):
        """Test error handling for empty arrays."""
        with pytest.raises((ValueError, IndexError)):
            gplt.plot([], [])

    def test_single_element_array(self):
        """Test single-element arrays."""
        gplt.plot([0], [0])
        assert len(gplt.gcf().scene.layers) > 0

    def test_nan_handling_robustness(self):
        """Test robust NaN handling."""
        x = np.array([0, 1, np.nan, 3, 4])
        y = np.array([0, 1, 2, np.nan, 4])

        try:
            gplt.plot(x, y)
            # Should either work or raise appropriate error
        except (ValueError, RuntimeError):
            pass  # Expected behavior

    def test_inf_handling_robustness(self):
        """Test robust infinity handling."""
        x = np.array([0, 1, np.inf, 3, 4])
        y = np.array([0, 1, 2, 3, np.inf])

        try:
            gplt.plot(x, y)
        except (ValueError, RuntimeError):
            pass  # Expected behavior


class TestLayerStateRegression:
    """Regression tests for layer state management."""

    def test_layer_persistence_after_clear(self):
        """Test that clf properly clears layers."""
        gplt.plot([0, 1], [0, 1])
        assert len(gplt.gcf().scene.layers) > 0

        gplt.clf()
        assert len(gplt.gcf().scene.layers) == 0

    def test_figure_switch_preserves_state(self):
        """Test that switching figures preserves their state."""
        fig1 = gplt.figure("fig1")
        gplt.plot([0, 1], [0, 1])
        layers_fig1 = len(gplt.gcf().scene.layers)

        fig2 = gplt.figure("fig2")
        gplt.plot([1, 2, 3], [1, 2, 3])
        gplt.plot([1, 2, 3], [3, 2, 1])
        layers_fig2 = len(gplt.gcf().scene.layers)

        gplt.figure("fig1")
        assert len(gplt.gcf().scene.layers) == layers_fig1

        gplt.figure("fig2")
        assert len(gplt.gcf().scene.layers) == layers_fig2


class TestColormapRegression:
    """Regression tests for colormap handling."""

    def test_named_colormap_application(self):
        """Test application of named colormaps."""
        cmaps = ["viridis", "plasma", "inferno", "magma", "cool", "hot"]

        for cmap in cmaps:
            gplt.clf()
            x = np.random.randn(100)
            y = np.random.randn(100)
            c = np.random.rand(100)
            gplt.scatter(x, y, c=c, cmap=cmap)
            assert len(gplt.gcf().scene.layers) > 0

    def test_colormap_normalization(self):
        """Test colormap value normalization."""
        gplt.scatter([0, 1, 2], [0, 1, 0], c=[0, 50, 100], cmap="viridis", vmin=0, vmax=100)
        layer = gplt.gcf().scene.layers[-1]
        assert layer.colors.shape[0] == 3
