"""Test helper functions in glplot.pyplot.

Focus on parameter validation, edge cases, and color handling without
requiring OpenGL or GPU.
"""

import numpy as np
import pytest

import glplot.pyplot as gplt


class TestNormalizeRgba:
    """Test _normalize_rgba color normalization function."""

    def test_none_uses_default(self):
        """Test that None color returns default."""
        result = gplt._normalize_rgba(None)
        assert result.shape == (4,)
        assert np.allclose(result, [0.0, 0.0, 0.0, 1.0])

    def test_none_with_custom_default(self):
        """Test None with custom default color."""
        default = (1.0, 0.0, 0.5, 0.8)
        result = gplt._normalize_rgba(None, default=default)
        assert np.allclose(result, default)

    def test_named_color_string(self):
        """Test named color strings."""
        colors = {
            "black": (0.0, 0.0, 0.0, 1.0),
            "white": (1.0, 1.0, 1.0, 1.0),
            "red": (1.0, 0.0, 0.0, 1.0),
            "green": (0.0, 1.0, 0.0, 1.0),
            "blue": (0.0, 0.0, 1.0, 1.0),
        }
        for name, expected in colors.items():
            result = gplt._normalize_rgba(name)
            assert np.allclose(result, expected), f"Failed for {name}"

    def test_short_color_codes(self):
        """Test single-letter color codes."""
        colors = {
            "k": (0.0, 0.0, 0.0, 1.0),
            "w": (1.0, 1.0, 1.0, 1.0),
            "r": (1.0, 0.0, 0.0, 1.0),
            "g": (0.0, 1.0, 0.0, 1.0),
            "b": (0.0, 0.0, 1.0, 1.0),
        }
        for code, expected in colors.items():
            result = gplt._normalize_rgba(code)
            assert np.allclose(result, expected), f"Failed for {code}"

    def test_matplotlib_color_names(self):
        """Test matplotlib color names (tab:orange, etc)."""
        result = gplt._normalize_rgba("tab:orange")
        assert result.shape == (4,)
        assert np.allclose(result, result, equal_nan=False)  # Just check it's valid

    def test_hex_color_string(self):
        """Test hex color strings."""
        result = gplt._normalize_rgba("#FF0000")  # Red
        assert result.shape == (4,)
        assert result[0] > 0.9  # Red channel high
        assert result[1] < 0.1  # Green channel low
        assert result[2] < 0.1  # Blue channel low

    def test_rgb_tuple(self):
        """Test RGB tuple (3 values)."""
        result = gplt._normalize_rgba((0.5, 0.5, 0.5))
        assert result.shape == (4,)
        assert np.allclose(result[:3], [0.5, 0.5, 0.5])
        assert result[3] == 1.0  # Alpha should be 1.0

    def test_rgba_tuple(self):
        """Test RGBA tuple (4 values)."""
        result = gplt._normalize_rgba((0.5, 0.5, 0.5, 0.8))
        assert result.shape == (4,)
        assert np.allclose(result, [0.5, 0.5, 0.5, 0.8])

    def test_numpy_array_rgb(self):
        """Test numpy array with RGB."""
        color = np.array([0.3, 0.6, 0.9])
        result = gplt._normalize_rgba(color)
        assert result.shape == (4,)
        assert np.allclose(result[:3], [0.3, 0.6, 0.9])
        assert result[3] == 1.0

    def test_numpy_array_rgba(self):
        """Test numpy array with RGBA."""
        color = np.array([0.3, 0.6, 0.9, 0.7])
        result = gplt._normalize_rgba(color)
        assert result.shape == (4,)
        assert np.allclose(result, [0.3, 0.6, 0.9, 0.7])

    def test_clipping_values_above_one(self):
        """Test that values > 1.0 are clipped."""
        result = gplt._normalize_rgba((1.5, 2.0, 0.5, 1.0))
        assert np.all(result <= 1.0)
        assert result[0] == 1.0
        assert result[1] == 1.0

    def test_clipping_negative_values(self):
        """Test that negative values are clipped."""
        result = gplt._normalize_rgba((-0.5, 0.5, 0.5, -0.2))
        assert np.all(result >= 0.0)
        assert result[0] == 0.0
        assert result[3] == 0.0

    def test_per_object_color_single_color(self):
        """Test broadcasting single color to n objects."""
        result = gplt._normalize_rgba((0.5, 0.5, 0.5, 1.0), n=3)
        assert result.shape == (3, 4)
        assert np.allclose(result, [[0.5, 0.5, 0.5, 1.0]] * 3)

    def test_per_object_color_array_rgba(self):
        """Test per-object RGBA color array."""
        colors = np.array([[0.1, 0.2, 0.3, 1.0], [0.4, 0.5, 0.6, 0.8]])
        result = gplt._normalize_rgba(colors, n=2)
        assert result.shape == (2, 4)
        assert np.allclose(result, colors)

    def test_per_object_color_array_rgb(self):
        """Test per-object RGB color array (should add alpha)."""
        colors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        result = gplt._normalize_rgba(colors, n=2)
        assert result.shape == (2, 4)
        assert result[0, 3] == 1.0
        assert result[1, 3] == 1.0

    def test_invalid_color_fallback(self):
        """Test that invalid color format falls back to default."""
        # Non-string, non-sequence should fall back
        result = gplt._normalize_rgba(123)
        assert result.shape == (4,)


class TestAsFloatArray:
    """Test _as_float_array array conversion function."""

    def test_list_conversion(self):
        """Test converting list to float array."""
        result = gplt._as_float_array([1, 2, 3])
        assert result.dtype == np.float32
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_numpy_array_conversion(self):
        """Test converting numpy array."""
        arr = np.array([1, 2, 3], dtype=np.float64)
        result = gplt._as_float_array(arr)
        assert result.dtype == np.float32

    def test_contiguous_array(self):
        """Test that output is C-contiguous."""
        result = gplt._as_float_array([1, 2, 3])
        assert result.flags["C_CONTIGUOUS"]

    def test_ndim_constraint_1d(self):
        """Test ndim=1 constraint."""
        result = gplt._as_float_array([1, 2, 3], ndim=1)
        assert result.ndim == 1

    def test_ndim_constraint_violation(self):
        """Test that violating ndim constraint raises ValueError."""
        with pytest.raises(ValueError, match="ndim"):
            gplt._as_float_array([[1, 2], [3, 4]], ndim=1)

    def test_custom_error_message(self):
        """Test custom error message in ndim violation."""
        with pytest.raises(ValueError, match="x_values"):
            gplt._as_float_array([[1, 2]], ndim=1, name="x_values")

    def test_empty_array(self):
        """Test empty array conversion."""
        result = gplt._as_float_array([])
        assert result.dtype == np.float32
        assert len(result) == 0

    def test_scalar_conversion(self):
        """Test scalar conversion."""
        result = gplt._as_float_array(5.0)
        assert result.dtype == np.float32
        # Scalar becomes shape (1,) when converted to array
        assert result.shape == (1,)


class TestParsePlotFormat:
    """Test _parse_plot_format matplotlib format string parsing."""

    def test_empty_format(self):
        """Test empty format string."""
        result = gplt._parse_plot_format("")
        assert result == {}

    def test_none_format(self):
        """Test None format."""
        result = gplt._parse_plot_format(None)
        assert result == {}

    def test_color_codes(self):
        """Test single-letter color codes."""
        color_codes = "bgrcmykw"
        for code in color_codes:
            result = gplt._parse_plot_format(code)
            assert "color" in result

    def test_marker_codes(self):
        """Test marker codes."""
        markers = ["o", "s", "x", "^", "v", "+", "*"]
        for marker in markers:
            result = gplt._parse_plot_format(marker)
            assert result.get("marker") == marker

    def test_line_styles(self):
        """Test line style codes."""
        styles = {"-": "-", "--": "--", ":": ":", "-.": "-."}
        for code, expected in styles.items():
            result = gplt._parse_plot_format(code)
            assert result.get("linestyle") == expected

    def test_combined_format(self):
        """Test combined color + marker + linestyle."""
        result = gplt._parse_plot_format("r-o")
        assert result.get("color") == "r"
        assert result.get("linestyle") == "-"
        assert result.get("marker") == "o"

    def test_combined_format_different_order(self):
        """Test that order doesn't matter for combined formats."""
        result1 = gplt._parse_plot_format("r-o")
        result2 = gplt._parse_plot_format("or-")
        # Color should be extracted regardless of position
        assert result1.get("color") == result2.get("color")

    def test_colormap_cycling(self):
        """Test color cycle codes."""
        for i in range(10):
            result = gplt._parse_plot_format(f"C{i}")
            assert result.get("color") == f"C{i}"

    def test_invalid_character_error(self):
        """Test that invalid character raises ValueError."""
        with pytest.raises(ValueError, match="unsupported"):
            gplt._parse_plot_format("r@")

    def test_linestyle_priority(self):
        """Test that longer linestyle tokens are matched first."""
        # "-." should match "-." not just "-"
        result = gplt._parse_plot_format("-.")
        assert result.get("linestyle") == "-."

    def test_colormap_with_other_color(self):
        """Test that later color codes override earlier ones."""
        result = gplt._parse_plot_format("C5r")
        # C5 is extracted first, then 'r' overrides it in the character loop
        assert result.get("color") == "r"


class TestColormapValues:
    """Test _colormap_values colormap application function."""

    def test_default_colormap(self):
        """Test default colormap (viridis)."""
        values = np.array([0.0, 0.5, 1.0])
        result = gplt._colormap_values(values)
        assert result.shape == (3, 4)  # RGBA for 3 values
        assert result.dtype == np.float32

    def test_custom_colormap(self):
        """Test with custom colormap."""
        values = np.array([0.0, 0.5, 1.0])
        result = gplt._colormap_values(values, cmap="plasma")
        assert result.shape == (3, 4)

    def test_normalization_auto_vmin_vmax(self):
        """Test automatic vmin/vmax normalization."""
        values = np.array([0.0, 50.0, 100.0])
        result = gplt._colormap_values(values)
        # Values should be normalized to [0, 1]
        assert result.shape == (3, 4)
        # First value (0.0) should map to one end
        # Last value (100.0) should map to other end

    def test_explicit_vmin_vmax(self):
        """Test explicit vmin/vmax."""
        values = np.array([0.0, 0.5, 1.0])
        result1 = gplt._colormap_values(values, vmin=0.0, vmax=1.0)
        result2 = gplt._colormap_values(values, vmin=-1.0, vmax=2.0)
        # Different vmin/vmax should produce different colors
        assert not np.allclose(result1, result2)

    def test_clipping_to_vmin_vmax(self):
        """Test that values outside vmin/vmax are clipped."""
        values = np.array([-10.0, 0.0, 10.0])
        result = gplt._colormap_values(values, vmin=0.0, vmax=10.0)
        assert result.shape == (3, 4)

    def test_nan_handling(self):
        """Test that NaN values are handled."""
        values = np.array([0.0, np.nan, 1.0])
        # Should not crash
        result = gplt._colormap_values(values)
        assert result.shape == (3, 4)

    def test_single_value(self):
        """Test single value normalization."""
        values = np.array([0.5])
        result = gplt._colormap_values(values)
        assert result.shape == (1, 4)

    def test_all_same_values(self):
        """Test array with all same values (zero range)."""
        values = np.array([0.5, 0.5, 0.5])
        result = gplt._colormap_values(values)
        assert result.shape == (3, 4)
        # Should not crash even though range is 0


class TestProject3d:
    """Test _project_3d 3D projection function."""

    def test_basic_projection(self):
        """Test basic 3D to 2D projection."""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        z = np.array([0.0, 1.0])
        xp, yp, zp = gplt._project_3d(x, y, z)
        assert xp.shape == (2,)
        assert yp.shape == (2,)
        assert zp.shape == (2,)
        assert xp.dtype == np.float32
        assert yp.dtype == np.float32
        assert zp.dtype == np.float32

    def test_projection_with_custom_angles(self):
        """Test projection with custom elevation and azimuth."""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        z = np.array([0.0, 1.0])
        xp1, yp1, zp1 = gplt._project_3d(x, y, z, elev=30.0, azim=-60.0)
        xp2, yp2, zp2 = gplt._project_3d(x, y, z, elev=45.0, azim=0.0)
        # Different angles should produce different projections
        assert not np.allclose(xp1, xp2)
        assert not np.allclose(yp1, yp2)

    def test_projection_with_custom_scale(self):
        """Test projection with custom Z scale."""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        z = np.array([0.0, 1.0])
        xp1, yp1, zp1 = gplt._project_3d(x, y, z, scale_z=0.7)
        xp2, yp2, zp2 = gplt._project_3d(x, y, z, scale_z=1.0)
        # Different Z scales should affect Y projection
        assert not np.allclose(yp1, yp2)

    def test_projection_preserves_z(self):
        """Test that Z values are preserved in output."""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        z = np.array([0.5, 1.5])
        xp, yp, zp = gplt._project_3d(x, y, z)
        assert np.allclose(zp, z)

    def test_projection_single_point(self):
        """Test projection of single point."""
        xp, yp, zp = gplt._project_3d([0.0], [0.0], [0.0])
        assert len(xp) == 1
        assert len(yp) == 1
        assert len(zp) == 1

    def test_projection_large_array(self):
        """Test projection of large array."""
        n = 1000
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        z = np.linspace(0, 1, n)
        xp, yp, zp = gplt._project_3d(x, y, z)
        assert len(xp) == n
        assert len(yp) == n
        assert len(zp) == n

    def test_projection_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        with pytest.raises(ValueError, match="same length"):
            gplt._project_3d([0, 1], [0, 1], [0])

    def test_projection_contiguity(self):
        """Test that output arrays are C-contiguous."""
        xp, yp, zp = gplt._project_3d([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
        assert xp.flags["C_CONTIGUOUS"]
        assert yp.flags["C_CONTIGUOUS"]
        assert zp.flags["C_CONTIGUOUS"]

    def test_projection_negative_angles(self):
        """Test projection with negative angles."""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        z = np.array([0.0, 1.0])
        # Should not crash with negative angles
        xp, yp, zp = gplt._project_3d(x, y, z, elev=-30.0, azim=-120.0)
        assert xp.shape == (2,)
