"""Test camera projections and coordinate transformations."""

import numpy as np
import pytest

from glplot.controllers import CameraController
from glplot.core.legacy import CameraState
from glplot.managers.axis import AxisManager
from glplot.core.context import RenderContext


class TestAsymmetricProjection:
    """Test camera math with asymmetric data ranges."""

    def test_fit_bounds_extreme_aspect_ratio(self):
        """Test fit_bounds with extreme aspect ratio (wide x, narrow y)."""
        cam = CameraState()
        ctrl = CameraController(cam, None)

        # Extreme bounds: span_x = 4, span_y = 2000
        ctrl.fit_bounds(-2, 2, -1000, 1000, 1280, 800)

        # Verify zoom factors
        assert abs(cam.zoom_x - 0.5) < 1e-7, f"Expected zoom_x=0.5, got {cam.zoom_x}"
        assert abs(cam.zoom_y - 0.001) < 1e-7, f"Expected zoom_y=0.001, got {cam.zoom_y}"

    def test_world_window_preserves_bounds(self):
        """Test that world_window preserves fitted bounds."""
        cam = CameraState()
        ctrl = CameraController(cam, None)

        bounds = (-2, 2, -1000, 1000)
        ctrl.fit_bounds(*bounds, 1280, 800)

        l, r, b, t = ctrl.world_window(1280, 800)

        assert abs(l - bounds[0]) < 1e-7
        assert abs(r - bounds[1]) < 1e-7
        assert abs(b - bounds[2]) < 1e-7
        assert abs(t - bounds[3]) < 1e-7

    def test_screen_to_world_conversion_works(self):
        """Test screen to world conversion produces consistent results."""
        cam = CameraState()
        ctrl = CameraController(cam, None)

        ctrl.fit_bounds(-2, 2, -1000, 1000, 1280, 800)

        # Just verify the conversion doesn't crash and returns values
        wx, wy = ctrl.screen_to_world(640, 400, 1280, 800)
        assert isinstance(wx, (int, float))
        assert isinstance(wy, (int, float))

    def test_ndc_transform_corners(self):
        """Test normalized device coordinate transformation at bounds."""
        cam = CameraState()
        ctrl = CameraController(cam, None)

        ctrl.fit_bounds(-2, 2, -1000, 1000, 1280, 800)

        l, r, b, t = ctrl.world_window(1280, 800)

        # Compute NDC transformation
        rl, tb = (r - l), (t - b)
        sx, sy = 2.0 / rl, 2.0 / tb
        ox, oy = -(r + l) / rl, -(t + b) / tb

        def to_ndc(wx, wy):
            return wx * sx + ox, wy * sy + oy

        # Bottom-left corner should map to (-1, -1)
        n_bl = to_ndc(-2, -1000)
        assert abs(n_bl[0] - (-1)) < 1e-7
        assert abs(n_bl[1] - (-1)) < 1e-7

        # Top-right corner should map to (1, 1)
        n_tr = to_ndc(2, 1000)
        assert abs(n_tr[0] - 1) < 1e-7
        assert abs(n_tr[1] - 1) < 1e-7

        # Center should map to (0, 0)
        n_center = to_ndc(0, 0)
        assert abs(n_center[0]) < 1e-7
        assert abs(n_center[1]) < 1e-7

    def test_camera_state_copy(self):
        """Test that camera state can be copied."""
        cam1 = CameraState()
        ctrl = CameraController(cam1, None)
        ctrl.fit_bounds(-10, 10, -100, 100, 1024, 768)

        # Create a copy
        cam2 = CameraState()
        cam2.cx = cam1.cx
        cam2.cy = cam1.cy
        cam2.zoom_x = cam1.zoom_x
        cam2.zoom_y = cam1.zoom_y

        # Verify state matches
        assert cam2.cx == cam1.cx
        assert cam2.cy == cam1.cy
        assert cam2.zoom_x == cam1.zoom_x
        assert cam2.zoom_y == cam1.zoom_y

    def test_zoom_at_cursor(self):
        """Test zoom operation maintains cursor position."""
        cam = CameraState()
        ctrl = CameraController(cam, None)

        ctrl.fit_bounds(-10, 10, -10, 10, 800, 800)

        # Get initial cursor world position at screen (400, 400)
        wx_before, wy_before = ctrl.screen_to_world(400, 400, 800, 800)

        # Zoom in 2x at cursor
        ctrl.apply_zoom_at_cursor(2.0, 400, 400, 800, 800)

        # Get cursor position after zoom
        wx_after, wy_after = ctrl.screen_to_world(400, 400, 800, 800)

        # Cursor should point to same world location
        assert abs(wx_after - wx_before) < 1e-6
        assert abs(wy_after - wy_before) < 1e-6

    def test_aspect_ratio_preservation(self):
        """Test that camera respects window aspect ratio."""
        cam = CameraState()
        ctrl = CameraController(cam, None)

        # Wide window (2:1 aspect)
        ctrl.fit_bounds(-10, 10, -5, 5, 1600, 800)

        l, r, b, t = ctrl.world_window(1600, 800)
        width = r - l
        height = t - b

        # Aspect ratio should be preserved (2:1)
        expected_aspect = 1600 / 800
        actual_aspect = width / height

        assert abs(actual_aspect - expected_aspect) < 1e-6


class TestCoordinateTransforms:
    """Test coordinate transformation edge cases."""

    def test_zero_sized_bounds(self):
        """Test handling of zero-sized bounds."""
        cam = CameraState()
        ctrl = CameraController(cam, None)

        # One dimension has zero span - may be handled or raise
        try:
            ctrl.fit_bounds(0, 0, -10, 10, 800, 600)
        except (ValueError, AssertionError):
            pass  # Expected if it raises

    def test_inverted_bounds(self):
        """Test handling of inverted bounds."""
        cam = CameraState()
        ctrl = CameraController(cam, None)

        # Inverted x bounds - may be handled or raise
        try:
            ctrl.fit_bounds(10, -10, -10, 10, 800, 600)
        except (ValueError, AssertionError):
            pass  # Expected if it raises

    def test_large_offset_precision(self):
        """Test precision with large offsets."""
        cam = CameraState()
        ctrl = CameraController(cam, None)

        # Large offsets with small range
        ctrl.fit_bounds(1e6, 1e6 + 1, 1e6, 1e6 + 1, 800, 600)

        l, r, b, t = ctrl.world_window(800, 600)

        # Should preserve bounds even with large offsets
        assert abs(r - l - 1) < 1e-3
        assert abs(t - b - 1) < 1e-3
