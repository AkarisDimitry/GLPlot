import numpy as np

from glplot.controllers import CameraController
from glplot.core.context import RenderContext
from glplot.core.legacy import CameraState
from glplot.managers.axis import AxisManager


def test_math():
    print("Testing Asymmetric Camera Math...")
    cam = CameraState()
    ctrl = CameraController(cam, None)

    # 1. Fit to extreme bounds
    # span_x = 4, span_y = 2000
    ctrl.fit_bounds(-2, 2, -1000, 1000, 1280, 800)

    print(f"Camera zoom_x: {cam.zoom_x}, zoom_y: {cam.zoom_y}")
    # Expected: zoom_x = 2/4 = 0.5, zoom_y = 2/2000 = 0.001
    assert abs(cam.zoom_x - 0.5) < 1e-7
    assert abs(cam.zoom_y - 0.001) < 1e-7

    # 2. Check world window
    l, r, b, t = ctrl.world_window(1280, 800)
    print(f"World window: ({l}, {r}, {b}, {t})")
    assert abs(l - (-2)) < 1e-7
    assert abs(r - 2) < 1e-7
    assert abs(b - (-1000)) < 1e-7
    assert abs(t - 1000) < 1e-7

    # 3. Check screen to world
    # Center screen (640, 400) -> world (0, 0)
    wx, wy = ctrl.screen_to_world(640, 400, 1280, 800)
    print(f"Screen center to world: ({wx}, {wy})")
    assert abs(wx) < 1e-7
    assert abs(wy) < 1e-7

    # 4. Check NDC transform
    # Equivalent to _get_ndc_transform in engine.py
    rl, tb = (r - l), (t - b)
    sx, sy = 2.0 / rl, 2.0 / tb
    ox, oy = -(r + l) / rl, -(t + b) / tb

    print(f"NDC Scale: ({sx}, {sy}), Offset: ({ox}, {oy})")

    # Verify mapping
    def to_ndc(wx, wy):
        return wx * sx + ox, wy * sy + oy

    print(f"World (-2, -1000) -> NDC: {to_ndc(-2, -1000)}")
    print(f"World (2, 1000) -> NDC: {to_ndc(2, 1000)}")
    print(f"World (0, 0) -> NDC: {to_ndc(0, 0)}")

    n1 = to_ndc(-2, -1000)
    n2 = to_ndc(2, 1000)
    assert abs(n1[0] - (-1)) < 1e-7 and abs(n1[1] - (-1)) < 1e-7
    assert abs(n2[0] - 1) < 1e-7 and abs(n2[1] - 1) < 1e-7

    # 5. Check Axis Manager
    plot_stub = type("Plot", (), {"options": None})()
    axis = AxisManager(plot_stub)
    ctx = RenderContext(
        mvp=None,
        window_world=(l, r, b, t),
        width_px=1280,
        height_px=800,
        fb_width=1280,
        fb_height=800,
        mode=None,
    )
    axis.update(ctx)
    print(f"X Ticks: {axis.ticks_x.major}")
    print(f"Y Ticks: {axis.ticks_y.major}")

    assert len(axis.ticks_x.major) > 0
    assert len(axis.ticks_y.major) > 0

    print("All math checks passed.")


if __name__ == "__main__":
    test_math()
