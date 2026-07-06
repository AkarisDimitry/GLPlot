#!/usr/bin/env python3
"""Validate runtime mathematical operations in GLPlot.

This diagnostic tool verifies that camera controller math operations
produce expected results at runtime.

Usage:
    python tools/validate_runtime_math.py
"""

import sys
import os

# Ensure GLPlot is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import glplot
from glplot.controllers import CameraController
from glplot.core.legacy import CameraState
from glplot.options import EngineOptions
import inspect


def validate_math():
    """Validate camera controller mathematical operations."""
    print("=" * 70)
    print("GLPlot Runtime Math Validation")
    print("=" * 70)

    print("\n--- ENVIRONMENT INFO ---")
    print(f"GLPlot Location: {glplot.__file__}")
    print(f"GLPlot Version: {glplot.__version__}")

    # Inspect CameraController source
    print("\n--- CAMERA CONTROLLER METHODS ---")
    print("world_window method:")
    try:
        source = inspect.getsource(CameraController.world_window)
        print(source[:200] + "..." if len(source) > 200 else source)
    except Exception as e:
        print(f"  (Could not retrieve source: {e})")

    # Test fit_bounds with asymmetric data
    print("\n--- TESTING ASYMMETRIC BOUNDS ---")
    cam = CameraState()
    ctrl = CameraController(cam, EngineOptions())

    test_cases = [
        ((-2, 2, -1000, 1000), (800, 600), "Extreme Y range"),
        ((-10, 10, -10, 10), (1600, 800), "Wide window"),
        ((0, 1, 0, 1), (400, 400), "Unit square"),
        ((1e6, 1e6 + 10, 1e6, 1e6 + 10), (800, 600), "Large offset"),
    ]

    all_passed = True

    for bounds, window_size, description in test_cases:
        print(f"\n  Test: {description}")
        print(f"    Bounds: {bounds}")
        print(f"    Window: {window_size}")

        try:
            xmin, xmax, ymin, ymax = bounds
            width, height = window_size

            ctrl.fit_bounds(xmin, xmax, ymin, ymax, width, height)

            l, r, b, t = ctrl.world_window(width, height)

            # Verify bounds are preserved
            tol = 1e-5
            checks = [
                (abs(l - xmin) < tol, f"Left: {l} ≈ {xmin}"),
                (abs(r - xmax) < tol, f"Right: {r} ≈ {xmax}"),
                (abs(b - ymin) < tol, f"Bottom: {b} ≈ {ymin}"),
                (abs(t - ymax) < tol, f"Top: {t} ≈ {ymax}"),
            ]

            passed = all(check[0] for check in checks)

            for check, desc in checks:
                status = "✓" if check else "✗"
                print(f"    {status} {desc}")

            if not passed:
                all_passed = False

        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            all_passed = False

    # Test screen to world conversion
    print("\n--- TESTING SCREEN-TO-WORLD CONVERSION ---")
    cam = CameraState()
    ctrl = CameraController(cam, EngineOptions())

    ctrl.fit_bounds(-10, 10, -10, 10, 800, 600)

    # Screen center should map to world center
    wx, wy = ctrl.screen_to_world(400, 300, 800, 600)

    print(f"  Screen center (400, 300) -> World ({wx:.6f}, {wy:.6f})")
    print(f"  Expected world center: (0, 0)")

    if abs(wx) < 1e-5 and abs(wy) < 1e-5:
        print("  ✓ Screen-to-world conversion correct")
    else:
        print("  ✗ Screen-to-world conversion incorrect")
        all_passed = False

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All math validations PASSED")
        print("=" * 70)
        return True
    else:
        print("✗ Some math validations FAILED")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = validate_math()
    sys.exit(0 if success else 1)
