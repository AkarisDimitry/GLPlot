#!/usr/bin/env python3
"""Diagnose camera state persistence and viewport behavior.

This tool helps debug camera controller behavior, including state
persistence across operations and viewport transformations.

Usage:
    python tools/diagnose_camera_state.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import numpy as np

import glplot.pyplot as gplt


def diagnose_camera_state():
    """Diagnose camera state persistence."""
    print("=" * 70)
    print("GLPlot Camera State Diagnostics")
    print("=" * 70)

    # Create a simple plot
    print("\n--- CREATING TEST PLOT ---")
    x = np.linspace(-10, 10, 1000)
    y = np.sin(x)

    gplt.figure("Camera Diagnostics", figsize=(8, 6))
    gplt.plot(x, y, color=(0.2, 0.4, 0.8, 1.0), label="sin(x)")
    gplt.xlabel("x")
    gplt.ylabel("y")
    gplt.title("Camera State Test")

    plot = gplt._get_or_create_plot()

    print(f"Figure created: {plot.title}")
    print(f"Initial dimensions: {plot.width}x{plot.height}")

    # Check initial camera state
    print("\n--- INITIAL CAMERA STATE ---")
    print(f"  Center: ({plot.camera.cx:.6f}, {plot.camera.cy:.6f})")
    print(f"  Zoom: ({plot.camera.zoom_x:.6f}, {plot.camera.zoom_y:.6f})")
    print(f"  Pan: ({plot.camera.pan_x:.6f}, {plot.camera.pan_y:.6f})")

    # Set explicit limits
    print("\n--- SETTING EXPLICIT LIMITS ---")
    print("  Setting xlim(-5, 5)")
    gplt.xlim(-5, 5)

    print(f"  Center after xlim: ({plot.camera.cx:.6f}, {plot.camera.cy:.6f})")
    print(f"  Zoom after xlim: ({plot.camera.zoom_x:.6f}, {plot.camera.zoom_y:.6f})")

    # Simulate pan
    print("\n--- SIMULATING PAN ---")
    initial_cx = plot.camera.cx
    print(f"  Initial cx: {initial_cx:.6f}")
    plot.camera.cx += 1.0
    print(f"  After +1.0 pan: {plot.camera.cx:.6f}")
    print(f"  Pan registered: {plot.camera.cx - initial_cx:.6f}")

    # Check view persistence
    print("\n--- CHECKING VIEW PERSISTENCE ---")
    print("  Current xlim:", (plot.xlim if hasattr(plot, "xlim") else "N/A"))
    print("  Current ylim:", (plot.ylim if hasattr(plot, "ylim") else "N/A"))

    # Test set_view with explicit limits
    print("\n--- TESTING SET_VIEW ---")
    print("  Calling set_view(xlim=(-5, 5), ylim=(-1, 1))")
    try:
        plot.set_view(xlim=(-5, 5), ylim=(-1, 1))
        print(
            f"  Camera after set_view: zoom_x={plot.camera.zoom_x:.6f}, zoom_y={plot.camera.zoom_y:.6f}"
        )
    except Exception as e:
        print(f"  Error: {e}")

    # Test window resize effect
    print("\n--- TESTING WINDOW RESIZE EFFECT ---")
    for aspect in [1.0, 2.0, 0.5]:
        width = int(800 * aspect)
        height = 800

        print(f"\n  Resizing to {width}x{height} (aspect={aspect:.1f})")
        plot.width = width
        plot.height = height

        try:
            plot.set_view(xlim=(-5, 5), ylim=(-1, 1))
            print(f"    Zoom after resize: ({plot.camera.zoom_x:.6f}, {plot.camera.zoom_y:.6f})")
        except Exception as e:
            print(f"    Error: {e}")

    # Check bounds computation
    print("\n--- CHECKING BOUNDS COMPUTATION ---")
    try:
        bounds = plot.compute_bounds()
        if bounds:
            xmin, xmax, ymin, ymax = bounds
            print(f"  Computed bounds: ({xmin:.2f}, {xmax:.2f}, {ymin:.2f}, {ymax:.2f})")
        else:
            print("  No bounds computed")
    except Exception as e:
        print(f"  Error computing bounds: {e}")

    print("\n" + "=" * 70)
    print("✓ Diagnostics completed")
    print("=" * 70)
    print("\nNotes:")
    print("  - This is a diagnostic tool for debugging camera behavior")
    print("  - No rendering occurs; all operations are computational")
    print("  - Check the output above for camera state consistency")


if __name__ == "__main__":
    try:
        diagnose_camera_state()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
