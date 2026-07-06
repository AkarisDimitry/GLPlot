import os
import sys

# Force local glplot import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glplot.pyplot as gplt
import numpy as np


def test_aspect_ratio():
    print("Testing aspect ratio fix...")

    # 1. Create data with extreme ranges
    x = np.linspace(-2, 2, 1000)
    y = 500 * np.sin(x * 10)

    gplt.figure("Aspect Ratio Test", width=800, height=600)
    gplt.plot(x, y, color="blue", label="Sinusoidal")

    # 2. Set extreme limits
    gplt.xlim(-2, 2)
    gplt.ylim(-1000, 1000)

    # 3. Verify limits via API
    lx = gplt.xlim()
    ly = gplt.ylim()

    print(f"Current X limits: {lx}")
    print(f"Current Y limits: {ly}")

    expected_x = (-2.0, 2.0)
    expected_y = (-1000.0, 1000.0)

    # Allow small floating point epsilon
    assert abs(lx[0] - expected_x[0]) < 1e-5
    assert abs(lx[1] - expected_x[1]) < 1e-5
    assert abs(ly[0] - expected_y[0]) < 1e-5
    assert abs(ly[1] - expected_y[1]) < 1e-5

    print("Success: API returns correct decoupled limits.")

    # 4. Test Zero Span (Horizontal Line)
    gplt.cla()
    gplt.plot([-10, 10], [5, 5], color="red", label="Horizontal")
    gplt.autoscale()

    lx_zero = gplt.xlim()
    ly_zero = gplt.ylim()
    print(f"Autoscale on horizontal line: X={lx_zero}, Y={ly_zero}")

    # Should have a sane Y range despite zero span in data
    assert ly_zero[1] > ly_zero[0]

    print("Verification complete.")
    # gplt.show() # Uncomment to see visually if running locally


if __name__ == "__main__":
    test_aspect_ratio()
