import time

import numpy as np

import glplot.pyplot as gplt


def test_polyline_colormapping():
    # Generate multiple overlapping polylines
    n_curves = 10
    x = np.linspace(-5, 5, 200)

    print(f"Creating {n_curves} polylines...")
    for i in range(n_curves):
        # Different frequencies and phases
        y = np.sin(x * (1 + i * 0.1) + i * 0.5) + i * 0.2
        label = f"Curve {i}"

        # We plot with a default color (black/gray),
        # but we expect it to change when 'Line Colormap' is toggled.
        gplt.plot(x, y, color=(0.2, 0.2, 0.2, 1.0), width=2.0, label=label)

    print("\nControls:")
    print(" - Press 'C' to toggle Line Colormap (once implemented, or use HUD)")
    print(" - Press 'D' to toggle Density/Exact view")

    # Enable colormap by default for this test
    plot = gplt.get_engine()
    plot.options.line_colormap_enabled = True
    plot.options.density_scheme_index = 0  # Inferno (usually)

    gplt.show()


if __name__ == "__main__":
    test_polyline_colormapping()
