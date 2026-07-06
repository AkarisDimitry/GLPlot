import numpy as np

import glplot.pyplot as gplt


def demo_read_density():
    n = 500000
    a = np.random.randn(n) * 0.15
    b = np.random.randn(n) * 0.15

    # Enable density mode
    gplt.figure("Read Density Example", density=True, hud=True)
    gplt.lines(a, b, x_range=(-1, 1), color="cyan")

    # We retrieve the GPULinePlot engine instance
    plot = gplt.gcf()

    # Force a single frame draw so densities are accumulated on the GPU before readback.
    plot._draw_exact_view()

    # Retrieve the densities as a 2D numpy array
    densities = plot.get_density_array()

    print("\n--- Density Array Extracted via API ---")
    print(f"Array shape: {densities.shape} (height, width)")
    print(f"Data type:   {densities.dtype}")
    print(f"Min density: {densities.min()}")
    print(f"Max density: {densities.max()}")
    print(f"Mean density: {densities.mean()}")
    print("----------------------------------------\n")

    gplt.show()


if __name__ == "__main__":
    demo_read_density()
