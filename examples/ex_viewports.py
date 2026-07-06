import numpy as np
import glplot.pyplot as gplt
import time


def demo_viewports_and_performance():
    """
    Verification script for deterministic viewports and mass-scale performance.
    """
    print("--- GLPlot Performance & Viewport Test ---")

    # 1. Dataset: 500,000 Polylines (Total 2.5M segments)
    # This specifically tests the new GPU Instanced PolylineRenderer
    n_curves = 100
    n_points = 5000
    x = np.linspace(-10, 10, n_points)

    print(f"Loading {n_curves} high-density polylines ({n_curves * n_points} points)...")
    for i in range(n_curves):
        offset = i * 0.1
        y = np.sin(x + offset) + np.random.normal(0, 0.05, n_points)
        color = (0.2, 0.5, 0.8, 0.4)
        gplt.plot(x, y, color=color, width=1.5, label=f"Curve {i}")

    # 2. Dataset: 1,000,000 Line Family (High-performance path)
    print("Loading 1,000,000 instanced lines...")
    n_lines = 1000000
    ab = np.zeros((n_lines, 2), dtype=np.float32)
    ab[:, 0] = np.random.normal(0, 0.5, n_lines)  # slopes
    ab[:, 1] = np.random.normal(0, 1.0, n_lines)  # intercepts
    gplt.lines(
        ab[:, 0], ab[:, 1], x_range=(-10, 10), alpha=0.01, width=1.0, label="Massive Dataset"
    )

    # 3. Deterministic Viewport
    # We want to start zoomed into a specific feature
    print("Setting deterministic initial view: X[-2, 2], Y[-1.5, 1.5]")
    gplt.xlim(-2, 2)
    gplt.ylim(-1.5, 1.5)

    gplt.title("Deterministic Viewport & 1.5M Primitives")

    print("Launching engine...")
    gplt.show()


if __name__ == "__main__":
    demo_viewports_and_performance()
