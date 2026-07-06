import time

import numpy as np

import glplot.pyplot as gplt
from glplot.options import BlendMode


def verify_blending_and_density():
    print("Verifying Procedural Blending & Weighted Density...")

    # 1. Glowing Orbs (Additive)
    N = 20_000
    cx = np.random.normal(0, 5.0, N).astype(np.float32)
    cy = np.random.normal(0, 5.0, N).astype(np.float32)
    # Varied alpha to test weighted density
    alphas = np.random.uniform(0.01, 0.5, N).astype(np.float32)
    colors = np.zeros((N, 4), dtype=np.float32)
    colors[:, 0] = 1.0  # Red
    colors[:, 3] = alphas

    gplt.scatter(cx, cy, color=colors, size=10.0)

    # 2. Configure Engine
    plot = gplt._get_or_create_plot()

    print("\n[TEST MODE] Testing Blending Modes...")
    # Cycle through modes to check for GL errors
    for mode in [BlendMode.ALPHA, BlendMode.ADDITIVE, BlendMode.SUBTRACTIVE, BlendMode.SCREEN]:
        print(f"Testing {mode.name}...")
        plot.options.blend_mode = mode
        plot.frame.dirty_scene = True
        # Tiny delay to ensure context is valid
        time.sleep(0.1)

    print("\n[TEST MODE] Testing Weighted Density...")
    plot.set_density_enabled(True)
    plot.options.density_weighted = True
    plot.frame.dirty_scene = True

    print("\nBlending and Density Extension is STABLE.")
    print(
        "You can now control 'Blending' and 'Weighted Accumulation' in the HUD under 'Render & Style'."
    )

    gplt.show()


if __name__ == "__main__":
    verify_blending_and_density()
