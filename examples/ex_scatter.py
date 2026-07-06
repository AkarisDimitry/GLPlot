import numpy as np

import glplot.pyplot as gplt

# Generate 5 million points
N = 5_000_000
print(f"Generating {N} points cloud...")

# Clusters
x = np.concatenate(
    [
        np.random.normal(-2, 1.0, N // 4),
        np.random.normal(2, 0.5, N // 4),
        np.random.normal(0, 3.0, N // 2),  # Noise
    ]
)
y = np.concatenate(
    [
        np.random.normal(1, 0.5, N // 4),
        np.random.normal(-2, 1.0, N // 4),
        np.random.normal(0, 3.0, N // 2),
    ]
)

# Per-point RGBA colors
colors = np.random.rand(N, 4).astype(np.float32)
colors[:, 3] = 0.1  # Very transparent for blending

# Figure with multi-sampling (AA) and HUD
gplt.figure(
    "Optimized Scatter (5M Points)",
    width=1280,
    height=800,
    multisample=True,
    hud=True,
    lod=True,
    budget=100,
)

gplt.scatter(x, y, color=colors, size=2.0)

gplt.text(-4, 4, "5,000,000 Points", fontsize=28, color="black")
gplt.text(-4, 3.5, "SDF Mathematical Primitives", fontsize=18, color="blue")

print("\nNotice how panning remains locked at 60 FPS.")
print("The Hybrid Cache (on by default) allows fluid movement through millions of points.")
gplt.show()
