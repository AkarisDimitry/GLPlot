import os

import numpy as np

import glplot.pyplot as plt

rng = np.random.default_rng(21)
n = 250_000
cluster = rng.integers(0, 4, size=n)
centers = np.array([[-2.4, -1.0], [-0.8, 1.6], [1.4, -1.4], [2.2, 1.1]], dtype=float)
spread = np.array([[0.45, 0.18], [0.25, 0.55], [0.65, 0.28], [0.38, 0.42]])
x = centers[cluster, 0] + rng.normal(scale=spread[cluster, 0])
y = centers[cluster, 1] + 0.55 * x + rng.normal(scale=spread[cluster, 1])
score = np.sin(2.4 * x) + np.cos(3.1 * y) + cluster

plt.figure("Gallery - Guides and Colormap", figsize=(9, 6))
plt.scatter(x, y, c=score, cmap="plasma", s=1.4, alpha=0.72, label="250k clustered samples")
plt.axhline(0, color="k", linestyle="--", linewidth=1)
plt.axvline(0, color="k", linestyle="--", linewidth=1)
plt.axline((0, 0), slope=0.55, color="cyan", linewidth=2, label="decision ridge")
plt.annotate(
    "dense branch",
    xy=(1.8, 0.0),
    xytext=(-2.9, 2.8),
    arrowprops={"color": "cyan", "width": 1.2},
    color="cyan",
)
plt.xlabel("x")
plt.ylabel("y")
plt.title("250k points with guides, colormap, and annotation")
plt.grid(True)
plt.legend()
plt.show()
plt.savefig("examples/gallery/results/05_guides_and_colormap.png")
