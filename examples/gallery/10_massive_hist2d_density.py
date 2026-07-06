import numpy as np

import glplot.pyplot as plt

rng = np.random.default_rng(123)
n = 1_000_000
theta = rng.uniform(0, 2 * np.pi, n)
arms = rng.integers(0, 5, n)
r = rng.gamma(shape=2.2, scale=0.8, size=n)
twist = theta + arms * 2 * np.pi / 5 + 0.9 * r
x = r * np.cos(twist) + rng.normal(scale=0.08, size=n)
y = r * np.sin(twist) + rng.normal(scale=0.08, size=n)

plt.figure("Gallery - Massive 2D Density", figsize=(8, 6))
plt.hist2d(x, y, bins=360, cmap="inferno", s=2.0, label="1M samples")
plt.title("1,000,000-point density histogram")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
# plt.show()
plt.savefig("examples/gallery/results/10_massive_hist2d_density.png")
