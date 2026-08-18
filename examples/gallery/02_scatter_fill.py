import numpy as np

import glplot.pyplot as plt

rng = np.random.default_rng(4)
n = 1_220_000
theta = rng.uniform(0, 18 * np.pi, n)
radius = 0.025 * theta + rng.gamma(1.5, 0.006, n)
x_pc = radius * np.cos(theta) + rng.normal(scale=0.012, size=n)
y_pc = radius * np.sin(theta) + rng.normal(scale=0.012, size=n)
formation_phase = theta + 6.0 * radius  # proxy for age along the spiral density wave

plt.figure("Gallery - Simulated Spiral Star Field", figsize=(9, 6))
plt.scatter(x_pc, y_pc, c=formation_phase, cmap="turbo", s=1.5, alpha=0.72)
plt.title(f"Simulated stellar density along a spiral arm (N = {n:,} stars)")
plt.xlabel("x (pc)")
plt.ylabel("y (pc)")
# plt.show()
plt.savefig("examples/gallery/results/02_scatter_fill.png")
