"""Scatter coloured and sized by data, with a colorbar.

c= maps a value through a colormap, s= sets the marker size. For millions of points the same
call stays interactive in a live window; the headless export is matplotlib and slower.
"""

import numpy as np

import glplot.pyplot as plt

rng = np.random.default_rng(0)
n = 2000
x = rng.normal(size=n)
y = 0.6 * x + rng.normal(scale=0.5, size=n)
distance = np.hypot(x, y)

plt.figure("Cloud", figsize=(7, 6))
plt.scatter(x, y, c=distance, cmap="viridis", s=8, alpha=0.8)
plt.colorbar(label="distance from origin")
plt.title("Correlated cloud")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("scatter_colormap.png")
