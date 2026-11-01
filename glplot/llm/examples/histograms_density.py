"""Histograms and 2-D density.

hist for one variable; hist2d / hexbin turn a point cloud into a density map -- the right tool
once a scatter would just be a blob.
"""

import numpy as np

import glplot.pyplot as plt

rng = np.random.default_rng(4)
x = np.concatenate([rng.normal(-1, 0.6, 50_000), rng.normal(1.5, 0.9, 50_000)])
y = 0.5 * x + rng.normal(scale=0.7, size=x.size)

fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
axs[0].hist(x, bins=80)
axs[0].set_title("histogram")
axs[0].set_xlabel("x")
axs[1].hist2d(x, y, bins=90, cmap="magma")
axs[1].set_title("hist2d (100k points)")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
plt.savefig("histograms_density.png")
