import numpy as np
from matplotlib import colormaps

import glplot.pyplot as plt

# Synthetic surface-height deviations from a white-light interferometer scanning a
# nominally flat optical reference. A broad measurement-noise floor sits under three
# sharper features: a cluster of shallow polishing scratches, a coating-ridge bump,
# and a handful of discrete micro-pit defects -- the kind of multimodal histogram a
# metrology bench actually produces.
rng = np.random.default_rng(7)
noise_floor = rng.normal(loc=0.0, scale=0.58, size=520_000)
polishing_scratches = rng.normal(loc=-2.25, scale=0.38, size=260_000)
coating_ridge = rng.gamma(shape=2.4, scale=0.58, size=320_000) + 0.45
micro_pits = rng.normal(loc=rng.choice([-1.15, 1.65, 2.85], size=160_000), scale=0.16)
deviations = np.r_[noise_floor, polishing_scratches, coating_ridge, micro_pits]
n_samples = deviations.size

counts, edges = np.histogram(deviations, bins=180, range=(-3.8, 4.3))
centers = 0.5 * (edges[:-1] + edges[1:])
width = float(np.diff(edges)[0])

cmap = colormaps["magma"]
shade = 0.12 + 0.80 * (counts / counts.max())

plt.figure("Gallery - Surface Deviation Histogram", figsize=(9, 5))
for xc, h, t in zip(centers, counts, shade):
    plt.bar([xc], [h], width=width * 0.98, color=cmap(t))

plt.title(f"Surface-height deviation histogram (N = {n_samples:,} interferometer samples)")
plt.xlabel("Height deviation (µm)")
plt.ylabel("Count")
plt.grid(True)
# plt.show()
plt.savefig("examples/gallery/results/03_bar_hist.png")
