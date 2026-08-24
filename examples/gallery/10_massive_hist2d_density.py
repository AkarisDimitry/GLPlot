import numpy as np

import glplot.pyplot as plt

# Simulated positions where beam particles strike a downstream tracking
# detector after multiple Coulomb scattering through a target: a dense
# central spot from the common small-angle deflections, surrounded by a
# diffuse halo from the rarer, large-angle single-scattering events.
rng = np.random.default_rng(123)
n = 10_000_000
theta = rng.uniform(0, 2 * np.pi, n)
r = rng.gamma(shape=2.2, scale=0.8, size=n)  # scattering-angle-like radius
mm_per_unit = 10.0  # detector spans roughly +/-100 mm
x = mm_per_unit * (r * np.cos(theta) + rng.normal(scale=0.08, size=n))
y = mm_per_unit * (r * np.sin(theta) + rng.normal(scale=0.08, size=n))

plt.figure("Gallery - Detector Impact Density", figsize=(8, 6))
# Bin over a tight +/-45 mm window rather than the raw data extent (rare
# large-angle outliers stretch that to +/-150 mm): this is the "zoom in" the
# populated core needs, and it keeps every displayed cell backed by enough of
# the 10M hits to read as a continuous density field instead of a halo of
# isolated single-count cells. hist2d() draws a real filled rectangular mesh
# (one tile per bin, sized exactly to the bin pitch), so unlike the old
# point-marker rendering there is no separate size knob to tune for gap-free
# tiling -- it tiles solidly by construction at any bin count.
lim = 45.0
plt.hist2d(
    x,
    y,
    bins=170,
    range=[[-lim, lim], [-lim, lim]],
    cmap="plasma",
    label=f"{n:,} hits",
)
plt.title(f"{n:,}-hit density map on a detector plane after multiple Coulomb scattering")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.legend()
# plt.show()
plt.savefig("examples/gallery/results/10_massive_hist2d_density.png")
