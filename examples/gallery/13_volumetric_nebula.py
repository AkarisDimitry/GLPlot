import numpy as np

import glplot.pyplot as plt

# Simulated emission nebula: gas and dust drifting in a four-armed spiral
# around a young star cluster, glowing brightest where density and radiation
# overlap. A single dense point cloud, shaded by local emission intensity --
# no secondary mark types layered on top.
#
# The raw gamma-distributed radius has a long tail of near-invisible stragglers
# that used to stretch the axis autoscale far past the actual spiral, leaving
# the bright core a small blob lost in an oversized box. Cutting the cloud off
# at a physical outer radius keeps the plotted nebula filling its axis box.
rng = np.random.default_rng(1313)
n = 1_750_000
outer_radius = 3.2  # ly, nebula edge
n_gen = 2_000_000  # oversample so the radius cut still leaves n points
theta = rng.uniform(0, 8 * np.pi, n_gen)
phi = rng.normal(0.0, 0.48, n_gen)
r = rng.gamma(2.0, 0.8, n_gen)
inside = r < outer_radius
theta, phi, r = theta[inside][:n], phi[inside][:n], r[inside][:n]
arm = rng.integers(0, 4, n) * np.pi / 2
twist = theta + arm + 0.62 * r
x = r * np.cos(twist) + rng.normal(scale=0.06, size=n)  # ly from cluster center
y = r * np.sin(twist) + rng.normal(scale=0.06, size=n)  # ly from cluster center
z = 0.65 * r * np.sin(phi) + 0.18 * np.sin(theta)  # ly from the disk midplane
emission = np.exp(-0.12 * r**2) + 0.28 * np.sin(3 * theta) ** 2

plt.figure("Gallery - Volumetric 3D Nebula", figsize=(9, 6))
plt.volume3d(
    x,
    y,
    z,
    emission,
    threshold=0.14,
    cmap="magma",
    alpha=0.42,
    s=1.2,
    elev=24,
    azim=-52,
    label=f"{n:,} points",
)
plt.title(f"{n:,}-point volumetric emission nebula, four-armed spiral structure")
plt.xlabel("x (ly)")
plt.ylabel("y (ly)")
plt.zlabel("z (ly)")
plt.legend()
# plt.show()
plt.savefig("examples/gallery/results/13_volumetric_nebula.png")
