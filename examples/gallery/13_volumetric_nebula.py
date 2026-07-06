import numpy as np

import glplot.pyplot as plt

rng = np.random.default_rng(1313)
n = 1_750_000
theta = rng.uniform(0, 8 * np.pi, n)
phi = rng.normal(0.0, 0.48, n)
r = rng.gamma(2.0, 0.8, n)
arm = rng.integers(0, 4, n) * np.pi / 2
twist = theta + arm + 0.62 * r
x = r * np.cos(twist) + rng.normal(scale=0.06, size=n)
y = r * np.sin(twist) + rng.normal(scale=0.06, size=n)
z = 0.65 * r * np.sin(phi) + 0.18 * np.sin(theta)
energy = np.exp(-0.12 * r**2) + 0.28 * np.sin(3 * theta) ** 2

plt.figure("Gallery - Volumetric 3D Nebula", figsize=(9, 6))
plt.volume3d(
    x,
    y,
    z,
    energy,
    threshold=0.14,
    cmap="magma",
    alpha=0.42,
    s=1.2,
    elev=24,
    azim=-52,
    label="volumetric samples",
)
plt.scatter3d(
    x[::450],
    y[::450],
    z[::450],
    c=energy[::450],
    cmap="turbo",
    s=2,
    elev=24,
    azim=-52,
    label="bright probes",
)
plt.title("750k-sample volumetric nebula")
plt.xlabel("x")
plt.ylabel("y")
plt.zlabel("z")
plt.legend()
# plt.show()
plt.savefig("examples/gallery/results/13_volumetric_nebula.png")
