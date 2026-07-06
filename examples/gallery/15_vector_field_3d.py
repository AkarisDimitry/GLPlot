import numpy as np

import glplot.pyplot as plt

rng = np.random.default_rng(1515)

grid = np.linspace(-3.0, 3.0, 21)
x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
r2 = x**2 + y**2 + z**2 + 0.35
u = -y / r2 + 0.18 * np.sin(2.2 * z)
v = x / r2 + 0.18 * np.cos(2.0 * z)
w = 0.35 * np.sin(1.4 * x) * np.cos(1.4 * y) - 0.12 * z
speed = np.sqrt(u**2 + v**2 + w**2)

cloud_n = 420_000
theta = rng.uniform(0, 10 * np.pi, cloud_n)
radius = rng.gamma(2.0, 0.72, cloud_n)
cloud_x = radius * np.cos(theta)
cloud_y = radius * np.sin(theta)
cloud_z = 0.28 * radius * np.sin(theta * 0.7) + rng.normal(scale=0.18, size=cloud_n)
cloud_energy = np.exp(-0.11 * radius**2) + 0.2 * np.sin(theta) ** 2

mask = speed.ravel() > np.quantile(speed, 0.58)

plt.figure("Gallery - 3D Vector Field", figsize=(9, 6))
plt.volume3d(
    cloud_x,
    cloud_y,
    cloud_z,
    cloud_energy,
    threshold=0.16,
    cmap="magma",
    alpha=0.22,
    s=0.9,
    elev=29,
    azim=-43,
    label="420k flow samples",
)
plt.quiver3d(
    x.ravel()[mask],
    y.ravel()[mask],
    z.ravel()[mask],
    u.ravel()[mask],
    v.ravel()[mask],
    w.ravel()[mask],
    scale=0.55,
    normalize=True,
    color=(0.15, 0.9, 1.0, 0.82),
    linewidth=0.75,
    head_length=0.16,
    head_width=0.07,
    elev=29,
    azim=-43,
    label="3D vector field",
)
plt.scatter3d([0], [0], [0], color="white", s=35, elev=29, azim=-43, label="core")
plt.title("3D vector field over volumetric flow")
plt.xlabel("x")
plt.ylabel("y")
plt.zlabel("z")
plt.legend()
plt.show()
plt.savefig("examples/gallery/results/15_vector_field_3d.png")
