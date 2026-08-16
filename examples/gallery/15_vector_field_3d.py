import numpy as np

import glplot.pyplot as plt

# Synthetic turbulent-jet rig: a swirling velocity field measured on a 3D
# lattice inside a wind-tunnel test section, with a faint aerosol tracer
# cloud (smoke visualization) drifting through the same volume as backdrop.
rng = np.random.default_rng(1515)

grid = np.linspace(-3.0, 3.0, 24)
x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
r2 = x**2 + y**2 + z**2 + 0.35
u = -y / r2 + 0.18 * np.sin(2.2 * z)
v = x / r2 + 0.18 * np.cos(2.0 * z)
w = 0.35 * np.sin(1.4 * x) * np.cos(1.4 * y) - 0.12 * z
speed = np.sqrt(u**2 + v**2 + w**2)

cloud_n = 420_000
# The raw gamma-distributed radius has a long tail reaching past the +-3 test
# section the velocity grid lives in, which used to stretch the shared axis
# autoscale far past the arrows and leave them a small tuft in a big empty
# box. Capping the tracer cloud near the grid's own extent keeps both layers
# filling the same tight axis box.
outer_radius = 3.4  # cm, roughly the wind-tunnel test-section half-width
n_gen = 460_000  # oversample so the radius cut still leaves cloud_n points
theta = rng.uniform(0, 10 * np.pi, n_gen)
radius = rng.gamma(2.0, 0.72, n_gen)
inside = radius < outer_radius
theta, radius = theta[inside][:cloud_n], radius[inside][:cloud_n]
cloud_x = radius * np.cos(theta)
cloud_y = radius * np.sin(theta)
cloud_z = 0.28 * radius * np.sin(theta * 0.7) + rng.normal(scale=0.18, size=cloud_n)
cloud_density = np.exp(-0.11 * radius**2) + 0.2 * np.sin(theta) ** 2

mask = speed.ravel() > np.quantile(speed, 0.62)
n_arrows = int(mask.sum())

plt.figure("Gallery - 3D Vector Field", figsize=(9, 6))
plt.volume3d(
    cloud_x,
    cloud_y,
    cloud_z,
    cloud_density,
    threshold=0.16,
    cmap="magma",
    alpha=0.16,
    s=0.9,
    elev=29,
    azim=-43,
    label="aerosol tracer cloud (420k samples)",
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
    color=(0.15, 0.95, 1.0, 0.92),
    linewidth=0.85,
    head_length=0.16,
    head_width=0.07,
    elev=29,
    azim=-43,
    label=f"velocity field ({n_arrows:,} vectors)",
)
plt.title(f"Swirling jet velocity field, N={n_arrows:,} vectors over tracer cloud")
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")
plt.zlabel("z (cm)")
plt.legend()
# plt.show()
plt.savefig("examples/gallery/results/15_vector_field_3d.png")
