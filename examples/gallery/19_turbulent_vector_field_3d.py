import numpy as np
import os
import glplot.pyplot as plt

rng = np.random.default_rng(1919)
elev = 27
azim = -49

# Dense Lorenz-like vector field sampled on a 3D lattice.
grid = np.linspace(-2.4, 2.4, 29)
x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
sigma = 5.4
rho = 10.8
beta = 1.75
u = sigma * (y - x) + 0.45 * np.sin(2.2 * z)
v = x * (rho - z * 2.6) - y + 0.55 * np.cos(1.9 * x)
w = x * y - beta * z + 0.35 * np.sin(2.7 * y)
speed = np.sqrt(u**2 + v**2 + w**2)

# Keep the strongest vectors plus a deterministic sample of mid-energy vectors
# so the field reads as structured instead of becoming a solid block.
flat_speed = speed.ravel()
strong = flat_speed > np.quantile(flat_speed, 0.77)
mid = (flat_speed > np.quantile(flat_speed, 0.54)) & (rng.random(flat_speed.size) < 0.18)
mask = strong | mid

# Massive glowing particle cloud following the same swirl family.
cloud_n = 560_000
t = rng.uniform(0.0, 18.0 * np.pi, cloud_n)
shell = rng.gamma(2.2, 0.42, cloud_n)
twist = 0.22 * np.sin(2.6 * t) + 0.18 * np.cos(1.4 * t)
cloud_x = shell * np.sin(t) + 0.28 * np.sin(3.0 * t)
cloud_y = shell * np.cos(1.17 * t) + 0.32 * np.cos(2.4 * t)
cloud_z = 0.58 * np.sin(0.52 * t + twist) + 0.32 * shell * np.cos(0.33 * t)
cloud_energy = np.exp(-0.22 * shell**2) + 0.33 * np.sin(1.7 * t) ** 2

plt.figure("Gallery - Turbulent 3D Vector Field", figsize=(10, 7), ssao=True)
plt.volume3d(
    cloud_x,
    cloud_y,
    cloud_z,
    cloud_energy,
    threshold=0.24,
    cmap="turbo",
    alpha=0.18,
    s=0.75,
    elev=elev,
    azim=azim,
    label="560k volumetric samples",
)
plt.quiver3d(
    x.ravel()[mask],
    y.ravel()[mask],
    z.ravel()[mask],
    u.ravel()[mask],
    v.ravel()[mask],
    w.ravel()[mask],
    scale=0.16,
    normalize=True,
    color=(0.08, 0.95, 1.0, 0.86),
    linewidth=0.62,
    head_length=0.20,
    head_width=0.08,
    elev=elev,
    azim=azim,
    label="adaptive 3D vector lattice",
)

trace_t = np.linspace(0.0, 9.0 * np.pi, 1_800)
for i, phase in enumerate(np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)):
    radius = 0.35 + 0.035 * i + 0.08 * np.sin(0.8 * trace_t + phase)
    tx = radius * trace_t / 3.5 * np.sin(0.72 * trace_t + phase)
    ty = radius * trace_t / 3.7 * np.cos(0.66 * trace_t + phase)
    tz = 1.25 * np.sin(0.35 * trace_t + phase) + 0.12 * np.cos(1.8 * trace_t)
    plt.plot3d(
        tx,
        ty,
        tz,
        color=(1.0, 0.88 - i * 0.035, 0.18 + i * 0.045, 0.72),
        linewidth=1.05,
        elev=elev,
        azim=azim,
        scale_z=1.0,
        label="stream traces" if i == 0 else None,
    )

plt.scatter3d(
    [0.0],
    [0.0],
    [0.0],
    color="white",
    s=45,
    elev=elev,
    azim=azim,
    label="field core",
)
plt.title("Massive turbulent 3D vector field")
plt.xlabel("flow x")
plt.ylabel("flow y")
plt.zlabel("flow z")
plt.legend()

# plt.show()
plt.savefig("examples/gallery/results/19_turbulent_vector_field_3d.png")
