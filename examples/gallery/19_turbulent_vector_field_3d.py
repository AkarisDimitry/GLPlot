import os

import numpy as np

import glplot.pyplot as plt

rng = np.random.default_rng(1919)
elev = 27
azim = -49

# Dense Lorenz-like vector field sampled on a 3D lattice. Kept a touch
# smaller than the particle cloud/stream traces below so the glowing core
# sets the frame and the lattice reads as texture around it, not as a grid
# that tiles every corner of the box.
grid = np.linspace(-2.0, 2.0, 29)
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
strong = flat_speed > np.quantile(flat_speed, 0.76)
mid = (flat_speed > np.quantile(flat_speed, 0.50)) & (rng.random(flat_speed.size) < 0.20)
mask = strong | mid

# Massive glowing particle cloud following the same swirl family. The radial
# term is soft-clipped (tanh) so the rare long tail of a raw gamma draw can't
# balloon the auto-scaled axes into a mostly-empty cube -- every point stays
# within a bounded shell around the core, which is what keeps the field
# looking dense and "zoomed in" instead of a speck lost in a huge box.
cloud_n = 950_000
t = rng.uniform(0.0, 18.0 * np.pi, cloud_n)
shell_raw = rng.gamma(2.2, 0.42, cloud_n)
shell = 2.0 * np.tanh(shell_raw / 2.0)
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
    threshold=0.22,
    cmap="inferno",
    alpha=0.17,
    s=0.72,
    elev=elev,
    azim=azim,
    label="950k volumetric samples",
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
    color=(0.15, 0.92, 1.0, 0.65),
    linewidth=0.55,
    head_length=0.20,
    head_width=0.08,
    elev=elev,
    azim=azim,
    label="adaptive 3D vector lattice",
)

# Bounded braided stream traces that orbit close to the core (rather than
# spiraling outward) so they reinforce the dense center instead of stretching
# the auto-scaled axes far beyond it.
n_traces = 16
trace_t = np.linspace(0.0, 6.0 * np.pi, 2_000)
for i, phase in enumerate(np.linspace(0.0, 2.0 * np.pi, n_traces, endpoint=False)):
    frac = i / (n_traces - 1)
    radius = 1.35 + 0.55 * frac + 0.16 * np.sin(0.35 * trace_t + 1.7 * phase)
    tx = radius * np.sin(0.85 * trace_t + phase)
    ty = radius * np.cos(0.95 * trace_t + phase * 1.3)
    tz = 0.85 * np.sin(0.4 * trace_t + phase) + 0.25 * np.cos(1.6 * trace_t + phase)
    plt.plot3d(
        tx,
        ty,
        tz,
        color=(1.0, 0.85 - 0.45 * frac, 0.15 + 0.55 * frac, 0.78),
        linewidth=1.15,
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
    s=55,
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
