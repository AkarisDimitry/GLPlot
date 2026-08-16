import numpy as np
from matplotlib import colormaps

import glplot.pyplot as plt

# Synthetic PIV-style measurement: a single vortex core in a microchannel,
# with a mild wavy perturbation superimposed on the rotational flow.
grid = np.linspace(-3.0, 3.0, 43)
x, y = np.meshgrid(grid, grid)


def velocity(px, py):
    r2 = px**2 + py**2 + 0.2
    uu = -py / r2 + 0.15 * np.sin(3 * py)
    vv = px / r2 + 0.15 * np.cos(3 * px)
    return uu, vv


u, v = velocity(x, y)
speed = np.hypot(u, v)

x_flat, y_flat = x.ravel(), y.ravel()
u_flat, v_flat = u.ravel(), v.ravel()
speed_flat = speed.ravel()

plt.figure("Gallery - Vector Field", figsize=(8, 6))

# Seed a cloud of tracer particles and advect them through the same analytic
# flow field the arrows sample (RK4 integration), plotting every sub-step so
# their curved streaks read like smoke caught mid-swirl in the vortex. This
# runs -- and is drawn -- before the quiver loop, so the particle haze sits
# underneath the arrows rather than competing with them.
rng = np.random.default_rng(7)
n_particles = 3000
px = rng.uniform(-3.0, 3.0, n_particles)
py = rng.uniform(-3.0, 3.0, n_particles)

dt = 0.05
n_steps = 7
vmin, vmax = float(speed_flat.min()), float(speed_flat.max())
n_particle_points = 0
for step in range(n_steps):
    k1u, k1v = velocity(px, py)
    k2u, k2v = velocity(px + 0.5 * dt * k1u, py + 0.5 * dt * k1v)
    k3u, k3v = velocity(px + 0.5 * dt * k2u, py + 0.5 * dt * k2v)
    k4u, k4v = velocity(px + dt * k3u, py + dt * k3v)
    px = px + (dt / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
    py = py + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    local_speed = np.hypot(k1u, k1v)
    fade = 0.12 + 0.38 * (step + 1) / n_steps
    plt.scatter(
        px,
        py,
        c=local_speed,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=3.0,
        alpha=fade,
    )
    n_particle_points += n_particles

# Colour-code every arrow by its local speed, using quantile bins from a
# single colormap -- the bins double as a discrete colorbar in the legend.
n_bins = 5
edges = np.quantile(speed_flat, np.linspace(0, 1, n_bins + 1))
cmap = colormaps["viridis"]
for i in range(n_bins):
    lo, hi = edges[i], edges[i + 1]
    sel = (
        (speed_flat >= lo) & (speed_flat <= hi)
        if i == n_bins - 1
        else ((speed_flat >= lo) & (speed_flat < hi))
    )
    if not np.any(sel):
        continue
    color = cmap((i + 0.5) / n_bins)
    plt.quiver(
        x_flat[sel],
        y_flat[sel],
        u_flat[sel],
        v_flat[sel],
        color=color,
        scale=0.42,
        width=0.8,
        head_width=0.06,
        head_length=0.095,
        label=f"{lo:.2f}-{hi:.2f} m/s",
    )

plt.title(
    f"Vortex flow field, N={x_flat.size:,} vectors + {n_particle_points:,} advected tracer particles"
)
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.legend()
# plt.show()
plt.savefig("examples/gallery/results/08_vector_field_quiver.png")
