"""Animated 2D vector field -- tracer particles genuinely advected through a flow.

Synthetic PIV (particle-image-velocimetry) recording of a benchtop drain-vortex tank: two
co-rotating stirring vortices are prescribed to orbit a common center above a weak central
drain, the classic setup used in laboratory studies of 2D vortex pairing/merging and of the
"bathtub vortex" that also shows up at oceanic-eddy scale. ~4,500 neutrally buoyant tracer
beads are seeded through the fluid and *advected* frame to frame -- their positions are
stepped through the analytic velocity field with 4th-order Runge-Kutta integration, not
replotted at random -- so the vortex cores, the saddle between them, and the converging
spiral into the drain all become visible purely through where the beads cluster and swirl.
Beads that spiral into the drain core (or drift past the tank rim) are recycled back to the
rim, the way a real PIV run keeps reseeding tracer density over a long recording.

A coarse 18x18 quiver grid overlays the same instantaneous field so the arrow directions and
the particle motion can be checked against each other directly.
"""

from __future__ import annotations

import numpy as np

import glplot.animation as animation
import glplot.pyplot as plt

FRAMES = 72
N_PARTICLES = 4500
DOMAIN_R = 0.34  # tank radius, meters
RIM_LO, RIM_HI = 0.26, 0.32  # respawn annulus near the rim
DRAIN_R = 0.018  # beads closer than this to the center are "drained" and recycled

R_ORBIT = 0.11  # radius the two vortex cores orbit the tank center at
CORE = 0.045  # Lamb-Oseen vortex core radius (solid-body rotation inside this)
GAMMA = 0.018  # circulation strength of each stirring vortex, m^2/s
SINK_STRENGTH = 6.0e-4  # weak central drain, m^3/s
SINK_CORE = 0.02

DT_SUB = 0.02  # seconds per Runge-Kutta sub-step
N_SUBSTEPS = 4  # sub-steps per animation frame -> smoother, more stable advection
_TOTAL_T = FRAMES * DT_SUB * N_SUBSTEPS  # total simulated seconds spanned by the clip
OMEGA_ORBIT = 2 * np.pi * 1.6 / _TOTAL_T  # the vortex pair completes 1.6 laps over the clip


def velocity(px: np.ndarray, py: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Velocity field (u, v) at (px, py), time t: two orbiting Lamb-Oseen vortices + a drain.

    Each vortex core is regularized (the ``1 - exp(-r^2/core^2)`` factor) so the induced
    speed does not blow up at the center -- solid-body rotation inside the core, the usual
    1/r swirl outside it, exactly like a real stirred vortex measured by PIV. The two cores
    are prescribed to orbit the tank center at a fixed radius, approximating the slow mutual
    rotation a same-sign vortex pair undergoes as it heads toward merging. A weak radial
    sink pulls fluid steadily toward the drain, regularized the same way near r=0.
    """
    theta = OMEGA_ORBIT * t
    centers = (
        (R_ORBIT * np.cos(theta), R_ORBIT * np.sin(theta)),
        (-R_ORBIT * np.cos(theta), -R_ORBIT * np.sin(theta)),
    )
    u = np.zeros_like(px)
    v = np.zeros_like(py)
    for cx, cy in centers:
        dx = px - cx
        dy = py - cy
        r2 = dx * dx + dy * dy
        factor = GAMMA / (2 * np.pi) * (1.0 - np.exp(-r2 / CORE**2)) / (r2 + 1e-8)
        u += -factor * dy
        v += factor * dx

    r2c = px * px + py * py
    sink = -SINK_STRENGTH / (r2c + SINK_CORE**2)
    u += sink * px
    v += sink * py
    return u, v


def rk4_step(px: np.ndarray, py: np.ndarray, t: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """One classic 4th-order Runge-Kutta step of dp/dt = velocity(p, t), field frozen over dt."""
    k1u, k1v = velocity(px, py, t)
    k2u, k2v = velocity(px + 0.5 * dt * k1u, py + 0.5 * dt * k1v, t)
    k3u, k3v = velocity(px + 0.5 * dt * k2u, py + 0.5 * dt * k2v, t)
    k4u, k4v = velocity(px + dt * k3u, py + dt * k3v, t)
    px_new = px + (dt / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
    py_new = py + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return px_new, py_new


# --- seed the tracer cloud once; update() only ever advects it from here on ---
rng = np.random.default_rng(11)
_seed_angle = rng.uniform(0.0, 2 * np.pi, N_PARTICLES)
_seed_radius = rng.uniform(0.04, DOMAIN_R * 0.97, N_PARTICLES)
px = _seed_radius * np.cos(_seed_angle)
py = _seed_radius * np.sin(_seed_angle)

# Coarse arrow grid, sampled fresh each frame from the same analytic field.
_grid = np.linspace(-DOMAIN_R * 0.92, DOMAIN_R * 0.92, 18)
QX, QY = np.meshgrid(_grid, _grid)
QX_FLAT, QY_FLAT = QX.ravel(), QY.ravel()

# Fixed color-scale range so speed colouring stays consistent frame to frame: sample the
# field across the whole orbit up front rather than recomputing vmin/vmax every frame (which
# would make the colormap flicker as the vortices sweep past a sample point).
_probe_t = np.linspace(0.0, _TOTAL_T, 24)
_speed_samples = []
for _t in _probe_t:
    _u, _v = velocity(px, py, float(_t))
    _speed_samples.append(np.hypot(_u, _v))
SPEED_VMIN = 0.0
SPEED_VMAX = float(np.percentile(np.concatenate(_speed_samples), 99.0))

fig = plt.figure("Gallery - Advected Flow Particles", figsize=(7.6, 6.6))
plt.plot_style("dark")


def update(frame: int):
    global px, py

    t = frame * DT_SUB * N_SUBSTEPS
    for _ in range(N_SUBSTEPS):
        px, py = rk4_step(px, py, t, DT_SUB)
        t += DT_SUB

    # Recycle beads that drained into the core or drifted past the rim, back onto the rim --
    # keeps tracer density roughly constant, the way a real PIV run keeps reseeding.
    r = np.hypot(px, py)
    lost = (r < DRAIN_R) | (r > DOMAIN_R)
    n_lost = int(lost.sum())
    if n_lost:
        ang = rng.uniform(0.0, 2 * np.pi, n_lost)
        rad = rng.uniform(RIM_LO, RIM_HI, n_lost)
        px[lost] = rad * np.cos(ang)
        py[lost] = rad * np.sin(ang)

    u, v = velocity(px, py, t)
    speed = np.hypot(u, v)

    qu, qv = velocity(QX_FLAT, QY_FLAT, t)

    plt.cla()
    plt.scatter(
        px,
        py,
        c=speed,
        cmap="plasma",
        vmin=SPEED_VMIN,
        vmax=SPEED_VMAX,
        s=5.5,
        alpha=0.85,
    )
    plt.quiver(
        QX_FLAT,
        QY_FLAT,
        qu,
        qv,
        color=(1.0, 1.0, 1.0, 0.5),
        scale=1.3,
        width=1.0,
        head_width=0.35,
        head_length=0.45,
    )
    plt.xlim(-DOMAIN_R * 1.05, DOMAIN_R * 1.05)
    plt.ylim(-DOMAIN_R * 1.05, DOMAIN_R * 1.05)
    plt.title(f"Drain-vortex tank -- {n_lost} beads recycled this frame")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    return []


ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=42)
# plt.show()
ani.save("examples/gallery/animations/results/05_advected_flow_particles.gif", fps=20)
