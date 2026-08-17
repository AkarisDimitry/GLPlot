"""Expanding density cloud -- a point-source explosion, shown as a live 2D histogram.

The scenario: a compact charge detonates at the origin at ``t = 0`` and throws off a
cloud of debris. Each fragment carries its own fixed ejection angle and speed (an
isotropic burst with a spread of muzzle velocities, like real shrapnel), so on its own
the ballistic term alone would paint a thin, expanding *shell* -- ``radius = speed * t``.
Layered on top is ordinary diffusive broadening: the fragments jostle through the
surrounding medium and pick up a random-walk displacement whose standard deviation grows
as ``sqrt(2 * D * t)``, the textbook solution of the 2D diffusion equation for a point
source. Because each particle's diffusive kick is a *fixed* per-particle Gaussian scaled
analytically by ``sqrt(t)`` (rather than an explicit step-by-step random walk), the whole
cloud's trajectory is one vectorised numpy expression per frame -- no per-frame state to
carry forward, no re-seeding, just "evaluate the closed-form position at time t".

The result is a fuzzy, expanding ring that blooms outward and thins as it grows -- more
"blast wave" than "static blob" -- rendered every frame as a fresh 2D histogram (moderate
grid, not literal per-particle markers), so a third of a million fragments costs no more
than the grid resolution does.
"""

from __future__ import annotations

import numpy as np

import glplot.animation as animation
import glplot.pyplot as plt

rng = np.random.default_rng(7)

N = 300_000  # fragments in the debris cloud
BINS = 160  # histogram grid resolution per axis
FRAMES = 72
FPS = 20

# --- Per-fragment constants, drawn once (a real explosion's fragments keep their own
# ejection angle and speed for the whole flight) ---------------------------------------
theta = rng.uniform(0.0, 2.0 * np.pi, N)  # ejection angle, isotropic burst
speed = rng.rayleigh(scale=6.0, size=N) + 1.5  # mm/s, spread of muzzle velocities
vx, vy = speed * np.cos(theta), speed * np.sin(theta)
zx, zy = rng.standard_normal(N), rng.standard_normal(N)  # fixed diffusive draw per fragment

D = 6.0  # mm^2/s, effective diffusion constant of the surrounding medium
T_MAX = 2.6  # s, total elapsed time shown
t_values = np.linspace(0.05, T_MAX, FRAMES)  # start just after t=0 (a point has no spread)

# Frozen at the bloom's own final extent (computed once, from the state at T_MAX) rather
# than re-fit to each frame's own current spread -- a per-frame span made the view itself
# keep zooming out as the cloud grew, which read as the *camera* pulling back rather than
# the explosion actually expanding. Early frames now show a small young cloud inside a
# frame already sized for the fully-grown one, which is the point: a fixed window you can
# watch the blast fill.
_spread_max = np.sqrt(2.0 * D * T_MAX)
_r97_max = np.percentile(np.hypot(vx * T_MAX + _spread_max * zx, vy * T_MAX + _spread_max * zy), 97.0)
SPAN = max(4.0, 1.15 * _r97_max)

fig = plt.figure("Gallery - Expanding Density Cloud", figsize=(7.6, 6.4))
plt.plot_style("dark")
plt.xlim(-SPAN, SPAN)
plt.ylim(-SPAN, SPAN)


def update(frame: int):
    t = t_values[frame]
    spread = np.sqrt(2.0 * D * t)
    x = vx * t + spread * zx
    y = vy * t + spread * zy

    counts, xedges, yedges = np.histogram2d(
        x, y, bins=BINS, range=[[-SPAN, SPAN], [-SPAN, SPAN]]
    )

    plt.cla()
    plt.imshow(
        np.log1p(counts.T),
        extent=(-SPAN, SPAN, -SPAN, SPAN),
        origin="lower",
        cmap="inferno",
        aspect="equal",
    )
    plt.xlim(-SPAN, SPAN)
    plt.ylim(-SPAN, SPAN)
    plt.title(f"Point-source explosion: t = {t:5.2f} s, N = {N:,} fragments")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    return []


ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=1000 // FPS)
# plt.show()
ani.save("examples/gallery/animations/results/06_expanding_density_cloud.gif", fps=FPS)
