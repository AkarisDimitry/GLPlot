"""Orbiting star cluster -- differential rotation winds a spiral galaxy's arms.

A textbook puzzle in galactic dynamics: if a spiral galaxy's disc rotated as a
single rigid body, its arms would just spin in place, never changing shape. Real
discs do not rotate rigidly -- they rotate *differentially*, with the angular
speed of an orbit falling off with radius roughly the way Kepler's third law
predicts for a point mass, ``omega(r) ~ 1 / sqrt(r)``. Stars near the core sweep
around many times faster than stars out in the halo.

Set a few thousand stars down along loose spiral arms and let each one orbit at
its own Keplerian rate and the arms visibly wind up tighter and tighter, lap by
lap -- exactly the "winding problem" that made astronomers realise persistent
spiral arms can't be the same clump of stars going around together forever, and
must instead be a slower-moving density wave that stars merely pass through.
This animation is the naive picture that motivated that idea: fixed stars,
differential rotation, watched over many orbits.

Every star's ``(x, y)`` is recomputed from its own orbital radius and its own
angular speed each frame -- ``theta(t) = theta_0 + omega(r) * t`` -- so nothing
here is a rigid-body rotation of a fixed image; each point genuinely advances
along its own circle at its own rate.
"""

from __future__ import annotations

import numpy as np

import glplot.animation as animation
import glplot.pyplot as plt

N_STARS = 20_000
N_ARMS = 3
FRAMES = 80
LIM = 6.8  # kpc, half-width of the fixed view window

rng = np.random.default_rng(42)

# Radius: an exponential disc profile (dense core, thinning halo), softened away from
# r = 0 so no star sits exactly on the (undefined) rotation axis.
radius = rng.exponential(scale=1.35, size=N_STARS) + 0.10
radius = np.clip(radius, 0.10, LIM - 0.3)

# Assign each star to one of N_ARMS logarithmic spiral arms, then scatter it around the
# arm's centreline -- tighter scatter near the core, looser out in the halo, which is
# what a real disc's arm width looks like.
arm_id = rng.integers(0, N_ARMS, N_STARS)
arm_offset = arm_id * (2.0 * np.pi / N_ARMS)
pitch = 1.65  # spiral pitch: how many radians of wind-up per e-fold of radius
theta0 = arm_offset + pitch * np.log(radius / 0.10)
theta0 += rng.normal(scale=0.10 + 0.05 * radius, size=N_STARS)
radius = radius * (1.0 + rng.normal(scale=0.035, size=N_STARS))
radius = np.clip(radius, 0.05, LIM - 0.2)

# Keplerian-like differential rotation: omega ~ 1/sqrt(r), softened near the core so the
# innermost stars spin fast but finite rather than blowing up.
OMEGA0 = 1.35
omega = OMEGA0 / np.sqrt(radius + 0.30)

# Colour by orbital radius -- a vivid, continuous readout of "how far out is this star"
# that a distant-galaxy photo can't give you directly.
color_by = radius

fig = plt.figure("Gallery - Orbiting Star Cluster", figsize=(7.6, 7.6))
plt.plot_style("neon")


def update(frame: int):
    t = frame * 0.11  # orbital-time units advanced per frame
    theta = theta0 + omega * t
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    plt.cla()
    plt.scatter(x, y, c=color_by, cmap="plasma", s=3.2, alpha=0.85)
    plt.set_aspect("equal")
    plt.xlim(-LIM, LIM)
    plt.ylim(-LIM, LIM)
    plt.title(f"Differential rotation winds the arms (N={N_STARS:,}, t={t:4.1f})")
    plt.xlabel("x (kpc)")
    plt.ylabel("y (kpc)")
    return []


ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=42)
# plt.show()
ani.save("examples/gallery/animations/results/01_orbiting_star_cluster.gif", fps=20)
