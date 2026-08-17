"""Rippling 3D surface -- two water drops on a still pond, animated.

Physical picture: a shallow pool of water at rest is struck by a drop at its centre at
t=0, then by a second drop off to one side at t=1.4s. Each drop launches an outward
travelling *wave packet* -- a ring of ripples whose envelope spreads and advances at the
**group velocity**, while the individual crests inside the ring advance faster, at the
**phase velocity**. For deep-water gravity-capillary waves the two are related by the
classic dispersion result phase = 2 x group: crests are born at the trailing edge of the
ring, race forward through it, and vanish at the leading edge -- the same thing you see
looking down at real ripples spreading from a dropped pebble. Each ring also widens and
loses energy as it spreads over its own growing circumference (~1/sqrt(r)) and to viscous
damping (~exp(-decay*t)). Because the wave equation is linear at these small amplitudes,
overlapping rings from the two drops simply add -- producing a genuine interference
pattern where the two wakes cross.

    z(x, y, t) = sum over drops of
                     A0 * ring_envelope(r, tau) * spreading(r) * damping(tau)
                        * cos(k*r - omega*tau) ,     r = distance to that drop's centre
                                                          (softened by the drop's own
                                                          finite size so the centre is a
                                                          rounded splash, not a point cusp)

sampled on a fixed 80x80 grid in millimetres and redrawn every frame with
``plt.plot_surface()``. Height is coloured with the ``turbo`` colormap so crests and
troughs read at a glance, and the camera slowly orbits and tilts around the pool for
extra dimensionality -- a static viewpoint would hide how the ripple rings curl in 3D.

Performance note: a fresh 80x80 = 6,400-vertex surface mesh recomputed and re-triangulated
every frame is the expensive part of this animation (3D surface rendering costs much more
per frame than a 2D scatter/line of the same size), so the grid is kept moderate and the
frame count modest -- see the per-frame timing check below before the full run.
"""

import time

import numpy as np

import glplot.animation as animation
import glplot.pyplot as plt

# ---------------------------------------------------------------------------
# Grid: an 80x80 pool surface, x/y in millimetres.
# ---------------------------------------------------------------------------
GRID_N = 80
POOL_HALF_WIDTH = 40.0  # mm
axis = np.linspace(-POOL_HALF_WIDTH, POOL_HALF_WIDTH, GRID_N)
X, Y = np.meshgrid(axis, axis)

WAVELENGTH = 10.0  # mm, distance between successive crests
K = 2.0 * np.pi / WAVELENGTH  # spatial wavenumber
OMEGA = 14.0  # angular frequency of individual crests (rad / animation-second)
PHASE_SPEED = OMEGA / K  # how fast individual crests race outward (mm / anim-s)
GROUP_SPEED = 0.5 * PHASE_SPEED  # deep-water dispersion: the packet/front travels at half that
DECAY_TIME = 6.0  # amplitude e-folding time (anim-seconds) -- viscous damping
RING_WIDTH0 = 5.0  # mm, radial width of the wave packet the instant it is launched
RING_SPREAD = 1.4  # mm / anim-s, how fast that ring broadens as it travels (dispersion)
AMPLITUDE0 = 6.0  # mm, peak height right at each drop
IMPACT_RADIUS = 3.0  # mm, finite size of the drop itself -- rounds off the centre so a
# purely radial cos(k*r) does not sample as a sharp cone tip exactly on the impact point
RAMP_TIME = 0.35  # anim-s for a drop's splash to fade in from zero -- avoids a wave
# popping into existence at full strength on the very frame each drop lands

# Two drops: (x0 mm, y0 mm, launch time in anim-seconds).
DROPS = [(0.0, 0.0, 0.0), (17.0, -14.0, 1.4)]

FRAMES = 72
FPS = 20
DT = 1.0 / FPS  # animation-seconds elapsed per frame


def ripple_height(t: float) -> np.ndarray:
    """Superposed ripple field from every drop launched by animation-time ``t``."""
    z = np.zeros_like(X)
    for x0, y0, t0 in DROPS:
        tau = t - t0  # time since that drop landed
        if tau < 0:
            continue
        r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2 + IMPACT_RADIUS**2)
        front = GROUP_SPEED * tau  # where the ring's envelope currently sits
        width = RING_WIDTH0 + RING_SPREAD * tau  # the ring broadens as it disperses
        envelope = np.exp(-0.5 * ((r - front) / width) ** 2)  # a travelling Gaussian ring
        spreading = 1.0 / np.sqrt(1.0 + r / 5.0)  # energy spreads over a growing circumference
        damping = np.exp(-tau / DECAY_TIME)
        ramp = min(1.0, tau / RAMP_TIME)  # fade the splash in smoothly instead of a hard start
        amplitude = AMPLITUDE0 * envelope * spreading * damping * ramp
        z += amplitude * np.cos(K * r - OMEGA * tau)
    return z


fig = plt.figure("Gallery - Rippling 3D Surface", figsize=(8, 6.5))
plt.plot_style("ide")
# Pin the view volume: x/y already match the fixed 80x80 grid regardless of height, but
# z (ripple height) otherwise autoscales to each frame's own current amplitude -- as the
# splash damps out over DECAY_TIME, the z-axis kept shrinking to hug the shrinking
# ripple, making a calming pond visually read as the whole scene zooming in rather than
# the waves genuinely dying down. 10mm covers the peak two-drop constructive overlap
# with margin.
plt.set_xlim3d(-POOL_HALF_WIDTH, POOL_HALF_WIDTH)
plt.set_ylim3d(-POOL_HALF_WIDTH, POOL_HALF_WIDTH)
plt.set_zlim3d(-10.0, 10.0)

_orbit_start = -55.0
_orbit_span = 70.0  # degrees swept across the animation, eased back and forth


def update(frame):
    t = frame * DT
    z = ripple_height(t)

    # Slow camera orbit + gentle tilt: eases back and forth across the animation so the
    # viewpoint keeps moving without ever spinning past a confusing angle.
    swing = 0.5 - 0.5 * np.cos(2.0 * np.pi * frame / FRAMES)  # 0 -> 1 -> 0
    azim = _orbit_start + _orbit_span * swing
    elev = 26.0 + 8.0 * np.sin(2.0 * np.pi * frame / FRAMES)

    plt.cla()
    plt.plot_surface(
        X,
        Y,
        z,
        cmap="turbo",
        elev=elev,
        azim=azim,
        scale_z=1.0,
        rstride=1,
        cstride=1,
        vmin=-0.85 * AMPLITUDE0,
        vmax=0.85 * AMPLITUDE0,
        alpha=0.95,
    )
    n_active = sum(1 for _, _, t0 in DROPS if t >= t0)
    plt.title(f"Ripples on a still pond, t = {t:4.2f} s   ({n_active} drop(s) in play)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.zlabel("Height (mm)")
    return []


if __name__ == "__main__":
    # --- one-frame timing check before committing to the full render ---------------
    t0 = time.perf_counter()
    update(FRAMES // 3)
    elapsed = time.perf_counter() - t0
    print(f"[timing] one frame: {elapsed:.3f}s -> ~{elapsed * FRAMES:.1f}s for {FRAMES} frames")

    ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=1000.0 / FPS)
    # plt.show()
    ani.save("examples/gallery/animations/results/10_rippling_3d_surface.gif", fps=FPS)
