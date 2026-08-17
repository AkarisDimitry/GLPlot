"""Growing 3D bar city -- an audio-reactive skyline driven by two sweeping wave sources.

Picture a public-art installation: a city block's worth of rooftop actuators, each one a
small hydraulic bar under a building, wired to a pair of low-frequency acoustic emitters
mounted on moving gantries overhead. As each emitter sweeps across the block it radiates a
travelling pressure wave -- concentric rings expanding outward from wherever the emitter
currently sits, exactly the way a moving loudspeaker's wavefronts do,

    p(r, t) = A * cos(k*r - omega*t) * exp(-r / decay),   r = distance to the (moving) source

-- and every rooftop actuator drives its bar's height in real time to the *local* pressure
it feels, summed over both emitters. Two emitters orbiting the block at different radii,
speeds, and carrier rates means the two ripple systems drift in and out of phase as they
sweep, so some sweeps light up a single rolling crest while others interfere into a
checkerboard of small peaks -- the skyline reads as "reacting to something" rather than
random flicker, exactly the way a real audio-reactive light/bar array pulses with a mix
instead of jittering independently per pixel.

Grid: 22 x 22 = 484 buildings (cheap flat-shaded box patches, not a literal per-frame
scatter draw), coloured by instantaneous height on a hot ``inferno`` ramp so the tallest,
most wave-driven bars glow brightest. The camera performs a slow partial orbit
(``elev``/``azim`` fed straight to ``bar3d`` each frame, following the pattern used by
``09_rotating_3d_starfield.py``) so the skyline's 3D structure -- and the wavefronts
rippling across its rooftops -- stay legible instead of collapsing into a flat top-down grid.
"""

from __future__ import annotations

import time

import numpy as np

import glplot.animation as animation
import glplot.pyplot as plt

# ---------------------------------------------------------------------------
# City grid: 22 x 22 lots on a regular block spacing, in metres.
# ---------------------------------------------------------------------------
GRID_N = 22
BLOCK_SPACING = 5.0  # m between adjacent lot centres
HALF = (GRID_N - 1) / 2.0
idx = np.arange(GRID_N) - HALF
gx, gy = np.meshgrid(idx, idx)
X = (gx * BLOCK_SPACING).ravel().astype(np.float64)
Y = (gy * BLOCK_SPACING).ravel().astype(np.float64)
N_BUILDINGS = X.size  # 484

BASE_HEIGHT = 3.0  # m, resting rooftop height with no wave energy present
MIN_HEIGHT = 1.0  # m, a building never fully vanishes even in a deep trough
FOOTPRINT = 3.6  # m, plan footprint side length (before the inter-lot gap)

# --- emitter 1: fast inner orbit ---
R1 = 22.0  # m, orbit radius
TURNS1 = 1.6  # orbits completed over the whole animation
PHASE1 = 0.4
AMP1 = 5.5  # m, peak height contribution right at the source
K1 = 2.0 * np.pi / 11.0  # spatial wavenumber -> ~11 m ring-to-ring spacing
DECAY1 = 26.0  # m, e-folding distance of the wave's reach
CARRIER1 = 0.34  # rad advanced per frame -> how fast rings race outward

# --- emitter 2: slower outer orbit, opposite sense, different carrier rate ---
R2 = 40.0  # m, orbit radius
TURNS2 = -0.9  # negative: orbits the other way
PHASE2 = 2.7
AMP2 = 4.0
K2 = 2.0 * np.pi / 15.0
DECAY2 = 34.0
CARRIER2 = 0.22

FRAMES = 76
FPS = 20


def city_heights(frame: int) -> np.ndarray:
    """Per-building rooftop height (m) driven by two moving, ringing wave sources."""
    t = frame / FRAMES  # 0..1 across the whole animation -- drives the orbital sweep

    theta1 = 2.0 * np.pi * TURNS1 * t + PHASE1
    cx1, cy1 = R1 * np.cos(theta1), R1 * np.sin(theta1)
    theta2 = 2.0 * np.pi * TURNS2 * t + PHASE2
    cx2, cy2 = R2 * np.cos(theta2), R2 * np.sin(theta2)

    r1 = np.hypot(X - cx1, Y - cy1)
    r2 = np.hypot(X - cx2, Y - cy2)

    wave_time = frame  # independent, faster-ticking phase clock for the ring carrier
    wave1 = AMP1 * np.cos(K1 * r1 - CARRIER1 * wave_time) * np.exp(-r1 / DECAY1)
    wave2 = AMP2 * np.cos(K2 * r2 - CARRIER2 * wave_time) * np.exp(-r2 / DECAY2)

    height = BASE_HEIGHT + wave1 + wave2
    return np.clip(height, MIN_HEIGHT, None)


VMAX = BASE_HEIGHT + AMP1 + AMP2  # fixed colour ceiling so colour = height, frame to frame

fig = plt.figure("Gallery - Growing 3D Bar City", figsize=(8, 6.5))
plt.plot_style("stage")
# Pin the view volume: without this the z-axis autoscales to each frame's own current
# bar heights (the color ceiling above was already fixed, but the axis range was not),
# so the skyline visibly grew a taller box behind it as the waves built up instead of
# staying in one stable frame. X/Y are pinned too for the same reason -- consistency
# with a real elev/azim-driven camera pan, which reads as filming a fixed volume.
_XY_LIM = HALF * BLOCK_SPACING + FOOTPRINT
plt.set_xlim3d(-_XY_LIM, _XY_LIM)
plt.set_ylim3d(-_XY_LIM, _XY_LIM)
plt.set_zlim3d(0.0, VMAX * 1.05)

_orbit_start, _orbit_span = -52.0, 46.0  # degrees swept across the animation, eased back and forth


def update(frame: int):
    height = city_heights(frame)

    # Slow, eased camera orbit + gentle elevation bob -- follows the moving wave sources
    # without ever spinning past a confusing angle.
    swing = 0.5 - 0.5 * np.cos(2.0 * np.pi * frame / FRAMES)  # 0 -> 1 -> 0
    azim = _orbit_start + _orbit_span * swing
    elev = 30.0 + 7.0 * np.sin(2.0 * np.pi * frame / FRAMES + 0.5)

    plt.cla()
    plt.bar3d(
        X,
        Y,
        np.zeros(N_BUILDINGS),
        FOOTPRINT,
        FOOTPRINT,
        height,
        c=height,
        cmap="inferno",
        vmin=0.0,
        vmax=VMAX,
        alpha=0.98,
        edge_color=(0.05, 0.02, 0.02, 0.4),
        edge_width=0.4,
        gap=0.18,
        shape="box",
        elev=elev,
        azim=azim,
    )
    t = frame / FRAMES
    plt.title(f"Wave-driven bar city: N={N_BUILDINGS} rooftops, t={t:4.2f}")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.zlabel("Height (m)")
    return []


if __name__ == "__main__":
    # --- one-frame timing check before committing to the full render ---------------
    t0 = time.perf_counter()
    update(FRAMES // 3)
    elapsed = time.perf_counter() - t0
    print(f"[timing] one frame: {elapsed:.3f}s -> ~{elapsed * FRAMES:.1f}s for {FRAMES} frames")

    ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=1000.0 / FPS)
    # plt.show()
    ani.save("examples/gallery/animations/results/11_growing_3d_bar_city.gif", fps=FPS)
