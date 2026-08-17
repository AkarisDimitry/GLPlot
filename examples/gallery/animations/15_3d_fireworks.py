"""3D fireworks -- particle shells igniting, expanding, and falling under gravity.

A real aerial shell firework is a two-stage rocket: a lifting charge sends the shell to
altitude, then a bursting charge inside detonates and flings dozens of pyrotechnic "stars"
outward in (very roughly) every direction at once -- a spherical shell of particles
expanding from a point. Each star is a compressed pellet of fuel plus a metal salt that
burns a specific hue -- sodium/iron for gold, copper compounds for blue, a strontium +
copper blend for magenta, barium for green -- and keeps burning, cooling, and fading for a
second or two before it gutters out, which is why a firework's color always *dies* rather
than switching off.

From ignition on, every star's position is the closed-form solution of a launch velocity
decaying exponentially under air drag, ``v(t) = v0 * exp(-k t)``, so the outward
displacement is ``v0 * (1 - exp(-k t)) / k`` -- fast expansion right at ignition, visibly
slowing as drag eats the momentum -- plus ordinary free-fall, ``-1/2 g t^2``, pulling the
whole shell down as it burns. A brief white pop at the origin stands in for the bursting
charge itself; every star then burns at full brightness in its own shell's color and dims
to black as it dies. That fade is carried entirely by *color*, not size or a per-point
alpha array: a headless GIF export of a 3D scatter only honors matplotlib's per-point
*color* (``layer.colors``) -- a per-point size array is silently flattened to one uniform
marker size on export -- so each burst gets its own short black -> hue colormap and
brightness is sampled from it as a scalar per star, which is what actually survives into
the saved frames. Four shells stagger their ignition across the run, at different
positions and each its own color, over a dark bay with a faint starfield and a slowly
panning camera.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import glplot.animation as animation
import glplot.pyplot as plt

FRAMES = 90
FPS = 20
DT = 1.0 / FPS  # seconds of simulated time per frame, matched to the saved playback rate

G = 9.8  # m/s^2 -- ordinary gravity, pulling every ember back down as it burns
DRAG_K = 1.0  # 1/s -- air-drag decay rate on each ember's launch velocity
FLASH_T = 0.12  # s -- how long the ignition point itself still shows a bright pop

N_PER_BURST = 850
ECHO_DT = DT * 0.5  # a half-frame-earlier sample, so fast-moving embers still read as a
ECHO_MULT = 0.40  # streak rather than a jump-cut between frames, dimmed on top of its age

# Fixed viewing volume, in meters. A headless GIF export re-autoscales a 3D axes to
# whatever is plotted that frame -- explicit axis limits are inert there -- so the box is
# pinned instead by drawing two invisible corner points at its exact extremes every frame
# (see ``_CORNERS`` below), the same "fake it with invisible anchor points" trick the
# 2D gallery already relies on for a fixed axis range.
XLIM, YLIM = 34.0, 34.0
ZLO, ZHI = 0.0, 48.0
_CORNERS = np.array([[-XLIM, -YLIM, ZLO], [XLIM, YLIM, ZHI]], dtype=np.float32)

BURST_SPECS = [
    # ignite_frame, origin (x, y, z) m, hue RGB, speed range m/s, life range s
    dict(ignite=0, origin=(-11.0, 5.0, 23.0), hue=(1.00, 0.55, 0.10), speed=(14.0, 22.0),
         life=1.55, jitter=0.30, seed=101),  # gold -- sodium/iron
    dict(ignite=22, origin=(10.0, -6.0, 25.0), hue=(0.10, 0.80, 1.00), speed=(15.0, 23.0),
         life=1.60, jitter=0.30, seed=202),  # electric blue -- copper
    dict(ignite=46, origin=(-2.0, 9.0, 22.0), hue=(1.00, 0.10, 0.60), speed=(14.0, 22.0),
         life=1.55, jitter=0.32, seed=303),  # magenta -- strontium + copper
    dict(ignite=68, origin=(-9.0, -9.0, 24.0), hue=(0.35, 1.00, 0.20), speed=(15.0, 24.0),
         life=1.60, jitter=0.30, seed=404),  # green -- barium
]


def _unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` directions drawn uniformly on the unit sphere (Marsaglia's normal-normalize trick)."""
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _build_burst(spec: dict) -> dict:
    """Precompute one shell's per-star launch geometry and its black -> hue colormap.

    Each star ignites already burning its characteristic color (the separate white pop
    drawn at the origin for ``FLASH_T`` seconds is the bursting charge itself, not the
    stars) -- so the body colormap only needs two stops, full hue fading straight to
    black, driven by a per-star brightness scalar that starts at 1 and decays with age.
    """
    rng = np.random.default_rng(spec["seed"])
    directions = _unit_vectors(N_PER_BURST, rng)
    speed = rng.uniform(spec["speed"][0], spec["speed"][1], N_PER_BURST)
    life = spec["life"] * rng.uniform(1.0 - spec["jitter"], 1.0 + spec["jitter"], N_PER_BURST)
    cmap = LinearSegmentedColormap.from_list(f"burst_{spec['seed']}", ["black", spec["hue"]])
    return {
        "ignite_frame": spec["ignite"],
        "origin": np.array(spec["origin"], dtype=np.float32),
        "v0": (directions * speed[:, None]).astype(np.float32),
        "life": life.astype(np.float32),
        "max_life": float(life.max()),
        "cmap": cmap,
    }


BURSTS = [_build_burst(spec) for spec in BURST_SPECS]

# A faint, fixed starfield for atmosphere -- lit with its own scalar brightness (a gentle
# per-star twinkle) through a plain black -> white colormap, same per-vertex-color mechanism
# the bursts use.
N_STARS = 260
_star_rng = np.random.default_rng(9)
star_x = _star_rng.uniform(-XLIM, XLIM, N_STARS).astype(np.float32)
star_y = _star_rng.uniform(-YLIM, YLIM, N_STARS).astype(np.float32)
star_z = _star_rng.uniform(8.0, ZHI, N_STARS).astype(np.float32)
star_phase = _star_rng.uniform(0.0, 2.0 * np.pi, N_STARS)
star_rate = _star_rng.uniform(0.5, 1.4, N_STARS)
STAR_CMAP = LinearSegmentedColormap.from_list("stars", ["black", "white"])

fig = plt.figure("Gallery - 3D Fireworks", figsize=(8.4, 7))
plt.plot_style("neon")  # true black sky; additive blend + glow makes the embers actually pop


def _burst_state(burst: dict, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ember positions, brightness in [0, 1], and an alive mask at burst-local time ``t`` (s).

    ``factor`` is the closed-form displacement of a velocity decaying as ``exp(-DRAG_K * t)``
    under drag; gravity is added separately on z since drag is only modeling air resistance
    on the launch impulse, not changing the constant downward pull.
    """
    factor = (1.0 - np.exp(-DRAG_K * t)) / DRAG_K
    pos = burst["origin"][None, :] + burst["v0"] * factor
    pos[:, 2] -= 0.5 * G * t * t
    life = burst["life"]
    age = np.clip(t / np.maximum(life, 1e-6), 0.0, 1.0)
    brightness = np.clip(1.0 - age**1.4, 0.0, 1.0)  # 1.0 (full hue) at t=0, 0.0 (black) at burnout
    alive = t <= life
    return pos, brightness, alive


def update(frame: int):
    elapsed = frame * DT
    phase = frame / FRAMES
    elev = 20.0 + 7.0 * np.sin(2.0 * np.pi * phase * 0.6 + 0.4)  # gentle up/down bob
    azim = -55.0 + 46.0 * phase  # slow one-way pan across the whole show

    plt.cla()

    # Invisible box-corner anchors -- pins the headless export's autoscaled axes to a fixed
    # volume every frame (see the ``_CORNERS`` comment above).
    plt.scatter3d(
        _CORNERS[:, 0], _CORNERS[:, 1], _CORNERS[:, 2],
        color=(0.0, 0.0, 0.0, 0.0), s=0.1, alpha=0.0, elev=elev, azim=azim,
    )

    star_bright = 0.55 + 0.45 * np.sin(star_phase + star_rate * elapsed * 2.4)
    star_bright = np.clip(star_bright, 0.12, 1.0)
    plt.scatter3d(
        star_x, star_y, star_z,
        c=star_bright, cmap=STAR_CMAP, vmin=0.0, vmax=1.0,
        s=1.3, alpha=0.55, elev=elev, azim=azim,
    )

    n_live = 0
    n_lit = 0
    for burst in BURSTS:
        t_main = elapsed - burst["ignite_frame"] * DT
        if t_main < -1e-9 or t_main > burst["max_life"]:
            continue
        n_lit += 1

        for dt_back, mult in ((0.0, 1.0), (ECHO_DT, ECHO_MULT)):
            t_sample = t_main - dt_back
            if t_sample < 0.0:
                continue
            pos, brightness, alive = _burst_state(burst, t_sample)
            if not np.any(alive):
                continue
            b = brightness[alive] * mult
            plt.scatter3d(
                pos[alive, 0], pos[alive, 1], pos[alive, 2],
                c=b, cmap=burst["cmap"], vmin=0.0, vmax=1.0,
                s=3.2, alpha=0.9, elev=elev, azim=azim,
            )
            n_live += int(alive.sum())

        if t_main <= FLASH_T:  # a brief white pop from the bursting charge itself
            ox, oy, oz = burst["origin"]
            flash_fade = 1.0 - t_main / FLASH_T
            plt.scatter3d(
                [ox], [oy], [oz], color=(1.0, 1.0, 1.0, 1.0),
                s=16.0, alpha=float(np.clip(flash_fade, 0.0, 1.0)), elev=elev, azim=azim,
            )

    plt.title(f"Fireworks over the bay -- t={elapsed:4.2f}s, {n_lit} burst(s) lit, {n_live:,} embers")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.zlabel("z (m)")
    return []


ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=40)
ani.save("examples/gallery/animations/results/15_3d_fireworks.gif", fps=FPS)
# plt.show()
