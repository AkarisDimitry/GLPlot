"""Rotating 3D starfield -- an orbiting camera around a simulated dwarf spiral galaxy.

The star positions are generated once, from a simple two-component model that shows up in
every real disk galaxy: a dense, roughly spherical central bulge of older stars, plus a
thin two-armed spiral disk laid out on a logarithmic spiral

    r(theta) = a * exp(b * theta)

(the same curve that describes nautilus shells and hurricane bands -- a constant pitch
angle between the arm and a circle at each radius). Star "brightness" falls off with
galactocentric radius, exp(-r / scale), which sets both point size and a per-frame
twinkle amplitude so the animation is not just a rigid image spinning in place.

Every frame keeps the star positions fixed and only moves the camera -- ``elev``/``azim``
passed straight to ``scatter3d`` each call, exactly like a real orbit around the cluster --
which is what makes the flat spiral read unambiguously as a 3D structure rather than a
static picture.
"""

import numpy as np

import glplot.animation as animation
import glplot.pyplot as plt

rng = np.random.default_rng(2609)

N_BULGE = 1_100
N_ARMS = 5_400
N = N_BULGE + N_ARMS  # total stars drawn every frame -- kept well under 10k for 3D scatter

# --- central bulge: an old, roughly spherical population near the galactic core ---
bulge_r = rng.gamma(2.0, 0.55, N_BULGE)  # kpc from center
bulge_theta = rng.uniform(0, 2 * np.pi, N_BULGE)
bulge_phi = np.arccos(rng.uniform(-1, 1, N_BULGE))
xb = bulge_r * np.sin(bulge_phi) * np.cos(bulge_theta)
yb = bulge_r * np.sin(bulge_phi) * np.sin(bulge_theta)
zb = 0.6 * bulge_r * np.cos(bulge_phi)  # slightly flattened bulge

# --- two-armed logarithmic spiral disk: r = a * exp(b*theta), two arms 180 deg apart ---
outer_radius = 7.5  # kpc, disk edge
pitch_b = 0.22  # spiral pitch parameter -- larger = more open arms
winds = 3.1 * np.pi  # how many radians of spiral to sample before hitting outer_radius
theta = rng.uniform(0, winds, N_ARMS)
arm_radius = 0.35 * np.exp(pitch_b * theta)
arm_radius = arm_radius / arm_radius.max() * outer_radius
arm_pick = rng.integers(0, 2, N_ARMS) * np.pi  # which of the two arms
scatter_spread = 0.16 + 0.10 * (arm_radius / outer_radius)  # arms fuzz out toward the rim
twist = theta + arm_pick + rng.normal(0.0, scatter_spread, N_ARMS)
xa = arm_radius * np.cos(twist)
ya = arm_radius * np.sin(twist)
disk_thickness = 0.22 * np.exp(-arm_radius / (0.6 * outer_radius)) + 0.03
za = rng.normal(0.0, 1.0, N_ARMS) * disk_thickness

x = np.concatenate([xb, xa]).astype(np.float32)
y = np.concatenate([yb, ya]).astype(np.float32)
z = np.concatenate([zb, za]).astype(np.float32)
radius = np.sqrt(x**2 + y**2 + z**2)  # galactocentric radius, used for color + brightness

brightness = np.exp(-radius / 2.6)  # dimmer with distance from the core, like real starlight
base_size = 2.0 + 9.0 * brightness  # bigger, brighter marks near the core
twinkle_phase = rng.uniform(0, 2 * np.pi, N)
twinkle_rate = rng.uniform(0.7, 1.6, N)

FRAMES = 72

fig = plt.figure("Gallery - Rotating 3D Starfield", figsize=(8, 7))
plt.plot_style("dark")


def update(frame: int):
    t = frame / FRAMES
    azim = -60.0 + 360.0 * t  # one full orbit around the galaxy over the whole animation
    elev = 24.0 + 11.0 * np.sin(2 * np.pi * t + 0.6)  # gentle up/down bob layered on the orbit

    # cheap per-star twinkle: a small sinusoidal size wobble, different phase/rate per star,
    # so the field visibly sparkles as it turns instead of reading as a frozen photograph.
    twinkle = 1.0 + 0.22 * np.sin(2 * np.pi * twinkle_rate * t * FRAMES / 18.0 + twinkle_phase)
    sizes = base_size * twinkle

    plt.cla()
    plt.scatter3d(
        x,
        y,
        z,
        c=radius,
        cmap="plasma_r",
        s=sizes,
        alpha=0.85,
        elev=elev,
        azim=azim,
        label=f"{N:,} stars",
    )
    plt.title(f"Rotating dwarf spiral galaxy ({N:,} stars, azim={azim:5.0f} deg)")
    plt.xlabel("x (kpc)")
    plt.ylabel("y (kpc)")
    plt.zlabel("z (kpc)")
    return []


ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=42)
# plt.show()
ani.save("examples/gallery/animations/results/09_rotating_3d_starfield.gif", fps=20)
