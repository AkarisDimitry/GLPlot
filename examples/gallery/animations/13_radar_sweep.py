"""PPI radar sweep -- a rotating beam lighting up contacts on a phosphor-afterglow screen.

A classic Plan Position Indicator (PPI) is the display behind both air-traffic and marine
surveillance radar: a narrow beam rotates continuously about the station at the centre,
and every echo it illuminates brightens instantly, then fades -- the old CRT sets used a
literal phosphor coating with a real afterglow half-life, so a contact stayed visible
between sweeps instead of blinking on and off once per rotation. This animation reproduces
that afterglow electronically: each of ``N_TARGETS`` contacts (ships and slow aircraft,
scattered at uniform *areal* density across the coverage circle -- ``r = R*sqrt(u)``, not
plain ``r = R*u``, is what keeps a uniform random draw from clumping near the centre) carries
its own persistent ``brightness`` in ``[0, 1]``. Every frame the beam advances by a fixed
angular step; any contact whose bearing the beam swept across during that step is set to
full brightness (a genuine hit test against the swept angular *interval*, not just the
instantaneous beam angle, so nothing is skipped between frames), and every other contact's
brightness decays geometrically (``brightness *= DECAY``) -- the discrete-time analogue of
exponential phosphor decay. Freshly hit contacts flash a hot amber-white before cooling
through green into the dim afterglow, and each contact also drifts slowly in range and
bearing (ships holding a heading, aircraft loitering) so the picture never quite repeats.

The beam itself is drawn twice over: a soft trailing wedge of ``N_TRAIL_LAYERS`` stacked,
increasingly narrow and opaque triangle fans (``add_patch`` with a fan of arc points, the
same fan-triangulation ``fill()`` uses internally) gives the classic tapered glow behind a
rotating radar line, and the 10-layer glow recipe (the same bright line redrawn
progressively wider and fainter) gives the leading edge itself a hot core. GLPlot has no
true polar projection -- ``gplt.polar()`` says so directly -- so every position here is
converted by hand with the ordinary ``x = r*cos(theta), y = r*sin(theta)`` and drawn on
plain Cartesian axes; the range rings and bearing spokes are drawn as ordinary circles and
line segments rather than a polar grid.
"""

from __future__ import annotations

import numpy as np

import glplot.animation as animation
import glplot.pyplot as plt

FRAMES = 84
MAX_R = 100.0  # km, instrumented range of the station
RING_RADII = (25.0, 50.0, 75.0, 100.0)  # km
N_TARGETS = 650
REVOLUTIONS = 2.75  # full beam rotations over the whole run -- non-integer so it never
# lands back on its opening bearing at the last frame
OMEGA_SWEEP = 2.0 * np.pi * REVOLUTIONS / FRAMES  # bearing advance per frame, rad
DECAY = 0.90  # per-frame afterglow decay -- ~10 frames to fade most of the way out
TRAIL_SPAN = np.radians(75.0)  # angular width of the trailing glow wedge
N_TRAIL_LAYERS = 16
ARC_PTS = 22

rng = np.random.default_rng(13)
r0 = MAX_R * np.sqrt(rng.uniform(0.02, 0.97, N_TARGETS))  # uniform areal density
theta0 = rng.uniform(0.0, 2.0 * np.pi, N_TARGETS)
radial_speed = rng.normal(0.0, 0.045, N_TARGETS)  # km/frame -- ships holding a heading
angular_speed = rng.normal(0.0, 0.0022, N_TARGETS)  # rad/frame -- slow bearing drift
return_strength = rng.uniform(0.35, 1.0, N_TARGETS)  # radar-cross-section proxy

brightness = np.zeros(N_TARGETS)  # persistent afterglow state, updated every frame

R_LO, R_HI = 6.0, MAX_R - 4.0  # keep contacts off the dead centre and the outer ring
R_SPAN = R_HI - R_LO
R_PERIOD = 2.0 * R_SPAN


def reflected_radius(frame: int) -> np.ndarray:
    """Each contact's range at ``frame``, bouncing off [R_LO, R_HI] like a triangle wave.

    A straight-line radial drift would eventually fly contacts off the scope or collapse
    them into the centre; folding the drift back on itself keeps every contact drifting
    forever within the coverage circle, the way a patrol holding a beat never truly leaves.
    """
    raw = r0 + radial_speed * frame
    m = np.mod(raw - R_LO, R_PERIOD)
    return R_LO + np.where(m > R_SPAN, R_PERIOD - m, m)


def fan_indices(n_verts: int) -> np.ndarray:
    """Triangle-fan indices from vertex 0 (the origin) -- same trick ``fill()`` uses."""
    fan = np.arange(1, n_verts - 1)
    return np.column_stack([np.zeros_like(fan), fan, fan + 1]).ravel().astype(np.uint32)


fig = plt.figure("Gallery - Radar Sweep", figsize=(7.6, 7.6))
plt.plot_style("neon")  # true black stage, additive blend -- the glow needs both

DIM_GREEN = np.array([0.03, 0.14, 0.05])
CORE_GREEN = np.array([0.25, 1.0, 0.35])
HOT_AMBER = np.array([1.0, 0.82, 0.30])
SWEEP_RGB = (0.45, 1.0, 0.55)
RING_RGB = (0.12, 0.65, 0.28)
SPOKE_RGB = (0.10, 0.45, 0.20)


def blip_colors(bright: np.ndarray) -> np.ndarray:
    """Afterglow colour ramp: dim green ghost -> vivid green -> hot amber at the instant of
    detection, so a contact's own colour tells you how long ago the beam passed over it.
    """
    lo_mix = np.clip(bright / 0.65, 0.0, 1.0)[:, None]
    hi_mix = np.clip((bright - 0.65) / 0.35, 0.0, 1.0)[:, None]
    rgb = DIM_GREEN[None, :] * (1.0 - lo_mix) + CORE_GREEN[None, :] * lo_mix
    rgb = rgb * (1.0 - hi_mix) + HOT_AMBER[None, :] * hi_mix
    return rgb


def update(frame: int):
    global brightness

    sweep_prev = OMEGA_SWEEP * (frame - 1) if frame > 0 else -OMEGA_SWEEP
    sweep_curr = OMEGA_SWEEP * frame
    theta = np.mod(theta0 + angular_speed * frame, 2.0 * np.pi)

    # A contact is "hit" this frame if its bearing fell inside the angular interval the
    # beam swept through since the previous frame -- checking the interval, not just the
    # instantaneous current angle, so a fast sweep can never step clean over a contact.
    ang_diff = np.mod(theta - sweep_prev, 2.0 * np.pi)
    hit = ang_diff <= OMEGA_SWEEP
    brightness = np.where(hit, 1.0, brightness * DECAY)

    r = reflected_radius(frame)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    plt.cla()

    # Range rings and bearing spokes -- GLPlot has no polar grid, so these are ordinary
    # circles/segments computed by hand.
    ring_theta = np.linspace(0.0, 2.0 * np.pi, 121)
    for radius in RING_RADII:
        plt.plot(
            radius * np.cos(ring_theta),
            radius * np.sin(ring_theta),
            color=(*RING_RGB, 0.55),
            linewidth=1.0,
            linestyle="--",
        )
        lx, ly = radius * np.cos(np.radians(50.0)), radius * np.sin(np.radians(50.0))
        plt.text(lx, ly, f"{radius:.0f} km", color=(*RING_RGB, 0.9))

    for deg in range(0, 360, 30):
        a = np.radians(deg)
        plt.plot(
            [0.0, MAX_R * np.cos(a)],
            [0.0, MAX_R * np.sin(a)],
            color=(*SPOKE_RGB, 0.35),
            linewidth=0.8,
        )

    # Trailing afterglow wedge behind the beam: stacked triangle fans, widest and faintest
    # first, narrowest and most opaque last, so painter's-order compositing builds a smooth
    # angular gradient rather than a hard-edged pie slice.
    for i in range(N_TRAIL_LAYERS, 0, -1):
        frac = i / N_TRAIL_LAYERS
        width = TRAIL_SPAN * frac
        arc = np.linspace(sweep_curr - width, sweep_curr, ARC_PTS)
        xs = MAX_R * np.cos(arc)
        ys = MAX_R * np.sin(arc)
        verts = np.vstack([[0.0, 0.0], np.column_stack([xs, ys])]).astype(np.float32)
        alpha = 0.018 + 0.05 * (1.0 - frac) ** 2
        # edge_color must be passed explicitly: the headless export path's default for an
        # unset edge_color is an opaque blue, which would outline every triangle of this
        # fan (visible as stray blue spokes) if left to fall back.
        plt.add_patch(
            verts,
            indices=fan_indices(len(verts)),
            mode="triangles",
            face_color=(*SWEEP_RGB, alpha),
            edge_color=(*SWEEP_RGB, 0.0),
        )

    # Beam leading edge: the 10-layer glow recipe (same line, progressively wider and
    # fainter) plus one crisp, fully-opaque core line on top.
    xt, yt = MAX_R * np.cos(sweep_curr), MAX_R * np.sin(sweep_curr)
    for i in range(1, 11):
        plt.plot([0.0, xt], [0.0, yt], color=(*SWEEP_RGB, 0.05), linewidth=1.4 + 1.35 * i)
    plt.plot([0.0, xt], [0.0, yt], color=(0.85, 1.0, 0.88, 1.0), linewidth=2.0)

    # Contacts, coloured continuously by afterglow + signal strength, but bucketed into a
    # handful of size bands: the headless (Agg) export path honours one marker size per
    # scatter() call, so a continuous brightness -> size taper has to be several calls
    # (same technique 08_rose_curve_trails.py uses for its comet tail).
    rgb = blip_colors(brightness)
    alpha = np.clip(0.06 + 0.9 * brightness, 0.0, 1.0) * (0.45 + 0.55 * return_strength)
    colors = np.concatenate([rgb, alpha[:, None]], axis=1)

    n_bands = 10
    edges = np.linspace(0.0, 1.0, n_bands + 1)
    for b in range(n_bands):
        lo, hi = edges[b], edges[b + 1]
        mask = (brightness >= lo) & (brightness <= hi if b == n_bands - 1 else brightness < hi)
        if not np.any(mask):
            continue
        band_center = 0.5 * (lo + hi)
        size = 2.6 + 10.0 * band_center**1.4
        plt.scatter(x[mask], y[mask], c=colors[mask], size=size)

    plt.set_aspect("equal")
    plt.xlim(-MAX_R * 1.08, MAX_R * 1.08)
    plt.ylim(-MAX_R * 1.08, MAX_R * 1.08)
    bearing_deg = np.degrees(sweep_curr) % 360.0
    plt.title(f"PPI radar sweep -- bearing {bearing_deg:05.1f} deg, {N_TARGETS} contacts")
    plt.xlabel("Range, east-west (km)")
    plt.ylabel("Range, north-south (km)")
    return []


ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=42)
# plt.show()
ani.save("examples/gallery/animations/results/13_radar_sweep.gif", fps=20)
