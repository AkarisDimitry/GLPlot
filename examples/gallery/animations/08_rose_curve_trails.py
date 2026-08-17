"""Rose (rhodonea) curve traced by a glowing comet, dragging a fading rainbow trail.

A rose curve is the polar graph ``r = cos(k*theta)`` -- the trace of a point on a disc
rotating inside another disc at an angular ratio ``k``, the exact geometry behind a
Spirograph toy. The same ``cos(k*theta)`` lobe shape also shows up in physics as the
angular far-field pattern of a multipole radiator (a dipole is one lobe, a quadrupole is
four, and so on) -- so a "petal count" here is not just decoration, it is literally what a
higher radiation order looks like plotted in polar coordinates. Both ``r`` and ``theta``
are dimensionless (a normalized field amplitude and an angle in radians); there is no
physical length scale to attach units to, so the axes are labeled accordingly.

Rather than snap the petal count to a fixed integer, ``k`` drifts slowly across the whole
animation -- the comet head is really tracing a slowly breathing family of rose curves, not
retracing one static shape, which is what keeps 80 frames from ever looking like a loop.

Every frame slices a fixed window out of one precomputed high-resolution curve (no
per-frame recomputation of the maths itself -- just a numpy slice), colors the visible
window by how recently each point was laid down (bright/opaque/large near the head, fading
to dim/transparent/tiny at the tail), and stacks the 10-layer "glow" recipe on the head
point so the comet has a genuine soft halo. Hue rotates continuously across the whole
animation via ``colorsys.hsv_to_rgb``, so the color itself is part of what is animating.
"""

import colorsys

import numpy as np

import glplot.animation as animation
import glplot.pyplot as plt

FRAMES = 80
STEPS_PER_FRAME = 40  # dense sub-steps appended to the trail each frame
MAX_TRAIL = 1200  # points kept on screen at once -- cheap, well under budget
TOTAL_STEPS = FRAMES * STEPS_PER_FRAME

# --- Precompute the whole curve once: one long rose-curve arc, petal count "breathing"
# slowly across its length so the shape keeps reshaping as the comet sweeps around it. ---
THETA_TOTAL = 9.0 * np.pi
theta_full = np.linspace(0.0, THETA_TOTAL, TOTAL_STEPS)
k_full = 6.0 + 2.6 * np.sin(0.16 * theta_full)  # petal order breathes ~3.4..8.6
r_full = np.cos(k_full * theta_full)
x_full = r_full * np.cos(theta_full)
y_full = r_full * np.sin(theta_full)

fig = plt.figure("Gallery - Rose Curve Comet Trail", figsize=(7.5, 7.5))
plt.plot_style("neon")  # true black stage: the glow only reads against real dark


def update(frame: int):
    end = (frame + 1) * STEPS_PER_FRAME
    start = max(0, end - MAX_TRAIL)
    xs = x_full[start:end]
    ys = y_full[start:end]
    n = xs.shape[0]

    # age_norm: 0 at the freshest point (the comet head, last in the slice) rising to 1 at
    # the oldest surviving point (the tail about to fall off the trail entirely).
    age = np.arange(n)[::-1]
    age_norm = age / max(MAX_TRAIL - 1, 1)

    # Hue rotates continuously across the whole animation (a bit more than one full turn
    # over FRAMES frames) plus a small extra sweep along the trail's own age, so each frame
    # reads as "rainbow comet" rather than "single flat color".
    base_hue = (frame / FRAMES * 1.35) % 1.0
    hues = (base_hue + 0.14 * age_norm) % 1.0
    sat = 0.95 - 0.15 * age_norm
    val = 1.0 - 0.55 * age_norm
    rgb = np.array(
        [colorsys.hsv_to_rgb(h, s, v) for h, s, v in zip(hues, sat, val)], dtype=np.float32
    )
    trail_alpha = (1.0 - age_norm) ** 1.6
    trail_colors = np.concatenate([rgb, trail_alpha[:, None]], axis=1)

    plt.cla()

    # The headless (Agg) export path that ``ani.save()`` renders through draws every point
    # of one ``scatter()`` call at the SAME marker size -- only per-point *color* survives
    # to a PNG/GIF frame, not a per-point size array. So the age -> size taper has to be a
    # handful of separate ``scatter()`` calls, one per age band (each with its own uniform
    # size), the same "many separate draws stack into one soft gradient" idea the gallery's
    # 10-layer line-glow recipe already relies on -- just applied to marker size instead of
    # line width. Color and alpha stay fully continuous *within* each band's array.
    n_bands = 18
    band_edges = np.linspace(0.0, 1.0, n_bands + 1)
    for b in range(n_bands):
        lo, hi = band_edges[b], band_edges[b + 1]
        mask = (age_norm >= lo) & (age_norm <= hi if b == n_bands - 1 else age_norm < hi)
        if not np.any(mask):
            continue
        band_center = 0.5 * (lo + hi)
        band_size = 3.0 + 12.0 * (1.0 - band_center) ** 1.3
        plt.scatter(xs[mask], ys[mask], c=trail_colors[mask], size=band_size)

    if n:
        hx, hy = float(xs[-1]), float(ys[-1])
        # A classic comet head is brilliant near-white-hot, not just a saturated version of
        # the tail color -- low saturation, full value, still tinted by the current hue so
        # it stays tied to the trail's own color story instead of reading as a foreign white
        # dot. That contrast is also what keeps the head visible when it happens to sit in
        # the crowded middle of the rose, where many trail strands cross.
        head_rgb = colorsys.hsv_to_rgb(base_hue, 0.22, 1.0)

        # 10-layer glow recipe on the comet head: the same point redrawn progressively
        # larger and fainter (each its own tiny ``scatter()`` call, so its size is actually
        # honored on export) so the stacked translucent copies blend into a soft halo, plus
        # one bright, fully opaque core drawn last on top -- sized past the trail's own
        # largest marker so the head always reads as the brightest, biggest thing on screen.
        n_glow = 10
        for i in range(1, n_glow + 1):
            plt.scatter([hx], [hy], c=[(*head_rgb, 0.09)], size=12.0 * (1.23**i))
        plt.scatter([hx], [hy], c=[(*head_rgb, 1.0)], size=20.0)

    # Frozen view: r = cos(k*theta) never exceeds |r| = 1, so a fixed +-1.3 comfortably
    # covers the curve at every petal count with a little margin -- same ceiling the
    # in-and-out "breathing" zoom used to reach at its widest, just held there instead of
    # continuously resizing the frame around a curve whose own extent barely changes.
    plt.xlim(-1.3, 1.3)
    plt.ylim(-1.3, 1.3)

    k_now = float(k_full[min(end, TOTAL_STEPS) - 1])
    plt.title(f"Rose curve r = cos({k_now:.2f}·θ) — comet trail ({n} pts)")
    plt.xlabel("x = r·cos(θ)  (dimensionless)")
    plt.ylabel("y = r·sin(θ)  (dimensionless)")
    return []


ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=40)
# plt.show()
ani.save("examples/gallery/animations/results/08_rose_curve_trails.gif", fps=20)
