import numpy as np

import glplot.pyplot as plt

rng = np.random.default_rng(7)

# A simulated wide-field survey: a sparse background population plus three dense
# clusters, millions of point sources total. The insets below are not a separate
# illustration bolted on afterwards -- they rasterize the very same point cloud at full
# detail inside each cluster's own footprint, so ``inset_axes()`` + ``imshow()`` are
# doing something a plain zoom-in on the scatter itself cannot: showing individual point
# structure that millions of overlapping markers wash out at the full-field scale.
FIELD = (0.0, 100.0)

n_background = 1_600_000
bg_x = rng.uniform(*FIELD, n_background)
bg_y = rng.uniform(*FIELD, n_background)
# A tight temperature spread: millions of independently-colored points at high per-point
# color variance render as visual static rather than a sky. Real sources at this density
# do vary, just not by much -- the three clusters below carry the dramatic color range.
bg_temp = np.clip(rng.normal(0.30, 0.03, n_background), 0.0, 1.0)

cluster_specs = [
    {
        "center": (20.0, 45.0),
        "spread": 3.6,
        "n": 260_000,
        "temp": 0.88,
        "color": (0.12, 0.46, 0.71),
    },
    {
        "center": (55.0, 18.0),
        "spread": 3.0,
        "n": 220_000,
        "temp": 0.55,
        "color": (1.00, 0.50, 0.05),
    },
    {
        "center": (80.0, 48.0),
        "spread": 3.3,
        "n": 240_000,
        "temp": 0.72,
        "color": (0.17, 0.63, 0.17),
    },
]
# Every cluster box (see `half = spread * 4` below) tops out at y <= 61.2, leaving the
# axes-fraction band above y=0.66 -- where the insets sit -- genuinely empty. The insets
# used to be positioned directly on top of the very clusters they were framing (two of
# the three cluster centers were at y=70/74, inside the insets' own 0.66-0.98 band).

xs, ys, temps = [bg_x], [bg_y], [bg_temp]
for spec in cluster_specs:
    cx, cy = spec["center"]
    xs.append(rng.normal(cx, spec["spread"], spec["n"]))
    ys.append(rng.normal(cy, spec["spread"], spec["n"]))
    temps.append(np.clip(rng.normal(spec["temp"], 0.10, spec["n"]), 0.0, 1.0))

x = np.concatenate(xs)
y = np.concatenate(ys)
temp = np.concatenate(temps)
n_total = x.size


def crop_to_image(cx, cy, half, color, n=220):
    """Rasterize the *actual* points inside this window into a true-color image.

    A full-detail crop of the same cloud the main scatter draws (not a synthetic
    stand-in): a fine 2D histogram of the real points in this window, tinted by the
    region's own highlight color and shaded by local density -- so the inset visually
    matches the box drawn around its source region on the main plot, standing in for
    the (unsupported) per-edge spine coloring the loop below still asks for.
    """
    xlim = (cx - half, cx + half)
    ylim = (cy - half, cy + half)
    mask = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
    count, _, _ = np.histogram2d(x[mask], y[mask], bins=n, range=[xlim, ylim])
    density = np.clip(count / max(count.max(), 1.0), 0.0, 1.0)
    glow = density[..., None] ** 0.45
    rgb = np.asarray(color)[None, None, :3] * glow
    img = np.clip(rgb, 0.0, 1.0)
    # histogram2d's first axis is x; transpose + flip so row 0 (top of the image) is
    # the largest y, matching imshow's default origin.
    return np.transpose(img, (1, 0, 2))[::-1], xlim, ylim


fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(x, y, c=temp, cmap="inferno", s=1.4, alpha=0.4, vmin=0.0, vmax=1.0)

inset_positions = [(0.03, 0.66), (0.35, 0.66), (0.67, 0.66)]

for spec, (left, bottom) in zip(cluster_specs, inset_positions):
    cx, cy = spec["center"]
    half = spec["spread"] * 4.0
    color = spec["color"]

    box_x = [cx - half, cx + half, cx + half, cx - half, cx - half]
    box_y = [cy - half, cy - half, cy + half, cy + half, cy - half]
    ax.plot(box_x, box_y, color=color, linewidth=2)

    img, xlim, ylim = crop_to_image(cx, cy, half, color)
    inset = ax.inset_axes([left, bottom, 0.28, 0.32])
    inset.imshow(img, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], origin="upper")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)

ax.set_xlabel("Field X (arcmin)")
ax.set_ylabel("Field Y (arcmin)")
ax.set_xlim(FIELD)
ax.set_ylim(FIELD)
ax.set_title(f"Wide-field survey (N = {n_total:,} sources) with full-detail cluster insets")
# plt.show()
plt.savefig("examples/gallery/results/27_inset_axes_image.png")
