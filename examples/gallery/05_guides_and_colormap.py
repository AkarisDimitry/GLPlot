import numpy as np
from scipy.ndimage import gaussian_filter

import glplot.pyplot as plt

# Synthetic backstory: a two-channel coincidence detector records paired pulse
# heights (Channel A, Channel B) across many distinct interaction populations
# (separate reaction/decay channels), some of which overlap in the plane.
# Colour reads as *event density* rather than raw scattered points -- built
# the same way GLPlot's own GPU density mode works: accumulate raw hits into
# a fine grid, then resolve that accumulation through a colormap. (This
# script pre-bakes that accumulate-then-resolve pipeline on the CPU with
# numpy + a gaussian blur so it exports to a static PNG; ``plt.density(True)``
# below flips on the real, live, GPU-accelerated version of the same idea for
# anyone who runs this with ``plt.show()`` instead of ``plt.savefig()``.)
rng = np.random.default_rng(21)

n_clusters = 16
centers = rng.uniform([-6.0, -4.5], [6.0, 4.5], size=(n_clusters, 2))
spread = rng.uniform(0.28, 0.8, size=(n_clusters, 2))
weights = rng.integers(6_000, 30_000, size=n_clusters)
n = int(weights.sum())

channel_a = np.concatenate(
    [rng.normal(centers[i, 0], spread[i, 0], size=int(w)) for i, w in enumerate(weights)]
)
channel_b = np.concatenate(
    [rng.normal(centers[i, 1], spread[i, 1], size=int(w)) for i, w in enumerate(weights)]
)

# Accumulate: bin the raw hits into a fine grid (the GPU path's additive
# blend into an RGBA32F accumulation texture). Resolve: blur + colormap (the
# GPU path's single-pass fullscreen resolve shader).
bins = 240
hist, xedges, yedges = np.histogram2d(channel_a, channel_b, bins=bins)
density_field = gaussian_filter(hist, sigma=2.0).T
xc = 0.5 * (xedges[:-1] + xedges[1:])
yc = 0.5 * (yedges[:-1] + yedges[1:])
X, Y = np.meshgrid(xc, yc)

plt.figure("Gallery - Guides and Density", figsize=(9, 6))
plt.density(True)  # GLPlot's real accumulate-then-resolve mode, for live use
plt.contourf(X, Y, density_field, levels=28, cmap="plasma")
# axhline()/axvline() span the *current* view window at call time, and that
# window is only fitted to the data once autoscale runs -- which normally
# happens on the first live frame, so it would still be the pre-data default
# in this headless export (a stray short cross at the origin instead of full
# guide lines). hlines()/vlines() are used directly instead, spanning the
# density field's own tight extent so the guides reach edge to edge without
# padding the final view beyond the field (as a plain autoscale() would).
plt.hlines(0, xc.min(), xc.max(), colors="white", linestyles="--", linewidth=1.0, alpha=0.55)
plt.vlines(0, yc.min(), yc.max(), colors="white", linestyles="--", linewidth=1.0, alpha=0.55)
plt.xlabel("Channel A pulse height (counts)")
plt.ylabel("Channel B pulse height (counts)")
plt.title(f"Coincidence density across {n_clusters} interaction populations ({n:,} events)")
# plt.show()
plt.savefig("examples/gallery/results/05_guides_and_colormap.png")
