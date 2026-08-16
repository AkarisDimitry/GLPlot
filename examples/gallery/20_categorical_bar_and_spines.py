import numpy as np
from matplotlib import colormaps

import glplot.pyplot as plt

# Catalytic hydrogenation yield across four candidate catalysts, each run in
# replicate. Demonstrates bar() with string categories, yerr= error bars, and
# despined axes (ax.spines[...].set_visible(False)) -- but each bar is not a
# single flat rectangle. It is built from its own dense column of jittered
# "raw run" points: millions of individual synthetic measurements stacked
# into the bar's silhouette and colored by height like a thermometer, so the
# bar is visibly *made of* the underlying data rather than a summary shape
# drawn over it. A normal plotting library can render a few thousand of
# these markers before the canvas grinds to a halt; GLPlot pushes the whole
# multi-million-point cloud through in a single call.
#
# On top of that raw-sample silhouette, each bar's own rectangle outline (and
# its error-bar caps) is redrawn as a neon-glow "tube": the same short polyline
# plotted ~10 times, each copy a touch wider and fainter, so the stacked
# translucent copies melt into a soft halo around a crisp bright core -- the
# real mplcyberpunk technique, just built from glplot.pyplot calls instead of
# a shader. One saturated hue per catalyst keeps the four bars distinguishable
# as glowing frames even before you read the x-axis.
rng = np.random.default_rng(42)
categories = ["none (control)", "Pd/C", "Pt/C", "Ni/Al2O3"]
means = [12.0, 68.5, 74.2, 55.1]
errors = [1.8, 3.4, 2.6, 4.1]
points_per_bar = 650_000  # -> 2.6M raw sample points across all four bars

# One saturated neon hue per catalyst -- distinct from the viridis "raw data"
# thermometer coloring used on the point cloud itself, so the glowing frame
# reads as a separate layer from the glassware it contains.
NEON_COLORS = [
    (0.00, 0.95, 1.00),  # electric cyan
    (1.00, 0.15, 0.75),  # hot magenta
    (1.00, 0.85, 0.00),  # gold
    (0.35, 1.00, 0.25),  # neon green
]


def glow_line(xs, ys, color, base_lw, n_layers=10, alpha_budget=0.3):
    """Draw a crisp core line, then stack progressively wider/fainter copies
    of the same line on top of it so they blend into a soft halo -- the
    mplcyberpunk neon-glow recipe, applied to a bar outline or cap instead
    of a plotted curve."""
    plt.plot(xs, ys, color=color, linewidth=base_lw, alpha=1.0, solid_capstyle="round")
    for i in range(1, n_layers + 1):
        plt.plot(
            xs,
            ys,
            color=color,
            linewidth=base_lw + 1.05 * i,
            alpha=alpha_budget / n_layers,
            solid_capstyle="round",
        )


plt.figure("Gallery - Reaction Yield by Catalyst", figsize=(9, 5))
plt.plot_style("neon")  # true-black background so the glow halos have room to pop

width = 0.8
cap_half = 0.08
cloud_x, cloud_y = [], []
for idx, (cat, mean, err, neon) in enumerate(zip(categories, means, errors, NEON_COLORS)):
    # A faint backdrop rectangle, tinted with the bar's own neon hue -- bar()
    # and yerr= still do real work here -- kept low-alpha so the raw point
    # cloud drawn on top reads as the bar itself, with a bright glowing
    # outline drawn later to frame it.
    plt.bar(
        [cat],
        [mean],
        yerr=[err],
        capsize=cap_half,
        color=(neon[0], neon[1], neon[2], 0.16),
        ecolor=(0.9, 0.95, 1.0, 0.85),
    )
    # The raw measurements a reported mean actually comes from: most of the
    # column fills the silhouette from the baseline up to the mean (biased
    # toward the top, the way repeated trials pile up near the true value),
    # plus a noisier cluster right at the top matching the reported spread.
    fill_n = int(points_per_bar * 0.85)
    spread_n = points_per_bar - fill_n
    x_fill = idx + rng.uniform(-width * 0.42, width * 0.42, fill_n)
    y_fill = mean * rng.uniform(0.0, 1.0, fill_n) ** 0.55
    x_spread = idx + rng.normal(0.0, width * 0.16, spread_n)
    y_spread = np.clip(rng.normal(mean, err, spread_n), 0.0, None)
    cloud_x.append(np.concatenate([x_fill, x_spread]))
    cloud_y.append(np.concatenate([y_fill, y_spread]))

cloud_x = np.concatenate(cloud_x)
cloud_y = np.concatenate(cloud_y)
plt.scatter(
    cloud_x,
    cloud_y,
    c=cloud_y,
    cmap="viridis",
    vmin=0.0,
    vmax=max(means) * 1.05,
    s=0.5,
    alpha=0.4,
)

# Glowing neon-tube frame around each bar, plus glowing error-bar caps. Cheap:
# each bar is just a handful of short polylines redrawn ~10 times (a few
# dozen points total per bar), not the raw million-point cloud -- so the
# 10-layer halo trick stays a crisp glow instead of degrading into generic
# blur or tanking headless render time.
for idx, (mean, err, neon) in enumerate(zip(means, errors, NEON_COLORS)):
    left, right = idx - width / 2, idx + width / 2
    rect_x = [left, left, right, right, left]
    rect_y = [0.0, mean, mean, 0.0, 0.0]
    glow_line(rect_x, rect_y, color=neon, base_lw=2.6)

    top_cap_y = mean + err
    bottom_cap_y = max(mean - err, 0.0)
    glow_line([idx - cap_half, idx + cap_half], [top_cap_y, top_cap_y], color=neon, base_lw=2.2)
    glow_line([idx - cap_half, idx + cap_half], [bottom_cap_y, bottom_cap_y], color=neon, base_lw=2.2)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.title(f"Reaction yield by catalyst (mean ± SD, n = 6 runs; N = {len(cloud_x):,} raw points)")
plt.xlabel("Catalyst")
plt.ylabel("Yield (%)")
# plt.show()
plt.savefig("examples/gallery/results/20_categorical_bar_and_spines.png")
