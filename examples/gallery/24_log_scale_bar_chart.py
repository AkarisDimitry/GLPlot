import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import glplot.pyplot as plt

# A full spectrum-analyzer sweep, not five hand-picked components: 1,500 spectral lines
# ranked by power and obeying a Zipf-like power law (power ~ rank^-alpha), the way real RF
# spectra crowd a strong carrier and a handful of harmonics over a long tail of weak lines
# near the noise floor. On a linear axis that whole tail is an indistinguishable sliver
# next to the carrier. A real log y-axis keeps every one of the 1,500 bars readable down
# to the noise floor; the baseline itself (0) is not representable on a log axis, so each
# bar's bottom edge runs off the bottom of the plot, matching how matplotlib clips a
# log-scaled bar at the axes edge. A Rectangle-patch-per-bar library would already be
# straining at this bar count -- GLPlot adds a full swarm of repeated-measurement samples
# on top, one small jittered dot per virtual sweep, without breaking a sweat.
rng = np.random.default_rng(24)
n_lines = 1_500
rank = np.arange(1, n_lines + 1)
alpha_exp = 1.15
carrier_power = 2.2e-2  # W, rank 1
jitter = 10.0 ** rng.normal(0.0, 0.18, n_lines)  # per-line log-normal scatter
power_w = carrier_power * rank.astype(float) ** -alpha_exp * jitter

log_power = np.log10(power_w)
shade = (log_power - log_power.min()) / (log_power.max() - log_power.min())

# Colored like a real VU meter / spectrum-analyzer display: green for the quiet tail down
# near the noise floor, climbing through yellow to red at the strong end -- signal LEVEL
# drives the color, not bar position, exactly like the LED ladder on a mixing desk.
vu_cmap = LinearSegmentedColormap.from_list("vu_meter", ["#28f06c", "#f5f01a", "#ff2438"])
bar_colors = vu_cmap(0.05 + 0.9 * shade)  # (n_lines, 4)

plt.figure("Gallery - Log-Scale Signal Power", figsize=(9, 5))
plt.plot_style("neon")  # black stage, so the LED caps actually glow

for r, p, color in zip(rank, power_w, bar_colors):
    plt.bar([r], [p], width=0.72, color=tuple(color))

# Peak-hold LED caps: a short, crisp horizontal segment sits exactly on top of each bar --
# the mplcyberpunk glow trick (same segment redrawn progressively wider and more
# transparent so the stacked strokes melt into a soft halo) applied per-bar instead of to
# one big line, so only the top edge lights up, not the whole bar frame. At 1,500 bars this
# is a few thousand two-point plot() calls, nowhere near the million-point series below, so
# it stays cheap even though every single bar gets its own glowing cap.
cap_half_width = 0.36
n_glow = 10
base_lw = 2.0
for r, p, color in zip(rank, power_w, bar_colors):
    x_cap = [r - cap_half_width, r + cap_half_width]
    y_cap = [p, p]
    plt.plot(x_cap, y_cap, color=tuple(color), linewidth=base_lw, alpha=1.0, solid_capstyle="round")
    for i in range(1, n_glow + 1):
        plt.plot(
            x_cap,
            y_cap,
            color=tuple(color),
            linewidth=base_lw + 1.05 * i,
            alpha=0.3 / n_glow,
            solid_capstyle="round",
        )

# Dot-composed overlay: each bar is also a tight swarm of repeated sweep measurements,
# log-normally scattered around its own line's power -- over a million individually
# addressable points riding on top of the 1,500 bars, sharing the same log y-axis.
reps = 1_000
rank_rep = np.repeat(rank, reps)
log_power_rep = np.repeat(log_power, reps)
measured = 10.0 ** (log_power_rep + rng.normal(0.0, 0.05, rank_rep.size))
x_swarm = rank_rep + rng.uniform(-0.3, 0.3, rank_rep.size)
swarm_colors = np.repeat(bar_colors, reps, axis=0).copy()
swarm_colors[:, 3] = 0.05
plt.scatter(x_swarm, measured, c=swarm_colors, s=0.6, edgecolors="none")

plt.yscale("log")
plt.xlabel("Spectral line rank (1 = strongest)")
plt.ylabel("Signal power (W, log scale)")
decades = log_power.max() - log_power.min()
plt.title(f"{n_lines:,} spectral lines, {decades:.1f} decades -- peak-hold LEDs")
# plt.show()
plt.savefig("examples/gallery/results/24_log_scale_bar_chart.png")
