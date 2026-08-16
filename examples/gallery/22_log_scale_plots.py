import numpy as np

import glplot.pyplot as plt

# Raw Monte-Carlo decay data instead of one smooth analytic curve: millions of
# individual, randomly timed detector counts drawn from a genuine
# non-homogeneous Poisson process with rate(t) = R0 * exp(-t/tau) -- a real
# radioactive-style decay, sampled as discrete *events*, not as a function
# evaluated on a grid. A normal plotting library can scatter a few thousand
# markers like this before it grinds to a halt; GLPlot renders the whole
# multi-million-count stream in one call. semilogy() still does the real
# log-axis work, on the glowing theoretical-rate curve drawn over the cloud.
rng = np.random.default_rng(7)
T = 10.0  # experiment duration, seconds
tau = 2.0  # decay time constant, seconds
R0 = 1_500_000.0  # initial event rate at t=0, counts/s

# The cumulative-rate function Lambda(t) = R0*tau*(1 - exp(-t/tau)) has a
# closed-form inverse, so every event's arrival time is drawn exactly by
# inversion sampling -- no rejection, no looping, fully vectorized even at
# millions of events.
lambda_total = R0 * tau * (1.0 - np.exp(-T / tau))
n_events = int(rng.poisson(lambda_total))
u = rng.uniform(0.0, lambda_total, n_events)
t_events = np.sort(-tau * np.log(1.0 - u / (R0 * tau)))

# Bin the raw arrivals finely, then hand *every event* the (jittered) count
# rate of the bin it landed in -- so each of the millions of points plotted
# is still one real simulated count, not a bin midpoint, and the cloud's
# thickness is the shot noise a real detector's raw stream actually shows.
n_bins = 500
bin_width = T / n_bins
bin_idx = np.clip((t_events / bin_width).astype(np.int64), 0, n_bins - 1)
counts_per_bin = np.bincount(bin_idx, minlength=n_bins)
rate_per_bin = counts_per_bin / bin_width
event_rate = rate_per_bin[bin_idx] * np.exp(rng.normal(0.0, 0.15, n_events))

plt.figure("Gallery - Exponential Decay (Log Scale)", figsize=(9, 5))
plt.plot_style("neon")

# The raw event stream stays as a dense starfield-like cloud drifting through
# black space -- millions of individual detector counts, dimmed just enough
# that the glowing theoretical curve reads as the brightest thing on the page.
plt.scatter(
    t_events,
    event_rate,
    c=event_rate,
    cmap="inferno",
    s=0.25,
    alpha=0.04,
    norm="log",
    label=f"raw MC events (N = {n_events:,})",
)

# Theoretical decay-rate curve: a real "physics data + glowing fit" reference,
# built with the mplcyberpunk glow recipe (stacked, thickening, near-transparent
# copies of the same line fusing into a soft halo around a crisp bright core).
t_ref = np.linspace(0.0, T, 400)
rate_ref = R0 * np.exp(-t_ref / tau)
glow_color = "#ffe066"  # warm gold -- pops hard against the black/inferno cloud

# A ratio-based lower bound (not an absolute one) keeps the underglow a
# thin, constant-width band hugging the line on this log axis, instead of a
# wash that floods the whole plot down to the axis floor.
plt.fill_between(t_ref, rate_ref, rate_ref * 0.6, color=glow_color, alpha=0.1)

base_lw = 1.8
n_layers = 10
for i in range(1, n_layers + 1):
    plt.plot(
        t_ref,
        rate_ref,
        color=glow_color,
        linewidth=base_lw + 1.05 * i,
        alpha=0.3 / n_layers,
        solid_capstyle="round",
    )
plt.semilogy(
    t_ref,
    rate_ref,
    color=glow_color,
    linewidth=base_lw,
    alpha=0.95,
    label="theoretical rate",
)

plt.grid(True)
plt.xlabel("Time (s)")
plt.ylabel("Count rate (counts/s)")
plt.title(f"Monte-Carlo decay events (semilogy, N = {n_events:,})")
plt.legend(loc="upper right")
# plt.show()
plt.savefig("examples/gallery/results/22_log_scale_plots.png")
