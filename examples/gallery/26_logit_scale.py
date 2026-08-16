import colorsys

import numpy as np

import glplot.pyplot as plt

# A pharmacological dose-response curve (a logistic fit to a binary trial outcome) lives
# entirely in (0, 1) and is most informative near the two ends, where a linear axis
# compresses everything into a razor-thin band against 0 or 1. logit is built for exactly
# this: it stretches both tails out and reads almost linearly around p=0.5.
N = 300
dose = np.linspace(-8, 8, N)  # log2(mg) offset from the EC50 dose
p = 1.0 / (1.0 + np.exp(-dose))

# What a normal plotting call never shows: not just the fitted curve, but the raw binary
# data it was fit to. Millions of individual simulated Bernoulli trials at a bank of
# discrete dose levels, jittered off y=0/y=1 so each trial is a real point rather than a
# count -- the outcomes pile up into a probability-mass texture that thins out exactly
# where the logit axis stretches, which is the whole point of the scale.
rng = np.random.default_rng(26)
n_dose_levels = 80
trials_per_level = 30_000
n_trials = n_dose_levels * trials_per_level  # 2,400,000

trial_dose_levels = np.linspace(-8, 8, n_dose_levels)
trial_p = 1.0 / (1.0 + np.exp(-trial_dose_levels))

level_idx = np.repeat(np.arange(n_dose_levels), trials_per_level)
success = rng.random(n_trials) < trial_p[level_idx]

level_spacing = trial_dose_levels[1] - trial_dose_levels[0]
trial_x = trial_dose_levels[level_idx] + rng.uniform(-0.5, 0.5, n_trials) * level_spacing
# Exponentially-distributed offset rather than a uniform one: most trials land right on
# the 0/1 rail and a thinning tail drifts inward, so each dose level's mass reads as an
# organic density glow instead of a hard-edged block. A small additive floor (not a clip)
# keeps every point strictly inside (0, 1) -- logit maps the exact endpoints to +/-inf --
# without creating the artificial pileup-on-the-boundary line a hard clip would draw.
offset = 0.0004 + rng.exponential(0.012, n_trials)
trial_y = np.where(success, 1.0 - offset, offset)

# Soft pastel chalk-stick success/failure colors instead of saturated primaries: the
# same hue family a real stick of chalk comes in, generated the same way as the
# 01_line_plot.py oscilloscope ensemble -- HSV with low saturation and near-full value,
# so the mint and coral read as soft strokes on the slate chalkboard rather than a
# harsh green/red clash. Alpha stays low so the 2.4M-point cloud still builds up as a
# soft density texture instead of curdling into a solid block.
mint_rgb = colorsys.hsv_to_rgb(0.42, 0.34, 0.97)  # pastel mint/green -- "success"
coral_rgb = colorsys.hsv_to_rgb(0.02, 0.40, 0.98)  # pastel coral/pink -- "failure"
trial_colors = np.empty((n_trials, 4), dtype=np.float32)
trial_colors[success] = (*mint_rgb, 0.035)
trial_colors[~success] = (*coral_rgb, 0.035)

plt.figure("Gallery - Dose-Response Curve (Logit Scale)", figsize=(9, 5))
plt.plot_style("chalk")  # slate chalkboard stage for the pastel trial cloud and fit curve
plt.scatter(trial_x, trial_y, c=trial_colors, s=1.4)

# The fitted curve is the "signal" cutting through the mint/coral trial cloud on either
# side, so it gets a soft pastel-gold core plus a real mplcyberpunk-style glow: the same
# curve redrawn several times, each copy a little wider and a lot more transparent, so
# the stacked translucent copies melt into a soft halo around the crisp core. Ten
# layers, linewidth growing by ~1 point per layer, alpha split so all ten only add up
# to 0.3 -- a chalk-gold hue rather than a saturated one so it stays in the pastel
# family while still reading as a distinct third color from the mint/coral cloud.
fit_color = colorsys.hsv_to_rgb(0.12, 0.32, 1.0)  # pastel chalk-gold
base_lw = 2.6
plt.plot(dose, p, color=fit_color, linewidth=base_lw, label="fitted response probability")
n_glow_layers = 10
for i in range(1, n_glow_layers + 1):
    plt.plot(
        dose,
        p,
        color=fit_color,
        linewidth=base_lw + 1.05 * i,
        alpha=0.3 / n_glow_layers,
        solid_capstyle="round",
    )

plt.yscale("logit")
plt.grid(True)
plt.xlabel("Dose (mg, log2 offset from EC50)")
plt.ylabel("Response probability")
plt.title(f"Dose-response fit over {n_trials / 1e6:.1f}M trials (logit scale)")
plt.legend(loc="upper left")
# plt.show()
plt.savefig("examples/gallery/results/26_logit_scale.png")
