import colorsys

import numpy as np

import glplot.pyplot as plt

# Chalk white -- matches the built-in "chalk" style preset's own ink color
# (glplot/gui/styles.py), used below for the one reference line that needs to
# read clearly against the chalkboard background plot_style("chalk") paints.
CHALK_INK = (0.949, 0.937, 0.902)


def chalk_color(hue: float, rng: np.random.Generator) -> tuple:
    """A pastel chalk-stick colour at ``hue`` -- soft saturation, near-full value."""
    sat = rng.uniform(0.22, 0.42)
    val = rng.uniform(0.90, 1.0)
    return colorsys.hsv_to_rgb(hue % 1.0, sat, val)


rng = np.random.default_rng(101)
n_channels = 10_072  # 72 original channels + 10,000 more, per the revision request
n_points = 180  # per-line resolution: the headless PNG export replays every
# polyline through its own individual matplotlib ax.plot() call (see
# _draw_layers() in glplot/utils/preview.py), so 10k+ full-resolution lines
# would make the static export impractically slow. The line COUNT stays at
# the requested 10,000-more; only points-per-line is trimmed.
t = np.linspace(0, 40 * np.pi, n_points)
t_ms = t * (50.0 / (40 * np.pi))  # a single 50 ms oscilloscope sweep window
stimulus = np.sin(t) + 0.35 * np.sin(2.7 * t)

plt.figure("Gallery - Oscilloscope Ensemble", figsize=(9, 5))
plt.plot_style("chalk", layers=False)  # chalkboard background, both live and in this export

# Fractions of the way through the ensemble, not raw indices: this keeps the
# frequency/amplitude/offset spread identical to the original 72-channel
# design no matter how many channels are packed into it, so 10,072 densely
# overlapping traces read as one dense "chalk dust" ensemble instead of a
# plot whose axes have blown out to a diagonal streak.
frac = np.linspace(0.0, 1.0, n_channels)
hues = (rng.uniform(0, 1) + frac + rng.uniform(-0.02, 0.02, n_channels)) % 1.0

for i in range(n_channels):
    t_frac = frac[i]
    phase = rng.uniform(0, 2 * np.pi)
    amp = 0.12 + 0.781 * t_frac
    freq = 1.1 + 0.852 * t_frac
    offset = 3.195 * t_frac
    channel = stimulus + amp * np.sin(freq * t + phase)
    color = chalk_color(hues[i], rng)
    alpha = rng.uniform(0.30, 0.46)
    plt.plot(t_ms, channel + offset, color=(*color, alpha), lw=0.5)

reference = stimulus + 0.7
plt.plot(t_ms, reference, color=(*CHALK_INK, 1.0), lw=3.0, label="shared stimulus (reference)")
plt.xlabel("Time (ms)")
plt.ylabel("Signal amplitude (a.u.)")
plt.title(f"Steady-state traces from {n_channels} nominally identical oscillator channels")
plt.grid(True)
plt.legend()
# plt.show()
plt.savefig("examples/gallery/results/01_line_plot.png")
