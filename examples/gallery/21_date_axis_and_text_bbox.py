import datetime

import matplotlib.dates as mdates
import numpy as np
from matplotlib import colormaps

import glplot.pyplot as plt

# Synthetic backstory: a weather station that logs a reading every few seconds
# across Jan-Apr 2024 -- millions of raw high-frequency samples, each with its
# own diurnal wobble and sensor noise. This turns it into a RIDGELINE / joyplot:
# one smoothed "shape of a day" trace per week, all drawn over the *same* 24-hour
# window and stacked bottom-to-top with a small y offset, color-graded across a
# wide multi-hue palette as the season turns from January to April -- but instead
# of showing *only* the 16 clean smoothed traces (which reads as a small, simple
# dataset no matter how big N really is), each ridge is redrawn underneath as a
# faint scatter of its own actual raw readings, so the millions of individual
# samples behind the average are visibly there, not just claimed in the title.
# The shared window is still real `datetime.datetime` values run through
# `matplotlib.dates.date2num` -- same documented mechanism as the original
# file's `datetime.date` x-axis, just standing in for "hour of day" instead of
# "day of year": the offset that stacks the ridges lives on the y-axis only.
rng = np.random.default_rng(3)
n_weeks = 16
n_bins = 96  # 15-minute resolution across the shared 24h window
offset_step = 3.2  # y-axis stacking step per ridge (see amplitude note below)
days = [datetime.date(2024, 1, 1) + datetime.timedelta(days=7 * i) for i in range(n_weeks)]

# A guaranteed winter -> spring warming drift (small random-walk wobble on top,
# so it is not a perfectly straight ramp) drives both which week gets called
# "warmest" and the cool-blue -> warm-red ridge coloring below.
warm_drift = np.linspace(1.0, 15.0, n_weeks)
wobble = np.cumsum(rng.normal(0, 1.0, n_weeks)) * 0.4
weekly_trend_c = warm_drift + wobble

# Each week gets its own diurnal "personality" -- longer, sunnier spring days
# swing wider than hazy January ones, plus a per-week phase jitter and a
# second harmonic (a rough stand-in for cloud cover) -- so the 16 ridges are
# clearly related but not sixteen identical clones of one sine wave.
diurnal_amp = np.linspace(2.6, 4.4, n_weeks) + rng.normal(0, 0.15, n_weeks)
phase_jitter = rng.normal(0, 0.05, n_weeks)
harmonic_amp = rng.uniform(0.3, 0.9, n_weeks)
harmonic_phase = rng.uniform(0, 2 * np.pi, n_weeks)

# The high-frequency record itself: one reading roughly every 4 seconds for the
# full 112-day span (n_weeks * 7 days) -- still millions of raw samples, each
# one binned below by which week it fell in and what time of day it was,
# rather than scattered on screen point-by-point.
n_readings = 3_000_000
t_numeric = np.sort(rng.uniform(0.0, n_weeks * 7.0, n_readings))
week_of_reading = np.clip((t_numeric // 7.0).astype(int), 0, n_weeks - 1)
day_fraction = t_numeric % 1.0
trend_at_t = np.interp(t_numeric, np.arange(n_weeks) * 7.0, weekly_trend_c)
diurnal = diurnal_amp[week_of_reading] * np.sin(
    2.0 * np.pi * (day_fraction - 0.3 + phase_jitter[week_of_reading])
) + harmonic_amp[week_of_reading] * np.sin(
    4.0 * np.pi * day_fraction + harmonic_phase[week_of_reading]
)
sensor_noise = rng.normal(0.0, 0.9, n_readings)
reading_c = trend_at_t + diurnal + sensor_noise

# Bin all 3,000,000 readings into (week, time-of-day) cells and average the
# *detrended* signal (diurnal wobble + sensor noise, weekly trend subtracted
# back out) -- with roughly two thousand raw readings landing in every one of
# the 16 * 96 cells, sensor noise all but cancels in the mean, leaving a clean
# per-week "shape of a day" trace without needing scipy for a real KDE.
hour_bin = np.clip((day_fraction * n_bins).astype(int), 0, n_bins - 1)
flat_bin = week_of_reading * n_bins + hour_bin
n_flat = n_weeks * n_bins
sums = np.bincount(flat_bin, weights=reading_c - trend_at_t, minlength=n_flat)
counts = np.bincount(flat_bin, minlength=n_flat)
bin_mean = (sums / np.maximum(counts, 1)).reshape(n_weeks, n_bins)


def _smoothed(y: np.ndarray, radius: int = 3, sigma: float = 1.3) -> np.ndarray:
    """A small Gaussian moving average -- the "smoothed trace" a ridgeline needs,
    without pulling in scipy just for a KDE."""
    xs = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(y, radius, mode="edge"), kernel, mode="valid")


# One shared 24-hour x-window for every ridge -- an arbitrary but real calendar
# day (`datetime.datetime` objects, not raw floats), reused unchanged by every
# week's curve below.
ref_day = datetime.datetime.combine(days[0], datetime.time())
hour_centers = (np.arange(n_bins) + 0.5) / n_bins * 24.0
x_dt = np.array([ref_day + datetime.timedelta(hours=float(h)) for h in hour_centers])
x_num = mdates.date2num(x_dt)

plt.figure("Gallery - Date Axis and Text Bbox", figsize=(9, 5))
plt.plot_style("neon")  # black background: the glow only pops against real dark

# "turbo" swings blue -> teal -> green -> gold -> red across the same Jan->Apr warming
# order "coolwarm" used, but it does it through *four* visibly distinct hue families
# instead of coolwarm's pale, nearly-white midpoint -- adjacent weeks read as different
# colors, not just different shades of the same two.
cmap = colormaps["turbo"]
warmest_two = set(np.argsort(weekly_trend_c)[-2:].tolist())
peak_week = int(np.argmax(weekly_trend_c))
peak_x = peak_y = None

# `t_numeric` is sorted, so `week_of_reading` is monotonic -- each week's raw readings
# occupy one contiguous slice of the 3,000,000-row arrays, found here with two
# `searchsorted` calls instead of an O(N) boolean mask per week.
week_bounds = np.searchsorted(t_numeric, np.arange(n_weeks + 1) * 7.0)
ref_day_num = mdates.date2num(ref_day)
n_raw_sample = 3_500  # per week -> 56,000 raw dots total across the figure

for w in range(n_weeks):
    trace = _smoothed(bin_mean[w])
    baseline = w * offset_step
    y = trace + baseline
    color = cmap(w / (n_weeks - 1))

    # fill_between() triangulates the band internally, and the headless export
    # strokes every triangle's own edges at the layer's default line_width --
    # visible as a faint diagonal hatch where adjacent triangles' shared edges
    # get drawn (and thus made twice as opaque) on top of each other. Zeroing
    # the width on the layer this call hands back keeps the fill a clean flat
    # wash instead.
    band = plt.fill_between(x_num, y, baseline, color=color, alpha=0.1)
    band.style.line_width = 0.0

    # The smoothed trace above is an *average* of ~187,500 raw readings per week --
    # without this, the plot just looks like 16 clean analytic curves and the "N =
    # 3,000,000" in the title is an unverifiable claim. Drawing a random sample of that
    # week's actual (still-noisy, still-diurnal, not-yet-averaged) readings as a faint
    # scatter puts the real data density back on screen: each dot lands at its own
    # true time-of-day and its own detrended reading, so the cloud naturally hugs and
    # thickens/thins around the smoothed line by exactly as much as the raw signal
    # really varies -- no synthetic jitter needed, it's already noisy.
    lo, hi = week_bounds[w], week_bounds[w + 1]
    raw_idx = rng.choice(np.arange(lo, hi), size=min(n_raw_sample, hi - lo), replace=False)
    raw_x = ref_day_num + day_fraction[raw_idx]
    raw_y = baseline + (reading_c[raw_idx] - trend_at_t[raw_idx])
    plt.scatter(raw_x, raw_y, color=color, s=1.4, alpha=0.05, edgecolors="none")

    label = f"warmest week ({days[w].strftime('%b %d')})" if w == peak_week else None
    # `plot()` coerces this real datetime array to date2num units internally --
    # the ridge line itself is what still demonstrates dates plotted directly.
    plt.plot(x_dt, y, color=color, linewidth=1.8, alpha=0.95, label=label)

    if w == peak_week:
        top_bin = int(np.argmax(trace))
        peak_x, peak_y = x_dt[top_bin], y[top_bin]

    # The full 10-layer glow recipe (mplcyberpunk's actual technique): the same
    # curve redrawn progressively thicker and fainter so the stacked translucent
    # copies blend into a soft halo around the crisp core already drawn above --
    # applied to every ridge now (not just the two warmest) so the whole figure
    # reads as a genuine neon display, not a plain rainbow with two glowing
    # exceptions. The two warmest ridges keep a second, brighter glow pass on
    # top so the spring end of the season still visibly stands out.
    n_layers = 10
    for i in range(1, n_layers + 1):
        plt.plot(
            x_num,
            y,
            color=color,
            linewidth=1.8 + 1.05 * i,
            alpha=0.22 / n_layers,
            solid_capstyle="round",
        )
    if w in warmest_two:
        for i in range(1, n_layers + 1):
            plt.plot(
                x_num,
                y,
                color=color,
                linewidth=1.8 + 1.05 * i,
                alpha=0.3 / n_layers,
                solid_capstyle="round",
            )

# GLPlot has no live date-tick *formatter* yet (the axis holds date2num floats
# underneath), so labeling a handful of round hours by hand is the same
# workaround the original file used for calendar days -- just at clock-time
# resolution now that the shared x-window is "a day" rather than "the season".
tick_hours = [0, 6, 12, 18, 24]
tick_dt = [ref_day + datetime.timedelta(hours=h) for h in tick_hours]
plt.xticks(mdates.date2num(tick_dt), [d.strftime("%H:%M") for d in tick_dt])
# Text has no data extent of its own for matplotlib's autoscale to consider (unlike a
# Line2D), so a label placed *outside* the plotted data's range renders outside the
# visible axes box, whatever the axis limits end up being -- a small enough offset to
# stay within the default 5% autoscale margin keeps it next to the point it's labeling.
#
# `legend()` always renders "upper right" in this headless export (it has no
# `loc=` of its own to move it, unlike matplotlib), so the annotation is
# anchored down and to the left of the peak rather than above it -- next to
# what it is labeling without the two boxes colliding.
plt.text(
    peak_x - datetime.timedelta(hours=2.5),
    peak_y - 2.2,
    "warmest week",
    fontsize=14,
    bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray"),
)
plt.title(f"Ridgeline of weekly diurnal cycles, Jan-Apr 2024 (N = {n_readings:,} readings)")
plt.xlabel("Time of day")
plt.ylabel("Temperature swing (°C)")
plt.legend()
# plt.show()
plt.savefig("examples/gallery/results/21_date_axis_and_text_bbox.png")
