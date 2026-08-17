"""Streamgraph flow -- a storm's swell spectrum, banded by frequency and centred.

An ocean wave buoy does not report a single "wave height": it reports a
*spectrum*, the wave energy broken out by frequency, because a real sea state
is a superposition of many wave trains of different periods riding on top of
each other -- long, slow groundswell from a distant storm mixed with short,
choppy wind waves raised locally. As a storm builds and fades offshore, that
spectrum's shape visibly breathes: energy piles up near whichever frequency
the dominant swell currently sits at, spreads or narrows as the sea state
organises or scatters, and the total energy across every band rises through
the storm's peak and falls away after.

This animation bins a synthetic buoy spectrum into ``N_BANDS`` frequency
bands (0.05-0.24 Hz, the usual wind-wave/swell range) and evolves each band's
energy from one shared, slowly wandering "storm state" -- a dominant
frequency, a spectral width, and an overall intensity, each its own smoothed
random walk plus a slow sinusoid -- so neighbouring bands swell and shrink
together the way a real moving spectral peak would, with a little independent
per-band texture noise layered on top so no two bands are ever identical.

The classic streamgraph trick is the *baseline*: instead of stacking every
band from ``y = 0`` the way an ordinary stacked-area chart does, each frame's
stack is re-centred by offsetting every band by ``-total(t) / 2`` so the whole
silhouette floats symmetrically around ``y = 0`` -- the "wiggle" look, drawn
here as ``N_BANDS`` separate :func:`~glplot.pyplot.fill_between` polygon
fills (cheap: tens of fills per frame, not a per-point scatter), one flowing
band per frequency, coloured across a single warm-to-cool hue sweep so the
stack itself reads as a small rainbow keyed to frequency. Every frame slides
a fixed-width time window one step forward across a precomputed storm
history, so old seconds scroll off the left edge as new ones enter on the
right.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import hsv_to_rgb

import glplot.animation as animation
import glplot.pyplot as plt

N_BANDS = 10
WINDOW = 200  # rolling buffer width, in time steps
FRAMES = 84
DT = 0.5  # simulated seconds per time step
TOTAL_STEPS = WINDOW + FRAMES

rng = np.random.default_rng(11)

# Frequency bins a real wave buoy might report: 0.05-0.24 Hz spans roughly 4-20 s
# periods, i.e. long groundswell up through short wind chop.
band_freq = np.linspace(0.05, 0.24, N_BANDS)


def build_storm_history(total_steps: int):
    """Precompute the full spectral-energy history once (one random trajectory).

    Three shared, slowly wandering "storm state" variables -- dominant frequency
    ``f0``, spectral width ``sigma``, and overall intensity -- each combine a slow
    sinusoid with a mean-reverting random walk (an AR(1) process: ``x <- 0.96*x +
    noise``, which stays bounded instead of drifting off forever). Every band's
    energy is a Gaussian bump of that shared storm centred on its own frequency,
    plus a little independent per-band noise so neighbouring bands move together
    but are never identical -- exactly how a real moving spectral peak would look
    sampled into discrete frequency bins.
    """
    times = np.empty(total_steps)
    energy = np.empty((N_BANDS, total_steps))
    f0_hist = np.empty(total_steps)

    walk_amp = 0.0
    walk_f0 = 0.0
    band_noise = np.zeros(N_BANDS)

    for k in range(total_steps):
        t = k * DT

        walk_amp = 0.96 * walk_amp + rng.normal(scale=0.05)
        storm = 1.3 + 0.55 * np.sin(2 * np.pi * t / 42.0 + 0.6)
        storm += 0.35 * np.sin(2 * np.pi * t / 19.0) + walk_amp
        storm = max(storm, 0.15)

        walk_f0 = 0.96 * walk_f0 + rng.normal(scale=0.004)
        f0 = 0.15 + 0.055 * np.sin(2 * np.pi * t / 70.0) + walk_f0
        f0 = float(np.clip(f0, band_freq[0], band_freq[-1]))

        sigma = 0.045 + 0.016 * np.sin(2 * np.pi * t / 33.0 + 1.1)
        sigma = max(sigma, 0.02)

        band_noise = 0.9 * band_noise + rng.normal(scale=0.035, size=N_BANDS)
        bump = storm * np.exp(-((band_freq - f0) ** 2) / (2.0 * sigma * sigma))
        vals = bump + 0.10 * storm + band_noise  # small pedestal keeps every band alive
        vals = np.clip(vals, 0.03, None)  # a streamgraph's bands can never go negative

        times[k] = t
        energy[:, k] = vals
        f0_hist[k] = f0

    return times, energy, f0_hist


times_full, energy_full, f0_full = build_storm_history(TOTAL_STEPS)

# Fixed vertical scale, sized from the whole precomputed run, so the frame doesn't
# rescale itself frame to frame -- only the bands' shapes move.
half_total_max = (energy_full.sum(axis=0) / 2.0).max()
Y_CAP = half_total_max * 1.15

# One distinct hue per band, swept warm-to-cool across the frequency axis so the
# stack itself reads as a small rainbow: low frequency (long groundswell) cool blue,
# high frequency (choppy wind waves) warm red.
hues = np.linspace(0.62, 0.01, N_BANDS)
hsv = np.stack([hues, np.full(N_BANDS, 0.72), np.full(N_BANDS, 0.93)], axis=1)
band_rgb = hsv_to_rgb(hsv)
band_colors = [(*band_rgb[i], 0.93) for i in range(N_BANDS)]

fig = plt.figure("Gallery - Streamgraph Flow", figsize=(9, 6))
plt.plot_style("solarized")


def update(frame: int):
    lo, hi = frame, frame + WINDOW
    t_win = times_full[lo:hi]
    v_win = energy_full[:, lo:hi]

    # Ordinary stacked cumulative sums, then the streamgraph baseline: shift every
    # band down by half the column's total so the whole silhouette is centred on
    # y = 0 instead of built up from a hard y = 0 floor.
    cum = np.cumsum(v_win, axis=0)
    total = cum[-1]
    baseline = -total / 2.0
    lower = baseline + np.vstack([np.zeros((1, WINDOW)), cum[:-1]])
    upper = baseline + cum

    plt.cla()
    for i in range(N_BANDS):
        plt.fill_between(t_win, upper[i], lower[i], color=band_colors[i])

    plt.xlim(t_win[0], t_win[-1])
    plt.ylim(-Y_CAP, Y_CAP)
    f0_now = f0_full[hi - 1]
    plt.title(
        f"Storm swell spectrum -- t={t_win[0]:.0f}-{t_win[-1]:.0f} s, "
        f"peak {f0_now:.2f} Hz ({1.0 / f0_now:.1f} s period)"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Spectral wave energy (a.u.)")
    return []


ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=42)
ani.save("examples/gallery/animations/results/14_streamgraph_flow.gif", fps=20)
# plt.show()
