"""Audio spectrum analyzer -- an animated equalizer display, green->yellow->red by level.

A real-time spectrum analyzer splits an audio signal into frequency bands (usually via an
FFT) and shows each band's instantaneous power as a bar, refreshed dozens of times a
second. This example synthesizes that behaviour rather than reading a real waveform: each
of 64 bands gets its own slowly-drifting envelope, built from a few superposed sine
components at different rates and phases (so neighbouring bands don't pulse in lockstep,
the way a real mix's bass/mid/treble energy drifts independently) plus a periodic bass
"kick" pulse that punches the low bands upward, the way a drum hit does on a real meter.

Bands are laid out on a log frequency axis (60 Hz - 16 kHz, the classic audio range) and
coloured by their *current* level on a green -> yellow -> red ramp, exactly like a VU meter:
quiet bands read green, loud ones shade toward red. A thin bright cap above each bar is a
peak-hold indicator -- it jumps to a new peak instantly but falls back slowly, the way
analog and digital meters both damp their peak marker so a transient spike stays readable
for a few frames instead of vanishing immediately.

Bars are cheap patches (not literal per-frame scatter marks), so 64 bars/frame redraw in a
fraction of a second regardless of frame count -- the "many points" here comes from the
band count x frame count x the peak-hold history, not from a single expensive draw call.
"""

import numpy as np
from matplotlib import colormaps

import glplot.animation as animation
import glplot.pyplot as plt

rng = np.random.default_rng(3)

N_BANDS = 64
FRAMES = 84
FLOOR_DB = -60.0  # meter floor -- silence reads as this, not zero height
CEIL_DB = 0.0  # 0 dBFS -- full scale

# Log-spaced band centre frequencies, 60 Hz to 16 kHz -- the range a consumer graphic
# equalizer covers, laid out the way a real one is: bass on the left, treble on the right.
band_freq = 60.0 * (16000.0 / 60.0) ** (np.arange(N_BANDS) / (N_BANDS - 1))
band_x = np.arange(N_BANDS)

# Each band's baseline loudness: a steep 1/f-ish tilt (bass mixes hotter than treble in
# most music) plus fixed per-band jitter so the resting spectrum isn't a perfectly smooth
# curve -- real program material never is. Tuned to sit mostly green/yellow at rest, so a
# bass kick or a bright transient reading red actually reads as an event, not the default.
base_level_db = -22.0 - 24.0 * (np.arange(N_BANDS) / (N_BANDS - 1)) + rng.normal(0.0, 2.0, N_BANDS)

# Per-band modulation: three sine components at different rates/phases per band, so bands
# drift independently instead of breathing in unison -- the same trick a granular-synthesis
# envelope follower uses to avoid a robotic, uniform pulse.
rate_a = rng.uniform(0.06, 0.14, N_BANDS)
rate_b = rng.uniform(0.15, 0.30, N_BANDS)
rate_c = rng.uniform(0.03, 0.07, N_BANDS)
phase_a = rng.uniform(0, 2 * np.pi, N_BANDS)
phase_b = rng.uniform(0, 2 * np.pi, N_BANDS)
phase_c = rng.uniform(0, 2 * np.pi, N_BANDS)
depth_db = rng.uniform(6.0, 10.0, N_BANDS)

# Bass "kick" -- a periodic envelope, strongest at the lowest few bands and fading out
# across the band range, like a real drum transient's spectral energy does. Kept tight to
# the bass end so the mids/highs stay green/yellow and only the low end punches into red.
KICK_PERIOD = 14.0
kick_weight = np.exp(-band_x / 6.0)

cmap = colormaps["RdYlGn"]  # 0 -> red, 1 -> green; we invert per-bar below

peak_db = np.full(N_BANDS, FLOOR_DB)  # peak-hold state, persists across frames
PEAK_DECAY_DB = 1.6  # dB the peak marker falls per frame once no new peak arrives


def band_levels_db(frame: int) -> np.ndarray:
    """Synthesized per-band signal power (dB), one frame of a fake spectrum analyzer."""
    t = frame
    wobble = (
        0.5 * np.sin(2 * np.pi * rate_a * t + phase_a)
        + 0.3 * np.sin(2 * np.pi * rate_b * t + phase_b)
        + 0.2 * np.sin(2 * np.pi * rate_c * t + phase_c)
    )
    level = base_level_db + depth_db * wobble

    kick_phase = (t % KICK_PERIOD) / KICK_PERIOD
    kick_env = max(0.0, np.sin(np.pi * kick_phase)) ** 6  # sharp attack, quick decay
    level = level + 20.0 * kick_env * kick_weight

    return np.clip(level, FLOOR_DB, CEIL_DB)


fig = plt.figure("Gallery - Spectrum Analyzer", figsize=(9, 5.6))
plt.plot_style("neon")


def update(frame: int):
    global peak_db

    level_db = band_levels_db(frame)

    # Peak-hold: jump up instantly on a new peak, otherwise fall slowly -- the standard
    # ballistic-meter behaviour that keeps a transient spike visible for a few frames.
    peak_db = np.maximum(level_db, peak_db - PEAK_DECAY_DB)

    frac = (level_db - FLOOR_DB) / (CEIL_DB - FLOOR_DB)  # 0..1, current loudness
    heights = level_db - FLOOR_DB  # bar height above the meter floor

    plt.cla()
    for i in range(N_BANDS):
        color = cmap(1.0 - frac[i])  # invert: quiet=green, loud=red
        plt.bar(
            [band_x[i]],
            [heights[i]],
            width=0.82,
            bottom=FLOOR_DB,
            color=color,
        )
        # Peak-hold cap: a thin bright sliver sitting at the held peak level.
        peak_h = peak_db[i] - FLOOR_DB
        plt.bar(
            [band_x[i]],
            [1.1],
            width=0.82,
            bottom=FLOOR_DB + peak_h - 1.1,
            color=(1.0, 1.0, 1.0, 0.9),
        )

    plt.xlim(-1, N_BANDS)
    plt.ylim(FLOOR_DB - 2, CEIL_DB + 4)

    tick_idx = np.linspace(0, N_BANDS - 1, 6).astype(int)

    def _hz_label(hz: float) -> str:
        return f"{hz / 1000:.1f}k" if hz >= 1000 else f"{hz:.0f}"

    plt.xticks(band_x[tick_idx], [_hz_label(band_freq[i]) for i in tick_idx])

    peak_dbfs = level_db.max()
    plt.title(f"64-band spectrum analyzer -- frame {frame:03d}  (peak {peak_dbfs:+.1f} dB)")
    plt.xlabel("Frequency band (Hz)")
    plt.ylabel("Signal power (dB)")
    return []


ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=40)
# plt.show()
ani.save("examples/gallery/animations/results/03_spectrum_analyzer_bars.gif", fps=22)
