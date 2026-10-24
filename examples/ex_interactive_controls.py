"""Signal-analysis bench: GLPlot's native controls driving a live analysis.

Run it and the window opens with a Controls panel beside the plot:

* **Signal** -- tone frequency (log slider), amplitude, 50 Hz mains hum, noise level, seed.
* **Filter** -- Butterworth low/high/band-pass; the controls that do not apply to the
  chosen type grey out. Order commits on release (``live=False``): drag freely.
* **Analysis** -- a time window (two cursors on the plot follow it) and a Results table:
  RMS, SNR before/after filtering and the dominant frequency inside the window.
* Buttons -- *New noise* (Mod+Alt+N), *Autoscale*, *Reset all* (asks first). All three
  are also in the command palette (Mod+K), under "Controls".

Right-click any control to reset it or copy its value; the toolbar saves named presets.

    python examples/ex_interactive_controls.py
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

import glplot.pyplot as plt

FS = 1000.0  # sample rate, Hz
t = np.arange(0.0, 4.0, 1.0 / FS)

fig = plt.figure("Signal analysis bench", figsize=(14, 8))
plt.title("Filtered signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

# -- controls ---------------------------------------------------------------------------

with plt.control_group("Signal"):
    tone = plt.slider("Tone", 0.5, 50.0, 5.0, log=True, unit="Hz", help="The clean tone")
    amplitude = plt.slider("Amplitude", 0.0, 2.0, 1.0, step=0.05)
    hum = plt.checkbox("Mains hum 50 Hz", True, help="Adds a 0.4-amplitude 50 Hz tone")
    noise = plt.slider("Noise sigma", 0.0, 1.0, 0.3, step=0.01)
    seed = plt.number("Seed", 0, integer=True, min=0)

with plt.control_group("Filter"):
    kind = plt.dropdown(
        "Type", {"Low-pass": "lowpass", "High-pass": "highpass", "Band-pass": "bandpass"}
    )
    cutoff = plt.slider("Cut-off", 0.5, 200.0, 15.0, log=True, unit="Hz")
    band = plt.range_slider("Band", 0.5, 200.0, (2.0, 20.0), step=0.5)
    order = plt.slider("Order", 1, 8, 4, step=1, integer=True, live=False)
    show_raw = plt.checkbox("Show raw signal", True)
    colour = plt.color_picker("Filtered colour", "tab:orange")

with plt.control_group("Analysis"):
    window = plt.range_slider("Window", 0.0, 4.0, (0.5, 3.5), step=0.01, fmt="%.2f s")

# -- the model --------------------------------------------------------------------------


def clean_signal(f: float, a: float) -> np.ndarray:
    return a * np.sin(2 * np.pi * f * t)


def measured_signal(f: float, a: float, with_hum: bool, sigma: float, rng_seed: int):
    y = clean_signal(f, a)
    if with_hum:
        y = y + 0.4 * np.sin(2 * np.pi * 50.0 * t)
    return y + np.random.default_rng(int(rng_seed)).normal(0.0, sigma, t.size)


def butterworth(y: np.ndarray, btype: str, fc: float, pass_band, n: int) -> np.ndarray:
    if btype == "bandpass":
        lo, hi = pass_band
        if hi - lo < 0.5:
            raise ValueError("the pass band is empty: widen Band")
        wn = (lo, hi)
    else:
        wn = fc
    sos = sps.butter(int(n), wn, btype=btype, fs=FS, output="sos")
    return sps.sosfiltfilt(sos, y)


def snr_db(estimate: np.ndarray, truth: np.ndarray) -> float:
    residual = estimate - truth
    return float(10 * np.log10(np.mean(truth**2) / max(np.mean(residual**2), 1e-20)))


# -- live views -------------------------------------------------------------------------

(raw_line,) = plt.plot(
    t, measured_signal(5.0, 1.0, True, 0.3, 0), color="0.65", lw=0.8, label="Raw"
)
left_cursor = plt.axvline(window.value[0], color="tab:green", ls="--", label="Analysis window")
right_cursor = plt.axvline(window.value[1], color="tab:green", ls="--", label="Analysis window")

signal_controls = dict(f=tone, a=amplitude, with_hum=hum, sigma=noise, rng_seed=seed)
filter_controls = dict(btype=kind, fc=cutoff, pass_band=band, n=order)


@plt.interact(
    plot_kwargs={"color": "tab:orange", "linewidth": 1.8}, **signal_controls, **filter_controls
)
def filtered(f, a, with_hum, sigma, rng_seed, btype, fc, pass_band, n):
    """Returns (t, y): GLPlot draws it as a curve and keeps it updated."""
    y = measured_signal(f, a, with_hum, sigma, rng_seed)
    raw_line.set_ydata(y)
    return t, butterworth(y, btype, fc, pass_band, n)


@plt.interact(**signal_controls, **filter_controls, span=window)
def analysis(f, a, with_hum, sigma, rng_seed, btype, fc, pass_band, n, span):
    """Returns a dict: shown in the Controls panel's Results table."""
    left_cursor.set_xdata([span[0]])
    right_cursor.set_xdata([span[1]])
    if filtered.result is None:  # the filter is failing; its error is on screen
        return {"Status": "no filtered signal (see the error above)"}
    sel = (t >= span[0]) & (t <= span[1])
    truth = clean_signal(f, a)[sel]
    raw = raw_line.get_ydata()[sel]
    clean = filtered.result[1][sel]
    spectrum = np.abs(np.fft.rfft(clean * np.hanning(clean.size)))
    freqs = np.fft.rfftfreq(clean.size, 1.0 / FS)
    return {
        "Samples in window": int(sel.sum()),
        "RMS raw": float(np.sqrt(np.mean(raw**2))),
        "RMS filtered": float(np.sqrt(np.mean(clean**2))),
        "SNR raw [dB]": snr_db(raw, truth),
        "SNR filtered [dB]": snr_db(clean, truth),
        "Dominant frequency [Hz]": float(freqs[1:][np.argmax(spectrum[1:])]),
    }


# -- reactions that are not "recompute a curve" -----------------------------------------


@kind.on_change
def _grey_out_unused(btype):
    band.enabled = btype == "bandpass"
    cutoff.enabled = btype != "bandpass"


_grey_out_unused(kind.value)
show_raw.on_change(lambda visible: setattr(raw_line.style, "visible", bool(visible)))
colour.on_change(lambda rgba: plt.setp(filtered.layer, color=rgba))


@plt.button("New noise", shortcut="Mod+Alt+N", help="Draw a fresh noise realisation")
def new_noise():
    seed.value = seed.value + 1


plt.button("Autoscale", lambda: fig.autoscale(), help="Frame every curve")
plt.button(
    "Reset all",
    lambda: plt.controls().reset(),
    confirm="Reset every control to its default value?",
)

plt.legend()
plt.xlim(0, 4)
plt.ylim(-3, 3)

if __name__ == "__main__":
    plt.show()
