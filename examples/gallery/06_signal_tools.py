import numpy as np

import glplot.pyplot as plt

# Synthetic backstory: an 8 s voltage trace sampled at 200 Hz, with sparse
# threshold-crossing events flagged by an onboard detector. The full 1600-sample
# trace compressed the step/errorbar/stem detail into illegible fuzz, so only a
# 2 s window (the one holding three detected events) is actually plotted -- xlim()
# does not crop the headless export (autoscale always wins there), so the zoom
# has to come from slicing the arrays themselves, not from a view-limit call.
rng = np.random.default_rng(606)
n_samples = 1600
sample_rate_hz = 200.0
sample_idx = np.arange(n_samples)
t_full = sample_idx / sample_rate_hz
trend = 0.4 * np.sin(t_full / 0.165) + 0.35 * np.sin(t_full / 0.0375) + 0.01 * t_full
voltage_full = trend + rng.normal(scale=0.18, size=n_samples)
events_full = np.arange(80, n_samples, 145)
event_strength = 1.3 + 0.45 * np.sin(events_full / 50)
voltage_full[events_full] += event_strength

window_samples = 400  # 2 s at 200 Hz
t = t_full[:window_samples]
voltage = voltage_full[:window_samples]
events = events_full[events_full < window_samples]

signal_color = "#2f6fed"

plt.figure("Gallery - Signal Tools", figsize=(9, 5))
plt.step(t, voltage, where="mid", color=signal_color, lw=1.4, label="voltage trace (2 s window)")
plt.errorbar(
    t[::16],
    voltage[::16],
    yerr=0.18 + 0.05 * np.sin(t[::16] * 22),
    fmt="o",
    color="#444444",
    ecolor="#999999",
    markersize=5,
    capsize=0.015,
    label="sampled uncertainty",
)
plt.stem(
    t[events],
    voltage[events] - 1.2,
    linefmt="C3-",
    markerfmt="C3o",
    basefmt="k-",
    bottom=-1.5,
    label="detected events",
)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("Voltage trace: 2 s window (of an 8 s, 200 Hz recording) with detected events")
plt.grid(True)
plt.legend()
# plt.show()
plt.savefig("examples/gallery/results/06_signal_tools.png")
