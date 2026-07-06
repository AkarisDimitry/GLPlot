import numpy as np
import glplot.pyplot as plt

rng = np.random.default_rng(606)
x = np.arange(1600)
trend = 0.002 * x + 0.75 * np.sin(x / 33.0) + 0.35 * np.sin(x / 7.5)
y = trend + rng.normal(scale=0.18, size=len(x))
events = np.arange(80, len(x), 145)
event_strength = 1.3 + 0.45 * np.sin(events / 50)
y[events] += event_strength

plt.figure("Gallery - Signal Tools", figsize=(9, 5))
plt.step(x, y, where="mid", color="tab:blue", lw=0.9, label="1600-sample step signal")
plt.errorbar(
    x[::32],
    y[::32],
    yerr=0.18 + 0.05 * np.sin(x[::32] / 9),
    fmt="ko",
    ecolor="tab:red",
    capsize=0.05,
    label="sampled uncertainty",
)
plt.stem(
    events,
    y[events] - 1.2,
    linefmt="C2-",
    markerfmt="C2o",
    basefmt="k-",
    bottom=-1.5,
    label="detected events",
)
plt.annotate(
    "largest spike",
    xy=(events[np.argmax(y[events])], np.max(y[events])),
    xytext=(900, 3.8),
    arrowprops={"color": "tab:red", "width": 1.1},
    color="tab:red",
)
plt.xlabel("sample")
plt.ylabel("value")
plt.title("Long signal with uncertainty and sparse event stems")
plt.grid(True)
plt.legend()
plt.show()
plt.savefig("examples/gallery/results/06_signal_tools.png")
