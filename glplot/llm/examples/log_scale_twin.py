"""Log axes and a second y axis (a Bode plot).

xscale/yscale accept 'linear', 'log', 'symlog', 'asinh' and 'logit'; values must be positive on
a log axis. twinx() returns a second axes that shares x and has its own y scale -- draw into the
returned object, not into plt.
"""

import numpy as np

import glplot.pyplot as plt

f = np.logspace(0, 4, 200)
gain = 1 / np.sqrt(1 + (f / 300) ** 2)
phase = -np.degrees(np.arctan(f / 300))

plt.figure("Bode", figsize=(8, 5))
plt.plot(f, gain, color="tab:blue", label="gain")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("frequency (Hz)")
plt.ylabel("gain", color="tab:blue")

ax2 = plt.twinx()
ax2.plot(f, phase, color="tab:red", label="phase")
ax2.set_ylabel("phase (deg)", color="tab:red")
plt.title("First-order low-pass")
plt.savefig("log_scale_twin.png")
