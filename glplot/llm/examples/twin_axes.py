"""Two y axes on one plot (twinx).

twinx() returns a second axes that shares x and has its own y scale; draw into the returned
object, not into plt (module-level plt.plot after twinx() still targets the first axes).
"""

import numpy as np

import glplot.pyplot as plt

t = np.linspace(0, 10, 300)

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(t, np.exp(-t / 4), color="tab:blue", label="concentration")
ax1.set_xlabel("time (h)")
ax1.set_ylabel("concentration (mg/L)", color="tab:blue")

ax2 = ax1.twinx()
ax2.plot(t, 20 + 15 * np.sin(t / 2), color="tab:red", label="temperature")
ax2.set_ylabel("temperature (C)", color="tab:red")
plt.savefig("twin_axes.png")
