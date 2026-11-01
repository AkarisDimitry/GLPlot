"""Porting a matplotlib script: usually only the import changes.

The same script runs in matplotlib and in GLPlot. Differences worth knowing: GLPlot opens a GPU
window on show(); savefig works headless; an option GLPlot cannot honour emits a one-time
MatplotlibCompatWarning rather than failing.
"""

import numpy as np

import glplot.pyplot as plt  # was: import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, np.sin(x), "r-", lw=2, label="sin")
ax.plot(x, np.cos(x), "b--", label="cos")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Ported")
ax.legend(loc="upper right")
ax.set_xlim(0, 2 * np.pi)
fig.savefig("matplotlib_port.png", dpi=150)
