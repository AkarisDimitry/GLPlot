"""Line plot with several series, a legend and labelled axes.

The smallest complete GLPlot script: figure, plot, label, save. Use plt.show() instead of
plt.savefig() for an interactive window (pan, zoom, right-click menus).
"""

import numpy as np

import glplot.pyplot as plt

t = np.linspace(0, 4 * np.pi, 400)

plt.figure("Waves", figsize=(8, 5))
plt.plot(t, np.sin(t), label="sin(t)")
plt.plot(t, np.cos(t), "--", label="cos(t)")
plt.plot(t, np.sin(t) * np.exp(-t / 6), color="tab:red", lw=2, label="damped")
plt.title("Three waves")
plt.xlabel("t (s)")
plt.ylabel("amplitude")
plt.grid(True)
plt.legend()
plt.savefig("line_plot.png")  # plt.show() opens the interactive window instead
