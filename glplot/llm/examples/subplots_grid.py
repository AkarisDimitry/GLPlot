"""A 2 x 2 grid of panels, each with its own kind of plot.

plt.subplots returns (figure, axes) like matplotlib. Every module-level function
(plot, scatter, hist, set_xlabel via ax.set_xlabel...) is also a method of an axes object.
"""

import numpy as np

import glplot.pyplot as plt

rng = np.random.default_rng(1)
x = np.linspace(0, 10, 200)

fig, axs = plt.subplots(2, 2, figsize=(9, 6))
axs[0, 0].plot(x, np.sin(x))
axs[0, 0].set_title("line")
axs[0, 1].scatter(rng.normal(size=300), rng.normal(size=300), s=6)
axs[0, 1].set_title("scatter")
axs[1, 0].hist(rng.normal(size=2000), bins=40)
axs[1, 0].set_title("histogram")
axs[1, 1].bar(["a", "b", "c", "d"], [3, 7, 5, 9])
axs[1, 1].set_title("bars")
plt.suptitle("Four panels")
plt.savefig("subplots_grid.png")
