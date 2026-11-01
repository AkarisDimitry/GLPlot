"""Categorical bars: with error bars, and grouped.

bar takes category names directly; grouped_bar takes a dict of series name -> values (or one
list per series). Inside a multi-panel figure call the methods of the axes (axs[i].bar...);
plt.subplot() with a different grid shape would rebuild the grid and empty the panels.
"""

import glplot.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))

cats = ["north", "south", "east", "west"]
axs[0].bar(cats, [12, 7, 9, 15], yerr=[1.0, 0.8, 1.2, 1.5], capsize=0.1)
axs[0].set_title("with error bars")
axs[0].set_ylabel("count")

axs[1].grouped_bar({"2023": [3, 5, 4], "2024": [4, 6, 6]}, tick_labels=["A", "B", "C"])
axs[1].set_title("grouped")
axs[1].legend()
plt.savefig("bars_categorical.png")
