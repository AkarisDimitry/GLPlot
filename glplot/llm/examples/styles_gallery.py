"""The same plot in several style presets.

plot_style('name') restyles the whole current figure (page, palette, line width, grid,
colormap), so one figure per preset. plot_styles() lists every preset with a description.
"""

import numpy as np

import glplot.pyplot as plt

x = np.linspace(0, 6, 200)

for preset in ["clean", "dark", "chalk", "neon"]:
    plt.figure(preset, figsize=(6, 4))
    plt.plot_style(preset)
    for k in range(4):
        plt.plot(x, np.sin(x + k * 0.7) / (1 + 0.2 * k), label=f"series {k}")
    plt.title(f"style: {preset}")
    plt.xlabel("x")
    plt.ylabel("amplitude")
    plt.legend()
    plt.savefig(f"styles_{preset}.png")

print(
    f"{len(plt.plot_styles())} presets; the first three:", [row[0] for row in plt.plot_styles()[:3]]
)
