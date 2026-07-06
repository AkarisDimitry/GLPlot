import os

import numpy as np

import glplot.pyplot as plt

rng = np.random.default_rng(1700)
grid = np.arange(-14, 15)
x, y = np.meshgrid(grid, grid)
r = np.hypot(x, y)
height = 0.35 + 5.0 * np.exp(-0.018 * r**2) * (0.35 + 0.65 * np.sin(0.7 * x + 0.4 * y) ** 2)
height += 0.25 * rng.random(height.shape)

plt.figure("Gallery - Square 3D Bars", figsize=(9, 6), ssao=True)
plt.bar3d(
    x.ravel(),
    y.ravel(),
    np.zeros(x.size),
    0.62,
    0.62,
    height.ravel(),
    c=height.ravel(),
    cmap="viridis",
    alpha=0.96,
    edge_color=(0.0, 0.12, 0.20, 0.52),
    edge_width=0.5,
    gap=0.16,
    shape="box",
    ssao=True,
    ssao_strength=0.36,
    label="square bars",
)
plt.title("Square 3D bars with edges and SSAO")
plt.xlabel("x")
plt.ylabel("y")
plt.zlabel("height")
plt.legend()
plt.savefig("examples/gallery/results/17_square_bars3d.png")
if os.environ.get("GLPLOT_SHOW_GUI") == "1":
    plt.show(density=True)
plt.show(density=True)
