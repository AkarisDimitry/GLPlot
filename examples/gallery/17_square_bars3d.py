import os

import numpy as np

import glplot.pyplot as plt

# Synthetic downtown skyline: a downtown core of tall towers rises out of a
# citywide grid of shorter blocks, whose heights ripple with a checkerboard
# district pattern plus small per-lot construction variation.
rng = np.random.default_rng(1700)
grid = np.arange(-18, 19)
x, y = np.meshgrid(grid, grid)
r = np.hypot(x, y)
district = np.abs(np.sin(0.33 * x) * np.cos(0.27 * y))
downtown = np.exp(-0.0035 * r**2)
height = 0.4 + 1.8 * district + 3.2 * downtown
height += 0.3 * rng.random(height.shape)
n_buildings = x.size

plt.figure("Gallery - Square 3D Bars (City Skyline)", figsize=(9, 6), ssao=True)
plt.bar3d(
    x.ravel(),
    y.ravel(),
    np.zeros(x.size),
    0.62,
    0.62,
    height.ravel(),
    c=height.ravel(),
    cmap="autumn",
    alpha=0.97,
    edge_color=(0.35, 0.16, 0.05, 0.35),
    edge_width=0.5,
    gap=0.16,
    shape="box",
    elev=32,
    azim=-38,
    ssao=True,
    ssao_strength=0.26,
)
plt.title(f"Synthetic downtown skyline: {n_buildings} square building footprints on a city grid")
plt.xlabel("Grid X (m)")
plt.ylabel("Grid Y (m)")
plt.zlabel("Building height (m)")
plt.savefig("examples/gallery/results/17_square_bars3d.png")
if os.environ.get("GLPLOT_SHOW_GUI") == "1":
    plt.show(density=True)
plt.show(density=True)
