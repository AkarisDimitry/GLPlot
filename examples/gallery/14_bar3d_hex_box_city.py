import numpy as np

import glplot.pyplot as plt

rng = np.random.default_rng(1414)
grid = np.arange(-9, 10)
x, y = np.meshgrid(grid, grid)
r = np.hypot(x, y)
height = 0.4 + 4.5 * np.exp(-0.045 * r**2) * (0.55 + 0.45 * np.sin(0.9 * x) ** 2)
height += 0.45 * rng.random(height.shape)

mask_hex = (x + y) % 2 == 0
mask_box = ~mask_hex

plt.figure("Gallery - 3D Bars Box and Hex", figsize=(9, 6), ssao=True)
plt.bar3d(
    x[mask_box],
    y[mask_box],
    np.zeros(mask_box.sum()),
    0.72,
    0.72,
    height[mask_box],
    c=height[mask_box],
    cmap="summer",
    alpha=0.94,
    edge_color=(0.0, 0.12, 0.18, 0.48),
    edge_width=0.48,
    gap=0.14,
    shape="box",
    elev=32,
    azim=-38,
    ssao=True,
    ssao_strength=0.38,
    label="box bars",
)
plt.bar3d(
    x[mask_hex],
    y[mask_hex],
    np.zeros(mask_hex.sum()),
    0.78,
    0.78,
    height[mask_hex],
    c=height[mask_hex],
    cmap="Wistia",
    alpha=0.95,
    edge_color=(0.20, 0.08, 0.0, 0.48),
    edge_width=0.52,
    gap=0.13,
    shape="hex",
    elev=32,
    azim=-38,
    ssao=True,
    ssao_strength=0.38,
    label="hex bars",
)
plt.plot3d(
    [-9, 9],
    [-9, 9],
    [0, 5],
    color="white",
    linewidth=2,
    elev=32,
    azim=-38,
    label="diagonal profile",
)
plt.title("Mixed box and hexagonal 3D bar field")
plt.xlabel("x")
plt.ylabel("y")
plt.zlabel("height")
plt.legend()
plt.show()
plt.savefig("examples/gallery/results/14_bar3d_hex_box_city.png")
