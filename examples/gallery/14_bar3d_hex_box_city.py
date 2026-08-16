import numpy as np

import glplot.pyplot as plt

rng = np.random.default_rng(1414)

# Synthetic downtown skyline: a city grid where building height decays
# radially from a downtown core and is modulated by a street-grid zoning
# ripple, with small random architectural variation per lot. Footprints
# alternate between square and hexagonal towers in a checkerboard, so the
# result reads as one coherent "bar city" built from two tower silhouettes.
block_spacing = 20.0  # m, distance between adjacent city lots
story_height = 8.5  # m per story

grid = np.arange(-9, 10)
gx, gy = np.meshgrid(grid, grid)
r = np.hypot(gx, gy)
stories = 0.4 + 4.5 * np.exp(-0.045 * r**2) * (0.55 + 0.45 * np.sin(0.9 * gx) ** 2)
stories += 0.45 * rng.random(stories.shape)
height_m = stories * story_height

x = gx * block_spacing
y = gy * block_spacing

mask_hex = (gx + gy) % 2 == 0
mask_box = ~mask_hex
n_buildings = gx.size

plt.figure("Gallery - Synthetic City Skyline (Box + Hex Towers)", figsize=(9, 6), ssao=True)
plt.bar3d(
    x[mask_box],
    y[mask_box],
    np.zeros(mask_box.sum()),
    0.72 * block_spacing,
    0.72 * block_spacing,
    height_m[mask_box],
    c=height_m[mask_box],
    cmap="YlOrRd",
    vmin=height_m.min(),
    vmax=height_m.max(),
    alpha=0.97,
    edge_color=(0.35, 0.16, 0.05, 0.35),
    edge_width=0.48,
    gap=0.14,
    shape="box",
    elev=32,
    azim=-38,
    ssao=True,
    ssao_strength=0.26,
    label="square towers",
)
plt.bar3d(
    x[mask_hex],
    y[mask_hex],
    np.zeros(mask_hex.sum()),
    0.78 * block_spacing,
    0.78 * block_spacing,
    height_m[mask_hex],
    c=height_m[mask_hex],
    cmap="YlOrRd",
    vmin=height_m.min(),
    vmax=height_m.max(),
    alpha=0.97,
    edge_color=(0.35, 0.16, 0.05, 0.35),
    edge_width=0.52,
    gap=0.13,
    shape="hex",
    elev=32,
    azim=-38,
    ssao=True,
    ssao_strength=0.26,
    label="hex towers",
)
plt.title(f"Synthetic downtown skyline: N = {n_buildings} square + hexagonal towers")
plt.xlabel("Position, east-west (m)")
plt.ylabel("Position, north-south (m)")
plt.zlabel("Building height (m)")
plt.legend()
# plt.show()
plt.savefig("examples/gallery/results/14_bar3d_hex_box_city.png")
