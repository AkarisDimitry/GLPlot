import numpy as np

import glplot.pyplot as plt

rng = np.random.default_rng(1800)
rows = np.arange(-12, 13)
cols = np.arange(-12, 13)
q, r = np.meshgrid(cols, rows)
x = q + 0.5 * (r % 2)
y = r * np.sqrt(3) / 2
rad = np.hypot(x, y)
height = 0.45 + 4.6 * np.exp(-0.016 * rad**2) * (0.5 + 0.5 * np.cos(0.8 * x - 1.1 * y) ** 2)
height += 0.28 * rng.random(height.shape)

plt.figure("Gallery - Hexagonal 3D Bars", figsize=(9, 6), ssao=True)
plt.bar3d(
    x.ravel(),
    y.ravel(),
    np.zeros(x.size),
    0.72,
    0.72,
    height.ravel(),
    c=height.ravel(),
    cmap="YlOrRd",
    alpha=0.96,
    edge_color=(0.18, 0.06, 0.0, 0.52),
    edge_width=0.56,
    gap=0.14,
    shape="hex",
    ssao=True,
    ssao_strength=0.36,
    label="hex bars",
)
plt.title("Hexagonal 3D bars with edges and SSAO")
plt.xlabel("x")
plt.ylabel("y")
plt.zlabel("height")
plt.legend()
plt.show()
plt.savefig("examples/gallery/results/18_hex_bars3d.png")
