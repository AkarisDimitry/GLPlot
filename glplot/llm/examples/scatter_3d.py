"""A coloured 3-D point cloud.

scatter3d takes x, y, z and the same c=/cmap=/s= options as scatter. A live window handles
millions of points; rotate with the mouse.
"""

import numpy as np

import glplot.pyplot as plt

rng = np.random.default_rng(3)
t = rng.uniform(0, 6 * np.pi, 3000)
x = np.cos(t) * (1 + 0.1 * rng.normal(size=t.size))
y = np.sin(t) * (1 + 0.1 * rng.normal(size=t.size))
z = t / (2 * np.pi)

plt.figure("Helix", figsize=(7.5, 6))
plt.scatter3d(x, y, z, c=z, cmap="plasma", s=3, elev=20, azim=-50)
plt.title("Noisy helix")
plt.xlabel("x")
plt.ylabel("y")
plt.zlabel("turns")
plt.savefig("scatter_3d.png")
