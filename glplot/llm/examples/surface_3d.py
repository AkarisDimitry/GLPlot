"""A 3-D surface with a chosen camera.

3-D plots are separate functions (plot_surface, scatter3d, plot3d, bar3d, quiver3d...) or any
call on axes made with projection='3d'. view_init sets elevation and azimuth in degrees.
"""

import numpy as np

import glplot.pyplot as plt

x = np.linspace(-3, 3, 80)
X, Y = np.meshgrid(x, x)
Z = np.sin(np.hypot(X, Y)) * np.exp(-0.1 * (X**2 + Y**2))

plt.figure("Surface", figsize=(8, 6))
plt.plot_surface(X, Y, Z, cmap="viridis")
plt.view_init(elev=30, azim=-60)
plt.title("sin(r) e^(-r^2/10)")
plt.xlabel("x")
plt.ylabel("y")
plt.zlabel("z")
plt.savefig("surface_3d.png")
