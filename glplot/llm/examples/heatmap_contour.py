"""A 2-D field as an image with contour lines and a colorbar.

imshow takes a 2-D array (row 0 is the top, like matplotlib); contour and contourf take
x, y, z grids. Use pcolormesh when the grid is not uniform.
"""

import numpy as np

import glplot.pyplot as plt

x = np.linspace(-3, 3, 200)
y = np.linspace(-2, 2, 150)
X, Y = np.meshgrid(x, y)
Z = np.sin(X * 2) * np.cos(Y * 3) * np.exp(-(X**2 + Y**2) / 4)

plt.figure("Field", figsize=(8, 5.5))
plt.imshow(Z, extent=(-3, 3, -2, 2), cmap="RdBu", origin="lower", aspect="auto")
plt.colorbar(label="z")
plt.contour(X, Y, Z, levels=8, colors="k", linewidths=0.6)
plt.title("z(x, y)")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("heatmap_contour.png")
