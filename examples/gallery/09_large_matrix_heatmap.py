import numpy as np
import glplot.pyplot as plt

n = 700
axis = np.linspace(-4, 4, n)
x, y = np.meshgrid(axis, axis)
matrix = (
    np.sin(7 * x) * np.cos(5 * y)
    + 0.55 * np.sin(11 * np.hypot(x, y))
    + 0.25 * np.cos(18 * np.arctan2(y, x))
)

plt.figure("Gallery - Large Matrix", figsize=(8, 6))
plt.imshow(matrix, extent=(-4, 4, -4, 4), cmap="viridis")
plt.contour_note = "matrix metadata is retained on the image layer"
plt.title("700 x 700 procedural matrix")
plt.xlabel("column coordinate")
plt.ylabel("row coordinate")
# plt.show()
plt.savefig("examples/gallery/results/09_large_matrix_heatmap.png")
