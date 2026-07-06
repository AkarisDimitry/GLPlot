import numpy as np

import glplot.pyplot as plt

n = 520
axis = np.linspace(-5, 5, n)
x, y = np.meshgrid(axis, axis)
r = np.hypot(x, y)
field = (
    np.sin(2.8 * x + 0.7 * np.cos(3 * y))
    + np.cos(3.6 * y - 0.5 * np.sin(2 * x))
    + 0.9 * np.sin(4.5 * r) / (1 + 0.08 * r**2)
)

plt.figure("Gallery - Contours and Pcolormesh", figsize=(9, 6))
plt.contourf(x, y, field, levels=34, cmap="twilight")
plt.contour(x, y, field, levels=18, colors="white", linewidths=0.35)
plt.pcolormesh(x[::3, ::3], y[::3, ::3], field[::3, ::3], cmap="viridis", alpha=0.2)
plt.axline((-4, -2), slope=0.45, color="cyan", linewidth=1.6, label="analysis slice")
plt.annotate(
    "phase knot",
    xy=(0.3, -0.2),
    xytext=(-4.2, 4.2),
    arrowprops={"color": "cyan", "width": 1.0},
    color="cyan",
)
plt.title("520 x 520 scalar field with contours")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
# plt.show()
plt.savefig("examples/gallery/results/11_contour_pcolormesh_field.png")
