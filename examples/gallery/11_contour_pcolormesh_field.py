import numpy as np

import glplot.pyplot as plt

# A synthetic gravitational potential surveyed over a 520 x 520 grid, as if
# mapped by a geodetic survey: a smooth regional trend plus a superposed ripple
# from local mass anomalies. contourf() renders the filled field and contour()
# overlays equipotential lines on top of it -- one topographic-map motif.
n = 520
axis = np.linspace(-5, 5, n)
x, y = np.meshgrid(axis, axis)
r = np.hypot(x, y)
field = (
    np.sin(2.8 * x + 0.7 * np.cos(3 * y))
    + np.cos(3.6 * y - 0.5 * np.sin(2 * x))
    + 0.9 * np.sin(4.5 * r) / (1 + 0.08 * r**2)
)

plt.figure("Gallery - Gravitational Potential Map", figsize=(9, 6))
plt.contourf(x, y, field, levels=34, cmap="twilight")
plt.contour(x, y, field, levels=18, colors="white", linewidths=0.35)
plt.title("Surveyed gravitational potential over a 520 x 520 grid")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
# plt.show()
plt.savefig("examples/gallery/results/11_contour_pcolormesh_field.png")
