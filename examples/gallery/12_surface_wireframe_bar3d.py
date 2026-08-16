import numpy as np

import glplot.pyplot as plt

# Standing-wave interference pattern on a vibrating membrane, sampled on a
# 155 x 155 grid, with its amplitude also flattened onto the floor plane
# beneath it -- matplotlib's classic ax.contourf(..., zdir='z', offset=...)
# "shadow" technique, built here from a dense floor-level point cloud coloured
# by the same field (GLPlot's contourf() does not support 3D axes yet). One
# coherent "magma" palette ties the surface, its wireframe scaffold, and the
# floor projection together.
axis = np.linspace(-3.5, 3.5, 155)
x, y = np.meshgrid(axis, axis)
r = np.hypot(x, y) + 1e-6
z = 1.8 * np.sin(3.4 * r) / (1 + 0.22 * r**2) + 0.32 * np.cos(2.5 * x) * np.sin(2.0 * y)

floor_z = np.full_like(z, -2.3)

plt.figure("Gallery - Interference Surface + Floor Projection", figsize=(9, 6))
plt.plot_surface(
    x,
    y,
    z,
    cmap="magma",
    elev=28,
    azim=-48,
    scale_z=0.92,
    rstride=2,
    cstride=2,
    s=2.6,
    alpha=0.92,
)
plt.plot_wireframe(
    x,
    y,
    z + 0.035,
    elev=28,
    azim=-48,
    scale_z=0.92,
    rstride=12,
    cstride=12,
    color=(0, 0, 0, 0.28),
    linewidth=0.45,
)
plt.scatter3d(
    x.ravel(),
    y.ravel(),
    floor_z.ravel(),
    c=z.ravel(),
    cmap="magma",
    s=13,
    alpha=0.9,
    elev=28,
    azim=-48,
    scale_z=0.92,
)
plt.title("Standing-wave interference on a 155x155 membrane, projected onto the floor")
plt.xlabel("Position x (mm)")
plt.ylabel("Position y (mm)")
plt.zlabel("Displacement z (mm)")
# plt.show()
plt.savefig("examples/gallery/results/12_surface_wireframe_bar3d.png")
