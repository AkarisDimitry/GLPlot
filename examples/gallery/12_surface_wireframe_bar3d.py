import numpy as np
import glplot.pyplot as plt


axis = np.linspace(-3.5, 3.5, 155)
x, y = np.meshgrid(axis, axis)
r = np.hypot(x, y) + 1e-6
z = 1.8 * np.sin(3.4 * r) / (1 + 0.22 * r**2) + 0.32 * np.cos(2.5 * x) * np.sin(2.0 * y)

bar_x = np.linspace(-3.0, 3.0, 9)
bar_y = np.full_like(bar_x, 3.25)
bar_z = np.zeros_like(bar_x)
bar_h = 0.45 + 1.35 * (np.sin(bar_x * 1.7) ** 2)

plt.figure("Gallery - 3D Surface", figsize=(9, 6))
plt.plot_surface(x, y, z, cmap="turbo", elev=28, azim=-48, scale_z=0.92, rstride=2, cstride=2, s=2.6, alpha=0.92, label="projected surface")
plt.plot_wireframe(x, y, z + 0.035, elev=28, azim=-48, scale_z=0.92, rstride=12, cstride=12, color=(0, 0, 0, 0.33), linewidth=0.45)
plt.bar3d(bar_x, bar_y, bar_z, 0.28, 0.28, bar_h, color=(1.0, 0.95, 0.15, 0.9), elev=28, azim=-48, scale_z=0.92, label="3D bars")
plt.title("Projected 3D surface, wireframe, and bars")
plt.xlabel("projected x")
plt.ylabel("projected y")
plt.legend()
#plt.show()
plt.savefig("examples/gallery/results/12_surface_wireframe_bar3d.png")
