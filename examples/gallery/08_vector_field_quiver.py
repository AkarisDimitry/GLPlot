import numpy as np
import glplot.pyplot as plt


grid = np.linspace(-3.0, 3.0, 31)
x, y = np.meshgrid(grid, grid)
r2 = x**2 + y**2 + 0.2
u = -y / r2 + 0.15 * np.sin(3 * y)
v = x / r2 + 0.15 * np.cos(3 * x)
speed = np.hypot(u, v)

plt.figure("Gallery - Vector Field", figsize=(8, 6))
plt.imshow(speed, extent=(-3, 3, -3, 3), cmap="magma", alpha=0.72)
plt.quiver(x, y, u, v, color="white", scale=0.42, width=0.7, head_width=0.055, head_length=0.09, label="flow")
plt.axhline(0, color="cyan", linestyle="--", linewidth=1)
plt.axvline(0, color="cyan", linestyle="--", linewidth=1)
plt.annotate("vortex core", xy=(0, 0), xytext=(-2.6, 2.5), arrowprops={"color": "cyan", "width": 1.2, "head_width": 0.16, "head_length": 0.22}, color="cyan")
plt.title("Vector field over density background")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.savefig("examples/gallery/results/08_vector_field_quiver.png")
