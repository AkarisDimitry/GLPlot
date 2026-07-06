import numpy as np
import glplot.pyplot as plt


rng = np.random.default_rng(77)
n = 180_000
t = rng.uniform(0, 24 * np.pi, n)
radius = 0.08 * t + 0.7 * rng.random(n)
x = radius * np.cos(t)
y = radius * np.sin(t)
z = 0.08 * t + rng.normal(scale=0.35, size=n)

plt.figure("Gallery - Projected 3D Cloud", figsize=(9, 6))
plt.scatter3d(x, y, z, c=z, cmap="turbo", s=1.2, alpha=0.75, elev=24, azim=-42, scale_z=0.85, label="180k points")
plt.plot3d(0.08 * t[:3000] * np.cos(t[:3000]), 0.08 * t[:3000] * np.sin(t[:3000]), 0.08 * t[:3000], "k-", lw=1.2, elev=24, azim=-42, scale_z=0.85, label="spine")
plt.title("Projected 3D spiral cloud")
plt.xlabel("projected x")
plt.ylabel("projected y")
plt.legend()
plt.show()
plt.savefig("examples/gallery/results/07_projected_3d_cloud.png")
