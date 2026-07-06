import numpy as np
import glplot.pyplot as plt

axis = np.linspace(-4.0, 4.0, 45)
x, y = np.meshgrid(axis, axis)
r = np.hypot(x, y)
z = 0.3 + 3.8 * np.exp(-0.075 * r**2) * (0.45 + 0.55 * np.sin(1.8 * x) ** 2)

plt.figure("Gallery - SSAO Comparison", figsize=(10, 6), ssao=True)
plt.bar3d(
    x.ravel() - 5.0,
    y.ravel(),
    np.zeros(x.size),
    0.13,
    0.13,
    z.ravel(),
    color=(0.2, 0.62, 1.0, 0.86),
    edge_color=(0.0, 0.0, 0.0, 0.35),
    edge_width=0.35,
    shape="box",
    ssao=False,
    label="SSAO off",
)
plt.bar3d(
    x.ravel() + 5.0,
    y.ravel(),
    np.zeros(x.size),
    0.13,
    0.13,
    z.ravel(),
    color=(1.0, 0.68, 0.12, 0.9),
    edge_color=(0.0, 0.0, 0.0, 0.52),
    edge_width=0.4,
    shape="box",
    ssao=True,
    ssao_strength=0.85,
    label="SSAO on",
)
plt.title("SSAO comparison on dense 3D bars")
plt.xlabel("x")
plt.ylabel("y")
plt.zlabel("height")
plt.legend()
plt.show()
plt.savefig("examples/gallery/results/16_ssao_comparison.png")
