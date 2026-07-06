import numpy as np

import glplot.pyplot as plt

rng = np.random.default_rng(4)
n = 1_220_000
theta = rng.uniform(0, 18 * np.pi, n)
radius = 0.025 * theta + rng.gamma(1.5, 0.08, n)
x_cloud = radius * np.cos(theta) + rng.normal(scale=0.07, size=n)
y_cloud = radius * np.sin(theta) + rng.normal(scale=0.07, size=n)
score = theta + 6 * radius

x = np.linspace(-2.8, 2.8, 1400)
upper = 1.35 * np.exp(-0.22 * x**2) + 0.18 * np.sin(7 * x)
lower = -1.15 * np.exp(-0.24 * x**2) + 0.12 * np.cos(5 * x)

plt.figure("Gallery - Scatter and Fill", figsize=(9, 6))
plt.fill_between(x, upper, lower, color="tab:blue", alpha=0.18, label="confidence band")
plt.scatter(x_cloud, y_cloud, c=score, cmap="turbo", s=1.5, alpha=0.72, label="220k spiral samples")
plt.plot(x, upper, "w-", lw=1.2)
plt.plot(x, lower, "w-", lw=1.2)
plt.title("220k-point scatter over filled nonlinear band")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
# plt.show()
plt.savefig("examples/gallery/results/02_scatter_fill.png")
