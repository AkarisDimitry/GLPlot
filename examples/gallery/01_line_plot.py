import numpy as np
import glplot.pyplot as plt


rng = np.random.default_rng(101)
x = np.linspace(0, 40 * np.pi, 12_000)
base = np.sin(x) + 0.35 * np.sin(2.7 * x)

plt.figure("Gallery - Dense Lines", figsize=(9, 5))
for i in range(42):
    phase = rng.uniform(0, 2 * np.pi)
    amp = 0.12 + 0.015 * i
    y = base + amp * np.sin((1.1 + i * 0.018) * x + phase)
    plt.plot(x, y + 0.055 * i, color=(0.05, 0.35 + 0.01 * i, 0.95, 0.17), lw=0.7)

envelope = base + 0.7
plt.plot(x, envelope, "r-", label="upper signal ridge", lw=2)
plt.plot(x[::80], (base - 0.75)[::80], "ko", label="sampled markers", ms=2.4)
plt.xlabel("x")
plt.ylabel("signal")
plt.title("42 layered high-resolution line series")
plt.grid(True)
plt.legend()
plt.show()
plt.savefig("examples/gallery/results/01_line_plot.png")

