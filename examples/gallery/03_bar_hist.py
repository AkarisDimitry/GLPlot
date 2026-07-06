import numpy as np
import glplot.pyplot as plt


rng = np.random.default_rng(7)
core = rng.normal(loc=0.0, scale=0.58, size=520_000)
left_tail = rng.normal(loc=-2.25, scale=0.38, size=260_000)
right_tail = rng.gamma(shape=2.4, scale=0.58, size=320_000) + 0.45
micro_modes = rng.normal(loc=rng.choice([-1.15, 1.65, 2.85], size=160_000), scale=0.16)
values = np.r_[core, left_tail, right_tail, micro_modes]

counts, edges = np.histogram(values, bins=180, range=(-3.8, 4.3))
centers = 0.5 * (edges[:-1] + edges[1:])
width = float(np.diff(edges)[0])
kernel = np.hanning(17)
kernel /= kernel.sum()
smooth = np.convolve(counts, kernel, mode="same")

palette = [
    (0.09, 0.20, 0.54, 0.82),
    (0.02, 0.48, 0.74, 0.82),
    (0.05, 0.68, 0.54, 0.84),
    (0.92, 0.70, 0.20, 0.86),
    (0.88, 0.22, 0.28, 0.82),
]

plt.figure("Gallery - Bars and Histogram", figsize=(9, 5))
for idx, (xc, h) in enumerate(zip(centers, counts)):
    t = idx / max(len(centers) - 1, 1)
    band = min(int(t * len(palette)), len(palette) - 1)
    plt.bar([xc], [h], width=width * 0.98, color=palette[band], alpha=palette[band][3])

x_fill = np.r_[centers[0] - width / 2, centers, centers[-1] + width / 2]
s_fill = np.r_[0.0, smooth, 0.0]
plt.fill_between(x_fill, s_fill, 0, color=(0.02, 0.03, 0.08, 0.16), label="smoothed mass")
plt.plot(x_fill, s_fill, color=(0.01, 0.03, 0.09, 1.0), lw=2.4, label="smoothed envelope")

q05, q50, q95 = np.quantile(values, [0.05, 0.50, 0.95])
plt.axvline(q05, color="white", linestyle="--", linewidth=1.0)
plt.axvline(q50, color="k", linestyle="-", linewidth=1.7, label="median")
plt.axvline(q95, color="white", linestyle="--", linewidth=1.0)
plt.annotate("rare right tail", xy=(q95, smooth[np.argmin(np.abs(centers - q95))]), xytext=(2.1, counts.max() * 0.92), arrowprops={"color": "tab:red", "width": 1.0}, color="tab:red")
plt.title("Spectral histogram of 1.26M multimodal samples")
plt.xlabel("value")
plt.ylabel("count")
plt.grid(True)
plt.legend()
plt.show()
plt.savefig("examples/gallery/results/03_bar_hist.png")
