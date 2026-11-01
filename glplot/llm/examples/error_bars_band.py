"""Measurements with error bars over a model curve with an uncertainty band.

errorbar draws points with +/- bars; fill_between shades between two curves (the band).
"""

import numpy as np

import glplot.pyplot as plt

rng = np.random.default_rng(2)
x = np.linspace(0, 10, 12)
truth = 2.0 * np.exp(-x / 5)
y = truth + rng.normal(scale=0.08, size=x.size)
sigma = np.full_like(x, 0.1)

xs = np.linspace(0, 10, 200)
model = 2.0 * np.exp(-xs / 5)

plt.figure("Fit", figsize=(8, 5))
plt.fill_between(xs, model - 0.15, model + 0.15, alpha=0.25, label="model +/- 0.15")
plt.plot(xs, model, label="model")
plt.errorbar(x, y, yerr=sigma, fmt="o", capsize=0.15, label="data")
plt.title("Decay with uncertainty")
plt.xlabel("time (s)")
plt.ylabel("signal (a.u.)")
plt.legend()
plt.savefig("error_bars_band.png")
