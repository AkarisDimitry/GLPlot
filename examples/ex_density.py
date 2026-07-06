import os
import sys

import numpy as np

# Force local glplot import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glplot.pyplot as gplt

factor = 1000

# Generate 10 million lines y = a*x + b
N = 10_000_000
print(f"Generating {N} lines. This tests memory throughput and GPU geometry performance.")
a = np.random.randn(N) * 2.0
b = np.random.randn(N) * 0.5 * factor

# Assign a random color with transparency
colors = np.random.rand(N, 4)
colors[:, 3] = 0.05  # alpha

# Configure figure with optimizations
gplt.figure(
    "Massive Density Map (10M Lines)",
    width=1280,
    height=800,
    density=True,
    lod=True,
    budget=200,  # High budget for dense detail
    clipping=True,
    hud=True,
)

# Performance Tuning: Render density at 0.5x resolution for massive speedup
# This is ideal when N is very large or GPU is limited.
gplt.options(density_resolution_scale=0.5, density_gain=1.5)

gplt.plot_lines(a, b, x_range=(-5, 5), colors=colors)

gplt.text(-4.5, 4.0, "10,000,000 Lines", fontsize=32, color="white")
gplt.text(-4.5, 3.5, "Optimized: 0.5x Resolution Scale", fontsize=20, color="gray")
gplt.xlim(-5, 5)
gplt.ylim(-factor, factor)
print("\nControls:")
print("  [ D ]            : Toggle Density/Exact view.")
print("  [ UP/DOWN ]      : Adjust density gain.")
gplt.show()
