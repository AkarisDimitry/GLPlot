import time

import numpy as np

import glplot.pyplot as gplt

# Stress Test: 50,000,000 lines
N = 50_000_000
print(f"STRESS TEST: Generating {N:,} lines. This will consume ~400MB of GPU memory.")

# Generate data clusters
a = np.random.normal(0, 1.0, N)
b = np.random.normal(0, 1.0, N)

# Optimization Setup
# We enable:
# - HUD: To monitor performance (FPS, budgets)
# - Budgeting: To ensure interactivity even during massive draw calls
# - Clipping: To avoid drawing lines outside the view
gplt.figure(
    "Expert Performance Stress Test (50M Lines)",
    hud=True,
    budget=5,  # Very aggressive budget for fluid interaction
    clipping=True,
    cache=True,
)

# Advanced Tuning: Fine-tune the Hybrid Cache
# - Increase refresh rate for more up-to-date impostors
# - Increase padding to reduce refresh frequency during slow pans
gplt.options(cache_refresh_hz=60.0, cache_padding=4.0, default_global_alpha=0.1)

gplt.plot_lines(a, b, x_range=(-5, 5))

print("\n--- PERFORMANCE GUIDE ---")
print("1. Interactive Path: Drag the mouse. The engine uses a reprojected cache.")
print("2. Exact Path: Release the mouse. The engine draws the exact geometry (LOD).")
print("3. HUD: Observe the FPS and Line Count in the top-left overlay.")

gplt.show()
