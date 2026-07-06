#!/usr/bin/env python3
"""
Colorful 2D Particle Burst - Simple & Beautiful Showcase

100,000 vibrant particles arranged in concentric circles with
smooth color gradients. Fully interactive at 60+ FPS.

Controls: Pan with mouse drag, zoom with scroll wheel
"""

import numpy as np
import glplot.pyplot as plt

# Create colorful particle burst
n = 100_000
angles = np.linspace(0, 2 * np.pi, n)
distances = np.sqrt(np.linspace(0, 1, n)) * 10
x = distances * np.cos(angles)
y = distances * np.sin(angles)

# Vibrant rainbow colors based on angle and distance
colors = np.mod(angles / np.pi + distances / 5, 1)  # Smooth color gradient

# Create plot with beautiful styling
plt.figure("✨ Particle Burst", figsize=(10, 10))
plt.scatter(x, y, c=colors, cmap="rainbow", s=2, alpha=0.8)
plt.xlim(-12, 12)
plt.ylim(-12, 12)
plt.title("100,000 Particles @ 60+ FPS", fontsize=14, color="white")
plt.grid(True, alpha=0.3)
plt.axis("equal")
plt.show()
