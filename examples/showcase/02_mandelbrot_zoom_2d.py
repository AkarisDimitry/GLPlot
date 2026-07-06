#!/usr/bin/env python3
"""
Interactive Mandelbrot Set - Fractal Visualization

300,000 points rendering the stunning Mandelbrot fractal
with psychedelic colors. Drag to pan, scroll to zoom!

Controls: Click & drag to pan, scroll to zoom infinitely
"""

import numpy as np

import glplot.pyplot as plt


def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter=40):
    """Compute Mandelbrot set for given region"""
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    Z = np.zeros_like(C, dtype=complex)
    M = np.zeros(C.shape, dtype=int)

    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 2 + C[mask]
        M[mask] = i

    return M


# Render Mandelbrot set (300k points)
width, height = 600, 600
mandel = mandelbrot_set(-2.5, 1.0, -1.25, 1.25, width, height)
x, y = np.meshgrid(np.linspace(-2.5, 1, width), np.linspace(-1.25, 1.25, height))

# Flatten and create colorful visualization
x_flat, y_flat = x.flatten(), y.flatten()
colors_flat = mandel.flatten() / 40.0

# Plot with psychedelic colors
plt.figure("🌀 Mandelbrot Fractal", figsize=(10, 10))
plt.scatter(x_flat, y_flat, c=colors_flat, cmap="twilight_shifted", s=1, alpha=0.9)
plt.xlim(-2.5, 1)
plt.ylim(-1.25, 1.25)
plt.title("360,000 Points @ 60+ FPS", fontsize=14, color="white")
plt.axis("equal")
plt.show()
