#!/usr/bin/env python3
"""
Spinning 3D Torus - Beautiful Donut Visualization

50,000 points forming a vibrant spinning torus with
smooth color gradients. Fully interactive in 3D!

Controls: Click & drag to rotate, scroll to zoom
"""

import numpy as np
import glplot.pyplot as plt

# Create 3D torus mesh
u = np.linspace(0, 2 * np.pi, 200)
v = np.linspace(0, 2 * np.pi, 250)
u_mesh, v_mesh = np.meshgrid(u, v)

# Torus parametric equations
R, r = 3, 1
x = (R + r * np.cos(v_mesh)) * np.cos(u_mesh)
y = (R + r * np.cos(v_mesh)) * np.sin(u_mesh)
z = r * np.sin(v_mesh)

# Flatten and create rainbow colors
x_flat = x.flatten()
y_flat = y.flatten()
z_flat = z.flatten()
colors = (np.arctan2(y_flat, x_flat) / np.pi + 1) * 0.5  # Azimuthal coloring

# Create 3D scatter plot
plt.figure("🍩 Spinning Torus", figsize=(10, 10))
plt.scatter3d(x_flat, y_flat, z_flat, c=colors, cmap="hsv", s=3, alpha=0.9)
plt.title("50,000 Points @ 60+ FPS", fontsize=14, color="white")
plt.show()
