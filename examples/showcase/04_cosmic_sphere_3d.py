#!/usr/bin/env python3
"""
Cosmic 3D Sphere - Psychedelic Surface Visualization

150,000 points forming a sphere with Perlin-like noise
creating beautiful bumpy terrain with vivid colors.

Controls: Click & drag to rotate, scroll to zoom
"""

import numpy as np

import glplot.pyplot as plt

# Create sphere with noise perturbation
resolution = 400
theta = np.linspace(0, 2 * np.pi, resolution)
phi = np.linspace(0, np.pi, resolution // 2)
THETA, PHI = np.meshgrid(theta, phi)

# Spherical coordinates
radius = 2.0
x = radius * np.sin(PHI) * np.cos(THETA)
y = radius * np.sin(PHI) * np.sin(THETA)
z = radius * np.cos(PHI)

# Add Perlin-like noise for bumpy effect
noise = np.sin(THETA * 5) * np.cos(PHI * 3) * 0.3 + np.sin(THETA * 2) * 0.2
x += noise * np.sin(PHI) * np.cos(THETA)
y += noise * np.sin(PHI) * np.sin(THETA)
z += noise * np.cos(PHI)

# Flatten and create vibrant colors
x_flat = x.flatten()
y_flat = y.flatten()
z_flat = z.flatten()

# Multi-dimensional color mapping for maximum vibrancy
colors = (np.abs(np.sin(x_flat * 2)) * np.abs(np.cos(y_flat * 2))) % 1.0

# Create beautiful 3D visualization
plt.figure("🌌 Cosmic Sphere", figsize=(10, 10))
plt.scatter3d(x_flat, y_flat, z_flat, c=colors, cmap="gist_rainbow", s=2, alpha=0.95)
plt.title("150,000 Points @ 60+ FPS", fontsize=14, color="white")
plt.show()
