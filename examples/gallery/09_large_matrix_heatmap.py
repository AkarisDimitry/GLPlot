import numpy as np

import glplot.pyplot as plt

# Coherent superposition of five plane waves crossing at equal angles, as if a
# beam were split by a five-facet diffraction grating and recombined on a
# screen. The result is a quasi-periodic interference pattern (a Penrose-like
# tiling) rather than plain noise -- a single, dense imshow() field.
n = 700
axis = np.linspace(-10, 10, n)  # mm, screen coordinates
x, y = np.meshgrid(axis, axis)

k = 2.4  # rad/mm, spatial frequency of each plane wave
angles = np.linspace(0, np.pi, 5, endpoint=False)
amplitude = sum(np.cos(k * (x * np.cos(a) + y * np.sin(a))) for a in angles)
intensity = amplitude**2  # interference intensity, arbitrary units

plt.figure("Gallery - Five-Beam Interference", figsize=(8, 6))
plt.imshow(intensity, extent=(-10, 10, -10, 10), cmap="viridis")
plt.title("Five-beam interference pattern on a 700 x 700 screen grid")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
# plt.show()
plt.savefig("examples/gallery/results/09_large_matrix_heatmap.png")
