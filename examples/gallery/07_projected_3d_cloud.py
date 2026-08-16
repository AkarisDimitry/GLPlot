import numpy as np

import glplot.pyplot as plt

# Simulated circumstellar dust halo: an isotropic cloud of grains orbiting a
# young star, densest near the core and thinning outward. A single point
# type, shaded by depth so the spherical structure reads in projection.
#
# The raw Gaussian has a handful of >4-sigma stragglers that used to stretch
# the axis autoscale far past the visible halo, leaving a wide empty margin
# around it in projection. Cutting the halo off at a physical outer radius
# (dust halos do have an edge) keeps the plotted cloud exactly filling its
# axis box instead of a sparse few points dragging the frame out.
rng = np.random.default_rng(77)
n = 1_000_000
sigma = 42.0  # AU, characteristic radius of the halo
outer_radius = 2.5 * sigma  # AU, halo edge
n_gen = 1_140_000  # oversample so the radius cut still leaves n grains
x = rng.normal(0.0, sigma, n_gen)
y = rng.normal(0.0, sigma, n_gen)
z = rng.normal(0.0, sigma, n_gen)
inside = (x**2 + y**2 + z**2) < outer_radius**2
x, y, z = x[inside][:n], y[inside][:n], z[inside][:n]

plt.figure("Gallery - Circumstellar Dust Halo", figsize=(9, 6))
plt.scatter3d(
    x,
    y,
    z,
    c=z,
    cmap="viridis",
    s=1.1,
    alpha=0.75,
    elev=22,
    azim=-42,
    label=f"{n:,} grains",
)
plt.title(f"Simulated circumstellar dust halo ({n:,} grains, colored by depth)")
plt.xlabel("x (AU)")
plt.ylabel("y (AU)")
plt.zlabel("z (AU)")
plt.legend()
# plt.show()
plt.savefig("examples/gallery/results/07_projected_3d_cloud.png")
