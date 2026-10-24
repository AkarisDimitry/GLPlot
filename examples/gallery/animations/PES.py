"""Exemplary 2D potential energy surface, rendered as a 3D GPU-accelerated surface.

A rugged multi-well landscape rather than a single idealized double well:
several Gaussian wells of random depth, width, and position (fixed seed,
reproducible) carved into a shallow confining bowl. Meant to look like the
kind of multi-basin PES that global-optimization methods in this thesis
(basin hopping / EZGA, nested sampling, ...) are built to navigate.
"""

import numpy as np

import glplot.pyplot as gplt


def _generate_wells(n=10, seed=5, extent=2.5):
    rng = np.random.default_rng(seed)
    depth = rng.uniform(2.5, 4.0, n)
    x0 = rng.uniform(-extent, extent, n)
    y0 = rng.uniform(-extent, extent, n)
    width = rng.uniform(0.2, 0.9, n)
    return list(zip(depth, x0, y0, width))


_WELLS = _generate_wells()
_CONFINE = 0.2


def pes(x, y):
    """Sum of Gaussian wells inside a shallow quadratic confinement."""
    v = _CONFINE * (x**2 + y**2)
    for depth, x0, y0, width in _WELLS:
        v -= depth * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * width**2))
    return v - v.min()


def main():
    x = np.linspace(-3.2, 3.2, 220)
    y = np.linspace(-3.2, 3.2, 220)
    X, Y = np.meshgrid(x, y)
    Z = pes(X, Y)

    gplt.figure("Potential Energy Surface", figsize=(9, 7))
    gplt.plot_surface(X, Y, Z, cmap="coolwarm")

    gplt.set_box_aspect((1, 1, 0.6))
    gplt.view_init(elev=35, azim=-60)

    gplt.xlabel("x")
    gplt.ylabel("y")
    gplt.zlabel("Energy")
    gplt.title("Exemplary 2D PES: multi-well landscape")

    # gplt.savefig("../plots/PES_3D.png", dpi=300)
    gplt.show()


if __name__ == "__main__":
    main()
