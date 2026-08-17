"""Fractal zoom -- deepening into a Mandelbrot seahorse spiral, colour-cycled.

The Mandelbrot set is the set of complex numbers ``c`` for which the recursion

    z_{n+1} = z_n^2 + c,   z_0 = 0

never escapes to infinity. A point's *colour* is how many iterations it took to
escape (its "escape time"); the boundary between escaping and non-escaping points
is the famously infinite-detail fractal edge. Diving toward a point ON that
boundary -- rather than the bulk interior -- reveals self-similar spirals and
filaments at every scale: that is the "zoom" this animation performs, honing in
on a classic seahorse-valley spiral at
``c = -0.743643887037151 + 0.13182590420533j`` and geometrically shrinking the
view window every frame while raising the iteration budget so newly-visible
detail keeps resolving instead of flattening into a blur.

Technique, following ``examples/ex_mandelbrot.py``'s CPU-baked pattern: the whole
escape-time grid is one vectorised numpy computation per frame (every pixel's
``z = z*z + c`` iterated together as an array op, not a Python-level per-pixel
loop), smoothed with the standard fractional-escape-count renormalisation so the
colour bands are continuous rather than stair-stepped, then log1p-compressed and
handed to ``imshow`` as a full RGB image -- built by remapping the smoothed
escape field through a colormap whose hue is slowly rotated frame-to-frame
(``turbo`` warped through HSV space) for a shimmering, ever-shifting palette on
top of the zoom itself.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

import glplot.animation as animation
import glplot.pyplot as plt


def mandelbrot(width: int, height: int, centre: complex, span: float, max_iter: int):
    """Vectorised escape-time grid over a window of the complex plane.

    Returns ``(extent, smooth_counts)`` -- the ``(x0, x1, y0, y1)`` the window
    covers and a ``(height, width)`` array of smoothed (fractional) escape
    counts, so a colormap over it does not stair-step into discrete bands.
    """
    aspect = width / height
    re = np.linspace(centre.real - span * aspect, centre.real + span * aspect, width)
    im = np.linspace(centre.imag - span, centre.imag + span, height)
    c = re[None, :] + 1j * im[:, None]

    z = np.zeros_like(c)
    count = np.zeros(c.shape, dtype=np.float64)
    alive = np.ones(c.shape, dtype=bool)

    for i in range(max_iter):
        z[alive] = z[alive] * z[alive] + c[alive]
        escaped = alive & (z.real * z.real + z.imag * z.imag > 4.0)
        zabs = np.abs(z[escaped])
        count[escaped] = (i + 1) - np.log2(np.log(zabs + 1e-12))
        alive[escaped] = False
        if not alive.any():
            break

    count[alive] = float(max_iter)  # still bounded after the budget: "in" the set
    extent = (re[0], re[-1], im[0], im[-1])
    return extent, count


def hue_rotated_rgb(values: np.ndarray, cmap_name: str, phase: float) -> np.ndarray:
    """Map a scalar field through a colormap, then rotate every pixel's hue by ``phase``.

    Gives the same escape-time structure a slow, shimmering palette drift across
    frames instead of a single static gradient -- extra flair on top of the zoom.
    """
    norm = (values - values.min()) / (np.ptp(values) + 1e-12)
    rgba = colormaps[cmap_name](norm)  # (H, W, 4), floats in [0, 1]
    rgb = rgba[..., :3]

    hsv = rgb_to_hsv(rgb)  # vectorised, no per-pixel Python loop
    hsv[..., 0] = (hsv[..., 0] + phase) % 1.0
    return hsv_to_rgb(hsv)


def main() -> None:
    target = -0.743643887037151 + 0.13182590420533j  # classic seahorse-valley spiral
    fig = plt.figure("Gallery - Fractal Zoom", figsize=(7, 7))
    plt.plot_style("dark")

    frames = 72
    grid = 460

    def update(frame: int):
        span = 1.4 * (0.93**frame)  # geometric zoom: constant magnification per frame
        max_iter = int(140 + frame * 7)  # deeper zoom needs more iterations to resolve
        extent, counts = mandelbrot(grid, grid, centre=target, span=span, max_iter=max_iter)
        compressed = np.log1p(counts)

        phase = (frame / frames) * 0.35  # slow hue drift, well under one full rotation
        rgb = hue_rotated_rgb(compressed, "turbo", phase)

        plt.cla()
        plt.imshow(rgb, extent=extent, origin="lower", aspect="equal")
        zoom_factor = 1.4 / span
        plt.title(f"Mandelbrot seahorse spiral -- zoom {zoom_factor:,.0f}x, {max_iter} iters")
        plt.xlabel("Re(c)")
        plt.ylabel("Im(c)")
        return []

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=40)
    ani.save("examples/gallery/animations/results/07_fractal_zoom.gif", fps=20)
    # plt.show()


if __name__ == "__main__":
    main()
