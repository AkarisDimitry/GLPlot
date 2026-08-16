"""Mandelbrot set — the recursive escape-time fractal, plotted and animated.

The Mandelbrot set is the set of complex numbers ``c`` for which the recursion

    z_{n+1} = z_n^2 + c,   z_0 = 0

stays bounded forever. In practice a point is "in" the set if ``|z_n|`` has not exceeded 2
after some iteration budget, and the *colour* of a point outside the set is how many
iterations it took to escape — the escape time. That is the recursion the title asks for:
one line, applied over and over, per point.

How each point could be computed on the GPU
--------------------------------------------
The escape loop is embarrassingly parallel: every pixel is independent, so the natural GPU
form is a **fragment shader** that runs the whole ``z = z*z + c`` loop for its own pixel
and writes the escape count as the colour. That is how a real-time Mandelbrot explorer
works — millions of pixels iterate at once, and a deep zoom stays interactive.

GLPlot does not ship that shader yet, so this example computes the escape counts on the
**CPU, vectorised with numpy** — every pixel of the grid iterated together as one array
operation, which is the same data-parallel idea one step short of the GPU — and then hands
the finished grid to GLPlot's GPU renderer as an image. The maths is identical; only *where
the loop runs* differs. The building block for a GPU port already lives in
``glplot.anim.processes`` as the ``mandelbrot_zoom`` process, whose ``mandelbrot_frame(t)``
is exactly the per-frame escape grid a fragment shader would produce.

There are now **two** ways to draw a Mandelbrot in GLPlot, and this file shows both:

* **CPU-baked** (``plot_still`` / ``animate_zoom`` below): compute the escape grid with
  numpy and hand the finished image to the renderer. Simple, portable, and what you do for
  any field you can only express in Python — but the pixels are fixed once computed, so
  zooming just magnifies them.
* **Live on the GPU** (``--gpu``): ``gplt.mandelbrot()`` / ``gplt.julia()`` draw a
  ``fractal`` layer whose fragment shader runs the escape loop per screen pixel every
  frame. Scroll to zoom and the boundary refines to the new scale on its own — this is the
  "recompute on zoom for more precision" version, and it is real-time. It is the GPU port
  the "how could each be computed on the GPU?" question is about.

Run it::

    python examples/ex_mandelbrot.py           # CPU-baked still, coloured by escape time
    python examples/ex_mandelbrot.py --zoom    # CPU-baked animated zoom into a seahorse
    python examples/ex_mandelbrot.py --gpu     # LIVE GPU set — scroll to zoom, it refines
"""

from __future__ import annotations

import sys

import numpy as np

import glplot.pyplot as plt


def mandelbrot(width: int, height: int, centre: complex, span: float, max_iter: int):
    """The escape-time grid over a window of the complex plane.

    Vectorised: the whole grid iterates as one numpy operation per step, so the Python
    loop runs ``max_iter`` times total, not ``max_iter * width * height`` times. A GPU
    fragment shader would drop even the ``max_iter`` loop onto the hardware and run every
    pixel's iterations concurrently — this is the CPU shadow of that.

    Returns ``(extent, smooth_counts)``: the ``(x0, x1, y0, y1)`` the window covers and a
    ``(height, width)`` array of smoothed escape values (continuous, so the colour bands do
    not stair-step).
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
        # Smooth (fractional) escape count: the standard renormalisation that turns the
        # integer iteration bands into a continuous field, so the colormap does not
        # stair-step. ``i`` is how many times the recursion has been applied so far.
        zabs = np.abs(z[escaped])
        count[escaped] = (i + 1) - np.log2(np.log(zabs + 1e-12))
        alive[escaped] = False
        if not alive.any():
            break

    count[alive] = float(max_iter)  # points still bounded: treat as "in the set"
    extent = (re[0], re[-1], im[0], im[-1])
    return extent, count


def plot_still() -> None:
    """A single high-detail Mandelbrot, coloured by escape time."""
    extent, counts = mandelbrot(900, 700, centre=-0.6 + 0.0j, span=1.25, max_iter=200)
    plt.figure("Mandelbrot set", figsize=(9, 7))
    # log1p compresses the huge dynamic range near the boundary so the filaments show.
    plt.imshow(
        np.log1p(counts),
        extent=extent,
        origin="lower",
        cmap="magma",
        aspect="equal",
    )
    plt.title("Mandelbrot set — coloured by escape time")
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.show()


def animate_zoom() -> None:
    """Zoom into a seahorse-valley point, one frame per escape grid.

    Uses ``glplot.animation`` (the matplotlib-compatible surface): a ``FuncAnimation`` whose
    update recomputes the grid at a smaller span each frame. Every frame is a fresh
    escape-time computation — the discrete progression the animation model calls a
    *step* rather than a blend, because there is no meaningful "half a Mandelbrot frame".
    """
    import glplot.animation as animation

    target = -0.743643887037151 + 0.13182590420533j  # a classic seahorse spiral
    fig = plt.figure("Mandelbrot zoom", figsize=(8, 8))
    frames = 120

    def update(frame: int):
        span = 1.4 * (0.94**frame)  # geometric zoom: constant magnification per frame
        max_iter = int(120 + frame * 6)  # deeper zoom needs more iterations to resolve
        extent, counts = mandelbrot(500, 500, centre=target, span=span, max_iter=max_iter)
        plt.cla()
        plt.imshow(np.log1p(counts), extent=extent, origin="lower", cmap="turbo", aspect="equal")
        plt.title(f"zoom {1.4 / span:,.0f}x   ({max_iter} iterations)")
        return []

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=40)
    # ani.save("mandelbrot_zoom.mp4", fps=25)   # needs imageio-ffmpeg or ffmpeg on PATH
    plt.show()


def show_gpu() -> None:
    """The live, GPU-computed set. Scroll to zoom — it refines to the new scale in real time.

    ``gplt.mandelbrot()`` adds a ``fractal`` layer; the escape loop runs in a fragment
    shader, once per screen pixel, so there is no grid and no recompute to trigger. The
    renderer raises the iteration budget as you zoom in, so the boundary keeps resolving.
    """
    plt.figure("Mandelbrot (live GPU)", figsize=(9, 7))
    plt.mandelbrot(center=(-0.6, 0.0), span=1.3, cmap="magma", gain=1.2)
    plt.title("Mandelbrot — live on the GPU (scroll to zoom)")
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.show()


if __name__ == "__main__":
    if "--gpu" in sys.argv:
        show_gpu()
    elif "--zoom" in sys.argv:
        animate_zoom()
    else:
        plot_still()
