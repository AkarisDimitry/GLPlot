"""Screen-sampled function plots — zoom in and the curve is recomputed, not magnified.

The idea
--------
``plot(x, y)`` stores a table. Zoom into it and you magnify the samples it was given: a
thousand points over (-1, 1) hold *one* sample in the window (1e-3, 2e-3), so a deep zoom
shows a straight line where the function actually oscillates eighty times. Zoom *out* and
the opposite waste happens — a million stored points crammed into a thousand pixels.

``function(f, xlim)`` stores the **function** and samples it against the **screen**: about
one sample per pixel column of whatever x range is currently visible, re-evaluated every
time the view moves. Three consequences, and they are the whole point:

* **constant resolution** — always exactly what the display can show;
* **constant cost** — ~2000 evaluations per frame at any zoom depth;
* **unbounded detail** — features finer than any fixed sampling resolve as you approach.

Compare with the live GPU fractal (``ex_mandelbrot.py --gpu``), the other view-driven layer
here. Both recompute from the view, but the fractal evaluates per *pixel of area*, so it
gets more expensive the deeper you go. A curve is one-dimensional: one sample per pixel
*column* is enough, so it stays cheap forever. Different geometry, different budget.

matplotlib has no equivalent — its zoom never re-samples, because there is nothing to
re-sample from. Desmos and GeoGebra work this way, and Datashader re-aggregates per view
for data, but within the scientific-Python plotting stack this is new.

When *not* to use it: measured data. There is no ``f`` to re-evaluate, and resampling
measurements would be inventing values that were never observed. ``plot`` is right there.

Run it::

    python examples/ex_function_plot.py            # sin(1/x) — scroll into x = 0
    python examples/ex_function_plot.py --compare  # the same curve, stored vs live
    python examples/ex_function_plot.py --zoo      # a page of hard functions
    python examples/ex_function_plot.py --poles    # tan and 1/x: gaps, not spikes
"""

from __future__ import annotations

import sys

import numpy as np

import glplot.pyplot as plt


def topologist_sine() -> None:
    """``sin(1/x)`` — the standard test for whether a plotter is sampling or storing.

    Every zoom towards the origin brings out oscillations that were not in the previous
    frame, because they were never computed until you asked to look there.
    """
    plt.figure("sin(1/x) — scroll into the origin", figsize=(10, 6))
    plt.function(lambda x: np.sin(1.0 / x), (-1.0, 1.0), ylim=(-1.3, 1.3), label="sin(1/x)")
    plt.title("sin(1/x) — scroll towards x = 0; the detail is computed on arrival")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def compare_stored_and_live() -> None:
    """The same function, both ways, in one figure. Zoom in and they diverge.

    The stored curve is a 2000-point table — a generous one, more than the live layer uses
    per frame. It looks identical at the full view. Scroll towards the origin and it turns
    into straight segments while the live one keeps resolving, because the table has
    nothing left to show and the function always does.
    """
    xs = np.linspace(-1.0, 1.0, 2000)
    with np.errstate(divide="ignore", invalid="ignore"):
        ys = np.sin(1.0 / xs)

    plt.figure("stored table vs live function", figsize=(10, 6))
    plt.plot(
        xs, ys, color=(0.85, 0.30, 0.25, 0.9), linewidth=3.0, label="plot() — 2000 stored points"
    )
    plt.function(
        lambda x: np.sin(1.0 / x),
        (-1.0, 1.0),
        ylim=(-1.3, 1.3),
        color=(0.15, 0.45, 0.75, 1.0),
        linewidth=1.5,
        label="function() — resampled per view",
    )
    plt.title("identical at this zoom — scroll into x = 0 and watch the red one give up")
    plt.legend()
    plt.show()


def zoo() -> None:
    """Functions that punish a fixed sampling, each in its own panel."""
    _fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    specs = [
        ("sin(1/x)", lambda x: np.sin(1.0 / x), (-0.5, 0.5), (-1.3, 1.3)),
        (
            "x·sin(1/x) — the squeeze",
            lambda x: x * np.sin(1.0 / x),
            (-0.3, 0.3),
            (-0.32, 0.32),
        ),
        (
            "Weierstrass — continuous, nowhere differentiable",
            lambda x: sum(0.5**k * np.cos(3.0**k * np.pi * x) for k in range(12)),
            (-1.0, 1.0),
            (-2.2, 2.2),
        ),
        (
            "a chirp — frequency grows without bound",
            lambda x: np.sin(x * x * 20.0),
            (-4.0, 4.0),
            (-1.3, 1.3),
        ),
    ]

    for ax, (title, f, xlim, ylim) in zip(np.ravel(axes), specs):
        plt.sca(ax)
        plt.function(f, xlim, ylim=ylim, label=title)
        plt.title(title)

    plt.suptitle("each panel resamples its own function against its own view — zoom any of them")
    plt.show()


def poles() -> None:
    """Non-finite values become gaps, which is what makes an asymptote readable.

    ``1/x`` is infinite at zero. A stored table would either miss the pole entirely or draw
    a vertical line across the whole plot joining +inf to -inf. Here the infinity is turned
    into a gap, so the two branches stay separate — and because ``sqrt`` is only real on
    the non-negatives, ``domain=`` stops it being evaluated where it is not defined at all.
    """
    plt.figure("poles and domains", figsize=(10, 6))
    plt.function(lambda x: 1.0 / x, (-3.0, 3.0), ylim=(-6.0, 6.0), label="1/x")
    plt.function(np.tan, (-3.0, 3.0), label="tan(x)")
    plt.function(np.sqrt, (-3.0, 3.0), domain=(0.0, np.inf), label="sqrt(x) — domain x ≥ 0")
    plt.title("1/x breaks at the pole; sqrt is simply not evaluated below zero")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    if "--compare" in sys.argv:
        compare_stored_and_live()
    elif "--zoo" in sys.argv:
        zoo()
    elif "--poles" in sys.argv:
        poles()
    else:
        topologist_sine()
