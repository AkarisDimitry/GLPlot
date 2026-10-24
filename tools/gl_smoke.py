#!/usr/bin/env python3
"""Render one frame through the real OpenGL path and fail if anything falls back.

The test suite runs without a window, so a broken GL context, a shader that no longer
compiles on a driver, or a silent fall back to matplotlib can pass every test. This script
asks for ``renderer="opengl"`` explicitly, so none of those can hide, and checks that real
pixels came out.

Run it before a release, on each OS you can reach:

    python tools/gl_smoke.py            # macOS, Windows, or Linux with a display
    xvfb-run -a python tools/gl_smoke.py  # headless Linux (what CI does)

Exit status is 0 on success, 1 on any failure.
"""

from __future__ import annotations

import sys
import warnings


def main() -> int:
    warnings.simplefilter("error", RuntimeWarning)  # the fallback is announced as one

    import numpy as np

    import glplot
    import glplot.pyplot as plt
    from glplot.animation import _animation_render_context, figure_to_rgb

    x = np.linspace(-2.0, 2.0, 48)
    grid_x, grid_y = np.meshgrid(x, x)
    height = np.exp(-(grid_x**2 + grid_y**2)) * np.cos(3.0 * grid_x)

    fig = plt.figure(figsize=(4, 3))
    plt.plot_surface(grid_x, grid_y, height)

    with _animation_render_context(fig, "opengl"):
        try:
            from OpenGL.GL import GL_RENDERER, GL_VERSION, glGetString

            renderer = glGetString(GL_RENDERER).decode()
            version = glGetString(GL_VERSION).decode()
        except Exception:  # noqa: BLE001 - the strings are only for the log
            renderer = version = "unknown"
        rgb = figure_to_rgb(fig)

    print(f"glplot {glplot.__version__}; GL {version} on {renderer}")
    colours = len(np.unique(rgb.reshape(-1, 3), axis=0))
    print(f"frame: {rgb.shape[1]}x{rgb.shape[0]}, {colours} colours")

    problems = []
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        problems.append(f"expected an (H, W, 3) uint8 frame, got {rgb.shape} {rgb.dtype}")
    elif rgb.std() < 5.0:
        problems.append(f"the frame is flat (std {rgb.std():.2f}); nothing was drawn")
    elif colours < 50:
        problems.append("fewer than 50 distinct colours; a shaded surface has many more")

    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    if not problems:
        print("OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
