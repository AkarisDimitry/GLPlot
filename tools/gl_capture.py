"""Capture what the real OpenGL window draws, as an image. Development tool.

``savefig`` before ``show()`` goes through the matplotlib preview (it has to: a GL context
may not exist), and that export has neither the engine's text, nor its glow and tone-map,
nor its 3D shading. The gallery thumbnails are meant to show what the *workstation*
draws, so they are read back from the real framebuffer instead.

The mechanism is the one in the project's own notes on verifying the live window: wrap
``glfw.swap_buffers`` and ``glReadPixels`` the default framebuffer just before the swap,
which is the one moment the back buffer holds the fully composited frame, imgui text
included. Two details are load-bearing:

* ``reactive_rendering`` is switched off. A static scene under the reactive loop sleeps and
  stops swapping, so a capture waiting for frame N would wait forever.
* The first frames are skipped. Frame 1 differs from the steady state (the bloom chain and
  the auto-framing settle over the first few frames).

Needs a display and a GPU. It is **not** used at runtime and not by the test suite (live GL
calls are kept out of pytest on purpose); it is run by hand to regenerate the committed
thumbnails, see ``tools/build_gallery_assets.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import imgui_bundle  # noqa: F401  -- must precede ``glfw`` (see glplot/__init__.py)
import glfw  # noqa: E402
from OpenGL.GL import GL_RGB, GL_UNSIGNED_BYTE, glFinish, glReadPixels  # noqa: E402

#: Frames drawn before the grab. The bloom chain allocates on the first resolve and the 3D
#: axes rebuild once the camera has settled; twelve is comfortably past both.
SETTLE_FRAMES = 12


def capture(plot: Any, *, frames: int = SETTLE_FRAMES) -> np.ndarray:
    """Open ``plot``'s window, draw ``frames`` frames and return the last one as RGB uint8.

    Blocks until the window closes itself; the window is visible for a moment, which is the
    price of reading a real framebuffer. The returned array is ``(fb_height, fb_width, 3)``
    in top-down row order, at the framebuffer's own (possibly 2x) resolution.
    """
    plot.options.reactive_rendering = False
    grabbed: dict = {}
    count = {"frames": 0}
    real_swap = glfw.swap_buffers

    def swap(window: Any) -> None:
        count["frames"] += 1
        if count["frames"] == frames:
            glFinish()
            width, height = plot.fb_width, plot.fb_height
            pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            grabbed["image"] = np.flipud(
                np.frombuffer(pixels, np.uint8).reshape((height, width, 3))
            ).copy()
            glfw.set_window_should_close(window, True)
        return real_swap(window)

    glfw.swap_buffers = swap
    try:
        plot.run()
    finally:
        glfw.swap_buffers = real_swap
    if "image" not in grabbed:
        raise RuntimeError("the window closed before a frame could be captured")
    return grabbed["image"]
