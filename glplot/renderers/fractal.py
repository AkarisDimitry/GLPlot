"""GPU renderer for :class:`~glplot.core.layers.FractalLayer`: a live escape-time field.

The design and its one honest limit are documented on :data:`glplot.utils.shaders.FRACTAL_FS`.
In short: a quad at the fractal's world extent is drawn with a fragment shader that runs
``z -> z^2 + c`` per screen fragment, so the field is recomputed at the screen's own
resolution every frame. Zooming the camera makes each pixel cover a finer world region and
the boundary re-detailes automatically — no CPU recompute, no grid to re-sample.

The renderer's only real job beyond drawing the quad is **adaptive iteration depth**: a
Mandelbrot boundary that is 200 iterations deep at the full view needs far more once the
camera has zoomed in a thousandfold, or the filaments stop resolving and flatten into
blobs. :meth:`_adaptive_iter` raises the budget with the log of the zoom, which is what
keeps a deep zoom looking like a fractal rather than a smudge.
"""

from __future__ import annotations

import ctypes as C
import math
from typing import TYPE_CHECKING

import numpy as np
from OpenGL.GL import *

from ..utils.gl_utils import link_program
from ..utils.shaders import FRACTAL_FS, FRACTAL_VS

if TYPE_CHECKING:
    from ..core.context import RenderContext
    from ..core.layers import FractalLayer
    from ..options import EngineOptions

#: Colour-scheme name -> ``apply_heatmap`` index (the same table the density path uses).
#: An unknown name falls back to magma, which is the fractal default.
_SCHEME_INDEX = {
    "classic": 0,
    "viridis": 1,
    "plasma": 2,
    "inferno": 3,
    "turbo": 4,
    "ink_fire": 5,
    "inkfire": 5,
    "magma": 6,
    "gray": 7,
    "grayscale": 7,
    "greys": 7,
    "ocean": 8,
    "hot": 9,
    "cool": 10,
}

#: Iteration budget added per power-of-two of zoom past the initial framing. 48 keeps a
#: 1000x zoom (~10 doublings) at ~680 iterations, which resolves the seahorse filaments
#: without stalling the frame.
_ITER_PER_OCTAVE = 48.0

#: Hard cap, matching the shader's ``ITER_CAP``. Past this the frame cost dominates and the
#: float32 precision wall (below) has usually been hit anyway.
_MAX_ITER = 2000

#: Visible span, as a fraction of the fractal's own width, below which float32 world
#: coordinates can no longer separate neighbouring pixels and the image pixelates. Used only
#: to stop *raising* the iteration count once detail can no longer appear — spending 2000
#: iterations to sharpen mush is pure waste.
_PRECISION_SPAN_FRAC = 3e-5

#: Fraction of the iteration budget spent while the view is *moving*.
#:
#: The fractal is the one layer whose cost scales with screen *area*: at a deep zoom the
#: shader runs ~700 iterations for every one of ~2.5M fragments, which is fine for a still
#: frame and far too slow to keep up with a scroll wheel. Zooming therefore felt laggy —
#: the frame simply could not be produced at wheel rate.
#:
#: So: draw the moving frames cheap and the settled frame full. A third of the iterations
#: is roughly three times the frame rate, and the difference is nearly invisible while the
#: image is sliding under the cursor; the instant motion stops, one more frame is requested
#: and drawn at the full budget. This is progressive refinement, the standard answer for a
#: per-pixel field, and it costs nothing when the view is still.
_INTERACTIVE_ITER_FRAC = 0.34

#: Never drop below this while moving — under ~60 iterations the set loses its shape and
#: the preview stops being a recognisable preview of what will land.
_MIN_INTERACTIVE_ITER = 60

#: Relative window change treated as "the view is moving" rather than float noise.
_MOTION_EPS = 1e-4


class FractalRenderer:
    """Draws a :class:`FractalLayer` as a live, zoom-refined GPU field."""

    def __init__(self, options: EngineOptions) -> None:
        self.options = options
        self.prog = 0
        self.vao = 0
        self.vbo = 0
        self.u_mvp = -1
        self.u_max_iter = -1
        self.u_type = -1
        self.u_julia_c = -1
        self.u_scheme = -1
        self.u_gain = -1
        self.u_alpha = -1
        self.u_inset = -1
        #: Last frame's world window, for the motion test in :meth:`_is_moving`.
        self._last_window: tuple | None = None
        #: True when the last frame was drawn cheap and a full-quality one is owed. The
        #: engine drains this (``consume_refine_request``) and marks the scene dirty, which
        #: is what wakes a reactive loop for the settle frame that would otherwise never
        #: come — nothing else changes once the wheel stops turning.
        self._refine_pending = False

    def initialize(self) -> None:
        self.prog = link_program(FRACTAL_VS, FRACTAL_FS)
        self.u_mvp = glGetUniformLocation(self.prog, "u_mvp")
        self.u_max_iter = glGetUniformLocation(self.prog, "u_max_iter")
        self.u_type = glGetUniformLocation(self.prog, "u_type")
        self.u_julia_c = glGetUniformLocation(self.prog, "u_julia_c")
        self.u_scheme = glGetUniformLocation(self.prog, "u_scheme")
        self.u_gain = glGetUniformLocation(self.prog, "u_gain")
        self.u_alpha = glGetUniformLocation(self.prog, "u_alpha")
        self.u_inset = glGetUniformLocation(self.prog, "u_inset_color")

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
        glBindVertexArray(0)

    def _quad(self, extent) -> np.ndarray:
        """Two triangles covering ``(x0, x1, y0, y1)`` in world coordinates."""
        x0, x1, y0, y1 = (float(v) for v in extent)
        return np.array(
            [x0, y0, x1, y0, x1, y1, x0, y0, x1, y1, x0, y1],
            dtype=np.float32,
        )

    def _adaptive_iter(self, layer: FractalLayer, ctx: RenderContext) -> int:
        """Iteration budget for this frame, raised with the zoom but capped.

        Zoom is the fractal's own width over the currently visible width. Once the visible
        span crosses the float32 precision floor the budget stops rising — beyond there
        more iterations only sharpen pixelation, never detail.
        """
        base_w = max(float(layer.extent[1]) - float(layer.extent[0]), 1e-30)
        l, r, _b, _t = ctx.window_world
        view_w = max(float(r) - float(l), 1e-30)
        zoom = base_w / view_w
        octaves = max(math.log2(zoom), 0.0) if zoom > 1.0 else 0.0
        if view_w < base_w * _PRECISION_SPAN_FRAC:
            # Freeze the budget at the precision wall.
            octaves = math.log2(1.0 / _PRECISION_SPAN_FRAC)
        return int(min(layer.max_iter + _ITER_PER_OCTAVE * octaves, _MAX_ITER))

    def _is_moving(self, ctx: RenderContext) -> bool:
        """Whether the view changed since the last drawn frame.

        Read from the window itself rather than from a drag flag, because the case that
        actually hurts is the *scroll wheel*: it zooms without any drag being active, and
        it is the interaction the user reported as laggy. A window comparison catches
        wheel, drag, keyboard and animated cameras alike, and needs no plumbing.
        """
        window = tuple(float(v) for v in ctx.window_world)
        previous, self._last_window = self._last_window, window
        if previous is None:
            return False  # the first frame of a still figure should be full quality
        scale = max(abs(previous[1] - previous[0]), 1e-30)
        return any(abs(a - b) > scale * _MOTION_EPS for a, b in zip(window, previous))

    def consume_refine_request(self) -> bool:
        """True once after a frame drawn at reduced quality; clears the flag.

        The engine calls this to decide whether to schedule one more frame. Consuming
        rather than peeking is what makes the sequence terminate: the refined frame is
        drawn at full budget, sets nothing, and the loop goes back to sleep.
        """
        pending, self._refine_pending = self._refine_pending, False
        return pending

    def draw(self, layer: FractalLayer, ctx: RenderContext) -> None:
        extent = getattr(layer, "extent", None)
        if extent is None:
            return

        # Progressive refinement: cheap while the view moves, full once it settles.
        max_iter = self._adaptive_iter(layer, ctx)
        if self._is_moving(ctx):
            max_iter = max(int(max_iter * _INTERACTIVE_ITER_FRAC), _MIN_INTERACTIVE_ITER)
            self._refine_pending = True

        glUseProgram(self.prog)
        glUniformMatrix4fv(self.u_mvp, 1, GL_TRUE, np.asarray(ctx.mvp, dtype=np.float32))
        glUniform1i(self.u_max_iter, max_iter)
        glUniform1i(self.u_type, 1 if str(layer.fractal_type) == "julia" else 0)
        cx, cy = layer.julia_c
        glUniform2f(self.u_julia_c, float(cx), float(cy))
        scheme = _SCHEME_INDEX.get(str(layer.cmap).strip().lower(), 6)
        glUniform1i(self.u_scheme, int(scheme))
        glUniform1f(self.u_gain, float(layer.gain))
        glUniform1f(self.u_alpha, float(ctx.global_alpha * layer.style.alpha))
        ic = layer.inset_color
        glUniform3f(self.u_inset, float(ic[0]), float(ic[1]), float(ic[2]))

        glBindVertexArray(self.vao)
        quad = self._quad(extent)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_DYNAMIC_DRAW)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
        layer.dirty.gpu_dirty = False
