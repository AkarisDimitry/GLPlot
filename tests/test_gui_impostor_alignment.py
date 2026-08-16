"""The pan impostor must land exactly where the exact view would draw the same data.

While a drag is in progress the engine stops re-rendering the scene and instead reprojects
a cached texture of it (that is the deliberate resolution drop). If that reprojection does
not agree with the real projection, the plot visibly jumps the moment you grab it, sits
offset for the whole drag, and snaps back on release.

The defect this pins: a world window does not span the viewport. ``mvp()`` insets it by the
axis margins, so world ``cur.x`` lands at screen x = margin_l, not 0. The impostor shader
used to map its full-screen UV straight to world coordinates, skipping the inset on both the
current view and the cached texture. Those two errors cancel only while cur == cache, and
the cache is captured 3x padded (``cache_padding``), so they never are. Measured before the
fix: 40px of displacement at the default 60px margin, 78px once the GUI rail widened it.

Everything here is CPU arithmetic mirroring the GLSL -- no OpenGL context is created.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.engine import GPULinePlot
from glplot.options import resolve_axis_margins

WIDTH = 1280
HEIGHT = 720


def _plot(margin_l: float) -> GPULinePlot:
    """A headless engine with a chosen left margin. No window, no GL context."""
    plot = GPULinePlot()
    plot.options.axis_margin_l = margin_l
    return plot


def _exact_screen_x(plot: GPULinePlot, world_x: float, window) -> float:
    """Ground truth: where the real projection puts ``world_x``."""
    mvp = plot.camera_controller.mvp(WIDTH, HEIGHT, window=window)
    ndc = mvp @ np.array([world_x, 0.0, 0.0, 1.0])
    return float((ndc[0] / ndc[3] * 0.5 + 0.5) * WIDTH)


def _impostor_screen_x(plot: GPULinePlot, world_x: float, current, cache) -> float:
    """Where the impostor shader puts ``world_x``, replayed on the CPU.

    Mirrors CACHE_IMPOSTOR_FS: screen uv -> inset t -> world -> cache inset -> texture uv.
    The texture itself was rendered through ``mvp()``, so the world value stored at a given
    texture uv is found by inverting the same inset mapping.
    """
    margin_l, margin_r, _, _ = resolve_axis_margins(plot.options)
    left = margin_l / WIDTH
    right = margin_r / WIDTH
    inset = 1.0 - left - right

    screen_uv = np.arange(WIDTH, dtype=float) / WIDTH
    t = (screen_uv - left) / inset
    world = current[0] + t * (current[1] - current[0])

    cache_t = (world - cache[0]) / (cache[1] - cache[0])
    texture_uv = left + cache_t * inset

    # Invert: which world value actually lives at that texture uv.
    stored_t = (texture_uv - left) / inset
    stored_world = cache[0] + stored_t * (cache[1] - cache[0])

    return float(np.argmin(np.abs(stored_world - world_x)))


def _windows(plot: GPULinePlot, pan_fraction: float):
    """The (current, cache) world windows for a pan of ``pan_fraction`` of the view."""
    controller = plot.camera_controller
    cache = controller.world_window(WIDTH, HEIGHT, padding=plot.options.cache_padding)
    rest = controller.world_window(WIDTH, HEIGHT)
    span = rest[1] - rest[0]
    current = (rest[0] + pan_fraction * span, rest[1] + pan_fraction * span, rest[2], rest[3])
    return current, cache


class TestImpostorMatchesTheExactProjection:
    """The impostor and the exact view must agree, at rest and mid-drag."""

    @pytest.mark.parametrize("margin_l", [60.0, 98.0])
    @pytest.mark.parametrize("pan_fraction", [0.0, 0.1, 0.3, -0.25])
    def test_no_displacement(self, margin_l: float, pan_fraction: float):
        """A feature must be drawn at the same pixel whichever path renders it."""
        plot = _plot(margin_l)
        current, cache = _windows(plot, pan_fraction)
        world_x = (current[0] + current[1]) * 0.5

        exact = _exact_screen_x(plot, world_x, current)
        impostor = _impostor_screen_x(plot, world_x, current, cache)

        assert impostor == pytest.approx(exact, abs=1.0), (
            f"impostor is {impostor - exact:+.1f}px off the exact projection "
            f"(margin_l={margin_l}, pan={pan_fraction:.0%})"
        )

    def test_the_old_mapping_really_was_broken(self):
        """Guards the guard: the pre-fix math must fail this, or the test proves nothing."""
        plot = _plot(98.0)
        current, cache = _windows(plot, 0.0)
        world_x = (current[0] + current[1]) * 0.5

        margin_l, margin_r, _, _ = resolve_axis_margins(plot.options)
        left, right = margin_l / WIDTH, margin_r / WIDTH
        inset = 1.0 - left - right

        # The shader as it shipped: v_uv straight to world, ignoring both insets.
        screen_uv = np.arange(WIDTH, dtype=float) / WIDTH
        world = current[0] + screen_uv * (current[1] - current[0])
        cache_t = (world - cache[0]) / (cache[1] - cache[0])
        stored_t = (cache_t - left) / inset
        stored_world = cache[0] + stored_t * (cache[1] - cache[0])
        old = float(np.argmin(np.abs(stored_world - world_x)))

        exact = _exact_screen_x(plot, world_x, current)
        assert abs(old - exact) > 20.0, "the old mapping no longer reproduces the bug"


class TestImpostorShaderContract:
    """The GLSL needs the margins, and the renderer has to hand them over."""

    def test_shader_takes_a_margins_uniform(self):
        """Without it the shader cannot know where the plot area starts."""
        from glplot.utils.shaders import CACHE_IMPOSTOR_FS

        assert "u_margins" in CACHE_IMPOSTOR_FS

    def test_shader_does_not_map_uv_straight_to_world(self):
        """The exact defect: `mix(u_cur_window.x, u_cur_window.y, v_uv.x)`."""
        from glplot.utils.shaders import CACHE_IMPOSTOR_FS

        assert "v_uv.x)" not in CACHE_IMPOSTOR_FS.replace(
            " ", ""
        ), "the impostor maps screen uv directly to world again, skipping the axis inset"

    def test_renderer_uploads_normalized_margins(self):
        """Normalised, not pixels: the cache texture is framebuffer-sized (DPR-scaled)."""
        from glplot.renderers.interaction import InteractionRenderer

        plot = _plot(98.0)
        plot.width, plot.height = WIDTH, HEIGHT
        renderer = InteractionRenderer(plot)

        left, right, bottom, top = renderer._normalized_margins()
        assert left == pytest.approx(98.0 / WIDTH)
        assert right == pytest.approx(20.0 / WIDTH)
        assert bottom == pytest.approx(40.0 / HEIGHT)
        assert top == pytest.approx(20.0 / HEIGHT)

    def test_normalized_margins_are_resolution_independent(self):
        """The same layout at 2x logical size must give the same fractions."""
        from glplot.renderers.interaction import InteractionRenderer

        plot = _plot(98.0)
        plot.width, plot.height = WIDTH, HEIGHT
        small = InteractionRenderer(plot)._normalized_margins()

        plot.width, plot.height = WIDTH, HEIGHT
        again = InteractionRenderer(plot)._normalized_margins()
        assert small == again
