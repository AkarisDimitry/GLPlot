"""The pan impostor must stand in for the exact scatter, not for a different picture.

Two defects are pinned here, both of which made ``examples/ex_scatter.py`` misbehave the
instant the camera moved. Both are about the *interaction cache*: the texture the engine
reprojects during a drag instead of re-rendering 5M points every frame.

1. **The cloud vanished.** The cache is cleared to transparent black and composited with
   ``GL_ONE, GL_ONE_MINUS_SRC_ALPHA``, i.e. it stores premultiplied RGB. Capturing it with
   plain ``glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)`` gets the colour channels
   right by accident and the alpha channel wrong on purpose: alpha lands as ``a*a``. At the
   ``alpha=0.1`` an overplotted cloud is drawn with, the cache recorded a hundredth of the
   coverage it painted, the composite barely dimmed the white page, and the scatter washed
   out to pure white. Measured on the real example before the fix: the impostor frame was
   uniformly 255 (min 251) where the exact frame it replaced went down to 77.

2. **The cloud darkened.** The cache is captured through a ``cache_padding`` times wider
   world window and magnified back by that factor. A point sprite keeps its size in
   *pixels* at any zoom, so the capture packed the same 5M markers into a ninth of the area
   -- the impostor carried 3.3x the ink of the exact frame. The capture now decimates by
   ``capture_scale**2``, which both restores the density and makes the capture that much
   cheaper (5,000,000 points -> 555,556 at the default 3x padding).

No OpenGL context is created: the blending test records the GL calls the engine makes, and
the LOD test is plain arithmetic on the renderer.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot import engine as engine_module
from glplot.core.context import RenderContext
from glplot.core.layers import LayerStyle, ScatterLayer
from glplot.engine import GPULinePlot
from glplot.options import BlendMode, EngineOptions, RenderMode
from glplot.renderers.scatter import ScatterRenderer
from glplot.utils import blending as blending_module

FB_W, FB_H = 2560, 1600

#: Blend factors are recorded by *name*, resolved against the engine's own namespace at call
#: time. Other suites in this repo stub ``OpenGL.GL`` out with a MagicMock, so importing the
#: enums here would bind different objects than the ones the engine passes to glBlendFunc --
#: these tests passed alone and failed in the full run.
_FACTOR_NAMES = (
    "GL_ONE",
    "GL_ZERO",
    "GL_SRC_ALPHA",
    "GL_ONE_MINUS_SRC_ALPHA",
    "GL_ONE_MINUS_SRC_COLOR",
)


# ---------------------------------------------------------------------------
# 1. Premultiplied-alpha capture
# ---------------------------------------------------------------------------


@pytest.fixture
def blend_calls(monkeypatch):
    """Record what ``_apply_blending_policy`` asks of GL, without a context.

    ``glEnable``/``glDisable`` fire from ``engine`` (``_apply_blending_policy`` turns
    ``GL_BLEND`` on there), while the mode-to-factor mapping moved to
    ``glplot.utils.blending.apply_blend_mode`` -- so the ``glBlendFunc*`` calls are recorded
    on the ``blending`` module, where they now happen. The factor enums are the same objects
    in both (each module does ``from OpenGL.GL import *``), so the name table resolves either
    way.
    """
    known = {id(getattr(engine_module, n)): n for n in _FACTOR_NAMES}
    name = lambda f: known.get(id(f), repr(f))  # noqa: E731

    calls: list[tuple] = []
    noop = lambda *a, **k: None  # noqa: E731
    monkeypatch.setattr(engine_module, "glEnable", noop)
    monkeypatch.setattr(engine_module, "glDisable", noop)
    monkeypatch.setattr(blending_module, "glBlendEquation", noop)
    monkeypatch.setattr(
        blending_module, "glBlendFunc", lambda s, d: calls.append(("func", name(s), name(d)))
    )
    monkeypatch.setattr(
        blending_module,
        "glBlendFuncSeparate",
        lambda s, d, sa, da: calls.append(("separate", name(s), name(d), name(sa), name(da))),
    )
    return calls


def test_direct_render_still_uses_plain_alpha_blending(blend_calls):
    """The on-screen path is untouched: same single ``glBlendFunc`` it always issued."""
    plot = GPULinePlot()
    plot.options.blend_mode = BlendMode.ALPHA
    plot._apply_blending_policy()
    assert blend_calls == [("func", "GL_SRC_ALPHA", "GL_ONE_MINUS_SRC_ALPHA")]


def test_cache_capture_accumulates_alpha_as_coverage(blend_calls):
    """Into the premultiplied cache, alpha must accumulate with ``GL_ONE``.

    ``GL_SRC_ALPHA`` there is what squared the coverage and bleached the impostor.
    """
    plot = GPULinePlot()
    plot.options.blend_mode = BlendMode.ALPHA
    plot._apply_blending_policy(premultiplied_target=True)
    assert blend_calls == [
        ("separate", "GL_SRC_ALPHA", "GL_ONE_MINUS_SRC_ALPHA", "GL_ONE", "GL_ONE_MINUS_SRC_ALPHA")
    ]


def test_cache_capture_keeps_the_colour_factors_of_its_blend_mode(blend_calls):
    """Only the alpha channel is corrected; the mode still decides how colour combines."""
    plot = GPULinePlot()
    plot.options.blend_mode = BlendMode.SCREEN
    plot._apply_blending_policy(premultiplied_target=True)
    assert blend_calls == [
        ("separate", "GL_ONE", "GL_ONE_MINUS_SRC_COLOR", "GL_ONE", "GL_ONE_MINUS_SRC_ALPHA")
    ]


def test_alpha_squaring_is_what_the_old_factors_did():
    """Why ``GL_SRC_ALPHA`` on the alpha channel bleaches a translucent cloud.

    Replays both blend functions on the CPU over the cache's transparent-black clear, then
    composites the result over the white page the way ``draw_cached_impostor`` does.
    """
    a, colour = 0.1, 0.0  # a black marker, the most visible thing on a white page

    def accumulate(alpha_src_factor: float) -> tuple[float, float]:
        rgb = acc = 0.0
        for _ in range(20):  # 20 overlapping markers, an ordinary count for a dense cloud
            rgb = colour * a + rgb * (1 - a)
            acc = a * alpha_src_factor + acc * (1 - a)
        return rgb, acc

    rgb_old, alpha_old = accumulate(a)  # GL_SRC_ALPHA: stores a*a
    rgb_new, alpha_new = accumulate(1.0)  # GL_ONE: stores a

    assert rgb_old == pytest.approx(rgb_new)  # colour was never the problem
    over_white_old = rgb_old + 1.0 * (1 - alpha_old)
    over_white_new = rgb_new + 1.0 * (1 - alpha_new)
    exact = (1 - a) ** 20  # what the same 20 markers give when drawn straight to the page

    assert over_white_new == pytest.approx(exact, abs=1e-6)
    assert over_white_old > 0.9  # bleached: 20 black markers and the page stays white
    assert over_white_old - exact > 0.7


# ---------------------------------------------------------------------------
# 2. Capture-scale density compensation
# ---------------------------------------------------------------------------


def _layer(n: int, point_size: float = 2.0, span: float = 10.0) -> ScatterLayer:
    """A cloud of ``n`` points filling a ``span``-wide square of world space."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-span / 2, span / 2, size=(n, 2)).astype(np.float32)
    layer = ScatterLayer(pts=pts)
    layer.style = LayerStyle(point_size=point_size)
    return layer


def _ctx(capture_scale: float, span: float = 10.0) -> RenderContext:
    """A context whose world window is ``capture_scale`` times the 10-unit view."""
    half = span * capture_scale / 2
    return RenderContext(
        mvp=np.eye(4, dtype=np.float32),
        window_world=(-half, half, -half, half),
        width_px=FB_W // 2,
        height_px=FB_H // 2,
        fb_width=FB_W,
        fb_height=FB_H,
        mode=RenderMode.INTERACTIVE,
        dpr=2.0,
        capture_scale=capture_scale,
    )


@pytest.fixture
def renderer() -> ScatterRenderer:
    return ScatterRenderer(EngineOptions())


def test_direct_render_is_never_decimated_by_this(renderer):
    """``capture_scale`` is 1.0 for every pass but the capture, and must cost nothing."""
    layer = _layer(2_000_000)
    assert renderer._lod_stride(layer, _ctx(1.0)) == 1


def test_capture_decimates_by_the_square_of_its_scale(renderer):
    """A 3x wider capture covers 9x the area, so it may draw a ninth of the points."""
    layer = _layer(2_000_000)
    assert renderer._lod_stride(layer, _ctx(3.0)) == 9


def test_compensation_tracks_the_configured_padding(renderer):
    layer = _layer(2_000_000)
    assert renderer._lod_stride(layer, _ctx(2.0)) == 4


def test_a_sparse_cloud_keeps_every_point(renderer):
    """Decimation may only remove overdraw.

    5k markers over a 2560x1600 framebuffer do not even cover it once, so there is no
    marker hiding behind another to drop -- taking 8 of every 9 would make points the user
    can see individually blink out for the length of the drag.
    """
    layer = _layer(5_000)
    assert renderer._lod_stride(layer, _ctx(3.0)) == 1


def test_the_cap_is_coverage_not_a_fixed_count(renderer):
    """The same cloud may be decimated once its markers are fat enough to overlap."""
    n = 300_000
    thin = renderer._lod_stride(_layer(n, point_size=1.0), _ctx(3.0))
    fat = renderer._lod_stride(_layer(n, point_size=8.0), _ctx(3.0))
    assert thin < fat == 9


def test_compensation_composes_with_the_zoom_out_budget(renderer):
    """Whichever reason to decimate is stronger wins; they are not multiplied.

    A cloud zoomed far out already has a stride from the points-per-pixel budget. The
    capture's own stride is the same arithmetic on a 9x smaller on-screen box, so it is
    already >= 9 there and the compensation is a floor, not a second cut.
    """
    layer = _layer(20_000_000, span=10.0)
    zoomed_out = _ctx(3.0, span=10_000.0)  # the cloud is a few px across
    assert renderer._lod_stride(layer, zoomed_out) >= 9
