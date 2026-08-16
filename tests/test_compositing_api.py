"""The public surface for 3D compositing: ``set_layer_compositing`` and the verb kwargs.

The renderer-side behaviour is in ``test_compositing_3d.py``; this pins the API that reaches
it -- that ``volume3d(blend="additive")`` and ``set_layer_compositing(layer, ...)`` land on
the three ``LayerStyle`` fields, that the string spellings resolve, and that the "hand it
back" sentinels (``blend="figure"``, ``auto_alpha=0``) clear an override.
"""

from __future__ import annotations

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.options import BlendMode


@pytest.fixture(autouse=True)
def _clean():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


class TestSetLayerCompositing:
    def test_string_blend_mode_resolves(self):
        layer = gplt.scatter3d([0, 1], [0, 1], [0, 1])
        gplt.set_layer_compositing(layer, blend="additive")
        assert layer.style.blend_mode is BlendMode.ADDITIVE

    def test_blend_mode_is_case_insensitive(self):
        layer = gplt.scatter3d([0, 1], [0, 1], [0, 1])
        gplt.set_layer_compositing(layer, blend="Screen")
        assert layer.style.blend_mode is BlendMode.SCREEN

    def test_an_enum_blend_mode_is_taken_directly(self):
        layer = gplt.scatter3d([0, 1], [0, 1], [0, 1])
        gplt.set_layer_compositing(layer, blend=BlendMode.SUBTRACTIVE)
        assert layer.style.blend_mode is BlendMode.SUBTRACTIVE

    def test_figure_hands_the_mode_back(self):
        layer = gplt.scatter3d([0, 1], [0, 1], [0, 1])
        gplt.set_layer_compositing(layer, blend="additive")
        gplt.set_layer_compositing(layer, blend="figure")
        assert layer.style.blend_mode is None

    def test_an_unknown_mode_raises(self):
        layer = gplt.scatter3d([0, 1], [0, 1], [0, 1])
        with pytest.raises(ValueError, match="unknown blend mode"):
            gplt.set_layer_compositing(layer, blend="glow")

    def test_depth_write_is_stored_as_a_bool(self):
        layer = gplt.scatter3d([0, 1], [0, 1], [0, 1])
        gplt.set_layer_compositing(layer, depth_write=False)
        assert layer.style.depth_write is False

    def test_auto_alpha_is_stored(self):
        layer = gplt.scatter3d([0, 1], [0, 1], [0, 1])
        gplt.set_layer_compositing(layer, auto_alpha=0.7)
        assert layer.style.auto_alpha == pytest.approx(0.7)

    def test_auto_alpha_zero_clears_it(self):
        layer = gplt.scatter3d([0, 1], [0, 1], [0, 1])
        gplt.set_layer_compositing(layer, auto_alpha=0.7)
        gplt.set_layer_compositing(layer, auto_alpha=0.0)
        assert layer.style.auto_alpha is None

    def test_an_omitted_argument_is_left_alone(self):
        layer = gplt.scatter3d([0, 1], [0, 1], [0, 1])
        gplt.set_layer_compositing(layer, blend="additive", auto_alpha=0.8)
        gplt.set_layer_compositing(layer, depth_write=True)  # touches only depth_write
        assert layer.style.blend_mode is BlendMode.ADDITIVE
        assert layer.style.auto_alpha == pytest.approx(0.8)
        assert layer.style.depth_write is True

    def test_it_returns_the_layer(self):
        layer = gplt.scatter3d([0, 1], [0, 1], [0, 1])
        assert gplt.set_layer_compositing(layer, blend="screen") is layer


class TestVerbKwargs:
    def test_volume3d_blend(self):
        x = np.linspace(0, 1, 20)
        layer = gplt.volume3d(x, x, x, x, blend="additive")
        assert layer.style.blend_mode is BlendMode.ADDITIVE

    def test_volume3d_auto_alpha(self):
        x = np.linspace(0, 1, 20)
        layer = gplt.volume3d(x, x, x, x, auto_alpha=0.9)
        assert layer.style.auto_alpha == pytest.approx(0.9)

    def test_scatter3d_kwargs(self):
        x = np.linspace(0, 1, 10)
        layer = gplt.scatter3d(x, x, x, blend="screen", depth_write=True)
        assert layer.style.blend_mode is BlendMode.SCREEN
        assert layer.style.depth_write is True

    def test_plot_surface_depth_write(self):
        g = np.linspace(-1, 1, 6)
        X, Y = np.meshgrid(g, g)
        Z = X**2 + Y**2
        layer = gplt.plot_surface(X, Y, Z, depth_write=False)
        assert layer.style.depth_write is False

    def test_a_verb_without_the_kwargs_leaves_the_defaults(self):
        x = np.linspace(0, 1, 10)
        layer = gplt.scatter3d(x, x, x)
        assert layer.style.blend_mode is None
        assert layer.style.depth_write is None
        assert layer.style.auto_alpha is None
