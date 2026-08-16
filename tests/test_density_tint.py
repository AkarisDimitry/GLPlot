"""Density mode painted in the layers' own colours.

The bug: the density accumulator was a single-channel buffer holding "how much landed on
this pixel", and the resolve turned that scalar into a colour through one global colormap.
A layer's own colour never entered the pass at all, so ``scatter(x, y, color="red")`` in
density mode came out in whatever ``density_scheme_index`` said -- setting a colour did
nothing, which is what was reported.

The buffer now carries ``(sum(w), sum(w*rgb))`` per pixel, so the resolve can recover the
weight-averaged colour of what landed there and paint with it at an opacity given by the
density. Verified on the real GL renderer: a red scatter resolves to (0.99, 0.25, 0.25) at
the centre, and a red and a blue scatter side by side stay red and blue and mix where they
overlap.

What is checked here is the *policy* -- which passes tint and which do not -- because that
is the part with a wrong answer that looks plausible. Tinting is deliberately not "the layer
has colours" (every layer does; scatter's default is black, and tinting on that paints the
image black on black) but "the caller asked for one".
"""

from __future__ import annotations

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.core.layers import PolylineLayer, ScatterLayer
from glplot.managers.renderer_manager import RendererManager


def marked(explicit: bool) -> ScatterLayer:
    layer = ScatterLayer()
    layer.metadata["explicit_color"] = explicit
    return layer


class TestTintPolicy:
    def test_a_layer_whose_colour_was_asked_for_turns_tinting_on(self):
        assert RendererManager.density_tint_active([marked(True)]) is True

    def test_a_layer_that_took_the_default_does_not(self):
        assert RendererManager.density_tint_active([marked(False)]) is False

    def test_a_layer_that_says_nothing_does_not(self):
        """Every layer type that never sets the key keeps the heatmap it always had."""
        assert RendererManager.density_tint_active([ScatterLayer(), PolylineLayer()]) is False

    def test_one_explicit_layer_is_enough(self):
        """The pass has a single resolve, so the mode is a property of the pass."""
        layers = [marked(False), ScatterLayer(), marked(True)]
        assert RendererManager.density_tint_active(layers) is True

    def test_an_empty_pass_is_not_tinted(self):
        assert RendererManager.density_tint_active([]) is False


class TestExplicitColorIsRecorded:
    """What ``pyplot`` marks, since the policy above is only as good as this flag."""

    def setup_method(self):
        gplt._cleanup_pyplot_state()

    def teardown_method(self):
        gplt._cleanup_pyplot_state()

    def test_scatter_with_a_named_colour(self):
        layer = gplt.scatter([0, 1], [0, 1], color="red")
        assert layer.metadata["explicit_color"] is True

    def test_scatter_with_colormapped_values(self):
        """``c=`` is a colour choice too -- and the one whose colours a density plot most
        obviously ought to keep."""
        layer = gplt.scatter([0, 1], [0, 1], c=[0.0, 1.0], cmap="viridis")
        assert layer.metadata["explicit_color"] is True

    def test_scatter_with_no_colour_at_all(self):
        """Its colours are the default black; tinting on that would paint black on black."""
        layer = gplt.scatter([0, 1], [0, 1])
        assert layer.metadata["explicit_color"] is False

    def test_plot_with_a_named_colour(self):
        (line,) = gplt.plot([0, 1], [0, 1], color="green")
        assert line.metadata["explicit_color"] is True

    def test_plot_taking_the_next_cycle_colour(self):
        """A cycle colour is one the *library* chose, and it has never governed density."""
        (line,) = gplt.plot([0, 1], [0, 1])
        assert line.metadata["explicit_color"] is False

    def test_a_marker_layer_is_marked_with_its_line(self):
        artists = gplt.plot([0, 1], [0, 1], marker="o", color="purple")
        assert [a.metadata["explicit_color"] for a in artists] == [True, True]

    def test_plot_lines_keeps_the_heatmap(self):
        """The bulk verb: the heatmap *is* the point, and its ``colors=`` is usually a flat
        fill chosen to weigh every line the same (the density benchmark passes white)."""
        gplt.plot_lines(np.array([0.5, 1.0]), np.array([0.0, 1.0]), x_range=(0.0, 1.0))
        layer = gplt.gcf().scene.layers[-1]
        assert not layer.metadata.get("explicit_color")


class TestEngineGate:
    """``density_tint_active`` is asked by the background pass too, and the two answers have
    to agree -- a tinted resolve is composited, so it needs the ordinary background."""

    def setup_method(self):
        gplt._cleanup_pyplot_state()

    def teardown_method(self):
        gplt._cleanup_pyplot_state()

    def test_off_when_density_is_off(self):
        fig = gplt.figure()
        gplt.scatter([0, 1], [0, 1], color="red")
        assert fig.display_density is False
        assert fig.density_tint_active() is False

    def test_on_for_an_explicitly_coloured_scatter_in_density_mode(self):
        fig = gplt.figure(density=True)
        gplt.scatter([0, 1], [0, 1], color="red")
        assert fig.density_tint_active() is True

    def test_off_for_a_default_scatter_in_density_mode(self):
        fig = gplt.figure(density=True)
        gplt.scatter([0, 1], [0, 1])
        assert fig.density_tint_active() is False

    def test_off_in_a_3d_scene(self):
        """Density is a 2D accumulator and does not run at all in 3D
        (``engine._density_active``), so neither does its tint."""
        fig = gplt.figure(density=True)
        gplt.scatter3d([0, 1], [0, 1], [0, 1], color="red")
        assert fig.density_tint_active() is False

    def test_a_hidden_layer_does_not_decide_the_mode(self):
        """The pass filters on visibility, so the gate has to as well."""
        fig = gplt.figure(density=True)
        layer = gplt.scatter([0, 1], [0, 1], color="red")
        layer.style.visible = False
        assert fig.density_tint_active() is False


class TestAccumulationBuffer:
    """The buffer's shape, which every accumulation shader has to agree on."""

    def test_the_weight_is_the_red_channel(self):
        """Both readbacks ask the driver for ``GL_RED``; putting the weight anywhere else
        would silently return a premultiplied colour sum as if it were a density."""
        from glplot.utils import shaders

        assert "return vec4(w, rgb * w);" in shaders.DENSITY_WEIGHT_TO_RGBA

    @pytest.mark.parametrize(
        "name",
        ["DENSITY_ACCUM_FS", "DENSITY_POINTS_FS", "WIDE_SEGMENT_DENSITY_FS"],
    )
    def test_every_accumulation_shader_writes_the_same_vec4(self, name):
        """Three shaders feed one buffer. A ``float`` output left behind in any of them
        would write the weight into red and leave the colour channels undefined."""
        from glplot.utils import shaders

        source = getattr(shaders, name)
        assert "layout(location=0) out vec4 FragValue;" in source
        assert "density_sample(" in source

    def test_the_resolve_reads_weight_and_colour_from_the_matching_channels(self):
        from glplot.utils import shaders

        assert "float val = acc.r;" in shaders.DENSITY_RESOLVE_FS
        assert "acc.gba / max(acc.r, 1e-6)" in shaders.DENSITY_RESOLVE_FS
