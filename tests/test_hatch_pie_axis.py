"""Hatching, patch edges, the round pie, and ``axis("off")`` in a saved figure.

All four were gaps in the *export*, which is a real matplotlib figure: ``hatch=`` was accepted and
ignored, an explicit edge on a ``fill_between`` stroked every triangle of its triangulation,
``pie`` came out an ellipse, and ``axis("off")`` was dropped by every 2D export. Inspected through
``_build_preview_figure`` so the assertions are about what was drawn.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.utils.preview import _boundary_segments, _build_preview_figure

X = np.linspace(0.0, 6.0, 60)


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


def export(fig):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _build_preview_figure(fig, 1.0)


def hatches(ax):
    """The hatch of every patch and collection on ``ax``, in draw order."""
    return [p.get_hatch() for p in ax.patches] + [c.get_hatch() for c in ax.collections]


# ---------------------------------------------------------------------------------------
# hatch=
# ---------------------------------------------------------------------------------------


class TestHatch:
    def test_bar(self):
        fig, ax = gplt.subplots()
        ax.bar([0, 1, 2], [1, 2, 3], hatch="//")
        assert hatches(export(fig).axes[0]) == ["//"] * 3

    def test_barh(self):
        fig, ax = gplt.subplots()
        ax.barh([0, 1], [3, 4], hatch="xx")
        assert hatches(export(fig).axes[0]) == ["xx", "xx"]

    def test_hist(self):
        fig, ax = gplt.subplots()
        ax.hist(np.random.default_rng(0).normal(size=100), bins=5, hatch="..")
        assert hatches(export(fig).axes[0]) == [".."] * 5

    def test_fill_between(self):
        fig, ax = gplt.subplots()
        ax.fill_between(X, np.sin(X), 0, hatch="//")
        assert hatches(export(fig).axes[0]) == ["//"]

    def test_fill_betweenx(self):
        fig, ax = gplt.subplots()
        ax.fill_betweenx(X, np.sin(X), 0, hatch="\\\\")
        assert hatches(export(fig).axes[0]) == ["\\\\"]

    def test_fill(self):
        fig, ax = gplt.subplots()
        ax.fill([0, 2, 1], [0, 0, 2], hatch="--")
        assert hatches(export(fig).axes[0]) == ["--"]

    def test_stackplot_cycles_a_sequence_over_the_bands(self):
        fig, ax = gplt.subplots()
        ax.stackplot(X, np.sin(X) + 2, np.cos(X) + 2, np.ones_like(X), hatch=["//", ".."])
        assert hatches(export(fig).axes[0]) == ["//", "..", "//"]

    def test_pie_takes_one_per_wedge(self):
        fig, ax = gplt.subplots()
        ax.pie([3, 2, 1], hatch=["//", "..", "xx"])
        assert hatches(export(fig).axes[0]) == ["//", "..", "xx"]

    def test_an_unhatched_patch_has_no_hatch(self):
        fig, ax = gplt.subplots()
        ax.bar([0, 1], [1, 2])
        assert hatches(export(fig).axes[0]) == [None, None] or not any(hatches(export(fig).axes[0]))

    def test_without_an_edge_the_hatch_is_black_and_has_no_outline(self):
        fig, ax = gplt.subplots()
        ax.bar([0], [1], color="tab:blue", hatch="//")
        (patch,) = export(fig).axes[0].patches
        assert tuple(patch.get_edgecolor()[:3]) == (0.0, 0.0, 0.0)
        assert patch.get_linewidth() == 0.0

    def test_with_an_edge_the_hatch_takes_its_colour_and_the_outline_stays(self):
        fig, ax = gplt.subplots()
        ax.bar([0], [1], color="tab:blue", hatch="//", edgecolor="tab:red", linewidth=2)
        (patch,) = export(fig).axes[0].patches
        assert patch.get_edgecolor()[:3] == pytest.approx((0.839, 0.153, 0.157), abs=0.01)
        assert patch.get_linewidth() == 2.0

    def test_the_window_fills_solid_and_the_call_says_so(self):
        fig, ax = gplt.subplots()
        with pytest.warns(gplt.MatplotlibCompatWarning, match="exported figures only"):
            (bar,) = ax.bar([0], [1], hatch="//")
        assert bar.metadata["hatch"] == "//"  # kept for the export, never drawn by the GL path

    def test_the_legend_handle_carries_the_hatch(self):
        fig, ax = gplt.subplots()
        ax.bar([0, 1], [1, 2], hatch="//", label="bars")
        ax.legend()
        legend = export(fig).axes[0].get_legend()
        assert legend.get_patches()[0].get_hatch() == "//"


# ---------------------------------------------------------------------------------------
# Patch edges: the outline of the shape, not of each triangle
# ---------------------------------------------------------------------------------------


class TestBoundarySegments:
    def test_a_quad_is_four_segments(self):
        vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        segments = _boundary_segments(vertices, [0, 1, 2, 0, 2, 3])
        assert segments.shape == (4, 2, 2)

    def test_the_shared_diagonal_is_not_on_the_boundary(self):
        vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        segments = _boundary_segments(vertices, [0, 1, 2, 0, 2, 3])
        diagonal = {tuple(map(tuple, seg)) for seg in segments} & {
            ((0.0, 0.0), (1.0, 1.0)),
            ((1.0, 1.0), (0.0, 0.0)),
        }
        assert not diagonal

    def test_a_fan_triangulated_polygon_has_one_segment_per_side(self):
        n = 7
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        vertices = np.column_stack([np.cos(angles), np.sin(angles)])
        fan = np.arange(1, n - 1)
        triangles = np.column_stack([np.zeros_like(fan), fan, fan + 1]).ravel()
        assert len(_boundary_segments(vertices, triangles)) == n

    def test_two_separate_shapes_keep_their_own_outlines(self):
        vertices = np.array([[0, 0], [1, 0], [0, 1], [5, 5], [6, 5], [5, 6]], dtype=float)
        assert len(_boundary_segments(vertices, [0, 1, 2, 3, 4, 5])) == 6


class TestPatchEdges:
    def test_fill_between_edge_is_one_outline_and_the_triangles_are_not_stroked(self):
        from matplotlib.collections import LineCollection, PolyCollection

        fig, ax = gplt.subplots()
        ax.fill_between(X, np.sin(X), 0, edgecolor="k", linewidth=2)
        axes = export(fig).axes[0]
        (fill,) = [c for c in axes.collections if isinstance(c, PolyCollection)]
        (outline,) = [c for c in axes.collections if isinstance(c, LineCollection)]
        assert float(fill.get_linewidth()[0]) == 0.0
        assert float(outline.get_linewidth()[0]) == 2.0
        # the outline of a band under a curve is far fewer segments than its triangles' edges
        assert len(outline.get_segments()) < 3 * (len(X) - 1)

    def test_no_explicit_edge_means_no_extra_outline(self):
        from matplotlib.collections import LineCollection

        fig, ax = gplt.subplots()
        ax.fill_between(X, np.sin(X), 0)
        assert not [c for c in export(fig).axes[0].collections if isinstance(c, LineCollection)]

    def test_fill_takes_facecolor_edgecolor_and_linewidth(self):
        fig, ax = gplt.subplots()
        (layer,) = ax.fill([0, 2, 1], [0, 0, 2], facecolor="tab:green", edgecolor="k", linewidth=3)
        assert layer.style.face_color[:3] == pytest.approx((0.173, 0.627, 0.173), abs=0.01)
        assert layer.style.line_width == 3.0 and layer.style.outline_enabled

    def test_fill_between_facecolor(self):
        fig, ax = gplt.subplots()
        layer = ax.fill_between(X, np.sin(X), 0, facecolor="tab:red", alpha=0.5)
        assert layer.style.face_color[:3] == pytest.approx((0.839, 0.153, 0.157), abs=0.01)
        assert layer.style.face_color[3] == pytest.approx(0.5)

    def test_the_edge_is_also_the_live_outline(self):
        fig, ax = gplt.subplots()
        layer = ax.fill_between(X, np.sin(X), 0, edgecolor="k", linewidth=2)
        assert layer.style.outline_enabled and layer.style.outline_width == 2.0
        assert layer.metadata["edge_as_outline"] is True

    def test_barh_takes_the_bar_edge_keywords(self):
        fig, ax = gplt.subplots()
        (bar,) = ax.barh([0], [3], color="tab:orange", edgecolor="k", linewidth=3)
        assert bar.style.outline_enabled and bar.style.outline_width == 3.0
        (patch,) = export(fig).axes[0].patches
        assert patch.get_linewidth() == 3.0 and len(patch.get_xy()) == 5

    def test_barh_ec_lw_aliases_and_tick_label(self):
        fig, ax = gplt.subplots()
        ax.barh([0, 1], [3, 4], ec="r", lw=2, tick_label=["a", "b"])
        assert list(ax.get_yticks()) == [0.0, 1.0]
        assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b"]

    def test_barh_log_is_accepted_and_says_it_does_nothing(self):
        fig, ax = gplt.subplots()
        with pytest.warns(gplt.MatplotlibCompatWarning, match="xscale"):
            ax.barh([0], [3], log=True)


# ---------------------------------------------------------------------------------------
# pie(): round, framed, no axes -- as matplotlib's own
# ---------------------------------------------------------------------------------------


class TestPie:
    def test_it_sets_an_equal_aspect(self):
        fig, ax = gplt.subplots()
        ax.pie([3, 2, 1])
        assert fig.panels[0].aspect_equal is True
        assert export(fig).axes[0].get_aspect() == 1.0

    def test_the_view_is_framed_around_the_centre(self):
        fig, ax = gplt.subplots()
        ax.pie([3, 2, 1], center=(2.0, -1.0))
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        assert (xlo + xhi) / 2 == pytest.approx(2.0) and (ylo + yhi) / 2 == pytest.approx(-1.0)
        assert xhi - xlo >= 2.5 - 1e-6 and yhi - ylo >= 2.5 - 1e-6

    def test_the_frame_and_ticks_are_gone_in_the_export(self):
        fig, ax = gplt.subplots()
        ax.pie([3, 2, 1])
        assert export(fig).axes[0].axison is False

    def test_frame_true_keeps_the_axes(self):
        fig, ax = gplt.subplots()
        ax.pie([3, 2, 1], frame=True)
        assert export(fig).axes[0].axison is True and export(fig).axes[0].get_aspect() == 1.0

    def test_the_wedges_are_round_whatever_the_figure_shape(self):
        fig, ax = gplt.subplots()
        ax.pie([1, 1])
        figure = export(fig)
        figure.canvas.draw()
        box = figure.axes[0].get_window_extent()
        xlo, xhi = figure.axes[0].get_xlim()
        ylo, yhi = figure.axes[0].get_ylim()
        pixels_per_x = box.width / (xhi - xlo)
        pixels_per_y = box.height / (yhi - ylo)
        assert pixels_per_x == pytest.approx(pixels_per_y, rel=1e-3)


# ---------------------------------------------------------------------------------------
# axis("off")
# ---------------------------------------------------------------------------------------


class TestAxisOff:
    def test_a_saved_figure_has_no_axes_apparatus(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        ax.axis("off")
        assert export(fig).axes[0].axison is False

    def test_set_axis_off_is_the_same(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        ax.set_axis_off()
        assert export(fig).axes[0].axison is False

    def test_the_title_stays(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        ax.set_title("kept")
        ax.axis("off")
        assert export(fig).axes[0].get_title() == "kept"

    def test_the_artists_stay(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        ax.axis("off")
        assert len(export(fig).axes[0].lines) == 1

    def test_it_is_per_axes(self):
        fig, axs = gplt.subplots(1, 3)
        for a in axs:
            a.plot(X, X)
        axs[1].axis("off")
        assert [a.axison for a in export(fig).axes] == [True, False, True]

    def test_axis_on_restores_it(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        ax.axis("off")
        ax.axis("on")
        assert export(fig).axes[0].axison is True

    def test_a_figure_that_never_called_it_is_unchanged(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        assert export(fig).axes[0].axison is True

    def test_it_applies_to_a_twin_panel_too(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        ax.twinx().plot(X, X * 100)
        ax.axis("off")
        assert export(fig).axes[0].axison is False
