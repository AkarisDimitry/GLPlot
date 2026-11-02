"""The 3Blue1Brown preset and the part of a preset that shapes the exported picture.

``PlotStyle.export_look`` is read by the headless export only (``utils/preview.py``): text face,
maths font, axis lines through the origin with tips, a NumberPlane grid. These tests read the
finished matplotlib figure through ``render_preview_bytes(inspect=...)``, the one render that
produced the bytes, so they check what is drawn rather than what was asked for.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from matplotlib.colors import to_hex

import glplot.pyplot as gplt
from glplot.gui import styles
from glplot.utils.preview import render_preview_bytes


@pytest.fixture(autouse=True)
def _close():
    yield
    gplt.close("all")


def finished(plot):
    """The exported figure's first axes, read while it is still open (and still in its fonts)."""
    seen = {}

    def inspect(fig):
        ax = fig.axes[0]
        fig.canvas.draw()
        seen["ax"] = ax
        seen["rc"] = {k: matplotlib.rcParams[k] for k in ("mathtext.fontset", "font.family")}
        seen["xticks"] = [t.get_text() for t in ax.get_xticklabels()]
        seen["xtick_values"] = list(ax.get_xticks())
        seen["legend"] = ax.get_legend()
        seen["artists"] = list(ax.lines)
        return None

    render_preview_bytes(plot, 1.0, inspect=inspect)
    return seen


def crossing_figure(style="3b1b"):
    plot = gplt.figure(figsize=(6, 4))
    if style:
        gplt.plot_style(style)
    x = np.linspace(-2, 2, 50)
    gplt.plot(x, x**2 - 1, label="parabola")
    gplt.xlim(-2.5, 2.5)
    gplt.ylim(-2, 4)
    gplt.xlabel("x")
    gplt.ylabel("y")
    gplt.grid(True)
    return plot


class TestThePreset:
    def test_it_is_manim_s_look(self):
        style = styles.get_style("3b1b")
        assert to_hex(style.palette[0][:3]) == "#58c4dd"  # BLUE_C
        assert to_hex(style.palette[1][:3]) == "#f7d96f"  # YELLOW_C
        assert style.background == (0.0, 0.0, 0.0) and style.cmap == "manim"
        look = style.export_look
        assert look.math_fontset == "cm" and look.axes == "cross" and look.arrow_tips
        assert to_hex(look.grid_color) == "#29abca"  # BLUE_D, a NumberPlane's lines
        assert look.faded_lines == 4

    def test_the_default_look_keeps_the_historical_export(self):
        """A look is opt-in: Clean (the default), Colour-safe and Dense export the plain box."""
        plain = [s.key for s in styles.STYLES if s.export_look is None]
        assert plain == ["clean", "colorsafe", "dense"]

    def test_it_is_recorded_on_the_figure_and_undone(self):
        plot = gplt.figure()
        state = styles.capture_state(plot)
        gplt.plot_style("3b1b")
        assert plot.options.export_look is styles.get_style("3b1b").export_look
        styles.restore_state(plot, state)
        assert plot.options.export_look is None


class TestWhatIsDrawnNext:
    def test_new_lines_take_the_presets_palette_and_width(self):
        gplt.figure()
        gplt.plot_style("3b1b")
        first = gplt.plot([0, 1], [0, 1])
        second = gplt.plot([0, 1], [1, 0])
        layers = gplt.gcf().scene.layers
        colours = [to_hex(tuple(layer.style.color[:3])) for layer in layers[-2:]]
        assert colours == ["#58c4dd", "#f7d96f"] and first is not None and second is not None
        assert layers[-1].style.line_width == pytest.approx(3.2)

    def test_the_cycle_continues_after_what_was_repainted(self):
        gplt.figure()
        gplt.plot([0, 1], [0, 1])
        gplt.plot_style("3b1b")  # repaints this line BLUE_C
        gplt.plot([0, 1], [1, 0])
        colours = [to_hex(tuple(layer.style.color[:3])) for layer in gplt.gcf().scene.layers]
        assert colours == ["#58c4dd", "#f7d96f"]

    def test_unset_text_is_the_presets_ink_and_black_without_one(self):
        gplt.figure()
        gplt.text(0, 0, "plain")
        assert tuple(gplt.gcf().scene.layers[-1].style.color[:3]) == (0.0, 0.0, 0.0)
        gplt.plot_style("3b1b")
        gplt.text(0, 0, "on black")
        assert tuple(gplt.gcf().scene.layers[-1].style.color[:3]) == (1.0, 1.0, 1.0)
        gplt.text(0, 0, "chosen", color="red")
        assert tuple(gplt.gcf().scene.layers[-1].style.color[:3]) == (1.0, 0.0, 0.0)


class TestTheExport:
    def test_axes_cross_at_the_origin_without_a_box(self):
        seen = finished(crossing_figure())
        ax = seen["ax"]
        assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
        assert ax.spines["bottom"].get_position() == ("data", 0.0)
        assert ax.spines["left"].get_position() == ("data", 0.0)

    def test_the_axis_lines_have_tips_and_the_titles_sit_beside_them(self):
        seen = finished(crossing_figure())
        tips = [line for line in seen["artists"] if line.get_marker() in (">", "^")]
        assert {line.get_marker() for line in tips} == {">", "^"}
        ax = seen["ax"]
        assert ax.yaxis.label.get_rotation() == 0
        assert ax.xaxis.label.get_horizontalalignment() == "right"

    def test_the_number_at_the_crossing_is_left_out(self):
        seen = finished(crossing_figure())
        labels = dict(zip(seen["xtick_values"], seen["xticks"]))
        assert labels[0.0] == "" and any(text for value, text in labels.items() if value != 0)

    def test_the_grid_is_a_number_plane(self):
        seen = finished(crossing_figure())
        ax = seen["ax"]
        major = ax.xaxis.get_gridlines()[0]
        assert to_hex(major.get_color()) == "#29abca" and major.get_alpha() == pytest.approx(0.6)
        assert ax.xaxis.get_minor_locator().__class__.__name__ == "AutoMinorLocator"

    def test_text_is_set_in_the_presets_faces(self):
        seen = finished(crossing_figure())
        assert seen["rc"]["mathtext.fontset"] == "cm"
        installed = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
        expected = [f for f in styles.get_style("3b1b").export_look.font_family if f in installed]
        assert seen["rc"]["font.family"] == expected

    def test_the_fonts_do_not_leak_out_of_the_export(self):
        before = matplotlib.rcParams["mathtext.fontset"]
        finished(crossing_figure())
        assert matplotlib.rcParams["mathtext.fontset"] == before

    def test_away_from_the_origin_the_lines_stay_on_the_edges(self):
        plot = crossing_figure()
        gplt.xlim(1, 3)
        gplt.ylim(1, 5)
        ax = finished(plot)["ax"]
        assert ax.spines["bottom"].get_position() != ("data", 0.0)
        assert not ax.spines["top"].get_visible()

    def test_a_preset_without_a_look_exports_the_usual_box(self):
        ax = finished(crossing_figure(style="colorsafe"))["ax"]
        assert ax.spines["top"].get_visible() and ax.spines["right"].get_visible()

    def test_a_removed_legend_is_not_exported(self):
        plot = crossing_figure(style=None)
        assert finished(plot)["legend"] is not None
        gplt.legend()
        gplt.gca().get_legend().remove()
        assert finished(plot)["legend"] is None


class TestTextBeingWritten:
    def left_edge(self, text, **kwargs):
        gplt.figure(figsize=(6, 3))
        gplt.plot([0, 10], [0, 1])
        gplt.xlim(0, 10)
        gplt.ylim(0, 1)
        gplt.text(5, 0.5, text, fontsize=20, ha="center", **kwargs)
        edges = []

        def inspect(fig):
            renderer = fig.canvas.get_renderer()
            (label,) = [t for t in fig.axes[0].texts]
            edges.append(label.get_window_extent(renderer).x0)

        render_preview_bytes(gplt.get_engine(), 1.0, inspect=inspect)
        gplt.close("all")
        return edges[0]

    def test_a_centred_text_being_written_starts_where_the_whole_one_does(self):
        whole = self.left_edge("Where does it come from?")
        partial = self.left_edge("Where", layout_text="Where does it come from?")
        alone = self.left_edge("Where")
        assert partial == pytest.approx(whole, abs=1.0), "written in place"
        assert alone > whole + 20, "without layout_text it would be centred on its own"
