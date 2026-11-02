"""The borrowed looks (newsrooms, chart grammars, creators, editor themes) and what a style
preset now reaches in a figure: the colour of every artist drawn without one, the export's
text colour, its faces and its axis furniture.

The export is read through ``render_preview_bytes(inspect=...)``: the matplotlib figure that
produced the bytes, so what is checked is what was drawn.
"""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pytest
from matplotlib.colors import to_hex

import glplot.pyplot as gplt
from glplot.gui import styles
from glplot.utils import preview
from glplot.utils.preview import _cm_safe, render_preview_bytes

BORROWED = (
    "kurzgesagt",
    "economist",
    "ft",
    "fivethirtyeight",
    "ggplot",
    "tufte",
    "nord",
    "catppuccin",
    "tokyonight",
    "material",
    "terminal",
)


@pytest.fixture(autouse=True)
def _close():
    yield
    gplt.close("all")


def hexes(layer):
    colors = np.asarray(layer.colors if getattr(layer, "colors", None) is not None else [])
    if colors.size:
        return to_hex(tuple(colors[0][:3]))
    return to_hex(tuple(layer.style.face_color[:3]))


def finished(plot):
    """The exported figure, read while it is still open (and still in its fonts)."""
    seen = {}

    def inspect(fig):
        fig.canvas.draw()
        ax = fig.axes[0]
        seen.update(
            fig=fig,
            ax=ax,
            family=list(matplotlib.rcParams["font.family"]),
            title_family=ax.title.get_fontfamily(),
            title_weight=ax.title.get_fontweight(),
            yticks_right=[t.label2.get_visible() for t in ax.yaxis.get_major_ticks()],
            xgrid=[line.get_visible() for line in ax.xaxis.get_gridlines()],
            ygrid=[line.get_visible() for line in ax.yaxis.get_gridlines()],
            ylabels=[t.get_color() for t in ax.get_yticklabels()],
        )
        return None

    render_preview_bytes(plot, 1.0, inspect=inspect)
    return seen


def chart(style, grid=None):
    plot = gplt.figure(figsize=(6, 4))
    gplt.plot_style(style)
    x = np.linspace(0, 10, 60)
    gplt.plot(x, np.sin(x) + 2, label="signal")
    gplt.bar([1, 3, 5], [-1, -0.5, -0.8], width=0.6, label="bars")
    gplt.title("Quarterly revenue")
    gplt.xlabel("time")
    gplt.ylabel("value")
    gplt.legend()
    if grid is not None:
        gplt.grid(grid)
    return plot


# ======================================================================================
# the registry
# ======================================================================================


class TestTheBorrowedLooks:
    def test_they_ship_after_3b1b_in_order(self):
        assert styles.STYLE_KEYS[-len(BORROWED) :] == BORROWED

    @pytest.mark.parametrize("key", BORROWED)
    def test_each_states_how_its_exports_look(self, key):
        look = styles.get_style(key).export_look
        assert look is not None
        assert look.axes in ("box", "open", "bottom", "none", "cross")
        assert look.grid_axis in ("both", "x", "y") and look.title_loc in ("center", "left")
        assert look.font_family and look.font_family[-1].startswith(
            "DejaVu"
        ), "the last face ships with matplotlib, so the list always resolves"

    @pytest.mark.parametrize("key", BORROWED)
    def test_each_exports_without_error(self, key):
        seen = finished(chart(key))
        assert seen["ax"].get_title() == "Quarterly revenue"

    def test_the_economist(self):
        seen = finished(chart("economist"))
        ax = seen["ax"]
        assert all(seen["yticks_right"]), "the numbers sit on the right"
        assert not any(seen["xgrid"]) and any(seen["ygrid"]), "horizontal rules only"
        assert ax.spines["bottom"].get_visible()
        assert not any(ax.spines[s].get_visible() for s in ("left", "top", "right"))
        assert ax.title.get_horizontalalignment() == "left" and seen["title_weight"] == "bold"
        tag = [a for a in ax.artists if type(a).__name__ == "AnnotationBbox"]
        assert len(tag) == 1, "the red tag above the title"
        assert ax.get_legend() is not None and not ax.get_legend().get_frame().get_visible()

    def test_ggplot_is_a_grey_panel_on_a_white_page_ruled_in_white(self):
        seen = finished(chart("ggplot"))
        ax = seen["ax"]
        assert to_hex(ax.get_facecolor()[:3]) == "#ebebeb"
        assert to_hex(seen["fig"].get_facecolor()[:3]) == "#ffffff"
        assert not any(spine.get_visible() for spine in ax.spines.values())
        assert any(seen["xgrid"]) and any(seen["ygrid"]), "the grid is on without grid(True)"
        assert to_hex(ax.xaxis.get_gridlines()[0].get_color()) == "#ffffff"

    def test_grid_false_still_wins_over_a_look_whose_grid_is_the_design(self):
        seen = finished(chart("fivethirtyeight", grid=False))
        assert not any(seen["xgrid"]) and not any(seen["ygrid"])

    def test_tufte_draws_a_range_frame(self):
        ax = finished(chart("tufte"))["ax"]
        assert ax.spines["left"].get_position() == ("outward", 10)
        low, high = ax.spines["left"].get_bounds()
        ticks = [t for t in ax.get_yticks() if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
        assert (low, high) == (ticks[0], ticks[-1])
        assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()

    def test_a_serif_headline_over_sans_text(self):
        seen = finished(chart("ft"))
        assert seen["title_family"] != seen["family"]


# ======================================================================================
# what a preset reaches
# ======================================================================================


class TestArtistsDrawnWithoutAColour:
    def test_an_unstyled_figure_keeps_glplots_defaults(self):
        gplt.figure()
        gplt.scatter([0, 1], [0, 1])
        gplt.bar([0, 1], [1, 2])
        layers = gplt.gcf().scene.layers
        assert hexes(layers[0]) == "#000000"
        assert hexes(layers[-1]) == to_hex((0.2, 0.4, 0.8))

    def test_on_a_styled_figure_they_walk_the_palette(self):
        gplt.figure()
        gplt.plot_style("dracula")
        palette = [to_hex(c[:3]) for c in styles.get_style("dracula").palette]
        gplt.plot([0, 1], [0, 1])
        gplt.scatter([0, 1], [0, 1])
        gplt.bar([0, 1], [1, 2])
        gplt.fill_between([0, 1], [0, 1], 0)
        layers = gplt.gcf().scene.layers
        line = to_hex(tuple(layers[0].style.color[:3]))
        assert [line, hexes(layers[1]), hexes(layers[2]), hexes(layers[-1])] == palette[:4]

    def test_a_wash_keeps_its_translucency(self):
        gplt.figure()
        gplt.plot_style("nord")
        layer = gplt.fill_between([0, 1], [0, 1], 0)
        layer = layer if hasattr(layer, "style") else gplt.gcf().scene.layers[-1]
        assert layer.style.face_color[3] == pytest.approx(0.35)

    def test_reference_lines_and_arrows_take_the_ink(self):
        gplt.figure()
        gplt.plot_style("tokyonight")
        ink = to_hex(styles.get_style("tokyonight").ink[:3])
        gplt.axhline(0.5)
        line = gplt.gcf().scene.layers[-1]
        assert to_hex(tuple(line.style.color[:3])) == ink

    def test_an_explicit_colour_always_wins(self):
        gplt.figure()
        gplt.plot_style("nord")
        gplt.scatter([0, 1], [0, 1], color="red")
        assert hexes(gplt.gcf().scene.layers[-1]) == "#ff0000"

    def test_cn_names_the_presets_own_cycle(self):
        gplt.figure()
        gplt.plot_style("catppuccin")
        gplt.plot([0, 1], [0, 1], color="C1")
        assert to_hex(tuple(gplt.gcf().scene.layers[-1].style.color[:3])) == "#fab387"
        gplt.figure()  # unstyled: matplotlib's own C1
        gplt.plot([0, 1], [0, 1], color="C1")
        assert to_hex(tuple(gplt.gcf().scene.layers[-1].style.color[:3])) == "#ff7f0e"


class TestTheExportsText:
    def test_tick_labels_are_written_in_the_presets_ink(self):
        plot = gplt.figure()
        gplt.plot_style("dense")  # a dark page with no look of its own: the ink is all it says
        gplt.plot([0, 1], [0, 1])
        seen = finished(plot)
        assert to_hex(seen["ylabels"][0]) == to_hex(styles.get_style("dense").ink[:3])

    def test_a_family_is_picked_by_the_weight_it_really_has(self, monkeypatch):
        """matplotlib registers one face per .ttc; a family that only has its bold
        registered must not set the body text, nor a regular-only one a bold title."""
        from matplotlib import font_manager

        fake = [
            font_manager.FontEntry(fname="a.ttc", name="OnlyBold", style="normal", weight=700),
            font_manager.FontEntry(fname="b.ttc", name="OnlyRegular", style="normal", weight=400),
            font_manager.FontEntry(fname="c.ttf", name="Both", style="normal", weight=400),
            font_manager.FontEntry(fname="d.ttf", name="Both", style="normal", weight="bold"),
        ]
        monkeypatch.setattr(font_manager.fontManager, "ttflist", fake)
        names = ("OnlyBold", "OnlyRegular", "Missing", "Both")
        assert preview._installed_families(names) == ["OnlyRegular", "Both"]
        assert preview._installed_families(names, bold=True) == ["OnlyBold", "Both"]


class TestComputerModernText:
    """matplotlib's cmr10 is laid out like TeX's OT1: some ASCII characters draw as other
    glyphs, and it has no accents. The 3b1b export writes those as one-symbol formulas."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("plain words", "plain words"),
            ("café au lait", "café au lait"),  # plain text: matplotlib's own fallback
            ("x > 0 and |x| < 1", "x $>$ 0 and $|$x$|$ $<$ 1"),
            ("a_b {c}", r"a$\_$b $\{$c$\}$"),
            ("café $x^2$", r"caf$\mathrm{é}$ $x^2$"),  # beside a formula: no fallback
            (r"cost \$5 and $y$", r"cost \$5 and $y$"),
            ("$5 > 3", r"\$5 $>$ 3"),  # a lone dollar becomes literal once there are formulas
        ],
    )
    def test_cm_safe(self, text, expected):
        assert _cm_safe(text) == expected

    def test_a_3b1b_export_draws_every_glyph(self):
        plot = gplt.figure(figsize=(6, 3.5))
        gplt.plot_style("3b1b")
        gplt.plot([0, 1], [0, 1])
        gplt.title("The slope of $x^2$ — café")
        gplt.xlabel("x > 0")
        gplt.text(0.1, 0.5, "naïve |y| $\\frac{a}{b}$ – done")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            render_preview_bytes(plot, 1.0)
        assert not [w for w in caught if "glyph" in str(w.message)]

    def test_the_rewrite_does_not_leak_out_of_the_export(self):
        from matplotlib.text import Text

        before = Text._preprocess_math
        render_preview_bytes(chart("3b1b"), 1.0)
        assert Text._preprocess_math is before


# ======================================================================================
# a catalogue a model can choose from
# ======================================================================================


class TestTheCatalogue:
    def test_every_preset_is_in_exactly_one_group(self):
        grouped = [key for _group, keys in styles.STYLE_GROUPS for key in keys]
        assert sorted(grouped) == sorted(styles.STYLE_KEYS)
        assert len(grouped) == len(set(grouped))

    def test_every_alias_names_a_preset(self):
        for alias, key in styles.STYLE_ALIASES.items():
            assert key in styles.STYLE_KEYS, alias
            assert alias == "".join(ch for ch in alias if ch.isalnum()), "aliases are folded"

    @pytest.mark.parametrize(
        "name, key",
        [
            ("economist", "economist"),
            ("The Economist", "economist"),
            ("the-economist", "economist"),
            ("538", "fivethirtyeight"),
            ("Five Thirty Eight", "fivethirtyeight"),
            ("Tokyo Night", "tokyonight"),
            ("3Blue1Brown", "3b1b"),
            ("manim", "3b1b"),
            ("ggplot2", "ggplot"),
            ("Financial Times", "ft"),
            ("Catppuccin Mocha", "catppuccin"),
            ("JOURNAL", "journal"),
        ],
    )
    def test_a_name_resolves_to_its_key(self, name, key):
        assert styles.resolve_style_key(name) == key

    @pytest.mark.parametrize("name", ["", "   ", "crayons", "the", "economistt"])
    def test_a_near_miss_resolves_to_nothing(self, name):
        assert styles.resolve_style_key(name) is None

    def test_plot_style_takes_a_name(self):
        gplt.figure()
        assert gplt.plot_style("The Economist") == "economist"

    def test_each_look_says_how_it_exports(self):
        assert "y numbers on the right" in styles.export_summary(styles.get_style("economist"))
        assert "LaTeX" in styles.export_summary(styles.get_style("3b1b"))
        assert styles.export_summary(styles.get_style("clean")) == ""

    def test_the_guide_lists_every_preset_by_use(self):
        from glplot import llm

        text = llm.guide("axes-styling")
        for key in styles.STYLE_KEYS:
            assert f"`{key}`" in text, key
        assert "**newsroom charts**" in text and "{{styles}}" not in text


class TestOverMcp:
    @pytest.fixture
    def box(self):
        from glplot.mcp.toolbox import Toolbox

        toolbox = Toolbox()
        yield toolbox
        toolbox.close_all()

    def test_a_figure_takes_a_style_by_name_and_says_what_it_got(self, box):
        result = box.create_figure(style="The Economist")
        assert result.data["style"] == "economist"
        assert "Saved images and videos get" in result.message
        assert box._figures[result.data["figure_id"]].spec["style"] == "economist"

    def test_set_style_records_the_key_for_the_script(self, box):
        figure = box.create_figure().data["figure_id"]
        box.add_plot(figure, "line", x=[0, 1, 2], y=[0, 1, 0])
        result = box.set_style(figure, "538")
        assert result.data == {"style": "fivethirtyeight", "name": "FiveThirtyEight"}
        script = box.export_script(figure).message
        assert "plt.plot_style('fivethirtyeight')" in script

    def test_an_unknown_style_suggests_and_lists_them_by_use(self, box):
        from glplot.mcp.state import McpError

        with pytest.raises(McpError, match="newsroom charts: economist") as caught:
            box.create_figure(style="economst")
        assert "economist" in str(caught.value).split("Available")[0], "the suggestion"

    def test_the_styles_resource_says_what_each_is_for(self, box):
        import json

        from glplot.mcp import catalog

        rows = {row["key"]: row for row in json.loads(catalog._styles(box, {}))}
        assert set(rows) == set(styles.STYLE_KEYS)
        assert rows["economist"]["use"] == "newsroom charts"
        assert rows["nord"]["page"] == "dark" and rows["ft"]["page"] == "light"
        assert "538" in rows["fivethirtyeight"]["aliases"]
        assert "title flush left" in rows["economist"]["export"]


# ======================================================================================
# the older presets' pictures, and colormaps
# ======================================================================================


class TestTheOlderLooks:
    def test_a_hand_drawn_look_wobbles_every_stroke_of_the_export(self):
        seen = finished(chart("chalk"))
        ax = seen["ax"]
        assert ax.spines["bottom"].get_sketch_params() is not None
        assert ax.lines[0].get_sketch_params() is not None
        assert matplotlib.rcParams["path.sketch"] is None, "and not outside it"

    def test_a_drawn_figure_is_the_same_every_time(self):
        first = render_preview_bytes(chart("hand"), 1.0)
        gplt.close("all")
        assert render_preview_bytes(chart("hand"), 1.0) == first, "an animation must not boil"

    def test_an_old_plate_has_its_ticks_inside(self):
        ax = finished(chart("vintage"))["ax"]
        tick = ax.xaxis.get_major_ticks()[0]
        assert tick._tickdir == "in"

    def test_a_talk_has_big_numbers_and_a_bold_title(self):
        seen = finished(chart("slide"))
        size = seen["ax"].get_yticklabels()[0].get_fontsize()
        assert size == pytest.approx(28.0) and seen["title_weight"] == "bold"

    @pytest.mark.parametrize("function", ["imshow", "contourf", "plot_surface"])
    def test_value_colours_take_the_presets_colormap(self, function):
        gplt.figure()
        gplt.plot_style("economist")
        z = np.outer(np.linspace(0, 1, 8), np.linspace(0, 1, 8))
        if function == "plot_surface":
            x, y = np.meshgrid(np.arange(8.0), np.arange(8.0))
            layer = gplt.plot_surface(x, y, z)
        elif function == "contourf":
            layer = gplt.contourf(z)
        else:
            layer = gplt.imshow(z)
        layers = gplt.gcf().scene.layers
        cmaps = {lay.metadata.get("cmap") for lay in layers if lay.metadata.get("cmap")}
        assert cmaps == {"Blues"}, (function, cmaps, layer)

    def test_an_explicit_colormap_still_wins(self):
        gplt.figure()
        gplt.plot_style("economist")
        gplt.imshow(np.eye(4), cmap="magma")
        assert gplt.gcf().scene.layers[-1].metadata["cmap"] == "magma"


# ======================================================================================
# legend placement (it was "reserved" and dropped)
# ======================================================================================


class TestLegendKeywords:
    def test_loc_places_the_exported_legend_and_the_live_one(self):
        plot = chart("economist")
        gplt.legend(loc="upper left")
        ax = finished(plot)["ax"]
        assert ax.get_legend()._loc == 2  # matplotlib's code for upper left
        assert plot.options.legend_loc == "upper left"

    def test_a_place_the_live_overlay_cannot_draw_falls_back_to_best_live_only(self):
        plot = chart("clean")
        gplt.legend(loc="center left")
        assert plot.options.legend_loc == "best"
        assert finished(plot)["ax"].get_legend()._loc == 6

    def test_numeric_codes_and_matplotlib_keywords(self):
        plot = chart("clean")
        gplt.legend(loc=4, ncol=2, title="series", frameon=False)
        legend = finished(plot)["ax"].get_legend()
        assert legend._loc == 4 and legend._ncols == 2
        assert legend.get_title().get_text() == "series"
        assert not legend.get_frame().get_visible()

    def test_an_explicit_frame_beats_the_looks_frameless_legend(self):
        plot = chart("economist")
        gplt.legend(frameon=True)
        assert finished(plot)["ax"].get_legend().get_frame().get_visible()

    def test_an_unknown_keyword_warns(self):
        gplt.figure()
        gplt.plot([0, 1], [0, 1], label="a")
        with pytest.warns(gplt.MatplotlibCompatWarning, match="nonsense"):
            gplt.legend(nonsense=3)

    def test_over_mcp(self):
        from glplot.mcp.state import McpError
        from glplot.mcp.toolbox import Toolbox

        box = Toolbox()
        try:
            figure = box.create_figure(style="ft").data["figure_id"]
            box.add_plot(figure, "line", x=[0, 1, 2], y=[0, 1, 0], label="a")
            box.set_axes(figure, legend_loc="lower right")
            assert "plt.legend(loc='lower right')" in box.export_script(figure).message
            with pytest.raises(McpError, match="Did you mean 'upper left'"):
                box.set_axes(figure, legend_loc="upper-left")
            with pytest.raises(McpError, match="legend=False"):
                box.set_axes(figure, legend=False, legend_loc="best")
        finally:
            box.close_all()


# ======================================================================================
# the review sees a colour the page swallows
# ======================================================================================


class TestInvisibleOnThePage:
    @pytest.fixture
    def box(self):
        from glplot.mcp.toolbox import Toolbox

        toolbox = Toolbox()
        yield toolbox
        toolbox.close_all()

    def codes(self, box, figure):
        return [i["code"] for i in box.render_figure(figure).data.get("issues", [])]

    def test_a_named_colour_that_matches_a_dark_page(self, box):
        figure = box.create_figure(style="dracula").data["figure_id"]
        box.add_plot(figure, "line", x=[0, 1, 2], y=[0, 1, 0], label="lost", color="#282A36")
        issues = box.render_figure(figure).data["issues"]
        lost = [i for i in issues if i["code"] == "invisible_on_page"]
        assert lost and "drop color=" in lost[0]["fix"] and "'lost'" in lost[0]["message"]

    def test_black_text_on_a_dark_style(self, box):
        figure = box.create_figure(style="tokyonight").data["figure_id"]
        box.add_plot(figure, "line", x=[0, 1, 2], y=[0, 1, 0], label="fine")
        box.add_annotation(figure, "text", x=1, y=0.5, text="a black note", color="black")
        assert "invisible_on_page" in self.codes(box, figure)

    @pytest.mark.parametrize("style", ["3b1b", "economist", "hand", "catppuccin", "ggplot"])
    def test_a_presets_own_colours_never_trip_it(self, box, style):
        figure = box.create_figure(style=style).data["figure_id"]
        for k in range(7):  # the whole palette
            box.add_plot(figure, "line", x=[0, 1], y=[k, k + 1], label=f"s{k}")
        box.add_plot(figure, "bar", x=[0, 1], y=[1, 2], label="bars")
        box.add_annotation(figure, "text", x=0.5, y=3, text="a note")
        assert "invisible_on_page" not in self.codes(box, figure)


class TestPromptsAndInstructions:
    def test_restyle_like_sends_the_look_to_set_style(self):
        from glplot.mcp import catalog

        text = catalog.find_prompt("restyle_like").build(figure_id="fig1", look="The Economist")
        assert "set_style(figure_id='fig1', preset='The Economist')" in text
        assert "legend_loc" in text and "invisible_on_page" in text

    def test_publication_figure_picks_the_look_by_venue(self):
        from glplot.mcp import catalog

        text = catalog.find_prompt("publication_figure").build(figure_id="fig1", venue="a blog")
        for key in ("journal", "slide", "economist", "material", "3b1b"):
            assert f"'{key}'" in text

    def test_the_server_instructions_say_a_look_is_one_argument(self):
        from glplot.mcp.guide import build_instructions

        text = build_instructions()
        assert "37 presets" in text and "'The Economist'" in text
        assert f"{len(styles.STYLE_KEYS)} presets" in text, "the count is the registry's"


class TestExportCost:
    def test_a_big_figure_keeps_the_hand_drawn_frame_but_not_on_its_data(self):
        plot = gplt.figure(figsize=(6, 4))
        gplt.plot_style("chalk")
        x = np.linspace(0, 1, preview.SKETCH_DATA_BUDGET + 10)
        gplt.plot(x, x)
        ax = finished(plot)["ax"]
        assert ax.spines["bottom"].get_sketch_params() is not None
        assert ax.lines[0].get_sketch_params() is None

    def test_a_long_title_is_fitted_without_redrawing_the_figure_per_step(self, monkeypatch):
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        draws = []
        real = FigureCanvasAgg.draw
        monkeypatch.setattr(FigureCanvasAgg, "draw", lambda self: draws.append(1) or real(self))
        plot = gplt.figure(figsize=(4, 3))
        gplt.plot([0, 1], [0, 1])
        gplt.title("A title far too long to fit across a figure this narrow, at any size " * 2)
        render_preview_bytes(plot, 1.0)
        assert len(draws) <= 3, "one to place, one after fitting, one to save"


class TestGradientPages:
    def test_a_gradient_preset_exports_its_gradient_behind_the_figure(self):
        from matplotlib.image import BboxImage

        seen = finished(chart("synthwave"))
        fig, ax = seen["fig"], seen["ax"]
        images = [a for a in fig.artists if isinstance(a, BboxImage)]
        assert len(images) == 1 and not images[0].get_in_layout()
        top, bottom = styles.get_style("synthwave").gradient
        pixels = images[0].get_array()
        assert np.allclose(pixels[0, 0], top, atol=1e-6)
        assert np.allclose(pixels[-1, 0], bottom, atol=1e-6)
        assert ax.get_facecolor()[3] == 0.0, "seen through the plotting area"
        assert len(fig.axes) == 1, "no extra axes for the review or the MCP to trip over"

    def test_a_flat_page_has_none(self):
        from matplotlib.image import BboxImage

        fig = finished(chart("economist"))["fig"]
        assert not [a for a in fig.artists if isinstance(a, BboxImage)]


class TestTextInABox:
    def test_text_in_a_light_box_on_a_dark_style_is_dark(self):
        gplt.figure()
        gplt.plot_style("neon")
        gplt.text(0, 0, "note", bbox=dict(facecolor="lightyellow"))
        assert tuple(gplt.gcf().scene.layers[-1].style.color[:3]) == (0.0, 0.0, 0.0)

    def test_the_ink_stays_when_it_reads_on_the_box(self):
        gplt.figure()
        gplt.plot_style("neon")
        gplt.text(0, 0, "note", bbox=dict(facecolor="black"))
        ink = tuple(styles.get_style("neon").ink[:3])
        assert tuple(gplt.gcf().scene.layers[-1].style.color[:3]) == pytest.approx(ink)

    def test_an_explicit_colour_still_wins(self):
        gplt.figure()
        gplt.plot_style("neon")
        gplt.text(0, 0, "note", color="red", bbox=dict(facecolor="lightyellow"))
        assert tuple(gplt.gcf().scene.layers[-1].style.color[:3]) == (1.0, 0.0, 0.0)


# ======================================================================================
# log axes the gallery found broken
# ======================================================================================


def exported_ylim(plot):
    seen = {}
    render_preview_bytes(plot, 1.0, inspect=lambda fig: seen.update(y=fig.axes[0].get_ylim()))
    return seen["y"]


class TestLogAxes:
    def test_a_log_bar_chart_shows_its_smallest_bar(self):
        plot = gplt.figure()
        gplt.bar([1, 2, 3], [2e-2, 1e-3, 1e-5])
        gplt.yscale("log")
        low, high = exported_ylim(plot)
        assert low < 1e-5 < 2e-2 < high, "the axis used to start a decade under the top bar"
        plot.autoscale()  # the live camera too
        assert 10 ** plot.get_ylim()[0] < 1e-5

    def test_a_scale_does_not_freeze_the_view(self):
        plot = gplt.figure()
        gplt.plot([1, 2, 3], [1e-2, 1e-1, 1.0])
        gplt.yscale("log")
        gplt.plot([1, 3], [1e-6, 1e-6])  # drawn after the scale, still in view
        assert exported_ylim(plot)[0] < 1e-6

    def test_limits_set_before_a_scale_survive_it(self):
        plot = gplt.figure()
        gplt.plot([1, 2, 3], [1, 10, 100])
        gplt.ylim(5, 50)
        gplt.yscale("log")
        assert exported_ylim(plot) == pytest.approx((5, 50))

    def test_a_handwriting_face_without_a_minus_sign_still_writes_one(self):
        from glplot.utils.preview import _minus_safe

        assert _minus_safe(r"$\mathdefault{10^{-3}}$") == r"$\mathdefault{10^{\mathrm{-}3}}$"
        assert _minus_safe("well-known") == "well-known", "text outside maths keeps its hyphen"
        plot = gplt.figure(figsize=(5, 3))
        gplt.plot_style("chalk")
        gplt.plot([0, 1, 2], [1e-3, 1e-2, 1e-1])
        gplt.yscale("log")
        import logging

        records = []
        handler = logging.Handler()
        handler.emit = records.append
        logging.getLogger("matplotlib.mathtext").addHandler(handler)
        try:
            render_preview_bytes(plot, 1.0)
        finally:
            logging.getLogger("matplotlib.mathtext").removeHandler(handler)
        assert not [r for r in records if "glyph" in r.getMessage()]
