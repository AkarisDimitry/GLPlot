"""A preset's own colormap: the ramp for anything coloured by value.

A palette answers "which line is which"; the colormap answers "how big is this one". Before
presets owned one, a synthwave page got a viridis heat map. These pin the pieces: every preset
names a colormap matplotlib can resolve, GLPlot's own six are well-formed ramps, applying a
preset can re-map the layers that already carry values (leaving the rest, and the data, alone),
undo puts the old ramp back exactly, and ``plot_style`` leaves the preset's ramp as the figure's
default without overriding an explicit choice.
"""

from __future__ import annotations

import dataclasses

import matplotlib
import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot import colormaps
from glplot.engine import GPULinePlot
from glplot.gui import layerops, layerops3d, styles


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


@pytest.fixture
def plot():
    return GPULinePlot(width=480, height=300, title="GLPlot")


def scene(plot):
    """One layer of each kind a colormap can reach, and two it cannot."""
    x = np.linspace(0.0, 1.0, 40)
    scatter = layerops.add_xy_layer(
        plot, x, x**2, kind="scatter", label="values", c=x, cmap="viridis"
    )
    image = layerops.add_image_layer(
        plot, np.add.outer(x, x), (0, 1, 0, 1), label="image", cmap="viridis"
    )
    plain = layerops.add_xy_layer(plot, x, x, kind="scatter", label="plain")
    line = layerops.add_xy_layer(plot, x, x, kind="line", label="line")
    return scatter, image, plain, line


def cmap_of(layer):
    return layer.metadata.get("cmap")


def ramp_colours(layer, name):
    """What the layer's colours would be if its scalars went through colormap ``name``."""
    return layerops.colormap_colors(np.asarray(layer.metadata["cvalues"], dtype=float), name)


def drawn_through(layer, name):
    return np.allclose(np.asarray(layer.colors)[:, :3], ramp_colours(layer, name)[:, :3], atol=1e-3)


class TestEveryPresetOwnsAColormap:
    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_it_resolves_in_matplotlib(self, key):
        name = colormaps.resolve(styles.get_style(key).cmap)
        assert name in matplotlib.colormaps

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_it_is_a_string(self, key):
        assert isinstance(styles.get_style(key).cmap, str) and styles.get_style(key).cmap

    def test_the_new_looks_each_have_a_ramp_made_for_them(self):
        for key, ramp in (
            ("aurora", "aurora"),
            ("synthwave", "synthwave"),
            ("sunset", "sunset"),
            ("ocean", "lagoon"),
            ("vintage", "parchment"),
            ("candy", "candy"),
        ):
            assert styles.get_style(key).cmap == ramp

    def test_a_ramp_for_a_dark_page_starts_dark_and_a_ramp_for_a_light_page_starts_light(self):
        """So 'low' dissolves into the page it is drawn on, whichever kind of page that is."""
        for key in ("aurora", "synthwave", "sunset", "ocean"):
            ramp = matplotlib.colormaps[colormaps.resolve(styles.get_style(key).cmap)]
            assert luminance(ramp(0.0)) < 0.2 < 0.8 < luminance(ramp(1.0)), key
        for key in ("vintage", "candy"):
            ramp = matplotlib.colormaps[colormaps.resolve(styles.get_style(key).cmap)]
            assert luminance(ramp(0.0)) > 0.8 > 0.2 > luminance(ramp(1.0)), key


def luminance(rgba):
    return 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]


class TestTheCustomRamps:
    @pytest.mark.parametrize("name", sorted(colormaps.CUSTOM_COLORMAPS))
    def test_lightness_only_ever_moves_one_way(self, name):
        """The property that lets a ramp answer 'which is bigger': no rainbow, no reversal."""
        ramp = matplotlib.colormaps[colormaps.resolve(name)]
        lum = np.array([luminance(ramp(v)) for v in np.linspace(0, 1, 128)])
        steps = np.diff(lum)
        assert (steps >= -1e-9).all() or (steps <= 1e-9).all()

    @pytest.mark.parametrize("name", sorted(colormaps.CUSTOM_COLORMAPS))
    def test_it_spans_a_real_range_of_lightness(self, name):
        ramp = matplotlib.colormaps[colormaps.resolve(name)]
        assert abs(luminance(ramp(0.0)) - luminance(ramp(1.0))) > 0.6

    def test_registration_is_idempotent(self):
        before = set(matplotlib.colormaps)
        colormaps.ensure_registered()
        colormaps._registered = False
        colormaps.ensure_registered()
        assert set(matplotlib.colormaps) == before

    def test_a_colormap_someone_else_registered_is_never_overwritten(self):
        probe = "glplot_test_probe"
        try:
            colormaps.CUSTOM_COLORMAPS[probe] = colormaps.CUSTOM_COLORMAPS["aurora"]
            foreign = matplotlib.colors.ListedColormap(["red", "blue"], name=probe)
            matplotlib.colormaps.register(foreign, name=probe)
            colormaps._registered = False
            colormaps.ensure_registered()
            assert matplotlib.colormaps[probe].N == 2  # the foreign two-colour map is still there
            assert colormaps.resolve(probe) == f"glplot_{probe}"
            assert f"glplot_{probe}" in matplotlib.colormaps
        finally:
            del colormaps.CUSTOM_COLORMAPS[probe]
            for name in (probe, f"glplot_{probe}"):
                if name in matplotlib.colormaps:
                    matplotlib.colormaps.unregister(name)
            colormaps._registered = False
            colormaps.ensure_registered()

    def test_names_that_are_not_ours_pass_through(self):
        assert colormaps.resolve("viridis") == "viridis"
        assert colormaps.resolve("no-such-map") == "no-such-map"


class TestApplyingAStyleRemapsValueLayers:
    def test_off_by_default_so_an_explicit_choice_survives(self, plot):
        scatter, image, *_ = scene(plot)
        styles.apply_style(plot, styles.get_style("synthwave"), layers=True)
        assert cmap_of(scatter) == "viridis" and cmap_of(image) == "viridis"

    def test_on_it_remaps_the_scatter_and_the_image(self, plot):
        scatter, image, *_ = scene(plot)
        styles.apply_style(plot, styles.get_style("synthwave"), layers=True, colormaps=True)
        assert cmap_of(scatter) == "synthwave" and cmap_of(image) == "synthwave"

    def test_the_data_behind_the_colours_is_untouched(self, plot):
        scatter, image, *_ = scene(plot)
        values = np.array(scatter.metadata["cvalues"])
        matrix = np.array(image.metadata["matrix"])
        styles.apply_style(plot, styles.get_style("sunset"), layers=True, colormaps=True)
        assert np.array_equal(scatter.metadata["cvalues"], values)
        assert np.array_equal(image.metadata["matrix"], matrix)

    def test_the_scatters_colours_really_change(self, plot):
        scatter, *_ = scene(plot)
        before = np.array(scatter.colors)
        styles.apply_style(plot, styles.get_style("candy"), layers=True, colormaps=True)
        assert not np.allclose(before[:, :3], scatter.colors[:, :3])

    def test_layers_with_no_value_colours_are_left_alone(self, plot):
        _, _, plain, line = scene(plot)
        styles.apply_style(plot, styles.get_style("ocean"), layers=False, colormaps=True)
        assert cmap_of(plain) is None and cmap_of(line) is None

    def test_it_works_with_layers_false_too(self, plot):
        """Re-mapping is its own switch, not a side effect of restyling colours."""
        scatter, *_ = scene(plot)
        styles.apply_style(plot, styles.get_style("ocean"), layers=False, colormaps=True)
        assert cmap_of(scatter) == "lagoon"

    def test_a_3d_surface_is_remapped_too(self, plot):
        plot.set_ndim(3)
        n = 8
        gx, gy = np.meshgrid(np.arange(float(n)), np.arange(float(n)))
        surface = layerops3d.add_xyz_layer(
            plot,
            gx.ravel(),
            gy.ravel(),
            (gx + gy).ravel(),
            kind="surface3d",
            label="s",
            cmap="viridis",
        )
        styles.apply_style(plot, styles.get_style("aurora"), layers=True, colormaps=True)
        assert cmap_of(surface) == "aurora"

    def test_it_records_the_ramp_on_the_figure_for_whatever_is_drawn_next(self, plot):
        styles.apply_style(plot, styles.get_style("sunset"), colormaps=True)
        assert plot._style_cmap == "sunset"

    def test_without_the_switch_the_figure_default_is_untouched(self, plot):
        styles.apply_style(plot, styles.get_style("sunset"))
        assert getattr(plot, "_style_cmap", None) is None

    def test_apply_colormaps_reports_how_many_it_changed(self, plot):
        scene(plot)
        assert styles.apply_colormaps(plot, styles.get_style("sunset")) == 2
        assert styles.apply_colormaps(plot, styles.get_style("sunset")) == 0  # already there


class TestUndo:
    def test_undo_puts_the_old_ramps_back_exactly(self, plot):
        scatter, image, *_ = scene(plot)
        colours = np.array(scatter.colors)
        command = styles.style_command(plot, styles.get_style("synthwave"), colormaps=True)
        command.do()
        assert cmap_of(scatter) == "synthwave"
        command.undo()
        assert cmap_of(scatter) == "viridis" and cmap_of(image) == "viridis"
        assert np.allclose(scatter.colors, colours)

    def test_a_pinned_range_survives_the_round_trip_and_autoscale_stays_autoscale(self, plot):
        scatter, image, *_ = scene(plot)
        layerops.set_layer_colormap(plot, image, vmin=0.25)
        command = styles.style_command(plot, styles.get_style("sunset"), colormaps=True)
        command.do()
        command.undo()
        assert image.metadata["vmin"] == 0.25 and image.metadata["vmax"] is None

    def test_a_command_that_did_not_ask_for_colormaps_records_none(self, plot):
        state = styles.capture_state(plot)
        assert state["colormaps"] == []

    def test_undo_skips_a_layer_that_was_deleted_in_between(self, plot):
        scatter, *_ = scene(plot)
        command = styles.style_command(plot, styles.get_style("sunset"), colormaps=True)
        command.do()
        layerops.remove_layer(plot, None, scatter)
        command.undo()  # must not raise


class TestPlotStyleApi:
    def test_plot_style_remaps_existing_layers_and_sets_the_figure_default(self):
        fig = gplt.figure()
        x = np.linspace(0, 1, 30)
        gplt.scatter(x, x, c=x)
        gplt.plot_style("sunset")
        layer = fig.scene.layers[-1]
        assert layer.metadata["cmap"] == "sunset"
        assert fig._style_cmap == "sunset"

    def test_the_next_plot_uses_the_styles_colormap_without_naming_one(self):
        fig = gplt.figure()
        gplt.plot_style("synthwave")
        x = np.linspace(0, 1, 30)
        gplt.scatter(x, x, c=x)
        # pyplot keeps no record of a *defaulted* colormap, so check the colours themselves.
        layer = fig.scene.layers[-1]
        assert drawn_through(layer, "synthwave") and not drawn_through(layer, "viridis")

    def test_an_explicit_cmap_still_wins(self):
        fig = gplt.figure()
        gplt.plot_style("synthwave")
        x = np.linspace(0, 1, 30)
        gplt.scatter(x, x, c=x, cmap="magma")
        assert fig.scene.layers[-1].metadata["cmap"] == "magma"

    def test_set_cmap_beats_the_styles_colormap(self):
        fig = gplt.figure()
        gplt.plot_style("synthwave")
        gplt.set_cmap("plasma")
        x = np.linspace(0, 1, 30)
        gplt.scatter(x, x, c=x)
        layer = fig.scene.layers[-1]
        assert drawn_through(layer, "plasma") and not drawn_through(layer, "synthwave")

    def test_colormap_false_leaves_every_colormap_alone(self):
        fig = gplt.figure()
        x = np.linspace(0, 1, 30)
        gplt.scatter(x, x, c=x)
        before = fig.scene.layers[-1].metadata["cmap"]
        gplt.plot_style("sunset", colormap=False)
        assert fig.scene.layers[-1].metadata["cmap"] == before
        assert getattr(fig, "_style_cmap", None) is None

    def test_the_style_colormap_belongs_to_its_figure(self):
        first = gplt.figure()
        gplt.plot_style("synthwave")
        second = gplt.figure()
        x = np.linspace(0, 1, 30)
        gplt.scatter(x, x, c=x)
        assert not drawn_through(second.scene.layers[-1], "synthwave")
        assert first._style_cmap == "synthwave"


class TestPresetCatalogue:
    def test_the_gallery_grew_by_the_looks_that_were_asked_for(self):
        assert styles.STYLE_KEYS[-6:] == (
            "aurora",
            "synthwave",
            "sunset",
            "ocean",
            "vintage",
            "candy",
        )
        assert len(styles.STYLES) == 25

    def test_dataclass_default_cmap_is_a_safe_fallback(self):
        assert dataclasses.fields(styles.PlotStyle)[0].name == "key"
        assert styles.PlotStyle("k", "n", "d").cmap == "viridis"
