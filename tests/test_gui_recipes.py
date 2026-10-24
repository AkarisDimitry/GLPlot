"""Recipes: the gallery examples as data, and what happens when one is pointed at a table.

Pure logic, no window and no imgui: every recipe is built on a windowless ``GPULinePlot`` the
way the thumbnail generator and the live preview build it, so what is checked here is the same
code the Gallery panel runs. The things worth pinning are the ones that fail *quietly*: a recipe
that draws but in the wrong palette, an undo that leaves the page restyled, a style that lands
on the 3-D axis box instead of the data, a column mapping that silently drops rows.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from glplot import colormaps
from glplot.core.camera3d import SYSTEM_3D_ARTISTS
from glplot.engine import GPULinePlot
from glplot.gui import layerops, layerops3d, recipes, styles

REPO = Path(__file__).resolve().parent.parent
ALL = recipes.RECIPES


def data_layers(plot):
    return [
        layer
        for layer in plot.scene.layers
        if layer.metadata.get("artist") not in SYSTEM_3D_ARTISTS
    ]


def sample(recipe):
    return recipes.sample_data(recipe)


def colour_of(layer):
    """The layer's colour: a polyline keeps it in its style, a scatter in its colour array."""
    if layer.layer_type == "polyline":
        return tuple(float(c) for c in layer.style.color)
    return tuple(np.asarray(layer.colors).reshape(-1, 4)[0])


def labels(recipe):
    return recipes.sample_labels(recipe)


# ----------------------------------------------------------------------------------
# The catalogue is complete data
# ----------------------------------------------------------------------------------


class TestCatalogue:
    def test_there_are_enough_recipes_and_keys_are_unique(self):
        assert len(ALL) >= 20
        assert len(set(recipes.RECIPE_KEYS)) == len(ALL)

    @pytest.mark.parametrize("recipe", ALL, ids=lambda r: r.key)
    def test_identity_fits_a_card(self, recipe):
        assert recipe.category in recipes.CATEGORIES
        assert 0 < len(recipe.name) <= 32
        assert 0 < len(recipe.description) <= 100

    @pytest.mark.parametrize("recipe", ALL, ids=lambda r: r.key)
    def test_everything_drawn_in_the_ui_is_ascii(self, recipe):
        """The workstation's font has no glyph for an ellipsis or a middle dot: it prints '?'."""
        shown = [recipe.name, recipe.description, recipe.title, *recipe.tags]
        for slot in recipe.slots:
            shown += [slot.label, slot.hint]
        for text in shown:
            assert text.isascii(), (recipe.key, text)

    @pytest.mark.parametrize("recipe", ALL, ids=lambda r: r.key)
    def test_kind_style_and_colormap_exist(self, recipe):
        known = set(layerops.KIND_KEYS) | set(layerops3d.KIND3D_KEYS) | {"image", "field"}
        assert recipe.kind in known
        assert recipe.style in styles.STYLE_KEYS
        if recipe.cmap is not None:
            import matplotlib

            assert colormaps.resolve(recipe.cmap) in matplotlib.colormaps

    @pytest.mark.parametrize("recipe", ALL, ids=lambda r: r.key)
    def test_options_are_ones_the_kind_accepts(self, recipe):
        if recipe.is_3d:
            layerops3d.resolve_kind3d_options(recipe.kind, dict(recipe.options))
        elif not recipe.is_special:
            layerops._resolve_options(recipe.kind, dict(recipe.options))

    @pytest.mark.parametrize("recipe", ALL, ids=lambda r: r.key)
    def test_layer_style_names_real_fields(self, recipe):
        from glplot.core.layers import LayerStyle

        style = LayerStyle()
        for name in recipe.layer_style:
            assert hasattr(style, name), (recipe.key, name)

    @pytest.mark.parametrize("recipe", ALL, ids=lambda r: r.key)
    def test_the_example_it_came_from_exists(self, recipe):
        assert recipe.example
        assert (REPO / "examples" / "gallery" / recipe.example).is_file(), recipe.example

    @pytest.mark.parametrize("recipe", ALL, ids=lambda r: r.key)
    def test_camera_only_on_3d_recipes(self, recipe):
        assert (recipe.camera is not None) == recipe.is_3d

    @pytest.mark.parametrize("recipe", ALL, ids=lambda r: r.key)
    def test_axes_name_a_slot_or_a_literal(self, recipe):
        slot_keys = {slot.key for slot in recipe.slots}
        for source in recipe.axes:
            assert source.startswith("=") or source in slot_keys, (recipe.key, source)

    def test_every_category_is_used_and_every_dimension_is_covered(self):
        assert {r.category for r in ALL} == set(recipes.CATEGORIES)
        assert {r.dimension for r in ALL} == {"2D", "3D"}

    def test_the_catalogue_shows_off_more_than_one_look(self):
        """A gallery where every card is Clean would not be showing the presets off."""
        assert len({r.style for r in ALL}) >= 10

    def test_get_recipe_never_guesses(self):
        assert recipes.get_recipe("surface3d").kind == "surface3d"
        with pytest.raises(KeyError):
            recipes.get_recipe("surfce3d")


# ----------------------------------------------------------------------------------
# Sample tables
# ----------------------------------------------------------------------------------


class TestSamples:
    @pytest.mark.parametrize("recipe", ALL, ids=lambda r: r.key)
    def test_sample_is_deterministic_finite_and_rectangular(self, recipe):
        first = recipes.sample_table(recipe)
        second = recipes.sample_table(recipe)
        assert list(first) == list(second)
        lengths = {len(values) for values in first.values()}
        assert len(lengths) == 1 and lengths.pop() > 1
        for name in first:
            assert first[name].dtype == np.float64
            assert np.array_equal(first[name], second[name]), name
            assert np.isfinite(first[name]).all(), name

    @pytest.mark.parametrize("recipe", ALL, ids=lambda r: r.key)
    def test_every_slot_a_sample_fills_names_a_real_column(self, recipe):
        table = recipes.sample_table(recipe)
        for slot, column in recipes.sample_assignment(recipe).items():
            assert column in table, (recipe.key, slot, column)
        # ...and every required, non-matrix slot is filled, or "Try with sample data" would fail.
        for slot in recipe.required_slots():
            if not slot.multi:
                assert slot.key in recipes.sample_assignment(recipe), (recipe.key, slot.key)

    @pytest.mark.parametrize("recipe", ALL, ids=lambda r: r.key)
    def test_the_sample_is_drawable(self, recipe):
        assert recipes.validate_data(recipe, sample(recipe)) is None

    def test_sample_row_count_can_be_overridden(self):
        recipe = recipes.get_recipe("signal_trace")
        assert len(next(iter(recipes.sample_table(recipe, 50).values()))) == 50

    def test_a_recipe_without_a_sample_says_so(self):
        bare = dataclasses.replace(recipes.get_recipe("signal_trace"), sample=None)
        with pytest.raises(recipes.RecipeError):
            recipes.sample_table(bare)


# ----------------------------------------------------------------------------------
# Guessing which column goes where
# ----------------------------------------------------------------------------------


class TestAssignColumns:
    def test_a_name_that_matches_an_alias_wins_over_position(self):
        recipe = recipes.get_recipe("signal_trace")
        got = recipes.assign_columns(recipe, ["noise", "signal", "time"])
        assert got["x"] == "time" and got["y"] == "signal"

    def test_unmatched_columns_fill_required_slots_left_to_right(self):
        recipe = recipes.get_recipe("bar_chart")
        got = recipes.assign_columns(recipe, ["alpha", "beta", "gamma"])
        assert (got["x"], got["y"]) == ("alpha", "beta")

    def test_matching_is_case_insensitive_and_keeps_the_real_name(self):
        recipe = recipes.get_recipe("signal_trace")
        assert recipes.assign_columns(recipe, ["Time", "Signal"]) == {
            "x": "Time",
            "y": "Signal",
            "c": None,
        }

    def test_optional_slots_are_not_guessed_without_a_name_hint(self):
        recipe = recipes.get_recipe("signal_trace")
        assert recipes.assign_columns(recipe, ["a", "b", "extra"])["c"] is None

    def test_an_optional_slot_is_filled_by_its_alias(self):
        recipe = recipes.get_recipe("cloud3d")
        got = recipes.assign_columns(recipe, ["x", "y", "z", "radius"])
        assert got["c"] == "radius"

    def test_one_column_is_plotted_against_the_row_number(self):
        got = recipes.assign_columns(recipes.get_recipe("signal_trace"), ["only"])
        assert got["x"] == recipes.INDEX and got["y"] == "only"

    def test_a_column_is_never_assigned_twice(self):
        got = recipes.assign_columns(recipes.get_recipe("surface3d"), ["x", "y", "z"])
        values = [v for v in got.values() if v]
        assert len(values) == len(set(values))

    def test_the_matrix_slot_takes_every_column_so_it_is_left_unassigned(self):
        got = recipes.assign_columns(recipes.get_recipe("heatmap_matrix"), ["a", "b", "c"])
        assert got == {"matrix": None}

    def test_fewer_columns_than_slots_leaves_the_rest_unassigned(self):
        got = recipes.assign_columns(recipes.get_recipe("surface3d"), ["a", "b"])
        assert got["z"] is None

    def test_no_columns_at_all(self):
        got = recipes.assign_columns(recipes.get_recipe("bar_chart"), [])
        assert got == {"x": None, "y": None}


# ----------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------


class TestValidation:
    def test_a_missing_required_slot_names_it(self):
        message = recipes.validate_data(recipes.get_recipe("signal_trace"), {"x": np.arange(5.0)})
        assert message and "Y" in message

    def test_mismatched_lengths_are_reported_with_the_counts(self):
        data = {"x": np.arange(5.0), "y": np.arange(4.0)}
        message = recipes.validate_data(recipes.get_recipe("signal_trace"), data)
        assert message and "different lengths" in message and "5" in message and "4" in message

    def test_empty_columns(self):
        data = {"x": np.array([]), "y": np.array([])}
        assert "empty" in recipes.validate_data(recipes.get_recipe("signal_trace"), data)

    def test_rows_with_a_gap_are_dropped_from_every_column_together(self):
        data = {"x": np.arange(6.0), "y": np.array([1, np.nan, 3, 4, np.inf, 6.0])}
        prepared = recipes.prepare_data(recipes.get_recipe("signal_trace"), data)
        assert prepared["x"].tolist() == [0.0, 2.0, 3.0, 5.0]
        assert prepared["y"].tolist() == [1.0, 3.0, 4.0, 6.0]

    def test_a_gap_in_an_unused_optional_column_costs_no_rows(self):
        recipe = recipes.get_recipe("signal_trace")
        data = {"x": np.arange(4.0), "y": np.arange(4.0)}
        assert len(recipes.prepare_data(recipe, data)["x"]) == 4

    def test_all_rows_missing(self):
        data = {"x": np.full(3, np.nan), "y": np.arange(3.0)}
        assert "missing" in recipes.validate_data(recipes.get_recipe("signal_trace"), data)

    def test_a_density_recipe_needs_enough_rows_to_bin(self):
        data = {"x": np.arange(3.0), "y": np.arange(3.0)}
        assert "at least" in recipes.validate_data(recipes.get_recipe("hist2d_heat"), data)

    def test_a_surface_needs_a_full_grid(self):
        rng = np.random.default_rng(0)
        scattered = {k: rng.random(40) for k in "xyz"}
        message = recipes.validate_data(recipes.get_recipe("surface3d"), scattered)
        assert message and "grid" in message

    def test_a_surface_accepts_a_real_grid(self):
        gx, gy = np.meshgrid(np.arange(5.0), np.arange(6.0))
        data = {"x": gx.ravel(), "y": gy.ravel(), "z": (gx * gy).ravel()}
        assert recipes.validate_data(recipes.get_recipe("surface3d"), data) is None

    def test_a_field_needs_a_range_in_both_directions(self):
        data = {"x": np.zeros(10), "y": np.arange(10.0), "z": np.arange(10.0)}
        assert "range" in recipes.validate_data(recipes.get_recipe("scattered_field"), data)

    def test_a_matrix_must_be_two_dimensional_and_at_least_2x2(self):
        recipe = recipes.get_recipe("heatmap_matrix")
        assert recipes.validate_data(recipe, {"matrix": np.arange(5.0)})
        assert recipes.validate_data(recipe, {"matrix": np.ones((1, 9))})
        assert recipes.validate_data(recipe, {"matrix": np.ones((4, 4))}) is None

    def test_too_many_rows_is_refused(self, monkeypatch):
        monkeypatch.setattr(recipes, "MAX_RECIPE_ROWS", 10)
        data = {"x": np.arange(11.0), "y": np.arange(11.0)}
        assert "more than" in recipes.validate_data(recipes.get_recipe("signal_trace"), data)

    def test_validation_never_raises_on_odd_input(self):
        recipe = recipes.get_recipe("signal_trace")
        assert recipes.validate_data(recipe, {}) is not None
        assert recipes.validate_data(recipe, {"x": [1, 2, 3], "y": [4, 5, 6]}) is None


# ----------------------------------------------------------------------------------
# Applying a recipe
# ----------------------------------------------------------------------------------


@pytest.fixture
def plot():
    return GPULinePlot(width=480, height=300, title="GLPlot")


def build(plot, recipe, **kwargs):
    kwargs.setdefault("labels", labels(recipe))
    return recipes.apply_recipe(plot, recipe, sample(recipe), **kwargs)


class TestApply:
    @pytest.mark.parametrize("recipe", ALL, ids=lambda r: r.key)
    def test_every_recipe_draws_from_its_own_sample(self, plot, recipe):
        created = build(plot, recipe)
        assert created, recipe.key
        assert all(layer in plot.scene.layers for layer in created)

    @pytest.mark.parametrize("recipe", [r for r in ALL if not r.is_special], ids=lambda r: r.key)
    def test_the_layer_carries_its_kind_so_the_scene_panel_can_edit_it(self, plot, recipe):
        created = build(plot, recipe)
        tag = (
            layerops3d.layer_kind3d(created[0]) if recipe.is_3d else layerops.layer_kind(created[0])
        )
        assert tag == recipe.kind

    @pytest.mark.parametrize(
        "recipe", [r for r in ALL if r.kind in ("image", "field")], ids=lambda r: r.key
    )
    def test_field_recipes_make_a_real_image_layer(self, plot, recipe):
        (layer,) = build(plot, recipe)
        assert layer.metadata["artist"] == "imshow"
        assert layer.metadata["matrix"].ndim == 2

    @pytest.mark.parametrize("recipe", ALL, ids=lambda r: r.key)
    def test_the_recipes_look_is_applied_to_the_page(self, plot, recipe):
        build(plot, recipe)
        style = styles.get_style(recipe.style)
        assert plot._style_key == recipe.style
        assert tuple(plot.options.visual.background_color) == tuple(style.background)
        assert plot.options.light_bg_mode == style.light_bg_mode

    @pytest.mark.parametrize("recipe", [r for r in ALL if r.is_3d], ids=lambda r: r.key)
    def test_a_3d_recipe_switches_to_3d_and_points_the_camera(self, plot, recipe):
        build(plot, recipe)
        assert plot.active_panel.is_3d()
        assert plot.camera3d.elev == pytest.approx(recipe.camera[0])
        assert plot.camera3d.azim == pytest.approx(recipe.camera[1])

    def test_a_flat_recipe_flattens_the_axis_box_and_undo_restores_it(self, plot):
        recipe = recipes.get_recipe("nebula3d")
        assert recipe.box_aspect is not None
        before = plot.camera3d.box_aspect
        command = recipes.recipe_command(plot, None, recipe, sample(recipe))
        command.do()
        assert tuple(plot.camera3d.box_aspect) == pytest.approx(recipe.box_aspect)
        command.undo()
        assert plot.camera3d.box_aspect == before

    @pytest.mark.parametrize("recipe", [r for r in ALL if not r.is_3d], ids=lambda r: r.key)
    def test_a_2d_recipe_stays_2d(self, plot, recipe):
        build(plot, recipe)
        assert not plot.active_panel.is_3d()

    def test_the_first_layer_takes_the_first_palette_colour_and_the_next_the_next(self, plot):
        recipe = dataclasses.replace(recipes.get_recipe("signal_trace"), style="clean")
        palette = styles.get_style("clean").palette
        first = build(plot, recipe)[0]
        second = build(plot, recipe)[0]
        assert colour_of(first) == pytest.approx(palette[0])
        assert colour_of(second) == pytest.approx(palette[1])

    def test_a_colour_by_column_is_mapped_through_the_styles_own_colormap(self, plot):
        recipe = recipes.get_recipe("starfield")
        (layer,) = build(plot, recipe)
        assert layer.metadata["cmap"] == colormaps.resolve(styles.get_style(recipe.style).cmap)
        assert layer.metadata["cvalues"] is not None

    def test_an_explicit_recipe_colormap_beats_the_styles(self, plot):
        recipe = dataclasses.replace(recipes.get_recipe("starfield"), cmap="magma")
        (layer,) = build(plot, recipe)
        assert layer.metadata["cmap"] == "magma"

    def test_the_colormapped_density_kinds_get_the_ramp_through_their_options(self, plot):
        recipe = recipes.get_recipe("hist2d_heat")
        (layer,) = build(plot, recipe)
        assert layer.metadata["gui_kind_options"]["cmap"] == colormaps.resolve(
            styles.get_style(recipe.style).cmap
        )

    def test_a_per_layer_style_reaches_the_data_layer(self, plot):
        recipe = recipes.get_recipe("starfield")
        (layer,) = build(plot, recipe)
        assert layer.style.alpha == pytest.approx(recipe.layer_style["alpha"])

    def test_the_3d_axis_box_is_never_styled_as_data(self, plot):
        """The box is layers too; an outline or additive blend meant for the arrows must not
        fatten and fog it (it did, before the filter)."""
        recipe = recipes.get_recipe("quiver3d_flow")
        build(plot, recipe)
        chrome = [l for l in plot.scene.layers if l.metadata.get("artist") in SYSTEM_3D_ARTISTS]
        assert chrome
        assert all(not l.style.outline_enabled for l in chrome)
        (data,) = data_layers(plot)
        assert data.style.outline_enabled

    def test_additive_compositing_lands_on_the_cloud_and_not_the_box(self, plot):
        build(plot, recipes.get_recipe("nebula3d"))
        (cloud,) = data_layers(plot)
        assert cloud.style.blend_mode is not None
        chrome = [l for l in plot.scene.layers if l.metadata.get("artist") in SYSTEM_3D_ARTISTS]
        assert all(l.style.blend_mode is None for l in chrome)

    def test_a_solid_recipe_uses_the_palette_not_the_colormap(self, plot):
        recipe = recipes.get_recipe("quiver3d_flow")
        assert recipe.solid
        build(plot, recipe)
        (layer,) = data_layers(plot)
        assert "cvalues" not in layer.metadata

    def test_rows_with_gaps_are_dropped_before_drawing(self, plot):
        recipe = recipes.get_recipe("signal_trace")
        data = {"x": np.arange(6.0), "y": np.array([1, np.nan, 3, 4, 5, 6.0])}
        (layer,) = recipes.apply_recipe(plot, recipe, data)
        assert len(layer.pts) == 5

    def test_invalid_data_is_refused_before_the_scene_is_touched(self, plot):
        recipe = recipes.get_recipe("signal_trace")
        with pytest.raises(recipes.RecipeError):
            recipes.apply_recipe(plot, recipe, {"x": np.arange(3.0)})
        assert plot.scene.layers == []
        assert getattr(plot, "_style_key", None) in (None, "")

    def test_a_per_point_size_column_is_scaled_into_marker_pixels(self, plot):
        recipe = recipes.get_recipe("bubble_chart")
        (layer,) = build(plot, recipe)
        sizes = np.asarray(layer.sizes)
        assert sizes.min() >= 4.0 and sizes.max() <= 22.0 and sizes.max() > sizes.min()

    def test_a_colormapped_scatter_is_round(self, plot):
        """``layerops`` used to tag every colour-by-column scatter ``marker="s"`` -- 'inert,
        nothing reads it' -- and the renderer reads it now: the Data panel's Color by drew
        squares while ``pyplot.scatter(c=...)`` drew circles."""
        (layer,) = build(plot, recipes.get_recipe("starfield"))
        assert layer.metadata.get("marker") is None
        x = np.arange(5.0)
        direct = layerops.add_xy_layer(plot, x, x, kind="scatter", label="c", c=x)
        assert direct.metadata.get("marker") is None


class TestLooks:
    def test_keep_leaves_the_page_alone_and_colours_from_the_current_style(self, plot):
        styles.apply_style(plot, styles.get_style("dracula"), layers=False)
        before = tuple(plot.options.visual.background_color)
        recipe = recipes.get_recipe("signal_trace")  # designed in chalk
        (layer,) = build(plot, recipe, look=recipes.KEEP_LOOK)
        assert tuple(plot.options.visual.background_color) == before
        assert plot._style_key == "dracula"
        assert colour_of(layer) == pytest.approx(styles.get_style("dracula").palette[0])

    def test_keep_on_an_unstyled_figure_falls_back_to_the_default_palette(self, plot):
        recipe = recipes.get_recipe("signal_trace")
        (layer,) = build(plot, recipe, look=recipes.KEEP_LOOK)
        assert colour_of(layer) == pytest.approx(styles.default_style().palette[0])

    def test_a_preset_key_overrides_the_recipes_own_look(self, plot):
        (layer,) = build(plot, recipes.get_recipe("signal_trace"), look="vintage")
        vintage = styles.get_style("vintage")
        assert plot._style_key == "vintage"
        assert tuple(plot.options.visual.background_color) == tuple(vintage.background)
        assert colour_of(layer) == pytest.approx(vintage.palette[0])

    def test_an_unknown_look_fails_loudly(self, plot):
        with pytest.raises(KeyError):
            build(plot, recipes.get_recipe("signal_trace"), look="no-such-style")

    def test_a_hand_drawn_look_wobbles_the_lines(self, plot):
        recipe = recipes.get_recipe("signal_trace")
        data = {"x": np.linspace(0, 1, 200), "y": np.zeros(200)}
        (layer,) = recipes.apply_recipe(plot, recipe, data, look="hand")
        assert float(np.ptp(layer.pts[:, 1])) > 0.0  # a flat line, until the hand shakes it


class TestAxisNames:
    def test_xy_axes_are_named_after_their_columns(self, plot):
        recipe = recipes.get_recipe("signal_trace")
        build(plot, recipe, labels={"x": "time", "y": "volts"}, title="T")
        assert (plot.xlabel, plot.ylabel, plot.title) == ("time", "volts", "T")

    def test_a_histogram_names_its_x_after_the_values_and_its_y_count(self, plot):
        build(plot, recipes.get_recipe("histogram"), labels={"y": "mass"})
        assert (plot.xlabel, plot.ylabel) == ("mass", "count")

    def test_a_line_family_does_not_call_its_x_axis_slope(self, plot):
        build(plot, recipes.get_recipe("line_family"), labels={"x": "slope", "y": "intercept"})
        assert (plot.xlabel, plot.ylabel) == ("x", "y")

    def test_3d_names_are_shortened_to_fit_beside_the_box(self, plot):
        recipe = recipes.get_recipe("surface3d")
        build(plot, recipe, labels={"x": "x", "y": "y", "z": "a_very_long_column_name"})
        assert len(plot.axes3d.zlabel) <= 10 and plot.axes3d.zlabel.endswith("…")

    def test_axis_names_resolves_literals_and_slots(self):
        recipe = recipes.get_recipe("histogram")
        assert recipes.axis_names(recipe, {"y": "mass"}) == {"x": "mass", "y": "count"}


class TestUndo:
    def command(self, plot, recipe, **kwargs):
        return recipes.recipe_command(
            plot, None, recipe, sample(recipe), labels=labels(recipe), **kwargs
        )

    def test_undo_removes_the_layers_and_puts_the_page_back(self, plot):
        before = tuple(plot.options.visual.background_color)
        recipe = recipes.get_recipe("signal_trace")
        command = self.command(plot, recipe)
        command.do()
        assert data_layers(plot) and plot._style_key == "chalk"
        command.undo()
        assert data_layers(plot) == []
        assert tuple(plot.options.visual.background_color) == before
        assert not getattr(plot, "_style_key", "")

    def test_undo_leaves_the_users_own_layers_alone(self, plot):
        mine = layerops.add_xy_layer(
            plot, np.arange(5.0), np.arange(5.0), kind="line", label="mine"
        )
        command = self.command(plot, recipes.get_recipe("starfield"))
        command.do()
        command.undo()
        assert data_layers(plot) == [mine]

    def test_undo_of_a_3d_recipe_returns_to_2d_and_keeps_the_box_out_of_the_way(self, plot):
        command = self.command(plot, recipes.get_recipe("surface3d"))
        command.do()
        assert plot.active_panel.is_3d()
        command.undo()
        assert not plot.active_panel.is_3d()
        assert data_layers(plot) == []

    def test_undo_restores_the_axis_names(self, plot):
        plot.xlabel, plot.ylabel = "mine", "also mine"
        command = self.command(plot, recipes.get_recipe("signal_trace"), title="Hello")
        command.do()
        assert plot.xlabel == "time"
        command.undo()
        assert (plot.xlabel, plot.ylabel) == ("mine", "also mine")

    def test_redo_draws_it_again_without_duplicating(self, plot):
        command = self.command(plot, recipes.get_recipe("signal_trace"))
        command.do()
        command.undo()
        command.do()
        assert len(data_layers(plot)) == 1

    def test_the_label_names_the_recipe_for_the_undo_menu(self, plot):
        assert "Signal trace" in self.command(plot, recipes.get_recipe("signal_trace")).label

    def test_the_dataset_name_prefixes_the_layer_label(self, plot):
        recipe = recipes.get_recipe("signal_trace")
        command = recipes.recipe_command(plot, None, recipe, sample(recipe), name="run7")
        command.do()
        assert data_layers(plot)[0].label == "run7: Signal trace"


class TestHeadlessRender:
    @pytest.mark.parametrize("key", ["signal_trace", "starfield", "surface3d", "heatmap_matrix"])
    def test_a_recipe_renders_to_an_rgb_array(self, key):
        recipe = recipes.get_recipe(key)
        rgb = recipes.render_preview_rgb(recipe, width=400, height=250)
        assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
        assert rgb.shape[0] > 100 and rgb.shape[1] > 100
        assert float(rgb.std()) > 1.0  # not a blank page

    def test_a_render_of_the_users_data_uses_that_data(self):
        recipe = recipes.get_recipe("signal_trace")
        flat = recipes.render_preview_rgb(
            recipe, {"x": np.arange(50.0), "y": np.zeros(50)}, look="clean", width=400, height=250
        )
        wavy = recipes.render_preview_rgb(
            recipe,
            {"x": np.arange(50.0), "y": np.sin(np.arange(50.0) / 3)},
            look="clean",
            width=400,
            height=250,
        )
        assert not np.array_equal(flat, wavy)

    def test_the_recipe_name_does_not_leak_into_a_legend(self):
        import matplotlib.pyplot as mpl

        from glplot.utils import preview

        captured = {}
        real = preview.render_preview_array

        def spy(plot, scale):
            fig = preview._build_preview_figure(plot, scale)
            captured["legend"] = [ax.get_legend() for ax in fig.axes]
            mpl.close(fig)
            return real(plot, scale)

        import glplot.utils.preview as preview_module

        original = preview_module.render_preview_array
        preview_module.render_preview_array = spy
        try:
            recipes.render_preview_rgb(recipes.get_recipe("signal_trace"), width=400, height=250)
        finally:
            preview_module.render_preview_array = original
        assert captured["legend"] and all(legend is None for legend in captured["legend"])

    def test_build_preview_plot_accepts_the_callers_own_figure_factory(self):
        made = []

        def factory():
            made.append(GPULinePlot(width=300, height=200, title="GLPlot"))
            return made[-1]

        plot = recipes.build_preview_plot(recipes.get_recipe("bar_chart"), new_plot=factory)
        assert plot is made[0] and data_layers(plot)
