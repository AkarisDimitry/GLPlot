"""``list_recipes`` and ``add_recipe``: the gallery's recipes, reachable by a language model.

Driven as plain ``Toolbox`` methods (no SDK), against the same headless renderer the other MCP
tools use. The properties worth pinning are the ones a model cannot see for itself: that the
columns it did *not* name were guessed sensibly and reported back, that a refusal says exactly
what to change, that the recipe's name does not turn up as a legend entry nobody asked for, that
a sample dataset is only kept if the drawing worked, and that the figure can still be edited
(remove_artist) afterwards.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from glplot.core.camera3d import SYSTEM_3D_ARTISTS
from glplot.gui import recipes, styles
from glplot.mcp.state import FileSandbox, Limits, McpError
from glplot.mcp.toolbox import Toolbox

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def box(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    toolbox = Toolbox(FileSandbox([root]))
    yield toolbox
    toolbox.close_all()


def fig(box, three_d=False, **kwargs):
    return box.create_figure(projection="3d" if three_d else "2d", **kwargs).data["figure_id"]


def signal(box, name="run", n=120):
    box.create_dataset(
        name, {"time": f"linspace(0, 6, {n})", "signal": "sin(time*3)", "other": "cos(time)"}
    )
    return name


def data_layers(box, figure_id):
    return [
        layer
        for layer in box._figures[figure_id].plot.scene.layers
        if layer.metadata.get("artist") not in SYSTEM_3D_ARTISTS
    ]


class TestListRecipes:
    def test_lists_every_recipe_as_json(self, box):
        result = box.list_recipes()
        rows = result.data["recipes"]
        assert [r["recipe"] for r in rows] == list(recipes.RECIPE_KEYS)
        json.dumps(result.data)
        assert str(len(rows)) in result.message

    def test_each_row_says_what_it_needs_and_how_it_looks(self, box):
        row = next(r for r in box.list_recipes().data["recipes"] if r["recipe"] == "starfield")
        assert row["dimension"] == "2D" and row["kind"] == "scatter" and row["look"] == "synthwave"
        slots = {s["slot"]: s for s in row["needs"]}
        assert slots["x"]["required"] and slots["c"]["required"] and not slots["s"]["required"]
        assert row["sample"] is True and "hint" in slots["c"]

    def test_the_matrix_recipe_says_it_uses_every_column(self, box):
        row = next(r for r in box.list_recipes().data["recipes"] if r["recipe"] == "heatmap_matrix")
        assert row["needs"][0]["uses"] == "every column of the dataset"

    def test_filtering_by_category(self, box):
        rows = box.list_recipes("3D").data["recipes"]
        assert rows and {r["category"] for r in rows} == {"3D"}
        assert {r["dimension"] for r in rows} == {"3D"}

    def test_an_unknown_category_suggests_and_lists(self, box):
        with pytest.raises(McpError, match="Did you mean 'Lines'") as info:
            box.list_recipes("Line")
        assert "3D" in str(info.value)

    def test_it_changes_nothing(self, box):
        box.list_recipes()
        assert box.figure_ids() == [] and len(box.datasets) == 0


class TestDrawFromADataset:
    def test_columns_are_guessed_drawn_and_reported(self, box):
        f = fig(box)
        signal(box)
        result = box.add_recipe(f, "signal_trace", dataset="run")
        assert result.data["columns"] == {"x": "time", "y": "signal"}
        assert result.data["recipe"] == "signal_trace" and result.data["dataset"] == "run"
        assert result.data["look"] == "chalk" and result.data["artist_id"] == "a1"
        assert len(data_layers(box, f)) == 1

    def test_the_drawing_uses_the_datasets_numbers(self, box):
        f = fig(box)
        signal(box)
        box.add_recipe(f, "signal_trace", dataset="run")
        (layer,) = data_layers(box, f)
        assert np.allclose(layer.pts[:, 0], box.datasets.column("run", "time"), atol=1e-5)

    def test_columns_override_only_what_is_named(self, box):
        f = fig(box)
        signal(box)
        result = box.add_recipe(f, "signal_trace", dataset="run", columns={"y": "other"})
        assert result.data["columns"] == {"x": "time", "y": "other"}

    def test_the_row_number_can_stand_in_for_a_column(self, box):
        f = fig(box)
        signal(box)
        result = box.add_recipe(f, "signal_trace", dataset="run", columns={"x": "(index)"})
        assert result.data["columns"]["x"] == "(index)"
        (layer,) = data_layers(box, f)
        assert layer.pts[:3, 0].tolist() == [0.0, 1.0, 2.0]
        assert box.render_figure(f).data["panels"][0]["xlabel"] == "row"

    def test_a_slot_can_be_emptied_with_null(self, box):
        f = fig(box)
        box.create_dataset("pts", {"x": "linspace(0,1,50)", "y": "x**2", "age": "x*3"})
        guessed = box.add_recipe(f, "starfield", dataset="pts").data["columns"]
        assert guessed["c"] == "age"
        with pytest.raises(McpError, match="Colour by"):
            box.add_recipe(fig(box), "starfield", dataset="pts", columns={"c": None})

    def test_axes_are_named_after_the_columns(self, box):
        f = fig(box)
        signal(box)
        box.add_recipe(f, "signal_trace", dataset="run")
        panel = box.render_figure(f).data["panels"][0]
        assert (panel["xlabel"], panel["ylabel"]) == ("time", "signal")

    def test_a_histogram_names_its_y_axis_count(self, box):
        f = fig(box)
        box.create_dataset("v", {"value": "noise(2000, 1, 3)"})
        box.add_recipe(f, "histogram", dataset="v")
        panel = box.render_figure(f).data["panels"][0]
        assert (panel["xlabel"], panel["ylabel"]) == ("value", "count")

    def test_the_matrix_recipe_uses_every_column(self, box):
        f = fig(box)
        box.create_dataset("m", {"a": "linspace(0,1,10)", "b": "a*2", "c": "a*3", "d": "a*4"})
        result = box.add_recipe(f, "heatmap_matrix", dataset="m")
        assert result.data["columns"] == {"matrix": "all 4 columns"}
        assert box.render_figure(f).image.startswith(PNG)

    def test_the_dataset_is_not_changed(self, box):
        f = fig(box)
        signal(box)
        before = box.datasets.column("run", "signal").copy()
        box.add_recipe(f, "signal_trace", dataset="run")
        assert np.array_equal(box.datasets.column("run", "signal"), before)

    def test_title_and_label(self, box):
        f = fig(box)
        signal(box)
        box.add_recipe(f, "signal_trace", dataset="run", title="Run 7", label="trace")
        box.set_axes(f, legend=True)
        panel = box.render_figure(f).data["panels"][0]
        assert panel["title"] == "Run 7" and panel["legend"] == ["trace"]

    def test_the_recipes_name_stays_out_of_the_legend_unless_asked(self, box):
        """A one-entry legend naming the recipe, over the plot, is noise nobody requested."""
        f = fig(box)
        signal(box)
        box.add_recipe(f, "signal_trace", dataset="run")
        box.set_axes(f, legend=True)
        assert "legend" not in box.render_figure(f).data["panels"][0]

    def test_the_recipes_name_stays_out_of_a_3d_legend_too(self, box):
        """The 3-D line branch of the export read ``layer.label`` directly and ignored the
        bookkeeping flag every other branch honours."""
        import matplotlib.pyplot as mpl

        from glplot.utils import preview

        f = fig(box, three_d=True)
        box.add_recipe(f, "quiver3d_flow", sample=True)
        figure = preview._build_preview_figure(box._figures[f].plot, 1.0)
        try:
            assert all(ax.get_legend_handles_labels()[1] == [] for ax in figure.axes)
        finally:
            mpl.close(figure)

    def test_but_a_label_asked_for_does_reach_the_3d_legend(self, box):
        """The positive control for the test above: without it, 'no entry' could just mean the
        export never draws legends for these layers at all."""
        import matplotlib.pyplot as mpl

        from glplot.utils import preview

        f = fig(box, three_d=True)
        box.add_recipe(f, "quiver3d_flow", sample=True, label="arrows")
        figure = preview._build_preview_figure(box._figures[f].plot, 1.0)
        try:
            labels = [t for ax in figure.axes for t in ax.get_legend_handles_labels()[1]]
        finally:
            mpl.close(figure)
        assert labels == ["arrows"]


class TestSampleData:
    def test_sample_adds_a_dataset_and_draws_from_it(self, box):
        f = fig(box)
        result = box.add_recipe(f, "starfield", sample=True)
        assert result.data["dataset"] == "starfield" and "starfield" in box.datasets
        assert box.datasets.get("starfield").n_rows() == recipes.get_recipe("starfield").sample_rows
        assert result.data["columns"] == {"x": "x_pc", "y": "y_pc", "c": "age"}
        assert "Added sample dataset" in result.message

    def test_a_second_sample_does_not_overwrite_the_first(self, box):
        box.add_recipe(fig(box), "bar_chart", sample=True)
        second = box.add_recipe(fig(box), "bar_chart", sample=True)
        assert second.data["dataset"] == "bar_chart_2"
        assert box.datasets.names() == ["bar_chart", "bar_chart_2"]

    def test_a_users_own_dataset_of_that_name_is_never_replaced(self, box):
        box.create_dataset("bar_chart", {"a": [1, 2, 3]})
        box.add_recipe(fig(box), "bar_chart", sample=True)
        assert box.datasets.column("bar_chart", "a").tolist() == [1.0, 2.0, 3.0]

    def test_the_sample_dataset_can_be_reused_by_name(self, box):
        box.add_recipe(fig(box), "signal_trace", sample=True)
        again = box.add_recipe(fig(box), "signal_trace", dataset="signal_trace")
        assert again.data["columns"] == {"x": "time", "y": "signal"}

    def test_the_dataset_is_kept_only_if_the_drawing_worked(self, box):
        """Nothing half-done is left behind when the figure turns out not to take the recipe."""
        f = fig(box)  # 2-D figure, 3-D recipe
        with pytest.raises(McpError):
            box.add_recipe(f, "surface3d", sample=True)
        assert len(box.datasets) == 0 and data_layers(box, f) == []

    def test_a_full_dataset_store_refuses_before_drawing(self, tmp_path):
        small = Toolbox(FileSandbox(()), Limits(max_datasets=1))
        try:
            small.create_dataset("only", {"a": [1, 2]})
            f = small.create_figure().data["figure_id"]
            with pytest.raises(McpError, match="Too many datasets"):
                small.add_recipe(f, "bar_chart", sample=True)
            assert data_layers(small, f) == []
        finally:
            small.close_all()

    @pytest.mark.parametrize("recipe", recipes.RECIPES, ids=lambda r: r.key)
    def test_every_recipe_draws_from_its_sample_and_renders(self, box, recipe):
        f = fig(box, three_d=recipe.is_3d)
        result = box.add_recipe(f, recipe.key, sample=True)
        assert result.data["layers"] >= 1 and result.data["look"] == recipe.style
        assert box.render_figure(f, scale=0.4).image.startswith(PNG)


class TestRefusals:
    def test_an_unknown_recipe_suggests_and_points_at_the_list(self, box):
        f = fig(box)
        with pytest.raises(McpError, match="Did you mean 'signal_trace'") as info:
            box.add_recipe(f, "signal_trac", sample=True)
        assert "list_recipes" in str(info.value)

    def test_no_data_says_how_to_supply_it_and_which_slots_exist(self, box):
        with pytest.raises(McpError, match="sample=true") as info:
            box.add_recipe(fig(box), "signal_trace")
        assert "Slots of 'signal_trace'" in str(info.value)

    def test_dataset_and_sample_together_are_ambiguous(self, box):
        signal(box)
        with pytest.raises(McpError, match="not both"):
            box.add_recipe(fig(box), "signal_trace", dataset="run", sample=True)

    def test_an_unknown_dataset_lists_the_real_ones(self, box):
        signal(box)
        with pytest.raises(McpError, match="run"):
            box.add_recipe(fig(box), "signal_trace", dataset="rn")

    def test_an_unknown_slot_lists_the_slots(self, box):
        signal(box)
        with pytest.raises(McpError, match="no slot 'q'") as info:
            box.add_recipe(fig(box), "signal_trace", dataset="run", columns={"q": "time"})
        assert "x (required)" in str(info.value)

    def test_an_unknown_column_lists_columns_suggests_and_mentions_the_row_number(self, box):
        signal(box)
        with pytest.raises(McpError, match="Did you mean 'signal'") as info:
            box.add_recipe(fig(box), "signal_trace", dataset="run", columns={"y": "signa"})
        assert "(index)" in str(info.value) and "time" in str(info.value)

    def test_a_non_string_column_is_refused(self, box):
        signal(box)
        with pytest.raises(McpError, match="column name"):
            box.add_recipe(fig(box), "signal_trace", dataset="run", columns={"y": 3})

    def test_the_matrix_slot_cannot_be_mapped_to_one_column(self, box):
        box.create_dataset("m", {"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(McpError, match="every column"):
            box.add_recipe(fig(box), "heatmap_matrix", dataset="m", columns={"matrix": "a"})

    def test_a_3d_recipe_on_a_2d_figure_says_what_to_create(self, box):
        with pytest.raises(McpError, match="projection='3d'"):
            box.add_recipe(fig(box), "surface3d", sample=True)

    def test_a_2d_recipe_on_a_3d_figure(self, box):
        with pytest.raises(McpError, match="2-D"):
            box.add_recipe(fig(box, three_d=True), "signal_trace", sample=True)

    def test_an_unknown_look_suggests_and_lists(self, box):
        with pytest.raises(McpError, match="Did you mean 'journal'") as info:
            box.add_recipe(fig(box), "signal_trace", sample=True, look="journl")
        assert "keep" in str(info.value) and "recipe" in str(info.value)

    def test_data_that_cannot_be_drawn_says_why_and_what_to_change(self, box):
        box.create_dataset(
            "pts", {"x": "noise(40,1,1)", "y": "noise(40,1,2)", "z": "noise(40,1,3)"}
        )
        f = fig(box, three_d=True)
        with pytest.raises(McpError, match="grid") as info:
            box.add_recipe(f, "surface3d", dataset="pts")
        assert "Slots of 'surface3d'" in str(info.value)
        assert data_layers(box, f) == []

    def test_a_missing_required_column_names_the_slot_and_the_columns_available(self, box):
        box.create_dataset("two", {"a": [1, 2, 3], "b": [3, 4, 5]})
        with pytest.raises(McpError, match="Colour by") as info:
            box.add_recipe(fig(box), "starfield", dataset="two")
        assert "Columns here: a, b" in str(info.value)

    def test_a_bad_panel_is_refused(self, box):
        signal(box)
        with pytest.raises(McpError, match="single panel"):
            box.add_recipe(fig(box), "signal_trace", dataset="run", panel=2)

    def test_a_refusal_leaves_the_figure_exactly_as_it_was(self, box):
        f = fig(box)
        signal(box)
        box.add_recipe(f, "signal_trace", dataset="run")
        before = box.describe_figure(f).data["artists"]
        for kwargs in ({"columns": {"y": "nope"}}, {"look": "nope"}, {"recipe": "nope"}):
            args = {"recipe": "signal_trace", "dataset": "run", **kwargs}
            with pytest.raises(McpError):
                box.add_recipe(f, **args)
        assert box.describe_figure(f).data["artists"] == before


class TestLooks:
    def test_the_recipes_look_is_applied_to_the_page(self, box):
        f = fig(box)
        box.add_recipe(f, "signal_trace", sample=True)
        plot = box._figures[f].plot
        assert plot._style_key == "chalk"
        assert tuple(plot.options.visual.background_color) == tuple(
            styles.get_style("chalk").background
        )

    def test_a_style_key_overrides_it(self, box):
        f = fig(box)
        result = box.add_recipe(f, "signal_trace", sample=True, look="vintage")
        assert result.data["look"] == "vintage"

    def test_keep_leaves_the_figures_current_look(self, box):
        f = fig(box, style="dracula")
        before = tuple(box._figures[f].plot.options.visual.background_color)
        result = box.add_recipe(f, "signal_trace", sample=True, look="keep")
        assert result.data["look"] == "dracula"
        assert tuple(box._figures[f].plot.options.visual.background_color) == before

    def test_a_later_set_style_still_wins(self, box):
        f = fig(box)
        box.add_recipe(f, "signal_trace", sample=True)
        box.set_style(f, "clean")
        assert box._figures[f].plot._style_key == "clean"


class TestTheFigureStaysEditable:
    def test_the_recipe_is_one_artist_and_removable(self, box):
        f = fig(box)
        box.add_recipe(f, "signal_trace", sample=True)
        (artist,) = box.describe_figure(f).data["artists"]
        assert artist["kind"] == "recipe:signal_trace" and artist["dataset"] == "signal_trace"
        box.remove_artist(f, artist["artist_id"])
        assert data_layers(box, f) == []

    def test_removing_a_3d_recipe_keeps_the_axis_box(self, box):
        f = fig(box, three_d=True)
        box.add_recipe(f, "surface3d", sample=True)
        chrome = [
            l
            for l in box._figures[f].plot.scene.layers
            if l.metadata.get("artist") in SYSTEM_3D_ARTISTS
        ]
        assert chrome
        box.remove_artist(f)
        assert data_layers(box, f) == []
        assert box.render_figure(f).image.startswith(PNG)

    def test_recipes_combine_with_the_other_tools(self, box):
        f = fig(box)
        signal(box)
        box.add_recipe(f, "signal_trace", dataset="run", look="keep")
        box.add_function(f, "0.5*sin(2*x)", 0, 6, label="model")
        box.add_annotation(f, "hline", y=0.0, label="zero")
        box.set_axes(f, title="Run vs model", legend=True, grid=True)
        panel = box.render_figure(f).data["panels"][0]
        assert panel["title"] == "Run vs model" and panel["legend"] == ["model", "zero"]
        assert len(box.describe_figure(f).data["artists"]) == 3

    def test_two_recipes_on_one_figure_take_successive_palette_colours(self, box):
        f = fig(box)
        signal(box)
        box.add_recipe(f, "signal_trace", dataset="run", look="clean")
        box.add_recipe(f, "signal_trace", dataset="run", look="keep", columns={"y": "other"})
        first, second = data_layers(box, f)
        assert tuple(first.style.color) != tuple(second.style.color)

    def test_it_draws_into_the_requested_panel(self, box):
        f = box.create_figure(rows=1, cols=2).data["figure_id"]
        signal(box)
        box.add_recipe(f, "signal_trace", dataset="run", panel=2)
        plot = box._figures[f].plot
        assert len(plot.panels[0].scene.layers) == 0 and len(plot.panels[1].scene.layers) == 1

    def test_save_works_after_a_recipe(self, box, tmp_path):
        f = fig(box)
        box.add_recipe(f, "starfield", sample=True)
        result = box.save_figure(f, "star.png")
        assert result.data["bytes"] > 1000


class TestIndexedFlows:
    def test_a_loaded_file_goes_straight_into_a_recipe(self, box, tmp_path):
        (box.sandbox.roots[0] / "run.csv").write_text(
            "time,signal\n" + "\n".join(f"{t},{np.sin(t):.4f}" for t in np.linspace(0, 6, 60))
        )
        box.load_dataset("run.csv", name="file")
        f = fig(box)
        result = box.add_recipe(f, "step_response", dataset="file")
        assert result.data["columns"] == {"x": "time", "y": "signal"}
        assert box.render_figure(f).image.startswith(PNG)

    def test_the_result_text_is_json_safe(self, box):
        result = box.add_recipe(fig(box), "signal_trace", sample=True)
        message, _, payload = result.text().partition("\n")
        assert message.startswith("Drew recipe") and json.loads(payload)["recipe"] == "signal_trace"

    def test_a_gap_in_the_data_is_dropped_not_fatal(self, box):
        box.create_dataset("gappy", {"x": [0, 1, 2, 3, 4, 5], "y": [1, None, 3, 4, 5, 6]})
        f = fig(box)
        box.add_recipe(f, "signal_trace", dataset="gappy")
        (layer,) = data_layers(box, f)
        assert len(layer.pts) == 5

    def test_a_png_is_returned_by_render_after_the_sample_path(self, box):
        f = fig(box)
        box.add_recipe(f, "hexbin_density", sample=True)
        data = box.render_figure(f).image
        from PIL import Image

        assert Image.open(io.BytesIO(data)).size[0] > 200
