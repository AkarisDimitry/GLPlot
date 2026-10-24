"""The Gallery panel, driven through the sanctioned headless imgui harness.

A real imgui context, a real frame, a real draw -- which is what catches a wrong imgui_bundle
signature or an unbalanced push/pop -- but no window and no GL: the figure the panel is built on
has no ``window``, so its thumbnails are placeholders and its preview job never starts. The
state machine is what is tested here (selection, column assignment, validation, apply, sample
data, undo); how the cards *look* is verified against the live window, not in pytest.
"""

from __future__ import annotations

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.core.camera3d import SYSTEM_3D_ARTISTS
from glplot.gui import recipes
from glplot.gui.commands import CommandQueue
from glplot.gui.datasets import Column, DataSet, DataStore
from glplot.gui.history import UndoStack

imgui = pytest.importorskip("imgui_bundle").imgui

from glplot.gui.panels import gallery as gallery_mod  # noqa: E402
from glplot.gui.panels.gallery import GalleryPanel, search_score  # noqa: E402
from glplot.gui.panels.thumbs import Thumbnails, fit_rect  # noqa: E402


class FakeWorkspace:
    def __init__(self, plot):
        self.plot = plot
        self.queue = CommandQueue()
        self.undo = UndoStack()
        self.store = DataStore()
        self.hud = None


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


@pytest.fixture
def harness():
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = 600, 1000
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    io.delta_time = 1 / 60.0
    yield imgui
    imgui.destroy_context(ctx)


@pytest.fixture
def ws():
    return FakeWorkspace(gplt.figure())


@pytest.fixture
def panel(ws):
    gallery = GalleryPanel(ws)
    gallery.async_preview = False
    return gallery


def frame(panel, count=1):
    for _ in range(count):
        imgui.new_frame()
        imgui.begin("Gallery")
        panel.draw()
        imgui.end()
        imgui.render()
        assert imgui.get_draw_data() is not None


def signal_dataset(name="signal", n=200):
    x = np.linspace(0.0, 10.0, n)
    return DataSet(name, [Column("t", x), Column("v", np.sin(x))])


def drain(ws):
    ws.queue.drain(ws.plot)


def data_layers(plot):
    return [l for l in plot.scene.layers if l.metadata.get("artist") not in SYSTEM_3D_ARTISTS]


class TestDrawing:
    def test_an_empty_workspace_draws_and_explains_what_to_do(self, harness, panel):
        frame(panel, 2)

    def test_every_state_draws_without_unbalanced_widgets(self, harness, panel, ws):
        ws.store.add(signal_dataset())
        frame(panel)
        for key in recipes.RECIPE_KEYS:
            panel.select(key)
            frame(panel)
        panel.select(None)
        frame(panel)

    def test_a_search_with_no_match_and_with_matches_draws(self, harness, panel):
        panel._query = "zzzzzz"
        frame(panel)
        panel._query = "surface"
        frame(panel)

    def test_the_category_chips_draw_at_a_narrow_width(self, harness, panel):
        imgui.get_io().display_size = 260, 900
        frame(panel, 2)

    def test_the_title_and_icon_are_the_panels_identity(self):
        assert GalleryPanel.title == "Gallery" and GalleryPanel.icon == "grid"
        assert GalleryPanel.default_open is False


class TestSelectionAndAssignment:
    def test_selecting_guesses_the_columns(self, panel, ws):
        ws.store.add(signal_dataset())
        panel.select("signal_trace")
        assert panel._assignment == {"x": "t", "y": "v", "c": None}

    def test_the_first_dataset_is_chosen_when_none_is(self, panel, ws):
        ws.store.add(signal_dataset("first"))
        ws.store.add(signal_dataset("second"))
        panel.select("signal_trace")
        panel._current_dataset()
        assert panel._dataset == "first"

    def test_a_deleted_dataset_falls_back_instead_of_crashing(self, harness, panel, ws):
        ds = ws.store.add(signal_dataset())
        panel.select("signal_trace")
        frame(panel)
        ws.store.remove(ds)
        frame(panel)
        assert panel._dataset is None

    def test_choosing_another_dataset_reassigns(self, panel, ws):
        ws.store.add(signal_dataset("a"))
        ws.store.add(
            DataSet("b", [Column("height", np.arange(5.0)), Column("depth", np.arange(5.0))])
        )
        panel.select("signal_trace")
        panel.choose_dataset("b")
        assert panel._assignment["x"] == "height"

    def test_assigning_a_column_changes_the_mapping_and_clears_the_verdict(self, panel, ws):
        ws.store.add(signal_dataset())
        panel.select("signal_trace")
        panel._problem_for(panel.selected_recipe, panel._current_dataset())
        panel.assign("y", "t")
        assert panel._problem_key is None and panel._assignment["y"] == "t"

    def test_an_unknown_recipe_is_refused_not_stored(self, panel):
        with pytest.raises(KeyError):
            panel.select("not-a-recipe")
        assert panel.selected_recipe is None

    def test_closing_the_detail(self, panel):
        panel.select("bar_chart")
        panel.select(None)
        assert panel.selected_recipe is None


class TestValidationDisablesApply:
    def test_no_dataset_is_a_reason(self, panel):
        panel.select("signal_trace")
        assert "dataset" in panel._problem_for(panel.selected_recipe, None)

    def test_a_missing_required_column_blocks_apply(self, panel, ws):
        ws.store.add(signal_dataset())
        panel.select("surface3d")
        assert panel._problem_for(panel.selected_recipe, panel._current_dataset())
        assert panel.apply_selected() is False
        assert len(ws.queue) == 0

    def test_the_verdict_is_computed_once_per_choice_not_once_per_frame(
        self, harness, panel, ws, monkeypatch
    ):
        ws.store.add(signal_dataset())
        panel.select("signal_trace")
        calls = []
        real = recipes.validate_data
        monkeypatch.setattr(recipes, "validate_data", lambda *a: calls.append(1) or real(*a))
        frame(panel, 5)
        assert len(calls) == 1
        panel.assign("y", "t")
        frame(panel, 5)
        assert len(calls) == 2


class TestApply:
    def test_apply_draws_the_recipe_as_one_undoable_step(self, panel, ws):
        ws.store.add(signal_dataset())
        panel.select("signal_trace")
        assert panel.apply_selected() is True
        assert data_layers(ws.plot) == []  # queued, not drawn from the draw callback
        drain(ws)
        (layer,) = data_layers(ws.plot)
        assert layer.label == "signal: Signal trace"
        assert ws.plot._style_key == "chalk"
        ws.undo.undo()
        assert data_layers(ws.plot) == []

    def test_the_look_choice_is_honoured(self, panel, ws):
        ws.store.add(signal_dataset())
        panel.select("signal_trace")
        panel.set_look("aurora")
        panel.apply_selected()
        drain(ws)
        assert ws.plot._style_key == "aurora"

    def test_keeping_the_current_look_leaves_the_page(self, panel, ws):
        ws.store.add(signal_dataset())
        before = tuple(ws.plot.options.visual.background_color)
        panel.select("signal_trace")
        panel.set_look(recipes.KEEP_LOOK)
        panel.apply_selected()
        drain(ws)
        assert tuple(ws.plot.options.visual.background_color) == before

    def test_the_axes_are_named_after_the_columns(self, panel, ws):
        ws.store.add(signal_dataset())
        panel.select("signal_trace")
        panel.apply_selected()
        drain(ws)
        assert (ws.plot.xlabel, ws.plot.ylabel) == ("t", "v")

    def test_the_row_number_can_stand_in_for_a_column(self, panel, ws):
        ws.store.add(DataSet("one", [Column("v", np.sin(np.linspace(0, 6, 80)))]))
        panel.select("signal_trace")
        assert panel._assignment["x"] == recipes.INDEX
        panel.apply_selected()
        drain(ws)
        (layer,) = data_layers(ws.plot)
        assert layer.pts[:, 0].tolist()[:3] == [0.0, 1.0, 2.0]
        assert ws.plot.xlabel == "row"

    def test_later_edits_to_the_table_do_not_reach_the_layer(self, panel, ws):
        """A recipe draws a copy: re-apply to pick up edits (it is not a live link)."""
        ds = ws.store.add(signal_dataset())
        panel.select("signal_trace")
        panel.apply_selected()
        ds.columns[1].values[:] = 99.0
        drain(ws)
        (layer,) = data_layers(ws.plot)
        assert float(layer.pts[:, 1].max()) < 5.0

    def test_the_matrix_recipe_uses_every_column(self, panel, ws):
        grid = np.random.default_rng(1).random((12, 5))
        ws.store.add(DataSet("m", [Column(f"c{i}", grid[:, i]) for i in range(5)]))
        panel.select("heatmap_matrix")
        assert panel._problem_for(panel.selected_recipe, panel._current_dataset()) is None
        panel.apply_selected()
        drain(ws)
        (layer,) = data_layers(ws.plot)
        assert layer.metadata["matrix"].shape == (12, 5)

    def test_a_3d_recipe_on_a_real_grid(self, panel, ws):
        gx, gy = np.meshgrid(np.arange(6.0), np.arange(7.0))
        ws.store.add(
            DataSet(
                "g",
                [Column("x", gx.ravel()), Column("y", gy.ravel()), Column("z", (gx * gy).ravel())],
            )
        )
        panel.select("surface3d")
        panel.apply_selected()
        drain(ws)
        assert ws.plot.active_panel.is_3d() and len(data_layers(ws.plot)) == 1


class TestTrySampleData:
    def test_one_click_adds_the_dataset_and_draws_the_recipe(self, panel, ws):
        panel.select("starfield")
        assert panel.try_sample() is True
        drain(ws)
        assert ws.store.names() == ["Colour-mapped scatter"]
        assert len(data_layers(ws.plot)) == 1
        assert panel._dataset == "Colour-mapped scatter"
        assert panel._assignment["c"] == "age"

    def test_it_is_a_single_undo_step_that_removes_both(self, panel, ws):
        panel.select("starfield")
        panel.try_sample()
        drain(ws)
        ws.undo.undo()
        assert ws.store.names() == [] and data_layers(ws.plot) == []
        ws.undo.redo()
        assert len(ws.store.names()) == 1 and len(data_layers(ws.plot)) == 1

    def test_the_sample_dataset_is_tagged_and_does_not_collide(self, panel, ws):
        panel.select("bar_chart")
        panel.try_sample()
        drain(ws)
        panel.try_sample()
        drain(ws)
        assert ws.store.names() == ["Bar chart", "Bar chart (2)"]
        assert ws.store.get("Bar chart").source == "sample"

    def test_every_recipe_can_be_tried(self, panel, ws):
        for key in recipes.RECIPE_KEYS:
            panel.select(key)
            assert panel.try_sample() is True
            drain(ws)
        assert len(ws.store.names()) == len(recipes.RECIPES)

    def test_with_nothing_selected_it_does_nothing(self, panel):
        assert panel.try_sample() is False


class TestSearchAndFilter:
    def test_category_filter(self, panel):
        panel._category = "3D"
        assert {r.category for r in panel._visible_recipes()} == {"3D"}
        panel._category = None
        assert len(panel._visible_recipes()) == len(recipes.RECIPES)

    def test_a_word_in_the_name_finds_it_first(self, panel):
        panel._query = "surface"
        assert panel._visible_recipes()[0].key == "surface3d"

    def test_a_tag_finds_a_recipe_whose_name_does_not_say_it(self, panel):
        panel._query = "oscilloscope"
        assert [r.key for r in panel._visible_recipes()] == ["signal_trace"]

    def test_a_word_in_the_description_finds_it(self, panel):
        panel._query = "chalk"
        assert "signal_trace" in [r.key for r in panel._visible_recipes()]

    def test_prose_is_not_fuzzy_matched(self, panel):
        """'heat' is a subsequence of two lines of nearly any description; it must not match
        everything (it did, when the palette's matcher ran over the whole description)."""
        panel._query = "heat"
        assert len(panel._visible_recipes()) < 6

    def test_an_abbreviation_still_finds_a_name(self, panel):
        panel._query = "hxbn"
        assert "hexbin_density" in [r.key for r in panel._visible_recipes()]

    def test_the_search_and_the_chip_combine(self, panel):
        panel._query = "3-d"
        panel._category = "Lines"
        assert panel._visible_recipes() == []

    def test_blank_search_matches_everything(self):
        assert all(search_score(r, "   ") > 0 for r in recipes.RECIPES)

    def test_a_prefix_beats_a_mid_name_hit(self):
        bar = recipes.get_recipe("bar_chart")
        assert (
            search_score(bar, "bar")
            > search_score(recipes.get_recipe("bars3d_city"), "city") - 1000
        )


class TestPreviewPlumbing:
    def test_no_window_means_no_preview_job(self, harness, panel, ws):
        ws.store.add(signal_dataset())
        panel.select("signal_trace")
        frame(panel, 3)
        assert panel._preview.job is None and panel._preview.ready is False

    def test_with_a_window_the_preview_is_rendered_and_shown(self, harness, panel, ws, monkeypatch):
        """A fake 'window' and a fake uploader: the cache turns on, the preview renders (inline,
        ``async_preview`` is off) and lands in the cache -- all without a GL call."""
        from glplot.gui.panels import thumbs

        uploads = []
        monkeypatch.setattr(
            thumbs, "gl_upload", lambda rgba: uploads.append(rgba.shape) or len(uploads)
        )
        monkeypatch.setattr(thumbs, "gl_delete", lambda texture_id: None)
        ws.plot.window = object()  # "a window exists": enough to turn the thumbnail cache on
        ws.store.add(signal_dataset())
        panel.select("signal_trace")
        frame(panel, 2)
        assert panel._preview.ready is True
        assert panel._thumbs.peek(("preview",)) is not None
        shapes = [s for s in uploads if s[0] > 400]
        assert shapes, "the live preview was uploaded at figure scale"

    def test_a_huge_dataset_is_thinned_for_the_preview_only(self, panel):
        big = {"x": np.arange(1_000_000.0), "y": np.arange(1_000_000.0)}
        thinned = panel._thin(recipes.get_recipe("signal_trace"), big)
        assert len(thinned["x"]) <= gallery_mod.PREVIEW_ROW_LIMIT
        assert len(big["x"]) == 1_000_000

    def test_a_lattice_is_never_thinned(self, panel):
        big = {k: np.arange(1_000_000.0) for k in "xyz"}
        assert panel._thin(recipes.get_recipe("surface3d"), big) is None


class TestThumbnails:
    def test_without_a_window_nothing_can_be_uploaded(self):
        thumbs = Thumbnails(lambda: None)
        thumbs.begin_frame()
        assert thumbs.available is False
        assert thumbs.get("k", lambda: np.zeros((2, 2, 4), np.uint8)) is None

    def test_a_window_turns_the_cache_on_and_a_new_window_forgets_the_old_ids(self, monkeypatch):
        from glplot.gui.panels import thumbs as module

        deleted = []
        monkeypatch.setattr(module, "gl_upload", lambda rgba: 7)
        monkeypatch.setattr(module, "gl_delete", lambda i: deleted.append(i))
        window = {"w": object()}
        holder = Thumbnails(lambda: window["w"])
        holder.begin_frame()
        assert holder.available
        assert holder.put("a", np.zeros((2, 2, 4), np.uint8)) is not None
        window["w"] = object()  # the window was closed and reopened: a new GL context
        holder.begin_frame()
        assert holder.peek("a") is None
        assert deleted == []  # forgotten, not deleted: those ids died with the old context

    def test_closing_the_window_turns_it_off_again(self, monkeypatch):
        from glplot.gui.panels import thumbs as module

        monkeypatch.setattr(module, "gl_upload", lambda rgba: 1)
        window = {"w": object()}
        holder = Thumbnails(lambda: window["w"])
        holder.begin_frame()
        assert holder.available
        window["w"] = None
        holder.begin_frame()
        assert not holder.available

    @pytest.mark.parametrize(
        "box, aspect, expected",
        [
            ((0, 0, 100, 100), 2.0, (0, 25, 100, 75)),
            ((0, 0, 100, 100), 0.5, (25, 0, 75, 100)),
            ((0, 0, 160, 100), 1.6, (0, 0, 160, 100)),
        ],
    )
    def test_fit_rect_keeps_the_aspect_and_centres(self, box, aspect, expected):
        assert fit_rect(box, aspect) == pytest.approx(expected)
