"""Test the Data panel's plot action: it must UPDATE the bound layer, not append.

No OpenGL context and no window: ``GPULinePlot()`` builds its state without one, the
panel is driven by calling its action methods directly (the imgui draw callback is not
involved), and every queued command is executed by draining the queue by hand -- which
is exactly where the engine drains it, at the top of ``_main_loop``.

The bug under test (SPEC_FIX P0-D, "solo se agregan plots en lugar de editar los
existentes"): pressing Plot on a dataset that was already plotted used to call
``add_xy_layer`` unconditionally, leaving two identically-labelled layers stacked on the
same data with only the first one bound -- so table edits updated one and the other was
stale forever.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.engine import GPULinePlot
from glplot.gui import layerops, notifications
from glplot.gui.datasets import Column, DataSet
from glplot.gui.panels.data_editor import (
    _can_replot_into,
    _read_csv_table,
    _write_csv_table,
)
from glplot.gui.panels.scene import _can_change_kind


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


def _panel():
    """A live plot plus its Data panel, with no window and no GL context."""
    plot = GPULinePlot()
    plot.options.enable_hud = True
    ws = plot.hud.workspace
    assert ws is not None
    return plot, ws, ws.panels["data"]


def _dataset(ws, name="t", n=5):
    """A bound-able x/y dataset registered in the workspace store."""
    ds = DataSet(name, [Column("x", np.arange(float(n))), Column("y", np.arange(float(n)) ** 2)])
    ws.store.add(ds)
    return ds


def _dataset_with_color(ws, name="t", n=8):
    """An x/y dataset plus a third column meant for a "Color by" encoding."""
    ds = DataSet(
        name,
        [
            Column("x", np.arange(float(n))),
            Column("y", np.arange(float(n)) ** 2),
            Column("c", np.linspace(0.0, 1.0, n)),
        ],
    )
    ws.store.add(ds)
    return ds


def _plot_once(ws, panel, ds, kind="line", **kwargs):
    """Press Plot and run the queued command, as the main loop would."""
    panel._plot_x, panel._plot_y, panel._plot_kind = "x", "y", kind
    panel._plot_dataset(ds, **kwargs)
    ws.queue.drain(ws.plot)


class TestPlotUpdatesBoundLayer:
    """Plot re-plots into the dataset's bound layer instead of adding a new one."""

    def test_second_plot_updates_instead_of_appending(self):
        """Plot twice -> exactly one layer, holding the newest data. Fails pre-fix."""
        plot, ws, panel = _panel()
        ds = _dataset(ws)
        panel._select(ds)

        _plot_once(ws, panel, ds)
        assert len(plot.scene.layers) == 1
        first_id = ds.layer_id
        assert first_id is not None

        ds.columns[1].values[0] = 42.0
        _plot_once(ws, panel, ds)

        assert len(plot.scene.layers) == 1
        assert ds.layer_id == first_id
        assert np.allclose(plot.scene.layers[0].pts[0], [0.0, 42.0])

    def test_repeated_plots_never_stack_duplicates(self):
        """Five presses of Plot leave one layer, not five."""
        plot, ws, panel = _panel()
        ds = _dataset(ws)
        panel._select(ds)
        for _ in range(5):
            _plot_once(ws, panel, ds)
        assert len(plot.scene.layers) == 1

    def test_updated_layer_still_follows_cell_edits(self):
        """The layer left standing after a re-plot is the bound one, not an orphan."""
        plot, ws, panel = _panel()
        ds = _dataset(ws)
        panel._select(ds)
        _plot_once(ws, panel, ds)
        _plot_once(ws, panel, ds)

        panel._edit_cell = (0, 1)
        panel._edit_text = "999"
        panel._acted = False
        panel._commit_edit(ds)
        ws.queue.drain(plot)

        assert len(plot.scene.layers) == 1
        assert np.allclose(plot.scene.layers[0].pts[0, 1], 999.0)

    def test_update_marks_gpu_dirty(self):
        """An in-place data swap must force the re-upload, or the plot stays stale."""
        plot, ws, panel = _panel()
        ds = _dataset(ws)
        panel._select(ds)
        _plot_once(ws, panel, ds)
        plot.scene.layers[0].dirty.gpu_dirty = False

        _plot_once(ws, panel, ds)
        assert plot.scene.layers[0].dirty.gpu_dirty is True

    def test_row_count_change_refits_colours(self):
        """A shorter table must not leave the colour VBO longer than the point count."""
        plot, ws, panel = _panel()
        ds = _dataset(ws, n=8)
        panel._select(ds)
        _plot_once(ws, panel, ds, kind="scatter")

        ds.delete_rows([0, 1, 2])
        _plot_once(ws, panel, ds, kind="scatter")

        layer = plot.scene.layers[0]
        assert len(plot.scene.layers) == 1
        assert len(layer.pts) == 5
        assert len(layer.colors) == 5


class TestPlotKindChange:
    """Changing Kind and pressing Plot converts the layer in place."""

    def test_kind_change_converts_the_bound_layer(self):
        """line -> scatter: one layer, of the new type, still bound."""
        plot, ws, panel = _panel()
        ds = _dataset(ws)
        panel._select(ds)
        _plot_once(ws, panel, ds, kind="line")
        assert plot.scene.layers[0].layer_type == "polyline"

        _plot_once(ws, panel, ds, kind="scatter")

        assert len(plot.scene.layers) == 1
        layer = plot.scene.layers[0]
        assert layer.layer_type == "scatter"
        assert ds.layer_id == layer.layer_id

    def test_kind_change_preserves_style_and_position(self):
        """Style, list index and selection survive the delete+recreate."""
        plot, ws, panel = _panel()
        ds = _dataset(ws)
        panel._select(ds)
        _plot_once(ws, panel, ds, kind="line")

        ws.queue.submit(
            lambda: layerops.add_xy_layer(plot, [0.0, 1.0], [0.0, 1.0], kind="line", label="other")
        )
        ws.queue.drain(plot)
        layer = layerops.find_layer(plot, ds.layer_id)
        layer.style.zorder = 7
        layer.style.alpha = 0.25
        layer.style.visible = False
        plot.interaction.selected_layer_id = layer.layer_id
        index = plot.scene.layers.index(layer)

        _plot_once(ws, panel, ds, kind="scatter")

        new = layerops.find_layer(plot, ds.layer_id)
        assert len(plot.scene.layers) == 2
        assert plot.scene.layers[index] is new
        assert new.style.zorder == 7
        assert np.isclose(new.style.alpha, 0.25)
        assert new.style.visible is False
        assert plot.interaction.selected_layer_id == new.layer_id

    def test_kind_change_is_undoable(self):
        """Undo restores the previous kind, the previous data and the binding."""
        plot, ws, panel = _panel()
        ds = _dataset(ws)
        panel._select(ds)
        _plot_once(ws, panel, ds, kind="line")
        before = plot.scene.layers[0].pts.copy()

        _plot_once(ws, panel, ds, kind="scatter")
        ws.undo.undo()
        ws.queue.drain(plot)

        assert len(plot.scene.layers) == 1
        layer = plot.scene.layers[0]
        assert layer.layer_type == "polyline"
        assert np.allclose(layer.pts, before)
        assert ds.layer_id == layer.layer_id


class TestColorEncodingSurvivesPanelKindChange:
    """The "Color by" column must survive a Kind change and re-Plot, not just the first
    Plot press -- and the panel must be able to say which column it is, so the Data
    panel reflects what the scene is actually showing rather than only what the "Color
    by" combo happens to have selected at that moment.
    """

    def test_kind_change_keeps_the_colour_by_column(self):
        plot, ws, panel = _panel()
        ds = _dataset_with_color(ws)
        panel._select(ds)
        panel._plot_x, panel._plot_y, panel._plot_kind = "x", "y", "scatter"
        panel._plot_c = "c"
        panel._plot_dataset(ds)
        ws.queue.drain(plot)

        layer = plot.scene.layers[0]
        assert layer.metadata.get("cvalues") is not None
        assert np.asarray(layer.metadata["cvalues"]) == pytest.approx(ds.get("c"), abs=1e-3)

        panel._plot_kind = "bar"
        panel._plot_dataset(ds)
        ws.queue.drain(plot)

        layer = plot.scene.layers[0]
        assert layer.layer_type != "scatter"
        assert layer.metadata.get("cvalues") is not None
        assert np.asarray(layer.metadata["cvalues"]) == pytest.approx(ds.get("c"), abs=1e-3)
        # A real per-bar mapping, not every bar painted the same flat colour.
        assert len(np.unique(layer.colors, axis=0)) > 1

    def test_status_line_reports_the_colour_column(self):
        plot, ws, panel = _panel()
        ds = _dataset_with_color(ws)
        panel._select(ds)
        panel._plot_x, panel._plot_y, panel._plot_kind = "x", "y", "scatter"
        panel._plot_c = "c"
        panel._plot_dataset(ds)
        ws.queue.drain(plot)

        layer = plot.scene.layers[0]
        assert panel._layer_color_column(ds, layer) == "c"

    def test_no_colour_encoding_reports_no_column(self):
        plot, ws, panel = _panel()
        ds = _dataset_with_color(ws)
        panel._select(ds)
        _plot_once(ws, panel, ds, kind="scatter")

        layer = plot.scene.layers[0]
        assert panel._layer_color_column(ds, layer) is None


class TestReplotRefusesToStrandALayer:
    """The Data panel must not offer a re-plot that nothing can undo.

    Re-plotting *into* a bound layer is a delete+recreate, and the only way back is to
    pick the layer's original kind out of the picker. A layer with no kind has no such
    entry -- so the button was a one-way door out of an ``imshow``, a guide or a 3-D
    layer, and the only notice was a status line *after* the fact, naming the layer_type
    ("scatter is not undoable") rather than the picture it had just thrown away.

    The Scene panel has always greyed its own type menu out on exactly this test
    (``_can_change_kind``); the Data panel reaching a different verdict about the same
    layer is the bug. ``_update_plot_layer`` already computes ``undoable = ... and
    old_kind is not None`` -- it simply does it one step too late to help.
    """

    def _pyplot_panel(self):
        """A Data panel on the *pyplot* figure, so ``gplt.*`` calls land in its scene.

        `_panel()`'s bare ``GPULinePlot()`` is invisible to pyplot, and these layers can
        only be built through the pyplot API -- there is no engine call that makes an
        ``imshow``.
        """
        plot = gplt.figure("stranding probe", hud=True)
        ws = plot.hud.workspace
        assert ws is not None
        return plot, ws, ws.panels["data"]

    def _kindless_bound(self, make_layer):
        """A dataset bound to a layer that has no kind -- the stranding setup."""
        plot, ws, panel = self._pyplot_panel()
        make_layer(plot)
        layer = plot.scene.layers[-1]
        assert layerops.layer_kind(layer) is None, "fixture no longer reproduces the setup"
        ds = DataSet.from_layer(layer)
        assert ds is not None, "the Data panel cannot even reach this layer; no door to shut"
        ws.store.add(ds)
        panel._select(ds)
        return plot, ws, panel, ds, layer

    @pytest.mark.parametrize(
        "name, make_layer",
        [
            # A field: the engine draws it as a scatter of one point per cell, which is
            # exactly why `from_layer` binds it and `layer_type` cannot be trusted. Note
            # `hist2d` is *not* here despite being a field too -- it is a `patch` now
            # (a real filled mesh, not a scatter of points; see `TestHist2dParity`), and
            # `DataSet.from_layer` refuses every patch outright (no editable tabular
            # form -- `test_patch_layer_has_no_tabular_form` in test_gui_datasets.py), so
            # it never reaches this "bound but must still refuse" scenario at all.
            ("imshow", lambda p: gplt.imshow(np.random.default_rng(0).random((8, 8)))),
            # A guide spans the view rather than passing through data.
            ("axhline", lambda p: gplt.axhline(0.5)),
            # 3-D: bindable (its vertices are x/y/z columns) and no 2-D kind rebuilds it.
            ("scatter3d", lambda p: gplt.scatter3d(np.zeros(4), np.zeros(4), np.zeros(4))),
        ],
    )
    def test_the_button_is_refused_on_a_kindless_layer(self, name, make_layer):
        plot, ws, panel, ds, layer = self._kindless_bound(make_layer)
        assert _can_replot_into(layer) is False, f"{name} would still be replotted away"

    def test_hist2d_cannot_even_be_bound_as_a_dataset(self):
        """Confirms the real, integration-level fact ``_kindless_bound``'s assertion
        would otherwise mask: ``hist2d`` moved from "bindable field, must still refuse
        replot" to "not bindable at all" the moment it became a ``patch`` (a real filled
        mesh) rather than a ``scatter`` of one sized point per cell. Same rule every
        patch already follows (``test_patch_layer_has_no_tabular_form`` in
        test_gui_datasets.py), checked here against the real artist, not a synthetic
        ``PatchLayer``, so a future change to ``hist2d``'s geometry cannot silently
        make it bindable again without this catching it.
        """
        gplt.figure("hist2d binding probe")
        gplt.hist2d(np.linspace(0.0, 1.0, 40), np.linspace(0.0, 1.0, 40))
        layer = gplt.gcf().scene.layers[-1]
        assert layer.layer_type == "patch"
        assert DataSet.from_layer(layer) is None

    @pytest.mark.parametrize(
        "name, make_layer",
        [
            ("imshow", lambda p: gplt.imshow(np.random.default_rng(0).random((8, 8)))),
            ("axhline", lambda p: gplt.axhline(0.5)),
        ],
    )
    def test_the_two_panels_agree_about_the_same_layer(self, name, make_layer):
        """One rule, asserted through both panels' own gates rather than re-stated."""
        plot, ws, panel, ds, layer = self._kindless_bound(make_layer)
        assert _can_replot_into(layer) == _can_change_kind(layer)

    def test_plot_as_new_layer_is_still_available(self):
        """The refusal must not cost the user the ability to see their columns."""
        plot, ws, panel, ds, layer = self._kindless_bound(
            lambda p: gplt.imshow(np.random.default_rng(0).random((8, 8)))
        )
        before = len(plot.scene.layers)
        _plot_once(ws, panel, ds, kind="scatter", as_new=True)
        assert len(plot.scene.layers) == before + 1
        assert plot.scene.layers[before] is not layer, "the field was replaced, not added to"

    @pytest.mark.parametrize(
        "name, make_layer",
        [
            ("plot", lambda p: gplt.plot([0.0, 1.0], [0.0, 1.0])),
            ("scatter", lambda p: gplt.scatter([0.0, 1.0], [0.0, 1.0])),
            (
                "plot_lines",
                lambda p: gplt.plot_lines(
                    np.array([0.5, 1.0], dtype=np.float32),
                    np.array([0.0, 1.0], dtype=np.float32),
                    x_range=(-3.0, 3.0),
                ),
            ),
        ],
    )
    def test_a_layer_with_a_kind_is_still_replottable(self, name, make_layer):
        """The gate must not overshoot: the normal path is the whole point of the panel."""
        plot, ws, panel = self._pyplot_panel()
        make_layer(plot)
        layer = plot.scene.layers[-1]
        assert layerops.layer_kind(layer) is not None, "fixture no longer discriminates"
        assert _can_replot_into(layer) is True

    def test_a_layer_the_panel_itself_built_stays_replottable(self):
        """Every kind the Data panel can create must remain updatable in place."""
        plot, ws, panel = _panel()
        ds = _dataset(ws)
        panel._select(ds)
        for kind in layerops.KIND_KEYS:
            _plot_once(ws, panel, ds, kind=kind)
            bound = layerops.find_layer(plot, ds.layer_id)
            assert bound is not None, f"{kind}: the panel lost its own layer"
            assert _can_replot_into(bound) is True, f"{kind}: stranded by its own builder"


class TestPlotAsNewLayer:
    """The explicit affordance for genuinely wanting a second layer."""

    def test_plot_as_new_layer_adds_an_unlinked_copy(self):
        """A second layer appears, with a distinct label, and the binding does not move."""
        plot, ws, panel = _panel()
        ds = _dataset(ws)
        panel._select(ds)
        _plot_once(ws, panel, ds)
        bound = ds.layer_id

        _plot_once(ws, panel, ds, as_new=True)

        assert len(plot.scene.layers) == 2
        assert ds.layer_id == bound
        labels = [layer.label for layer in plot.scene.layers]
        assert len(set(labels)) == 2, f"copies must be distinguishable in Scene: {labels}"

    def test_plot_as_new_layer_is_undoable(self):
        """Undo removes the copy and leaves the bound layer alone."""
        plot, ws, panel = _panel()
        ds = _dataset(ws)
        panel._select(ds)
        _plot_once(ws, panel, ds)
        bound = ds.layer_id

        _plot_once(ws, panel, ds, as_new=True)
        ws.undo.undo()
        ws.queue.drain(plot)

        assert len(plot.scene.layers) == 1
        assert ds.layer_id == bound


class TestStaleBinding:
    """The bound layer can be deleted from the Scene panel under the dataset's feet."""

    def test_plot_after_the_bound_layer_was_deleted_rebinds(self):
        """Plot creates a fresh layer and re-binds, rather than resurrecting an id."""
        plot, ws, panel = _panel()
        ds = _dataset(ws)
        panel._select(ds)
        _plot_once(ws, panel, ds)
        dead = layerops.find_layer(plot, ds.layer_id)

        ws.queue.submit(lambda: layerops.remove_layer(plot, plot.hud, dead))
        ws.queue.drain(plot)
        assert layerops.find_layer(plot, ds.layer_id) is None

        _plot_once(ws, panel, ds)

        assert len(plot.scene.layers) == 1
        assert layerops.find_layer(plot, ds.layer_id) is plot.scene.layers[0]

    def test_bound_layer_lookup_is_none_when_deleted(self):
        """_bound_layer answers honestly, which is what the button label reads."""
        plot, ws, panel = _panel()
        ds = _dataset(ws)
        panel._select(ds)
        _plot_once(ws, panel, ds)
        assert panel._bound_layer(ds) is not None

        layer = layerops.find_layer(plot, ds.layer_id)
        ws.queue.submit(lambda: layerops.remove_layer(plot, plot.hud, layer))
        ws.queue.drain(plot)
        assert panel._bound_layer(ds) is None


class TestLayerKindIdentity:
    """layerops' kind bookkeeping, the thing that decides update vs rebuild."""

    def test_kind_tag_round_trips(self):
        """add_xy_layer's kind is recoverable from the layer it produced."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, [0.0, 1.0], [0.0, 1.0], kind="line", label="l")
        layerops.tag_layer_kind(layer, "line")
        assert layerops.layer_kind(layer) == "line"
        assert layerops.layer_matches_kind(layer, "line")
        assert not layerops.layer_matches_kind(layer, "scatter")

    def test_untagged_layer_resolves_by_type(self):
        """A layer from the plain engine API is still recognised as line/scatter."""
        plot = GPULinePlot()
        plot.add_scatter(np.zeros(3), np.zeros(3), np.ones((3, 4), dtype=np.float32), label="s")
        assert layerops.layer_kind(plot.scene.layers[-1]) == "scatter"

    def test_unknown_type_has_no_kind(self):
        """A type no kind produces resolves to None -- callers must rebuild, not update."""
        plot = GPULinePlot()
        plot.add_patch(
            np.zeros((3, 2), dtype=np.float32),
            np.arange(3, dtype=np.uint32),
            mode="triangles",
            label="p",
        )
        assert layerops.layer_kind(plot.scene.layers[-1]) is None

    def test_layer_xy_returns_copies(self):
        """The undo snapshot must not alias the live array."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, [0.0, 1.0], [2.0, 3.0], kind="line", label="l")
        x, y = layerops.layer_xy(layer)
        layer.pts[0, 0] = 99.0
        assert np.allclose(x, [0.0, 1.0])
        assert np.allclose(y, [2.0, 3.0])


class TestFilterIsAView:
    """The row filter hides rows; it does not delete them (SPEC_FIX P1-I)."""

    def test_filter_hides_rows_without_touching_the_data(self):
        """Filtering leaves every row in the table and only shrinks the view."""
        plot, ws, panel = _panel()
        ds = _dataset(ws, n=6)
        panel._select(ds)
        panel._filter_text = "y > 4"
        panel._apply_filter(ds)
        assert ds.n_rows() == 6
        assert panel._view_rows_count(ds) == 3
        assert panel._view_rows.tolist() == [3, 4, 5]

    def test_view_rows_map_to_data_rows(self):
        """A view row reads the cell of the data row it stands for."""
        plot, ws, panel = _panel()
        ds = _dataset(ws, n=6)
        panel._select(ds)
        panel._filter_text = "y > 4"
        panel._apply_filter(ds)
        assert panel._data_row(0) == 3
        assert ds.get_cell(panel._data_row(0), 1) == 9.0

    def test_a_bad_predicate_reports_and_shows_everything(self):
        """A predicate that does not evaluate is an error, not an empty table."""
        plot, ws, panel = _panel()
        ds = _dataset(ws)
        panel._select(ds)
        panel._filter_text = "y > "
        panel._apply_filter(ds)
        assert panel._error
        assert panel._view_rows is None

    def test_structural_edits_are_refused_while_filtered(self):
        """Paste/insert/sort speak contiguous rows; the visible ones are not.

        The alternative was to silently act on rows the user cannot see.
        """
        plot, ws, panel = _panel()
        ds = _dataset(ws, n=6)
        panel._select(ds)
        panel._filter_text = "y > 4"
        panel._apply_filter(ds)
        for action in (panel._insert_row, panel._delete_rows):
            panel._error = ""
            panel._acted = False
            action(ds)
            assert "filter" in panel._error
        assert ds.n_rows() == 6

    def test_a_row_count_change_drops_the_filter(self):
        """The mask is row indices; a table that gains or loses rows invalidates it."""
        plot, ws, panel = _panel()
        ds = _dataset(ws, n=6)
        panel._select(ds)
        panel._filter_text = "y > 4"
        panel._apply_filter(ds)
        ds.delete_row(0)
        panel._validate_filter(ds)
        assert panel._view_rows is None

    def test_apply_to_data_deletes_the_hidden_rows_undoably(self):
        """'Apply to data' is the destructive half, and it is undoable."""
        plot, ws, panel = _panel()
        ds = _dataset(ws, n=6)
        panel._select(ds)
        panel._filter_text = "y > 4"
        panel._apply_filter(ds)
        panel._acted = False
        panel._apply_filter_to_data(ds)
        ws.queue.drain(plot)
        assert ds.n_rows() == 3
        assert np.allclose(ds.get("x"), [3.0, 4.0, 5.0])
        ws.undo.undo()
        ws.queue.drain(plot)
        assert ds.n_rows() == 6

    def test_filter_to_new_dataset_leaves_the_bound_one_alone(self):
        """The safe path for a plotted dataset: derive, do not destroy."""
        plot, ws, panel = _panel()
        ds = _dataset(ws, n=6)
        panel._select(ds)
        _plot_once(ws, panel, ds)
        panel._filter_text = "y > 4"
        panel._apply_filter(ds)
        panel._filter_to_new_dataset(ds)
        ws.queue.drain(plot)
        assert ds.n_rows() == 6
        derived = ws.store.get("t filtered")
        assert derived is not None and derived.n_rows() == 3
        assert derived.layer_id is None


class TestTransformColumn:
    """Transforming a column with an expression, undoably."""

    def test_transform_in_place_is_undoable(self):
        """y = y*2 rewrites the column and Ctrl+Z puts it back."""
        plot, ws, panel = _panel()
        ds = _dataset(ws)
        panel._select(ds)
        before = ds.get("y").copy()
        panel._acted = False
        panel._transform(ds, "y", "y * 2", None)
        ws.queue.drain(plot)
        assert np.allclose(ds.get("y"), before * 2)
        ws.undo.undo()
        ws.queue.drain(plot)
        assert np.allclose(ds.get("y"), before)

    def test_transform_updates_the_bound_layer_live(self):
        """A transform reaches the plot through the same live link a cell edit uses."""
        plot, ws, panel = _panel()
        ds = _dataset(ws)
        panel._select(ds)
        _plot_once(ws, panel, ds)
        layer = layerops.find_layer(plot, ds.layer_id)
        panel._acted = False
        panel._transform(ds, "y", "y * 2", None)
        ws.queue.drain(plot)
        assert np.allclose(layer.pts[:, 1], ds.get("y"))

    def test_a_bad_expression_reports_and_changes_nothing(self):
        """The error surfaces in the panel, not inside a queued command with no audience."""
        plot, ws, panel = _panel()
        ds = _dataset(ws)
        panel._select(ds)
        before = ds.get("y").copy()
        panel._acted = False
        panel._transform(ds, "y", "y + nope", None)
        ws.queue.drain(plot)
        assert panel._error
        assert np.allclose(ds.get("y"), before)


class TestMoveColumnsAndRows:
    """Reordering from the panel is undoable and resolved by name."""

    def test_move_column_is_undoable(self):
        """Dragging a header reorders the columns and Ctrl+Z restores the order."""
        plot, ws, panel = _panel()
        ds = _dataset(ws)
        panel._select(ds)
        panel._acted = False
        panel._move_column(ds, 0, 1)
        ws.queue.drain(plot)
        assert ds.column_names() == ["y", "x"]
        ws.undo.undo()
        ws.queue.drain(plot)
        assert ds.column_names() == ["x", "y"]

    def test_move_rows_moves_the_whole_selection(self):
        """Dragging a selected row moves every selected row, like a file manager."""
        plot, ws, panel = _panel()
        ds = _dataset(ws, n=5)
        panel._select(ds)
        panel._anchor, panel._cursor = (0, 0), (1, 1)
        panel._acted = False
        panel._move_rows(ds, 0, 4)
        ws.queue.drain(plot)
        assert np.allclose(ds.get("x"), [2.0, 3.0, 0.0, 1.0, 4.0])

    def test_move_rows_moves_only_the_dragged_row_when_unselected(self):
        """Dragging a row outside the selection moves just that row."""
        plot, ws, panel = _panel()
        ds = _dataset(ws, n=5)
        panel._select(ds)
        panel._anchor = panel._cursor = (0, 0)
        panel._acted = False
        panel._move_rows(ds, 3, 0)
        ws.queue.drain(plot)
        assert np.allclose(ds.get("x"), [3.0, 0.0, 1.0, 2.0, 4.0])


class TestPlotTypes:
    """Every kind the picker offers is reachable, bound and updatable (SPEC_FIX P1-I)."""

    @pytest.mark.parametrize("kind", list(layerops.KIND_KEYS))
    def test_each_kind_plots_one_bound_layer(self, kind):
        """One dataset, one layer, whatever the type."""
        plot, ws, panel = _panel()
        ds = _dataset(ws, n=6)
        panel._select(ds)
        panel._set_plot_kind(kind)
        _plot_once(ws, panel, ds, kind=kind)
        assert len(plot.scene.layers) == 1
        layer = layerops.find_layer(plot, ds.layer_id)
        assert layer is not None
        assert layerops.layer_kind(layer) == kind
        assert layer.layer_type == layerops.kind_spec(kind).layer_type

    @pytest.mark.parametrize("kind", list(layerops.KIND_KEYS))
    def test_replot_never_appends(self, kind):
        """The 10c bug, for every new type: re-plotting updates the bound layer."""
        plot, ws, panel = _panel()
        ds = _dataset(ws, n=6)
        panel._select(ds)
        panel._set_plot_kind(kind)
        _plot_once(ws, panel, ds, kind=kind)
        _plot_once(ws, panel, ds, kind=kind)
        assert len(plot.scene.layers) == 1

    @pytest.mark.parametrize("kind", list(layerops.KIND_KEYS))
    def test_each_kind_follows_a_cell_edit(self, kind):
        """A derived layer re-derives its geometry from the table, live.

        Two shapes of assertion, because the kinds genuinely differ. Most plot the value,
        so the edited 99.0 has to appear in the geometry. The rest *summarise* the column
        -- ``hist``/``hist2d`` bin it into counts and cell centres, ``boxplot`` reduces it
        to quartiles (where 99.0 lands outside the whiskers and is not drawn at all) --
        so 99.0 will never appear in theirs. For those the property is that the geometry
        moved at all, which is still the whole question ("did the edit reach the layer?")
        and is stronger than the blanket pass the old ``or kind == "hist"`` handed out.
        """

        def geometry_of(layer):
            """The layer's data array, whatever attribute its type keeps it in.

            Routed through ``layer_xy`` rather than a ``pts``-or-``vertices`` guess: the
            attribute is per layer_type (a line_family's columns live in ``ab``), and
            ``layerops`` already owns that mapping. Guessing here is what made this test
            raise ``AttributeError`` the moment a kind with a third attribute existed.
            """
            xy = layerops.layer_xy(layer)
            if xy is not None:
                return np.column_stack(xy)
            return np.array(layer.vertices, copy=True)

        plot, ws, panel = _panel()
        ds = _dataset(ws, n=6)
        panel._select(ds)
        panel._set_plot_kind(kind)
        _plot_once(ws, panel, ds, kind=kind)
        layer = layerops.find_layer(plot, ds.layer_id)

        before = geometry_of(layer)
        ds.set_cell(0, 1, 99.0)
        ws.queue.submit(lambda: panel._sync_layer_now(ds))
        ws.queue.drain(plot)
        after = geometry_of(layer)

        assert layer.dirty.gpu_dirty is True
        if kind in ("hist", "hist2d", "boxplot"):
            assert after.shape != before.shape or not np.array_equal(after, before)
        else:
            assert float(np.nanmax(after[:, 1])) >= 99.0

    def test_bar_geometry_is_one_layer_of_quads(self):
        """pyplot.bar spends one patch per bar; a bound dataset can only own one layer."""
        plot, ws, panel = _panel()
        ds = _dataset(ws, n=6)
        panel._select(ds)
        panel._set_plot_kind("bar")
        _plot_once(ws, panel, ds, kind="bar")
        layer = layerops.find_layer(plot, ds.layer_id)
        assert layer.vertices.shape == (24, 2)
        assert layer.indices.shape == (36,)
        assert layer.indices.dtype == np.uint32
        assert layer.vertices.dtype == np.float32
        assert layer.mode == "triangles"

    def test_derived_bindings_do_not_write_back(self):
        """A bar's binding must never column-stack the table over its corners."""
        plot, ws, panel = _panel()
        ds = _dataset(ws, n=6)
        panel._select(ds)
        panel._set_plot_kind("bar")
        _plot_once(ws, panel, ds, kind="bar")
        assert ds.binding.derived is True
        assert ds.binding.kind == "bar"
        layer = layerops.find_layer(plot, ds.layer_id)
        assert ds.write_back(layer) is False

    def test_kind_options_reach_the_geometry(self):
        """The bins spinner is real: it changes how many bars a histogram has."""
        plot, ws, panel = _panel()
        ds = _dataset(ws, n=20)
        panel._select(ds)
        panel._set_plot_kind("hist")
        panel._plot_opts["bins"] = 4
        _plot_once(ws, panel, ds, kind="hist")
        layer = layerops.find_layer(plot, ds.layer_id)
        assert layer.vertices.shape == (16, 2)

    def test_switching_kind_resets_its_options(self):
        """'bins' means nothing to a bar chart, and layerops rejects it rather than ignore it."""
        plot, ws, panel = _panel()
        panel._set_plot_kind("hist")
        panel._plot_opts["bins"] = 7
        panel._set_plot_kind("bar")
        assert "bins" not in panel._current_options()
        assert "bar_width" in panel._current_options()


class TestChangeLayerType:
    """set_layer_kind: the shared type-change (SPEC_FIX user decision 2)."""

    def _bar(self):
        plot, ws, panel = _panel()
        ds = _dataset(ws, n=5)
        panel._select(ds)
        panel._set_plot_kind("bar")
        _plot_once(ws, panel, ds, kind="bar")
        return plot, ws, panel, ds, layerops.find_layer(plot, ds.layer_id)

    def test_type_change_preserves_style_position_and_selection(self):
        """The user's decision, tested where the logic lives rather than per panel."""
        plot, ws, panel, ds, bar = self._bar()
        other = layerops.add_xy_layer(plot, [0.0, 1.0], [0.0, 1.0], kind="line", label="other")
        layerops.move_layer(plot, layerops._index_of(plot.scene.layers, other), 0)
        index = layerops._index_of(plot.scene.layers, bar)
        bar.style.zorder = 9
        bar.style.alpha = 0.25
        plot.interaction.selected_layer_id = bar.layer_id

        ws.queue.submit(
            lambda: layerops.set_layer_kind(plot, plot.hud, bar, "scatter", store=ws.store)
        )
        ws.queue.drain(plot)

        new = layerops.find_layer(plot, ds.layer_id)
        assert new.layer_type == "scatter"
        assert new.style.zorder == 9
        assert new.style.alpha == 0.25
        assert layerops._index_of(plot.scene.layers, new) == index
        assert plot.interaction.selected_layer_id == new.layer_id
        assert len(plot.scene.layers) == 2

    def test_type_change_repoints_the_dataset(self):
        """Miss this and the next Plot adds a second layer -- bug 10c, re-armed."""
        plot, ws, panel, ds, bar = self._bar()
        old_id = ds.layer_id
        ws.queue.submit(
            lambda: layerops.set_layer_kind(plot, plot.hud, bar, "line", store=ws.store)
        )
        ws.queue.drain(plot)
        assert ds.layer_id != old_id
        assert layerops.find_layer(plot, ds.layer_id) is not None
        assert ds.binding.kind == "line"
        assert ds.binding.derived is False
        _plot_once(ws, panel, ds, kind="line")
        assert len(plot.scene.layers) == 1

    def test_type_change_carries_the_source_data_not_the_geometry(self):
        """A bar's vertices are its corners; converting must plot the columns."""
        plot, ws, panel, ds, bar = self._bar()
        ws.queue.submit(
            lambda: layerops.set_layer_kind(plot, plot.hud, bar, "line", store=ws.store)
        )
        ws.queue.drain(plot)
        new = layerops.find_layer(plot, ds.layer_id)
        assert np.allclose(new.pts[:, 0], ds.get("x"))
        assert np.allclose(new.pts[:, 1], ds.get("y"))

    def test_type_change_refuses_a_layer_it_cannot_read(self):
        """A patch nobody tagged has no source columns; guessing would draw scaffolding."""
        plot, ws, panel = _panel()
        plot.add_patch(
            np.zeros((4, 2), dtype=np.float32),
            None,
            mode="strip",
            face_color=(1, 0, 0, 1),
            label="raw",
        )
        raw = plot.scene.layers[-1]
        assert layerops.set_layer_kind(plot, plot.hud, raw, "line") is None
        assert len(plot.scene.layers) == 1

    def test_same_kind_is_a_no_op(self):
        """Setting the type it already has must not delete and recreate it."""
        plot, ws, panel, ds, bar = self._bar()
        assert layerops.set_layer_kind(plot, plot.hud, bar, "bar") is bar
        assert layerops.find_layer(plot, ds.layer_id) is bar


class TestKindRegistry:
    """The registry is the picker's source of truth."""

    def test_unsupported_kinds_are_reported_not_offered(self):
        """The types a table cannot reach are documented, and none of them is a kind."""
        assert layerops.UNSUPPORTED_KINDS
        for name in layerops.UNSUPPORTED_KINDS:
            assert name not in layerops.KIND_KEYS
        assert "errorbar" in layerops.UNSUPPORTED_KINDS

    def test_unknown_kind_raises(self):
        """A kind the registry does not know is an error, not a silent line plot."""
        with pytest.raises(ValueError):
            layerops.kind_spec("pie")

    def test_unknown_option_raises(self):
        """A silently dropped option is a bug the user debugs by staring at the plot."""
        plot = GPULinePlot()
        with pytest.raises(ValueError):
            layerops.add_xy_layer(
                plot, [0.0, 1.0], [0.0, 1.0], kind="bar", label="b", options={"bins": 5}
            )

    def test_update_layer_xy_refuses_a_derived_layer(self):
        """Writing raw x/y over a step's staircase would draw a line calling itself a step."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(
            plot, [0.0, 1.0, 2.0], [0.0, 1.0, 4.0], kind="step", label="s"
        )
        with pytest.raises(ValueError):
            layerops.update_layer_xy(plot, layer, [0.0, 1.0, 2.0], [1.0, 2.0, 3.0])

    def test_auto_bar_width_survives_degenerate_x(self):
        """A zero-width bar is invisible and reads as a bug, so never return 0."""
        assert layerops.auto_bar_width(np.array([1.0])) == 1.0
        assert layerops.auto_bar_width(np.array([2.0, 2.0])) == 1.0
        assert layerops.auto_bar_width(np.array([np.nan, np.inf])) == 1.0
        assert layerops.auto_bar_width(np.array([0.0, 1.0, 2.0])) == pytest.approx(0.8)

    def test_hist_of_an_all_nan_column_draws_nothing(self):
        """np.histogram cannot bin nan; empty geometry is 'draw nothing', not a raise."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, [0.0, 1.0], [np.nan, np.nan], kind="hist", label="h")
        assert len(layer.vertices) == 0
        assert layer.get_intrinsic_bounds() is None


# ----------------------------------------------------------------------------------
# CSV import/export: read/write correctness, then background dispatch/poll/cancel.
# Backgrounded so an arbitrarily large file does not freeze the interface (see
# glplot/gui/panels/data_editor.py's _import_csv/_export_csv docstrings and
# glplot/gui/background.py). This was an untested gap before -- no prior test in this
# file exercised _import_csv/_export_csv at all.
# ----------------------------------------------------------------------------------


def _wait_until(predicate, *, timeout=2.0, interval=0.005):
    """Poll ``predicate`` until it's truthy or ``timeout`` seconds have elapsed --
    robust against scheduler jitter, matching test_gui_mathlab.py's helper of the same
    shape. Unlike that one, no _draw_frame is driven here: _import_csv/_export_csv/
    _poll_csv_job are plain methods, not imgui draw code (see this file's own
    established convention of calling panel action methods directly)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s")


class TestReadWriteCsvTable:
    """_read_csv_table/_write_csv_table: pure, module-level, safe to background."""

    def test_round_trips_a_table(self, tmp_path):
        path = str(tmp_path / "out.csv")
        headers = ["x", "y"]
        array = np.array([[0.0, 1.0], [2.0, 4.0], [3.0, 9.0]])
        _write_csv_table(path, headers, array)
        read_headers, read_array = _read_csv_table(path)
        assert read_headers == headers
        assert np.allclose(read_array, array)

    def test_missing_file_raises_a_could_not_read_message(self, tmp_path):
        path = str(tmp_path / "does_not_exist.csv")
        with pytest.raises(RuntimeError, match="Could not read"):
            _read_csv_table(path)

    def test_unwritable_directory_raises_a_could_not_write_message(self, tmp_path):
        path = str(tmp_path / "no_such_dir" / "out.csv")
        with pytest.raises(RuntimeError, match="Could not write"):
            _write_csv_table(path, ["x"], np.array([[1.0]]))

    def test_malformed_content_raises_with_the_path_named(self, tmp_path):
        """An empty file has no header/body at all -- parse_table's own ValueError,
        surfaced with the offending path named, not just its bare message."""
        path = str(tmp_path / "empty.csv")
        with open(path, "w") as fh:
            fh.write("")
        with pytest.raises(RuntimeError, match=r"empty\.csv"):
            _read_csv_table(path)


class TestCsvImportExportSync:
    """panel._import_csv()/_export_csv(): the synchronous path, unchanged behavior."""

    def _sync_panel(self):
        plot, ws, panel = _panel()
        plot._is_test_mode = True  # deterministic: same-call-returns-final-state
        return plot, ws, panel

    def test_import_creates_a_dataset_from_the_path_field(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("x,y\n1,2\n3,4\n")
        _plot, ws, panel = self._sync_panel()
        panel._csv_path = str(path)

        panel._import_csv()

        assert "data" in ws.store.names()
        ds = ws.store.get("data")
        assert ds.n_rows() == 2
        assert ds.n_cols() == 2
        assert panel._error == ""
        assert "Imported" in panel._status

    def test_import_of_a_missing_file_sets_error_not_a_dataset(self, tmp_path):
        _plot, ws, panel = self._sync_panel()
        panel._csv_path = str(tmp_path / "nope.csv")
        before = set(ws.store.names())

        panel._import_csv()

        assert set(ws.store.names()) == before
        assert "Could not read" in panel._error

    def test_export_writes_the_current_dataset(self, tmp_path):
        path = tmp_path / "out.csv"
        _plot, ws, panel = self._sync_panel()
        ds = _dataset(ws, name="e")
        panel._csv_path = str(path)

        panel._export_csv(ds)

        assert path.exists()
        headers, array = _read_csv_table(str(path))
        assert headers == ["x", "y"]
        assert array.shape[0] == ds.n_rows()
        assert "Exported" in panel._status

    def test_export_to_an_unwritable_path_sets_error(self, tmp_path):
        _plot, ws, panel = self._sync_panel()
        ds = _dataset(ws, name="e2")
        panel._csv_path = str(tmp_path / "missing_dir" / "out.csv")

        panel._export_csv(ds)

        assert "Could not write" in panel._error


class TestCsvAsyncEnabled:
    """_async_enabled(): identical contract to MathLabPanel's/PipelinePanel's."""

    def test_default_is_asynchronous_under_a_real_plot(self):
        """Unlike Math Lab/Pipeline's _FakePlot-based tests, this file's own
        convention is a REAL GPULinePlot -- confirms _async_enabled works against the
        real engine class, not just a mock that happens to lack the attribute."""
        _plot, _ws, panel = _panel()
        assert panel._async_enabled() is True

    def test_is_test_mode_true_is_synchronous(self):
        plot, _ws, panel = _panel()
        plot._is_test_mode = True
        assert panel._async_enabled() is False

    def test_force_async_overrides_is_test_mode(self):
        plot, _ws, panel = _panel()
        plot._is_test_mode = True
        panel._force_async = True
        assert panel._async_enabled() is True


class TestCsvAsyncCompute:
    """Background CSV import/export: dispatch/poll/settle/cancel/retry/notify.

    _import_csv/_export_csv/_poll_csv_job are plain methods (imgui only enters through
    _draw_io_section's buttons/busy-row, tested separately below), so these are driven
    by calling them directly -- matching this file's own established convention -- and
    waited on via _wait_until's poll-until-condition loop, never a fixed sleep.
    """

    def _async_panel(self):
        plot, ws, panel = _panel()
        plot._is_test_mode = True  # belt-and-suspenders; _force_async is authoritative
        panel._force_async = True
        return plot, ws, panel

    def test_import_is_pending_not_blocking_then_settles(self, tmp_path, monkeypatch):
        path = tmp_path / "data.csv"
        path.write_text("x,y\n1,2\n3,4\n")
        _plot, ws, panel = self._async_panel()
        panel._csv_path = str(path)
        monkeypatch.setattr(
            "glplot.gui.panels.data_editor._read_csv_table",
            lambda p: (time.sleep(0.05), _read_csv_table(p))[1],
        )

        start = time.monotonic()
        panel._import_csv()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1, "dispatching a background import must not block"
        assert panel._csv_job is not None
        assert "data" not in ws.store.names()

        _wait_until(lambda: (panel._poll_csv_job(), "data" in ws.store.names())[1])
        assert panel._csv_job is None
        ds = ws.store.get("data")
        assert ds.n_rows() == 2

    def test_second_import_click_while_pending_is_ignored(self, tmp_path, monkeypatch):
        path = tmp_path / "big.csv"
        path.write_text("x,y\n1,2\n")
        _plot, ws, panel = self._async_panel()
        panel._csv_path = str(path)
        monkeypatch.setattr(
            "glplot.gui.panels.data_editor._read_csv_table",
            lambda p: (time.sleep(0.2), _read_csv_table(p))[1],
        )
        panel._import_csv()
        job = panel._csv_job
        panel._import_csv()  # must not replace the in-flight job
        assert panel._csv_job is job

    def test_export_settles_and_writes_the_file(self, tmp_path, monkeypatch):
        path = tmp_path / "out.csv"
        _plot, ws, panel = self._async_panel()
        ds = _dataset(ws, name="e")
        panel._csv_path = str(path)
        monkeypatch.setattr(
            "glplot.gui.panels.data_editor._write_csv_table",
            lambda p, h, a: (time.sleep(0.05), _write_csv_table(p, h, a))[1],
        )

        panel._export_csv(ds)
        assert not path.exists()  # not yet -- still running in the background

        _wait_until(lambda: (panel._poll_csv_job(), path.exists())[1])
        assert "Exported" in panel._status

    def test_import_failure_surfaces_via_poll(self, tmp_path):
        _plot, ws, panel = self._async_panel()
        panel._csv_path = str(tmp_path / "missing.csv")

        panel._import_csv()
        _wait_until(lambda: (panel._poll_csv_job(), panel._error != "")[1])
        assert "Could not read" in panel._error

    def test_cancel_stops_waiting_immediately(self, tmp_path, monkeypatch):
        path = tmp_path / "big.csv"
        path.write_text("x,y\n1,2\n")
        _plot, ws, panel = self._async_panel()
        panel._csv_path = str(path)
        monkeypatch.setattr(
            "glplot.gui.panels.data_editor._read_csv_table",
            lambda p: (time.sleep(0.3), _read_csv_table(p))[1],
        )
        panel._import_csv()
        job = panel._csv_job
        job.cancel()

        panel._poll_csv_job()
        assert panel._csv_job is None
        assert "data" not in ws.store.names()
        assert job.poll().state == "cancelled"

    def test_notification_pushed_for_a_slow_import(self, tmp_path, monkeypatch):
        path = tmp_path / "data.csv"
        path.write_text("x,y\n1,2\n")
        _plot, ws, panel = self._async_panel()
        panel._csv_path = str(path)
        monkeypatch.setattr("glplot.gui.panels.data_editor._ASYNC_NOTIFY_THRESHOLD_SECONDS", 0.0)
        notifications.clear()

        panel._import_csv()
        _wait_until(lambda: (panel._poll_csv_job(), "data" in ws.store.names())[1])

        toasts = notifications.active()
        assert len(toasts) == 1
        assert toasts[0].kind == "success"
        notifications.clear()

    def test_no_notification_below_the_threshold(self, tmp_path, monkeypatch):
        path = tmp_path / "data.csv"
        path.write_text("x,y\n1,2\n")
        _plot, ws, panel = self._async_panel()
        panel._csv_path = str(path)
        monkeypatch.setattr("glplot.gui.panels.data_editor._ASYNC_NOTIFY_THRESHOLD_SECONDS", 999.0)
        notifications.clear()

        panel._import_csv()
        _wait_until(lambda: (panel._poll_csv_job(), "data" in ws.store.names())[1])
        assert notifications.active() == []
        notifications.clear()


class TestCsvAsyncUiIntegration:
    """_draw_io_section's busy row and Cancel button, driven through a real headless
    imgui frame -- a lighter-weight harness than the rest of this file's, scoped to
    just this method (which is all that touches imgui in the CSV import/export path)."""

    @pytest.fixture
    def imgui_context(self):
        imgui_bundle = pytest.importorskip("imgui_bundle")
        imgui = imgui_bundle.imgui
        ctx = imgui.create_context()
        io = imgui.get_io()
        io.display_size = 800, 600
        io.delta_time = 1 / 60.0
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        yield io
        imgui.destroy_context(ctx)

    def _draw_io(self, panel, ds):
        from imgui_bundle import imgui

        imgui.new_frame()
        imgui.set_next_window_pos((0.0, 0.0))
        imgui.set_next_window_size((400.0, 200.0))
        imgui.begin("##test")
        panel._draw_io_section(ds)
        imgui.end()
        imgui.render()

    def test_busy_row_appears_while_pending(self, imgui_context, tmp_path, monkeypatch):
        from imgui_bundle import imgui

        path = tmp_path / "big.csv"
        path.write_text("x,y\n1,2\n")
        _plot, ws, panel = _panel()
        panel._force_async = True
        panel._csv_path = str(path)
        monkeypatch.setattr(
            "glplot.gui.panels.data_editor._read_csv_table",
            lambda p: (time.sleep(0.2), _read_csv_table(p))[1],
        )
        ds = _dataset(ws, name="d")
        panel._import_csv()

        texts = []
        real_text = imgui.text

        def spy_text(s, *a, **k):
            texts.append(str(s))
            return real_text(s, *a, **k)

        monkeypatch.setattr(imgui, "text", spy_text)
        self._draw_io(panel, ds)
        assert any("Importing" in t for t in texts)

    def test_cancel_button_click_reaches_the_job(self, imgui_context, tmp_path, monkeypatch):
        from imgui_bundle import imgui

        path = tmp_path / "big.csv"
        path.write_text("x,y\n1,2\n")
        _plot, ws, panel = _panel()
        panel._force_async = True
        panel._csv_path = str(path)
        monkeypatch.setattr(
            "glplot.gui.panels.data_editor._read_csv_table",
            lambda p: (time.sleep(0.3), _read_csv_table(p))[1],
        )
        ds = _dataset(ws, name="d")
        panel._import_csv()
        job = panel._csv_job
        assert job is not None

        real_button = imgui.button

        def spy_button(label, *a, **k):
            if label.startswith("Cancel##"):
                return True
            return real_button(label, *a, **k)

        monkeypatch.setattr(imgui, "button", spy_button)
        self._draw_io(panel, ds)

        assert job.poll().state == "cancelled"

    def test_import_export_buttons_hidden_while_pending(self, imgui_context, tmp_path, monkeypatch):
        path = tmp_path / "big.csv"
        path.write_text("x,y\n1,2\n")
        _plot, ws, panel = _panel()
        panel._force_async = True
        panel._csv_path = str(path)
        monkeypatch.setattr(
            "glplot.gui.panels.data_editor._read_csv_table",
            lambda p: (time.sleep(0.2), _read_csv_table(p))[1],
        )
        ds = _dataset(ws, name="d")
        panel._import_csv()

        tooltips = []
        real_icon_button = None
        from glplot.gui import icons as icons_module

        real_icon_button = icons_module.icon_button

        def spy_icon_button(id_str, shape, *a, **k):
            tooltips.append(k.get("tooltip"))
            return real_icon_button(id_str, shape, *a, **k)

        monkeypatch.setattr(icons_module, "icon_button", spy_icon_button)
        self._draw_io(panel, ds)
        assert "Import CSV" not in tooltips
        assert "Export CSV" not in tooltips


# ----------------------------------------------------------------------------------
# Transform-section parameter sliders + live preview -- "excel + desmos": free
# variables in a Transform expression (a*x + b) become sliders (glplot/gui/
# expressions.py's free_variables(), already built for panels/functions.py's
# identical Desmos-style slider feature), and the resulting column is previewed live
# before Apply bakes the current slider values in (see DataEditorPanel._transform's
# own docstring on why this is a snapshot, not a live link).
# ----------------------------------------------------------------------------------


def _dataset_xy(ws, name="t", n=5):
    ds = DataSet(name, [Column("x", np.arange(float(n))), Column("y", np.arange(float(n)) * 2.0)])
    ws.store.add(ds)
    return ds


class TestSyncTransformParams:
    """_sync_transform_params(): auto-detecting sliders from the expression text."""

    def test_detects_free_variables_excluding_columns(self):
        _plot, ws, panel = _panel()
        ds = _dataset_xy(ws)
        panel._transform_expr = "a * x + b"
        panel._sync_transform_params(ds)
        assert panel._transform_param_order == ["a", "b"]
        assert set(panel._transform_params) == {"a", "b"}

    def test_column_names_are_never_treated_as_parameters(self):
        _plot, ws, panel = _panel()
        ds = _dataset_xy(ws)
        panel._transform_expr = "x + y"
        panel._sync_transform_params(ds)
        assert panel._transform_param_order == []

    def test_new_params_default_to_one(self):
        _plot, ws, panel = _panel()
        ds = _dataset_xy(ws)
        panel._transform_expr = "k * x"
        panel._sync_transform_params(ds)
        assert panel._transform_params["k"].value == 1.0

    def test_is_gated_on_the_expression_text_changing(self):
        """Test that re-syncing the SAME text is a no-op -- a per-frame call must not
        re-detect (and potentially reset) sliders every single frame."""
        _plot, ws, panel = _panel()
        ds = _dataset_xy(ws)
        panel._transform_expr = "a * x"
        panel._sync_transform_params(ds)
        panel._transform_params["a"].value = 7.0
        panel._sync_transform_params(ds)  # same text again
        assert panel._transform_params["a"].value == 7.0

    def test_an_existing_slider_value_survives_editing_the_expression_further(self):
        """Test that tweaking a already has a dialled-in value that a*x -> a*x+b keeps."""
        _plot, ws, panel = _panel()
        ds = _dataset_xy(ws)
        panel._transform_expr = "a * x"
        panel._sync_transform_params(ds)
        panel._transform_params["a"].value = 3.0
        panel._transform_expr = "a * x + b"
        panel._sync_transform_params(ds)
        assert panel._transform_params["a"].value == 3.0
        assert panel._transform_params["b"].value == 1.0

    def test_a_dropped_parameter_stays_remembered_but_leaves_the_order(self):
        """Test that removing 'b' from the text drops it from what's DRAWN, but the
        slider state itself is not discarded -- typing it back restores the value."""
        _plot, ws, panel = _panel()
        ds = _dataset_xy(ws)
        panel._transform_expr = "a * x + b"
        panel._sync_transform_params(ds)
        panel._transform_params["b"].value = 9.0
        panel._transform_expr = "a * x"
        panel._sync_transform_params(ds)
        assert panel._transform_param_order == ["a"]
        assert panel._transform_params["b"].value == 9.0

    def test_an_invalid_expression_leaves_the_previous_sliders_in_place(self):
        """Test that a mid-typing/hostile expression does not yank sliders away."""
        _plot, ws, panel = _panel()
        ds = _dataset_xy(ws)
        panel._transform_expr = "a * x"
        panel._sync_transform_params(ds)
        panel._transform_expr = "a * x +"  # incomplete -- invalid
        panel._sync_transform_params(ds)
        assert panel._transform_param_order == ["a"]

    def test_hash_never_becomes_a_parameter(self):
        """Test that '#' (the auto-bound row index) is excluded, same as it is for
        panels/functions.py's identical slider feature."""
        _plot, ws, panel = _panel()
        ds = _dataset_xy(ws)
        panel._transform_expr = "a * # + b"
        panel._sync_transform_params(ds)
        assert panel._transform_param_order == ["a", "b"]

    def test_caps_remembered_parameters(self):
        _plot, ws, panel = _panel()
        ds = _dataset_xy(ws)
        from glplot.gui.panels.data_editor import _MAX_REMEMBERED_PARAMS

        for i in range(_MAX_REMEMBERED_PARAMS + 5):
            panel._transform_expr = f"p{i}"
            panel._sync_transform_params(ds)
        assert len(panel._transform_params) <= _MAX_REMEMBERED_PARAMS


class TestTransformPreview:
    """_draw_transform_preview(): a live, cached preview of the transform's output."""

    @pytest.fixture
    def imgui_context(self):
        imgui_bundle = pytest.importorskip("imgui_bundle")
        imgui = imgui_bundle.imgui
        ctx = imgui.create_context()
        io = imgui.get_io()
        io.display_size = 800, 600
        io.delta_time = 1 / 60.0
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        yield io
        imgui.destroy_context(ctx)

    def _draw(self, panel, fn):
        from imgui_bundle import imgui

        imgui.new_frame()
        imgui.set_next_window_pos((0.0, 0.0))
        imgui.set_next_window_size((400.0, 300.0))
        imgui.begin("##test")
        fn()
        imgui.end()
        imgui.render()

    def test_computes_the_expected_values(self, imgui_context):
        _plot, ws, panel = _panel()
        ds = _dataset_xy(ws)
        panel._transform_expr = "a * x + b"
        panel._sync_transform_params(ds)
        panel._transform_params["a"].value = 2.0
        panel._transform_params["b"].value = 1.0

        self._draw(panel, lambda: panel._draw_transform_preview(ds))

        assert panel._transform_preview_error == ""
        assert np.allclose(panel._transform_preview_values, 2.0 * ds.get("x") + 1.0)

    def test_recomputes_when_a_slider_moves(self, imgui_context):
        _plot, ws, panel = _panel()
        ds = _dataset_xy(ws)
        panel._transform_expr = "a * x"
        panel._sync_transform_params(ds)
        self._draw(panel, lambda: panel._draw_transform_preview(ds))
        first = panel._transform_preview_values.copy()

        panel._transform_params["a"].value = 5.0
        self._draw(panel, lambda: panel._draw_transform_preview(ds))
        assert not np.allclose(first, panel._transform_preview_values)
        assert np.allclose(panel._transform_preview_values, 5.0 * ds.get("x"))

    def test_does_not_recompute_when_nothing_changed(self, imgui_context, monkeypatch):
        _plot, ws, panel = _panel()
        ds = _dataset_xy(ws)
        panel._transform_expr = "a * x"
        panel._sync_transform_params(ds)
        self._draw(panel, lambda: panel._draw_transform_preview(ds))

        calls = []
        real_eval = DataSet.eval_expression

        def spy_eval(self, expr, **kwargs):
            calls.append(expr)
            return real_eval(self, expr, **kwargs)

        monkeypatch.setattr(DataSet, "eval_expression", spy_eval)
        self._draw(panel, lambda: panel._draw_transform_preview(ds))
        assert calls == []

    def test_a_bad_expression_shows_an_error_not_a_crash(self, imgui_context):
        """'x.real' -- attribute access -- is rejected by the AST allowlist itself
        (unlike 'x + nope', where 'nope' just becomes a new parameter, not an error:
        that is the whole point of this feature, so it is not a case that can test
        error handling)."""
        _plot, ws, panel = _panel()
        ds = _dataset_xy(ws)
        panel._transform_expr = "x.real"
        panel._sync_transform_params(ds)
        self._draw(panel, lambda: panel._draw_transform_preview(ds))
        assert panel._transform_preview_error != ""
        assert panel._transform_preview_values is None

    def test_a_row_count_change_invalidates_the_cache(self, imgui_context):
        _plot, ws, panel = _panel()
        ds = _dataset_xy(ws, n=5)
        panel._transform_expr = "a * x"
        panel._sync_transform_params(ds)
        self._draw(panel, lambda: panel._draw_transform_preview(ds))
        assert panel._transform_preview_values.size == 5

        ds.add_column("z")  # does not change n_rows, but exercises the key regardless
        ds.insert_row(0)
        self._draw(panel, lambda: panel._draw_transform_preview(ds))
        assert panel._transform_preview_values.size == 6


class TestTransformApplyWithSliders:
    """The panel-level "Apply" path: bakes in the CURRENT slider values."""

    def test_apply_bakes_in_the_slider_values(self):
        plot, ws, panel = _panel()
        ds = _dataset_xy(ws)
        panel._select(ds)
        panel._transform_expr = "a * x + b"
        panel._sync_transform_params(ds)
        panel._transform_params["a"].value = 3.0
        panel._transform_params["b"].value = 2.0

        variables = {n: p.value for n, p in panel._transform_params.items()}
        panel._transform(ds, "y", "a * x + b", None, variables=variables)
        ws.queue.drain(plot)

        assert np.allclose(ds.get("y"), 3.0 * ds.get("x") + 2.0)

    def test_apply_into_a_new_column_with_sliders(self):
        plot, ws, panel = _panel()
        ds = _dataset_xy(ws)
        panel._select(ds)
        panel._transform_expr = "a * x"
        panel._sync_transform_params(ds)
        panel._transform_params["a"].value = 10.0

        variables = {n: p.value for n, p in panel._transform_params.items()}
        panel._transform(ds, "y", "a * x", "scaled", variables=variables)
        ws.queue.drain(plot)

        assert "scaled" in ds.column_names()
        assert np.allclose(ds.get("scaled"), 10.0 * ds.get("x"))

    def test_apply_is_undoable_with_sliders(self):
        plot, ws, panel = _panel()
        ds = _dataset_xy(ws)
        panel._select(ds)
        before = ds.get("y").copy()
        panel._transform_expr = "a * x"
        panel._sync_transform_params(ds)
        panel._transform_params["a"].value = 100.0

        variables = {n: p.value for n, p in panel._transform_params.items()}
        panel._transform(ds, "y", "a * x", None, variables=variables)
        ws.queue.drain(plot)
        assert np.allclose(ds.get("y"), 100.0 * ds.get("x"))

        ws.undo.undo()
        ws.queue.drain(plot)
        assert np.allclose(ds.get("y"), before)

    def test_slider_values_are_a_one_time_snapshot_not_a_live_link(self):
        """The recommended, user-confirmed design: Apply bakes in today's slider
        values; the column does NOT keep recomputing when the slider moves again
        afterward (that would need a wholly different, live-reactive column concept
        this codebase deliberately does not have -- see the memory on this decision)."""
        plot, ws, panel = _panel()
        ds = _dataset_xy(ws)
        panel._select(ds)
        panel._transform_expr = "a * x"
        panel._sync_transform_params(ds)
        panel._transform_params["a"].value = 2.0

        variables = {n: p.value for n, p in panel._transform_params.items()}
        panel._transform(ds, "y", "a * x", "scaled", variables=variables)
        ws.queue.drain(plot)
        first = ds.get("scaled").copy()

        panel._transform_params["a"].value = 999.0  # moving the slider afterward...
        assert np.array_equal(ds.get("scaled"), first)  # ...must not touch the column


class TestTransformSliderUiIntegration:
    """Driven through the real headless imgui harness (_draw_filter_section directly
    -- it lives inside a collapsed-by-default section in the real panel tree, so
    tests call it directly rather than fighting that collapse state, matching this
    file's existing conventions elsewhere in this class)."""

    @pytest.fixture
    def imgui_context(self):
        imgui_bundle = pytest.importorskip("imgui_bundle")
        imgui = imgui_bundle.imgui
        ctx = imgui.create_context()
        io = imgui.get_io()
        io.display_size = 800, 700
        io.delta_time = 1 / 60.0
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        yield io
        imgui.destroy_context(ctx)

    def _draw(self, panel, ds):
        from imgui_bundle import imgui

        imgui.new_frame()
        imgui.set_next_window_pos((0.0, 0.0))
        imgui.set_next_window_size((500.0, 700.0))
        imgui.begin("##test")
        panel._draw_filter_section(ds)
        imgui.end()
        imgui.render()

    def test_sliders_appear_for_a_parametrized_expression(self, imgui_context, monkeypatch):
        """A real imgui.slider_float call is drawn per detected parameter -- checked
        by spying on the widget itself, not on 'Parameters' (that text is imgui's own
        collapsing_header label, drawn internally, never through a plain imgui.text()
        call a Python-level spy could ever observe)."""
        from imgui_bundle import imgui

        _plot, ws, panel = _panel()
        ds = _dataset_xy(ws)
        panel._transform_col = "y"
        panel._transform_expr = "a * x + b"

        labels = []
        real_slider = imgui.slider_float

        def spy_slider(label, *a, **k):
            labels.append(label)
            return real_slider(label, *a, **k)

        monkeypatch.setattr(imgui, "slider_float", spy_slider)
        for _ in range(3):
            labels.clear()
            self._draw(panel, ds)

        assert labels == ["a", "b"]

    def test_full_flow_type_drag_apply(self, imgui_context, monkeypatch):
        """Type an expression, drag a slider (simulated directly, imgui sliders are
        not scriptable via synthetic mouse events in this headless harness), click
        Apply, and confirm the committed column used the dragged value."""
        from imgui_bundle import imgui

        plot, ws, panel = _panel()
        ds = _dataset_xy(ws)
        panel._select(ds)
        panel._transform_col = "y"
        panel._transform_expr = "a * x"
        panel._transform_to_new = True
        panel._transform_target = "scaled"

        for _ in range(2):
            self._draw(panel, ds)
        assert "a" in panel._transform_params
        panel._transform_params["a"].value = 4.0

        real_button = imgui.button

        def spy_button(label, *a, **k):
            if label == "Apply":
                return True
            return real_button(label, *a, **k)

        monkeypatch.setattr(imgui, "button", spy_button)
        self._draw(panel, ds)
        ws.queue.drain(plot)

        assert "scaled" in ds.column_names()
        assert np.allclose(ds.get("scaled"), 4.0 * ds.get("x"))

    def test_reset_button_resets_the_parameter_to_one(self, imgui_context, monkeypatch):
        _plot, ws, panel = _panel()
        ds = _dataset_xy(ws)
        panel._transform_col = "y"
        panel._transform_expr = "a * x"
        for _ in range(2):
            self._draw(panel, ds)
        panel._transform_params["a"].value = 42.0

        def spy_icon_button(id_str, shape, *a, **k):
            if shape == "refresh":
                return True
            return False

        from glplot.gui import icons as icons_module

        monkeypatch.setattr(icons_module, "icon_button", spy_icon_button)
        self._draw(panel, ds)

        assert panel._transform_params["a"].value == 1.0


class TestIndexAxisOption:
    """ "(index)" as a real X/Y/Z axis pick, not just a colour/size encoding.

    A table with no natural "against what" column (a bare 2-column x,y series, say)
    can be plotted against its own row number, mirroring the "Color by"/"Size by"
    pickers' pre-existing ``_INDEX_OPTION`` handling (see ``_resolve_encoding_arrays``).
    """

    @pytest.fixture
    def imgui_context(self):
        imgui_bundle = pytest.importorskip("imgui_bundle")
        imgui = imgui_bundle.imgui
        ctx = imgui.create_context()
        io = imgui.get_io()
        io.display_size = 800, 600
        io.delta_time = 1 / 60.0
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        yield io
        imgui.destroy_context(ctx)

    def _draw_2d(self, panel, ds):
        from imgui_bundle import imgui

        imgui.new_frame()
        imgui.set_next_window_pos((0.0, 0.0))
        imgui.set_next_window_size((400.0, 400.0))
        imgui.begin("##test")
        panel._draw_plot_section(ds)
        imgui.end()
        imgui.render()

    def _draw_3d(self, panel, ds, names):
        from imgui_bundle import imgui

        imgui.new_frame()
        imgui.set_next_window_pos((0.0, 0.0))
        imgui.set_next_window_size((400.0, 400.0))
        imgui.begin("##test")
        panel._draw_plot_section_3d(ds, names)
        imgui.end()
        imgui.render()

    # -- picker offers the option -------------------------------------------------

    def test_index_option_is_selectable_as_2d_x_column(self):
        plot, ws, panel = _panel()
        ds = _dataset(ws)
        panel._plot_x, panel._plot_y = panel._INDEX_OPTION, "y"
        panel._plot_kind = "line"
        # A stale-pick reset would clobber this back to a real column; confirm the
        # validity list widened to include the sentinel keeps it in place.
        names = ds.column_names()
        options = [panel._INDEX_OPTION] + list(names)
        assert panel._plot_x in options

    def test_index_option_is_selectable_as_2d_y_column(self):
        plot, ws, panel = _panel()
        ds = _dataset(ws)
        panel._plot_x, panel._plot_y = "x", panel._INDEX_OPTION
        names = ds.column_names()
        options = [panel._INDEX_OPTION] + list(names)
        assert panel._plot_y in options

    # -- plotting synthesizes arange data ------------------------------------------

    def test_plotting_with_index_x_produces_arange_data(self):
        plot, ws, panel = _panel()
        ds = _dataset(ws, n=6)
        panel._select(ds)
        panel._plot_x, panel._plot_y, panel._plot_kind = panel._INDEX_OPTION, "y", "line"
        panel._plot_dataset(ds)
        ws.queue.drain(plot)

        layer = plot.scene.layers[0]
        assert np.allclose(layer.pts[:, 0], np.arange(6, dtype=np.float64))
        assert np.allclose(layer.pts[:, 1], ds.get("y"))

    def test_plotting_with_index_y_produces_arange_data(self):
        plot, ws, panel = _panel()
        ds = _dataset(ws, n=6)
        panel._select(ds)
        panel._plot_x, panel._plot_y, panel._plot_kind = "x", panel._INDEX_OPTION, "line"
        panel._plot_dataset(ds)
        ws.queue.drain(plot)

        layer = plot.scene.layers[0]
        assert np.allclose(layer.pts[:, 0], ds.get("x"))
        assert np.allclose(layer.pts[:, 1], np.arange(6, dtype=np.float64))

    def test_index_option_is_selectable_as_3d_z_column(self):
        plot, ws, panel = _panel()
        ds = _dataset_with_color(ws, n=6)
        panel._select(ds)
        panel._plot_x, panel._plot_y, panel._plot_z = "x", "c", panel._INDEX_OPTION
        panel._plot_kind3d = "scatter3d"
        panel._plot_dataset_3d(ds)
        ws.queue.drain(plot)

        layer = next(ly for ly in plot.scene.layers if ly.layer_type == "scatter3d")
        pts = np.asarray(layer.vertices)
        assert np.allclose(pts[:, 0], ds.get("x"))
        assert np.allclose(pts[:, 1], ds.get("c"))
        assert np.allclose(pts[:, 2], np.arange(6, dtype=np.float64))

    def test_index_option_is_selectable_as_3d_x_column(self):
        plot, ws, panel = _panel()
        ds = _dataset_with_color(ws, n=6)
        panel._select(ds)
        panel._plot_x, panel._plot_y, panel._plot_z = panel._INDEX_OPTION, "x", "c"
        panel._plot_kind3d = "scatter3d"
        panel._plot_dataset_3d(ds)
        ws.queue.drain(plot)

        layer = next(ly for ly in plot.scene.layers if ly.layer_type == "scatter3d")
        pts = np.asarray(layer.vertices)
        assert np.allclose(pts[:, 0], np.arange(6, dtype=np.float64))
        assert np.allclose(pts[:, 1], ds.get("x"))
        assert np.allclose(pts[:, 2], ds.get("c"))

    # -- stale-pick validity check does not silently discard the sentinel ---------

    def test_index_pick_survives_a_redraw_without_being_reset_to_a_real_column(self, imgui_context):
        plot, ws, panel = _panel()
        ds = _dataset(ws)
        panel._select(ds)
        panel._plot_x = panel._INDEX_OPTION
        panel._plot_y = "y"

        for _ in range(3):
            self._draw_2d(panel, ds)

        assert panel._plot_x == panel._INDEX_OPTION

    def test_index_pick_survives_a_3d_redraw_without_being_reset(self, imgui_context):
        plot, ws, panel = _panel()
        ds = _dataset_with_color(ws, n=6)
        panel._select(ds)
        panel._plot_ndim = 3
        panel._ndim_touched = True
        panel._plot_x = "x"
        panel._plot_y = "c"
        panel._plot_z = panel._INDEX_OPTION
        names = ds.column_names()

        for _ in range(3):
            self._draw_3d(panel, ds, names)

        assert panel._plot_z == panel._INDEX_OPTION
