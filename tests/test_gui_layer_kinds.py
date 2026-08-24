"""Test that the Scene panel's "Change type" offers what it can actually deliver.

A type change is delete+recreate, so the *only* way back to the layer you started with is
to pick its original kind out of the same menu. That makes two silent failures equivalent
to data loss, and both had shipped:

* the picker probed ``layerops`` for a registry name that never existed and quietly fell
  back to a hardcoded ``("line", "scatter")``, so five of the seven kinds could not be
  picked -- and a ``bar`` or ``hist`` layer had no way home;
* ``layer_kind`` inferred a kind from ``layer_type`` alone, and the engine draws a field
  (``imshow``, ``pcolormesh``, ``hist2d``, ``contour``) as a scatter of one point per
  cell -- so an image offered to become a line, and came back as bare points.

Both are the same class of bug: an answer that is wrong rather than absent. The menu
greying out is a fine outcome and is asserted as one; a menu that offers a lie is not.
No OpenGL and no GPU.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.engine import GPULinePlot
from glplot.gui import layerops
from glplot.gui.panels.functions import FunctionsPanel
from glplot.gui.panels.mathlab import _LAYER_KINDS as _mathlab_kinds
from glplot.gui.panels.scene import _available_kinds, _can_change_kind


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


@pytest.fixture
def xy():
    x = np.linspace(0.0, 10.0, 12)
    return x, np.sin(x)


def _pyplot_layers(call) -> list:
    """The layers one ``pyplot`` call leaves behind, on a figure of its own."""
    gplt.figure("kind probe")
    call()
    return list(gplt.gcf().scene.layers)


class TestAvailableKinds:
    def test_offers_every_kind_the_registry_implements(self):
        """The picker and the builder must agree, or a kind is unreachable."""
        assert set(_available_kinds()) == set(layerops.KIND_KEYS)

    def test_every_panel_that_picks_a_kind_reads_the_registry(self):
        """Scene, Mathlab and Functions each had their own copy of the pair.

        One registry, no copies: each of these was independently hardcoded to
        ``("line", "scatter")``, so a kind added to ``layerops`` reached exactly none of
        them. Asserting on the panels' own constants is what makes the next kind show up
        everywhere instead of in whichever panel someone remembered.
        """
        assert set(_mathlab_kinds) == set(layerops.KIND_KEYS)
        assert set(_available_kinds()) == set(layerops.KIND_KEYS)
        source = inspect.getsource(FunctionsPanel._draw_actions)
        assert "layerops.KIND_KEYS" in source
        assert '("line", "scatter")' not in source

    def test_offers_the_derived_kinds(self):
        """The five that the fallback used to hide. Named, so a silent drop is loud."""
        assert {"step", "stem", "area", "bar", "hist"}.issubset(set(_available_kinds()))

    def test_every_offered_kind_can_actually_be_built(self, xy):
        """The other half of the contract: offering a kind whose creation raises."""
        plot = GPULinePlot()
        for kind in _available_kinds():
            layer = layerops.add_xy_layer(plot, *xy, kind=kind, label=f"L-{kind}")
            assert layerops.layer_kind(layer) == kind


class TestRoundTrip:
    """Every kind must be reachable *and* returnable -- the user's way back."""

    @pytest.mark.parametrize("kind", layerops.KIND_KEYS)
    def test_line_to_kind_and_back_restores_the_data(self, kind, xy):
        x, y = xy
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, x, y, kind="line", label="rt")

        changed = layerops.replot_layer_xy(
            plot, None, layer, *layerops.layer_source_xy(layer), kind=kind, label="rt"
        )
        source = layerops.layer_source_xy(changed)
        assert source is not None, f"{kind} layer stranded: nothing to rebuild it from"

        back = layerops.replot_layer_xy(plot, None, changed, *source, kind="line", label="rt")
        restored = layerops.layer_source_xy(back)
        assert restored is not None
        assert restored[0] == pytest.approx(x)
        assert restored[1] == pytest.approx(y)

    @pytest.mark.parametrize("kind", layerops.KIND_KEYS)
    def test_the_menu_stays_open_on_the_result(self, kind, xy):
        """A change must never strand a layer in its new type.

        Asserted through the panel's own gate, not a re-implementation of it: the whole
        bug was the gate asking the wrong question. ``bar``/``area``/``hist`` are patches
        with no ``layer_xy`` at all, so asking the geometry greys the menu out on exactly
        the layers a type change has just produced.
        """
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, *xy, kind=kind, label="stay")
        assert _can_change_kind(layer) is True

    @pytest.mark.parametrize("kind", layerops.KIND_KEYS)
    def test_the_original_kind_is_still_on_the_menu(self, kind, xy):
        """The user's actual way home: change away, then find the old kind offered."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, *xy, kind=kind, label="home")

        changed = layerops.replot_layer_xy(
            plot, None, layer, *layerops.layer_source_xy(layer), kind="scatter", label="home"
        )
        assert _can_change_kind(changed), f"a {kind} changed to scatter cannot change again"
        assert kind in _available_kinds()


class TestKindOptions:
    """Every kind's parameters must be reachable from every panel that builds one.

    They existed all along and only the Data panel drew them, so a bar plotted from
    Functions, Mathlab, or a Scene type change was stuck at its default width with no
    control anywhere in sight.
    """

    @pytest.mark.parametrize("kind", layerops.KIND_KEYS)
    def test_a_layers_options_are_readable_back(self, kind, xy):
        """What the Scene inspector edits: the live layer's own options."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, *xy, kind=kind, label="opts")
        assert layerops.layer_kind_options(layer) == layerops.default_kind_options(kind)

    def test_options_survive_the_rebuild_that_applies_them(self, xy):
        """Editing bar width must re-derive the bars, not silently keep the old ones."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, *xy, kind="bar", label="w")
        wide = layerops.replot_layer_xy(
            plot,
            None,
            layer,
            *layerops.layer_source_xy(layer),
            kind="bar",
            label="w",
            options={"baseline": 0.0, "bar_width": 2.0},
        )
        assert layerops.layer_kind_options(wide)["bar_width"] == pytest.approx(2.0)

    def test_bar_width_actually_changes_the_geometry(self, xy):
        """The option has to reach the vertices, or the control is a placebo."""
        plot = GPULinePlot()
        narrow = layerops.add_xy_layer(
            plot, *xy, kind="bar", label="n", options={"baseline": 0.0, "bar_width": 0.1}
        )

        def span(layer):
            return float(layer.vertices[:, 0].max() - layer.vertices[:, 0].min())

        thin = span(narrow)
        wide = layerops.replot_layer_xy(
            plot,
            None,
            narrow,
            *layerops.layer_source_xy(narrow),
            kind="bar",
            label="n",
            options={"baseline": 0.0, "bar_width": 3.0},
        )
        assert span(wide) > thin

    def test_line_and_scatter_declare_no_options(self):
        """The editor draws nothing for them, which is why it keys off the dict."""
        assert layerops.default_kind_options("line") == {}
        assert layerops.default_kind_options("scatter") == {}


class TestColorEncodingSurvivesKindChange:
    """``scatter(c=...)``/colour-by-column must not evaporate on a plot-type change.

    ``replot_layer_xy`` used to rebuild a different-kind layer with only a flat
    representative colour, dropping any per-row colour mapping the Style or Data panel
    had going -- even though the exact same information (``layer.metadata["cvalues"]``)
    was already sitting right there and :func:`layerops.replot_layer_xyz` (the 3D twin)
    already knew to reuse it.
    """

    @pytest.fixture
    def c_xy(self):
        x = np.linspace(0.0, 10.0, 12)
        c = np.linspace(0.0, 1.0, 12)
        return x, np.sin(x), c

    def test_scatter_to_bar_keeps_the_colour_column(self, c_xy):
        x, y, c = c_xy
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, x, y, kind="scatter", label="c", c=c, cmap="plasma")

        bar = layerops.replot_layer_xy(plot, None, layer, x, y, kind="bar", label="c")

        assert bar.metadata.get("cvalues") is not None
        assert np.asarray(bar.metadata["cvalues"]) == pytest.approx(c, abs=1e-3)
        assert bar.metadata.get("cmap") == "plasma"
        # A real per-bar mapping, not every bar painted the same flat colour.
        assert len(np.unique(bar.colors, axis=0)) > 1

    def test_round_trip_through_line_keeps_the_colour_column(self, c_xy):
        x, y, c = c_xy
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, x, y, kind="scatter", label="c", c=c, cmap="viridis")

        line = layerops.replot_layer_xy(plot, None, layer, x, y, kind="line", label="c")
        back = layerops.replot_layer_xy(plot, None, line, x, y, kind="scatter", label="c")

        assert back.metadata.get("cvalues") is not None
        assert np.asarray(back.metadata["cvalues"]) == pytest.approx(c, abs=1e-3)
        assert len(np.unique(back.colors, axis=0)) > 1

    def test_switching_to_a_colormapped_kind_uses_its_own_values_not_the_old_column(self, c_xy):
        """hist2d owns its colours (density); it must not inherit the old per-row ``c``."""
        x, y, c = c_xy
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, x, y, kind="scatter", label="c", c=c, cmap="viridis")

        heat = layerops.replot_layer_xy(plot, None, layer, x, y, kind="hist2d", label="c")

        stored = np.asarray(heat.metadata["cvalues"])
        # hist2d bins the rows onto a grid; its cell count is not the row count, which is
        # exactly what keeps the old per-row ``c`` from being mistaken for it.
        assert stored.size != len(x)

    def test_edit_that_changes_row_count_drops_the_stale_mapping_instead_of_misapplying_it(
        self, c_xy
    ):
        x, y, c = c_xy
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, x, y, kind="scatter", label="c", c=c, cmap="viridis")

        shorter_x, shorter_y = x[:-2], y[:-2]
        rebuilt = layerops.replot_layer_xy(
            plot, None, layer, shorter_x, shorter_y, kind="bar", label="c"
        )
        # No crash, and no colour array recovered from the old (now mismatched) column.
        assert rebuilt.metadata.get("cvalues") is None


class TestKindOptionsEditorWidget:
    """``widgets.kind_options_editor`` draws every numeric field as a click-and-drag
    slider (``imgui.drag_int``/``drag_float``) rather than a +/- stepper
    (``imgui.input_int``/``input_float``) -- a single click still opens keyboard entry
    (Dear ImGui's own drag-widget behaviour), a click-and-drag now also works, which a
    plain input field never supported. A real headless imgui frame, no GL: this is the
    one function in the file that actually calls into imgui, so it gets its own
    lightweight harness rather than the module's usual GL-less unit tests.
    """

    @pytest.fixture
    def imgui_context(self):
        imgui_bundle = pytest.importorskip("imgui_bundle")
        imgui_mod = imgui_bundle.imgui
        ctx = imgui_mod.create_context()
        io = imgui_mod.get_io()
        io.display_size = 800, 600
        io.delta_time = 1 / 60.0
        io.backend_flags |= imgui_mod.BackendFlags_.renderer_has_textures
        yield io
        imgui_mod.destroy_context(ctx)

    def _draw(self, opts):
        from imgui_bundle import imgui as imgui_mod

        from glplot.gui import widgets

        imgui_mod.new_frame()
        imgui_mod.set_next_window_pos((0.0, 0.0))
        imgui_mod.set_next_window_size((400.0, 400.0))
        imgui_mod.begin("##test")
        changed = widgets.kind_options_editor(opts)
        imgui_mod.end()
        imgui_mod.render()
        return changed

    @pytest.mark.parametrize("kind", ["hist2d", "hexbin", "bar", "hist", "boxplot"])
    def test_every_numeric_field_draws_without_raising(self, imgui_context, kind):
        opts = layerops.default_kind_options(kind)
        changed = self._draw(opts)  # must not raise
        assert changed is False  # nothing dragged, so nothing changed

    def test_hist2ds_new_options_are_present(self, imgui_context):
        opts = layerops.default_kind_options("hist2d")
        assert set(opts) == {"bins", "mincnt", "pad", "cmap"}
        self._draw(opts)

    def test_hexbins_new_options_are_present(self, imgui_context):
        opts = layerops.default_kind_options("hexbin")
        assert set(opts) == {"gridsize", "mincnt", "pad", "cmap"}
        self._draw(opts)

    def test_drag_int_is_used_for_bins_and_gridsize_and_mincnt(self):
        """The actual UX request: a stepper (input_int) must not have crept back in."""
        import inspect

        from glplot.gui import widgets

        source = inspect.getsource(widgets.kind_options_editor)
        assert "imgui.input_int" not in source
        assert "imgui.input_float" not in source
        assert source.count("imgui.drag_int") >= 3  # bins, gridsize, mincnt
        assert "imgui.drag_float" in source


class TestHeatMap:
    """``hist2d`` is the one heat map two columns express: a density of the rows.

    It is also the first kind whose colours come from the data rather than the caller,
    so it is the first to prove the ``colormapped`` path: a per-cell colour VBO, and the
    scalars retained beside it so the colormap stays re-mappable afterwards.

    Cells render as a real filled rectangular mesh (one quad per surviving cell), not a
    scatter of sized point markers standing in for cell centres -- a point's fixed pixel
    size cannot track the actual bin width, so it either leaves gaps (too small) or
    overlaps (too large) depending on zoom, and cannot represent a non-square cell at
    all. This makes ``hist2d`` architecturally identical to ``hexbin`` (see
    ``TestHexDensity``): a ``colormapped`` ``layer_type="patch"`` kind whose geometry is
    ``(verts, indices, mode, per-cell count)``, not ``(points, values)``.
    """

    @pytest.fixture
    def cloud(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0.0, 1.0, 2000)
        return x, x * 0.6 + rng.normal(0.0, 0.7, 2000)

    @staticmethod
    def _cell_key_map(vertices: np.ndarray, values: np.ndarray) -> dict:
        """(rounded bottom-left corner x, y) -> value, sidestepping ordering entirely."""
        corners = vertices[0::4].astype(np.float64)
        return {(round(cx, 3), round(cy, 3)): float(v) for (cx, cy), v in zip(corners, values)}

    def test_matches_the_picture_pyplot_draws(self, cloud):
        """Same numbers in, same picture out -- or the GUI is a second implementation."""
        x, y = cloud
        gplt.figure("ref")
        _counts, _xe, _ye, ref = gplt.hist2d(x, y, bins=24, cmap="magma")

        plot = GPULinePlot()
        gui = layerops.add_xy_layer(
            plot, x, y, kind="hist2d", label="heat", options={"bins": 24, "cmap": "magma"}
        )
        assert gui.vertices.shape == ref.vertices.shape
        assert gui.indices.shape == ref.indices.shape
        assert gui.mode == ref.mode == "triangles"

        ref_map = self._cell_key_map(ref.vertices, ref.metadata["cvalues"])
        gui_map = self._cell_key_map(gui.vertices, gui.metadata["cvalues"])
        assert ref_map == gui_map

    def test_cells_are_coloured_by_density(self, cloud):
        """The whole point: one colour per cell, not one colour for the layer."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, *cloud, kind="hist2d", label="heat")
        assert len(np.unique(layer.colors, axis=0)) > 1
        # 4 colour rows (one per corner) per cell, all identical -- the tile draws as
        # one solid colour, not a gradient across its own corners.
        n_cells = len(layer.vertices) // 4
        for i in range(min(n_cells, 20)):
            rows = layer.colors[i * 4 : (i + 1) * 4]
            assert np.allclose(rows, rows[0])

    def test_the_scalars_are_retained_alongside_the_colours(self, cloud):
        """The counts are tagged exactly as hexbin/bar tag theirs, for the same reason:
        the colours are already baked into a per-vertex VBO, and without the scalars
        beside them a re-map has nothing to work from.

        ``layer_colormap_kind`` does not recognise a *patch* as remappable (its
        "values2d" path assumes one colour row per geometry row, true for a scatter's
        points but not for a rectangle's 4-row fan) -- the same gap ``hexbin``/``bar``'s
        own colouring already has, not a regression from this kind switching from
        scatter to patch. Rebuilding via a changed ``cmap``/``bins`` option (the
        actually-wired path, exercised below) is what the Data Editor's kind picker
        uses.
        """
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, *cloud, kind="hist2d", label="heat")
        assert layer.metadata.get("cvalues") is not None
        assert len(layer.metadata["cvalues"]) == len(layer.vertices) // 4

    def test_the_cmap_option_reaches_the_colours(self, cloud):
        """Two colormaps must not produce the same VBO, or the control is a placebo."""
        plot = GPULinePlot()
        a = layerops.add_xy_layer(
            plot, *cloud, kind="hist2d", label="a", options={"bins": 16, "cmap": "magma"}
        )
        magma = np.array(a.colors, copy=True)
        b = layerops.replot_layer_xy(
            plot,
            None,
            a,
            *layerops.layer_source_xy(a),
            kind="hist2d",
            label="a",
            options={"bins": 16, "cmap": "viridis"},
        )
        assert not np.allclose(magma, b.colors)

    def test_changing_bins_recolours_as_well_as_regrids(self, cloud):
        """A bin change rescales every count; stale colours on a new grid would lie.

        ``replot_layer_xy`` re-derives *in place* on a same-kind change, so ``layer``
        and ``rebuilt`` are the same object by the time it returns -- the "before"
        vertex count has to be captured first, or the comparison is an object compared
        against itself (see ``TestHexDensity`` for where this bit once already).
        """
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(
            plot, *cloud, kind="hist2d", label="h", options={"bins": 10, "cmap": "magma"}
        )
        original_vertex_count = len(layer.vertices)
        rebuilt = layerops.replot_layer_xy(
            plot,
            None,
            layer,
            *layerops.layer_source_xy(layer),
            kind="hist2d",
            label="h",
            options={"bins": 40, "cmap": "magma"},
        )
        assert len(rebuilt.colors) == len(rebuilt.vertices)
        assert len(rebuilt.vertices) != original_vertex_count

    def test_empty_cells_are_dropped(self, cloud):
        """A full grid of mostly-zero cells paints the background and hides the data."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(
            plot, *cloud, kind="hist2d", label="h", options={"bins": 30, "cmap": "magma"}
        )
        assert len(layer.vertices) // 4 < 30 * 30

    def test_survives_a_column_of_nothing_but_nan(self):
        """`histogram2d` cannot bin non-finite rows; empty geometry is 'draw nothing'."""
        plot = GPULinePlot()
        nan = np.full(8, np.nan)
        layer = layerops.add_xy_layer(plot, nan, nan, kind="hist2d", label="empty")
        assert len(layer.vertices) == 0
        assert len(layer.indices) == 0

    def test_tiles_share_edges_with_no_gap_or_overlap(self, cloud):
        """The whole point of a real mesh over sized point markers: adjacent cells'
        shared edge is the exact same coordinate on both sides, at any bin count."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, *cloud, kind="hist2d", label="h", options={"bins": 12})
        xs = np.round(layer.vertices[:, 0].astype(np.float64), 6)
        # Every distinct x column present is either a left or right edge of some cell;
        # a real tiling reuses each interior edge twice (once as a cell's right side,
        # once as its neighbour's left) -- a point-cloud approximation has no such
        # constraint at all, since a marker's edges are never other markers' edges.
        unique_xs, counts = np.unique(xs, return_counts=True)
        assert len(unique_xs) >= 2
        assert (counts >= 2).sum() >= 1, "no shared interior edge found -- cells do not tile"

    def _cell_widths(self, layer):
        """Every cell's own (x1 - x0), reading the rectangle's own two x-corners."""
        xs = layer.vertices[:, 0].astype(np.float64).reshape(-1, 4)
        return xs.max(axis=1) - xs.min(axis=1)

    def _rows_of_adjacent_x_spans(self, layer):
        """Group cells into histogram rows (same y0, a real ``np.histogram2d`` row
        shares it whatever the padding), each row's cells sorted left to right as
        ``(x0, x1)`` pairs -- what "does cell i touch/gap/overlap cell i+1" needs.

        ``_rect_geometry``'s own vertex order is (left,bottom), (left,top), (right,top),
        (right,bottom), so corners 0/2 give the left/right edge x, corner 0 the row's y.
        """
        verts = layer.vertices.astype(np.float64).reshape(-1, 4, 2)
        x0, x1, y0 = verts[:, 0, 0], verts[:, 2, 0], np.round(verts[:, 0, 1], 6)
        rows = []
        for row_y in np.unique(y0):
            in_row = y0 == row_y
            order = np.argsort(x0[in_row])
            rows.append((x0[in_row][order], x1[in_row][order]))
        return rows

    def test_mincnt_hides_sparse_cells(self):
        """The GUI's own name for `pyplot.hist2d`'s `cmin`: fewer than this many points
        in a cell, and it is not drawn at all -- not coloured at the low end, gone."""
        # A dense cluster at the origin plus a handful of scattered singletons far away,
        # so "bins=6" puts every singleton in its own count-1 cell.
        rng = np.random.default_rng(5)
        dense = rng.normal(0.0, 0.05, 500)
        x = np.concatenate([dense, [5.0, -5.0, 5.0, -5.0]])
        y = np.concatenate([dense, [5.0, 5.0, -5.0, -5.0]])
        plot = GPULinePlot()
        all_cells = layerops.add_xy_layer(
            plot, x, y, kind="hist2d", label="all", options={"bins": 6, "mincnt": 0}
        )
        filtered = layerops.add_xy_layer(
            plot, x, y, kind="hist2d", label="filtered", options={"bins": 6, "mincnt": 2}
        )
        n_all = len(all_cells.vertices) // 4
        n_filtered = len(filtered.vertices) // 4
        assert n_filtered < n_all
        assert filtered.metadata["cvalues"].min() >= 2

    def test_mincnt_default_changes_nothing(self, cloud):
        """0 (the default) must reproduce exactly what omitting the option already did."""
        plot = GPULinePlot()
        without = layerops.add_xy_layer(
            plot, *cloud, kind="hist2d", label="a", options={"bins": 15}
        )
        explicit_zero = layerops.add_xy_layer(
            plot, *cloud, kind="hist2d", label="b", options={"bins": 15, "mincnt": 0}
        )
        assert np.array_equal(without.vertices, explicit_zero.vertices)

    def test_pad_zero_reproduces_the_pre_existing_exact_fit(self, cloud):
        plot = GPULinePlot()
        without = layerops.add_xy_layer(
            plot, *cloud, kind="hist2d", label="a", options={"bins": 15}
        )
        explicit_zero = layerops.add_xy_layer(
            plot, *cloud, kind="hist2d", label="b", options={"bins": 15, "pad": 0.0}
        )
        assert np.array_equal(without.vertices, explicit_zero.vertices)

    def test_positive_pad_grows_every_cell_by_that_percentage(self, cloud):
        plot = GPULinePlot()
        base = layerops.add_xy_layer(
            plot, *cloud, kind="hist2d", label="a", options={"bins": 12, "pad": 0.0}
        )
        grown = layerops.add_xy_layer(
            plot, *cloud, kind="hist2d", label="b", options={"bins": 12, "pad": 20.0}
        )
        base_w, grown_w = self._cell_widths(base), self._cell_widths(grown)
        assert len(base_w) == len(grown_w)
        np.testing.assert_allclose(grown_w, base_w * 1.20, rtol=1e-5)

    def _adjacent_pair_gaps(self, cloud, bins, pad):
        """(padded x1[i] - padded x0[i+1]) for every pair of cells that were genuinely
        adjacent grid columns in the unpadded (pad=0) reference -- a survivor list can
        skip an empty column, so "next in the sorted list" is not always "next in the
        grid"; only a pair the *unpadded* geometry already had touching (x1==x0) proves
        that. Positive means overlap, negative means a gap, zero means still touching.
        """
        plot = GPULinePlot()
        base = layerops.add_xy_layer(
            plot, *cloud, kind="hist2d", label="ref", options={"bins": bins, "pad": 0.0}
        )
        padded = layerops.add_xy_layer(
            plot, *cloud, kind="hist2d", label="h", options={"bins": bins, "pad": pad}
        )
        base_rows = self._rows_of_adjacent_x_spans(base)
        padded_rows = self._rows_of_adjacent_x_spans(padded)
        assert len(base_rows) == len(padded_rows)

        gaps = []
        for (bx0, bx1), (px0, px1) in zip(base_rows, padded_rows):
            assert len(bx0) == len(px0)  # same surviving columns, pad changes extents only
            touching = np.isclose(bx1[:-1], bx0[1:])
            gaps.append(px1[:-1][touching] - px0[1:][touching])
        return np.concatenate(gaps)

    def test_positive_pad_makes_adjacent_cells_overlap(self, cloud):
        """The stated purpose: closing a hairline seam by deliberately overlapping."""
        gaps = self._adjacent_pair_gaps(cloud, bins=12, pad=25.0)
        assert len(gaps) > 0, "fixture produced no genuinely adjacent pair to check"
        assert (gaps > 0).all(), "grown cells did not overlap their neighbour"

    def test_negative_pad_shrinks_cells_and_opens_a_gap(self, cloud):
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(
            plot, *cloud, kind="hist2d", label="h", options={"bins": 12, "pad": -20.0}
        )
        widths = self._cell_widths(layer)
        base = layerops.add_xy_layer(
            plot, *cloud, kind="hist2d", label="ref", options={"bins": 12, "pad": 0.0}
        )
        base_widths = self._cell_widths(base)
        np.testing.assert_allclose(widths, base_widths * 0.80, rtol=1e-5)

        gaps = self._adjacent_pair_gaps(cloud, bins=12, pad=-20.0)
        assert len(gaps) > 0, "fixture produced no genuinely adjacent pair to check"
        assert (gaps < 0).all(), "shrunk cells must not still touch their neighbour"

    def test_extreme_negative_pad_cannot_invert_the_geometry(self, cloud):
        """However far the slider is pushed, a cell's width must stay positive."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(
            plot, *cloud, kind="hist2d", label="h", options={"bins": 10, "pad": -99.0}
        )
        assert (self._cell_widths(layer) > 0.0).all()


class TestHexDensity:
    """``hexbin`` is a hexagonal density: the honest alternative to ``hist2d`` for a
    genuinely dense scatter, tiling the plane without a square grid's rows/columns
    drawing the eye where the data has none.

    It is the first ``colormapped`` kind whose geometry is a real filled mesh rather
    than points (hist2d's cells draw as sized dots -- ``layerops.py``'s own docstring
    calls out the gaps that leaves). That makes it the first to prove the *patch* half
    of the colormapped contract: ``add_xy_layer``'s patch branch and
    ``update_layer_kind_data``'s colormapped branch previously only ever unpacked a
    3-tuple (verts, indices, mode) or a 2-tuple (points, values) respectively --
    hexbin's geometry is 4 elements, (verts, indices, mode, per-hexagon counts), which
    is what actually exercises the fix.
    """

    @pytest.fixture
    def cloud(self):
        rng = np.random.default_rng(11)
        x = rng.normal(0.0, 2.0, 3000)
        return x, x * 0.4 + rng.normal(0.0, 1.0, 3000)

    @staticmethod
    def _hexagon_key_map(vertices: np.ndarray, values: np.ndarray) -> dict:
        """(rounded hub x, rounded hub y) -> value, sidestepping ordering entirely."""
        hubs = vertices[0::7].astype(np.float64)
        return {(round(cx, 3), round(cy, 3)): float(v) for (cx, cy), v in zip(hubs, values)}

    def test_matches_the_picture_pyplot_draws(self, cloud):
        """Same numbers in, same picture out -- or the GUI is a second implementation.

        ``gridsize=23`` is deliberate, not arbitrary: layerops derives ``ny`` from
        ``nx`` with ``round()``, pyplot's own ``hexbin()`` with ``int()`` truncation --
        the two heuristics agree at 23 (both give 13) and disagree at plenty of other
        values, so this pins a gridsize where the comparison isolates the lattice
        math the two implementations share, not the aspect-ratio heuristic they don't.
        """
        x, y = cloud
        gplt.figure("ref")
        ref = gplt.hexbin(x, y, gridsize=23, cmap="magma")

        plot = GPULinePlot()
        gui = layerops.add_xy_layer(
            plot, x, y, kind="hexbin", label="hex", options={"gridsize": 23, "cmap": "magma"}
        )
        assert gui.vertices.shape == ref.vertices.shape
        assert gui.indices.shape == ref.indices.shape
        assert gui.mode == ref.mode == "triangles"

        ref_map = self._hexagon_key_map(ref.vertices, ref.metadata["cvalues"])
        gui_map = self._hexagon_key_map(gui.vertices, gui.metadata["cvalues"])
        assert ref_map == gui_map

    def test_hexagons_are_coloured_by_density(self, cloud):
        """The whole point: one colour per hexagon, not one colour for the layer."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, *cloud, kind="hexbin", label="hex")
        assert len(np.unique(layer.colors, axis=0)) > 1
        # 7 colour rows (hub + 6 corners) per hexagon, all identical -- the shape draws
        # as one solid colour, not a gradient across its own corners.
        n_hex = len(layer.vertices) // 7
        for i in range(min(n_hex, 20)):
            rows = layer.colors[i * 7 : (i + 1) * 7]
            assert np.allclose(rows, rows[0])

    def test_the_scalars_are_retained_alongside_the_colours(self, cloud):
        """The counts are tagged exactly as hist2d/bar tag theirs, for the same reason:
        the colours are already baked into a per-vertex VBO, and without the scalars
        beside them a re-map has nothing to work from.

        ``layer_colormap_kind`` itself does not yet recognise a *patch* as remappable
        (its "values2d" path assumes one colour row per geometry row, true for a
        scatter's points but not for a hexagon's 7-row fan) -- the same pre-existing
        gap ``bar``'s own ``c=`` colouring has always had, not something introduced
        here. Rebuilding via a changed ``cmap``/``gridsize`` option (the actually-wired
        path, exercised below) is what the Data Editor's kind picker uses.
        """
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, *cloud, kind="hexbin", label="hex")
        assert layer.metadata.get("cvalues") is not None
        assert len(layer.metadata["cvalues"]) == len(layer.vertices) // 7

    def test_the_cmap_option_reaches_the_colours(self, cloud):
        """Two colormaps must not produce the same VBO, or the control is a placebo."""
        plot = GPULinePlot()
        a = layerops.add_xy_layer(
            plot, *cloud, kind="hexbin", label="a", options={"gridsize": 20, "cmap": "magma"}
        )
        magma = np.array(a.colors, copy=True)
        b = layerops.replot_layer_xy(
            plot,
            None,
            a,
            *layerops.layer_source_xy(a),
            kind="hexbin",
            label="a",
            options={"gridsize": 20, "cmap": "viridis"},
        )
        assert not np.allclose(magma, b.colors)

    def test_changing_gridsize_recolours_as_well_as_regrids(self, cloud):
        """A gridsize change moves every hexagon; stale colours would lie about them.

        ``replot_layer_xy`` re-derives *in place* on a same-kind change (that's the
        whole point -- the layer's identity survives), so ``layer`` and ``rebuilt`` are
        the same object by the time it returns; the "before" snapshot has to be taken
        first, or the comparison is an object compared against itself.
        """
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(
            plot, *cloud, kind="hexbin", label="h", options={"gridsize": 10, "cmap": "magma"}
        )
        original_vertex_count = len(layer.vertices)
        rebuilt = layerops.replot_layer_xy(
            plot,
            None,
            layer,
            *layerops.layer_source_xy(layer),
            kind="hexbin",
            label="h",
            options={"gridsize": 40, "cmap": "magma"},
        )
        assert len(rebuilt.colors) == len(rebuilt.vertices)
        assert len(rebuilt.vertices) != original_vertex_count

    def test_in_place_update_follows_a_live_link(self, cloud):
        """``update_layer_kind_data`` is what makes a bound layer follow a cell edit."""
        x, y = cloud
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, x, y, kind="hexbin", label="hex")
        original_vertices = np.array(layer.vertices, copy=True)
        original_colors = np.array(layer.colors, copy=True)

        # Two tight, far-apart clusters redistribute the counts unevenly across
        # hexagons; a uniform spatial rescale would not (the relative density per cell
        # is roughly invariant under scaling x and its cell width together), so this is
        # what actually exercises whether the colours were rederived, not just resized.
        rng = np.random.default_rng(23)
        n = len(x)
        x2 = np.concatenate([rng.normal(-8.0, 0.3, n // 2), rng.normal(8.0, 0.3, n - n // 2)])
        y2 = np.concatenate([rng.normal(-8.0, 0.3, n // 2), rng.normal(8.0, 0.3, n - n // 2)])

        changed = layerops.update_layer_kind_data(plot, layer, x2, y2, kind="hexbin")
        assert changed is True
        assert len(layer.colors) == len(layer.vertices)
        moved = layer.vertices.shape != original_vertices.shape or not np.allclose(
            layer.vertices, original_vertices
        )
        assert moved, "hexagon geometry did not change after the source data moved"
        recoloured = layer.colors.shape != original_colors.shape or not np.allclose(
            layer.colors, original_colors
        )
        assert recoloured, "hexagon colours did not follow the regridded counts"

    def test_empty_cells_are_dropped(self, cloud):
        """A hexagon per grid cell, densely packed, would paint the background too."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(
            plot, *cloud, kind="hexbin", label="h", options={"gridsize": 30, "cmap": "magma"}
        )
        assert len(layer.vertices) // 7 < 30 * 30

    def test_survives_every_point_landing_on_the_same_spot(self):
        """A zero-width extent divides by zero on the way; one hexagon holds them all."""
        plot = GPULinePlot()
        same = np.full(40, 5.0)
        layer = layerops.add_xy_layer(plot, same, same, kind="hexbin", label="degenerate")
        assert len(layer.vertices) // 7 == 1
        assert layer.metadata["cvalues"][0] == 40

    def test_gridsize_zero_is_auto_not_a_single_hexagon(self, cloud):
        """0 means "pick one", the same convention hist2d's ``bins`` uses -- not zero."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(
            plot, *cloud, kind="hexbin", label="auto", options={"gridsize": 0}
        )
        n_hex = len(layer.vertices) // 7
        assert 1 < n_hex < len(cloud[0])

    def test_survives_a_column_of_nothing_but_nan(self):
        """No finite rows to bin; empty geometry is 'draw nothing', same as hist2d."""
        plot = GPULinePlot()
        nan = np.full(8, np.nan)
        layer = layerops.add_xy_layer(plot, nan, nan, kind="hexbin", label="empty")
        assert len(layer.vertices) == 0
        assert len(layer.indices) == 0

    def test_gridsize_is_capped_before_it_ever_reaches_the_grid(self):
        """The same runaway-allocation guard ``hist2d``'s ``bins`` gets, for the same
        reason -- checked at the capping function itself, not by actually materialising
        a multi-million-hexagon mesh: ``_hexbin_geometry`` resolves ``nx`` through the
        exact ``_hist_bin_count`` hist2d already uses, so a pathological gridsize is
        bounded the same way a pathological bin count is, without a second cap to drift
        out of step with the first.
        """
        assert layerops._hist_bin_count(1000, 10**9) == layerops.MAX_HIST_BINS

    @staticmethod
    def _hex_radii(layer):
        """Every hexagon's own hub-to-corner distance, averaged over its six corners
        (they are all equal by construction, so this is really just reading it back)."""
        verts = layer.vertices.astype(np.float64).reshape(-1, 7, 2)
        hub = verts[:, 0:1, :]
        corners = verts[:, 1:, :]
        return np.linalg.norm(corners - hub, axis=2).mean(axis=1)

    def test_mincnt_hides_sparse_hexagons(self):
        """The same name and semantics ``pyplot.hexbin``'s own ``mincnt`` already has."""
        rng = np.random.default_rng(5)
        dense = rng.normal(0.0, 0.05, 500)
        x = np.concatenate([dense, [5.0, -5.0, 5.0, -5.0]])
        y = np.concatenate([dense, [5.0, 5.0, -5.0, -5.0]])
        plot = GPULinePlot()
        all_hex = layerops.add_xy_layer(
            plot, x, y, kind="hexbin", label="all", options={"gridsize": 6, "mincnt": 0}
        )
        filtered = layerops.add_xy_layer(
            plot, x, y, kind="hexbin", label="filtered", options={"gridsize": 6, "mincnt": 2}
        )
        assert len(filtered.vertices) // 7 < len(all_hex.vertices) // 7
        assert filtered.metadata["cvalues"].min() >= 2

    def test_mincnt_default_changes_nothing(self, cloud):
        plot = GPULinePlot()
        without = layerops.add_xy_layer(
            plot, *cloud, kind="hexbin", label="a", options={"gridsize": 15}
        )
        explicit_zero = layerops.add_xy_layer(
            plot, *cloud, kind="hexbin", label="b", options={"gridsize": 15, "mincnt": 0}
        )
        assert np.array_equal(without.vertices, explicit_zero.vertices)

    def test_pad_zero_reproduces_the_pre_existing_exact_fit(self, cloud):
        plot = GPULinePlot()
        without = layerops.add_xy_layer(
            plot, *cloud, kind="hexbin", label="a", options={"gridsize": 15}
        )
        explicit_zero = layerops.add_xy_layer(
            plot, *cloud, kind="hexbin", label="b", options={"gridsize": 15, "pad": 0.0}
        )
        assert np.array_equal(without.vertices, explicit_zero.vertices)

    def test_pad_scales_every_hexagons_own_radius(self, cloud):
        plot = GPULinePlot()
        base = layerops.add_xy_layer(
            plot, *cloud, kind="hexbin", label="a", options={"gridsize": 12, "pad": 0.0}
        )
        grown = layerops.add_xy_layer(
            plot, *cloud, kind="hexbin", label="b", options={"gridsize": 12, "pad": 20.0}
        )
        shrunk = layerops.add_xy_layer(
            plot, *cloud, kind="hexbin", label="c", options={"gridsize": 12, "pad": -20.0}
        )
        base_r, grown_r, shrunk_r = (
            self._hex_radii(base),
            self._hex_radii(grown),
            self._hex_radii(shrunk),
        )
        assert len(base_r) == len(grown_r) == len(shrunk_r)
        np.testing.assert_allclose(grown_r, base_r * 1.20, rtol=1e-5)
        np.testing.assert_allclose(shrunk_r, base_r * 0.80, rtol=1e-5)

    def test_extreme_negative_pad_cannot_invert_the_geometry(self, cloud):
        """However far the slider is pushed, a hexagon's radius must stay positive."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(
            plot, *cloud, kind="hexbin", label="h", options={"gridsize": 10, "pad": -99.0}
        )
        assert (self._hex_radii(layer) > 0.0).all()


class TestBoxPlot:
    """Tukey's definition, checked against matplotlib's own implementation of it.

    A box plot is the one kind here that is *arithmetic* rather than layout: getting the
    whiskers subtly wrong draws a plausible box that misreports the spread, which no
    amount of looking at it would reveal.
    """

    @pytest.fixture
    def sample(self):
        rng = np.random.default_rng(3)
        y = np.concatenate([rng.normal(10.0, 2.0, 300), [25.0, 26.0, -6.0]])
        return np.linspace(0.0, 4.0, len(y)), y

    def _stats(self, y, whis=1.5):
        from matplotlib import cbook

        return cbook.boxplot_stats(np.asarray(y, dtype=np.float64), whis=whis)[0]

    def test_quartiles_match_matplotlib(self, sample):
        x, y = sample
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, x, y, kind="boxplot", label="b")
        ref = self._stats(y)
        ys = layer.pts[:, 1]
        # float32 geometry against float64 stats: ~7 digits is all it can carry.
        for value in (ref["q1"], ref["med"], ref["q3"]):
            assert np.isclose(ys, value, rtol=1e-6).any(), f"{value} missing from the box"

    def test_whiskers_reach_the_furthest_point_inside_the_fence(self, sample):
        """Not the min/max of the data: the outliers must not stretch them."""
        x, y = sample
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, x, y, kind="boxplot", label="b")
        ref = self._stats(y)
        assert np.isclose(float(layer.pts[:, 1].min()), ref["whislo"], rtol=1e-6)
        assert np.isclose(float(layer.pts[:, 1].max()), ref["whishi"], rtol=1e-6)
        assert float(layer.pts[:, 1].max()) < float(np.max(y))  # 25/26 are fliers

    def test_whis_widens_the_whiskers(self, sample):
        """The option has to reach the arithmetic, or the control is a placebo."""
        x, y = sample
        plot = GPULinePlot()
        tight = layerops.add_xy_layer(
            plot, x, y, kind="boxplot", label="b", options={"whis": 0.5, "box_width": 0.0}
        )
        narrow = float(tight.pts[:, 1].max())
        loose = layerops.replot_layer_xy(
            plot,
            None,
            tight,
            *layerops.layer_source_xy(tight),
            kind="boxplot",
            label="b",
            options={"whis": 3.0, "box_width": 0.0},
        )
        assert float(loose.pts[:, 1].max()) > narrow

    def test_box_width_reaches_the_geometry(self, sample):
        x, y = sample
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(
            plot, x, y, kind="boxplot", label="b", options={"whis": 1.5, "box_width": 2.0}
        )
        span = float(layer.pts[:, 0].max() - layer.pts[:, 0].min())
        assert span == pytest.approx(2.0, rel=1e-5)

    def test_survives_a_column_of_nothing_but_nan(self):
        plot = GPULinePlot()
        nan = np.full(8, np.nan)
        layer = layerops.add_xy_layer(plot, nan, nan, kind="boxplot", label="empty")
        assert len(layer.pts) == 0

    def test_survives_a_single_row(self):
        """Every quartile collapses onto one value; the IQR is zero, not a crash."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, [1.0], [5.0], kind="boxplot", label="one")
        assert np.allclose(layer.pts[:, 1], 5.0)


class TestFieldArtistsHaveNoKind:
    """A field is drawn as a scatter of cells. That is a renderer detail, not a kind."""

    @pytest.mark.parametrize(
        "name, call",
        [
            ("imshow", lambda: gplt.imshow(np.sin(np.outer(np.linspace(0, 3, 16), np.ones(16))))),
            (
                "pcolormesh",
                lambda: gplt.pcolormesh(
                    *np.meshgrid(np.linspace(-2, 2, 8), np.linspace(-2, 2, 8)),
                    np.zeros((8, 8)),
                ),
            ),
            ("hist2d", lambda: gplt.hist2d(np.linspace(0, 1, 40), np.linspace(0, 1, 40))),
        ],
    )
    def test_field_layer_has_no_kind(self, name, call):
        for layer in _pyplot_layers(call):
            assert layerops.layer_kind(layer) is None, (
                f"{name} resolves to a kind, so the menu offers a conversion that "
                "throws the field away and cannot rebuild it"
            )

    def test_imshow_renders_as_a_scatter(self):
        """The premise: this is why layer_type alone cannot answer the question."""
        layers = _pyplot_layers(lambda: gplt.imshow(np.zeros((8, 8))))
        assert [str(getattr(layer, "layer_type", "")) for layer in layers] == ["scatter"]

    @pytest.mark.parametrize("name", ["axhline", "axvline"])
    def test_guides_have_no_kind(self, name):
        """A guide spans the view; it is not a line through data."""
        for layer in _pyplot_layers(lambda: getattr(gplt, name)(0.5)):
            assert layerops.layer_kind(layer) is None


class TestLineFamilyIsAKind:
    """``gplt.plot_lines`` builds a ``line_family``, and it was in neither list.

    Not in ``KIND_KEYS``, so the picker could not offer it; not in ``UNSUPPORTED_KINDS``
    either, which is the register of what a two-column table genuinely cannot reach. It
    fell between the two -- and the Data panel *does* ingest it (``DataSet.from_layer``
    reads its ``ab`` as columns ``a``/``b``), so it would happily replot the family as one
    of the other kinds with nothing on the menu to put it back. Same class of bug as the
    two in this module's docstring: the third one to ship.

    It belongs in the registry rather than in ``UNSUPPORTED_KINDS`` because it *is* two
    columns -- a slope and an intercept per row -- unlike errorbar (needs a third), imshow
    (needs a grid) or pie (is a composition).
    """

    A = np.array([0.5, -1.0, 2.0, 0.25], dtype=np.float32)
    B = np.array([0.0, 1.0, -0.5, 2.0], dtype=np.float32)

    def _family(self):
        gplt.figure("fam", density=True)
        gplt.plot_lines(self.A, self.B, x_range=(-3.0, 3.0))
        plot = gplt.gcf()
        return plot, plot._primary_line_layer

    def test_plot_lines_layer_has_a_kind(self):
        """The whole bug in one line: this used to be None."""
        _, layer = self._family()
        assert layerops.layer_kind(layer) == "line_family"

    def test_the_menu_is_open_on_it(self):
        _, layer = self._family()
        assert _can_change_kind(layer) is True

    def test_the_picker_offers_it(self):
        assert "line_family" in _available_kinds()

    def test_it_is_not_also_listed_as_unsupported(self):
        """The two lists must not disagree about the same kind."""
        assert not any("line_family" in key for key in layerops.UNSUPPORTED_KINDS)

    def test_options_recover_the_layers_real_x_range(self):
        """Not the registry placeholder: the family is drawn over (-3, 3), not (-1, 1).

        A default here would silently crop the lines on the way back -- same data, wrong
        picture, and nothing on screen to say why.
        """
        _, layer = self._family()
        assert layerops.layer_kind_options(layer) == {"x_lo": -3.0, "x_hi": 3.0}

    def test_round_trip_away_and_back_restores_the_family(self):
        """The user's report: change it, then get it back.

        Fails on the old code at the very first step -- ``layer_source_xy`` returned None,
        so there was nothing to rebuild from and no ``line_family`` to rebuild it as.
        """
        plot, layer = self._family()
        options = layerops.layer_kind_options(layer)
        source = layerops.layer_source_xy(layer)
        assert source is not None, "family stranded: nothing to rebuild it from"

        away = layerops.replot_layer_xy(plot, None, layer, *source, kind="scatter", label="away")
        assert layerops.layer_kind(away) == "scatter"

        back = layerops.replot_layer_xy(
            plot,
            None,
            away,
            *layerops.layer_source_xy(away),
            kind="line_family",
            label="back",
            options=options,
        )
        assert layerops.layer_kind(back) == "line_family"
        assert back.ab[:, 0] == pytest.approx(self.A)
        assert back.ab[:, 1] == pytest.approx(self.B)
        assert tuple(map(float, back.x_range)) == (-3.0, 3.0)

    def test_builder_returns_the_engines_singleton_not_the_last_layer(self):
        """The family is inserted at index 0 and reused, unlike every appending add_*.

        Reading ``layers[-1]`` would tag whatever was plotted last -- so this fails with
        the kind tag landing on the wrong layer entirely.
        """
        plot = GPULinePlot()
        plot.add_scatter(
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.ones((2, 4), dtype=np.float32),
            label="decoy",
        )
        family = layerops.add_xy_layer(
            plot,
            self.A,
            self.B,
            kind="line_family",
            label="fam",
            options={"x_lo": -1.0, "x_hi": 1.0},
        )
        assert family is plot._primary_line_layer
        assert family.label == "fam"
        assert plot.scene.layers[-1].label == "decoy", "the decoy was mislabelled"

    def test_a_degenerate_x_range_is_refused(self):
        """x_hi <= x_lo draws every line as a point; better to say so than to render it."""
        plot = GPULinePlot()
        with pytest.raises(ValueError, match="x_hi > x_lo"):
            layerops.add_xy_layer(
                plot,
                self.A,
                self.B,
                kind="line_family",
                label="bad",
                options={"x_lo": 1.0, "x_hi": 1.0},
            )


class TestPlainArtistsKeepTheirKind:
    """The whitelist must not overshoot: a real line is still a line."""

    def test_plot_is_a_line(self):
        layers = _pyplot_layers(lambda: gplt.plot([0.0, 1.0], [0.0, 1.0]))
        assert layerops.layer_kind(layers[0]) == "line"

    def test_scatter_is_a_scatter(self):
        layers = _pyplot_layers(lambda: gplt.scatter([0.0, 1.0], [0.0, 1.0]))
        assert layerops.layer_kind(layers[0]) == "scatter"

    def test_raw_engine_layer_keeps_the_legacy_inference(self):
        """No artist tag at all: the pre-artist layers the fallback exists for."""
        plot = GPULinePlot()
        x = np.linspace(0.0, 1.0, 8, dtype=np.float32)
        plot.add_line_strip(x, x, color=(1.0, 1.0, 1.0, 1.0), label="raw")
        assert layerops.layer_kind(plot.scene.layers[0]) == "line"

    def test_an_explicit_tag_still_wins(self):
        """The tag is the answer whenever it exists; the artist gate is a fallback."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, [0.0, 1.0], [0.0, 1.0], kind="bar", label="tagged")
        layer.metadata["artist"] = "imshow"
        assert layerops.layer_kind(layer) == "bar"
