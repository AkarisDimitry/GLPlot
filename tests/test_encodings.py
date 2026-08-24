"""Tests for data-driven visual encodings (colour and size as dimensions).

Colour-as-a-dimension (``c=`` numeric->colormap, RGBA per point) already had coverage; this
file focuses on **size as a dimension** -- the ``s=`` array path added so marker size can
follow a variable, matching matplotlib's per-point ``s``. State only, no GL context needed.
"""

from __future__ import annotations

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.engine import GPULinePlot
from glplot.gui import layerops
from glplot.gui.datasets import Column, DataSet


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


class TestScatterSizeDimension:
    def test_scalar_size_leaves_sizes_none(self):
        lay = gplt.scatter([0, 1, 2], [0, 1, 2], s=12.0)
        assert lay.sizes is None
        assert lay.style.point_size == pytest.approx(12.0)

    def test_array_size_sets_per_point_sizes(self):
        sizes = np.array([4.0, 8.0, 16.0, 32.0])
        lay = gplt.scatter([0, 1, 2, 3], [0, 0, 0, 0], s=sizes)
        assert lay.sizes is not None
        assert np.allclose(lay.sizes, sizes)

    def test_array_size_retained_as_svalues_metadata(self):
        sizes = [5.0, 10.0, 15.0]
        lay = gplt.scatter([0, 1, 2], [1, 2, 3], s=sizes)
        assert lay.metadata.get("svalues") is not None
        assert np.allclose(lay.metadata["svalues"], sizes)

    def test_representative_scalar_is_the_mean(self):
        sizes = np.array([10.0, 20.0, 30.0])
        lay = gplt.scatter([0, 1, 2], [0, 0, 0], s=sizes)
        assert lay.style.point_size == pytest.approx(20.0)

    def test_wrong_length_size_array_raises(self):
        with pytest.raises(ValueError, match="one per point"):
            gplt.scatter([0, 1, 2], [0, 1, 2], s=np.array([1.0, 2.0]))

    def test_length_one_size_is_treated_as_scalar(self):
        lay = gplt.scatter([0, 1, 2], [0, 1, 2], s=np.array([7.0]))
        assert lay.sizes is None
        assert lay.style.point_size == pytest.approx(7.0)

    def test_color_and_size_dimensions_compose(self):
        n = 6
        vals = np.linspace(0, 1, n)
        sizes = np.linspace(4, 40, n)
        lay = gplt.scatter(np.arange(n), np.zeros(n), c=vals, cmap="viridis", s=sizes)
        # Colour came from a colormap (cvalues retained) AND size is per-point.
        assert lay.metadata.get("cvalues") is not None
        assert lay.sizes is not None and len(lay.sizes) == n


class TestAddXyLayerEncodings:
    """The GUI/layerops path: ``add_xy_layer`` mapping columns to colour and size."""

    def _plot(self):
        return GPULinePlot()

    def test_c_column_colormaps_the_scatter(self):
        e = self._plot()
        x = np.arange(8.0)
        lay = layerops.add_xy_layer(
            e, x, np.sin(x), kind="scatter", label="s", c=np.linspace(0, 1, 8), cmap="plasma"
        )
        assert lay.metadata.get("cvalues") is not None
        assert lay.metadata.get("cmap") == "plasma"

    def test_s_column_sets_per_point_sizes(self):
        e = self._plot()
        x = np.arange(8.0)
        lay = layerops.add_xy_layer(
            e, x, np.sin(x), kind="scatter", label="s", s=np.linspace(5, 25, 8)
        )
        assert lay.sizes is not None and len(lay.sizes) == 8
        assert lay.metadata.get("svalues") is not None

    def test_plain_scatter_unaffected(self):
        e = self._plot()
        x = np.arange(8.0)
        lay = layerops.add_xy_layer(e, x, np.sin(x), kind="scatter", label="s")
        assert lay.sizes is None
        assert lay.metadata.get("cvalues") is None

    def test_line_c_column_colors_each_vertex(self):
        e = self._plot()
        x = np.arange(8.0)
        lay = layerops.add_xy_layer(
            e, x, np.sin(x), kind="line", label="l", c=np.linspace(0, 1, 8), cmap="turbo"
        )
        assert layerops.layer_kind(lay) == "line"
        # One RGBA per vertex, so a segment can gradient from end to end.
        assert lay.colors is not None and lay.colors.shape == (8, 4)
        assert lay.metadata.get("cvalues") is not None

    def test_plain_line_stays_flat(self):
        e = self._plot()
        x = np.arange(8.0)
        lay = layerops.add_xy_layer(e, x, np.sin(x), kind="line", label="l")
        assert lay.colors is None

    def test_bar_c_column_colors_each_bar(self):
        e = self._plot()
        x = np.arange(6.0)
        y = np.array([3.0, 7, 2, 9, 5, 6])
        lay = layerops.add_xy_layer(e, x, y, kind="bar", label="b", c=y, cmap="viridis")
        # 6 bars x 4 verts each => a per-vertex colour array of 24 rows.
        assert lay.colors is not None and lay.colors.shape == (24, 4)
        assert lay.metadata.get("cvalues") is not None

    def test_bar_without_c_stays_flat(self):
        e = self._plot()
        x = np.arange(4.0)
        lay = layerops.add_xy_layer(e, x, np.array([1.0, 2, 3, 4]), kind="bar", label="b")
        assert lay.metadata.get("cvalues") is None


class TestDataPanelEncodings:
    """The Data panel resolving colour/size column pickers into arrays for a scatter plot."""

    def _panel(self):
        plot = GPULinePlot()
        plot.options.enable_hud = True
        ws = plot.hud.workspace
        return plot, ws, ws.panels["data"]

    def _dataset(self, ws, n=6):
        ds = DataSet(
            "t",
            [
                Column("x", np.arange(float(n))),
                Column("y", np.arange(float(n)) ** 2),
                Column("temp", np.linspace(10.0, 20.0, n)),
                Column("mass", np.linspace(100.0, 1000.0, n)),
            ],
        )
        ws.store.add(ds)
        return ds

    def test_size_column_is_normalised_into_pixel_range(self):
        plot, ws, panel = self._panel()
        ds = self._dataset(ws)
        panel._plot_kind = "scatter"
        panel._plot_s = "mass"
        _, s, _ = panel._resolve_encoding_arrays(ds)
        lo, hi = panel._SIZE_PX_RANGE
        assert s is not None
        assert float(np.min(s)) == pytest.approx(lo, abs=1e-3)
        assert float(np.max(s)) == pytest.approx(hi, abs=1e-3)

    def test_color_column_passed_raw_with_cmap(self):
        plot, ws, panel = self._panel()
        ds = self._dataset(ws)
        panel._plot_kind = "scatter"
        panel._plot_c = "temp"
        panel._plot_cmap = "inferno"
        c, _, cmap = panel._resolve_encoding_arrays(ds)
        assert c is not None and np.allclose(np.asarray(c), np.linspace(10.0, 20.0, 6))
        assert cmap == "inferno"

    def test_plotting_scatter_with_encodings_produces_data_driven_layer(self):
        plot, ws, panel = self._panel()
        ds = self._dataset(ws)
        panel._plot_x, panel._plot_y, panel._plot_kind = "x", "y", "scatter"
        panel._plot_c, panel._plot_s, panel._plot_cmap = "temp", "mass", "viridis"
        panel._plot_dataset(ds)
        ws.queue.drain(plot)
        scatters = [ly for ly in plot.scene.layers if ly.layer_type == "scatter"]
        assert scatters, "a scatter layer should have been created"
        lay = scatters[-1]
        assert lay.sizes is not None
        assert lay.metadata.get("cvalues") is not None

    def test_none_encodings_give_plain_scatter(self):
        plot, ws, panel = self._panel()
        ds = self._dataset(ws)
        panel._plot_x, panel._plot_y, panel._plot_kind = "x", "y", "scatter"
        panel._plot_c, panel._plot_s = "", ""
        panel._plot_dataset(ds)
        ws.queue.drain(plot)
        scatters = [ly for ly in plot.scene.layers if ly.layer_type == "scatter"]
        assert scatters and scatters[-1].sizes is None

    def test_index_color_resolves_to_row_position(self):
        plot, ws, panel = self._panel()
        ds = self._dataset(ws, n=6)
        panel._plot_kind = "scatter"
        panel._plot_c = panel._INDEX_OPTION
        c, _, _ = panel._resolve_encoding_arrays(ds)
        assert c is not None and np.allclose(np.asarray(c), np.arange(6))

    def test_index_size_normalises_like_a_real_column(self):
        plot, ws, panel = self._panel()
        ds = self._dataset(ws, n=6)
        panel._plot_kind = "scatter"
        panel._plot_s = panel._INDEX_OPTION
        _, s, _ = panel._resolve_encoding_arrays(ds)
        lo, hi = panel._SIZE_PX_RANGE
        assert s is not None
        assert float(np.min(s)) == pytest.approx(lo, abs=1e-3)
        assert float(np.max(s)) == pytest.approx(hi, abs=1e-3)
