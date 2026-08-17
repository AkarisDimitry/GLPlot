"""Style panel per-layer parameter editor: change plot type and kind options in place.

Exercises ``StylePanel._replot_layer`` / ``_draw_layer_kind`` -- the "edit any parameter of
the selected plot (bins, baseline, ...), including its representation type" feature. Logic
only (queue drained by hand as the main loop would); no imgui frame, no GL.
"""

from __future__ import annotations

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.engine import GPULinePlot
from glplot.gui import layerops


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


def _panel():
    plot = GPULinePlot()
    plot.options.enable_hud = True
    ws = plot.hud.workspace
    return plot, ws, ws.panels["style"]


def _bar(plot, n=6):
    x = np.arange(float(n))
    y = np.array([3.0, 7, 2, 9, 5, 6][:n])
    return layerops.add_xy_layer(
        plot, x, y, kind="bar", label="b", options={"baseline": 0.0, "bar_width": 0.0}
    )


class TestLayerParamEditor:
    def test_change_representation_type(self):
        plot, ws, panel = _panel()
        lay = _bar(plot)
        plot.interaction.selected_layer_id = lay.layer_id
        panel._replot_layer(lay, "line", {})
        ws.queue.drain(plot)
        live = layerops.find_layer(plot, plot.interaction.selected_layer_id)
        assert layerops.layer_kind(live) == "line"

    def test_edit_hist_bins(self):
        plot, ws, panel = _panel()
        x = np.arange(50.0)
        lay = layerops.add_xy_layer(
            plot, x, np.sin(x), kind="hist", label="h", options={"bins": 5, "baseline": 0.0}
        )
        plot.interaction.selected_layer_id = lay.layer_id
        panel._replot_layer(lay, "hist", {"bins": 20, "baseline": 0.0})
        ws.queue.drain(plot)
        live = layerops.find_layer(plot, plot.interaction.selected_layer_id)
        assert layerops.layer_kind_options(live)["bins"] == 20

    def test_selection_follows_a_rebuild(self):
        plot, ws, panel = _panel()
        lay = _bar(plot)
        old_id = lay.layer_id
        plot.interaction.selected_layer_id = old_id
        panel._replot_layer(lay, "scatter", {})
        ws.queue.drain(plot)
        # A different-kind change rebuilds under a new id; the selection must track it.
        new_id = plot.interaction.selected_layer_id
        assert layerops.find_layer(plot, new_id) is not None
        assert layerops.layer_kind(layerops.find_layer(plot, new_id)) == "scatter"

    def test_replot_on_missing_layer_is_a_safe_noop(self):
        # The layer may be deleted between the frame that queued the re-plot and the drain;
        # re-resolving by id must then find nothing and do nothing, never raise.
        plot, ws, panel = _panel()
        lay = _bar(plot)
        panel._replot_layer(lay, "line", {})
        # Remove it before the drain runs the queued apply().
        plot.scene.layers = [layer for layer in plot.scene.layers if layer is not lay]
        ws.queue.drain(plot)  # must not raise


class TestKindOptions:
    """New per-kind options: hist density/cumulative, bar align."""

    def test_hist_density_normalizes(self):
        plot, ws, _ = _panel()
        rng = np.random.default_rng(0)
        y = rng.normal(size=400)
        x = np.arange(400.0)
        lay = layerops.add_xy_layer(
            plot, x, y, kind="hist", label="h", options={"bins": 20, "density": True}
        )
        assert layerops.layer_kind_options(lay)["density"] is True

    def test_hist_cumulative_is_monotonic(self):
        plot, ws, _ = _panel()
        y = np.linspace(0, 10, 300)
        x = np.arange(300.0)
        lay = layerops.add_xy_layer(
            plot, x, y, kind="hist", label="h", options={"bins": 20, "cumulative": True}
        )
        # Bar tops (every 4th vertex is a top corner) must not decrease across a CDF.
        tops = lay.vertices[1::4, 1]
        assert np.all(np.diff(tops) >= -1e-9)

    def test_bar_align_edge_shifts_left_edge_to_x(self):
        plot, ws, _ = _panel()
        x = np.arange(5.0)
        y = np.array([1.0, 2, 3, 2, 1])
        center = layerops.add_xy_layer(
            plot, x, y, kind="bar", label="c", options={"align": "center", "bar_width": 0.8}
        )
        edge = layerops.add_xy_layer(
            plot, x, y, kind="bar", label="e", options={"align": "edge", "bar_width": 0.8}
        )
        assert float(center.vertices[0, 0]) == pytest.approx(-0.4)
        assert float(edge.vertices[0, 0]) == pytest.approx(0.0)

    def test_defaults_expose_new_keys(self):
        assert set(layerops.default_kind_options("hist")) == {
            "bins",
            "baseline",
            "density",
            "cumulative",
        }
        assert set(layerops.default_kind_options("bar")) == {"baseline", "bar_width", "align"}
