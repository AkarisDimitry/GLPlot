"""Test that the Scene panel's layer order agrees with the renderer's draw order.

The panel used to list ``scene.layers`` while the renderer sorted by ``style.zorder``
(``renderer_manager.py:96``), so with any non-default zorder the list and the picture
disagreed and a drag-reorder appeared to do nothing. These cover the layerops
primitives that fix it. No OpenGL or GPU.
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


def _plot_with(labels, zorders=None):
    """A plot holding one polyline per label, with optional explicit zorders."""
    plot = GPULinePlot()
    x = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    for label in labels:
        plot.add_line_strip(x, x, color=(1.0, 1.0, 1.0, 1.0), label=label)
    for layer, zorder in zip(plot.scene.layers, zorders or []):
        layer.style.zorder = zorder
    return plot


class TestDrawOrder:
    """draw_order must reproduce what the renderer does, not what the list says."""

    def test_matches_the_renderer_sort(self):
        """renderer_manager sorts by zorder; so does this."""
        plot = _plot_with(["a", "b", "c"], [5, 0, 2])
        assert [layer.label for layer in layerops.draw_order(plot)] == ["b", "c", "a"]

    def test_ties_keep_list_order(self):
        """The renderer's sort is stable, so equal zorders fall back to list order."""
        plot = _plot_with(["a", "b", "c"], [0, 0, 0])
        assert [layer.label for layer in layerops.draw_order(plot)] == ["a", "b", "c"]

    def test_empty_scene(self):
        """No layers, no order, no exception."""
        assert layerops.draw_order(GPULinePlot()) == []

    def test_does_not_mutate_the_scene(self):
        """It is a read-only view: the list itself must be untouched."""
        plot = _plot_with(["a", "b"], [5, 0])
        layerops.draw_order(plot)
        assert [layer.label for layer in plot.scene.layers] == ["a", "b"]


class TestZorderIsAuthoritative:
    """The predicate that decides whether a naive list-order panel would be lying."""

    def test_default_zorders_mean_list_order_is_truth(self):
        """The common case: nothing to explain, no badge to show."""
        assert layerops.zorder_is_authoritative(_plot_with(["a", "b"])) is False

    def test_detects_the_disagreement(self):
        """REGRESSION for the reported bug: list says a,b; the renderer draws b,a."""
        plot = _plot_with(["a", "b"], [5, 0])
        assert layerops.zorder_is_authoritative(plot) is True
        assert [layer.label for layer in plot.scene.layers] != [
            layer.label for layer in layerops.draw_order(plot)
        ]

    def test_ascending_zorders_are_not_a_disagreement(self):
        """Non-default zorders are fine as long as they agree with the list."""
        assert layerops.zorder_is_authoritative(_plot_with(["a", "b"], [1, 7])) is False


class TestRenumberZorder:
    """Renumbering must change the meaning without changing the picture."""

    def test_preserves_draw_order_exactly(self):
        """It derives the new numbers from the current draw order, so it is a visual no-op."""
        plot = _plot_with(["a", "b", "c"], [50, 0, -3])
        before = [layer.label for layer in layerops.draw_order(plot)]
        layerops.renumber_zorder(plot)
        assert [layer.label for layer in layerops.draw_order(plot)] == before

    def test_makes_list_order_equal_draw_order(self):
        """Afterwards scene.layers alone answers 'what draws on top'."""
        plot = _plot_with(["a", "b", "c"], [50, 0, -3])
        layerops.renumber_zorder(plot)
        assert [layer.label for layer in plot.scene.layers] == ["c", "b", "a"]
        assert layerops.zorder_is_authoritative(plot) is False

    def test_ranks_from_zero(self):
        """The chosen values are ranks, not the user's original numbers."""
        plot = _plot_with(["a", "b", "c"], [50, 0, -3])
        layerops.renumber_zorder(plot)
        assert [layer.style.zorder for layer in plot.scene.layers] == [0, 1, 2]

    def test_is_idempotent(self):
        """A second call must report no change."""
        plot = _plot_with(["a", "b"], [5, 0])
        layerops.renumber_zorder(plot)
        assert layerops.renumber_zorder(plot) is False

    def test_never_rebinds_scene_layers(self):
        """hud.py:249 and the panels hold a reference to this exact list object."""
        plot = _plot_with(["a", "b"], [5, 0])
        held = plot.scene.layers
        layerops.renumber_zorder(plot)
        assert plot.scene.layers is held


class TestMoveLayerToIndex:
    """The drag gesture, with the lie fixed."""

    def test_old_move_layer_does_not_move_the_picture(self):
        """REGRESSION: this is the bug. move_layer is list-only, by design (§1.7)."""
        plot = _plot_with(["a", "b"], [5, 0])
        before = [layer.label for layer in layerops.draw_order(plot)]
        layerops.move_layer(plot, 0, 1)
        assert [layer.label for layer in layerops.draw_order(plot)] == before

    def test_move_to_index_changes_the_draw_order(self):
        """The fix: dragging commits, so the picture follows the row."""
        plot = _plot_with(["a", "b"], [5, 0])
        target = plot.scene.layers[0]
        assert layerops.move_layer_to_index(plot, target.layer_id, 0) is True
        assert [layer.label for layer in layerops.draw_order(plot)] == ["a", "b"]

    def test_move_to_index_reconciles_both_orderings(self):
        """List and zorder must agree afterwards, not merely be consistent."""
        plot = _plot_with(["a", "b", "c"], [5, 0, 2])
        target = plot.scene.layers[2]
        layerops.move_layer_to_index(plot, target.layer_id, 0)
        assert [layer.label for layer in plot.scene.layers] == [
            layer.label for layer in layerops.draw_order(plot)
        ]
        assert layerops.zorder_is_authoritative(plot) is False

    def test_a_no_op_drag_still_commits(self):
        """Dropping a row on itself is still a statement that list order is meant."""
        plot = _plot_with(["a", "b"], [5, 0])
        target = plot.scene.layers[1]  # 'b' is already first in draw order
        layerops.move_layer_to_index(plot, target.layer_id, 0)
        assert layerops.zorder_is_authoritative(plot) is False

    def test_unknown_id_is_ignored(self):
        """A layer can be deleted between the drop and the drain."""
        plot = _plot_with(["a", "b"])
        assert layerops.move_layer_to_index(plot, 123456789, 0) is False

    def test_index_is_clamped(self):
        """An out-of-range drop lands at the end rather than raising."""
        plot = _plot_with(["a", "b", "c"])
        target = plot.scene.layers[0]
        assert layerops.move_layer_to_index(plot, target.layer_id, 99) is True
        assert plot.scene.layers[-1].label == "a"

    def test_applies_the_dirty_incantation(self):
        """A reorder is a scene mutation and must survive the latch (§1.4)."""
        plot = _plot_with(["a", "b"], [5, 0])
        plot.frame.dirty_scene = False
        plot.cache.refresh_requested = False
        plot.cache.capture_window = (0.0, 1.0, 0.0, 1.0)
        layerops.move_layer_to_index(plot, plot.scene.layers[0].layer_id, 0)
        assert plot.frame.dirty_scene is True
        assert plot.cache.refresh_requested is True
        assert plot.cache.capture_window is None


class TestLayerBounds:
    """The Scene inspector's bounds readout."""

    def test_reports_intrinsic_bounds(self):
        """Computed live from the arrays; there is no bounds cache (§1.9)."""
        plot = GPULinePlot()
        plot.add_line_strip(
            np.array([0.0, 2.0], dtype=np.float32),
            np.array([1.0, 5.0], dtype=np.float32),
            color=(1.0, 1.0, 1.0, 1.0),
            label="b",
        )
        assert layerops.layer_bounds(plot, plot.scene.layers[0]) == (0.0, 2.0, 1.0, 5.0)

    def test_applies_translation(self):
        """Must match what autoscale frames (renderer_manager.py:199-203)."""
        plot = GPULinePlot()
        plot.add_line_strip(
            np.array([0.0, 2.0], dtype=np.float32),
            np.array([1.0, 5.0], dtype=np.float32),
            color=(1.0, 1.0, 1.0, 1.0),
            label="b",
        )
        layer = plot.scene.layers[0]
        layer.translation = (10.0, -1.0)
        assert layerops.layer_bounds(plot, layer) == (10.0, 12.0, 0.0, 4.0)

    def test_text_layer_has_no_bounds(self):
        """TextLayer returns None by design; that is an answer, not an error."""
        plot = GPULinePlot()
        plot.add_text(0.5, 0.5, "hi", label="t")
        assert layerops.layer_bounds(plot, plot.scene.layers[0]) is None

    def test_element_count(self):
        """The stats readout's other half."""
        plot = _plot_with(["a"])
        assert layerops.layer_element_count(plot.scene.layers[0]) == 8
