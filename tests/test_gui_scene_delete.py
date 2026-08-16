"""Test the Scene panel's per-row delete button.

The toolbar already had a trash, but it acts on the *selection* and is disabled until
there is one, so removing a single plot meant first discovering that you have to select
it. The row button is the one people reach for.

Its one real hazard is scope. A trash sitting on a row promises to delete that row, so it
must delete exactly that layer even when the row happens to be part of a multi-selection
-- taking the other four along would be a surprise that undo exists to survive but nobody
should need. That is what most of this file pins down.

No GPU. The draw tests use the headless imgui harness (CONTRACT §2.10).
"""

from __future__ import annotations

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.engine import GPULinePlot
from glplot.gui import layerops
from glplot.gui.panels.scene import ScenePanel
from glplot.gui.workspace import Workspace


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


@pytest.fixture
def scene():
    """A plot with three named line layers, plus a live panel over it."""
    plot = GPULinePlot()
    ws = Workspace(plot)
    panel = ScenePanel(ws)
    x = np.linspace(0.0, 10.0, 20)
    for name in ("A", "B", "C"):
        layerops.add_xy_layer(plot, x, np.sin(x), kind="line", label=name)
    return plot, ws, panel


def _labels(plot):
    return [layer.label for layer in plot.scene.layers]


def _click_row_trash(monkeypatch, panel, layer):
    """Press the row's trash for ``layer`` and return.

    Goes through ``_draw_row_delete`` rather than calling ``_delete_layers`` directly,
    because the button's *scope* is the thing worth testing and that decision lives in
    the draw method. Faking the click at ``icon_button`` is what makes that reachable —
    imgui has no way to synthesise a hit-tested press headlessly.
    """
    import glplot.gui.panels.scene as scene_mod

    monkeypatch.setattr(scene_mod, "icon_button", lambda *a, **k: True)
    panel._draw_row_delete(layer)


class TestRowDelete:
    def test_deletes_the_row_it_sits_on(self, scene, monkeypatch):
        plot, ws, panel = scene
        _click_row_trash(monkeypatch, panel, plot.scene.layers[1])
        ws.queue.drain(plot)
        assert _labels(plot) == ["A", "C"]

    def test_ignores_the_selection(self, scene, monkeypatch):
        """The whole point: the row's trash is about the row, not about what is selected."""
        plot, ws, panel = scene
        panel._selection = {plot.scene.layers[0].layer_id, plot.scene.layers[2].layer_id}
        _click_row_trash(monkeypatch, panel, plot.scene.layers[1])
        ws.queue.drain(plot)
        assert _labels(plot) == ["A", "C"], "the selected layers were taken along"

    def test_deleting_a_selected_row_leaves_its_neighbours(self, scene, monkeypatch):
        """Even when the row *is* in the selection, only it goes."""
        plot, ws, panel = scene
        panel._selection = {layer.layer_id for layer in plot.scene.layers}
        _click_row_trash(monkeypatch, panel, plot.scene.layers[0])
        ws.queue.drain(plot)
        assert _labels(plot) == ["B", "C"]

    def test_undo_puts_it_back_at_its_index(self, scene, monkeypatch):
        plot, ws, panel = scene
        _click_row_trash(monkeypatch, panel, plot.scene.layers[1])
        ws.queue.drain(plot)
        ws.undo.undo()
        ws.queue.drain(plot)
        assert _labels(plot) == ["A", "B", "C"]

    def test_not_clicking_deletes_nothing(self, scene, monkeypatch):
        """The other half of the harness: prove the fake click is what fires it."""
        import glplot.gui.panels.scene as scene_mod

        plot, ws, panel = scene
        monkeypatch.setattr(scene_mod, "icon_button", lambda *a, **k: False)
        panel._draw_row_delete(plot.scene.layers[1])
        ws.queue.drain(plot)
        assert _labels(plot) == ["A", "B", "C"]

    def test_deleting_every_row_one_by_one_empties_the_scene(self, scene, monkeypatch):
        plot, ws, panel = scene
        while plot.scene.layers:
            _click_row_trash(monkeypatch, panel, plot.scene.layers[0])
            ws.queue.drain(plot)
        assert _labels(plot) == []

    def test_the_toolbar_trash_still_takes_the_whole_selection(self, scene):
        """The row button must not have narrowed the toolbar's, which is the bulk one."""
        plot, ws, panel = scene
        panel._selection = {plot.scene.layers[0].layer_id, plot.scene.layers[1].layer_id}
        panel._delete_layers(panel._selected_layers())
        ws.queue.drain(plot)
        assert _labels(plot) == ["C"]


class TestRowDeleteDraws:
    """Through the real imgui frame: a button nobody can render is not a button."""

    def _harness(self):
        imgui = pytest.importorskip("imgui")
        imgui.create_context()
        io = imgui.get_io()
        io.display_size = 900, 700
        io.fonts.get_tex_data_as_rgba32()
        io.fonts.texture_id = 1
        io.delta_time = 1 / 60.0
        return imgui

    def test_the_panel_still_draws_with_the_button_in_the_row(self, scene):
        imgui = self._harness()
        plot, _ws, panel = scene
        imgui.new_frame()
        imgui.begin("Scene")
        panel.draw()
        imgui.end()
        imgui.render()
        assert imgui.get_draw_data() is not None

    def test_the_button_id_is_keyed_on_something_unique(self, scene):
        """The button keys its imgui id on `layer_id`; duplicates would cross-fire.

        Asserts the *premise* — that `layer_id` is unique per layer — rather than
        re-deriving the id string, which would only check this test against itself.
        """
        plot, _ws, _panel = scene
        ids = [layer.layer_id for layer in plot.scene.layers]
        assert len(set(ids)) == len(ids)
        assert None not in ids

    def test_a_long_label_does_not_break_the_row(self, scene):
        """A smoke test, and named as one: imgui does not raise on a clipped widget.

        The row reserves the button's width out of the label's, so it should stay in
        view — but nothing here can see pixels, so this only proves the layout survives
        a label far longer than the panel.
        """
        imgui = self._harness()
        plot, _ws, panel = scene
        plot.scene.layers[0].label = "a" * 400

        imgui.new_frame()
        imgui.begin("Scene")
        panel.draw()
        imgui.end()
        imgui.render()
        assert imgui.get_draw_data() is not None
