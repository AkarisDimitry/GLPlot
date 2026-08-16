"""Test the multi-element selection model and rect picking.

Covers glplot.core.legacy.Selection, the picking offset table and its
vectorised id decode, and the engine's shift+drag marquee gesture. The
readback is exercised against a synthetic pick buffer with a stubbed
glReadPixels, so nothing here requires OpenGL or a GPU.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot import engine as eng
from glplot.core.legacy import SceneData, Selection
from glplot.managers import picking as picking_mod
from glplot.managers.picking import PickingManager
from glplot.options import EngineOptions
from glplot.renderers.base import GLOffscreenTarget

# Take the modifier constants from the engine's OWN glfw reference, which is the
# object it compares `mods` against. A plain `import glfw` here is unreliable:
# tests/test_camera_anisotropy.py replaces sys.modules["glfw"] with a MagicMock
# at import time and never restores it, so a later import may get the mock.
_glfw = eng.glfw
_GLFW_MOCKED = isinstance(_glfw, MagicMock)
needs_real_glfw = pytest.mark.skipif(
    _GLFW_MOCKED,
    reason="another test module replaced sys.modules['glfw'] with a MagicMock at import time",
)
# Likewise for OpenGL: if sys.modules["OpenGL.GL"] was mocked before picking.py
# ran, its `from OpenGL.GL import *` bound no names and the readback cannot even
# resolve GL_RED_INTEGER. Nothing here needs a GPU -- only the symbols.
needs_gl_symbols = pytest.mark.skipif(
    not hasattr(picking_mod, "glReadPixels"),
    reason="another test module replaced sys.modules['OpenGL.GL'] with a MagicMock at import time",
)


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


def _layer(layer_id, layer_type, count, visible=True, has_gl=True):
    """Build a minimal stand-in exposing only what the offset table reads."""
    ly = types.SimpleNamespace()
    ly.layer_id = layer_id
    ly.layer_type = layer_type
    ly.style = types.SimpleNamespace(visible=visible)
    ly.ab = np.zeros((count, 2), dtype=np.float32) if layer_type == "line_family" else None
    if has_gl:
        ly._gl = types.SimpleNamespace(count=count)
    return ly


class TestSelection:
    """Test the Selection container."""

    def test_add_accepts_scalar_iterable_and_ndarray(self):
        """add() coerces ints, lists and numpy arrays to plain int elements."""
        sel = Selection()
        sel.add(1, 3)
        sel.add(1, [4, 5])
        sel.add(1, np.array([5, 6]))
        assert sel.by_layer == {1: {3, 4, 5, 6}}
        assert all(isinstance(i, int) for i in sel.by_layer[1])

    def test_add_empty_does_not_create_a_layer(self):
        """Adding nothing must not leave an empty layer key behind."""
        sel = Selection()
        sel.add(1, [])
        assert sel.by_layer == {}
        assert sel.is_empty()

    def test_remove_drops_the_layer_when_it_empties(self):
        """remove() ignores unselected indices and prunes empty layers."""
        sel = Selection()
        sel.add(1, [1, 2])
        sel.remove(1, [2, 99])
        assert sel.by_layer == {1: {1}}
        sel.remove(1, 1)
        assert sel.by_layer == {}
        sel.remove(42, 0)  # unknown layer is a no-op

    def test_toggle_flips_membership(self):
        """toggle() removes selected indices and adds unselected ones."""
        sel = Selection()
        sel.add(1, [1, 2])
        sel.toggle(1, [2, 3])
        assert sel.by_layer == {1: {1, 3}}
        sel.toggle(1, [1, 3])
        assert sel.by_layer == {}

    def test_set_replaces_the_whole_selection(self):
        """set() drops every other layer, not just the named one."""
        sel = Selection()
        sel.add(1, [1])
        sel.add(2, [2])
        sel.set(3, [7])
        assert sel.by_layer == {3: {7}}

    def test_set_all_replaces_and_update_merges(self):
        """set_all() replaces the mapping; update() merges additively."""
        sel = Selection()
        sel.add(1, [0])
        sel.set_all({2: [1, 2], 3: [0]})
        assert sel.by_layer == {2: {1, 2}, 3: {0}}
        sel.update({2: [3], 4: [9]})
        assert sel.by_layer == {2: {1, 2, 3}, 3: {0}, 4: {9}}

    def test_invert_over_an_element_count(self):
        """invert() complements the layer's selection against n elements."""
        sel = Selection()
        sel.add(1, [0, 2])
        sel.invert(1, 5)
        assert sel.by_layer == {1: {1, 3, 4}}

    def test_invert_of_everything_clears_the_layer(self):
        """Inverting a full selection leaves no empty key behind."""
        sel = Selection()
        sel.add(1, [0, 1])
        sel.invert(1, 2)
        assert sel.by_layer == {}

    def test_invert_of_nothing_selects_all(self):
        """Inverting an untouched layer selects every element."""
        sel = Selection()
        sel.invert(1, 3)
        assert sel.by_layer == {1: {0, 1, 2}}

    def test_count_layers_and_contains(self):
        """count()/layers()/contains() report the selection accurately."""
        sel = Selection()
        sel.add(1, [0, 1, 2])
        sel.add(2, [5])
        assert sel.count() == 4
        assert len(sel) == 4
        assert sel.count_in(1) == 3
        assert sel.count_in(99) == 0
        assert sel.layers() == [1, 2]
        assert sel.contains(1, 2)
        assert not sel.contains(1, 9)
        assert not sel.contains(99, 0)

    def test_indices_are_sorted(self):
        """indices() returns a sorted int64 array."""
        sel = Selection()
        sel.add(1, [5, 0, 3])
        idx = sel.indices(1)
        assert idx.tolist() == [0, 3, 5]
        assert idx.dtype == np.int64
        assert sel.indices(99).size == 0

    def test_as_mask(self):
        """as_mask() marks exactly the selected elements."""
        sel = Selection()
        sel.add(1, [0, 3])
        mask = sel.as_mask(1, 5)
        assert mask.dtype == bool
        assert mask.tolist() == [True, False, False, True, False]
        assert sel.as_mask(99, 3).tolist() == [False, False, False]

    def test_as_mask_ignores_stale_out_of_range_indices(self):
        """A selection outliving a shrunken layer must clip, not raise."""
        sel = Selection()
        sel.add(1, [0, 99])
        assert sel.as_mask(1, 2).tolist() == [True, False]
        assert sel.as_mask(1, 0).tolist() == []

    def test_as_mask_selects_the_data_it_names(self):
        """The mask is usable directly to extract the selected points."""
        x = np.arange(10.0)
        sel = Selection()
        sel.add(1, [2, 7])
        assert x[sel.as_mask(1, len(x))].tolist() == [2.0, 7.0]

    def test_clear_and_discard_layer(self):
        """clear() empties everything; discard_layer() drops one layer."""
        sel = Selection()
        sel.add(1, [0])
        sel.add(2, [0])
        sel.discard_layer(1)
        assert sel.layers() == [2]
        sel.discard_layer(404)  # unknown layer is a no-op
        sel.clear()
        assert sel.is_empty()
        assert sel.count() == 0

    def test_copy_is_independent(self):
        """copy() must not alias the per-layer sets."""
        sel = Selection()
        sel.add(1, [0])
        other = sel.copy()
        other.add(1, [1])
        other.add(2, [0])
        assert sel.by_layer == {1: {0}}
        assert other.by_layer == {1: {0, 1}, 2: {0}}

    def test_repr_reports_size(self):
        """repr() summarises element and layer counts."""
        sel = Selection()
        sel.add(1, [0, 1])
        assert "2 elements" in repr(sel)
        assert "1 layers" in repr(sel)

    def test_interaction_state_carries_an_independent_selection(self):
        """Each InteractionState gets its own Selection, not a shared default."""
        from glplot.core.legacy import InteractionState

        a, b = InteractionState(), InteractionState()
        a.selection.add(1, [0])
        assert b.selection.is_empty()
        assert a.selection.count() == 1


class TestOffsetTable:
    """Test the picking id offset table."""

    def test_table_mirrors_draw_pick_scene(self):
        """Only layers draw_pick_scene actually draws contribute id ranges."""
        scene = SceneData()
        scene.layers = [
            _layer(10, "scatter", 5),
            _layer(11, "scatter", 0),  # empty
            _layer(12, "line_family", 3),
            _layer(13, "polyline", 1),
            _layer(14, "scatter", 4, visible=False),  # invisible
            _layer(15, "patch", 1),
            _layer(16, "scatter", 2, has_gl=False),  # never uploaded
            _layer(17, "scatter", 2),
        ]
        layers, starts, counts = PickingManager._offset_table(scene)
        assert [ly.layer_id for ly in layers] == [10, 12, 13, 15, 17]
        assert starts.tolist() == [0, 5, 8, 9, 10]
        assert counts.tolist() == [5, 3, 1, 1, 2]

    def test_starts_are_strictly_increasing(self):
        """searchsorted decoding requires unambiguous, strictly ordered starts."""
        scene = SceneData()
        scene.layers = [_layer(i, "scatter", i + 1) for i in range(6)]
        _, starts, _ = PickingManager._offset_table(scene)
        assert np.all(np.diff(starts) > 0)

    def test_empty_scene_yields_an_empty_table(self):
        """A scene with no pickable layers decodes nothing."""
        layers, starts, counts = PickingManager._offset_table(SceneData())
        assert layers == []
        assert starts.size == 0 and counts.size == 0


class TestDecodeIds:
    """Test the vectorised id decode."""

    def _table(self):
        scene = SceneData()
        scene.layers = [
            _layer(10, "scatter", 5),
            _layer(12, "line_family", 3),
            _layer(13, "polyline", 1),
        ]
        return PickingManager._offset_table(scene)

    def test_every_id_decodes_to_its_owner(self):
        """Exhaustively, ids 1..total map to the right (layer, element)."""
        layers, starts, counts = self._table()
        total = int(starts[-1] + counts[-1])
        ids = np.arange(1, total + 1, dtype=np.int64)
        owner, element = PickingManager._decode_ids(ids, starts, counts)
        got = [(layers[int(j)].layer_id, int(e)) for j, e in zip(owner, element)]
        expected = [(10, 0), (10, 1), (10, 2), (10, 3), (10, 4)]
        expected += [(12, 0), (12, 1), (12, 2), (13, 0)]
        assert got == expected

    def test_ids_are_one_based(self):
        """Id 1 is the first element; 0 means 'nothing'."""
        _, starts, counts = self._table()
        owner, element = PickingManager._decode_ids(np.array([1]), starts, counts)
        assert (int(owner[0]), int(element[0])) == (0, 0)
        owner, _ = PickingManager._decode_ids(np.array([0]), starts, counts)
        assert int(owner[0]) == -1

    def test_out_of_range_ids_are_unowned(self):
        """Negative, zero and past-the-end ids decode to owner -1."""
        _, starts, counts = self._table()
        ids = np.array([-5, 0, 10, 11, 9999], dtype=np.int64)
        owner, _ = PickingManager._decode_ids(ids, starts, counts)
        assert owner.tolist() == [-1, -1, -1, -1, -1]

    def test_decode_against_an_empty_table(self):
        """Decoding with no layers owns nothing and does not raise."""
        owner, element = PickingManager._decode_ids(
            np.array([1, 2]), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        )
        assert owner.tolist() == [-1, -1]
        assert element.tolist() == [0, 0]

    def test_decode_is_vectorised_over_many_ids(self):
        """A large id set decodes in one shot, not per element."""
        _, starts, counts = self._table()
        ids = np.random.randint(1, 10, size=100_000).astype(np.int64)
        owner, element = PickingManager._decode_ids(ids, starts, counts)
        assert owner.shape == ids.shape
        assert np.all(owner >= 0)


class _FakeTargetPicker(PickingManager):
    """PickingManager reading a synthetic in-memory pick buffer instead of a GPU."""

    def __init__(self, buffer):
        super().__init__(EngineOptions())
        h, w = buffer.shape
        self.buffer = buffer  # GL row order: row 0 is the BOTTOM of the window
        self.target = GLOffscreenTarget(fbo=1, tex=1, width=w, height=h)
        self.last_read = None

    def install(self, monkeypatch):
        monkeypatch.setattr(picking_mod, "glBindFramebuffer", lambda *a: None)
        monkeypatch.setattr(picking_mod, "glPixelStorei", lambda *a: None)
        monkeypatch.setattr(picking_mod, "glReadPixels", self._read)
        return self

    def _read(self, x, y, w, h, fmt, typ):
        self.last_read = (x, y, w, h)
        return np.ascontiguousarray(self.buffer[y : y + h, x : x + w], dtype=np.int32)


@needs_gl_symbols
class TestPickRectReadback:
    """Test rect readback geometry against a synthetic pick buffer."""

    W, H = 16, 12

    def _scene(self):
        scene = SceneData()
        scene.layers = [_layer(77, "scatter", self.W * self.H)]
        return scene

    def _picker(self, monkeypatch):
        # id = 1 + x + y_gl * W, so element_idx == id - 1 decodes to the pixel.
        ys, xs = np.mgrid[0 : self.H, 0 : self.W]
        return _FakeTargetPicker((1 + xs + ys * self.W).astype(np.int32)).install(monkeypatch)

    def test_rect_returns_exactly_the_enclosed_ids(self, monkeypatch):
        """A window rect decodes to the ids of the pixels it covers."""
        pm = self._picker(monkeypatch)
        got = pm.pick_rect_readback(2.5, 3.5, 4.5, 5.5, self._scene())
        # Window rows 3..5 are GL rows H-1-3=8 down to H-1-5=6; x stays 2..4.
        expected = {(1 + x + y * self.W) - 1 for y in (6, 7, 8) for x in (2, 3, 4)}
        assert got == {77: expected}
        assert len(got[77]) == 9

    def test_y_is_flipped(self, monkeypatch):
        """The TOP window row must read the HIGHEST GL row, and vice versa."""
        pm = self._picker(monkeypatch)
        scene = self._scene()
        top = pm.pick_rect_readback(0.5, 0.5, self.W - 0.5, 0.5, scene)
        assert {e // self.W for e in top[77]} == {self.H - 1}
        bottom = pm.pick_rect_readback(0.5, self.H - 0.5, self.W - 0.5, self.H - 0.5, scene)
        assert {e // self.W for e in bottom[77]} == {0}

    def test_every_window_row_inverse_maps_exactly(self, monkeypatch):
        """Row r (sampled at its centre) maps to GL row H-1-r, none missing."""
        pm = self._picker(monkeypatch)
        scene = self._scene()
        seen = {}
        for r in range(self.H):
            hit = pm.pick_rect_readback(0.5, r + 0.5, self.W - 0.5, r + 0.5, scene)
            rows = {e // self.W for e in hit[77]}
            assert len(rows) == 1
            seen[r] = rows.pop()
        assert seen == {r: self.H - 1 - r for r in range(self.H)}

    def test_x_is_not_flipped(self, monkeypatch):
        """x passes straight through; only y is flipped."""
        pm = self._picker(monkeypatch)
        scene = self._scene()
        for col in (0, 1, self.W - 1):
            hit = pm.pick_rect_readback(col + 0.5, 0.5, col + 0.5, self.H - 0.5, scene)
            assert {e % self.W for e in hit[77]} == {col}

    def test_corners_may_be_given_in_any_order(self, monkeypatch):
        """A drag up-left selects the same rect as a drag down-right."""
        pm = self._picker(monkeypatch)
        scene = self._scene()
        a = pm.pick_rect_readback(2.5, 3.5, 4.5, 5.5, scene)
        for corners in [(4.5, 5.5, 2.5, 3.5), (4.5, 3.5, 2.5, 5.5), (2.5, 5.5, 4.5, 3.5)]:
            assert pm.pick_rect_readback(*corners, scene) == a

    def test_zero_area_rect_reads_a_single_pixel(self, monkeypatch):
        """A degenerate rect falls back to one pixel, not an empty read."""
        pm = self._picker(monkeypatch)
        got = pm.pick_rect_readback(5.5, 6.5, 5.5, 6.5, self._scene())
        assert pm.last_read[2:] == (1, 1)
        assert got == {77: {(1 + 5 + (self.H - 1 - 6) * self.W) - 1}}

    def test_rect_is_clamped_to_the_target(self, monkeypatch):
        """An overshooting drag clamps instead of reading out of bounds."""
        pm = self._picker(monkeypatch)
        got = pm.pick_rect_readback(-500, -500, 500, 500, self._scene())
        assert len(got[77]) == self.W * self.H
        x, y, w, h = pm.last_read
        assert (x, y) == (0, 0)
        assert w <= self.W and h <= self.H

    def test_rect_entirely_outside_selects_nothing(self, monkeypatch):
        """A fully off-target rect must not clamp onto a real edge pixel."""
        pm = self._picker(monkeypatch)
        scene = self._scene()
        for corners in [
            (-50, -50, -10, -10),
            (self.W + 5, 5, self.W + 20, 20),
            (5, self.H + 5, 20, self.H + 20),
        ]:
            assert pm.pick_rect_readback(*corners, scene) == {}

    def test_empty_pick_buffer_selects_nothing(self, monkeypatch):
        """Zeroes mean 'nothing drawn there'."""
        pm = _FakeTargetPicker(np.zeros((self.H, self.W), dtype=np.int32)).install(monkeypatch)
        assert pm.pick_rect_readback(0.5, 0.5, 9.5, 9.5, self._scene()) == {}

    def test_rect_over_the_area_cap_is_refused(self, monkeypatch):
        """The area cap keeps a pathological readback from running."""
        pm = self._picker(monkeypatch)
        got = pm.pick_rect_readback(0.5, 0.5, 9.5, 9.5, self._scene(), max_pixels=4)
        assert got == {}

    def test_multiple_layers_decode_into_separate_sets(self, monkeypatch):
        """Ids spanning two layers split by layer_id with per-layer indices."""
        pm = self._picker(monkeypatch)
        scene = SceneData()
        scene.layers = [
            _layer(1, "scatter", 10),  # ids 1..10
            _layer(2, "line_family", 5),  # ids 11..15
        ]
        # GL row 0 (window bottom) holds ids 1..16 across x.
        got = pm.pick_rect_readback(0.5, self.H - 0.5, 14.5, self.H - 0.5, scene)
        assert got == {1: set(range(10)), 2: set(range(5))}

    def test_uploaded_polyline_decodes_as_one_element(self, monkeypatch):
        """polyline/patch contribute a single element for the whole layer."""
        pm = self._picker(monkeypatch)
        scene = SceneData()
        scene.layers = [_layer(4, "polyline", 1)]  # id 1 only
        got = pm.pick_rect_readback(0.5, self.H - 0.5, 0.5, self.H - 0.5, scene)
        assert got == {4: {0}}

    def test_readback_with_no_target_is_safe(self):
        """A picker that was never initialized reads nothing rather than crashing."""
        pm = PickingManager(EngineOptions())
        assert pm.pick_rect_readback(0, 0, 10, 10, SceneData()) == {}


@needs_real_glfw
class TestMarqueeGesture:
    """Test the engine's shift+drag marquee wiring (headless, no GL)."""

    def _plot(self, monkeypatch, rect_result=None):
        from glplot import engine as eng
        from glplot.engine import GPULinePlot

        plot = GPULinePlot()
        # Retina: logical 800x600 over a 1600x1200 framebuffer (DPR 2).
        plot.width, plot.height = 800, 600
        plot.fb_width, plot.fb_height = 1600, 1200
        cursor = [0.0, 0.0]
        monkeypatch.setattr(eng.glfw, "get_cursor_pos", lambda w: tuple(cursor))
        monkeypatch.setattr(eng.glfw, "get_time", lambda: 0.0)
        monkeypatch.setattr(plot.hud, "wants_mouse", lambda: False)
        monkeypatch.setattr(plot.hud, "on_mouse_button", lambda *a: None)
        calls = {"drew": 0, "rect": None}

        def fake_draw(*a, **k):
            calls["drew"] += 1

        def fake_rect(x0, y0, x1, y1, scene):
            calls["rect"] = (x0, y0, x1, y1)
            return {} if rect_result is None else rect_result

        monkeypatch.setattr(plot.picking, "draw_pick_scene", fake_draw)
        monkeypatch.setattr(plot.picking, "pick_rect_readback", fake_rect)
        return plot, cursor, calls

    def _press(self, plot, cursor, x, y, mods):
        cursor[:] = [x, y]
        plot._on_mouse_button(None, _glfw.MOUSE_BUTTON_LEFT, _glfw.PRESS, mods)

    def _release(self, plot, cursor, x, y, mods):
        cursor[:] = [x, y]
        plot._on_mouse_button(None, _glfw.MOUSE_BUTTON_LEFT, _glfw.RELEASE, mods)

    def test_shift_drag_enters_marquee_mode(self, monkeypatch):
        """Shift+Drag is the marquee now, no longer a layer move."""
        plot, cursor, _ = self._plot(monkeypatch)
        self._press(plot, cursor, 100, 50, _glfw.MOD_SHIFT)
        assert plot.interaction.drag_mode == "marquee"

    def test_marquee_rect_is_dpr_scaled(self, monkeypatch):
        """The readback rect is in framebuffer pixels, not logical units."""
        plot, cursor, calls = self._plot(monkeypatch, {5: {1, 2}})
        self._press(plot, cursor, 100, 50, _glfw.MOD_SHIFT)
        plot._on_cursor(None, 200, 130)
        self._release(plot, cursor, 200, 130, _glfw.MOD_SHIFT)
        assert calls["rect"] == (200.0, 100.0, 400.0, 260.0)

    def test_marquee_draws_the_current_view_before_reading_back(self, monkeypatch):
        """The pick target must hold this view before the rect is read."""
        plot, cursor, calls = self._plot(monkeypatch, {5: {1}})
        self._press(plot, cursor, 10, 10, _glfw.MOD_SHIFT)
        plot._on_cursor(None, 90, 90)
        self._release(plot, cursor, 90, 90, _glfw.MOD_SHIFT)
        assert calls["drew"] == 1

    def test_marquee_populates_the_selection(self, monkeypatch):
        """A marquee release replaces the selection with everything it covered."""
        plot, cursor, _ = self._plot(monkeypatch, {5: {1, 2, 3}, 9: {0}})
        self._press(plot, cursor, 10, 10, _glfw.MOD_SHIFT)
        plot._on_cursor(None, 90, 90)
        self._release(plot, cursor, 90, 90, _glfw.MOD_SHIFT)
        assert plot.interaction.selection.by_layer == {5: {1, 2, 3}, 9: {0}}
        assert plot.interaction.selection.count() == 4

    def test_plain_marquee_replaces_the_previous_selection(self, monkeypatch):
        """Without Alt, a marquee is a fresh selection."""
        plot, cursor, _ = self._plot(monkeypatch, {7: {42}})
        plot.interaction.selection.set(1, [0, 1, 2])
        self._press(plot, cursor, 10, 10, _glfw.MOD_SHIFT)
        plot._on_cursor(None, 90, 90)
        self._release(plot, cursor, 90, 90, _glfw.MOD_SHIFT)
        assert plot.interaction.selection.by_layer == {7: {42}}

    def test_alt_makes_the_marquee_additive(self, monkeypatch):
        """Shift+Alt+Drag adds to the selection instead of replacing it."""
        plot, cursor, _ = self._plot(monkeypatch, {7: {43}, 8: {0}})
        plot.interaction.selection.set(7, [42])
        self._press(plot, cursor, 10, 10, _glfw.MOD_SHIFT | _glfw.MOD_ALT)
        assert plot.interaction.selection_additive
        plot._on_cursor(None, 90, 90)
        self._release(plot, cursor, 90, 90, _glfw.MOD_SHIFT | _glfw.MOD_ALT)
        assert plot.interaction.selection.by_layer == {7: {42, 43}, 8: {0}}

    def test_marquee_does_not_move_the_camera(self, monkeypatch):
        """The view must not pan under the rubber band."""
        plot, cursor, _ = self._plot(monkeypatch, {5: {1}})
        before = (plot.camera.cx, plot.camera.cy)
        self._press(plot, cursor, 100, 100, _glfw.MOD_SHIFT)
        plot._on_cursor(None, 300, 250)
        assert (plot.camera.cx, plot.camera.cy) == before
        self._release(plot, cursor, 300, 250, _glfw.MOD_SHIFT)
        assert (plot.camera.cx, plot.camera.cy) == before

    def test_shift_click_without_drag_defers_a_single_pick(self, monkeypatch):
        """A click under the drag threshold picks one element on release."""
        plot, cursor, calls = self._plot(monkeypatch)
        plot.frame.dirty_pick = False  # FrameState starts dirty; clear the baseline
        self._press(plot, cursor, 300, 300, _glfw.MOD_SHIFT)
        assert not plot.frame.dirty_pick, "the press must not request a pick yet"
        assert plot.interaction.pick_press_requested
        self._release(plot, cursor, 301, 300, _glfw.MOD_SHIFT)
        assert plot.frame.dirty_pick
        assert plot.interaction.explicit_pick_requested
        assert plot.interaction.selection_pick_requested
        assert calls["rect"] is None  # no rect pick for a click
        assert not plot.interaction.pick_press_requested

    def test_plain_drag_still_pans(self, monkeypatch):
        """Plain drag must keep panning; nothing regresses for existing users."""
        plot, cursor, _ = self._plot(monkeypatch)
        before = (plot.camera.cx, plot.camera.cy)
        self._press(plot, cursor, 400, 300, 0)
        assert plot.interaction.drag_mode == "pan"
        plot._on_cursor(None, 450, 320)
        assert (plot.camera.cx, plot.camera.cy) != before

    def test_ctrl_drag_still_moves_a_picked_layer(self, monkeypatch):
        """Ctrl+Drag remains the layer-move gesture."""
        plot, cursor, _ = self._plot(monkeypatch)
        plot.add_scatter(
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.ones((2, 4), dtype=np.float32),
            label="s",
        )
        layer = plot.scene.layers[-1]
        monkeypatch.setattr(
            plot.picking,
            "pick_readback",
            lambda sx, sy, scene: {
                "type": "scatter",
                "layer_id": layer.layer_id,
                "element_idx": 0,
                "layer": layer,
            },
        )
        self._press(plot, cursor, 400, 300, _glfw.MOD_CONTROL)
        assert plot.interaction.drag_mode == "move"
        before = layer.translation
        plot._on_cursor(None, 450, 300)
        assert layer.translation != before

    def test_ctrl_drag_does_not_touch_the_selection(self, monkeypatch):
        """Resolving a move gesture must not rewrite the user's selection."""
        plot, cursor, _ = self._plot(monkeypatch)
        plot.interaction.selection.set(99, [1, 2])
        monkeypatch.setattr(
            plot.picking,
            "pick_readback",
            lambda sx, sy, scene: {
                "type": "scatter",
                "layer_id": 3,
                "element_idx": 0,
                "layer": None,
            },
        )
        self._press(plot, cursor, 400, 300, _glfw.MOD_CONTROL)
        assert plot.interaction.selection.by_layer == {99: {1, 2}}


@needs_real_glfw
class TestSinglePickFeedsSelection:
    """Test that the existing single-pick path populates the Selection."""

    def _plot(self, monkeypatch, hit):
        from glplot.engine import GPULinePlot

        plot = GPULinePlot()
        plot.interaction.last_mouse = (10.0, 10.0)
        monkeypatch.setattr(plot.picking, "draw_pick_scene", lambda *a, **k: None)
        monkeypatch.setattr(plot.picking, "pick_readback", lambda sx, sy, scene: hit)
        return plot

    def test_pick_sets_one_element(self, monkeypatch):
        """An explicit pick selects exactly the picked element."""
        hit = {"type": "scatter", "layer_id": 3, "element_idx": 77, "layer": None}
        plot = self._plot(monkeypatch, hit)
        plot._run_picking_pass(update_selection=True)
        assert plot.interaction.selection.by_layer == {3: {77}}
        assert plot.interaction.selected_layer_id == 3

    def test_pick_replaces_by_default(self, monkeypatch):
        """A non-additive pick drops the previous selection."""
        hit = {"type": "scatter", "layer_id": 3, "element_idx": 1, "layer": None}
        plot = self._plot(monkeypatch, hit)
        plot.interaction.selection.set(8, [0, 1])
        plot._run_picking_pass(update_selection=True)
        assert plot.interaction.selection.by_layer == {3: {1}}

    def test_additive_pick_toggles(self, monkeypatch):
        """Shift+Alt+Click toggles the element within the selection."""
        hit = {"type": "scatter", "layer_id": 3, "element_idx": 1, "layer": None}
        plot = self._plot(monkeypatch, hit)
        plot.interaction.selection_additive = True
        plot.interaction.selection.set(3, [0])
        plot._run_picking_pass(update_selection=True)
        assert plot.interaction.selection.by_layer == {3: {0, 1}}
        plot._run_picking_pass(update_selection=True)
        assert plot.interaction.selection.by_layer == {3: {0}}

    def test_pick_without_update_selection_leaves_it_alone(self, monkeypatch):
        """Hover/scroll/resize re-picks must not rewrite the selection."""
        hit = {"type": "scatter", "layer_id": 3, "element_idx": 1, "layer": None}
        plot = self._plot(monkeypatch, hit)
        plot.interaction.selection.set(8, [0])
        plot._run_picking_pass()
        assert plot.interaction.selection.by_layer == {8: {0}}
        assert plot.interaction.selected_layer_id == 3  # legacy behaviour intact

    def test_miss_clears_the_selection_only_when_explicit(self, monkeypatch):
        """Shift+Click on empty space deselects; a hover miss does not."""
        plot = self._plot(monkeypatch, None)
        plot.interaction.selection.set(8, [0])
        plot._run_picking_pass(update_selection=False)
        assert plot.interaction.selection.by_layer == {8: {0}}
        plot._run_picking_pass(update_selection=True)
        assert plot.interaction.selection.is_empty()

    def test_additive_miss_keeps_the_selection(self, monkeypatch):
        """Shift+Alt+Click on empty space must not wipe the selection."""
        plot = self._plot(monkeypatch, None)
        plot.interaction.selection.set(8, [0])
        plot.interaction.selection_additive = True
        plot._run_picking_pass(update_selection=True)
        assert plot.interaction.selection.by_layer == {8: {0}}
