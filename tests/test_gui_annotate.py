"""Test the annotation helpers and the annotate panel's tool mode.

The layerops half is pure logic and runs with no OpenGL and no GPU: ``GPULinePlot()``
constructs without a window, so the scene-mutation paths are exercised against a real
engine rather than a mock. The panel half uses the sanctioned headless imgui harness
(CONTRACT §2.10) — ``create_context`` plus a ``new_frame``/``end_frame`` pair, which
needs no GL context either.
"""

from __future__ import annotations

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.engine import GPULinePlot
from glplot.gui import layerops
from glplot.gui.commands import CommandQueue
from glplot.gui.history import UndoStack


@pytest.fixture(autouse=True)
def clean_state():
    """The repo-wide pyplot state reset."""
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


@pytest.fixture
def plot():
    """A real engine with one line layer and a framed view. No window, no GL."""
    p = GPULinePlot()
    p.width, p.height = 900, 700
    xs = np.linspace(0.0, 10.0, 50)
    p.add_line_strip(xs, np.sin(xs), label="sin")
    p.autoscale()
    return p


class FakeWorkspace:
    """The three fields the tool controller reads off a workspace.

    A real ``Workspace`` needs imgui to construct; the tool logic does not, and testing
    it through a stand-in is what keeps these tests running on a GL-less host.
    """

    def __init__(self, plot):
        self.plot = plot
        self.queue = CommandQueue()
        self.undo = UndoStack()
        self.hud = None
        self._frame = 0

    def drain(self):
        """Run the queue against the plot, as ``_main_loop`` does at the frame top."""
        self.queue.drain(self.plot)


class TestAnnotationRegistry:
    """Test the annotation kind registry and its honesty about what it cannot do."""

    def test_every_key_resolves_to_a_spec(self):
        """Each advertised kind has a spec whose key matches its registry entry."""
        for key in layerops.ANNOTATION_KEYS:
            assert layerops.annotation_spec(key).key == key

    def test_unknown_kind_raises(self):
        """An unknown kind raises rather than silently producing nothing."""
        with pytest.raises(ValueError, match="annotation kind must be one of"):
            layerops.annotation_spec("ellipse")

    def test_unreachable_shapes_are_documented_not_faked(self):
        """The shapes left out are recorded with a reason, and are not in the registry."""
        assert layerops.UNREACHABLE_ANNOTATIONS
        for name, reason in layerops.UNREACHABLE_ANNOTATIONS.items():
            assert name not in layerops.ANNOTATION_KEYS
            assert len(reason) > 40, f"{name} needs a real reason, not a shrug"

    def test_text_fontsize_is_declared_export_only(self):
        """text_size_px has no GL consumer; the spec must not claim it is live."""
        spec = layerops.annotation_spec("text")
        assert "text_size_px" in spec.export_only_fields
        assert "text_size_px" not in spec.live_fields

    def test_rect_does_not_offer_dead_edge_color(self):
        """patch.py reads face_color only, so edge_color must not be advertised."""
        spec = layerops.annotation_spec("rect")
        assert "face_color" in spec.live_fields
        assert "edge_color" not in spec.live_fields


class TestTextAnnotations:
    """Test text annotation creation, editing and the legacy-mirror contract."""

    def test_add_reaches_scene_layers(self, plot):
        """A text annotation lands in scene.layers as a tagged TextLayer."""
        layer = layerops.add_text_annotation(plot, 2.0, 3.0, "Peak", fontsize=14)
        assert layer in plot.scene.layers
        assert layer.layer_type == "text"
        assert layer.text == "Peak"
        assert layerops.annotation_kind(layer) == "text"

    def test_add_sets_the_dirty_flags_add_text_forgets(self, plot):
        """engine.add_text sets only dirty_ui; the helper must repair dirty_scene.

        Text draws in the scene pass (engine.py:979), which is gated on dirty_scene —
        without this the label does not appear until an unrelated event wakes the loop.
        """
        plot.frame.dirty_scene = False
        plot.cache.refresh_requested = False
        plot.cache.capture_window = (0.0, 1.0, 0.0, 1.0)

        layerops.add_text_annotation(plot, 1.0, 1.0, "hi")

        assert plot.frame.dirty_scene is True
        assert plot.cache.refresh_requested is True
        assert plot.cache.capture_window is None

    def test_add_through_the_queue_is_the_supported_path(self, plot):
        """A queued placement reaches the scene with the epilogue applied (§1.5)."""
        ws = FakeWorkspace(plot)
        ws.queue.submit(lambda: layerops.add_text_annotation(plot, 1.0, 2.0, "queued"))
        assert not any(la.layer_type == "text" for la in plot.scene.layers)

        ws.drain()

        texts = [la for la in plot.scene.layers if la.layer_type == "text"]
        assert len(texts) == 1
        assert texts[0].text == "queued"
        assert plot.frame.dirty_scene is True
        assert plot.cache.refresh_requested is True

    def test_explicit_label_is_always_passed(self, plot):
        """The engine's index-based default label is unstable and must never be used."""
        layer = layerops.add_text_annotation(plot, 0.0, 0.0, "Some annotation text here")
        assert layer.label
        assert not layer.label.startswith("Text: ") or "Some annotation" in layer.label

    def test_update_edits_in_place(self, plot):
        """Editing rewrites the layer the renderer reads, not a copy."""
        layer = layerops.add_text_annotation(plot, 1.0, 1.0, "before")
        layerops.update_text_annotation(plot, layer, text="after", x=5.0, y=6.0)
        assert (layer.text, layer.x, layer.y) == ("after", 5.0, 6.0)

    def test_update_keeps_the_legacy_text_mirror_in_sync(self, plot):
        """scene.texts must track the edit or the removal purge silently misses it.

        _purge_legacy_mirrors matches the mirror row by x/y/str, so a stale row is not
        cosmetic: it pins the string forever and can match a *different* text layer that
        later occupies the old position.
        """
        layer = layerops.add_text_annotation(plot, 1.0, 1.0, "before")
        assert plot.scene.texts[0]["str"] == "before"

        layerops.update_text_annotation(plot, layer, text="after", x=9.0)

        assert plot.scene.texts[0]["str"] == "after"
        assert plot.scene.texts[0]["x"] == 9.0

        layerops.remove_layer(plot, None, layer)
        assert plot.scene.texts == []

    def test_position_applies_translation(self, plot):
        """renderers/text.py projects (x + tx, y + ty); the readout must agree."""
        layer = layerops.add_text_annotation(plot, 1.0, 2.0, "t")
        layer.translation = (10.0, 20.0)
        assert layerops.text_annotation_position(layer) == (11.0, 22.0)

    def test_update_ignores_a_non_text_layer(self, plot):
        """A wrong-typed layer is a no-op, not an AttributeError mid-drain."""
        line = plot.scene.layers[0]
        layerops.update_text_annotation(plot, line, text="nope")
        assert not hasattr(line, "text") or line.text != "nope"


class TestShapeAnnotations:
    """Test the shapes that are genuinely reachable from existing primitives."""

    def test_arrow_is_a_shaft_and_a_head_in_one_group(self, plot):
        """pyplot.arrow's structure: a polyline plus a filled triangle, grouped as one."""
        layers = layerops.add_arrow_annotation(plot, 0.0, 0.0, 2.0, 1.0)
        assert [la.layer_type for la in layers] == ["polyline", "patch"]
        groups = {layerops.annotation_group(la) for la in layers}
        assert len(groups) == 1, "both halves must share a group id or the list shows two rows"

    def test_arrow_head_geometry_matches_pyplot(self, plot):
        """The head is the tip plus two base corners, perpendicular to the shaft."""
        head = layerops.arrow_head_geometry(0.0, 0.0, 1.0, 0.0, head_width=0.2, head_length=0.4)
        assert head.shape == (3, 2)
        assert np.allclose(head[0], [1.0, 0.0])
        assert np.allclose(sorted(head[1:, 1]), [-0.1, 0.1])

    def test_zero_length_arrow_has_no_head(self, plot):
        """A degenerate shaft yields no head rather than a divide-by-zero."""
        assert layerops.arrow_head_geometry(1.0, 1.0, 0.0, 0.0, head_width=1, head_length=1) is None
        layers = layerops.add_arrow_annotation(plot, 1.0, 1.0, 0.0, 0.0)
        assert len(layers) == 1

    def test_patch_arrays_are_the_dtypes_add_patch_requires(self, plot):
        """add_patch does no dtype coercion (§1.5); float32/uint32 are mandatory."""
        layers = layerops.add_arrow_annotation(plot, 0.0, 0.0, 1.0, 1.0)
        head = [la for la in layers if la.layer_type == "patch"][0]
        assert head.vertices.dtype == np.float32
        assert head.indices.dtype == np.uint32
        assert head.vertices.flags["C_CONTIGUOUS"]

    def test_rect_geometry_normalises_its_corners(self):
        """A box dragged right-to-left is the same rect, not a flipped winding."""
        forward, _ = layerops.rect_annotation_geometry(0.0, 0.0, 2.0, 3.0)
        backward, _ = layerops.rect_annotation_geometry(2.0, 3.0, 0.0, 0.0)
        assert np.allclose(forward, backward)

    def test_rect_is_two_triangles(self, plot):
        """add_patch mode 'triangles' with six indices over four corners."""
        layer = layerops.add_rect_annotation(plot, 0.0, 0.0, 1.0, 1.0)
        assert layer.mode == "triangles"
        assert layer.vertices.shape == (4, 2)
        assert layer.indices.tolist() == [0, 1, 2, 0, 2, 3]
        assert layer.vertices.dtype == np.float32
        assert layer.indices.dtype == np.uint32

    def test_hline_and_vline_span_the_requested_range(self, plot):
        """hlines/vlines semantics: a finite rule with an explicit span."""
        h = layerops.add_line_annotation(plot, kind="hline", value=2.0, lo=0.0, hi=10.0)
        assert np.allclose(h.pts[:, 1], [2.0, 2.0])
        assert np.allclose(h.pts[:, 0], [0.0, 10.0])

        v = layerops.add_line_annotation(plot, kind="vline", value=3.0, lo=-1.0, hi=1.0)
        assert np.allclose(v.pts[:, 0], [3.0, 3.0])
        assert np.allclose(v.pts[:, 1], [-1.0, 1.0])

    def test_line_annotation_rejects_an_unknown_kind(self, plot):
        """Only hline/vline; anything else is a caller bug and must say so."""
        with pytest.raises(ValueError, match="kind must be 'hline' or 'vline'"):
            layerops.add_line_annotation(plot, kind="diagonal", value=0.0, lo=0.0, hi=1.0)


class TestAnnotationGrouping:
    """Test that the list sees one annotation per row, not one per layer."""

    def test_groups_collapse_multi_layer_annotations(self, plot):
        """An arrow's two layers are one row; the other kinds are one layer each."""
        layerops.add_text_annotation(plot, 1.0, 1.0, "t")
        layerops.add_arrow_annotation(plot, 0.0, 0.0, 1.0, 1.0)
        layerops.add_rect_annotation(plot, 0.0, 0.0, 1.0, 1.0)

        groups = layerops.annotation_groups(plot)
        assert [(kind, len(layers)) for _g, kind, layers in groups] == [
            ("text", 1),
            ("arrow", 2),
            ("rect", 1),
        ]

    def test_non_annotation_layers_are_not_listed(self, plot):
        """The scene's own data layers must never appear in the annotation list."""
        layerops.add_text_annotation(plot, 1.0, 1.0, "t")
        groups = layerops.annotation_groups(plot)
        assert len(groups) == 1
        assert plot.scene.layers[0] not in groups[0][2]

    def test_is_annotation_is_false_for_plain_layers(self, plot):
        """A line added by the engine is not an annotation."""
        assert layerops.is_annotation(plot.scene.layers[0]) is False
        assert layerops.annotation_kind(plot.scene.layers[0]) is None

    def test_tagging_preserves_other_metadata(self, plot):
        """metadata is load-bearing (§4.4) and the tag must not stomp it."""
        layer = layerops.add_rect_annotation(plot, 0.0, 0.0, 1.0, 1.0)
        layer.metadata["artist"] = "custom"
        layerops.tag_annotation(layer, "rect", layerops.annotation_group(layer))
        assert layer.metadata["artist"] == "custom"


class TestAnnotationStyle:
    """Test restyling across an annotation's parts."""

    def test_style_fans_out_across_fields_each_layer_actually_has(self, plot):
        """An arrow's shaft reads style.color and its head reads face_color.

        set_layer_style raises on an unknown field by design, so a single picker driving
        both halves depends on set_annotation_style filtering per layer.
        """
        layers = layerops.add_arrow_annotation(plot, 0.0, 0.0, 1.0, 1.0, color=(1, 0, 0, 1))
        layerops.set_annotation_style(plot, layers, color=(0, 1, 0, 1), face_color=(0, 1, 0, 1))

        shaft = [la for la in layers if la.layer_type == "polyline"][0]
        head = [la for la in layers if la.layer_type == "patch"][0]
        assert tuple(shaft.style.color) == (0, 1, 0, 1)
        assert tuple(head.style.face_color) == (0, 1, 0, 1)

    def test_style_survives_a_field_the_layer_lacks(self, plot):
        """A text layer has no line_width; passing one must not raise."""
        layer = layerops.add_text_annotation(plot, 0.0, 0.0, "t")
        layerops.set_annotation_style(plot, [layer], line_width=4.0, color=(1, 1, 0, 1))
        assert tuple(layer.style.color) == (1, 1, 0, 1)

    def test_annotation_color_seeds_from_the_real_field(self, plot):
        """The picker seeds from style.color, or face_color when that is all there is."""
        rect = layerops.add_rect_annotation(plot, 0, 0, 1, 1, face_color=(0.2, 0.4, 0.6, 0.5))
        assert layerops.annotation_color([rect]) == pytest.approx((0.2, 0.4, 0.6, 0.5))

        text = layerops.add_text_annotation(plot, 0, 0, "t", color=(1.0, 0.0, 0.0, 1.0))
        assert layerops.annotation_color([text]) == pytest.approx((1.0, 0.0, 0.0, 1.0))

    def test_removing_a_group_removes_every_part(self, plot):
        """Deleting an arrow deletes both halves, through the full §1.6 ritual."""
        layers = layerops.add_arrow_annotation(plot, 0.0, 0.0, 1.0, 1.0)
        layerops.remove_annotation_group(plot, None, layers)
        assert not any(layerops.is_annotation(la) for la in plot.scene.layers)


class TestToolController:
    """Test the tool mode's state machine. Requires imgui to import."""

    def _tool(self, plot):
        annotate = pytest.importorskip("glplot.gui.panels.annotate")
        return annotate, annotate.ToolController(FakeWorkspace(plot))

    def test_default_mode_is_pan(self, plot):
        """Plain-drag must stay pan by default: nothing regresses for existing users."""
        annotate, tool = self._tool(plot)
        assert tool.mode == annotate.TOOL_PAN

    def test_unknown_mode_raises(self, plot):
        """A typo'd mode must not silently arm nothing."""
        _annotate, tool = self._tool(plot)
        with pytest.raises(ValueError, match="unknown tool mode"):
            tool.set_mode("lasso")

    def test_switching_mode_abandons_the_in_flight_gesture(self, plot):
        """A half-drawn rect must not land when the user switches to the text tool."""
        annotate, tool = self._tool(plot)
        tool.set_mode(annotate.TOOL_RECT)
        tool._drag_from = (10.0, 10.0)
        tool.set_mode(annotate.TOOL_TEXT)
        assert tool._drag_from is None

    def test_select_is_not_a_tool_mode(self, plot):
        """Select is deliberately absent — shift+drag is the marquee (see the module doc)."""
        annotate, _tool = self._tool(plot)
        assert "select" not in {spec[0] for spec in annotate.TOOL_SPECS}

    def test_every_tool_icon_exists(self, plot):
        """An unknown icon name draws a placeholder; the rail must not ship one."""
        annotate, _tool = self._tool(plot)
        from glplot.gui import icons

        for _mode, _label, icon, _hint in annotate.TOOL_SPECS:
            assert icon in icons.ICON_SHAPES

    def test_get_tool_is_idempotent_per_workspace(self, plot):
        """The mode must survive a panel being closed and reopened."""
        annotate = pytest.importorskip("glplot.gui.panels.annotate")
        ws = FakeWorkspace(plot)
        first = annotate.get_tool(ws)
        annotate.set_tool_mode(ws, annotate.TOOL_RECT)
        assert annotate.get_tool(ws) is first
        assert annotate.tool_mode(ws) == annotate.TOOL_RECT


class TestToolPlacement:
    """Test that a tool gesture becomes a queued, undoable scene mutation."""

    def test_placement_is_deferred_then_undoable(self, plot):
        """A text commit reaches scene.layers only on drain, and undo removes it."""
        annotate = pytest.importorskip("glplot.gui.panels.annotate")
        ws = FakeWorkspace(plot)
        tool = annotate.get_tool(ws)
        tool.set_mode(annotate.TOOL_TEXT)
        tool._pending_xy = (2.0, 3.0)
        tool._pending_text = "Peak"

        annotate._commit_text(ws, tool)
        assert not any(
            la.layer_type == "text" for la in plot.scene.layers
        ), "must not mutate in draw"

        ws.drain()
        texts = [la for la in plot.scene.layers if la.layer_type == "text"]
        assert len(texts) == 1 and texts[0].text == "Peak"
        assert ws.undo.peek_undo() == "Add text annotation"

        ws.undo.undo()
        assert not any(la.layer_type == "text" for la in plot.scene.layers)

        ws.undo.redo()
        assert len([la for la in plot.scene.layers if la.layer_type == "text"]) == 1

    def test_empty_text_places_nothing(self, plot):
        """Committing a blank box must not add an invisible, unclickable layer."""
        annotate = pytest.importorskip("glplot.gui.panels.annotate")
        ws = FakeWorkspace(plot)
        tool = annotate.get_tool(ws)
        tool._pending_xy = (1.0, 1.0)
        tool._pending_text = "   "
        annotate._commit_text(ws, tool)
        ws.drain()
        assert not any(la.layer_type == "text" for la in plot.scene.layers)

    def test_emptying_an_existing_box_deletes_it(self, plot):
        """A text layer with no string renders nothing and cannot be clicked again."""
        annotate = pytest.importorskip("glplot.gui.panels.annotate")
        ws = FakeWorkspace(plot)
        layer = layerops.add_text_annotation(plot, 1.0, 1.0, "gone soon")
        tool = annotate.get_tool(ws)
        tool._pending_xy = (1.0, 1.0)
        tool._editing = layer
        tool._editing_before = "gone soon"
        tool._pending_text = ""

        annotate._commit_text(ws, tool)
        ws.drain()
        assert layer not in plot.scene.layers

    def test_editing_text_is_undoable(self, plot):
        """An edit restores the previous string exactly."""
        annotate = pytest.importorskip("glplot.gui.panels.annotate")
        ws = FakeWorkspace(plot)
        layer = layerops.add_text_annotation(plot, 1.0, 1.0, "before")
        tool = annotate.get_tool(ws)
        tool._pending_xy = (1.0, 1.0)
        tool._editing = layer
        tool._editing_before = "before"
        tool._pending_text = "after"

        annotate._commit_text(ws, tool)
        ws.drain()
        assert layer.text == "after"

        ws.undo.undo()
        assert layer.text == "before"

    def test_delete_undo_rebuilds_rather_than_reinserts(self, plot):
        """§1.6 frees the layers' GL, so undo must recreate them, geometry intact.

        Re-inserting the same objects would leave every renderer's _gl handle naming a
        freed VAO — the failure this rebuild path exists to avoid.
        """
        annotate = pytest.importorskip("glplot.gui.panels.annotate")
        ws = FakeWorkspace(plot)
        layers = layerops.add_arrow_annotation(plot, 0.0, 0.0, 2.0, 1.0, color=(1, 0, 0, 1))
        head_before = np.array([la for la in layers if la.layer_type == "patch"][0].vertices)

        annotate._delete_group(ws, layers, "Delete annotation")
        ws.drain()
        assert not any(layerops.is_annotation(la) for la in plot.scene.layers)

        ws.undo.undo()
        restored = [la for la in plot.scene.layers if layerops.annotation_kind(la) == "arrow"]
        assert len(restored) == 2
        head_after = [la for la in restored if la.layer_type == "patch"][0].vertices
        assert np.allclose(head_after, head_before)
        assert restored[0] not in layers, "undo must build new layers, not revive freed ones"


class TestToolProjection:
    """Test the screen/world mapping the click-to-place and hit-test depend on."""

    def test_screen_to_world_round_trips_through_world_to_screen(self, plot):
        """The classic placement bug: the box lands where you did not click.

        world_to_screen reproduces renderers/text.py's projection and screen_to_world is
        the camera's own inverse; if they disagree, a placed label drifts from the cursor.
        """
        imgui = pytest.importorskip("imgui")
        annotate = pytest.importorskip("glplot.gui.panels.annotate")
        imgui.create_context()

        for sx, sy in [(100.0, 100.0), (450.0, 350.0), (800.0, 600.0), (123.5, 456.5)]:
            world = annotate.screen_to_world(plot, sx, sy)
            assert world is not None
            back = annotate.world_to_screen(plot, world[0], world[1])
            assert back is not None
            assert back[0] == pytest.approx(sx, abs=1e-2)
            assert back[1] == pytest.approx(sy, abs=1e-2)

    def test_projection_answers_none_without_a_sized_window(self):
        """A zero-sized plot must return None, not divide by zero mid-frame."""
        annotate = pytest.importorskip("glplot.gui.panels.annotate")
        p = GPULinePlot()
        p.width, p.height = 0, 0
        assert annotate.world_to_screen(p, 1.0, 1.0) is None
        assert annotate.screen_to_world(p, 1.0, 1.0) is None


class TestAnnotatePanelDraws:
    """Test the panel body through the headless imgui harness (CONTRACT §2.10)."""

    def _harness(self):
        imgui = pytest.importorskip("imgui")
        imgui.create_context()
        io = imgui.get_io()
        io.display_size = 900, 700
        io.fonts.get_tex_data_as_rgba32()
        io.fonts.texture_id = 1
        io.delta_time = 1 / 60.0
        return imgui

    def test_panel_draws_every_kind_without_raising(self, plot):
        """One of every annotation, each row expanded — exercises every widget branch."""
        imgui = self._harness()
        annotate = pytest.importorskip("glplot.gui.panels.annotate")

        ws = FakeWorkspace(plot)
        panel = annotate.AnnotatePanel(ws)

        layerops.add_text_annotation(plot, 5.0, 0.5, "Peak", fontsize=14)
        layerops.add_arrow_annotation(plot, 1.0, 0.0, 2.0, 0.5)
        layerops.add_line_annotation(plot, kind="hline", value=0.25, lo=0.0, hi=10.0)
        layerops.add_line_annotation(plot, kind="vline", value=3.0, lo=-1.0, hi=1.0)
        layerops.add_rect_annotation(plot, 6.0, -0.5, 8.0, 0.5)

        for group, _kind, _layers in layerops.annotation_groups(plot):
            imgui.new_frame()
            imgui.begin("Annotate")
            panel._selected = group
            panel.draw()
            imgui.end()
            imgui.render()
            assert imgui.get_draw_data() is not None

    def test_panel_draws_when_empty(self, plot):
        """The empty state is a real state — it must not need an annotation to render."""
        imgui = self._harness()
        annotate = pytest.importorskip("glplot.gui.panels.annotate")
        panel = annotate.AnnotatePanel(FakeWorkspace(plot))

        imgui.new_frame()
        imgui.begin("Annotate")
        panel.draw()
        imgui.end()
        imgui.render()

    def test_every_tool_mode_draws_its_overlay_and_preview(self, plot):
        """Each armed tool draws its overlay, live preview and hint without raising."""
        imgui = self._harness()
        annotate = pytest.importorskip("glplot.gui.panels.annotate")
        ws = FakeWorkspace(plot)
        panel = annotate.AnnotatePanel(ws)
        tool = annotate.get_tool(ws)

        for mode, _label, _icon, _hint in annotate.TOOL_SPECS:
            tool.set_mode(mode)
            tool._drag_from, tool._drag_to = (100.0, 100.0), (300.0, 250.0)
            ws._frame += 1
            imgui.new_frame()
            imgui.begin("Annotate")
            panel.draw()
            imgui.end()
            imgui.render()

    def test_pan_mode_submits_no_overlay(self, plot):
        """The no-regression guarantee: in Pan the module must not capture the mouse.

        draw_canvas_layer returning without a begin() is what leaves want_capture_mouse
        False over the canvas, so engine._on_mouse_button does not early-out and plain
        drag still pans.
        """
        imgui = self._harness()
        annotate = pytest.importorskip("glplot.gui.panels.annotate")
        ws = FakeWorkspace(plot)
        tool = annotate.get_tool(ws)
        assert tool.mode == annotate.TOOL_PAN

        io = imgui.get_io()
        for _ in range(3):
            io.mouse_pos = 450, 350
            imgui.new_frame()
            annotate.draw_canvas_layer(ws)
            ws._frame += 1
            imgui.end_frame()
            assert io.want_capture_mouse is False

    def test_armed_tool_captures_the_mouse_from_the_engine(self, plot):
        """The overlay is the whole mechanism: with a tool armed, the engine must not pan.

        engine._on_mouse_button early-outs on hud.wants_mouse(), which is
        io.want_capture_mouse — so this flipping True is what disables the camera while
        a tool is live, with no engine edit at all.
        """
        imgui = self._harness()
        annotate = pytest.importorskip("glplot.gui.panels.annotate")
        ws = FakeWorkspace(plot)
        annotate.set_tool_mode(ws, annotate.TOOL_TEXT)

        io = imgui.get_io()
        captured = []
        for _ in range(3):
            io.mouse_pos = 450, 350
            imgui.new_frame()
            annotate.draw_canvas_layer(ws)
            ws._frame += 1
            imgui.end_frame()
            captured.append(io.want_capture_mouse)

        assert captured[-1] is True, "an armed tool must take the mouse from the camera"

    def test_draw_canvas_layer_is_idempotent_within_a_frame(self, plot):
        """Both the workspace and the panel may call it; the second must be a no-op."""
        imgui = self._harness()
        annotate = pytest.importorskip("glplot.gui.panels.annotate")
        ws = FakeWorkspace(plot)
        annotate.set_tool_mode(ws, annotate.TOOL_RECT)

        imgui.new_frame()
        annotate.draw_canvas_layer(ws)
        annotate.draw_canvas_layer(ws)  # would duplicate the window id if it ran twice
        imgui.end_frame()
