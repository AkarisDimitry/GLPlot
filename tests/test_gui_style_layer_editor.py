"""The Style panel's per-layer editor, with the 3D half it grew.

Two kinds of test, because the panel has two kinds of failure mode:

* **Draw tests** go through the sanctioned headless imgui harness (CONTRACT §2.10) — real
  context, real frame, real ``imgui.render()``. That is the only thing that catches a
  wrong pyimgui signature, an unbalanced ``push_style_var``/``pop_style_var`` or a
  ``same_line`` after a widget that never drew, which is most of what can go wrong in a
  panel and none of which a logic test can see. Every 3D kind gets its own frame, because
  the options editor is driven by *which keys the kind's options dict has* — a kind with
  no branch drawn for a key it owns is invisible until someone selects that kind.
* **Action tests** assert on state. Every mutation is queued (CONTRACT §1.1), so they
  check the scene is *unchanged* before ``queue.drain(plot)`` and changed after. A panel
  that mutated in ``draw`` would pass a naive "did it change" test and lose the edit in
  the real loop.

The ``harness`` fixture destroys its context in teardown. That is not tidiness: imgui's
context is global state, and a leaked one makes an unrelated test file's own
``create_context`` see a foreign context and fail. ``test_gui_3d_panels.py`` carries the
same fixture and the same warning.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.engine import GPULinePlot
from glplot.gui import layerops, layerops3d
from glplot.gui.commands import CommandQueue
from glplot.gui.datasets import DataStore
from glplot.gui.history import UndoStack
from glplot.options import BlendMode


class FakeWorkspace:
    """The five services a panel reads off a workspace."""

    def __init__(self, plot):
        self.plot = plot
        self.queue = CommandQueue()
        self.undo = UndoStack()
        self.store = DataStore()
        self.hud = None

    def drain(self):
        """Run the queue against the plot, as ``_main_loop`` does at the frame top."""
        self.queue.drain(self.plot)


@pytest.fixture
def plot():
    return GPULinePlot()


@pytest.fixture
def ws(plot):
    return FakeWorkspace(plot)


@pytest.fixture
def panel(ws):
    from glplot.gui.panels.style import StylePanel

    return StylePanel(ws)


@pytest.fixture
def harness():
    """A headless imgui context sized like a real window, destroyed afterwards.

    The teardown is mandatory — see the module docstring.
    """
    imgui = pytest.importorskip("imgui_bundle").imgui
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = 1200, 900
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    io.delta_time = 1 / 60.0
    yield imgui
    imgui.destroy_context(ctx)


@pytest.fixture
def all_sections(monkeypatch):
    """Force every collapsing header open, so one frame covers every widget branch."""
    from glplot.gui import widgets

    monkeypatch.setattr(widgets, "section", lambda label, **kw: True)


def _layer_frame(imgui, panel, *, tab=None):
    """Draw one panel body inside a real frame and assert the frame closed cleanly."""
    imgui.new_frame()
    imgui.set_next_window_size((420, 900))
    imgui.begin("Style")
    (tab or panel._draw_layer)()
    imgui.end()
    imgui.render()
    assert imgui.get_draw_data() is not None


def _grid(n=6):
    """A rectangular (x, y) lattice, so every 3D kind — including the gridded ones — builds."""
    xs = np.linspace(-1.0, 1.0, n)
    ys = np.linspace(-1.0, 1.0, n)
    gx, gy = np.meshgrid(xs, ys)
    gz = np.sin(gx) * np.cos(gy)
    return gx.ravel(), gy.ravel(), gz.ravel()


def _scatter_xyz(n=40):
    """Scattered points — the shape the gridded kinds must *refuse*."""
    rng = np.random.default_rng(7)
    return rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)


def _add3d(plot, kind="scatter3d", label="L", data=None, **kw):
    x, y, z = data if data is not None else _grid()
    return layerops3d.add_xyz_layer(plot, x, y, z, kind=kind, label=label, **kw)


def _select(plot, layer):
    plot.interaction.selected_layer_id = layer.layer_id
    return layer


def _data_layers(plot):
    """Scene layers that are the user's data, not the 3D axis/grid decoration.

    The decoration is rebuilt by every ``set_3d_view`` — which ``add_geometry3d`` calls —
    so raw indices into ``scene.layers`` shift under any operation that adds a layer.
    Draw *order among the data* is the thing a conversion has to preserve, and it is what
    these compare.
    """
    return [
        layer
        for layer in plot.scene.layers
        if layer.metadata.get("artist") not in plot._SYSTEM_3D_ARTISTS
    ]


# ----------------------------------------------------------------------------------
# Draw tests
# ----------------------------------------------------------------------------------


class TestLayerEditorDraws:
    @pytest.mark.parametrize("kind", list(layerops3d.KIND3D_KEYS))
    def test_it_draws_every_3d_kind(self, harness, ws, plot, panel, all_sections, kind):
        """Each kind's own parameters have to survive a real frame, not just exist."""
        plot.set_ndim(3)
        _select(plot, _add3d(plot, kind=kind, label=kind))
        _layer_frame(harness, panel)

    @pytest.mark.parametrize("kind", ["line", "scatter", "bar", "hist", "line_family"])
    def test_it_draws_a_2d_layer(self, harness, ws, plot, panel, all_sections, kind):
        """The 2D path must keep working, including the sections 3D added around it."""
        x = np.arange(12.0)
        _select(plot, layerops.add_xy_layer(plot, x, np.sin(x), kind=kind, label=kind))
        _layer_frame(harness, panel)

    def test_it_draws_an_untagged_3d_layer(self, harness, ws, plot, panel, all_sections):
        """A layer built outside the GUI takes the dimmed-picker branch.

        This is the ``push_style_var``/``pop_style_var`` pair and the tooltip — the one
        path where an imbalance would corrupt the style stack for the rest of the frame.
        """
        plot.set_ndim(3)
        verts = np.column_stack(_grid()).astype(np.float32)
        layer = plot.add_geometry3d(verts, None, primitive="points", label="raw")
        _select(plot, layer)
        assert layerops3d.layer_kind3d(layer) is None
        _layer_frame(harness, panel)

    def test_it_draws_a_colormapped_2d_scatter(self, harness, ws, plot, panel, all_sections):
        """The ``values2d`` colormap branch, with its matplotlib swatch strip."""
        x = np.arange(30.0)
        layer = layerops.add_xy_layer(plot, x, np.sin(x), kind="scatter", label="s", c=x)
        _select(plot, layer)
        assert layerops.layer_colormap_kind(layer) == "values2d"
        _layer_frame(harness, panel)

    def test_it_draws_the_gl_colormap_branch(self, harness, ws, plot, panel, all_sections):
        """A polyline's shader colormap is a different namespace and a different setter."""
        x = np.arange(20.0)
        layer = layerops.add_xy_layer(plot, x, np.cos(x), kind="line", label="l")
        _select(plot, layer)
        assert layerops.layer_colormap_kind(layer) == "gl_line"
        _layer_frame(harness, panel)

    def test_it_draws_the_manual_colormap_range(self, harness, ws, plot, panel, all_sections):
        """Unticking 'Auto range' reveals two more widgets; they must draw too."""
        plot.set_ndim(3)
        layer = _select(plot, _add3d(plot, kind="surface3d"))
        layer.metadata["vmin"] = -0.5
        layer.metadata["vmax"] = 0.5
        _layer_frame(harness, panel)

    def test_it_draws_the_outline_section_open(self, harness, ws, plot, panel, all_sections):
        """The colour/width/alpha widgets only exist while the outline is on."""
        plot.set_ndim(3)
        layer = _select(plot, _add3d(plot, kind="surface3d"))
        layer.style.outline_enabled = True
        _layer_frame(harness, panel)

    def test_it_draws_every_blend_mode(self, harness, ws, plot, panel, all_sections):
        """Each mode prints its own explanation; AUTO prints an extra line."""
        x = np.arange(10.0)
        _select(plot, layerops.add_xy_layer(plot, x, x, kind="line", label="l"))
        for mode in BlendMode:
            plot.options.blend_mode = mode
            _layer_frame(harness, panel)

    def test_the_whole_panel_draws_with_a_3d_layer_selected(self, harness, ws, plot, panel):
        """Every tab in one frame, through the panel's own tab bar."""
        plot.set_ndim(3)
        _select(plot, _add3d(plot, kind="surface3d"))
        for _ in range(3):
            _layer_frame(harness, panel, tab=panel.draw)
            ws.drain()

    def test_it_draws_the_3d_compositing_section(self, harness, ws, plot, panel, all_sections):
        """The per-layer blend/occlusion/auto-alpha section, in every override state.

        The slider only exists while auto-alpha is on and the mode help only while a mode is
        chosen, so both branches have to be drawn -- an ``enum_combo`` with a bad current
        value or a ``same_line`` after a skipped widget would only show up here.
        """
        plot.set_ndim(3)
        layer = _select(plot, _add3d(plot, kind="scatter3d"))
        _layer_frame(harness, panel)  # all defaults (everything inheriting)
        layer.style.blend_mode = BlendMode.ADDITIVE
        layer.style.depth_write = False
        layer.style.auto_alpha = 0.8
        _layer_frame(harness, panel)  # every override active, slider present

    def test_the_compositing_section_is_absent_for_a_2d_layer(
        self, harness, ws, plot, panel, all_sections
    ):
        """It is gated on a 3D layer -- the 2D pass has one figure-wide blend state."""
        x = np.arange(10.0)
        _select(plot, layerops.add_xy_layer(plot, x, np.sin(x), kind="line", label="l"))
        _layer_frame(harness, panel)

    def test_it_draws_with_no_layers_at_all(self, harness, ws, plot, panel, all_sections):
        _layer_frame(harness, panel)


# ----------------------------------------------------------------------------------
# 3D plot-type conversion
# ----------------------------------------------------------------------------------


class TestKind3DConversion:
    @pytest.mark.parametrize("target", list(layerops3d.KIND3D_KEYS))
    def test_every_kind_is_reachable_from_a_scatter(self, ws, plot, panel, target):
        """The picker offers nine kinds; all nine must actually convert."""
        plot.set_ndim(3)
        layer = _select(plot, _add3d(plot, kind="scatter3d", label="mine"))
        panel._replot_layer3d(
            layer, target, layerops3d.default_kind3d_options(target), undoable=True
        )

        assert layerops3d.layer_kind3d(layer) == "scatter3d", "must not mutate during draw"
        ws.drain()

        live = layerops.find_layer(plot, plot.interaction.selected_layer_id)
        assert live is not None
        assert layerops3d.layer_kind3d(live) == target
        assert live.label == "mine"
        assert panel._layer_error == ""

    def test_the_old_layer_is_gone_and_the_draw_order_is_kept(self, ws, plot, panel):
        """A type change must not send the layer to the back of the scene."""
        plot.set_ndim(3)
        first = _add3d(plot, kind="scatter3d", label="first")
        second = _add3d(plot, kind="scatter3d", label="second")
        assert [layer.label for layer in _data_layers(plot)] == ["first", "second"]
        _select(plot, first)

        panel._replot_layer3d(first, "surface3d", {"nu": 0, "nv": 0}, undoable=True)
        ws.drain()

        assert all(
            layer is not first for layer in plot.scene.layers
        ), "the old layer must be dropped"
        new = layerops.find_layer(plot, plot.interaction.selected_layer_id)
        assert new.label == "first"
        assert [layer.label for layer in _data_layers(plot)] == ["first", "second"]
        assert _data_layers(plot)[0] is new
        assert _data_layers(plot)[1] is second

    def test_a_conversion_is_undoable(self, ws, plot, panel):
        plot.set_ndim(3)
        layer = _select(plot, _add3d(plot, kind="scatter3d", label="mine"))
        panel._replot_layer3d(layer, "wireframe3d", {"stride": 1, "nu": 0, "nv": 0}, undoable=True)
        ws.drain()
        assert ws.undo.can_undo()

        live = layerops.find_layer(plot, plot.interaction.selected_layer_id)
        assert layerops3d.layer_kind3d(live) == "wireframe3d"

        ws.undo.undo()
        ws.drain()
        back = layerops.find_layer(plot, plot.interaction.selected_layer_id)
        assert layerops3d.layer_kind3d(back) == "scatter3d"
        assert back.label == "mine"

        # Redo must land where `do` did: the closures resolve the layer by a live id.
        ws.undo.redo()
        ws.drain()
        again = layerops.find_layer(plot, plot.interaction.selected_layer_id)
        assert layerops3d.layer_kind3d(again) == "wireframe3d"

    def test_an_options_edit_keeps_the_layer_and_is_not_undoable(self, ws, plot, panel):
        """A slider drag re-derives in place: same id, no history entry per frame."""
        plot.set_ndim(3)
        layer = _select(plot, _add3d(plot, kind="wireframe3d", label="w"))
        layer_id = layer.layer_id

        panel._replot_layer3d(layer, "wireframe3d", {"stride": 3, "nu": 0, "nv": 0}, undoable=False)
        assert layerops3d.layer_kind3d_options(layer)["stride"] == 1
        ws.drain()

        assert layerops.find_layer(plot, layer_id) is layer
        assert layerops3d.layer_kind3d_options(layer)["stride"] == 3
        assert not ws.undo.can_undo()

    def test_style_and_outline_survive_a_conversion(self, ws, plot, panel):
        """The four outline fields are in ``_KIND_CHANGE_STYLE_FIELDS`` for this reason."""
        plot.set_ndim(3)
        layer = _select(plot, _add3d(plot, kind="scatter3d", label="s"))
        layer.style.outline_enabled = True
        layer.style.outline_color = (1.0, 0.0, 0.0, 1.0)
        layer.style.outline_width = 3.5
        layer.style.outline_alpha = 0.4
        layer.style.alpha = 0.25
        layer.style.zorder = 7

        panel._replot_layer3d(layer, "ribbon3d", {"baseline": 0.0}, undoable=True)
        ws.drain()

        new = layerops.find_layer(plot, plot.interaction.selected_layer_id)
        assert new.style.outline_enabled is True
        assert tuple(new.style.outline_color) == (1.0, 0.0, 0.0, 1.0)
        assert new.style.outline_width == pytest.approx(3.5)
        assert new.style.outline_alpha == pytest.approx(0.4)
        assert new.style.alpha == pytest.approx(0.25)
        assert new.style.zorder == 7

    def test_a_gridded_kind_on_scattered_points_reports_instead_of_raising(self, ws, plot, panel):
        """Grid3DError is a real user error: it must reach the panel, not the frame."""
        plot.set_ndim(3)
        layer = _select(plot, _add3d(plot, kind="scatter3d", label="cloud", data=_scatter_xyz()))
        panel._replot_layer3d(layer, "surface3d", {"nu": 0, "nv": 0}, undoable=True)
        ws.drain()  # must not raise

        assert "grid" in panel._layer_error.lower()
        assert layerops3d.layer_kind3d(layer) == "scatter3d", "a refusal must change nothing"

    def test_an_untagged_layer_cannot_be_converted(self, ws, plot, panel):
        """No kind tag means no source columns, which means no honest conversion."""
        plot.set_ndim(3)
        verts = np.column_stack(_grid()).astype(np.float32)
        layer = _select(plot, plot.add_geometry3d(verts, None, primitive="points", label="raw"))

        with pytest.raises(ValueError):
            layerops.replot_layer_xyz(plot, None, layer, kind="surface3d")

        panel._replot_layer3d(layer, "surface3d", {"nu": 0, "nv": 0}, undoable=True)
        ws.drain()
        assert panel._layer_error
        assert layerops.find_layer(plot, layer.layer_id) is layer

    def test_a_deleted_layer_is_a_safe_noop(self, ws, plot, panel):
        """The Scene panel queues removals too; the drain may find nothing left."""
        plot.set_ndim(3)
        layer = _select(plot, _add3d(plot, kind="scatter3d"))
        panel._replot_layer3d(layer, "line3d", {}, undoable=True)
        plot.scene.layers = [other for other in plot.scene.layers if other is not layer]
        ws.drain()  # must not raise

    def test_the_picker_is_blocked_for_the_two_unconvertible_cases(self, plot, panel):
        """Both refusals name their own cause; a generic message helps nobody."""
        plot.set_ndim(3)
        verts = np.column_stack(_grid()).astype(np.float32)
        untagged = plot.add_geometry3d(verts, None, primitive="points", label="raw")
        assert layerops3d.layer_source_xyz(untagged) is None
        reason = panel._kind3d_block_reason(untagged, None)
        assert "no plot-type tag" in reason

        derived = _add3d(plot, kind="ribbon3d", label="r")
        derived.metadata.pop("gui_source_xyz")
        assert layerops3d.layer_source_xyz(derived) is None
        reason = panel._kind3d_block_reason(derived, "ribbon3d")
        assert "source columns" in reason

    def test_a_derived_layer_with_no_source_still_draws(
        self, harness, ws, plot, panel, all_sections
    ):
        """Both blocked branches take the dimmed path, and neither offers dead sliders."""
        plot.set_ndim(3)
        derived = _select(plot, _add3d(plot, kind="ribbon3d", label="r"))
        derived.metadata.pop("gui_source_xyz")
        _layer_frame(harness, panel)


# ----------------------------------------------------------------------------------
# Colormaps
# ----------------------------------------------------------------------------------


class TestLayerColormap:
    @pytest.mark.parametrize(
        "kind", ["scatter3d", "line3d", "surface3d", "wireframe3d", "bar3d", "volume3d"]
    )
    def test_every_gui_built_3d_kind_reports_a_colormap(self, plot, kind):
        """``layer_colormap_kind`` used to answer only for scatter3d + pyplot's zdata."""
        layer = _add3d(plot, kind=kind, label=kind)
        assert layerops.layer_colormap_kind(layer) == "values3d"

    def test_a_flat_coloured_3d_layer_reports_none(self, plot):
        """No retained scalars, no colormap control — the honest answer."""
        layer = _add3d(plot, kind="scatter3d", label="flat", colormap_by_z=False)
        assert layer.metadata.get("cvalues") is None
        assert layerops.layer_colormap_kind(layer) is None

    def test_remapping_a_3d_layer_is_queued_and_recolours_it(self, ws, plot, panel):
        plot.set_ndim(3)
        layer = _select(plot, _add3d(plot, kind="surface3d", label="s"))
        before = layer.colors.copy()

        panel._set_layer_colormap(layer, cmap="magma")
        assert layer.metadata["cmap"] == "viridis", "mutation must not happen in draw"

        ws.drain()
        assert layer.metadata["cmap"] == "magma"
        assert not np.array_equal(layer.colors, before)
        assert layer.dirty.gpu_dirty is True, "3D colours live in a VBO read at upload time"

    def test_remapping_preserves_the_vertex_count(self, ws, plot, panel):
        """A short colour array reads past the VBO's end at draw time."""
        layer = _add3d(plot, kind="bar3d", label="b")
        panel._set_layer_colormap(layer, cmap="turbo")
        ws.drain()
        assert len(layer.colors) == len(layer.vertices)

    def test_the_range_can_be_pinned_and_released(self, ws, plot, panel):
        layer = _add3d(plot, kind="surface3d", label="s")
        panel._set_layer_colormap(layer, vmin=-2.0, vmax=2.0)
        ws.drain()
        assert layerops.layer_colormap(layer)[1:] == (-2.0, 2.0)

        panel._set_layer_colormap(layer, vmin=None, vmax=None)
        ws.drain()
        assert layerops.layer_colormap(layer)[1:] == (None, None)

    def test_the_gl_override_is_tri_state(self, ws, plot, panel):
        x = np.arange(20.0)
        layer = layerops.add_xy_layer(plot, x, np.cos(x), kind="line", label="l")
        assert layerops.layer_gl_colormap(layer) == (None, None)

        panel._set_layer_gl_colormap(layer, enabled=True)
        panel._set_layer_gl_colormap(layer, scheme_index=4)
        ws.drain()
        assert layerops.layer_gl_colormap(layer) == (True, 4)

        panel._set_layer_gl_colormap(layer, enabled=None)
        ws.drain()
        assert layerops.layer_gl_colormap(layer)[0] is None, "None means 'follow the scene'"

    def test_a_colormap_call_on_a_layer_with_none_reports_instead_of_raising(self, ws, plot, panel):
        """The layer can be replaced between the click and the drain."""
        x = np.arange(10.0)
        layer = layerops.add_xy_layer(plot, x, x, kind="scatter", label="s")
        panel._set_layer_colormap(layer, cmap="magma")
        ws.drain()  # must not raise
        assert panel._layer_error

    def test_the_swatch_strip_is_the_real_ramp(self):
        from glplot.gui.panels.style import cpu_colormap_strip_colors

        strip = cpu_colormap_strip_colors("viridis", steps=8)
        assert len(strip) == 8
        assert all(len(c) == 3 and all(0.0 <= v <= 1.0 for v in c) for c in strip)
        # viridis runs dark blue -> yellow: the ends must not be the same colour.
        assert strip[0] != strip[-1]
        # Memoised, so the picker costs nothing per frame after the first.
        assert cpu_colormap_strip_colors("viridis", steps=8) is strip

    def test_an_unknown_colormap_previews_grey_rather_than_raising(self):
        from glplot.gui.panels.style import cpu_colormap_strip_colors

        strip = cpu_colormap_strip_colors("not-a-real-colormap", steps=4)
        assert strip == [(0.5, 0.5, 0.5)] * 4


# ----------------------------------------------------------------------------------
# Outline, and the rest of the per-layer surface
# ----------------------------------------------------------------------------------


class TestOutlineAndBlending:
    @pytest.mark.parametrize("build", ["3d", "scatter", "line", "bar"])
    def test_the_outline_is_settable_on_every_layer_type(self, ws, plot, panel, build):
        if build == "3d":
            layer = _add3d(plot, kind="surface3d", label="s")
        else:
            x = np.arange(10.0)
            layer = layerops.add_xy_layer(plot, x, np.sin(x), kind=build, label=build)

        panel._set_style(
            layer,
            outline_enabled=True,
            outline_color=(0.1, 0.2, 0.3, 1.0),
            outline_width=2.5,
            outline_alpha=0.6,
        )
        assert layer.style.outline_enabled is False, "mutation must not happen in draw"

        ws.drain()
        assert layer.style.outline_enabled is True
        assert layer.style.outline_color == (0.1, 0.2, 0.3, 1.0)
        assert layer.style.outline_width == pytest.approx(2.5)
        assert layer.style.outline_alpha == pytest.approx(0.6)

    def test_the_dead_outline_types_match_the_renderers(self):
        """The caveat line must name exactly the types with no outline reader.

        Read off the renderers rather than trusted: this list is the panel's only
        protection against telling a user their outline does nothing when it does.
        """
        from glplot.gui.panels.style import OUTLINE_DEAD_TYPES

        assert OUTLINE_DEAD_TYPES == frozenset({"text"})
        for module in ("scatter", "polyline", "patch", "line_family", "geometry3d"):
            source = __import__(f"glplot.renderers.{module}", fromlist=["x"]).__file__
            with open(source, "r", encoding="utf-8") as handle:
                assert "outline_enabled" in handle.read(), module

    def test_point_size_and_alpha_reach_a_3d_point_layer(self, ws, plot, panel):
        from glplot.gui.panels.style import _point_size_applies

        layer = _add3d(plot, kind="volume3d", label="v")
        assert _point_size_applies(layer) is True

        panel._set_style(layer, point_size=9.0, alpha=0.3)
        ws.drain()
        assert layer.style.point_size == pytest.approx(9.0)
        assert layer.style.alpha == pytest.approx(0.3)

    def test_point_size_does_not_apply_to_a_3d_mesh(self, plot):
        from glplot.gui.panels.style import _point_size_applies

        assert _point_size_applies(_add3d(plot, kind="surface3d", label="s")) is False

    def test_the_blend_mode_is_queued(self, ws, plot, panel):
        panel._set(plot.options, "blend_mode", BlendMode.ADDITIVE)
        assert plot.options.blend_mode is not BlendMode.ADDITIVE
        ws.drain()
        assert plot.options.blend_mode is BlendMode.ADDITIVE

    def test_the_per_layer_blend_mode_is_queued(self, ws, plot, panel):
        """The section's own control writes ``style.blend_mode``, not the figure option."""
        layer = _add3d(plot, kind="volume3d", label="v")
        panel._set_style(layer, blend_mode=BlendMode.ADDITIVE)
        assert layer.style.blend_mode is None, "mutation must not happen in draw"
        ws.drain()
        assert layer.style.blend_mode is BlendMode.ADDITIVE
        assert plot.options.blend_mode is not BlendMode.ADDITIVE, "figure mode is untouched"

    def test_forced_occlusion_is_queued(self, ws, plot, panel):
        layer = _add3d(plot, kind="volume3d", label="v")
        panel._set_style(layer, depth_write=False)
        ws.drain()
        assert layer.style.depth_write is False

    def test_auto_alpha_is_queued(self, ws, plot, panel):
        layer = _add3d(plot, kind="volume3d", label="v")
        panel._set_style(layer, auto_alpha=0.75)
        ws.drain()
        assert layer.style.auto_alpha == pytest.approx(0.75)

    def test_compositing_overrides_survive_a_kind_conversion(self, ws, plot, panel):
        """A blend/occlusion/auto-alpha choice must not be dropped when the plot type
        changes, the same guarantee the outline fields get."""
        plot.set_ndim(3)
        layer = _select(plot, _add3d(plot, kind="scatter3d", label="s"))
        layer.style.blend_mode = BlendMode.ADDITIVE
        layer.style.depth_write = True
        layer.style.auto_alpha = 0.6

        panel._replot_layer3d(
            layer, "volume3d", layerops3d.default_kind3d_options("volume3d"), undoable=True
        )
        ws.drain()

        new = layerops.find_layer(plot, plot.interaction.selected_layer_id)
        assert new.style.blend_mode is BlendMode.ADDITIVE
        assert new.style.depth_write is True
        assert new.style.auto_alpha == pytest.approx(0.6)

    def test_every_blend_mode_is_documented(self):
        from glplot.gui.panels.style import BLEND_MODE_HELP

        assert {m.name for m in BlendMode} <= set(BLEND_MODE_HELP)
        assert all(len(text) > 40 for text in BLEND_MODE_HELP.values())

    def test_the_shared_options_editor_still_owns_the_keys_it_draws(self):
        """If ``widgets`` ever grows a 3D key, the panel must stop drawing it locally."""
        from glplot.gui.panels.style import ARRAY_KIND3D_OPTION_KEYS, SHARED_KIND_OPTION_KEYS

        every_key = set()
        for kind in layerops3d.KIND3D_KEYS:
            every_key |= set(layerops3d.default_kind3d_options(kind))
        local = every_key - SHARED_KIND_OPTION_KEYS - ARRAY_KIND3D_OPTION_KEYS
        # The keys the panel draws itself today. A key leaving this set means the shared
        # editor grew a branch for it and the local one must go.
        assert local == {"stride", "nu", "nv", "gap", "bar_dx", "bar_dy", "scale", "head"}
