"""Test the 3D GUI: the View 3D and Objects 3D panels, and the other panels' 3D paths.

Panel *bodies* are exercised through the sanctioned headless imgui harness (CONTRACT
§2.10): create a context, open a frame, call ``draw``, render, and assert the frame closed.
That catches the whole class of mistakes a pure-logic test cannot — a wrong imgui call
signature, an unbalanced begin/end, a missing pop — which is most of what can go wrong in
a panel.

The action half is asserted on state: every mutation is queued (CONTRACT §1.1), so the
tests drain the queue and then look at the scene.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.engine import GPULinePlot
from glplot.gui import generators3d, layerops3d
from glplot.gui.commands import CommandQueue
from glplot.gui.datasets import Column, DataSet, DataStore
from glplot.gui.history import UndoStack


class FakeWorkspace:
    """The four services a panel reads off a workspace.

    A real ``Workspace`` needs imgui to construct and builds every panel; a stand-in keeps
    these tests to the panel under test.
    """

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
def harness():
    """A headless imgui context sized like a real window, destroyed afterwards.

    The teardown is not optional here. imgui's context is *global state*, and this file
    sorts before ``test_gui_actions.py``; a leaked context left current made that file's
    own ``create_context``/``destroy_context`` fixture see a foreign context and five of
    its dispatch tests fail. Cleaning up is what keeps the suite order-independent.
    """
    imgui = pytest.importorskip("imgui")
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = 1200, 900
    io.fonts.get_tex_data_as_rgba32()
    io.fonts.texture_id = 1
    io.delta_time = 1 / 60.0
    yield imgui
    imgui.destroy_context(ctx)


@pytest.fixture
def all_sections(monkeypatch):
    """Force every collapsing header open, so one frame covers every widget branch."""
    from glplot.gui import widgets

    monkeypatch.setattr(widgets, "section", lambda label, **kw: True)


def _frame(imgui, panel, title="Panel"):
    imgui.new_frame()
    imgui.begin(title)
    panel.draw()
    imgui.end()
    imgui.render()
    assert imgui.get_draw_data() is not None


def _curve(n=120):
    t = np.linspace(0.0, 8.0, n)
    return np.cos(t), np.sin(t), t / 4.0


def _data_layers(plot):
    """Scene layers that are the user's data, not the 3D decoration."""
    return [
        layer
        for layer in plot.scene.layers
        if layer.metadata.get("artist") not in plot._SYSTEM_3D_ARTISTS
    ]


class TestView3DPanel:
    def _panel(self, ws):
        from glplot.gui.panels.view3d import View3DPanel

        return View3DPanel(ws)

    def test_it_draws_the_empty_2d_state(self, harness, ws):
        """A 2D figure gets an explanation, not a wall of dead 3D controls."""
        _frame(harness, self._panel(ws), "View 3D")

    def test_it_draws_every_section_in_3d(self, harness, ws, plot, all_sections):
        plot.set_ndim(3)
        layerops3d.add_xyz_layer(plot, *_curve(), kind="scatter3d", label="c")
        panel = self._panel(ws)
        for _ in range(3):
            _frame(harness, panel, "View 3D")
            ws.drain()

    def test_mode_switch_is_queued_and_applied(self, ws, plot):
        panel = self._panel(ws)
        panel._set_ndim(3)
        assert plot.is_3d_scene() is False, "mutation must not happen in draw"
        ws.drain()
        assert plot.is_3d_scene() is True

    def test_preset_snaps_the_camera(self, ws, plot):
        plot.set_ndim(3)
        panel = self._panel(ws)
        panel._set_preset("top")
        ws.drain()
        assert plot.camera3d.elev == pytest.approx(90.0)

    def test_camera_setter_goes_through_set_3d_view(self, ws, plot):
        plot.set_ndim(3)
        layerops3d.add_xyz_layer(plot, *_curve(), kind="scatter3d", label="c")
        panel = self._panel(ws)
        panel._set_camera(elev=42.0)
        ws.drain()
        assert plot.camera3d.elev == pytest.approx(42.0)
        # The layer's copy of the camera must follow, or it draws under the old view.
        layer = _data_layers(plot)[0]
        assert layer.metadata["camera"]["elev"] == pytest.approx(42.0)

    def test_axes_toggle_removes_the_artist(self, ws, plot):
        plot.set_ndim(3)
        layerops3d.add_xyz_layer(plot, *_curve(), kind="scatter3d", label="c")
        assert any(layer.metadata.get("artist") == "grid3d" for layer in plot.scene.layers)
        panel = self._panel(ws)
        panel._set_axes("show_grid", False)
        ws.drain()
        assert not any(layer.metadata.get("artist") == "grid3d" for layer in plot.scene.layers)

    def test_box_aspect_setter(self, ws, plot):
        plot.set_ndim(3)
        panel = self._panel(ws)
        panel._set_box_aspect((1.0, 1.0, 1.0))
        ws.drain()
        assert plot.camera3d.box_aspect == (1.0, 1.0, 1.0)
        panel._set_box_aspect(None)
        ws.drain()
        assert plot.camera3d.box_aspect is None

    def test_spin_toggle_round_trips(self, ws, plot):
        plot.set_ndim(3)
        panel = self._panel(ws)
        panel._toggle_spin()
        ws.drain()
        assert plot.camera3d.auto_spin != 0.0
        panel._toggle_spin()
        ws.drain()
        assert plot.camera3d.auto_spin == 0.0

    def test_current_view_label_recognises_a_preset_and_a_custom_angle(self, ws, plot):
        plot.set_ndim(3)
        panel = self._panel(ws)
        plot.set_3d_preset("front")
        assert panel._current_view_label() == "Front (−Y)"
        plot.set_3d_view(elev=17.0, azim=3.0)
        assert panel._current_view_label() == "Custom"

    def test_aspect_label_matches_the_presets(self, ws, plot):
        panel = self._panel(ws)
        assert panel._aspect_label(None) == "Data units"
        assert panel._aspect_label((1.0, 1.0, 1.0)) == "Cube (1:1:1)"
        assert panel._aspect_label((3.0, 1.0, 7.0)) == "Custom"

    def test_ssao_setter_reaches_the_options(self, ws, plot):
        plot.set_ndim(3)
        panel = self._panel(ws)
        panel._set_ssao("enabled", True)
        ws.drain()
        assert plot.options.visual.ssao.enabled is True


class TestObjects3DPanel:
    def _panel(self, ws):
        from glplot.gui.panels.objects3d import Objects3DPanel

        return Objects3DPanel(ws)

    def test_it_draws_for_every_generator(self, harness, ws, all_sections):
        panel = self._panel(ws)
        for category, _label in generators3d.CATEGORIES:
            panel.category = category
            for spec in generators3d.by_category(category):
                panel._select(spec.key)
                _frame(harness, panel, "Objects 3D")
                ws.drain()

    @pytest.mark.parametrize(
        "key", [generators3d.by_category(c)[0].key for c, _ in generators3d.CATEGORIES]
    )
    def test_plotting_one_of_each_category(self, ws, plot, key):
        panel = self._panel(ws)
        panel._select(key)
        panel.samples = 600
        panel._action_plot()
        ws.drain()
        assert not panel._error, panel._error
        assert plot.is_3d_scene() is True
        assert len(_data_layers(plot)) == 1
        # The table is registered too, so a generated object is an ordinary dataset.
        assert len(ws.store) == 1

    def test_plot_is_undoable(self, ws, plot):
        panel = self._panel(ws)
        panel._select("sphere")
        panel.samples = 400
        panel._action_plot()
        ws.drain()
        assert len(_data_layers(plot)) == 1
        ws.undo.undo()
        ws.drain()
        assert _data_layers(plot) == []
        assert len(ws.store) == 0

    def test_create_dataset_makes_no_layer(self, ws, plot):
        panel = self._panel(ws)
        panel._select("helix")
        panel.samples = 300
        panel._action_create_dataset()
        ws.drain()
        assert len(ws.store) == 1
        assert _data_layers(plot) == []

    def test_switching_object_drops_a_stale_kind_override(self, ws):
        """ "Draw it as stems" makes sense for a curve and not for a vector field."""
        panel = self._panel(ws)
        panel._select("helix")
        panel.kind = "stem3d"
        panel._select("abc_flow")
        assert panel.kind is None
        assert panel.current_kind() == "quiver3d"

    def test_parameters_persist_per_object(self, ws):
        panel = self._panel(ws)
        panel._select("helix")
        panel.current_params()["turns"] = 17.0
        panel._select("sphere")
        panel._select("helix")
        assert panel.current_params()["turns"] == pytest.approx(17.0)

    def test_surface_options_carry_the_sampling_shape(self, ws):
        panel = self._panel(ws)
        panel._select("torus")
        panel.samples = 900
        spec, table = panel._build_table()
        options = panel._plot_options(spec, table)
        assert options["nu"] * options["nv"] == len(table["x"])

    def test_quiver_options_carry_the_vector_columns(self, ws):
        panel = self._panel(ws)
        panel._select("vortex")
        spec, table = panel._build_table()
        options = panel._plot_options(spec, table)
        assert set(("u", "v", "w")) <= set(options)

    def test_labels_are_unique_across_repeats(self, ws, plot):
        panel = self._panel(ws)
        panel._select("sphere")
        panel.samples = 400
        panel._action_plot()
        ws.drain()
        panel._action_plot()
        ws.drain()
        labels = [layer.label for layer in _data_layers(plot)]
        assert len(labels) == len(set(labels)) == 2


class TestDataPanel3D:
    """The 2D/3D toggle, the Z column and the 3D kinds."""

    def _panel(self, ws):
        from glplot.gui.panels.data_editor import DataEditorPanel

        return DataEditorPanel(ws)

    def _dataset(self, ws, nx=7, ny=5):
        gx, gy = np.meshgrid(np.linspace(-2, 2, nx), np.linspace(-1, 1, ny))
        ds = DataSet(
            "grid",
            [
                Column("x", gx.ravel()),
                Column("y", gy.ravel()),
                Column("z", np.sin(gx * gy).ravel()),
                Column("u", np.cos(gx).ravel()),
                Column("v", np.sin(gy).ravel()),
                Column("w", np.ones(gx.size)),
            ],
        )
        ws.store.add(ds)
        return ds

    def test_the_picker_defaults_to_the_figure(self, harness, ws, plot, all_sections):
        panel = self._panel(ws)
        self._dataset(ws)
        panel._current = ws.store.datasets[0]
        _frame(harness, panel, "Data")
        assert panel._plot_ndim == 2
        plot.set_ndim(3)
        _frame(harness, panel, "Data")
        assert panel._plot_ndim == 3

    def test_an_explicit_toggle_wins_over_the_figure(self, harness, ws, plot):
        panel = self._panel(ws)
        self._dataset(ws)
        panel._current = ws.store.datasets[0]
        plot.set_ndim(3)
        panel._plot_ndim = 2
        panel._ndim_touched = True
        _frame(harness, panel, "Data")
        assert panel._plot_ndim == 2

    @pytest.mark.parametrize("kind", list(layerops3d.KIND3D_KEYS))
    def test_every_3d_kind_plots_from_the_table(self, harness, ws, plot, kind):
        panel = self._panel(ws)
        ds = self._dataset(ws)
        panel._current = ds
        panel._plot_ndim = 3
        panel._ndim_touched = True
        panel._plot_x, panel._plot_y, panel._plot_z = "x", "y", "z"
        panel._set_plot_kind3d(kind)
        if kind == "quiver3d":
            panel._plot_u, panel._plot_v, panel._plot_w = "u", "v", "w"
        _frame(harness, panel, "Data")
        ws.drain()
        panel._plot_dataset_3d(ds)
        ws.drain()
        assert not panel._error, panel._error
        assert plot.is_3d_scene() is True
        layer = _data_layers(plot)[0]
        assert layerops3d.layer_kind3d(layer) == kind

    def test_a_quiver_with_no_vector_columns_is_refused(self, ws, plot):
        panel = self._panel(ws)
        ds = self._dataset(ws)
        panel._current = ds
        panel._plot_ndim = 3
        panel._plot_x, panel._plot_y, panel._plot_z = "x", "y", "z"
        panel._set_plot_kind3d("quiver3d")
        panel._plot_u = panel._plot_v = panel._plot_w = ""
        panel._plot_dataset_3d(ds)
        assert "U/V/W" in panel._error

    def test_the_3d_plot_is_undoable(self, ws, plot):
        panel = self._panel(ws)
        ds = self._dataset(ws)
        panel._current = ds
        panel._plot_ndim = 3
        panel._plot_x, panel._plot_y, panel._plot_z = "x", "y", "z"
        panel._plot_dataset_3d(ds)
        ws.drain()
        assert len(_data_layers(plot)) == 1
        ws.undo.undo()
        ws.drain()
        assert _data_layers(plot) == []

    def test_a_size_column_is_rescaled_into_pixels(self, ws):
        panel = self._panel(ws)
        pixels = panel._size_pixels(np.array([0.0, 1.0, 2.0]))
        lo, hi = panel._SIZE_PX_RANGE
        assert float(pixels.min()) == pytest.approx(lo)
        assert float(pixels.max()) == pytest.approx(hi)

    def test_a_constant_size_column_lands_mid_range(self, ws):
        panel = self._panel(ws)
        pixels = panel._size_pixels(np.full(4, 3.0))
        lo, hi = panel._SIZE_PX_RANGE
        assert np.allclose(pixels, 0.5 * (lo + hi))


class TestFunctionsPanel3D:
    """``z = f(x, y)`` as a heat map, a surface or a wireframe."""

    def _panel(self, ws):
        from glplot.gui.panels.functions import FunctionsPanel

        panel = FunctionsPanel(ws)
        panel.expr = "sin(x) * cos(y)"
        panel.x_min, panel.x_max, panel.samples = -3.0, 3.0, 24
        panel.y_min, panel.y_max, panel.y_samples = -3.0, 3.0, 18
        return panel

    def test_the_expression_is_detected_as_a_field(self, harness, ws, all_sections):
        panel = self._panel(ws)
        _frame(harness, panel, "Functions")
        assert panel.is_field is True

    @pytest.mark.parametrize("output", ["image", "surface", "wireframe"])
    def test_each_output_draws_and_plots(self, harness, ws, plot, output, all_sections):
        panel = self._panel(ws)
        panel.field_output = output
        _frame(harness, panel, "Functions")
        ws.drain()
        panel._action_plot_field()
        ws.drain()
        assert panel._status_ok, panel._status
        assert len(_data_layers(plot)) == 1
        assert plot.is_3d_scene() is (output != "image")

    def test_the_surface_has_one_vertex_per_grid_sample(self, ws, plot):
        panel = self._panel(ws)
        panel.field_output = "surface"
        panel._action_plot_field()
        ws.drain()
        layer = _data_layers(plot)[0]
        assert len(layer.vertices) == panel.samples * panel.y_samples
        assert layer.indices is not None

    def test_the_wireframe_stride_thins_the_mesh(self, ws, plot):
        panel = self._panel(ws)
        panel.field_output = "wireframe"
        panel.wire_stride = 1
        panel._action_plot_field()
        ws.drain()
        dense = len(_data_layers(plot)[0].vertices)

        plot.scene.layers.clear()
        panel.wire_stride = 4
        panel._action_plot_field()
        ws.drain()
        assert len(_data_layers(plot)[0].vertices) < dense

    def test_the_z_axis_is_named_after_the_expression(self, ws, plot):
        panel = self._panel(ws)
        panel.field_output = "surface"
        panel._action_plot_field()
        ws.drain()
        assert plot.axes3d.zlabel == panel.expr


class TestDynamicsPanel3D:
    """A Z axis turns a 2D projection into a space curve."""

    def _panel(self, ws):
        from glplot.gui.panels.dynamics import DynamicsPanel

        return DynamicsPanel(ws)

    def test_the_default_is_still_a_2d_projection(self, harness, ws, plot, all_sections):
        panel = self._panel(ws)
        _frame(harness, panel, "Dynamics")
        ws.drain()
        assert panel.z_axis == ""
        panel._action_plot()
        ws.drain()
        assert plot.is_3d_scene() is False

    def test_choosing_a_z_axis_plots_a_space_curve(self, harness, ws, plot, all_sections):
        panel = self._panel(ws)
        panel.x_axis, panel.y_axis, panel.z_axis = "x", "y", "z"
        _frame(harness, panel, "Dynamics")
        ws.drain()
        panel._action_plot()
        ws.drain()
        assert plot.is_3d_scene() is True
        layer = _data_layers(plot)[0]
        assert layerops3d.layer_kind3d(layer) == "line3d"

    def test_the_box_is_labelled_with_the_phase_variables(self, harness, ws, plot, all_sections):
        panel = self._panel(ws)
        panel.x_axis, panel.y_axis, panel.z_axis = "x", "y", "z"
        # The panel integrates from `draw` (change-gated), so a frame has to come first.
        # That is also the only order the real GUI can produce: the button is drawn by it.
        _frame(harness, panel, "Dynamics")
        panel._action_plot()
        ws.drain()
        assert (plot.axes3d.xlabel, plot.axes3d.ylabel, plot.axes3d.zlabel) == ("x", "y", "z")

    def test_the_curve_is_coloured_along_time(self, harness, ws, plot, all_sections):
        """A single-coloured orbit hides the direction of travel."""
        panel = self._panel(ws)
        panel.x_axis, panel.y_axis, panel.z_axis = "x", "y", "z"
        _frame(harness, panel, "Dynamics")
        panel._action_plot()
        ws.drain()
        layer = _data_layers(plot)[0]
        assert len(np.unique(layer.colors, axis=0)) > 1

    def test_the_3d_plot_is_undoable(self, harness, ws, plot, all_sections):
        panel = self._panel(ws)
        panel.x_axis, panel.y_axis, panel.z_axis = "x", "y", "z"
        _frame(harness, panel, "Dynamics")
        panel._action_plot()
        ws.drain()
        assert len(_data_layers(plot)) == 1
        ws.undo.undo()
        ws.drain()
        assert _data_layers(plot) == []


class TestEngineOverlayPass:
    """The engine's own overlay pass, which is where the 3D labels are painted.

    The regression this exists for: ``_draw_overlays`` calls ``axes3d.draw_labels``, and
    that call is reachable *only* from a live imgui frame — test mode skips the overlay
    pass entirely, and the panel harness never goes through the engine. A wrong pyimgui
    signature there therefore passed every test and crashed the window on the first 3D
    plot with an axis title. These drive the engine's own method.
    """

    def _scene(self, plot, *, titles: bool):
        plot.set_ndim(3)
        layerops3d.add_xyz_layer(plot, *_curve(), kind="scatter3d", label="c")
        if titles:
            plot.axes3d.xlabel = "X axis"
            plot.axes3d.ylabel = "Y axis"
            plot.axes3d.zlabel = "Z axis"
            plot.set_3d_view()

    def _overlay_frame(self, imgui, plot):
        imgui.new_frame()
        plot._draw_overlays()
        imgui.render()
        assert imgui.get_draw_data() is not None

    def test_the_overlay_pass_runs_on_a_3d_scene(self, harness, plot):
        self._scene(plot, titles=False)
        self._overlay_frame(harness, plot)

    def test_the_overlay_pass_runs_with_axis_titles(self, harness, plot):
        """Titles take the scaled-text path; numbers do not. This is the crashing case."""
        self._scene(plot, titles=True)
        assert any(a.kind == "title" for a in plot._axes3d_labels)
        self._overlay_frame(harness, plot)

    def test_the_overlay_pass_survives_an_empty_3d_scene(self, harness, plot):
        """A figure pinned to 3D with no data yet must not fall back to 2D spines.

        ``figure(projection="3d")`` puts you in exactly this state before you plot
        anything, and it is the one case where "is this 3D" cannot be inferred from the
        layers — so the explicit mode has to answer it.
        """
        plot.set_ndim(3)
        assert plot._is_pure_3d_scene() is True
        self._overlay_frame(harness, plot)

    def test_a_2d_figure_still_takes_the_2d_branch(self, plot):
        """The inference path is untouched for anything that never asked for a mode.

        Only the predicate is asserted, not the draw: the 2D overlay needs the axis
        renderer, which only exists after GL initialisation.
        """
        plot.add_scatter(np.arange(5.0), np.arange(5.0), None)
        assert plot._is_pure_3d_scene() is False


class TestWorkspaceIntegration:
    """The rail, the action table and the keymap, on a real workspace."""

    @pytest.fixture
    def workspace(self, harness):
        plot = GPULinePlot()
        plot.options.enable_hud = True
        ws = plot.hud.workspace
        assert ws is not None
        return ws

    def test_both_3d_panels_are_registered(self, workspace):
        assert "view3d" in workspace.panels
        assert "objects3d" in workspace.panels

    def test_the_3d_actions_exist_and_are_documented(self, workspace):
        actions = [a for a in workspace.registry._actions if a.category == "3D"]
        assert len(actions) >= 15
        for action in actions:
            assert action.title and action.description
            assert action.icon

    def test_no_two_actions_share_a_chord_except_the_known_one(self, workspace):
        """A duplicate chord means only the earlier action ever runs."""
        from collections import Counter

        counts = Counter(str(a.chord) for a in workspace.registry._actions if a.chord)
        duplicates = {chord for chord, n in counts.items() if n > 1}
        offenders = {
            chord: [a.id for a in workspace.registry._actions if str(a.chord) == chord]
            for chord in duplicates
        }
        # `panel.dynamics` vs `view.autoscale` on Mod+0 predates the 3D work; the point of
        # this test is that nothing added since has made it worse.
        assert all(
            set(ids) == {"panel.dynamics", "view.autoscale"} for ids in offenders.values()
        ), offenders

    def test_every_3d_action_runs_without_raising(self, workspace):
        plot = workspace.plot
        plot.set_ndim(3)
        layerops3d.add_xyz_layer(plot, *_curve(), kind="scatter3d", label="c")
        for action in [a for a in workspace.registry._actions if a.category == "3D"]:
            action.run()
        workspace.queue.drain(plot)

    def test_the_3d_actions_are_disabled_in_a_2d_figure(self, workspace):
        """Greyed out rather than absent: an absent entry says 3D does not exist."""
        gated = [
            a
            for a in workspace.registry._actions
            if a.category == "3D" and a.id != "view3d.toggle_mode"
        ]
        assert gated
        assert all(a.is_enabled() is False for a in gated)
        workspace.plot.set_ndim(3)
        assert all(a.is_enabled() is True for a in gated)
