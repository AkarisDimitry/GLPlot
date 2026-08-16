"""The Objects 3D panel's live thumbnail: its projection, its geometry and its drawing.

Three layers, because they fail in three different ways.

**The projection** (:func:`glplot.gui.widgets.project_points_3d` and friends) is pure numpy
and is tested as pure numpy: no context, no frame, just "does a framed sphere land inside
the rect and does a vertex behind the eye come back as nan".

**The geometry** is the panel deciding what to hand the widget. Tested on state, because
the interesting questions -- is it capped, is it change-gated, does the colormap reach it,
does orbiting avoid rebuilding it -- are all answerable without drawing.

**The drawing** goes through the sanctioned headless imgui harness (CONTRACT §2.10) and
draws *every* generator in the catalogue with every section forced open. That is not
belt-and-braces: this widget calls ``draw_list.add_text``, ``add_triangle_filled`` and
``add_line`` directly, and pyimgui's signatures for those are not the C++ ones -- an
``add_text`` written with a font argument type-checks fine, passes any state-only test,
and crashes the window on the first frame. It shipped once. The only test that catches it
is one that actually submits the draw list, which is why every case here ends in
``imgui.render()`` and an assertion on the draw data.

The ``harness`` fixture destroys its context on teardown, for the reason
``test_gui_3d_panels.py`` documents: imgui's context is global state, and a leaked one
makes unrelated files fail depending on collection order.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.core.camera3d import Camera3D
from glplot.engine import GPULinePlot
from glplot.gui import generators3d, layerops3d, widgets
from glplot.gui.commands import CommandQueue
from glplot.gui.datasets import DataStore
from glplot.gui.history import UndoStack


class FakeWorkspace:
    """The four services a panel reads off a workspace (see ``test_gui_3d_panels.py``)."""

    def __init__(self, plot):
        self.plot = plot
        self.queue = CommandQueue()
        self.undo = UndoStack()
        self.store = DataStore()
        self.hud = None

    def drain(self):
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

    The teardown is not optional: imgui's context is global state, and a leaked one left
    current makes any later file's own create/destroy fixture see a foreign context.
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
    monkeypatch.setattr(widgets, "section", lambda label, **kw: True)


def _panel(ws):
    from glplot.gui.panels.objects3d import Objects3DPanel

    return Objects3DPanel(ws)


def _frame(imgui, panel, title="Objects 3D"):
    imgui.new_frame()
    imgui.begin(title)
    panel.draw()
    imgui.end()
    imgui.render()
    assert imgui.get_draw_data() is not None


def _every_spec():
    """Every generator in the catalogue, in category order."""
    return [spec for cat, _ in generators3d.CATEGORIES for spec in generators3d.by_category(cat)]


def _sphere_points(n=30):
    u = np.linspace(0.0, 2.0 * np.pi, n)
    v = np.linspace(0.0, np.pi, n)
    uu, vv = np.meshgrid(u, v)
    return np.column_stack(
        [
            (np.sin(vv) * np.cos(uu)).ravel(),
            (np.sin(vv) * np.sin(uu)).ravel(),
            np.cos(vv).ravel(),
        ]
    )


RECT = (0.0, 0.0, 300.0, 190.0)


# ----------------------------------------------------------------------------------
# The projection (pure numpy, no context)
# ----------------------------------------------------------------------------------


class TestProjection:
    def test_a_framed_object_lands_inside_the_rect(self):
        pts = _sphere_points()
        bounds = widgets.bounds3d(pts)
        mvp = Camera3D().mvp(300.0 / 190.0, bounds)
        sx, sy, depth = widgets.project_points_3d(pts, mvp, RECT)
        assert np.isfinite(sx).all() and np.isfinite(sy).all()
        assert RECT[0] <= sx.min() and sx.max() <= RECT[2]
        assert RECT[1] <= sy.min() and sy.max() <= RECT[3]
        # And it is not a dot in the middle: the default framing has to actually use the box.
        assert (sy.max() - sy.min()) > 0.4 * (RECT[3] - RECT[1])
        assert np.isfinite(depth).all()

    def test_the_fill_factor_magnifies_without_moving_the_centre(self):
        pts = _sphere_points()
        bounds = widgets.bounds3d(pts)
        camera = Camera3D()

        def span(box):
            sx, sy, _ = widgets.project_points_3d(pts, camera.mvp(300.0 / 190.0, box), RECT)
            return sy.max() - sy.min(), 0.5 * (sx.max() + sx.min())

        loose, centre_loose = span(bounds)
        tight, centre_tight = span(widgets._scaled_bounds(bounds, 1.0 / 1.35))
        assert tight > loose
        assert centre_tight == pytest.approx(centre_loose, abs=1.0)

    def test_depth_orders_near_before_far(self):
        """The painter's sort is only a depth cue if the sign is right."""
        camera = Camera3D()
        pts = _sphere_points()
        bounds = widgets.bounds3d(pts)
        _sx, _sy, depth = widgets.project_points_3d(pts, camera.mvp(1.5, bounds), RECT)
        nearest = pts[int(np.argmin(depth))]
        farthest = pts[int(np.argmax(depth))]
        # The nearest vertex is the one furthest along the direction of the eye.
        assert float(nearest @ camera.direction()) > float(farthest @ camera.direction())

    def test_a_vertex_behind_the_eye_is_nan_not_mirrored(self):
        """A naive divide folds it across the screen; the polyline splitter needs nan."""
        camera = Camera3D()
        pts = _sphere_points()
        mvp = camera.mvp(1.5, widgets.bounds3d(pts))
        behind = np.array([[0.0, 0.0, 0.0], [1e4, 1e4, 1e4]])
        sx, sy, depth = widgets.project_points_3d(behind, mvp, RECT)
        assert np.isfinite(sx[0])
        assert np.isnan(sx[1]) and np.isnan(sy[1]) and np.isnan(depth[1])

    def test_an_orthographic_camera_projects_too(self):
        pts = _sphere_points()
        camera = Camera3D(projection="orthographic")
        sx, sy, _ = widgets.project_points_3d(pts, camera.mvp(1.5, widgets.bounds3d(pts)), RECT)
        assert np.isfinite(sx).all() and np.isfinite(sy).all()

    def test_empty_and_broken_input_never_raise(self):
        mvp = Camera3D().mvp(1.5, None)
        assert widgets.project_points_3d(np.zeros((0, 3)), mvp, RECT)[0].shape == (0,)
        nan_mvp = np.full((4, 4), np.nan)
        sx, _sy, _d = widgets.project_points_3d(_sphere_points(4), nan_mvp, RECT)
        assert np.isnan(sx).all()
        assert widgets.project_points_3d(_sphere_points(4), np.eye(3), RECT)[0].shape == (16,)

    def test_as_points3d_refuses_anything_that_is_not_nx3(self):
        assert widgets.as_points3d(np.zeros((5, 3))).shape == (5, 3)
        assert widgets.as_points3d(np.zeros((5, 2))).shape == (0, 3)
        assert widgets.as_points3d(np.zeros(5)).shape == (0, 3)
        assert widgets.as_points3d([[1, 2, 3], [4, 5]]).shape == (0, 3)
        assert widgets.as_points3d(None).shape == (0, 3)

    def test_bounds_ignore_non_finite_rows(self):
        pts = np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 1.0], [2.0, 2.0, 2.0]])
        assert widgets.bounds3d(pts) == (0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
        assert widgets.bounds3d(np.full((3, 3), np.nan)) is None
        assert widgets.bounds3d(np.zeros((0, 3))) is None

    def test_thinning_is_a_no_op_below_the_limit(self):
        assert widgets._thin_indices(10, 20) is None
        thinned = widgets._thin_indices(1000, 100)
        assert thinned is not None and len(thinned) <= 100
        assert thinned[0] == 0 and thinned[-1] == 999

    def test_the_depth_fade_runs_from_far_to_near(self):
        fade = widgets._depth_fade(np.array([-1.0, 0.0, 1.0]))
        assert fade[0] == pytest.approx(1.0)
        assert fade[-1] == pytest.approx(widgets._SCENE3D_FAR_ALPHA)
        # A single depth plane must not divide by zero.
        assert np.allclose(widgets._depth_fade(np.full(4, 0.5)), 1.0)
        assert np.allclose(widgets._depth_fade(np.full(4, np.nan)), 1.0)


# ----------------------------------------------------------------------------------
# The geometry the panel hands the widget
# ----------------------------------------------------------------------------------


class TestPreviewGeometry:
    def test_every_generator_builds_drawable_finite_geometry(self, ws):
        """A thumbnail that is blank for one object in the catalogue is a broken catalogue."""
        panel = _panel(ws)
        for spec in _every_spec():
            panel._select(spec.key)
            panel._refresh_preview()
            panel._refresh_drawable()
            drawable = panel._drawable
            assert drawable is not None, spec.key
            points = drawable["points"]
            assert len(points) > 0, spec.key
            assert np.isfinite(points).any(), spec.key
            assert widgets.bounds3d(points) is not None, spec.key
            has_primitive = (
                drawable["dots"]
                or drawable.get("segments") is not None
                or drawable.get("triangles") is not None
            )
            assert has_primitive, spec.key

    @pytest.mark.parametrize("kind", list(layerops3d.KIND3D_KEYS))
    def test_every_kind_stays_under_the_widget_caps(self, ws, kind):
        """The caps are a floor under the frame time; the panel must not lean on them."""
        panel = _panel(ws)
        for spec in _every_spec():
            panel._select(spec.key)
            panel.kind = kind
            panel._refresh_preview()
            panel._refresh_drawable()
            drawable = panel._drawable
            segments = drawable.get("segments")
            triangles = drawable.get("triangles")
            if segments is not None:
                assert len(segments) <= widgets.MAX_SCENE3D_SEGMENTS, (spec.key, kind)
            if triangles is not None:
                assert len(triangles) <= widgets.MAX_SCENE3D_TRIANGLES, (spec.key, kind)
            if drawable["dots"]:
                assert len(drawable["points"]) <= widgets.MAX_SCENE3D_POINTS, (spec.key, kind)

    def test_the_colours_are_one_per_vertex(self, ws):
        panel = _panel(ws)
        for spec in _every_spec():
            panel._select(spec.key)
            panel._refresh_preview()
            panel._refresh_drawable()
            drawable = panel._drawable
            assert drawable["colors"].shape == (len(drawable["points"]), 4), spec.key
            assert np.isfinite(drawable["colors"]).all(), spec.key

    def test_the_colormap_choice_reaches_the_thumbnail(self, ws):
        """A preview in the wrong colormap does not predict anything."""
        panel = _panel(ws)
        panel._select("sphere")
        panel._refresh_preview()
        panel._refresh_drawable()
        viridis = panel._drawable["colors"].copy()
        panel.cmap = "magma"
        panel._refresh_drawable()
        assert not np.allclose(viridis, panel._drawable["colors"])

    def test_the_colour_column_drives_the_gradient(self, ws):
        """A helix is coloured along t, so the two ends of the path differ."""
        panel = _panel(ws)
        panel._select("helix")
        panel._refresh_preview()
        panel._refresh_drawable()
        colors = panel._drawable["colors"]
        assert not np.allclose(colors[0], colors[-1])

    def test_alpha_is_applied_but_floored(self, ws):
        """A volume cloud at alpha 0.02 works by stacking a million points; this has 2000."""
        panel = _panel(ws)
        panel._select("ball")
        panel.alpha = 0.5
        panel._refresh_preview()
        panel._refresh_drawable()
        assert panel._drawable["colors"][:, 3].max() == pytest.approx(0.5, abs=1e-5)
        panel.alpha = 0.02
        panel._refresh_drawable()
        from glplot.gui.panels.objects3d import PREVIEW_MIN_ALPHA

        assert panel._drawable["colors"][:, 3].max() == pytest.approx(PREVIEW_MIN_ALPHA, abs=1e-5)

    def test_a_surface_keeps_its_lattice_through_the_decimation(self, ws):
        """Dropping arbitrary rows from a surface makes nu * nv != N and it stops gridding."""
        from glplot.gui.panels.objects3d import PREVIEW_SAMPLES

        panel = _panel(ws)
        panel._select("torus")
        spec = panel.spec()
        n = spec.grid_shape(PREVIEW_SAMPLES)
        rows, shape = panel._preview_rows("surface3d", spec, n[0] * n[1])
        assert shape is not None
        assert shape[0] * shape[1] == len(rows)
        assert len(np.unique(rows)) == len(rows)

    def test_a_kind_the_object_cannot_satisfy_falls_back_to_points(self, ws):
        """A space curve has no lattice, so 'surface' cannot draw it -- say so, show it anyway."""
        panel = _panel(ws)
        panel._select("helix")
        panel.kind = "surface3d"
        panel._refresh_preview()
        panel._refresh_drawable()
        assert panel._drawable["dots"] is True
        assert panel._drawable["triangles"] is None
        assert panel._preview_note
        assert len(panel._drawable["points"]) > 0

    def test_the_drawable_is_change_gated(self, ws):
        """Rebuilding a triangulation every frame is what makes a slider unusable."""
        panel = _panel(ws)
        panel._select("torus")
        panel._refresh_preview()
        panel._refresh_drawable()
        first = panel._drawable
        panel._refresh_drawable()
        assert panel._drawable is first
        panel.current_params()["R"] = 3.0
        panel._refresh_preview()
        panel._refresh_drawable()
        assert panel._drawable is not first

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda p: setattr(p, "cmap", "turbo"),
            lambda p: setattr(p, "kind", "scatter3d"),
            lambda p: setattr(p, "alpha", 0.4),
            lambda p: setattr(p, "_color_override", "z"),
        ],
    )
    def test_every_appearance_input_invalidates_the_drawable(self, ws, mutate):
        panel = _panel(ws)
        panel._select("torus")
        panel._refresh_preview()
        panel._refresh_drawable()
        first = panel._drawable
        mutate(panel)
        panel._refresh_drawable()
        assert panel._drawable is not first

    def test_orbiting_does_not_rebuild_the_geometry(self, ws):
        """The camera re-projects cached vertices; a drag must not re-run the generator."""
        panel = _panel(ws)
        panel._select("klein")
        panel._refresh_preview()
        panel._refresh_drawable()
        first = panel._drawable
        panel.preview_camera.orbit(37.0, 11.0)
        panel._refresh_drawable()
        assert panel._drawable is first

    def test_a_failed_generation_leaves_the_last_good_thumbnail(self, ws, monkeypatch):
        panel = _panel(ws)
        panel._select("sphere")
        panel._refresh_preview()
        panel._refresh_drawable()
        good = panel._drawable
        assert good is not None

        def boom(*args, **kwargs):
            raise generators3d.GeneratorError("nope")

        monkeypatch.setattr(generators3d.Generator3D, "generate", boom)
        panel.current_params()["radius"] = 4.0
        panel._refresh_preview()
        panel._refresh_drawable()
        assert panel._error == "nope"
        assert len(panel._drawable["points"]) > 0


# ----------------------------------------------------------------------------------
# The drawing
# ----------------------------------------------------------------------------------


class TestPreviewDraws:
    def test_it_draws_the_thumbnail_for_every_generator(self, harness, ws, all_sections):
        """Every object in the catalogue, drawn for real, with every section open.

        This is the test that catches a wrong pyimgui signature. A state-only assertion
        cannot: ``draw_list.add_text`` with the C++ font overload is a perfectly ordinary
        Python call right up until the frame is submitted.
        """
        panel = _panel(ws)
        specs = _every_spec()
        assert len(specs) >= 27, "the catalogue shrank; this test is meant to cover all of it"
        for spec in specs:
            panel._select(spec.key)
            panel._refresh_preview()
            panel._refresh_drawable()
            assert panel._drawable is not None and len(panel._drawable["points"]) > 0, spec.key
            _frame(harness, panel)

    @pytest.mark.parametrize("kind", list(layerops3d.KIND3D_KEYS))
    @pytest.mark.parametrize("key", ["helix", "sphere", "gaussian_cloud", "vortex"])
    def test_it_draws_every_kind_of_every_shape(self, harness, ws, all_sections, key, kind):
        """One object per category crossed with every 3D kind -- points, lines and triangles."""
        panel = _panel(ws)
        panel._select(key)
        panel.kind = kind
        _frame(harness, panel)

    def test_it_draws_while_spinning(self, harness, ws, all_sections):
        panel = _panel(ws)
        panel._select("torus_knot")
        before = panel.preview_camera.azim
        for _ in range(4):
            _frame(harness, panel)
        assert panel.preview_camera.azim != before

    def test_it_draws_with_the_spin_off(self, harness, ws, all_sections):
        panel = _panel(ws)
        panel.preview_spin = False
        before = panel.preview_camera.azim
        _frame(harness, panel)
        assert panel.preview_camera.azim == before

    def test_it_draws_an_empty_scene(self, harness, ws, all_sections):
        """A generator that produced nothing yet must not blank or crash the panel."""
        panel = _panel(ws)
        panel._preview = {}
        panel._drawable = panel._build_drawable()
        _frame(harness, panel)

    def test_it_draws_with_an_error_showing(self, harness, ws, all_sections, monkeypatch):
        panel = _panel(ws)
        panel._select("shell")

        def boom(*args, **kwargs):
            raise generators3d.GeneratorError("out of range")

        monkeypatch.setattr(generators3d.Generator3D, "generate", boom)
        panel.current_params()["turns"] = 9.0
        _frame(harness, panel)
        assert panel._error == "out of range"

    def test_the_bare_widget_draws_every_primitive_mix(self, harness):
        """``mini_scene3d`` on its own: dots, lines, triangles, all three, and nothing."""
        pts = _sphere_points(12)
        n = len(pts)
        colors = np.tile(np.array([0.2, 0.6, 0.9, 1.0]), (n, 1))
        segments = np.column_stack([np.arange(n - 1), np.arange(1, n)])
        triangles = np.column_stack([np.arange(n - 2), np.arange(1, n - 1), np.arange(2, n)])
        cases = [
            dict(dots=True),
            dict(segments=segments),
            dict(triangles=triangles),
            dict(dots=True, segments=segments, triangles=triangles, colors=colors),
            dict(dots=True, show_box=False),
            dict(dots=True, colors=np.zeros((3, 4))),  # wrong length -> default colour
        ]
        for case in cases:
            harness.new_frame()
            harness.begin("W")
            widgets.mini_scene3d("bare", pts, camera=Camera3D(), **case)
            harness.end()
            harness.render()
            assert harness.get_draw_data() is not None

    def test_the_bare_widget_survives_degenerate_input(self, harness):
        cases = [
            (np.zeros((0, 3)), {}),
            (np.full((5, 3), np.nan), {"dots": True}),
            (np.zeros((5, 3)), {"dots": True}),  # every point coincident
            (_sphere_points(5), {"segments": np.array([[0, 999]])}),  # out-of-range index
            (_sphere_points(5), {"triangles": np.array([[0, 1]])}),  # wrong arity
            (_sphere_points(5), {"segments": np.zeros((0, 2), dtype=int)}),
            ("not an array", {"dots": True}),
        ]
        for points, kwargs in cases:
            harness.new_frame()
            harness.begin("W")
            widgets.mini_scene3d("bad", points, camera=Camera3D(), **kwargs)
            harness.end()
            harness.render()
            assert harness.get_draw_data() is not None

    def test_a_camera_that_cannot_build_a_matrix_is_survived(self, harness):
        class Broken:
            def mvp(self, aspect, bounds):
                raise RuntimeError("no matrix for you")

        harness.new_frame()
        harness.begin("W")
        widgets.mini_scene3d("broken", _sphere_points(5), camera=Broken(), dots=True)
        harness.end()
        harness.render()
        assert harness.get_draw_data() is not None

    def test_the_vectorised_colour_pack_matches_imgui(self, harness):
        """The thumbnail packs a few thousand ImU32s per frame without calling imgui."""
        harness.new_frame()
        harness.begin("W")
        samples = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.5],
                [0.2, 0.4, 0.6, 0.25],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
        packed = widgets._pack_u32(samples).tolist()
        expected = [harness.get_color_u32_rgba(*row) for row in samples.tolist()]
        harness.end()
        harness.render()
        assert packed == expected
        assert widgets._pack_u32(np.zeros((2, 3))).size == 0


# ----------------------------------------------------------------------------------
# Orbit, and the angle it carries
# ----------------------------------------------------------------------------------


class TestOrbitDrag:
    def _drag(self, imgui, camera, start=(100.0, 100.0), delta=(40.0, 20.0)):
        """Press inside the thumbnail and drag: hover frame, press frame, move frame."""
        io = imgui.get_io()
        points = _sphere_points(8)
        moved = []

        def frame():
            imgui.new_frame()
            imgui.set_next_window_position(0.0, 0.0)
            imgui.set_next_window_size(400.0, 300.0)
            imgui.begin("Orbit", flags=imgui.WINDOW_NO_TITLE_BAR)
            moved.append(
                widgets.mini_scene3d("orbit", points, camera=camera, dots=True, height=200.0)
            )
            imgui.end()
            imgui.render()
            assert imgui.get_draw_data() is not None

        io.mouse_pos = start
        frame()
        io.mouse_down[0] = True
        frame()
        io.mouse_pos = (start[0] + delta[0], start[1] + delta[1])
        frame()
        io.mouse_down[0] = False
        return moved

    def test_dragging_orbits_the_camera_and_reports_it(self, harness):
        camera = Camera3D()
        moved = self._drag(harness, camera)
        assert moved[-1] is True
        assert moved[0] is False
        # Right and down, matplotlib's convention: azimuth falls, elevation rises.
        assert camera.azim == pytest.approx(-45.0 - 40.0 * widgets.SCENE3D_ORBIT_DEG_PER_PX)
        assert camera.elev == pytest.approx(28.0 + 20.0 * widgets.SCENE3D_ORBIT_DEG_PER_PX)

    def test_orbit_can_be_switched_off(self, harness):
        camera = Camera3D()
        io = harness.get_io()
        io.mouse_pos = (100.0, 100.0)
        for pressed in (False, True, True):
            io.mouse_down[0] = pressed
            io.mouse_pos = (io.mouse_pos[0] + 20.0, io.mouse_pos[1])
            harness.new_frame()
            harness.begin("NoOrbit")
            widgets.mini_scene3d("static", _sphere_points(8), camera=camera, dots=True, orbit=False)
            harness.end()
            harness.render()
        io.mouse_down[0] = False
        assert camera.azim == pytest.approx(-45.0)

    def test_a_camera_without_orbit_is_not_called(self, harness):
        """The widget documents Camera3D but must not explode on a stand-in."""

        class NoOrbit:
            def mvp(self, aspect, bounds):
                return Camera3D().mvp(aspect, bounds)

        io = harness.get_io()
        io.mouse_pos = (100.0, 100.0)
        io.mouse_down[0] = True
        harness.new_frame()
        harness.begin("W")
        assert (
            widgets.mini_scene3d("noorbit", _sphere_points(6), camera=NoOrbit(), dots=True) is False
        )
        harness.end()
        harness.render()
        io.mouse_down[0] = False
        assert harness.get_draw_data() is not None


class TestOrientationCarry:
    def test_the_panel_stops_spinning_once_the_user_takes_over(
        self, harness, ws, all_sections, monkeypatch
    ):
        """A deliberate angle beats a decorative one."""
        panel = _panel(ws)
        monkeypatch.setattr(widgets, "mini_scene3d", lambda *a, **k: True)
        assert panel.preview_spin is True
        _frame(harness, panel)
        assert panel.preview_spin is False
        assert panel._preview_oriented is True

    def test_no_angle_is_carried_until_the_user_orbits(self, ws):
        """Otherwise Plot would inherit whatever azimuth the spin was passing through."""
        panel = _panel(ws)
        assert panel._preview_view() is None
        panel.preview_camera.orbit(30.0, 10.0)
        assert panel._preview_view() is None, "spinning alone must not count as a choice"
        panel._preview_oriented = True
        assert panel._preview_view() == pytest.approx((28.0 + 10.0, -45.0 + 30.0, 0.0))

    def test_plot_adopts_the_previewed_angle(self, ws, plot):
        panel = _panel(ws)
        panel._select("sphere")
        panel.samples = 400
        panel.preview_camera.elev = 61.0
        panel.preview_camera.azim = 17.0
        panel._preview_oriented = True
        panel._action_plot()
        ws.drain()
        assert not panel._error, panel._error
        assert plot.camera3d.elev == pytest.approx(61.0)
        assert plot.camera3d.azim == pytest.approx(17.0)

    def test_plot_leaves_the_figure_view_alone_when_the_preview_was_not_touched(self, ws, plot):
        panel = _panel(ws)
        panel._select("sphere")
        panel.samples = 400
        plot.set_ndim(3)
        plot.set_3d_view(elev=5.0, azim=95.0)
        panel.preview_camera.orbit(120.0, -40.0)  # spin, not a user choice
        panel._action_plot()
        ws.drain()
        assert plot.camera3d.elev == pytest.approx(5.0)
        assert plot.camera3d.azim == pytest.approx(95.0)

    def test_the_carried_angle_is_captured_at_click_time(self, ws, plot):
        """Undo/redo must reproduce the picture the command made, not a later spin."""
        panel = _panel(ws)
        panel._select("helix")
        panel.samples = 400
        panel.preview_camera.elev = 12.0
        panel.preview_camera.azim = -100.0
        panel._preview_oriented = True
        panel._action_plot()
        ws.drain()
        panel.preview_camera.orbit(180.0, 40.0)
        ws.undo.undo()
        ws.drain()
        ws.undo.redo()
        ws.drain()
        assert plot.camera3d.elev == pytest.approx(12.0)
        assert plot.camera3d.azim == pytest.approx(-100.0)

    def test_reset_view_clears_the_choice(self, harness, ws, all_sections, monkeypatch):
        panel = _panel(ws)
        monkeypatch.setattr(widgets, "mini_scene3d", lambda *a, **k: False)
        panel._preview_oriented = True
        panel.preview_spin = False
        panel.preview_camera.orbit(90.0, 20.0)
        _frame(harness, panel)
        assert panel._preview_view() is not None
        # Drive the button by hand: the harness cannot click, so run what it runs.
        panel.preview_camera.reset()
        panel.preview_spin = True
        panel._preview_oriented = False
        assert panel._preview_view() is None
        assert panel.preview_camera.azim == pytest.approx(-45.0)
        assert panel.preview_camera.elev == pytest.approx(28.0)
