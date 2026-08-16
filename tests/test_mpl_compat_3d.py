"""Test the ``Axes3D`` half of the matplotlib-parity surface, against real matplotlib.

A compatibility audit of GLPlot's 3D API found that roughly one sixth of ``Axes3D`` was
reachable and that several of the reachable parts diverged *silently*. This file pins the
fixes for the findings that had teeth, and it pins them the only way a compatibility claim
can be pinned: by asking matplotlib what it does and asserting GLPlot agrees.

Three kinds of assertion appear here, and the mix is deliberate.

1. **Signature parity**, via ``inspect.signature`` against the real
   ``mpl_toolkits.mplot3d.Axes3D``. A signature that drifts is how
   ``ax.view_init(30, 45, 0, "y")`` -- legal matplotlib -- became a ``TypeError``.
2. **The idiom actually running.** A matching signature proves nothing about behaviour, so
   every fix also runs the matplotlib spelling against GLPlot and inspects what was drawn.
   The audit's worst findings were calls that *succeeded* and drew the wrong picture.
3. **Warnings where GLPlot cannot comply.** A keyword accepted and dropped in silence is
   the failure this module exists to remove; where GLPlot cannot honour a matplotlib
   keyword the contract is that it says so, once, as a ``MatplotlibCompatWarning``.

No OpenGL and no GPU: ``GPULinePlot`` constructs headless and ``show()`` is never called.
matplotlib is driven on the Agg backend for the same reason.
"""

from __future__ import annotations

import inspect
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes as MplAxes  # noqa: E402
from matplotlib.figure import Figure as MplFigure  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402

import glplot.pyplot as gplt  # noqa: E402


@pytest.fixture(autouse=True)
def clean_state():
    """Every test starts from a fresh pyplot module state.

    ``_cleanup_pyplot_state`` also clears the ``MatplotlibCompatWarning`` "already said
    this" registry, without which the first test to trip a warning would silence it for
    every test after it and make "does this warn?" depend on execution order.
    """
    gplt._cleanup_pyplot_state()
    plt.close("all")
    yield
    gplt._cleanup_pyplot_state()
    plt.close("all")


@pytest.fixture
def ax3d():
    """A genuine 3D axes obtained the way a matplotlib script obtains one."""
    fig = gplt.figure("compat3d")
    return fig.add_subplot(projection="3d")


@pytest.fixture
def mpl3d():
    """The same thing in real matplotlib, for side-by-side assertions."""
    return plt.figure().add_subplot(projection="3d")


def _grid(n: int = 20):
    x, y = np.meshgrid(np.linspace(-2.0, 2.0, n), np.linspace(-2.0, 2.0, n))
    return x, y, np.sin(np.sqrt(x**2 + y**2))


def _bare_signature(func, drop_self: bool = False) -> inspect.Signature:
    """``func``'s signature with every annotation stripped.

    Names, kinds, order and defaults are the contract a caller depends on; annotations are
    not, and GLPlot annotates where matplotlib does not, so comparing raw ``Signature``
    objects would fail on a difference nobody can observe from a call site.
    """
    params = list(inspect.signature(func).parameters.values())
    if drop_self:
        params = params[1:]
    return inspect.Signature([p.replace(annotation=inspect.Parameter.empty) for p in params])


def _signature_without_self(func) -> inspect.Signature:
    """An unbound ``Axes3D`` method's signature as a module-level function would spell it."""
    return _bare_signature(func, drop_self=True)


def _layer_kinds(fig) -> list:
    """The ``layer_type`` of every non-decoration layer on the figure's active panel."""
    system = set(getattr(type(fig), "_SYSTEM_3D_ARTISTS", ()) or ())
    return [
        layer.layer_type for layer in fig.scene.layers if layer.metadata.get("artist") not in system
    ]


# ======================================================================================
# Group 1 -- crashes and silent wrong output
# ======================================================================================


class TestSetProjType:
    """D1: ``set_proj_type("ortho", np.inf)`` used to brick the figure."""

    def test_signature_matches_matplotlib(self):
        assert _signature_without_self(Axes3D.set_proj_type) == _bare_signature(gplt.set_proj_type)

    def test_proj_type_is_required_as_in_matplotlib(self, ax3d, mpl3d):
        with pytest.raises(TypeError):
            mpl3d.set_proj_type()
        with pytest.raises(TypeError):
            ax3d.set_proj_type()

    def test_ortho_with_inf_does_not_zero_the_field_of_view(self, ax3d):
        """THE crash. ``fov = degrees(2*atan(1/inf))`` is 0 and every matrix divides by it.

        matplotlib documents ``np.inf`` as the orthographic focal length, so this is not an
        exotic call: it is the spelling a ported script contains.
        """
        before = ax3d.figure.camera3d.fov
        ax3d.set_proj_type("ortho", np.inf)
        assert ax3d.figure.camera3d.fov == before > 0.0

    def test_the_figure_still_works_after_ortho_inf(self, ax3d):
        """The audit's exact follow-on failures: every later 3D call raised here."""
        ax3d.set_proj_type("ortho", np.inf)
        ax3d.set_proj_type("persp")
        ax3d.set_box_aspect((1, 1, 1))
        ax3d.scatter3d([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
        assert ax3d.figure.camera3d.fov > 0.0

    def test_matplotlib_accepts_ortho_inf_too(self, mpl3d):
        """The parity half: this call is legal there, so it must be legal here."""
        assert mpl3d.set_proj_type("ortho", np.inf) is None

    def test_finite_focal_length_with_ortho_is_refused_as_in_matplotlib(self, ax3d, mpl3d):
        with pytest.raises(ValueError) as mpl_err:
            mpl3d.set_proj_type("ortho", 2)
        with pytest.raises(ValueError) as gl_err:
            ax3d.set_proj_type("ortho", 2)
        assert str(gl_err.value) == str(mpl_err.value)

    def test_unknown_proj_type_is_refused(self, ax3d, mpl3d):
        with pytest.raises(ValueError):
            mpl3d.set_proj_type("isometric")
        with pytest.raises(ValueError) as err:
            ax3d.set_proj_type("isometric")
        assert "'persp'" in str(err.value) and "'ortho'" in str(err.value)

    def test_non_positive_focal_length_is_refused_with_matplotlibs_message(self, ax3d, mpl3d):
        with pytest.raises(ValueError) as mpl_err:
            mpl3d.set_proj_type("persp", 0)
        with pytest.raises(ValueError) as gl_err:
            ax3d.set_proj_type("persp", 0)
        assert str(gl_err.value) == str(mpl_err.value)

    def test_persp_focal_length_sets_the_equivalent_field_of_view(self, ax3d):
        ax3d.set_proj_type("persp", 1.0)
        assert ax3d.figure.camera3d.fov == pytest.approx(90.0)

    def test_bare_persp_leaves_the_field_of_view_alone(self, ax3d):
        """Selecting perspective is not a request to re-frame the scene."""
        ax3d.set_proj_type("persp", 0.5)
        chosen = ax3d.figure.camera3d.fov
        ax3d.set_proj_type("ortho")
        ax3d.set_proj_type("persp")
        assert ax3d.figure.camera3d.fov == chosen


class TestViewInitSignature:
    """D6: ``vertical_axis`` and ``share`` were keyword-only."""

    def test_signature_matches_matplotlib(self):
        assert _signature_without_self(Axes3D.view_init) == _bare_signature(gplt.view_init)

    def test_the_full_positional_call_works(self, ax3d, mpl3d):
        mpl3d.view_init(30, 45, 0, "y")
        ax3d.view_init(30, 45, 0, "y")
        assert (ax3d.elev, ax3d.azim, ax3d.roll) == (30.0, 45.0, 0.0)
        assert ax3d.figure.camera3d.up_axis == "y"

    def test_a_bare_call_restores_z_as_the_up_axis(self, ax3d):
        """matplotlib's ``vertical_axis`` default is ``"z"``, not "leave it alone"."""
        ax3d.view_init(vertical_axis="y")
        ax3d.view_init(elev=10)
        assert ax3d.figure.camera3d.up_axis == "z"


class TestPlotIsThreeDAware:
    """D2: ``ax.plot(x, y, z)`` drew a flat 2D picture with z read as a second series."""

    def test_matplotlib_draws_a_line3d(self, mpl3d):
        (line,) = mpl3d.plot([0, 1, 2], [0, 1, 2], [0, 1, 2])
        assert type(line).__name__ == "Line3D"

    def test_three_positionals_draw_one_3d_line(self, ax3d):
        t = np.linspace(0.0, 1.0, 8)
        artists = ax3d.plot(t, t, t)
        assert len(artists) == 1
        assert _layer_kinds(ax3d.figure) == ["wireframe3d"]

    def test_the_z_data_is_the_z_data(self, ax3d):
        artists = ax3d.plot([0.0, 1.0], [0.0, 1.0], [0.0, 10.0])
        assert artists[0].vertices[:, 2].tolist() == [0.0, 10.0]

    def test_zs_keyword_is_honoured(self, ax3d):
        t = np.linspace(0.0, 1.0, 5)
        artists = ax3d.plot(t, t, zs=t, zdir="z")
        assert _layer_kinds(ax3d.figure) == ["wireframe3d"]
        assert artists[0].vertices[:, 2] == pytest.approx(t.repeat(2)[1:-1], abs=1e-6)

    def test_a_scalar_zs_is_broadcast(self, ax3d):
        artists = ax3d.plot([0.0, 1.0], [0.0, 1.0], zs=3.0)
        assert artists[0].vertices[:, 2].tolist() == [3.0, 3.0]

    def test_zdir_juggles_the_axes_the_way_matplotlib_does(self, ax3d):
        """``zdir="y"`` projects the 2D curve onto a wall; it is not decoration."""
        from mpl_toolkits.mplot3d import art3d

        xs, ys, zs = np.array([1.0]), np.array([2.0]), np.array([3.0])
        expected = art3d.juggle_axes(xs, ys, zs, "y")
        artists = ax3d.plot([1.0, 1.0], [2.0, 2.0], zs=3.0, zdir="y")
        got = artists[0].vertices[0]
        assert got == pytest.approx([float(e[0]) for e in expected], abs=1e-6)

    def test_a_format_string_still_works_in_3d(self, ax3d):
        artists = ax3d.plot([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], "r-")
        assert _layer_kinds(ax3d.figure) == ["wireframe3d"]
        assert artists[0].colors[0][:3] == pytest.approx([1.0, 0.0, 0.0])

    def test_two_positionals_are_a_line_at_z_zero(self, ax3d):
        """matplotlib's ``zs`` default is 0, so a 2D call on 3D axes lands on the floor."""
        artists = ax3d.plot([0.0, 1.0], [0.0, 1.0])
        assert artists[0].vertices[:, 2].tolist() == [0.0, 0.0]

    def test_the_2d_path_is_untouched(self):
        gplt.figure("flat")
        artists = gplt.plot([0.0, 1.0], [0.0, 1.0], "b-", [0.0, 1.0], [1.0, 0.0], "r:")
        assert len(artists) == 2
        assert all(a.layer_type == "polyline" for a in artists)

    def test_zs_on_2d_axes_warns_instead_of_vanishing(self):
        gplt.figure("flat")
        with pytest.warns(gplt.MatplotlibCompatWarning, match="zs"):
            gplt.plot([0.0, 1.0], [0.0, 1.0], zs=[0.0, 1.0])

    def test_zs_twice_is_a_type_error(self, ax3d):
        with pytest.raises(TypeError, match="zs"):
            ax3d.plot([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], zs=[0.0, 1.0])

    def test_an_unknown_zdir_is_refused(self, ax3d):
        with pytest.raises(ValueError, match="zdir"):
            ax3d.plot([0.0, 1.0], [0.0, 1.0], zdir="w")


class TestScatterIsThreeDAware:
    """D2: ``ax.scatter(x, y, z)`` bound ``z`` to GLPlot's third parameter, ``color``."""

    def test_matplotlib_draws_a_path3dcollection(self, mpl3d):
        coll = mpl3d.scatter([0, 1], [0, 1], [0, 1])
        assert type(coll).__name__ == "Path3DCollection"

    def test_three_positionals_draw_a_3d_point_cloud(self, ax3d):
        layer = ax3d.scatter([0.0, 1.0], [0.0, 1.0], [0.0, 10.0])
        assert layer.layer_type == "scatter3d"
        assert layer.vertices.shape[1] == 3
        assert layer.vertices[:, 2].tolist() == [0.0, 10.0]

    def test_zs_keyword_is_honoured(self, ax3d):
        layer = ax3d.scatter([0.0, 1.0], [0.0, 1.0], zs=[2.0, 3.0])
        assert layer.vertices[:, 2].tolist() == [2.0, 3.0]

    def test_zdir_is_honoured(self, ax3d):
        layer = ax3d.scatter([1.0], [2.0], 3.0, "y")
        assert layer.vertices[0].tolist() == [1.0, 3.0, 2.0]

    def test_the_third_positional_is_not_the_colour(self, ax3d):
        """The precise shape of the old bug: z became a viridis ramp on a flat scatter."""
        layer = ax3d.scatter([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        assert layer.vertices.shape == (3, 3)

    def test_an_explicit_colour_beats_the_z_ramp(self, ax3d):
        layer = ax3d.scatter([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], color="red")
        assert layer.colors[0][:3] == pytest.approx([1.0, 0.0, 0.0])

    def test_c_still_maps_a_colormap(self, ax3d):
        layer = ax3d.scatter([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], c=[0.0, 1.0], cmap="plasma")
        assert not np.allclose(layer.colors[0], layer.colors[1])

    def test_depthshade_false_warns_rather_than_vanishing(self, ax3d):
        with pytest.warns(gplt.MatplotlibCompatWarning, match="depthshade"):
            ax3d.scatter([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], depthshade=False)

    def test_the_2d_signature_is_still_what_introspection_reports(self):
        """The dispatcher must not cost the 2D signature; ``__wrapped__`` preserves it."""
        params = inspect.signature(gplt.scatter).parameters
        for name in ("x", "y", "color", "size", "c", "s", "cmap", "vmin", "vmax"):
            assert name in params

    def test_the_2d_path_is_untouched(self):
        gplt.figure("flat")
        layer = gplt.scatter([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], c=[0.0, 0.5, 1.0], cmap="viridis")
        assert layer.layer_type == "scatter"

    def test_positional_2d_arguments_still_bind_the_old_way(self):
        """``scatter(x, y, color, size)`` is GLPlot's historical positional order."""
        gplt.figure("flat")
        layer = gplt.scatter([0.0, 1.0], [0.0, 1.0], "red", 12.0)
        assert layer.colors[0][:3] == pytest.approx([1.0, 0.0, 0.0])

    def test_duplicate_binding_is_a_type_error(self, ax3d):
        with pytest.raises(TypeError, match="zs"):
            ax3d.scatter([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], zs=[0.0, 1.0])


class TestScaleZNoLongerRewritesTheData:
    """D4: every 3D artist multiplied z by 0.7, so the z axis was in fabricated units."""

    @pytest.mark.parametrize(
        "artist",
        ["plot3d", "scatter3d", "plot_surface", "plot_wireframe", "bar3d"],
    )
    def test_the_default_is_one(self, artist):
        assert inspect.signature(getattr(gplt, artist)).parameters["scale_z"].default == 1.0

    def test_scatter3d_stores_the_z_it_was_given(self, ax3d):
        layer = ax3d.scatter3d([0.0, 1.0], [0.0, 1.0], [0.0, 10.0])
        assert layer.vertices[:, 2].tolist() == [0.0, 10.0]

    def test_plot3d_stores_the_z_it_was_given(self, ax3d):
        (layer,) = ax3d.plot3d([0.0, 1.0], [0.0, 1.0], [0.0, 10.0])
        assert layer.vertices[:, 2].max() == pytest.approx(10.0)

    def test_bar3d_bars_are_as_tall_as_asked(self, ax3d):
        """The audit's number: the box top used to come out at 0.7 for a dz of 1."""
        artists = ax3d.bar3d([0.0], [0.0], [0.0], 1.0, 1.0, 1.0)
        assert artists[0].vertices[:, 2].max() == pytest.approx(1.0)

    def test_plot_surface_keeps_the_z_range(self, ax3d):
        x, y, z = _grid(8)
        layer = ax3d.plot_surface(x, y, z)
        assert layer.vertices[:, 2].max() == pytest.approx(float(z.max()), rel=1e-5)

    def test_the_z_limit_read_back_is_in_data_units(self, ax3d):
        """``set_zlim()`` reported 0.7x the truth, which is what made it a data bug."""
        ax3d.scatter3d([0.0, 1.0], [0.0, 1.0], [0.0, 10.0])
        lo, hi = ax3d.set_zlim()
        assert lo <= 0.0 and hi >= 10.0

    def test_an_explicit_scale_z_still_works(self, ax3d):
        layer = ax3d.scatter3d([0.0, 1.0], [0.0, 1.0], [0.0, 10.0], scale_z=0.5)
        assert layer.vertices[:, 2].tolist() == [0.0, 5.0]


# ======================================================================================
# Group 2 -- the entry points a ported script hits on line 2
# ======================================================================================


class TestFigureAxesConstructors:
    """Missing #1: ``fig.add_subplot`` did not exist at all."""

    @pytest.mark.parametrize("name", ["add_subplot", "add_axes"])
    def test_matplotlib_has_it_and_so_do_we(self, name):
        assert hasattr(MplFigure, name)
        assert callable(getattr(gplt.figure("f"), name))

    def test_the_canonical_3d_script_survives_line_two(self):
        """The whole point: this is how nearly every mplot3d example begins."""
        x, y, z = _grid(10)
        fig = gplt.figure("canonical")
        ax = fig.add_subplot(projection="3d")
        ax.plot_surface(x, y, z)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=30, azim=-60)
        assert fig.is_3d_scene() is True
        assert ax.get_zlabel() == "z"

    def test_add_subplot_returns_an_axes_proxy(self):
        ax = gplt.figure("f").add_subplot(projection="3d")
        assert isinstance(ax, gplt.AxesProxy)

    @pytest.mark.parametrize("args", [(), (1, 1, 1), (2, 2, 3), (223,)])
    def test_every_grid_spelling_matplotlib_takes(self, args):
        fig = gplt.figure("f")
        assert isinstance(fig.add_subplot(*args), gplt.AxesProxy)

    def test_add_subplot_makes_the_new_axes_current(self):
        fig = gplt.figure("f")
        fig.add_subplot(1, 2, 1)
        second = fig.add_subplot(1, 2, 2)
        assert fig.active_panel_index == second._index

    def test_add_axes_honours_the_projection(self):
        fig = gplt.figure("f")
        # The rectangle cannot be placed (GLPlot's panels come from a grid), and saying so
        # is the contract -- so the warning is expected here, not incidental.
        with pytest.warns(gplt.MatplotlibCompatWarning):
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="3d")
        assert isinstance(ax, gplt.AxesProxy)
        assert fig.is_3d_scene() is True

    def test_add_axes_without_a_rectangle_is_silent(self):
        fig = gplt.figure("f")
        with warnings.catch_warnings():
            warnings.simplefilter("error", gplt.MatplotlibCompatWarning)
            assert isinstance(fig.add_axes(projection="3d"), gplt.AxesProxy)


class TestGcaAndAxesReturnAnAxes:
    """D8: both returned the bare figure, which carries none of the Axes API."""

    def test_gca_gives_a_working_axes(self):
        gplt.figure("f")
        ax = gplt.gca()
        assert isinstance(ax, gplt.AxesProxy)
        assert ax.figure is gplt.gcf()

    def test_the_audits_failing_snippet_now_runs(self):
        """``ax = plt.axes(projection="3d"); ax.plot_surface(...)`` -- the #2 3D idiom."""
        x, y, z = _grid(8)
        ax = gplt.axes(projection="3d")
        ax.plot_surface(x, y, z)
        ax.set_zlim(-1, 1)
        ax.view_init(30, 45)
        ax.set_zlabel("z")
        ax.set_xlabel("x")
        assert ax.get_zlim() == (-1.0, 1.0)

    def test_gca_follows_the_active_panel(self):
        fig, axs = gplt.subplots(1, 2)
        axs[1]._activate()
        assert gplt.gca()._index == axs[1]._index


class TestAxes3DMethodNames:
    """Missing #2-#8: names that were one dictionary line away from working."""

    #: Every name here is a real ``Axes3D`` attribute -- the test asserts that too, so the
    #: list cannot drift into asserting parity with something matplotlib does not have.
    NAMES = [
        "set_zlabel",
        "get_zlabel",
        "set_zlim",
        "get_zlim",
        "set_zlim3d",
        "set_xlim3d",
        "set_ylim3d",
        "set_zticks",
        "get_zticks",
        "set_zticklabels",
        "set_zscale",
        "invert_zaxis",
        "set_axis_off",
        "set_axis_on",
        "plot3D",
        "scatter3D",
        "view_init",
        "set_proj_type",
        "set_box_aspect",
        "get_box_aspect",
        "plot_surface",
        "plot_wireframe",
        "bar3d",
        "voxels",
    ]

    @pytest.mark.parametrize("name", NAMES)
    def test_matplotlib_has_it_and_so_do_we(self, ax3d, name):
        assert hasattr(Axes3D, name), f"{name} is not an Axes3D method; fix the test"
        assert callable(getattr(ax3d, name))

    def test_set_zlabel_writes_the_label(self, ax3d):
        ax3d.set_zlabel("height")
        assert ax3d.get_zlabel() == "height"
        assert ax3d.figure.axes3d.zlabel == "height"

    def test_get_zlim_reads_what_set_zlim_wrote(self, ax3d):
        ax3d.scatter3d([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
        ax3d.set_zlim(-2.0, 2.0)
        assert ax3d.get_zlim() == (-2.0, 2.0)

    def test_get_zticks_returns_the_ticks_on_the_plot(self, ax3d):
        ax3d.scatter3d([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
        ax3d.set_zlim(0.0, 1.0)
        ticks = ax3d.get_zticks()
        assert isinstance(ticks, np.ndarray) and ticks.size
        assert ticks.min() >= 0.0 and ticks.max() <= 1.0

    def test_get_zticks_signature_matches_matplotlib(self):
        assert _signature_without_self(Axes3D.get_zticks) == _bare_signature(gplt.get_zticks)

    def test_set_zticks_applies_the_count_and_says_what_it_cannot_do(self, ax3d):
        ax3d.scatter3d([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
        with pytest.warns(gplt.MatplotlibCompatWarning, match="ticks"):
            ax3d.set_zticks([0.0, 0.5, 1.0])
        assert ax3d.figure.axes3d.tick_count == 3

    def test_set_zticklabels_warns_rather_than_vanishing(self, ax3d):
        with pytest.warns(gplt.MatplotlibCompatWarning, match="labels"):
            ax3d.set_zticklabels(["a", "b"])

    def test_set_zscale_accepts_linear_and_warns_on_log(self, ax3d):
        ax3d.set_zscale("linear")
        with pytest.warns(gplt.MatplotlibCompatWarning):
            ax3d.set_zscale("log")
        with pytest.raises(ValueError):
            ax3d.set_zscale("nonsense")

    def test_invert_zaxis_warns_rather_than_pretending(self, ax3d):
        with pytest.warns(gplt.MatplotlibCompatWarning, match="invert_zaxis"):
            ax3d.invert_zaxis()

    def test_set_axis_off_and_on_drive_the_3d_decoration(self, ax3d):
        ax3d.scatter3d([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
        ax3d.set_axis_off()
        assert ax3d.figure.axes3d.show_axes is False
        ax3d.set_axis_on()
        assert ax3d.figure.axes3d.show_axes is True

    def test_set_axis_off_also_works_on_2d_axes(self):
        """matplotlib's spelling has to work in both projections; a script cannot choose."""
        gplt.figure("flat")
        gplt.plot([0.0, 1.0], [0.0, 1.0])
        gplt.gca().set_axis_off()
        assert gplt.gcf().options.axis_show_frame is False

    def test_the_capital_d_aliases_draw_in_3d(self, ax3d):
        """``plot3D``/``scatter3D`` are matplotlib's own names, used across the gallery."""
        assert Axes3D.plot3D is Axes3D.plot
        assert Axes3D.scatter3D is Axes3D.scatter
        ax3d.plot3D([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
        ax3d.scatter3D([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
        assert _layer_kinds(ax3d.figure) == ["wireframe3d", "scatter3d"]

    def test_elev_azim_roll_read_back_after_view_init(self, ax3d, mpl3d):
        mpl3d.view_init(elev=12, azim=34, roll=5)
        assert (mpl3d.elev, mpl3d.azim, mpl3d.roll) == (12, 34, 5)
        ax3d.view_init(elev=12, azim=34, roll=5)
        assert (ax3d.elev, ax3d.azim, ax3d.roll) == (12.0, 34.0, 5.0)

    def test_the_angles_are_writable_as_they_are_in_matplotlib(self, ax3d):
        ax3d.elev = 40.0
        ax3d.azim = -20.0
        ax3d.roll = 3.0
        assert (ax3d.figure.camera3d.elev, ax3d.figure.camera3d.azim) == (40.0, -20.0)
        assert ax3d.figure.camera3d.roll == 3.0

    def test_reading_an_angle_does_not_steal_the_current_axes(self):
        """A question must not have a side effect: reading panel 1 must not select it."""
        fig, axs = gplt.subplots(1, 2, subplot_kw={"projection": "3d"})
        axs[0]._activate()
        _ = axs[1].azim
        assert fig.active_panel_index == axs[0]._index


# ======================================================================================
# Group 3 -- swallowed keywords
# ======================================================================================


class TestPlotSurfaceKeywords:
    """D5: every matplotlib keyword was accepted and dropped without a word."""

    def test_color_paints_a_flat_surface(self, ax3d):
        x, y, z = _grid(8)
        layer = ax3d.plot_surface(x, y, z, color="red")
        assert np.allclose(layer.colors, layer.colors[0])
        assert layer.colors[0][:3] == pytest.approx([1.0, 0.0, 0.0])

    def test_matplotlib_takes_color_too(self, mpl3d):
        x, y, z = _grid(8)
        assert mpl3d.plot_surface(x, y, z, color="red") is not None

    def test_rcount_and_ccount_decimate_the_mesh(self, ax3d):
        x, y, z = _grid(20)
        full = ax3d.plot_surface(x, y, z)
        coarse = ax3d.plot_surface(x, y, z, rcount=5, ccount=5)
        assert len(coarse.vertices) == 25 < len(full.vertices)

    def test_a_count_and_a_stride_together_are_refused(self, ax3d):
        x, y, z = _grid(8)
        with pytest.raises(ValueError, match="rcount"):
            ax3d.plot_surface(x, y, z, rcount=4, rstride=2)

    def test_vmin_and_vmax_pin_the_colormap(self, ax3d):
        x, y, z = _grid(8)
        wide = ax3d.plot_surface(x, y, z, vmin=-100.0, vmax=100.0)
        tight = ax3d.plot_surface(x, y, z)
        assert not np.allclose(wide.colors, tight.colors)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"shade": False},
            {"lightsource": object()},
            {"facecolors": np.zeros((8, 8, 4))},
            {"edgecolor": "k"},
            {"linewidth": 0.0},
            {"antialiased": False},
            {"norm": object()},
        ],
    )
    def test_what_cannot_be_honoured_warns(self, ax3d, kwargs):
        x, y, z = _grid(8)
        with pytest.warns(gplt.MatplotlibCompatWarning):
            ax3d.plot_surface(x, y, z, **kwargs)

    def test_an_unknown_keyword_warns_rather_than_vanishing(self, ax3d):
        x, y, z = _grid(8)
        with pytest.warns(gplt.MatplotlibCompatWarning, match="nonsense"):
            ax3d.plot_surface(x, y, z, nonsense=1)


class TestPlotWireframeKeywords:
    def test_rcount_and_ccount_are_honoured(self, ax3d):
        x, y, z = _grid(20)
        coarse = ax3d.plot_wireframe(x, y, z, rcount=2, ccount=2)
        fine = ax3d.plot_wireframe(x, y, z, rcount=20, ccount=20)
        assert len(coarse[0].vertices) < len(fine[0].vertices)

    @pytest.mark.parametrize("kwargs", [{"linestyle": "--"}, {"cmap": "viridis"}])
    def test_what_cannot_be_honoured_warns(self, ax3d, kwargs):
        x, y, z = _grid(8)
        with pytest.warns(gplt.MatplotlibCompatWarning):
            ax3d.plot_wireframe(x, y, z, **kwargs)


class TestBar3dKeywords:
    """D7 and the ``zsort``/``shade``/``lightsource`` no-ops."""

    def test_the_positional_order_matches_matplotlib(self):
        mpl_names = list(_signature_without_self(Axes3D.bar3d).parameters)
        gl_names = list(inspect.signature(gplt.bar3d).parameters)
        assert gl_names[:10] == mpl_names[:10]

    def test_a_positional_colour_works_here_as_it_does_there(self, ax3d, mpl3d):
        assert mpl3d.bar3d([0], [0], [0], 1, 1, 1, "r") is not None
        artists = ax3d.bar3d([0.0], [0.0], [0.0], 1.0, 1.0, 1.0, "r")
        reds = artists[0].colors[:, 0]
        assert reds.max() > 0.0
        assert artists[0].colors[:, 1].max() == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "kwargs", [{"zsort": "max"}, {"shade": False}, {"lightsource": object()}]
    )
    def test_what_cannot_be_honoured_warns(self, ax3d, kwargs):
        with pytest.warns(gplt.MatplotlibCompatWarning):
            ax3d.bar3d([0.0], [0.0], [0.0], 1.0, 1.0, 1.0, **kwargs)

    def test_the_defaults_do_not_warn(self, ax3d):
        with warnings.catch_warnings():
            warnings.simplefilter("error", gplt.MatplotlibCompatWarning)
            ax3d.bar3d([0.0], [0.0], [0.0], 1.0, 1.0, 1.0)


class TestAxisLimitAliases:
    """D3: ``zmin``/``zmax`` and friends were popped by nobody and returned unchanged."""

    def test_matplotlib_accepts_them(self, mpl3d):
        assert mpl3d.set_zlim(zmin=-3, zmax=3) == (-3.0, 3.0)

    def test_set_zlim_honours_zmin_and_zmax(self, ax3d):
        ax3d.scatter3d([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
        assert ax3d.set_zlim(zmin=-3.0, zmax=3.0) == (-3.0, 3.0)
        assert ax3d.set_zlim() == (-3.0, 3.0)

    def test_set_xlim3d_honours_xmin_and_xmax(self, ax3d):
        ax3d.scatter3d([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
        assert ax3d.set_xlim3d(xmin=1.0, xmax=2.0) == (1.0, 2.0)
        assert ax3d.figure.axes3d.xlim == (1.0, 2.0)

    def test_set_ylim3d_honours_ymin_and_ymax(self, ax3d):
        ax3d.scatter3d([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
        assert ax3d.set_ylim3d(ymin=1.0, ymax=2.0) == (1.0, 2.0)

    def test_naming_one_end_twice_is_a_type_error_as_in_matplotlib(self, ax3d, mpl3d):
        with pytest.raises(TypeError):
            mpl3d.set_zlim(0, zmin=1)
        with pytest.raises(TypeError, match="Cannot pass both"):
            ax3d.set_zlim(0.0, zmin=1.0)

    def test_an_unknown_keyword_is_a_type_error(self, ax3d):
        with pytest.raises(TypeError, match="unexpected keyword"):
            ax3d.set_zlim(zmim=-3.0)


class TestGrid3dKeywords:
    @pytest.mark.parametrize("kwargs", [{"which": "major"}, {"axis": "z"}])
    def test_what_cannot_be_honoured_warns(self, ax3d, kwargs):
        with pytest.warns(gplt.MatplotlibCompatWarning):
            ax3d.grid3d(True, **kwargs)

    def test_the_visible_flag_still_works(self, ax3d):
        ax3d.grid3d(False)
        assert ax3d.figure.axes3d.show_grid is False


class TestLayoutHelpersHonourTheProjection:
    """D9: ``subplot2grid`` and ``subplot_mosaic`` accepted ``projection`` and dropped it."""

    def test_matplotlib_gives_axes3d_for_subplot2grid(self):
        assert isinstance(plt.subplot2grid((2, 2), (0, 0), projection="3d"), Axes3D)

    def test_subplot2grid_makes_a_real_3d_panel(self):
        gplt.figure("f")
        ax = gplt.subplot2grid((2, 2), (0, 0), projection="3d")
        assert ax.panel.ndim == 3
        assert ax.figure.is_3d_scene() is True
        ax.plot_surface(*_grid(6))

    def test_subplot_kw_applies_to_every_mosaic_panel(self):
        fig, axd = gplt.subplot_mosaic("AB", subplot_kw={"projection": "3d"})
        assert [p.ndim for p in fig.panels] == [3, 3]

    def test_per_subplot_kw_applies_to_the_named_panel_only(self):
        fig, axd = gplt.subplot_mosaic("AB", per_subplot_kw={"A": {"projection": "3d"}})
        assert axd["A"].panel.ndim == 3
        assert axd["B"].panel.ndim != 3

    def test_a_tuple_key_names_several_panels(self):
        fig, axd = gplt.subplot_mosaic("AB", per_subplot_kw={("A", "B"): {"projection": "3d"}})
        assert [p.ndim for p in fig.panels] == [3, 3]

    def test_the_mosaic_does_not_leave_a_stray_active_panel(self):
        fig, axd = gplt.subplot_mosaic("AB", subplot_kw={"projection": "3d"})
        assert 0 <= fig.active_panel_index < len(fig.panels)


# ======================================================================================
# Group 4 -- the safety issues
# ======================================================================================


class TestVoxels:
    """D10: ``voxels`` was an alias of ``volume3d``, with entirely different arguments."""

    def test_the_signature_matches_matplotlib(self):
        assert _signature_without_self(Axes3D.voxels) == _bare_signature(gplt.voxels)

    def test_matplotlib_takes_a_boolean_array(self, mpl3d):
        filled = np.zeros((3, 3, 3), dtype=bool)
        filled[1, 1, 1] = True
        assert isinstance(mpl3d.voxels(filled), dict)

    def test_the_audits_failing_snippet_now_runs(self, ax3d):
        filled = np.zeros((3, 3, 3), dtype=bool)
        filled[1, 1, 1] = True
        drawn = ax3d.voxels(filled)
        assert list(drawn) == [(1, 1, 1)]

    def test_a_cube_occupies_its_unit_cell(self, ax3d):
        filled = np.zeros((2, 2, 2), dtype=bool)
        filled[1, 0, 0] = True
        (layer,) = set(ax3d.voxels(filled).values())
        assert layer.vertices[:, 0].min() == pytest.approx(1.0)
        assert layer.vertices[:, 0].max() == pytest.approx(2.0)

    def test_the_coordinate_form_is_accepted(self, ax3d):
        filled = np.zeros((2, 2, 2), dtype=bool)
        filled[0, 0, 0] = True
        x, y, z = np.indices((3, 3, 3))
        assert list(ax3d.voxels(x * 2.0, y * 2.0, z * 2.0, filled)) == [(0, 0, 0)]

    def test_facecolors_are_honoured(self, ax3d):
        filled = np.zeros((2, 2, 2), dtype=bool)
        filled[0, 0, 0] = True
        (layer,) = set(ax3d.voxels(filled, facecolors="red").values())
        assert layer.colors[:, 0].max() > 0.0
        assert layer.colors[:, 1].max() == pytest.approx(0.0)

    def test_an_empty_array_draws_nothing(self, ax3d):
        assert ax3d.voxels(np.zeros((2, 2, 2), dtype=bool)) == {}

    def test_a_non_3d_array_is_refused(self, ax3d):
        with pytest.raises(ValueError, match="3D array"):
            ax3d.voxels(np.zeros((2, 2), dtype=bool))

    def test_shade_false_warns(self, ax3d):
        filled = np.zeros((2, 2, 2), dtype=bool)
        filled[0, 0, 0] = True
        with pytest.warns(gplt.MatplotlibCompatWarning, match="shade"):
            ax3d.voxels(filled, shade=False)

    def test_volume3d_is_still_its_own_function(self, ax3d):
        """The extension survives the un-aliasing; only the name collision is gone."""
        assert gplt.voxels is not gplt.volume3d
        assert ax3d.volume3d([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]) is not None


class TestAxesProxyDenyList:
    """The secondary hazard: ``ax.show`` / ``ax.clf`` / ``ax.subplots`` all resolved."""

    DENIED = ["show", "savefig", "close", "clf", "subplots", "suptitle", "figure_does_not_exist"]

    @pytest.mark.parametrize("name", ["show", "savefig", "close", "clf", "subplots", "suptitle"])
    def test_figure_level_names_are_not_axes_methods(self, ax3d, name):
        assert not hasattr(MplAxes, name), f"{name} is an Axes method in matplotlib"
        with pytest.raises(AttributeError):
            getattr(ax3d, name)

    def test_hasattr_answers_honestly(self, ax3d):
        """Duck-typed code probes with ``hasattr``; a True here misclassifies the object."""
        assert not hasattr(ax3d, "show")
        assert not hasattr(ax3d, "clf")

    def test_the_message_says_where_to_call_it_instead(self, ax3d):
        with pytest.raises(AttributeError, match="figure-level"):
            ax3d.savefig

    @pytest.mark.parametrize(
        "name", ["plot", "scatter", "cla", "clear", "legend", "grid", "text", "margins"]
    )
    def test_real_axes_methods_are_untouched(self, ax3d, name):
        assert hasattr(MplAxes, name)
        assert callable(getattr(ax3d, name))

    def test_the_figure_is_still_reachable(self, ax3d):
        assert ax3d.figure is gplt.gcf()
        assert callable(ax3d.figure.savefig)

    def test_nothing_denied_is_a_real_matplotlib_axes_member(self):
        """Guards the list itself: denying a genuine Axes name would break parity."""
        assert [n for n in gplt._AXES_DENIED_NAMES if hasattr(MplAxes, n)] == []


# ======================================================================================
# Regression guard for the 2D surface
# ======================================================================================


class TestNo2DRegressions:
    """The 3D dispatch must be invisible to a 2D figure."""

    def test_a_plain_2d_script_is_unchanged(self):
        gplt.figure("flat", width=800, height=600)
        gplt.plot([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], "b-", label="d")
        gplt.scatter([0.0, 1.0], [1.0, 0.0], c=[0.0, 1.0], cmap="viridis")
        gplt.xlabel("x")
        gplt.ylabel("y")
        gplt.title("t")
        gplt.xlim(0.0, 2.0)
        gplt.grid(True)
        gplt.legend()
        assert gplt.gcf().is_3d_scene() is False

    def test_2d_axes_still_reject_nothing_they_used_to_accept(self):
        fig, axs = gplt.subplots(2, 2)
        axs[0, 0].plot([0.0, 1.0], [0.0, 1.0])
        axs[0, 0].set_xlim(0.0, 1.0)
        axs[0, 0].set_xlabel("x")
        assert axs[0, 0].get_xlim() == (0.0, 1.0)

    def test_mixing_a_2d_and_a_3d_panel_keeps_them_apart(self):
        fig, axs = gplt.subplots(1, 2)
        axs[1].set_projection("3d")
        axs[0].plot([0.0, 1.0], [0.0, 1.0])
        axs[1].plot([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
        assert [layer.layer_type for layer in fig.panels[0].scene.layers] == ["polyline"]
        assert "wireframe3d" in [layer.layer_type for layer in fig.panels[1].scene.layers]
