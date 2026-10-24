"""The object-oriented half of the matplotlib surface: ``fig.<method>()`` and ``ax.<method>()``.

``test_mpl_parity_surface.py`` holds ``matplotlib.pyplot`` -- the functional API -- to its
compat contract. This file is the same contract for what a script does to the objects
``plt.subplots()`` hands back, which is most of what real matplotlib code contains:
``fig.tight_layout()``, ``fig.savefig(...)``, ``ax.set(xlabel=..., ylim=...)``,
``ax.get_xlim()``.

Nothing here opens a window: ``GPULinePlot`` constructs without one and every export goes
through the headless matplotlib path.
"""

from __future__ import annotations

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot import animation
from glplot.utils.preview import _collapse_uniform_colors


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


X = np.linspace(0.0, 6.0, 200)


# ---------------------------------------------------------------------------------------
# fig.savefig: used to take only (filename, scale) and segfault without a GL window
# ---------------------------------------------------------------------------------------


class TestFigureSavefig:
    def test_writes_a_file_with_no_window(self, tmp_path):
        fig, ax = gplt.subplots()
        ax.plot(X, np.sin(X))
        out = tmp_path / "a.png"
        fig.savefig(str(out))
        assert out.stat().st_size > 1000

    def test_takes_matplotlibs_keywords(self, tmp_path):
        fig, ax = gplt.subplots()
        ax.plot(X, np.sin(X))
        out = tmp_path / "b.png"
        fig.savefig(str(out), dpi=150, bbox_inches="tight", transparent=True, facecolor="w")
        assert out.stat().st_size > 1000

    def test_export_works_headless_too(self, tmp_path):
        gplt.plot(X, np.sin(X))
        out = tmp_path / "c.png"
        gplt.export(str(out))
        assert out.stat().st_size > 1000

    def test_saves_that_figure_not_the_current_one(self, tmp_path):
        a, ax_a = gplt.subplots()
        ax_a.plot(X, np.sin(X))
        gplt.figure()  # now current, and empty
        out = tmp_path / "d.png"
        a.savefig(str(out))
        assert gplt.gcf() is not a
        assert out.stat().st_size > 5000  # the curve is in it, not a blank canvas


# ---------------------------------------------------------------------------------------
# Figure methods
# ---------------------------------------------------------------------------------------


class TestFigureMethods:
    def test_methods_act_on_their_own_figure(self):
        a, _ = gplt.subplots()
        b, _ = gplt.subplots()  # b is current
        assert a.gca().figure is a
        a.suptitle("a")
        a.tight_layout()
        # ...and, as in matplotlib, talking to `a` does not change what plt.plot() draws into.
        assert gplt.gcf() is b

    def test_fig_subplots_lays_out_the_figure_it_is_called_on(self):
        fig = gplt.figure()
        other = gplt.figure()
        axs = fig.subplots(2, 1, sharex=True)
        assert len(fig.panels) == 2
        assert len(other.panels) == 1
        assert axs.shape == (2,)

    def test_fig_subplots_squeezes_like_matplotlib(self):
        fig = gplt.figure()
        assert fig.subplots().figure is fig

    def test_axes_lists_every_panel_in_order(self):
        fig, axs = gplt.subplots(1, 3)
        assert [a._index for a in fig.axes] == [0, 1, 2]
        assert fig.get_axes() == fig.axes

    def test_a_proxy_equals_another_proxy_for_the_same_panel(self):
        fig, axs = gplt.subplots(1, 2)
        assert axs[0] in fig.axes
        assert axs[1] == fig.axes[1]
        assert axs[0] != axs[1]
        assert len({axs[0], fig.axes[0]}) == 1

    def test_number_follows_registration(self):
        fig = gplt.figure(7)
        assert fig.number == 7

    def test_size_in_inches_round_trips(self):
        fig = gplt.figure(figsize=(8, 5))
        assert fig.get_size_inches().tolist() == [8.0, 5.0]
        fig.set_size_inches(6, 4)
        assert fig.get_size_inches().tolist() == [6.0, 4.0]
        assert (fig.width, fig.height) == (600, 400)
        fig.set_size_inches((3, 2))
        assert fig.get_size_inches().tolist() == [3.0, 2.0]
        fig.set_figwidth(9)
        assert fig.get_figwidth() == 9.0 and fig.get_figheight() == 2.0

    def test_facecolor_and_patch_agree(self):
        fig = gplt.figure()
        fig.set_facecolor("white")
        assert fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
        fig.patch.set_facecolor("black")
        assert fig.get_facecolor() == (0.0, 0.0, 0.0, 1.0)

    @pytest.mark.parametrize(
        "call",
        [
            lambda f: f.align_labels(),
            lambda f: f.align_xlabels(),
            lambda f: f.autofmt_xdate(),
            lambda f: f.set_tight_layout(True),
            lambda f: f.set_layout_engine("constrained"),
            lambda f: f.patch.set_alpha(0),
        ],
    )
    def test_layout_only_methods_run_and_say_they_do_nothing(self, call):
        fig, _ = gplt.subplots()
        with pytest.warns(gplt.MatplotlibCompatWarning, match="no effect"):
            call(fig)

    @pytest.mark.parametrize(
        "call",
        [
            lambda f: f.tight_layout(),
            lambda f: f.subplots_adjust(hspace=0.3),
            lambda f: f.suptitle("title"),
            lambda f: f.text(0.5, 0.5, "hi"),
            lambda f: f.legend(),
            lambda f: f.gca(),
        ],
    )
    def test_delegated_methods_run(self, call):
        fig, ax = gplt.subplots()
        ax.plot(X, np.sin(X), label="s")
        call(fig)


# ---------------------------------------------------------------------------------------
# plt.sca: used to ignore an Axes entirely
# ---------------------------------------------------------------------------------------


class TestSca:
    def test_sca_redirects_the_next_pyplot_call(self):
        fig, axs = gplt.subplots(1, 2)
        gplt.sca(axs[1])
        gplt.plot([0, 1], [0, 1])
        assert [len(p.scene.layers) for p in fig.panels] == [0, 1]

    def test_sca_makes_the_axes_figure_current(self):
        a, axs_a = gplt.subplots(1, 2)
        gplt.subplots()
        gplt.sca(axs_a[0])
        assert gplt.gcf() is a

    def test_sca_still_accepts_a_figure(self):
        a, _ = gplt.subplots()
        gplt.subplots()
        gplt.sca(a)
        assert gplt.gcf() is a


# ---------------------------------------------------------------------------------------
# Axes methods
# ---------------------------------------------------------------------------------------


class TestAxesGetters:
    def test_labels_and_title_round_trip(self):
        _, ax = gplt.subplots()
        ax.set_xlabel("time")
        ax.set_ylabel("value")
        ax.set_title("T")
        assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == ("time", "value", "T")
        assert ax.get_title("left") == ""

    def test_labels_are_per_panel(self):
        _, axs = gplt.subplots(1, 2)
        axs[0].set_xlabel("a")
        axs[1].set_xlabel("b")
        assert (axs[0].get_xlabel(), axs[1].get_xlabel()) == ("a", "b")

    def test_scale_names(self):
        _, ax = gplt.subplots()
        assert (ax.get_xscale(), ax.get_yscale()) == ("linear", "linear")
        ax.set_yscale("log")
        assert ax.get_yscale() == "log"

    def test_ticks_and_tick_labels_agree(self):
        _, ax = gplt.subplots()
        ax.plot(X, np.sin(X))
        ticks = ax.get_xticks()
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert len(ticks) == len(labels) > 1
        assert all(ax.get_xlim()[0] <= t <= ax.get_xlim()[1] for t in ticks)

    def test_getters_leave_the_current_axes_alone(self):
        fig, axs = gplt.subplots(1, 2)
        gplt.sca(axs[1])
        axs[0].get_xticks()
        axs[0].get_xbound()
        axs[0].get_yticklabels()
        assert fig.active_panel_index == 1

    def test_legend_handles_and_labels_skip_unlabelled_layers(self):
        _, ax = gplt.subplots()
        ax.plot(X, np.sin(X), label="sin")
        ax.plot(X, np.cos(X))
        ax.plot(X, X, label="_hidden")
        handles, labels = ax.get_legend_handles_labels()
        assert labels == ["sin"]
        assert len(handles) == 1

    def test_get_lines_and_has_data(self):
        _, ax = gplt.subplots()
        assert not ax.has_data()
        ax.plot(X, np.sin(X))
        assert ax.has_data() and len(ax.get_lines()) == 1

    def test_position_round_trips(self):
        fig, ax = gplt.subplots()
        ax.set_position([0.1, 0.2, 0.3, 0.4])
        assert np.allclose(ax.get_position().bounds, (0.1, 0.2, 0.3, 0.4))
        assert ax.get_figure() is fig


class TestAxesSetters:
    def test_set_applies_each_property(self):
        _, ax = gplt.subplots()
        ax.plot(X, np.sin(X))
        ax.set(xlabel="a", ylabel="b", title="c", xlim=(0, 3), ylim=(-2, 2))
        assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == ("a", "b", "c")
        assert tuple(np.round(ax.get_xlim(), 6)) == (0.0, 3.0)
        assert tuple(np.round(ax.get_ylim(), 6)) == (-2.0, 2.0)

    def test_set_rejects_an_unknown_property(self):
        _, ax = gplt.subplots()
        with pytest.raises(AttributeError, match="bogus"):
            ax.set(bogus=1)

    def test_update_is_set_with_a_dict(self):
        _, ax = gplt.subplots()
        ax.update({"xlabel": "q"})
        assert ax.get_xlabel() == "q"

    def test_bounds(self):
        _, ax = gplt.subplots()
        ax.plot(X, np.sin(X))
        ax.set_xbound(1, 4)
        assert ax.get_xbound() == (1.0, 4.0)
        ax.set_xbound(upper=5)
        assert ax.get_xbound() == (1.0, 5.0)

    def test_margins_and_autoscale_view_run(self):
        _, ax = gplt.subplots()
        ax.plot(X, np.sin(X))
        ax.set_xmargin(0.1)
        ax.set_ymargin(0.2)
        ax.autoscale_view()
        ax.relim()

    def test_facecolor_sets_the_background(self):
        fig, ax = gplt.subplots()
        ax.set_facecolor("white")
        assert ax.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
        ax.patch.set_facecolor("black")
        assert fig.get_facecolor() == (0.0, 0.0, 0.0, 1.0)

    def test_facecolor_on_a_grid_says_it_is_figure_wide(self):
        _, axs = gplt.subplots(1, 2)
        with pytest.warns(gplt.MatplotlibCompatWarning, match="whole figure"):
            axs[0].set_facecolor("white")

    @pytest.mark.parametrize(
        "name,args",
        [
            ("label_outer", ()),
            ("set_axisbelow", (True,)),
            ("set_rasterized", (True,)),
            ("set_animated", (True,)),
            ("set_zorder", (3,)),
        ],
    )
    def test_cosmetic_setters_run_and_say_they_do_nothing(self, name, args):
        _, ax = gplt.subplots()
        with pytest.warns(gplt.MatplotlibCompatWarning, match="no effect"):
            getattr(ax, name)(*args)

    @pytest.mark.parametrize(
        "name",
        ["get_size_inches", "get_axes", "set_size_inches", "align_labels", "number"],
    )
    def test_figure_only_names_do_not_leak_onto_an_axes(self, name):
        _, ax = gplt.subplots()
        with pytest.raises(AttributeError, match="figure-level"):
            getattr(ax, name)


# ---------------------------------------------------------------------------------------
# Limits: matplotlib autoscales as data arrives; GLPlot used to wait for show()
# ---------------------------------------------------------------------------------------


class TestLimitsBeforeAnyRender:
    def test_get_xlim_after_plot_is_the_data_range(self):
        _, ax = gplt.subplots()
        ax.plot(X, np.sin(X))
        lo, hi = ax.get_xlim()
        assert lo == pytest.approx(-0.3) and hi == pytest.approx(6.3)

    def test_set_xlim_alone_leaves_y_fitted_to_the_data(self):
        # Used to pin y at the (-1, 1) default, so 100*sin(x) exported as an empty frame.
        _, ax = gplt.subplots()
        ax.plot(X, 100 * np.sin(X))
        ax.set_xlim(0, 3)
        lo, hi = ax.get_ylim()
        assert lo < -90 and hi > 90

    def test_set_ylim_alone_leaves_x_fitted_to_the_data(self):
        _, ax = gplt.subplots()
        ax.plot(X, np.sin(X))
        ax.set_ylim(-2, 2)
        lo, hi = ax.get_xlim()
        assert lo == pytest.approx(-0.3) and hi == pytest.approx(6.3)

    def test_reading_a_limit_does_not_decide_the_view(self):
        fig, ax = gplt.subplots()
        ax.plot([0, 1], [0, 1])
        ax.get_xlim()
        assert fig._needs_initial_autoscale  # the fit is still owed
        ax.plot([0, 50], [0, 50])
        assert ax.get_xlim()[1] == pytest.approx(52.5)

    def test_an_empty_axes_keeps_its_default_view(self):
        _, ax = gplt.subplots()
        assert ax.get_xlim() == (-1.0, 1.0)

    def test_a_pinned_view_is_returned_as_pinned(self):
        _, ax = gplt.subplots()
        ax.plot(X, np.sin(X))
        ax.set_xlim(1, 2)
        ax.set_ylim(-5, 5)
        assert tuple(np.round(ax.get_xlim(), 6)) == (1.0, 2.0)
        assert tuple(np.round(ax.get_ylim(), 6)) == (-5.0, 5.0)


# ---------------------------------------------------------------------------------------
# Things the new Figure methods must not disturb
# ---------------------------------------------------------------------------------------


class TestAnimationStillSeesAnEngine:
    def test_a_glplot_figure_is_not_mistaken_for_a_matplotlib_one(self):
        fig, _ = gplt.subplots()
        assert hasattr(fig, "get_size_inches")  # the new method is what made this a risk
        assert not animation._is_mpl_figure(fig)

    def test_a_real_matplotlib_figure_still_is(self):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        assert animation._is_mpl_figure(Figure())

    def test_frame_size_prediction_still_uses_the_engines_arithmetic(self):
        fig = gplt.figure(width=1280, height=800)
        assert animation.figure_pixel_size(fig) == (1536, 960)


# ---------------------------------------------------------------------------------------
# Export speed: a uniform colour array used to put Agg on its per-point path
# ---------------------------------------------------------------------------------------


class TestCollapseUniformColors:
    def test_identical_rows_collapse_to_one(self):
        colors = np.tile([0.1, 0.4, 0.7, 1.0], (1000, 1))
        out = _collapse_uniform_colors(colors)
        assert out.shape == (1, 4)
        assert out[0].tolist() == [0.1, 0.4, 0.7, 1.0]

    def test_distinct_rows_are_left_alone(self):
        colors = np.random.default_rng(0).random((50, 4))
        assert _collapse_uniform_colors(colors) is colors

    def test_one_odd_row_is_enough_to_keep_them_all(self):
        colors = np.tile([0.1, 0.4, 0.7, 1.0], (100, 1))
        colors[57, 0] = 0.9
        assert _collapse_uniform_colors(colors).shape == (100, 4)

    @pytest.mark.parametrize("value", [None, np.zeros((1, 4)), np.zeros(4)])
    def test_degenerate_inputs_pass_through(self, value):
        assert _collapse_uniform_colors(value) is value

    def test_a_uniform_scatter_still_exports(self, tmp_path):
        rng = np.random.default_rng(0)
        gplt.scatter(rng.normal(size=5000), rng.normal(size=5000), s=3)
        out = tmp_path / "s.png"
        gplt.savefig(str(out))
        assert out.stat().st_size > 1000
