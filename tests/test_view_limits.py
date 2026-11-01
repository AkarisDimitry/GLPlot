"""Axis limits behave as matplotlib's do: per axis, and in data units on a scaled axis.

Two long-standing differences, both of which a script written from matplotlib habit hits at once:

* ``ylim(...)`` on an axes with no data yet used to pin *both* axes, so x stayed at its empty
  default (-1, 1) when the data arrived. matplotlib turns autoscale off for the axis you set only.
* On a log (or symlog/asinh/logit) axis the engine's camera speaks ``log10``, and ``xlim``,
  ``axis``, ``axhline`` and friends passed that through as if it were data: ``xlim(10, 1000)``
  framed 10**10..10**1000, and a reference line was drawn at coordinates the log axis then dropped.
"""

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.utils.preview import _build_preview_figure

F = np.logspace(0, 4, 100)
GAIN = 1.0 / np.sqrt(1.0 + (F / 300.0) ** 2)


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


def exported(panel=0):
    fig = _build_preview_figure(gplt.get_engine())
    ax = fig.axes[panel]
    fig.canvas.draw()
    return tuple(map(float, ax.get_xlim())), tuple(map(float, ax.get_ylim()))


def covers(limits, values, tolerance=0.0):
    return limits[0] <= np.min(values) + tolerance and limits[1] >= np.max(values) - tolerance


class TestOneAxisAtATime:
    def test_ylim_before_data_leaves_x_following_the_data(self):
        gplt.ylim(-5, 5)
        gplt.plot([0, 10], [0, 1])
        xlim, ylim = exported()
        assert ylim == (-5.0, 5.0)
        assert covers(xlim, [0, 10]) and xlim != (-1.0, 1.0)

    def test_xlim_before_data_leaves_y_following_the_data(self):
        gplt.xlim(0, 5)
        gplt.plot([0, 10], [0, 100])
        xlim, ylim = exported()
        assert xlim == (0.0, 5.0)
        assert covers(ylim, [0, 100])

    def test_xlim_after_plotting_keeps_the_other_axis_fitted(self):
        gplt.plot([0, 10], [0, 100])
        gplt.xlim(0, 5)
        xlim, ylim = exported()
        assert xlim == (0.0, 5.0) and covers(ylim, [0, 100])

    def test_the_free_axis_follows_data_added_after_the_limit(self):
        gplt.plot([0, 10], [0, 100])
        gplt.xlim(0, 5)
        gplt.plot([0, 10], [0, 300])
        xlim, ylim = exported()
        assert xlim == (0.0, 5.0) and covers(ylim, [0, 300])

    def test_both_limits_pin_both_axes(self):
        gplt.xlim(0, 5)
        gplt.ylim(0, 1)
        gplt.plot([0, 10], [0, 100])
        assert exported() == ((0.0, 5.0), (0.0, 1.0))

    def test_an_explicit_autoscale_takes_both_axes_back(self):
        gplt.ylim(-5, 5)
        gplt.plot([0, 10], [0, 1])
        gplt.autoscale()
        xlim, ylim = exported()
        assert covers(xlim, [0, 10]) and covers(ylim, [0, 1]) and ylim[1] < 2

    def test_an_inverted_pinned_axis_stays_inverted(self):
        gplt.ylim(5, -5)
        gplt.plot([0, 10], [0, 1])
        _, ylim = exported()
        assert ylim == (5.0, -5.0)

    def test_panels_are_pinned_independently(self):
        _, axs = gplt.subplots(1, 2)
        axs[0].plot([0, 10], [0, 1])
        axs[1].plot([0, 10], [0, 1])
        axs[0].set_ylim(-5, 5)
        left_x, left_y = exported(0)
        right_x, right_y = exported(1)
        assert left_y == (-5.0, 5.0) and covers(left_x, [0, 10])
        assert covers(right_x, [0, 10]) and covers(right_y, [0, 1]) and right_y[1] < 2

    def test_the_engine_fit_honours_the_pin_too(self):
        """The live window runs the same initial fit; it must not overwrite a pinned axis."""
        gplt.ylim(-5, 5)
        gplt.plot([0, 10], [0, 1])
        engine = gplt.get_engine()
        engine._autoscale_all_panels()
        left, right = engine.get_xlim()
        bottom, top = engine.get_ylim()
        assert (bottom, top) == pytest.approx((-5.0, 5.0))
        assert left < 0 < 10 < right and (left, right) != (-1.0, 1.0)

    def test_reading_a_limit_does_not_change_what_gets_fitted(self):
        gplt.ylim(-5, 5)
        gplt.plot([0, 10], [0, 1])
        gplt.xlim()  # a read
        gplt.plot([0, 20], [0, 1])
        assert covers(exported()[0], [0, 20])


class TestScaledAxes:
    def log_figure(self):
        gplt.plot(F, GAIN)
        gplt.xscale("log")
        gplt.yscale("log")

    def test_limits_are_set_and_read_in_data_units(self):
        self.log_figure()
        assert gplt.xlim(10, 1000) == (10, 1000)
        gplt.ylim(0.1, 1)
        assert gplt.xlim() == pytest.approx((10.0, 1000.0))
        assert gplt.ylim() == pytest.approx((0.1, 1.0))

    def test_the_export_frames_exactly_what_was_asked(self):
        self.log_figure()
        gplt.xlim(10, 1000)
        gplt.ylim(0.1, 1)
        xlim, ylim = exported()
        assert xlim == pytest.approx((10.0, 1000.0)) and ylim == pytest.approx((0.1, 1.0))

    def test_an_end_a_log_axis_cannot_show_is_ignored_with_a_warning(self):
        self.log_figure()
        gplt.xlim(10, 1000)
        with pytest.warns(UserWarning, match="cannot be shown on a log axis"):
            gplt.xlim(-5, 100)
        assert gplt.xlim() == pytest.approx((10.0, 100.0))

    def test_an_end_given_alone_moves_only_that_end(self):
        self.log_figure()
        gplt.xlim(10, 1000)
        gplt.xlim(left=100)
        assert gplt.xlim() == pytest.approx((100.0, 1000.0))

    def test_axis_reads_and_writes_data_units(self):
        self.log_figure()
        gplt.axis((1, 1e4, 0.01, 2))
        assert gplt.axis() == pytest.approx((1.0, 1e4, 0.01, 2.0))
        xlim, ylim = exported()
        assert xlim == pytest.approx((1.0, 1e4)) and ylim == pytest.approx((0.01, 2.0))
        gplt.axis(xmin=10)
        assert gplt.xlim()[0] == pytest.approx(10.0)

    def test_symlog_limits_round_trip(self):
        gplt.plot(np.linspace(-50, 50, 100), np.linspace(-50, 50, 100) ** 3)
        gplt.xscale("symlog")
        gplt.xlim(-20, 30)
        assert gplt.xlim() == pytest.approx((-20.0, 30.0))
        assert exported()[0] == pytest.approx((-20.0, 30.0))

    def test_reference_lines_are_drawn_in_data_coordinates(self):
        self.log_figure()
        gplt.xlim(10, 1000)
        gplt.ylim(0.1, 1)
        horizontal = gplt.axhline(0.5)
        vertical = gplt.axvline(100)
        assert np.asarray(horizontal.pts)[:, 0] == pytest.approx([10.0, 1000.0])
        assert np.asarray(horizontal.pts)[:, 1] == pytest.approx([0.5, 0.5])
        assert np.asarray(vertical.pts)[:, 1] == pytest.approx([0.1, 1.0])

    def test_a_fraction_of_a_log_axis_is_a_fraction_of_the_drawn_axis(self):
        """Half way along a decade-spanning log axis is the geometric, not arithmetic, mean."""
        self.log_figure()
        gplt.xlim(1, 10000)
        line = gplt.axhline(0.5, xmin=0.5, xmax=0.75)
        xs = np.asarray(line.pts)[:, 0]
        assert xs == pytest.approx([100.0, 1000.0], rel=1e-3)

    def test_spans_text_and_axline_use_data_units(self):
        self.log_figure()
        gplt.xlim(10, 1000)
        gplt.ylim(0.1, 1)
        span = gplt.axvspan(20, 200, ymin=0.0, ymax=1.0)
        verts = np.asarray(span.vertices)
        assert verts[:, 1].min() == pytest.approx(0.1) and verts[:, 1].max() == pytest.approx(1.0)
        line = gplt.axline((100, 0.5), slope=0.0)
        assert np.asarray(line.pts)[:, 0] == pytest.approx([10.0, 1000.0])

    def test_text_in_axes_fractions_lands_inside_a_log_axis(self):
        self.log_figure()
        gplt.xlim(1, 10000)
        gplt.ylim(0.01, 1)
        gplt.text(0.5, 0.5, "mid", transform=gplt.gca().transAxes)
        layer = gplt.get_engine().scene.layers[-1]
        assert layer.x == pytest.approx(100.0, rel=1e-3)
        assert layer.y == pytest.approx(0.1, rel=1e-3)

    def test_linear_axes_are_untouched(self):
        gplt.plot([0, 10], [0, 4])
        gplt.xlim(2, 8)
        assert gplt.xlim() == (2, 8)
        assert gplt.axis() == pytest.approx((2.0, 8.0, *gplt.ylim()))
        line = gplt.axhline(1.0, xmin=0.25, xmax=0.75)
        assert np.asarray(line.pts)[:, 0] == pytest.approx([3.5, 6.5])

    def test_the_figure_with_everything_set_exports_with_the_data_visible(self, tmp_path):
        self.log_figure()
        gplt.xlim(10, 1000)
        gplt.ylim(0.1, 1)
        gplt.axhline(0.5, color="red")
        gplt.axvline(300, color="green")
        gplt.savefig(str(tmp_path / "bode.png"))
        assert (tmp_path / "bode.png").stat().st_size > 5000


class TestThroughMcp:
    """The same behaviour reached the way a model reaches it: ``set_axes`` on a toolbox figure."""

    @pytest.fixture
    def box(self):
        from glplot.mcp.toolbox import Toolbox

        toolbox = Toolbox()
        toolbox.create_dataset("bode", {"f": "logspace(0, 4, 100)", "g": "1/sqrt(1+(f/300)**2)"})
        toolbox.create_figure()
        toolbox.add_plot("fig1", "line", dataset="bode", x="f", y="g")
        yield toolbox
        toolbox.close_all()

    def resolved(self, box):
        panel = box.render_figure("fig1").data["panels"][0]
        return tuple(panel["xlim"]), tuple(panel["ylim"])

    def test_a_scale_and_limits_in_one_call_both_take_effect(self, box):
        done = box.set_axes(
            "fig1", xscale="log", yscale="log", xlim=[10, 1000], ylim=[0.1, 1], title="Bode"
        )
        assert done.data["applied"].index("xscale") < done.data["applied"].index("xlim")
        xlim, ylim = self.resolved(box)
        assert xlim == pytest.approx((10.0, 1000.0)) and ylim == pytest.approx((0.1, 1.0))

    def test_limits_alone_later_still_work_on_a_log_axis(self, box):
        box.set_axes("fig1", xscale="log")
        box.set_axes("fig1", xlim=[100, 5000])
        assert self.resolved(box)[0] == pytest.approx((100.0, 5000.0))

    def test_one_limit_leaves_the_other_axis_fitted(self, box):
        box.set_axes("fig1", ylim=[0, 2])
        xlim, ylim = self.resolved(box)
        assert ylim == pytest.approx((0.0, 2.0)) and covers(xlim, [1, 10000])

    def test_logspace_and_geomspace_build_the_x_values_a_log_plot_needs(self, box):
        box.create_dataset("g", {"x": "geomspace(1, 1000, 4)"})
        head = box.describe_dataset("g", 4).data["head"]
        assert [row[0] for row in head] == pytest.approx([1.0, 10.0, 100.0, 1000.0])
