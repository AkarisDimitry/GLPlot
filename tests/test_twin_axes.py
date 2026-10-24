"""``twinx`` / ``twiny``: a second y (or x) scale for one panel.

Three layers, tested separately because they fail separately:

* the *model* (``glplot.core.twin``): which layers belong to the twin, and the affine relation
  that maps their values into the host's viewport;
* the *live window* path: the axis manager's twin ticks, and what survives a host re-fit;
* the *export*: a real matplotlib twin axes drawn from the values the user plotted.

Nothing here opens a window; the window is exercised with a disposable script (see the
project notes), and its tick placement is checked through ``AxisManager`` directly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.core import twin as twin_core
from glplot.utils.preview import _build_preview_figure

T = np.linspace(0.0, 10.0, 200)
PRESSURE = 1000.0 * np.cos(T) + 5000.0  # true range 4000..6000, nothing like sin's -1..1


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


def make(**twin_kwargs):
    """``(fig, host, twin, twin_line)``: a sine on the host, ``PRESSURE`` on the twin."""
    fig, host = gplt.subplots()
    host.plot(T, np.sin(T), label="temperature")
    twin = host.twinx()
    (line,) = twin.plot(T, PRESSURE, label="pressure")
    return fig, host, twin, line


def host_window(fig, axis="y"):
    l, r, b, t = fig.camera_controller.world_window(fig.width, fig.height)
    return (b, t) if axis == "y" else (l, r)


def export(fig):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _build_preview_figure(fig, 1.0)


# ---------------------------------------------------------------------------------------
# The axes objects
# ---------------------------------------------------------------------------------------


class TestAxesObjects:
    def test_a_twin_is_its_own_axes_on_the_same_panel(self):
        fig, host, twin, _ = make()
        assert twin != host and hash(twin) != hash(host)
        assert twin.panel is host.panel and twin.figure is fig
        assert twin == host.twinx()  # asking twice gives the same twin

    def test_module_level_twinx_uses_the_current_axes(self):
        gplt.plot(T, T)
        twin = gplt.twinx()
        assert twin.panel is gplt.gca().panel

    def test_twiny_is_the_x_twin(self):
        _, host, _, _ = make()
        assert host.twiny()._twin_axis == "x"

    def test_the_twin_does_not_become_the_current_axes(self):
        fig, host, twin, _ = make()
        gplt.plot(T, T * 2)  # pyplot-style call after twinx(): still the host
        assert len(host.get_lines()) == 2 and len(twin.get_lines()) == 1

    def test_each_axes_lists_only_its_own_artists(self):
        _, host, twin, line = make()
        assert len(host.get_lines()) == 1 and host.get_lines()[0] is not line
        assert twin.get_lines() == [line]
        assert host.has_data() and twin.has_data()
        assert [*host.get_legend_handles_labels()[1]] == ["temperature"]
        assert [*twin.get_legend_handles_labels()[1]] == ["pressure"]

    def test_a_twin_with_no_data_of_its_own(self):
        _, host, _, _ = make()
        empty = host.twiny()
        assert not empty.has_data() and empty.get_lines() == []

    def test_labels_belong_to_the_twinned_axis_only(self):
        _, host, twin, _ = make()
        twin.set_ylabel("kPa")
        twin.set_xlabel("shared x")
        assert (twin.get_ylabel(), host.get_ylabel()) == ("kPa", "")
        assert (twin.get_xlabel(), host.get_xlabel()) == ("shared x", "shared x")

    def test_a_twin_label_takes_colour_and_size(self):
        _, _, twin, _ = make()
        twin.set_ylabel("kPa", color="tab:red", fontsize=14)
        state = twin._twin
        assert (state.label, state.label_color, state.label_fontsize) == ("kPa", "tab:red", 14.0)

    def test_the_label_object(self):
        _, _, twin, _ = make()
        twin.yaxis.label.set_text("kPa")
        twin.yaxis.label.set_color("r")
        assert twin.get_ylabel() == "kPa" and twin._twin.label_color == "r"

    def test_tick_params_colours_only_the_twin(self):
        _, host, twin, _ = make()
        twin.tick_params(axis="y", colors="tab:red")
        assert twin._twin.tick_color == "tab:red"
        assert host.figure.options.axis_tick_color is None

    def test_tick_params_on_the_shared_axis_reach_the_host(self):
        fig, _, twin, _ = make()
        twin.tick_params(axis="x", labelsize=9)
        assert fig.options.axis_tick_fontsize == 9.0

    def test_other_axis_methods_are_the_hosts(self):
        _, host, twin, _ = make()
        host.set_xlim(2, 4)
        assert twin.get_xlim() == (2.0, 4.0)
        twin.set_xlim(3, 5)
        assert host.get_xlim() == (3.0, 5.0)


# ---------------------------------------------------------------------------------------
# Limits, scale, ticks of the twin
# ---------------------------------------------------------------------------------------


class TestTwinLimits:
    def test_the_host_fits_its_own_data_only(self):
        fig, host, _, _ = make()
        lo, hi = host.get_ylim()
        assert -1.5 < lo < -1.0 and 1.0 < hi < 1.5  # a sine, not 4000..6000

    def test_the_twin_fits_its_own_data(self):
        _, _, twin, _ = make()
        lo, hi = twin.get_ylim()
        assert 3800 < lo < 4000 and 6000 < hi < 6200

    def test_the_x_range_is_shared(self):
        _, host, twin, _ = make()
        assert host.get_xlim() == twin.get_xlim()

    def test_set_ylim_on_the_twin(self):
        fig, host, twin, _ = make()
        twin.set_ylim(0, 10000)
        assert twin.get_ylim() == (0.0, 10000.0)
        fig._autoscale_all_panels()
        shown = twin_core.derived_range(twin._twin, host_window(fig))
        assert shown == pytest.approx((0.0, 10000.0), rel=1e-6, abs=1e-6)

    def test_reversed_twin_limits_invert_the_twin(self):
        _, _, twin, _ = make()
        twin.set_ylim(6000, 4000)
        assert twin.get_ylim() == (6000.0, 4000.0) and twin.yaxis_inverted()

    def test_partial_twin_limits(self):
        _, _, twin, _ = make()
        before = twin.get_ylim()
        twin.set_ylim(bottom=0)
        assert twin.get_ylim() == (0.0, before[1])

    def test_set_ybound_and_get_ybound(self):
        _, _, twin, _ = make()
        twin.set_ybound(1000, 7000)
        assert twin.get_ybound() == (1000.0, 7000.0)

    def test_invert_yaxis_on_the_twin(self):
        _, host, twin, _ = make()
        twin.invert_yaxis()
        assert twin.yaxis_inverted() and not host.yaxis_inverted()
        twin.yaxis.set_inverted(False)
        assert not twin.yaxis_inverted()

    def test_log_scale_on_the_twin(self):
        fig, host, twin, line = make()
        twin.set_yscale("log")
        assert twin.get_yscale() == "log" and host.get_yscale() == "linear"
        fig._autoscale_all_panels()
        lo, hi = host_window(fig)
        assert lo - 1e-3 <= float(line.pts[:, 1].min()) and float(line.pts[:, 1].max()) <= hi + 1e-3

    def test_an_unknown_scale_raises(self):
        _, _, twin, _ = make()
        with pytest.raises(ValueError, match="unsupported scale"):
            twin.set_yscale("sideways")

    def test_set_yticks_pins_the_twin_ticks(self):
        _, host, twin, _ = make()
        twin.set_yticks([4000, 5000, 6000])
        assert list(twin.get_yticks()) == [4000.0, 5000.0, 6000.0]
        assert host.figure.options.axis_tick_values_y is None  # not the host's

    def test_set_yticklabels_pairs_with_the_pinned_ticks(self):
        _, _, twin, _ = make()
        twin.set_yticks([4000, 6000])
        twin.set_yticklabels(["low", "high"])
        assert [t.get_text() for t in twin.get_yticklabels()] == ["low", "high"]

    def test_set_yticklabels_with_the_wrong_count_raises(self):
        _, _, twin, _ = make()
        twin.set_yticks([4000, 6000])
        with pytest.raises(ValueError, match="2 ticks"):
            twin.set_yticklabels(["only one"])

    def test_the_automatic_ticks_are_round_numbers_in_the_twins_units(self):
        _, _, twin, _ = make()
        ticks = twin.get_yticks()
        assert len(ticks) >= 3 and ticks.min() >= 3800 and ticks.max() <= 6200
        assert np.allclose(ticks % 250, 0)

    def test_the_formatter_object_is_the_twins(self):
        from matplotlib.ticker import FuncFormatter

        _, host, twin, _ = make()
        formatter = FuncFormatter(lambda v, p: f"{v:.0f} kPa")
        twin.yaxis.set_major_formatter(formatter)
        assert twin.yaxis.get_major_formatter() is formatter
        assert host.yaxis.get_major_formatter() is None

    def test_a_fixed_locator_pins_the_twin_ticks(self):
        from matplotlib.ticker import FixedLocator

        _, _, twin, _ = make()
        twin.yaxis.set_major_locator(FixedLocator([4500, 5500]))
        assert list(twin.get_yticks()) == [4500.0, 5500.0]

    def test_an_unmapped_locator_is_kept_and_warned(self):
        from matplotlib.ticker import LogLocator

        _, _, twin, _ = make()
        with pytest.warns(gplt.MatplotlibCompatWarning, match="LogLocator"):
            twin.yaxis.set_major_locator(LogLocator())
        assert isinstance(twin.yaxis.get_major_locator(), LogLocator)

    def test_the_twinx_x_methods_still_reach_the_host_and_twiny_y_methods_too(self):
        _, host, _, _ = make()
        top = host.twiny()
        top.set_ylim(-3, 3)  # y is shared on a twiny: the host's
        assert host.get_ylim() == (-3.0, 3.0)


# ---------------------------------------------------------------------------------------
# The relation: baked copies, true values, and what a host re-fit does
# ---------------------------------------------------------------------------------------


class TestRelation:
    def test_true_values_are_kept_and_the_baked_copy_fits_the_host(self):
        fig, host, _, line = make()
        fig._autoscale_all_panels()
        assert float(line.get_ydata().min()) == pytest.approx(PRESSURE.min(), abs=0.01)
        assert float(line.get_ydata().max()) == pytest.approx(PRESSURE.max(), abs=0.01)
        lo, hi = host_window(fig)
        assert lo <= float(line.pts[:, 1].min()) and float(line.pts[:, 1].max()) <= hi
        assert line.metadata["twin_axis"] == "y"

    def test_nothing_is_baked_before_the_host_has_been_fitted(self):
        fig, _, _, line = make()
        assert fig.panels[0].twins["y"].affine is None
        assert float(line.pts[:, 1].min()) == pytest.approx(PRESSURE.min(), abs=0.01)

    def test_a_limit_read_before_the_first_render_does_not_commit_the_fit(self):
        fig, host, twin, line = make()
        host.get_ylim()
        twin.get_ylim()
        host.get_yticks()
        assert fig.panels[0].twins["y"].affine is None
        assert float(line.pts[:, 1].min()) == pytest.approx(PRESSURE.min(), abs=0.01)

    def test_a_programmatic_host_limit_leaves_the_twin_where_it_was(self):
        # ax1.set_ylim(...) does not move ax2 in matplotlib.
        fig, host, twin, _ = make()
        fig._autoscale_all_panels()
        before = twin_core.derived_range(twin._twin, host_window(fig))
        host.set_ylim(-5, 5)
        after = twin_core.derived_range(twin._twin, host_window(fig))
        assert after == pytest.approx(before)

    def test_a_navigation_gesture_moves_the_twin_with_the_host(self):
        fig, host, twin, _ = make()
        fig._autoscale_all_panels()
        affine = twin._twin.affine
        fig.set_view(ylim=(-0.5, 0.5), twin_follow=True)
        assert twin._twin.affine == affine
        lo, hi = twin_core.derived_range(twin._twin, host_window(fig))
        assert (hi - lo) < 1000  # zoomed in on the middle of 4000..6000

    def test_a_twin_limit_survives_a_later_host_limit(self):
        fig, host, twin, _ = make()
        twin.set_ylim(0, 10000)
        fig._autoscale_all_panels()
        host.set_ylim(-3, 3)
        assert twin_core.derived_range(twin._twin, host_window(fig)) == pytest.approx(
            (0.0, 10000.0)
        )

    def test_explicit_autoscale_keeps_the_twin_range(self):
        fig, host, twin, _ = make()
        fig._autoscale_all_panels()
        before = twin_core.derived_range(twin._twin, host_window(fig))
        host.autoscale()
        assert twin_core.derived_range(twin._twin, host_window(fig)) == pytest.approx(before)

    def test_an_inverted_host_does_not_invert_the_twin(self):
        # ax1.invert_yaxis() leaves ax2 upright, so in *host* coordinates it runs the other way.
        fig, host, twin, _ = make()
        fig._autoscale_all_panels()
        upright = twin._twin.affine[0]
        host.invert_yaxis()
        assert twin._twin.affine[0] == pytest.approx(-upright)
        assert not twin.yaxis_inverted()
        host.invert_yaxis()
        assert twin._twin.affine[0] == pytest.approx(upright)

    def test_an_inverted_twin_on_an_upright_host(self):
        fig, _, twin, _ = make()
        twin.invert_yaxis()
        fig._autoscale_all_panels()
        assert twin._twin.affine[0] < 0

    def test_a_log_host_gets_the_inverse_scale_applied_to_baked_values(self):
        fig, host, twin, line = make()
        host.set_yscale("log")
        host.cla()
        host.plot(T, np.exp(T / 3.0))
        twin = host.twinx()
        (line,) = twin.plot(T, PRESSURE)
        fig._autoscale_all_panels()
        lo, hi = host_window(fig)  # log10 space
        uploaded = np.log10(line.pts[:, 1])  # what the renderer does to pts
        assert lo - 1e-3 <= uploaded.min() and uploaded.max() <= hi + 1e-3

    def test_set_ydata_on_a_twin_line_replaces_the_true_values_and_remaps(self):
        fig, _, _, line = make()
        fig._autoscale_all_panels()
        line.set_ydata(PRESSURE - 50.0)  # inside the range the twin is showing
        assert float(line.get_ydata().max()) == pytest.approx(PRESSURE.max() - 50.0, abs=0.01)
        lo, hi = host_window(fig)
        assert lo <= float(line.pts[:, 1].min()) and float(line.pts[:, 1].max()) <= hi

    def test_set_data_before_the_fit_is_just_stored(self):
        _, _, _, line = make()
        line.set_data(T[:50], PRESSURE[:50])
        assert len(line.get_xdata()) == 50

    def test_bars_scatter_and_text_can_live_on_a_twin(self):
        fig, host, _, _ = make()
        twin = host.twinx()
        bars = twin.bar([1, 2], [10, 20])
        points = twin.scatter([1, 2], [100, 200])
        note = twin.text(3, 150, "hi")
        assert all(b.metadata["twin_axis"] == "y" for b in bars)
        assert points.metadata["twin_axis"] == "y" and note is not None
        fig._autoscale_all_panels()
        assert len(twin.patches) == 2 and len(host.patches) == 0

    def test_layers_that_cannot_be_mapped_stay_on_the_host_and_say_so(self):
        _, host, _, _ = make()
        twin = host.twinx()
        with pytest.warns(gplt.MatplotlibCompatWarning, match="first axes"):
            twin.imshow(np.zeros((4, 4)))
        assert len(host.get_images()) == 1 and twin.get_images() == []

    def test_a_twin_added_after_the_host_was_fitted_is_fitted_immediately(self):
        fig, host, _, _ = make()
        fig._autoscale_all_panels()
        late = host.twiny()  # make() already created the 'y' twin
        (line,) = late.plot(T * 1000.0, np.sin(T))
        assert fig.panels[0].twins["x"].affine is not None
        lo, hi = host_window(fig, "x")
        assert lo <= float(line.pts[:, 0].min()) and float(line.pts[:, 0].max()) <= hi

    def test_twiny_maps_x_and_shares_y(self):
        fig, host, _, _ = make()
        top = host.twiny()
        (line,) = top.plot(T * 1000.0, np.sin(T))
        fig._autoscale_all_panels()
        assert line.metadata["twin_axis"] == "x"
        lo, hi = host_window(fig, "x")
        assert lo <= float(line.pts[:, 0].min()) and float(line.pts[:, 0].max()) <= hi
        assert float(line.get_xdata().max()) == pytest.approx(10000.0, abs=0.5)
        assert top.get_ylim() == host.get_ylim()

    def test_an_only_twin_panel_still_gets_a_sane_host_range(self):
        fig, host = gplt.subplots()
        twin = host.twinx()
        twin.plot(T, PRESSURE)
        fig._autoscale_all_panels()
        lo, hi = host.get_ylim()
        assert np.isfinite([lo, hi]).all() and hi > lo

    def test_cla_forgets_the_twins(self):
        fig, host, _, _ = make()
        host.cla()
        assert fig.panels[0].twins == {}

    def test_twins_are_per_panel(self):
        fig, axs = gplt.subplots(1, 2)
        for a in axs:
            a.plot(T, np.sin(T))
        axs[1].twinx().plot(T, PRESSURE)
        assert fig.panels[0].twins == {} and "y" in fig.panels[1].twins
        fig._autoscale_all_panels()
        assert axs[0].get_ylim()[1] < 2 and axs[1].get_ylim()[1] < 2


# ---------------------------------------------------------------------------------------
# Live ticks (the axis manager), without a window
# ---------------------------------------------------------------------------------------


class TestLiveTicks:
    def refresh(self, fig):
        gplt._refresh_axis_ticks(fig)
        return fig.axis_manager.twin_ticks

    def test_twin_ticks_are_in_the_twins_units_at_host_positions(self):
        fig, _, twin, _ = make()
        fig._autoscale_all_panels()
        ticks = self.refresh(fig)["y"]
        lo, hi = host_window(fig)
        assert len(ticks.major) >= 3 and ticks.major.min() >= lo and ticks.major.max() <= hi
        values = [float(label) for label in ticks.labels]
        assert min(values) >= 3800 and max(values) <= 6200
        # the label at each position is the twin value that position stands for
        a, b = twin._twin.affine
        for position, value in zip(ticks.major, values):
            assert (position - b) / a == pytest.approx(value, rel=1e-3)

    def test_no_twin_ticks_without_a_twin(self):
        fig, host = gplt.subplots()
        host.plot(T, T)
        assert self.refresh(fig) == {}

    def test_no_twin_ticks_before_the_fit(self):
        fig, _, _, _ = make()
        assert self.refresh(fig) == {}

    def test_zooming_the_host_reticks_the_twin(self):
        fig, _, _, _ = make()
        fig._autoscale_all_panels()
        wide = self.refresh(fig)["y"]
        fig.set_view(ylim=(-0.1, 0.1), twin_follow=True)
        zoomed = [float(v) for v in self.refresh(fig)["y"].labels]
        assert max(zoomed) - min(zoomed) < max(float(v) for v in wide.labels) - 3800

    def test_pinned_twin_ticks_and_labels(self):
        fig, _, twin, _ = make()
        twin.set_yticks([4500, 5500], ["low", "high"])
        fig._autoscale_all_panels()
        assert self.refresh(fig)["y"].labels == ["low", "high"]

    def test_pinned_ticks_outside_the_view_are_dropped(self):
        fig, _, twin, _ = make()
        twin.set_yticks([0, 5000, 99999])
        fig._autoscale_all_panels()
        assert [float(v) for v in self.refresh(fig)["y"].labels] == [5000.0]

    def test_a_formatter_labels_the_live_twin_ticks(self):
        from matplotlib.ticker import FuncFormatter

        fig, _, twin, _ = make()
        twin.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:.1f}k"))
        fig._autoscale_all_panels()
        assert all(label.endswith("k") for label in self.refresh(fig)["y"].labels)

    def test_twiny_ticks_run_along_x(self):
        fig, host, _, _ = make()
        top = host.twiny()
        top.plot(T * 1000.0, np.sin(T))
        fig._autoscale_all_panels()
        ticks = self.refresh(fig)["x"]
        values = [float(v) for v in ticks.labels]
        assert max(values) > 5000

    def test_a_log_twin_gets_decade_ticks(self):
        fig, host, _, _ = make()
        twin = host.twiny()
        twin.plot(10.0 ** np.linspace(0, 4, 50), np.linspace(0, 1, 50))
        twin.set_xscale("log")
        fig._autoscale_all_panels()
        labels = self.refresh(fig)["x"].labels
        assert "10" in labels and "1000" in labels


# ---------------------------------------------------------------------------------------
# Export: a real matplotlib twin axes
# ---------------------------------------------------------------------------------------


class TestExport:
    def test_twinx_exports_as_a_real_twin_with_the_true_values(self):
        fig, _, _, _ = make()
        host_ax, twin_ax = export(fig).axes
        assert host_ax.get_shared_x_axes().joined(host_ax, twin_ax)
        assert host_ax.get_ylim()[1] < 2  # the sine only
        lo, hi = twin_ax.get_ylim()
        assert 3800 < lo < 4000 and 6000 < hi < 6200
        ydata = twin_ax.lines[0].get_ydata()
        assert ydata.min() == pytest.approx(PRESSURE.min(), abs=0.01)
        assert len(host_ax.lines) == 1

    def test_the_export_does_not_depend_on_the_bake(self):
        fig, _, _, _ = make()
        fig._autoscale_all_panels()  # the window has opened: pts are now mapped
        _, twin_ax = export(fig).axes
        assert twin_ax.lines[0].get_ydata().max() == pytest.approx(PRESSURE.max(), abs=0.01)

    def test_label_colour_and_tick_colour_reach_the_twin_axes(self):
        fig, _, twin, _ = make()
        twin.set_ylabel("kPa", color="tab:red", fontsize=18)
        twin.tick_params(axis="y", colors="tab:red")
        _, twin_ax = export(fig).axes
        assert twin_ax.get_ylabel() == "kPa"
        assert twin_ax.yaxis.label.get_fontsize() == 18
        assert twin_ax.yaxis.label.get_color() == "tab:red"
        assert twin_ax.yaxis.get_ticklabels()[0].get_color() == "tab:red"

    def test_limits_scale_ticks_and_direction_reach_the_twin_axes(self):
        fig, _, twin, _ = make()
        twin.set_ylim(8000, 1000)
        twin.set_yticks([2000, 4000])
        _, twin_ax = export(fig).axes
        assert twin_ax.get_ylim() == (8000.0, 1000.0)
        assert list(twin_ax.get_yticks()) == [2000.0, 4000.0]

    def test_log_twin(self):
        fig, _, twin, _ = make()
        twin.set_yscale("log")
        _, twin_ax = export(fig).axes
        assert twin_ax.get_yscale() == "log"

    def test_the_formatter_reaches_the_twin_axes_only(self):
        from matplotlib.ticker import FuncFormatter

        fig, _, twin, _ = make()
        twin.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}!"))
        host_ax, twin_ax = export(fig).axes
        twin_ax.figure.canvas.draw()
        assert all(t.get_text().endswith("!") for t in twin_ax.get_yticklabels() if t.get_text())
        assert not any(t.get_text().endswith("!") for t in host_ax.get_yticklabels())

    def test_one_legend_covers_both_axes(self):
        fig, host, _, _ = make()
        host.legend()
        host_ax, _ = export(fig).axes
        assert [t.get_text() for t in host_ax.get_legend().get_texts()] == [
            "temperature",
            "pressure",
        ]

    def test_twiny(self):
        fig, host = gplt.subplots()
        host.plot(T, np.sin(T))
        top = host.twiny()
        top.plot(T * 1000.0, np.sin(T))
        top.set_xlabel("ms")
        host_ax, top_ax = export(fig).axes
        assert top_ax.get_xlabel() == "ms"
        assert top_ax.get_xlim()[1] > 5000 and host_ax.get_xlim()[1] < 20
        assert host_ax.get_shared_y_axes().joined(host_ax, top_ax)

    def test_twinx_and_twiny_together(self):
        fig, host, _, _ = make()
        host.twiny().plot(T * 1000.0, np.sin(T))
        assert len(export(fig).axes) == 3

    def test_room_is_left_for_the_twin_labels(self):
        fig, _, _, _ = make()
        host_ax, _ = export(fig).axes
        assert host_ax.get_position().x1 < 0.92  # a plain figure ends at 0.96

    def test_room_is_left_in_a_grid_cell_too(self):
        fig, axs = gplt.subplots(1, 2)
        for a in axs:
            a.plot(T, np.sin(T))
        axs[1].twinx().plot(T, PRESSURE)
        plain, with_twin, _ = export(fig).axes
        assert with_twin.get_position().width < plain.get_position().width

    def test_a_panel_without_twins_exports_one_axes(self):
        fig, host = gplt.subplots()
        host.plot(T, T)
        assert len(export(fig).axes) == 1

    def test_twin_bars_and_text_are_drawn_on_the_twin_axes(self):
        fig, host, _, _ = make()
        twin = host.twinx()
        twin.bar([1, 2], [10, 20])
        twin.text(3, 150, "hi")
        axes = export(fig).axes
        assert len(axes) == 2
        assert any(len(a.patches) == 2 for a in axes)
        assert any(t.get_text() == "hi" for a in axes for t in a.texts)

    def test_a_dark_style_inks_the_twin_but_keeps_a_colour_the_caller_chose(self):
        fig, _, twin, _ = make()
        fig.options.visual.background_color = (0.0, 0.0, 0.0)
        twin.set_ylabel("kPa")
        host_ax, twin_ax = export(fig).axes
        assert twin_ax.yaxis.label.get_color() == host_ax.yaxis.label.get_color()
        twin.tick_params(axis="y", colors="tab:red")
        _, twin_ax = export(fig).axes
        assert twin_ax.yaxis.get_ticklabels()[0].get_color() == "tab:red"

    def test_savefig_writes_a_file(self, tmp_path):
        fig, _, _, _ = make()
        out = tmp_path / "twin.png"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.savefig(str(out))
        assert out.stat().st_size > 5000


# ---------------------------------------------------------------------------------------
# secondary_xaxis / secondary_yaxis: converted units, no data of their own
# ---------------------------------------------------------------------------------------

C_TO_F = (lambda c: c * 9.0 / 5.0 + 32.0, lambda f: (f - 32.0) * 5.0 / 9.0)
CELSIUS = np.linspace(0.0, 100.0, 50)


def make_secondary(location="top", functions=C_TO_F):
    fig, ax = gplt.subplots()
    ax.plot(CELSIUS, CELSIUS * 3.0)
    return fig, ax, ax.secondary_xaxis(location, functions=functions)


def secondary_ticks(fig):
    gplt._refresh_axis_ticks(fig)
    return fig.axis_manager.secondary_ticks


class TestSecondaryAxes:
    def test_it_is_registered_on_the_panel_and_does_not_touch_the_data(self):
        fig, ax, top = make_secondary()
        (state,) = fig.panels[0].secondaries
        assert (state.axis, state.location) == ("x", "top") and top._state is state
        assert len(ax.get_lines()) == 1 and ax.get_xlim() == (-5.0, 105.0)

    def test_locations_are_validated_per_axis(self):
        _, ax, _ = make_secondary()
        with pytest.raises(ValueError, match="location"):
            ax.secondary_xaxis("left")
        with pytest.raises(ValueError, match="location"):
            ax.secondary_yaxis("top")
        assert ax.secondary_yaxis("right")._state.axis == "y"

    def test_an_axes_fraction_goes_to_the_nearest_edge_and_says_so(self):
        _, ax, _ = make_secondary()
        with pytest.warns(gplt.MatplotlibCompatWarning, match="nearest edge"):
            near_top = ax.secondary_xaxis(0.9, functions=C_TO_F)
        assert near_top._state.location == "top"

    def test_functions_must_be_a_pair(self):
        _, ax, _ = make_secondary()
        with pytest.raises(TypeError, match="forward, inverse"):
            ax.secondary_xaxis("top", functions=(abs,))

    def test_identity_when_no_functions_are_given(self):
        fig, ax = gplt.subplots()
        ax.plot(CELSIUS, CELSIUS)
        ax.secondary_xaxis("top")
        fig._autoscale_all_panels()
        sec, ticks = secondary_ticks(fig)[0]
        assert [float(v) for v in ticks.labels] == pytest.approx(list(ticks.major), rel=1e-6)

    def test_ticks_are_round_numbers_in_the_new_units_at_the_old_positions(self):
        fig, _, _ = make_secondary()
        fig._autoscale_all_panels()
        ((sec, ticks),) = secondary_ticks(fig)
        fahrenheit = [float(v) for v in ticks.labels]
        assert all(v % 20 == 0 for v in fahrenheit) and len(fahrenheit) >= 4
        for position, value in zip(ticks.major, fahrenheit):
            assert C_TO_F[0](position) == pytest.approx(value, abs=1e-3)

    def test_zooming_the_host_reticks_it(self):
        fig, _, _ = make_secondary()
        fig._autoscale_all_panels()
        wide = secondary_ticks(fig)[0][1]
        fig.set_view(xlim=(20, 30), twin_follow=True)
        zoomed = secondary_ticks(fig)[0][1]
        assert max(float(v) for v in zoomed.labels) < max(float(v) for v in wide.labels)
        assert len(zoomed.labels) >= 3

    def test_a_non_linear_conversion(self):
        # period = 1 / frequency: the ticks are round periods, unevenly spaced in frequency.
        fig, ax = gplt.subplots()
        freq = np.linspace(1.0, 10.0, 50)
        ax.plot(freq, freq)
        ax.secondary_xaxis("top", functions=(lambda f: 1.0 / f, lambda p: 1.0 / p))
        fig._autoscale_all_panels()
        ((_, ticks),) = secondary_ticks(fig)
        periods = [float(v) for v in ticks.labels]
        gaps = np.diff(sorted(ticks.major))
        assert len(periods) >= 3 and not np.allclose(gaps, gaps[0])
        for position, period in zip(ticks.major, periods):
            assert 1.0 / position == pytest.approx(period, rel=1e-3)

    def test_a_function_that_raises_skips_the_axis_instead_of_the_frame(self):
        fig, _, _ = make_secondary(functions=(lambda v: 1 / 0, lambda v: v))
        fig._autoscale_all_panels()
        assert secondary_ticks(fig) == []

    def test_a_function_answering_nan_is_skipped(self):
        fig, _, _ = make_secondary(functions=(lambda v: v * np.nan, lambda v: v))
        fig._autoscale_all_panels()
        assert secondary_ticks(fig) == []

    def test_pinned_ticks_and_labels(self):
        fig, _, top = make_secondary()
        top.set_xticks([50, 100, 150], ["fifty", "hundred", "one-fifty"])
        fig._autoscale_all_panels()
        assert secondary_ticks(fig)[0][1].labels == ["fifty", "hundred", "one-fifty"]

    def test_pinned_ticks_outside_the_view_are_dropped(self):
        fig, _, top = make_secondary()
        top.set_xticks([100, 5000])
        fig._autoscale_all_panels()
        assert secondary_ticks(fig)[0][1].labels == ["100"]

    def test_set_functions_changes_the_conversion(self):
        fig, _, top = make_secondary()
        fig._autoscale_all_panels()
        top.set_functions((lambda c: c + 273.15, lambda k: k - 273.15))
        labels = [float(v) for v in secondary_ticks(fig)[0][1].labels]
        assert min(labels) >= 250  # Kelvin now, not Fahrenheit

    def test_a_y_secondary_along_the_right_edge(self):
        fig, ax = gplt.subplots()
        ax.plot(CELSIUS, np.linspace(1.5, 2.5, 50))
        ax.secondary_yaxis("right", functions=(lambda m: m * 3.28084, lambda ft: ft / 3.28084))
        fig._autoscale_all_panels()
        ((sec, ticks),) = secondary_ticks(fig)
        assert sec.axis == "y"
        for position, feet in zip(ticks.major, ticks.labels):
            assert position * 3.28084 == pytest.approx(float(feet), abs=1e-2)

    def test_label_colour_and_tick_colour(self):
        _, _, top = make_secondary()
        top.set_xlabel("Fahrenheit", color="tab:red", fontsize=14)
        top.tick_params(colors="tab:red")
        state = top._state
        assert (state.label, state.label_color, state.label_fontsize) == (
            "Fahrenheit",
            "tab:red",
            14.0,
        )
        assert state.tick_color == "tab:red" and top.get_xlabel() == "Fahrenheit"

    def test_set_color_colours_everything(self):
        _, _, top = make_secondary()
        top.set_color("g")
        assert top._state.tick_color == "g" and top._state.label_color == "g"

    def test_color_keyword_at_creation(self):
        _, ax, _ = make_secondary()
        assert ax.secondary_yaxis("right", color="b")._state.tick_color == "b"

    def test_unknown_setters_are_accepted_and_inert(self):
        _, _, top = make_secondary()
        with pytest.warns(gplt.MatplotlibCompatWarning, match="no effect"):
            top.set_alpha(0.5)
        with pytest.warns(gplt.MatplotlibCompatWarning, match="no effect"):
            top.xaxis.set_visible(True)

    def test_text_style_keywords_on_the_label_are_ignored_with_a_warning(self):
        _, _, top = make_secondary()
        with pytest.warns(gplt.MatplotlibCompatWarning, match="fixed text style"):
            top.set_xlabel("F", fontweight="bold")

    def test_ticklabels_need_pinned_ticks(self):
        _, _, top = make_secondary()
        with pytest.raises(ValueError, match="set_ticks"):
            top.set_xticklabels(["a"])

    def test_cla_forgets_secondaries(self):
        fig, ax, _ = make_secondary()
        ax.cla()
        assert fig.panels[0].secondaries == []

    def test_module_level_function_uses_the_current_axes(self):
        gplt.plot(CELSIUS, CELSIUS)
        strip = gplt.secondary_xaxis("top", functions=C_TO_F)
        assert strip._state in gplt.gcf().panels[0].secondaries

    def test_a_twin_and_a_secondary_can_share_a_panel(self):
        fig, ax, _ = make_secondary()
        ax.twinx().plot(CELSIUS, CELSIUS * 100.0)
        fig._autoscale_all_panels()
        assert len(secondary_ticks(fig)) == 1  # also refreshes the manager
        assert "y" in fig.axis_manager.twin_ticks


class TestSecondaryExport:
    def test_it_is_matplotlibs_own_secondary_axis(self):
        fig, _, _ = make_secondary()
        (host_ax,) = export(fig).axes
        assert len(host_ax.child_axes) == 1

    def test_label_and_colours_reach_it(self):
        fig, _, top = make_secondary()
        top.set_xlabel("Fahrenheit", color="tab:red")
        top.tick_params(colors="tab:red")
        (host_ax,) = export(fig).axes
        (strip,) = host_ax.child_axes
        assert strip.get_xlabel() == "Fahrenheit"
        assert strip.xaxis.label.get_color() == "tab:red"

    def test_the_conversion_is_applied(self):
        fig, _, _ = make_secondary()
        (host_ax,) = export(fig).axes
        host_ax.figure.canvas.draw()
        (strip,) = host_ax.child_axes
        low, high = strip.get_xlim()
        assert low == pytest.approx(C_TO_F[0](host_ax.get_xlim()[0]))
        assert high == pytest.approx(C_TO_F[0](host_ax.get_xlim()[1]))

    def test_pinned_ticks_reach_it(self):
        fig, _, top = make_secondary()
        top.set_xticks([50, 100], ["a", "b"])
        (host_ax,) = export(fig).axes
        host_ax.figure.canvas.draw()
        (strip,) = host_ax.child_axes
        assert [t.get_text() for t in strip.get_xticklabels()] == ["a", "b"]

    def test_a_y_secondary(self):
        fig, ax = gplt.subplots()
        ax.plot(CELSIUS, np.linspace(1.5, 2.5, 50))
        ax.secondary_yaxis("right", functions=(lambda m: m * 3.28, lambda ft: ft / 3.28))
        (host_ax,) = export(fig).axes
        host_ax.figure.canvas.draw()  # a secondary axis' limits are computed when it draws
        (strip,) = host_ax.child_axes
        assert strip.get_ylim()[1] == pytest.approx(host_ax.get_ylim()[1] * 3.28)

    def test_room_is_left_for_a_top_or_right_secondary_only(self):
        fig, ax = gplt.subplots()
        ax.plot(CELSIUS, CELSIUS)
        ax.secondary_xaxis("top", functions=C_TO_F)
        (with_top,) = export(fig).axes
        fig2, ax2 = gplt.subplots()
        ax2.plot(CELSIUS, CELSIUS)
        ax2.secondary_xaxis("bottom", functions=C_TO_F)
        (with_bottom,) = export(fig2).axes
        assert with_top.get_position().y1 < with_bottom.get_position().y1

    def test_a_dark_style_inks_a_secondary_but_keeps_a_colour_the_caller_chose(self):
        fig, _, top = make_secondary()
        fig.options.visual.background_color = (0.0, 0.0, 0.0)
        top.set_xlabel("F")
        (host_ax,) = export(fig).axes
        (strip,) = host_ax.child_axes
        assert strip.xaxis.label.get_color() == host_ax.xaxis.label.get_color()
        top.tick_params(colors="tab:red")
        (host_ax,) = export(fig).axes
        (strip,) = host_ax.child_axes
        assert strip.xaxis.get_ticklabels()[0].get_color() == "tab:red"

    def test_a_panel_without_secondaries_has_no_child_axes(self):
        fig, ax = gplt.subplots()
        ax.plot(CELSIUS, CELSIUS)
        (host_ax,) = export(fig).axes
        assert host_ax.child_axes == []

    def test_savefig_writes_a_file(self, tmp_path):
        fig, _, top = make_secondary()
        top.set_xlabel("F")
        out = tmp_path / "secondary.png"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.savefig(str(out))
        assert out.stat().st_size > 5000
