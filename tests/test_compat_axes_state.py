"""Per-axes state, inverted axes, and the matplotlib idioms that used to raise or be dropped.

Companion to ``test_figure_axes_oo_surface.py``. That file covers the methods that exist on
``fig``/``ax``; this one covers behaviour that has to hold *between panels* and *through the
export*, which is where the real bugs were: ``axs[0].set_xticks(...)`` changing every panel,
``fig.suptitle`` overwriting a panel's title, ``set_ylim(hi, lo)`` producing a +-1e12 view.

Nothing here opens a window. The export is inspected through ``_build_preview_figure``, which
returns the real matplotlib figure a ``savefig`` would write, so assertions are about what was
actually drawn rather than about what was stored.
"""

from __future__ import annotations

import io
import warnings

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.controllers import CameraController
from glplot.core.legacy import CameraState
from glplot.options import EngineOptions
from glplot.renderers.axis import _screen_edges
from glplot.utils.preview import _build_preview_figure, _spaced_rect

X = np.linspace(0.0, 10.0, 100)


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


def export(fig):
    """The matplotlib figure ``fig.savefig()`` would write."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _build_preview_figure(fig, 1.0)


# ---------------------------------------------------------------------------------------
# Inverted axes
# ---------------------------------------------------------------------------------------


class TestInvertedAxes:
    def test_reversed_limits_invert_instead_of_exploding(self):
        # fit_bounds used to clip a negative span to zoom_min and report +-1e12.
        _, ax = gplt.subplots()
        ax.plot(X, X)
        ax.set_xlim(10, 0)
        lo, hi = ax.get_xlim()
        assert (lo, hi) == (10.0, 0.0)
        assert ax.xaxis_inverted() and not ax.yaxis_inverted()

    def test_set_ylim_top_bottom_order_is_the_direction(self):
        _, ax = gplt.subplots()
        ax.plot(X, X)
        ax.set_ylim(5, 0)
        assert ax.get_ylim() == (5.0, 0.0)
        ax.set_ylim(0, 5)
        assert ax.get_ylim() == (0.0, 5.0)
        assert not ax.yaxis_inverted()

    def test_invert_axis_toggles_and_keeps_the_range(self):
        _, ax = gplt.subplots()
        ax.plot(X, X)
        lo, hi = ax.get_ylim()
        ax.invert_yaxis()
        assert ax.get_ylim() == (hi, lo) and ax.yaxis_inverted()
        ax.invert_yaxis()
        assert ax.get_ylim() == (lo, hi) and not ax.yaxis_inverted()

    def test_a_partial_limit_call_keeps_the_direction(self):
        _, ax = gplt.subplots()
        ax.plot(X, X)
        ax.invert_yaxis()
        ax.set_xlim(0, 5)  # only x given
        assert ax.yaxis_inverted()

    def test_get_xlim_after_invert_is_reversed(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        ax.invert_xaxis()
        left, right = ax.get_xlim()
        assert left > right

    def test_yaxis_set_inverted(self):
        _, ax = gplt.subplots()
        ax.plot(X, X)
        ax.yaxis.set_inverted(True)
        assert ax.yaxis.get_inverted()
        ax.yaxis.set_inverted(False)
        assert not ax.yaxis_inverted()

    def test_projection_mirrors_and_screen_to_world_inverts_it(self):
        options = EngineOptions()
        camera = CameraState(cx=5.0, cy=5.0, zoom_x=0.2, zoom_y=0.2)
        controller = CameraController(camera, options)
        px, py = 400, 300
        plain = controller.mvp(800, 600)
        controller.camera.flip_y = True
        flipped = controller.mvp(800, 600)
        assert flipped[1][1] == pytest.approx(-plain[1][1])
        assert flipped[0][0] == pytest.approx(plain[0][0])
        # screen -> world must still be the exact inverse of mvp
        wx, wy = controller.screen_to_world(px, py, 800, 600)
        clip = flipped.astype(float) @ np.array([wx, wy, 0.0, 1.0])
        sx = (clip[0] + 1.0) * 0.5 * 800
        sy = (1.0 - clip[1]) * 0.5 * 600
        assert sx == pytest.approx(px, abs=1e-3) and sy == pytest.approx(py, abs=1e-3)

    def test_fit_bounds_sorts_a_reversed_pair(self):
        camera = CameraState()
        CameraController(camera, EngineOptions()).fit_bounds(10, 0, 4, -4, 800, 600)
        assert camera.cx == pytest.approx(5.0) and camera.zoom_x == pytest.approx(0.2)
        assert camera.cy == pytest.approx(0.0) and camera.zoom_y == pytest.approx(0.25)

    def test_chrome_stays_on_the_screens_bottom_edge_when_y_is_inverted(self):
        win = (0.0, 10.0, 0.0, 5.0)
        options, camera = EngineOptions(), CameraState(cx=5.0, cy=2.5, zoom_x=0.2, zoom_y=0.4)
        controller = CameraController(camera, options)
        left, right, bottom, top, fx, fy = _screen_edges(controller.mvp(800, 600), win)
        assert (bottom, top, fy) == (0.0, 5.0, False)
        camera.flip_y = True
        left, right, bottom, top, fx, fy = _screen_edges(controller.mvp(800, 600), win)
        assert (bottom, top, fy) == (5.0, 0.0, True)

    def test_inversion_is_shared_across_linked_axes(self):
        fig, axs = gplt.subplots(1, 2, sharey=True)
        for a in axs:
            a.plot(X, X)
        axs[0].invert_yaxis()
        assert axs[1].yaxis_inverted()

    def test_imshow_defaults_to_matplotlibs_extent_and_inverted_y(self):
        _, ax = gplt.subplots()
        ax.imshow(np.zeros((20, 30)))  # origin='upper'
        assert ax.get_images()[0].metadata["extent"] == (-0.5, 29.5, 19.5, -0.5)
        assert ax.yaxis_inverted() and not ax.xaxis_inverted()

    def test_imshow_lower_origin_is_not_inverted(self):
        _, ax = gplt.subplots()
        ax.imshow(np.zeros((20, 30)), origin="lower")
        assert not ax.yaxis_inverted()

    def test_a_contour_over_an_image_does_not_un_invert_it(self):
        _, ax = gplt.subplots()
        ax.imshow(np.zeros((20, 30)))
        gx, gy = np.meshgrid(np.arange(30), np.arange(20))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.contour(gx, gy, gx + gy)
        assert ax.yaxis_inverted()

    def test_export_of_an_inverted_axis_is_inverted(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        ax.invert_yaxis()
        exported = export(fig).axes[0]
        assert exported.yaxis_inverted() and not exported.xaxis_inverted()

    def test_export_of_pinned_reversed_limits(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        ax.set_xlim(8, 2)
        assert export(fig).axes[0].get_xlim() == (8.0, 2.0)


class TestPartialLimits:
    @pytest.mark.parametrize(
        "call,expect",
        [
            (lambda a: a.set_xlim(left=0), "x"),
            (lambda a: a.set_xlim(right=5), "x"),
            (lambda a: a.set_ylim(top=2), "y"),
            (lambda a: a.set_ylim(bottom=-2), "y"),
        ],
    )
    def test_one_end_moves_and_the_other_stays(self, call, expect):
        _, ax = gplt.subplots()
        ax.plot(X, np.sin(X))
        before = ax.get_xlim() if expect == "x" else ax.get_ylim()
        call(ax)
        after = ax.get_xlim() if expect == "x" else ax.get_ylim()
        assert before != after
        assert (before[0] == after[0]) != (before[1] == after[1])

    def test_module_level_none_end(self):
        gplt.plot(X, X)
        gplt.xlim(None, 5)
        assert gplt.xlim()[1] == 5.0


# ---------------------------------------------------------------------------------------
# Per-axes state
# ---------------------------------------------------------------------------------------


class TestPerPanelState:
    def test_pinned_ticks_belong_to_one_panel(self):
        fig, axs = gplt.subplots(1, 2)
        for a in axs:
            a.plot(X, X)
        axs[0].set_xticks([0, 5, 10])
        axs[1].set_xticks([1, 2])
        assert list(axs[0].get_xticks()) == [0.0, 5.0, 10.0]
        assert list(axs[1].get_xticks()) == [1.0, 2.0]

    def test_scale_belongs_to_one_panel(self):
        fig, axs = gplt.subplots(1, 3)
        axs[1].set_yscale("log")
        assert [a.get_yscale() for a in axs] == ["linear", "log", "linear"]

    def test_scale_survives_switching_back_and_forth(self):
        fig, axs = gplt.subplots(1, 2)
        axs[0].set_xscale("log")
        axs[1].set_xscale("symlog")
        for _ in range(3):
            assert axs[0].get_xscale() == "log" and axs[1].get_xscale() == "symlog"

    def test_formatter_and_locator_belong_to_one_panel(self):
        from matplotlib.ticker import FuncFormatter, MaxNLocator

        fig, axs = gplt.subplots(1, 2)
        fmt = FuncFormatter(lambda v, p: f"{v:g}s")
        axs[0].xaxis.set_major_formatter(fmt)
        axs[0].xaxis.set_major_locator(MaxNLocator(4))
        assert axs[0].xaxis.get_major_formatter() is fmt
        assert axs[1].xaxis.get_major_formatter() is None
        assert axs[1].xaxis.get_major_locator() is None

    def test_rotation_belongs_to_one_panel(self):
        fig, axs = gplt.subplots(1, 2)
        for a in axs:
            a.plot(X, X)
        axs[1].tick_params(axis="x", rotation=30)
        assert fig.axes_option(axs[0].panel, "axis_tick_rotation_x") == 0.0
        assert fig.axes_option(axs[1].panel, "axis_tick_rotation_x") == 30.0

    def test_legend_belongs_to_one_panel(self):
        fig, axs = gplt.subplots(1, 2)
        axs[0].plot(X, X, label="a")
        axs[1].plot(X, X, label="b")
        axs[1].legend()
        assert axs[0].get_legend() is None
        assert axs[1].get_legend() is not None
        axs[1].get_legend().remove()
        assert axs[1].get_legend() is None

    def test_new_panels_start_from_the_defaults(self):
        fig = gplt.figure()
        fig.set_panels([])  # one fresh panel
        gplt.xticks([1, 2, 3])
        fig, axs = gplt.subplots(1, 2)
        assert fig.options.axis_tick_values_x is None
        assert axs[1].get_xscale() == "linear"

    def test_set_panels_resets_what_the_old_panel_set(self):
        fig, ax = gplt.subplots()
        ax.set_yscale("log")
        axs = fig.subplots(1, 2)
        assert [a.get_yscale() for a in axs] == ["linear", "linear"]

    def test_a_single_panel_figure_behaves_as_before(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        gplt.xticks([0, 5, 10])
        gplt.yscale("linear")
        assert list(ax.get_xticks()) == [0.0, 5.0, 10.0]
        assert fig.options.axis_tick_values_x == (0.0, 5.0, 10.0)

    def test_reading_another_panels_state_does_not_change_the_active_one(self):
        fig, axs = gplt.subplots(1, 2)
        axs[0].set_xscale("log")
        gplt.sca(axs[1])
        axs[0].get_xscale()
        assert fig.active_panel_index == 1

    def test_split_view_keeps_the_active_panels_settings(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        ax.set_yscale("log")
        fig.split_view(1, 2)
        assert fig.options.axis_scale_y == "log"
        assert fig.axes_option(fig.panels[1], "axis_scale_y") == "linear"


class TestPerPanelExport:
    def test_each_exported_axes_has_its_own_scale(self):
        fig, axs = gplt.subplots(1, 2)
        for a in axs:
            a.plot(X + 1, X + 1)
        axs[1].set_yscale("log")
        exported = export(fig).axes
        assert [a.get_yscale() for a in exported] == ["linear", "log"]

    def test_each_exported_axes_has_its_own_ticks(self):
        fig, axs = gplt.subplots(1, 2)
        for a in axs:
            a.plot(X, X)
        axs[0].set_xticks([0, 5, 10])
        axs[1].set_xticks([2, 4])
        exported = export(fig).axes
        assert list(exported[0].get_xticks()) == [0.0, 5.0, 10.0]
        assert list(exported[1].get_xticks()) == [2.0, 4.0]

    def test_a_formatter_reaches_only_its_own_axes(self):
        from matplotlib.ticker import FuncFormatter

        fig, axs = gplt.subplots(1, 2)
        for a in axs:
            a.plot(X, X)
        axs[0].xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}s"))
        first, second = export(fig).axes
        for ax in (first, second):
            ax.figure.canvas.draw()
        assert all(t.get_text().endswith("s") for t in first.get_xticklabels() if t.get_text())
        assert not any(t.get_text().endswith("s") for t in second.get_xticklabels())

    def test_rotation_reaches_only_its_own_axes(self):
        fig, axs = gplt.subplots(1, 2)
        for a in axs:
            a.plot(X, X)
        gplt.setp(axs[0].get_xticklabels(), rotation=45)
        first, second = export(fig).axes
        assert first.get_xticklabels()[0].get_rotation() == 45.0
        assert second.get_xticklabels()[0].get_rotation() == 0.0

    def test_a_panels_legend_is_drawn_only_when_it_asked(self):
        fig, axs = gplt.subplots(1, 3)
        for a in axs:
            a.plot(X, X, label="a")
        axs[0].legend()
        gplt.sca(axs[0])  # primary
        legends = [a.get_legend() is not None for a in export(fig).axes]
        assert legends[0] is True  # the primary panel's historical behaviour
        axs[1].legend()
        gplt.sca(axs[0])
        legends = [a.get_legend() is not None for a in export(fig).axes]
        assert legends == [True, True, False]

    def test_export_restores_the_active_panel_and_its_state(self):
        fig, axs = gplt.subplots(1, 2)
        axs[0].set_xscale("log")
        axs[1].set_xscale("symlog")
        before = fig.active_panel_index
        export(fig)
        assert fig.active_panel_index == before
        assert [a.get_xscale() for a in axs] == ["log", "symlog"]


class TestFigureTitles:
    def test_suptitle_does_not_replace_a_panel_title(self):
        fig, axs = gplt.subplots(1, 2)
        axs[0].set_title("left")
        axs[1].set_title("right")
        with pytest.warns(gplt.MatplotlibCompatWarning, match="exported figures only"):
            fig.suptitle("whole figure")
        assert (axs[0].get_title(), axs[1].get_title()) == ("left", "right")
        exported = export(fig)
        assert exported._suptitle.get_text() == "whole figure"
        assert [a.get_title() for a in exported.axes] == ["left", "right"]

    def test_suptitle_on_a_single_panel_is_still_its_title(self):
        fig, ax = gplt.subplots()
        fig.suptitle("only")
        assert ax.get_title() == "only"

    def test_supxlabel_and_supylabel_reach_the_export(self):
        fig, axs = gplt.subplots(1, 2)
        for a in axs:
            a.plot(X, X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.supxlabel("time")
            fig.supylabel("value")
        exported = export(fig)
        assert exported._supxlabel.get_text() == "time"
        assert exported._supylabel.get_text() == "value"


class TestPanelLayoutInExport:
    def test_grid_panels_are_spaced_apart_so_labels_have_room(self):
        fig, axs = gplt.subplots(1, 2)
        for a in axs:
            a.plot(X, X)
        left, right = export(fig).axes
        assert left.get_position().x1 < right.get_position().x0

    def test_an_explicit_rect_is_left_exactly_where_it_was_put(self):
        fig, axs = gplt.subplots(1, 2)
        axs[1].set_position([0.55, 0.1, 0.3, 0.3])
        panels = fig.panels
        assert _spaced_rect(panels[1], panels, 0.2, 0.2) == (0.55, 0.1, 0.3, 0.3)

    def test_an_edge_with_no_neighbour_is_not_pulled_in(self):
        fig, axs = gplt.subplots(1, 2)
        x0, y0, w, h = _spaced_rect(fig.panels[0], fig.panels, 0.2, 0.2)
        assert (x0, y0, h) == (0.0, 0.0, 1.0) and w < 0.5

    def test_set_visible_false_hides_a_panel_from_the_export(self):
        fig, axs = gplt.subplots(1, 3)
        for a in axs:
            a.plot(X, X)
        axs[2].set_visible(False)
        assert len(export(fig).axes) == 2
        assert axs[2].get_visible() is False

    def test_delaxes_leaves_a_blank_cell_and_other_axes_keep_their_index(self):
        fig, axs = gplt.subplots(1, 3)
        for a in axs:
            a.plot(X, X)
        fig.delaxes(axs[1])
        assert [a._index for a in fig.axes] == [0, 2]
        assert axs[2] in fig.axes and axs[1] not in fig.axes
        assert len(export(fig).axes) == 2

    def test_delaxes_on_the_only_axes_clears_the_figure(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        fig.delaxes(ax)
        assert fig.scene.layers == []


# ---------------------------------------------------------------------------------------
# Export drawing
# ---------------------------------------------------------------------------------------


class TestExportDrawing:
    def test_a_bar_is_one_polygon_not_two_triangles(self):
        fig, ax = gplt.subplots()
        ax.bar([0, 1, 2], [3, 5, 2], color="tab:orange", edgecolor="k", linewidth=2)
        patches = export(fig).axes[0].patches
        assert len(patches) == 3
        assert all(len(p.get_xy()) == 5 for p in patches)  # 4 corners + the closing point

    def test_bar_edgecolor_and_linewidth_reach_the_export(self):
        fig, ax = gplt.subplots()
        ax.bar([0], [1], color="tab:orange", edgecolor="k", linewidth=3)
        patch = export(fig).axes[0].patches[0]
        assert patch.get_edgecolor()[:3] == (0.0, 0.0, 0.0)
        assert patch.get_linewidth() == 3.0

    def test_unstyled_bar_edge_matches_its_face(self):
        fig, ax = gplt.subplots()
        ax.bar([0], [1], color="tab:orange")
        patch = export(fig).axes[0].patches[0]
        assert patch.get_edgecolor() == patch.get_facecolor()

    def test_hist_edgecolor_reaches_the_export(self):
        fig, ax = gplt.subplots()
        ax.hist(np.random.default_rng(0).normal(size=200), bins=8, edgecolor="k", facecolor="g")
        patches = export(fig).axes[0].patches
        assert len(patches) == 8
        assert all(p.get_edgecolor()[:3] == (0.0, 0.0, 0.0) for p in patches)

    def test_bar_tick_label_pins_the_ticks(self):
        fig, ax = gplt.subplots()
        ax.bar([0, 1, 2], [1, 2, 3], tick_label=["a", "b", "c"])
        exported = export(fig).axes[0]
        exported.figure.canvas.draw()
        assert [t.get_text() for t in exported.get_xticklabels()] == ["a", "b", "c"]

    def test_bar_hatch_says_it_is_ignored(self):
        _, ax = gplt.subplots()
        with pytest.warns(gplt.MatplotlibCompatWarning, match="hatch"):
            ax.bar([0, 1], [1, 2], hatch="//")

    def test_text_in_axes_fractions_is_pinned_in_the_export(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        ax.text(0.05, 0.95, "corner", transform=ax.transAxes, va="top")
        exported = export(fig).axes[0]
        (text,) = [t for t in exported.texts if t.get_text() == "corner"]
        assert text.get_position() == (0.05, 0.95)
        assert text.get_transform() == exported.transAxes
        assert text.get_va() == "top"

    def test_text_alignment_and_rotation_reach_the_export(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        ax.text(5, 5, "t", ha="center", va="bottom", rotation=30)
        (text,) = [t for t in export(fig).axes[0].texts if t.get_text() == "t"]
        assert (text.get_ha(), text.get_va(), text.get_rotation()) == ("center", "bottom", 30.0)

    def test_savefig_to_a_buffer(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        assert buffer.getvalue()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_a_uniform_scatter_exports_in_one_colour_collection(self):
        fig, ax = gplt.subplots()
        ax.scatter(X, X, s=4)
        (collection,) = export(fig).axes[0].collections
        assert len(collection.get_facecolor()) == 1


# ---------------------------------------------------------------------------------------
# Ticks
# ---------------------------------------------------------------------------------------


class TestTicks:
    def test_no_negative_zero_label(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        ax.set_xlim(-1, 1)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "-0" not in labels and "-0.0" not in labels and "0.0" in labels

    def test_formatter_labels_the_live_ticks(self):
        from matplotlib.ticker import FuncFormatter

        fig, ax = gplt.subplots()
        ax.plot(X, X)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}s"))
        assert all(t.get_text().endswith("s") for t in ax.get_xticklabels())

    def test_a_format_string_is_a_formatter(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        ax.yaxis.set_major_formatter("{x:.1f}%")
        # StrMethodFormatter is called with (value, position): the live labels end in '%'
        assert all(t.get_text().endswith("%") for t in ax.get_yticklabels())

    @pytest.mark.parametrize(
        "locator,expect",
        [
            ("MaxNLocator", {"axis_tick_count_x": 5}),
            ("MultipleLocator", {"axis_tick_step_x": 2.0}),
            ("FixedLocator", {"axis_tick_values_x": (1.0, 2.0)}),
            ("NullLocator", {"axis_tick_values_x": ()}),
        ],
    )
    def test_locators_map_onto_the_live_ticker(self, locator, expect):
        from matplotlib import ticker

        built = {
            "MaxNLocator": lambda: ticker.MaxNLocator(4),
            "MultipleLocator": lambda: ticker.MultipleLocator(2.0),
            "FixedLocator": lambda: ticker.FixedLocator([1.0, 2.0]),
            "NullLocator": lambda: ticker.NullLocator(),
        }[locator]()
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        ax.xaxis.set_major_locator(built)
        for name, value in expect.items():
            assert getattr(fig.options, name) == value

    def test_an_unmapped_locator_is_kept_for_the_export_and_warned(self):
        from matplotlib.ticker import LogLocator

        fig, ax = gplt.subplots()
        ax.plot(X, X)
        with pytest.warns(gplt.MatplotlibCompatWarning, match="LogLocator"):
            ax.xaxis.set_major_locator(LogLocator())
        assert isinstance(ax.xaxis.get_major_locator(), LogLocator)

    def test_set_xticks_then_labels_pairs_with_the_pinned_ticks(self):
        # Used to raise "got 3 labels but 1 ticks are currently on the x-axis".
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        ax.set_xticks([0, 5, 10])
        ax.set_xticklabels(["a", "b", "c"])
        assert list(ax.get_xticks()) == [0.0, 5.0, 10.0]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]

    def test_ticks_read_before_the_first_render_belong_to_the_data(self):
        fig, ax = gplt.subplots()
        ax.plot(X, 100 * np.sin(X))
        assert max(abs(ax.get_yticks())) > 50  # not the (-1, 1) default's ticks

    def test_xticks_rotation_keyword(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        with pytest.warns(gplt.MatplotlibCompatWarning, match="exported figures only"):
            gplt.xticks(rotation=45)
        assert fig.options.axis_tick_rotation_x == 45.0

    def test_rotation_accepts_matplotlibs_words(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gplt.xticks(rotation="vertical")
        assert fig.options.axis_tick_rotation_x == 90.0

    def test_bad_rotation_word_raises(self):
        gplt.plot(X, X)
        with pytest.raises(ValueError, match="rotation"):
            gplt.xticks(rotation="sideways")

    def test_tick_params_labelrotation(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.tick_params(axis="y", labelrotation=60)
        assert (fig.options.axis_tick_rotation_x, fig.options.axis_tick_rotation_y) == (0.0, 60.0)

    def test_setp_on_tick_labels_rotates_them(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gplt.setp(ax.get_xticklabels(), rotation=45)
        assert fig.options.axis_tick_rotation_x == 45.0


class TestTextProperties:
    @pytest.mark.parametrize(
        "call",
        [
            lambda ax: ax.set_title("t", fontweight="bold", loc="left", pad=8),
            lambda ax: ax.set_xlabel("x", labelpad=6, fontstyle="italic", rotation=10),
            lambda ax: ax.set_ylabel("y", rotation=0, va="center", family="serif"),
        ],
    )
    def test_text_style_keywords_are_accepted_and_said_to_be_ignored(self, call):
        _, ax = gplt.subplots()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call(ax)
        assert any(issubclass(w.category, gplt.MatplotlibCompatWarning) for w in caught)

    def test_a_typo_is_still_an_error(self):
        _, ax = gplt.subplots()
        with pytest.raises(TypeError):
            ax.set_title("t", fontweigth="bold")

    def test_the_text_still_reaches_the_axes(self):
        _, ax = gplt.subplots()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.set_title("hello", fontweight="bold")
        assert ax.get_title() == "hello"


# ---------------------------------------------------------------------------------------
# plot(): columns, artist handles, containers
# ---------------------------------------------------------------------------------------


class TestPlotColumns:
    def test_a_2d_y_draws_one_line_per_column(self):
        _, ax = gplt.subplots()
        artists = ax.plot(X, np.c_[np.sin(X), np.cos(X), X])
        assert len(artists) == 3

    def test_a_lone_2d_array_is_columns_against_the_index(self):
        _, ax = gplt.subplots()
        artists = ax.plot(np.random.default_rng(0).random((10, 4)))
        assert len(artists) == 4
        assert list(artists[0].get_xdata()) == list(range(10))

    def test_x_and_y_matrices_pair_column_by_column(self):
        _, ax = gplt.subplots()
        Xm = np.column_stack([X, X + 1])
        Ym = np.column_stack([X, X * 2])
        artists = ax.plot(Xm, Ym)
        assert len(artists) == 2
        assert artists[1].get_xdata()[0] == pytest.approx(1.0)

    def test_labels_are_handed_out_per_column(self):
        _, ax = gplt.subplots()
        artists = ax.plot(X, np.c_[np.sin(X), np.cos(X)], label=["sin", "cos"])
        assert [a.label for a in artists] == ["sin", "cos"]

    def test_one_label_for_several_columns_labels_each(self):
        _, ax = gplt.subplots()
        artists = ax.plot(X, np.c_[np.sin(X), np.cos(X)], label="both")
        assert [a.label for a in artists] == ["both", "both"]

    def test_mismatched_matrix_shapes_still_raise(self):
        _, ax = gplt.subplots()
        with pytest.raises(ValueError):
            ax.plot(np.zeros((5, 2)), np.zeros((6, 2)))

    def test_ordinary_1d_calls_are_untouched(self):
        _, ax = gplt.subplots()
        assert len(ax.plot(X, X)) == 1
        assert len(ax.plot(X, X, "r--", X, -X, "b-")) == 2


class TestArtistHandles:
    def test_style_setters_write_the_layer_style(self):
        _, ax = gplt.subplots()
        (line,) = ax.plot(X, X)
        line.set_color("red")
        line.set_linewidth(4)
        line.set_alpha(0.5)
        assert line.get_color() == (1.0, 0.0, 0.0, 1.0)
        assert (line.get_linewidth(), line.get_alpha()) == (4.0, 0.5)

    def test_set_label_replaces_the_bookkeeping_name(self):
        _, ax = gplt.subplots()
        (line,) = ax.plot(X, X)
        line.set_label("mine")
        assert line.get_label() == "mine"
        assert ax.get_legend_handles_labels()[1] == ["mine"]

    def test_set_visible_and_zorder(self):
        _, ax = gplt.subplots()
        (line,) = ax.plot(X, X)
        line.set_visible(False)
        line.set_zorder(5)
        assert line.get_visible() is False and line.get_zorder() == 5.0

    def test_scatter_set_color_recolours_every_point(self):
        _, ax = gplt.subplots()
        sc = ax.scatter(X, X, c=X)
        sc.set_color("blue")
        assert np.allclose(sc.colors, [0.0, 0.0, 1.0, 1.0])

    def test_setp_on_a_layer(self):
        _, ax = gplt.subplots()
        (line,) = ax.plot(X, X)
        gplt.setp(line, lw=3, color="g", alpha=0.25)
        assert line.get_linewidth() == 3.0 and line.get_alpha() == 0.25

    def test_setp_warns_about_what_it_cannot_set(self):
        _, ax = gplt.subplots()
        (line,) = ax.plot(X, X)
        with pytest.warns(gplt.MatplotlibCompatWarning, match="bogus"):
            gplt.setp(line, bogus=1)

    def test_containers_follow_the_layers(self):
        _, ax = gplt.subplots()
        ax.plot(X, X)
        ax.scatter(X, X)
        ax.bar([0, 1], [1, 2])
        ax.text(1, 1, "t")
        assert (len(ax.lines), len(ax.collections), len(ax.patches), len(ax.texts)) == (1, 1, 2, 1)

    def test_lines_is_still_callable_as_glplots_own_lines(self):
        _, ax = gplt.subplots()
        ax.plot(X, X)
        assert callable(ax.lines)  # ``ax.lines(...)`` is GLPlot's many-lines-in-one-call API
        assert ax.lines[0] is ax.lines[-1]  # and ``ax.lines[i]`` is plain list indexing

    def test_get_children_and_window_extent(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        assert len(ax.get_children()) >= 1
        box = ax.get_window_extent()
        assert (box.width, box.height) == (float(fig.width), float(fig.height))

    def test_add_line_draws_a_matplotlib_line2d(self):
        from matplotlib.lines import Line2D

        _, ax = gplt.subplots()
        ax.add_line(Line2D([0, 1, 2], [0, 1, 0], color="r", label="seg"))
        assert len(ax.lines) == 1 and ax.lines[0].label == "seg"

    def test_set_prop_cycle_chooses_the_next_colours(self):
        _, ax = gplt.subplots()
        ax.set_prop_cycle(color=["red", "blue"])
        first, second, third = (ax.plot(X, X + i)[0] for i in range(3))
        assert first.get_color()[:3] == (1.0, 0.0, 0.0)
        assert second.get_color()[:3] == (0.0, 0.0, 1.0)
        assert third.get_color()[:3] == (1.0, 0.0, 0.0)


class TestColorbarAxes:
    def test_the_bars_axes_accepts_matplotlibs_calls(self):
        fig, ax = gplt.subplots()
        image = ax.imshow(np.random.default_rng(0).random((8, 8)))
        bar = fig.colorbar(image, ax=ax)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bar.ax.tick_params(labelsize=8)
            bar.ax.set_yticklabels(["a", "b"])
            bar.ax.yaxis.set_ticks_position("left")
        bar.ax.set_yticks([0.2, 0.8])
        assert fig.panels[0].colorbars[0].ticks == [0.2, 0.8]


class TestFigureDpiAndCanvas:
    def test_dpi_round_trips_and_sets_inches(self):
        fig = gplt.figure(figsize=(6, 4), dpi=150)
        assert fig.dpi == 150.0
        assert tuple(fig.get_size_inches()) == (6.0, 4.0)
        fig.dpi = 100
        assert tuple(fig.get_size_inches()) == (9.0, 6.0)

    def test_canvas_reports_the_pixel_size(self):
        fig = gplt.figure(width=640, height=480)
        assert fig.canvas.get_width_height() == (640, 480)

    def test_an_engines_dpi_is_not_an_animation_frame_dpi(self):
        # `fig.dpi` exists now; the animation frame size must not start following it.
        from glplot import animation

        fig = gplt.figure(figsize=(6, 4), dpi=150)
        assert animation._default_dpi(fig) == animation._default_dpi(object())


# ---------------------------------------------------------------------------------------
# plt.close: only a figure object used to be understood, so close("all") closed nothing
# ---------------------------------------------------------------------------------------


class TestClose:
    def test_close_all_closes_every_figure(self):
        for _ in range(3):
            gplt.figure()
        gplt.close("all")
        assert gplt._ALL_PLOTS == [] and gplt._CURRENT_PLOT is None

    def test_close_no_argument_closes_the_current_figure(self):
        first = gplt.figure()
        second = gplt.figure()
        gplt.close()
        assert second not in gplt._ALL_PLOTS and first in gplt._ALL_PLOTS
        assert gplt.gcf() is first

    def test_close_by_figure_object(self):
        fig = gplt.figure()
        keep = gplt.figure()
        gplt.close(fig)
        assert fig not in gplt._ALL_PLOTS and keep in gplt._ALL_PLOTS

    def test_close_by_number_and_by_name(self):
        numbered = gplt.figure(7)
        named = gplt.figure("results")
        gplt.close(7)
        assert numbered not in gplt._ALL_PLOTS and named in gplt._ALL_PLOTS
        gplt.close("results")
        assert named not in gplt._ALL_PLOTS

    def test_closing_a_number_does_not_let_figure_hand_the_old_one_back(self):
        old = gplt.figure(5)
        gplt.close(5)
        assert gplt.figure(5) is not old

    def test_an_unknown_figure_is_ignored(self):
        fig = gplt.figure()
        gplt.close(99)
        gplt.close("nope")
        assert fig in gplt._ALL_PLOTS

    def test_a_closed_figure_is_released(self):
        import gc
        import weakref

        gplt.imshow(np.random.default_rng(0).random((50, 50)))
        ref = weakref.ref(gplt.gcf())
        gplt.close("all")
        gc.collect()
        assert ref() is None


# ---------------------------------------------------------------------------------------
# Autoscale: one panel's limits must not leave the others unfitted
# ---------------------------------------------------------------------------------------


class TestPerPanelAutoscale:
    def test_fitting_skips_a_panel_whose_view_was_set(self):
        fig, axs = gplt.subplots(1, 3)
        for a in axs:
            a.plot(X, 100 * X)
        axs[0].set_ylim(0, 10)  # pinned: must stay exactly as asked
        fig._autoscale_all_panels()
        assert axs[0].get_ylim() == (0.0, 10.0)
        for other in axs[1:]:
            assert other.get_ylim()[1] > 900  # fitted to the 0..1000 data

    def test_a_log_scale_on_one_panel_does_not_stop_the_rest_fitting(self):
        # `set_yscale` autoscales its own panel and used to switch the whole figure's
        # initial fit off, so the other panels opened empty at (-1, 1).
        fig, axs = gplt.subplots(1, 2)
        for a in axs:
            a.plot(X + 1, X + 1)
        axs[1].set_yscale("log")
        fig._autoscale_all_panels()
        assert axs[0].get_xlim()[1] > 10

    def test_the_first_fit_is_not_a_deliberate_pin(self):
        fig, axs = gplt.subplots(1, 2)
        for a in axs:
            a.plot(X, X)
        fig._autoscale_all_panels()
        assert not any(p.view_pinned for p in fig.panels)
        assert not any(p.needs_initial_autoscale for p in fig.panels)

    def test_a_fit_is_done_once(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        fig._autoscale_all_panels()
        ax.plot([0, 1000], [0, 1000])  # data added after the window opened
        fig._autoscale_all_panels()
        assert ax.get_xlim()[1] < 100

    def test_reading_a_limit_uses_this_panels_pending_fit_not_the_figures_flag(self):
        fig, axs = gplt.subplots(1, 2)
        for a in axs:
            a.plot(X, 100 * X)
        axs[0].set_xlim(0, 3)  # turns the figure-wide flag off
        assert axs[1].get_ylim()[1] > 900  # axs[1] still reports the fit it is owed

    def test_an_explicit_autoscale_call_pins_the_view(self):
        fig, ax = gplt.subplots()
        ax.plot(X, X)
        ax.autoscale(enable=True)
        assert fig.active_panel.view_pinned


class TestBarEdgeLive:
    def test_edge_is_also_the_live_outline(self):
        fig, ax = gplt.subplots()
        (bar,) = ax.bar([0], [1], color="tab:orange", edgecolor="k", linewidth=3)
        assert bar.style.outline_enabled is True
        assert bar.style.outline_width == 3.0
        assert bar.style.outline_color[:3] == (0.0, 0.0, 0.0)

    def test_the_export_does_not_stroke_the_edge_twice(self):
        fig, ax = gplt.subplots()
        ax.bar([0], [1], color="tab:orange", edgecolor="k", linewidth=3)
        (patch,) = export(fig).axes[0].patches
        assert not patch.get_path_effects()

    def test_an_unstyled_bar_has_no_outline(self):
        fig, ax = gplt.subplots()
        (bar,) = ax.bar([0], [1])
        assert bar.style.outline_enabled is False
