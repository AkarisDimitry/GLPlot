"""Tests for the multi-panel / subplots system.

Covers the four layers that make ``plt.subplots(2, 2)`` real:
  * ``core.layout`` -- grid / mosaic / span geometry,
  * ``core.panel.Panel`` -- rectangle maths (pixels, hit-testing, local cursor),
  * the engine's panel model -- delegation, set/add/split/merge, shared-axis sync,
  * the pyplot facade -- ``subplots`` / ``subplot`` / ``subplot2grid`` / ``subplot_mosaic``
    and the ``AxesProxy`` that routes per-panel plotting.

None of these need a GL context: they exercise state and geometry, not rendering.
"""

from __future__ import annotations

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.core import layout
from glplot.core.panel import Panel
from glplot.engine import GPULinePlot
from glplot.options import EngineOptions


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


# ----------------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------------


class TestLayout:
    def test_grid_returns_one_spec_per_cell_in_row_major_order(self):
        specs = layout.grid(2, 3)
        assert len(specs) == 6
        assert [(s.row, s.col) for s in specs] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    def test_grid_top_row_sits_above_bottom_row(self):
        specs = layout.grid(2, 1)
        top = next(s for s in specs if s.row == 0)
        bottom = next(s for s in specs if s.row == 1)
        # Bottom-left origin: the top row's y0 is greater than the bottom row's.
        assert top.rect_frac[1] > bottom.rect_frac[1]

    def test_grid_cells_stay_inside_the_unit_square(self):
        for s in layout.grid(3, 3):
            x0, y0, w, h = s.rect_frac
            assert 0.0 <= x0 and 0.0 <= y0
            assert x0 + w <= 1.0 + 1e-9
            assert y0 + h <= 1.0 + 1e-9

    def test_grid_cells_do_not_overlap(self):
        specs = layout.grid(2, 2)
        # Two cells in the same row must not overlap horizontally.
        r0 = sorted((s for s in specs if s.row == 0), key=lambda s: s.col)
        left, right = r0[0].rect_frac, r0[1].rect_frac
        assert left[0] + left[2] <= right[0] + 1e-9

    def test_mosaic_string_and_list_agree(self):
        from_str = layout.mosaic([["A", "B"], ["C", "D"]])
        assert {s.name for s in from_str} == {"A", "B", "C", "D"}

    def test_mosaic_spanning_label_covers_its_block(self):
        specs = layout.mosaic([["A", "A"], ["B", "C"]])
        a = next(s for s in specs if s.name == "A")
        b = next(s for s in specs if s.name == "B")
        # A spans the whole top row, so it is wider than a single bottom cell.
        assert a.rect_frac[2] > b.rect_frac[2]
        assert a.colspan == 2 and a.rowspan == 1

    def test_mosaic_dot_is_an_empty_cell(self):
        specs = layout.mosaic([["A", "."], [".", "B"]])
        assert {s.name for s in specs} == {"A", "B"}

    def test_grid_span_covers_multiple_cells(self):
        spec = layout.grid_span(3, 3, (0, 0), rowspan=1, colspan=3)
        # Spanning all three columns of the top row is ~full width.
        assert spec.rect_frac[2] > 0.8
        assert spec.colspan == 3


# ----------------------------------------------------------------------------
# Panel geometry
# ----------------------------------------------------------------------------


class TestPanelGeometry:
    def test_full_panel_pixel_rect_is_the_whole_framebuffer(self):
        p = Panel(EngineOptions(), rect_frac=(0.0, 0.0, 1.0, 1.0))
        assert p.pixel_rect(800, 600) == (0, 0, 800, 600)

    def test_quadrant_pixel_rect(self):
        # Top-right quadrant in bottom-left-origin fractions.
        p = Panel(EngineOptions(), rect_frac=(0.5, 0.5, 0.5, 0.5))
        assert p.pixel_rect(800, 600) == (400, 300, 400, 300)

    def test_contains_window_px_flips_y(self):
        # Top-left quadrant: rect (0, 0.5, 0.5, 0.5) in bottom-left origin.
        p = Panel(EngineOptions(), rect_frac=(0.0, 0.5, 0.5, 0.5))
        # A cursor near the window top-left (small y in GLFW top-left origin) is inside it.
        assert p.contains_window_px(100, 50, 800, 600)
        # A cursor near the window bottom is not.
        assert not p.contains_window_px(100, 550, 800, 600)

    def test_local_cursor_offsets_by_the_panel_origin(self):
        # Top-right quadrant.
        p = Panel(EngineOptions(), rect_frac=(0.5, 0.5, 0.5, 0.5))
        lx, ly = p.local_cursor(500, 100, 800, 600)
        # Panel left edge is at x=400, top edge at y=0 (it is the top half).
        assert lx == pytest.approx(100.0)
        assert ly == pytest.approx(100.0)

    def test_pixel_size(self):
        p = Panel(EngineOptions(), rect_frac=(0.25, 0.25, 0.5, 0.5))
        assert p.pixel_size(800, 600) == (400, 300)


# ----------------------------------------------------------------------------
# Engine panel model
# ----------------------------------------------------------------------------


class TestEnginePanels:
    def test_default_engine_has_one_full_panel(self):
        e = GPULinePlot()
        assert len(e.panels) == 1
        assert e.panels[0].rect_frac == (0.0, 0.0, 1.0, 1.0)

    def test_state_delegates_to_active_panel(self):
        e = GPULinePlot()
        e.set_panels(layout.grid(1, 2))
        e.active_panel_index = 0
        assert e.scene is e.panels[0].scene
        assert e.camera is e.panels[0].camera
        assert e.camera_controller is e.panels[0].camera_controller
        e.active_panel_index = 1
        assert e.scene is e.panels[1].scene
        assert e.camera_controller.camera is e.camera

    def test_set_panels_makes_fresh_empty_panels(self):
        e = GPULinePlot()
        e.set_panels(layout.grid(2, 2))
        assert len(e.panels) == 4
        assert all(len(p.scene.layers) == 0 for p in e.panels)
        assert e.active_panel_index == 0

    def test_add_panel_appends_and_activates(self):
        e = GPULinePlot()
        spec = layout.grid_span(2, 2, (0, 0))
        e.add_panel(spec)
        assert len(e.panels) == 2
        assert e.active_panel_index == 1

    def test_split_view_keeps_active_panel_content(self):
        e = GPULinePlot()
        marker = object()
        e.panels[0].scene.layers.append(marker)  # type: ignore[arg-type]
        e.split_view(2, 2)
        assert len(e.panels) == 4
        assert marker in e.panels[0].scene.layers
        assert all(len(p.scene.layers) == 0 for p in e.panels[1:])

    def test_merge_view_collapses_to_full_window(self):
        e = GPULinePlot()
        e.split_view(2, 2)
        e.active_panel_index = 2
        marker = object()
        e.panels[2].scene.layers.append(marker)  # type: ignore[arg-type]
        e.merge_view()
        assert len(e.panels) == 1
        assert e.panels[0].rect_frac == (0.0, 0.0, 1.0, 1.0)
        assert marker in e.panels[0].scene.layers

    def test_panel_index_at_maps_quadrants(self):
        e = GPULinePlot(width=1000, height=800)
        e.width, e.height = 1000, 800
        e.set_panels(layout.grid(2, 2))
        assert e._panel_index_at(100, 100) == 0  # top-left
        assert e._panel_index_at(900, 100) == 1  # top-right
        assert e._panel_index_at(100, 700) == 2  # bottom-left
        assert e._panel_index_at(900, 700) == 3  # bottom-right

    def test_panel_index_at_returns_none_in_gutter(self):
        e = GPULinePlot(width=1000, height=800)
        e.width, e.height = 1000, 800
        # An explicit gap leaves dead space between panels; the default is now edge-to-edge.
        e.set_panels(layout.grid(2, 2, wspace=0.1, hspace=0.1, outer=0.1))
        assert e._panel_index_at(500, 400) is None


class TestSharedAxes:
    def _linked(self, sharex, sharey):
        fig, axs = gplt.subplots(1, 2, sharex=sharex, sharey=sharey)
        return fig

    def test_sharex_propagates_x_only(self):
        fig = self._linked(sharex=True, sharey=False)
        p0, p1 = fig.panels
        fig.active_panel_index = 0
        p0.camera.cx, p0.camera.zoom_x = 42.0, 3.0
        p0.camera.cy, p0.camera.zoom_y = 5.0, 7.0
        fig._sync_shared_axes()
        assert p1.camera.cx == 42.0
        assert p1.camera.zoom_x == 3.0
        assert p1.camera.cy != 5.0  # y is independent

    def test_sharey_propagates_y_only(self):
        fig = self._linked(sharex=False, sharey=True)
        p0, p1 = fig.panels
        fig.active_panel_index = 0
        p0.camera.cy, p0.camera.zoom_y = 9.0, 4.0
        p0.camera.cx = 1.0
        fig._sync_shared_axes()
        assert p1.camera.cy == 9.0
        assert p1.camera.zoom_y == 4.0
        assert p1.camera.cx != 1.0

    def test_independent_by_default(self):
        fig, axs = gplt.subplots(1, 2)
        assert fig.panels[0].sharex_group is None
        assert fig.panels[0].sharey_group is None


# ----------------------------------------------------------------------------
# pyplot facade
# ----------------------------------------------------------------------------


class TestSubplotsFacade:
    def test_subplots_1x1_returns_single_proxy(self):
        fig, ax = gplt.subplots()
        assert isinstance(ax, gplt.AxesProxy)
        assert len(fig.panels) == 1

    def test_subplots_grid_returns_object_array(self):
        fig, axs = gplt.subplots(2, 3)
        assert axs.shape == (2, 3)
        assert len(fig.panels) == 6
        assert axs[0, 0] is not axs[1, 2]

    def test_subplots_single_row_squeezes_to_1d(self):
        fig, axs = gplt.subplots(1, 3)
        assert axs.shape == (3,)

    def test_axes_plot_targets_its_own_panel(self):
        fig, axs = gplt.subplots(1, 2)
        axs[0].plot([0.0, 1.0], [0.0, 1.0])
        fig.active_panel_index = 0
        assert any(ly.layer_type == "polyline" for ly in fig.scene.layers)
        fig.active_panel_index = 1
        assert not any(ly.layer_type == "polyline" for ly in fig.scene.layers)

    def test_axes_set_xlim_aliases_to_module_xlim(self):
        fig, ax = gplt.subplots()
        ax.plot([0.0, 10.0], [0.0, 1.0])
        ax.set_xlim(2.0, 8.0)
        lo, hi = ax.get_xlim()
        assert lo == pytest.approx(2.0, abs=0.5)
        assert hi == pytest.approx(8.0, abs=0.5)

    def test_subplot_reuses_grid_across_calls(self):
        a1 = gplt.subplot(2, 1, 1)
        a2 = gplt.subplot(2, 1, 2)
        assert a1 is not a2
        assert len(gplt.gcf().panels) == 2  # not rebuilt on the second call

    def test_subplot_rebuilds_on_shape_change(self):
        gplt.subplot(2, 1, 1)
        gplt.subplot(2, 2, 1)
        assert len(gplt.gcf().panels) == 4

    def test_subplot2grid_accumulates_spanning_panels(self):
        gplt.subplot2grid((3, 3), (0, 0), colspan=3)
        gplt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
        fig = gplt.gcf()
        assert len(fig.panels) == 2
        assert (fig.panels[0].rowspan, fig.panels[0].colspan) == (1, 3)
        assert (fig.panels[1].rowspan, fig.panels[1].colspan) == (2, 2)

    def test_subplot_mosaic_makes_a_panel_per_name(self):
        fig, axd = gplt.subplot_mosaic("AB;CD")
        assert set(axd) == {"A", "B", "C", "D"}
        assert len({id(ax) for ax in axd.values()}) == 4
        assert len(fig.panels) == 4

    def test_proxy_figure_and_panel_accessors(self):
        fig, axs = gplt.subplots(1, 2)
        assert axs[0].figure is fig
        assert axs[1].panel is fig.panels[1]


class TestInsetAxes:
    """``ax.inset_axes(...)`` used to be entirely missing (AttributeError)."""

    def test_inset_axes_adds_a_panel_scaled_into_the_parent_box(self):
        fig, ax = gplt.subplots()
        inset = ax.inset_axes([0.2, 0.3, 0.4, 0.5])
        assert isinstance(inset, gplt.AxesProxy)
        assert len(fig.panels) == 2
        # The parent panel is the full window (0, 0, 1, 1), so the inset's rect is the
        # bounds unchanged; this is the case that would hide a broken offset/scale.
        assert inset.panel.rect_frac == pytest.approx((0.2, 0.3, 0.4, 0.5))

    def test_inset_axes_rect_is_relative_to_its_own_parent_not_the_figure(self):
        fig, axs = gplt.subplots(1, 2)
        parent = axs[1]
        px0, py0, pw, ph = parent.panel.rect_frac
        inset = parent.inset_axes([0.5, 0.0, 0.5, 1.0])
        expected = (px0 + 0.5 * pw, py0, 0.5 * pw, ph)
        assert inset.panel.rect_frac == pytest.approx(expected)

    def test_inset_axes_leaves_the_parent_as_the_current_axes(self):
        fig, ax = gplt.subplots()
        ax.inset_axes([0.1, 0.1, 0.2, 0.2])
        assert gplt.gca().panel is ax.panel

    def test_inset_axes_plot_targets_the_inset_not_the_parent(self):
        fig, ax = gplt.subplots()
        inset = ax.inset_axes([0.1, 0.1, 0.2, 0.2])
        inset.plot([0.0, 1.0], [0.0, 1.0])
        assert not any(ly.layer_type == "polyline" for ly in ax.panel.scene.layers)
        assert any(ly.layer_type == "polyline" for ly in inset.panel.scene.layers)


class TestSpinesStub:
    """``ax.spines[...]`` setters must never raise, even the ones added after the fact."""

    def test_set_edgecolor_and_linewidth_do_not_raise(self):
        fig, ax = gplt.subplots()
        for spine in ax.spines.values():
            spine.set_edgecolor("tab:blue")
            spine.set_linewidth(2)


class TestStylePanelLayout:
    """The Style panel's Scene-tab Layout section splits/merges the window into panels."""

    def _panel(self):
        plot = GPULinePlot()
        plot.options.enable_hud = True
        ws = plot.hud.workspace
        return plot, ws, ws.panels["style"]

    def test_split_from_style_panel(self):
        plot, ws, style = self._panel()
        style._apply_layout(2, 2)
        ws.queue.drain(plot)
        assert len(plot.panels) == 4

    def test_merge_from_style_panel_keeps_content(self):
        plot, ws, style = self._panel()
        marker = object()
        plot.panels[0].scene.layers.append(marker)  # type: ignore[arg-type]
        style._apply_layout(2, 3)
        ws.queue.drain(plot)
        assert len(plot.panels) == 6
        style._apply_layout(1, 1)  # "Single" merges back
        ws.queue.drain(plot)
        assert len(plot.panels) == 1
        assert marker in plot.panels[0].scene.layers


class TestStylePanelAxisLinking:
    """The Layout section's Link X / Link Y controls share axes across panels."""

    def _panel(self):
        plot = GPULinePlot()
        plot.options.enable_hud = True
        ws = plot.hud.workspace
        return plot, ws, ws.panels["style"]

    def test_split_with_link_x_links_only_x(self):
        plot, ws, style = self._panel()
        style._layout_sharex, style._layout_sharey = True, False
        style._apply_layout(1, 2)
        ws.queue.drain(plot)
        assert plot.panels[0].sharex_group is not None
        assert plot.panels[0].sharey_group is None

    def test_toggle_sharing_on_existing_split(self):
        plot, ws, style = self._panel()
        style._apply_layout(1, 2)
        ws.queue.drain(plot)
        style._layout_sharey = True
        style._apply_sharing()
        ws.queue.drain(plot)
        assert plot.panels[0].sharey_group is not None

    def test_set_shared_axes_propagates_on_sync(self):
        plot = GPULinePlot()
        plot.set_panels(layout.grid(1, 2))
        plot.set_shared_axes(True, False)
        plot.active_panel_index = 0
        plot.panels[0].camera.cx = 12.0
        plot.panels[0].camera.cy = 3.0
        plot._sync_shared_axes()
        assert plot.panels[1].camera.cx == 12.0  # x shared
        assert plot.panels[1].camera.cy != 3.0  # y independent

    def test_single_panel_has_no_groups(self):
        plot = GPULinePlot()
        plot.set_shared_axes(True, True)  # nothing to share with one panel
        assert plot.panels[0].sharex_group is None
        assert plot.panels[0].sharey_group is None


class TestLayoutCompactness:
    """Grid spacing controls: retile_current reflows without losing content."""

    def test_retile_reflows_without_losing_content(self):
        plot = GPULinePlot()
        plot.set_panels(layout.grid(2, 2))
        for i, p in enumerate(plot.panels):
            p.scene.layers.append(("m", i))
        ok = plot.retile_current(0.0, 0.0, 0.0)
        assert ok is True
        # gap=0, outer=0 => panels fill the unit square edge to edge.
        assert plot.panels[0].rect_frac == (0.0, 0.5, 0.5, 0.5)
        assert all(p.scene.layers[-1] == ("m", i) for i, p in enumerate(plot.panels))

    def test_retile_single_panel_is_noop(self):
        plot = GPULinePlot()
        assert plot.retile_current(0.0, 0.0, 0.0) is False

    def test_retile_skips_mosaic(self):
        plot = GPULinePlot()
        plot.set_panels(layout.mosaic([["A", "A"], ["B", "C"]]))  # spanning cells
        assert plot.retile_current(0.0, 0.0, 0.0) is False

    def test_split_view_accepts_spacing(self):
        plot = GPULinePlot()
        plot.split_view(1, 2, wspace=0.0, outer=0.0)
        assert plot.panels[0].rect_frac[0] == 0.0
        assert plot.panels[1].rect_frac[0] == pytest.approx(0.5)
