"""Test the reusable ImGui widgets in glplot.gui.widgets, focused on mini_plot.

Driven through the same headless imgui harness as test_gui_mathlab.py (CONTRACT 2.10): a
real context, real frames, real draw-list calls, no OpenGL or GPU at any point. mini_plot
draws entirely with draw_list primitives rather than returning a value, so most of what can
be asserted here is "it does not raise" for every input shape it documents handling, plus a
spy on the internal color-packing helper to prove background_color/grid_color actually reach
the draw calls rather than being silently ignored.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

imgui = pytest.importorskip("imgui_bundle").imgui

from glplot.gui import widgets  # noqa: E402


@pytest.fixture
def imgui_context():
    """A headless imgui context with a font atlas but no GL (CONTRACT 2.10)."""
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = 800, 600
    io.delta_time = 1 / 60.0
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    yield io
    imgui.destroy_context(ctx)


def _in_window(fn) -> None:
    """Run ``fn()`` inside one begin/end'd window/frame, the shape every draw-list widget
    needs (a valid cursor position, a draw list, content region)."""
    imgui.new_frame()
    imgui.set_next_window_pos((0.0, 0.0))
    imgui.set_next_window_size((400.0, 300.0))
    imgui.begin("##test")
    fn()
    imgui.end()
    imgui.render()


class TestMiniPlotSmoke:
    """mini_plot must not raise for any of the input shapes its docstring promises."""

    def test_plain_series(self, imgui_context):
        y = np.sin(np.linspace(0.0, 10.0, 200))
        _in_window(lambda: widgets.mini_plot("p", y))

    def test_with_explicit_x(self, imgui_context):
        x = np.linspace(0.0, 5.0, 100)
        y = np.cos(x)
        _in_window(lambda: widgets.mini_plot("p", y, x=x))

    def test_with_overlay_and_markers(self, imgui_context):
        x = np.linspace(0.0, 5.0, 100)
        y = np.sin(x)
        overlay = np.cos(x)
        markers = (x[::10], y[::10])
        _in_window(
            lambda: widgets.mini_plot("p", y, x=x, overlay=overlay, markers=markers, label="s")
        )

    def test_empty_series(self, imgui_context):
        _in_window(lambda: widgets.mini_plot("p", np.array([])))

    def test_all_nan_series(self, imgui_context):
        y = np.full(50, np.nan)
        _in_window(lambda: widgets.mini_plot("p", y))

    def test_series_with_nan_gaps(self, imgui_context):
        y = np.sin(np.linspace(0.0, 10.0, 200))
        y[50:60] = np.nan
        _in_window(lambda: widgets.mini_plot("p", y))

    def test_mismatched_x_y_length_is_truncated_not_fatal(self, imgui_context):
        y = np.sin(np.linspace(0.0, 10.0, 200))
        x = np.linspace(0.0, 10.0, 150)
        _in_window(lambda: widgets.mini_plot("p", y, x=x))

    def test_single_point(self, imgui_context):
        _in_window(lambda: widgets.mini_plot("p", np.array([1.0]), x=np.array([0.0])))

    def test_zero_height_and_width_are_clamped_not_fatal(self, imgui_context):
        y = np.sin(np.linspace(0.0, 10.0, 50))
        _in_window(lambda: widgets.mini_plot("p", y, height=0.0))


class TestMiniPlotBackgroundGridColorDefaults:
    """background_color/grid_color are new, optional kwargs -- omitting them must be a
    no-op relative to the pre-existing behaviour (every other caller of mini_plot, e.g. the
    Functions panel's generator preview, passes neither)."""

    def test_omitting_them_does_not_raise(self, imgui_context):
        y = np.sin(np.linspace(0.0, 10.0, 50))
        _in_window(lambda: widgets.mini_plot("p", y))

    def test_none_is_equivalent_to_omitting_them(self, imgui_context):
        y = np.sin(np.linspace(0.0, 10.0, 50))
        _in_window(lambda: widgets.mini_plot("p", y, background_color=None, grid_color=None))


class TestMiniPlotBackgroundGridColorPlumbing:
    """background_color/grid_color must actually reach the packed draw colors, not just be
    accepted and dropped -- spy on the internal packing helper mini_plot calls for every
    color it draws with."""

    def _spy_u32_calls(self, monkeypatch):
        calls = []
        original = widgets._u32

        def spy(color: Any, alpha_scale: float = 1.0) -> int:
            calls.append((tuple(float(c) for c in color), alpha_scale))
            return original(color, alpha_scale)

        monkeypatch.setattr(widgets, "_u32", spy)
        return calls

    def test_custom_background_color_is_packed(self, imgui_context, monkeypatch):
        calls = self._spy_u32_calls(monkeypatch)
        y = np.sin(np.linspace(0.0, 10.0, 100))
        custom_bg = (0.9, 0.1, 0.1)
        _in_window(lambda: widgets.mini_plot("p", y, background_color=custom_bg))

        packed_colors = {color for color, _scale in calls}
        assert (0.9, 0.1, 0.1, 1.0) in packed_colors

    def test_custom_grid_color_is_packed(self, imgui_context, monkeypatch):
        calls = self._spy_u32_calls(monkeypatch)
        y = np.sin(np.linspace(0.0, 10.0, 100))  # crosses zero -> the zero-line is drawn too
        custom_grid = (0.1, 0.8, 0.1)
        _in_window(lambda: widgets.mini_plot("p", y, grid_color=custom_grid))

        packed_colors = {color for color, _scale in calls}
        assert (0.1, 0.8, 0.1, 1.0) in packed_colors

    def test_default_background_is_the_theme_token_not_the_new_default_arg(
        self, imgui_context, monkeypatch
    ):
        """Omitting background_color must still resolve to theme.get_color("panel_bg"),
        proving the new parameter did not change the no-argument code path."""
        from glplot.gui import theme

        calls = self._spy_u32_calls(monkeypatch)
        y = np.sin(np.linspace(0.0, 10.0, 100))
        _in_window(lambda: widgets.mini_plot("p", y))

        packed_colors = {color for color, _scale in calls}
        assert tuple(theme.get_color("panel_bg")) in packed_colors


class TestMiniPlotBand:
    """band=(lower, upper) fills a confidence/prediction band behind the series."""

    def test_band_draws_without_raising(self, imgui_context):
        x = np.linspace(0.0, 10.0, 100)
        y = np.sin(x)
        band = (y - 0.3, y + 0.3)
        _in_window(lambda: widgets.mini_plot("p", y, x=x, band=band))

    def test_band_with_gaps_draws_without_raising(self, imgui_context):
        x = np.linspace(0.0, 10.0, 100)
        y = np.sin(x)
        lo, hi = y - 0.3, y + 0.3
        lo = lo.copy()
        hi = hi.copy()
        lo[40:50] = np.nan
        hi[40:50] = np.nan
        _in_window(lambda: widgets.mini_plot("p", y, x=x, band=(lo, hi)))

    def test_all_nan_band_draws_without_raising(self, imgui_context):
        x = np.linspace(0.0, 10.0, 50)
        y = np.sin(x)
        nan = np.full(50, np.nan)
        _in_window(lambda: widgets.mini_plot("p", y, x=x, band=(nan, nan)))

    def test_mismatched_band_length_is_truncated_not_fatal(self, imgui_context):
        x = np.linspace(0.0, 10.0, 100)
        y = np.sin(x)
        short_lo = np.full(30, -1.0)
        short_hi = np.full(40, 1.0)
        _in_window(lambda: widgets.mini_plot("p", y, x=x, band=(short_lo, short_hi)))

    def test_no_band_omits_it_without_raising(self, imgui_context):
        x = np.linspace(0.0, 10.0, 50)
        y = np.sin(x)
        _in_window(lambda: widgets.mini_plot("p", y, x=x))

    def test_band_extends_the_autoscaled_y_range(self, imgui_context, monkeypatch):
        """A band wider than the line itself must not be clipped by autoscale -- the
        y-range union has to include it, or a wide band silently looks truncated."""
        calls = []
        real_finite_range = widgets._finite_range

        def spy(*arrays):
            result = real_finite_range(*arrays)
            calls.append(result)
            return result

        monkeypatch.setattr(widgets, "_finite_range", spy)

        x = np.linspace(0.0, 10.0, 50)
        y = np.zeros(50)
        band = (np.full(50, -100.0), np.full(50, 100.0))
        _in_window(lambda: widgets.mini_plot("p", y, x=x, band=band))

        # The y-range call (second _finite_range call: first is for x) must have folded
        # the band in, i.e. its result spans at least [-100, 100].
        y_range_calls = [c for c in calls if c is not None and c[0] <= -50.0]
        assert y_range_calls, "the band's extreme values never reached _finite_range for y"

    def test_custom_band_color_is_packed(self, imgui_context, monkeypatch):
        calls = []
        original = widgets._u32

        def spy(color: Any, alpha_scale: float = 1.0) -> int:
            calls.append((tuple(float(c) for c in color), alpha_scale))
            return original(color, alpha_scale)

        monkeypatch.setattr(widgets, "_u32", spy)

        x = np.linspace(0.0, 10.0, 100)
        y = np.sin(x)
        custom_band_color = (0.3, 0.6, 0.9)
        _in_window(
            lambda: widgets.mini_plot("p", y, x=x, band=(y - 0.3, y + 0.3), band_color=custom_band_color)
        )

        packed_colors = {color for color, _scale in calls}
        assert (0.3, 0.6, 0.9, 1.0) in packed_colors


class TestStatsTable:
    """stats_table: a real bordered two-column table for label/value rows."""

    def test_draws_without_raising(self, imgui_context):
        rows = [("Mean", "1.23"), ("Std", "0.45"), ("N", "100")]
        _in_window(lambda: widgets.stats_table("##t", ("Statistic", "Value"), rows))

    def test_empty_rows_draws_without_raising(self, imgui_context):
        _in_window(lambda: widgets.stats_table("##t", ("Statistic", "Value"), []))

    def test_rows_render_as_text(self, imgui_context, monkeypatch):
        """The label/value pairs must actually reach imgui.text/text_disabled, not just
        the table scaffolding."""
        seen: list = []
        real_text = imgui.text

        def spy_text(s, *a, **k):
            seen.append(s)
            return real_text(s, *a, **k)

        monkeypatch.setattr(widgets.imgui, "text", spy_text)

        rows = [("Mean", "1.23"), ("Std", "0.45")]
        _in_window(lambda: widgets.stats_table("##t", ("Statistic", "Value"), rows))

        assert "1.23" in seen
        assert "0.45" in seen


class TestMiniScatterSmoke:
    """mini_scatter must not raise for any input shape its docstring promises."""

    def test_plain_points(self, imgui_context):
        rng = np.random.default_rng(0)
        x, y = rng.normal(0, 1, 200), rng.normal(0, 1, 200)
        _in_window(lambda: widgets.mini_scatter("s", x, y))

    def test_with_point_colors(self, imgui_context):
        rng = np.random.default_rng(1)
        x, y = rng.normal(0, 1, 100), rng.normal(0, 1, 100)
        labels = rng.integers(0, 3, 100).astype(np.float64)
        _in_window(lambda: widgets.mini_scatter("s", x, y, point_colors=labels))

    def test_with_unclustered_label(self, imgui_context):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        labels = np.array([-1.0, 0.0, 0.0, 1.0])
        _in_window(lambda: widgets.mini_scatter("s", x, y, point_colors=labels))

    def test_with_markers(self, imgui_context):
        rng = np.random.default_rng(2)
        x, y = rng.normal(0, 1, 100), rng.normal(0, 1, 100)
        markers = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        _in_window(lambda: widgets.mini_scatter("s", x, y, markers=markers))

    def test_empty_points(self, imgui_context):
        _in_window(lambda: widgets.mini_scatter("s", np.array([]), np.array([])))

    def test_all_nan_points(self, imgui_context):
        nan = np.full(20, np.nan)
        _in_window(lambda: widgets.mini_scatter("s", nan, nan))

    def test_points_with_some_nan(self, imgui_context):
        x = np.array([0.0, np.nan, 2.0, 3.0])
        y = np.array([0.0, 1.0, np.nan, 3.0])
        _in_window(lambda: widgets.mini_scatter("s", x, y))

    def test_mismatched_length_is_truncated_not_fatal(self, imgui_context):
        x = np.linspace(0.0, 10.0, 100)
        y = np.linspace(0.0, 10.0, 50)
        _in_window(lambda: widgets.mini_scatter("s", x, y))

    def test_mismatched_point_colors_length_is_truncated_not_fatal(self, imgui_context):
        x = np.linspace(0.0, 10.0, 100)
        y = np.linspace(0.0, 10.0, 100)
        labels = np.zeros(30)
        _in_window(lambda: widgets.mini_scatter("s", x, y, point_colors=labels))

    def test_more_than_max_plot_points_is_subsampled_not_fatal(self, imgui_context):
        rng = np.random.default_rng(3)
        n = widgets.MAX_PLOT_POINTS * 3
        x, y = rng.normal(0, 1, n), rng.normal(0, 1, n)
        labels = rng.integers(0, 4, n).astype(np.float64)
        _in_window(lambda: widgets.mini_scatter("s", x, y, point_colors=labels))

    def test_single_point(self, imgui_context):
        _in_window(lambda: widgets.mini_scatter("s", np.array([1.0]), np.array([1.0])))

    def test_custom_palette(self, imgui_context):
        rng = np.random.default_rng(4)
        x, y = rng.normal(0, 1, 50), rng.normal(0, 1, 50)
        labels = rng.integers(0, 2, 50).astype(np.float64)
        palette = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0)]
        _in_window(lambda: widgets.mini_scatter("s", x, y, point_colors=labels, palette=palette))

    def test_background_and_grid_color_overrides(self, imgui_context):
        rng = np.random.default_rng(5)
        x, y = rng.normal(0, 1, 50), rng.normal(0, 1, 50)
        _in_window(
            lambda: widgets.mini_scatter(
                "s", x, y, background_color=(0.1, 0.1, 0.1), grid_color=(0.9, 0.9, 0.9)
            )
        )


class TestMiniHeatmapSmoke:
    """mini_heatmap must not raise for any grid shape its docstring promises."""

    def _grid(self, nx=10, ny=8):
        rng = np.random.default_rng(0)
        counts = rng.uniform(0.0, 10.0, (nx, ny))
        x_edges = np.linspace(0.0, 1.0, nx + 1)
        y_edges = np.linspace(0.0, 1.0, ny + 1)
        return counts, x_edges, y_edges

    def test_plain_grid(self, imgui_context):
        counts, xe, ye = self._grid()
        _in_window(lambda: widgets.mini_heatmap("h", counts, xe, ye))

    def test_with_overlay_points(self, imgui_context):
        counts, xe, ye = self._grid()
        pts = (np.array([0.1, 0.5, 0.9]), np.array([0.2, 0.4, 0.8]))
        _in_window(lambda: widgets.mini_heatmap("h", counts, xe, ye, overlay_points=pts))

    def test_with_label(self, imgui_context):
        counts, xe, ye = self._grid()
        _in_window(lambda: widgets.mini_heatmap("h", counts, xe, ye, label="density"))

    def test_zero_grid_shape(self, imgui_context):
        _in_window(
            lambda: widgets.mini_heatmap("h", np.zeros((0, 0)), np.array([0.0]), np.array([0.0]))
        )

    def test_degenerate_edges_does_not_raise(self, imgui_context):
        """x_hi <= x_lo (all points at the same x) must not divide by zero."""
        counts = np.zeros((3, 3))
        edges = np.array([1.0, 1.0, 1.0, 1.0])
        _in_window(lambda: widgets.mini_heatmap("h", counts, edges, edges))

    def test_nan_in_grid_does_not_raise(self, imgui_context):
        counts, xe, ye = self._grid()
        counts[0, 0] = np.nan
        _in_window(lambda: widgets.mini_heatmap("h", counts, xe, ye))

    def test_shape_mismatch_with_edges_falls_back_gracefully(self, imgui_context):
        """A grid whose shape does not match (len(x_edges)-1, len(y_edges)-1) must be
        treated as degenerate, not raise or index out of bounds."""
        counts = np.ones((5, 5))
        xe = np.linspace(0.0, 1.0, 4)  # implies nx=3, not 5
        ye = np.linspace(0.0, 1.0, 4)
        _in_window(lambda: widgets.mini_heatmap("h", counts, xe, ye))

    def test_unknown_colormap_falls_back_not_raising(self, imgui_context):
        counts, xe, ye = self._grid()
        _in_window(lambda: widgets.mini_heatmap("h", counts, xe, ye, cmap="not_a_real_cmap"))

    def test_background_and_grid_color_overrides(self, imgui_context):
        counts, xe, ye = self._grid()
        _in_window(
            lambda: widgets.mini_heatmap(
                "h", counts, xe, ye, background_color=(0.05, 0.05, 0.05), grid_color=(0.9, 0.9, 0.9)
            )
        )

    def test_large_grid_does_not_raise(self, imgui_context):
        """A grid at the panel's own clamp ceiling (100x100) must still draw in one frame."""
        counts, xe, ye = self._grid(nx=100, ny=100)
        _in_window(lambda: widgets.mini_heatmap("h", counts, xe, ye))

    def test_more_than_max_plot_points_overlay_is_subsampled_not_fatal(self, imgui_context):
        counts, xe, ye = self._grid()
        rng = np.random.default_rng(1)
        n = widgets.MAX_PLOT_POINTS * 2
        pts = (rng.uniform(0, 1, n), rng.uniform(0, 1, n))
        _in_window(lambda: widgets.mini_heatmap("h", counts, xe, ye, overlay_points=pts))
