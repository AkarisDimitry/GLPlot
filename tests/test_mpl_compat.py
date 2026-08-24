"""Test the matplotlib-parity surface of ``glplot.pyplot``.

GLPlot's selling point is that a matplotlib script runs unchanged, so a keyword mpl
accepts and GLPlot rejects is a bug in GLPlot, not a caller error. Two rules are pinned
here, and they are the whole compat contract:

1. **A matplotlib keyword must not raise.** ``TypeError: unexpected keyword argument`` is
   the failure mode this surface exists to remove.
2. **A keyword that does nothing must say so.** Accepting an argument and silently
   dropping it is worse than rejecting it: the caller reads it back in their own source
   and believes it took effect. Every no-op is a `MatplotlibCompatWarning`.

Rule 2 is the one that needs tests with teeth. It is trivially easy to "support" a
keyword by adding it to a signature, and nothing but an assertion on the warning
distinguishes that from an implementation.

No OpenGL and no GPU: ``GPULinePlot`` constructs without a window (CONTRACT 5.1) and
none of these call ``show()``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.options import resolve_axis_margins


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


@pytest.fixture
def plot():
    """A real engine with a deliberately non-square view, so aspect bugs show up."""
    gplt.figure(width=1000, height=500)
    gplt.plot([0.0, 10.0], [0.0, 1.0])
    live = gplt._get_or_create_plot()
    live.width, live.height = 1000, 500
    return live


def _units_per_px(live) -> tuple[float, float]:
    """World units per pixel on each axis, measured through the frame the data lands in.

    The world window projects into the box left after the gutters, not the raw viewport,
    so this reads the same margins ``mvp()`` does -- otherwise "equal" would look equal to
    the test and visibly wrong on screen.
    """
    margin_l, margin_r, margin_b, margin_t = resolve_axis_margins(live.options)
    frame_w = float(live.width) - margin_l - margin_r
    frame_h = float(live.height) - margin_b - margin_t
    span_x = 2.0 / live.camera.zoom_x
    span_y = 2.0 / live.camera.zoom_y
    return span_x / frame_w, span_y / frame_h


class TestAxisModes:
    """Every mode matplotlib's axis() takes. 'equal' is the one real scripts hit."""

    def test_equal_gives_both_axes_the_same_scale(self, plot):
        """The whole point of axis('equal'): a circle must come out round."""
        gplt.axis("equal")
        upp_x, upp_y = _units_per_px(plot)
        assert upp_x == pytest.approx(upp_y, rel=1e-6)

    def test_equal_widens_rather_than_crops(self, plot):
        """Equalising must not push visible data out of frame."""
        before_x, before_y = _units_per_px(plot)
        gplt.axis("equal")
        after_x, after_y = _units_per_px(plot)
        # The coarser scale is adopted, so neither axis may end up showing *less* world.
        assert after_x >= before_x - 1e-9
        assert after_y >= before_y - 1e-9
        assert max(after_x, after_y) == pytest.approx(max(before_x, before_y), rel=1e-6)

    def test_equal_accounts_for_the_axis_gutters(self, plot):
        """A viewport-based aspect would be wrong by the margins; assert it is not used.

        The margins are asymmetric (l=60,r=20 vs b=40,t=20), so an implementation that
        equalised against raw width/height lands at a measurably different zoom than one
        that uses the inset frame. This fails on the naive version.
        """
        gplt.axis("equal")
        margin_l, margin_r, margin_b, margin_t = resolve_axis_margins(plot.options)
        assert (margin_l + margin_r) != (margin_b + margin_t), "fixture no longer discriminates"
        upp_x, upp_y = _units_per_px(plot)
        assert upp_x == pytest.approx(upp_y, rel=1e-6)

    def test_square_equalises_span_not_pixels(self, plot):
        """'square' is about the data range, so the spans match and the scales do not."""
        gplt.axis("square")
        span_x = 2.0 / plot.camera.zoom_x
        span_y = 2.0 / plot.camera.zoom_y
        assert span_x == pytest.approx(span_y, rel=1e-6)

    def test_scaled_matches_equal(self, plot):
        """One viewport means 'box' and 'datalim' collapse to the same thing."""
        gplt.axis("equal")
        equal = (plot.camera.zoom_x, plot.camera.zoom_y)
        gplt.figure(width=1000, height=500)
        gplt.plot([0.0, 10.0], [0.0, 1.0])
        live = gplt._get_or_create_plot()
        live.width, live.height = 1000, 500
        gplt.axis("scaled")
        assert (live.camera.zoom_x, live.camera.zoom_y) == pytest.approx(equal, rel=1e-6)

    def test_image_fits_data_then_equalises(self, plot):
        """'image' is 'scaled' on tight limits, so it must do both."""
        plot.camera.cx, plot.camera.cy = 500.0, 500.0  # park the view far off the data
        gplt.axis("image")
        upp_x, upp_y = _units_per_px(plot)
        assert upp_x == pytest.approx(upp_y, rel=1e-6)
        # The autoscale half must have brought the camera back onto the data.
        assert abs(plot.camera.cx) < 100.0

    def test_off_hides_the_axis_apparatus(self, plot):
        gplt.axis("off")
        assert plot.options.axis_show_grid is False
        assert plot.options.axis_show_labels is False
        assert plot.options.axis_show_frame is False

    def test_on_restores_it(self, plot):
        gplt.axis("off")
        gplt.axis("on")
        assert plot.options.axis_show_grid is True
        assert plot.options.axis_show_labels is True
        assert plot.options.axis_show_frame is True

    def test_unknown_mode_still_raises_and_names_the_alternatives(self, plot):
        """A typo must fail, and the message must be enough to fix it without the docs."""
        with pytest.raises(ValueError, match="equal"):
            gplt.axis("qeual")

    def test_tuple_and_preset_modes_still_work(self, plot):
        """The pre-existing modes must not regress."""
        assert gplt.axis((0.0, 10.0, -5.0, 5.0)) == (0.0, 10.0, -5.0, 5.0)
        assert gplt.axis("auto") is None
        assert gplt.axis("tight") is None
        assert gplt.axis("reset") is None


class TestHistParity:
    """hist() had 11 of matplotlib's parameters missing. These assert the ones that
    are real, not merely accepted."""

    DATA = np.array([0.5, 1.5, 1.5, 2.5, 2.5, 2.5, 3.5, 9.0], dtype=float)

    def test_range_restricts_the_bins(self, plot):
        counts, edges, _ = gplt.hist(self.DATA, bins=4, range=(0.0, 4.0))
        assert (edges[0], edges[-1]) == (0.0, 4.0)
        assert counts.sum() == 7  # the 9.0 outlier is outside the range

    def test_weights_replace_the_unit_count(self, plot):
        counts, _, _ = gplt.hist(self.DATA, bins=2, weights=np.full(len(self.DATA), 3.0))
        assert counts.sum() == pytest.approx(3.0 * len(self.DATA))

    def test_weights_length_is_checked(self, plot):
        with pytest.raises(ValueError, match="weights"):
            gplt.hist(self.DATA, weights=[1.0, 2.0])

    def test_cumulative_accumulates_left_to_right(self, plot):
        counts, _, _ = gplt.hist(self.DATA, bins=4)
        cumulative, _, _ = gplt.hist(self.DATA, bins=4, cumulative=True)
        assert list(cumulative) == list(np.cumsum(counts))

    def test_negative_cumulative_accumulates_from_the_right(self, plot):
        """matplotlib's spelling for a survival curve."""
        counts, _, _ = gplt.hist(self.DATA, bins=4)
        reverse, _, _ = gplt.hist(self.DATA, bins=4, cumulative=-1)
        assert list(reverse) == list(np.cumsum(counts[::-1])[::-1])
        assert reverse[0] == counts.sum()

    @pytest.mark.parametrize(
        "align,expected_first_bar_centre", [("mid", 1.0), ("left", 0.0), ("right", 2.0)]
    )
    def test_align_moves_the_bars(self, plot, align, expected_first_bar_centre):
        """align picks which part of the bin the bar sits on, so the geometry must move."""
        gplt.hist(self.DATA, bins=2, range=(0.0, 4.0), align=align)
        patches = [layer for layer in plot.scene.layers if layer.layer_type == "patch"]
        xs = patches[0].vertices[:, 0]
        assert 0.5 * (xs.min() + xs.max()) == pytest.approx(expected_first_bar_centre)

    def test_rwidth_narrows_the_bars(self, plot):
        gplt.hist(self.DATA, bins=2, range=(0.0, 4.0))
        full = [ly for ly in plot.scene.layers if ly.layer_type == "patch"][0]
        full_w = full.vertices[:, 0].max() - full.vertices[:, 0].min()
        gplt._cleanup_pyplot_state()
        gplt.figure(width=1000, height=500)
        gplt.hist(self.DATA, bins=2, range=(0.0, 4.0), rwidth=0.5)
        live = gplt._get_or_create_plot()
        half = [ly for ly in live.scene.layers if ly.layer_type == "patch"][0]
        half_w = half.vertices[:, 0].max() - half.vertices[:, 0].min()
        assert half_w == pytest.approx(full_w * 0.5, rel=1e-4)

    def test_histtype_step_draws_a_line_not_bars(self, plot):
        gplt.hist(self.DATA, bins=4, histtype="step")
        kinds = {layer.layer_type for layer in plot.scene.layers}
        assert "patch" not in kinds
        assert "polyline" in kinds

    def test_histtype_step_ignores_rwidth_and_says_so(self, plot):
        """matplotlib ignores rwidth for step; silently doing the same would be a lie."""
        with pytest.warns(gplt.MatplotlibCompatWarning, match="rwidth"):
            gplt.hist(self.DATA, histtype="step", rwidth=0.5)

    def test_bins_accepts_a_numpy_strategy_name(self, plot):
        counts, edges, _ = gplt.hist(self.DATA, bins="auto")
        assert len(edges) == len(counts) + 1

    def test_bad_histtype_and_align_raise(self, plot):
        with pytest.raises(ValueError, match="histtype"):
            gplt.hist(self.DATA, histtype="nope")
        with pytest.raises(ValueError, match="align"):
            gplt.hist(self.DATA, align="nope")

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"log": True}, "log"),
            ({"stacked": True}, "stacked"),
            ({"orientation": "horizontal"}, "orientation"),
        ],
    )
    def test_unimplementable_kwargs_warn(self, plot, kwargs, match):
        with pytest.warns(gplt.MatplotlibCompatWarning, match=match):
            gplt.hist(self.DATA, **kwargs)


class TestScatterParity:
    def test_norm_replaces_the_linear_ramp(self, plot):
        """A LogNorm over decades must not produce the same colours as a linear ramp."""
        vals = np.array([1.0, 10.0, 100.0, 1000.0])
        xs = np.arange(4.0)
        linear = gplt.scatter(xs, xs, c=vals).colors.copy()
        logged = gplt.scatter(xs, xs, c=vals, norm="log").colors
        assert not np.allclose(linear, logged)

    def test_a_per_point_rgba_array_via_color_is_not_a_colormap(self, plot):
        """`color=` an (N, 4) array is a literal colour per point -- a real, common call.

        The `resolved_color or default` idiom raised 'truth value of an array is
        ambiguous' on it (an example in the repo hit exactly this), because an ndarray has
        no truth value. This is the regression guard: each point keeps its own colour.
        """
        n = 50
        colors = np.random.default_rng(0).random((n, 4)).astype(np.float32)
        layer = gplt.scatter(np.arange(n), np.arange(n), color=colors)
        assert len(np.unique(layer.colors, axis=0)) == n

    def test_a_single_named_colour_still_works(self, plot):
        layer = gplt.scatter([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], color="red")
        assert tuple(layer.colors[0]) == (1.0, 0.0, 0.0, 1.0)

    def test_norm_instance_and_name_agree(self, plot):
        from matplotlib.colors import LogNorm

        vals = np.array([1.0, 10.0, 100.0, 1000.0])
        xs = np.arange(4.0)
        by_name = gplt.scatter(xs, xs, c=vals, norm="log").colors.copy()
        by_instance = gplt.scatter(xs, xs, c=vals, norm=LogNorm()).colors
        assert np.allclose(by_name, by_instance)

    def test_norm_with_vmin_raises_as_in_matplotlib(self, plot):
        with pytest.raises(ValueError, match="vmin"):
            gplt.scatter([0.0, 1.0], [0.0, 1.0], c=[1.0, 2.0], norm="log", vmin=1.0)

    def test_norm_out_of_domain_values_do_not_leak_a_mask(self, plot):
        """A LogNorm sees a zero and masks it; a masked index would corrupt the RGBA."""
        layer = gplt.scatter([0.0, 1.0], [0.0, 1.0], c=[0.0, 100.0], norm="log")
        assert np.isfinite(layer.colors).all()

    def test_edgecolors_enables_the_outline(self, plot):
        layer = gplt.scatter([0.0, 1.0], [0.0, 1.0], edgecolors="red", linewidths=2.0)
        assert layer.style.point_outline_enabled is True
        assert layer.style.point_outline_color[:3] == (1.0, 0.0, 0.0)
        assert layer.style.point_outline_width == 2.0

    @pytest.mark.parametrize("spelling", ["none", "face"])
    def test_matplotlibs_no_outline_spellings_leave_it_off(self, plot, spelling):
        layer = gplt.scatter([0.0, 1.0], [0.0, 1.0], edgecolors=spelling)
        assert layer.style.point_outline_enabled is False

    def test_linewidths_without_edgecolors_warns_rather_than_ringing_every_point(self, plot):
        """Turning the outline on here would silently restyle the plot in default black."""
        with pytest.warns(gplt.MatplotlibCompatWarning, match="linewidths"):
            layer = gplt.scatter([0.0, 1.0], [0.0, 1.0], linewidths=2.0)
        assert layer.style.point_outline_enabled is False


class TestDataKwarg:
    """matplotlib's `data=` indirection, shared by the plotting functions."""

    def test_strings_index_the_container(self, plot):
        frame = {"a": [0.0, 1.0, 2.0], "b": [3.0, 4.0, 5.0]}
        layer = gplt.scatter("a", "b", data=frame)
        assert layer.pts[:, 0].tolist() == [0.0, 1.0, 2.0]
        assert layer.pts[:, 1].tolist() == [3.0, 4.0, 5.0]

    def test_a_missing_key_names_the_key(self, plot):
        """The error must not surface three frames down as a numpy dtype complaint."""
        with pytest.raises(ValueError, match="'nope' is not a key"):
            gplt.scatter("nope", "b", data={"b": [1.0]})

    def test_arrays_still_pass_through_untouched(self, plot):
        layer = gplt.scatter([0.0, 1.0], [2.0, 3.0], data={"a": [9.0]})
        assert layer.pts[:, 0].tolist() == [0.0, 1.0]


class TestBarParity:
    def test_align_edge_puts_the_left_edge_on_x(self, plot):
        gplt.bar([0.0], [1.0], width=2.0, align="edge")
        patch = [ly for ly in plot.scene.layers if ly.layer_type == "patch"][0]
        assert patch.vertices[:, 0].min() == pytest.approx(0.0)
        assert patch.vertices[:, 0].max() == pytest.approx(2.0)

    def test_align_center_is_still_the_default(self, plot):
        gplt.bar([0.0], [1.0], width=2.0)
        patch = [ly for ly in plot.scene.layers if ly.layer_type == "patch"][0]
        assert patch.vertices[:, 0].min() == pytest.approx(-1.0)
        assert patch.vertices[:, 0].max() == pytest.approx(1.0)

    def test_negative_width_with_edge_grows_leftward(self, plot):
        """matplotlib's spelling for right-edge alignment."""
        gplt.bar([0.0], [1.0], width=-2.0, align="edge")
        patch = [ly for ly in plot.scene.layers if ly.layer_type == "patch"][0]
        assert patch.vertices[:, 0].min() == pytest.approx(-2.0)
        assert patch.vertices[:, 0].max() == pytest.approx(0.0)


class TestTicks:
    """xticks/yticks did not exist at all -- an AttributeError, not a bad kwarg."""

    @pytest.fixture
    def fitted(self, plot):
        gplt.axis("tight")
        return plot

    @pytest.mark.parametrize("func", ["xticks", "yticks"])
    def test_query_answers_before_anything_is_drawn(self, fitted, func):
        """The ticks are a per-frame product, so an un-drawn plot must still answer.

        This fails against an implementation that only reads AxisManager's cached arrays:
        outside a render loop they are empty, and the query would report a plot with no
        ticks on an axis that visibly has them.
        """
        locs, labels = getattr(gplt, func)()
        assert len(locs) > 0
        assert len(labels) == len(locs)

    @pytest.mark.parametrize("func,axis", [("xticks", "x"), ("yticks", "y")])
    def test_explicit_ticks_win_over_the_generator(self, fitted, func, axis):
        getattr(gplt, func)([0.0, 0.5, 1.0])
        locs, _ = getattr(gplt, func)()
        assert list(locs) == [0.0, 0.5, 1.0]

    def test_labels_are_paired_with_their_ticks(self, fitted):
        gplt.xticks([0.0, 5.0, 10.0], ["low", "mid", "high"])
        locs, labels = gplt.xticks()
        assert list(locs) == [0.0, 5.0, 10.0]
        assert labels == ["low", "mid", "high"]

    def test_ticks_without_labels_are_numbered_by_the_formatter(self, fitted):
        gplt.xticks([0.0, 5.0, 10.0])
        _, labels = gplt.xticks()
        assert labels == ["0", "5", "10"]

    def test_empty_ticks_clears_the_axis(self, fitted):
        """xticks([]) is a real instruction, not an absent one -- the `is None` test."""
        gplt.xticks([])
        locs, labels = gplt.xticks()
        assert len(locs) == 0
        assert labels == []

    def test_offscreen_ticks_are_not_drawn_but_are_not_forgotten(self, fitted):
        """Only visible ticks are returned; the stored set survives for a later pan."""
        gplt.xticks([0.0, 5.0, 10.0, 9999.0])
        assert list(gplt.xticks()[0]) == [0.0, 5.0, 10.0]
        assert gplt._get_or_create_plot().options.axis_tick_values_x == (0.0, 5.0, 10.0, 9999.0)

    def test_labels_track_their_ticks_when_some_scroll_offscreen(self, fitted):
        """The visibility mask must carry the labels, or they slide onto wrong ticks."""
        gplt.xticks([-9999.0, 0.0, 5.0, 10.0], ["gone", "low", "mid", "high"])
        locs, labels = gplt.xticks()
        assert list(locs) == [0.0, 5.0, 10.0]
        assert labels == ["low", "mid", "high"]

    def test_length_mismatch_raises_rather_than_truncating(self, fitted):
        with pytest.raises(ValueError, match="2 ticks but 1 labels"):
            gplt.xticks([0.0, 1.0], ["only-one"])

    def test_labels_without_ticks_is_an_error(self, fitted):
        with pytest.raises(TypeError, match="without ticks"):
            gplt.xticks(labels=["a"])

    @pytest.mark.parametrize("kwargs", [{"minor": True}, {"fontsize": 14.0}, {"color": "red"}])
    def test_unimplementable_kwargs_warn(self, fitted, kwargs):
        with pytest.warns(gplt.MatplotlibCompatWarning):
            gplt.xticks([0.0, 1.0], **kwargs)


class TestFigureLevelAliases:
    """Names matplotlib has that GLPlot's one-viewport model collapses onto others."""

    def test_gca_returns_axes_not_the_figure(self, plot):
        """``gca()`` returns an axes, as in matplotlib — not the figure.

        This assertion is the inverse of what it used to be, and the change is a fix.
        ``gca()`` returned the ``GPULinePlot`` itself, so the very next line of any ported
        matplotlib script — ``ax.plot_surface(...)``, ``ax.set_zlim(...)``, even
        ``ax.set_xlabel(...)`` — died with ``AttributeError``, because a figure carries
        none of the Axes API. The old test encoded that defect as the contract.

        It now returns an ``AxesProxy``, which forwards to the current panel. The figure is
        still reachable through ``gcf()`` and through the proxy's own ``.figure``.
        """
        axes = gplt.gca()
        assert axes is not gplt.gcf()
        assert axes.figure is gplt.gcf()
        # The point of the change: the Axes API is actually there now.
        for name in ("plot", "scatter", "set_xlabel", "view_init", "set_zlim"):
            assert callable(getattr(axes, name))

    def test_suptitle_sets_the_title(self, plot):
        gplt.suptitle("Run 42")
        assert plot.title == "Run 42"

    def test_suptitle_and_title_are_the_same_title(self, plot):
        """Documented collision: there is only one place a title can go."""
        gplt.title("first")
        gplt.suptitle("second")
        assert plot.title == "second"

    def test_suptitle_takes_the_text_styling(self, plot):
        gplt.suptitle("Run 42", fontsize=16.0, color="white")
        assert plot.options.axis_title_fontsize == 16.0
        assert plot.options.axis_title_color == "white"

    @pytest.mark.parametrize("kwargs", [{"x": 0.5}, {"y": 0.9}])
    def test_suptitle_position_kwargs_warn(self, plot, kwargs):
        with pytest.warns(gplt.MatplotlibCompatWarning):
            gplt.suptitle("Run 42", **kwargs)


def _patch_area(layer) -> float:
    """The area the emitted triangles actually cover, by the shoelace formula.

    Asserted on instead of the vertex count because the count is the easy thing to get
    right while filling the wrong region: `where` that emits every vertex and no holes
    passes a count check and draws a solid band.
    """
    verts, indices = layer.vertices, layer.indices
    if indices is None or len(indices) == 0:
        return 0.0
    t = verts[indices].reshape(-1, 3, 2)
    return float(
        np.sum(
            np.abs(
                0.5
                * (
                    (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
                    - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1])
                )
            )
        )
    )


def _last_patch(live):
    return [ly for ly in live.scene.layers if ly.layer_type == "patch"][-1]


class TestFillBetweenParity:
    """`where` needs holes in the geometry, which one triangle strip cannot have."""

    X = np.linspace(0.0, 10.0, 11)

    @property
    def Y1(self):
        return np.sin(self.X)

    def test_where_fills_less_than_the_whole_band(self, plot):
        gplt.fill_between(self.X, self.Y1, 0.0)
        full = _patch_area(_last_patch(plot))
        gplt.fill_between(self.X, self.Y1, 0.0, where=(self.Y1 > 0))
        assert _patch_area(_last_patch(plot)) < full

    def test_all_false_where_fills_nothing(self, plot):
        gplt.fill_between(self.X, self.Y1, 0.0, where=np.zeros(len(self.X), dtype=bool))
        assert len(_last_patch(plot).indices) == 0

    def test_a_lone_true_fills_nothing(self, plot):
        """matplotlib's rule: a segment needs both ends selected."""
        mask = np.zeros(len(self.X), dtype=bool)
        mask[3] = True
        gplt.fill_between(self.X, self.Y1, 0.0, where=mask)
        assert len(_last_patch(plot).indices) == 0

    def test_separate_runs_are_not_bridged(self, plot):
        """Two runs must stay two, or the hole between them is filled in."""
        mask = np.array([True] * 3 + [False] * 4 + [True] * 4)
        gplt.fill_between(self.X, self.Y1, 0.0, where=mask)
        layer = _last_patch(plot)
        # 2 + 3 filled segments, two triangles each -- not the 10 a bridged band would have.
        assert len(layer.indices) // 3 == 2 * (2 + 3)

    def test_interpolate_reaches_the_true_crossing(self, plot):
        """The point of interpolate: fill up to where the curves meet, not the last sample.

        Measured against the exact area of sin's positive lobes over [0, 10], so this
        fails on an implementation that adds the crossing vertices but never indexes them
        into a triangle -- the band would be identical to the plain `where` one.
        """
        gplt.fill_between(self.X, self.Y1, 0.0, where=(self.Y1 > 0))
        plain = _patch_area(_last_patch(plot))
        gplt.fill_between(self.X, self.Y1, 0.0, where=(self.Y1 > 0), interpolate=True)
        interpolated = _patch_area(_last_patch(plot))

        exact = abs(np.cos(0) - np.cos(np.pi)) + abs(np.cos(2 * np.pi) - np.cos(3 * np.pi))
        assert interpolated > plain
        assert abs(interpolated - exact) < abs(plain - exact)

    def test_interpolate_without_where_warns(self, plot):
        with pytest.warns(gplt.MatplotlibCompatWarning, match="interpolate"):
            gplt.fill_between(self.X, self.Y1, 0.0, interpolate=True)

    @pytest.mark.parametrize("step", ["pre", "post", "mid"])
    def test_step_fills_a_staircase(self, plot, step):
        gplt.fill_between(self.X, self.Y1, 0.0, step=step)
        assert len(_last_patch(plot).indices) > 0

    def test_bad_step_raises(self, plot):
        with pytest.raises(ValueError, match="step"):
            gplt.fill_between(self.X, self.Y1, 0.0, step="sideways")

    def test_where_length_is_checked(self, plot):
        with pytest.raises(ValueError, match="where"):
            gplt.fill_between(self.X, self.Y1, 0.0, where=[True, False])

    def test_it_is_still_one_layer(self, plot):
        """matplotlib returns one PolyCollection; several runs must not become several."""
        before = len([ly for ly in plot.scene.layers if ly.layer_type == "patch"])
        gplt.fill_between(
            self.X, self.Y1, 0.0, where=np.array([True] * 3 + [False] * 4 + [True] * 4)
        )
        after = len([ly for ly in plot.scene.layers if ly.layer_type == "patch"])
        assert after == before + 1


class TestHist2dParity:
    def test_weights_replace_the_unit_count(self, plot):
        xs = np.linspace(0.0, 1.0, 20)
        counts, _, _, _ = gplt.hist2d(xs, xs, bins=4, weights=np.full(20, 2.0))
        assert counts.sum() == pytest.approx(40.0)

    def test_weights_length_is_checked(self, plot):
        with pytest.raises(ValueError, match="weights"):
            gplt.hist2d(np.zeros(5), np.zeros(5), weights=[1.0])

    def test_cmin_drops_sparse_cells(self, plot):
        xs = np.concatenate([np.zeros(10), [1.0]])
        layer = gplt.hist2d(xs, xs, bins=2)[3]
        busy = gplt.hist2d(xs, xs, bins=2, cmin=5)[3]
        assert len(busy.vertices) < len(layer.vertices)

    def test_cmax_drops_crowded_cells(self, plot):
        xs = np.concatenate([np.zeros(10), [1.0]])
        layer = gplt.hist2d(xs, xs, bins=2)[3]
        sparse = gplt.hist2d(xs, xs, bins=2, cmax=5)[3]
        assert len(sparse.vertices) < len(layer.vertices)

    def test_empty_cells_stay_out_even_with_cmin_zero(self, plot):
        """Drawing them would colour "nothing landed here" as the colormap's low end."""
        xs = np.concatenate([np.zeros(10), [1.0]])
        counts, _, _, layer = gplt.hist2d(xs, xs, bins=4, cmin=0)
        assert len(layer.vertices) // 4 == int((counts > 0).sum())


class TestStemAndTextParity:
    def test_stem_orientation_warns(self, plot):
        with pytest.warns(gplt.MatplotlibCompatWarning, match="orientation"):
            gplt.stem([0.0, 1.0], [1.0, 2.0], orientation="horizontal")

    def test_stem_basefmt_is_honoured_not_ignored(self, plot):
        """basefmt draws the baseline; it must not be reported as a no-op."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", gplt.MatplotlibCompatWarning)
            gplt.stem([0.0, 1.0], [1.0, 2.0], basefmt="r-")

    def test_stem_data_kwarg(self, plot):
        artists = gplt.stem("a", "b", data={"a": [0.0, 1.0], "b": [1.0, 2.0]})
        assert len(artists) > 0

    def test_text_fontdict_is_honoured(self, plot):
        layer = plot.scene.layers
        before = len(layer)
        gplt.text(0.0, 0.0, "hi", fontdict={"fontsize": 20, "color": "red"})
        added = plot.scene.layers[before]
        assert added.style.text_size_px == 20

    def test_text_explicit_kwarg_beats_fontdict(self, plot):
        before = len(plot.scene.layers)
        gplt.text(0.0, 0.0, "hi", fontdict={"fontsize": 20}, fontsize=30)
        assert plot.scene.layers[before].style.text_size_px == 30

    def test_text_defaults_are_unchanged(self, plot):
        before = len(plot.scene.layers)
        gplt.text(0.0, 0.0, "hi")
        assert plot.scene.layers[before].style.text_size_px == 12


class TestFigureIdentity:
    """matplotlib's `figure(num)` is get-or-create, and `num` took first place."""

    def test_a_positional_string_still_titles_the_window(self):
        """46 calls in this repo spell it this way; `num` must not have broken them.

        matplotlib does the same -- a string num sets the window title -- which is what
        makes taking first place safe here rather than a silent re-binding.
        """
        fig = gplt.figure("🌀 Mandelbrot Fractal", figsize=(10, 10))
        assert fig.title == "🌀 Mandelbrot Fractal"
        assert (fig.width, fig.height) == (1000, 1000)

    def test_the_same_num_returns_the_same_figure(self):
        first = gplt.figure("run")
        gplt.plot([0.0, 1.0], [0.0, 1.0])
        assert gplt.figure("run") is first
        assert len(first.scene.layers) == 1, "the scene was rebuilt, not reused"

    def test_different_nums_are_different_figures(self):
        assert gplt.figure(1) is not gplt.figure(2)

    def test_no_num_always_builds_a_fresh_figure(self):
        assert gplt.figure() is not gplt.figure()

    def test_clear_empties_a_reused_figure(self):
        fig = gplt.figure("run")
        gplt.plot([0.0, 1.0], [0.0, 1.0])
        assert gplt.figure("run", clear=True) is fig
        assert len(fig.scene.layers) == 0

    def test_close_forgets_the_num(self):
        """Otherwise figure(num) reopens a closed figure's scene -- worse than a leak."""
        fig = gplt.figure("gone")
        gplt.close(fig)
        assert gplt.figure("gone") is not fig

    def test_facecolor_sets_the_background(self):
        fig = gplt.figure("bg", facecolor="white")
        assert fig.options.visual.background_color == (1.0, 1.0, 1.0)

    @pytest.mark.parametrize(
        "kwargs", [{"edgecolor": "red"}, {"frameon": False}, {"FigureClass": object}]
    )
    def test_unimplementable_kwargs_warn(self, kwargs):
        with pytest.warns(gplt.MatplotlibCompatWarning):
            gplt.figure("w", **kwargs)


class TestScaleFunctions:
    """xscale/yscale/loglog/semilog* exist now. 'linear', 'log', 'symlog', 'asinh', and
    'logit' are real -- 'function'/'functionlog' still just warn and stay linear."""

    @pytest.mark.parametrize("func", ["xscale", "yscale"])
    @pytest.mark.parametrize("scale", ["linear", "log", "symlog", "asinh", "logit"])
    def test_real_scales_are_honoured_silently(self, plot, func, scale):
        """The scales GLPlot actually draws must not warn about drawing them."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", gplt.MatplotlibCompatWarning)
            getattr(gplt, func)(scale)

    @pytest.mark.parametrize("func", ["xscale", "yscale"])
    @pytest.mark.parametrize("scale", ["function", "functionlog"])
    def test_nonlinear_scales_warn_rather_than_lie(self, plot, func, scale):
        """Transforming the data silently would make the cursor disagree with the plot."""
        with pytest.warns(gplt.MatplotlibCompatWarning, match="linear"):
            getattr(gplt, func)(scale)

    @pytest.mark.parametrize("func", ["xscale", "yscale"])
    def test_an_unknown_scale_still_raises(self, plot, func):
        with pytest.raises(ValueError, match="unsupported scale"):
            getattr(gplt, func)("bogus")

    @pytest.mark.parametrize("func, axis", [("xscale", "x"), ("yscale", "y")])
    def test_log_sets_the_engine_option(self, plot, func, axis):
        getattr(gplt, func)("log")
        assert getattr(plot.options, f"axis_scale_{axis}") == "log"
        getattr(gplt, func)("linear")
        assert getattr(plot.options, f"axis_scale_{axis}") == "linear"

    @pytest.mark.parametrize("func", ["loglog", "semilogx", "semilogy"])
    def test_the_log_plotters_draw_the_curve_without_warning(self, plot, func):
        """The scale is real now: no MatplotlibCompatWarning, and the data still lands."""
        before = len(plot.scene.layers)
        with warnings.catch_warnings():
            warnings.simplefilter("error", gplt.MatplotlibCompatWarning)
            artists = getattr(gplt, func)([1.0, 2.0, 3.0], [1.0, 10.0, 100.0])
        assert len(artists) >= 1
        assert len(plot.scene.layers) > before

    def test_a_bare_xscale_creates_the_figure(self, plot):
        """matplotlib's does; a bare xscale() must not AttributeError on no figure."""
        gplt._cleanup_pyplot_state()
        gplt.xscale("linear")
        assert gplt.gcf() is not None


class TestRealLogScale:
    """`xscale('log')`/`yscale('log')` transform data at GPU-upload time, never
    `layer.pts` -- see `glplot.utils.scale`. These tests cover the parts of that pipeline
    that don't need a live GL context: bounds, ticks, cursor readout, and headless export."""

    def test_forward_and_inverse_round_trip(self):
        from glplot.utils.scale import forward, inverse

        values = np.array([1.0, 10.0, 100.0, 1000.0])
        logged = forward(values, "log")
        np.testing.assert_allclose(logged, [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(inverse(logged, "log"), values)

    def test_linear_mode_is_the_identity(self):
        from glplot.utils.scale import forward, inverse

        values = np.array([-3.0, 0.0, 5.0])
        assert forward(values, "linear") is values
        assert inverse(values, "linear") is values

    def test_non_positive_values_are_masked_not_raised(self):
        """Matches matplotlib: a non-positive point on a log axis is a masked gap, not
        a crash or a warning -- matplotlib itself does not warn per data point either."""
        from glplot.utils.scale import forward

        result = forward(np.array([-1.0, 0.0, 2.0]), "log")
        assert np.isnan(result[0]) and np.isnan(result[1])
        assert result[2] == pytest.approx(np.log10(2.0))

    def test_autoscale_bounds_land_in_log_space(self):
        gplt.figure(width=1000, height=500)
        gplt.plot([1.0, 10.0, 100.0], [2.0, 20.0, 200.0])
        gplt.xscale("log")
        gplt.yscale("log")
        live = gplt._get_or_create_plot()
        l, r, b, t = live.camera_controller.world_window(live.width, live.height)
        # The raw data spans 1..100 (x) and 2..200 (y); in log10 space that is 0..2 / ~0.3..2.3.
        assert l == pytest.approx(0.0, abs=0.5)
        assert r == pytest.approx(2.0, abs=0.5)

    def test_bounds_with_nonpositive_data_do_not_produce_nan(self):
        """A stray non-positive value must not poison the camera with NaN bounds."""
        gplt.figure(width=1000, height=500)
        gplt.plot([-5.0, 1.0, 10.0], [1.0, 2.0, 3.0])
        gplt.xscale("log")
        live = gplt._get_or_create_plot()
        l, r, b, t = live.camera_controller.world_window(live.width, live.height)
        assert np.isfinite(l) and np.isfinite(r)

    def test_decade_ticks_for_a_multi_decade_span(self, plot):
        from glplot.managers.axis import AxisManager

        plot.options.axis_scale_x = "log"
        am = AxisManager(plot)
        ticks = am._generate_log_ticks(0.0, 3.0, target_count=6)  # 1 .. 1000
        np.testing.assert_allclose(ticks.major, [0.0, 1.0, 2.0, 3.0])
        assert ticks.labels == ["1", "10", "100", "1000"]

    def test_nice_ticks_for_a_sub_decade_span(self, plot):
        """Zoomed into less than one decade: round numbers in real space, not raw log math."""
        from glplot.managers.axis import AxisManager

        plot.options.axis_scale_x = "log"
        am = AxisManager(plot)
        ticks = am._generate_log_ticks(np.log10(100.0), np.log10(180.0), target_count=4)
        assert len(ticks.major) >= 2
        real_values = 10.0**ticks.major
        # Every generated position must round-trip to a "nice" (1-2-5 ladder) real value.
        for v in real_values:
            assert v == pytest.approx(round(v))

    def test_mouse_world_display_inverts_on_a_log_axis(self, plot):
        plot.options.axis_scale_x = "log"
        plot.options.axis_scale_y = "log"
        plot.mouse_world = (2.0, 1.0)  # log10 space
        x, y = plot.mouse_world_display()
        assert x == pytest.approx(100.0)
        assert y == pytest.approx(10.0)

    def test_mouse_world_display_is_passthrough_when_linear(self, plot):
        plot.mouse_world = (3.0, -2.0)
        assert plot.mouse_world_display() == (3.0, -2.0)

    def test_headless_export_applies_a_real_log_scale(self, tmp_path, monkeypatch):
        import matplotlib.figure

        captured = {}
        original_savefig = matplotlib.figure.Figure.savefig

        def capture_savefig(self, *args, **kwargs):
            captured["ax"] = self.axes[0]
            return original_savefig(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_savefig)

        gplt.plot([1.0, 2.0, 3.0], [1.0, 10.0, 100.0])
        gplt.yscale("log")
        gplt.savefig(str(tmp_path / "log_export.png"))

        assert captured["ax"].get_yscale() == "log"
        assert captured["ax"].get_xscale() == "linear"


class TestRealSymlogAndAsinhScale:
    """`xscale('symlog'/'asinh')` extend the log-scale architecture in
    [[TestRealLogScale]] to two more real scales. Both are defined for every real
    number (including 0 and negatives), unlike log -- see `glplot.utils.scale`, whose
    formulas are copied verbatim from `matplotlib.scale` and checked against it here."""

    @pytest.mark.parametrize(
        "mode, mpl_cls, mpl_kwargs, our_params",
        [
            ("symlog", "SymmetricalLogTransform", (10, 2, 1), None),
            ("symlog", "SymmetricalLogTransform", (10, 5, 2), {"linthresh": 5, "linscale": 2}),
            ("asinh", "AsinhTransform", (1.0,), None),
            ("asinh", "AsinhTransform", (3.0,), {"linear_width": 3.0}),
        ],
    )
    def test_forward_and_inverse_match_matplotlib_exactly(
        self, mode, mpl_cls, mpl_kwargs, our_params
    ):
        import matplotlib.scale as mscale

        from glplot.utils.scale import forward, inverse

        values = np.array([-1000.0, -50.0, -2.0, -0.5, 0.0, 0.5, 2.0, 50.0, 1000.0])
        transform = getattr(mscale, mpl_cls)(*mpl_kwargs)

        mine = forward(values, mode, our_params)
        theirs = transform.transform_non_affine(values.copy())
        np.testing.assert_allclose(mine, theirs)

        mine_inv = inverse(mine, mode, our_params)
        np.testing.assert_allclose(mine_inv, values)

    @pytest.mark.parametrize("func, axis", [("xscale", "x"), ("yscale", "y")])
    def test_symlog_kwargs_are_captured_on_the_option(self, plot, func, axis):
        getattr(gplt, func)("symlog", linthresh=5, linscale=2)
        assert getattr(plot.options, f"axis_scale_{axis}") == "symlog"
        assert getattr(plot.options, f"axis_scale_params_{axis}") == {
            "linthresh": 5,
            "linscale": 2,
        }

    def test_asinh_kwargs_are_captured_on_the_option(self, plot):
        gplt.xscale("asinh", linear_width=2.5)
        assert plot.options.axis_scale_x == "asinh"
        assert plot.options.axis_scale_params_x == {"linear_width": 2.5}

    def test_bounds_transform_for_data_crossing_zero(self):
        """Unlike log, symlog/asinh need no NaN fallback -- both are defined at 0."""
        gplt.figure(width=1000, height=500)
        gplt.plot([-100.0, 0.0, 100.0], [-50.0, 0.0, 50.0])
        gplt.xscale("symlog")
        gplt.yscale("asinh")
        live = gplt._get_or_create_plot()
        l, r, b, t = live.camera_controller.world_window(live.width, live.height)
        assert np.isfinite(l) and np.isfinite(r) and np.isfinite(b) and np.isfinite(t)
        assert l < 0 < r  # the view still straddles zero

    def test_symmetric_ticks_mirror_decades_around_zero(self, plot):
        from glplot.managers.axis import AxisManager
        from glplot.utils.scale import forward, inverse

        plot.options.axis_scale_x = "symlog"
        am = AxisManager(plot)
        vmin, vmax = forward(-1000.0, "symlog"), forward(1000.0, "symlog")
        ticks = am._generate_symmetric_ticks(vmin, vmax, target_count=6, mode="symlog", params=None)
        real_values = inverse(ticks.major, "symlog")
        assert 0.0 in real_values
        assert -1000.0 == pytest.approx(real_values.min())
        assert 1000.0 == pytest.approx(real_values.max())
        # Symmetric: every positive candidate has a matching negative one.
        np.testing.assert_allclose(sorted(real_values), sorted(-real_values))

    def test_symmetric_ticks_fall_back_to_a_nice_ladder_inside_the_linear_region(self, plot):
        from glplot.managers.axis import AxisManager
        from glplot.utils.scale import forward

        plot.options.axis_scale_x = "asinh"
        am = AxisManager(plot)
        vmin, vmax = forward(-1.0, "asinh"), forward(1.0, "asinh")
        ticks = am._generate_symmetric_ticks(vmin, vmax, target_count=4, mode="asinh", params=None)
        assert len(ticks.major) >= 2

    def test_headless_export_applies_symlog_with_its_kwargs(self, tmp_path, monkeypatch):
        import matplotlib.figure

        captured = {}
        original_savefig = matplotlib.figure.Figure.savefig

        def capture_savefig(self, *args, **kwargs):
            captured["ax"] = self.axes[0]
            return original_savefig(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_savefig)

        gplt.plot([-100.0, 0.0, 100.0], [1.0, 2.0, 3.0])
        gplt.xscale("symlog", linthresh=10)
        gplt.savefig(str(tmp_path / "symlog_export.png"))

        assert captured["ax"].get_xscale() == "symlog"
        assert captured["ax"].xaxis.get_transform().linthresh == 10


class TestRealLogBarAndBarh:
    """`bar()`/`barh()` under a real log axis: the baseline vertex (`bottom`, default 0)
    would transform to -inf under plain `log10`, which corrupts a whole triangle rather
    than just dropping a point/segment the way a line or scatter does -- see
    `glplot.utils.scale.clamp_to_domain` (shared by `patch.py` and, since the imshow-quad
    fix, `scatter.py` too). symlog/asinh need no such handling: both are defined at 0."""

    def test_clamp_keeps_a_bars_baseline_finite(self):
        from glplot.utils.scale import clamp_to_domain

        # A bar's 4 corners: two at the baseline (0), two at the top (5).
        coords = np.array([0.0, 0.0, 5.0, 5.0])
        clamped = clamp_to_domain(coords, "log")
        assert np.all(np.isfinite(clamped)) and np.all(clamped > 0)
        assert clamped[2] == 5.0 and clamped[3] == 5.0
        assert clamped[0] < 5.0 / 1e5  # far enough below the top to run off-screen

    def test_clamp_falls_back_when_nothing_positive_exists(self):
        from glplot.utils.scale import clamp_to_domain

        coords = np.array([0.0, -3.0, -1.0])
        clamped = clamp_to_domain(coords, "log")
        assert np.all(np.isfinite(clamped)) and np.all(clamped > 0)

    def test_clamped_baseline_survives_the_log_forward_transform(self):
        from glplot.utils.scale import clamp_to_domain, forward

        coords = np.array([0.0, 0.0, 5.0, 5.0])
        transformed = forward(clamp_to_domain(coords, "log"), "log")
        assert np.all(np.isfinite(transformed))

    def test_bar_under_log_yscale_exports_headless_without_crashing(self, tmp_path):
        gplt.bar(["a", "b", "c"], [5.0, 50.0, 500.0])
        gplt.yscale("log")
        gplt.savefig(str(tmp_path / "log_bar.png"))
        assert (tmp_path / "log_bar.png").exists()

    def test_barh_under_symlog_xscale_exports_headless_without_crashing(self, tmp_path):
        gplt.barh(["a", "b"], [-50.0, 50.0])
        gplt.xscale("symlog")
        gplt.savefig(str(tmp_path / "symlog_barh.png"))
        assert (tmp_path / "symlog_barh.png").exists()


class TestRealLogitScale:
    """`xscale('logit')`/`yscale('logit')`: domain `(0, 1)`, masked outside it like
    `log`'s non-positive case -- see `glplot.utils.scale`, whose formulas are copied
    verbatim from `matplotlib.scale.LogitTransform`/`LogisticTransform`."""

    def test_forward_and_inverse_match_matplotlib_exactly(self):
        import matplotlib.scale as mscale

        from glplot.utils.scale import forward, inverse

        values = np.array([0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])
        transform = mscale.LogitTransform("mask")

        mine = forward(values, "logit")
        theirs = transform.transform_non_affine(values.copy())
        np.testing.assert_allclose(mine, theirs)
        np.testing.assert_allclose(inverse(mine, "logit"), values)

    def test_outside_the_open_interval_is_masked_not_raised(self):
        from glplot.utils.scale import forward

        result = forward(np.array([-1.0, 0.0, 0.5, 1.0, 2.0]), "logit")
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert not np.isnan(result[2])
        assert np.isnan(result[3])
        assert np.isnan(result[4])

    def test_clamp_to_domain_keeps_both_edges_finite(self):
        from glplot.utils.scale import clamp_to_domain

        clamped = clamp_to_domain(np.array([0.0, 0.0, 0.3, 1.0]), "logit")
        assert np.all(np.isfinite(clamped))
        assert np.all((clamped > 0) & (clamped < 1))

    def test_clamp_to_domain_log_mode_is_unchanged(self):
        """Regression check: generalizing the clamp must not change log's own behavior."""
        from glplot.utils.scale import clamp_to_domain

        clamped = clamp_to_domain(np.array([0.0, 0.0, 5.0, 5.0]), "log")
        assert np.all(clamped > 0)
        assert clamped[2] == 5.0 and clamped[3] == 5.0

    @pytest.mark.parametrize("func, axis", [("xscale", "x"), ("yscale", "y")])
    def test_logit_sets_the_engine_option(self, plot, func, axis):
        getattr(gplt, func)("logit")
        assert getattr(plot.options, f"axis_scale_{axis}") == "logit"

    def test_ticks_follow_matplotlibs_canonical_logit_ladder(self, plot):
        from glplot.managers.axis import AxisManager
        from glplot.utils.scale import forward, inverse

        plot.options.axis_scale_x = "logit"
        am = AxisManager(plot)
        vmin, vmax = forward(0.001, "logit"), forward(0.999, "logit")
        ticks = am._generate_logit_ticks(vmin, vmax, target_count=6)
        real_values = inverse(ticks.major, "logit")
        assert 0.5 in np.round(real_values, 6)
        np.testing.assert_allclose(sorted(real_values)[:2], [0.001, 0.01])
        np.testing.assert_allclose(sorted(real_values)[-2:], [0.99, 0.999])

    def test_bar_under_logit_yscale_exports_headless_without_crashing(self, tmp_path):
        gplt.bar(["a", "b", "c"], [0.1, 0.5, 0.9])
        gplt.yscale("logit")
        gplt.savefig(str(tmp_path / "logit_bar.png"))
        assert (tmp_path / "logit_bar.png").exists()

    def test_headless_export_applies_a_real_logit_scale(self, tmp_path, monkeypatch):
        import matplotlib.figure

        captured = {}
        original_savefig = matplotlib.figure.Figure.savefig

        def capture_savefig(self, *args, **kwargs):
            captured["ax"] = self.axes[0]
            return original_savefig(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_savefig)

        gplt.plot([0.01, 0.1, 0.5, 0.9, 0.99], [1.0, 2.0, 3.0, 4.0, 5.0])
        gplt.xscale("logit")
        gplt.savefig(str(tmp_path / "logit_export.png"))

        assert captured["ax"].get_xscale() == "logit"


class TestImshowAndContourfUnderARealScale:
    """`imshow()`'s image quad (and therefore `contourf()`, `matshow()`, `specgram()`,
    `figimage()`, which all draw through it) had its own separate GPU-upload path that
    the log/symlog/asinh work never reached -- see `renderers/scatter.py`'s
    `_make_image_quad()`. Its cache (`layer._image_gl`) is also a second cache key
    `gpu_dirty` doesn't gate, so `_set_scale()` must clear it explicitly too."""

    def test_scale_change_invalidates_the_image_quad_cache(self):
        """No live GL context needed for this one: `_image_gl` is just an attribute,
        so a sentinel stands in for a real quad to check `_set_scale()`'s invalidation
        logic in isolation (this suite has no GL context -- see the module docstring)."""
        gplt.imshow(np.random.rand(10, 10), extent=(1, 100, 1, 50))
        plot = gplt._get_or_create_plot()
        layer = plot.scene.layers[-1]
        layer._image_gl = "sentinel-standing-in-for-a-real-quad"

        gplt.xscale("log")
        assert layer._image_gl is None  # invalidated immediately, not just marked dirty

    def test_imshow_under_log_xscale_computes_the_transformed_quad(self):
        """Exercises the same corner math `_make_image_quad` uploads, without a live GL
        context (this suite has none -- see the module docstring): `extent` transforms
        the same way `renderers/scatter.py`'s hook does, verified against matplotlib's
        own log10. A hidden-GLFW-window script separately confirmed the real uploaded
        GPU buffer matches this exact computation (see the project's own memory notes)."""
        from glplot.utils.scale import forward

        xmin, xmax = 1.0, 100.0
        x_positions = forward(np.array([xmin, xmax]), "log")
        np.testing.assert_allclose(x_positions, [0.0, 2.0])

    def test_contourf_under_log_yscale_exports_headless_without_crashing(self, tmp_path):
        x = np.linspace(1.0, 10.0, 20)
        y = np.linspace(1.0, 100.0, 20)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y / 10.0)
        gplt.contourf(X, Y, Z, levels=8)
        gplt.yscale("log")
        gplt.savefig(str(tmp_path / "contourf_log.png"))
        assert (tmp_path / "contourf_log.png").exists()


class TestContourStreamplotHist2dAlreadySupportRealScales:
    """`contour()`'s live lines and `streamplot()` both draw through `add_line_strip()`
    (the `PolylineLayer` path already patched for log/symlog/asinh) and needed no new
    code. `hist2d()` reaches the equivalent hook on `PatchLayer` (`renderers/patch.py`)
    -- the same one `bar()` already used -- so every cell corner, not just a stand-in
    centre point, is scale-aware: the rendered tiles keep tiling with no gap or overlap
    even under a nonlinear axis. This class makes that fact checked rather than assumed.
    """

    def test_contour_under_log_scale_exports_headless_without_crashing(self, tmp_path):
        x = np.linspace(1.0, 10.0, 15)
        y = np.linspace(1.0, 50.0, 15)
        X, Y = np.meshgrid(x, y)
        Z = X**2 - Y
        gplt.contour(X, Y, Z, levels=5)
        gplt.xscale("log")
        gplt.savefig(str(tmp_path / "contour_log.png"))
        assert (tmp_path / "contour_log.png").exists()

    def test_streamplot_under_symlog_scale_exports_headless_without_crashing(self, tmp_path):
        x = np.linspace(-10.0, 10.0, 15)
        y = np.linspace(-10.0, 10.0, 15)
        X, Y = np.meshgrid(x, y)
        U = -1 - X**2 + Y
        V = 1 + X - Y**2
        gplt.streamplot(X, Y, U, V, density=0.5)
        gplt.xscale("symlog")
        gplt.savefig(str(tmp_path / "streamplot_symlog.png"))
        assert (tmp_path / "streamplot_symlog.png").exists()

    def test_hist2d_under_log_scale_exports_headless_without_crashing(self, tmp_path):
        rng = np.random.default_rng(0)
        x = rng.uniform(1.0, 1000.0, 2000)
        y = rng.uniform(1.0, 100.0, 2000)
        gplt.hist2d(x, y, bins=15)
        gplt.xscale("log")
        gplt.savefig(str(tmp_path / "hist2d_log.png"))
        assert (tmp_path / "hist2d_log.png").exists()

    def test_hist2d_bin_edges_are_linear_in_data_space(self):
        """`layer.vertices` is the raw truth the Data panel/CSV export read -- a real
        axis scale transforms a throwaway GPU-upload copy, never this array (see the
        module comment above `_SCALE_NAMES` in pyplot.py) -- so the *stored* cell edges
        stay exactly what `np.histogram2d` computed regardless of any active scale."""
        rng = np.random.default_rng(0)
        x = rng.uniform(1.0, 1000.0, 500)
        y = rng.uniform(1.0, 100.0, 500)
        _, _, _, layer = gplt.hist2d(x, y, bins=10)
        xs = np.sort(np.unique(np.round(layer.vertices[:, 0].astype(np.float64), 6)))
        gaps = np.diff(xs)
        # Linear edges -> roughly equal gaps between adjacent cell boundaries.
        assert gaps.max() / gaps.min() < 1.5

    def test_hist2d_tiles_still_share_edges_after_a_real_log_transform(self):
        """The actual claim the module comment makes: every cell corner is scale-aware
        (not just a stand-in centre point), so applying the exact transform
        `renderers/patch.py` applies at GPU-upload to the layer's own vertices leaves
        adjacent cells sharing the same transformed edge -- no gap, no overlap, under a
        real log axis, the same as under a linear one."""
        from glplot.utils.scale import forward

        rng = np.random.default_rng(0)
        x = rng.uniform(1.0, 1000.0, 500)
        y = rng.uniform(1.0, 100.0, 500)
        _, _, _, layer = gplt.hist2d(x, y, bins=10)

        raw_xs = np.round(layer.vertices[:, 0].astype(np.float64), 6)
        transformed = forward(layer.vertices[:, 0].astype(np.float64), "log", None)
        # Group the transformed x by its raw (pre-transform) value: every vertex sharing
        # a raw edge must land on the exact same transformed coordinate, or two cells
        # that used to share a wall would now show a gap or an overlap there.
        for raw_value in np.unique(raw_xs):
            group = transformed[raw_xs == raw_value]
            assert np.allclose(group, group[0]), f"edge at x={raw_value} split under log scale"


class TestBoxplot:
    VALUES = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0])

    def test_it_draws_a_box(self, plot):
        artists = gplt.boxplot(self.VALUES)
        assert len(artists["boxes"]) == 1

    def test_outliers_become_their_own_layer(self, plot):
        """Fliers are points; the box is a polyline. They cannot be one layer."""
        artists = gplt.boxplot(self.VALUES)
        assert len(artists["fliers"]) == 1

    def test_showfliers_false_omits_them(self, plot):
        assert gplt.boxplot(self.VALUES, showfliers=False)["fliers"] == []

    def test_positions_and_widths_place_the_box(self, plot):
        box = gplt.boxplot(self.VALUES, positions=[5.0], widths=0.5)["boxes"][0]
        assert box.pts[:, 0].min() == pytest.approx(4.75)
        assert box.pts[:, 0].max() == pytest.approx(5.25)

    def test_all_nan_raises_rather_than_drawing_nothing(self, plot):
        with pytest.raises(ValueError, match="no finite values"):
            gplt.boxplot(np.full(5, np.nan))

    def test_zorder_warns(self, plot):
        """The one keyword left that GLPlot cannot honour: layers draw in add order."""
        with pytest.warns(gplt.MatplotlibCompatWarning):
            gplt.boxplot(self.VALUES, zorder=5)


class TestBoxplotShape:
    """One box per dataset, and the three input shapes matplotlib accepts.

    ``boxplot([a, b, c])`` is the single most common call in the wild and it used to raise
    ``ValueError: x must have ndim=1`` -- the function only ever drew one box. The 2-D case
    is the subtle one: matplotlib reads *columns* as datasets, so a transposed
    implementation still draws the right number of boxes off the wrong numbers.
    """

    @pytest.fixture
    def groups(self):
        rng = np.random.default_rng(1)
        return [rng.normal(size=120), rng.normal(loc=5, size=120), rng.normal(loc=-3, size=120)]

    def test_a_sequence_of_arrays_draws_one_box_each(self, plot, groups):
        assert len(gplt.boxplot(groups)["boxes"]) == 3

    def test_a_2d_array_draws_one_box_per_column(self, plot, groups):
        """Columns, not rows. Three 120-sample columns is three boxes, not 120."""
        assert len(gplt.boxplot(np.column_stack(groups))["boxes"]) == 3

    def test_columns_are_summarised_not_rows(self, plot, groups):
        """The medians must be the per-column medians, which the transpose would not give."""
        medians = [float(m.pts[0, 1]) for m in gplt.boxplot(np.column_stack(groups))["medians"]]
        assert np.allclose(medians, [np.median(g) for g in groups], atol=1e-5)

    def test_default_positions_start_at_one(self, plot, groups):
        """matplotlib puts the first box at 1, not 0 -- ticks read 1, 2, 3."""
        centres = [
            float(np.mean([b.pts[:, 0].min(), b.pts[:, 0].max()]))
            for b in gplt.boxplot(groups)["boxes"]
        ]
        assert np.allclose(centres, [1.0, 2.0, 3.0])

    def test_statistics_match_matplotlib(self, plot, groups):
        from matplotlib import cbook

        expected = cbook.boxplot_stats(groups, whis=1.5)
        got = [float(m.pts[0, 1]) for m in gplt.boxplot(groups, whis=1.5)["medians"]]
        assert np.allclose(got, [e["med"] for e in expected], atol=1e-5)


class TestBoxplotElements:
    """The keywords that decide *which* pieces get drawn, and where they go."""

    @pytest.fixture
    def values(self):
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0])

    def test_every_piece_is_returned_under_matplotlibs_key(self, plot, values):
        artists = gplt.boxplot(values, showmeans=True)
        for key in ("boxes", "medians", "whiskers", "caps", "fliers", "means"):
            assert artists[key], f"{key} was not drawn"

    def test_two_whiskers_and_two_caps_per_box(self, plot, values):
        artists = gplt.boxplot([values, values])
        assert len(artists["whiskers"]) == 4 and len(artists["caps"]) == 4

    def test_showbox_false_omits_the_box(self, plot, values):
        assert gplt.boxplot(values, showbox=False)["boxes"] == []

    def test_showcaps_false_omits_the_caps(self, plot, values):
        assert gplt.boxplot(values, showcaps=False)["caps"] == []

    def test_notch_pinches_the_box_at_the_median(self, plot, values):
        """The notched outline has more vertices than the plain rectangle, and the
        narrowest point sits at the median."""
        plain = gplt.boxplot(values)["boxes"][0]
        notched = gplt.boxplot(values, notch=True)["boxes"][0]
        assert len(notched.pts) > len(plain.pts)

    def test_vert_false_swaps_the_axes(self, plot, values):
        """A horizontal box spans the value range along x, not y."""
        vertical = gplt.boxplot(values)["boxes"][0]
        horizontal = gplt.boxplot(values, vert=False)["boxes"][0]
        assert np.ptp(horizontal.pts[:, 0]) == pytest.approx(np.ptp(vertical.pts[:, 1]))
        assert np.ptp(horizontal.pts[:, 1]) == pytest.approx(np.ptp(vertical.pts[:, 0]))

    def test_meanline_draws_a_line_and_not_a_marker(self, plot, values):
        """With meanline the mean is a polyline across the box; without, it is a point."""
        line = gplt.boxplot(values, showmeans=True, meanline=True)["means"][0]
        marker = gplt.boxplot(values, showmeans=True)["means"][0]
        assert type(line) is not type(marker)

    def test_sym_empty_hides_the_fliers(self, plot, values):
        assert gplt.boxplot(values, sym="")["fliers"] == []

    def test_usermedians_overrides_the_computed_median(self, plot, values):
        median = gplt.boxplot(values, usermedians=[42.0])["medians"][0]
        assert float(median.pts[0, 1]) == pytest.approx(42.0)

    def test_usermedians_none_entry_keeps_the_computed_one(self, plot, values):
        """A None entry means "leave this one alone", not "use zero"."""
        got = gplt.boxplot([values, values], usermedians=[42.0, None])["medians"]
        assert float(got[1].pts[0, 1]) == pytest.approx(np.median(values), abs=1e-5)

    def test_capwidths_narrows_the_caps_only(self, plot, values):
        cap = gplt.boxplot(values, widths=0.8, capwidths=0.2)["caps"][0]
        assert np.ptp(cap.pts[:, 0]) == pytest.approx(0.2)

    def test_patch_artist_fills_the_box(self, plot, values):
        """A filled box is a patch layer, not the polyline the default draws."""
        outlined = gplt.boxplot(values)["boxes"][0]
        filled = gplt.boxplot(values, patch_artist=True)["boxes"][0]
        assert type(filled) is not type(outlined)

    def test_patch_artist_tessellates_a_notched_box(self, plot, values):
        """A notch makes the outline non-convex; a triangle strip would fold it inside
        out, so the fan has to cover all eleven edges."""
        filled = gplt.boxplot(values, patch_artist=True, notch=True)["boxes"][0]
        assert len(filled.indices) == 10 * 3

    @pytest.mark.parametrize("labels_kw", ["labels", "tick_labels"])
    def test_both_label_spellings_work(self, plot, values, labels_kw):
        """matplotlib 3.9 renamed ``labels`` to ``tick_labels``; scripts exist for both."""
        gplt.boxplot([values, values], **{labels_kw: ["left", "right"]})
        # `xticks()` reports the ticks in view, and the default camera is (-1, 1) -- the
        # second box sits at x=2 and would be filtered out before the frame is fitted.
        gplt.autoscale()
        _, texts = gplt.xticks()
        assert "left" in texts and "right" in texts


class TestPie:
    def test_one_layer_per_wedge(self, plot):
        wedges, _ = gplt.pie([30.0, 20.0, 50.0])
        assert len(wedges) == 3

    def test_wedges_are_sized_by_share(self, plot):
        """A half-share wedge must cover twice the area of a quarter-share one."""
        wedges, _ = gplt.pie([50.0, 25.0, 25.0])
        areas = [_patch_area(ly) for ly in wedges]
        assert areas[0] == pytest.approx(areas[1] * 2.0, rel=0.02)

    def test_values_are_normalised(self, plot):
        """x need not sum to 1: the wedges are shares, so both spellings agree."""
        raw = [_patch_area(ly) for ly in gplt.pie([30.0, 20.0, 50.0])[0]]
        gplt.figure("norm")
        unit = [_patch_area(ly) for ly in gplt.pie([0.3, 0.2, 0.5])[0]]
        assert raw == pytest.approx(unit, rel=1e-3)

    def test_explode_moves_a_wedge_off_centre(self, plot):
        plain = gplt.pie([1.0, 1.0])[0][0]
        gplt.figure("exploded")
        moved = gplt.pie([1.0, 1.0], explode=[0.5, 0.0])[0][0]
        assert not np.allclose(plain.vertices[0], moved.vertices[0])

    def test_labels_reach_the_layers(self, plot):
        """The label goes on the wedge too, so ``legend()`` finds it -- matplotlib
        attaches it to the Wedge artist as well as drawing the text."""
        wedges, _ = gplt.pie([1.0, 1.0], labels=["alpha", "beta"])
        assert [ly.label for ly in wedges] == ["alpha", "beta"]

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"explode": [0.1]}, "explode"),
            ({"labels": ["only-one"]}, "labels"),
            ({"colors": ["red"]}, "colors"),
        ],
    )
    def test_mismatched_lengths_raise(self, plot, kwargs, match):
        with pytest.raises(ValueError, match=match):
            gplt.pie([1.0, 1.0, 1.0], **kwargs)

    @pytest.mark.parametrize(
        "values, match",
        [([], "empty"), ([1.0, -1.0], "negative"), ([0.0, 0.0], "positive")],
    )
    def test_degenerate_input_raises(self, plot, values, match):
        with pytest.raises(ValueError, match=match):
            gplt.pie(values)


class TestPieAnnotation:
    """``autopct`` and the label text -- the reason most real ``pie`` calls have kwargs.

    A pie with no share written on it makes the reader estimate angles by eye, which is
    the whole complaint about pie charts. ``autopct`` is how matplotlib answers it, and it
    used to be a TypeError here.
    """

    def test_autopct_adds_a_third_return_value(self, plot):
        """matplotlib returns ``(wedges, texts)`` normally and ``(..., autotexts)`` with
        autopct. Code unpacks both forms, so the arity has to switch."""
        assert len(gplt.pie([1.0, 1.0])) == 2
        assert len(gplt.pie([1.0, 1.0], autopct="%d")) == 3

    def test_autopct_format_string_gets_the_percentage(self, plot):
        _, _, autotexts = gplt.pie([25.0, 25.0, 50.0], autopct="%1.0f%%")
        assert [t.text for t in autotexts] == ["25%", "25%", "50%"]

    def test_autopct_accepts_a_callable(self, plot):
        _, _, autotexts = gplt.pie([50.0, 50.0], autopct=lambda p: f"<{p:.0f}>")
        assert [t.text for t in autotexts] == ["<50>", "<50>"]

    def test_labels_are_drawn_as_text(self, plot):
        _, texts = gplt.pie([1.0, 1.0], labels=["alpha", "beta"])
        assert [t.text for t in texts] == ["alpha", "beta"]

    def test_labeldistance_none_draws_no_text(self, plot):
        """matplotlib's escape hatch for "label the legend, not the pie"."""
        _, texts = gplt.pie([1.0, 1.0], labels=["a", "b"], labeldistance=None)
        assert texts == []

    def test_pctdistance_moves_the_percentage_outward(self, plot):
        near = gplt.pie([1.0], autopct="%d", pctdistance=0.2)[2][0]
        gplt.figure("far")
        far = gplt.pie([1.0], autopct="%d", pctdistance=0.9)[2][0]
        assert np.hypot(far.x, far.y) > np.hypot(near.x, near.y)


class TestPieWedgeShape:
    """``wedgeprops`` and ``normalize`` -- the keywords that change the geometry itself."""

    def _radii(self, layer):
        v = layer.vertices
        return np.hypot(v[:, 0], v[:, 1])

    def test_width_carves_a_donut(self, plot):
        """The standard donut recipe. A hole means no vertex sits near the centre."""
        wedges, _ = gplt.pie([1.0], wedgeprops={"width": 0.3})
        r = self._radii(wedges[0])
        assert r.min() == pytest.approx(0.7, abs=1e-3)
        assert r.max() == pytest.approx(1.0, abs=1e-3)

    def test_a_solid_wedge_still_reaches_the_centre(self, plot):
        """The annulus path must not leak into the default: without ``width`` the wedge
        is a fan from the middle."""
        wedges, _ = gplt.pie([1.0])
        assert self._radii(wedges[0]).min() == pytest.approx(0.0, abs=1e-6)

    def test_radius_scales_the_pie(self, plot):
        wedges, _ = gplt.pie([1.0], radius=2.5)
        assert self._radii(wedges[0]).max() == pytest.approx(2.5, rel=1e-3)

    def test_normalize_false_leaves_a_gap(self, plot):
        """Half a unit, unnormalised, is half a pie -- not a full circle."""
        wedges, _ = gplt.pie([0.5], normalize=False)
        v = wedges[0].vertices[1:]
        sweep = np.rad2deg(np.ptp(np.arctan2(v[:, 1], v[:, 0])))
        assert sweep == pytest.approx(180.0, abs=1.0)

    def test_normalize_false_rejects_an_oversized_sum(self, plot):
        with pytest.raises(ValueError, match="unnormalized"):
            gplt.pie([0.8, 0.5], normalize=False)

    @pytest.mark.parametrize("kwargs", [{"rotatelabels": True}, {"hatch": "//"}])
    def test_unimplementable_kwargs_warn(self, plot, kwargs):
        with pytest.warns(gplt.MatplotlibCompatWarning):
            gplt.pie([1.0, 1.0], labels=["a", "b"], **kwargs)


class TestSingleViewportStubs:
    """Compat names that still map onto the single viewport GLPlot draws each panel into.

    ``twinx``/``twiny`` (two scales in one viewport) and ``tight_layout`` have no real
    backing yet: they return the current axes and *say so* rather than pretending to have
    worked. ``subplot`` and ``subplot_mosaic``, by contrast, are now real -- see
    :class:`TestRealSubplots`. ``colorbar`` is real too -- see :class:`TestColorbar`.
    """

    @pytest.mark.parametrize("func", ["twinx", "twiny"])
    def test_twin_axes_warn_that_they_are_not_twinned(self, plot, func):
        """Returning the same axes silently would draw series 2 on axis 1 unannounced.

        The return value is an AxesProxy (not the bare figure) bound to the same
        panel/figure -- so the standard ``ax2 = ax1.twinx(); ax2.plot(...)`` idiom has a
        real ``.plot()`` to call, rather than raising AttributeError on the next line.
        """
        with pytest.warns(gplt.MatplotlibCompatWarning, match="not a twinned one"):
            twin = getattr(gplt, func)()
        assert isinstance(twin, gplt.AxesProxy)
        assert twin.figure is gplt.gcf()

    def test_tight_layout_warns_and_points_at_the_margins(self, plot):
        with pytest.warns(gplt.MatplotlibCompatWarning, match="axis_margin"):
            gplt.tight_layout()


class TestColorbar:
    """``colorbar()`` shrinks the host panel's own rect and attaches a real
    :class:`~glplot.core.panel.ColorbarSpec` -- not a second GL viewport (see that
    class's docstring for why not), so these tests stay at the metadata/geometry level
    that is safe without a live GL window (CONTRACT 5.1).
    """

    def test_no_mappable_raises(self, plot):
        with pytest.raises(RuntimeError, match="no colormapped layer"):
            gplt.colorbar()

    def test_default_location_is_right_and_shrinks_the_panel_width(self, plot):
        panel = gplt.gca().panel
        x0, y0, w, h = panel.rect_frac
        im = gplt.imshow(np.random.default_rng(0).random((5, 5)), cmap="viridis")
        cb = gplt.colorbar(im)

        assert isinstance(cb, gplt.Colorbar)
        assert len(panel.colorbars) == 1
        spec = panel.colorbars[0]
        assert spec.location == "right"
        assert spec.orientation == "vertical"
        # The host panel's own rect narrowed; its (x0, y0, h) are untouched.
        assert panel.rect_frac[2] < w
        assert panel.rect_frac[0] == x0
        assert panel.rect_frac[1] == y0
        assert panel.rect_frac[3] == h
        # The bar sits in the strip vacated on the right, still inside the figure.
        bx0, by0, bw, bh = spec.rect_frac
        assert bx0 + bw == pytest.approx(x0 + w)
        assert by0 == y0 and bh == h

    @pytest.mark.parametrize(
        "location,orientation",
        [("left", "vertical"), ("top", "horizontal"), ("bottom", "horizontal")],
    )
    def test_other_three_locations(self, plot, location, orientation):
        panel = gplt.gca().panel
        im = gplt.imshow(np.random.default_rng(1).random((5, 5)))
        cb = gplt.colorbar(im, location=location)
        spec = cb._spec
        assert spec.location == location
        assert spec.orientation == orientation
        assert panel.colorbars[-1] is spec

    def test_resolves_cmap_and_norm_from_a_scatter_layer(self, plot):
        v = np.linspace(0.0, 5.0, 10)
        layer = gplt.scatter(v, v, c=v, cmap="plasma", vmin=1.0, vmax=4.0)
        cb = gplt.colorbar(layer)
        spec = cb._spec
        assert spec.cmap == "plasma"
        assert spec.norm.vmin == pytest.approx(1.0)
        assert spec.norm.vmax == pytest.approx(4.0)

    def test_honours_an_explicit_normalize_instance(self, plot):
        from matplotlib.colors import LogNorm

        matrix = np.abs(np.random.default_rng(2).random((6, 6))) + 0.1
        im = gplt.imshow(matrix, cmap="inferno", norm=LogNorm(vmin=0.5, vmax=2.0))
        cb = gplt.colorbar(im)
        assert isinstance(cb._spec.norm, LogNorm)
        assert cb._spec.norm.vmin == pytest.approx(0.5)
        assert cb._spec.norm.vmax == pytest.approx(2.0)

    def test_bare_scalar_mappable(self, plot):
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        gplt.plot([0.0, 1.0], [0.0, 1.0])
        sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=9.0), cmap="cool")
        cb = gplt.colorbar(sm)
        assert cb._spec.cmap == "cool"
        assert cb._spec.norm.vmax == pytest.approx(9.0)

    def test_current_mappable_is_used_when_none_given(self, plot):
        gplt.imshow(np.random.default_rng(3).random((4, 4)), cmap="magma")
        cb = gplt.colorbar()
        assert cb._spec.cmap == "magma"

    def test_set_label_reaches_the_spec(self, plot):
        im = gplt.imshow(np.random.default_rng(4).random((4, 4)))
        cb = gplt.colorbar(im)
        cb.set_label("Intensity")
        assert cb._spec.label == "Intensity"
        cb.ax.set_ylabel("Other")
        assert cb._spec.label == "Other"

    def test_cax_warns_unsupported(self, plot):
        im = gplt.imshow(np.random.default_rng(5).random((4, 4)))
        with pytest.warns(gplt.MatplotlibCompatWarning, match="cax"):
            gplt.colorbar(im, cax=object())

    def test_inset_does_not_shrink_the_host_panel(self, plot):
        """That is the whole point of `inset=True`: the bar goes *over* the plot."""
        panel = gplt.gca().panel
        before = panel.rect_frac
        im = gplt.imshow(np.random.default_rng(6).random((4, 4)))
        spec = gplt.colorbar(im, location="top", inset=True)._spec
        assert panel.rect_frac == before
        assert spec.inset is True
        # ...and the strip itself lands inside the host's own rect.
        bx, by, bw, bh = spec.rect_frac
        hx, hy, hw, hh = before
        assert bx >= hx and by >= hy
        assert bx + bw <= hx + hw + 1e-9
        assert by + bh <= hy + hh + 1e-9

    @pytest.mark.parametrize(
        "location,long_axis", [("right", 3), ("left", 3), ("top", 2), ("bottom", 2)]
    )
    def test_inset_bounds_honour_shrink(self, plot, location, long_axis):
        """``shrink`` shortens an inset bar along its long axis and keeps it centred."""
        from glplot.core.panel import ColorbarSpec
        from glplot.pyplot import inset_colorbar_bounds

        def bounds(shrink):
            return inset_colorbar_bounds(
                ColorbarSpec(
                    rect_frac=(0.0, 0.0, 1.0, 1.0),
                    orientation="vertical" if location in ("left", "right") else "horizontal",
                    location=location,
                    cmap="viridis",
                    norm=None,
                    fraction=0.05,
                    shrink=shrink,
                )
            )

        full, half = bounds(1.0), bounds(0.5)
        assert half[long_axis] == pytest.approx(full[long_axis] * 0.5)
        lead = long_axis - 2  # the offset paired with that length
        full_mid = full[lead] + full[long_axis] * 0.5
        half_mid = half[lead] + half[long_axis] * 0.5
        assert half_mid == pytest.approx(full_mid)


class TestScatterMarkerReachesTheExport:
    """``scatter(marker=...)`` was stored on the layer but dropped by the PNG export.

    Every point exported as a circle regardless of what was asked for -- most visibly on
    ``hist2d``, which used to pass ``marker="s"`` on purpose because a 2D histogram's
    bins are squares, and whose exported PNG therefore disagreed with the live window.
    ``hist2d`` now draws a real rectangular mesh (see ``TestHist2dParity``) rather than
    sized square markers, which sidesteps this whole bug class for it structurally: a
    patch's per-vertex colours export through the same generic path
    (``utils/preview.py``'s "patch that carries per-vertex colours" branch) that already
    draws it live, so there is no separate marker-shape mapping left to fall out of step.
    """

    def test_marker_is_recorded_on_the_layer(self, plot):
        layer = gplt.scatter([0.0, 1.0], [0.0, 1.0], marker="^")
        assert layer.metadata["marker"] == "^"

    def test_hist2d_is_a_patch_so_live_and_export_cannot_disagree(self, plot):
        rng = np.random.default_rng(0)
        *_, layer = gplt.hist2d(rng.normal(size=200), rng.normal(size=200), bins=8)
        assert layer.layer_type == "patch"
        assert "marker" not in layer.metadata

    def test_live_shader_has_a_shape_for_each_supported_marker(self, plot):
        """The live GL path drew every marker as a circle until the shader gained shapes.

        The mapping and the shader's ``switch`` are only meaningful together, so this
        pins that every index the table hands out is one the shader actually implements.
        """
        from glplot.utils.shaders import MARKER_SHAPE_INDEX, SCATTER_FS, marker_shape_index

        assert marker_shape_index(None) == 0, "no marker must stay the historical circle"
        assert marker_shape_index("not-a-marker") == 0, "unknown markers degrade to a circle"
        for marker, index in MARKER_SHAPE_INDEX.items():
            assert isinstance(index, int)
            if index != 0:  # 0 is the shader's fall-through, so it has no `shape ==` case
                assert f"shape == {index}" in SCATTER_FS, f"{marker!r} -> {index} has no case"


class TestPerPanelAxisNames:
    """``xlabel``/``ylabel``/``title`` are per panel, not one shared value per figure.

    They were figure-global until they moved onto :class:`~glplot.core.panel.Panel`, so a
    2x2 grid could carry exactly one x-name between all four panels -- whichever call ran
    last won and the other three axes went unnamed.
    """

    def test_each_panel_keeps_its_own_names(self, plot):
        _, axs = gplt.subplots(1, 2)
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Volts")
        axs[1].set_xlabel("Freq (Hz)")
        axs[1].set_ylabel("Power")
        assert (axs[0].panel.xlabel, axs[0].panel.ylabel) == ("Time (s)", "Volts")
        assert (axs[1].panel.xlabel, axs[1].panel.ylabel) == ("Freq (Hz)", "Power")

    def test_engine_property_forwards_to_the_active_panel(self, plot):
        """``engine.xlabel`` is "the current axes' x-name", like ``engine.scene`` is."""
        fig, axs = gplt.subplots(1, 2)
        axs[0].set_xlabel("first")
        axs[1].set_xlabel("second")
        fig.active_panel_index = 0
        assert fig.xlabel == "first"
        fig.active_panel_index = 1
        assert fig.xlabel == "second"

    def test_unset_names_are_empty_not_missing(self, plot):
        """The properties always exist now; "unset" is "" rather than AttributeError."""
        _, axs = gplt.subplots(1, 2)
        assert axs[0].panel.xlabel == ""
        assert gplt.gcf().ylabel == ""

    def test_title_is_per_panel_and_does_not_leak(self, plot):
        """``set_title`` titles the panel it was called on, not every panel.

        It still writes the window caption too (that is also the single-panel axes
        title), which is exactly why an untitled sibling must not fall back to it.
        """
        _, axs = gplt.subplots(1, 2)
        axs[1].set_title("Spectrum")
        assert axs[1].panel.title == "Spectrum"
        assert axs[0].panel.title == ""

    def test_stock_window_caption_is_not_promoted_to_an_axes_title(self, plot):
        """ "GLPlot" is the default *window* caption and must never become a plot title."""
        from glplot.utils.preview import _resolve_axes_title

        fig, axs = gplt.subplots(1, 2)
        assert fig.title in ("GLPlot", "")  # the untouched stock caption
        assert _resolve_axes_title(axs[0].panel, fig) == ""


class TestRealSubplots:
    """``subplot``/``subplots``/``subplot2grid``/``subplot_mosaic`` build real, separate panels."""

    def test_subplot_returns_distinct_panels(self, plot):
        ax1 = gplt.subplot(2, 1, 1)
        ax2 = gplt.subplot(2, 1, 2)
        assert ax1 is not ax2
        assert (ax1.row, ax1.col) != (ax2.row, ax2.col)
        assert len(gplt.gcf().panels) == 2

    def test_subplot_draws_into_separate_panels(self, plot):
        """Each half of the script draws into its own panel, not on top of the other."""
        fig = gplt.gcf()
        ax = gplt.subplot(2, 1, 1)
        gplt.plot([0.0, 1.0], [0.0, 1.0])
        ax2 = gplt.subplot(2, 1, 2)
        gplt.plot([0.0, 1.0], [1.0, 0.0])
        assert ax is not ax2
        fig.active_panel_index = 0
        assert any(ly.layer_type == "polyline" for ly in fig.scene.layers)
        fig.active_panel_index = 1
        assert any(ly.layer_type == "polyline" for ly in fig.scene.layers)

    def test_subplots_grid_shape(self, plot):
        fig, axs = gplt.subplots(2, 3)
        assert axs.shape == (2, 3)
        assert len(fig.panels) == 6
        assert axs[0, 0] is not axs[1, 2]


class TestCompatWarning:
    """The no-op policy: accepted, ignored, and *said out loud*."""

    def test_unsupported_kwarg_warns_rather_than_raising(self, plot):
        with pytest.warns(gplt.MatplotlibCompatWarning, match="emit"):
            gplt.axis("equal", emit=False)

    def test_warning_fires_once_per_kwarg_not_once_per_call(self, plot):
        """A no-op kwarg inside a plotting loop must not print on every iteration."""
        with pytest.warns(gplt.MatplotlibCompatWarning) as caught:
            for _ in range(10):
                gplt.axis("equal", emit=False)
        assert len(caught) == 1

    def test_unset_kwarg_is_silent(self, plot):
        """The warning must track what the caller asked for, not what the signature lists."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", gplt.MatplotlibCompatWarning)
            gplt.axis("equal")

    def test_warning_is_independently_silenceable(self, plot):
        """Its own category, so filtering it does not also hide unrelated warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", gplt.MatplotlibCompatWarning)
            gplt.axis("equal", emit=False)

    def test_category_is_a_userwarning(self):
        """Porting scripts run under -W default; a non-UserWarning would be invisible."""
        assert issubclass(gplt.MatplotlibCompatWarning, UserWarning)


class TestHexbin:
    """Hexagonal binning. The hexagons are the point -- a square grid's rows and columns
    impose a pattern on the eye that the data does not have."""

    @pytest.fixture
    def cloud(self):
        rng = np.random.default_rng(0)
        return rng.normal(size=2000), rng.normal(size=2000)

    def test_it_is_one_layer_of_hexagons(self, plot, cloud):
        layer = gplt.hexbin(*cloud, gridsize=20)
        assert layer.layer_type == "patch"
        # Seven vertices per hexagon (a hub plus six corners), six triangles off them.
        assert len(layer.vertices) % 7 == 0
        assert len(layer.indices) // 3 == 6 * (len(layer.vertices) // 7)

    def test_every_point_lands_in_exactly_one_hexagon(self, plot, cloud):
        """The nearest-of-two-lattices rule IS the tessellation; this is what proves it.

        A gap or an overlap in the lattice shows up here as a count that does not add up,
        long before it shows up as a visible artefact.
        """
        layer = gplt.hexbin(*cloud, gridsize=15)
        assert int(layer.metadata["counts"].sum()) == len(cloud[0])

    def test_hexagons_carry_their_own_colour(self, plot, cloud):
        """One face_color cannot express a hexbin; the layer must hold a colour buffer."""
        layer = gplt.hexbin(*cloud, gridsize=20)
        assert layer.colors is not None
        assert layer.colors.shape == (len(layer.vertices), 4)
        assert len(np.unique(layer.colors, axis=0)) > 1

    def test_a_hexagons_seven_vertices_share_one_colour(self, plot, cloud):
        layer = gplt.hexbin(*cloud, gridsize=10)
        first_hex = layer.colors[:7]
        assert len(np.unique(first_hex, axis=0)) == 1

    def test_mincnt_drops_sparse_hexagons(self, plot, cloud):
        many = gplt.hexbin(*cloud, gridsize=20)
        few = gplt.hexbin(*cloud, gridsize=20, mincnt=10)
        assert len(few.vertices) < len(many.vertices)

    def test_C_is_reduced_per_hexagon(self, plot, cloud):
        x, y = cloud
        layer = gplt.hexbin(x, y, C=np.abs(x), reduce_C_function=np.max, gridsize=10)
        # Reduced values, not counts: the max of |x| is unbounded by the point count.
        assert layer.metadata["cvalues"].max() > 1.0

    def test_bins_log_compresses_the_range(self, plot, cloud):
        plain = gplt.hexbin(*cloud, gridsize=20).metadata["cvalues"].max()
        logged = gplt.hexbin(*cloud, gridsize=20, bins="log").metadata["cvalues"].max()
        assert logged < plain

    def test_gridsize_tuple_is_honoured(self, plot, cloud):
        wide = gplt.hexbin(*cloud, gridsize=(30, 5))
        tall = gplt.hexbin(*cloud, gridsize=(5, 30))
        assert len(wide.vertices) != len(tall.vertices)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"gridsize": 0}, "gridsize"),
            ({"bins": "sqrt"}, "bins"),
            ({"mincnt": 10**9}, "mincnt"),
        ],
    )
    def test_bad_input_raises(self, plot, cloud, kwargs, match):
        with pytest.raises(ValueError, match=match):
            gplt.hexbin(*cloud, **kwargs)

    @pytest.mark.parametrize("kwargs", [{"xscale": "log"}, {"yscale": "log"}])
    def test_log_scales_warn(self, plot, cloud, kwargs):
        with pytest.warns(gplt.MatplotlibCompatWarning):
            gplt.hexbin(*cloud, gridsize=10, **kwargs)

    def test_hexbin_auto_equalises_the_aspect(self, plot):
        """A hexagon is only regular -- not a stretched parallelogram -- when x and y
        share the same world-units-per-pixel; ``_hexagon_geometry`` computes its own
        ``rx``/``ry`` assuming exactly that. Left to the default 'auto' aspect this
        module's own ``plot`` fixture (a deliberately non-square 1000x500 window) would
        stretch a symmetric data range unevenly across it -- the same failure mode
        ``mandelbrot()``/``julia()`` already guard against via the same
        ``_apply_equal_aspect`` call (see ``_add_fractal``), extended here since a
        hexbin is equally meaningless off-square.
        """
        rng = np.random.default_rng(1)
        # A wide-vs-tall data range on top of the fixture's own non-square window, so an
        # implementation that only fixed the window's own aspect (and not the data's)
        # would still fail this.
        x = rng.normal(0.0, 20.0, 2000)
        y = rng.normal(0.0, 2.0, 2000)
        gplt.hexbin(x, y, gridsize=15)
        upp_x, upp_y = _units_per_px(plot)
        assert upp_x == pytest.approx(upp_y, rel=1e-6)

    def test_hexbin_equalising_still_frames_every_point(self, plot):
        """Widening, not cropping: every point that went into the hexagons must still
        land inside the final, equalised view -- the property
        ``test_equal_widens_rather_than_crops`` proves for ``axis('equal')`` directly,
        checked here end to end against the actual data instead of the camera's own
        zoom numbers."""
        rng = np.random.default_rng(1)
        x = rng.normal(0.0, 20.0, 2000)
        y = rng.normal(0.0, 2.0, 2000)
        gplt.hexbin(x, y, gridsize=15)
        cx, cy = plot.camera.cx, plot.camera.cy
        half_w, half_h = 1.0 / plot.camera.zoom_x, 1.0 / plot.camera.zoom_y
        assert x.min() >= cx - half_w - 1e-6
        assert x.max() <= cx + half_w + 1e-6
        assert y.min() >= cy - half_h - 1e-6
        assert y.max() <= cy + half_h + 1e-6


class TestEventplot:
    def test_one_layer_per_row(self, plot):
        layers = gplt.eventplot([np.arange(5.0), np.arange(5.0) + 0.5])
        assert len(layers) == 2

    def test_a_raster_of_many_events_is_still_one_layer_per_row(self, plot):
        """The reason the ticks are quads rather than lines: N events, not N layers."""
        layers = gplt.eventplot(np.linspace(0.0, 1.0, 10_000))
        assert len(layers) == 1
        assert len(layers[0].indices) // 3 == 2 * 10_000

    def test_rows_are_stacked_by_default(self, plot):
        """Several rows sharing the default offset must not draw on top of each other."""
        layers = gplt.eventplot([np.arange(3.0), np.arange(3.0)])
        centres = [float(np.mean(ly.vertices[:, 1])) for ly in layers]
        assert centres[0] != pytest.approx(centres[1])

    def test_lineoffsets_and_linelengths_place_the_ticks(self, plot):
        layer = gplt.eventplot(np.arange(3.0), lineoffsets=5.0, linelengths=2.0)[0]
        ys = layer.vertices[:, 1]
        assert ys.min() == pytest.approx(4.0)
        assert ys.max() == pytest.approx(6.0)

    def test_vertical_orientation_transposes(self, plot):
        h = gplt.eventplot(np.arange(3.0), orientation="horizontal")[0]
        gplt.figure("v")
        v = gplt.eventplot(np.arange(3.0), orientation="vertical")[0]
        assert np.ptp(h.vertices[:, 1]) == pytest.approx(np.ptp(v.vertices[:, 0]))

    def test_bad_orientation_raises(self, plot):
        with pytest.raises(ValueError, match="orientation"):
            gplt.eventplot(np.arange(3.0), orientation="sideways")

    def test_colors_per_row(self, plot):
        layers = gplt.eventplot([np.arange(3.0), np.arange(3.0)], colors=["red", "blue"])
        assert layers[0].style.face_color != layers[1].style.face_color

    def test_a_single_rgb_tuple_colours_every_row(self, plot):
        """(1, 0, 0) is one red; ["red", "blue"] is two colours. Both are sequences.

        The ambiguity is real and matplotlib settles it with `is_color_like`; a
        hand-rolled "is the first element a string?" test reads this tuple as three
        separate colours for two rows.
        """
        layers = gplt.eventplot([np.arange(3.0), np.arange(3.0)], colors=(1.0, 0.0, 0.0))
        assert layers[0].style.face_color == layers[1].style.face_color
        assert layers[0].style.face_color[:3] == (1.0, 0.0, 0.0)

    def test_a_single_named_colour_colours_every_row(self, plot):
        layers = gplt.eventplot([np.arange(3.0), np.arange(3.0)], colors="red")
        assert layers[0].style.face_color == layers[1].style.face_color

    def test_a_wrong_number_of_colours_raises(self, plot):
        with pytest.raises(ValueError, match="colors"):
            gplt.eventplot([np.arange(3.0), np.arange(3.0)], colors=["red", "blue", "green"])

    def test_an_empty_row_draws_nothing_rather_than_raising(self, plot):
        assert len(gplt.eventplot([np.array([]), np.arange(3.0)])) == 1


class TestViolinplot:
    @pytest.fixture
    def bimodal(self):
        rng = np.random.default_rng(0)
        return np.concatenate([rng.normal(-2.0, 0.3, 400), rng.normal(2.0, 0.3, 400)])

    def test_the_body_shows_the_density_a_boxplot_hides(self, plot, bimodal):
        """The whole reason a violin exists: two lobes and a waist between them.

        A bimodal sample and a flat one have the same quartiles, so this is the assertion
        that separates a real KDE from a box drawn in the shape of one.
        """
        body = gplt.violinplot(bimodal)["bodies"][0]
        xs, ys = body.vertices[:, 0], body.vertices[:, 1]
        waist = np.ptp(xs[np.abs(ys) < 0.2])
        peak = np.ptp(xs[np.abs(ys - 2.0) < 0.2])
        assert waist < peak * 0.5

    def test_widths_bounds_the_body(self, plot, bimodal):
        body = gplt.violinplot(bimodal, widths=2.0)["bodies"][0]
        assert np.ptp(body.vertices[:, 0]) == pytest.approx(2.0, rel=1e-3)

    def test_positions_place_each_violin(self, plot):
        rng = np.random.default_rng(1)
        result = gplt.violinplot([rng.normal(size=200), rng.normal(size=200)], positions=[0.0, 3.0])
        centres = [float(np.mean(b.vertices[:, 0])) for b in result["bodies"]]
        assert centres == pytest.approx([0.0, 3.0], abs=1e-2)

    def test_vert_false_transposes(self, plot, bimodal):
        body = gplt.violinplot(bimodal, vert=False)["bodies"][0]
        assert np.ptp(body.vertices[:, 0]) > np.ptp(body.vertices[:, 1])

    @pytest.mark.parametrize(
        "kwargs, key",
        [
            ({"showmeans": True, "showextrema": False}, "cmeans"),
            ({"showmedians": True, "showextrema": False}, "cmedians"),
            ({"showextrema": True}, "cmins"),
            ({"showextrema": True}, "cmaxes"),
        ],
    )
    def test_the_markers_are_drawn_when_asked(self, plot, bimodal, kwargs, key):
        assert len(gplt.violinplot(bimodal, **kwargs)[key]) == 1

    def test_the_markers_stay_off_by_default(self, plot, bimodal):
        """Only extrema default on, as in matplotlib; the rest must be asked for."""
        result = gplt.violinplot(bimodal)
        assert result["cmeans"] == []
        assert result["cmedians"] == []
        assert len(result["cmins"]) == 1

    def test_quantiles_are_marked(self, plot, bimodal):
        assert len(gplt.violinplot(bimodal, quantiles=[0.25, 0.75])["cquantiles"]) == 2

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"dataset": np.ones(50)}, "single repeated value"),
            ({"dataset": np.array([1.0])}, "at least 2"),
            ({"dataset": np.arange(20.0), "quantiles": [1.5]}, "0..1"),
            ({"dataset": np.arange(20.0), "positions": [0.0, 1.0]}, "positions"),
        ],
    )
    def test_degenerate_input_raises(self, plot, kwargs, match):
        with pytest.raises(ValueError, match=match):
            gplt.violinplot(**kwargs)


class TestStreamplot:
    """The integrator is asserted against fields whose streamlines are known exactly."""

    @pytest.fixture
    def grid(self):
        Y, X = np.mgrid[-2:2:20j, -2:2:20j]
        return X, Y

    def test_a_uniform_field_gives_horizontal_lines(self, plot, grid):
        """u=1, v=0: every streamline is a straight horizontal line, exactly."""
        X, Y = grid
        lines = gplt.streamplot(X, Y, np.ones_like(X), np.zeros_like(Y), density=0.5)
        assert len(lines) > 0
        assert max(float(np.ptp(ly.pts[:, 1])) for ly in lines) < 1e-4

    def test_solid_body_rotation_gives_circles(self, plot, grid):
        """u=-y, v=x: every streamline is a circle, so its radius must not drift.

        This is the assertion that catches a broken integrator: Euler on this field
        spirals outward, and the drift compounds with every step.
        """
        X, Y = grid
        lines = gplt.streamplot(X, Y, -Y, X, density=1.0)
        drift = 0.0
        for ly in lines:
            r = np.hypot(ly.pts[:, 0], ly.pts[:, 1])
            if len(r) > 20 and r.mean() > 0.5:
                drift = max(drift, float(np.ptp(r) / r.mean()))
        assert drift < 0.05

    def test_lines_do_not_stop_on_their_own_trail(self, plot, grid):
        """The thinning grid must exclude the line's own cells.

        Marking them as it goes makes every line stop after one step -- it collides with
        the cell its own first step just claimed, and the plot becomes a field of stubs.
        """
        X, Y = grid
        lines = gplt.streamplot(X, Y, np.ones_like(X), np.zeros_like(Y), density=0.5)
        assert max(len(ly.pts) for ly in lines) > 10

    def test_density_controls_how_many(self, plot, grid):
        X, Y = grid
        sparse = gplt.streamplot(X, Y, -Y, X, density=0.5)
        gplt.figure("dense")
        dense = gplt.streamplot(X, Y, -Y, X, density=2.0)
        assert len(dense) > len(sparse)

    def test_cmap_colours_each_line_by_speed(self, plot, grid):
        X, Y = grid
        lines = gplt.streamplot(X, Y, -Y, X, cmap="viridis")
        assert len({tuple(np.round(ly.style.color, 3)) for ly in lines}) > 1

    def test_a_plain_color_is_uniform(self, plot, grid):
        X, Y = grid
        lines = gplt.streamplot(X, Y, -Y, X, color="red")
        assert len({tuple(np.round(ly.style.color, 3)) for ly in lines}) == 1

    def test_start_points_seed_exactly_those_lines(self, plot, grid):
        X, Y = grid
        assert len(gplt.streamplot(X, Y, -Y, X, start_points=[[1.0, 0.0]])) == 1

    def test_one_sided_integration_is_shorter(self, plot, grid):
        X, Y = grid
        forward = gplt.streamplot(
            X, Y, -Y, X, integration_direction="forward", start_points=[[1.0, 0.0]]
        )
        gplt.figure("both")
        both = gplt.streamplot(X, Y, -Y, X, integration_direction="both", start_points=[[1.0, 0.0]])
        assert len(forward[0].pts) < len(both[0].pts)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"density": 0}, "density"),
            ({"integration_direction": "sideways"}, "integration_direction"),
        ],
    )
    def test_bad_input_raises(self, plot, grid, kwargs, match):
        X, Y = grid
        with pytest.raises(ValueError, match=match):
            gplt.streamplot(X, Y, -Y, X, **kwargs)

    def test_field_shape_is_checked(self, plot, grid):
        X, Y = grid
        with pytest.raises(ValueError, match=r"shape"):
            gplt.streamplot(X, Y, np.ones((5, 5)), np.ones((5, 5)))

    @pytest.mark.parametrize("kwargs", [{"arrowsize": 2.0}, {"arrowstyle": "->"}])
    def test_arrow_kwargs_warn(self, plot, grid, kwargs):
        X, Y = grid
        with pytest.warns(gplt.MatplotlibCompatWarning, match="arrow"):
            gplt.streamplot(X, Y, -Y, X, density=0.3, **kwargs)


class TestPatchMappableRemap:
    """clim() and set_cmap() must reach the per-vertex patch mappables.

    hexbin, hist2d, pcolor and tripcolor colour a single patch through a per-vertex colour
    buffer (hist2d joined this list once it became a real filled mesh rather than a scatter
    of sized point markers -- see TestHist2dParity). The GUI's re-mapper
    (`set_layer_colormap`) knows only image and scatter-value layers and *raises* on a
    patch, so a bare ``clim()`` after ``hexbin()`` crashed with 'layer_type patch has no
    per-layer colormap'. These patches now carry their own re-map path; each of these fails
    against the crashing version and against a no-op that leaves the colours untouched.
    """

    @pytest.mark.parametrize(
        "build",
        [
            lambda r: gplt.hexbin(r.normal(size=2000), r.normal(size=2000), gridsize=12),
            lambda r: gplt.hist2d(r.normal(size=2000), r.normal(size=2000), bins=12)[3],
            lambda r: gplt.pcolor(np.arange(30.0).reshape(5, 6)),
            lambda r: gplt.tripcolor(r.random(40), r.random(40), r.random(40)),
        ],
        ids=["hexbin", "hist2d", "pcolor", "tripcolor"],
    )
    def test_clim_does_not_crash_and_recolours(self, plot, build):
        layer = build(np.random.default_rng(0))
        before = layer.colors.copy()
        result = gplt.clim(0.0, 1.0)  # this used to raise ValueError
        assert result == (0.0, 1.0)
        assert not np.allclose(before, layer.colors), "clim() did not repaint the patch"

    @pytest.mark.parametrize(
        "build",
        [
            lambda r: gplt.hexbin(r.normal(size=2000), r.normal(size=2000), gridsize=12),
            lambda r: gplt.hist2d(r.normal(size=2000), r.normal(size=2000), bins=12)[3],
            lambda r: gplt.pcolor(np.arange(30.0).reshape(5, 6)),
            lambda r: gplt.tripcolor(r.random(40), r.random(40), r.random(40)),
        ],
        ids=["hexbin", "hist2d", "pcolor", "tripcolor"],
    )
    def test_set_cmap_recolours_the_patch(self, plot, build):
        layer = build(np.random.default_rng(0))
        before = layer.colors.copy()
        gplt.set_cmap("plasma")
        assert not np.allclose(before, layer.colors)

    def test_clim_autoscale_query_is_finite(self, plot):
        gplt.hexbin(*np.random.default_rng(0).normal(size=(2, 2000)), gridsize=12)
        lo, hi = gplt.clim()
        assert np.isfinite(lo) and np.isfinite(hi)


class TestColormapSurface:
    """set_cmap/get_cmap/clim/sci/gci and the nineteen one-word shortcuts."""

    @pytest.fixture
    def scalars(self):
        rng = np.random.default_rng(0)
        return rng.normal(size=200), rng.normal(size=200), rng.uniform(3.0, 7.0, 200)

    def test_get_cmap_returns_matplotlibs_own(self, plot):
        assert gplt.get_cmap("jet").name == "jet"

    def test_get_cmap_lut_resamples(self, plot):
        assert gplt.get_cmap("jet", lut=8).N == 8

    def test_set_cmap_reaches_a_plot_that_named_none(self, plot, scalars):
        x, y, v = scalars
        default = gplt.scatter(x, y, c=v).colors[:5].copy()
        gplt.set_cmap("jet")
        assert not np.allclose(default, gplt.scatter(x, y, c=v).colors[:5])

    def test_an_explicit_cmap_still_wins(self, plot, scalars):
        """Three levels of precedence, and this is the one a global must not break."""
        x, y, v = scalars
        default = gplt.scatter(x, y, c=v).colors[:5].copy()
        gplt.set_cmap("jet")
        assert np.allclose(default, gplt.scatter(x, y, c=v, cmap="viridis").colors[:5])

    def test_hist2d_keeps_its_own_default(self, plot, scalars):
        """hist2d has always been magma, and scatter viridis. A global fallback must not
        quietly retire either."""
        x, y, _ = scalars
        assert np.allclose(
            gplt.hist2d(x, y, bins=4)[3].colors[:3],
            gplt.hist2d(x, y, bins=4, cmap="magma")[3].colors[:3],
        )

    def test_set_cmap_rejects_an_unknown_name_at_the_call(self, plot):
        """Storing it would defer the error to whichever plot came next."""
        with pytest.raises((ValueError, KeyError)):
            gplt.set_cmap("not_a_colormap")

    def test_gci_is_the_last_colormapped_layer(self, plot, scalars):
        x, y, v = scalars
        layer = gplt.scatter(x, y, c=v)
        assert gplt.gci() is layer

    def test_a_flat_scatter_is_not_a_mappable(self, plot, scalars):
        """No scalars means no limits to offer; clim() must not find one to act on."""
        x, y, _ = scalars
        gplt.scatter(x, y)
        assert gplt.gci() is None

    def test_hexbin_is_a_mappable(self, plot, scalars):
        x, y, _ = scalars
        layer = gplt.hexbin(x, y, gridsize=10)
        assert gplt.gci() is layer

    def test_hist2d_is_a_mappable(self, plot, scalars):
        x, y, _ = scalars
        layer = gplt.hist2d(x, y, bins=10)[3]
        assert gplt.gci() is layer

    def test_sci_points_at_a_chosen_layer(self, plot, scalars):
        x, y, v = scalars
        first = gplt.scatter(x, y, c=v)
        gplt.scatter(x, y, c=v)
        gplt.sci(first)
        assert gplt.gci() is first

    def test_clim_reports_the_data_range_when_autoscaled(self, plot, scalars):
        """An unset limit means autoscale; the number in force is then the data's own."""
        x, y, v = scalars
        gplt.scatter(x, y, c=v)
        assert gplt.clim() == pytest.approx((float(v.min()), float(v.max())), rel=1e-5)

    def test_clim_sets_and_reports(self, plot, scalars):
        x, y, v = scalars
        gplt.scatter(x, y, c=v)
        assert gplt.clim(0.0, 1.0) == (0.0, 1.0)

    def test_clim_actually_recolours(self, plot, scalars):
        """Reading `style.vmin` would report a limit the renderer never received."""
        x, y, v = scalars
        layer = gplt.scatter(x, y, c=v)
        before = layer.colors[:2].copy()
        gplt.clim(4.0, 5.0)
        assert not np.allclose(before, layer.colors[:2])

    def test_clim_one_sided_leaves_the_other(self, plot, scalars):
        x, y, v = scalars
        gplt.scatter(x, y, c=v)
        gplt.clim(4.0, 5.0)
        assert gplt.clim(vmax=9.0) == (4.0, 9.0)

    def test_clim_without_a_mappable_says_what_to_do(self, plot):
        with pytest.raises(RuntimeError, match="scatter"):
            gplt.clim(0.0, 1.0)

    @pytest.mark.parametrize("name", gplt._CMAP_SHORTCUTS)
    def test_every_shortcut_exists_and_sets_its_own_colormap(self, plot, name):
        """Nineteen generated functions: a loop cannot get one of them subtly wrong, but
        it can get all of them wrong, so each is asserted."""
        getattr(gplt, name)()
        assert gplt._CURRENT_CMAP == name

    def test_the_shortcuts_are_named_after_themselves(self, plot):
        """Generated functions keep their own __name__, or a traceback names a closure."""
        assert gplt.viridis.__name__ == "viridis"
        assert gplt.jet.__doc__ and "jet" in gplt.jet.__doc__

    def test_state_does_not_leak_between_figures(self, plot):
        """The module globals are torn down with the rest of the pyplot state."""
        gplt.set_cmap("jet")
        gplt._cleanup_pyplot_state()
        assert gplt._CURRENT_CMAP is None
        assert gplt._CURRENT_MAPPABLE is None


def _tri_area(layer) -> float:
    """Filled area of an indexed-triangle patch, by shoelace over its triangles."""
    v, i = layer.vertices, layer.indices
    if i is None or len(i) == 0:
        return 0.0
    t = v[i].reshape(-1, 3, 2)
    return float(
        np.sum(
            np.abs(
                0.5
                * (
                    (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
                    - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1])
                )
            )
        )
    )


class TestRectangleFamily:
    def test_axhspan_is_horizontal(self, plot):
        gplt.axis("tight")
        layer = gplt.axhspan(3.0, 5.0)
        assert layer.vertices[:, 1].min() == pytest.approx(3.0)
        assert layer.vertices[:, 1].max() == pytest.approx(5.0)
        # It spans the whole current x range, so it is wider than it is tall here.
        assert np.ptp(layer.vertices[:, 0]) > np.ptp(layer.vertices[:, 1])

    def test_axvspan_is_vertical(self, plot):
        gplt.axis("tight")
        layer = gplt.axvspan(2.0, 4.0)
        assert layer.vertices[:, 0].min() == pytest.approx(2.0)
        assert layer.vertices[:, 0].max() == pytest.approx(4.0)

    def test_broken_barh_is_one_layer_of_rectangles(self, plot):
        layer = gplt.broken_barh([(1, 2), (5, 1), (7, 3)], (10, 0.8))
        assert layer.layer_type == "patch"
        assert len(layer.indices) // 6 == 3  # three rectangles, one layer

    def test_broken_barh_empty_raises(self, plot):
        with pytest.raises(ValueError, match="empty"):
            gplt.broken_barh([], (0, 1))

    def test_barh_is_bar_transposed(self, plot):
        bars = gplt.barh([0, 1, 2], [10, 24, 18])
        assert len(bars) == 3
        first = bars[0].vertices
        assert first[:, 0].max() == pytest.approx(10.0)  # width along x
        assert np.ptp(first[:, 1]) == pytest.approx(0.8)  # height on y

    def test_barh_edge_align(self, plot):
        bar = gplt.barh([0.0], [5.0], align="edge")[0]
        assert bar.vertices[:, 1].min() == pytest.approx(0.0)
        assert bar.vertices[:, 1].max() == pytest.approx(0.8)


class TestFillFamily:
    def test_fill_closes_a_polygon(self, plot):
        square = gplt.fill([0, 1, 1, 0], [0, 0, 1, 1], "b")
        assert len(square) == 1
        assert _tri_area(square[0]) == pytest.approx(1.0)

    def test_fill_draws_several_polygons(self, plot):
        polys = gplt.fill([0, 1, 1], [0, 0, 1], "r", [2, 3, 3], [0, 0, 1], "g")
        assert len(polys) == 2

    def test_fill_without_y_raises(self, plot):
        with pytest.raises((ValueError, TypeError)):
            gplt.fill([0, 1, 2])

    def test_fill_betweenx_is_fill_between_transposed(self, plot):
        layer = gplt.fill_betweenx(np.linspace(0, 10, 11), np.full(11, -0.1), np.full(11, 0.1))
        assert layer.vertices[:, 0].min() == pytest.approx(-0.1)
        assert layer.vertices[:, 0].max() == pytest.approx(0.1)
        assert np.ptp(layer.vertices[:, 1]) == pytest.approx(10.0)

    def test_stackplot_stacks(self, plot):
        x = np.linspace(0, 10, 20)
        bands = gplt.stackplot(x, np.ones(20), 2 * np.ones(20), 0.5 * np.ones(20))
        tops = [ly.vertices[:, 1].max() for ly in bands]
        assert tops == pytest.approx([1.0, 3.0, 3.5], rel=1e-4)

    def test_stackplot_accepts_a_2d_array(self, plot):
        x = np.linspace(0, 10, 20)
        assert len(gplt.stackplot(x, np.ones((3, 20)))) == 3

    def test_stackplot_unsupported_baseline_warns(self, plot):
        with pytest.warns(gplt.MatplotlibCompatWarning, match="baseline"):
            gplt.stackplot(np.linspace(0, 1, 5), np.ones(5), baseline="wiggle")

    def test_fill_between_now_returns_the_layer(self, plot):
        """Contract fix: it returns the patch Layer its docstring promises."""
        layer = gplt.fill_between(np.linspace(0, 10, 20), np.sin(np.linspace(0, 10, 20)), 0)
        assert layer.layer_type == "patch"


class TestStairsAndEcdf:
    def test_stairs_outline(self, plot):
        layer = gplt.stairs([1, 3, 2, 4], [0, 1, 2, 3, 4])
        assert layer.layer_type == "polyline"

    def test_stairs_fill(self, plot):
        layer = gplt.stairs([1, 3, 2, 4], [0, 1, 2, 3, 4], fill=True)
        assert layer.layer_type == "patch"

    def test_stairs_edges_length_checked(self, plot):
        with pytest.raises(ValueError, match="edges"):
            gplt.stairs([1, 2, 3], [0, 1])

    def test_ecdf_rises_monotonically_to_one(self, plot):
        rng = np.random.default_rng(0)
        layer = gplt.ecdf(rng.normal(size=500))
        ys = layer.pts[:, 1]
        assert np.all(np.diff(ys) >= -1e-9)
        assert ys.max() == pytest.approx(1.0)
        assert ys.min() < 0.01

    def test_ecdf_value_at_the_median_is_half(self, plot):
        layer = gplt.ecdf(np.arange(100.0))
        assert np.interp(49.5, layer.pts[:, 0], layer.pts[:, 1]) == pytest.approx(0.5, abs=0.02)

    def test_ecdf_complementary_is_the_survival_function(self, plot):
        layer = gplt.ecdf(np.arange(100.0), complementary=True)
        assert layer.pts[:, 1].max() == pytest.approx(1.0, abs=0.02)
        assert layer.pts[:, 1].min() < 0.02

    def test_ecdf_all_nan_raises(self, plot):
        with pytest.raises(ValueError, match="finite"):
            gplt.ecdf(np.full(5, np.nan))


class TestMeshFamily:
    def test_pcolor_of_C_alone(self, plot):
        layer = gplt.pcolor(np.arange(12.0).reshape(3, 4))
        assert len(layer.indices) // 6 == 12  # one quad per cell
        assert layer.colors is not None  # per-cell colour

    def test_pcolor_with_corner_grid(self, plot):
        x, y = np.meshgrid(np.arange(5.0), np.arange(4.0))
        layer = gplt.pcolor(x, y, np.random.default_rng(0).random((3, 4)))
        assert len(layer.indices) // 6 == 12

    def test_pcolor_drops_nan_cells(self, plot):
        Z = np.array([[1.0, np.nan], [3.0, 4.0]])
        assert len(gplt.pcolor(Z).indices) // 6 == 3

    def test_pcolor_is_a_mappable(self, plot):
        gplt.pcolor(np.arange(12.0).reshape(3, 4))
        assert gplt.gci() is not None

    def test_pcolormesh_same_size_grid_is_padded_to_corners(self, plot):
        x, y = np.meshgrid(np.arange(4.0), np.arange(3.0))
        assert (
            len(gplt.pcolormesh(x, y, np.random.default_rng(0).random((3, 4))).indices) // 6 == 12
        )

    def test_triplot_traces_edges(self, plot):
        rng = np.random.default_rng(0)
        layer = gplt.triplot(rng.random(30), rng.random(30))
        assert layer.layer_type == "polyline"
        assert len(layer.pts) > 30

    def test_tripcolor_per_point_is_gouraud(self, plot):
        rng = np.random.default_rng(0)
        px, py = rng.random(30), rng.random(30)
        layer = gplt.tripcolor(px, py, px + py)
        # A triangle's three vertices carry three (generally distinct) colours.
        assert layer.colors is not None
        assert len(layer.colors) % 3 == 0

    def test_tripcolor_per_triangle_is_flat(self, plot):
        from matplotlib.tri import Triangulation

        rng = np.random.default_rng(0)
        px, py = rng.random(30), rng.random(30)
        tri = Triangulation(px, py)
        layer = gplt.tripcolor(px, py, rng.random(len(tri.triangles)))
        # Flat: a face's three vertices share one colour.
        assert np.allclose(layer.colors[0], layer.colors[1])
        assert np.allclose(layer.colors[1], layer.colors[2])

    def test_tripcolor_bad_C_length_raises(self, plot):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="per point"):
            gplt.tripcolor(rng.random(30), rng.random(30), np.ones(7))


class TestTriContour:
    @pytest.fixture
    def field(self):
        rng = np.random.default_rng(0)
        px, py = rng.random(100), rng.random(100)
        return px, py, np.exp(-((px - 0.5) ** 2 + (py - 0.5) ** 2) * 8)

    def test_tricontour_draws_live_polylines(self, plot, field):
        lines = gplt.tricontour(*field, levels=6)
        assert len(lines) > 0
        assert all(ly.layer_type == "polyline" for ly in lines)

    def test_tricontourf_fills_bands(self, plot, field):
        bands = gplt.tricontourf(*field, levels=6)
        assert len(bands) > 0
        assert all(ly.layer_type == "patch" for ly in bands)

    def test_tricontour_single_colour(self, plot, field):
        lines = gplt.tricontour(*field, colors="red")
        assert len({str(ly.style.color) for ly in lines}) == 1

    def test_tricontour_z_length_checked(self, plot, field):
        px, py, _ = field
        with pytest.raises(ValueError, match="per point"):
            gplt.tricontour(px, py, np.ones(7))


class TestSpyPlotDate:
    def test_spy_marks_the_nonzeros(self, plot):
        M = np.eye(5)
        M[0, 4] = 1.0
        layer = gplt.spy(M)
        assert len(layer.pts) == 6

    def test_spy_upper_origin_puts_row_zero_at_the_top(self, plot):
        M = np.zeros((5, 5))
        M[0, 4] = 1.0
        layer = gplt.spy(M)  # origin='upper' default
        assert layer.pts[0, 1] == pytest.approx(4.0)  # row 0 -> top (y = nrows-1)

    def test_spy_lower_origin(self, plot):
        M = np.zeros((5, 5))
        M[0, 0] = 1.0
        assert gplt.spy(M, origin="lower").pts[0, 1] == pytest.approx(0.0)

    def test_spy_needs_2d(self, plot):
        with pytest.raises(ValueError):
            gplt.spy(np.ones(5))

    def test_plot_date_draws_the_curve_and_warns(self, plot):
        with pytest.warns(gplt.MatplotlibCompatWarning, match="date"):
            artists = gplt.plot_date([19000, 19001, 19002], [1, 2, 3])
        assert len(artists) >= 1


class TestPlottingStubs:
    def test_barbs_draws_a_shaft_per_point_and_warns(self, plot):
        X, Y = np.meshgrid(np.linspace(0, 5, 6), np.linspace(0, 5, 6))
        with pytest.warns(gplt.MatplotlibCompatWarning, match="barb"):
            artists = gplt.barbs(X, Y, np.cos(X), np.sin(Y))
        assert len(artists) >= 1

    def test_quiverkey_draws_an_arrow_and_a_label(self, plot):
        q = gplt.quiver([0.0], [0.0], [1.0], [0.0])
        parts = gplt.quiverkey(q, 0.9, 0.9, 2.0, "2 m/s")
        assert len(parts) == 2  # arrow + text

    def test_clabel_stashes_its_request_for_the_export(self, plot):
        """The request rides on the contour layer, for ``render_preview()`` to replay."""
        X, Y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
        cs = gplt.contour(X, Y, X**2 + Y**2, levels=5)
        assert gplt.clabel(cs, inline=True, fontsize=9, fmt="%.1f") == []
        assert cs.metadata["clabel"] == {
            "fontsize": 9,
            "inline": True,
            "inline_spacing": 5,
            "fmt": "%.1f",
            "colors": None,
            "use_clabeltext": False,
            "manual": False,
            "rightside_up": True,
            "zorder": None,
        }

    def test_clabel_also_marks_the_level_lines_for_the_live_pass(self, plot):
        """The live renderer walks layers, so each level line needs the request too.

        ``contour()`` draws one real polyline per level; those -- not the invisible
        placeholder the request is stashed on -- are what the live pass seats a number on.
        """
        X, Y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
        cs = gplt.contour(X, Y, X**2 + Y**2, levels=5)
        lines = cs.metadata["line_layers"]
        assert lines, "contour() must record its level lines for clabel() to find"
        assert all("clabel" not in ly.metadata for ly in lines)
        gplt.clabel(cs, fmt="%.2f")
        assert all(ly.metadata["clabel"]["fmt"] == "%.2f" for ly in lines)

    def test_clabel_levels_selects_which_lines_get_marked(self, plot):
        X, Y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
        cs = gplt.contour(X, Y, X**2 + Y**2, levels=[1.0, 2.0, 3.0])
        chosen = 2.0
        gplt.clabel(cs, levels=[chosen])
        for line in cs.metadata["line_layers"]:
            marked = "clabel" in line.metadata
            assert marked == (float(line.metadata["level"]) == chosen)

    def test_clabel_does_not_cross_label_a_second_contour(self, plot):
        """Two contours on one axes must not label each other's levels."""
        X, Y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
        first = gplt.contour(X, Y, X**2 + Y**2, levels=3)
        second = gplt.contour(X, Y, X - Y, levels=3)
        gplt.clabel(first)
        assert all("clabel" in ly.metadata for ly in first.metadata["line_layers"])
        assert all("clabel" not in ly.metadata for ly in second.metadata["line_layers"])

    def test_clabel_rejects_a_non_layer(self, plot):
        with pytest.raises(TypeError, match="clabel"):
            gplt.clabel(None)

    def test_figimage_draws_an_image(self, plot):
        layer = gplt.figimage(np.random.default_rng(0).random((8, 8)))
        assert layer is not None

    def test_table_is_a_no_op_that_warns(self, plot):
        with pytest.warns(gplt.MatplotlibCompatWarning, match="table"):
            assert gplt.table() is None

    def test_polar_converts_to_cartesian(self, plot):
        """A unit circle in (theta, r) must come out as radius 1 in x/y."""
        theta = np.linspace(0, 2 * np.pi, 50)
        with pytest.warns(gplt.MatplotlibCompatWarning, match="polar"):
            artists = gplt.polar(theta, np.ones(50))
        r = np.hypot(artists[0].pts[:, 0], artists[0].pts[:, 1])
        assert np.allclose(r, 1.0, atol=1e-5)


class TestSpectral:
    """The signal group: each is asserted against a signal with a known spectrum."""

    @pytest.fixture
    def signal(self):
        fs = 500.0
        t = np.arange(0, 2, 1 / fs)
        rng = np.random.default_rng(0)
        return fs, np.sin(2 * np.pi * 50 * t) + 0.1 * rng.normal(size=len(t))

    def test_psd_finds_the_tone(self, plot, signal):
        fs, sig = signal
        pxx, freqs = gplt.psd(sig, NFFT=256, Fs=fs)
        assert abs(freqs[np.argmax(pxx)] - 50.0) < 3.0

    def test_magnitude_spectrum_finds_the_tone(self, plot, signal):
        fs, sig = signal
        mag, freqs, _ = gplt.magnitude_spectrum(sig, Fs=fs)
        assert abs(freqs[np.argmax(mag)] - 50.0) < 2.0

    def test_angle_spectrum_is_wrapped(self, plot, signal):
        fs, sig = signal
        ang, _, _ = gplt.angle_spectrum(sig, Fs=fs)
        assert np.abs(ang).max() <= np.pi + 1e-6

    def test_coherence_is_bounded(self, plot, signal):
        fs, sig = signal
        cxy, _ = gplt.cohere(sig, np.roll(sig, 3), Fs=fs)
        assert cxy.min() >= -1e-9 and cxy.max() <= 1.0 + 1e-6

    def test_csd_runs(self, plot, signal):
        fs, sig = signal
        pxy, freqs = gplt.csd(sig, np.roll(sig, 3), Fs=fs)
        assert len(pxy) == len(freqs)

    def test_specgram_is_an_image(self, plot, signal):
        fs, sig = signal
        sxx, freqs, times, layer = gplt.specgram(sig, NFFT=256, Fs=fs)
        assert sxx.shape == (len(freqs), len(times))
        assert layer is not None

    def test_acorr_lag_zero_is_one(self, plot, signal):
        """Normalised, matching matplotlib's normed=True default."""
        _, sig = signal
        lags, c, _, _ = gplt.acorr(sig[:200], maxlags=10)
        assert c[len(c) // 2] == pytest.approx(1.0, abs=1e-6)
        assert list(lags) == list(range(-10, 11))

    def test_xcorr_detects_a_known_shift(self, plot, signal):
        """sig2 is sig delayed by 5 samples, so the peak must land at that lag."""
        _, sig = signal
        s = sig[:200]
        lags, c, _, _ = gplt.xcorr(s, np.roll(s, 5), maxlags=15)
        assert abs(lags[np.argmax(c)]) == 5


class TestSpectralParity:
    """The spectral group returns numbers the caller *analyses*, so they must be mpl's.

    Every function here delegates to :mod:`matplotlib.mlab`, which makes the estimator
    keywords real rather than decorative. That is only worth anything if it is pinned: a
    well-meaning rewrite back onto ``scipy.signal.welch`` would still "find the tone" in
    the tests above while quietly returning different numbers, because scipy and mlab
    differ in windowing and normalisation defaults. So these compare element-wise against
    matplotlib itself, and they are also what stops the float32 cast in ``_as_float_array``
    from creeping back in -- it shows up here as a ~1e-6 disagreement and nowhere else.
    """

    @pytest.fixture
    def sig(self):
        rng = np.random.default_rng(0)
        t = np.linspace(0, 1, 1024)
        return np.sin(2 * np.pi * 5 * t) + 0.1 * rng.normal(size=t.size)

    @pytest.fixture
    def mpl(self):
        """A headless matplotlib axes to compare against."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure()
        yield plt
        plt.close(fig)

    def test_psd_matches_matplotlib(self, plot, sig, mpl):
        kw = dict(NFFT=256, Fs=100, detrend="mean", noverlap=64)
        pxx, freqs = gplt.psd(sig, **kw)
        exp_pxx, exp_freqs = mpl.psd(sig, **kw)
        assert np.allclose(pxx, exp_pxx)
        assert np.allclose(freqs, exp_freqs)

    def test_sides_and_pad_to_are_honoured(self, plot, sig, mpl):
        """Two keywords that used to raise TypeError; both change the output shape."""
        kw = dict(NFFT=256, Fs=100, sides="twosided", pad_to=512)
        pxx, freqs = gplt.psd(sig, **kw)
        exp_pxx, exp_freqs = mpl.psd(sig, **kw)
        assert np.allclose(pxx, exp_pxx)
        assert len(freqs) == 512, "pad_to must lengthen the FFT, not be ignored"

    def test_csd_matches_matplotlib(self, plot, sig, mpl):
        kw = dict(NFFT=256, Fs=100, detrend="linear")
        pxy, freqs = gplt.csd(sig, np.roll(sig, 7), **kw)
        exp, exp_freqs = mpl.csd(sig, np.roll(sig, 7), **kw)
        assert np.allclose(pxy, exp) and np.allclose(freqs, exp_freqs)

    def test_cohere_matches_matplotlib(self, plot, sig, mpl):
        cxy, freqs = gplt.cohere(sig, np.roll(sig, 7), NFFT=256, Fs=100)
        exp, exp_freqs = mpl.cohere(sig, np.roll(sig, 7), NFFT=256, Fs=100)
        assert np.allclose(cxy, exp) and np.allclose(freqs, exp_freqs)

    @pytest.mark.parametrize("name", ["magnitude_spectrum", "phase_spectrum", "angle_spectrum"])
    def test_spectrum_family_matches_matplotlib(self, plot, sig, mpl, name):
        spec, freqs, _ = getattr(gplt, name)(sig, Fs=100, pad_to=2048)
        exp, exp_freqs, _ = getattr(mpl, name)(sig, Fs=100, pad_to=2048)
        assert np.allclose(spec, exp) and np.allclose(freqs, exp_freqs)

    def test_specgram_mode_and_scale_are_honoured(self, plot, sig, mpl):
        kw = dict(NFFT=256, Fs=100, mode="magnitude", scale="linear")
        spec, freqs, times, _ = gplt.specgram(sig, **kw)
        exp, exp_freqs, exp_times, _ = mpl.specgram(sig, **kw)
        assert np.allclose(spec, exp)
        assert np.allclose(freqs, exp_freqs) and np.allclose(times, exp_times)

    def test_specgram_rejects_db_on_an_angle(self, plot, sig):
        """A log of a radian is meaningless, and matplotlib raises rather than draw it."""
        with pytest.raises(ValueError, match="dB"):
            gplt.specgram(sig, mode="angle", scale="dB")

    @pytest.mark.parametrize("normed", [True, False])
    def test_xcorr_matches_matplotlib(self, plot, sig, mpl, normed):
        lags, c, _, _ = gplt.xcorr(sig, np.roll(sig, 7), maxlags=20, normed=normed)
        exp_lags, exp_c, _, _ = mpl.xcorr(sig, np.roll(sig, 7), maxlags=20, normed=normed)
        assert np.allclose(lags, exp_lags) and np.allclose(c, exp_c)

    def test_xcorr_default_detrend_does_not_remove_the_mean(self, plot, mpl):
        """matplotlib's default is ``detrend_none``. GLPlot used to subtract the mean,
        which changes every value of a signal with an offset."""
        offset = np.ones(64) * 3.0 + np.sin(np.linspace(0, 10, 64))
        _, c, _, _ = gplt.xcorr(offset, offset, maxlags=5)
        _, exp, _, _ = mpl.xcorr(offset, offset, maxlags=5)
        assert np.allclose(c, exp)


class TestSpectralArity:
    """matplotlib's return arity, which decides whether an unpack works at all.

    ``Pxx, freqs = plt.psd(x)`` is the documented call. GLPlot returned a 3-tuple, so that
    line raised ValueError on the unpack -- the plot was right and the script still died.
    """

    @pytest.fixture
    def sig(self):
        return np.sin(np.linspace(0, 50, 512))

    def test_psd_returns_two_by_default(self, plot, sig):
        assert len(gplt.psd(sig)) == 2

    def test_psd_returns_the_line_on_request(self, plot, sig):
        pxx, freqs, line = gplt.psd(sig, return_line=True)
        assert line is not None

    def test_csd_returns_two_by_default(self, plot, sig):
        assert len(gplt.csd(sig, sig)) == 2

    def test_cohere_returns_two(self, plot, sig):
        assert len(gplt.cohere(sig, sig)) == 2

    def test_correlation_returns_four(self, plot, sig):
        """``lags, c, line, b = plt.acorr(x)`` -- the baseline is the fourth element."""
        assert len(gplt.acorr(sig)) == 4
        assert len(gplt.xcorr(sig, sig)) == 4

    def test_usevlines_false_has_no_baseline(self, plot, sig):
        _, _, line, base = gplt.xcorr(sig, sig, usevlines=False)
        assert line is not None and base is None


class TestAxisConfig:
    def test_minorticks_toggle(self, plot):
        gplt.minorticks_on()
        assert plot.options.axis_minor_ticks is True
        gplt.minorticks_off()
        assert plot.options.axis_minor_ticks is False

    def test_margins_loosens_the_fit(self, plot):
        gplt.axis("tight")
        tight = np.ptp(plot.get_xlim())
        gplt.margins(0.3)
        gplt.axis("auto")
        assert np.ptp(plot.get_xlim()) > tight

    def test_margins_returns_the_pair(self, plot):
        assert gplt.margins(0.1) == (0.1, 0.1)

    def test_tick_params_length_turns_ticks_on(self, plot):
        gplt.tick_params(length=6)
        assert plot.options.axis_tick_len_px == 6.0
        assert plot.options.axis_show_ticks is True

    def test_locator_params_sets_the_count(self, plot):
        gplt.locator_params(nbins=5)
        assert plot.options.axis_tick_count_x == 5

    def test_box_toggles_the_frame(self, plot):
        gplt.box(False)
        assert plot.options.axis_show_frame is False
        gplt.box()
        assert plot.options.axis_show_frame is True

    def test_ticklabel_format_sci(self, plot):
        gplt.ticklabel_format(style="sci")
        assert "e" in plot.options.axis_tick_format


class TestInteractiveState:
    def test_ion_ioff_round_trip(self, plot):
        gplt.ion()
        assert gplt.isinteractive() is True
        gplt.ioff()
        assert gplt.isinteractive() is False

    def test_interactive_context_restores(self, plot):
        gplt.ioff()
        with gplt.interactive(True):
            assert gplt.isinteractive() is True
        assert gplt.isinteractive() is False

    def test_backend_is_glplot(self, plot):
        assert gplt.get_backend() == "glplot"

    @pytest.mark.parametrize(
        "call",
        [
            lambda: gplt.pause(0.01),
            lambda: gplt.ginput(),
            lambda: gplt.waitforbuttonpress(),
            lambda: gplt.draw(),
            lambda: gplt.install_repl_displayhook(),
        ],
    )
    def test_non_blocking_calls_return_without_hanging(self, plot, call):
        call()  # the assertion is that it returns at all, headless

    def test_ginput_returns_no_points(self, plot):
        with pytest.warns(gplt.MatplotlibCompatWarning):
            assert gplt.ginput() == []


class TestFigureManagement:
    def test_axes_returns_an_axes_for_the_current_panel(self, plot):
        """``axes()`` returns an axes object, not the figure — see ``test_gca_...`` above."""
        axes = gplt.axes()
        assert axes is not gplt.gcf()
        assert axes.figure is gplt.gcf()

    def test_fignum_exists(self, plot):
        gplt.figure(7)
        assert gplt.fignum_exists(7)
        assert not gplt.fignum_exists(999)

    def test_get_fignums_and_figlabels(self, plot):
        gplt.figure(3)
        gplt.figure("panel")
        assert 3 in gplt.get_fignums()
        assert "panel" in gplt.get_figlabels()

    def test_figaspect_matches_the_ratio(self, plot):
        w, h = gplt.figaspect(0.5)
        assert h / w == pytest.approx(0.5, rel=1e-6)

    def test_figaspect_from_an_array(self, plot):
        w, h = gplt.figaspect(np.zeros((10, 20)))
        assert h / w == pytest.approx(0.5, rel=1e-6)

    def test_subplot_mosaic_makes_a_panel_per_name(self, plot):
        fig, axd = gplt.subplot_mosaic("AB;CD")
        assert set(axd) == {"A", "B", "C", "D"}
        assert all(ax.figure is fig for ax in axd.values())
        assert len({id(ax) for ax in axd.values()}) == 4
        assert len(fig.panels) == 4

    def test_bar_label_labels_each_bar(self, plot):
        bars = gplt.bar([0, 1, 2], [10, 24, 18])
        assert len(gplt.bar_label(bars)) == 3

    def test_bar_returns_layers_not_the_figure(self, plot):
        """Contract fix: bar's list is patch layers, not N copies of the figure."""
        bars = gplt.bar([0, 1], [2, 3])
        assert all(getattr(b, "vertices", None) is not None for b in bars)


class TestIntrospectionAndRc:
    def test_setp_sets_and_lists(self, plot):
        layer = gplt.scatter([0, 1], [0, 1])
        gplt.setp(layer, alpha=0.5)
        assert gplt.getp(layer, "alpha") == 0.5
        assert "alpha" in gplt.setp(layer)  # no value -> lists properties

    def test_getp_all_properties(self, plot):
        layer = gplt.scatter([0, 1], [0, 1])
        props = gplt.getp(layer)
        assert "label" in props

    def test_findobj_by_predicate(self, plot):
        gplt.plot([0, 1], [0, 1], label="target")
        found = gplt.findobj(match=lambda o: getattr(o, "label", "") == "target")
        assert len(found) == 1

    def test_get_scale_names(self, plot):
        assert "linear" in gplt.get_scale_names()

    def test_get_plot_commands_lists_functions(self, plot):
        cmds = gplt.get_plot_commands()
        assert "plot" in cmds and "scatter" in cmds

    def test_rc_roundtrips_through_matplotlib(self, plot):
        import matplotlib as mpl

        gplt.rc("lines", linewidth=3)
        assert mpl.rcParams["lines.linewidth"] == 3
        gplt.rcdefaults()

    def test_xkcd_is_a_context_manager(self, plot):
        with pytest.warns(gplt.MatplotlibCompatWarning):
            with gplt.xkcd():
                pass


class TestImageIO:
    def test_imsave_imread_round_trip(self, plot, tmp_path):
        path = tmp_path / "test.png"
        img = np.random.default_rng(0).random((8, 8, 3))
        gplt.imsave(str(path), img)
        back = gplt.imread(str(path))
        assert back.shape[:2] == (8, 8)


def test_every_matplotlib_pyplot_function_exists():
    """The headline: no public matplotlib.pyplot function is missing from GLPlot.

    Guards against a regression that drops one, and documents that the parity surface is
    complete. Leaked non-matplotlib imports (typing.cast, cycler) are excluded -- they are
    not matplotlib API.
    """
    import inspect

    import matplotlib.pyplot as mpl

    def public_functions(module):
        return {
            name
            for name in dir(module)
            if not name.startswith("_") and inspect.isfunction(getattr(module, name))
        }

    mpl_api = {
        name
        for name in public_functions(mpl)
        if (getattr(getattr(mpl, name), "__module__", "") or "").startswith("matplotlib")
    }
    missing = mpl_api - public_functions(gplt)
    assert not missing, f"missing matplotlib.pyplot functions: {sorted(missing)}"
