"""Test axis annotation, tick generation and the Style panel's Axes tab.

The headline is :class:`TestAnnotationReachesPixels`. ``gplt.xlabel()`` set
``plot.xlabel`` and nothing read it -- the only consumers were in the matplotlib savefig
fallback -- so the call drew nothing at all in the live window. These tests fail on the
old code because the old ``AxisRenderer`` emitted tick labels and nothing else.

No OpenGL and no GPU at any point: ``AxisRenderer._draw_labels`` is pure imgui, and
``AxisManager`` is pure numpy, so both run under the sanctioned headless harness
(CONTRACT 2.10). ``GPULinePlot`` itself constructs without a window.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.core.context import RenderContext
from glplot.managers.axis import AxisManager, _decimals_for_step
from glplot.options import STOCK_WINDOW_TITLES, EngineOptions, RenderMode

imgui = pytest.importorskip("imgui")

from glplot.renderers.axis import AxisRenderer, _ImDrawVert  # noqa: E402

_WIDTH, _HEIGHT = 1280, 720


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


@pytest.fixture
def imgui_context():
    """A headless imgui context with a font atlas but no GL (CONTRACT 2.10)."""
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = _WIDTH, _HEIGHT
    io.delta_time = 1 / 60.0
    io.fonts.get_tex_data_as_rgba32()
    io.fonts.texture_id = 1
    yield io
    imgui.destroy_context(ctx)


@pytest.fixture
def plot():
    """A real engine with one line. Constructs no window (CONTRACT 5.1)."""
    t = np.linspace(0.0, 10.0, 64)
    gplt.plot(t, np.sin(t))
    live = gplt._get_or_create_plot()
    live.width, live.height = _WIDTH, _HEIGHT
    return live


def _label_vertices(live: Any) -> List[Tuple[float, float]]:
    """One frame of `_draw_labels`, returning every glyph vertex it emitted."""
    imgui.new_frame()
    window = live.camera_controller.world_window(_WIDTH, _HEIGHT)
    ctx = RenderContext(
        mvp=live.camera_controller.mvp(_WIDTH, _HEIGHT),
        window_world=window,
        width_px=_WIDTH,
        height_px=_HEIGHT,
        fb_width=_WIDTH,
        fb_height=_HEIGHT,
        mode=RenderMode.EXACT,
    )
    live.axis_manager.update(ctx)
    draw_list = imgui.get_background_draw_list()
    first = draw_list.vtx_buffer_size
    AxisRenderer(live.options)._draw_labels(live.axis_manager, ctx)
    count = draw_list.vtx_buffer_size - first
    verts = (_ImDrawVert * (first + count)).from_address(int(draw_list.vtx_buffer_data))
    points = [(verts[i].x, verts[i].y) for i in range(first, first + count)]
    imgui.end_frame()
    return points


def _bbox(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


class TestAnnotationReachesPixels:
    """gplt.xlabel/ylabel/title must draw. Each of these fails on the old renderer."""

    def test_xlabel_draws(self, imgui_context, plot):
        """gplt.xlabel() must emit glyphs, not just set an attribute nothing reads."""
        before = len(_label_vertices(plot))
        gplt.xlabel("Time (s)")
        assert len(_label_vertices(plot)) > before

    def test_ylabel_draws(self, imgui_context, plot):
        """gplt.ylabel() must emit glyphs."""
        before = len(_label_vertices(plot))
        gplt.ylabel("Amplitude (V)")
        assert len(_label_vertices(plot)) > before

    def test_title_draws(self, imgui_context, plot):
        """A caption the caller chose must be drawn on the plot."""
        before = len(_label_vertices(plot))
        plot.set_title("Damped response")
        assert len(_label_vertices(plot)) > before

    def test_gui_option_channel_draws(self, imgui_context, plot):
        """The GUI's option channel must work as well as the pyplot attribute channel."""
        before = len(_label_vertices(plot))
        plot.options.axis_xlabel = "Time (s)"
        assert len(_label_vertices(plot)) > before

    def test_option_wins_over_attribute(self, imgui_context, plot):
        """When both channels are set the option is the one drawn."""
        gplt.xlabel("x")
        plot.options.axis_xlabel = "a much longer label"
        long_count = len(_label_vertices(plot))
        plot.options.axis_xlabel = ""
        assert long_count > len(_label_vertices(plot))


class TestTitleFontsizeAndColor:
    """``title(fontsize=..., color=...)`` must reach pixels, not just be stored.

    Both kwargs exist for matplotlib parity, and a parity kwarg that parses and then does
    nothing is worse than one that raises TypeError: the caller reads the number back in
    the source and believes it. Each of these fails against an implementation that only
    records the option.
    """

    def _title_glyphs(self, live: Any, **kwargs: Any) -> List[Tuple[float, float]]:
        """The title's glyph vertices alone, isolated from the tick labels.

        Goes through ``gplt.title()`` rather than poking ``options`` directly, because the
        kwarg -> option -> pixel path is the whole thing under test; setting the option by
        hand would pass against a ``title()`` that drops its arguments on the floor.

        The baseline frame suppresses the title with a stock caption rather than by being
        taken first: these tests measure the same plot twice, so a "before it was set"
        baseline is only empty on the first call. ``_draw_annotations`` draws the title
        last of the three, and this fixture sets no x/y label, so the tail of the diff is
        the title's run.
        """
        live.set_title(next(iter(STOCK_WINDOW_TITLES)))
        before = _label_vertices(live)
        gplt.title("Damped response", **kwargs)
        after = _label_vertices(live)
        return after[len(before) :]

    def test_fontsize_widens_the_title(self, imgui_context, plot):
        """A bigger fontsize must emit a wider glyph run, not the same one."""
        x0, _, x1, _ = _bbox(self._title_glyphs(plot))
        bx0, _, bx1, _ = _bbox(self._title_glyphs(plot, fontsize=24.0))
        # 24pt against the 12pt matplotlib default is a 2x scale.
        assert (bx1 - bx0) == pytest.approx((x1 - x0) * 2.0, rel=0.1)

    def test_fontsize_stays_centred_as_it_grows(self, imgui_context, plot):
        """Scaling about the anchor must not walk the title off its centre."""
        x0, _, x1, _ = _bbox(self._title_glyphs(plot))
        bx0, _, bx1, _ = _bbox(self._title_glyphs(plot, fontsize=24.0))
        assert 0.5 * (bx0 + bx1) == pytest.approx(0.5 * (x0 + x1), abs=2.0)

    def test_default_fontsize_is_unscaled(self, imgui_context, plot):
        """12pt is matplotlib's default title size, so it must be a no-op scale."""
        unset = _bbox(self._title_glyphs(plot))
        assert _bbox(self._title_glyphs(plot, fontsize=12.0)) == pytest.approx(unset, abs=0.5)

    def test_omitting_fontsize_restores_the_default(self, imgui_context, plot):
        """An unset kwarg means the default, not "keep whatever the last call set"."""
        big = _bbox(self._title_glyphs(plot, fontsize=24.0))
        plain = _bbox(self._title_glyphs(plot))
        assert (plain[2] - plain[0]) < (big[2] - big[0]) * 0.75

    def test_color_changes_the_ink(self, imgui_context, plot):
        """color= must reach the vertex colour, not just the options dataclass."""

        def title_color() -> int:
            imgui.new_frame()
            window = plot.camera_controller.world_window(_WIDTH, _HEIGHT)
            ctx = RenderContext(
                mvp=plot.camera_controller.mvp(_WIDTH, _HEIGHT),
                window_world=window,
                width_px=_WIDTH,
                height_px=_HEIGHT,
                fb_width=_WIDTH,
                fb_height=_HEIGHT,
                mode=RenderMode.EXACT,
            )
            plot.axis_manager.update(ctx)
            draw_list = imgui.get_background_draw_list()
            first = draw_list.vtx_buffer_size
            AxisRenderer(plot.options)._draw_labels(plot.axis_manager, ctx)
            count = draw_list.vtx_buffer_size - first
            verts = (_ImDrawVert * (first + count)).from_address(int(draw_list.vtx_buffer_data))
            # The title is the last run emitted, so its colour is the last vertex's.
            col = verts[first + count - 1].col
            imgui.end_frame()
            return col

        gplt.title("Damped response")
        default = title_color()
        gplt.title("Damped response", color="red")
        assert title_color() != default

    def test_omitting_color_restores_the_default_ink(self, imgui_context, plot):
        """As with fontsize: unset means default, not "inherit the last call"."""
        gplt.title("Damped response")
        assert plot.options.axis_title_color is None
        gplt.title("Damped response", color="red")
        assert plot.options.axis_title_color == "red"
        gplt.title("Damped response")
        assert plot.options.axis_title_color is None

    def test_unparseable_color_falls_back_instead_of_raising(self, imgui_context, plot):
        """A bad colour must not take the frame down mid-draw."""
        gplt.title("Damped response", color="not-a-real-colour")
        assert len(_label_vertices(plot)) > 0


class TestLabelFontsizeAndColor:
    """xlabel/ylabel take the same Text kwargs as title. ``xlabel("x", fontsize=12)``
    is everyday matplotlib and used to be a TypeError."""

    def _label_glyphs(self, live: Any, which: str, **kwargs: Any) -> List[Tuple[float, float]]:
        """One label's glyph run, isolated by drawing the frame without it first."""
        fn = getattr(gplt, which)
        fn("")
        before = _label_vertices(live)
        fn("Amplitude", **kwargs)
        after = _label_vertices(live)
        return after[len(before) :]

    @pytest.mark.parametrize("which", ["xlabel", "ylabel"])
    def test_fontsize_grows_the_label(self, imgui_context, plot, which):
        """The y-label goes through the rotation path, so both need their own case."""
        plain = _bbox(self._label_glyphs(plot, which))
        big = _bbox(self._label_glyphs(plot, which, fontsize=20.0))
        plain_extent = max(plain[2] - plain[0], plain[3] - plain[1])
        big_extent = max(big[2] - big[0], big[3] - big[1])
        # 20pt against matplotlib's 10pt label default is a 2x scale.
        assert big_extent == pytest.approx(plain_extent * 2.0, rel=0.15)

    @pytest.mark.parametrize("which", ["xlabel", "ylabel"])
    def test_default_fontsize_is_unscaled(self, imgui_context, plot, which):
        """10pt is matplotlib's axes.labelsize, so it must be the no-op scale."""
        unset = _bbox(self._label_glyphs(plot, which))
        assert _bbox(self._label_glyphs(plot, which, fontsize=10.0)) == pytest.approx(
            unset, abs=0.5
        )

    @pytest.mark.parametrize("which", ["xlabel", "ylabel"])
    def test_color_is_stored_for_the_renderer(self, imgui_context, plot, which):
        getattr(gplt, which)("Amplitude", color="red")
        assert getattr(plot.options, f"axis_{which}_color") == "red"

    @pytest.mark.parametrize("which", ["xlabel", "ylabel"])
    def test_omitting_style_restores_the_default(self, imgui_context, plot, which):
        fn = getattr(gplt, which)
        fn("Amplitude", fontsize=20.0, color="red")
        fn("Amplitude")
        assert getattr(plot.options, f"axis_{which}_fontsize") is None
        assert getattr(plot.options, f"axis_{which}_color") is None

    def test_ylabel_still_rotates_when_scaled(self, imgui_context, plot):
        """Scaling must compose with the rotation, not replace it.

        Both transforms are applied in one pass over the glyph vertices, so getting the
        composition wrong silently drops one of them.
        """
        box = _bbox(self._label_glyphs(plot, "ylabel", fontsize=20.0))
        # Rotated text is taller than it is wide; unrotated "Amplitude" is the reverse.
        assert (box[3] - box[1]) > (box[2] - box[0])


class TestTextKwargParity:
    """The Text-property surface title/xlabel/ylabel share: fontdict and the no-ops."""

    def test_fontdict_is_honoured(self, imgui_context, plot):
        gplt.title("T", fontdict={"fontsize": 20.0, "color": "red"})
        assert plot.options.axis_title_fontsize == 20.0
        assert plot.options.axis_title_color == "red"

    def test_fontdict_accepts_matplotlibs_size_alias(self, imgui_context, plot):
        gplt.title("T", fontdict={"size": 20.0})
        assert plot.options.axis_title_fontsize == 20.0

    def test_explicit_kwarg_beats_fontdict(self, imgui_context, plot):
        """matplotlib's precedence: the more specific, more visible spelling wins."""
        gplt.title("T", fontdict={"fontsize": 20.0}, fontsize=30.0)
        assert plot.options.axis_title_fontsize == 30.0

    def test_unsupported_fontdict_key_warns(self, imgui_context, plot):
        """The no-op policy has to reach inside the dict, not just the signature."""
        with pytest.warns(gplt.MatplotlibCompatWarning, match="family"):
            gplt.title("T", fontdict={"family": "serif"})

    @pytest.mark.parametrize(
        "func,kwarg",
        [("title", "label"), ("xlabel", "xlabel"), ("ylabel", "ylabel")],
    )
    def test_matplotlibs_parameter_name_is_keyword_passable(self, imgui_context, plot, func, kwarg):
        """plt.xlabel(xlabel="Time") is legal matplotlib, so it must be legal here."""
        getattr(gplt, func)(**{kwarg: "Text"})
        assert len(_label_vertices(plot)) > 0

    @pytest.mark.parametrize("func", ["title", "xlabel", "ylabel"])
    def test_legacy_s_spelling_still_works(self, imgui_context, plot, func):
        """This is on PyPI; renaming the parameter must not TypeError released code."""
        getattr(gplt, func)(s="Text")
        assert len(_label_vertices(plot)) > 0

    @pytest.mark.parametrize("func", ["title", "xlabel", "ylabel"])
    def test_missing_text_names_the_matplotlib_parameter(self, imgui_context, plot, func):
        """The error must point at the supported spelling, not the legacy one."""
        with pytest.raises(TypeError, match="missing required argument"):
            getattr(gplt, func)()

    @pytest.mark.parametrize(
        "func,kwargs",
        [
            ("title", {"loc": "left"}),
            ("title", {"pad": 20.0}),
            ("title", {"y": 0.9}),
            ("xlabel", {"labelpad": 10.0}),
            ("xlabel", {"loc": "left"}),
            ("ylabel", {"labelpad": 10.0}),
            ("ylabel", {"loc": "top"}),
        ],
    )
    def test_positional_no_ops_warn_rather_than_raise(self, imgui_context, plot, func, kwargs):
        with pytest.warns(gplt.MatplotlibCompatWarning):
            getattr(gplt, func)("text", **kwargs)


class TestDefaultsUnchanged:
    """The stock look must not move. This is a published library."""

    def test_untitled_plot_draws_no_annotation(self, imgui_context, plot):
        """A default plot draws tick labels only -- no label, no title."""
        baseline = len(_label_vertices(plot))
        gplt.xlabel("Time (s)")
        gplt.ylabel("Amplitude")
        plot.set_title("A title")
        annotated = len(_label_vertices(plot))
        # Strip all three again; the frame must return to exactly the baseline.
        plot.xlabel = ""
        plot.ylabel = ""
        plot.set_title("GLPlot")
        assert annotated > baseline
        assert len(_label_vertices(plot)) == baseline

    def test_stock_caption_is_not_promoted_to_a_title(self, imgui_context, plot):
        """The default window caption must not caption the plot with the product name."""
        assert plot.title in STOCK_WINDOW_TITLES
        baseline = len(_label_vertices(plot))
        plot.set_title("Hybrid GPU Line Renderer")
        assert len(_label_vertices(plot)) == baseline

    def test_minor_ticks_and_marks_are_off_by_default(self):
        """Both restyle every plot, so neither may default on."""
        options = EngineOptions()
        assert options.axis_minor_ticks is False
        assert options.axis_show_ticks is False


class TestYLabelRotation:
    """The y-label is rotated, not stacked and not left horizontal."""

    def test_ylabel_is_rotated_ninety_degrees(self, imgui_context, plot):
        """Its on-screen box must be the transpose of the text's own box."""
        before = set(_label_vertices(plot))
        text = "Amplitude (V)"
        gplt.ylabel(text)
        added = [p for p in _label_vertices(plot) if p not in before]
        assert added, "the y-label emitted no glyphs"

        text_w, text_h = imgui.calc_text_size(text)
        x0, y0, x1, y1 = _bbox(added)
        # Rotated: the run is as wide as the text is tall, and as tall as it is wide.
        assert (x1 - x0) <= text_h + 2.0
        assert (y1 - y0) >= text_w * 0.7

    def test_ylabel_sits_in_the_left_gutter(self, imgui_context, plot):
        """It must land inside the reserved margin, clear of the frame."""
        before = set(_label_vertices(plot))
        gplt.ylabel("Amplitude (V)")
        added = [p for p in _label_vertices(plot) if p not in before]
        _, _, x1, _ = _bbox(added)
        assert x1 < plot.options.axis_margin_l

    def test_ylabel_is_centred_on_the_frame(self, imgui_context, plot):
        """Vertically centred between the top and bottom spines."""
        before = set(_label_vertices(plot))
        gplt.ylabel("Amplitude (V)")
        added = [p for p in _label_vertices(plot) if p not in before]
        _, y0, _, y1 = _bbox(added)
        frame_mid = 0.5 * (plot.options.axis_margin_t + (_HEIGHT - plot.options.axis_margin_b))
        assert abs(0.5 * (y0 + y1) - frame_mid) < 2.0


class TestAnnotationPlacement:
    """Placement follows the gutters rather than a fixed magic number."""

    def test_xlabel_sits_below_the_tick_labels(self, imgui_context, plot):
        """The x-label must clear the tick labels, not overprint them."""
        ticks_only = _label_vertices(plot)
        plot.options.axis_xlabel = "Time (s)"
        added = [p for p in _label_vertices(plot) if p not in set(ticks_only)]
        assert added
        _, label_top, _, label_bottom = _bbox(added)
        _, _, _, ticks_bottom = _bbox(ticks_only)
        assert label_top >= ticks_bottom
        assert label_bottom <= _HEIGHT

    def test_xlabel_is_centred_on_the_frame(self, imgui_context, plot):
        """Horizontally centred between the left and right spines."""
        ticks_only = set(_label_vertices(plot))
        plot.options.axis_xlabel = "Time (s)"
        added = [p for p in _label_vertices(plot) if p not in ticks_only]
        x0, _, x1, _ = _bbox(added)
        frame_mid = 0.5 * (plot.options.axis_margin_l + (_WIDTH - plot.options.axis_margin_r))
        assert abs(0.5 * (x0 + x1) - frame_mid) < 3.0

    def test_widening_the_left_gutter_moves_the_ylabel_out(self, imgui_context, plot):
        """A GUI that raises axis_margin_l to clear its own chrome must gain room.

        This is the coupling the rail fix depends on: the y-label is placed off the
        projected spine, so it tracks the gutter instead of a baked-in offset.
        """

        def ylabel_bbox() -> Tuple[float, float, float, float]:
            plot.options.axis_ylabel = ""
            without = set(_label_vertices(plot))
            plot.options.axis_ylabel = "Amplitude (V)"
            added = [p for p in _label_vertices(plot) if p not in without]
            assert added, "the y-label emitted no glyphs"
            return _bbox(added)

        narrow = ylabel_bbox()
        plot.options.axis_margin_l = 120.0
        wide = ylabel_bbox()
        # The whole label moved right with the spine, and still clears the frame.
        assert wide[0] > narrow[0]
        assert wide[2] < plot.options.axis_margin_l


class TestTickGeneration:
    """Tick count, spacing and subdivision."""

    def _ticks(self, options: EngineOptions, window=(0.0, 10.0, -1.0, 1.0)) -> AxisManager:
        holder = type("P", (), {"options": options})()
        manager = AxisManager(holder)
        manager.update(
            RenderContext(
                mvp=np.eye(4, dtype=np.float32),
                window_world=window,
                width_px=_WIDTH,
                height_px=_HEIGHT,
                fb_width=_WIDTH,
                fb_height=_HEIGHT,
                mode=RenderMode.EXACT,
            )
        )
        return manager

    def test_default_ticks_are_the_historical_nice_numbers(self):
        """The 1-2-5 ladder must be untouched."""
        assert self._ticks(EngineOptions()).ticks_x.labels == [str(i) for i in range(11)]

    def test_tick_count_controls_density(self):
        """A lower target count yields fewer ticks."""
        options = EngineOptions()
        options.axis_tick_count_x = 3
        assert len(self._ticks(options).ticks_x.major) < 11

    def test_tick_step_pins_the_spacing(self):
        """An explicit step overrides the nice ladder."""
        options = EngineOptions()
        options.axis_tick_step_x = 2.5
        assert np.allclose(self._ticks(options).ticks_x.major, [0.0, 2.5, 5.0, 7.5, 10.0])

    def test_pinned_step_labels_the_ticks_it_actually_drew(self):
        """A 2.5 step used to print '0, 2, 5, 8': the labels must match the positions."""
        options = EngineOptions()
        options.axis_tick_step_x = 2.5
        assert self._ticks(options).ticks_x.labels == ["0.0", "2.5", "5.0", "7.5", "10.0"]

    def test_absurd_step_is_clamped(self):
        """A hand-typed step must not stall the frame with a million labels."""
        options = EngineOptions()
        options.axis_tick_step_x = 1e-9
        assert len(self._ticks(options).ticks_x.major) <= 1001

    def test_minor_ticks_are_off_by_default(self):
        """No subdivision without asking."""
        assert len(self._ticks(EngineOptions()).ticks_x.minor) == 0

    def test_minor_ticks_subdivide_the_major_step(self):
        """Each major interval gains `subdivisions - 1` minors."""
        options = EngineOptions()
        options.axis_minor_ticks = True
        options.axis_minor_subdivisions = 4
        ticks = self._ticks(options).ticks_x
        assert len(ticks.minor) > 0
        assert np.allclose(np.diff(np.sort(ticks.minor))[0], ticks.step / 4, atol=1e-9)

    def test_minor_ticks_never_coincide_with_a_major(self):
        """A minor drawn under a major is a double-strength grid line."""
        options = EngineOptions()
        options.axis_minor_ticks = True
        ticks = self._ticks(options).ticks_x
        for value in ticks.minor:
            assert not np.isclose(value, ticks.major, atol=1e-9).any()

    def test_inverted_window_yields_no_ticks(self):
        """Preserved behaviour: vmin >= vmax is empty, not an exception."""
        assert len(self._ticks(EngineOptions(), window=(10.0, 0.0, 1.0, -1.0)).ticks_x.major) == 0


class TestTickFormat:
    """The label formatter."""

    def _labels(self, options: EngineOptions, window=(0.0, 10.0, -1.0, 1.0)) -> List[str]:
        holder = type("P", (), {"options": options})()
        manager = AxisManager(holder)
        manager.update(
            RenderContext(
                mvp=np.eye(4, dtype=np.float32),
                window_world=window,
                width_px=_WIDTH,
                height_px=_HEIGHT,
                fb_width=_WIDTH,
                fb_height=_HEIGHT,
                mode=RenderMode.EXACT,
            )
        )
        return manager.ticks_x.labels

    def test_printf_form(self):
        """%.2f is a thing a scientist types."""
        options = EngineOptions()
        options.axis_tick_format = "%.2f"
        assert self._labels(options)[0] == "0.00"

    def test_str_format_form(self):
        """So is {:.3g}."""
        options = EngineOptions()
        options.axis_tick_format = "{:.1f}"
        assert self._labels(options)[0] == "0.0"

    def test_bare_spec_form(self):
        """And so is a bare format spec."""
        options = EngineOptions()
        options.axis_tick_format = ".1e"
        assert self._labels(options)[0] == "0.0e+00"

    def test_invalid_format_falls_back_instead_of_raising(self):
        """The GUI edits this live: it is invalid on almost every keystroke."""
        options = EngineOptions()
        options.axis_tick_format = "%.2q"
        assert self._labels(options) == self._labels(EngineOptions())

    def test_tiny_range_is_not_a_column_of_zeros(self):
        """A nanometre axis must carry information in its labels."""
        labels = self._labels(EngineOptions(), window=(0.0, 3e-9, -1.0, 1.0))
        assert len(set(labels)) == len(labels)
        assert all("e" in label for label in labels)

    def test_ordinary_ranges_stay_fixed_point(self):
        """Scientific notation must not leak into everyday plots."""
        assert self._labels(EngineOptions()) == [str(i) for i in range(11)]

    @pytest.mark.parametrize("exponent", range(-9, 10))
    @pytest.mark.parametrize("mantissa", [1.0, 2.0, 5.0])
    def test_decimals_match_the_historical_rule_on_every_nice_step(self, mantissa, exponent):
        """The precision rule must be bit-identical to the old one on the 1-2-5 ladder."""
        step = mantissa * (10.0**exponent)
        historical = max(0, int(-np.floor(np.log10(step)))) if step < 1 else 0
        assert _decimals_for_step(step) == historical


class TestAxesTab:
    """The Style panel's Axes tab, through the headless harness (CONTRACT 2.10)."""

    def _panel(self, live: Any) -> Any:
        from glplot.gui.commands import CommandQueue
        from glplot.gui.panels.style import StylePanel

        workspace = type(
            "WS",
            (),
            {"plot": live, "queue": CommandQueue(), "store": None, "undo": None, "hud": None},
        )()
        return StylePanel(workspace)

    def _draw_axes_tab(self, panel: Any) -> None:
        imgui.new_frame()
        imgui.set_next_window_size(360, 640)
        imgui.begin("Style")
        panel._draw_axes()
        imgui.end()
        imgui.render()

    def test_axes_tab_draws_without_error(self, imgui_context, plot):
        """Every widget in the tab must survive a real frame."""
        self._draw_axes_tab(self._panel(plot))

    def test_axes_tab_draws_with_annotation_set(self, imgui_context, plot):
        """Including when the labels come from the pyplot channel."""
        gplt.xlabel("Time (s)")
        gplt.ylabel("Amplitude (V)")
        plot.set_title("A title")
        self._draw_axes_tab(self._panel(plot))

    def test_axes_tab_draws_with_ticks_enabled(self, imgui_context, plot):
        """The conditional sub-widgets (subdivisions, tick length) must draw too."""
        plot.options.axis_minor_ticks = True
        plot.options.axis_show_ticks = True
        self._draw_axes_tab(self._panel(plot))

    def test_tab_reads_back_the_pyplot_channel(self, imgui_context, plot):
        """A label set by gplt.xlabel() must show in the box, not look unset."""
        gplt.xlabel("Time (s)")
        panel = self._panel(plot)
        assert (
            str(getattr(plot.options, "axis_xlabel", "") or "")
            or str(getattr(panel.plot, "xlabel", "") or "")
        ) == "Time (s)"

    def test_tab_does_not_offer_a_stock_caption_as_a_title(self, imgui_context, plot):
        """The Title box mirrors the renderer: blank for a default window."""
        from glplot.gui.panels.style import _plot_title

        assert _plot_title(plot) == ""
        plot.set_title("Real")
        assert _plot_title(plot) == "Real"
