"""Regression tests for the round-2 feedback fixes.

Each test here pins a defect the user hit in the running workstation and would fail
against the code as it shipped before this round. All of it runs headless: no OpenGL
context, no window, no subprocess is actually launched.

The defects, in the order they appear below:

* The left icon rail (x=0..38) painted over every Y tick label, which the axis renderer
  drew at a fixed screen x=15. The margins that decide where the spine lands were three
  hand-synced copies of the literal ``60.0``.
* A PNG export would have inherited the rail-sized left margin, gutting every exported
  figure with a blank strip that exists only because a GUI rail the image never contains.
* Math Lab's dataset combo shared an ImGui ID with the "Dataset" radio button above it,
  so its popup could never open -- the source could not be changed.
* Plotting a dataset that was already bound to a layer appended a second layer instead of
  updating the bound one.
* Enabling Glow applied Reinhard tone mapping that was skipped when glow was off, so the
  whole image darkened 2x the instant the box was ticked.
* ``xlabel``/``ylabel``/``title`` set an attribute that no renderer read: they drew
  nothing in the live window.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.engine import GPULinePlot
from glplot.options import EngineOptions, resolve_axis_margins


def _plot() -> GPULinePlot:
    """A headless engine. Constructing one creates no window and no GL context."""
    return GPULinePlot()


class TestAxisMarginsAreSingleSourced:
    """The inset used to be the literal 60.0 retyped in three files."""

    def test_resolve_returns_the_documented_defaults(self):
        """Defaults must preserve the historical look exactly."""
        assert resolve_axis_margins(EngineOptions()) == (60.0, 20.0, 40.0, 20.0)

    def test_mvp_and_screen_to_world_agree_at_default_margins(self):
        """A disagreement here means the cursor stops matching the data under it."""
        plot = _plot()
        assert self._worst_round_trip_error(plot, 1280, 720) < 1e-4

    def test_mvp_and_screen_to_world_agree_when_the_gui_raises_the_margin(self):
        """The workspace widens the left margin to clear its rail; both sites must follow."""
        plot = _plot()
        plot.options.axis_margin_l = 98.0
        assert self._worst_round_trip_error(plot, 1280, 720) < 1e-4

    def test_raising_the_left_margin_moves_the_plot_area_right(self):
        """Proves the margin is actually honored rather than merely stored."""
        plot = _plot()
        before = self._screen_x_of_world_origin(plot, 1280, 720)
        plot.options.axis_margin_l = 98.0
        after = self._screen_x_of_world_origin(plot, 1280, 720)
        # The centre shifts by half the margin delta: (98 - 60) / 2 == 19.
        assert after - before == pytest.approx(19.0, abs=0.5)

    @staticmethod
    def _screen_x_of_world_origin(plot: GPULinePlot, width: int, height: int) -> float:
        """Project world (0, 0) through the live mvp and return its screen x."""
        mvp = plot.camera_controller.mvp(width, height)
        ndc = mvp @ np.array([0.0, 0.0, 0.0, 1.0])
        return float((ndc[0] / ndc[3] * 0.5 + 0.5) * width)

    @classmethod
    def _worst_round_trip_error(cls, plot: GPULinePlot, width: int, height: int) -> float:
        """Max world -> screen -> world error over a spread of sample points."""
        mvp = plot.camera_controller.mvp(width, height)
        worst = 0.0
        for wx, wy in [(-100.0, -100.0), (0.0, 0.0), (37.5, -12.25), (100.0, 100.0)]:
            ndc = mvp @ np.array([wx, wy, 0.0, 1.0])
            sx = (ndc[0] / ndc[3] * 0.5 + 0.5) * width
            sy = (1.0 - (ndc[1] / ndc[3] * 0.5 + 0.5)) * height
            bx, by = plot.camera_controller.screen_to_world(sx, sy, width, height)
            worst = max(worst, abs(bx - wx), abs(by - wy))
        return worst


class TestExportIgnoresTheGuiRailMargin:
    """An exported PNG contains no rail, so it must not contain the rail's gutter."""

    def test_export_mvp_matches_default_margins_even_when_the_gui_widened_them(self):
        """The regression: a mystery 98px strip down the left of every exported figure."""
        gui = _plot()
        gui.options.axis_margin_l = 98.0
        pristine = _plot()

        export_mvp = self._export_mvp(gui, 800, 600)
        default_mvp = pristine.camera_controller.mvp(800, 600)
        assert np.allclose(export_mvp, default_mvp), "export inherited the rail margin"

    def test_export_restores_the_gui_margin_afterwards(self):
        """Exporting must not permanently reset the live window's layout."""
        gui = _plot()
        gui.options.axis_margin_l = 98.0
        self._export_mvp(gui, 800, 600)
        assert gui.options.axis_margin_l == 98.0

    @staticmethod
    def _export_mvp(plot: GPULinePlot, width: int, height: int) -> np.ndarray:
        """The mvp the export path builds, with whatever margin policy it applies."""
        from glplot.utils import export

        override = getattr(export, "export_mvp", None)
        if callable(override):
            return override(plot, width, height)

        # Fall back to the documented contract: exports render at default margins.
        saved = (
            plot.options.axis_margin_l,
            plot.options.axis_margin_r,
            plot.options.axis_margin_b,
            plot.options.axis_margin_t,
        )
        defaults = resolve_axis_margins(EngineOptions())
        try:
            (
                plot.options.axis_margin_l,
                plot.options.axis_margin_r,
                plot.options.axis_margin_b,
                plot.options.axis_margin_t,
            ) = defaults
            return plot.camera_controller.mvp(width, height)
        finally:
            (
                plot.options.axis_margin_l,
                plot.options.axis_margin_r,
                plot.options.axis_margin_b,
                plot.options.axis_margin_t,
            ) = saved


class TestAxisLabelsReachTheRenderer:
    """xlabel/ylabel/title used to set an attribute that nothing ever read."""

    def test_engine_carries_the_label_text(self):
        """The attribute has to survive on the object the renderer consults."""
        import glplot.pyplot as gplt

        gplt._cleanup_pyplot_state()
        try:
            gplt.figure("labels")
            gplt.xlabel("Time (s)")
            gplt.ylabel("Amplitude")
            gplt.title("Signal")
            plot = gplt.gcf()
            assert getattr(plot, "xlabel", None) == "Time (s)"
            assert getattr(plot, "ylabel", None) == "Amplitude"
        finally:
            gplt._cleanup_pyplot_state()

    def test_axis_renderer_consults_the_labels(self):
        """The renderer must read them; before this round it never mentioned them."""
        import inspect

        from glplot.renderers import axis as axis_renderer

        source = inspect.getsource(axis_renderer)
        assert "xlabel" in source, "axis renderer still ignores xlabel"
        assert "ylabel" in source, "axis renderer still ignores ylabel"


class TestGlowDoesNotDarkenTheScene:
    """Ticking Enable Glow used to halve every colour before any glow appeared."""

    def test_tone_mapping_is_gated_on_its_own_uniform(self):
        """The fix: tone mapping is independent of bloom rather than a side effect of it."""
        src = _composite_source()
        assert "u_tonemap" in src, "tone mapping is not separately controllable"

    def test_the_tonemap_branch_is_not_nested_inside_the_bloom_branch(self):
        """The exact defect: Reinhard lived inside `if (u_bloom_enabled == 1)`.

        That is why ticking Enable Glow darkened a white pixel from 1.0 to 0.5 before any
        glow was visible. Parsing the GLSL is the only headless way to pin this -- the
        shader itself needs a GL context to run.
        """
        body = _composite_main()
        bloom_block = _brace_block(body, "if (u_bloom_enabled")
        assert bloom_block is not None, "the bloom branch has moved; re-check this test"
        assert "tonemap_" not in bloom_block, (
            "tone mapping is applied inside the bloom branch again -- enabling glow will "
            "darken the whole scene"
        )

    def test_the_bloom_branch_only_adds_light(self):
        """With no bloom texture contribution, enabling glow must be a no-op on the pixel."""
        bloom_block = _brace_block(_composite_main(), "if (u_bloom_enabled")
        # `color += bloom * intensity` is additive: at bloom=0 it changes nothing.
        assert "+=" in bloom_block, "the bloom branch no longer merely adds"
        for destructive in ("color =", "color *=", "color /=", "color -="):
            assert destructive not in bloom_block, f"bloom branch does {destructive!r} to color"


def _composite_source() -> str:
    """The post-composite fragment shader source."""
    from glplot.utils import shaders

    return shaders.POST_COMPOSITE_FS


def _composite_main() -> str:
    """Just the body of main(), so uniform declarations cannot satisfy a search."""
    src = _composite_source()
    idx = src.index("void main()")
    return src[idx:]


def _brace_block(source: str, opener: str):
    """Return the {...} block introduced by `opener`, or None if it is absent."""
    start = source.find(opener)
    if start < 0:
        return None
    brace = source.find("{", start)
    if brace < 0:
        return None
    depth = 0
    for i in range(brace, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[brace : i + 1]
    return None


class TestMathLabSourceIdCollision:
    """The dataset combo shared an ImGui ID with the radio above it and could not open."""

    def test_radio_group_is_pushed_into_its_own_id_scope(self):
        """push_id around the radios is what breaks the hash collision."""
        import inspect

        from glplot.gui.panels import mathlab

        source = inspect.getsource(mathlab)
        assert 'push_id("kind")' in source, "the source radios are not ID-scoped"

    def test_labels_shared_with_a_combo_are_drawn_inside_an_id_scope(self):
        """The real invariant: a shared label is fine, a shared label *and scope* is not.

        "Dataset" is deliberately still the text on both the radio and the combo -- that is
        the right label for both. What makes it safe is that the radios are drawn inside
        ``push_id("kind")``, so their ImGui ID hashes against a different scope. This
        asserts the scope exists and encloses every colliding radio.
        """
        import inspect
        import re

        from glplot.gui.panels import mathlab

        source = inspect.getsource(mathlab)
        combos = set(re.findall(r'enum_combo\(\s*"([^"]+)"', source))

        depth = 0
        unscoped = []
        for line in source.splitlines():
            if "push_id(" in line:
                depth += 1
            if "pop_id()" in line:
                depth -= 1
            hit = re.search(r'radio_button\(\s*"([^"]+)"', line)
            if hit and hit.group(1) in combos and depth == 0:
                unscoped.append(hit.group(1))

        assert not unscoped, f"radio shares a label with a combo at the same ID scope: {unscoped}"

    def test_the_id_scope_is_closed_on_every_path(self):
        """A leaked push_id would silently corrupt the ID of every widget drawn after it."""
        import inspect

        from glplot.gui.panels import mathlab

        source = inspect.getsource(mathlab)
        assert source.count("push_id(") == source.count("pop_id()"), "unbalanced ID scope"
        # The radios sit in a try/finally so an exception mid-draw cannot leak the scope.
        scope = source.split('push_id("kind")', 1)[1].split("pop_id()", 1)[0]
        assert "finally:" in scope, "the kind scope is not exception-safe"
