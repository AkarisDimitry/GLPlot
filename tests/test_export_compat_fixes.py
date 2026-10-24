"""Headless-export and plot() fixes that made GLPlot disagree with matplotlib.

Each class pins one thing a matplotlib script gets for free: a marker-only format string
meaning markers only, error bars taking the data's colour, a scalar boundary in
``fill_between``, a violin that is a body rather than a comb, and ``set_aspect('equal')``
surviving into a saved file. The export ones inspect the matplotlib figure the preview
builds, rather than pixels, so they say what is wrong when they fail.
"""

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.utils.preview import _build_preview_figure


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


def _preview_axes():
    fig = _build_preview_figure(gplt.get_engine())
    return fig, fig.axes[0]


class TestMarkerOnlyFormat:
    """``'o'`` / ``'ro'`` name a marker and no line, so no line is drawn."""

    @pytest.mark.parametrize("fmt", ["o", "ro", "C1s", "^"])
    def test_marker_only_format_draws_no_line(self, fmt):
        layers = gplt.plot([0, 1, 2], [0, 1, 4], fmt)
        assert [layer.layer_type for layer in layers] == ["scatter"]

    @pytest.mark.parametrize("fmt", ["o-", "-o", "r--s", "o:"])
    def test_format_with_a_line_keeps_it(self, fmt):
        types = {layer.layer_type for layer in gplt.plot([0, 1, 2], [0, 1, 4], fmt)}
        assert types == {"scatter", "polyline"}

    def test_explicit_linestyle_overrides_the_inferred_none(self):
        layers = gplt.plot([0, 1, 2], [0, 1, 4], "o", linestyle="--")
        assert {layer.layer_type for layer in layers} == {"scatter", "polyline"}

    def test_errorbar_fmt_o_has_no_connecting_line(self):
        artists = gplt.errorbar([0, 1, 2], [1, 2, 3], yerr=0.2, fmt="o")
        data_layers = [a for a in artists if a.metadata.get("artist") in ("line", "marker")]
        assert data_layers and all(a.metadata["artist"] == "marker" for a in data_layers)


class TestErrorbarColour:
    """matplotlib's ``ecolor`` defaults to the colour the data is drawn in."""

    @staticmethod
    def _bar_and_data_colours(artists):
        bars = [a for a in artists if a.layer_type == "polyline" and a.style.line_width <= 1.0]
        data = [a for a in artists if a.layer_type == "scatter"]
        return bars, data

    def test_bars_follow_an_explicit_colour(self):
        artists = gplt.errorbar([0, 1], [1, 2], yerr=0.1, fmt="o", color="red")
        bars, data = self._bar_and_data_colours(artists)
        assert bars and data
        assert all(tuple(b.style.color) == tuple(data[0].colors[0]) for b in bars)

    def test_bars_follow_the_colour_letter_in_fmt(self):
        artists = gplt.errorbar([0, 1], [1, 2], yerr=0.1, fmt="gs")
        bars, data = self._bar_and_data_colours(artists)
        assert all(tuple(b.style.color) == tuple(data[0].colors[0]) for b in bars)

    def test_bars_follow_the_next_cycle_colour_not_black(self):
        artists = gplt.errorbar([0, 1], [1, 2], yerr=0.1, fmt="o")
        bars, data = self._bar_and_data_colours(artists)
        assert tuple(bars[0].style.color)[:3] != (0.0, 0.0, 0.0)
        assert tuple(bars[0].style.color) == tuple(data[0].colors[0])

    def test_ecolor_still_wins(self):
        artists = gplt.errorbar([0, 1], [1, 2], yerr=0.1, fmt="o", color="red", ecolor="blue")
        bars, data = self._bar_and_data_colours(artists)
        assert tuple(bars[0].style.color) != tuple(data[0].colors[0])

    def test_only_one_cycle_slot_is_used(self):
        gplt.errorbar([0, 1], [1, 2], yerr=0.1, fmt="o")
        first = gplt.plot([0, 1], [0, 1])[0]
        # The errorbar took slot 0; the next line must take slot 1, not skip to 2.
        assert tuple(first.style.color) == tuple(gplt._normalize_rgba(gplt._color_cycle()[1]))


class TestFillBetweenScalarBoundary:
    """``fill_between(x, 0, y)`` -- a scalar on either side -- broadcasts against x."""

    def test_scalar_y1(self):
        x = np.linspace(0, 1, 5)
        layer = gplt.fill_between(x, 0, x + 1)
        assert layer.vertices is not None and len(layer.vertices) >= 4

    def test_scalar_on_both_sides(self):
        layer = gplt.fill_between(np.linspace(0, 1, 5), 0, 1)
        assert layer.vertices is not None

    def test_scalar_y1_matches_the_explicit_array(self):
        x = np.linspace(0, 1, 5)
        a = gplt.fill_between(x, 0, x + 1)
        gplt._cleanup_pyplot_state()
        b = gplt.fill_between(x, np.zeros_like(x), x + 1)
        np.testing.assert_allclose(a.vertices, b.vertices)

    def test_scalar_x1_in_fill_betweenx(self):
        y = np.linspace(0, 1, 5)
        layer = gplt.fill_betweenx(y, 0, y + 1)
        assert layer.vertices is not None

    def test_mismatched_length_still_raises(self):
        with pytest.raises(ValueError, match="same length"):
            gplt.fill_between(np.linspace(0, 1, 5), np.zeros(3), 1)


class TestViolinExport:
    """A strip patch's vertices interleave its two sides; its outline is one side then back."""

    def test_violin_body_is_exported_as_an_outline_not_a_zigzag(self):
        rng = np.random.default_rng(0)
        gplt.violinplot([rng.normal(size=300)])
        _, ax = _preview_axes()
        (poly,) = ax.patches
        xy = poly.get_xy()[:-1]  # drop the closing vertex
        # In strip order consecutive vertices alternate left/right (x jumps across the body
        # every step); in outline order x moves smoothly and y only reverses once.
        steps = np.abs(np.diff(xy[:, 0]))
        assert np.count_nonzero(steps > 0.2) <= 2
        reversals = np.count_nonzero(np.diff(np.sign(np.diff(xy[:, 1]))) != 0)
        assert reversals <= 2

    def test_a_layer_with_indices_is_not_reordered(self):
        verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        gplt.add_patch(
            verts, indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32), mode="triangles"
        )
        _, ax = _preview_axes()
        assert len(ax.collections) == 1 and not ax.patches


class TestAspectSurvivesExport:
    """``set_aspect('equal')`` is baked into the live camera; the saved figure re-derives it."""

    @staticmethod
    def _ellipse():
        t = np.linspace(0, 2 * np.pi, 100)
        gplt.plot(10 * np.cos(t), np.sin(t))

    def test_default_is_auto(self):
        self._ellipse()
        _, ax = _preview_axes()
        assert ax.get_aspect() == "auto"

    @pytest.mark.parametrize(
        "call",
        [
            lambda: gplt.set_aspect("equal"),
            lambda: gplt.axis("equal"),
            lambda: gplt.axis("scaled"),
            lambda: gplt.gca().set_aspect("equal"),
        ],
    )
    def test_equal_aspect_reaches_the_export(self, call):
        self._ellipse()
        call()
        _, ax = _preview_axes()
        assert ax.get_aspect() == 1.0
        assert ax.get_adjustable() == "datalim"

    def test_equal_aspect_widens_the_short_axis_instead_of_cropping(self):
        self._ellipse()
        gplt.set_aspect("equal")
        fig, ax = _preview_axes()
        fig.canvas.draw()
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        assert x0 <= -10 and x1 >= 10  # nothing the caller had in view was cropped
        assert (y1 - y0) > 2  # y was widened past its own +-1 data range to match x's scale

    @pytest.mark.parametrize("undo", [lambda: gplt.set_aspect("auto"), lambda: gplt.axis("auto")])
    def test_auto_puts_it_back(self, undo):
        self._ellipse()
        gplt.set_aspect("equal")
        undo()
        _, ax = _preview_axes()
        assert ax.get_aspect() == "auto"

    def test_axis_tight_leaves_the_aspect_alone(self):
        self._ellipse()
        gplt.set_aspect("equal")
        gplt.axis("tight")
        _, ax = _preview_axes()
        assert ax.get_aspect() == 1.0

    def test_axis_square_is_not_a_lasting_aspect(self):
        self._ellipse()
        gplt.axis("square")
        _, ax = _preview_axes()
        assert ax.get_aspect() == "auto"

    def test_per_panel_in_a_subplot_grid(self):
        _, axs = gplt.subplots(1, 2)
        t = np.linspace(0, 2 * np.pi, 100)
        axs[0].plot(10 * np.cos(t), np.sin(t))
        axs[1].plot(10 * np.cos(t), np.sin(t))
        axs[0].set_aspect("equal")
        fig = _build_preview_figure(gplt.get_engine())
        assert [a.get_aspect() for a in fig.axes[:2]] == [1.0, "auto"]
