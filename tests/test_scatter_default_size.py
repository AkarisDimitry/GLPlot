"""The default scatter marker must be as large as matplotlib's, relative to the figure.

It was not: GLPlot's default size is a *pixel diameter* (10 px), and the PNG export handed
that number straight to ``ax.scatter(s=...)``, which reads it as a pt^2 **area** -- so a
default marker exported at 3.2 pt where matplotlib draws 6 pt. Compounding it, ``savefig``
renders at ``figsize * scale`` (scale=2) while still sizing markers in points, halving the
relative size again. Measured end to end, the default marker came out at 0.32x matplotlib's
relative diameter -- about a tenth of its area.

The last test here is a real pixel measurement rather than a unit assertion, because that
is the only thing that would have caught this: every individual number involved was
defensible on its own.

Agg only -- no GL window, no imgui (see the project's note on live-GL in pytest).
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import glplot.pyplot as gplt
from glplot.utils.preview import DEFAULT_SCATTER_SIZE_PT2

#: A colour nothing else in these figures uses, so the marker is unambiguous to find.
MARKER_RGB = (255, 0, 0)


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


def _relative_diameter(path) -> float:
    """The marker's width as a fraction of the image width.

    Relative, not absolute: the export renders on a 2x canvas, so raw pixel counts are not
    comparable between it and a plain matplotlib figure. Comparing the fraction of the
    figure the marker covers is the comparison a reader actually makes.
    """
    rgb = np.array(Image.open(path).convert("RGB")).astype(int)
    distance = np.abs(rgb - np.array(MARKER_RGB)).sum(axis=-1)
    ys, xs = np.nonzero(distance < 120)
    assert len(xs) > 0, f"no marker found in {path}"
    return (xs.max() - xs.min() + 1) / rgb.shape[1]


class TestSizeIsExplicitFlag:
    """The export needs to tell "nobody chose a size" from "somebody chose 10"."""

    def test_a_bare_scatter_records_no_chosen_size(self):
        gplt.figure()
        layer = gplt.scatter([0.0, 1.0], [0.0, 1.0])
        assert layer.metadata["size_is_explicit"] is False
        assert layer.style.point_size == 10.0, "the live pixel default must not change"

    @pytest.mark.parametrize(
        "kwargs,expected_px",
        [({"s": 100.0}, 100.0), ({"size": 7.0}, 7.0)],
    )
    def test_a_chosen_size_is_recorded_and_kept(self, kwargs, expected_px):
        gplt.figure()
        layer = gplt.scatter([0.0, 1.0], [0.0, 1.0], **kwargs)
        assert layer.metadata["size_is_explicit"] is True
        assert layer.style.point_size == expected_px

    def test_the_legacy_positional_size_still_counts_as_chosen(self):
        """``scatter(x, y, color, size)`` is GLPlot's own long-standing spelling."""
        gplt.figure()
        layer = gplt.scatter([0.0, 1.0], [0.0, 1.0], "red", 12.0)
        assert layer.metadata["size_is_explicit"] is True
        assert layer.style.point_size == 12.0

    def test_per_point_sizes_count_as_chosen(self):
        gplt.figure()
        layer = gplt.scatter([0.0, 1.0], [0.0, 1.0], s=[5.0, 9.0])
        assert layer.metadata["size_is_explicit"] is True
        assert layer.metadata["svalues"] is not None


class TestExportedMarkerSize:
    """What the reconstruction actually hands matplotlib."""

    def _exported_scatter_kwargs(self, monkeypatch, **scatter_kwargs):
        seen = {}

        def spy(self, *args, **kwargs):  # noqa: ANN001 - matplotlib Axes.scatter
            seen.update(kwargs)
            return _real(self, *args, **kwargs)

        from matplotlib.axes import Axes

        _real = Axes.scatter
        monkeypatch.setattr(Axes, "scatter", spy)

        gplt.figure(figsize=(4, 4))
        gplt.scatter([0.5], [0.5], color="red", **scatter_kwargs)
        gplt.savefig(str(self.path))
        return seen

    @pytest.fixture(autouse=True)
    def _tmp_path(self, tmp_path):
        self.path = tmp_path / "out.png"

    def test_unchosen_size_uses_the_matplotlib_scaled_default(self, monkeypatch):
        seen = self._exported_scatter_kwargs(monkeypatch)
        assert seen["s"] == DEFAULT_SCATTER_SIZE_PT2

    def test_a_chosen_size_is_passed_through_untouched(self, monkeypatch):
        """Every gallery figure that tuned ``s=`` must keep exporting identically."""
        seen = self._exported_scatter_kwargs(monkeypatch, s=110.0)
        assert seen["s"] == 110.0


class TestRenderedAgainstMatplotlib:
    """The end-to-end check: does a default marker *look* matplotlib-sized?"""

    def _matplotlib_reference(self, path) -> float:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as mpl

        fig, ax = mpl.subplots(figsize=(4, 4), dpi=120)
        ax.scatter([0.5], [0.5], color="red")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.savefig(str(path), dpi=120)
        mpl.close(fig)
        return _relative_diameter(path)

    def test_default_marker_matches_matplotlibs_relative_size(self, tmp_path):
        reference = self._matplotlib_reference(tmp_path / "mpl.png")

        gplt.figure(figsize=(4, 4))
        gplt.scatter([0.5], [0.5], color="red")
        out = tmp_path / "gl.png"
        gplt.savefig(str(out))
        measured = _relative_diameter(out)

        ratio = measured / reference
        # Generous either way: this is a "is it the same order of size" guard, not a
        # pixel-exact pin. It fails loudly at the 0.32x this used to be, and would fail
        # just as loudly if someone over-corrected.
        assert 0.75 <= ratio <= 1.35, (
            f"default marker is {ratio:.2f}x matplotlib's relative diameter "
            f"({measured:.4f} vs {reference:.4f} of figure width)"
        )

    def test_a_chosen_size_still_scales_with_the_number(self, tmp_path):
        """Sanity: the explicit path is untouched and still monotonic."""
        sizes = []
        for s in (40.0, 400.0):
            gplt._cleanup_pyplot_state()
            gplt.figure(figsize=(4, 4))
            gplt.scatter([0.5], [0.5], color="red", s=s)
            out = tmp_path / f"gl_{s}.png"
            gplt.savefig(str(out))
            sizes.append(_relative_diameter(out))
        assert sizes[1] > sizes[0]
