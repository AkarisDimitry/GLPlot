"""Test the raster handoff in ``glplot.utils.mpl_bridge``.

A snapshot is a picture plus a claim about what the picture means: ``rgba`` is the
pixels, ``extent`` says which world rectangle they cover. The bridge is where that claim
gets cashed, and both halves of it are silently breakable -- a raster handed over upside
down still renders, and an extent that disagrees with the pixels still draws axes. Both
produce a plausible plot that is simply wrong, which is exactly the kind of bug that
survives a code review. So the orientation contract is pinned here with an asymmetric
pattern, and the 3D case with the fact that a projection has no 2D data mapping at all.

No OpenGL and no GPU: the snapshots are built by hand.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from glplot.utils.mpl_bridge import GLPlotSnapshot, snapshot_to_matplotlib

RED = (255, 0, 0)
BLUE = (0, 0, 255)


def _snapshot(**overrides) -> GLPlotSnapshot:
    """A snapshot whose top half is red and bottom half blue, in top-row-first order.

    That is the order ``capture_snapshot`` produces: ``glReadPixels`` reads bottom-up and
    the flip puts row 0 back on top, matching what every image library assumes.
    """
    h, w = 40, 20
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[: h // 2] = [*RED, 255]
    rgba[h // 2 :] = [*BLUE, 255]
    kwargs = dict(
        rgba=rgba,
        extent=(0.0, 10.0, 100.0, 200.0),
        xlim=(0.0, 10.0),
        ylim=(100.0, 200.0),
        width_px=w,
        height_px=h,
        transparent=False,
    )
    kwargs.update(overrides)
    return GLPlotSnapshot(**kwargs)


def _color_at(fig, ax, x: float, y: float):
    """The rendered RGB at data coordinate ``(x, y)`` -- what a reader actually sees."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    px, py = ax.transData.transform((x, y))
    return tuple(buf[int(round(buf.shape[0] - py)), int(round(px))][:3])


class TestOrientation:
    """Row 0 of ``rgba`` is the top row, and it must land at the top of the axes."""

    def test_top_row_renders_at_high_y(self):
        fig, ax, _ = snapshot_to_matplotlib(_snapshot())
        assert _color_at(fig, ax, 5.0, 195.0) == RED
        matplotlib.pyplot.close(fig)

    def test_bottom_row_renders_at_low_y(self):
        fig, ax, _ = snapshot_to_matplotlib(_snapshot())
        assert _color_at(fig, ax, 5.0, 105.0) == BLUE
        matplotlib.pyplot.close(fig)

    def test_y_axis_is_not_inverted(self):
        """The axes must read bottom-up, or the ticks contradict the picture."""
        fig, ax, _ = snapshot_to_matplotlib(_snapshot())
        bottom, top = ax.get_ylim()
        assert bottom < top
        matplotlib.pyplot.close(fig)


class TestExtent:
    def test_limits_follow_the_extent(self):
        fig, ax, _ = snapshot_to_matplotlib(_snapshot())
        assert ax.get_xlim() == pytest.approx((0.0, 10.0))
        assert ax.get_ylim() == pytest.approx((100.0, 200.0))
        matplotlib.pyplot.close(fig)


class TestProjected3D:
    """A 3D projection has no 2D data mapping, so it must not be given data axes."""

    def test_axes_are_hidden(self):
        fig, ax, _ = snapshot_to_matplotlib(_snapshot(projected_3d=True))
        assert ax.axison is False
        matplotlib.pyplot.close(fig)

    def test_no_data_extent_is_claimed(self):
        """Ticks pinned to `extent` would label a rotated volume with 2D numbers."""
        fig, ax, artist = snapshot_to_matplotlib(_snapshot(projected_3d=True))
        left, right, bottom, top = artist.get_extent()
        assert (left, right) != (0.0, 10.0)
        assert (bottom, top) != (100.0, 200.0)
        matplotlib.pyplot.close(fig)

    def test_projection_is_not_stretched(self):
        fig, ax, _ = snapshot_to_matplotlib(_snapshot(projected_3d=True))
        assert ax.get_aspect() == 1.0
        matplotlib.pyplot.close(fig)

    def test_defaults_to_flat(self):
        """Only a 3D scene opts in; everything else keeps its data axes."""
        assert _snapshot().projected_3d is False
