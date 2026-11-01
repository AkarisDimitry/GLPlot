"""A log (or symlog/asinh) axis set *after* the data exports with the data in frame.

``xscale("log")`` re-fits the camera in log10 space and pins the view; the export used to hand
those log10 numbers to a matplotlib axis as if they were data, so ``plot(); xscale("log")`` --
the usual order -- saved an empty plot. Setting the scale first, or ``loglog()``, always worked
because the view was then left to matplotlib's own autoscale.
"""

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.utils.preview import _build_preview_figure

F = np.logspace(0, 4, 200)
GAIN = 1.0 / np.sqrt(1.0 + (F / 300.0) ** 2)


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


def _limits():
    fig = _build_preview_figure(gplt.get_engine())
    ax = fig.axes[0]
    fig.canvas.draw()
    return ax.get_xlim(), ax.get_ylim()


def _assert_frames(xlim, ylim):
    assert xlim[0] <= F.min() and xlim[1] >= F.max()
    assert ylim[0] <= GAIN.min() and ylim[1] >= GAIN.max()


def test_scale_set_after_plotting_frames_the_data():
    gplt.plot(F, GAIN)
    gplt.xscale("log")
    gplt.yscale("log")
    _assert_frames(*_limits())


def test_scale_set_before_plotting_frames_the_data():
    gplt.xscale("log")
    gplt.yscale("log")
    gplt.plot(F, GAIN)
    _assert_frames(*_limits())


def test_loglog_frames_the_data():
    gplt.loglog(F, GAIN)
    _assert_frames(*_limits())


def test_one_log_axis_leaves_the_other_linear():
    gplt.plot(F, GAIN)
    gplt.xscale("log")
    xlim, ylim = _limits()
    assert xlim[0] <= F.min() and xlim[1] >= F.max()
    assert ylim[0] <= GAIN.min() and ylim[1] >= GAIN.max()
    assert ylim[1] < 2.0  # still the linear 0..1 range, not a log10 number


@pytest.mark.parametrize("scale", ["symlog", "asinh"])
def test_other_real_scales_frame_the_data(scale):
    x = np.linspace(-50, 50, 200)
    gplt.plot(x, x**3)
    gplt.xscale(scale)
    gplt.yscale(scale)
    xlim, ylim = _limits()
    assert xlim[0] <= -50 and xlim[1] >= 50
    assert ylim[0] <= (-50) ** 3 and ylim[1] >= 50**3


def test_linear_axes_are_unchanged():
    gplt.plot(F, GAIN)
    xlim, _ = _limits()
    assert xlim[0] < 0 and xlim[1] > F.max()  # matplotlib-style margins around 1..1e4
    assert xlim[1] < 2 * F.max()


def test_each_panel_uses_its_own_scale():
    _, axs = gplt.subplots(1, 2)
    axs[0].plot(F, GAIN)
    axs[1].plot(F, GAIN)
    axs[0].set_xscale("log")
    fig = _build_preview_figure(gplt.get_engine())
    fig.canvas.draw()
    left, right = fig.axes[0], fig.axes[1]
    assert left.get_xscale() == "log" and right.get_xscale() == "linear"
    assert left.get_xlim()[0] > 0 and left.get_xlim()[1] >= F.max()
    assert right.get_xlim()[0] <= 0 or right.get_xlim()[1] > F.max()
