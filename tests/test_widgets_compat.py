"""Test glplot.widgets: matplotlib's widget API on GLPlot's native controls.

The reference is matplotlib's own Slider demo -- ``fig.add_axes`` for the widget, an
``update`` callback ending in ``fig.canvas.draw_idle()`` -- which must run unchanged
apart from the import.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import glplot.pyplot as plt
from glplot import controls as ctl
from glplot.widgets import Button, CheckButtons, RadioButtons, RangeSlider, Slider, TextBox


@pytest.fixture(autouse=True)
def clean_state():
    plt._cleanup_pyplot_state()
    yield
    plt._cleanup_pyplot_state()


def _widget_axes(fig):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the rectangle is not honoured, and says so
        return fig.add_axes([0.25, 0.1, 0.65, 0.03])


def test_the_matplotlib_slider_demo_runs():
    t = np.linspace(0, 1, 500)
    fig, ax = plt.subplots()
    (line,) = ax.plot(t, np.sin(2 * np.pi * 3 * t))
    freq = Slider(_widget_axes(fig), "Frequency", 0.1, 30, valinit=3)
    amp = Slider(
        ax=_widget_axes(fig),
        label="Amplitude",
        valmin=0,
        valmax=10,
        valinit=5,
        orientation="vertical",
    )

    def update(val):
        line.set_ydata(amp.val * np.sin(2 * np.pi * freq.val * t))
        fig.canvas.draw_idle()

    freq.on_changed(update)
    amp.on_changed(update)
    reset = Button(_widget_axes(fig), "Reset", hovercolor="0.975")
    reset.on_clicked(lambda event: (freq.reset(), amp.reset()))

    freq.set_val(10)
    assert np.abs(line.get_ydata()).max() == pytest.approx(5, abs=1e-3)
    amp.val = 2
    reset.click()
    assert (freq.val, amp.val) == (3.0, 5.0)
    assert fig.controls.keys() == ["frequency", "amplitude", "reset"]
    assert isinstance(freq.control, ctl.Slider)


def test_pyplot_exports_the_widget_classes():
    assert plt.Slider is Slider and plt.Button is Button


def test_callbacks_receive_matplotlib_arguments():
    plt.figure()
    b = Button(None, "Go")
    events = []
    b.on_clicked(events.append)
    b.click()
    assert events[0].name == "button_press_event" and events[0].button == 1


def test_disconnect_and_eventson():
    plt.figure()
    s = Slider(None, "s", 0, 1)
    seen = []
    cid = s.on_changed(seen.append)
    s.eventson = False
    s.set_val(0.2)
    s.eventson = True
    s.set_val(0.3)
    s.disconnect(cid)
    s.set_val(0.4)
    assert seen == [0.3]


def test_slider_valstep_and_range_changes():
    plt.figure()
    s = Slider(None, "s", 0, 10, valinit=2, valstep=0.5)
    s.set_val(3.3)
    assert s.val == 3.5
    s.set_max(3)
    assert s.val == 3.0 and s.valmax == 3.0
    with pytest.raises(ValueError, match="array valstep"):
        Slider(None, "t", 0, 1, valstep=[0, 0.5, 1])
    s.set_active(False)
    assert not s.control.enabled and not s.get_active()


def test_range_slider():
    plt.figure()
    r = RangeSlider(None, "band", 0, 100, valinit=(20, 60))
    seen = []
    r.on_changed(seen.append)
    r.set_max(80)
    assert r.val == (20.0, 80.0) and seen == [(20.0, 80.0)]


def test_check_buttons():
    plt.figure()
    cb = CheckButtons(None, ["a", "b", "c"], [True, False, True])
    toggled = []
    cid = cb.on_clicked(toggled.append)
    cb.set_active(1)
    assert cb.get_status() == [True, True, True] and toggled == ["b"]
    assert cb.get_checked_labels() == ["a", "b", "c"]
    cb.disconnect(cid)
    cb.clear()
    assert cb.get_status() == [False] * 3 and toggled == ["b"]
    with pytest.raises(ValueError):
        CheckButtons(None, ["a"], [True, False])


def test_radio_buttons():
    plt.figure()
    rb = RadioButtons(None, ["x", "y", "z"], active=1)
    picked = []
    rb.on_clicked(picked.append)
    rb.set_active(2)
    assert (rb.value_selected, rb.index_selected, picked) == ("z", 2, ["z"])
    assert rb.control.style == "radio"
    with pytest.raises(ValueError):
        rb.set_active(7)


def test_text_box():
    plt.figure()
    tb = TextBox(None, "f(x) =", "x**2")
    submitted = []
    tb.on_submit(submitted.append)
    assert not tb.control.live
    tb.set_val("x**3")
    assert tb.text == "x**3" and submitted == ["x**3"]
    tb.on_text_change(lambda text: None)
    assert tb.control.live


def test_the_figure_canvas_is_not_an_event_bus():
    fig = plt.figure()
    fig.frame.dirty_scene = False
    fig.canvas.draw_idle()
    assert fig.frame.dirty_scene and fig.canvas.figure is fig
    assert not hasattr(fig.canvas, "mpl_connect") and not hasattr(fig.canvas, "get_renderer")
