"""matplotlib.widgets, on GLPlot's native controls.

``Slider``, ``RangeSlider``, ``Button``, ``CheckButtons``, ``RadioButtons`` and ``TextBox``
with matplotlib's constructor signatures and callback API, so a matplotlib widget script
runs with its import changed::

    from glplot.widgets import Slider, Button          # was: from matplotlib.widgets ...

    fig, ax = plt.subplots()
    (line,) = ax.plot(t, np.sin(2 * np.pi * 3 * t))
    freq = Slider(fig.add_axes([0.25, 0.1, 0.65, 0.03]), "Frequency", 0.1, 30, valinit=3)

    def update(val):
        line.set_ydata(np.sin(2 * np.pi * freq.val * t))
        fig.canvas.draw_idle()

    freq.on_changed(update)

Each wrapper owns a :mod:`glplot.controls` control, which the workstation's Controls panel
draws; the callbacks run through the same frame-coalesced pipeline as ``gplt.slider``.

**Where the widgets appear.** matplotlib positions a widget inside the axes it is given.
GLPlot draws every control in the Controls panel instead, so the ``ax`` argument only
identifies the figure. The usual ``fig.add_axes([...])`` for a widget still works (it warns
that the rectangle is not honoured); ``ax=None`` means the current figure.

Not reproduced: visual styling (colours, handle style, orientation) and ``useblit``, which
have no meaning for a panel control. They are accepted and ignored.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from . import controls as ctl

__all__ = ["Button", "CheckButtons", "RadioButtons", "RangeSlider", "Slider", "TextBox", "Widget"]


def _figure_for(ax: Any) -> Any:
    """The figure a widget's ``ax`` belongs to (the current figure for ``None``)."""
    from . import pyplot as _pp
    from .engine import GPULinePlot

    if ax is None:
        return _pp.gcf()
    if isinstance(ax, GPULinePlot):
        return ax
    # By type, not duck-typing: AxesProxy forwards unknown attributes to pyplot functions,
    # so `hasattr(ax, "controls")` is True for the wrong reason.
    fig = ax.__dict__.get("_fig") if hasattr(ax, "__dict__") else None
    if isinstance(fig, GPULinePlot):
        return fig
    return _pp.gcf()


def _label_text(label: Any, fallback: str) -> str:
    text = str(label).strip() if label is not None else ""
    return text or fallback


class _Event:
    """The stand-in for matplotlib's ``MouseEvent`` handed to ``Button.on_clicked``.

    Callbacks almost never read it; the attributes a handler might test exist.
    """

    def __init__(self, ax: Any, name: str = "button_press_event") -> None:
        self.name = name
        self.inaxes = ax
        self.button = 1
        self.canvas = getattr(_figure_for(ax), "canvas", None)
        self.x = self.y = self.xdata = self.ydata = None
        self.key = None
        self.dblclick = False


class Widget:
    """Base: the owned control, and matplotlib's connect/disconnect bookkeeping."""

    def __init__(self, ax: Any, control: ctl.Control) -> None:
        self.ax = ax
        self.figure = _figure_for(ax)
        self._control = self.figure.controls.add(control)
        self._cids: Dict[int, Callable[..., Any]] = {}
        self._counter = itertools.count()
        #: matplotlib's global switch: with it False, no callback fires.
        self.eventson = True

    @property
    def control(self) -> ctl.Control:
        """The native :mod:`glplot.controls` control behind this widget."""
        return self._control

    @property
    def active(self) -> bool:
        return self._control.enabled

    @active.setter
    def active(self, value: bool) -> None:
        self._control.enabled = bool(value)

    def set_active(self, active: bool) -> None:
        self.active = active

    def get_active(self) -> bool:
        return self.active

    def ignore(self, event: Any) -> bool:
        return not self.active

    def _connect(self, fn: Callable[..., Any], adapt: Callable[[Any], Any]) -> int:
        def observer(value: Any) -> None:
            if self.eventson:
                fn(adapt(value))

        cid = next(self._counter)
        self._cids[cid] = observer
        self._control.on_change(observer)
        return cid

    def disconnect(self, cid: int) -> None:
        """Remove the callback with connection id ``cid``."""
        observer = self._cids.pop(cid, None)
        if observer is not None:
            self._control.disconnect(observer)

    def disconnect_events(self) -> None:
        for cid in list(self._cids):
            self.disconnect(cid)


class Slider(Widget):
    """``matplotlib.widgets.Slider``: ``val``, ``set_val``, ``on_changed``, ``reset``."""

    def __init__(
        self,
        ax: Any,
        label: str,
        valmin: float,
        valmax: float,
        *,
        valinit: float = 0.5,
        valfmt: Optional[str] = None,
        closedmin: bool = True,
        closedmax: bool = True,
        slidermin: Any = None,
        slidermax: Any = None,
        dragging: bool = True,
        valstep: Any = None,
        orientation: str = "horizontal",
        **kwargs: Any,
    ) -> None:
        step = None
        if valstep is not None:
            if np.ndim(valstep) != 0:
                raise ValueError(
                    "glplot.widgets.Slider: an array valstep is not supported; use a "
                    "RadioButtons / gplt.dropdown for a fixed set of values"
                )
            step = float(valstep)
        self.valmin, self.valmax = float(valmin), float(valmax)
        self.valinit = valinit
        self.valstep = valstep
        self.orientation = orientation
        self.label = _label_text(label, "Slider")
        control = ctl.Slider(
            self.label,
            self.valmin,
            self.valmax,
            valinit,
            step,
            fmt=valfmt,
            live=bool(dragging),
            key=kwargs.pop("key", None),
            help=kwargs.pop("help", ""),
        )
        super().__init__(ax, control)

    @property
    def val(self) -> float:
        return self._control.value

    @val.setter
    def val(self, value: float) -> None:
        self.set_val(value)

    def set_val(self, val: float) -> None:
        self._control.set(val)

    def set_min(self, vmin: float) -> None:
        """Move the lower end of the range (the value is re-clamped)."""
        self._rebuild(min_=float(vmin))

    def set_max(self, vmax: float) -> None:
        """Move the upper end of the range (the value is re-clamped)."""
        self._rebuild(max_=float(vmax))

    def _rebuild(self, *, min_: Optional[float] = None, max_: Optional[float] = None) -> None:
        c = self._control
        lo = c.min if min_ is None else min_
        hi = c.max if max_ is None else max_
        if not lo < hi:
            raise ValueError(f"slider range must satisfy min < max, got [{lo}, {hi}]")
        c.min, c.max = lo, hi
        self.valmin, self.valmax = lo, hi
        c.set(c.value)  # re-clamp through the normal path so callbacks see the change

    def on_changed(self, func: Callable[[float], Any]) -> int:
        """Call ``func(val)`` whenever the value changes. Returns a connection id."""
        return self._connect(func, lambda v: v)

    def reset(self) -> None:
        self._control.reset()


class RangeSlider(Widget):
    """``matplotlib.widgets.RangeSlider``: ``val`` is a ``(min, max)`` tuple."""

    def __init__(
        self,
        ax: Any,
        label: str,
        valmin: float,
        valmax: float,
        *,
        valinit: Optional[Sequence[float]] = None,
        valfmt: Optional[str] = None,
        valstep: Any = None,
        dragging: bool = True,
        **kwargs: Any,
    ) -> None:
        step = None if valstep is None else float(valstep)
        self.valmin, self.valmax = float(valmin), float(valmax)
        self.valinit = valinit
        self.label = _label_text(label, "Range")
        control = ctl.RangeSlider(
            self.label,
            self.valmin,
            self.valmax,
            None if valinit is None else tuple(valinit),
            step,
            fmt=valfmt,
            live=bool(dragging),
            key=kwargs.pop("key", None),
        )
        super().__init__(ax, control)

    @property
    def val(self) -> tuple:
        return self._control.value

    def set_val(self, val: Sequence[float]) -> None:
        self._control.set(tuple(val))

    def set_min(self, vmin: float) -> None:
        self.set_val((vmin, self.val[1]))

    def set_max(self, vmax: float) -> None:
        self.set_val((self.val[0], vmax))

    def on_changed(self, func: Callable[[tuple], Any]) -> int:
        return self._connect(func, lambda v: v)

    def reset(self) -> None:
        self._control.reset()


class Button(Widget):
    """``matplotlib.widgets.Button``: ``on_clicked(func)``, ``func(event)`` per click."""

    def __init__(
        self,
        ax: Any,
        label: str,
        image: Any = None,
        color: Any = "0.85",
        hovercolor: Any = "0.95",
        *,
        useblit: bool = True,
        **kwargs: Any,
    ) -> None:
        self.label = _label_text(label, "Button")
        control = ctl.Button(
            self.label,
            shortcut=kwargs.pop("shortcut", None),
            confirm=kwargs.pop("confirm", None),
            key=kwargs.pop("key", None),
            help=kwargs.pop("help", ""),
        )
        super().__init__(ax, control)

    def on_clicked(self, func: Callable[[Any], Any]) -> int:
        """Call ``func(event)`` on every click. Returns a connection id."""
        return self._connect(func, lambda _button: _Event(self.ax))

    def click(self) -> None:
        """Press the button from code."""
        self._control.click()


class CheckButtons(Widget):
    """``matplotlib.widgets.CheckButtons``: one checkbox per label.

    ``on_clicked(func)`` receives the label that was toggled; ``get_status()`` the states.
    """

    def __init__(
        self,
        ax: Any,
        labels: Sequence[str],
        actives: Optional[Sequence[bool]] = None,
        *,
        useblit: bool = True,
        **kwargs: Any,
    ) -> None:
        self.labels = [str(label) for label in labels]
        if not self.labels:
            raise ValueError("CheckButtons needs at least one label")
        states = list(actives) if actives is not None else [False] * len(self.labels)
        if len(states) != len(self.labels):
            raise ValueError("actives must have one entry per label")
        self.ax = ax
        self.figure = _figure_for(ax)
        self._checks: List[ctl.Checkbox] = [
            self.figure.controls.add(ctl.Checkbox(label, bool(state)))
            for label, state in zip(self.labels, states)
        ]
        self._control = self._checks[0]
        self._cids: Dict[int, List[Callable[..., Any]]] = {}  # type: ignore[assignment]
        self._counter = itertools.count()
        self.eventson = True

    @property
    def active(self) -> bool:
        return all(c.enabled for c in self._checks)

    @active.setter
    def active(self, value: bool) -> None:
        for c in self._checks:
            c.enabled = bool(value)

    def get_status(self) -> List[bool]:
        return [bool(c.value) for c in self._checks]

    def get_checked_labels(self) -> List[str]:
        return [label for label, c in zip(self.labels, self._checks) if c.value]

    def set_active(self, index: int, state: Optional[bool] = None) -> None:  # type: ignore[override]  # noqa: E501
        """Toggle (or set) the ``index``-th box -- matplotlib's meaning, not "enable"."""
        if not 0 <= index < len(self._checks):
            raise ValueError(f"invalid CheckButtons index: {index}")
        check = self._checks[index]
        check.set(not check.value if state is None else bool(state))

    def clear(self) -> None:
        for c in self._checks:
            c.set(False)

    def on_clicked(self, func: Callable[[str], Any]) -> int:
        cid = next(self._counter)
        observers = []
        for label, check in zip(self.labels, self._checks):

            def observer(_value: Any, label: str = label) -> None:
                if self.eventson:
                    func(label)

            check.on_change(observer)
            observers.append(observer)
        self._cids[cid] = observers  # type: ignore[assignment]
        return cid

    def disconnect(self, cid: int) -> None:
        for check, observer in zip(self._checks, self._cids.pop(cid, [])):  # type: ignore[arg-type]
            check.disconnect(observer)


class RadioButtons(Widget):
    """``matplotlib.widgets.RadioButtons``: ``value_selected``, ``index_selected``."""

    def __init__(
        self,
        ax: Any,
        labels: Sequence[str],
        active: int = 0,
        activecolor: Any = None,
        *,
        useblit: bool = True,
        **kwargs: Any,
    ) -> None:
        self.labels = [str(label) for label in labels]
        if not 0 <= int(active) < len(self.labels):
            raise ValueError(f"active index {active} out of range")
        title = _label_text(kwargs.pop("label", None), "Choice")
        style = "radio" if len(self.labels) <= 5 else "combo"
        control = ctl.Dropdown(
            title,
            self.labels,
            self.labels[int(active)],
            style=style,
            key=kwargs.pop("key", None),
        )
        super().__init__(ax, control)

    @property
    def value_selected(self) -> str:
        return self._control.value

    @property
    def index_selected(self) -> int:
        return self._control.index

    def set_active(self, index: int) -> None:  # type: ignore[override]
        """Select the ``index``-th option -- matplotlib's meaning, not "enable"."""
        if not 0 <= index < len(self.labels):
            raise ValueError(f"invalid RadioButton index: {index}")
        self._control.set(self.labels[index])

    def on_clicked(self, func: Callable[[str], Any]) -> int:
        return self._connect(func, lambda v: v)


class TextBox(Widget):
    """``matplotlib.widgets.TextBox``: ``on_submit`` (Enter / focus loss), ``on_text_change``."""

    def __init__(
        self,
        ax: Any,
        label: str,
        initial: str = "",
        *,
        textalignment: str = "left",
        **kwargs: Any,
    ) -> None:
        self.label = _label_text(label, "Text")
        control = ctl.TextBox(self.label, initial, key=kwargs.pop("key", None))
        super().__init__(ax, control)
        self._change_cids: Dict[int, Callable[..., Any]] = {}

    @property
    def text(self) -> str:
        return self._control.value

    def set_val(self, val: str) -> None:
        self._control.set(val)

    def on_submit(self, func: Callable[[str], Any]) -> int:
        """Call ``func(text)`` when the text is committed."""
        return self._connect(func, lambda v: v)

    def on_text_change(self, func: Callable[[str], Any]) -> int:
        """Call ``func(text)`` as the user types (switches the box to per-key updates)."""
        self._control.live = True
        return self._connect(func, lambda v: v)

    def begin_typing(self) -> None:
        """No-op: focus is the panel's business."""

    def stop_typing(self) -> None:
        """No-op: focus is the panel's business."""
