"""Interactive controls: sliders, buttons, toggles and menus bound to Python callbacks.

This is the model behind ``gplt.slider``, ``gplt.button``, ``gplt.interact`` and the
workstation's **Controls** panel. It is pure logic -- no imgui, no GL -- so every rule in
here is testable headless, and a script that defines controls runs unchanged under
``savefig`` (the controls simply keep their current values).

Three objects matter:

* :class:`Control` and its subclasses -- one per widget. A control owns a *value*, coerces
  and validates everything assigned to it (a slider clamps and snaps to its step, a
  dropdown refuses an option it does not have), and remembers its default so it can be
  reset.
* :class:`Binding` -- a function wired to a set of controls. It re-runs with the current
  values whenever one of them changes. What it returns is put to work: an ``(x, y)`` pair
  is drawn as a curve the binding owns and keeps up to date, and a mapping is shown in
  the panel as a table of results.
* :class:`ControlSet` -- the per-figure registry (``fig.controls``). It orders the
  controls into groups, owns the bindings, and runs the change pipeline.

**When callbacks run.** Outside the window (a script before ``show()``, a test, a
headless export) a change runs its callbacks *immediately*, like any Python call, and an
exception propagates to whoever made the change -- a typo in a callback is a normal
traceback. While the window is open, changes are *coalesced*: the set records which
controls moved, and the engine runs the callbacks once at the top of the next frame, on
the GL thread, with the latest values. Dragging a slider across 40 positions in one frame
costs one callback run, not 40, and callbacks may freely mutate the scene because they
are not running inside an imgui draw callback (CONTRACT §1.1). Inside the window an
exception is caught, recorded on the control or binding that raised it, and shown in the
panel instead of tearing down the render loop.

**Buttons are events, not values.** A button's clicks are queued individually, so two
clicks in one frame run the callback twice; a slider moved twice runs its callbacks once.
"""

from __future__ import annotations

import builtins
import inspect
import json
import logging
import math
import re
import threading
import time
import traceback
import weakref
from collections import OrderedDict, deque
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    ClassVar,
    Deque,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "Binding",
    "Button",
    "Checkbox",
    "ColorPicker",
    "Control",
    "ControlError",
    "ControlSet",
    "Dropdown",
    "Number",
    "RangeSlider",
    "Readout",
    "Slider",
    "TextBox",
    "control_from_spec",
]

#: How many rounds of "a callback changed another control" a single flush follows before
#: it concludes two callbacks are feeding each other and stops. Legitimate chains (a
#: button that moves a slider that redraws a curve) are two or three rounds deep.
MAX_CASCADE_ROUNDS = 32

#: Default group name for controls created without one.
DEFAULT_GROUP = "General"

#: Longest text a :class:`TextBox` accepts. Also the imgui input buffer size.
MAX_TEXT_LENGTH = 4096

_KEY_SANITISER = re.compile(r"[^0-9a-zA-Z_]+")


class ControlError(ValueError):
    """A control was defined or assigned with something it cannot represent."""


def _slug(label: str) -> str:
    """``"Cut-off (Hz)"`` -> ``"cut_off_hz"``: a stable Python identifier for a label."""
    key = _KEY_SANITISER.sub("_", str(label).strip()).strip("_").lower()
    if not key:
        key = "control"
    if key[0].isdigit():
        key = "_" + key
    return key


def _pretty(name: str) -> str:
    """``"cutoff_hz"`` -> ``"Cutoff hz"``: a readable label for a parameter name."""
    text = str(name).replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else str(name)


def _finite(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ControlError(f"{what}: expected a number, got {value!r}") from None
    if not math.isfinite(number):
        raise ControlError(f"{what}: expected a finite number, got {value!r}")
    return number


def _decimals_for(step: float) -> int:
    """How many decimals display a multiple of ``step`` exactly (0.25 -> 2, 5 -> 0)."""
    if step <= 0:
        return 3
    text = f"{step:.10f}".rstrip("0")
    return len(text.split(".")[1]) if "." in text else 0


class _Callback:
    """A user callback plus how many positional arguments it wants (0 or 1).

    ``on_change(lambda: ...)`` and ``on_change(lambda value: ...)`` are both natural, and
    matplotlib's ``Slider.on_changed`` passes the value, so the arity is read once here
    instead of making every user write ``*_``.
    """

    __slots__ = ("fn", "takes_arg")

    def __init__(self, fn: Callable[..., Any]) -> None:
        if not callable(fn):
            raise ControlError(f"callback must be callable, got {fn!r}")
        self.fn = fn
        self.takes_arg = _accepts_positional(fn)

    def __call__(self, arg: Any) -> Any:
        return self.fn(arg) if self.takes_arg else self.fn()


def _accepts_positional(fn: Callable[..., Any]) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):  # builtins without a signature: assume they take one
        return True
    for param in params:
        if param.kind in (param.VAR_POSITIONAL, param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            return True
    return False


# ======================================================================================
# Controls
# ======================================================================================


class Control:
    """Base class: a labelled value with validation, a default and change observers.

    Subclasses set :attr:`kind`, implement :meth:`_coerce`, and assign their own
    attributes *before* calling ``super().__init__`` (which coerces the initial value).
    """

    #: Widget type, used by the panel to pick a renderer and by :meth:`describe`.
    kind: ClassVar[str] = "control"
    #: False for controls that do not feed bindings (buttons, readouts).
    is_input: ClassVar[bool] = True

    def __init__(
        self,
        label: str,
        value: Any,
        *,
        key: Optional[str] = None,
        help: str = "",  # noqa: A002 - the natural name for a tooltip, as in argparse
        group: Optional[str] = None,
        enabled: bool = True,
        visible: bool = True,
        on_change: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not isinstance(label, str) or not label.strip():
            raise ControlError(f"a control needs a non-empty text label, got {label!r}")
        self.label = label.strip()
        if key is None:
            key = _slug(self.label)
        elif not (isinstance(key, str) and key.isidentifier()):
            raise ControlError(f"control key must be a Python identifier, got {key!r}")
        #: Identifier used by ``controls["key"]``, snapshots and ``interact`` matching.
        self.key: str = key
        self.help = str(help or "")
        self.group: Optional[str] = group
        #: A disabled control is drawn greyed out and ignores the GUI.
        self.enabled = bool(enabled)
        #: A hidden control keeps its value and its bindings; it is just not drawn.
        self.visible = bool(visible)
        #: Message of the last exception raised by one of this control's observers.
        self.error: Optional[str] = None
        self._owner: Optional[ControlSet] = None
        self._observers: List[_Callback] = []
        self._value = self._coerce(value)
        #: What :meth:`reset` returns to.
        self.default = self._value
        #: Bumped on every accepted change; lets a GUI notice a change it did not make.
        self.version = 0
        if on_change is not None:
            self.on_change(on_change)

    # -- value -----------------------------------------------------------------------

    @property
    def value(self) -> Any:
        """The current value. Assigning goes through :meth:`set` (validation included)."""
        return self._value

    @value.setter
    def value(self, new: Any) -> None:
        self.set(new)

    def set(self, value: Any) -> bool:
        """Validate and store ``value``; run the change pipeline if it actually changed.

        Returns True when the value changed. Raises :class:`ControlError` for a value the
        control cannot represent (the old value is kept).
        """
        if not self.is_input and self.kind != "readout":
            raise ControlError(f"{self.kind} {self.label!r} has no settable value")
        new = self._coerce(value)
        if self._same(new, self._value):
            return False
        self._value = new
        self.version += 1
        self._changed()
        return True

    def reset(self) -> bool:
        """Return to :attr:`default`. Returns True if that was a change."""
        return self.set(self.default)

    def _coerce(self, value: Any) -> Any:  # pragma: no cover - abstract
        return value

    @staticmethod
    def _same(a: Any, b: Any) -> bool:
        try:
            return bool(a == b)
        except Exception:  # e.g. comparing ndarrays: treat as changed
            return a is b

    # -- observers -------------------------------------------------------------------

    def on_change(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Call ``fn(value)`` (or ``fn()``) after every change. Usable as a decorator."""
        self._observers.append(_Callback(fn))
        return fn

    def disconnect(self, fn: Callable[..., Any]) -> bool:
        """Remove an observer added with :meth:`on_change`. Returns True if it was found."""
        for i, cb in enumerate(self._observers):
            if cb.fn is fn:
                del self._observers[i]
                return True
        return False

    def _changed(self) -> None:
        owner = self._owner
        if owner is not None:
            owner._notify(self)
            return
        # A free-standing control (not yet added to a figure) behaves like a plain
        # observable: observers run now, and their exceptions are the caller's.
        for cb in list(self._observers):
            cb(self._value)

    # -- serialisation ---------------------------------------------------------------

    def to_json(self) -> Any:
        """The value as something ``json.dumps`` accepts."""
        return self._value

    def describe(self) -> Dict[str, Any]:
        """A plain-dict summary, for debugging, docs and the panel's "copy" action."""
        return {"kind": self.kind, "key": self.key, "label": self.label, "value": self.to_json()}

    @property
    def owner(self) -> Optional["ControlSet"]:
        """The :class:`ControlSet` this control belongs to, if any."""
        return self._owner

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.key}={self._value!r}>"


class Slider(Control):
    """A number chosen from ``[min, max]``, optionally snapped to ``step``.

    Args:
        label: Text shown beside the slider.
        min, max: The range. ``min < max``; both finite.
        value: Initial value; the middle of the range by default (the geometric middle
            for a ``log`` slider).
        step: Snap to ``min + k*step``. ``None`` means continuous.
        integer: Hold an ``int``. Implies a step of at least 1.
        log: Logarithmic travel -- for a frequency or a tolerance spanning decades.
            Requires ``min > 0``.
        fmt: printf-style display format (``"%.2f Hz"``). Chosen from ``step`` if omitted.
        live: When False the value only commits when the user releases the slider, so an
            expensive callback does not run on every pixel of a drag.
        unit: Suffix appended to the displayed value when ``fmt`` is not given.
    """

    kind = "slider"

    def __init__(
        self,
        label: str,
        min: float = 0.0,  # noqa: A002 - matches ipywidgets / the call users write
        max: float = 1.0,  # noqa: A002
        value: Optional[float] = None,
        step: Optional[float] = None,
        *,
        integer: bool = False,
        log: bool = False,
        fmt: Optional[str] = None,
        live: bool = True,
        unit: str = "",
        **kwargs: Any,
    ) -> None:
        lo = _finite(min, f"slider {label!r} min")
        hi = _finite(max, f"slider {label!r} max")
        if not lo < hi:
            raise ControlError(f"slider {label!r}: min ({lo:g}) must be less than max ({hi:g})")
        if log and lo <= 0:
            raise ControlError(f"slider {label!r}: a log slider needs min > 0, got {lo:g}")
        if step is not None:
            step = _finite(step, f"slider {label!r} step")
            if step <= 0:
                raise ControlError(f"slider {label!r}: step must be positive, got {step:g}")
        if integer:
            lo, hi = float(math.ceil(lo)), float(math.floor(hi))
            if not lo < hi:
                raise ControlError(f"integer slider {label!r} has no integer in its range")
            step = float(builtins.max(1, round(step or 1)))
        self.min = lo
        self.max = hi
        self.step: Optional[float] = step
        self.integer = bool(integer)
        self.log = bool(log)
        self.live = bool(live)
        self.unit = str(unit or "")
        self.fmt = fmt or self._auto_fmt()
        if value is None:
            value = math.sqrt(lo * hi) if log else (lo + hi) / 2.0
        super().__init__(label, value, **kwargs)

    def _auto_fmt(self) -> str:
        if self.integer:
            base = "%d"
        elif self.step is not None:
            base = f"%.{_decimals_for(self.step)}f"
        elif self.log:
            base = "%.4g"
        else:
            span = self.max - self.min
            base = "%.3f" if span <= 1000 else "%.1f"
        return f"{base} {self.unit}" if self.unit else base

    def _coerce(self, value: Any) -> Union[int, float]:
        x = _finite(value, f"slider {self.label!r}")
        x = builtins.min(builtins.max(x, self.min), self.max)
        if self.step:
            k = round((x - self.min) / self.step)
            x = builtins.min(self.min + k * self.step, self.max)
            x = float(f"{x:.12g}")  # strip the 0.30000000000000004 noise
        if self.integer:
            return int(round(x))
        return float(x)

    @property
    def span(self) -> float:
        return self.max - self.min

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info.update(min=self.min, max=self.max, step=self.step, log=self.log, integer=self.integer)
        return info


class RangeSlider(Control):
    """An interval ``(lo, hi)`` inside ``[min, max]`` -- a band-pass, a fit window, a crop."""

    kind = "range"

    def __init__(
        self,
        label: str,
        min: float = 0.0,  # noqa: A002
        max: float = 1.0,  # noqa: A002
        value: Optional[Sequence[float]] = None,
        step: Optional[float] = None,
        *,
        integer: bool = False,
        fmt: Optional[str] = None,
        live: bool = True,
        **kwargs: Any,
    ) -> None:
        # Reuse Slider's validation and snapping for each end rather than re-deriving it.
        self._axis = Slider(label, min, max, None, step, integer=integer, fmt=fmt)
        self.min, self.max, self.step = self._axis.min, self._axis.max, self._axis.step
        self.integer = self._axis.integer
        self.fmt = self._axis.fmt
        self.live = bool(live)
        if value is None:
            value = (self.min, self.max)
        super().__init__(label, value, **kwargs)

    def _coerce(self, value: Any) -> Tuple[Any, Any]:
        try:
            lo, hi = value
        except (TypeError, ValueError):
            raise ControlError(
                f"range {self.label!r}: expected a (low, high) pair, got {value!r}"
            ) from None
        lo, hi = self._axis._coerce(lo), self._axis._coerce(hi)
        if lo > hi:
            lo, hi = hi, lo
        return (lo, hi)

    def to_json(self) -> Any:
        return list(self._value)


class Checkbox(Control):
    """A boolean switch."""

    kind = "checkbox"

    def __init__(self, label: str, value: bool = False, **kwargs: Any) -> None:
        super().__init__(label, value, **kwargs)

    def _coerce(self, value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            return bool(value)
        # A string is rejected on purpose: bool("false") is True, which is never meant.
        raise ControlError(f"checkbox {self.label!r}: expected True/False, got {value!r}")


class Dropdown(Control):
    """One choice from a fixed list of options.

    ``options`` may be a list of values (shown via ``str``), a list of ``(label, value)``
    pairs, or a ``{label: value}`` dict. Assigning accepts either an option's value or its
    label. ``style="radio"`` draws radio buttons instead of a combo box.
    """

    kind = "dropdown"

    def __init__(
        self,
        label: str,
        options: Union[Sequence[Any], Mapping[str, Any]],
        value: Any = None,
        *,
        style: str = "combo",
        **kwargs: Any,
    ) -> None:
        self.labels, self.values = self._normalise(label, options)
        if style not in ("combo", "radio"):
            raise ControlError(f"dropdown {label!r}: style must be 'combo' or 'radio'")
        self.style = style
        if value is None:
            value = self.values[0]
        super().__init__(label, value, **kwargs)

    @staticmethod
    def _normalise(label: str, options: Any) -> Tuple[List[str], List[Any]]:
        if isinstance(options, Mapping):
            pairs = [(str(k), v) for k, v in options.items()]
        else:
            try:
                items = list(options)
            except TypeError:
                raise ControlError(f"dropdown {label!r}: options must be a list or dict") from None
            if items and all(
                isinstance(it, tuple) and len(it) == 2 and isinstance(it[0], str) for it in items
            ):
                pairs = [(it[0], it[1]) for it in items]
            else:
                pairs = [(str(it), it) for it in items]
        if not pairs:
            raise ControlError(f"dropdown {label!r} needs at least one option")
        labels = [p[0] for p in pairs]
        if len(set(labels)) != len(labels):
            raise ControlError(f"dropdown {label!r}: option labels must be unique, got {labels}")
        return labels, [p[1] for p in pairs]

    def _find(self, value: Any) -> int:
        for i, option in enumerate(self.values):
            if option is value or self._same(option, value):
                return i
        if isinstance(value, str) and value in self.labels:
            return self.labels.index(value)
        raise ControlError(
            f"dropdown {self.label!r}: {value!r} is not one of {', '.join(self.labels)}"
        )

    def _coerce(self, value: Any) -> Any:
        return self.values[self._find(value)]

    @property
    def index(self) -> int:
        """Position of the current option."""
        return self._find(self._value)

    @property
    def selected_label(self) -> str:
        """Label of the current option."""
        return self.labels[self.index]

    def to_json(self) -> Any:
        # The label, not the value: values may be functions or arrays, labels are strings.
        return self.selected_label


class TextBox(Control):
    """Free text. Commits on Enter / focus loss unless ``live=True``."""

    kind = "text"

    def __init__(
        self,
        label: str,
        value: str = "",
        *,
        placeholder: str = "",
        multiline: bool = False,
        live: bool = False,
        **kwargs: Any,
    ) -> None:
        self.placeholder = str(placeholder or "")
        self.multiline = bool(multiline)
        self.live = bool(live)
        super().__init__(label, value, **kwargs)

    def _coerce(self, value: Any) -> str:
        text = "" if value is None else str(value)
        if len(text) > MAX_TEXT_LENGTH:
            raise ControlError(
                f"text {self.label!r}: {len(text)} characters exceeds the {MAX_TEXT_LENGTH} limit"
            )
        return text


class Number(Control):
    """A number typed or dragged, optionally bounded. For values with no natural range."""

    kind = "number"

    def __init__(
        self,
        label: str,
        value: float = 0.0,
        *,
        min: Optional[float] = None,  # noqa: A002
        max: Optional[float] = None,  # noqa: A002
        step: Optional[float] = None,
        integer: bool = False,
        fmt: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.min = None if min is None else _finite(min, f"number {label!r} min")
        self.max = None if max is None else _finite(max, f"number {label!r} max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ControlError(f"number {label!r}: min must not exceed max")
        self.integer = bool(integer)
        self.step = None if step is None else _finite(step, f"number {label!r} step")
        if self.step is not None and self.step <= 0:
            raise ControlError(f"number {label!r}: step must be positive")
        self.fmt = fmt or ("%d" if integer else "%.6g")
        super().__init__(label, value, **kwargs)

    def _coerce(self, value: Any) -> Union[int, float]:
        x = _finite(value, f"number {self.label!r}")
        if self.min is not None:
            x = builtins.max(x, self.min)
        if self.max is not None:
            x = builtins.min(x, self.max)
        return int(round(x)) if self.integer else float(x)


def _to_rgba(value: Any) -> Tuple[float, float, float, float]:
    """Any reasonable colour spelling -> an RGBA tuple of floats in [0, 1]."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("#") and len(text) in (7, 9):
            try:
                parts = [int(text[i : i + 2], 16) / 255.0 for i in range(1, len(text), 2)]
            except ValueError:
                raise ControlError(f"not a colour: {value!r}") from None
            return tuple(parts + [1.0] * (4 - len(parts)))  # type: ignore[return-value]
        try:
            from matplotlib.colors import to_rgba

            return tuple(float(c) for c in to_rgba(text))  # type: ignore[return-value]
        except Exception:
            raise ControlError(f"not a colour: {value!r}") from None
    try:
        parts = [float(c) for c in value]
    except (TypeError, ValueError):
        raise ControlError(f"not a colour: {value!r}") from None
    if len(parts) not in (3, 4) or not all(math.isfinite(c) for c in parts):
        raise ControlError(f"a colour needs 3 or 4 finite components, got {value!r}")
    if builtins.max(parts) > 1.0:  # 0-255 integers
        parts = [c / 255.0 for c in parts]
    if len(parts) == 3:
        parts.append(1.0)
    return tuple(builtins.min(builtins.max(c, 0.0), 1.0) for c in parts)  # type: ignore[return-value]  # noqa: E501


class ColorPicker(Control):
    """A colour, held as an RGBA tuple of floats in [0, 1]. Accepts hex, names, tuples."""

    kind = "color"

    def __init__(self, label: str, value: Any = "#1f77b4", *, alpha: bool = False, **kwargs: Any):
        self.alpha = bool(alpha)
        super().__init__(label, value, **kwargs)

    def _coerce(self, value: Any) -> Tuple[float, float, float, float]:
        return _to_rgba(value)

    @property
    def hex(self) -> str:
        """``"#rrggbb"`` (``"#rrggbbaa"`` when the colour is translucent)."""
        comps = [int(round(c * 255)) for c in self._value]
        if comps[3] == 255:
            comps = comps[:3]
        return "#" + "".join(f"{c:02x}" for c in comps)

    def to_json(self) -> Any:
        return self.hex


class Button(Control):
    """A push button that runs its callbacks once per click.

    Args:
        label: Button text.
        callback: Called as ``fn()`` or ``fn(button)`` per click. More via :meth:`on_click`.
        shortcut: Keyboard chord (``"Mod+R"``, ``"F5"``). The button is also listed in the
            command palette (``Mod+K``) under "Controls" whether or not it has one.
        confirm: If given, clicking asks this question first and runs only on OK.
        icon: A workstation icon name drawn on the button (``"refresh"``, ``"play"``...).
    """

    kind = "button"
    is_input = False

    def __init__(
        self,
        label: str,
        callback: Optional[Callable[..., Any]] = None,
        *,
        shortcut: Optional[str] = None,
        confirm: Optional[str] = None,
        icon: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if shortcut is not None:
            _validate_chord(shortcut, label)
        self.shortcut = shortcut
        self.confirm = confirm
        self.icon = icon
        #: Number of clicks so far.
        self.clicks = 0
        super().__init__(label, 0, **kwargs)
        if callback is not None:
            self.on_click(callback)

    def _coerce(self, value: Any) -> int:
        return int(value)

    def on_click(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Run ``fn()`` (or ``fn(button)``) on every click. Usable as a decorator."""
        return self.on_change(fn)

    def click(self) -> None:
        """Press the button programmatically -- exactly what the GUI does."""
        if not self.enabled:
            return
        self.clicks += 1
        self._value = self.clicks
        self.version += 1
        owner = self._owner
        if owner is not None:
            owner._notify(self)
            return
        for cb in list(self._observers):
            cb(self)

    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator form: ``@gplt.button("Randomize")`` registers the function below it."""
        if not callable(fn):
            raise ControlError("use button.click() to press a button; calling it registers fn")
        self.on_click(fn)
        return fn

    def to_json(self) -> Any:
        return None


def _validate_chord(spec: str, label: str) -> None:
    try:
        from .gui.keys import parse
    except Exception:  # pragma: no cover - the keys module is pure python
        return
    try:
        parse(spec)
    except ValueError as exc:
        raise ControlError(f"button {label!r}: invalid shortcut {spec!r} ({exc})") from None


class Readout(Control):
    """A read-only value shown in the panel -- a statistic, a fit result, a status line.

    Assign to :attr:`value` from any callback; it never triggers bindings.
    """

    kind = "readout"
    is_input = False

    def __init__(self, label: str, value: Any = "", *, fmt: Optional[str] = None, **kwargs: Any):
        self.fmt = fmt
        super().__init__(label, value, **kwargs)

    def _coerce(self, value: Any) -> Any:
        return value

    @property
    def text(self) -> str:
        """The value as displayed."""
        return format_value(self._value, self.fmt)

    def to_json(self) -> Any:
        return self.text


def format_value(value: Any, fmt: Optional[str] = None) -> str:
    """Display text for a readout or a binding output."""
    if fmt:
        try:
            return fmt % value if "%" in fmt else format(value, fmt)
        except Exception:
            pass
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if not math.isfinite(x):
            return str(x)
        if x != 0 and (abs(x) >= 1e6 or abs(x) < 1e-3):
            return f"{x:.4e}"
        return f"{x:.6g}"
    if isinstance(value, np.ndarray):
        if value.size <= 6:
            return np.array2string(value, precision=4, separator=", ")
        return f"array{value.shape} {value.dtype}"
    return str(value)


# ======================================================================================
# interact() abbreviations
# ======================================================================================


def control_from_spec(name: str, spec: Any) -> Control:
    """Turn an ``interact`` keyword into a control, ipywidgets-style.

    ============================  ==================================================
    ``(0, 10)``                   float :class:`Slider`
    ``(0, 10, 0.5)``              float slider with a step
    ``(1, 100, 1)``               integer slider (every element an ``int``)
    ``True`` / ``False``          :class:`Checkbox`
    ``["sin", "cos"]`` / dict     :class:`Dropdown`
    ``"text"``                    :class:`TextBox`
    ``3.5`` / ``7``               :class:`Number`
    a :class:`Control`            used as-is
    ============================  ==================================================
    """
    label = _pretty(name)
    if isinstance(spec, Control):
        return spec
    if isinstance(spec, (bool, np.bool_)):
        return Checkbox(label, bool(spec), key=name)
    if isinstance(spec, str):
        return TextBox(label, spec, key=name)
    if isinstance(spec, (int, float, np.integer, np.floating)):
        integer = isinstance(spec, (int, np.integer))
        return Number(label, spec, integer=integer, key=name)
    if isinstance(spec, Mapping):
        return Dropdown(label, spec, key=name)
    if (
        isinstance(spec, tuple)
        and len(spec) in (2, 3)
        and all(
            isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
            for v in spec
        )
    ):
        integer = len(spec) == 3 and all(isinstance(v, (int, np.integer)) for v in spec)
        step = spec[2] if len(spec) == 3 else None
        return Slider(label, spec[0], spec[1], step=step, integer=integer, key=name)
    if isinstance(spec, (list, tuple)):
        return Dropdown(label, list(spec), key=name)
    raise ControlError(
        f"cannot make a control for {name}={spec!r}: use (min, max[, step]), a bool, a list "
        "of options, a string, a number, or a Control"
    )


# ======================================================================================
# Bindings
# ======================================================================================


def _as_xy(result: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Recognise curve data in a binding's return value, or None if it is not curve data."""
    if isinstance(result, tuple) and len(result) == 2:
        try:
            x = np.asarray(result[0], dtype=np.float64)
            y = np.asarray(result[1], dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if x.ndim == 1 and y.ndim == 1 and len(x) == len(y) and len(x) > 0:
            return x, y
        return None
    if isinstance(result, np.ndarray) or (
        isinstance(result, list) and result and all(isinstance(v, (int, float)) for v in result)
    ):
        try:
            arr = np.asarray(result, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if arr.ndim == 1 and len(arr) > 0:
            return np.arange(len(arr), dtype=np.float64), arr
        if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
            return arr[:, 0], arr[:, 1]
    return None


class Binding:
    """A function wired to controls; re-run whenever one of them changes.

    Created by :meth:`ControlSet.bind` (what ``gplt.interact`` calls). Calling the binding
    runs it now with the current values and returns the function's result.
    """

    def __init__(
        self,
        owner: "ControlSet",
        fn: Callable[..., Any],
        params: Mapping[str, Control],
        *,
        label: Optional[str] = None,
        autoscale: bool = False,
        kind: str = "line",
        plot_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if kind not in ("line", "scatter"):
            raise ControlError(f"interact kind must be 'line' or 'scatter', got {kind!r}")
        self._owner = owner
        self.fn = fn
        #: Parameter name -> control. The function is called as ``fn(**{name: value})``.
        self.params: Dict[str, Control] = dict(params)
        self.label = label or _pretty(getattr(fn, "__name__", "interact"))
        self.autoscale = bool(autoscale)
        self.kind = kind
        self.plot_kwargs = dict(plot_kwargs or {})
        #: Paused bindings do not run on changes (calling one still runs it).
        self.enabled = True
        self.result: Any = None
        #: ``{name: value}`` from the last run that returned a mapping; shown in the panel.
        self.outputs: Dict[str, Any] = {}
        #: The curve this binding draws and owns, once it has returned ``(x, y)`` data.
        self.layer: Any = None
        self.error: Optional[str] = None
        self.traceback: Optional[str] = None
        self.runs = 0
        self.last_ms = 0.0

    @property
    def controls(self) -> Tuple[Control, ...]:
        return tuple(self.params.values())

    def kwargs(self) -> Dict[str, Any]:
        return {name: control.value for name, control in self.params.items()}

    def depends_on(self, changed_ids: Any) -> bool:
        return any(id(control) in changed_ids for control in self.params.values())

    def __call__(self) -> Any:
        return self.run()

    def run(self) -> Any:
        """Run now with the current values (subject to the set's error policy)."""
        self._owner._run_binding(self)
        return self.result

    def remove(self) -> None:
        """Unbind. The curve it drew stays on the figure."""
        self._owner.unbind(self)

    # -- executed by ControlSet ------------------------------------------------------

    def _execute(self) -> None:
        t0 = time.perf_counter()
        try:
            result = self.fn(**self.kwargs())
        except BaseException:
            # No stale result: a binding reading this one's .result must see "nothing
            # current", not the output of values the user has already moved away from.
            # (The curve it drew stays -- the last good picture, with the error beside it.)
            self.result = None
            raise
        finally:
            self.runs += 1
            self.last_ms = (time.perf_counter() - t0) * 1000.0
        self.result = result
        self._consume(result)
        if self.autoscale:
            plot = self._owner.plot
            if plot is not None and hasattr(plot, "autoscale"):
                plot.autoscale()

    def _consume(self, result: Any) -> None:
        if result is None:
            return
        if isinstance(result, Mapping):
            self.outputs = {str(k): v for k, v in result.items()}
            return
        data = _as_xy(result)
        if data is not None:
            self._draw(*data)

    def _draw(self, x: np.ndarray, y: np.ndarray) -> None:
        plot = self._owner.plot
        if plot is None:
            return
        layer = self.layer
        alive = layer is not None and any(
            existing is layer for existing in getattr(plot.scene, "layers", ())
        )
        if alive:
            if hasattr(layer, "set_data"):
                layer.set_data(x, y)
                return
        self.layer = self._create_layer(plot, x, y)

    def _create_layer(self, plot: Any, x: np.ndarray, y: np.ndarray) -> Any:
        from . import pyplot as _pp

        previous = _pp._CURRENT_PLOT
        _pp._CURRENT_PLOT = plot
        try:
            kwargs = {"label": self.label}
            kwargs.update(self.plot_kwargs)
            if self.kind == "scatter":
                layer = _pp.scatter(x, y, **kwargs)
            else:
                layer = _pp.plot(x, y, **kwargs)[0]
        finally:
            _pp._CURRENT_PLOT = previous
        layer.metadata["controls_binding"] = self.label
        return layer

    def __repr__(self) -> str:
        names = ", ".join(self.params)
        return f"<Binding {self.label!r} ({names})>"


# ======================================================================================
# The per-figure set
# ======================================================================================


class ControlSet:
    """Every control and binding of one figure, plus the change pipeline.

    Reached as ``fig.controls`` (``gplt.controls()`` for the current figure). Supports
    ``controls["key"]``, ``"key" in controls``, ``len`` and iteration in display order.
    """

    def __init__(self, plot: Any = None) -> None:
        self._plot_ref = weakref.ref(plot) if plot is not None else None
        self._controls: "OrderedDict[str, Control]" = OrderedDict()
        self._bindings: List[Binding] = []
        self._pending: "OrderedDict[int, Control]" = OrderedDict()
        self._clicks: Deque[Button] = deque()
        self._lock = threading.RLock()
        self._live = False
        self._waker: Optional[Callable[[], None]] = None
        self._batch_depth = 0
        self._flushing = False
        self._group_stack: List[str] = []
        self._logged_errors: Dict[int, str] = {}
        #: Bumped when controls or bindings are added or removed (not on value changes).
        self.structure_version = 0
        #: Named snapshots, see :meth:`save_preset`.
        self.presets: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        #: Wall time of the last flush that ran anything, in milliseconds.
        self.last_flush_ms = 0.0

    # -- plumbing --------------------------------------------------------------------

    @property
    def plot(self) -> Any:
        """The owning figure, or None for a free-standing set."""
        return self._plot_ref() if self._plot_ref is not None else None

    @property
    def live(self) -> bool:
        """True while the window's main loop is running (changes are then coalesced)."""
        return self._live

    def _set_live(self, live: bool, waker: Optional[Callable[[], None]] = None) -> None:
        """Engine hook: switch between immediate and frame-coalesced dispatch."""
        with self._lock:
            self._live = bool(live)
            self._waker = waker if live else None
        if not live and self.pending:
            # Changes made in the last frame before the window closed still get their
            # callbacks, now in script mode.
            try:
                self.flush()
            except Exception:  # pragma: no cover - reported, not raised, at teardown
                logger.exception("A control callback raised while the window closed.")

    @property
    def pending(self) -> bool:
        """Whether changes are waiting for :meth:`flush`."""
        return bool(self._pending or self._clicks)

    # -- membership ------------------------------------------------------------------

    def add(self, control: Control) -> Control:
        """Add ``control`` and return it. A clashing key gets a ``_2``, ``_3``... suffix."""
        if not isinstance(control, Control):
            raise ControlError(f"expected a Control, got {control!r}")
        if control._owner is self:
            return control
        if control._owner is not None:
            raise ControlError(f"{control!r} already belongs to another figure")
        with self._lock:
            base, n = control.key, 2
            while control.key in self._controls:
                control.key = f"{base}_{n}"
                n += 1
            if control.group is None and self._group_stack:
                control.group = self._group_stack[-1]
            control._owner = self
            self._controls[control.key] = control
            self.structure_version += 1
        self._wake()
        return control

    def remove(self, control: Union[str, Control]) -> None:
        """Remove a control, and every binding that reads it."""
        target = self[control] if isinstance(control, str) else control
        with self._lock:
            if self._controls.get(target.key) is not target:
                raise KeyError(target.key)
            del self._controls[target.key]
            target._owner = None
            self._pending.pop(id(target), None)
            self._bindings = [b for b in self._bindings if target not in b.controls]
            self.structure_version += 1
        self._wake()

    def clear(self) -> None:
        """Remove every control, binding and pending change."""
        with self._lock:
            for control in self._controls.values():
                control._owner = None
            self._controls.clear()
            self._bindings.clear()
            self._pending.clear()
            self._clicks.clear()
            self.structure_version += 1
        self._wake()

    def __getitem__(self, key: str) -> Control:
        try:
            return self._controls[key]
        except KeyError:
            known = ", ".join(self._controls) or "none"
            raise KeyError(f"no control {key!r} (have: {known})") from None

    def get(self, key: str, default: Any = None) -> Any:
        return self._controls.get(key, default)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, Control):
            return self._controls.get(key.key) is key
        return key in self._controls

    def __iter__(self) -> Iterator[Control]:
        return iter(list(self._controls.values()))

    def __len__(self) -> int:
        return len(self._controls)

    def __bool__(self) -> bool:
        # A set with no controls is still a real object; "if fig.controls:" asking "are
        # there any" is the reading that matches len(), so be explicit about it.
        return bool(self._controls)

    def keys(self) -> List[str]:
        return list(self._controls)

    @property
    def bindings(self) -> List[Binding]:
        return list(self._bindings)

    def groups(self) -> List[Tuple[str, List[Control]]]:
        """Controls grouped for display, groups in first-appearance order."""
        out: "OrderedDict[str, List[Control]]" = OrderedDict()
        for control in self._controls.values():
            out.setdefault(control.group or DEFAULT_GROUP, []).append(control)
        return list(out.items())

    @contextmanager
    def group(self, name: str) -> Iterator[None]:
        """Controls created inside ``with controls.group("Filter"):`` land in that group."""
        self._group_stack.append(str(name))
        try:
            yield
        finally:
            self._group_stack.pop()

    # -- values ----------------------------------------------------------------------

    def values(self) -> Dict[str, Any]:
        """``{key: value}`` of every input control."""
        return {k: c.value for k, c in self._controls.items() if c.is_input}

    def set_values(self, mapping: Optional[Mapping[str, Any]] = None, **values: Any) -> None:
        """Assign several controls at once; callbacks run once, after all of them."""
        merged = dict(mapping or {})
        merged.update(values)
        with self.batch():
            for key, value in merged.items():
                self[key].set(value)

    def reset(self, *keys: str) -> None:
        """Reset the named controls (all input controls when none are named)."""
        targets = [self[k] for k in keys] if keys else [c for c in self if c.is_input]
        with self.batch():
            for control in targets:
                control.reset()

    def snapshot(self) -> Dict[str, Any]:
        """JSON-ready ``{key: value}`` of every input control."""
        return {k: c.to_json() for k, c in self._controls.items() if c.is_input}

    def restore(self, snapshot: Mapping[str, Any], *, strict: bool = False) -> List[str]:
        """Apply a :meth:`snapshot`. Unknown keys are skipped (returned) unless ``strict``."""
        skipped: List[str] = []
        with self.batch():
            for key, value in snapshot.items():
                control = self._controls.get(key)
                if control is None or not control.is_input:
                    if strict:
                        raise KeyError(f"snapshot names an unknown control {key!r}")
                    skipped.append(key)
                    continue
                control.set(value)
        return skipped

    def save_preset(self, name: str) -> Dict[str, Any]:
        """Remember the current values under ``name`` (replacing an existing preset)."""
        name = str(name).strip()
        if not name:
            raise ControlError("a preset needs a name")
        snap = self.snapshot()
        with self._lock:
            self.presets[name] = snap
            self.structure_version += 1
        return snap

    def load_preset(self, name: str) -> None:
        """Restore the values saved by :meth:`save_preset`."""
        try:
            snap = self.presets[name]
        except KeyError:
            raise KeyError(
                f"no preset {name!r} (have: {', '.join(self.presets) or 'none'})"
            ) from None
        self.restore(snap)

    def delete_preset(self, name: str) -> None:
        with self._lock:
            del self.presets[name]
            self.structure_version += 1

    def save(self, path: str) -> None:
        """Write the current values and every preset to a JSON file."""
        payload = {"values": self.snapshot(), "presets": dict(self.presets)}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)

    def load(self, path: str) -> List[str]:
        """Read a file written by :meth:`save`. Returns the keys it could not apply."""
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise ControlError(f"{path}: not a controls file")
        with self._lock:
            for name, snap in dict(payload.get("presets") or {}).items():
                if isinstance(snap, dict):
                    self.presets[str(name)] = snap
            self.structure_version += 1
        return self.restore(payload.get("values") or {})

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Defer callbacks until the block ends, then run each affected one once."""
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
        if self._batch_depth == 0 and not self._live and not self._flushing and self.pending:
            self.flush()

    # -- bindings --------------------------------------------------------------------

    def bind(
        self,
        fn: Callable[..., Any],
        params: Mapping[str, Control],
        *,
        label: Optional[str] = None,
        autoscale: bool = False,
        kind: str = "line",
        plot_kwargs: Optional[Dict[str, Any]] = None,
        run_now: bool = True,
    ) -> Binding:
        """Wire ``fn`` to ``params`` (name -> control); returns the :class:`Binding`.

        Controls in ``params`` that are not in the set yet are added. With ``run_now``
        the function runs once immediately, so the figure starts in a consistent state.
        """
        for control in params.values():
            if not isinstance(control, Control):
                raise ControlError(f"interact parameters must be controls, got {control!r}")
            if not control.is_input:
                raise ControlError(f"{control.kind} {control.label!r} cannot feed a function")
            self.add(control)
        binding = Binding(
            self, fn, params, label=label, autoscale=autoscale, kind=kind, plot_kwargs=plot_kwargs
        )
        with self._lock:
            self._bindings.append(binding)
            self.structure_version += 1
        if run_now:
            self._run_binding(binding)
        self._wake()
        return binding

    def unbind(self, binding: Binding) -> None:
        with self._lock:
            self._bindings = [b for b in self._bindings if b is not binding]
            self.structure_version += 1

    def interact(
        self,
        fn: Callable[..., Any],
        *,
        autoscale: bool = False,
        kind: str = "line",
        label: Optional[str] = None,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        run_now: bool = True,
        **specs: Any,
    ) -> Binding:
        """Build controls for ``fn``'s parameters and bind them. See ``gplt.interact``.

        Each parameter of ``fn`` is resolved, in order, from: a keyword in ``specs`` (an
        abbreviation or a :class:`Control`), an existing control whose key is the
        parameter name, or else the parameter's own default (left out of the call). A
        required parameter matching none of these is an error, raised now rather than on
        the first slider move.
        """
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            raise ControlError(f"interact needs a Python function, got {fn!r}") from None
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())
        unknown = [
            name for name in specs if name not in signature.parameters and not accepts_kwargs
        ]
        if unknown:
            raise ControlError(
                f"interact: {getattr(fn, '__name__', 'fn')}() has no parameter "
                f"{', '.join(repr(u) for u in unknown)}"
            )
        params: "OrderedDict[str, Control]" = OrderedDict()
        missing: List[str] = []
        for name, param in signature.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if name in specs:
                control = control_from_spec(name, specs[name])
                spec = specs[name]
                carries_value = isinstance(spec, (Control, bool, str, int, float, np.generic))
                if not carries_value and param.default is not param.empty:
                    # The function's own default is the natural starting point:
                    # ``def f(amp=1.0)`` with ``amp=(0, 5)`` starts at 1.0, not 2.5.
                    try:
                        control._value = control.default = control._coerce(param.default)
                    except ControlError:
                        pass
                params[name] = control
            elif name in self._controls and self._controls[name].is_input:
                params[name] = self._controls[name]
            elif param.default is param.empty:
                missing.append(name)
        for name, spec in specs.items():  # extras routed into **kwargs
            if name not in params:
                params[name] = control_from_spec(name, spec)
        # Controls this call creates are grouped under the function's name, so each
        # interact() reads as one titled block -- unless a `with group(...)` is active.
        heading = label or _pretty(getattr(fn, "__name__", "interact"))
        for name, control in params.items():
            created = name in specs and not isinstance(specs[name], Control)
            if created and control.group is None and not self._group_stack:
                control.group = heading
        if missing:
            raise ControlError(
                f"interact: no control for parameter(s) {', '.join(missing)} of "
                f"{getattr(fn, '__name__', 'fn')}(); pass e.g. {missing[0]}=(0, 1)"
            )
        return self.bind(
            fn,
            params,
            label=label,
            autoscale=autoscale,
            kind=kind,
            plot_kwargs=plot_kwargs,
            run_now=run_now,
        )

    # -- the change pipeline ---------------------------------------------------------

    def _notify(self, control: Control) -> None:
        with self._lock:
            if isinstance(control, Button):
                self._clicks.append(control)
            elif control.is_input:
                self._pending[id(control)] = control
                self._pending.move_to_end(id(control))
            else:
                # A readout changed: nothing to run, but the panel should redraw.
                self._wake()
                return
        if self._live:
            self._wake()
            return
        if self._batch_depth or self._flushing:
            return  # picked up by the enclosing batch / the running flush's next round
        self.flush()

    def flush(self) -> int:
        """Run every callback owed to the changes since the last flush.

        Order: each changed control's own observers (in change order) and each button
        click, repeated until they stop changing controls; then every binding that reads
        any control changed so far -- once, with the settled values. Changes made by the
        bindings start another pass. At most :data:`MAX_CASCADE_ROUNDS` rounds in all, so
        two callbacks feeding each other cannot hang the window. Returns how many
        callbacks ran.

        The engine calls this at the top of each frame; outside the window a change calls
        it directly. Re-entrant calls are no-ops.
        """
        if self._flushing:
            return 0
        self._flushing = True
        ran = 0
        touched = False
        t0 = time.perf_counter()
        try:
            # Observers and clicks first, followed to a fixed point, so a button that moves
            # three sliders reruns a binding reading all three once -- not once per slider.
            # Then the bindings, once each, with the settled values. A binding that itself
            # moves a control starts another pass.
            owed: "OrderedDict[int, Control]" = OrderedDict()
            for _round in range(MAX_CASCADE_ROUNDS):
                with self._lock:
                    changed = list(self._pending.values())
                    self._pending.clear()
                    clicks = list(self._clicks)
                    self._clicks.clear()
                if changed or clicks:
                    touched = True
                    for control in changed:
                        owed[id(control)] = control
                        for cb in list(control._observers):
                            ran += self._invoke(control, cb, control.value)
                    for button in clicks:
                        for cb in list(button._observers):
                            ran += self._invoke(button, cb, button)
                    continue
                if not owed:
                    break
                changed_ids = set(owed)
                owed.clear()
                with self._lock:
                    bindings = list(self._bindings)
                for binding in bindings:
                    if binding.enabled and binding.depends_on(changed_ids):
                        ran += self._run_binding(binding, _in_flush=True)
            else:
                with self._lock:
                    looping = ", ".join(c.label for c in self._pending.values())
                    self._pending.clear()
                    self._clicks.clear()
                logger.warning(
                    "Controls: callbacks kept changing each other (%s) for %d rounds; "
                    "stopped to keep the window responsive.",
                    looping,
                    MAX_CASCADE_ROUNDS,
                )
        finally:
            self._flushing = False
        if ran:
            self.last_flush_ms = (time.perf_counter() - t0) * 1000.0
        if touched:
            # Even with no callback registered: a gplt.function() closure may read the
            # control directly, and it only re-evaluates when told to.
            self._after_callbacks()
        return ran

    def _invoke(self, source: Control, cb: _Callback, arg: Any) -> int:
        if not self._live:
            cb(arg)
            source.error = None
            return 1
        try:
            cb(arg)
        except Exception as exc:  # shown in the panel; the render loop must survive
            self._record_error(source, exc)
        else:
            source.error = None
            self._logged_errors.pop(id(source), None)
        return 1

    def _run_binding(self, binding: Binding, *, _in_flush: bool = False) -> int:
        if not self._live:
            binding._execute()
            binding.error = binding.traceback = None
        else:
            try:
                binding._execute()
            except Exception as exc:
                binding.traceback = traceback.format_exc()
                self._record_error(binding, exc)
            else:
                binding.error = binding.traceback = None
                self._logged_errors.pop(id(binding), None)
        if not _in_flush:
            self._after_callbacks()
        return 1

    def _record_error(self, source: Any, exc: BaseException) -> None:
        message = f"{type(exc).__name__}: {exc}"
        source.error = message
        # Log and toast once per distinct failure: a callback that raises on every frame
        # of a drag must not write sixty identical tracebacks a second.
        if self._logged_errors.get(id(source)) != message:
            self._logged_errors[id(source)] = message
            name = getattr(source, "label", "callback")
            logger.error("Control callback %r failed: %s", name, message, exc_info=exc)
            try:
                from .gui import notifications

                notifications.push(f"{name}: {message}", kind="error")
            except Exception:  # pragma: no cover - notifications are best effort
                pass

    def errors(self) -> List[Tuple[str, str]]:
        """``(label, message)`` for every control or binding whose last run failed."""
        out = [(c.label, c.error) for c in self._controls.values() if c.error]
        out += [(b.label, b.error) for b in self._bindings if b.error]
        return out  # type: ignore[return-value]

    def _after_callbacks(self) -> None:
        plot = self.plot
        if plot is None:
            return
        repoint_legacy_mirrors(plot)
        for layer in list(getattr(getattr(plot, "scene", None), "layers", ()) or ()):
            invalidate = getattr(layer, "invalidate", None)
            if callable(invalidate):
                invalidate()
        frame = getattr(plot, "frame", None)
        if frame is not None:
            frame.dirty_scene = True
            frame.dirty_ui = True
        cache = getattr(plot, "cache", None)
        if cache is not None:
            cache.capture_window = None
            cache.refresh_requested = True

    def _wake(self) -> None:
        waker = self._waker
        if waker is not None:
            try:
                waker()
            except Exception:  # pragma: no cover - waking is best effort
                pass

    def __repr__(self) -> str:
        return f"<ControlSet {len(self._controls)} controls, {len(self._bindings)} bindings>"


def repoint_legacy_mirrors(plot: Any) -> None:
    """Point ``scene.scatters``/``scene.strips`` at arrays a layer setter replaced.

    ``engine.add_scatter``/``add_line_strip`` keep a legacy mirror that shares the layer's
    ``pts`` array by identity, and ``layerops.remove_layer`` purges mirrors by that same
    identity. A setter that swaps in a new array records the one it replaced; this
    re-points the mirror so it neither pins the old copy nor survives the layer's removal.
    """
    scene = getattr(plot, "scene", None)
    if scene is None:
        return
    for layer in getattr(scene, "layers", ()):
        old = getattr(layer, "_superseded_pts", None)
        if old is None:
            continue
        mirrors = getattr(scene, "scatters" if layer.layer_type == "scatter" else "strips", None)
        for mirror in mirrors or ():
            if getattr(mirror, "pts", None) is old:
                mirror.pts = layer.pts
                break
        layer._superseded_pts = None
