"""Time and parameters for MCP figures: controls, bindings and multi-stage timelines.

A figure the model builds is *static*. Three small ideas make it move or respond, without any
tool ever running model-written code:

* a **control** is a named value -- a slider, a toggle, a choice or a button -- that the figure's
  properties can depend on;
* a **binding** says "this property of that artist (or axes, or dataset column) is this
  expression" -- ``linewidth = 1 + 3*t``, ``azim = 30 + 360*t``, ``y = sin(k*x - phase)``;
* a **timeline** turns a frame count and a list of *stages* into one array per variable, so a
  stage can draw a curve in, the next can sweep a parameter and a third can orbit the camera.

Everything here is a pure function of plain data: no GL, no matplotlib figure, no MCP SDK. The
toolbox replays a figure once per frame with the evaluated values (``Toolbox._build``); this
module only decides *what the values are*. That split is why a whole animation can be checked --
every variable at every frame -- before a single image is rendered.

Expressions use the workstation's safe evaluator (:func:`glplot.mcp.state.evaluate_expression`),
plus the easing helpers in :data:`HELPERS`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..gui.expressions import SAFE_NAMES, _unit
from .state import (
    DEFAULT_LIMITS,
    AnimationLimits,
    Limits,
    McpError,
    evaluate_expression,
    is_identifier,
    suggest,
)

__all__ = [
    "Binding",
    "CONTROL_KINDS",
    "Control",
    "EASINGS",
    "HELPERS",
    "RESERVED_NAMES",
    "Timeline",
    "build_timeline",
    "control_variable",
    "coerce_property",
    "evaluate_value",
    "reveal_text",
    "make_control",
    "parse_target",
    "set_path",
    "set_control_value",
    "timeline_guide",
]


# ----------------------------------------------------------------------------------
# Easing and helper functions available in every expression
# ----------------------------------------------------------------------------------


#: The easing and interpolation functions of the shared expression vocabulary, by name.
HELPERS: Dict[str, Callable[..., Any]] = {
    name: SAFE_NAMES[name]
    for name in (
        "lerp",
        "smoothstep",
        "smootherstep",
        "ease_in",
        "ease_out",
        "ease_in_out",
        "ease_sine",
        "there_and_back",
        "rush_into",
        "rush_from",
        "pingpong",
        "ramp",
        "pulse",
    )
}

#: Easing functions a stage or a ``tween`` can name.
EASINGS: Dict[str, Callable[[Any], np.ndarray]] = {
    "linear": _unit,
    "in": HELPERS["ease_in"],
    "out": HELPERS["ease_out"],
    "in_out": HELPERS["ease_in_out"],
    "smooth": HELPERS["smoothstep"],
    "smoother": HELPERS["smootherstep"],
    "sine": HELPERS["ease_sine"],
    # manim's own: its ``smooth`` is ``smoother`` above; these are the other three it uses most
    "rush_into": HELPERS["rush_into"],
    "rush_from": HELPERS["rush_from"],
    # goes to the target and comes back, so the stage ends where it began (a pulse)
    "there_and_back": HELPERS["there_and_back"],
}


def ease_end(name: Optional[str]) -> float:
    """Where an easing finishes: 1 for all of them except ``there_and_back`` (0)."""
    return float(EASINGS[name or "linear"](np.array(1.0)))


#: Names a variable may not take: the clock, the stage position and the helpers.
RESERVED_NAMES = frozenset({"t", "frame", "n", "time", "s", "stage", "x", "y", "z"} | set(HELPERS))


# ----------------------------------------------------------------------------------
# Limits specific to animation
# ----------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
# Controls
# ----------------------------------------------------------------------------------

CONTROL_KINDS = ("slider", "toggle", "choice", "button")


@dataclass
class Control:
    """A named value a figure can depend on. ``kind`` decides how the value behaves."""

    name: str
    kind: str
    label: str
    value: Any
    default: Any
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    integer: bool = False
    log: bool = False
    options: Optional[List[str]] = None
    #: A button's effect: control name -> the value it sets (``None``: reset every control).
    sets: Optional[Dict[str, Any]] = None

    def describe(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name, "kind": self.kind, "label": self.label}
        if self.kind == "slider":
            out.update(value=self.value, min=self.minimum, max=self.maximum)
            if self.step is not None:
                out["step"] = self.step
            if self.integer:
                out["integer"] = True
            if self.log:
                out["log"] = True
        elif self.kind == "toggle":
            out["value"] = bool(self.value)
        elif self.kind == "choice":
            out.update(value=self.value, options=list(self.options or []))
            out["index"] = (self.options or []).index(self.value)
        else:
            out["sets"] = dict(self.sets) if self.sets is not None else "all controls to default"
        return out


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise McpError(f"{what} must be a finite number, got {value!r}.")
    return float(value)


def make_control(
    name: str,
    kind: str = "slider",
    *,
    label: Optional[str] = None,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    value: Any = None,
    step: Optional[float] = None,
    integer: bool = False,
    log: bool = False,
    options: Optional[Sequence[str]] = None,
    sets: Optional[Mapping[str, Any]] = None,
) -> Control:
    """A validated :class:`Control`; every refusal names the argument and what would work."""
    if not is_identifier(name) or name in RESERVED_NAMES:
        raise McpError(
            f"Control name {name!r} must be an identifier and not one of the reserved names "
            f"({', '.join(sorted(RESERVED_NAMES))}): it is a variable in expressions."
        )
    if kind not in CONTROL_KINDS:
        raise McpError(
            f"Unknown control kind {kind!r}.{suggest(kind, CONTROL_KINDS)} "
            f"Use one of: {', '.join(CONTROL_KINDS)}."
        )
    text = label if label else name
    if kind == "slider":
        if minimum is None or maximum is None:
            raise McpError("A slider needs min and max (for example min=0, max=10).")
        low, high = _number(minimum, "min"), _number(maximum, "max")
        if low >= high:
            raise McpError(f"min ({low:g}) must be below max ({high:g}).")
        if log and low <= 0:
            raise McpError("log=true needs min > 0 (a log slider spans decades).")
        if step is not None and _number(step, "step") <= 0:
            raise McpError("step must be positive.")
        start = (math.sqrt(low * high) if log else (low + high) / 2.0) if value is None else value
        control = Control(
            name, kind, text, 0.0, 0.0, low, high, None if step is None else float(step),
            bool(integer), bool(log),
        )  # fmt: skip
        control.value = control.default = set_control_value(control, start)
        return control
    if kind == "toggle":
        if value is not None and not isinstance(value, (bool, int)):
            raise McpError(f"A toggle's value is true or false, got {value!r}.")
        flag = bool(value) if value is not None else False
        return Control(name, kind, text, flag, flag)
    if kind == "choice":
        names = [str(o) for o in (options or [])]
        if len(names) < 2 or len(set(names)) != len(names):
            raise McpError(
                "A choice needs options: two or more distinct names, e.g. ['sin', 'cos']."
            )
        chosen = names[0] if value is None else str(value)
        if chosen not in names:
            raise McpError(f"value {chosen!r} is not one of the options: {', '.join(names)}.")
        return Control(name, kind, text, chosen, chosen, options=names)
    if sets is not None:
        if not isinstance(sets, Mapping) or not sets:
            raise McpError("A button's sets is an object like {'a': 1, 'b': 0.5}.")
        sets = {str(k): v for k, v in sets.items()}
    return Control(name, kind, text, None, None, sets=dict(sets) if sets else None)


def set_control_value(control: Control, raw: Any) -> Any:
    """``raw`` coerced to a valid value of ``control`` (a slider snaps; out of range is refused)."""
    if control.kind == "slider":
        number = _number(raw, f"The value of slider {control.name!r}")
        low, high = float(control.minimum), float(control.maximum)
        if not low - 1e-12 <= number <= high + 1e-12:
            raise McpError(
                f"{number:g} is outside slider {control.name!r}'s range {low:g}..{high:g}."
            )
        number = min(max(number, low), high)
        if control.step:
            number = low + round((number - low) / control.step) * control.step
            number = min(max(number, low), high)
        if control.integer:
            number = float(round(number))
        return float(f"{number:.12g}")
    if control.kind == "toggle":
        if isinstance(raw, bool) or (isinstance(raw, int) and raw in (0, 1)):
            return bool(raw)
        raise McpError(f"Toggle {control.name!r} takes true or false, got {raw!r}.")
    if control.kind == "choice":
        options = control.options or []
        if isinstance(raw, int) and not isinstance(raw, bool) and 0 <= raw < len(options):
            return options[raw]
        if str(raw) not in options:
            raise McpError(
                f"{raw!r} is not an option of {control.name!r}.{suggest(str(raw), options)} "
                f"Options: {', '.join(options)}."
            )
        return str(raw)
    raise McpError(f"{control.name!r} is a button; it has no value. Press it with set_control.")


def control_variable(control: Control) -> Optional[float]:
    """The number an expression sees for ``control`` (a choice is its index; a button has none)."""
    if control.kind == "slider":
        return float(control.value)
    if control.kind == "toggle":
        return 1.0 if control.value else 0.0
    if control.kind == "choice":
        return float((control.options or []).index(control.value))
    return None


# ----------------------------------------------------------------------------------
# Bindings
# ----------------------------------------------------------------------------------

_ARTIST = re.compile(r"^a\d+$")
_AXES = re.compile(r"^axes(?::(\d+))?$")


def parse_target(target: str) -> Tuple[str, Any]:
    """``('artist', 'a3')``, ``('axes', panel)`` or ``('data', 'name')`` for a binding target."""
    if not isinstance(target, str):
        raise McpError(f"target must be a string, got {type(target).__name__}.")
    if _ARTIST.match(target):
        return "artist", target
    match = _AXES.match(target)
    if match:
        return "axes", int(match.group(1) or 1)
    if target.startswith("data:") and len(target) > 5:
        return "data", target[5:]
    raise McpError(
        f"Unknown binding target {target!r}. Use an artist id ('a1'), 'axes' or 'axes:2' "
        "(a panel's "
        "title, labels, limits, camera) or 'data:<dataset>' (a column recomputed each time)."
    )


@dataclass
class Binding:
    """One property of one target, defined by a value specification (see :func:`evaluate_value`)."""

    target: str
    prop: str
    value: Any

    def describe(self) -> Dict[str, Any]:
        return {"target": self.target, "property": self.prop, "value": self.value}


#: Properties that are whole numbers or true/false, though an expression gives a float.
INTEGER_PROPERTIES = frozenset({"trail", "bins", "samples", "resolution", "levels", "panel"})
BOOLEAN_PROPERTIES = frozenset(
    {"head", "colorbar", "grid", "legend", "equal_aspect", "hide_axes", "sample", "integer"}
)


def reveal_text(text: str, fraction: float) -> str:
    """The first ``fraction`` (0..1) of ``text``, as written so far.

    A ``$...$`` formula is one piece -- half a formula cannot be typeset -- and every other
    character is one, so a title is written letter by letter and its maths appears whole.
    """
    import re

    pieces = re.findall(r"\$[^$]*\$|.", str(text), flags=re.S)
    share = min(max(float(fraction), 0.0), 1.0)
    count = int(np.ceil(share * len(pieces) - 1e-9))
    return "".join(pieces[:count])


def coerce_property(prop: str, value: Any) -> Any:
    """``value`` as the type the property takes (an expression's 150.0 is the integer 150)."""
    leaf = prop.rsplit(".", 1)[-1]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if leaf in INTEGER_PROPERTIES:
        return int(round(value))
    if leaf in BOOLEAN_PROPERTIES:
        return bool(value)
    return value


def set_path(args: Dict[str, Any], path: str, value: Any) -> None:
    """Set ``args['a']['b']`` for ``path='a.b'``, creating the dictionaries on the way."""
    keys = path.split(".")
    if not all(is_identifier(k) for k in keys):
        raise McpError(
            f"{path!r} is not a property name (use letters, digits and '_', '.' to nest)."
        )
    cursor = args
    for key in keys[:-1]:
        nested = cursor.get(key)
        if nested is None:
            nested = cursor[key] = {}
        if not isinstance(nested, dict):
            raise McpError(f"Cannot set {path!r}: {key!r} is not an object here.")
        cursor = nested
    cursor[keys[-1]] = value


# ----------------------------------------------------------------------------------
# Value specifications
# ----------------------------------------------------------------------------------

_FORMAT_FIELD = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)(?::([^{}]*))?\}")
_FORMAT_SPEC = re.compile(r"^[-+ #0-9.,_%eEfFgGd]*$")


def _expression_array(
    expression: str, context: Mapping[str, Any], n: int, limits: Limits, what: str
) -> np.ndarray:
    """``expression`` evaluated over the context as an array of ``n`` finite numbers."""
    result = evaluate_expression(expression, {**HELPERS, **context}, limits=limits, what=what)
    try:
        array = np.asarray(result, dtype=np.float64)
    except (TypeError, ValueError):
        raise McpError(f"{what} {expression!r} did not produce numbers.") from None
    if array.ndim == 0:
        array = np.full(n, float(array))
    elif array.shape != (n,):
        raise McpError(
            f"{what} {expression!r} produced shape {array.shape}; it must be a single number "
            "per frame (use t, time, s or a variable, not x)."
        )
    if not np.all(np.isfinite(array)):
        bad = int(np.flatnonzero(~np.isfinite(array))[0])
        raise McpError(
            f"{what} {expression!r} is not finite at frame {bad} (check for 1/0, log(0))."
        )
    return array


def _colors(spec: Any, at: np.ndarray, what: str) -> List[str]:
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap, to_hex

    try:
        if isinstance(spec, str):
            cmap = colormaps[spec]
        else:
            if not isinstance(spec, (list, tuple)) or len(spec) < 2:
                raise ValueError("a colormap name or a list of two or more colours")
            cmap = LinearSegmentedColormap.from_list("binding", list(spec))
    except (KeyError, ValueError) as exc:
        raise McpError(
            f"{what}: colormap must be a name like 'viridis' or a list of colours ({exc})."
        ) from None
    return [to_hex(cmap(float(v))) for v in np.clip(at, 0.0, 1.0)]


def _format_strings(
    template: str, context: Mapping[str, np.ndarray], n: int, what: str
) -> List[str]:
    """``template`` with ``{name}`` / ``{name:.2f}`` replaced per frame. No attribute access.

    ``{{`` and ``}}`` are literal braces. Only a variable name and a plain number format are
    accepted, so a template cannot reach into objects the way ``str.format`` allows.
    """
    escaped = template.replace("{{", "\x00").replace("}}", "\x01")
    leftover = _FORMAT_FIELD.sub("", escaped)
    if "{" in leftover or "}" in leftover:
        raise McpError(
            f"{what}: format {template!r} has a brace that is not {{name}} or {{name:.2f}}; "
            "write {{ and }} for literal braces."
        )
    names = {m.group(1) for m in _FORMAT_FIELD.finditer(escaped)}
    for name in names:
        if name not in context:
            known = sorted(k for k in context if k not in HELPERS)
            raise McpError(
                f"{what}: format uses {{{name}}}, which is not a variable.{suggest(name, known)} "
                f"Available: {', '.join(known)}."
            )
    out: List[str] = []
    for i in range(n):

        def fill(match: "re.Match[str]") -> str:
            spec = match.group(2) or ""
            if not _FORMAT_SPEC.match(spec):
                raise McpError(
                    f"{what}: format spec {spec!r} is not a plain number format like .2f."
                )
            value = np.asarray(context[match.group(1)])
            item = float(value if value.ndim == 0 else value[i])
            try:
                if not spec:
                    return f"{item:g}"
                return format(int(round(item)) if spec.endswith("d") else item, spec)
            except ValueError as exc:
                raise McpError(f"{what}: cannot format {item!r} with {spec!r} ({exc}).") from None

        out.append(_FORMAT_FIELD.sub(fill, escaped).replace("\x00", "{").replace("\x01", "}"))
    return out


def evaluate_value(
    spec: Any,
    context: Mapping[str, Any],
    n: int,
    *,
    limits: Limits = DEFAULT_LIMITS,
    what: str = "value",
    text: bool = False,
) -> List[Any]:
    """The value ``spec`` takes at each of ``n`` frames, as plain Python values.

    ``context`` maps names to arrays of length ``n`` (or scalars). With ``text`` set the property
    is plain text (a colour, a label, a title...): a string is then that text, not an expression.
    A specification is

    * a number -- constant;
    * a string -- an expression: ``"1 + 3*t"``, ``"amp * sin(2*pi*time)"``;
    * a list -- each item is itself a specification, the result a list per frame
      (``xlim = ["0", "10*t"]``);
    * ``{"colormap": "viridis" | [colours], "at": expr}`` -- a colour that moves along a ramp;
    * ``{"choose": [...], "index": expr}`` -- one item of a list, picked by a number;
    * ``{"format": "t = {time:.2f} s"}`` -- text with the variables filled in.
    """
    if isinstance(spec, bool):
        return [spec] * n
    if isinstance(spec, (int, float)):
        if not math.isfinite(spec):
            raise McpError(f"{what}: {spec!r} is not finite.")
        return [spec] * n
    if isinstance(spec, str):
        if text:
            return [spec] * n
        array = _expression_array(spec, context, n, limits, what)
        return [float(v) for v in array]
    if isinstance(spec, (list, tuple)):
        columns = [evaluate_value(item, context, n, limits=limits, what=what) for item in spec]
        return [list(row) for row in zip(*columns)] if columns else [[] for _ in range(n)]
    if isinstance(spec, Mapping):
        keys = set(spec)
        if "colormap" in keys:
            if "at" not in keys:
                raise McpError(
                    f"{what}: a colormap needs 'at', the expression (0..1) along the ramp."
                )
            at = _expression_array(str(spec["at"]), context, n, limits, what)
            return _colors(spec["colormap"], at, what)
        if "choose" in keys:
            items = spec["choose"]
            if not isinstance(items, (list, tuple)) or not items or "index" not in keys:
                raise McpError(f"{what}: choose needs a non-empty list and an 'index' expression.")
            index = _expression_array(str(spec["index"]), context, n, limits, what)
            picked = np.clip(np.floor(index + 1e-9).astype(int), 0, len(items) - 1)
            return [items[int(i)] for i in picked]
        if "format" in keys:
            if not isinstance(spec["format"], str):
                raise McpError(f"{what}: format must be a string like 't = {{time:.1f}} s'.")
            return _format_strings(spec["format"], context, n, what)
        raise McpError(
            f"{what}: an object value needs one of 'colormap' (+at), 'choose' (+index) or "
            "'format'; "
            f"got keys {sorted(keys)}."
        )
    raise McpError(
        f"{what}: cannot use {type(spec).__name__} here. Give a number, an expression string, "
        "a list, or an object with colormap/choose/format."
    )


# ----------------------------------------------------------------------------------
# The timeline
# ----------------------------------------------------------------------------------


@dataclass
class Timeline:
    """Every variable at every frame of an animation, and where its stages begin and end."""

    n: int
    fps: float
    loop: bool
    t: np.ndarray
    frame: np.ndarray
    time: np.ndarray
    s: np.ndarray
    stage: np.ndarray
    stages: List[Dict[str, Any]]
    variables: Dict[str, np.ndarray]
    #: Variables a stage can move (numbers); the others are derived from them by expression.
    parameters: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.n / self.fps

    def arrays(self) -> Dict[str, Any]:
        """The context expressions are evaluated in: clock, stage position and variables."""
        base: Dict[str, Any] = {
            "t": self.t,
            "frame": self.frame.astype(np.float64),
            "n": float(self.n),
            "time": self.time,
            "s": self.s,
            "stage": self.stage.astype(np.float64),
        }
        base.update(self.variables)
        return base

    def at(self, index: int) -> Dict[str, float]:
        """Scalars of every name at frame ``index`` (what one rendered frame sees)."""
        return {
            key: float(value if np.ndim(value) == 0 else value[index])
            for key, value in self.arrays().items()
        }

    def summary(self) -> Dict[str, Any]:
        """JSON-safe: stage spans and each variable's first, last, min and max."""
        out: Dict[str, Any] = {
            "frames": self.n,
            "fps": self.fps,
            "duration_s": round(self.duration, 4),
            "stages": [dict(s) for s in self.stages],
        }
        table: Dict[str, Any] = {}
        for name, values in self.variables.items():
            table[name] = {
                "first": _short(values[0]),
                "last": _short(values[-1]),
                "min": _short(values.min()),
                "max": _short(values.max()),
            }
        if table:
            out["variables"] = table
        return out


def _short(value: Any) -> float:
    return float(f"{float(value):.6g}")


def _frames_for(spec: Mapping[str, Any], fps: float, what: str) -> Optional[int]:
    if spec.get("frames") is not None:
        count = spec["frames"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise McpError(f"{what}: frames must be a whole number of at least 1, got {count!r}.")
        return count
    if spec.get("duration") is not None:
        seconds = _number(spec["duration"], f"{what}: duration")
        if seconds <= 0:
            raise McpError(f"{what}: duration must be positive (seconds).")
        return max(1, int(round(seconds * fps)))
    return None


def _ease(name: Any, what: str) -> Callable[[Any], np.ndarray]:
    key = "linear" if name is None else str(name)
    if key not in EASINGS:
        raise McpError(
            f"{what}: unknown ease {key!r}.{suggest(key, EASINGS)} "
            f"Use one of: {', '.join(EASINGS)}."
        )
    return EASINGS[key]


def build_timeline(
    *,
    frames: Optional[int] = None,
    duration: Optional[float] = None,
    fps: float = 30.0,
    loop: bool = False,
    stages: Optional[Sequence[Mapping[str, Any]]] = None,
    variables: Optional[Mapping[str, Any]] = None,
    controls: Optional[Mapping[str, float]] = None,
    limits: Limits = DEFAULT_LIMITS,
    animation_limits: Optional[AnimationLimits] = None,
) -> Timeline:
    """Resolve an animation's clock, stages and variables into per-frame arrays.

    ``variables`` maps a name to a **number** (a parameter a stage may move; it holds that value
    until then) or an **expression** string (derived each frame from ``t``, ``time``, ``s``,
    ``stage`` and the parameters, e.g. ``"phase": "2*pi*t"``). ``controls`` are the figure's
    current control values; they act as parameters too, so a stage can sweep a slider.

    A stage ``{"name", "frames" | "duration", "ease", "animate": {...}}`` moves parameters.
    Each ``animate`` entry is a target number (from wherever the parameter is), ``[from, to]``,
    ``{"from", "to", "ease"}`` or an expression string in ``s`` (0..1 across the stage). A stage
    starts where the previous one left off and **ends exactly on its target**; afterwards the
    parameter holds its value.
    """
    caps = animation_limits or limits.animation
    fps = _number(fps, "fps")
    if not 1.0 <= fps <= 120.0:
        raise McpError("fps must be between 1 and 120.")
    stage_specs = list(stages or [])
    if len(stage_specs) > caps.max_stages:
        raise McpError(f"{len(stage_specs)} stages; the limit is {caps.max_stages}.")
    for index, spec in enumerate(stage_specs):
        if not isinstance(spec, Mapping):
            raise McpError(
                f"stages[{index}] must be an object like {{'duration': 2, 'animate': {{...}}}}."
            )
        unknown = set(spec) - {"name", "frames", "duration", "ease", "animate"}
        if unknown:
            raise McpError(
                f"stages[{index}] has unknown key(s) {sorted(unknown)}. Allowed: name, frames, "
                "duration, ease, animate."
            )

    # -- how many frames, and where each stage sits ---------------------------------
    counts: List[Optional[int]] = [
        _frames_for(spec, fps, f"stages[{i}]") for i, spec in enumerate(stage_specs)
    ]
    if stage_specs:
        total_hint = _frames_for({"frames": frames, "duration": duration}, fps, "the animation")
        open_stages = [i for i, c in enumerate(counts) if c is None]
        if open_stages:
            if total_hint is None:
                names = ", ".join(str(stage_specs[i].get("name", i)) for i in open_stages)
                raise McpError(
                    f"stage {names} has no length: give it frames or duration, or give the "
                    "animation a duration to share among the stages without one."
                )
            left = max(len(open_stages), total_hint - sum(c for c in counts if c is not None))
            for i in open_stages:
                counts[i] = max(1, left // len(open_stages))
        n = int(sum(c for c in counts if c is not None))
    else:
        total = _frames_for({"frames": frames, "duration": duration}, fps, "the animation")
        if total is None:
            raise McpError("Say how long the animation is: frames=60, or duration=2 (seconds).")
        n = total
    if n < 2:
        raise McpError("An animation needs at least 2 frames.")
    if n > caps.max_frames:
        raise McpError(
            f"{n} frames is over the limit of {caps.max_frames}. Lower fps or the duration "
            f"(at {fps:g} fps that is {caps.max_frames / fps:.1f} s)."
        )

    frame = np.arange(n)
    time = frame / fps
    t = frame / n if loop else frame / (n - 1)
    s = t.copy()
    stage_index = np.zeros(n, dtype=int)
    spans: List[Dict[str, Any]] = []
    cursor = 0
    for i, count in enumerate(counts):
        assert count is not None
        end = cursor + count
        stage_index[cursor:end] = i
        s[cursor:end] = 1.0 if count == 1 else np.linspace(0.0, 1.0, count)
        spans.append(
            {
                "name": str(stage_specs[i].get("name", f"stage{i + 1}")),
                "start": cursor,
                "end": end,
                "frames": count,
            }
        )
        cursor = end

    # -- declared variables ---------------------------------------------------------
    parameters: Dict[str, np.ndarray] = {}
    derived: Dict[str, str] = {}
    for key, value in (controls or {}).items():
        parameters[key] = np.full(n, float(value))
    for key, value in (variables or {}).items():
        if not is_identifier(key) or key in RESERVED_NAMES:
            raise McpError(
                f"Variable name {key!r} must be an identifier and not reserved "
                f"({', '.join(sorted(RESERVED_NAMES))})."
            )
        if isinstance(value, str):
            parameters.pop(key, None)
            derived[key] = value
        elif isinstance(value, (bool, int, float)):
            number = float(value)
            if not math.isfinite(number):
                raise McpError(f"Variable {key!r} must be finite.")
            derived.pop(key, None)
            parameters[key] = np.full(n, number)
        else:
            raise McpError(
                f"Variable {key!r}: give a number (a parameter) or an expression string, "
                f"got {type(value).__name__}."
            )

    clock: Dict[str, Any] = {
        "t": t, "frame": frame.astype(np.float64), "n": float(n), "time": time,
        "s": s, "stage": stage_index.astype(np.float64),
    }  # fmt: skip

    def context(lo: int, hi: int) -> Dict[str, Any]:
        sliced = {k: (v[lo:hi] if np.ndim(v) else v) for k, v in clock.items()}
        sliced.update({k: v[lo:hi] for k, v in parameters.items()})
        return sliced

    # -- stages move parameters ------------------------------------------------------
    for i, spec in enumerate(stage_specs):
        lo, hi = spans[i]["start"], spans[i]["end"]
        label = f"stage {spans[i]['name']!r}"
        stage_ease = _ease(spec.get("ease"), label)
        moves = spec.get("animate") or {}
        if not isinstance(moves, Mapping):
            raise McpError(f"{label}: animate must be an object like {{'amp': [1, 3]}}.")
        for name, move in moves.items():
            if name in derived:
                raise McpError(
                    f"{label} animates {name!r}, which is an expression in variables. Make it a "
                    f"number there (its starting value) so a stage can move it, or animate the "
                    f"parameters it depends on."
                )
            ease = stage_ease
            start: Optional[float] = None
            stop: Optional[float] = None
            expression: Optional[str] = None
            if isinstance(move, str):
                expression = move
            elif isinstance(move, bool) or isinstance(move, (int, float)):
                stop = _number(move, f"{label}: {name}")
            elif isinstance(move, (list, tuple)):
                if len(move) != 2:
                    raise McpError(
                        f"{label}: {name} as a list is [from, to], got {len(move)} items."
                    )
                start, stop = _number(move[0], f"{label}: {name} from"), _number(
                    move[1], f"{label}: {name} to"
                )
            elif isinstance(move, Mapping):
                unknown = set(move) - {"from", "to", "ease", "expr"}
                if unknown:
                    raise McpError(
                        f"{label}: {name} has unknown key(s) {sorted(unknown)}; "
                        "use from, to, ease, expr."
                    )
                if "expr" in move:
                    expression = str(move["expr"])
                else:
                    if "to" not in move:
                        raise McpError(f"{label}: {name} needs 'to' (or 'expr').")
                    stop = _number(move["to"], f"{label}: {name} to")
                if "from" in move:
                    start = _number(move["from"], f"{label}: {name} from")
                if "ease" in move:
                    ease = _ease(move["ease"], f"{label}: {name}")
            else:
                raise McpError(
                    f"{label}: {name} must be a number, [from, to], an object or an expression."
                )
            if name not in parameters:
                if start is None:
                    raise McpError(
                        f"{label} animates {name!r}, which has no starting value. Declare it "
                        f"(variables={{'{name}': <start>}}) or give [from, to]."
                    )
                if not is_identifier(name) or name in RESERVED_NAMES:
                    raise McpError(f"{label}: {name!r} is not a usable variable name.")
                parameters[name] = np.full(n, start)
            values = parameters[name]
            if expression is not None:
                piece = _expression_array(
                    expression, {**context(lo, hi)}, hi - lo, limits, f"{label}: {name}"
                )
                values[lo:hi] = piece
                values[hi:] = piece[-1]
            else:
                begin = float(values[lo]) if start is None else start
                weight = ease(s[lo:hi])
                values[lo:hi] = begin + (stop - begin) * weight  # type: ignore[operator]
                # where the easing ends: the target, or the start again for there_and_back
                end = float(ease(np.array(1.0)))
                values[hi:] = (
                    float(stop) if end == 1.0 else begin + (float(stop) - begin) * end
                )  # type: ignore[arg-type]

    # -- derived variables ------------------------------------------------------------
    resolved: Dict[str, np.ndarray] = dict(parameters)
    for name, expression in derived.items():
        scope = {**clock, **resolved}
        resolved[name] = _expression_array(expression, scope, n, limits, f"variable {name!r}")

    return Timeline(
        n=n,
        fps=fps,
        loop=bool(loop),
        t=t,
        frame=frame,
        time=time,
        s=s,
        stage=stage_index,
        stages=spans,
        variables=resolved,
        parameters=list(parameters),
    )


def timeline_guide() -> str:
    """The animation vocabulary as Markdown, for the model-facing guide."""
    easings = ", ".join(f"`{name}`" for name in EASINGS)
    helpers = ", ".join(f"`{name}`" for name in HELPERS)
    return (
        f"Easings: {easings}. Helper functions usable in any animation expression: {helpers}. "
        "Reserved names: " + ", ".join(f"`{n}`" for n in sorted(RESERVED_NAMES)) + "."
    )
