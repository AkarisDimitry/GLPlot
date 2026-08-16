"""A pure, ordered chain of signal transforms -- the engine behind the Pipeline panel.

The Pipeline panel lets a user snap together a sequence of Math Lab operations
(``baseline -> smooth -> peaks`` and the like) with click-and-drag. All of the *logic*
of that -- what steps exist, what each one does to an ``(x, y)`` pair, how a chain runs,
how a chain is reordered -- lives here, with no imgui and no engine, so it is fully
unit-testable headless. The panel in :mod:`glplot.gui.panels.pipeline` is only the
drag-and-drop shell around it.

Only **signal -> signal** operations belong in a chain: each step takes an ``(x, y)``
and returns a new ``(x, y)`` the next step consumes. Terminal analyses (peaks, FFT,
histogram, a fit) are deliberately absent -- they end a chain rather than continue it, so
they stay on the Math Lab tabs where their rich output has somewhere to go.

Every ``apply`` delegates to :mod:`glplot.gui.mathops`, so the numerics -- and their scipy
fallbacks -- are shared with the Math Lab tabs and cannot drift from them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from glplot.gui import mathops

__all__ = [
    "ParamSpec",
    "StepType",
    "STEP_TYPES",
    "STEP_KEYS",
    "PipelineStep",
    "new_step",
    "run_pipeline",
    "move_step",
    "PipelineResult",
]


# ---------------------------------------------------------------------------
# parameter schema -- lets the panel render an editor for any step generically
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    """One editable parameter of a step, plus enough metadata to render an editor.

    ``kind`` is one of ``"enum"``, ``"int"``, ``"float"``, ``"logfloat"``. ``logfloat``
    stores and edits ``log10(value)`` on a slider so a control can span decades (used for
    the AsLS smoothness), and the step's ``apply`` raises it back to a linear value.
    """

    name: str
    kind: str
    label: str
    options: Tuple[str, ...] = ()
    vmin: float = 0.0
    vmax: float = 0.0
    help: str = ""


# ---------------------------------------------------------------------------
# the transforms -- each takes (x, y, params) and returns a new (x, y)
# ---------------------------------------------------------------------------


def _as_xy(x: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def _sorted_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ascending-x order, matching mathops -- steps that need a sorted grid share this."""
    if x.size < 2 or bool(np.all(np.diff(x) >= 0.0)):
        return x, y
    order = np.argsort(x, kind="stable")
    return x[order], y[order]


def _apply_smooth(x: Any, y: Any, p: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    xa, ya = _as_xy(x, y)
    return xa, mathops.smooth(ya, method=str(p["method"]), window=int(p["window"]))


def _apply_baseline(x: Any, y: Any, p: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    xa, ya = _as_xy(x, y)
    kind = str(p["kind"])
    if kind == "asls":
        baseline = mathops.baseline_asls(ya, lam=10.0 ** float(p["lam_log10"]), p=float(p["p"]))
        return xa, ya - baseline
    return xa, mathops.detrend(xa, ya, kind=kind)


def _apply_normalize(x: Any, y: Any, p: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    xa, ya = _as_xy(x, y)
    return xa, mathops.normalize(ya, mode=str(p["mode"]))


def _apply_derivative(x: Any, y: Any, p: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = _sorted_xy(*_as_xy(x, y))
    return xs, mathops.derivative(xs, ys, order=int(p["order"]), method=str(p["method"]))


def _apply_integral(x: Any, y: Any, p: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = _sorted_xy(*_as_xy(x, y))
    return xs, mathops.integrate(xs, ys, method=str(p["method"]), initial=float(p["initial"]))


def _apply_filter(x: Any, y: Any, p: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = _sorted_xy(*_as_xy(x, y))
    n = xs.size
    if n < 2 or float(xs[-1] - xs[0]) <= 0.0:
        raise ValueError("filter needs at least 2 samples over a non-zero x span")
    nyquist = 0.5 * (n - 1) / float(xs[-1] - xs[0])
    btype = str(p["btype"])
    low = min(max(float(p["cutoff"]), 1e-6), 1.0 - 1e-6) * nyquist
    if btype in ("bandpass", "bandstop"):
        high = min(max(float(p["cutoff_hi"]), 1e-6), 1.0 - 1e-6) * nyquist
        cutoff: Any = (low, high)
    else:
        cutoff = low
    return xs, mathops.butter_filter(xs, ys, cutoff, btype=btype, order=int(p["order"]))


def _apply_resample(x: Any, y: Any, p: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    return mathops.resample(x, y, int(p["n"]), kind=str(p["kind"]))


# ---------------------------------------------------------------------------
# the step registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepType:
    """A kind of pipeline step: its label, its parameters, and how it transforms a signal."""

    key: str
    label: str
    apply: Callable[[Any, Any, Dict[str, Any]], Tuple[np.ndarray, np.ndarray]]
    params: Tuple[ParamSpec, ...]
    defaults: Dict[str, Any]
    summary: str = ""


# Presentation order: conditioning first (baseline, smooth, filter), then shape-changing
# transforms. This is also the order the palette shows them in.
_STEP_LIST: Tuple[StepType, ...] = (
    StepType(
        key="baseline",
        label="Baseline",
        apply=_apply_baseline,
        params=(
            ParamSpec("kind", "enum", "Baseline", ("linear", "constant", "asls")),
            ParamSpec("lam_log10", "logfloat", "Smoothness", vmin=2.0, vmax=9.0),
            ParamSpec("p", "float", "Asymmetry", vmin=0.001, vmax=0.5),
        ),
        defaults={"kind": "linear", "lam_log10": 6.0, "p": 0.01},
        summary="remove a constant, linear or curved background",
    ),
    StepType(
        key="smooth",
        label="Smooth",
        apply=_apply_smooth,
        params=(
            ParamSpec(
                "method", "enum", "Method", ("moving_average", "gaussian", "savgol", "median")
            ),
            ParamSpec("window", "int", "Window", vmin=1.0, vmax=9999.0),
        ),
        defaults={"method": "moving_average", "window": 11},
        summary="reduce noise with a moving window",
    ),
    StepType(
        key="filter",
        label="Filter",
        apply=_apply_filter,
        params=(
            ParamSpec("btype", "enum", "Type", ("lowpass", "highpass", "bandpass", "bandstop")),
            ParamSpec("cutoff", "float", "Cutoff", vmin=0.001, vmax=0.999),
            ParamSpec("cutoff_hi", "float", "High cutoff", vmin=0.001, vmax=0.999),
            ParamSpec("order", "int", "Order", vmin=1.0, vmax=8.0),
        ),
        defaults={"btype": "lowpass", "cutoff": 0.25, "cutoff_hi": 0.4, "order": 4},
        summary="Butterworth frequency filter (cutoff as a fraction of Nyquist)",
    ),
    StepType(
        key="normalize",
        label="Normalize",
        apply=_apply_normalize,
        params=(ParamSpec("mode", "enum", "Mode", ("minmax", "zscore", "max_abs", "area")),),
        defaults={"mode": "minmax"},
        summary="rescale onto a standard range",
    ),
    StepType(
        key="derivative",
        label="Derivative",
        apply=_apply_derivative,
        params=(
            ParamSpec("order", "int", "Order", vmin=1.0, vmax=8.0),
            ParamSpec("method", "enum", "Method", ("central", "forward", "savgol")),
        ),
        defaults={"order": 1, "method": "central"},
        summary="differentiate with respect to x",
    ),
    StepType(
        key="integral",
        label="Integral",
        apply=_apply_integral,
        params=(
            ParamSpec("method", "enum", "Method", ("trapezoid", "simpson", "rectangle")),
            ParamSpec("initial", "float", "Constant", vmin=0.0, vmax=0.0),
        ),
        defaults={"method": "trapezoid", "initial": 0.0},
        summary="cumulative integral with respect to x",
    ),
    StepType(
        key="resample",
        label="Resample",
        apply=_apply_resample,
        params=(
            ParamSpec("n", "int", "Samples", vmin=2.0, vmax=1_000_000.0),
            ParamSpec("kind", "enum", "Kind", ("linear", "cubic")),
        ),
        defaults={"n": 512, "kind": "linear"},
        summary="resample onto an even grid",
    ),
)

#: key -> StepType.
STEP_TYPES: Dict[str, StepType] = {st.key: st for st in _STEP_LIST}

#: Step keys in palette order.
STEP_KEYS: Tuple[str, ...] = tuple(st.key for st in _STEP_LIST)


# ---------------------------------------------------------------------------
# a pipeline: an ordered list of configured steps
# ---------------------------------------------------------------------------


@dataclass
class PipelineStep:
    """One configured step in a chain: a step kind plus its own parameter values.

    ``params`` is a *copy* of the step type's defaults, owned by this instance -- two
    Smooth steps in one chain must be able to hold different windows.
    """

    kind: str
    params: Dict[str, Any] = field(default_factory=dict)


def new_step(kind: str) -> PipelineStep:
    """A fresh :class:`PipelineStep` of ``kind`` with its own copy of the default params.

    Raises:
        KeyError: If ``kind`` is not a registered step (a programming error -- the palette
            only ever offers real keys).
    """
    return PipelineStep(kind=kind, params=dict(STEP_TYPES[kind].defaults))


def move_step(steps: List[PipelineStep], src: int, dst: int) -> None:
    """Move the step at index ``src`` to index ``dst``, shifting the rest -- the reorder a
    drag performs. Out-of-range indices are a no-op, so a stray drag cannot raise.
    """
    n = len(steps)
    if n == 0 or src == dst or not (0 <= src < n) or not (0 <= dst < n):
        return
    step = steps.pop(src)
    steps.insert(dst, step)


@dataclass
class PipelineResult:
    """The outcome of running a chain: the final ``(x, y)`` plus any per-step failure.

    ``failed_index``/``error`` are None on success. On failure they name the step that
    raised and its message, and ``x``/``y`` hold the signal *as of the step before* the
    failure, so the preview can still show how far the chain got.
    """

    x: np.ndarray
    y: np.ndarray
    failed_index: Optional[int] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.failed_index is None


def run_pipeline(x: Any, y: Any, steps: List[PipelineStep]) -> PipelineResult:
    """Run ``steps`` in order over ``(x, y)``, each step feeding the next.

    A step that raises stops the chain: the result carries the index and message of the
    failing step and the signal produced by the steps before it. Every step's numerics are
    the same :mod:`glplot.gui.mathops` the Math Lab tabs use.

    Never raises for a bad step -- a broken pipeline is a message to show, not a crash in a
    draw callback. It *does* propagate a genuinely malformed input (non-numeric x/y) from
    the initial coercion, which is a caller error rather than a step failure.
    """
    cx, cy = _as_xy(x, y)
    for index, step in enumerate(steps):
        step_type = STEP_TYPES.get(step.kind)
        if step_type is None:
            return PipelineResult(cx, cy, index, f"unknown step {step.kind!r}")
        try:
            cx, cy = step_type.apply(cx, cy, step.params)
        except Exception as exc:  # ValueError from mathops, and anything defensive
            return PipelineResult(cx, cy, index, str(exc))
    return PipelineResult(cx, cy)
