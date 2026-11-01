"""Data analysis a language model can ask for by name: fit, smooth, FFT, peaks, filters, ...

The workstation's Math Lab (:mod:`glplot.gui.mathops`) already holds well-tested numerical
routines. This module exposes the useful ones through one declarative registry, so a model that
has loaded a table can *analyse* it -- fit a curve, find peaks, take a spectrum -- and then plot the
result, instead of only drawing the raw numbers.

Every operation is a row in :data:`OPERATIONS`: its name, what it reads, what it produces, and the
options it accepts, each with a type, a default and a description. That one table is the validator
(unknown or out-of-range options are refused with the valid ones named), the generated guide topic
and the tool description, so the three cannot disagree.

An operation produces one of three things:

* a **column** the same length as the input, added to the dataset (``smooth``, ``fit``...);
* a new **table** of a different length (``fft``, ``histogram``, ``peaks``...);
* a **report** only (``describe``, ``correlate``).

Pure logic: numpy and :mod:`glplot.gui.mathops`, no MCP SDK and no GL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..gui import mathops
from .state import Limits, McpError, suggest

__all__ = [
    "OPERATIONS",
    "Operation",
    "Option",
    "Outcome",
    "operation",
    "operations_guide",
    "operations_summary",
    "validate_options",
]


# ----------------------------------------------------------------------------------
# Declarations
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Option:
    """One operation option: enough to validate a value and to document it."""

    name: str
    kind: str  # "choice" | "int" | "float" | "bool" | "text" | "floats"
    description: str
    default: Any = None
    choices: Tuple[str, ...] = ()
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    required: bool = False

    def describe(self) -> str:
        """``name (type, default): description`` for the guide."""
        if self.kind == "choice":
            shape = "one of " + ", ".join(self.choices)
        elif self.kind in ("int", "float"):
            low = "" if self.minimum is None else f"{self.minimum:g}"
            high = "" if self.maximum is None else f"{self.maximum:g}"
            shape = f"{self.kind}" + (f" {low}..{high}" if low or high else "")
        elif self.kind == "floats":
            shape = "list of numbers"
        else:
            shape = self.kind
        default = "required" if self.required else f"default {self.default!r}"
        return f"`{self.name}` ({shape}; {default}) -- {self.description}"


@dataclass
class Outcome:
    """What an operation returns, before the toolbox stores anything."""

    column: Optional[np.ndarray] = None
    table: Optional[Dict[str, np.ndarray]] = None
    report: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    column_suffix: Optional[str] = None
    #: One short sentence with the result that matters ("R^2 0.996 ..."), for the reply's message.
    headline: str = ""
    #: (add_plot kind, x column, y column) that draws a table result, for the reply's plot_hint.
    plot: Optional[Tuple[str, str, str]] = None
    #: Category labels of columns of ``table`` (``{column: [labels]}``), carried to the new dataset.
    categories: Dict[str, List[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class Operation:
    name: str
    summary: str
    #: "column", "table" or "report": the main thing it hands back.
    produces: str
    run: Callable[["Context"], Outcome]
    options: Tuple[Option, ...] = ()
    #: Reads x (an abscissa). When x is not given the row index is used.
    uses_x: bool = True
    #: x must be a real column (a relationship between two series is meaningless against an index).
    x_required: bool = False
    #: Say so when x defaulted to the row index. Off where x only refines the result (a lag axis).
    index_note: bool = True
    #: Default suffix of the column added by a "column" operation: ``<y>_<suffix>``.
    suffix: str = ""
    example: str = ""

    def option(self, name: str) -> Option:
        for candidate in self.options:
            if candidate.name == name:
                return candidate
        raise KeyError(name)


@dataclass
class Context:
    """The validated inputs an operation sees."""

    x: np.ndarray
    y: np.ndarray
    options: Dict[str, Any]
    x_name: str
    y_name: str
    x_is_index: bool
    limits: Limits
    #: Every column of the dataset, for operations that group by another column.
    columns: Mapping[str, np.ndarray] = field(default_factory=dict)
    #: Labels of the dataset's category columns.
    categories: Mapping[str, List[str]] = field(default_factory=dict)


# ----------------------------------------------------------------------------------
# Option validation
# ----------------------------------------------------------------------------------


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise McpError(f"{what} must be a number, got {value!r}.")
    try:
        number = float(value)
    except ValueError:
        raise McpError(f"{what} must be a number, got {value!r}.") from None
    if not math.isfinite(number):
        raise McpError(f"{what} must be finite, got {value!r}.")
    return number


def _coerce(op: Operation, option: Option, value: Any) -> Any:
    what = f"option {option.name!r} of {op.name}"
    if option.kind == "choice":
        text = str(value).strip().lower()
        if text not in option.choices:
            raise McpError(
                f"{what} must be one of: {', '.join(option.choices)}; got {value!r}."
                f"{suggest(text, option.choices)}"
            )
        return text
    if option.kind in ("int", "float"):
        number = _number(value, what)
        if option.kind == "int":
            if number != int(number):
                raise McpError(f"{what} must be a whole number, got {value!r}.")
            number = int(number)
        if option.minimum is not None and number < option.minimum:
            raise McpError(f"{what} must be at least {option.minimum:g}, got {number}.")
        if option.maximum is not None and number > option.maximum:
            raise McpError(f"{what} must be at most {option.maximum:g}, got {number}.")
        return number
    if option.kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        raise McpError(f"{what} must be true or false, got {value!r}.")
    if option.kind == "floats":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = [value]
        if not isinstance(value, (list, tuple)) or not value:
            raise McpError(f"{what} must be a non-empty list of numbers, got {value!r}.")
        if len(value) > 32:
            raise McpError(f"{what} takes at most 32 numbers, got {len(value)}.")
        return [_number(item, f"{what}[{index}]") for index, item in enumerate(value)]
    if option.kind == "text":
        if not isinstance(value, str) or not value.strip():
            raise McpError(f"{what} must be non-empty text, got {value!r}.")
        return value.strip()
    raise McpError(f"{what}: unsupported option type {option.kind!r}.")  # pragma: no cover


def validate_options(op: Operation, given: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """``given`` checked against ``op``'s declared options, with defaults filled in.

    Unknown names are refused (and the valid ones listed) rather than ignored: a misspelt option
    that silently did nothing would leave the model believing it had smoothed with window 51 when
    it had used the default.
    """
    given = {} if given is None else dict(given)
    if not isinstance(given, dict):
        raise McpError("options must be an object of name -> value.")
    known = [option.name for option in op.options]
    unknown = sorted(set(given) - set(known))
    if unknown:
        hint = suggest(unknown[0], known)
        raise McpError(
            f"{op.name} has no option(s) {', '.join(map(repr, unknown))}.{hint} "
            f"Valid options: {', '.join(known) if known else 'none'}."
        )
    out: Dict[str, Any] = {}
    for option in op.options:
        if option.name in given and given[option.name] is not None:
            out[option.name] = _coerce(op, option, given[option.name])
        elif option.required:
            raise McpError(f"{op.name} needs the option {option.name!r}: {option.description}")
        else:
            out[option.name] = option.default
    return out


# ----------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------


def _round(value: Any, digits: int = 6) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(number):
        return None
    return 0.0 if number == 0.0 else float(f"{number:.{digits}g}")


def _round_map(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: _round(value) for key, value in mapping.items()}


def _spacing(ctx: Context) -> Tuple[float, bool]:
    """Median step of x and whether it is uniform: what a filter or an FFT assumes."""
    steps = np.diff(ctx.x[np.isfinite(ctx.x)])
    if steps.size == 0 or not np.all(steps > 0):
        return float("nan"), False
    step = float(np.median(steps))
    return step, bool(np.allclose(steps, step, rtol=1e-3, atol=0.0))


# ----------------------------------------------------------------------------------
# Operations
# ----------------------------------------------------------------------------------


def _run_describe(ctx: Context) -> Outcome:
    report = _round_map(mathops.describe_extended(ctx.y))
    return Outcome(
        report=report,
        headline=f"mean {report['mean']}, std {report['std']}, median {report['median']}, "
        f"n {report['n']:g}",
    )


_FIT_MODELS = (
    "linear",
    "polynomial",
    "gaussian",
    "lorentzian",
    "exp_decay",
    "power_law",
    "sigmoid",
    "custom",
    "multipeak",
)


def _run_fit(ctx: Context) -> Outcome:
    o = ctx.options
    name = o["model"]
    notes: List[str] = []
    if name == "polynomial":
        degree = int(o["degree"])
        if ctx.x.size <= degree:
            raise McpError(f"A degree-{degree} polynomial needs more than {degree} rows.")
        coefficients, fitted = mathops.fit_polynomial(ctx.x, ctx.y, degree)
        try:
            _, _, covariance = mathops.fit_polynomial_covariance(ctx.x, ctx.y, degree)
            errors = mathops.fit_errors(covariance)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            errors = np.full(len(coefficients), np.nan)
        names = [f"c{degree - k}" for k in range(degree + 1)]
        label = f"polynomial (degree {degree})"
        formula = " + ".join(f"c{degree - k}*x^{degree - k}" for k in range(degree + 1))
        values = np.asarray(coefficients, dtype=float)
    else:
        if name == "custom":
            if not o["expression"]:
                raise McpError(
                    "fit with model='custom' needs options.expression, e.g. 'a*exp(-x/b)+c' "
                    "(a function of x; every other name becomes a fitted parameter)."
                )
            model: Any = mathops.custom_model(o["expression"])
        elif name == "multipeak":
            model = mathops.multipeak_model(int(o["peaks"]), o["shape"])
        else:
            model = mathops.resolve_model(name)
        runner = mathops.fit_model_robust if o["robust"] else mathops.fit_model
        try:
            values, covariance, fitted, names = runner(ctx.x, ctx.y, model, p0=o["p0"])
        except RuntimeError as exc:
            raise McpError(
                f"The {name} fit did not converge ({exc}). Try another model, a better "
                "starting point with options.p0 (one number per parameter), or smooth/resample "
                "the data first."
            ) from exc
        errors = mathops.fit_errors(covariance)
        label, formula = model.label, model.formula
    stats = mathops.fit_statistics(ctx.y, fitted, len(values))
    report = {
        "model": label,
        "formula": formula,
        "parameters": [
            {"name": n, "value": _round(v), "stderr": _round(e)}
            for n, v, e in zip(names, values, errors)
        ],
        "statistics": {
            key: _round(stats[key]) for key in ("r_squared", "rmse", "reduced_chi_squared", "n")
        },
    }
    if stats["r_squared"] < 0.5:
        notes.append(
            "Low R^2: this model describes the data poorly. Plot the fit next to the data, then "
            "try another model (or options.p0 for a better starting point)."
        )
    if o["robust"]:
        notes.append("Outlier-robust fit (soft_l1 loss).")
    shown = ", ".join(f"{p['name']}={p['value']}" for p in report["parameters"][:6])
    if len(report["parameters"]) > 6:
        shown += ", ..."
    headline = f"{label}: R^2 {report['statistics']['r_squared']} ({shown})"
    return Outcome(
        column=np.asarray(fitted, dtype=float), report=report, notes=notes, headline=headline
    )


def _run_smooth(ctx: Context) -> Outcome:
    o = ctx.options
    extra: Dict[str, Any] = {}
    if o["method"] == "savgol":
        extra["polyorder"] = int(o["polyorder"])
    if o["method"] == "gaussian" and o["sigma"] is not None:
        extra["sigma"] = float(o["sigma"])
    if o["method"] == "ema" and o["alpha"] is not None:
        extra["alpha"] = float(o["alpha"])
    result = mathops.smooth(ctx.y, method=o["method"], window=int(o["window"]), **extra)
    notes = []
    if int(o["window"]) % 2 == 0 and o["method"] != "ema":
        notes.append(
            "Even windows are made odd (+1) so the output is not shifted by half a sample."
        )
    return Outcome(column=np.asarray(result, dtype=float), notes=notes)


def _run_rolling(ctx: Context) -> Outcome:
    o = ctx.options
    result = mathops.rolling_stat(ctx.y, int(o["window"]), stat=o["stat"], center=o["center"])
    return Outcome(column=np.asarray(result, dtype=float), column_suffix=f"roll_{o['stat']}")


def _run_derivative(ctx: Context) -> Outcome:
    o = ctx.options
    result = mathops.derivative(
        ctx.x,
        ctx.y,
        order=int(o["order"]),
        method=o["method"],
        window=int(o["window"]),
        polyorder=int(o["polyorder"]),
    )
    return Outcome(
        column=np.asarray(result, dtype=float),
        column_suffix="d1" if int(o["order"]) == 1 else "d2",
    )


def _run_integrate(ctx: Context) -> Outcome:
    method = ctx.options["method"]
    cumulative = mathops.integrate(ctx.x, ctx.y, method=method)
    total = mathops.definite_integral(ctx.x, ctx.y, method=method)
    return Outcome(
        column=np.asarray(cumulative, dtype=float),
        report={"total": _round(total), "method": method},
        headline=f"total integral {_round(total)}",
    )


def _run_fft(ctx: Context) -> Outcome:
    step, uniform = _spacing(ctx)
    frequency, amplitude = mathops.fft_spectrum(ctx.x, ctx.y)
    report: Dict[str, Any] = {"sample_spacing": _round(step), "uniform": uniform}
    notes: List[str] = []
    if not uniform:
        notes.append(
            "x is not evenly spaced, so the spectrum is only approximate: resample to an even grid "
            "first (operation 'resample')."
        )
    if ctx.x_is_index:
        notes.append("No x column was given: frequencies are in cycles per sample.")
    if amplitude.size > 1:
        peak = 1 + int(np.argmax(amplitude[1:]))
        report["dominant_frequency"] = _round(frequency[peak])
        report["dominant_amplitude"] = _round(amplitude[peak])
        report["dc_amplitude"] = _round(amplitude[0])
    headline = (
        f"dominant frequency {report['dominant_frequency']} "
        f"(amplitude {report['dominant_amplitude']})"
        if "dominant_frequency" in report
        else "spectrum computed"
    )
    return Outcome(
        table={"frequency": frequency, "amplitude": amplitude},
        report=report,
        notes=notes,
        headline=headline,
        plot=("line", "frequency", "amplitude"),
    )


def _run_peaks(ctx: Context) -> Outcome:
    o = ctx.options
    positions, heights, props = mathops.find_peaks(
        ctx.x,
        ctx.y,
        height=o["height"],
        prominence=o["prominence"],
        distance=o["distance"],
        width=o["width"],
    )
    count = int(len(positions))
    cap = 500
    notes = []
    if count > cap:
        keep = np.argsort(np.asarray(props["prominences"]))[::-1][:cap]
        keep.sort()
        positions, heights = np.asarray(positions)[keep], np.asarray(heights)[keep]
        props = {key: np.asarray(value)[keep] for key, value in props.items()}
        notes.append(
            f"{count} peaks found; the {cap} most prominent are kept. Raise options.prominence."
        )
    if count == 0:
        notes.append("No peaks found: lower options.prominence/height or smooth less.")
    table = {
        "position": np.asarray(positions, dtype=float),
        "height": np.asarray(heights, dtype=float),
        "prominence": np.asarray(props.get("prominences", []), dtype=float),
        "width": np.asarray(props.get("widths", []), dtype=float),
        "area": np.asarray(props.get("areas", []), dtype=float),
    }
    ranked = np.argsort(table["prominence"])[::-1][:5]
    report = {
        "count": count,
        "strongest": [
            {"position": _round(table["position"][i]), "height": _round(table["height"][i])}
            for i in ranked
        ],
    }
    strongest = report["strongest"][0] if report["strongest"] else None
    headline = f"{count} peak{'s' if count != 1 else ''}" + (
        f"; strongest at {strongest['position']} (height {strongest['height']})"
        if strongest
        else ""
    )
    return Outcome(
        table=table,
        report=report,
        notes=notes,
        headline=headline,
        plot=("scatter", "position", "height"),
    )


def _run_filter(ctx: Context) -> Outcome:
    o = ctx.options
    btype, cutoff = o["btype"], list(o["cutoff"])
    need = 2 if btype in ("bandpass", "bandstop") else 1
    if len(cutoff) != need:
        raise McpError(
            f"filter with btype={btype!r} needs {need} cutoff value(s) "
            f"({'low, high' if need == 2 else 'one'}), got {len(cutoff)}."
        )
    step, uniform = _spacing(ctx)
    notes = []
    if not uniform:
        notes.append("x is not evenly spaced; the filter assumes a constant sample rate.")
    rate = 1.0 / step if step and math.isfinite(step) else float("nan")
    nyquist = 0.5 * rate
    if math.isfinite(nyquist) and any(c >= nyquist for c in cutoff):
        raise McpError(
            f"cutoff {cutoff} must be below the Nyquist frequency {nyquist:g} "
            f"(sampling every {step:g} in x = {rate:g} samples per unit)."
        )
    result = mathops.iir_filter(
        ctx.x,
        ctx.y,
        cutoff if need == 2 else cutoff[0],
        btype=btype,
        family=o["family"],
        order=int(o["order"]),
    )
    return Outcome(
        column=np.asarray(result, dtype=float),
        report={"sample_rate": _round(rate), "nyquist": _round(nyquist)},
        notes=notes,
    )


def _run_detrend(ctx: Context) -> Outcome:
    result = mathops.detrend(ctx.x, ctx.y, kind=ctx.options["kind"])
    return Outcome(column=np.asarray(result, dtype=float))


def _run_normalize(ctx: Context) -> Outcome:
    result = mathops.normalize(ctx.y, mode=ctx.options["mode"])
    return Outcome(column=np.asarray(result, dtype=float))


def _run_resample(ctx: Context) -> Outcome:
    o = ctx.options
    if int(o["n"]) > ctx.limits.max_analysis_rows:
        raise McpError(f"n must be at most {ctx.limits.max_analysis_rows}.")
    new_x, new_y = mathops.resample(ctx.x, ctx.y, int(o["n"]), kind=o["kind"])
    x_name = "index" if ctx.x_is_index else ctx.x_name
    y_name = ctx.y_name if ctx.y_name != x_name else f"{ctx.y_name}_y"
    return Outcome(
        table={x_name: np.asarray(new_x), y_name: np.asarray(new_y)},
        headline=f"{int(o['n'])} points on an even grid",
        plot=("line", x_name, y_name),
    )


def _run_histogram(ctx: Context) -> Outcome:
    o = ctx.options
    edges, heights, centers = mathops.histogram(ctx.y, bins=int(o["bins"]), density=o["density"])
    left, right = np.asarray(edges)[:-1], np.asarray(edges)[1:]
    column = "density" if o["density"] else "count"
    return Outcome(
        table={
            "bin_center": np.asarray(centers, dtype=float),
            "bin_left": left.astype(float),
            "bin_right": right.astype(float),
            column: np.asarray(heights, dtype=float),
        },
        report={"bins": int(len(centers))},
        headline=f"{int(len(centers))} bins",
        plot=("bar", "bin_center", column),
    )


def _run_correlate(ctx: Context) -> Outcome:
    report = _round_map(mathops.correlate(ctx.x, ctx.y))
    strength = abs(report.get("pearson_r") or 0.0)
    word = "strong" if strength >= 0.7 else "moderate" if strength >= 0.4 else "weak"
    return Outcome(
        report=report,
        notes=[f"A {word} linear relationship (|r| = {strength:.2f})."],
        headline=f"Pearson r {report.get('pearson_r')}, Spearman rho {report.get('spearman_rho')}, "
        f"R^2 {report.get('r_squared')}",
    )


def _run_autocorrelation(ctx: Context) -> Outcome:
    o = ctx.options
    lags, values = mathops.autocorrelation(ctx.y, max_lag=o["max_lag"], detrend=o["detrend"])
    step, uniform = _spacing(ctx)
    table = {"lag": np.asarray(lags, dtype=float), "autocorrelation": np.asarray(values)}
    report: Dict[str, Any] = {}
    if uniform and not ctx.x_is_index:
        table["lag_x"] = table["lag"] * step
    return Outcome(
        table=table,
        report=report,
        headline=f"{len(values)} lags",
        plot=("line", "lag_x" if "lag_x" in table else "lag", "autocorrelation"),
    )


_AGGREGATES = ("mean", "sum", "median", "min", "max", "std", "count")


def _run_aggregate(ctx: Context) -> Outcome:
    o = ctx.options
    by = o["by"]
    if by not in ctx.columns:
        raise McpError(
            f"Unknown column {by!r} for options.by.{suggest(by, ctx.columns)} "
            f"Columns: {', '.join(ctx.columns)}."
        )
    if by == ctx.y_name:
        raise McpError("options.by must be a different column from y (the one being summarised).")
    keys = np.asarray(ctx.columns[by], dtype=float)
    groups = np.unique(keys[np.isfinite(keys)])
    if groups.size == 0:
        raise McpError(f"Column {by!r} has no values to group by.")
    if groups.size > 5000:
        raise McpError(
            f"{groups.size} distinct values of {by!r}: too many groups to summarise. Group by a "
            "category or a coarser column (e.g. a rounded copy made with create_dataset)."
        )
    stat = o["stat"]
    functions = {
        "mean": np.mean,
        "sum": np.sum,
        "median": np.median,
        "min": np.min,
        "max": np.max,
        "std": np.std,
    }
    results = np.full(groups.size, np.nan)
    counts = np.zeros(groups.size)
    for i, group in enumerate(groups):
        members = ctx.y[keys == group]
        members = members[np.isfinite(members)]
        counts[i] = members.size
        if stat == "count":
            results[i] = members.size
        elif members.size:
            results[i] = functions[stat](members)
    label = "".join(c if c.isalnum() or c == "_" else "_" for c in ctx.y_name).strip("_") or "y"
    value_name = f"{label}_{stat}"
    labels = ctx.categories.get(by)
    report_rows = []
    for key, value, n in list(zip(groups, results, counts))[:20]:
        if labels and 0 <= int(key) < len(labels):
            name: Any = labels[int(key)]
        else:
            name = int(key) if float(key).is_integer() else _round(key)
        report_rows.append({by: name, stat: _round(value), "n": int(n)})
    table = {by: groups, value_name: results, "count": counts}
    top = np.nanargmax(results) if np.isfinite(results).any() else None
    headline = f"{stat} of {ctx.y_name} by {by}: {groups.size} groups" + (
        f"; largest {report_rows[int(top)][by]} ({report_rows[int(top)][stat]})"
        if top is not None and top < len(report_rows)
        else ""
    )
    return Outcome(
        table=table,
        report={"groups": report_rows, "stat": stat, "by": by},
        headline=headline,
        plot=("bar", by, value_name),
        categories={by: list(labels)} if labels else {},
    )


_WINDOW = Option(
    "window", "int", "Window length in samples (made odd).", default=11, minimum=1, maximum=100001
)
_POLYORDER = Option(
    "polyorder",
    "int",
    "Polynomial order for savgol (below the window).",
    default=3,
    minimum=1,
    maximum=10,
)

OPERATIONS: Tuple[Operation, ...] = (
    Operation(
        "describe",
        "Summary statistics of y: mean, std, median, percentiles, skewness, kurtosis, outliers.",
        "report",
        _run_describe,
        uses_x=False,
        example='analyze_dataset("run", "describe", y="v")',
    ),
    Operation(
        "fit",
        "Least-squares fit of a model to y(x); adds the fitted curve as a column and reports "
        "parameters with standard errors and R^2.",
        "column",
        _run_fit,
        options=(
            Option(
                "model",
                "choice",
                "Model family. 'custom' takes options.expression; 'multipeak' sums several peaks.",
                default="linear",
                choices=_FIT_MODELS,
            ),
            Option(
                "degree", "int", "Polynomial degree (model=polynomial).", 2, minimum=1, maximum=12
            ),
            Option(
                "expression",
                "text",
                "model=custom: a function of x such as 'a*exp(-x/b)+c'; every other name is a "
                "fitted parameter.",
                default=None,
            ),
            Option("peaks", "int", "model=multipeak: number of peaks.", 2, minimum=1, maximum=10),
            Option(
                "shape",
                "choice",
                "model=multipeak: peak shape.",
                default="gaussian",
                choices=("gaussian", "lorentzian"),
            ),
            Option(
                "p0",
                "floats",
                "Starting values, one per parameter, when the automatic guess does not converge.",
                default=None,
            ),
            Option("robust", "bool", "Down-weight outliers (soft_l1 loss).", default=False),
        ),
        suffix="fit",
        example='analyze_dataset("run", "fit", x="t", y="v", options={"model": "exp_decay"})',
    ),
    Operation(
        "smooth",
        "Smooth y; adds a same-length column.",
        "column",
        _run_smooth,
        uses_x=False,
        options=(
            Option(
                "method",
                "choice",
                "Smoothing method (ema is causal; the others are centred).",
                default="moving_average",
                choices=("moving_average", "gaussian", "savgol", "median", "ema"),
            ),
            _WINDOW,
            _POLYORDER,
            Option(
                "sigma",
                "float",
                "gaussian: kernel width in samples (default window/6).",
                minimum=0.01,
            ),
            Option(
                "alpha",
                "float",
                "ema: smoothing factor 0-1 (default 2/(window+1)).",
                minimum=0.0001,
                maximum=1,
            ),
        ),
        suffix="smooth",
        example=(
            'analyze_dataset("run", "smooth", y="v", options={"method": "savgol", "window": 31})'
        ),
    ),
    Operation(
        "rolling",
        "A moving-window statistic of y (mean, std, median, min, max).",
        "column",
        _run_rolling,
        uses_x=False,
        options=(
            Option(
                "window", "int", "Window length in samples.", default=11, minimum=1, maximum=100001
            ),
            Option(
                "stat",
                "choice",
                "Statistic.",
                default="mean",
                choices=("mean", "std", "median", "min", "max"),
            ),
            Option("center", "bool", "Centre the window on each sample.", default=True),
        ),
        example='analyze_dataset("run", "rolling", y="v", options={"window": 25, "stat": "std"})',
    ),
    Operation(
        "derivative",
        "dy/dx (or d2y/dx2); adds a column.",
        "column",
        _run_derivative,
        options=(
            Option("order", "int", "1 or 2.", default=1, minimum=1, maximum=2),
            Option(
                "method",
                "choice",
                "central differences or a Savitzky-Golay derivative.",
                default="central",
                choices=("central", "savgol"),
            ),
            _WINDOW,
            _POLYORDER,
        ),
        example='analyze_dataset("run", "derivative", x="t", y="v")',
    ),
    Operation(
        "integrate",
        "Cumulative integral of y over x; adds a column and reports the total.",
        "column",
        _run_integrate,
        options=(
            Option(
                "method",
                "choice",
                "Quadrature rule.",
                default="trapezoid",
                choices=("trapezoid", "simpson", "rectangle"),
            ),
        ),
        suffix="integral",
        example='analyze_dataset("run", "integrate", x="t", y="v")',
    ),
    Operation(
        "fft",
        "One-sided amplitude spectrum; creates a table (frequency, amplitude) and reports the "
        "dominant frequency.",
        "table",
        _run_fft,
        example='analyze_dataset("signal", "fft", x="t", y="v")',
    ),
    Operation(
        "peaks",
        "Local maxima of y(x) with height, prominence, width and area; creates a table.",
        "table",
        _run_peaks,
        options=(
            Option("height", "float", "Minimum peak height."),
            Option(
                "prominence",
                "float",
                "Minimum prominence (how far a peak stands above its surroundings).",
            ),
            Option("distance", "int", "Minimum separation between peaks, in samples.", minimum=1),
            Option("width", "float", "Minimum width, in x units."),
        ),
        example=(
            'analyze_dataset("spectrum", "peaks", x="nm", y="counts", options={"prominence": 50})'
        ),
    ),
    Operation(
        "filter",
        "Zero-phase IIR filter (low/high/band-pass/stop); adds a column. Cut-offs are in "
        "cycles per unit of x.",
        "column",
        _run_filter,
        options=(
            Option(
                "cutoff",
                "floats",
                "One cut-off (low/high-pass) or two [low, high] (band).",
                required=True,
            ),
            Option(
                "btype",
                "choice",
                "Filter type.",
                default="lowpass",
                choices=("lowpass", "highpass", "bandpass", "bandstop"),
            ),
            Option(
                "family",
                "choice",
                "Filter family.",
                default="butter",
                choices=("butter", "cheby1", "cheby2", "bessel"),
            ),
            Option("order", "int", "Filter order.", default=4, minimum=1, maximum=12),
        ),
        suffix="filtered",
        example='analyze_dataset("signal", "filter", x="t", y="v", options={"cutoff": 5})',
    ),
    Operation(
        "detrend",
        "Remove a linear or constant trend from y; adds a column.",
        "column",
        _run_detrend,
        options=(
            Option(
                "kind",
                "choice",
                "What to remove.",
                default="linear",
                choices=("linear", "constant"),
            ),
        ),
        suffix="detrended",
        example='analyze_dataset("run", "detrend", x="t", y="v")',
    ),
    Operation(
        "normalize",
        "Rescale y (to 0-1, to z-scores, by its max, or to unit area); adds a column.",
        "column",
        _run_normalize,
        uses_x=False,
        options=(
            Option(
                "mode",
                "choice",
                "Scaling.",
                default="minmax",
                choices=("minmax", "zscore", "max_abs", "area"),
            ),
        ),
        suffix="norm",
        example='analyze_dataset("run", "normalize", y="v", options={"mode": "zscore"})',
    ),
    Operation(
        "resample",
        "Resample y(x) onto n evenly spaced x values; creates a new table.",
        "table",
        _run_resample,
        options=(
            Option("n", "int", "Number of output points.", required=True, minimum=2),
            Option(
                "kind", "choice", "Interpolation.", default="linear", choices=("linear", "cubic")
            ),
        ),
        example='analyze_dataset("run", "resample", x="t", y="v", options={"n": 1000})',
    ),
    Operation(
        "histogram",
        "Histogram of y's values; creates a table (bin_center, bin_left, bin_right, "
        "count|density).",
        "table",
        _run_histogram,
        uses_x=False,
        options=(
            Option(
                "bins", "int", "Number of equal-width bins.", default=50, minimum=1, maximum=10000
            ),
            Option("density", "bool", "Normalise to a probability density.", default=False),
        ),
        example='analyze_dataset("run", "histogram", y="v", options={"bins": 30})',
    ),
    Operation(
        "correlate",
        "Linear relationship between two columns: Pearson r, Spearman rho, slope, intercept, R^2.",
        "report",
        _run_correlate,
        x_required=True,
        example='analyze_dataset("run", "correlate", x="dose", y="response")',
    ),
    Operation(
        "aggregate",
        "Group the rows by another column (a category such as a city, or any numeric column) and "
        "summarise y in each group; creates a table (the group column, the statistic, count).",
        "table",
        _run_aggregate,
        uses_x=False,
        index_note=False,
        options=(
            Option(
                "by", "text", "Column to group by (categories keep their labels).", required=True
            ),
            Option(
                "stat",
                "choice",
                "What to compute in each group.",
                default="mean",
                choices=_AGGREGATES,
            ),
        ),
        example='analyze_dataset("weather", "aggregate", y="temp", options={"by": "city"})',
    ),
    Operation(
        "autocorrelation",
        "Normalised autocorrelation of y versus lag; creates a table.",
        "table",
        _run_autocorrelation,
        index_note=False,
        options=(
            Option(
                "max_lag",
                "int",
                "Largest lag in samples (default: about half the data).",
                minimum=1,
            ),
            Option("detrend", "bool", "Remove the mean first.", default=True),
        ),
        example='analyze_dataset("signal", "autocorrelation", y="v", options={"max_lag": 200})',
    ),
)

_BY_NAME = {op.name: op for op in OPERATIONS}


def operation(name: str) -> Operation:
    """The operation called ``name``; an unknown one lists the valid names and the near miss."""
    key = str(name).strip().lower().replace("-", "_")
    if key in _BY_NAME:
        return _BY_NAME[key]
    raise McpError(
        f"Unknown operation {name!r}.{suggest(key, _BY_NAME)} "
        f"Available: {', '.join(_BY_NAME)}. See get_guide('mcp-analysis')."
    )


# ----------------------------------------------------------------------------------
# Generated documentation
# ----------------------------------------------------------------------------------


def operations_summary() -> str:
    """One line per operation, for the tool description."""
    return "; ".join(f"{op.name} ({op.produces})" for op in OPERATIONS)


def operations_guide() -> str:
    """The markdown reference for ``analyze_dataset``, generated from :data:`OPERATIONS`."""
    lines = [
        "`analyze_dataset(dataset, operation, y=..., x=..., options={...}, output=...)` runs a "
        "numerical",
        "routine on columns of a dataset and keeps the result so it can be plotted:",
        "",
        "* **column** operations add a same-length column to the dataset (named `<y>_<suffix>` "
        "unless you give `output`) -- plot it next to the data with `add_plot`;",
        "* **table** operations create a *new* dataset (named `<dataset>_<operation>` unless you "
        "give `output`) because the result has a different length;",
        "* **report** operations only return numbers.",
        "",
        "`x` defaults to the row index (unit spacing) when an operation needs one and none is "
        "given; `y` defaults to the only other column. Unknown or out-of-range options are refused "
        "with the valid ones named. The input is capped (see `server_info`); `resample` first for "
        "bigger data.",
        "",
    ]
    for op in OPERATIONS:
        lines.append(f"### {op.name} -- {op.produces}")
        lines.append("")
        lines.append(op.summary)
        reads = "y and x" if op.uses_x else "y"
        lines.append(f"Reads {reads}" + (" (x required)." if op.x_required else "."))
        if op.produces == "column":
            lines.append(f"Adds the column `<y>_{op.suffix or op.name}`.")
        if op.options:
            lines.append("")
            lines.extend(f"* {option.describe()}" for option in op.options)
        if op.example:
            lines.append("")
            lines.append(f"    {op.example}")
        lines.append("")
    lines.append(
        "A typical sequence: `create_dataset` or `load_dataset` -> "
        "`analyze_dataset(... 'fit' ...)` -> `add_plot` the data, `add_plot` the fitted column "
        "on the same figure -> "
        "`render_figure`. The reply includes a `plot_hint` with the call to draw the result."
    )
    return "\n".join(lines).rstrip() + "\n"
