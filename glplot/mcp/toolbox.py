"""The MCP tools as plain methods. Pure logic: no MCP SDK, no GL window.

:class:`Toolbox` is everything a language model can *do* to GLPlot -- create figures,
build datasets, add plots, restyle, render -- written as ordinary Python methods that take
JSON-shaped arguments and return a :class:`ToolResult`. The SDK adapter in
:mod:`glplot.mcp.server` only describes these methods to the protocol; it adds no behaviour.
That split is what lets the whole surface be tested without an MCP client, and what keeps
this module importable on a machine that has not installed the optional ``mcp`` extra.

**The model drives a headless figure.** Nothing here opens a window: figures are real
:class:`~glplot.engine.GPULinePlot` objects built through :mod:`glplot.pyplot` (so a script
and a tool call build the same scene), and a picture is the matplotlib-based preview export
that ``savefig`` already uses when no GL context exists. That is the one renderer that runs
unchanged on a laptop, a CI runner and a container with no display.

**State.** pyplot keeps a process-global "current figure", matplotlib is not thread-safe,
and the SDK runs synchronous tools on worker threads. Every public method therefore runs
under one re-entrant lock and re-selects its figure (and panel) on entry; nothing depends on
what the previous call left current.

**Handles, not layers.** One ``add_plot`` call can create many scene layers (a bar chart is
one patch per bar; ``errorbar`` is a dozen polylines). The model is handed an *artist id*
per call instead, and removal takes the whole group back out.
"""

from __future__ import annotations

import contextlib
import functools
import json
import math
import sys
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .. import pyplot as gplt
from ..core.camera3d import SYSTEM_3D_ARTISTS
from ..gui import dataio, layerops, recipes, styles
from ..gui.datasets import Column, DataSet
from ..gui.expressions import ExpressionError, evaluate_2d
from ..pyplot import MatplotlibCompatWarning
from ..utils.preview import render_preview_bytes
from .state import (
    DEFAULT_LIMITS,
    DatasetStore,
    FileSandbox,
    Limits,
    McpError,
    as_float_array,
    check_name,
    evaluate_expression,
    guarded_constructors,
    is_identifier,
    slug,
    suggest,
)

__all__ = ["Toolbox", "ToolResult"]

SeriesRef = Union[str, Sequence[Optional[float]], None]

#: pyplot figure numbers start here, clear of the small integers a script run in the same
#: process (a test, a notebook) would pick for its own figures.
_FIGURE_NUM_BASE = 9000

#: ``add_plot`` kinds, in the order the model is shown them.
PLOT_KINDS: Tuple[str, ...] = (
    "line",
    "scatter",
    "step",
    "stem",
    "bar",
    "barh",
    "area",
    "errorbar",
    "hist",
    "box",
    "violin",
    "pie",
    "hist2d",
    "hexbin",
)
#: ``add_field`` kinds. The last two draw a 3-D surface and need a ``projection="3d"`` figure.
FIELD_KINDS: Tuple[str, ...] = (
    "image",
    "contour",
    "contourf",
    "pcolormesh",
    "surface",
    "wireframe",
)
ANNOTATION_KINDS: Tuple[str, ...] = ("text", "hline", "vline")

#: What pyplot raises for a mistake in the arguments or the state it was handed (a bad colour,
#: mismatched lengths, ``colorbar()`` with nothing to attach to). Each message already says what
#: was wrong, so it is passed to the model as-is; anything else is a bug and is left to propagate.
_USER_ERRORS = (ValueError, TypeError, KeyError, IndexError, RuntimeError, ExpressionError)
#: Kinds drawn in black (or one fixed blue) unless told otherwise; they take the next colour
#: from the figure's cycle instead, as ``plot()`` does.
_CYCLED_KINDS = {"scatter", "errorbar", "bar", "barh", "area", "hist"}
#: Kinds that read *one* series of values (from ``y``, else ``x``) rather than ``x`` vs ``y``.
_SINGLE_SERIES = {"hist", "box", "violin", "pie"}
#: Kinds that cannot be drawn without a ``y`` ... all the x-vs-y ones.
_XY_KINDS = set(PLOT_KINDS) - _SINGLE_SERIES


@dataclass
class ToolResult:
    """What a tool hands back: a sentence, structured data, and maybe a picture."""

    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    image: Optional[bytes] = None
    image_format: str = "png"

    def text(self) -> str:
        """The message, then the data as one compact JSON line the model can parse."""
        if not self.data:
            return self.message
        payload = json.dumps(_finite(self.data), separators=(",", ":"), default=_json_default)
        return f"{self.message}\n{payload}"


def _finite(value: Any) -> Any:
    """``value`` with every ``nan``/``inf`` replaced by ``None``, recursively.

    ``json.dumps`` writes them as the bare words ``NaN``/``Infinity``, which are not JSON and
    which a strict client rejects -- losing the whole reply for one empty statistic.
    """
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _round(value: Any, digits: int = 6) -> Any:
    """Round a float for a compact, JSON-safe report; ``nan``/``inf`` become ``None``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(number):
        return None
    if number == 0.0:
        return 0.0
    return float(f"{number:.{digits}g}")


@dataclass
class _Artist:
    """One ``add_*`` call's worth of scene layers, addressed by a single id."""

    artist_id: str
    kind: str
    panel: int
    label: str
    layer_ids: List[int]
    points: Optional[int] = None
    dataset: Optional[str] = None

    def describe(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "artist_id": self.artist_id,
            "kind": self.kind,
            "panel": self.panel,
            "layers": len(self.layer_ids),
        }
        if self.label:
            out["label"] = self.label
        if self.points is not None:
            out["points"] = self.points
        if self.dataset:
            out["dataset"] = self.dataset
        return out


@dataclass
class _Figure:
    figure_id: str
    plot: Any
    rows: int
    cols: int
    projection: str
    width: int
    height: int
    artists: List[_Artist] = field(default_factory=list)
    next_artist: int = 1
    #: The integer key this figure is registered under in :mod:`glplot.pyplot`.
    num: int = 0
    #: Panels (1-based) whose axes render with equal x/y scaling. The headless preview does
    #: not honour ``axis("equal")`` (the pie below would come out an ellipse), so the export
    #: applies it to the matplotlib axes itself; this is what it consults.
    equal_panels: set = field(default_factory=set)

    @property
    def panels(self) -> int:
        return self.rows * self.cols

    @property
    def is_3d(self) -> bool:
        return self.projection == "3d"


def _tool(method: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
    """Serialise a tool, report its warnings, and turn pyplot's complaints into McpErrors.

    pyplot raises ``ValueError``/``TypeError`` for exactly the mistakes a model makes (a bad
    colour, mismatched lengths, an unknown colormap). Those become :class:`McpError` with the
    original text, which already says what was wrong. Anything else is a bug and propagates.
    ``warnings`` raised during the call -- GLPlot warns when an argument "has no effect" --
    are collected into the result so the model learns it, instead of them landing in a log
    nobody reads.
    """

    @functools.wraps(method)
    def wrapper(self: "Toolbox", *args: Any, **kwargs: Any) -> ToolResult:
        with self._lock, contextlib.redirect_stdout(sys.stderr):
            # stdout is the protocol channel under the stdio transport: one stray print() from
            # a library below would be read by the client as a malformed message.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    result = method(self, *args, **kwargs)
                except McpError:
                    raise
                except _USER_ERRORS as exc:
                    raise McpError(f"{method.__name__}: {exc}") from exc
            notes = sorted(
                {str(w.message) for w in caught if issubclass(w.category, MatplotlibCompatWarning)}
            )
            if notes:
                result.data.setdefault("warnings", notes)
            return result

    return wrapper


class Toolbox:
    """Figures and datasets for one MCP session, and the operations on them."""

    def __init__(
        self,
        sandbox: Optional[FileSandbox] = None,
        limits: Limits = DEFAULT_LIMITS,
    ) -> None:
        self.limits = limits
        self.sandbox = sandbox if sandbox is not None else FileSandbox(())
        self.datasets = DatasetStore(limits)
        self._figures: Dict[str, _Figure] = {}
        self._next_figure = 1
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    @_tool
    def create_figure(
        self,
        title: Optional[str] = None,
        width: int = 1000,
        height: int = 640,
        rows: int = 1,
        cols: int = 1,
        projection: str = "2d",
        style: Optional[str] = None,
    ) -> ToolResult:
        """Create an empty figure (a grid of ``rows`` x ``cols`` panels) and return its id."""
        lo, hi = self.limits.min_figure_px, self.limits.max_figure_px
        for name, value in (("width", width), ("height", height)):
            if not isinstance(value, int) or not lo <= value <= hi:
                raise McpError(f"{name} must be an integer between {lo} and {hi} pixels.")
        for name, value in (("rows", rows), ("cols", cols)):
            if not isinstance(value, int) or not 1 <= value <= 6:
                raise McpError(f"{name} must be an integer between 1 and 6.")
        if projection not in ("2d", "3d"):
            raise McpError("projection must be '2d' or '3d'.")
        if len(self._figures) >= self.limits.max_figures:
            raise McpError(
                f"Too many open figures ({self.limits.max_figures}). Close one with "
                f"close_figure: {', '.join(self._figures)}."
            )

        figure_id = f"fig{self._next_figure}"
        self._next_figure += 1
        three_d = projection == "3d"
        # An integer ``num`` and the stock window title: a string ``num`` doubles as the window
        # caption, and a non-stock caption is drawn as the plot's title -- every render would
        # carry "mcp-fig1" above the axes. The real title is set below, only if one was asked.
        num = _FIGURE_NUM_BASE + int(figure_id[3:])
        projection_kw = {"projection": "3d"} if three_d else {}
        if rows * cols > 1:
            # ``subplots`` rather than a figure plus ``subplot`` calls: it is the one that
            # takes ``wspace``/``hspace``. Panels placed any other way sit edge to edge, so
            # one panel's tick numbers print over its neighbour's axis.
            plot, _ = gplt.subplots(
                rows,
                cols,
                squeeze=False,
                num=num,
                title="GLPlot",
                width=width,
                height=height,
                wspace=0.22,
                hspace=0.32,
                **projection_kw,
            )
        else:
            plot = gplt.figure(num=num, title="GLPlot", width=width, height=height, **projection_kw)
        entry = _Figure(figure_id, plot, rows, cols, projection, width, height, num=num)
        self._figures[figure_id] = entry
        try:
            if entry.panels > 1:
                self._activate(entry, 1)
            if style:
                self._apply_style(style)
            if title:
                (gplt.suptitle if entry.panels > 1 else gplt.title)(title)
        except Exception:
            self._discard(figure_id)
            raise
        return ToolResult(
            f"Created {figure_id} ({width}x{height}px, {rows}x{cols} panel"
            f"{'s' if entry.panels > 1 else ''}, {projection}).",
            {"figure_id": figure_id, "panels": entry.panels},
        )

    @_tool
    def close_figure(self, figure_id: str) -> ToolResult:
        """Close a figure and free its data."""
        self._figure(figure_id)
        self._discard(figure_id)
        return ToolResult(f"Closed {figure_id}.", {"open_figures": list(self._figures)})

    @_tool
    def describe_figure(self, figure_id: Optional[str] = None) -> ToolResult:
        """Describe one figure -- or, with no id, list them all -- without sending a picture."""
        if figure_id is None:
            listing = [
                {
                    "figure_id": f.figure_id,
                    "projection": f.projection,
                    "panels": f.panels,
                    "artists": len(f.artists),
                }
                for f in self._figures.values()
            ]
            return ToolResult(
                f"{len(listing)} open figure{'s' if len(listing) != 1 else ''}.",
                {"figures": listing},
            )
        entry = self._figure(figure_id)
        view = self._resolve_view(entry)
        data = {
            "figure_id": entry.figure_id,
            "size_px": [entry.width, entry.height],
            "projection": entry.projection,
            "grid": [entry.rows, entry.cols],
            "artists": [a.describe() for a in entry.artists],
            "panels": view,
        }
        return ToolResult(f"{entry.figure_id}: {len(entry.artists)} artists.", data)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    @_tool
    def create_dataset(
        self,
        name: str,
        columns: Mapping[str, Union[Sequence[Optional[float]], str]],
        base: Optional[str] = None,
    ) -> ToolResult:
        """Build a table from number lists and/or expressions over earlier columns."""
        check_name(name, "dataset name")
        if not isinstance(columns, Mapping) or not columns:
            raise McpError("columns must be an object like {'t': 'linspace(0, 10, 500)'}.")
        if len(columns) > 256:
            raise McpError("At most 256 columns per dataset.")

        built: Dict[str, np.ndarray] = {}
        length: Optional[int] = None
        if base is not None:
            source = self.datasets.get(base)
            length = source.n_rows()
            for column in source.columns:
                built[column.name] = np.array(column.values, dtype=np.float64)

        for column_name, spec in columns.items():
            if not isinstance(column_name, str) or not is_identifier(column_name):
                raise McpError(
                    f"Column name {column_name!r} must be an identifier (letters, digits and "
                    "'_', not starting with a digit) so later expressions can use it."
                )
            what = f"column {column_name!r}"
            if isinstance(spec, str):
                result = evaluate_expression(spec, built, limits=self.limits, what=what)
                values = as_float_array(result, what, length=length, limits=self.limits)
            elif isinstance(spec, (list, tuple)):
                values = as_float_array(
                    self._inline(spec, what), what, length=length, limits=self.limits
                )
            else:
                raise McpError(f"{what}: give a list of numbers or an expression string.")
            if length is None:
                length = values.size
            built[column_name] = values

        table = DataSet(name, [Column(key, value) for key, value in built.items()])
        table.source = "mcp"
        replaced = self.datasets.put(table)
        verb = "Replaced" if replaced else "Created"
        return ToolResult(
            f"{verb} dataset {name!r} ({table.n_rows()} rows x {table.n_cols()} columns).",
            self.datasets.describe(name, rows=3),
        )

    @_tool
    def load_dataset(self, path: str, name: Optional[str] = None) -> ToolResult:
        """Load a CSV/TSV/TXT, NPY/NPZ, JSON or image file from the sandbox as a dataset."""
        resolved = self.sandbox.read_path(path)
        dataset_name = check_name(name, "dataset name") if name else slug(resolved.stem)
        try:
            table = dataio.load_dataset(str(resolved), name=dataset_name)
        except dataio.DataIOError as exc:
            raise McpError(str(exc)) from exc
        table.name = dataset_name
        table.source = "mcp"
        replaced = self.datasets.put(table)
        verb = "Replaced" if replaced else "Loaded"
        return ToolResult(
            f"{verb} {resolved.name} as dataset {dataset_name!r} "
            f"({table.n_rows()} rows x {table.n_cols()} columns).",
            self.datasets.describe(dataset_name, rows=3),
        )

    @_tool
    def describe_dataset(self, name: Optional[str] = None, rows: int = 5) -> ToolResult:
        """Show a dataset's columns, statistics and first rows -- or list every dataset."""
        if name is None:
            listing = [
                {
                    "name": n,
                    "rows": self.datasets.get(n).n_rows(),
                    "columns": self.datasets.get(n).column_names(),
                }
                for n in self.datasets.names()
            ]
            return ToolResult(
                f"{len(listing)} dataset{'s' if len(listing) != 1 else ''}.", {"datasets": listing}
            )
        data = self.datasets.describe(name, rows)
        return ToolResult(f"Dataset {name!r}: {data['rows']} rows.", data)

    @_tool
    def delete_dataset(self, name: str) -> ToolResult:
        """Forget a dataset. Figures already drawn from it are unaffected."""
        self.datasets.remove(name)
        return ToolResult(f"Deleted dataset {name!r}.", {"datasets": self.datasets.names()})

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    @_tool
    def add_plot(
        self,
        figure_id: str,
        kind: str,
        dataset: Optional[str] = None,
        x: SeriesRef = None,
        y: SeriesRef = None,
        z: SeriesRef = None,
        error: SeriesRef = None,
        y2: SeriesRef = None,
        columns: Optional[Sequence[str]] = None,
        color: Optional[str] = None,
        color_by: SeriesRef = None,
        size_by: SeriesRef = None,
        cmap: Optional[str] = None,
        label: Optional[str] = None,
        alpha: Optional[float] = None,
        linewidth: Optional[float] = None,
        linestyle: Optional[str] = None,
        marker: Optional[str] = None,
        size: Optional[float] = None,
        bins: Optional[int] = None,
        width: Optional[float] = None,
        tick_labels: Optional[Sequence[str]] = None,
        colorbar: bool = False,
        panel: int = 1,
    ) -> ToolResult:
        """Add data to a figure as a line, scatter, bar, histogram, ... (see PLOT_KINDS)."""
        entry = self._figure(figure_id)
        if kind not in PLOT_KINDS:
            raise McpError(
                f"Unknown kind {kind!r}.{suggest(kind, PLOT_KINDS)} "
                f"Use one of: {', '.join(PLOT_KINDS)}."
            )
        self._check_projection_for_plot(entry, kind, z)

        req = _PlotRequest(kind=kind, dataset=dataset)
        if kind in _SINGLE_SERIES:
            series = y if y is not None else x
            if kind in ("box", "violin") and columns:
                if dataset is None:
                    raise McpError("columns names columns of a dataset, so `dataset` is required.")
                req.groups = [self.datasets.column(dataset, c) for c in columns]
                req.group_names = list(columns)
                req.n = max(len(g) for g in req.groups)
            else:
                req.y = self._series(series, dataset, "y (the values)", required=True)
                req.n = req.y.size
        else:
            req.y = self._series(y, dataset, "y", required=True)
            req.x = self._series(x, dataset, "x", required=False)
            if req.x is None:
                req.x = np.arange(req.y.size, dtype=np.float64)
            if req.x.size != req.y.size:
                raise McpError(
                    f"x has {req.x.size} values but y has {req.y.size}; they must match."
                )
            req.n = req.y.size
            if z is not None:
                req.z = self._series(z, dataset, "z", required=True)
                if req.z.size != req.n:
                    raise McpError(f"z has {req.z.size} values but x and y have {req.n}.")
            for key, ref in (
                ("error", error),
                ("y2", y2),
                ("color_by", color_by),
                ("size_by", size_by),
            ):
                if ref is not None:
                    values = self._series(ref, dataset, key, required=True)
                    if values.size != req.n and not (key == "error" and values.size == 1):
                        raise McpError(f"{key} has {values.size} values but x and y have {req.n}.")
                    setattr(req, key, values)
            if kind == "errorbar" and req.error is None:
                raise McpError(
                    "kind='errorbar' needs `error` (the +/- size, per point or a single number)."
                )
        if req.n == 0:
            raise McpError("There is no data to plot (0 values).")

        req.options = {
            "color": color,
            "alpha": alpha,
            "linewidth": linewidth,
            "linestyle": linestyle,
            "marker": marker,
            "size": size,
            "label": label,
            "cmap": cmap,
            "bins": bins,
            "width": width,
        }
        req.tick_labels = list(tick_labels) if tick_labels else None
        req.color_label = color_by if isinstance(color_by, str) else None
        req.colorbar = bool(colorbar)

        self._activate(entry, panel)
        if color is None and req.color_by is None and kind in _CYCLED_KINDS:
            # GLPlot draws a bare scatter or errorbar in black; matplotlib walks its colour
            # cycle. A model overlaying two series needs them told apart without having to
            # name a colour for each, so take the figure's next cycle colour (the one
            # ``plot()`` is already advancing) rather than the black default.
            req.options["color"] = gplt._next_cycle_color(entry.plot)
        if kind == "area" and alpha is None:
            # A translucent fill is exported as one outlined quad per sample, and the
            # overlapping outlines show as vertical hatching. Opaque is seamless.
            req.options["alpha"] = 1.0
        artist = self._record(entry, panel, kind, label or "", lambda: _DRAW[kind](req), req)
        if kind == "pie":
            entry.equal_panels.add(panel)
        artist.dataset = dataset
        note = f" Ignored for kind={kind!r}: {', '.join(req.ignored)}." if req.ignored else ""
        return ToolResult(
            f"Added {kind} {artist.artist_id} ({req.n} points) to {figure_id}.{note}",
            (
                {
                    "artist_id": artist.artist_id,
                    "layers": len(artist.layer_ids),
                    "ignored": req.ignored,
                }
                if req.ignored
                else {"artist_id": artist.artist_id, "layers": len(artist.layer_ids)}
            ),
        )

    @_tool
    def add_function(
        self,
        figure_id: str,
        expression: str,
        x_min: float,
        x_max: float,
        samples: int = 500,
        parameters: Optional[Mapping[str, float]] = None,
        kind: str = "line",
        label: Optional[str] = None,
        color: Optional[str] = None,
        alpha: Optional[float] = None,
        linewidth: Optional[float] = None,
        linestyle: Optional[str] = None,
        panel: int = 1,
    ) -> ToolResult:
        """Plot ``y = expression(x)`` over ``[x_min, x_max]`` without building a dataset."""
        if kind not in ("line", "area", "scatter", "step"):
            raise McpError("kind must be one of: line, area, scatter, step.")
        if not (math.isfinite(x_min) and math.isfinite(x_max)) or x_min >= x_max:
            raise McpError("x_min and x_max must be finite numbers with x_min < x_max.")
        if not isinstance(samples, int) or not 2 <= samples <= self.limits.max_array_elements:
            raise McpError(
                f"samples must be an integer between 2 and {self.limits.max_array_elements}."
            )
        variables: Dict[str, Any] = {}
        for key, value in (parameters or {}).items():
            if not is_identifier(key) or key == "x":
                raise McpError(f"Parameter name {key!r} must be an identifier other than 'x'.")
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise McpError(f"Parameter {key!r} must be a finite number.")
            variables[key] = float(value)
        grid = np.linspace(float(x_min), float(x_max), samples)
        variables["x"] = grid
        result = evaluate_expression(expression, variables, limits=self.limits, what="expression")
        values = as_float_array(
            result, f"expression {expression!r}", length=samples, limits=self.limits
        )
        outcome = self.add_plot(
            figure_id,
            kind,
            x=grid,
            y=values,
            label=label if label is not None else expression,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            panel=panel,
        )
        outcome.message = (
            f"Plotted y = {expression} on [{x_min:g}, {x_max:g}] as {outcome.data['artist_id']}."
        )
        return outcome

    @_tool
    def add_field(
        self,
        figure_id: str,
        kind: str = "image",
        expression: Optional[str] = None,
        matrix: Optional[Sequence[Sequence[float]]] = None,
        file: Optional[str] = None,
        x_range: Optional[Sequence[float]] = None,
        y_range: Optional[Sequence[float]] = None,
        resolution: Optional[int] = None,
        parameters: Optional[Mapping[str, float]] = None,
        cmap: Optional[str] = None,
        levels: Optional[int] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        alpha: Optional[float] = None,
        label: Optional[str] = None,
        colorbar: bool = False,
        panel: int = 1,
    ) -> ToolResult:
        """Draw a 2-D scalar field ``z(x, y)`` as an image, contours or a 3-D surface."""
        entry = self._figure(figure_id)
        if kind not in FIELD_KINDS:
            raise McpError(
                f"Unknown kind {kind!r}.{suggest(kind, FIELD_KINDS)} "
                f"Use one of: {', '.join(FIELD_KINDS)}."
            )
        surface = kind in ("surface", "wireframe")
        if surface and not entry.is_3d:
            raise McpError(f"kind={kind!r} needs a 3-D figure; create one with projection='3d'.")
        if not surface and entry.is_3d:
            raise McpError(
                f"kind={kind!r} is 2-D but {figure_id} is a 3-D figure. Use kind='surface' or "
                "'wireframe', or create a 2-D figure."
            )
        sources = [
            name
            for name, value in (("expression", expression), ("matrix", matrix), ("file", file))
            if value is not None
        ]
        if len(sources) != 1:
            raise McpError("Give exactly one of expression, matrix or file.")
        x0, x1 = self._range(x_range if x_range is not None else (0.0, 1.0), "x_range")
        y0, y1 = self._range(y_range if y_range is not None else (0.0, 1.0), "y_range")

        rgb: Optional[np.ndarray] = None
        if expression is not None:
            n = int(resolution or (80 if surface else 200))
            biggest = int(self.limits.max_array_elements**0.5)
            if not 2 <= n <= biggest:
                raise McpError(
                    f"resolution must be between 2 and {biggest} (cells per side), got {n}."
                )
            xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
            variables: Dict[str, Any] = dict(guarded_constructors(self.limits.max_array_elements))
            for key, value in (parameters or {}).items():
                if not is_identifier(key) or key in ("x", "y"):
                    raise McpError(
                        f"Parameter name {key!r} must be an identifier other than 'x'/'y'."
                    )
                variables[key] = float(value)
            if len(expression) > self.limits.max_expression_chars:
                raise McpError(
                    f"expression is longer than {self.limits.max_expression_chars} characters."
                )
            try:
                z_grid = evaluate_2d(expression, xs, ys, variables=variables)
            except ExpressionError as exc:
                raise McpError(f"Could not evaluate expression {expression!r}: {exc}") from exc
        elif matrix is not None:
            z_grid = self._matrix(matrix, "matrix")
            xs, ys = np.linspace(x0, x1, z_grid.shape[1]), np.linspace(y0, y1, z_grid.shape[0])
        else:
            z_grid, rgb = self._field_file(file)
            shape = (rgb if rgb is not None else z_grid).shape
            xs, ys = np.linspace(x0, x1, shape[1]), np.linspace(y0, y1, shape[0])
        if rgb is not None and kind != "image":
            raise McpError("A colour image file can only be drawn with kind='image'.")
        if z_grid is not None and z_grid.size > self.limits.max_array_elements:
            raise McpError(
                f"The field has {z_grid.size} cells; the limit is {self.limits.max_array_elements}."
            )

        grid_x, grid_y = np.meshgrid(xs, ys)
        self._activate(entry, panel)
        options = {"cmap": cmap, "alpha": alpha, "label": label, "vmin": vmin, "vmax": vmax}

        # What each pyplot function accepts. An option the kind cannot use is dropped, not
        # an error: a model that sends cmap with a wireframe should still get its wireframe.
        accepted = {
            "image": ("cmap", "alpha", "label", "vmin", "vmax"),
            "contour": ("cmap", "alpha", "label"),
            "contourf": ("cmap", "alpha"),
            "pcolormesh": ("cmap", "alpha", "vmin", "vmax"),
            "surface": ("cmap", "alpha", "label", "vmin", "vmax"),
            "wireframe": ("label",),
        }[kind]
        options = {
            k: v
            for k, v in (
                ("cmap", cmap),
                ("alpha", alpha),
                ("label", label),
                ("vmin", vmin),
                ("vmax", vmax),
            )
            if v is not None and k in accepted
        }
        dropped = [
            k
            for k, v in (
                ("cmap", cmap),
                ("alpha", alpha),
                ("label", label),
                ("vmin", vmin),
                ("vmax", vmax),
            )
            if v is not None and k not in accepted
        ]

        def draw() -> Any:
            if kind == "image":
                if rgb is not None:
                    return gplt.imshow(rgb, **options)
                extent = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))
                return gplt.imshow(z_grid, origin="lower", extent=extent, aspect="auto", **options)
            if kind in ("contour", "contourf"):
                fn = gplt.contour if kind == "contour" else gplt.contourf
                return fn(grid_x, grid_y, z_grid, levels=int(levels or 10), **options)
            if kind == "pcolormesh":
                return gplt.pcolormesh(grid_x, grid_y, z_grid, **options)
            fn = gplt.plot_surface if kind == "surface" else gplt.plot_wireframe
            return fn(grid_x, grid_y, z_grid, **options)

        artist = self._record(entry, panel, kind, label or "", draw, None)
        shown = rgb if rgb is not None else z_grid
        artist.points = int(shown.shape[0] * shown.shape[1])
        note_extra = ""
        if colorbar:
            try:
                gplt.colorbar()
            except _USER_ERRORS as exc:
                # The plot itself was added; losing the bar must not lose the call.
                note_extra = f" Colorbar not added: {exc}"
        data: Dict[str, Any] = {"artist_id": artist.artist_id, "layers": len(artist.layer_ids)}
        if dropped:
            data["ignored"] = dropped
        return ToolResult(
            f"Added {kind} {artist.artist_id} ({xs.size}x{ys.size} cells) to {figure_id}."
            + (f" Ignored for kind={kind!r}: {', '.join(dropped)}." if dropped else "")
            + note_extra,
            data,
        )

    @_tool
    def list_recipes(self, category: Optional[str] = None) -> ToolResult:
        """List the ready-made recipes: what each draws, the columns it reads, its default look."""
        if category is not None and category not in recipes.CATEGORIES:
            raise McpError(
                f"Unknown category {category!r}.{suggest(category, recipes.CATEGORIES)} "
                f"Categories: {', '.join(recipes.CATEGORIES)}."
            )
        rows = [_describe_recipe(r) for r in recipes.RECIPES if category in (None, r.category)]
        scope = f" in {category}" if category else ""
        return ToolResult(f"{len(rows)} recipes{scope}.", {"recipes": rows})

    @_tool
    def add_recipe(
        self,
        figure_id: str,
        recipe: str,
        dataset: Optional[str] = None,
        columns: Optional[Mapping[str, Optional[str]]] = None,
        sample: bool = False,
        look: str = recipes.RECIPE_LOOK,
        title: Optional[str] = None,
        label: Optional[str] = None,
        panel: int = 1,
    ) -> ToolResult:
        """Draw a ready-made recipe (a plot kind plus the look it was designed in) from a dataset."""  # noqa: E501
        entry = self._figure(figure_id)
        spec = self._recipe(recipe)
        if spec.is_3d != entry.is_3d:
            have = "3-D" if entry.is_3d else "2-D"
            need = "projection='3d'" if spec.is_3d else "the default 2-D projection"
            raise McpError(
                f"{spec.key!r} is a {spec.dimension} recipe but {figure_id} is a {have} figure. "
                f"Create a new figure with {need} for it."
            )
        if sample and dataset is not None:
            raise McpError("Give either dataset or sample=true, not both.")
        if not sample and dataset is None:
            raise McpError(
                "add_recipe needs data: pass dataset=<name> (from create_dataset or load_dataset), "
                f"or sample=true to use the recipe's own example table. {_slot_help(spec)}"
            )
        if look not in (recipes.RECIPE_LOOK, recipes.KEEP_LOOK) and look not in styles.STYLE_KEYS:
            options = (recipes.RECIPE_LOOK, recipes.KEEP_LOOK) + styles.STYLE_KEYS
            raise McpError(
                f"Unknown look {look!r}.{suggest(look, options)} Use 'recipe' (the recipe's own), "
                f"'keep' (leave the figure's current look) or a style key: "
                f"{', '.join(styles.STYLE_KEYS)}."
            )

        if sample:
            if spec.sample is None:
                raise McpError(f"Recipe {spec.key!r} has no sample data.")
            name = self.datasets.unique_name(spec.key)
            self.datasets.check_room(name)
            table = recipes.sample_table(spec)
            source = DataSet(name, [Column(key, values) for key, values in table.items()])
            source.source = "mcp"
            assignment: Dict[str, Optional[str]] = dict(recipes.sample_assignment(spec))
        else:
            source = self.datasets.get(dataset)
            assignment = dict(recipes.assign_columns(spec, source.column_names()))
        self._apply_overrides(spec, source, assignment, columns)

        table = {column.name: column.values for column in source.columns}
        data = {
            key: np.array(values, copy=True)
            for key, values in recipes.slot_arrays(spec, table, assignment, source.n_rows()).items()
        }
        problem = recipes.validate_data(spec, data)
        if problem:
            raise McpError(f"{problem} {_slot_help(spec)} Columns here: {', '.join(table)}.")

        labels = recipes.slot_labels(spec, assignment)

        def draw() -> Any:
            created = recipes.apply_recipe(
                entry.plot,
                spec,
                data,
                name=source.name,
                look=look,
                labels=labels,
                title=title,
            )
            for layer in created:
                if label:
                    layer.label = label
                else:
                    # The layer is named after the recipe so the scene can list it; the export
                    # reads any explicit label as "put me in the legend", which would box the
                    # recipe's name over the plot. Bookkeeping labels stay out of it, and a
                    # `label` the caller gives is the way to ask for an entry.
                    layer.metadata["auto_label"] = True

        self._activate(entry, panel)
        artist = self._record(entry, panel, f"recipe:{spec.key}", label or "", draw, None)
        first = next(iter(data.values()))
        artist.points = int(first.size)
        artist.dataset = source.name
        if sample:
            self.datasets.put(source)

        used = {slot: column for slot, column in assignment.items() if column}
        if spec.kind == "image":
            used = {"matrix": f"all {source.n_cols()} columns"}
        applied = getattr(entry.plot, "_style_key", "") or ""
        note = f" Added sample dataset {source.name!r}." if sample else ""
        return ToolResult(
            f"Drew recipe {spec.key!r} as {artist.artist_id} on {figure_id} "
            f"({first.shape[0]} rows).{note}",
            {
                "artist_id": artist.artist_id,
                "layers": len(artist.layer_ids),
                "recipe": spec.key,
                "dataset": source.name,
                "columns": used,
                "look": applied,
            },
        )

    def _recipe(self, key: str) -> recipes.Recipe:
        try:
            return recipes.get_recipe(key)
        except KeyError:
            raise McpError(
                f"Unknown recipe {key!r}.{suggest(key, recipes.RECIPE_KEYS)} "
                "Call list_recipes to see them."
            ) from None

    def _apply_overrides(
        self,
        spec: recipes.Recipe,
        source: DataSet,
        assignment: Dict[str, Optional[str]],
        columns: Optional[Mapping[str, Optional[str]]],
    ) -> None:
        """Fold the caller's ``columns`` map into the guessed assignment, refusing nonsense."""
        if not columns:
            return
        if not isinstance(columns, Mapping):
            raise McpError(
                "columns must be an object mapping slot -> column, e.g. {'y': 'signal'}."
            )
        slots = {slot.key: slot for slot in spec.slots}
        names = source.column_names()
        for key, column in columns.items():
            if key not in slots:
                raise McpError(
                    f"Recipe {spec.key!r} has no slot {key!r}.{suggest(key, slots)} "
                    f"{_slot_help(spec)}"
                )
            if slots[key].multi:
                raise McpError(
                    f"Slot {key!r} of {spec.key!r} takes every column of the dataset; "
                    "it cannot be mapped to one."
                )
            if column in (None, ""):
                assignment[key] = None
            elif not isinstance(column, str):
                raise McpError(
                    f"columns[{key!r}] must be a column name, got {type(column).__name__}."
                )
            elif column == recipes.INDEX or column in names:
                assignment[key] = column
            else:
                raise McpError(
                    f"{source.name!r} has no column {column!r} (columns: {', '.join(names)})."
                    f"{suggest(column, names)} "
                    f"Use '{recipes.INDEX}' for the row number."
                )

    @_tool
    def add_annotation(
        self,
        figure_id: str,
        kind: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        text: Optional[str] = None,
        color: Optional[str] = None,
        linestyle: Optional[str] = None,
        linewidth: Optional[float] = None,
        fontsize: Optional[float] = None,
        label: Optional[str] = None,
        panel: int = 1,
    ) -> ToolResult:
        """Add a text note, a horizontal line (at ``y``) or a vertical line (at ``x``)."""
        entry = self._figure(figure_id)
        if kind not in ANNOTATION_KINDS:
            raise McpError(f"Unknown kind {kind!r}. Use one of: {', '.join(ANNOTATION_KINDS)}.")
        if entry.is_3d:
            raise McpError("Annotations are 2-D only; this is a 3-D figure.")
        self._activate(entry, panel)
        style = {
            k: v
            for k, v in (
                ("color", color),
                ("linestyle", linestyle),
                ("linewidth", linewidth),
                ("label", label),
            )
            if v is not None
        }

        if kind == "text":
            if x is None or y is None or not text:
                raise McpError("kind='text' needs x, y and text.")
            text_style = {
                k: v for k, v in (("color", color), ("fontsize", fontsize)) if v is not None
            }
            draw = lambda: gplt.text(float(x), float(y), str(text), **text_style)  # noqa: E731
        elif kind == "hline":
            if y is None:
                raise McpError("kind='hline' needs y.")
            draw = lambda: gplt.axhline(float(y), **style)  # noqa: E731
        else:
            if x is None:
                raise McpError("kind='vline' needs x.")
            draw = lambda: gplt.axvline(float(x), **style)  # noqa: E731

        artist = self._record(entry, panel, kind, label or (text or ""), draw, None)
        return ToolResult(
            f"Added {kind} {artist.artist_id} to {figure_id}.",
            {"artist_id": artist.artist_id, "layers": len(artist.layer_ids)},
        )

    @_tool
    def remove_artist(self, figure_id: str, artist_id: Optional[str] = None) -> ToolResult:
        """Remove one artist (everything one add_* call drew), or all of them if no id is given."""
        entry = self._figure(figure_id)
        targets = list(entry.artists)
        if artist_id is not None:
            targets = [a for a in entry.artists if a.artist_id == artist_id]
            if not targets:
                known = ", ".join(a.artist_id for a in entry.artists) or "none"
                raise McpError(f"{figure_id} has no artist {artist_id!r} (artists: {known}).")
        for artist in targets:
            self._activate(entry, artist.panel)
            scene = entry.plot.scene
            for layer in [l for l in scene.layers if l.layer_id in set(artist.layer_ids)]:
                layerops.remove_layer(entry.plot, None, layer)
            entry.artists.remove(artist)
        remaining = [a.artist_id for a in entry.artists]
        return ToolResult(
            f"Removed {len(targets)} artist{'s' if len(targets) != 1 else ''} from {figure_id}.",
            {"remaining": remaining},
        )

    # ------------------------------------------------------------------
    # Axes and style
    # ------------------------------------------------------------------

    @_tool
    def set_axes(
        self,
        figure_id: str,
        panel: int = 1,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        zlabel: Optional[str] = None,
        xlim: Optional[Sequence[float]] = None,
        ylim: Optional[Sequence[float]] = None,
        zlim: Optional[Sequence[float]] = None,
        xscale: Optional[str] = None,
        yscale: Optional[str] = None,
        grid: Optional[bool] = None,
        legend: Optional[bool] = None,
        equal_aspect: Optional[bool] = None,
        hide_axes: Optional[bool] = None,
        elev: Optional[float] = None,
        azim: Optional[float] = None,
    ) -> ToolResult:
        """Set titles, labels, limits, scales, grid, legend and (3-D) camera on one panel."""
        entry = self._figure(figure_id)
        self._activate(entry, panel)
        applied: List[str] = []

        def do(name: str, fn: Callable[..., Any], *args: Any) -> None:
            self._call(name, fn, *args)
            applied.append(name)

        if title is not None:
            do("title", gplt.title, title)
        if xlabel is not None:
            do("xlabel", gplt.xlabel, xlabel)
        if ylabel is not None:
            do("ylabel", gplt.ylabel, ylabel)
        if zlabel is not None:
            if not entry.is_3d:
                raise McpError("zlabel is only meaningful on a 3-D figure.")
            do("zlabel", gplt.zlabel, zlabel)
        for name, limits_, fn in (
            ("xlim", xlim, gplt.xlim),
            ("ylim", ylim, gplt.ylim),
            ("zlim", zlim, getattr(gplt, "set_zlim", None)),
        ):
            if limits_ is not None:
                lo, hi = self._range(limits_, name)
                if name == "zlim" and not entry.is_3d:
                    raise McpError("zlim is only meaningful on a 3-D figure.")
                do(name, fn, lo, hi)
        for name, value, fn in (("xscale", xscale, gplt.xscale), ("yscale", yscale, gplt.yscale)):
            if value is not None:
                if entry.panels > 1:
                    # GLPlot keeps one axis scale per *figure* (``engine.options``), and the
                    # export applies it to a single panel. Accepting it here would draw the
                    # log axis on a panel the model did not name, so say so instead.
                    raise McpError(
                        f"{name} is figure-wide in GLPlot and cannot be set on one panel of a "
                        f"{entry.rows}x{entry.cols} figure. Put the log-scaled plot in its own "
                        "single-panel figure."
                    )
                if value not in ("linear", "log", "symlog", "logit"):
                    raise McpError(f"{name} must be one of: linear, log, symlog, logit.")
                do(name, fn, value)
        if grid is not None:
            do("grid", gplt.grid, bool(grid))
        if legend:
            do("legend", gplt.legend)
        if equal_aspect is not None:
            do("equal_aspect", gplt.axis, "equal" if equal_aspect else "auto")
            (entry.equal_panels.add if equal_aspect else entry.equal_panels.discard)(panel)
        if hide_axes is not None:
            do("hide_axes", gplt.axis, "off" if hide_axes else "on")
        if elev is not None or azim is not None:
            if not entry.is_3d:
                raise McpError("elev/azim set the 3-D camera; this is a 2-D figure.")
            do("view", lambda: gplt.view_init(elev=elev, azim=azim))
        if not applied:
            raise McpError("Nothing to set: pass at least one of title, xlabel, ylabel, xlim, ...")
        return ToolResult(
            f"Updated panel {panel} of {figure_id}: {', '.join(applied)}.", {"applied": applied}
        )

    @_tool
    def set_style(self, figure_id: str, preset: str) -> ToolResult:
        """Apply a GLPlot style preset (``chalk``, ``journal``, ``clean``...) to the whole figure."""  # noqa: E501
        entry = self._figure(figure_id)
        self._activate(entry, 1)
        key = self._apply_style(preset)
        return ToolResult(f"Applied style {key!r} to {figure_id}.", {"style": key})

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @_tool
    def render_figure(self, figure_id: str, scale: float = 1.0) -> ToolResult:
        """Render the figure to a PNG the model can look at, with a text summary of the axes."""
        entry = self._figure(figure_id)
        if not 0.25 <= float(scale) <= 3.0:
            raise McpError("scale must be between 0.25 and 3.0.")
        data, view = self._render(entry, float(scale), "png", self.limits.max_image_edge)
        return ToolResult(
            f"Rendered {figure_id}.",
            {"figure_id": figure_id, "artists": len(entry.artists), "panels": view},
            image=data,
        )

    @_tool
    def save_figure(
        self,
        figure_id: str,
        path: str,
        scale: float = 1.0,
        transparent: bool = False,
        overwrite: bool = False,
    ) -> ToolResult:
        """Save the figure as PNG, SVG, PDF or JPEG inside the server's output directory."""
        entry = self._figure(figure_id)
        if not 0.25 <= float(scale) <= 4.0:
            raise McpError("scale must be between 0.25 and 4.0.")
        target = self.sandbox.write_path(path, overwrite=bool(overwrite))
        fmt = target.suffix.lower().lstrip(".")
        fmt = "jpeg" if fmt == "jpg" else fmt
        data, _ = self._render(entry, float(scale), fmt, 6000, transparent=bool(transparent))
        target.write_bytes(data)
        return ToolResult(
            f"Saved {figure_id} to {target} ({len(data) / 1024:.0f} KiB).",
            {"path": str(target), "bytes": len(data)},
        )

    def render_png(self, figure_id: str) -> bytes:
        """The figure as PNG bytes, for the resource endpoint. Raises :class:`McpError`."""
        with self._lock, contextlib.redirect_stdout(sys.stderr):
            return self._render(self._figure(figure_id), 1.0, "png", self.limits.max_image_edge)[0]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _figure(self, figure_id: str) -> _Figure:
        entry = self._figures.get(figure_id)
        if entry is None:
            known = ", ".join(self._figures) or "none open"
            raise McpError(
                f"No figure {figure_id!r} (open: {known}).{suggest(figure_id, self._figures)} "
                "Create one with create_figure."
            )
        return entry

    def figure_ids(self) -> List[str]:
        """Ids of the open figures."""
        with self._lock:
            return list(self._figures)

    def _discard(self, figure_id: str) -> None:
        entry = self._figures.pop(figure_id, None)
        if entry is not None:
            gplt.close(entry.plot)

    def close_all(self) -> None:
        """Close every figure. Called on shutdown, and by tests to leave pyplot clean."""
        with self._lock:
            for figure_id in list(self._figures):
                self._discard(figure_id)

    def _activate(self, entry: _Figure, panel: int) -> None:
        """Make ``entry`` pyplot's current figure and ``panel`` (1-based) its active axes."""
        if not isinstance(panel, int) or not 1 <= panel <= entry.panels:
            raise McpError(
                f"panel must be between 1 and {entry.panels} for {entry.figure_id}"
                f"{' (it has a single panel)' if entry.panels == 1 else ''}."
            )
        gplt.figure(num=entry.num)
        # Point the figure at the panel directly, the way ``AxesProxy`` does. Calling
        # ``subplot(rows, cols, i)`` to select it would also rebuild the grid with the
        # default spacing, undoing the ``wspace``/``hspace`` it was created with.
        entry.plot.active_panel_index = panel - 1

    def _call(self, what: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except McpError:
            raise
        except _USER_ERRORS as exc:
            raise McpError(f"{what}: {exc}") from exc

    def _apply_style(self, preset: str) -> str:
        try:
            return gplt.plot_style(preset)
        except ValueError as exc:
            keys = [row[0] for row in gplt.plot_styles()]
            raise McpError(
                f"Unknown style {preset!r}.{suggest(preset, keys)} Available: {', '.join(keys)}."
            ) from exc

    def _inline(self, values: Sequence[Optional[float]], what: str) -> np.ndarray:
        if len(values) > self.limits.max_inline_values:
            raise McpError(
                f"{what}: {len(values)} inline values exceeds {self.limits.max_inline_values}. "
                "Write the data to a file and use load_dataset, or build it with an expression."
            )
        return np.array([np.nan if v is None else v for v in values])

    def _series(
        self, ref: SeriesRef, dataset: Optional[str], what: str, *, required: bool
    ) -> Optional[np.ndarray]:
        """One data series from a column name (needs ``dataset``) or inline numbers."""
        if ref is None:
            if required:
                raise McpError(f"{what} is required for this kind.")
            return None
        if isinstance(ref, str):
            if dataset is None:
                raise McpError(
                    f"{what}={ref!r} names a column, so `dataset` is required. Create one with "
                    "create_dataset (it accepts expressions) or load_dataset, then pass its name."
                )
            return self.datasets.column(dataset, ref)
        if isinstance(ref, np.ndarray):
            # Only ever produced inside the server (``add_function``); the inline-size cap
            # exists for what arrives over the wire, and this did not.
            return as_float_array(ref, what, limits=self.limits)
        if isinstance(ref, (list, tuple)):
            return as_float_array(self._inline(ref, what), what, limits=self.limits)
        raise McpError(
            f"{what}: give a column name or a list of numbers, got {type(ref).__name__}."
        )

    def _check_projection_for_plot(self, entry: _Figure, kind: str, z: SeriesRef) -> None:
        if entry.is_3d:
            if kind not in ("line", "scatter"):
                raise McpError(
                    f"{entry.figure_id} is a 3-D figure: add_plot supports only kind='line' and "
                    "'scatter' there (with z); use add_field "
                    "kind='surface'/'wireframe' for surfaces."
                )
            if z is None:
                raise McpError("z is required to plot on a 3-D figure.")
        elif z is not None:
            raise McpError(
                f"z was given but {entry.figure_id} is 2-D. Create the figure with projection='3d'."
            )

    @staticmethod
    def _range(value: Sequence[float], what: str) -> Tuple[float, float]:
        try:
            lo, hi = (float(value[0]), float(value[1]))
            if len(value) != 2:
                raise ValueError
        except (TypeError, ValueError, IndexError):
            raise McpError(f"{what} must be two numbers [low, high].") from None
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo == hi:
            raise McpError(
                f"{what} must be two distinct finite numbers, got [{value[0]}, {value[1]}]."
            )
        return lo, hi

    def _matrix(self, matrix: Sequence[Sequence[float]], what: str) -> np.ndarray:
        try:
            grid = np.array(
                [[np.nan if v is None else v for v in row] for row in matrix], dtype=np.float64
            )
        except (TypeError, ValueError) as exc:
            raise McpError(
                f"{what}: rows must all have the same length and contain numbers ({exc})."
            ) from exc
        if grid.ndim != 2 or min(grid.shape) < 2:
            raise McpError(f"{what}: expected a 2-D grid of at least 2x2, got shape {grid.shape}.")
        if grid.size > self.limits.max_inline_values:
            raise McpError(
                f"{what}: {grid.size} inline cells exceeds {self.limits.max_inline_values}. "
                "Save it as .npy and pass file=, or use an expression."
            )
        return grid

    def _field_file(self, file: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """A 2-D grid (and, for a colour image, its RGB array) read from the sandbox."""
        path = self.sandbox.read_path(file)
        extension = path.suffix.lower()
        if extension == ".npy":
            try:
                grid = np.load(path, allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise McpError(f"Could not read {file!r}: {exc}") from exc
            if grid.ndim != 2:
                raise McpError(f"{file!r} holds a {grid.ndim}-D array; a field must be 2-D.")
            return np.asarray(grid, dtype=np.float64), None
        if extension in dataio.IMAGE_EXTENSIONS:
            import matplotlib.image as mimage

            try:
                pixels = mimage.imread(str(path))
            except Exception as exc:
                raise McpError(f"Could not read image {file!r}: {exc}") from exc
            if pixels.shape[0] * pixels.shape[1] > dataio.MAX_IMAGE_PIXELS * 4:
                raise McpError(
                    f"{file!r} is {pixels.shape[1]}x{pixels.shape[0]}; "
                    "downscale it below ~4 megapixels."
                )
            if pixels.ndim == 2:
                return np.asarray(pixels, dtype=np.float64), None
            return np.asarray(pixels[..., 0], dtype=np.float64), pixels
        raise McpError(
            f"A field file must be .npy or an image ({', '.join(dataio.IMAGE_EXTENSIONS)}), "
            f"not {extension!r}."
        )

    def _record(
        self,
        entry: _Figure,
        panel: int,
        kind: str,
        label: str,
        draw: Callable[[], Any],
        req: Optional["_PlotRequest"],
    ) -> _Artist:
        """Run ``draw`` and bundle the scene layers it created into one :class:`_Artist`."""
        scene = entry.plot.scene
        before = {layer.layer_id for layer in scene.layers}
        try:
            draw()
        except McpError:
            self._remove_new_layers(entry, before)
            raise
        except _USER_ERRORS:
            self._remove_new_layers(entry, before)
            raise
        created = [
            layer
            for layer in entry.plot.scene.layers
            if layer.layer_id not in before
            and layer.metadata.get("artist") not in SYSTEM_3D_ARTISTS
        ]
        artist = _Artist(
            artist_id=f"a{entry.next_artist}",
            kind=kind,
            panel=panel,
            label=label,
            layer_ids=[layer.layer_id for layer in created],
            points=None if req is None else req.n,
        )
        entry.next_artist += 1
        entry.artists.append(artist)
        return artist

    def _remove_new_layers(self, entry: _Figure, before: set) -> None:
        """Undo a half-finished draw so a failed call leaves the figure as it found it."""
        for layer in [
            l
            for l in entry.plot.scene.layers
            if l.layer_id not in before and l.metadata.get("artist") not in SYSTEM_3D_ARTISTS
        ]:
            layerops.remove_layer(entry.plot, None, layer)

    def _render(
        self,
        entry: _Figure,
        scale: float,
        fmt: str,
        max_edge: int,
        *,
        transparent: bool = False,
    ) -> Tuple[bytes, List[Dict[str, Any]]]:
        """Render to ``fmt`` bytes and read the resolved axes from the same matplotlib figure."""
        view: List[Dict[str, Any]] = []

        def inspect(fig: Any) -> Dict[str, Any]:
            if entry.equal_panels:
                data_axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
                for index, ax in enumerate(data_axes, start=1):
                    if index in entry.equal_panels and not hasattr(ax, "get_zlim"):
                        ax.set_aspect("equal")
            view.extend(_summarise_axes(fig))
            if fmt in ("png", "jpeg"):
                edge = max(fig.get_size_inches()) * fig.dpi
                if edge > max_edge:
                    # Lower the *export* dpi: ``savefig`` does not read ``fig.dpi`` back.
                    return {"dpi": fig.dpi * max_edge / edge}
            return {}

        self._activate(entry, 1)
        kwargs: Dict[str, Any] = {"bbox_inches": "tight", "pad_inches": 0.15}
        if transparent:
            kwargs["transparent"] = True
        if fmt == "jpeg":
            kwargs["pil_kwargs"] = {"quality": 92}
        try:
            data = render_preview_bytes(entry.plot, scale, format=fmt, inspect=inspect, **kwargs)
        except McpError:
            raise
        except (ValueError, TypeError) as exc:
            raise McpError(f"Could not render {entry.figure_id}: {exc}") from exc
        return data, view

    def _resolve_view(self, entry: _Figure) -> List[Dict[str, Any]]:
        """What the finished picture shows per panel (limits, labels, legend), at low cost."""
        return self._render(entry, 0.25, "png", 200)[1]


@dataclass
class _PlotRequest:
    """``add_plot``'s arguments after validation, in the shape the drawing functions want."""

    kind: str
    dataset: Optional[str] = None
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    error: Optional[np.ndarray] = None
    y2: Optional[np.ndarray] = None
    color_by: Optional[np.ndarray] = None
    size_by: Optional[np.ndarray] = None
    groups: Optional[List[np.ndarray]] = None
    group_names: Optional[List[str]] = None
    n: int = 0
    options: Dict[str, Any] = field(default_factory=dict)
    tick_labels: Optional[List[str]] = None
    color_label: Optional[str] = None
    colorbar: bool = False
    ignored: List[str] = field(default_factory=list)

    def style(self, mapping: Mapping[str, str]) -> Dict[str, Any]:
        """Translate the options this kind understands to pyplot kwargs; note the rest.

        An option the kind does not use is *reported*, not an error: a model that passes
        ``marker='o'`` to a bar chart should learn it did nothing rather than lose the call.
        """
        out: Dict[str, Any] = {}
        for name, value in self.options.items():
            if value is None:
                continue
            if name in mapping:
                out[mapping[name]] = value
            elif name not in self.ignored:
                self.ignored.append(name)
        return out


def _slot_help(spec: recipes.Recipe) -> str:
    """One line telling the model which slots a recipe has and how to fill one."""
    parts = [f"{slot.key} ({'required' if slot.required else 'optional'})" for slot in spec.slots]
    first = next((slot.key for slot in spec.slots if slot.required and not slot.multi), None)
    example = f" Example: columns={{'{first}': '<column>'}}." if first else ""
    return f"Slots of {spec.key!r}: {', '.join(parts)}.{example}"


def _describe_recipe(spec: recipes.Recipe) -> Dict[str, Any]:
    """A recipe as JSON for ``list_recipes``: what it draws, what it reads, how it looks."""
    needs = []
    for slot in spec.slots:
        row: Dict[str, Any] = {
            "slot": slot.key,
            "label": slot.label,
            "required": slot.required,
        }
        if slot.hint:
            row["hint"] = slot.hint
        if slot.multi:
            row["uses"] = "every column of the dataset"
        needs.append(row)
    return {
        "recipe": spec.key,
        "name": spec.name,
        "category": spec.category,
        "dimension": spec.dimension,
        "kind": spec.kind,
        "description": spec.description,
        "needs": needs,
        "look": spec.style,
        "sample": spec.sample is not None,
    }


def _summarise_axes(fig: Any) -> List[Dict[str, Any]]:
    """Per-axes facts a model cannot read off pixels reliably: limits, labels, legend text."""
    panels: List[Dict[str, Any]] = []
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>":
            continue
        entry: Dict[str, Any] = {"panel": len(panels) + 1}
        if ax.get_title():
            entry["title"] = ax.get_title()
        entry["xlabel"], entry["ylabel"] = ax.get_xlabel(), ax.get_ylabel()
        entry["xlim"] = [_round(v) for v in ax.get_xlim()]
        entry["ylim"] = [_round(v) for v in ax.get_ylim()]
        if hasattr(ax, "get_zlim"):
            entry["zlabel"] = ax.get_zlabel()
            entry["zlim"] = [_round(v) for v in ax.get_zlim()]
        for axis in ("x", "y"):
            scale = getattr(ax, f"get_{axis}scale")()
            if scale != "linear":
                entry[f"{axis}scale"] = scale
        legend = ax.get_legend()
        if legend is not None:
            entry["legend"] = [t.get_text() for t in legend.get_texts()]
        panels.append(entry)
    return panels


# ----------------------------------------------------------------------------------
# Drawing functions, one per add_plot kind
# ----------------------------------------------------------------------------------


def _draw_line(r: _PlotRequest) -> Any:
    kw = r.style(
        {
            "color": "color",
            "alpha": "alpha",
            "linewidth": "linewidth",
            "linestyle": "linestyle",
            "marker": "marker",
            "size": "markersize",
            "label": "label",
        }
    )
    if r.z is not None:
        return gplt.plot(r.x, r.y, r.z, **kw)
    return gplt.plot(r.x, r.y, **kw)


def _draw_scatter(r: _PlotRequest) -> Any:
    kw = r.style(
        {
            "color": "color",
            "alpha": "alpha",
            "marker": "marker",
            "size": "size",
            "label": "label",
            "cmap": "cmap",
        }
    )
    if r.color_by is not None:
        kw["c"] = r.color_by
        kw.pop("color", None)
    if r.size_by is not None:
        span = float(np.nanmax(r.size_by) - np.nanmin(r.size_by))
        unit = (r.size_by - np.nanmin(r.size_by)) / span if span > 0 else np.full(r.n, 0.5)
        kw["size"] = 4.0 + 24.0 * unit
    if r.z is not None:
        layer = gplt.scatter(r.x, r.y, r.z, **kw)
    else:
        layer = gplt.scatter(r.x, r.y, **kw)
    if r.colorbar and r.color_by is not None:
        gplt.colorbar(label=r.color_label) if r.color_label else gplt.colorbar()
    return layer


def _draw_step(r: _PlotRequest) -> Any:
    kw = r.style(
        {
            "color": "color",
            "alpha": "alpha",
            "linewidth": "linewidth",
            "linestyle": "linestyle",
            "label": "label",
        }
    )
    return gplt.step(r.x, r.y, where="mid", **kw)


def _draw_stem(r: _PlotRequest) -> Any:
    kw = r.style({"label": "label"})  # colour and the rest are not stem options: reported
    return gplt.stem(r.x, r.y, **kw)


def _draw_bar(r: _PlotRequest) -> Any:
    kw = r.style({"color": "color", "alpha": "alpha", "label": "label", "width": "width"})
    if r.error is not None:
        kw["yerr"] = r.error
    bars = gplt.bar(r.x, r.y, **kw)
    if r.tick_labels:
        gplt.xticks(list(r.x), r.tick_labels)
    return bars


def _draw_barh(r: _PlotRequest) -> Any:
    kw = r.style({"color": "color", "alpha": "alpha", "label": "label"})
    bars = gplt.barh(r.x, r.y, **kw)
    if r.tick_labels:
        gplt.yticks(list(r.x), r.tick_labels)
    return bars


def _draw_area(r: _PlotRequest) -> Any:
    kw = r.style({"color": "color", "alpha": "alpha", "label": "label"})
    return gplt.fill_between(r.x, r.y, r.y2 if r.y2 is not None else 0.0, **kw)


def _draw_errorbar(r: _PlotRequest) -> Any:
    kw = r.style({"color": "color", "label": "label"})
    return gplt.errorbar(r.x, r.y, yerr=r.error, **kw)


def _draw_hist(r: _PlotRequest) -> Any:
    kw = r.style({"color": "color", "alpha": "alpha", "label": "label", "bins": "bins"})
    return gplt.hist(r.y, **kw)


def _draw_box(r: _PlotRequest) -> Any:
    groups = r.groups if r.groups is not None else [r.y]
    r.style({})
    result = gplt.boxplot(groups)
    if r.group_names:
        gplt.xticks(list(range(1, len(groups) + 1)), r.group_names)
    return result


def _draw_violin(r: _PlotRequest) -> Any:
    groups = r.groups if r.groups is not None else [r.y]
    r.style({})
    result = gplt.violinplot(groups)
    if r.group_names:
        gplt.xticks(list(range(1, len(groups) + 1)), r.group_names)
    return result


def _draw_pie(r: _PlotRequest) -> Any:
    if np.any(r.y < 0):
        raise McpError("pie needs non-negative values.")
    r.style({})
    return gplt.pie(r.y, labels=r.tick_labels)


def _draw_hist2d(r: _PlotRequest) -> Any:
    kw = r.style({"cmap": "cmap", "bins": "bins"})
    return gplt.hist2d(r.x, r.y, **kw)


def _draw_hexbin(r: _PlotRequest) -> Any:
    kw = r.style({"cmap": "cmap"})
    return gplt.hexbin(r.x, r.y, **kw)


_DRAW: Dict[str, Callable[[_PlotRequest], Any]] = {
    "line": _draw_line,
    "scatter": _draw_scatter,
    "step": _draw_step,
    "stem": _draw_stem,
    "bar": _draw_bar,
    "barh": _draw_barh,
    "area": _draw_area,
    "errorbar": _draw_errorbar,
    "hist": _draw_hist,
    "box": _draw_box,
    "violin": _draw_violin,
    "pie": _draw_pie,
    "hist2d": _draw_hist2d,
    "hexbin": _draw_hexbin,
}
