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
import copy
import functools
import inspect
import json
import math
import sys
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib import colors as mcolors

from .. import pyplot as gplt
from ..core.camera3d import SYSTEM_3D_ARTISTS
from ..gui import dataio, layerops, recipes, styles
from ..gui.datasets import Column, DataSet
from ..gui.expressions import ExpressionError, evaluate_2d
from ..pyplot import MatplotlibCompatWarning
from ..utils.preview import render_preview_bytes
from . import animator, choreography, dynamics, motion, video
from . import script as scripting
from . import simulate as simulation
from . import textdata
from .review import review_figure
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
ANNOTATION_KINDS: Tuple[str, ...] = ("text", "hline", "vline", "arrow")

#: What pyplot raises for a mistake in the arguments or the state it was handed (a bad colour,
#: mismatched lengths, ``colorbar()`` with nothing to attach to). Each message already says what
#: was wrong, so it is passed to the model as-is; anything else is a bug and is left to propagate.
_USER_ERRORS = (ValueError, TypeError, KeyError, IndexError, RuntimeError, ExpressionError)
#: Kinds drawn in black (or one fixed blue) unless told otherwise; they take the next colour
#: from the figure's cycle instead, as ``plot()`` does.
_CYCLED_KINDS = {"scatter", "errorbar", "bar", "barh", "area", "hist"}
#: Kinds that read *one* series of values (from ``y``, else ``x``) rather than ``x`` vs ``y``.
_SINGLE_SERIES = {"hist", "box", "violin", "pie"}
#: Kinds that can be drawn on a second (right-hand) y axis.
_TWIN_KINDS = {"line", "scatter", "step", "bar", "area", "errorbar"}
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
        # ensure_ascii=False: a title like "T (°C)" or a label with µ/Å/accents is read by the model
        # as written, not as \\u00b0 escapes it has to decode in its head.
        payload = json.dumps(
            _finite(self.data), separators=(",", ":"), default=_json_default, ensure_ascii=False
        )
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
class _Op:
    """One successful scene-building tool call, kept so the figure can be rebuilt from it.

    Rebuilding is how ``update_artist`` changes something already drawn, how a control or an
    animation frame gives a property another value, and what ``export_script`` turns into code.
    ``args`` are exactly the arguments the caller passed (not the defaults), so a replay makes
    the same call.
    """

    tool: str
    args: Dict[str, Any]
    artist_id: Optional[str] = None
    #: Choices the toolbox made for the caller (the colour taken from the figure's cycle...),
    #: kept so an exported script makes the very same call.
    resolved: Dict[str, Any] = field(default_factory=dict)


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
    #: Panels whose legend was switched off on purpose (set_axes(legend=False)).
    legend_off: set = field(default_factory=set)
    #: What the last reviewed render found (see :mod:`glplot.mcp.review`).
    issues: List[Dict[str, Any]] = field(default_factory=list)
    #: The arguments ``create_figure`` was called with (title, style, grid, size...).
    spec: Dict[str, Any] = field(default_factory=dict)
    #: Every scene-building call so far, in order (see :class:`_Op`).
    log: List[_Op] = field(default_factory=list)
    #: The right-hand y axis of a panel (1-based), created by the first ``axis="right"`` plot.
    twins: Dict[int, Any] = field(default_factory=dict)
    #: Named values the figure's properties can depend on, and the properties that do.
    controls: Dict[str, dynamics.Control] = field(default_factory=dict)
    bindings: List[dynamics.Binding] = field(default_factory=list)
    #: The last ``animate`` specification (what ``export_script`` turns into an animation).
    animation: Optional[Dict[str, Any]] = None
    #: What that animation rendered, for ``compose_video``: ``{signature, frames, fps, bytes}``.
    #: Only reused while :meth:`Toolbox._signature` of the figure is unchanged.
    rendered: Optional[Dict[str, Any]] = None

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
                {
                    str(w.message)
                    for w in caught
                    if issubclass(w.category, MatplotlibCompatWarning)
                    # Every MCP picture is the export, which does draw hatching.
                    and "is drawn in exported figures only" not in str(w.message)
                }
            )
            if notes:
                result.data.setdefault("warnings", notes)
            return result

    return wrapper


def _scene_op(method: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
    """Record a successful scene-building call on its figure, so the figure can be replayed.

    Only the outermost call is recorded: ``add_function`` draws by calling ``add_plot``, and the
    replay must repeat ``add_function``, not both. Nothing is recorded while replaying.
    """
    signature = inspect.signature(method)

    @functools.wraps(method)
    def wrapper(self: "Toolbox", *args: Any, **kwargs: Any) -> ToolResult:
        bound = signature.bind(self, *args, **kwargs)
        figure_id = bound.arguments.get("figure_id")
        outermost = self._op_depth == 0 and not self._replaying
        if outermost:
            self._resolved = {}
        self._op_depth += 1
        try:
            result = method(self, *args, **kwargs)
        finally:
            self._op_depth -= 1
        entry = self._figures.get(figure_id) if outermost else None
        if entry is not None:
            given = {
                key: copy.deepcopy(value)
                for key, value in bound.arguments.items()
                if key not in ("self", "figure_id")
            }
            self._log_operation(entry, method.__name__, given, result)
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
        #: Nesting of scene-building calls, and whether a figure is being replayed (see _scene_op).
        self._op_depth = 0
        self._replaying = False
        self._resolved: Dict[str, Any] = {}
        self._next_scratch = 1
        #: Figure ids whose rendered frames are kept, oldest first (see _remember_frames).
        self._frame_order: List[str] = []

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
        spec = {
            "title": title,
            "width": width,
            "height": height,
            "rows": rows,
            "cols": cols,
            "projection": projection,
            "style": style,
        }
        entry = self._make_entry(figure_id, spec, _FIGURE_NUM_BASE + int(figure_id[3:]))
        self._figures[figure_id] = entry
        return ToolResult(
            f"Created {figure_id} ({width}x{height}px, {rows}x{cols} panel"
            f"{'s' if entry.panels > 1 else ''}, {projection}).",
            {"figure_id": figure_id, "panels": entry.panels},
        )

    def _make_entry(self, figure_id: str, spec: Mapping[str, Any], num: int) -> _Figure:
        """A new empty figure for ``spec`` (the arguments of ``create_figure``); not registered."""
        width, height = spec["width"], spec["height"]
        rows, cols, projection = spec["rows"], spec["cols"], spec["projection"]
        three_d = projection == "3d"
        # An integer ``num`` and the stock window title: a string ``num`` doubles as the window
        # caption, and a non-stock caption is drawn as the plot's title -- every render would
        # carry "mcp-fig1" above the axes. The real title is set below, only if one was asked.
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
        entry = _Figure(
            figure_id, plot, rows, cols, projection, width, height, num=num, spec=dict(spec)
        )
        try:
            if entry.panels > 1:
                self._activate(entry, 1)
            if spec.get("style"):
                self._apply_style(spec["style"])
            if spec.get("title"):
                (gplt.suptitle if entry.panels > 1 else gplt.title)(spec["title"])
        except Exception:
            gplt.close(plot)
            raise
        return entry

    # ------------------------------------------------------------------
    # Replay: the figure rebuilt from what was asked of it
    # ------------------------------------------------------------------

    def _log_operation(
        self, entry: _Figure, tool: str, args: Dict[str, Any], result: ToolResult
    ) -> None:
        """Remember a scene-building call (see :func:`_scene_op`)."""
        if tool == "set_axes" and entry.log:
            last = entry.log[-1]
            if last.tool == "set_axes" and last.args.get("panel", 1) == args.get("panel", 1):
                last.args.update(args)  # the same panel set again: later values win
                return
        cap = self.limits.animation.max_operations
        if len(entry.log) >= cap:
            raise McpError(
                f"{entry.figure_id} already holds {cap} operations. Start a new figure, or "
                "remove artists you no longer need."
            )
        artist_id = result.data.get("artist_id") if isinstance(result.data, dict) else None
        entry.log.append(_Op(tool, args, artist_id, dict(self._resolved)))
        self._resolved = {}

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _object_arguments(tool: str) -> frozenset:
        """Names of the arguments of ``tool`` that are objects (``parameters``, ``columns``...)."""
        parameters = inspect.signature(getattr(Toolbox, tool)).parameters
        return frozenset(name for name, p in parameters.items() if "Mapping" in str(p.annotation))

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _text_arguments(tool: str) -> frozenset:
        """Names of the arguments of ``tool`` that are plain text (colour, label, title...)."""
        parameters = inspect.signature(getattr(Toolbox, tool)).parameters
        plain = ("str", "Optional[str]")
        return frozenset(
            name
            for name, p in parameters.items()
            if str(p.annotation).replace("typing.", "") in plain
            and name not in ("self", "figure_id")
        )

    def _check_property(self, entry: _Figure, kind: str, ident: Any, prop: str) -> str:
        """The tool a binding's ``prop`` is an argument of (and a readable error when it is not)."""
        if kind == "artist":
            op = next((o for o in entry.log if o.artist_id == ident), None)
            tool = op.tool if op else "set_axes"
        else:
            tool = "set_axes"
        hidden = ("self", "figure_id") + (("panel",) if tool == "set_axes" else ())
        allowed = [
            name
            for name in inspect.signature(getattr(Toolbox, tool)).parameters
            if name not in hidden
        ]
        head = prop.split(".")[0]
        if head not in allowed:
            what = (
                f"{ident} was drawn by {tool}" if kind == "artist" else "axes are set by set_axes"
            )
            raise McpError(
                f"{what}, which has no argument {head!r}.{suggest(head, allowed)} "
                f"Arguments: {', '.join(allowed)}."
            )
        self._check_path(tool, prop)
        return tool

    def _check_path(self, tool: str, path: str) -> None:
        """A dotted ``path`` may only reach into an argument that is an object."""
        head = path.split(".")[0]
        if "." in path and head not in self._object_arguments(tool):
            objects = sorted(self._object_arguments(tool)) or ["(none)"]
            raise McpError(
                f"{path!r}: {head!r} is not an object argument of {tool}, so it has no "
                f"parts to set. Object arguments: {', '.join(objects)}."
            )

    def _scratch_id(self, entry: _Figure) -> str:
        self._next_scratch += 1
        return f"{entry.figure_id}~{self._next_scratch}"

    def _build(
        self,
        entry: _Figure,
        *,
        ops: Optional[Sequence[_Op]] = None,
        artist_args: Optional[Mapping[str, Mapping[str, Any]]] = None,
        axes_args: Optional[Mapping[int, Mapping[str, Any]]] = None,
        datasets: Optional[DatasetStore] = None,
    ) -> _Figure:
        """Replay ``ops`` (default: the figure's own log) into a brand-new, unregistered figure.

        ``artist_args`` maps an artist id to ``{property path: value}`` replacing arguments of
        the call that drew it (``None`` removes one, back to its default); ``axes_args`` maps a
        panel to ``set_axes`` arguments applied last; ``datasets`` stands in for the dataset
        store (an animation frame's recomputed columns). The caller owns the result and must
        :meth:`_adopt` or :meth:`_release` it. Failure leaves no trace.
        """
        scratch = self._scratch_id(entry)
        built = self._make_entry(
            scratch, entry.spec, _FIGURE_NUM_BASE + 100_000 + self._next_scratch
        )
        self._figures[scratch] = built
        saved_store, saved_flag = self.datasets, self._replaying
        self._replaying = True
        if datasets is not None:
            self.datasets = datasets
        try:
            for op in entry.log if ops is None else ops:
                args = copy.deepcopy(op.args)
                for path, value in ((artist_args or {}).get(op.artist_id or "") or {}).items():
                    self._check_path(op.tool, path)
                    if value is None:
                        args.pop(path, None)
                    else:
                        dynamics.set_path(args, path, value)
                if op.artist_id:
                    built.next_artist = int(op.artist_id[1:])
                try:
                    getattr(self, op.tool)(scratch, **args)
                except McpError as exc:
                    where = f" ({op.artist_id})" if op.artist_id else ""
                    text = str(exc).replace(scratch, entry.figure_id)
                    raise McpError(
                        f"Could not rebuild {entry.figure_id}: {op.tool}{where} failed: {text}"
                    ) from exc
            for panel, values in (axes_args or {}).items():
                if values:
                    try:
                        self.set_axes(scratch, panel=panel, **values)
                    except McpError as exc:
                        text = str(exc).replace(scratch, entry.figure_id)
                        raise McpError(f"Setting axes of panel {panel}: {text}") from exc
            built.next_artist = max(built.next_artist, entry.next_artist)
            return built
        except BaseException:
            self._release(built)
            raise
        finally:
            self.datasets, self._replaying = saved_store, saved_flag

    def _release(self, built: _Figure) -> None:
        """Throw away a figure made by :meth:`_build`."""
        self._figures.pop(built.figure_id, None)
        gplt.close(built.plot)

    def _adopt(self, entry: _Figure, built: _Figure) -> None:
        """Make the rebuilt figure ``entry``'s own: its scene replaces the old one."""
        old_plot = entry.plot
        self._figures.pop(built.figure_id, None)
        entry.plot, entry.num = built.plot, built.num
        entry.artists, entry.next_artist = built.artists, built.next_artist
        entry.equal_panels, entry.twins = built.equal_panels, built.twins
        entry.legend_off = built.legend_off
        gplt.close(old_plot)

    def _context_values(self, entry: _Figure) -> Dict[str, float]:
        """The number each control currently stands for, by name."""
        out: Dict[str, float] = {}
        for name, control in entry.controls.items():
            value = dynamics.control_variable(control)
            if value is not None:
                out[name] = value
        return out

    def _bound_arguments(
        self,
        entry: _Figure,
        bindings: Sequence[dynamics.Binding],
        context: Mapping[str, Any],
        n: int,
    ) -> Tuple[List[Dict[str, Dict[str, Any]]], List[Dict[int, Dict[str, Any]]], List[Any]]:
        """Evaluate scalar bindings for ``n`` frames: per-frame artist and axes arguments.

        The third item lists the ``data:`` bindings, which are computed per frame from the
        dataset's own columns and so are not evaluated here.
        """
        per_artist: List[Dict[str, Dict[str, Any]]] = [dict() for _ in range(n)]
        per_axes: List[Dict[int, Dict[str, Any]]] = [dict() for _ in range(n)]
        data: List[Any] = []
        known = {a.artist_id for a in entry.artists}
        for binding in bindings:
            kind, ident = dynamics.parse_target(binding.target)
            if kind == "data":
                self.datasets.get(ident)
                data.append(binding)
                continue
            if kind == "artist" and ident not in known:
                raise McpError(
                    f"No artist {ident!r} in {entry.figure_id} (artists: "
                    f"{', '.join(sorted(known)) or 'none'})."
                )
            if kind == "axes" and not 1 <= ident <= entry.panels:
                raise McpError(
                    f"{binding.target!r}: {entry.figure_id} has {entry.panels} panel(s)."
                )
            tool = self._check_property(entry, kind, ident, binding.prop)
            values = dynamics.evaluate_value(
                binding.value,
                context,
                n,
                limits=self.limits,
                what=f"{binding.target}.{binding.prop}",
                text=binding.prop in self._text_arguments(tool),
            )
            for i, value in enumerate(values):
                value = dynamics.coerce_property(binding.prop, value)
                if kind == "artist":
                    per_artist[i].setdefault(ident, {})[binding.prop] = value
                else:
                    per_axes[i].setdefault(ident, {})[binding.prop] = value
        return per_artist, per_axes, data

    def _frame_datasets(
        self, data_bindings: Sequence[dynamics.Binding], scalars: Mapping[str, float]
    ) -> Optional[DatasetStore]:
        """A dataset store whose bound columns are recomputed for one frame (None: no change)."""
        if not data_bindings:
            return None
        store = DatasetStore(self.limits)
        store._sets = dict(self.datasets._sets)
        copies: Dict[str, DataSet] = {}
        variables = {**dynamics.HELPERS, **scalars}
        for binding in data_bindings:
            _, name = dynamics.parse_target(binding.target)
            table = copies.get(name)
            if table is None:
                source = store.get(name)
                table = DataSet(name, [Column(c.name, c.values.copy()) for c in source.columns])
                table.source = source.source
                table.categories = {k: list(v) for k, v in source.categories.items()}
                table.date_columns = dict(source.date_columns)
                copies[name] = table
            rows = table.n_rows()
            scope = dict(variables)
            scope.update({c.name: np.asarray(c.values, dtype=np.float64) for c in table.columns})
            expression = binding.value
            if not isinstance(expression, str):
                raise McpError(
                    f"data:{name}.{binding.prop}: the value of a data binding is an expression "
                    "string over the dataset's columns, e.g. 'sin(k*x - phase)'."
                )
            result = evaluate_expression(
                expression, scope, limits=self.limits, what=f"data:{name}.{binding.prop}"
            )
            values = as_float_array(
                result, f"data:{name}.{binding.prop}", length=rows, limits=self.limits
            )
            for column in table.columns:
                if column.name == binding.prop:
                    column.values = values
                    column.invalidate()
                    break
            else:
                table.columns.append(Column(binding.prop, values))
        for name, table in copies.items():
            store._sets[name] = table
        return store

    def _refresh(self, entry: _Figure, ops: Optional[Sequence[_Op]] = None) -> None:
        """Rebuild ``entry`` from ``ops`` (default: its log) with bindings at the control values.

        With ``ops`` given they become the figure's log once the rebuild has succeeded.
        """
        built = self._build_for(entry, ops=ops)
        self._adopt(entry, built)
        if ops is not None:
            entry.log = list(ops)

    def _build_for(
        self,
        entry: _Figure,
        *,
        ops: Optional[Sequence[_Op]] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> _Figure:
        """A rebuilt copy of ``entry`` with its bindings evaluated at the (overridden) controls."""
        controls = {k: copy.copy(v) for k, v in entry.controls.items()}
        for name, raw in (params or {}).items():
            if name not in controls:
                known = ", ".join(controls) or "none -- add one with add_control"
                raise McpError(
                    f"{entry.figure_id} has no control {name!r}.{suggest(name, controls)} "
                    f"Controls: {known}."
                )
            if controls[name].kind == "button":
                raise McpError(
                    f"{name!r} is a button; params sets values (sliders, toggles, choices)."
                )
            controls[name].value = dynamics.set_control_value(controls[name], raw)
        values = {
            n: dynamics.control_variable(c)
            for n, c in controls.items()
            if dynamics.control_variable(c) is not None
        }
        context = {k: np.full(1, v) for k, v in values.items()}
        artist_args, axes_args, data = self._bound_arguments(entry, entry.bindings, context, 1)
        datasets = self._frame_datasets(data, values)
        return self._build(
            entry,
            ops=ops,
            artist_args=artist_args[0],
            axes_args=axes_args[0],
            datasets=datasets,
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
        recipes_by_artist = {op.artist_id: op for op in entry.log if op.artist_id}
        artists = []
        for artist in entry.artists:
            item = artist.describe()
            op = recipes_by_artist.get(artist.artist_id)
            if op is not None:
                item["recipe"] = {"tool": op.tool, **_brief(op.args)}
            artists.append(item)
        data: Dict[str, Any] = {
            "figure_id": entry.figure_id,
            "size_px": [entry.width, entry.height],
            "projection": entry.projection,
            "grid": [entry.rows, entry.cols],
            "artists": artists,
            "panels": view,
        }
        if entry.controls:
            data["controls"] = [c.describe() for c in entry.controls.values()]
        if entry.bindings:
            data["bindings"] = [b.describe() for b in entry.bindings]
        if entry.animation:
            data["animation"] = _brief(entry.animation)
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
        table.mcp_origin = {
            "kind": "create",
            "columns": copy.deepcopy(dict(columns)),
            "base": base,
            "names": table.column_names(),
        }
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
        table.mcp_origin = {
            "kind": "file",
            "path": str(resolved),
            "names": table.column_names(),
        }
        encoding: Dict[str, Any] = {}
        if resolved.suffix.lower() not in (
            *dataio.ARRAY_EXTENSIONS,
            *dataio.JSON_EXTENSIONS,
            *dataio.IMAGE_EXTENSIONS,
        ):
            try:
                text = dataio.text_columns(str(resolved))
            except dataio.DataIOError:
                text = {}
            if text:
                encoding = textdata.interpret_text_columns(table, text)
        replaced = self.datasets.put(table)
        verb = "Replaced" if replaced else "Loaded"
        sentence = textdata.describe_encoding(encoding)
        return ToolResult(
            f"{verb} {resolved.name} as dataset {dataset_name!r} "
            f"({table.n_rows()} rows x {table.n_cols()} columns)."
            + (f" {sentence}" if sentence else ""),
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
    def analyze_dataset(
        self,
        dataset: str,
        operation: str,
        y: Optional[str] = None,
        x: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
        output: Optional[str] = None,
    ) -> ToolResult:
        """Run a numerical analysis (fit, smooth, FFT, peaks, filter...) and keep the result."""
        from . import analysis

        op = analysis.operation(operation)
        table = self.datasets.get(dataset)
        rows = table.n_rows()
        if rows > self.limits.max_analysis_rows:
            raise McpError(
                f"Dataset {dataset!r} has {rows} rows; analyze_dataset works on at most "
                f"{self.limits.max_analysis_rows}. Reduce it first (create_dataset with base= and "
                "a resampled expression, or load a smaller file)."
            )
        if rows < 2:
            raise McpError(f"Dataset {dataset!r} has {rows} row(s); an analysis needs at least 2.")
        names = table.column_names()
        notes: List[str] = []

        # x: a real column, or the row index (unit spacing) when the operation wants one and none
        # was named. Operations that compare two series insist on a real column.
        x_is_index = True
        x_name = "(index)"
        x_values = np.arange(rows, dtype=np.float64)
        if x is not None and x != "(index)":
            if not op.uses_x:
                notes.append(f"{op.name} does not use x; the x column {x!r} was ignored.")
            else:
                x_values = np.array(self.datasets.column(dataset, x), dtype=np.float64)
                x_is_index, x_name = False, x
        elif op.x_required:
            raise McpError(
                f"{op.name} compares two series: give both x and y columns "
                f"(columns: {', '.join(names)})."
            )

        if y is None:
            candidates = [n for n in names if n != x_name]
            if len(candidates) != 1:
                raise McpError(
                    f"Which column should {op.name} use? Pass y=... (columns: {', '.join(names)})."
                )
            y = candidates[0]
        y_values = np.array(self.datasets.column(dataset, y), dtype=np.float64)

        ctx = analysis.Context(
            x=x_values,
            y=y_values,
            options=analysis.validate_options(op, options),
            x_name=x_name,
            y_name=y,
            x_is_index=x_is_index,
            limits=self.limits,
            columns={
                column.name: np.asarray(column.values, dtype=np.float64) for column in table.columns
            },
            categories=table.categories,
        )
        outcome = op.run(ctx)
        notes = outcome.notes + notes
        if x_is_index and op.uses_x and op.index_note and not op.x_required:
            notes.append("No x column given: the row index (unit spacing) was used as x.")

        data: Dict[str, Any] = {
            "operation": op.name,
            "dataset": dataset,
            "y": y,
            "x": None if x_is_index else x_name,
            "rows": rows,
            "result": outcome.report,
        }
        hint = None
        if outcome.column is not None:
            column_name = self._analysis_column_name(output, y, x_name, outcome, op)
            values = np.asarray(outcome.column, dtype=np.float64)
            if values.shape != (rows,):
                raise McpError(f"{op.name} produced {values.size} values for {rows} rows.")
            table.remove_column(column_name)
            table.add_column(column_name, values)
            data["created"] = {"column": column_name, "dataset": dataset}
            where = "" if x_is_index else f", x={x_name!r}"
            hint = f"add_plot(figure_id, 'line', dataset={dataset!r}{where}, y={column_name!r})"
            summary = f"added column {column_name!r} to {dataset!r}"
        elif outcome.table is not None:
            new_name = self._analysis_table_name(output, dataset, op.name)
            lengths = {len(v) for v in outcome.table.values()}
            if len(lengths) != 1:
                raise McpError(f"{op.name} produced columns of different lengths.")
            result_table = DataSet(
                new_name,
                [Column(key, np.asarray(v, dtype=np.float64)) for key, v in outcome.table.items()],
            )
            result_table.source = "mcp"
            for column_name, labels in outcome.categories.items():
                if column_name in outcome.table:
                    result_table.categories[column_name] = list(labels)
            replaced = self.datasets.put(result_table)
            data["created"] = {
                "dataset": new_name,
                "columns": list(outcome.table),
                "rows": result_table.n_rows(),
                "replaced": replaced,
            }
            if outcome.plot is not None:
                kind, px, py = outcome.plot
                hint = f"add_plot(figure_id, {kind!r}, dataset={new_name!r}, x={px!r}, y={py!r})"
            summary = f"created dataset {new_name!r} ({result_table.n_rows()} rows)"
        else:
            summary = "report only; nothing was stored"
        if hint:
            data["plot_hint"] = hint
        if notes:
            data["notes"] = notes
        headline = f"{outcome.headline}; " if outcome.headline else ""
        return ToolResult(f"{op.name} of {y!r}: {headline}{summary}.", data)

    @_tool
    def simulate(
        self,
        name: str,
        equations: Mapping[str, str],
        initial: Mapping[str, float],
        t_end: float,
        t_start: float = 0.0,
        steps: int = 1000,
        parameters: Optional[Mapping[str, float]] = None,
        derived: Optional[Mapping[str, str]] = None,
        method: str = "rk45",
        substeps: int = 4,
    ) -> ToolResult:
        """Integrate ordinary differential equations into a dataset: t, each state, derived."""
        check_name(name, "dataset name")
        self.datasets.check_room(name)
        table, report = simulation.simulate(
            name,
            equations,
            initial,
            t_end=t_end,
            t_start=t_start,
            steps=steps,
            parameters=parameters,
            derived=derived,
            method=method,
            substeps=substeps,
            limits=self.limits,
        )
        table.source = "mcp"
        table.mcp_origin = {
            "kind": "simulate",
            "names": table.column_names(),
            "spec": {
                "equations": dict(equations),
                "initial": dict(initial),
                "t_start": t_start,
                "t_end": t_end,
                "steps": steps,
                "parameters": dict(parameters or {}),
                "derived": dict(derived or {}),
                "method": str(method).lower(),
            },
        }
        replaced = self.datasets.put(table)
        states = report["states"]
        hint = f"add_plot(figure_id, 'line', dataset='{name}', x='t', y='{states[0]}')"
        if len(states) > 1:
            hint += (
                f"; phase portrait: add_plot(figure_id, 'line', dataset='{name}', "
                f"x='{states[0]}', y='{states[1]}', reveal=..., head=true)"
            )
        return ToolResult(
            f"Solved {len(states)} equation{'s' if len(states) != 1 else ''} with "
            f"{report['method']} over t = {t_start:g}..{t_end:g} into dataset {name!r} "
            f"({steps} rows).",
            {
                "dataset": name,
                "replaced": replaced,
                "report": report,
                "plot_hint": hint,
                **self.datasets.describe(name, rows=3),
            },
        )

    @_tool
    def select_rows(
        self,
        dataset: str,
        where: Optional[str] = None,
        equals: Optional[Mapping[str, Any]] = None,
        sort_by: Optional[str] = None,
        descending: bool = False,
        every: Optional[int] = None,
        limit: Optional[int] = None,
        output: Optional[str] = None,
    ) -> ToolResult:
        """Keep some rows of a dataset (filter, sort, thin out) as a new dataset."""
        table = self.datasets.get(dataset)
        rows = table.n_rows()
        if where is None and not equals and sort_by is None and every is None and limit is None:
            raise McpError(
                "Nothing to do: pass where=, equals=, sort_by=, every= or limit= "
                "(for example where='temp > 10', equals={'city': 'Lima'})."
            )
        for name, value in (("every", every), ("limit", limit)):
            if value is not None and (not isinstance(value, int) or value < 1):
                raise McpError(f"{name} must be a whole number of at least 1, got {value!r}.")
        names = table.column_names()
        mask = np.ones(rows, dtype=bool)
        steps: List[str] = []

        if where is not None:
            try:
                mask &= table.row_mask(where)
            except ExpressionError as exc:
                raise McpError(f"Could not evaluate where {where!r}: {exc}") from exc
            steps.append(f"where {where}")

        if equals:
            if not isinstance(equals, Mapping):
                raise McpError("equals must be an object like {'city': 'Lima'}.")
            for column_name, wanted in equals.items():
                if column_name not in names:
                    raise McpError(
                        f"equals: no column {column_name!r}.{suggest(str(column_name), names)} "
                        f"Columns: {', '.join(names)}."
                    )
                values = np.asarray(table.get(column_name), dtype=np.float64)
                choices = list(wanted) if isinstance(wanted, (list, tuple)) else [wanted]
                labels = table.categories.get(column_name)
                if labels is not None:
                    codes = []
                    for choice in choices:
                        if not isinstance(choice, str) or choice not in labels:
                            raise McpError(
                                f"equals: {choice!r} is not a label of {column_name!r}."
                                f"{suggest(str(choice), labels)} Labels: {', '.join(labels)}."
                            )
                        codes.append(float(labels.index(choice)))
                    mask &= np.isin(values, codes)
                else:
                    try:
                        numbers = [float(choice) for choice in choices]
                    except (TypeError, ValueError):
                        raise McpError(
                            f"equals: column {column_name!r} is numeric; give numbers, "
                            f"got {choices!r}."
                        ) from None
                    mask &= np.isin(values, numbers)
                steps.append(f"{column_name} in {choices}")

        picked = np.nonzero(mask)[0]
        if picked.size == 0:
            raise McpError(
                f"No rows match ({', '.join(steps) or 'no filter'}); {dataset!r} has {rows} rows. "
                "Check the condition against describe_dataset's min/max per column."
            )
        if sort_by is not None:
            if sort_by not in names:
                raise McpError(
                    f"sort_by: no column {sort_by!r}.{suggest(sort_by, names)} "
                    f"Columns: {', '.join(names)}."
                )
            key = np.asarray(table.get(sort_by), dtype=np.float64)[picked]
            finite = np.isfinite(key)
            order = np.argsort(key[finite], kind="stable")
            if descending:
                # Negating the key keeps ties in their original order (unlike reversing).
                order = np.argsort(-key[finite], kind="stable")
            picked = np.concatenate([picked[finite][order], picked[~finite]])
            steps.append(f"sorted by {sort_by}{' (descending)' if descending else ''}")
        if every is not None and every > 1:
            picked = picked[::every]
            steps.append(f"every {every}th row")
        if limit is not None:
            picked = picked[:limit]
            steps.append(f"first {limit}")

        name = (
            check_name(output, "output dataset name")
            if output
            else self.datasets.unique_name(check_name(f"{dataset}_rows"[:64], "output name"))
        )
        kept = DataSet(
            name,
            [Column(c.name, np.asarray(c.values, dtype=np.float64)[picked]) for c in table.columns],
        )
        kept.source = "mcp"
        kept.categories = {key: list(labels) for key, labels in table.categories.items()}
        kept.date_columns = dict(table.date_columns)
        replaced = self.datasets.put(kept)
        return ToolResult(
            f"Kept {picked.size} of {rows} rows of {dataset!r} ({'; '.join(steps)}) as dataset "
            f"{name!r}.",
            {
                "dataset": name,
                "rows_in": rows,
                "rows_out": int(picked.size),
                "steps": steps,
                "replaced": replaced,
                **self.datasets.describe(name, rows=3),
            },
        )

    @staticmethod
    def _identifier(text: str) -> str:
        """``text`` as a column name an expression can use (spaces and symbols become ``_``)."""
        import re

        cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_") or "col"
        return cleaned if not cleaned[0].isdigit() else f"c_{cleaned}"

    def _analysis_column_name(
        self, output: Optional[str], y: str, x_name: str, outcome: Any, op: Any
    ) -> str:
        if output is not None:
            if not isinstance(output, str) or not is_identifier(output):
                raise McpError(
                    f"output {output!r} must be a column name made of letters, digits and '_'."
                )
            name = output
        else:
            name = f"{self._identifier(y)}_{outcome.column_suffix or op.suffix or op.name}"
        if name in (y, x_name):
            raise McpError(
                f"The result column would overwrite the input column {name!r}. "
                "Pass output=... to choose another name."
            )
        return name

    def _analysis_table_name(self, output: Optional[str], dataset: str, operation: str) -> str:
        if output is not None:
            return check_name(output, "output dataset name")
        return self.datasets.unique_name(check_name(f"{dataset}_{operation}"[:64], "output name"))

    @_tool
    def delete_dataset(self, name: str) -> ToolResult:
        """Forget a dataset. Figures already drawn from it are unaffected."""
        self.datasets.remove(name)
        return ToolResult(f"Deleted dataset {name!r}.", {"datasets": self.datasets.names()})

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    @_tool
    @_scene_op
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
        hatch: Optional[str] = None,
        axis: str = "left",
        reveal: Optional[float] = None,
        trail: Optional[int] = None,
        head: bool = False,
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
            "hatch": hatch,
        }
        if axis not in ("left", "right"):
            raise McpError("axis must be 'left' or 'right' (a second y axis on the right).")
        if axis == "right" and (entry.is_3d or kind not in _TWIN_KINDS):
            raise McpError(
                "axis='right' draws on a second y axis; it is for 2-D "
                f"{', '.join(sorted(_TWIN_KINDS))} (this is kind={kind!r} in "
                f"{'a 3-D' if entry.is_3d else 'a 2-D'} figure)."
            )
        motion_notes = self._apply_motion(req, kind, reveal, trail)
        req.tick_labels = list(tick_labels) if tick_labels else None
        notes = self._category_labels(req, dataset, kind, x, y, color_by, tick_labels)
        notes.extend(motion_notes)
        req.color_label = color_by if isinstance(color_by, str) else None
        req.colorbar = bool(colorbar)

        self._activate(entry, panel)
        req.api = self._axes_api(entry, panel, axis)
        if head:
            if kind not in ("line", "scatter", "step"):
                raise McpError(
                    "head=true puts a dot on the last point: use kind line, scatter or step."
                )
            req.head = True
        if color is None and req.color_by is None and kind in _CYCLED_KINDS:
            # GLPlot draws a bare scatter or errorbar in black; matplotlib walks its colour
            # cycle. A model overlaying two series needs them told apart without having to
            # name a colour for each, so take the figure's next cycle colour (the one
            # ``plot()`` is already advancing) rather than the black default.
            req.options["color"] = gplt._next_cycle_color(entry.plot)
            self._resolved["color"] = req.options["color"]
        if kind == "area" and alpha is None:
            # A translucent fill is exported as one outlined quad per sample, and the
            # overlapping outlines show as vertical hatching. Opaque is seamless.
            req.options["alpha"] = 1.0
            self._resolved["alpha"] = 1.0
        artist = self._record(entry, panel, kind, label or "", lambda: _DRAW[kind](req), req)
        if kind == "pie":
            entry.equal_panels.add(panel)
        artist.dataset = dataset
        note = f" Ignored for kind={kind!r}: {', '.join(req.ignored)}." if req.ignored else ""
        data: Dict[str, Any] = {"artist_id": artist.artist_id, "layers": len(artist.layer_ids)}
        if req.ignored:
            data["ignored"] = req.ignored
        if notes:
            data["notes"] = notes
        return ToolResult(
            f"Added {kind} {artist.artist_id} ({req.n} points) to {figure_id}.{note}", data
        )

    def _apply_motion(
        self, req: "_PlotRequest", kind: str, reveal: Optional[float], trail: Optional[int]
    ) -> List[str]:
        """Show only part of the data: the first ``reveal`` fraction, the last ``trail`` of that.

        This is what makes a curve draw itself in, or a moving point drag a tail behind it, when
        the numbers change from frame to frame; used on its own it is a partial plot.
        """
        if reveal is None and trail is None:
            return []
        if kind not in _XY_KINDS:
            raise McpError(
                f"reveal/trail cut a series short, so they need an x-vs-y kind "
                f"({', '.join(sorted(_XY_KINDS))}); kind={kind!r} reads one set of values."
            )
        total = req.n
        count = total
        # plot3d needs two points to make a line, so a 3-D line never shows fewer (reveal=0 and
        # trail=1 then show the first segment, which is how a curve starts to draw itself)
        floor = 2 if kind == "line" and req.z is not None and total >= 2 else 1
        if reveal is not None:
            if isinstance(reveal, bool) or not isinstance(reveal, (int, float)):
                raise McpError("reveal is a fraction between 0 and 1 of the points to show.")
            if not math.isfinite(reveal) or not 0.0 <= reveal <= 1.0:
                raise McpError(f"reveal must be between 0 and 1, got {reveal!r}.")
            count = max(floor, min(total, int(math.ceil(reveal * total - 1e-9))))
        first = 0
        if trail is not None:
            if isinstance(trail, bool) or not isinstance(trail, int) or trail < 1:
                raise McpError(
                    "trail is a whole number of points (at least 1) kept behind the head."
                )
            first = max(0, count - max(trail, floor))
        if first == 0 and count == total:
            return []
        for name in ("x", "y", "z", "error", "y2", "color_by", "size_by"):
            values = getattr(req, name)
            if values is not None and values.size == total:
                setattr(req, name, values[first:count])
        if req.tick_labels and len(req.tick_labels) == total:
            req.tick_labels = req.tick_labels[first:count]
        req.n = count - first
        req.visible = (first, count, total)
        return []

    def _axes_api(self, entry: _Figure, panel: int, axis: str) -> Any:
        """What drawing calls go to: pyplot itself, or the panel's right-hand twin axes."""
        if axis != "right":
            return gplt
        twin = entry.twins.get(panel)
        if twin is None:
            twin = entry.twins[panel] = gplt.twinx()
        return twin

    def _category_labels(
        self,
        req: Any,
        dataset: Optional[str],
        kind: str,
        x: SeriesRef,
        y: SeriesRef,
        color_by: SeriesRef,
        tick_labels: Optional[Sequence[str]],
    ) -> List[str]:
        """Use the labels of a category column as tick/slice names; say what the other uses mean.

        A category column stores the *index* of each row's label. For a bar or pie that index is
        only a position, so its labels become the tick/slice names automatically; anywhere else the
        numbers are what is plotted and the reply says which label each one stands for.
        """
        if dataset is None:
            return []
        table = self.datasets.get(dataset)
        notes: List[str] = []

        def labels_of(ref: SeriesRef) -> Optional[List[str]]:
            return table.categories.get(ref) if isinstance(ref, str) else None

        def per_point(labels: List[str], values: np.ndarray) -> List[str]:
            return [
                labels[int(v)] if np.isfinite(v) and 0 <= int(v) < len(labels) else ""
                for v in values
            ]

        x_labels = labels_of(x)
        if tick_labels is None and x_labels and kind in ("bar", "barh") and req.x is not None:
            req.tick_labels = per_point(x_labels, req.x)
            notes.append(f"Tick labels taken from the categories of {x!r}.")
            x_labels = None
        elif tick_labels is None and x_labels and kind == "pie":
            values = self._series(x, dataset, "x", required=True)
            if values.size == req.y.size:
                req.tick_labels = per_point(x_labels, values)
                notes.append(f"Slice names taken from the categories of {x!r}.")
                x_labels = None
        for role, ref in (("x", x), ("y", y), ("color_by", color_by)):
            labels = x_labels if role == "x" else labels_of(ref)
            if labels:
                shown = ", ".join(f"{i}={label}" for i, label in enumerate(labels[:8]))
                notes.append(
                    f"{role} column {ref!r} is categorical: its values are label indices "
                    f"({shown}{', ...' if len(labels) > 8 else ''})."
                )
        return notes

    @_tool
    @_scene_op
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
        reveal: Optional[float] = None,
        head: bool = False,
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
            reveal=reveal,
            head=head,
        )
        outcome.message = (
            f"Plotted y = {expression} on [{x_min:g}, {x_max:g}] as {outcome.data['artist_id']}."
        )
        return outcome

    @_tool
    @_scene_op
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
    @_scene_op
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
    @_scene_op
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
        dx: Optional[float] = None,
        dy: Optional[float] = None,
        alpha: Optional[float] = None,
        reveal: Optional[float] = None,
        coords: str = "data",
        align: str = "left",
        background: bool = False,
    ) -> ToolResult:
        """Add a text note, a horizontal or vertical line, or an arrow (a vector)."""
        entry = self._figure(figure_id)
        if kind not in ANNOTATION_KINDS:
            raise McpError(f"Unknown kind {kind!r}. Use one of: {', '.join(ANNOTATION_KINDS)}.")
        if entry.is_3d:
            raise McpError("Annotations are 2-D only; this is a 3-D figure.")
        if coords not in ("data", "axes"):
            raise McpError("coords is 'data' (x, y in data units) or 'axes' (0..1 of the box).")
        if coords == "axes" and kind != "text":
            raise McpError("coords='axes' places text; lines and arrows are in data units.")
        if align not in ("left", "center", "right"):
            raise McpError("align is 'left', 'center' or 'right'.")
        for name, value in (("alpha", alpha), ("reveal", reveal)):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= value <= 1.0
            ):
                raise McpError(f"{name} is a number between 0 and 1.")
        if background and kind != "text":
            raise McpError("background puts a box behind text; it applies to kind='text'.")
        if reveal is not None and kind not in ("text", "arrow"):
            raise McpError("reveal applies to text (letters written so far) and arrows (grown).")
        self._activate(entry, panel)
        # Every annotation gets an explicit colour: the preset's ink when none is given (a black
        # default vanishes on a dark page), an arrow the next colour of the cycle like a curve.
        # Recorded so an exported script draws the same thing.
        if color is None:
            if kind == "arrow":
                base = gplt._next_cycle_color(entry.plot)
            else:
                ink = gplt._style_ink(entry.plot)
                base = mcolors.to_hex(ink[:3]) if ink is not None else "#000000"
            self._resolved["color"] = base
        else:
            base = color
        try:
            rgba = list(mcolors.to_rgba(base))
        except ValueError as exc:
            raise McpError(f"color {base!r} is not a colour: {exc}") from exc
        if alpha is not None:
            rgba[3] *= float(alpha)
        shade = tuple(rgba)
        style = {
            k: v
            for k, v in (
                ("color", shade),
                ("linestyle", linestyle),
                ("linewidth", linewidth),
                ("label", label),
            )
            if v is not None
        }

        if kind == "text":
            if x is None or y is None or not text:
                raise McpError("kind='text' needs x, y and text.")
            shown = dynamics.reveal_text(str(text), 1.0 if reveal is None else float(reveal))
            text_style: Dict[str, Any] = {"color": shade}
            if align != "left":  # left is what text() does anyway
                text_style["ha"] = align
            if reveal is not None:  # written so far: placed where it sits in the whole text
                text_style["layout_text"] = str(text)
            if background:
                # a box in the page colour hides the grid and axis lines behind the text
                page = mcolors.to_hex(tuple(entry.plot.options.visual.background_color)[:3])
                self._resolved["page"] = page
                text_style["bbox"] = {"facecolor": page, "edgecolor": "none", "pad": 4}
            if fontsize is not None:
                text_style["fontsize"] = fontsize
            if coords == "axes":
                text_style["transform"] = gplt.gca().transAxes
            draw = lambda: gplt.text(float(x), float(y), shown, **text_style)  # noqa: E731
        elif kind == "hline":
            if y is None:
                raise McpError("kind='hline' needs y.")
            draw = lambda: gplt.axhline(float(y), **style)  # noqa: E731
        elif kind == "vline":
            if x is None:
                raise McpError("kind='vline' needs x.")
            draw = lambda: gplt.axvline(float(x), **style)  # noqa: E731
        else:
            if x is None or y is None or dx is None or dy is None:
                raise McpError("kind='arrow' needs x, y (the tail) and dx, dy (tail to tip).")
            grow = 1.0 if reveal is None else float(reveal)
            ux, uy = float(dx) * grow, float(dy) * grow
            length = math.hypot(ux, uy)
            # a head in proportion to the arrow, like manim's (at most a quarter of its length)
            head = 0.22 * length
            arrow_style = {"color": shade, "width": linewidth if linewidth is not None else 3.0}
            if label is not None:
                arrow_style["label"] = label

            def draw() -> Any:
                if length <= 1e-12:  # fully shrunk: nothing to draw, but keep the artist
                    return gplt.plot([float(x)], [float(y)], color=(0, 0, 0, 0))
                return gplt.arrow(
                    float(x),
                    float(y),
                    ux,
                    uy,
                    head_length=head,
                    head_width=0.6 * head,
                    **arrow_style,
                )

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
        gone = {a.artist_id for a in targets}
        entry.log = [op for op in entry.log if op.artist_id not in gone]
        entry.bindings = [b for b in entry.bindings if b.target not in gone]
        remaining = [a.artist_id for a in entry.artists]
        return ToolResult(
            f"Removed {len(targets)} artist{'s' if len(targets) != 1 else ''} from {figure_id}.",
            {"remaining": remaining},
        )

    @_tool
    def update_artist(
        self, figure_id: str, artist_id: str, changes: Mapping[str, Any]
    ) -> ToolResult:
        """Change something already drawn: colour, width, data columns, label... or refresh it.

        ``changes`` holds arguments of the call that drew the artist (see its ``recipe`` in
        ``describe_figure``); a ``None`` value puts that argument back to its default; a dotted
        name reaches inside an object argument (``"parameters.a"``). An empty ``changes``
        redraws the artist from the datasets as they are now.
        """
        entry = self._figure(figure_id)
        index = next((i for i, op in enumerate(entry.log) if op.artist_id == artist_id), None)
        if index is None:
            known = ", ".join(a.artist_id for a in entry.artists) or "none"
            raise McpError(f"{figure_id} has no artist {artist_id!r} (artists: {known}).")
        if not isinstance(changes, Mapping):
            raise McpError("changes must be an object like {'color': 'tab:red', 'linewidth': 3}.")
        op = entry.log[index]
        allowed = [
            name
            for name in inspect.signature(getattr(Toolbox, op.tool)).parameters
            if name not in ("self", "figure_id")
        ]
        revised = copy.deepcopy(op)
        for path, value in changes.items():
            head = str(path).split(".")[0]
            if head not in allowed:
                raise McpError(
                    f"{artist_id} was drawn by {op.tool}, which has no argument {head!r}."
                    f"{suggest(head, allowed)} Arguments: {', '.join(allowed)}."
                )
            self._check_path(op.tool, str(path))
            if value is None:
                revised.args.pop(path, None)
            else:
                dynamics.set_path(revised.args, str(path), value)
        ops = list(entry.log)
        ops[index] = revised
        self._refresh(entry, ops)
        artist = next(a for a in entry.artists if a.artist_id == artist_id)
        return ToolResult(
            f"Updated {artist_id} of {figure_id}"
            + (f" ({', '.join(map(str, changes))})." if changes else " (redrawn from the data)."),
            {"artist": artist.describe(), "recipe": {"tool": op.tool, **revised.args}},
        )

    # ------------------------------------------------------------------
    # Controls and bindings: parameters the figure depends on
    # ------------------------------------------------------------------

    @_tool
    def add_control(
        self,
        figure_id: str,
        name: str,
        kind: str = "slider",
        min: Optional[float] = None,  # noqa: A002 - the argument name a model expects
        max: Optional[float] = None,  # noqa: A002
        value: Any = None,
        step: Optional[float] = None,
        options: Optional[Sequence[str]] = None,
        label: Optional[str] = None,
        integer: bool = False,
        log: bool = False,
        sets: Optional[Mapping[str, Any]] = None,
    ) -> ToolResult:
        """Define a slider, toggle, choice or button: a named value expressions can use."""
        entry = self._figure(figure_id)
        control = dynamics.make_control(
            name,
            kind,
            label=label,
            minimum=min,
            maximum=max,
            value=value,
            step=step,
            integer=integer,
            log=log,
            options=options,
            sets=sets,
        )
        if name not in entry.controls and len(entry.controls) >= self.limits.animation.max_controls:
            raise McpError(
                f"{figure_id} already has {self.limits.animation.max_controls} controls."
            )
        if control.kind == "button" and control.sets:
            for target in control.sets:
                if target not in entry.controls:
                    raise McpError(
                        f"The button sets {target!r}, which is not a control of {figure_id}."
                        f"{suggest(target, entry.controls)} Add that control first."
                    )
        replaced = name in entry.controls
        entry.controls[name] = control
        return ToolResult(
            f"{'Replaced' if replaced else 'Added'} {control.kind} {name!r} on {figure_id}. "
            "Use its name in expressions (bind_property, animate).",
            {
                "control": control.describe(),
                "controls": [c.describe() for c in entry.controls.values()],
            },
        )

    @_tool
    def set_control(self, figure_id: str, name: str, value: Any = None) -> ToolResult:
        """Move a control (the figure is redrawn at the new value), or press a button."""
        entry = self._figure(figure_id)
        control = entry.controls.get(name)
        if control is None:
            known = ", ".join(entry.controls) or "none -- add one with add_control"
            raise McpError(
                f"{figure_id} has no control {name!r}.{suggest(name, entry.controls)} "
                f"Controls: {known}."
            )
        before = {n: c.value for n, c in entry.controls.items()}
        if control.kind == "button":
            targets = (
                control.sets
                if control.sets is not None
                else {n: c.default for n, c in entry.controls.items() if c.kind != "button"}
            )
            for target, raw in targets.items():
                other = entry.controls.get(target)
                if other is None:
                    raise McpError(f"The button sets {target!r}, which no longer exists.")
                other.value = dynamics.set_control_value(other, raw)
        else:
            if value is None:
                raise McpError(f"Give the new value of {name!r}.")
            control.value = dynamics.set_control_value(control, value)
        try:
            self._refresh(entry)
        except McpError:
            for n, v in before.items():
                entry.controls[n].value = v
            raise
        changed = {n: c.value for n, c in entry.controls.items() if before.get(n) != c.value}
        return ToolResult(
            f"{'Pressed' if control.kind == 'button' else 'Set'} {name!r}"
            + (f": {changed}." if changed else " (nothing changed)."),
            {"changed": changed, "controls": [c.describe() for c in entry.controls.values()]},
        )

    @_tool
    def bind_property(
        self, figure_id: str, target: str, property: str, value: Any = None  # noqa: A002
    ) -> ToolResult:
        """Make a property follow an expression of the controls (and, in animate, of time)."""
        entry = self._figure(figure_id)
        kind, ident = dynamics.parse_target(target)
        if not isinstance(property, str) or not property:
            raise McpError(
                "property is the name to drive: 'linewidth', 'xlim', 'azim', 'parameters.a'."
            )
        previous = list(entry.bindings)
        entry.bindings = [
            b for b in entry.bindings if not (b.target == target and b.prop == property)
        ]
        if value is None:
            removed = len(previous) - len(entry.bindings)
            if not removed:
                raise McpError(f"{target}.{property} has no binding to remove.")
        else:
            if len(entry.bindings) >= self.limits.animation.max_bindings:
                entry.bindings = previous
                raise McpError(
                    f"{figure_id} already has {self.limits.animation.max_bindings} bindings."
                )
            entry.bindings.append(dynamics.Binding(target, property, copy.deepcopy(value)))
        try:
            self._refresh(entry)
        except BaseException:
            entry.bindings = previous
            raise
        return ToolResult(
            (
                f"{target}.{property} now follows {value!r}."
                if value is not None
                else f"Removed the binding of {target}.{property}."
            )
            + (" Move controls with set_control; animate uses them too." if entry.controls else ""),
            {"bindings": [b.describe() for b in entry.bindings], "kind": kind, "target": ident},
        )

    # ------------------------------------------------------------------
    # Axes and style
    # ------------------------------------------------------------------

    @_tool
    @_scene_op
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
        y2label: Optional[str] = None,
        y2lim: Optional[Sequence[float]] = None,
        y2scale: Optional[str] = None,
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
        # Scales first: switching one re-fits the view and would throw away a limit set before it.
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
        if grid is not None:
            do("grid", gplt.grid, bool(grid))
        if legend:
            do("legend", gplt.legend)
            entry.legend_off.discard(panel)
        elif legend is False:
            do("legend", lambda: setattr(gplt.gcf(), "legend_show", False))
            entry.legend_off.add(panel)
        if equal_aspect is not None:
            do("equal_aspect", gplt.axis, "equal" if equal_aspect else "auto")
            (entry.equal_panels.add if equal_aspect else entry.equal_panels.discard)(panel)
        if hide_axes is not None:
            do("hide_axes", gplt.axis, "off" if hide_axes else "on")
        if elev is not None or azim is not None:
            if not entry.is_3d:
                raise McpError("elev/azim set the 3-D camera; this is a 2-D figure.")
            do("view", lambda: gplt.view_init(elev=elev, azim=azim))
        if y2label is not None or y2lim is not None or y2scale is not None:
            twin = entry.twins.get(panel)
            if twin is None:
                raise McpError(
                    f"Panel {panel} has no right-hand y axis yet: draw something on it first with "
                    "add_plot(..., axis='right')."
                )
            if y2label is not None:
                do("y2label", twin.set_ylabel, y2label)
            if y2scale is not None:
                if y2scale not in ("linear", "log", "symlog", "logit"):
                    raise McpError("y2scale must be one of: linear, log, symlog, logit.")
                do("y2scale", twin.set_yscale, y2scale)
            if y2lim is not None:
                lo, hi = self._range(y2lim, "y2lim")
                do("y2lim", twin.set_ylim, lo, hi)
        if not applied:
            raise McpError("Nothing to set: pass at least one of title, xlabel, ylabel, xlim, ...")
        return ToolResult(
            f"Updated panel {panel} of {figure_id}: {', '.join(applied)}.", {"applied": applied}
        )

    @_tool
    @_scene_op
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
    def render_figure(
        self, figure_id: str, scale: float = 1.0, params: Optional[Mapping[str, Any]] = None
    ) -> ToolResult:
        """Render the figure to a PNG the model can look at, with a text summary of the axes."""
        entry = self._figure(figure_id)
        if not 0.25 <= float(scale) <= 3.0:
            raise McpError("scale must be between 0.25 and 3.0.")
        shown = entry
        if params:
            shown = self._build_for(entry, params=params)
        try:
            data, view = self._render(
                shown, float(scale), "png", self.limits.max_image_edge, review=True
            )
            found = shown.issues
        finally:
            if shown is not entry:
                self._release(shown)
        worst = sum(1 for item in found if item["severity"] == "warning")
        note = (
            f" {len(found)} thing{'s' if len(found) != 1 else ''} to consider"
            + (f" ({worst} warning{'s' if worst != 1 else ''})" if worst else "")
            + ": see issues."
            if found
            else " No issues found."
        )
        return ToolResult(
            f"Rendered {figure_id}.{note}",
            {
                "figure_id": figure_id,
                "artists": len(entry.artists),
                "panels": view,
                "issues": found,
            },
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
        params: Optional[Mapping[str, Any]] = None,
    ) -> ToolResult:
        """Save the figure as PNG, SVG, PDF or JPEG inside the server's output directory."""
        entry = self._figure(figure_id)
        if not 0.25 <= float(scale) <= 4.0:
            raise McpError("scale must be between 0.25 and 4.0.")
        target = self.sandbox.write_path(path, overwrite=bool(overwrite))
        fmt = target.suffix.lower().lstrip(".")
        fmt = "jpeg" if fmt == "jpg" else fmt
        shown = self._build_for(entry, params=params) if params else entry
        try:
            data, _ = self._render(shown, float(scale), fmt, 6000, transparent=bool(transparent))
        finally:
            if shown is not entry:
                self._release(shown)
        target.write_bytes(data)
        return ToolResult(
            f"Saved {figure_id} to {target} ({len(data) / 1024:.0f} KiB).",
            {"path": str(target), "bytes": len(data)},
        )

    @_tool
    def animate(
        self,
        figure_id: str,
        output: Optional[str] = None,
        frames: Optional[int] = None,
        duration: Optional[float] = None,
        fps: float = 30.0,
        loop: bool = False,
        stages: Optional[Sequence[Mapping[str, Any]]] = None,
        variables: Optional[Mapping[str, Any]] = None,
        bindings: Optional[Sequence[Mapping[str, Any]]] = None,
        scale: Optional[float] = None,
        hold_start: float = 0.0,
        hold_end: float = 0.0,
        sheet_frames: int = 8,
        frame: Optional[int] = None,
        overwrite: bool = False,
        repeat: int = 0,
        quality: str = "good",
    ) -> ToolResult:
        """Animate a figure: properties follow expressions of time, across stages. See the guide."""
        entry = self._figure(figure_id)
        # stage effects (create, write, fade_in, indicate...) become stage variables + bindings
        stages, variables, bindings, plan = choreography.compile_effects(
            self, entry, stages, variables, bindings
        )
        keep: Dict[str, Any] = {}
        data, image, headline = animator.run_animation(
            self,
            entry,
            output=output,
            frames=frames,
            duration=duration,
            fps=fps,
            loop=loop,
            stages=stages,
            variables=variables,
            bindings=bindings,
            scale=scale,
            hold_start=hold_start,
            hold_end=hold_end,
            sheet_frames=sheet_frames,
            frame=frame,
            overwrite=overwrite,
            repeat=repeat,
            quality=quality,
            keep=keep if frame is None else None,
        )
        if frame is None:
            entry.animation = {
                "frames": frames,
                "duration": duration,
                "fps": fps,
                "loop": loop,
                "stages": copy.deepcopy(stages),
                "variables": copy.deepcopy(variables),
                "bindings": copy.deepcopy(bindings),
                "hold_start": hold_start,
                "hold_end": hold_end,
            }
            if plan:
                entry.animation["effects"] = copy.deepcopy(plan["stages"])
            self._remember_frames(entry, keep)
        if plan:
            data["effects"] = plan["stages"]
            if plan["notes"]:
                data.setdefault("notes", []).extend(plan["notes"])
        return ToolResult(headline, data, image=image)

    @_tool
    def compose_video(
        self,
        segments: Sequence[Any],
        output: Optional[str] = None,
        fps: Optional[float] = None,
        transition: str = "fade",
        transition_s: float = 0.6,
        size: Optional[Sequence[int]] = None,
        audio: Optional[str] = None,
        quality: str = "good",
        repeat: int = 0,
        overwrite: bool = False,
        sheet_frames: int = 12,
    ) -> ToolResult:
        """Cut several animated figures (the scenes) together into one video."""
        data, image, headline = video.compose(
            self,
            segments,
            output=output,
            fps=fps,
            transition=transition,
            transition_s=transition_s,
            size=size,
            audio=audio,
            quality=quality,
            repeat=repeat,
            overwrite=overwrite,
            sheet_frames=sheet_frames,
        )
        return ToolResult(headline, data, image=image)

    # -- rendered frames, kept between animate and compose_video ---------------------------

    def _signature(self, entry: _Figure) -> str:
        """A digest of everything a figure's animation frames depend on.

        The calls that drew it, its animation, its controls and bindings, and the contents of
        every dataset (a replaced dataset changes what the same calls draw).
        """
        import hashlib

        digest = hashlib.sha256()

        def feed(value: Any) -> None:
            if isinstance(value, np.ndarray):
                digest.update(str(value.dtype).encode() + str(value.shape).encode())
                digest.update(np.ascontiguousarray(value).tobytes())
            elif isinstance(value, Mapping):
                for key in sorted(value, key=str):
                    digest.update(repr(key).encode())
                    feed(value[key])
            elif isinstance(value, (list, tuple)):
                digest.update(b"[")
                for item in value:
                    feed(item)
                digest.update(b"]")
            else:
                digest.update(repr(value).encode())

        feed([(op.tool, op.args, op.artist_id, op.resolved) for op in entry.log])
        feed(entry.spec)
        feed(entry.animation)
        feed({name: control.value for name, control in entry.controls.items()})
        feed([binding.describe() for binding in entry.bindings])
        for name in sorted(self.datasets.names()):
            table = self.datasets.get(name)
            for column in table.columns:
                feed([name, column.name, np.asarray(column.values)])
        return digest.hexdigest()

    def _remember_frames(self, entry: _Figure, keep: Mapping[str, Any]) -> None:
        """Keep what ``animate`` rendered, within the server's frame-cache budget (LRU)."""
        frames = list(keep.get("frames") or [])
        if not frames:
            return
        size = sum(len(f) for f in frames)
        budget = self.limits.animation.max_frame_cache_bytes
        entry.rendered = None
        if entry.figure_id in self._frame_order:
            self._frame_order.remove(entry.figure_id)
        if size > budget:
            return
        used = size + sum(
            f.rendered["bytes"] for f in self._figures.values() if f.rendered is not None
        )
        while used > budget and self._frame_order:
            oldest = self._figures.get(self._frame_order.pop(0))
            if oldest is not None and oldest.rendered is not None:
                used -= oldest.rendered["bytes"]
                oldest.rendered = None
        entry.rendered = {
            "signature": self._signature(entry),
            "frames": frames,
            "fps": float(keep["fps"]),
            "bytes": size,
        }
        self._frame_order.append(entry.figure_id)

    def _cached_frames(self, entry: _Figure) -> Optional[Tuple[List[bytes], float]]:
        """The frames ``animate`` kept for ``entry``, if the figure has not changed since."""
        kept = entry.rendered
        if kept is not None and kept["signature"] == self._signature(entry):
            return kept["frames"], kept["fps"]
        return None

    def _frames_for(
        self, entry: _Figure, started: float, budget_s: float
    ) -> Tuple[List[bytes], float, str]:
        """The frames of ``entry``'s last animation: kept ones if still valid, else rendered."""
        kept = self._cached_frames(entry)
        if kept is not None:
            return kept[0], kept[1], "cached"
        import time as clock

        if clock.monotonic() - started > budget_s:
            raise McpError(
                f"compose_video has used its {budget_s:.0f} s rendering budget before "
                f"{entry.figure_id}. animate each scene first (its frames are kept), then compose."
            )
        spec = {k: v for k, v in entry.animation.items() if k != "effects"}
        keep: Dict[str, Any] = {}
        animator.run_animation(
            self,
            entry,
            output=None,
            frames=spec.get("frames"),
            duration=spec.get("duration"),
            fps=spec.get("fps", 30.0),
            loop=bool(spec.get("loop")),
            stages=spec.get("stages"),
            variables=spec.get("variables"),
            bindings=spec.get("bindings"),
            scale=None,
            hold_start=spec.get("hold_start", 0.0),
            hold_end=spec.get("hold_end", 0.0),
            sheet_frames=1,
            frame=None,
            overwrite=False,
            repeat=0,
            quality="good",
            keep=keep,
        )
        self._remember_frames(entry, keep)
        return list(keep["frames"]), float(keep["fps"]), "rendered"

    def _still_frame(self, entry: _Figure) -> bytes:
        """One picture of a figure with no animation, at about an animation frame's size."""
        return animator.still_frame(self, entry)

    @_tool
    def export_script(self, figure_id: str, kind: str = "figure") -> ToolResult:
        """The Python script that rebuilds this figure (also interactive, or as an animation)."""
        entry = self._figure(figure_id)
        text, notes = scripting.export_script(self, entry, kind)
        limit = 120_000
        if len(text) > limit:
            raise McpError(
                f"The script is {len(text):,} characters, over the {limit:,} this tool returns "
                "(inline data is the usual cause). Draw from expressions or smaller datasets."
            )
        lines = text.count("\n")
        data: Dict[str, Any] = {"kind": kind, "lines": lines}
        if notes:
            data["notes"] = notes
        return ToolResult(
            f"Python script for {figure_id} ({kind}, {lines} lines). Save it as a .py file and run "
            f"it with python:\n```python\n{text}```",
            data,
        )

    # ------------------------------------------------------------------
    # Knowledge: what GLPlot can do and how to use it
    #
    # Read-only and SDK-free, so a client that lists tools but never reads resources (most do)
    # can still learn the whole library, and the built-in stdio server serves them unchanged.
    # ------------------------------------------------------------------

    #: Longest reply these tools return, in characters. A docstring or guide is bounded in practice;
    #: this is the guard that keeps one call from flooding a model's context.
    MAX_KNOWLEDGE_CHARS = 60_000

    def _guide_topics(self) -> List[Any]:
        from . import guide as guide_module

        return guide_module.mcp_topics(self.sandbox, self.limits)

    @_tool
    def get_guide(self, topic: Optional[str] = None) -> ToolResult:
        """A guide topic as Markdown, or the list of topics when none is named."""
        from .. import llm

        extra = self._guide_topics()
        text = llm.guide(topic, extra)
        if len(text) > self.MAX_KNOWLEDGE_CHARS:
            text = text[: self.MAX_KNOWLEDGE_CHARS] + "\n\n[truncated]"
        listing = [t.id for t in llm.guides.list_topics(tuple(extra))]
        return ToolResult(text, {"topic": topic or None, "topics": listing} if not topic else {})

    @_tool
    def search_api(
        self, query: str = "", limit: int = 8, category: Optional[str] = None
    ) -> ToolResult:
        """Find GLPlot Python functions by name or what they do; no query lists categories."""
        from .. import llm

        limit = int(limit)
        if not 1 <= limit <= 40:
            raise McpError("limit must be between 1 and 40.")
        valid = [c["id"] for c in llm.api_categories()]
        if category and category not in valid:
            raise McpError(
                f"Unknown category {category!r}.{suggest(category, valid)} "
                f"Valid: {', '.join(valid)}."
            )
        if not str(query).strip():
            cats = llm.api_categories()
            lines = [f"{c['title']} [{c['id']}]: {', '.join(c['names'])}" for c in cats]
            return ToolResult(
                "GLPlot API by category (pass words to search, or a name to get_api_doc):\n"
                + "\n".join(lines),
                {"categories": [{"id": c["id"], "count": len(c["names"])} for c in cats]},
            )
        hits = llm.search_api(query, limit=limit, category=category)
        if not hits:
            return ToolResult(
                f"No function matches {query!r}. Try other words, a matplotlib name, or call "
                "search_api with no query to list every category.",
                {"matches": []},
            )
        rows = [f"{e.name}{e.brief} -- {e.summary}" for e in hits]
        return ToolResult(
            f"{len(hits)} match(es) for {query!r}; call get_api_doc(name) for the full docstring.\n"
            + "\n".join(rows),
            {"matches": [{"name": e.name, "category": e.category, "usage": e.usage} for e in hits]},
        )

    @_tool
    def get_api_doc(self, name: str) -> ToolResult:
        """The signature and full docstring of one GLPlot function or class."""
        from .. import llm

        try:
            info = llm.describe_api(name)
        except KeyError as exc:
            raise McpError(str(exc)) from exc
        doc = str(info["doc"]) or "(no docstring)"
        text = f"{info['usage']}\n\ncategory: {info['category']}\n\n{doc}"
        if len(text) > self.MAX_KNOWLEDGE_CHARS:
            text = text[: self.MAX_KNOWLEDGE_CHARS] + "\n\n[truncated]"
        return ToolResult(
            text,
            {"name": info["name"], "category": info["category"], "signature": info["signature"]},
        )

    @_tool
    def get_example(self, name: Optional[str] = None) -> ToolResult:
        """A runnable Python example script, or the list of examples when none is named."""
        from .. import llm

        if not name:
            items = llm.examples()
            lines = [f"{e['name']}: {e['title']}" for e in items]
            return ToolResult(
                "Runnable GLPlot scripts (each ends in plt.savefig, so it works with no display). "
                "Call get_example(name) for the code.\n" + "\n".join(lines),
                {"examples": [e["name"] for e in items]},
            )
        try:
            item = llm.example(name)
        except KeyError as exc:
            raise McpError(str(exc)) from exc
        return ToolResult(
            f"{item['title']}\n\n{item['description']}\n\n```python\n{item['code']}```",
            {"name": item["name"], "filename": item["filename"]},
        )

    #: Set by whichever server wraps this toolbox ("mcp-sdk" or "builtin").
    server_flavor: str = "unknown"

    @_tool
    def server_info(self) -> ToolResult:
        """Versions, what this server may do (files, limits), and what is open right now."""
        from .. import llm
        from . import analysis as analysis_module

        info = llm.capabilities()
        files = (
            {"enabled": True, "roots": [str(r) for r in self.sandbox.roots]}
            if self.sandbox.enabled
            else {"enabled": False, "note": "started without --root: no file reads or writes"}
        )
        info["server"] = {
            "implementation": self.server_flavor,
            "files": files,
            "plot_kinds": list(PLOT_KINDS),
            "field_kinds": list(FIELD_KINDS),
            "annotation_kinds": list(ANNOTATION_KINDS),
            "analysis_operations": [op.name for op in analysis_module.OPERATIONS],
            "animation": {
                "formats": {
                    name: ("available" if why is None else why)
                    for name, why in motion.available_formats().items()
                },
                "easings": list(dynamics.EASINGS),
                "functions": list(dynamics.HELPERS),
                "max_frames": self.limits.animation.max_frames,
                "max_stages": self.limits.animation.max_stages,
                "max_seconds": self.limits.animation.max_seconds,
                "ode_methods": list(simulation.METHODS),
                "control_kinds": list(dynamics.CONTROL_KINDS),
                "script_kinds": list(scripting.SCRIPT_KINDS),
            },
            "limits": {
                "max_figures": self.limits.max_figures,
                "max_datasets": self.limits.max_datasets,
                "max_inline_values": self.limits.max_inline_values,
                "max_dataset_rows": self.limits.max_dataset_rows,
                "max_analysis_rows": self.limits.max_analysis_rows,
                "figure_px": [self.limits.min_figure_px, self.limits.max_figure_px],
                "max_image_edge": self.limits.max_image_edge,
            },
            "open_figures": list(self._figures),
            "datasets": self.datasets.names(),
        }
        return ToolResult("GLPlot server information.", info)

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
        review: bool = False,
        crop: Optional[Any] = None,
        bounds: Optional[List[Any]] = None,
        dpi: Optional[float] = None,
    ) -> Tuple[bytes, List[Dict[str, Any]]]:
        """Render to ``fmt`` bytes and read the resolved axes from the same matplotlib figure.

        With ``review`` the finished figure is also checked (:func:`review_figure`) and the
        findings are left in ``entry.issues``.
        """
        view: List[Dict[str, Any]] = []

        def inspect(fig: Any) -> Dict[str, Any]:
            if entry.equal_panels:
                data_axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
                for index, ax in enumerate(data_axes, start=1):
                    if index in entry.equal_panels and not hasattr(ax, "get_zlim"):
                        ax.set_aspect("equal")
            view.extend(_summarise_axes(fig))
            if bounds is not None:
                bounds.append(fig.get_tightbbox(fig.canvas.get_renderer()))
            if review:
                labelled: Dict[int, List[str]] = {}
                for artist in entry.artists:
                    # a note or an arrow is not a series a legend would list
                    if artist.label and artist.kind not in ("text", "arrow"):
                        if artist.panel not in entry.legend_off:
                            labelled.setdefault(artist.panel, []).append(artist.label)
                entry.issues = review_figure(
                    fig, labelled=labelled, no_legend=sorted(entry.legend_off)
                )
            if fmt in ("png", "jpeg") and crop is None and dpi is None:
                edge = max(fig.get_size_inches()) * fig.dpi
                if edge > max_edge:
                    # Lower the *export* dpi: ``savefig`` does not read ``fig.dpi`` back.
                    return {"dpi": fig.dpi * max_edge / edge}
            return {}

        self._activate(entry, 1)
        # An animation passes one fixed ``crop`` (a Bbox in inches) for all its frames: a tight
        # crop per frame would fit each to its own text, so the frames would differ in size and
        # the axes would slide about. ``bounds`` collects each figure's tight box to build it.
        kwargs: Dict[str, Any] = (
            {"bbox_inches": crop, "pad_inches": 0.0}
            if crop is not None
            else {"bbox_inches": "tight", "pad_inches": 0.15}
        )
        if dpi is not None:
            kwargs["dpi"] = float(dpi)
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
    #: Where drawing calls go (``gplt`` or a twin axes) and the span of the data actually shown.
    api: Any = None
    visible: Optional[Tuple[int, int, int]] = None
    head: bool = False

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


def _brief(value: Any, limit: int = 12) -> Any:
    """``value`` with long lists replaced by a count, so a description stays short."""
    if isinstance(value, Mapping):
        return {k: _brief(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > limit and not any(isinstance(v, (Mapping, list, tuple)) for v in value):
            return f"[{len(value)} values]"
        return [_brief(v, limit) for v in value]
    return value


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


def _api(r: _PlotRequest) -> Any:
    """The object drawing calls go to: pyplot, or the panel's right-hand twin axes."""
    return r.api if r.api is not None else gplt


def _head_marker(r: _PlotRequest, layers: Any) -> None:
    """A dot on the last point shown, in the colour of the line (the 'head' of a moving curve)."""
    if not r.head or r.n == 0:
        return
    from matplotlib.colors import to_hex

    layer = layers[0] if isinstance(layers, (list, tuple)) and layers else layers
    colour = r.options.get("color")
    if colour is None:
        try:
            colour = to_hex(tuple(float(v) for v in layer.style.color))
        except (AttributeError, TypeError, ValueError):
            colour = None
    size = r.options.get("size")
    diameter = float(size) * 1.6 if size else 10.0
    dot: Dict[str, Any] = {
        "size": diameter if r.z is not None else _px_area(diameter),
        "marker": "o",
    }
    if colour is not None:
        dot["color"] = colour
    args = (r.x[-1:], r.y[-1:]) + ((r.z[-1:],) if r.z is not None else ())
    _api(r).scatter(*args, **dot)


def _px_area(diameter: Any) -> Any:
    """A 2-D marker's ``size`` (a diameter in px, as the tools document it) as the area in pt^2
    the exported picture's ``scatter`` takes -- the conversion 3-D markers already get, so a
    size means the same thing in both."""
    from ..utils.preview import _point_diameter_px_to_area_pt2

    return _point_diameter_px_to_area_pt2(diameter, gplt.gcf())


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
        layers = _api(r).plot(r.x, r.y, r.z, **kw)
    else:
        layers = _api(r).plot(r.x, r.y, **kw)
    _head_marker(r, layers)
    return layers


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
    if r.z is None and kw.get("size") is not None:
        kw["size"] = _px_area(kw["size"])
    if r.z is not None:
        layer = _api(r).scatter(r.x, r.y, r.z, **kw)
    else:
        layer = _api(r).scatter(r.x, r.y, **kw)
    if r.colorbar and r.color_by is not None:
        gplt.colorbar(label=r.color_label) if r.color_label else gplt.colorbar()
    _head_marker(r, layer)
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
    layers = _api(r).step(r.x, r.y, where="mid", **kw)
    _head_marker(r, layers)
    return layers


def _draw_stem(r: _PlotRequest) -> Any:
    kw = r.style({"label": "label"})  # colour and the rest are not stem options: reported
    return gplt.stem(r.x, r.y, **kw)


def _draw_bar(r: _PlotRequest) -> Any:
    kw = r.style(
        {"color": "color", "alpha": "alpha", "label": "label", "width": "width", "hatch": "hatch"}
    )
    if r.error is not None:
        kw["yerr"] = r.error
    bars = _api(r).bar(r.x, r.y, **kw)
    if r.tick_labels:
        gplt.xticks(list(r.x), r.tick_labels)
    return bars


def _draw_barh(r: _PlotRequest) -> Any:
    kw = r.style({"color": "color", "alpha": "alpha", "label": "label", "hatch": "hatch"})
    bars = gplt.barh(r.x, r.y, **kw)
    if r.tick_labels:
        gplt.yticks(list(r.x), r.tick_labels)
    return bars


def _draw_area(r: _PlotRequest) -> Any:
    kw = r.style({"color": "color", "alpha": "alpha", "label": "label", "hatch": "hatch"})
    return _api(r).fill_between(r.x, r.y, r.y2 if r.y2 is not None else 0.0, **kw)


def _draw_errorbar(r: _PlotRequest) -> Any:
    kw = r.style({"color": "color", "label": "label"})
    return _api(r).errorbar(r.x, r.y, yerr=r.error, **kw)


def _draw_hist(r: _PlotRequest) -> Any:
    kw = r.style(
        {"color": "color", "alpha": "alpha", "label": "label", "bins": "bins", "hatch": "hatch"}
    )
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
