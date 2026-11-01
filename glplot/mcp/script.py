"""``export_script``: the Python a person would write to get the figure the model built.

The model works through tool calls, but its user usually wants a script they can read, change and
keep. Every figure records the calls that built it (see ``Toolbox._log_operation``), so the
script is generated from that record -- not from the picture -- and is meant to be edited:

* ``figure``       the figure at the controls' current values, saved as a PNG;
* ``interactive``  the controls become real sliders, toggles, menus and buttons (``plt.slider``,
  ``plt.checkbox``, ``plt.dropdown``, ``plt.button``) that redraw the figure, and ``plt.show()``
  opens the window;
* ``animation``    the last ``animate`` specification becomes a ``FuncAnimation``: the timeline
  as numpy arrays, one ``update(i)`` that redraws with that frame's values, and ``ani.save``
  for GIF or MP4.

Datasets are rebuilt in the script from how they were made: expressions become numpy, ``simulate``
becomes a ``solve_ivp`` call, a loaded file becomes a load. Anything else (the output of an
analysis, say) is embedded as numbers when small, and reported in ``notes`` when it is not.

The module only reads the toolbox's records; it runs nothing and imports no GL. Tests execute the
scripts it writes and compare them with the toolbox's own renders.
"""

from __future__ import annotations

import ast
import inspect
import math
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..gui import expressions
from ..gui.expressions import SAFE_NAMES
from . import dynamics
from .state import McpError

__all__ = ["export_script", "SCRIPT_KINDS"]

SCRIPT_KINDS = ("figure", "interactive", "animation")

#: Below this many numbers a computed dataset is written into the script as a literal.
_INLINE_LIMIT = 4000


@dataclass
class Raw:
    """A fragment of code, written as is where a value is expected."""

    code: str


def _whole(value: Any, default: int) -> Any:
    """A whole-number argument: bound code (already an int) is kept, a plain number is made one."""
    return value if isinstance(value, Raw) else int(value or default)


def lit(value: Any) -> str:
    """``value`` as a Python literal (a :class:`Raw` is its code)."""
    if isinstance(value, Raw):
        return value.code
    if value is None or isinstance(value, (bool, str)):
        return repr(value)
    if isinstance(value, (int, np.integer)):
        return repr(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "np.nan"
        if math.isinf(number):
            return "np.inf" if number > 0 else "-np.inf"
        return repr(number)
    if isinstance(value, np.ndarray):
        return lit(value.tolist())
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(lit(v) for v in value) + "]"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{lit(k)}: {lit(v)}" for k, v in value.items()) + "}"
    return repr(value)


# ----------------------------------------------------------------------------------
# Expressions -> numpy
# ----------------------------------------------------------------------------------

_CONSTANTS = {"pi": "np.pi", "e": "np.e", "tau": "2 * np.pi", "inf": "np.inf", "nan": "np.nan"}


class _Translator(ast.NodeTransformer):
    """Rewrite an expression's names as numpy (or helper) names, leaving variables alone."""

    def __init__(self, variables: Mapping[str, str], helpers: "Dict[str, None]") -> None:
        self.variables = variables
        self.helpers = helpers

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802 - ast API
        name = node.id
        if name in self.variables:
            return ast.copy_location(ast.Name(id=self.variables[name], ctx=node.ctx), node)
        if name in dynamics.HELPERS:
            self.helpers.setdefault(name)
            return node
        if name in SAFE_NAMES:
            value = SAFE_NAMES[name]
            if callable(value):
                if getattr(np, name, None) is value:
                    return ast.copy_location(
                        ast.Attribute(
                            value=ast.Name(id="np", ctx=ast.Load()), attr=name, ctx=node.ctx
                        ),
                        node,
                    )
                self.helpers.setdefault(name)
                return node
            return ast.copy_location(
                ast.parse(_CONSTANTS.get(name, repr(value)), mode="eval").body, node
            )
        return node


def translate(
    expression: str,
    helpers: "Dict[str, None]",
    variables: Optional[Mapping[str, str]] = None,
) -> str:
    """The numpy source of an expression of the safe language."""
    tree = ast.parse(expression.strip(), mode="eval")
    tree = _Translator(variables or {}, helpers).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


#: ``noise`` as written into a script: the same draw, without the server's size guard (which
#: reads a module constant the script does not have).
_NOISE = '''def noise(n_or_x, scale=1.0, seed=None):
    """Gaussian noise: ``noise(500)`` draws 500 values, ``noise(x, 0.1)`` matches x's shape."""
    arr = np.asarray(n_or_x)
    shape = (int(float(arr)),) if arr.ndim == 0 else arr.shape
    return np.random.default_rng(seed).normal(0.0, float(scale), size=shape)'''


def _helper_source(name: str) -> str:
    if name == "noise":
        return _NOISE
    function = dynamics.HELPERS.get(name) or SAFE_NAMES[name]
    return textwrap.dedent(inspect.getsource(function)).rstrip()


# ----------------------------------------------------------------------------------
# The generator
# ----------------------------------------------------------------------------------


def _spread(value: Any) -> str:
    """Arguments for a call that takes (low, high): ``0, 10`` for a literal pair, else ``*code``."""
    if isinstance(value, (list, tuple)) and not isinstance(value, Raw):
        return ", ".join(lit(v) for v in value)
    return f"*{lit(value)}"


def _ident(name: str) -> str:
    cleaned = re.sub(r"\W", "_", name)
    return cleaned if cleaned and not cleaned[0].isdigit() else f"_{cleaned}"


class _Writer:
    def __init__(self) -> None:
        self.lines: List[str] = []
        self.level = 0

    def add(self, text: str = "") -> None:
        if not text:
            self.lines.append("")
            return
        for part in text.split("\n"):
            self.lines.append(("    " * self.level + part) if part.strip() else "")

    def call(self, head: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> None:
        parts = [lit(a) for a in args] + [
            f"{k}={lit(v)}" for k, v in kwargs.items() if v is not None
        ]
        one = f"{head}({', '.join(parts)})"
        if len(one) + 4 * self.level <= 96 or not parts:
            self.add(one)
            return
        self.add(f"{head}(")
        self.level += 1
        for part in parts:
            self.add(part + ",")
        self.level -= 1
        self.add(")")

    def indent(self) -> "_Indent":
        return _Indent(self)

    def text(self) -> str:
        return "\n".join(self.lines)


class _Indent:
    def __init__(self, writer: _Writer) -> None:
        self.writer = writer

    def __enter__(self) -> None:
        self.writer.level += 1

    def __exit__(self, *exc: Any) -> None:
        self.writer.level -= 1


class _Script:
    """Builds one script from a figure's record."""

    def __init__(self, box: Any, entry: Any, kind: str) -> None:
        self.box = box
        self.entry = entry
        self.kind = kind
        self.notes: List[str] = []
        self.helpers: Dict[str, None] = {}
        self.uses_ramp = False
        self.uses_fade = False
        self.uses_reveal_text = False
        #: In an interactive or animated script every control and variable is a local name, so a
        #: function's own parameters are written under ``p_`` names that cannot collide with them.
        self.dynamic = kind != "figure"
        self.datasets: Dict[str, None] = {}
        self.twins: set = set()
        self.panel_now: Optional[int] = None
        self.body = _Writer()

    # -- data ------------------------------------------------------------------------

    def ds(self, name: str) -> str:
        self.datasets.setdefault(name)
        return "d_" + _ident(name)

    def series(self, ref: Any, dataset: Optional[str]) -> Any:
        if ref is None or isinstance(ref, Raw):
            return ref
        if isinstance(ref, str):
            return Raw(f"{self.ds(dataset)}[{ref!r}]")
        return Raw(f"np.array({lit(list(ref))}, dtype=float)")

    def _order(self, wanted: Sequence[str]) -> List[str]:
        """``wanted`` dataset names plus the ones they were built from, base first."""
        out: List[str] = []

        def visit(name: str) -> None:
            if name in out or name not in self.box.datasets:
                return
            origin = getattr(self.box.datasets.get(name), "mcp_origin", None) or {}
            if origin.get("kind") == "create" and origin.get("base"):
                visit(origin["base"])
            out.append(name)

        for name in wanted:
            visit(name)
        return out

    def data_section(self, writer: _Writer) -> None:
        if not self.datasets:
            return
        for name in self._order(list(self.datasets)):
            table = self.box.datasets.get(name)
            origin = getattr(table, "mcp_origin", None) or {}
            var = "d_" + _ident(name)
            kind = origin.get("kind")
            writer.add(f"# dataset {name!r}")
            if kind == "create":
                self._create(writer, var, name, origin)
            elif kind == "simulate":
                self._simulate(writer, var, origin)
            elif kind == "file":
                writer.add(
                    f"# loaded from a file the model read: adjust the path, and the delimiter if "
                    f"it is not a comma\n{var}_raw = np.genfromtxt({origin['path']!r}, "
                    "delimiter=',', "
                    "names=True)\n"
                    f"{var} = {{key: np.asarray({var}_raw[key], dtype=float) "
                    f"for key in {var}_raw.dtype.names}}"
                )
                if table.categories:
                    self.notes.append(
                        f"Text columns of {name!r} are numbers (label indices) after genfromtxt; "
                        "map them with the category lists in the comments."
                    )
                    for column, labels in table.categories.items():
                        writer.add(f"# {column}: {lit(labels)}")
            else:
                self._values(writer, var, name, table)
            known = set(origin.get("names") or [])
            extras = [c for c in table.column_names() if known and c not in known]
            if extras:
                self._extra_columns(writer, var, name, table, extras)
            writer.add()

    def _create(self, writer: _Writer, var: str, name: str, origin: Mapping[str, Any]) -> None:
        columns = origin["columns"]
        base = origin.get("base")
        writer.add(f"def _make_{_ident(name)}():")
        with writer.indent():
            names: List[str] = []
            if base:
                writer.add(f"base = {'d_' + _ident(base)}")
                for column in self.box.datasets.get(base).column_names():
                    writer.add(f"{column} = base[{column!r}]")
                    names.append(column)
            for column, spec in columns.items():
                if isinstance(spec, str):
                    code = translate(spec, self.helpers)
                    size = "None" if not names else f"len({names[0]})"
                    writer.add(f"{column} = _as_column({code}, {size})")
                    self.helpers.setdefault("_as_column")
                else:
                    writer.add(f"{column} = np.array({lit(list(spec))}, dtype=float)")
                if column not in names:
                    names.append(column)
            writer.add("return {" + ", ".join(f"{n!r}: {n}" for n in names) + "}")
        writer.add(f"{var} = _make_{_ident(name)}()")

    def _simulate(self, writer: _Writer, var: str, origin: Mapping[str, Any]) -> None:
        spec = origin["spec"]
        states = list(spec["equations"])
        parameters = spec.get("parameters") or {}
        self.imports_scipy = True
        writer.add(f"def _rhs_{_ident(var)}(t, y):")
        with writer.indent():
            for key, value in parameters.items():
                writer.add(f"{key} = {lit(float(value))}")
            writer.add(f"{', '.join(states)}{',' if len(states) == 1 else ''} = y")
            rhs = [translate(spec["equations"][s], self.helpers) for s in states]
            writer.add("return [" + ", ".join(rhs) + "]")
        t0, t1, steps = float(spec["t_start"]), float(spec["t_end"]), int(spec["steps"])
        method = {
            "rk45": "RK45",
            "rk23": "RK23",
            "dop853": "DOP853",
            "radau": "Radau",
            "bdf": "BDF",
            "lsoda": "LSODA",
        }
        chosen = method.get(spec.get("method", "rk45"), "RK45")
        if spec.get("method") == "rk4":
            self.notes.append(
                "simulate used its fixed-step RK4; the script uses RK45 (same answer to ~1e-8)."
            )
        writer.add(f"_t = np.linspace({t0!r}, {t1!r}, {steps})")
        init = [float(spec["initial"][s]) for s in states]
        writer.add(
            f"_sol = solve_ivp(_rhs_{_ident(var)}, ({t0!r}, {t1!r}), {lit(init)}, t_eval=_t, "
            f"method={chosen!r}, rtol=1e-8, atol=1e-10)"
        )
        writer.add(
            f"{var} = {{'t': _t, "
            + ", ".join(f"{s!r}: _sol.y[{i}]" for i, s in enumerate(states))
            + "}"
        )
        if spec.get("derived"):
            scope = [f"{k} = {lit(float(v))}" for k, v in parameters.items()]
            for line in scope:
                writer.add(line)
            writer.add("t = _t")
            for i, s in enumerate(states):
                writer.add(f"{s} = _sol.y[{i}]")
            for column, expression in spec["derived"].items():
                writer.add(f"{column} = _as_column({translate(expression, self.helpers)}, {steps})")
                writer.add(f"{var}[{column!r}] = {column}")
                self.helpers.setdefault("_as_column")

    def _extra_columns(
        self, writer: _Writer, var: str, name: str, table: Any, extras: Sequence[str]
    ) -> None:
        """Columns added after the dataset was made (an analysis result): numbers, or a note."""
        if table.n_rows() * len(extras) > _INLINE_LIMIT:
            self.notes.append(
                f"Column(s) {', '.join(extras)} of dataset {name!r} were computed by "
                "analyze_dataset inside the MCP server and are too big to embed; the script "
                "raises until you compute them (or load them) into the dict."
            )
            writer.add(
                f"raise SystemExit('columns {', '.join(extras)} of {name} were computed by the "
                f"MCP server: add them to d_{_ident(name)}')"
            )
            return
        writer.add(f"# columns computed by analyze_dataset: {', '.join(extras)}")
        for column in extras:
            values = np.asarray(table.get(column), dtype=float)
            writer.add(f"{var}[{column!r}] = np.array({lit([float(f'{v:.8g}') for v in values])})")

    def _values(self, writer: _Writer, var: str, name: str, table: Any) -> None:
        total = table.n_rows() * table.n_cols()
        if total > _INLINE_LIMIT:
            self.notes.append(
                f"Dataset {name!r} ({table.n_rows()} rows) was computed inside the MCP server and "
                "is too big to embed; the script raises until you supply it "
                f"(np.load / np.genfromtxt into the dict d_{_ident(name)})."
            )
            writer.add(
                f"raise SystemExit('dataset {name} has {table.n_rows()} rows computed by the MCP "
                f"server: load it here into d_{_ident(name)}')"
            )
            writer.add(f"{var} = {{}}")
            return
        writer.add(f"{var} = {{")
        with writer.indent():
            for column in table.columns:
                values = np.asarray(column.values, dtype=float)
                writer.add(
                    f"{column.name!r}: np.array({lit([float(f'{v:.8g}') for v in values])}),"
                )
        writer.add("}")

    # -- figure ----------------------------------------------------------------------

    def figure(self, writer: _Writer) -> None:
        spec = self.entry.spec
        rows, cols = spec["rows"], spec["cols"]
        kwargs: Dict[str, Any] = {"width": spec["width"], "height": spec["height"]}
        if spec["projection"] == "3d":
            kwargs["projection"] = "3d"
        if rows * cols > 1:
            parts = {"squeeze": False, **kwargs, "wspace": 0.22, "hspace": 0.32}
            writer.call("fig, axs = plt.subplots", [rows, cols], parts)
        else:
            writer.call("fig = plt.figure", [], kwargs)
        if spec.get("style"):
            writer.add(f"plt.plot_style({spec['style']!r})")
        if spec.get("title"):
            writer.add(f"plt.{'suptitle' if rows * cols > 1 else 'title'}({spec['title']!r})")
        self.panel_now = 1 if rows * cols > 1 else None
        self.twins = set()

    def panel(self, writer: _Writer, panel: int) -> None:
        rows, cols = self.entry.spec["rows"], self.entry.spec["cols"]
        if rows * cols == 1 or self.panel_now == panel:
            return
        r, c = divmod(panel - 1, cols)
        writer.add(f"plt.sca(axs[{r}][{c}])")
        self.panel_now = panel

    # -- operations -----------------------------------------------------------------

    def merged(self, op: Any, overrides: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
        import copy

        args = copy.deepcopy(op.args)
        for path, value in (overrides.get(op.artist_id or "") or {}).items():
            if value is None:
                args.pop(path, None)
            else:
                dynamics.set_path(args, path, value)
        return args

    def ops(
        self,
        writer: _Writer,
        overrides: Mapping[str, Mapping[str, Any]],
        axes_overrides: Mapping[int, Mapping[str, Any]],
    ) -> None:
        for op in self.entry.log:
            args = self.merged(op, overrides)
            writer.add(f"# {op.tool}" + (f" ({op.artist_id})" if op.artist_id else ""))
            getattr(self, f"op_{op.tool}")(writer, op, args)
        for panel, values in axes_overrides.items():
            if values:
                writer.add(f"# bound axes properties of panel {panel}")
                self.op_set_axes(writer, None, {"panel": panel, **values})

    def _color(self, op: Any, args: Mapping[str, Any]) -> Any:
        """The colour to pass: the caller's, else (for the cycled kinds) the one actually drawn."""
        color = args.get("color")
        if color is not None:
            return color
        return op.resolved.get("color")

    def _drawn_color(self, artist_id: Optional[str]) -> Optional[str]:
        from matplotlib.colors import to_hex

        artist = next((a for a in self.entry.artists if a.artist_id == artist_id), None)
        if artist is None or not artist.layer_ids:
            return None
        for layer in self.entry.plot.scene.layers:
            if layer.layer_id == artist.layer_ids[0]:
                for name in ("color", "face_color", "edge_color"):
                    try:
                        return to_hex(tuple(float(v) for v in getattr(layer.style, name)))
                    except (AttributeError, TypeError, ValueError):
                        continue
        return None

    def op_add_function(self, writer: _Writer, op: Any, args: Dict[str, Any]) -> None:
        expression = args["expression"]
        rename = {"x": "x_"}
        for key, value in (args.get("parameters") or {}).items():
            name = f"p_{key}" if self.dynamic else key
            writer.add(f"{name} = {lit(value)}")
            rename[key] = name
        samples = args.get("samples", 500)
        writer.add(f"x_ = np.linspace({lit(args['x_min'])}, {lit(args['x_max'])}, {lit(samples)})")
        code = translate(expression, self.helpers, rename)
        writer.add(f"y_ = _as_column({code}, {lit(samples)})")
        self.helpers.setdefault("_as_column")
        plot_args = {
            k: v
            for k, v in args.items()
            if k not in ("expression", "x_min", "x_max", "samples", "parameters")
        }
        plot_args.update(kind=args.get("kind", "line"), x=Raw("x_"), y=Raw("y_"))
        if plot_args.get("label") is None:
            plot_args["label"] = expression
        self.plot(writer, op, plot_args)

    def op_add_plot(self, writer: _Writer, op: Any, args: Dict[str, Any]) -> None:
        self.plot(writer, op, args)

    def plot(self, writer: _Writer, op: Any, a: Dict[str, Any]) -> None:
        kind = a["kind"]
        dataset = a.get("dataset")
        panel = a.get("panel", 1)
        self.panel(writer, panel)
        api = "plt"
        if a.get("axis") == "right":
            if panel not in self.twins:
                writer.add(
                    "ax2 = plt.twinx()" if len(self.twins) == 0 else f"ax2_{panel} = plt.twinx()"
                )
                self.twins.add(panel)
            api = "ax2" if len(self.twins) == 1 else f"ax2_{panel}"
        single = kind in ("hist", "box", "violin", "pie")
        names = ("x", "y", "z", "error", "y2", "color_by", "size_by")
        values: Dict[str, Any] = {}
        if single:
            if kind in ("box", "violin") and a.get("columns"):
                groups = "[" + ", ".join(f"{self.ds(dataset)}[{c!r}]" for c in a["columns"]) + "]"
                values["groups"] = Raw(groups)
            else:
                values["y"] = self.series(
                    a.get("y") if a.get("y") is not None else a.get("x"), dataset
                )
        else:
            for key in names:
                if a.get(key) is not None:
                    values[key] = self.series(a[key], dataset)
            if "x" not in values and "y" in values:
                values["x"] = Raw(f"np.arange(len({lit(values['y'])}), dtype=float)")
        for key, code in values.items():
            if lit(code) != f"{key}_":
                writer.add(f"{key}_ = {lit(code)}")
        ref = {k: Raw(f"{k}_") for k in values}

        reveal, trail = a.get("reveal"), a.get("trail")
        if (reveal is not None or trail is not None) and not single:
            writer.add("_n = len(y_)")
            # the server shows a 3-D line at least two points (plot3d cannot draw fewer)
            floor = "min(_n, 2)" if kind == "line" and "z" in values else "1"
            writer.add(
                "_count = "
                + (
                    "_n"
                    if reveal is None
                    else f"max({floor}, min(_n, int(np.ceil(({lit(reveal)}) * _n - 1e-9))))"
                )
            )
            writer.add(
                "_first = "
                + ("0" if trail is None else f"max(0, _count - max({lit(trail)}, {floor}))")
            )
            for key in values:
                writer.add(f"{key}_ = {key}_[_first:_count]")

        labels = self._tick_labels(a, kind, dataset)
        color = self._color(op, a)
        if kind == "area" and a.get("alpha") is None:
            a = dict(a, alpha=1.0)
        common = {
            "color": color,
            "alpha": a.get("alpha"),
            "linewidth": a.get("linewidth"),
            "linestyle": a.get("linestyle"),
            "marker": a.get("marker"),
            "label": a.get("label"),
        }
        pos = [ref[k] for k in ("x", "y", "z") if k in ref]
        if kind == "line":
            kw = {
                k: common[k]
                for k in ("color", "alpha", "linewidth", "linestyle", "marker", "label")
            }
            kw["markersize"] = a.get("size")
            writer.call(f"{api}.plot", pos, kw)
            if a.get("head"):
                self._head(writer, api, ref, a, color or self._drawn_color(op.artist_id))
        elif kind == "scatter":
            flat = "z" not in ref
            kw = {k: common[k] for k in ("color", "alpha", "marker", "label")}
            kw["size"] = self.area(a.get("size")) if flat else a.get("size")
            kw["cmap"] = a.get("cmap")
            if "color_by" in ref:
                kw["c"] = ref["color_by"]
                kw.pop("color")
            if "size_by" in ref:
                writer.add("_s = size_by_")
                writer.add("_span = np.nanmax(_s) - np.nanmin(_s)")
                writer.add(
                    "_unit = (_s - np.nanmin(_s)) / _span if _span > 0 else np.full(len(_s), 0.5)"
                )
                kw["size"] = Raw("4.0 + 24.0 * _unit")
                if flat:
                    kw["size"] = self.area(kw["size"])
            writer.call(f"{api}.scatter", pos, kw)
            if a.get("colorbar") and "color_by" in ref:
                writer.add(
                    f"plt.colorbar(label={a['color_by']!r})"
                    if isinstance(a.get("color_by"), str)
                    else "plt.colorbar()"
                )
            if a.get("head"):
                self._head(writer, api, ref, a, color)
        elif kind == "step":
            kw = {k: common[k] for k in ("color", "alpha", "linewidth", "linestyle", "label")}
            writer.call(f"{api}.step", pos, {"where": "mid", **kw})
            if a.get("head"):
                self._head(writer, api, ref, a, color or self._drawn_color(op.artist_id))
        elif kind == "stem":
            writer.call("plt.stem", pos, {"label": a.get("label")})
        elif kind in ("bar", "barh"):
            kw = {
                "color": color,
                "alpha": a.get("alpha"),
                "label": a.get("label"),
                "hatch": a.get("hatch"),
            }
            if kind == "bar":
                kw["width"] = a.get("width")
                if "error" in ref:
                    kw["yerr"] = ref["error"]
            writer.call(f"{api}.{kind}", pos, kw)
            if labels:
                writer.add(
                    f"plt.{'xticks' if kind == 'bar' else 'yticks'}(list(x_), {lit(labels)})"
                )
        elif kind == "area":
            writer.call(
                f"{api}.fill_between",
                [ref["x"], ref["y"], ref.get("y2", 0.0)],
                {
                    "color": color,
                    "alpha": a.get("alpha"),
                    "label": a.get("label"),
                    "hatch": a.get("hatch"),
                },
            )
        elif kind == "errorbar":
            writer.call(
                f"{api}.errorbar",
                [ref["x"], ref["y"]],
                {"yerr": ref.get("error"), "color": color, "label": a.get("label")},
            )
        elif kind == "hist":
            writer.call(
                "plt.hist",
                [ref["y"]],
                {
                    "color": color,
                    "alpha": a.get("alpha"),
                    "label": a.get("label"),
                    "bins": a.get("bins"),
                    "hatch": a.get("hatch"),
                },
            )
        elif kind in ("box", "violin"):
            data = Raw("groups_") if "groups" in values else Raw("[y_]")
            if "groups" in values:
                writer.add(f"groups_ = {lit(values['groups'])}")
                del values["groups"]
            writer.add(f"plt.{'boxplot' if kind == 'box' else 'violinplot'}({lit(data)})")
            names_ = a.get("columns")
            if names_:
                writer.add(f"plt.xticks(list(range(1, {len(names_)} + 1)), {lit(list(names_))})")
        elif kind == "pie":
            writer.call("plt.pie", [ref["y"]], {"labels": labels})
            writer.add("plt.axis('equal')")
        elif kind == "hist2d":
            writer.call(
                "plt.hist2d", [ref["x"], ref["y"]], {"cmap": a.get("cmap"), "bins": a.get("bins")}
            )
        elif kind == "hexbin":
            writer.call("plt.hexbin", [ref["x"], ref["y"]], {"cmap": a.get("cmap")})

    def _head(
        self, writer: _Writer, api: str, ref: Mapping[str, Any], a: Mapping[str, Any], color: Any
    ) -> None:
        size = a.get("size")
        pos = [Raw(f"{k}_[-1:]") for k in ("x", "y", "z") if k in ref]
        if isinstance(size, Raw):
            dot: Any = Raw(f"({size.code}) * 1.6")
        else:
            dot = float(size) * 1.6 if size else 10.0
        writer.call(
            f"{api}.scatter",
            pos,
            {
                "size": dot if "z" in ref else self.area(dot),
                "marker": "o",
                "color": color,
            },
        )

    def area(self, diameter: Any) -> Any:
        """A 2-D marker's diameter in px as the area the server hands to scatter (see
        ``toolbox._px_area``): the same number when it is one, the same arithmetic when bound."""
        if diameter is None:
            return None
        dpi = float(getattr(self.entry.plot, "_figure_dpi", 100.0) or 100.0)
        if isinstance(diameter, Raw):
            return Raw(f"(np.maximum({diameter.code}, 0.5) * 72.0 / {dpi!r}) ** 2")
        from ..utils.preview import _point_diameter_px_to_area_pt2

        return _point_diameter_px_to_area_pt2(float(diameter), self.entry.plot)

    def _tick_labels(
        self, a: Mapping[str, Any], kind: str, dataset: Optional[str]
    ) -> Optional[List[str]]:
        if a.get("tick_labels"):
            return list(a["tick_labels"])
        if dataset and kind in ("bar", "barh", "pie") and isinstance(a.get("x"), str):
            table = self.box.datasets.get(dataset)
            labels = table.categories.get(a["x"])
            if labels:
                codes = np.asarray(table.get(a["x"]), dtype=float)
                return [
                    labels[int(c)] if np.isfinite(c) and 0 <= int(c) < len(labels) else ""
                    for c in codes
                ]
        return None

    def op_add_field(self, writer: _Writer, op: Any, a: Dict[str, Any]) -> None:
        kind = a.get("kind", "image")
        self.panel(writer, a.get("panel", 1))
        x0, x1 = a.get("x_range") or (0.0, 1.0)
        y0, y1 = a.get("y_range") or (0.0, 1.0)
        surface = kind in ("surface", "wireframe")
        if a.get("expression") is not None:
            n = lit(_whole(a.get("resolution"), 80 if surface else 200))
            writer.add(f"xs = np.linspace({lit(x0)}, {lit(x1)}, {n})")
            writer.add(f"ys = np.linspace({lit(y0)}, {lit(y1)}, {n})")
            writer.add("x, y = np.meshgrid(xs, ys)")
            rename = {}
            for key, value in (a.get("parameters") or {}).items():
                name = f"p_{key}" if self.dynamic else key
                writer.add(f"{name} = {lit(value)}")
                rename[key] = name
            code = translate(a["expression"], self.helpers, rename)
            writer.add(f"z_ = _as_column({code}, None) * np.ones_like(x)")
            self.helpers.setdefault("_as_column")
        elif a.get("matrix") is not None:
            matrix = np.asarray(a["matrix"], dtype=float)
            writer.add(f"z_ = np.array({lit(matrix.tolist())})")
            writer.add(f"xs = np.linspace({lit(x0)}, {lit(x1)}, z_.shape[1])")
            writer.add(f"ys = np.linspace({lit(y0)}, {lit(y1)}, z_.shape[0])")
            writer.add("x, y = np.meshgrid(xs, ys)")
        else:
            self.notes.append(
                "add_field(file=...) reads a file the model had access to; "
                "load it into z_ yourself."
            )
            writer.add(f"z_ = np.load({str(a.get('file'))!r})  # adjust")
            writer.add(f"xs = np.linspace({lit(x0)}, {lit(x1)}, z_.shape[1])")
            writer.add(f"ys = np.linspace({lit(y0)}, {lit(y1)}, z_.shape[0])")
            writer.add("x, y = np.meshgrid(xs, ys)")
        accepted = {
            "image": ("cmap", "alpha", "label", "vmin", "vmax"),
            "contour": ("cmap", "alpha", "label"),
            "contourf": ("cmap", "alpha"),
            "pcolormesh": ("cmap", "alpha", "vmin", "vmax"),
            "surface": ("cmap", "alpha", "label", "vmin", "vmax"),
            "wireframe": ("label",),
        }[kind]
        kw = {k: a.get(k) for k in accepted}
        if kind == "image":
            writer.call(
                "plt.imshow",
                [Raw("z_")],
                {
                    "origin": "lower",
                    "extent": Raw(f"({lit(x0)}, {lit(x1)}, {lit(y0)}, {lit(y1)})"),
                    "aspect": "auto",
                    **kw,
                },
            )
        elif kind in ("contour", "contourf"):
            writer.call(
                f"plt.{kind}",
                [Raw("x"), Raw("y"), Raw("z_")],
                {"levels": _whole(a.get("levels"), 10), **kw},
            )
        elif kind == "pcolormesh":
            writer.call("plt.pcolormesh", [Raw("x"), Raw("y"), Raw("z_")], kw)
        else:
            writer.call(
                "plt.plot_surface" if kind == "surface" else "plt.plot_wireframe",
                [Raw("x"), Raw("y"), Raw("z_")],
                kw,
            )
        if a.get("colorbar"):
            writer.add("plt.colorbar()")

    def op_add_recipe(self, writer: _Writer, op: Any, a: Dict[str, Any]) -> None:
        self.panel(writer, a.get("panel", 1))
        self.imports_recipes = True
        key = a["recipe"]
        writer.add(f"_spec = recipes.get_recipe({key!r})")
        if a.get("sample"):
            writer.add("_table = recipes.sample_table(_spec)")
            writer.add("_assign = dict(recipes.sample_assignment(_spec))")
            writer.add("_rows = len(next(iter(_table.values())))")
        else:
            var = self.ds(a["dataset"])
            writer.add(f"_table = {{k: np.asarray(v, dtype=float) for k, v in {var}.items()}}")
            writer.add("_assign = dict(recipes.assign_columns(_spec, list(_table)))")
            if a.get("columns"):
                writer.add(f"_assign.update({lit(dict(a['columns']))})")
            writer.add("_rows = len(next(iter(_table.values())))")
        writer.add("_data = recipes.slot_arrays(_spec, _table, _assign, _rows)")
        kw = {
            "name": a.get("dataset") or key,
            "look": a.get("look", "recipe"),
            "labels": Raw("recipes.slot_labels(_spec, _assign)"),
            "title": a.get("title"),
        }
        writer.call("_layers = recipes.apply_recipe", [Raw("fig"), Raw("_spec"), Raw("_data")], kw)
        writer.add("for _layer in _layers:")
        with writer.indent():
            if a.get("label"):
                writer.add(f"_layer.label = {a['label']!r}")
            else:
                # A recipe's layers are named after it; the export reads a label as "put me in
                # the legend", so they are marked as bookkeeping, as the server does.
                writer.add("_layer.metadata['auto_label'] = True")
        self.notes.append(
            "add_recipe uses GLPlot's recipe module (glplot.gui.recipes); it has no "
            "matplotlib equivalent."
        )

    def op_add_annotation(self, writer: _Writer, op: Any, a: Dict[str, Any]) -> None:
        self.panel(writer, a.get("panel", 1))
        kind = a["kind"]
        color = a.get("color") or op.resolved.get("color")
        shade: Any = color
        if a.get("alpha") is not None:
            self.uses_fade = True
            shade = Raw(f"_fade({lit(color)}, {lit(a['alpha'])})")
        style = {
            "color": shade,
            "linestyle": a.get("linestyle"),
            "linewidth": a.get("linewidth"),
            "label": a.get("label"),
        }
        if kind == "text":
            text: Any = a["text"]
            if a.get("reveal") is not None:
                self.uses_reveal_text = True
                text = Raw(f"reveal_text({lit(a['text'])}, {lit(a['reveal'])})")
            kw: Dict[str, Any] = {"color": shade, "fontsize": a.get("fontsize")}
            if a.get("align", "left") != "left":
                kw["ha"] = a["align"]
            if a.get("coords", "data") == "axes":
                kw["transform"] = Raw("plt.gca().transAxes")
            if a.get("reveal") is not None:
                kw["layout_text"] = a["text"]
            if a.get("background"):
                page = op.resolved.get("page", "#ffffff")
                kw["bbox"] = {"facecolor": page, "edgecolor": "none", "pad": 4}
            writer.call("plt.text", [a["x"], a["y"], text], kw)
        elif kind == "hline":
            writer.call("plt.axhline", [a["y"]], style)
        elif kind == "vline":
            writer.call("plt.axvline", [a["x"]], style)
        else:
            grow = a.get("reveal")
            scale = "" if grow is None else f" * ({lit(grow)})"
            writer.add(f"_dx, _dy = ({lit(a['dx'])}){scale}, ({lit(a['dy'])}){scale}")
            writer.add("_head = 0.22 * float(np.hypot(_dx, _dy))")
            writer.add("if _head > 0:")
            with writer.indent():
                kw = {
                    "head_length": Raw("_head"),
                    "head_width": Raw("0.6 * _head"),
                    "color": shade,
                    "width": a.get("linewidth") if a.get("linewidth") is not None else 3.0,
                    "label": a.get("label"),
                }
                writer.call("plt.arrow", [a["x"], a["y"], Raw("_dx"), Raw("_dy")], kw)

    def op_set_style(self, writer: _Writer, op: Any, a: Dict[str, Any]) -> None:
        writer.add(f"plt.plot_style({a['preset']!r})")

    def op_set_axes(self, writer: _Writer, op: Any, a: Dict[str, Any]) -> None:
        panel = a.get("panel", 1)
        self.panel(writer, panel)
        simple = (
            ("title", "plt.title"),
            ("xlabel", "plt.xlabel"),
            ("ylabel", "plt.ylabel"),
            ("zlabel", "plt.zlabel"),
        )
        for key, fn in simple:
            if a.get(key) is not None:
                writer.add(f"{fn}({lit(a[key])})")
        for key, fn in (("xscale", "plt.xscale"), ("yscale", "plt.yscale")):
            if a.get(key) is not None:
                writer.add(f"{fn}({lit(a[key])})")
        for key, fn in (("xlim", "plt.xlim"), ("ylim", "plt.ylim"), ("zlim", "plt.set_zlim")):
            if a.get(key) is not None:
                writer.add(f"{fn}({_spread(a[key])})")
        if a.get("grid") is not None:
            writer.add(f"plt.grid({lit(bool(a['grid']))})")
        if a.get("legend"):
            writer.add("plt.legend()")
        elif a.get("legend") is False:
            writer.add("plt.legend()")
            writer.add("plt.gca().get_legend().remove()  # no legend")
        if a.get("equal_aspect") is not None:
            writer.add(f"plt.axis({'equal' if a['equal_aspect'] else 'auto'!r})")
        if a.get("hide_axes") is not None:
            writer.add(f"plt.axis({'off' if a['hide_axes'] else 'on'!r})")
        if a.get("elev") is not None or a.get("azim") is not None:
            writer.call("plt.view_init", [], {"elev": a.get("elev"), "azim": a.get("azim")})
        twin = "ax2" if len(self.twins) <= 1 else f"ax2_{panel}"
        if a.get("y2label") is not None:
            writer.add(f"{twin}.set_ylabel({lit(a['y2label'])})")
        if a.get("y2scale") is not None:
            writer.add(f"{twin}.set_yscale({lit(a['y2scale'])})")
        if a.get("y2lim") is not None:
            writer.add(f"{twin}.set_ylim({_spread(a['y2lim'])})")

    # -- dynamic values -----------------------------------------------------------------

    def value_code(
        self, spec: Any, scalars: Mapping[str, str], prop: str = "", text: bool = False
    ) -> Any:
        """A binding's value as code (a :class:`Raw`) or a literal, in a scope where each
        timeline variable is a scalar named as in ``scalars``.

        ``prop`` and ``text`` say what the value is for, so the script agrees with the server: a
        text property (a colour, a title) takes a plain string as it is, and a whole-number or
        true/false property gets an ``int`` / ``bool`` rather than the float an expression gives.
        """
        if isinstance(spec, (bool, int, float)):
            return dynamics.coerce_property(prop, spec)
        if isinstance(spec, str):
            if text:
                return spec
            code = translate(spec, self.helpers, scalars)
            leaf = prop.rsplit(".", 1)[-1]
            if leaf in dynamics.INTEGER_PROPERTIES:
                code = f"int(round({code}))"
            elif leaf in dynamics.BOOLEAN_PROPERTIES:
                code = f"bool({code})"
            return Raw(code)
        if isinstance(spec, (list, tuple)):
            return Raw("[" + ", ".join(lit(self.value_code(item, scalars)) for item in spec) + "]")
        if isinstance(spec, Mapping):
            if "colormap" in spec:
                self.uses_ramp = True
                at = translate(str(spec["at"]), self.helpers, scalars)
                return Raw(f"_ramp({lit(spec['colormap'])}, {at})")
            if "choose" in spec:
                items = list(spec["choose"])
                index = translate(str(spec["index"]), self.helpers, scalars)
                return Raw(
                    f"{lit(items)}[int(np.clip(np.floor({index} + 1e-9), 0, {len(items) - 1}))]"
                )
            if "format" in spec:
                template = spec["format"]
                used = sorted(set(re.findall(r"\{([A-Za-z_]\w*)", template)))
                fields = ", ".join(f"{name}={scalars.get(name, name)}" for name in used)
                return Raw(f"{lit(template)}.format({fields})")
        raise McpError(f"Cannot write {spec!r} as code.")


# ----------------------------------------------------------------------------------
# Whole scripts
# ----------------------------------------------------------------------------------


def _header(kind: str, entry: Any, title: str) -> List[str]:
    return [
        f'"""{title}',
        "",
        f"Generated by GLPlot's MCP server from figure {entry.figure_id}. Run it with Python;",
        "every number below is meant to be edited.",
        '"""',
        "",
    ]


def _control_lines(script: _Script, writer: _Writer) -> None:
    entry = script.entry
    for name, c in entry.controls.items():
        var = f"_{name}"
        if c.kind == "slider":
            extra = {
                k: v
                for k, v in (
                    ("step", c.step),
                    ("integer", c.integer or None),
                    ("log", c.log or None),
                )
                if v
            }
            writer.call(f"{var} = plt.slider", [c.label, c.minimum, c.maximum, c.value], extra)
        elif c.kind == "toggle":
            writer.add(f"{var} = plt.checkbox({lit(c.label)}, {lit(bool(c.value))})")
        elif c.kind == "choice":
            writer.add(
                f"{var} = plt.dropdown({lit(c.label)}, {lit(list(c.options))}, {lit(c.value)})"
            )
    for name, c in entry.controls.items():
        if c.kind != "button":
            continue
        writer.add(f"def _press_{name}():")
        with writer.indent():
            targets = (
                c.sets
                if c.sets is not None
                else {n: o.default for n, o in entry.controls.items() if o.kind != "button"}
            )
            for target, value in targets.items():
                writer.add(f"_{target}.value = {lit(value)}")
            if not targets:
                writer.add("pass")
        writer.add(f"plt.button({lit(c.label)}, _press_{name})")


def _control_scalars(entry: Any) -> Tuple[List[str], Dict[str, str]]:
    """Lines that read each control into a local, and the map of names (identity)."""
    lines: List[str] = []
    names: Dict[str, str] = {}
    for name, c in entry.controls.items():
        if c.kind == "slider":
            lines.append(f"{name} = _{name}.value")
        elif c.kind == "toggle":
            lines.append(f"{name} = 1.0 if _{name}.value else 0.0")
        elif c.kind == "choice":
            lines.append(f"{name} = float({lit(list(c.options))}.index(_{name}.value))")
        else:
            continue
        names[name] = name
    return lines, names


def _bindings_by_target(bindings: Sequence[Any], scalars: Mapping[str, str], script: _Script):
    per_artist: Dict[str, Dict[str, Any]] = {}
    per_axes: Dict[int, Dict[str, Any]] = {}
    data: List[Any] = []
    for binding in bindings:
        kind, ident = dynamics.parse_target(binding.target)
        if kind == "data":
            data.append((ident, binding))
            continue
        tool = script.box._check_property(script.entry, kind, ident, binding.prop)
        code = script.value_code(
            binding.value,
            scalars,
            prop=binding.prop,
            text=binding.prop in script.box._text_arguments(tool),
        )
        if kind == "artist":
            per_artist.setdefault(ident, {})[binding.prop] = code
        else:
            per_axes.setdefault(ident, {})[binding.prop] = code
    return per_artist, per_axes, data


def _data_rebinding(
    script: _Script,
    writer: _Writer,
    data: Sequence[Any],
    scalars: Mapping[str, str],
    *,
    restart: bool = True,
) -> None:
    """Code that recomputes the bound columns of a dataset (from a fresh copy when ``restart``)."""
    done: Dict[str, None] = {}
    for dataset, binding in data:
        var = script.ds(dataset)
        if dataset not in done and restart:
            writer.add(f"{var} = dict({var}_base)")
        done[dataset] = None
        columns = script.box.datasets.get(dataset).column_names()
        for column in columns:
            writer.add(f"{column} = {var}[{column!r}]")
        names = {k: v for k, v in scalars.items() if k not in columns}
        code = translate(str(binding.value), script.helpers, names)
        writer.add(f"{var}[{binding.prop!r}] = _as_column({code}, len({var}[{columns[0]!r}]))")
        script.helpers.setdefault("_as_column")
        if binding.prop not in columns:
            columns.append(binding.prop)


def _timeline_code(
    script: _Script, writer: _Writer, spec: Mapping[str, Any]
) -> Tuple[Any, Dict[str, str]]:
    """Code that builds the animation's variable arrays; returns the timeline and name map."""
    entry = script.entry
    controls = script.box._context_values(entry)
    timeline = dynamics.build_timeline(
        frames=spec.get("frames"),
        duration=spec.get("duration"),
        fps=spec.get("fps", 30.0),
        loop=bool(spec.get("loop")),
        stages=spec.get("stages"),
        variables=spec.get("variables"),
        controls=controls,
        limits=script.box.limits,
    )
    n = timeline.n
    writer.add(f"N = {n}")
    writer.add(f"FPS = {lit(float(timeline.fps))}")
    writer.add("_frame = np.arange(N, dtype=float)")
    writer.add("_time = _frame / FPS")
    writer.add(f"_t = _frame / {'N' if timeline.loop else '(N - 1)'}")
    writer.add("_s = _t.copy()")
    writer.add("_stage = np.zeros(N)")
    rename = {"t": "_t", "frame": "_frame", "time": "_time", "s": "_s", "stage": "_stage"}
    stages = list(spec.get("stages") or [])
    for index, span in enumerate(timeline.stages if stages else []):
        lo, hi = span["start"], span["end"]
        writer.add(f"# stage {span['name']!r}: frames {lo}..{hi - 1}")
        writer.add(f"_stage[{lo}:{hi}] = {index}")
        writer.add(
            f"_s[{lo}:{hi}] = " + ("1.0" if hi - lo == 1 else f"np.linspace(0.0, 1.0, {hi - lo})")
        )
    params = list(timeline.parameters)
    for name in params:
        start = float(timeline.variables[name][0])
        writer.add(f"_tl_{name} = np.full(N, {lit(start)})")
        rename[name] = f"_tl_{name}"
    for index, stage in enumerate(stages):
        span = timeline.stages[index]
        lo, hi = span["start"], span["end"]
        ease_name = stage.get("ease") or "linear"
        for name, move in (stage.get("animate") or {}).items():
            writer.add(f"# stage {span['name']!r}: {name}")
            arr = f"_tl_{name}"
            expression = start_value = stop_value = None
            move_ease = ease_name
            if isinstance(move, str):
                expression = move
            elif isinstance(move, (int, float)):
                stop_value = float(move)
            elif isinstance(move, (list, tuple)):
                start_value, stop_value = float(move[0]), float(move[1])
            else:
                expression = move.get("expr")
                if expression is None:
                    stop_value = float(move["to"])
                if move.get("from") is not None:
                    start_value = float(move["from"])
                move_ease = move.get("ease") or ease_name
            if expression is not None:
                local = {p: f"_tl_{p}[{lo}:{hi}]" for p in params}
                for key in ("t", "s", "time", "frame", "stage"):
                    local[key] = f"{rename[key]}[{lo}:{hi}]"
                code = translate(expression, script.helpers, local)
                writer.add(f"{arr}[{lo}:{hi}] = _as_column({code}, {hi - lo})")
                writer.add(f"{arr}[{hi}:] = {arr}[{hi - 1}]")
                script.helpers.setdefault("_as_column")
            else:
                begin = lit(start_value) if start_value is not None else f"{arr}[{lo}]"
                writer.add(f"_begin = {begin}")
                shape = "_s[%d:%d]" % (lo, hi)
                weight = shape if move_ease == "linear" else f"{_EASE_FUNCTION[move_ease]}({shape})"
                if move_ease != "linear":
                    script.helpers.setdefault(_EASE_FUNCTION[move_ease])
                writer.add(f"{arr}[{lo}:{hi}] = _begin + ({lit(stop_value)} - _begin) * {weight}")
                # what the server holds afterwards: the target, or the start for a pulse
                end = dynamics.ease_end(move_ease)
                writer.add(
                    f"{arr}[{hi}:] = "
                    + (
                        lit(stop_value)
                        if end == 1.0
                        else f"_begin + ({lit(stop_value)} - _begin) * {lit(end)}"
                    )
                )
    derived = {k: v for k, v in (spec.get("variables") or {}).items() if isinstance(v, str)}
    for name, expression in derived.items():
        rename[name] = f"_tl_{name}"
        writer.add(f"_tl_{name} = _as_column({translate(expression, script.helpers, rename)}, N)")
        script.helpers.setdefault("_as_column")
    names = {n: n for n in list(rename) + list(derived)}
    return timeline, names


_EASE_FUNCTION = {
    "in": "ease_in",
    "out": "ease_out",
    "in_out": "ease_in_out",
    "smooth": "smoothstep",
    "smoother": "smootherstep",
    "sine": "ease_sine",
    "rush_into": "rush_into",
    "rush_from": "rush_from",
    "there_and_back": "there_and_back",
}


def _array_name(name: str) -> str:
    """The timeline array a variable lives in (the clock's are ``_t``, variables ``_tl_x``)."""
    return "_" + name if name in ("t", "frame", "time", "s", "stage") else "_tl_" + name


_AS_COLUMN = '''def _as_column(value, n=None):
    """An expression's result as a 1-D float array (a single number is repeated n times)."""
    array = np.asarray(value, dtype=float)
    return array if array.ndim else np.full(n or 1, float(array))'''

_FADE = '''def _fade(color, alpha):
    """``color`` with its opacity multiplied by ``alpha``."""
    from matplotlib.colors import to_rgba

    r, g, b, a = to_rgba(color)
    return (r, g, b, a * float(alpha))'''

_RAMP = '''def _ramp(colors, at):
    """A colour (hex) at position ``at`` (0..1) along a colormap name or a list of colours."""
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap, to_hex

    if isinstance(colors, str):
        cmap = colormaps[colors]
    else:
        cmap = LinearSegmentedColormap.from_list("c", colors)
    return to_hex(cmap(float(np.clip(at, 0.0, 1.0))))'''


def export_script(
    box: Any, entry: Any, kind: str = "figure", path: Optional[str] = None
) -> Tuple[str, List[str]]:
    """The script for ``entry`` and a list of things to know about it."""
    if kind not in SCRIPT_KINDS:
        raise McpError(f"kind must be one of: {', '.join(SCRIPT_KINDS)}.")
    if not entry.log:
        raise McpError(f"{entry.figure_id} has nothing drawn yet: add something first.")
    script = _Script(box, entry, kind)
    script.imports_scipy = False
    script.imports_recipes = False
    spec = entry.animation
    if kind == "animation" and not spec:
        raise McpError(
            f"{entry.figure_id} has no animation yet: call animate(...) first (it records the "
            "specification this script is written from)."
        )
    if kind == "interactive" and not entry.controls:
        raise McpError(
            f"{entry.figure_id} has no controls: add_control(...) and bind_property(...) first."
        )

    # The figure's operations are written first (into ``body``) so that we know which datasets,
    # helpers and imports they use before the preamble is written.
    body = script.body
    scalars: Dict[str, str] = {}
    out = f"{entry.figure_id}"
    if kind == "figure":
        # Bindings are evaluated once, at the controls' current values; the script states the
        # result (data columns a binding recomputes are recomputed in the script).
        context = {n: np.full(1, v) for n, v in box._context_values(entry).items()}
        artist_args, axes_args, data = box._bound_arguments(entry, entry.bindings, context, 1)
        literal = {n: lit(float(v[0])) for n, v in context.items()}
        script.figure(body)
        pairs = [(dynamics.parse_target(b.target)[1], b) for b in data]
        _data_rebinding(script, body, pairs, literal, restart=False)
        script.ops(
            body,
            {a: dict(p) for a, p in artist_args[0].items()},
            {p: dict(v) for p, v in axes_args[0].items()},
        )
        body.add("")
        body.add(f"plt.savefig({out + '.png'!r}, bbox_inches='tight', pad_inches=0.15)")
        body.add("# plt.show()   # to look at it in a window instead")
    elif kind == "interactive":
        reads, scalars = _control_scalars(entry)
        script.figure(body)
        body.add()
        _control_lines(script, body)
        body.add()
        body.add("def build():")
        with body.indent():
            body.add('"""Redraw the figure at the current control values."""')
            for line in reads:
                body.add(line)
            per_artist, per_axes, data = _bindings_by_target(entry.bindings, scalars, script)
            if entry.spec["rows"] * entry.spec["cols"] > 1:
                body.add("for _ax in axs.flat:")
                with body.indent():
                    body.add("plt.sca(_ax)")
                    body.add("plt.cla()")
            else:
                body.add("plt.cla()")
            if entry.spec.get("title") and entry.spec["rows"] * entry.spec["cols"] == 1:
                body.add(f"plt.title({entry.spec['title']!r})")
            script.panel_now = None
            script.twins = set()
            _data_rebinding(script, body, data, scalars)
            script.ops(body, per_artist, per_axes)
        body.add()
        body.add("# redraw whenever a control moves")
        for name, c in entry.controls.items():
            if c.kind != "button":
                body.add(f"_{name}.on_change(lambda _value: build())")
        body.add("build()")
        body.add("plt.show()")
    else:
        script.figure(body)
        body.add()
        body.add("# ---- the timeline: every variable at every frame ----")
        timeline, names = _timeline_code(script, body, spec)
        extra = _bindings_from(spec.get("bindings"))
        merged = {(b.target, b.prop): b for b in entry.bindings}
        merged.update({(b.target, b.prop): b for b in extra})
        local = {n: n for n in names}
        per_artist, per_axes, data = _bindings_by_target(list(merged.values()), local, script)
        body.add()
        body.add("def update(i):")
        with body.indent():
            body.add('"""Draw frame i: every bound property takes that frame\'s value."""')
            for n in names:
                body.add(f"{n} = {_array_name(n)}[i]")
            body.add("n = float(N)")
            if entry.spec["rows"] * entry.spec["cols"] > 1:
                body.add("for _ax in axs.flat:")
                with body.indent():
                    body.add("plt.sca(_ax)")
                    body.add("plt.cla()")
            else:
                body.add("plt.cla()")
            if entry.spec.get("title") and entry.spec["rows"] * entry.spec["cols"] == 1:
                body.add(f"plt.title({entry.spec['title']!r})")
            script.panel_now = None
            script.twins = set()
            _data_rebinding(script, body, data, local)
            script.ops(body, per_artist, per_axes)
            body.add("return []")
        body.add()
        body.add("ani = animation.FuncAnimation(fig, update, frames=N, interval=1000.0 / FPS)")
        body.add("")
        body.add("if __name__ == '__main__':")
        with body.indent():
            body.add(f"ani.save({out + '.gif'!r}, fps=FPS)   # a .mp4 name needs ffmpeg")
            body.add("# plt.show()   # play it in a window instead")

    # -- preamble -------------------------------------------------------------------
    # The datasets are written first (into ``data``): rebuilding them is what decides whether
    # scipy, numpy helpers or the column helper are needed.
    data = _Writer()
    script.data_section(data)
    if kind in ("animation", "interactive"):
        # Datasets that data bindings recompute keep a pristine copy to restart from.
        bound = list(entry.bindings) + (
            _bindings_from(spec.get("bindings")) if kind == "animation" and spec else []
        )
        for name in script.datasets:
            if any(dynamics.parse_target(b.target) == ("data", name) for b in bound):
                var = "d_" + _ident(name)
                data.add(f"{var}_base = {var}")
    head = _Writer()
    for line in _header(
        kind,
        entry,
        {"figure": "A figure", "interactive": "An interactive figure", "animation": "An animation"}[
            kind
        ],
    ):
        head.add(line)
    head.add("from __future__ import annotations")
    head.add()
    head.add("import numpy as np")
    if kind == "animation":
        head.add("import glplot.animation as animation")
    head.add("import glplot.pyplot as plt")
    if script.imports_scipy:
        head.add("from scipy.integrate import solve_ivp")
    if script.imports_recipes:
        head.add("from glplot.gui import recipes")
    head.add()
    if "_as_column" in script.helpers:
        head.add(_AS_COLUMN)
        head.add()
    if any(h in script.helpers for h in dynamics.HELPERS):
        head.add(textwrap.dedent(inspect.getsource(expressions._unit)).rstrip())
        head.add()
    for name in list(script.helpers):
        if name != "_as_column" and (name in dynamics.HELPERS or name in SAFE_NAMES):
            head.add(_helper_source(name))
            head.add()
    if script.uses_ramp:
        head.add(_RAMP)
        head.add()
    if script.uses_fade:
        head.add(_FADE)
        head.add()
    if script.uses_reveal_text:
        head.add(textwrap.dedent(inspect.getsource(dynamics.reveal_text)).rstrip())
        head.add()
    head.lines.extend(data.lines)
    text = head.text().rstrip() + "\n\n" + body.text().rstrip() + "\n"
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text, script.notes


def _bindings_from(items: Optional[Sequence[Mapping[str, Any]]]) -> List[Any]:
    out: List[Any] = []
    for item in items or []:
        out.append(dynamics.Binding(str(item["target"]), str(item["property"]), item["value"]))
    return out
