"""State and guard rails behind the MCP server. Pure logic: no MCP SDK, no GL, no imgui.

A language model is driving the other end of this server, which changes what "bad input"
means. A person who types ``arange(1e12)`` into the workstation's expression box is
hurting their own session; a model that writes it is making an ordinary mistake it cannot
see coming. Everything here exists to turn those mistakes into a short, readable
:class:`McpError` instead of a frozen process or a read of ``~/.ssh``:

* :class:`Limits` -- one place that says how big things may get.
* :class:`FileSandbox` -- the only door to the filesystem. Reads and writes are confined to
  explicit root directories, to a fixed set of extensions, and writes never overwrite
  unless asked to.
* :class:`DatasetStore` -- named tables the model builds once and plots many times, so a
  large series never has to travel through the conversation.
* :func:`evaluate_expression` -- the GUI's AST-validated evaluator
  (:mod:`glplot.gui.expressions`), with its array *constructors* swapped for size-capped
  ones. That evaluator already refuses attribute access, builtins and runaway ``**``; what
  it does not bound is ``zeros(10**12)``, which is fine behind a text box and not fine
  behind a tool call.

Every message a model can read is written to be acted on: it names the offending argument
and says what would have been accepted.
"""

from __future__ import annotations

import difflib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from ..gui import dataio
from ..gui.datasets import DataSet
from ..gui.expressions import SAFE_NAMES, ExpressionError, evaluate

__all__ = [
    "DEFAULT_LIMITS",
    "DatasetStore",
    "FileSandbox",
    "Limits",
    "McpError",
    "evaluate_expression",
    "expression_vocabulary",
    "suggest",
]


class McpError(Exception):
    """A failure the model should read and act on: a bad argument, a missing figure.

    Distinct from a crash. The server adapter turns this into the SDK's ``ToolError``, whose
    message reaches the model verbatim; any *other* exception is reported to it only as
    "error executing tool", because an arbitrary traceback is neither safe nor useful to
    hand over.
    """


@dataclass(frozen=True)
class Limits:
    """How big things may get. Frozen: a server's limits do not change under it."""

    #: Values accepted inline in one tool call, per array. Past this the model should write
    #: a file and ``load_dataset`` it -- a 200k-number argument is most of a context window.
    max_inline_values: int = 200_000
    #: Rows in a dataset, whether loaded from a file or built from expressions.
    max_dataset_rows: int = 5_000_000
    #: Elements any array built by an expression constructor (``linspace``, ``zeros``...)
    #: may have, and cells in a 2-D field (``resolution ** 2``).
    max_array_elements: int = 4_000_000
    #: Figures kept alive at once. Each holds its layers' arrays.
    max_figures: int = 24
    #: Datasets kept alive at once.
    max_datasets: int = 64
    #: Figure size bounds, in pixels, per side.
    min_figure_px: int = 200
    max_figure_px: int = 3000
    #: Longest edge of an image handed back to the model. Larger pictures are downscaled by
    #: lowering the render dpi: a model sees roughly this many pixels whatever it asked for.
    max_image_edge: int = 1600
    #: Largest file ``load_dataset`` / ``add_field`` will open.
    max_read_bytes: int = 256 * 1024 * 1024
    #: Longest expression, in characters. The evaluator has its own (larger) cap; this is
    #: the one a model is told about.
    max_expression_chars: int = 2000


#: The limits a server uses unless told otherwise. ``Limits`` is frozen, so sharing one
#: instance as a default argument is safe.
DEFAULT_LIMITS = Limits()


# ----------------------------------------------------------------------------------
# Suggestions
# ----------------------------------------------------------------------------------


def suggest(name: str, options: Iterable[str]) -> str:
    """``" Did you mean 'x'?"`` for a near miss, else an empty string. For error messages."""
    close = difflib.get_close_matches(str(name), [str(o) for o in options], n=1, cutoff=0.6)
    return f" Did you mean {close[0]!r}?" if close else ""


# ----------------------------------------------------------------------------------
# Filesystem
# ----------------------------------------------------------------------------------


class FileSandbox:
    """The only way a tool touches the filesystem.

    A path is accepted when, after symlinks are resolved, it sits inside one of ``roots``.
    Resolution happens *before* the containment test, so a symlink inside a root that
    points outside it is refused rather than followed.

    With no roots every call raises: file access is opt-in. A server launched by an MCP
    client inherits whatever working directory the client had, so the working directory is
    never a default root.
    """

    #: What ``load_dataset`` and ``add_field(file=...)`` may open.
    READ_EXTENSIONS: Sequence[str] = dataio.SUPPORTED_EXTENSIONS
    #: What ``save_figure`` may write. Image formats only: a tool that can write ``.py`` or
    #: ``.sh`` next to the user's work is a tool that can be talked into planting one.
    WRITE_EXTENSIONS: Sequence[str] = (".png", ".svg", ".pdf", ".jpg", ".jpeg")

    def __init__(
        self,
        roots: Sequence[os.PathLike[str] | str] = (),
        *,
        max_read_bytes: int = Limits.max_read_bytes,
    ) -> None:
        resolved: List[Path] = []
        for root in roots:
            path = Path(root).expanduser().resolve()
            if not path.is_dir():
                raise ValueError(f"Sandbox root {str(root)!r} is not a directory.")
            if path == Path(path.anchor):
                raise ValueError(
                    f"Refusing {str(path)!r} as a sandbox root: it is the whole filesystem. "
                    "Pass the directory that holds your data."
                )
            if path not in resolved:
                resolved.append(path)
        self._roots: List[Path] = resolved
        self._max_read_bytes = int(max_read_bytes)

    @property
    def roots(self) -> List[Path]:
        """The accepted root directories, resolved. The first is where relative paths go."""
        return list(self._roots)

    @property
    def enabled(self) -> bool:
        """False when no root was given, in which case every file operation is refused."""
        return bool(self._roots)

    def _locate(self, path: str) -> Path:
        if not self._roots:
            raise McpError(
                "File access is disabled on this server. Restart it with --root DIRECTORY "
                "(repeatable) to allow reading data from, and saving images into, that "
                "directory. Until then use create_dataset with inline values or expressions."
            )
        if not isinstance(path, str) or not path.strip():
            raise McpError("path must be a non-empty string.")
        candidate = Path(path.strip()).expanduser()
        if not candidate.is_absolute():
            candidate = self._roots[0] / candidate
        resolved = candidate.resolve()
        if not any(resolved == root or root in resolved.parents for root in self._roots):
            allowed = ", ".join(str(r) for r in self._roots)
            raise McpError(
                f"{path!r} is outside the directories this server may use ({allowed}). "
                "Use a path inside one of them; relative paths are taken from the first."
            )
        return resolved

    def read_path(self, path: str) -> Path:
        """Resolve ``path`` for reading: inside a root, a regular file, a known extension."""
        resolved = self._locate(path)
        if not resolved.exists():
            raise McpError(f"File not found: {path!r} (looked for {resolved}).")
        if not resolved.is_file():
            raise McpError(f"{path!r} is not a regular file.")
        extension = resolved.suffix.lower()
        if extension not in self.READ_EXTENSIONS:
            raise McpError(
                f"Cannot read {extension or 'a file with no extension'!r} files. "
                f"Supported: {', '.join(self.READ_EXTENSIONS)}."
            )
        size = resolved.stat().st_size
        if size > self._max_read_bytes:
            raise McpError(
                f"{path!r} is {size / 1e6:.0f} MB; this server reads at most "
                f"{self._max_read_bytes / 1e6:.0f} MB. Downsample it first."
            )
        return resolved

    def write_path(self, path: str, *, overwrite: bool = False) -> Path:
        """Resolve ``path`` for writing an image; refuses to replace a file unless asked."""
        resolved = self._locate(path)
        extension = resolved.suffix.lower()
        if extension not in self.WRITE_EXTENSIONS:
            raise McpError(
                f"Cannot save a {extension or 'extensionless'!r} file. "
                f"Use one of: {', '.join(self.WRITE_EXTENSIONS)}."
            )
        if resolved.is_dir():
            raise McpError(f"{path!r} is a directory; give a file name such as 'figure.png'.")
        if resolved.exists() and not overwrite:
            raise McpError(
                f"{path!r} already exists. Pass overwrite=true to replace it, or choose "
                "another name."
            )
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved


# ----------------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------------

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def check_name(name: str, what: str = "name") -> str:
    """Validate a dataset name: 1-64 characters of letters, digits, ``_``, ``.`` and ``-``."""
    if not isinstance(name, str) or not _NAME_PATTERN.match(name):
        raise McpError(
            f"Invalid {what} {name!r}: use 1-64 characters from letters, digits, '_', '.' "
            "and '-' (no spaces)."
        )
    return name


def slug(text: str, fallback: str = "data") -> str:
    """A valid dataset name derived from arbitrary text, such as a file's base name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.\-]+", "_", str(text)).strip("._-")[:64]
    return cleaned or fallback


def is_identifier(name: str) -> bool:
    """True if ``name`` can be referenced from an expression."""
    return bool(_IDENTIFIER.match(name)) and "__" not in name


class DatasetStore:
    """Named :class:`~glplot.gui.datasets.DataSet` tables, bounded in count and size.

    The GUI's own table type, on purpose: equal-length float64 columns, cached statistics,
    and the importers in :mod:`glplot.gui.dataio` produce it directly, so a file means
    exactly what it means in the workstation's Data panel.
    """

    def __init__(self, limits: Limits = DEFAULT_LIMITS) -> None:
        self._limits = limits
        self._sets: Dict[str, DataSet] = {}

    def __len__(self) -> int:
        return len(self._sets)

    def __contains__(self, name: object) -> bool:
        return name in self._sets

    def names(self) -> List[str]:
        """Dataset names in creation order."""
        return list(self._sets)

    def check_room(self, name: str) -> None:
        """Raise :class:`McpError` if a dataset called ``name`` could not be stored now.

        Lets a caller that stores a dataset *after* doing something else (drawing a recipe from
        its sample table) find out first, instead of drawing and then failing to keep the data.
        """
        if name not in self._sets and len(self._sets) >= self._limits.max_datasets:
            raise McpError(
                f"Too many datasets ({self._limits.max_datasets}). Delete one with "
                f"delete_dataset first: {', '.join(self._sets)}."
            )

    def unique_name(self, base: str) -> str:
        """``base``, or ``base_2``, ``base_3``... -- the first name not already taken."""
        if base not in self._sets:
            return base
        index = 2
        while f"{base}_{index}" in self._sets:
            index += 1
        return f"{base}_{index}"

    def put(self, dataset: DataSet) -> bool:
        """Store ``dataset`` under its own name; True when it replaced an existing one."""
        check_name(dataset.name, "dataset name")
        replaced = dataset.name in self._sets
        self.check_room(dataset.name)
        if dataset.n_rows() > self._limits.max_dataset_rows:
            raise McpError(
                f"Dataset {dataset.name!r} has {dataset.n_rows()} rows; the limit is "
                f"{self._limits.max_dataset_rows}. Downsample it first."
            )
        self._sets[dataset.name] = dataset
        return replaced

    def get(self, name: str) -> DataSet:
        """The dataset called ``name``, or an :class:`McpError` that lists what exists."""
        if name in self._sets:
            return self._sets[name]
        known = ", ".join(self._sets) if self._sets else "none yet"
        raise McpError(
            f"No dataset named {name!r} (available: {known}).{suggest(name, self._sets)} "
            "Create one with create_dataset or load_dataset."
        )

    def remove(self, name: str) -> None:
        """Forget ``name``; raises :class:`McpError` if it does not exist."""
        self.get(name)
        del self._sets[name]

    def column(self, dataset: str, column: str) -> np.ndarray:
        """A copy of one column's values, with a helpful error for a wrong column name."""
        table = self.get(dataset)
        values = table.get(column)
        if values is None:
            names = table.column_names()
            raise McpError(
                f"Dataset {dataset!r} has no column {column!r} "
                f"(columns: {', '.join(names)}).{suggest(column, names)}"
            )
        return np.array(values, dtype=np.float64)

    def describe(self, name: str, rows: int = 5) -> Dict[str, Any]:
        """A JSON-safe summary: shape, per-column statistics and the first ``rows`` rows."""
        table = self.get(name)
        rows = max(0, min(int(rows), 50))
        columns = []
        for column in table.columns:
            stats = column.stats()
            columns.append(
                {
                    "name": column.name,
                    **{key: _finite_or_none(value) for key, value in stats.items()},
                    "nan_count": int(np.isnan(column.values).sum()),
                }
            )
        head = [
            [_finite_or_none(column.values[i]) for column in table.columns]
            for i in range(min(rows, table.n_rows()))
        ]
        return {
            "name": table.name,
            "rows": table.n_rows(),
            "columns": columns,
            "head": head,
            "source": table.source,
        }


def _finite_or_none(value: Any) -> Optional[float]:
    """``nan``/``inf`` are not valid JSON; report them as ``null``."""
    number = float(value)
    return number if math.isfinite(number) else None


# ----------------------------------------------------------------------------------
# Expressions
# ----------------------------------------------------------------------------------


def _count(value: Any, what: str, limit: int) -> int:
    """Validate an element count taken from an expression and return it as an ``int``."""
    try:
        number = float(np.prod(np.asarray(value, dtype=np.float64)))
    except (TypeError, ValueError) as exc:
        raise ExpressionError(f"{what}: expected a number of elements, got {value!r}") from exc
    if not math.isfinite(number) or number < 0:
        raise ExpressionError(f"{what}: {value!r} is not a valid number of elements")
    if number > limit:
        raise ExpressionError(
            f"{what}: {number:.4g} elements exceeds the limit of {limit}. "
            "Use fewer points (an array of a few thousand is plenty for a plot)."
        )
    return int(number)


def guarded_constructors(limit: int) -> Dict[str, Callable[..., Any]]:
    """Size-capped stand-ins for the evaluator's array constructors.

    The evaluator lets a bound variable shadow a safe name, so these are passed as ordinary
    ``variables``: no change to :mod:`glplot.gui.expressions`, which the workstation relies
    on being unrestricted.
    """
    base_noise = SAFE_NAMES["noise"]

    def linspace(start, stop, num=50, *args, **kwargs):
        _count(num, "linspace()", limit)
        return np.linspace(start, stop, int(num), *args, **kwargs)

    def arange(start, stop=None, step=1.0, *args, **kwargs):
        low, high = (0.0, start) if stop is None else (start, stop)
        if float(step) == 0.0:
            raise ExpressionError("arange(): step must not be zero")
        _count(max(0.0, math.ceil((float(high) - float(low)) / float(step))), "arange()", limit)
        return np.arange(start, stop, step, *args, **kwargs)

    def zeros(shape, *args, **kwargs):
        _count(shape, "zeros()", limit)
        return np.zeros(shape, *args, **kwargs)

    def ones(shape, *args, **kwargs):
        _count(shape, "ones()", limit)
        return np.ones(shape, *args, **kwargs)

    def full(shape, fill_value, *args, **kwargs):
        _count(shape, "full()", limit)
        return np.full(shape, fill_value, *args, **kwargs)

    def noise(n_or_x, scale=1.0, seed=None):
        arr = np.asarray(n_or_x)
        _count(float(arr) if arr.ndim == 0 else arr.shape, "noise()", limit)
        return base_noise(n_or_x, scale, seed)

    return {
        "linspace": linspace,
        "arange": arange,
        "zeros": zeros,
        "ones": ones,
        "full": full,
        "noise": noise,
    }


def evaluate_expression(
    expression: str,
    variables: Optional[Mapping[str, Any]] = None,
    *,
    limits: Limits = DEFAULT_LIMITS,
    what: str = "expression",
) -> Any:
    """Evaluate ``expression`` safely, with ``variables`` bound, and return the raw result.

    The model-facing wrapper around :func:`glplot.gui.expressions.evaluate`: the same
    validated AST and the same names, but with capped array constructors, a length limit,
    and every failure reported as an :class:`McpError` that quotes the expression.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise McpError(f"{what} must be a non-empty string.")
    if len(expression) > limits.max_expression_chars:
        raise McpError(
            f"{what} is {len(expression)} characters; the limit is "
            f"{limits.max_expression_chars}. Split it into several columns."
        )
    bindings: Dict[str, Any] = guarded_constructors(limits.max_array_elements)
    for key, value in (variables or {}).items():
        if is_identifier(key):
            bindings[key] = value
    try:
        return evaluate(expression, bindings)
    except ExpressionError as exc:
        raise McpError(f"Could not evaluate {what} {expression!r}: {exc}") from exc


def as_float_array(
    value: Any, what: str, *, length: Optional[int] = None, limits: Limits = DEFAULT_LIMITS
) -> np.ndarray:
    """Coerce an expression result or an inline list to a 1-D float64 array.

    A scalar is broadcast to ``length`` when one is given. Anything else that is not a
    plain real 1-D numeric array is an :class:`McpError` naming ``what``.
    """
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise McpError(f"{what}: could not read the values as numbers ({exc}).") from exc
    if np.iscomplexobj(array):
        raise McpError(f"{what}: complex values; use abs(...) to plot the magnitude.")
    if not (np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)):
        raise McpError(
            f"{what}: expected numbers but got {array.dtype}. Strings cannot be plotted; "
            "pass category names through tick_labels instead."
        )
    array = array.astype(np.float64, copy=False)
    if array.ndim == 0:
        if length is None:
            raise McpError(f"{what}: a single number needs a length to repeat to.")
        return np.full(length, float(array))
    if array.ndim != 1:
        raise McpError(f"{what}: expected a flat list of numbers, got shape {array.shape}.")
    if array.size > limits.max_dataset_rows:
        raise McpError(
            f"{what}: {array.size} values exceeds the limit of {limits.max_dataset_rows}."
        )
    if length is not None and array.size != length:
        if array.size == 1:
            return np.full(length, float(array[0]))
        raise McpError(f"{what}: has {array.size} values but the other columns have {length}.")
    return np.ascontiguousarray(array)


def expression_vocabulary() -> Dict[str, List[str]]:
    """The names an expression may use, grouped for the model-facing guide.

    Generated from :data:`glplot.gui.expressions.SAFE_NAMES` so the guide cannot drift from
    what the evaluator really accepts.
    """
    constants = [k for k, v in SAFE_NAMES.items() if not callable(v)]
    functions = sorted(k for k, v in SAFE_NAMES.items() if callable(v))
    return {"functions": functions, "constants": sorted(constants)}
