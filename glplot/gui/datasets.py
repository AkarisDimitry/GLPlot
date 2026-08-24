"""The GUI data model: named tables of equal-length float64 columns.

Pure logic -- numpy and stdlib only, no imgui, no engine imports (CONTRACT 5.1 rule 7).
The only engine contact is duck-typed: :meth:`DataSet.write_back` reads and assigns plain
attributes on a layer object that is passed in.

Layer backing strategy (the float32-vs-float64 resolution)
---------------------------------------------------------
Engine layers store geometry as **float32** ``(N, k)`` arrays (``pts``, ``ab``,
``vertices``). A float64 *view* of a float32 column is not merely undesirable, it is
impossible: ``pts[:, 0].view(np.float64)`` raises ``ValueError: To change to a dtype of a
different size, the last axis must be contiguous``, and ``astype`` never shares memory.

So this module uses **copy-in / explicit write-back**, not views:

* :meth:`DataStore.from_layer` copies the layer's arrays into float64 columns and records
  a :class:`LayerBinding` (which attribute, which column names, which dtype).
* :meth:`DataSet.write_back` reassembles the bound columns into a C-contiguous array in
  the layer's native dtype, assigns it, and sets ``layer.dirty.gpu_dirty = True``.

``Column.values`` is therefore **always float64 and always owned** -- never a view into a
layer. Three reasons this beats views, beyond the dtype being unrepresentable:

1. Views cannot survive structural edits. ``insert_row`` / ``delete_row`` / ``add_column``
   must reallocate, which silently severs a view and kills the live link exactly when the
   user thinks it is working. Write-back handles row and column growth natively.
2. A view write from ``set_cell`` would mutate live GPU-backed data from inside an imgui
   draw callback, which CONTRACT 1.1 forbids, and would leave ``gpu_dirty`` unset.
3. The data editor already routes commits through a queued ``update_layer_xy``, so it
   never depended on view aliasing to begin with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_STAT_KEYS = ("min", "max", "mean", "std")

# layer_type -> (attribute holding the geometry, default column names)
_LAYER_BINDINGS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "scatter": ("pts", ("x", "y")),
    "polyline": ("pts", ("x", "y")),
    "line_family": ("ab", ("a", "b")),
    "geometry3d": ("vertices", ("x", "y", "z")),
    "scatter3d": ("vertices", ("x", "y", "z")),
    "volume3d": ("vertices", ("x", "y", "z")),
    "wireframe3d": ("vertices", ("x", "y", "z")),
    "surface3d": ("vertices", ("x", "y", "z")),
}

# Layer types with no editable tabular form. ``patch`` is listed explicitly because it
# also carries a ``vertices`` attribute and would otherwise be caught by the duck-typed
# fallback below.
_UNSUPPORTED_LAYER_TYPES = frozenset({"text", "patch", "semantic_line", "semantic_span"})


def _next_available(base: str, taken: Sequence[str]) -> str:
    """Return ``base`` if free, else ``base (2)``, ``base (3)``, ..."""
    if base not in taken:
        return base
    index = 2
    while f"{base} ({index})" in taken:
        index += 1
    return f"{base} ({index})"


@dataclass(eq=False)
class Column:
    """One named float64 column.

    ``eq=False`` is required: a generated ``__eq__`` would compare ndarrays and raise
    "truth value of an array is ambiguous".

    ``values`` is coerced to 1-D float64 on construction and is always owned by this
    Column -- never a view into an engine layer (see the module docstring).
    """

    name: str
    values: np.ndarray
    _stats: Optional[Dict[str, float]] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1:
            values = np.ravel(values)
        self.values = values

    def invalidate(self) -> None:
        """Drop the cached statistics. Call after any in-place mutation of ``values``."""
        self._stats = None

    def stats(self) -> Dict[str, float]:
        """Return nan-safe ``{min, max, mean, std}``, cached until :meth:`invalidate`.

        Caching is keyed on nothing at all -- it is invalidated explicitly by every
        mutator on :class:`DataSet`. That is deliberate: an ``id()``-keyed cache is unsafe
        because CPython reuses ids of freed arrays, and it cannot see in-place writes.
        """
        if self._stats is None:
            values = self.values
            if values.size == 0 or bool(np.all(np.isnan(values))):
                self._stats = dict.fromkeys(_STAT_KEYS, float("nan"))
            else:
                self._stats = {
                    "min": float(np.nanmin(values)),
                    "max": float(np.nanmax(values)),
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                }
        return self._stats

    @property
    def min(self) -> float:
        """Nan-safe minimum (nan if the column is empty or all-nan)."""
        return self.stats()["min"]

    @property
    def max(self) -> float:
        """Nan-safe maximum (nan if the column is empty or all-nan)."""
        return self.stats()["max"]

    @property
    def mean(self) -> float:
        """Nan-safe mean (nan if the column is empty or all-nan)."""
        return self.stats()["mean"]

    @property
    def std(self) -> float:
        """Nan-safe standard deviation (nan if the column is empty or all-nan)."""
        return self.stats()["std"]


@dataclass(eq=False)
class LayerBinding:
    """Records how a :class:`DataSet` maps back onto an engine layer.

    Attributes
    ----------
    attr
        The layer attribute holding the geometry: ``"pts"``, ``"ab"`` or ``"vertices"``.
    columns
        Dataset column names, in geometry-column order. Kept in sync by
        :meth:`DataSet.rename_column`.
    dtype
        The layer's native dtype at bind time, used as the write-back fallback.
    kind
        The ``layerops`` plot kind the layer was built with (``"line"``, ``"bar"``...),
        or None for a layer bound without one. Advisory here -- this module never
        imports ``layerops``; it is carried so the editor can re-derive the geometry.
    derived
        True when the layer's geometry is *computed* from ``columns`` rather than being
        them (a bar's 4N corners, a step's staircase). :meth:`DataSet.write_back` and
        :meth:`DataSet.sync_from_layer` both refuse on a derived binding: column-stacking
        N table rows onto 4N corners would not raise, it would quietly draw nonsense.
    """

    attr: str
    columns: List[str]
    dtype: np.dtype
    kind: Optional[str] = None
    derived: bool = False


def _resolve_layer_binding(layer: Any) -> Optional[Tuple[str, List[str], np.ndarray]]:
    """Resolve ``layer`` to ``(attr, column_names, array)``, or None if not tabular."""
    layer_type = getattr(layer, "layer_type", "") or ""
    if layer_type in _UNSUPPORTED_LAYER_TYPES:
        return None

    spec = _LAYER_BINDINGS.get(layer_type)
    if spec is None:
        # Unknown layer type: duck-type it. Order matters -- ``ab`` before ``pts``.
        for attr, names in (("ab", ("a", "b")), ("pts", ("x", "y")), ("vertices", ("x", "y", "z"))):
            if getattr(layer, attr, None) is not None:
                spec = (attr, names)
                break
    if spec is None:
        return None

    attr, default_names = spec
    array = getattr(layer, attr, None)
    if array is None:
        return None
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[1] < 1:
        return None

    n_cols = array.shape[1]
    names = list(default_names[:n_cols])
    # Tolerate geometry wider than the default names rather than truncating data.
    names.extend(f"c{i + 1}" for i in range(len(names), n_cols))
    return attr, names, array


class DataSet:
    """A named table of equal-length float64 columns."""

    def __init__(self, name: str, columns: Optional[List[Column]] = None) -> None:
        self.name = name
        self.columns: List[Column] = list(columns) if columns else []
        self.layer_id: Optional[int] = None
        self.source: str = "manual"
        self.binding: Optional[LayerBinding] = None

        lengths = {len(column.values) for column in self.columns}
        if len(lengths) > 1:
            raise ValueError(
                f"DataSet {name!r}: all columns must have equal length, "
                f"got lengths {sorted(lengths)}."
            )

    def __repr__(self) -> str:
        return f"<DataSet {self.name!r} {self.n_rows()}x{self.n_cols()} source={self.source!r}>"

    # -- shape -----------------------------------------------------------------

    def n_rows(self) -> int:
        """Number of rows; 0 when the dataset has no columns."""
        return len(self.columns[0].values) if self.columns else 0

    def n_cols(self) -> int:
        """Number of columns."""
        return len(self.columns)

    def column_names(self) -> List[str]:
        """The column names, in order."""
        return [column.name for column in self.columns]

    # -- lookup ----------------------------------------------------------------

    def _find(self, name: str) -> Optional[Column]:
        for column in self.columns:
            if column.name == name:
                return column
        return None

    def get(self, name: str) -> Optional[np.ndarray]:
        """Return the live values array of column ``name``, or None if absent."""
        column = self._find(name)
        return None if column is None else column.values

    def index_of(self, name: str) -> int:
        """Return the position of column ``name``, or -1 if absent."""
        for i, column in enumerate(self.columns):
            if column.name == name:
                return i
        return -1

    # -- columns ---------------------------------------------------------------

    def add_column(self, name: str, values: Optional[np.ndarray] = None) -> Column:
        """Append a column, returning it.

        ``values=None`` fills with ``zeros(n_rows)``. When the dataset is empty, the first
        column defines ``n_rows``. Name collisions are suffixed ``" (2)"``, ``" (3)"``...
        so the returned Column's ``name`` may differ from the requested one.
        """
        if values is None:
            data = np.zeros(self.n_rows(), dtype=np.float64)
        else:
            data = np.ravel(np.asarray(values, dtype=np.float64))
            if self.columns and len(data) != self.n_rows():
                raise ValueError(
                    f"Column {name!r} has {len(data)} values but the dataset has "
                    f"{self.n_rows()} rows."
                )
        column = Column(_next_available(name, self.column_names()), data)
        self.columns.append(column)
        return column

    def remove_column(self, name: str) -> bool:
        """Remove column ``name``. Returns False if it does not exist."""
        column = self._find(name)
        if column is None:
            return False
        self.columns.remove(column)
        if self.binding is not None and name in self.binding.columns:
            # The geometry can no longer be reassembled; drop the live link rather than
            # write back a wrong-shaped array.
            self.unbind()
        return True

    def move_column(self, name: str, index: int) -> bool:
        """Move column ``name`` to position ``index`` (clamped). False if it is absent.

        Deliberately does **not** unbind, where :meth:`remove_column` must: the binding
        names its columns, it does not index them, so reordering the table cannot make
        the geometry unreassemblable. The layer keeps its x/y; only the spreadsheet's
        column order changes.
        """
        current = self.index_of(name)
        if current < 0:
            return False
        target = max(0, min(int(index), self.n_cols() - 1))
        if target == current:
            return True
        self.columns.insert(target, self.columns.pop(current))
        return True

    def rename_column(self, old: str, new: str) -> bool:
        """Rename a column. Returns False if ``old`` does not exist.

        Collisions with other columns are suffixed as in :meth:`add_column`. A bound
        column's name is updated in the :class:`LayerBinding` too, so renaming does not
        break the layer link.
        """
        column = self._find(old)
        if column is None:
            return False
        if new == old:
            return True
        others = [name for name in self.column_names() if name != old]
        resolved = _next_available(new, others)
        column.name = resolved
        if self.binding is not None:
            self.binding.columns = [
                resolved if name == old else name for name in self.binding.columns
            ]
        return True

    # -- rows ------------------------------------------------------------------

    def _check_row(self, index: int) -> int:
        n_rows = self.n_rows()
        if index < 0:
            index += n_rows
        if not 0 <= index < n_rows:
            raise IndexError(f"Row {index} out of range for {n_rows} rows.")
        return index

    def insert_row(self, index: int) -> None:
        """Insert a zero-filled row at ``index`` (clamped to ``[0, n_rows]``)."""
        index = max(0, min(int(index), self.n_rows()))
        for column in self.columns:
            column.values = np.insert(column.values, index, 0.0)
            column.invalidate()

    def delete_row(self, index: int) -> None:
        """Delete the row at ``index``. Negative indices count from the end."""
        index = self._check_row(index)
        for column in self.columns:
            column.values = np.delete(column.values, index)
            column.invalidate()

    def delete_rows(self, indices: Sequence[int]) -> None:
        """Delete several rows at once. Out-of-range and duplicate indices are ignored."""
        n_rows = self.n_rows()
        valid = sorted({i + n_rows if i < 0 else i for i in indices})
        valid = [i for i in valid if 0 <= i < n_rows]
        if not valid:
            return
        for column in self.columns:
            column.values = np.delete(column.values, valid)
            column.invalidate()

    def move_rows(self, indices: Sequence[int], dest: int) -> bool:
        """Move the rows at ``indices`` to just before the row currently at ``dest``.

        The moved rows keep their relative order and end up contiguous, and every column
        is permuted identically -- a row is a row across the whole table, so moving one
        in a single column would silently re-pair the data.

        ``dest`` names a row in the table *as it is now* (which is what a drag-and-drop
        gesture knows: "I dropped on row 7"), and is resolved against the rows that stay.
        That indirection is the only definition that survives a multi-row move: with rows
        0 and 5 selected, "drop at 3" cannot mean "index 3 of a list that still contains
        them". ``dest = n_rows`` moves to the end. Out-of-range and duplicate indices are
        ignored; returns False when nothing moved.
        """
        n_rows = self.n_rows()
        if n_rows == 0:
            return False
        moving = sorted({i + n_rows if i < 0 else i for i in indices})
        moving = [i for i in moving if 0 <= i < n_rows]
        if not moving or len(moving) == n_rows:
            return False

        moving_set = set(moving)
        rest = [i for i in range(n_rows) if i not in moving_set]
        # Where dest lands among the survivors: every moved row before it shifts it left.
        target = max(0, min(int(dest), n_rows))
        insert_at = target - sum(1 for i in moving if i < target)
        insert_at = max(0, min(insert_at, len(rest)))

        order = np.array(rest[:insert_at] + moving + rest[insert_at:], dtype=np.intp)
        if np.array_equal(order, np.arange(n_rows)):
            return False
        for column in self.columns:
            column.values = column.values[order]
            column.invalidate()
        return True

    def filter_rows(self, mask: Any) -> int:
        """Keep only the rows where ``mask`` is True. Returns the number dropped.

        ``mask`` is a boolean array of ``n_rows`` (a nan-valued predicate result counts
        as False -- ``nan > 0`` is False, and a row whose value is missing is not a row
        that matched). This is the destructive half of the filter: the panel shows the
        mask as a view first, and only calls this for "apply to data".
        """
        n_rows = self.n_rows()
        if n_rows == 0:
            return 0
        keep = np.asarray(mask)
        if keep.dtype != np.bool_:
            keep = np.nan_to_num(np.asarray(keep, dtype=np.float64), nan=0.0) != 0.0
        keep = np.ravel(keep)
        if keep.shape[0] != n_rows:
            raise ValueError(f"mask has {keep.shape[0]} entries but the table has {n_rows} rows.")
        dropped = int(n_rows - int(np.count_nonzero(keep)))
        if dropped == 0:
            return 0
        for column in self.columns:
            column.values = column.values[keep]
            column.invalidate()
        return dropped

    # -- expressions over the columns ------------------------------------------

    def bindable_columns(self) -> Dict[str, np.ndarray]:
        """The columns an expression can name, keyed by name.

        Only Python identifiers, and never a dunder: a column called ``"y (2)"`` or
        ``"1"`` has no name an expression could reference. Excluding them here (rather
        than letting the evaluator raise) is what lets the panel *tell the user* which
        columns are usable, and is why :meth:`transform_column` can still target a
        column whose own name is unspeakable.
        """
        usable: Dict[str, np.ndarray] = {}
        for column in self.columns:
            if column.name.isidentifier() and "__" not in column.name:
                usable[column.name] = column.values
        return usable

    def eval_expression(
        self, expr: str, *, variables: Optional[Mapping[str, Any]] = None
    ) -> np.ndarray:
        """Evaluate ``expr`` with the columns bound as names -> float64 of ``n_rows``.

        The one primitive behind "transform a column" (``y * 2``), "filter rows"
        (``(y > 0) & (x < 10)``) and a parametrized transform (``a * y + b``, ``a``/
        ``b`` from Data Editor's Transform-section sliders): ``expressions.evaluate``
        is the hardened, AST-allowlisted evaluator, and the columns -- plus, now,
        ``variables`` -- are simply its ``variables``.

        Not ``expressions.evaluate_1d``: that binds the name ``x`` to a domain it is
        handed, which would either shadow a real ``x`` column or invent one for a table
        that has none. Here every name comes from the table (and ``variables``), and
        nothing else exists.

        Args:
            expr: The expression text.
            variables: Extra name -> value bindings on top of the table's own columns
                (e.g. a slider's current parameter values). A column always wins on a
                name collision -- real data over a stale/same-named parameter -- though
                in practice a caller should exclude existing column names when deciding
                what counts as a free parameter in the first place (see
                :func:`~glplot.gui.expressions.free_variables`'s ``exclude``).

        A scalar result broadcasts to ``n_rows`` (``0`` is a legal transform); booleans
        become 0.0/1.0. Raises :class:`~glplot.gui.expressions.ExpressionError` -- and
        only that -- for anything else, including a length mismatch.
        """
        from .expressions import ExpressionError, evaluate

        n_rows = self.n_rows()
        bindings: Dict[str, Any] = dict(variables) if variables else {}
        bindings.update(self.bindable_columns())
        result = evaluate(expr, bindings)

        try:
            arr = np.asarray(result)
        except Exception as exc:  # pragma: no cover - numpy rejects the object
            raise ExpressionError(f"expression produced a value numpy cannot use: {exc}") from exc

        if np.iscomplexobj(arr):
            raise ExpressionError(
                "expression produced complex values; wrap it in abs(...) to use the magnitude"
            )
        if not (np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)):
            raise ExpressionError(
                f"expression produced a non-numeric result (dtype {arr.dtype}); expected numbers"
            )

        arr = arr.astype(np.float64, copy=False)
        if arr.ndim == 0:
            return np.full(n_rows, float(arr), dtype=np.float64)
        if arr.ndim != 1:
            raise ExpressionError(f"expression produced a {arr.ndim}-D result; expected 1-D")
        if arr.shape[0] == 1:
            return np.full(n_rows, float(arr[0]), dtype=np.float64)
        if arr.shape[0] != n_rows:
            raise ExpressionError(
                f"expression produced {arr.shape[0]} values but the table has {n_rows} rows"
            )
        return np.array(arr, dtype=np.float64, copy=True)

    def row_mask(self, expr: str) -> np.ndarray:
        """Evaluate ``expr`` as a row predicate -> bool array of ``n_rows``.

        Anything non-zero is a match, so ``y`` and ``y != 0`` mean the same thing, and a
        nan never matches (a row with no value did not pass the test).

        Re-raises the two mistakes Python's operator rules make unavoidable here with an
        actionable message instead of numpy's. Both are worth catching because the
        obvious spelling is the broken one: ``y > 0 & x < 10`` parses as
        ``y > (0 & x) < 10`` -- ``&`` binds tighter than ``>`` -- and ``and``/``or`` ask
        an array for a single truth value.
        """
        from .expressions import ExpressionError

        try:
            values = self.eval_expression(expr)
        except ExpressionError as exc:
            message = str(exc)
            if "bitwise_and" in message or "bitwise_or" in message or "bitwise_xor" in message:
                raise ExpressionError(
                    f"{message}\nParenthesise each comparison: '&' binds tighter than '>', "
                    "so write (y > 0) & (x < 10), not y > 0 & x < 10."
                ) from exc
            if "truth value" in message and "ambiguous" in message:
                raise ExpressionError(
                    f"{message}\nUse '&' and '|' instead of 'and'/'or': they compare the "
                    "whole column at once."
                ) from exc
            raise
        return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=1.0) != 0.0

    def transform_column(
        self,
        name: str,
        expr: str,
        *,
        target: Optional[str] = None,
        variables: Optional[Mapping[str, Any]] = None,
    ) -> Column:
        """Write ``expr`` over the columns into a column, and return it.

        ``target=None`` overwrites ``name`` in place; otherwise a column named ``target``
        is appended (or overwritten if it already exists, which is what makes re-running
        a transform idempotent instead of piling up ``z (2)``, ``z (3)``...).

        ``variables`` passes through to :meth:`eval_expression` unchanged -- a
        parametrized transform's current slider values, baked in at the moment this is
        called (this is the "Apply" of Data Editor's Transform-section sliders; it does
        not keep the column live-linked to them afterward).

        The result is computed **before** anything is written, so an expression that
        raises leaves the table exactly as it was -- there is no half-applied transform.
        """
        source = self._find(name)
        if source is None:
            raise ValueError(f"Column {name!r} not found in dataset {self.name!r}.")

        values = self.eval_expression(expr, variables=variables)

        if target is None or target == name:
            source.values = values
            source.invalidate()
            return source

        existing = self._find(target)
        if existing is not None:
            existing.values = values
            existing.invalidate()
            return existing
        return self.add_column(target, values)

    # -- cells -----------------------------------------------------------------

    def _check_col(self, col: int) -> int:
        n_cols = self.n_cols()
        if col < 0:
            col += n_cols
        if not 0 <= col < n_cols:
            raise IndexError(f"Column {col} out of range for {n_cols} columns.")
        return col

    def set_cell(self, row: int, col: int, value: float) -> None:
        """Write one cell. ``value`` may be nan."""
        column = self.columns[self._check_col(col)]
        column.values[self._check_row(row)] = float(value)
        column.invalidate()

    def get_cell(self, row: int, col: int) -> float:
        """Read one cell."""
        column = self.columns[self._check_col(col)]
        return float(column.values[self._check_row(row)])

    # -- conversion ------------------------------------------------------------

    def to_xy(self, x_col: str, y_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return the ``(x, y)`` arrays of two columns.

        These are the live column arrays, not copies. Callers that hand them to the engine
        go through ``layerops.add_xy_layer``, which coerces to contiguous float32 and
        therefore copies anyway.
        """
        x = self.get(x_col)
        if x is None:
            raise ValueError(f"Column {x_col!r} not found in dataset {self.name!r}.")
        y = self.get(y_col)
        if y is None:
            raise ValueError(f"Column {y_col!r} not found in dataset {self.name!r}.")
        return x, y

    def to_array(self) -> np.ndarray:
        """Return the whole table as a ``(n_rows, n_cols)`` float64 array (a copy)."""
        if not self.columns:
            return np.zeros((0, 0), dtype=np.float64)
        return np.column_stack([column.values for column in self.columns])

    def copy(self, new_name: Optional[str] = None) -> "DataSet":
        """Deep-copy the dataset.

        The copy never inherits the layer link: ``layer_id`` and ``binding`` are cleared,
        because two datasets writing back to the same layer would fight each other. A copy
        of a ``"layer"``-sourced dataset therefore becomes ``"derived"``; any other source
        is preserved.
        """
        columns = [Column(column.name, column.values.copy()) for column in self.columns]
        clone = DataSet(new_name if new_name is not None else f"{self.name} copy", columns)
        clone.source = "derived" if self.source == "layer" else self.source
        return clone

    # -- layer link ------------------------------------------------------------

    @classmethod
    def from_layer(cls, layer: Any, name: Optional[str] = None) -> Optional["DataSet"]:
        """Build an unregistered DataSet backed by ``layer``.

        Columns are **float64 copies** of the layer's native (usually float32) geometry --
        not views; see the module docstring for why views are impossible here. Cell edits
        reach the layer through :meth:`write_back`, which the data editor calls from inside
        a queued command.

        Mapping: ``scatter``/``polyline`` -> ``pts`` -> ``x, y``; ``line_family`` -> ``ab``
        -> ``a, b``; 3D layers -> ``vertices`` -> ``x, y, z``. Returns None for layer types
        with no editable tabular form (``text``, ``patch``).

        Sets ``layer_id`` and ``source = "layer"`` on the result.
        """
        resolved = _resolve_layer_binding(layer)
        if resolved is None:
            return None
        attr, names, array = resolved

        columns = [Column(names[i], array[:, i].astype(np.float64)) for i in range(array.shape[1])]
        label = getattr(layer, "label", "") or ""
        dataset = cls(name or label or f"Layer {getattr(layer, 'layer_id', '?')}", columns)
        dataset.layer_id = getattr(layer, "layer_id", None)
        dataset.source = "layer"
        dataset.binding = LayerBinding(attr=attr, columns=list(names), dtype=array.dtype)
        return dataset

    def unbind(self) -> None:
        """Drop the layer link, leaving the data intact as a standalone table."""
        self.binding = None
        self.layer_id = None
        if self.source == "layer":
            self.source = "derived"

    def is_bound(self) -> bool:
        """True if this dataset can still :meth:`write_back` to its layer."""
        return self.binding is not None

    def write_back(self, layer: Any) -> bool:
        """Push the bound columns back onto ``layer``'s geometry array.

        Reassembles the bound columns into a C-contiguous array in the layer's native
        dtype (float32 for engine geometry), assigns it to the bound attribute, and sets
        ``layer.dirty.gpu_dirty = True`` per CONTRACT 1.4 -- the array is swapped in place,
        so the renderer must be told to re-upload.

        Row-count changes are handled: the array is rebuilt from scratch, so inserting or
        deleting rows in the editor works and the layer simply gets a different-length
        array.

        This does NOT set the scene dirty flags. It is the caller's job to run this inside
        a queued command, whose drain epilogue supplies them (CONTRACT 1.3). Calling it
        from a draw callback would mutate live GPU-backed data mid-frame.

        Returns False -- without touching ``layer`` -- when the dataset is not bound, when
        the binding is ``derived`` (the layer's geometry is computed from the columns, so
        only the code that computed it may rewrite it), when ``layer`` is not the bound
        layer, or when a bound column has been removed.
        """
        binding = self.binding
        if binding is None or binding.derived:
            return False
        if self.layer_id is not None and getattr(layer, "layer_id", None) != self.layer_id:
            return False

        arrays: List[np.ndarray] = []
        for name in binding.columns:
            values = self.get(name)
            if values is None:
                return False
            arrays.append(values)
        if not arrays:
            return False

        current = getattr(layer, binding.attr, None)
        dtype = binding.dtype
        if current is not None:
            current_dtype = np.asarray(current).dtype
            if current_dtype.kind == "f":
                dtype = current_dtype

        setattr(layer, binding.attr, np.ascontiguousarray(np.column_stack(arrays), dtype=dtype))
        dirty = getattr(layer, "dirty", None)
        if dirty is not None:
            dirty.gpu_dirty = True
        return True

    def sync_from_layer(self, layer: Any) -> bool:
        """Re-read the bound columns from ``layer``, discarding local edits.

        The inverse of :meth:`write_back`, for when code outside the editor has replaced a
        layer's arrays. Returns False if not bound, if the binding is ``derived`` (the
        geometry is not the columns and reading it back would import the scaffolding into
        the table), if ``layer`` is not the bound layer, or if the geometry no longer has
        the bound shape.
        """
        binding = self.binding
        if binding is None or binding.derived:
            return False
        if self.layer_id is not None and getattr(layer, "layer_id", None) != self.layer_id:
            return False

        array = getattr(layer, binding.attr, None)
        if array is None:
            return False
        array = np.asarray(array)
        if array.ndim != 2 or array.shape[1] != len(binding.columns):
            return False

        for i, name in enumerate(binding.columns):
            column = self._find(name)
            if column is None:
                return False
            column.values = array[:, i].astype(np.float64)
            column.invalidate()
        return True


class DataStore:
    """The workspace's collection of datasets, keyed by unique name."""

    def __init__(self) -> None:
        self.datasets: List[DataSet] = []

    def __len__(self) -> int:
        return len(self.datasets)

    def __iter__(self):
        return iter(self.datasets)

    def names(self) -> List[str]:
        """The dataset names, in order."""
        return [dataset.name for dataset in self.datasets]

    def unique_name(self, base: str) -> str:
        """Return ``base`` if free, else ``base (2)``, ``base (3)``, ..."""
        return _next_available(base, self.names())

    def add(self, ds: DataSet) -> DataSet:
        """Register ``ds``, renaming it in place if its name collides. Returns ``ds``.

        Re-adding an already-registered dataset is a no-op (it is not renamed).
        """
        if any(existing is ds for existing in self.datasets):
            return ds
        ds.name = self.unique_name(ds.name)
        self.datasets.append(ds)
        return ds

    def remove(self, ds: DataSet) -> bool:
        """Unregister ``ds``. Returns False if it was not registered."""
        for i, existing in enumerate(self.datasets):
            if existing is ds:
                del self.datasets[i]
                return True
        return False

    def get(self, name: str) -> Optional[DataSet]:
        """Return the dataset named ``name``, or None."""
        for dataset in self.datasets:
            if dataset.name == name:
                return dataset
        return None

    def get_by_layer_id(self, layer_id: int) -> Optional[DataSet]:
        """Return the dataset bound to ``layer_id``, or None."""
        for dataset in self.datasets:
            if dataset.layer_id == layer_id:
                return dataset
        return None

    def from_layer(self, layer: Any) -> Optional[DataSet]:
        """Build a DataSet backed by ``layer`` **and register it** in this store.

        Thin wrapper over :meth:`DataSet.from_layer` -- see there for the copy/write-back
        semantics and the layer-type mapping. Returns None for layer types with no
        editable tabular form (``text``, ``patch``), in which case nothing is registered.

        If a dataset for this layer already exists it is returned as-is rather than
        duplicated, so repeated scene syncs are idempotent.
        """
        layer_id = getattr(layer, "layer_id", None)
        if layer_id is not None:
            existing = self.get_by_layer_id(layer_id)
            if existing is not None:
                return existing
        dataset = DataSet.from_layer(layer)
        if dataset is None:
            return None
        return self.add(dataset)
