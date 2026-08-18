"""The spreadsheet data editor — "editar agregar datos, copiar y pegar datos".

The primary panel of the workstation, and the standalone app's entry point: paste a block
of numbers from Excel and get a dataset (and a plot) in one click.

Three design decisions worth knowing before editing this file
------------------------------------------------------------

**1. Virtualization is hand-rolled, because pyimgui 2.0 has no list clipper.**
Verified: ``[n for n in dir(imgui.core) if "lipper" in n] == []`` — there is no
``ListClipper`` binding at all. So :meth:`DataEditorPanel._draw_grid` computes the visible
row range from ``get_scroll_y()`` / ``get_window_height()`` (both of which, verified,
report the *table's inner scroll window* when queried inside a ``TABLE_SCROLL_Y`` table)
and emits only those rows, bracketed by two spacer rows of ``dummy(1, n * row_h)``. The
spacers are what give the scrollbar its full range, so layout is driven by them rather
than by the row estimate: an off-by-a-few estimate costs a few wasted rows, never a
visual shift. That is why :data:`_OVERSCAN` is cheap insurance rather than a fudge factor.
At 1e6 rows this emits ~30 rows per frame instead of 1e6.

Row height must be *uniform* for the arithmetic to hold, including the row that is being
edited (an ``input_text`` is taller than a ``selectable``). ``row_h = get_frame_height() +
cell_padding.y * 2`` passed as ``table_next_row``'s ``min_row_height`` was measured to
produce exactly-uniform 23px rows with a mix of both widgets.

**2. Undo granularity is chosen per operation, not uniformly.**
A full-table snapshot of a 1e6-row dataset is 16 MB. Taking one per committed keystroke
would be indefensible, so:

* single cell edit -> :meth:`_commit_edit`, an O(1) command holding two floats;
* range edit that fits (paste-in-place, cut, delete) -> :meth:`_apply_block`, which
  snapshots only the affected sub-block;
* structural change (paste that grows, insert/delete row, add/remove column, sort) ->
  :meth:`_apply_structural`, the only path that snapshots the whole table.

Every ``do``/``undo`` closure resolves columns **by index at run time** rather than
capturing :class:`~glplot.gui.datasets.Column` objects. This is load-bearing:
:meth:`_restore` rebuilds ``ds.columns`` with fresh objects, so a captured reference from
a command lower in the stack would be silently orphaned after a structural undo/redo.

**3. Mutation goes through the queue, per CONTRACT §1.1.**
``Panel.push_command`` submits ``UndoStack.push(cmd)`` to the queue, so ``cmd.do()`` runs
at the drain on the GL thread — never in the draw callback. Consequently a command's
``do`` is the *forward mutation* and its ``undo`` is a restore, and ``do`` must be
re-runnable from the undone state (it is: every mutation here is deterministic given the
data it reads). :meth:`_sync_layer_now` is called from inside those closures and is
therefore drain-only.

Known limitation (API, not choice)
----------------------------------
The spec asks for "typing enters edit mode". pyimgui 2.0 exposes only *write* access to
the character queue (``io.add_input_character`` / ``clear_input_characters``); there is no
readable ``input_queue_characters``, and no ``KEY_F2``/digit key constants. Edit mode is
therefore entered with **double-click or Enter**, which covers the same workflow.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from .. import clipboard, icons, layerops, layerops3d, theme, widgets
from ..datasets import Column, DataSet
from ..expressions import ExpressionError
from ..history import Command, is_snapshot_safe
from .base import Panel

try:
    from imgui_bundle import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):
    IMGUI_AVAILABLE = False
    imgui = None

logger = logging.getLogger(__name__)

__all__ = ["DataEditorPanel"]

#: ``ImGuiFocusedFlags_RootAndChildWindows``. pyimgui 2.0 exports no ``FOCUSED_*``
#: constants at all (``[n for n in dir(imgui) if n.startswith("FOCUSED")] == []``), but
#: ``is_window_focused`` accepts the raw flags int. The value is confirmed by the
#: ``HOVERED_*`` mirrors, which pyimgui *does* export and which Dear ImGui defines with
#: the same bit layout: ``HOVERED_CHILD_WINDOWS == 1``, ``HOVERED_ROOT_WINDOW == 2``,
#: ``HOVERED_ROOT_AND_CHILD_WINDOWS == 3``. We need child windows included because a
#: ``TABLE_SCROLL_Y`` table lives in an inner child window: clicking a cell focuses that
#: child, and a bare ``is_window_focused()`` on the panel then returns False — verified.
_FOCUSED_ROOT_AND_CHILD_WINDOWS = 3

#: Extra rows rendered above/below the viewport, absorbing row-range estimation error.
_OVERSCAN = 4

#: Grid columns are fixed-width so the horizontal scroll range is stable while scrolling.
_CELL_WIDTH = 92.0

#: Printf format for a cell's on-screen text: compact, but never lies by rounding to "0".
_DISPLAY_FMT = "%.6g"

#: Printf format when a cell opens for editing — full round-trip precision.
_EDIT_FMT = "%.10g"

#: Drag-and-drop payload types. imgui matches source and target on this string, so two
#: distinct types are what stop a dragged column from being dropped onto a row.
_DND_COLUMN = "GLPLOT_COLUMN"
_DND_ROW = "GLPLOT_ROW"

_NEW_DATASET_ROWS = 16


def _can_replot_into(layer: Any) -> bool:
    """True when re-plotting ``layer`` from a table is recoverable.

    The Data panel's primary button re-plots *into* the bound layer, and a kind change is
    a delete+recreate. That is safe only for a layer the picker could rebuild -- which is
    exactly what having a kind means. A layer with none (an ``imshow`` and every other
    field, which the engine draws as a scatter of cells; an ``axhline`` guide; a 3-D
    layer) is bindable all the same, because ``DataSet.from_layer`` reads its ``pts``
    happily -- so without this test the button offers to convert the picture into a line
    and nothing on the menu puts it back. ``_update_plot_layer`` already reaches the same
    conclusion one step too late: it computes ``undoable = ... and old_kind is not None``
    *after* the layer is gone.

    Deliberately the same question :func:`panels.scene._can_change_kind` asks, so the two
    panels cannot disagree about whether a layer's type may be changed. The source check
    is the second half of it: a kind whose source columns are unknowable cannot be rebuilt
    from either, and undo would have nothing to restore.
    """
    return layerops.layer_kind(layer) is not None and layerops.layer_source_xy(layer) is not None


def _fmt_display(value: float) -> str:
    """Format a cell for display. nan renders as an empty cell, as in Excel."""
    if math.isnan(value):
        return ""
    return _DISPLAY_FMT % value


def _fmt_edit(value: float) -> str:
    """Format a cell for the edit box. nan opens as empty text."""
    if math.isnan(value):
        return ""
    return _EDIT_FMT % value


def _parse_cell(text: str) -> Optional[float]:
    """Parse edited cell text to a float.

    Empty text and the literal ``nan`` both mean "no value" -> ``nan``. Returns ``None``
    when the text is not a number at all, which the caller surfaces as an error rather
    than silently writing a zero.
    """
    stripped = text.strip()
    if not stripped:
        return float("nan")
    try:
        return float(stripped)
    except (TypeError, ValueError):
        return None


def _headers_are_generated(headers: List[str]) -> bool:
    """True when ``headers`` are :func:`clipboard.parse_table`'s invented ``col1..colN``.

    ``parse_table`` always returns headers, inventing them when the pasted text had no
    header row. It does not report which happened, so this reconstructs the invented
    pattern: on a match we keep the dataset's existing column names, on a mismatch the
    user pasted real names worth adopting.
    """
    return all(name == f"col{i + 1}" for i, name in enumerate(headers))


class DataEditorPanel(Panel):
    """Spreadsheet editor over a :class:`~glplot.gui.datasets.DataSet`.

    Range selection, Excel-grade clipboard (keyboard *and* icon buttons), undoable cell
    edits, virtualized rendering, and a live link that re-uploads a bound layer as you
    type.
    """

    title = "Data"
    icon = "table"
    default_open = True

    def __init__(self, ws: Any) -> None:
        super().__init__(ws)
        self._current: Optional[DataSet] = None

        # Selection: an inclusive rectangle spanned by anchor..cursor, in (row, col).
        self._anchor: Tuple[int, int] = (0, 0)
        self._cursor: Tuple[int, int] = (0, 0)
        self._selecting = False

        # Cell edit state.
        self._edit_cell: Optional[Tuple[int, int]] = None
        self._edit_text = ""
        self._edit_focus = False

        self._scroll_to_row: Optional[int] = None
        #: One grid mutation per frame. Two actions racing in a single frame would each
        #: snapshot the same pre-state and the second undo would revert both.
        self._acted = False

        self._error = ""
        self._status = ""

        self._copy_headers = False
        self._csv_path = os.path.join(os.getcwd(), "data.csv")
        self._plot_x = ""
        self._plot_y = ""
        self._plot_kind = "line"
        self._plot_opts: dict = layerops.default_kind_options("line")
        # Data-driven encodings for scatter: a column name maps that variable onto colour
        # (through _plot_cmap) or marker size. "" means "no mapping" (flat colour / uniform size).
        self._plot_c = ""
        self._plot_s = ""
        self._plot_cmap = "viridis"

        # -- 3D plotting -------------------------------------------------------
        # The picker is dimensionality-aware: a table with three columns can become a
        # surface as readily as two can become a line, and which set of plot types is
        # offered is the *only* thing that changes. Kept as separate state from the 2D
        # picker (rather than one kind field that switches registry) so flipping the
        # toggle back and forth does not lose either side's column choices.
        self._plot_ndim = 2
        #: True once the user has touched the 2D/3D toggle. Until then the picker follows
        #: the figure, so opening the Data panel on a 3D plot offers 3D types immediately.
        self._ndim_touched = False
        self._plot_z = ""
        self._plot_kind3d = "scatter3d"
        self._plot_opts3d: dict = layerops3d.default_kind3d_options("scatter3d")
        #: Vector columns for ``quiver3d`` — the one kind that needs six columns.
        self._plot_u = ""
        self._plot_v = ""
        self._plot_w = ""

        self._rename_text = ""
        self._new_column_name = "z"

        # Filter: a non-destructive *view*. ``_view_rows`` holds the data-row indices the
        # grid shows, in table order; None means "no filter, view == data". Selection and
        # every (row, col) in this panel are in **view** coordinates and go through
        # :meth:`_data_row`; only the mutators, which require an unfiltered table, may
        # treat the two as the same number.
        self._filter_text = ""
        self._view_rows: Optional[np.ndarray] = None
        self._filter_applied = ""
        #: n_rows when the filter was applied. Any change invalidates ``_view_rows``,
        #: whose indices would otherwise point into a table that no longer has them.
        self._filter_n_rows = 0

        self._transform_col = ""
        self._transform_expr = ""
        self._transform_to_new = False
        self._transform_target = "z"

    # -- dataset selection -----------------------------------------------------

    def _current_dataset(self) -> Optional[DataSet]:
        """Return the edited dataset, healing a stale reference after a removal."""
        datasets = self.store.datasets
        if not datasets:
            self._current = None
            return None
        if self._current is None or not any(ds is self._current for ds in datasets):
            self._select(datasets[0])
        return self._current

    def _select(self, ds: Optional[DataSet]) -> None:
        """Make ``ds`` current and reset per-dataset UI state."""
        self._current = ds
        self._anchor = (0, 0)
        self._cursor = (0, 0)
        self._edit_cell = None
        self._selecting = False
        self._error = ""
        # The grid keeps one scroll position across datasets (same table id), so rewind
        # it explicitly rather than landing mid-way down a dataset the user just opened.
        self._scroll_to_row = 0
        self._clear_filter()
        self._filter_text = ""
        if ds is not None:
            names = ds.column_names()
            self._plot_x = names[0] if names else ""
            self._plot_y = names[1] if len(names) > 1 else self._plot_x
            self._transform_col = names[0] if names else ""

    # -- the filter view -------------------------------------------------------

    def _clear_filter(self) -> None:
        """Drop the row filter, showing the whole table again. Never touches the data."""
        self._view_rows = None
        self._filter_applied = ""
        self._filter_n_rows = 0

    @property
    def _filtered(self) -> bool:
        """True while a row filter is hiding part of the table."""
        return self._view_rows is not None

    def _view_rows_count(self, ds: DataSet) -> int:
        """How many rows the grid shows: the whole table, or the filter's matches."""
        return ds.n_rows() if self._view_rows is None else int(len(self._view_rows))

    def _data_row(self, row: int) -> int:
        """Map a **view** row (what the grid and the selection speak) to a data row."""
        if self._view_rows is None:
            return row
        if 0 <= row < len(self._view_rows):
            return int(self._view_rows[row])
        return -1

    def _validate_filter(self, ds: DataSet) -> None:
        """Drop a filter whose indices can no longer be trusted.

        The mask is a snapshot of row *indices*, so anything that changes the row count
        underneath it (an undo from the global toolbar, a Math Lab op writing back)
        leaves it pointing at rows that moved or no longer exist. Rebuilding silently
        would be worse than dropping it: the user would be looking at a filter of one
        table while reading the label of another.
        """
        if self._view_rows is None:
            return
        if ds.n_rows() != self._filter_n_rows:
            self._clear_filter()
            self._status = "Filter cleared — the table's rows changed underneath it."

    def _apply_filter(self, ds: DataSet) -> None:
        """Evaluate the predicate and show only the matching rows. Non-destructive."""
        expr = self._filter_text.strip()
        if not expr:
            self._clear_filter()
            return
        try:
            mask = ds.row_mask(expr)
        except ExpressionError as exc:
            self._error = f"Filter: {exc}"
            return
        self._error = ""
        self._view_rows = np.nonzero(mask)[0].astype(np.intp)
        self._filter_applied = expr
        self._filter_n_rows = ds.n_rows()
        self._anchor = self._cursor = (0, self._cursor[1])
        self._scroll_to_row = 0
        self._status = f"Filter shows {len(self._view_rows)} of {ds.n_rows()} rows"

    def _clamp_selection(self, ds: DataSet) -> None:
        """Keep anchor/cursor inside the dataset after rows or columns disappear."""
        max_row = max(0, self._view_rows_count(ds) - 1)
        max_col = max(0, ds.n_cols() - 1)

        def clamp(cell: Tuple[int, int]) -> Tuple[int, int]:
            return (min(max(cell[0], 0), max_row), min(max(cell[1], 0), max_col))

        self._anchor = clamp(self._anchor)
        self._cursor = clamp(self._cursor)
        if self._edit_cell is not None:
            row, col = self._edit_cell
            if not (0 <= row < ds.n_rows() and 0 <= col < ds.n_cols()):
                self._edit_cell = None

    def _range(self, ds: DataSet) -> Tuple[int, int, int, int]:
        """The selected rectangle as inclusive ``(r0, r1, c0, c1)``, in **view** rows."""
        r0, r1 = sorted((self._anchor[0], self._cursor[0]))
        c0, c1 = sorted((self._anchor[1], self._cursor[1]))
        r1 = min(r1, max(0, self._view_rows_count(ds) - 1))
        c1 = min(c1, max(0, ds.n_cols() - 1))
        return r0, r1, c0, c1

    def _selected_data_rows(self, ds: DataSet) -> List[int]:
        """The selection's rows as *data* indices — the same list, filter or no filter."""
        r0, r1, _c0, _c1 = self._range(ds)
        rows = [self._data_row(row) for row in range(r0, r1 + 1)]
        return [row for row in rows if row >= 0]

    def _require_unfiltered(self, action: str) -> bool:
        """Refuse a structural edit while a filter is up, and say so. True when allowed.

        Not squeamishness: paste, insert, sort and reorder all speak in contiguous row
        indices, and under a filter the visible rows are neither contiguous nor all of
        them. The choices were "silently act on rows the user cannot see", "quietly act
        on a different rectangle than the one highlighted", or this. Cell edits and Copy
        stay live, because both address one visible cell at a time.
        """
        if not self._filtered:
            return True
        self._error = (
            f"{action} is not available while a filter is active — the rows you can see "
            "are not the rows the table has. Clear the filter, or apply it to the data."
        )
        return False

    # -- undo plumbing ---------------------------------------------------------

    @staticmethod
    def _snapshot(ds: DataSet) -> Tuple[List[str], List[np.ndarray]]:
        """Copy the whole table (names + values). Only for structural changes."""
        return ([c.name for c in ds.columns], [c.values.copy() for c in ds.columns])

    @staticmethod
    def _restore(ds: DataSet, snap: Tuple[List[str], List[np.ndarray]]) -> None:
        """Rebuild ``ds.columns`` from a snapshot.

        The arrays are copied again on the way in: a snapshot is replayed on every
        undo/redo cycle, so handing over the stored array would let a later in-place cell
        write corrupt the history.
        """
        names, arrays = snap
        ds.columns = [Column(name, values.copy()) for name, values in zip(names, arrays)]

    def _sync_layer_now(self, ds: DataSet) -> None:
        """Push ``ds`` back onto its bound layer. **Inside a queued command only.**

        This is the "edición en vivo" link: it runs from the ``do``/``undo`` closures,
        which ``Panel.push_command`` executes at the queue drain (on the GL thread), so
        the plot follows the table as the user types. The drain epilogue supplies the
        §1.4 dirty flags; ``update_layer_xy`` / ``write_back`` set ``gpu_dirty``.

        Three paths, because a layer's geometry is not always its columns: a *derived*
        kind (bar, step, hist) is re-derived by the same code that built it, so the
        live link and the Plot button cannot draw different pictures from one table; a
        line/scatter has its ``pts`` swapped; anything else falls back to ``write_back``.
        """
        if ds.layer_id is None:
            return
        layer = layerops.find_layer(self.plot, ds.layer_id)
        if layer is None:
            # The layer was deleted from the scene panel; degrade to a plain table.
            ds.unbind()
            return
        binding = ds.binding
        if binding is not None and binding.derived and binding.kind:
            x = ds.get(binding.columns[0])
            y = ds.get(binding.columns[-1])
            if x is not None and y is not None:
                updated = layerops.update_layer_kind_data(
                    self.plot,
                    layer,
                    x,
                    y,
                    kind=binding.kind,
                    options=layerops.layer_kind_options(layer),
                )
                if updated:
                    return
        if binding is not None and binding.attr == "pts" and len(binding.columns) == 2:
            x = ds.get(binding.columns[0])
            y = ds.get(binding.columns[1])
            if x is not None and y is not None:
                layerops.update_layer_xy(self.plot, layer, x, y)
                return
        if ds.write_back(layer):
            layerops.mark_scene_dirty(self.plot)

    def _apply_structural(self, ds: DataSet, label: str, mutate: Callable[[], None]) -> bool:
        """Queue ``mutate`` as an undoable command, snapshotting the whole table.

        ``mutate`` runs at the drain and must be deterministic given the restored
        pre-state, so that redo reproduces it exactly.
        """
        if self._acted:
            return False
        self._acted = True

        safe = is_snapshot_safe(*[c.values for c in ds.columns])
        before = self._snapshot(ds) if safe else None

        def do() -> None:
            mutate()
            self._sync_layer_now(ds)

        if before is None:
            self.push_command(Command.not_undoable(label, do))
            self._status = f"{label} — dataset too large to undo"
            return True

        def undo() -> None:
            self._restore(ds, before)
            self._sync_layer_now(ds)

        self.push_command(Command(label=label, do=do, undo=undo))
        self._status = label
        return True

    def _apply_block(
        self, ds: DataSet, label: str, r0: int, r1: int, c0: int, c1: int, block: Any
    ) -> bool:
        """Write ``block`` into the inclusive rectangle, undoably.

        Snapshots only the rectangle, so zeroing or pasting over a selection inside a
        million-row table costs the size of the selection, not the size of the table.
        """
        if self._acted:
            return False
        if ds.n_rows() == 0 or ds.n_cols() == 0:
            return False
        self._acted = True

        n_rows = r1 - r0 + 1
        new = np.asarray(block, dtype=np.float64).reshape(n_rows, c1 - c0 + 1)
        old = np.column_stack([ds.columns[c].values[r0 : r1 + 1] for c in range(c0, c1 + 1)])

        def write(data: np.ndarray) -> None:
            # Resolve by index, never by captured Column: _restore rebuilds the objects.
            if r1 >= ds.n_rows() or c1 >= ds.n_cols():
                logger.debug("Stale block edit %r ignored: dataset shape changed.", label)
                return
            for j in range(c0, c1 + 1):
                column = ds.columns[j]
                column.values[r0 : r1 + 1] = data[:, j - c0]
                column.invalidate()
            self._sync_layer_now(ds)

        if not is_snapshot_safe(old):
            self.push_command(Command.not_undoable(label, lambda: write(new)))
            self._status = f"{label} — selection too large to undo"
            return True

        self.push_command(Command(label=label, do=lambda: write(new), undo=lambda: write(old)))
        self._status = label
        return True

    # -- cell editing ----------------------------------------------------------

    def _begin_edit(self, ds: DataSet, row: int, col: int) -> None:
        """Open the cell at view ``(row, col)`` for editing."""
        if not (0 <= row < self._view_rows_count(ds) and 0 <= col < ds.n_cols()):
            return
        self._edit_cell = (row, col)
        self._edit_text = _fmt_edit(ds.get_cell(self._data_row(row), col))
        self._edit_focus = True
        self._anchor = self._cursor = (row, col)

    def _cancel_edit(self) -> None:
        """Abandon the in-progress edit, leaving the cell untouched."""
        self._edit_cell = None
        self._edit_text = ""
        self._edit_focus = False

    def _commit_edit(self, ds: DataSet) -> None:
        """Commit the in-progress edit as an O(1) undoable command.

        Deliberately does *not* snapshot the table: one command holds the old and new
        float, so a keystroke costs bytes rather than the 16 MB a 1e6-row snapshot would.
        """
        if self._edit_cell is None:
            return
        view_row, col = self._edit_cell
        text = self._edit_text
        self._cancel_edit()

        # Resolve to a data row here and commit in data coordinates: the command outlives
        # the filter that was up when it was typed, and an undo must not depend on it.
        row = self._data_row(view_row)
        if not (0 <= row < ds.n_rows() and 0 <= col < ds.n_cols()):
            return
        value = _parse_cell(text)
        if value is None:
            self._error = f"{text!r} is not a number. Leave a cell empty for nan."
            return

        old = ds.get_cell(row, col)
        if old == value or (math.isnan(old) and math.isnan(value)):
            return
        if self._acted:
            return
        self._acted = True

        def write(target: float) -> None:
            if not (0 <= row < ds.n_rows() and 0 <= col < ds.n_cols()):
                return
            column = ds.columns[col]
            column.values[row] = target
            column.invalidate()
            self._sync_layer_now(ds)

        label = f"Edit {ds.name}[{row}, {ds.columns[col].name}]"
        self.push_command(Command(label=label, do=lambda: write(value), undo=lambda: write(old)))
        self._status = label

    def _move_cursor(self, ds: DataSet, d_row: int, d_col: int, *, extend: bool = False) -> None:
        """Move the cursor, clamped; ``extend`` keeps the anchor to grow the range."""
        row = min(max(self._cursor[0] + d_row, 0), max(0, self._view_rows_count(ds) - 1))
        col = min(max(self._cursor[1] + d_col, 0), max(0, ds.n_cols() - 1))
        self._cursor = (row, col)
        if not extend:
            self._anchor = self._cursor
        self._scroll_to_row = row

    # -- clipboard -------------------------------------------------------------

    def _copy_range(self, ds: DataSet) -> None:
        """Copy the selected rectangle to the clipboard as TSV.

        Copies what is *visible*: under a filter the selection's rows are gathered by
        index, so "filter, select all, copy" hands over exactly the matching rows —
        which is most of the reason to have a filter at all.
        """
        if ds.n_rows() == 0 or ds.n_cols() == 0:
            return
        _r0, _r1, c0, c1 = self._range(ds)
        rows = self._selected_data_rows(ds)
        if not rows:
            return
        index = np.asarray(rows, dtype=np.intp)
        headers = [ds.columns[c].name for c in range(c0, c1 + 1)]
        data = np.column_stack([ds.columns[c].values[index] for c in range(c0, c1 + 1)])
        text = clipboard.format_table(headers, data, include_header=self._copy_headers)
        imgui.set_clipboard_text(text)
        self._status = f"Copied {len(rows)} x {c1 - c0 + 1} cells"

    def _read_clipboard_table(self) -> Optional[Tuple[List[str], np.ndarray]]:
        """Parse the clipboard, surfacing parse failures in the panel's error box."""
        try:
            text = imgui.get_clipboard_text() or ""
        except Exception as exc:  # pragma: no cover - platform clipboard failure
            self._error = f"Could not read the clipboard: {exc}"
            return None
        try:
            return clipboard.parse_table(text)
        except ValueError as exc:
            self._error = str(exc)
            return None

    def _paste_range(self, ds: DataSet) -> None:
        """Paste at the selection's top-left, growing the dataset like Excel does."""
        if not self._require_unfiltered("Paste"):
            return
        parsed = self._read_clipboard_table()
        if parsed is None:
            return
        headers, data = parsed
        if data.size == 0:
            self._error = "Nothing to paste: the clipboard holds no data rows."
            return

        r0, _r1, c0, _c1 = self._range(ds)
        n_rows, n_cols = data.shape
        grows = (r0 + n_rows) > ds.n_rows() or (c0 + n_cols) > ds.n_cols()

        if not grows:
            self._apply_block(
                ds, f"Paste {n_rows} x {n_cols}", r0, r0 + n_rows - 1, c0, c0 + n_cols - 1, data
            )
        else:
            named = None if _headers_are_generated(headers) else headers

            def mutate() -> None:
                self._ensure_shape(ds, r0 + n_rows, c0 + n_cols, named, c0)
                for j in range(n_cols):
                    column = ds.columns[c0 + j]
                    column.values[r0 : r0 + n_rows] = data[:, j]
                    column.invalidate()

            self._apply_structural(ds, f"Paste {n_rows} x {n_cols} (grow)", mutate)

        self._anchor = (r0, c0)
        self._cursor = (r0 + n_rows - 1, c0 + n_cols - 1)

    @staticmethod
    def _ensure_shape(
        ds: DataSet,
        rows: int,
        cols: int,
        names: Optional[List[str]],
        col_offset: int,
    ) -> None:
        """Grow ``ds`` to at least ``rows`` x ``cols``, filling new cells with nan.

        New cells are nan rather than zero: a zero is a claim about the data, an empty
        cell is not, and :func:`clipboard.parse_table` already reads blanks as nan.
        """
        while ds.n_cols() < cols:
            index = ds.n_cols()
            hint = None
            if names is not None and col_offset <= index < col_offset + len(names):
                hint = names[index - col_offset]
            column = ds.add_column(hint or f"col{index + 1}")
            column.values[:] = np.nan
            column.invalidate()

        pad = rows - ds.n_rows()
        if pad > 0:
            for column in ds.columns:
                column.values = np.concatenate([column.values, np.full(pad, np.nan)])
                column.invalidate()

    def _cut_range(self, ds: DataSet) -> None:
        """Copy the selection, then blank it."""
        if not self._require_unfiltered("Cut"):
            return
        self._copy_range(ds)
        self._clear_range(ds, label="Cut")

    def _clear_range(self, ds: DataSet, *, label: str = "Clear") -> None:
        """Zero every cell in the selection."""
        if ds.n_rows() == 0 or ds.n_cols() == 0:
            return
        if not self._require_unfiltered("Clear"):
            return
        r0, r1, c0, c1 = self._range(ds)
        zeros = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.float64)
        self._apply_block(ds, f"{label} {r1 - r0 + 1} x {c1 - c0 + 1} cells", r0, r1, c0, c1, zeros)

    def _dataset_from_table(self, headers: List[str], data: np.ndarray, base: str) -> DataSet:
        """Build and register a dataset from a parsed table, and select it."""
        ds = DataSet(self.store.unique_name(base))
        for j, name in enumerate(headers):
            ds.add_column(name, data[:, j])
        ds.source = "clipboard"
        self.store.add(ds)
        self._select(ds)
        return ds

    def _paste_as_new_dataset(self) -> None:
        """Clipboard -> brand-new dataset, in one click. The standalone app's front door."""
        parsed = self._read_clipboard_table()
        if parsed is None:
            return
        headers, data = parsed
        ds = self._dataset_from_table(headers, data, "Pasted")
        self.push_command(
            Command(
                label=f"Paste as dataset {ds.name!r}",
                do=lambda: self.store.add(ds),
                undo=lambda: self.store.remove(ds),
            )
        )
        self._status = f"Pasted {ds.n_rows()} x {ds.n_cols()} into {ds.name!r}"

    # -- keyboard --------------------------------------------------------------

    @staticmethod
    def _pressed(key: "imgui.Key") -> bool:
        """True on the rising edge of an ``imgui.Key`` member."""
        return imgui.is_key_pressed(key)

    def _handle_keys(self, ds: DataSet, focused: bool) -> None:
        """Grid keyboard handling. Runs before the table so edits land the same frame."""
        if not focused:
            return
        io = imgui.get_io()
        # key_super keeps Cmd+C working on macOS, where Ctrl is not the modifier.
        ctrl = io.key_ctrl or io.key_super

        if self._edit_cell is not None:
            # Esc first and unconditionally: relying on the input's own deactivation to
            # signal a cancel cannot distinguish Esc from a click-away (which commits).
            if self._pressed(imgui.Key.escape):
                self._cancel_edit()
            elif self._pressed(imgui.Key.tab):
                self._commit_edit(ds)
                self._move_cursor(ds, 0, -1 if io.key_shift else 1)
                self._begin_edit(ds, *self._cursor)
            return

        # Another text field (rename, CSV path) owns the keyboard — do not hijack it.
        if io.want_text_input:
            return

        if ctrl and self._pressed(imgui.Key.c):
            self._copy_range(ds)
        elif ctrl and self._pressed(imgui.Key.v):
            self._paste_range(ds)
        elif ctrl and self._pressed(imgui.Key.x):
            self._cut_range(ds)
        elif ctrl and self._pressed(imgui.Key.a):
            self._anchor = (0, 0)
            self._cursor = (max(0, ds.n_rows() - 1), max(0, ds.n_cols() - 1))
        elif self._pressed(imgui.Key.delete) or self._pressed(imgui.Key.backspace):
            self._clear_range(ds)
        elif self._pressed(imgui.Key.enter) or self._pressed(imgui.Key.keypad_enter):
            self._begin_edit(ds, *self._cursor)
        elif self._pressed(imgui.Key.tab):
            self._move_cursor(ds, 0, -1 if io.key_shift else 1)
        elif self._pressed(imgui.Key.left_arrow):
            self._move_cursor(ds, 0, -1, extend=io.key_shift)
        elif self._pressed(imgui.Key.right_arrow):
            self._move_cursor(ds, 0, 1, extend=io.key_shift)
        elif self._pressed(imgui.Key.up_arrow):
            self._move_cursor(ds, -1, 0, extend=io.key_shift)
        elif self._pressed(imgui.Key.down_arrow):
            self._move_cursor(ds, 1, 0, extend=io.key_shift)
        elif self._pressed(imgui.Key.page_up):
            self._move_cursor(ds, -20, 0, extend=io.key_shift)
        elif self._pressed(imgui.Key.page_down):
            self._move_cursor(ds, 20, 0, extend=io.key_shift)
        elif self._pressed(imgui.Key.home):
            self._move_cursor(ds, -ds.n_rows(), 0, extend=io.key_shift)
        elif self._pressed(imgui.Key.end):
            self._move_cursor(ds, ds.n_rows(), 0, extend=io.key_shift)

    # -- drawing ---------------------------------------------------------------

    def draw(self) -> None:
        """Render the panel body."""
        if not IMGUI_AVAILABLE:
            return
        self._acted = False
        focused = imgui.is_window_focused(_FOCUSED_ROOT_AND_CHILD_WINDOWS)

        self._draw_dataset_bar()
        ds = self._current_dataset()
        if ds is None:
            self._draw_empty_state()
            return

        self._validate_filter(ds)
        self._clamp_selection(ds)
        self._handle_keys(ds, focused)
        self._draw_toolbar(ds)

        if self._error:
            widgets.error_box(self._error)

        self._draw_sections(ds)
        self._draw_status(ds)
        self._draw_grid(ds)

        if not imgui.is_mouse_down(0):
            self._selecting = False

    def _draw_dataset_bar(self) -> None:
        """Dataset combo + New / Duplicate / Delete."""
        datasets = self.store.datasets
        names = [ds.name for ds in datasets]
        current = self._current.name if self._current is not None else ""

        imgui.set_next_item_width(-190.0)
        if names:
            changed, picked = widgets.enum_combo("##dataset", current, names)
            if changed:
                for ds in datasets:
                    if ds.name == picked:
                        self._select(ds)
                        break
        else:
            imgui.text_disabled("No datasets")

        imgui.same_line()
        if icons.icon_button("##ds_new", "plus", tooltip="New dataset"):
            self._new_dataset()
        imgui.same_line()
        if icons.icon_button(
            "##ds_dup", "copy", tooltip="Duplicate dataset", enabled=self._current is not None
        ):
            self._duplicate_dataset()
        imgui.same_line()
        if icons.icon_button(
            "##ds_del", "trash", tooltip="Delete dataset", enabled=self._current is not None
        ):
            imgui.open_popup("Delete dataset?")

        message = (
            f"Delete the dataset {self._current.name!r}?\n"
            "The layer it plotted, if any, stays on the scene."
            if self._current is not None
            else ""
        )
        decision = widgets.confirm_popup("Delete dataset?", message)
        if decision:
            self._delete_dataset()

    def _draw_empty_state(self) -> None:
        """The first-run view: the one-click path from clipboard to data."""
        imgui.spacing()
        imgui.text_disabled("No dataset yet.")
        imgui.spacing()
        theme.push_accent()
        pasted = imgui.button("Paste data from clipboard", (240.0, 0.0))
        theme.pop_accent()
        if pasted:
            self._paste_as_new_dataset()
        imgui.same_line()
        if imgui.button("New empty dataset", (160.0, 0.0)):
            self._new_dataset()
        imgui.spacing()
        imgui.text_disabled("Copy a block of numbers in Excel, then click Paste.")
        if self._error:
            widgets.error_box(self._error)

    def _draw_toolbar(self, ds: DataSet) -> None:
        """Icon toolbar. Every keyboard shortcut is reachable here by mouse too.

        This is not redundancy for its own sake: Ctrl+C only fires while the grid has
        focus, so without buttons the clipboard would be unreachable after clicking any
        other widget — and the user explicitly asked for icons.
        """
        has_cells = ds.n_rows() > 0 and ds.n_cols() > 0

        if icons.icon_button(
            "##copy", "copy", tooltip="Copy selection (Ctrl+C)", enabled=has_cells
        ):
            self._copy_range(ds)
        imgui.same_line()
        if icons.icon_button("##cut", "cut", tooltip="Cut selection (Ctrl+X)", enabled=has_cells):
            self._cut_range(ds)
        imgui.same_line()
        if icons.icon_button("##paste", "paste", tooltip="Paste at selection (Ctrl+V)"):
            self._paste_range(ds)
        imgui.same_line()
        if icons.icon_button(
            "##clear", "close", tooltip="Clear selection to 0 (Delete)", enabled=has_cells
        ):
            self._clear_range(ds)

        widgets.toolbar_separator()
        if icons.icon_button("##paste_new", "table", tooltip="Paste as NEW dataset"):
            self._paste_as_new_dataset()

        widgets.toolbar_separator()
        if icons.icon_button("##row_add", "plus", tooltip="Insert row above selection"):
            self._insert_row(ds)
        imgui.same_line()
        if icons.icon_button(
            "##row_del", "minus", tooltip="Delete selected rows", enabled=has_cells
        ):
            self._delete_rows(ds)

        widgets.toolbar_separator()
        if icons.icon_button(
            "##filter",
            "filter",
            tooltip=(
                f"Filtered: {self._filter_applied} — click to show every row again"
                if self._filtered
                else "Filter rows by an expression (see 'Filter & transform')"
            ),
            active=self._filtered,
        ):
            if self._filtered:
                self._clear_filter()
                self._status = "Filter cleared"
            else:
                self._apply_filter(ds)

        widgets.toolbar_separator()
        spec = layerops.kind_spec(self._plot_kind)
        plot_tip = (
            f"Update the linked layer with this dataset ({spec.label})"
            if self._bound_layer(ds) is not None
            else f"Plot this dataset as {spec.label}"
        )
        if icons.icon_button("##plot", spec.icon, tooltip=plot_tip, enabled=ds.n_cols() >= 2):
            self._plot_dataset(ds)

        imgui.same_line()
        imgui.dummy((8.0, 1.0))
        imgui.same_line()
        _, self._copy_headers = imgui.checkbox("Copy headers", self._copy_headers)

    def _draw_status(self, ds: DataSet) -> None:
        """One line: shape, selection, live-link state."""
        r0, r1, c0, c1 = self._range(ds)
        imgui.text_disabled(
            f"{ds.n_rows()} rows x {ds.n_cols()} cols   |   "
            f"sel {r1 - r0 + 1}x{c1 - c0 + 1} @ ({r0}, {c0})   |   {ds.source}"
        )
        if self._filtered:
            imgui.same_line()
            r, g, b, a = theme.get_color("warn")
            imgui.text_colored(
                (r, g, b, a), f"|   filtered: {self._view_rows_count(ds)} rows shown"
            )
        if ds.layer_id is not None:
            imgui.same_line()
            imgui.text_disabled("|")
            imgui.same_line()
            if self._bound_layer(ds) is None:
                r, g, b, a = theme.get_color("warn")
                imgui.text_colored((r, g, b, a), f"layer {ds.layer_id} was deleted")
            else:
                r, g, b, a = theme.get_color("ok")
                imgui.text_colored((r, g, b, a), f"live -> layer {ds.layer_id}")
        if self._status:
            imgui.same_line()
            imgui.text_disabled(f"   {self._status}")

    def _draw_sections(self, ds: DataSet) -> None:
        """Collapsed-by-default detail sections, above the grid."""
        if widgets.section("Columns", default_open=False):
            self._draw_columns_section(ds)
        if widgets.section("Filter & transform", default_open=False):
            self._draw_filter_section(ds)
        if widgets.section("Plot & link", default_open=False):
            self._draw_plot_section(ds)
        if widgets.section("Import / Export", default_open=False):
            self._draw_io_section(ds)

    def _draw_columns_section(self, ds: DataSet) -> None:
        """Add / rename / remove / move / sort columns."""
        names = ds.column_names()
        cursor_col = self._cursor[1]
        current = names[cursor_col] if 0 <= cursor_col < len(names) else ""

        imgui.set_next_item_width(150.0)
        changed, self._new_column_name = imgui.input_text("##newcol", self._new_column_name)
        imgui.same_line()
        if icons.icon_button("##col_add", "plus", tooltip="Add column"):
            self._add_column(ds, self._new_column_name or "col")

        if current:
            imgui.set_next_item_width(150.0)
            renamed, self._rename_text = imgui.input_text(
                "##rename",
                self._rename_text or current,
                flags=imgui.InputTextFlags_.enter_returns_true,
            )
            imgui.same_line()
            commit = icons.icon_button("##col_ren", "check", tooltip=f"Rename {current!r}")
            if (renamed or commit) and self._rename_text and self._rename_text != current:
                self._rename_column(ds, current, self._rename_text)
                self._rename_text = ""
            imgui.same_line()
            if icons.icon_button("##col_del", "trash", tooltip=f"Remove column {current!r}"):
                self._remove_column(ds, current)

            imgui.text_disabled(f"Sort rows by {current!r}:")
            imgui.same_line()
            if icons.icon_button("##sort_asc", "chevron_up", tooltip="Sort ascending"):
                self._sort_by(ds, cursor_col, descending=False)
            imgui.same_line()
            if icons.icon_button("##sort_desc", "chevron_down", tooltip="Sort descending"):
                self._sort_by(ds, cursor_col, descending=True)

            imgui.text_disabled(f"Move {current!r}:")
            imgui.same_line()
            if icons.icon_button(
                "##col_left", "chevron_left", tooltip="Move left", enabled=cursor_col > 0
            ):
                self._move_column(ds, cursor_col, cursor_col - 1)
            imgui.same_line()
            if icons.icon_button(
                "##col_right",
                "chevron_right",
                tooltip="Move right",
                enabled=cursor_col < ds.n_cols() - 1,
            ):
                self._move_column(ds, cursor_col, cursor_col + 1)
            widgets.help_marker(
                "Columns and rows can also be dragged: drag a column header onto another "
                "header to reorder the columns, or drag the row number in the left gutter "
                "onto another row to move rows. Dragging a row that is part of the "
                "selection moves the whole selection."
            )

    def _draw_filter_section(self, ds: DataSet) -> None:
        """Filter rows by a predicate, and transform a column by an expression."""
        usable = ", ".join(sorted(ds.bindable_columns())) or (
            "(none — an expression can only name a column whose name is a plain "
            "identifier; rename them)"
        )
        imgui.text_disabled("Rows where:")
        imgui.same_line()
        imgui.set_next_item_width(-150.0)
        entered, self._filter_text = imgui.input_text(
            "##filter", self._filter_text, flags=imgui.InputTextFlags_.enter_returns_true
        )
        imgui.same_line()
        if imgui.button("Filter", (60.0, 0.0)) or entered:
            self._apply_filter(ds)
        imgui.same_line()
        if imgui.button("Show all", (70.0, 0.0)):
            self._clear_filter()
            self._status = "Filter cleared"
        widgets.help_marker(
            "A predicate over the columns, e.g. (y > 0) & (x < 10).\n\n"
            "Parenthesise each comparison: in Python '&' binds tighter than '>', so "
            "'y > 0 & x < 10' is an error, not a filter. Use '&' and '|', not "
            "'and'/'or' — those cannot compare a whole column.\n\n"
            "Filtering only HIDES rows; the table is untouched until you press 'Apply "
            "to data'. While a filter is up, cell edits and Copy work on the rows you "
            "can see; anything that moves rows around is disabled, because the rows on "
            "screen are not all the rows there are.\n\n"
            f"Columns you can name here: {usable}"
        )

        if self._filtered:
            r, g, b, a = theme.get_color("warn")
            imgui.text_colored(
                (r, g, b, a),
                f"FILTERED VIEW — showing {self._view_rows_count(ds)} of {ds.n_rows()} rows "
                f"where {self._filter_applied}   (the table still has all "
                f"{ds.n_rows()})",
            )
            if imgui.button("Apply to data", (110.0, 0.0)):
                self._apply_filter_to_data(ds)
            imgui.same_line()
            if imgui.button("To new dataset", (120.0, 0.0)):
                self._filter_to_new_dataset(ds)
            imgui.same_line()
            if ds.is_bound():
                imgui.text_disabled("(this dataset is plotted — 'To new dataset' is safer)")
            else:
                imgui.text_disabled("(deletes the hidden rows; undoable)")

        imgui.separator()

        names = ds.column_names()
        if not names:
            return
        if self._transform_col not in names:
            self._transform_col = names[0]
        imgui.set_next_item_width(110.0)
        _, self._transform_col = widgets.enum_combo("##tcol", self._transform_col, names)
        imgui.same_line()
        imgui.text("=")
        imgui.same_line()
        imgui.set_next_item_width(-90.0)
        applied, self._transform_expr = imgui.input_text(
            "##texpr", self._transform_expr, flags=imgui.InputTextFlags_.enter_returns_true
        )
        imgui.same_line()
        if (imgui.button("Apply", (80.0, 0.0)) or applied) and self._transform_expr.strip():
            target = self._transform_target.strip() if self._transform_to_new else None
            self._transform(ds, self._transform_col, self._transform_expr.strip(), target or None)

        _, self._transform_to_new = imgui.checkbox("Into a new column", self._transform_to_new)
        if self._transform_to_new:
            imgui.same_line()
            imgui.set_next_item_width(110.0)
            _, self._transform_target = imgui.input_text("##tnew", self._transform_target)
        widgets.help_marker(
            "An expression over the columns, evaluated for every row: y*2, y - mean(y), "
            "sqrt(x**2 + y**2). Undoable.\n\n"
            "Unchecked, it overwrites the chosen column in place. Checked, it writes a "
            "new column (re-running with the same name overwrites it rather than piling "
            "up copies).\n\n"
            "This is a per-column edit of the table. For operations that produce a new "
            "curve from (x, y) — integrate, smooth, fit, FFT — use Math Lab, which "
            "previews before it commits."
        )

    def _draw_plot_section(self, ds: DataSet) -> None:
        """Choose x/y/kind and push the dataset onto the scene.

        The primary button *updates* the bound layer rather than adding another one; the
        secondary button is the explicit way to get a second, unlinked copy. Its label
        says which of the two will happen before the click, because after the click the
        only difference is a layer count.
        """
        names = ds.column_names()
        if len(names) < 2:
            imgui.text_disabled("Need at least two columns to plot.")
            return

        self._draw_dimension_toggle(names)
        if self._plot_ndim == 3:
            self._draw_plot_section_3d(ds, names)
            return

        if self._plot_x not in names:
            self._plot_x = names[0]
        if self._plot_y not in names:
            self._plot_y = names[1]

        _, self._plot_x = widgets.enum_combo("X column", self._plot_x, names)
        _, self._plot_y = widgets.enum_combo("Y column", self._plot_y, names)

        labels = [layerops.kind_spec(k).label for k in layerops.KIND_KEYS]
        current = layerops.kind_spec(self._plot_kind).label
        changed, picked = widgets.enum_combo("Plot type", current, labels)
        if changed:
            self._set_plot_kind(layerops.KIND_KEYS[labels.index(picked)])
        widgets.help_marker(self._kind_help())
        self._draw_kind_options()
        self._draw_encoding_combos(names)

        bound = self._bound_layer(ds)
        stale = bound is None and ds.layer_id is not None
        # A bound layer with no kind cannot be re-plotted *into*: the re-plot is a
        # delete+recreate, nothing on the picker rebuilds it (that is what having no kind
        # means), and `_update_plot_layer` already knows the result is not undoable. So the
        # primary button here is a one-way door out of an imshow, a guide or a 3-D layer,
        # and the only notice was a status line *after* the layer was gone -- naming the
        # layer_type ("scatter is not undoable") rather than the picture it threw away.
        # The Scene panel greys its own type menu out on exactly this test; this is the
        # same refusal, and it leaves "Plot as new layer" as the way to see the columns.
        strands_layer = bound is not None and not _can_replot_into(bound)
        if bound is None:
            action = "Plot"
        elif layerops.layer_matches_kind(bound, self._plot_kind):
            action = "Update plot"
        else:
            action = f"Replot as {self._plot_kind}"

        theme.push_accent()
        if strands_layer:
            imgui.push_style_var(imgui.StyleVar_.alpha, imgui.get_style().alpha * 0.5)
        if imgui.button(action, (130.0, 0.0)) and not strands_layer:
            self._plot_dataset(ds)
        if strands_layer:
            imgui.pop_style_var(1)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    f"{bound.label!r} is a {bound.layer_type} layer that no plot type can "
                    "rebuild, so replotting into it could not be undone.\n\n"
                    "Use 'Plot as new layer' to see these columns without replacing it, "
                    "or 'Unlink' first if you meant this table to take the layer over."
                )
        theme.pop_accent()
        widgets.help_marker(
            "'Plot' renders this dataset into the layer it is linked to, replacing that "
            "layer's data instead of stacking a second copy on top of it. Changing Kind "
            "and pressing it converts the layer in place, keeping its colour, style and "
            "position in the Scene list.\n\n"
            "'Plot as new layer' is the way to get an additional, independent layer: it "
            "is NOT linked, so it will not follow later edits to this table."
        )

        if bound is not None:
            imgui.same_line()
            if imgui.button("Plot as new layer", (130.0, 0.0)):
                self._plot_dataset(ds, as_new=True)

        if ds.layer_id is not None:
            imgui.same_line()
            if imgui.button("Unlink", (90.0, 0.0)):
                ds.unbind()
                self._status = "Unlinked from layer"

        if stale:
            r, g, b, a = theme.get_color("warn")
            imgui.text_colored(
                (r, g, b, a), "Linked layer no longer exists — Plot will create a fresh one."
            )
        elif bound is not None:
            kind_name = layerops.layer_kind(bound) or bound.layer_type
            imgui.text_disabled(
                f"Linked to {bound.label!r} ({kind_name}) — cell edits update it live."
            )
        else:
            imgui.text_disabled("(plotting links the dataset for live editing)")

    # -- 3D plotting ---------------------------------------------------------------

    def _draw_dimension_toggle(self, names: List[str]) -> None:
        """The 2D / 3D switch above the column pickers.

        Follows the figure until the user touches it: opening this panel on a 3D plot
        should already be offering 3D types, and opening it on a 2D one should not be
        showing a Z picker for a table that has no third column to put in it.
        """
        if not self._ndim_touched:
            wanted = 3 if self.plot.is_3d_scene() else 2
            if wanted != self._plot_ndim:
                self._plot_ndim = wanted

        imgui.text("Plot as")
        for index, (label, value) in enumerate((("2D", 2), ("3D", 3))):
            imgui.same_line()
            if imgui.radio_button(f"{label}##plotndim", self._plot_ndim == value):
                self._plot_ndim = value
                self._ndim_touched = True
        imgui.same_line()
        widgets.help_marker(
            "2D plots two columns; 3D plots three. The plot types offered change with it "
            "— a surface, a 3D scatter and a vector field are only reachable from three "
            "columns, and a histogram or a line family only from two.\n\n"
            "Plotting in 3D switches the whole figure to a 3D view."
        )
        if self._plot_ndim == 3 and len(names) < 3:
            r, g, b, a = theme.get_color("warn")
            imgui.text_colored(
                (r, g, b, a),
                "This table has two columns. Add a third (Columns section) to plot in 3D.",
            )

    def _draw_plot_section_3d(self, ds: DataSet, names: List[str]) -> None:
        """The 3D half of the plot picker: x/y/z, a 3D plot type, and its encodings."""
        if len(names) < 3:
            return
        if self._plot_x not in names:
            self._plot_x = names[0]
        if self._plot_y not in names:
            self._plot_y = names[1]
        if self._plot_z not in names:
            self._plot_z = names[2]

        _, self._plot_x = widgets.enum_combo("X column", self._plot_x, names)
        _, self._plot_y = widgets.enum_combo("Y column", self._plot_y, names)
        _, self._plot_z = widgets.enum_combo("Z column", self._plot_z, names)

        keys = list(layerops3d.KIND3D_KEYS)
        labels = [layerops3d.kind3d_spec(k).label for k in keys]
        current = layerops3d.kind3d_spec(self._plot_kind3d).label
        changed, picked = widgets.enum_combo("Plot type", current, labels)
        if changed:
            self._set_plot_kind3d(keys[labels.index(picked)])
        widgets.help_marker(self._kind3d_help())

        spec = layerops3d.kind3d_spec(self._plot_kind3d)
        if spec.needs_grid:
            imgui.text_disabled("Needs a rectangular (x, y) grid — every combination present once.")

        # ``quiver3d`` takes three more columns. Drawn before the generic options editor
        # so the six geometry pickers stay together.
        if self._plot_kind3d == "quiver3d":
            self._draw_vector_combos(names)

        # The generic editor is driven by which keys the options dict has, so it picks up
        # baseline/stride/scale/head without knowing any kind's name. The vector columns
        # are arrays, not scalars, so they are hidden from it.
        widgets.kind_options_editor(
            {k: v for k, v in self._plot_opts3d.items() if k not in ("u", "v", "w")}
        )
        self._draw_encoding_combos_3d(names)

        theme.push_accent()
        if imgui.button("Plot in 3D", (130.0, 0.0)):
            self._plot_dataset_3d(ds)
        theme.pop_accent()
        imgui.same_line()
        widgets.help_marker(
            "Adds a 3D layer and switches the figure to a 3D view. Unlike the 2D "
            "path this always adds a layer rather than replotting into a bound one: a "
            "dataset binds one layer, and a 3D layer built from three columns is not the "
            "same object as the 2D layer the same table may already own."
        )
        imgui.text_disabled(f"{ds.n_rows()} rows -> {self._plot_kind3d}")

    def _set_plot_kind3d(self, kind: str) -> None:
        """Switch the 3D kind, resetting its parameters to that kind's defaults."""
        self._plot_kind3d = kind
        self._plot_opts3d = layerops3d.default_kind3d_options(kind)

    def _kind3d_help(self) -> str:
        """Tooltip for the 3D plot-type picker."""
        lines = ["What three columns can be plotted as:", ""]
        lines += [
            f"  {layerops3d.kind3d_spec(k).label} — {layerops3d.kind3d_spec(k).description}"
            for k in layerops3d.KIND3D_KEYS
        ]
        return "\n".join(lines)

    def _draw_vector_combos(self, names: List[str]) -> None:
        """The U/V/W pickers, for the one 3D kind that needs six columns."""
        none = "(none)"
        options = [none] + list(names)
        for label, attr in (("U (dx)", "_plot_u"), ("V (dy)", "_plot_v"), ("W (dz)", "_plot_w")):
            current = getattr(self, attr) or none
            if current not in options:
                current = none
            _, picked = widgets.enum_combo(label, current, options)
            setattr(self, attr, "" if picked == none else picked)

    def _draw_encoding_combos_3d(self, names: List[str]) -> None:
        """Colour-by / size-by for 3D. Size applies only to the point kinds."""
        spec = layerops3d.kind3d_spec(self._plot_kind3d)
        none = "(none)"
        options = [none] + list(names)

        cur_c = self._plot_c if self._plot_c in names else none
        _, picked_c = widgets.enum_combo(
            "Color by",
            cur_c,
            options,
            help="A column mapped through the colormap. With none chosen the layer is "
            "coloured by z, which is what makes a surface read as a height field.",
        )
        self._plot_c = "" if picked_c == none else picked_c
        cur_cmap = self._plot_cmap if self._plot_cmap in self._ENCODING_CMAPS else "viridis"
        _, self._plot_cmap = widgets.enum_combo("Colormap", cur_cmap, list(self._ENCODING_CMAPS))

        if spec.primitive == "points":
            cur_s = self._plot_s if self._plot_s in names else none
            _, picked_s = widgets.enum_combo("Size by", cur_s, options)
            self._plot_s = "" if picked_s == none else picked_s

    def _current_options3d(self) -> dict:
        """The selected 3D kind's options, including the vector columns for a quiver."""
        resolved = layerops3d.default_kind3d_options(self._plot_kind3d)
        resolved.update({k: v for k, v in self._plot_opts3d.items() if k in resolved})
        return resolved

    def _plot_dataset_3d(self, ds: DataSet) -> None:
        """Queue a 3D layer built from the x/y/z pickers, switching the figure to 3D."""
        names = ds.column_names()
        for attr in ("_plot_x", "_plot_y", "_plot_z"):
            if getattr(self, attr) not in names:
                self._error = "Pick three columns to plot in 3D."
                return

        x = ds.get(self._plot_x)
        y = ds.get(self._plot_y)
        z = ds.get(self._plot_z)
        if x is None or y is None or z is None:
            self._error = "Pick three columns to plot in 3D."
            return

        options = self._current_options3d()
        if self._plot_kind3d == "quiver3d":
            for key, attr in (("u", "_plot_u"), ("v", "_plot_v"), ("w", "_plot_w")):
                column = getattr(self, attr)
                options[key] = ds.get(column) if column in names else None
            if all(options.get(k) is None for k in ("u", "v", "w")):
                self._error = "A vector field needs at least one of the U/V/W columns."
                return

        c = ds.get(self._plot_c) if self._plot_c in names else None
        s = None
        if (
            self._plot_s in names
            and layerops3d.kind3d_spec(self._plot_kind3d).primitive == "points"
        ):
            s = self._size_pixels(ds.get(self._plot_s))

        plot = self.plot
        hud = self.hud
        kind = self._plot_kind3d
        cmap = self._plot_cmap
        label = self._unique_layer_label(f"{ds.name}: {self._plot_z}")
        created: dict = {}

        def do() -> None:
            # 3D before the layer lands, so `add_geometry3d`'s camera sync frames it.
            if not plot.active_panel.is_3d():
                plot.set_ndim(3)
            created["layer"] = layerops3d.add_xyz_layer(
                plot, x, y, z, kind=kind, label=label, options=options, c=c, s=s, cmap=cmap
            )
            plot.frame_3d_view()

        def undo() -> None:
            layer = created.pop("layer", None)
            if layer is not None:
                layerops.remove_layer(plot, hud, layer)

        self.push_command(Command(label=f"Plot {label!r} in 3D", do=do, undo=undo))
        self._error = ""
        self._status = f"Plotting {label!r} as {kind}"

    def _size_pixels(self, values: Any) -> Optional[np.ndarray]:
        """Rescale a data column into the marker-size pixel band, as the 2D path does."""
        if values is None:
            return None
        vals = np.asarray(values, dtype=np.float64).ravel()
        lo_px, hi_px = self._SIZE_PX_RANGE
        finite = vals[np.isfinite(vals)]
        if finite.size == 0 or float(np.ptp(finite)) < 1e-12:
            return np.full(vals.shape, 0.5 * (lo_px + hi_px), dtype=np.float32)
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        norm = np.clip((vals - vmin) / (vmax - vmin), 0.0, 1.0)
        return (lo_px + norm * (hi_px - lo_px)).astype(np.float32)

    def _set_plot_kind(self, kind: str) -> None:
        """Switch the picker's kind, resetting its parameters to that kind's defaults.

        Per-kind options are not shared: ``bins`` means nothing to a bar chart, and
        ``layerops`` rejects an option a kind has no use for rather than ignoring it.
        """
        self._plot_kind = kind
        self._plot_opts = layerops.default_kind_options(kind)

    def _current_options(self) -> dict:
        """The selected kind's parameters, guaranteed to be ones that kind accepts.

        ``layerops`` rejects an option a kind has no use for (a silently ignored ``bins``
        would be a bug you debug by staring at the plot), so the picker must not hand it
        leftovers from a kind the user has since switched away from.
        """
        resolved = layerops.default_kind_options(self._plot_kind)
        resolved.update({k: v for k, v in self._plot_opts.items() if k in resolved})
        return resolved

    def _kind_help(self) -> str:
        """Tooltip for the plot-type picker: what each type is, and what is missing."""
        lines = ["What this table can be plotted as:", ""]
        lines += [
            f"  {layerops.kind_spec(k).label} — {layerops.kind_spec(k).description}"
            for k in layerops.KIND_KEYS
        ]
        lines += ["", "Not offered from a table, and why:", ""]
        lines += [f"  {name} — {reason}" for name, reason in layerops.UNSUPPORTED_KINDS.items()]
        lines += [
            "",
            "Changing the type of a plotted dataset converts the layer in place: it "
            "keeps its colour, style, position in the Scene list and selection. The "
            "layer id changes underneath (the engine bakes the type in at creation), "
            "which is why this is the only place that should do it.",
        ]
        return "\n".join(lines)

    def _draw_kind_options(self) -> None:
        """The parameters of the selected plot type. Nothing at all for line/scatter."""
        widgets.kind_options_editor(self._plot_opts)

    #: Colormaps offered for a data-driven colour encoding, common perceptual maps first.
    _ENCODING_CMAPS = (
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "cividis",
        "coolwarm",
        "turbo",
        "jet",
        "hot",
        "cool",
        "spring",
        "rainbow",
    )

    def _draw_encoding_combos(self, names: List[str]) -> None:
        """Colour-by / size-by column pickers -- marker size and colour as data dimensions.

        Only shown for the ``scatter`` kind: it is the plot type where per-point colour and
        size read naturally. "(none)" keeps the flat colour / uniform size. When a colour
        column is chosen a colormap picker appears; the mapping is exactly ``scatter(c=...)``.
        """
        # Colour applies to scatter (per point), line (per vertex) and bar (per bar);
        # size is a scatter-only encoding.
        allows_color = self._plot_kind in ("scatter", "line", "bar")
        allows_size = self._plot_kind == "scatter"
        if not allows_color:
            return
        none = "(none)"
        options = [none] + list(names)

        cur_c = self._plot_c if self._plot_c in names else none
        _, picked_c = widgets.enum_combo("Color by", cur_c, options)
        self._plot_c = "" if picked_c == none else picked_c
        if self._plot_c:
            cur_cmap = self._plot_cmap if self._plot_cmap in self._ENCODING_CMAPS else "viridis"
            _, self._plot_cmap = widgets.enum_combo(
                "Colormap", cur_cmap, list(self._ENCODING_CMAPS)
            )

        if allows_size:
            cur_s = self._plot_s if self._plot_s in names else none
            _, picked_s = widgets.enum_combo("Size by", cur_s, options)
            self._plot_s = "" if picked_s == none else picked_s

    #: Pixel range a size-by column is normalised into, so any column scale is usable.
    _SIZE_PX_RANGE = (6.0, 30.0)

    def _resolve_encoding_arrays(self, ds: DataSet):
        """Return ``(c, s, cmap)`` for the current scatter encodings, or ``(None, None, cmap)``.

        Reads the chosen colour/size columns off ``ds``. Colour is passed raw (the colormap
        normalises it); size is rescaled from the column's own range into a sensible pixel
        band -- a data-unit value used directly as pixels would be unusable at most scales.
        A column that has vanished (renamed, deleted) silently degrades to no mapping.
        """
        if self._plot_kind not in ("scatter", "line", "bar"):
            return None, None, self._plot_cmap
        names = ds.column_names()
        c = ds.get(self._plot_c) if self._plot_c in names else None
        # Size is a scatter-only encoding; bars take colour but not per-bar size.
        s_name = self._plot_s if self._plot_kind == "scatter" else ""
        s_raw = ds.get(s_name) if s_name in names else None
        s = None
        if s_raw is not None:
            vals = np.asarray(s_raw, dtype=np.float64).ravel()
            lo_px, hi_px = self._SIZE_PX_RANGE
            finite = vals[np.isfinite(vals)]
            if finite.size == 0 or float(np.ptp(finite)) < 1e-12:
                s = np.full(vals.shape, 0.5 * (lo_px + hi_px), dtype=np.float32)
            else:
                vmin, vmax = float(np.min(finite)), float(np.max(finite))
                norm = np.clip((vals - vmin) / (vmax - vmin), 0.0, 1.0)
                s = (lo_px + norm * (hi_px - lo_px)).astype(np.float32)
        return c, s, self._plot_cmap

    def _draw_io_section(self, ds: DataSet) -> None:
        """CSV import/export by path (pyimgui has no native file dialog)."""
        imgui.set_next_item_width(-70.0)
        _, self._csv_path = imgui.input_text("##csv", self._csv_path)
        imgui.same_line()
        if icons.icon_button("##imp", "upload", tooltip="Import CSV"):
            self._import_csv()
        imgui.same_line()
        if icons.icon_button("##exp", "download", tooltip="Export CSV"):
            self._export_csv(ds)

    # -- the grid --------------------------------------------------------------

    def _draw_grid(self, ds: DataSet) -> None:
        """The virtualized spreadsheet.

        Row arithmetic is in **view** rows throughout — with no filter the view is the
        table, and with one it is the matching subset. Virtualization is unaffected: the
        spacers reserve ``view_rows * row_h`` either way.
        """
        n_rows, n_cols = self._view_rows_count(ds), ds.n_cols()
        if n_cols == 0:
            imgui.text_disabled("No columns. Add one in the Columns section, or paste data.")
            return

        flags = (
            imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders
            | imgui.TableFlags_.resizable
            | imgui.TableFlags_.scroll_x
            | imgui.TableFlags_.scroll_y
            | imgui.TableFlags_.sizing_fixed_fit
        )
        # TABLE_REORDERABLE is deliberately absent: it reorders the *view* only, which
        # would silently desync from the data order that copy/paste and x/y selection use.

        height = max(120.0, imgui.get_content_region_avail().y)
        table = imgui.begin_table("##grid", n_cols + 1, flags, (0.0, height))
        if not table:
            return

        index_width = imgui.calc_text_size(str(max(ds.n_rows(), 1))).x + 16.0
        imgui.table_setup_column("#", imgui.TableColumnFlags_.width_fixed, index_width)
        for column in ds.columns:
            imgui.table_setup_column(column.name, imgui.TableColumnFlags_.width_fixed, _CELL_WIDTH)
        imgui.table_setup_scroll_freeze(1, 1)
        self._draw_header_row(ds)

        style = imgui.get_style()
        # Measured to give exactly-uniform rows for both selectable and input_text cells.
        row_h = imgui.get_frame_height() + style.cell_padding.y * 2.0
        view_h = imgui.get_window_height()
        scroll_y = imgui.get_scroll_y()

        if self._scroll_to_row is not None:
            scroll_y = self._scroll_into_view(self._scroll_to_row, row_h, view_h, scroll_y)
            self._scroll_to_row = None

        # Clamp against *this* dataset's content height before deriving the row range.
        # imgui keys scroll state to the table id, so switching from a 1e6-row dataset to
        # a short one reports the old, huge scroll for one frame (imgui clamps it only
        # after seeing the new content). Without this, `first` lands past `last` and the
        # grid renders completely empty until the user scrolls. Same hazard after
        # deleting rows or undoing a large paste.
        scroll_y = min(max(scroll_y, 0.0), max(0.0, n_rows * row_h - view_h))

        first = max(0, int(scroll_y / row_h) - _OVERSCAN)
        last = min(n_rows, int((scroll_y + view_h) / row_h) + _OVERSCAN + 1)
        first = min(first, last)

        self._spacer_row(first * row_h)
        for row in range(first, last):
            imgui.table_next_row(0, row_h)
            self._draw_row(ds, row, n_cols)
        self._spacer_row((n_rows - last) * row_h)

        imgui.end_table()

    def _draw_header_row(self, ds: DataSet) -> None:
        """The header row: drag a header to reorder, right-click it to sort/rename.

        Hand-rolled rather than ``table_headers_row()`` because that helper submits the
        whole row and hands back nothing to attach behaviour to. ``table_header`` is the
        same widget it would have drawn, one column at a time, so the header keeps its
        native look while becoming a drag source and a drop target.

        ``TABLE_REORDERABLE`` would have been the lazy way to get the drag, and it is
        deliberately not set (see ``_draw_grid``): it reorders the *view* only, leaving
        the data order that copy, paste and the x/y pickers speak silently different
        from the one on screen. Reordering the columns themselves cannot desync.
        """
        imgui.table_next_row(imgui.TableRowFlags_.headers)
        imgui.table_set_column_index(0)
        imgui.table_header("#")
        for index, column in enumerate(ds.columns):
            imgui.table_set_column_index(index + 1)
            imgui.push_id(f"hdr{index}")
            imgui.table_header(column.name)
            self._column_drag_drop(ds, index)
            self._column_context_menu(ds, index, column.name)
            imgui.pop_id()

    def _column_drag_drop(self, ds: DataSet, index: int) -> None:
        """Make the last item a column drag source and drop target."""
        source = imgui.begin_drag_drop_source()
        if source:
            # Payload is a Python int id (CONTRACT §2.6); the index is all the target needs.
            imgui.set_drag_drop_payload_py_id(_DND_COLUMN, index)
            imgui.text(f"Move column {ds.columns[index].name!r}")
            imgui.end_drag_drop_source()

        target = imgui.begin_drag_drop_target()
        if target:
            # accept_drag_drop_payload_py_id asserts unless it is nested inside an active
            # target block, so it can never be hoisted out of this branch.
            payload = imgui.accept_drag_drop_payload_py_id(_DND_COLUMN)
            if payload is not None:
                self._move_column(ds, self._payload_index(payload), index)
            imgui.end_drag_drop_target()

    def _column_context_menu(self, ds: DataSet, index: int, name: str) -> None:
        """Right-click a header: sort, move, rename, remove — all undoable."""
        popup = imgui.begin_popup_context_item(f"##colmenu{index}")
        if not popup:
            return
        imgui.text_disabled(name)
        imgui.separator()
        if imgui.selectable("Sort ascending")[0]:
            self._sort_by(ds, index, descending=False)
        if imgui.selectable("Sort descending")[0]:
            self._sort_by(ds, index, descending=True)
        imgui.separator()
        if imgui.selectable(
            "Move left", False, 0 if index > 0 else imgui.SelectableFlags_.disabled
        )[0]:
            self._move_column(ds, index, index - 1)
        last = index < ds.n_cols() - 1
        if imgui.selectable("Move right", False, 0 if last else imgui.SelectableFlags_.disabled)[0]:
            self._move_column(ds, index, index + 1)
        imgui.separator()
        if imgui.selectable("Transform...")[0]:
            self._transform_col = name
            self._status = f"Transform {name!r}: see the Filter & transform section"
        if imgui.selectable("Remove column")[0]:
            self._remove_column(ds, name)
        imgui.end_popup()

    @staticmethod
    def _payload_index(payload: Any) -> int:
        """Decode a drag payload back to an index. -1 when it is not one of ours."""
        try:
            return int(payload.data_id)
        except (TypeError, ValueError, AttributeError):  # pragma: no cover - defensive
            return -1

    @staticmethod
    def _scroll_into_view(row: int, row_h: float, view_h: float, scroll_y: float) -> float:
        """Scroll so ``row`` is visible; returns the scroll to use for this frame.

        ``set_scroll_y`` only takes effect next frame, so the new value is returned and
        used for the row-range estimate immediately — otherwise a keyboard-driven jump
        would render one frame of blank rows.
        """
        top = row * row_h
        bottom = top + row_h
        if top < scroll_y:
            scroll_y = top
        elif bottom > scroll_y + view_h:
            scroll_y = bottom - view_h
        imgui.set_scroll_y(scroll_y)
        return scroll_y

    @staticmethod
    def _spacer_row(height: float) -> None:
        """Reserve the height of the rows we skipped, keeping the scrollbar honest."""
        if height <= 0.0:
            return
        imgui.table_next_row()
        imgui.table_set_column_index(0)
        imgui.dummy((1.0, height))

    def _draw_row(self, ds: DataSet, row: int, n_cols: int) -> None:
        """One view row: index gutter + cells.

        The gutter shows the **data** row number, not the view position: under a filter
        the whole point is to know which rows of the table you are looking at, and a
        renumbered 0..N would make the filtered view unreadable against the original.
        """
        r0, r1, c0, c1 = self._range(ds)
        row_selected = r0 <= row <= r1
        data_row = self._data_row(row)

        imgui.table_set_column_index(0)
        if imgui.selectable(f"{data_row}##r{row}", row_selected)[0]:
            self._anchor = (row, 0)
            self._cursor = (row, max(0, n_cols - 1))
        self._row_drag_drop(ds, data_row)

        for col in range(n_cols):
            imgui.table_set_column_index(col + 1)
            self._draw_cell(ds, row, col, row_selected and c0 <= col <= c1)

    def _row_drag_drop(self, ds: DataSet, data_row: int) -> None:
        """Make the row gutter a drag source and drop target, moving whole rows.

        Dragging moves **every selected row** when the dragged one is part of the
        selection, and just the dragged one otherwise — the same rule as a file manager,
        and the only one that makes a multi-row move reachable without a modifier key.

        Disabled under a filter: the drop target is a view position, and the rows between
        two visible ones are not visible, so "drop here" would have no honest meaning.
        """
        if self._filtered:
            return

        source = imgui.begin_drag_drop_source()
        if source:
            imgui.set_drag_drop_payload_py_id(_DND_ROW, data_row)
            rows = self._selected_data_rows(ds)
            count = len(rows) if data_row in rows else 1
            imgui.text(f"Move {count} row(s)")
            imgui.end_drag_drop_source()

        target = imgui.begin_drag_drop_target()
        if target:
            payload = imgui.accept_drag_drop_payload_py_id(_DND_ROW)
            if payload is not None:
                self._move_rows(ds, self._payload_index(payload), data_row)
            imgui.end_drag_drop_target()

    def _draw_cell(self, ds: DataSet, view_row: int, col: int, selected: bool) -> None:
        """One cell: an edit box when editing, otherwise a selectable."""
        row = view_row
        data_row = self._data_row(row)
        if self._edit_cell == (row, col):
            imgui.set_next_item_width(-1.0)
            if self._edit_focus:
                imgui.set_keyboard_focus_here()
                self._edit_focus = False
            entered, self._edit_text = imgui.input_text(
                f"##e{row}_{col}",
                self._edit_text,
                flags=imgui.InputTextFlags_.enter_returns_true
                | imgui.InputTextFlags_.auto_select_all,
            )
            if entered:
                self._commit_edit(ds)
                self._move_cursor(ds, 1, 0)
            elif imgui.is_item_deactivated():
                # Clicked away: Excel commits. Esc already cancelled in _handle_keys.
                self._commit_edit(ds)
            return

        clicked = imgui.selectable(
            f"{_fmt_display(ds.get_cell(data_row, col))}##c{row}_{col}",
            selected,
            imgui.SelectableFlags_.allow_double_click,
        )[0]

        if imgui.is_item_hovered():
            io = imgui.get_io()
            if imgui.is_mouse_double_clicked(0):
                self._begin_edit(ds, row, col)
            elif imgui.is_mouse_clicked(0):
                if io.key_shift:
                    self._cursor = (row, col)
                else:
                    self._anchor = self._cursor = (row, col)
                self._selecting = True
            elif self._selecting and imgui.is_mouse_dragging(0):
                self._cursor = (row, col)
        elif clicked:
            self._anchor = self._cursor = (row, col)

        if self._cursor == (row, col):
            # Outline the active cell so it reads apart from the rest of the range.
            x0, y0 = imgui.get_item_rect_min()
            x1, y1 = imgui.get_item_rect_max()
            imgui.get_window_draw_list().add_rect(
                (x0, y0), (x1, y1), theme.color_u32("accent"), rounding=0.0, thickness=1.5
            )

    # -- dataset / structural actions -----------------------------------------

    def _new_dataset(self) -> None:
        """Create an empty x/y sheet and select it."""
        ds = DataSet(self.store.unique_name("Data"))
        ds.add_column("x", np.full(_NEW_DATASET_ROWS, np.nan))
        ds.add_column("y", np.full(_NEW_DATASET_ROWS, np.nan))
        self.store.add(ds)
        self._select(ds)
        self.push_command(
            Command(
                label=f"New dataset {ds.name!r}",
                do=lambda: self.store.add(ds),
                undo=lambda: self.store.remove(ds),
            )
        )

    def _duplicate_dataset(self) -> None:
        """Copy the current dataset (unlinked from any layer) and select it."""
        source = self._current
        if source is None:
            return
        ds = source.copy(self.store.unique_name(f"{source.name} copy"))
        self.store.add(ds)
        self._select(ds)
        self.push_command(
            Command(
                label=f"Duplicate to {ds.name!r}",
                do=lambda: self.store.add(ds),
                undo=lambda: self.store.remove(ds),
            )
        )

    def _delete_dataset(self) -> None:
        """Remove the current dataset. The layer it drove, if any, stays on the scene."""
        ds = self._current
        if ds is None:
            return
        self.store.remove(ds)
        self._select(None)
        self.push_command(
            Command(
                label=f"Delete dataset {ds.name!r}",
                do=lambda: self.store.remove(ds),
                undo=lambda: self.store.add(ds),
            )
        )

    def _insert_row(self, ds: DataSet) -> None:
        """Insert a zero row above the selection."""
        if not self._require_unfiltered("Insert row"):
            return
        row = self._range(ds)[0]
        self._apply_structural(ds, f"Insert row {row}", lambda: ds.insert_row(row))

    def _delete_rows(self, ds: DataSet) -> None:
        """Delete every row the selection touches."""
        if not self._require_unfiltered("Delete rows"):
            return
        rows = self._selected_data_rows(ds)
        if not rows:
            return
        self._apply_structural(ds, f"Delete {len(rows)} row(s)", lambda: ds.delete_rows(rows))

    def _move_column(self, ds: DataSet, src: int, dst: int) -> None:
        """Move column ``src`` to position ``dst``, undoably.

        Resolved by *name* inside the command, not by the index the drag reported: the
        undo stack replays ``do`` from a restored state, and an index captured from a
        different column order would move the wrong column on redo.
        """
        if not (0 <= src < ds.n_cols()) or src == dst:
            return
        name = ds.columns[src].name
        target = max(0, min(dst, ds.n_cols() - 1))
        self._apply_structural(ds, f"Move column {name!r}", lambda: ds.move_column(name, target))

    def _move_rows(self, ds: DataSet, dragged: int, dest: int) -> None:
        """Move the dragged rows (or the whole selection containing them) before ``dest``."""
        if not self._require_unfiltered("Move rows"):
            return
        selected = self._selected_data_rows(ds)
        rows = selected if dragged in selected else [dragged]
        if not rows or dest in rows:
            return
        self._apply_structural(ds, f"Move {len(rows)} row(s)", lambda: ds.move_rows(rows, dest))

    def _apply_filter_to_data(self, ds: DataSet) -> None:
        """Delete the rows the filter hides. The destructive half — undoable."""
        expr = self._filter_applied or self._filter_text.strip()
        if not expr:
            return
        try:
            mask = ds.row_mask(expr)
        except ExpressionError as exc:
            self._error = f"Filter: {exc}"
            return
        dropped = int(np.count_nonzero(~mask))
        if dropped == 0:
            self._status = "Filter matches every row — nothing to delete."
            return
        # Clear the view first: _apply_structural snapshots the table, and the mask's
        # indices stop meaning anything the moment the rows go.
        self._clear_filter()
        self._apply_structural(
            ds,
            f"Filter: keep {int(np.count_nonzero(mask))}, drop {dropped}",
            lambda: ds.filter_rows(mask),
        )

    def _filter_to_new_dataset(self, ds: DataSet) -> None:
        """Copy the matching rows into a new dataset, leaving this one untouched.

        The default for a *bound* dataset, and the reason it exists: filtering in place
        re-uploads the layer, so one typo'd predicate silently rewrites what is plotted
        with only Ctrl+Z between the user and the data they came with.
        """
        expr = self._filter_applied or self._filter_text.strip()
        if not expr:
            return
        try:
            mask = ds.row_mask(expr)
        except ExpressionError as exc:
            self._error = f"Filter: {exc}"
            return

        clone = ds.copy(self.store.unique_name(f"{ds.name} filtered"))
        clone.filter_rows(mask)
        clone.source = "derived"
        self.store.add(clone)
        self._select(clone)
        self.push_command(
            Command(
                label=f"Filter to {clone.name!r}",
                do=lambda: self.store.add(clone),
                undo=lambda: self.store.remove(clone),
            )
        )
        self._status = f"{clone.name!r}: {clone.n_rows()} of {ds.n_rows()} rows"

    def _transform(self, ds: DataSet, name: str, expr: str, target: Optional[str]) -> None:
        """Write an expression over the columns into a column, undoably.

        Validated here, in the draw callback, rather than inside the queued command:
        a bad expression must report itself in the panel's error box, and by the time
        the command runs at the drain there is nobody left to tell.
        """
        if not self._require_unfiltered("Transform"):
            return
        try:
            ds.eval_expression(expr)
        except ExpressionError as exc:
            self._error = f"Transform: {exc}"
            return
        self._error = ""
        label = f"{target or name} = {expr}"
        self._apply_structural(ds, label, lambda: ds.transform_column(name, expr, target=target))

    def _add_column(self, ds: DataSet, name: str) -> None:
        """Append a zero column."""
        self._apply_structural(ds, f"Add column {name!r}", lambda: ds.add_column(name))

    def _remove_column(self, ds: DataSet, name: str) -> None:
        """Drop a column. If it backed the layer link, DataSet.remove_column unbinds."""
        self._apply_structural(ds, f"Remove column {name!r}", lambda: ds.remove_column(name))

    def _rename_column(self, ds: DataSet, old: str, new: str) -> None:
        """Rename a column, keeping any layer binding intact."""
        self._apply_structural(ds, f"Rename {old!r} -> {new!r}", lambda: ds.rename_column(old, new))

    def _sort_by(self, ds: DataSet, col: int, *, descending: bool) -> None:
        """Reorder every column by the sort order of column ``col``.

        Stable, so equal keys keep their relative order. nan sorts last ascending (numpy's
        convention) and therefore first descending.
        """
        if not (0 <= col < ds.n_cols()):
            return
        if not self._require_unfiltered("Sort"):
            return
        name = ds.columns[col].name

        def mutate() -> None:
            index = ds.index_of(name)
            if index < 0:
                return
            order = np.argsort(ds.columns[index].values, kind="stable")
            if descending:
                order = order[::-1]
            for column in ds.columns:
                column.values = column.values[order]
                column.invalidate()

        self._apply_structural(ds, f"Sort by {name!r}", mutate)

    # -- plotting --------------------------------------------------------------

    def _bound_layer(self, ds: DataSet) -> Optional[Any]:
        """The live layer ``ds`` is bound to, or None. Read-only; safe from ``draw``.

        None with ``ds.layer_id is not None`` is a real state, not a bug: the layer was
        deleted from the Scene panel and the binding is stale.
        """
        if ds.layer_id is None:
            return None
        return layerops.find_layer(self.plot, ds.layer_id)

    def _bind(self, ds: DataSet, layer: Any, x_name: str, y_name: str) -> None:
        """Point ``ds`` at ``layer``. **Inside a queued command only.**

        Delegates to ``layerops.bind_dataset``, which is also what a type change from the
        Scene or Style panel calls — the binding's shape (which attribute, and whether the
        geometry is derived from the columns or *is* them) must not depend on which panel
        the user happened to press.
        """
        layerops.bind_dataset(ds, layer, x_name, y_name)

    def _plot_dataset(self, ds: DataSet, *, as_new: bool = False) -> None:
        """Render ``ds`` on the scene: **update its bound layer**, or add a new one.

        The distinction is the whole point. A dataset already bound to a live layer
        re-plots *into* that layer (``layerops.replot_layer_xy``), because appending
        instead would leave two identically-labelled layers stacked on the same data,
        only one of them bound — so table edits would update one and the other would
        stay frozen at the values it was born with, with nothing on screen to say which
        is which. ``as_new=True`` is the explicit opt-in for genuinely wanting a second,
        independent copy; it never steals the binding.

        Three edge cases, all real and all handled here rather than downstream:
        the bound layer was deleted from the Scene panel (rebind by creating a new one);
        the row count changed (``update_layer_xy`` re-fits the colours); the layer's kind
        differs from the requested one (delete+recreate preserving style/index/selection,
        and re-point the binding at the new ``layer_id``).
        """
        names = ds.column_names()
        if self._plot_x not in names or self._plot_y not in names:
            if len(names) < 2:
                self._error = "Need at least two columns to plot."
                return
            self._plot_x, self._plot_y = names[0], names[1]

        try:
            x, y = ds.to_xy(self._plot_x, self._plot_y)
        except ValueError as exc:
            self._error = str(exc)
            return

        target = None if as_new else self._bound_layer(ds)
        if target is None:
            self._add_plot_layer(ds, x, y, as_new=as_new)
        else:
            self._update_plot_layer(ds, target, x, y)

    def _add_plot_layer(self, ds: DataSet, x: Any, y: Any, *, as_new: bool) -> None:
        """Queue an ``add_xy_layer``, binding the dataset unless this is an extra copy."""
        plot = self.plot
        hud = self.hud
        kind = self._plot_kind
        options = self._current_options()
        x_name, y_name = self._plot_x, self._plot_y
        base = f"{ds.name}: {y_name}"
        label = self._unique_layer_label(base) if as_new else base
        # A stale binding (the bound layer was deleted) must be dropped, or the new
        # layer would be created while ds.layer_id still names a corpse.
        stale = ds.layer_id is not None
        bind = not as_new
        created: dict = {}
        # Resolve colour/size encodings now (while ds is current); passed to add_xy_layer so
        # a scatter can carry a data-driven colour and/or size. None for other kinds.
        c_enc, s_enc, cmap_enc = self._resolve_encoding_arrays(ds)

        def do() -> None:
            if bind and stale:
                ds.unbind()
            layer = layerops.add_xy_layer(
                plot,
                x,
                y,
                kind=kind,
                label=label,
                options=options,
                c=c_enc,
                s=s_enc,
                cmap=cmap_enc,
            )
            created["layer"] = layer
            if bind:
                self._bind(ds, layer, x_name, y_name)

        def undo() -> None:
            layer = created.pop("layer", None)
            if layer is not None:
                layerops.remove_layer(plot, hud, layer)
            if bind:
                ds.unbind()

        verb = "Plot as new layer" if as_new else "Plot"
        self.push_command(Command(label=f"{verb} {label!r}", do=do, undo=undo))
        if as_new:
            self._status = f"Added independent layer {label!r} (not linked)"
        elif stale:
            self._status = f"Linked layer was gone — re-plotted {label!r}"
        else:
            self._status = f"Plotting {label!r}"

    def _update_plot_layer(self, ds: DataSet, layer: Any, x: Any, y: Any) -> None:
        """Queue a re-plot **into** ``layer``, the layer ``ds`` is already bound to."""
        plot = self.plot
        hud = self.hud
        kind = self._plot_kind
        options = self._current_options()
        x_name, y_name = self._plot_x, self._plot_y
        label = f"{ds.name}: {y_name}"

        old_kind = layerops.layer_kind(layer)
        same_kind = old_kind == kind
        old_label = getattr(layer, "label", "") or label
        old_options = layerops.layer_kind_options(layer)
        old_columns = list(ds.binding.columns) if ds.binding is not None else [x_name, y_name]

        # Undo restores the layer's previous data, so it needs a copy of that data — the
        # *source* data, not the geometry: undoing a bar->line must restore the columns
        # the bars came from, and layer_xy would hand back the bars' corners. It can only
        # restore the previous kind if that kind is knowable.
        before = layerops.layer_source_xy(layer)
        if before is not None and not is_snapshot_safe(*before):
            before = None
        undoable = before is not None and old_kind is not None
        created: dict = {}

        def do() -> None:
            # Re-resolve: an earlier command in this same drain may have removed it.
            target = layerops.find_layer(plot, ds.layer_id)
            if target is None:
                new_layer = layerops.add_xy_layer(
                    plot, x, y, kind=kind, label=label, options=options
                )
            else:
                new_layer = layerops.replot_layer_xy(
                    plot, hud, target, x, y, kind=kind, label=label, options=options
                )
            created["layer_id"] = getattr(new_layer, "layer_id", None)
            self._bind(ds, new_layer, x_name, y_name)

        def undo() -> None:
            target = layerops.find_layer(plot, created.pop("layer_id", None))
            if target is None or before is None or old_kind is None:
                return
            restored = layerops.replot_layer_xy(
                plot,
                hud,
                target,
                before[0],
                before[1],
                kind=old_kind,
                label=old_label,
                options=old_options,
            )
            self._bind(ds, restored, old_columns[0], old_columns[-1])

        if same_kind:
            action = f"Update layer {label!r}"
        else:
            action = f"Replot {label!r} as {kind}"

        if undoable:
            self.push_command(Command(label=action, do=do, undo=undo))
            self._status = action
        else:
            self.push_command(Command.not_undoable(action, do))
            reason = (
                "too large to undo" if before is None else f"{layer.layer_type} is not undoable"
            )
            self._status = f"{action} — {reason}"

    def _unique_layer_label(self, base: str) -> str:
        """``base``, suffixed until no live layer already carries it.

        Two layers with the same label are indistinguishable in the Scene panel, which
        is precisely the confusion an extra copy must not create.
        """
        taken = {str(getattr(layer, "label", "")) for layer in self.plot.scene.layers}
        if base not in taken:
            return base
        for n in range(2, len(taken) + 3):
            candidate = f"{base} ({n})"
            if candidate not in taken:
                return candidate
        return base

    # -- csv -------------------------------------------------------------------

    def _import_csv(self) -> None:
        """Read the CSV at the path field into a new dataset."""
        path = self._csv_path.strip()
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as handle:
                text = handle.read()
        except OSError as exc:
            self._error = f"Could not read {path!r}: {exc}"
            return
        try:
            headers, data = clipboard.parse_table(text)
        except ValueError as exc:
            self._error = f"{path!r}: {exc}"
            return
        base = os.path.splitext(os.path.basename(path))[0] or "Imported"
        ds = self._dataset_from_table(headers, data, base)
        ds.source = "manual"
        self._status = f"Imported {ds.n_rows()} x {ds.n_cols()} from {base}"

    def _export_csv(self, ds: DataSet) -> None:
        """Write the current dataset to the path field as CSV."""
        path = self._csv_path.strip()
        try:
            text = clipboard.format_table(ds.column_names(), ds.to_array(), delimiter=",")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
        except (OSError, ValueError) as exc:
            self._error = f"Could not write {path!r}: {exc}"
            return
        self._status = f"Exported {ds.n_rows()} rows to {path}"
