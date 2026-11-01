"""Text columns in a loaded table: dates become numbers, names become categories.

GLPlot's tables hold float64 columns only, and the file importer turns a cell it cannot read as a
number into ``nan``. That is right for a stray ``n/a`` in a measurement and wrong for a column of
city names or dates, whose every value silently vanished -- a model that loaded a weather CSV saw
``city: nan_count 6`` and no hint that the information was ever there.

:func:`interpret_text_columns` gives such a column a meaning the rest of the toolbox can use:

* every non-empty cell is an ISO date/time (``2020-01-05``, ``2020-01-05 10:30``) -> the column
  becomes *days since 1970-01-01* (a float, so it plots, fits and differences like any number);
* otherwise -> a **category** column: the stored value is the index of the label (first
  appearance order), and the labels live in ``DataSet.categories``. ``select_rows(equals=...)``,
  ``analyze_dataset(operation='aggregate')`` and bar charts read them.

Pure logic: no MCP SDK, no GL.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ..gui.datasets import DataSet

__all__ = ["DATE_UNIT", "MAX_CATEGORIES", "interpret_text_columns", "parse_date"]

#: How a date column is expressed.
DATE_UNIT = "days since 1970-01-01"
#: More distinct values than this is free text or an identifier, not a category worth encoding.
MAX_CATEGORIES = 500

_EPOCH = datetime(1970, 1, 1)


def parse_date(text: str) -> Optional[float]:
    """``text`` as days since 1970-01-01, or None if it is not an ISO date/time."""
    cell = text.strip()
    if not cell:
        return None
    try:
        moment = datetime.fromisoformat(cell)
    except ValueError:
        return None
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return (moment - _EPOCH).total_seconds() / 86400.0


def _set_values(table: DataSet, name: str, values: np.ndarray) -> None:
    for column in table.columns:
        if column.name == name:
            column.values = values
            column.invalidate()
            return
    raise KeyError(name)


def interpret_text_columns(table: DataSet, text: Mapping[str, List[str]]) -> Dict[str, Any]:
    """Give each text column of ``table`` a numeric meaning, in place; return what was done.

    ``text`` maps a column name to its raw cells (see :func:`glplot.gui.dataio.text_columns`).
    The report has ``categories`` (name -> labels), ``dates`` (name -> unit) and ``ignored``
    (name -> why a column was left empty); keys with nothing to say are omitted.
    """
    categories: Dict[str, List[str]] = {}
    dates: Dict[str, str] = {}
    ignored: Dict[str, str] = {}
    for name, cells in text.items():
        if len(cells) != table.n_rows():
            ignored[name] = "its rows do not line up with the numeric table"
            continue
        filled = [cell for cell in cells if cell]
        parsed = [parse_date(cell) for cell in filled]
        if filled and all(value is not None for value in parsed):
            lookup = {cell: value for cell, value in zip(filled, parsed)}
            values = np.array([lookup[c] if c else np.nan for c in cells], dtype=np.float64)
            _set_values(table, name, values)
            table.date_columns[name] = DATE_UNIT
            dates[name] = DATE_UNIT
            continue
        labels = list(dict.fromkeys(filled))
        if len(labels) > MAX_CATEGORIES:
            ignored[name] = (
                f"{len(labels)} distinct text values (more than {MAX_CATEGORIES}): looks like "
                "free text or an id, so it was left empty"
            )
            continue
        index = {label: float(i) for i, label in enumerate(labels)}
        values = np.array([index[c] if c else np.nan for c in cells], dtype=np.float64)
        _set_values(table, name, values)
        table.categories[name] = labels
        categories[name] = labels
    report: Dict[str, Any] = {}
    if categories:
        report["categories"] = categories
    if dates:
        report["dates"] = dates
    if ignored:
        report["ignored"] = ignored
    return report


def describe_encoding(report: Mapping[str, Any]) -> str:
    """One sentence for a reply: what the text columns became."""
    parts: List[str] = []
    for name, labels in (report.get("categories") or {}).items():
        shown = ", ".join(f"{i}={label}" for i, label in enumerate(labels[:6]))
        more = f", ... ({len(labels)} in all)" if len(labels) > 6 else ""
        parts.append(f"{name}: {len(labels)} categories ({shown}{more})")
    for name, unit in (report.get("dates") or {}).items():
        parts.append(f"{name}: dates as {unit}")
    for name, why in (report.get("ignored") or {}).items():
        parts.append(f"{name}: {why}")
    return "Text columns -> " + "; ".join(parts) + "." if parts else ""
