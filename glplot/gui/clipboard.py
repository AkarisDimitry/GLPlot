"""Pure text <-> table codec for the GUI clipboard path.

This module is the transport codec behind Ctrl+C / Ctrl+V in the data editor. It is
deliberately pure logic: it imports only numpy and the stdlib, never imgui, so it is
unit-testable headless (see CONTRACT 5.1 rule 7).

Quoted fields are handled by the stdlib :mod:`csv` module rather than by hand, so an
Excel paste like ``"a,b"\\t1`` round-trips correctly.
"""

from __future__ import annotations

import csv
import io
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Delimiter candidates, in the priority order fixed by the spec. ``None`` is the
# implicit fallback and means "runs of arbitrary whitespace".
_DELIMITER_PRIORITY: Tuple[str, ...] = ("\t", ",", ";")

_BOM = "﻿"


def _normalize_newlines(text: str) -> str:
    """Collapse CRLF and bare CR to LF and strip a leading UTF-8 BOM.

    Excel writes CRLF; classic Mac exports write bare CR; CSV exports often carry a BOM.
    Normalizing inside quoted fields too is intentional -- an embedded newline stays a
    newline, it just loses its carriage return.
    """
    if text.startswith(_BOM):
        text = text[len(_BOM) :]
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _parse_float(cell: str) -> Optional[float]:
    """Return the float value of ``cell``, or None if it is empty or non-numeric."""
    cell = cell.strip()
    if not cell:
        return None
    try:
        return float(cell)
    except (TypeError, ValueError):
        return None


def detect_delimiter(text: str) -> Optional[str]:
    """Guess the delimiter of ``text``.

    Returns one of ``"\\t"``, ``","``, ``";"`` -- the first of those present in the text,
    in that priority order -- or ``None`` meaning "split on runs of whitespace".

    The priority order is what makes tab-separated Excel data containing quoted commas
    resolve to tab rather than comma. It also means European ``1,5;2,5`` decimal-comma
    data resolves to comma (wrong) rather than semicolon; that trade-off is fixed by the
    spec and is not second-guessed here.
    """
    for candidate in _DELIMITER_PRIORITY:
        if candidate in text:
            return candidate
    return None


def _split_rows(text: str, delimiter: Optional[str]) -> List[List[str]]:
    """Split normalized ``text`` into rows of raw string cells.

    Fully empty rows are dropped, which is what makes a trailing newline (and stray blank
    lines in a paste) harmless.
    """
    rows: List[List[str]] = []
    if delimiter is None:
        # No quoting semantics exist in the whitespace path, so str.split is exact.
        for line in text.split("\n"):
            cells = line.split()
            if cells:
                rows.append(cells)
        return rows

    try:
        # csv.reader over a file-like object correctly rejoins newlines that live inside
        # quoted fields, which naive line splitting would tear apart.
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        for cells in reader:
            if any(cell.strip() for cell in cells):
                rows.append(list(cells))
    except csv.Error as exc:
        raise ValueError(f"Could not parse the table: {exc}") from exc
    return rows


def _looks_like_header(row: Sequence[str]) -> bool:
    """True if ``row`` is a header row.

    A row is a header when it holds at least one cell that is non-empty and does not
    parse as a float. Empty cells deliberately do NOT count: a body row with a hole
    (``",5"``) is data with a missing value, not a header, and treating it as one would
    produce a column literally named "".
    """
    return any(cell.strip() and _parse_float(cell) is None for cell in row)


def _make_headers(row: Sequence[str], n_cols: int) -> List[str]:
    """Build ``n_cols`` header names from ``row``, generating names for empty/missing cells."""
    headers: List[str] = []
    for i in range(n_cols):
        cell = row[i].strip() if i < len(row) else ""
        headers.append(cell if cell else f"col{i + 1}")
    return headers


def parse_table(text: str) -> Tuple[List[str], np.ndarray]:
    """Parse clipboard/file text into ``(headers, data)``.

    Handles Excel TSV (tabs + CRLF), CSV, semicolon-separated and whitespace-separated
    text, quoted fields containing the delimiter, bare CR line endings, a leading BOM,
    trailing newlines and ragged rows.

    Semantics:

    * The delimiter is auto-detected by :func:`detect_delimiter`.
    * The first row is treated as a header when it holds any non-empty, non-numeric cell;
      otherwise headers are generated as ``["col1", "col2", ...]``.
    * Non-numeric and empty body cells become ``nan``.
    * Ragged rows are padded with ``nan`` out to the widest row.

    Returns
    -------
    (headers, data)
        ``len(headers) == data.shape[1]`` and ``data`` is a C-contiguous float64 array of
        shape ``(n_rows, n_cols)``. A header-only input yields ``n_rows == 0`` rather
        than an error -- it is a valid, empty table.

    Raises
    ------
    ValueError
        With a human-readable message when the input is empty or unparseable.
    """
    if not text or not text.strip():
        raise ValueError("Nothing to paste: the input is empty.")

    normalized = _normalize_newlines(text)
    delimiter = detect_delimiter(normalized)
    rows = _split_rows(normalized, delimiter)
    if not rows:
        raise ValueError("Nothing to paste: no non-empty rows were found.")

    n_cols = max(len(row) for row in rows)
    if n_cols == 0:
        raise ValueError("Nothing to paste: no columns were found.")

    if _looks_like_header(rows[0]):
        headers = _make_headers(rows[0], n_cols)
        body = rows[1:]
    else:
        headers = _make_headers([], n_cols)
        body = rows

    data = np.full((len(body), n_cols), np.nan, dtype=np.float64)
    for i, row in enumerate(body):
        for j in range(min(len(row), n_cols)):
            value = _parse_float(row[j])
            if value is not None:
                data[i, j] = value
    return headers, data


def format_table(
    headers: Sequence[str],
    data: np.ndarray,
    *,
    delimiter: str = "\t",
    include_header: bool = True,
    fmt: str = "%.10g",
) -> str:
    """Format a table as delimited text for the clipboard.

    ``nan`` renders as an empty cell, which is what :func:`parse_table` reads back as
    ``nan``, so ``parse_table(format_table(h, d))`` round-trips (given non-numeric header
    names -- numeric-looking headers are indistinguishable from data on the way back).

    Quoting is delegated to :mod:`csv` with ``QUOTE_MINIMAL``, so a header or value
    containing the delimiter, a quote or a newline is quoted rather than corrupting the
    column layout. Rows are terminated with ``"\\n"`` (never CRLF); Excel accepts either.

    Parameters
    ----------
    headers
        One name per column. Only required when ``include_header`` is True.
    data
        Shape ``(n_rows, n_cols)``. A 1-D array is accepted as a single column.
    delimiter
        Exactly one character.
    fmt
        Printf-style format applied to every non-nan value.

    Raises
    ------
    ValueError
        On a non-2-D array, a multi-character delimiter, or a header/column count
        mismatch when ``include_header`` is True.
    """
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise ValueError(f"delimiter must be exactly one character, got {delimiter!r}.")

    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"data must be 1-D or 2-D, got {arr.ndim} dimensions.")

    header_list = list(headers)
    if include_header and len(header_list) != arr.shape[1]:
        raise ValueError(
            f"Header/column mismatch: {len(header_list)} headers for {arr.shape[1]} columns."
        )

    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=delimiter, lineterminator="\n")
    if include_header:
        writer.writerow(header_list)
    for row in arr:
        writer.writerow(["" if math.isnan(value) else fmt % value for value in row])
    return buf.getvalue()
