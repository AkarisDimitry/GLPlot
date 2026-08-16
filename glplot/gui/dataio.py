"""File importers: numpy archives, JSON, delimited text and images -> :class:`DataSet`.

Pure logic -- numpy, stdlib and (lazily, for images only) matplotlib. No imgui, no engine
imports (CONTRACT 5.1 rule 7), exactly like :mod:`glplot.gui.datasets`,
:mod:`glplot.gui.dynamics` and :mod:`glplot.gui.generators3d`. An import panel is a shell
over :func:`load_dataset`; nothing here knows a window exists.

One entry point
---------------
:func:`load_dataset` dispatches on the file extension and always returns a
:class:`~glplot.gui.datasets.DataSet` or raises :class:`DataIOError` (a ``ValueError``, so
callers that already catch ``ValueError`` around ``load_table_file`` need no new except
clause). Every message names the file and what was wrong with it -- the convention
established by :func:`glplot.gui.app.load_table_file`.

The per-format loaders are public too (:func:`load_array_file`, :func:`load_json_file`,
:func:`load_text_file`, :func:`load_image`) for a caller that already knows what it has,
but a panel should use :func:`load_dataset`.

Why the table is always float64 columns
---------------------------------------
Because :class:`~glplot.gui.datasets.DataSet` is: it holds equal-length float64 columns and
nothing else. So every importer's job is the same -- decide what the *columns* are, and
turn each one into a float64 array in which anything unrepresentable is ``nan``. That is
also why a string field does not raise: :func:`glplot.gui.clipboard.parse_table` already
established that a non-numeric cell is ``nan`` rather than an error, and an importer that
refused a file because one label column was not a number would be useless on real data.

What is *not* silently guessed
------------------------------
* A 2-D array's **second axis is the columns**, always. No transposition heuristic: a
  ``(3, 5000)`` array becomes 5000 columns of 3 rows, loudly wrong, instead of sometimes
  swapping x and y depending on which side happened to be longer.
* Pickled arrays are refused (``np.load(..., allow_pickle=False)``). Unpickling executes
  arbitrary code, and "open this file" must never mean "run this file".
* Float image data is taken as-is rather than auto-ranged; see :func:`load_image`.

Images
------
See :func:`load_image` for the full contract. In short: an image comes back as **both** a
2-D luminance matrix (``ImageImport.luminance``, for ``imshow`` / height-field / surface
use) and a long ``(x, y, lum, r, g, b)`` table (:meth:`ImageImport.to_dataset`), it is
block-averaged down to :data:`MAX_IMAGE_PIXELS` if it is bigger, and the dataset name says
so when it was.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .clipboard import detect_delimiter, parse_table
from .datasets import Column, DataSet

#: Extensions read as delimited text through :func:`glplot.gui.clipboard.parse_table`.
#: Also the fallback for an unknown extension -- a delimited file called ``.data`` is far
#: more likely than a format this module has never heard of.
TEXT_EXTENSIONS: Tuple[str, ...] = (".csv", ".tsv", ".txt", ".dat", ".tab", ".asc", ".text")

#: Characters that start a comment line in a text file: ``#`` (numpy, gnuplot, most
#: instruments) and ``%`` (MATLAB, LaTeX-adjacent exports). ``//`` is honoured too. A data
#: line never begins with one of these, so no numeric row is ever mistaken for a comment.
COMMENT_MARKERS: Tuple[str, ...] = ("#", "%")

#: Extensions read with :func:`numpy.load`.
ARRAY_EXTENSIONS: Tuple[str, ...] = (".npy", ".npz")

#: Extensions read with :mod:`json`.
JSON_EXTENSIONS: Tuple[str, ...] = (".json",)

#: Extensions read with ``matplotlib.image.imread``. Everything past ``.png`` needs Pillow,
#: which matplotlib itself depends on, so in practice all of these work wherever GLPlot
#: does -- and :func:`load_image` says so explicitly if one does not.
IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".ppm",
    ".pgm",
    ".webp",
)

#: Everything :func:`load_dataset` recognises, for a file-dialog filter.
SUPPORTED_EXTENSIONS: Tuple[str, ...] = tuple(
    sorted(set(TEXT_EXTENSIONS + ARRAY_EXTENSIONS + JSON_EXTENSIONS + IMAGE_EXTENSIONS))
)

#: Pixel budget for an imported image. One megapixel is ~1e6 table rows and, at six
#: columns of float64, ~48 MB -- already the largest table the Data panel is comfortable
#: with. Bigger images are block-averaged down (see :func:`load_image`); the reduction is
#: reported in the dataset name, never silently.
MAX_IMAGE_PIXELS = 1_000_000

#: Rec. 709 luma weights, the ones sRGB displays are built around, applied to the stored
#: (gamma-encoded) channel values. Rejected alternatives: the flat mean, which makes blue
#: as bright as green and turns a colour photograph into mush; Rec. 601 (0.299/0.587/
#: 0.114), which is right for analogue NTSC and wrong for every file written this century;
#: and linearising first, which is more correct photometrically but silently changes the
#: numbers of anyone importing an image as *data* rather than as a picture.
LUMA_WEIGHTS: Tuple[float, float, float] = (0.2126, 0.7152, 0.0722)


class DataIOError(ValueError):
    """Raised for an unreadable or unusable file.

    A :class:`ValueError` subclass so a caller that already catches ``ValueError`` around
    the existing text loader catches this too -- the same relationship
    :class:`glplot.gui.generators3d.GeneratorError` has to its callers.
    """


# ----------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------


def file_stem(path: str) -> str:
    """The base name of ``path`` without its extension, or ``"data"`` if that is empty."""
    return os.path.splitext(os.path.basename(str(path)))[0] or "data"


def extension_of(path: str) -> str:
    """The lower-cased extension of ``path``, including the dot (``""`` when there is none)."""
    return os.path.splitext(str(path))[1].lower()


def is_image(path: str) -> bool:
    """True when ``path``'s extension is one :func:`load_image` handles."""
    return extension_of(path) in IMAGE_EXTENSIONS


def is_supported(path: str) -> bool:
    """True when ``path``'s extension is one :func:`load_dataset` recognises by name.

    False does **not** mean the file will fail: an unknown extension is read as delimited
    text. It means "this module has no specific reader for it".
    """
    return extension_of(path) in SUPPORTED_EXTENSIONS


def _as_float_column(values: Any, where: str) -> np.ndarray:
    """Coerce ``values`` to a 1-D float64 array, or raise naming ``where``.

    Booleans become 0.0/1.0 and integers widen. Complex data raises rather than dropping
    the imaginary part in silence -- ``abs()`` and ``.real`` are different answers and only
    the user knows which one was meant.
    """
    array = np.asarray(values)
    if array.dtype == object or array.dtype.kind in "SUV":
        # An object/string array is per-element: parse what parses, nan for the rest.
        return np.array([_to_float(item) for item in np.ravel(array)], dtype=np.float64)
    if np.iscomplexobj(array):
        raise DataIOError(
            f"{where} holds complex numbers; a table column is real. "
            "Save the magnitude or the real part instead."
        )
    return np.ravel(np.asarray(array, dtype=np.float64))


def _to_float(value: Any) -> float:
    """One JSON/object value as a float, or ``nan`` if it is not a number.

    ``None``, ``True``/``False``, numeric strings and nested containers all have an obvious
    answer here (nan, 1.0/0.0, the parsed number, nan), and none of them is worth an
    exception: a records file with one text field is still a perfectly good table.
    """
    if value is None or isinstance(value, (list, dict, tuple)):
        return float("nan")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _dataset_from_columns(
    name: str, names: Sequence[str], arrays: Sequence[np.ndarray], source: str, path: str
) -> DataSet:
    """Assemble a named DataSet, raising a file-scoped error on a length disagreement."""
    lengths = {len(a) for a in arrays}
    if len(lengths) > 1:
        detail = ", ".join(f"{n} ({len(a)})" for n, a in zip(names, arrays))
        raise DataIOError(
            f"Could not import {path}: its columns have different lengths ({detail}); "
            "a table needs one value per column per row."
        )
    dataset = DataSet(name, [Column(n, a) for n, a in zip(names, arrays)])
    dataset.source = source
    return dataset


def _broadcast_scalars(
    columns: Dict[str, np.ndarray], scalars: Dict[str, float], n_rows: int
) -> None:
    """Expand scalar entries to full columns, in place.

    A JSON object or an ``.npz`` mixing arrays with single values ("``t``: [...], ``dt``:
    0.01") is the normal shape of an exported run, and the scalar is a real per-row
    constant. Broadcasting it beats dropping it, and beats refusing the file.
    """
    for key, value in scalars.items():
        columns[key] = np.full(n_rows, value, dtype=np.float64)


# ----------------------------------------------------------------------------------
# Delimited text
# ----------------------------------------------------------------------------------


def strip_comments(text: str) -> Tuple[str, Optional[str]]:
    """Remove comment lines, returning ``(body, header_hint)``.

    A line whose first non-blank character is in :data:`COMMENT_MARKERS` is a comment. This
    exists because files have preambles and clipboards do not, and because handing the
    preamble to :func:`glplot.gui.clipboard.parse_table` is not merely untidy -- it is
    wrong. ``# t  y`` is a perfectly good non-numeric first row, so parse_table would adopt
    it as the header and every column would come back shifted by one, named ``#``, ``t``,
    ``y``. Rejected alternative: teaching parse_table about comments, which would change the
    behaviour of paste for everyone to fix a problem only files have.

    The *last* comment line before the first data line is returned as ``header_hint``,
    stripped of its marker: the numpy/gnuplot convention is that this line names the
    columns, and :func:`load_text_file` uses it when the body itself carries no header.
    Earlier comment lines are prose and are discarded.
    """
    kept: List[str] = []
    hint: Optional[str] = None
    seen_data = False
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped[:1] in COMMENT_MARKERS or stripped[:2] == "//":
            if not seen_data:
                hint = stripped.lstrip("#%/ \t")
            continue
        if stripped:
            seen_data = True
        kept.append(line)
    return "\n".join(kept), (hint or None)


def _header_from_hint(hint: str, n_cols: int) -> Optional[List[str]]:
    """Split a comment-line header hint into exactly ``n_cols`` names, or None.

    Anything else -- a prose comment, a units line, a mismatched count -- returns None and
    the generated ``col1 .. colN`` names stand. Guessing a partial header would be worse
    than not guessing: half the columns would be mislabelled with no way to tell which.
    """
    delimiter = detect_delimiter(hint)
    cells = hint.split(delimiter) if delimiter else hint.split()
    cells = [cell.strip() for cell in cells]
    if len(cells) != n_cols or not all(cells):
        return None
    return cells


def load_text_file(path: str, *, name: Optional[str] = None) -> DataSet:
    """Read a delimited-text file (CSV/TSV/semicolon/whitespace) into a DataSet.

    The parsing itself is :func:`glplot.gui.clipboard.parse_table` -- the same codec the
    editor's Ctrl+V uses, deliberately not a second implementation, so a file and a paste
    of that file's contents produce identical tables (header detection, quoted fields,
    ragged rows padded with nan, all of it).

    This module owns the three things a *file* adds and a clipboard paste never has:

    * decoding (utf-8 with a BOM stripped, undecodable bytes replaced rather than fatal),
    * **comment lines** -- see :func:`strip_comments`, and
    * a file-scoped error message.
    """
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
            text = handle.read()
    except OSError as exc:
        raise DataIOError(f"Could not read {path}: {exc}") from exc

    body, hint = strip_comments(text)
    if not body.strip():
        # parse_table's own message for this case says "nothing to paste", which is the
        # wrong noun for a file and would not tell the user that the comment stripper is
        # what emptied it.
        detail = "it holds only comments" if hint is not None else "it is empty"
        raise DataIOError(f"Could not parse {path}: {detail}.")
    try:
        headers, data = parse_table(body)
    except ValueError as exc:
        raise DataIOError(f"Could not parse {path}: {exc}") from exc

    if hint is not None and headers == [f"col{i + 1}" for i in range(data.shape[1])]:
        recovered = _header_from_hint(hint, data.shape[1])
        if recovered is not None:
            headers = recovered

    columns = [Column(headers[i], data[:, i]) for i in range(data.shape[1])]
    dataset = DataSet(name or file_stem(path), columns)
    dataset.source = "file"
    return dataset


# ----------------------------------------------------------------------------------
# numpy archives
# ----------------------------------------------------------------------------------


def _columns_from_2d(array: np.ndarray, base: str, where: str) -> List[Tuple[str, np.ndarray]]:
    """Split a 2-D array into ``(name, values)`` pairs, one per column of its second axis.

    A single-column array keeps ``base`` unsuffixed: an ``(n, 1)`` member called ``y`` is
    the column ``y``, not the column ``y1``.
    """
    if array.shape[1] == 0:
        raise DataIOError(f"{where} has zero columns.")
    if array.shape[1] == 1:
        return [(base, _as_float_column(array[:, 0], where))]
    return [(f"{base}{i + 1}", _as_float_column(array[:, i], where)) for i in range(array.shape[1])]


def load_npy_file(path: str, *, name: Optional[str] = None) -> DataSet:
    """Read a ``.npy`` array into a DataSet.

    * 1-D ``(n,)`` -- and ``(n, 1)``, which is the same table -- becomes a single column
      named after the file.
    * 2-D ``(n, k)`` becomes ``k`` columns ``col1 .. colk`` -- the second axis is the
      columns, always (see the module docstring).
    * 0-D and 3-D-or-more raise: a scalar is not a table, and an ``(n, m, k)`` array has no
      column reading that is not a guess. Slice or reshape it first.
    """
    array = _load_numpy(path)
    if not isinstance(array, np.ndarray):  # pragma: no cover - np.load's own contract
        raise DataIOError(f"Could not import {path}: it does not hold an array.")

    stem = file_stem(path)
    if array.ndim == 1 or (array.ndim == 2 and array.shape[1] == 1):
        pairs = [(stem, _as_float_column(array, str(path)))]
    elif array.ndim == 2:
        pairs = _columns_from_2d(array, "col", str(path))
    else:
        raise DataIOError(
            f"Could not import {path}: it holds a {array.ndim}-D array of shape "
            f"{tuple(array.shape)}; a table needs 1-D or 2-D data."
        )
    return _dataset_from_columns(
        name or stem, [p[0] for p in pairs], [p[1] for p in pairs], "file", str(path)
    )


def load_npz_file(path: str, *, name: Optional[str] = None) -> DataSet:
    """Read a ``.npz`` archive into a DataSet: one column per member.

    Each 1-D member becomes a column under its own name -- which is the point of an
    ``.npz`` over an ``.npy``, the names are the schema. A 2-D member ``m`` of ``k``
    columns becomes ``m1 .. mk``; a 0-D member becomes a constant column (see
    :func:`_broadcast_scalars`). Members must agree in length, and the error names the
    ones that did not.
    """
    archive = _load_numpy(path)
    members = list(getattr(archive, "files", []))
    if not members:
        raise DataIOError(f"Could not import {path}: the archive is empty.")

    columns: Dict[str, np.ndarray] = {}
    scalars: Dict[str, float] = {}
    try:
        for member in members:
            values = archive[member]
            where = f"{path}: member {member!r}"
            if values.ndim == 0:
                scalars[member] = _to_float(values.item())
            elif values.ndim == 1:
                columns[member] = _as_float_column(values, where)
            elif values.ndim == 2:
                for column_name, column in _columns_from_2d(values, member, where):
                    columns[column_name] = column
            else:
                raise DataIOError(
                    f"Could not import {path}: member {member!r} is {values.ndim}-D "
                    f"(shape {tuple(values.shape)}); a table needs 1-D or 2-D data."
                )
    finally:
        close = getattr(archive, "close", None)
        if close is not None:
            close()

    if not columns:
        if not scalars:  # pragma: no cover - unreachable: members is non-empty
            raise DataIOError(f"Could not import {path}: it holds no usable columns.")
        # An archive of nothing but scalars is a single record -- one row, one column each.
        columns = {key: np.array([value], dtype=np.float64) for key, value in scalars.items()}
        scalars = {}

    n_rows = len(next(iter(columns.values())))
    _broadcast_scalars(columns, scalars, n_rows)
    ordered = [key for key in columns]
    return _dataset_from_columns(
        name or file_stem(path), ordered, [columns[k] for k in ordered], "file", str(path)
    )


def _load_numpy(path: str) -> Any:
    """``np.load`` with pickling off and every failure turned into a DataIOError."""
    try:
        return np.load(path, allow_pickle=False)
    except OSError as exc:
        raise DataIOError(f"Could not read {path}: {exc}") from exc
    except ValueError as exc:
        if "allow_pickle" in str(exc):
            # numpy's own message is accurate but says nothing about what to do next.
            raise DataIOError(
                f"Could not read {path}: it holds a pickled (object) array. Loading one "
                "would execute the code inside it, so it is refused; re-save it with a "
                "numeric dtype."
            ) from exc
        raise DataIOError(f"Could not read {path}: {exc}") from exc


def load_array_file(path: str, *, name: Optional[str] = None) -> DataSet:
    """Read ``.npy`` or ``.npz``, dispatching on the extension."""
    if extension_of(path) == ".npz":
        return load_npz_file(path, name=name)
    return load_npy_file(path, name=name)


# ----------------------------------------------------------------------------------
# JSON
# ----------------------------------------------------------------------------------


def _columns_from_records(records: Sequence[Mapping[str, Any]], path: str) -> Dict[str, list]:
    """Column-ise a list of JSON objects.

    Keys are collected in first-seen order **across all records**, not from the first one:
    real record files leave a field out when it is missing, and taking the first record as
    the schema would drop every column that happens to be absent from row 0. A record that
    lacks a key contributes ``nan`` there.
    """
    columns: Dict[str, list] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise DataIOError(
                f"Could not import {path}: entry {index} is a {type(record).__name__}, "
                "but the list starts with objects. Records must all be objects."
            )
        for key in record:
            columns.setdefault(str(key), [])
    for record in records:
        for key, values in columns.items():
            values.append(_to_float(record.get(key)))
    return columns


def load_json_file(path: str, *, name: Optional[str] = None) -> DataSet:
    """Read a ``.json`` file into a DataSet.

    Four shapes are understood, which between them cover what tools actually emit:

    * ``[{"t": 0, "y": 1}, ...]`` -- a list of records. One column per key, in first-seen
      order, ``nan`` where a record omits it.
    * ``{"t": [...], "y": [...]}`` -- an object of equal-length arrays. Scalar values
      alongside them become constant columns.
    * ``[[1, 2], [3, 4]]`` -- a list of rows, named ``col1 .. colk`` and padded with
      ``nan`` when ragged.
    * ``[1, 2, 3]`` -- a flat list, one column named after the file.

    Anything else raises with the shape it actually found. Non-numeric values become
    ``nan`` rather than failing the import.
    """
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise DataIOError(f"Could not read {path}: {exc}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DataIOError(f"Could not parse {path}: it is not valid JSON ({exc}).") from exc

    stem = file_stem(path)

    if isinstance(payload, dict):
        columns: Dict[str, np.ndarray] = {}
        scalars: Dict[str, float] = {}
        for key, value in payload.items():
            if isinstance(value, (list, tuple)):
                columns[str(key)] = _as_float_column(
                    [_to_float(item) for item in value], f"{path}: key {key!r}"
                )
            elif isinstance(value, dict):
                raise DataIOError(
                    f"Could not import {path}: key {key!r} holds a nested object; this "
                    "reader takes an object of arrays, not a tree. Flatten it first."
                )
            else:
                scalars[str(key)] = _to_float(value)
        if not columns:
            if not scalars:
                raise DataIOError(f"Could not import {path}: the object is empty.")
            columns = {k: np.array([v], dtype=np.float64) for k, v in scalars.items()}
            scalars = {}
        _broadcast_scalars(columns, scalars, len(next(iter(columns.values()))))
        keys = list(columns)
        return _dataset_from_columns(
            name or stem, keys, [columns[k] for k in keys], "file", str(path)
        )

    if isinstance(payload, list):
        if not payload:
            raise DataIOError(f"Could not import {path}: the list is empty.")
        if isinstance(payload[0], dict):
            record_columns = _columns_from_records(payload, str(path))
            keys = list(record_columns)
            arrays = [np.asarray(record_columns[k], dtype=np.float64) for k in keys]
            return _dataset_from_columns(name or stem, keys, arrays, "file", str(path))
        if isinstance(payload[0], (list, tuple)):
            width = max(len(row) if isinstance(row, (list, tuple)) else 1 for row in payload)
            data = np.full((len(payload), width), np.nan, dtype=np.float64)
            for i, row in enumerate(payload):
                cells = row if isinstance(row, (list, tuple)) else [row]
                for j, cell in enumerate(cells[:width]):
                    data[i, j] = _to_float(cell)
            names = [f"col{j + 1}" for j in range(width)]
            return _dataset_from_columns(
                name or stem, names, [data[:, j] for j in range(width)], "file", str(path)
            )
        values = np.array([_to_float(item) for item in payload], dtype=np.float64)
        return _dataset_from_columns(name or stem, [stem], [values], "file", str(path))

    raise DataIOError(
        f"Could not import {path}: its top level is a {type(payload).__name__}; expected a "
        "list of records, an object of arrays, or a list of rows."
    )


# ----------------------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageImport:
    """An imported image, in both of the forms a plotting tool needs.

    Attributes
    ----------
    name
        Display name, including the imported size and the reduction factor when the image
        was downsampled. Used as the dataset name by :meth:`to_dataset`.
    luminance
        ``(h, w)`` float64 in ``0 .. 1``, in **image order**: row 0 is the top row, as it
        is on disk and as ``imshow(..., origin="upper")`` expects. This is the height field
        / matrix form.
    rgb
        ``(h, w, 3)`` float64 in ``0 .. 1``. A grayscale source gives three identical
        channels, so the shape of this array never depends on the file's channel count.
    alpha
        ``(h, w)`` float64, or None when the file had no alpha channel.
    step
        Block size used to downsample: 1 means the image was imported at full resolution,
        ``k`` means each pixel here is the mean of a ``k x k`` block of the original.
    source_shape
        ``(H, W)`` of the file on disk, before any downsampling.
    """

    name: str
    luminance: np.ndarray
    rgb: np.ndarray
    alpha: Optional[np.ndarray]
    step: int
    source_shape: Tuple[int, int]

    @property
    def height(self) -> int:
        """Rows in the imported image (after downsampling)."""
        return int(self.luminance.shape[0])

    @property
    def width(self) -> int:
        """Columns in the imported image (after downsampling)."""
        return int(self.luminance.shape[1])

    @property
    def downsampled(self) -> bool:
        """True when the imported image is smaller than the file on disk."""
        return self.step > 1

    def to_dataset(self, name: Optional[str] = None) -> DataSet:
        """The long form: one row per pixel, columns ``x, y, lum, r, g, b`` (and ``a``).

        Coordinates are in **original-image pixel units**: ``x = column * step`` and
        ``y = (h - 1 - row) * step``. Two consequences, both deliberate:

        * ``y`` is flipped, so the picture is upright in a viewer whose y axis points up.
          The matrix in :attr:`luminance` is *not* flipped -- it stays in image order for
          ``imshow``. The two conventions disagree because their consumers do.
        * The extent does not change when the cap kicks in, so the same image imported at
          two different resolutions overlays itself instead of shrinking.

        The ``(x, y)`` pairs form a full rectangular lattice, so
        :func:`glplot.gui.layerops3d.grid_shape` detects it and the table can be plotted as
        a surface or a wireframe height field with no re-sorting.

        ``a`` is present only when the file had an alpha channel, and it is *not* applied
        to ``r/g/b`` or to ``lum``: compositing needs a background colour, and inventing
        one would put numbers in the table that are in no file. Composite explicitly if you
        want it (``r * a`` over black, ``r * a + (1 - a)`` over white).
        """
        h, w = self.height, self.width
        step = float(self.step)
        xs = np.tile(np.arange(w, dtype=np.float64) * step, h)
        ys = np.repeat((h - 1 - np.arange(h, dtype=np.float64)) * step, w)
        names = ["x", "y", "lum", "r", "g", "b"]
        arrays = [
            xs,
            ys,
            self.luminance.ravel(),
            self.rgb[:, :, 0].ravel(),
            self.rgb[:, :, 1].ravel(),
            self.rgb[:, :, 2].ravel(),
        ]
        if self.alpha is not None:
            names.append("a")
            arrays.append(self.alpha.ravel())
        dataset = DataSet(name or self.name, [Column(n, a) for n, a in zip(names, arrays)])
        dataset.source = "image"
        return dataset


def _normalize_pixels(pixels: np.ndarray, path: str) -> np.ndarray:
    """Scale raw image data to float64 in ``0 .. 1``.

    Integer dtypes are divided by their own maximum (255 for the uint8 arrays Pillow
    returns, 65535 for 16-bit). Float data is taken **as-is**: matplotlib's PNG path
    already returns ``0 .. 1``, and auto-ranging by the observed min/max would make two
    crops of the same image incomparable -- a dark tile would come back looking like a
    bright one. A float file that is really ``0 .. 255`` therefore imports as ``0 .. 255``,
    which is visible in the column statistics rather than hidden by a rescale.
    """
    if pixels.dtype.kind in "ui":
        info = np.iinfo(pixels.dtype)
        return pixels.astype(np.float64) / float(info.max)
    if pixels.dtype.kind == "b":
        return pixels.astype(np.float64)
    if pixels.dtype.kind != "f":  # pragma: no cover - imread returns nothing else
        raise DataIOError(f"Could not import {path}: unsupported pixel dtype {pixels.dtype}.")
    return pixels.astype(np.float64)


def _block_reduce(pixels: np.ndarray, max_pixels: int) -> Tuple[np.ndarray, int]:
    """Average ``pixels`` down in square blocks until it fits ``max_pixels``.

    Returns ``(reduced, step)``. Block **mean** rather than decimation: an image imported
    as data is usually a field, and the mean of a block is the honest low-resolution value
    of that field, whereas taking every k-th pixel aliases -- a fine grating would come
    back as a coarse one that is not in the original at all.

    The trailing rows and columns that do not fill a whole block are cropped (at most
    ``step - 1`` of each), because padding them would fabricate edge pixels.

    One stated limitation: the block is square, to preserve the aspect ratio, so an extreme
    strip (``1 x 10^7``) cannot be reduced along its short axis and comes back over the
    cap. That shape is not worth a second code path.
    """
    h, w = pixels.shape[:2]
    total = h * w
    if total <= max_pixels or max_pixels <= 0:
        return pixels, 1
    step = int(math.ceil(math.sqrt(total / float(max_pixels))))
    step = max(1, min(step, h, w))
    if step == 1:
        return pixels, 1
    hb, wb = h // step, w // step
    cropped = pixels[: hb * step, : wb * step]
    shape = (hb, step, wb, step) + pixels.shape[2:]
    return cropped.reshape(shape).mean(axis=(1, 3)), step


def load_image(path: str, *, max_pixels: int = MAX_IMAGE_PIXELS) -> ImageImport:
    """Read an image file into an :class:`ImageImport`.

    Loading is ``matplotlib.image.imread`` -- matplotlib is already a hard dependency, so
    this adds nothing to the install, and it is the same reader ``imshow`` uses. The import
    is deferred to call time rather than done at module level so that importing a CSV does
    not pay for matplotlib.

    Colour reduction
    ----------------
    ``lum = 0.2126 R + 0.7152 G + 0.0722 B`` (:data:`LUMA_WEIGHTS`, Rec. 709), applied to
    the stored channel values without linearising -- see that constant for the rejected
    alternatives. A grayscale file is already its own luminance and is copied into all
    three channels unchanged, so ``lum == r == g == b`` exactly.

    Alpha
    -----
    Kept as a separate channel (:attr:`ImageImport.alpha`, and an ``a`` column) and applied
    to nothing. RGB and luminance are the values stored in the file, whatever the alpha
    says about them. Compositing requires choosing a background colour, and choosing one
    here would silently write invented numbers into a table the user is about to do
    arithmetic on. Note the consequence: a fully transparent region still has whatever
    colour the encoder left there, often black.

    Size
    ----
    Images above ``max_pixels`` (default :data:`MAX_IMAGE_PIXELS`) are block-averaged down
    by an integer factor -- see :func:`_block_reduce` -- and the factor and the original
    size go into :attr:`ImageImport.name`, so a downsampled import cannot be mistaken for a
    full-resolution one.

    Raises
    ------
    DataIOError
        If the file cannot be read or decoded, or if it does not hold 2-D image data.
    """
    try:
        from matplotlib import image as mpimg
    except ImportError as exc:  # pragma: no cover - matplotlib is a hard dependency
        raise DataIOError(f"Could not import {path}: matplotlib is required ({exc}).") from exc

    try:
        raw = mpimg.imread(path)
    except FileNotFoundError as exc:
        raise DataIOError(f"Could not read {path}: {exc}") from exc
    except Exception as exc:
        # imread raises whatever its backend raises (SyntaxError from Pillow on a truncated
        # file, ValueError on an unknown format...). None of those help; name the file.
        raise DataIOError(
            f"Could not read {path} as an image: {type(exc).__name__}: {exc}"
        ) from exc

    pixels = np.asarray(raw)
    if pixels.ndim not in (2, 3) or pixels.size == 0:
        raise DataIOError(
            f"Could not import {path}: it decoded to shape {tuple(pixels.shape)}, which is "
            "not 2-D image data."
        )
    if pixels.ndim == 3 and pixels.shape[2] not in (1, 2, 3, 4):
        raise DataIOError(
            f"Could not import {path}: it has {pixels.shape[2]} channels; expected 1 "
            "(gray), 2 (gray+alpha), 3 (RGB) or 4 (RGBA)."
        )

    source_shape = (int(pixels.shape[0]), int(pixels.shape[1]))
    pixels, step = _block_reduce(pixels, int(max_pixels))
    values = _normalize_pixels(pixels, str(path))

    if values.ndim == 2:
        gray, alpha = values, None
    elif values.shape[2] == 1:
        gray, alpha = values[:, :, 0], None
    elif values.shape[2] == 2:
        gray, alpha = values[:, :, 0], values[:, :, 1]
    else:
        gray, alpha = None, (values[:, :, 3] if values.shape[2] == 4 else None)

    if gray is not None:
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        lum = gray
    else:
        rgb = values[:, :, :3]
        weights = np.asarray(LUMA_WEIGHTS, dtype=np.float64)
        lum = rgb @ weights

    h, w = int(lum.shape[0]), int(lum.shape[1])
    label = f"{file_stem(path)} {w}x{h}"
    if step > 1:
        label += f" (downsampled {step}x from {source_shape[1]}x{source_shape[0]})"
    return ImageImport(
        name=label,
        luminance=np.ascontiguousarray(lum, dtype=np.float64),
        rgb=np.ascontiguousarray(rgb, dtype=np.float64),
        alpha=None if alpha is None else np.ascontiguousarray(alpha, dtype=np.float64),
        step=int(step),
        source_shape=source_shape,
    )


def load_image_file(
    path: str, *, name: Optional[str] = None, max_pixels: int = MAX_IMAGE_PIXELS
) -> DataSet:
    """Read an image straight into its long ``(x, y, lum, r, g, b)`` table.

    Shorthand for ``load_image(path).to_dataset()``. Use :func:`load_image` instead when
    the 2-D luminance matrix is wanted as well -- the dataset does not carry it, only the
    (griddable) long form.
    """
    return load_image(path, max_pixels=max_pixels).to_dataset(name)


# ----------------------------------------------------------------------------------
# The entry point
# ----------------------------------------------------------------------------------


def load_dataset(
    path: str, *, name: Optional[str] = None, max_pixels: int = MAX_IMAGE_PIXELS
) -> DataSet:
    """Import any supported file into a :class:`~glplot.gui.datasets.DataSet`.

    The one function a panel should call. Dispatch is by extension:

    ===================  ==========================================================
    ``.npy`` ``.npz``    :func:`load_array_file`
    ``.json``            :func:`load_json_file`
    image extensions     :func:`load_image_file` (see :data:`IMAGE_EXTENSIONS`)
    anything else        :func:`load_text_file` -- CSV/TSV/whitespace, via
                         :func:`glplot.gui.clipboard.parse_table`
    ===================  ==========================================================

    Text is the fallback rather than an error because an unknown extension on a delimited
    file (``.data``, ``.out``, no extension at all) is common and a format this module has
    never heard of is not; a genuinely binary file then fails in the text parser, which
    still names it.

    The returned dataset is **not** registered in any DataStore -- the caller does that,
    inside a queued command, exactly as the generator and dynamics panels do. Its ``name``
    defaults to the file's base name (for an image, the base name plus the imported size),
    and ``source`` is ``"file"`` (``"image"`` for images).

    Raises
    ------
    DataIOError
        A ``ValueError``, naming the file and what was wrong with it.
    """
    extension = extension_of(path)
    if extension in ARRAY_EXTENSIONS:
        return load_array_file(path, name=name)
    if extension in JSON_EXTENSIONS:
        return load_json_file(path, name=name)
    if extension in IMAGE_EXTENSIONS:
        return load_image_file(path, name=name, max_pixels=max_pixels)
    return load_text_file(path, name=name)
