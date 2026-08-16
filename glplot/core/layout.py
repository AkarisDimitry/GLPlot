"""Figure layouts: turn a grid / mosaic / explicit spec into panel rectangles.

Every function here returns a list of :class:`PanelSpec` -- a rectangle in figure fractions
``(x0, y0, w, h)`` with a bottom-left origin (the convention :class:`~glplot.core.panel.Panel`
stores and ``glViewport`` consumes), plus an optional name and its ``(row, col)`` origin so
callers can index panels the matplotlib way.

The engine consumes these to (re)build ``self.panels``. Spacing and outer margins are
expressed as figure fractions, kept deliberately simple: this is the *panel* layout (how the
window is carved into regions), distinct from the per-panel *axis gutters* (``axis_margin_*``
in the options), which each panel still applies inside its own rectangle for ticks/labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

#: Default outer margin of the whole grid, as a figure fraction, on each side. 0 by default so
#: panels reach the window edge; each panel still reserves its own axis gutter (``axis_margin_*``)
#: inside its rect for ticks and labels, which is separate from this.
DEFAULT_OUTER_MARGIN = 0.0
#: Default gap between cells, as a fraction of the figure. 0 packs panels edge to edge; the
#: per-panel axis gutters still keep their tick labels from touching.
DEFAULT_WSPACE = 0.0
DEFAULT_HSPACE = 0.0


@dataclass
class PanelSpec:
    """One panel's placement: a bottom-left-origin figure-fraction rect plus its grid origin."""

    rect_frac: Tuple[float, float, float, float]  # (x0, y0, w, h)
    name: Optional[str] = None
    row: int = 0  # top row is 0 (matplotlib order)
    col: int = 0  # left column is 0
    rowspan: int = 1
    colspan: int = 1


def _cell_edges(
    n: int,
    outer_lo: float,
    outer_hi: float,
    spacing: float,
    ratios: Optional[Sequence[float]] = None,
) -> List[Tuple[float, float]]:
    """Return ``n`` ``(start, end)`` fraction intervals along one axis.

    ``outer_lo``/``outer_hi`` are the margins reserved before the first and after the last
    cell; ``spacing`` is the gap between adjacent cells. All are figure fractions.

    ``ratios`` gives the cells relative sizes (matplotlib's ``width_ratios`` /
    ``height_ratios``). The *gaps* are not scaled with them -- spacing is a fixed figure
    fraction, so only the space left after the gaps is divided in proportion, which is what
    keeps a 3:1 split reading as 3:1 rather than 3:1 plus an uneven gutter.
    """
    n = max(1, int(n))
    usable = max(1e-6, 1.0 - outer_lo - outer_hi - spacing * (n - 1))
    if ratios is None:
        weights = [1.0] * n
    else:
        weights = [float(r) for r in ratios]
        if len(weights) != n:
            raise ValueError(f"expected {n} ratios, got {len(weights)}")
        if any(w <= 0 for w in weights):
            raise ValueError("ratios must be positive")
    total = sum(weights)

    edges = []
    pos = outer_lo
    for w in weights:
        cell = usable * w / total
        edges.append((pos, pos + cell))
        pos += cell + spacing
    return edges


def grid(
    nrows: int,
    ncols: int,
    *,
    wspace: float = DEFAULT_WSPACE,
    hspace: float = DEFAULT_HSPACE,
    outer: float = DEFAULT_OUTER_MARGIN,
    width_ratios: Optional[Sequence[float]] = None,
    height_ratios: Optional[Sequence[float]] = None,
) -> List[PanelSpec]:
    """Rectangles for an ``nrows`` x ``ncols`` grid, in row-major (matplotlib) order.

    Row 0 is the *top* row, matching ``plt.subplots``: the returned list is ordered so that
    ``specs[row * ncols + col]`` is the panel at that grid position.

    ``width_ratios`` / ``height_ratios`` give the columns and rows relative sizes, as in
    matplotlib. ``height_ratios`` is indexed top-down like the rows themselves, so
    ``height_ratios=[3, 1]`` makes the *top* row the tall one.
    """
    nrows = max(1, int(nrows))
    ncols = max(1, int(ncols))

    x_edges = _cell_edges(ncols, outer, outer, wspace, width_ratios)
    # y in figure fractions is bottom-origin; row 0 (top) maps to the highest interval.
    # The ratios are reversed alongside the edges so entry 0 still describes the top row.
    y_ratios = None if height_ratios is None else list(height_ratios)[::-1]
    y_edges_top_down = list(reversed(_cell_edges(nrows, outer, outer, hspace, y_ratios)))

    specs: List[PanelSpec] = []
    for r in range(nrows):
        y0, y1 = y_edges_top_down[r]
        for c in range(ncols):
            x0, x1 = x_edges[c]
            specs.append(PanelSpec(rect_frac=(x0, y0, x1 - x0, y1 - y0), row=r, col=c))
    return specs


def grid_span(
    nrows: int,
    ncols: int,
    loc: Tuple[int, int],
    rowspan: int = 1,
    colspan: int = 1,
    *,
    wspace: float = DEFAULT_WSPACE,
    hspace: float = DEFAULT_HSPACE,
    outer: float = DEFAULT_OUTER_MARGIN,
) -> PanelSpec:
    """One rectangle spanning ``rowspan`` x ``colspan`` cells of an ``nrows`` x ``ncols`` grid.

    ``loc`` is ``(row, col)`` of the top-left cell (row 0 = top). Backs ``subplot2grid``.
    """
    x_edges = _cell_edges(max(1, ncols), outer, outer, wspace)
    y_edges_top_down = list(reversed(_cell_edges(max(1, nrows), outer, outer, hspace)))

    r0, c0 = int(loc[0]), int(loc[1])
    r1 = min(max(1, nrows) - 1, r0 + max(1, rowspan) - 1)
    c1 = min(max(1, ncols) - 1, c0 + max(1, colspan) - 1)

    x0 = x_edges[c0][0]
    x1 = x_edges[c1][1]
    y_top = y_edges_top_down[r0][1]
    y_bot = y_edges_top_down[r1][0]
    return PanelSpec(
        rect_frac=(x0, y_bot, x1 - x0, y_top - y_bot),
        row=r0,
        col=c0,
        rowspan=max(1, rowspan),
        colspan=max(1, colspan),
    )


def mosaic(
    spec: Sequence[Sequence[str]],
    *,
    wspace: float = DEFAULT_WSPACE,
    hspace: float = DEFAULT_HSPACE,
    outer: float = DEFAULT_OUTER_MARGIN,
) -> List[PanelSpec]:
    """Rectangles for a matplotlib ``subplot_mosaic`` layout given as a 2-D list of labels.

    Each distinct label becomes one panel spanning the (contiguous, rectangular) block of
    cells it covers. ``"."`` marks an empty cell (no panel). Example::

        mosaic([["A", "A"],
                ["B", "C"]])

    yields three panels: ``A`` spanning the top row, ``B`` bottom-left, ``C`` bottom-right.
    The result is ordered by first appearance of each label, scanning top-to-bottom then
    left-to-right.
    """
    rows = [list(r) for r in spec]
    nrows = len(rows)
    ncols = max((len(r) for r in rows), default=0)
    if nrows == 0 or ncols == 0:
        return []

    x_edges = _cell_edges(ncols, outer, outer, wspace)
    y_edges_top_down = list(reversed(_cell_edges(nrows, outer, outer, hspace)))

    # Bounding box of every label's cells, in grid coordinates.
    order: List[str] = []
    bounds: Dict[str, List[int]] = {}  # label -> [rmin, rmax, cmin, cmax]
    for r in range(nrows):
        for c in range(ncols):
            label = rows[r][c] if c < len(rows[r]) else "."
            if label == "." or label == "":
                continue
            if label not in bounds:
                bounds[label] = [r, r, c, c]
                order.append(label)
            else:
                b = bounds[label]
                b[0], b[1] = min(b[0], r), max(b[1], r)
                b[2], b[3] = min(b[2], c), max(b[3], c)

    specs: List[PanelSpec] = []
    for label in order:
        rmin, rmax, cmin, cmax = bounds[label]
        x0 = x_edges[cmin][0]
        x1 = x_edges[cmax][1]
        y_top = y_edges_top_down[rmin][1]
        y_bot = y_edges_top_down[rmax][0]
        specs.append(
            PanelSpec(
                rect_frac=(x0, y_bot, x1 - x0, y_top - y_bot),
                name=label,
                row=rmin,
                col=cmin,
                rowspan=rmax - rmin + 1,
                colspan=cmax - cmin + 1,
            )
        )
    return specs


def explicit(rects: Sequence[Tuple[float, float, float, float]]) -> List[PanelSpec]:
    """Panels at arbitrary ``(x0, y0, w, h)`` figure-fraction rectangles (bottom-left origin)."""
    return [PanelSpec(rect_frac=tuple(float(v) for v in rect)) for rect in rects]


def full() -> List[PanelSpec]:
    """A single panel covering the whole figure -- the default, pre-multi-panel layout."""
    return [PanelSpec(rect_frac=(0.0, 0.0, 1.0, 1.0))]
