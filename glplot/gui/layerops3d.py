"""3D plot kinds: how three columns of a table become a 3D layer.

The 2D counterpart of this module is the ``_KIND_REGISTRY`` in
:mod:`glplot.gui.layerops`, and the design is deliberately the same one: a kind is a
:class:`PlotKind3D` record whose ``geom`` maps ``(x, y, z, options)`` to renderer-ready
geometry, and *both* the builder and the in-place updater call it, so "what Plot draws" and
"what a cell edit redraws" cannot drift apart.

Why a separate registry rather than more entries in the 2D one
--------------------------------------------------------------
``layerops`` answers "what can a **two**-column table become". Its kinds all compile to
``polyline``/``scatter``/``patch``/``line_family``, all share one ``(xs, ys)`` signature,
and :func:`~glplot.gui.layerops.set_layer_kind` freely converts between any two of them —
a line and a bar chart are the same numbers drawn differently.

A 3D kind takes three columns (six, for ``quiver3d``), compiles to a
:class:`~glplot.core.layers.Layer3D`, and is *not* convertible to or from a 2D kind: there
is no z to invent when converting a line into a surface, and no way to keep one when going
the other way. Two registries with one shared metadata tag says that plainly; one merged
registry would need a "which dimensionality" field on every entry and a conversion matrix
that is empty in both off-diagonal blocks.

Until now the GUI could not build *any* of these. ``layerops.UNSUPPORTED_KINDS`` said so
outright — "a 3-D layer needs the scene's 3-D camera and axes, which the Data panel does
not own. Use gplt.scatter3d from code." — which stopped being true when the camera became
panel state (:class:`glplot.core.camera3d.Camera3D`).

Gridded kinds
-------------
``surface3d``, ``wireframe3d`` and ``mesh3d`` need the ``(x, y)`` pairs to form a regular
lattice: a surface is defined by its *connectivity*, and a bag of scattered points has
none. :func:`grid_shape` detects the lattice and :func:`reshape_to_grid` recovers it in
row-major order regardless of which column varies fastest. Scattered input raises
:class:`Grid3DError` with the counts it found, because "surface needs gridded data" is
only actionable if it also says what it saw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Metadata key holding the 3D kind, shared with the 2D registry's tag so one lookup
#: answers "what kind is this layer" for both. See :func:`layer_kind3d`.
KIND3D_METADATA_KEY = "gui_kind"

#: Largest grid a surface kind will triangulate. ``nx * ny`` cells become ``2 * nx * ny``
#: triangles and six vertices each; past this the upload dominates the frame and the user
#: wants a density view, not a mesh.
MAX_GRID_CELLS = 4_000_000

#: Largest number of arrows ``quiver3d`` will build. Each is 6 vertices (shaft + two head
#: barbs), so this is ~1.2M vertices — the point where one more arrow stops adding
#: information to the picture.
MAX_ARROWS = 200_000


class Grid3DError(ValueError):
    """Raised when a kind needs gridded ``(x, y)`` and the columns are not a lattice.

    A :class:`ValueError` subclass so panel actions that already catch ``ValueError``
    catch this too, matching :class:`glplot.gui.dynamics.DynamicsError`.
    """


# ----------------------------------------------------------------------------------
# Grid detection
# ----------------------------------------------------------------------------------


def grid_shape(xs: np.ndarray, ys: np.ndarray) -> Optional[Tuple[int, int]]:
    """``(nx, ny)`` if the ``(x, y)`` pairs form a full rectangular lattice, else None.

    "Full" is the strict test — every one of the ``nx * ny`` combinations present exactly
    once — because a partially filled lattice would triangulate into a surface with holes
    silently stitched shut. Detection is by unique counts and a length check, which is
    O(n log n) and, unlike a tolerance-based clustering, has no threshold to tune.
    """
    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    n = xs.size
    if n == 0 or ys.size != n:
        return None
    ux = np.unique(xs)
    uy = np.unique(ys)
    nx, ny = int(ux.size), int(uy.size)
    if nx < 2 or ny < 2 or nx * ny != n:
        return None
    # Unique counts multiplying out is necessary but not sufficient: (0,0),(0,0),(1,1),(1,1)
    # has 2 x 2 = 4 too. Confirm every cell is occupied exactly once.
    ix = np.searchsorted(ux, xs)
    iy = np.searchsorted(uy, ys)
    seen = np.zeros(nx * ny, dtype=bool)
    seen[ix * ny + iy] = True
    if not bool(seen.all()):
        return None
    return nx, ny


def reshape_to_grid(
    xs: np.ndarray, ys: np.ndarray, zs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover ``(X, Y, Z)`` as ``(ny, nx)`` arrays from three flat columns.

    Row-major with x along the columns and y along the rows — matplotlib's ``meshgrid``
    convention, so a grid built by the Functions panel and one pasted into the Data panel
    triangulate identically. Raises :class:`Grid3DError` when the pairs are not a lattice.
    """
    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    zs = np.asarray(zs, dtype=np.float64).ravel()
    shape = grid_shape(xs, ys)
    if shape is None:
        raise Grid3DError(
            f"this plot type needs the x/y columns to form a regular grid; got "
            f"{len(np.unique(xs))} distinct x and {len(np.unique(ys))} distinct y "
            f"over {xs.size} rows, which is not a full lattice. Use 'scatter 3D' for "
            f"scattered points, or build the grid in the Functions panel (z = f(x, y))."
        )
    nx, ny = shape
    if nx * ny > MAX_GRID_CELLS:
        raise Grid3DError(f"grid is {nx}x{ny}; the limit is {MAX_GRID_CELLS} points.")
    ux = np.unique(xs)
    uy = np.unique(ys)
    ix = np.searchsorted(ux, xs)
    iy = np.searchsorted(uy, ys)
    grid_z = np.empty((ny, nx), dtype=np.float64)
    grid_z[iy, ix] = zs
    grid_x = np.tile(ux, (ny, 1))
    grid_y = np.tile(uy[:, None], (1, nx))
    return grid_x, grid_y, grid_z


# ----------------------------------------------------------------------------------
# Geometry builders
# ----------------------------------------------------------------------------------


def _xyz(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """The three columns as a contiguous float32 ``(N, 3)`` vertex array."""
    return np.ascontiguousarray(
        np.column_stack(
            [
                np.asarray(xs, dtype=np.float64).ravel(),
                np.asarray(ys, dtype=np.float64).ravel(),
                np.asarray(zs, dtype=np.float64).ravel(),
            ]
        ),
        dtype=np.float32,
    )


def points_geometry(xs, ys, zs, opts: Dict[str, Any]):
    """One vertex per row. The identity kind: ``scatter3d`` and ``volume3d``."""
    verts = _xyz(xs, ys, zs)
    return verts, None, verts[:, 2].astype(np.float64)


def path_geometry(xs, ys, zs, opts: Dict[str, Any]):
    """The rows connected in table order, expanded to ``GL_LINES`` pairs.

    Expanded rather than drawn as a strip because the 3D renderer's primitive map has no
    ``GL_LINE_STRIP`` entry — the same expansion :func:`glplot.pyplot.plot3d` performs, so
    a path built here and one built from code are the same array.
    """
    verts = _xyz(xs, ys, zs)
    if len(verts) < 2:
        return verts, None, verts[:, 2].astype(np.float64)
    segs = np.empty(((len(verts) - 1) * 2, 3), dtype=np.float32)
    segs[0::2] = verts[:-1]
    segs[1::2] = verts[1:]
    return segs, None, segs[:, 2].astype(np.float64)


def stem_geometry(xs, ys, zs, opts: Dict[str, Any]):
    """A vertical segment from the baseline up to each point."""
    base = float(opts.get("baseline", 0.0))
    verts = _xyz(xs, ys, zs)
    n = len(verts)
    segs = np.empty((n * 2, 3), dtype=np.float32)
    segs[0::2, :2] = verts[:, :2]
    segs[0::2, 2] = base
    segs[1::2] = verts
    return segs, None, segs[:, 2].astype(np.float64)


def ribbon_geometry(xs, ys, zs, opts: Dict[str, Any]):
    """A filled band hanging from the path down to the baseline plane.

    Two triangles per segment. The 3D equivalent of ``fill_between``: it reads the height
    profile of a trajectory far better than the bare line does, because the eye gets a
    surface to judge the depth of.
    """
    base = float(opts.get("baseline", 0.0))
    verts = _xyz(xs, ys, zs)
    if len(verts) < 2:
        return np.zeros((0, 3), dtype=np.float32), None, np.zeros(0)
    top = verts
    bottom = verts.copy()
    bottom[:, 2] = base
    n = len(verts) - 1
    tris = np.empty((n * 6, 3), dtype=np.float32)
    tris[0::6] = bottom[:-1]
    tris[1::6] = bottom[1:]
    tris[2::6] = top[1:]
    tris[3::6] = bottom[:-1]
    tris[4::6] = top[1:]
    tris[5::6] = top[:-1]
    return tris, None, tris[:, 2].astype(np.float64)


def resolve_grid(
    xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, opts: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(X, Y, Z)`` as ``(nv, nu)`` arrays, from an explicit shape or by detection.

    Two kinds of surface reach here and they are gridded in *different spaces*:

    * A **height field** — ``z = f(x, y)`` pasted as a table, or built by the Functions
      panel — is a lattice in ``(x, y)``, and :func:`reshape_to_grid` recovers it.
    * A **parametric surface** — a sphere, a torus, a Möbius strip — is a lattice in its
      own ``(u, v)`` parameters and emphatically *not* in ``(x, y)``: a sphere's x and y
      both cycle, so no amount of sorting turns them into a rectangular grid. Its
      connectivity is only knowable from the sampling shape, which the generator knows and
      passes down as ``nu``/``nv``.

    Detection alone therefore cannot serve both, and a surface kind that only detected
    would reject every parametric surface in the catalogue as "not gridded". An explicit
    ``nu * nv == N`` wins; absent that, the ``(x, y)`` lattice is detected as before.
    """
    nu = int(opts.get("nu", 0) or 0)
    nv = int(opts.get("nv", 0) or 0)
    n = int(xs.size)
    if nu > 1 and nv > 1 and nu * nv == n:
        # Row-major with u along the columns: the order `np.meshgrid(u, v).ravel()`
        # produces, which is what every generator in the catalogue emits.
        return xs.reshape(nv, nu), ys.reshape(nv, nu), zs.reshape(nv, nu)
    if (nu > 1 or nv > 1) and nu * nv != n:
        raise Grid3DError(
            f"the grid shape {nu}x{nv} does not match the {n} rows given. Set both "
            f"'grid u' and 'grid v' so they multiply to the row count, or set them to 0 "
            f"to detect a rectangular (x, y) lattice instead."
        )
    return reshape_to_grid(xs, ys, zs)


def surface_geometry(xs, ys, zs, opts: Dict[str, Any]):
    """A triangulated surface over a gridded ``(x, y)`` or an explicit ``(nu, nv)`` shape.

    Indexed rather than expanded: a ``nx * ny`` grid has ``2 * (nx-1) * (ny-1)`` triangles,
    and sharing the vertices between them costs a third of the memory an expanded array
    would and uploads once.
    """
    grid_x, grid_y, grid_z = resolve_grid(
        np.asarray(xs, dtype=np.float64).ravel(),
        np.asarray(ys, dtype=np.float64).ravel(),
        np.asarray(zs, dtype=np.float64).ravel(),
        opts,
    )
    ny, nx = grid_z.shape
    verts = np.ascontiguousarray(
        np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]), dtype=np.float32
    )
    # Corner indices of every cell, as (ny-1, nx-1) blocks.
    rows = np.arange(ny - 1)[:, None]
    cols = np.arange(nx - 1)[None, :]
    i00 = (rows * nx + cols).ravel()
    i10 = i00 + 1
    i01 = i00 + nx
    i11 = i01 + 1
    indices = np.empty(i00.size * 6, dtype=np.uint32)
    indices[0::6] = i00
    indices[1::6] = i10
    indices[2::6] = i11
    indices[3::6] = i00
    indices[4::6] = i11
    indices[5::6] = i01
    return verts, indices, grid_z.ravel()


def wireframe_geometry(xs, ys, zs, opts: Dict[str, Any]):
    """The grid's row and column lines, as ``GL_LINES`` pairs.

    ``stride`` thins the mesh: a 400x400 grid drawn at every line is 320k segments of
    solid ink, in which the surface is invisible. Every 4th line reads as a mesh.
    """
    grid_x, grid_y, grid_z = resolve_grid(
        np.asarray(xs, dtype=np.float64).ravel(),
        np.asarray(ys, dtype=np.float64).ravel(),
        np.asarray(zs, dtype=np.float64).ravel(),
        opts,
    )
    ny, nx = grid_z.shape
    stride = max(1, int(opts.get("stride", 1)))
    pts = np.stack([grid_x, grid_y, grid_z], axis=-1)

    segments: List[np.ndarray] = []
    # Rows: connect along x. Always include the last row so the mesh has a closed edge.
    row_idx = sorted(set(list(range(0, ny, stride)) + [ny - 1]))
    for j in row_idx:
        line = pts[j]
        seg = np.empty(((nx - 1) * 2, 3))
        seg[0::2] = line[:-1]
        seg[1::2] = line[1:]
        segments.append(seg)
    col_idx = sorted(set(list(range(0, nx, stride)) + [nx - 1]))
    for i in col_idx:
        line = pts[:, i]
        seg = np.empty(((ny - 1) * 2, 3))
        seg[0::2] = line[:-1]
        seg[1::2] = line[1:]
        segments.append(seg)

    if not segments:
        return np.zeros((0, 3), dtype=np.float32), None, np.zeros(0)
    verts = np.ascontiguousarray(np.concatenate(segments, axis=0), dtype=np.float32)
    return verts, None, verts[:, 2].astype(np.float64)


def _box_geometry(
    centres: np.ndarray, dx: float, dy: float, heights: np.ndarray, base: float
) -> np.ndarray:
    """Axis-aligned boxes as expanded ``GL_TRIANGLES`` — 36 vertices each.

    Expanded rather than indexed on purpose: each face wants its own colour when the bars
    are colour-mapped, and shared corners cannot carry two.
    """
    n = len(centres)
    hx, hy = dx * 0.5, dy * 0.5
    x0 = centres[:, 0] - hx
    x1 = centres[:, 0] + hx
    y0 = centres[:, 1] - hy
    y1 = centres[:, 1] + hy
    z0 = np.full(n, base, dtype=np.float64)
    z1 = heights

    # The eight corners of every box, each (n, 3).
    def corner(cx, cy, cz):
        return np.column_stack([cx, cy, cz])

    c000, c100 = corner(x0, y0, z0), corner(x1, y0, z0)
    c110, c010 = corner(x1, y1, z0), corner(x0, y1, z0)
    c001, c101 = corner(x0, y0, z1), corner(x1, y0, z1)
    c111, c011 = corner(x1, y1, z1), corner(x0, y1, z1)

    quads = [
        (c001, c101, c111, c011),  # top
        (c000, c010, c110, c100),  # bottom
        (c000, c100, c101, c001),  # -y
        (c010, c011, c111, c110),  # +y
        (c000, c001, c011, c010),  # -x
        (c100, c110, c111, c101),  # +x
    ]
    out = np.empty((n * 36, 3), dtype=np.float32)
    for face, (a, b, c, d) in enumerate(quads):
        block = face * 6
        for offset, corner_pts in enumerate((a, b, c, a, c, d)):
            out[block + offset :: 36] = corner_pts
    return out


def bar_geometry(xs, ys, zs, opts: Dict[str, Any]):
    """One solid box per row, standing on the ``z = baseline`` plane."""
    verts = _xyz(xs, ys, zs).astype(np.float64)
    n = len(verts)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32), None, np.zeros(0)
    base = float(opts.get("baseline", 0.0))
    dx = float(opts.get("bar_dx", 0.0)) or auto_spacing(verts[:, 0])
    dy = float(opts.get("bar_dy", 0.0)) or auto_spacing(verts[:, 1])
    gap = float(np.clip(float(opts.get("gap", 0.15)), 0.0, 0.9))
    dx *= 1.0 - gap
    dy *= 1.0 - gap
    boxes = _box_geometry(verts, dx, dy, verts[:, 2], base)
    # One colour value per box, repeated over its 36 vertices, so a colour-mapped bar
    # chart shades whole bars rather than gradient-filling each face.
    values = np.repeat(verts[:, 2], 36)
    return boxes, None, values


def auto_spacing(values: np.ndarray) -> float:
    """A sensible bar footprint: 80% of the median gap between distinct coordinates.

    The same rule :func:`glplot.gui.layerops.auto_bar_width` uses in 2D, so a 3D bar chart
    and its 2D projection have bars of the same relative width.
    """
    unique = np.unique(np.asarray(values, dtype=np.float64))
    if unique.size < 2:
        return 1.0
    gaps = np.diff(unique)
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return 1.0
    return float(np.median(gaps)) * 0.8


def quiver_geometry(xs, ys, zs, opts: Dict[str, Any]):
    """Arrows from ``(x, y, z)`` along ``(u, v, w)``, as ``GL_LINES``.

    The vector components arrive through ``options`` rather than as positional columns
    because the kind signature is ``(x, y, z)`` for every entry in the registry; a kind
    that needed six positional arrays would force every other one to grow three unused
    parameters. The Data panel fills them from three more column pickers.

    Each arrow is a shaft plus two head barbs, drawn in the plane most perpendicular to
    the shaft so the head is visible from any viewing angle.
    """
    verts = _xyz(xs, ys, zs).astype(np.float64)
    n = len(verts)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32), None, np.zeros(0)
    if n > MAX_ARROWS:
        raise ValueError(f"quiver3d is limited to {MAX_ARROWS} arrows; got {n}.")

    def component(name: str) -> np.ndarray:
        raw = opts.get(name)
        if raw is None:
            return np.zeros(n, dtype=np.float64)
        arr = np.asarray(raw, dtype=np.float64).ravel()
        if arr.size == 1:
            return np.full(n, float(arr[0]))
        if arr.size != n:
            raise ValueError(f"quiver3d '{name}' has {arr.size} values but there are {n} rows.")
        return arr

    uvw = np.column_stack([component("u"), component("v"), component("w")])
    scale = float(opts.get("scale", 1.0))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    uvw = uvw * scale

    tips = verts + uvw
    lengths = np.linalg.norm(uvw, axis=1)
    safe = np.where(lengths < 1e-12, 1.0, lengths)
    direction = uvw / safe[:, None]

    # A reference axis that is never parallel to the shaft, chosen per arrow so the cross
    # product cannot collapse: whichever world axis the shaft is least aligned with.
    ref = np.zeros_like(direction)
    least = np.argmin(np.abs(direction), axis=1)
    ref[np.arange(n), least] = 1.0
    side = np.cross(direction, ref)
    side_norm = np.linalg.norm(side, axis=1)
    side = side / np.where(side_norm < 1e-12, 1.0, side_norm)[:, None]

    head = float(np.clip(float(opts.get("head", 0.25)), 0.02, 0.9))
    back = tips - direction * (lengths * head)[:, None]
    spread = side * (lengths * head * 0.45)[:, None]
    barb_a = back + spread
    barb_b = back - spread

    out = np.empty((n * 6, 3), dtype=np.float32)
    out[0::6] = verts
    out[1::6] = tips
    out[2::6] = tips
    out[3::6] = barb_a
    out[4::6] = tips
    out[5::6] = barb_b
    return out, None, np.repeat(lengths, 6)


# ----------------------------------------------------------------------------------
# The registry
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class PlotKind3D:
    """One 3D plot type: how three columns become a :class:`Layer3D`.

    Attributes
    ----------
    key
        Stable identifier stored in ``metadata[KIND3D_METADATA_KEY]``.
    label
        What the picker shows.
    layer_type
        The ``layer_type`` the kind compiles to. Several kinds share one (``line3d``,
        ``stem3d`` and ``wireframe3d`` are all ``wireframe3d`` layers) — which is exactly
        why the kind tag exists and ``layer_type`` cannot be the identity.
    primitive
        ``"points"``, ``"lines"`` or ``"triangles"``: what the renderer draws.
    derived
        True when the vertices are *computed* from the columns rather than being them, so
        a bound dataset must re-derive instead of writing rows back over them. Only
        ``scatter3d`` and ``volume3d`` are row-for-row.
    icon
        An :data:`glplot.gui.icons.ICON_SHAPES` name.
    geom
        ``(xs, ys, zs, options) -> (vertices, indices|None, values|None)``. ``values`` is
        the per-vertex scalar a colormap is applied to when the layer is colour-mapped.
    needs_grid
        True when the kind requires a rectangular ``(x, y)`` lattice.
    description
        One paragraph shown in the picker's help, saying what the kind draws and what it
        cannot.
    """

    key: str
    label: str
    layer_type: str
    primitive: str
    derived: bool
    icon: str
    geom: Callable[[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]], Any]
    description: str
    needs_grid: bool = False


#: Default options per kind. A kind's options dict is validated against these keys, so an
#: unknown option is a caller bug rather than a silently ignored typo — the same contract
#: :func:`glplot.gui.layerops._resolve_options` enforces in 2D.
_KIND3D_DEFAULT_OPTIONS: Dict[str, Dict[str, Any]] = {
    "scatter3d": {},
    "line3d": {},
    "stem3d": {"baseline": 0.0},
    "ribbon3d": {"baseline": 0.0},
    "bar3d": {"baseline": 0.0, "bar_dx": 0.0, "bar_dy": 0.0, "gap": 0.15},
    # ``nu``/``nv`` name the sampling shape of a *parametric* surface, whose connectivity
    # is not recoverable from (x, y). 0 means "detect an (x, y) lattice" — see
    # :func:`resolve_grid`.
    "surface3d": {"nu": 0, "nv": 0},
    "wireframe3d": {"stride": 1, "nu": 0, "nv": 0},
    "volume3d": {},
    "quiver3d": {"u": None, "v": None, "w": None, "scale": 1.0, "head": 0.25},
}


_KIND3D_REGISTRY: Dict[str, PlotKind3D] = {
    "scatter3d": PlotKind3D(
        key="scatter3d",
        label="scatter 3D",
        layer_type="scatter3d",
        primitive="points",
        derived=False,
        icon="chart_scatter3d",
        geom=points_geometry,
        description="One point per row. The only 3D kind whose vertices are the table's "
        "rows, so it is the one a cell edit can write straight back to.",
    ),
    "line3d": PlotKind3D(
        key="line3d",
        label="line 3D (path)",
        layer_type="wireframe3d",
        primitive="lines",
        derived=True,
        icon="chart_line3d",
        geom=path_geometry,
        description="Connect the rows in table order — a trajectory through space. The "
        "row order is the curve, so sort the table if the rows are not already in path "
        "order.",
    ),
    "stem3d": PlotKind3D(
        key="stem3d",
        label="stems 3D",
        layer_type="wireframe3d",
        primitive="lines",
        derived=True,
        icon="chart_stem3d",
        geom=stem_geometry,
        description="A vertical drop from each point to the baseline plane. Reads the "
        "height of scattered samples far better than bare points, which give the eye no "
        "reference for depth.",
    ),
    "ribbon3d": PlotKind3D(
        key="ribbon3d",
        label="ribbon 3D",
        layer_type="mesh3d",
        primitive="triangles",
        derived=True,
        icon="chart_ribbon3d",
        geom=ribbon_geometry,
        description="A filled band hanging from the path to the baseline — fill_between "
        "in 3D. The surface gives the trajectory a depth cue the line alone cannot.",
    ),
    "bar3d": PlotKind3D(
        key="bar3d",
        label="bars 3D",
        layer_type="mesh3d",
        primitive="triangles",
        derived=True,
        icon="chart_bar3d",
        geom=bar_geometry,
        description="A solid box per row standing on the baseline plane. Footprint 0 = "
        "auto (80% of the median spacing on that axis). Best on gridded (x, y); on "
        "scattered points the boxes overlap.",
    ),
    "surface3d": PlotKind3D(
        key="surface3d",
        label="surface (needs a grid)",
        layer_type="mesh3d",
        primitive="triangles",
        derived=True,
        icon="chart_surface3d",
        geom=surface_geometry,
        needs_grid=True,
        description="Triangulate z over a rectangular (x, y) lattice. The x/y columns "
        "must be a full grid — every combination present exactly once. Build one in the "
        "Functions panel with z = f(x, y), or use 'scatter 3D' for scattered points.",
    ),
    "wireframe3d": PlotKind3D(
        key="wireframe3d",
        label="wireframe (needs a grid)",
        layer_type="wireframe3d",
        primitive="lines",
        derived=True,
        icon="chart_wireframe3d",
        geom=wireframe_geometry,
        needs_grid=True,
        description="The grid's row and column lines. 'Stride' thins the mesh: on a fine "
        "grid every line is a sheet of solid ink, and every 4th reads as a surface.",
    ),
    "volume3d": PlotKind3D(
        key="volume3d",
        label="volumetric cloud",
        layer_type="volume3d",
        primitive="points",
        derived=False,
        icon="chart_volume3d",
        geom=points_geometry,
        description="The same points as 'scatter 3D', drawn as a translucent cloud: with "
        "a low alpha the density of overlapping samples becomes the picture. Millions of "
        "points stay interactive.",
    ),
    "quiver3d": PlotKind3D(
        key="quiver3d",
        label="vector field (quiver)",
        layer_type="wireframe3d",
        primitive="lines",
        derived=True,
        icon="chart_quiver3d",
        geom=quiver_geometry,
        description="An arrow at each point along (u, v, w), taken from three more "
        "columns. 'Scale' multiplies every vector; 'head' is the barb length as a "
        "fraction of the arrow.",
    ),
}

#: Every 3D kind key, in picker order.
KIND3D_KEYS: Tuple[str, ...] = tuple(_KIND3D_REGISTRY)

#: The kinds whose vertices line up row-for-row with the table, so a bound dataset can
#: write cell edits straight back onto them.
DIRECT_KIND3D_KEYS: Tuple[str, ...] = tuple(
    k for k, spec in _KIND3D_REGISTRY.items() if not spec.derived
)


def kind3d_spec(kind: str) -> PlotKind3D:
    """The :class:`PlotKind3D` for ``kind``. Raises ValueError on an unknown key."""
    spec = _KIND3D_REGISTRY.get(str(kind).strip().lower())
    if spec is None:
        raise ValueError(f"kind must be one of {', '.join(KIND3D_KEYS)}, got {kind!r}")
    return spec


def is_kind3d(kind: Optional[str]) -> bool:
    """True when ``kind`` names a 3D kind. Used to route a tag to the right registry."""
    return kind is not None and str(kind).strip().lower() in _KIND3D_REGISTRY


def kind3d_is_derived(kind: Optional[str]) -> bool:
    """True when ``kind``'s vertices are computed rather than the columns themselves.

    Unknown kinds answer True: refusing to write back is the safe side of the bet, the
    same rule :func:`glplot.gui.layerops.kind_is_derived` follows.
    """
    if kind is None:
        return True
    spec = _KIND3D_REGISTRY.get(str(kind).strip().lower())
    return True if spec is None else spec.derived


def default_kind3d_options(kind: str) -> Dict[str, Any]:
    """A fresh, mutable options dict for ``kind`` — the panel's starting state."""
    return dict(_KIND3D_DEFAULT_OPTIONS.get(str(kind).strip().lower(), {}))


def resolve_kind3d_options(kind: str, options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge ``options`` over the kind's defaults, rejecting keys it has no use for."""
    resolved = default_kind3d_options(kind)
    if not options:
        return resolved
    unknown = [key for key in options if key not in resolved]
    if unknown:
        raise ValueError(
            f"kind {kind!r} has no option(s) {', '.join(sorted(unknown))}; "
            f"it accepts {', '.join(sorted(resolved)) or '(none)'}"
        )
    resolved.update(options)
    return resolved


def layer_kind3d(layer: Any) -> Optional[str]:
    """The 3D kind ``layer`` was built with, or None if it is not a tagged 3D layer."""
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    tagged = metadata.get(KIND3D_METADATA_KEY)
    return str(tagged) if is_kind3d(tagged) else None


def layer_kind3d_options(layer: Any) -> Dict[str, Any]:
    """The options ``layer`` was last built with, merged over its kind's defaults."""
    kind = layer_kind3d(layer)
    if kind is None:
        return {}
    stored = getattr(layer, "metadata", {}).get("gui_kind_options")
    resolved = default_kind3d_options(kind)
    if isinstance(stored, dict):
        resolved.update({k: v for k, v in stored.items() if k in resolved})
    return resolved


def tag_layer_kind3d(
    layer: Any,
    kind: str,
    *,
    options: Optional[Dict[str, Any]] = None,
    source_xyz: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> None:
    """Record on ``layer`` which kind built it, with what options and from what columns.

    ``source_xyz`` is retained only for *derived* kinds, and is what lets the Style panel
    switch a surface to a wireframe without going back to the dataset: the layer's own
    vertices are the triangulation, not the samples it came from.
    """
    spec = kind3d_spec(kind)
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        return
    metadata[KIND3D_METADATA_KEY] = spec.key
    metadata["gui_kind_options"] = dict(options or {})
    if source_xyz is not None:
        metadata["gui_source_xyz"] = tuple(
            np.asarray(a, dtype=np.float64).ravel().copy() for a in source_xyz
        )
    else:
        metadata.pop("gui_source_xyz", None)


def stretch_to_vertices(raw: Any, n_verts: int) -> Optional[np.ndarray]:
    """Spread a per-**row** column over a kind's per-**vertex** array.

    Every derived 3D kind emits more vertices than the table has rows, and each does it
    differently: a path emits ``2(N-1)``, a ribbon ``6(N-1)``, a bar chart ``36N``, a
    stem plot ``2N``. A colour column is one value per row, so it has to be spread over
    whichever of those the kind produced.

    Three cases, most exact first:

    1. lengths already agree — use it as is;
    2. an exact multiple (stems, bars) — :func:`numpy.repeat`, so each row's value covers
       exactly its own vertices;
    3. anything else (paths, ribbons) — resample along the normalised row parameter. The
       geometry walks the rows in order, so a linear resample lands each vertex between
       its two neighbouring rows' values, which is the gradient the renderer would
       interpolate anyway.

    Returns None only when there is nothing to map. The previous code did integer division
    and, when it did not divide exactly, silently fell through to a *flat* colour — so
    ``line3d`` with ``c=t`` drew a single-coloured curve and the colour column vanished
    with no error. That is the failure this function exists to remove.
    """
    values = np.asarray(raw, dtype=np.float64).ravel()
    n = int(values.size)
    n_verts = int(n_verts)
    if n == 0 or n_verts <= 0:
        return None
    if n == n_verts:
        return values
    if n_verts % n == 0:
        return np.repeat(values, n_verts // n)
    source = np.linspace(0.0, 1.0, n)
    target = np.linspace(0.0, 1.0, n_verts)
    return np.interp(target, source, values)


def add_xyz_layer(
    plot: Any,
    x: Any,
    y: Any,
    z: Any,
    *,
    kind: str,
    label: str,
    color: Optional[Sequence[float]] = None,
    width: float = 1.5,
    size: float = 3.0,
    alpha: float = 1.0,
    options: Optional[Dict[str, Any]] = None,
    c: Optional[Any] = None,
    s: Optional[Any] = None,
    cmap: str = "viridis",
    colormap_by_z: bool = True,
) -> Any:
    """Build a 3D layer of ``kind`` from three columns and return it, tagged with its kind.

    The 3D twin of :func:`glplot.gui.layerops.add_xy_layer`, and it follows the same
    rules: it routes through the engine's public ``add_*`` API (CONTRACT §1.5), it always
    demands an explicit ``label`` (the engine's index-based default renames itself as the
    scene changes and cannot be used to find the layer again), and it tags the result so
    the layer can be re-derived and re-typed later.

    Colour resolves in three steps, most specific first:

    1. an explicit ``c`` column, mapped through ``cmap`` — colour as a data dimension;
    2. otherwise the kind's own scalar (usually z), when ``colormap_by_z`` is set — which
       is what makes a surface read as a height field instead of a flat sheet;
    3. otherwise the flat ``color``.

    **Queued commands only.** Creating a layer mutates the scene, so this must run from
    inside a queued command, never a draw callback.
    """
    from .layerops import colormap_colors, mark_scene_dirty

    spec = kind3d_spec(kind)
    if not label or not str(label).strip():
        raise ValueError("label is required: the engine's index-based default is unstable")

    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    zs = np.asarray(z, dtype=np.float64).ravel()
    if not (xs.size == ys.size == zs.size):
        raise ValueError(f"x, y and z must be the same length; got {xs.size}, {ys.size}, {zs.size}")

    opts = resolve_kind3d_options(spec.key, options)
    vertices, indices, values = spec.geom(xs, ys, zs, opts)
    n_verts = len(vertices)

    # -- colours ----------------------------------------------------------------
    cvalues: Optional[np.ndarray] = None
    if c is not None:
        raw = np.asarray(c, dtype=np.float64).ravel()
        per_vertex = stretch_to_vertices(raw, n_verts)
        if per_vertex is not None:
            colors = colormap_colors(per_vertex, str(cmap))
            cvalues = per_vertex
        else:
            colors = None
    elif colormap_by_z and values is not None and len(values) == n_verts:
        colors = colormap_colors(np.asarray(values, dtype=np.float64), str(cmap))
        cvalues = np.asarray(values, dtype=np.float64)
    else:
        colors = None

    if colors is None:
        rgba = _normalize_rgba3d(color)
        colors = np.tile(np.asarray(rgba, dtype=np.float32), (n_verts, 1))
    colors = np.ascontiguousarray(colors, dtype=np.float32)
    if alpha != 1.0:
        colors = colors.copy()
        colors[:, 3] *= float(np.clip(alpha, 0.0, 1.0))

    # -- per-point sizes --------------------------------------------------------
    sizes_arr = None
    if s is not None and spec.primitive == "points":
        raw = np.asarray(s, dtype=np.float32).ravel()
        if raw.size == n_verts:
            sizes_arr = raw

    layer = plot.add_geometry3d(
        vertices,
        colors,
        indices=indices,
        primitive=spec.primitive,
        layer_type=spec.layer_type,
        label=str(label),
        point_size=float(size),
        line_width=float(width),
        sizes=sizes_arr,
    )

    if cvalues is not None:
        layer.metadata["cvalues"] = cvalues
        layer.metadata["cmap"] = str(cmap)
        layer.style.use_colormap = True
        layer.style.cmap = str(cmap)
    if sizes_arr is not None:
        layer.metadata["svalues"] = sizes_arr
    tag_layer_kind3d(layer, spec.key, options=opts, source_xyz=(xs, ys, zs))
    mark_scene_dirty(plot)
    return layer


def _normalize_rgba3d(color: Optional[Sequence[float]]) -> Tuple[float, float, float, float]:
    """Coerce a colour to RGBA, defaulting to the renderer's own fallback blue.

    That default is deliberately ``geometry3d.py``'s ``(0.1, 0.45, 1.0, 1.0)`` rather than
    black: a 3D layer with no colours falls through to it at upload time, so matching it
    means an explicitly-defaulted layer and an implicitly-defaulted one look the same.
    """
    if color is None:
        return (0.1, 0.45, 1.0, 1.0)
    values = [float(v) for v in color]
    if len(values) == 3:
        values.append(1.0)
    if len(values) != 4:
        raise ValueError(f"color must be RGB or RGBA, got {len(values)} components")
    return (values[0], values[1], values[2], values[3])


def update_layer_xyz(
    plot: Any,
    layer: Any,
    x: Any,
    y: Any,
    z: Any,
    *,
    options: Optional[Dict[str, Any]] = None,
) -> None:
    """Re-derive ``layer``'s geometry from new columns, keeping its kind and styling.

    The in-place counterpart to :func:`add_xyz_layer`, and it calls the *same* ``geom``, so
    a surface redrawn after a cell edit is triangulated exactly as it was when it was
    created. Raises ValueError when the layer carries no 3D kind tag — geometry this
    module did not build must not be overwritten by it.

    **Queued commands only.**
    """
    from .layerops import colormap_colors, mark_scene_dirty

    kind = layer_kind3d(layer)
    if kind is None:
        raise ValueError("layer has no 3D kind tag; its geometry cannot be re-derived")
    spec = kind3d_spec(kind)

    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    zs = np.asarray(z, dtype=np.float64).ravel()
    opts = resolve_kind3d_options(
        kind, options if options is not None else layer_kind3d_options(layer)
    )
    vertices, indices, values = spec.geom(xs, ys, zs, opts)

    layer.vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    layer.indices = None if indices is None else np.ascontiguousarray(indices, dtype=np.uint32)

    # Re-map the colours only when they *were* mapped. A layer the user has since given a
    # flat colour must keep it: re-deriving the geometry is not a request to restyle.
    cmap = layer.metadata.get("cmap")
    if cmap and values is not None and len(values) == len(layer.vertices):
        layer.colors = colormap_colors(np.asarray(values, dtype=np.float64), str(cmap))
        layer.metadata["cvalues"] = np.asarray(values, dtype=np.float64)
    elif layer.colors is not None and len(layer.colors) != len(layer.vertices):
        # The vertex count changed and the old colour array no longer fits. Collapse to the
        # first colour rather than leaving a mismatched buffer, which uploads as garbage.
        layer.colors = np.tile(layer.colors[0], (len(layer.vertices), 1)).astype(np.float32)

    layer.dirty.gpu_dirty = True
    layer.dirty.bounds_dirty = True
    tag_layer_kind3d(layer, kind, options=opts, source_xyz=(xs, ys, zs))
    # The bounds moved, so the shared box and the auto-fit distance have to follow.
    plot.set_3d_view()
    mark_scene_dirty(plot)


def layer_source_xyz(layer: Any) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The ``(x, y, z)`` a derived 3D layer was built from, or its vertices for a direct one.

    None for a 3D layer this module did not build (one from ``pyplot.plot_surface``, say),
    which is the signal that its kind cannot be changed in place.
    """
    stored = getattr(layer, "metadata", {}).get("gui_source_xyz")
    if isinstance(stored, tuple) and len(stored) == 3:
        return tuple(np.asarray(a, dtype=np.float64) for a in stored)  # type: ignore[return-value]
    kind = layer_kind3d(layer)
    if kind is None or kind3d_is_derived(kind):
        return None
    verts = getattr(layer, "vertices", None)
    if verts is None:
        return None
    verts = np.asarray(verts, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        return None
    return verts[:, 0].copy(), verts[:, 1].copy(), verts[:, 2].copy()
