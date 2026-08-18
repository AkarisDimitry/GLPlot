"""The 3D objects panel: pick a surface or a space curve, tune it, plot it.

A thin, stateful shell over :mod:`glplot.gui.generators3d`, the way the Dynamics panel is a
shell over :mod:`glplot.gui.dynamics`. It owns nothing but the current selection, the
parameter values and the sample count; every array comes from a generator's pure ``build``.

What this is for
----------------
An empty 3D figure is a harder starting point than an empty 2D one. In 2D the user types
``sin(x)`` and has a curve. In 3D the objects worth looking at are parametrised in two
variables at once — a torus is three coupled expressions in ``(u, v)`` — so "type your
surface" is a wall rather than a feature. Twenty-seven named objects with sliders is what
makes the 3D mode usable on the first click instead of the tenth.

Everything it produces is an **ordinary dataset**. It lands in the shared
:class:`~glplot.gui.datasets.DataStore` exactly like a pasted CSV, so it can be edited cell
by cell, filtered, transformed and re-plotted with any other kind. The catalogue is a
source of data, not a special kind of it.

**Change-gated preview.** ``draw`` runs every frame, and regenerating a 20k-point surface
there would make every slider drag unusable. The preview is capped to
:data:`PREVIEW_SAMPLES` and only rebuilt when a signature of the inputs changes; Plot and
Create-dataset regenerate at the full requested resolution. This is the same discipline
the Dynamics and Functions panels use, for the same reason.

**The thumbnail.** That capped table used to be *reported* -- "preview: 3969 rows" -- which
told the user nothing about what a torus knot with q=7 looks like, and picking an object
out of twenty-seven by name and tooltip is guesswork. :meth:`_draw_preview` draws it
instead: a real 3-D thumbnail through :func:`glplot.gui.widgets.mini_scene3d`, spinning by
default and draggable to orbit, rebuilt only when the same signature changes.

It is built by running *the plot kind's own* ``geom`` function
(:mod:`glplot.gui.layerops3d`) over a decimated slice of the preview table and colouring
the result through :func:`glplot.gui.layerops.colormap_colors` -- the same two calls
:func:`~glplot.gui.layerops3d.add_xyz_layer` makes. A rebuilt-from-scratch approximation
would drift from the thing it claims to predict on the first change to either; sharing the
functions means "the thumbnail is wrong" and "the plot is wrong" cannot be different bugs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...core.camera3d import Camera3D
from .. import generators3d, layerops, layerops3d, widgets
from ..datasets import Column, DataSet
from ..generators3d import CATEGORIES, Generator3D, GeneratorError
from ..history import Command
from ..icons import icon_button
from .base import Panel

try:
    from imgui_bundle import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - exercised only on GL-less systems
    IMGUI_AVAILABLE = False
    imgui = None


#: The preview never generates more than this many rows regardless of the requested count,
#: so dragging a parameter stays interactive with a million points queued behind it.
PREVIEW_SAMPLES = 4000

#: How many generator buttons fit on a row in the picker grid.
PICKER_COLUMNS = 4

#: Height of the 3D thumbnail in pixels. Tall enough that a torus knot is legible, short
#: enough that the parameter sliders stay on screen under it in a docked panel.
PREVIEW_HEIGHT = 190.0

#: Degrees of azimuth the thumbnail turns per second while spinning. Slow: the rotation is
#: there to disambiguate depth, and anything brisk enough to notice is fast enough to
#: fight the user while they read a slider.
PREVIEW_SPIN_DEG_PER_S = 22.0

#: Rows of the preview table the thumbnail keeps, per plot kind. The numbers differ
#: because the kinds *derive* wildly different vertex counts from the same rows -- a path
#: emits 2(N-1) vertices, a ribbon 6(N-1), a bar chart 36N, a quiver 6N -- so a single row
#: budget would leave a scatter starved and a bar chart over the widget's caps. Each is
#: the row count whose derived geometry lands just under the matching
#: ``widgets.MAX_SCENE3D_*`` ceiling.
PREVIEW_ROWS: Dict[str, int] = {
    "scatter3d": 2000,
    "volume3d": 2000,
    "line3d": 1600,
    "stem3d": 900,
    "ribbon3d": 700,
    "bar3d": 140,
    "quiver3d": 380,
}
PREVIEW_ROWS_DEFAULT = 1200

#: Per-axis samples kept from a surface generator's ``(u, v)`` lattice. 30 x 30 is 1682
#: triangles or 1740 wireframe segments, both inside the widget's caps, and it is the
#: coarseness at which a sphere still reads as a sphere.
PREVIEW_GRID_SIDE = 30

#: Floor under the alpha the thumbnail draws with. The Appearance alpha is applied so the
#: preview predicts the result, but a ``volume3d`` cloud at alpha 0.02 works by *stacking*
#: a million translucent points, and the thumbnail has two thousand -- reproducing the
#: number literally would show an empty box for a plot that comes out perfectly visible.
PREVIEW_MIN_ALPHA = 0.18

#: Fallback colour for geometry with no colour-mappable scalar. Deliberately the same
#: literal :func:`glplot.gui.layerops3d._normalize_rgba3d` falls back to, so an uncoloured
#: preview and the layer it predicts are the same blue.
PREVIEW_FALLBACK_RGBA = (0.1, 0.45, 1.0, 1.0)

#: Sample-count slider bounds. The ceiling is well below
#: :data:`generators3d.MAX_SAMPLES` because this is a slider, not a text field — a
#: hundred million is reachable by typing into the Data panel, not by an accidental drag.
MIN_SAMPLES_UI = 64
MAX_SAMPLES_UI = 2_000_000

_HELP_SAMPLES = (
    "Rows generated. A surface samples the square root of this per axis, so 40 000 is a "
    "200x200 mesh. Curves and clouds use it directly."
)

_HELP_KIND = (
    "How the columns are drawn. Each object has a natural default — a surface generator "
    "asks for 'surface', a curve for 'line 3D' — but any kind that accepts three columns "
    "will work, so a sphere can be drawn as a point cloud and a helix as stems."
)

_HELP_COLOR = (
    "The column mapped through the colormap. Most objects carry a natural one (a radius, "
    "a cluster index, the path parameter); with none, z is used, which is what makes a "
    "surface read as a height field."
)


class Objects3DPanel(Panel):
    """Generate 3D surfaces, curves, clouds, lattices and vector fields."""

    title = "Objects 3D"
    icon = "chart_surface3d"
    default_open = False

    def __init__(self, ws: Any) -> None:
        super().__init__(ws)
        self.category: str = CATEGORIES[0][0]
        self.key: str = generators3d.GENERATOR_KEYS[0]
        #: generator key -> its live parameter values, so switching object and coming back
        #: restores what the user dialled in rather than resetting to the defaults.
        self.params: Dict[str, Dict[str, float]] = {}
        self.samples: int = 20_000
        #: Plot kind override, or None to use the generator's own default.
        self.kind: Optional[str] = None
        self.cmap: str = "viridis"
        self.point_size: float = 2.5
        self.line_width: float = 1.5
        self.alpha: float = 1.0
        #: Colour-column override chosen in Appearance, or None to use the generator's own.
        self._color_override: Optional[str] = None
        self._status: str = ""
        self._error: str = ""

        #: Cached preview table and the signature it was built from.
        self._preview: Optional[Dict[str, np.ndarray]] = None
        self._preview_sig: Optional[tuple] = None

        #: The thumbnail's own camera. Independent of the figure's on purpose: orbiting a
        #: preview is a question about the preview, and having it swing the real 3D scene
        #: under the user's plots would make the panel unusable next to existing work.
        self.preview_camera: Camera3D = Camera3D()
        #: Spin the thumbnail. On by default: one still frame of a projected wireframe is
        #: ambiguous about depth, and half a turn resolves it without the user doing
        #: anything. The first drag turns it off (see :meth:`_draw_preview`).
        self.preview_spin: bool = True
        #: True once the user has orbited the thumbnail by hand, which is what makes the
        #: angle worth carrying to the plotted layer. Without this flag Plot would inherit
        #: whatever azimuth the *spin* happened to be passing through, which is noise.
        self._preview_oriented: bool = False
        #: Cached drawable geometry (vertices, colours, primitives) and its signature.
        self._drawable: Optional[Dict[str, Any]] = None
        self._drawable_sig: Optional[tuple] = None
        #: Why the thumbnail is showing something other than the chosen kind, if it is.
        self._preview_note: str = ""

    # -- state -----------------------------------------------------------------

    def spec(self) -> Generator3D:
        """The selected generator."""
        return generators3d.generator(self.key)

    def current_params(self) -> Dict[str, float]:
        """The selected generator's live parameters, seeded from its defaults once."""
        if self.key not in self.params:
            self.params[self.key] = self.spec().defaults()
        return self.params[self.key]

    def current_kind(self) -> str:
        """The plot kind to build with: the override, or the generator's default."""
        return self.kind or self.spec().kind

    def _signature(self) -> tuple:
        """Everything the preview depends on. Any change rebuilds it, nothing else does."""
        params = self.current_params()
        return (self.key, tuple(sorted((k, float(v)) for k, v in params.items())))

    def _refresh_preview(self) -> None:
        """Rebuild the capped preview table if any input changed."""
        signature = self._signature()
        if signature == self._preview_sig and self._preview is not None:
            return
        self._preview_sig = signature
        try:
            self._preview = self.spec().generate(self.current_params(), PREVIEW_SAMPLES)
            self._error = ""
        except (GeneratorError, ValueError, ArithmeticError) as exc:
            # Keep the last good preview on screen; an out-of-range parameter should show
            # its message, not blank the panel.
            self._error = str(exc)

    # -- thumbnail geometry ----------------------------------------------------

    def _drawable_signature(self) -> tuple:
        """Everything the thumbnail's *geometry* depends on.

        A superset of :meth:`_signature`: the table changes with the object and its
        parameters, but the drawable also changes with the plot kind, the colormap, the
        colour column and the alpha. Camera angle is deliberately absent -- orbiting
        re-projects the cached geometry and must not rebuild it, or the drag would run the
        generator sixty times a second.
        """
        return (
            self._signature(),
            self.current_kind(),
            self.cmap,
            self._color_override,
            round(float(self.alpha), 4),
        )

    def _preview_rows(
        self, kind: str, spec: Generator3D, n: int
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        """Which of the ``n`` preview rows the thumbnail keeps, and the grid shape left.

        Two regimes. A gridded kind has to keep the *lattice*: dropping arbitrary rows from
        a surface would leave ``nu * nv != N`` and :func:`~glplot.gui.layerops3d.resolve_grid`
        would reject it, so the sub-sample is taken along u and v separately and the
        reduced shape is returned with it. Everything else strides evenly, which for a
        cloud is an unbiased sub-sample and for a path is a coarser path.
        """
        if kind in ("surface3d", "wireframe3d"):
            shape = spec.grid_shape(PREVIEW_SAMPLES)
            if shape is not None and shape[0] * shape[1] == n:
                nu, nv = shape
                iu = np.unique(np.linspace(0, nu - 1, min(nu, PREVIEW_GRID_SIDE)).astype(np.int64))
                iv = np.unique(np.linspace(0, nv - 1, min(nv, PREVIEW_GRID_SIDE)).astype(np.int64))
                # Row-major with u along the columns -- the order every generator emits and
                # the order `resolve_grid` reshapes with.
                return (iv[:, None] * nu + iu[None, :]).ravel(), (int(iu.size), int(iv.size))
            # Not a lattice. The kind will raise and `_build_drawable` falls back to
            # points, so fall through to the stride below rather than keeping every row --
            # a fallback that lands over the widget's cap is still a fallback that is too
            # big to draw.

        budget = PREVIEW_ROWS.get(kind, PREVIEW_ROWS_DEFAULT)
        if n <= budget:
            return np.arange(n, dtype=np.int64), None
        return np.unique(np.linspace(0, n - 1, budget).astype(np.int64)), None

    def _preview_colors(self, values: Optional[np.ndarray], n_verts: int) -> np.ndarray:
        """Per-vertex RGBA for ``n_verts`` vertices, by the layer builder's own rules.

        The three steps and their order are :func:`~glplot.gui.layerops3d.add_xyz_layer`'s:
        an explicit colour column stretched over the derived vertices, else the kind's own
        scalar (usually z), else the flat fallback. Same functions, same result.
        """
        per_vertex: Optional[np.ndarray] = None
        if values is not None:
            per_vertex = layerops3d.stretch_to_vertices(values, n_verts)
        if per_vertex is None:
            return np.tile(np.asarray(PREVIEW_FALLBACK_RGBA, dtype=np.float32), (n_verts, 1))
        return layerops.colormap_colors(per_vertex, self.cmap)

    def _build_drawable(self) -> Dict[str, Any]:
        """Vertices, colours and primitive indices for the thumbnail. Never raises.

        The vertex array and its primitives come from the plot kind's own ``geom``, so
        picking "stems 3D" for a helix previews stems and picking "surface" previews a
        triangulated surface. When the kind cannot accept this object -- asking for a
        surface over a space curve, which has no lattice -- the fallback is the object's
        points plus a note, rather than a blank box: "I cannot draw that" is less useful
        than "here is where it is, and here is why it will not plot".
        """
        table = self._preview or {}
        spec = self.spec()
        kind = self.current_kind()
        cx, cy, cz = spec.plot_columns
        xs = np.asarray(table.get(cx, ()), dtype=np.float64)
        ys = np.asarray(table.get(cy, ()), dtype=np.float64)
        zs = np.asarray(table.get(cz, ()), dtype=np.float64)
        note = ""

        if xs.size == 0 or ys.size != xs.size or zs.size != xs.size:
            empty = np.zeros((0, 3), dtype=np.float64)
            return {"points": empty, "colors": None, "dots": True, "note": note}

        rows, shape = self._preview_rows(kind, spec, int(xs.size))
        colour_values = self._color_values(spec, table)
        if colour_values is not None:
            colour_values = np.asarray(colour_values, dtype=np.float64)[rows]

        options: Dict[str, Any] = {}
        if shape is not None:
            options["nu"], options["nv"] = shape
        if kind == "quiver3d":
            for name in ("u", "v", "w"):
                if name in table:
                    options[name] = np.asarray(table[name], dtype=np.float64)[rows]

        px, py, pz = xs[rows], ys[rows], zs[rows]
        # Resolved outside the try so the failure path cannot fail the same way twice.
        kind3d = layerops3d.kind3d_spec(kind)
        try:
            opts = layerops3d.resolve_kind3d_options(kind, options or None)
            vertices, indices, values = kind3d.geom(px, py, pz, opts)
            primitive = kind3d.primitive
        except (ValueError, ArithmeticError, IndexError) as exc:
            vertices = np.column_stack([px, py, pz])
            indices, values, primitive = None, pz.astype(np.float64), "points"
            note = f"{kind3d.label}: {exc} Showing the points instead."

        vertices = np.asarray(vertices, dtype=np.float64)
        n_verts = int(len(vertices))
        source = colour_values if colour_values is not None else values
        colors = np.array(self._preview_colors(source, n_verts), dtype=np.float32, copy=True)
        if colors.shape[0] == n_verts and n_verts:
            colors[:, 3] *= max(float(self.alpha), PREVIEW_MIN_ALPHA)

        segments: Optional[np.ndarray] = None
        triangles: Optional[np.ndarray] = None
        if primitive == "lines":
            # Every ``lines`` kind emits GL_LINES: consecutive vertex pairs, one segment
            # each. That is the renderer's contract, not an assumption about the shape.
            segments = np.arange((n_verts // 2) * 2, dtype=np.int64).reshape(-1, 2)
        elif primitive == "triangles":
            if indices is not None:
                triangles = np.asarray(indices, dtype=np.int64).ravel()
                triangles = triangles[: (triangles.size // 3) * 3].reshape(-1, 3)
            else:
                triangles = np.arange((n_verts // 3) * 3, dtype=np.int64).reshape(-1, 3)

        return {
            "points": vertices,
            "colors": colors,
            "segments": segments,
            "triangles": triangles,
            "dots": primitive == "points",
            "note": note,
        }

    def _refresh_drawable(self) -> None:
        """Rebuild the thumbnail geometry if any input it depends on changed.

        Change-gated for the same reason the table is, and one step further: the geometry
        step runs a triangulation and a matplotlib colormap lookup over a few thousand
        vertices, which is affordable per *edit* and not per frame.
        """
        signature = self._drawable_signature()
        if signature == self._drawable_sig and self._drawable is not None:
            return
        self._drawable_sig = signature
        self._drawable = self._build_drawable()
        self._preview_note = str(self._drawable.get("note", ""))

    # -- frame -----------------------------------------------------------------

    def draw(self) -> None:
        if not IMGUI_AVAILABLE:
            return
        self._refresh_preview()
        self._refresh_drawable()

        self._draw_category_tabs()
        self._draw_picker()
        imgui.separator()
        self._draw_description()
        if widgets.section("Preview", default_open=True):
            self._draw_preview()
        if widgets.section("Parameters", default_open=True):
            self._draw_params()
        if widgets.section("Appearance", default_open=False):
            self._draw_appearance()
        imgui.separator()
        self._draw_actions()
        if self._error:
            widgets.error_box(self._error)
        elif self._status:
            imgui.text_disabled(self._status)

    def _draw_category_tabs(self) -> None:
        if not imgui.begin_tab_bar("##obj3d_cats"):
            return
        for key, label in CATEGORIES:
            selected, _ = imgui.begin_tab_item(label)
            if selected:
                if self.category != key:
                    self.category = key
                    members = generators3d.by_category(key)
                    if members:
                        self._select(members[0].key)
                imgui.end_tab_item()
        imgui.end_tab_bar()

    def _select(self, key: str) -> None:
        """Switch the selected object, dropping any stale plot-kind override."""
        if key == self.key:
            return
        self.key = key
        # The override is per-object by nature: "draw it as stems" makes sense for a curve
        # and not for a vector field, so carrying it across a switch would produce a kind
        # the new object cannot satisfy.
        self.kind = None
        self._status = ""
        self._error = ""

    def _draw_picker(self) -> None:
        """A grid of icon buttons, one per generator in the current category."""
        members = generators3d.by_category(self.category)
        for index, spec in enumerate(members):
            if index % PICKER_COLUMNS:
                imgui.same_line()
            if icon_button(
                f"##gen_{spec.key}",
                spec.icon,
                size=30.0,
                tooltip=f"{spec.label}\n{spec.description}",
                active=spec.key == self.key,
            ):
                self._select(spec.key)

    def _draw_description(self) -> None:
        spec = self.spec()
        imgui.text(spec.label)
        imgui.text_wrapped(spec.description)
        rows = 0 if self._preview is None else len(next(iter(self._preview.values()), ()))
        imgui.text_disabled(f"columns: {', '.join(spec.columns)}   preview: {rows} rows")

    def _draw_preview(self) -> None:
        """The 3D thumbnail and its two controls.

        The spin is advanced here rather than by a timer because a panel body is the only
        place with a frame delta to advance it *by*; ``io.delta_time`` is what the engine's
        own :meth:`~glplot.core.camera3d.Camera3D.advance_spin` runs on too.
        """
        drawable = self._drawable or {}
        camera = self.preview_camera
        if self.preview_spin:
            camera.orbit(PREVIEW_SPIN_DEG_PER_S * float(imgui.get_io().delta_time), 0.0)

        orbited = widgets.mini_scene3d(
            "obj3d_preview",
            drawable.get("points"),
            camera=camera,
            colors=drawable.get("colors"),
            segments=drawable.get("segments"),
            triangles=drawable.get("triangles"),
            dots=bool(drawable.get("dots")),
            height=PREVIEW_HEIGHT,
            label=self.spec().label,
            hint="drag to orbit",
            # The Appearance point size is a diameter in pixels; the thumbnail wants a
            # radius, and clamps it because a 20 px dot on a 190 px box is a solid fill.
            point_radius=float(np.clip(self.point_size * 0.5, 1.0, 4.0)),
            thickness=float(np.clip(self.line_width, 0.75, 3.0)),
        )
        if orbited:
            # A deliberate orientation beats a decorative one: stop spinning, and remember
            # that this angle is the user's, so Plot can adopt it.
            self.preview_spin = False
            self._preview_oriented = True

        changed, value = imgui.checkbox("Spin", self.preview_spin)
        if changed:
            self.preview_spin = bool(value)
        imgui.same_line()
        if imgui.button("Reset view"):
            camera.reset()
            self.preview_spin = True
            self._preview_oriented = False
        imgui.same_line()
        widgets.help_marker(
            "A CPU-projected thumbnail of the object as the chosen plot type will draw "
            "it, in the chosen colormap. Drag inside it to orbit; once you do, Plot "
            "adopts that angle for the figure instead of the default isometric view."
        )
        imgui.text_disabled(
            "elev %.0f°   azim %.0f°%s"
            % (
                camera.elev,
                camera.azim,
                "   (Plot will use this)" if self._preview_oriented else "",
            )
        )
        if self._preview_note:
            imgui.text_wrapped(self._preview_note)

    def _draw_params(self) -> None:
        spec = self.spec()
        params = self.current_params()
        if not spec.params:
            imgui.text_disabled("This object has no parameters.")
        for param in spec.params:
            value = float(params.get(param.name, param.default))
            if param.integer:
                changed, new_value = imgui.slider_int(
                    param.label, int(round(value)), int(param.vmin), int(param.vmax)
                )
            else:
                changed, new_value = widgets.labeled_slider_float(
                    param.label, value, param.vmin, param.vmax, fmt="%.3f"
                )
            if changed:
                params[param.name] = param.clamp(float(new_value))

        imgui.spacing()
        changed, value = imgui.slider_int(
            "Samples", int(self.samples), MIN_SAMPLES_UI, MAX_SAMPLES_UI
        )
        if changed:
            self.samples = int(value)
        imgui.same_line()
        widgets.help_marker(_HELP_SAMPLES)
        if spec.category == "surface":
            side = generators3d.surface_resolution(self.samples)
            imgui.text_disabled(f"mesh: {side} x {side}")

        if imgui.button("Reset parameters"):
            self.params[self.key] = spec.defaults()

    def _draw_appearance(self) -> None:
        spec = self.spec()
        kinds = list(layerops3d.KIND3D_KEYS)
        labels = [layerops3d.kind3d_spec(k).label for k in kinds]
        current = layerops3d.kind3d_spec(self.current_kind()).label
        changed, picked = widgets.enum_combo("Plot type", current, labels, help=_HELP_KIND)
        if changed and picked in labels:
            self.kind = kinds[labels.index(picked)]

        # Colour source: any column, or "z" when the object names none.
        options = ["(use z)"] + list(spec.columns)
        current_color = spec.color_column or "(use z)"
        changed, picked = widgets.enum_combo("Colour by", current_color, options, help=_HELP_COLOR)
        if changed:
            self._color_override = None if picked == "(use z)" else picked

        changed, value = widgets.enum_combo(
            "Colormap",
            self.cmap,
            ("viridis", "magma", "plasma", "inferno", "turbo", "cividis", "coolwarm"),
        )
        if changed:
            self.cmap = str(value)

        changed, value = widgets.labeled_slider_float(
            "Point size", self.point_size, 0.5, 20.0, fmt="%.1f px"
        )
        if changed:
            self.point_size = float(value)
        changed, value = widgets.labeled_slider_float(
            "Line width", self.line_width, 0.5, 8.0, fmt="%.1f px"
        )
        if changed:
            self.line_width = float(value)
        changed, value = widgets.labeled_slider_float("Alpha", self.alpha, 0.02, 1.0, fmt="%.2f")
        if changed:
            self.alpha = float(value)

    def _draw_actions(self) -> None:
        if imgui.button("Plot"):
            self._action_plot()
        imgui.same_line()
        if imgui.button("Create dataset"):
            self._action_create_dataset()
        imgui.same_line()
        widgets.help_marker(
            "Plot generates the object and adds it to the scene as a layer, with its "
            "table registered in the Data panel. Create dataset makes the table only, "
            "for editing or transforming before it is drawn."
        )

    # -- actions ---------------------------------------------------------------

    def _build_table(self) -> Tuple[Generator3D, Dict[str, np.ndarray]]:
        """Generate at the full requested resolution. Raises on a bad parameter set."""
        spec = self.spec()
        return spec, spec.generate(self.current_params(), self.samples)

    def _dataset_from(self, spec: Generator3D, table: Dict[str, np.ndarray]) -> DataSet:
        """Wrap a generated table in an unregistered :class:`DataSet`."""
        dataset = DataSet(
            self.store.unique_name(spec.label),
            [Column(name, np.asarray(values, dtype=np.float64)) for name, values in table.items()],
        )
        dataset.source = "generated"
        return dataset

    def _color_values(
        self, spec: Generator3D, table: Dict[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        """The column to colour-map by: the user's override, then the generator's own."""
        name = self._color_override if self._color_override is not None else spec.color_column
        if not name:
            return None
        return table.get(name)

    def _plot_options(self, spec: Generator3D, table: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """The plot-kind options this object's output needs.

        Two kinds need more than the three geometry columns: a surface needs its sampling
        shape (its ``(x, y)`` is not a lattice — see
        :func:`glplot.gui.layerops3d.resolve_grid`), and a quiver needs the vector
        columns. Both come from the generator, so neither is something the user has to
        wire up by hand.
        """
        kind = self.current_kind()
        options: Dict[str, Any] = {}
        if kind in ("surface3d", "wireframe3d"):
            shape = spec.grid_shape(self.samples)
            if shape is not None:
                options["nu"], options["nv"] = shape
        if kind == "quiver3d":
            for name in ("u", "v", "w"):
                if name in table:
                    options[name] = table[name]
        return options

    def _preview_view(self) -> Optional[Tuple[float, float, float]]:
        """The thumbnail's ``(elev, azim, roll)`` if the user chose it, else None.

        Carrying the angle over is nearly free — :meth:`~glplot.engine.GPULinePlot.frame_3d_view`
        re-fits the distance and leaves the orientation alone, so setting the three angles
        before it runs is the whole implementation — and it is what "see what I am about
        to add" implies: an object framed the way the preview showed it. What is *not*
        carried is the distance, because the preview frames one object and the figure may
        already hold several.
        """
        if not self._preview_oriented:
            return None
        camera = self.preview_camera
        return float(camera.elev), float(camera.azim), float(camera.roll)

    def _action_plot(self) -> None:
        """Generate, register the table, and add the layer — as one undoable command."""
        try:
            spec, table = self._build_table()
        except (GeneratorError, ValueError, ArithmeticError) as exc:
            self._error = str(exc)
            return

        kind = self.current_kind()
        try:
            options = layerops3d.resolve_kind3d_options(kind, self._plot_options(spec, table))
        except ValueError as exc:
            self._error = str(exc)
            return

        dataset = self._dataset_from(spec, table)
        cx, cy, cz = spec.plot_columns
        colors = self._color_values(spec, table)
        plot = self.plot
        store = self.store
        label = plot_label = self._unique_label(spec.label)
        created: List[Any] = []
        # Captured now, not read from the panel inside `do`: a command can be redone long
        # after the thumbnail has spun on, and a redo must reproduce the picture it made
        # the first time. None means "the user never orbited the preview", in which case
        # the figure keeps whatever view it already had.
        view = self._preview_view()

        def do() -> None:
            # Ensure the figure is 3D *before* the layer lands, so `add_geometry3d`'s
            # camera sync frames it rather than leaving it under a 2D projection.
            if not plot.active_panel.is_3d():
                plot.set_ndim(3)
            if view is not None:
                plot.set_3d_view(elev=view[0], azim=view[1], roll=view[2])
            layer = layerops3d.add_xyz_layer(
                plot,
                table[cx],
                table[cy],
                table[cz],
                kind=kind,
                label=plot_label,
                options=options or None,
                c=colors,
                cmap=self.cmap,
                size=self.point_size,
                width=self.line_width,
                alpha=self.alpha,
            )
            created.append(layer)
            store.add(dataset)
            dataset.layer_id = getattr(layer, "layer_id", None)
            plot.frame_3d_view()

        def undo() -> None:
            store.remove(dataset)
            for layer in created:
                layerops.remove_layer(plot, self.hud, layer)
            created.clear()

        self.push_command(Command(f"Plot {spec.label}", do, undo))
        self._error = ""
        self._status = f"Plotted {label} ({len(table[cx])} rows) as {kind}."

    def _action_create_dataset(self) -> None:
        """Generate the table and register it, without drawing anything."""
        try:
            spec, table = self._build_table()
        except (GeneratorError, ValueError, ArithmeticError) as exc:
            self._error = str(exc)
            return

        dataset = self._dataset_from(spec, table)
        store = self.store

        def do() -> None:
            store.add(dataset)

        def undo() -> None:
            store.remove(dataset)

        self.push_command(Command(f"Create {spec.label} dataset", do, undo))
        self._error = ""
        self._status = f"Created dataset {dataset.name!r} ({dataset.n_rows()} rows)."

    def _unique_label(self, base: str) -> str:
        """A layer label not already taken, so two spheres are tellable apart."""
        taken = {str(getattr(layer, "label", "")) for layer in self.plot.scene.layers}
        if base not in taken:
            return base
        index = 2
        while f"{base} ({index})" in taken:
            index += 1
        return f"{base} ({index})"
