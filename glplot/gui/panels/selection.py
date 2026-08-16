"""The Selection panel — what is selected, and what you can do with it.

The user's literal ask was "botones en la parte de seleccion para seleccionar todo":
**Select All / Select None / Invert**, as real buttons, scoped by a layer combo. The rest
of the panel is what makes a selection worth having — a header that says what is
selected, per-layer breakdown, statistics, and the actions that turn a selection into
data: Create dataset from selection, Delete selected points, Isolate, Zoom to selection.

Every mutation is deferred to the command queue (CONTRACT §1.1); the discrete ones are
undoable. The buttons are registered as actions in ``workspace._register_actions``, so
they are reachable from the command palette too.

Four decisions worth knowing before editing this file:

* **Element ids mean exactly what the picking pass says they mean.**
  :func:`selection_element_count` mirrors ``PickingManager._offset_table`` deliberately
  and must keep mirroring it: a point index for ``scatter``, a line index for
  ``line_family``, and a single element standing for the whole layer on ``polyline`` /
  ``patch``. Select All over a polyline therefore selects *one* element, not one per
  vertex. ``layerops.layer_element_count`` is **not** usable here — it returns
  ``len(pts)`` for a polyline, which would invent ids the picker can never produce and
  make Invert produce a selection that nothing can ever match.

* **The highlight is pushed onto the layers, not read from the panel.** The renderer
  cannot see ``plot.interaction.selection`` (it gets a layer and a ``RenderContext``,
  and neither carries it), so :func:`sync_selection_highlight` copies the selection onto
  ``layer._selected_indices`` and bumps ``layer._selection_version``; ``ScatterRenderer``
  reads those two attributes and nothing else. That keeps ``renderers/scatter.py`` free
  of any import from ``glplot.gui``.

* **The sync runs in the drain, never in the draw.** It is a scene mutation in the sense
  that matters (§1.2): setting ``dirty_scene`` from inside a draw callback loses the
  latch at ``engine.py:1238``. So the panel only *detects* a divergence during the draw
  (:func:`selection_needs_sync`, a pure read) and submits the sync; the §1.3 epilogue
  then sets the flags and drops ``cache.capture_window``, which is what rebuilds the
  impostor with the new highlight baked in. The submit is self-limiting: once the layers
  match the selection the predicate goes quiet, so this cannot spin the queue.

* **Statistics and Create-dataset are scatter-only, and say so.** ``describe`` on a
  ``line_family``'s selected elements would summarise its ``(a, b)`` line parameters and
  label them "x" and "y", which is a lie about what the numbers are. A polyline
  contributes one opaque element id, so there is nothing per-point to summarise at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from .. import layerops, widgets
from ..datasets import Column, DataSet
from ..history import Command, is_snapshot_safe
from ..icons import icon_button
from ..mathops import describe
from .base import Panel

try:
    import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - imgui is a hard dependency in CI
    IMGUI_AVAILABLE = False
    imgui = None

if TYPE_CHECKING:
    from ..workspace import Workspace

__all__ = [
    "SelectionPanel",
    "selectable_layers",
    "selection_element_count",
    "selection_needs_sync",
    "sync_selection_highlight",
]

#: Layer types that render a selection highlight today. ``line_family`` is picked and
#: selected correctly but draws no highlight — see the module note in the panel body.
HIGHLIGHTED_TYPES = ("scatter",)

#: Layer types whose selected elements are individually addressable x/y points, i.e. the
#: ones Statistics / Create dataset / Delete / Zoom can honestly work on.
POINTWISE_TYPES = ("scatter",)


def selection_element_count(layer: Any) -> int:
    """How many selectable elements ``layer`` contributes, per the picking id scheme.

    Mirrors ``PickingManager._offset_table``: per point for ``scatter``, per line for
    ``line_family``, one for the whole layer on ``polyline`` / ``patch``, and zero for
    anything the picker never draws (text, 3D). Returns 0 for an empty or invisible
    layer, so Select All over it is a no-op rather than an invented id.
    """
    layer_type = str(getattr(layer, "layer_type", ""))
    if layer_type == "scatter":
        pts = getattr(layer, "pts", None)
        return 0 if pts is None else int(len(pts))
    if layer_type == "line_family":
        ab = getattr(layer, "ab", None)
        return 0 if ab is None else int(len(ab))
    if layer_type in ("polyline", "patch"):
        return 1
    return 0


def selectable_layers(plot: Any) -> List[Any]:
    """Visible layers that contribute at least one element id, in scene order."""
    try:
        layers = list(plot.scene.layers)
    except Exception:  # pragma: no cover - defensive
        return []
    out = []
    for layer in layers:
        if not getattr(getattr(layer, "style", None), "visible", False):
            continue
        if selection_element_count(layer) > 0:
            out.append(layer)
    return out


def _selection_of(plot: Any) -> Optional[Any]:
    """The live :class:`~glplot.core.legacy.Selection`, or None on an engine without one."""
    return getattr(getattr(plot, "interaction", None), "selection", None)


def _highlight_indices(selection: Any, layer: Any) -> np.ndarray:
    """The uint32 indices the renderer should highlight for ``layer`` (sorted)."""
    layer_id = getattr(layer, "layer_id", None)
    if layer_id is None:
        return np.empty(0, dtype=np.uint32)
    idx = selection.indices(layer_id)
    return np.asarray(idx, dtype=np.uint32)


def selection_needs_sync(plot: Any) -> bool:
    """True when a layer's highlight buffer disagrees with the live selection.

    Pure read — safe from a draw callback, which is the whole point: the workspace calls
    this every frame and only pays for :func:`sync_selection_highlight` when something
    actually moved.
    """
    selection = _selection_of(plot)
    if selection is None:
        return False
    try:
        layers = list(plot.scene.layers)
    except Exception:  # pragma: no cover - defensive
        return False
    for layer in layers:
        if str(getattr(layer, "layer_type", "")) not in HIGHLIGHTED_TYPES:
            continue
        want = _highlight_indices(selection, layer)
        have = getattr(layer, "_selected_indices", None)
        if have is None:
            if want.size:
                return True
            continue
        if have.size != want.size or not np.array_equal(have, want):
            return True
    return False


def sync_selection_highlight(plot: Any) -> bool:
    """Copy the live selection onto the layers' highlight buffers. Returns True if changed.

    **Queued commands only** (see the module docstring): it must be the drain, not the
    draw, that marks the scene dirty, or the flag is cleared at ``engine.py:1238`` before
    it ever gates a render and the highlight does not appear until an unrelated event
    wakes the loop.

    Bumping ``_selection_version`` is what tells ``ScatterRenderer`` to re-upload its
    element buffer; without it the renderer would keep drawing the previous set.
    """
    selection = _selection_of(plot)
    if selection is None:
        return False
    try:
        layers = list(plot.scene.layers)
    except Exception:  # pragma: no cover - defensive
        return False

    changed = False
    for layer in layers:
        if str(getattr(layer, "layer_type", "")) not in HIGHLIGHTED_TYPES:
            continue
        want = _highlight_indices(selection, layer)
        have = getattr(layer, "_selected_indices", None)
        if have is not None and have.size == want.size and np.array_equal(have, want):
            continue
        layer._selected_indices = want
        layer._selection_version = int(getattr(layer, "_selection_version", 0)) + 1
        changed = True

    if changed:
        layerops.mark_scene_dirty(plot)
    return changed


def _fmt_count(n: int) -> str:
    """Thousands-separated count."""
    return "{0:,}".format(int(n))


def _selected_xy(plot: Any, selection: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Concatenated x/y of every selected point across the point-wise layers.

    None when nothing point-wise is selected. Applies ``layer.translation``, because that
    is what ``get_bounds`` does (``renderer_manager.py:199-203``) and what the user sees.
    """
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    try:
        layers = list(plot.scene.layers)
    except Exception:  # pragma: no cover - defensive
        return None
    for layer in layers:
        if str(getattr(layer, "layer_type", "")) not in POINTWISE_TYPES:
            continue
        layer_id = getattr(layer, "layer_id", None)
        if layer_id is None or selection.count_in(layer_id) == 0:
            continue
        pts = getattr(layer, "pts", None)
        if pts is None or len(pts) == 0:
            continue
        idx = selection.indices(layer_id)
        idx = idx[(idx >= 0) & (idx < len(pts))]
        if idx.size == 0:
            continue
        tx, ty = getattr(layer, "translation", (0.0, 0.0))
        xs.append(np.asarray(pts[idx, 0], dtype=np.float64) + float(tx))
        ys.append(np.asarray(pts[idx, 1], dtype=np.float64) + float(ty))
    if not xs:
        return None
    return np.concatenate(xs), np.concatenate(ys)


class SelectionPanel(Panel):
    """What is selected on the canvas, and the verbs that act on it."""

    title = "Selection"
    icon = "target"
    default_open = False

    def __init__(self, ws: Workspace) -> None:
        super().__init__(ws)
        #: Scope for the three buttons: None means "All layers", else a ``layer_id``.
        self.scope_layer_id: Optional[int] = None
        self._new_name = "Selection"

    # -- scope -----------------------------------------------------------------

    def scoped_layers(self) -> List[Any]:
        """The layers the buttons act on: all selectable ones, or just the scoped one."""
        layers = selectable_layers(self.plot)
        if self.scope_layer_id is None:
            return layers
        return [ly for ly in layers if getattr(ly, "layer_id", None) == self.scope_layer_id]

    def _scope_label(self) -> str:
        if self.scope_layer_id is None:
            return "All layers"
        for layer in selectable_layers(self.plot):
            if getattr(layer, "layer_id", None) == self.scope_layer_id:
                return str(getattr(layer, "label", "?"))
        return "All layers"

    # -- the three buttons -----------------------------------------------------

    def select_all(self) -> None:
        """Select every element of every layer in scope. Queued, not undoable."""
        targets = [
            (getattr(ly, "layer_id", None), selection_element_count(ly))
            for ly in self.scoped_layers()
        ]
        targets = [(lid, n) for lid, n in targets if lid is not None and n > 0]
        if not targets:
            return
        scope_all = self.scope_layer_id is None
        selection = _selection_of(self.plot)
        if selection is None:
            return

        def run() -> None:
            if scope_all:
                selection.clear()
            for layer_id, count in targets:
                selection.add(layer_id, np.arange(count, dtype=np.int64))
            sync_selection_highlight(self.plot)

        self.submit(run)

    def select_none(self) -> None:
        """Clear the selection — the whole selection, or just the scoped layer's."""
        selection = _selection_of(self.plot)
        if selection is None:
            return
        scope = self.scope_layer_id
        ids = [getattr(ly, "layer_id", None) for ly in self.scoped_layers()]

        def run() -> None:
            if scope is None:
                selection.clear()
            else:
                for layer_id in ids:
                    if layer_id is not None:
                        selection.discard_layer(layer_id)
            sync_selection_highlight(self.plot)

        self.submit(run)

    def invert(self) -> None:
        """Invert the selection within every layer in scope."""
        targets = [
            (getattr(ly, "layer_id", None), selection_element_count(ly))
            for ly in self.scoped_layers()
        ]
        targets = [(lid, n) for lid, n in targets if lid is not None and n > 0]
        if not targets:
            return
        selection = _selection_of(self.plot)
        if selection is None:
            return

        def run() -> None:
            for layer_id, count in targets:
                selection.invert(layer_id, count)
            sync_selection_highlight(self.plot)

        self.submit(run)

    # -- actions ---------------------------------------------------------------

    def has_selection(self) -> bool:
        """True when anything at all is selected."""
        selection = _selection_of(self.plot)
        return selection is not None and not selection.is_empty()

    def has_pointwise_selection(self) -> bool:
        """True when at least one *point* is selected (not just a line or a patch)."""
        selection = _selection_of(self.plot)
        if selection is None:
            return False
        return _selected_xy(self.plot, selection) is not None

    def create_dataset(self) -> None:
        """Create a DataStore dataset from the selected points. Undoable.

        The payoff action: the new dataset lands in the Data panel and is bindable from
        Math Lab like any other, so a marquee on the canvas becomes a table you can
        differentiate, smooth or re-plot.
        """
        selection = _selection_of(self.plot)
        if selection is None:
            return
        xy = _selected_xy(self.plot, selection)
        if xy is None:
            return
        xs, ys = xy
        store = self.store
        name = store.unique_name(self._new_name.strip() or "Selection")
        holder: Dict[str, Any] = {}

        def do() -> None:
            ds = holder.get("ds")
            if ds is None:
                ds = DataSet(name, [Column("x", xs), Column("y", ys)])
                holder["ds"] = ds
            store.add(ds)

        def undo() -> None:
            ds = holder.get("ds")
            if ds is not None:
                store.remove(ds)

        self.push_command(Command(label=f"Create dataset '{name}'", do=do, undo=undo))
        self.ws.open_panel("data")

    def delete_selected(self) -> None:
        """Delete the selected points from their layers. Undoable when affordable.

        Deleting renumbers every index after the deleted one, so the layer's selection is
        dropped rather than left pointing at whatever slid into its place.
        """
        selection = _selection_of(self.plot)
        if selection is None:
            return
        plot = self.plot
        jobs: List[Tuple[Any, np.ndarray, np.ndarray, np.ndarray]] = []
        for layer in list(plot.scene.layers):
            if str(getattr(layer, "layer_type", "")) not in POINTWISE_TYPES:
                continue
            layer_id = getattr(layer, "layer_id", None)
            if layer_id is None or selection.count_in(layer_id) == 0:
                continue
            xy = layerops.layer_xy(layer)
            if xy is None:
                continue
            old_x, old_y = xy
            mask = selection.as_mask(layer_id, len(old_x))
            if not mask.any() or mask.all():
                # All-or-nothing: deleting every point would leave an empty layer that
                # renders nothing and cannot be re-selected. Scene > Delete removes a
                # whole layer properly (§1.6); this action refuses to half-do it.
                continue
            jobs.append((layer, old_x, old_y, mask))
        if not jobs:
            return

        snapshots = [arr for _, ox, oy, _ in jobs for arr in (ox, oy)]
        undoable = is_snapshot_safe(*snapshots)
        saved = {
            id(layer): selection.indices(getattr(layer, "layer_id")) for layer, _, _, _ in jobs
        }

        def do() -> None:
            for layer, old_x, old_y, mask in jobs:
                keep = ~mask
                layerops.update_layer_xy(plot, layer, old_x[keep], old_y[keep])
                selection.discard_layer(getattr(layer, "layer_id"))
            sync_selection_highlight(plot)

        def undo() -> None:
            for layer, old_x, old_y, _ in jobs:
                layerops.update_layer_xy(plot, layer, old_x, old_y)
                selection.add(getattr(layer, "layer_id"), saved[id(layer)])
            sync_selection_highlight(plot)

        total = int(sum(int(m.sum()) for _, _, _, m in jobs))
        label = f"Delete {_fmt_count(total)} selected points"
        if undoable:
            self.push_command(Command(label=label, do=do, undo=undo))
        else:
            self.push_command(Command.not_undoable(label, do))

    def isolate(self) -> None:
        """Hide every layer that has nothing selected.

        Layer visibility, not point deletion: non-destructive and reversible from the
        Scene panel's eye. Hiding a layer also removes it from the picking pass
        (``picking.py:98`` skips invisible layers), so an isolated view picks only what
        it shows — which is the point of isolating.
        """
        selection = _selection_of(self.plot)
        if selection is None or selection.is_empty():
            return
        plot = self.plot
        victims = []
        for layer in list(plot.scene.layers):
            layer_id = getattr(layer, "layer_id", None)
            if layer_id is None:
                continue
            if selection.count_in(layer_id) == 0 and getattr(layer.style, "visible", False):
                victims.append(layer)
        if not victims:
            return

        def do() -> None:
            for layer in victims:
                layerops.set_layer_style(plot, layer, visible=False)

        def undo() -> None:
            for layer in victims:
                layerops.set_layer_style(plot, layer, visible=True)

        self.push_command(Command(label="Isolate selection", do=do, undo=undo))

    def zoom_to_selection(self) -> None:
        """Frame the selected points. Queued (it calls into the camera), not undoable."""
        selection = _selection_of(self.plot)
        if selection is None:
            return
        xy = _selected_xy(self.plot, selection)
        if xy is None:
            return
        xs, ys = xy
        finite = np.isfinite(xs) & np.isfinite(ys)
        if not finite.any():
            return
        xs, ys = xs[finite], ys[finite]
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        # A single point (or an axis-aligned line) has zero extent on an axis; framing it
        # verbatim divides by zero in fit_bounds. Pad to something the user can see.
        px = (x1 - x0) * 0.1 or max(abs(x0) * 0.1, 0.5)
        py = (y1 - y0) * 0.1 or max(abs(y0) * 0.1, 0.5)
        plot = self.plot

        def run() -> None:
            plot.set_view(xlim=(x0 - px, x1 + px), ylim=(y0 - py, y1 + py))

        self.submit(run)

    # -- draw ------------------------------------------------------------------

    def draw(self) -> None:
        """Render the panel body."""
        if not IMGUI_AVAILABLE:  # pragma: no cover - imgui is a hard dependency in CI
            return
        selection = _selection_of(self.plot)
        if selection is None:
            imgui.text_colored("This engine build has no selection model.", 0.9, 0.35, 0.35)
            return

        # One id scope for the whole panel. Round 2 lost a combo to a label collision
        # between a radio and a combo that shared a name; the buttons here reuse verbs
        # ("Delete", "Isolate") that other panels also use.
        imgui.push_id("selection_panel")
        try:
            self._draw_header(selection)
            imgui.separator()
            self._draw_scope()
            self._draw_buttons()
            imgui.separator()
            self._draw_breakdown(selection)
            self._draw_stats(selection)
            imgui.separator()
            self._draw_actions()
        finally:
            imgui.pop_id()

    def _draw_header(self, selection: Any) -> None:
        """ "1,284 points across 2 layers" — what is selected, in one line."""
        total = selection.count()
        n_layers = len(selection.layers())
        if total == 0:
            imgui.text_colored("Nothing selected", 0.48, 0.51, 0.56, 1.0)
            imgui.text_colored(
                "Shift+drag on the canvas, or use Select All.", 0.48, 0.51, 0.56, 1.0
            )
            return
        noun = "element" if total == 1 else "elements"
        plural = "layer" if n_layers == 1 else "layers"
        imgui.text(f"{_fmt_count(total)} {noun} across {n_layers} {plural}")

    def _draw_scope(self) -> None:
        """The layer combo the three buttons are scoped by."""
        layers = selectable_layers(self.plot)
        names = ["All layers"] + [str(getattr(ly, "label", "?")) for ly in layers]
        ids: List[Optional[int]] = [None] + [getattr(ly, "layer_id", None) for ly in layers]
        current = ids.index(self.scope_layer_id) if self.scope_layer_id in ids else 0
        imgui.set_next_item_width(-1.0)
        changed, index = imgui.combo("##scope", current, names)
        if changed and 0 <= index < len(ids):
            self.scope_layer_id = ids[index]
        if imgui.is_item_hovered():
            imgui.set_tooltip("Which layers Select All / None / Invert act on.")

    def _draw_buttons(self) -> None:
        """The explicit ask: Select All / Select None / Invert, as icon buttons."""
        enabled = bool(selectable_layers(self.plot))
        if icon_button("sel_all", "check", tooltip="Select All", enabled=enabled):
            self.select_all()
        imgui.same_line()
        if icon_button("sel_none", "close", tooltip="Select None", enabled=self.has_selection()):
            self.select_none()
        imgui.same_line()
        if icon_button("sel_invert", "refresh", tooltip="Invert Selection", enabled=enabled):
            self.invert()
        imgui.same_line()
        imgui.text_colored(f"scope: {self._scope_label()}", 0.48, 0.51, 0.56, 1.0)

    def _draw_breakdown(self, selection: Any) -> None:
        """Per-layer "42 / 500" rows, so a multi-layer selection is legible."""
        if selection.is_empty():
            return
        if not widgets.section("Per layer", default_open=True):
            return
        by_id = {getattr(ly, "layer_id", None): ly for ly in selectable_layers(self.plot)}
        for layer_id in selection.layers():
            layer = by_id.get(layer_id)
            count = selection.count_in(layer_id)
            if layer is None:
                # Selected ids for a layer that is gone or hidden. Say so rather than
                # dropping the row: the count is real and it is in the header total.
                widgets.stat_row(f"(layer {layer_id})", f"{_fmt_count(count)} — hidden/removed")
                continue
            total = selection_element_count(layer)
            label = str(getattr(layer, "label", "?"))
            widgets.stat_row(label, f"{_fmt_count(count)} / {_fmt_count(total)}")

    def _draw_stats(self, selection: Any) -> None:
        """``mathops.describe`` over the selected x and y."""
        xy = _selected_xy(self.plot, selection)
        if xy is None:
            return
        if not widgets.section("Statistics", default_open=True):
            return
        xs, ys = xy
        try:
            sx, sy = describe(xs), describe(ys)
        except ValueError:  # pragma: no cover - _selected_xy never yields empty
            return
        table = imgui.begin_table(
            "stats", 3, imgui.TABLE_ROW_BACKGROUND | imgui.TABLE_BORDERS_INNER
        )
        # .opened, per CONTRACT §2.2 — and end_table() only inside the guard.
        if table.opened:
            imgui.table_setup_column("")
            imgui.table_setup_column("x")
            imgui.table_setup_column("y")
            imgui.table_headers_row()
            for key in ("n", "min", "max", "mean", "median", "std"):
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(key)
                imgui.table_next_column()
                imgui.text(_fmt_count(sx[key]) if key == "n" else f"{sx[key]:.6g}")
                imgui.table_next_column()
                imgui.text(_fmt_count(sy[key]) if key == "n" else f"{sy[key]:.6g}")
            imgui.end_table()

    def _draw_actions(self) -> None:
        """Create dataset / Delete / Isolate / Zoom — every one of them queued."""
        pointwise = self.has_pointwise_selection()
        any_sel = self.has_selection()

        imgui.set_next_item_width(-1.0)
        changed, value = imgui.input_text("##dsname", self._new_name)
        if changed:
            self._new_name = value
        if imgui.is_item_hovered():
            imgui.set_tooltip("Name for the dataset created from the selection.")

        if icon_button(
            "sel_dataset", "table", tooltip="Create dataset from selection", enabled=pointwise
        ):
            self.create_dataset()
        imgui.same_line()
        if icon_button("sel_delete", "trash", tooltip="Delete selected points", enabled=pointwise):
            self.delete_selected()
        imgui.same_line()
        if icon_button(
            "sel_isolate", "filter", tooltip="Isolate (hide unselected layers)", enabled=any_sel
        ):
            self.isolate()
        imgui.same_line()
        if icon_button("sel_zoom", "zoom_in", tooltip="Zoom to selection", enabled=pointwise):
            self.zoom_to_selection()

        if any_sel and not pointwise:
            imgui.text_colored(
                "Selected elements are lines or patches: no per-point data to extract.",
                0.48,
                0.51,
                0.56,
                1.0,
            )
