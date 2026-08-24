"""The Scene panel — the layer tree and the scene-wide actions.

One row per layer: visibility eye, colour chip, type chip, name (inline-renameable),
inline opacity and a size readout. Rows drag to reorder, right-click for rename /
duplicate / isolate / change type / delete. Above them sit a filter box and the
scene-wide actions; below them, an inspector for the selected layer.

Every mutation is deferred to the command queue (CONTRACT §1.1). Nothing here touches
``plot.scene`` directly — the draw callback only ever *collects intent*, exactly as the
existing ``hud._draw_layers_panel`` defers its ``pop``/``insert`` until after the
enumeration loop (``hud.py:296-301``); "after the loop" simply becomes "in the drain".

Five deliberate decisions worth knowing before editing this file:

* **Rows are listed in DRAW order, not list order, and dragging commits it.** This is
  the panel's one correctness fix, not a cosmetic one. The renderer sorts by
  ``style.zorder`` (``renderer_manager.py:96``) and uses list order only as the tie-break,
  while ``layerops.move_layer`` deliberately does not touch ``zorder`` — so on the old
  code the panel showed one order, the screen showed another, and dragging a row moved
  the row without moving the picture. Both halves are fixed: rows come from
  ``layerops.draw_order`` so the list cannot lie, and a drag goes through
  ``layerops.move_layer_to_index``, which renumbers ``zorder`` to the dropped order so
  the gesture means what it looks like. A ``z:N`` badge appears on every row whenever
  zorder is what is actually ranking them, so the rule is visible rather than magic.
* **Only discrete edits are undoable.** Delete, duplicate, rename and type-change push
  onto the :class:`~glplot.gui.history.UndoStack`. Visibility, isolate, drag-reorder,
  the colour chip and the opacity drag do not: a colour or alpha drag fires on every
  frame of the gesture and would bury the history under hundreds of entries, and the
  others are one click to reverse by hand. Clear is explicitly *not* undoable — see
  :meth:`ScenePanel._clear_scene`.
* **The colour chip is only editable where colour is actually live** (§4.4). On a
  scatter or a line family the colour lives in a per-vertex VBO and ``style.color`` is
  read by nothing, so the chip renders as a read-only swatch with a tooltip that says
  why. Wiring it would be a lie, per §4.1. The same gate governs every inspector
  control: a widget is drawn only where the field it writes is read at draw time.
* **Drag payloads carry ``layer_id``, not the row index.** The index is resolved at
  drain time. Indices are captured in the draw callback but applied a frame later, and
  anything else queued in between (a delete from the context menu, a duplicate) shifts
  them — an index captured at draw time can name a different layer by the time it runs.
* **Selection is a set, and the HUD's single id is its anchor.** ``hud.state`` holds one
  id because the engine's picking does; the panel keeps the multi-selection beside it
  and treats any *external* change to that id (a click in the viewport, another panel)
  as a fresh single selection. See :meth:`ScenePanel._sync_selection`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ...core.camera3d import SYSTEM_3D_ARTISTS
from .. import layerops, layerops3d, widgets
from ..history import Command
from ..icons import icon_button, icon_inline
from ..layerops import DEFAULT_LAYER_COLOR
from ..widgets import confirm_popup
from .base import Panel

if TYPE_CHECKING:
    from ..workspace import Workspace

try:
    from imgui_bundle import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):
    IMGUI_AVAILABLE = False
    imgui = None


#: Drag-and-drop payload type for row reordering.
_DND_TYPE = "GLPLOT_LAYER_ORDER"

#: Modal id. Must match between ``open_popup`` and ``confirm_popup``.
_CLEAR_POPUP = "Clear scene?"

#: ``layer_type`` -> icon name. Unknown types fall back to ``_DEFAULT_ICON``; an unknown
#: *icon* would only draw a placeholder, so a new layer type degrades rather than raises.
_TYPE_ICONS: Dict[str, str] = {
    "scatter": "chart_scatter",
    "polyline": "chart_line",
    "line_family": "wave",
    "patch": "chart_bar",
    "text": "info",
    "semantic_line": "minus",
    "semantic_span": "filter",
    "geometry3d": "axes3d",
    "scatter3d": "chart_scatter3d",
    "volume3d": "chart_volume3d",
    "wireframe3d": "chart_wireframe3d",
    "mesh3d": "chart_surface3d",
    "surface3d": "chart_surface3d",
    "bars3d": "chart_bar3d",
}
_DEFAULT_ICON = "layers"

#: Short human names for the row's type chip. The raw ``layer_type`` is still shown in
#: the inspector; this is the glanceable version.
_TYPE_LABELS: Dict[str, str] = {
    "scatter": "scatter",
    "polyline": "line",
    "line_family": "lines",
    "patch": "patch",
    "text": "text",
    "semantic_line": "ruler",
    "semantic_span": "span",
    "geometry3d": "3d",
    "scatter3d": "3d pts",
    "volume3d": "volume",
    "wireframe3d": "wire",
    "mesh3d": "mesh",
    "surface3d": "surface",
    "bars3d": "3d bars",
}

#: ``layer_type``s whose ``style.alpha`` is read by nothing, so no opacity control is
#: drawn for them. Text draws through imgui's draw list and takes its alpha from
#: ``style.color`` (``text.py:59``), not from ``style.alpha`` (§4.4).
_ALPHA_DEAD_TYPES = frozenset({"text"})

_ROW_ICON_SIZE = 18.0
_CHIP_SIZE = 16.0
_ALPHA_WIDTH = 44.0
#: Height reserved below the layer list for the inspector, by open/closed state.
_INSPECTOR_OPEN_H = 208.0
_INSPECTOR_SHUT_H = 30.0


def _available_kinds() -> Tuple[str, ...]:
    """The ``add_xy_layer`` kinds the type chip may offer.

    ``layerops.KIND_KEYS`` is the single source of truth for what is reachable (its
    ``add_xy_layer`` raises on anything else), so it is read straight, at call time.
    Nothing is caught and nothing is defaulted: this used to probe for a registry under a
    name ``layerops`` never published and quietly fall back to a hardcoded
    ``("line", "scatter")``, which hid the other five kinds for as long as the misspelling
    survived -- and hid *itself*, because a fallback that works looks like a registry that
    is empty. A kind the user cannot pick is a kind they cannot return a layer to.
    """
    return tuple(str(k) for k in layerops.KIND_KEYS)


def _can_change_kind(layer: Any) -> bool:
    """True when ``layer`` can honestly be rebuilt as another plot type.

    False for a layer with no kind (a field, a guide, a 3D mesh) and for one with no
    source columns to rebuild from (a line family, whose columns are a/b rather than x/y;
    a ``pyplot`` patch, which recorded none).

    It asks :func:`layerops.layer_source_xy`, never ``layer_xy``, for the same reason
    :meth:`ScenePanel._change_kind` rebuilds from it: a ``bar``/``area``/``hist`` layer is
    a patch and has no ``layer_xy`` at all, so asking the geometry greys the menu out on
    exactly the layers a type change has just produced -- stranding them in their new
    type with no way back to the one they started in.
    """
    return layerops.layer_kind(layer) is not None and layerops.layer_source_xy(layer) is not None


def _layer_type_icon(layer: Any) -> str:
    """Icon name for a layer's type, preferring its 3D *kind* where it has one.

    ``layer_type`` is too coarse in 3D: a path, a stem plot, a wireframe and a vector
    field are all ``wireframe3d`` layers, and showing one icon for the four of them makes
    the Scene list unreadable in exactly the case with the most layer types. The kind tag
    (:func:`glplot.gui.layerops3d.layer_kind3d`) is what distinguishes them, and it is
    present on everything the GUI built.
    """
    kind = layerops3d.layer_kind3d(layer)
    if kind is not None:
        return layerops3d.kind3d_spec(kind).icon
    return _TYPE_ICONS.get(str(getattr(layer, "layer_type", "")), _DEFAULT_ICON)


def _type_label(layer: Any) -> str:
    """Short type name for the row chip — the 3D kind where there is one."""
    kind = layerops3d.layer_kind3d(layer)
    if kind is not None:
        return kind
    layer_type = str(getattr(layer, "layer_type", ""))
    return _TYPE_LABELS.get(layer_type, layer_type or "layer")


def _layer_size_text(layer: Any) -> str:
    """Human size readout. Mirrors ``HudManager._layer_size_text`` (``hud.py:62-69``).

    Reimplemented rather than imported: importing ``HudManager`` into a panel would
    invert the layering (the HUD owns the workspace, not the other way round).
    """
    vertices = getattr(layer, "vertices", None)
    if vertices is not None:
        return f"{len(vertices):,} verts"
    pts = getattr(layer, "pts", None)
    if pts is not None:
        return f"{len(pts):,} pts"
    ab = getattr(layer, "ab", None)
    if ab is not None:
        return f"{len(ab):,} lines"
    return ""


def _display_label(layer: Any) -> str:
    """The visible row name.

    ``##`` is imgui's id separator: a layer literally labelled ``"a##b"`` would render
    as ``"a"`` and silently lose half its name, so the separator is defanged for display
    only. The layer's real ``label`` is never modified.
    """
    name = str(getattr(layer, "label", "") or getattr(layer, "layer_type", "") or "layer")
    return name.replace("##", "# #") if "##" in name else name


def _fmt_num(value: float) -> str:
    """Compact fixed/scientific formatting for a bounds or stats readout."""
    if not np.isfinite(value):
        return "-"
    magnitude = abs(value)
    if magnitude != 0.0 and (magnitude < 1e-3 or magnitude >= 1e5):
        return f"{value:.3e}"
    return f"{value:.4g}"


def _color_field(layer: Any) -> Optional[str]:
    """The ``LayerStyle`` field the colour chip should edit, or None if none is live.

    Per the §4.4 applicability matrix:

    * ``patch`` fills from ``style.face_color``; its ``style.color`` is read by nothing.
    * ``polyline`` (``polyline.py:173``) and ``text`` (``text.py:59``) read
      ``style.color`` as a uniform at draw time.
    * 3D layers use ``style.color`` **only** as the fallback for ``colors is None``
      (``geometry3d.py:130``); with per-vertex colours present it is inert.
    * ``scatter`` and ``line_family`` bake colour into a per-instance VBO — ``style.color``
      is dead for both.
    """
    layer_type = str(getattr(layer, "layer_type", ""))
    if layer_type == "patch":
        return "face_color"
    if layer_type in ("polyline", "text"):
        return "color"
    if layer_type.endswith("3d") and getattr(layer, "colors", None) is None:
        return "color"
    return None


def _effective_color(layer: Any) -> Tuple[float, float, float, float]:
    """The colour to *show* in the chip, which is not always the colour it can edit.

    Falls back to the first per-vertex colour so the swatch on a scatter still tells the
    truth about what is on screen, and finally to the engine's own default.
    """
    style = getattr(layer, "style", None)
    if style is not None:
        # A patch is drawn from face_color; its style.color is read by nothing.
        if str(getattr(layer, "layer_type", "")) == "patch":
            if getattr(style, "face_color", None) is not None:
                return _as_rgba(style.face_color)
        elif getattr(style, "color", None) is not None:
            return _as_rgba(style.color)

    colors = getattr(layer, "colors", None)
    if colors is not None:
        arr = np.asarray(colors, dtype=np.float64)
        if arr.ndim == 2 and len(arr) > 0:
            return _as_rgba(arr[0])
        if arr.ndim == 1 and arr.size in (3, 4):
            return _as_rgba(arr)
    return DEFAULT_LAYER_COLOR


def _as_rgba(value: Any) -> Tuple[float, float, float, float]:
    """Coerce an RGB/RGBA sequence to a 4-float tuple, tolerating junk."""
    try:
        vals = [float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1)]
    except (TypeError, ValueError):
        return DEFAULT_LAYER_COLOR
    if len(vals) == 3:
        vals.append(1.0)
    if len(vals) != 4:
        return DEFAULT_LAYER_COLOR
    return (vals[0], vals[1], vals[2], vals[3])


class ScenePanel(Panel):
    """Layer tree + scene-wide actions."""

    title = "Scene"
    icon = "layers"
    default_open = True

    def __init__(self, ws: Workspace) -> None:
        super().__init__(ws)
        # Inline rename state. Only one row can be renaming at a time.
        self._rename_id: Optional[int] = None
        self._rename_buf: str = ""
        self._focus_rename: bool = False
        # Isolate/solo state: the soloed layer and the visibility to restore on exit.
        self._solo_id: Optional[int] = None
        self._solo_saved: Dict[int, bool] = {}
        # Multi-selection. ``_anchor_id`` is the shift-click pivot; ``_synced_id`` is the
        # last HUD id this panel wrote, so an external change can be told from its own.
        self._selection: Set[int] = set()
        self._anchor_id: Optional[int] = None
        self._synced_id: Optional[int] = None
        # Name filter. Case-insensitive substring over label and type.
        self._filter: str = ""
        self._inspector_open: bool = True

    # -- draw ------------------------------------------------------------------

    def draw(self) -> None:
        """Render the panel body. The workspace owns the window."""
        if not IMGUI_AVAILABLE:
            return

        # Draw order, not list order: with any non-default zorder the two disagree and
        # scene.layers is the one that is wrong about the picture (layerops.draw_order).
        #
        # The 3D decoration is filtered out. The axis box, the back panes, the grid walls
        # and the tick marks are implemented as ordinary ``Layer3D`` objects — that is what
        # lets the renderer draw them with no special case — but they are *chrome the
        # engine added*, not data the user plotted. Listing them here put four rows the
        # user never created at the top of their layer list, offered to delete and reorder
        # them (both of which the next camera move silently undoes, because
        # ``ensure_3d_axes`` rebuilds them), and buried the actual data. They are
        # configured in the View 3D panel, which is where a user looks for "show the grid".
        ordered = [
            layer
            for layer in layerops.draw_order(self.plot)
            if layer.metadata.get("artist") not in SYSTEM_3D_ARTISTS
        ]
        self._sync_solo(ordered)
        self._sync_selection(ordered)
        self._draw_toolbar(ordered)
        imgui.separator()

        if not ordered:
            imgui.spacing()
            imgui.text_disabled("No layers yet.")
            imgui.text_disabled("Paste data with Ctrl+V, or write a function.")
            return

        self._draw_filter()
        rows = self._filtered(ordered)
        self._draw_list_caption(ordered, rows)
        self._draw_layer_list(rows)
        self._draw_inspector()

    def _draw_filter(self) -> None:
        """The search box. Filtering is display-only — it never touches the scene."""
        icon_inline("search", size=_ROW_ICON_SIZE)
        imgui.same_line()
        imgui.push_item_width(imgui.get_content_region_avail().x - 26.0)
        # buffer_length stays at its -1 default (CONTRACT §2.4).
        _, text = imgui.input_text("##filter", self._filter)
        imgui.pop_item_width()
        self._filter = text
        if self._filter:
            imgui.same_line()
            if icon_button("##clear_filter", "close", size=_ROW_ICON_SIZE, tooltip="Clear filter"):
                self._filter = ""

    def _filtered(self, ordered: Sequence[Any]) -> List[Any]:
        """The rows to show: a case-insensitive substring match on label and type."""
        needle = self._filter.strip().lower()
        if not needle:
            return list(ordered)
        return [
            layer
            for layer in ordered
            if needle in _display_label(layer).lower()
            or needle in str(getattr(layer, "layer_type", "")).lower()
        ]

    def _draw_list_caption(self, ordered: Sequence[Any], rows: Sequence[Any]) -> None:
        """The count line above the rows, and the honest version of "drag to reorder"."""
        total = len(ordered)
        if len(rows) != total:
            imgui.text_disabled(f"{len(rows)} of {total} layers")
        else:
            imgui.text_disabled(f"{total} layer{'s' if total != 1 else ''}")

        selected = len(self._selection)
        if selected > 1:
            imgui.same_line()
            imgui.text_disabled(f"- {selected} selected")

        imgui.same_line()
        if self._filter.strip():
            # Dragging while filtered would drop the row next to a neighbour that is not
            # its real neighbour, so the gesture is disabled rather than made surprising.
            imgui.text_disabled("- filtered")
        else:
            imgui.text_disabled("- top draws last")
        imgui.same_line()
        widgets.help_marker(
            "Rows are in render order: the bottom row draws first, the top row draws "
            "over it. Drag a row to reorder; that also renumbers every layer's Z-Order "
            "to match, so the list and the picture always agree. A filtered list cannot "
            "be dragged, because the row you drop next to may not be the real neighbour."
        )
        imgui.spacing()

    def _draw_layer_list(self, rows: Sequence[Any]) -> None:
        """The scrolling layer rows. Collects reorder intent; applies it after the loop."""
        to_move: Optional[Tuple[int, int]] = None

        reserve = _INSPECTOR_OPEN_H if self._inspector_open else _INSPECTOR_SHUT_H
        if not self._selection:
            reserve = _INSPECTOR_SHUT_H
        # Negative height means "remaining minus abs" (§2.6), which keeps the inspector
        # on screen instead of letting a long list push it below the panel.
        # end_child is unconditional per CONTRACT §2.6 — begin_child returning
        # not-visible still opens a scope that must be closed.
        child_visible = imgui.begin_child("##layer_list", (0.0, -reserve))
        if child_visible:
            draggable = not self._filter.strip()
            show_z = layerops.zorder_is_authoritative(self.plot)
            for row, layer in enumerate(rows):
                moved = self._draw_layer_row(layer, row, rows, draggable, show_z)
                if moved is not None:
                    to_move = moved
        imgui.end_child()

        if to_move is not None:
            self._move_layer(*to_move)

    def _draw_layer_row(
        self,
        layer: Any,
        row: int,
        rows: Sequence[Any],
        draggable: bool,
        show_z: bool,
    ) -> Optional[Tuple[int, int]]:
        """One layer row. Returns a ``(src_id, dst_id)`` reorder request, or None."""
        moved: Optional[Tuple[int, int]] = None
        layer_id = layer.layer_id
        imgui.push_id(str(layer_id))
        try:
            self._draw_visibility_toggle(layer)
            imgui.same_line()
            self._draw_color_chip(layer)
            imgui.same_line()
            # The type chip. icon_inline reserves its space with imgui.dummy, which is a
            # real item, so the tooltip attaches to it.
            icon_inline(_layer_type_icon(layer), size=_ROW_ICON_SIZE)
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"{_type_label(layer)} - right-click to change type")
            imgui.same_line()

            if self._rename_id == layer_id:
                self._draw_rename_field(layer)
            else:
                moved = self._draw_name_row(layer, row, rows, draggable, show_z)
        finally:
            # pop_id on every path: an unbalanced id stack corrupts every later widget
            # in the frame, not just this row.
            imgui.pop_id()
        return moved

    def _draw_visibility_toggle(self, layer: Any) -> None:
        """The eye. Hidden layers are skipped by both rendering and autoscale."""
        visible = bool(layer.style.visible)
        shape = "eye" if visible else "eye_off"
        tooltip = "Hide layer" if visible else "Show layer"
        if icon_button("##vis", shape, size=_ROW_ICON_SIZE, tooltip=tooltip):
            self._set_visible(layer, not visible)
            # A hand toggle invalidates the saved isolate state: restoring it later
            # would silently undo what the user just asked for.
            if self._solo_id is not None:
                self._solo_id = None
                self._solo_saved = {}

    def _draw_color_chip(self, layer: Any) -> None:
        """The colour swatch — editable only where the field is live (§4.4)."""
        rgba = _effective_color(layer)
        field = _color_field(layer)

        if field is None:
            imgui.color_button(
                "##color_ro",
                rgba,
                imgui.ColorEditFlags_.no_tooltip,
                (_CHIP_SIZE, _CHIP_SIZE),
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Per-element colour (baked per vertex).\n"
                    "Not editable here - recolour the data instead."
                )
            return

        changed, color = imgui.color_edit4(
            "##color",
            rgba,
            imgui.ColorEditFlags_.no_inputs
            | imgui.ColorEditFlags_.no_label
            | imgui.ColorEditFlags_.alpha_preview_half,
        )
        if changed:
            self._set_color(layer, field, _as_rgba(color))

    def _draw_name_row(
        self,
        layer: Any,
        row: int,
        rows: Sequence[Any],
        draggable: bool,
        show_z: bool,
    ) -> Optional[Tuple[int, int]]:
        """Name selectable + drag source/target + context menu + opacity + readouts."""
        moved: Optional[Tuple[int, int]] = None
        size_text = _layer_size_text(layer)
        z_text = f"z:{int(layer.style.zorder)}" if show_z else ""
        alpha_live = str(getattr(layer, "layer_type", "")) not in _ALPHA_DEAD_TYPES

        # Reserve every trailing widget's width so a long label cannot push them out of
        # view. calc_text_size is the only honest measure — the default font is
        # fixed-width, but the readouts are not fixed-length.
        avail = imgui.get_content_region_avail().x
        trailing = imgui.calc_text_size(size_text).x if size_text else 0.0
        if z_text:
            trailing += imgui.calc_text_size(z_text).x + 8.0
        if alpha_live:
            trailing += _ALPHA_WIDTH + 8.0
        trailing += _ROW_ICON_SIZE + 8.0  # the delete button, always drawn
        name_w = max(60.0, avail - trailing - 12.0)

        imgui.align_text_to_frame_padding()
        # The FIRST return value is the click; the second is a *toggled* selection state
        # that reads True on every idle frame while the row is selected (verified with
        # the §2.10 harness). ``hud.py:273`` unpacks the second and gets away with it only
        # because re-selecting the same id is idempotent — here it would re-fire selection
        # every frame and arm the double-click rename on clicks meant for other widgets.
        clicked, _ = imgui.selectable(
            f"{_display_label(layer)}##name",
            layer.layer_id in self._selection,
            imgui.SelectableFlags_.allow_double_click,
            (name_w, 0.0),
        )
        if clicked:
            self._click_select(layer, row, rows)
            if imgui.is_mouse_double_clicked(0):
                self._begin_rename(layer)

        if draggable:
            moved = self._draw_drag_drop(layer)
        self._draw_context_menu(layer)

        if alpha_live:
            imgui.same_line()
            self._draw_row_alpha(layer)
        if z_text:
            imgui.same_line()
            imgui.text_disabled(z_text)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Z-Order. Shown because these layers are ranked by it rather than "
                    "by list position. Drag a row to renumber them all to list order."
                )
        if size_text:
            imgui.same_line()
            imgui.text_disabled(size_text)
        imgui.same_line()
        self._draw_row_delete(layer)
        return moved

    def _draw_row_delete(self, layer: Any) -> None:
        """Delete this row's layer, from the row itself.

        The toolbar already had a trash, but it acts on the *selection* and is greyed out
        until there is one — so deleting one plot meant discovering that you must select
        it first. This is the button people look for: it names the layer it will delete
        and needs nothing selected.

        Deliberately **this layer only**, never the selection, even when the row is part of
        one: a trash sitting on a row promises to delete that row, and quietly taking four
        other layers with it would be the kind of surprise that undo exists to survive but
        nobody should need. The toolbar's trash is still the one for a whole selection, and
        its tooltip says how many it will take.
        """
        if icon_button(
            f"##del_{layer.layer_id}",
            "trash",
            size=_ROW_ICON_SIZE,
            tooltip=f"Delete '{_display_label(layer)}'",
        ):
            self._delete_layers([layer])

    def _draw_row_alpha(self, layer: Any) -> None:
        """Inline opacity. Applies to the whole selection when this row is part of it."""
        imgui.push_item_width(_ALPHA_WIDTH)
        changed, alpha = imgui.drag_float(
            "##alpha", float(layer.style.alpha), 0.005, 0.0, 1.0, "%.2f"
        )
        imgui.pop_item_width()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Opacity (drag). Applies to every selected layer.")
        if changed:
            targets = self._selected_layers()
            if layer.layer_id not in self._selection:
                targets = [layer]
            self._set_alpha(targets, float(alpha))

    def _draw_drag_drop(self, layer: Any) -> Optional[Tuple[int, int]]:
        """Drag-reorder, following the working pattern at ``hud.py:282-299``.

        The payload is the source ``layer_id`` rather than ``hud.py``'s row index: this
        request is applied a frame later in the drain, by which time an index may name a
        different layer. ``accept_drag_drop_payload`` must stay nested inside the target
        block — outside one it raises ``ImGuiError`` on an internal assert (§2.6).
        """
        moved: Optional[Tuple[int, int]] = None

        if imgui.begin_drag_drop_source():
            imgui.set_drag_drop_payload_py_id(_DND_TYPE, layer.layer_id)
            imgui.text(f"Moving {_display_label(layer)}...")
            imgui.end_drag_drop_source()

        if imgui.begin_drag_drop_target():
            payload = imgui.accept_drag_drop_payload_py_id(_DND_TYPE)
            if payload:
                src_id = payload.data_id
                if src_id is not None and src_id != layer.layer_id:
                    moved = (src_id, layer.layer_id)
            imgui.end_drag_drop_target()
        return moved

    def _draw_context_menu(self, layer: Any) -> None:
        """Right-click menu for the row."""
        popup_open = imgui.begin_popup_context_item("##row_menu")
        if not popup_open:
            return

        # A right-click on a row outside the selection acts on that row alone, which is
        # what every file manager does: the alternative silently deletes layers the user
        # is no longer looking at.
        if layer.layer_id not in self._selection:
            self._set_selection({layer.layer_id}, anchor=layer.layer_id)
        targets = self._selected_layers() or [layer]
        many = len(targets) > 1
        suffix = f" ({len(targets)} layers)" if many else ""

        if imgui.menu_item("Rename", "", False, not many)[0]:
            self._begin_rename(layer)
        if imgui.menu_item(f"Duplicate{suffix}", "", False)[0]:
            self._duplicate_layers(targets)

        self._draw_type_menu(layer, many)

        soloed = self._solo_id == layer.layer_id
        if imgui.menu_item("Exit isolate" if soloed else "Isolate", "", soloed, not many)[0]:
            self._toggle_solo(layer)

        visible = bool(layer.style.visible)
        if imgui.menu_item(f"{'Hide' if visible else 'Show'}{suffix}", "", False)[0]:
            self._set_visible_many(targets, not visible)

        imgui.separator()
        if imgui.menu_item(f"Delete{suffix}", "", False)[0]:
            self._delete_layers(targets)
        imgui.end_popup()

    def _draw_type_menu(self, layer: Any, many: bool) -> None:
        """The "Change type" submenu. Absent where a type change is not expressible."""
        kind = layerops.layer_kind(layer)
        kinds = _available_kinds()
        can_change = not many and _can_change_kind(layer)

        menu_open = imgui.begin_menu("Change type", can_change)
        if not menu_open:
            if not can_change and imgui.is_item_hovered():
                imgui.set_tooltip(
                    "This layer is not built from an x/y pair, so it cannot be "
                    "rebuilt as another plot type."
                )
            return
        for option in kinds:
            if imgui.menu_item(option, "", option == kind)[0] and option != kind:
                self._change_kind(layer, option)
        imgui.end_menu()

    def _draw_rename_field(self, layer: Any) -> None:
        """Inline rename. Enter commits, Escape cancels, click-away commits."""
        if self._focus_rename:
            imgui.set_keyboard_focus_here()
            self._focus_rename = False

        imgui.push_item_width(max(80.0, imgui.get_content_region_avail().x - 8.0))
        # buffer_length is left at its -1 default: an explicit length shorter than the
        # text silently truncates it (CONTRACT §2.4).
        entered, text = imgui.input_text(
            "##rename",
            self._rename_buf,
            flags=imgui.InputTextFlags_.enter_returns_true | imgui.InputTextFlags_.auto_select_all,
        )
        imgui.pop_item_width()
        self._rename_buf = text

        if entered:
            self._commit_rename(layer, text)
        elif imgui.is_item_deactivated():
            # Escape reverts imgui's own buffer, so the returned text is already the
            # original; check the key rather than trusting the value.
            if imgui.is_key_pressed(imgui.Key.escape):
                self._cancel_rename()
            else:
                self._commit_rename(layer, text)

    # -- toolbar ---------------------------------------------------------------

    def _draw_toolbar(self, ordered: Sequence[Any]) -> None:
        """Scene-wide actions. Buttons that need a layer are disabled without one."""
        has_layers = bool(ordered)
        targets = self._selected_layers()
        many = len(targets) > 1
        suffix = f" ({len(targets)} layers)" if many else ""

        if icon_button(
            "##autoscale", "target", tooltip="Autoscale to visible data", enabled=has_layers
        ):
            self.submit(lambda: self.plot.autoscale())
        imgui.same_line()
        if icon_button("##reset_view", "home", tooltip="Reset view"):
            self.submit(lambda: self.plot.reset_view())

        imgui.same_line()
        imgui.text_disabled("|")
        imgui.same_line()

        if icon_button(
            "##duplicate", "copy", tooltip=f"Duplicate selected{suffix}", enabled=bool(targets)
        ):
            self._duplicate_layers(targets)
        imgui.same_line()
        if icon_button(
            "##delete", "trash", tooltip=f"Delete selected{suffix}", enabled=bool(targets)
        ):
            self._delete_layers(targets)

        imgui.same_line()
        imgui.text_disabled("|")
        imgui.same_line()

        if icon_button("##clear", "close", tooltip="Clear scene", enabled=has_layers):
            imgui.open_popup(_CLEAR_POPUP)

        if self._solo_id is not None:
            imgui.same_line()
            if icon_button("##unsolo", "eye", tooltip="Exit isolate (show all)", active=True):
                self._exit_solo()

        # Must be drawn every frame, not only on the click that opens it.
        if confirm_popup(
            _CLEAR_POPUP,
            "Remove every layer from the scene?\n\nThis cannot be undone, and it clears "
            "the undo history.",
        ):
            self._clear_scene()

    # -- inspector -------------------------------------------------------------

    def _draw_inspector(self) -> None:
        """Details for the selection: type, bounds, stats, opacity, colormap, SSAO."""
        targets = self._selected_layers()
        if not targets:
            return

        imgui.separator()
        self._inspector_open = widgets.section("Inspector")
        if not self._inspector_open:
            return

        if len(targets) > 1:
            self._draw_multi_inspector(targets)
            return

        layer = targets[0]
        child_visible = imgui.begin_child("##inspector", (0.0, 0.0))
        if child_visible:
            imgui.push_id(str(layer.layer_id))
            try:
                self._draw_inspector_identity(layer)
                self._draw_inspector_bounds(layer)
                self._draw_inspector_colormap(layer)
                self._draw_inspector_ssao(layer)
            finally:
                imgui.pop_id()
        imgui.end_child()

    def _draw_multi_inspector(self, targets: Sequence[Any]) -> None:
        """The multi-selection body: one opacity control and the combined bounds."""
        imgui.text_disabled(f"{len(targets)} layers selected")
        alpha_targets = [
            layer
            for layer in targets
            if str(getattr(layer, "layer_type", "")) not in _ALPHA_DEAD_TYPES
        ]
        if alpha_targets:
            shared = float(alpha_targets[0].style.alpha)
            changed, alpha = widgets.labeled_slider_float("Opacity", shared, 0.0, 1.0)
            if changed:
                self._set_alpha(alpha_targets, alpha)

        boxes = [layerops.layer_bounds(self.plot, layer) for layer in targets]
        boxes = [b for b in boxes if b is not None]
        if boxes:
            arr = np.asarray(boxes, dtype=np.float64)
            widgets.stat_row(
                "X range", f"{_fmt_num(arr[:, 0].min())} .. {_fmt_num(arr[:, 1].max())}"
            )
            widgets.stat_row(
                "Y range", f"{_fmt_num(arr[:, 2].min())} .. {_fmt_num(arr[:, 3].max())}"
            )

    def _draw_inspector_identity(self, layer: Any) -> None:
        """Type chip, change-type combo and z-order."""
        imgui.text_disabled(
            f"{_type_label(layer)}   ({getattr(layer, 'layer_type', '?')})   "
            f"id {getattr(layer, 'layer_id', '?')}"
        )

        kind = layerops.layer_kind(layer)
        if _can_change_kind(layer):
            changed, option = widgets.enum_combo(
                "Plot type",
                kind or "",
                _available_kinds(),
                help=(
                    "Rebuilds the layer as another plot type, keeping its colour, "
                    "width, opacity, z-order, list position, name and selection. The "
                    "layer's internal id changes, so anything holding the old id is "
                    "re-pointed for you. Undoable."
                ),
            )
            if changed:
                self._change_kind(layer, option)
            else:
                self._draw_kind_options(layer)

        changed, zorder = imgui.drag_int("Z-Order", int(layer.style.zorder), 0.1, -100, 100)
        imgui.same_line()
        widgets.help_marker(
            "Higher draws on top. Editing this re-ranks the list above, because the "
            "list is shown in render order."
        )
        if changed:
            self._set_zorder(layer, int(zorder))

    def _draw_kind_options(self, layer: Any) -> None:
        """The selected layer's per-kind parameters, re-derived as they change.

        Without this the type change is only half an answer: it hands you a ``bar``, and
        the width that decides whether the bars touch or barely show was a number only
        the Data panel could reach -- and only for layers it had created itself.

        Deferred and **not undoable**, matching :meth:`_set_alpha` and
        :meth:`_set_zorder`: these are spinboxes the user steps through, and one history
        entry per intermediate value would bury every real edit under the sweep. The
        rebuild is the same ``replot_layer_xy`` a type change runs, so the layer id
        changes here too and the dataset is re-pointed the same way.
        """
        opts = layerops.layer_kind_options(layer)
        if not opts:
            return
        if not widgets.kind_options_editor(opts):
            return

        kind = layerops.layer_kind(layer)
        if kind is None:  # pragma: no cover - gated by _can_change_kind
            return

        plot, hud = self.plot, self.hud

        def apply() -> None:
            data = layerops.layer_source_xy(layer)
            if data is None:  # pragma: no cover - type changed under us
                return
            old_id = getattr(layer, "layer_id", None)
            new_layer = layerops.replot_layer_xy(
                plot,
                hud,
                layer,
                data[0],
                data[1],
                kind=kind,
                label=getattr(layer, "label", None),
                options=opts,
            )
            new_id = getattr(new_layer, "layer_id", None)
            self._rebind_dataset(old_id, new_id)
            if old_id in self._selection:
                self._selection.discard(old_id)
                self._selection.add(new_id)
            if self._anchor_id == old_id:
                self._anchor_id = new_id
            if self._solo_id == old_id:
                self._solo_id = new_id

        self.submit(apply)

    def _draw_inspector_bounds(self, layer: Any) -> None:
        """Extent and element count — what autoscale would frame for this layer."""
        count = layerops.layer_element_count(layer)
        if count is not None:
            widgets.stat_row("Elements", f"{count:,}")

        bounds = layerops.layer_bounds(self.plot, layer)
        if bounds is None:
            # Honest, not an error: a text layer has no intrinsic bounds by design
            # (§1.9) and never participates in autoscale.
            imgui.text_disabled("No bounds (this layer does not autoscale).")
            return
        xmin, xmax, ymin, ymax = bounds
        widgets.stat_row("X range", f"{_fmt_num(xmin)} .. {_fmt_num(xmax)}")
        widgets.stat_row("Y range", f"{_fmt_num(ymin)} .. {_fmt_num(ymax)}")
        widgets.stat_row("Extent", f"{_fmt_num(xmax - xmin)} x {_fmt_num(ymax - ymin)}")

    def _draw_inspector_colormap(self, layer: Any) -> None:
        """Per-layer colormap. Two mechanisms, two namespaces — never mixed (§4.5)."""
        kind = layerops.layer_colormap_kind(layer)
        if kind is None:
            return
        if kind == "gl_line":
            self._draw_gl_colormap(layer)
            return

        if not widgets.section("Colormap"):
            return
        cmap, vmin, vmax = layerops.layer_colormap(layer)
        changed, name = widgets.enum_combo("Map", cmap or "viridis", layerops.CPU_COLORMAP_NAMES)
        if changed:
            self._set_colormap(layer, cmap=name)

        auto = vmin is None and vmax is None
        changed, want_auto = imgui.checkbox("Auto range", auto)
        imgui.same_line()
        widgets.help_marker(
            "Auto maps the colours across the data's full range. Turn it off to pin the "
            "range, so several layers can share one scale."
        )
        if changed:
            if want_auto:
                self._set_colormap(layer, vmin=None, vmax=None)
            else:
                # Seed from the data rather than 0..1: the first keystroke into an
                # unseeded field would silently rescale the whole layer.
                data_range = layerops.layer_colormap_data_range(layer)
                lo, hi = data_range if data_range is not None else (0.0, 1.0)
                self._set_colormap(layer, vmin=lo, vmax=hi)

        if not auto:
            data_range = layerops.layer_colormap_data_range(layer)
            lo, hi = data_range if data_range is not None else (0.0, 1.0)
            changed_lo, new_lo = imgui.input_float("vmin", float(vmin if vmin is not None else lo))
            if changed_lo:
                self._set_colormap(layer, vmin=float(new_lo))
            changed_hi, new_hi = imgui.input_float("vmax", float(vmax if vmax is not None else hi))
            if changed_hi:
                self._set_colormap(layer, vmax=float(new_hi))

    def _draw_gl_colormap(self, layer: Any) -> None:
        """The shader colormap override for a polyline / line family.

        Three-state on purpose. This colormap is keyed on the *vertex id*, not on data
        values, and it is scene-global by default (``options.line_colormap_enabled``);
        "Follow scene" is that default and is what keeps the global toggle working.
        """
        if not widgets.section("Colormap (shader)"):
            return
        enabled, index = layerops.layer_gl_colormap(layer)
        modes = ("Follow scene", "On", "Off")
        current = modes[0] if enabled is None else (modes[1] if enabled else modes[2])
        changed, mode = widgets.enum_combo(
            "Mode",
            current,
            modes,
            help=(
                "Colours the line along its length by vertex index - it maps position, "
                "not data values. 'Follow scene' uses the scene-wide toggle in Style; "
                "'On'/'Off' override it for this layer only."
            ),
        )
        if changed:
            self._set_gl_colormap(layer, enabled=None if mode == modes[0] else mode == modes[1])

        names = layerops.GL_COLORMAP_NAMES
        follow = index is None
        scheme = int(index) if index is not None else int(self.plot.options.density_scheme_index)
        scheme = max(0, min(scheme, len(names) - 1))
        changed, name = widgets.enum_combo(
            "Scheme", f"{names[scheme]}{' (scene)' if follow else ''}", names
        )
        if changed:
            self._set_gl_colormap(layer, scheme_index=names.index(name))
        if not follow:
            if imgui.small_button("Follow scene scheme"):
                self._set_gl_colormap(layer, scheme_index=None)

    def _draw_inspector_ssao(self, layer: Any) -> None:
        """Per-layer ambient occlusion — live at ``geometry3d.py:197-198``, never exposed."""
        if not str(getattr(layer, "layer_type", "")).endswith("3d"):
            return
        if not widgets.section("Ambient Occlusion"):
            return

        metadata = getattr(layer, "metadata", {}) or {}
        ssao = getattr(getattr(self.plot.options.visual, "ssao", None), "enabled", False)
        forced = bool(metadata.get("ssao", False))

        changed, value = imgui.checkbox("Force on for this layer", forced)
        imgui.same_line()
        widgets.help_marker(
            "geometry3d ORs this with the scene's SSAO switch, so a layer can opt IN "
            "while the scene is off, but cannot opt OUT while the scene is on. "
            "Unticking this does not disable SSAO if the scene has it enabled."
        )
        if changed:
            self._set_ssao(layer, enabled=value)

        if ssao and not forced:
            imgui.text_disabled("Scene SSAO is on; this layer is shaded regardless.")

        default_strength = float(
            getattr(getattr(self.plot.options.visual, "ssao", None), "strength", 0.45)
        )
        override = metadata.get("ssao_strength")
        strength = float(override if override is not None else default_strength)
        changed, value = widgets.labeled_slider_float("Strength", strength, 0.0, 1.0)
        if changed:
            self._set_ssao(layer, strength=value)
        if override is not None:
            imgui.same_line()
            if imgui.small_button("Reset##ssao_strength"):
                self._set_ssao(layer, strength=None)

    # -- selection -------------------------------------------------------------

    def _selected_layer_id(self) -> Optional[int]:
        """The selected layer id, preferring the HUD's copy (it drives the sync block)."""
        hud = self.hud
        state = getattr(hud, "state", None)
        if state is not None:
            return getattr(state, "selected_layer_id", None)
        interaction = getattr(self.plot, "interaction", None)
        return getattr(interaction, "selected_layer_id", None)

    def _selected_layer(self) -> Optional[Any]:
        """The primary selected layer object, or None if nothing is selected."""
        return layerops.find_layer(self.plot, self._selected_layer_id())

    def _selected_layers(self) -> List[Any]:
        """Every selected layer, in draw order. Empty when nothing is selected."""
        return [
            layer for layer in layerops.draw_order(self.plot) if layer.layer_id in self._selection
        ]

    def _sync_selection(self, ordered: Sequence[Any]) -> None:
        """Reconcile the panel's set with the HUD's single id and with the live scene.

        The engine's picking and the other panels only know one selected id. When that
        id changes underneath us — a click in the viewport, a Style-panel layer combo —
        it is a *new* selection and the set collapses to it. Distinguishing that from
        this panel's own writes is what ``_synced_id`` is for; without it, every frame
        after a multi-select would see "hud id is not the whole set" and fight itself.
        """
        live = {layer.layer_id for layer in ordered}
        self._selection &= live
        if self._anchor_id not in live:
            self._anchor_id = None

        current = self._selected_layer_id()
        if current != self._synced_id:
            self._synced_id = current
            self._selection = {current} if current in live else set()
            self._anchor_id = current if current in live else None

    def _click_select(self, layer: Any, row: int, rows: Sequence[Any]) -> None:
        """Apply the click's modifiers: plain replaces, ctrl/cmd toggles, shift extends."""
        io = imgui.get_io()
        # key_super is Cmd on macOS, where Cmd-click is the platform's toggle gesture.
        toggle = bool(io.key_ctrl or io.key_super)
        extend = bool(io.key_shift)

        if extend and self._anchor_id is not None:
            anchor_row = next(
                (i for i, la in enumerate(rows) if la.layer_id == self._anchor_id), None
            )
            if anchor_row is not None:
                lo, hi = sorted((anchor_row, row))
                span = {la.layer_id for la in rows[lo : hi + 1]}
                # Extend from the anchor without dropping what ctrl-click added, which is
                # what makes ctrl and shift composable rather than mutually exclusive.
                self._set_selection(self._selection | span, anchor=self._anchor_id)
                return

        if toggle:
            selection = set(self._selection)
            if layer.layer_id in selection:
                selection.discard(layer.layer_id)
            else:
                selection.add(layer.layer_id)
            self._set_selection(selection, anchor=layer.layer_id)
            return

        self._set_selection({layer.layer_id}, anchor=layer.layer_id)

    def _set_selection(self, ids: Set[int], *, anchor: Optional[int] = None) -> None:
        """Set the selection and push its primary id to the HUD."""
        self._selection = set(ids)
        self._anchor_id = anchor
        primary = anchor if anchor in self._selection else next(iter(self._selection), None)
        self._select_id(primary)

    def _select(self, layer: Any) -> None:
        """Select a single layer (the pre-multi-select entry point)."""
        self._set_selection({layer.layer_id}, anchor=layer.layer_id)

    def _select_id(self, layer_id: Optional[int]) -> None:
        """Write the primary id to the HUD.

        Writes ``hud.state.selected_layer_id`` and lets the existing sync block
        (``hud.py:122-136``) propagate it to ``plot.interaction`` on the next frame —
        writing both here would make the block read the change as an *engine*-side pick
        and stop honouring later UI selections. Selection is not scene state, so this
        needs no queue.
        """
        self._synced_id = layer_id
        hud = self.hud
        state = getattr(hud, "state", None)
        if state is not None:
            state.selected_layer_id = layer_id
            return
        interaction = getattr(self.plot, "interaction", None)
        if interaction is not None:
            interaction.selected_layer_id = layer_id

    # -- mutations (all deferred) ---------------------------------------------

    def _set_visible(self, layer: Any, visible: bool) -> None:
        """Toggle visibility, deferred."""
        plot = self.plot
        self.submit(lambda: layerops.set_layer_style(plot, layer, visible=visible))

    def _set_visible_many(self, layers: Sequence[Any], visible: bool) -> None:
        """Toggle visibility on a whole selection, deferred as one command."""
        plot = self.plot
        targets = list(layers)

        def apply() -> None:
            for layer in targets:
                layerops.set_layer_style(plot, layer, visible=visible)

        self.submit(apply)

    def _set_color(self, layer: Any, field: str, rgba: Tuple[float, float, float, float]) -> None:
        """Set the live colour field, deferred.

        Not undoable on purpose: ``color_edit4`` reports a change on every frame of a
        drag, so recording it would push hundreds of entries for one gesture.
        ``set_layer_style`` handles the 3D ``gpu_dirty`` re-upload trap for us.
        """
        plot = self.plot
        self.submit(lambda: layerops.set_layer_style(plot, layer, **{field: rgba}))

    def _set_alpha(self, layers: Sequence[Any], alpha: float) -> None:
        """Set opacity across a selection, deferred. Not undoable (a per-frame drag)."""
        plot = self.plot
        targets = list(layers)
        value = float(np.clip(alpha, 0.0, 1.0))

        def apply() -> None:
            for layer in targets:
                layerops.set_layer_style(plot, layer, alpha=value)

        self.submit(apply)

    def _set_zorder(self, layer: Any, zorder: int) -> None:
        """Set an explicit z-order, deferred. Not undoable (a per-frame drag)."""
        plot = self.plot
        self.submit(lambda: layerops.set_layer_style(plot, layer, zorder=int(zorder)))

    def _set_colormap(self, layer: Any, **fields: Any) -> None:
        """Set the data colormap, deferred.

        Routed through ``layerops.set_layer_colormap`` rather than ``set_layer_style``.
        That is not a preference: ``LayerStyle`` *has* ``cmap``/``vmin``/``vmax`` fields,
        so ``set_layer_style(cmap=...)`` is accepted and writes them — and no renderer
        reads them (§4.1). The call succeeds and the picture never changes.
        """
        plot = self.plot
        self.submit(lambda: layerops.set_layer_colormap(plot, layer, **fields))

    def _set_gl_colormap(self, layer: Any, **fields: Any) -> None:
        """Set the shader colormap override, deferred."""
        plot = self.plot
        self.submit(lambda: layerops.set_layer_gl_colormap(plot, layer, **fields))

    def _set_ssao(self, layer: Any, **fields: Any) -> None:
        """Set the per-layer SSAO override, deferred."""
        plot = self.plot
        self.submit(lambda: layerops.set_layer_ssao(plot, layer, **fields))

    def _move_layer(self, src_id: int, dst_id: int) -> None:
        """Reorder, deferred. Indices are resolved at drain time from the ids.

        ``move_layer_to_index``, not ``move_layer``: the latter moves the row in
        ``scene.layers`` and leaves ``style.zorder`` alone, so the renderer — which sorts
        by zorder — keeps drawing the old order and the drag appears to do nothing.
        """
        plot = self.plot

        def apply() -> None:
            ordered = layerops.draw_order(plot)
            dst = next((i for i, la in enumerate(ordered) if la.layer_id == dst_id), -1)
            if dst < 0:
                return  # A layer went away between the drop and the drain.
            layerops.move_layer_to_index(plot, src_id, dst)

        self.submit(apply)

    def _duplicate_layers(self, layers: Sequence[Any]) -> None:
        """Duplicate one or more layers, undoably, as a single history entry."""
        targets = [layer for layer in layers if layer is not None]
        if not targets:
            return
        plot = self.plot
        hud = self.hud
        label = (
            f"Duplicate {len(targets)} layers"
            if len(targets) > 1
            else f"Duplicate {_display_label(targets[0])}"
        )
        # The clones only exist once ``do`` has run, so ``undo`` cannot close over them
        # directly. Redo makes *new* clones, which is fine: do/undo/do lands on the same
        # state, which is all Command requires.
        made: Dict[str, List[Any]] = {}

        def do() -> None:
            made["clones"] = [layerops.duplicate_layer(plot, layer) for layer in targets]

        def undo() -> None:
            for clone in made.pop("clones", []):
                layerops.remove_layer(plot, hud, clone)

        self.push_command(Command(label=label, do=do, undo=undo))

    def _delete_layers(self, layers: Sequence[Any]) -> None:
        """Delete layers through the full §1.6 teardown, undoably where that is honest.

        Undo re-inserts the very same layer objects at their old indices. That is sound
        because ``remove_layer`` sets ``layer._gl = None`` and every renderer recreates
        its buffers when ``_gl`` is None (``scatter.py:135-137``, ``polyline.py:163-165``),
        so a layer simply re-uploads on its next draw.

        The engine's ``_primary_line_layer`` is the exception: removing it also resets
        ``scene.lines`` and ``_cpu_line_copy``, which feed adaptive alpha and picking.
        Re-inserting the layer would not rebuild those, so the undo would be a quiet lie
        and the whole command is marked not-undoable instead — which makes the stack drop
        the history rather than let a later undo replay across the gap.
        """
        targets = [layer for layer in layers if layer is not None]
        if not targets:
            return
        plot = self.plot
        hud = self.hud
        label = (
            f"Delete {len(targets)} layers"
            if len(targets) > 1
            else f"Delete {_display_label(targets[0])}"
        )

        # Indices are captured now, in the draw callback, and restored in reverse order
        # so each insert lands before the list has shifted under it.
        indexed = []
        for layer in targets:
            index = next(
                (i for i, la in enumerate(plot.scene.layers) if la is layer),
                len(plot.scene.layers),
            )
            indexed.append((index, layer))
        indexed.sort(key=lambda pair: pair[0])
        primary = getattr(plot, "_primary_line_layer", None)
        has_primary = any(layer is primary for _, layer in indexed)

        def do() -> None:
            for _, layer in indexed:
                if self._solo_id == layer.layer_id:
                    self._solo_id = None
                    self._solo_saved = {}
                self._selection.discard(layer.layer_id)
                layerops.remove_layer(plot, hud, layer)

        if has_primary:
            self.push_command(Command.not_undoable(label, do))
            return

        def undo() -> None:
            layers_list = plot.scene.layers
            for index, layer in indexed:
                layers_list.insert(min(index, len(layers_list)), layer)
                layer.dirty.gpu_dirty = True
            layerops.mark_scene_dirty(plot)

        self.push_command(Command(label=label, do=do, undo=undo))

    def _change_kind(self, layer: Any, kind: str) -> None:
        """Rebuild a layer as another plot type, undoably.

        Delegates to ``layerops.replot_layer_xy``, which owns the delete+recreate:
        ``layer_type`` is baked in at construction and there is no conversion API, so a
        type change is necessarily a new layer. It carries the style, translation, list
        position and selection across; the **``layer_id`` necessarily changes**, and
        re-pointing whatever held the old one is documented as the caller's job — which
        is why :meth:`_rebind_dataset` runs here.

        Undo is the same operation aimed back at the old kind. ``state`` tracks the live
        object across each swap, because every swap replaces it.
        """
        if layerops.layer_source_xy(layer) is None:
            return
        old_kind = layerops.layer_kind(layer)
        if old_kind is None or old_kind == kind:
            return

        plot = self.plot
        hud = self.hud
        state: Dict[str, Any] = {"layer": layer}
        label = _display_label(layer)

        def swap(target_kind: str):
            def run() -> None:
                current = state["layer"]
                # The columns it was plotted from, never its geometry: a bar's vertices
                # are its corners and a step's pts are its staircase, so rebuilding off
                # those would plot the scaffolding and stair an already-stepped line.
                data = layerops.layer_source_xy(current)
                if data is None:  # pragma: no cover - type changed under us
                    return
                old_id = getattr(current, "layer_id", None)
                new_layer = layerops.replot_layer_xy(
                    plot,
                    hud,
                    current,
                    data[0],
                    data[1],
                    kind=target_kind,
                    label=getattr(current, "label", None),
                )
                state["layer"] = new_layer
                new_id = getattr(new_layer, "layer_id", None)
                self._rebind_dataset(old_id, new_id)
                if old_id in self._selection:
                    self._selection.discard(old_id)
                    self._selection.add(new_id)
                if self._anchor_id == old_id:
                    self._anchor_id = new_id
                if self._solo_id == old_id:
                    self._solo_id = new_id

            return run

        self.push_command(
            Command(
                label=f"Change {label} to {kind}",
                do=swap(kind),
                undo=swap(old_kind),
            )
        )

    def _rebind_dataset(self, old_id: Optional[int], new_id: Optional[int]) -> None:
        """Re-point the DataSet that was bound to ``old_id`` at ``new_id``.

        A type change gives the layer a new id, and ``DataSet.layer_id`` is a plain int:
        left alone it names a layer that no longer exists, and the table silently stops
        driving the plot while still looking bound.
        """
        if old_id is None or new_id is None or old_id == new_id:
            return
        store = self.store
        if store is None:
            return
        dataset = store.get_by_layer_id(old_id)
        if dataset is not None:
            dataset.layer_id = new_id

    def _clear_scene(self) -> None:
        """Remove every layer, deferred.

        Routed through ``layerops.gui_clear`` — ``plot.clear()`` is forbidden (§1.8): it
        rebinds a fresh ``SceneData``, leaking every GPU buffer and dangling
        ``_primary_line_layer``.

        Not undoable: restoring the layer list is easy, but ``gui_clear`` also resets
        ``scene.lines``, ``_cpu_line_copy``, ``_primary_line_layer`` and ``picked_info``,
        and a partial restore that looks complete is worse than an honest refusal. The
        confirmation modal is what stands in for undo here, and ``not_undoable`` makes
        the stack drop a history whose entries now point at removed layers.
        """
        plot = self.plot
        hud = self.hud

        def do() -> None:
            self._solo_id = None
            self._solo_saved = {}
            self._selection = set()
            self._anchor_id = None
            layerops.gui_clear(plot, hud)

        self.push_command(Command.not_undoable("Clear scene", do))

    # -- isolate / solo --------------------------------------------------------

    def _sync_solo(self, layers: Sequence[Any]) -> None:
        """Drop isolate state whose layer has left the scene, so the eye button clears."""
        if self._solo_id is None:
            return
        if not any(layer.layer_id == self._solo_id for layer in layers):
            self._solo_id = None
            self._solo_saved = {}

    def _toggle_solo(self, layer: Any) -> None:
        """Isolate a layer, or exit isolate if it is already the soloed one."""
        if self._solo_id == layer.layer_id:
            self._exit_solo()
            return

        plot = self.plot
        target_id = layer.layer_id
        # Snapshot before the queued command runs: this is the pre-isolate truth, and
        # by drain time everything is already hidden.
        self._solo_saved = {la.layer_id: bool(la.style.visible) for la in plot.scene.layers}
        self._solo_id = target_id

        def isolate() -> None:
            for other in plot.scene.layers:
                layerops.set_layer_style(plot, other, visible=other.layer_id == target_id)

        self.submit(isolate)

    def _exit_solo(self) -> None:
        """Restore the visibility captured when isolate was entered."""
        plot = self.plot
        saved = dict(self._solo_saved)
        self._solo_id = None
        self._solo_saved = {}

        def restore() -> None:
            for layer in plot.scene.layers:
                if layer.layer_id in saved:
                    layerops.set_layer_style(plot, layer, visible=saved[layer.layer_id])

        self.submit(restore)

    # -- rename ----------------------------------------------------------------

    def _begin_rename(self, layer: Any) -> None:
        """Enter inline rename on a row."""
        self._rename_id = layer.layer_id
        self._rename_buf = str(getattr(layer, "label", "") or "")
        self._focus_rename = True

    def _cancel_rename(self) -> None:
        """Leave rename without touching the layer."""
        self._rename_id = None
        self._rename_buf = ""
        self._focus_rename = False

    def _commit_rename(self, layer: Any, text: str) -> None:
        """Commit a rename, undoably. An empty or unchanged name is a no-op."""
        new = str(text).strip()
        old = str(getattr(layer, "label", "") or "")
        self._cancel_rename()
        if not new or new == old:
            return

        plot = self.plot

        def do() -> None:
            layerops.set_layer_style(plot, layer, label=new)

        def undo() -> None:
            layerops.set_layer_style(plot, layer, label=old)

        self.push_command(Command(label=f"Rename to {new}", do=do, undo=undo))
