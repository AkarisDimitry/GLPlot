"""The Pipeline panel: chain signal transforms with click-and-drag.

Where Math Lab applies *one* operation at a time, this panel snaps several together into a
reusable chain -- ``baseline -> smooth -> normalize``, say -- and previews the whole chain
live before committing it. Steps are added by dragging (or clicking) a type out of the
palette, reordered by dragging one over another, and configured in place. The chain runs
top to bottom, each step feeding the next.

All of the numerics and the chain logic live in :mod:`glplot.gui.pipeline`, which is pure
and headless-testable; this module is only the imgui shell -- the drag-and-drop, the param
editors, the preview and the Apply command. As with every panel it never mutates the live
scene from ``draw`` (CONTRACT 1.1): Apply is queued through the CommandQueue and recorded
on the UndoStack, exactly as Math Lab does.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from glplot.gui import icons, layerops, pipeline, theme, widgets
from glplot.gui.datasets import Column, DataSet
from glplot.gui.history import Command
from glplot.gui.panels.base import Panel

try:
    import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - exercised only on GL-less systems
    IMGUI_AVAILABLE = False
    imgui = None

logger = logging.getLogger(__name__)

__all__ = ["PipelinePanel"]

_PREVIEW_HEIGHT = 160.0

# Drag-and-drop payload types. ADD carries a step key (palette -> chain); MOVE carries a
# source index (reorder within the chain).
_PAYLOAD_ADD = "glplot_pipe_add"
_PAYLOAD_MOVE = "glplot_pipe_move"

# Geometry a layer may expose, probed in this order (ab before pts, as datasets.py does).
_LAYER_GEOMETRY: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("ab", ("a", "b")),
    ("pts", ("x", "y")),
    ("vertices", ("x", "y", "z")),
)
_LAYER_UNSUPPORTED = frozenset({"text", "patch", "semantic_line", "semantic_span"})

_APPLY_MODES: Tuple[Tuple[str, str], ...] = (
    ("new_layer", "New layer"),
    ("new_dataset", "New dataset"),
)


class PipelinePanel(Panel):
    """Build and run an ordered chain of Math Lab signal transforms, with a live preview."""

    title = "Pipeline"
    icon = "link"
    default_open = False

    def __init__(self, ws: Any) -> None:
        super().__init__(ws)
        self._source_kind = "dataset"
        self._ds_name = ""
        self._ds_x = ""
        self._ds_y = ""
        self._layer_id: Optional[int] = None
        self._layer_x = ""
        self._layer_y = ""

        #: The chain, top to bottom.
        self._steps: List[pipeline.PipelineStep] = []
        #: A structural edit deferred to after the step loop -- mutating the list while
        #: iterating it (from a drag drop) would corrupt the very loop drawing it.
        self._pending: Optional[Tuple[Any, ...]] = None

        self._apply_mode = "new_layer"
        self._layer_kind = "line"
        self._layer_opts: Dict[str, Any] = layerops.default_kind_options(self._layer_kind)
        self._apply_name = ""
        self._notice: Optional[str] = None

    # -- lifecycle -----------------------------------------------------------------

    def draw(self) -> None:
        if not IMGUI_AVAILABLE:
            return
        self._draw_notice()
        source = self._draw_source()
        imgui.separator()
        if source is None:
            return
        x, y, label = source

        self._draw_palette()
        self._draw_steps()
        self._apply_pending()

        result = pipeline.run_pipeline(x, y, self._steps)
        imgui.separator()
        self._draw_preview(x, y, label, result)
        if self._steps and result.ok and result.y.size:
            imgui.separator()
            self._draw_apply(result, label)

    def _draw_notice(self) -> None:
        if self._notice is None:
            return
        widgets.error_box(self._notice)
        if icons.icon_button("##pipe_dismiss", "close", size=18.0, tooltip="Dismiss"):
            self._notice = None

    # -- source --------------------------------------------------------------------

    def _draw_source(self) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
        """Draw the source selector and resolve it to ``(x, y, label)`` or None."""
        imgui.push_id("pipe_source")
        try:
            imgui.text("Source")
            imgui.same_line()
            widgets.help_marker(
                "The signal the chain runs on: two columns of a dataset, or the geometry "
                "of a plotted layer."
            )
            imgui.push_id("kind")
            try:
                if imgui.radio_button("Dataset", self._source_kind == "dataset"):
                    self._source_kind = "dataset"
                imgui.same_line()
                if imgui.radio_button("Plotted layer", self._source_kind == "layer"):
                    self._source_kind = "layer"
            finally:
                imgui.pop_id()

            imgui.push_item_width(-140.0)
            try:
                if self._source_kind == "dataset":
                    return self._resolve_dataset()
                return self._resolve_layer()
            finally:
                imgui.pop_item_width()
        finally:
            imgui.pop_id()

    def _resolve_dataset(self) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
        names = self.store.names()
        if not names:
            widgets.error_box("No datasets yet. Paste data or create one in the Data panel.")
            return None
        if self._ds_name not in names:
            self._ds_name = names[0]
        _c, self._ds_name = widgets.enum_combo("Dataset", self._ds_name, names)
        dataset = self.store.get(self._ds_name)
        if dataset is None or not dataset.column_names() or dataset.n_rows() == 0:
            widgets.error_box("Select a dataset with rows.")
            return None
        columns = dataset.column_names()
        if self._ds_x not in columns:
            self._ds_x = columns[0]
        if self._ds_y not in columns:
            self._ds_y = columns[1] if len(columns) > 1 else columns[0]
        _cx, self._ds_x = widgets.enum_combo("X column", self._ds_x, columns)
        _cy, self._ds_y = widgets.enum_combo("Y column", self._ds_y, columns)
        x = dataset.get(self._ds_x)
        y = dataset.get(self._ds_y)
        if x is None or y is None:
            widgets.error_box("Pick an x and a y column.")
            return None
        return (
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            f"{dataset.name}.{self._ds_y}",
        )

    def _resolve_layer(self) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
        candidates: List[Tuple[Any, Tuple[str, List[str], np.ndarray]]] = []
        for layer in self.plot.scene.layers:
            table = _layer_table(layer)
            if table is not None:
                candidates.append((layer, table))
        if not candidates:
            widgets.error_box("No layer in the scene has tabular x/y data to operate on.")
            return None
        entries = [_layer_display(layer) for layer, _t in candidates]
        current = ""
        for (layer, _t), entry in zip(candidates, entries):
            if getattr(layer, "layer_id", None) == self._layer_id:
                current = entry
                break
        if not current:
            current = entries[0]
            self._layer_id = getattr(candidates[0][0], "layer_id", None)
        _c, chosen = widgets.enum_combo("Layer", current, entries)
        index = entries.index(chosen) if chosen in entries else 0
        layer, (attr, cols, array) = candidates[index]
        self._layer_id = getattr(layer, "layer_id", None)
        if self._layer_x not in cols:
            self._layer_x = cols[0]
        if self._layer_y not in cols:
            self._layer_y = cols[1]
        _cx, self._layer_x = widgets.enum_combo("X column", self._layer_x, cols)
        _cy, self._layer_y = widgets.enum_combo("Y column", self._layer_y, cols)
        x = np.asarray(array[:, cols.index(self._layer_x)], dtype=np.float64)
        y = np.asarray(array[:, cols.index(self._layer_y)], dtype=np.float64)
        return x, y, str(getattr(layer, "label", None) or "layer")

    # -- palette -------------------------------------------------------------------

    def _draw_palette(self) -> None:
        """The addable step types. Each is a drag source and a click-to-add button."""
        imgui.text("Add step")
        imgui.same_line()
        widgets.help_marker(
            "Drag a step into the chain below, or click to append it. Drag steps over one "
            "another to reorder. The chain runs top to bottom."
        )
        avail = max(1.0, imgui.get_content_region_available()[0])
        per_row = max(1, int(avail // 96.0))
        for i, key in enumerate(pipeline.STEP_KEYS):
            step_type = pipeline.STEP_TYPES[key]
            if i % per_row != 0:
                imgui.same_line()
            if imgui.button(step_type.label, 90.0, 0.0):
                self._steps.append(pipeline.new_step(key))
            if imgui.begin_drag_drop_source():
                imgui.set_drag_drop_payload(_PAYLOAD_ADD, key.encode("ascii"))
                imgui.text(step_type.label)
                imgui.end_drag_drop_source()
            if imgui.is_item_hovered():
                imgui.set_tooltip(step_type.summary)

    # -- steps ---------------------------------------------------------------------

    def _draw_steps(self) -> None:
        """The chain: one reorderable, configurable row per step."""
        imgui.spacing()
        if not self._steps:
            imgui.text_disabled("Empty chain -- add a step above.")
            self._draw_drop_zone(0)
            return
        for index, step in enumerate(self._steps):
            self._draw_step_row(index, step)
        # A trailing target so a drag past the last row appends.
        self._draw_drop_zone(len(self._steps))

    def _draw_step_row(self, index: int, step: pipeline.PipelineStep) -> None:
        step_type = pipeline.STEP_TYPES.get(step.kind)
        if step_type is None:  # pragma: no cover - defensive
            return
        imgui.push_id(f"pipe_step_{index}")
        try:
            # Drag handle + header. The selectable is the grab surface and the drop target.
            imgui.selectable(f"=  {index + 1}. {step_type.label}", False)
            if imgui.begin_drag_drop_source():
                imgui.set_drag_drop_payload(_PAYLOAD_MOVE, str(index).encode("ascii"))
                imgui.text(step_type.label)
                imgui.end_drag_drop_source()
            self._accept_drops(index)

            imgui.same_line()
            if icons.icon_button("##remove", "trash", size=16.0, tooltip="Remove step"):
                self._pending = ("remove", index)

            imgui.indent(12.0)
            imgui.push_item_width(-120.0)
            try:
                self._edit_params(step, step_type)
            finally:
                imgui.pop_item_width()
                imgui.unindent(12.0)
        finally:
            imgui.pop_id()

    def _accept_drops(self, index: int) -> None:
        """Accept a reorder (MOVE) or an insert-here (ADD) dropped onto row ``index``."""
        if not imgui.begin_drag_drop_target():
            return
        try:
            move = imgui.accept_drag_drop_payload(_PAYLOAD_MOVE)
            if move is not None:
                self._pending = ("move", int(bytes(move).decode("ascii")), index)
            add = imgui.accept_drag_drop_payload(_PAYLOAD_ADD)
            if add is not None:
                self._pending = ("insert", bytes(add).decode("ascii"), index)
        finally:
            imgui.end_drag_drop_target()

    def _draw_drop_zone(self, index: int) -> None:
        """An append target at the end of the list (and the whole area when empty)."""
        imgui.dummy(1.0, 6.0)
        if not imgui.begin_drag_drop_target():
            return
        try:
            move = imgui.accept_drag_drop_payload(_PAYLOAD_MOVE)
            if move is not None:
                self._pending = ("move", int(bytes(move).decode("ascii")), max(0, index - 1))
            add = imgui.accept_drag_drop_payload(_PAYLOAD_ADD)
            if add is not None:
                self._pending = ("insert", bytes(add).decode("ascii"), index)
        finally:
            imgui.end_drag_drop_target()

    def _apply_pending(self) -> None:
        """Run the structural edit a drag or a click deferred, now the loop is done."""
        pending = self._pending
        self._pending = None
        if pending is None:
            return
        action = pending[0]
        if action == "remove":
            i = int(pending[1])
            if 0 <= i < len(self._steps):
                self._steps.pop(i)
        elif action == "move":
            pipeline.move_step(self._steps, int(pending[1]), int(pending[2]))
        elif action == "insert":
            key, at = str(pending[1]), int(pending[2])
            if key in pipeline.STEP_TYPES:
                self._steps.insert(max(0, min(at, len(self._steps))), pipeline.new_step(key))

    def _param_visible(self, step: pipeline.PipelineStep, name: str) -> bool:
        """Hide parameters that do not apply to the current mode of a step."""
        if step.kind == "baseline" and name in ("lam_log10", "p"):
            return step.params.get("kind") == "asls"
        if step.kind == "filter" and name == "cutoff_hi":
            return step.params.get("btype") in ("bandpass", "bandstop")
        return True

    def _edit_params(self, step: pipeline.PipelineStep, step_type: pipeline.StepType) -> None:
        for spec in step_type.params:
            if not self._param_visible(step, spec.name):
                continue
            value = step.params.get(spec.name, 0)
            if spec.kind == "enum":
                _c, new = widgets.enum_combo(spec.label, str(value), spec.options)
                step.params[spec.name] = new
            elif spec.kind == "int":
                changed, new = imgui.input_int(spec.label, int(value))
                if changed:
                    step.params[spec.name] = int(min(max(new, spec.vmin), spec.vmax))
            elif spec.kind == "logfloat":
                _c, new = widgets.labeled_slider_float(
                    spec.label, float(value), spec.vmin, spec.vmax, fmt="1e%.1f"
                )
                step.params[spec.name] = float(new)
            else:  # float
                if spec.vmin == 0.0 and spec.vmax == 0.0:
                    _c, new = widgets.labeled_drag_float(spec.label, float(value), speed=0.01)
                else:
                    _c, new = widgets.labeled_slider_float(
                        spec.label, float(value), spec.vmin, spec.vmax
                    )
                step.params[spec.name] = float(new)

    # -- preview -------------------------------------------------------------------

    def _draw_preview(
        self,
        x: np.ndarray,
        y: np.ndarray,
        label: str,
        result: pipeline.PipelineResult,
    ) -> None:
        if not result.ok and result.failed_index is not None:
            failed = self._steps[result.failed_index]
            st = pipeline.STEP_TYPES.get(failed.kind)
            name = st.label if st is not None else failed.kind
            widgets.error_box(f"Step {result.failed_index + 1} ({name}) failed: {result.error}")

        if not self._steps:
            widgets.mini_plot(
                "pipe_preview",
                y,
                x=x,
                height=_PREVIEW_HEIGHT,
                label=label,
                color=theme.get_color("muted"),
            )
            return

        overlay = self._overlay_on(result.x, x, y)
        imgui.text_disabled(f"Chain output ({len(self._steps)} steps)")
        widgets.mini_plot(
            "pipe_preview",
            result.y,
            x=result.x,
            overlay=overlay,
            height=_PREVIEW_HEIGHT,
            label="result",
            color=theme.get_color("accent"),
            overlay_color=theme.get_color("warn"),
        )
        imgui.text_colored("--", *theme.get_color("accent"))
        imgui.same_line()
        imgui.text("result")
        if overlay is not None:
            imgui.same_line()
            imgui.text_colored("--", *theme.get_color("warn"))
            imgui.same_line()
            imgui.text("source")

    @staticmethod
    def _overlay_on(result_x: np.ndarray, x: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
        """The source interpolated onto the result's x grid, for a comparable overlay."""
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        if result_x.size == 0 or xa.size < 2:
            return None
        order = np.argsort(xa, kind="stable")
        return np.interp(np.asarray(result_x, dtype=np.float64), xa[order], ya[order])

    # -- apply ---------------------------------------------------------------------

    def _draw_apply(self, result: pipeline.PipelineResult, label: str) -> None:
        imgui.text("Apply as")
        imgui.same_line()
        for i, (candidate, text) in enumerate(_APPLY_MODES):
            if i:
                imgui.same_line()
            if imgui.radio_button(text, self._apply_mode == candidate):
                self._apply_mode = candidate

        imgui.push_item_width(-140.0)
        try:
            if self._apply_mode == "new_layer":
                changed, self._layer_kind = widgets.enum_combo(
                    "Kind", self._layer_kind, layerops.KIND_KEYS
                )
                if changed:
                    self._layer_opts = layerops.default_kind_options(self._layer_kind)
                widgets.kind_options_editor(self._layer_opts)
            _c, self._apply_name = imgui.input_text("Name", self._apply_name)
            imgui.same_line()
            widgets.help_marker(f"Leave empty for the default: {self._default_name(label)!r}")
        finally:
            imgui.pop_item_width()

        theme.push_accent()
        clicked = imgui.button("Apply", 130.0, 0.0)
        theme.pop_accent()
        imgui.same_line()
        widgets.help_marker("Queued for the next frame and recorded on the undo stack.")
        if clicked:
            self._submit_apply(result, label)

    def _default_name(self, label: str) -> str:
        return f"{label} pipeline".strip()

    def _output_name(self, label: str) -> str:
        return self._apply_name.strip() or self._default_name(label)

    def _submit_apply(self, result: pipeline.PipelineResult, label: str) -> None:
        x = np.array(result.x, dtype=np.float64)
        y = np.array(result.y, dtype=np.float64)
        name = self._output_name(label)
        if self._apply_mode == "new_layer":
            cmd = self._command_new_layer(x, y, name)
        else:
            cmd = self._command_new_dataset(x, y, name)
        self.submit(lambda: self._push(cmd))

    def _push(self, cmd: Command) -> None:
        if not self.undo.push(cmd):
            self._notice = (
                f"{cmd.label}: applied, but the data is too large to snapshot for undo. "
                f"The undo history was cleared."
            )

    def _command_new_layer(self, x: np.ndarray, y: np.ndarray, name: str) -> Command:
        plot = self.plot
        hud = self.hud
        kind = self._layer_kind
        options = dict(self._layer_opts)
        holder: Dict[str, Any] = {}

        def do() -> None:
            holder["layer"] = layerops.add_xy_layer(
                plot, x, y, kind=kind, label=name, options=options
            )

        def undo() -> None:
            layer = holder.pop("layer", None)
            if layer is not None:
                layerops.remove_layer(plot, hud, layer)

        return Command(label=f"Plot {name!r}", do=do, undo=undo)

    def _command_new_dataset(self, x: np.ndarray, y: np.ndarray, name: str) -> Command:
        store = self.store
        dataset = DataSet(name, [Column("x", x), Column("y", y)])
        dataset.source = "derived"

        def do() -> None:
            store.add(dataset)

        def undo() -> None:
            store.remove(dataset)

        return Command(label=f"Create dataset {name!r}", do=do, undo=undo)


# ----------------------------------------------------------------------------------
# layer geometry probe (a compact local copy; the panel needs only x/y columns)
# ----------------------------------------------------------------------------------


def _layer_table(layer: Any) -> Optional[Tuple[str, List[str], np.ndarray]]:
    """Resolve ``layer`` to ``(attr, column_names, array)``, or None if not tabular."""
    layer_type = getattr(layer, "layer_type", "") or ""
    if layer_type in _LAYER_UNSUPPORTED:
        return None
    for attr, default_names in _LAYER_GEOMETRY:
        array = getattr(layer, attr, None)
        if array is None:
            continue
        arr = np.asarray(array)
        if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 1:
            continue
        names = list(default_names[: arr.shape[1]])
        names.extend(f"c{i + 1}" for i in range(len(names), arr.shape[1]))
        return attr, names, arr
    return None


def _layer_display(layer: Any) -> str:
    """A stable, unique-per-layer combo entry."""
    lbl = getattr(layer, "label", None) or getattr(layer, "layer_type", "layer")
    return f"{lbl} [{getattr(layer, 'layer_id', '?')}]"
