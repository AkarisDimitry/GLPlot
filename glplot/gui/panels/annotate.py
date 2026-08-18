"""Annotations: the text tool, the shape tools, and the annotation list.

Two things live here. The **tool mode** (:class:`ToolController`) is a canvas-level
interaction state — Pan, Text, Arrow, H-line, V-line, Rect — and the **panel**
(:class:`AnnotatePanel`) is the list that adds, edits, restyles and deletes what the
tools place. They are one module because they are one feature: the list's "Add" buttons
arm the tools, and the tools' output is the list's rows.

How the tools get the mouse without touching the engine
-------------------------------------------------------
``engine._on_mouse_button`` (``engine.py:1473``) and ``_on_scroll`` both begin with::

    self.hud.on_mouse_button(window, button, action, mods)
    if self.hud.wants_mouse():
        return

and ``hud.wants_mouse()`` is ``imgui.get_io().want_capture_mouse`` (``hud.py:141``). So
an imgui window with a full-canvas ``invisible_button`` over the plot makes the engine
early-out of its own camera handling *for free*, with no engine edit at all. That is
exactly what :func:`draw_canvas_layer` puts up — but **only when a tool other than Pan is
armed**. In Pan mode no overlay is submitted, ``want_capture_mouse`` stays False over the
canvas, and plain-drag reaches the engine and pans precisely as it always has. Nothing
regresses for a user who never opens this panel, because in the default mode this module
draws nothing.

Verified headless (``scratchpad/probe_capture.py``, ``probe_click.py``): with the overlay
up, ``want_capture_mouse`` is True from the second frame (imgui resolves hover from the
previous frame's item rects, so arming a tool costs exactly one frame — a click cannot
land in that window because the click that armed the tool was itself over the rail); with
no overlay it stays False. The button reports ``active`` on press, ``is_mouse_dragging``
with a live delta while held, and ``clicked`` on release — which is why a *click* here is
"released with a drag delta under :data:`_CLICK_SLOP_PX`" rather than ``clicked`` alone.

Why there is no Select tool
---------------------------
Pan / Select / Text was the shape the brief suggested, and Select is deliberately absent.
A Select mode means plain-drag draws a marquee, and the marquee is resolved inside
``engine._on_mouse_button``'s ``drag_mode`` ladder — the exact lines the selection work
owns this phase. Worse, the overlay above *swallows* the press before the engine sees it,
so a Select mode built here could not reuse that ladder at all; it would have to re-drive
``engine._run_marquee_pick`` behind the engine's back, duplicating the state machine that
already exists. Shift+Drag is already the marquee and needs no mode. So the rail offers
Pan plus the tools that place something, and selection stays where it belongs.

CONTRACT §1.1 applies to every line below: this module runs inside an imgui draw
callback, so nothing here touches ``plot.scene`` directly — placements, edits and drags
all go through :meth:`~glplot.gui.workspace.Workspace.queue`. That is not belt-and-braces
for text specifically: ``engine.add_text`` sets ``frame.dirty_ui`` but **not**
``dirty_scene`` (``engine.py:450``) even though text draws in the scene pass, so a label
added outside the queue does not appear until an unrelated event happens to wake the
reactive loop. The queue's epilogue is what fixes it (§1.5).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import icons, layerops
from ..history import Command
from .base import Panel

try:
    from imgui_bundle import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - GL-less import guard
    IMGUI_AVAILABLE = False
    imgui = None

try:
    import glfw

    GLFW_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - GL-less import guard
    GLFW_AVAILABLE = False
    glfw = None

logger = logging.getLogger(__name__)

#: Tool ids. ``PAN`` is the default and is the *absence* of an overlay, not a mode with
#: behaviour of its own — see the module docstring.
TOOL_PAN = "pan"
TOOL_TEXT = "text"
TOOL_ARROW = "arrow"
TOOL_HLINE = "hline"
TOOL_VLINE = "vline"
TOOL_RECT = "rect"

#: ``(id, label, icon, hint)`` per tool, in rail order. The hint is the status line the
#: overlay shows while the tool is armed: a modal cursor with no visible statement of
#: what it does is how a tool mode becomes a bug report.
TOOL_SPECS: Tuple[Tuple[str, str, str, str], ...] = (
    (TOOL_PAN, "Pan", "home", "Drag to pan, scroll to zoom."),
    (TOOL_TEXT, "Text", "table", "Click to place a text box. Drag one to move it."),
    (TOOL_ARROW, "Arrow", "chevron_right", "Drag from tail to tip."),
    (TOOL_HLINE, "H line", "minus", "Click to place a horizontal rule across the view."),
    (TOOL_VLINE, "V line", "minus", "Click to place a vertical rule across the view."),
    (TOOL_RECT, "Rect", "grid", "Drag a box from corner to corner."),
)

_TOOL_LABELS: Dict[str, str] = {spec[0]: spec[1] for spec in TOOL_SPECS}

#: A release whose drag delta stayed under this many pixels is a click, not a drag.
#: imgui reports ``clicked`` on release even after a 300px drag, so the two gestures are
#: told apart by distance here rather than by trusting ``clicked``.
_CLICK_SLOP_PX = 4.0

#: A shape drag shorter than this in either axis is discarded on release. Without it a
#: stray click in Rect mode leaves a zero-area patch in the scene that cannot be seen,
#: cannot be picked, and can only be found in the layers list.
_MIN_DRAG_PX = 3.0

#: Attribute the lazily-built :class:`ToolController` is cached under on the workspace.
_TOOL_ATTR = "_annotate_tool"

#: imgui window flags for the canvas overlay: invisible, inert, unsaved, and never
#: focus-stealing. ``NO_BRING_TO_FRONT_ON_FOCUS`` matters — without it, pressing on the
#: canvas raises the overlay above every panel window and the panels stop taking clicks.
_OVERLAY_FLAGS = 0
if IMGUI_AVAILABLE:  # pragma: no branch - constant folding, guarded for a GL-less import
    _OVERLAY_FLAGS = (
        imgui.WindowFlags_.no_title_bar
        | imgui.WindowFlags_.no_resize
        | imgui.WindowFlags_.no_move
        | imgui.WindowFlags_.no_scrollbar
        | imgui.WindowFlags_.no_saved_settings
        | imgui.WindowFlags_.no_collapse
        | imgui.WindowFlags_.no_background
        | imgui.WindowFlags_.no_bring_to_front_on_focus
        | imgui.WindowFlags_.no_focus_on_appearing
    )


# ----------------------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------------------


def world_to_screen(plot: Any, x: float, y: float) -> Optional[Tuple[float, float]]:
    """Project world ``(x, y)`` to window pixels, or None when it is off-view.

    Deliberately goes through ``camera_controller.mvp`` and the same NDC-to-pixel
    arithmetic as ``renderers/text.py:39-54`` rather than inverting
    ``controllers.screen_to_world``. The renderer's path is what decides where the glyphs
    actually land, so reproducing it is what makes the hit-test below agree with what the
    user sees. An independently-derived inverse would be correct only for as long as the
    two happened to stay in step — and the margins (which the workspace *changes*, see
    ``RAIL_AXIS_MARGIN_L``) are exactly the kind of thing that pulls them apart.
    """
    controller = getattr(plot, "camera_controller", None)
    if controller is None:
        return None
    width = int(getattr(plot, "width", 0) or 0)
    height = int(getattr(plot, "height", 0) or 0)
    if width <= 0 or height <= 0:
        return None

    try:
        mvp = controller.mvp(width, height)
    except Exception:  # pragma: no cover - defensive; a malformed camera must not kill the frame
        return None

    ndc = mvp @ np.array([float(x), float(y), 0.0, 1.0], dtype=np.float32)
    if ndc[3] != 0:
        ndc = ndc / ndc[3]
    if not np.all(np.isfinite(ndc[:2])):
        return None
    return (
        float((ndc[0] + 1.0) * 0.5 * width),
        float((1.0 - ndc[1]) * 0.5 * height),
    )


def screen_to_world(plot: Any, sx: float, sy: float) -> Optional[Tuple[float, float]]:
    """Window pixels to world coordinates, or None when the camera cannot answer."""
    controller = getattr(plot, "camera_controller", None)
    if controller is None:
        return None
    width = int(getattr(plot, "width", 0) or 0)
    height = int(getattr(plot, "height", 0) or 0)
    if width <= 0 or height <= 0:
        return None
    try:
        wx, wy = controller.screen_to_world(float(sx), float(sy), width, height)
    except Exception:  # pragma: no cover - defensive
        return None
    if not (np.isfinite(wx) and np.isfinite(wy)):
        return None
    return (float(wx), float(wy))


def _view_window(plot: Any) -> Optional[Tuple[float, float, float, float]]:
    """The current world window ``(l, r, b, t)``, or None."""
    controller = getattr(plot, "camera_controller", None)
    if controller is None:
        return None
    width = int(getattr(plot, "width", 0) or 0)
    height = int(getattr(plot, "height", 0) or 0)
    if width <= 0 or height <= 0:
        return None
    try:
        window = controller.world_window(width, height)
        return tuple(float(v) for v in window)  # type: ignore[return-value]
    except Exception:  # pragma: no cover - defensive
        return None


def _text_screen_rect(plot: Any, layer: Any) -> Optional[Tuple[float, float, float, float]]:
    """A text layer's on-screen bounding box, matching how the renderer places it.

    ``renderers/text.py:66`` calls ``add_text(screen_x, screen_y, ...)``, and imgui's
    ``add_text`` treats its position as the **top-left** of the string — so the box grows
    right and down from the projected point, and ``calc_text_size`` measures it in the
    same default font the renderer uses.
    """
    if getattr(layer, "layer_type", None) != "text":
        return None
    wx, wy = layerops.text_annotation_position(layer)
    projected = world_to_screen(plot, wx, wy)
    if projected is None:
        return None
    sx, sy = projected
    text = str(getattr(layer, "text", "") or "")
    width, height = imgui.calc_text_size(text if text else " ")
    return (sx, sy, sx + max(float(width), 6.0), sy + max(float(height), 6.0))


# ----------------------------------------------------------------------------------
# Tool state
# ----------------------------------------------------------------------------------


class ToolController:
    """The canvas tool mode and the in-progress gesture. One per workspace.

    Held on the workspace rather than on the panel because the tools must keep working
    with the Annotate panel closed — a modal cursor that silently reverts when a window
    is dismissed is worse than no modal cursor. :func:`get_tool` builds it on demand, so
    the workspace needs no constructor change to host it.
    """

    def __init__(self, ws: Any) -> None:
        self.ws = ws

        #: The armed tool. Always one of :data:`TOOL_SPECS`' ids.
        self.mode: str = TOOL_PAN

        #: Style the next placement will use. Shared by every tool on purpose: placing
        #: three arrows in a row should not mean setting the colour three times.
        self.color: Tuple[float, float, float, float] = (0.9, 0.25, 0.25, 1.0)
        self.fill_color: Tuple[float, float, float, float] = (0.3, 0.6, 1.0, 0.35)
        self.fontsize: int = 12
        self.line_width: float = 1.5

        #: In-progress drag, in *screen* pixels: ``(x0, y0)`` press, ``(x1, y1)`` now.
        self._drag_from: Optional[Tuple[float, float]] = None
        self._drag_to: Optional[Tuple[float, float]] = None

        #: The pending inline text box: ``(world_x, world_y)`` plus its buffer. Non-None
        #: exactly while the inline editor is open.
        self._pending_xy: Optional[Tuple[float, float]] = None
        self._pending_text: str = ""
        self._pending_focus: bool = False

        #: The text layer being edited inline (None when placing a new one), and the
        #: string it held when the edit began — the undo entry's "before".
        self._editing: Optional[Any] = None
        self._editing_before: str = ""

        #: The text layer being dragged, its original ``(x, y)``, and the world offset
        #: from its anchor to the press point (so it does not snap under the cursor).
        self._moving: Optional[Any] = None
        self._move_origin: Optional[Tuple[float, float]] = None
        self._move_grab: Tuple[float, float] = (0.0, 0.0)

        #: Frame guard. :func:`draw_canvas_layer` is called by both the workspace (once
        #: wired) and the panel, so that the tools work either way; this makes the second
        #: call in a frame a no-op instead of a duplicate window id and a double-handled
        #: click.
        self._last_frame: int = -1

    # -- mode ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Arm ``mode``, abandoning any gesture the previous tool had in flight."""
        if mode not in _TOOL_LABELS:
            raise ValueError(f"unknown tool mode {mode!r}; expected one of {sorted(_TOOL_LABELS)}")
        if mode == self.mode:
            return
        self.cancel()
        self.mode = mode

    def cancel(self) -> None:
        """Drop every in-flight gesture. Never mutates the scene, so it is always safe."""
        self._drag_from = None
        self._drag_to = None
        self._pending_xy = None
        self._pending_text = ""
        self._pending_focus = False
        self._editing = None
        self._editing_before = ""
        self._moving = None
        self._move_origin = None
        self._move_grab = (0.0, 0.0)

    @property
    def is_editing(self) -> bool:
        """True while the inline text editor owns the keyboard."""
        return self._pending_xy is not None

    # -- deferral --------------------------------------------------------------

    def _submit(self, fn: Any) -> None:
        """Defer a scene mutation to the queue drain (CONTRACT §1.3)."""
        self.ws.queue.submit(fn)

    def _push(self, cmd: Command) -> None:
        """Defer ``cmd`` *and* record it — the undoable :meth:`_submit`.

        ``UndoStack.push`` runs ``cmd.do`` itself, so pushing from inside the queued
        closure is what keeps the mutation on the GL thread at the top of the loop,
        exactly as ``Panel.push_command`` does.
        """
        undo = self.ws.undo
        self.ws.queue.submit(lambda: undo.push(cmd))


def get_tool(ws: Any) -> ToolController:
    """The workspace's :class:`ToolController`, built on first use.

    The whole integration surface for the tool mode is this function plus
    :func:`draw_tool_rail` and :func:`draw_canvas_layer`. Nothing here needs the
    workspace to grow a field, which is what keeps this feature out of ``workspace.py``.
    """
    tool = getattr(ws, _TOOL_ATTR, None)
    if tool is None:
        tool = ToolController(ws)
        setattr(ws, _TOOL_ATTR, tool)
    return tool


def tool_mode(ws: Any) -> str:
    """The armed tool id. ``TOOL_PAN`` unless something armed another."""
    return get_tool(ws).mode


def set_tool_mode(ws: Any, mode: str) -> None:
    """Arm ``mode``. Raises ValueError on an unknown id."""
    get_tool(ws).set_mode(mode)


# ----------------------------------------------------------------------------------
# The workspace-facing draw entry points
# ----------------------------------------------------------------------------------


def draw_tool_rail(ws: Any, *, size: float = 22.0) -> None:
    """Draw the tool-mode buttons. Call from inside the workspace's rail window.

    The integration point for ``Workspace.draw_toolbar``::

        from .panels import annotate
        annotate.draw_tool_rail(self)

    Safe to omit: the same buttons are in the panel's own header, so the tools are
    reachable either way — the rail is the ergonomic home for them, not the only one.
    """
    if not IMGUI_AVAILABLE:
        return
    tool = get_tool(ws)
    for mode, label, icon, hint in TOOL_SPECS:
        if icons.icon_button(
            f"##tool_{mode}",
            icon,
            size=size,
            tooltip=f"{label} — {hint}",
            active=tool.mode == mode,
        ):
            # A second click on the armed tool returns to Pan: a modal cursor needs an
            # exit that does not require knowing which button is the neutral one.
            tool.set_mode(TOOL_PAN if tool.mode == mode else mode)
        imgui.spacing()


def draw_canvas_layer(ws: Any) -> None:
    """Draw the canvas overlay and run the armed tool. Once per frame, any caller.

    The integration point for ``Workspace.draw`` — call it *after* ``_draw_panels`` so
    the inline text editor stacks above the panel windows::

        from .panels import annotate
        annotate.draw_canvas_layer(self)

    Calling it twice in one frame is a no-op the second time (see
    ``ToolController._last_frame``), so wiring it into the workspace does not double up
    with the panel's own fallback call. In ``TOOL_PAN`` with no editor open it submits
    **no window at all**, which is what leaves ``want_capture_mouse`` False and the
    engine's pan untouched.
    """
    if not IMGUI_AVAILABLE:
        return
    tool = get_tool(ws)
    frame = int(getattr(ws, "_frame", 0))
    if tool._last_frame == frame:
        return
    tool._last_frame = frame

    if tool.mode == TOOL_PAN and not tool.is_editing:
        return

    plot = ws.plot
    width = float(getattr(plot, "width", 0) or 0)
    height = float(getattr(plot, "height", 0) or 0)
    if width <= 0 or height <= 0:
        return

    if tool.mode != TOOL_PAN:
        _draw_overlay(ws, tool, width, height)
    if tool.is_editing:
        _draw_inline_editor(ws, tool)


# ----------------------------------------------------------------------------------
# Overlay
# ----------------------------------------------------------------------------------


def _draw_overlay(ws: Any, tool: ToolController, width: float, height: float) -> None:
    """The full-canvas hit target, the live preview, and the gesture state machine."""
    imgui.set_next_window_pos((0.0, 0.0))
    imgui.set_next_window_size((width, height))
    imgui.set_next_window_bg_alpha(0.0)
    imgui.begin("##glplot_canvas_tool", flags=_OVERLAY_FLAGS)

    imgui.set_cursor_screen_pos((0.0, 0.0))
    imgui.invisible_button("##glplot_canvas_hit", (max(width, 1.0), max(height, 1.0)))
    hovered = imgui.is_item_hovered()
    active = imgui.is_item_active()

    if hovered and not tool.is_editing:
        imgui.set_mouse_cursor(
            imgui.MouseCursor_.text_input if tool.mode == TOOL_TEXT else imgui.MouseCursor_.hand
        )

    mouse = tuple(float(v) for v in imgui.get_io().mouse_pos)
    _run_gesture(ws, tool, hovered, active, mouse)
    _draw_preview(ws, tool)
    _draw_hint(tool, width, height)

    imgui.end()


def _run_gesture(
    ws: Any, tool: ToolController, hovered: bool, active: bool, mouse: Tuple[float, float]
) -> None:
    """Advance the press/drag/release state machine for the armed tool."""
    if tool.is_editing:
        # The inline editor owns the mouse: a click landing behind it must not place a
        # second box while the first is still uncommitted.
        return

    # Press.
    if active and tool._drag_from is None:
        tool._drag_from = mouse
        tool._drag_to = mouse
        if tool.mode == TOOL_TEXT:
            _begin_text_press(ws, tool, mouse)
        return

    if tool._drag_from is None:
        return

    # Held.
    if active:
        tool._drag_to = mouse
        if tool._moving is not None:
            _drag_text(ws, tool, mouse)
        return

    # Released (or the press was lost — a window change, a mode switch mid-drag).
    press = tool._drag_from
    release = tool._drag_to or mouse
    tool._drag_from = None
    tool._drag_to = None
    if not hovered and tool._moving is None:
        tool.cancel()
        return
    _finish_gesture(ws, tool, press, release)


def _begin_text_press(ws: Any, tool: ToolController, mouse: Tuple[float, float]) -> None:
    """A press in Text mode: grab an existing box to move, or prepare to place one."""
    layer = _hit_test_text(ws, mouse)
    if layer is None:
        return
    wx, wy = layerops.text_annotation_position(layer)
    world = screen_to_world(ws.plot, mouse[0], mouse[1])
    tool._moving = layer
    tool._move_origin = (float(getattr(layer, "x", 0.0)), float(getattr(layer, "y", 0.0)))
    tool._move_grab = (0.0, 0.0) if world is None else (world[0] - wx, world[1] - wy)


def _drag_text(ws: Any, tool: ToolController, mouse: Tuple[float, float]) -> None:
    """Live-move the dragged text box. Queued, not undoable — the release records that."""
    layer = tool._moving
    if layer is None:
        return
    world = screen_to_world(ws.plot, mouse[0], mouse[1])
    if world is None:
        return
    tx, ty = getattr(layer, "translation", (0.0, 0.0))
    new_x = world[0] - tool._move_grab[0] - float(tx)
    new_y = world[1] - tool._move_grab[1] - float(ty)
    plot = ws.plot
    ws.queue.submit(lambda: layerops.update_text_annotation(plot, layer, x=new_x, y=new_y))


def _finish_gesture(
    ws: Any, tool: ToolController, press: Tuple[float, float], release: Tuple[float, float]
) -> None:
    """Resolve a completed gesture into a placement, a move, or nothing."""
    dx = release[0] - press[0]
    dy = release[1] - press[1]
    is_click = float(np.hypot(dx, dy)) < _CLICK_SLOP_PX

    if tool._moving is not None:
        _commit_text_move(ws, tool, is_click)
        return

    if tool.mode == TOOL_TEXT:
        if is_click:
            _open_new_text(ws, tool, release)
        return

    if tool.mode in (TOOL_HLINE, TOOL_VLINE):
        _place_rule(ws, tool, release)
        return

    if abs(dx) < _MIN_DRAG_PX and abs(dy) < _MIN_DRAG_PX:
        # A stray click in a drag tool: placing a zero-area rect or a zero-length arrow
        # would add an invisible, unpickable layer to the scene.
        return
    if tool.mode == TOOL_ARROW:
        _place_arrow(ws, tool, press, release)
    elif tool.mode == TOOL_RECT:
        _place_rect(ws, tool, press, release)


def _hit_test_text(ws: Any, mouse: Tuple[float, float]) -> Optional[Any]:
    """The topmost text annotation under ``mouse``, or None.

    Reverse scene order: later layers draw over earlier ones, so the last match is the
    one the user is actually pointing at.
    """
    plot = ws.plot
    for layer in reversed(list(plot.scene.layers)):
        if layerops.annotation_kind(layer) != "text":
            continue
        if not getattr(getattr(layer, "style", None), "visible", True):
            continue
        rect = _text_screen_rect(plot, layer)
        if rect is None:
            continue
        x0, y0, x1, y1 = rect
        if x0 <= mouse[0] <= x1 and y0 <= mouse[1] <= y1:
            return layer
    return None


# ----------------------------------------------------------------------------------
# Placement
# ----------------------------------------------------------------------------------


def _open_new_text(ws: Any, tool: ToolController, mouse: Tuple[float, float]) -> None:
    """Open the inline editor for a new box at ``mouse``. Nothing is added until commit."""
    world = screen_to_world(ws.plot, mouse[0], mouse[1])
    if world is None:
        return
    tool._pending_xy = world
    tool._pending_text = ""
    tool._pending_focus = True
    tool._editing = None
    tool._editing_before = ""


def open_text_editor(ws: Any, layer: Any) -> None:
    """Open the inline editor over an existing text ``layer``. Used by the list's Edit."""
    tool = get_tool(ws)
    tool.cancel()
    tool._pending_xy = layerops.text_annotation_position(layer)
    tool._pending_text = str(getattr(layer, "text", "") or "")
    tool._pending_focus = True
    tool._editing = layer
    tool._editing_before = tool._pending_text


def _commit_text_move(ws: Any, tool: ToolController, is_click: bool) -> None:
    """Finish a text drag: one undo entry for the whole gesture, or reopen the editor.

    A press-and-release without movement on an existing box is a click *on* it, which
    means "edit this one" — the same gesture a user expects from every other canvas.
    """
    layer = tool._moving
    origin = tool._move_origin
    tool._moving = None
    tool._move_origin = None
    if layer is None or origin is None:
        return

    if is_click:
        open_text_editor(ws, layer)
        return

    plot = ws.plot
    final = (float(getattr(layer, "x", 0.0)), float(getattr(layer, "y", 0.0)))
    if final == origin:
        return

    # do() re-applies the position the live drag already reached, so pushing this is a
    # no-op on screen; undo() is the point of it.
    tool._push(
        Command(
            label="Move annotation",
            do=lambda: layerops.update_text_annotation(plot, layer, x=final[0], y=final[1]),
            undo=lambda: layerops.update_text_annotation(plot, layer, x=origin[0], y=origin[1]),
        )
    )


def _commit_text(ws: Any, tool: ToolController) -> None:
    """Commit the inline editor: add a new box, or rewrite the edited one. Undoable."""
    text = tool._pending_text.strip()
    layer = tool._editing
    before = tool._editing_before
    xy = tool._pending_xy
    plot = ws.plot
    hud = getattr(ws, "hud", None)
    tool._pending_xy = None
    tool._pending_text = ""
    tool._editing = None
    tool._editing_before = ""

    if layer is not None:
        if text == before:
            return
        if not text:
            # Emptying an existing box deletes it. The alternative is a layer that
            # renders nothing, cannot be clicked (its hit rect is empty) and can only be
            # removed from the Scene panel.
            _delete_group(ws, [layer], "Delete annotation")
            return
        tool._push(
            Command(
                label="Edit annotation text",
                do=lambda: layerops.update_text_annotation(plot, layer, text=text),
                undo=lambda: layerops.update_text_annotation(plot, layer, text=before),
            )
        )
        return

    if not text or xy is None:
        return

    color = tool.color
    fontsize = tool.fontsize
    created: List[Any] = []

    def do() -> None:
        created.clear()
        created.append(
            layerops.add_text_annotation(plot, xy[0], xy[1], text, fontsize=fontsize, color=color)
        )

    def undo() -> None:
        for made in created:
            layerops.remove_layer(plot, hud, made)
        created.clear()

    tool._push(Command(label="Add text annotation", do=do, undo=undo))


def _place_rule(ws: Any, tool: ToolController, mouse: Tuple[float, float]) -> None:
    """Place an h/v rule at ``mouse``, spanning the current view."""
    world = screen_to_world(ws.plot, mouse[0], mouse[1])
    window = _view_window(ws.plot)
    if world is None or window is None:
        return
    left, right, bottom, top = window
    kind = tool.mode
    value = world[1] if kind == TOOL_HLINE else world[0]
    lo, hi = (left, right) if kind == TOOL_HLINE else (bottom, top)
    _place(
        ws,
        tool,
        f"Add {_TOOL_LABELS[kind].lower()}",
        lambda plot: [
            layerops.add_line_annotation(
                plot,
                kind=kind,
                value=value,
                lo=lo,
                hi=hi,
                color=tool.color,
                width=tool.line_width,
            )
        ],
    )


def _place_arrow(
    ws: Any, tool: ToolController, press: Tuple[float, float], release: Tuple[float, float]
) -> None:
    """Place an arrow from the press point to the release point."""
    tail = screen_to_world(ws.plot, press[0], press[1])
    tip = screen_to_world(ws.plot, release[0], release[1])
    if tail is None or tip is None:
        return
    dx = tip[0] - tail[0]
    dy = tip[1] - tail[1]
    # The head is sized in world units, so it must be derived from the arrow's own
    # length: a fixed 0.12 (pyplot.arrow's default) is invisible on a 1e6-wide axis and
    # swallows the shaft on a 1e-6 one.
    length = float(np.hypot(dx, dy))
    head_length = length * 0.22
    head_width = length * 0.14
    _place(
        ws,
        tool,
        "Add arrow",
        lambda plot: layerops.add_arrow_annotation(
            plot,
            tail[0],
            tail[1],
            dx,
            dy,
            color=tool.color,
            width=tool.line_width,
            head_width=head_width,
            head_length=head_length,
        ),
    )


def _place_rect(
    ws: Any, tool: ToolController, press: Tuple[float, float], release: Tuple[float, float]
) -> None:
    """Place a filled rectangle spanning the drag."""
    corner0 = screen_to_world(ws.plot, press[0], press[1])
    corner1 = screen_to_world(ws.plot, release[0], release[1])
    if corner0 is None or corner1 is None:
        return
    _place(
        ws,
        tool,
        "Add rectangle",
        lambda plot: [
            layerops.add_rect_annotation(
                plot,
                corner0[0],
                corner0[1],
                corner1[0],
                corner1[1],
                face_color=tool.fill_color,
            )
        ],
    )


def _place(ws: Any, tool: ToolController, label: str, build: Any) -> None:
    """Queue an undoable placement. ``build(plot)`` returns the layers it created.

    The created layers are captured in a list the closures share rather than returned,
    because ``build`` does not run until the drain — and redo must re-run it, so the
    undo closure has to read whatever the *latest* ``do`` made, not the first one.
    """
    plot = ws.plot
    hud = getattr(ws, "hud", None)
    created: List[Any] = []

    def do() -> None:
        created.clear()
        created.extend(build(plot))

    def undo() -> None:
        for layer in created:
            layerops.remove_layer(plot, hud, layer)
        created.clear()

    tool._push(Command(label=label, do=do, undo=undo))


def _delete_group(ws: Any, layers: Sequence[Any], label: str) -> None:
    """Queue an undoable delete of an annotation's layers.

    Undo re-creates rather than re-inserts: the §1.6 ritual has already released the
    layers' GL handles by then, so putting the same objects back would leave every
    renderer's ``_gl`` naming freed VAOs. Re-running the builder is the only correct
    inverse, which is why the geometry is snapshotted here before anything is removed.
    """
    plot = ws.plot
    hud = getattr(ws, "hud", None)
    rebuilders = [_rebuilder(layer) for layer in layers]
    if any(rebuild is None for rebuild in rebuilders):
        # Something in this group cannot be reconstructed. Deleting anyway and lying
        # about undo would be worse than refusing to record it: UndoStack.push wipes the
        # history for a not-undoable command, which is exactly the honest outcome.
        current = list(layers)
        ws.queue.submit(
            lambda: ws.undo.push(
                Command.not_undoable(
                    label, lambda: layerops.remove_annotation_group(plot, hud, current)
                )
            )
        )
        return

    current = list(layers)

    def do() -> None:
        layerops.remove_annotation_group(plot, hud, current)
        current.clear()

    def undo() -> None:
        current.clear()
        for rebuild in rebuilders:
            current.append(rebuild(plot))  # type: ignore[misc]

    ws.queue.submit(lambda: ws.undo.push(Command(label=label, do=do, undo=undo)))


def _rebuilder(layer: Any) -> Optional[Any]:
    """A ``plot -> layer`` closure that recreates ``layer``, or None if it cannot.

    Snapshots every array *now*, while the layer is still live, so the closure holds
    plain numpy rather than a reference to a layer whose GL has since been freed.
    """
    kind = layerops.annotation_kind(layer)
    style = getattr(layer, "style", None)
    if kind is None or style is None:
        return None
    label = str(getattr(layer, "label", "") or "")
    group = layerops.annotation_group(layer)

    if kind == "text":
        x = float(getattr(layer, "x", 0.0))
        y = float(getattr(layer, "y", 0.0))
        text = str(getattr(layer, "text", "") or "")
        fontsize = int(getattr(style, "text_size_px", 12) or 12)
        color = getattr(style, "color", None)
        return lambda plot: layerops.add_text_annotation(
            plot, x, y, text, label=label, fontsize=fontsize, color=color, group=group
        )

    pts = getattr(layer, "pts", None)
    if kind in ("hline", "vline") and isinstance(pts, np.ndarray) and len(pts) >= 2:
        saved = np.array(pts, dtype=np.float32, copy=True)
        color = getattr(style, "color", None)
        width = float(getattr(style, "line_width", 1.0) or 1.0)
        value = float(saved[0, 1] if kind == "hline" else saved[0, 0])
        span = saved[:, 0] if kind == "hline" else saved[:, 1]
        lo, hi = float(np.min(span)), float(np.max(span))
        return lambda plot: layerops.add_line_annotation(
            plot,
            kind=kind,
            value=value,
            lo=lo,
            hi=hi,
            label=label,
            color=color,
            width=width,
            group=group,
        )

    vertices = getattr(layer, "vertices", None)
    if isinstance(vertices, np.ndarray) and len(vertices):
        saved_v = np.ascontiguousarray(np.array(vertices, dtype=np.float32, copy=True))
        indices = getattr(layer, "indices", None)
        saved_i = (
            None
            if indices is None
            else np.ascontiguousarray(np.array(indices, dtype=np.uint32, copy=True))
        )
        mode = str(getattr(layer, "mode", "strip"))
        face = getattr(style, "face_color", None)

        def rebuild_patch(plot: Any) -> Any:
            plot.add_patch(saved_v, saved_i, mode=mode, face_color=face, label=label)
            made = plot.scene.layers[-1]
            layerops.tag_annotation(made, kind, group)
            layerops.mark_scene_dirty(plot)
            return made

        return rebuild_patch

    if isinstance(pts, np.ndarray) and len(pts):
        saved_p = np.array(pts, dtype=np.float32, copy=True)
        color = getattr(style, "color", None)
        width = float(getattr(style, "line_width", 1.0) or 1.0)

        def rebuild_strip(plot: Any) -> Any:
            plot.add_line_strip(
                saved_p[:, 0],
                saved_p[:, 1],
                color=layerops.normalize_color(color),
                width=width,
                label=label,
            )
            made = plot.scene.layers[-1]
            layerops.tag_annotation(made, kind, group)
            layerops.mark_scene_dirty(plot)
            return made

        return rebuild_strip

    return None


# ----------------------------------------------------------------------------------
# Overlay chrome
# ----------------------------------------------------------------------------------


def _u32(rgba: Sequence[float]) -> int:
    """Pack an RGBA 4-tuple for a draw list."""
    values = list(rgba) + [1.0] * (4 - len(rgba))
    return imgui.get_color_u32(tuple(float(v) for v in values[:4]))


def _draw_preview(ws: Any, tool: ToolController) -> None:
    """Show what the in-flight gesture will place, in this window's draw list.

    In this window's list, not the background one: the background list draws *behind*
    every imgui window, so a preview there would be occluded by the panels it is meant
    to be drawn over.
    """
    dl = imgui.get_window_draw_list()

    if tool.mode == TOOL_TEXT:
        _draw_text_handles(ws, dl)
        return

    press, now = tool._drag_from, tool._drag_to
    if press is None or now is None:
        return
    color = _u32(tool.color)

    if tool.mode == TOOL_ARROW:
        dl.add_line((press[0], press[1]), (now[0], now[1]), color, max(tool.line_width, 1.0))
        length = float(np.hypot(now[0] - press[0], now[1] - press[1]))
        if length > 1e-6:
            ux, uy = (now[0] - press[0]) / length, (now[1] - press[1]) / length
            head = min(14.0, length * 0.35)
            bx, by = now[0] - ux * head, now[1] - uy * head
            dl.add_triangle_filled(
                (now[0], now[1]),
                (bx - uy * head * 0.5, by + ux * head * 0.5),
                (bx + uy * head * 0.5, by - ux * head * 0.5),
                color,
            )
    elif tool.mode == TOOL_RECT:
        dl.add_rect_filled(
            (min(press[0], now[0]), min(press[1], now[1])),
            (max(press[0], now[0]), max(press[1], now[1])),
            _u32(tool.fill_color),
        )
        dl.add_rect(
            (min(press[0], now[0]), min(press[1], now[1])),
            (max(press[0], now[0]), max(press[1], now[1])),
            color,
            rounding=0.0,
            thickness=1.0,
        )


def _draw_text_handles(ws: Any, dl: Any) -> None:
    """Outline every text annotation while the Text tool is armed.

    The one affordance the tool cannot do without: the glyphs have no background box
    (see :data:`~glplot.gui.layerops.UNREACHABLE_ANNOTATIONS`), so without an outline
    there is nothing on screen that says a string is draggable.
    """
    plot = ws.plot
    color = imgui.get_color_u32((0.55, 0.78, 1.0, 0.65))
    for layer in plot.scene.layers:
        if layerops.annotation_kind(layer) != "text":
            continue
        if not getattr(getattr(layer, "style", None), "visible", True):
            continue
        rect = _text_screen_rect(plot, layer)
        if rect is None:
            continue
        dl.add_rect(
            (rect[0] - 2.0, rect[1] - 2.0),
            (rect[2] + 2.0, rect[3] + 2.0),
            color,
            rounding=2.0,
            thickness=1.0,
        )


def _draw_hint(tool: ToolController, width: float, height: float) -> None:
    """A bottom-centred status line naming the armed tool and how to leave it."""
    hint = next((spec[3] for spec in TOOL_SPECS if spec[0] == tool.mode), "")
    label = _TOOL_LABELS.get(tool.mode, tool.mode)
    message = f"{label}: {hint}   (Esc to return to Pan)"
    text_w, text_h = imgui.calc_text_size(message)
    x = (width - float(text_w)) * 0.5
    y = height - float(text_h) - 14.0
    dl = imgui.get_window_draw_list()
    dl.add_rect_filled(
        (x - 8.0, y - 4.0),
        (x + float(text_w) + 8.0, y + float(text_h) + 4.0),
        imgui.get_color_u32((0.0, 0.0, 0.0, 0.55)),
        4.0,
    )
    dl.add_text((x, y), imgui.get_color_u32((0.9, 0.9, 0.92, 1.0)), message)


def _draw_inline_editor(ws: Any, tool: ToolController) -> None:
    """The floating text field at the box's own position — type where it will appear."""
    plot = ws.plot
    xy = tool._pending_xy
    if xy is None:
        return
    screen = world_to_screen(plot, xy[0], xy[1])
    if screen is None:
        # The box scrolled off-view mid-edit (a keyboard pan, a resize). Committing at a
        # position that can no longer be projected is fine; leaving the field pinned to
        # a stale pixel is not.
        _commit_text(ws, tool)
        return

    width = float(getattr(plot, "width", 0) or 0)
    height = float(getattr(plot, "height", 0) or 0)
    box_w = 240.0
    pos_x = min(max(screen[0] - 6.0, 0.0), max(width - box_w - 4.0, 0.0))
    pos_y = min(max(screen[1] - 6.0, 0.0), max(height - 40.0, 0.0))

    imgui.set_next_window_pos((pos_x, pos_y))
    imgui.set_next_window_size((box_w, 0.0))
    imgui.begin(
        "##glplot_text_edit",
        flags=(
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_saved_settings
            | imgui.WindowFlags_.always_auto_resize
        ),
    )

    if tool._pending_focus:
        imgui.set_keyboard_focus_here()
        tool._pending_focus = False

    imgui.push_item_width(box_w - 16.0)
    # buffer_length omitted deliberately (CONTRACT §2.4): an explicit length silently
    # truncates, and this field is the annotation's only content.
    entered, value = imgui.input_text(
        "##glplot_text_edit_field",
        tool._pending_text,
        flags=imgui.InputTextFlags_.enter_returns_true | imgui.InputTextFlags_.auto_select_all,
    )
    imgui.pop_item_width()
    tool._pending_text = value

    imgui.text_disabled("Enter to place, Esc to cancel")
    imgui.end()

    if entered:
        _commit_text(ws, tool)
    elif GLFW_AVAILABLE and imgui.is_key_pressed(imgui.Key.escape):
        tool.cancel()


def handle_escape(ws: Any) -> bool:
    """Escape handling for the tools. Returns True when it consumed the key.

    Offered to ``Workspace._dispatch_keys`` so Escape can leave a tool before the global
    keymap sees it. Optional — the inline editor handles its own Escape either way; this
    only adds "Escape returns the canvas to Pan".
    """
    tool = get_tool(ws)
    if tool.is_editing:
        tool.cancel()
        return True
    if tool.mode != TOOL_PAN:
        tool.set_mode(TOOL_PAN)
        return True
    return False


# ----------------------------------------------------------------------------------
# The panel
# ----------------------------------------------------------------------------------


class AnnotatePanel(Panel):
    """Add, edit, restyle and delete annotations. The list behind the canvas tools."""

    title = "Annotate"
    icon = "table"
    default_open = False

    def __init__(self, ws: Any) -> None:
        super().__init__(ws)
        self._selected: Optional[str] = None

    @property
    def tool(self) -> ToolController:
        """The shared :class:`ToolController`."""
        return get_tool(self.ws)

    def draw(self) -> None:
        """Render the panel body."""
        if not IMGUI_AVAILABLE:
            return

        # Fallback so the tools work before the workspace wires draw_canvas_layer in.
        # Idempotent per frame, so having both callers is harmless (see that function).
        draw_canvas_layer(self.ws)

        self._draw_tools()
        imgui.separator()
        self._draw_defaults()
        imgui.separator()
        self._draw_list()

    # -- sections --------------------------------------------------------------

    def _draw_tools(self) -> None:
        """The tool picker, mirroring the rail so the panel is self-sufficient."""
        imgui.text("Tool")
        tool = self.tool
        imgui.push_id("annotate_tools")
        for index, (mode, label, icon, hint) in enumerate(TOOL_SPECS):
            if index:
                imgui.same_line()
            if icons.icon_button(
                f"##panel_tool_{mode}",
                icon,
                size=24.0,
                tooltip=f"{label} — {hint}",
                active=tool.mode == mode,
            ):
                tool.set_mode(TOOL_PAN if tool.mode == mode else mode)
        imgui.pop_id()
        imgui.text_disabled(next((s[3] for s in TOOL_SPECS if s[0] == tool.mode), ""))

    def _draw_defaults(self) -> None:
        """Style for the *next* placement."""
        tool = self.tool
        imgui.text("New annotation style")
        imgui.push_id("annotate_defaults")

        changed, color = imgui.color_edit4("Color", tool.color)
        if changed:
            tool.color = (float(color[0]), float(color[1]), float(color[2]), float(color[3]))

        changed, fill = imgui.color_edit4("Fill", tool.fill_color)
        if changed:
            tool.fill_color = (float(fill[0]), float(fill[1]), float(fill[2]), float(fill[3]))

        changed, width = imgui.slider_float("Line width", tool.line_width, 0.5, 8.0, "%.1f")
        if changed:
            tool.line_width = float(width)

        changed, size = imgui.slider_int("Font size", tool.fontsize, 6, 48)
        if changed:
            tool.fontsize = int(size)
        _export_only_note("Font size is honoured by savefig only")

        imgui.pop_id()

    def _draw_list(self) -> None:
        """One row per annotation, grouped so an arrow is a row and not two."""
        groups = layerops.annotation_groups(self.plot)
        imgui.text(f"Annotations ({len(groups)})")
        imgui.same_line()
        if imgui.small_button("Delete all##annotate_delete_all") and groups:
            for group, _kind, layers in groups:
                _delete_group(self.ws, layers, "Delete annotation")
            self._selected = None

        if not groups:
            imgui.text_disabled("None yet. Pick a tool above, then click the plot.")
            self._draw_unreachable()
            return

        for group, kind, layers in groups:
            # Every row draws the same widget labels; without a per-group id scope they
            # would share one imgui id and the last row would win every click. This is
            # the collision that is invisible until the whole list stops responding.
            imgui.push_id(f"annotation_{group}")
            self._draw_row(group, kind, layers)
            imgui.pop_id()

        imgui.separator()
        self._draw_unreachable()

    def _draw_row(self, group: str, kind: str, layers: List[Any]) -> None:
        """One annotation: header, then its controls when expanded."""
        spec = layerops.annotation_spec(kind)
        first = layers[0]
        style = getattr(first, "style", None)
        visible = bool(getattr(style, "visible", True))

        if icons.icon_button(
            "##vis",
            "eye" if visible else "eye_off",
            size=18.0,
            tooltip="Hide" if visible else "Show",
        ):
            self._set_style(layers, "Toggle annotation", visible=not visible)
        imgui.same_line()

        title = _row_title(kind, layers)
        expanded = self._selected == group
        if imgui.selectable(f"{spec.label}: {title}", expanded)[0]:
            self._selected = None if expanded else group
        imgui.same_line()
        if icons.icon_button("##del", "trash", size=18.0, tooltip="Delete"):
            _delete_group(self.ws, layers, "Delete annotation")
            if self._selected == group:
                self._selected = None
            return

        if not expanded:
            return

        imgui.indent()
        self._draw_row_body(kind, spec, layers, first)
        imgui.unindent()

    def _draw_row_body(
        self, kind: str, spec: layerops.AnnotationKind, layers: List[Any], first: Any
    ) -> None:
        """The expanded controls, gated on what the kind's renderers actually read."""
        if kind == "text":
            imgui.text_disabled(
                f"at ({', '.join(f'{v:.4g}' for v in layerops.text_annotation_position(first))})"
            )
            if imgui.small_button("Edit text##edit"):
                set_tool_mode(self.ws, TOOL_TEXT)
                open_text_editor(self.ws, first)

        if "color" in spec.live_fields or "face_color" in spec.live_fields:
            current = layerops.annotation_color(layers)
            changed, value = imgui.color_edit4("Color", current)
            if changed:
                rgba = (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
                # Fanned across both fields: an arrow's shaft reads style.color and its
                # head reads face_color, and set_annotation_style drops the one a given
                # layer has no slot for.
                self._set_style(layers, "Recolor annotation", color=rgba, face_color=rgba)

        if "line_width" in spec.live_fields:
            width = float(getattr(getattr(first, "style", None), "line_width", 1.0) or 1.0)
            changed, value = imgui.slider_float("Width", width, 0.5, 8.0, "%.1f")
            if changed:
                self._set_style(layers, "Restyle annotation", line_width=float(value))

        if "alpha" in spec.live_fields:
            alpha = float(getattr(getattr(first, "style", None), "alpha", 1.0) or 1.0)
            changed, value = imgui.slider_float("Alpha", alpha, 0.0, 1.0, "%.2f")
            if changed:
                self._set_style(layers, "Restyle annotation", alpha=float(value))

        if "zorder" in spec.live_fields:
            zorder = int(getattr(getattr(first, "style", None), "zorder", 0) or 0)
            changed, value = imgui.drag_float("Z order", float(zorder), 0.2, -100.0, 100.0, "%.0f")
            if changed:
                self._set_style(layers, "Reorder annotation", zorder=int(value))

        if spec.export_only_fields:
            size = int(getattr(getattr(first, "style", None), "text_size_px", 12) or 12)
            changed, value = imgui.slider_int("Font size", size, 6, 48)
            if changed:
                self._set_style(layers, "Resize annotation", text_size_px=int(value))
            _export_only_note("Font size is honoured by savefig only")

    def _draw_unreachable(self) -> None:
        """State plainly what this panel does not offer, and why.

        Read straight from :data:`~glplot.gui.layerops.UNREACHABLE_ANNOTATIONS`. A user
        hunting for a background box or an ellipse deserves the reason in the UI rather
        than the conclusion that the panel is unfinished.
        """
        expanded = imgui.collapsing_header("Not available, and why")
        if not expanded:
            return
        imgui.indent()
        for name, reason in layerops.UNREACHABLE_ANNOTATIONS.items():
            imgui.push_id(f"unreachable_{name}")
            imgui.bullet_text(name)
            imgui.indent()
            imgui.push_text_wrap_pos(0.0)
            imgui.text_disabled(reason)
            imgui.pop_text_wrap_pos()
            imgui.unindent()
            imgui.pop_id()
        imgui.unindent()

    # -- mutation --------------------------------------------------------------

    def _set_style(self, layers: List[Any], label: str, **fields: Any) -> None:
        """Restyle an annotation, undoably, capturing the before-state per layer."""
        plot = self.plot
        targets = list(layers)
        before: List[Dict[str, Any]] = []
        for layer in targets:
            style = getattr(layer, "style", None)
            snapshot: Dict[str, Any] = {}
            for name in fields:
                if name in layerops.LAYER_LEVEL_FIELDS:
                    snapshot[name] = getattr(layer, name, None)
                elif style is not None and hasattr(style, name):
                    snapshot[name] = getattr(style, name)
            before.append(snapshot)

        def do() -> None:
            layerops.set_annotation_style(plot, targets, **fields)

        def undo() -> None:
            for layer, snapshot in zip(targets, before):
                if snapshot:
                    layerops.set_annotation_style(plot, [layer], **snapshot)

        self.push_command(Command(label=label, do=do, undo=undo))


def _row_title(kind: str, layers: List[Any]) -> str:
    """A short, identifying label for a list row."""
    first = layers[0]
    if kind == "text":
        text = str(getattr(first, "text", "") or "")
        return (text[:28] + "…") if len(text) > 28 else (text or "(empty)")
    return str(getattr(first, "label", "") or kind)


def _export_only_note(message: str) -> None:
    """Badge a control whose field no GL renderer reads (CONTRACT §4.1).

    ``style.text_size_px`` is live in ``utils/preview.py:300`` and nowhere else, so the
    slider does something real — just not on screen. Hiding it would lose a working
    export control; showing it unlabelled would read as a broken one.
    """
    imgui.same_line()
    imgui.text_disabled("(export only)")
    if imgui.is_item_hovered():
        imgui.set_tooltip(message)
