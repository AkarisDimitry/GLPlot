"""The workspace: menu bar, icon rail, the floating panel windows, and the keymap.

Layer 3 of the GUI — the only module that knows the full panel set exists. It owns the
three shared services (:class:`~glplot.gui.datasets.DataStore`,
:class:`~glplot.gui.history.UndoStack`, :class:`~glplot.gui.commands.CommandQueue`) and
hands them to every panel through :class:`~glplot.gui.panels.base.Panel`'s proxies, so
there is exactly one of each per plot.

It also owns the fourth service: the :class:`~glplot.gui.actions.ActionRegistry`, built in
exactly one place (:meth:`Workspace._register_actions`). Three consumers read it and none
of them has a list of its own — the keymap dispatches from it (:meth:`Workspace.draw`),
the command palette searches it, and Help's shortcut sheet is generated from it. A
shortcut therefore cannot exist without also being searchable and documented, which is the
entire point of routing every action through one table.

**No docking.** pyimgui 2.0 / Dear ImGui 1.82 ships no docking branch (CONTRACT §2), so
the layout is a left icon rail plus free-floating windows placed with ``FIRST_USE_EVER``
and restorable via "Reset layout". imgui persists whatever the user drags them to in its
own ini; the ``FIRST_USE_EVER`` positions here are only the first-run arrangement.

The workspace draws inside ``hud.update()``, which runs inside an imgui frame *after* the
scene pass — so every line here is bound by CONTRACT §1.1: no direct scene mutation. Even
undo/redo is deferred to the queue, because an undo callback rewrites layer arrays. Every
``Action.run`` registered below obeys the same rule.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from ..options import DEFAULT_AXIS_MARGINS
from . import icons, keys, layerops, theme
from .actions import ActionRegistry
from .commands import CommandQueue
from .datasets import DataStore
from .history import UndoStack
from .panels.base import Panel

if TYPE_CHECKING:
    from ..engine import GPULinePlot

try:
    import imgui

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

#: Size of each rail icon button (side length, points) and the window padding around the
#: single column of them. The rail width is *derived* from these below so the three can never
#: drift apart: a smaller icon or tighter padding always yields a narrower rail.
RAIL_ICON_SIZE = 24.0
RAIL_PAD = 6.0

#: Width of the left icon rail, in points: one icon column plus padding on both sides.
RAIL_WIDTH = RAIL_ICON_SIZE + 2.0 * RAIL_PAD

#: Left axis gutter, in pixels, while the workstation is mounted: the engine's stock gutter
#: plus the rail's own width.
#:
#: The rail is an opaque imgui window pinned at x=0, but the Y tick labels are painted into
#: ``get_background_draw_list()``, which draws *behind* every imgui window. At the stock
#: 60px gutter the labels land under the rail and are simply gone -- the one defect a user
#: cannot work around. Widening the gutter by exactly the rail's width slides the plot's
#: left spine clear of it, and the labels (right-aligned against the spine, see
#: ``renderers/axis.py``) follow it out.
#:
#: Screen-only. Exports must render with :data:`~glplot.options.DEFAULT_AXIS_MARGINS`;
#: a PNG has no rail, so inheriting this would bake in a mystery gutter. ``utils/export.py``
#: restores the defaults for the duration of the render.
RAIL_AXIS_MARGIN_L = RAIL_WIDTH + 60.0

#: Height of the HUD's status overlay strip, which the rail must clear.
_STATUS_STRIP_H = 30.0

#: imgui's ``FocusedFlags_RootAndChildWindows``. Not exposed as a named constant by
#: pyimgui 2.0 — the same literal the data editor uses.
_FOCUSED_ROOT_AND_CHILD_WINDOWS = 3

#: First-run window geometry as ``(x, y, w, h)`` fractions of the viewport, keyed by panel.
#: Fractions rather than pixels so the first-run layout is sane on a 1280x720 laptop and on
#: a 4K display, instead of piling every window into the top-left corner of one and
#: stranding them off-screen on the other.
_LAYOUT: Dict[str, Tuple[float, float, float, float]] = {
    "data": (0.00, 0.00, 0.52, 0.52),
    "functions": (0.00, 0.53, 0.34, 0.47),
    "mathlab": (0.35, 0.53, 0.37, 0.47),
    "pipeline": (0.35, 0.05, 0.37, 0.47),
    "scene": (0.73, 0.00, 0.27, 0.40),
    "style": (0.73, 0.41, 0.27, 0.59),
    "selection": (0.52, 0.00, 0.21, 0.44),
    "help": (0.08, 0.05, 0.62, 0.85),
    # The two 3D panels sit where their 2D counterparts do: the view controls over the
    # Style column (both are "how the figure looks"), the object catalogue over the
    # Functions column (both are "where data comes from").
    "view3d": (0.72, 0.30, 0.28, 0.70),
    "objects3d": (0.00, 0.40, 0.34, 0.60),
    # The timeline is a full-width strip along the bottom, the way every editing tool
    # arranges one: its x axis IS time, so a narrow window makes it unusable. The
    # presentation list is a tall left column, because it is read as a running order.
    "timeline": (0.00, 0.60, 1.00, 0.40),
    "presentation": (0.00, 0.02, 0.26, 0.58),
}

#: Panel key -> chord spec for the toggle+focus actions, in rail order.
_PANEL_CHORDS: Dict[str, str] = {
    "data": "Mod+1",
    "functions": "Mod+2",
    "dynamics": "Mod+0",
    "mathlab": "Mod+3",
    "style": "Mod+4",
    "scene": "Mod+5",
    "help": "Mod+7",
    "selection": "Mod+8",
    "pipeline": "Mod+9",
    # Mod+1..0 are all taken (Mod+6 is the profiler), and the obvious mnemonics are too
    # (Mod+Shift+V is "paste as new dataset", Mod+Shift+O is free but reads as "open").
    # These two were checked against the whole table: no other action claims them.
    "view3d": "Mod+Shift+B",  # B for box — the 3D axis box the panel controls
    "objects3d": "Mod+Shift+G",  # G for generate
    "timeline": "Mod+Shift+T",  # T for timeline
    # R for "run the presentation": Mod+Shift+P is already the palette's VS Code binding.
    "presentation": "Mod+Shift+R",
}


#: The 3D orientations bound to the bare number row, as ``(view key, label, icon)`` in
#: **digit order**: the first entry is 1, the second 2, and so on up to 9. ``0`` is not in
#: the table — it is "reset the view", registered separately, which is why it sits at the
#: far end of the row.
#:
#: Why bare digits rather than a modifier. Snapping to a standard angle is the single
#: most-repeated gesture in 3D authoring, and it is worth a *whole* key: the workflow this
#: exists for is "press 5, press K, move the playhead, press 1, press K" — a camera move
#: built out of six keystrokes and no dialogs. ``keys.is_text_safe`` is False for a bare
#: digit, so the dispatcher suppresses every one of these for as long as any text field
#: holds the keyboard; typing "3" into a duration box cannot spin the camera.
#:
#: The order is the one the panel's own preset combo uses, six faces first (the answers to
#: "what does this look like from the side") and then the three isometrics, so the mapping
#: is learnable rather than arbitrary. ``Mod+1``..``Mod+9`` are the panel toggles and are
#: untouched — a digit means "look from here", Mod+digit means "show me that panel".
_VIEW_DIGITS: Tuple[Tuple[str, str, str], ...] = (
    ("front", "Front", "view_front"),  # 1
    ("back", "Back", "view_front"),  # 2
    ("left", "Left", "view_right"),  # 3
    ("right", "Right", "view_right"),  # 4
    ("top", "Top", "view_top"),  # 5
    ("bottom", "Bottom", "view_top"),  # 6
    ("iso", "Isometric", "view_iso"),  # 7
    ("iso_front_left", "Iso front-left", "view_iso"),  # 8
    ("iso_back_right", "Iso back-right", "view_iso"),  # 9
)

#: View key -> the digit that selects it, derived from :data:`_VIEW_DIGITS` so the two can
#: never disagree about which key is bound to what.
_VIEW_DIGIT_OF: Dict[str, str] = {key: str(i + 1) for i, (key, _, _) in enumerate(_VIEW_DIGITS)}


def _panel_classes() -> List[Tuple[str, Any]]:
    """Import the panel classes lazily and return them as ``(key, class)`` pairs.

    Imported inside the function, not at module scope: a panel that fails to import (a
    syntax error mid-refactor, a missing optional dep) must degrade to "that panel is
    missing" rather than taking down the whole workspace and, with it, the HUD.
    """
    pairs: List[Tuple[str, Any]] = []
    specs = (
        ("data", "data_editor", "DataEditorPanel"),
        ("functions", "functions", "FunctionsPanel"),
        ("dynamics", "dynamics", "DynamicsPanel"),
        ("objects3d", "objects3d", "Objects3DPanel"),
        ("mathlab", "mathlab", "MathLabPanel"),
        ("pipeline", "pipeline", "PipelinePanel"),
        ("scene", "scene", "ScenePanel"),
        ("view3d", "view3d", "View3DPanel"),
        ("timeline", "timeline", "TimelinePanel"),
        ("presentation", "presentation", "PresentationPanel"),
        ("style", "style", "StylePanel"),
        ("selection", "selection", "SelectionPanel"),
        ("help", "help", "HelpPanel"),
    )
    from importlib import import_module

    for key, module_name, class_name in specs:
        try:
            module = import_module(f"{__package__}.panels.{module_name}")
            pairs.append((key, getattr(module, class_name)))
        except Exception:  # pragma: no cover - defensive
            logger.exception("Panel %r failed to import; continuing without it.", key)
    return pairs


def _chord_text(spec: str) -> str:
    """Format a chord spec for display, degrading to the spec if it will not parse."""
    try:
        return keys.format(keys.parse(spec))
    except ValueError:  # pragma: no cover - the specs here are literals
        return spec


class Workspace:
    """The mathematical workstation's shell: menu, rail, panels, palette, keymap.

    One per plot, created lazily by :class:`~glplot.managers.hud.HudManager` only when
    imgui is importable. Construction is cheap and headless-safe; nothing here touches GL.
    """

    def __init__(self, plot: GPULinePlot, hud: Any = None) -> None:
        self.plot = plot
        self._hud = hud

        # The three shared services. Panels reach them through Panel's proxies; nothing
        # else may construct a second one, or half the UI would talk to a dead store.
        self.store = DataStore()
        self.undo = UndoStack()
        self.queue = CommandQueue()

        self.panels: Dict[str, Panel] = {}
        self.open: Dict[str, bool] = {}
        for key, cls in _panel_classes():
            try:
                panel = cls(self)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Panel %r failed to construct; continuing without it.", key)
                continue
            self.panels[key] = panel
            self.open[key] = bool(getattr(cls, "default_open", False))

        #: The command palette. **Not** in ``self.panels``: it is a centred modal overlay
        #: that owns its own window, so the generic ``begin(panel.title, True)`` in
        #: `_draw_panels` would give it a title bar, a close box and a rail icon.
        self.palette: Optional[Any] = self._build_palette()

        #: The one action table. Everything keyboard-, palette- or Help-reachable is here.
        self.registry = ActionRegistry()
        self._register_actions()

        self._theme_applied = False
        self._frame = 0
        #: Countdown of frames during which window geometry is forced (see `reset_layout`).
        self._force_layout = 1
        self._status = ""
        self._show_hint = True
        #: Key of the panel window that held focus as of the last drawn frame. Drives
        #: "Close focused panel" and "Cycle panels"; None when the focus is elsewhere.
        self._focused_panel: Optional[str] = None

        self._widen_axis_gutter_for_rail()

    def _widen_axis_gutter_for_rail(self) -> None:
        """Reserve gutter for the rail so it stops covering the Y tick labels.

        Called once at construction -- the Workspace *is* the mount: `HudManager` builds it
        only when the workstation is actually going to be drawn, so a plain `GPULinePlot`
        with no workstation keeps the stock 60px gutter untouched.

        Only ever *widens*. If an embedder has already set a gutter roomier than the rail
        needs, theirs wins; we would otherwise silently narrow a margin someone chose.
        """
        options = getattr(self.plot, "options", None)
        if options is None:  # pragma: no cover - defensive
            return
        try:
            current = float(getattr(options, "axis_margin_l", DEFAULT_AXIS_MARGINS[0]))
            if current < RAIL_AXIS_MARGIN_L:
                options.axis_margin_l = RAIL_AXIS_MARGIN_L
        except (TypeError, ValueError):  # pragma: no cover - defensive
            logger.debug("Could not widen the axis gutter for the rail.", exc_info=True)

    def _build_palette(self) -> Optional[Any]:
        """Construct the palette, or None if its module will not import.

        Same policy as the panels: a workstation with no palette is a degraded
        workstation, a workstation that will not construct is no workstation at all.
        """
        try:
            from .panels.palette import CommandPalette

            return CommandPalette(self)
        except Exception:  # pragma: no cover - defensive
            logger.exception("The command palette failed to construct; continuing without it.")
            return None

    # -- workspace/HUD glue ----------------------------------------------------

    @property
    def hud(self) -> Optional[Any]:
        """The owning ``HudManager``, or None. Needed for the §1.6 removal ritual."""
        if self._hud is not None:
            return self._hud
        return getattr(self.plot, "hud", None)

    def _hud_state(self) -> Optional[Any]:
        """The HUD's ``HudState``, or None when there is no HUD."""
        return getattr(self.hud, "state", None)

    def _controller(self) -> Optional[Any]:
        """The HUD's ``HudController``, or None. It owns the engine-level verbs."""
        return getattr(self.hud, "controller", None)

    # -- frame -----------------------------------------------------------------

    def draw(self) -> None:
        """Draw the whole workspace. Called once per frame from ``HudManager.update``."""
        if not IMGUI_AVAILABLE:
            return
        if not self._theme_applied:
            theme.apply_theme(dark=not getattr(self.plot.options, "light_bg_mode", False))
            self._theme_applied = True

        # Before the panels draw, per ActionRegistry.dispatch's contract: io.want_text_input
        # describes the last completed frame, so asking at the top of this one is the only
        # read that cannot mistake a field the user is typing in for a free keyboard.
        self._dispatch_keys()

        self.draw_menu()
        self.draw_toolbar()
        self._draw_panels()
        # After the panels: the palette is an overlay, and a window submitted later sits on
        # top of one submitted earlier.
        if self.palette is not None:
            self.palette.draw()
        self._draw_empty_hint()

        # Cheap, but not free: rebuilding a dataset for every layer copies its geometry to
        # float64. Once a second at 60fps is responsive enough for layers created from code
        # while the window is already up, which is the only case this exists for.
        if self._frame % 60 == 0:
            self.sync_datasets_from_scene()

        self._sync_selection_highlight()

        self._frame += 1
        if self._force_layout > 0:
            self._force_layout -= 1

    def _sync_selection_highlight(self) -> None:
        """Keep the renderers' highlight buffers in step with the live selection.

        Detect here (a pure read, safe in a draw callback), apply in the drain. The
        engine's marquee writes ``interaction.selection`` directly, so the panel's own
        queued commands are not the only source of change and something has to notice;
        doing the write here instead would set ``dirty_scene`` inside the draw callback
        and lose the latch at ``engine.py:1238`` (CONTRACT §1.2), leaving the highlight
        invisible until an unrelated event woke the loop.

        Self-limiting: after the queued sync runs, the layers agree with the selection
        and the predicate goes quiet, so this cannot spin the queue frame after frame.
        """
        try:
            from .panels.selection import selection_needs_sync, sync_selection_highlight
        except Exception:  # pragma: no cover - defensive: panel module failed to import
            return
        plot = self.plot
        if selection_needs_sync(plot):
            self.queue.submit(lambda: sync_selection_highlight(plot))

    def _dispatch_keys(self) -> None:
        """Run the keymap for this frame.

        Normally that is one call to :meth:`~glplot.gui.actions.ActionRegistry.dispatch`.
        The palette is the single exception: while it is open it owns the keyboard, and its
        own navigation keys overlap the global map (``Mod+N``/``Mod+P`` move the selection
        *and* are bound to "New Dataset" and nothing respectively). Rather than duplicate
        the keymap, the exception is narrow and reads from the same registry: only the two
        chords that toggle the palette are honoured, so ``Mod+K`` still closes it.
        """
        if self.palette is not None and getattr(self.palette, "visible", False):
            for action_id in ("global.palette", "global.palette_alt"):
                action = self.registry.get(action_id)
                if action is None or action.chord is None:
                    continue
                if keys.pressed(action.chord):
                    action.run()
                    return
            return
        self.registry.dispatch()

    def reset_layout(self) -> None:
        """Restore the first-run window arrangement and reopen the default panels.

        Two frames of forced geometry, not one: a window that is currently collapsed or was
        never submitted needs a frame to exist before ``set_next_window_*`` on it sticks.
        """
        self._force_layout = 2
        for key, panel in self.panels.items():
            self.open[key] = bool(getattr(panel, "default_open", False))

    # ======================================================================================
    # THE ACTION TABLE — every user-facing action in the workstation is registered here.
    #
    # One place, on purpose: the keymap, the palette and Help > Keyboard all read this
    # registry and none of them keeps a list of its own, so an action added here is bound,
    # searchable and documented in one edit — and cannot be any two of those without the
    # third. Rules for a new entry:
    #
    #   * `id` is dotted and stable (an MRU list persists it).
    #   * `description` is mandatory in practice: the palette shows it as the subtitle and
    #     Help renders it under the title. An action with no description is an action the
    #     user must already understand to find.
    #   * `run` may not touch plot.scene / plot.frame / plot.cache directly — CONTRACT §1.1.
    #     Use self.queue.submit, or call a panel method that already does.
    #   * An action that needs a panel opens and focuses it first: a shortcut that silently
    #     acts on an invisible panel is indistinguishable from one that did nothing.
    #   * Chords must be unique. `registry.conflicts()` reports duplicates among *enabled*
    #     actions and Help surfaces them; the table below keeps them unique unconditionally.
    # ======================================================================================

    def _register_actions(self) -> None:
        """Build the one action table. See the banner above before editing."""
        add = self.registry.add

        # -- Global ------------------------------------------------------------------
        add(
            "global.palette",
            "Command Palette",
            "Global",
            self._act_palette,
            chord="Mod+K",
            icon="search",
            keywords=("search", "smart search", "find", "commands", "go to"),
            description="Search every command, layer, dataset, function and help topic.",
            is_enabled=lambda: self.palette is not None,
        )
        add(
            "global.palette_alt",
            "Command Palette (alternate binding)",
            "Global",
            self._act_palette,
            chord="Mod+Shift+P",
            icon="search",
            keywords=("search", "vscode", "commands"),
            description="The same palette on the VS Code binding, for that muscle memory.",
            is_enabled=lambda: self.palette is not None,
        )
        add(
            "global.help",
            "Help",
            "Global",
            lambda: self._act_help(None),
            chord="F1",
            icon="info",
            keywords=("manual", "docs", "guide", "tour", "shortcuts"),
            description="Open Help: the tour, the live keymap, and the expression reference.",
            is_enabled=lambda: "help" in self.panels,
        )
        add(
            "global.undo",
            "Undo",
            "Global",
            self.undo_action,
            chord="Mod+Z",
            icon="undo",
            keywords=("history", "revert", "back"),
            description="Undo the last edit. One stack covers cells, layers and Math Lab.",
            is_enabled=self.undo.can_undo,
        )
        add(
            "global.redo",
            "Redo",
            "Global",
            self.redo_action,
            chord="Mod+Shift+Z",
            icon="redo",
            keywords=("history", "forward"),
            description="Redo the edit that was last undone.",
            is_enabled=self.undo.can_redo,
        )
        add(
            "global.redo_alt",
            "Redo (alternate binding)",
            "Global",
            self.redo_action,
            chord="Mod+Y",
            icon="redo",
            keywords=("history", "forward", "windows"),
            description="The same redo on the Windows binding.",
            is_enabled=self.undo.can_redo,
        )
        add(
            "global.export_png",
            "Export PNG",
            "Global",
            self._act_export_png,
            chord="Mod+S",
            icon="save",
            keywords=("save", "screenshot", "image", "render"),
            description="Write the current view to plot_<timestamp>.png at the export scale.",
            is_enabled=lambda: self._controller() is not None,
        )
        add(
            "global.reset_layout",
            "Reset Layout",
            "Global",
            self.reset_layout,
            icon="refresh",
            keywords=("windows", "arrange", "default", "panels"),
            description="Put every panel window back where it started and reopen the defaults.",
        )
        add(
            "global.quit",
            "Quit",
            "Global",
            self._act_quit,
            chord="Mod+Q",
            icon="close",
            keywords=("exit", "close window"),
            description="Close the plot window and end the session.",
            is_enabled=lambda: getattr(self.plot, "window", None) is not None,
        )

        # -- Panels ------------------------------------------------------------------
        for key, chord in _PANEL_CHORDS.items():
            panel = self.panels.get(key)
            if panel is None:
                continue
            title = panel.title
            add(
                f"panel.{key}",
                f"Toggle {title} Panel",
                "Panels",
                self._panel_toggler(key),
                chord=chord,
                icon=getattr(panel, "icon", "layers"),
                keywords=("panel", "window", "show", "hide", title.lower()),
                description=f"Show the {title} panel and focus it, or hide it if it is up.",
                is_checked=lambda key=key: bool(self.open.get(key, False)),
            )
        add(
            "panel.profiler",
            "Toggle Profiler",
            "Panels",
            self._act_toggle_profiler,
            chord="Mod+6",
            icon="sliders",
            keywords=("fps", "performance", "timings", "overlay"),
            description="Show the HUD's frame-time and GPU-timing overlay.",
            is_enabled=lambda: self._hud_state() is not None,
            is_checked=lambda: bool(getattr(self._hud_state(), "show_profiler", False)),
        )
        add(
            "panel.close",
            "Close Focused Panel",
            "Panels",
            self._act_close_focused,
            chord="Mod+W",
            icon="close",
            keywords=("hide", "dismiss", "window"),
            description="Hide the panel window that currently has focus.",
            is_enabled=lambda: self._focused_panel in self.panels,
        )
        add(
            "panel.cycle",
            "Cycle Panels",
            "Panels",
            self._act_cycle_panels,
            chord="Mod+Tab",
            icon="chevron_right",
            keywords=("next", "switch", "focus", "window"),
            description="Focus the next open panel, wrapping at the end.",
            is_enabled=lambda: any(self.open.get(k, False) for k in self.panels),
        )

        # -- View --------------------------------------------------------------------
        add(
            "view.autoscale",
            "Autoscale",
            "View",
            self._act_autoscale,
            chord="Mod+0",
            icon="target",
            keywords=("fit", "frame", "zoom to fit", "bounds"),
            description="Frame every visible layer. Text layers never contribute bounds.",
            is_enabled=lambda: self._controller() is not None,
        )
        add(
            "view.reset",
            "Reset View",
            "View",
            self._act_reset_view,
            chord="Mod+Home",
            icon="home",
            keywords=("camera", "home", "recenter"),
            description="Return the camera to its initial framing.",
            is_enabled=lambda: self._controller() is not None,
        )
        add(
            "view.split_grid",
            "Split View: 2x2 Grid",
            "View",
            lambda: self._act_split(2, 2),
            icon="grid",
            keywords=("panels", "subplots", "tile", "divide", "layout", "multi"),
            description="Divide the window into a 2x2 grid of panels, keeping the current "
            "plot in the top-left. Each panel pans and zooms on its own.",
            is_enabled=lambda: self._controller() is not None,
        )
        add(
            "view.split_cols",
            "Split View: Side by Side",
            "View",
            lambda: self._act_split(1, 2),
            icon="grid",
            keywords=("panels", "subplots", "columns", "divide", "layout"),
            description="Split the window into two side-by-side panels, keeping the current "
            "plot on the left.",
            is_enabled=lambda: self._controller() is not None,
        )
        add(
            "view.split_rows",
            "Split View: Stacked",
            "View",
            lambda: self._act_split(2, 1),
            icon="grid",
            keywords=("panels", "subplots", "rows", "stack", "divide", "layout"),
            description="Split the window into two stacked panels, keeping the current plot "
            "on top.",
            is_enabled=lambda: self._controller() is not None,
        )
        add(
            "view.merge",
            "Merge Panels",
            "View",
            self._act_merge,
            icon="home",
            keywords=("panels", "single", "unsplit", "collapse", "one", "layout"),
            description="Collapse back to a single full-window panel, keeping the active "
            "panel's plot.",
            is_enabled=lambda: len(getattr(self.plot, "panels", [])) > 1,
        )
        add(
            "view.zoom_in",
            "Zoom In",
            "View",
            lambda: self._act_zoom(1.0),
            chord="Mod+=",
            icon="zoom_in",
            keywords=("magnify", "closer", "scale"),
            description="Zoom in about the centre of the viewport.",
            is_enabled=self._can_zoom,
        )
        add(
            "view.zoom_out",
            "Zoom Out",
            "View",
            lambda: self._act_zoom(-1.0),
            chord="Mod+-",
            icon="zoom_out",
            keywords=("shrink", "further", "scale"),
            description="Zoom out about the centre of the viewport.",
            is_enabled=self._can_zoom,
        )
        # -- 3D ----------------------------------------------------------------------
        # Every one of these is a no-op in a 2D figure, so they all gate on `_is_3d`
        # rather than being hidden: the palette showing a greyed "Top view" tells the
        # user 3D exists, where an absent entry tells them nothing.
        add(
            "view3d.toggle_mode",
            "Toggle 2D / 3D",
            "3D",
            self._act_toggle_ndim,
            chord="Mod+Shift+M",  # M for mode. Mod+Shift+D is "duplicate layer".
            icon="cube",
            keywords=("3d", "dimension", "projection", "mode", "two", "three"),
            description="Switch this figure between a 2D and a 3D view. A 3D figure keeps "
            "its own camera, axis box and plot types.",
            is_checked=self._is_3d,
        )
        for key, label, icon in _VIEW_DIGITS:
            digit = _VIEW_DIGIT_OF[key]
            add(
                f"view3d.preset_{key}",
                f"3D View: {label}",
                "3D",
                self._preset_setter(key),
                chord=digit,
                icon=icon,
                keywords=("3d", "camera", "orientation", "view", "angle", label.lower()),
                description=f"Look at the scene from {label.lower()}, keeping the current "
                f"zoom and pan. The bare {digit} key, so the standard angles are one "
                "keystroke each — press one, then K to key it.",
                is_enabled=self._is_3d,
            )
        add(
            "view3d.reset",
            "3D View: Reset",
            "3D",
            lambda: self.queue.submit(self.plot.reset_3d_view),
            chord="0",
            icon="home",
            keywords=("3d", "camera", "home", "default", "isometric"),
            description="Back to the default isometric framing: no dolly, no pan, no roll. "
            "The bare 0 key, next to the nine angles on 1-9.",
            is_enabled=self._is_3d,
        )
        add(
            "view3d.fit",
            "3D View: Fit Data",
            "3D",
            lambda: self.queue.submit(self.plot.frame_3d_view),
            icon="target",
            keywords=("3d", "camera", "frame", "zoom to fit", "bounds"),
            description="Re-fit the camera to the data, keeping the current orientation.",
            is_enabled=self._is_3d,
        )
        add(
            "view3d.toggle_projection",
            "3D: Perspective / Orthographic",
            "3D",
            self._act_toggle_projection,
            icon="perspective",
            keywords=("3d", "projection", "ortho", "persp", "parallel", "camera"),
            description="Swap the 3D projection. Orthographic keeps parallel lines "
            "parallel, so lengths stay comparable anywhere in the box.",
            is_enabled=self._is_3d,
            is_checked=lambda: self.plot.camera3d.projection == "orthographic",
        )
        add(
            "view3d.toggle_spin",
            "3D: Auto-rotate",
            "3D",
            self._act_toggle_spin,
            icon="spin",
            keywords=("3d", "turntable", "rotate", "animate", "spin"),
            description="Turn the scene continuously about the vertical axis.",
            is_enabled=self._is_3d,
            is_checked=lambda: bool(self.plot.camera3d.auto_spin),
        )
        add(
            "view3d.toggle_box_aspect",
            "3D: Cube / Data Aspect",
            "3D",
            self._act_toggle_box_aspect,
            icon="box_aspect",
            keywords=("3d", "aspect", "scale", "cube", "normalise", "equal"),
            description="Rescale the three axes to the same on-screen length, or back to "
            "raw data units. Changes the picture, not the numbers.",
            is_enabled=self._is_3d,
            is_checked=lambda: self.plot.camera3d.box_aspect is not None,
        )
        for label, name, icon in (
            ("Axis Box", "show_box", "cube"),
            ("Floor", "show_floor", "grid"),
            ("Grid Walls", "show_grid", "grid"),
            ("Tick Marks", "show_ticks", "axes3d"),
            ("Tick Numbers", "show_tick_labels", "info"),
        ):
            add(
                f"view3d.toggle_{name}",
                f"3D: Toggle {label}",
                "3D",
                self._axes3d_toggler(name),
                icon=icon,
                keywords=("3d", "axes", "decoration", label.lower()),
                description=f"Show or hide the 3D {label.lower()}.",
                is_enabled=self._is_3d,
                is_checked=lambda name=name: bool(getattr(self.plot.axes3d, name)),
            )

        add(
            "view.toggle_density",
            "Toggle Density Rendering",
            "View",
            self._act_toggle_density,
            icon="grid",
            keywords=("heatmap", "accumulator", "overplot", "millions"),
            description="Swap exact lines for the density accumulator. Also the D hotkey.",
            is_enabled=lambda: self._controller() is not None,
            is_checked=lambda: bool(getattr(self.plot, "display_density", False)),
        )
        add(
            "view.toggle_grid",
            "Toggle Grid",
            "View",
            self._option_toggler("axis_show_grid"),
            icon="grid",
            keywords=("axes", "lines", "background"),
            description="Show or hide the axis grid.",
            is_checked=self._option_reader("axis_show_grid"),
        )
        add(
            "view.toggle_labels",
            "Toggle Axis Labels",
            "View",
            self._option_toggler("axis_show_labels"),
            icon="info",
            keywords=("axes", "ticks", "numbers"),
            description="Show or hide the axis tick labels.",
            is_checked=self._option_reader("axis_show_labels"),
        )
        add(
            "view.toggle_frame",
            "Toggle Axis Frame",
            "View",
            self._option_toggler("axis_show_frame"),
            icon="chart_bar",
            keywords=("axes", "border", "box"),
            description="Show or hide the box drawn around the plot area.",
            is_checked=self._option_reader("axis_show_frame"),
        )
        add(
            "view.next_colormap",
            "Next Colormap",
            "View",
            lambda: self._act_cycle_scheme(1),
            icon="palette",
            keywords=("density", "scheme", "viridis", "colours", "colors"),
            description="Step forward through the 11 density colour schemes.",
            is_enabled=lambda: self._controller() is not None,
        )
        add(
            "view.prev_colormap",
            "Previous Colormap",
            "View",
            lambda: self._act_cycle_scheme(-1),
            icon="palette",
            keywords=("density", "scheme", "colours", "colors"),
            description="Step back through the 11 density colour schemes.",
            is_enabled=lambda: self._controller() is not None,
        )
        add(
            "view.toggle_hud",
            "Toggle HUD",
            "View",
            self._act_toggle_hud,
            icon="eye",
            keywords=("interface", "overlay", "hide ui", "chrome"),
            description="Hide the whole interface, panels included, to see the plot bare.",
            is_enabled=lambda: self._controller() is not None,
            is_checked=lambda: bool(getattr(self.plot.options, "enable_hud", False)),
        )

        # -- Data --------------------------------------------------------------------
        add(
            "data.new",
            "New Dataset",
            "Data",
            self._act_data_new,
            chord="Mod+N",
            icon="plus",
            keywords=("table", "sheet", "empty", "create"),
            description="Create an empty table and open it in the Data panel.",
            is_enabled=self._has_data_panel,
        )
        add(
            "data.paste_as_new",
            "Paste as New Dataset",
            "Data",
            self._act_data_paste_as_new,
            chord="Mod+Shift+V",
            icon="paste",
            keywords=("clipboard", "excel", "tsv", "import", "numbers"),
            description="Turn the clipboard's table into a fresh dataset. Excel pastes as-is.",
            is_enabled=self._has_data_panel,
        )
        add(
            "data.duplicate",
            "Duplicate Dataset",
            "Data",
            self._act_data_duplicate,
            icon="copy",
            keywords=("copy", "clone", "table"),
            description="Copy the current dataset, unbound from any layer.",
            is_enabled=self._has_dataset,
        )
        add(
            "data.delete",
            "Delete Dataset",
            "Data",
            self._act_data_delete,
            icon="trash",
            keywords=("remove", "drop", "table"),
            description="Delete the current dataset. Any layer it plotted stays on the scene.",
            is_enabled=self._has_dataset,
        )
        add(
            "data.import_csv",
            "Import CSV",
            "Data",
            self._act_data_import,
            chord="Mod+O",
            icon="folder",
            keywords=("open", "file", "load", "tsv", "read"),
            description="Read the CSV at the path in the Data panel's I/O section.",
            is_enabled=self._has_data_panel,
        )
        add(
            "data.export_csv",
            "Export CSV",
            "Data",
            self._act_data_export,
            chord="Mod+E",
            icon="download",
            keywords=("save", "write", "file"),
            description="Write the current dataset to the path in the Data panel's I/O section.",
            is_enabled=self._has_dataset,
        )
        add(
            "data.plot",
            "Plot Dataset",
            "Data",
            self._act_data_plot,
            chord="Mod+Enter",
            icon="chart_line",
            keywords=("draw", "curve", "scatter", "link", "x y"),
            description="Plot the current dataset's x/y columns as a layer bound to the table.",
            is_enabled=self._has_dataset,
        )
        add(
            "data.add_row",
            "Add Row",
            "Data",
            self._act_data_add_row,
            icon="plus",
            keywords=("insert", "table", "grid"),
            description="Insert a zero row above the selection.",
            is_enabled=self._has_dataset,
        )
        add(
            "data.add_column",
            "Add Column",
            "Data",
            self._act_data_add_column,
            icon="plus",
            keywords=("insert", "field", "series", "table"),
            description="Append a zero column named after the Data panel's column field.",
            is_enabled=self._has_dataset,
        )
        # No chord: Mod+A belongs to whichever surface has the keyboard, and while the
        # Data panel is focused the grid already handles Ctrl+A itself
        # (`data_editor._handle_keys`, which returns early when unfocused). Keeping the
        # chord here would have bound Mod+A to two enabled actions the moment the grid
        # had focus — `registry.conflicts()` must stay empty — while doing nothing the
        # grid was not already doing. See `sel.select_all` for the other half.
        add(
            "data.select_all",
            "Select All Cells",
            "Data",
            self._act_data_select_all,
            icon="table",
            keywords=("selection", "grid", "everything"),
            description="Select the whole table, ready to copy or clear.",
            is_enabled=self._has_dataset,
        )

        # -- Selection ---------------------------------------------------------------
        # Mod+A is scoped by focus rather than duplicated: it is enabled only where the
        # canvas selection is the thing the user means. `_select_all_has_focus` excludes
        # the Data panel (whose grid owns Ctrl+A itself) and every other panel, so a
        # Mod+A typed into a text field goes to that field — imgui's own input_text does
        # select-all — instead of silently selecting a million points behind it. Exactly
        # one action carries the chord, so `registry.conflicts()` is empty by
        # construction rather than by luck.
        add(
            "sel.select_all",
            "Select All Points",
            "Selection",
            self._act_sel_select_all,
            chord="Mod+A",
            icon="check",
            keywords=("selection", "everything", "canvas", "points", "marquee"),
            description="Select every element in the Selection panel's scope.",
            is_enabled=self._select_all_enabled,
        )
        add(
            "sel.select_none",
            "Select None",
            "Selection",
            self._act_sel_select_none,
            chord="Mod+Shift+A",
            icon="close",
            keywords=("selection", "deselect", "clear", "nothing"),
            description="Clear the canvas selection.",
            is_enabled=self._has_canvas_selection,
        )
        add(
            "sel.invert",
            "Invert Selection",
            "Selection",
            self._act_sel_invert,
            # Not Mod+I: that is math.integrate. Mod+Shift+I is both free and the
            # binding every image editor already uses for Invert Selection.
            chord="Mod+Shift+I",
            icon="refresh",
            keywords=("selection", "flip", "opposite", "swap"),
            description="Select what is not selected, within the panel's scope.",
            is_enabled=self._has_selectable_layers,
        )
        add(
            "sel.create_dataset",
            "Create Dataset from Selection",
            "Selection",
            self._act_sel_create_dataset,
            icon="table",
            keywords=("selection", "extract", "table", "data", "new"),
            description="Copy the selected points into a new dataset in the Data panel.",
            is_enabled=self._has_pointwise_selection,
        )
        add(
            "sel.delete",
            "Delete Selected Points",
            "Selection",
            self._act_sel_delete,
            icon="trash",
            keywords=("selection", "remove", "erase", "points"),
            description="Delete the selected points from their layers.",
            is_enabled=self._has_pointwise_selection,
        )
        add(
            "sel.isolate",
            "Isolate Selection",
            "Selection",
            self._act_sel_isolate,
            icon="filter",
            keywords=("selection", "hide", "only", "focus"),
            description="Hide every layer that has nothing selected.",
            is_enabled=self._has_canvas_selection,
        )
        add(
            "sel.zoom",
            "Zoom to Selection",
            "Selection",
            self._act_sel_zoom,
            icon="zoom_in",
            keywords=("selection", "frame", "fit", "view"),
            description="Frame the selected points in the viewport.",
            is_enabled=self._has_pointwise_selection,
        )

        # -- Functions ---------------------------------------------------------------
        add(
            "func.new",
            "New Function",
            "Functions",
            self._act_func_new,
            icon="function",
            keywords=("expression", "formula", "curve", "f(x)"),
            description="Reset the Functions panel to a fresh expression and open it.",
            is_enabled=self._has_functions_panel,
        )
        add(
            "func.plot",
            "Plot Function",
            "Functions",
            self._act_func_plot,
            chord="Mod+Shift+Enter",
            icon="chart_line",
            keywords=("expression", "draw", "evaluate", "curve"),
            description="Evaluate the Functions panel's expression and add it as a layer.",
            is_enabled=self._has_functions_panel,
        )
        add(
            "func.insert_preset",
            "Insert Function Preset…",
            "Functions",
            self._act_func_presets,
            icon="wave",
            keywords=("template", "example", "sine", "gaussian", "library"),
            description="Browse the function presets in the palette and plot one.",
            is_enabled=lambda: self.palette is not None and "functions" in self.panels,
        )

        # -- Math --------------------------------------------------------------------
        for op_id, op_key, op_title, op_chord, op_icon, op_words, op_desc in _MATH_OPS:
            add(
                op_id,
                op_title,
                "Math",
                self._math_opener(op_key),
                chord=op_chord,
                icon=op_icon,
                keywords=op_words,
                description=op_desc,
                is_enabled=self._has_mathlab_panel,
            )

        # -- Layer -------------------------------------------------------------------
        add(
            "layer.delete",
            "Delete Layer",
            "Layer",
            self._act_layer_delete,
            chord="Mod+Delete",
            icon="trash",
            keywords=("remove", "drop", "scene"),
            description="Remove the selected layer, releasing its GPU buffers. Undoable.",
            is_enabled=self._has_selected_layer,
        )
        add(
            "layer.duplicate",
            "Duplicate Layer",
            "Layer",
            self._act_layer_duplicate,
            chord="Mod+Shift+D",
            icon="copy",
            keywords=("copy", "clone", "scene"),
            description="Clone the selected layer, geometry and style included.",
            is_enabled=self._has_selected_layer,
        )
        add(
            "layer.toggle_visibility",
            "Toggle Layer Visibility",
            "Layer",
            self._act_layer_visibility,
            icon="eye",
            keywords=("hide", "show", "eye", "scene"),
            description="Hide or show the selected layer without deleting it.",
            is_enabled=self._has_selected_layer,
            is_checked=self._selected_layer_visible,
        )
        add(
            "layer.solo",
            "Isolate Layer",
            "Layer",
            self._act_layer_solo,
            icon="filter",
            keywords=("solo", "only", "focus", "hide others"),
            description="Show only the selected layer; run it again to restore the rest.",
            is_enabled=lambda: self._has_selected_layer() and "scene" in self.panels,
        )
        add(
            "layer.rename",
            "Rename Layer",
            "Layer",
            self._act_layer_rename,
            chord="F2",
            icon="link",
            keywords=("label", "name", "title"),
            description="Edit the selected layer's label inline in the Scene panel.",
            is_enabled=lambda: self._has_selected_layer() and "scene" in self.panels,
        )
        add(
            "layer.move_up",
            "Move Layer Up",
            "Layer",
            lambda: self._act_layer_move(-1),
            chord="Mod+Shift+Up",
            icon="chevron_up",
            keywords=("reorder", "order", "raise", "front"),
            description="Move the selected layer earlier in the list. Style's zorder still wins.",
            is_enabled=self._has_selected_layer,
        )
        add(
            "layer.move_down",
            "Move Layer Down",
            "Layer",
            lambda: self._act_layer_move(1),
            chord="Mod+Shift+Down",
            icon="chevron_down",
            keywords=("reorder", "order", "lower", "back"),
            description="Move the selected layer later in the list. Style's zorder still wins.",
            is_enabled=self._has_selected_layer,
        )
        add(
            "layer.select_next",
            "Select Next Layer",
            "Layer",
            lambda: self._act_layer_step(1),
            chord="Alt+Down",
            icon="chevron_down",
            keywords=("selection", "cycle", "scene"),
            description="Select the next layer in the scene, wrapping at the end.",
            is_enabled=self._has_layers,
        )
        add(
            "layer.select_prev",
            "Select Previous Layer",
            "Layer",
            lambda: self._act_layer_step(-1),
            chord="Alt+Up",
            icon="chevron_up",
            keywords=("selection", "cycle", "scene"),
            description="Select the previous layer in the scene, wrapping at the start.",
            is_enabled=self._has_layers,
        )

        # -- Animation ---------------------------------------------------------------
        # Chord notes, all checked against the whole table above:
        #   * `Space` is the transport binding everybody already has in their fingers, and
        #     it is safe here because `keys.is_text_safe` is False for a bare key: the
        #     dispatcher suppresses it for as long as any text field holds the keyboard, so
        #     typing a space into a layer name cannot start playback.
        #   * The frame and scene steps carry Alt for the same reason `layer.select_next`
        #     does — Mod+Left/Right belong to the text fields' own word-jump.
        #   * F5 is what every slide tool binds "present" to, and a function key stays live
        #     while typing, which is exactly right for the one action a presenter needs.
        add(
            "anim.timeline",
            "Animation: Timeline Editor",
            "Animation",
            self._panel_toggler("timeline"),
            icon="timeline",
            keywords=("animate", "keyframe", "track", "scrub", "playhead", "transport"),
            description="Show the Timeline panel: transport, scrubber, keyframe tracks and "
            f"easing. Also {_chord_text(_PANEL_CHORDS['timeline'])}.",
            is_enabled=self._has_timeline_panel,
            is_checked=lambda: bool(self.open.get("timeline", False)),
        )
        add(
            "anim.presentation",
            "Animation: Presentation Editor",
            "Animation",
            self._panel_toggler("presentation"),
            icon="play",
            keywords=("scenes", "chapters", "slides", "talk", "present", "story"),
            description=(
                "Show the Presentation panel: the timeline's scenes as chapters you can "
                "name, reorder and step through. Also "
                f"{_chord_text(_PANEL_CHORDS['presentation'])}."
            ),
            is_enabled=self._has_presentation_panel,
            is_checked=lambda: bool(self.open.get("presentation", False)),
        )
        add(
            "anim.play_pause",
            "Play / Pause",
            "Animation",
            self._act_anim_play,
            chord="Space",
            icon="play",
            keywords=("animate", "transport", "run", "stop", "playback"),
            description="Start or stop playback of this panel's timeline. Opens the "
            "Timeline panel, which is what drives the playhead.",
            is_enabled=self._has_timeline_panel,
            is_checked=self._is_playing,
        )
        add(
            "anim.stop",
            "Stop and Rewind",
            "Animation",
            self._act_anim_stop,
            chord="Mod+.",
            icon="stop",
            keywords=("animate", "transport", "rewind", "reset", "start"),
            description="Stop playback and put the playhead back at zero.",
            is_enabled=self._has_timeline_panel,
        )
        add(
            "anim.next_frame",
            "Next Frame",
            "Animation",
            lambda: self._act_anim_step(1),
            chord="Alt+Right",
            icon="step_forward",
            keywords=("animate", "step", "frame", "forward", "advance"),
            description="Advance the playhead one frame at the timeline's frame rate.",
            is_enabled=self._has_timeline_panel,
        )
        add(
            "anim.prev_frame",
            "Previous Frame",
            "Animation",
            lambda: self._act_anim_step(-1),
            chord="Alt+Left",
            icon="step_back",
            keywords=("animate", "step", "frame", "back", "previous"),
            description="Move the playhead back one frame.",
            is_enabled=self._has_timeline_panel,
        )
        add(
            "anim.next_scene",
            "Next Scene",
            "Animation",
            lambda: self._act_anim_scene(1),
            chord="Alt+PageDown",
            icon="chevron_right",
            keywords=("presentation", "chapter", "slide", "advance", "clicker"),
            description="Jump to the next scene — the presentation's 'click to continue'.",
            is_enabled=self._has_scenes,
        )
        add(
            "anim.prev_scene",
            "Previous Scene",
            "Animation",
            lambda: self._act_anim_scene(-1),
            chord="Alt+PageUp",
            icon="chevron_left",
            keywords=("presentation", "chapter", "slide", "back"),
            description="Jump back to the previous scene.",
            is_enabled=self._has_scenes,
        )
        add(
            "anim.capture",
            "Capture Keyframe",
            "Animation",
            self._act_anim_capture,
            chord="K",
            icon="keyframe",
            keywords=("animate", "record", "key", "capture", "pose", "snapshot", "view"),
            description="Record the figure as it looks right now at the playhead, creating "
            "whatever tracks are needed. Works on an empty timeline, which is what makes it "
            "the first thing to press: aim the view, K, move the playhead, aim again, K.",
            is_enabled=self._has_timeline_panel,
        )
        add(
            "anim.add_key",
            "Add Keyframe on the Selected Track",
            "Animation",
            self._act_anim_key,
            chord="Mod+Shift+K",
            icon="keyframe",
            keywords=("animate", "record", "key", "track", "selected"),
            description="Record the current value of the selected track at the playhead — "
            "or of every track when none is selected. Creates nothing; use Capture for "
            "that.",
            is_enabled=self._has_tracks,
        )
        # One palette entry per preset. Registered from the panel module's own table, not
        # from a copy of it here: a preset that existed in the panel and not in the palette
        # would be a feature only the people who already found the panel can use.
        for preset in self._animation_presets():
            add(
                f"anim.preset_{preset.key}",
                f"Animation Preset: {preset.label}",
                "Animation",
                self._preset_builder(preset.key),
                icon=preset.icon,
                keywords=("animate", "preset", "quick", "build", "keyframes", preset.key),
                description=preset.description,
                is_enabled=self._preset_enabler(preset.key),
            )
        add(
            "anim.present",
            "Start Presentation",
            "Animation",
            self._act_anim_present,
            chord="F5",
            icon="play",
            keywords=("presentation", "slides", "talk", "full", "show", "run"),
            description="Enter presentation playback from the current scene, holding at the "
            "end of each one so a click advances. Press it again to leave.",
            is_enabled=self._has_scenes,
            is_checked=self._is_presenting,
        )

    # -- action predicates -----------------------------------------------------

    def _has_data_panel(self) -> bool:
        """True when the Data panel imported and constructed."""
        return "data" in self.panels

    def _has_functions_panel(self) -> bool:
        """True when the Functions panel imported and constructed."""
        return "functions" in self.panels

    def _has_mathlab_panel(self) -> bool:
        """True when the Math Lab panel imported and constructed."""
        return "mathlab" in self.panels

    def _has_dataset(self) -> bool:
        """True when the Data panel exists and the store has something to act on."""
        return "data" in self.panels and len(self.store) > 0

    # -- selection -------------------------------------------------------------
    # The panel owns the verbs (scope, undo entries, dirty flags); these are thin
    # adapters so the same code answers the button, the palette and the chord.

    def _selection_panel(self) -> Optional[Any]:
        """The Selection panel, or None if its module would not import."""
        return self.panels.get("selection")

    def _has_selectable_layers(self) -> bool:
        """True when at least one visible layer contributes element ids to pick."""
        panel = self._selection_panel()
        return panel is not None and bool(panel.scoped_layers())

    def _has_canvas_selection(self) -> bool:
        """True when anything is selected on the canvas."""
        panel = self._selection_panel()
        return panel is not None and panel.has_selection()

    def _has_pointwise_selection(self) -> bool:
        """True when the selection contains real x/y points, not just lines or patches."""
        panel = self._selection_panel()
        return panel is not None and panel.has_pointwise_selection()

    def _select_all_enabled(self) -> bool:
        """Whether Mod+A currently means "select all on the canvas".

        Only when the canvas or the Selection panel itself has the keyboard. Any other
        focused panel keeps its own Ctrl+A: the Data grid selects its cells
        (`data_editor._handle_keys`), and an ``input_text`` anywhere selects its text —
        both of which a user pressing Cmd+A in that window plainly meant.
        """
        if self._focused_panel not in (None, "selection"):
            return False
        return self._has_selectable_layers()

    def _act_sel_select_all(self) -> None:
        """Select every element in the Selection panel's scope."""
        panel = self._selection_panel()
        if panel is not None:
            panel.select_all()

    def _act_sel_select_none(self) -> None:
        """Clear the canvas selection."""
        panel = self._selection_panel()
        if panel is not None:
            panel.select_none()

    def _act_sel_invert(self) -> None:
        """Invert the selection within the panel's scope."""
        panel = self._selection_panel()
        if panel is not None:
            panel.invert()

    def _act_sel_create_dataset(self) -> None:
        """Copy the selected points into a new dataset."""
        panel = self._selection_panel()
        if panel is not None:
            panel.create_dataset()

    def _act_sel_delete(self) -> None:
        """Delete the selected points from their layers."""
        panel = self._selection_panel()
        if panel is not None:
            panel.delete_selected()

    def _act_sel_isolate(self) -> None:
        """Hide every layer that has nothing selected."""
        panel = self._selection_panel()
        if panel is not None:
            panel.isolate()

    def _act_sel_zoom(self) -> None:
        """Frame the selected points."""
        panel = self._selection_panel()
        if panel is not None:
            panel.zoom_to_selection()

    def _has_layers(self) -> bool:
        """True when the scene has at least one layer."""
        try:
            return bool(self.plot.scene.layers)
        except Exception:  # pragma: no cover - defensive: a plot without a scene
            return False

    def _has_selected_layer(self) -> bool:
        """True when a layer is selected and still in the scene."""
        return self._selected_layer() is not None

    def _selected_layer_visible(self) -> bool:
        """Visibility of the selected layer, for the toggle's check mark."""
        layer = self._selected_layer()
        if layer is None:
            return False
        return bool(getattr(getattr(layer, "style", None), "visible", True))

    def _can_zoom(self) -> bool:
        """True when the engine has a camera controller and a real viewport to zoom in."""
        if getattr(self.plot, "camera_controller", None) is None:
            return False
        return float(getattr(self.plot, "width", 0) or 0) > 0.0

    # -- action helpers --------------------------------------------------------

    def _selected_layer_id(self) -> Optional[int]:
        """The selected layer id, preferring the HUD's copy (it drives the sync block)."""
        state = self._hud_state()
        if state is not None:
            return getattr(state, "selected_layer_id", None)
        interaction = getattr(self.plot, "interaction", None)
        return getattr(interaction, "selected_layer_id", None)

    def _selected_layer(self) -> Optional[Any]:
        """The selected layer object, or None if nothing is selected or it is gone."""
        return layerops.find_layer(self.plot, self._selected_layer_id())

    def _select_layer(self, layer: Any) -> None:
        """Select ``layer``, mirroring ``ScenePanel._select``.

        The HUD's copy wins where a HUD exists: ``hud.py:122-136`` propagates it to
        ``plot.interaction`` on the next frame, and writing both would make that block read
        the change as an engine-side pick and stop honouring later UI selections. Selection
        is not scene state, so this needs no queue.
        """
        layer_id = getattr(layer, "layer_id", None)
        if layer_id is None:
            return
        state = self._hud_state()
        if state is not None:
            state.selected_layer_id = layer_id
            return
        interaction = getattr(self.plot, "interaction", None)
        if interaction is not None:
            interaction.selected_layer_id = layer_id

    def open_panel(self, key: str) -> Optional[Panel]:
        """Open panel ``key``, focus it next frame, and return it (None if it is missing).

        Every action that acts on a panel calls this first. A shortcut that mutated an
        invisible panel would be indistinguishable from one that did nothing at all.
        """
        panel = self.panels.get(key)
        if panel is None:
            self._status = f"The {key} panel is unavailable."
            return None
        self.open[key] = True
        self._focus(key)
        return panel

    def _panel_toggler(self, key: str) -> Callable[[], None]:
        """The run for a ``panel.*`` action: open+focus, or hide if it already has focus.

        Hiding only on *focus* rather than on "already open" is what makes the chord a
        usable toggle in a multi-window layout: ``Mod+1`` from the Functions panel brings
        Data forward (the useful thing) instead of closing a window you were not looking at.
        """

        def run() -> None:
            if self.open.get(key, False) and self._focused_panel == key:
                self.open[key] = False
                if self._focused_panel == key:
                    self._focused_panel = None
                return
            self.open_panel(key)

        return run

    def _option_toggler(self, name: str) -> Callable[[], None]:
        """A run that flips the boolean ``options.<name>`` through the queue."""
        options = self.plot.options

        def run() -> None:
            def apply() -> None:
                setattr(options, name, not bool(getattr(options, name, False)))

            self.queue.submit(apply)

        return run

    def _option_reader(self, name: str) -> Callable[[], bool]:
        """An ``is_checked`` predicate reading the live ``options.<name>``."""
        return lambda: bool(getattr(self.plot.options, name, False))

    # -- 3D actions ------------------------------------------------------------
    #
    # Every one of these queues its work: ``set_3d_view`` rebuilds the axis-box geometry,
    # which is scene mutation, and CONTRACT §1.1 forbids that from a draw callback just as
    # firmly for the camera as for a layer.

    def _is_3d(self) -> bool:
        """Whether the current figure is 3D. The gate on every 3D action."""
        try:
            return bool(self.plot.is_3d_scene())
        except Exception:  # pragma: no cover - defensive
            return False

    def _act_toggle_ndim(self) -> None:
        """Flip the figure between 2D and 3D, pinning it either way.

        Pinned rather than reset to "auto": the user asking for 3D on an empty figure
        wants the 3D tools *now*, and auto would answer 2D until a 3D layer existed.
        """
        plot = self.plot
        target = 2 if self._is_3d() else 3

        def apply() -> None:
            plot.set_ndim(target)

        self.queue.submit(apply)

    def _preset_setter(self, key: str) -> Callable[[], None]:
        """The run for a named 3D orientation."""
        plot = self.plot

        def run() -> None:
            def apply() -> None:
                plot.set_3d_preset(key)

            self.queue.submit(apply)

        return run

    def _act_toggle_projection(self) -> None:
        plot = self.plot

        def apply() -> None:
            current = plot.camera3d.projection
            plot.set_3d_view(
                projection="perspective" if current == "orthographic" else "orthographic"
            )

        self.queue.submit(apply)

    def _act_toggle_spin(self) -> None:
        plot = self.plot

        def apply() -> None:
            camera = plot.camera3d
            camera.auto_spin = 0.0 if camera.auto_spin else 24.0
            plot.frame.dirty_scene = True

        self.queue.submit(apply)

    def _act_toggle_box_aspect(self) -> None:
        plot = self.plot

        def apply() -> None:
            camera = plot.camera3d
            camera.box_aspect = None if camera.box_aspect is not None else (1.0, 1.0, 1.0)
            plot.set_3d_view()

        self.queue.submit(apply)

    def _axes3d_toggler(self, name: str) -> Callable[[], None]:
        """The run for one of the 3D decoration switches."""
        plot = self.plot

        def run() -> None:
            def apply() -> None:
                axes = plot.axes3d
                setattr(axes, name, not bool(getattr(axes, name, False)))
                plot.set_3d_view()

            self.queue.submit(apply)

        return run

    # -- Animation actions -----------------------------------------------------
    #
    # Every one of these goes *through the panel* rather than reaching for
    # ``plot.active_panel.timeline`` directly. That is not indirection for its own sake:
    # the panels own the queueing, the frame snapping and the selection, so a chord that
    # bypassed them would seek to an unsnapped time and key a track the user never picked.
    # It also means playback keeps running, because only a drawn panel advances the
    # playhead (see ``panels/timeline.queue_playback``).

    def _timeline_panel(self) -> Optional[Any]:
        """The Timeline panel, or None if its module would not import."""
        return self.panels.get("timeline")

    def _presentation_panel(self) -> Optional[Any]:
        """The Presentation panel, or None if its module would not import."""
        return self.panels.get("presentation")

    def _has_timeline_panel(self) -> bool:
        return "timeline" in self.panels

    def _has_presentation_panel(self) -> bool:
        return "presentation" in self.panels

    def _timeline(self) -> Optional[Any]:
        """The active panel's :class:`~glplot.core.timeline.Timeline`, or None."""
        try:
            return self.plot.active_panel.timeline
        except Exception:  # pragma: no cover - defensive: a plot without panels
            return None

    def _is_playing(self) -> bool:
        """Whether the timeline is running, for the transport's check mark."""
        timeline = self._timeline()
        return bool(getattr(timeline, "playing", False))

    def _is_presenting(self) -> bool:
        """Whether presentation playback is running."""
        panel = self._presentation_panel()
        return bool(getattr(panel, "_presenting", False))

    def _has_tracks(self) -> bool:
        """True when there is something keyable to key."""
        timeline = self._timeline()
        return bool(getattr(timeline, "tracks", None)) and self._has_timeline_panel()

    def _has_scenes(self) -> bool:
        """True when the timeline has chapters to step through."""
        timeline = self._timeline()
        return bool(getattr(timeline, "scenes", None)) and self._has_presentation_panel()

    def _act_anim_play(self) -> None:
        """Toggle playback, opening the Timeline panel because it is what advances it."""
        panel = self.open_panel("timeline")
        if panel is not None:
            panel.toggle_play()

    def _act_anim_stop(self) -> None:
        panel = self._timeline_panel()
        if panel is not None:
            panel.stop()

    def _act_anim_step(self, delta: int) -> None:
        """Step the playhead. No panel opened: the frame changes on screen regardless."""
        panel = self._timeline_panel()
        if panel is not None:
            panel.step_frames(int(delta))

    def _act_anim_key(self) -> None:
        """Key at the playhead, opening the Timeline so the new diamond is visible."""
        panel = self.open_panel("timeline")
        if panel is None:
            return
        count = panel.key_at_playhead()
        self._status = f"Keyed {count} track{'s' if count != 1 else ''}."

    def _act_anim_capture(self) -> None:
        """Capture the figure's current state, opening the Timeline to show the result.

        The panel is opened rather than merely used: a capture that created three tracks
        behind a closed window would look exactly like a shortcut that did nothing, and
        this is the one action a user is most likely to press before they know the panel
        exists.
        """
        panel = self.open_panel("timeline")
        if panel is None:
            return
        count = panel.capture()
        if count:
            self._status = f"Captured {count} propert{'y' if count == 1 else 'ies'}."
        else:
            self._status = "Nothing to capture in this figure."

    def _animation_presets(self) -> List[Any]:
        """The Timeline panel's preset table, or empty if its module would not import.

        Read through the module rather than the panel instance: the actions are registered
        once at construction, and a panel that failed to import must cost a missing menu
        section rather than a missing workspace.
        """
        try:
            from .panels.timeline import PRESETS

            return list(PRESETS)
        except Exception:  # pragma: no cover - defensive: GL-less import
            return []

    def _preset_builder(self, key: str) -> Callable[[], None]:
        """The run for one preset: open the Timeline and build it there."""

        def run() -> None:
            panel = self.open_panel("timeline")
            if panel is None:
                return
            if panel.build_preset(key):
                self._status = f"Built the {key.replace('_', ' ')} animation."
            else:
                self._status = getattr(panel, "_error", "") or "That preset does not apply here."

        return run

    def _preset_enabler(self, key: str) -> Callable[[], bool]:
        """Whether a preset can do anything in this figure — the palette greys it if not."""

        def enabled() -> bool:
            if not self._has_timeline_panel():
                return False
            try:
                from .panels.timeline import PRESET_BY_KEY, preset_available

                preset = PRESET_BY_KEY.get(key)
                return preset is not None and preset_available(self.plot, preset)
            except Exception:  # pragma: no cover - defensive
                return False

        return enabled

    def _act_anim_scene(self, delta: int) -> None:
        panel = self._presentation_panel()
        if panel is None:
            return
        panel.next_scene() if delta > 0 else panel.previous_scene()

    def _act_anim_present(self) -> None:
        """Enter or leave presentation playback."""
        panel = self.open_panel("presentation")
        if panel is not None:
            panel.toggle_presentation()

    def _math_opener(self, op_key: str) -> Callable[[], None]:
        """The run for a Math action: open Math Lab on that operation's tab.

        The operation is not applied. Every Math Lab op previews before it commits, and a
        shortcut that silently rewrote the selected layer would be the one thing the panel
        was built to prevent — the chord takes you to the operation, Apply is still yours.
        """

        def run() -> None:
            panel = self.open_panel("mathlab")
            if panel is None:
                return
            show = getattr(panel, "show_operation", None)
            if callable(show):
                show(op_key)

        return run

    def _data_target(self) -> Tuple[Optional[Any], Optional[Any]]:
        """The Data panel and its current dataset, ready for a mutation this frame.

        Clears the panel's ``_acted`` gate first. That flag exists so two clicks in one
        frame cannot each snapshot the same pre-state, and the panel clears it at the top of
        its own ``draw`` — but actions dispatch *before* that, so a True left over from the
        previous frame would silently swallow this shortcut. Clearing it here restores the
        invariant the flag actually encodes: at most one grid mutation per frame.
        """
        panel = self.panels.get("data")
        if panel is None:
            return (None, None)
        panel._acted = False
        return (panel, panel._current_dataset())

    # -- action runs -----------------------------------------------------------

    def _act_palette(self) -> None:
        """Open the palette, or close it if it is already up."""
        if self.palette is not None:
            self.palette.toggle()

    def _act_help(self, topic: Optional[str]) -> None:
        """Open Help, at ``topic`` when the panel offers a way in."""
        panel = self.open_panel("help")
        if panel is None:
            return
        show = getattr(panel, "show", None)
        if callable(show):
            show(topic)

    def _act_export_png(self) -> None:
        """Queue a PNG export. Queued because savefig reads the live scene."""
        controller = self._controller()
        if controller is None:
            return
        self.queue.submit(controller.export)
        self._status = "Exporting PNG…"

    def _act_quit(self) -> None:
        """Ask GLFW to close the window. The main loop notices on its next iteration."""
        window = getattr(self.plot, "window", None)
        if window is None or not GLFW_AVAILABLE:
            return
        try:
            glfw.set_window_should_close(window, True)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Could not close the window.")

    def _act_toggle_profiler(self) -> None:
        """Flip the HUD's profiler overlay. Pure HUD state; no queue needed."""
        state = self._hud_state()
        if state is None:
            return
        state.show_profiler = not bool(getattr(state, "show_profiler", False))

    def _act_close_focused(self) -> None:
        """Hide the focused panel window."""
        key = self._focused_panel
        if key in self.panels:
            self.open[key] = False
            self._focused_panel = None

    def _act_cycle_panels(self) -> None:
        """Focus the next open panel after the focused one, wrapping."""
        order = [key for key in self.panels if self.open.get(key, False)]
        if not order:
            return
        try:
            index = order.index(self._focused_panel) if self._focused_panel in order else -1
        except ValueError:  # pragma: no cover - the membership test precedes this
            index = -1
        target = order[(index + 1) % len(order)]
        self._focus(target)
        self._focused_panel = target

    def _act_autoscale(self) -> None:
        """Queue an autoscale (CONTRACT §1.9: never call it outside the queue)."""
        controller = self._controller()
        if controller is not None:
            self.queue.submit(controller.autoscale)

    def _act_reset_view(self) -> None:
        """Queue a camera reset."""
        controller = self._controller()
        if controller is not None:
            self.queue.submit(controller.reset_view)

    def _act_split(self, nrows: int, ncols: int) -> None:
        """Queue a re-tile of the window into an ``nrows`` x ``ncols`` grid of panels."""
        plot = self.plot
        if getattr(plot, "split_view", None) is None:
            return

        def apply() -> None:
            plot.split_view(nrows, ncols)

        self.queue.submit(apply)

    def _act_merge(self) -> None:
        """Queue a collapse back to a single full-window panel."""
        plot = self.plot
        if getattr(plot, "merge_view", None) is None:
            return

        def apply() -> None:
            plot.merge_view()

        self.queue.submit(apply)

    def _act_zoom(self, direction: float) -> None:
        """Queue a zoom about the viewport centre, matching the engine's `=`/`-` keys."""
        plot = self.plot

        def apply() -> None:
            controller = getattr(plot, "camera_controller", None)
            width = float(getattr(plot, "width", 0) or 0)
            height = float(getattr(plot, "height", 0) or 0)
            if controller is None or width <= 0.0 or height <= 0.0:
                return
            step = float(getattr(plot.options, "zoom_scroll_factor", 1.1) or 1.1)
            factor = step if direction > 0 else 1.0 / step
            controller.apply_zoom_at_cursor(factor, width * 0.5, height * 0.5, width, height)

        self.queue.submit(apply)

    def _act_toggle_density(self) -> None:
        """Queue the density toggle."""
        controller = self._controller()
        if controller is not None:
            self.queue.submit(controller.toggle_density)

    def _act_cycle_scheme(self, direction: int) -> None:
        """Queue a density colormap step."""
        controller = self._controller()
        if controller is not None:
            self.queue.submit(lambda: controller.cycle_scheme(direction))

    def _act_toggle_hud(self) -> None:
        """Queue the HUD toggle. It hides this whole interface — the palette says so."""
        controller = self._controller()
        if controller is not None:
            self.queue.submit(controller.toggle_hud)

    def _act_data_new(self) -> None:
        """Create an empty dataset and show it."""
        panel, _ = self._data_target()
        if panel is None:
            return
        panel._new_dataset()
        self.open_panel("data")

    def _act_data_paste_as_new(self) -> None:
        """Turn the clipboard's table into a new dataset and show it."""
        panel, _ = self._data_target()
        if panel is None:
            return
        panel._paste_as_new_dataset()
        self.open_panel("data")

    def _act_data_duplicate(self) -> None:
        """Duplicate the current dataset."""
        panel, dataset = self._data_target()
        if panel is None or dataset is None:
            return
        panel._duplicate_dataset()
        self.open_panel("data")

    def _act_data_delete(self) -> None:
        """Delete the current dataset. The Data panel's own button confirms first; this
        does not — a shortcut the user typed deliberately is its own confirmation, and undo
        is one keystroke away."""
        panel, dataset = self._data_target()
        if panel is None or dataset is None:
            return
        panel._delete_dataset()
        self.open_panel("data")

    def _act_data_import(self) -> None:
        """Import the CSV at the Data panel's path field."""
        panel, _ = self._data_target()
        if panel is None:
            return
        self.open_panel("data")
        panel._import_csv()

    def _act_data_export(self) -> None:
        """Export the current dataset to the Data panel's path field."""
        panel, dataset = self._data_target()
        if panel is None or dataset is None:
            return
        self.open_panel("data")
        panel._export_csv(dataset)

    def _act_data_plot(self) -> None:
        """Plot the current dataset with the Data panel's chosen x/y columns."""
        panel, dataset = self._data_target()
        if panel is None or dataset is None:
            return
        self.open_panel("data")
        panel._plot_dataset(dataset)

    def _act_data_add_row(self) -> None:
        """Insert a zero row above the selection."""
        panel, dataset = self._data_target()
        if panel is None or dataset is None:
            return
        self.open_panel("data")
        panel._insert_row(dataset)

    def _act_data_add_column(self) -> None:
        """Append a zero column, named from the Data panel's column field."""
        panel, dataset = self._data_target()
        if panel is None or dataset is None:
            return
        self.open_panel("data")
        name = str(getattr(panel, "_new_column_name", "") or "z")
        panel._add_column(dataset, name)

    def _act_data_select_all(self) -> None:
        """Select the whole table, exactly as the grid's own Ctrl+A does."""
        panel, dataset = self._data_target()
        if panel is None or dataset is None:
            return
        panel._anchor = (0, 0)
        panel._cursor = (max(0, dataset.n_rows() - 1), max(0, dataset.n_cols() - 1))
        self.open_panel("data")

    def _act_func_new(self) -> None:
        """Reset the Functions panel to a fresh expression."""
        panel = self.open_panel("functions")
        if panel is None:
            return
        panel.expr = "a * sin(b * x)"
        # _sync_params re-derives the slider list from the new text. Without it the panel
        # would keep the previous expression's parameters until the user typed a character.
        panel._sync_params()

    def _act_func_plot(self) -> None:
        """Evaluate the Functions panel's expression and add it as a layer."""
        panel = self.open_panel("functions")
        if panel is None:
            return
        panel._sync_params()
        panel._action_plot()

    def _act_func_presets(self) -> None:
        """Open the palette on the ``=`` provider: presets, plus a live expression."""
        if self.palette is None:
            return
        self.palette.open()
        # open() resets the query by design; seed it afterwards so the palette lands on the
        # function provider with an empty search rather than on the command list.
        self.palette.query = "="

    def _act_layer_delete(self) -> None:
        """Delete the selected layer through the Scene panel's undoable command."""
        layer = self._selected_layer()
        panel = self.panels.get("scene")
        if layer is None or panel is None:
            return
        panel._delete_layer(layer)

    def _act_layer_duplicate(self) -> None:
        """Duplicate the selected layer."""
        layer = self._selected_layer()
        panel = self.panels.get("scene")
        if layer is None or panel is None:
            return
        panel._duplicate_layer(layer)

    def _act_layer_visibility(self) -> None:
        """Hide or show the selected layer.

        Goes through ``layerops.set_layer_style`` in the queue rather than assigning
        ``style.visible``: that helper handles the 3D ``gpu_dirty`` re-upload trap and the
        dirty-flag epilogue (CONTRACT §1.4).
        """
        layer = self._selected_layer()
        if layer is None:
            return
        plot = self.plot
        visible = not bool(getattr(getattr(layer, "style", None), "visible", True))
        self.queue.submit(lambda: layerops.set_layer_style(plot, layer, visible=visible))

    def _act_layer_solo(self) -> None:
        """Isolate the selected layer, or leave isolate if it is already the soloed one."""
        layer = self._selected_layer()
        panel = self.panels.get("scene")
        if layer is None or panel is None:
            return
        panel._toggle_solo(layer)
        self.open_panel("scene")

    def _act_layer_rename(self) -> None:
        """Start an inline rename on the selected layer's Scene row."""
        layer = self._selected_layer()
        panel = self.panels.get("scene")
        if layer is None or panel is None:
            return
        panel._begin_rename(layer)
        self.open_panel("scene")

    def _act_layer_move(self, delta: int) -> None:
        """Move the selected layer ``delta`` places in the list order."""
        layer = self._selected_layer()
        panel = self.panels.get("scene")
        if layer is None or panel is None:
            return
        try:
            layers = list(self.plot.scene.layers)
        except Exception:  # pragma: no cover - defensive
            return
        ids = [getattr(la, "layer_id", None) for la in layers]
        try:
            index = ids.index(layer.layer_id)
        except ValueError:  # pragma: no cover - _selected_layer just found it
            return
        target = index + delta
        if not 0 <= target < len(layers):
            self._status = "The layer is already at the end of the list."
            return
        panel._move_layer(layer.layer_id, ids[target])

    def _act_layer_step(self, delta: int) -> None:
        """Select the next/previous layer, wrapping."""
        try:
            layers = list(self.plot.scene.layers)
        except Exception:  # pragma: no cover - defensive
            return
        if not layers:
            return
        current = self._selected_layer_id()
        ids = [getattr(la, "layer_id", None) for la in layers]
        index = ids.index(current) if current in ids else -1
        self._select_layer(layers[(index + delta) % len(layers)])
        if "scene" in self.panels:
            self.open_panel("scene")

    # -- menu ------------------------------------------------------------------

    def draw_menu(self) -> None:
        """The main menu bar. The workspace owns it exclusively.

        Dear ImGui supports exactly one main menu bar per frame, so ``HudManager`` must not
        also call ``_draw_main_menu`` while a workspace exists — the HUD's View/Actions
        items are reproduced here instead, driving the same ``HudState``/``HudController``.

        Every item that has a registered action goes through :meth:`_menu_action` rather
        than calling the run directly, so the label, the shortcut hint and the enabled state
        all come from the registry and cannot drift from the keymap.
        """
        if not imgui.begin_main_menu_bar():
            return

        if imgui.begin_menu("File"):
            self._menu_action("data.new")
            self._menu_action("data.import_csv")
            self._menu_action("data.export_csv")
            imgui.separator()
            self._menu_action("global.export_png")
            self._menu_action("global.reset_layout")
            imgui.separator()
            self._menu_action("global.quit")
            imgui.end_menu()

        if imgui.begin_menu("Edit"):
            undo_label = self.undo.peek_undo()
            redo_label = self.undo.peek_redo()
            self._menu_action("global.undo", label=f"Undo {undo_label}" if undo_label else None)
            self._menu_action("global.redo", label=f"Redo {redo_label}" if redo_label else None)
            imgui.separator()
            self._menu_action("sel.select_all")
            self._menu_action("sel.select_none")
            self._menu_action("sel.invert")
            imgui.separator()
            self._menu_action("sel.create_dataset")
            self._menu_action("sel.delete")
            self._menu_action("sel.isolate")
            self._menu_action("sel.zoom")
            imgui.separator()
            self._menu_action("data.select_all")
            imgui.end_menu()

        if imgui.begin_menu("Panels"):
            for key, panel in self.panels.items():
                action = self.registry.get(f"panel.{key}")
                chord = keys.format(action.chord) if action and action.chord else None
                clicked, _ = imgui.menu_item(panel.title, chord, self.open.get(key, False))
                if clicked:
                    self.open[key] = not self.open.get(key, False)
                    if self.open[key]:
                        self._focus(key)
            imgui.separator()
            self._menu_action("panel.profiler")
            imgui.end_menu()

        state = self._hud_state()
        if state is not None and imgui.begin_menu("View"):
            self._menu_action("view.autoscale")
            self._menu_action("view.reset")
            imgui.separator()
            if imgui.begin_menu("Split into panels"):
                self._menu_action("view.split_grid")
                self._menu_action("view.split_cols")
                self._menu_action("view.split_rows")
                imgui.separator()
                self._menu_action("view.merge")
                imgui.end_menu()
            imgui.separator()
            self._menu_action("view.toggle_density")
            self._menu_action("view.toggle_grid")
            self._menu_action("view.toggle_labels")
            self._menu_action("view.toggle_frame")
            imgui.separator()
            _, state.show_status_overlay = imgui.menu_item(
                "Status Overlay", None, state.show_status_overlay
            )
            _, state.show_selection = imgui.menu_item("Selection Info", None, state.show_selection)
            _, state.show_analysis = imgui.menu_item("Analysis", None, state.show_analysis)
            changed, hint = imgui.menu_item("Getting Started Hint", None, self._show_hint)
            if changed:
                self._show_hint = hint
            imgui.end_menu()

        if imgui.begin_menu("Help"):
            self._menu_action("global.help")
            self._menu_action("global.palette")
            imgui.end_menu()

        self._draw_menu_hint()
        imgui.end_main_menu_bar()

    def _menu_action(self, action_id: str, *, label: Optional[str] = None) -> None:
        """Render a menu item for a registered action.

        The shortcut column, the enabled state and the check mark are read from the
        registry — the same source the keymap and Help read — so a menu entry cannot claim
        a binding the keymap does not have.
        """
        action = self.registry.get(action_id)
        if action is None:  # pragma: no cover - every id here is registered above
            return
        shortcut = keys.format(action.chord) if action.chord is not None else None
        checked = self.registry.checked(action)
        clicked, _ = imgui.menu_item(
            label or action.title,
            shortcut,
            bool(checked),
            self.registry.enabled(action),
        )
        if clicked:
            action.run()

    def _draw_menu_hint(self) -> None:
        """Right-align the status line and the palette hint in the menu bar.

        The hint is the discoverability budget of the whole feature: the palette is the one
        thing that leads to everything else, and a shortcut nobody knows about does not
        exist. It is a real button — clicking it opens the palette — because a user who
        does not yet trust the keyboard still needs a way in.
        """
        hint = f"{_chord_text('Mod+K')} to search"
        hint_w = imgui.calc_text_size(hint)[0] + 18.0
        if self._status:
            status = f"{self._status}   "
            width = imgui.calc_text_size(status)[0]
            if imgui.get_window_width() - imgui.get_cursor_pos_x() > width + hint_w:
                imgui.same_line(imgui.get_window_width() - width - hint_w)
                imgui.text_disabled(status)

        if imgui.get_window_width() - imgui.get_cursor_pos_x() > hint_w:
            imgui.same_line(imgui.get_window_width() - hint_w)
            if imgui.small_button(hint):
                self._act_palette()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Search commands, layers, datasets, functions and help")

    # -- rail ------------------------------------------------------------------

    def draw_toolbar(self) -> None:
        """The left icon rail: search, one latching toggle per panel, plus undo/redo.

        A window rather than a child region so it floats above the plot and its buttons
        take input priority over the camera.
        """
        state = self._hud_state()
        menu_h = imgui.get_frame_height()
        top = menu_h
        if state is not None and getattr(state, "show_status_overlay", False):
            top += _STATUS_STRIP_H

        viewport_h = float(imgui.get_io().display_size[1])
        rail_h = max(0.0, viewport_h - top)
        imgui.set_next_window_position(0.0, top, imgui.ALWAYS)
        imgui.set_next_window_size(RAIL_WIDTH, rail_h, imgui.ALWAYS)
        flags = (
            imgui.WINDOW_NO_TITLE_BAR
            | imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_SCROLLBAR
            | imgui.WINDOW_NO_SAVED_SETTINGS
            | imgui.WINDOW_NO_COLLAPSE
        )
        # Tight padding so the rail hugs its single icon column instead of inheriting the
        # roomy 10px window padding the floating panels use. RAIL_WIDTH is derived from this
        # same RAIL_PAD, so the icons stay centred at any size.
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (RAIL_PAD, RAIL_PAD))
        # The global theme sets window_min_size to (200, 100) for the roomy floating panels.
        # imgui clamps *every* window up to that minimum, so set_next_window_size above is
        # overruled and the rail balloons to 200px wide. Override the minimum for this window
        # only (pushed before begin, read inside begin) so the rail can be its true narrow width.
        imgui.push_style_var(imgui.STYLE_WINDOW_MIN_SIZE, (RAIL_WIDTH, 0.0))
        imgui.begin("##glplot_rail", flags=flags)

        # The palette leads the rail: it is the front door to every other action, and it is
        # the one control whose tooltip teaches a shortcut the user did not know existed.
        if icons.icon_button(
            "##rail_palette",
            "search",
            size=RAIL_ICON_SIZE,
            tooltip=f"Search everything ({_chord_text('Mod+K')})",
            active=self.palette is not None and getattr(self.palette, "visible", False),
            enabled=self.palette is not None,
        ):
            self._act_palette()
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        for key, panel in self.panels.items():
            is_open = self.open.get(key, False)
            if icons.icon_button(
                f"##rail_{key}",
                getattr(panel, "icon", "layers"),
                size=RAIL_ICON_SIZE,
                tooltip=self._panel_tooltip(key, panel.title),
                active=is_open,
            ):
                self.open[key] = not is_open
                if self.open[key]:
                    self._focus(key)
            imgui.spacing()

        imgui.separator()
        imgui.spacing()

        if icons.icon_button(
            "##rail_undo",
            "undo",
            size=RAIL_ICON_SIZE,
            tooltip=self._shortcut_tooltip("Undo", self.undo.peek_undo(), "global.undo"),
            enabled=self.undo.can_undo(),
        ):
            self.undo_action()
        imgui.spacing()
        if icons.icon_button(
            "##rail_redo",
            "redo",
            size=RAIL_ICON_SIZE,
            tooltip=self._shortcut_tooltip("Redo", self.undo.peek_redo(), "global.redo"),
            enabled=self.undo.can_redo(),
        ):
            self.redo_action()

        imgui.end()
        imgui.pop_style_var(2)

    def _panel_tooltip(self, key: str, title: str) -> str:
        """ "Data (⌘1)" — the rail is where most users will meet the panel chords."""
        action = self.registry.get(f"panel.{key}")
        if action is None or action.chord is None:
            return title
        return f"{title} ({keys.format(action.chord)})"

    def _shortcut_tooltip(self, verb: str, label: Optional[str], action_id: str) -> str:
        """Build "Undo Edit cell (⌘Z)" — the label is what makes it worth reading.

        The chord comes from the registry rather than a literal: the rail cannot advertise
        a binding the keymap does not actually have.
        """
        action = self.registry.get(action_id)
        chord = keys.format(action.chord) if action is not None and action.chord else ""
        head = f"{verb} {label}" if label else verb
        return f"{head} ({chord})" if chord else head

    # -- panels ----------------------------------------------------------------

    def _draw_panels(self) -> None:
        """Draw every open panel in its own window. The workspace owns begin/end."""
        focused: Optional[str] = None
        for key, panel in self.panels.items():
            if not self.open.get(key, False):
                continue
            self._place_window(key)
            # imgui.begin must always be paired with end, expanded or not — unlike
            # begin_child, whose end is also unconditional but for a different reason.
            expanded, opened = imgui.begin(panel.title, True)
            if not opened:
                self.open[key] = False
            # Read inside the window, where "is this window focused" is answerable at all.
            # Child windows count: the data grid is a child, and Mod+W from inside it must
            # still mean "close Data".
            if imgui.is_window_focused(_FOCUSED_ROOT_AND_CHILD_WINDOWS):
                focused = key
            if expanded:
                try:
                    panel.draw()
                except Exception:
                    # A panel that raises mid-body has already pushed imgui state we cannot
                    # know about, but the frame survives because end() below still runs and
                    # imgui tolerates an unbalanced stack within a window. Losing one panel
                    # beats losing the render loop.
                    logger.exception("Panel %r raised while drawing.", key)
                    imgui.text_colored("This panel failed to draw; see the log.", 0.9, 0.35, 0.35)
            imgui.end()
        self._focused_panel = focused

    def _place_window(self, key: str) -> None:
        """Apply the first-run (or forced) geometry for panel ``key``."""
        layout = _LAYOUT.get(key)
        if layout is None:
            return
        io = imgui.get_io()
        vw = float(io.display_size[0])
        vh = float(io.display_size[1])
        if vw <= 0.0 or vh <= 0.0:  # pragma: no cover - degenerate/minimized frame
            return

        menu_h = imgui.get_frame_height()
        state = self._hud_state()
        top = menu_h + (_STATUS_STRIP_H if getattr(state, "show_status_overlay", False) else 0.0)
        left = RAIL_WIDTH + 4.0
        usable_w = max(120.0, vw - left - 4.0)
        usable_h = max(120.0, vh - top - 4.0)

        fx, fy, fw, fh = layout
        condition = imgui.ALWAYS if self._force_layout > 0 else imgui.FIRST_USE_EVER
        imgui.set_next_window_position(left + fx * usable_w, top + fy * usable_h, condition)
        imgui.set_next_window_size(fw * usable_w - 6.0, fh * usable_h - 6.0, condition)

    def _draw_empty_hint(self) -> None:
        """The standalone app's first-run affordance: say what to do with an empty scene.

        Shown only while there is genuinely nothing to look at — no layers *and* no data —
        so it never nags a user who is working. Dismissable from the View menu.
        """
        if not self._show_hint:
            return
        if len(self.store) > 0 or len(self.plot.scene.layers) > 0:
            return

        io = imgui.get_io()
        vw = float(io.display_size[0])
        vh = float(io.display_size[1])
        imgui.set_next_window_position(vw * 0.5, vh * 0.62, imgui.ALWAYS, 0.5, 0.5)
        flags = (
            imgui.WINDOW_NO_TITLE_BAR
            | imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_SAVED_SETTINGS
            | imgui.WINDOW_ALWAYS_AUTO_RESIZE
            | imgui.WINDOW_NO_COLLAPSE
        )
        imgui.begin("##glplot_hint", flags=flags)
        imgui.text("Nothing plotted yet.")
        imgui.spacing()
        imgui.text_disabled("Paste a table with Ctrl+V in the Data panel,")
        imgui.text_disabled("or write a function like  a*sin(b*x)  in Functions.")
        imgui.spacing()
        # The palette line sits with the other two first-run affordances, not below them:
        # this window is the only thing an empty workstation shows, so it is the one place
        # every user is guaranteed to read.
        imgui.text_disabled(
            f"Press {_chord_text('Mod+K')} to search every command, "
            f"or {_chord_text('F1')} for help."
        )
        imgui.spacing()
        if imgui.button("Open Data"):
            self.open_panel("data")
        imgui.same_line()
        if imgui.button("Open Functions"):
            self.open_panel("functions")
        imgui.same_line()
        if imgui.button(f"Search  {_chord_text('Mod+K')}"):
            self._act_palette()
        imgui.same_line()
        if imgui.button("Help"):
            self._act_help("start")
        imgui.same_line()
        if imgui.button("Dismiss"):
            self._show_hint = False
        imgui.end()

    def _focus(self, key: str) -> None:
        """Raise panel ``key`` to the front on the next frame, if it exists."""
        panel = self.panels.get(key)
        if panel is None or not IMGUI_AVAILABLE:
            return
        try:
            imgui.set_window_focus_labeled(panel.title)
        except Exception:  # pragma: no cover - no current imgui context (headless test)
            logger.debug("Could not focus panel %r outside a frame.", key)

    # -- undo/redo -------------------------------------------------------------

    def undo_action(self) -> None:
        """Queue an undo. Deferred because undo callbacks rewrite live layer arrays."""

        def run() -> None:
            label = self.undo.undo()
            self._status = f"Undo: {label}" if label else "Nothing to undo"

        self.queue.submit(run)

    def redo_action(self) -> None:
        """Queue a redo. Deferred for the same reason as :meth:`undo_action`."""

        def run() -> None:
            label = self.undo.redo()
            self._status = f"Redo: {label}" if label else "Nothing to redo"

        self.queue.submit(run)

    # -- scene sync ------------------------------------------------------------

    def sync_datasets_from_scene(self) -> None:
        """Reconcile the DataStore with layers created from code.

        Layers plotted before ``show()`` (or from a script while the window is up) have no
        dataset until this runs, and a user who deletes a layer must not be left with a
        dataset that writes back into a freed one. ``DataStore.from_layer`` is idempotent,
        so re-registering an already-tracked layer is a no-op.
        """
        try:
            layers = list(self.plot.scene.layers)
        except Exception:  # pragma: no cover - defensive
            return

        live_ids = set()
        for layer in layers:
            layer_id = getattr(layer, "layer_id", None)
            if layer_id is None:
                continue
            live_ids.add(layer_id)
            try:
                self.store.from_layer(layer)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Could not build a dataset for layer %r.", layer_id)

        # Unbind, never remove: the user may have edited the table, and silently deleting
        # their work because the layer went away would be worse than an orphan table.
        for dataset in list(self.store.datasets):
            if dataset.layer_id is not None and dataset.layer_id not in live_ids:
                dataset.unbind()

    def __repr__(self) -> str:
        return (
            f"<Workspace panels={list(self.panels)} "
            f"actions={len(self.registry.all())} queued={len(self.queue)}>"
        )


#: The Math actions, as ``(id, mathlab tab key, title, chord, icon, keywords, description)``.
#: A table rather than nine near-identical ``add`` calls: they differ only in their data, and
#: the Math Lab tab key is the one field that must match ``panels/mathlab.py::_TABS``.
_MATH_OPS: Tuple[Tuple[str, str, str, Optional[str], str, Tuple[str, ...], str], ...] = (
    (
        "math.integrate",
        "integral",
        "Integrate",
        "Mod+I",
        "integral",
        ("integral", "cumulative", "area", "trapezoid", "antiderivative"),
        "Open Math Lab on the cumulative integral of the selected layer.",
    ),
    (
        "math.derivative",
        "derivative",
        "Derivative",
        "Mod+D",
        "derivative",
        ("differentiate", "gradient", "slope", "rate"),
        "Open Math Lab on the derivative of the selected layer.",
    ),
    (
        "math.smooth",
        "smooth",
        "Smooth",
        "Mod+Shift+S",
        "smooth",
        ("filter", "savitzky", "golay", "gaussian", "denoise", "moving average"),
        "Open Math Lab on smoothing: moving average, Savitzky-Golay or Gaussian.",
    ),
    (
        "math.noise",
        "noise",
        "Add Noise",
        None,
        "noise",
        ("random", "gaussian", "jitter", "perturb"),
        "Open Math Lab on adding noise to the selected layer.",
    ),
    (
        "math.normalize",
        "normalize",
        "Normalize",
        None,
        "sliders",
        ("scale", "minmax", "zscore", "standardize", "rescale"),
        "Open Math Lab on normalising the selected layer.",
    ),
    (
        "math.fft",
        "fft",
        "FFT",
        None,
        "wave",
        ("fourier", "spectrum", "frequency", "transform"),
        "Open Math Lab on the frequency spectrum of the selected layer.",
    ),
    (
        "math.fit",
        "fit",
        "Fit Polynomial",
        None,
        "chart_line",
        ("regression", "polyfit", "trend", "least squares", "curve fit"),
        "Open Math Lab on a least-squares polynomial fit of the selected layer.",
    ),
    (
        "math.stats",
        "stats",
        "Statistics",
        None,
        "info",
        ("mean", "median", "std", "describe", "summary"),
        "Open Math Lab on the summary statistics of the selected layer.",
    ),
    (
        "math.resample",
        "resample",
        "Resample",
        None,
        "refresh",
        ("interpolate", "grid", "decimate", "upsample"),
        "Open Math Lab on resampling the selected layer onto a new x grid.",
    ),
)
