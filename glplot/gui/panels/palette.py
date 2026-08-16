"""The command palette -- "smart search", the keyboard front door to the workstation.

Opens on ``Mod+K`` or ``Mod+Shift+P`` and searches *everything*: registered commands,
the layers in the scene, the datasets in the store, the function presets, the help
topics -- and, uniquely, a **live expression**: typing ``=sin(x)*exp(-x/5)`` and pressing
Enter plots that curve without ever visiting the Functions panel.

**This is not a rail panel.** It subclasses :class:`~glplot.gui.panels.base.Panel` purely
for the ``plot``/``store``/``queue``/``submit`` proxies; it owns its own window because it
is a centred modal overlay, not a dockable body. The workspace must therefore call
:meth:`CommandPalette.draw` directly and must **not** put it in ``ws.panels`` -- doing so
would wrap it in a second ``imgui.begin`` and give it a title bar and a rail icon.

**Providers.** Each search source implements :class:`SearchProvider` (a ``prefix`` and a
``results(query)``). With no prefix the palette blends and re-ranks every provider; with a
prefix it consults exactly one. The table:

======  =========================  ==============================================
Prefix  Provider                   A hit does
======  =========================  ==============================================
*none*  everything, blended        (whatever the underlying hit does)
``>``   commands                   runs the action
``@``   layers                     selects it and focuses the Scene panel
``#``   datasets                   opens it in the Data editor
``=``   presets + live expression  plots the curve
``?``   help topics                opens Help at that topic
======  =========================  ==============================================

Three details separate a palette that feels professional from a styled listbox, and each
one is load-bearing here:

* **Focus on the opening frame only.** ``set_keyboard_focus_here`` is called before the
  input and *only* on the frame the palette opens (:attr:`CommandPalette._focus_input`).
  Calling it every frame re-grabs focus after every click and makes the field impossible
  to leave.
* **Matched characters are highlighted.** :mod:`glplot.gui.fuzzy` returns the matched
  indices precisely so the title can be drawn in segments with the accent on the matches.
  Without it the ranking looks arbitrary; with it the search visibly understands the
  query.
* **The selection stays on screen.** Arrowing past the fold scrolls the list by the
  minimum amount needed (:meth:`CommandPalette._apply_scroll`), which is why the rows are
  drawn by hand at known offsets rather than left to imgui's layout.

Disabled hits are listed greyed, never hidden (the reasoning is in
:meth:`glplot.gui.actions.ActionRegistry.search`). Every hit that touches the scene goes
through the deferred queue, per CONTRACT section 1.1 -- in practice by delegating to an
``Action.run`` or to a ``FunctionsPanel`` method that already does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from .. import fuzzy, icons, keys, theme
from ..keys import Chord
from .base import Panel

try:
    import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - imgui is a hard dependency in CI
    IMGUI_AVAILABLE = False
    imgui = None

logger = logging.getLogger(__name__)

__all__ = [
    "IMGUI_AVAILABLE",
    "CommandPalette",
    "CommandProvider",
    "DatasetProvider",
    "FunctionProvider",
    "HelpProvider",
    "LayerProvider",
    "SearchHit",
    "SearchProvider",
]


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------

#: Overlay width in points, clamped to the viewport on small displays.
PALETTE_WIDTH = 640.0

#: Height of one result row. Two lines of text (title + subtitle) plus breathing room.
ROW_HEIGHT = 42.0

#: Height of a category header row, used only for the un-typed listing (see
#: :meth:`CommandPalette._build_rows`).
HEADER_HEIGHT = 22.0

#: Rows visible before the list scrolls. Fractional on purpose: a half-row peeking under
#: the fold is what tells the user there is more to see.
VISIBLE_ROWS = 8.5

#: Hard cap on rendered hits. Nobody scrolls past 60, and every row costs a fuzzy score.
MAX_HITS = 60

#: Most-recently-used action ids kept in memory (never persisted).
MRU_SIZE = 8

_ICON_SLOT = 30.0  # horizontal room reserved for the leading icon
_PAD_X = 12.0  # inner left/right padding of a row
_CHIP_PAD = 5.0  # horizontal padding inside a category chip

#: Sources and what they are prefixed with. Rendered as the hint line under an empty
#: input -- a prefix nobody knows about is a prefix that does not exist.
_PREFIX_HINTS: Tuple[Tuple[str, str], ...] = (
    (">", "commands"),
    ("@", "layers"),
    ("#", "datasets"),
    ("=", "functions"),
    ("?", "help"),
)

#: ``layer_type`` -> icon name, mirroring ``panels/scene.py``. Duplicated rather than
#: imported: the palette must not depend on a sibling panel's private table, and a new
#: layer type degrades to the default icon in both places independently.
_LAYER_ICONS: Dict[str, str] = {
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
_LAYER_ICON_DEFAULT = "layers"

# Parsed once at import: `parse` raises on a bad spec, and a chord table that is wrong is
# better discovered at import than from inside a draw callback at 60Hz. "Mod" matches
# Ctrl *or* Cmd (see the keys module), so one binding covers Ctrl+P on every platform.
_CHORD_ESCAPE: Chord = keys.parse("Escape")
_CHORD_UP: Chord = keys.parse("Up")
_CHORD_DOWN: Chord = keys.parse("Down")
_CHORD_CTRL_P: Chord = keys.parse("Mod+P")
_CHORD_CTRL_N: Chord = keys.parse("Mod+N")
_CHORD_ENTER: Chord = keys.parse("Enter")

#: Help tabs, per SPEC section 5. Used when ``panels/help.py`` exposes no topic table of
#: its own -- see :meth:`HelpProvider._topics`.
_FALLBACK_HELP_TOPICS: Tuple[Tuple[str, str, str], ...] = (
    ("getting_started", "Getting Started", "A five-step tour of the workstation"),
    ("keyboard", "Keyboard", "Every shortcut, generated from the action registry"),
    ("expressions", "Expressions", "Every function and constant you can write in f(x)"),
    ("panels", "Panels", "What each panel is for and its local keys"),
    ("tips", "Tips", "The non-obvious wins"),
)


# --------------------------------------------------------------------------------------
# The provider interface
# --------------------------------------------------------------------------------------


@dataclass
class SearchHit:
    """One row in the results list.

    ``matches`` are indices into ``title`` (from :func:`glplot.gui.fuzzy.score`) and are
    drawn in the accent colour. ``score`` is what the blended, un-prefixed search ranks
    on, so providers must produce it with ``fuzzy.score`` rather than inventing a scale.

    ``mru_id`` is a stable id -- an ``Action.id`` -- recorded in the palette's recent list
    when the hit runs. Transient hits (a layer, a live expression) leave it ``None``:
    there is nothing stable to remember.
    """

    title: str
    subtitle: str
    category: str
    icon: Optional[str]
    chord_text: str
    run: Callable[[], None]
    score: int
    matches: List[int] = field(default_factory=list)
    enabled: bool = True
    mru_id: Optional[str] = None


class SearchProvider(Protocol):  # pragma: no cover - a typing surface, not runtime code
    """A source of :class:`SearchHit` rows.

    ``prefix`` is the single character that selects this provider exclusively, or ``None``
    for a provider that only ever participates in the blended search.

    A provider may additionally define ``prefixed_results(query)``. The palette calls it
    instead of :meth:`results` when the provider's own prefix is active, which is how
    :class:`FunctionProvider` offers to plot a live expression under ``=`` without
    offering it in the blended search -- typing ``sin`` should find the *Sine preset*, not
    volunteer to plot a curve the user never asked for.
    """

    prefix: Optional[str]

    def results(self, query: str) -> List[SearchHit]:
        """Rank this source against ``query``. An empty query means "list everything"."""
        ...


def _panel_of(ws: Any, key: str) -> Optional[Any]:
    """The workspace's panel under ``key``, or None if it failed to import."""
    panels = getattr(ws, "panels", None)
    if not isinstance(panels, dict):
        return None
    return panels.get(key)


def _open_and_focus(ws: Any, key: str) -> None:
    """Open panel ``key`` and raise it to the front on the next frame.

    Focus goes through ``set_window_focus_labeled`` on the panel's public ``title`` rather
    than ``Workspace._focus``: the palette must not reach into a private method of the
    object that owns it, and the title is the panel contract's documented window key.
    """
    panel = _panel_of(ws, key)
    if panel is None:
        return
    open_map = getattr(ws, "open", None)
    if isinstance(open_map, dict):
        open_map[key] = True
    if IMGUI_AVAILABLE:
        try:
            imgui.set_window_focus_labeled(panel.title)
        except Exception:  # pragma: no cover - defensive: never break a draw callback
            logger.exception("Could not focus panel %r", key)


def _best(query: str, *texts: str) -> Optional[Tuple[int, List[int]]]:
    """Score ``query`` against several fields, returning the best ``(score, indices)``.

    The indices returned are always the **first** field's -- the title -- so a hit found
    by its subtitle still highlights the right characters, or none at all. Highlighting
    positions from a string the row never draws is worse than no highlight.
    """
    title_hit = fuzzy.score(query, texts[0]) if texts else None
    best_score = title_hit[0] if title_hit is not None else -1
    for text in texts[1:]:
        other = fuzzy.score(query, text)
        if other is not None and other[0] > best_score:
            best_score = other[0]
    if best_score < 0:
        return None
    return (best_score, list(title_hit[1]) if title_hit is not None else [])


# --------------------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------------------


class CommandProvider:
    """Every registered :class:`~glplot.gui.actions.Action`, ranked by the registry.

    Thin by design: :meth:`glplot.gui.actions.ActionRegistry.search` already applies the
    field weights, ranks disabled actions last and returns title indices. Re-ranking here
    would be a second, divergent opinion about the same question.
    """

    prefix: Optional[str] = ">"

    def __init__(self, ws: Any) -> None:
        self._ws = ws

    def _registry(self) -> Optional[Any]:
        """The workspace's registry, or None.

        ``getattr`` rather than an attribute access: the palette is constructible before
        ``Workspace._register_actions`` has run, and a half-built workspace must degrade
        to an empty command list rather than raise inside a draw callback.
        """
        return getattr(self._ws, "registry", None)

    def results(self, query: str) -> List[SearchHit]:
        """Rank the registry against ``query``."""
        registry = self._registry()
        if registry is None:
            return []
        hits: List[SearchHit] = []
        for action, score, matches in registry.search(query):
            chord = getattr(action, "chord", None)
            hits.append(
                SearchHit(
                    title=action.title,
                    subtitle=action.description,
                    category=action.category,
                    icon=action.icon,
                    chord_text=keys.format(chord) if isinstance(chord, Chord) else "",
                    run=action.run,
                    score=score,
                    matches=matches,
                    enabled=registry.enabled(action),
                    mru_id=action.id,
                )
            )
        return hits


class LayerProvider:
    """The layers in the live scene. A hit selects one and focuses the Scene panel."""

    prefix: Optional[str] = "@"

    def __init__(self, ws: Any) -> None:
        self._ws = ws

    def results(self, query: str) -> List[SearchHit]:
        """Rank the scene's layers by label and type."""
        try:
            layers = list(self._ws.plot.scene.layers)
        except Exception:  # pragma: no cover - defensive: a plot without a scene
            return []

        hits: List[SearchHit] = []
        for layer in layers:
            label = _layer_label(layer)
            layer_type = str(getattr(layer, "layer_type", "") or "layer")
            hit = _best(query, label, layer_type)
            if hit is None:
                continue
            score, matches = hit
            visible = bool(getattr(getattr(layer, "style", None), "visible", True))
            subtitle = f"{layer_type} - {_layer_size_text(layer)}".rstrip(" -")
            if not visible:
                subtitle += " - hidden"
            hits.append(
                SearchHit(
                    title=label,
                    subtitle=subtitle,
                    category="Layer",
                    icon=_LAYER_ICONS.get(layer_type, _LAYER_ICON_DEFAULT),
                    chord_text="",
                    run=self._make_run(layer),
                    score=score,
                    matches=matches,
                )
            )
        return hits

    def _make_run(self, layer: Any) -> Callable[[], None]:
        """Select ``layer`` and show it in the Scene panel.

        Selection is not scene state -- it is a HUD/interaction id, read by
        ``renderer_manager`` only for the selection outline -- so it needs no queue. The
        write mirrors ``ScenePanel._select``: the HUD's copy wins where a HUD exists,
        because ``hud.py:122-136`` propagates it to ``plot.interaction`` on the next
        frame and writing both would make that block read it as an engine-side pick.
        """

        def run() -> None:
            layer_id = getattr(layer, "layer_id", None)
            if layer_id is None:
                return
            state = getattr(getattr(self._ws, "hud", None), "state", None)
            if state is not None:
                state.selected_layer_id = layer_id
            else:
                interaction = getattr(self._ws.plot, "interaction", None)
                if interaction is not None:
                    interaction.selected_layer_id = layer_id
            _open_and_focus(self._ws, "scene")

        return run


class DatasetProvider:
    """The datasets in the store. A hit opens one in the Data editor."""

    prefix: Optional[str] = "#"

    def __init__(self, ws: Any) -> None:
        self._ws = ws

    def results(self, query: str) -> List[SearchHit]:
        """Rank the store's datasets by name and column names."""
        try:
            datasets = list(self._ws.store.datasets)
        except Exception:  # pragma: no cover - defensive
            return []

        hits: List[SearchHit] = []
        for dataset in datasets:
            name = str(getattr(dataset, "name", "") or "dataset")
            # DataSet.column_names and .n_rows are *methods*, not properties -- the one
            # place this module's shape assumptions had to bend to the existing code.
            columns = [str(c) for c in dataset.column_names()]
            hit = _best(query, name, " ".join(columns))
            if hit is None:
                continue
            score, matches = hit
            rows = int(dataset.n_rows())
            subtitle = f"{rows:,} rows x {len(columns)} cols"
            if columns:
                subtitle += ": " + ", ".join(columns[:4])
                if len(columns) > 4:
                    subtitle += ", ..."
            hits.append(
                SearchHit(
                    title=name,
                    subtitle=subtitle,
                    category="Data",
                    icon="table",
                    chord_text="",
                    run=self._make_run(dataset),
                    score=score,
                    matches=matches,
                )
            )
        return hits

    def _make_run(self, dataset: Any) -> Callable[[], None]:
        """Make ``dataset`` the Data editor's current sheet and focus it.

        Goes through the panel's ``_select`` where it exists: that resets the cursor,
        the selection rectangle and the scroll, which is exactly what "open this dataset"
        should mean. Assigning ``_current`` alone would land the user mid-way down the
        previous dataset with a selection pointing at rows this one may not have.
        """

        def run() -> None:
            panel = _panel_of(self._ws, "data")
            if panel is not None:
                select = getattr(panel, "_select", None)
                if callable(select):
                    select(dataset)
                else:  # pragma: no cover - only if the editor drops _select
                    panel._current = dataset
            _open_and_focus(self._ws, "data")

        return run


class FunctionProvider:
    """Function presets, plus -- under ``=`` only -- a live expression.

    The live expression is the palette's flagship: ``=sin(x)*exp(-x/5)`` then Enter plots
    that curve. It is offered **only** when the ``=`` prefix is active
    (:meth:`prefixed_results`), never in the blended search, because a bare ``sin`` should
    surface the Sine preset rather than volunteer to plot something the user was merely
    searching for.

    Both paths delegate to :class:`~glplot.gui.panels.functions.FunctionsPanel`, which
    already validates, evaluates, derives the parameter sliders and pushes an undoable
    command onto the queue. Reimplementing any of that here would be a second, drifting
    copy of the expression pipeline -- and a CONTRACT section 1.1 violation waiting to
    happen the first time it was edited without the queue in mind.
    """

    prefix: Optional[str] = "="

    def __init__(self, ws: Any) -> None:
        self._ws = ws

    def results(self, query: str) -> List[SearchHit]:
        """Rank the function presets. No live expression -- see the class docstring."""
        presets = self._presets()
        hits: List[SearchHit] = []
        for preset in presets:
            hit = _best(query, preset.label, preset.expr)
            if hit is None:
                continue
            score, matches = hit
            hits.append(
                SearchHit(
                    title=preset.label,
                    subtitle=f"Plot  {preset.expr}",
                    category="Function",
                    icon=getattr(preset, "icon", "function"),
                    chord_text="",
                    run=self._make_preset_run(preset),
                    score=score,
                    matches=matches,
                    enabled=_panel_of(self._ws, "functions") is not None,
                )
            )
        return hits

    def prefixed_results(self, query: str) -> List[SearchHit]:
        """Presets, headed by a live-expression hit once ``query`` is non-empty."""
        hits = self.results(query)
        live = self._live_hit(query)
        if live is not None:
            hits.insert(0, live)
        return hits

    @staticmethod
    def _presets() -> Sequence[Any]:
        """The ``FunctionsPanel`` preset table.

        Imported lazily: ``panels/functions.py`` pulls in numpy and the expression
        machinery, and the palette must stay constructible even if that import breaks.
        """
        try:
            from .functions import PRESETS

            return PRESETS
        except Exception:  # pragma: no cover - defensive
            logger.exception("Could not load the function presets")
            return ()

    def _live_hit(self, query: str) -> Optional[SearchHit]:
        """A "plot this expression" hit, greyed with the parse error when invalid.

        Validation runs per keystroke. It is an ``ast.parse`` of a short string plus a
        node walk -- microseconds -- and it buys the thing that makes the feature usable:
        the row says *why* an expression will not plot while you are still typing it,
        instead of accepting Enter and failing silently somewhere else.
        """
        expr = query.strip()
        if not expr:
            return None

        try:
            from .. import expressions
        except Exception:  # pragma: no cover - defensive
            return None

        panel = _panel_of(self._ws, "functions")
        enabled = panel is not None
        try:
            free = expressions.free_variables(expr)
            subtitle = "Plot this expression in Functions"
            if free:
                names = ", ".join(free)
                subtitle = f"Plot it; {names} become parameter sliders"
        except expressions.ExpressionError as exc:
            enabled = False
            subtitle = str(exc)
        except Exception as exc:  # pragma: no cover - defensive
            enabled = False
            subtitle = str(exc)

        if panel is None:
            subtitle = "The Functions panel is unavailable"

        # A large, fixed score: with the `=` prefix active the user has already declared
        # this is what they want, so it heads the list regardless of how a preset ranks.
        return SearchHit(
            title=f"f(x) = {expr}",
            subtitle=subtitle,
            category="Function",
            icon="function",
            chord_text=keys.format(_CHORD_ENTER),
            run=self._make_live_run(expr),
            score=1_000_000,
            matches=[],
            enabled=enabled,
        )

    def _make_preset_run(self, preset: Any) -> Callable[[], None]:
        """Load ``preset`` into the Functions panel, plot it, and show the panel."""

        def run() -> None:
            panel = _panel_of(self._ws, "functions")
            if panel is None:
                return
            panel._apply_preset(preset)
            # _apply_preset resets the sync gate; _sync_params then re-derives the slider
            # list, which _action_plot's bindings read. Without this the plot would be
            # evaluated against the *previous* expression's parameters.
            panel._sync_params()
            panel._action_plot()
            _open_and_focus(self._ws, "functions")

        return run

    def _make_live_run(self, expr: str) -> Callable[[], None]:
        """Plot ``expr`` through the Functions panel, then show it there.

        The panel keeps the expression and its freshly-derived sliders, so "plot from the
        palette" lands the user somewhere they can immediately tune what they just made --
        rather than in front of a curve with no visible provenance.
        """

        def run() -> None:
            panel = _panel_of(self._ws, "functions")
            if panel is None:
                return
            panel.expr = expr
            panel._sync_params()
            panel._action_plot()
            _open_and_focus(self._ws, "functions")

        return run


class HelpProvider:
    """Help topics. A hit opens the Help panel at that topic."""

    prefix: Optional[str] = "?"

    def __init__(self, ws: Any) -> None:
        self._ws = ws

    def results(self, query: str) -> List[SearchHit]:
        """Rank the help topics by title and summary."""
        hits: List[SearchHit] = []
        for topic_id, title, summary in self._topics():
            hit = _best(query, title, summary)
            if hit is None:
                continue
            score, matches = hit
            hits.append(
                SearchHit(
                    title=title,
                    subtitle=summary,
                    category="Help",
                    icon="info",
                    chord_text="",
                    run=self._make_run(topic_id),
                    score=score,
                    matches=matches,
                    enabled=_panel_of(self._ws, "help") is not None,
                )
            )
        return hits

    @staticmethod
    def _topics() -> Sequence[Tuple[str, str, str]]:
        """``(id, title, summary)`` per topic, from ``panels/help.py`` where it offers one.

        Duck-typed against ``help.TOPICS`` / ``help.HELP_TOPICS`` so the Help panel stays
        the source of truth for its own contents, with SPEC section 5's five tabs as the
        fallback. Help is authored concurrently with this module: a hard import of a
        specific symbol would make the palette's ``?`` provider hostage to a name that has
        not settled, and an empty ``?`` reads as a broken palette.
        """
        try:
            from . import help as help_panel
        except Exception:
            return _FALLBACK_HELP_TOPICS

        table = getattr(help_panel, "TOPICS", None)
        if table is None:
            table = getattr(help_panel, "HELP_TOPICS", None)
        if not table:
            return _FALLBACK_HELP_TOPICS

        topics: List[Tuple[str, str, str]] = []
        for entry in table:
            try:
                if isinstance(entry, (tuple, list)):
                    parts = list(entry) + ["", ""]
                    topics.append((str(parts[0]), str(parts[1] or parts[0]), str(parts[2])))
                else:
                    topic_id = str(getattr(entry, "id", "") or getattr(entry, "key", ""))
                    title = str(getattr(entry, "title", "") or topic_id)
                    summary = str(
                        getattr(entry, "summary", "") or getattr(entry, "description", "")
                    )
                    if topic_id:
                        topics.append((topic_id, title, summary))
            except Exception:  # pragma: no cover - defensive
                continue
        return topics or _FALLBACK_HELP_TOPICS

    def _make_run(self, topic_id: str) -> Callable[[], None]:
        """Open Help and, if it supports it, jump to ``topic_id``.

        The topic jump is duck-typed for the same reason :meth:`_topics` is. If Help
        exposes no way in, the panel still opens -- degrading to "the right panel, wrong
        tab" rather than to nothing happening at all.
        """

        def run() -> None:
            panel = _panel_of(self._ws, "help")
            if panel is not None:
                for name in ("open_topic", "show_topic", "select_topic"):
                    method = getattr(panel, name, None)
                    if callable(method):
                        try:
                            method(topic_id)
                        except Exception:  # pragma: no cover - defensive
                            logger.exception("Help.%s(%r) failed", name, topic_id)
                        break
                else:
                    if hasattr(panel, "active_topic"):
                        panel.active_topic = topic_id
            _open_and_focus(self._ws, "help")

        return run


# --------------------------------------------------------------------------------------
# Row model
# --------------------------------------------------------------------------------------


@dataclass
class _Row:
    """One rendered line: either a category header or a hit. Headers are not selectable."""

    height: float
    top: float
    hit: Optional[SearchHit] = None
    header: str = ""


def _layer_label(layer: Any) -> str:
    """The layer's display name, with imgui's ``##`` id separator defanged.

    A layer literally labelled ``"a##b"`` would otherwise render as ``"a"`` and silently
    lose half its name. The layer's real ``label`` is never touched.
    """
    name = str(getattr(layer, "label", "") or getattr(layer, "layer_type", "") or "layer")
    return name.replace("##", "# #") if "##" in name else name


def _layer_size_text(layer: Any) -> str:
    """Human size readout for a layer, matching the Scene panel's wording."""
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


# --------------------------------------------------------------------------------------
# The palette
# --------------------------------------------------------------------------------------


class CommandPalette(Panel):
    """The centred search overlay. Owns its own window; see the module docstring.

    The workspace drives it with three calls::

        ws.palette = CommandPalette(ws)
        ...
        ws.palette.toggle()   # from the Mod+K / Mod+Shift+P actions
        ws.palette.draw()     # once per frame, after the panels
    """

    title = "Command Palette"
    icon = "search"
    default_open = False

    def __init__(self, ws: Any) -> None:
        super().__init__(ws)
        self.visible = False
        self.query = ""
        self.selected = 0

        self.providers: List[Any] = [
            CommandProvider(ws),
            LayerProvider(ws),
            DatasetProvider(ws),
            FunctionProvider(ws),
            HelpProvider(ws),
        ]

        #: Action ids, most recent first. In-memory only, per SPEC section 4.
        self.recent: List[str] = []

        #: True for exactly one frame after :meth:`open` -- the focus grab. Every frame
        #: would re-steal focus after any click and make the field impossible to leave.
        self._focus_input = False
        #: Set when the keyboard moves the selection; consumed by :meth:`_apply_scroll`.
        self._scroll_to_selected = False
        self._rows: List[_Row] = []

    # -- lifecycle ---------------------------------------------------------------------

    def open(self) -> None:
        """Show the palette, empty and focused.

        The query is always reset. A palette that reopens on the last search is a palette
        you must clear before you can use it, which costs more than the rare case where
        you wanted the same query twice.
        """
        self.visible = True
        self.query = ""
        self.selected = 0
        self._focus_input = True
        self._scroll_to_selected = True

    def close(self) -> None:
        """Hide the palette."""
        self.visible = False
        self._focus_input = False

    def toggle(self) -> None:
        """Open the palette, or close it if it is already open. Bind Mod+K here."""
        if self.visible:
            self.close()
        else:
            self.open()

    # -- search ------------------------------------------------------------------------

    def _split_query(self) -> Tuple[Optional[Any], str]:
        """Resolve the typed text to ``(provider, query)``.

        A leading prefix character selects one provider exclusively and is stripped from
        the query; anything else blends every provider. ``"> "`` with a trailing space is
        the same as ``">"`` -- VS Code inserts that space, and so does muscle memory.
        """
        text = self.query
        if not text:
            return (None, "")
        head = text[0]
        for provider in self.providers:
            if provider.prefix and head == provider.prefix:
                return (provider, text[1:].lstrip())
        return (None, text.strip())

    def _gather(self) -> List[SearchHit]:
        """Run the providers and rank their hits.

        With a prefix, exactly one provider answers and its own order is preserved -- it
        knows its domain better than a re-sort would. Without one, everything is blended
        and sorted on the shared :mod:`glplot.gui.fuzzy` scale, except for the empty query
        (see :meth:`_default_hits`).
        """
        provider, query = self._split_query()

        if provider is not None:
            method = getattr(provider, "prefixed_results", None)
            hits = method(query) if callable(method) else provider.results(query)
            return hits[:MAX_HITS]

        if not query:
            return self._default_hits()

        hits: List[SearchHit] = []
        for candidate in self.providers:
            try:
                hits.extend(candidate.results(query))
            except Exception:  # pragma: no cover - defensive
                logger.exception("Search provider %r failed", type(candidate).__name__)

        # Disabled last, then by score, then the tighter title. `sorted` is stable, so
        # provider order breaks any remaining tie deterministically.
        hits.sort(key=lambda h: (not h.enabled, -h.score, len(h.title)))
        return hits[:MAX_HITS]

    def _default_hits(self) -> List[SearchHit]:
        """The un-typed listing: recents first, then every command in authored order.

        Only commands. Listing every layer and dataset before the user has typed anything
        would bury the actions under whatever happens to be loaded, and the actions are
        what the palette is for.
        """
        provider = self.providers[0]
        try:
            hits = provider.results("")
        except Exception:  # pragma: no cover - defensive
            logger.exception("Command provider failed")
            return []

        by_id = {hit.mru_id: hit for hit in hits if hit.mru_id}
        recent: List[SearchHit] = []
        for action_id in self.recent:
            hit = by_id.get(action_id)
            if hit is not None:
                recent.append(hit)

        recent_ids = {hit.mru_id for hit in recent}
        rest = [hit for hit in hits if hit.mru_id not in recent_ids]
        for hit in recent:
            hit.category = "Recently Used"
        return (recent + rest)[:MAX_HITS]

    def _build_rows(self, hits: Sequence[SearchHit]) -> List[_Row]:
        """Lay the hits out into rows, inserting category headers for an empty query.

        Headers appear **only** when nothing is typed. Once there is a query the list is
        ranked by score, so a category header would falsely imply the rows under it are a
        group when they are merely adjacent.

        Every row carries its own ``top``, which is what lets :meth:`_apply_scroll` do
        exact, minimal scrolling and lets the hit test be arithmetic rather than a walk.
        """
        grouped = not self.query.strip()
        rows: List[_Row] = []
        y = 0.0
        last_category = None
        for hit in hits:
            if grouped and hit.category != last_category:
                rows.append(_Row(height=HEADER_HEIGHT, top=y, header=hit.category))
                y += HEADER_HEIGHT
                last_category = hit.category
            rows.append(_Row(height=ROW_HEIGHT, top=y, hit=hit))
            y += ROW_HEIGHT
        return rows

    def _hit_rows(self) -> List[int]:
        """Indices into :attr:`_rows` that are selectable (i.e. not headers)."""
        return [i for i, row in enumerate(self._rows) if row.hit is not None]

    # -- input -------------------------------------------------------------------------

    def _move(self, delta: int, count: int) -> None:
        """Move the selection by ``delta``, wrapping at both ends.

        Wrapping is deliberate: Up on the first row lands on the last, which is the
        fastest route to the bottom of a short list and is what every palette does.
        """
        if count <= 0:
            return
        self.selected = (self.selected + delta) % count
        self._scroll_to_selected = True

    def _handle_keys(self, count: int) -> None:
        """Arrow / Ctrl-P / Ctrl-N navigation and Escape.

        Read with raw GLFW keycodes through :mod:`glplot.gui.keys` rather than
        ``imgui.KEY_*``: those constants exist only for a handful of keys and are
        ``io.key_map`` indices, not keycodes (see the keys module docstring). ``repeat``
        is on for the arrows so holding one scrolls the list.

        Enter is **not** handled here -- it comes back from ``input_text`` via
        ``ENTER_RETURNS_TRUE`` in :meth:`draw`. Two mechanisms for one key would run the
        selected hit twice on the same frame.
        """
        if keys.pressed(_CHORD_ESCAPE):
            self.close()
            return
        if keys.pressed(_CHORD_DOWN, repeat=True) or keys.pressed(_CHORD_CTRL_N):
            self._move(1, count)
        elif keys.pressed(_CHORD_UP, repeat=True) or keys.pressed(_CHORD_CTRL_P):
            self._move(-1, count)

    def _run(self, hit: SearchHit) -> None:
        """Close, then run ``hit``. Order matters; so does not trusting ``run``.

        Closing first frees the keyboard focus the palette is holding, so a hit that
        focuses a panel or a text field actually gets it. And ``run`` is arbitrary code
        reached from a draw callback: letting it raise would unbalance the imgui stack
        mid-frame, so it is contained and logged.
        """
        if not hit.enabled:
            return
        if hit.mru_id:
            self._remember(hit.mru_id)
        self.close()
        try:
            hit.run()
        except Exception:
            logger.exception("Palette hit %r raised", hit.title)

    def _remember(self, action_id: str) -> None:
        """Push ``action_id`` to the front of the recents, de-duplicated and capped."""
        if action_id in self.recent:
            self.recent.remove(action_id)
        self.recent.insert(0, action_id)
        del self.recent[MRU_SIZE:]

    # -- drawing -----------------------------------------------------------------------

    def draw(self) -> None:
        """Draw the overlay. Call once per frame from ``Workspace.draw``; cheap when closed.

        Unlike a rail panel this opens and closes its own window: the palette is a modal
        overlay, and the workspace's generic ``begin(panel.title, True)`` would give it a
        title bar, a close box and a saved position.
        """
        if not IMGUI_AVAILABLE or not self.visible:
            return

        io = imgui.get_io()
        vw = float(io.display_size[0])
        vh = float(io.display_size[1])
        if vw <= 0.0 or vh <= 0.0:  # pragma: no cover - minimised/degenerate frame
            return

        hits = self._gather()
        self._rows = self._build_rows(hits)
        hit_rows = self._hit_rows()
        if hit_rows:
            self.selected = max(0, min(self.selected, len(hit_rows) - 1))
        else:
            self.selected = 0

        self._handle_keys(len(hit_rows))
        if not self.visible:  # Escape closed us this frame
            return

        self._draw_backdrop(vw, vh)

        width = min(PALETTE_WIDTH, vw - 40.0)
        imgui.set_next_window_position(vw * 0.5, vh * 0.14, imgui.ALWAYS, 0.5, 0.0)
        imgui.set_next_window_size(width, 0.0, imgui.ALWAYS)
        imgui.set_next_window_focus()
        flags = (
            imgui.WINDOW_NO_TITLE_BAR
            | imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_COLLAPSE
            | imgui.WINDOW_NO_SAVED_SETTINGS
            | imgui.WINDOW_ALWAYS_AUTO_RESIZE
            | imgui.WINDOW_NO_SCROLLBAR
        )
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (10.0, 10.0))
        imgui.begin("##glplot_palette", flags=flags)
        try:
            submitted = self._draw_input()
            self._draw_hint_line()
            self._draw_results(hit_rows)
            self._draw_footer(len(hit_rows))
        finally:
            # Unconditional: begin/end must balance even if a provider's row drawing
            # raised, or every window submitted after this one is lost.
            imgui.end()
            imgui.pop_style_var(1)

        if submitted and hit_rows:
            row = self._rows[hit_rows[self.selected]]
            if row.hit is not None:
                self._run(row.hit)

    def _draw_backdrop(self, vw: float, vh: float) -> None:
        """Dim everything behind the palette.

        The *background* draw list, not the foreground: it renders under every imgui
        window, so the palette and the panels stay legible while the plot behind them
        recedes. The foreground list would paint over the palette itself.
        """
        dl = imgui.get_background_draw_list()
        dl.add_rect_filled(0.0, 0.0, vw, vh, imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 0.45))

    def _draw_input(self) -> bool:
        """The search field. Returns True when Enter was pressed this frame."""
        icons.icon_inline("search", size=18.0, color="text_dim")
        imgui.same_line()

        if self._focus_input:
            # The opening frame only -- see the class docstring for why.
            imgui.set_keyboard_focus_here()
            self._focus_input = False

        imgui.push_item_width(-1.0)
        # buffer_length omitted per CONTRACT section 2.4: an explicit length silently
        # truncates, the default grows.
        submitted, value = imgui.input_text(
            "##palette_query",
            self.query,
            flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
        )
        imgui.pop_item_width()

        if value != self.query:
            self.query = value
            # The old index points into the old result list; anything but a reset would
            # land the user on an unrelated row as the ranking shifts under them.
            self.selected = 0
            self._scroll_to_selected = True
        return bool(submitted)

    def _draw_hint_line(self) -> None:
        """The prefix legend, shown only under an empty input."""
        if self.query:
            return
        parts = [f"{prefix} {name}" for prefix, name in _PREFIX_HINTS]
        imgui.text_disabled("   ".join(parts))

    def _draw_results(self, hit_rows: Sequence[int]) -> None:
        """The scrolling result list."""
        if not self._rows:
            imgui.spacing()
            imgui.text_disabled("No matches.")
            imgui.spacing()
            return

        total = sum(row.height for row in self._rows)
        list_h = min(total, ROW_HEIGHT * VISIBLE_ROWS)

        imgui.push_style_var(imgui.STYLE_ITEM_SPACING, (0.0, 0.0))
        visible = imgui.begin_child("##palette_results", 0.0, list_h, border=False)
        try:
            if visible:
                self._apply_scroll(hit_rows)
                width = imgui.get_content_region_available()[0]
                for index, row in enumerate(self._rows):
                    if row.hit is None:
                        self._draw_header(row, width)
                    else:
                        self._draw_hit_row(index, row, width, hit_rows)
        finally:
            # end_child is unconditional even when begin_child returned False
            # (CONTRACT section 2.2).
            imgui.end_child()
            imgui.pop_style_var(1)

    def _apply_scroll(self, hit_rows: Sequence[int]) -> None:
        """Keep the selected row on screen, scrolling by the minimum needed.

        Done by arithmetic on the rows' known ``top`` offsets rather than with
        ``set_scroll_here_y``, which recentres the selection on every move: arrowing down
        one row would jump the whole list, and the eye loses its place. This only nudges
        when the selection actually falls outside the viewport.
        """
        if not self._scroll_to_selected or not hit_rows:
            return
        self._scroll_to_selected = False

        row = self._rows[hit_rows[min(self.selected, len(hit_rows) - 1)]]
        # A header immediately above the selection is part of it: scroll to reveal both,
        # or the first row of a group always arrives with its title cut off.
        top = row.top
        index = hit_rows[min(self.selected, len(hit_rows) - 1)]
        if index > 0 and self._rows[index - 1].hit is None:
            top = self._rows[index - 1].top
        bottom = row.top + row.height

        scroll = imgui.get_scroll_y()
        view_h = imgui.get_window_height()
        if top < scroll:
            imgui.set_scroll_y(top)
        elif bottom > scroll + view_h:
            imgui.set_scroll_y(bottom - view_h)

    def _draw_header(self, row: _Row, width: float) -> None:
        """A category header inside the list."""
        dl = imgui.get_window_draw_list()
        ox, oy = imgui.get_cursor_screen_pos()
        imgui.dummy(width, row.height)
        dl.add_text(
            ox + _PAD_X,
            oy + (row.height - imgui.get_text_line_height()) * 0.5,
            theme.color_u32("text_dim"),
            row.header.upper(),
        )

    def _draw_hit_row(self, index: int, row: _Row, width: float, hit_rows: Sequence[int]) -> None:
        """One result row: icon, highlighted title, subtitle, category chip, shortcut.

        Drawn entirely by hand into the draw list over an ``invisible_button``. imgui's
        ``selectable`` cannot do the two-line layout, the per-character accent on the
        matched indices, or the right-aligned chord -- and those are exactly the details
        that separate this from a listbox.
        """
        hit = row.hit
        if hit is None:  # pragma: no cover - the caller routes headers to _draw_header
            return

        dl = imgui.get_window_draw_list()
        ox, oy = imgui.get_cursor_screen_pos()

        # get_cursor_screen_pos before the button: after it the cursor has advanced and
        # everything would draw one row low (CONTRACT section 3).
        clicked = imgui.invisible_button(f"##palette_row_{index}", width, row.height)
        hovered = imgui.is_item_hovered()
        is_selected = bool(hit_rows) and hit_rows[self.selected] == index

        x0, y0 = ox, oy
        x1, y1 = ox + width, oy + row.height

        if is_selected:
            dl.add_rect_filled(x0, y0, x1, y1, theme.color_u32("accent", 0.22), 4.0)
            dl.add_rect_filled(x0, y0, x0 + 2.5, y1, theme.color_u32("accent"), 0.0)
        elif hovered and hit.enabled:
            dl.add_rect_filled(x0, y0, x1, y1, theme.color_u32("text", 0.06), 4.0)

        dim = 0.45 if not hit.enabled else 1.0
        line_h = imgui.get_text_line_height()
        title_y = y0 + (row.height - line_h * 2.0 - 3.0) * 0.5
        sub_y = title_y + line_h + 3.0

        if hit.icon:
            icons.draw_icon(
                dl,
                hit.icon,
                x0 + _PAD_X + 8.0,
                (y0 + y1) * 0.5,
                8.0,
                theme.color_u32("text_dim", 0.9 * dim),
            )

        # Right-hand furniture first: it decides where the title has to stop.
        right = x1 - _PAD_X
        if hit.chord_text:
            w = imgui.calc_text_size(hit.chord_text)[0]
            dl.add_text(right - w, title_y, theme.color_u32("text_dim", dim), hit.chord_text)
            right -= w + 10.0
        if hit.category:
            right = self._draw_chip(dl, hit.category, right, title_y, line_h, dim)

        text_x = x0 + _PAD_X + _ICON_SLOT
        text_right = max(text_x + 20.0, right - 8.0)

        # Clip rather than ellipsise: a long title is truncated at a pixel boundary, but
        # its highlighted prefix -- the part that explains the match -- always survives.
        dl.push_clip_rect(text_x, y0, text_right, y1, True)
        self._draw_highlighted(dl, hit.title, hit.matches, text_x, title_y, dim)
        if hit.subtitle:
            dl.add_text(text_x, sub_y, theme.color_u32("muted", dim), hit.subtitle)
        dl.pop_clip_rect()

        if clicked:
            self._run(hit)

    def _draw_chip(
        self, dl: Any, text: str, right: float, y: float, line_h: float, dim: float
    ) -> float:
        """Draw a right-aligned category chip ending at ``right``; return its left edge."""
        w = imgui.calc_text_size(text)[0]
        x0 = right - w - _CHIP_PAD * 2.0
        dl.add_rect_filled(
            x0, y - 1.0, right, y + line_h + 1.0, theme.color_u32("raised_bg", dim), 3.0
        )
        dl.add_text(x0 + _CHIP_PAD, y, theme.color_u32("text_dim", dim), text)
        return x0 - 8.0

    def _draw_highlighted(
        self, dl: Any, text: str, matches: Sequence[int], x: float, y: float, dim: float
    ) -> None:
        """Draw ``text``, accenting the characters at ``matches``.

        The title is split into runs of matched/unmatched characters and each run is drawn
        as one ``add_text`` at an x advanced by its measured width. Per-character calls
        would be correct too, but they lose the font's kerning within a run and cost one
        ``calc_text_size`` per glyph on every row of every frame.

        ``matches`` come straight from :func:`glplot.gui.fuzzy.score` and index the
        original string, so no re-derivation is needed here -- and none is wanted: a
        second, independent guess at which characters matched is a second chance to be
        wrong about it.
        """
        base = theme.color_u32("text", dim)
        if not matches:
            dl.add_text(x, y, base, text)
            return

        accent = theme.color_u32("accent_hover", dim)
        marked = set(matches)
        cursor = x
        start = 0
        for i in range(1, len(text) + 1):
            at_end = i == len(text)
            if at_end or ((i in marked) != (start in marked)):
                segment = text[start:i]
                dl.add_text(cursor, y, accent if start in marked else base, segment)
                cursor += imgui.calc_text_size(segment)[0]
                start = i

    def _draw_footer(self, count: int) -> None:
        """The key legend and the result count. Discoverability costs one line."""
        imgui.separator()
        imgui.text_disabled(f"{count} result{'' if count == 1 else 's'}")
        legend = "up/down navigate    enter run    esc close"
        width = imgui.calc_text_size(legend)[0]
        avail = imgui.get_content_region_available()[0]
        if avail > width + 8.0:
            imgui.same_line()
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + avail - width)
            imgui.text_disabled(legend)

    def __repr__(self) -> str:
        return f"<CommandPalette visible={self.visible} query={self.query!r}>"
