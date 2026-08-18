"""The action registry: one definition per user-facing action, three consumers.

Every action the user can invoke is registered here exactly once. The keymap dispatches
from this registry, the command palette searches it, and the Help panel's shortcut sheet
is *generated* from it. That is the point: a hand-written shortcut list in Help drifts out
of sync with the keymap the first time someone rebinds a key, and nothing catches it.
Here, Help cannot lie -- it renders whatever :meth:`ActionRegistry.all` says.

**Not to be confused with :mod:`glplot.gui.commands`.** That module is the deferred-
mutation ``CommandQueue`` required by CONTRACT section 1 (draw callbacks may not touch
``plot.scene``). This module is a catalogue of invocable UI actions. An action's ``run``
that mutates the scene is expected to *submit to* that queue; the two are layered, not
alternatives.

Pure logic apart from reading ``io`` in :meth:`ActionRegistry.dispatch`, so the imgui
import stays guarded and the module imports headless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import fuzzy
from .keys import Chord, is_text_safe
from .keys import parse as parse_chord
from .keys import pressed as chord_pressed

try:
    from imgui_bundle import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - imgui is a hard dependency in CI
    IMGUI_AVAILABLE = False
    imgui = None

logger = logging.getLogger(__name__)

__all__ = ["Action", "ActionRegistry", "IMGUI_AVAILABLE"]

# Relative weight of each searched field. A title hit is the real thing; matching only a
# category or a hidden keyword is a weaker signal and must not outrank it.
_WEIGHT_TITLE = 1.0
_WEIGHT_KEYWORD = 0.75
_WEIGHT_CATEGORY = 0.5


@dataclass
class Action:
    """One user-facing action: what it is called, what it does, how to reach it.

    Attributes:
        id: Dotted, stable identifier (``"data.paste_as_new"``). Stable across releases --
            it is what an MRU list or a user keymap persists.
        title: Human label, shown in the palette and the Help sheet.
        category: Palette grouping and Help section ("Data", "View", "Math", ...).
        run: Zero-argument callable. If it mutates the scene it must go through
            ``ws.queue.submit`` (CONTRACT section 1), never touch ``plot.scene`` directly.
        chord: Keyboard binding, or ``None`` for palette-only actions.
        icon: A name from :data:`glplot.gui.icons.ICON_SHAPES`, or ``None``.
        keywords: Extra search terms that are not in the title ("clipboard", "excel").
        description: One line, shown in the palette subtitle and the Help sheet.
        is_enabled: Predicate; ``None`` means always enabled.
        is_checked: Predicate for toggles; the palette shows a check. ``None`` for
            non-toggles.
    """

    id: str
    title: str
    category: str
    run: Callable[[], None]
    chord: Optional[Chord] = None
    icon: Optional[str] = None
    keywords: Tuple[str, ...] = ()
    description: str = ""
    is_enabled: Optional[Callable[[], bool]] = None
    is_checked: Optional[Callable[[], bool]] = None
    # Registration order, assigned by the registry. Breaks score ties deterministically
    # and gives the palette a sensible order for the empty query.
    order: int = field(default=0, compare=False)


class ActionRegistry:
    """Catalogue of every invocable action, with search and keyboard dispatch."""

    def __init__(self) -> None:
        self._actions: List[Action] = []
        self._by_id: Dict[str, Action] = {}

    # -- registration ------------------------------------------------------------------

    def register(self, action: Action) -> Action:
        """Register ``action`` and return it.

        Raises:
            ValueError: if ``action.id`` is already registered. A duplicate id is always a
                bug -- two actions would answer to one MRU entry and one keymap slot --
                and it is silent unless we raise.
        """
        if action.id in self._by_id:
            raise ValueError("duplicate action id {0!r}".format(action.id))
        action.order = len(self._actions)
        self._actions.append(action)
        self._by_id[action.id] = action
        return action

    def add(
        self,
        id: str,  # noqa: A002 - matches the Action field name
        title: str,
        category: str,
        run: Callable[[], None],
        **kw: object,
    ) -> Action:
        """Construct and register an :class:`Action` in one call.

        ``chord`` may be given as a spec string (``chord="Mod+K"``) as well as a
        :class:`~glplot.gui.keys.Chord`; the string is parsed here so the ~55 call sites
        in ``workspace._register_actions`` stay readable. ``keywords`` may be any
        sequence and is normalised to a tuple.

        Raises:
            ValueError: on a duplicate id, or on an unparseable chord spec.
        """
        chord = kw.pop("chord", None)
        if isinstance(chord, str):
            chord = parse_chord(chord)
        keywords = kw.pop("keywords", ())
        if isinstance(keywords, str):
            keywords = (keywords,)
        elif keywords is not None:
            keywords = tuple(keywords)  # type: ignore[arg-type]

        action = Action(
            id=id,
            title=title,
            category=category,
            run=run,
            chord=chord,  # type: ignore[arg-type]
            keywords=keywords,  # type: ignore[arg-type]
            **kw,  # type: ignore[arg-type]
        )
        return self.register(action)

    # -- lookup ------------------------------------------------------------------------

    def get(self, id: str) -> Optional[Action]:  # noqa: A002 - matches the Action field
        """Return the action with this id, or ``None``."""
        return self._by_id.get(id)

    def all(self) -> List[Action]:
        """Every registered action, in registration order. A copy: callers may not mutate."""
        return list(self._actions)

    def categories(self) -> List[str]:
        """Distinct categories in first-registration order.

        Not sorted alphabetically: registration order is authored order, which lets
        ``_register_actions`` decide that "Global" heads the Help sheet and "Layer" tails
        it without a separate ordering table.
        """
        seen: List[str] = []
        for action in self._actions:
            if action.category not in seen:
                seen.append(action.category)
        return seen

    def enabled(self, action: Action) -> bool:
        """Whether ``action`` is currently invocable.

        ``is_enabled=None`` means always. A predicate that *raises* is logged and treated
        as disabled: this runs once per action per frame, so letting it propagate would
        take the whole GUI down at 60Hz, and running an action whose own guard just
        crashed is the more dangerous of the two failure modes.
        """
        if action.is_enabled is None:
            return True
        try:
            return bool(action.is_enabled())
        except Exception:
            logger.exception(
                "is_enabled predicate failed for action %r; treating as disabled", action.id
            )
            return False

    def checked(self, action: Action) -> Optional[bool]:
        """Toggle state for ``action``, or ``None`` if it is not a toggle.

        Same failure policy as :meth:`enabled`: a raising predicate degrades to ``None``
        (draw a toggle with no check) rather than killing the frame.
        """
        if action.is_checked is None:
            return None
        try:
            return bool(action.is_checked())
        except Exception:
            logger.exception("is_checked predicate failed for action %r", action.id)
            return None

    # -- search ------------------------------------------------------------------------

    def search(self, query: str) -> List[Tuple[Action, int, List[int]]]:
        """Rank actions against ``query``; returns ``(action, score, title_indices)``.

        Searches title, category and keywords; the best-scoring field wins, after the
        field weights above. **The returned indices are always title indices** (empty when
        the title itself did not match) -- the palette highlights the title, so indices
        into a keyword it never draws would highlight the wrong characters.

        Disabled actions are ranked last but still listed, greyed. Hiding them makes the
        palette feel broken: the user searches for the thing they know exists, gets no
        result, and cannot tell whether they mistyped or it is unavailable.

        An empty query returns every action in registration order (enabled first), which
        is what the palette lists before the user types anything.
        """
        if not query:
            return [
                (a, 0, [])
                for a in sorted(self._actions, key=lambda a: (not self.enabled(a), a.order))
            ]

        results: List[Tuple[Action, int, List[int]]] = []
        for action in self._actions:
            title_hit = fuzzy.score(query, action.title)
            best = -1
            if title_hit is not None:
                best = int(title_hit[0] * _WEIGHT_TITLE)

            cat_hit = fuzzy.score(query, action.category)
            if cat_hit is not None:
                best = max(best, int(cat_hit[0] * _WEIGHT_CATEGORY))

            for keyword in action.keywords:
                kw_hit = fuzzy.score(query, keyword)
                if kw_hit is not None:
                    best = max(best, int(kw_hit[0] * _WEIGHT_KEYWORD))

            if best < 0:
                continue
            indices = list(title_hit[1]) if title_hit is not None else []
            results.append((action, best, indices))

        # Disabled last, then by score, then shortest title (a query is likelier to mean
        # the tighter match), then registration order so the ranking is fully stable.
        results.sort(key=lambda r: (not self.enabled(r[0]), -r[1], len(r[0].title), r[0].order))
        return results

    # -- keyboard ----------------------------------------------------------------------

    def dispatch(self) -> Optional[str]:
        """Test every bound chord; run the first enabled match. Returns its id, or ``None``.

        Call once per frame, **before the panels draw** -- and that placement is load-
        bearing, not stylistic. Dear ImGui computes ``io.want_text_input`` during
        ``end_frame`` from whichever widget is active, so any read of it reports the
        *previous* frame's state (verified: a field focused on frame N first reports
        ``want_text_input`` on frame N+2, and it stays set for one frame after the field
        goes away). Dispatching at the top of the frame therefore asks exactly the right
        question -- "was a text field focused as of the last completed frame" -- and errs
        on the safe side, suppressing bare letters one frame longer rather than leaking a
        stray hotkey into a field the user is still typing in.

        **The text-input rule.** When imgui reports ``io.want_text_input`` -- i.e. an
        ``input_text`` has keyboard focus -- only chords for which
        :func:`glplot.gui.keys.is_text_safe` holds are dispatched: those carrying Mod, the
        function keys, and Escape. Everything else is suppressed. Without this, typing
        ``sin(x)`` into the function generator would fire the bare-letter actions bound to
        ``s``, ``i`` and ``n``, which is exactly the class of bug that made every text
        field in the engine's own HUD unusable. ``Mod+Z`` and ``F1`` still work while
        typing, because no text field can produce them as input.

        "First match" is registration order. Two enabled actions on one chord is a
        conflict -- only the earlier one ever runs -- which is why :meth:`conflicts`
        exists and Help surfaces it rather than leaving it mysterious.
        """
        if not IMGUI_AVAILABLE or imgui is None:
            return None
        try:
            want_text = bool(imgui.get_io().want_text_input)
        except Exception:  # pragma: no cover - no current imgui context
            return None

        for action in self._actions:
            chord = action.chord
            if chord is None:
                continue
            if want_text and not is_text_safe(chord):
                continue
            if not chord_pressed(chord):
                continue
            if not self.enabled(action):
                # A disabled action swallows its chord rather than falling through to a
                # conflicting one: the binding belongs to it, it is merely unavailable.
                continue
            try:
                action.run()
            except Exception:
                logger.exception("action %r raised", action.id)
            return action.id
        return None

    def conflicts(self) -> List[Tuple[Chord, List[Action]]]:
        """Chords bound to more than one currently-enabled action.

        Only the first such action can ever fire (see :meth:`dispatch`), so a conflict is
        a silent dead binding. Help > Keyboard renders this list, which turns "why doesn't
        my shortcut work" into something the user can see.
        """
        by_chord: Dict[Chord, List[Action]] = {}
        for action in self._actions:
            if action.chord is None or not self.enabled(action):
                continue
            by_chord.setdefault(action.chord, []).append(action)
        return [(chord, actions) for chord, actions in by_chord.items() if len(actions) > 1]
