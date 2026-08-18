"""Keyboard chord model for the workstation (parse, match, format).

The mechanism, verified against the installed ``imgui-bundle`` (Dear ImGui bindings)
and ``glfw`` rather than assumed:

* ``imgui.Key`` is an enum of named keys (``imgui.Key.a``, ``imgui.Key.f1``,
  ``imgui.Key._0``, ...) -- ``imgui.is_key_pressed`` takes one of those, never a raw
  integer keycode.
* GLFW keycodes are a different, unrelated numbering (GLFW's own frozen ABI -- see
  :data:`KEY_NAMES`). There is no arithmetic relationship between the two, so a keycode
  must be translated through an explicit table before it can reach imgui.
* :data:`_GLFW_TO_IMGUI_KEY` is that table, built once at import time to mirror
  ``GlfwRenderer._map_keys`` (``imgui_bundle.python_backends.glfw_backend``).
  :func:`pressed` looks a chord's key up in it and calls
  ``imgui.is_key_pressed(imgui_key)``; a key absent from the table (e.g. F13-F25, which
  the backend never wires to a key event) falls back to ``False`` rather than raising.

**Keycodes are hardcoded literals, not read from ``glfw``.** They are part of GLFW's
frozen public ABI, and hardcoding them keeps :data:`KEY_NAMES`, :func:`parse`,
:func:`format` and :func:`key_label` fully functional when ``glfw`` is not importable
(headless CI, GL-less boxes). Only :func:`pressed` needs a live library, and it returns
``False`` when one is missing. ``tests/test_gui_keys.py`` cross-checks every literal
against the real ``glfw`` module when it is importable, so drift cannot go unnoticed.

**"Mod" is one token, deliberately.** A binding is written ``"Mod+K"`` once and matches
Cmd on macOS and Ctrl everywhere else, because :func:`pressed` accepts ``key_ctrl`` *or*
``key_super``. Two per-platform binding tables would be two things to keep in sync.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Dict, Optional

try:
    import glfw

    GLFW_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - glfw is a hard dependency in CI
    GLFW_AVAILABLE = False
    glfw = None

try:
    from imgui_bundle import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - imgui is a hard dependency in CI
    IMGUI_AVAILABLE = False
    imgui = None

logger = logging.getLogger(__name__)

__all__ = [
    "GLFW_AVAILABLE",
    "IMGUI_AVAILABLE",
    "KEY_NAMES",
    "Chord",
    "format",
    "is_text_safe",
    "key_label",
    "parse",
    "pressed",
]


# --------------------------------------------------------------------------------------
# Keycode table (GLFW's frozen ABI values -- see the module docstring)
# --------------------------------------------------------------------------------------

KEY_NAMES: Dict[str, int] = {
    # Printable ASCII
    "Space": 32,
    "'": 39,
    ",": 44,
    "-": 45,
    ".": 46,
    "/": 47,
    ";": 59,
    "=": 61,
    "[": 91,
    "\\": 92,
    "]": 93,
    "`": 96,
    # Navigation / editing
    "Escape": 256,
    "Enter": 257,
    "Tab": 258,
    "Backspace": 259,
    "Insert": 260,
    "Delete": 261,
    "Right": 262,
    "Left": 263,
    "Down": 264,
    "Up": 265,
    "PageUp": 266,
    "PageDown": 267,
    "Home": 268,
    "End": 269,
    "CapsLock": 280,
    "PrintScreen": 283,
    "Pause": 284,
    "KeypadEnter": 335,
}

# Digits 0-9 -> 48-57, letters A-Z -> 65-90, function keys F1-F25 -> 290-314.
KEY_NAMES.update({str(d): 48 + d for d in range(10)})
KEY_NAMES.update({chr(ord("A") + i): 65 + i for i in range(26)})
KEY_NAMES.update({"F{0}".format(n): 289 + n for n in range(1, 26)})

#: Function keys are dispatchable while a text field has focus (see :func:`is_text_safe`).
_FUNCTION_KEYS = frozenset(range(290, 315))
_KEY_ESCAPE = 256

# --------------------------------------------------------------------------------------
# GLFW keycode -> imgui.Key translation
# --------------------------------------------------------------------------------------
#
# imgui-bundle's ``is_key_pressed`` takes an ``imgui.Key`` enum member, not a raw
# keycode. This table mirrors ``GlfwRenderer._map_keys``
# (``imgui_bundle.python_backends.glfw_backend``) for every key in :data:`KEY_NAMES`.
# F13-F25 have no entry: the backend never wires them to a key event, so
# ``is_key_pressed`` on them would only ever read False -- :func:`pressed` treats a
# missing entry as "no mapping" and returns False, the same observable result.
_GLFW_TO_IMGUI_KEY: Dict[int, "imgui.Key"] = {}
if IMGUI_AVAILABLE and imgui is not None:
    _GLFW_TO_IMGUI_KEY.update(
        {
            KEY_NAMES["Space"]: imgui.Key.space,
            KEY_NAMES["'"]: imgui.Key.apostrophe,
            KEY_NAMES[","]: imgui.Key.comma,
            KEY_NAMES["-"]: imgui.Key.minus,
            KEY_NAMES["."]: imgui.Key.period,
            KEY_NAMES["/"]: imgui.Key.slash,
            KEY_NAMES[";"]: imgui.Key.semicolon,
            KEY_NAMES["="]: imgui.Key.equal,
            KEY_NAMES["["]: imgui.Key.left_bracket,
            KEY_NAMES["\\"]: imgui.Key.backslash,
            KEY_NAMES["]"]: imgui.Key.right_bracket,
            KEY_NAMES["`"]: imgui.Key.grave_accent,
            KEY_NAMES["Escape"]: imgui.Key.escape,
            KEY_NAMES["Enter"]: imgui.Key.enter,
            KEY_NAMES["Tab"]: imgui.Key.tab,
            KEY_NAMES["Backspace"]: imgui.Key.backspace,
            KEY_NAMES["Insert"]: imgui.Key.insert,
            KEY_NAMES["Delete"]: imgui.Key.delete,
            KEY_NAMES["Right"]: imgui.Key.right_arrow,
            KEY_NAMES["Left"]: imgui.Key.left_arrow,
            KEY_NAMES["Down"]: imgui.Key.down_arrow,
            KEY_NAMES["Up"]: imgui.Key.up_arrow,
            KEY_NAMES["PageUp"]: imgui.Key.page_up,
            KEY_NAMES["PageDown"]: imgui.Key.page_down,
            KEY_NAMES["Home"]: imgui.Key.home,
            KEY_NAMES["End"]: imgui.Key.end,
            KEY_NAMES["CapsLock"]: imgui.Key.caps_lock,
            KEY_NAMES["PrintScreen"]: imgui.Key.print_screen,
            KEY_NAMES["Pause"]: imgui.Key.pause,
            KEY_NAMES["KeypadEnter"]: imgui.Key.keypad_enter,
        }
    )
    # Digits 0-9 -> imgui.Key._0.._9 (leading underscore: a Python identifier can't
    # start with a digit, so imgui-bundle's enum members are spelled that way).
    _GLFW_TO_IMGUI_KEY.update(
        {KEY_NAMES[str(d)]: getattr(imgui.Key, "_{0}".format(d)) for d in range(10)}
    )
    # Letters A-Z -> imgui.Key.a..z
    _GLFW_TO_IMGUI_KEY.update(
        {KEY_NAMES[chr(ord("A") + i)]: getattr(imgui.Key, chr(ord("a") + i)) for i in range(26)}
    )
    # F1-F12 only -- see the "no entry" note above for F13-F25.
    _GLFW_TO_IMGUI_KEY.update(
        {KEY_NAMES["F{0}".format(n)]: getattr(imgui.Key, "f{0}".format(n)) for n in range(1, 13)}
    )

# Aliases accepted by parse() but never produced by key_label(): a single canonical
# spelling per keycode keeps the reverse lookup unambiguous.
_ALIASES: Dict[str, str] = {
    "esc": "Escape",
    "return": "Enter",
    "ret": "Enter",
    "del": "Delete",
    "ins": "Insert",
    "pgup": "PageUp",
    "pgdn": "PageDown",
    "pagedn": "PageDown",
    "leftarrow": "Left",
    "rightarrow": "Right",
    "uparrow": "Up",
    "downarrow": "Down",
    "spacebar": "Space",
    "bksp": "Backspace",
    "padenter": "KeypadEnter",
    "kpenter": "KeypadEnter",
    "numpadenter": "KeypadEnter",
    "comma": ",",
    "period": ".",
    "dot": ".",
    "slash": "/",
    "backslash": "\\",
    "minus": "-",
    "dash": "-",
    "equal": "=",
    "equals": "=",
    "grave": "`",
    "backtick": "`",
    "semicolon": ";",
    "apostrophe": "'",
    "quote": "'",
    "leftbracket": "[",
    "rightbracket": "]",
}

# Case-insensitive lookup index: lowercased name (canonical or alias) -> keycode.
_LOOKUP: Dict[str, int] = {name.lower(): code for name, code in KEY_NAMES.items()}
_LOOKUP.update({alias: KEY_NAMES[canon] for alias, canon in _ALIASES.items()})

# Reverse lookup for key_label(), built from the canonical table only.
_CODE_TO_NAME: Dict[int, str] = {code: name for name, code in KEY_NAMES.items()}

_MOD_TOKENS = frozenset({"mod", "cmd", "command", "ctrl", "control", "super", "win", "meta"})
_SHIFT_TOKENS = frozenset({"shift"})
_ALT_TOKENS = frozenset({"alt", "option", "opt"})

# Display overrides. macOS spells the editing keys as glyphs; everyone else uses words.
_MAC_LABELS: Dict[int, str] = {
    256: "⎋",  # Escape
    257: "↩",  # Enter
    258: "⇥",  # Tab
    259: "⌫",  # Backspace
    260: "⎀",  # Insert
    261: "⌦",  # Delete (forward)
    262: "→",  # Right
    263: "←",  # Left
    264: "↓",  # Down
    265: "↑",  # Up
    266: "⇞",  # PageUp
    267: "⇟",  # PageDown
    268: "↖",  # Home
    269: "↘",  # End
    335: "⌤",  # KeypadEnter
}

_GENERIC_LABELS: Dict[int, str] = {
    262: "→",
    263: "←",
    264: "↓",
    265: "↑",
}


def _is_mac() -> bool:
    """Read ``sys.platform`` at call time so tests can monkeypatch it."""
    return sys.platform == "darwin"


@dataclass(frozen=True)
class Chord:
    """A key plus an exact modifier state.

    ``key`` is a raw GLFW keycode (see :data:`KEY_NAMES`). ``mod`` is the portable
    "Mod" modifier: Cmd on macOS, Ctrl elsewhere. Frozen and hashable so chords can key
    the conflict map in :mod:`glplot.gui.actions`.
    """

    key: int
    mod: bool = False
    shift: bool = False
    alt: bool = False


def parse(spec: str) -> Chord:
    """Parse a chord spec such as ``"Mod+Shift+P"``, ``"F1"``, ``"Delete"`` or ``"Mod+1"``.

    Tokens are separated by ``+`` and are case-insensitive. ``Ctrl``, ``Cmd``, ``Super``,
    ``Win``, ``Meta`` and ``Mod`` all mean the same portable Mod modifier -- callers write
    one binding and it works on every platform. ``Alt``/``Option``/``Opt`` and ``Shift``
    are the other two.

    Raises:
        ValueError: on an empty spec, an unknown key name, a missing key, or more than
            one non-modifier token.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("empty chord spec")

    # Split on '+' but keep a literal trailing '+' usable as nothing: GLFW has no PLUS
    # keycode (it is Shift+Equal), so such a spec is a genuine error, not a special case.
    tokens = [tok.strip() for tok in spec.split("+")]
    mod = shift = alt = False
    key: Optional[int] = None

    for token in tokens:
        if not token:
            raise ValueError("malformed chord spec {0!r}: empty token".format(spec))
        low = token.lower()
        if low in _MOD_TOKENS:
            mod = True
        elif low in _SHIFT_TOKENS:
            shift = True
        elif low in _ALT_TOKENS:
            alt = True
        else:
            if key is not None:
                raise ValueError(
                    "malformed chord spec {0!r}: more than one non-modifier key".format(spec)
                )
            code = _LOOKUP.get(low)
            if code is None:
                raise ValueError("unknown key name {0!r} in chord spec {1!r}".format(token, spec))
            key = code

    if key is None:
        raise ValueError("chord spec {0!r} has modifiers but no key".format(spec))
    return Chord(key=key, mod=mod, shift=shift, alt=alt)


def pressed(chord: Chord, *, repeat: bool = False) -> bool:
    """True on the frame ``chord`` fires: its key went down **and** modifiers match exactly.

    The exact match is the whole point. ``is_key_pressed(KEY_Z) and io.key_ctrl`` -- the
    obvious spelling -- is also true for ``Mod+Shift+Z``, so binding undo to ``Mod+Z`` and
    redo to ``Mod+Shift+Z`` would run *undo on every redo*. Every modifier is therefore
    compared against the chord, including the ones the chord does not ask for.

    Mod matches ``io.key_ctrl`` **or** ``io.key_super``, which is what lets a single
    ``"Mod+K"`` binding serve macOS and everything else.

    Returns ``False`` when imgui is unavailable, no context is current, or ``chord.key``
    has no imgui-side mapping (see :data:`_GLFW_TO_IMGUI_KEY`), so callers in a headless
    process need no guard of their own.
    """
    if not IMGUI_AVAILABLE or imgui is None or not GLFW_AVAILABLE:
        return False
    try:
        io = imgui.get_io()
    except Exception:  # pragma: no cover - no current imgui context
        return False

    # Modifiers first: three attribute reads reject the overwhelming majority of chords
    # before touching is_key_pressed.
    if bool(io.key_ctrl or io.key_super) != bool(chord.mod):
        return False
    if bool(io.key_shift) != bool(chord.shift):
        return False
    if bool(io.key_alt) != bool(chord.alt):
        return False

    imgui_key = _GLFW_TO_IMGUI_KEY.get(chord.key)
    if imgui_key is None:
        return False
    try:
        return bool(imgui.is_key_pressed(imgui_key, repeat))
    except Exception:  # pragma: no cover - defensive: unexpected imgui-side failure
        return False


def is_text_safe(chord: Chord) -> bool:
    """True if ``chord`` may still fire while a text field has keyboard focus.

    A bare-letter chord must not: typing ``d`` into a data cell would otherwise run
    "Toggle Density". A chord carrying Mod, a function key, or Escape is unambiguous --
    no text field produces those as input -- so they stay live. This predicate is the
    single rule :meth:`glplot.gui.actions.ActionRegistry.dispatch` consults.

    Note Shift and Alt do **not** make a chord text-safe: ``Shift+A`` is how you type a
    capital A, and on many layouts Alt+key produces a character.
    """
    return chord.mod or chord.key in _FUNCTION_KEYS or chord.key == _KEY_ESCAPE


def key_label(key: int) -> str:
    """Human-readable name for a GLFW keycode, platform-appropriate.

    macOS renders the editing keys as glyphs (``↩``, ``⌫``, ``⌦``);
    elsewhere they are words. Unknown codes degrade to ``"Key<code>"`` rather than
    raising -- a label is never worth crashing a draw callback over.
    """
    if _is_mac():
        label = _MAC_LABELS.get(key)
        if label is not None:
            return label
    else:
        label = _GENERIC_LABELS.get(key)
        if label is not None:
            return label
    name = _CODE_TO_NAME.get(key)
    if name is not None:
        return name
    return "Key{0}".format(key)


def format(chord: Chord) -> str:  # noqa: A001 - the spec names this `format`
    """Render ``chord`` for display: ``"⇧⌘P"`` on macOS, ``"Ctrl+Shift+P"`` elsewhere.

    macOS uses no separators and Apple's canonical modifier order
    (Control, Option, Shift, Command) -- since our Mod *is* Command there, it sorts last.
    Other platforms use the conventional ``Ctrl+Alt+Shift+Key`` with ``+`` separators.
    """
    if _is_mac():
        # Apple order: ctrl, option, shift, command. Chord.mod is Command on macOS.
        parts = []
        if chord.alt:
            parts.append("⌥")
        if chord.shift:
            parts.append("⇧")
        if chord.mod:
            parts.append("⌘")
        parts.append(key_label(chord.key))
        return "".join(parts)

    parts = []
    if chord.mod:
        parts.append("Ctrl")
    if chord.alt:
        parts.append("Alt")
    if chord.shift:
        parts.append("Shift")
    parts.append(key_label(chord.key))
    return "+".join(parts)
