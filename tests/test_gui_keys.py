"""Test the keyboard chord model in glplot.gui.keys.

Focus on parse/format/key_label round-trips, per-platform Mod rendering, and --
the one that actually matters -- exact modifier matching in pressed(), which is
what keeps undo off the redo chord.

pressed() is exercised against a REAL imgui context driven headlessly (no OpenGL,
no GPU, no window): the context is created, io.keys_down is indexed by raw GLFW
keycode exactly as GlfwRenderer.keyboard_callback does it, and new_frame/end_frame
are stepped so is_key_pressed sees genuine rising edges. Mocking imgui away here
would defeat the purpose -- the whole module rests on the claim that
is_key_pressed() accepts a GLFW keycode, and only a real frame can falsify it.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

import imgui
import pytest

from glplot.gui import keys
from glplot.gui.keys import Chord, format, is_text_safe, key_label, parse, pressed

#: Keycodes under test come from the module's own table, never from a module-level
#: ``import glfw``. tests/test_camera_anisotropy.py installs
#: ``sys.modules["glfw"] = MagicMock()`` at import time and never restores it, and it
#: collects before this file, so a plain ``import glfw`` here yields MagicMock attributes
#: instead of keycodes. keys.py is immune to that because its keycodes are literals --
#: which is exactly the property _real_glfw() below exists to police.
KEY = keys.KEY_NAMES

# Canonical name -> the glfw attribute suffix, where the two spellings differ.
# Any name absent from this map is expected to match ``glfw.KEY_<NAME>`` verbatim.
_GLFW_ALIASES = {
    "Space": "SPACE",
    "'": "APOSTROPHE",
    ",": "COMMA",
    "-": "MINUS",
    ".": "PERIOD",
    "/": "SLASH",
    ";": "SEMICOLON",
    "=": "EQUAL",
    "[": "LEFT_BRACKET",
    "\\": "BACKSLASH",
    "]": "RIGHT_BRACKET",
    "`": "GRAVE_ACCENT",
    "Escape": "ESCAPE",
    "Enter": "ENTER",
    "Tab": "TAB",
    "Backspace": "BACKSPACE",
    "Insert": "INSERT",
    "Delete": "DELETE",
    "Right": "RIGHT",
    "Left": "LEFT",
    "Down": "DOWN",
    "Up": "UP",
    "PageUp": "PAGE_UP",
    "PageDown": "PAGE_DOWN",
    "Home": "HOME",
    "End": "END",
    "CapsLock": "CAPS_LOCK",
    "PrintScreen": "PRINT_SCREEN",
    "Pause": "PAUSE",
    "KeypadEnter": "KP_ENTER",
}


def _real_glfw():
    """Import the genuine glfw, bypassing any MagicMock a sibling test left in sys.modules.

    Only the KEY_NAMES cross-check needs this: its entire job is to prove the module's
    hardcoded literals still equal GLFW's, which a mock would answer with a shrug.
    sys.modules is restored exactly as found so no other test sees a change.
    """
    saved = sys.modules.get("glfw")
    if isinstance(getattr(saved, "KEY_A", None), int):
        return saved
    sys.modules.pop("glfw", None)
    try:
        return importlib.import_module("glfw")
    except Exception:  # pragma: no cover - glfw is a hard dependency in CI
        pytest.skip("glfw is not importable")
    finally:
        if saved is None:
            sys.modules.pop("glfw", None)
        else:
            sys.modules["glfw"] = saved


def _workspace_chord_specs():
    """Every chord spec the workspace and palette actually bind, from the real tables.

    Imported rather than hand-listed so the coverage assertion cannot rot: a new
    binding with a typo'd key name fails this test on the next run.
    """
    from glplot.gui import workspace as ws

    specs = set(ws._PANEL_CHORDS.values())
    specs.update(op[3] for op in ws._MATH_OPS if op[3])
    import re
    from pathlib import Path

    for mod in (ws, __import__("glplot.gui.panels.palette", fromlist=["x"])):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        specs.update(re.findall(r'chord="([^"]+)"', src))
        specs.update(re.findall(r'keys\.parse\("([^"]+)"\)', src))
    return sorted(specs)


class _Keyboard:
    """Drives a real headless imgui context one frame at a time.

    Mirrors GlfwRenderer.keyboard_callback: io.keys_down is indexed by raw GLFW
    keycode, and the modifier booleans are set from the GLFW mods bitfield.
    """

    def __init__(self, io) -> None:
        """Retain the io of the context under test."""
        self.io = io

    def step(self, down=(), ctrl=False, shift=False, alt=False, super_=False) -> None:
        """Run one full frame with exactly these keys and modifiers held down."""
        for i in range(512):
            self.io.keys_down[i] = False
        for code in down:
            self.io.keys_down[code] = True
        self.io.key_ctrl = ctrl
        self.io.key_shift = shift
        self.io.key_alt = alt
        self.io.key_super = super_
        imgui.new_frame()
        imgui.begin("probe")
        imgui.end()
        imgui.end_frame()

    def idle(self) -> None:
        """Run one frame with nothing held, to clear any rising edge."""
        self.step()

    def tap(self, *specs, down=(), **mods) -> tuple:
        """Release everything, then press ``down``+``mods``, and probe each spec.

        Returns one bool per spec, evaluated on the rising-edge frame -- the only
        frame on which a non-repeating chord may fire.
        """
        self.idle()
        self.step(down=down, **mods)
        return tuple(pressed(parse(s)) for s in specs)


@pytest.fixture
def kbd():
    """A real imgui context, created and destroyed per test. No GL required."""
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = 800, 600
    io.fonts.get_tex_data_as_rgba32()
    io.fonts.texture_id = 1
    io.delta_time = 1.0 / 60.0
    yield _Keyboard(io)
    imgui.destroy_context(ctx)


class TestKeyNames:
    """Test the KEY_NAMES keycode table against the real glfw module."""

    def test_every_keycode_matches_real_glfw(self):
        """Hardcoded literals must equal glfw's own constants, or pressed() is a lie."""
        glfw = _real_glfw()
        mismatched = []
        for name, code in keys.KEY_NAMES.items():
            attr = "KEY_" + _GLFW_ALIASES.get(name, name)
            actual = getattr(glfw, attr, None)
            if actual != code:
                mismatched.append((name, code, actual))
        assert mismatched == []

    def test_keycodes_fit_the_keys_down_array(self):
        """Every keycode must index io.keys_down, whose 512 slots are the real bound."""
        assert all(0 <= code < 512 for code in keys.KEY_NAMES.values())
        assert _real_glfw().KEY_LAST < 512

    def test_canonical_names_are_unique_per_keycode(self):
        """One name per code, or key_label()'s reverse lookup would be ambiguous."""
        codes = list(keys.KEY_NAMES.values())
        assert len(codes) == len(set(codes))

    def test_covers_letters_digits_and_function_keys(self):
        """The vocabulary the action set draws on must be present."""
        glfw = _real_glfw()
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            assert KEY[letter] == getattr(glfw, "KEY_" + letter)
        for digit in range(10):
            assert KEY[str(digit)] == getattr(glfw, "KEY_%d" % digit)
        for n in range(1, 26):
            assert KEY["F%d" % n] == getattr(glfw, "KEY_F%d" % n)

    def test_the_glfw_module_is_not_a_prerequisite(self):
        """keys.py must never read keycodes out of glfw at import time.

        A sibling test file replaces sys.modules["glfw"] with a MagicMock and never
        restores it. The literals are what keep KEY_NAMES correct regardless.
        """
        assert isinstance(KEY["Z"], int)
        assert not isinstance(keys.KEY_NAMES["Z"], MagicMock)

    def test_covers_every_chord_the_workspace_binds(self):
        """Each real binding must parse -- an unknown key name is a startup crash."""
        specs = _workspace_chord_specs()
        assert len(specs) >= 30, "chord tables did not load; the scan found nothing"
        for spec in specs:
            assert isinstance(parse(spec), Chord), spec

    def test_workspace_bindings_survive_a_canonical_round_trip(self):
        """Every real binding must re-parse from its canonical spec.

        Note this goes through _canonical_spec(), NOT format(). format() is a
        display renderer -- it emits "⇧⌘P" and "Alt+↑" -- and was never meant to
        be machine-readable. The invariant that does hold is that KEY_NAMES round-
        trips: every keycode a chord can carry has a name that parses back to it.
        """
        for spec in _workspace_chord_specs():
            chord = parse(spec)
            assert parse(_canonical_spec(chord)) == chord, spec


#: Reverse of the public KEY_NAMES table: keycode -> canonical spec name.
_CANONICAL_NAMES = {code: name for name, code in keys.KEY_NAMES.items()}


def _canonical_spec(chord: Chord) -> str:
    """Render ``chord`` as a parseable spec string, independent of platform.

    This is deliberately not format(): format() is for humans and emits glyphs.
    """
    parts = []
    if chord.mod:
        parts.append("Mod")
    if chord.shift:
        parts.append("Shift")
    if chord.alt:
        parts.append("Alt")
    parts.append(_CANONICAL_NAMES[chord.key])
    return "+".join(parts)


def _format_as(platform: str, chord: Chord) -> str:
    """Render ``chord`` as ``platform`` would, by swapping sys.platform around the call."""
    saved = sys.platform
    try:
        sys.platform = platform
        return format(chord)
    finally:
        sys.platform = saved


class TestParse:
    """Test parse() token handling, aliases and error reporting."""

    def test_parses_a_plain_key(self):
        """A bare key name yields a chord with no modifiers."""
        assert parse("F1") == Chord(key=KEY["F1"])
        assert parse("Delete") == Chord(key=KEY["Delete"])

    def test_parses_modifier_combinations(self):
        """Mod/Shift/Alt each set their own flag and nothing else."""
        assert parse("Mod+K") == Chord(key=KEY["K"], mod=True)
        assert parse("Mod+Shift+P") == Chord(key=KEY["P"], mod=True, shift=True)
        assert parse("Shift+Enter") == Chord(key=KEY["Enter"], shift=True)
        assert parse("Alt+Up") == Chord(key=KEY["Up"], alt=True)
        assert parse("Mod+Shift+Alt+Z") == Chord(key=KEY["Z"], mod=True, shift=True, alt=True)

    def test_parses_a_digit_key(self):
        """ "Mod+1" is a panel binding and must resolve to the digit keycode, not text."""
        assert parse("Mod+1") == Chord(key=KEY["1"], mod=True)

    def test_token_names_are_case_insensitive(self):
        """Spec says case-insensitive; the action tables rely on it."""
        assert parse("mod+shift+p") == parse("MOD+SHIFT+P") == parse("Mod+Shift+P")
        assert parse("delete") == parse("DELETE") == parse("Delete")

    def test_ctrl_cmd_and_mod_are_the_same_token(self):
        """One portable binding: Ctrl, Cmd, Super, Win, Meta and Mod all mean Mod."""
        target = Chord(key=KEY["Z"], mod=True)
        for token in ("Mod", "Ctrl", "Control", "Cmd", "Command", "Super", "Win", "Meta"):
            assert parse(token + "+Z") == target, token

    def test_alt_and_option_are_the_same_token(self):
        """macOS calls it Option; the parser accepts both spellings."""
        for token in ("Alt", "Option", "Opt"):
            assert parse(token + "+Down") == Chord(key=KEY["Down"], alt=True), token

    def test_key_aliases_resolve_to_canonical_codes(self):
        """Alias spellings must land on the same keycode as the canonical name."""
        assert parse("Esc") == parse("Escape")
        assert parse("Return") == parse("Enter")
        assert parse("Del") == parse("Delete")
        assert parse("PgUp") == parse("PageUp")
        assert parse("Comma") == parse(",")
        assert parse("Backslash") == parse("\\")

    def test_whitespace_around_tokens_is_tolerated(self):
        """ " Mod + Shift + P " is the same chord; authored tables get sloppy."""
        assert parse(" Mod + Shift + P ") == parse("Mod+Shift+P")

    def test_round_trips_through_parse(self):
        """parse() is idempotent: re-parsing a chord's canonical spec is a no-op."""
        for spec in ("Mod+K", "Mod+Shift+P", "F1", "Delete", "Shift+Enter", "Mod+1", "Alt+Up"):
            chord = parse(spec)
            assert parse(_canonical_spec(chord)) == chord, spec

    def test_every_key_name_round_trips(self):
        """Every name in the public table must parse back to its own keycode."""
        for name, code in keys.KEY_NAMES.items():
            assert parse(name) == Chord(key=code), name

    def test_rejects_unknown_key_names(self):
        """An unknown key name must raise, not silently bind nothing."""
        for spec in ("Mod+Nope", "Banana", "Mod+Shift+F99", "Mod+KEY_K"):
            with pytest.raises(ValueError):
                parse(spec)

    def test_unknown_key_error_names_the_offending_token(self):
        """The message must point at the bad token; a chord table has ~55 rows."""
        with pytest.raises(ValueError, match="unknown key name"):
            parse("Mod+Nope")

    def test_rejects_empty_and_modifier_only_specs(self):
        """Empty, blank, or modifiers with no key are all errors."""
        for spec in ("", "   ", "Mod", "Mod+Shift", "Shift"):
            with pytest.raises(ValueError):
                parse(spec)

    def test_rejects_malformed_specs(self):
        """Empty tokens and two non-modifier keys are errors, not guesses."""
        for spec in ("Mod+", "+K", "Mod++K", "Mod+K+J", "A+B"):
            with pytest.raises(ValueError):
                parse(spec)

    def test_rejects_non_string_input(self):
        """A non-string spec raises ValueError rather than AttributeError."""
        for spec in (None, 42, ["Mod", "K"]):
            with pytest.raises(ValueError):
                parse(spec)

    def test_chord_is_frozen_and_hashable(self):
        """Chords key the conflict map in actions.py, so they must hash."""
        chord = parse("Mod+K")
        assert {chord: 1}[parse("Mod+K")] == 1
        with pytest.raises(Exception):
            chord.key = 5


class TestModPerPlatform:
    """Test that Mod renders as Cmd on macOS and Ctrl elsewhere."""

    def test_mod_renders_as_command_glyph_on_macos(self, monkeypatch):
        """On darwin, Mod is the Command key and shows as the loop glyph."""
        monkeypatch.setattr(sys, "platform", "darwin")
        assert format(parse("Mod+K")) == "⌘K"

    def test_mod_renders_as_ctrl_word_on_linux(self, monkeypatch):
        """Off darwin, Mod is Ctrl and spells itself out."""
        monkeypatch.setattr(sys, "platform", "linux")
        assert format(parse("Mod+K")) == "Ctrl+K"

    def test_mod_renders_as_ctrl_word_on_windows(self, monkeypatch):
        """win32 takes the same non-mac branch as linux."""
        monkeypatch.setattr(sys, "platform", "win32")
        assert format(parse("Mod+K")) == "Ctrl+K"

    def test_platform_is_read_at_call_time(self, monkeypatch):
        """Not captured at import: the same chord must re-render per platform."""
        chord = parse("Mod+Shift+P")
        monkeypatch.setattr(sys, "platform", "darwin")
        mac = format(chord)
        monkeypatch.setattr(sys, "platform", "linux")
        assert format(chord) != mac


class TestFormat:
    """Test format() output on both platforms."""

    def test_macos_uses_apple_order_and_no_separators(self, monkeypatch):
        """Apple order is Control, Option, Shift, Command -- Mod (Command) sorts last."""
        monkeypatch.setattr(sys, "platform", "darwin")
        assert format(parse("Mod+Shift+P")) == "⇧⌘P"
        assert format(parse("Mod+Alt+Shift+P")) == "⌥⇧⌘P"
        assert format(parse("Alt+P")) == "⌥P"
        assert "+" not in format(parse("Mod+Shift+P"))

    def test_linux_uses_ctrl_alt_shift_order_with_plus(self, monkeypatch):
        """Conventional ordering and separators off darwin."""
        monkeypatch.setattr(sys, "platform", "linux")
        assert format(parse("Mod+Shift+P")) == "Ctrl+Shift+P"
        assert format(parse("Mod+Alt+Shift+P")) == "Ctrl+Alt+Shift+P"
        assert format(parse("Alt+Down")) == "Alt+↓"

    def test_unmodified_chord_is_just_the_key(self, monkeypatch):
        """No modifiers means no prefix on either platform."""
        for platform in ("darwin", "linux"):
            monkeypatch.setattr(sys, "platform", platform)
            assert format(parse("F1")) == "F1"

    def test_macos_spells_editing_keys_as_glyphs(self, monkeypatch):
        """macOS users expect the glyphs; words there look wrong."""
        monkeypatch.setattr(sys, "platform", "darwin")
        assert format(parse("Delete")) == "⌦"
        assert format(parse("Mod+Enter")) == "⌘↩"
        assert format(parse("Mod+Home")) == "⌘↖"
        assert format(parse("Escape")) == "⎋"

    def test_linux_spells_editing_keys_as_words(self, monkeypatch):
        """Off darwin the same keys are words, but arrows stay as arrows."""
        monkeypatch.setattr(sys, "platform", "linux")
        assert format(parse("Delete")) == "Delete"
        assert format(parse("Mod+Enter")) == "Ctrl+Enter"
        assert format(parse("Escape")) == "Escape"
        assert format(parse("Mod+Shift+Up")) == "Ctrl+Shift+↑"

    def test_output_is_never_empty(self):
        """A shortcut hint that renders as nothing is worse than no hint."""
        for spec in _workspace_chord_specs():
            for platform in ("darwin", "linux"):
                assert _format_as(platform, parse(spec)).strip(), (spec, platform)


class TestKeyLabel:
    """Test key_label() naming and its degradation path."""

    def test_labels_canonical_names_off_mac(self, monkeypatch):
        """Named keys come back as their canonical spelling."""
        monkeypatch.setattr(sys, "platform", "linux")
        assert key_label(KEY["A"]) == "A"
        assert key_label(KEY["F1"]) == "F1"
        assert key_label(KEY["Space"]) == "Space"
        assert key_label(KEY["PageUp"]) == "PageUp"

    def test_arrows_are_glyphs_on_every_platform(self, monkeypatch):
        """Arrows read better as glyphs and are safe Latin-adjacent output."""
        for platform in ("darwin", "linux"):
            monkeypatch.setattr(sys, "platform", platform)
            assert key_label(KEY["Up"]) == "↑"
            assert key_label(KEY["Down"]) == "↓"
            assert key_label(KEY["Left"]) == "←"
            assert key_label(KEY["Right"]) == "→"

    def test_unknown_keycode_degrades_instead_of_raising(self):
        """A label is never worth crashing a draw callback over."""
        assert key_label(9999) == "Key9999"

    def test_every_canonical_name_labels_back(self, monkeypatch):
        """The reverse lookup must cover the whole table on the non-glyph platform."""
        monkeypatch.setattr(sys, "platform", "linux")
        for name, code in keys.KEY_NAMES.items():
            label = key_label(code)
            assert label == name or code in (
                KEY["Up"],
                KEY["Down"],
                KEY["Left"],
                KEY["Right"],
            ), name


class TestIsTextSafe:
    """Test which chords may still fire while a text field has focus."""

    def test_bare_letters_are_not_text_safe(self):
        """Typing 'd' in a data cell must not run Toggle Density."""
        assert not is_text_safe(parse("D"))
        assert not is_text_safe(parse("A"))
        assert not is_text_safe(parse("1"))

    def test_mod_chords_are_text_safe(self):
        """No text field produces Mod+K as input, so it stays live."""
        assert is_text_safe(parse("Mod+K"))
        assert is_text_safe(parse("Mod+Shift+P"))
        assert is_text_safe(parse("Mod+1"))

    def test_function_keys_and_escape_are_text_safe(self):
        """F1 and Escape are unambiguous while typing."""
        assert is_text_safe(parse("F1"))
        assert is_text_safe(parse("F2"))
        assert is_text_safe(parse("Escape"))

    def test_shift_and_alt_alone_do_not_confer_safety(self):
        """Shift+A is how you type a capital A; Alt+key is a character on many layouts."""
        assert not is_text_safe(parse("Shift+A"))
        assert not is_text_safe(parse("Alt+Up"))


class TestPressedHeadless:
    """Test pressed() against a real imgui context stepped frame by frame."""

    def test_nothing_fires_on_an_idle_frame(self, kbd):
        """No keys down, no chord fires."""
        kbd.idle()
        assert not pressed(parse("Mod+Z"))
        assert not pressed(parse("Z"))
        assert not pressed(parse("F1"))

    def test_glfw_keycode_reaches_is_key_pressed(self, kbd):
        """The module's core claim: is_key_pressed() accepts a raw GLFW keycode.

        P has no imgui.KEY_P constant, so this can only work via the keys_down index.
        """
        assert not hasattr(imgui, "KEY_P")
        assert kbd.tap("P", down=[KEY["P"]]) == (True,)

    def test_fires_only_on_the_rising_edge(self, kbd):
        """A held key must not re-fire every frame with repeat=False."""
        kbd.idle()
        kbd.step(down=[KEY["P"]])
        assert pressed(parse("P"))
        kbd.step(down=[KEY["P"]])
        assert not pressed(parse("P"))

    def test_mod_matches_ctrl(self, kbd):
        """Mod is Ctrl on non-mac hardware."""
        assert kbd.tap("Mod+Z", down=[KEY["Z"]], ctrl=True) == (True,)

    def test_mod_matches_super(self, kbd):
        """Mod is Cmd on macOS: one binding, both platforms."""
        assert kbd.tap("Mod+Z", down=[KEY["Z"]], super_=True) == (True,)

    def test_mod_chord_does_not_fire_without_mod(self, kbd):
        """A bare Z must not trigger the Mod+Z binding."""
        assert kbd.tap("Mod+Z", down=[KEY["Z"]]) == (False,)

    def test_bare_chord_does_not_fire_with_mod_held(self, kbd):
        """Exact match runs both ways: Ctrl+Z must not fire a bare Z binding."""
        assert kbd.tap("Z", down=[KEY["Z"]], ctrl=True) == (False,)

    def test_wrong_key_does_not_fire(self, kbd):
        """Modifiers matching is not enough; the key must match too."""
        assert kbd.tap("Mod+K", down=[KEY["Z"]], ctrl=True) == (False,)

    def test_undo_does_not_fire_on_redo(self, kbd):
        """THE TRAP: Mod+Z must NOT fire while Shift is held.

        Undo is Mod+Z and redo is Mod+Shift+Z. Without an exact modifier match,
        is_key_pressed(Z) and io.key_ctrl is true for both, so every redo would
        also run an undo and the history would sit still while the user hammers
        the key. This is the single assertion the module exists for.
        """
        undo, redo = kbd.tap("Mod+Z", "Mod+Shift+Z", down=[KEY["Z"]], ctrl=True, shift=True)
        assert not undo, "Mod+Z fired while Shift was held: undo runs on every redo"
        assert redo

    def test_shift_variant_does_not_fire_on_the_plain_chord(self, kbd):
        """The converse: Mod+Shift+Z must not fire on a plain Mod+Z press."""
        undo, redo = kbd.tap("Mod+Z", "Mod+Shift+Z", down=[KEY["Z"]], ctrl=True)
        assert undo
        assert not redo

    def test_alt_is_matched_exactly_too(self, kbd):
        """Alt is not a wildcard either -- Alt+Up must not fire a bare Up binding."""
        bare, alt = kbd.tap("Up", "Alt+Up", down=[KEY["Up"]], alt=True)
        assert not bare
        assert alt

    def test_every_modifier_must_match_at_once(self, kbd):
        """Only the fully-specified chord fires when all three modifiers are held."""
        specs = ("Z", "Mod+Z", "Shift+Z", "Alt+Z", "Mod+Shift+Z", "Mod+Shift+Alt+Z")
        results = kbd.tap(*specs, down=[KEY["Z"]], ctrl=True, shift=True, alt=True)
        assert results == (False, False, False, False, False, True)

    def test_distinct_bindings_do_not_collide(self, kbd):
        """The real Mod+K / Mod+Shift+P palette bindings must stay independent."""
        k, p = kbd.tap("Mod+K", "Mod+Shift+P", down=[KEY["K"]], ctrl=True)
        assert k
        assert not p

    def test_repeat_true_refires_while_held(self, kbd):
        """repeat=True is the opt-in for auto-repeat; the default must not do this."""
        kbd.idle()
        kbd.step(down=[KEY["Down"]])
        assert pressed(parse("Down"), repeat=True)
        kbd.step(down=[KEY["Down"]])
        assert not pressed(parse("Down"))

    def test_function_key_fires(self, kbd):
        """F1 opens Help and has no imgui.KEY_F1 constant to lean on."""
        assert not hasattr(imgui, "KEY_F1")
        assert kbd.tap("F1", down=[KEY["F1"]]) == (True,)

    def test_digit_chord_fires(self, kbd):
        """Mod+1 focuses the Data panel; digits have no imgui.KEY_* constants either."""
        assert kbd.tap("Mod+1", down=[KEY["1"]], ctrl=True) == (True,)

    def test_punctuation_chord_fires(self, kbd):
        """Mod+= / Mod+- are the zoom bindings and live in the punctuation range."""
        zoom_in, zoom_out = kbd.tap("Mod+=", "Mod+-", down=[KEY["="]], ctrl=True)
        assert zoom_in
        assert not zoom_out

    def test_every_workspace_binding_fires_on_its_own_key(self, kbd):
        """End-to-end: each real binding fires when its exact chord is pressed.

        This is the assertion that catches a KEY_NAMES entry that parses but points
        at the wrong keycode -- a table typo that no round-trip test can see.
        """
        for spec in _workspace_chord_specs():
            chord = parse(spec)
            fired = kbd.tap(
                spec,
                down=[chord.key],
                ctrl=chord.mod,
                shift=chord.shift,
                alt=chord.alt,
            )
            assert fired == (True,), spec


class TestPressedWithoutContext:
    """Test that pressed() is safe in a headless process with no imgui context."""

    def test_returns_false_with_no_current_context(self):
        """Callers in a GL-less process need no guard of their own."""
        assert imgui.get_current_context() is None
        assert not pressed(parse("Mod+Z"))

    def test_returns_false_when_imgui_is_missing(self, monkeypatch):
        """The guarded import nulls the name; pressed() must notice, not crash."""
        monkeypatch.setattr(keys, "IMGUI_AVAILABLE", False)
        monkeypatch.setattr(keys, "imgui", None)
        assert not pressed(parse("Mod+Z"))

    def test_returns_false_when_glfw_is_missing(self, monkeypatch):
        """No glfw means no window means no keys; keycodes are still hardcoded."""
        monkeypatch.setattr(keys, "GLFW_AVAILABLE", False)
        assert not pressed(parse("Mod+Z"))

    def test_parse_and_format_work_without_either_library(self, monkeypatch):
        """The keycode table is hardcoded precisely so Help renders headless."""
        monkeypatch.setattr(keys, "IMGUI_AVAILABLE", False)
        monkeypatch.setattr(keys, "imgui", None)
        monkeypatch.setattr(keys, "GLFW_AVAILABLE", False)
        monkeypatch.setattr(keys, "glfw", None)
        assert parse("Mod+Shift+P") == Chord(key=80, mod=True, shift=True)
        assert _format_as("linux", parse("Mod+K")) == "Ctrl+K"
        assert key_label(80) == "P"
