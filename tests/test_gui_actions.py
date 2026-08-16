"""Test the action registry in glplot.gui.actions.

Focus on registration, ranked search, chord-conflict detection and keyboard
dispatch. No OpenGL and no GPU: the dispatch tests drive a real imgui context
through the sanctioned headless harness (create_context + new_frame/end_frame),
which needs no GL context, and every action is a real Action object with a real
callable rather than a mock.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from glplot.gui.actions import Action, ActionRegistry
from glplot.gui.keys import Chord
from glplot.gui.keys import parse as parse_chord

try:
    import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - imgui is a hard dependency in CI
    IMGUI_AVAILABLE = False
    imgui = None

# GLFW keycodes, hardcoded exactly as glplot.gui.keys does (frozen GLFW ABI).
KEY_D = 68
KEY_Z = 90
KEY_F1 = 290
KEY_ESCAPE = 256


class _Recorder:
    """A real zero-argument callable that records how often it ran."""

    def __init__(self, log: Optional[List[str]] = None, name: str = "") -> None:
        self.calls = 0
        self._log = log
        self._name = name

    def __call__(self) -> None:
        """Record one invocation, and its order if a shared log was given."""
        self.calls += 1
        if self._log is not None:
            self._log.append(self._name)


def _action(id_: str, title: str, category: str = "Test", **kw: object) -> Action:
    """Build a real Action with a recording `run`, for tests that do not need the callable."""
    return Action(id=id_, title=title, category=category, run=_Recorder(), **kw)  # type: ignore


class TestRegistration:
    """Test ActionRegistry.register/add and the lookup surface."""

    def test_register_returns_the_action_and_assigns_order(self):
        """register hands the action back and stamps registration order onto it."""
        reg = ActionRegistry()
        first = reg.register(_action("a.one", "One"))
        second = reg.register(_action("a.two", "Two"))
        assert first.order == 0
        assert second.order == 1
        assert reg.get("a.one") is first
        assert reg.get("missing") is None

    def test_duplicate_id_raises(self):
        """A second action with an existing id is a bug and must raise, not shadow."""
        reg = ActionRegistry()
        reg.register(_action("data.paste", "Paste as New Dataset", "Data"))
        with pytest.raises(ValueError):
            reg.register(_action("data.paste", "Something Else", "Data"))

    def test_duplicate_id_leaves_the_registry_untouched(self):
        """A rejected duplicate must not appear in all() nor bump the order counter."""
        reg = ActionRegistry()
        reg.register(_action("a.one", "One"))
        with pytest.raises(ValueError):
            reg.register(_action("a.one", "Dup"))
        reg.register(_action("a.two", "Two"))
        assert [a.id for a in reg.all()] == ["a.one", "a.two"]
        assert reg.get("a.two").order == 1

    def test_all_returns_a_copy_in_registration_order(self):
        """all() is a snapshot: mutating it must not corrupt the registry."""
        reg = ActionRegistry()
        reg.register(_action("a.one", "One"))
        reg.register(_action("a.two", "Two"))
        snapshot = reg.all()
        snapshot.clear()
        assert [a.id for a in reg.all()] == ["a.one", "a.two"]

    def test_add_parses_a_chord_spec_string(self):
        """add() accepts chord="Mod+K" so the ~55 registration call sites stay readable."""
        reg = ActionRegistry()
        action = reg.add("global.palette", "Command Palette", "Global", _Recorder(), chord="Mod+K")
        assert action.chord == parse_chord("Mod+K")
        assert action.chord.mod is True
        assert reg.get("global.palette") is action

    def test_add_accepts_a_chord_object_unchanged(self):
        """A Chord passed to add() is stored as-is, not re-parsed."""
        reg = ActionRegistry()
        chord = Chord(key=KEY_F1)
        action = reg.add("global.help", "Help", "Global", _Recorder(), chord=chord)
        assert action.chord is chord

    def test_add_rejects_an_unknown_key_name(self):
        """An unparseable chord spec must raise rather than register a dead binding."""
        reg = ActionRegistry()
        with pytest.raises(ValueError):
            reg.add("bad.chord", "Bad", "Global", _Recorder(), chord="Mod+Nope")

    def test_add_normalises_keywords(self):
        """keywords may be a bare string or any sequence; it is stored as a tuple."""
        reg = ActionRegistry()
        one = reg.add("a.one", "One", "Data", _Recorder(), keywords="clipboard")
        many = reg.add("a.two", "Two", "Data", _Recorder(), keywords=["excel", "tsv"])
        assert one.keywords == ("clipboard",)
        assert many.keywords == ("excel", "tsv")

    def test_add_forwards_the_remaining_action_fields(self):
        """description/icon/is_enabled/is_checked reach the Action untouched."""
        reg = ActionRegistry()
        action = reg.add(
            "view.density",
            "Toggle Density",
            "View",
            _Recorder(),
            description="Switch the density renderer on or off",
            icon="gear",
            is_enabled=lambda: True,
            is_checked=lambda: True,
        )
        assert action.description == "Switch the density renderer on or off"
        assert action.icon == "gear"
        assert reg.enabled(action) is True
        assert reg.checked(action) is True

    def test_categories_are_distinct_and_in_first_registration_order(self):
        """categories() is authored order, not alphabetical: Global heads the Help sheet."""
        reg = ActionRegistry()
        reg.register(_action("g.one", "One", "Global"))
        reg.register(_action("d.one", "Two", "Data"))
        reg.register(_action("g.two", "Three", "Global"))
        reg.register(_action("m.one", "Four", "Math"))
        assert reg.categories() == ["Global", "Data", "Math"]


class TestPredicates:
    """Test the is_enabled / is_checked callbacks."""

    def test_enabled_defaults_to_true(self):
        """is_enabled=None means the action is always invocable."""
        reg = ActionRegistry()
        action = reg.register(_action("a.one", "One"))
        assert reg.enabled(action) is True

    def test_enabled_honors_the_callback_live(self):
        """enabled() re-reads the predicate every call; state changes are picked up."""
        reg = ActionRegistry()
        state = {"ok": False}
        action = reg.register(_action("a.one", "One", is_enabled=lambda: state["ok"]))
        assert reg.enabled(action) is False
        state["ok"] = True
        assert reg.enabled(action) is True

    def test_enabled_coerces_a_truthy_return_to_bool(self):
        """A predicate returning a non-bool truthy value still yields a real bool."""
        reg = ActionRegistry()
        action = reg.register(_action("a.one", "One", is_enabled=lambda: ["non-empty"]))
        assert reg.enabled(action) is True

    def test_enabled_treats_a_raising_predicate_as_disabled(self):
        """A crashing guard must disable the action, not take the 60Hz frame down."""

        def boom() -> bool:
            raise RuntimeError("predicate exploded")

        reg = ActionRegistry()
        action = reg.register(_action("a.one", "One", is_enabled=boom))
        assert reg.enabled(action) is False

    def test_checked_returns_none_for_non_toggles(self):
        """is_checked=None marks a plain command: the palette draws no check."""
        reg = ActionRegistry()
        action = reg.register(_action("a.one", "One"))
        assert reg.checked(action) is None

    def test_checked_honors_the_callback_live(self):
        """checked() reflects the current toggle state on every call."""
        reg = ActionRegistry()
        state = {"on": True}
        action = reg.register(_action("a.one", "One", is_checked=lambda: state["on"]))
        assert reg.checked(action) is True
        state["on"] = False
        assert reg.checked(action) is False

    def test_checked_degrades_to_none_when_the_predicate_raises(self):
        """A crashing toggle predicate draws no check rather than killing the frame."""

        def boom() -> bool:
            raise RuntimeError("predicate exploded")

        reg = ActionRegistry()
        action = reg.register(_action("a.one", "One", is_checked=boom))
        assert reg.checked(action) is None


class TestConflicts:
    """Test chord-conflict detection surfaced in Help > Keyboard."""

    def test_two_enabled_actions_on_one_chord_conflict(self):
        """Only the first can ever fire, so the pair must be reported."""
        reg = ActionRegistry()
        chord = parse_chord("Mod+I")
        first = reg.register(_action("math.integrate", "Integrate", "Math", chord=chord))
        second = reg.register(_action("data.import", "Import CSV", "Data", chord=chord))
        conflicts = reg.conflicts()
        assert len(conflicts) == 1
        found_chord, actions = conflicts[0]
        assert found_chord == chord
        assert actions == [first, second]

    def test_equal_chords_from_separate_parses_conflict(self):
        """Chord is a frozen value type, so two separate parses of one spec collide."""
        reg = ActionRegistry()
        reg.register(_action("a.one", "One", chord=parse_chord("Mod+Shift+P")))
        reg.register(_action("a.two", "Two", chord=parse_chord("mod+shift+p")))
        assert len(reg.conflicts()) == 1

    def test_a_disabled_action_does_not_conflict(self):
        """A disabled action cannot fire, so it is not competing for the chord."""
        reg = ActionRegistry()
        chord = parse_chord("Mod+I")
        reg.register(_action("math.integrate", "Integrate", "Math", chord=chord))
        reg.register(
            _action("data.import", "Import CSV", "Data", chord=chord, is_enabled=lambda: False)
        )
        assert reg.conflicts() == []

    def test_conflicts_reappear_when_the_disabled_action_becomes_enabled(self):
        """conflicts() is evaluated live, not cached at registration."""
        reg = ActionRegistry()
        chord = parse_chord("Mod+I")
        state = {"ok": False}
        reg.register(_action("math.integrate", "Integrate", "Math", chord=chord))
        reg.register(
            _action("data.import", "Import", "Data", chord=chord, is_enabled=lambda: state["ok"])
        )
        assert reg.conflicts() == []
        state["ok"] = True
        assert len(reg.conflicts()) == 1

    def test_distinct_chords_and_unbound_actions_do_not_conflict(self):
        """Exact modifier state is part of the chord: Mod+Z and Mod+Shift+Z are different."""
        reg = ActionRegistry()
        reg.register(_action("edit.undo", "Undo", "Global", chord=parse_chord("Mod+Z")))
        reg.register(_action("edit.redo", "Redo", "Global", chord=parse_chord("Mod+Shift+Z")))
        reg.register(_action("edit.none", "Palette Only", "Global"))
        assert reg.conflicts() == []

    def test_three_actions_on_one_chord_are_reported_together(self):
        """A conflict group lists every enabled claimant, in registration order."""
        reg = ActionRegistry()
        chord = parse_chord("Delete")
        for i in range(3):
            reg.register(_action("a.{0}".format(i), "Action {0}".format(i), chord=chord))
        conflicts = reg.conflicts()
        assert len(conflicts) == 1
        assert [a.id for a in conflicts[0][1]] == ["a.0", "a.1", "a.2"]


class TestSearch:
    """Test ranked search across title, category and keywords."""

    def _registry(self) -> ActionRegistry:
        """A small registry with realistic titles/categories/keywords."""
        reg = ActionRegistry()
        reg.add("math.integrate", "Integrate", "Math", _Recorder(), keywords=("cumulative", "area"))
        reg.add("style.point_size", "Point Size", "Style", _Recorder())
        reg.add(
            "data.paste_as_new",
            "Paste as New Dataset",
            "Data",
            _Recorder(),
            keywords=("clipboard", "excel"),
        )
        reg.add("view.density", "Toggle Density", "View", _Recorder())
        return reg

    def test_search_finds_by_title(self):
        """The obvious case: a title subsequence matches."""
        reg = self._registry()
        ids = [a.id for a, _score, _idx in reg.search("integrate")]
        assert "math.integrate" in ids

    def test_search_finds_by_category(self):
        """Typing a category name surfaces its actions even when no title matches."""
        reg = self._registry()
        results = reg.search("style")
        assert [a.id for a, _s, _i in results] == ["style.point_size"]

    def test_search_finds_by_keyword(self):
        """Hidden keywords are searched: "excel" finds Paste as New Dataset."""
        reg = self._registry()
        results = reg.search("excel")
        assert [a.id for a, _s, _i in results] == ["data.paste_as_new"]

    def test_keyword_only_match_returns_no_title_indices(self):
        """Indices are always title indices; a keyword hit highlights nothing."""
        reg = self._registry()
        action, score, indices = reg.search("excel")[0]
        assert action.id == "data.paste_as_new"
        assert score > 0
        assert indices == []

    def test_title_match_indices_are_the_matched_title_positions(self):
        """The returned indices really index the title characters that matched."""
        reg = self._registry()
        action, _score, indices = reg.search("integ")[0]
        assert action.id == "math.integrate"
        assert "".join(action.title[i] for i in indices).lower() == "integ"

    def test_no_match_returns_empty(self):
        """A query matching no field at all yields no rows."""
        reg = self._registry()
        assert reg.search("zzzqqq") == []

    def test_prefix_beats_mid_word(self):
        """ "int" must rank Integrate above Point Size, the spec's canonical example."""
        reg = self._registry()
        ids = [a.id for a, _s, _i in reg.search("int")]
        assert ids[0] == "math.integrate"
        assert "style.point_size" in ids

    def test_title_hit_outranks_a_category_only_hit(self):
        """The field weights exist so a real title match wins over a category match."""
        reg = ActionRegistry()
        reg.add("data.wrong", "Nothing Relevant", "Data", _Recorder())
        reg.add("misc.data", "Data Editor", "Misc", _Recorder())
        ids = [a.id for a, _s, _i in reg.search("data")]
        assert ids[0] == "misc.data"

    def test_scores_are_descending(self):
        """Results come out ranked best-first."""
        reg = self._registry()
        scores = [s for _a, s, _i in reg.search("ate")]
        assert scores == sorted(scores, reverse=True)

    def test_disabled_actions_rank_last_but_are_still_returned(self):
        """Hiding a disabled action makes the palette feel broken; grey it instead."""
        reg = ActionRegistry()
        reg.add("a.off", "Integrate", "Math", _Recorder(), is_enabled=lambda: False)
        reg.add("a.on", "Integrate Later", "Math", _Recorder())
        ids = [a.id for a, _s, _i in reg.search("integrate")]
        assert ids == ["a.on", "a.off"]

    def test_a_disabled_action_ranks_last_despite_a_better_score(self):
        """The enabled/disabled split dominates the score in the sort key."""
        reg = ActionRegistry()
        reg.add("a.off", "Integrate", "Math", _Recorder(), is_enabled=lambda: False)
        reg.add("a.on", "Definite Integral Of Signal", "Math", _Recorder())
        results = reg.search("integ")
        assert [a.id for a, _s, _i in results] == ["a.on", "a.off"]
        assert results[0][1] < results[1][1]

    def test_empty_query_returns_everything_enabled_first(self):
        """The palette lists all commands before the user types, disabled at the bottom."""
        reg = ActionRegistry()
        reg.add("a.one", "One", "Global", _Recorder())
        reg.add("a.off", "Two", "Global", _Recorder(), is_enabled=lambda: False)
        reg.add("a.three", "Three", "Global", _Recorder())
        results = reg.search("")
        assert [a.id for a, _s, _i in results] == ["a.one", "a.three", "a.off"]
        assert all(score == 0 and idx == [] for _a, score, idx in results)

    def test_ranking_is_stable_across_repeated_calls(self):
        """Ties break on registration order, so the list does not jitter per keystroke."""
        reg = self._registry()
        first = [a.id for a, _s, _i in reg.search("e")]
        second = [a.id for a, _s, _i in reg.search("e")]
        assert first == second


@pytest.mark.skipif(not IMGUI_AVAILABLE, reason="imgui not importable")
class TestDispatch:
    """Test one-action-per-frame keyboard dispatch against a real imgui context."""

    @pytest.fixture
    def io(self):
        """A headless imgui context per the CONTRACT probe harness; no GL required."""
        ctx = imgui.create_context()
        gui_io = imgui.get_io()
        gui_io.display_size = 800, 600
        gui_io.fonts.get_tex_data_as_rgba32()
        gui_io.fonts.texture_id = 1
        gui_io.delta_time = 1.0 / 60.0
        yield gui_io
        imgui.destroy_context(ctx)

    def _frame(self, io, registry=None, keys=(), ctrl=False, shift=False, alt=False, field=False):
        """Run one frame: dispatch first (as workspace does), then optionally a text field.

        `keys` are GLFW keycodes held down for this frame only, so the next frame sees a
        fresh rising edge. Returns the dispatched action id, or None.
        """
        for key in keys:
            io.keys_down[key] = True
        io.key_ctrl = ctrl
        io.key_shift = shift
        io.key_alt = alt
        imgui.new_frame()
        imgui.begin("harness")
        fired = registry.dispatch() if registry is not None else None
        if field:
            imgui.input_text("cell", "abc")
        imgui.end()
        imgui.end_frame()
        for key in keys:
            io.keys_down[key] = False
        io.key_ctrl = io.key_shift = io.key_alt = False
        return fired

    def _focus_field(self, io):
        """Focus a text field and settle until io.want_text_input reports True.

        want_text_input is computed during end_frame, so a field focused on frame N is
        first visible to a top-of-frame reader on frame N+2 (see dispatch's docstring).
        """
        imgui.new_frame()
        imgui.begin("harness")
        imgui.set_keyboard_focus_here()
        imgui.input_text("cell", "abc")
        imgui.end()
        imgui.end_frame()
        for _ in range(4):
            self._frame(io, field=True)
        assert io.want_text_input, "harness failed to focus a text field"

    def test_no_chord_pressed_dispatches_nothing(self, io):
        """An idle frame returns None and runs nothing."""
        reg = ActionRegistry()
        run = _Recorder()
        reg.add("view.density", "Toggle Density", "View", run, chord="D")
        assert self._frame(io, reg) is None
        assert run.calls == 0

    def test_bare_letter_chord_fires_and_returns_its_id(self, io):
        """With no text field focused, D runs its action exactly once and returns its id."""
        reg = ActionRegistry()
        run = _Recorder()
        reg.add("view.density", "Toggle Density", "View", run, chord="D")
        assert self._frame(io, reg, keys=(KEY_D,)) == "view.density"
        assert run.calls == 1

    def test_a_held_key_fires_only_on_the_rising_edge(self, io):
        """is_key_pressed(repeat=False): holding D must not re-fire every frame."""
        reg = ActionRegistry()
        run = _Recorder()
        reg.add("view.density", "Toggle Density", "View", run, chord="D")
        io.keys_down[KEY_D] = True
        try:
            assert self._frame(io, reg) == "view.density"
            assert self._frame(io, reg) is None
        finally:
            io.keys_down[KEY_D] = False
        assert run.calls == 1

    def test_actions_without_a_chord_are_skipped(self, io):
        """A palette-only action has no binding and must never be dispatched."""
        reg = ActionRegistry()
        palette_only = _Recorder()
        bound = _Recorder()
        reg.add("misc.only", "Palette Only", "Misc", palette_only)
        reg.add("view.density", "Toggle Density", "View", bound, chord="D")
        assert self._frame(io, reg, keys=(KEY_D,)) == "view.density"
        assert palette_only.calls == 0
        assert bound.calls == 1

    def test_exactly_one_action_runs_when_two_chords_match(self, io):
        """Two different chords down in one frame: only the first registered runs."""
        log: List[str] = []
        reg = ActionRegistry()
        first = _Recorder(log, "first")
        second = _Recorder(log, "second")
        reg.add("view.density", "Toggle Density", "View", first, chord="D")
        reg.add("edit.undo", "Undo", "Global", second, chord="Z")
        assert self._frame(io, reg, keys=(KEY_D, KEY_Z)) == "view.density"
        assert log == ["first"]
        assert second.calls == 0

    def test_exactly_one_action_runs_when_two_actions_share_a_chord(self, io):
        """A conflicting binding: the earlier registration wins, the later never runs."""
        reg = ActionRegistry()
        first = _Recorder()
        second = _Recorder()
        reg.add("a.first", "First", "Test", first, chord="D")
        reg.add("a.second", "Second", "Test", second, chord="D")
        assert self._frame(io, reg, keys=(KEY_D,)) == "a.first"
        assert first.calls == 1
        assert second.calls == 0
        assert len(reg.conflicts()) == 1

    def test_a_disabled_action_does_not_run_and_yields_its_chord(self, io):
        """dispatch runs the first *enabled* match: a disabled claimant is passed over."""
        reg = ActionRegistry()
        off = _Recorder()
        on = _Recorder()
        reg.add("a.off", "Off", "Test", off, chord="D", is_enabled=lambda: False)
        reg.add("a.on", "On", "Test", on, chord="D")
        assert self._frame(io, reg, keys=(KEY_D,)) == "a.on"
        assert off.calls == 0
        assert on.calls == 1

    def test_a_disabled_sole_binding_dispatches_nothing(self, io):
        """With no enabled claimant the frame dispatches nothing at all."""
        reg = ActionRegistry()
        off = _Recorder()
        reg.add("a.off", "Off", "Test", off, chord="D", is_enabled=lambda: False)
        assert self._frame(io, reg, keys=(KEY_D,)) is None
        assert off.calls == 0

    def test_modifiers_must_match_exactly(self, io):
        """Mod+Z must not fire while Shift is held, or undo runs on every redo."""
        reg = ActionRegistry()
        undo = _Recorder()
        redo = _Recorder()
        reg.add("edit.undo", "Undo", "Global", undo, chord="Mod+Z")
        reg.add("edit.redo", "Redo", "Global", redo, chord="Mod+Shift+Z")
        assert self._frame(io, reg, keys=(KEY_Z,), ctrl=True, shift=True) == "edit.redo"
        assert undo.calls == 0
        assert redo.calls == 1

    def test_a_raising_action_is_logged_and_still_returns_its_id(self, io, caplog):
        """One broken action must not kill the frame; dispatch reports what it ran."""

        def boom() -> None:
            raise RuntimeError("action exploded")

        reg = ActionRegistry()
        later = _Recorder()
        reg.add("a.boom", "Boom", "Test", boom, chord="D")
        reg.add("a.later", "Later", "Test", later, chord="D")
        with caplog.at_level("ERROR"):
            assert self._frame(io, reg, keys=(KEY_D,)) == "a.boom"
        assert later.calls == 0
        assert "a.boom" in caplog.text

    def test_bare_letter_is_suppressed_while_a_text_field_has_focus(self, io):
        """Typing "d" in a cell must not toggle density -- the whole point of the rule."""
        reg = ActionRegistry()
        run = _Recorder()
        reg.add("view.density", "Toggle Density", "View", run, chord="D")
        self._focus_field(io)
        assert self._frame(io, reg, keys=(KEY_D,), field=True) is None
        assert run.calls == 0

    def test_a_mod_chord_still_fires_while_a_text_field_has_focus(self, io):
        """No text field produces Mod+Z as input, so undo stays live while typing."""
        reg = ActionRegistry()
        run = _Recorder()
        reg.add("edit.undo", "Undo", "Global", run, chord="Mod+Z")
        self._focus_field(io)
        assert self._frame(io, reg, keys=(KEY_Z,), ctrl=True, field=True) == "edit.undo"
        assert run.calls == 1

    def test_a_function_key_still_fires_while_a_text_field_has_focus(self, io):
        """F1 opens Help even mid-word: a function key is never text input."""
        reg = ActionRegistry()
        run = _Recorder()
        reg.add("global.help", "Help", "Global", run, chord="F1")
        self._focus_field(io)
        assert self._frame(io, reg, keys=(KEY_F1,), field=True) == "global.help"
        assert run.calls == 1

    def test_escape_still_fires_while_a_text_field_has_focus(self, io):
        """Escape is text-safe: closing the palette from its own input must work."""
        reg = ActionRegistry()
        run = _Recorder()
        reg.add("global.close", "Close", "Global", run, chord="Escape")
        self._focus_field(io)
        assert self._frame(io, reg, keys=(KEY_ESCAPE,), field=True) == "global.close"
        assert run.calls == 1

    def test_shift_letter_is_suppressed_while_a_text_field_has_focus(self, io):
        """Shift+A is how you type a capital A: Shift does not make a chord text-safe."""
        reg = ActionRegistry()
        run = _Recorder()
        reg.add("data.select_all", "Select All", "Data", run, chord="Shift+D")
        self._focus_field(io)
        assert self._frame(io, reg, keys=(KEY_D,), shift=True, field=True) is None
        assert run.calls == 0

    def test_a_suppressed_bare_letter_does_not_block_a_later_mod_chord(self, io):
        """Suppression skips the action, it does not abort the scan of the registry."""
        reg = ActionRegistry()
        bare = _Recorder()
        undo = _Recorder()
        reg.add("view.density", "Toggle Density", "View", bare, chord="D")
        reg.add("edit.undo", "Undo", "Global", undo, chord="Mod+Z")
        self._focus_field(io)
        fired = self._frame(io, reg, keys=(KEY_D, KEY_Z), ctrl=True, field=True)
        assert fired == "edit.undo"
        assert bare.calls == 0
        assert undo.calls == 1

    def test_the_bare_letter_fires_again_once_the_field_loses_focus(self, io):
        """The rule is conditional on focus, not a permanent disable."""
        reg = ActionRegistry()
        run = _Recorder()
        reg.add("view.density", "Toggle Density", "View", run, chord="D")
        self._focus_field(io)
        assert self._frame(io, reg, keys=(KEY_D,), field=True) is None
        for _ in range(4):
            self._frame(io)
        assert not io.want_text_input
        assert self._frame(io, reg, keys=(KEY_D,)) == "view.density"
        assert run.calls == 1
