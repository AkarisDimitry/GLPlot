"""Test the fuzzy subsequence matcher in glplot.gui.fuzzy.

Focus on subsequence semantics, the matched-index contract and — the substance —
ranking quality: prefix beats mid-word, exact substring beats scattered, consecutive
runs beat gaps, and results are deterministic. ``fuzzy`` is a pure module, so nothing
here requires OpenGL, a GPU, imgui or a window.
"""

from __future__ import annotations

import itertools
import random

import pytest

from glplot.gui.fuzzy import (
    BONUS_BOUNDARY,
    BONUS_CONSEC,
    BONUS_FIRST,
    MAX_LEADING_PENALTY,
    PENALTY_GAP,
    PENALTY_LEADING,
    SCORE_MATCH,
    score,
)

# Real action titles, lifted from workspace._register_actions. The palette ranks these
# against every keystroke, so they are the only candidates whose ranking actually ships.
REGISTRY_TITLES = [
    "Command Palette",
    "Help",
    "Undo",
    "Redo",
    "Export PNG",
    "Reset Layout",
    "Quit",
    "Open Data",
    "Open Functions",
    "Close Focused Panel",
    "Cycle Panels",
    "Autoscale",
    "Reset View",
    "Next Colormap",
    "Previous Colormap",
    "New Dataset",
    "Paste as New Dataset",
    "Duplicate Dataset",
    "Delete Dataset",
    "Import CSV",
    "Export CSV",
    "Plot Dataset",
    "Add Row",
    "Add Column",
    "Select All Cells",
    "New Function",
    "Plot Function",
    "Integrate",
    "Derivative",
    "Add Noise",
    "Normalize",
    "Resample",
    "Fit Polynomial",
    "Delete Layer",
    "Duplicate Layer",
    "Isolate Layer",
    "Rename Layer",
    "Move Layer Up",
    "Move Layer Down",
    "Select Next Layer",
    "Select Previous Layer",
]


def rank(query, titles=None):
    """Rank ``titles`` best-first the way the palette does: score desc, stable on ties."""
    candidates = REGISTRY_TITLES if titles is None else titles
    scored = [(t, score(query, t)) for t in candidates]
    hits = [(t, s[0]) for t, s in scored if s is not None]
    return sorted(hits, key=lambda pair: -pair[1])


def score_of(query, text):
    """Return just the numeric score, asserting the pair matched at all."""
    result = score(query, text)
    assert result is not None, f"expected {query!r} to match {text!r}"
    return result[0]


class TestSubsequenceSemantics:
    """Test what counts as a match at all, independent of ranking."""

    def test_exact_string_matches_itself(self):
        """A query identical to the text is trivially a subsequence of it."""
        assert score("Integrate", "Integrate") is not None

    def test_contiguous_substring_matches(self):
        """A literal substring is the simplest kind of subsequence."""
        assert score("egra", "Integrate") is not None

    def test_scattered_subsequence_matches(self):
        """Characters may be spread out, as long as they appear in order."""
        assert score("ngt", "Integrate") is not None

    def test_out_of_order_characters_do_not_match(self):
        """Order is part of the contract: the same letters reversed are not a match."""
        assert score("tni", "Integrate") is None

    def test_absent_character_does_not_match(self):
        """One character missing from the text kills the whole match."""
        assert score("intz", "Integrate") is None

    def test_query_longer_than_text_does_not_match(self):
        """A query with more characters than the text cannot be a subsequence."""
        assert score("Integrate", "Int") is None

    def test_repeated_query_char_needs_repeated_text_char(self):
        """Each query character consumes a distinct text position."""
        assert score("aa", "Resample") is None
        assert score("aa", "Add as") is not None

    def test_nonempty_query_never_matches_empty_text(self):
        """There is nothing in an empty string for a real query to match."""
        assert score("a", "") is None

    def test_every_registry_title_matches_its_own_prefix(self):
        """Sanity across the shipping action set: a title always matches its prefix."""
        for title in REGISTRY_TITLES:
            assert score(title[:3], title) is not None


class TestMatchedIndices:
    """Test that returned indices really are the matched positions."""

    def test_indices_select_the_query_characters(self):
        """text[i] for each returned index must reproduce the query, in order."""
        result = score("ngt", "Integrate")
        assert result is not None
        _, indices = result
        assert "".join("Integrate"[i] for i in indices).lower() == "ngt"

    def test_indices_are_ascending(self):
        """A subsequence alignment is monotonic; the palette slices on that assumption."""
        result = score("sl", "Select Layer")
        assert result is not None
        _, indices = result
        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)

    def test_indices_length_equals_query_length(self):
        """One index per query character, no more and no less."""
        result = score("pand", "Paste as New Dataset")
        assert result is not None
        assert len(result[1]) == len("pand")

    def test_indices_index_the_original_not_the_lowered_text(self):
        """Indices are offsets into the caller's string, so they survive mixed case."""
        result = score("pnd", "Paste New Dataset")
        assert result is not None
        _, indices = result
        assert [("Paste New Dataset")[i] for i in indices] == ["P", "N", "D"]

    def test_indices_hold_for_every_registry_title(self):
        """The index contract must hold for every real title the palette highlights."""
        for title in REGISTRY_TITLES:
            for query in ("e", "at", "ls", "ne"):
                result = score(query, title)
                if result is None:
                    continue
                _, indices = result
                assert indices == sorted(indices)
                assert "".join(title[i] for i in indices).lower() == query

    def test_indices_point_at_the_literal_run_when_one_exists(self):
        """When the text literally contains the query, highlight that run, not a scatter."""
        result = score("int", "Integrate")
        assert result is not None
        assert result[1] == [0, 1, 2]


class TestEmptyQuery:
    """Test the list-everything path the palette uses before the user types."""

    def test_empty_query_returns_zero_and_no_indices(self):
        """The documented contract: empty query -> (0, [])."""
        assert score("", "Integrate") == (0, [])

    def test_empty_query_matches_every_registry_title(self):
        """An empty query lists all commands rather than filtering them out."""
        for title in REGISTRY_TITLES:
            assert score("", title) == (0, [])

    def test_empty_query_matches_empty_text(self):
        """Emptiness short-circuits before the empty-text rejection."""
        assert score("", "") == (0, [])

    def test_empty_query_scores_all_titles_equally(self):
        """Uniform zero means the palette's own order (MRU, category) survives."""
        assert {s for _, s in rank("")} == {0}
        assert len(rank("")) == len(REGISTRY_TITLES)


class TestCaseInsensitivity:
    """Test that case never affects matching, and only affects bonuses via boundaries."""

    def test_upper_query_matches_lower_text(self):
        """Matching lowercases both sides."""
        assert score("INT", "integrate") is not None

    def test_lower_query_matches_upper_text(self):
        """The reverse direction must behave identically."""
        assert score("int", "INTEGRATE") is not None

    def test_query_case_does_not_change_the_score(self):
        """Only the text's case carries information (word humps); the query's does not."""
        for variant in ("int", "INT", "Int", "iNt"):
            assert score_of(variant, "Integrate") == score_of("int", "Integrate")

    def test_query_case_does_not_change_the_indices(self):
        """Highlight positions must be stable under query case."""
        assert score("PASTE", "Paste as New Dataset") == score("paste", "Paste as New Dataset")

    def test_mixed_case_query_matches_mixed_case_title(self):
        """The realistic sloppy-typing case."""
        result = score("PlOt DaT", "Plot Dataset")
        assert result is not None
        assert "".join("Plot Dataset"[i] for i in result[1]).lower() == "plot dat"


class TestPrefixBeatsMidWord:
    """Test the headline ranking rule: a prefix hit outranks a mid-word hit."""

    def test_int_ranks_integrate_over_point_size(self):
        """The spec's worked example. Both contain a literal 'int'; the prefix wins."""
        assert score_of("int", "Integrate") > score_of("int", "Point Size")

    def test_int_ranks_integrate_first_in_the_real_registry(self):
        """End-to-end: typing 'int' in the palette must put Integrate at the top."""
        assert rank("int")[0][0] == "Integrate"

    def test_prefix_wins_across_realistic_pairs(self):
        """The rule is not a single tuned coincidence."""
        pairs = [
            ("del", "Delete Layer", "Model Delta"),
            ("norm", "Normalize", "Renormalized"),
            ("res", "Resample", "Preset Restore"),
            ("exp", "Export CSV", "Reexport"),
        ]
        for query, prefixed, midword in pairs:
            assert score_of(query, prefixed) > score_of(query, midword)

    def test_first_char_bonus_exceeds_boundary_bonus(self):
        """The mechanism: index 0 must be worth strictly more than a word boundary."""
        assert BONUS_FIRST > BONUS_BOUNDARY

    def test_word_boundary_beats_interior(self):
        """The weaker sibling rule: after a separator beats buried mid-word."""
        assert score_of("l", "Delete Layer") > score_of("l", "Isolate")

    def test_boundary_detected_after_separators_and_humps(self):
        """Separators, camelCase humps and digits all start a word."""
        for text in ("add noise", "add_noise", "add-noise", "add.noise", "addNoise"):
            assert score_of("n", text) > score_of("n", "canoe")


class TestExactSubstringBeatsScattered:
    """Test that a literal run always outranks the same characters scattered."""

    def test_exact_substring_outranks_scattered_in_longer_text(self):
        """'nor' is literal in Normalize, scattered in the alternative."""
        assert score_of("nor", "Normalize") > score_of("nor", "New Function Or")

    def test_exact_substring_outranks_scatter_within_one_registry_pair(self):
        """'dat' is a run in Dataset, scattered in Duplicate Alt."""
        assert score_of("dat", "Dataset") > score_of("dat", "Duplicate Alt")

    def test_exact_run_wins_when_both_start_at_index_zero(self):
        """Strip the prefix bonus out of the comparison; the run alone must decide."""
        assert score_of("plot", "Plot Dataset") > score_of("plot", "Polynomial Fit Out")

    def test_literal_run_preferred_over_earlier_scattered_alignment(self):
        """The DP's raison d'etre: the greedy first-match alignment is not always best."""
        result = score("bbc", "baddebbc")
        assert result is not None
        assert result[1] == [5, 6, 7]

    def test_dataset_queries_rank_literal_titles_first(self):
        """Typing a whole word must surface the titles containing it."""
        top = [title for title, _ in rank("dataset")[:5]]
        assert "New Dataset" in top
        assert "Delete Dataset" in top


class TestConsecutiveRunsBeatGaps:
    """Test the compounding run bonus and the gap penalties."""

    def test_adjacent_pair_beats_gapped_pair(self):
        """Same two characters, same start: adjacency must win."""
        assert score_of("ab", "abxx") > score_of("ab", "axbx")

    def test_run_bonus_compounds_with_length(self):
        """A 3-run must be worth more than a 2-run plus a detached character."""
        assert score_of("abc", "abcx") > score_of("abc", "abxc")

    def test_longer_gap_costs_more(self):
        """The gap penalty is linear in the number of skipped characters."""
        near = score_of("ab", "axb")
        far = score_of("ab", "axxxb")
        assert near > far
        assert near - far == 2 * PENALTY_GAP

    def test_leading_penalty_makes_an_earlier_match_win(self):
        """Unmatched leading characters cost, so the earlier of two equals wins."""
        assert score_of("z", "az") > score_of("z", "aaaz")

    def test_leading_penalty_is_capped(self):
        """A very late match is penalised, but not driven arbitrarily negative."""
        deep = score_of("z", "a" * 200 + "z")
        assert deep == SCORE_MATCH - MAX_LEADING_PENALTY

    def test_leading_penalty_rate_matches_the_constant(self):
        """Below the cap, each skipped leading character costs PENALTY_LEADING."""
        assert score_of("z", "az") - score_of("z", "aaz") == PENALTY_LEADING

    def test_score_constants_are_positively_signed(self):
        """Bonuses reward and penalties cost; a sign flip would invert every ranking."""
        for bonus in (SCORE_MATCH, BONUS_FIRST, BONUS_BOUNDARY, BONUS_CONSEC):
            assert bonus > 0
        for penalty in (PENALTY_GAP, PENALTY_LEADING):
            assert penalty > 0


class TestDeterminismAndStability:
    """Test that the palette does not reshuffle under the user's cursor."""

    def test_repeated_calls_return_identical_results(self):
        """score() is pure: same inputs, same score and same indices, every time."""
        for query in ("int", "sl", "paste", "e"):
            for title in REGISTRY_TITLES:
                first = score(query, title)
                assert all(score(query, title) == first for _ in range(5))

    def test_ranking_is_reproducible_across_runs(self):
        """The whole ranked list must be byte-identical when recomputed."""
        assert rank("dat") == rank("dat")

    def test_ties_preserve_input_order(self):
        """Equal scores must keep registration order — a stable sort over score alone."""
        titles = ["Add Row", "Add Column"]
        assert score_of("add", titles[0]) == score_of("add", titles[1])
        assert [t for t, _ in rank("add", titles)] == titles
        assert [t for t, _ in rank("add", list(reversed(titles)))] == list(reversed(titles))

    def test_ranking_independent_of_candidate_order(self):
        """Each candidate is scored in isolation; shuffling inputs cannot change scores."""
        shuffled = list(REGISTRY_TITLES)
        random.Random(1234).shuffle(shuffled)
        assert dict(rank("lay", shuffled)) == dict(rank("lay"))

    def test_scores_are_plain_ints(self):
        """The palette sorts and compares these; a float would ruin tie stability."""
        result = score("int", "Integrate")
        assert result is not None
        assert isinstance(result[0], int)
        assert all(isinstance(i, int) for i in result[1])

    def test_matching_a_query_implies_matching_every_prefix_of_it(self):
        """Typing forward only narrows: a candidate cannot drop out and come back."""
        for query in ("dataset", "layer", "polynomial"):
            survivors = [t for t in REGISTRY_TITLES if score(query, t) is not None]
            assert survivors, query
            for length in range(1, len(query)):
                for title in survivors:
                    assert score(query[:length], title) is not None, (query, length, title)


class TestOptimalAlignment:
    """Cross-check the dynamic program against brute-force enumeration."""

    @staticmethod
    def _score_alignment(text, combo):
        """Score one concrete alignment: the independent re-implementation of the rules."""
        boundary = [False] * len(text)
        for i in range(1, len(text)):
            prev, cur = text[i - 1], text[i]
            if prev in " \t.-_/\\:()[]{}+,":
                boundary[i] = True
            elif cur.isupper() and prev.islower():
                boundary[i] = True
            elif cur.isdigit() and not prev.isdigit():
                boundary[i] = True

        def char_bonus(j):
            """Positional value of matching at index j, independent of alignment."""
            if j == 0:
                return SCORE_MATCH + BONUS_FIRST
            return SCORE_MATCH + (BONUS_BOUNDARY if boundary[j] else 0)

        total = char_bonus(combo[0]) - min(combo[0] * PENALTY_LEADING, MAX_LEADING_PENALTY)
        run = 1
        for k in range(1, len(combo)):
            cur, prev = combo[k], combo[k - 1]
            if cur == prev + 1:
                run += 1
                total += BONUS_CONSEC * run + char_bonus(cur)
            else:
                run = 1
                total += char_bonus(cur) - PENALTY_GAP * (cur - prev - 1)
        return total

    @classmethod
    def _brute(cls, query, text):
        """Score every valid alignment exhaustively and return the best score."""
        if not query:
            return 0
        lowered_q, lowered_t = query.lower(), text.lower()
        best = None
        for combo in itertools.combinations(range(len(text)), len(query)):
            if "".join(lowered_t[j] for j in combo) != lowered_q:
                continue
            total = cls._score_alignment(text, combo)
            if best is None or total > best:
                best = total
        return best

    @pytest.mark.parametrize(
        "query,text",
        [
            ("int", "Integrate"),
            ("int", "Point Size"),
            ("sl", "Select Layer"),
            ("bbc", "baddebbc"),
            ("dat", "Paste as New Dataset"),
            ("nl", "Normalize"),
            ("ee", "Delete Layer"),
        ],
    )
    def test_dp_finds_the_optimum_on_known_cases(self, query, text):
        """The DP's score must equal the true maximum over all alignments."""
        assert score_of(query, text) == self._brute(query, text)

    def test_dp_finds_the_optimum_on_random_strings(self):
        """Fuzz the alphabet that provokes repeats, gaps and boundaries."""
        rng = random.Random(20240716)
        for _ in range(400):
            text = "".join(rng.choice("abc de") for _ in range(rng.randint(1, 9)))
            query = "".join(rng.choice("abcd") for _ in range(rng.randint(1, 3)))
            result = score(query, text)
            expected = self._brute(query, text)
            assert (result is None) == (expected is None), (query, text)
            if result is not None:
                assert result[0] == expected, (query, text)
                assert "".join(text[i] for i in result[1]) == query

    def test_reported_indices_are_worth_exactly_the_reported_score(self):
        """The highlighted alignment must be the one that earned the score, re-scored."""
        rng = random.Random(99)
        checked = 0
        for _ in range(200):
            text = "".join(rng.choice("abc de") for _ in range(rng.randint(1, 9)))
            query = "".join(rng.choice("abcd") for _ in range(rng.randint(1, 3)))
            result = score(query, text)
            if result is None:
                continue
            reported, indices = result
            assert self._score_alignment(text, indices) == reported, (query, text)
            checked += 1
        assert checked > 50, "fuzz produced too few matches to be meaningful"

    @pytest.mark.parametrize("query,text", [("int", "Integrate"), ("sl", "Select Layer")])
    def test_registry_indices_are_worth_the_reported_score(self, query, text):
        """The same equality on real titles, where the highlight is user-visible."""
        reported, indices = score(query, text)
        assert self._score_alignment(text, indices) == reported
