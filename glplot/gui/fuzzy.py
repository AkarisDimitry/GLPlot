"""Fuzzy subsequence matcher behind the command palette's smart search.

Pure logic: no imgui, no numpy, no I/O. Ranks ~200 candidates per keystroke.

**Why a dynamic program and not a greedy scan.** The obvious implementation walks the
text once, taking the first character that matches. It mis-ranks the moment a query
character occurs twice. Query ``"sl"`` against ``"Select Layer"``: the greedy scan takes
the ``l`` of "Se(l)ect" and reports a cramped mid-word match, never discovering that
skipping it to reach the ``L`` of "Layer" earns the word-boundary bonus and scores far
higher. A DP evaluates both and keeps the better one. The cost is bounded and small (see
the complexity note on :func:`score`), and ranking quality is the entire product value of
a palette.

**The bonus structure**, in the order that decides real rankings:

* ``BONUS_FIRST`` -- matching at index 0. Makes a prefix beat a mid-word hit:
  ``"int"`` scores 148 on "Integrate" but 104 on "Point Size", so *Integrate* wins even
  though both contain a literal, consecutive ``"int"``.
* ``BONUS_BOUNDARY`` -- matching just after a separator (space ``.`` ``_`` ``-`` ``/``)
  or on a lower->upper transition (``camelCase``, ``PascalCase``).
* ``BONUS_CONSEC`` -- **compounding** with run length, so the Nth consecutive character
  is worth N times the bonus. This is what makes an exact substring always outrank a
  scattered subsequence of the same characters: the run grows quadratically while a
  scattered match collects gap penalties instead.
* ``PENALTY_GAP`` / ``PENALTY_LEADING`` -- linear costs for skipped characters, the
  second for characters skipped before the *first* match.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

__all__ = ["score"]

# Tuned so that the invariants in the module docstring hold; tests/test_gui_fuzzy.py and
# the ranking harness pin the behaviour these numbers produce, not the numbers themselves.
SCORE_MATCH = 16  # baseline value of matching one character
BONUS_FIRST = 40  # match at index 0 -- a prefix hit
BONUS_BOUNDARY = 24  # match at a word boundary
BONUS_CONSEC = 12  # per unit of run length; compounds (Nth in a run is worth N * this)
PENALTY_GAP = 3  # per character skipped between two matches
PENALTY_LEADING = 2  # per character skipped before the first match
MAX_LEADING_PENALTY = 20  # cap: a late match should not be driven arbitrarily negative

_BOUNDARY_CHARS = frozenset(" \t.-_/\\:()[]{}+,")

_NEG_INF = float("-inf")


def _boundary_flags(text: str) -> List[bool]:
    """Per-index "starts a word" flags for ``text`` (the original, un-lowered string).

    Index 0 is not flagged: it earns the larger :data:`BONUS_FIRST` instead, and stacking
    both would let a 1-char prefix match outrank a longer, better one.
    """
    flags = [False] * len(text)
    for i in range(1, len(text)):
        prev = text[i - 1]
        cur = text[i]
        if prev in _BOUNDARY_CHARS:
            flags[i] = True
        elif cur.isupper() and prev.islower():
            # camelCase / PascalCase hump
            flags[i] = True
        elif cur.isdigit() and not prev.isdigit():
            flags[i] = True
    return flags


def _prune(cell: Dict[int, Tuple[float, int, int]]) -> Dict[int, Tuple[float, int, int]]:
    """Drop dominated (run, score) entries from one DP cell.

    A longer run is never worse for the future -- the next character's bonus grows with
    it -- so an entry is dominated when some entry with a longer-or-equal run also scores
    at least as well. Real titles rarely produce more than two entries; this exists to
    stop repeated-character input (``"aaaa"`` against ``"aaaaaaaa"``) from carrying a
    frontier of length ``len(query)`` through every cell.
    """
    kept: Dict[int, Tuple[float, int, int]] = {}
    best_score = _NEG_INF
    for run in sorted(cell, reverse=True):
        entry = cell[run]
        if entry[0] > best_score:
            kept[run] = entry
            best_score = entry[0]
    return kept


def _best_per_cell(cells: List[Dict[int, Tuple[float, int, int]]]) -> List[Tuple[float, int]]:
    """Collapse each cell to its best ``(score, run)``, for the gap branch and the finish.

    The gap branch pays a penalty and restarts the run at 1, so the predecessor's own run
    length is irrelevant to it and plain max-by-score is the right choice here -- unlike
    the consecutive branch, which must see every run (see the note in :func:`score`).
    """
    out: List[Tuple[float, int]] = []
    for cell in cells:
        best_score = _NEG_INF
        best_run = 0
        for run, entry in cell.items():
            if entry[0] > best_score:
                best_score = entry[0]
                best_run = run
        out.append((best_score, best_run))
    return out


def score(query: str, text: str) -> Optional[Tuple[int, List[int]]]:
    """Score ``query`` against ``text``, returning ``(score, matched_indices)`` or ``None``.

    The match is a case-insensitive **subsequence**: every character of ``query`` must
    appear in ``text`` in order, but not necessarily adjacently. ``None`` means no such
    subsequence exists.

    The returned indices are the positions in ``text`` of the best-scoring alignment, in
    ascending order and the same length as ``query``. They are not a diagnostic --
    the palette draws the title in segments and accents exactly these characters, and
    that highlight is what makes the search feel like it understands the query. They
    index the **original** ``text``, so they are safe to slice with directly.

    An empty query returns ``(0, [])`` -- it matches everything, which is how the palette
    lists all commands before the user types.

    Complexity is ``O(len(query) * len(text) * f)`` after an ``O(len(text))`` reject scan,
    where ``f`` is the per-cell run frontier -- 1 or 2 for real titles, and bounded by
    ``len(query)`` for pathological repeated-character input. Measured: ~0.5ms to rank 240
    realistic action titles, i.e. the full palette re-ranks well inside one frame. Note
    non-matching candidates, the common case while typing, exit at the reject scan.
    """
    if not query:
        return (0, [])
    if not text:
        return None

    q = query.lower()
    t = text.lower()
    n, m = len(q), len(t)
    if n > m:
        return None

    # Cheap reject: bail before allocating the DP when no subsequence exists at all.
    # This is the common case while the user keeps typing, so it carries the throughput.
    pos = 0
    for ch in q:
        pos = t.find(ch, pos)
        if pos < 0:
            return None
        pos += 1

    boundary = _boundary_flags(text)

    # Per-character positional value, independent of the alignment.
    char_bonus: List[int] = [0] * m
    for j in range(m):
        if j == 0:
            char_bonus[j] = SCORE_MATCH + BONUS_FIRST
        elif boundary[j]:
            char_bonus[j] = SCORE_MATCH + BONUS_BOUNDARY
        else:
            char_bonus[j] = SCORE_MATCH

    # Candidate text positions per query character, so rows skip non-matching columns.
    occurrences: Dict[str, List[int]] = {}
    for j, ch in enumerate(t):
        occurrences.setdefault(ch, []).append(j)

    # DP state, one row per query character. The state is (text position, run length) --
    # NOT text position alone. Keeping only the best-scoring alignment per position is
    # wrong, and subtly so: a lower-scoring alignment that ends in a longer consecutive
    # run can overtake it one character later, because the run bonus compounds. Verified
    # against a brute-force enumerator, which caught exactly this on q="bbc" t="baddebbc"
    # (max-score-per-position scores 97 via b..bc; the true optimum is 98 via the literal
    # "bbc" run). Highlighting a scattered match when the text literally contains the
    # query is the single most visible way a palette can look broken.
    #
    # cells[j] maps run length -> (score, parent_position, parent_run). A run of length r
    # at (i, j) is only feasible when q[i-r+1:i+1] == t[j-r+1:j+1], so in real titles each
    # cell holds one or two entries; _prune keeps the frontier from growing on pathological
    # repeated-character input.
    prev_cells: List[Dict[int, Tuple[float, int, int]]] = [{} for _ in range(m)]
    for j in occurrences.get(q[0], ()):
        prev_cells[j][1] = (
            char_bonus[j] - min(j * PENALTY_LEADING, MAX_LEADING_PENALTY),
            -1,
            0,
        )
    rows: List[List[Dict[int, Tuple[float, int, int]]]] = [prev_cells]
    prev_best = _best_per_cell(prev_cells)

    for i in range(1, n):
        cells: List[Dict[int, Tuple[float, int, int]]] = [{} for _ in range(m)]
        qi = q[i]

        # `gap_val` is G(j) = max over p <= j-2 of (best[p] - PENALTY_GAP * (j-1-p)): the
        # best way to reach j from a *non-adjacent* predecessor. p == j-1 is excluded by
        # construction -- an adjacent predecessor is a consecutive run, handled below --
        # and G is advanced only at the end of each column, so it never sees j-1. Because
        # the penalty is linear in distance, the running maximum satisfies
        #   G(j+1) = max(G(j), best[j-1]) - PENALTY_GAP
        # which keeps each row O(m) instead of O(m^2).
        gap_val = _NEG_INF
        gap_src = -1
        gap_run = 0

        for j in range(1, m):
            # q[i] for i > 0 can never match at j == 0 (q[i-1] would have nowhere to go),
            # so the loop starts at 1 and every best[j - 1] read below is legal.
            if t[j] == qi:
                cell = cells[j]

                # Extend every consecutive run reaching j-1: no gap, compounding bonus.
                for prev_run, (prev_score, _, _) in prev_cells[j - 1].items():
                    r = prev_run + 1
                    total = prev_score + BONUS_CONSEC * r + char_bonus[j]
                    current = cell.get(r)
                    if current is None or total > current[0]:
                        cell[r] = (total, j - 1, prev_run)

                # Jump from the best non-adjacent predecessor, paying the gap. This
                # necessarily starts a fresh run of 1.
                if gap_val > _NEG_INF:
                    total = gap_val + char_bonus[j]
                    current = cell.get(1)
                    if current is None or total > current[0]:
                        cell[1] = (total, gap_src, gap_run)

                if len(cell) > 1:
                    cells[j] = _prune(cell)

            # Advance G(j) -> G(j+1), admitting j-1 as a newly non-adjacent predecessor.
            cand_score, cand_run = prev_best[j - 1]
            if cand_score >= gap_val:
                gap_val = cand_score
                gap_src = j - 1
                gap_run = cand_run
            gap_val -= PENALTY_GAP

        prev_cells = cells
        prev_best = _best_per_cell(cells)
        rows.append(cells)

    # Best terminal (position, run), then walk the parent chain back for the indices.
    end_j = -1
    end_run = 0
    end_score = _NEG_INF
    for j in range(m):
        score_j, run_j = prev_best[j]
        if score_j > end_score:
            end_score = score_j
            end_j = j
            end_run = run_j
    if end_j < 0 or end_score == _NEG_INF:
        return None  # unreachable given the reject scan; never return a bogus match

    indices: List[int] = [0] * n
    j, r = end_j, end_run
    for i in range(n - 1, -1, -1):
        indices[i] = j
        j, r = rows[i][j][r][1], rows[i][j][r][2]

    return (int(end_score), indices)
