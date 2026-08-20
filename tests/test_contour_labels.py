"""Placement/formatting logic for the live inline contour labels.

The drawing itself needs a GL context and an imgui frame, which pytest must not open (see
the project's own note: live-GL in the suite passes alone and fails in combination). Every
decision *before* the draw call is pure, so that is what is pinned here: which run of the
polyline the label lands on, and what the number reads as.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.renderers.contour_labels import (
    _candidate_anchors,
    _format_level,
    _label_anchor,
    _longest_run,
    _overlaps,
)


class _Layer:
    def __init__(self, pts):
        self.pts = np.asarray(pts, dtype=np.float32)


def _identity_projector():
    """Treat world coordinates as window pixels, so anchors are readable by eye."""

    def project(points):
        return np.asarray(points, dtype=np.float64)

    return project


class TestLongestRun:
    """The label goes on the longest *visible* run, so panning never orphans it."""

    def test_picks_the_longer_of_two_runs(self):
        mask = np.array([1, 1, 0, 1, 1, 1, 0], dtype=bool)
        assert _longest_run(mask) == (3, 3)

    def test_run_touching_the_end_is_counted(self):
        mask = np.array([1, 0, 1, 1, 1], dtype=bool)
        assert _longest_run(mask) == (2, 3)

    def test_all_visible(self):
        assert _longest_run(np.ones(4, dtype=bool)) == (0, 4)

    def test_none_visible(self):
        assert _longest_run(np.zeros(4, dtype=bool)) == (0, 0)


class TestLabelAnchor:
    VIEW = (0.0, 0.0, 100.0, 100.0)  # left, top, right, bottom

    def test_anchors_at_the_middle_of_the_visible_run(self):
        pts = [(10.0, 50.0), (20.0, 50.0), (30.0, 50.0)]
        anchor = _label_anchor(_Layer(pts), _identity_projector(), self.VIEW)
        assert anchor == (20.0, 50.0)

    def test_offscreen_vertices_are_excluded(self):
        """The two points outside the view must not drag the label off the line."""
        pts = [(-500.0, 50.0), (40.0, 50.0), (50.0, 50.0), (60.0, 50.0), (900.0, 50.0)]
        anchor = _label_anchor(_Layer(pts), _identity_projector(), self.VIEW)
        assert anchor == (50.0, 50.0)

    def test_fully_offscreen_contour_gets_no_label(self):
        pts = [(-10.0, -10.0), (-20.0, -20.0), (-30.0, -30.0)]
        assert _label_anchor(_Layer(pts), _identity_projector(), self.VIEW) is None

    def test_degenerate_geometry_is_refused(self):
        assert _label_anchor(_Layer([(1.0, 1.0)]), _identity_projector(), self.VIEW) is None

    def test_non_finite_projection_is_not_labelled(self):
        """A log axis maps out-of-domain points to NaN; those are not placements."""
        pts = [(np.nan, np.nan), (np.nan, np.nan)]
        assert _label_anchor(_Layer(pts), _identity_projector(), self.VIEW) is None


class TestCollisionAvoidance:
    """Nested contours all peak at nearly one screen angle, so one anchor is not enough.

    Placing every level at the midpoint of its run stacked a dipole's whole family into a
    single unreadable column; the renderer now walks alternative anchors and drops a label
    outright rather than piling it on a neighbour.
    """

    VIEW = (0.0, 0.0, 100.0, 100.0)

    def test_several_distinct_candidates_are_offered(self):
        pts = [(float(x), 50.0) for x in range(0, 100, 5)]
        candidates = _candidate_anchors(_Layer(pts), _identity_projector(), self.VIEW)
        assert len(candidates) >= 3
        assert len(set(candidates)) >= 3, "candidates must actually differ"

    def test_first_candidate_is_the_documented_midpoint(self):
        pts = [(float(x), 50.0) for x in range(0, 100, 5)]
        args = (_Layer(pts), _identity_projector(), self.VIEW)
        assert _candidate_anchors(*args)[0] == _label_anchor(*args)

    def test_overlapping_rects_are_detected(self):
        assert _overlaps((0, 0, 10, 10), [(5, 5, 15, 15)])

    def test_touching_rects_do_not_count_as_overlapping(self):
        assert not _overlaps((0, 0, 10, 10), [(10, 0, 20, 10)])

    def test_disjoint_rects_are_fine(self):
        assert not _overlaps((0, 0, 10, 10), [(50, 50, 60, 60)])


class TestFormatLevel:
    @pytest.mark.parametrize(
        "fmt,expected",
        [
            (None, "2.5"),
            ("%.2f", "2.50"),
            ("{:.3f}", "2.500"),
        ],
    )
    def test_formats(self, fmt, expected):
        assert _format_level(2.5, fmt) == expected

    def test_callable_format(self):
        assert _format_level(2.5, lambda v: f"<{v:.1f}>") == "<2.5>"

    def test_a_broken_format_falls_back_rather_than_drawing_nothing(self):
        assert _format_level(2.5, "%d %d") == "2.5"

    def test_missing_level_is_empty(self):
        assert _format_level(None, None) == ""
