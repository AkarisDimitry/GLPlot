"""``grouped_bar`` lays groups out the way matplotlib does: bars side by side inside a group,
``group_spacing`` bar-widths of air between groups, and no group overlapping the next.

The width used to be ``group_spacing / (n + ...)``, so with the defaults a two-series group was
1.5 units wide on a 1.0 pitch and neighbouring groups overlapped.
"""

import numpy as np
import pytest

import glplot.pyplot as gplt


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


def _extent(layers):
    """(left, right) of one series' bars."""
    verts = np.vstack([np.asarray(layer.vertices) for layer in layers])
    return float(verts[:, 0].min()), float(verts[:, 0].max())


def _bar_width(series):
    verts = np.asarray(series[0].vertices)
    return float(verts[:, 0].max() - verts[:, 0].min())


@pytest.mark.parametrize("n_series", [1, 2, 3, 5])
def test_groups_on_unit_pitch_do_not_overlap(n_series):
    heights = {f"s{i}": [3.0, 5.0, 4.0] for i in range(n_series)}
    series = gplt.grouped_bar(heights)
    width = _bar_width(series[0])
    # The group is n bars wide and the gap to the next group is group_spacing (1.5) bars.
    assert width == pytest.approx(1.0 / (n_series + 1.5))
    group_width = n_series * width
    assert group_width < 1.0
    left_of_second_group = min(_extent([s[1]])[0] for s in series)
    right_of_first_group = max(_extent([s[0]])[1] for s in series)
    assert left_of_second_group - right_of_first_group == pytest.approx(1.5 * width)


def test_bars_within_a_group_touch_by_default_and_gap_widens_them():
    touching = gplt.grouped_bar({"a": [1.0, 2.0], "b": [1.0, 2.0]})
    a_right = _extent([touching[0][0]])[1]
    b_left = _extent([touching[1][0]])[0]
    assert b_left == pytest.approx(a_right)

    gplt._cleanup_pyplot_state()
    spaced = gplt.grouped_bar({"a": [1.0, 2.0], "b": [1.0, 2.0]}, bar_spacing=0.5)
    width = _bar_width(spaced[0])
    assert _extent([spaced[1][0]])[0] - _extent([spaced[0][0]])[1] == pytest.approx(0.5 * width)


def test_groups_are_centred_on_their_positions():
    series = gplt.grouped_bar({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    for category in range(3):
        left = _extent([series[0][category]])[0]
        right = _extent([series[1][category]])[1]
        assert (left + right) / 2.0 == pytest.approx(float(category))


def test_width_follows_the_pitch_of_custom_positions():
    series = gplt.grouped_bar({"a": [1.0, 2.0], "b": [2.0, 1.0]}, positions=[0.0, 10.0])
    assert _bar_width(series[0]) == pytest.approx(10.0 / (2 + 1.5))


def test_a_single_category_still_gets_a_sane_width():
    series = gplt.grouped_bar({"a": [1.0], "b": [2.0]})
    assert _bar_width(series[0]) == pytest.approx(1.0 / (2 + 1.5))
