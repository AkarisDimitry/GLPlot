"""``plt.subplot(r, c, i)`` after ``plt.subplots(r, c)`` selects a panel; no grid rebuild.

``subplot`` already reused a grid *it* had made, but ``subplots`` never recorded its shape, so the
very common ``fig, axs = plt.subplots(1, 2); ...; plt.subplot(1, 2, 2)`` took the shape for a new
one and emptied every panel drawn so far.
"""

import pytest

import glplot.pyplot as gplt


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


def layers_per_panel():
    return [len(panel.scene.layers) for panel in gplt.get_engine().panels]


def test_subplot_after_subplots_keeps_what_was_drawn():
    _, axs = gplt.subplots(1, 2)
    axs[0].plot([1, 2, 3], [1, 4, 9])
    gplt.subplot(1, 2, 2)
    gplt.plot([1, 2, 3], [3, 2, 1])
    assert layers_per_panel() == [1, 1]


def test_reselecting_a_panel_targets_it():
    _, axs = gplt.subplots(2, 2)
    axs[0, 0].plot([0, 1], [0, 1])
    gplt.subplot(2, 2, 4)
    gplt.scatter([0, 1], [1, 0])
    gplt.subplot(2, 2, 1)
    gplt.plot([0, 1], [1, 1])
    assert layers_per_panel() == [2, 0, 0, 1]


def test_the_spacing_given_to_subplots_survives():
    gplt.subplots(1, 2, wspace=0.6)
    before = [tuple(panel.rect_frac) for panel in gplt.get_engine().panels]
    gplt.subplot(1, 2, 2)
    assert [tuple(panel.rect_frac) for panel in gplt.get_engine().panels] == before


def test_a_different_shape_still_rebuilds_the_grid():
    _, axs = gplt.subplots(1, 2)
    axs[0].plot([0, 1], [0, 1])
    gplt.subplot(2, 1, 1)
    assert len(gplt.get_engine().panels) == 2 and layers_per_panel() == [0, 0]
