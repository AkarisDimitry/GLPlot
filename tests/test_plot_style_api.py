"""The ``gplt.plot_style`` / ``gplt.plot_styles`` public API.

The style presets began GUI-only (``gui/styles.py`` + the Style panel). This is the code
door onto the same registry, so a preset can be chosen from a script — a gallery example,
a savefig batch — as well as from a click. The behaviour it owns:

* ``plot_styles()`` enumerates the catalogue as plain triples;
* ``plot_style(key_or_name)`` applies one and returns its key;
* ``plot_style()`` reports the last one applied;
* a bad name raises rather than silently doing nothing.

The *effect* of applying a style (background, palette, 3D box) is ``gui/styles.py``'s job
and is tested there against the whole catalogue; this file tests only the API layer.
"""

from __future__ import annotations

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.gui import styles


@pytest.fixture(autouse=True)
def _clean():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


def _line():
    x = np.linspace(0.0, 10.0, 64)
    gplt.plot(x, np.sin(x))


class TestPlotStylesCatalogue:
    def test_it_lists_every_preset_in_registry_order(self):
        listed = gplt.plot_styles()
        assert [k for k, _, _ in listed] == list(styles.STYLE_KEYS)

    def test_each_row_is_key_name_description(self):
        for key, name, description in gplt.plot_styles():
            assert styles.get_style(key).name == name
            assert styles.get_style(key).description == description

    def test_the_new_surface_looks_are_listed(self):
        keys = {k for k, _, _ in gplt.plot_styles()}
        assert {"chalk", "marker", "hand", "kids"} <= keys


class TestPlotStyleApply:
    def test_applying_by_key_returns_the_key(self):
        _line()
        assert gplt.plot_style("marker") == "marker"

    def test_applying_by_display_name_is_case_insensitive(self):
        _line()
        assert gplt.plot_style("Crayon") == "kids"
        assert gplt.plot_style("whiteboard marker") == "marker"

    def test_it_actually_restyles_the_page(self):
        _line()
        gplt.plot_style("kids")
        plot = gplt._get_or_create_plot()
        # Crayon's warm manila page — proof the apply reached the options, not just the key.
        assert tuple(round(c, 2) for c in plot.options.visual.background_color) == (
            0.99,
            0.96,
            0.87,
        )

    def test_no_argument_reports_the_current_style(self):
        _line()
        assert gplt.plot_style() == ""  # nothing applied yet
        gplt.plot_style("chalk")
        assert gplt.plot_style() == "chalk"

    def test_an_unknown_name_raises_and_changes_nothing(self):
        _line()
        gplt.plot_style("marker")
        with pytest.raises(ValueError, match="unknown style"):
            gplt.plot_style("crayons")  # close, but not a key
        assert gplt.plot_style() == "marker"  # the failed call left the previous one

    def test_layers_false_leaves_a_layers_colour_alone(self):
        x = np.linspace(0.0, 10.0, 64)
        line = gplt.plot(x, np.sin(x), color="red")[0]
        before = tuple(float(c) for c in line.style.color)
        gplt.plot_style("kids", layers=False)  # page only
        after = tuple(float(c) for c in line.style.color)
        assert after == before  # the caller's red survived

    def test_layers_true_repaints_a_uniform_line(self):
        x = np.linspace(0.0, 10.0, 64)
        line = gplt.plot(x, np.sin(x), color="red")[0]
        gplt.plot_style("marker")  # layers=True default
        # marker's first palette entry is its black, not the red we started with.
        painted = tuple(round(float(c), 3) for c in line.style.color[:3])
        assert painted == (0.13, 0.14, 0.16)
