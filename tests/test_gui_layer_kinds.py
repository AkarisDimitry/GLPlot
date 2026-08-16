"""Test that the Scene panel's "Change type" offers what it can actually deliver.

A type change is delete+recreate, so the *only* way back to the layer you started with is
to pick its original kind out of the same menu. That makes two silent failures equivalent
to data loss, and both had shipped:

* the picker probed ``layerops`` for a registry name that never existed and quietly fell
  back to a hardcoded ``("line", "scatter")``, so five of the seven kinds could not be
  picked -- and a ``bar`` or ``hist`` layer had no way home;
* ``layer_kind`` inferred a kind from ``layer_type`` alone, and the engine draws a field
  (``imshow``, ``pcolormesh``, ``hist2d``, ``contour``) as a scatter of one point per
  cell -- so an image offered to become a line, and came back as bare points.

Both are the same class of bug: an answer that is wrong rather than absent. The menu
greying out is a fine outcome and is asserted as one; a menu that offers a lie is not.
No OpenGL and no GPU.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.engine import GPULinePlot
from glplot.gui import layerops
from glplot.gui.panels.functions import FunctionsPanel
from glplot.gui.panels.mathlab import _LAYER_KINDS as _mathlab_kinds
from glplot.gui.panels.scene import _available_kinds, _can_change_kind


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


@pytest.fixture
def xy():
    x = np.linspace(0.0, 10.0, 12)
    return x, np.sin(x)


def _pyplot_layers(call) -> list:
    """The layers one ``pyplot`` call leaves behind, on a figure of its own."""
    gplt.figure("kind probe")
    call()
    return list(gplt.gcf().scene.layers)


class TestAvailableKinds:
    def test_offers_every_kind_the_registry_implements(self):
        """The picker and the builder must agree, or a kind is unreachable."""
        assert set(_available_kinds()) == set(layerops.KIND_KEYS)

    def test_every_panel_that_picks_a_kind_reads_the_registry(self):
        """Scene, Mathlab and Functions each had their own copy of the pair.

        One registry, no copies: each of these was independently hardcoded to
        ``("line", "scatter")``, so a kind added to ``layerops`` reached exactly none of
        them. Asserting on the panels' own constants is what makes the next kind show up
        everywhere instead of in whichever panel someone remembered.
        """
        assert set(_mathlab_kinds) == set(layerops.KIND_KEYS)
        assert set(_available_kinds()) == set(layerops.KIND_KEYS)
        source = inspect.getsource(FunctionsPanel._draw_actions)
        assert "layerops.KIND_KEYS" in source
        assert '("line", "scatter")' not in source

    def test_offers_the_derived_kinds(self):
        """The five that the fallback used to hide. Named, so a silent drop is loud."""
        assert {"step", "stem", "area", "bar", "hist"}.issubset(set(_available_kinds()))

    def test_every_offered_kind_can_actually_be_built(self, xy):
        """The other half of the contract: offering a kind whose creation raises."""
        plot = GPULinePlot()
        for kind in _available_kinds():
            layer = layerops.add_xy_layer(plot, *xy, kind=kind, label=f"L-{kind}")
            assert layerops.layer_kind(layer) == kind


class TestRoundTrip:
    """Every kind must be reachable *and* returnable -- the user's way back."""

    @pytest.mark.parametrize("kind", layerops.KIND_KEYS)
    def test_line_to_kind_and_back_restores_the_data(self, kind, xy):
        x, y = xy
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, x, y, kind="line", label="rt")

        changed = layerops.replot_layer_xy(
            plot, None, layer, *layerops.layer_source_xy(layer), kind=kind, label="rt"
        )
        source = layerops.layer_source_xy(changed)
        assert source is not None, f"{kind} layer stranded: nothing to rebuild it from"

        back = layerops.replot_layer_xy(plot, None, changed, *source, kind="line", label="rt")
        restored = layerops.layer_source_xy(back)
        assert restored is not None
        assert restored[0] == pytest.approx(x)
        assert restored[1] == pytest.approx(y)

    @pytest.mark.parametrize("kind", layerops.KIND_KEYS)
    def test_the_menu_stays_open_on_the_result(self, kind, xy):
        """A change must never strand a layer in its new type.

        Asserted through the panel's own gate, not a re-implementation of it: the whole
        bug was the gate asking the wrong question. ``bar``/``area``/``hist`` are patches
        with no ``layer_xy`` at all, so asking the geometry greys the menu out on exactly
        the layers a type change has just produced.
        """
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, *xy, kind=kind, label="stay")
        assert _can_change_kind(layer) is True

    @pytest.mark.parametrize("kind", layerops.KIND_KEYS)
    def test_the_original_kind_is_still_on_the_menu(self, kind, xy):
        """The user's actual way home: change away, then find the old kind offered."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, *xy, kind=kind, label="home")

        changed = layerops.replot_layer_xy(
            plot, None, layer, *layerops.layer_source_xy(layer), kind="scatter", label="home"
        )
        assert _can_change_kind(changed), f"a {kind} changed to scatter cannot change again"
        assert kind in _available_kinds()


class TestKindOptions:
    """Every kind's parameters must be reachable from every panel that builds one.

    They existed all along and only the Data panel drew them, so a bar plotted from
    Functions, Mathlab, or a Scene type change was stuck at its default width with no
    control anywhere in sight.
    """

    @pytest.mark.parametrize("kind", layerops.KIND_KEYS)
    def test_a_layers_options_are_readable_back(self, kind, xy):
        """What the Scene inspector edits: the live layer's own options."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, *xy, kind=kind, label="opts")
        assert layerops.layer_kind_options(layer) == layerops.default_kind_options(kind)

    def test_options_survive_the_rebuild_that_applies_them(self, xy):
        """Editing bar width must re-derive the bars, not silently keep the old ones."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, *xy, kind="bar", label="w")
        wide = layerops.replot_layer_xy(
            plot,
            None,
            layer,
            *layerops.layer_source_xy(layer),
            kind="bar",
            label="w",
            options={"baseline": 0.0, "bar_width": 2.0},
        )
        assert layerops.layer_kind_options(wide)["bar_width"] == pytest.approx(2.0)

    def test_bar_width_actually_changes_the_geometry(self, xy):
        """The option has to reach the vertices, or the control is a placebo."""
        plot = GPULinePlot()
        narrow = layerops.add_xy_layer(
            plot, *xy, kind="bar", label="n", options={"baseline": 0.0, "bar_width": 0.1}
        )
        span = lambda l: float(l.vertices[:, 0].max() - l.vertices[:, 0].min())
        thin = span(narrow)
        wide = layerops.replot_layer_xy(
            plot,
            None,
            narrow,
            *layerops.layer_source_xy(narrow),
            kind="bar",
            label="n",
            options={"baseline": 0.0, "bar_width": 3.0},
        )
        assert span(wide) > thin

    def test_line_and_scatter_declare_no_options(self):
        """The editor draws nothing for them, which is why it keys off the dict."""
        assert layerops.default_kind_options("line") == {}
        assert layerops.default_kind_options("scatter") == {}


class TestHeatMap:
    """``hist2d`` is the one heat map two columns express: a density of the rows.

    It is also the first kind whose colours come from the data rather than the caller,
    so it is the first to prove the ``colormapped`` path: a per-cell colour VBO, and the
    scalars retained beside it so the colormap stays re-mappable afterwards.
    """

    @pytest.fixture
    def cloud(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0.0, 1.0, 2000)
        return x, x * 0.6 + rng.normal(0.0, 0.7, 2000)

    def test_matches_the_picture_pyplot_draws(self, cloud):
        """Same numbers in, same picture out -- or the GUI is a second implementation."""
        x, y = cloud
        gplt.figure("ref")
        _counts, _xe, _ye, ref = gplt.hist2d(x, y, bins=24, cmap="magma")

        plot = GPULinePlot()
        gui = layerops.add_xy_layer(
            plot, x, y, kind="hist2d", label="heat", options={"bins": 24, "cmap": "magma"}
        )
        assert gui.pts.shape == ref.pts.shape
        assert np.allclose(np.sort(gui.pts, axis=0), np.sort(ref.pts, axis=0))
        assert np.allclose(np.sort(gui.colors, axis=0), np.sort(ref.colors, axis=0))

    def test_cells_are_coloured_by_density(self, cloud):
        """The whole point: one colour per cell, not one colour for the layer."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, *cloud, kind="hist2d", label="heat")
        assert len(np.unique(layer.colors, axis=0)) > 1

    def test_the_colormap_stays_remappable(self, cloud):
        """Without the retained scalars the picker would have nothing to re-map."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, *cloud, kind="hist2d", label="heat")
        assert layer.metadata.get("cvalues") is not None
        assert layerops.layer_colormap_kind(layer) == "values2d"

    def test_the_cmap_option_reaches_the_colours(self, cloud):
        """Two colormaps must not produce the same VBO, or the control is a placebo."""
        plot = GPULinePlot()
        a = layerops.add_xy_layer(
            plot, *cloud, kind="hist2d", label="a", options={"bins": 16, "cmap": "magma"}
        )
        magma = np.array(a.colors, copy=True)
        b = layerops.replot_layer_xy(
            plot,
            None,
            a,
            *layerops.layer_source_xy(a),
            kind="hist2d",
            label="a",
            options={"bins": 16, "cmap": "viridis"},
        )
        assert not np.allclose(magma, b.colors)

    def test_changing_bins_recolours_as_well_as_regrids(self, cloud):
        """A bin change rescales every count; stale colours on a new grid would lie."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(
            plot, *cloud, kind="hist2d", label="h", options={"bins": 10, "cmap": "magma"}
        )
        rebuilt = layerops.replot_layer_xy(
            plot,
            None,
            layer,
            *layerops.layer_source_xy(layer),
            kind="hist2d",
            label="h",
            options={"bins": 40, "cmap": "magma"},
        )
        assert len(rebuilt.colors) == len(rebuilt.pts)

    def test_empty_cells_are_dropped(self, cloud):
        """A full grid of mostly-zero cells paints the background and hides the data."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(
            plot, *cloud, kind="hist2d", label="h", options={"bins": 30, "cmap": "magma"}
        )
        assert len(layer.pts) < 30 * 30

    def test_survives_a_column_of_nothing_but_nan(self):
        """`histogram2d` cannot bin non-finite rows; empty geometry is 'draw nothing'."""
        plot = GPULinePlot()
        nan = np.full(8, np.nan)
        layer = layerops.add_xy_layer(plot, nan, nan, kind="hist2d", label="empty")
        assert len(layer.pts) == 0


class TestBoxPlot:
    """Tukey's definition, checked against matplotlib's own implementation of it.

    A box plot is the one kind here that is *arithmetic* rather than layout: getting the
    whiskers subtly wrong draws a plausible box that misreports the spread, which no
    amount of looking at it would reveal.
    """

    @pytest.fixture
    def sample(self):
        rng = np.random.default_rng(3)
        y = np.concatenate([rng.normal(10.0, 2.0, 300), [25.0, 26.0, -6.0]])
        return np.linspace(0.0, 4.0, len(y)), y

    def _stats(self, y, whis=1.5):
        from matplotlib import cbook

        return cbook.boxplot_stats(np.asarray(y, dtype=np.float64), whis=whis)[0]

    def test_quartiles_match_matplotlib(self, sample):
        x, y = sample
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, x, y, kind="boxplot", label="b")
        ref = self._stats(y)
        ys = layer.pts[:, 1]
        # float32 geometry against float64 stats: ~7 digits is all it can carry.
        for value in (ref["q1"], ref["med"], ref["q3"]):
            assert np.isclose(ys, value, rtol=1e-6).any(), f"{value} missing from the box"

    def test_whiskers_reach_the_furthest_point_inside_the_fence(self, sample):
        """Not the min/max of the data: the outliers must not stretch them."""
        x, y = sample
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, x, y, kind="boxplot", label="b")
        ref = self._stats(y)
        assert np.isclose(float(layer.pts[:, 1].min()), ref["whislo"], rtol=1e-6)
        assert np.isclose(float(layer.pts[:, 1].max()), ref["whishi"], rtol=1e-6)
        assert float(layer.pts[:, 1].max()) < float(np.max(y))  # 25/26 are fliers

    def test_whis_widens_the_whiskers(self, sample):
        """The option has to reach the arithmetic, or the control is a placebo."""
        x, y = sample
        plot = GPULinePlot()
        tight = layerops.add_xy_layer(
            plot, x, y, kind="boxplot", label="b", options={"whis": 0.5, "box_width": 0.0}
        )
        narrow = float(tight.pts[:, 1].max())
        loose = layerops.replot_layer_xy(
            plot,
            None,
            tight,
            *layerops.layer_source_xy(tight),
            kind="boxplot",
            label="b",
            options={"whis": 3.0, "box_width": 0.0},
        )
        assert float(loose.pts[:, 1].max()) > narrow

    def test_box_width_reaches_the_geometry(self, sample):
        x, y = sample
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(
            plot, x, y, kind="boxplot", label="b", options={"whis": 1.5, "box_width": 2.0}
        )
        span = float(layer.pts[:, 0].max() - layer.pts[:, 0].min())
        assert span == pytest.approx(2.0, rel=1e-5)

    def test_survives_a_column_of_nothing_but_nan(self):
        plot = GPULinePlot()
        nan = np.full(8, np.nan)
        layer = layerops.add_xy_layer(plot, nan, nan, kind="boxplot", label="empty")
        assert len(layer.pts) == 0

    def test_survives_a_single_row(self):
        """Every quartile collapses onto one value; the IQR is zero, not a crash."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, [1.0], [5.0], kind="boxplot", label="one")
        assert np.allclose(layer.pts[:, 1], 5.0)


class TestFieldArtistsHaveNoKind:
    """A field is drawn as a scatter of cells. That is a renderer detail, not a kind."""

    @pytest.mark.parametrize(
        "name, call",
        [
            ("imshow", lambda: gplt.imshow(np.sin(np.outer(np.linspace(0, 3, 16), np.ones(16))))),
            (
                "pcolormesh",
                lambda: gplt.pcolormesh(
                    *np.meshgrid(np.linspace(-2, 2, 8), np.linspace(-2, 2, 8)),
                    np.zeros((8, 8)),
                ),
            ),
            ("hist2d", lambda: gplt.hist2d(np.linspace(0, 1, 40), np.linspace(0, 1, 40))),
        ],
    )
    def test_field_layer_has_no_kind(self, name, call):
        for layer in _pyplot_layers(call):
            assert layerops.layer_kind(layer) is None, (
                f"{name} resolves to a kind, so the menu offers a conversion that "
                "throws the field away and cannot rebuild it"
            )

    def test_imshow_renders_as_a_scatter(self):
        """The premise: this is why layer_type alone cannot answer the question."""
        layers = _pyplot_layers(lambda: gplt.imshow(np.zeros((8, 8))))
        assert [str(getattr(l, "layer_type", "")) for l in layers] == ["scatter"]

    @pytest.mark.parametrize("name", ["axhline", "axvline"])
    def test_guides_have_no_kind(self, name):
        """A guide spans the view; it is not a line through data."""
        for layer in _pyplot_layers(lambda: getattr(gplt, name)(0.5)):
            assert layerops.layer_kind(layer) is None


class TestLineFamilyIsAKind:
    """``gplt.plot_lines`` builds a ``line_family``, and it was in neither list.

    Not in ``KIND_KEYS``, so the picker could not offer it; not in ``UNSUPPORTED_KINDS``
    either, which is the register of what a two-column table genuinely cannot reach. It
    fell between the two -- and the Data panel *does* ingest it (``DataSet.from_layer``
    reads its ``ab`` as columns ``a``/``b``), so it would happily replot the family as one
    of the other kinds with nothing on the menu to put it back. Same class of bug as the
    two in this module's docstring: the third one to ship.

    It belongs in the registry rather than in ``UNSUPPORTED_KINDS`` because it *is* two
    columns -- a slope and an intercept per row -- unlike errorbar (needs a third), imshow
    (needs a grid) or pie (is a composition).
    """

    A = np.array([0.5, -1.0, 2.0, 0.25], dtype=np.float32)
    B = np.array([0.0, 1.0, -0.5, 2.0], dtype=np.float32)

    def _family(self):
        gplt.figure("fam", density=True)
        gplt.plot_lines(self.A, self.B, x_range=(-3.0, 3.0))
        plot = gplt.gcf()
        return plot, plot._primary_line_layer

    def test_plot_lines_layer_has_a_kind(self):
        """The whole bug in one line: this used to be None."""
        _, layer = self._family()
        assert layerops.layer_kind(layer) == "line_family"

    def test_the_menu_is_open_on_it(self):
        _, layer = self._family()
        assert _can_change_kind(layer) is True

    def test_the_picker_offers_it(self):
        assert "line_family" in _available_kinds()

    def test_it_is_not_also_listed_as_unsupported(self):
        """The two lists must not disagree about the same kind."""
        assert not any("line_family" in key for key in layerops.UNSUPPORTED_KINDS)

    def test_options_recover_the_layers_real_x_range(self):
        """Not the registry placeholder: the family is drawn over (-3, 3), not (-1, 1).

        A default here would silently crop the lines on the way back -- same data, wrong
        picture, and nothing on screen to say why.
        """
        _, layer = self._family()
        assert layerops.layer_kind_options(layer) == {"x_lo": -3.0, "x_hi": 3.0}

    def test_round_trip_away_and_back_restores_the_family(self):
        """The user's report: change it, then get it back.

        Fails on the old code at the very first step -- ``layer_source_xy`` returned None,
        so there was nothing to rebuild from and no ``line_family`` to rebuild it as.
        """
        plot, layer = self._family()
        options = layerops.layer_kind_options(layer)
        source = layerops.layer_source_xy(layer)
        assert source is not None, "family stranded: nothing to rebuild it from"

        away = layerops.replot_layer_xy(plot, None, layer, *source, kind="scatter", label="away")
        assert layerops.layer_kind(away) == "scatter"

        back = layerops.replot_layer_xy(
            plot,
            None,
            away,
            *layerops.layer_source_xy(away),
            kind="line_family",
            label="back",
            options=options,
        )
        assert layerops.layer_kind(back) == "line_family"
        assert back.ab[:, 0] == pytest.approx(self.A)
        assert back.ab[:, 1] == pytest.approx(self.B)
        assert tuple(map(float, back.x_range)) == (-3.0, 3.0)

    def test_builder_returns_the_engines_singleton_not_the_last_layer(self):
        """The family is inserted at index 0 and reused, unlike every appending add_*.

        Reading ``layers[-1]`` would tag whatever was plotted last -- so this fails with
        the kind tag landing on the wrong layer entirely.
        """
        plot = GPULinePlot()
        plot.add_scatter(
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.ones((2, 4), dtype=np.float32),
            label="decoy",
        )
        family = layerops.add_xy_layer(
            plot,
            self.A,
            self.B,
            kind="line_family",
            label="fam",
            options={"x_lo": -1.0, "x_hi": 1.0},
        )
        assert family is plot._primary_line_layer
        assert family.label == "fam"
        assert plot.scene.layers[-1].label == "decoy", "the decoy was mislabelled"

    def test_a_degenerate_x_range_is_refused(self):
        """x_hi <= x_lo draws every line as a point; better to say so than to render it."""
        plot = GPULinePlot()
        with pytest.raises(ValueError, match="x_hi > x_lo"):
            layerops.add_xy_layer(
                plot,
                self.A,
                self.B,
                kind="line_family",
                label="bad",
                options={"x_lo": 1.0, "x_hi": 1.0},
            )


class TestPlainArtistsKeepTheirKind:
    """The whitelist must not overshoot: a real line is still a line."""

    def test_plot_is_a_line(self):
        layers = _pyplot_layers(lambda: gplt.plot([0.0, 1.0], [0.0, 1.0]))
        assert layerops.layer_kind(layers[0]) == "line"

    def test_scatter_is_a_scatter(self):
        layers = _pyplot_layers(lambda: gplt.scatter([0.0, 1.0], [0.0, 1.0]))
        assert layerops.layer_kind(layers[0]) == "scatter"

    def test_raw_engine_layer_keeps_the_legacy_inference(self):
        """No artist tag at all: the pre-artist layers the fallback exists for."""
        plot = GPULinePlot()
        x = np.linspace(0.0, 1.0, 8, dtype=np.float32)
        plot.add_line_strip(x, x, color=(1.0, 1.0, 1.0, 1.0), label="raw")
        assert layerops.layer_kind(plot.scene.layers[0]) == "line"

    def test_an_explicit_tag_still_wins(self):
        """The tag is the answer whenever it exists; the artist gate is a fallback."""
        plot = GPULinePlot()
        layer = layerops.add_xy_layer(plot, [0.0, 1.0], [0.0, 1.0], kind="bar", label="tagged")
        layer.metadata["artist"] = "imshow"
        assert layerops.layer_kind(layer) == "bar"
