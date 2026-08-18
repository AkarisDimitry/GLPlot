"""Test the live legend renderer in glplot.renderers.legend.

Focus on smart grouping, the two show/hide channels, and the drawn overlay's
geometry, without requiring OpenGL or a GPU. The grouping half is pure numpy and
needs no context at all; the drawing half uses the headless imgui harness
(create_context + new_frame), which creates no GL context.
"""

from __future__ import annotations

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.renderers.legend import (
    LEGEND_LOCATIONS,
    MORE_LABEL,
    build_entries,
    collect_legend_items,
    draw_legend,
    group_legend_entries,
    layer_colors,
    resolve_legend_show,
    resolve_max_items,
    split_indexed_label,
)

RED = (1.0, 0.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


@pytest.fixture
def imgui_frame():
    """An imgui context and open frame, per CONTRACT section 2.10. No GL involved."""
    imgui = pytest.importorskip("imgui_bundle").imgui
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = 1280, 800
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    io.delta_time = 1 / 60.0
    imgui.new_frame()
    yield imgui
    imgui.end_frame()
    imgui.destroy_context(ctx)


class TestSplitIndexedLabel:
    """Test the trailing-index detector that indexed-family grouping is built on."""

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("Curve 0", ("Curve", 0)),
            ("Curve 100", ("Curve", 100)),
            ("Curve 007", ("Curve", 7)),
            ("Trial_7", ("Trial", 7)),
            ("Run-12", ("Run", 12)),
            ("C0", ("C", 0)),
        ],
    )
    def test_indexed_labels_split(self, label, expected):
        """A trailing index is found across space, underscore, hyphen or nothing."""
        assert split_indexed_label(label) == expected

    @pytest.mark.parametrize("label", ["Signal", "Data 3.5", "0", "12", "+95 more layers"])
    def test_non_indexed_labels_rejected(self, label):
        """Rejections matter: a false positive silently merges unrelated series.

        A bare number has no stem to name the group after, and "Data 3.5"/"Data 3.6" are
        decimals rather than an indexed family.
        """
        assert split_indexed_label(label) is None


class TestGrouping:
    """Test the three collapses that keep a legend from becoming a wall of rows."""

    def test_indexed_family_collapses_to_one_entry(self):
        """The headline case: 100 curves plotted in a loop become ONE row."""
        items = [(f"Curve {i}", (RED,)) for i in range(100)]
        entries = group_legend_entries(items)
        assert len(entries) == 1
        assert entries[0].label == "Curve (×100)"
        assert entries[0].count == 100

    def test_exact_duplicates_collapse_with_a_count(self):
        """Repeated identical labels fold into one entry carrying the count."""
        entries = group_legend_entries([("Data", (RED,))] * 3 + [("Other", (BLUE,))])
        assert [e.label for e in entries] == ["Data (×3)", "Other"]
        assert entries[0].count == 3

    def test_lone_indexed_label_is_left_alone(self):
        """A single 'Curve 0' must not be rewritten to 'Curve (×1)' -- that is pure loss."""
        entries = group_legend_entries([("Curve 0", (RED,))])
        assert [e.label for e in entries] == ["Curve 0"]

    def test_distinct_stems_do_not_merge(self):
        """Two families keep their own identities."""
        items = [(f"Curve {i}", (RED,)) for i in range(3)]
        items += [(f"Trial {i}", (BLUE,)) for i in range(3)]
        entries = group_legend_entries(items)
        assert [e.label for e in entries] == ["Curve (×3)", "Trial (×3)"]

    def test_max_items_produces_a_more_entry(self):
        """Overflow folds into the shared '+N more layers' wording."""
        entries = group_legend_entries([(f"S{chr(97 + i)}", (RED,)) for i in range(8)], max_items=5)
        assert len(entries) == 6
        assert entries[-1].label == MORE_LABEL.format(n=3)
        assert entries[-1].count == 3

    def test_more_entry_counts_layers_not_rows(self):
        """The hidden count reports layers, so it cannot under-report a folded family."""
        items = [("Solo", (RED,))] + [(f"Curve {i}", (BLUE,)) for i in range(10)]
        entries = group_legend_entries(items, max_items=1)
        assert entries[-1].label == MORE_LABEL.format(n=10)

    def test_grouping_runs_before_truncation(self):
        """100 curves under a limit of 5 fit in one row rather than being guillotined.

        Truncating first would give "Curve 0..4, +95 more" -- the exact wall of rows the
        grouping exists to prevent.
        """
        items = [(f"Curve {i}", (RED,)) for i in range(100)] + [("Noise", (BLUE,))]
        entries = group_legend_entries(items, max_items=5)
        assert [e.label for e in entries] == ["Curve (×100)", "Noise"]

    def test_order_is_first_appearance(self):
        """Rows come out in layer order, which is the order the user plotted in."""
        entries = group_legend_entries([("Zeta", (RED,)), ("Alpha", (BLUE,))])
        assert [e.label for e in entries] == ["Zeta", "Alpha"]

    def test_empty_input_yields_no_entries(self):
        """Nothing labelled means nothing drawn."""
        assert group_legend_entries([]) == []


class TestGroupedColors:
    """Test that a grouped entry never claims a single colour it does not have."""

    def test_uniform_family_gets_one_swatch_color(self):
        """Members that agree on colour get a flat swatch."""
        entries = group_legend_entries([(f"Curve {i}", (RED,)) for i in range(5)])
        assert entries[0].colors == (RED,)

    def test_varied_family_gets_a_gradient_pair(self):
        """Members that disagree get two colours, so the swatch draws as a gradient."""
        items = [(f"Curve {i}", ((i / 4.0, 0.0, 1.0 - i / 4.0, 1.0),)) for i in range(5)]
        entries = group_legend_entries(items)
        assert len(entries[0].colors) == 2
        assert entries[0].colors[0] != entries[0].colors[1]


class TestLayerColors:
    """Test that swatch colours are read from what the layer actually draws with."""

    def test_polyline_uses_style_color(self):
        """gplt.plot assigns a cycle colour to style.color; the swatch must match it."""
        layer = gplt.plot([0, 1], [0, 1], label="L")[0]
        assert layer_colors(layer) == (tuple(float(c) for c in layer.style.color),)

    def test_colormapped_scatter_yields_a_gradient(self):
        """A per-point colour ramp has no single colour, so it reports two."""
        x = np.linspace(0, 1, 200)
        layer = gplt.scatter(x, x, c=x, cmap="viridis", label="Ramp")
        colors = layer_colors(layer)
        assert len(colors) == 2
        assert colors[0] != colors[1]

    def test_uniform_scatter_yields_a_flat_color(self):
        """A single-colour scatter reports exactly that colour."""
        x = np.linspace(0, 1, 50)
        layer = gplt.scatter(x, x, color=(1.0, 0.0, 0.0, 1.0), label="Flat")
        assert layer_colors(layer) == (RED,)

    def test_patch_uses_face_color(self):
        """Patches fill with face_color, so that is what the swatch shows."""
        from glplot.core.layers import PatchLayer

        layer = PatchLayer(label="P")
        layer.style.face_color = BLUE
        assert layer_colors(layer) == (BLUE,)


class TestCollectItems:
    """Test which layers reach the legend at all."""

    def test_empty_label_is_skipped(self):
        """A layer with no label has nothing to say in a legend.

        Note pyplot never produces one: `add_line_strip` stamps an index-based default
        ("Polyline 0"), which `pyplot.legend()` has always returned. This guards layers
        constructed directly, and is why the check is on the empty string rather than on
        some notion of "the user did not name it".
        """
        from glplot.core.layers import PolylineLayer

        anon = PolylineLayer(pts=np.zeros((2, 2), dtype=np.float32), label="")
        named = PolylineLayer(pts=np.zeros((2, 2), dtype=np.float32), label="Named")
        assert [label for label, _ in collect_legend_items([anon, named])] == ["Named"]

    def test_default_labels_group_by_stem(self):
        """Unlabelled plots get "Polyline 0..N" defaults, which fold into one row.

        This is the smart grouping earning its keep on the laziest possible input: a
        loop of bare gplt.plot() calls.
        """
        for _ in range(20):
            gplt.plot([0, 1], [0, 1])
        plot = gplt._get_or_create_plot()
        entries = group_legend_entries(collect_legend_items(plot.scene.layers))
        assert [e.label for e in entries] == ["Polyline (×20)"]

    def test_hidden_layers_are_skipped(self):
        """Naming something the user cannot see is noise."""
        gplt.plot([0, 1], [0, 1], label="Shown")
        hidden = gplt.plot([0, 1], [1, 0], label="Hidden")[0]
        hidden.style.visible = False
        plot = gplt._get_or_create_plot()
        assert [label for label, _ in collect_legend_items(plot.scene.layers)] == ["Shown"]

    def test_system_3d_scaffolding_is_skipped(self):
        """axis3d/floor3d are chrome the engine added, not data the user plotted."""
        layer = gplt.plot([0, 1], [0, 1], label="Chrome")[0]
        layer.metadata["artist"] = "floor3d"
        plot = gplt._get_or_create_plot()
        assert collect_legend_items(plot.scene.layers) == []


class TestPyplotLegend:
    """Test that legend() keeps its published contract while lighting up the window."""

    def test_return_value_is_unchanged_deduped_labels(self):
        """The return value is still the deduped label list -- published API."""
        gplt.plot([0, 1], [0, 1], label="A")
        gplt.plot([0, 1], [1, 0], label="B")
        gplt.plot([0, 1], [1, 1], label="A")
        assert gplt.legend() == ["A", "B"]

    def test_max_items_return_semantics_are_unchanged(self):
        """max_items still truncates the returned list with '+N more layers'."""
        for i in range(8):
            gplt.plot([0, 1], [i, i], label=f"S{chr(97 + i)}")
        labels = gplt.legend(max_items=5)
        assert len(labels) == 6
        assert labels[-1] == "+3 more layers"

    def test_legend_labels_attribute_still_set(self):
        """plot.legend_labels keeps mirroring the return value."""
        gplt.plot([0, 1], [0, 1], label="A")
        labels = gplt.legend()
        assert gplt._get_or_create_plot().legend_labels == labels

    def test_legend_turns_the_live_overlay_on(self):
        """The headline fix: legend() must make the live window draw something."""
        gplt.plot([0, 1], [0, 1], label="A")
        plot = gplt._get_or_create_plot()
        assert resolve_legend_show(plot.options, plot) is False
        gplt.legend()
        assert resolve_legend_show(plot.options, plot) is True

    def test_live_entries_agree_with_returned_labels(self):
        """The drawn rows are the grouping of exactly what legend() returned.

        Same layers, same label set, one grouping function -- so the live window and the
        caller's list cannot describe different plots.
        """
        for i in range(100):
            gplt.plot([0, 1], [i, i], label=f"Curve {i}")
        plot = gplt._get_or_create_plot()
        labels = gplt.legend()
        plot.legend_max_items = None
        drawn = [e.label for e in build_entries(plot)]
        regrouped = [e.label for e in group_legend_entries([(v, (RED,)) for v in labels])]
        assert drawn == regrouped == ["Curve (×100)"]

    def test_legend_max_items_reaches_the_renderer(self):
        """legend(max_items=N) constrains the drawn rows, not just the return value."""
        for i in range(8):
            gplt.plot([0, 1], [i, i], label=f"S{chr(97 + i)}")
        plot = gplt._get_or_create_plot()
        gplt.legend(max_items=2)
        assert resolve_max_items(plot.options, plot) == 2
        assert [e.label for e in build_entries(plot)][-1] == MORE_LABEL.format(n=6)


class TestShowResolution:
    """Test the two independent show/hide channels."""

    def test_default_is_off(self):
        """An unasked-for legend on every existing plot would be a restyling."""
        plot = gplt._get_or_create_plot()
        assert plot.options.legend_show is None
        assert resolve_legend_show(plot.options, plot) is False

    def test_gui_option_can_hide_what_legend_turned_on(self):
        """The tri-state option is why a GUI checkbox can win over legend()."""
        gplt.plot([0, 1], [0, 1], label="A")
        plot = gplt._get_or_create_plot()
        gplt.legend()
        plot.options.legend_show = False
        assert resolve_legend_show(plot.options, plot) is False

    def test_gui_option_can_show_without_legend_call(self):
        """The option alone is enough; it does not depend on the pyplot channel."""
        plot = gplt._get_or_create_plot()
        plot.options.legend_show = True
        assert resolve_legend_show(plot.options, plot) is True


class TestDrawLegend:
    """Test the drawn overlay through the headless imgui harness."""

    def test_draws_nothing_when_off(self, imgui_frame):
        """Off by default means not one vertex."""
        gplt.plot([0, 1], [0, 1], label="A")
        plot = gplt._get_or_create_plot()
        before = len(imgui_frame.get_background_draw_list().vtx_buffer)
        assert draw_legend(plot) is None
        assert len(imgui_frame.get_background_draw_list().vtx_buffer) == before

    def test_draws_geometry_once_enabled(self, imgui_frame):
        """legend() makes real vertices appear on the background draw list."""
        gplt.plot([0, 1], [0, 1], label="A")
        plot = gplt._get_or_create_plot()
        gplt.legend()
        before = len(imgui_frame.get_background_draw_list().vtx_buffer)
        rect = draw_legend(plot)
        assert rect is not None
        assert len(imgui_frame.get_background_draw_list().vtx_buffer) > before

    def test_draws_nothing_when_no_labels(self, imgui_frame):
        """An enabled legend with nothing to say still draws nothing."""
        plot = gplt._get_or_create_plot()
        plot.options.legend_show = True
        assert draw_legend(plot) is None

    @pytest.mark.parametrize("loc", ["upper left", "upper right", "lower left", "lower right"])
    def test_corner_locations_land_in_their_corner(self, imgui_frame, loc):
        """Each named corner puts the box on the right side of both midlines."""
        gplt.plot([0, 1], [0, 1], label="A")
        plot = gplt._get_or_create_plot()
        gplt.legend()
        plot.options.legend_loc = loc
        x0, y0, x1, y1 = draw_legend(plot)
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        assert (cx < plot.width * 0.5) == ("left" in loc)
        assert (cy < plot.height * 0.5) == ("upper" in loc)

    def test_best_avoids_the_data(self, imgui_frame):
        """'best' puts the box where the data is not."""
        gplt.plot([0.0, 0.02], [0.98, 1.0], label="Blob")
        plot = gplt._get_or_create_plot()
        gplt.legend()
        plot.options.legend_loc = "best"
        plot.camera_controller.fit_bounds(0, 1, 0, 1, plot.width, plot.height)
        x0, y0, _, _ = draw_legend(plot)
        # The blob is upper-left, so the legend must not be.
        assert not (x0 < plot.width * 0.5 and y0 < plot.height * 0.5)

    def test_best_does_not_twitch_during_a_pan(self, imgui_frame):
        """A legend that hops corners as you drag is not 'best', whatever it scores."""
        gplt.plot([0.0, 0.02], [0.98, 1.0], label="Blob")
        plot = gplt._get_or_create_plot()
        gplt.legend()
        plot.options.legend_loc = "best"
        plot.camera_controller.fit_bounds(0, 1, 0, 1, plot.width, plot.height)
        first = draw_legend(plot)
        for _ in range(40):
            plot.camera.cx += 1e-4
            assert draw_legend(plot) == first

    def test_best_relocates_when_genuinely_obstructed(self, imgui_frame):
        """Hysteresis must damp noise without freezing the box under the data."""
        gplt.plot([0.90, 1.0], [0.90, 1.0], label="Blob")
        plot = gplt._get_or_create_plot()
        gplt.legend()
        plot.options.legend_loc = "best"
        plot.camera_controller.fit_bounds(0, 1, 0, 1, plot.width, plot.height)
        before = draw_legend(plot)
        for _ in range(40):
            plot.camera.cx += 0.9 / 40.0
            after = draw_legend(plot)
        assert after[0] != before[0]

    def test_font_scale_scales_the_box(self, imgui_frame):
        """legend_font_scale must move real geometry, not just be stored."""
        gplt.plot([0, 1], [0, 1], label="A")
        plot = gplt._get_or_create_plot()
        gplt.legend()
        plot.options.legend_loc = "upper left"
        x0, y0, x1, y1 = draw_legend(plot)
        plot.options.legend_font_scale = 2.0
        bx0, by0, bx1, by1 = draw_legend(plot)
        assert (bx1 - bx0) == pytest.approx((x1 - x0) * 2.0, rel=0.05)
        assert (by1 - by0) == pytest.approx((y1 - y0) * 2.0, rel=0.05)

    def test_font_scale_emits_scaled_glyphs(self, imgui_frame):
        """The ctypes vertex-scaling path must actually widen the glyph run."""
        gplt.plot([0, 1], [0, 1], label="Wide Label Here")
        plot = gplt._get_or_create_plot()
        gplt.legend()

        def glyph_span(scale):
            plot.options.legend_font_scale = scale
            dl = imgui_frame.get_background_draw_list()
            first = len(dl.vtx_buffer)
            draw_legend(plot)
            n = len(dl.vtx_buffer) - first
            xs = [dl.vtx_buffer[i].pos.x for i in range(first, first + n)]
            return max(xs) - min(xs)

        assert glyph_span(2.0) > glyph_span(1.0) * 1.5

    def test_background_and_border_toggles_change_geometry(self, imgui_frame):
        """Both toggles must remove real vertices, or they are lies on a panel."""
        gplt.plot([0, 1], [0, 1], label="A")
        plot = gplt._get_or_create_plot()
        gplt.legend()

        def verts():
            dl = imgui_frame.get_background_draw_list()
            before = len(dl.vtx_buffer)
            draw_legend(plot)
            return len(dl.vtx_buffer) - before

        full = verts()
        plot.options.legend_background = False
        no_bg = verts()
        plot.options.legend_border = False
        neither = verts()
        assert no_bg < full
        assert neither < no_bg

    def test_hidden_layer_drops_its_row(self, imgui_frame):
        """The legend reflects visibility live, in the same frame."""
        gplt.plot([0, 1], [0, 1], label="A")
        b = gplt.plot([0, 1], [1, 0], label="B")[0]
        plot = gplt._get_or_create_plot()
        gplt.legend()
        assert len(build_entries(plot)) == 2
        b.style.visible = False
        assert [e.label for e in build_entries(plot)] == ["A"]

    def test_all_documented_locations_are_accepted(self, imgui_frame):
        """Every value in LEGEND_LOCATIONS must actually draw."""
        gplt.plot([0, 1], [0, 1], label="A")
        plot = gplt._get_or_create_plot()
        gplt.legend()
        for loc in LEGEND_LOCATIONS:
            plot.options.legend_loc = loc
            assert draw_legend(plot) is not None

    def test_unknown_location_falls_back_to_best(self, imgui_frame):
        """A bad option must not break the frame."""
        gplt.plot([0, 1], [0, 1], label="A")
        plot = gplt._get_or_create_plot()
        gplt.legend()
        plot.options.legend_loc = "nonsense"
        assert draw_legend(plot) is not None
