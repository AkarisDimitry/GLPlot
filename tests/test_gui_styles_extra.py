"""Test the *whole* style registry as a product: every preset, ticks, and 3D.

``test_gui_styles.py`` pins the mechanism — one apply function, undo, the hand-drawn
transform. This file pins the **catalogue**: that every preset in :data:`glplot.gui.styles.STYLES`,
old or new, is complete, in range, readable, distinct from its neighbours, renderable as a
card in the Style panel, and safe to drop onto a real ``GPULinePlot()``. It is written
against ``STYLE_KEYS`` rather than a list of names on purpose: a preset added later gets
all of this coverage for free, and a preset that cannot pass it does not ship.

Everything here runs headless. ``GPULinePlot()`` constructs without a window and
``apply_style`` never touches GL, so a preset can be applied, undone and inspected with no
GPU in the room.

**Readability is asserted numerically.** A style whose ink is invisible on its own page is
not a look, it is a bug, and "someone will notice in the screenshot" is not a test. The
contrast checks below use the WCAG relative-luminance ratio, which is the only widely
agreed answer to "can a person see this on that".
"""

from __future__ import annotations

import itertools
import math
from dataclasses import fields as dataclass_fields

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.core.camera3d import Axes3DOptions
from glplot.engine import GPULinePlot
from glplot.gui import styles
from glplot.gui.history import UndoStack
from glplot.options import BlendMode, EngineOptions
from glplot.utils.shaders import DENSITY_SCHEMES


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------


def _plot_with_layers() -> GPULinePlot:
    """A windowless 2D plot carrying a polyline pair and a uniform-colour scatter."""
    plot = GPULinePlot()
    x = np.linspace(0.0, 10.0, 128).astype(np.float32)
    plot.add_line_strip(x, np.sin(x), color=(1.0, 0.0, 0.0, 1.0), width=9.0, label="L1")
    plot.add_line_strip(x, np.cos(x), color=(0.0, 1.0, 0.0, 1.0), width=9.0, label="L2")
    plot.add_scatter(x, np.sin(x), np.tile(np.float32([1, 0, 0, 1]), (128, 1)), 20.0, label="S")
    return plot


def _plot_3d() -> GPULinePlot:
    """A windowless 3D plot: one point cloud, plus whatever decoration the engine adds.

    ``add_geometry3d`` calls ``set_3d_view`` itself, so the box/floor/grid/tick layers exist
    from the start — which is what makes "did the preset rebuild them" observable.
    """
    plot = GPULinePlot()
    t = np.linspace(0.0, 6.0 * np.pi, 200)
    verts = np.column_stack([np.cos(t), np.sin(t), t / 6.0]).astype(np.float32)
    plot.add_geometry3d(verts, primitive="points", layer_type="scatter3d", label="helix")
    return plot


def _artists(plot: GPULinePlot) -> set:
    """The ``metadata["artist"]`` tags of the engine's 3D decoration layers."""
    return {
        layer.metadata.get("artist") for layer in plot.scene.layers if layer.metadata.get("artist")
    }


def _relative_luminance(color) -> float:
    """WCAG relative luminance of a linear-ish sRGB triple."""

    def channel(value: float) -> float:
        v = float(value)
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(color[0]) + 0.7152 * channel(color[1]) + 0.0722 * channel(color[2])


def _contrast(a, b) -> float:
    """WCAG contrast ratio, 1.0 (identical) to 21.0 (black on white)."""
    la, lb = _relative_luminance(a), _relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _in_unit_range(values) -> bool:
    return all(0.0 <= float(v) <= 1.0 for v in values)


def _all_styles():
    return [styles.get_style(key) for key in styles.STYLE_KEYS]


# ----------------------------------------------------------------------------------
# Every preset is complete data
# ----------------------------------------------------------------------------------


class TestEveryPresetIsComplete:
    """Walk the whole registry and assert each preset is leaf data a renderer can eat."""

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_identity_is_present_and_card_sized(self, key):
        """The Style panel draws the name and the pitch beside a 132px thumbnail."""
        style = styles.get_style(key)
        assert style.key == key
        assert style.key.replace("_", "").isalnum(), key
        assert style.key == style.key.lower(), key
        assert style.name.strip(), key
        assert len(style.name) <= 32, key
        assert style.description.strip(), key
        # The card gives the pitch one unwrapped line; past ~80 characters it runs off the
        # end of the widest card the panel will draw.
        assert len(style.description) <= 80, key

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_colors_are_in_range(self, key):
        """Out-of-range colour components are undefined behaviour in the imgui packers."""
        style = styles.get_style(key)
        assert len(style.background) == 3 and _in_unit_range(style.background), key
        assert len(style.ink) == 4 and _in_unit_range(style.ink), key
        assert style.palette, key
        for color in style.palette:
            assert len(color) == 4 and _in_unit_range(color), key
        assert len(style.point_outline_color) == 4, key
        assert _in_unit_range(style.point_outline_color), key
        if style.grid_color is not None:
            assert len(style.grid_color) == 3 and _in_unit_range(style.grid_color), key
        if style.gradient is not None:
            top, bottom = style.gradient
            assert _in_unit_range(top) and _in_unit_range(bottom), key

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_geometry_is_inside_the_panel_sliders(self, key):
        """A preset must land somewhere the Layer tab's sliders can still express."""
        style = styles.get_style(key)
        assert 0.1 <= style.line_width <= 20.0, key  # style.py's Line Width slider
        assert 1.0 <= style.point_size <= 100.0, key  # style.py's Point Size slider
        assert 0.1 <= style.point_outline_width <= 5.0, key
        assert 0.0 < style.global_alpha <= 1.0, key
        assert 0.0 <= style.grid_alpha <= 1.0, key

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_scene_and_effect_fields_are_valid(self, key):
        """Every enum/index a preset writes must be one the engine actually implements."""
        style = styles.get_style(key)
        assert isinstance(style.blend_mode, BlendMode), key
        assert isinstance(style.antialiasing, bool), key
        assert 0 <= style.colormap_index < len(DENSITY_SCHEMES), key
        assert style.tonemap in (styles.TONEMAP_NONE, styles.TONEMAP_REINHARD, styles.TONEMAP_ACES)
        # EffectManager.grain_amount clamps to 0.5; a preset past it is asking for
        # something the pipeline will silently refuse.
        assert 0.0 <= style.grain <= 0.5, key
        glow = style.glow
        assert 0.0 <= glow.threshold <= 2.0, key
        assert 0.0 <= glow.intensity <= 4.0, key
        assert 0.0 <= glow.radius_px <= 64.0, key
        assert 0.0 <= glow.knee <= 1.0, key

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_hand_drawn_parameters_are_sane(self, key):
        """Even a preset that never wobbles carries the parameters the panel seeds from."""
        style = styles.get_style(key)
        assert style.hand_amplitude > 0.0, key
        assert style.hand_wavelength > 0.0, key
        assert int(style.hand_seed) >= 0, key

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_tick_settings_are_valid_or_absent(self, key):
        """``None`` means 'no opinion'; anything else must be a drawable tick policy."""
        style = styles.get_style(key)
        if style.ticks is None:
            return
        assert isinstance(style.ticks, styles.TickSettings), key
        assert isinstance(style.ticks.show, bool), key
        assert 0.0 < style.ticks.length_px <= 20.0, key
        assert isinstance(style.ticks.minor, bool), key
        # Below 2 there is no interval to subdivide, and AxisManager would emit nothing.
        assert style.ticks.minor_subdivisions >= 2, key

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_axes3d_settings_are_valid(self, key):
        """The 3D half of a preset is data too, and every field must land somewhere."""
        style = styles.get_style(key)
        axes = style.axes3d
        assert isinstance(axes, styles.Axes3DSettings), key
        for name in ("show_box", "show_floor", "show_grid", "show_ticks"):
            assert isinstance(getattr(axes, name), bool), (key, name)
        assert isinstance(axes.show_tick_labels, bool), key
        assert isinstance(axes.show_axis_labels, bool), key
        # 0 means "leave the count to the per-axis screen-length rule", which is the
        # default and what a preset that has no opinion about tick density should say.
        assert axes.tick_count == 0 or 2 <= axes.tick_count <= 20, key
        assert 0.0 <= axes.pad <= 0.5, key

    def test_axes3d_fields_all_exist_on_the_engine_dataclass(self):
        """The bridge is by name, so a rename in camera3d.py must fail here, not silently."""
        engine_fields = {f.name for f in dataclass_fields(Axes3DOptions)}
        preset_fields = {f.name for f in dataclass_fields(styles.Axes3DSettings)}
        assert set(styles.AXES3D_FIELDS) == preset_fields
        assert preset_fields <= engine_fields

    def test_axes3d_defaults_match_the_engine_defaults(self):
        """A preset that says nothing about 3D must write what the engine would have used.

        Otherwise adding the 3D slot would have silently restyled every existing figure.
        """
        engine = Axes3DOptions()
        preset = styles.Axes3DSettings()
        for name in styles.AXES3D_FIELDS:
            assert getattr(preset, name) == getattr(engine, name), name

    def test_the_axes3d_slot_did_not_change_the_default_preset(self):
        """Clean is the startup look: in 3D it must still be exactly the stock box."""
        assert styles.default_style().axes3d == styles.Axes3DSettings()


# ----------------------------------------------------------------------------------
# Every preset is legible
# ----------------------------------------------------------------------------------


class TestEveryPresetIsReadable:
    """A look whose data disappears into its own page is not a look."""

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_ink_reads_against_the_page(self, key):
        """Labels, titles and text layers are all drawn in ``ink``."""
        style = styles.get_style(key)
        assert _contrast(style.ink, style.background) >= 4.5, key

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_the_first_series_reads_against_the_page(self, key):
        """Most plots have one line, and it is always ``palette[0]``."""
        style = styles.get_style(key)
        assert _contrast(style.palette[0], style.background) >= 4.5, key

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_no_palette_entry_vanishes(self, key):
        """The floor is Hand-drawn's pale yellow on paper, which is the historical limit."""
        style = styles.get_style(key)
        for index, color in enumerate(style.palette):
            assert _contrast(color, style.background) >= 1.5, (key, index)

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_palette_entries_are_distinguishable_from_each_other(self, key):
        """Two series painted the same colour is a plot with a missing series."""
        style = styles.get_style(key)
        for a, b in itertools.combinations(style.palette, 2):
            assert a != b, key
            assert math.dist(a[:3], b[:3]) >= 0.09, (key, a, b)

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_the_grid_never_shouts_over_the_data(self, key):
        """Blueprint's 0.30 is the loudest grid that is still a grid and not a wall."""
        style = styles.get_style(key)
        if style.show_grid:
            assert style.grid_alpha <= 0.35, key

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_light_bg_mode_agrees_with_the_background(self, key):
        """It drives the imgui chrome theme and the 3D decoration ink; a lie here is visible."""
        style = styles.get_style(key)
        lum = _relative_luminance(style.background)
        assert style.light_bg_mode == (lum > 0.18), key


# ----------------------------------------------------------------------------------
# The catalogue as a catalogue
# ----------------------------------------------------------------------------------


class TestTheCatalogueIsUseful:
    """The registry has to cover the jobs a professional actually hands a plot to."""

    def test_every_job_has_a_preset(self):
        """Paper, presentation (light and dark), workstation, engineering, density, access."""
        for key in (
            "journal",  # a manuscript figure
            "print",  # greyscale / photocopier
            "colorsafe",  # colour-vision deficiency
            "slide",  # projector, lights up
            "stage",  # projector, lights down
            "ide",  # a dark workstation
            "blueprint",  # engineering / CAD
            "dense",  # a million points
        ):
            assert styles.get_style(key).key == key

    def test_presets_are_not_colour_permutations(self):
        """Two presets that differ only in hue are one preset and a wasted click."""
        signatures = [
            (
                s.background,
                s.palette,
                round(s.line_width, 3),
                round(s.point_size, 3),
                s.blend_mode,
                s.show_grid,
                s.show_frame,
            )
            for s in _all_styles()
        ]
        assert len(set(signatures)) == len(signatures)
        assert len({s.palette for s in _all_styles()}) == len(styles.STYLE_KEYS)

    def test_the_gallery_is_balanced(self):
        """Enough of both pages that neither kind of user is browsing someone else's list."""
        light = [s for s in _all_styles() if s.light_bg_mode]
        dark = [s for s in _all_styles() if not s.light_bg_mode]
        assert len(light) >= 4
        assert len(dark) >= 4

    def test_effects_stay_opt_in(self):
        """The module's whole thesis: presets are looks, not light. Only Neon glows.

        Grain and the hand-drawn jitter are *surface texture*, not light, so both are shared
        by the small family of looks that imitate a real drawing surface -- grain is the
        tooth of slate, dry-erase board and waxed paper; the jitter is the unsteady hand of
        a sketch and a child's crayon. What stays a strict singleton is glow, which is the
        one genuinely additive effect and the one the module was written to stop overusing.
        """
        assert [s.key for s in _all_styles() if s.glow.enabled] == ["neon"]
        assert [s.key for s in _all_styles() if s.grain > 0.0] == ["chalk", "marker", "kids"]
        assert [s.key for s in _all_styles() if s.hand_drawn] == ["hand", "kids"]

    def test_clean_is_still_first_and_still_the_default(self):
        """The gallery is ordered by medium, and the default has to be the first card."""
        assert styles.STYLE_KEYS[0] == styles.DEFAULT_STYLE_KEY == "clean"


class TestStyleCardInputs:
    """Everything ``StylePanel._style_card`` reads must be there and usable."""

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_thumbnail_has_what_it_needs(self, key):
        """The card paints background/gradient, grid, three palette curves and a frame."""
        style = styles.get_style(key)
        assert len(style.palette) >= 3, key  # the thumbnail draws three curves
        assert max(1.0, float(style.line_width) * 0.8) >= 1.0, key
        grid = style.grid_color or (0.2, 0.2, 0.2)
        assert _in_unit_range(grid), key
        assert isinstance(style.show_frame, bool), key

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_preview_curves_render(self, key):
        """The card runs the real jitter for a hand-drawn preset; it must not fail or NaN."""
        from glplot.gui.panels import style as style_panel

        curves = style_panel._preview_curves(styles.get_style(key))
        assert len(curves) == 3, key
        for curve in curves:
            assert len(curve) >= 2, key
            for px, py in curve:
                assert math.isfinite(px) and math.isfinite(py), key
                # The thumbnail multiplies these by its own 132x44 box, so anything far
                # outside the unit square would be drawn off the card.
                assert -0.2 <= px <= 1.2 and -0.2 <= py <= 1.2, key

    def test_palette_preview_never_runs_out(self):
        """The swatch strip asks for more entries than some palettes have."""
        for style in _all_styles():
            assert len(styles.palette_preview(style, 12)) == 12


# ----------------------------------------------------------------------------------
# Applying every preset to a real plot
# ----------------------------------------------------------------------------------


def _assert_options_are_valid(plot: GPULinePlot, key: str) -> None:
    """Nothing a preset wrote may leave the engine holding a value it cannot render."""
    options = plot.options
    assert isinstance(options.blend_mode, BlendMode), key
    assert isinstance(options.enable_antialiasing, bool), key
    assert 0.0 <= options.default_global_alpha <= 1.0, key
    assert 0.0 <= options.axis_grid_alpha <= 1.0, key
    assert len(tuple(options.axis_grid_color)) == 3, key
    assert _in_unit_range(tuple(options.axis_grid_color)), key
    assert 0 <= options.density_scheme_index < len(DENSITY_SCHEMES), key
    assert _in_unit_range(tuple(options.visual.background_color)[:3]), key
    assert options.axis_tick_len_px >= 0.0, key
    assert options.axis_minor_subdivisions >= 2, key
    glow = options.visual.glow
    assert glow.threshold >= 0.0 and glow.intensity >= 0.0 and glow.radius_px >= 0.0, key
    assert getattr(options.visual, "tonemap_index") in (0, 1, 2), key
    assert 0.0 <= getattr(options.visual, "grain_amount") <= 0.5, key


class TestApplyingEveryPreset:
    """Every preset, on a real windowless plot, end to end."""

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_leaves_the_options_valid(self, key):
        plot = _plot_with_layers()
        assert styles.apply_style(plot, styles.get_style(key)) == 0
        _assert_options_are_valid(plot, key)

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_writes_the_page_it_promised(self, key):
        plot = _plot_with_layers()
        style = styles.get_style(key)
        styles.apply_style(plot, style)

        options = plot.options
        assert tuple(options.visual.background_color) == style.background, key
        assert options.blend_mode is style.blend_mode, key
        assert options.axis_show_grid is style.show_grid, key
        assert options.axis_show_frame is style.show_frame, key
        assert options.axis_grid_alpha == pytest.approx(style.grid_alpha), key
        assert options.density_scheme_index == style.colormap_index, key
        assert options.light_bg_mode is style.light_bg_mode, key
        assert options.visual.gradient_background.enabled == (style.gradient is not None), key

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_paints_the_layers_from_the_palette(self, key):
        plot = _plot_with_layers()
        style = styles.get_style(key)
        styles.apply_style(plot, style)

        assert plot.scene.layers[0].style.color == style.palette[0], key
        assert plot.scene.layers[1].style.color == style.palette[1], key
        assert plot.scene.layers[0].style.line_width == pytest.approx(style.line_width), key
        assert plot.scene.layers[2].style.point_size == pytest.approx(style.point_size), key
        assert plot.scene.layers[2].style.point_outline_enabled is style.point_outline, key

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_survives_the_per_frame_background_pass(self, key):
        """``_apply_background_mode`` runs every frame and used to repaint over presets."""
        plot = _plot_with_layers()
        style = styles.get_style(key)
        styles.apply_style(plot, style)
        plot._apply_background_mode()
        assert tuple(plot.options.visual.background_color) == style.background, key

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_is_idempotent(self, key):
        """Clicking the same card twice must not compound anything."""
        plot = _plot_with_layers()
        style = styles.get_style(key)
        styles.apply_style(plot, style)
        first = (
            tuple(plot.options.visual.background_color),
            plot.scene.layers[0].style.color,
            plot.scene.layers[0].pts.copy(),
        )
        styles.apply_style(plot, style)
        assert tuple(plot.options.visual.background_color) == first[0], key
        assert plot.scene.layers[0].style.color == first[1], key
        assert np.array_equal(plot.scene.layers[0].pts, first[2]), key

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_page_only_mode_keeps_the_users_colours(self, key):
        """The startup path applies every preset with ``layers=False``."""
        plot = _plot_with_layers()
        styles.apply_style(plot, styles.get_style(key), layers=False)
        assert plot.scene.layers[0].style.color == (1.0, 0.0, 0.0, 1.0), key
        assert plot.scene.layers[0].style.line_width == pytest.approx(9.0), key

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_applies_to_an_empty_scene(self, key):
        """A preset is applied before any data exists, every single session."""
        plot = GPULinePlot()
        assert styles.apply_style(plot, styles.get_style(key)) == 0
        _assert_options_are_valid(plot, key)

    def test_switching_between_every_pair_leaves_a_valid_plot(self):
        """The real usage: click through the gallery. No pair may leave a broken scene."""
        plot = _plot_with_layers()
        for key in styles.STYLE_KEYS:
            for other in styles.STYLE_KEYS:
                styles.apply_style(plot, styles.get_style(key))
                styles.apply_style(plot, styles.get_style(other))
                _assert_options_are_valid(plot, f"{key}->{other}")


# ----------------------------------------------------------------------------------
# Ticks
# ----------------------------------------------------------------------------------


class TestTicks:
    """The tick policy: opinionated presets write it, the rest keep their hands off."""

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_an_opinionated_preset_writes_all_four_options(self, key):
        style = styles.get_style(key)
        if style.ticks is None:
            pytest.skip(f"{key} has no tick opinion")
        plot = _plot_with_layers()
        styles.apply_style(plot, style)
        options = plot.options
        assert options.axis_show_ticks is style.ticks.show
        assert options.axis_tick_len_px == pytest.approx(style.ticks.length_px)
        assert options.axis_minor_ticks is style.ticks.minor
        assert options.axis_minor_subdivisions == style.ticks.minor_subdivisions

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_an_abstaining_preset_leaves_the_users_ticks_alone(self, key):
        """``gplt.tick_params(length=...)`` before ``show()`` is a choice, not a default."""
        style = styles.get_style(key)
        if style.ticks is not None:
            pytest.skip(f"{key} owns the ticks")
        plot = _plot_with_layers()
        plot.options.axis_show_ticks = True
        plot.options.axis_tick_len_px = 13.0
        plot.options.axis_minor_ticks = True
        plot.options.axis_minor_subdivisions = 4

        styles.apply_style(plot, style)

        assert plot.options.axis_show_ticks is True
        assert plot.options.axis_tick_len_px == pytest.approx(13.0)
        assert plot.options.axis_minor_ticks is True
        assert plot.options.axis_minor_subdivisions == 4

    def test_the_presets_that_predate_ticks_all_abstain(self):
        """Regression: adding the field must not have changed a single old preset."""
        for key in ("clean", "dark", "chalk", "hand", "neon", "print", "minimal"):
            assert styles.get_style(key).ticks is None, key

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_undo_restores_the_tick_options(self, key):
        plot = _plot_with_layers()
        plot.options.axis_show_ticks = True
        plot.options.axis_tick_len_px = 11.0
        plot.options.axis_minor_ticks = True
        plot.options.axis_minor_subdivisions = 3

        undo = UndoStack()
        undo.push(styles.style_command(plot, styles.get_style(key)))
        undo.undo()

        assert plot.options.axis_show_ticks is True
        assert plot.options.axis_tick_len_px == pytest.approx(11.0)
        assert plot.options.axis_minor_ticks is True
        assert plot.options.axis_minor_subdivisions == 3


# ----------------------------------------------------------------------------------
# 3D
# ----------------------------------------------------------------------------------


class TestAxes3D:
    """A preset reaches the 3D box, and only the box — never the camera."""

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_writes_the_decoration_onto_the_panel(self, key):
        plot = _plot_3d()
        style = styles.get_style(key)
        styles.apply_style(plot, style)
        for name in styles.AXES3D_FIELDS:
            assert getattr(plot.axes3d, name) == getattr(style.axes3d, name), (key, name)

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_rebuilds_the_decoration_layers(self, key):
        """A ``show_floor=False`` nobody rebuilds is a setting that does nothing."""
        plot = _plot_3d()
        style = styles.get_style(key)
        styles.apply_style(plot, style)

        artists = _artists(plot)
        assert ("floor3d" in artists) is style.axes3d.show_floor, key
        assert ("grid3d" in artists) is style.axes3d.show_grid, key
        assert ("ticks3d" in artists) is style.axes3d.show_ticks, key
        assert ("axis3d" in artists) is style.axes3d.show_box, key

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_never_moves_the_camera(self, key):
        """Where the viewer is standing is not a style."""
        plot = _plot_3d()
        plot.set_3d_view(elev=41.0, azim=-17.0, fov=52.0)
        before = (plot.camera3d.elev, plot.camera3d.azim, plot.camera3d.fov)
        styles.apply_style(plot, styles.get_style(key))
        assert (plot.camera3d.elev, plot.camera3d.azim, plot.camera3d.fov) == before, key

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_keeps_the_users_axis_titles(self, key):
        """``gplt.zlabel()`` is data about the plot, not decoration the style owns."""
        plot = _plot_3d()
        plot.axes3d.xlabel, plot.axes3d.ylabel, plot.axes3d.zlabel = "x [m]", "y [m]", "z [m]"
        styles.apply_style(plot, styles.get_style(key))
        assert (plot.axes3d.xlabel, plot.axes3d.ylabel, plot.axes3d.zlabel) == (
            "x [m]",
            "y [m]",
            "z [m]",
        ), key

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_keeps_explicit_axis_limits(self, key):
        """An explicit ``set_zlim`` outranks any look."""
        plot = _plot_3d()
        plot.axes3d.zlim = (-2.0, 7.0)
        styles.apply_style(plot, styles.get_style(key))
        assert plot.axes3d.zlim == (-2.0, 7.0), key

    def test_styles_every_panel_of_a_split_figure(self):
        """The style is a property of the figure; the 3D box is per panel."""
        plot = _plot_3d()
        plot.split_view(2, 2)
        assert len(plot.panels) == 4

        styles.apply_style(plot, styles.get_style("blueprint"))
        for panel in plot.panels:
            assert panel.axes3d.show_floor is False
            assert panel.axes3d.tick_count == 6

    def test_undo_restores_the_decoration(self):
        plot = _plot_3d()
        plot.axes3d.show_floor = True
        plot.axes3d.tick_count = 9
        plot.axes3d.pad = 0.2

        undo = UndoStack()
        undo.push(styles.style_command(plot, styles.get_style("dense")))
        assert plot.axes3d.show_floor is False
        undo.undo()

        assert plot.axes3d.show_floor is True
        assert plot.axes3d.tick_count == 9
        assert plot.axes3d.pad == pytest.approx(0.2)

    def test_redo_reapplies_the_decoration(self):
        plot = _plot_3d()
        undo = UndoStack()
        undo.push(styles.style_command(plot, styles.get_style("minimal")))
        undo.undo()
        undo.redo()
        assert plot.axes3d.show_grid is False
        assert plot.axes3d.show_box is True

    def test_a_2d_figure_grows_no_3d_layers(self):
        """The 3D pass must be inert on the 2D plots that are 95% of the library's use."""
        plot = _plot_with_layers()
        before = len(plot.scene.layers)
        for key in styles.STYLE_KEYS:
            styles.apply_style(plot, styles.get_style(key))
            assert len(plot.scene.layers) == before, key
            assert _artists(plot) == set(), key

    def test_a_stub_plot_without_panels_still_applies(self):
        """``_axes3d_targets`` has to tolerate the hand-rolled plots tests are full of."""

        class _Stub:
            def __init__(self) -> None:
                self.options = EngineOptions()
                self.scene = type("S", (), {"layers": []})()
                self.frame = type("F", (), {"dirty_scene": False, "dirty_ui": False})()
                self.cache = type("C", (), {"refresh_requested": False, "capture_window": "x"})()

        stub = _Stub()
        assert styles.apply_style(stub, styles.get_style("stage")) == 0
        assert stub.options.light_bg_mode is False

    def test_3d_layer_colours_come_from_the_palette(self):
        """A 3D layer with no per-vertex colours is 'a colour', and the style owns it."""
        plot = _plot_3d()
        style = styles.get_style("stage")
        styles.apply_style(plot, style)
        helix = next(layer for layer in plot.scene.layers if getattr(layer, "label", "") == "helix")
        assert tuple(helix.style.color) in style.palette

    def _box_ink(self, plot: GPULinePlot):
        box = next(layer for layer in plot.scene.layers if layer.metadata.get("artist") == "axis3d")
        return tuple(float(v) for v in box.colors[0])

    def test_decoration_ink_follows_the_page_not_the_palette(self):
        """The box is drawn in the engine's own ink, picked from the page's luminance.

        This is the third way a preset reaches 3D (after the decoration flags and the layer
        colours) and it is entirely indirect: ``engine._axis3d_color`` keeps an on-dark and
        an on-light pair and chooses between them from ``visual.background_color``, so
        writing the background and *then* rebuilding is what swaps the box from
        dark-on-white to light-on-black.

        Asserted as a relation, not as literal RGBA: the engine owns those four colours and
        may retune them, but a light page must never get light chrome. And a preset must
        not paint the box from its palette at all — the decoration is chrome, not a series.
        """
        plot = _plot_3d()
        light_style = styles.get_style("journal")  # white page
        dark_style = styles.get_style("stage")  # near-black page

        styles.apply_style(plot, light_style)
        light_ink = self._box_ink(plot)
        styles.apply_style(plot, dark_style)
        dark_ink = self._box_ink(plot)

        assert _relative_luminance(dark_ink) > _relative_luminance(light_ink)
        assert _contrast(light_ink, light_style.background) > 1.5
        assert _contrast(dark_ink, dark_style.background) > 1.5
        assert light_ink not in light_style.palette
        assert dark_ink not in dark_style.palette

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_the_box_is_visible_on_every_preset_page(self, key):
        """Whatever page a preset paints, the 3D box has to be visible against it.

        The regression this pins is real and recent: the decoration ink used to be chosen
        from the ``light_bg_mode`` flag rather than from the colour actually cleared, so any
        disagreement between the two painted a near-white box onto a white page. A preset
        is exactly the thing that can create that disagreement, since it sets the page and
        the mode in the same click.
        """
        plot = _plot_3d()
        style = styles.get_style(key)
        styles.apply_style(plot, style)
        if not style.axes3d.show_box:
            pytest.skip(f"{key} draws no box")
        assert _contrast(self._box_ink(plot), style.background) > 1.5, key
