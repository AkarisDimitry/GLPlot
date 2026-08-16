"""Test the plot-style preset system in glplot.gui.styles.

Presets are pure data applied by one function, so all of this runs against a windowless
``GPULinePlot()`` with no OpenGL and no GPU. The two regression classes at the bottom pin
bugs that were live before this module existed: the engine repainting the background on
every frame, and the composite pass having no grain uniform at all.
"""

from __future__ import annotations

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.engine import GPULinePlot
from glplot.gui import styles
from glplot.gui.history import UndoStack
from glplot.options import BlendMode


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


def _plot_with_layers() -> GPULinePlot:
    """A windowless plot carrying one of every layer type a preset restyles."""
    plot = GPULinePlot()
    x = np.linspace(0.0, 10.0, 128).astype(np.float32)
    plot.add_line_strip(x, np.sin(x), color=(1.0, 0.0, 0.0, 1.0), width=9.0, label="L1")
    plot.add_line_strip(x, np.cos(x), color=(0.0, 1.0, 0.0, 1.0), width=9.0, label="L2")
    plot.add_scatter(x, np.sin(x), np.tile(np.float32([1, 0, 0, 1]), (128, 1)), 20.0, label="S")
    return plot


def _colormapped_scatter(plot: GPULinePlot) -> object:
    """A scatter whose per-point colours carry data, not decoration."""
    x = np.linspace(0.0, 1.0, 64).astype(np.float32)
    colors = np.zeros((64, 4), dtype=np.float32)
    colors[:, 0] = np.linspace(0.0, 1.0, 64)
    colors[:, 3] = 1.0
    plot.add_scatter(x, x, colors, label="CMAP")
    return plot.scene.layers[-1]


class TestStyleRegistry:
    """Test that presets are data, not code."""

    def test_every_promised_preset_ships(self):
        """SPEC_FIX P1-F names seven looks; all seven exist."""
        for key in ("clean", "dark", "chalk", "hand", "neon", "print", "minimal"):
            assert styles.get_style(key).key == key

    def test_the_four_drawing_surface_looks_ship(self):
        """Chalkboard, whiteboard, pencil-on-paper and crayon-on-manila are a family."""
        for key in ("chalk", "marker", "hand", "kids"):
            assert styles.get_style(key).key == key

    def test_whiteboard_marker_is_a_bold_clean_light_board(self):
        """Marker is the *bold* board look: near-white page, thick strokes, no grid, and —
        unlike Chalk and Crayon — no wobble, because a marker sketch is drawn steadily."""
        marker = styles.get_style("marker")
        assert min(marker.background) > 0.9  # a light board
        assert marker.light_bg_mode is True
        assert marker.line_width >= 3.0  # a fat nib
        assert marker.show_grid is False and marker.show_frame is False  # a blank board
        assert marker.hand_drawn is False  # bold and clean, not shaky
        assert marker.grain > 0.0  # a faint dry-erase tooth

    def test_crayon_is_a_coarser_wobblier_hand_drawn(self):
        """Kids is Hand-drawn turned up: it wobbles too, but more, on fatter strokes, and it
        must not wobble *identically* (a shared seed would draw the two looks the same)."""
        kids = styles.get_style("kids")
        hand = styles.get_style("hand")
        assert kids.hand_drawn is True
        assert kids.hand_amplitude > hand.hand_amplitude  # a less steady hand
        assert kids.hand_wavelength > hand.hand_wavelength  # a slower sway
        assert kids.hand_seed != hand.hand_seed  # not the same wobble
        assert kids.line_width > hand.line_width  # fatter crayon
        assert kids.point_size > hand.point_size  # bigger dots

    def test_keys_are_unique(self):
        """A duplicate key would shadow a preset in the registry and in the UI."""
        assert len(styles.STYLE_KEYS) == len(set(styles.STYLE_KEYS))

    def test_clean_is_the_default(self):
        """The user asked for 'plots mas limpios' as the default look."""
        assert styles.DEFAULT_STYLE_KEY == "clean"
        assert styles.default_style().background == (1.0, 1.0, 1.0)

    def test_presets_are_immutable_data(self):
        """Frozen: a preset that a panel could mutate is not a preset."""
        with pytest.raises(Exception):
            styles.default_style().line_width = 5.0

    def test_every_preset_has_a_palette_and_a_name(self):
        """The panel cycles the palette by layer index; an empty one would ZeroDivide."""
        for style in styles.STYLES:
            assert style.palette, style.key
            assert style.name and style.description, style.key
            for color in style.palette:
                assert len(color) == 4, style.key

    def test_unknown_key_raises(self):
        """Never guess: a typo must not silently return the default."""
        with pytest.raises(KeyError):
            styles.get_style("chartreuse")

    def test_only_neon_glows(self):
        """The whole point of the round: glow is opt-in, not the house style."""
        glowing = [s.key for s in styles.STYLES if s.glow.enabled]
        assert glowing == ["neon"]


class TestApplyStyle:
    """Test the single apply function."""

    def test_applies_page_and_layers(self):
        """One call restyles background, palette, line width and point size."""
        plot = _plot_with_layers()
        styles.apply_style(plot, styles.get_style("clean"))

        clean = styles.get_style("clean")
        assert tuple(plot.options.visual.background_color) == clean.background
        assert plot.scene.layers[0].style.color == clean.palette[0]
        assert plot.scene.layers[1].style.color == clean.palette[1]
        assert plot.scene.layers[0].style.line_width == clean.line_width
        assert plot.scene.layers[2].style.point_size == clean.point_size

    def test_layers_false_leaves_layers_alone(self):
        """The startup default must not overwrite a colour the user passed to plot()."""
        plot = _plot_with_layers()
        styles.apply_style(plot, styles.get_style("dark"), layers=False)

        assert plot.scene.layers[0].style.color == (1.0, 0.0, 0.0, 1.0)
        assert plot.scene.layers[0].style.line_width == 9.0
        assert tuple(plot.options.visual.background_color) == styles.get_style("dark").background

    def test_uniform_scatter_colors_are_repainted(self):
        """A single-colour scatter is 'a colour', and the style owns it."""
        plot = _plot_with_layers()
        styles.apply_style(plot, styles.get_style("dark"))
        expected = styles.get_style("dark").palette[2]
        assert np.allclose(plot.scene.layers[2].colors[0], expected)
        assert plot.scene.layers[2].dirty.gpu_dirty

    def test_colormapped_scatter_is_never_flattened(self):
        """Per-point colours carry information; a preset changes the mood, not the data."""
        plot = _plot_with_layers()
        layer = _colormapped_scatter(plot)
        before = layer.colors.copy()
        styles.apply_style(plot, styles.get_style("neon"))
        assert np.array_equal(layer.colors, before)

    def test_sets_the_dirty_flag_incantation(self):
        """CONTRACT 1.4 — without this the restyle waits for an unrelated event."""
        plot = _plot_with_layers()
        plot.frame.dirty_scene = False
        plot.cache.refresh_requested = False
        plot.cache.capture_window = "stale"

        styles.apply_style(plot, styles.get_style("chalk"))

        assert plot.frame.dirty_scene is True
        assert plot.cache.refresh_requested is True
        assert plot.cache.capture_window is None

    def test_drives_the_optional_post_fields(self):
        """tonemap/knee/grain are read by effects.py via getattr; a preset must set them."""
        plot = _plot_with_layers()
        styles.apply_style(plot, styles.get_style("chalk"))
        assert plot.options.visual.grain_amount == pytest.approx(0.055)
        assert plot.options.visual.tonemap_index == styles.TONEMAP_NONE

        styles.apply_style(plot, styles.get_style("neon"))
        assert plot.options.visual.tonemap_index == styles.TONEMAP_ACES
        assert plot.options.visual.glow.enabled is True
        assert plot.options.visual.glow.knee == pytest.approx(0.5)
        assert plot.options.blend_mode is BlendMode.ADDITIVE

    def test_gradient_is_off_unless_the_preset_asks(self):
        """Only Chalk ships a gradient; the rest must clear it, not inherit one."""
        plot = _plot_with_layers()
        styles.apply_style(plot, styles.get_style("chalk"))
        assert plot.options.visual.gradient_background.enabled is True

        styles.apply_style(plot, styles.get_style("minimal"))
        assert plot.options.visual.gradient_background.enabled is False

    def test_empty_scene_is_fine(self):
        """Applying a style before any data exists must not raise."""
        plot = GPULinePlot()
        assert styles.apply_style(plot, styles.get_style("hand")) == 0


class TestStyleUndo:
    """Test that applying a preset is reversible — SPEC_FIX P1-F's required regression."""

    def _snapshot(self, plot):
        options = plot.options
        return {
            "bg": tuple(options.visual.background_color),
            "blend": options.blend_mode,
            "aa": options.enable_antialiasing,
            "alpha": options.default_global_alpha,
            "grid": options.axis_show_grid,
            "frame": options.axis_show_frame,
            "labels": options.axis_show_labels,
            "grid_alpha": options.axis_grid_alpha,
            "grid_color": tuple(options.axis_grid_color),
            "scheme": options.density_scheme_index,
            "light": options.light_bg_mode,
            "glow": (
                options.visual.glow.enabled,
                options.visual.glow.threshold,
                options.visual.glow.intensity,
                options.visual.glow.radius_px,
            ),
            "gradient": (
                options.visual.gradient_background.enabled,
                tuple(options.visual.gradient_background.top_color),
                tuple(options.visual.gradient_background.bottom_color),
            ),
            "tonemap": getattr(options.visual, "tonemap_index", "<absent>"),
            "grain": getattr(options.visual, "grain_amount", "<absent>"),
            "knee": getattr(options.visual.glow, "knee", "<absent>"),
            "l0_color": plot.scene.layers[0].style.color,
            "l0_width": plot.scene.layers[0].style.line_width,
            "l0_pts": plot.scene.layers[0].pts.copy(),
            "l2_size": plot.scene.layers[2].style.point_size,
            "l2_colors": plot.scene.layers[2].colors.copy(),
        }

    def _assert_same(self, before, after):
        for key in before:
            if isinstance(before[key], np.ndarray):
                assert np.array_equal(before[key], after[key]), key
            else:
                assert before[key] == after[key], key

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_undo_restores_every_field(self, key):
        """Apply any preset, undo, and every field this module touches is back."""
        plot = _plot_with_layers()
        before = self._snapshot(plot)

        undo = UndoStack()
        assert undo.push(styles.style_command(plot, styles.get_style(key))) is True
        undo.undo()

        self._assert_same(before, self._snapshot(plot))

    def test_undo_removes_fields_the_options_never_declared(self):
        """tonemap_index/grain_amount do not exist on VisualOptions; undo must not add them.

        Restoring a value the class never declared would freeze a default the library has
        not chosen yet, and would break the moment options.py grows the real field.
        """
        plot = _plot_with_layers()
        assert not hasattr(plot.options.visual, "grain_amount")

        undo = UndoStack()
        undo.push(styles.style_command(plot, styles.get_style("chalk")))
        assert hasattr(plot.options.visual, "grain_amount")

        undo.undo()
        assert not hasattr(plot.options.visual, "grain_amount")
        assert not hasattr(plot.options.visual, "tonemap_index")

    def test_redo_reapplies(self):
        """The command is a real Command: undo then redo lands on the styled scene."""
        plot = _plot_with_layers()
        undo = UndoStack()
        undo.push(styles.style_command(plot, styles.get_style("dark")))
        undo.undo()
        undo.redo()
        assert tuple(plot.options.visual.background_color) == styles.get_style("dark").background

    def test_undo_survives_a_deleted_layer(self):
        """A layer removed between apply and undo is skipped, not resurrected."""
        plot = _plot_with_layers()
        undo = UndoStack()
        undo.push(styles.style_command(plot, styles.get_style("print")))
        plot.scene.layers.pop(1)
        undo.undo()
        assert len(plot.scene.layers) == 2

    def test_label_names_the_preset(self):
        """The undo menu has to say what it will undo."""
        plot = _plot_with_layers()
        undo = UndoStack()
        undo.push(styles.style_command(plot, styles.get_style("chalk")))
        assert undo.peek_undo() == "Style: Chalk"


class TestHandDrawn:
    """Test the hand-drawn transform: geometry, seeded, capped."""

    def test_is_seeded_and_stable(self):
        """A per-frame reseed would shimmer; the same seed must redraw the same line."""
        x = np.linspace(0.0, 10.0, 200)
        y = np.sin(x)
        a = styles.jitter_polyline(x, y, seed=7)
        b = styles.jitter_polyline(x, y, seed=7)
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])

    def test_different_seed_is_a_different_hand(self):
        """Otherwise the seed control is a lie."""
        x = np.linspace(0.0, 10.0, 200)
        y = np.sin(x)
        a = styles.jitter_polyline(x, y, seed=7)
        c = styles.jitter_polyline(x, y, seed=8)
        assert not np.array_equal(a[1], c[1])

    def test_actually_displaces(self):
        """A straight line must come out visibly wobbled, not merely resampled."""
        x = np.linspace(0.0, 1.0, 50)
        y = np.zeros(50)
        jx, jy = styles.jitter_polyline(x, y, amplitude=0.01, seed=3)
        assert np.abs(jy).max() > 0.0

    def test_amplitude_is_scale_free(self):
        """Same wobble, relative to the data, at 1e-9 and 1e9."""
        x = np.linspace(0.0, 1.0, 200)
        y = np.sin(x * 6.0)
        _, small = styles.jitter_polyline(x * 1e-9, y * 1e-9, seed=5)
        _, big = styles.jitter_polyline(x * 1e9, y * 1e9, seed=5)
        assert np.allclose(small / 1e-9, big / 1e9, atol=1e-4)

    def test_ends_are_pinned(self):
        """A stroke that starts mid-wobble reads as a broken line."""
        x = np.linspace(0.0, 1.0, 100)
        y = np.zeros(100)
        jx, jy = styles.jitter_polyline(x, y, amplitude=0.02, seed=1)
        assert abs(float(jy[0])) < 1e-6
        assert abs(float(jy[-1])) < 1e-6

    def test_refuses_oversized_input(self):
        """Above the cap it returns None so the UI can say so instead of stalling."""
        n = styles.HAND_DRAWN_MAX_POINTS + 1
        assert styles.jitter_polyline(np.zeros(n), np.zeros(n)) is None

    def test_refuses_gaps(self):
        """Resampling across a NaN would draw a line where the author asked for a hole."""
        x = np.array([0.0, 1.0, np.nan, 3.0])
        y = np.array([0.0, 1.0, np.nan, 3.0])
        assert styles.jitter_polyline(x, y) is None

    def test_degenerate_input_survives(self):
        """One point, or a zero-length path, must not raise or divide by zero."""
        assert styles.jitter_polyline([1.0], [2.0])[0].tolist() == [1.0]
        jx, jy = styles.jitter_polyline([1.0, 1.0], [2.0, 2.0])
        assert np.allclose(jx, 1.0) and np.allclose(jy, 2.0)

    def test_applies_to_lines_only(self):
        """Scatter has no path to wobble along; line_family's ab columns are not a curve."""
        plot = _plot_with_layers()
        scatter_pts = plot.scene.layers[2].pts.copy()
        styles.apply_style(plot, styles.get_style("hand"))
        assert not np.array_equal(plot.scene.layers[0].pts, scatter_pts[: len(scatter_pts)])
        assert np.array_equal(plot.scene.layers[2].pts, scatter_pts)

    def test_does_not_compound(self):
        """Hand -> Hand must not wobble the wobble."""
        plot = _plot_with_layers()
        styles.apply_style(plot, styles.get_style("hand"))
        once = plot.scene.layers[0].pts.copy()
        styles.apply_style(plot, styles.get_style("hand"))
        assert np.array_equal(plot.scene.layers[0].pts, once)

    def test_another_preset_restores_the_true_geometry(self):
        """The transform is a view of the data, and must be exactly reversible."""
        plot = _plot_with_layers()
        original = plot.scene.layers[0].pts.copy()
        styles.apply_style(plot, styles.get_style("hand"))
        assert not np.array_equal(plot.scene.layers[0].pts, original)

        styles.apply_style(plot, styles.get_style("clean"))
        assert np.array_equal(plot.scene.layers[0].pts, original)

    def test_undo_of_hand_drawn_restores_geometry(self):
        """Undo is the other way back out, and it must not leave a jittered copy behind."""
        plot = _plot_with_layers()
        original = plot.scene.layers[0].pts.copy()
        undo = UndoStack()
        undo.push(styles.style_command(plot, styles.get_style("hand")))
        undo.undo()
        assert np.array_equal(plot.scene.layers[0].pts, original)

    def test_oversized_layers_is_reported_not_silently_skipped(self):
        """The panel calls this from draw to warn before the click."""
        plot = GPULinePlot()
        n = styles.HAND_DRAWN_MAX_POINTS + 10
        plot.add_line_strip(np.zeros(n), np.zeros(n), color=(1, 1, 1, 1), label="HUGE")
        assert styles.oversized_layers(plot) == ["HUGE"]
        assert styles.apply_style(plot, styles.get_style("hand")) == 1

    def test_oversized_probe_is_side_effect_free(self):
        """It runs from a draw callback, so it must not touch the scene."""
        plot = _plot_with_layers()
        plot.frame.dirty_scene = False
        styles.oversized_layers(plot)
        assert plot.frame.dirty_scene is False


class TestFactoryDefault:
    """Test the guard on the startup default."""

    def test_a_fresh_plot_is_factory_default(self):
        """The constructor's own colours are nobody's choice."""
        assert styles.is_factory_default(GPULinePlot()) is True

    def test_a_styled_plot_is_not(self):
        """Applying any preset must make the panel stop volunteering another."""
        plot = GPULinePlot()
        styles.apply_style(plot, styles.get_style("chalk"), layers=False)
        assert styles.is_factory_default(plot) is False

    def test_a_user_background_is_respected(self):
        """gplt users who set a background before show() meant it."""
        plot = GPULinePlot()
        plot.options.visual.background_color = (0.2, 0.4, 0.6)
        assert styles.is_factory_default(plot) is False

    def test_a_user_option_is_respected(self):
        """Same for a blend mode or a colormap."""
        plot = GPULinePlot()
        plot.options.blend_mode = BlendMode.SCREEN
        assert styles.is_factory_default(plot) is False


class TestChalkGrain:
    """Test the composite pass's grain, which exists for the Chalk preset.

    Lives here rather than in ``test_effects_post.py`` because grain is a style feature:
    the shader uniform is its plumbing. GL cannot run headless, so the GLSL is asserted to
    declare and use the uniform, and the CPU-side policy is tested on EffectManager, which
    constructs without a window.
    """

    def _manager(self):
        from glplot.managers.effects import EffectManager

        return EffectManager(GPULinePlot())

    def test_grain_is_off_by_default(self):
        """VisualOptions has no grain_amount field; absent must mean a no-op composite."""
        manager = self._manager()
        assert not hasattr(manager.options.visual, "grain_amount")
        assert manager.grain_amount() == 0.0

    def test_grain_reads_the_visual_option_when_present(self):
        """The same optional-field contract effects.py already uses for tonemap_index."""
        manager = self._manager()
        manager.options.visual.grain_amount = 0.055
        assert manager.grain_amount() == pytest.approx(0.055)

    def test_grain_rejects_garbage_and_clamps(self):
        """A style file or a stale pickle must never blow out the image."""
        manager = self._manager()
        manager.options.visual.grain_amount = "chalk"
        assert manager.grain_amount() == 0.0
        manager.options.visual.grain_amount = float("nan")
        assert manager.grain_amount() == 0.0
        manager.options.visual.grain_amount = 99.0
        assert manager.grain_amount() == 0.5
        manager.options.visual.grain_amount = -1.0
        assert manager.grain_amount() == 0.0

    def test_grain_keeps_the_post_pass_alive_without_reactive_rendering(self):
        """Otherwise the grain silently vanishes for anyone who turns reactive off."""
        manager = self._manager()
        manager.options.reactive_rendering = False
        manager.options.visual.glow.enabled = False
        assert manager.any_post_enabled() is False

        manager.options.visual.grain_amount = 0.05
        assert manager.any_post_enabled() is True

    def test_composite_declares_and_uses_the_grain_uniform(self):
        """The CPU policy above is worthless if the shader never reads it."""
        from glplot.utils import shaders

        source = shaders.POST_COMPOSITE_FS
        assert "uniform float u_grain;" in source
        assert "u_grain > 0.0" in source
        assert "grain_hash" in source

    def test_grain_is_static_not_animated(self):
        """A time-varying grain would crawl, and would freeze under reactive rendering."""
        from glplot.utils import shaders

        assert "u_time" not in shaders.POST_COMPOSITE_FS

    def test_grain_is_applied_after_tone_mapping(self):
        """Grain is the paper, not the light: tone-mapping it would wash it out."""
        from glplot.utils import shaders

        source = shaders.POST_COMPOSITE_FS
        assert source.index("u_tonemap == 1") < source.index("u_grain > 0.0")

    def test_only_the_surface_looks_ship_grain(self):
        """Grain is paper tooth, not a default: it belongs to the looks that imitate a real
        drawing surface (slate, dry-erase board, waxed manila) and nothing else."""
        grainy = [s.key for s in styles.STYLES if s.grain > 0.0]
        assert grainy == ["chalk", "marker", "kids"]


class TestBackgroundModeLatch:
    """Regression: the engine repainted the background on EVERY frame.

    ``_apply_background_mode`` is called once per frame from ``_main_loop`` and used to run
    its body unconditionally, overwriting ``background_color``, the whole gradient and
    ``axis_grid_color`` from two hardcoded pairs. Every one of these tests fails on the old
    code: a style preset, the Style panel's colour picker and its gradient checkbox were
    all dead on screen, and the public option was simply unsettable.
    """

    def test_constructor_still_seeds_the_stock_background(self):
        """Default behaviour is preserved: a fresh plot looks exactly as it always did."""
        plot = GPULinePlot()
        assert plot.options.light_bg_mode is True
        assert tuple(plot.options.visual.background_color) == (1.0, 1.0, 1.0)
        assert tuple(plot.options.axis_grid_color) == (0.8, 0.8, 0.8)

    def test_a_user_set_background_survives_a_frame(self):
        """plot.options.visual.background_color is a public option and must stick."""
        plot = GPULinePlot()
        plot.options.visual.background_color = (0.2, 0.4, 0.6)
        plot._apply_background_mode()
        assert tuple(plot.options.visual.background_color) == (0.2, 0.4, 0.6)

    def test_a_gradient_toggled_off_stays_off(self):
        """The Effects tab's Enable Gradient checkbox was re-ticked every frame."""
        plot = GPULinePlot()
        plot.options.visual.gradient_background.enabled = False
        plot._apply_background_mode()
        assert plot.options.visual.gradient_background.enabled is False

    @pytest.mark.parametrize("key", styles.STYLE_KEYS)
    def test_every_preset_survives_a_frame(self, key):
        """The whole preset system is worthless if the loop repaints over it."""
        plot = GPULinePlot()
        style = styles.get_style(key)
        styles.apply_style(plot, style, layers=False)
        plot._apply_background_mode()

        assert tuple(plot.options.visual.background_color) == style.background
        assert plot.options.visual.gradient_background.enabled == (style.gradient is not None)
        expected_grid = style.grid_color or styles.AUTO_GRID_COLOR
        assert tuple(plot.options.axis_grid_color) == tuple(expected_grid)

    def test_toggling_the_mode_still_restyles(self):
        """The feature the per-frame call was there for keeps working."""
        plot = GPULinePlot()
        plot.set_background_mode(False)
        assert tuple(plot.options.visual.background_color) == (0.0, 0.0, 0.0)
        assert tuple(plot.options.axis_grid_color) == (0.2, 0.2, 0.2)

        plot.set_background_mode(True)
        assert tuple(plot.options.visual.background_color) == (1.0, 1.0, 1.0)

    def test_setting_the_option_directly_still_restyles_on_the_next_frame(self):
        """Anyone flipping options.light_bg_mode by hand (HUD 'B') gets the stock repaint."""
        plot = GPULinePlot()
        plot.options.light_bg_mode = False
        plot._apply_background_mode()
        assert tuple(plot.options.visual.background_color) == (0.0, 0.0, 0.0)

    def test_adopting_a_mode_without_the_preset_leaves_colours_alone(self):
        """apply_preset=False is how a style preset claims 'dark page, but mine'."""
        plot = GPULinePlot()
        plot.options.visual.background_color = (0.1, 0.2, 0.3)
        plot.set_background_mode(False, apply_preset=False)
        assert plot.options.light_bg_mode is False
        assert tuple(plot.options.visual.background_color) == (0.1, 0.2, 0.3)
        plot._apply_background_mode()
        assert tuple(plot.options.visual.background_color) == (0.1, 0.2, 0.3)
