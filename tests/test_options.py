"""Test options and configuration classes in glplot.options.

Focus on enum options, dataclass initialization, and default values
without requiring OpenGL or GPU.
"""

import pytest

from glplot.options import (
    BlendMode,
    EngineOptions,
    GlobalStyleOverrides,
    GlowOptions,
    GradientBackgroundOptions,
    RenderMode,
    RuntimePolicy,
    SSAOOptions,
    VisualOptions,
)


class TestRenderMode:
    """Test RenderMode enum."""

    def test_enum_members_exist(self):
        """Test that expected enum members exist."""
        assert hasattr(RenderMode, "EXACT")
        assert hasattr(RenderMode, "INTERACTIVE")

    def test_enum_member_values_unique(self):
        """Test that enum members have unique values."""
        assert RenderMode.EXACT != RenderMode.INTERACTIVE

    def test_enum_iteration(self):
        """Test enum iteration."""
        members = list(RenderMode)
        assert len(members) >= 2

    def test_enum_comparison(self):
        """Test enum comparison."""
        mode = RenderMode.EXACT
        assert mode == RenderMode.EXACT
        assert mode != RenderMode.INTERACTIVE


class TestBlendMode:
    """Test BlendMode enum."""

    def test_all_blend_modes_exist(self):
        """Test that all blend modes are defined."""
        assert hasattr(BlendMode, "ALPHA")
        assert hasattr(BlendMode, "ADDITIVE")
        assert hasattr(BlendMode, "SUBTRACTIVE")
        assert hasattr(BlendMode, "SCREEN")
        assert hasattr(BlendMode, "AUTO")
        assert hasattr(BlendMode, "OFF")

    def test_blend_modes_unique(self):
        """Test that blend modes have unique values."""
        modes = [BlendMode.ALPHA, BlendMode.ADDITIVE, BlendMode.AUTO, BlendMode.OFF]
        values = [mode.value for mode in modes]
        assert len(values) == len(set(values))

    def test_blend_mode_iteration(self):
        """Test iterating over blend modes."""
        modes = list(BlendMode)
        assert len(modes) == 6

    def test_default_blend_mode(self):
        """Test that AUTO is a valid default."""
        assert BlendMode.AUTO is not None


class TestGlowOptions:
    """Test GlowOptions dataclass."""

    def test_default_initialization(self):
        """Test default GlowOptions."""
        glow = GlowOptions()
        assert glow.enabled is False
        assert glow.threshold == 0.7
        assert glow.intensity == 0.8
        assert glow.radius_px == 6.0
        assert glow.resolution_scale == 0.5

    def test_custom_initialization(self):
        """Test custom GlowOptions."""
        glow = GlowOptions(
            enabled=True, threshold=0.5, intensity=1.0, radius_px=10.0, resolution_scale=1.0
        )
        assert glow.enabled is True
        assert glow.threshold == 0.5
        assert glow.intensity == 1.0
        assert glow.radius_px == 10.0
        assert glow.resolution_scale == 1.0

    def test_partial_initialization(self):
        """Test partial custom initialization."""
        glow = GlowOptions(enabled=True, intensity=0.5)
        assert glow.enabled is True
        assert glow.intensity == 0.5
        assert glow.threshold == 0.7  # Default

    def test_threshold_range(self):
        """Test threshold parameter variations."""
        glow_low = GlowOptions(threshold=0.1)
        glow_high = GlowOptions(threshold=0.9)
        assert glow_low.threshold != glow_high.threshold

    def test_intensity_range(self):
        """Test intensity parameter variations."""
        glow_low = GlowOptions(intensity=0.1)
        glow_high = GlowOptions(intensity=2.0)
        assert glow_low.intensity != glow_high.intensity

    def test_enabled_toggle(self):
        """Test toggling enabled flag."""
        glow_on = GlowOptions(enabled=True)
        glow_off = GlowOptions(enabled=False)
        assert glow_on.enabled != glow_off.enabled


class TestGradientBackgroundOptions:
    """Test GradientBackgroundOptions dataclass."""

    def test_default_initialization(self):
        """Test default GradientBackgroundOptions."""
        grad = GradientBackgroundOptions()
        assert grad.enabled is False
        assert grad.top_color == (1.0, 1.0, 1.0)
        assert grad.bottom_color == (0.95, 0.97, 1.0)

    def test_custom_colors(self):
        """Test custom color initialization."""
        grad = GradientBackgroundOptions(
            enabled=True, top_color=(0.0, 0.0, 0.0), bottom_color=(1.0, 1.0, 1.0)
        )
        assert grad.top_color == (0.0, 0.0, 0.0)
        assert grad.bottom_color == (1.0, 1.0, 1.0)

    def test_color_tuple_format(self):
        """Test that colors are tuples."""
        grad = GradientBackgroundOptions()
        assert isinstance(grad.top_color, tuple)
        assert isinstance(grad.bottom_color, tuple)
        assert len(grad.top_color) == 3
        assert len(grad.bottom_color) == 3

    def test_enabled_toggle(self):
        """Test toggling enabled flag."""
        grad_on = GradientBackgroundOptions(enabled=True)
        grad_off = GradientBackgroundOptions(enabled=False)
        assert grad_on.enabled != grad_off.enabled


class TestSSAOOptions:
    """Test SSAOOptions dataclass."""

    def test_default_initialization(self):
        """Test default SSAOOptions."""
        ssao = SSAOOptions()
        assert ssao.enabled is False
        assert ssao.strength == 0.45
        assert ssao.radius == 1.0

    def test_custom_initialization(self):
        """Test custom SSAOOptions."""
        ssao = SSAOOptions(enabled=True, strength=0.8, radius=2.0)
        assert ssao.enabled is True
        assert ssao.strength == 0.8
        assert ssao.radius == 2.0

    def test_strength_variation(self):
        """Test varying strength values."""
        ssao_weak = SSAOOptions(strength=0.1)
        ssao_strong = SSAOOptions(strength=1.0)
        assert ssao_weak.strength < ssao_strong.strength

    def test_radius_variation(self):
        """Test varying radius values."""
        ssao_small = SSAOOptions(radius=0.5)
        ssao_large = SSAOOptions(radius=5.0)
        assert ssao_small.radius < ssao_large.radius


class TestGlobalStyleOverrides:
    """Test GlobalStyleOverrides dataclass."""

    def test_default_initialization(self):
        """Test default GlobalStyleOverrides."""
        overrides = GlobalStyleOverrides()
        assert overrides.enabled is True
        assert overrides.alpha_multiplier == 1.0
        assert overrides.line_width_multiplier == 1.0
        assert overrides.point_size_multiplier == 1.0

    def test_all_disabled(self):
        """Test disabling all overrides."""
        overrides = GlobalStyleOverrides(enabled=False)
        assert overrides.enabled is False

    def test_alpha_multiplier(self):
        """Test alpha multiplier variations."""
        overrides_low = GlobalStyleOverrides(alpha_multiplier=0.5)
        overrides_high = GlobalStyleOverrides(alpha_multiplier=2.0)
        assert overrides_low.alpha_multiplier < overrides_high.alpha_multiplier

    def test_line_width_multiplier(self):
        """Test line width multiplier variations."""
        overrides_thin = GlobalStyleOverrides(line_width_multiplier=0.5)
        overrides_thick = GlobalStyleOverrides(line_width_multiplier=3.0)
        assert overrides_thin.line_width_multiplier < overrides_thick.line_width_multiplier

    def test_point_size_multiplier(self):
        """Test point size multiplier variations."""
        overrides_small = GlobalStyleOverrides(point_size_multiplier=0.5)
        overrides_large = GlobalStyleOverrides(point_size_multiplier=4.0)
        assert overrides_small.point_size_multiplier < overrides_large.point_size_multiplier

    def test_zero_multipliers(self):
        """Test zero multiplier values."""
        overrides = GlobalStyleOverrides(
            alpha_multiplier=0.0, line_width_multiplier=0.0, point_size_multiplier=0.0
        )
        assert overrides.alpha_multiplier == 0.0
        assert overrides.line_width_multiplier == 0.0
        assert overrides.point_size_multiplier == 0.0


class TestVisualOptions:
    """Test VisualOptions dataclass."""

    def test_default_initialization(self):
        """Test default VisualOptions."""
        visual = VisualOptions()
        assert visual.background_color == (0.0, 0.0, 0.0)
        assert isinstance(visual.glow, GlowOptions)
        assert isinstance(visual.gradient_background, GradientBackgroundOptions)
        assert isinstance(visual.ssao, SSAOOptions)
        assert isinstance(visual.overrides, GlobalStyleOverrides)

    def test_custom_background_color(self):
        """Test custom background color."""
        visual = VisualOptions(background_color=(1.0, 1.0, 1.0))
        assert visual.background_color == (1.0, 1.0, 1.0)

    def test_nested_glow_options(self):
        """Test nested GlowOptions."""
        glow = GlowOptions(enabled=True, intensity=0.5)
        visual = VisualOptions(glow=glow)
        assert visual.glow.enabled is True
        assert visual.glow.intensity == 0.5

    def test_nested_ssao_options(self):
        """Test nested SSAOOptions."""
        ssao = SSAOOptions(enabled=True, strength=0.7)
        visual = VisualOptions(ssao=ssao)
        assert visual.ssao.enabled is True
        assert visual.ssao.strength == 0.7

    def test_nested_overrides(self):
        """Test nested GlobalStyleOverrides."""
        overrides = GlobalStyleOverrides(alpha_multiplier=0.8)
        visual = VisualOptions(overrides=overrides)
        assert visual.overrides.alpha_multiplier == 0.8

    def test_multiple_visual_options_independent(self):
        """Test that multiple VisualOptions instances are independent."""
        visual1 = VisualOptions()
        visual2 = VisualOptions()
        visual2.glow.enabled = True
        assert visual1.glow.enabled is False


class TestEngineOptions:
    """Test EngineOptions dataclass."""

    def test_default_initialization(self):
        """Test default EngineOptions."""
        opts = EngineOptions()
        assert opts.window_width == 1400
        assert opts.window_height == 900
        assert opts.title == "Hybrid GPU Line Renderer"
        assert opts.lod_enabled is True
        assert opts.default_global_alpha == 1.0
        assert opts.enable_hud is False
        assert opts.axis_show_grid is True

    def test_custom_window_size(self):
        """Test custom window dimensions."""
        opts = EngineOptions(window_width=800, window_height=600)
        assert opts.window_width == 800
        assert opts.window_height == 600

    def test_custom_title(self):
        """Test custom window title."""
        opts = EngineOptions(title="My Plot")
        assert opts.title == "My Plot"

    def test_lod_parameters(self):
        """Test level-of-detail parameters."""
        opts = EngineOptions(lod_enabled=False, lod_target_coverage=0.5)
        assert opts.lod_enabled is False
        assert opts.lod_target_coverage == 0.5

    def test_interaction_parameters(self):
        """Test interaction parameters."""
        opts = EngineOptions(
            drag_threshold_px=5.0, hover_pick_hz=1.0, zoom_scroll_factor=1.15
        )
        assert opts.drag_threshold_px == 5.0
        assert opts.hover_pick_hz == 1.0
        assert opts.zoom_scroll_factor == 1.15

    def test_density_parameters(self):
        """Test density rendering parameters."""
        opts = EngineOptions(density_gain=2.0, density_is_log=False)
        assert opts.density_gain == 2.0
        assert opts.density_is_log is False

    def test_visual_effects_options(self):
        """Test visual effects options."""
        opts = EngineOptions()
        assert isinstance(opts.visual, VisualOptions)

    def test_blend_mode_setting(self):
        """Test blend mode in engine options."""
        opts_auto = EngineOptions(blend_mode=BlendMode.AUTO)
        opts_additive = EngineOptions(blend_mode=BlendMode.ADDITIVE)
        assert opts_auto.blend_mode == BlendMode.AUTO
        assert opts_additive.blend_mode == BlendMode.ADDITIVE

    def test_antialiasing_toggle(self):
        """Test antialiasing enable/disable."""
        opts_on = EngineOptions(enable_antialiasing=True)
        opts_off = EngineOptions(enable_antialiasing=False)
        assert opts_on.enable_antialiasing is True
        assert opts_off.enable_antialiasing is False

    def test_reactive_rendering_toggle(self):
        """Test reactive rendering enable/disable."""
        opts_on = EngineOptions(reactive_rendering=True)
        opts_off = EngineOptions(reactive_rendering=False)
        assert opts_on.reactive_rendering is True
        assert opts_off.reactive_rendering is False

    def test_export_scale(self):
        """Test export scale parameter."""
        opts_2x = EngineOptions(export_scale=2.0)
        opts_1x = EngineOptions(export_scale=1.0)
        assert opts_2x.export_scale == 2.0
        assert opts_1x.export_scale == 1.0

    def test_picking_parameters(self):
        """Test picking-related parameters."""
        opts = EngineOptions(shift_required_for_picking=False, picking_radius_px=10)
        assert opts.shift_required_for_picking is False
        assert opts.picking_radius_px == 10

    def test_grid_and_axis_parameters(self):
        """Test grid and axis visibility parameters."""
        opts = EngineOptions(
            axis_show_grid=False, axis_show_labels=False, axis_show_frame=False
        )
        assert opts.axis_show_grid is False
        assert opts.axis_show_labels is False
        assert opts.axis_show_frame is False

    def test_grid_styling(self):
        """Test grid color and alpha."""
        opts = EngineOptions(axis_grid_alpha=0.2, axis_grid_color=(0.5, 0.5, 0.5))
        assert opts.axis_grid_alpha == 0.2
        assert opts.axis_grid_color == (0.5, 0.5, 0.5)


class TestRuntimePolicy:
    """Test RuntimePolicy dataclass."""

    def test_default_initialization(self):
        """Test default RuntimePolicy."""
        policy = RuntimePolicy()
        assert policy.blending_enabled is True
        assert policy.current_mode == RenderMode.EXACT
        assert policy.picking_enabled_this_frame is False
        assert policy.hud_enabled_this_frame is False

    def test_disable_blending(self):
        """Test disabling blending."""
        policy = RuntimePolicy(blending_enabled=False)
        assert policy.blending_enabled is False

    def test_change_render_mode(self):
        """Test changing render mode."""
        policy_exact = RuntimePolicy(current_mode=RenderMode.EXACT)
        policy_interactive = RuntimePolicy(current_mode=RenderMode.INTERACTIVE)
        assert policy_exact.current_mode == RenderMode.EXACT
        assert policy_interactive.current_mode == RenderMode.INTERACTIVE

    def test_enable_picking_this_frame(self):
        """Test picking enable flag."""
        policy_off = RuntimePolicy(picking_enabled_this_frame=False)
        policy_on = RuntimePolicy(picking_enabled_this_frame=True)
        assert policy_off.picking_enabled_this_frame is False
        assert policy_on.picking_enabled_this_frame is True

    def test_enable_hud_this_frame(self):
        """Test HUD enable flag."""
        policy_off = RuntimePolicy(hud_enabled_this_frame=False)
        policy_on = RuntimePolicy(hud_enabled_this_frame=True)
        assert policy_off.hud_enabled_this_frame is False
        assert policy_on.hud_enabled_this_frame is True

    def test_multiple_policies_independent(self):
        """Test that multiple policy instances are independent."""
        policy1 = RuntimePolicy()
        policy2 = RuntimePolicy()
        policy2.blending_enabled = False
        assert policy1.blending_enabled is True


class TestOptionsConsistency:
    """Test consistency across option classes."""

    def test_all_dataclass_instantiation(self):
        """Test that all dataclass options can be instantiated."""
        classes = [
            GlowOptions,
            GradientBackgroundOptions,
            SSAOOptions,
            GlobalStyleOverrides,
            VisualOptions,
            EngineOptions,
            RuntimePolicy,
        ]
        for cls in classes:
            instance = cls()
            assert instance is not None

    def test_visual_options_contains_all_sub_options(self):
        """Test that VisualOptions contains all nested option types."""
        visual = VisualOptions()
        assert isinstance(visual.glow, GlowOptions)
        assert isinstance(visual.gradient_background, GradientBackgroundOptions)
        assert isinstance(visual.ssao, SSAOOptions)
        assert isinstance(visual.overrides, GlobalStyleOverrides)

    def test_engine_options_contains_visual_options(self):
        """Test that EngineOptions contains VisualOptions."""
        engine = EngineOptions()
        assert isinstance(engine.visual, VisualOptions)
