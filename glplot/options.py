from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto


class RenderMode(Enum):
    EXACT = auto()
    INTERACTIVE = auto()


class BlendMode(Enum):
    ALPHA = auto()  # Standard transparency (SrcAlpha, OneMinusSrcAlpha)
    ADDITIVE = auto()  # Glowing accumulation (SrcAlpha, One)
    SUBTRACTIVE = auto()  # Anti-glow (RevSub, SrcAlpha, One)
    SCREEN = auto()  # Lightening (One, OneMinusSrcColor)
    AUTO = auto()  # Smart switching based on count
    OFF = auto()  # No blending


@dataclass
class GlowOptions:
    enabled: bool = False
    threshold: float = 0.7
    intensity: float = 0.8
    radius_px: float = 6.0
    resolution_scale: float = 0.5


@dataclass
class GradientBackgroundOptions:
    enabled: bool = False
    top_color: tuple = (1.0, 1.0, 1.0)
    bottom_color: tuple = (0.95, 0.97, 1.0)


@dataclass
class SSAOOptions:
    """Lightweight screen-space-style ambient occlusion controls for 3D layers."""

    enabled: bool = False
    strength: float = 0.45
    radius: float = 1.0


@dataclass
class GlobalStyleOverrides:
    """Centralized multipliers to adjust the entire scene at once."""

    enabled: bool = True
    alpha_multiplier: float = 1.0
    line_width_multiplier: float = 1.0
    point_size_multiplier: float = 1.0


@dataclass
class VisualOptions:
    background_color: tuple = (0.0, 0.0, 0.0)
    glow: GlowOptions = field(default_factory=GlowOptions)
    gradient_background: GradientBackgroundOptions = field(
        default_factory=GradientBackgroundOptions
    )
    ssao: SSAOOptions = field(default_factory=SSAOOptions)
    overrides: GlobalStyleOverrides = field(default_factory=GlobalStyleOverrides)


@dataclass
class EngineOptions:
    window_width: int = 1400
    window_height: int = 900
    title: str = "Hybrid GPU Line Renderer"

    # Quality / scale policy
    lod_enabled: bool = True
    lod_target_coverage: float = 0.35
    default_global_alpha: float = 1.0
    default_line_budget_per_px: int = 8
    global_line_width: float = 1.0
    interaction_budget_lines_per_screen_px: int = 2
    auto_disable_blending_threshold: int = 100_000_000

    # Interaction
    drag_threshold_px: float = 4.0
    hover_pick_hz: float = 0.0  # 0 means disabled; picking is shift-only
    cache_refresh_hz: float = 10.0
    cache_padding: float = 3.0
    cache_safe_margin: float = 0.15
    zoom_scroll_factor: float = 1.10

    # Navigation tuning (Phase 2 additions planned)
    pan_key_fraction: float = 0.08
    zoom_key_factor: float = 1.15
    zoom_scroll_factor: float = 1.10
    box_zoom_min_pixels: int = 8

    # Feature policies
    enable_hud: bool = False  # ask user beforehand, or set explicitly

    # Axis / Framework visibility
    axis_show_grid: bool = True
    axis_show_labels: bool = True
    axis_show_frame: bool = True
    axis_grid_alpha: float = 0.1
    axis_grid_color: tuple = (0.2, 0.2, 0.2)

    enable_density_interaction_path: bool = True
    enable_cache_interaction_path: bool = True
    enable_clipping_optimization: bool = True
    enable_multisample: bool = False
    enable_antialiasing: bool = True
    always_lod: bool = False
    reactive_rendering: bool = True

    # Picking policy
    shift_required_for_picking: bool = True
    picking_radius_px: int = 5

    # Export
    export_scale: float = 2.0

    # Density rendering
    density_gain: float = 1.0
    density_resolution_scale: float = 1.0  # 1.0 = full-res, 0.5 = faster
    density_scheme_index: int = 9
    density_gain_step: float = 1.25
    density_log_scale: float = 3.0  # Divisor for log normalization
    density_weighted: bool = False  # Accumulate alpha instead of 1.0
    density_invert: bool = True
    density_is_log: bool = True  # Enable logarithmic colormap scaling by default
    light_bg_mode: bool = True
    density_light_to_color: bool = (
        True  # Go from white to color (instead of white to color to black)
    )

    # Style
    blend_mode: BlendMode = BlendMode.AUTO
    line_colormap_enabled: bool = False
    enable_auto_alpha: bool = True  # Scale alpha based on N

    # Visual Effects
    visual: VisualOptions = field(default_factory=VisualOptions)


@dataclass
class RuntimePolicy:
    # Mutable runtime decisions derived from dataset size and state.
    blending_enabled: bool = True
    current_mode: RenderMode = RenderMode.EXACT
    picking_enabled_this_frame: bool = False
    hud_enabled_this_frame: bool = False
