from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator, Optional, Tuple, Union

#: The stock content inset ``(left, right, bottom, top)`` in pixels, reserved inside the
#: viewport for the axis gutters. These are the values the engine has always used and are
#: what an export must render with: they describe the figure, not whatever chrome a GUI
#: happens to have parked on top of the window. See :meth:`EngineOptions.axis_margins`.
DEFAULT_AXIS_MARGINS: Tuple[float, float, float, float] = (60.0, 20.0, 40.0, 20.0)

#: Tight per-panel gutters (l, r, b, t) used when the window is split into several panels, so
#: the frames pack closely instead of each reserving the full (or rail-widened) left gutter.
#: The left is kept above the icon rail's width so the left column's y labels still clear it.
PANEL_COMPACT_MARGINS: Tuple[float, float, float, float] = (46.0, 10.0, 28.0, 16.0)

#: Window captions that must NOT be promoted to an on-plot title. ``gplt.title()`` writes
#: the OS window caption (``engine.set_title``), and ``EngineOptions.title`` /
#: ``GPULinePlot.__init__`` default to these -- so treating that attribute as a plot title
#: unconditionally would caption every default window with its own product name. Anything
#: else is a caption the caller chose deliberately. See ``AxisRenderer._resolve_title``.
STOCK_WINDOW_TITLES = frozenset({"GLPlot", "Hybrid GPU Line Renderer"})

#: A colour as the public API accepts it: a matplotlib name ("white", "C0", "#ff8800") or an
#: RGB/RGBA tuple of floats. Kept un-normalised in the options on purpose -- the string form is
#: what a GUI wants to round-trip, and `AxisRenderer._ink` resolves it at draw time through the
#: same `_normalize_rgba` every other colour argument goes through.
ColorOption = Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]


def resolve_axis_margins(options: object) -> Tuple[float, float, float, float]:
    """The content inset ``(left, right, bottom, top)`` in pixels for ``options``.

    THE reader for the axis gutters. `CameraController.mvp`, `CameraController.screen_to_world`
    and the density resolve pass's UV clip box all come through here, so they cannot drift
    apart -- and `mvp`/`screen_to_world` are exact inverses, so drift there would silently
    decouple the cursor from the data under it.

    Tolerates option objects without the fields (older pickles, hand-rolled stubs in tests)
    by falling back to :data:`DEFAULT_AXIS_MARGINS`.
    """
    getter = getattr(options, "axis_margins", None)
    if callable(getter):
        return getter()
    return (
        float(getattr(options, "axis_margin_l", DEFAULT_AXIS_MARGINS[0])),
        float(getattr(options, "axis_margin_r", DEFAULT_AXIS_MARGINS[1])),
        float(getattr(options, "axis_margin_b", DEFAULT_AXIS_MARGINS[2])),
        float(getattr(options, "axis_margin_t", DEFAULT_AXIS_MARGINS[3])),
    )


@contextmanager
def override_axis_margins(
    options: object, margins: Tuple[float, float, float, float]
) -> Iterator[None]:
    """Render with ``margins`` as the gutters, whatever the live window is using.

    THE writer, counterpart to :func:`resolve_axis_margins`. An offscreen render builds
    its projection from the very same ``camera_controller.mvp()`` as the live view, so it
    would otherwise inherit whatever gutters the window happens to be using -- including
    the rail-sized ones a GUI widens for its own chrome.

    Restores the caller's values on the way out, including on exception: the live view
    must not be left mis-projected because an offscreen render failed.
    """
    saved = (
        getattr(options, "axis_margin_l", DEFAULT_AXIS_MARGINS[0]),
        getattr(options, "axis_margin_r", DEFAULT_AXIS_MARGINS[1]),
        getattr(options, "axis_margin_b", DEFAULT_AXIS_MARGINS[2]),
        getattr(options, "axis_margin_t", DEFAULT_AXIS_MARGINS[3]),
    )
    try:
        (
            options.axis_margin_l,
            options.axis_margin_r,
            options.axis_margin_b,
            options.axis_margin_t,
        ) = margins
        yield
    finally:
        (
            options.axis_margin_l,
            options.axis_margin_r,
            options.axis_margin_b,
            options.axis_margin_t,
        ) = saved


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

    # Axis content inset, in pixels. The projection compresses the world into the region
    # left after these gutters, leaving room for tick labels outside the frame. Read the
    # four of them through `axis_margins()` and never re-type the numbers: `mvp()` and
    # `screen_to_world()` must agree exactly or the cursor stops matching the data under it.
    axis_margin_l: float = DEFAULT_AXIS_MARGINS[0]
    axis_margin_r: float = DEFAULT_AXIS_MARGINS[1]
    axis_margin_b: float = DEFAULT_AXIS_MARGINS[2]
    axis_margin_t: float = DEFAULT_AXIS_MARGINS[3]

    # Axis annotation. Empty means "not set" for all three, which is why the defaults draw
    # nothing and the stock look is unchanged.
    #
    # `axis_xlabel`/`axis_ylabel` are the GUI's channel. The `pyplot` channel is the
    # `plot.xlabel`/`plot.ylabel` attributes that `gplt.xlabel()`/`gplt.ylabel()` set --
    # historically written by the public API and read by nothing but the matplotlib savefig
    # fallback, so `gplt.xlabel(...)` drew nothing on screen. `AxisRenderer` now resolves
    # both, option first, so each channel works and the GUI wins when they disagree.
    axis_xlabel: str = ""
    axis_ylabel: str = ""

    # The *plot* title, drawn above the frame. Distinct from `title`, which is the OS
    # window caption -- `gplt.title()` sets that one (`engine.set_title`). When this is
    # empty the renderer falls back to the window title, but only once it stops being one
    # of the stock captions, so a default window does not caption itself. See
    # `AxisRenderer._resolve_title`.
    axis_title: str = ""

    # Per-annotation text styling, from the matplotlib Text kwargs `title()`, `xlabel()`
    # and `ylabel()` forward here. None means "not set" and resolves to the stock look:
    # unscaled, and the automatic light-on-dark ink the tick labels already use. The
    # *_fontsize values are matplotlib POINTS, not pixels -- `AxisRenderer._font_scale`
    # divides them by the point size matplotlib itself defaults each annotation to, so a
    # caller's 14 means what it means there. See `_MPL_DEFAULT_TITLE_PT`.
    axis_title_fontsize: Optional[float] = None
    axis_title_color: Optional[ColorOption] = None
    axis_xlabel_fontsize: Optional[float] = None
    axis_xlabel_color: Optional[ColorOption] = None
    axis_ylabel_fontsize: Optional[float] = None
    axis_ylabel_color: Optional[ColorOption] = None

    # Tick label styling, from `gplt.tick_params(labelsize=..., labelcolor=...)` or the GUI.
    # Same "None means unset" convention as the annotation styling above: unscaled, automatic
    # light-on-dark ink. `axis_tick_fontsize` is matplotlib POINTS, resolved the same way as
    # `axis_xlabel_fontsize` -- see `AxisRenderer._font_scale`.
    axis_tick_fontsize: Optional[float] = None
    axis_tick_color: Optional[ColorOption] = None

    # Axis scale mode, from `gplt.xscale()`/`yscale()`. "linear", "log", "symlog", and
    # "asinh" are real -- see `glplot.utils.scale`. Every layer's GPU buffer is built from
    # this at upload time; `layer.pts` itself is never transformed, so the Data panel and CSV
    # export keep reading real values regardless of what the axis is doing on screen.
    axis_scale_x: str = "linear"
    axis_scale_y: str = "linear"

    # Extra keyword arguments a caller passed to `xscale()`/`yscale()` -- `linthresh`/
    # `linscale`/`base` for "symlog", `linear_width` for "asinh". `glplot.utils.scale`
    # fills in matplotlib's own defaults for any key left unset.
    axis_scale_params_x: dict = field(default_factory=dict)
    axis_scale_params_y: dict = field(default_factory=dict)

    # Tick generation. 0 means "auto": the count is derived from the viewport size, and the
    # step from the 1-2-5 nice-number ladder. A non-zero `axis_tick_step_*` pins the spacing
    # outright and takes precedence over the count.
    axis_tick_count_x: int = 0
    axis_tick_count_y: int = 0
    axis_tick_step_x: float = 0.0
    axis_tick_step_y: float = 0.0

    # Explicit tick positions, from `gplt.xticks()`/`gplt.yticks()`. These outrank every
    # knob above: a caller who named the positions does not want them re-derived from the
    # viewport. None means "not set" (generate them); an EMPTY tuple is a real instruction
    # -- `xticks([])` is matplotlib's way of clearing an axis -- which is why this cannot
    # be a plain tuple defaulting to (), and why readers must test `is None`, not falsiness.
    #
    # Positions are kept in full, unfiltered by the view: the axis draws only what is
    # on-screen, but `xticks()` must be able to give back what it was handed, and a pan
    # must not silently discard ticks it scrolled past.
    axis_tick_values_x: Optional[Tuple[float, ...]] = None
    axis_tick_values_y: Optional[Tuple[float, ...]] = None

    # Labels for the positions above, positionally paired. None means "use the formatter",
    # so `xticks(ticks)` renumbers without inventing text. Only meaningful alongside the
    # matching axis_tick_values_*.
    axis_tick_labels_x: Optional[Tuple[str, ...]] = None
    axis_tick_labels_y: Optional[Tuple[str, ...]] = None

    # Minor ticks: `axis_minor_subdivisions` intervals per major step. Off by default.
    axis_minor_ticks: bool = False
    axis_minor_subdivisions: int = 5

    # Tick mark glyphs drawn inward from the spines. Off by default: the axis has only ever
    # drawn grid lines and spines, and turning marks on unasked would restyle every plot.
    axis_show_ticks: bool = False
    axis_tick_len_px: float = 5.0

    # Legend. `legend_show` is deliberately tri-state: None means "not configured", and the
    # renderer then follows `plot.legend_show`, which `gplt.legend()` sets. A plain bool
    # could not express the difference between "unset" and "the user hid it", so a GUI
    # checkbox would have no way to turn off a legend that `legend()` turned on. Setting it
    # True or False overrides the pyplot channel outright -- the same "option wins when set"
    # rule the axis annotations follow. Default None draws nothing, so the stock look of
    # every existing plot is unchanged until `legend()` is called.
    legend_show: Optional[bool] = None

    # One of `renderers.legend.LEGEND_LOCATIONS`. "best" scores the four corners by the
    # projected data area each would cover and takes the emptiest, ties going to
    # "upper right" -- which is the corner the matplotlib savefig path hardcodes.
    legend_loc: str = "best"

    # Rows beyond this fold into a single "+N more layers" entry. Matches
    # `utils.preview.MAX_LEGEND_ITEMS` so the live window and the exported PNG cut at the
    # same place. None means no limit. `legend(max_items=...)` overrides it per call.
    # Note this applies to *grouped* rows: 100 curves named "Curve 0..99" collapse to one
    # entry long before this limit ever sees them.
    legend_max_items: Optional[int] = 5

    # Scales the whole legend -- swatches, padding and glyphs. Clamped to [0.5, 4.0].
    legend_font_scale: float = 1.0

    # The panel's fill and outline. Both auto-pick their colour from background luminance,
    # exactly as the grid and tick labels do, so there is no dead colour option here.
    legend_background: bool = True
    legend_border: bool = True

    # Tick label format. Empty selects the automatic formatter, which derives its precision
    # from the step and switches to scientific notation outside 1e-4 .. 1e+6. Otherwise:
    # a printf form ("%.2f", "%.1e"), a str.format form ("{:.3g}"), or a bare format spec
    # (".3g"). A form that raises falls back to automatic rather than breaking the frame.
    axis_tick_format: str = ""

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

    def axis_margins(self) -> Tuple[float, float, float, float]:
        """The content inset ``(left, right, bottom, top)`` in pixels.

        The single source of truth for the axis gutters. Every consumer -- the projection
        in :meth:`~glplot.controllers.CameraController.mvp`, its inverse
        :meth:`~glplot.controllers.CameraController.screen_to_world`, and the density
        resolve pass's UV clip box -- reads it from here. They must never disagree.
        """
        return (
            float(self.axis_margin_l),
            float(self.axis_margin_r),
            float(self.axis_margin_b),
            float(self.axis_margin_t),
        )

    def set_axis_margins(self, left: float, right: float, bottom: float, top: float) -> None:
        """Set all four gutters at once. Counterpart to :meth:`axis_margins`."""
        self.axis_margin_l = float(left)
        self.axis_margin_r = float(right)
        self.axis_margin_b = float(bottom)
        self.axis_margin_t = float(top)


@dataclass
class RuntimePolicy:
    # Mutable runtime decisions derived from dataset size and state.
    blending_enabled: bool = True
    current_mode: RenderMode = RenderMode.EXACT
    picking_enabled_this_frame: bool = False
    hud_enabled_this_frame: bool = False
