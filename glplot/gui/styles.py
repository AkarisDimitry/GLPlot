"""Plot styles — presets that restyle the whole scene in one click.

The brief for this module was a complaint, and a precise one: the glow made plots look
*less* clear, not more. What the user asked for is cleaner, clearer plots and a choice of
*looks* — chalk, hand-drawn, publication — not more light. So this module is deliberately
not an effects module. It is a small registry of :class:`PlotStyle` values, plus exactly
one function that applies one (:func:`apply_style`) and one that makes that application
undoable (:func:`style_command`).

**Presets are data.** A :class:`PlotStyle` is a frozen dataclass of leaf values; adding a
look means adding an entry to :data:`STYLES`, not writing code. Nothing here branches on
the preset's name — if it did, the registry would stop being data the first time someone
special-cased "chalk".

**Everything here is pure.** No imgui, no GL, no ``plot.window``: the module imports
headless and every function is unit-testable against a windowless ``GPULinePlot()``. The
GL-adjacent half of the job (dirty flags, re-uploads) is delegated to
:mod:`glplot.gui.layerops`, which owns it.

**Frame safety.** :func:`apply_style` and :func:`restore_state` mutate the live scene, so
they are *queued-command only* (CONTRACT §1.1) exactly like every ``layerops`` function.
:func:`style_command` returns a :class:`~glplot.gui.history.Command` for
``Panel.push_command``, which does the queueing.

Two fields the presets drive do not exist on :class:`~glplot.options.VisualOptions` yet
(``tonemap_index``, ``grain_amount``) and one does not exist on
:class:`~glplot.options.GlowOptions` (``knee``). The post-processing chain already reads
all three through ``getattr`` with a default, and so does this module: a preset sets them
if they are wanted and the value is honoured whether or not the dataclass ever declares
the field. See ``managers/effects.py``'s ``tonemap_index``/``glow_knee``.

**What a preset does not touch.** Per-layer ``alpha``, ``zorder``, ``visible``, ``label``
and data are the user's, not the style's. Colours are only rewritten where the layer's
colour is *uniform*: a scatter carrying per-point colours is carrying information, and a
preset that flattened it would destroy the plot to change its mood.

**A preset reaches 3D too.** A look is not a look if it only applies to half the figures
the library can draw. Three of the fields here land on a 3D scene:
:class:`Axes3DSettings` writes the panel's
:class:`~glplot.core.camera3d.Axes3DOptions` (box, floor, grid, ticks, labels, tick
count, padding), ``light_bg_mode`` already picks the palette
``engine._axis3d_color`` paints the decoration with, and the layer pass colours 3D
layers whose colour is uniform. What a preset deliberately does *not* set on 3D is the
camera: where you are standing is not a style.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..options import BlendMode
from . import layerops
from .history import Command

__all__ = [
    "AUTO_GRID_COLOR",
    "AXES3D_FIELDS",
    "Axes3DSettings",
    "DEFAULT_STYLE_KEY",
    "GlowSettings",
    "HAND_DRAWN_MAX_POINTS",
    "PlotStyle",
    "STYLES",
    "STYLE_KEYS",
    "TickSettings",
    "apply_style",
    "oversized_layers",
    "capture_state",
    "get_style",
    "hand_drawn_layers",
    "is_factory_default",
    "jitter_polyline",
    "restore_state",
    "style_command",
]

RGB = Tuple[float, float, float]
RGBA = Tuple[float, float, float, float]

#: The sentinel ``renderers/axis.py`` reads as "pick the grid ink from the background
#: luminance". A preset with ``grid_color=None`` writes this, i.e. asks for auto contrast.
AUTO_GRID_COLOR: RGB = (0.2, 0.2, 0.2)

#: Above this many points a polyline is left alone by the hand-drawn transform. The
#: transform resamples and re-uploads the whole curve, so the cost is linear in N and paid
#: on the GL thread; at 50k points it is a few milliseconds, and past that it would stall
#: the frame for a wobble too fine to see anyway. Panels must say so rather than hang.
HAND_DRAWN_MAX_POINTS = 50_000

#: Where :func:`hand_drawn_layers` stashes a layer's pre-jitter data, so the transform is
#: reversible and — more importantly — never applied twice on top of itself. Kept under
#: ``layer.metadata``, which the GUI must preserve anyway (CONTRACT §4.4).
_SOURCE_XY_KEY = "_glplot_style_source_xy"

#: Tone-map operator indices understood by ``POST_COMPOSITE_FS`` (mirrors
#: ``managers/effects.py``; duplicated rather than imported because ``effects`` does
#: ``from OpenGL.GL import *`` at module level and would break headless import).
TONEMAP_NONE = 0
TONEMAP_REINHARD = 1
TONEMAP_ACES = 2

#: Marker for "this attribute did not exist before we set it", so undo can remove it
#: again instead of freezing a value the class never declared.
_MISSING = object()


# ----------------------------------------------------------------------------------
# The preset dataclasses
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class GlowSettings:
    """The bloom knobs a preset owns. ``enabled=False`` is the norm, not the exception."""

    enabled: bool = False
    threshold: float = 0.7
    intensity: float = 0.8
    radius_px: float = 6.0
    #: Soft-knee half-width, relative to the threshold. Read via ``getattr`` by
    #: ``EffectManager.glow_knee``; harmless when ``GlowOptions`` has no such field.
    knee: float = 0.5


@dataclass(frozen=True)
class TickSettings:
    """Axis tick decoration: the marks themselves, their length, and their subdivisions.

    A separate object rather than four more fields on :class:`PlotStyle` because a preset
    is allowed to have *no opinion* here (``PlotStyle.ticks = None``), and "no opinion" has
    to be expressible in one place. Ticks are the one piece of axis decoration a user
    routinely sets from code (``gplt.tick_params(length=...)``, ``gplt.minorticks_on()``)
    before ever opening the workstation, so the presets that shipped before this class
    existed keep ``None`` and leave that choice alone; the new ones state a policy, because
    "tick marks instead of a grid" *is* the journal look and "long ticks" *is* what makes a
    slide readable from the back of the room.

    Every field lands on a value ``renderers/axis.py`` reads at draw time
    (``axis.py:358,393,429``).
    """

    #: ``options.axis_show_ticks`` — the major tick marks on the frame.
    show: bool = True
    #: ``options.axis_tick_len_px`` — major tick length in pixels, DPI-scaled by the
    #: renderer. Minor ticks are drawn at a fixed fraction of it.
    length_px: float = 5.0
    #: ``options.axis_minor_ticks`` — subdivisions between the majors.
    minor: bool = False
    #: ``options.axis_minor_subdivisions`` — intervals per major step, so 5 draws four
    #: marks between each pair of majors.
    minor_subdivisions: int = 5


@dataclass(frozen=True)
class Axes3DSettings:
    """What a preset says about the decoration around a **3D** scene.

    Mirrors the drawing half of :class:`~glplot.core.camera3d.Axes3DOptions` field for
    field, and nothing else: the camera (elevation, azimuth, projection, distance) is
    deliberately absent, because where the viewer is standing is not a style and a preset
    that moved the camera would throw away the orbit the user just dialled in.

    The defaults are exactly ``Axes3DOptions``' own, so a preset that says nothing about 3D
    still writes the values the engine would have used anyway — the field is a statement of
    the look, never an accident of the dataclass.
    """

    #: The bounding-box wireframe. Off is the 3D equivalent of ``show_frame=False``.
    show_box: bool = True
    #: The shaded plane at ``z = zmin``. It reads as a table the data sits on; a technical
    #: look usually wants it gone, a presentation look usually wants it for depth.
    show_floor: bool = True
    #: Grid lines ruled across the three back walls — the 3D counterpart of ``show_grid``.
    show_grid: bool = True
    #: Tick marks along the three front edges.
    show_ticks: bool = True
    #: The numbers beside those ticks. Off makes the box a frame rather than a scale, which
    #: is right for a density cloud and wrong for anything anyone has to read a value off.
    show_tick_labels: bool = True
    #: The x/y/z axis titles.
    show_axis_labels: bool = True
    #: Target ticks per axis, or 0 to leave it automatic (the density then follows the
    #: projected length of each axis, so it re-labels as the camera dollies). A preset only
    #: pins this when the look depends on it: fewer = a cleaner box that survives being
    #: shrunk onto a slide; more = a box you can measure against.
    tick_count: int = 0
    #: Padding around the data bounds, as a fraction of each axis' extent. Small crops
    #: tight (a figure), large gives the scene air (a slide).
    pad: float = 0.08


#: The :class:`~glplot.core.camera3d.Axes3DOptions` attributes :func:`apply_style` writes,
#: named once so the apply, the capture and the restore cannot drift apart. Every name here
#: is also a field of :class:`Axes3DSettings`.
AXES3D_FIELDS: Tuple[str, ...] = (
    "show_box",
    "show_floor",
    "show_grid",
    "show_ticks",
    "show_tick_labels",
    "show_axis_labels",
    "tick_count",
    "pad",
)


@dataclass(frozen=True)
class PlotStyle:
    """One complete look, as leaf data.

    Every field is a value some renderer or post pass actually reads (the audit is in
    ``brief_options-style-surface.md``); a field nothing reads would make the preset lie
    about what it changed, which is the same disease as a dead slider.
    """

    key: str
    name: str
    description: str

    # -- background ------------------------------------------------------------
    background: RGB = (1.0, 1.0, 1.0)
    #: ``None`` disables the gradient and leaves the flat ``background`` clear colour.
    gradient: Optional[Tuple[RGB, RGB]] = None

    # -- palette ---------------------------------------------------------------
    #: Cycled over the scene's layers in list order. Never empty.
    palette: Tuple[RGBA, ...] = ((0.0, 0.45, 0.70, 1.0),)
    #: Text/annotation colour. Also the fallback when ``palette`` runs out on a
    #: single-colour element that wants to disappear into the page.
    ink: RGBA = (0.10, 0.10, 0.12, 1.0)

    # -- geometry --------------------------------------------------------------
    line_width: float = 1.5
    point_size: float = 6.0
    point_outline: bool = False
    point_outline_color: RGBA = (1.0, 1.0, 1.0, 1.0)
    point_outline_width: float = 1.0

    # -- scene -----------------------------------------------------------------
    blend_mode: BlendMode = BlendMode.ALPHA
    antialiasing: bool = True
    global_alpha: float = 1.0

    # -- axes ------------------------------------------------------------------
    show_grid: bool = True
    show_frame: bool = True
    show_labels: bool = True
    grid_alpha: float = 0.10
    #: ``None`` means auto-contrast (see :data:`AUTO_GRID_COLOR`).
    grid_color: Optional[RGB] = None
    #: Tick decoration, or ``None`` for "this preset has no opinion, leave the user's".
    #: See :class:`TickSettings` for why the abstention is a value and not a default.
    ticks: Optional[TickSettings] = None

    # -- 3D axis decoration ----------------------------------------------------
    #: What the box around a 3D scene looks like. Always applied — a preset that only
    #: styled 2D would leave half the library's figures looking like a different product.
    axes3d: Axes3DSettings = field(default_factory=Axes3DSettings)

    # -- density ---------------------------------------------------------------
    #: Index into ``shaders.DENSITY_SCHEMES``; the heatmap look that goes with this style.
    colormap_index: int = 5
    light_bg_mode: bool = True

    # -- effects ---------------------------------------------------------------
    glow: GlowSettings = field(default_factory=GlowSettings)
    tonemap: int = TONEMAP_NONE
    #: Static value-noise overlay in the composite pass, 0 = off. Static on purpose: this
    #: is paper grain, not film grain, and an animated one would crawl under the cursor.
    grain: float = 0.0

    # -- hand-drawn geometry ---------------------------------------------------
    hand_drawn: bool = False
    #: Displacement amplitude as a fraction of the layer's own bounding box, so the wobble
    #: looks the same on a curve spanning 1e-9 and one spanning 1e9.
    hand_amplitude: float = 0.0035
    #: Noise wavelength, same normalised units. Low frequency = a drawn line, not fur.
    hand_wavelength: float = 0.045
    #: Seeded, and stable across frames by construction: the jitter is computed once at
    #: apply time and stored. A per-frame reseed would shimmer.
    hand_seed: int = 7


# ----------------------------------------------------------------------------------
# The registry
# ----------------------------------------------------------------------------------

#: Okabe-Ito, minus its yellow (illegible as a thin line on white). Chosen over the
#: matplotlib default cycle because it survives all three common colour-vision
#: deficiencies *and* stays distinguishable in greyscale-ish print.
_CLEAN_PALETTE: Tuple[RGBA, ...] = (
    (0.000, 0.447, 0.698, 1.0),  # #0072B2 blue
    (0.835, 0.369, 0.000, 1.0),  # #D55E00 vermillion
    (0.000, 0.620, 0.451, 1.0),  # #009E73 bluish green
    (0.800, 0.475, 0.655, 1.0),  # #CC79A7 reddish purple
    (0.902, 0.624, 0.000, 1.0),  # #E69F00 orange
    (0.337, 0.706, 0.914, 1.0),  # #56B4E9 sky blue
    (0.100, 0.100, 0.120, 1.0),  # near-black
)

_DARK_PALETTE: Tuple[RGBA, ...] = (
    (0.310, 0.765, 0.969, 1.0),  # #4FC3F7
    (1.000, 0.541, 0.396, 1.0),  # #FF8A65
    (0.506, 0.780, 0.518, 1.0),  # #81C784
    (0.729, 0.408, 0.784, 1.0),  # #BA68C8
    (1.000, 0.835, 0.310, 1.0),  # #FFD54F
    (0.302, 0.816, 0.882, 1.0),  # #4DD0E1
    (0.941, 0.384, 0.573, 1.0),  # #F06292
)

_CHALK_PALETTE: Tuple[RGBA, ...] = (
    (0.949, 0.937, 0.902, 1.0),  # chalk white
    (0.969, 0.773, 0.624, 1.0),  # peach
    (0.659, 0.855, 0.863, 1.0),  # pale cyan
    (0.780, 0.914, 0.690, 1.0),  # pale green
    (0.957, 0.651, 0.753, 1.0),  # pink
    (1.000, 0.886, 0.604, 1.0),  # pale yellow
    (0.765, 0.710, 0.890, 1.0),  # lavender
)

_HAND_PALETTE: Tuple[RGBA, ...] = (
    (0.110, 0.110, 0.110, 1.0),  # pencil
    (0.886, 0.290, 0.200, 1.0),  # #E24A33
    (0.204, 0.541, 0.741, 1.0),  # #348ABD
    (0.596, 0.557, 0.835, 1.0),  # #988ED5
    (0.557, 0.729, 0.259, 1.0),  # #8EBA42
    (0.984, 0.757, 0.369, 1.0),  # #FBC15E
    (0.467, 0.467, 0.467, 1.0),  # #777777
)

#: Dry-erase marker inks. Led by black because a whiteboard sketch's first stroke almost
#: always is, then the four colours every marker set ships with, then two more. Each is a
#: *saturated* ink rather than a pastel: a whiteboard has no tooth to soften a colour, so
#: dry-erase pigment reads bolder than chalk and every entry stays well clear of the page.
_MARKER_PALETTE: Tuple[RGBA, ...] = (
    (0.130, 0.140, 0.160, 1.0),  # marker black
    (0.120, 0.310, 0.720, 1.0),  # marker blue
    (0.800, 0.130, 0.150, 1.0),  # marker red
    (0.110, 0.550, 0.300, 1.0),  # marker green
    (0.470, 0.160, 0.600, 1.0),  # marker purple
    (0.860, 0.440, 0.050, 1.0),  # marker orange
    (0.780, 0.130, 0.470, 1.0),  # marker magenta
)

#: Crayon primaries: the loud, opaque, slightly-uneven colours of a child's drawing. Led by
#: red (the crayon that always wears out first), and kept saturated because wax crayon lays
#: down as solid colour, not a wash. The yellow is pulled towards goldenrod so it survives on
#: the warm manila page rather than dissolving into it, which is the one thing a real crayon
#: yellow does *not* do and the one the eye most forgives.
_KIDS_PALETTE: Tuple[RGBA, ...] = (
    (0.800, 0.130, 0.130, 1.0),  # crayon red
    (0.150, 0.350, 0.800, 1.0),  # crayon blue
    (0.180, 0.600, 0.250, 1.0),  # crayon green
    (0.550, 0.200, 0.680, 1.0),  # crayon purple
    (0.900, 0.250, 0.550, 1.0),  # crayon pink
    (0.950, 0.520, 0.100, 1.0),  # crayon orange
    (0.860, 0.680, 0.060, 1.0),  # crayon goldenrod
)

_NEON_PALETTE: Tuple[RGBA, ...] = (
    (0.000, 0.898, 1.000, 1.0),  # cyan
    (1.000, 0.176, 0.584, 1.0),  # magenta
    (0.486, 1.000, 0.000, 1.0),  # lime
    (1.000, 0.769, 0.000, 1.0),  # amber
    (0.702, 0.533, 1.000, 1.0),  # violet
    (0.094, 1.000, 0.835, 1.0),  # aqua
    (1.000, 0.322, 0.322, 1.0),  # red
)

#: Lightness-separated greys. Distinguishable *because* of the value steps, which is the
#: whole point: a colour palette that merges into three identical greys on a photocopier
#: is not grayscale-safe, however pretty it is on screen.
_PRINT_PALETTE: Tuple[RGBA, ...] = (
    (0.000, 0.000, 0.000, 1.0),
    (0.550, 0.550, 0.550, 1.0),
    (0.275, 0.275, 0.275, 1.0),
    (0.720, 0.720, 0.720, 1.0),
    (0.420, 0.420, 0.420, 1.0),
    (0.150, 0.150, 0.150, 1.0),
    (0.640, 0.640, 0.640, 1.0),
)

#: Nord. Low chroma, high separation — reads as quiet rather than washed out.
_MINIMAL_PALETTE: Tuple[RGBA, ...] = (
    (0.298, 0.337, 0.416, 1.0),  # #4C566A
    (0.749, 0.380, 0.416, 1.0),  # #BF616A
    (0.369, 0.506, 0.675, 1.0),  # #5E81AC
    (0.639, 0.745, 0.549, 1.0),  # #A3BE8C
    (0.816, 0.529, 0.439, 1.0),  # #D08770
    (0.706, 0.557, 0.678, 1.0),  # #B48EAD
    (0.561, 0.737, 0.733, 1.0),  # #8FBCBB
)

#: Dark, desaturated and led by black, because the first series in a journal figure is
#: almost always the one printed alone. Every entry stays under ~55% luminance so a 1.2px
#: stroke still reads at 300dpi on white — the failure mode of a screen palette on paper is
#: a pale line that survives the PDF and vanishes in the issue.
_JOURNAL_PALETTE: Tuple[RGBA, ...] = (
    (0.078, 0.078, 0.094, 1.0),  # near-black
    (0.000, 0.353, 0.612, 1.0),  # #005A9C deep blue
    (0.776, 0.239, 0.110, 1.0),  # #C63D1C brick
    (0.000, 0.451, 0.400, 1.0),  # #007366 pine
    (0.435, 0.278, 0.549, 1.0),  # #6F468C aubergine
    (0.541, 0.451, 0.098, 1.0),  # #8A7319 bronze
    (0.427, 0.435, 0.463, 1.0),  # #6D6F76 slate
)

#: Paul Tol's *muted* qualitative scheme, minus the two entries that go pale on white.
#: Chosen over Okabe-Ito for this preset because Tol's set is separated by lightness as
#: well as hue, so it survives protanopia, deuteranopia, tritanopia *and* a greyscale
#: printer — Okabe-Ito (used by Clean) survives the first three and merges under the last.
_COLORSAFE_PALETTE: Tuple[RGBA, ...] = (
    (0.200, 0.133, 0.533, 1.0),  # #332288 indigo
    (0.800, 0.400, 0.467, 1.0),  # #CC6677 rose
    (0.267, 0.667, 0.600, 1.0),  # #44AA99 teal
    (0.533, 0.133, 0.333, 1.0),  # #882255 wine
    (0.600, 0.600, 0.200, 1.0),  # #999933 olive
    (0.667, 0.267, 0.600, 1.0),  # #AA4499 purple
    (0.067, 0.467, 0.200, 1.0),  # #117733 green
)

#: Fully saturated primaries. A projector's gamut and contrast are both worse than the
#: monitor the figure was made on, and a subtle palette is the first casualty: these are
#: chosen to still be six *different* colours after the lamp, the screen and the room have
#: each taken their cut.
_SLIDE_PALETTE: Tuple[RGBA, ...] = (
    (0.043, 0.435, 0.820, 1.0),  # #0B6FD1 blue
    (0.878, 0.235, 0.122, 1.0),  # #E03C1F red
    (0.106, 0.620, 0.271, 1.0),  # #1B9E45 green
    (0.482, 0.247, 0.749, 1.0),  # #7B3FBF violet
    (0.910, 0.643, 0.000, 1.0),  # #E8A400 gold
    (0.000, 0.627, 0.659, 1.0),  # #00A0A8 teal
    (0.098, 0.098, 0.118, 1.0),  # near-black
)

#: The same six decisions as :data:`_SLIDE_PALETTE`, lifted for a black stage: on a dark
#: page the eye loses saturation before it loses brightness, so these trade chroma for
#: luminance rather than the other way round.
_STAGE_PALETTE: Tuple[RGBA, ...] = (
    (0.310, 0.659, 1.000, 1.0),  # #4FA8FF
    (1.000, 0.478, 0.271, 1.0),  # #FF7A45
    (0.290, 0.871, 0.502, 1.0),  # #4ADE80
    (0.780, 0.573, 0.918, 1.0),  # #C792EA
    (1.000, 0.827, 0.302, 1.0),  # #FFD34D
    (0.302, 0.878, 0.878, 1.0),  # #4DE0E0
    (1.000, 0.420, 0.616, 1.0),  # #FF6B9D
)

#: One Dark, the syntax palette this look borrows wholesale. Deliberately: the point of an
#: IDE theme is that the plot stops being a bright rectangle in the middle of a dark
#: workspace, and matching an existing, heavily-tuned dark scheme gets there faster than
#: inventing a seventh set of accents.
_IDE_PALETTE: Tuple[RGBA, ...] = (
    (0.380, 0.686, 0.937, 1.0),  # #61AFEF blue
    (0.878, 0.424, 0.459, 1.0),  # #E06C75 red
    (0.596, 0.765, 0.475, 1.0),  # #98C379 green
    (0.776, 0.471, 0.867, 1.0),  # #C678DD magenta
    (0.898, 0.753, 0.482, 1.0),  # #E5C07B yellow
    (0.337, 0.714, 0.761, 1.0),  # #56B6C2 cyan
    (0.671, 0.698, 0.749, 1.0),  # #ABB2BF foreground
)

#: Bright, light-toned and few-hued, because under additive blending colour is a budget:
#: two overlapping strokes make a third colour, and a palette of deep saturated inks turns
#: an overplot into mud. These sit high enough on the value axis that a pile-up reads as
#: *density* rather than as a new series.
_DENSE_PALETTE: Tuple[RGBA, ...] = (
    (0.350, 0.850, 1.000, 1.0),  # ice
    (1.000, 0.451, 0.647, 1.0),  # rose
    (0.549, 1.000, 0.600, 1.0),  # mint
    (1.000, 0.800, 0.349, 1.0),  # sand
    (0.749, 0.600, 1.000, 1.0),  # lilac
    (0.400, 1.000, 0.898, 1.0),  # aqua
    (1.000, 0.549, 0.400, 1.0),  # coral
)

#: Cyanotype ink: whites and pale washes, the way a real blueprint's lines are the *absence*
#: of dye. Kept close in hue on purpose — an engineering drawing separates things by
#: linework and annotation, not by rainbow, and a magenta trace on a blueprint reads as a
#: mistake.
_BLUEPRINT_PALETTE: Tuple[RGBA, ...] = (
    (0.902, 0.941, 0.980, 1.0),  # drawing white
    (0.549, 0.792, 0.949, 1.0),  # pale blue
    (1.000, 0.776, 0.443, 1.0),  # amber (the annotation colour)
    (0.643, 0.898, 0.749, 1.0),  # mint
    (0.980, 0.671, 0.671, 1.0),  # salmon
    (0.792, 0.784, 0.949, 1.0),  # periwinkle
    (0.667, 0.831, 0.871, 1.0),  # pale cyan
)


#: Solarized's own accent set, in the order they sit around Ethan Schoonover's published
#: hue wheel (yellow -> orange -> red -> violet -> blue -> cyan -> green), minus magenta:
#: seven is this module's convention, and magenta is the entry whose hue sits nearest red
#: once the wheel is docked down to size.
_SOLARIZED_PALETTE: Tuple[RGBA, ...] = (
    (0.710, 0.537, 0.000, 1.0),  # #B58900 yellow
    (0.796, 0.294, 0.086, 1.0),  # #CB4B16 orange
    (0.863, 0.196, 0.184, 1.0),  # #DC322F red
    (0.424, 0.443, 0.769, 1.0),  # #6C71C4 violet
    (0.149, 0.545, 0.824, 1.0),  # #268BD2 blue
    (0.165, 0.631, 0.596, 1.0),  # #2AA198 cyan
    (0.522, 0.600, 0.000, 1.0),  # #859900 green
)

#: Dracula's full accent set — all seven survive, because the theme itself only ships
#: seven and none of them crowds another once they're off a syntax highlighter and onto a
#: plot, where there's no adjacent black text for two of them to compete with.
_DRACULA_PALETTE: Tuple[RGBA, ...] = (
    (0.741, 0.576, 0.976, 1.0),  # #BD93F9 purple
    (0.314, 0.980, 0.482, 1.0),  # #50FA7B green
    (1.000, 0.475, 0.776, 1.0),  # #FF79C6 pink
    (0.545, 0.914, 0.992, 1.0),  # #8BE9FD cyan
    (0.945, 0.980, 0.549, 1.0),  # #F1FA8C yellow
    (1.000, 0.722, 0.424, 1.0),  # #FFB86C orange
    (1.000, 0.333, 0.333, 1.0),  # #FF5555 red
)

#: Gruvbox's "bright" accent row — the variant the scheme's own source specifies for use
#: against its dark background (over "neutral", which is tuned for the light background
#: instead), so this is the exact row a gruvbox terminal already shows.
_GRUVBOX_PALETTE: Tuple[RGBA, ...] = (
    (0.996, 0.502, 0.098, 1.0),  # #FE8019 orange
    (0.514, 0.647, 0.596, 1.0),  # #83A598 blue
    (0.980, 0.741, 0.184, 1.0),  # #FABD2F yellow
    (0.984, 0.286, 0.204, 1.0),  # #FB4934 red
    (0.557, 0.753, 0.486, 1.0),  # #8EC07C aqua
    (0.827, 0.525, 0.608, 1.0),  # #D3869B purple
    (0.722, 0.733, 0.149, 1.0),  # #B8BB26 green
)


# -- shared 3D decorations -----------------------------------------------------------
# Named rather than repeated, because these are three *policies* and the presets that share
# one should visibly share it.

#: Everything on, at the engine's own defaults. What a look that has no quarrel with the
#: stock 3D box asks for — including Clean, so the default preset stays a true no-op in 3D.
_AXES3D_DEFAULT = Axes3DSettings()

#: The measuring box: no shaded floor (it is decoration, and it hides the data nearest the
#: viewer), a fine grid and tight padding. For anything a reader takes a number off.
_AXES3D_TECHNICAL = Axes3DSettings(show_floor=False, tick_count=6, pad=0.05)

#: The presentation box: floor on for depth, few ticks so the numbers survive being 3m
#: away, and generous padding so the scene is not touching its own frame on the screen.
_AXES3D_PRESENTATION = Axes3DSettings(tick_count=4, pad=0.12)

#: Frame only: the box survives (a 3D scene without one has no depth cues at all), the grid
#: and the numbers do not. For a look whose 2D half has already dropped the grid.
_AXES3D_BARE = Axes3DSettings(
    show_floor=False,
    show_grid=False,
    show_ticks=False,
    show_tick_labels=False,
    tick_count=3,
)


#: Ordered by the question the user is answering, not by age: what goes on paper, what goes
#: in a room, what stays on a screen, what belongs on a drawing board, and last the four
#: looks that are a voice rather than a medium. A gallery sorted by "when we shipped it"
#: makes the reader scan all fourteen every time.
STYLES: Tuple[PlotStyle, ...] = (
    PlotStyle(
        key="clean",
        name="Clean / Publication",
        description="White page, thin colour-blind-safe lines, a grid you have to look for.",
        background=(1.0, 1.0, 1.0),
        palette=_CLEAN_PALETTE,
        ink=(0.10, 0.10, 0.12, 1.0),
        line_width=1.6,
        point_size=5.0,
        blend_mode=BlendMode.ALPHA,
        antialiasing=True,
        global_alpha=1.0,
        show_grid=True,
        show_frame=True,
        show_labels=True,
        grid_alpha=0.08,
        grid_color=None,
        colormap_index=5,  # Ink Fire (White BG)
        light_bg_mode=True,
        glow=GlowSettings(enabled=False),
        tonemap=TONEMAP_NONE,
        axes3d=_AXES3D_DEFAULT,
    ),
    PlotStyle(
        # FOR: the figure that goes into a manuscript, at one column wide, in black and
        # white or close to it. Everything here follows from being read on paper at 8cm:
        # hairline strokes (a 1.6px screen line prints fat), a near-black ink that survives
        # a scanner, tick marks and minor ticks *instead* of a grid — a printed grid is
        # chartjunk that competes with the data for the same thin black ink — and a tight
        # 3D box with no shaded floor.
        #
        # SACRIFICES: it is quiet to the point of frailty. Projected, the 1.2px strokes
        # disappear; on a dark workstation it is a white rectangle in the face; and with
        # more than ~4 series the low-chroma palette starts asking for a legend.
        key="journal",
        name="Journal",
        description="Figure-ready: hairline ink on white, tick marks instead of a grid.",
        background=(1.0, 1.0, 1.0),
        palette=_JOURNAL_PALETTE,
        ink=(0.055, 0.055, 0.070, 1.0),
        line_width=1.2,
        point_size=4.0,
        blend_mode=BlendMode.ALPHA,
        antialiasing=True,
        global_alpha=1.0,
        show_grid=False,
        show_frame=True,
        show_labels=True,
        grid_alpha=0.0,
        grid_color=(0.0, 0.0, 0.0),
        ticks=TickSettings(show=True, length_px=4.0, minor=True, minor_subdivisions=5),
        colormap_index=1,  # Viridis — the perceptually uniform default most journals ask for
        light_bg_mode=True,
        glow=GlowSettings(enabled=False),
        tonemap=TONEMAP_NONE,
        axes3d=_AXES3D_TECHNICAL,
    ),
    PlotStyle(
        # FOR: a figure that has to work for a reader with a colour-vision deficiency, and
        # keep working when they print it. This is not Clean with different colours: hue is
        # demoted from *the* channel to *a* channel. Strokes are 2.4px because judging a hue
        # is much harder on a thin line than a fat one; points are large and white-outlined
        # so overlapping markers stay countable without relying on colour separation; and
        # the density colormap is Viridis, which is monotonic in lightness, rather than a
        # two-hue ramp that a dichromat reads as one colour.
        #
        # SACRIFICES: ink weight. At 2.4px a dense scatter or twenty series turn into a
        # blanket, and the muted Tol palette is less punchy than Clean's Okabe-Ito.
        key="colorsafe",
        name="Colour-vision safe",
        description="Tol's muted set, thick strokes, outlined points. Hue is never alone.",
        background=(1.0, 1.0, 1.0),
        palette=_COLORSAFE_PALETTE,
        ink=(0.10, 0.10, 0.12, 1.0),
        line_width=2.4,
        point_size=8.0,
        point_outline=True,
        point_outline_color=(1.0, 1.0, 1.0, 1.0),
        point_outline_width=1.4,
        blend_mode=BlendMode.ALPHA,
        antialiasing=True,
        global_alpha=1.0,
        show_grid=True,
        show_frame=True,
        show_labels=True,
        grid_alpha=0.07,
        grid_color=None,
        ticks=TickSettings(show=True, length_px=5.0, minor=False),
        colormap_index=1,  # Viridis: monotonic lightness, safe under all three deficiencies
        light_bg_mode=True,
        glow=GlowSettings(enabled=False),
        tonemap=TONEMAP_NONE,
        axes3d=_AXES3D_DEFAULT,
    ),
    PlotStyle(
        key="print",
        name="Print",
        description="Greyscale-safe: separated by lightness, so a photocopier keeps them apart.",
        background=(1.0, 1.0, 1.0),
        palette=_PRINT_PALETTE,
        ink=(0.0, 0.0, 0.0, 1.0),
        line_width=1.8,
        point_size=6.0,
        point_outline=True,
        point_outline_color=(1.0, 1.0, 1.0, 1.0),
        point_outline_width=1.2,
        blend_mode=BlendMode.ALPHA,
        antialiasing=True,
        show_grid=True,
        grid_alpha=0.06,
        grid_color=(0.0, 0.0, 0.0),
        colormap_index=7,  # Grayscale
        light_bg_mode=True,
        glow=GlowSettings(enabled=False),
        tonemap=TONEMAP_NONE,
        # No shaded floor: on a greyscale printer the floor is a flat grey wash that eats
        # the value range the palette is using to keep the series apart.
        axes3d=_AXES3D_TECHNICAL,
    ),
    PlotStyle(
        # FOR: a lit room and a projector. Every dimension is pushed one step past what
        # looks right on a monitor 50cm away, because that is what survives 10m, a washed-out
        # lamp and a screen with half the contrast of the display it was authored on: 3.4px
        # strokes, 11px points with a white halo so they read against a line, saturated
        # primaries, long 9px ticks, and few of them.
        #
        # SACRIFICES: density and nuance. Two nearby series merge into one fat band, a
        # thousand-point scatter becomes a solid shape, and printed it is a waste of toner.
        key="slide",
        name="Slide (light)",
        description="Projector-proof: heavy strokes, big outlined points, long ticks.",
        background=(1.0, 1.0, 1.0),
        palette=_SLIDE_PALETTE,
        ink=(0.060, 0.060, 0.080, 1.0),
        line_width=3.4,
        point_size=11.0,
        point_outline=True,
        point_outline_color=(1.0, 1.0, 1.0, 1.0),
        point_outline_width=1.6,
        blend_mode=BlendMode.ALPHA,
        antialiasing=True,
        global_alpha=1.0,
        show_grid=True,
        show_frame=True,
        show_labels=True,
        grid_alpha=0.14,
        grid_color=None,
        ticks=TickSettings(show=True, length_px=9.0, minor=False),
        colormap_index=4,  # Turbo: the most separable ramp once a projector has flattened it
        light_bg_mode=True,
        glow=GlowSettings(enabled=False),
        tonemap=TONEMAP_NONE,
        axes3d=_AXES3D_PRESENTATION,
    ),
    PlotStyle(
        # FOR: the same room with the lights down, and for a dark slide template. Identical
        # geometry to Slide — the room did not change, only the page — but the palette
        # trades chroma for luminance, and the point halo is the background colour rather
        # than white, so overlapping markers still separate.
        #
        # The page is 0.05 rather than pure black on purpose: a projector cannot render
        # black, and authoring against 0.0 hides banding that the room will happily show.
        #
        # SACRIFICES: everything Slide sacrifices, plus print — a dark-page figure sent to a
        # printer either comes back a black rectangle or gets silently inverted.
        key="stage",
        name="Stage (dark slide)",
        description="The slide look at night: luminous ink on a stage-black page.",
        background=(0.050, 0.052, 0.060),
        palette=_STAGE_PALETTE,
        ink=(0.950, 0.960, 0.980, 1.0),
        line_width=3.4,
        point_size=11.0,
        point_outline=True,
        point_outline_color=(0.050, 0.052, 0.060, 1.0),
        point_outline_width=1.6,
        blend_mode=BlendMode.ALPHA,
        antialiasing=True,
        global_alpha=1.0,
        show_grid=True,
        show_frame=True,
        show_labels=True,
        grid_alpha=0.20,
        grid_color=None,
        ticks=TickSettings(show=True, length_px=9.0, minor=False),
        colormap_index=2,  # Plasma: bright across its whole range, so it holds up unlit
        light_bg_mode=False,
        glow=GlowSettings(enabled=False),
        tonemap=TONEMAP_NONE,
        axes3d=_AXES3D_PRESENTATION,
    ),
    PlotStyle(
        key="dark",
        name="Dark",
        description="Charcoal page, luminous palette. The same plot, at night.",
        background=(0.086, 0.098, 0.118),
        palette=_DARK_PALETTE,
        ink=(0.88, 0.90, 0.94, 1.0),
        line_width=1.6,
        point_size=5.0,
        blend_mode=BlendMode.ALPHA,
        antialiasing=True,
        show_grid=True,
        grid_alpha=0.16,
        grid_color=None,
        colormap_index=1,  # Viridis
        light_bg_mode=False,
        glow=GlowSettings(enabled=False),
        tonemap=TONEMAP_NONE,
        axes3d=_AXES3D_DEFAULT,
    ),
    PlotStyle(
        # FOR: the plot you keep open all day beside an editor. Dark is a *page*; this is a
        # *panel*: One Dark's charcoal and accents, so the figure sits in a dark workspace
        # without being the brightest thing on the desk, and an explicit low-contrast grid
        # instead of the auto one, which picks a light grey that glares at 2am.
        #
        # SACRIFICES: it is tuned for a calibrated monitor in a dim room. Projected or
        # printed it is muddy, and the deliberately quiet ink is too quiet for a slide.
        key="ide",
        name="Workstation",
        description="One-dark workstation: calm accents on charcoal, a quiet grid.",
        background=(0.157, 0.173, 0.212),
        palette=_IDE_PALETTE,
        ink=(0.671, 0.698, 0.749, 1.0),
        line_width=1.8,
        point_size=5.5,
        blend_mode=BlendMode.ALPHA,
        antialiasing=True,
        global_alpha=1.0,
        show_grid=True,
        show_frame=True,
        show_labels=True,
        grid_alpha=0.14,
        grid_color=(0.36, 0.40, 0.47),
        ticks=TickSettings(show=True, length_px=5.0, minor=False),
        colormap_index=1,  # Viridis
        light_bg_mode=False,
        glow=GlowSettings(enabled=False),
        tonemap=TONEMAP_NONE,
        axes3d=_AXES3D_DEFAULT,
    ),
    PlotStyle(
        # FOR: a million points, where the answer is the shape of the cloud and not any one
        # series. Decoration is stripped to nothing — no frame, no grid, quiet ink — because
        # at that scale every drawn pixel that is not data is in the way. Hairline strokes,
        # 2px points and ADDITIVE blending at 45% turn overlap into brightness, so a pile-up
        # reads as density instead of clipping to a flat blob.
        #
        # No tone map on purpose: Reinhard would be the honest operator for a genuine
        # overplot, and it also halves the midtones of the 50-point plot someone will
        # inevitably try this on first. The look has to be legible at both ends.
        #
        # SACRIFICES: individual series. Additive blending mixes colours, so two crossing
        # curves make a third; a single faint line looks underexposed; and this is the wrong
        # answer for any plot small enough to read point by point.
        key="dense",
        name="High density",
        description="Nothing but data: hairlines, additive light, no frame, no grid.",
        background=(0.020, 0.020, 0.028),
        palette=_DENSE_PALETTE,
        ink=(0.750, 0.780, 0.840, 1.0),
        line_width=1.0,
        # 3px and 55% are the floor found by rendering this preset onto a 15-point scatter:
        # below them a *sparse* plot reads as underexposed rather than as dense, and a look
        # that is only legible at its design scale is a trap.
        point_size=3.0,
        blend_mode=BlendMode.ADDITIVE,
        antialiasing=True,
        global_alpha=0.55,
        show_grid=False,
        show_frame=False,
        show_labels=True,
        grid_alpha=0.0,
        grid_color=None,
        # Ticks stay: with no frame and no grid they are the only thing left that says what
        # the axis means, and they cost four short marks.
        ticks=TickSettings(show=True, length_px=4.0, minor=False),
        colormap_index=1,  # Viridis — the scheme density mode is actually for
        light_bg_mode=False,
        glow=GlowSettings(enabled=False),
        tonemap=TONEMAP_NONE,
        # The box without its furniture: a cloud does not need a floor or wall grids, and
        # tick numbers over a point cloud are unreadable anyway.
        axes3d=_AXES3D_BARE,
    ),
    PlotStyle(
        # FOR: engineering drawings, CAD-adjacent figures, anything measured. The cyanotype
        # page and white linework are the recognisable half; the working half is the grid,
        # which is at 0.30 alpha with minor subdivisions and an explicit pale-blue ink —
        # here the grid is not chartjunk, it is the instrument, and a 6-tick box with tight
        # padding is what you read coordinates off.
        #
        # SACRIFICES: the strong page hue tints every judgement of colour, so a plot whose
        # colours mean something (a colormapped scatter) should not wear this. And the loud
        # grid is exactly wrong for a dense scene — it competes with the data by design.
        key="blueprint",
        name="Blueprint",
        description="Cyanotype: white linework, minor ticks, a grid you're meant to use.",
        background=(0.043, 0.180, 0.318),
        palette=_BLUEPRINT_PALETTE,
        ink=(0.902, 0.941, 0.980, 1.0),
        line_width=1.6,
        point_size=5.0,
        blend_mode=BlendMode.ALPHA,
        antialiasing=True,
        global_alpha=1.0,
        show_grid=True,
        show_frame=True,
        show_labels=True,
        grid_alpha=0.30,
        grid_color=(0.62, 0.80, 0.95),
        ticks=TickSettings(show=True, length_px=6.0, minor=True, minor_subdivisions=5),
        colormap_index=10,  # Cool — stays inside the drawing's own blue-cyan world
        light_bg_mode=False,
        glow=GlowSettings(enabled=False),
        tonemap=TONEMAP_NONE,
        axes3d=_AXES3D_TECHNICAL,
    ),
    PlotStyle(
        key="minimal",
        name="Minimal",
        description="No grid, no frame. Just the data and its labels.",
        background=(0.988, 0.988, 0.984),
        palette=_MINIMAL_PALETTE,
        ink=(0.18, 0.20, 0.25, 1.0),
        line_width=1.8,
        point_size=6.0,
        blend_mode=BlendMode.ALPHA,
        antialiasing=True,
        show_grid=False,
        show_frame=False,
        show_labels=True,
        grid_alpha=0.0,
        colormap_index=1,
        light_bg_mode=True,
        glow=GlowSettings(enabled=False),
        tonemap=TONEMAP_NONE,
        # The 2D half already dropped the grid and the frame; the 3D half drops the floor,
        # the wall grids and the tick numbers and keeps the box, which is the only depth cue
        # a 3D scene has.
        axes3d=_AXES3D_BARE,
    ),
    PlotStyle(
        key="chalk",
        name="Chalk",
        description="Slate board, soft pastel strokes, a dusting of grain.",
        background=(0.118, 0.161, 0.149),
        gradient=((0.137, 0.184, 0.169), (0.094, 0.129, 0.122)),
        palette=_CHALK_PALETTE,
        ink=(0.949, 0.937, 0.902, 1.0),
        line_width=2.4,
        point_size=7.0,
        blend_mode=BlendMode.ALPHA,
        antialiasing=True,
        global_alpha=0.92,
        show_grid=True,
        show_frame=False,
        grid_alpha=0.12,
        grid_color=(0.80, 0.82, 0.78),
        colormap_index=10,  # Cool
        light_bg_mode=False,
        glow=GlowSettings(enabled=False),
        tonemap=TONEMAP_NONE,
        grain=0.055,
        # Chalk keeps the full box: the whole look is *drawn on a surface*, and the floor
        # and wall grids are what make a 3D scene read as drawn on one rather than floating.
        axes3d=_AXES3D_DEFAULT,
    ),
    PlotStyle(
        # FOR: the whiteboard-sketch look — a quick, bold explanation, the kind drawn in a
        # standup or a lecture. The recognisable half is the near-white glossy board and the
        # saturated dry-erase inks; the working half is that it is *bold and clean* where
        # Chalk is soft and Hand-drawn is shaky. Strokes are 3.2px (a marker is a fat nib),
        # 90% opaque so crossing strokes darken the way real ink does, and there is no grid
        # or frame because a whiteboard starts blank.
        #
        # SACRIFICES: with no grid and one flat page colour it is a poor instrument — you
        # sketch a relationship on it, you do not read coordinates off it. And the bold nib
        # turns a dense scatter into a solid mass, same as every other thick-stroke look.
        key="marker",
        name="Whiteboard marker",
        description="Bold dry-erase ink on a blank board. A sketch, not a measurement.",
        background=(0.965, 0.975, 0.975),
        palette=_MARKER_PALETTE,
        ink=(0.130, 0.140, 0.160, 1.0),
        line_width=3.2,
        point_size=8.5,
        blend_mode=BlendMode.ALPHA,
        antialiasing=True,
        global_alpha=0.90,
        show_grid=False,
        show_frame=False,
        show_labels=True,
        grid_alpha=0.0,
        colormap_index=5,  # Ink Fire (White BG)
        light_bg_mode=True,
        glow=GlowSettings(enabled=False),
        tonemap=TONEMAP_NONE,
        # A faint dry-erase tooth. A real board is glossier than slate, so this is a third of
        # Chalk's grain — just enough to keep the page from reading as a flat digital white.
        grain=0.022,
        # A whiteboard sketch is drawn on a blank board: keep the box for depth, drop the
        # ruled walls and the numbers, exactly as the 2D half drops its grid and frame.
        axes3d=_AXES3D_BARE,
    ),
    PlotStyle(
        key="hand",
        name="Hand-drawn",
        description="Paper, no grid, and every line redrawn with a shaky hand.",
        background=(0.988, 0.980, 0.965),
        palette=_HAND_PALETTE,
        ink=(0.11, 0.11, 0.11, 1.0),
        line_width=2.6,
        point_size=8.0,
        blend_mode=BlendMode.ALPHA,
        antialiasing=True,
        show_grid=False,
        show_frame=False,
        show_labels=True,
        grid_alpha=0.06,
        colormap_index=5,
        light_bg_mode=True,
        glow=GlowSettings(enabled=False),
        tonemap=TONEMAP_NONE,
        hand_drawn=True,
        hand_amplitude=0.0035,
        hand_wavelength=0.045,
        hand_seed=7,
        # The jitter is a 2D transform, so the 3D half of this look is carried entirely by
        # what it leaves out: paper, a bare box, no ruled walls.
        axes3d=_AXES3D_BARE,
    ),
    PlotStyle(
        # FOR: the crayon-drawing look — loud, chunky and cheerful. It is Hand-drawn turned
        # up: the same shaky-hand jitter but *coarser* (a child's wobble is bigger and
        # slower than an adult sketching a figure), fat 4.5px strokes, big 13px dots, and a
        # bright opaque primary palette on warm manila paper. The waxy tooth of crayon is the
        # grain; the wobble is what stops any line being a ruler-straight adult line.
        #
        # SACRIFICES: precision, entirely and on purpose. Fat wobbling strokes cannot carry a
        # dense plot or a value anyone measures, and the jitter — like Hand-drawn's — resamples
        # each polyline, so it steps aside above ``HAND_DRAWN_MAX_POINTS`` and leaves big
        # curves straight.
        key="kids",
        name="Crayon",
        description="Wax crayon on manila: fat wobbling strokes in loud primaries.",
        background=(0.990, 0.960, 0.870),
        palette=_KIDS_PALETTE,
        ink=(0.200, 0.160, 0.130, 1.0),
        line_width=4.5,
        point_size=13.0,
        point_outline=True,
        # A waxy dark rim, not a white halo: a crayon dot has a darker edge where the wax
        # piles up, and the rim is what keeps two overlapping dots countable on the busy page.
        point_outline_color=(0.220, 0.170, 0.120, 1.0),
        point_outline_width=1.6,
        blend_mode=BlendMode.ALPHA,
        antialiasing=True,
        global_alpha=1.0,  # crayon is opaque; there is no wash in a child's drawing
        show_grid=False,
        show_frame=False,
        show_labels=True,
        grid_alpha=0.0,
        colormap_index=4,  # Turbo — the rainbow a kid actually reaches for
        light_bg_mode=True,
        glow=GlowSettings(enabled=False),
        tonemap=TONEMAP_NONE,
        grain=0.05,  # the tooth of wax on paper
        hand_drawn=True,
        # Coarser and larger than Hand-drawn: a bigger amplitude for a less steady hand and a
        # longer wavelength so the wobble is a child's slow sway, not a fine tremor. A
        # different seed so the two looks do not wobble identically.
        hand_amplitude=0.0080,
        hand_wavelength=0.070,
        hand_seed=11,
        # Same reasoning as Hand-drawn: paper, a bare box, no ruled walls.
        axes3d=_AXES3D_BARE,
    ),
    PlotStyle(
        key="neon",
        name="Neon",
        description="Black, additive, and glowing. The look this build shipped with.",
        background=(0.0, 0.0, 0.0),
        palette=_NEON_PALETTE,
        ink=(0.90, 0.95, 1.0, 1.0),
        line_width=1.4,
        point_size=5.0,
        blend_mode=BlendMode.ADDITIVE,
        antialiasing=True,
        show_grid=True,
        grid_alpha=0.12,
        grid_color=None,
        colormap_index=9,  # Hot
        light_bg_mode=False,
        # ACES, not Reinhard: additive neon overshoots 1.0 constantly, and ACES rolls the
        # highlights off instead of halving the midtones the way Reinhard does. This is
        # the one preset that asks for a tone map, and it asks explicitly.
        glow=GlowSettings(enabled=True, threshold=0.65, intensity=0.85, radius_px=8.0, knee=0.5),
        tonemap=TONEMAP_ACES,
        axes3d=_AXES3D_DEFAULT,
    ),
    PlotStyle(
        # FOR: a developer who already lives in Solarized — vim, tmux, a terminal — and
        # wants the plot to stop being the one window that breaks the theme. The palette is
        # Ethan Schoonover's own accent set at its own published hex values, not a re-tuned
        # copy, and the ink is base1, the exact tone the Solarized spec itself names for
        # body text on its dark background.
        #
        # SACRIFICES: Solarized was tuned for terminal text at 10-14pt, not a data line —
        # its signature restraint (every accent close in *lightness*, by design, so nothing
        # shouts over the prose) means a 12-series plot asks for a legend sooner than
        # Slide's loud primaries would.
        key="solarized",
        name="Solarized",
        description="Ethan Schoonover's precision palette: a teal-navy page, muted named accents.",
        background=(0.000, 0.169, 0.212),  # base03 #002B36
        palette=_SOLARIZED_PALETTE,
        ink=(0.576, 0.631, 0.631, 1.0),  # base1 #93A1A1 — Solarized's own dark-mode body text
        line_width=1.7,
        point_size=5.5,
        blend_mode=BlendMode.ALPHA,
        antialiasing=True,
        global_alpha=1.0,
        show_grid=True,
        show_frame=True,
        show_labels=True,
        grid_alpha=0.16,
        grid_color=(0.345, 0.431, 0.459),  # base01 #586E75
        colormap_index=8,  # Ocean — stays in the palette's own teal-blue register
        light_bg_mode=False,
        glow=GlowSettings(enabled=False),
        tonemap=TONEMAP_NONE,
        axes3d=_AXES3D_DEFAULT,
    ),
    PlotStyle(
        # FOR: the look this build's audience will recognise on sight — Dracula is one of
        # the most-installed themes on every editor marketplace it ships for, so borrowing
        # it wholesale (the way Workstation borrows One Dark) buys instant familiarity
        # instead of inventing an eighth "dark theme" nobody asked for. All seven of the
        # theme's own accents survive; nothing here is re-tuned.
        #
        # SACRIFICES: the palette is built to sit *behind* syntax-highlighted text, not to
        # carry a plot on its own — pink and purple, its two loudest colours, are close
        # enough in lightness that a dense multi-series plot wants point outlines to tell
        # them apart, and this preset doesn't turn them on.
        key="dracula",
        name="Dracula",
        description="The web's best-loved dark theme: violet night, candy-bright accents.",
        background=(0.157, 0.165, 0.212),  # #282A36
        palette=_DRACULA_PALETTE,
        ink=(0.973, 0.973, 0.949, 1.0),  # #F8F8F2 foreground
        line_width=1.8,
        point_size=6.0,
        blend_mode=BlendMode.ALPHA,
        antialiasing=True,
        global_alpha=1.0,
        show_grid=True,
        show_frame=True,
        show_labels=True,
        grid_alpha=0.15,
        grid_color=(0.384, 0.447, 0.643),  # #6272A4 "Comment"
        colormap_index=6,  # Magma — echoes the palette's own purple-pink-orange range
        light_bg_mode=False,
        glow=GlowSettings(enabled=False),
        tonemap=TONEMAP_NONE,
        axes3d=_AXES3D_DEFAULT,
    ),
    PlotStyle(
        # FOR: the "retro groove" look, warm where every other dark preset here is cool —
        # Dark, Workstation, Solarized and Dracula all sit in blue-grey or blue-black;
        # Gruvbox's neutrals are true warm grey, unfiltered by any blue tint, and the
        # accents are the theme's own "bright" row, the variant its source tunes
        # specifically to read against this background rather than a re-mix.
        #
        # SACRIFICES: the warmth. On a cool monitor profile the whole page can drift toward
        # looking merely under-lit rather than intentionally warm, and the low-saturation
        # neutrals it's famous for mean the grid needs its own colour (below) rather than
        # the module's usual auto-contrast, which would pick a grey too close to the page.
        key="gruvbox",
        name="Gruvbox",
        description="Retro-groove neutrals: warm charcoal, sun-baked orange, olive and aqua.",
        background=(0.157, 0.157, 0.157),  # #282828 dark0
        palette=_GRUVBOX_PALETTE,
        ink=(0.922, 0.859, 0.698, 1.0),  # #EBDBB2 light1
        line_width=1.8,
        point_size=6.0,
        blend_mode=BlendMode.ALPHA,
        antialiasing=True,
        global_alpha=1.0,
        show_grid=True,
        show_frame=True,
        show_labels=True,
        grid_alpha=0.16,
        grid_color=(0.314, 0.286, 0.271),  # #504945 dark2
        colormap_index=3,  # Inferno — echoes the palette's own black-to-orange warmth
        light_bg_mode=False,
        glow=GlowSettings(enabled=False),
        tonemap=TONEMAP_NONE,
        axes3d=_AXES3D_DEFAULT,
    ),
)

#: The look the workstation opens with, per the user's steer ("plots más limpios").
DEFAULT_STYLE_KEY = "clean"

STYLE_KEYS: Tuple[str, ...] = tuple(s.key for s in STYLES)

_BY_KEY: Dict[str, PlotStyle] = {s.key: s for s in STYLES}


def get_style(key: str) -> PlotStyle:
    """The preset named ``key``. Raises ``KeyError`` on an unknown key — never guesses."""
    return _BY_KEY[str(key)]


# ----------------------------------------------------------------------------------
# Hand-drawn geometry
# ----------------------------------------------------------------------------------


def _value_noise(t: np.ndarray, *, wavelength: float, seed: int) -> np.ndarray:
    """Smooth, seeded, band-limited noise over ``t``, in ``[-1, 1]``.

    Classic value noise: random values on a lattice of spacing ``wavelength``, joined with
    a smoothstep. Deterministic in ``seed`` — the same seed gives the same wobble forever,
    which is the whole reason the hand-drawn transform can live in a 60fps loop without
    boiling. Two octaves, because one octave reads as a sine wave rather than a hand.
    """
    span = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    out = np.zeros(len(t), dtype=np.float64)
    amplitude = 1.0
    total = 0.0
    for octave in range(2):
        wl = max(wavelength * (0.5**octave), 1e-6)
        count = int(span / wl) + 4
        rng = np.random.default_rng((int(seed) * 1000003 + octave) & 0xFFFFFFFF)
        lattice = rng.uniform(-1.0, 1.0, size=count + 2)

        u = (t - t[0]) / wl
        i = np.clip(np.floor(u).astype(np.intp), 0, count - 1)
        f = u - i
        f = f * f * (3.0 - 2.0 * f)  # smoothstep: C1-continuous across every lattice cell
        out += amplitude * (lattice[i] * (1.0 - f) + lattice[i + 1] * f)
        total += amplitude
        amplitude *= 0.5
    return out / max(total, 1e-9)


def jitter_polyline(
    x: Any,
    y: Any,
    *,
    amplitude: float = 0.0035,
    wavelength: float = 0.045,
    seed: int = 7,
    max_points: int = HAND_DRAWN_MAX_POINTS,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """A hand-drawn copy of a polyline, or ``None`` when it is too big to be worth it.

    Geometry, not a shader: the curve is resampled by arc length and each vertex is pushed
    along the local *normal* by smooth low-frequency noise. Normal displacement is what
    makes it read as a drawn stroke — displacing along x/y instead wobbles the parametrisation
    and looks like jitter noise on an oscilloscope.

    The work happens in the curve's own normalised bounding box, so ``amplitude`` and
    ``wavelength`` are fractions of the plot rather than data units: a curve spanning
    nanometres and one spanning parsecs get the same-looking hand.

    ``None`` (rather than a slow answer) above ``max_points``: the transform is O(N) on the
    GL thread and the caller is expected to say "too big" in the UI, per SPEC_FIX P1-F.

    Pure: same inputs, same output, every time. Never touches the engine's renderers, which
    stay deterministic.
    """
    xs = np.asarray(x, dtype=np.float64).reshape(-1)
    ys = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(xs) != len(ys):
        raise ValueError(f"x and y must be the same length, got {len(xs)} and {len(ys)}")
    if len(xs) < 2:
        return xs.astype(np.float32), ys.astype(np.float32)
    if len(xs) > int(max_points):
        return None

    finite = np.isfinite(xs) & np.isfinite(ys)
    if not bool(np.all(finite)):
        # A gap (NaN) is a real feature of a polyline; resampling across it would draw a
        # line where the author asked for a hole. Refuse rather than lie.
        return None

    x_lo, x_hi = float(xs.min()), float(xs.max())
    y_lo, y_hi = float(ys.min()), float(ys.max())
    x_span = x_hi - x_lo
    y_span = y_hi - y_lo
    sx = x_span if x_span > 0.0 else 1.0
    sy = y_span if y_span > 0.0 else 1.0

    nx = (xs - x_lo) / sx
    ny = (ys - y_lo) / sy

    seg = np.hypot(np.diff(nx), np.diff(ny))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    length = float(arc[-1])
    if not np.isfinite(length) or length <= 0.0:
        return xs.astype(np.float32), ys.astype(np.float32)

    # Resample: the wobble needs ~8 vertices per noise wavelength to be a curve rather than
    # a polygon, and a 2-point straight line has nowhere to put one otherwise.
    spacing = max(wavelength / 8.0, 1e-6)
    count = int(min(float(max_points), max(float(len(xs)), length / spacing + 2.0)))
    s = np.linspace(0.0, length, count)
    # arc is non-decreasing but can repeat on duplicate points; np.interp takes the last
    # match, which is what we want (a duplicate vertex contributes no arc).
    rx = np.interp(s, arc, nx)
    ry = np.interp(s, arc, ny)

    tx = np.gradient(rx)
    ty = np.gradient(ry)
    norm = np.hypot(tx, ty)
    norm[norm <= 1e-12] = 1.0
    # Left normal of the unit tangent.
    px = -ty / norm
    py = tx / norm

    noise = _value_noise(s, wavelength=wavelength, seed=int(seed))
    # Taper to zero at both ends: a stroke that starts mid-wobble looks like a broken line,
    # and for a closed-ish curve it keeps the join from splitting.
    ramp = max(length * 0.02, 1e-9)
    taper = np.clip(np.minimum(s, length - s) / ramp, 0.0, 1.0)
    disp = noise * float(amplitude) * taper

    jx = rx + px * disp
    jy = ry + py * disp

    return (
        np.ascontiguousarray(jx * sx + x_lo, dtype=np.float32),
        np.ascontiguousarray(jy * sy + y_lo, dtype=np.float32),
    )


def _hand_drawn_candidates(plot: Any) -> List[Any]:
    """Layers the hand-drawn transform applies to: plain 2D polylines, nothing else.

    Scatter has no path to wobble along, ``line_family``'s ``ab`` columns are not an (x, y)
    curve, and a 3D mesh's normals are not this module's business.
    """
    return [
        layer
        for layer in list(getattr(plot.scene, "layers", []))
        if str(getattr(layer, "layer_type", "")) == "polyline"
    ]


def _source_xy(layer: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The layer's pre-jitter data, if it is currently jittered."""
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    stored = metadata.get(_SOURCE_XY_KEY)
    if not isinstance(stored, tuple) or len(stored) != 2:
        return None
    return stored


def oversized_layers(plot: Any) -> List[str]:
    """Labels of the polylines the hand-drawn transform would refuse, as a read-only probe.

    Pure and side-effect free, so a panel may call it straight from a draw callback to warn
    *before* the user clicks, rather than reporting a silent no-op afterwards.
    """
    names: List[str] = []
    for layer in _hand_drawn_candidates(plot):
        source = _source_xy(layer)
        count = len(source[0]) if source is not None else len(getattr(layer, "pts", ()))
        if count > HAND_DRAWN_MAX_POINTS:
            names.append(str(getattr(layer, "label", "")) or "(unnamed)")
    return names


def hand_drawn_layers(plot: Any, style: PlotStyle) -> int:
    """Apply (or remove) the hand-drawn transform across the scene. Returns #skipped.

    **Queued commands only.** Idempotent by construction: the untouched data is stashed
    under ``metadata`` on the first pass and every later pass re-jitters *that*, so
    switching Hand-drawn → Chalk → Hand-drawn does not compound the wobble, and any other
    preset restores the true geometry.

    The return value is the number of layers left alone for being larger than
    :data:`HAND_DRAWN_MAX_POINTS` (or non-finite), which the panel reports rather than
    silently doing nothing.
    """
    skipped = 0
    for layer in _hand_drawn_candidates(plot):
        source = _source_xy(layer)
        if not style.hand_drawn:
            if source is not None:
                layerops.update_layer_xy(plot, layer, source[0], source[1])
                layer.metadata.pop(_SOURCE_XY_KEY, None)
            continue

        current = layerops.layer_xy(layer)
        if current is None:
            continue
        base = source if source is not None else current
        jittered = jitter_polyline(
            base[0],
            base[1],
            amplitude=style.hand_amplitude,
            wavelength=style.hand_wavelength,
            seed=style.hand_seed,
        )
        if jittered is None:
            skipped += 1
            continue
        if source is None:
            layer.metadata[_SOURCE_XY_KEY] = (base[0], base[1])
        layerops.update_layer_xy(plot, layer, jittered[0], jittered[1])
    return skipped


# ----------------------------------------------------------------------------------
# Applying a style
# ----------------------------------------------------------------------------------


def _uniform_color(colors: Any) -> Optional[RGBA]:
    """The single RGBA of a uniform per-element colour array, else ``None``.

    ``None`` means "this layer's colours carry information" — a colormapped scatter, a
    per-index gradient on a line family — and a preset must not overwrite them. Uniform
    arrays are the ones that are merely *a colour*, and those are the style's to change.
    """
    if colors is None:
        return None
    arr = np.asarray(colors)
    if arr.ndim == 1 and arr.shape[0] in (3, 4):
        vals = [float(c) for c in arr]
        if len(vals) == 3:
            vals.append(1.0)
        return (vals[0], vals[1], vals[2], vals[3])
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != 4:
        return None
    if not bool(np.all(arr == arr[0])):
        return None
    row = arr[0]
    return (float(row[0]), float(row[1]), float(row[2]), float(row[3]))


def _repaint_uniform_colors(layer: Any, rgba: RGBA) -> None:
    """Repaint a uniform per-element colour array, preserving its shape exactly.

    Shape is load-bearing: ``scatter.py:123`` sets the draw count from ``len(layer.pts)``
    but uploads ``layer.colors`` at whatever length it has, so turning a ``(4,)`` colour
    into a ``(1, 4)`` one would leave the colour VBO one row long and the draw call
    fetching past its end. Callers must have checked :func:`_uniform_color` first.
    """
    arr = np.asarray(layer.colors)
    new = np.asarray(rgba, dtype=np.float32)
    if arr.ndim == 1:
        new = new[: arr.shape[0]]
    else:
        new = np.tile(new, (arr.shape[0], 1))
    layer.colors = np.ascontiguousarray(new, dtype=np.float32)
    layer.dirty.gpu_dirty = True


def _set_optional(target: Any, name: str, value: Any) -> None:
    """Set a field that the dataclass may not declare yet (``tonemap_index`` et al.)."""
    setattr(target, name, value)


def _get_optional(target: Any, name: str) -> Any:
    """Read an optional field, returning :data:`_MISSING` when it is genuinely absent."""
    return getattr(target, name, _MISSING)


def _restore_optional(target: Any, name: str, value: Any) -> None:
    """Undo counterpart of :func:`_set_optional`, including "it was not there"."""
    if value is _MISSING:
        try:
            delattr(target, name)
        except AttributeError:
            pass
        return
    setattr(target, name, value)


def _style_color_for(layer: Any, index: int, style: PlotStyle) -> RGBA:
    """The palette entry this layer gets: cycled by position, ink for text."""
    if str(getattr(layer, "layer_type", "")) == "text":
        return style.ink
    palette = style.palette or (style.ink,)
    return palette[index % len(palette)]


def _apply_layer(plot: Any, layer: Any, index: int, style: PlotStyle) -> None:
    """Restyle one layer, touching only the fields its renderer actually reads."""
    layer_type = str(getattr(layer, "layer_type", ""))
    color = _style_color_for(layer, index, style)
    fields: Dict[str, Any] = {}

    if layer_type == "polyline":
        fields["color"] = color
        fields["line_width"] = style.line_width
    elif layer_type == "line_family":
        fields["line_width"] = style.line_width
    elif layer_type == "text":
        fields["color"] = style.ink
    elif layer_type == "scatter":
        fields["point_size"] = style.point_size
        fields["point_outline_enabled"] = style.point_outline
        if style.point_outline:
            fields["point_outline_color"] = style.point_outline_color
            fields["point_outline_width"] = style.point_outline_width
    elif layer_type == "patch":
        # Patch fill is the only colour patch.py reads; keep the author's opacity, which
        # is usually the difference between a fill and a wash.
        face = getattr(layer.style, "face_color", None)
        if face is not None:
            alpha = float(face[3]) if len(face) > 3 else 1.0
            fields["face_color"] = (color[0], color[1], color[2], alpha)
    elif layer_type.endswith("3d"):
        if getattr(layer, "colors", None) is None:
            # set_layer_style sets gpu_dirty for this case — geometry3d reads style.color
            # at upload, not at draw (CONTRACT §1.4).
            fields["color"] = color
        if str(getattr(layer, "primitive", "")) == "points":
            fields["point_size"] = style.point_size

    if fields:
        layerops.set_layer_style(plot, layer, **fields)

    # Per-element colour arrays: rewrite only the uniform ones (see _uniform_color).
    if layer_type in ("scatter", "line_family"):
        if _uniform_color(getattr(layer, "colors", None)) is not None:
            _repaint_uniform_colors(layer, color)


def _axes3d_targets(plot: Any) -> List[Any]:
    """Every :class:`~glplot.core.camera3d.Axes3DOptions` in the figure, panels included.

    A style is a property of the *figure*, exactly like ``options.axis_show_grid``, but the
    3D decoration is per panel (``core/panel.py:56``). Styling only the active panel would
    leave a 2x2 split with three panels in the old look and one in the new — so this
    returns all of them and the caller writes all of them.

    Falls back to the engine's own ``axes3d`` property (which forwards to the active panel)
    for anything that is not panel-shaped, and to an empty list for the hand-rolled plot
    stubs the test-suite is full of.
    """
    targets: List[Any] = []
    for panel in getattr(plot, "panels", None) or ():
        axes = getattr(panel, "axes3d", None)
        if axes is not None:
            targets.append(axes)
    if not targets:
        axes = getattr(plot, "axes3d", None)
        if axes is not None:
            targets.append(axes)
    return targets


def _apply_axes3d(plot: Any, style: PlotStyle) -> None:
    """Write the preset's 3D decoration onto every panel. Does not touch the camera."""
    for axes in _axes3d_targets(plot):
        for name in AXES3D_FIELDS:
            if hasattr(axes, name):
                setattr(axes, name, getattr(style.axes3d, name))


def _rebuild_axes3d(plot: Any) -> None:
    """Re-emit the 3D box/floor/grid/tick layers so a decoration change is visible now.

    The four decoration layers are *geometry*, rebuilt by ``engine.ensure_3d_axes`` — a
    ``show_floor = False`` that nobody rebuilds after is a checkbox that does nothing until
    the next orbit. ``set_3d_view()`` with no arguments is the engine's own "re-sync and
    rebuild" entry point, and it is what the View 3D panel calls for the same reason.

    Gated on the panel actually being 3D: on a 2D figure the call would be a no-op that
    still walked the scene, and on a stub plot it does not exist at all.
    """
    is_3d = getattr(plot, "is_3d_scene", None)
    rebuild = getattr(plot, "set_3d_view", None)
    if callable(is_3d) and callable(rebuild) and is_3d():
        rebuild()


def apply_style(plot: Any, style: PlotStyle, *, layers: bool = True) -> int:
    """Apply ``style`` to the live scene. **The** one function that applies a preset.

    **Queued commands only** (CONTRACT §1.1): it writes options and layer arrays, and the
    hand-drawn pass re-uploads geometry.

    ``layers=False`` restyles the page only — background, grid, blending, effects — and
    leaves every layer's colour and width alone. That is the right mode for applying the
    default look at startup, where the user's own ``gplt.plot(..., color=...)`` must win.
    The *page* includes the 3D box: axis decoration is decoration whoever set it, so it is
    written in both modes, while the 3D camera is never touched in either.

    Returns the number of layers the hand-drawn transform skipped for being too large; 0
    for every preset that is not hand-drawn. Callers surface it, per SPEC_FIX P1-F.
    """
    options = plot.options
    visual = options.visual

    visual.background_color = tuple(style.background)

    grad = visual.gradient_background
    if style.gradient is None:
        grad.enabled = False
    else:
        grad.enabled = True
        grad.top_color = tuple(style.gradient[0])
        grad.bottom_color = tuple(style.gradient[1])

    glow = visual.glow
    glow.enabled = bool(style.glow.enabled)
    glow.threshold = float(style.glow.threshold)
    glow.intensity = float(style.glow.intensity)
    glow.radius_px = float(style.glow.radius_px)
    _set_optional(glow, "knee", float(style.glow.knee))

    _set_optional(visual, "tonemap_index", int(style.tonemap))
    _set_optional(visual, "grain_amount", float(style.grain))

    options.blend_mode = style.blend_mode
    options.enable_antialiasing = bool(style.antialiasing)
    options.default_global_alpha = float(style.global_alpha)
    options.axis_show_grid = bool(style.show_grid)
    options.axis_show_frame = bool(style.show_frame)
    options.axis_show_labels = bool(style.show_labels)
    options.axis_grid_alpha = float(style.grid_alpha)
    options.axis_grid_color = tuple(style.grid_color) if style.grid_color else AUTO_GRID_COLOR
    options.density_scheme_index = int(style.colormap_index)

    # Ticks only when the preset has an opinion. ``None`` is not "the defaults", it is
    # "leave whatever gplt.tick_params / minorticks_on set" — see :class:`TickSettings`.
    if style.ticks is not None:
        options.axis_show_ticks = bool(style.ticks.show)
        options.axis_tick_len_px = float(style.ticks.length_px)
        options.axis_minor_ticks = bool(style.ticks.minor)
        options.axis_minor_subdivisions = int(style.ticks.minor_subdivisions)

    _apply_axes3d(plot, style)

    # light_bg_mode is not a colour, it is a claim about the page: the workspace picks the
    # imgui chrome theme from it (``workspace.py:252``) and the engine keeps a stock
    # background per mode. Adopt the mode WITHOUT the stock background — this preset has
    # its own, and ``set_background_mode(apply_preset=False)`` is exactly the door for that.
    # The getattr guard is for hand-rolled plot stubs, which tests are full of.
    set_mode = getattr(plot, "set_background_mode", None)
    if callable(set_mode):
        set_mode(bool(style.light_bg_mode), apply_preset=False)
    else:  # pragma: no cover - only reachable on a stub without the engine API
        options.light_bg_mode = bool(style.light_bg_mode)

    skipped = 0
    if layers:
        for index, layer in enumerate(list(getattr(plot.scene, "layers", []))):
            _apply_layer(plot, layer, index, style)
        skipped = hand_drawn_layers(plot, style)

    # Last, and after ``set_background_mode``: rebuilding the 3D decoration re-reads
    # ``light_bg_mode`` for its ink (``engine._axis3d_color``), so doing it earlier would
    # paint the box in the outgoing preset's colours.
    _rebuild_axes3d(plot)

    layerops.mark_scene_dirty(plot)
    return skipped


# ----------------------------------------------------------------------------------
# Undo
# ----------------------------------------------------------------------------------

#: The plain ``options`` scalars a preset writes. Captured and restored by name so the two
#: halves cannot drift: a field added to :func:`apply_style` but forgotten here would make
#: undo quietly lossy, which is worse than not offering undo at all.
#:
#: The four tick fields are captured unconditionally even though only some presets write
#: them: restoring a value to itself costs nothing, and deciding what to capture from the
#: *incoming* preset would make undo depend on which style you happened to pick.
_OPTION_FIELDS = (
    "blend_mode",
    "enable_antialiasing",
    "default_global_alpha",
    "axis_show_grid",
    "axis_show_frame",
    "axis_show_labels",
    "axis_grid_alpha",
    "axis_grid_color",
    "axis_show_ticks",
    "axis_tick_len_px",
    "axis_minor_ticks",
    "axis_minor_subdivisions",
    "density_scheme_index",
    "light_bg_mode",
)

_GLOW_FIELDS = ("enabled", "threshold", "intensity", "radius_px")
_GRADIENT_FIELDS = ("enabled", "top_color", "bottom_color")

#: Per-layer style fields any preset may write (the union across layer types). Captured
#: unconditionally: restoring a field the preset happened not to touch is a no-op, whereas
#: missing one is a leak.
_LAYER_STYLE_FIELDS = (
    "color",
    "face_color",
    "line_width",
    "point_size",
    "point_outline_enabled",
    "point_outline_color",
    "point_outline_width",
)


def capture_state(plot: Any, *, layers: bool = True) -> Dict[str, Any]:
    """Everything :func:`apply_style` is about to overwrite, as plain data.

    Snapshot/restore rather than a hand-written inverse, because the inverse of "apply a
    preset" is "the exact scene before it", and only a snapshot can promise that. It stays
    cheap because the only arrays it copies are the hand-drawn sources (already capped at
    :data:`HAND_DRAWN_MAX_POINTS`) and uniform colours, which collapse to one RGBA.
    """
    options = plot.options
    visual = options.visual

    state: Dict[str, Any] = {
        "options": {name: getattr(options, name) for name in _OPTION_FIELDS},
        "background_color": tuple(visual.background_color),
        "glow": {name: getattr(visual.glow, name) for name in _GLOW_FIELDS},
        "glow_knee": _get_optional(visual.glow, "knee"),
        "gradient": {name: getattr(visual.gradient_background, name) for name in _GRADIENT_FIELDS},
        "tonemap_index": _get_optional(visual, "tonemap_index"),
        "grain_amount": _get_optional(visual, "grain_amount"),
        # One dict per panel, in panel order. Positional rather than keyed because panels
        # have no stable identity across a split/merge; :func:`restore_state` zips, so a
        # figure that was re-tiled between the apply and the undo restores the panels it
        # still has and drops the rest instead of raising.
        "axes3d": [
            {name: getattr(axes, name) for name in AXES3D_FIELDS if hasattr(axes, name)}
            for axes in _axes3d_targets(plot)
        ],
        "layers": [],
    }
    if not layers:
        return state

    entries: List[Dict[str, Any]] = []
    for layer in list(getattr(plot.scene, "layers", [])):
        style = layer.style
        entry: Dict[str, Any] = {
            "layer_id": getattr(layer, "layer_id", None),
            "style": {
                name: getattr(style, name) for name in _LAYER_STYLE_FIELDS if hasattr(style, name)
            },
            "uniform_color": _uniform_color(getattr(layer, "colors", None)),
        }
        source = _source_xy(layer)
        if source is not None:
            entry["source_xy"] = source
            entry["jittered_xy"] = layerops.layer_xy(layer)
        elif str(getattr(layer, "layer_type", "")) == "polyline":
            entry["plain_xy"] = True
        entries.append(entry)

    state["layers"] = entries
    return state


def _restore_layer(plot: Any, entry: Dict[str, Any]) -> None:
    """Put one layer back exactly as :func:`capture_state` found it."""
    layer = layerops.find_layer(plot, entry.get("layer_id"))
    if layer is None:
        return

    fields = {
        name: value for name, value in entry.get("style", {}).items() if hasattr(layer.style, name)
    }
    if fields:
        layerops.set_layer_style(plot, layer, **fields)

    uniform = entry.get("uniform_color")
    if uniform is not None and _uniform_color(getattr(layer, "colors", None)) is not None:
        _repaint_uniform_colors(layer, uniform)

    source = entry.get("source_xy")
    if source is not None:
        # It was jittered before: put the jittered data and its stash back together, or the
        # next preset would take the wobble for the truth.
        jittered = entry.get("jittered_xy")
        if jittered is not None:
            layerops.update_layer_xy(plot, layer, jittered[0], jittered[1])
        layer.metadata[_SOURCE_XY_KEY] = source
    elif entry.get("plain_xy"):
        stashed = _source_xy(layer)
        if stashed is not None:
            layerops.update_layer_xy(plot, layer, stashed[0], stashed[1])
            layer.metadata.pop(_SOURCE_XY_KEY, None)


def restore_state(plot: Any, state: Dict[str, Any]) -> None:
    """Undo of :func:`apply_style`. **Queued commands only.**

    Layers that disappeared between the capture and the undo are skipped, not resurrected:
    the scene panel's removal is its own undo entry and owns that.
    """
    options = plot.options
    visual = options.visual

    for name, value in state["options"].items():
        if name == "light_bg_mode":
            # Same reasoning as in apply_style: writing the flag plainly would re-arm the
            # engine's latch and let it repaint the background we are busy restoring.
            set_mode = getattr(plot, "set_background_mode", None)
            if callable(set_mode):
                set_mode(bool(value), apply_preset=False)
                continue
        setattr(options, name, value)
    visual.background_color = state["background_color"]

    for name, value in state["glow"].items():
        setattr(visual.glow, name, value)
    _restore_optional(visual.glow, "knee", state["glow_knee"])

    for name, value in state["gradient"].items():
        setattr(visual.gradient_background, name, value)

    _restore_optional(visual, "tonemap_index", state["tonemap_index"])
    _restore_optional(visual, "grain_amount", state["grain_amount"])

    for axes, captured in zip(_axes3d_targets(plot), state.get("axes3d", [])):
        for name, value in captured.items():
            setattr(axes, name, value)

    for entry in state.get("layers", []):
        _restore_layer(plot, entry)

    _rebuild_axes3d(plot)
    layerops.mark_scene_dirty(plot)


def style_command(plot: Any, style: PlotStyle, *, layers: bool = True) -> Command:
    """An undoable "apply this preset", for ``Panel.push_command``.

    The state is captured when ``do`` first *runs* (inside the drain), not when the command
    is built (inside the draw callback) — otherwise a click would snapshot a scene that
    another queued command was still about to change.
    """
    captured: Dict[str, Any] = {}

    def do() -> None:
        if not captured:
            captured.update(capture_state(plot, layers=layers))
        apply_style(plot, style, layers=layers)

    def undo() -> None:
        if captured:
            restore_state(plot, captured)

    return Command(label=f"Style: {style.name}", do=do, undo=undo)


# ----------------------------------------------------------------------------------
# Startup default
# ----------------------------------------------------------------------------------

#: The fields that decide whether anyone has styled this plot yet. Compared against a
#: pristine ``EngineOptions``, not against hardcoded literals, so this cannot rot when a
#: default changes.
#: The two backgrounds ``engine._apply_background_mode`` seeds at construction, keyed by
#: ``light_bg_mode``. A plot sitting on either of them has not been styled by anyone: the
#: dataclass default (black) is *not* the answer, because the constructor has already run.
_STOCK_BACKGROUNDS: Dict[bool, RGB] = {True: (1.0, 1.0, 1.0), False: (0.0, 0.0, 0.0)}

_FACTORY_PROBE = (
    "blend_mode",
    "enable_antialiasing",
    "axis_show_grid",
    "axis_show_frame",
    "axis_grid_alpha",
    "density_scheme_index",
)


def is_factory_default(plot: Any) -> bool:
    """True when nothing has styled this plot — no background, no options touched.

    The workstation opens on Clean, but only for a plot that has not been styled already.
    A user who set ``visual.background_color`` or a blend mode before ``show()`` meant it,
    and having a GUI silently overwrite their choice on mount would be its own bug report.
    """
    from ..options import EngineOptions

    reference = EngineOptions()
    options = plot.options
    stock = _STOCK_BACKGROUNDS[bool(getattr(options, "light_bg_mode", False))]
    if tuple(options.visual.background_color)[:3] != stock:
        return False
    for name in _FACTORY_PROBE:
        if getattr(options, name) != getattr(reference, name):
            return False
    return options.visual.glow.enabled == reference.visual.glow.enabled


def default_style() -> PlotStyle:
    """The preset the workstation opens with."""
    return get_style(DEFAULT_STYLE_KEY)


def palette_preview(style: PlotStyle, count: int = 7) -> Sequence[RGBA]:
    """The first ``count`` palette entries, cycled — for a swatch strip in the UI."""
    palette = style.palette or (style.ink,)
    return [palette[i % len(palette)] for i in range(max(0, int(count)))]
