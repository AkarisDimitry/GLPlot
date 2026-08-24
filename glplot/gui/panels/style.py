"""The style editor — every live knob that changes how the plot looks, in seven tabs.

Styles / Scene / Layer / Density / Effects / Axes / Performance.

**Styles comes first on purpose.** The other six tabs are a hundred knobs, and a hundred
knobs is not a look — it is homework. The Styles tab is the answer to "I just want a
cleaner plot": one click, whole scene, undoable. The knobs are still there for the tenth
percent the preset did not guess.

**Wiring policy.** A widget appears here only when some renderer actually reads the
field at draw time (the audit lives in ``brief_options-style-surface.md``). A slider
bound to a field nothing reads is worse than a missing slider: it looks like it works,
so the user blames their data. The fields deliberately absent are, with the reason:

* ``ssao.radius`` — no ``u_ssao_radius`` uniform exists.
* ``glow.resolution_scale`` — ``effects.py:311`` hardcodes ``0.5``.
* ``overrides.enabled`` — the multipliers always apply; the master switch is ignored.
* ``style.pickable`` — picking reads ``style.visible`` (``picking.py:87,189``).
* ``style.use_colormap`` / ``cmap`` / ``vmin`` / ``vmax`` — shaders use
  ``options.line_colormap_enabled``; imshow reads ``layer.metadata``, not the style.
* ``style.edge_color`` / ``text_size_px`` — matplotlib export path only, never GL.
* ``hover_pick_hz``, ``pan_key_fraction``, ``zoom_key_factor``, ``box_zoom_min_pixels``,
  ``picking_radius_px``, ``always_lod``, ``density_log_scale``,
  ``enable_density_interaction_path``, ``interaction_budget_lines_per_screen_px`` —
  zero readers anywhere.

Per-layer widgets are gated on ``layer.layer_type`` against the applicability matrix in
the same brief: ``line_width`` never appears for scatter, and so on.

**The 3D half of the Layer tab.** A selected 3D layer used to get three camera sliders and
nothing else: no plot type, no per-kind parameters, no palette. It now gets the same editor
a 2D layer does, routed through the second registry —
:mod:`glplot.gui.layerops3d` — because a 3D kind takes three columns and compiles to a
:class:`~glplot.core.layers.Layer3D`, and is not convertible to or from a 2D kind. What
that buys: a surface can become a wireframe or a bar field in place, every option of the
kind it lands in is editable, and its colormap can be re-mapped, all from the layer the
user already has selected. The one thing that is *not* here is per-**layer** blending —
see :meth:`StylePanel._draw_layer_blending` for why it cannot be.

``point_outline_*`` (the 2D scatter's per-marker ring) is still 2D only: ``geometry3d`` has
no ``u_point_outline_width_px``. The general ``outline_*`` fields are what ring a 3D
marker, and they are wired for every layer type — see
:meth:`StylePanel._draw_layer_outline`.

Every mutation goes through :meth:`Panel.submit` per CONTRACT §1.1 — including the plain
option scalars. Writing ``options.foo`` straight from ``draw`` would land after the scene
pass and lose the dirty-flag latch at ``engine.py:1238``, so the change would not show up
until an unrelated event woke the loop. The one-frame deferral is invisible during a drag
because the drain runs at the top of the loop, before ``hud.update()`` redraws the widget.

Style edits are intentionally *not* pushed onto the undo stack: a slider drag would bury
the user's real history under a hundred entries per second. Applying a *preset* is the
exception — it is one discrete act that changes the whole scene at once, which is exactly
what undo is for.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...core import layout as _layout
from ...options import DEFAULT_AXIS_MARGINS, STOCK_WINDOW_TITLES, BlendMode
from ...utils.shaders import DENSITY_SCHEMES, eval_colormap
from .. import layerops, layerops3d, styles, widgets
from ..history import Command
from .base import Panel

try:
    from imgui_bundle import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - exercised only on GL-less systems
    IMGUI_AVAILABLE = False
    imgui = None


__all__ = ["StylePanel", "colormap_strip_colors", "cpu_colormap_strip_colors"]


# -- the per-layer-type applicability matrix ------------------------------------------
# Straight from brief_options-style-surface.md §3. Widening any of these sets without a
# matching renderer change re-creates the dead-slider problem this panel exists to avoid.

#: ``style.line_width`` reaches a shader only here (``polyline.py:171``,
#: ``line_family.py:189``). It is dead on 3D: no ``glLineWidth`` call exists in the repo.
LINE_WIDTH_TYPES = frozenset({"polyline", "line_family"})

#: ``style.color`` is read at draw time only here (``polyline.py:173``, ``text.py:59``).
#: scatter/patch/line_family bake colour into a VBO; 3D is handled separately below.
COLOR_LIVE_TYPES = frozenset({"polyline", "text"})

#: ``style.face_color`` — the patch fill (``patch.py:112-113``), unreachable from any
#: GUI before this panel.
FACE_COLOR_TYPES = frozenset({"patch"})

#: Point outlines — the 2D scatter's own per-marker ring — are live only on 2D scatter
#: (``scatter.py``'s ``u_point_outline_*``). Not to be confused with the general
#: ``outline_*`` silhouette below, which is a different feature on different uniforms.
OUTLINE_TYPES = frozenset({"scatter"})

#: Layer types with no reader for the general ``style.outline_*`` silhouette. Every other
#: type has one — ``geometry3d`` for all three 3D primitives, and ``scatter``/``polyline``/
#: ``patch``/``line_family`` in 2D — so this is a short deny-list rather than the usual
#: allow-list: ``text`` packs ``style.color`` and nothing else (``text.py``), and a layer
#: type added later is far more likely to want an outline than not.
OUTLINE_DEAD_TYPES = frozenset({"text"})

#: Text layers have no alpha uniform — ``text.py`` packs ``style.color`` and nothing else.
ALPHA_DEAD_TYPES = frozenset({"text"})


#: Paired with ``renderers/axis.py``'s ``AUTO_GRID_COLOR``. While ``axis_grid_color``
#: holds exactly this value the renderer picks the grid colour from the background
#: luminance, which is the historical look; any other value is used verbatim. Duplicated
#: rather than imported because ``renderers/axis.py`` does ``from OpenGL.GL import *`` at
#: module level and importing it here would break headless import (CONTRACT §5.1).
AUTO_GRID_COLOR = (0.2, 0.2, 0.2)

#: Paired with ``renderers/axis.py``'s ``_MPL_DEFAULT_TITLE_PT``/``_MPL_DEFAULT_LABEL_PT``:
#: the matplotlib point size a tick/label/title's ``fontsize`` option is scaled against, so
#: the slider default lands on "unscaled" rather than an arbitrary pixel count. Duplicated
#: for the same headless-import reason as ``AUTO_GRID_COLOR`` above.
_MPL_DEFAULT_TITLE_PT = 12.0
_MPL_DEFAULT_LABEL_PT = 10.0

#: The colour a patch fill is seeded with when the user enables it on a layer whose
#: ``face_color`` is None (which is how ``add_patch`` leaves it).
DEFAULT_FACE_COLOR = (0.30, 0.55, 0.90, 1.0)

#: ``LayerStyle.outline_color``'s own default, repeated here so the picker opens on the
#: value the renderer would have used rather than on an invented one.
DEFAULT_OUTLINE_COLOR = (0.0, 0.0, 0.0, 1.0)

#: What each blend mode is *for*, in the terms someone choosing between them needs. Keyed
#: by ``BlendMode`` name so a mode added to :mod:`glplot.options` shows up in the picker
#: with a "(no description yet)" rather than silently inheriting another mode's text.
BLEND_MODE_HELP: Dict[str, str] = {
    "ALPHA": (
        "Ordinary transparency (src.a, 1-src.a). The right default for anything with a "
        "surface: layers cover what is behind them in proportion to their alpha, and a "
        "50% layer over white reads as the colour you picked."
    ),
    "ADDITIVE": (
        "Light accumulates (src.a, 1). Overlapping translucent geometry gets *brighter* "
        "instead of averaging out, so a point cloud shows its density and the tails stay "
        "visible at alpha 0.02. Needs a dark background — on white there is nothing left "
        "to add to, and everything washes out."
    ),
    "SUBTRACTIVE": (
        "Additive run backwards (reverse subtract). Overlapping geometry gets darker, "
        "which is the same density readout inverted — the mode for accumulating ink on a "
        "light page."
    ),
    "SCREEN": (
        "Lightening (1, 1-src.colour). Like additive but saturating: it cannot blow past "
        "white, so dense regions stay separable where additive would clip them all to "
        "the same flat white. Also a dark-background mode."
    ),
    "AUTO": (
        "Alpha blending until the scene gets big, then blending switches off entirely "
        "past the Perf tab's 'Blending Cutoff' primitive count. The performance escape "
        "hatch, not a look."
    ),
    "OFF": (
        "No blending at all. Every fragment overwrites what is under it, so alpha does "
        "nothing and draw order decides the picture. The fastest mode, and the honest "
        "choice for opaque 3D geometry where the depth buffer already sorts things."
    ),
}

#: The option keys :func:`glplot.gui.widgets.kind_options_editor` draws today. The 3D
#: registry's dicts also carry ``stride``/``nu``/``nv``/``gap``/``bar_dx``/``bar_dy``/
#: ``scale``/``head``, which the shared editor has no branch for, so
#: :meth:`StylePanel._kind3d_options_editor` draws those and hands it only what it knows.
#:
#: Split this way rather than passing the whole dict and drawing "the rest" from a
#: hardcoded list of the keys it lacks: with that arrangement, the day the shared editor
#: learns ``stride`` the panel would draw *two* stride controls with two ids. This way the
#: worst case is a control that stays local for longer than it needed to.
SHARED_KIND_OPTION_KEYS = frozenset(
    {"where", "bins", "bar_width", "align", "density", "cumulative", "baseline", "whis", "cmap"}
)

#: ``layerops3d`` option keys that are per-point *arrays* (a quiver's vector columns), not
#: scalars. No slider can edit a column, so they are hidden from every options editor; the
#: Data panel's 3D section is where they are chosen.
ARRAY_KIND3D_OPTION_KEYS = frozenset({"u", "v", "w"})

_CPU_STRIP_CACHE: Dict[Tuple[str, int], List[Tuple[float, float, float]]] = {}


def _plot_title(plot: Any) -> str:
    """The window caption, but only once it is one the caller actually chose.

    Mirrors ``AxisRenderer._resolve_title`` so the Title box shows exactly what the
    renderer draws: blank for a default window (whose caption is the product name), the
    caption once ``gplt.title()`` has set one.
    """
    caption = str(getattr(plot, "title", "") or "")
    return "" if caption in STOCK_WINDOW_TITLES else caption


#: Colour handed to a polyline whose ``style.color`` is None. Matches the renderer's own
#: fallback (``polyline.py:173``) so enabling the picker does not change the look.
DEFAULT_LINE_COLOR = (0.0, 0.0, 0.0, 1.0)

#: Geometry of one colormap swatch in the picker.
SWATCH_WIDTH = 108.0
SWATCH_HEIGHT = 16.0
SWATCH_STEPS = 32

_STRIP_CACHE: Dict[Tuple[int, bool, bool, int], List[Tuple[float, float, float]]] = {}

#: Tone-map operators ``POST_COMPOSITE_FS`` implements, indexed by ``visual.tonemap_index``.
#: Positional: the index *is* the uniform value.
TONEMAP_NAMES = ("Off", "Reinhard", "ACES")

#: ``EffectManager.glow_knee``'s fallback, for the slider's initial position. Duplicated
#: rather than imported: ``managers/effects.py`` does ``from OpenGL.GL import *`` at module
#: level, so importing it here would break headless import (CONTRACT §5.1).
DEFAULT_GLOW_KNEE = 0.5

#: Geometry of one preset card in the Styles tab.
CARD_HEIGHT = 52.0
THUMB_WIDTH = 132.0
THUMB_HEIGHT = 44.0
PAD = 4.0

#: Sample count for a thumbnail curve. Enough for a smooth sine at 132px, few enough that
#: the hand-drawn jitter of all three curves stays a sub-millisecond one-off.
_PREVIEW_SAMPLES = 40

_PREVIEW_CACHE: Dict[str, List[List[Tuple[float, float]]]] = {}


def _auto_grid_color(background: Sequence[float]) -> Tuple[float, float, float]:
    """What ``renderers/axis.py`` picks for the grid when auto-contrast is on.

    Thin alias kept for the call sites already in this file; the real implementation is
    :func:`glplot.gui.styles.auto_grid_color`, public so other GUI code (e.g. Math Lab's
    style-matched preview) can share it instead of a third copy.
    """
    return styles.auto_grid_color(background)


def _preview_curves(style: Any) -> List[List[Tuple[float, float]]]:
    """Three curves in the thumbnail's unit square, in the preset's own hand.

    Memoised per style key: the hand-drawn variant runs the real
    :func:`glplot.gui.styles.jitter_polyline`, so the card shows the actual transform
    rather than an artist's impression of it — but it runs once, not once per frame.
    """
    cached = _PREVIEW_CACHE.get(style.key)
    if cached is not None:
        return cached

    curves: List[List[Tuple[float, float]]] = []
    for index in range(3):
        phase = index * 0.8
        decay = 0.30 - index * 0.06
        xs: List[float] = []
        ys: List[float] = []
        for i in range(_PREVIEW_SAMPLES):
            t = i / float(_PREVIEW_SAMPLES - 1)
            xs.append(0.06 + 0.88 * t)
            # A damped sine per curve: it exercises curvature, slope and a flat run, which
            # is where a line style either reads or does not.
            ys.append(0.5 - decay * math.sin(t * 5.0 + phase) * (1.0 - 0.35 * t))
        if style.hand_drawn:
            jittered = styles.jitter_polyline(
                xs,
                ys,
                # Exaggerated: the amplitude is a fraction of the *drawing*, and at 44px a
                # faithful 0.35% wobble is a sixth of a pixel. The card shows the character
                # of the transform at thumbnail scale, not its exact magnitude.
                amplitude=style.hand_amplitude * 10.0,
                wavelength=style.hand_wavelength * 2.0,
                seed=style.hand_seed + index,
            )
            if jittered is not None:
                xs = [float(v) for v in jittered[0]]
                ys = [float(v) for v in jittered[1]]
        curves.append(list(zip(xs, ys)))

    _PREVIEW_CACHE[style.key] = curves
    return curves


def _optional_float(target: Any, name: str, default: float) -> float:
    """Read a post-processing field the options dataclass may not declare yet.

    ``visual.tonemap_index``, ``visual.grain_amount`` and ``glow.knee`` are read by
    ``managers/effects.py`` through ``getattr`` with a default, so the pipeline honours them
    whether or not :mod:`glplot.options` has caught up. This is the panel's half of that
    arrangement: show the value the shader will actually use, and let ``_set`` write it.
    """
    raw = getattr(target, name, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _clamp01(value: float) -> float:
    """Clamp to the 0..1 range imgui's colour packers require."""
    return max(0.0, min(1.0, float(value)))


def colormap_strip_colors(
    scheme_index: int,
    *,
    invert: bool,
    light_to_color: bool,
    steps: int = SWATCH_STEPS,
) -> List[Tuple[float, float, float]]:
    """The RGB ramp of a density scheme, as the density pass would paint it.

    The ``invert`` / ``light_to_color`` remapping mirrors ``hud.py:555-586`` exactly, so a
    swatch previews what the plot will actually look like rather than the raw colormap.

    Pure (``eval_colormap`` is the CPU twin of the GLSL ``colormap()``) and memoised —
    the cache is bounded at ``11 schemes x 2 x 2`` entries, so the picker costs nothing
    per frame after the first.
    """
    key = (int(scheme_index), bool(invert), bool(light_to_color), int(steps))
    cached = _STRIP_CACHE.get(key)
    if cached is not None:
        return cached

    colors: List[Tuple[float, float, float]] = []
    denom = float(max(1, int(steps) - 1))
    for i in range(int(steps)):
        v = i / denom
        if invert:
            norm = 1.0 - 0.75 * v if light_to_color else 1.0 - v
        else:
            norm = 0.75 * v if light_to_color else v
        c = eval_colormap(int(scheme_index), norm)
        colors.append((_clamp01(c[0]), _clamp01(c[1]), _clamp01(c[2])))

    _STRIP_CACHE[key] = colors
    return colors


def cpu_colormap_strip_colors(
    name: str, *, steps: int = SWATCH_STEPS
) -> List[Tuple[float, float, float]]:
    """The RGB ramp of a **matplotlib** colormap, for the per-layer palette picker.

    A separate function from :func:`colormap_strip_colors` because the two preview
    genuinely different things. That one previews the *density pass*, whose ramps live in
    GLSL (``shaders.colormap()``) and are re-mapped by the invert / light-to-color options
    before they reach the screen. This one previews a **CPU** colormap: the colours a
    colour-mapped layer carries in its per-vertex VBO, produced by
    :func:`glplot.gui.layerops.colormap_colors` — the same call the re-map itself makes,
    so the swatch cannot disagree with the result of clicking it.

    Memoised per ``(name, steps)``. The cache is bounded by the length of
    :data:`glplot.gui.layerops.CPU_COLORMAP_NAMES`, and each entry costs one matplotlib
    lookup, so the picker is free after its first frame.

    An unknown colormap name yields a flat mid-grey strip rather than raising: the name
    can come from a layer built in code (``cmap="my_custom_map"``), and a panel that
    crashes the frame because it cannot preview a colormap would be worse than one that
    admits it cannot.
    """
    key = (str(name), int(steps))
    cached = _CPU_STRIP_CACHE.get(key)
    if cached is not None:
        return cached

    denom = float(max(1, int(steps) - 1))
    samples = [i / denom for i in range(int(steps))]
    try:
        rgba = layerops.colormap_colors(samples, str(name))
        colors = [(_clamp01(row[0]), _clamp01(row[1]), _clamp01(row[2])) for row in rgba]
    except Exception:  # pragma: no cover - unknown/removed matplotlib colormap
        colors = [(0.5, 0.5, 0.5)] * int(steps)

    _CPU_STRIP_CACHE[key] = colors
    return colors


def _is_3d(layer: Any) -> bool:
    """True for every ``*3d`` layer type (geometry/wireframe/scatter/volume/bars/mesh)."""
    return str(getattr(layer, "layer_type", "")).endswith("3d")


def _point_size_applies(layer: Any) -> bool:
    """Whether ``style.point_size`` reaches a shader for this layer.

    Live on 2D scatter (``scatter.py:150``) and on 3D — but ``geometry3d.py:192`` feeds
    ``gl_PointSize``, so it only does anything when the primitive is actually points.
    """
    layer_type = str(getattr(layer, "layer_type", ""))
    if layer_type == "scatter":
        return True
    if _is_3d(layer):
        return str(getattr(layer, "primitive", "points")) == "points"
    return False


def _color_mode(layer: Any) -> Optional[str]:
    """How ``style.color`` behaves for this layer: "live", "3d_upload", or None (dead).

    "3d_upload" means the colour is only the fallback for ``layer.colors is None`` and is
    read at GPU-upload time — ``layerops.set_layer_style`` sets ``gpu_dirty`` to make that
    land live. A 3D layer carrying per-vertex colours ignores ``style.color`` by design,
    so it reports None and the picker stays hidden.
    """
    layer_type = str(getattr(layer, "layer_type", ""))
    if layer_type in COLOR_LIVE_TYPES:
        return "live"
    if _is_3d(layer):
        return "3d_upload" if getattr(layer, "colors", None) is None else None
    return None


def _as_rgba(color: Any, fallback: Tuple[float, float, float, float]) -> Tuple[float, ...]:
    """Normalise a stored colour to a 4-tuple, tolerating None and 3-component values."""
    if color is None:
        return fallback
    try:
        vals = [float(c) for c in color]
    except (TypeError, ValueError):
        return fallback
    if len(vals) == 3:
        vals.append(1.0)
    if len(vals) != 4:
        return fallback
    return tuple(vals)


class StylePanel(Panel):
    """Rebuild of the old Render + Inspector panels: richer, tabbed, and honest.

    Honest in the specific sense that every control here moves something on screen; see
    the module docstring for what was left out and why.
    """

    title = "Style"
    icon = "palette"
    default_open = True

    def __init__(self, ws: Any) -> None:
        super().__init__(ws)
        #: Remembers the last layer the user picked here, so the Layer tab does not jump
        #: around when the scene's selection is cleared from elsewhere.
        self._layer_id: Optional[int] = None
        #: The preset last applied *from this panel*. None means "we do not know" — the
        #: scene may have been styled by hand or from code, and claiming a preset is
        #: active when a dozen knobs have moved since would be a small lie.
        self._style_key: Optional[str] = None
        #: Whether applying a preset also recolours existing layers, or only the page.
        self._style_layers: bool = True
        #: Live overrides for the hand-drawn transform, seeded from the preset.
        self._hand_amplitude: float = styles.get_style("hand").hand_amplitude
        self._hand_wavelength: float = styles.get_style("hand").hand_wavelength
        self._hand_seed: int = styles.get_style("hand").hand_seed
        #: The startup default is applied once, and only to an unstyled plot. See
        #: :meth:`_maybe_apply_default`.
        self._default_applied: bool = False
        #: Custom rows x cols for the Layout section's "Split" button.
        self._layout_rows: int = 2
        self._layout_cols: int = 2
        #: Whether new/updated splits link the panels' x / y axes (pan/zoom moves all).
        self._layout_sharex: bool = False
        self._layout_sharey: bool = False
        #: How tightly panels pack: gap between cells and outer margin (figure fractions).
        #: Default to 0 (edge to edge); the sliders adjust it live.
        self._layout_spacing: float = _layout.DEFAULT_WSPACE
        self._layout_outer: float = _layout.DEFAULT_OUTER_MARGIN
        #: Why the last queued layer edit failed, shown until the next one succeeds.
        #: Needed because the work happens in the drain, one frame after the click: a
        #: conversion that raises (``surface`` on scattered points is the common one) has
        #: no frame left to report into, and swallowing it silently is how a picker comes
        #: to look broken. Panel state, not scene state, so a queued closure may write it.
        self._layer_error: str = ""

    # -- deferred writes -------------------------------------------------------

    def _set(self, target: Any, name: str, value: Any) -> None:
        """Queue ``target.name = value`` (CONTRACT §1.1 — never write from ``draw``)."""

        def apply() -> None:
            setattr(target, name, value)

        self.submit(apply)

    def _set_style(self, layer: Any, **fields: Any) -> None:
        """Queue a per-layer style change, re-resolving the layer at drain time.

        The layer could be deleted between this frame and the drain (the scene panel
        queues removals too), so the closure looks it up by id rather than closing over
        the object and restyling a detached layer.
        """
        plot = self.plot
        layer_id = getattr(layer, "layer_id", None)

        def apply() -> None:
            live = layerops.find_layer(plot, layer_id)
            if live is not None:
                layerops.set_layer_style(plot, live, **fields)

        self.submit(apply)

    def _set_camera(self, layer: Any, key: str, value: float) -> None:
        """Queue a per-layer 3D camera change (``metadata["camera"]``, ``engine.py:238``).

        ``metadata`` is load-bearing, so this rewrites the camera sub-dict rather than
        replacing the whole mapping.
        """
        plot = self.plot
        layer_id = getattr(layer, "layer_id", None)

        def apply() -> None:
            live = layerops.find_layer(plot, layer_id)
            if live is None:
                return
            camera = dict(live.metadata.get("camera", {}))
            camera[key] = float(value)
            live.metadata["camera"] = camera
            layerops.mark_scene_dirty(plot)

        self.submit(apply)

    # -- layer selection -------------------------------------------------------

    def _selected_layer_id(self) -> Optional[int]:
        """The active layer id, preferring the HUD's selection, then the engine's."""
        hud = self.hud
        state = getattr(hud, "state", None) if hud is not None else None
        if state is not None and getattr(state, "selected_layer_id", None) is not None:
            return state.selected_layer_id
        interaction = getattr(self.plot, "interaction", None)
        if interaction is not None:
            return getattr(interaction, "selected_layer_id", None)
        return self._layer_id

    def _select_layer(self, layer_id: Optional[int]) -> None:
        """Queue a selection change, keeping the HUD and the engine in agreement."""
        self._layer_id = layer_id
        plot = self.plot
        hud = self.hud

        def apply() -> None:
            interaction = getattr(plot, "interaction", None)
            if interaction is not None:
                interaction.selected_layer_id = layer_id
            state = getattr(hud, "state", None) if hud is not None else None
            if state is not None:
                state.selected_layer_id = layer_id
            plot.frame.dirty_ui = True

        self.submit(apply)

    def _current_layer(self) -> Optional[Any]:
        """Resolve the selected layer object, falling back to the first in the scene."""
        layers = list(getattr(self.plot.scene, "layers", []))
        if not layers:
            return None
        layer_id = self._selected_layer_id()
        if layer_id is not None:
            found = layerops.find_layer(self.plot, layer_id)
            if found is not None:
                return found
        return layers[0]

    def _scene_has_3d(self) -> bool:
        """Whether any layer is 3D — drives the '2D only' warnings and the SSAO tab."""
        return any(_is_3d(layer) for layer in getattr(self.plot.scene, "layers", []))

    # -- lifecycle -------------------------------------------------------------

    def draw(self) -> None:
        """Render the seven tabs. The workspace owns the window."""
        if not IMGUI_AVAILABLE:
            return

        self._maybe_apply_default()

        tab_bar_open = imgui.begin_tab_bar("##style_tabs")
        if not tab_bar_open:
            return

        for label, body in (
            ("Styles", self._draw_styles),
            ("Scene", self._draw_scene),
            ("Layer", self._draw_layer),
            ("Density", self._draw_density),
            ("Effects", self._draw_effects),
            ("Axes", self._draw_axes),
            ("Perf", self._draw_performance),
        ):
            selected, _ = imgui.begin_tab_item(label)
            if selected:
                body()
                imgui.end_tab_item()

        imgui.end_tab_bar()

    # -- Styles ----------------------------------------------------------------

    def _maybe_apply_default(self) -> None:
        """Open the workstation on the Clean preset — but only on an unstyled plot.

        Once per panel lifetime, and gated on :func:`styles.is_factory_default`: a user who
        set a background or a blend mode before ``show()`` meant it, and a GUI that
        overwrote their choice on mount would be a bug report, not a default.

        Not pushed onto the undo stack: the state it replaces is "the engine's constructor
        defaults", which nobody chose, and an undo entry the user did not create sitting at
        the bottom of their history is noise. Applied with ``layers=False`` for the same
        reason — ``gplt.plot(color=...)`` before ``show()`` is an explicit choice.
        """
        if self._default_applied:
            return
        self._default_applied = True

        plot = self.plot
        if not styles.is_factory_default(plot):
            return

        style = styles.default_style()
        self._style_key = style.key

        def apply() -> None:
            styles.apply_style(plot, style, layers=False)

        self.submit(apply)

    def _draw_styles(self) -> None:
        """The preset gallery: one click, whole scene, undoable."""
        imgui.text_wrapped(
            "A style sets the page, the palette, the grid, the line weight and the "
            "effects in one go. Pick one, then fine-tune in the other tabs."
        )
        imgui.spacing()

        changed, value = imgui.checkbox("Restyle existing layers", self._style_layers)
        if changed:
            self._style_layers = value
        imgui.same_line()
        widgets.help_marker(
            "On: existing layers take the style's palette, line width and point size. "
            "Off: only the page changes — background, grid, blending, effects — and every "
            "layer keeps the colour you gave it.\n\n"
            "Layers whose colours carry data (a colormapped scatter, a per-index gradient) "
            "are never repainted either way."
        )
        imgui.separator()

        for style in styles.STYLES:
            if self._style_card(style):
                self._apply_style(style)

        imgui.separator()
        self._draw_hand_drawn_controls()

    def _apply_style(self, style: Any) -> None:
        """Queue an undoable apply of ``style`` (CONTRACT §1.1 — never from ``draw``)."""
        self._style_key = style.key
        self.push_command(styles.style_command(self.plot, style, layers=self._style_layers))

    def _style_card(self, style: Any) -> bool:
        """One preset: a live thumbnail of the look, its name and a one-line pitch.

        The thumbnail is drawn with the preset's own background, grid, palette and line
        width — and, for Hand-drawn, its own jitter — so what the card shows is what the
        click does. A name and a colour swatch would not have answered "what does chalk
        look like".
        """
        selected = style.key == self._style_key
        width = max(180.0, min(float(imgui.get_content_region_avail()[0]), 420.0))

        draw_list = imgui.get_window_draw_list()
        # CONTRACT §3: read the cursor BEFORE the item — afterwards it has already advanced.
        ox, oy = imgui.get_cursor_screen_pos()
        clicked = imgui.invisible_button(f"##style_{style.key}", (width, CARD_HEIGHT))
        hovered = imgui.is_item_hovered()

        self._draw_style_thumbnail(draw_list, style, ox + PAD, oy + PAD)

        text_x = ox + PAD + THUMB_WIDTH + 10.0
        draw_list.add_text(
            (text_x, oy + PAD + 2.0),
            imgui.get_color_u32((1.0, 1.0, 1.0, 1.0 if selected or hovered else 0.85)),
            style.name,
        )
        draw_list.add_text(
            (text_x, oy + PAD + 20.0),
            imgui.get_color_u32((0.65, 0.66, 0.70, 1.0)),
            style.description,
        )

        if selected:
            border = imgui.get_color_u32((0.55, 0.78, 1.0, 1.0))
            thickness = 2.0
        elif hovered:
            border = imgui.get_color_u32((1.0, 1.0, 1.0, 0.45))
            thickness = 1.0
        else:
            border = imgui.get_color_u32((1.0, 1.0, 1.0, 0.12))
            thickness = 1.0
        draw_list.add_rect(
            (ox, oy), (ox + width, oy + CARD_HEIGHT), border, rounding=4.0, thickness=thickness
        )
        return clicked

    def _draw_style_thumbnail(self, draw_list: Any, style: Any, x: float, y: float) -> None:
        """The preset's own look, at 132x44."""
        x1 = x + THUMB_WIDTH
        y1 = y + THUMB_HEIGHT

        if style.gradient is None:
            bg = style.background
            draw_list.add_rect_filled(
                (x, y), (x1, y1), imgui.get_color_u32((bg[0], bg[1], bg[2], 1.0)), rounding=2.0
            )
        else:
            top, bottom = style.gradient
            top_u32 = imgui.get_color_u32((top[0], top[1], top[2], 1.0))
            bottom_u32 = imgui.get_color_u32((bottom[0], bottom[1], bottom[2], 1.0))
            # add_rect_filled_multi_color takes its four corners in TL, TR, BR, BL order.
            draw_list.add_rect_filled_multi_color(
                (x, y), (x1, y1), top_u32, top_u32, bottom_u32, bottom_u32
            )

        if style.show_grid and style.grid_alpha > 0.0:
            grid = style.grid_color or _auto_grid_color(style.background)
            # The thumbnail is 44px tall; the live grid alpha would be invisible at that
            # size, so it is floored just enough to read as "there is a grid".
            alpha = max(float(style.grid_alpha), 0.18)
            col = imgui.get_color_u32((grid[0], grid[1], grid[2], alpha))
            for frac in (0.33, 0.66):
                draw_list.add_line((x, y + THUMB_HEIGHT * frac), (x1, y + THUMB_HEIGHT * frac), col)
            for frac in (0.25, 0.5, 0.75):
                draw_list.add_line((x + THUMB_WIDTH * frac, y), (x + THUMB_WIDTH * frac, y1), col)

        for index, curve in enumerate(_preview_curves(style)):
            color = style.palette[index % len(style.palette)]
            points = [(x + px * THUMB_WIDTH, y + py * THUMB_HEIGHT) for px, py in curve]
            draw_list.add_polyline(
                points,
                imgui.get_color_u32((color[0], color[1], color[2], color[3])),
                thickness=max(1.0, float(style.line_width) * 0.8),
                flags=0,
            )

        if style.show_frame:
            draw_list.add_rect(
                (x, y),
                (x1, y1),
                imgui.get_color_u32((0.5, 0.5, 0.5, 0.5)),
                rounding=2.0,
                thickness=1.0,
            )

    def _draw_hand_drawn_controls(self) -> None:
        """Hand-drawn's parameters, plus an honest word about what it costs.

        Shown always, not only while Hand-drawn is active: the sliders are how you find out
        what the wobble does, and hiding them until after the click is the same mistake as
        hiding Apply below the fold.
        """
        if not widgets.section("Hand-drawn", default_open=False):
            return

        imgui.text_wrapped(
            "Hand-drawn is geometry, not a filter: each line is resampled and pushed along "
            "its own normal by seeded low-frequency noise. The seed is fixed, so the wobble "
            "holds still instead of shimmering."
        )
        imgui.spacing()

        changed, value = widgets.labeled_slider_float(
            "Wobble",
            float(self._hand_amplitude),
            0.0005,
            0.02,
            fmt="%.4f",
            help="Displacement, as a fraction of the layer's own extent. Scale-free.",
        )
        if changed:
            self._hand_amplitude = value

        changed, value = widgets.labeled_slider_float(
            "Wavelength",
            float(self._hand_wavelength),
            0.005,
            0.20,
            fmt="%.3f",
            help="Distance between wobbles. Short = fur; long = a relaxed hand.",
        )
        if changed:
            self._hand_wavelength = value

        changed, value = imgui.drag_int("Seed", int(self._hand_seed), 0.2, 0, 9999)
        if changed:
            self._hand_seed = int(value)
        imgui.same_line()
        widgets.help_marker("A different hand. The same seed always redraws the same line.")

        if imgui.button("Apply Hand-drawn"):
            self._apply_style(
                replace(
                    styles.get_style("hand"),
                    hand_amplitude=float(self._hand_amplitude),
                    hand_wavelength=float(self._hand_wavelength),
                    hand_seed=int(self._hand_seed),
                )
            )

        imgui.text_disabled(
            f"Applies to 2D lines only, up to {styles.HAND_DRAWN_MAX_POINTS:,} pts."
        )
        oversized = styles.oversized_layers(self.plot)
        if oversized:
            names = ", ".join(oversized[:3]) + (" ..." if len(oversized) > 3 else "")
            widgets.error_box(
                f"Too large to redraw by hand, will be left as-is ({len(oversized)}): {names}"
            )

    # -- Scene -----------------------------------------------------------------

    def _draw_scene(self) -> None:
        """Scene-wide colour, blending, alpha and the global multipliers."""
        options = self.plot.options
        visual = options.visual

        self._draw_layout_section()

        if widgets.section("Background"):
            changed, color = imgui.color_edit3("Clear Color", visual.background_color[:3])
            if changed:
                self._set(visual, "background_color", tuple(color))
            imgui.text_disabled("Density mode overrides this with the colormap's low end.")

        if widgets.section("Blending"):
            names = [mode.name for mode in BlendMode]
            changed, name = widgets.enum_combo(
                "Blend Mode",
                options.blend_mode.name,
                names,
                help=(
                    "AUTO switches on point count. ADDITIVE and SCREEN accumulate light "
                    "and read best on a dark background."
                ),
            )
            if changed:
                self._set(options, "blend_mode", BlendMode[name])

            changed, alpha = widgets.labeled_slider_float(
                "Global Alpha",
                float(options.default_global_alpha),
                0.0,
                1.0,
                help="Multiplies every layer's own alpha. Read-only in the old HUD.",
            )
            if changed:
                self._set(options, "default_global_alpha", alpha)

            changed, auto_alpha = imgui.checkbox("Auto Alpha by N", options.enable_auto_alpha)
            if changed:
                self._set(options, "enable_auto_alpha", auto_alpha)
            imgui.same_line()
            widgets.help_marker(
                "Fades dense scenes automatically so overlapping lines stay readable. "
                "Scales Global Alpha down as the point count rises."
            )

        if widgets.section("Line Colormap"):
            changed, enabled = imgui.checkbox(
                "Colormap Lines by Index", options.line_colormap_enabled
            )
            if changed:
                self._set(options, "line_colormap_enabled", enabled)
            imgui.same_line()
            widgets.help_marker(
                "Colours each line/instance by its index using the Density tab's scheme "
                "instead of its own colour. Affects polylines, line families and the "
                "exact renderer. Previously reachable only by keybind."
            )
            imgui.text_disabled(f"Scheme: {self._scheme_name(options.density_scheme_index)}")

        if widgets.section("Antialiasing"):
            changed, aa = imgui.checkbox("Antialiasing", options.enable_antialiasing)
            if changed:
                self._set(options, "enable_antialiasing", aa)
            imgui.same_line()
            widgets.help_marker(
                "Line families and the exact renderer only. Scatter, patches and 3D are "
                "unaffected — use multisampling (Perf tab) for those."
            )

        if widgets.section("Global Overrides"):
            imgui.text_disabled("Multiplies every 2D layer's style at once.")
            if self._scene_has_3d():
                imgui.text_disabled("3D layers ignore alpha and point-size multipliers.")

            overrides = visual.overrides
            changed, value = widgets.labeled_slider_float(
                "Alpha x",
                float(overrides.alpha_multiplier),
                0.0,
                2.0,
                help="2D only. geometry3d.py:191 does not apply it.",
            )
            if changed:
                self._set(overrides, "alpha_multiplier", value)

            changed, value = widgets.labeled_slider_float(
                "Line Width x",
                float(overrides.line_width_multiplier),
                0.1,
                5.0,
                help="Polyline and line-family only.",
            )
            if changed:
                self._set(overrides, "line_width_multiplier", value)

            changed, value = widgets.labeled_slider_float(
                "Point Size x",
                float(overrides.point_size_multiplier),
                0.1,
                5.0,
                help="2D scatter only. geometry3d.py:192 does not apply it.",
            )
            if changed:
                self._set(overrides, "point_size_multiplier", value)

    def _draw_layout_section(self) -> None:
        """Split the window into panels (real subplots), from the Style panel.

        Figure-level, so it sits on the Scene tab beside the background and grid controls --
        the same operations as the View -> Split into panels menu, plus a custom rows x cols,
        in a place the user is already in while styling. Every change goes through the queue.
        """
        plot = self.plot
        if getattr(plot, "panels", None) is None or getattr(plot, "split_view", None) is None:
            return
        if not widgets.section("Layout"):
            return

        n = len(plot.panels)
        imgui.text_disabled(f"{n} panel{'s' if n != 1 else ''} — split the window into subplots.")

        presets = (("Single", 1, 1), ("1x2", 1, 2), ("2x1", 2, 1), ("2x2", 2, 2), ("3x3", 3, 3))
        for i, (label, r, c) in enumerate(presets):
            if i:
                imgui.same_line()
            if imgui.button(label):
                self._apply_layout(r, c)

        imgui.set_next_item_width(70.0)
        _cr, self._layout_rows = imgui.input_int("Rows", int(self._layout_rows))
        imgui.same_line()
        imgui.set_next_item_width(70.0)
        _cc, self._layout_cols = imgui.input_int("Cols", int(self._layout_cols))
        self._layout_rows = max(1, min(int(self._layout_rows), 6))
        self._layout_cols = max(1, min(int(self._layout_cols), 6))
        imgui.same_line()
        if imgui.button("Split##custom"):
            self._apply_layout(self._layout_rows, self._layout_cols)
        imgui.same_line()
        widgets.help_marker(
            "Each panel is an independent axes: click one to make it current, then plot into "
            "it or pan/zoom it on its own. 'Single' merges back to one, keeping the active "
            "panel's plot."
        )

        # Compactness: the gap between panels and the outer margin. Reflows the current grid
        # live (keeping every panel's plot), and seeds the spacing of the next split.
        cs, self._layout_spacing = widgets.labeled_slider_float(
            "Gap",
            float(self._layout_spacing),
            0.0,
            0.12,
            help="Space between panels, as a figure fraction. 0 packs them edge to edge.",
        )
        cm, self._layout_outer = widgets.labeled_slider_float(
            "Margin",
            float(self._layout_outer),
            0.0,
            0.10,
            help="Space around the whole grid, at the window edge.",
        )
        imgui.same_line()
        if imgui.button("Compact"):
            # A one-click tight preset.
            self._layout_spacing, self._layout_outer = 0.0, 0.0
            cs = True
        if (cs or cm) and n > 1:
            self._apply_spacing()

        # Tight labels: shrink each panel's axis gutter (the space its tick labels sit in) so
        # the frames pack closely, instead of every panel reserving the full/rail-widened one.
        tight = getattr(plot, "_panel_margins", None) is not None
        ct, tight = imgui.checkbox("Tight labels", tight)
        imgui.same_line()
        widgets.help_marker(
            "Give each panel a small gutter for its tick labels instead of the full one, so "
            "split frames sit close together. Turn off if long labels get clipped."
        )
        if ct:
            self._apply_tight_labels(tight)

        # Axis linking: which axes are locked together across panels, so pan/zoom on one
        # moves them all. Toggling applies immediately to the current split.
        cx, self._layout_sharex = imgui.checkbox("Link X axis", self._layout_sharex)
        imgui.same_line()
        cy, self._layout_sharey = imgui.checkbox("Link Y axis", self._layout_sharey)
        imgui.same_line()
        widgets.help_marker(
            "Lock the chosen axis across all panels: panning or zooming one panel pans/zooms "
            "every panel on that axis (matplotlib's sharex / sharey). Needs more than one panel."
        )
        if cx or cy:
            self._apply_sharing()

    def _apply_layout(self, nrows: int, ncols: int) -> None:
        """Queue a re-tile into ``nrows`` x ``ncols`` panels (or a merge to one)."""
        plot = self.plot
        sharex, sharey = self._layout_sharex, self._layout_sharey
        gap, outer = float(self._layout_spacing), float(self._layout_outer)

        def apply() -> None:
            if nrows <= 1 and ncols <= 1:
                if getattr(plot, "merge_view", None) is not None:
                    plot.merge_view()
            elif getattr(plot, "split_view", None) is not None:
                plot.split_view(nrows, ncols, wspace=gap, hspace=gap, outer=outer)
                # split_view clears any prior links, so re-apply the chosen ones.
                if getattr(plot, "set_shared_axes", None) is not None:
                    plot.set_shared_axes(sharex, sharey)

        self.submit(apply)

    def _apply_spacing(self) -> None:
        """Queue a live reflow of the current grid to the chosen gap / margin (keeps content)."""
        plot = self.plot
        gap, outer = float(self._layout_spacing), float(self._layout_outer)

        def apply() -> None:
            if getattr(plot, "retile_current", None) is not None:
                plot.retile_current(gap, gap, outer)

        self.submit(apply)

    def _apply_tight_labels(self, tight: bool) -> None:
        """Queue toggling the compact per-panel axis gutters (``engine._panel_margins``)."""
        plot = self.plot
        from ...options import PANEL_COMPACT_MARGINS

        def apply() -> None:
            plot._panel_margins = PANEL_COMPACT_MARGINS if tight else None
            plot.frame.dirty_scene = True
            plot.frame.dirty_ui = True

        self.submit(apply)

    def _apply_sharing(self) -> None:
        """Queue an axis-link change on the current panels (no re-tile)."""
        plot = self.plot
        sharex, sharey = self._layout_sharex, self._layout_sharey

        def apply() -> None:
            if getattr(plot, "set_shared_axes", None) is not None:
                plot.set_shared_axes(sharex, sharey)

        self.submit(apply)

    def _scheme_name(self, index: int) -> str:
        """Name of a density scheme, safe against an out-of-range index."""
        if 0 <= int(index) < len(DENSITY_SCHEMES):
            return DENSITY_SCHEMES[int(index)]
        return f"#{index}"

    # -- Layer -----------------------------------------------------------------

    def _draw_layer(self) -> None:
        """Per-layer style, gated on the layer type's real applicability matrix."""
        layers = list(getattr(self.plot.scene, "layers", []))
        if not layers:
            imgui.text_disabled("No layers yet.")
            imgui.text_disabled("Plot a function or paste some data to get started.")
            return

        layer = self._current_layer()
        if layer is None:
            imgui.text_disabled("No layer selected.")
            return

        self._draw_layer_selector(layers, layer)
        imgui.separator()

        layer_type = str(getattr(layer, "layer_type", ""))
        imgui.text_disabled(f"Type: {layer_type}   ID: {getattr(layer, 'layer_id', '?')}")

        changed, label = imgui.input_text("Label", getattr(layer, "label", ""))
        if changed:
            self._set_style(layer, label=label)

        if self._layer_error:
            widgets.error_box(self._layer_error)

        self._draw_layer_kind(layer)

        style = layer.style

        if widgets.section("Visibility & Stacking"):
            changed, visible = imgui.checkbox("Visible", style.visible)
            if changed:
                self._set_style(layer, visible=visible)
            imgui.same_line()
            widgets.help_marker(
                "Hidden layers are skipped by the renderer, by autoscale bounds and by " "picking."
            )

            if layer_type not in ALPHA_DEAD_TYPES:
                changed, alpha = widgets.labeled_slider_float("Alpha", float(style.alpha), 0.0, 1.0)
                if changed:
                    self._set_style(layer, alpha=alpha)

            changed, zorder = imgui.drag_int("Z-Order", int(style.zorder), 0.1, -100, 100)
            if changed:
                self._set_style(layer, zorder=int(zorder))
            imgui.same_line()
            widgets.help_marker(
                "Render order: higher draws on top. Sorted stably, so layers sharing a "
                "z-order keep their list order. Dragging in the Scene panel reorders the "
                "list, not this."
            )

        self._draw_layer_colors(layer, layer_type, style)
        self._draw_layer_colormap(layer)
        self._draw_layer_geometry(layer, layer_type, style)
        self._draw_layer_outline(layer, style)
        if _is_3d(layer):
            self._draw_layer_compositing_3d(layer, style)
        self._draw_layer_blending()

        # translation is applied by the 2D renderers' u_offset uniform and by
        # renderer_manager.py:199-203 for bounds; geometry3d has no offset uniform, so a
        # 3D layer gets its per-layer camera instead (metadata["camera"]).
        if _is_3d(layer):
            self._draw_layer_camera(layer)
        else:
            self._draw_layer_transform(layer)

    def _draw_layer_kind(self, layer: Any) -> None:
        """Plot type and per-kind parameters (bins, baseline, bar width, ...) for the layer.

        The heart of "edit any parameter of the selected plot": it changes *what* a layer is
        (scatter <-> line <-> bar <-> hist <-> ...) and every option of its kind, re-plotting
        from the layer's own retained source data. Only shown when the layer has an editable
        kind whose source is recoverable -- an imshow, a guide or a hand-plotted layer with no
        retained source cannot be rebuilt here, so the section stays hidden rather than lying.

        A 3D layer is routed to :meth:`_draw_layer_kind3d` instead of being tested against
        the 2D registry. Not a shortcut: ``layerops.layer_kind`` reads the *same* metadata
        key ``layerops3d`` writes (both are ``"gui_kind"``), so a scatter3d asked for its 2D
        kind answers ``"scatter3d"`` -- a string no 2D ``kind_spec`` accepts. Dispatching on
        the layer's dimensionality first is what keeps one tag serving two registries.
        """
        if _is_3d(layer):
            self._draw_layer_kind3d(layer)
            return

        kind = layerops.layer_kind(layer)
        if kind is None or layerops.layer_source_xy(layer) is None:
            return
        if not widgets.section("Plot Parameters"):
            return

        # Change the representation. Conversions a two-column table cannot reach are rejected
        # by the re-plot (caught below), so every kind can be offered without a crash risk.
        labels = [layerops.kind_spec(k).label for k in layerops.KIND_KEYS]
        changed_kind, picked = widgets.enum_combo(
            "Plot type", layerops.kind_spec(kind).label, labels
        )
        new_kind = layerops.KIND_KEYS[labels.index(picked)] if changed_kind else kind

        # Per-kind parameters. Read the layer's current options; on a kind change, seed the
        # new kind's defaults but carry over any option both kinds share.
        opts = dict(layerops.layer_kind_options(layer) or layerops.default_kind_options(kind))
        if changed_kind:
            merged = layerops.default_kind_options(new_kind)
            merged.update({k: v for k, v in opts.items() if k in merged})
            opts = merged
        changed_opts = widgets.kind_options_editor(opts)
        # line/scatter have no geometric parameters -- their look is colour, line width and
        # point size, which live in the sections below. Say so rather than leaving a blank.
        if not opts:
            imgui.text_disabled("No geometric parameters for this type.")
            imgui.text_disabled("Style it with Color / Line width / Point size below.")

        if changed_kind or changed_opts:
            self._replot_layer(layer, new_kind, opts)

    def _replot_layer(self, layer: Any, kind: str, options: Any) -> None:
        """Queue a re-plot of ``layer`` into ``kind`` with ``options``, from its source xy.

        Re-resolves the layer at drain time (the scene panel may delete it first) and re-points
        the selection when a kind change rebuilds the layer under a new id. A conversion the
        data shape cannot support is swallowed rather than crashing the frame.
        """
        plot = self.plot
        hud = self.hud
        layer_id = getattr(layer, "layer_id", None)
        opts = dict(options)

        def apply() -> None:
            live = layerops.find_layer(plot, layer_id)
            if live is None:
                return
            src = layerops.layer_source_xy(live)
            if src is None:
                return
            x, y = src
            label = getattr(live, "label", "") or "layer"
            try:
                new_layer = layerops.replot_layer_xy(
                    plot, hud, live, x, y, kind=str(kind), label=label, options=opts
                )
            except Exception:
                return
            new_id = getattr(new_layer, "layer_id", None)
            if new_id is not None and new_id != layer_id:
                self._layer_id = new_id
                interaction = getattr(plot, "interaction", None)
                if interaction is not None:
                    interaction.selected_layer_id = new_id
                state = getattr(hud, "state", None) if hud is not None else None
                if state is not None:
                    state.selected_layer_id = new_id
            layerops.mark_scene_dirty(plot)

        self.submit(apply)

    # -- Layer: the 3D plot type ----------------------------------------------

    def _draw_layer_kind3d(self, layer: Any) -> None:
        """Plot type and per-kind parameters for a **3D** layer.

        The 3D twin of :meth:`_draw_layer_kind`, and the answer to "select a 3D object and
        change what it is": the nine kinds of :data:`glplot.gui.layerops3d.KIND3D_KEYS` are
        all reachable from any layer that recorded its source columns, so a scatter becomes
        a surface, a surface a wireframe, a path a ribbon — keeping the label, the colour,
        the style, the position in the Scene list and the selection.

        Unlike the 2D version this section is drawn *even when the type cannot be changed*,
        with the picker dimmed and the reason in its tooltip. The 2D one can afford to
        vanish because a 2D layer that cannot be re-typed usually has nothing else to edit
        here either; a 3D layer that cannot be re-typed (one from ``gplt.plot_surface``,
        say) still has parameters, a palette and an outline worth reaching, and hiding the
        lot because one control is unavailable would be the wrong trade. The dim-plus-
        tooltip form is the house convention (``data_editor.py:1194-1207``).
        """
        kind = layerops3d.layer_kind3d(layer)
        convertible = kind is not None and layerops3d.layer_source_xyz(layer) is not None
        if not widgets.section("Plot Parameters"):
            return

        keys = list(layerops3d.KIND3D_KEYS)
        labels = [layerops3d.kind3d_spec(k).label for k in keys]
        current = layerops3d.kind3d_spec(kind).label if kind is not None else "(unknown)"

        if not convertible:
            imgui.push_style_var(imgui.StyleVar_.alpha, imgui.get_style().alpha * 0.5)
        changed_kind, picked = widgets.enum_combo("Plot type", current, labels)
        if not convertible:
            imgui.pop_style_var(1)
            if imgui.is_item_hovered():
                imgui.set_tooltip(self._kind3d_block_reason(layer, kind))
            # The popup still opens (STYLE_ALPHA only dims), so the refusal has to be here
            # rather than in the widget: a pick that did anything would rebuild the layer
            # from geometry that is not its data.
            changed_kind = False
        imgui.same_line()
        widgets.help_marker(self._kind3d_help())

        new_kind = keys[labels.index(picked)] if changed_kind else kind
        if new_kind is None:
            imgui.text_disabled("This layer was not built by the GUI, so it carries no kind.")
            return

        spec = layerops3d.kind3d_spec(new_kind)
        imgui.text_disabled(spec.description)
        if spec.needs_grid:
            imgui.text_disabled("Needs a rectangular (x, y) grid, or an explicit u x v shape.")

        if not convertible:
            # The parameters are equally unreachable, and for the same reason: every one of
            # them is an argument to the kind's ``geom``, which re-derives the geometry from
            # the columns this layer did not record. Drawing sliders that could only produce
            # an error banner would be the dead-control problem in a new place.
            imgui.text_disabled("Its parameters cannot be edited here either: re-deriving")
            imgui.text_disabled("the geometry needs the source columns it did not record.")
            return

        # Per-kind parameters. Read what the layer was built with; on a kind change seed the
        # new kind's defaults and carry over any option both kinds share (a baseline means
        # the same thing to stems, ribbons and bars, and losing it on every switch would
        # make the picker feel like it resets your work).
        opts = dict(layerops3d.layer_kind3d_options(layer))
        if changed_kind:
            merged = layerops3d.default_kind3d_options(new_kind)
            merged.update({k: v for k, v in opts.items() if k in merged})
            opts = merged
        changed_opts = self._kind3d_options_editor(opts)

        if not any(k not in ARRAY_KIND3D_OPTION_KEYS for k in opts):
            imgui.text_disabled("No geometric parameters for this type.")
            imgui.text_disabled("Style it with Color / Colormap / Point size below.")
        if new_kind == "quiver3d" and all(opts.get(k) is None for k in ("u", "v", "w")):
            imgui.text_disabled("No U/V/W columns on this layer: every arrow has zero length.")
            imgui.text_disabled("Build a vector field from the Data panel's 3D plot section.")

        if changed_kind or changed_opts:
            self._replot_layer3d(layer, new_kind, opts, undoable=changed_kind)

    def _kind3d_block_reason(self, layer: Any, kind: Optional[str]) -> str:
        """Why this 3D layer's type cannot be changed — the tooltip on the dimmed picker.

        Two distinct causes with two distinct fixes, so they get two distinct messages: a
        picker that greys out with one generic "unavailable" for both is how a user ends up
        re-plotting a layer that only needed a different panel.
        """
        if kind is None:
            return (
                f"{getattr(layer, 'label', '?')!r} carries no plot-type tag: it was built "
                "outside the GUI (gplt.plot_surface, gplt.scatter3d, add_geometry3d), so "
                "nothing here knows which of the nine kinds produced its vertices — or "
                "whether they are samples at all.\n\n"
                "Re-plot the columns from the Data panel's 3D section to get a layer whose "
                "type is editable. Everything else on this tab works on it as it is."
            )
        return (
            f"{getattr(layer, 'label', '?')!r} is a {kind} layer whose source columns were "
            "not recorded, and a derived kind's vertices are its geometry rather than its "
            "data — a surface's are the triangulation, a quiver's are the arrows.\n\n"
            "Converting off them would plot the scaffolding: a wireframe of a wireframe. "
            "Re-plot from the Data panel's 3D section to get the columns back."
        )

    def _kind3d_help(self) -> str:
        """The picker's tooltip: every 3D kind and what it draws.

        Built from the registry rather than written out, so a kind added to
        :mod:`glplot.gui.layerops3d` documents itself here. Same text the Data panel shows,
        for the same reason: the two pickers offer the same nine kinds and must not
        describe them differently.
        """
        lines = ["What this layer's three columns can be drawn as:", ""]
        lines += [
            f"  {layerops3d.kind3d_spec(k).label} — {layerops3d.kind3d_spec(k).description}"
            for k in layerops3d.KIND3D_KEYS
        ]
        return "\n".join(lines)

    def _kind3d_options_editor(self, opts: Dict[str, Any]) -> bool:
        """Edit a 3D kind's parameters in place. Returns True if any changed.

        Two halves, and the split is bookkeeping rather than design: the keys the shared
        :func:`glplot.gui.widgets.kind_options_editor` already draws
        (:data:`SHARED_KIND_OPTION_KEYS` — ``baseline`` is the only one a 3D kind uses
        today) go to it, so a stem plot's baseline is the same control in the Data panel,
        the Math Lab and here. The rest — the grid, bar-footprint and arrow parameters that
        exist only in 3D — are drawn here.

        Driven by which keys ``opts`` *has*, exactly as the shared editor is, so a kind that
        grows a parameter gets its control the moment ``layerops3d`` defaults it. The vector
        columns are arrays and are hidden (:data:`ARRAY_KIND3D_OPTION_KEYS`).
        """
        shared = {k: v for k, v in opts.items() if k in SHARED_KIND_OPTION_KEYS}
        changed = widgets.kind_options_editor(shared) if shared else False
        opts.update(shared)

        width = 120.0
        if "stride" in opts:
            imgui.set_next_item_width(width)
            hit, value = imgui.input_int("Stride", int(opts["stride"]))
            opts["stride"] = max(1, min(int(value), 256))
            changed |= hit
            imgui.same_line()
            widgets.help_marker(
                "Draw every Nth grid line. On a fine mesh every line is a sheet of solid "
                "ink; every 4th reads as a surface. The last row and column are always "
                "drawn, so the mesh keeps a closed edge."
            )

        if "nu" in opts or "nv" in opts:
            for label, key in (("Grid u", "nu"), ("Grid v", "nv")):
                if key not in opts:
                    continue
                imgui.set_next_item_width(width)
                hit, value = imgui.input_int(label, int(opts[key] or 0))
                opts[key] = max(0, int(value))
                changed |= hit
            imgui.same_line()
            widgets.help_marker(
                "The sampling shape of a parametric surface (a sphere, a torus), whose "
                "connectivity is not recoverable from x and y — both cycle. u x v must "
                "equal the row count.\n\n"
                "0 x 0 means 'detect a rectangular (x, y) lattice instead', which is what "
                "a height field z = f(x, y) wants."
            )

        if "bar_dx" in opts or "bar_dy" in opts:
            for label, key in (("Bar dx (0 = auto)", "bar_dx"), ("Bar dy (0 = auto)", "bar_dy")):
                if key not in opts:
                    continue
                imgui.set_next_item_width(width)
                hit, value = imgui.input_float(label, float(opts[key]))
                opts[key] = max(0.0, float(value))
                changed |= hit
            imgui.same_line()
            widgets.help_marker(
                "The footprint of each box in data units. 0 = 80% of the median spacing "
                "on that axis, which leaves a visible gap on gridded data."
            )

        if "gap" in opts:
            hit, value = widgets.labeled_slider_float(
                "Gap",
                float(opts["gap"]),
                0.0,
                0.9,
                fmt="%.2f",
                help="Fraction of each bar's footprint left empty. 0 packs them solid.",
            )
            if hit:
                opts["gap"] = float(value)
            changed |= hit

        if "scale" in opts:
            imgui.set_next_item_width(width)
            hit, value = imgui.input_float("Vector scale", float(opts["scale"]))
            opts["scale"] = float(value)
            changed |= hit
            imgui.same_line()
            widgets.help_marker(
                "Multiplies every (u, v, w) before it is drawn. The vectors are in data "
                "units, so this is what makes a field of small gradients visible at all."
            )

        if "head" in opts:
            hit, value = widgets.labeled_slider_float(
                "Arrow head",
                float(opts["head"]),
                0.02,
                0.9,
                fmt="%.2f",
                help="Barb length as a fraction of the arrow. Clamped to 0.02..0.9.",
            )
            if hit:
                opts["head"] = float(value)
            changed |= hit

        return bool(changed)

    def _replot_layer3d(
        self, layer: Any, kind: str, options: Dict[str, Any], *, undoable: bool
    ) -> None:
        """Queue a rebuild of ``layer`` as the 3D ``kind`` with ``options``.

        ``undoable`` splits the two things this serves, following the rule the rest of the
        panel follows (see the module docstring): a **type change** is one discrete act that
        replaces the layer, so it goes on the undo stack; an **options edit** is a slider
        drag that can fire sixty times a second and re-derives the same layer in place, so
        it does not — a hundred history entries per second would bury the user's real work,
        and dragging the stride back is already the inverse.

        The layer is re-resolved by id at drain time: the Scene panel queues removals too,
        and restyling a detached layer would silently do nothing. A conversion the data
        shape cannot support (``surface`` on scattered points is the one that happens)
        raises out of ``layerops3d``, and its message — which names the counts it saw — is
        put on the panel rather than swallowed, because by then the frame that could have
        pre-checked it is gone.
        """
        plot = self.plot
        hud = self.hud
        layer_id = getattr(layer, "layer_id", None)
        opts = dict(options)
        kind_key = str(kind)
        previous_kind = layerops3d.layer_kind3d(layer)
        previous_opts = dict(layerops3d.layer_kind3d_options(layer))
        # Survives the rebuild so undo can find the layer the new id belongs to.
        holder: Dict[str, Any] = {"id": layer_id}

        def convert(target_id: Optional[int], to_kind: str, to_opts: Dict[str, Any]) -> None:
            live = layerops.find_layer(plot, target_id)
            if live is None:
                return
            label = getattr(live, "label", "") or to_kind
            try:
                new_layer = layerops.replot_layer_xyz(
                    plot, hud, live, kind=to_kind, options=dict(to_opts), label=label
                )
            except ValueError as exc:
                self._layer_error = f"{label}: {exc}"
                return
            self._layer_error = ""
            new_id = getattr(new_layer, "layer_id", None)
            holder["id"] = new_id
            if new_id is not None and new_id != target_id:
                self._layer_id = new_id
                interaction = getattr(plot, "interaction", None)
                if interaction is not None:
                    interaction.selected_layer_id = new_id
                state = getattr(hud, "state", None) if hud is not None else None
                if state is not None:
                    state.selected_layer_id = new_id
            layerops.mark_scene_dirty(plot)

        if not undoable or previous_kind is None:
            self.submit(lambda: convert(holder["id"], kind_key, opts))
            return

        self.push_command(
            Command(
                label=f"Change plot type to {kind_key}",
                do=lambda: convert(holder["id"], kind_key, opts),
                undo=lambda: convert(holder["id"], previous_kind, previous_opts),
            )
        )

    def _draw_layer_selector(self, layers: Sequence[Any], current: Any) -> None:
        """A combo over every layer in the scene, synced with the shared selection."""
        names: List[str] = []
        for i, layer in enumerate(layers):
            label = str(getattr(layer, "label", "")) or "(unnamed)"
            names.append(f"{i}: {label} [{getattr(layer, 'layer_type', '?')}]")

        index = 0
        for i, layer in enumerate(layers):
            if layer is current:
                index = i
                break

        imgui.push_item_width(-1)
        changed, new_index = imgui.combo("##layer_select", index, names)
        imgui.pop_item_width()
        if changed and 0 <= new_index < len(layers):
            self._select_layer(getattr(layers[new_index], "layer_id", None))

    def _draw_layer_colors(self, layer: Any, layer_type: str, style: Any) -> None:
        """Colour widgets, shown only where the renderer reads the field."""
        mode = _color_mode(layer)
        has_face = layer_type in FACE_COLOR_TYPES
        if mode is None and not has_face:
            if _is_3d(layer):
                if not widgets.section("Color"):
                    return
                imgui.text_disabled("Per-vertex colors; style.color is unused.")
            return

        if not widgets.section("Color"):
            return

        if mode is not None:
            fallback = DEFAULT_LINE_COLOR
            changed, color = imgui.color_edit4("Color", _as_rgba(style.color, fallback))
            if changed:
                self._set_style(layer, color=tuple(color))
            if mode == "3d_upload":
                imgui.text_disabled("Fallback color; re-uploads the vertex buffer.")

        if has_face:
            fill_on = style.face_color is not None
            changed, new_fill = imgui.checkbox("Fill", fill_on)
            if changed:
                self._set_style(layer, face_color=DEFAULT_FACE_COLOR if new_fill else None)
            imgui.same_line()
            widgets.help_marker(
                "Patches draw their fill only when a face color is set, and add_patch "
                "leaves it unset."
            )
            if fill_on:
                changed, color = imgui.color_edit4(
                    "Fill Color", _as_rgba(style.face_color, DEFAULT_FACE_COLOR)
                )
                if changed:
                    self._set_style(layer, face_color=tuple(color))

    # -- Layer: palette --------------------------------------------------------

    def _draw_layer_colormap(self, layer: Any) -> None:
        """The selected layer's palette, and the range it is mapped over.

        Gated on :func:`glplot.gui.layerops.layer_colormap_kind`, which is the honest
        answer to "does this layer *have* a colormap": a layer whose colours were handed in
        as literal RGBA has none, and drawing a picker for it would be a control that
        cannot do anything. The three mechanisms it reports need three different setters,
        so they are three branches here rather than one picker pretending to be generic:

        * ``values2d`` / ``values3d`` / ``image`` — colours computed on the CPU from
          retained scalars. Re-mapping recomputes the per-vertex colour array
          (:func:`glplot.gui.layerops.set_layer_colormap`) and re-uploads it. This is the
          branch a GUI-built 3D layer takes: ``add_xyz_layer`` retains ``cvalues``/``cmap``
          on every kind precisely so the palette stays changeable afterwards.
        * ``gl_line`` — a polyline or line family coloured by a *shader* colormap keyed on
          the vertex/instance id rather than on data. Different names
          (:data:`glplot.gui.layerops.GL_COLORMAP_NAMES`), an index rather than a name, and
          a third state that matters: "follow the scene", which is every layer's default
          and what keeps the Scene tab's global toggle working.

        The swatch strips are the real ramps (see :func:`cpu_colormap_strip_colors`), not
        colour chips: a name like "cividis" or "turbo" tells you nothing about what the
        plot will look like, which is the entire reason a picker beats a text field here.
        """
        kind = layerops.layer_colormap_kind(layer)
        if kind is None:
            return
        if not widgets.section("Colormap"):
            return

        if kind == "gl_line":
            self._draw_gl_colormap_picker(layer)
            return

        cmap, vmin, vmax = layerops.layer_colormap(layer)
        current = cmap or "viridis"
        if kind == "values3d":
            imgui.text_disabled("Maps the values this layer was coloured from (z, or a column).")
        elif kind == "image":
            imgui.text_disabled("Maps the image matrix. The texture is rebuilt on the GPU.")

        names = list(layerops.CPU_COLORMAP_NAMES)
        if current not in names:
            # A layer built in code may carry any name matplotlib knows; show it rather
            # than silently previewing someone else's palette as if it were selected.
            names.insert(0, current)
        for name in names:
            if self._cmap_swatch(
                f"lcmap_{name}", name, cpu_colormap_strip_colors(name), name == current
            ):
                self._set_layer_colormap(layer, cmap=name)

        imgui.separator()
        auto = vmin is None and vmax is None
        changed, new_auto = imgui.checkbox("Auto range", auto)
        if changed:
            if new_auto:
                self._set_layer_colormap(layer, vmin=None, vmax=None)
            else:
                # Seed the manual range with what autoscaling was already using, so
                # unticking the box does not move the picture before the user drags.
                lo, hi = self._colormap_data_range(layer)
                self._set_layer_colormap(layer, vmin=lo, vmax=hi)
        imgui.same_line()
        widgets.help_marker(
            "On: the ends of the ramp track the data's own min and max. Off: pin them, "
            "which is what makes two layers (or two figures) comparable — the same colour "
            "means the same number only when the range is fixed."
        )

        if not auto:
            lo, hi = self._colormap_data_range(layer)
            changed, value = widgets.labeled_drag_float(
                "vmin", float(vmin if vmin is not None else lo), speed=0.01, fmt="%.4g"
            )
            if changed:
                self._set_layer_colormap(layer, vmin=value)
            changed, value = widgets.labeled_drag_float(
                "vmax", float(vmax if vmax is not None else hi), speed=0.01, fmt="%.4g"
            )
            if changed:
                self._set_layer_colormap(layer, vmax=value)

    def _colormap_data_range(self, layer: Any) -> Tuple[float, float]:
        """``(min, max)`` of the scalars a layer's colours are mapped from.

        Only used to seed the manual vmin/vmax fields with the values autoscaling was
        already using. Falls back to ``(0, 1)`` when the scalars are gone or all
        non-finite, because a seed of ``(inf, -inf)`` would be a worse lie than a
        placeholder the user is about to type over anyway.
        """
        metadata = getattr(layer, "metadata", None)
        values = None
        if isinstance(metadata, dict):
            values = metadata.get("cvalues")
            if values is None:
                values = metadata.get("zdata")
            if values is None:
                values = metadata.get("matrix")
        if values is None:
            return (0.0, 1.0)
        try:
            arr = np.asarray(values, dtype=np.float64).ravel()
            finite = arr[np.isfinite(arr)]
        except (TypeError, ValueError):  # pragma: no cover - exotic retained payload
            return (0.0, 1.0)
        if finite.size == 0:
            return (0.0, 1.0)
        return (float(finite.min()), float(finite.max()))

    def _draw_gl_colormap_picker(self, layer: Any) -> None:
        """The shader colormap override for a polyline / line family.

        Three states, not two, and the third is the default: ``None`` means *follow the
        scene*, i.e. obey the Scene tab's "Colormap Lines by Index" and the Density tab's
        scheme. Only a layer the user has explicitly overridden departs from it — which is
        what lets the global toggle keep meaning something after this picker has been
        touched on one layer.
        """
        enabled, index = layerops.layer_gl_colormap(layer)
        options = self.plot.options
        states = ("Follow scene", "On", "Off")
        current = states[0] if enabled is None else (states[1] if enabled else states[2])
        changed, picked = widgets.enum_combo(
            "Colormap by index",
            current,
            list(states),
            help=(
                "Colours each line by its index through the shader colormap instead of by "
                "its own colour. This is not a data colormap: the ramp is keyed on which "
                "line it is, not on any value.\n\n"
                "'Follow scene' is the default and obeys the Scene tab's global toggle."
            ),
        )
        if changed:
            self._set_layer_gl_colormap(
                layer, enabled=None if picked == states[0] else (picked == states[1])
            )

        effective = int(index if index is not None else options.density_scheme_index)
        invert = bool(options.density_invert)
        ltc = bool(options.density_light_to_color)
        for i, name in enumerate(layerops.GL_COLORMAP_NAMES):
            colors = colormap_strip_colors(i, invert=invert, light_to_color=ltc)
            if self._cmap_swatch(f"glcmap_{i}", name, colors, i == effective):
                self._set_layer_gl_colormap(layer, scheme_index=i)
        if index is None:
            imgui.text_disabled("Scheme follows the Density tab; click one to pin it here.")

    def _cmap_swatch(
        self, key: str, name: str, colors: Sequence[Tuple[float, float, float]], selected: bool
    ) -> bool:
        """One gradient strip plus its name. Returns True when clicked.

        The same shape as :meth:`_colormap_swatch` but taking the ramp rather than a
        density-scheme index, because the per-layer picker has two namespaces to draw
        (matplotlib names and GL scheme indices) and only one of them is an index. Follows
        CONTRACT §3: read the cursor *before* the invisible button, or the strip lands a
        slot late.
        """
        draw_list = imgui.get_window_draw_list()
        ox, oy = imgui.get_cursor_screen_pos()

        clicked = imgui.invisible_button(f"##{key}", (SWATCH_WIDTH, SWATCH_HEIGHT))
        hovered = imgui.is_item_hovered()

        step = SWATCH_WIDTH / float(max(len(colors), 1))
        for i, rgb in enumerate(colors):
            x0 = ox + i * step
            # +1 on the right edge: adjacent fills must overlap or seams show through.
            draw_list.add_rect_filled(
                (x0, oy),
                (x0 + step + 1.0, oy + SWATCH_HEIGHT),
                imgui.get_color_u32((rgb[0], rgb[1], rgb[2], 1.0)),
            )

        if selected:
            border, thickness = imgui.get_color_u32((1.0, 1.0, 1.0, 1.0)), 2.0
        elif hovered:
            border, thickness = imgui.get_color_u32((1.0, 1.0, 1.0, 0.7)), 1.5
        else:
            border, thickness = imgui.get_color_u32((0.0, 0.0, 0.0, 0.4)), 1.0
        draw_list.add_rect(
            (ox, oy),
            (ox + SWATCH_WIDTH, oy + SWATCH_HEIGHT),
            border,
            rounding=2.0,
            thickness=thickness,
        )

        imgui.same_line()
        if selected:
            imgui.text(name)
        else:
            imgui.text_disabled(name)
        return clicked

    def _set_layer_colormap(self, layer: Any, **fields: Any) -> None:
        """Queue a data-colormap change, re-resolving the layer at drain time.

        The refusal :func:`glplot.gui.layerops.set_layer_colormap` raises for a layer with
        no colormap is caught and shown rather than crashing the frame: the layer can be
        replaced between this frame and the drain (a re-plot from the Data panel), and the
        replacement need not have one.
        """
        plot = self.plot
        layer_id = getattr(layer, "layer_id", None)

        def apply() -> None:
            live = layerops.find_layer(plot, layer_id)
            if live is None:
                return
            try:
                layerops.set_layer_colormap(plot, live, **fields)
            except ValueError as exc:
                self._layer_error = str(exc)
                return
            self._layer_error = ""

        self.submit(apply)

    def _set_layer_gl_colormap(self, layer: Any, **fields: Any) -> None:
        """Queue a shader-colormap override, re-resolving the layer at drain time."""
        plot = self.plot
        layer_id = getattr(layer, "layer_id", None)

        def apply() -> None:
            live = layerops.find_layer(plot, layer_id)
            if live is None:
                return
            try:
                layerops.set_layer_gl_colormap(plot, live, **fields)
            except ValueError as exc:
                self._layer_error = str(exc)
                return
            self._layer_error = ""

        self.submit(apply)

    def _draw_layer_geometry(self, layer: Any, layer_type: str, style: Any) -> None:
        """Line width, point size and point outlines, per the applicability matrix."""
        shows_width = layer_type in LINE_WIDTH_TYPES
        shows_size = _point_size_applies(layer)
        if not shows_width and not shows_size:
            return

        if not widgets.section("Geometry"):
            return

        if shows_width:
            changed, width = widgets.labeled_slider_float(
                "Line Width", float(style.line_width), 0.1, 20.0, help="In pixels, DPI-scaled."
            )
            if changed:
                self._set_style(layer, line_width=width)

        if shows_size:
            changed, size = widgets.labeled_slider_float(
                "Point Size", float(style.point_size), 1.0, 100.0, help="In pixels, DPI-scaled."
            )
            if changed:
                self._set_style(layer, point_size=size)

        if layer_type in OUTLINE_TYPES:
            imgui.separator()
            changed, enabled = imgui.checkbox("Point Outline", style.point_outline_enabled)
            if changed:
                self._set_style(layer, point_outline_enabled=enabled)
            imgui.same_line()
            widgets.help_marker(
                "The 2D scatter's own per-marker ring (matplotlib's edgecolors). Separate "
                "from the general Outline section below, which is a silhouette of the "
                "whole layer — a layer can legitimately want one and not the other."
            )
            if style.point_outline_enabled:
                changed, color = imgui.color_edit4(
                    "Outline Color", _as_rgba(style.point_outline_color, DEFAULT_OUTLINE_COLOR)
                )
                if changed:
                    self._set_style(layer, point_outline_color=tuple(color))
                changed, width = widgets.labeled_slider_float(
                    "Outline Width", float(style.point_outline_width), 0.1, 5.0
                )
                if changed:
                    self._set_style(layer, point_outline_width=width)
        elif shows_size and _is_3d(layer):
            # The 3D marker edge is not this pair of fields — `geometry3d` has no
            # `u_outline_width_px`. Point at the section that does ring a 3D point rather
            # than showing controls that would do nothing, or nothing at all.
            imgui.text_disabled("Marker edge for 3D points: see Outline / Silhouette below.")

    def _draw_layer_outline(self, layer: Any, style: Any) -> None:
        """Outline / silhouette: enable, colour, width, alpha. Every layer type.

        Shown for 2D and 3D alike because the four fields are one feature with one meaning
        — "draw this layer's boundary" — and gating the control per renderer would make the
        same layer's outline appear and disappear as it is converted between kinds. Only
        the types in :data:`OUTLINE_DEAD_TYPES` get a caveat.

        What it draws, stated plainly because these are not the same picture:
        ``Geometry3DRenderer`` grows a **point** sprite and rings it analytically (one
        pass, exact), and dilates **lines and meshes** by redrawing them a dozen times at
        pixel offsets on a circle with a depth bias, which yields the layer's outer
        silhouette *and the boundary of its holes* — not interior creases, and not a
        contour shared with a neighbouring layer.

        Width is in *logical* pixels: the renderer multiplies by ``ctx.dpr``, so an outline
        keeps its apparent thickness on a Retina display and in a scaled ``savefig``.
        Alpha is deliberately independent of the layer's own: an outline on a translucent
        layer exists to keep it readable, so it has to be able to stay opaque while the
        layer fades. The scene-wide fade still applies to both.
        """
        if not widgets.section("Outline / Silhouette", default_open=False):
            return

        enabled = bool(getattr(style, "outline_enabled", False))
        changed, value = imgui.checkbox("Outline", enabled)
        if changed:
            self._set_style(layer, outline_enabled=value)
        imgui.same_line()
        widgets.help_marker(
            "Draws the layer's boundary in its own colour: a ring around each point, a "
            "dilated silhouette around lines and meshes.\n\n"
            "What it is for: keeping overlapping translucent layers apart, making a "
            "surface read as a solid object rather than a gradient, and giving a point "
            "cloud an edge on a background of the same brightness."
        )

        if not enabled:
            imgui.text_disabled("Off: the layer renders exactly as it did before.")
            return

        changed, color = imgui.color_edit4(
            "Silhouette Color",
            _as_rgba(getattr(style, "outline_color", None), DEFAULT_OUTLINE_COLOR),
        )
        if changed:
            self._set_style(layer, outline_color=tuple(color))

        changed, width = widgets.labeled_slider_float(
            "Silhouette Width",
            float(getattr(style, "outline_width", 1.5)),
            0.0,
            8.0,
            fmt="%.1f px",
            help="Logical pixels, DPI-scaled by the renderer. 0 disables it as surely as "
            "the checkbox does. Past ~4 px on a mesh the dilation samples start to show.",
        )
        if changed:
            self._set_style(layer, outline_width=width)

        changed, alpha = widgets.labeled_slider_float(
            "Silhouette Alpha",
            float(getattr(style, "outline_alpha", 1.0)),
            0.0,
            1.0,
            help="Independent of the layer's own alpha, so a 20% point cloud can keep a "
            "solid edge. A translucent silhouette on a mesh can read unevenly where the "
            "dilation copies overlap.",
        )
        if changed:
            self._set_style(layer, outline_alpha=alpha)

        if str(getattr(layer, "layer_type", "")) in OUTLINE_DEAD_TYPES:
            imgui.text_disabled("This layer type has no outline renderer; the fields are")
            imgui.text_disabled("stored but nothing draws them.")

    def _draw_layer_compositing_3d(self, layer: Any, style: Any) -> None:
        """Per-layer blend mode, occlusion and automatic alpha -- for 3D layers only.

        These three are real per-layer controls (unlike :meth:`_draw_layer_blending`, which
        stays figure-wide): ``Geometry3DRenderer.draw`` reads ``style.blend_mode``,
        ``style.depth_write`` and ``style.auto_alpha`` off each layer and overrides the
        figure's state for the length of that layer's draw. The 2D pass still sets one blend
        state for the whole frame, which is why this section is gated on a 3D layer.
        """
        if not widgets.section("Compositing (this layer)", default_open=False):
            return

        # Blend mode. "Figure" is the None state -- inherit options.blend_mode -- and is
        # first so the default reads as "nothing overridden".
        INHERIT = "Figure default"
        mode_names = [INHERIT] + [m.name for m in BlendMode]
        current = INHERIT if style.blend_mode is None else style.blend_mode.name
        changed, name = widgets.enum_combo(
            "Blend Mode##layer3d",
            current,
            mode_names,
            help=(
                "How THIS layer merges with the scene, overriding the figure's mode for its "
                "own draw. 'Additive' is the one to reach for on a volume3d: overlapping "
                "points brighten instead of averaging, so density reads as light."
            ),
        )
        if changed:
            self._set_style(layer, blend_mode=None if name == INHERIT else BlendMode[name])
        if style.blend_mode is not None:
            imgui.text_wrapped(BLEND_MODE_HELP.get(style.blend_mode.name, ""))

        # Occlusion. Three states -- inherit / force on / force off -- so a tri-state combo
        # rather than a checkbox, whose two states cannot express "let the renderer decide".
        OCC = ["Auto (by translucency)", "Always occlude", "Never occlude"]
        occ_current = (
            OCC[0] if style.depth_write is None else (OCC[1] if style.depth_write else OCC[2])
        )
        changed, occ = widgets.enum_combo(
            "Occludes##layer3d",
            occ_current,
            OCC,
            help=(
                "Whether this layer hides what is drawn behind it. Auto turns writes off for "
                "translucent points and lines and on for meshes. Force it off to see through "
                "a surface's own far side; on to make a dense cloud solid."
            ),
        )
        if changed:
            self._set_style(layer, depth_write={OCC[0]: None, OCC[1]: True, OCC[2]: False}[occ])

        # Automatic alpha. 0 is "off" (use the layer's own alpha); above 0 is the target
        # opacity a covered pixel is solved towards.
        auto_on = style.auto_alpha is not None
        changed, auto_on = imgui.checkbox("Auto alpha##layer3d", auto_on)
        imgui.same_line()
        widgets.help_marker(
            "Solve the per-point alpha so a typical covered pixel reaches the target "
            "opacity under the current view -- keeps a cloud's density readable across "
            "zoom instead of saturating out and vanishing in. Point clouds only."
        )
        if changed:
            self._set_style(layer, auto_alpha=0.9 if auto_on else None)
        if style.auto_alpha is not None:
            t_changed, target = imgui.slider_float(
                "Target##autoalpha", float(style.auto_alpha), 0.05, 1.0
            )
            if t_changed:
                self._set_style(layer, auto_alpha=float(target))

    def _draw_layer_blending(self) -> None:
        """The **figure-wide** blend mode. Per-layer 3D compositing is a separate section
        (:meth:`_draw_layer_compositing_3d`); this one stays figure-level because the 2D
        pass sets one blend state for the whole frame.

        ``engine._apply_blending_policy`` sets this once, before the scene pass, from
        ``options.blend_mode``, and ``policy.py`` decides whether blending is on at all from
        the whole scene's primitive count. For a 2D layer there is nothing per-layer to set;
        for a 3D one the per-layer override lives in the section above and this remains the
        default it falls back to.

        Every mode gets a sentence on what it is *for* rather than its GL factors, because
        "SrcAlpha, One" does not answer "which one do I want for a million translucent
        points" — :data:`BLEND_MODE_HELP` does.
        """
        if not widgets.section("Blending (whole figure)", default_open=False):
            return

        options = self.plot.options
        names = [mode.name for mode in BlendMode]
        current = options.blend_mode.name
        changed, name = widgets.enum_combo(
            "Blend Mode",
            current,
            names,
            help=(
                "Applies to the whole figure, not to the selected layer: the engine sets "
                "one blend state before drawing the scene, so there is nothing per-layer "
                "to set. The same control lives on the Scene tab."
            ),
        )
        if changed:
            self._set(options, "blend_mode", BlendMode[name])

        imgui.text_wrapped(BLEND_MODE_HELP.get(name, "(no description yet)"))

        if options.blend_mode == BlendMode.AUTO:
            imgui.text_disabled(
                f"Blending turns off past {int(options.auto_disable_blending_threshold):,} "
                "primitives (Perf tab)."
            )

    def _draw_layer_transform(self, layer: Any) -> None:
        """World-space offset. 2D only — ``geometry3d`` has no offset uniform."""
        if not widgets.section("Transform", default_open=False):
            return

        tx, ty = getattr(layer, "translation", (0.0, 0.0))
        changed, values = imgui.drag_float2("Offset", (float(tx), float(ty)), 0.05)
        if changed:
            self._set_style(layer, translation=(values[0], values[1]))
        imgui.same_line()
        widgets.help_marker("World-space shift. Autoscale bounds follow it.")

        if imgui.button("Reset Offset"):
            self._set_style(layer, translation=(0.0, 0.0))

    def _draw_layer_camera(self, layer: Any) -> None:
        """The scene's 3D camera, reachable from the layer editor.

        These three sliders used to write ``metadata["camera"]`` on the selected layer and
        claimed to "override the scene camera for this layer only". They never really did:
        every ``set_3d_view`` re-synced the whole scene onto one camera, so a per-layer
        override survived exactly until the next orbit — and the label made that look like
        a bug in the drag rather than in the promise.

        Now that the camera is genuine panel state
        (:class:`glplot.core.camera3d.Camera3D`), these write it. The full set of controls
        — projection, roll, pan, box aspect, axis decoration — lives in the View 3D panel;
        this stays because "nudge the elevation while I look at this layer" is a real
        thing to want without changing panels.
        """
        if not widgets.section("3D Camera", default_open=False):
            return

        camera = self.plot.camera3d

        changed, value = widgets.labeled_slider_float("Elevation", float(camera.elev), -90.0, 90.0)
        if changed:
            self._set_scene_camera(elev=value)

        changed, value = widgets.labeled_slider_float("Azimuth", float(camera.azim), -180.0, 180.0)
        if changed:
            self._set_scene_camera(azim=value)

        changed, value = widgets.labeled_slider_float("FOV", float(camera.fov), 15.0, 90.0)
        if changed:
            self._set_scene_camera(fov=value)

        imgui.text_disabled("One camera per panel. Full controls: View 3D panel.")

    def _set_scene_camera(self, **fields: Any) -> None:
        """Queue a scene-camera change through ``set_3d_view`` so the axis box follows."""
        plot = self.plot

        def apply() -> None:
            plot.set_3d_view(**fields)

        self.submit(apply)

    # -- Density ---------------------------------------------------------------

    def _draw_density(self) -> None:
        """The density/heatmap pass and its colormap."""
        options = self.plot.options
        plot = self.plot

        enabled = bool(getattr(plot, "display_density", False))
        changed, new_enabled = imgui.checkbox("Density Mode (Heatmap)", enabled)
        if changed:
            self.submit(lambda: plot.set_density_enabled(new_enabled))
        imgui.same_line()
        widgets.help_marker(
            "Accumulates overlapping geometry into a heatmap instead of drawing each "
            "primitive. The only readable way to look at millions of lines."
        )

        if not enabled:
            imgui.text_disabled("Settings below apply once density mode is on.")

        imgui.separator()

        if widgets.section("Colormap"):
            self._draw_colormap_picker(options)

        if widgets.section("Mapping"):
            changed, value = imgui.checkbox("Invert", options.density_invert)
            if changed:
                self._set(options, "density_invert", value)
            imgui.same_line()
            widgets.help_marker("Flips the ramp: dense areas take the colormap's low end.")

            label = "Light to Color" if options.density_invert else "Dark to Color"
            changed, value = imgui.checkbox(label, options.density_light_to_color)
            if changed:
                self._set(options, "density_light_to_color", value)
            imgui.same_line()
            widgets.help_marker(
                "Uses only the first 75% of the ramp, so it runs white-to-color instead "
                "of white-to-color-to-black. Usually the cleaner look on paper."
            )

            changed, value = imgui.checkbox("Logarithmic", options.density_is_log)
            if changed:
                self._set(options, "density_is_log", value)
            imgui.same_line()
            widgets.help_marker("Compresses the dynamic range so sparse structure survives.")

            changed, value = imgui.checkbox("Weighted Accumulation", options.density_weighted)
            if changed:
                self._set(options, "density_weighted", value)
            imgui.same_line()
            widgets.help_marker(
                "Accumulates each primitive's alpha instead of a flat 1.0, so faint "
                "layers contribute less."
            )

            changed, value = imgui.checkbox("Light Background Mode", options.light_bg_mode)
            if changed:
                # Not a plain option write: the engine owns a stock background per mode and
                # only repaints it when the flag changes (engine.set_background_mode). The
                # checkbox means "give me the light page", so it wants the repaint.
                plot_ref = self.plot
                self.submit(lambda: plot_ref.set_background_mode(value))
            imgui.same_line()
            widgets.help_marker(
                "Switches the page to the stock white or black background and matching "
                "grid ink, and the panels to a light or dark theme. The Styles tab sets "
                "this for you, with its own background instead of the stock one."
            )

        if widgets.section("Intensity"):
            changed, gain = widgets.labeled_drag_float(
                "Gain",
                float(options.density_gain),
                speed=1.0,
                vmin=0.1,
                vmax=10000.0,
                fmt="%.1f",
                help="Multiplies the accumulated count before the colormap.",
            )
            if changed:
                self._set(options, "density_gain", gain)

            changed, step = widgets.labeled_drag_float(
                "Gain Step",
                float(options.density_gain_step),
                speed=0.01,
                vmin=1.01,
                vmax=4.0,
                fmt="%.2f",
                help="Factor applied per keyboard gain nudge.",
            )
            if changed:
                self._set(options, "density_gain_step", step)

            changed, scale = widgets.labeled_slider_float(
                "Inner Resolution",
                float(options.density_resolution_scale),
                0.1,
                1.0,
                fmt="%.2f",
                help="0.5x is ~4x faster; 1.0x is sharpest.",
            )
            if changed:
                self._set(options, "density_resolution_scale", scale)

    def _draw_colormap_picker(self, options: Any) -> None:
        """The 11 ``DENSITY_SCHEMES`` as clickable gradient swatches.

        Driven off ``DENSITY_SCHEMES`` rather than a hardcoded list, so a scheme added to
        ``shaders.py`` shows up here for free.
        """
        invert = bool(options.density_invert)
        ltc = bool(options.density_light_to_color)
        current = int(options.density_scheme_index)

        for index, name in enumerate(DENSITY_SCHEMES):
            if self._colormap_swatch(index, name, index == current, invert, ltc):
                self._set(options, "density_scheme_index", index)

    def _colormap_swatch(
        self, index: int, name: str, selected: bool, invert: bool, light_to_color: bool
    ) -> bool:
        """One gradient strip + name. Returns True when clicked.

        Follows the ``hud.py:555-586`` pattern: read the cursor *before* the invisible
        button (CONTRACT §3 — afterwards it has already advanced and the strip lands a
        slot late), then paint into the window draw list.
        """
        draw_list = imgui.get_window_draw_list()
        ox, oy = imgui.get_cursor_screen_pos()

        clicked = imgui.invisible_button(f"##cmap_{index}", (SWATCH_WIDTH, SWATCH_HEIGHT))
        hovered = imgui.is_item_hovered()

        colors = colormap_strip_colors(index, invert=invert, light_to_color=light_to_color)
        step = SWATCH_WIDTH / float(len(colors))
        for i, (r, g, b) in enumerate(colors):
            x0 = ox + i * step
            # +1 on the right edge: adjacent fills must overlap or seams show through.
            draw_list.add_rect_filled(
                (x0, oy), (x0 + step + 1.0, oy + SWATCH_HEIGHT), imgui.get_color_u32((r, g, b, 1.0))
            )

        if selected:
            border = imgui.get_color_u32((1.0, 1.0, 1.0, 1.0))
            thickness = 2.0
        elif hovered:
            border = imgui.get_color_u32((1.0, 1.0, 1.0, 0.7))
            thickness = 1.5
        else:
            border = imgui.get_color_u32((0.0, 0.0, 0.0, 0.4))
            thickness = 1.0
        draw_list.add_rect(
            (ox, oy),
            (ox + SWATCH_WIDTH, oy + SWATCH_HEIGHT),
            border,
            rounding=2.0,
            thickness=thickness,
        )

        imgui.same_line()
        if selected:
            imgui.text(name)
        else:
            imgui.text_disabled(name)
        return clicked

    # -- Effects ---------------------------------------------------------------

    def _draw_effects(self) -> None:
        """Glow, gradient background and SSAO."""
        visual = self.plot.options.visual

        if widgets.section("Glow"):
            glow = visual.glow
            changed, enabled = imgui.checkbox("Enable Glow", glow.enabled)
            if changed:
                self._set(glow, "enabled", enabled)
            imgui.same_line()
            widgets.help_marker("Bright-pass blur composited back over the scene.")

            changed, value = widgets.labeled_slider_float(
                "Threshold",
                float(glow.threshold),
                0.0,
                1.0,
                help="Only pixels brighter than this bloom.",
            )
            if changed:
                self._set(glow, "threshold", value)

            changed, value = widgets.labeled_slider_float(
                "Intensity", float(glow.intensity), 0.0, 3.0
            )
            if changed:
                self._set(glow, "intensity", value)

            changed, value = widgets.labeled_slider_float(
                "Radius (px)", float(glow.radius_px), 1.0, 32.0
            )
            if changed:
                self._set(glow, "radius_px", value)

            changed, value = widgets.labeled_slider_float(
                "Knee",
                _optional_float(glow, "knee", DEFAULT_GLOW_KNEE),
                0.0,
                1.0,
                fmt="%.2f",
                help=(
                    "How softly a pixel enters the glow, as a fraction of the threshold. "
                    "0 is a hard cut, which makes edges pop in and out as you pan."
                ),
            )
            if changed:
                self._set(glow, "knee", value)

        if widgets.section("Image"):
            names = list(TONEMAP_NAMES)
            index = int(_optional_float(visual, "tonemap_index", 0.0))
            current = names[index] if 0 <= index < len(names) else names[0]
            changed, name = widgets.enum_combo(
                "Tone Map",
                current,
                names,
                help=(
                    "Rolls values above 1.0 back into range. Off is the honest default: "
                    "tone mapping used to switch itself on with Glow, which halved every "
                    "white pixel the moment you ticked the box. Turn it on when additive "
                    "blending or glow is blowing out your highlights — ACES keeps colour "
                    "in a clipped highlight better than Reinhard, which greys everything."
                ),
            )
            if changed:
                self._set(visual, "tonemap_index", names.index(name))

            changed, value = widgets.labeled_slider_float(
                "Grain",
                _optional_float(visual, "grain_amount", 0.0),
                0.0,
                0.30,
                fmt="%.3f",
                help=(
                    "Static value noise over the finished image — the tooth in the paper "
                    "for the Chalk style. Locked to the pixel grid, so it does not crawl."
                ),
            )
            if changed:
                self._set(visual, "grain_amount", value)

        if widgets.section("Gradient Background"):
            grad = visual.gradient_background
            changed, enabled = imgui.checkbox("Enable Gradient", grad.enabled)
            if changed:
                self._set(grad, "enabled", enabled)
            imgui.same_line()
            widgets.help_marker("Replaces the flat clear color with a vertical ramp.")

            changed, color = imgui.color_edit3("Top", grad.top_color[:3])
            if changed:
                self._set(grad, "top_color", tuple(color))

            changed, color = imgui.color_edit3("Bottom", grad.bottom_color[:3])
            if changed:
                self._set(grad, "bottom_color", tuple(color))

        if widgets.section("Ambient Occlusion"):
            ssao = visual.ssao
            if not self._scene_has_3d():
                imgui.text_disabled("3D layers only — this scene has none.")

            changed, enabled = imgui.checkbox("Enable SSAO", ssao.enabled)
            if changed:
                self._set(ssao, "enabled", enabled)
            imgui.same_line()
            widgets.help_marker(
                "Depth-based shading that makes 3D geometry read as solid. Per-layer "
                "metadata can force it on individually."
            )

            changed, value = widgets.labeled_slider_float(
                "Strength", float(ssao.strength), 0.0, 1.0
            )
            if changed:
                self._set(ssao, "strength", value)

    # -- Axes ------------------------------------------------------------------

    def _draw_axes(self) -> None:
        """Grid, labels, frame, annotation, ticks — and the gutters they all live in.

        Everything here is engine-side work landed alongside this tab. Before it, the five
        grid/visibility options below were the *entire* axis surface: ``xlabel``/``ylabel``
        were attributes that only the matplotlib savefig fallback ever read, and tick
        density and formatting were hardcoded literals in ``managers/axis.py``.

        A scale control (log/symlog/asinh) is deliberately absent here and is not an
        oversight in the engine sense -- ``xscale()``/``yscale()`` are real now (see
        ``glplot.utils.scale``: each transforms a layer's data at GPU-upload time rather
        than the projection itself, so ``controllers.py``'s ortho matrix, its inverse in
        ``screen_to_world``, and the density accumulator never needed to change). What is
        still missing is purely a GUI gap: nothing in this tab reads or writes
        ``options.axis_scale_x/y`` yet, so a script-set scale is invisible here and there
        is no dropdown to set one from the Style panel. Wiring that in is its own round.
        """
        options = self.plot.options

        self._draw_axis_annotation(options)
        self._draw_axis_ticks(options)

        if widgets.section("Visibility"):
            changed, value = imgui.checkbox("Show Grid", options.axis_show_grid)
            if changed:
                self._set(options, "axis_show_grid", value)

            changed, value = imgui.checkbox("Show Labels", options.axis_show_labels)
            if changed:
                self._set(options, "axis_show_labels", value)

            changed, value = imgui.checkbox("Show Frame", options.axis_show_frame)
            if changed:
                self._set(options, "axis_show_frame", value)

        if widgets.section("Grid"):
            changed, value = widgets.labeled_slider_float(
                "Grid Alpha", float(options.axis_grid_alpha), 0.0, 1.0, fmt="%.2f"
            )
            if changed:
                self._set(options, "axis_grid_alpha", value)

            grid_color = _as_rgba(options.axis_grid_color, AUTO_GRID_COLOR + (1.0,))
            auto = tuple(grid_color[:3]) == AUTO_GRID_COLOR
            changed, new_auto = imgui.checkbox("Auto Contrast", auto)
            if changed:
                # Leaving auto seeds a mid grey: any value other than the sentinel is an
                # override, so seeding with the sentinel itself would be a no-op.
                self._set(
                    options, "axis_grid_color", AUTO_GRID_COLOR if new_auto else (0.5, 0.5, 0.5)
                )
            imgui.same_line()
            widgets.help_marker(
                "On: the grid picks dark or light ink from the background luminance. "
                "Off: it uses the color below."
            )

            if not auto:
                changed, color = imgui.color_edit3("Grid Color", grid_color[:3])
                if changed:
                    self._set(options, "axis_grid_color", tuple(color))

        self._draw_axis_gutters(options)

    def _draw_text_style_controls(
        self,
        options: Any,
        label: str,
        fontsize_attr: str,
        color_attr: str,
        default_pt: float,
    ) -> None:
        """Font size + colour for one text element (a tick, x/y-label, or the title).

        Mirrors the Grid Color idiom above: font size is unset (auto, matching the stock
        look) until touched, and colour defaults to "Auto Contrast" -- the luminance-derived
        ink the renderer has always used, so a plot with a dark background does not go blind
        the moment this tab is opened. Turning Auto Contrast off seeds black, the matplotlib
        default and what a caller asking for "colour, black by default" actually wants.
        """
        changed, value = widgets.labeled_slider_float(
            f"{label} Font Size##{fontsize_attr}",
            float(getattr(options, fontsize_attr, None) or default_pt),
            6.0,
            48.0,
            fmt="%.0f pt",
        )
        if changed:
            self._set(options, fontsize_attr, value)

        color = getattr(options, color_attr, None)
        auto = color is None
        changed, new_auto = imgui.checkbox(f"{label} Auto Color##{color_attr}", auto)
        if changed:
            self._set(options, color_attr, None if new_auto else (0.0, 0.0, 0.0))
        imgui.same_line()
        widgets.help_marker(
            "On: picks dark or light ink from the background luminance, same as always. "
            "Off: uses the color below (black by default)."
        )

        if not auto:
            rgba = _as_rgba(color, (0.0, 0.0, 0.0, 1.0))
            changed, new_color = imgui.color_edit3(f"{label} Color##{color_attr}", rgba[:3])
            if changed:
                self._set(options, color_attr, tuple(new_color))

    def _draw_axis_annotation(self, options: Any) -> None:
        """The x-label, y-label and plot title.

        These read back from the same two channels the renderer resolves, so a label set by
        ``gplt.xlabel()`` shows up in the box rather than the box looking empty next to a
        label that is plainly on screen.
        """
        if not widgets.section("Labels"):
            return

        for label, option_name, attr_name, fontsize_attr, color_attr in (
            ("X Label", "axis_xlabel", "xlabel", "axis_xlabel_fontsize", "axis_xlabel_color"),
            ("Y Label", "axis_ylabel", "ylabel", "axis_ylabel_fontsize", "axis_ylabel_color"),
        ):
            current = str(getattr(options, option_name, "") or "") or str(
                getattr(self.plot, attr_name, "") or ""
            )
            changed, value = imgui.input_text(label, current)
            if changed:
                self._set(options, option_name, value)
            self._draw_text_style_controls(
                options, label, fontsize_attr, color_attr, _MPL_DEFAULT_LABEL_PT
            )

        current_title = str(getattr(options, "axis_title", "") or "") or _plot_title(self.plot)
        changed, value = imgui.input_text("Title", current_title)
        if changed:
            self._set(options, "axis_title", value)
        imgui.same_line()
        widgets.help_marker(
            "The title drawn above the frame. Separate from the OS window caption, which "
            "is what gplt.title() sets -- an untitled window is not captioned on the plot."
        )
        self._draw_text_style_controls(
            options, "Title", "axis_title_fontsize", "axis_title_color", _MPL_DEFAULT_TITLE_PT
        )

        widgets.help_marker(
            "gplt.xlabel() / gplt.ylabel() write these too. The Y label is drawn rotated, "
            "reading upward. All three live in the gutters below and are hidden with "
            "Show Labels."
        )

    def _draw_axis_ticks(self, options: Any) -> None:
        """Tick density, spacing, subdivision, marks, label format, font and colour."""
        if not widgets.section("Ticks", default_open=False):
            return

        self._draw_text_style_controls(
            options, "Tick Label", "axis_tick_fontsize", "axis_tick_color", _MPL_DEFAULT_LABEL_PT
        )

        # 0 == auto for all four. Exposed as ints/floats with an explicit "Auto" caption
        # rather than a checkbox pair, because 0 is already the engine's sentinel.
        for label, name, help_text in (
            ("Count X", "axis_tick_count_x", "Target number of X ticks. 0 = auto from width."),
            ("Count Y", "axis_tick_count_y", "Target number of Y ticks. 0 = auto from height."),
        ):
            changed, value = imgui.input_int(label, int(getattr(options, name, 0) or 0))
            if changed:
                self._set(options, name, max(0, value))
            imgui.same_line()
            widgets.help_marker(help_text)

        for label, name in (("Step X", "axis_tick_step_x"), ("Step Y", "axis_tick_step_y")):
            changed, value = widgets.labeled_drag_float(
                label,
                float(getattr(options, name, 0.0) or 0.0),
                speed=0.01,
                vmin=0.0,
                vmax=0.0,
                fmt="%.4g",
                help="Pin the spacing in data units. 0 = the automatic 1-2-5 ladder. "
                "Overrides the count above.",
            )
            if changed:
                self._set(options, name, max(0.0, value))

        changed, value = imgui.checkbox(
            "Minor Ticks", bool(getattr(options, "axis_minor_ticks", False))
        )
        if changed:
            self._set(options, "axis_minor_ticks", value)

        if getattr(options, "axis_minor_ticks", False):
            changed, value = imgui.input_int(
                "Subdivisions", int(getattr(options, "axis_minor_subdivisions", 5) or 5)
            )
            if changed:
                # Below 2 there is nothing to subdivide; the generator returns empty and the
                # checkbox above would look broken.
                self._set(options, "axis_minor_subdivisions", max(2, value))

        changed, value = imgui.checkbox(
            "Tick Marks", bool(getattr(options, "axis_show_ticks", False))
        )
        if changed:
            self._set(options, "axis_show_ticks", value)
        imgui.same_line()
        widgets.help_marker(
            "Short strokes inward from the bottom and left spines. Off by default: the "
            "axis has only ever drawn grid lines and spines."
        )

        if getattr(options, "axis_show_ticks", False):
            changed, value = widgets.labeled_slider_float(
                "Tick Length",
                float(getattr(options, "axis_tick_len_px", 5.0)),
                0.0,
                20.0,
                fmt="%.0f px",
            )
            if changed:
                self._set(options, "axis_tick_len_px", value)

        changed, value = imgui.input_text(
            "Tick Format", str(getattr(options, "axis_tick_format", "") or "")
        )
        if changed:
            self._set(options, "axis_tick_format", value)
        imgui.same_line()
        widgets.help_marker(
            "Empty = automatic: precision from the tick step, scientific outside "
            "1e-4..1e+6. Or give a form: %.2f, {:.3g}, or .3g. "
            "A form that does not apply falls back to automatic."
        )

    def _draw_axis_gutters(self, options: Any) -> None:
        """The four content-inset margins the labels and annotation live inside.

        Exposed because they are the only thing standing between a long tick label -- or a
        y-label -- and the frame. The renderer places annotation inside whatever gutter it
        is given rather than inventing space, so this is the knob that gives it room.
        """
        if not widgets.section("Gutters", default_open=False):
            return

        for label, name in (
            ("Left", "axis_margin_l"),
            ("Right", "axis_margin_r"),
            ("Bottom", "axis_margin_b"),
            ("Top", "axis_margin_t"),
        ):
            changed, value = widgets.labeled_slider_float(
                label, float(getattr(options, name, 0.0)), 0.0, 200.0, fmt="%.0f px"
            )
            if changed:
                self._set(options, name, value)

        widgets.help_marker(
            "Pixels reserved outside the frame for tick labels and annotation. The "
            "projection reads these, so the cursor keeps tracking the data. Exports "
            "always use the stock gutters, never these."
        )

        if imgui.button("Reset Gutters"):
            self._set(options, "axis_margin_l", DEFAULT_AXIS_MARGINS[0])
            self._set(options, "axis_margin_r", DEFAULT_AXIS_MARGINS[1])
            self._set(options, "axis_margin_b", DEFAULT_AXIS_MARGINS[2])
            self._set(options, "axis_margin_t", DEFAULT_AXIS_MARGINS[3])

    # -- Performance -----------------------------------------------------------

    def _draw_performance(self) -> None:
        """LOD, caching, interaction thresholds and export scale."""
        options = self.plot.options

        if widgets.section("Level of Detail"):
            changed, value = imgui.checkbox("LOD Enabled", options.lod_enabled)
            if changed:
                self._set(options, "lod_enabled", value)
            imgui.same_line()
            widgets.help_marker("Drops primitives while interacting to hold the frame rate.")

            changed, value = widgets.labeled_slider_float(
                "Complexity Budget",
                float(options.lod_target_coverage),
                0.05,
                1.0,
                fmt="%.2f",
                help="Target fraction of screen coverage. Lower = more aggressive culling.",
            )
            if changed:
                self._set(options, "lod_target_coverage", value)

            changed, value = imgui.drag_int(
                "Line Budget / px", int(options.default_line_budget_per_px), 0.2, 1, 64
            )
            if changed:
                self._set(options, "default_line_budget_per_px", int(value))
            imgui.same_line()
            widgets.help_marker("Lines per screen pixel the LOD aims to keep.")

            changed, value = widgets.labeled_drag_float(
                "LOD Cost Hint",
                float(options.global_line_width),
                speed=0.05,
                vmin=0.1,
                vmax=20.0,
                fmt="%.2f",
                help=(
                    "NOT the line width — nothing draws with this. It is only the width "
                    "the LOD assumes when estimating cost. Set per-layer Line Width to "
                    "actually change thickness."
                ),
            )
            if changed:
                self._set(options, "global_line_width", value)

            changed, value = widgets.labeled_drag_float(
                "Blending Cutoff",
                float(options.auto_disable_blending_threshold),
                speed=100000.0,
                vmin=0.0,
                vmax=1e9,
                fmt="%.0f",
                help="Above this primitive count AUTO blend mode turns blending off.",
            )
            if changed:
                self._set(options, "auto_disable_blending_threshold", int(value))

        if widgets.section("Rendering"):
            changed, value = imgui.checkbox("Reactive Rendering", options.reactive_rendering)
            if changed:
                self._set(options, "reactive_rendering", value)
            imgui.same_line()
            widgets.help_marker(
                "On: redraw only when something changed (low CPU, low power). Off: redraw "
                "continuously — burns CPU, but papers over a missed redraw if you ever "
                "see a stale frame."
            )

            changed, value = imgui.checkbox(
                "Clipping Optimization", options.enable_clipping_optimization
            )
            if changed:
                self._set(options, "enable_clipping_optimization", value)
            imgui.same_line()
            widgets.help_marker("Skips geometry outside the view. Turn off only to debug.")

            changed, value = imgui.checkbox(
                "Cached Interaction Path", options.enable_cache_interaction_path
            )
            if changed:
                self._set(options, "enable_cache_interaction_path", value)
            imgui.same_line()
            widgets.help_marker(
                "Pans and zooms a cached image instead of redrawing the scene. Much "
                "smoother on big data; the sharp image returns when you let go."
            )

        if widgets.section("Cache Tuning", default_open=False):
            changed, value = widgets.labeled_slider_float(
                "Refresh (Hz)",
                float(options.cache_refresh_hz),
                1.0,
                60.0,
                fmt="%.1f",
                help="How often the cached impostor may be re-captured.",
            )
            if changed:
                self._set(options, "cache_refresh_hz", value)

            changed, value = widgets.labeled_slider_float(
                "Padding",
                float(options.cache_padding),
                1.0,
                10.0,
                fmt="%.1f",
                help="Screens of extra area captured around the view, so a pan stays inside it.",
            )
            if changed:
                self._set(options, "cache_padding", value)

            changed, value = widgets.labeled_slider_float(
                "Safe Margin",
                float(options.cache_safe_margin),
                0.0,
                1.0,
                fmt="%.2f",
                help="How far into the padding you may pan before a re-capture is forced.",
            )
            if changed:
                self._set(options, "cache_safe_margin", value)

        if widgets.section("Interaction", default_open=False):
            changed, value = imgui.checkbox(
                "Shift Required for Picking", options.shift_required_for_picking
            )
            if changed:
                self._set(options, "shift_required_for_picking", value)
            imgui.same_line()
            widgets.help_marker("Off: hovering picks directly, which costs a pick per frame.")

            changed, value = widgets.labeled_slider_float(
                "Drag Threshold (px)",
                float(options.drag_threshold_px),
                0.0,
                20.0,
                fmt="%.1f",
                help="Movement before a press becomes a drag rather than a click.",
            )
            if changed:
                self._set(options, "drag_threshold_px", value)

            changed, value = widgets.labeled_slider_float(
                "Zoom / Scroll Notch",
                float(options.zoom_scroll_factor),
                1.01,
                2.0,
                fmt="%.2f",
                help="Zoom factor per scroll click.",
            )
            if changed:
                self._set(options, "zoom_scroll_factor", value)

        if widgets.section("Export", default_open=False):
            changed, value = widgets.labeled_slider_float(
                "Export Scale",
                float(options.export_scale),
                0.5,
                8.0,
                fmt="%.1f",
                help="Supersampling factor for savefig. 2.0 gives a crisp print image.",
            )
            if changed:
                self._set(options, "export_scale", value)

        if widgets.section("Window (read-only)", default_open=False):
            imgui.text_disabled("Fixed at construction time.")
            widgets.stat_row("Width", str(options.window_width))
            widgets.stat_row("Height", str(options.window_height))
            widgets.stat_row("Title", str(options.title))
            widgets.stat_row("Multisample", "on" if options.enable_multisample else "off")
