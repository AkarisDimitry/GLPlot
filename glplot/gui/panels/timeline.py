"""The Timeline panel: transport, scrubber, keyframe tracks and easing.

The editing half of :mod:`glplot.core.timeline`. The model there is complete and pure —
keyframes, easing, scenes, a playhead and the playback rules — and knows nothing about
imgui; this panel is the only thing that lets a person *author* it.

Why it looks the way it does
----------------------------
Every animation tool converged on the same three-part layout because the three questions
are genuinely separate, and the panel follows it rather than inventing a fourth:

* a **transport** (where the clip runs from and how fast),
* a **scrubber** (where the playhead is *now* — the single most-used control here, so it
  gets a full-width, 38px-tall hit target rather than a slider row), and
* a **track view** (what is animated, and when), with each keyframe drawn as a diamond at
  its own time so that two tracks' keys line up on the eye rather than in a table.

The diamonds, the row backgrounds and the playhead are drawn into the window's draw list
because imgui has no widget for any of them; the *interaction* is still an
``invisible_button`` per row, so hover, focus and click-through obey imgui's own stacking
rules instead of a hand-rolled hit test that ignores the window in front.

CONTRACT §1.1 applies to every line of :meth:`TimelinePanel.draw`
-----------------------------------------------------------------
Nothing here mutates the timeline (or the scene) directly, drag included: a drag computes
an **absolute** target time from the mouse position and queues ``Track.move`` for it, so a
frame of latency between the gesture and the drain cannot accumulate error the way a
delta would.

Playback
--------
Nothing in the engine advances a :class:`~glplot.core.timeline.Timeline` yet, so the GUI
does: :func:`queue_playback` queues one wall-clock advance per frame. Wall-clock, not
per-frame, so an animation plays at the speed it was authored at whether the window is
managing 12fps or 144 — and the last wall time is stamped on the timeline itself, so the
Presentation panel can call the same function without the two double-advancing.

:func:`apply_timeline` is the minimal applier that makes playback *visible*: layer style
properties and the 3D camera. It is deliberately small and whitelisted. When the engine
grows a real applier this becomes redundant rather than wrong — both do the same
idempotent writes.

Capture
-------
Authoring a track by naming a property in a text field is the wall every first-time user
hits: you have to know that the 3D camera's azimuth is spelled ``azim`` before you can
animate anything at all. :func:`capture_state` and :func:`capture_keyframes` are the way
past it — they read what the figure looks like *right now* and key it, **creating whatever
tracks are missing**. That turns authoring into the gesture every 3D tool already has:
point the camera, capture; move the playhead, point it somewhere else, capture again.

The scopes (:data:`CAPTURE_SCOPES`) exist because "capture" is genuinely ambiguous. A
turntable wants the camera and nothing else; a reveal wants the layers' opacity; a pose
wants both. Capturing everything unconditionally would bury a two-track camera move under
thirty style tracks nobody asked for, so the scope is a visible, remembered choice rather
than a guess.
"""

from __future__ import annotations

import logging
import math
import time as _time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...core.camera3d import MAX_ELEV, MAX_FOV, MIN_FOV, STANDARD_VIEWS
from ...core.timeline import (
    COMMON_FPS,
    EASING_LABELS,
    LOOP_MODES,
    MAX_DURATION,
    MIN_DURATION,
    STEP,
    Keyframe,
    Timeline,
    Track,
    is_blendable,
)
from .. import widgets
from ..icons import icon_button
from ..theme import COLORS
from .base import Panel

try:
    import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - exercised only on GL-less systems
    IMGUI_AVAILABLE = False
    imgui = None

logger = logging.getLogger(__name__)

__all__ = [
    "CAMERA2D_TARGET",
    "CAMERA_TARGET",
    "CAMERA_TARGETS",
    "CAPTURE_SCOPES",
    "CAPTURE_SCOPE_LABELS",
    "GRAND_TOUR_VIEWS",
    "PRESETS",
    "PRESET_BY_KEY",
    "Preset",
    "TimelinePanel",
    "apply_preset",
    "apply_timeline",
    "capture_keyframes",
    "capture_state",
    "live_value",
    "preset_available",
    "queue_playback",
    "snap_time",
    "track_live_value",
    "turntable",
]

#: Width of the per-track label gutter (toggles + name + delete). Shared by the ruler and
#: every row: the two must indent identically or the playhead lies about where a key is.
_LABEL_W = 208.0

#: Height of one track row, and of the scrubber strip above them.
_ROW_H = 24.0
_RULER_H = 38.0

#: Reserved for the track list's scrollbar. Subtracted from the time axis *unconditionally*
#: so the axis does not shift by 16px the moment a sixth track appears.
_SCROLLBAR_W = 16.0

#: Half-diagonal of a keyframe diamond, and the click radius around it. The hit radius is
#: larger than the glyph on purpose — a 5px diamond is a 5px target otherwise.
_KEY_HALF = 5.5
_KEY_HIT = 9.0

#: Minimum height reserved for the scrolling track list.
_TRACKS_H = 132.0

#: The target string that means "this panel's 3D camera" rather than a layer.
#:
#: ``"camera3d"``, **not** ``"camera"``, and that is a correctness fix rather than a
#: rename: :mod:`glplot.anim.applier` — which is what the engine's own
#: ``_advance_timelines`` runs during playback — resolves ``"camera"`` to the *2D* camera
#: and ``"camera3d"`` to the 3D one. A track authored here as ``"camera"`` therefore
#: scrubbed correctly (this module's applier drove the 3D camera) and then did nothing at
#: all when you pressed Play, which is the single most confusing failure this panel could
#: have. ``"camera"`` is still *accepted* below so timelines authored before the fix keep
#: working; nothing new is written with it.
CAMERA_TARGET = "camera3d"

#: The target string for the 2D camera — pan and zoom. Animatable in a 2D figure exactly
#: as the pose is in a 3D one, and the same name :mod:`glplot.anim.applier` uses.
CAMERA2D_TARGET = "camera"

#: Every spelling :func:`apply_timeline` accepts for a camera. Which camera is meant is
#: decided by the *property*, not by the string, so the legacy ``("camera", "azim")`` still
#: drives the 3D camera while ``("camera", "cx")`` drives the 2D one.
CAMERA_TARGETS: Tuple[str, ...] = (CAMERA_TARGET, CAMERA2D_TARGET, "camera2d")

#: Layer style fields :func:`apply_timeline` is willing to drive. A whitelist rather than
#: "whatever ``LayerStyle`` has": an animation that silently rewrote ``label`` or a
#: colormap range would be a very confusing way to find out a track was misnamed.
_ANIMATABLE_STYLE: Tuple[str, ...] = (
    "alpha",
    "visible",
    "zorder",
    "line_width",
    "color",
    "face_color",
    "point_size",
    "size",
)

#: 3D camera fields :func:`apply_timeline` is willing to drive, via ``set_3d_view``.
_ANIMATABLE_CAMERA: Tuple[str, ...] = (
    "elev",
    "azim",
    "roll",
    "distance",
    "fov",
)

#: 2D camera fields :func:`apply_timeline` is willing to drive: the centre and the zoom.
#: The same four :data:`glplot.anim.primitives.CAMERA2D_PROPS` names, so a track authored
#: by the panel and one authored by a script are the same track.
_ANIMATABLE_CAMERA2D: Tuple[str, ...] = ("cx", "cy", "zoom_x", "zoom_y")

#: Style properties a capture records for each layer. A strict subset of
#: :data:`_ANIMATABLE_STYLE`: ``zorder`` and the size aliases are real, animatable
#: properties but nobody reaches for "capture" meaning to pin them, and every extra name
#: here is another track the user has to read past to find the one they wanted.
_CAPTURE_STYLE: Tuple[str, ...] = ("alpha", "visible", "color", "line_width", "point_size")

#: What a capture records. Ordered as the picker offers them, commonest first.
#:
#: * ``view``   — the point of view: the 3D pose in a 3D figure, pan and zoom in a 2D one.
#: * ``layers`` — every layer's opacity, visibility, colour and weight.
#: * ``all``    — both of the above.
#: * ``tracks`` — only what is already animated. The "re-key this pose" scope: it creates
#:   nothing, so it cannot surprise you with tracks you did not ask for.
CAPTURE_SCOPES: Tuple[str, ...] = ("view", "layers", "all", "tracks")

#: Human labels for :data:`CAPTURE_SCOPES`, for the picker.
CAPTURE_SCOPE_LABELS: Dict[str, str] = {
    "view": "View (camera)",
    "layers": "Layers (style)",
    "all": "Everything",
    "tracks": "Existing tracks only",
}

#: Turns a one-click turntable sweeps, and the easing it uses. Linear because an eased
#: orbit accelerates and brakes, which reads as the scene lurching rather than turning.
_TURNTABLE_TURNS = 1.0
_TURNTABLE_EASING = "linear"

#: Attribute stamped on a :class:`~glplot.core.timeline.Timeline` with the wall time of the
#: last queued advance. Lives on the timeline, not on a panel, so two panels driving the
#: same timeline in one frame advance it once between them (see :func:`queue_playback`).
_WALL_ATTR = "_glplot_gui_wall"

_HELP_SCRUB = (
    "Drag anywhere on the strip to move the playhead. The readout shows the time and the "
    "frame number; with 'Snap to frames' on, the playhead can only land on a real frame, "
    "so stepping and scrubbing agree."
)

_HELP_TRACKS = (
    "One row per animated property. Each diamond is a keyframe at its own time - drag it "
    "sideways to retime it, click it to edit its value and easing below. The eye mutes a "
    "track without deleting it; the wave toggles interpolation off, which holds each value "
    "until the next key (what you want for a discrete progression, a label or a flag)."
)

_HELP_LOOP = (
    "What happens at the end: 'once' stops, 'loop' restarts, 'pingpong' plays back "
    "towards the start. Ping-pong is the honest way to show that a transform is "
    "reversible."
)

_HELP_CAPTURE = (
    "Record the figure as it looks right now, at the playhead ({count} properties). "
    "Tracks that do not exist yet are created, so this works on an empty timeline - it is "
    "the quickest way to start.\n\n"
    "The workflow: aim the view, press K, move the playhead, aim it somewhere else, press "
    "K again. In a 3D figure the number keys 1-9 snap to the standard angles, so a "
    "camera move is a handful of keystrokes.\n\n"
    "The picker says what 'the figure' means: the camera, every layer's style, both, or "
    "only what is already animated."
)

_HELP_TURNTABLE = (
    "Turntable: sweep the camera a full turn around the scene across the whole timeline. "
    "Starts from the angle you are looking from now."
)

_HELP_EMPTY = (
    "Two ways in.\n\n"
    "Capture records the figure as it is now and creates whatever tracks it needs. Do it "
    "again at another point on the playhead and you have an animation - that is the whole "
    "loop, and it is the one that gives you exactly what you aimed at.\n\n"
    "A preset builds a whole animation at once: a turntable, a tour of the standard "
    "angles, a staggered reveal of the layers. Every one of them lands as ordinary "
    "keyframes you can then drag, retime and re-ease - there is nothing special about a "
    "preset's tracks once they exist.\n\n"
    "'New track' below is the manual route, for a property neither one covers."
)

_HELP_PRESETS = (
    "A canned animation, built as ordinary keyframes on ordinary tracks - so you can drag "
    "them, retime them and change their easing exactly as if you had keyed them by hand.\n\n"
    "Presets that need something this figure does not have (a 3D camera, a layer) say so "
    "instead of offering a button that would do nothing."
)

_HELP_NEW_TRACK = (
    "The 3D camera animates elev, azim, roll, distance and fov; the 2D camera animates cx, "
    "cy and the two zooms; a layer animates alpha, visible, color, line_width and the point "
    "sizes. Tick 'Type it by hand' for anything else - it is still keyable, it simply has "
    "nothing driving it yet.\n\n"
    "Most of the time Capture is the faster route: it creates these tracks for you."
)


# ----------------------------------------------------------------------------------
# Pure helpers (no imgui) — unit-testable headless
# ----------------------------------------------------------------------------------


def snap_time(timeline: Timeline, seconds: float, *, snap: bool = True) -> float:
    """Clamp ``seconds`` into the timeline, optionally onto the nearest frame."""
    value = float(np.clip(float(seconds), 0.0, max(timeline.duration, 0.0)))
    if not snap or timeline.fps <= 0.0:
        return value
    return float(np.clip(round(value * timeline.fps) / timeline.fps, 0.0, timeline.duration))


def _is_scalar(value: Any) -> bool:
    """True for a plain number — the one value kind the inspector can edit in place."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _describe_value(value: Any) -> str:
    """A one-line readout for any keyframe value, arrays included."""
    if value is None:
        return "-"
    if _is_scalar(value):
        return f"{float(value):.6g}"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, np.ndarray):
        return f"array{tuple(value.shape)} {value.dtype}"
    if isinstance(value, (list, tuple)):
        head = ", ".join(f"{float(v):.4g}" if _is_scalar(v) else str(v) for v in value[:4])
        return f"({head}{', ...' if len(value) > 4 else ''})"
    return str(value)


def _tick_step(duration: float, width: float) -> float:
    """A round tick interval giving roughly one label per 90 px."""
    if duration <= 0.0 or width <= 0.0:
        return 1.0
    target = duration * (90.0 / width)
    if target <= 0.0 or not math.isfinite(target):
        return duration
    magnitude = 10.0 ** math.floor(math.log10(target))
    for step in (1.0, 2.0, 5.0, 10.0):
        if target <= step * magnitude:
            return step * magnitude
    return 10.0 * magnitude


def _fmt_time(seconds: float, step: float) -> str:
    """Format a ruler label with just enough decimals for the tick spacing."""
    if step >= 1.0:
        return f"{seconds:.0f}s"
    if step >= 0.1:
        return f"{seconds:.1f}"
    return f"{seconds:.2f}"


def _prop_options(target: Any) -> Tuple[str, ...]:
    """The properties worth offering for ``target``, for the "New track" picker.

    A whitelist, matching what :func:`apply_timeline` will actually drive. Anything outside
    it is still keyable by hand — the picker has a free-text escape hatch — it simply has
    nothing driving it yet, and putting it in a dropdown would imply otherwise.
    """
    if isinstance(target, str):
        key = target.lower()
        if key == CAMERA2D_TARGET or key == "camera2d":
            return _ANIMATABLE_CAMERA2D
        if key in CAMERA_TARGETS:
            return _ANIMATABLE_CAMERA
    return _ANIMATABLE_STYLE


def track_label(track: Track) -> str:
    """The visible name of a track: its label, or ``target.prop``.

    ``##`` is imgui's id separator, so a label containing it would silently lose half
    itself; the real ``Track.label`` is never modified.
    """
    name = str(track.label) if track.label else f"{track.target}.{track.prop}"
    return name.replace("##", "# #") if "##" in name else name


def _apply_rest(plot: Any, values: Dict[Tuple[Any, str], Any]) -> int:
    """Hand whatever this module's whitelist did not claim to the engine's own applier.

    Import is lazy and the failure is silent: :mod:`glplot.anim` is optional to the core
    (the engine imports it lazily for the same reason), and a figure that never animates
    must not pay for it.
    """
    try:
        from ...anim.applier import apply_values
    except Exception:  # pragma: no cover - the package may not be installed
        return 0
    try:
        return int(apply_values(plot, values))
    except Exception:  # pragma: no cover - defensive: the applier never raises
        logger.debug("Timeline could not apply %d extra properties.", len(values), exc_info=True)
        return 0


def apply_timeline(plot: Any, timeline: Timeline) -> int:
    """Push every animated value at the playhead onto the scene. Returns how many landed.

    The panel's own minimal applier, and the reason "play" does something visible at all:
    the engine has no timeline applier yet. Two target kinds are understood — a layer id
    (drives :data:`_ANIMATABLE_STYLE` through ``layerops.set_layer_style``, which owns the
    3D re-upload trap) and :data:`CAMERA_TARGET` (drives :data:`_ANIMATABLE_CAMERA` through
    ``set_3d_view``, which rebuilds the axis box).

    Anything the whitelist does not claim is handed to :mod:`glplot.anim.applier` — the
    same module the engine runs during playback. That fallback is what keeps **scrubbing and
    playing showing the same picture**: a preset that keys ``draw_fraction`` or a colormap
    would otherwise animate under Play and sit inert under the playhead, which reads as the
    preset being broken. The whitelist still goes first, because it owns the legacy
    ``"camera"`` spelling that the engine's applier resolves to the *other* camera.

    **Must be called from the queue**, never from a draw callback: both of those calls are
    scene mutation (CONTRACT §1.1). Nothing raises out of here — a track naming a property
    that no longer exists simply does not apply, because a timeline is scrubbed
    continuously and a half-edited track must not take the frame down.
    """
    values = timeline.evaluate()
    if not values:
        return 0

    try:
        from .. import layerops
    except Exception:  # pragma: no cover - defensive: GL-less import
        return 0

    layers: Dict[Any, Any] = {}
    try:
        for layer in plot.scene.layers:
            layers[getattr(layer, "layer_id", None)] = layer
    except Exception:  # pragma: no cover - defensive: a plot without a scene
        return 0

    applied = 0
    camera_fields: Dict[str, Any] = {}
    camera2d_fields: Dict[str, float] = {}
    rest: Dict[Tuple[Any, str], Any] = {}
    for (target, prop), value in values.items():
        if value is None:
            continue
        if str(target).lower() in CAMERA_TARGETS:
            # By property, not by target string: the legacy ``"camera"`` spelling means the
            # 3D camera when it carries a pose and the 2D one when it carries pan or zoom.
            if prop in _ANIMATABLE_CAMERA:
                camera_fields[prop] = float(value) if _is_scalar(value) else value
                continue
            if prop in _ANIMATABLE_CAMERA2D and _is_scalar(value):
                camera2d_fields[prop] = float(value)
                continue
            rest[(target, prop)] = value
            continue
        layer = layers.get(target)
        if layer is None or prop not in _ANIMATABLE_STYLE:
            rest[(target, prop)] = value
            continue
        try:
            layerops.set_layer_style(plot, layer, **{prop: value})
            applied += 1
        except Exception:
            logger.debug("Timeline could not set %r on layer %r.", prop, target, exc_info=True)

    if rest:
        applied += _apply_rest(plot, rest)

    if camera2d_fields:
        try:
            camera = plot.camera
            for name, number in camera2d_fields.items():
                if name.startswith("zoom"):
                    # An overshooting easing (back, elastic) really can drive a zoom key
                    # through zero on its way to the target, and the projection divides by
                    # it. Clamped to the camera's own limits, as the engine's applier does.
                    lo = float(getattr(camera, "zoom_min", 1e-12))
                    hi = float(getattr(camera, "zoom_max", 1e12))
                    number = float(np.clip(number, lo, hi))
                setattr(camera, name, number)
            layerops.mark_scene_dirty(plot)
            applied += len(camera2d_fields)
        except Exception:
            logger.debug("Timeline could not drive the 2D camera.", exc_info=True)

    if camera_fields:
        try:
            if plot.is_3d_scene():
                # `set_3d_view` rebuilds the axis-box geometry, which is the whole reason
                # it must be the entry point in a 3D figure — and the whole reason it must
                # NOT be one in a 2D figure, where it would grow 3D decoration nobody
                # asked for. The camera state is still animated either way.
                plot.set_3d_view(**camera_fields)
            else:
                for name, value in camera_fields.items():
                    setattr(plot.camera3d, name, value)
            applied += len(camera_fields)
        except Exception:
            logger.debug("Timeline could not drive the 3D camera.", exc_info=True)
    return applied


# ----------------------------------------------------------------------------------
# Capture — reading the figure's current state back out as keyframes
# ----------------------------------------------------------------------------------


def live_value(plot: Any, target: Any, prop: str, fallback: Any = None) -> Any:
    """What ``target``'s ``prop`` is on ``plot`` *right now*, or ``fallback``.

    The single place "read the live value of an animatable property" is implemented, so
    capture, the panel's key-at-playhead button and any future auto-key all agree about
    where a value comes from. Never raises: this is called once per track per capture and
    an exotic target must degrade to ``fallback``, not take the frame down.
    """
    prop = str(prop)
    try:
        if isinstance(target, str) and target.lower() in CAMERA_TARGETS:
            if prop in _ANIMATABLE_CAMERA:
                value = getattr(plot.camera3d, prop, None)
                return fallback if value is None else value
            if prop in _ANIMATABLE_CAMERA2D:
                value = getattr(plot.camera, prop, None)
                return fallback if value is None else value
            return fallback
        for layer in plot.scene.layers:
            if getattr(layer, "layer_id", None) != target:
                continue
            style = getattr(layer, "style", None)
            if style is not None and hasattr(style, prop):
                return getattr(style, prop)
            if hasattr(layer, prop):
                return getattr(layer, prop)
    except Exception:  # pragma: no cover - defensive: a plot without a scene
        logger.debug("Could not read %r from %r.", prop, target, exc_info=True)
    return fallback


def track_live_value(plot: Any, track: Track, time: float) -> Any:
    """:func:`live_value` for a whole track, with the fallbacks a keyframe needs.

    Falls back to the track's own interpolated value, then to ``0.0`` — a keyframe with no
    value at all would be a hole in the curve.
    """
    value = live_value(plot, track.target, track.prop)
    if value is not None:
        return value
    value = track.value_at(float(time))
    return 0.0 if value is None else value


def _capture_view(plot: Any) -> List[Tuple[Any, str, Any]]:
    """The point of view: the 3D pose in a 3D figure, pan and zoom in a 2D one.

    Which one is chosen from the figure, not from a setting, because "capture the view" in
    a 2D figure can only mean the 2D camera — offering a 3D pose there would key five
    properties nothing draws.
    """
    try:
        is_3d = bool(plot.is_3d_scene())
    except Exception:  # pragma: no cover - defensive: an engine without the probe
        is_3d = False
    target = CAMERA_TARGET if is_3d else CAMERA2D_TARGET
    props = _ANIMATABLE_CAMERA if is_3d else _ANIMATABLE_CAMERA2D

    out: List[Tuple[Any, str, Any]] = []
    for prop in props:
        value = live_value(plot, target, prop)
        # ``distance=None`` means "fit the data" and is a real camera state, but it is not
        # a keyframe value: a track holding None applies nothing, so it would be a row in
        # the editor that does nothing forever.
        if value is not None:
            out.append((target, prop, value))
    return out


def _capture_layers(plot: Any) -> List[Tuple[Any, str, Any]]:
    """Every layer's :data:`_CAPTURE_STYLE` properties, in scene order."""
    out: List[Tuple[Any, str, Any]] = []
    try:
        layers = list(plot.scene.layers)
    except Exception:  # pragma: no cover - defensive: a plot without a scene
        return out
    for layer in layers:
        layer_id = getattr(layer, "layer_id", None)
        if layer_id is None:
            continue
        style = getattr(layer, "style", None)
        if style is None:
            continue
        for prop in _CAPTURE_STYLE:
            if not hasattr(style, prop):
                continue
            value = getattr(style, prop)
            if value is not None:
                out.append((layer_id, prop, value))
    return out


def capture_state(
    plot: Any, scope: str = "view", *, timeline: Optional[Timeline] = None
) -> List[Tuple[Any, str, Any]]:
    """Read what ``scope`` covers off the live figure as ``(target, prop, value)``.

    A pure read — nothing here mutates the plot or the timeline, which is what lets the
    panel call it from ``draw`` to count what a capture *would* record and then queue the
    writes separately.

    ``scope="tracks"`` needs ``timeline`` (it captures exactly what is already animated);
    the other scopes ignore it.

    Raises:
        ValueError: on an unknown scope. A typo that silently captured nothing would look
            exactly like a figure with nothing to capture.
    """
    key = str(scope).strip().lower()
    if key not in CAPTURE_SCOPES:
        raise ValueError(
            f"unknown capture scope {scope!r}; expected one of {', '.join(CAPTURE_SCOPES)}"
        )

    if key == "tracks":
        if timeline is None:
            return []
        return [
            (track.target, track.prop, track_live_value(plot, track, timeline.time))
            for track in timeline.tracks
        ]

    out: List[Tuple[Any, str, Any]] = []
    if key in ("view", "all"):
        out.extend(_capture_view(plot))
    if key in ("layers", "all"):
        out.extend(_capture_layers(plot))
    return out


def capture_keyframes(
    plot: Any,
    timeline: Timeline,
    *,
    scope: str = "view",
    time: Optional[float] = None,
    easing_name: str = "smooth",
) -> List[Track]:
    """Key the figure's current state onto ``timeline``, **creating the missing tracks**.

    The gesture the whole capture design exists for: point the camera, press capture, move
    the playhead, point it somewhere else, press capture again — and an animation exists
    without anyone having typed the word ``azim``.

    **Queued commands only** (CONTRACT §1.1): this writes the timeline, and the timeline
    drives the scene.

    A value that cannot be blended (a visibility flag, a colormap name) is keyed with
    :data:`~glplot.core.timeline.STEP` easing, and a *fresh* track holding one is switched
    to non-interpolating. Only a fresh one: overriding that flag on a track the user
    already configured would silently undo their choice.

    Returns the tracks that were written, in capture order.
    """
    entries = capture_state(plot, scope, timeline=timeline)
    if not entries:
        return []
    at = timeline.time if time is None else float(time)

    touched: List[Track] = []
    for target, prop, value in entries:
        track = timeline.track(target, prop)
        if track is None:  # pragma: no cover - create=True never returns None
            continue
        blendable = is_blendable(value)
        if not blendable and not track.keyframes:
            track.interpolate = False
        track.add(at, value, STEP if not blendable else easing_name)
        touched.append(track)
    timeline.fit_duration()
    return touched


def turntable(
    plot: Any,
    timeline: Timeline,
    *,
    turns: float = _TURNTABLE_TURNS,
    start: float = 0.0,
    duration: Optional[float] = None,
) -> List[Track]:
    """Sweep the 3D camera's azimuth through ``turns`` whole turns. Returns the tracks.

    The one-click animation, and the reason the empty timeline has a button rather than a
    paragraph: a turntable is what most people want the first time they animate a 3D
    figure, and building it by hand means knowing the property name, the two keyframe
    times *and* that the second value must be 360 rather than 0.

    Delegates to :class:`glplot.anim.primitives.Orbit` rather than writing two keyframes
    here, so the turntable this button makes and the one a script makes are the same
    object — including the baked intermediate keys that make the sweep readable in the
    track view.

    **Queued commands only** (CONTRACT §1.1).
    """
    from ...anim.primitives import Orbit

    span = float(timeline.duration) if duration is None else float(duration)
    orbit = Orbit.from_camera(plot.camera3d, turns=float(turns), easing=_TURNTABLE_EASING)
    return orbit.apply(timeline, CAMERA_TARGET, float(start), max(span, MIN_DURATION))


# ----------------------------------------------------------------------------------
# Presets — a whole animation from one click
# ----------------------------------------------------------------------------------
#
# Every builder below is (plot, timeline, start, duration) -> None and writes through
# ``glplot.anim.primitives``, not by hand. That is the point of routing them through the
# verbs: the turntable this menu builds and the turntable a script builds are the same
# keyframes, so a preset is a starting point you can then edit, script, or read as an
# example — rather than a black box the GUI alone knows how to make.


#: The orientations "Grand tour" visits, in order. Chosen to avoid the ±180° azimuth seam:
#: an interpolation across it unwinds the long way round, which reads as the scene
#: spinning backwards for no reason.
GRAND_TOUR_VIEWS: Tuple[str, ...] = ("front", "iso", "top", "iso_back_right")

#: Degrees either side of the current azimuth that "Rock" sways.
_ROCK_DEGREES = 22.0

#: What "Push in" multiplies the field of view by. Narrower lens, same framing centre.
_PUSH_IN_FACTOR = 0.55

#: What "Zoom in" multiplies the 2D camera's zoom by.
_ZOOM_2D_FACTOR = 2.5

#: Overlap between consecutive layers in a staggered preset — Manim's ``LaggedStart``
#: default. 0 would flash them all at once, 1 would make a strict queue.
_REVEAL_LAG = 0.25


def _layer_ids(plot: Any) -> List[Any]:
    """Every layer id in scene order. Empty when the figure has no layers."""
    try:
        return [
            layer.layer_id
            for layer in plot.scene.layers
            if getattr(layer, "layer_id", None) is not None
        ]
    except Exception:  # pragma: no cover - defensive: a plot without a scene
        return []


def _key_curve(
    timeline: Timeline,
    target: Any,
    prop: str,
    start: float,
    duration: float,
    points: Sequence[Tuple[float, Any]],
    easing_name: str = "smooth",
) -> Track:
    """Write ``(fraction, value)`` pairs onto one track as keyframes. Returns the track.

    For the shapes no verb expresses — an oscillation is a *there and back and there
    again*, which :class:`~glplot.anim.primitives.CameraMoveTo` would need four of.
    """
    track = timeline.track(target, prop)
    assert track is not None  # create=True
    for fraction, value in points:
        track.add(start + duration * float(fraction), value, easing_name)
    return track


def _stagger_over_layers(
    timeline: Timeline,
    animations: Sequence[Tuple[Any, Any]],
    start: float,
    duration: float,
    lag_ratio: float = _REVEAL_LAG,
) -> None:
    """Schedule one ``(target, verb)`` per layer, overlapping, to fill exactly ``duration``.

    The per-play run time is solved for rather than guessed: with Manim's lag rule a group
    of ``n`` plays ends at ``per * (1 + lag * (n - 1))``, so filling a 5-second timeline
    with eight layers means each fade is 2.2s, not 5s (which would overrun by 4x) and not
    0.6s (which would finish in the first fifth and look like a glitch).
    """
    from ...anim.primitives import Play, group

    count = len(animations)
    if not count:
        return
    per = float(duration) / max(1.0 + lag_ratio * (count - 1), 1e-9)
    plays = [
        Play(animation=animation, target=target, duration=per) for target, animation in animations
    ]
    group(timeline, plays, start=float(start), lag_ratio=lag_ratio, duration=per)


def _build_turntable(plot: Any, timeline: Timeline, start: float, duration: float) -> None:
    turntable(plot, timeline, start=start, duration=duration)


def _build_grand_tour(plot: Any, timeline: Timeline, start: float, duration: float) -> None:
    from ...anim.primitives import CameraMoveTo

    per = float(duration) / len(GRAND_TOUR_VIEWS)
    at = float(start)
    for index, name in enumerate(GRAND_TOUR_VIEWS):
        elev, azim, roll = STANDARD_VIEWS[name]
        if index == 0:
            # The first leg has to be seeded from the live camera: INHERIT on a fresh track
            # has nothing to inherit, so the leg would collapse to its destination key and
            # the tour would begin by snapping rather than moving.
            verb: Any = CameraMoveTo.from_camera(plot.camera3d, elev=elev, azim=azim, roll=roll)
        else:
            verb = CameraMoveTo(elev=elev, azim=azim, roll=roll)
        verb.apply(timeline, CAMERA_TARGET, at, per)
        at += per


def _build_rock(plot: Any, timeline: Timeline, start: float, duration: float) -> None:
    base = float(getattr(plot.camera3d, "azim", 0.0))
    amp = _ROCK_DEGREES
    _key_curve(
        timeline,
        CAMERA_TARGET,
        "azim",
        start,
        duration,
        ((0.0, base), (0.25, base + amp), (0.5, base), (0.75, base - amp), (1.0, base)),
        "smooth",
    )


def _build_push_in(plot: Any, timeline: Timeline, start: float, duration: float) -> None:
    from ...anim.primitives import ZoomTo

    camera = plot.camera3d
    narrowed = float(np.clip(float(camera.fov) * _PUSH_IN_FACTOR, MIN_FOV, MAX_FOV))
    ZoomTo.from_camera(camera, fov=narrowed).apply(timeline, CAMERA_TARGET, start, duration)


def _build_tilt_up(plot: Any, timeline: Timeline, start: float, duration: float) -> None:
    from ...anim.primitives import CameraMoveTo

    CameraMoveTo.from_camera(plot.camera3d, elev=float(MAX_ELEV)).apply(
        timeline, CAMERA_TARGET, start, duration
    )


def _build_reveal(plot: Any, timeline: Timeline, start: float, duration: float) -> None:
    from ...anim.primitives import FadeIn

    pairs = []
    for layer in getattr(plot.scene, "layers", []):
        layer_id = getattr(layer, "layer_id", None)
        if layer_id is None:
            continue
        # Fade up to the alpha the layer actually has, not to 1: a layer the user
        # deliberately left at 0.3 must not be opaque when the reveal finishes.
        alpha = float(getattr(getattr(layer, "style", None), "alpha", 1.0) or 1.0)
        pairs.append((layer_id, FadeIn(to=alpha)))
    _stagger_over_layers(timeline, pairs, start, duration)


def _build_fade_out(plot: Any, timeline: Timeline, start: float, duration: float) -> None:
    from ...anim.primitives import FadeOut

    pairs = []
    for layer in getattr(plot.scene, "layers", []):
        layer_id = getattr(layer, "layer_id", None)
        if layer_id is None:
            continue
        alpha = float(getattr(getattr(layer, "style", None), "alpha", 1.0) or 1.0)
        pairs.append((layer_id, FadeOut(start_alpha=alpha)))
    _stagger_over_layers(timeline, pairs, start, duration)


def _build_draw_on(plot: Any, timeline: Timeline, start: float, duration: float) -> None:
    from ...anim.primitives import Create

    _stagger_over_layers(timeline, [(lid, Create()) for lid in _layer_ids(plot)], start, duration)


def _build_zoom_in_2d(plot: Any, timeline: Timeline, start: float, duration: float) -> None:
    from ...anim.primitives import ZoomTo

    camera = plot.camera
    zoom = float(getattr(camera, "zoom_x", 1.0)) * _ZOOM_2D_FACTOR
    ZoomTo.from_camera(camera, zoom=zoom).apply(timeline, CAMERA2D_TARGET, start, duration)


@dataclass(frozen=True)
class Preset:
    """One canned animation: what it is called, what it needs, and how to build it.

    ``requires`` is the figure it makes sense in — ``"3d"``, ``"2d"``, ``"layers"`` (at
    least one layer, any dimensionality) or ``"any"``. Presets that do not apply are shown
    disabled rather than hidden, for the reason the 3D actions are: a greyed "Turntable"
    tells a 2D user that 3D exists, an absent one tells them nothing.
    """

    key: str
    label: str
    description: str
    requires: str
    build: Callable[[Any, Timeline, float, float], None]
    #: A name from :data:`glplot.gui.icons.ICON_SHAPES`, for the command palette.
    icon: str = "timeline"


#: Every preset, in menu order: the camera moves first (the commonest thing to want), then
#: the layer reveals, then the 2D move.
PRESETS: Tuple[Preset, ...] = (
    Preset(
        "turntable",
        "Turntable (360 deg)",
        "One full turn of the camera, starting from the angle you are looking from now. "
        "The signature 3D preset, and the one that reads a point cloud's structure best.",
        "3d",
        _build_turntable,
        "orbit",
    ),
    Preset(
        "grand_tour",
        "Grand tour (4 angles)",
        "Front, isometric, top, then back round to isometric — a leg per quarter of the "
        "timeline. The quickest way to show a shape from every side.",
        "3d",
        _build_grand_tour,
        "view_iso",
    ),
    Preset(
        "rock",
        "Rock back and forth",
        "A gentle sway either side of the current azimuth. Parallax without a full turn, "
        "which is what makes depth read in a still-ish figure.",
        "3d",
        _build_rock,
        "spin",
    ),
    Preset(
        "push_in",
        "Push in (narrow the lens)",
        "Narrows the field of view, so the scene grows without the camera moving. Works in "
        "both projections.",
        "3d",
        _build_push_in,
        "zoom_in",
    ),
    Preset(
        "tilt_up",
        "Tilt to top-down",
        "Raises the elevation to a plan view, keeping the azimuth. Good for showing that a "
        "3D structure has a 2D footprint.",
        "3d",
        _build_tilt_up,
        "view_top",
    ),
    Preset(
        "reveal",
        "Reveal layers one by one",
        "Fades each layer up in turn, overlapping. Every layer ends at the opacity it "
        "already has, so nothing is left brighter than you set it.",
        "layers",
        _build_reveal,
        "layers",
    ),
    Preset(
        "draw_on",
        "Draw on (trace the geometry)",
        "Draws each layer along its own path — a line traces itself, a scatter fills in "
        "point by point. Staggered across the layers.",
        "layers",
        _build_draw_on,
        "chart_line",
    ),
    Preset(
        "fade_out",
        "Fade out",
        "Takes every layer down to transparent, staggered. The ending half of a reveal; "
        "apply it from the playhead to put it after something else.",
        "layers",
        _build_fade_out,
        "eye_off",
    ),
    Preset(
        "zoom_in_2d",
        "Zoom in (2D)",
        "Zooms the 2D camera in on the current centre. The 2D counterpart of Push in.",
        "2d",
        _build_zoom_in_2d,
        "zoom_in",
    ),
)

#: :data:`PRESETS` by key, for lookup from an action id or a saved setting.
PRESET_BY_KEY: Dict[str, Preset] = {preset.key: preset for preset in PRESETS}

#: Why a preset is unavailable, phrased to complete "<preset label> ...". Shown in place of
#: the Build button, because a greyed button that says nothing is the same as a broken one.
PRESET_REQUIREMENT_LABELS: Dict[str, str] = {
    "3d": "needs a 3D figure",
    "2d": "needs a 2D figure",
    "layers": "needs at least one layer",
}


def preset_available(plot: Any, preset: Preset) -> bool:
    """Whether ``preset`` can do anything in this figure. Never raises."""
    try:
        if preset.requires == "3d":
            return bool(plot.is_3d_scene())
        if preset.requires == "2d":
            return not bool(plot.is_3d_scene())
        if preset.requires == "layers":
            return bool(_layer_ids(plot))
    except Exception:  # pragma: no cover - defensive: an engine without the probe
        return False
    return True


def apply_preset(
    plot: Any,
    timeline: Timeline,
    key: str,
    *,
    start: float = 0.0,
    duration: Optional[float] = None,
) -> List[Track]:
    """Build the preset called ``key`` onto ``timeline``. Returns the tracks it touched.

    ``duration`` defaults to the whole timeline from ``start``, which is what "quick
    animation" means — the preset fills the clip. Pass ``start`` to chain one after
    another (reveal, then turntable).

    **Queued commands only** (CONTRACT §1.1).

    Raises:
        ValueError: on an unknown key. A typo that silently built nothing would look
            exactly like a preset that had nothing to do.
    """
    preset = PRESET_BY_KEY.get(str(key).strip().lower())
    if preset is None:
        raise ValueError(f"unknown preset {key!r}; expected one of {', '.join(PRESET_BY_KEY)}")
    start = float(start)
    span = (float(timeline.duration) - start) if duration is None else float(duration)
    span = max(span, MIN_DURATION)

    # Counted by identity rather than reported by the builders: the sequencing helpers
    # return an end time, not a track list, and a diff answers "what did this touch"
    # without every builder having to thread the bookkeeping through.
    before = {id(track): len(track.keyframes) for track in timeline.tracks}
    preset.build(plot, timeline, start, span)
    timeline.fit_duration()
    return [t for t in timeline.tracks if len(t.keyframes) != before.get(id(t), 0)]


def engine_drives_playback(panel: Panel) -> bool:
    """Whether the engine's own main loop is advancing timelines.

    ``GPULinePlot._advance_timelines`` steps **every panel's** timeline once per frame from
    wall clock and applies the result through :mod:`glplot.anim.applier`. Where that exists,
    this panel must not advance the clock as well — two independent advances in one frame
    play the animation at double speed, and the second one is invisible in every test that
    exercises the panel alone.

    Probed rather than assumed so the panel still works against an engine that predates the
    hook, which is the state it was written against.
    """
    return callable(getattr(panel.plot, "_advance_timelines", None))


def queue_playback(panel: Panel) -> None:
    """Queue this frame's wall-clock advance of ``panel``'s timeline.

    **A no-op when the engine drives playback** (see :func:`engine_drives_playback`). The
    engine's hook landed after this panel was written; keeping both would double the
    playback rate.

    On an engine without the hook this is the fallback, and it is safe to call from more
    than one panel in the same frame: the last wall time is stamped on the timeline, so the
    second caller sees a ``dt`` of a few microseconds and the two together advance exactly
    once. That is what lets the Presentation panel drive playback while the Timeline panel
    is closed, and both of them while it is open.
    """
    if engine_drives_playback(panel):
        return

    timeline = panel.plot.active_panel.timeline
    now = _time.perf_counter()

    def apply() -> None:
        last = getattr(timeline, _WALL_ATTR, None)
        setattr(timeline, _WALL_ATTR, now)
        # A paused timeline still re-stamps: otherwise resuming after a minute of editing
        # would hand `advance` a 60-second dt (which it rejects) and stall for one frame.
        if not timeline.playing or last is None:
            return
        if timeline.advance(now - float(last)):
            apply_timeline(panel.plot, timeline)

    panel.submit(apply)


# ----------------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------------


def _col(name: str, alpha: Optional[float] = None) -> int:
    """Pack a theme token for draw-list use, tolerating a theme that never applied."""
    rgba = COLORS.get(name, (0.85, 0.86, 0.88, 1.0))
    return imgui.get_color_u32_rgba(rgba[0], rgba[1], rgba[2], rgba[3] if alpha is None else alpha)


def _diamond(draw_list: Any, cx: float, cy: float, half: float, fill: int, edge: int) -> None:
    """One keyframe glyph: a filled diamond with a contrasting outline."""
    draw_list.add_quad_filled(cx, cy - half, cx + half, cy, cx, cy + half, cx - half, cy, fill)
    draw_list.add_quad(cx, cy - half, cx + half, cy, cx, cy + half, cx - half, cy, edge, 1.2)


def _playhead(draw_list: Any, x: float, y0: float, y1: float, col: int, head: bool) -> None:
    """The vertical playhead, optionally with the triangular grab handle on top."""
    draw_list.add_line(x, y0, x, y1, col, 1.6)
    if head:
        draw_list.add_triangle_filled(x - 6.0, y0, x + 6.0, y0, x, y0 + 9.0, col)


class TimelinePanel(Panel):
    """Transport, scrubber, keyframe tracks and the keyframe inspector."""

    title = "Timeline"
    icon = "timeline"
    default_open = False

    def __init__(self, ws: Any) -> None:
        super().__init__(ws)
        #: Selection is by *object identity*, never by index: ``Track.move`` re-sorts the
        #: keyframes, so an index captured on mouse-down names a different key the moment
        #: a drag crosses its neighbour — which is exactly when it matters.
        self._sel_track: Optional[Track] = None
        self._sel_key: Optional[Keyframe] = None
        self._drag_track: Optional[Track] = None
        self._drag_key: Optional[Keyframe] = None
        #: Snap the playhead and keyframe drags onto whole frames.
        self._snap: bool = True
        #: "New track" authoring fields.
        self._new_target: str = CAMERA_TARGET
        self._new_prop: str = "azim"
        #: Whether the property field is a free-text box rather than the known-property
        #: picker. Sticky, so a custom name typed by hand is not swallowed the moment it
        #: happens to equal a known one.
        self._custom_prop: bool = False
        #: What Capture records. Remembered across captures: a user who is animating layer
        #: opacity presses the button repeatedly and must not have to re-pick each time.
        self._capture_scope: str = "view"
        #: The preset the picker has selected, and whether it builds from the playhead
        #: rather than from zero. Off by default: "quick animation" means the preset fills
        #: the clip; the playhead option is what lets two presets be chained.
        self._preset: str = PRESETS[0].key
        self._preset_from_playhead: bool = False
        self._error: str = ""
        #: Width of the time axis in pixels, measured by the scrubber and reused verbatim
        #: by every track row. Measuring it twice is what misaligns the diamonds from the
        #: playhead the moment the track list grows a scrollbar and its content narrows.
        self._axis_w: float = 1.0

    # -- state -----------------------------------------------------------------

    @property
    def timeline(self) -> Timeline:
        """The active panel's timeline. One per panel, like the cameras."""
        return self.plot.active_panel.timeline

    def _sync_selection(self) -> None:
        """Drop a selection whose track or keyframe has left the timeline."""
        timeline = self.timeline
        if self._sel_track is not None and self._sel_track not in timeline.tracks:
            self._sel_track = None
            self._sel_key = None
        if self._sel_key is not None:
            track = self._sel_track
            if track is None or not any(k is self._sel_key for k in track.keyframes):
                self._sel_key = None
        if self._drag_track is not None and self._drag_track not in timeline.tracks:
            self._drag_track = None
            self._drag_key = None

    # -- queued mutations ------------------------------------------------------
    # Everything below defers (CONTRACT §1.1). The timeline drives the scene through
    # `apply_timeline`, so writing it from `draw` would lose the dirty-flag latch exactly
    # as a direct layer edit would.

    def _seek(self, seconds: float) -> None:
        """Move the playhead, snapping it to a frame when asked, and re-apply."""
        timeline, plot = self.timeline, self.plot
        target = snap_time(timeline, seconds, snap=self._snap)

        def apply() -> None:
            timeline.seek(target)
            apply_timeline(plot, timeline)

        self.submit(apply)

    def _step_frames(self, count: int) -> None:
        timeline, plot = self.timeline, self.plot

        def apply() -> None:
            timeline.step_frames(int(count))
            apply_timeline(plot, timeline)

        self.submit(apply)

    def _toggle_play(self) -> None:
        timeline = self.timeline
        self.submit(timeline.toggle)

    def _stop(self) -> None:
        timeline, plot = self.timeline, self.plot

        def apply() -> None:
            timeline.stop()
            apply_timeline(plot, timeline)

        self.submit(apply)

    def _set_field(self, name: str, value: Any) -> None:
        """Set one scalar timeline field through its own setter where there is one."""
        timeline = self.timeline
        setters = {
            "duration": timeline.set_duration,
            "fps": timeline.set_fps,
            "loop": timeline.set_loop,
        }

        def apply() -> None:
            setter = setters.get(name)
            if setter is not None:
                setter(value)
            else:
                setattr(timeline, name, value)

        self.submit(apply)

    def _fit_duration(self) -> None:
        timeline = self.timeline
        self.submit(lambda: timeline.fit_duration())

    def _set_track_flag(self, track: Track, name: str, value: bool) -> None:
        self.submit(lambda: setattr(track, name, bool(value)))

    def _remove_track(self, track: Track) -> None:
        timeline = self.timeline

        def apply() -> None:
            timeline.remove_track(track.target, track.prop)

        if self._sel_track is track:
            self._sel_track = None
            self._sel_key = None
        self.submit(apply)

    def _add_track(self, target: Any, prop: str) -> None:
        timeline = self.timeline
        prop = str(prop).strip()
        if not prop:
            self._error = "A track needs a property name."
            return
        self._error = ""

        def apply() -> None:
            timeline.track(target, prop)

        self.submit(apply)

    def _move_key(self, track: Track, key: Keyframe, seconds: float) -> None:
        """Retime one keyframe. The index is resolved at drain time, by identity."""
        timeline = self.timeline
        plot = self.plot
        target = snap_time(timeline, seconds, snap=self._snap)

        def apply() -> None:
            index = next((i for i, k in enumerate(track.keyframes) if k is key), None)
            if index is None:  # The key was deleted between the drag and the drain.
                return
            track.move(index, target)
            apply_timeline(plot, timeline)

        self.submit(apply)

    def _key_at_playhead(self, track: Track) -> None:
        """Key the property's current value at the playhead — the core authoring gesture."""
        timeline, plot = self.timeline, self.plot
        value = self._live_value(track)
        easing = self._sel_key.easing if self._sel_key is not None else "smooth"

        def apply() -> None:
            track.add(timeline.time, value, easing)
            timeline.fit_duration()
            apply_timeline(plot, timeline)

        self.submit(apply)

    def _capture(self, scope: str) -> None:
        """Queue a capture of everything ``scope`` covers at the playhead."""
        timeline, plot = self.timeline, self.plot
        at = snap_time(timeline, timeline.time, snap=self._snap)

        def apply() -> None:
            capture_keyframes(plot, timeline, scope=scope, time=at)
            apply_timeline(plot, timeline)

        self.submit(apply)

    def _turntable(self) -> None:
        """Queue a one-turn azimuth sweep across the whole timeline."""
        self._apply_preset("turntable")

    def _apply_preset(self, key: str) -> None:
        """Queue a preset build over the whole timeline, or from the playhead."""
        timeline, plot = self.timeline, self.plot
        start = (
            snap_time(timeline, timeline.time, snap=self._snap)
            if self._preset_from_playhead
            else 0.0
        )

        def apply() -> None:
            apply_preset(plot, timeline, key, start=start)
            apply_timeline(plot, timeline)

        self.submit(apply)

    def _remove_key(self, track: Track, key: Keyframe) -> None:
        def apply() -> None:
            index = next((i for i, k in enumerate(track.keyframes) if k is key), None)
            if index is not None:
                del track.keyframes[index]

        if self._sel_key is key:
            self._sel_key = None
        self.submit(apply)

    def _set_key_field(self, key: Keyframe, name: str, value: Any) -> None:
        timeline, plot = self.timeline, self.plot

        def apply() -> None:
            setattr(key, name, value)
            apply_timeline(plot, timeline)

        self.submit(apply)

    # -- public verbs (the workspace's Animation actions call these) -----------
    # Thin wrappers on the queued mutations above, so a chord, the palette and the button
    # in the transport all run the same code — including the panel's snap setting, which
    # a workspace action reaching past the panel would silently ignore.

    def toggle_play(self) -> None:
        """Start or stop playback."""
        self._toggle_play()

    def stop(self) -> None:
        """Stop and rewind to the start."""
        self._stop()

    def step_frames(self, count: int) -> None:
        """Move the playhead ``count`` frames (negative for back)."""
        self._step_frames(int(count))

    def seek(self, seconds: float) -> None:
        """Move the playhead to ``seconds``."""
        self._seek(float(seconds))

    def key_at_playhead(self) -> int:
        """Key the current value of the selected track, or of every track. Returns how many.

        "Every track" when nothing is selected is the behaviour of a record button: the
        user pressing a global shortcut means "capture this pose", not "capture whichever
        row I last clicked".
        """
        targets = [self._sel_track] if self._sel_track is not None else list(self.timeline.tracks)
        targets = [t for t in targets if t is not None]
        for track in targets:
            self._key_at_playhead(track)
        return len(targets)

    def capture(self, scope: Optional[str] = None) -> int:
        """Capture the figure's current state at the playhead. Returns how many properties.

        Unlike :meth:`key_at_playhead` this **creates the tracks it needs**, which is the
        whole difference: it works on an empty timeline, so it is the first thing a new
        user can press and get an animation out of.

        The count is computed here rather than at drain time because
        :func:`capture_state` is a pure read — that is what lets a shortcut report "Captured
        5 properties" in the same frame the key was pressed.
        """
        key = self._capture_scope if scope is None else str(scope).strip().lower()
        if key not in CAPTURE_SCOPES:
            self._error = f"Unknown capture scope {scope!r}."
            return 0
        self._error = ""
        count = len(capture_state(self.plot, key, timeline=self.timeline))
        if count:
            self._capture(key)
        return count

    def capture_count(self, scope: Optional[str] = None) -> int:
        """How many properties a capture would record right now. A pure read."""
        key = self._capture_scope if scope is None else str(scope).strip().lower()
        if key not in CAPTURE_SCOPES:
            return 0
        try:
            return len(capture_state(self.plot, key, timeline=self.timeline))
        except Exception:  # pragma: no cover - defensive: an exotic scene
            return 0

    def add_turntable(self) -> None:
        """Build a one-turn camera sweep across the whole timeline."""
        self._turntable()

    def build_preset(self, key: str) -> bool:
        """Build the preset called ``key``. False (with an error shown) when it cannot.

        Both refusals are reported rather than silent: an unknown key is a caller bug, and
        "this preset needs a 3D figure" is the answer a user pressing it in a 2D figure
        actually needs.
        """
        preset = PRESET_BY_KEY.get(str(key).strip().lower())
        if preset is None:
            self._error = f"Unknown preset {key!r}."
            return False
        if not preset_available(self.plot, preset):
            self._error = f"{preset.label} {PRESET_REQUIREMENT_LABELS.get(preset.requires, '')}."
            return False
        self._error = ""
        self._preset = preset.key
        self._apply_preset(preset.key)
        return True

    # -- value probing ---------------------------------------------------------

    def _live_value(self, track: Track) -> Any:
        """What ``track``'s property is *right now*, for "key the current value"."""
        return track_live_value(self.plot, track, self.timeline.time)

    # -- frame -----------------------------------------------------------------

    def draw(self) -> None:
        if not IMGUI_AVAILABLE:
            return
        self._sync_selection()
        timeline = self.timeline

        # Playback is driven from here: nothing in the engine advances a Timeline yet, and
        # a queued advance keeps the loop awake (a non-empty queue forces a render).
        queue_playback(self)

        self._draw_transport(timeline)
        imgui.spacing()
        self._draw_capture()
        imgui.spacing()
        self._draw_settings(timeline)
        imgui.spacing()
        self._draw_scrubber(timeline)
        imgui.spacing()
        self._draw_tracks(timeline)
        self._draw_inspector(timeline)
        if timeline.tracks:
            # In the empty state the picker is already up in the call-to-action row; a
            # second copy of it there would be two controls holding one setting.
            self._draw_presets()
        if self._error:
            widgets.error_box(self._error)

    # -- transport -------------------------------------------------------------

    def _draw_transport(self, timeline: Timeline) -> None:
        """The button row every video tool has, in the order every video tool has it."""
        playing = bool(timeline.playing)

        if icon_button("##tl_start", "skip_start", tooltip="Go to the start"):
            self._seek(0.0)
        imgui.same_line()
        if icon_button("##tl_prev", "step_back", tooltip="Previous frame"):
            self._step_frames(-1)
        imgui.same_line()
        if icon_button(
            "##tl_play",
            "pause" if playing else "play",
            tooltip="Pause (Space)" if playing else "Play (Space)",
            active=playing,
        ):
            self._toggle_play()
        imgui.same_line()
        if icon_button("##tl_next", "step_forward", tooltip="Next frame"):
            self._step_frames(1)
        imgui.same_line()
        if icon_button("##tl_end", "skip_end", tooltip="Go to the end"):
            self._seek(timeline.duration)
        imgui.same_line()
        if icon_button("##tl_stop", "stop", tooltip="Stop and rewind"):
            self._stop()

        imgui.same_line()
        imgui.text_disabled("|")
        imgui.same_line()

        loop = str(timeline.loop)
        if icon_button(
            "##tl_loop",
            "loop",
            tooltip=f"Loop mode: {loop} - click to cycle",
            active=loop != "once",
        ):
            self._set_field("loop", LOOP_MODES[(LOOP_MODES.index(loop) + 1) % len(LOOP_MODES)])
        imgui.same_line()
        if icon_button("##tl_fit", "refresh", tooltip="Fit the duration to every keyframe"):
            self._fit_duration()

        imgui.same_line()
        imgui.text_disabled("|")
        imgui.same_line()
        imgui.text_disabled(loop)
        imgui.same_line()
        widgets.help_marker(_HELP_LOOP)

    def _draw_capture(self) -> None:
        """The record button, and it is the widest one here on purpose.

        Everything else in this panel edits an animation that already exists. This is the
        control that *starts* one, and it is the only one a first-time user can press
        without first knowing what a track is — so it gets a labelled button at the top
        rather than an icon in the track header.
        """
        count = self.capture_count()
        if imgui.button("Capture keyframe", 150.0, 0.0):
            self.capture()
        if imgui.is_item_hovered():
            imgui.set_tooltip(_HELP_CAPTURE.format(count=count))
        imgui.same_line()

        labels = [CAPTURE_SCOPE_LABELS[name] for name in CAPTURE_SCOPES]
        imgui.push_item_width(170.0)
        changed, picked = widgets.enum_combo(
            "##tl_capture_scope",
            CAPTURE_SCOPE_LABELS.get(self._capture_scope, self._capture_scope),
            labels,
        )
        imgui.pop_item_width()
        if changed and picked in labels:
            self._capture_scope = CAPTURE_SCOPES[labels.index(picked)]

        imgui.same_line()
        imgui.text_disabled(f"{count} propert{'y' if count == 1 else 'ies'}")
        imgui.same_line()
        widgets.help_marker(_HELP_CAPTURE.format(count=count))

        # The turntable is offered only where it means something. In a 2D figure there is
        # no azimuth to sweep, and a button that keys a property nothing reads would be a
        # worse first experience than no button at all.
        if self._is_3d():
            imgui.same_line()
            if icon_button("##tl_turntable", "orbit", tooltip=_HELP_TURNTABLE):
                self._turntable()

    def _is_3d(self) -> bool:
        """Whether this figure draws in 3D, tolerating an engine without the probe."""
        try:
            return bool(self.plot.is_3d_scene())
        except Exception:  # pragma: no cover - defensive
            return False

    def _draw_settings(self, timeline: Timeline) -> None:
        """Frame rate, duration, speed — the three numbers the transport runs on."""
        options = [str(f) for f in COMMON_FPS]
        current = (
            str(int(timeline.fps)) if float(timeline.fps).is_integer() else f"{timeline.fps:g}"
        )
        imgui.push_item_width(84.0)
        changed, picked = widgets.enum_combo(
            "fps",
            current,
            options,
            help="Frames per second. 60 matches the display; 24 is film; 12 is for stepping "
            "through a discrete process where every frame is a state.",
        )
        imgui.pop_item_width()
        if changed:
            try:
                self._set_field("fps", float(picked))
            except ValueError:  # pragma: no cover - the options are literals
                pass

        imgui.same_line()
        imgui.push_item_width(110.0)
        changed, value = imgui.drag_float(
            "seconds",
            float(timeline.duration),
            0.05,
            float(MIN_DURATION),
            float(MAX_DURATION),
            "%.2f",
        )
        imgui.pop_item_width()
        if changed:
            self._set_field("duration", float(value))

        imgui.same_line()
        imgui.push_item_width(120.0)
        changed, speed = imgui.slider_float("speed", float(timeline.speed), -4.0, 4.0, "%.2fx")
        imgui.pop_item_width()
        if changed:
            self._set_field("speed", float(speed))
        imgui.same_line()
        widgets.help_marker(
            "Playback rate multiplier. Negative plays backwards, which is the quickest way "
            "to check that an animation reads in both directions."
        )

        changed, snap = imgui.checkbox("Snap to frames", bool(self._snap))
        if changed:
            self._snap = bool(snap)
        imgui.same_line()
        imgui.text_disabled(
            f"   t = {timeline.time:.3f}s     frame {timeline.frame_index()} / "
            f"{timeline.frame_count - 1}"
        )

    # -- scrubber --------------------------------------------------------------

    def _time_axis(self) -> Tuple[float, float]:
        """``(x0, width)`` of the time axis in screen pixels, for the *current* cursor row.

        Both the ruler and every track row call this at the same indent, which is what
        makes a keyframe at t and the playhead at t land on the same pixel.
        """
        avail = float(imgui.get_content_region_available().x)
        width = max(80.0, avail - _LABEL_W - _SCROLLBAR_W)
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + _LABEL_W)
        self._axis_w = width
        return float(imgui.get_cursor_screen_pos().x), width

    def _x_of(self, timeline: Timeline, x0: float, width: float, seconds: float) -> float:
        span = max(float(timeline.duration), 1e-9)
        return x0 + float(np.clip(seconds / span, 0.0, 1.0)) * width

    def _t_of(self, timeline: Timeline, x0: float, width: float, x: float) -> float:
        return float(np.clip((x - x0) / max(width, 1e-6), 0.0, 1.0)) * timeline.duration

    def _draw_scrubber(self, timeline: Timeline) -> None:
        """The playhead strip: a ruler, the ticks, and the biggest hit target in the panel."""
        imgui.text("Playhead")
        imgui.same_line()
        widgets.help_marker(_HELP_SCRUB)

        x0, width = self._time_axis()
        y0 = float(imgui.get_cursor_screen_pos().y)
        imgui.invisible_button("##tl_scrub", width, _RULER_H)
        hovered = imgui.is_item_hovered()
        active = imgui.is_item_active()
        y1 = y0 + _RULER_H

        draw_list = imgui.get_window_draw_list()
        draw_list.add_rect_filled(x0, y0, x0 + width, y1, _col("frame_bg"), 3.0)
        draw_list.add_rect(x0, y0, x0 + width, y1, _col("border"), 3.0, 0, 1.0)

        step = _tick_step(timeline.duration, width)
        tick_col = _col("axis", 0.65)
        label_col = _col("muted")
        count = int(timeline.duration / step) + 1 if step > 0 else 0
        for i in range(min(count + 1, 200)):
            seconds = i * step
            if seconds > timeline.duration:
                break
            x = self._x_of(timeline, x0, width, seconds)
            draw_list.add_line(x, y1 - 8.0, x, y1, tick_col, 1.0)
            draw_list.add_text(x + 3.0, y0 + 3.0, label_col, _fmt_time(seconds, step))

        # Elapsed fill: the cheapest possible "how far through am I" cue.
        head_x = self._x_of(timeline, x0, width, timeline.time)
        draw_list.add_rect_filled(x0, y1 - 5.0, head_x, y1 - 1.0, _col("accent", 0.55), 2.0)
        _playhead(draw_list, head_x, y0, y1, _col("accent"), True)

        if hovered:
            imgui.set_tooltip(
                f"{self._t_of(timeline, x0, width, imgui.get_mouse_pos().x):.3f}s "
                f"of {timeline.duration:.3f}s"
            )
        if active:
            self._seek(self._t_of(timeline, x0, width, imgui.get_mouse_pos().x))

    # -- tracks ----------------------------------------------------------------

    def _draw_tracks(self, timeline: Timeline) -> None:
        """The track list: labels and toggles on the left, keyframe diamonds on the right."""
        imgui.text(f"Tracks ({len(timeline.tracks)})")
        imgui.same_line()
        widgets.help_marker(_HELP_TRACKS)
        imgui.same_line()
        if icon_button(
            "##tl_key_add",
            "keyframe",
            tooltip="Key the selected track's current value at the playhead",
            enabled=self._sel_track is not None,
        ):
            if self._sel_track is not None:
                self._key_at_playhead(self._sel_track)
        imgui.same_line()
        if icon_button(
            "##tl_key_del",
            "minus",
            tooltip="Delete the selected keyframe",
            enabled=self._sel_key is not None and self._sel_track is not None,
        ):
            if self._sel_track is not None and self._sel_key is not None:
                self._remove_key(self._sel_track, self._sel_key)

        if not timeline.tracks:
            self._draw_empty_state()
            return

        # Zero window padding so the child's content starts at the same x as the ruler's;
        # any padding here would offset every diamond from the playhead above it.
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0.0, 0.0))
        height = max(_TRACKS_H, min(len(timeline.tracks) * _ROW_H + 6.0, 320.0))
        # end_child is unconditional per CONTRACT §2.6.
        child = imgui.begin_child("##tl_tracks", 0.0, height, border=False)
        if child.visible:
            for index, track in enumerate(list(timeline.tracks)):
                self._draw_track_row(timeline, track, index)
        imgui.end_child()
        imgui.pop_style_var(1)

        imgui.spacing()
        self._draw_new_track()

    def _draw_track_row(self, timeline: Timeline, track: Track, index: int) -> None:
        """One track: toggles, name, delete, then its keyframes on the shared time axis."""
        imgui.push_id(f"tl_track_{index}")
        try:
            selected = track is self._sel_track
            enabled = bool(track.enabled)

            if icon_button(
                "##en",
                "eye" if enabled else "eye_off",
                size=18.0,
                tooltip="Mute this track" if enabled else "Un-mute this track",
            ):
                self._set_track_flag(track, "enabled", not enabled)
            imgui.same_line()
            if icon_button(
                "##interp",
                "smooth" if track.interpolate else "minus",
                size=18.0,
                tooltip=(
                    "Interpolating - values between keys are blended"
                    if track.interpolate
                    else "Holding - each key's value is held until the next"
                ),
                active=not track.interpolate,
            ):
                self._set_track_flag(track, "interpolate", not track.interpolate)
            imgui.same_line()

            imgui.align_text_to_frame_padding()
            clicked, _ = imgui.selectable(
                f"{track_label(track)}##name", selected, 0, _LABEL_W - 96.0, 0.0
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    f"target: {track.target}\nproperty: {track.prop}\n"
                    f"{len(track.keyframes)} keyframe(s)"
                )
            if clicked:
                self._sel_track = track
                self._sel_key = None
            imgui.same_line()
            imgui.text_disabled(f"{len(track.keyframes)}")
            imgui.same_line()
            if icon_button("##del", "trash", size=18.0, tooltip="Delete this track"):
                self._remove_track(track)

            # An explicit offset, not a bare same_line: the left column's widgets are
            # variable-width, and letting the axis start wherever they happened to end
            # would put this row's diamonds a few pixels off every other row's.
            imgui.same_line(_LABEL_W)
            self._draw_track_keys(timeline, track, index)
        finally:
            # pop_id on every path: an unbalanced id stack corrupts every later widget in
            # the frame, not only this row.
            imgui.pop_id()

    def _draw_track_keys(self, timeline: Timeline, track: Track, index: int) -> None:
        """The row's time axis: background, keyframe diamonds, and the playhead through it."""
        width = max(40.0, float(self._axis_w))
        pos = imgui.get_cursor_screen_pos()
        x0, y0 = float(pos.x), float(pos.y)
        imgui.invisible_button("##keys", width, _ROW_H)
        activated = imgui.is_item_activated()
        active = imgui.is_item_active()
        deactivated = imgui.is_item_deactivated()
        y1 = y0 + _ROW_H
        cy = y0 + _ROW_H * 0.5

        draw_list = imgui.get_window_draw_list()
        selected_track = track is self._sel_track
        background = "raised_bg" if selected_track else ("frame_bg" if index % 2 == 0 else "bg")
        draw_list.add_rect_filled(x0, y0, x0 + width, y1 - 1.0, _col(background), 2.0)
        draw_list.add_line(x0, cy, x0 + width, cy, _col("border"), 1.0)

        muted = not track.enabled
        fill = _col("muted" if muted else "accent", 0.55 if muted else 1.0)
        edge = _col("bg", 0.9)
        sel_fill = _col("warn")
        for key in track.keyframes:
            x = self._x_of(timeline, x0, width, key.time)
            if key is self._sel_key and selected_track:
                _diamond(draw_list, x, cy, _KEY_HALF + 2.0, sel_fill, _col("text"))
            else:
                _diamond(draw_list, x, cy, _KEY_HALF, fill, edge)

        head_x = self._x_of(timeline, x0, width, timeline.time)
        _playhead(draw_list, head_x, y0, y1, _col("accent", 0.75), False)

        if activated:
            mouse_x = float(imgui.get_mouse_pos().x)
            hit = self._key_near(timeline, track, x0, width, mouse_x)
            self._sel_track = track
            if hit is not None:
                self._sel_key = hit
                self._drag_track, self._drag_key = track, hit
            else:
                self._sel_key = None
                self._drag_track = self._drag_key = None
                self._seek(self._t_of(timeline, x0, width, mouse_x))
        elif active and self._drag_track is track and self._drag_key is not None:
            self._move_key(
                track, self._drag_key, self._t_of(timeline, x0, width, imgui.get_mouse_pos().x)
            )
        if deactivated and self._drag_track is track:
            self._drag_track = self._drag_key = None

    def _key_near(
        self, timeline: Timeline, track: Track, x0: float, width: float, mouse_x: float
    ) -> Optional[Keyframe]:
        """The keyframe under the cursor, or None. Nearest wins where two overlap."""
        best: Optional[Keyframe] = None
        best_dx = _KEY_HIT
        for key in track.keyframes:
            dx = abs(self._x_of(timeline, x0, width, key.time) - mouse_x)
            if dx <= best_dx:
                best, best_dx = key, dx
        return best

    def _draw_empty_state(self) -> None:
        """What an empty timeline offers: two buttons, not a paragraph.

        The old empty state explained what a track was and pointed at a disclosure holding
        a free-text property field — which is to say it asked the user to already know the
        answer. These two buttons each produce a working animation in one press, and the
        explanation is still there underneath for whoever wants it.
        """
        # One row, not a paragraph and a button stack: this panel's first-run window is a
        # 40%-tall strip and everything above (transport, capture, settings, scrubber)
        # already fills most of it. An empty state that pushes its own call to action below
        # the fold is worse than no empty state at all.
        imgui.text_disabled("Nothing is animated yet.")
        imgui.same_line()
        count = self.capture_count()
        if imgui.button("Capture the current view", 200.0, 0.0):
            self.capture()
        if imgui.is_item_hovered():
            imgui.set_tooltip(_HELP_CAPTURE.format(count=count))
        imgui.same_line()
        imgui.text_disabled("or")
        imgui.same_line()
        self._draw_preset_picker(inline=True)
        imgui.same_line()
        widgets.help_marker(_HELP_EMPTY)

        self._draw_new_track()

    def _selected_preset(self) -> Preset:
        """The preset the picker has selected, falling back to the first."""
        return PRESET_BY_KEY.get(self._preset, PRESETS[0])

    def _draw_preset_picker(self, *, inline: bool) -> None:
        """The combo and the Build button. Shared by the empty state and the section.

        ``inline`` only narrows the combo — the behaviour is identical, because two paths
        that build a preset slightly differently is exactly the kind of divergence that
        makes "it worked in the other place" bug reports.
        """
        options = [preset.label for preset in PRESETS]
        current = self._selected_preset()
        imgui.push_item_width(210.0 if inline else 250.0)
        changed, picked = widgets.enum_combo("##tl_preset", current.label, options)
        imgui.pop_item_width()
        if changed and picked in options:
            self._preset = PRESETS[options.index(picked)].key
            current = self._selected_preset()

        imgui.same_line()
        if preset_available(self.plot, current):
            if imgui.button("Build", 70.0, 0.0):
                self.build_preset(current.key)
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"{current.label}\n\n{current.description}")
        else:
            imgui.align_text_to_frame_padding()
            imgui.text_disabled(PRESET_REQUIREMENT_LABELS.get(current.requires, "unavailable"))

    def _draw_presets(self) -> None:
        """The "Quick animations" section: the picker, where it starts, and what it does."""
        if not widgets.section("Quick animations", default_open=False):
            return
        self._draw_preset_picker(inline=False)
        imgui.same_line()
        widgets.help_marker(_HELP_PRESETS)

        changed, from_playhead = imgui.checkbox(
            "Build from the playhead", bool(self._preset_from_playhead)
        )
        if changed:
            self._preset_from_playhead = bool(from_playhead)
        imgui.same_line()
        widgets.help_marker(
            "Off, a preset fills the whole timeline. On, it starts at the playhead and "
            "runs to the end - which is how two presets are chained: reveal the layers, "
            "move the playhead, then turn the camera."
        )
        imgui.text_wrapped(self._selected_preset().description)

    def _draw_new_track(self) -> None:
        """Authoring: pick a target and a property, get an empty track to key into."""
        if not widgets.section("New track", default_open=False):
            return
        targets = self._target_options()
        labels = [label for label, _ in targets]
        current = next(
            (label for label, value in targets if str(value) == str(self._new_target)),
            labels[0] if labels else CAMERA_TARGET,
        )
        changed, picked = widgets.enum_combo(
            "Target",
            current,
            labels,
            help="What is animated: this panel's 3D camera, its 2D pan and zoom, or one of "
            "the scene's layers.",
        )
        if changed:
            for label, value in targets:
                if label == picked:
                    self._new_target = value
                    # The property list is per target, so a stale name from the previous
                    # one ("azim" on a layer) would create a track nothing can drive.
                    options = _prop_options(value)
                    self._new_prop = options[0] if options else self._new_prop
                    break

        # A picker rather than a text field, because "type the property name" is the step
        # that stopped people: nothing in the UI told them the azimuth is spelled `azim`.
        # The free-text escape hatch stays, one checkbox away, for the properties this
        # whitelist does not know about.
        options = list(_prop_options(self._new_target))
        if self._custom_prop or self._new_prop not in options:
            imgui.push_item_width(160.0)
            _, prop = imgui.input_text("Property", self._new_prop)
            imgui.pop_item_width()
            self._new_prop = prop
        else:
            imgui.push_item_width(160.0)
            picked_prop, prop = widgets.enum_combo("Property", self._new_prop, options)
            imgui.pop_item_width()
            if picked_prop:
                self._new_prop = prop
        imgui.same_line()
        widgets.help_marker(_HELP_NEW_TRACK)

        changed, custom = imgui.checkbox("Type it by hand", bool(self._custom_prop))
        if changed:
            self._custom_prop = bool(custom)
            if not custom and self._new_prop not in options and options:
                self._new_prop = options[0]

        if imgui.button("Add track"):
            self._add_track(self._new_target, self._new_prop)

    def _target_options(self) -> List[Tuple[str, Any]]:
        """``(label, target)`` for both cameras and every layer in the scene."""
        options: List[Tuple[str, Any]] = [
            ("3D camera (pose)", CAMERA_TARGET),
            ("2D camera (pan & zoom)", CAMERA2D_TARGET),
        ]
        try:
            for layer in self.plot.scene.layers:
                layer_id = getattr(layer, "layer_id", None)
                if layer_id is None:
                    continue
                name = str(getattr(layer, "label", "") or getattr(layer, "layer_type", "layer"))
                options.append((f"{name} (#{layer_id})", layer_id))
        except Exception:  # pragma: no cover - defensive: a plot without a scene
            pass
        return options

    # -- inspector -------------------------------------------------------------

    def _draw_inspector(self, timeline: Timeline) -> None:
        """The selected keyframe: when it is, what it holds, and how it leaves."""
        if not widgets.section("Keyframe", default_open=True):
            return
        track, key = self._sel_track, self._sel_key
        if track is None:
            imgui.text_disabled("Select a track to see its keyframes.")
            return
        imgui.text_disabled(f"{track_label(track)}   -   {track.target}.{track.prop}")
        if key is None:
            imgui.text_disabled("Click a diamond to select a keyframe.")
            return

        imgui.push_item_width(140.0)
        changed, seconds = imgui.drag_float(
            "Time", float(key.time), 0.01, 0.0, float(timeline.duration), "%.3f s"
        )
        imgui.pop_item_width()
        if changed:
            self._move_key(track, key, float(seconds))
        imgui.same_line()
        imgui.text_disabled(f"frame {timeline.frame_index(key.time)}")

        if _is_scalar(key.value):
            imgui.push_item_width(140.0)
            changed, value = imgui.drag_float("Value", float(key.value), 0.01, 0.0, 0.0, "%.5g")
            imgui.pop_item_width()
            if changed:
                self._set_key_field(key, "value", float(value))
        else:
            widgets.stat_row("Value", _describe_value(key.value))
            imgui.text_disabled(
                "Read-only: this value is not a single number, so it is edited where it "
                "was recorded, not here."
            )

        names = list(EASING_LABELS.keys())
        labels = [EASING_LABELS[name] for name in names]
        changed, picked = widgets.enum_combo(
            "Easing",
            EASING_LABELS.get(key.easing, key.easing),
            labels,
            help="The curve used to LEAVE this keyframe. 'Step (discrete)' holds the value "
            "for the whole segment - that is the easing for a progression whose "
            "in-between states do not exist.",
        )
        if changed and picked in labels:
            self._set_key_field(key, "easing", names[labels.index(picked)])

        imgui.spacing()
        if imgui.button("Key at playhead"):
            self._key_at_playhead(track)
        imgui.same_line()
        if imgui.button("Delete keyframe"):
            self._remove_key(track, key)
