"""The bridge from an evaluated timeline to a live scene.

:meth:`glplot.core.timeline.Timeline.evaluate` returns ``{(target, prop): value}`` and
knows nothing about layers, cameras or GPU buffers — that separation is what makes the
timeline testable without an engine. This module is the other half: it takes that dict and
a :class:`~glplot.engine.GPULinePlot` and performs the mutations.

**CONTRACT §1.1 — queued commands only.** Every public function here writes to the live
scene, so none of them may be called from an imgui draw callback. The engine clears the
dirty-flag latch later in the same frame (``engine.py:1238``), so a mutation issued from
``draw`` sets flags that are wiped before anything reads them: the change lands in the data
and never reaches the screen, which is the worst of both worlds. Call these from a queued
command (``glplot.gui.commands.CommandQueue``) at the top of the main loop, or from a
script that is not in a draw at all.

Mutation goes through :mod:`glplot.gui.layerops`
------------------------------------------------
Not because it is shorter — it is not — but because the dirty-flag rules are not
guessable from the outside. Two of them bite here:

* :func:`~glplot.gui.layerops.set_layer_style` knows the **3D colour trap**:
  ``renderers/geometry3d.py`` reads ``style.color`` at *GPU-upload* time, not at draw time,
  so setting a 3D layer's colour without ``gpu_dirty`` leaves the old VBO on screen and the
  animation appears to do nothing. Reimplementing "set the attribute" here would
  reintroduce that bug in a place nobody would think to look for it.
* :func:`~glplot.gui.layerops.mark_scene_dirty` is a three-line incantation
  (``frame.dirty_scene``, ``cache.capture_window = None``, ``cache.refresh_requested``) and
  only the middle one kills the stale interaction impostor, which otherwise keeps drawing
  the pre-mutation scene while the user drags.

Geometry additionally sets ``layer.dirty.gpu_dirty`` — a vertex array swap that does not is
invisible, because the buffer on the card is what gets drawn.

Nothing here raises for bad input
---------------------------------
An unknown target, an unknown property, a deleted layer, a malformed value: all skipped,
counted, logged at debug level. This is not defensiveness, it is the only correct
behaviour for the thing being modelled — **a timeline outlives the layers it was keyed
against**. A user deletes a layer at 3pm and presses play at 3.01pm; the tracks that
referenced it are still there (they are also still *useful*, in case the layer comes back
via undo), and playback must not stop. The same argument covers a saved presentation opened
against a different dataset.

The one thing that is not silent is the return value: every entry point returns the number
of properties actually applied, so a caller that wants to notice "my whole animation is
being skipped" can.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from ..gui.layerops import (
    GL_COLORMAP_NAMES,
    _fit_colors,
    _swap_pts,
    find_layer,
    layer_colormap_kind,
    mark_scene_dirty,
    set_layer_colormap,
    set_layer_gl_colormap,
    set_layer_style,
)

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------
# Target names
# ----------------------------------------------------------------------------------

#: The 2D camera (:class:`glplot.core.legacy.CameraState`). ``"camera2d"`` is accepted as a
#: synonym so a script that names both cameras explicitly reads symmetrically.
TARGET_CAMERA: Tuple[str, ...] = ("camera", "camera2d")

#: The 3D camera (:class:`glplot.core.camera3d.Camera3D`).
TARGET_CAMERA3D = "camera3d"

#: The 3D decoration (:class:`glplot.core.camera3d.Axes3DOptions`).
TARGET_AXES3D = "axes3d"

#: The panel's own :class:`~glplot.core.timeline.Timeline`.
TARGET_TIMELINE = "timeline"

#: Engine-wide render options (:class:`glplot.options.EngineOptions`).
TARGET_OPTIONS = "options"

#: Where a progressively-revealed layer's untruncated geometry is parked, so
#: :data:`~glplot.anim.primitives.PROP_DRAW_FRACTION` can grow it back. Under a private-ish
#: name because ``metadata`` is a shared namespace whose other keys (``artist``,
#: ``capabilities``, ``camera``, ``cvalues``) are load-bearing elsewhere.
FULL_GEOMETRY_KEY = "_anim_full_geometry"

#: The same, for a text layer's untruncated string.
FULL_TEXT_KEY = "_anim_full_text"


# ----------------------------------------------------------------------------------
# Coercion — each raises ValueError, and the caller turns that into "skip"
# ----------------------------------------------------------------------------------


def _as_float(value: Any) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"expected a finite number, got {value!r}")
    return out


def _as_optional_float(value: Any) -> Optional[float]:
    """``None`` survives: it is a real state (``Camera3D.distance = None`` = "fit the data")."""
    return None if value is None else _as_float(value)


def _as_positive_float(value: Any) -> float:
    out = _as_float(value)
    if out <= 0.0:
        raise ValueError(f"expected a positive number, got {value!r}")
    return out


def _as_nonneg_float(value: Any) -> float:
    return max(0.0, _as_float(value))


def _as_unit_float(value: Any) -> float:
    """Clamped to ``[0, 1]`` rather than rejected: an easing that overshoots (``back``,
    ``elastic``) legitimately produces an alpha of 1.03 mid-flight, and clamping is what
    every renderer would do with it anyway."""
    return float(np.clip(_as_float(value), 0.0, 1.0))


def _as_bool(value: Any) -> bool:
    return bool(value)


def _as_int(value: Any) -> int:
    return int(round(_as_float(value)))


def _as_rgba(value: Any) -> Tuple[float, float, float, float]:
    parts = [float(c) for c in np.asarray(value, dtype=np.float64).reshape(-1)]
    if len(parts) == 3:
        parts.append(1.0)
    if len(parts) != 4:
        raise ValueError(f"color must have 3 or 4 components, got {len(parts)}")
    return (parts[0], parts[1], parts[2], parts[3])


def _as_vec(value: Any, n: int) -> Tuple[float, ...]:
    parts = [float(c) for c in np.asarray(value, dtype=np.float64).reshape(-1)]
    if len(parts) != n:
        raise ValueError(f"expected {n} components, got {len(parts)}")
    return tuple(parts)


def _as_xy(value: Any) -> Tuple[float, float]:
    x, y = _as_vec(value, 2)
    return (x, y)


def _as_vec3(value: Any) -> Tuple[float, float, float]:
    x, y, z = _as_vec(value, 3)
    return (x, y, z)


def _as_optional_vec3(value: Any) -> Optional[Tuple[float, float, float]]:
    return None if value is None else _as_vec3(value)


def _as_limits(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    lo, hi = _as_vec(value, 2)
    return (lo, hi)


def _as_str(value: Any) -> str:
    return str(value)


# ----------------------------------------------------------------------------------
# Layer properties
# ----------------------------------------------------------------------------------

#: Style fields an animation may write, and how each value is coerced first.
#:
#: An allowlist rather than "anything :class:`LayerStyle` has", for two reasons.
#: :func:`set_layer_style` *raises* on an unknown field, and a raise from inside playback is
#: exactly what this module must not do; and several style fields (``use_colormap``,
#: ``vmin``/``vmax``) have a second, authoritative home in ``metadata`` that a raw write
#: would silently disagree with — those go through
#: :func:`~glplot.gui.layerops.set_layer_colormap` instead.
LAYER_STYLE_PROPS: Dict[str, Callable[[Any], Any]] = {
    "alpha": _as_unit_float,
    "visible": _as_bool,
    "zorder": _as_int,
    "color": _as_rgba,
    "edge_color": _as_rgba,
    "face_color": _as_rgba,
    "line_width": _as_nonneg_float,
    "point_size": _as_nonneg_float,
    "outline_alpha": _as_unit_float,
    "outline_width": _as_nonneg_float,
    "outline_color": _as_rgba,
    "text_size_px": _as_positive_float,
    # Lives on the layer, not on ``layer.style`` — ``set_layer_style`` knows that.
    "translation": _as_xy,
}

#: Property names that mean "replace the vertex array". Aliases on purpose: a verb should
#: not have to know whether its target is a 2D scatter (``pts``) or a 3D mesh
#: (``vertices``), so both names resolve to whichever attribute the layer actually carries.
GEOMETRY_PROPS: Tuple[str, ...] = ("pts", "vertices")


def _geometry_attr(layer: Any) -> Optional[str]:
    """``"pts"`` or ``"vertices"`` — whichever this layer keeps its geometry in.

    No layer class has both (scatter/polyline have ``pts``; patch and every 3D layer have
    ``vertices``), so the choice is unambiguous.
    """
    for attr in ("pts", "vertices"):
        if hasattr(layer, attr):
            return attr
    return None


def _fit_1d(values: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    """Resize a per-vertex 1-D array (``sizes``) to ``n``, nearest-neighbour.

    Same reasoning as :func:`glplot.gui.layerops._fit_colors`: the renderers size the draw
    call from the vertex count but upload these buffers at whatever length they happen to
    be, so a short one is read past its end.
    """
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32).ravel()
    if len(arr) == n or len(arr) == 0:
        return arr if len(arr) == n else None
    idx = np.clip(np.rint(np.linspace(0, len(arr) - 1, n)).astype(np.intp), 0, len(arr) - 1)
    return np.ascontiguousarray(arr[idx], dtype=np.float32)


def _set_geometry(plot: Any, layer: Any, value: Any) -> bool:
    """Swap a layer's vertex array. The geometry-morph path.

    Refuses (returns False, changes nothing) when:

    * the value is not an ``(N, D)`` float array;
    * ``D`` disagrees with the layer's current geometry — 2D and 3D vertices are not
      interchangeable and padding a z of zero would read as the shape collapsing;
    * the vertex count changed **and** the layer carries an index buffer. A triangle list
      or a wireframe's ``indices`` name specific vertices; keeping them across a resize
      makes the draw call read past the end of the VBO, which is a GPU fault rather than a
      wrong picture. :class:`~glplot.anim.primitives.Transform`'s resampling exists partly
      so this case does not come up: matched counts morph fine.

    ``gpu_dirty`` is mandatory here, not cosmetic — see the module docstring.
    """
    attr = _geometry_attr(layer)
    if attr is None:
        return False

    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    if arr.ndim != 2 or arr.shape[1] not in (2, 3) or len(arr) == 0:
        raise ValueError(f"geometry must be a non-empty (N, 2) or (N, 3) array, got {arr.shape}")

    current = getattr(layer, attr, None)
    if isinstance(current, np.ndarray) and current.ndim == 2 and current.shape[1] != arr.shape[1]:
        raise ValueError(
            f"cannot replace {current.shape[1]}-D geometry with {arr.shape[1]}-D geometry"
        )
    count_changed = not (isinstance(current, np.ndarray) and len(current) == len(arr))
    if count_changed and getattr(layer, "indices", None) is not None:
        raise ValueError("cannot change the vertex count of an indexed layer")

    if attr == "pts":
        # Not a bare assignment: ``_swap_pts`` also re-fits ``colors`` and re-points the
        # legacy ``scene.scatters``/``strips`` mirror at the new array. An orphaned mirror
        # both pins the old array against GC and survives ``remove_layer``'s identity purge,
        # so skipping it leaks a full copy of the geometry per morph.
        _swap_pts(plot, layer, arr)
    else:
        layer.vertices = arr
        if getattr(layer, "colors", None) is not None:
            layer.colors = _fit_colors(layer.colors, len(arr))
        if getattr(layer, "sizes", None) is not None:
            layer.sizes = _fit_1d(layer.sizes, len(arr))

    layer.dirty.gpu_dirty = True
    layer.dirty.bounds_dirty = True
    return True


def _set_draw_fraction(plot: Any, layer: Any, value: Any) -> bool:
    """Show only the first ``fraction`` of a layer's geometry — the ``Create`` mechanism.

    The full array is parked in ``metadata[FULL_GEOMETRY_KEY]`` the first time this runs and
    is what every later fraction slices, so the reveal is idempotent and reversible; at
    fraction 1 the original array object is restored, not a copy of it.

    Rejected: keying the vertex array itself with progressively longer prefixes. It works,
    and it stores the whole geometry once per keyframe — for the 10⁵-vertex layers this
    engine exists for, a two-second reveal at 60fps would be gigabytes of keyframes.

    Limitations, both real:

    * An **indexed** layer is refused. Truncating the vertices leaves the index buffer
      naming vertices that no longer exist.
    * A layer that is *also* being morphed on the same span will fight this: the parked
      "full" array is the geometry as it stood when the reveal began, and the morph writes
      over what the reveal produced. Do one or the other on a given span.
    """
    fraction = float(np.clip(_as_float(value), 0.0, 1.0))
    attr = _geometry_attr(layer)
    if attr is None:
        return False
    if getattr(layer, "indices", None) is not None:
        raise ValueError("cannot progressively reveal an indexed layer")

    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        return False

    full = metadata.get(FULL_GEOMETRY_KEY)
    if full is None:
        full = getattr(layer, attr, None)
        if not isinstance(full, np.ndarray) or full.ndim != 2 or len(full) == 0:
            return False
        metadata[FULL_GEOMETRY_KEY] = full

    if fraction >= 1.0:
        wanted = full
    else:
        # A polyline needs two vertices before it draws anything at all, so the floor is 2
        # rather than 0: at fraction 0 the layer shows a degenerate stub instead of
        # vanishing, which is what "not drawn yet" looks like in Manim too. Callers that
        # want it gone should pair the reveal with ``Hide``/``FadeIn``.
        count = int(np.clip(round(fraction * len(full)), min(2, len(full)), len(full)))
        wanted = np.ascontiguousarray(full[:count])

    current = getattr(layer, attr, None)
    if isinstance(current, np.ndarray) and len(current) == len(wanted):
        # Handled, but the reveal has not advanced a whole vertex since the last frame.
        # Re-uploading an identical buffer is the single most expensive no-op available.
        return True

    if attr == "pts":
        _swap_pts(plot, layer, wanted)
    else:
        layer.vertices = wanted
        if getattr(layer, "colors", None) is not None:
            layer.colors = _fit_colors(layer.colors, len(wanted))
        if getattr(layer, "sizes", None) is not None:
            layer.sizes = _fit_1d(layer.sizes, len(wanted))

    layer.dirty.gpu_dirty = True
    layer.dirty.bounds_dirty = True
    return True


def _set_text_fraction(plot: Any, layer: Any, value: Any) -> bool:
    """Show only the first ``fraction`` of a text layer's string — the ``Write`` mechanism.

    Character-by-character. The engine draws text through imgui's font atlas and has no
    glyph outlines to trace, so Manim's stroke-by-stroke reveal is not reachable; a
    typewriter is the honest approximation.
    """
    if not hasattr(layer, "text"):
        return False
    fraction = float(np.clip(_as_float(value), 0.0, 1.0))

    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    full = metadata.get(FULL_TEXT_KEY)
    if full is None:
        full = str(getattr(layer, "text", "") or "")
        if not full:
            return False
        metadata[FULL_TEXT_KEY] = full

    wanted = full if fraction >= 1.0 else full[: int(round(fraction * len(full)))]
    if str(getattr(layer, "text", "")) != wanted:
        layer.text = wanted
    return True


def _set_cmap(plot: Any, layer: Any, value: Any) -> bool:
    """Switch a layer's colormap, through whichever of the two mechanisms it has.

    :func:`~glplot.gui.layerops.layer_colormap_kind` is the dispatcher, and both setters it
    points at raise for a layer that has neither — which is a perfectly ordinary thing for
    an animation to be pointed at, so the "no colormap here" case returns False instead.

    ``"gl_line"`` layers (polyline, line_family) take a *shader* colormap indexed into
    :data:`~glplot.gui.layerops.GL_COLORMAP_NAMES`, not a matplotlib name; both an index and
    a name are accepted and a name that is not in that list is refused rather than silently
    rendering as Classic.
    """
    kind = layer_colormap_kind(layer)
    if kind is None:
        return False

    if kind == "gl_line":
        if isinstance(value, str):
            names = [n.lower() for n in GL_COLORMAP_NAMES]
            if value.strip().lower() not in names:
                raise ValueError(
                    f"{value!r} is not a shader colormap; expected one of "
                    f"{', '.join(GL_COLORMAP_NAMES)}"
                )
            index = names.index(value.strip().lower())
        else:
            index = _as_int(value)
        set_layer_gl_colormap(plot, layer, enabled=True, scheme_index=index)
        return True

    set_layer_colormap(plot, layer, cmap=_as_str(value))
    return True


#: Layer properties whose application may move the geometry, and therefore the scene's
#: bounds. A 3D one of these forces a re-frame; see :func:`apply_values`.
_GEOMETRY_MOVING_PROPS: Tuple[str, ...] = GEOMETRY_PROPS + ("draw_fraction",)


def _apply_layer_prop(plot: Any, layer: Any, prop: str, value: Any) -> bool:
    """Apply one property to one layer. Raises on a bad value; the caller turns that into a skip."""
    if prop in LAYER_STYLE_PROPS:
        set_layer_style(plot, layer, **{prop: LAYER_STYLE_PROPS[prop](value)})
        return True
    if prop in GEOMETRY_PROPS:
        return _set_geometry(plot, layer, value)
    if prop == "draw_fraction":
        return _set_draw_fraction(plot, layer, value)
    if prop == "text_fraction":
        return _set_text_fraction(plot, layer, value)
    if prop == "cmap":
        return _set_cmap(plot, layer, value)
    return False


# ----------------------------------------------------------------------------------
# Camera / axes / timeline / options properties
# ----------------------------------------------------------------------------------

#: :class:`~glplot.core.camera3d.Camera3D` fields an animation may write.
#:
#: ``projection`` and ``up_axis`` are handled separately: they are strings validated by
#: setter methods, not values to coerce.
CAMERA3D_PROPS: Dict[str, Callable[[Any], Any]] = {
    "elev": _as_float,
    "azim": _as_float,
    "roll": _as_float,
    "distance": _as_optional_float,
    "fov": _as_positive_float,
    "pan": _as_vec3,
    "box_aspect": _as_optional_vec3,
    "auto_spin": _as_float,
}

#: :class:`~glplot.core.legacy.CameraState` fields. ``zoom`` is the legacy property that
#: scales both axes together, and it is kept because it is what a 2D "zoom to" means.
CAMERA2D_PROPS: Tuple[str, ...] = ("cx", "cy", "zoom_x", "zoom_y", "zoom")

#: :class:`~glplot.core.camera3d.Axes3DOptions` fields, with their coercions. Decoration,
#: so animating them is how a presentation reveals the grid or the tick labels on cue.
AXES3D_PROPS: Dict[str, Callable[[Any], Any]] = {
    "show_axes": _as_bool,
    "show_box": _as_bool,
    "show_floor": _as_bool,
    "show_grid": _as_bool,
    "show_ticks": _as_bool,
    "show_tick_labels": _as_bool,
    "show_axis_labels": _as_bool,
    "tick_count": _as_int,
    "pad": _as_nonneg_float,
    "xlabel": _as_str,
    "ylabel": _as_str,
    "zlabel": _as_str,
    "xlim": _as_limits,
    "ylim": _as_limits,
    "zlim": _as_limits,
}

#: Timeline fields an animation may write. Deliberately **not** ``time`` or ``duration``:
#: a track that moves the playhead is evaluated *from* the playhead, so it would feed back
#: on itself and the animation would either freeze or run away depending on the sign.
#: ``speed`` is safe (it only scales the next :meth:`Timeline.advance`) and is the one that
#: is actually wanted, for a slow-motion beat inside a presentation.
TIMELINE_PROPS: Dict[str, Callable[[Any], Any]] = {
    "speed": _as_float,
    "fps": _as_positive_float,
}


def _apply_camera3d(plot: Any, camera: Any, prop: str, value: Any) -> bool:
    if prop in CAMERA3D_PROPS:
        setattr(camera, prop, CAMERA3D_PROPS[prop](value))
        return True
    if prop == "projection":
        camera.set_projection(_as_str(value))  # raises on an unknown mode -> skipped
        return True
    if prop == "up_axis":
        camera.set_up_axis(_as_str(value))
        return True
    return False


def _apply_camera2d(plot: Any, camera: Any, prop: str, value: Any) -> bool:
    if prop not in CAMERA2D_PROPS:
        return False
    number = _as_float(value)
    if prop in ("cx", "cy"):
        setattr(camera, prop, number)
        return True
    # Zoom is clamped to the camera's own limits rather than merely to "positive": the
    # projection divides by it, and an overshooting easing (``back``, ``elastic``) really
    # can drive a zoom keyframe through zero on its way to the target.
    lo = float(getattr(camera, "zoom_min", 1e-12))
    hi = float(getattr(camera, "zoom_max", 1e12))
    setattr(camera, prop, float(np.clip(number, lo, hi)))
    return True


def _apply_axes3d(plot: Any, axes: Any, prop: str, value: Any) -> bool:
    if prop not in AXES3D_PROPS:
        return False
    setattr(axes, prop, AXES3D_PROPS[prop](value))
    return True


def _apply_timeline(plot: Any, timeline: Any, prop: str, value: Any) -> bool:
    if prop not in TIMELINE_PROPS:
        return False
    setattr(timeline, prop, TIMELINE_PROPS[prop](value))
    return True


def _apply_options(plot: Any, prop: str, value: Any) -> bool:
    """Write an engine option, coercing to the type the option already holds.

    ``global_alpha`` is special-cased because it lives on the *engine*
    (``plot.global_alpha``, seeded from ``options.default_global_alpha``) and it is the one
    option an animation is genuinely likely to want — a whole-scene fade.

    Everything else is type-directed rather than allow-listed: :class:`EngineOptions` is a
    flat settings record with fifty-odd fields and no side effects on assignment, so an
    allowlist here would be a second list to keep in step with it for no safety gained. The
    existence check (``hasattr``) is what stops a typo from inventing a field, and the
    coercion is what stops a float landing in an int option.
    """
    if prop == "global_alpha":
        plot.global_alpha = _as_unit_float(value)
        return True

    options = getattr(plot, "options", None)
    if options is None or not hasattr(options, prop):
        return False
    current = getattr(options, prop)
    if isinstance(current, bool):
        setattr(options, prop, _as_bool(value))
    elif isinstance(current, (int, np.integer)):
        setattr(options, prop, _as_int(value))
    elif isinstance(current, (float, np.floating)):
        setattr(options, prop, _as_float(value))
    elif isinstance(current, str):
        setattr(options, prop, _as_str(value))
    elif isinstance(current, tuple):
        setattr(options, prop, tuple(float(v) for v in np.asarray(value).reshape(-1)))
    else:
        return False
    return True


# ----------------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------------


def _resolve_panel(plot: Any, panel: Any) -> Any:
    if panel is not None:
        return panel
    return getattr(plot, "active_panel", None)


def _resolve_layer(plot: Any, panel: Any, layer_id: int) -> Any:
    """The layer with ``layer_id`` in ``panel``, or None.

    Routed through :func:`~glplot.gui.layerops.find_layer` for the active panel (which is
    what ``plot.scene`` delegates to) and by a direct scan otherwise, so an animation on a
    background panel of a split figure does not silently address the foreground one.
    """
    if panel is None or panel is getattr(plot, "active_panel", None):
        return find_layer(plot, layer_id)
    for layer in getattr(getattr(panel, "scene", None), "layers", ()) or ():
        if getattr(layer, "layer_id", None) == layer_id:
            return layer
    return None


def _mark_dirty(plot: Any, panel: Any) -> None:
    """:func:`~glplot.gui.layerops.mark_scene_dirty`, extended to a non-active panel.

    ``mark_scene_dirty`` reaches ``plot.cache``, which is a property delegating to the
    *active* panel — correct for every existing caller (they all run against the panel the
    user is interacting with) and not enough here, because a timeline is per panel and a
    split figure animates all of them at once.
    """
    mark_scene_dirty(plot)
    active = getattr(plot, "active_panel", None)
    if panel is not None and panel is not active:
        cache = getattr(panel, "cache", None)
        if cache is not None:
            cache.capture_window = None
            cache.refresh_requested = True


def apply_values(
    plot: Any,
    values: Mapping[Tuple[Any, str], Any],
    *,
    panel: Any = None,
) -> int:
    """Apply an evaluated timeline to the scene. Returns how many properties landed.

    **Queued commands only** (CONTRACT §1.1).

    Parameters
    ----------
    plot
        The :class:`~glplot.engine.GPULinePlot`.
    values
        ``{(target, prop): value}`` — exactly what
        :meth:`glplot.core.timeline.Timeline.evaluate` returns.
    panel
        Which panel the targets are resolved against. Defaults to the active one, which is
        what a single-panel figure wants and what every non-panel-aware caller means.

    Target resolution
    -----------------
    ==================  ========================================================
    ``target``          resolves to
    ==================  ========================================================
    ``int``             the layer with that ``layer_id`` (missing → skipped)
    ``"camera"``        the panel's 2D :class:`CameraState`
    ``"camera2d"``      the same
    ``"camera3d"``      the panel's :class:`Camera3D`
    ``"axes3d"``        the panel's :class:`Axes3DOptions`
    ``"timeline"``      the panel's :class:`Timeline`
    ``"options"``       :class:`EngineOptions` (plus ``global_alpha`` on the engine)
    anything else       skipped
    ==================  ========================================================

    The count that comes back is the number of ``(target, prop)`` pairs this module
    *handled* — resolved to something real and wrote (or confirmed) — not the number of
    pixels that changed. A held ``draw_fraction`` that has not advanced a whole vertex since
    the last frame still counts, because the property is correctly applied; it just needed
    no GPU work.

    Everything that cannot be applied is skipped and logged at debug level; nothing raises.
    See the module docstring for why that is a requirement rather than a convenience.
    """
    if not values:
        return 0

    panel = _resolve_panel(plot, panel)
    applied = 0
    touched_camera3d = False
    touched_3d_geometry = False

    for key, value in values.items():
        try:
            target, prop = key
        except (TypeError, ValueError):
            logger.debug("anim.applier: malformed timeline key %r", key)
            continue
        prop = str(prop)

        try:
            if isinstance(target, (int, np.integer)) and not isinstance(target, bool):
                layer = _resolve_layer(plot, panel, int(target))
                if layer is None:
                    # The commonest skip by far, and the one the whole design is built
                    # around: the layer was deleted and the track outlived it.
                    logger.debug("anim.applier: no layer %s for %r", target, prop)
                    continue
                changed = _apply_layer_prop(plot, layer, prop, value)
                if (
                    changed
                    and prop in _GEOMETRY_MOVING_PROPS
                    and str(getattr(layer, "layer_type", "")).endswith("3d")
                ):
                    touched_3d_geometry = True

            elif isinstance(target, str) and target in TARGET_CAMERA:
                changed = _apply_camera2d(plot, panel.camera, prop, value)

            elif target == TARGET_CAMERA3D:
                changed = _apply_camera3d(plot, panel.camera3d, prop, value)
                touched_camera3d = touched_camera3d or changed

            elif target == TARGET_AXES3D:
                changed = _apply_axes3d(plot, panel.axes3d, prop, value)
                touched_camera3d = touched_camera3d or changed

            elif target == TARGET_TIMELINE:
                changed = _apply_timeline(plot, panel.timeline, prop, value)

            elif target == TARGET_OPTIONS:
                changed = _apply_options(plot, prop, value)

            else:
                logger.debug("anim.applier: unknown target %r", target)
                continue
        except Exception:
            # Every failure mode is the same failure mode from the caller's point of view:
            # this frame does not get that property. A raise would stop playback, and a
            # timeline is exactly the kind of state that goes stale on purpose.
            logger.debug("anim.applier: could not apply %r to %r", prop, target, exc_info=True)
            continue

        if changed:
            applied += 1
        else:
            logger.debug("anim.applier: %r is not animatable on %r", prop, target)

    if applied:
        _mark_dirty(plot, panel)

    # Batched, not per property: both of these walk every layer in the scene, and a
    # thirty-track camera move would otherwise pay for them thirty times a frame.
    active = getattr(plot, "active_panel", None)
    if panel is active:
        try:
            if touched_3d_geometry:
                # Bounds moved, so the shared axis box and the auto-fit distance have to
                # follow — same call ``layerops3d.update_layer_xyz`` makes, and it re-syncs
                # the camera on the way through.
                plot.set_3d_view()
            elif touched_camera3d:
                # The renderer builds its matrix per layer out of ``metadata["camera"]``,
                # so a camera write that skips this changes nothing on screen.
                plot.sync_3d_camera()
        except Exception:  # pragma: no cover - engine without a 3D pipeline
            logger.debug("anim.applier: 3D re-sync failed", exc_info=True)

    return applied


def apply_timeline(
    plot: Any,
    *,
    panel: Any = None,
    timeline: Any = None,
    time: Optional[float] = None,
) -> int:
    """Evaluate a timeline and apply it. The one-liner a playback loop calls.

    **Queued commands only** (CONTRACT §1.1).

    ``timeline`` defaults to the panel's own; ``time`` to that timeline's playhead. Returns
    the number of properties applied, so a caller can tell "nothing is keyed" from
    "everything I keyed is stale".
    """
    panel = _resolve_panel(plot, panel)
    if timeline is None:
        timeline = getattr(panel, "timeline", None)
    if timeline is None:
        return 0
    return apply_values(plot, timeline.evaluate(time), panel=panel)


def reset_reveal(layer: Any) -> bool:
    """Undo a progressive reveal, restoring the layer's full geometry and text.

    The parked arrays are the only state this module leaves behind, and they are not
    self-clearing: a layer left mid-``Create`` when the timeline is deleted would keep its
    truncated geometry forever. Returns True if anything was restored.

    Does **not** set dirty flags — the caller knows whether it is inside a queued command
    and whether anything else changed too. Pair it with
    :func:`~glplot.gui.layerops.mark_scene_dirty`.
    """
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        return False

    restored = False
    full = metadata.pop(FULL_GEOMETRY_KEY, None)
    if full is not None:
        attr = _geometry_attr(layer)
        if attr is not None:
            setattr(layer, attr, full)
            if getattr(layer, "colors", None) is not None:
                layer.colors = _fit_colors(layer.colors, len(full))
            if getattr(layer, "sizes", None) is not None:
                layer.sizes = _fit_1d(layer.sizes, len(full))
            layer.dirty.gpu_dirty = True
            layer.dirty.bounds_dirty = True
            restored = True

    text = metadata.pop(FULL_TEXT_KEY, None)
    if text is not None and hasattr(layer, "text"):
        layer.text = text
        restored = True

    return restored


__all__ = [
    "AXES3D_PROPS",
    "CAMERA2D_PROPS",
    "CAMERA3D_PROPS",
    "FULL_GEOMETRY_KEY",
    "FULL_TEXT_KEY",
    "GEOMETRY_PROPS",
    "LAYER_STYLE_PROPS",
    "TARGET_AXES3D",
    "TARGET_CAMERA",
    "TARGET_CAMERA3D",
    "TARGET_OPTIONS",
    "TARGET_TIMELINE",
    "TIMELINE_PROPS",
    "apply_timeline",
    "apply_values",
    "reset_reveal",
]
