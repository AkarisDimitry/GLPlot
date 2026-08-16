"""Manim-like animation verbs that write keyframes into a :class:`~glplot.core.timeline.Timeline`.

Why verbs at all
----------------
Hand-authored keyframes are the assembly language of animation: everything is expressible
and nothing is short. Manim's whole leverage is that an animation is a **verb applied to
an object over a span** — ``self.play(FadeIn(dot), run_time=1)`` — so a five-line script
produces a minute of video. This module is that vocabulary for GLPlot.

The one rule every verb obeys
-----------------------------
**A verb writes keyframes. It never drives the clock and never touches the scene.**

That is the difference between this and a callback-per-frame animation library, and it
buys three things that matter more than they sound:

* **Scrubbable.** The state at t = 3.2s is computed, not accumulated, so a timeline can be
  dragged backwards, seeked into the middle, or rendered out of order.
* **Deterministic export.** ``Timeline.bake()`` walks the frame grid and gets the same
  numbers every run, at any frame rate, on any machine.
* **Composable.** Two verbs on the same property are two sets of keyframes on one track.
  Nothing has to arbitrate between competing callbacks because there are no callbacks.

Pure numpy + stdlib, exactly like :mod:`glplot.core.timeline`: no GL, no imgui, no engine
import. :mod:`glplot.anim.applier` is the only module that knows what a layer is.

Verbs are records, not closures
-------------------------------
Every verb is a frozen dataclass. A closure over engine state would have been shorter to
write and is what most animation APIs do, but a closure cannot be compared, cannot be
printed usefully in a timeline panel, cannot be serialised into a saved presentation, and
cannot be tested without an engine. A record can. The cost is that a verb must be *told*
the values it needs (``Transform`` is handed both vertex arrays; ``ColorTo`` is handed the
colour to start from) rather than reading them off a live layer — and where that is
tedious there is a ``from_layer`` / ``from_camera`` constructor that does the reading once,
at authoring time, and stores plain numbers.

Start values, and the :data:`INHERIT` sentinel
----------------------------------------------
A two-keyframe animation needs to know where it starts. Three answers, in the order you
should reach for them:

1. **An explicit start** — ``FadeIn(start_alpha=0.0)``. The default for every verb where a
   sensible neutral value exists (alpha starts at 0 for a fade-in, translation at the
   origin), because it makes the verb self-contained and reproducible.
2. **:data:`INHERIT`** — take the start value from whatever the track already holds at the
   animation's start time. This is what makes verbs chain: ``succession`` of a
   ``MoveTo`` then a ``Shift`` needs the second to begin where the first ended, and
   ``INHERIT`` is how it finds out without the author restating the number.
3. **A snapshot constructor** — ``CameraMoveTo.from_camera(fig.camera3d, azim=90)`` reads
   the live values *now* and freezes them into the record.

``INHERIT`` against an *empty* track has no answer, so the verb writes only its
destination keyframe. :meth:`~glplot.core.timeline.Track.value_at` holds outside the keyed
range, so the property then simply snaps to its destination and stays there. That is a
visible, understandable failure rather than a silent wrong number, but it *is* a failure —
if a verb appears to do nothing but jump, an unresolvable ``INHERIT`` is the first thing to
check.

Baked curves
------------
Some verbs cannot be expressed as two keyframes because the property does not travel in a
straight line between them. Rotating a shape 180° with two vertex keyframes interpolates
*positions*, so the shape collapses through the pivot and comes out the other side inverted
— it does not rotate. Those verbs **bake**: they emit many keyframes along the true path,
each holding the value at an already-eased fraction, and mark every one ``linear`` so the
track does not ease a second time between them. :data:`ROTATE_DEGREES_PER_KEY` sets how
finely.

Rejected: giving :class:`~glplot.core.timeline.Track` a "rotation" interpolation mode. It
would have made the timeline know about pivots and axes, which is exactly the coupling that
module's docstring exists to prevent, and it would still not have covered the next
non-linear path anyone wants.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..core.timeline import STEP, Timeline, Track, ease

# ----------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------

#: How long a verb runs when nobody says. One second is Manim's ``run_time`` default and is
#: long enough to read at 60fps without feeling slow.
DEFAULT_DURATION = 1.0

#: The easing every verb starts with. ``smooth`` (zero velocity at both ends) rather than
#: ``linear``, for the same reason Manim's default rate function is ``smooth``: a linear
#: move starts and stops with a visible jolt.
DEFAULT_EASING = "smooth"

#: Manim's ``LaggedStart`` default, and the stagger that reads as "one after another, but
#: overlapping" rather than as a queue.
DEFAULT_LAG_RATIO = 0.25

#: Degrees of rotation per baked keyframe. At 15° the chord's sagitta is
#: ``1 - cos(7.5°) ≈ 0.86%`` of the radius, i.e. under a pixel for a 100px-radius shape —
#: below that the extra keyframes cost memory and buy nothing visible.
ROTATE_DEGREES_PER_KEY = 15.0

#: Keyframes an :class:`Orbit` writes per full turn. An orbit keys an *angle*, which does
#: travel in a straight line, so two keyframes would be exact — these exist only so that a
#: track scrubbed in a GUI shows the sweep as a sequence of keys rather than as one long
#: segment, and so an easing curve is visible in the key spacing.
ORBIT_KEYS_PER_TURN = 8

# -- property names ----------------------------------------------------------------
#
# Spelled once here and used by both this module and ``glplot.anim.applier``, so a rename
# is one edit rather than a silent mismatch between what a verb writes and what the applier
# looks for.

#: Layer opacity, 0..1. :class:`~glplot.core.layers.LayerStyle.alpha`.
PROP_ALPHA = "alpha"
#: Layer visibility. Boolean, therefore never interpolated.
PROP_VISIBLE = "visible"
#: Primary RGBA. :class:`~glplot.core.layers.LayerStyle.color`.
PROP_COLOR = "color"
#: ``(dx, dy)`` layer-space offset. :attr:`~glplot.core.layers.BaseLayer.translation`.
PROP_TRANSLATION = "translation"
#: Fraction of the layer's geometry that is drawn, 0..1 — what :class:`Create` keys.
PROP_DRAW_FRACTION = "draw_fraction"
#: Fraction of a text layer's string that is shown, 0..1 — what :class:`Write` keys.
PROP_TEXT_FRACTION = "text_fraction"
#: The vertex array itself. ``"pts"`` and ``"vertices"`` are aliases as far as the applier
#: is concerned: it resolves whichever attribute the layer actually carries, so a verb does
#: not have to know whether its target is a 2D scatter or a 3D mesh.
PROP_VERTICES = "pts"
#: Colormap *name* (a string) — held, never blended.
PROP_CMAP = "cmap"

#: Every :class:`~glplot.core.camera3d.Camera3D` field a camera verb may key. Numeric and
#: sequence-valued, so all of them interpolate; ``projection`` and ``up_axis`` are strings
#: and are deliberately absent (animate them with :class:`Set`, which holds).
CAMERA3D_PROPS: Tuple[str, ...] = (
    "elev",
    "azim",
    "roll",
    "distance",
    "fov",
    "pan",
    "box_aspect",
)

#: The 2D camera's animatable fields (:class:`~glplot.core.legacy.CameraState`).
CAMERA2D_PROPS: Tuple[str, ...] = ("cx", "cy", "zoom_x", "zoom_y")


# ----------------------------------------------------------------------------------
# Sentinels and relative values
# ----------------------------------------------------------------------------------


class _Sentinel:
    """A named singleton. Two are needed and ``None`` can be neither of them.

    ``None`` is a *real value* in this domain — ``Camera3D.distance = None`` means "fit the
    data" and ``LayerStyle.color = None`` means "use the renderer's fallback" — so using it
    to mean "unspecified" would make those two states unreachable.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return self._name

    def __bool__(self) -> bool:
        return False


#: "Take this value from whatever the track already holds at the start time." See the
#: module docstring for what happens when the track is empty.
INHERIT = _Sentinel("INHERIT")

#: "This property is not part of this verb." Used by the multi-property camera verbs, where
#: every field is optional and ``None`` is a meaningful value for at least one of them.
UNSET = _Sentinel("UNSET")


@dataclass(frozen=True)
class Delta:
    """A value expressed *relative* to wherever the property already is.

    ``Shift`` and ``Orbit`` are relative by definition — "move 3 to the right", "turn twice
    more" — and they must stay relative all the way to :meth:`Animation.apply`, because the
    value they are relative *to* is only known once the track is in hand.

    ``default`` is the base to use when there is nothing to be relative to (an empty track
    and no explicit start). It is the property's neutral value: ``(0, 0)`` for a
    translation, ``0`` for an angle.
    """

    amount: Any
    default: Any = 0.0

    def resolve(self, base: Any) -> Any:
        """``base + amount``, elementwise for sequences and arrays."""
        if base is None or isinstance(base, _Sentinel):
            base = self.default
        amount = self.amount
        if isinstance(base, np.ndarray) or isinstance(amount, np.ndarray):
            return np.asarray(base, dtype=np.float64) + np.asarray(amount, dtype=np.float64)
        if isinstance(base, (list, tuple)):
            if isinstance(amount, (list, tuple, np.ndarray)):
                parts = [float(b) + float(a) for b, a in zip(base, np.asarray(amount).ravel())]
            else:
                parts = [float(b) + float(amount) for b in base]
            return tuple(parts) if isinstance(base, tuple) else parts
        return float(base) + float(amount)


# ----------------------------------------------------------------------------------
# Geometry helpers — public, because a caller building a custom verb needs them too
# ----------------------------------------------------------------------------------


def resample_vertices(vertices: Any, count: int) -> np.ndarray:
    """Resample an ``(N, D)`` vertex array to ``count`` rows by normalised index.

    Each output row ``i`` is the linear blend of the input rows around
    ``i / (count - 1)`` of the way through the array. Not arc length: arc length is the
    better parameterisation for an unevenly sampled *path*, but it is meaningless for a
    point cloud, it moves vertices even when the counts already match, and it needs a
    special case for closed and self-intersecting curves. Index parameterisation is
    shape-preserving for the common case (an evenly sampled curve) and at least
    well-defined for the rest.

    A single input row is repeated; an empty array raises, because there is no shape to
    resample and returning zeros would put a phantom vertex at the origin.
    """
    arr = np.asarray(vertices, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"vertices must be 2-D (N, D), got shape {arr.shape}")
    count = int(count)
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if len(arr) == 0:
        raise ValueError("cannot resample an empty vertex array")
    if len(arr) == count:
        return np.ascontiguousarray(arr, dtype=np.float32)
    if len(arr) == 1:
        return np.ascontiguousarray(np.repeat(arr, count, axis=0), dtype=np.float32)

    src = np.linspace(0.0, 1.0, len(arr))
    dst = np.linspace(0.0, 1.0, count)
    out = np.column_stack([np.interp(dst, src, arr[:, c]) for c in range(arr.shape[1])])
    return np.ascontiguousarray(out, dtype=np.float32)


def match_vertex_counts(a: Any, b: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(a, b)`` resampled to a common vertex count.

    **The vertex-count-mismatch policy, stated once:** both arrays are resampled to
    ``max(len(a), len(b))`` — the longer one is returned untouched, the shorter one is
    stretched onto the longer one's index grid (see :func:`resample_vertices`).

    Rejected alternatives:

    * **Truncate to the shorter.** Cheapest, and wrong: the detail of the more complex
      shape is thrown away permanently, so a morph from a 12-point square to a
      1000-point circle produces a 12-point circle at the end.
    * **Hold** — what :func:`glplot.core.timeline.interpolate_values` does on its own for
      mismatched shapes. Correct as a *default* (it refuses to invent geometry the user
      never keyed) but useless as a morph: nothing moves.
    * **Optimal point matching** (Hungarian assignment, as Manim's ``Transform`` does for
      submobjects). Genuinely better-looking, ``O(n³)``, and unusable at the vertex counts
      this engine exists to draw.

    The consequence to know about: when the counts differ, the *shorter* shape gains
    duplicate-ish vertices spread along its own path. For a polyline that is invisible
    (extra points on a line are still on the line). For a scatter it is not — a 10-point
    cloud morphing into a 100-point cloud starts as 10 clusters of 10 coincident points
    that fan out, which reads as the points "splitting". That is the honest picture of what
    the interpolation is doing, and there is no vertex-count-preserving alternative.

    Raises when the two have different column counts (2D vs 3D): a plane and a volume are
    not the same kind of thing and blending them would silently drop or invent an axis.
    """
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    if va.ndim != 2 or vb.ndim != 2:
        raise ValueError(f"both arrays must be 2-D, got {va.shape} and {vb.shape}")
    if va.shape[1] != vb.shape[1]:
        raise ValueError(
            f"cannot morph between {va.shape[1]}-D and {vb.shape[1]}-D geometry "
            f"(shapes {va.shape} and {vb.shape})"
        )
    n = max(len(va), len(vb))
    return resample_vertices(va, n), resample_vertices(vb, n)


def _pivot_of(arr: np.ndarray, pivot: Optional[Sequence[float]]) -> np.ndarray:
    """The rotation/scale centre: an explicit ``pivot``, else the vertex centroid.

    The centroid rather than the bounding-box centre, because it is what "spin this in
    place" means for a shape whose vertices are unevenly distributed, and because it needs
    no special case for a degenerate (zero-extent) axis.
    """
    if pivot is None:
        return np.asarray(arr, dtype=np.float64).mean(axis=0)
    p = np.asarray(pivot, dtype=np.float64).ravel()
    if len(p) != arr.shape[1]:
        raise ValueError(f"pivot must have {arr.shape[1]} components, got {len(p)}")
    return p


def rotate_vertices(
    vertices: Any,
    angle_deg: float,
    pivot: Optional[Sequence[float]] = None,
    axis: str = "z",
) -> np.ndarray:
    """Rotate an ``(N, 2)`` or ``(N, 3)`` vertex array about ``pivot``.

    2D rotates in the plane and ignores ``axis``. 3D rotates about the named world axis
    through the pivot — not about an arbitrary direction, because a named axis covers every
    case a plot actually asks for and an arbitrary one would need a rotation-vector
    convention this codebase does not otherwise have.
    """
    arr = np.asarray(vertices, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(f"vertices must be (N, 2) or (N, 3), got shape {arr.shape}")
    centre = _pivot_of(arr, pivot)
    local = arr - centre
    theta = math.radians(float(angle_deg))
    c, s = math.cos(theta), math.sin(theta)

    if arr.shape[1] == 2:
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    else:
        key = str(axis).strip().lower()
        if key == "x":
            rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
        elif key == "y":
            rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
        elif key == "z":
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        else:
            raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}")

    return np.ascontiguousarray(local @ rot.T + centre, dtype=np.float32)


def scale_vertices(
    vertices: Any,
    factor: Union[float, Sequence[float]],
    pivot: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Scale an ``(N, D)`` vertex array about ``pivot``. ``factor`` is scalar or per-axis."""
    arr = np.asarray(vertices, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"vertices must be 2-D (N, D), got shape {arr.shape}")
    centre = _pivot_of(arr, pivot)
    if isinstance(factor, (list, tuple, np.ndarray)):
        f = np.asarray(factor, dtype=np.float64).ravel()
        if len(f) != arr.shape[1]:
            raise ValueError(f"factor must have {arr.shape[1]} components, got {len(f)}")
    else:
        f = float(factor)
    return np.ascontiguousarray((arr - centre) * f + centre, dtype=np.float32)


def eased_fractions(easing_name: str, steps: int) -> List[float]:
    """``steps + 1`` already-eased progress values from 0 to 1 inclusive.

    What "baking" means here: instead of two keyframes and an eased segment between them,
    a verb emits ``steps + 1`` keyframes at *uniform times* whose values sit at these
    (non-uniform) fractions of the journey, and marks them ``linear``. The piecewise-linear
    track then approximates the eased curve, and — the point of the whole exercise — the
    intermediate values can be anything, not just a blend of the endpoints.
    """
    steps = max(1, int(steps))
    return [ease(easing_name, i / steps) for i in range(steps + 1)]


# ----------------------------------------------------------------------------------
# The span: what a verb hands to the timeline
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class PropertySpan:
    """One property's journey over a verb's span, in *fraction of duration* coordinates.

    Kept separate from the verb so that :meth:`Animation.apply` is written once and every
    verb reduces to "describe your spans". ``keys`` are ``(fraction, value)`` pairs with
    fractions in ``[0, 1]``; ``apply`` maps them onto wall-clock times.

    A value may be :data:`INHERIT` (only useful at fraction 0) or a :class:`Delta`; both are
    resolved by ``apply`` against the value the track already holds. Everything else is
    passed through untouched, so a span can carry a colour tuple, a numpy array, or a
    string.
    """

    prop: str
    keys: Tuple[Tuple[float, Any], ...]
    easing: str = DEFAULT_EASING
    #: False forces the whole track to hold. For values that cannot be blended: a colormap
    #: name, a visibility flag. Mirrors :attr:`glplot.core.timeline.Track.interpolate`.
    interpolate: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "prop", str(self.prop))
        object.__setattr__(self, "keys", tuple((float(f), v) for f, v in self.keys))


def span(
    prop: str,
    start: Any,
    end: Any,
    *,
    easing_name: str = DEFAULT_EASING,
    interpolate: bool = True,
) -> PropertySpan:
    """The ordinary two-keyframe span: ``start`` at the beginning, ``end`` at the end."""
    return PropertySpan(
        prop=prop, keys=((0.0, start), (1.0, end)), easing=easing_name, interpolate=interpolate
    )


def baked_span(prop: str, values: Sequence[Any], *, interpolate: bool = True) -> PropertySpan:
    """A span of pre-eased values at uniform times, every key ``linear``.

    The easing is already in the *values* (see :func:`eased_fractions`); marking the keys
    ``linear`` is what stops the track easing them a second time and turning a smooth
    rotation into a sequence of little lurches.
    """
    n = len(values)
    if n < 2:
        raise ValueError(f"a baked span needs at least two values, got {n}")
    keys = tuple((i / (n - 1), v) for i, v in enumerate(values))
    return PropertySpan(prop=prop, keys=keys, easing="linear", interpolate=interpolate)


# ----------------------------------------------------------------------------------
# The base verb
# ----------------------------------------------------------------------------------


class Animation:
    """Base class for every verb: a record that knows how to write itself into a timeline.

    Subclasses are ``@dataclass(frozen=True)`` and implement :meth:`spans`. They each
    declare their own ``easing`` field rather than inheriting one, because a dataclass base
    with a defaulted field forces every subclass field to be defaulted *and* pushes
    ``easing`` to the front of the positional signature, which would make
    ``Transform(source, dest)`` unwritable.
    """

    #: Only a fallback for verbs (like :class:`Wait`) that have no easing of their own.
    easing: str = DEFAULT_EASING

    # -- to implement ----------------------------------------------------------

    def spans(self) -> Tuple[PropertySpan, ...]:
        """The properties this verb animates. Override."""
        raise NotImplementedError

    # -- the shared machinery --------------------------------------------------

    def apply(
        self,
        timeline: Timeline,
        target: Any,
        start: float = 0.0,
        duration: float = DEFAULT_DURATION,
        *,
        fit: bool = True,
        **overrides: Any,
    ) -> List[Track]:
        """Write this verb's keyframes onto ``target``'s tracks in ``timeline``.

        Parameters
        ----------
        timeline
            Where the keyframes go. Usually ``panel.timeline``.
        target
            The identifier :mod:`glplot.anim.applier` will resolve: a ``layer_id``, or one
            of ``"camera"`` / ``"camera3d"`` / ``"axes3d"`` / ``"timeline"`` / ``"options"``.
            Opaque here — this module never learns what a layer is.
        start, duration
            The span in seconds. A ``duration`` of 0 collapses every keyframe onto one
            instant; since :meth:`Track.add` replaces a key at the same time, that leaves
            exactly the destination value, i.e. a snap. Negative durations are clamped to 0.
        fit
            Grow the timeline's duration to cover the keyframes just written. On by
            default because a verb scheduled past the end of a 5-second default timeline
            would otherwise be unreachable, and "my animation stops early" is a much worse
            surprise than "my timeline got longer".
        **overrides
            Field overrides for this one application, applied with
            :func:`dataclasses.replace` — ``FadeIn().apply(tl, 1, easing="bounce")``. An
            unknown field raises, deliberately: a silently ignored typo in an override is a
            bug the author would debug by staring at the animation.

        Returns the tracks that were touched, in span order.
        """
        spec = dataclasses.replace(self, **overrides) if overrides else self
        start = float(start)
        duration = max(0.0, float(duration))

        touched: List[Track] = []
        for property_span in spec.spans():
            track = timeline.track(target, property_span.prop)
            assert track is not None  # create=True
            # Read the base *before* writing, or the first key we add would become its own
            # start value and INHERIT would resolve to whatever we are about to write.
            base: Any = track.value_at(start) if track.keyframes else None

            track.interpolate = bool(property_span.interpolate)
            wrote_any = False
            for fraction, value in property_span.keys:
                if isinstance(value, _Sentinel):
                    if value is INHERIT and base is not None:
                        resolved = base
                    else:
                        # Nothing to inherit and nothing to invent: skip the key rather
                        # than fabricating a start the author never specified.
                        continue
                elif isinstance(value, Delta):
                    resolved = value.resolve(base)
                else:
                    resolved = value

                if fraction <= 0.0:
                    # Later Deltas in this span measure from the resolved start, not from
                    # the track's previous value — "shift by 3 from where this move begins".
                    base = resolved
                track.add(start + duration * fraction, resolved, property_span.easing)
                wrote_any = True

            if wrote_any:
                touched.append(track)

        if fit:
            timeline.fit_duration()
        return touched

    # -- convenience -----------------------------------------------------------

    def play(self, target: Any, duration: float = DEFAULT_DURATION, delay: float = 0.0) -> "Play":
        """Bind this verb to a target and a duration, for the sequencing helpers."""
        return Play(animation=self, target=target, duration=duration, delay=delay)

    def props(self) -> Tuple[str, ...]:
        """The property names this verb writes. Handy for a timeline panel's summary."""
        return tuple(s.prop for s in self.spans())


# ----------------------------------------------------------------------------------
# Opacity
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class FadeIn(Animation):
    """Bring a layer up from transparent.

    ``prop`` exists so the same verb can fade a 3D layer's *outline*
    (``prop="outline_alpha"``) independently of the layer itself, which
    :class:`~glplot.core.layers.LayerStyle` supports on purpose and which is the one
    genuinely useful variant.
    """

    to: float = 1.0
    start_alpha: Any = 0.0
    prop: str = PROP_ALPHA
    easing: str = DEFAULT_EASING

    def spans(self) -> Tuple[PropertySpan, ...]:
        return (span(self.prop, self.start_alpha, float(self.to), easing_name=self.easing),)


@dataclass(frozen=True)
class FadeOut(Animation):
    """Take a layer down to transparent.

    ``start_alpha`` defaults to 1.0 — :class:`~glplot.core.layers.LayerStyle`'s own default
    — rather than to :data:`INHERIT`, so a bare ``FadeOut()`` on a never-keyed layer does
    the obvious thing instead of snapping to invisible. Pass ``start_alpha=INHERIT`` to
    continue from whatever a previous verb left behind.
    """

    to: float = 0.0
    start_alpha: Any = 1.0
    prop: str = PROP_ALPHA
    easing: str = DEFAULT_EASING

    def spans(self) -> Tuple[PropertySpan, ...]:
        return (span(self.prop, self.start_alpha, float(self.to), easing_name=self.easing),)


@dataclass(frozen=True)
class Show(Animation):
    """Switch a layer visible at the start of the span. One held keyframe, no blending.

    ``visible`` is a boolean: "half visible" has no meaning a renderer could draw, so the
    track is marked non-interpolating and the key is placed at fraction 0 — the switch
    happens when the verb *starts*, which is what "show it now" means.
    """

    prop: str = PROP_VISIBLE

    def spans(self) -> Tuple[PropertySpan, ...]:
        return (PropertySpan(self.prop, ((0.0, True),), easing=STEP, interpolate=False),)


@dataclass(frozen=True)
class Hide(Animation):
    """Switch a layer invisible at the start of the span. See :class:`Show`."""

    prop: str = PROP_VISIBLE

    def spans(self) -> Tuple[PropertySpan, ...]:
        return (PropertySpan(self.prop, ((0.0, False),), easing=STEP, interpolate=False),)


@dataclass(frozen=True)
class Set(Animation):
    """Snap one property to one value at the start of the span, and hold it.

    The escape hatch for everything that cannot be blended — a colormap name, a projection
    mode, a label — and the verb that keeps this module from needing one class per string
    property. ``Set("projection", "orthographic")``.
    """

    prop: str = ""
    value: Any = None

    def spans(self) -> Tuple[PropertySpan, ...]:
        if not self.prop:
            raise ValueError("Set needs a prop name")
        return (PropertySpan(self.prop, ((0.0, self.value),), easing=STEP, interpolate=False),)


# ----------------------------------------------------------------------------------
# Progressive reveal
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Create(Animation):
    """Draw a layer's geometry on, progressively — Manim's ``Create``.

    Keys :data:`PROP_DRAW_FRACTION` from 0 to 1. The applier turns that fraction into a
    prefix of the layer's vertex array, so a polyline draws itself along its own path and a
    scatter fills in point by point. Nothing about *how* is decided here: the verb only says
    "this much of it".

    Not implemented as a vertex morph from a degenerate shape, which would have needed no
    new property: a morph from a point interpolates *positions*, so the curve would grow
    outwards from the middle rather than being traced from one end, and the intermediate
    frames would show a shape the data never had.
    """

    from_fraction: float = 0.0
    to_fraction: float = 1.0
    prop: str = PROP_DRAW_FRACTION
    easing: str = DEFAULT_EASING

    def spans(self) -> Tuple[PropertySpan, ...]:
        return (
            span(
                self.prop,
                float(self.from_fraction),
                float(self.to_fraction),
                easing_name=self.easing,
            ),
        )


@dataclass(frozen=True)
class Uncreate(Animation):
    """Un-draw a layer's geometry, progressively. :class:`Create` in reverse."""

    from_fraction: float = 1.0
    to_fraction: float = 0.0
    prop: str = PROP_DRAW_FRACTION
    easing: str = DEFAULT_EASING

    def spans(self) -> Tuple[PropertySpan, ...]:
        return (
            span(
                self.prop,
                float(self.from_fraction),
                float(self.to_fraction),
                easing_name=self.easing,
            ),
        )


@dataclass(frozen=True)
class Write(Animation):
    """Reveal a text layer's string progressively — Manim's ``Write``.

    Keys :data:`PROP_TEXT_FRACTION`; the applier slices the string. Character-by-character,
    not stroke-by-stroke as Manim does: this engine draws text through imgui's font atlas
    and has no access to glyph outlines, so there are no strokes to trace. The result reads
    as a typewriter rather than as handwriting, which is the honest thing to render given
    what the text pipeline can do.

    ``linear`` by default: an eased typewriter types fast in the middle and stalls at the
    ends, which reads as a stutter rather than as typing.
    """

    from_fraction: float = 0.0
    to_fraction: float = 1.0
    prop: str = PROP_TEXT_FRACTION
    easing: str = "linear"

    def spans(self) -> Tuple[PropertySpan, ...]:
        return (
            span(
                self.prop,
                float(self.from_fraction),
                float(self.to_fraction),
                easing_name=self.easing,
            ),
        )


@dataclass(frozen=True)
class Unwrite(Animation):
    """Hide a text layer's string progressively. :class:`Write` in reverse."""

    from_fraction: float = 1.0
    to_fraction: float = 0.0
    prop: str = PROP_TEXT_FRACTION
    easing: str = "linear"

    def spans(self) -> Tuple[PropertySpan, ...]:
        return (
            span(
                self.prop,
                float(self.from_fraction),
                float(self.to_fraction),
                easing_name=self.easing,
            ),
        )


# ----------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Transform(Animation):
    """Morph one vertex array into another — Manim's ``Transform``.

    Both arrays are stored in the record, at authoring time, as ordinary keyframe values;
    the timeline then blends them elementwise like any other pair of equal-shape arrays
    (:func:`glplot.core.timeline.interpolate_values`). No engine access, no callbacks, and
    the whole morph is visible in the timeline as two keyframes.

    **Vertex-count mismatch.** When ``resample`` is True (the default) both endpoints are
    put onto a common vertex count with :func:`match_vertex_counts` — the longer count wins
    and the shorter shape is stretched onto its index grid. Read that function's docstring
    for the alternatives that were rejected and for the one visible consequence (a growing
    *scatter* appears to split, because its new points start out coincident). With
    ``resample=False`` the arrays are stored as given, and a mismatch makes the timeline
    hold the source for the whole span — i.e. nothing moves. That is a deliberate, if
    useless, option: it is the only way to say "do not invent vertices".

    2D and 3D geometry cannot be morphed into each other; :func:`match_vertex_counts`
    raises rather than padding a z of zero, which would look like the shape collapsing onto
    the floor.
    """

    source: Any = None
    dest: Any = None
    prop: str = PROP_VERTICES
    resample: bool = True
    easing: str = DEFAULT_EASING

    @classmethod
    def from_layers(cls, source_layer: Any, dest_layer: Any, **kw: Any) -> "Transform":
        """Snapshot two layers' vertex arrays into a record.

        Duck-typed on ``pts`` then ``vertices``, which is every geometry-bearing layer this
        engine has. The arrays are *copied*, so a later edit to either layer does not
        silently rewrite the animation.
        """

        def _verts(layer: Any) -> np.ndarray:
            for attr in ("pts", "vertices"):
                arr = getattr(layer, attr, None)
                if arr is not None:
                    return np.array(arr, dtype=np.float32, copy=True)
            raise ValueError(
                f"layer {getattr(layer, 'label', layer)!r} has no pts/vertices to morph"
            )

        return cls(source=_verts(source_layer), dest=_verts(dest_layer), **kw)

    def resolved(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(source, dest)`` as the keyframes will store them."""
        if self.source is None or self.dest is None:
            raise ValueError("Transform needs both a source and a dest vertex array")
        if not self.resample:
            return (
                np.ascontiguousarray(self.source, dtype=np.float32),
                np.ascontiguousarray(self.dest, dtype=np.float32),
            )
        return match_vertex_counts(self.source, self.dest)

    def spans(self) -> Tuple[PropertySpan, ...]:
        source, dest = self.resolved()
        return (span(self.prop, source, dest, easing_name=self.easing),)


#: Manim calls it ``Transform``; the graphics literature calls it a morph. Same verb.
Morph = Transform


@dataclass(frozen=True)
class Rotate(Animation):
    """Spin a vertex array about a pivot — Manim's ``Rotate``.

    **Baked**, and it has to be: two keyframes would interpolate vertex *positions*, so a
    90° turn would cut the corner and a 180° turn would collapse the shape through the
    pivot and turn it inside out. This emits ``steps + 1`` keyframes along the real arc, at
    already-eased fractions, each marked ``linear``. ``steps`` defaults to one key per
    :data:`ROTATE_DEGREES_PER_KEY` degrees.

    ``pivot`` defaults to the vertex centroid, i.e. "spin in place". ``axis`` is ignored for
    2D geometry.
    """

    vertices: Any = None
    angle: float = 360.0
    pivot: Optional[Sequence[float]] = None
    axis: str = "z"
    steps: Optional[int] = None
    prop: str = PROP_VERTICES
    easing: str = DEFAULT_EASING

    @classmethod
    def from_layer(cls, layer: Any, angle: float = 360.0, **kw: Any) -> "Rotate":
        """Snapshot a layer's vertex array into a record. See :meth:`Transform.from_layers`."""
        for attr in ("pts", "vertices"):
            arr = getattr(layer, attr, None)
            if arr is not None:
                return cls(vertices=np.array(arr, dtype=np.float32, copy=True), angle=angle, **kw)
        raise ValueError(f"layer {getattr(layer, 'label', layer)!r} has no pts/vertices to rotate")

    def step_count(self) -> int:
        if self.steps is not None:
            return max(1, int(self.steps))
        return max(2, int(math.ceil(abs(float(self.angle)) / ROTATE_DEGREES_PER_KEY)))

    def spans(self) -> Tuple[PropertySpan, ...]:
        if self.vertices is None:
            raise ValueError("Rotate needs a vertex array")
        base = np.asarray(self.vertices, dtype=np.float64)
        angle = float(self.angle)
        values = [
            rotate_vertices(base, angle * f, pivot=self.pivot, axis=self.axis)
            for f in eased_fractions(self.easing, self.step_count())
        ]
        return (baked_span(self.prop, values),)


@dataclass(frozen=True)
class ScaleTo(Animation):
    """Grow or shrink a vertex array about a pivot — Manim's ``.scale()``.

    Two keyframes are exact here, unlike :class:`Rotate`: scaling is linear in the
    vertices, so blending the endpoint positions gives the same answer as blending the
    factor. ``factor`` is a scalar or one value per axis.
    """

    vertices: Any = None
    factor: Union[float, Sequence[float]] = 1.0
    pivot: Optional[Sequence[float]] = None
    prop: str = PROP_VERTICES
    easing: str = DEFAULT_EASING

    @classmethod
    def from_layer(cls, layer: Any, factor: Union[float, Sequence[float]], **kw: Any) -> "ScaleTo":
        """Snapshot a layer's vertex array into a record."""
        for attr in ("pts", "vertices"):
            arr = getattr(layer, attr, None)
            if arr is not None:
                return cls(vertices=np.array(arr, dtype=np.float32, copy=True), factor=factor, **kw)
        raise ValueError(f"layer {getattr(layer, 'label', layer)!r} has no pts/vertices to scale")

    def spans(self) -> Tuple[PropertySpan, ...]:
        if self.vertices is None:
            raise ValueError("ScaleTo needs a vertex array")
        base = np.ascontiguousarray(np.asarray(self.vertices, dtype=np.float32))
        scaled = scale_vertices(base, self.factor, pivot=self.pivot)
        return (span(self.prop, base, scaled, easing_name=self.easing),)


# ----------------------------------------------------------------------------------
# Placement
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class MoveTo(Animation):
    """Translate a layer to an absolute offset — Manim's ``.move_to()``.

    Keys :data:`PROP_TRANSLATION`, the ``(dx, dy)`` offset every 2D layer carries, rather
    than rewriting vertices: it is one pair of floats per frame instead of an array, the
    renderer already applies it, and it composes with a geometry morph on the same layer.
    3D layers carry the attribute too but no 3D renderer reads it — move a 3D layer with
    :class:`Transform` on shifted vertices, or with a camera :attr:`pan`.
    """

    to: Sequence[float] = (0.0, 0.0)
    start_at: Any = (0.0, 0.0)
    prop: str = PROP_TRANSLATION
    easing: str = DEFAULT_EASING

    def spans(self) -> Tuple[PropertySpan, ...]:
        dest = tuple(float(v) for v in self.to)
        return (span(self.prop, self.start_at, dest, easing_name=self.easing),)


@dataclass(frozen=True)
class Shift(Animation):
    """Translate a layer *by* an offset — Manim's ``.shift()``.

    Relative by construction: the destination is a :class:`Delta`, resolved at apply time
    against ``start_at`` (or, with ``start_at=INHERIT``, against whatever the track already
    holds). ``(0, 0)`` is the right default start because that is what
    :attr:`BaseLayer.translation` is on a fresh layer.
    """

    by: Sequence[float] = (0.0, 0.0)
    start_at: Any = (0.0, 0.0)
    prop: str = PROP_TRANSLATION
    easing: str = DEFAULT_EASING

    def spans(self) -> Tuple[PropertySpan, ...]:
        amount = tuple(float(v) for v in self.by)
        neutral = tuple(0.0 for _ in amount)
        return (
            span(
                self.prop,
                self.start_at,
                Delta(amount, default=neutral),
                easing_name=self.easing,
            ),
        )


# ----------------------------------------------------------------------------------
# Colour
# ----------------------------------------------------------------------------------


def _rgba(color: Any) -> Tuple[float, float, float, float]:
    """Normalise an RGB/RGBA sequence to a 4-tuple. A 3-tuple gains alpha 1."""
    values = [float(c) for c in np.asarray(color, dtype=np.float64).reshape(-1)]
    if len(values) == 3:
        values.append(1.0)
    if len(values) != 4:
        raise ValueError(f"color must have 3 or 4 components, got {len(values)}")
    return (values[0], values[1], values[2], values[3])


@dataclass(frozen=True)
class ColorTo(Animation):
    """Cross-fade a layer's colour — Manim's ``.set_color()`` under an animation.

    Interpolated in straight RGBA, which is what every other colour path in this codebase
    does (``pyplot``, the colormap helpers, the style panel). Perceptually a fade through
    linear RGB dips in brightness between complementary colours; a CIELAB path would not,
    but it would also disagree with every static colour this engine draws, and one
    consistent model beats two correct ones.

    ``start_color`` defaults to :data:`INHERIT`. There is no neutral colour to fall back on
    — black and the fallback blue are both wrong most of the time — so on a never-keyed
    track this writes only the destination and the colour snaps. Use
    :meth:`from_layer` to capture the layer's current colour instead.
    """

    to: Any = (1.0, 1.0, 1.0, 1.0)
    start_color: Any = INHERIT
    prop: str = PROP_COLOR
    easing: str = DEFAULT_EASING

    @classmethod
    def from_layer(cls, layer: Any, to: Any, **kw: Any) -> "ColorTo":
        """Snapshot ``layer.style.color`` as the start, falling back to opaque white."""
        current = getattr(getattr(layer, "style", None), "color", None)
        start = _rgba(current) if current is not None else (1.0, 1.0, 1.0, 1.0)
        return cls(to=to, start_color=start, **kw)

    def spans(self) -> Tuple[PropertySpan, ...]:
        start = self.start_color
        if not isinstance(start, _Sentinel) and start is not None:
            start = _rgba(start)
        return (span(self.prop, start, _rgba(self.to), easing_name=self.easing),)


@dataclass(frozen=True)
class ColorMapTo(Animation):
    """Switch a layer's colormap. A **held** keyframe — colormap names cannot be blended.

    "Half viridis, half magma" is not a colormap, and the halfway string would not name one
    either, so the track is marked non-interpolating and the switch lands at the start of
    the span. The blendable version of this verb does not exist and should not: to
    cross-fade between two colour mappings, put the layer's two colourings on two layers and
    :class:`FadeIn` one over the other.
    """

    to: str = "viridis"
    prop: str = PROP_CMAP

    def spans(self) -> Tuple[PropertySpan, ...]:
        return (PropertySpan(self.prop, ((0.0, str(self.to)),), easing=STEP, interpolate=False),)


# ----------------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraMoveTo(Animation):
    """Fly the 3D camera to a pose. Apply it to the ``"camera3d"`` target.

    Every field is optional and an omitted one (:data:`UNSET`) is not keyed at all, so
    ``CameraMoveTo(azim=90)`` sweeps the azimuth and leaves elevation, roll and dolly
    exactly as the user left them. That is the difference between a camera *verb* and a
    camera *preset*, and it is why the fields are ``UNSET``-defaulted rather than
    ``None``-defaulted: ``distance=None`` is a real pose ("fit the data") and has to stay
    reachable.

    ``start`` is a mapping of the pose the move begins from. Omitted, every span inherits
    from its track — which is right in a sequence and a snap on a fresh timeline, so
    :meth:`from_camera` is the constructor to reach for when animating a live figure.
    """

    elev: Any = UNSET
    azim: Any = UNSET
    roll: Any = UNSET
    distance: Any = UNSET
    fov: Any = UNSET
    pan: Any = UNSET
    box_aspect: Any = UNSET
    start: Optional[Mapping[str, Any]] = None
    easing: str = DEFAULT_EASING

    @classmethod
    def from_camera(cls, camera: Any, *, easing: str = DEFAULT_EASING, **to: Any) -> "CameraMoveTo":
        """Snapshot ``camera``'s current pose as the start, for exactly the props in ``to``.

        Only the props being animated are captured. Snapshotting the whole pose would key
        (and therefore pin) elevation and roll on a verb that only meant to sweep the
        azimuth, which is the surprise this whole class is arranged to avoid.
        """
        unknown = [k for k in to if k not in CAMERA3D_PROPS]
        if unknown:
            raise ValueError(
                f"unknown camera property/properties {', '.join(sorted(unknown))}; "
                f"expected some of {', '.join(CAMERA3D_PROPS)}"
            )
        start = {name: getattr(camera, name) for name in to}
        return cls(start=start, easing=easing, **to)

    def spans(self) -> Tuple[PropertySpan, ...]:
        start = dict(self.start or {})
        out: List[PropertySpan] = []
        for name in CAMERA3D_PROPS:
            dest = getattr(self, name)
            if isinstance(dest, _Sentinel):
                continue
            out.append(span(name, start.get(name, INHERIT), dest, easing_name=self.easing))
        return tuple(out)


@dataclass(frozen=True)
class Orbit(Animation):
    """Sweep the 3D camera's azimuth by whole turns. Apply it to the ``"camera3d"`` target.

    The signature verb of a 3D presentation, and the reason it is not just
    ``CameraMoveTo(azim=...)``: the destination is *relative* (a :class:`Delta`), so
    ``Orbit(turns=2)`` means two more turns from wherever the camera is, and the keyframe
    value really is 720 rather than wrapping to 0 — a wrapped angle would make the camera
    unwind at the seam.

    ``linear`` by default: an eased orbit accelerates and brakes, which reads as the scene
    lurching rather than turning.

    ``start_azim`` defaults to 0. :data:`INHERIT` would be the more composable default, but
    on a fresh timeline it has nothing to inherit and the sweep would degenerate to a single
    destination key (see the module docstring); 0 always produces a real turn. Pass
    ``start_azim=INHERIT`` inside a sequence, or use :meth:`from_camera` to start from where
    the camera actually is.
    """

    turns: float = 1.0
    start_azim: Any = 0.0
    steps: Optional[int] = None
    prop: str = "azim"
    easing: str = "linear"

    @classmethod
    def from_camera(cls, camera: Any, turns: float = 1.0, **kw: Any) -> "Orbit":
        """Sweep from the camera's current azimuth."""
        return cls(turns=turns, start_azim=float(getattr(camera, "azim", 0.0)), **kw)

    def step_count(self) -> int:
        if self.steps is not None:
            return max(1, int(self.steps))
        return max(1, int(round(abs(float(self.turns)) * ORBIT_KEYS_PER_TURN)))

    def spans(self) -> Tuple[PropertySpan, ...]:
        sweep = 360.0 * float(self.turns)
        steps = self.step_count()
        # Baked, but only for legibility: an angle *is* linear in time, so two keys would
        # give identical playback. The intermediate keys make the sweep readable in a
        # keyframe editor and let a non-linear easing show up in the key spacing.
        fractions = eased_fractions(self.easing, steps)
        keys: List[Tuple[float, Any]] = []
        for i, eased in enumerate(fractions):
            value: Any = self.start_azim if i == 0 else Delta(sweep * eased, default=0.0)
            keys.append((i / steps, value))
        return (PropertySpan(self.prop, tuple(keys), easing="linear"),)


@dataclass(frozen=True)
class ZoomTo(Animation):
    """Dolly or zoom. Works on both cameras, because the caller says which one.

    Apply it to ``"camera3d"`` with ``distance=`` (and optionally ``fov=``), or to
    ``"camera"`` with ``zoom=`` / ``zoom_x=`` / ``zoom_y=`` / ``cx=`` / ``cy=``. One verb
    rather than two because "get closer" is one intent, and the target argument already
    disambiguates which camera is meant — a ``ZoomTo3D`` would only be a longer way to say
    the same thing.

    ``zoom`` is shorthand for setting ``zoom_x`` and ``zoom_y`` together; an explicit
    ``zoom_x``/``zoom_y`` wins over it, which is what lets an anisotropic zoom be expressed
    without repeating the isotropic part.
    """

    distance: Any = UNSET
    fov: Any = UNSET
    zoom: Any = UNSET
    zoom_x: Any = UNSET
    zoom_y: Any = UNSET
    cx: Any = UNSET
    cy: Any = UNSET
    start: Optional[Mapping[str, Any]] = None
    easing: str = DEFAULT_EASING

    @classmethod
    def from_camera(cls, camera: Any, *, easing: str = DEFAULT_EASING, **to: Any) -> "ZoomTo":
        """Snapshot the start values for exactly the props in ``to`` off a live camera."""
        expanded = dict(to)
        if "zoom" in expanded:
            value = expanded.pop("zoom")
            expanded.setdefault("zoom_x", value)
            expanded.setdefault("zoom_y", value)
        start = {name: getattr(camera, name) for name in expanded if hasattr(camera, name)}
        return cls(start=start, easing=easing, **to)

    def spans(self) -> Tuple[PropertySpan, ...]:
        start = dict(self.start or {})
        dests: List[Tuple[str, Any]] = []

        if not isinstance(self.zoom, _Sentinel):
            for name in ("zoom_x", "zoom_y"):
                if isinstance(getattr(self, name), _Sentinel):
                    dests.append((name, self.zoom))
        for name in ("distance", "fov", "zoom_x", "zoom_y", "cx", "cy"):
            value = getattr(self, name)
            if not isinstance(value, _Sentinel):
                dests.append((name, value))

        return tuple(
            span(name, start.get(name, INHERIT), value, easing_name=self.easing)
            for name, value in dests
        )


# ----------------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Wait(Animation):
    """Animate nothing, for a while. Manim's ``self.wait()``.

    Writes no keyframes and therefore touches no tracks; its whole purpose is to consume a
    slot in a :func:`succession` so the beat between two verbs is stated in the script
    rather than smuggled in as an arithmetic offset on the next ``start``.
    """

    def spans(self) -> Tuple[PropertySpan, ...]:
        return ()


# ----------------------------------------------------------------------------------
# Sequencing
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Play:
    """An animation bound to a target and a run time — the unit the schedulers order.

    A verb does not carry its own target (``apply`` takes one) because the same
    ``FadeIn()`` should be usable on twenty layers without twenty records. The schedulers
    need the pairing though, so this is where it lives.
    """

    animation: Animation
    target: Any
    duration: float = DEFAULT_DURATION
    #: Extra offset inside this play's slot, in seconds. Shifts when the animation starts
    #: without changing where the *next* one begins in a ``lag_ratio=0`` group, which is how
    #: a hand-tuned accent is added to an otherwise parallel group.
    delay: float = 0.0


#: What the schedulers accept in place of a :class:`Play`: ``(animation, target)`` or
#: ``(animation, target, duration)``. Shorthand, because a five-verb succession written out
#: with ``Play(...)`` five times is mostly punctuation.
PlayLike = Union[Play, Tuple[Any, ...]]


def _coerce_play(item: PlayLike, default_duration: float) -> Play:
    if isinstance(item, Play):
        return item
    if isinstance(item, tuple) and len(item) in (2, 3, 4):
        animation, target = item[0], item[1]
        duration = float(item[2]) if len(item) > 2 else float(default_duration)
        delay = float(item[3]) if len(item) > 3 else 0.0
        return Play(animation=animation, target=target, duration=duration, delay=delay)
    raise TypeError(
        "expected a Play or an (animation, target[, duration[, delay]]) tuple, "
        f"got {type(item).__name__}"
    )


def group(
    timeline: Timeline,
    plays: Iterable[PlayLike],
    *,
    start: float = 0.0,
    lag_ratio: float = 1.0,
    duration: float = DEFAULT_DURATION,
    fit: bool = True,
) -> float:
    """Schedule ``plays`` with Manim's lag-ratio rule. Returns the time the group ends.

    The rule, taken verbatim from ``AnimationGroup.build_animations_with_timings``: each
    play starts at the running cursor, and the cursor then advances by
    ``lag_ratio * run_time`` — *not* by the full run time. So

    * ``lag_ratio = 1`` → each play starts exactly when the previous one ends: a
      **succession**;
    * ``lag_ratio = 0`` → every cursor advance is zero, so they all start together:
      **parallel**;
    * anything between → an overlap, which is the stagger that makes a group of twenty
      dots read as a wave instead of as a flash.

    Reimplemented rather than approximated because "lag_ratio" is a term of art that
    animators already know the meaning of, and a lag_ratio here that behaved differently
    from a lag_ratio there would be worse than not having one.

    ``duration`` is the run time used for plays given as bare ``(animation, target)``
    tuples. ``fit`` is passed through to :meth:`Animation.apply`.
    """
    start = float(start)
    lag_ratio = float(lag_ratio)
    cursor = start
    end = start
    for item in plays:
        play = _coerce_play(item, duration)
        at = cursor + float(play.delay)
        run_time = max(0.0, float(play.duration))
        play.animation.apply(timeline, play.target, at, run_time, fit=fit)
        end = max(end, at + run_time)
        cursor = at + lag_ratio * run_time
    if fit:
        timeline.fit_duration()
    return end


def succession(
    timeline: Timeline,
    *plays: PlayLike,
    start: float = 0.0,
    duration: float = DEFAULT_DURATION,
    fit: bool = True,
) -> float:
    """One after another. ``group`` with ``lag_ratio=1``. Returns the end time.

    The returned end time is what makes scripts chain::

        t = succession(tl, (FadeIn(), a), (Create(), b))
        t = succession(tl, (FadeOut(), a), start=t)
    """
    return group(timeline, plays, start=start, lag_ratio=1.0, duration=duration, fit=fit)


def parallel(
    timeline: Timeline,
    *plays: PlayLike,
    start: float = 0.0,
    duration: float = DEFAULT_DURATION,
    fit: bool = True,
) -> float:
    """All at once. ``group`` with ``lag_ratio=0``. Returns the end time.

    The end time is the *longest* play's end, not the last one's, so a parallel group of
    unequal durations chains correctly.
    """
    return group(timeline, plays, start=start, lag_ratio=0.0, duration=duration, fit=fit)


def stagger(
    timeline: Timeline,
    *plays: PlayLike,
    start: float = 0.0,
    lag_ratio: float = DEFAULT_LAG_RATIO,
    duration: float = DEFAULT_DURATION,
    fit: bool = True,
) -> float:
    """Overlapping, one behind the next — Manim's ``LaggedStart``. Returns the end time."""
    return group(timeline, plays, start=start, lag_ratio=lag_ratio, duration=duration, fit=fit)


def lagged(
    timeline: Timeline,
    animation: Animation,
    targets: Iterable[Any],
    *,
    start: float = 0.0,
    duration: float = DEFAULT_DURATION,
    lag_ratio: float = DEFAULT_LAG_RATIO,
    fit: bool = True,
) -> float:
    """Apply **one** verb to many targets, staggered. Returns the end time.

    The shape most Manim one-liners actually have (``LaggedStart(*[FadeIn(d) for d in
    dots])``) and the reason a verb does not carry its own target: one record, twenty
    layers, one line.
    """
    plays = [Play(animation=animation, target=t, duration=duration) for t in targets]
    return group(timeline, plays, start=start, lag_ratio=lag_ratio, duration=duration, fit=fit)


__all__ = [
    "Animation",
    "CAMERA2D_PROPS",
    "CAMERA3D_PROPS",
    "CameraMoveTo",
    "ColorMapTo",
    "ColorTo",
    "Create",
    "DEFAULT_DURATION",
    "DEFAULT_EASING",
    "DEFAULT_LAG_RATIO",
    "Delta",
    "FadeIn",
    "FadeOut",
    "Hide",
    "INHERIT",
    "Morph",
    "MoveTo",
    "ORBIT_KEYS_PER_TURN",
    "Orbit",
    "PROP_ALPHA",
    "PROP_CMAP",
    "PROP_COLOR",
    "PROP_DRAW_FRACTION",
    "PROP_TEXT_FRACTION",
    "PROP_TRANSLATION",
    "PROP_VERTICES",
    "PROP_VISIBLE",
    "Play",
    "PropertySpan",
    "ROTATE_DEGREES_PER_KEY",
    "Rotate",
    "ScaleTo",
    "Set",
    "Shift",
    "Show",
    "Transform",
    "UNSET",
    "Uncreate",
    "Unwrite",
    "Wait",
    "Write",
    "ZoomTo",
    "baked_span",
    "eased_fractions",
    "group",
    "lagged",
    "match_vertex_counts",
    "parallel",
    "resample_vertices",
    "rotate_vertices",
    "scale_vertices",
    "span",
    "stagger",
    "succession",
]
