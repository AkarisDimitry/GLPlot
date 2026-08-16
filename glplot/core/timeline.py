"""The timeline: keyframes, easing, scenes and a playhead, stored as plain data.

Pure numpy + stdlib. No GL, no imgui, no engine import — the same discipline
:mod:`glplot.core.camera3d` follows, and for the same reason: an animation model you can
only exercise by watching it play is an animation model you cannot test.

The model
---------
Four objects, and the whole system is their composition:

* :class:`Keyframe` — a value pinned at a time, plus the easing used to *leave* it.
* :class:`Track` — one animated property of one target, as a sorted list of keyframes.
  :meth:`Track.value_at` is the only place interpolation happens.
* :class:`Scene` — a named span of time. Scenes are what make a *presentation*: they give
  the timeline chapters you can jump between, play one at a time, and step through with a
  clicker, rather than one undifferentiated ribbon.
* :class:`Timeline` — the tracks, the scenes, the frame rate, and the playhead, plus the
  playback rules (loop, ping-pong, hold).

Continuous *and* discrete, deliberately
---------------------------------------
Two kinds of animation matter here and they are not the same thing:

* **Interpolated** — a camera sweeping from one angle to another, a colour fading, a point
  moving. Values between keyframes are computed. This is what easing is for.
* **Discrete / progression** — frame 12 of a Game of Life run, iteration 300 of a
  Mandelbrot zoom, the 40th step of an ODE integration. There is no "between": asking for
  a value at t = 3.5 must yield the state at step 3, not a blend of steps 3 and 4.

:data:`STEP` is the easing that expresses the second kind, and
:attr:`Track.interpolate` turns it off wholesale for a track whose values cannot be
blended at all (strings, booleans, ragged arrays). Blending two Game-of-Life grids into a
half-alive board would be a bug that *looks* like an artistic choice, so the model refuses
it rather than making it representable.

What a track's ``value`` may be
-------------------------------
Floats, sequences of floats (a colour, a 3-vector), and numpy arrays of equal shape —
those interpolate elementwise, which is what makes a geometry morph one line of code.
Anything else is carried through as a step value. :func:`interpolate_values` is the single
place that decides, so a new value type is one function to extend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

#: Frame rates the GUI offers. 60 is the interactive default; 24 is film; 30 and 50 are
#: broadcast; 12 is for stepping through a discrete process where every frame is a state.
COMMON_FPS: Tuple[int, ...] = (12, 24, 25, 30, 50, 60)

#: Default frame rate: the display rate, so playback in the window and the exported file
#: agree without resampling.
DEFAULT_FPS = 60.0

#: Hard ceiling on a timeline's duration in seconds. An hour of 60fps frames is 216 000
#: renders; past that the user wants a batch script, not a scrubber.
MAX_DURATION = 3600.0

#: Smallest usable duration. Below one frame there is nothing to play.
MIN_DURATION = 1.0 / 240.0

#: How playback behaves when the playhead reaches the end.
LOOP_MODES: Tuple[str, ...] = ("once", "loop", "pingpong")


class TimelineError(ValueError):
    """Raised for a malformed track, scene or playback request.

    A :class:`ValueError` subclass so callers that already catch ``ValueError`` — every
    panel action does — catch this too, matching :class:`glplot.gui.dynamics.DynamicsError`
    and :class:`glplot.gui.generators3d.GeneratorError`.
    """


# ----------------------------------------------------------------------------------
# Easing
# ----------------------------------------------------------------------------------
#
# Every easing maps [0, 1] -> [0, 1] and satisfies f(0) == 0 and f(1) == 1, so a segment
# always starts and ends exactly on its keyframes no matter which curve is chosen. The
# exception is `step`, which is deliberately discontinuous at 1 — that IS its meaning.


def _linear(t: float) -> float:
    return t


def _step(t: float) -> float:
    """Hold the start value for the whole segment. The discrete-progression easing."""
    return 0.0 if t < 1.0 else 1.0


def _smooth(t: float) -> float:
    """Smoothstep: zero velocity at both ends. The most useful default after linear."""
    return t * t * (3.0 - 2.0 * t)


def _smoother(t: float) -> float:
    """Smootherstep: zero velocity *and* acceleration at both ends."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _ease_in(t: float) -> float:
    return t * t


def _ease_out(t: float) -> float:
    return 1.0 - (1.0 - t) ** 2


def _ease_in_out(t: float) -> float:
    return 2.0 * t * t if t < 0.5 else 1.0 - ((-2.0 * t + 2.0) ** 2) / 2.0


def _ease_in_cubic(t: float) -> float:
    return t**3


def _ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def _ease_in_out_cubic(t: float) -> float:
    return 4.0 * t**3 if t < 0.5 else 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


def _ease_out_back(t: float) -> float:
    """Overshoots past 1 then settles. Reads as "arrives with weight"."""
    c1, c3 = 1.70158, 2.70158
    return 1.0 + c3 * (t - 1.0) ** 3 + c1 * (t - 1.0) ** 2


def _ease_out_elastic(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    c4 = (2.0 * math.pi) / 3.0
    return 2.0 ** (-10.0 * t) * math.sin((t * 10.0 - 0.75) * c4) + 1.0


def _ease_out_bounce(t: float) -> float:
    n1, d1 = 7.5625, 2.75
    if t < 1.0 / d1:
        return n1 * t * t
    if t < 2.0 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    if t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    t -= 2.625 / d1
    return n1 * t * t + 0.984375


#: The easing name used for a discrete progression. Named rather than spelled inline
#: because several modules test against it.
STEP = "step"

#: Every easing, by name, in menu order. The GUI iterates this, so adding a curve here is
#: all it takes to offer it.
EASINGS: Dict[str, Callable[[float], float]] = {
    "linear": _linear,
    STEP: _step,
    "smooth": _smooth,
    "smoother": _smoother,
    "ease_in": _ease_in,
    "ease_out": _ease_out,
    "ease_in_out": _ease_in_out,
    "ease_in_cubic": _ease_in_cubic,
    "ease_out_cubic": _ease_out_cubic,
    "ease_in_out_cubic": _ease_in_out_cubic,
    "back": _ease_out_back,
    "elastic": _ease_out_elastic,
    "bounce": _ease_out_bounce,
}

#: Human labels for the easing picker.
EASING_LABELS: Dict[str, str] = {
    "linear": "Linear",
    STEP: "Step (discrete)",
    "smooth": "Smooth",
    "smoother": "Smoother",
    "ease_in": "Ease in",
    "ease_out": "Ease out",
    "ease_in_out": "Ease in-out",
    "ease_in_cubic": "Ease in (cubic)",
    "ease_out_cubic": "Ease out (cubic)",
    "ease_in_out_cubic": "Ease in-out (cubic)",
    "back": "Back (overshoot)",
    "elastic": "Elastic",
    "bounce": "Bounce",
}


def easing(name: str) -> Callable[[float], float]:
    """The easing function called ``name``. Raises :class:`TimelineError` if unknown."""
    fn = EASINGS.get(str(name).strip().lower())
    if fn is None:
        raise TimelineError(
            f"unknown easing {name!r}; expected one of {', '.join(sorted(EASINGS))}"
        )
    return fn


def ease(name: str, t: float) -> float:
    """Apply easing ``name`` to a normalised ``t``, clamped to ``[0, 1]``."""
    return float(easing(name)(float(np.clip(t, 0.0, 1.0))))


# ----------------------------------------------------------------------------------
# Value interpolation
# ----------------------------------------------------------------------------------


def is_blendable(value: Any) -> bool:
    """Whether two values of this kind can meaningfully be mixed.

    Numbers, sequences of numbers and float arrays can. Strings, booleans and anything
    else cannot — and a boolean is the interesting case: ``True`` half way to ``False``
    has no meaning a plot could draw, so "visible at 0.5" must hold rather than blend.
    """
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return True
    if isinstance(value, np.ndarray):
        return value.dtype.kind in "fiu"
    if isinstance(value, (list, tuple)):
        return bool(value) and all(
            isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
            for v in value
        )
    return False


def interpolate_values(start: Any, end: Any, t: float) -> Any:
    """Blend ``start`` towards ``end`` by an already-eased ``t``.

    The single place the value model is decided. Falls back to holding ``start`` whenever
    the two cannot be blended — mismatched shapes, mismatched types, non-numeric values —
    rather than raising, because a timeline is scrubbed continuously and a keyframe pair
    the user is midway through editing must not take the frame down. The refusal is
    visible (the value simply does not move), which is the right feedback.
    """
    if t <= 0.0:
        return start
    if t >= 1.0:
        return end
    if not (is_blendable(start) and is_blendable(end)):
        return start

    if isinstance(start, np.ndarray) or isinstance(end, np.ndarray):
        a = np.asarray(start, dtype=np.float64)
        b = np.asarray(end, dtype=np.float64)
        if a.shape != b.shape:
            # A geometry morph between different vertex counts is a real thing to want,
            # but resampling one to the other silently would change what the user keyed.
            # Held instead; ``glplot.anim`` is where an explicit morph belongs.
            return start
        return a + (b - a) * float(t)

    if isinstance(start, (list, tuple)) or isinstance(end, (list, tuple)):
        a = list(start) if isinstance(start, (list, tuple)) else [start]
        b = list(end) if isinstance(end, (list, tuple)) else [end]
        if len(a) != len(b):
            return start
        blended = [float(x) + (float(y) - float(x)) * float(t) for x, y in zip(a, b)]
        return tuple(blended) if isinstance(start, tuple) else blended

    return float(start) + (float(end) - float(start)) * float(t)


# ----------------------------------------------------------------------------------
# Keyframes and tracks
# ----------------------------------------------------------------------------------


@dataclass
class Keyframe:
    """A value pinned at a time, and the easing used to *leave* it.

    The easing belongs to the keyframe the segment starts at, not to the segment or to the
    track. That is the convention every animation tool uses, and it is the one that lets a
    single track hold a snap, then a smooth glide, then a bounce, without needing a
    separate object per segment.
    """

    time: float
    value: Any
    easing: str = "smooth"

    def __post_init__(self) -> None:
        self.time = float(self.time)
        if not np.isfinite(self.time):
            raise TimelineError(f"keyframe time must be finite, got {self.time!r}")
        if str(self.easing).strip().lower() not in EASINGS:
            raise TimelineError(
                f"unknown easing {self.easing!r}; expected one of {', '.join(sorted(EASINGS))}"
            )
        self.easing = str(self.easing).strip().lower()


@dataclass
class Track:
    """One animated property of one target, as keyframes over time.

    Attributes
    ----------
    target
        What is animated, as a stable identifier the applier resolves — a ``layer_id``,
        ``"camera"``, ``"axes"``, an option name. Deliberately opaque here: the timeline
        knows *when* a value changes, and nothing about what a layer is.
    prop
        Which property of that target.
    keyframes
        Sorted by time. :meth:`add` keeps them sorted; do not append directly.
    interpolate
        False forces every segment to hold (as if every easing were ``step``). For a track
        whose values cannot be blended — a Game-of-Life grid, a label, a visibility flag —
        this states the intent once instead of relying on every keyframe being set to
        ``step`` and staying that way.
    enabled
        A muted track is kept but not evaluated, so a user can audition without deleting.
    """

    target: Any
    prop: str
    keyframes: List[Keyframe] = field(default_factory=list)
    interpolate: bool = True
    enabled: bool = True
    label: str = ""

    def __post_init__(self) -> None:
        self.prop = str(self.prop)
        self._sort()

    def _sort(self) -> None:
        self.keyframes.sort(key=lambda k: k.time)

    # -- editing ---------------------------------------------------------------

    def add(self, time: float, value: Any, easing_name: str = "smooth") -> Keyframe:
        """Add or replace the keyframe at ``time``, and return it.

        Replaces rather than appends when a key already sits at that time: two keyframes at
        one instant have no defined order, and the second would be unreachable. "Key the
        current value again" is by far the commonest reason to land on an existing time.
        """
        key = Keyframe(time=time, value=value, easing=easing_name)
        for i, existing in enumerate(self.keyframes):
            if abs(existing.time - key.time) <= 1e-9:
                self.keyframes[i] = key
                return key
        self.keyframes.append(key)
        self._sort()
        return key

    def remove_at(self, time: float, tolerance: float = 1e-6) -> bool:
        """Remove the keyframe at ``time``. False when there is none."""
        for i, existing in enumerate(self.keyframes):
            if abs(existing.time - float(time)) <= tolerance:
                del self.keyframes[i]
                return True
        return False

    def move(self, index: int, time: float) -> bool:
        """Move keyframe ``index`` to ``time``, re-sorting. False when out of range."""
        if not 0 <= int(index) < len(self.keyframes):
            return False
        self.keyframes[int(index)].time = float(time)
        self._sort()
        return True

    def clear(self) -> None:
        self.keyframes.clear()

    # -- queries ---------------------------------------------------------------

    def is_empty(self) -> bool:
        return not self.keyframes

    def time_range(self) -> Optional[Tuple[float, float]]:
        """``(first, last)`` keyframe time, or None when the track is empty."""
        if not self.keyframes:
            return None
        return self.keyframes[0].time, self.keyframes[-1].time

    def value_at(self, time: float) -> Any:
        """The track's value at ``time``. The only place interpolation happens.

        Outside the keyed range the value is **held**, not extrapolated: an eased curve
        continued past its last key produces values the user never specified, and for a
        bounce or an overshoot those are wildly out of range.
        """
        keys = self.keyframes
        if not keys:
            return None
        time = float(time)
        if time <= keys[0].time:
            return keys[0].value
        if time >= keys[-1].time:
            return keys[-1].value

        # Binary search for the segment. A track can hold thousands of keys once it comes
        # from a baked simulation, and this is called once per track per frame.
        times = [k.time for k in keys]
        hi = int(np.searchsorted(times, time, side="right"))
        lo = max(hi - 1, 0)
        start, end = keys[lo], keys[min(hi, len(keys) - 1)]

        span = end.time - start.time
        if span <= 1e-12:
            return end.value
        raw = (time - start.time) / span
        if not self.interpolate:
            return start.value
        return interpolate_values(start.value, end.value, ease(start.easing, raw))

    def bake(self, times: Sequence[float]) -> List[Any]:
        """The value at each of ``times`` — for export, where every frame is wanted."""
        return [self.value_at(t) for t in times]

    def key(self) -> Tuple[Any, str]:
        """``(target, prop)`` — the identity a timeline indexes tracks by."""
        return (self.target, self.prop)


# ----------------------------------------------------------------------------------
# Scenes
# ----------------------------------------------------------------------------------


@dataclass
class Scene:
    """A named span of the timeline — a chapter.

    Scenes are what turn a timeline into a *presentation*. Without them a talk is one
    ribbon you have to scrub blindly; with them there is a list you can jump between, play
    one at a time, and advance with a clicker. The span is a view onto the same tracks, not
    a container for them: a track may cross scene boundaries, which is what lets a camera
    move continue across a chapter change.
    """

    name: str
    start: float
    end: float
    description: str = ""
    #: Hold at the end of this scene instead of running on, so a presenter controls the
    #: pace. This is the "advance on click" behaviour of every slide tool.
    hold: bool = True

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.start = float(self.start)
        self.end = float(self.end)
        if not (self.end > self.start):
            raise TimelineError(
                f"scene {self.name!r} needs end > start, got ({self.start}, {self.end})"
            )

    @property
    def duration(self) -> float:
        return self.end - self.start

    def contains(self, time: float) -> bool:
        """True when ``time`` falls in this scene. Half-open, so scenes can abut."""
        return self.start <= float(time) < self.end

    def progress(self, time: float) -> float:
        """How far through the scene ``time`` is, clamped to ``[0, 1]``."""
        return float(np.clip((float(time) - self.start) / max(self.duration, 1e-12), 0.0, 1.0))


# ----------------------------------------------------------------------------------
# The timeline
# ----------------------------------------------------------------------------------


class Timeline:
    """Tracks, scenes, a frame rate and a playhead.

    Owned per :class:`~glplot.core.panel.Panel`, beside the 2D and 3D cameras, so a split
    figure can animate one panel while another stays still — the same reason the camera
    lives there.

    Playback is advanced from **wall-clock deltas**, not per frame, so an animation runs at
    the speed it was authored at whether the window is managing 12fps or 144. The frame
    *grid* (:meth:`frame_times`) is separate and is what export walks: playback may skip
    frames, an export never may.
    """

    def __init__(
        self,
        duration: float = 5.0,
        fps: float = DEFAULT_FPS,
        loop: str = "loop",
    ) -> None:
        self.duration = float(np.clip(duration, MIN_DURATION, MAX_DURATION))
        self.fps = float(fps) if fps and fps > 0 else DEFAULT_FPS
        self.set_loop(loop)

        self.tracks: List[Track] = []
        self.scenes: List[Scene] = []

        #: Current time in seconds. Always inside ``[0, duration]``.
        self.time: float = 0.0
        self.playing: bool = False
        #: Playback rate multiplier. Negative plays backwards, which is genuinely useful
        #: for checking that an animation reads both ways.
        self.speed: float = 1.0
        #: Direction, flipped by ping-pong. Never set this directly.
        self._direction: int = 1

    # -- configuration ---------------------------------------------------------

    def set_loop(self, mode: str) -> None:
        key = str(mode).strip().lower()
        if key not in LOOP_MODES:
            raise TimelineError(f"loop must be one of {', '.join(LOOP_MODES)}, got {mode!r}")
        self.loop = key

    def set_duration(self, seconds: float) -> None:
        """Change the length, keeping the playhead inside it."""
        self.duration = float(np.clip(seconds, MIN_DURATION, MAX_DURATION))
        self.time = float(np.clip(self.time, 0.0, self.duration))

    def set_fps(self, fps: float) -> None:
        self.fps = float(fps) if fps and fps > 0 else DEFAULT_FPS

    def fit_duration(self, *, minimum: float = 1.0) -> float:
        """Grow the duration to cover every keyframe and scene. Returns the new duration.

        Never shrinks: a user who set a long timeline on purpose and then deleted a late
        keyframe should not have the end pulled out from under the playhead.
        """
        latest = minimum
        for track in self.tracks:
            span = track.time_range()
            if span is not None:
                latest = max(latest, span[1])
        for scene in self.scenes:
            latest = max(latest, scene.end)
        if latest > self.duration:
            self.set_duration(latest)
        return self.duration

    # -- frames ----------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        """Number of frames at the current rate. Always at least one."""
        return max(1, int(round(self.duration * self.fps)) + 1)

    def frame_times(self) -> np.ndarray:
        """Every frame's time, inclusive of both ends. What an export walks."""
        return np.linspace(0.0, self.duration, self.frame_count)

    def frame_index(self, time: Optional[float] = None) -> int:
        """The frame number containing ``time`` (the playhead by default)."""
        t = self.time if time is None else float(time)
        return int(np.clip(round(t * self.fps), 0, self.frame_count - 1))

    def seek_frame(self, index: int) -> float:
        """Put the playhead on frame ``index``. Returns the new time."""
        index = int(np.clip(int(index), 0, self.frame_count - 1))
        self.time = float(np.clip(index / self.fps, 0.0, self.duration))
        return self.time

    def step_frames(self, count: int = 1) -> float:
        """Move ``count`` frames (negative for back). Returns the new time."""
        return self.seek_frame(self.frame_index() + int(count))

    # -- transport -------------------------------------------------------------

    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def toggle(self) -> bool:
        self.playing = not self.playing
        return self.playing

    def stop(self) -> None:
        """Pause and rewind — the transport verb ``pause`` deliberately is not."""
        self.playing = False
        self.time = 0.0
        self._direction = 1

    def seek(self, time: float) -> float:
        """Put the playhead at ``time``, clamped. Returns it."""
        self.time = float(np.clip(float(time), 0.0, self.duration))
        return self.time

    def advance(self, dt: float) -> bool:
        """Move the playhead by ``dt`` wall-clock seconds. True when it moved.

        A large ``dt`` — the loop blocked on a modal, an export, a slow first frame — is
        clamped rather than applied, because jumping half a second of animation reads as a
        glitch and, in ping-pong, could skip a whole bounce.
        """
        if not self.playing or self.duration <= 0.0:
            return False
        dt = float(dt)
        if not (0.0 < dt <= 0.25):
            return False

        delta = dt * float(self.speed) * self._direction
        t = self.time + delta

        if self.loop == "loop":
            self.time = float(t % self.duration) if self.duration > 0 else 0.0
        elif self.loop == "pingpong":
            if t > self.duration:
                self.time = max(0.0, 2.0 * self.duration - t)
                self._direction = -1
            elif t < 0.0:
                self.time = min(self.duration, -t)
                self._direction = 1
            else:
                self.time = t
        else:  # "once"
            self.time = float(np.clip(t, 0.0, self.duration))
            if self.time >= self.duration or self.time <= 0.0:
                self.playing = False
        return True

    # -- tracks ----------------------------------------------------------------

    def track(self, target: Any, prop: str, *, create: bool = True) -> Optional[Track]:
        """The track for ``(target, prop)``, created on demand unless ``create=False``."""
        for existing in self.tracks:
            if existing.target == target and existing.prop == str(prop):
                return existing
        if not create:
            return None
        created = Track(target=target, prop=str(prop))
        self.tracks.append(created)
        return created

    def add_track(self, track: Track) -> Track:
        """Register ``track``, replacing any existing one with the same identity."""
        for i, existing in enumerate(self.tracks):
            if existing.key() == track.key():
                self.tracks[i] = track
                return track
        self.tracks.append(track)
        return track

    def remove_track(self, target: Any, prop: str) -> bool:
        for i, existing in enumerate(self.tracks):
            if existing.target == target and existing.prop == str(prop):
                del self.tracks[i]
                return True
        return False

    def tracks_for(self, target: Any) -> List[Track]:
        """Every track animating ``target``."""
        return [t for t in self.tracks if t.target == target]

    def discard_target(self, target: Any) -> int:
        """Drop every track for ``target``. Call when a layer is deleted."""
        before = len(self.tracks)
        self.tracks = [t for t in self.tracks if t.target != target]
        return before - len(self.tracks)

    def key(
        self,
        target: Any,
        prop: str,
        value: Any,
        *,
        time: Optional[float] = None,
        easing_name: str = "smooth",
    ) -> Keyframe:
        """Key ``value`` at ``time`` (the playhead by default). The one-liner for the GUI."""
        track = self.track(target, prop)
        assert track is not None  # create=True
        return track.add(self.time if time is None else float(time), value, easing_name)

    def is_empty(self) -> bool:
        return not any(not t.is_empty() for t in self.tracks)

    # -- evaluation ------------------------------------------------------------

    def evaluate(self, time: Optional[float] = None) -> Dict[Tuple[Any, str], Any]:
        """Every animated value at ``time`` (the playhead by default).

        Returns ``{(target, prop): value}``. Empty and muted tracks contribute nothing, so
        the caller can apply the whole dict blindly — an entry exists only when something
        actually asked for that property to have a value now.
        """
        t = self.time if time is None else float(time)
        out: Dict[Tuple[Any, str], Any] = {}
        for track in self.tracks:
            if not track.enabled or track.is_empty():
                continue
            out[track.key()] = track.value_at(t)
        return out

    def bake(self) -> List[Dict[Tuple[Any, str], Any]]:
        """:meth:`evaluate` at every frame — the whole animation as data, for export."""
        return [self.evaluate(t) for t in self.frame_times()]

    # -- scenes ----------------------------------------------------------------

    def add_scene(self, name: str, start: float, end: float, description: str = "") -> Scene:
        """Append a scene, keeping the list sorted by start time."""
        scene = Scene(name=name, start=start, end=end, description=description)
        self.scenes.append(scene)
        self.scenes.sort(key=lambda s: s.start)
        self.fit_duration()
        return scene

    def remove_scene(self, name: str) -> bool:
        for i, scene in enumerate(self.scenes):
            if scene.name == str(name):
                del self.scenes[i]
                return True
        return False

    def scene_at(self, time: Optional[float] = None) -> Optional[Scene]:
        """The scene containing ``time``, or None between scenes."""
        t = self.time if time is None else float(time)
        for scene in self.scenes:
            if scene.contains(t):
                return scene
        # The last scene owns its own end instant, so the final frame is not "nowhere".
        if self.scenes and abs(t - self.scenes[-1].end) < 1e-9:
            return self.scenes[-1]
        return None

    def scene_index(self, time: Optional[float] = None) -> int:
        """Index of the scene containing ``time``, or -1."""
        scene = self.scene_at(time)
        return self.scenes.index(scene) if scene is not None else -1

    def go_to_scene(self, index: int) -> Optional[Scene]:
        """Put the playhead at the start of scene ``index``. Returns it, or None."""
        if not self.scenes:
            return None
        index = int(np.clip(int(index), 0, len(self.scenes) - 1))
        scene = self.scenes[index]
        self.seek(scene.start)
        return scene

    def next_scene(self) -> Optional[Scene]:
        """Advance to the next scene — the presentation "click to continue"."""
        current = self.scene_index()
        return self.go_to_scene(0 if current < 0 else current + 1)

    def previous_scene(self) -> Optional[Scene]:
        current = self.scene_index()
        return self.go_to_scene(max(current - 1, 0) if current >= 0 else 0)

    def auto_scenes(self, count: int = 4, prefix: str = "Scene") -> List[Scene]:
        """Split the timeline into ``count`` equal scenes, replacing any existing ones.

        The quickest way to get a presentation out of an animation that was authored as
        one continuous run.
        """
        count = max(1, int(count))
        self.scenes = []
        edges = np.linspace(0.0, self.duration, count + 1)
        for i in range(count):
            self.scenes.append(
                Scene(name=f"{prefix} {i + 1}", start=float(edges[i]), end=float(edges[i + 1]))
            )
        return self.scenes

    # -- misc ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<Timeline {self.duration:g}s @{self.fps:g}fps "
            f"{len(self.tracks)} tracks, {len(self.scenes)} scenes, t={self.time:.3f}>"
        )
