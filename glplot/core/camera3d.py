"""The 3D camera: a scene's point of view, stored as plain data.

Before this module the 3D "camera" was three floats (``elev``, ``azim``, ``fov``) copied
into every 3D layer's ``metadata["camera"]`` dict, and the view matrix was rebuilt from
them inside :mod:`glplot.renderers.geometry3d`. That was enough to orbit a point cloud and
nothing else: there was no target to pan towards, no roll, no orthographic projection, and
no way to say "these three axes should read as a cube even though z spans a million and x
spans one".

:class:`Camera3D` is that state promoted to a real object, owned by
:class:`~glplot.core.panel.Panel` alongside the 2D :class:`~glplot.core.legacy.CameraState`.
It is pure numpy + stdlib: no GL, no imgui, no engine import, so it is testable headless
and safe to construct anywhere.

Coordinate conventions
----------------------
``up_axis="z"`` (the default, and matplotlib's) puts the vertical along **+z**, with
``azim`` measured in the xy plane from +x towards +y and ``elev`` measured up from that
plane. The eye therefore sits at::

    eye = centre + distance * (cos(elev)cos(azim), cos(elev)sin(azim), sin(elev))

which is exactly the expression the old renderer inlined, so a camera left at its defaults
reproduces the previous picture matrix-for-matrix. ``up_axis="y"`` swaps to the graphics
convention (+y up, azim in the xz plane) for users coming from game/DCC tools.

**The pole.** At ``elev = ±90`` the view direction is parallel to the up vector and
``look_at`` degenerates into a matrix of NaNs — the old code had no guard, so a "top view"
was unreachable. :meth:`basis` detects the near-parallel case and derives the up vector
from ``azim`` instead, which makes the pole an ordinary orientation and lets the standard
Top/Bottom presets exist at all.

Framing versus panning
----------------------
The camera never stores an absolute target. It stores :attr:`pan` — a world-space offset
*added to whatever centre the scene currently has*. That indirection is what keeps a plot
auto-centred: replacing a layer's data moves the scene's bounds, the centre follows, and
the view stays framed without anyone recomputing a target. ``pan = (0, 0, 0)`` is always
"look at the middle of the data", which is what :meth:`reset` restores.

``distance = None`` means the same thing on the radial axis: "whatever framing fits the
data". It resolves through :meth:`resolve_distance` at matrix-build time, so a camera that
has never been dollied re-frames itself when the data changes, and one that has been
dollied keeps the distance the user chose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, MutableMapping, Optional, Tuple

import numpy as np

Bounds3D = Tuple[float, float, float, float, float, float]
Vec3 = Tuple[float, float, float]

#: ``metadata["artist"]`` values of the layers GLPlot generates to *decorate* a 3D scene
#: rather than to show data: the bounding box, the floor, the grid walls and the tick
#: marks. They are ordinary :class:`~glplot.core.layers.Layer3D` objects so the renderer
#: needs no special case, which means every consumer that reasons about "the user's data"
#: has to exclude them — the autoscale, the legend, the Scene tree, the HUD counter and
#: the matplotlib export bridge all do.
#:
#: Canonical here rather than on the engine because half those consumers cannot import the
#: engine without a cycle. Two of them used to carry their own ``("axis3d", "floor3d")``
#: literal and both silently went stale the moment a third artist was added.
SYSTEM_3D_ARTISTS = frozenset({"axis3d", "floor3d", "grid3d", "ticks3d"})

#: Multiple of the scene's bounding radius used as the *starting guess* for the viewport
#: fit, and as the fallback when there is no viewport to fit against. 2.9 is the literal
#: the pre-camera renderer used, kept so a camera with no aspect to work from reproduces
#: the historical framing exactly.
DEFAULT_DISTANCE_FACTOR = 2.9

#: Fraction of the viewport's half-extent the data box is fitted to by
#: :meth:`Camera3D.fit_distance`. Below 1 leaves room for the tick labels and axis titles,
#: which are drawn *outside* the box and would otherwise be clipped at the frame edge.
FIT_FILL = 0.86

#: Hard limits on the dolly, matching the engine's historical ``np.clip(dist, 1e-3, 1e6)``.
MIN_DISTANCE = 1e-3
MAX_DISTANCE = 1e6

#: Elevation limits. Exactly ±90 is legal (:meth:`basis` handles the pole); beyond it the
#: view would flip through the pole and the azimuth would read backwards, which no plotting
#: tool wants from a drag gesture.
MIN_ELEV = -90.0
MAX_ELEV = 90.0

#: Field-of-view limits. Below ~5° perspective is indistinguishable from orthographic and
#: the near/far spread collapses; above ~120° the distortion makes axes unreadable.
MIN_FOV = 5.0
MAX_FOV = 120.0

#: The two projection modes, as accepted by :meth:`Camera3D.set_projection`.
PROJECTIONS: Tuple[str, ...] = ("perspective", "orthographic")

#: Aliases so ``"persp"`` / ``"ortho"`` (matplotlib's ``set_proj_type`` spelling) work.
_PROJECTION_ALIASES: Dict[str, str] = {
    "persp": "perspective",
    "perspective": "perspective",
    "p": "perspective",
    "ortho": "orthographic",
    "orthographic": "orthographic",
    "o": "orthographic",
}

#: Named camera orientations as ``(elev, azim, roll)`` in degrees, in menu order.
#:
#: The six axis-aligned views are what an engineering drawing calls its projections, and
#: the four isometrics are the corners a data plot is normally read from. ``"iso"`` is the
#: default orientation, so "Home" and "Isometric" agree.
STANDARD_VIEWS: Dict[str, Tuple[float, float, float]] = {
    "iso": (28.0, -45.0, 0.0),
    "top": (90.0, -90.0, 0.0),
    "bottom": (-90.0, -90.0, 0.0),
    "front": (0.0, -90.0, 0.0),
    "back": (0.0, 90.0, 0.0),
    "left": (0.0, 180.0, 0.0),
    "right": (0.0, 0.0, 0.0),
    "iso_front_left": (28.0, -135.0, 0.0),
    "iso_front_right": (28.0, -45.0, 0.0),
    "iso_back_left": (28.0, 135.0, 0.0),
    "iso_back_right": (28.0, 45.0, 0.0),
    "corner": (35.264, -45.0, 0.0),
}

#: Human labels for :data:`STANDARD_VIEWS`, for menus and the command palette.
STANDARD_VIEW_LABELS: Dict[str, str] = {
    "iso": "Isometric",
    "top": "Top (+Z)",
    "bottom": "Bottom (−Z)",
    "front": "Front (−Y)",
    "back": "Back (+Y)",
    "left": "Left (−X)",
    "right": "Right (+X)",
    "iso_front_left": "Iso front-left",
    "iso_front_right": "Iso front-right",
    "iso_back_left": "Iso back-left",
    "iso_back_right": "Iso back-right",
    "corner": "True isometric",
}


# ----------------------------------------------------------------------------------
# Matrix primitives
# ----------------------------------------------------------------------------------


def normalize(v: np.ndarray) -> np.ndarray:
    """Unit-length ``v``, or ``v`` unchanged when it is too short to have a direction."""
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.asarray(v, dtype=np.float32)
    return np.asarray(v, dtype=np.float32) / n


def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Row-major OpenGL perspective matrix (vertical FOV in degrees).

    The field of view is clamped to :data:`MIN_FOV`/:data:`MAX_FOV` here rather than
    trusted. A zero fov gives ``tan(0) == 0`` and a ``ZeroDivisionError`` on the very next
    line, and because this matrix is rebuilt every frame that does not merely fail once —
    it makes the figure permanently unusable. It is reachable from the public API
    (matplotlib spells an orthographic camera ``set_proj_type("ortho", focal_length=inf)``,
    and a naive fov-from-focal-length conversion sends that to 0), so the guard belongs at
    the bottom where nothing can route around it.
    """
    fovy_deg = float(np.clip(float(fovy_deg), MIN_FOV, MAX_FOV))
    f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    out = np.zeros((4, 4), dtype=np.float32)
    out[0, 0] = f / max(aspect, 1e-6)
    out[1, 1] = f
    out[2, 2] = (far + near) / (near - far)
    out[2, 3] = (2.0 * far * near) / (near - far)
    out[3, 2] = -1.0
    return out


def orthographic(
    left: float, right: float, bottom: float, top: float, near: float, far: float
) -> np.ndarray:
    """Row-major OpenGL orthographic matrix.

    ``near`` may be negative — an orthographic frustum is a box, not a cone, so pulling the
    near plane behind the eye is legal and is how :meth:`Camera3D.projection_matrix` keeps
    geometry from clipping when the user dollies inside their own data.
    """
    rl = max(right - left, 1e-9)
    tb = max(top - bottom, 1e-9)
    fn = far - near
    if abs(fn) < 1e-9:
        fn = 1e-9
    out = np.eye(4, dtype=np.float32)
    out[0, 0] = 2.0 / rl
    out[1, 1] = 2.0 / tb
    out[2, 2] = -2.0 / fn
    out[0, 3] = -(right + left) / rl
    out[1, 3] = -(top + bottom) / tb
    out[2, 3] = -(far + near) / fn
    return out


def look_at(eye: np.ndarray, centre: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Row-major OpenGL view matrix looking from ``eye`` at ``centre``."""
    eye = np.asarray(eye, dtype=np.float32)
    centre = np.asarray(centre, dtype=np.float32)
    f = normalize(centre - eye)
    s = normalize(np.cross(f, np.asarray(up, dtype=np.float32)))
    u = np.cross(s, f)
    out = np.eye(4, dtype=np.float32)
    out[0, :3] = s
    out[1, :3] = u
    out[2, :3] = -f
    out[0, 3] = -float(np.dot(s, eye))
    out[1, 3] = -float(np.dot(u, eye))
    out[2, 3] = float(np.dot(f, eye))
    return out


def bounds_centre_radius(bounds: Optional[Bounds3D]) -> Tuple[np.ndarray, float]:
    """Split ``(xmin, xmax, ymin, ymax, zmin, zmax)`` into a centre and a bounding radius.

    A None or degenerate box yields the origin and a radius of 1, so a camera can always
    build a finite matrix even against an empty scene.
    """
    if bounds is None:
        return np.zeros(3, dtype=np.float32), 1.0
    mins = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float32)
    maxs = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float32)
    if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
        return np.zeros(3, dtype=np.float32), 1.0
    centre = 0.5 * (mins + maxs)
    radius = max(float(np.linalg.norm(maxs - mins)) * 0.5, 1e-3)
    return centre, radius


# ----------------------------------------------------------------------------------
# The camera
# ----------------------------------------------------------------------------------


@dataclass
class Camera3D:
    """Where the 3D scene is viewed from, and how it is projected.

    Attributes
    ----------
    elev, azim, roll
        Orientation in degrees. ``elev`` rises out of the horizontal plane, ``azim``
        sweeps around the up axis, ``roll`` spins about the line of sight. Clamped by
        :meth:`orbit`, not by assignment — a direct write is trusted, so deserialising a
        stored view never silently changes it.
    distance
        Eye-to-target distance in world units, or None for "fit the data"
        (:data:`DEFAULT_DISTANCE_FACTOR` times the scene radius).
    fov
        Vertical field of view in degrees. Also governs the orthographic box height, so
        toggling projection preserves the framing at the target plane rather than jumping.
    projection
        ``"perspective"`` or ``"orthographic"``.
    up_axis
        ``"z"`` (matplotlib/engineering) or ``"y"`` (graphics).
    pan
        World-space offset added to the scene centre. See the module docstring for why
        this is an offset and not an absolute target.
    box_aspect
        ``(ax, ay, az)`` relative axis lengths, or None to draw in raw data units. None is
        the default because it is what GLPlot has always done and what every existing
        example expects; ``(1, 1, 1)`` gives matplotlib's cube-normalised look, in which a
        z that spans a million and an x that spans one read as the same length.
    auto_spin
        Degrees of azimuth added per second by :meth:`advance_spin`. 0 disables it.
    """

    elev: float = 28.0
    azim: float = -45.0
    roll: float = 0.0
    distance: Optional[float] = None
    fov: float = 42.0
    projection: str = "perspective"
    up_axis: str = "z"
    pan: Vec3 = (0.0, 0.0, 0.0)
    box_aspect: Optional[Vec3] = None
    auto_spin: float = 0.0

    # -- basic mutators --------------------------------------------------------

    def copy(self) -> "Camera3D":
        """An independent copy — tuples are immutable, so a shallow dataclass copy is deep."""
        return Camera3D(
            elev=self.elev,
            azim=self.azim,
            roll=self.roll,
            distance=self.distance,
            fov=self.fov,
            projection=self.projection,
            up_axis=self.up_axis,
            pan=tuple(self.pan),  # type: ignore[arg-type]
            box_aspect=None if self.box_aspect is None else tuple(self.box_aspect),  # type: ignore[arg-type]
            auto_spin=self.auto_spin,
        )

    def reset(self) -> None:
        """Restore the default isometric framing, dropping every dolly, pan and roll."""
        elev, azim, roll = STANDARD_VIEWS["iso"]
        self.elev, self.azim, self.roll = elev, azim, roll
        self.distance = None
        self.fov = 42.0
        self.pan = (0.0, 0.0, 0.0)

    def set_projection(self, projection: str) -> None:
        """Set ``perspective``/``orthographic``, accepting matplotlib's ``persp``/``ortho``."""
        key = _PROJECTION_ALIASES.get(str(projection).strip().lower())
        if key is None:
            raise ValueError(
                f"projection must be one of {', '.join(PROJECTIONS)} "
                f"(or 'persp'/'ortho'), got {projection!r}"
            )
        self.projection = key

    def set_up_axis(self, axis: str) -> None:
        """Set the vertical axis: ``"z"`` (default) or ``"y"``."""
        key = str(axis).strip().lower()
        if key not in ("z", "y"):
            raise ValueError(f"up_axis must be 'z' or 'y', got {axis!r}")
        self.up_axis = key

    def apply_view(self, name: str) -> None:
        """Snap the orientation to a :data:`STANDARD_VIEWS` preset, keeping pan and dolly.

        Deliberately leaves ``distance`` and ``pan`` alone: "look at this from the top" is
        a question about orientation, and re-framing at the same time would throw away a
        zoom the user set up on purpose. "Home" (:meth:`reset`) is the verb that does both.
        """
        preset = STANDARD_VIEWS.get(str(name).strip().lower())
        if preset is None:
            raise ValueError(f"unknown view {name!r}; expected one of {', '.join(STANDARD_VIEWS)}")
        self.elev, self.azim, self.roll = preset

    # -- gestures --------------------------------------------------------------

    def orbit(self, d_azim: float, d_elev: float) -> None:
        """Rotate around the target. Elevation clamps at the poles, azimuth wraps."""
        self.azim = (float(self.azim) + float(d_azim)) % 360.0
        if self.azim > 180.0:
            self.azim -= 360.0
        self.elev = float(np.clip(float(self.elev) + float(d_elev), MIN_ELEV, MAX_ELEV))

    def spin(self, d_roll: float) -> None:
        """Roll about the line of sight, wrapped to (-180, 180]."""
        roll = (float(self.roll) + float(d_roll)) % 360.0
        self.roll = roll - 360.0 if roll > 180.0 else roll

    def dolly(self, factor: float, radius: float = 1.0) -> None:
        """Multiply the eye distance by ``factor`` (>1 pulls back, <1 moves in).

        ``radius`` seeds the distance the first time a camera that has never been dollied
        is zoomed, so the gesture works before anything has framed the scene.
        """
        current = self.resolve_distance(radius)
        self.distance = float(np.clip(current * float(factor), MIN_DISTANCE, MAX_DISTANCE))

    def pan_pixels(
        self, dx_px: float, dy_px: float, viewport_height: float, radius: float = 1.0
    ) -> None:
        """Slide the target by a screen-space drag, in pixels.

        The conversion is exact at the target plane for both projections: a perspective
        frustum and the orthographic box :meth:`projection_matrix` derives from it are the
        same size there, which is the whole reason the ortho height is defined from
        ``fov`` and ``distance`` rather than being an independent knob.
        """
        height = max(float(viewport_height), 1.0)
        distance = self.resolve_distance(radius)
        world_per_px = 2.0 * distance * math.tan(math.radians(self.fov) * 0.5) / height
        right, up, _ = self.basis()
        # Screen y grows downward, so a downward drag must move the data down, i.e. the
        # target up. Both signs are inverted relative to the cursor: the gesture is
        # "grab the scene and push it", not "move the camera".
        offset = (-float(dx_px) * right + float(dy_px) * up) * world_per_px
        self.pan = (
            float(self.pan[0]) + float(offset[0]),
            float(self.pan[1]) + float(offset[1]),
            float(self.pan[2]) + float(offset[2]),
        )

    def advance_spin(self, dt: float) -> bool:
        """Apply :attr:`auto_spin` for ``dt`` seconds. True when the view actually moved."""
        if not self.auto_spin:
            return False
        self.orbit(float(self.auto_spin) * float(dt), 0.0)
        return True

    def frame(self, bounds: Optional[Bounds3D], *, margin: float = 1.0) -> None:
        """Re-frame on ``bounds``: clear the pan and set a distance that fits it."""
        self.pan = (0.0, 0.0, 0.0)
        if bounds is None:
            self.distance = None
            return
        _, radius = bounds_centre_radius(self.transform_bounds(bounds))
        self.distance = float(
            np.clip(radius * DEFAULT_DISTANCE_FACTOR * float(margin), MIN_DISTANCE, MAX_DISTANCE)
        )

    def corners(self, bounds: Optional[Bounds3D]) -> np.ndarray:
        """The eight corners of ``bounds`` after :meth:`model_matrix`, as ``(8, 3)``."""
        framed = self.transform_bounds(bounds)
        if framed is None:
            return np.zeros((0, 3), dtype=np.float64)
        xs = (framed[0], framed[1])
        ys = (framed[2], framed[3])
        zs = (framed[4], framed[5])
        return np.array([(x, y, z) for x in xs for y in ys for z in zs], dtype=np.float64)

    def fit_distance(
        self,
        bounds: Optional[Bounds3D],
        aspect: float,
        *,
        fill: float = FIT_FILL,
        iterations: int = 4,
    ) -> Optional[float]:
        """The eye distance at which ``bounds`` fills ``fill`` of the viewport.

        Fits the **projected box**, not the bounding sphere. That distinction is the whole
        point: a sphere fit is orientation-independent and therefore has to assume the
        worst case, so a long thin dataset — which is most real data — was framed as if it
        were a ball of its own diagonal and ended up occupying about a third of the canvas
        with white space all round. Fitting what is actually on screen recovers that space,
        and it re-fits as the camera turns, which is what makes a plot look composed rather
        than merely centred.

        Solved by iteration rather than in closed form. Under perspective the projected
        extent is not exactly proportional to ``1/distance`` — the corners are at different
        depths — so a single division overshoots on a box seen end-on. Four rounds converge
        to well under a percent, and each is one 8x4 matrix multiply.

        Returns None when there is nothing to frame.
        """
        framed = self.transform_bounds(bounds)
        if framed is None:
            return None
        _, radius = bounds_centre_radius(framed)
        aspect = max(float(aspect), 1e-6)
        fill = float(np.clip(fill, 0.05, 1.0))
        corners = np.column_stack([self.corners(bounds), np.ones(8)])

        distance = max(radius * DEFAULT_DISTANCE_FACTOR, MIN_DISTANCE)
        saved = self.distance
        try:
            for _ in range(max(1, int(iterations))):
                self.distance = distance
                clip = corners @ np.asarray(self.mvp(aspect, bounds), dtype=np.float64).T
                w = clip[:, 3]
                # A corner behind the eye cannot be measured; back off hard and retry.
                if np.any(w <= 1e-9):
                    distance = min(distance * 2.0, MAX_DISTANCE)
                    continue
                ndc = clip[:, :2] / w[:, None]
                extent = float(np.max(np.abs(ndc)))
                if extent <= 1e-9:
                    break
                distance = float(np.clip(distance * extent / fill, MIN_DISTANCE, MAX_DISTANCE))
        finally:
            self.distance = saved
        return distance

    def centre_offset(
        self, bounds: Optional[Bounds3D], aspect: float
    ) -> Tuple[float, float, float]:
        """The world-space pan that puts ``bounds``' projected centre in the middle.

        The camera aims at the box's *geometric* centre, which is not where the box lands
        on screen: under perspective the near corners project further from the axis than
        the far ones, so an off-axis box drifts towards one side of the frame. Measuring
        the projected extent and correcting for it is the difference between "centred in
        world space" and "centred in the picture", and only the second one is visible.

        The conversion is exact at the target plane, where the frustum's half-height is
        ``distance * tan(fov/2)`` — the same relation :meth:`pan_pixels` uses, so a
        programmatic re-centre and a mouse pan move the target in the same units.
        """
        framed = self.transform_bounds(bounds)
        if framed is None:
            return (0.0, 0.0, 0.0)
        aspect = max(float(aspect), 1e-6)
        corners = np.column_stack([self.corners(bounds), np.ones(8)])
        clip = corners @ np.asarray(self.mvp(aspect, bounds), dtype=np.float64).T
        w = clip[:, 3]
        if np.any(w <= 1e-9):
            return tuple(self.pan)  # type: ignore[return-value]
        ndc = clip[:, :2] / w[:, None]
        # The centre of the projected *extent*, not the mean of the corners: the extent is
        # what has to sit inside the frame.
        ndc_centre = 0.5 * (ndc.min(axis=0) + ndc.max(axis=0))

        _, radius = bounds_centre_radius(framed)
        distance = self.resolve_distance(radius)
        half_h = distance * math.tan(math.radians(self.fov) * 0.5)
        right, up, _ = self.basis()
        offset = right * float(ndc_centre[0]) * half_h * aspect + up * float(ndc_centre[1]) * half_h
        return (
            float(self.pan[0]) + float(offset[0]),
            float(self.pan[1]) + float(offset[1]),
            float(self.pan[2]) + float(offset[2]),
        )

    def frame_viewport(
        self, bounds: Optional[Bounds3D], aspect: float, *, fill: float = FIT_FILL
    ) -> None:
        """Dolly and pan so ``bounds`` fills and is centred in the viewport.

        Distance and centring are solved together, twice: re-centring changes which
        corners are furthest from the axis, which changes the distance that fits them.
        Two rounds settle it to well under a pixel.
        """
        self.pan = (0.0, 0.0, 0.0)
        if bounds is None:
            self.distance = None
            return
        for _ in range(2):
            self.distance = self.fit_distance(bounds, aspect, fill=fill)
            self.pan = self.centre_offset(bounds, aspect)

    # -- derived geometry ------------------------------------------------------

    def resolve_distance(self, radius: float) -> float:
        """The eye distance actually used: :attr:`distance`, or a fit to ``radius``."""
        if self.distance is None:
            return max(float(radius) * DEFAULT_DISTANCE_FACTOR, MIN_DISTANCE)
        return max(float(self.distance), MIN_DISTANCE)

    def direction(self) -> np.ndarray:
        """Unit vector from the target towards the eye."""
        el = math.radians(float(self.elev))
        az = math.radians(float(self.azim))
        if self.up_axis == "y":
            return np.array(
                [math.cos(el) * math.cos(az), math.sin(el), math.cos(el) * math.sin(az)],
                dtype=np.float32,
            )
        return np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
            dtype=np.float32,
        )

    def world_up(self) -> np.ndarray:
        """The world's vertical axis for this camera's convention."""
        if self.up_axis == "y":
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The camera's ``(right, up, forward)`` unit vectors in world space.

        ``forward`` points *from the eye towards the target* (so it is the negative of
        :meth:`direction`). At the poles the world up vector is parallel to the view and
        cannot define a right vector; there the horizontal reference is derived from
        ``azim`` instead, which is what makes a top-down view a usable orientation rather
        than a matrix of NaNs.
        """
        forward = -self.direction()
        up_ref = self.world_up()
        if abs(float(np.dot(forward, up_ref))) > 0.9999:
            # At the pole: pick the up vector that keeps the azimuth meaning what it means
            # everywhere else, so spinning azim at elev=90 rotates the plan view in place.
            az = math.radians(float(self.azim))
            if self.up_axis == "y":
                up_ref = np.array([-math.cos(az), 0.0, -math.sin(az)], dtype=np.float32)
            else:
                up_ref = np.array([-math.cos(az), -math.sin(az), 0.0], dtype=np.float32)
            # Looking from below, the same reference would mirror the view; flip it back.
            if float(self.elev) < 0.0:
                up_ref = -up_ref

        right = normalize(np.cross(forward, up_ref))
        up = normalize(np.cross(right, forward))

        if self.roll:
            # Rodrigues about the view axis. Rolling the basis rather than the up vector
            # keeps right/up orthonormal by construction at any angle.
            theta = math.radians(float(self.roll))
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            right_r = right * cos_t + np.cross(forward, right) * sin_t
            up_r = up * cos_t + np.cross(forward, up) * sin_t
            right, up = normalize(right_r), normalize(up_r)
        return right, up, forward

    # -- box aspect ------------------------------------------------------------

    def scale_factors(self, bounds: Optional[Bounds3D]) -> np.ndarray:
        """Per-axis world scale implied by :attr:`box_aspect`, as a length-3 array.

        ``box_aspect=None`` gives ``(1, 1, 1)``: raw data units, GLPlot's historical
        behaviour. Otherwise each axis is scaled so the box's extents end up proportional
        to ``box_aspect``, and the whole thing is renormalised by the geometric mean so the
        scene keeps roughly its original size — without that, normalising a z that spans
        1e6 would shrink the model by six orders of magnitude and every distance, near
        plane and point size tuned against the old scale would be wrong.
        """
        if self.box_aspect is None or bounds is None:
            return np.ones(3, dtype=np.float32)
        extents = np.array(
            [
                max(float(bounds[1]) - float(bounds[0]), 1e-12),
                max(float(bounds[3]) - float(bounds[2]), 1e-12),
                max(float(bounds[5]) - float(bounds[4]), 1e-12),
            ],
            dtype=np.float64,
        )
        aspect = np.array([max(float(a), 1e-9) for a in self.box_aspect], dtype=np.float64)
        scale = aspect / extents
        # Geometric mean of 1 keeps the overall size; a plain max/sum normalisation would
        # bias towards whichever axis happened to be largest.
        gm = float(np.exp(np.mean(np.log(scale))))
        if gm > 0.0 and np.isfinite(gm):
            scale = scale / gm
        return np.asarray(scale, dtype=np.float32)

    def model_matrix(self, bounds: Optional[Bounds3D]) -> np.ndarray:
        """The 4x4 that applies :attr:`box_aspect`, scaling about the box centre.

        Identity when ``box_aspect`` is None, so the raw-units path costs one matrix
        multiply by the identity and nothing else changes.
        """
        scale = self.scale_factors(bounds)
        out = np.eye(4, dtype=np.float32)
        if np.allclose(scale, 1.0):
            return out
        centre, _ = bounds_centre_radius(bounds)
        out[0, 0], out[1, 1], out[2, 2] = float(scale[0]), float(scale[1]), float(scale[2])
        out[:3, 3] = centre - scale * centre
        return out

    def transform_bounds(self, bounds: Optional[Bounds3D]) -> Optional[Bounds3D]:
        """``bounds`` after :meth:`model_matrix` — what the camera actually frames."""
        if bounds is None:
            return None
        scale = self.scale_factors(bounds)
        if np.allclose(scale, 1.0):
            return bounds
        centre, _ = bounds_centre_radius(bounds)
        mins = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float64)
        maxs = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float64)
        mins = centre + scale * (mins - centre)
        maxs = centre + scale * (maxs - centre)
        return (
            float(mins[0]),
            float(maxs[0]),
            float(mins[1]),
            float(maxs[1]),
            float(mins[2]),
            float(maxs[2]),
        )

    # -- matrices --------------------------------------------------------------

    def eye(self, centre: np.ndarray, radius: float) -> np.ndarray:
        """Eye position for a scene centred at ``centre`` with bounding ``radius``."""
        target = np.asarray(centre, dtype=np.float32) + np.asarray(self.pan, dtype=np.float32)
        return target + self.resolve_distance(radius) * self.direction()

    def view_matrix(self, centre: np.ndarray, radius: float) -> np.ndarray:
        """The view matrix for a scene centred at ``centre`` with bounding ``radius``."""
        target = np.asarray(centre, dtype=np.float32) + np.asarray(self.pan, dtype=np.float32)
        _, up, _ = self.basis()
        return look_at(self.eye(centre, radius), target, up)

    def clip_planes(self, radius: float) -> Tuple[float, float]:
        """``(near, far)`` that enclose a ``radius``-sized scene at the current distance.

        Perspective keeps the historical ``near = 0.02 * radius`` — depth precision is
        scarce in a perspective frustum and that value was tuned against it — but the far
        plane now follows the dolly, so pulling back no longer clips the far half of the
        data (it used to be pinned at ``12 * radius``). Orthographic depth is linear, so it
        can afford a symmetric box around the target and never clip at all, which is what
        lets the user dolly inside their own point cloud.
        """
        radius = max(float(radius), 1e-6)
        distance = self.resolve_distance(radius)
        if self.projection == "orthographic":
            return distance - radius * 4.0, distance + radius * 4.0
        near = max(radius * 0.02, 1e-4)
        far = max(distance + radius * 4.0, radius * 12.0, near + 1.0)
        return near, far

    def projection_matrix(self, aspect: float, radius: float) -> np.ndarray:
        """Projection matrix for a viewport of ``aspect`` and a ``radius``-sized scene."""
        near, far = self.clip_planes(radius)
        if self.projection == "orthographic":
            distance = self.resolve_distance(radius)
            half_h = max(distance * math.tan(math.radians(self.fov) * 0.5), 1e-6)
            half_w = half_h * max(float(aspect), 1e-6)
            return orthographic(-half_w, half_w, -half_h, half_h, near, far)
        return perspective(self.fov, aspect, near, far)

    def mvp(self, aspect: float, bounds: Optional[Bounds3D]) -> np.ndarray:
        """The full model-view-projection matrix for a scene occupying ``bounds``."""
        model = self.model_matrix(bounds)
        framed = self.transform_bounds(bounds)
        centre, radius = bounds_centre_radius(framed)
        view = self.view_matrix(centre, radius)
        proj = self.projection_matrix(aspect, radius)
        return np.asarray(proj @ view @ model, dtype=np.float32)

    # -- serialisation ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """The camera as the plain dict the renderer reads out of ``metadata["camera"]``.

        Every key the old three-float dict carried is still here and still means the same
        thing, so a layer whose metadata predates this module keeps rendering identically
        and only misses the features it never had.
        """
        return {
            "elev": float(self.elev),
            "azim": float(self.azim),
            "roll": float(self.roll),
            "fov": float(self.fov),
            "distance": None if self.distance is None else float(self.distance),
            "projection": str(self.projection),
            "up_axis": str(self.up_axis),
            "pan": tuple(float(v) for v in self.pan),
            "box_aspect": (
                None if self.box_aspect is None else tuple(float(v) for v in self.box_aspect)
            ),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Camera3D":
        """Rebuild a camera from :meth:`to_dict`, tolerating missing and unknown keys.

        Unknown keys are ignored and missing ones fall back to the field defaults, so a
        view saved by an older or newer GLPlot loads instead of raising.
        """
        camera = cls()
        if not data:
            return camera
        for name in ("elev", "azim", "roll", "fov"):
            if data.get(name) is not None:
                try:
                    setattr(camera, name, float(data[name]))
                except (TypeError, ValueError):
                    pass
        if data.get("distance") is not None:
            try:
                camera.distance = float(data["distance"])
            except (TypeError, ValueError):
                pass
        if data.get("projection"):
            try:
                camera.set_projection(str(data["projection"]))
            except ValueError:
                pass
        if data.get("up_axis"):
            try:
                camera.set_up_axis(str(data["up_axis"]))
            except ValueError:
                pass
        pan = data.get("pan")
        if pan is not None:
            try:
                camera.pan = (float(pan[0]), float(pan[1]), float(pan[2]))
            except (TypeError, ValueError, IndexError):
                pass
        box = data.get("box_aspect")
        if box is not None:
            try:
                camera.box_aspect = (float(box[0]), float(box[1]), float(box[2]))
            except (TypeError, ValueError, IndexError):
                camera.box_aspect = None
        return camera


# ----------------------------------------------------------------------------------
# 3D axes decoration
# ----------------------------------------------------------------------------------


@dataclass
class Axes3DOptions:
    """What decoration is drawn around a 3D scene.

    Split from :class:`Camera3D` on purpose: these answer "what is drawn", the camera
    answers "from where". A user who toggles the floor is not moving the camera, and a
    saved view should be able to carry one without the other.
    """

    #: Master switch. False removes the box, floor, grid, ticks and labels in one go —
    #: this is the flag the historical ``view3d["show_axes"]`` mapped to.
    show_axes: bool = True
    #: The bounding-box wireframe.
    show_box: bool = True
    #: The shaded plane at z = zmin.
    show_floor: bool = True
    #: Grid lines ruled across the three back walls.
    show_grid: bool = True
    #: Tick marks along the three front edges.
    show_ticks: bool = True
    #: Numeric tick labels, drawn as screen-space text over the projected tick positions.
    show_tick_labels: bool = True
    #: The x/y/z axis titles.
    show_axis_labels: bool = True
    #: Requested ticks per axis, or **0 for automatic** — the default, and the same
    #: convention the 2D renderer uses for ``axis_tick_count_x``/``_y``.
    #:
    #: Automatic means the count is derived per axis from how long that axis is *on screen*
    #: (:func:`glplot.renderers.axes3d.adaptive_tick_counts`), so dollying in subdivides the
    #: axis and dollying out thins it, instead of stretching or crushing one fixed set of
    #: numbers. A positive value pins every axis to it.
    #:
    #: Either way the actual count comes from a nice-number rounding, so this is a target
    #: rather than a promise, matching the 2D axis renderer.
    tick_count: int = 0
    #: Axis titles, in x/y/z order.
    xlabel: str = ""
    ylabel: str = ""
    zlabel: str = ""
    #: Padding applied to the data bounds before the box is drawn, as a fraction of each
    #: axis' extent. 0.08 is the literal ``ensure_3d_axes`` has always used.
    pad: float = 0.08

    #: Explicit per-axis limits as ``(lo, hi)``, or None for "fit the data".
    #:
    #: An explicit limit does three things at once, and all three are needed for it to
    #: mean anything: the box is drawn at exactly that range (no padding — the user asked
    #: for *that* interval), the camera frames it, and the 3D fragment shader discards
    #: geometry outside it. Without the last one a limit would only move the walls and
    #: leave the data spilling through them.
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    zlim: Optional[Tuple[float, float]] = None

    def limits(self) -> Tuple[Optional[Tuple[float, float]], ...]:
        """The three limits as a tuple, in x/y/z order."""
        return (self.xlim, self.ylim, self.zlim)

    def has_limits(self) -> bool:
        """True when any axis has an explicit limit."""
        return any(limit is not None for limit in self.limits())

    def copy(self) -> "Axes3DOptions":
        return Axes3DOptions(**{f: getattr(self, f) for f in self.__dataclass_fields__})


# ----------------------------------------------------------------------------------
# Back-compatible dict facade
# ----------------------------------------------------------------------------------

#: Keys :class:`View3DProxy` forwards to the camera, and the attribute each maps to.
_CAMERA_KEYS: Dict[str, str] = {
    "elev": "elev",
    "azim": "azim",
    "roll": "roll",
    "fov": "fov",
    "distance": "distance",
    "projection": "projection",
    "up_axis": "up_axis",
    "pan": "pan",
    "box_aspect": "box_aspect",
    "auto_spin": "auto_spin",
}

#: Keys forwarded to the axes options.
_AXES_KEYS: Dict[str, str] = {
    "show_axes": "show_axes",
    "show_box": "show_box",
    "show_floor": "show_floor",
    "show_grid": "show_grid",
    "show_ticks": "show_ticks",
    "show_tick_labels": "show_tick_labels",
    "show_axis_labels": "show_axis_labels",
    "tick_count": "tick_count",
    "xlabel": "xlabel",
    "ylabel": "ylabel",
    "zlabel": "zlabel",
}


class View3DProxy(MutableMapping):
    """``engine.view3d`` as it always was — a dict — backed by a real camera.

    ``view3d`` was a plain dict of five keys read and written all over the engine, the HUD,
    the preview bridge and the Style panel. Rather than rewrite every one of those call
    sites (and break any user script that pokes ``fig.view3d["azim"]``), this presents the
    same mapping interface and forwards each key to :class:`Camera3D` or
    :class:`Axes3DOptions`. Writing ``view3d["elev"] = 40`` really does move the camera.

    Keys outside the two tables are stored locally rather than rejected, so an embedder
    stashing its own value in there keeps working.
    """

    __slots__ = ("_camera", "_axes", "_extra")

    def __init__(self, camera: Camera3D, axes: Axes3DOptions) -> None:
        self._camera = camera
        self._axes = axes
        self._extra: Dict[str, Any] = {}

    # The two halves, for code that would rather not go through the mapping.
    @property
    def camera(self) -> Camera3D:
        return self._camera

    @property
    def axes(self) -> Axes3DOptions:
        return self._axes

    def _rebind(self, camera: Camera3D, axes: Axes3DOptions) -> None:
        """Re-point the proxy at another panel's state (used by the engine's delegation)."""
        self._camera = camera
        self._axes = axes

    def __getitem__(self, key: str) -> Any:
        attr = _CAMERA_KEYS.get(key)
        if attr is not None:
            return getattr(self._camera, attr)
        attr = _AXES_KEYS.get(key)
        if attr is not None:
            return getattr(self._axes, attr)
        return self._extra[key]

    def __setitem__(self, key: str, value: Any) -> None:
        attr = _CAMERA_KEYS.get(key)
        if attr is not None:
            if attr == "projection" and value is not None:
                self._camera.set_projection(str(value))
            elif attr == "up_axis" and value is not None:
                self._camera.set_up_axis(str(value))
            elif attr == "distance":
                self._camera.distance = None if value is None else float(value)
            elif attr in ("pan", "box_aspect"):
                setattr(
                    self._camera, attr, None if value is None else tuple(float(v) for v in value)
                )
            else:
                setattr(self._camera, attr, float(value))
            return
        attr = _AXES_KEYS.get(key)
        if attr is not None:
            current = getattr(self._axes, attr)
            if isinstance(current, bool):
                setattr(self._axes, attr, bool(value))
            elif isinstance(current, int):
                setattr(self._axes, attr, int(value))
            elif isinstance(current, str):
                setattr(self._axes, attr, str(value))
            else:
                setattr(self._axes, attr, value)
            return
        self._extra[key] = value

    def __delitem__(self, key: str) -> None:
        if key in _CAMERA_KEYS or key in _AXES_KEYS:
            raise KeyError(f"{key!r} is a built-in view key and cannot be deleted")
        del self._extra[key]

    def __iter__(self) -> Iterator[str]:
        yield from _CAMERA_KEYS
        yield from _AXES_KEYS
        yield from self._extra

    def __len__(self) -> int:
        return len(_CAMERA_KEYS) + len(_AXES_KEYS) + len(self._extra)

    def __repr__(self) -> str:
        return f"View3DProxy({dict(self)!r})"
