"""A catalogue of 3D datasets, stored as data rather than as code.

Pure numpy and stdlib -- no imgui, no engine import (CONTRACT 5.1 rule 7). The 3D panels
are thin shells over this module, exactly as the Dynamics panel is a shell over
:mod:`glplot.gui.dynamics` and the Functions panel over :mod:`glplot.gui.expressions`.

Why a catalogue at all
----------------------
A 3D workstation with no data in it is unusable in a way a 2D one is not. In 2D the user
types ``sin(x)`` and gets a curve; the domain is one interval and the Functions panel owns
it. In 3D the interesting objects are *parametrised surfaces and space curves* -- a torus
is ``(R + r cos v) cos u`` in three coordinates at once -- and asking a user to type three
coupled expressions before they can see anything is a wall, not a feature. Sixty named
objects with sliders is the difference between "3D works" and "3D is usable".

The catalogue is chosen for *coverage of the cases that behave differently*, not for
length: a minimal surface that self-intersects (Enneper) next to one that does not
(catenoid), an immersion of the projective plane (Boy) next to a map of it with genuine
singularities (Roman), the two standard manifold-learning sets (Swiss roll, S-curve) next
to the two-moons classifier set, four crystal packings plus the aperiodic one. Anything
that can only be reached by moving another object's sliders is deliberately absent -- a
trefoil is ``torus_knot`` at ``p = 2, q = 3``, so it is not a separate entry.

Each :class:`Generator3D` is a literal: its parameters with slider bounds, the plot kind it
wants, and one pure function from ``(params, samples)`` to named columns. A generated
dataset is therefore an ordinary table -- it lands in the DataStore like a pasted CSV, can
be edited cell by cell, transformed, filtered and re-plotted. Nothing about it is special
once it exists, which is the point: the catalogue is a *source* of data, not a kind of it.

Shapes
------
* **Curves** return ``t, x, y, z`` -- one row per sample, in path order.
* **Surfaces** return ``u, v, x, y, z`` on a full rectangular ``n x n`` lattice, in the
  row-major order :func:`glplot.gui.layerops3d.reshape_to_grid` expects, so the
  ``surface3d`` and ``wireframe3d`` kinds accept them without a re-sort.
* **Clouds** return ``x, y, z`` plus a scalar worth colouring by -- a radius, a cluster
  label, or the manifold coordinate an embedding is supposed to recover.
* **Lattices** return ``x, y, z`` plus a scalar, like clouds, but from a deterministic
  site list rather than an RNG. They are a separate category because the sample count
  means something different: it sets the number of *unit cells*, so the row count is
  rounded to whole cells instead of honoured exactly.
* **Fields** return ``x, y, z, u, v, w`` -- position and vector -- for ``quiver3d``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

#: Hard ceiling on generated rows. Above this the cost is the *table*, not the plot: the
#: Data panel holds float64 columns and the spreadsheet has to be able to show them.
MAX_SAMPLES = 4_000_000

#: Smallest usable sample count.
MIN_SAMPLES = 4

#: Default sample count for a curve, and the target total for a surface (whose per-axis
#: resolution is the square root of this).
DEFAULT_SAMPLES = 20_000

#: The categories, in menu order, with their human labels.
CATEGORIES: Tuple[Tuple[str, str], ...] = (
    ("curve", "Space curves"),
    ("surface", "Surfaces"),
    ("cloud", "Point clouds"),
    ("lattice", "Lattices"),
    ("field", "Vector fields"),
)


class GeneratorError(ValueError):
    """Raised for an unknown generator or an out-of-range request.

    A :class:`ValueError` subclass so panel actions that already catch ``ValueError``
    catch this too, matching :class:`glplot.gui.dynamics.DynamicsError`.
    """


@dataclass(frozen=True)
class Param3D:
    """One generator parameter: its default and the slider bounds around it."""

    name: str
    default: float
    vmin: float
    vmax: float
    label: str = ""
    integer: bool = False

    def clamp(self, value: float) -> float:
        """``value`` forced into range (and to an integer where the parameter is one)."""
        out = float(np.clip(float(value), self.vmin, self.vmax))
        return float(round(out)) if self.integer else out


@dataclass(frozen=True)
class Generator3D:
    """One named 3D object: what it is called, what it takes, and how it is built.

    Attributes
    ----------
    key
        Stable machine identifier; the :data:`GENERATORS` dict key.
    label
        Human name, used for the dataset and the layer it produces.
    category
        One of :data:`CATEGORIES`' keys — drives the picker's grouping.
    icon
        An :data:`glplot.gui.icons.ICON_SHAPES` name. Advisory: an unknown name draws a
        placeholder rather than raising.
    description
        One line, shown as a tooltip.
    kind
        The :mod:`glplot.gui.layerops3d` plot kind this object wants by default. A surface
        generator asks for ``surface3d``, a curve for ``line3d``, and so on — so "generate"
        can be one click rather than a generate-then-choose-a-type sequence.
    columns
        The column names ``build`` returns, in order.
    plot_columns
        Which three of them are the geometry, as ``(x, y, z)``.
    color_column
        The column worth colour-mapping by default, or None for "use z".
    params
        Parameters with slider bounds.
    build
        ``(params, samples) -> {name: values}``. Pure: same inputs, same table.
    """

    key: str
    label: str
    category: str
    icon: str
    description: str
    kind: str
    columns: Tuple[str, ...]
    plot_columns: Tuple[str, str, str]
    build: Callable[[Mapping[str, float], int], Dict[str, np.ndarray]]
    params: Tuple[Param3D, ...] = ()
    color_column: Optional[str] = None

    def defaults(self) -> Dict[str, float]:
        """A fresh, mutable dict of this generator's default parameter values."""
        return {p.name: p.default for p in self.params}

    def resolve(self, params: Optional[Mapping[str, float]]) -> Dict[str, float]:
        """Merge ``params`` over the defaults, clamping each to its slider bounds.

        Unknown keys are ignored rather than rejected: a panel that keeps one parameter
        dict per generator and switches between them would otherwise raise every time the
        user changed object.
        """
        resolved = self.defaults()
        for p in self.params:
            if params and p.name in params:
                resolved[p.name] = p.clamp(params[p.name])
        return resolved

    def generate(
        self, params: Optional[Mapping[str, float]] = None, samples: int = DEFAULT_SAMPLES
    ) -> Dict[str, np.ndarray]:
        """Build the table. The one entry point; ``build`` is never called directly."""
        n = int(np.clip(int(samples), MIN_SAMPLES, MAX_SAMPLES))
        table = self.build(self.resolve(params), n)
        lengths = {len(v) for v in table.values()}
        if len(lengths) > 1:  # pragma: no cover - a generator bug, not a user error
            raise GeneratorError(
                f"generator {self.key!r} produced ragged columns: {sorted(lengths)}"
            )
        return table

    def grid_shape(self, samples: int = DEFAULT_SAMPLES) -> Optional[Tuple[int, int]]:
        """``(nu, nv)`` for a surface generator at ``samples``, or None for the others.

        A parametric surface's triangulation is only recoverable from its sampling shape
        (a sphere's ``(x, y)`` is not a lattice — see
        :func:`glplot.gui.layerops3d.resolve_grid`), so this is what the caller passes
        into the ``surface3d``/``wireframe3d`` options.
        """
        if self.category != "surface":
            return None
        n = surface_resolution(int(np.clip(int(samples), MIN_SAMPLES, MAX_SAMPLES)))
        return n, n

    def kind_options(self, samples: int = DEFAULT_SAMPLES) -> Dict[str, Any]:
        """The plot-kind options this generator's output needs, ready to pass along.

        Empty for everything except surfaces, which need their sampling shape, and vector
        fields, whose ``u``/``v``/``w`` come from columns the caller supplies.
        """
        shape = self.grid_shape(samples)
        if shape is None:
            return {}
        return {"nu": shape[0], "nv": shape[1]}


# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------


def surface_resolution(samples: int) -> int:
    """Per-axis sample count a surface generator uses for a total of about ``samples``.

    The single definition of the sampling shape. :func:`_grid_uv` builds the lattice from
    it and :meth:`Generator3D.grid_shape` reports it to the caller, so the ``nu``/``nv``
    handed to :func:`glplot.gui.layerops3d.resolve_grid` cannot disagree with the array
    that was actually generated — which would triangulate the surface inside out.
    """
    return max(3, int(math.sqrt(max(int(samples), MIN_SAMPLES))))


def _grid_uv(
    samples: int,
    u_range: Tuple[float, float],
    v_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """A rectangular ``(u, v)`` lattice with about ``samples`` points, row-major.

    Returned flat and with ``v`` varying slowest — the order
    :func:`glplot.gui.layerops3d.resolve_grid` reshapes with. Building it here rather than
    in each surface generator means every surface in the catalogue is griddable by
    construction, and a new one cannot get the ordering wrong.
    """
    n = surface_resolution(samples)
    u = np.linspace(u_range[0], u_range[1], n)
    v = np.linspace(v_range[0], v_range[1], n)
    uu, vv = np.meshgrid(u, v)  # (n, n), v along rows
    return uu.ravel(), vv.ravel()


def _rng(seed: float) -> np.random.Generator:
    """A seeded generator, so a cloud is reproducible from its parameters alone.

    Reproducibility is not a nicety here: the panel re-runs ``generate`` on every slider
    drag, and an unseeded cloud would reshuffle itself continuously while the user tried
    to tune its width.
    """
    return np.random.default_rng(int(seed) & 0x7FFFFFFF)


def _spow(values: np.ndarray, exponent: float) -> np.ndarray:
    """``sign(v) * |v| ** exponent`` — the signed power the superquadrics are defined with.

    A bare ``v ** e`` is not usable: for a negative ``v`` and a fractional ``e`` numpy
    returns ``nan`` (the real branch does not exist), which would punch holes through
    three of the four quadrants of every superellipsoid.
    """
    return np.sign(values) * np.abs(values) ** exponent


def _cell_side(samples: int, per_cell: int) -> int:
    """Unit cells per axis so that ``side**3 * per_cell`` is about ``samples``.

    Shared by every crystal lattice below so that "10000 points" means the same density of
    sites whether the cell carries one atom or eight.
    """
    return max(1, int(round((max(int(samples), MIN_SAMPLES) / float(per_cell)) ** (1.0 / 3.0))))


def _lattice_table(points: np.ndarray, jitter: float, spacing: float, seed: float) -> Dict:
    """The common tail of every lattice generator: jitter the sites and label them.

    ``r`` is the distance from the centroid rather than from the origin, so colouring by
    it reads as "how far out in the crystal" for any cell count.
    """
    if jitter:
        points = points + _rng(seed).normal(scale=jitter * spacing, size=points.shape)
    return {
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],
        "r": np.linalg.norm(points - points.mean(axis=0), axis=1),
    }


# ----------------------------------------------------------------------------------
# Curves
# ----------------------------------------------------------------------------------


def _helix(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    turns, radius, pitch = p["turns"], p["radius"], p["pitch"]
    t = np.linspace(0.0, turns * 2.0 * np.pi, n)
    return {"t": t, "x": radius * np.cos(t), "y": radius * np.sin(t), "z": pitch * t}


def _torus_knot(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    pp, qq, radius = p["p"], p["q"], p["radius"]
    t = np.linspace(0.0, 2.0 * np.pi, n)
    r = radius * (2.0 + np.cos(qq * t))
    return {"t": t, "x": r * np.cos(pp * t), "y": r * np.sin(pp * t), "z": -radius * np.sin(qq * t)}


def _lissajous3d(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    a, b, c, delta = p["a"], p["b"], p["c"], p["delta"]
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return {
        "t": t,
        "x": np.sin(a * t + delta),
        "y": np.sin(b * t),
        "z": np.sin(c * t),
    }


def _spherical_spiral(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    turns, radius = p["turns"], p["radius"]
    t = np.linspace(-np.pi / 2.0 + 1e-3, np.pi / 2.0 - 1e-3, n)
    return {
        "t": t,
        "x": radius * np.cos(t) * np.cos(turns * t),
        "y": radius * np.cos(t) * np.sin(turns * t),
        "z": radius * np.sin(t),
    }


def _conical_spiral(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    turns, height, radius = p["turns"], p["height"], p["radius"]
    t = np.linspace(0.0, 1.0, n)
    a = turns * 2.0 * np.pi * t
    return {"t": t, "x": radius * t * np.cos(a), "y": radius * t * np.sin(a), "z": height * t}


def _viviani(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    a = p["radius"]
    t = np.linspace(-2.0 * np.pi, 2.0 * np.pi, n)
    return {
        "t": t,
        "x": a * (1.0 + np.cos(t)),
        "y": a * np.sin(t),
        "z": 2.0 * a * np.sin(t / 2.0),
    }


def _figure_eight_knot(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """The 4_1 knot: the simplest knot that is *not* a torus knot.

    Worth having next to :func:`_torus_knot` precisely because it cannot be reached from
    it — no (p, q) makes a torus knot into a figure-eight — so the two together cover both
    halves of the small-knot table people actually draw.
    """
    scale = p["scale"]
    t = np.linspace(0.0, 2.0 * np.pi, n)
    ring = 2.0 + np.cos(2.0 * t)
    return {
        "t": t,
        "x": scale * ring * np.cos(3.0 * t),
        "y": scale * ring * np.sin(3.0 * t),
        "z": scale * np.sin(4.0 * t),
    }


def _granny_knot(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """The granny knot (the connected sum of two trefoils) as a Fourier series.

    The coefficients are the standard closed-form parametrisation; they are literals
    rather than derived because there is nothing to derive — the curve *is* those twelve
    numbers. ``scale`` divides by 100 so the default lands in the same box as every other
    curve here instead of at a radius of 150.
    """
    scale = p["scale"] / 100.0
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return {
        "t": t,
        "x": scale
        * (-22.0 * np.cos(t) - 128.0 * np.sin(t) - 44.0 * np.cos(3.0 * t) - 78.0 * np.sin(3.0 * t)),
        "y": scale
        * (
            -10.0 * np.cos(2.0 * t)
            - 27.0 * np.sin(2.0 * t)
            + 38.0 * np.cos(4.0 * t)
            + 46.0 * np.sin(4.0 * t)
        ),
        "z": scale * (70.0 * np.cos(3.0 * t) - 40.0 * np.sin(3.0 * t)),
    }


def _coiled_coil(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """A helix wound around a helix — a telephone cord, a coiled-coil protein, a solenoid.

    Built as a small circle of radius ``minor`` riding a large helix rather than by
    transporting a frame along it: the closed form is exact, cheap and cannot flip its
    normal, which a numerically transported frame can at an inflection point.
    """
    turns, coils = p["turns"], p["coils"]
    major, minor, pitch = p["radius"], p["minor"], p["pitch"]
    t = np.linspace(0.0, turns * 2.0 * np.pi, n)
    ring = major + minor * np.cos(coils * t)
    return {
        "t": t,
        "x": ring * np.cos(t),
        "y": ring * np.sin(t),
        "z": pitch * t + minor * np.sin(coils * t),
    }


def _rose3d(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """A rhodonea (rose) curve in the plane, lifted out of it by a second harmonic.

    ``t`` always sweeps a full ``2 pi``: an odd ``k`` retraces its own petals over the
    second half, which is harmless, whereas stopping at ``pi`` would silently draw half a
    flower for every even ``k``.
    """
    k, amp, radius = p["k"], p["amp"], p["radius"]
    t = np.linspace(0.0, 2.0 * np.pi, n)
    r = radius * np.cos(k * t)
    return {
        "t": t,
        "x": r * np.cos(t),
        "y": r * np.sin(t),
        "z": amp * np.sin(2.0 * k * t),
    }


def _twisted_cubic(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """``(t, t², t³)`` — the smallest non-planar algebraic curve.

    The standard example of a rational normal curve, and the cleanest object for checking
    that a 3D view is not quietly collapsing one axis: no two coordinates are related by a
    linear map anywhere along it.
    """
    extent, scale = p["extent"], p["scale"]
    t = np.linspace(-extent, extent, n)
    return {"t": t, "x": t, "y": t * t, "z": scale * t**3}


# ----------------------------------------------------------------------------------
# Surfaces
# ----------------------------------------------------------------------------------


def _sphere(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    radius = p["radius"]
    u, v = _grid_uv(n, (0.0, 2.0 * np.pi), (0.0, np.pi))
    return {
        "u": u,
        "v": v,
        "x": radius * np.sin(v) * np.cos(u),
        "y": radius * np.sin(v) * np.sin(u),
        "z": radius * np.cos(v),
    }


def _torus(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    big, small = p["R"], p["r"]
    u, v = _grid_uv(n, (0.0, 2.0 * np.pi), (0.0, 2.0 * np.pi))
    return {
        "u": u,
        "v": v,
        "x": (big + small * np.cos(v)) * np.cos(u),
        "y": (big + small * np.cos(v)) * np.sin(u),
        "z": small * np.sin(v),
    }


def _mobius(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    radius, width = p["radius"], p["width"]
    u, v = _grid_uv(n, (0.0, 2.0 * np.pi), (-width, width))
    return {
        "u": u,
        "v": v,
        "x": (radius + v * np.cos(u / 2.0)) * np.cos(u),
        "y": (radius + v * np.cos(u / 2.0)) * np.sin(u),
        "z": v * np.sin(u / 2.0),
    }


def _klein(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """The figure-8 immersion — the Klein bottle form that fits in a plotting box."""
    radius = p["radius"]
    u, v = _grid_uv(n, (0.0, 2.0 * np.pi), (0.0, 2.0 * np.pi))
    half = u / 2.0
    s = np.sin(v)
    c = np.sin(2.0 * v)
    common = radius + np.cos(half) * s - np.sin(half) * c
    return {
        "u": u,
        "v": v,
        "x": common * np.cos(u),
        "y": common * np.sin(u),
        "z": np.sin(half) * s + np.cos(half) * c,
    }


def _saddle(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    extent, scale = p["extent"], p["scale"]
    u, v = _grid_uv(n, (-extent, extent), (-extent, extent))
    return {"u": u, "v": v, "x": u, "y": v, "z": scale * (u * u - v * v)}


def _ripple(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    extent, freq, amp = p["extent"], p["freq"], p["amp"]
    u, v = _grid_uv(n, (-extent, extent), (-extent, extent))
    r = np.hypot(u, v)
    # sinc rather than a bare cosine: the decay keeps the outer rings from reading as a
    # flat corrugated sheet, which is the shape people actually picture as "ripple".
    return {"u": u, "v": v, "x": u, "y": v, "z": amp * np.sinc(freq * r / np.pi)}


def _gaussian_bump(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    extent, sigma, amp = p["extent"], p["sigma"], p["amp"]
    u, v = _grid_uv(n, (-extent, extent), (-extent, extent))
    return {
        "u": u,
        "v": v,
        "x": u,
        "y": v,
        "z": amp * np.exp(-(u * u + v * v) / (2.0 * max(sigma, 1e-6) ** 2)),
    }


def _cylinder(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    radius, height = p["radius"], p["height"]
    u, v = _grid_uv(n, (0.0, 2.0 * np.pi), (0.0, height))
    return {"u": u, "v": v, "x": radius * np.cos(u), "y": radius * np.sin(u), "z": v}


def _cone(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    radius, height = p["radius"], p["height"]
    u, v = _grid_uv(n, (0.0, 2.0 * np.pi), (0.0, 1.0))
    return {
        "u": u,
        "v": v,
        "x": radius * (1.0 - v) * np.cos(u),
        "y": radius * (1.0 - v) * np.sin(u),
        "z": height * v,
    }


def _hyperboloid(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    radius, height = p["radius"], p["height"]
    u, v = _grid_uv(n, (0.0, 2.0 * np.pi), (-height, height))
    scale = np.sqrt(1.0 + v * v)
    return {
        "u": u,
        "v": v,
        "x": radius * scale * np.cos(u),
        "y": radius * scale * np.sin(u),
        "z": v,
    }


def _shell(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """A logarithmic seashell — a spiral tube, the classic parametric-surface showpiece."""
    turns, radius, taper = p["turns"], p["radius"], p["taper"]
    u, v = _grid_uv(n, (0.0, turns * 2.0 * np.pi), (0.0, 2.0 * np.pi))
    growth = np.exp(taper * u)
    tube = radius * growth
    return {
        "u": u,
        "v": v,
        "x": growth * (1.0 + np.cos(v) * radius) * np.cos(u),
        "y": growth * (1.0 + np.cos(v) * radius) * np.sin(u),
        "z": growth * (np.sin(v) * radius) + 0.25 * u * tube,
    }


#: Where Dini's surface and Kuen's surface have their ``v`` domain cut. Both carry a
#: ``log(tan(v/2))`` term that runs to -inf at ``v = 0``; the surface is unbounded there,
#: so the only honest options are "clip the domain" or "emit inf". Clipping at 0.1 rad
#: costs about one screen pixel of the trumpet and keeps every column finite.
_LOG_TAN_CLIP = 0.1


def _enneper(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """Enneper's minimal surface: zero mean curvature, and self-intersecting from |u| > 1.

    The reference *non-embedded* minimal surface — the case that catches a renderer which
    assumes a surface never crosses itself.
    """
    extent, scale = p["extent"], p["scale"]
    u, v = _grid_uv(n, (-extent, extent), (-extent, extent))
    return {
        "u": u,
        "v": v,
        "x": scale * (u - u**3 / 3.0 + u * v * v),
        "y": scale * (v - v**3 / 3.0 + v * u * u),
        "z": scale * (u * u - v * v),
    }


def _catenoid(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """The catenoid: the only minimal surface of revolution, and a real soap film.

    ``height`` is capped well below the point where ``cosh`` overflows — the flare grows
    exponentially, so a tall thin catenoid is numerically fine but visually useless.
    """
    waist, height = p["waist"], p["height"]
    u, v = _grid_uv(n, (0.0, 2.0 * np.pi), (-height, height))
    c = max(float(waist), 1e-6)
    ring = c * np.cosh(v / c)
    return {"u": u, "v": v, "x": ring * np.cos(u), "y": ring * np.sin(u), "z": v}


def _helicoid(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """The helicoid: minimal, ruled, and the catenoid's associate — a spiral ramp.

    Every ``v`` line is straight, so its wireframe is a family of straight rules through
    the axis; that makes it the clearest check that a surface's grid lines are being drawn
    along the parameter directions and not along x/y.
    """
    turns, radius, pitch = p["turns"], p["radius"], p["pitch"]
    u, v = _grid_uv(n, (0.0, turns * 2.0 * np.pi), (-radius, radius))
    return {"u": u, "v": v, "x": v * np.cos(u), "y": v * np.sin(u), "z": pitch * u}


def _dini(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """Dini's surface: a pseudosphere twisted along its axis, constant negative curvature.

    The ``v`` domain is clipped at :data:`_LOG_TAN_CLIP` because the surface is genuinely
    unbounded as ``v -> 0``.
    """
    radius, twist, turns = p["radius"], p["twist"], p["turns"]
    u, v = _grid_uv(n, (0.0, turns * 2.0 * np.pi), (_LOG_TAN_CLIP, 2.0))
    return {
        "u": u,
        "v": v,
        "x": radius * np.cos(u) * np.sin(v),
        "y": radius * np.sin(u) * np.sin(v),
        "z": radius * (np.cos(v) + np.log(np.tan(v / 2.0))) + twist * u,
    }


def _roman(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """Steiner's Roman surface: the real projective plane mapped into 3-space.

    Written as ``r² (YZ, ZX, XY)`` on the unit sphere rather than as three expanded
    trigonometric polynomials, because that form is the definition — antipodal points of
    the sphere map to the same place, which is exactly what "the projective plane" means.
    """
    radius = p["radius"]
    u, v = _grid_uv(n, (0.0, 2.0 * np.pi), (0.0, np.pi))
    sx = np.sin(v) * np.cos(u)
    sy = np.sin(v) * np.sin(u)
    sz = np.cos(v)
    scale = radius * radius
    return {"u": u, "v": v, "x": scale * sy * sz, "y": scale * sz * sx, "z": scale * sx * sy}


def _boy(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """Boy's surface: an *immersion* of the projective plane — no pinch points.

    The companion to the Roman surface: same underlying space, but this one has no
    singular points, only self-intersections. The denominator is bounded below by
    ``2 - sqrt(2) ~ 0.586``, so no clamping is needed anywhere on the domain.
    """
    scale = p["scale"]
    u, v = _grid_uv(n, (0.0, np.pi), (0.0, np.pi))
    root2 = math.sqrt(2.0)
    denom = 2.0 - root2 * np.sin(3.0 * u) * np.sin(2.0 * v)
    cos2v = np.cos(v) ** 2
    sin2v = np.sin(2.0 * v)
    return {
        "u": u,
        "v": v,
        "x": scale * (root2 * np.cos(2.0 * u) * cos2v + np.cos(u) * sin2v) / denom,
        "y": scale * (root2 * np.sin(2.0 * u) * cos2v - np.sin(u) * sin2v) / denom,
        "z": scale * 3.0 * cos2v / denom,
    }


def _supertoroid(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """A superquadric torus: a torus whose tube and whose ring are both superellipses.

    ``e1``/``e2`` near 1 give the ordinary torus, near 0 a square-sectioned square ring,
    above 2 a pinched star. One slider pair covers a whole family of CAD/graphics
    primitives, which is why superquadrics are worth a catalogue entry at all.
    """
    big, small, e1, e2 = p["R"], p["r"], p["e1"], p["e2"]
    u, v = _grid_uv(n, (-np.pi, np.pi), (-np.pi, np.pi))
    ring = big + small * _spow(np.cos(v), e2)
    return {
        "u": u,
        "v": v,
        "x": ring * _spow(np.cos(u), e1),
        "y": ring * _spow(np.sin(u), e1),
        "z": small * _spow(np.sin(v), e2),
    }


def _superellipsoid(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """A superquadric ellipsoid: sphere at ``e = 1``, box as ``e -> 0``, octahedron at 2."""
    radius, e1, e2 = p["radius"], p["e1"], p["e2"]
    u, v = _grid_uv(n, (-np.pi, np.pi), (-np.pi / 2.0, np.pi / 2.0))
    band = _spow(np.cos(v), e1)
    return {
        "u": u,
        "v": v,
        "x": radius * band * _spow(np.cos(u), e2),
        "y": radius * band * _spow(np.sin(u), e2),
        "z": radius * _spow(np.sin(v), e1),
    }


def _kuen(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """Kuen's surface: constant negative curvature with a very un-obvious shape.

    A Bäcklund transform of the pseudosphere. Like Dini's it carries ``log(tan(v/2))``, so
    its ``v`` domain is clipped at :data:`_LOG_TAN_CLIP` at both ends.
    """
    extent, scale = p["extent"], p["scale"]
    u, v = _grid_uv(n, (-extent, extent), (_LOG_TAN_CLIP, np.pi - _LOG_TAN_CLIP))
    s = np.sin(v)
    denom = 1.0 + (u * s) ** 2
    return {
        "u": u,
        "v": v,
        "x": scale * 2.0 * (np.cos(u) + u * np.sin(u)) * s / denom,
        "y": scale * 2.0 * (np.sin(u) - u * np.cos(u)) * s / denom,
        "z": scale * (np.log(np.tan(0.5 * v)) + 2.0 * np.cos(v) / denom),
    }


def _pseudosphere(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """The tractricoid: the surface of revolution of a tractrix, curvature -1 everywhere.

    Written in the ``sech u`` / ``u - tanh u`` form rather than the ``log(tan)`` one on
    purpose — this parametrisation is finite on the whole domain, so unlike Dini's and
    Kuen's surfaces it needs no clipping and shows both trumpets around the cusp.
    """
    radius, extent = p["radius"], p["extent"]
    u, v = _grid_uv(n, (-extent, extent), (0.0, 2.0 * np.pi))
    sech = 1.0 / np.cosh(u)
    return {
        "u": u,
        "v": v,
        "x": radius * sech * np.cos(v),
        "y": radius * sech * np.sin(v),
        "z": radius * (u - np.tanh(u)),
    }


def _monkey_saddle(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """``z = x³ - 3xy²``: three descending valleys, so a monkey gets somewhere for its tail.

    The standard counter-example to the second-derivative test — the Hessian is singular at
    the origin, which is a degenerate critical point rather than a saddle or an extremum.
    """
    extent, scale = p["extent"], p["scale"]
    u, v = _grid_uv(n, (-extent, extent), (-extent, extent))
    return {"u": u, "v": v, "x": u, "y": v, "z": scale * (u**3 - 3.0 * u * v * v)}


def _knot_tube(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """A circular tube swept along a (p, q) torus knot — a knot with a *surface*.

    The sweep needs a frame on the curve. It is taken from the Frenet formulas with the
    two derivatives computed by central differences rather than symbolically: the
    expressions for ``c''`` of a torus knot are a page of trigonometry that nobody can
    check by reading, while a central difference at ``h = 1e-3`` is accurate to ~1e-10
    here and is obviously the derivative. The normal is only ill-defined where the
    curvature vanishes, which a torus knot's never does; the denominators are floored
    anyway so a pathological (p, q) degrades to a twisted tube instead of ``nan``.
    """
    pp, qq, tube = p["p"], p["q"], p["tube"]

    def centre(t: np.ndarray) -> np.ndarray:
        ring = 2.0 + np.cos(qq * t)
        return np.stack([ring * np.cos(pp * t), ring * np.sin(pp * t), -np.sin(qq * t)], axis=-1)

    u, v = _grid_uv(n, (0.0, 2.0 * np.pi), (0.0, 2.0 * np.pi))
    h = 1e-3
    ahead, behind = centre(u + h), centre(u - h)
    here = centre(u)
    d1 = (ahead - behind) / (2.0 * h)
    d2 = (ahead - 2.0 * here + behind) / (h * h)

    tangent = d1 / np.maximum(np.linalg.norm(d1, axis=-1, keepdims=True), 1e-12)
    radial = d2 - tangent * np.sum(d2 * tangent, axis=-1, keepdims=True)
    normal = radial / np.maximum(np.linalg.norm(radial, axis=-1, keepdims=True), 1e-12)
    binormal = np.cross(tangent, normal)

    offset = tube * (np.cos(v)[:, None] * normal + np.sin(v)[:, None] * binormal)
    pts = here + offset
    return {"u": u, "v": v, "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]}


# ----------------------------------------------------------------------------------
# Clouds and lattices
# ----------------------------------------------------------------------------------


def _gaussian_cloud(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    rng = _rng(p["seed"])
    pts = rng.normal(scale=(p["sx"], p["sy"], p["sz"]), size=(n, 3))
    return {
        "x": pts[:, 0],
        "y": pts[:, 1],
        "z": pts[:, 2],
        "r": np.linalg.norm(pts, axis=1),
    }


def _ball(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """Uniform inside a sphere — the cube-root radius is what makes it uniform by volume."""
    rng = _rng(p["seed"])
    direction = rng.normal(size=(n, 3))
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    radius = p["radius"] * np.cbrt(rng.random(n))
    pts = direction * radius[:, None]
    return {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2], "r": radius}


def _sphere_shell(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    rng = _rng(p["seed"])
    direction = rng.normal(size=(n, 3))
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    radius = p["radius"] * (1.0 + p["jitter"] * rng.normal(size=n))
    pts = direction * radius[:, None]
    return {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2], "r": radius}


def _random_walk(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    rng = _rng(p["seed"])
    steps = rng.normal(scale=p["step"], size=(n, 3))
    if p["drift"]:
        steps[:, 2] += p["drift"] * p["step"]
    pts = np.cumsum(steps, axis=0)
    return {
        "t": np.arange(n, dtype=np.float64),
        "x": pts[:, 0],
        "y": pts[:, 1],
        "z": pts[:, 2],
    }


def _blobs(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """Gaussian clusters — the shape every clustering demo needs and nothing else makes."""
    rng = _rng(p["seed"])
    k = max(1, int(p["clusters"]))
    centres = rng.uniform(-p["spread"], p["spread"], size=(k, 3))
    which = rng.integers(0, k, size=n)
    pts = centres[which] + rng.normal(scale=p["sigma"], size=(n, 3))
    return {
        "x": pts[:, 0],
        "y": pts[:, 1],
        "z": pts[:, 2],
        "cluster": which.astype(np.float64),
    }


def _multivariate_normal(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """A normal cloud with a *settable covariance* — the tilted ellipsoid, not the box.

    The three sliders are correlations, not covariance entries, because correlations are
    the readable parametrisation: each is bounded by ±1 and means the same thing whatever
    the widths are.

    Not every correlation triple is a covariance, though: ``(0.9, 0.9, -0.9)`` is not
    positive semi-definite and no cloud has it. Rather than refuse (a slider the user
    cannot move through is worse than useless) the matrix is repaired by clipping its
    eigenvalues at zero and re-assembling — the nearest PSD matrix in the Frobenius sense.
    Inside the valid region the repair is the identity, so the common case is exact.

    ``d`` is the Mahalanobis distance from the mean, which is free here: it is the norm of
    the *uncorrelated* draw before the covariance is applied. Colour by it and the
    iso-surfaces are ellipsoidal shells rather than spheres.
    """
    rng = _rng(p["seed"])
    sigma = np.array([p["sx"], p["sy"], p["sz"]], dtype=np.float64)
    rxy, rxz, ryz = p["rxy"], p["rxz"], p["ryz"]
    corr = np.array([[1.0, rxy, rxz], [rxy, 1.0, ryz], [rxz, ryz, 1.0]], dtype=np.float64)
    values, vectors = np.linalg.eigh(corr)
    root = (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T
    raw = rng.normal(size=(n, 3))
    pts = (raw @ root) * sigma
    return {
        "x": pts[:, 0],
        "y": pts[:, 1],
        "z": pts[:, 2],
        "d": np.linalg.norm(raw, axis=1),
    }


def _swiss_roll(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """The Swiss roll: a 2-D sheet rolled up in 3-D, the standard manifold-learning set.

    Same construction as ``sklearn.datasets.make_swiss_roll`` (``t`` over
    ``1.5 pi .. 4.5 pi``), rolled about **z** rather than about y so the sheet stands up in
    a z-up viewer. ``t`` is returned as a column: it is the true position along the
    manifold, which is what an embedding is supposed to recover, so colouring by it is how
    you read whether one did.
    """
    rng = _rng(p["seed"])
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.random(n))
    height, noise = p["height"], p["noise"]
    pts = np.column_stack([t * np.cos(t), t * np.sin(t), height * (rng.random(n) - 0.5)])
    if noise:
        pts = pts + rng.normal(scale=noise, size=pts.shape)
    return {"t": t, "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]}


def _s_curve(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """The S-curve: the other standard manifold set — a sheet folded, not rolled.

    Where the Swiss roll is a spiral, this one is a single S-shaped fold: an embedding can
    unroll it without tearing, which makes the pair together the usual before/after test.
    """
    rng = _rng(p["seed"])
    t = 3.0 * np.pi * (rng.random(n) - 0.5)
    width, noise = p["width"], p["noise"]
    pts = np.column_stack(
        [np.sin(t), width * (rng.random(n) - 0.5), np.sign(t) * (np.cos(t) - 1.0)]
    )
    if noise:
        pts = pts + rng.normal(scale=noise, size=pts.shape)
    return {"t": t, "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]}


def _two_moons(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """Two interleaved half-circles — the classic not-linearly-separable classifier test.

    The set is two-dimensional by nature; ``lift`` separates the classes along z so it can
    be *used* in a 3D viewer, and at ``lift = 0`` you get the textbook planar version drawn
    in the z = 0 plane. ``moon`` is the class label, so colour by it to see the ground
    truth.
    """
    rng = _rng(p["seed"])
    first = n // 2
    second = n - first
    a = np.pi * rng.random(first)
    b = np.pi * rng.random(second)
    gap, lift, noise = p["gap"], p["lift"], p["noise"]
    x = np.concatenate([np.cos(a), 1.0 - np.cos(b)])
    y = np.concatenate([np.sin(a), gap - np.sin(b)])
    label = np.concatenate([np.zeros(first), np.ones(second)])
    z = lift * (label - 0.5)
    pts = np.column_stack([x, y, z])
    if noise:
        pts = pts + rng.normal(scale=noise, size=pts.shape)
    return {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2], "moon": label}


def _uniform_cube(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """Uniform in an axis-aligned box — the null hypothesis for anything density-shaped.

    Every clustering, kernel-density and nearest-neighbour result needs a "what does
    *nothing* look like" reference, and this is it. The per-axis half-widths also make it
    the quickest way to fill a specific viewing box with points.
    """
    rng = _rng(p["seed"])
    pts = rng.uniform(-1.0, 1.0, size=(n, 3)) * np.array([p["sx"], p["sy"], p["sz"]])
    return {
        "x": pts[:, 0],
        "y": pts[:, 1],
        "z": pts[:, 2],
        "r": np.linalg.norm(pts, axis=1),
    }


def _cubic_lattice(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    side = max(2, int(round(n ** (1.0 / 3.0))))
    a = p["spacing"]
    jitter = p["jitter"]
    grid = np.arange(side, dtype=np.float64) * a
    gx, gy, gz = np.meshgrid(grid, grid, grid, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    if jitter:
        pts = pts + _rng(p["seed"]).normal(scale=jitter * a, size=pts.shape)
    return {
        "x": pts[:, 0],
        "y": pts[:, 1],
        "z": pts[:, 2],
        "r": np.linalg.norm(pts - pts.mean(axis=0), axis=1),
    }


def _fcc_lattice(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """Face-centred cubic: the packing most crystals actually use."""
    side = max(1, int(round((max(n, 4) / 4.0) ** (1.0 / 3.0))))
    a = p["spacing"]
    basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    cells = np.arange(side, dtype=np.float64)
    gx, gy, gz = np.meshgrid(cells, cells, cells, indexing="ij")
    origins = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    pts = (origins[:, None, :] + basis[None, :, :]).reshape(-1, 3) * a
    if p["jitter"]:
        pts = pts + _rng(p["seed"]).normal(scale=p["jitter"] * a, size=pts.shape)
    return {
        "x": pts[:, 0],
        "y": pts[:, 1],
        "z": pts[:, 2],
        "r": np.linalg.norm(pts - pts.mean(axis=0), axis=1),
    }


def _bcc_lattice(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """Body-centred cubic: a cube with an atom at its centre — iron, tungsten, the alkalis."""
    side = _cell_side(n, 2)
    basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    cells = np.arange(side, dtype=np.float64)
    gx, gy, gz = np.meshgrid(cells, cells, cells, indexing="ij")
    origins = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    pts = (origins[:, None, :] + basis[None, :, :]).reshape(-1, 3) * p["spacing"]
    return _lattice_table(pts, p["jitter"], p["spacing"], p["seed"])


def _diamond_lattice(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """Diamond cubic: two interpenetrating FCC lattices offset by a quarter body diagonal.

    Silicon, germanium and diamond itself. Eight sites per cell, so it needs the fewest
    cells of any lattice here to look like anything — start at a low sample count.
    """
    side = _cell_side(n, 8)
    fcc = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    basis = np.vstack([fcc, fcc + 0.25])
    cells = np.arange(side, dtype=np.float64)
    gx, gy, gz = np.meshgrid(cells, cells, cells, indexing="ij")
    origins = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    pts = (origins[:, None, :] + basis[None, :, :]).reshape(-1, 3) * p["spacing"]
    return _lattice_table(pts, p["jitter"], p["spacing"], p["seed"])


def _hcp_lattice(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """Hexagonal close-packed: the *other* densest packing — ABAB where FCC is ABCABC.

    Built from its true (non-orthogonal) primitive vectors with the ideal ``c/a =
    sqrt(8/3)``, so the twelve nearest neighbours really are equidistant. Magnesium,
    titanium, zinc, and every stack of oranges that is not FCC.
    """
    side = _cell_side(n, 2)
    a = p["spacing"]
    a1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([0.5, math.sqrt(3.0) / 2.0, 0.0])
    a3 = np.array([0.0, 0.0, math.sqrt(8.0 / 3.0)])
    # The B layer sits over the centroid of the A triangle, which in this 60-degree cell
    # is (a1 + a2) / 3 -- at (2/3, 1/3) it would sit over an edge and the packing would be
    # neither close nor hexagonal.
    basis = np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0, 0.5]])
    cells = np.arange(side, dtype=np.float64)
    gi, gj, gk = np.meshgrid(cells, cells, cells, indexing="ij")
    frac = np.column_stack([gi.ravel(), gj.ravel(), gk.ravel()])
    frac = (frac[:, None, :] + basis[None, :, :]).reshape(-1, 3)
    pts = a * (frac[:, 0:1] * a1 + frac[:, 1:2] * a2 + frac[:, 2:3] * a3)
    return _lattice_table(pts, p["jitter"], a, p["seed"])


def _honeycomb(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """A graphene sheet — a triangular lattice with a two-atom basis — stacked AB.

    The honeycomb is not a Bravais lattice, which is the whole point of including it: it
    is the smallest structure that needs a *basis*, and getting it wrong (one atom per
    cell) silently produces a triangular lattice instead. ``spacing`` is the bond length;
    the interlayer distance is graphite's, 2.36 bond lengths.
    """
    layers = max(1, int(p["layers"]))
    bond = p["spacing"]
    side = max(2, int(round(math.sqrt(max(n, MIN_SAMPLES) / (2.0 * layers)))))
    a1 = bond * np.array([1.5, math.sqrt(3.0) / 2.0, 0.0])
    a2 = bond * np.array([1.5, -math.sqrt(3.0) / 2.0, 0.0])
    basis = np.array([[0.0, 0.0, 0.0], [bond, 0.0, 0.0]])
    cells = np.arange(side, dtype=np.float64)
    gi, gj = np.meshgrid(cells, cells, indexing="ij")
    sheet = gi.ravel()[:, None] * a1 + gj.ravel()[:, None] * a2
    sheet = (sheet[:, None, :] + basis[None, :, :]).reshape(-1, 3)
    stack = []
    for layer in range(layers):
        shifted = sheet.copy()
        if layer % 2:  # AB stacking: every other sheet slides by one bond vector.
            shifted[:, 0] += bond
        shifted[:, 2] = layer * 2.36 * bond
        stack.append(shifted)
    pts = np.vstack(stack)
    return _lattice_table(pts, p["jitter"], bond, p["seed"])


def _fibonacci_quasilattice(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """The product of three Fibonacci chains: a lattice with no period, in golden ratio.

    Each axis is the canonical 1-D quasicrystal — the Sturmian word
    ``floor((k+1)/tau) - floor(k/tau)`` read as a sequence of long and short gaps whose
    ratio is ``tau``. It is aperiodic (no translation maps it to itself) yet perfectly
    ordered, which is exactly the property a quasicrystal has and a jittered lattice does
    not.

    It is honestly *not* a Penrose tiling: Penrose point sets come from a cut-and-project
    with a pentagonal window and have no representation as an axis product. This is the
    3-D construction that is both standard and exactly reproducible in ten lines; the
    diffraction-pattern intuition (sharp peaks at irrationally spaced positions) carries
    over unchanged.
    """
    side = _cell_side(n, 1)
    tau = (1.0 + math.sqrt(5.0)) / 2.0
    k = np.arange(side, dtype=np.float64)
    long_gap = (np.floor((k + 1.0) / tau) - np.floor(k / tau)) > 0.5
    gaps = np.where(long_gap, tau, 1.0)
    axis = np.concatenate([[0.0], np.cumsum(gaps)[:-1]])
    axis = p["spacing"] * axis / float(gaps.mean())
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    return _lattice_table(pts, p["jitter"], p["spacing"], p["seed"])


def _fibonacci_sphere(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """The golden-angle spiral: the standard way to put N points *evenly* on a sphere.

    Not a random cloud and not a lat/long grid — a lat/long grid piles points at the poles,
    and a random one leaves gaps. Equal-area rings in z plus a golden-angle azimuth give a
    point set whose nearest-neighbour distance is nearly constant, which is what makes it
    the usual choice for quadrature directions, environment sampling and icon-quality
    sphere markers.
    """
    radius, jitter = p["radius"], p["jitter"]
    i = np.arange(n, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * i / float(n)
    ring = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    theta = np.pi * (1.0 + math.sqrt(5.0)) * i
    pts = np.column_stack([ring * np.cos(theta), ring * np.sin(theta), z]) * radius
    if jitter:
        pts = pts + _rng(p["seed"]).normal(scale=jitter * radius, size=pts.shape)
    return {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2], "t": i / float(n)}


# ----------------------------------------------------------------------------------
# Vector fields
# ----------------------------------------------------------------------------------


def _field_grid(samples: int, extent: float) -> Tuple[np.ndarray, ...]:
    """A cubic sample lattice for a vector field, with about ``samples`` sites."""
    side = max(2, int(round(max(samples, 8) ** (1.0 / 3.0))))
    axis = np.linspace(-extent, extent, side)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    return gx.ravel(), gy.ravel(), gz.ravel()


def _abc_flow(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """The Arnold-Beltrami-Childress flow: the standard chaotic-advection test field."""
    x, y, z = _field_grid(n, p["extent"])
    a, b, c = p["A"], p["B"], p["C"]
    return {
        "x": x,
        "y": y,
        "z": z,
        "u": a * np.sin(z) + c * np.cos(y),
        "v": b * np.sin(x) + a * np.cos(z),
        "w": c * np.sin(y) + b * np.cos(x),
    }


def _dipole_field(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """A magnetic dipole aligned with +z — the textbook field, singular at the origin."""
    x, y, z = _field_grid(n, p["extent"])
    m = p["moment"]
    r2 = x * x + y * y + z * z
    r = np.sqrt(np.maximum(r2, 1e-6))
    r5 = np.maximum(r**5, 1e-9)
    return {
        "x": x,
        "y": y,
        "z": z,
        "u": m * (3.0 * x * z) / r5,
        "v": m * (3.0 * y * z) / r5,
        "w": m * (3.0 * z * z - r2) / r5,
    }


def _vortex_field(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """A columnar vortex about +z with an axial updraft: rotation you can see."""
    x, y, z = _field_grid(n, p["extent"])
    strength, lift = p["strength"], p["lift"]
    r2 = np.maximum(x * x + y * y, 1e-6)
    return {
        "x": x,
        "y": y,
        "z": z,
        "u": -strength * y / r2,
        "v": strength * x / r2,
        "w": np.full_like(z, lift),
    }


def _taylor_green(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """The Taylor-Green vortex: the canonical initial condition for transition to turbulence.

    ``w`` is zero everywhere, and that is the definition rather than an omission -- the
    Taylor-Green field is two-dimensional in its velocity and three-dimensional only in
    where that velocity lives. The vertical motion appears later, from the Navier-Stokes
    evolution, which this module does not do.
    """
    x, y, z = _field_grid(n, p["extent"])
    amp, k = p["amp"], p["k"]
    return {
        "x": x,
        "y": y,
        "z": z,
        "u": amp * np.sin(k * x) * np.cos(k * y) * np.cos(k * z),
        "v": -amp * np.cos(k * x) * np.sin(k * y) * np.cos(k * z),
        "w": np.zeros_like(z),
    }


def _hill_vortex(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """Hill's spherical vortex: a ball of rotational flow inside potential flow.

    An exact solution of the Euler equations and the standard model of a rising thermal or
    a spherical bubble wake. The two branches are computed for every site and selected with
    :func:`numpy.where` rather than indexed, which keeps it a single vectorised expression;
    the exterior branch's ``1/r**3`` is evaluated (and discarded) inside the sphere, so its
    denominator is floored to keep that discarded arithmetic finite.
    """
    x, y, z = _field_grid(n, p["extent"])
    a, speed = p["radius"], p["speed"]
    rho = np.hypot(x, y)
    r = np.hypot(rho, z)
    safe_r = np.maximum(r, 1e-9)
    safe_rho = np.maximum(rho, 1e-9)

    inside_rho = 1.5 * speed * rho * z / (a * a)
    inside_z = 1.5 * speed * (1.0 - (2.0 * rho * rho + z * z) / (a * a))

    cos_t, sin_t = z / safe_r, rho / safe_r
    ratio = (a / safe_r) ** 3
    v_r = speed * cos_t * (1.0 - ratio)
    v_t = -speed * sin_t * (1.0 + 0.5 * ratio)
    outside_rho = v_r * sin_t + v_t * cos_t
    outside_z = v_r * cos_t - v_t * sin_t

    is_inside = r <= a
    radial = np.where(is_inside, inside_rho, outside_rho)
    return {
        "x": x,
        "y": y,
        "z": z,
        "u": radial * x / safe_rho,
        "v": radial * y / safe_rho,
        "w": np.where(is_inside, inside_z, outside_z),
    }


def _linear_field(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """``(a x, b y, c z)`` — the whole family of linear fixed points on three sliders.

    All three positive is a source, all three negative a sink, mixed signs a saddle, and
    ``a + b + c = 0`` is incompressible. This is what a nonlinear field looks like near any
    non-degenerate zero, which is why it is worth having as a first-class object rather
    than as three separate entries.
    """
    x, y, z = _field_grid(n, p["extent"])
    return {"x": x, "y": y, "z": z, "u": p["a"] * x, "v": p["b"] * y, "w": p["c"] * z}


def _roessler_field(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """The velocity field of the Rössler system — the attractor's *cause*, not its orbit.

    The Dynamics panel integrates this system into a trajectory; drawing the field it comes
    from is the complementary view, and the one that shows why the orbit stretches in the
    plane and folds through the ``z(x - c)`` term.
    """
    x, y, z = _field_grid(n, p["extent"])
    return {
        "x": x,
        "y": y,
        "z": z,
        "u": -y - z,
        "v": x + p["a"] * y,
        "w": p["b"] + z * (x - p["c"]),
    }


def _lorenz_field(p: Mapping[str, float], n: int) -> Dict[str, np.ndarray]:
    """The Lorenz velocity field, sampled on a cube centred on the origin.

    The cube is centred rather than placed over the attractor's own ``z`` range (roughly 0
    to 50) so the two fixed points at ``z = rho - 1`` sit inside the box together with
    their mirror images; that symmetry is the readable part of the picture.
    """
    x, y, z = _field_grid(n, p["extent"])
    return {
        "x": x,
        "y": y,
        "z": z,
        "u": p["sigma"] * (y - x),
        "v": x * (p["rho"] - z) - y,
        "w": x * y - p["beta"] * z,
    }


# ----------------------------------------------------------------------------------
# The catalogue
# ----------------------------------------------------------------------------------


def _p(name: str, default: float, vmin: float, vmax: float, label: str = "", integer: bool = False):
    return Param3D(
        name=name, default=default, vmin=vmin, vmax=vmax, label=label or name, integer=integer
    )


GENERATORS: Dict[str, Generator3D] = {
    # -- curves ---------------------------------------------------------------
    "helix": Generator3D(
        key="helix",
        label="Helix",
        category="curve",
        icon="chart_line3d",
        description="A circle advancing along z. The simplest space curve there is.",
        kind="line3d",
        columns=("t", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        color_column="t",
        build=_helix,
        params=(
            _p("turns", 6.0, 0.5, 60.0),
            _p("radius", 1.0, 0.05, 10.0),
            _p("pitch", 0.15, -2.0, 2.0),
        ),
    ),
    "torus_knot": Generator3D(
        key="torus_knot",
        label="Torus knot (p, q)",
        category="curve",
        icon="chart_line3d",
        description="A curve winding p times one way and q the other around a torus. "
        "Coprime p and q give a genuine knot; a common factor gives a link.",
        kind="line3d",
        columns=("t", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        color_column="t",
        build=_torus_knot,
        params=(
            _p("p", 2.0, 1.0, 12.0, integer=True),
            _p("q", 3.0, 1.0, 12.0, integer=True),
            _p("radius", 1.0, 0.1, 5.0),
        ),
    ),
    "lissajous3d": Generator3D(
        key="lissajous3d",
        label="Lissajous 3D",
        category="curve",
        icon="wave",
        description="Three sinusoids at different frequencies. Rational ratios close; "
        "irrational ones fill a box.",
        kind="line3d",
        columns=("t", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        color_column="t",
        build=_lissajous3d,
        params=(
            _p("a", 3.0, 1.0, 20.0),
            _p("b", 2.0, 1.0, 20.0),
            _p("c", 5.0, 1.0, 20.0),
            _p("delta", 0.5 * math.pi, 0.0, 2.0 * math.pi),
        ),
    ),
    "spherical_spiral": Generator3D(
        key="spherical_spiral",
        label="Spherical spiral",
        category="curve",
        icon="chart_line3d",
        description="A loxodrome: constant-bearing winding from pole to pole.",
        kind="line3d",
        columns=("t", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        color_column="t",
        build=_spherical_spiral,
        params=(_p("turns", 24.0, 1.0, 200.0), _p("radius", 1.0, 0.1, 10.0)),
    ),
    "conical_spiral": Generator3D(
        key="conical_spiral",
        label="Conical spiral",
        category="curve",
        icon="chart_line3d",
        description="A spiral whose radius grows with height — a cone traced by a curve.",
        kind="line3d",
        columns=("t", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        color_column="t",
        build=_conical_spiral,
        params=(
            _p("turns", 8.0, 0.5, 60.0),
            _p("height", 2.0, 0.1, 20.0),
            _p("radius", 1.0, 0.05, 10.0),
        ),
    ),
    "viviani": Generator3D(
        key="viviani",
        label="Viviani's curve",
        category="curve",
        icon="chart_line3d",
        description="Where a sphere meets a cylinder through its centre — a figure-eight "
        "in space.",
        kind="line3d",
        columns=("t", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        color_column="t",
        build=_viviani,
        params=(_p("radius", 1.0, 0.1, 10.0),),
    ),
    "figure_eight_knot": Generator3D(
        key="figure_eight_knot",
        label="Figure-eight knot",
        category="curve",
        icon="chart_line3d",
        description="The 4_1 knot — the simplest knot that is not a torus knot, so no "
        "(p, q) in the entry above can reach it.",
        kind="line3d",
        columns=("t", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        color_column="t",
        build=_figure_eight_knot,
        params=(_p("scale", 1.0, 0.1, 10.0),),
    ),
    "granny_knot": Generator3D(
        key="granny_knot",
        label="Granny knot",
        category="curve",
        icon="chart_line3d",
        description="Two trefoils of the same handedness tied in series — the knot you get "
        "when you tie your shoes wrong. A twelve-term Fourier curve.",
        kind="line3d",
        columns=("t", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        color_column="t",
        build=_granny_knot,
        params=(_p("scale", 1.0, 0.1, 10.0),),
    ),
    "coiled_coil": Generator3D(
        key="coiled_coil",
        label="Coiled coil",
        category="curve",
        icon="chart_line3d",
        description="A helix wound around a helix: a telephone cord, a solenoid winding, a "
        "supercoiled filament. 'coils' is the small helix's turns per big one.",
        kind="line3d",
        columns=("t", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        color_column="t",
        build=_coiled_coil,
        params=(
            _p("turns", 4.0, 0.5, 40.0),
            _p("coils", 20.0, 1.0, 200.0),
            _p("radius", 1.0, 0.05, 10.0),
            _p("minor", 0.2, 0.01, 3.0),
            _p("pitch", 0.1, -2.0, 2.0),
        ),
    ),
    "rose3d": Generator3D(
        key="rose3d",
        label="Rose curve (3D)",
        category="curve",
        icon="wave",
        description="A rhodonea r = cos(k·t) lifted out of its plane. Odd k gives k petals, "
        "even k gives 2k — the classic surprise of polar plotting.",
        kind="line3d",
        columns=("t", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        color_column="t",
        build=_rose3d,
        params=(
            _p("k", 5.0, 1.0, 20.0),
            _p("radius", 1.0, 0.1, 10.0),
            _p("amp", 0.3, -3.0, 3.0),
        ),
    ),
    "twisted_cubic": Generator3D(
        key="twisted_cubic",
        label="Twisted cubic",
        category="curve",
        icon="chart_line3d",
        description="(t, t², t³): the smallest non-planar algebraic curve, and the standard "
        "example of a rational normal curve.",
        kind="line3d",
        columns=("t", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        color_column="t",
        build=_twisted_cubic,
        params=(_p("extent", 2.0, 0.2, 10.0), _p("scale", 0.5, -5.0, 5.0)),
    ),
    # -- surfaces -------------------------------------------------------------
    "sphere": Generator3D(
        key="sphere",
        label="Sphere",
        category="surface",
        icon="chart_surface3d",
        description="The reference surface. Useful for checking that a view, an aspect "
        "ratio or a colormap behaves.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_sphere,
        params=(_p("radius", 1.0, 0.1, 10.0),),
    ),
    "torus": Generator3D(
        key="torus",
        label="Torus",
        category="surface",
        icon="chart_surface3d",
        description="A tube of radius r bent into a circle of radius R.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_torus,
        params=(_p("R", 1.0, 0.1, 10.0), _p("r", 0.35, 0.01, 5.0)),
    ),
    "mobius": Generator3D(
        key="mobius",
        label="Möbius strip",
        category="surface",
        icon="chart_surface3d",
        description="One side, one edge. A half-twist band — the standard test that a "
        "renderer is not assuming closed, orientable surfaces.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_mobius,
        params=(_p("radius", 1.0, 0.1, 10.0), _p("width", 0.4, 0.02, 3.0)),
    ),
    "klein": Generator3D(
        key="klein",
        label="Klein bottle (figure-8)",
        category="surface",
        icon="chart_surface3d",
        description="The figure-8 immersion of the Klein bottle — the form that fits in "
        "a finite box.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_klein,
        params=(_p("radius", 2.0, 0.5, 10.0),),
    ),
    "saddle": Generator3D(
        key="saddle",
        label="Saddle (x² − y²)",
        category="surface",
        icon="chart_surface3d",
        description="The hyperbolic paraboloid: a maximum along one axis and a minimum "
        "along the other through the same point.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_saddle,
        params=(_p("extent", 2.0, 0.2, 20.0), _p("scale", 0.5, -5.0, 5.0)),
    ),
    "ripple": Generator3D(
        key="ripple",
        label="Ripple (sinc)",
        category="surface",
        icon="chart_surface3d",
        description="Concentric decaying waves. The decay is what keeps the outer rings "
        "from reading as a flat corrugated sheet.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_ripple,
        params=(
            _p("extent", 8.0, 0.5, 50.0),
            _p("freq", 2.0, 0.1, 20.0),
            _p("amp", 1.0, -10.0, 10.0),
        ),
    ),
    "gaussian_bump": Generator3D(
        key="gaussian_bump",
        label="Gaussian bump",
        category="surface",
        icon="chart_surface3d",
        description="A single smooth peak — a clean height field for testing lighting, "
        "colormaps and contours.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_gaussian_bump,
        params=(
            _p("extent", 3.0, 0.2, 30.0),
            _p("sigma", 1.0, 0.05, 10.0),
            _p("amp", 1.0, -10.0, 10.0),
        ),
    ),
    "cylinder": Generator3D(
        key="cylinder",
        label="Cylinder",
        category="surface",
        icon="chart_surface3d",
        description="An open tube along z.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_cylinder,
        params=(_p("radius", 1.0, 0.05, 10.0), _p("height", 2.0, 0.1, 20.0)),
    ),
    "cone": Generator3D(
        key="cone",
        label="Cone",
        category="surface",
        icon="chart_surface3d",
        description="A cone tapering to a point at the top.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_cone,
        params=(_p("radius", 1.0, 0.05, 10.0), _p("height", 2.0, 0.1, 20.0)),
    ),
    "hyperboloid": Generator3D(
        key="hyperboloid",
        label="Hyperboloid",
        category="surface",
        icon="chart_surface3d",
        description="A one-sheet hyperboloid — doubly ruled, so its wireframe is made of "
        "straight lines despite the curve.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_hyperboloid,
        params=(_p("radius", 1.0, 0.05, 10.0), _p("height", 1.5, 0.1, 10.0)),
    ),
    "shell": Generator3D(
        key="shell",
        label="Seashell",
        category="surface",
        icon="chart_surface3d",
        description="A logarithmic spiral tube. Heavy on triangles — lower the sample "
        "count before raising the turns.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_shell,
        params=(
            _p("turns", 4.0, 0.5, 12.0),
            _p("radius", 0.35, 0.02, 1.5),
            _p("taper", 0.18, 0.0, 0.6),
        ),
    ),
    "enneper": Generator3D(
        key="enneper",
        label="Enneper surface",
        category="surface",
        icon="chart_surface3d",
        description="A minimal surface that crosses itself past |u| = 1 — the reference "
        "non-embedded minimal surface.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_enneper,
        params=(_p("extent", 1.6, 0.2, 3.0), _p("scale", 1.0, 0.05, 5.0)),
    ),
    "catenoid": Generator3D(
        key="catenoid",
        label="Catenoid",
        category="surface",
        icon="chart_surface3d",
        description="The only minimal surface of revolution — the soap film between two "
        "rings. 'waist' is the neck radius.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_catenoid,
        params=(_p("waist", 0.6, 0.25, 5.0), _p("height", 1.5, 0.1, 3.0)),
    ),
    "helicoid": Generator3D(
        key="helicoid",
        label="Helicoid",
        category="surface",
        icon="chart_surface3d",
        description="A spiral ramp: minimal, ruled, and the catenoid's associate surface — "
        "one bends into the other without stretching.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_helicoid,
        params=(
            _p("turns", 1.5, 0.1, 12.0),
            _p("radius", 1.0, 0.05, 10.0),
            _p("pitch", 0.3, -3.0, 3.0),
        ),
    ),
    "dini": Generator3D(
        key="dini",
        label="Dini's surface",
        category="surface",
        icon="chart_surface3d",
        description="A pseudosphere dragged along a helix: constant negative curvature, so "
        "it is a piece of the hyperbolic plane you can hold.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_dini,
        params=(
            _p("radius", 1.0, 0.1, 5.0),
            _p("twist", 0.2, -2.0, 2.0),
            _p("turns", 2.0, 0.25, 8.0),
        ),
    ),
    "roman": Generator3D(
        key="roman",
        label="Roman surface (Steiner)",
        category="surface",
        icon="chart_surface3d",
        description="The projective plane mapped into 3-space with tetrahedral symmetry. "
        "Its six pinch points are real singularities, not artefacts.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_roman,
        params=(_p("radius", 1.4, 0.2, 5.0),),
    ),
    "boy": Generator3D(
        key="boy",
        label="Boy's surface",
        category="surface",
        icon="chart_surface3d",
        description="The projective plane again, but immersed: self-intersections only, no "
        "pinch points. The counterpart to the Roman surface.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_boy,
        params=(_p("scale", 1.0, 0.1, 8.0),),
    ),
    "supertoroid": Generator3D(
        key="supertoroid",
        label="Supertoroid",
        category="surface",
        icon="chart_surface3d",
        description="A torus with superellipse cross-sections. e = 1 is the ordinary torus, "
        "e → 0 a square ring, e > 2 a pinched star.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_supertoroid,
        params=(
            _p("R", 1.0, 0.1, 10.0),
            _p("r", 0.4, 0.01, 5.0),
            _p("e1", 1.0, 0.1, 4.0),
            _p("e2", 1.0, 0.1, 4.0),
        ),
    ),
    "superellipsoid": Generator3D(
        key="superellipsoid",
        label="Superellipsoid",
        category="surface",
        icon="chart_surface3d",
        description="Sphere at e = 1, rounded box as e → 0, octahedron at e = 2. One family "
        "covering most of the primitives a CAD kernel ships with.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_superellipsoid,
        params=(
            _p("radius", 1.0, 0.1, 10.0),
            _p("e1", 0.5, 0.1, 4.0),
            _p("e2", 0.5, 0.1, 4.0),
        ),
    ),
    "kuen": Generator3D(
        key="kuen",
        label="Kuen's surface",
        category="surface",
        icon="chart_surface3d",
        description="Constant negative curvature with none of the symmetry that usually "
        "comes with it — a Bäcklund transform of the pseudosphere.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_kuen,
        params=(_p("extent", 4.0, 1.0, 8.0), _p("scale", 1.0, 0.1, 5.0)),
    ),
    "pseudosphere": Generator3D(
        key="pseudosphere",
        label="Pseudosphere",
        category="surface",
        icon="chart_surface3d",
        description="The tractricoid: curvature −1 everywhere, the hyperbolic answer to the "
        "sphere. Two trumpets meeting at a cusp.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_pseudosphere,
        params=(_p("radius", 1.0, 0.1, 10.0), _p("extent", 3.0, 0.5, 5.0)),
    ),
    "monkey_saddle": Generator3D(
        key="monkey_saddle",
        label="Monkey saddle",
        category="surface",
        icon="chart_surface3d",
        description="z = x³ − 3xy²: three valleys, one for each leg and one for the tail. "
        "The standard degenerate critical point.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_monkey_saddle,
        params=(_p("extent", 1.5, 0.2, 20.0), _p("scale", 1.0, -5.0, 5.0)),
    ),
    "knot_tube": Generator3D(
        key="knot_tube",
        label="Knot tube (p, q)",
        category="surface",
        icon="chart_surface3d",
        description="A circular tube swept along a (p, q) torus knot, so the knot can be "
        "shaded and lit instead of drawn as a hairline.",
        kind="surface3d",
        columns=("u", "v", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        build=_knot_tube,
        params=(
            _p("p", 2.0, 1.0, 12.0, integer=True),
            _p("q", 3.0, 1.0, 12.0, integer=True),
            _p("tube", 0.25, 0.02, 1.0),
        ),
    ),
    # -- clouds ---------------------------------------------------------------
    "gaussian_cloud": Generator3D(
        key="gaussian_cloud",
        label="Gaussian cloud",
        category="cloud",
        icon="chart_volume3d",
        description="An anisotropic normal cloud. The per-axis widths make it the quickest "
        "way to see what a 3D scatter does with millions of points.",
        kind="scatter3d",
        columns=("x", "y", "z", "r"),
        plot_columns=("x", "y", "z"),
        color_column="r",
        build=_gaussian_cloud,
        params=(
            _p("sx", 1.0, 0.01, 20.0),
            _p("sy", 1.0, 0.01, 20.0),
            _p("sz", 1.0, 0.01, 20.0),
            _p("seed", 0.0, 0.0, 9999.0, integer=True),
        ),
    ),
    "ball": Generator3D(
        key="ball",
        label="Uniform ball",
        category="cloud",
        icon="chart_volume3d",
        description="Uniform by volume inside a sphere — not the same as uniform radius, "
        "which piles points at the centre.",
        kind="volume3d",
        columns=("x", "y", "z", "r"),
        plot_columns=("x", "y", "z"),
        color_column="r",
        build=_ball,
        params=(_p("radius", 1.0, 0.05, 20.0), _p("seed", 0.0, 0.0, 9999.0, integer=True)),
    ),
    "sphere_shell": Generator3D(
        key="sphere_shell",
        label="Spherical shell",
        category="cloud",
        icon="chart_volume3d",
        description="Points on a sphere with a little radial noise — a hollow cloud, which "
        "is the case that shows whether depth cues are working.",
        kind="scatter3d",
        columns=("x", "y", "z", "r"),
        plot_columns=("x", "y", "z"),
        color_column="r",
        build=_sphere_shell,
        params=(
            _p("radius", 1.0, 0.05, 20.0),
            _p("jitter", 0.05, 0.0, 1.0),
            _p("seed", 0.0, 0.0, 9999.0, integer=True),
        ),
    ),
    "random_walk": Generator3D(
        key="random_walk",
        label="Random walk 3D",
        category="cloud",
        icon="chart_line3d",
        description="A Brownian path. 'Drift' biases each step along z, turning the walk "
        "into a trajectory.",
        kind="line3d",
        columns=("t", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        color_column="t",
        build=_random_walk,
        params=(
            _p("step", 1.0, 0.01, 10.0),
            _p("drift", 0.0, -2.0, 2.0),
            _p("seed", 0.0, 0.0, 9999.0, integer=True),
        ),
    ),
    "blobs": Generator3D(
        key="blobs",
        label="Clusters (blobs)",
        category="cloud",
        icon="chart_scatter3d",
        description="Gaussian clusters at random centres, with the cluster index as a "
        "column — colour by it to see the labels.",
        kind="scatter3d",
        columns=("x", "y", "z", "cluster"),
        plot_columns=("x", "y", "z"),
        color_column="cluster",
        build=_blobs,
        params=(
            _p("clusters", 4.0, 1.0, 40.0, integer=True),
            _p("spread", 3.0, 0.1, 30.0),
            _p("sigma", 0.5, 0.01, 10.0),
            _p("seed", 0.0, 0.0, 9999.0, integer=True),
        ),
    ),
    "multivariate_normal": Generator3D(
        key="multivariate_normal",
        label="Multivariate normal",
        category="cloud",
        icon="chart_volume3d",
        description="A normal cloud with a settable covariance: three widths and three "
        "correlations. Impossible correlation triples snap to the nearest real covariance.",
        kind="scatter3d",
        columns=("x", "y", "z", "d"),
        plot_columns=("x", "y", "z"),
        color_column="d",
        build=_multivariate_normal,
        params=(
            _p("sx", 1.0, 0.01, 20.0),
            _p("sy", 1.0, 0.01, 20.0),
            _p("sz", 1.0, 0.01, 20.0),
            _p("rxy", 0.6, -1.0, 1.0),
            _p("rxz", 0.0, -1.0, 1.0),
            _p("ryz", -0.3, -1.0, 1.0),
            _p("seed", 0.0, 0.0, 9999.0, integer=True),
        ),
    ),
    "swiss_roll": Generator3D(
        key="swiss_roll",
        label="Swiss roll",
        category="cloud",
        icon="chart_scatter3d",
        description="A 2D sheet rolled into 3D — the standard manifold-learning test set. "
        "Colour by t, the true position along the sheet, to grade an embedding.",
        kind="scatter3d",
        columns=("t", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        color_column="t",
        build=_swiss_roll,
        params=(
            _p("height", 20.0, 0.5, 100.0),
            _p("noise", 0.0, 0.0, 2.0),
            _p("seed", 0.0, 0.0, 9999.0, integer=True),
        ),
    ),
    "s_curve": Generator3D(
        key="s_curve",
        label="S-curve",
        category="cloud",
        icon="chart_scatter3d",
        description="The other standard manifold set: a sheet folded into an S rather than "
        "rolled, so it unfolds without tearing.",
        kind="scatter3d",
        columns=("t", "x", "y", "z"),
        plot_columns=("x", "y", "z"),
        color_column="t",
        build=_s_curve,
        params=(
            _p("width", 2.0, 0.1, 20.0),
            _p("noise", 0.0, 0.0, 1.0),
            _p("seed", 0.0, 0.0, 9999.0, integer=True),
        ),
    ),
    "two_moons": Generator3D(
        key="two_moons",
        label="Two moons",
        category="cloud",
        icon="chart_scatter3d",
        description="Two interleaved crescents — the not-linearly-separable classifier "
        "test. 'lift' pulls the classes apart along z; at 0 it is the planar original.",
        kind="scatter3d",
        columns=("x", "y", "z", "moon"),
        plot_columns=("x", "y", "z"),
        color_column="moon",
        build=_two_moons,
        params=(
            _p("gap", 0.5, -1.0, 2.0),
            _p("lift", 0.5, 0.0, 5.0),
            _p("noise", 0.06, 0.0, 1.0),
            _p("seed", 0.0, 0.0, 9999.0, integer=True),
        ),
    ),
    "uniform_cube": Generator3D(
        key="uniform_cube",
        label="Uniform box",
        category="cloud",
        icon="chart_volume3d",
        description="Uniform inside an axis-aligned box: the null hypothesis every density, "
        "clustering and nearest-neighbour result has to be read against.",
        kind="volume3d",
        columns=("x", "y", "z", "r"),
        plot_columns=("x", "y", "z"),
        color_column="r",
        build=_uniform_cube,
        params=(
            _p("sx", 1.0, 0.01, 20.0),
            _p("sy", 1.0, 0.01, 20.0),
            _p("sz", 1.0, 0.01, 20.0),
            _p("seed", 0.0, 0.0, 9999.0, integer=True),
        ),
    ),
    # -- lattices -------------------------------------------------------------
    "cubic_lattice": Generator3D(
        key="cubic_lattice",
        label="Cubic lattice",
        category="lattice",
        icon="grid",
        description="A simple cubic array of sites, optionally jittered — the reference "
        "geometry for anything crystallographic.",
        kind="scatter3d",
        columns=("x", "y", "z", "r"),
        plot_columns=("x", "y", "z"),
        color_column="r",
        build=_cubic_lattice,
        params=(
            _p("spacing", 1.0, 0.01, 10.0),
            _p("jitter", 0.0, 0.0, 0.5),
            _p("seed", 0.0, 0.0, 9999.0, integer=True),
        ),
    ),
    "fcc_lattice": Generator3D(
        key="fcc_lattice",
        label="FCC lattice",
        category="lattice",
        icon="grid",
        description="Face-centred cubic — the packing most metals actually adopt, four "
        "sites per unit cell.",
        kind="scatter3d",
        columns=("x", "y", "z", "r"),
        plot_columns=("x", "y", "z"),
        color_column="r",
        build=_fcc_lattice,
        params=(
            _p("spacing", 1.0, 0.01, 10.0),
            _p("jitter", 0.0, 0.0, 0.5),
            _p("seed", 0.0, 0.0, 9999.0, integer=True),
        ),
    ),
    "bcc_lattice": Generator3D(
        key="bcc_lattice",
        label="BCC lattice",
        category="lattice",
        icon="grid",
        description="Body-centred cubic: a cube with one more atom at its centre. Iron, "
        "tungsten, chromium and the alkali metals.",
        kind="scatter3d",
        columns=("x", "y", "z", "r"),
        plot_columns=("x", "y", "z"),
        color_column="r",
        build=_bcc_lattice,
        params=(
            _p("spacing", 1.0, 0.01, 10.0),
            _p("jitter", 0.0, 0.0, 0.5),
            _p("seed", 0.0, 0.0, 9999.0, integer=True),
        ),
    ),
    "diamond_lattice": Generator3D(
        key="diamond_lattice",
        label="Diamond lattice",
        category="lattice",
        icon="cube",
        description="Two FCC lattices offset by a quarter of the body diagonal — diamond, "
        "silicon, germanium. Eight sites per cell, so keep the sample count low.",
        kind="scatter3d",
        columns=("x", "y", "z", "r"),
        plot_columns=("x", "y", "z"),
        color_column="r",
        build=_diamond_lattice,
        params=(
            _p("spacing", 1.0, 0.01, 10.0),
            _p("jitter", 0.0, 0.0, 0.5),
            _p("seed", 0.0, 0.0, 9999.0, integer=True),
        ),
    ),
    "hcp_lattice": Generator3D(
        key="hcp_lattice",
        label="HCP lattice",
        category="lattice",
        icon="grid",
        description="Hexagonal close packing — ABAB stacking where FCC is ABCABC. The same "
        "density, a different symmetry: magnesium, titanium, zinc.",
        kind="scatter3d",
        columns=("x", "y", "z", "r"),
        plot_columns=("x", "y", "z"),
        color_column="r",
        build=_hcp_lattice,
        params=(
            _p("spacing", 1.0, 0.01, 10.0),
            _p("jitter", 0.0, 0.0, 0.5),
            _p("seed", 0.0, 0.0, 9999.0, integer=True),
        ),
    ),
    "honeycomb": Generator3D(
        key="honeycomb",
        label="Honeycomb (graphene)",
        category="lattice",
        icon="grid",
        description="A graphene sheet: a triangular lattice with a two-atom basis, so it is "
        "not a Bravais lattice. Stack the layers AB and it is graphite.",
        kind="scatter3d",
        columns=("x", "y", "z", "r"),
        plot_columns=("x", "y", "z"),
        color_column="r",
        build=_honeycomb,
        params=(
            _p("spacing", 1.0, 0.01, 10.0),
            _p("layers", 1.0, 1.0, 20.0, integer=True),
            _p("jitter", 0.0, 0.0, 0.5),
            _p("seed", 0.0, 0.0, 9999.0, integer=True),
        ),
    ),
    "quasilattice": Generator3D(
        key="quasilattice",
        label="Fibonacci quasilattice",
        category="lattice",
        icon="grid",
        description="Aperiodic but perfectly ordered: two spacings in the golden ratio, in "
        "Fibonacci order, on every axis. A quasicrystal, not a jittered lattice.",
        kind="scatter3d",
        columns=("x", "y", "z", "r"),
        plot_columns=("x", "y", "z"),
        color_column="r",
        build=_fibonacci_quasilattice,
        params=(
            _p("spacing", 1.0, 0.01, 10.0),
            _p("jitter", 0.0, 0.0, 0.5),
            _p("seed", 0.0, 0.0, 9999.0, integer=True),
        ),
    ),
    "fibonacci_sphere": Generator3D(
        key="fibonacci_sphere",
        label="Fibonacci sphere",
        category="lattice",
        icon="chart_scatter3d",
        description="N points spread evenly over a sphere by the golden angle — neither "
        "clumped like a random cloud nor pole-heavy like a lat/long grid.",
        kind="scatter3d",
        columns=("x", "y", "z", "t"),
        plot_columns=("x", "y", "z"),
        color_column="t",
        build=_fibonacci_sphere,
        params=(
            _p("radius", 1.0, 0.05, 20.0),
            _p("jitter", 0.0, 0.0, 0.5),
            _p("seed", 0.0, 0.0, 9999.0, integer=True),
        ),
    ),
    # -- fields ---------------------------------------------------------------
    "abc_flow": Generator3D(
        key="abc_flow",
        label="ABC flow",
        category="field",
        icon="chart_quiver3d",
        description="Arnold-Beltrami-Childress: an exact Euler solution whose streamlines "
        "are chaotic. The standard chaotic-advection test field.",
        kind="quiver3d",
        columns=("x", "y", "z", "u", "v", "w"),
        plot_columns=("x", "y", "z"),
        build=_abc_flow,
        params=(
            _p("A", 1.0, -3.0, 3.0),
            _p("B", 0.7, -3.0, 3.0),
            _p("C", 0.5, -3.0, 3.0),
            _p("extent", 3.14159, 0.5, 12.0),
        ),
    ),
    "dipole": Generator3D(
        key="dipole",
        label="Dipole field",
        category="field",
        icon="chart_quiver3d",
        description="A magnetic dipole along +z. Singular at the origin, so the arrows "
        "near the centre are long by construction — lower 'scale' to read it.",
        kind="quiver3d",
        columns=("x", "y", "z", "u", "v", "w"),
        plot_columns=("x", "y", "z"),
        build=_dipole_field,
        params=(_p("moment", 1.0, -5.0, 5.0), _p("extent", 2.0, 0.2, 12.0)),
    ),
    "vortex": Generator3D(
        key="vortex",
        label="Vortex column",
        category="field",
        icon="chart_quiver3d",
        description="Rotation about +z with a uniform updraft. The clearest field for "
        "checking that arrow orientation is being drawn correctly.",
        kind="quiver3d",
        columns=("x", "y", "z", "u", "v", "w"),
        plot_columns=("x", "y", "z"),
        build=_vortex_field,
        params=(
            _p("strength", 1.0, -5.0, 5.0),
            _p("lift", 0.2, -3.0, 3.0),
            _p("extent", 2.0, 0.2, 12.0),
        ),
    ),
    "taylor_green": Generator3D(
        key="taylor_green",
        label="Taylor–Green vortex",
        category="field",
        icon="chart_quiver3d",
        description="The canonical initial condition for the transition to turbulence. Its "
        "vertical velocity is zero by definition — the 3D motion appears as it evolves.",
        kind="quiver3d",
        columns=("x", "y", "z", "u", "v", "w"),
        plot_columns=("x", "y", "z"),
        build=_taylor_green,
        params=(
            _p("amp", 1.0, -5.0, 5.0),
            _p("k", 1.0, 0.1, 5.0),
            _p("extent", 3.14159, 0.5, 12.0),
        ),
    ),
    "hill_vortex": Generator3D(
        key="hill_vortex",
        label="Hill's spherical vortex",
        category="field",
        icon="chart_quiver3d",
        description="An exact Euler solution: a ball of rotational flow inside potential "
        "flow. The model of a rising thermal, and of a spherical bubble's wake.",
        kind="quiver3d",
        columns=("x", "y", "z", "u", "v", "w"),
        plot_columns=("x", "y", "z"),
        build=_hill_vortex,
        params=(
            _p("radius", 1.0, 0.1, 8.0),
            _p("speed", 1.0, -5.0, 5.0),
            _p("extent", 2.0, 0.2, 12.0),
        ),
    ),
    "linear_field": Generator3D(
        key="linear_field",
        label="Source / sink / saddle",
        category="field",
        icon="chart_quiver3d",
        description="(a·x, b·y, c·z): all signs positive is a source, all negative a sink, "
        "mixed a saddle, and a+b+c = 0 incompressible. Every fixed point looks like this "
        "close up.",
        kind="quiver3d",
        columns=("x", "y", "z", "u", "v", "w"),
        plot_columns=("x", "y", "z"),
        build=_linear_field,
        params=(
            _p("a", 1.0, -3.0, 3.0),
            _p("b", 1.0, -3.0, 3.0),
            _p("c", -2.0, -3.0, 3.0),
            _p("extent", 2.0, 0.2, 12.0),
        ),
    ),
    "roessler_field": Generator3D(
        key="roessler_field",
        label="Rössler flow field",
        category="field",
        icon="chart_quiver3d",
        description="The velocity field the Rössler attractor lives in — the stretching in "
        "the plane and the fold through z·(x − c), before any trajectory is drawn.",
        kind="quiver3d",
        columns=("x", "y", "z", "u", "v", "w"),
        plot_columns=("x", "y", "z"),
        build=_roessler_field,
        params=(
            _p("a", 0.2, -1.0, 1.0),
            _p("b", 0.2, -1.0, 1.0),
            _p("c", 5.7, 0.1, 20.0),
            _p("extent", 8.0, 0.5, 30.0),
        ),
    ),
    "lorenz_field": Generator3D(
        key="lorenz_field",
        label="Lorenz flow field",
        category="field",
        icon="chart_quiver3d",
        description="The Lorenz system as a field on a cube centred at the origin, so both "
        "non-trivial fixed points and their mirror images sit inside the box.",
        kind="quiver3d",
        columns=("x", "y", "z", "u", "v", "w"),
        plot_columns=("x", "y", "z"),
        build=_lorenz_field,
        params=(
            _p("sigma", 10.0, 0.1, 30.0),
            _p("rho", 28.0, 0.1, 100.0),
            _p("beta", 2.6667, 0.1, 10.0),
            _p("extent", 20.0, 1.0, 60.0),
        ),
    ),
}

#: Every generator key, in catalogue order.
GENERATOR_KEYS: Tuple[str, ...] = tuple(GENERATORS)


def generator(key: str) -> Generator3D:
    """The :class:`Generator3D` named ``key``. Raises :class:`GeneratorError` if unknown."""
    spec = GENERATORS.get(str(key).strip().lower())
    if spec is None:
        raise GeneratorError(
            f"unknown generator {key!r}; expected one of {', '.join(GENERATOR_KEYS)}"
        )
    return spec


def by_category(category: str) -> Tuple[Generator3D, ...]:
    """Every generator in ``category``, in catalogue order."""
    return tuple(g for g in GENERATORS.values() if g.category == category)


def category_label(category: str) -> str:
    """The human label for a category key, or the key itself if it is unknown."""
    for key, label in CATEGORIES:
        if key == category:
            return label
    return category
