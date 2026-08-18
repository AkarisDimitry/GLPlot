"""A catalogue of steppable simulations, stored as data rather than as code.

Pure numpy and stdlib -- no imgui, no engine import, no OpenGL (CONTRACT 5.1 rule 7). The
animation panels and the timeline are thin shells over this module, exactly as the 3D
panels are a shell over :mod:`glplot.gui.generators3d` and the Dynamics panel over
:mod:`glplot.gui.dynamics`. Everything here runs headless, which is why it is testable.

Why a catalogue, and why it is *not* a generator catalogue
----------------------------------------------------------
:mod:`glplot.gui.generators3d` answers "give me a table". This module answers "give me a
table *per frame*, forever". That is a different shape of object and pretending otherwise
would cost the caller correctness: a generator is a pure function of its parameters, while
a process carries state that only makes sense as the accumulation of every step before it.
An N-body cluster at frame 900 is not computable from the parameters alone in any useful
sense -- it *is* the 900 steps.

So the record here is :class:`Process`, three functions and a description of what they
produce:

* ``initial_state(params) -> State`` -- frame 0.
* ``step(state, params, dt) -> State`` -- frame *n* to frame *n+1*.
* ``state_to_table(state, params) -> {column: array}`` -- any frame as an ordinary table,
  so every existing plotting path (layers, colormaps, the DataStore) accepts it unchanged.

Continuous versus discrete time -- stated in the type
-----------------------------------------------------
A pendulum has a timestep. Conway's Game of Life does not: one step is one *generation*,
and there is no smaller amount of it. Collapsing the two -- giving Life a fake ``dt=1.0``
and calling it seconds -- would make ``state.time`` a lie and would let a caller ask for
half a generation, which has no meaning. So :attr:`Process.time_model` is
:data:`TIME_CONTINUOUS` or :data:`TIME_DISCRETE` and the difference is enforced:

* Continuous processes advance ``state.time`` by exactly the ``dt`` they were given, and
  subdivide it internally into :attr:`Process.substeps` integrator sub-steps for accuracy.
* Discrete processes advance ``state.time`` by exactly ``1.0`` (a generation count, not
  seconds) and **raise** :class:`ProcessError` if a caller passes a ``dt`` at all.

Table layouts
-------------
Two shapes, declared per process in :attr:`Process.table_layout`, because a caller that
autoscales axes needs to know which it is getting:

* :data:`LAYOUT_SNAPSHOT` -- one row per particle/cell/pixel, row count fixed by the
  parameters. N-body, Life, the fractal rasters, the grids.
* :data:`LAYOUT_HISTORY` -- one row per frame so far; the trajectory *is* the picture.
  The projectile arc, the pendulum's phase history, the two-body orbit. Row count grows
  to :data:`MAX_TRAIL` and then scrolls, dropping the oldest row.

Cost ceilings
-------------
A frame must be affordable at interactive rates, so every unbounded quantity has a
documented ceiling and the value at that ceiling was measured, not guessed (numpy 1.26 /
Python 3.12 on an Apple-silicon laptop, single core):

===========================  ==================  =========================================
Ceiling                      Value               Measured cost at the ceiling
===========================  ==================  =========================================
:data:`MAX_BODIES`           256 bodies          ~0.35 ms per O(N^2) force evaluation
:data:`MAX_NODES`            4096 masses         ~0.05 ms per chain sub-step
:data:`MAX_GRID_SIDE`        512 cells/side      Life 0.8 ms/step, Gray-Scott 2.7 ms/step
:data:`MAX_CELLS`            262144 cells        the 512x512 grid above
:data:`MAX_ITERATIONS`       1000 iterations     per-pixel escape budget
:data:`MAX_ESCAPE_WORK`      16e6 point-iters    ~12 ms typical, ~55 ms worst-case frame
:data:`MAX_POINTS_PER_FRAME` 65536 IFS points    ~2.0 ms per chaos-game frame
:data:`MAX_TRAIL`            20000 rows          history tables scroll past this
:data:`MAX_ACCUM_POINTS`     2e6 points          32 MB of chaos-game buffer, then it drops
===========================  ==================  =========================================

The escape-time ceiling is the binding one and deserves its number spelled out.
:data:`MAX_ESCAPE_WORK` counts *pixels times iterations*, and cost scales with that product
only loosely: what really matters is how many pixels are *inside* the set, because those
run the whole budget while the ones outside leave in a handful of iterations and are
dropped from the working arrays. So the same budget costs very different amounts at
different depths, and both numbers are worth stating:

* At the ceiling (256x256 at 244 effective iterations, or 192x192 at 434), a **deep zoom**
  frame -- the worst case, where most pixels never escape -- measures **36 ms**, about
  28 fps.
* The **defaults** (192x192 at 256 iterations) measure **4 ms** on the opening whole-set
  view and **26 ms** once the zoom is inside the seahorse valley, so roughly 38 fps at the
  slowest point of the default animation. Julia at the same resolution is **9 ms**.

Raising the resolution past those numbers trades frame rate for detail and the caller is
entitled to do it, up to the ceiling, after which :func:`escape_budget` silently lowers
the iteration count -- never the resolution -- and reports the value it actually used in
``state.data["max_iter"]``.

Every other process in the catalogue costs **under 0.3 ms per frame at its defaults**,
which is why the ceilings above are expressed per-step rather than per-frame: for them the
frame budget is never the binding constraint, the plot is.

What the integrators actually conserve
--------------------------------------
:meth:`Process.total_energy` exists so this is checkable rather than asserted, and the
numbers below are measured over 5000 frames at each process's defaults, as a fraction of
the initial energy. "Band" is peak-to-peak; "drift" is where it ended up:

============================  =========  ==========  ==========================
Process                       Integrator Band        Drift after 5000 frames
============================  =========  ==========  ==========================
:data:`TWO_BODY`              Verlet     7.2e-5      1.3e-7
:data:`NBODY`                 Verlet     1.4e-3      9.8e-5
:data:`SPRING_CHAIN`          Verlet     6.4e-6      4.7e-6
:data:`WAVE2D`                sympl.Euler 6.7e-2     1.2e-3
:data:`PENDULUM` (undriven)   RK4        3.1e-9      2.9e-9
:data:`DOUBLE_PENDULUM`       RK4        4.9e-8      1.7e-8
============================  =========  ==========  ==========================

The symplectic entries have a *band* and no trend: over 20000 frames the N-body energy's
per-quartile mean settles at -29.767, -29.772, -29.773, -29.773 and stays there. That is
the property that matters, and it is the property forward Euler does not have -- an Euler
two-body orbit gains energy monotonically and unwinds into a spiral within a few hundred
steps, which is not a small quantitative error but a qualitatively wrong picture.

The NBODY row is measured on Accelerate/arm64; six mutually gravitating bodies is a
chaotic system, so a different libm/BLAS (OpenBLAS/x86_64, as in Windows and Linux CI)
lands on a different point of the same bounded orbit family and measures a band around
5x larger. The other rows are far enough from chaos, or short enough, not to show this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------------------
# Cost ceilings. Every one of these bounds a quantity that would otherwise grow without
# limit -- either per frame (work) or across frames (memory).
# ----------------------------------------------------------------------------------

#: Hard ceiling on gravitating bodies. The force evaluation is O(N^2) with no tree code:
#: 256 bodies is ~0.35 ms per evaluation, which at 4 sub-steps is ~1.4 ms of a frame.
#: A Barnes-Hut tree would raise this by orders of magnitude and is deliberately not here
#: -- the interesting demonstrations are a handful of bodies, not a galaxy.
MAX_BODIES = 256

#: Hard ceiling on masses in a spring chain / discretised string. 4096 nodes is ~0.05 ms
#: per Verlet sub-step; the limit is really the plot, since the chain is drawn as a line.
MAX_NODES = 4096

#: Hard ceiling on a square grid's side, for Life, the automata and the PDE grids.
MAX_GRID_SIDE = 512

#: Hard ceiling on total cells in any grid, so a 512x8 grid is legal but 512x512 is the
#: largest square one. At 512x512: Life 0.8 ms/step, Gray-Scott 2.7 ms/step.
MAX_CELLS = 262_144

#: Smallest usable grid side. Below 4 cells a neighbourhood wraps onto itself twice and
#: the automaton rules stop meaning what they say.
MIN_GRID_SIDE = 4

#: Hard ceiling on escape-time iterations per pixel.
MAX_ITERATIONS = 1000

#: Hard ceiling on escape-time work per frame, in pixel-iterations. See the module
#: docstring: ~12 ms typical and ~55 ms worst case at this value.
MAX_ESCAPE_WORK = 16_000_000

#: The zoom floor for a float64 escape-time raster. A pixel of a 512-wide view at this
#: span is 2e-15 across, and float64 resolves about 2.2e-16 near a centre of order 1 --
#: so this is roughly ten ulps per pixel, the last depth at which the image is still an
#: image rather than a quantised blockiness. Zooming past it is refused by clamping, not
#: by raising, because a zoom animation should stop deepening rather than stop playing.
#: Going deeper genuinely requires arbitrary-precision arithmetic, which is out of scope.
MIN_SPAN = 1e-12

#: Hard ceiling on chaos-game points generated per frame (~2.0 ms at this value).
MAX_POINTS_PER_FRAME = 65_536

#: Hard ceiling on accumulated chaos-game points. Two float64 columns plus an int8 map
#: index is ~34 MB here; past it the buffer drops its oldest points.
MAX_ACCUM_POINTS = 2_000_000

#: Hard ceiling on rows in a :data:`LAYOUT_HISTORY` table. Past it the history scrolls.
MAX_TRAIL = 20_000

#: Hard ceiling on retained rows of an elementary-CA space-time diagram.
MAX_CA_ROWS = 1024

#: Hard ceiling on integrator sub-steps per frame. Above ~64 the accuracy gain is below
#: float64 round-off for every system here and the frame cost is all that is left.
MAX_SUBSTEPS = 64

#: Hard ceiling on frames :meth:`Process.run` will produce in one call. Stepping is
#: normally driven by the caller's timeline one frame at a time; ``run`` is for tests and
#: for offline export, and an unbounded ``run`` is an unbounded memory request.
MAX_FRAMES = 100_000

#: Bailout radius for the escape-time iteration. 2 is the mathematically minimal choice
#: (a point leaving the disc of radius 2 provably never returns); 4 costs a fraction of an
#: iteration more and makes the smooth/normalised iteration count visibly less banded,
#: because the overshoot past the bailout is proportionally smaller.
ESCAPE_RADIUS = 4.0

# ----------------------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------------------

#: A process whose steps carry a real timestep in seconds. ``dt`` is meaningful and the
#: process subdivides it internally.
TIME_CONTINUOUS = "continuous"

#: A process whose steps are indivisible: one step is one generation. ``dt`` is *not*
#: meaningful and passing one is an error, not a rounding.
TIME_DISCRETE = "discrete"

#: The two time models, in menu order, with their human labels.
TIME_MODELS: Tuple[Tuple[str, str], ...] = (
    (TIME_CONTINUOUS, "Continuous (integrated)"),
    (TIME_DISCRETE, "Discrete (generations)"),
)

#: One row per particle/cell/pixel; row count fixed by the parameters, not the frame.
LAYOUT_SNAPSHOT = "snapshot"

#: One row per frame so far; the accumulated trajectory is the picture. Row count grows
#: until :data:`MAX_TRAIL`, then the table scrolls.
LAYOUT_HISTORY = "history"

#: The categories, in menu order, with their human labels.
CATEGORIES: Tuple[Tuple[str, str], ...] = (
    ("mechanics", "Newtonian mechanics"),
    ("automaton", "Cellular automata"),
    ("grid", "Continuous grids"),
    ("fractal", "Fractals"),
)


class ProcessError(ValueError):
    """Raised for an unknown process, an impossible request, or a misused time model.

    A :class:`ValueError` subclass so panel actions that already catch ``ValueError``
    catch this too, matching :class:`glplot.gui.generators3d.GeneratorError` and
    :class:`glplot.gui.dynamics.DynamicsError`.
    """


@dataclass(frozen=True)
class ProcessParam:
    """One process parameter: its default and the slider bounds around it.

    Deliberately the same shape and the same ``clamp`` contract as
    :class:`glplot.gui.generators3d.Param3D`, so one piece of GUI code can render sliders
    for both catalogues. It is duplicated rather than imported because this package must
    not depend on :mod:`glplot.gui` -- the dependency would run the wrong way and would
    drag an import-heavy subpackage into a module whose whole value is being headless.
    """

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


@dataclass(frozen=True, eq=False)
class State:
    """One instant of a running process: which frame, what time, and the arrays.

    Attributes
    ----------
    frame
        Steps taken since :meth:`Process.initial_state`. Always increments by exactly one
        per :meth:`Process.step`, for both time models -- it is the frame counter a
        timeline scrubs, and it must not depend on the physics.
    time
        Simulation time. Seconds for a continuous process; a generation count for a
        discrete one, where it always equals ``float(frame)``. Reading it as seconds for a
        discrete process is the caller's error and the docstring above says so.
    data
        The process-specific arrays and scalars. Column-shaped entries are what
        :meth:`Process.state_to_table` exposes; everything else (a Life grid, the current
        Julia ``c``, an effective iteration count) lives here for callers that want the
        raw object rather than a table.

    ``eq=False`` on purpose: the default dataclass ``__eq__`` would compare ``data``
    dict-wise, and comparing two dicts of numpy arrays raises "truth value of an array is
    ambiguous". Identity comparison is both correct and what every caller actually wants.
    ``frozen=True`` prevents renumbering a frame by accident; the arrays inside ``data``
    are mutable but every ``advance`` in this module builds new ones rather than writing
    into the old, so treating them as read-only is safe and is the documented contract.
    """

    frame: int
    time: float
    data: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """``state[name]`` -- the named entry, with a message that names the alternatives."""
        try:
            return self.data[key]
        except KeyError:
            raise ProcessError(
                f"state has no entry {key!r} at frame {self.frame}; it holds: "
                f"{', '.join(sorted(self.data)) or '(nothing)'}."
            ) from None

    def get(self, key: str, default: Any = None) -> Any:
        """The named entry, or ``default`` when it is absent."""
        return self.data.get(key, default)

    def advanced(self, dt: float, data: Dict[str, Any]) -> "State":
        """The successor state: one frame on, ``dt`` further in time, holding ``data``."""
        return State(frame=self.frame + 1, time=self.time + float(dt), data=data)


@dataclass(frozen=True)
class Process:
    """One named simulation: what it is called, what it takes, and how it evolves.

    Attributes
    ----------
    key
        Stable machine identifier; the :data:`PROCESSES` dict key.
    label
        Human name, used for the dataset and the layer a frame produces.
    category
        One of :data:`CATEGORIES`' keys -- drives the picker's grouping.
    time_model
        :data:`TIME_CONTINUOUS` or :data:`TIME_DISCRETE`. See the module docstring; the
        distinction is enforced by :meth:`step`, not merely advertised.
    icon
        A :data:`glplot.gui.icons.ICON_SHAPES` name. Advisory: an unknown name draws a
        placeholder rather than raising, so this module never needs to import the GUI.
    description
        One line, shown as a tooltip.
    kind
        The :mod:`glplot.gui.layerops` plot kind a frame wants by default, so "animate"
        can be one click rather than an animate-then-choose-a-type sequence.
    columns
        The column names :meth:`state_to_table` returns, in order.
    plot_columns
        Which two of them are the geometry, as ``(x, y)``.
    setup
        ``params -> data``. Builds frame 0's arrays. Pure and reproducible: a process with
        randomness takes a ``seed`` parameter and uses nothing else.
    advance
        ``(state, params, dt) -> data``. One step. Pure: the same state and parameters give
        the same successor, every time. ``dt`` is the *frame* time for a continuous
        process (subdivided internally into :attr:`substeps`) and exactly ``0.0`` for a
        discrete one, where it must be ignored.
    tabulate
        ``(state, params) -> {column: array}``. Cheap by design: everything expensive is
        done once in ``advance`` and cached in ``state.data``, so a caller may re-tabulate
        a frame -- to re-colour it, say -- without recomputing the physics.
    params
        Parameters with slider bounds.
    dt
        Default frame time for a continuous process. Exactly ``0.0`` for a discrete one,
        checked at construction.
    substeps
        Integrator sub-steps per frame, for continuous processes. ``dt`` stays the frame
        time; the integrator uses ``dt / substeps`` internally, so ``state.time`` is exact
        while accuracy is decoupled from the frame rate.
    color_column
        The column worth colour-mapping by default, or None for "use y".
    table_layout
        :data:`LAYOUT_SNAPSHOT` or :data:`LAYOUT_HISTORY`.
    energy
        ``(state, params) -> float`` for the mechanical systems, else None. Present so a
        caller (and the test suite) can check that an integrator is behaving: for the
        conservative systems this quantity must stay bounded forever, and a slow secular
        drift in it is the signature of the wrong integrator.
    """

    key: str
    label: str
    category: str
    time_model: str
    icon: str
    description: str
    kind: str
    columns: Tuple[str, ...]
    plot_columns: Tuple[str, str]
    setup: Callable[[Mapping[str, float]], Dict[str, Any]]
    advance: Callable[[State, Mapping[str, float], float], Dict[str, Any]]
    tabulate: Callable[[State, Mapping[str, float]], Dict[str, np.ndarray]]
    params: Tuple[ProcessParam, ...] = ()
    dt: float = 0.0
    substeps: int = 1
    color_column: Optional[str] = None
    table_layout: str = LAYOUT_SNAPSHOT
    energy: Optional[Callable[[State, Mapping[str, float]], float]] = None

    def __post_init__(self) -> None:
        """Reject a malformed catalogue entry at import time rather than at first click."""
        if self.time_model not in dict(TIME_MODELS):
            raise ProcessError(
                f"process {self.key!r} has time_model {self.time_model!r}; it must be "
                f"{TIME_CONTINUOUS!r} or {TIME_DISCRETE!r}."
            )
        if self.time_model == TIME_DISCRETE and self.dt != 0.0:
            raise ProcessError(
                f"process {self.key!r} is discrete but declares dt={self.dt!r}; a discrete "
                "process advances one generation per step and has no timestep. Set dt=0.0."
            )
        if self.time_model == TIME_CONTINUOUS and not (math.isfinite(self.dt) and self.dt > 0.0):
            raise ProcessError(
                f"process {self.key!r} is continuous but declares dt={self.dt!r}; a "
                "continuous process needs a finite, positive default timestep."
            )
        if self.table_layout not in (LAYOUT_SNAPSHOT, LAYOUT_HISTORY):
            raise ProcessError(
                f"process {self.key!r} has table_layout {self.table_layout!r}; it must be "
                f"{LAYOUT_SNAPSHOT!r} or {LAYOUT_HISTORY!r}."
            )
        if not 1 <= int(self.substeps) <= MAX_SUBSTEPS:
            raise ProcessError(
                f"process {self.key!r} asks for {self.substeps} sub-steps; the range is "
                f"1..{MAX_SUBSTEPS}."
            )
        missing = [c for c in self.plot_columns if c not in self.columns]
        if missing:
            raise ProcessError(
                f"process {self.key!r} plots {missing} which it does not produce; its "
                f"columns are {', '.join(self.columns)}."
            )
        if self.color_column is not None and self.color_column not in self.columns:
            raise ProcessError(
                f"process {self.key!r} colours by {self.color_column!r} which it does not "
                f"produce; its columns are {', '.join(self.columns)}."
            )
        for p in self.params:
            if not p.vmin <= p.default <= p.vmax:
                raise ProcessError(
                    f"process {self.key!r} parameter {p.name!r} defaults to {p.default} "
                    f"outside its own bounds [{p.vmin}, {p.vmax}]."
                )

    # -- introspection ---------------------------------------------------------

    @property
    def is_continuous(self) -> bool:
        """True when ``dt`` is meaningful for this process."""
        return self.time_model == TIME_CONTINUOUS

    @property
    def is_discrete(self) -> bool:
        """True when one step is one indivisible generation."""
        return self.time_model == TIME_DISCRETE

    def defaults(self) -> Dict[str, float]:
        """A fresh, mutable dict of this process's default parameter values."""
        return {p.name: p.default for p in self.params}

    def resolve(self, params: Optional[Mapping[str, float]]) -> Dict[str, float]:
        """Merge ``params`` over the defaults, clamping each to its slider bounds.

        Unknown keys are ignored rather than rejected, matching
        :meth:`glplot.gui.generators3d.Generator3D.resolve`: a panel that keeps one
        parameter dict per process and switches between them would otherwise raise every
        time the user changed process.
        """
        resolved = self.defaults()
        for p in self.params:
            if params and p.name in params:
                resolved[p.name] = p.clamp(params[p.name])
        return resolved

    def timestep(self, dt: Optional[float] = None) -> float:
        """The frame time one :meth:`step` will advance, honouring the time model.

        Always ``0.0`` for a discrete process, and asking for anything else is an error --
        that is the whole point of the distinction. For a continuous one, ``dt`` overrides
        the stored default and must be finite and non-zero (a negative ``dt`` is allowed:
        every integrator here is time-reversible enough to run an animation backwards).
        """
        if self.is_discrete:
            if dt is not None:
                raise ProcessError(
                    f"process {self.key!r} is discrete: one step is one generation and it "
                    f"has no timestep, but dt={dt!r} was given. Drop the dt argument; use "
                    "state.frame to count generations."
                )
            return 0.0
        value = float(self.dt if dt is None else dt)
        if not math.isfinite(value) or value == 0.0:
            raise ProcessError(
                f"process {self.key!r} needs a finite, non-zero dt (got {dt!r}); its "
                f"default is {self.dt}."
            )
        return value

    # -- running ---------------------------------------------------------------

    def initial_state(self, params: Optional[Mapping[str, float]] = None) -> State:
        """Frame 0. Reproducible from ``params`` alone -- randomness comes from ``seed``."""
        return State(frame=0, time=0.0, data=self.setup(self.resolve(params)))

    def step(
        self,
        state: State,
        params: Optional[Mapping[str, float]] = None,
        dt: Optional[float] = None,
        *,
        substeps: Optional[int] = None,
    ) -> State:
        """Advance one frame. The one entry point; ``advance`` is never called directly.

        ``substeps`` overrides the process's integrator subdivision for this call only --
        a timeline that is dropping frames can lower it to keep up, at a cost in accuracy
        that is visible in :meth:`total_energy`. Ignored by discrete processes, which have
        nothing to subdivide.
        """
        resolved = self.resolve(params)
        frame_dt = self.timestep(dt)
        if substeps is not None:
            n = int(np.clip(int(substeps), 1, MAX_SUBSTEPS))
            resolved["_substeps"] = float(n)
        else:
            resolved["_substeps"] = float(self.substeps)
        return state.advanced(
            1.0 if self.is_discrete else frame_dt, self.advance(state, resolved, frame_dt)
        )

    def state_to_table(
        self, state: State, params: Optional[Mapping[str, float]] = None
    ) -> Dict[str, np.ndarray]:
        """This frame as an ordinary table of equal-length float64 columns.

        The output lands in the DataStore like a pasted CSV: nothing about an animation
        frame is special once it is a table, which is what lets every existing layer kind,
        colormap and transform apply to it without knowing a process produced it.
        """
        table = self.tabulate(state, self.resolve(params))
        if len({len(v) for v in table.values()}) > 1:
            shapes = ", ".join(f"{name}={len(column)}" for name, column in table.items())
            raise ProcessError(
                f"process {self.key!r} produced ragged columns at frame {state.frame} "
                f"({shapes}); every column of one frame must have the same length. This "
                "is a bug in the process, not in the request."
            )
        return table

    def total_energy(
        self, state: State, params: Optional[Mapping[str, float]] = None
    ) -> Optional[float]:
        """The conserved quantity for a mechanical system, or None where there is none.

        For the symplectic systems (:data:`NBODY`, :data:`TWO_BODY`, :data:`SPRING_CHAIN`,
        :data:`WAVE2D`) this oscillates within a bounded band forever and never drifts. For
        the RK4 systems it drifts slowly, at fourth order in the sub-step. Either way it is
        the number that tells you whether the integration is still meaningful.
        """
        if self.energy is None:
            return None
        return float(self.energy(state, self.resolve(params)))

    def frames(
        self,
        params: Optional[Mapping[str, float]] = None,
        count: int = 100,
        dt: Optional[float] = None,
        *,
        substeps: Optional[int] = None,
    ) -> Iterator[State]:
        """Yield ``count + 1`` states, starting at frame 0. Lazy, so it costs one frame.

        Raises :class:`ProcessError` above :data:`MAX_FRAMES` -- an unbounded run is an
        unbounded memory request, and a caller that wants one should drive :meth:`step`
        itself and discard what it has finished with.
        """
        n = int(count)
        if not 0 <= n <= MAX_FRAMES:
            raise ProcessError(f"count must be between 0 and {MAX_FRAMES:,} (got {count}).")
        resolved = self.resolve(params)
        state = self.initial_state(resolved)
        yield state
        for _ in range(n):
            state = self.step(state, resolved, dt, substeps=substeps)
            yield state

    def run(
        self,
        params: Optional[Mapping[str, float]] = None,
        count: int = 100,
        dt: Optional[float] = None,
        *,
        substeps: Optional[int] = None,
    ) -> List[State]:
        """:meth:`frames` collected into a list. Convenience for tests and offline export."""
        return list(self.frames(params, count, dt, substeps=substeps))


# ----------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------


def _rng(seed: float, *stream: float) -> np.random.Generator:
    """A generator seeded from ``(seed, *stream)``, so every draw is reproducible.

    ``stream`` is how a *step* gets randomness without carrying a mutable generator
    through the state: a process passes its frame index and gets an independent,
    reproducible stream for that frame. Threading a live ``np.random.Generator`` through
    ``State`` was the alternative and was rejected -- it would make a state non-copyable
    in any useful sense and would make "re-run frame 900" mean something different from
    "step to frame 900".
    """
    return np.random.default_rng([int(seed) & 0x7FFF_FFFF, *(int(s) & 0x7FFF_FFFF for s in stream)])


def _substeps(p: Mapping[str, float]) -> int:
    """The sub-step count :meth:`Process.step` stashed in the resolved parameters."""
    return max(1, int(p.get("_substeps", 1.0)))


def _grid_shape(p: Mapping[str, float]) -> Tuple[int, int]:
    """``(height, width)`` for a grid process, clamped to the cell ceilings.

    Both sides are clamped to :data:`MIN_GRID_SIDE`..:data:`MAX_GRID_SIDE` first, then the
    height is lowered if the product would exceed :data:`MAX_CELLS`. Clamping rather than
    raising because these arrive from sliders, and a slider that throws is a slider that
    cannot be dragged past its neighbour's limit.
    """
    w = int(np.clip(int(p.get("width", 128.0)), MIN_GRID_SIDE, MAX_GRID_SIDE))
    h = int(np.clip(int(p.get("height", w)), MIN_GRID_SIDE, MAX_GRID_SIDE))
    if w * h > MAX_CELLS:
        h = max(MIN_GRID_SIDE, MAX_CELLS // w)
    return h, w


def _grid_coords(height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """Flat ``(x, y)`` cell centres, row-major, with row 0 at the *top* of the picture.

    Grids are stored with row 0 first, as every array is; plots have y increasing upward.
    Flipping here rather than in each process is what stops half the catalogue drawing
    upside down.
    """
    xs = np.arange(width, dtype=np.float64)
    ys = np.arange(height - 1, -1, -1, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    return gx.ravel(), gy.ravel()


def _append_history(
    old: Mapping[str, np.ndarray], new: Mapping[str, float], limit: int = MAX_TRAIL
) -> Dict[str, np.ndarray]:
    """Append one row to a history table, scrolling once it reaches ``limit`` rows.

    Concatenating a whole table per frame is O(rows) per step and so O(rows^2) over a run,
    which for the 20000-row ceiling is 4e8 float copies spread over 20000 frames -- about
    20 microseconds a frame, below the noise of everything else in a frame. A growing
    doubling buffer would be faster and would make :class:`State` mutable in a way that
    breaks "re-tabulate an old frame"; the trade is not worth it.
    """
    out: Dict[str, np.ndarray] = {}
    for name, column in old.items():
        joined = np.concatenate([column, np.asarray([new[name]], dtype=np.float64)])
        out[name] = joined[-limit:] if joined.size > limit else joined
    return out


def _verlet(
    pos: np.ndarray,
    vel: np.ndarray,
    accel: Callable[[np.ndarray], np.ndarray],
    dt: float,
    substeps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Velocity-Verlet over one frame, subdivided into ``substeps`` sub-steps.

    **Why velocity-Verlet and not Euler or RK4.** Verlet is symplectic and time-reversible,
    so it conserves a *modified* Hamiltonian exactly; the true energy then oscillates
    inside a band of width O(h^2) and never leaves it, however long the run. Forward Euler
    does the opposite: it injects energy at every step, and a circular orbit integrated
    with it visibly spirals outward within a few hundred steps. That is not a cosmetic
    difference for this module -- an orbit demo whose orbit unwinds is simply wrong, and a
    user watching it learns something false. RK4 is fourth-order accurate but not
    symplectic, so its energy error is small yet *secular*: it creeps in one direction
    forever, and it costs four force evaluations per step instead of one.

    ``accel`` must depend on position only. A velocity-dependent force (drag, damping)
    breaks the splitting Verlet is built on; those systems use :func:`_rk4` instead, which
    costs nothing in principle because they are not conservative anyway.
    """
    h = dt / substeps
    a = accel(pos)
    for _ in range(substeps):
        vel = vel + 0.5 * h * a
        pos = pos + h * vel
        a = accel(pos)
        vel = vel + 0.5 * h * a
    return pos, vel


def _rk4(
    y: np.ndarray,
    t: float,
    deriv: Callable[[float, np.ndarray], np.ndarray],
    dt: float,
    substeps: int,
) -> Tuple[np.ndarray, float]:
    """Classical RK4 over one frame, subdivided into ``substeps`` sub-steps.

    Used for the systems Verlet cannot take: velocity-dependent forces (the projectile's
    drag, the pendulum's damping), explicit time dependence (the pendulum's drive) and the
    double pendulum, whose kinetic energy couples the two angular momenta through an
    angle-dependent mass matrix and so is *not separable* -- there is no explicit leapfrog
    for it. Returns the new state and the new time, both exact to the sub-step.
    """
    h = dt / substeps
    for _ in range(substeps):
        k1 = deriv(t, y)
        k2 = deriv(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = deriv(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = deriv(t + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
    return y, t


def escape_budget(pixels: int, max_iter: int) -> int:
    """The iteration count an escape-time raster of ``pixels`` may actually afford.

    :data:`MAX_ESCAPE_WORK` is a ceiling on *pixels times iterations*, so something has to
    give when a caller asks for both a large raster and a deep budget. Iterations give,
    not resolution: a lower-resolution image is visibly worse everywhere, while a lower
    iteration count only mislabels the pixels nearest the boundary as "inside". The value
    used is reported in ``state.data["max_iter"]`` so a GUI can say so.
    """
    n = max(1, int(pixels))
    want = int(np.clip(int(max_iter), 1, MAX_ITERATIONS))
    return max(1, min(want, MAX_ESCAPE_WORK // n))


def _escape_time(z0: np.ndarray, c: np.ndarray, max_iter: int) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised escape-time iteration of ``z -> z^2 + c``. Returns ``(count, smooth)``.

    Both inputs are broadcast against each other, which is what lets one function serve
    both fractals: Mandelbrot passes ``z0 = 0`` with ``c`` the pixel grid, Julia passes
    ``z0`` the pixel grid with ``c`` a scalar.

    The loop compacts its active set every iteration -- escaped pixels are dropped from
    the arrays rather than masked in place. That is the difference between paying
    ``pixels * max_iter`` always and paying it only where the set actually is: on a
    whole-set view most pixels leave in under ten iterations, and the measured frame cost
    is 3-5x lower than the naive version. On a deep zoom into the set almost nothing
    escapes and the compaction buys nothing, which is exactly the worst case the module
    docstring quotes.

    ``count`` is the iteration at which the orbit left the disc of radius
    :data:`ESCAPE_RADIUS`, or ``max_iter`` for a point that never left (i.e. "inside", to
    the accuracy of the budget). ``smooth`` is the standard normalised iteration count
    ``n + 1 - log2(log|z|)``, which removes the integer banding from a colour ramp; it is
    exact only in the limit of a large bailout radius, and at radius 4 the residual is
    well under one iteration.
    """
    shape = np.broadcast_shapes(np.shape(z0), np.shape(c))
    z = np.array(np.broadcast_to(np.asarray(z0, dtype=np.complex128), shape), copy=True).ravel()
    cc = np.array(np.broadcast_to(np.asarray(c, dtype=np.complex128), shape), copy=True).ravel()
    n = z.size

    iters = int(max_iter)
    count = np.full(n, float(iters), dtype=np.float64)
    smooth = np.full(n, float(iters), dtype=np.float64)
    alive = np.arange(n)
    radius2 = ESCAPE_RADIUS * ESCAPE_RADIUS

    with np.errstate(all="ignore"):
        for i in range(iters):
            z = z * z + cc
            mag2 = z.real * z.real + z.imag * z.imag
            escaped = mag2 > radius2
            if not escaped.any():
                continue
            hit = alive[escaped]
            count[hit] = float(i + 1)
            mag = np.sqrt(mag2[escaped])
            smooth[hit] = (
                (i + 1) + 1.0 - np.log(np.log(np.maximum(mag, 1.0 + 1e-12))) / math.log(2.0)
            )
            keep = ~escaped
            if not keep.any():
                break
            z = z[keep]
            cc = cc[keep]
            alive = alive[keep]

    np.clip(smooth, 0.0, float(iters), out=smooth)
    return count, smooth


def _raster_axes(
    center: Tuple[float, float], span: float, side: int
) -> Tuple[np.ndarray, np.ndarray]:
    """The real and imaginary sample axes of a square view, low to high."""
    half = 0.5 * span
    re = np.linspace(center[0] - half, center[0] + half, side)
    im = np.linspace(center[1] - half, center[1] + half, side)
    return re, im


def _raster_table(state: State, p: Mapping[str, float]) -> Dict[str, np.ndarray]:
    """Unpack a cached escape-time raster into columns. Cheap: no arithmetic on pixels."""
    re, im = state["re"], state["im"]
    gx, gy = np.meshgrid(re, im)
    return {
        "x": gx.ravel(),
        "y": gy.ravel(),
        "iter": state["count"],
        "smooth": state["smooth"],
    }


# ----------------------------------------------------------------------------------
# Newtonian mechanics
# ----------------------------------------------------------------------------------


def _nbody_accel(pos: np.ndarray, mass: np.ndarray, g: float, soft: float) -> np.ndarray:
    """Softened pairwise gravity, O(N^2), fully vectorised.

    The softening ``eps`` is not a fudge: an exact 1/r^2 force is singular, and a close
    pass with any finite step ejects a body at absurd speed, which looks like a bug and is
    really a resolution failure. Softening replaces the potential with
    ``-G m_i m_j / sqrt(r^2 + eps^2)``, whose gradient is this force, so the *softened*
    energy in :func:`_nbody_energy` is the exact conserved quantity of what is integrated.
    Using the unsoftened potential in the energy check would show a spurious drift on every
    close encounter and would blame the integrator for it.
    """
    delta = pos[None, :, :] - pos[:, None, :]
    r2 = np.einsum("ijk,ijk->ij", delta, delta) + soft * soft
    inv = mass[None, :] / (r2 * np.sqrt(r2))
    np.fill_diagonal(inv, 0.0)
    return g * np.einsum("ij,ijk->ik", inv, delta)


def _nbody_energy(state: State, p: Mapping[str, float]) -> float:
    """Kinetic plus softened potential energy -- the quantity Verlet keeps bounded."""
    pos, vel, mass = state["pos"], state["vel"], state["mass"]
    kinetic = 0.5 * float(np.sum(mass * np.einsum("ij,ij->i", vel, vel)))
    delta = pos[None, :, :] - pos[:, None, :]
    r = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta) + p["softening"] ** 2)
    pair = mass[None, :] * mass[:, None] / r
    np.fill_diagonal(pair, 0.0)
    potential = -0.5 * p["G"] * float(np.sum(pair))
    return kinetic + potential


def _nbody_setup(p: Mapping[str, float]) -> Dict[str, Any]:
    n = int(np.clip(int(p["bodies"]), 2, MAX_BODIES))
    rng = _rng(p["seed"])
    mass = np.full(n, float(p["body_mass"]), dtype=np.float64)
    mass[0] = float(p["central_mass"])

    # A disc of satellites round a heavy centre, each launched at its local circular speed.
    # A uniform random cloud was the alternative and is a worse demo: it collapses, ejects
    # most of its members in the first few hundred steps, and shows nothing recognisable.
    angle = rng.uniform(0.0, 2.0 * np.pi, n)
    radius = float(p["radius"]) * np.sqrt(rng.uniform(0.15, 1.0, n))
    pos = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    pos[0] = 0.0
    speed = np.sqrt(p["G"] * mass[0] / np.maximum(radius, 1e-9)) * float(p["spin"])
    vel = np.column_stack([-speed * np.sin(angle), speed * np.cos(angle)])
    vel[0] = 0.0

    # Zero the total momentum so the cluster stays in frame instead of drifting off it.
    vel -= np.sum(mass[:, None] * vel, axis=0) / np.sum(mass)
    return {"pos": pos, "vel": vel, "mass": mass}


def _nbody_advance(state: State, p: Mapping[str, float], dt: float) -> Dict[str, Any]:
    mass = state["mass"]
    g, soft = float(p["G"]), float(p["softening"])
    pos, vel = _verlet(
        state["pos"],
        state["vel"],
        lambda q: _nbody_accel(q, mass, g, soft),
        dt,
        _substeps(p),
    )
    return {"pos": pos, "vel": vel, "mass": mass}


def _nbody_table(state: State, p: Mapping[str, float]) -> Dict[str, np.ndarray]:
    pos, vel, mass = state["pos"], state["vel"], state["mass"]
    return {
        "body": np.arange(pos.shape[0], dtype=np.float64),
        "x": pos[:, 0].copy(),
        "y": pos[:, 1].copy(),
        "vx": vel[:, 0].copy(),
        "vy": vel[:, 1].copy(),
        "speed": np.linalg.norm(vel, axis=1),
        "mass": mass.copy(),
    }


def _two_body_setup(p: Mapping[str, float]) -> Dict[str, Any]:
    """Start at apoapsis, where the eccentricity fixes the speed in one line.

    The vis-viva relation at apoapsis for a relative orbit of semi-major axis ``a`` and
    eccentricity ``e`` is ``v = sqrt(GM (1 - e) / (a (1 + e)))`` at ``r = a (1 + e)``.
    Starting there rather than at periapsis means the slowest, best-resolved part of the
    orbit is integrated first, and a run that is cut short still shows the ellipse.
    """
    ecc = float(np.clip(p["eccentricity"], 0.0, 0.95))
    a = float(p["semi_major"])
    gm = float(p["GM"])
    ratio = float(np.clip(p["mass_ratio"], 1e-3, 1.0))

    r = a * (1.0 + ecc)
    v = math.sqrt(max(gm * (1.0 - ecc) / (a * (1.0 + ecc)), 0.0))
    rel_pos = np.array([r, 0.0])
    rel_vel = np.array([0.0, v])

    total = 1.0 + ratio
    m1, m2 = 1.0 / total, ratio / total  # masses summing to 1; GM carries the scale
    pos = np.array([-m2 * rel_pos, m1 * rel_pos])
    vel = np.array([-m2 * rel_vel, m1 * rel_vel])
    mass = np.array([m1, m2])

    history = {
        "t": np.zeros(1),
        "x1": pos[0:1, 0].copy(),
        "y1": pos[0:1, 1].copy(),
        "x2": pos[1:2, 0].copy(),
        "y2": pos[1:2, 1].copy(),
        "r": np.array([float(np.linalg.norm(rel_pos))]),
    }
    return {"pos": pos, "vel": vel, "mass": mass, "history": history}


def _two_body_accel(pos: np.ndarray, mass: np.ndarray, gm: float, soft: float) -> np.ndarray:
    delta = pos[1] - pos[0]
    r2 = float(delta @ delta) + soft * soft
    scale = gm / (r2 * math.sqrt(r2))
    return np.array([mass[1] * scale * delta, -mass[0] * scale * delta])


def _two_body_advance(state: State, p: Mapping[str, float], dt: float) -> Dict[str, Any]:
    mass = state["mass"]
    gm, soft = float(p["GM"]), float(p["softening"])
    pos, vel = _verlet(
        state["pos"],
        state["vel"],
        lambda q: _two_body_accel(q, mass, gm, soft),
        dt,
        _substeps(p),
    )
    row = {
        "t": state.time + dt,
        "x1": pos[0, 0],
        "y1": pos[0, 1],
        "x2": pos[1, 0],
        "y2": pos[1, 1],
        "r": float(np.linalg.norm(pos[1] - pos[0])),
    }
    limit = int(np.clip(int(p["trail"]), 2, MAX_TRAIL))
    return {
        "pos": pos,
        "vel": vel,
        "mass": mass,
        "history": _append_history(state["history"], row, limit),
    }


def _two_body_energy(state: State, p: Mapping[str, float]) -> float:
    pos, vel, mass = state["pos"], state["vel"], state["mass"]
    kinetic = 0.5 * float(np.sum(mass * np.einsum("ij,ij->i", vel, vel)))
    delta = pos[1] - pos[0]
    r = math.sqrt(float(delta @ delta) + p["softening"] ** 2)
    return kinetic - p["GM"] * mass[0] * mass[1] / r


def two_body_period(params: Optional[Mapping[str, float]] = None) -> float:
    """Kepler's third law for :data:`TWO_BODY`: ``T = 2 pi sqrt(a^3 / GM)``.

    Exposed because it is the exact statement the integration can be checked against --
    after one period the relative position must return to where it started, and how far
    off it lands is a direct, physical measure of the integrator's error.
    """
    p = TWO_BODY.resolve(params)
    return 2.0 * math.pi * math.sqrt(p["semi_major"] ** 3 / max(p["GM"], 1e-12))


def _projectile_deriv(p: Mapping[str, float]) -> Callable[[float, np.ndarray], np.ndarray]:
    """State ``[x, y, vx, vy]``; quadratic drag on the velocity *relative to the wind*."""
    g = float(p["gravity"])
    k = float(p["drag"]) / max(float(p["mass"]), 1e-6)
    wind = float(p["wind"])

    def deriv(_t: float, s: np.ndarray) -> np.ndarray:
        vx, vy = s[2] - wind, s[3]
        speed = math.hypot(vx, vy)
        return np.array([s[2], s[3], -k * speed * vx, -g - k * speed * vy])

    return deriv


def _projectile_setup(p: Mapping[str, float]) -> Dict[str, Any]:
    theta = math.radians(float(p["angle"]))
    speed = float(p["speed"])
    state = np.array([0.0, float(p["height"]), speed * math.cos(theta), speed * math.sin(theta)])
    history = {
        "t": np.zeros(1),
        "x": state[0:1].copy(),
        "y": state[1:2].copy(),
        "vx": state[2:3].copy(),
        "vy": state[3:4].copy(),
        "speed": np.array([speed]),
    }
    return {"state": state, "history": history}


def _projectile_advance(state: State, p: Mapping[str, float], dt: float) -> Dict[str, Any]:
    """RK4 with the ground handled between sub-steps.

    The bounce is applied at sub-step boundaries, so the impact is resolved to ``dt /
    substeps`` and not exactly: the ball penetrates the ground by at most one sub-step's
    travel before it is reflected. Root-finding the crossing time would fix that and is
    not worth it here -- at the default 4 sub-steps of a 0.02 s frame the error is a
    millimetre at ballistic speeds, invisible in the plot and irrelevant to the shape of
    the arc. It is a real approximation and this is it, stated.

    A ball with restitution below 1 bounces infinitely often in finite time (the classic
    Zeno sequence), so a fixed-step integration would micro-bounce forever, jittering the
    trajectory by a fraction of a millimetre and never settling. The cutoff is physical
    rather than a magic number: once a rebound is too slow to survive one sub-step of free
    fall (``vy <= g h``), the ball is put at rest on the ground and stays there.
    """
    y = state["state"]
    deriv = _projectile_deriv(p)
    n = _substeps(p)
    h = dt / n
    t = state.time
    restitution = float(p["restitution"])
    at_rest = float(p["gravity"]) * abs(h)
    for _ in range(n):
        y, t = _rk4(y, t, deriv, h, 1)
        if y[1] < 0.0:
            y = np.array([y[0], -y[1] * restitution, y[2], -y[3] * restitution])
            if y[3] <= at_rest:
                y = np.array([y[0], 0.0, y[2], 0.0])
    row = {
        "t": state.time + dt,
        "x": y[0],
        "y": y[1],
        "vx": y[2],
        "vy": y[3],
        "speed": math.hypot(y[2], y[3]),
    }
    limit = int(np.clip(int(p["trail"]), 2, MAX_TRAIL))
    return {"state": y, "history": _append_history(state["history"], row, limit)}


def _pendulum_setup(p: Mapping[str, float]) -> Dict[str, Any]:
    y = np.array([math.radians(float(p["theta0"])), float(p["omega0"])])
    history = {
        "t": np.zeros(1),
        "theta": y[0:1].copy(),
        "omega": y[1:2].copy(),
        "x": np.array([float(p["length"]) * math.sin(y[0])]),
        "y": np.array([-float(p["length"]) * math.cos(y[0])]),
    }
    return {"state": y, "history": history}


def _pendulum_deriv(p: Mapping[str, float]) -> Callable[[float, np.ndarray], np.ndarray]:
    """``theta'' = -(g/L) sin(theta) - b theta' + A cos(omega t)`` as a first-order pair."""
    w2 = float(p["gravity"]) / max(float(p["length"]), 1e-6)
    damp = float(p["damping"])
    drive = float(p["drive"])
    omega = float(p["drive_freq"])

    def deriv(t: float, s: np.ndarray) -> np.ndarray:
        return np.array([s[1], -w2 * math.sin(s[0]) - damp * s[1] + drive * math.cos(omega * t)])

    return deriv


def _pendulum_advance(state: State, p: Mapping[str, float], dt: float) -> Dict[str, Any]:
    y, _t = _rk4(state["state"], state.time, _pendulum_deriv(p), dt, _substeps(p))
    length = float(p["length"])
    row = {
        "t": state.time + dt,
        "theta": y[0],
        "omega": y[1],
        "x": length * math.sin(y[0]),
        "y": -length * math.cos(y[0]),
    }
    limit = int(np.clip(int(p["trail"]), 2, MAX_TRAIL))
    return {"state": y, "history": _append_history(state["history"], row, limit)}


def _pendulum_energy(state: State, p: Mapping[str, float]) -> float:
    """``E = 1/2 m L^2 w^2 + m g L (1 - cos t)``, conserved only when undamped and undriven."""
    theta, omega = state["state"]
    m, length, g = float(p["mass"]), float(p["length"]), float(p["gravity"])
    return 0.5 * m * length * length * omega * omega + m * g * length * (1.0 - math.cos(theta))


def _double_deriv(p: Mapping[str, float]) -> Callable[[float, np.ndarray], np.ndarray]:
    """The exact double-pendulum equations of motion in ``[t1, t2, w1, w2]``.

    Written out rather than derived from a Lagrangian at runtime because these are the
    equations everyone checks against, and a symbolic derivation would add a dependency
    for no accuracy. The denominator ``2 m1 + m2 - m2 cos(2 t1 - 2 t2)`` is bounded below
    by ``2 m1`` and so never vanishes -- the system has no coordinate singularity.
    """
    m1, m2 = float(p["mass1"]), float(p["mass2"])
    l1, l2 = float(p["length1"]), float(p["length2"])
    g = float(p["gravity"])

    def deriv(_t: float, s: np.ndarray) -> np.ndarray:
        t1, t2, w1, w2 = s
        d = t1 - t2
        den = 2.0 * m1 + m2 - m2 * math.cos(2.0 * d)
        a1 = (
            -g * (2.0 * m1 + m2) * math.sin(t1)
            - m2 * g * math.sin(t1 - 2.0 * t2)
            - 2.0 * math.sin(d) * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * math.cos(d))
        ) / (l1 * den)
        a2 = (
            2.0
            * math.sin(d)
            * (
                w1 * w1 * l1 * (m1 + m2)
                + g * (m1 + m2) * math.cos(t1)
                + w2 * w2 * l2 * m2 * math.cos(d)
            )
        ) / (l2 * den)
        return np.array([w1, w2, a1, a2])

    return deriv


def _double_positions(s: np.ndarray, p: Mapping[str, float]) -> Tuple[float, float, float, float]:
    l1, l2 = float(p["length1"]), float(p["length2"])
    x1 = l1 * math.sin(s[0])
    y1 = -l1 * math.cos(s[0])
    return x1, y1, x1 + l2 * math.sin(s[1]), y1 - l2 * math.cos(s[1])


def _double_setup(p: Mapping[str, float]) -> Dict[str, Any]:
    y = np.array(
        [
            math.radians(float(p["theta1"])),
            math.radians(float(p["theta2"])),
            float(p["omega1"]),
            float(p["omega2"]),
        ]
    )
    x1, y1, x2, y2 = _double_positions(y, p)
    history = {
        "t": np.zeros(1),
        "theta1": y[0:1].copy(),
        "theta2": y[1:2].copy(),
        "x1": np.array([x1]),
        "y1": np.array([y1]),
        "x": np.array([x2]),
        "y": np.array([y2]),
    }
    return {"state": y, "history": history}


def _double_advance(state: State, p: Mapping[str, float], dt: float) -> Dict[str, Any]:
    y, _t = _rk4(state["state"], state.time, _double_deriv(p), dt, _substeps(p))
    x1, y1, x2, y2 = _double_positions(y, p)
    row = {
        "t": state.time + dt,
        "theta1": y[0],
        "theta2": y[1],
        "x1": x1,
        "y1": y1,
        "x": x2,
        "y": y2,
    }
    limit = int(np.clip(int(p["trail"]), 2, MAX_TRAIL))
    return {"state": y, "history": _append_history(state["history"], row, limit)}


def _double_energy(state: State, p: Mapping[str, float]) -> float:
    """Total energy of the double pendulum, with the cross term the coupling produces."""
    t1, t2, w1, w2 = state["state"]
    m1, m2 = float(p["mass1"]), float(p["mass2"])
    l1, l2 = float(p["length1"]), float(p["length2"])
    g = float(p["gravity"])
    kinetic = 0.5 * m1 * (l1 * w1) ** 2 + 0.5 * m2 * (
        (l1 * w1) ** 2 + (l2 * w2) ** 2 + 2.0 * l1 * l2 * w1 * w2 * math.cos(t1 - t2)
    )
    potential = -(m1 + m2) * g * l1 * math.cos(t1) - m2 * g * l2 * math.cos(t2)
    return kinetic + potential


def _chain_setup(p: Mapping[str, float]) -> Dict[str, Any]:
    """A string of ``n`` masses with fixed ends: either a plucked corner or a pure mode."""
    n = int(np.clip(int(p["nodes"]), 3, MAX_NODES))
    x = np.linspace(0.0, 1.0, n)
    mode = int(p["mode"])
    amp = float(p["amplitude"])
    if mode <= 0:
        # A pluck: a triangle peaked at `pluck`. Its corner is a discontinuity in slope,
        # which excites every mode at once -- the interesting case, and the one that shows
        # dispersion if the discretisation has any.
        at = float(np.clip(p["pluck"], 0.02, 0.98))
        y = amp * np.where(x <= at, x / at, (1.0 - x) / max(1.0 - at, 1e-6))
    else:
        y = amp * np.sin(mode * np.pi * x)
    y[0] = y[-1] = 0.0
    return {"x": x, "y": y, "vy": np.zeros(n)}


def _chain_accel(y: np.ndarray, k_over_m: float) -> np.ndarray:
    """The discrete Laplacian: this *is* the wave equation, one lattice spacing at a time."""
    a = np.zeros_like(y)
    a[1:-1] = k_over_m * (y[:-2] - 2.0 * y[1:-1] + y[2:])
    return a


def _chain_advance(state: State, p: Mapping[str, float], dt: float) -> Dict[str, Any]:
    """Verlet on the interior nodes, with damping applied as an exact exponential decay.

    Splitting the linear drag out of the Verlet step and integrating it exactly
    (``v *= exp(-c h)``) keeps the undamped case bit-for-bit symplectic, so the energy test
    for this process is testing the integrator and not the damping model. Folding ``-c v``
    into the Verlet kick instead would have made the ``damping = 0`` case indistinguishable
    but the general case first-order and slightly energy-generating at large ``c``.
    """
    k_over_m = float(p["stiffness"]) / max(float(p["mass"]), 1e-9)
    n = _substeps(p)
    y, vy = _verlet(state["y"], state["vy"], lambda q: _chain_accel(q, k_over_m), dt, n)
    damping = float(p["damping"])
    if damping:
        vy = vy * math.exp(-damping * dt)
    y[0] = y[-1] = 0.0
    vy[0] = vy[-1] = 0.0
    return {"x": state["x"], "y": y, "vy": vy}


def _chain_energy(state: State, p: Mapping[str, float]) -> float:
    """Kinetic plus the spring energy stored in every link."""
    y, vy = state["y"], state["vy"]
    m, k = float(p["mass"]), float(p["stiffness"])
    return 0.5 * m * float(np.sum(vy * vy)) + 0.5 * k * float(np.sum(np.diff(y) ** 2))


def _chain_table(state: State, p: Mapping[str, float]) -> Dict[str, np.ndarray]:
    x = state["x"]
    return {
        "i": np.arange(x.size, dtype=np.float64),
        "x": x.copy(),
        "y": state["y"].copy(),
        "vy": state["vy"].copy(),
    }


def _history_table(state: State, p: Mapping[str, float]) -> Dict[str, np.ndarray]:
    """The accumulated trajectory, newest row last. Shared by every history-layout process."""
    return {name: column.copy() for name, column in state["history"].items()}


# ----------------------------------------------------------------------------------
# Cellular automata
# ----------------------------------------------------------------------------------

#: Classic Life seeds as ASCII art, ``#`` alive. Named so a GUI can offer them by name and
#: so the test suite can assert their textbook behaviour: the blinker has period 2, the
#: block never changes, the glider reproduces itself translated by (1, 1) after four
#: generations, and the gun emits one glider every 30.
LIFE_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "block": ("##", "##"),
    "blinker": ("###",),
    "toad": (".###", "###."),
    "glider": (".#.", "..#", "###"),
    "lwss": ("#..#.", "....#", "#...#", ".####"),
    "r_pentomino": (".##", "##.", ".#."),
    "acorn": (".#.....", "...#...", "##..###"),
    "gosper_gun": (
        "........................#...........",
        "......................#.#...........",
        "............##......##............##",
        "...........#...#....##............##",
        "##........#.....#...##..............",
        "##........#...#.##....#.#...........",
        "..........#.....#.......#...........",
        "...........#...#....................",
        "............##......................",
    ),
    "random": (),  # filled from `density` and `seed` rather than from art
}

#: Seed names in menu order. The ``pattern`` parameter is an index into this tuple, which
#: is how a named choice fits the numeric-slider convention the rest of the catalogue uses.
LIFE_PATTERN_NAMES: Tuple[str, ...] = tuple(LIFE_PATTERNS)


def life_pattern_index(name: str) -> int:
    """The ``pattern`` parameter value that selects the named seed."""
    try:
        return LIFE_PATTERN_NAMES.index(str(name))
    except ValueError:
        raise ProcessError(
            f"unknown Life pattern {name!r}; expected one of " f"{', '.join(LIFE_PATTERN_NAMES)}."
        ) from None


def _pattern_array(name: str) -> np.ndarray:
    """One ASCII seed as a ``(rows, cols)`` uint8 grid, validated for a ragged drawing."""
    art = LIFE_PATTERNS[name]
    widths = {len(row) for row in art}
    if len(widths) > 1:  # pragma: no cover - a catalogue typo, caught at import by tests
        raise ProcessError(f"Life pattern {name!r} has ragged rows: widths {sorted(widths)}.")
    return np.array([[1 if ch == "#" else 0 for ch in row] for row in art], dtype=np.uint8)


def _place(grid: np.ndarray, block: np.ndarray, top: int, left: int, name: str) -> None:
    h, w = grid.shape
    bh, bw = block.shape
    if bh > h or bw > w:
        raise ProcessError(
            f"Life pattern {name!r} is {bw}x{bh} but the grid is {w}x{h}; raise width and "
            f"height to at least {bw}x{bh}, or choose a smaller pattern."
        )
    top = int(np.clip(top, 0, h - bh))
    left = int(np.clip(left, 0, w - bw))
    grid[top : top + bh, left : left + bw] = block


def _life_setup(p: Mapping[str, float]) -> Dict[str, Any]:
    h, w = _grid_shape(p)
    grid = np.zeros((h, w), dtype=np.uint8)
    name = LIFE_PATTERN_NAMES[int(np.clip(int(p["pattern"]), 0, len(LIFE_PATTERN_NAMES) - 1))]
    if name == "random":
        grid = (_rng(p["seed"]).random((h, w)) < float(p["density"])).astype(np.uint8)
    else:
        block = _pattern_array(name)
        # The gun needs empty space to its lower right to shoot into; everything else
        # reads best centred.
        if name == "gosper_gun":
            _place(grid, block, 1, 1, name)
        else:
            _place(grid, block, (h - block.shape[0]) // 2, (w - block.shape[1]) // 2, name)
    return {"grid": grid, "age": grid.astype(np.int32)}


def _neighbour_count(grid: np.ndarray, toroidal: bool) -> np.ndarray:
    """The eight-neighbour sum of every cell, vectorised, in one padded array.

    Padding with a wrapped border for the toroidal case and a dead border for the bounded
    one collapses the two topologies into a single code path -- eight slice adds, no
    Python loop over cells. At the 512x512 ceiling this is 0.8 ms. A per-cell loop at that
    size is about four seconds, which is not an animation.
    """
    pad = np.pad(grid.astype(np.int16), 1, mode="wrap" if toroidal else "constant")
    return (
        pad[:-2, :-2]
        + pad[:-2, 1:-1]
        + pad[:-2, 2:]
        + pad[1:-1, :-2]
        + pad[1:-1, 2:]
        + pad[2:, :-2]
        + pad[2:, 1:-1]
        + pad[2:, 2:]
    )


def _life_advance(state: State, p: Mapping[str, float], _dt: float) -> Dict[str, Any]:
    """B3/S23: born on exactly three neighbours, surviving on two or three."""
    grid = state["grid"]
    n = _neighbour_count(grid, bool(p["toroidal"]))
    alive = ((n == 3) | ((grid == 1) & (n == 2))).astype(np.uint8)
    age = np.where(alive == 1, state["age"] + 1, 0).astype(np.int32)
    return {"grid": alive, "age": age}


def _life_table(state: State, p: Mapping[str, float]) -> Dict[str, np.ndarray]:
    """The living cells only, as ``(x, y, age)``.

    Dead cells are not emitted: at the 512x512 ceiling a full grid is 262144 rows of which
    typically under 10% are alive, and a scatter of the living cells is both the cheaper
    table and the picture people expect. The complete grid stays available as
    ``state.data["grid"]`` for a caller that wants to render it as an image.
    """
    grid = state["grid"]
    rows, cols = np.nonzero(grid)
    return {
        "x": cols.astype(np.float64),
        "y": (grid.shape[0] - 1 - rows).astype(np.float64),
        "age": state["age"][rows, cols].astype(np.float64),
    }


def _ca_setup(p: Mapping[str, float]) -> Dict[str, Any]:
    width = int(np.clip(int(p["width"]), MIN_GRID_SIDE, MAX_GRID_SIDE))
    row = np.zeros(width, dtype=np.uint8)
    if int(p["init"]) <= 0:
        row[width // 2] = 1  # the single seed cell that makes rule 30 a triangle
    else:
        row = (_rng(p["seed"]).random(width) < float(p["density"])).astype(np.uint8)
    return {"row": row, "history": row[None, :].copy(), "origin": 0}


def _ca_advance(state: State, p: Mapping[str, float], _dt: float) -> Dict[str, Any]:
    """One generation of a Wolfram elementary rule, as an eight-entry table lookup.

    The rule number *is* its truth table: bit ``4L + 2C + R`` of the number gives the new
    centre cell for that neighbourhood. Indexing a length-8 array with the packed
    neighbourhood does the whole row at once.
    """
    row = state["row"]
    rule = int(np.clip(int(p["rule"]), 0, 255))
    table = ((rule >> np.arange(8, dtype=np.uint8)) & 1).astype(np.uint8)
    if bool(p["toroidal"]):
        left, right = np.roll(row, 1), np.roll(row, -1)
    else:
        left = np.concatenate([[0], row[:-1]]).astype(np.uint8)
        right = np.concatenate([row[1:], [0]]).astype(np.uint8)
    new = table[(left << 2) | (row << 1) | right]

    limit = int(np.clip(int(p["rows"]), 2, MAX_CA_ROWS))
    history = np.concatenate([state["history"], new[None, :]])
    origin = int(state["origin"])
    if history.shape[0] > limit:
        origin += history.shape[0] - limit
        history = history[-limit:]
    return {"row": new, "history": history, "origin": origin}


def _ca_table(state: State, p: Mapping[str, float]) -> Dict[str, np.ndarray]:
    """The live cells of the space-time diagram: ``x`` across, ``y`` downward in time.

    ``y`` uses the absolute generation number, not the buffer row, so the picture keeps
    scrolling smoothly once the history has filled and starts dropping its oldest rows.
    """
    history = state["history"]
    rows, cols = np.nonzero(history)
    return {
        "x": cols.astype(np.float64),
        "y": -(rows + int(state["origin"])).astype(np.float64),
        "gen": (rows + int(state["origin"])).astype(np.float64),
    }


# ----------------------------------------------------------------------------------
# Continuous grids
# ----------------------------------------------------------------------------------


def _laplacian(a: np.ndarray) -> np.ndarray:
    """The periodic five-point Laplacian at unit grid spacing, four rolls and a subtract."""
    return (
        np.roll(a, 1, axis=0)
        + np.roll(a, -1, axis=0)
        + np.roll(a, 1, axis=1)
        + np.roll(a, -1, axis=1)
        - 4.0 * a
    )


def _gray_scott_setup(p: Mapping[str, float]) -> Dict[str, Any]:
    h, w = _grid_shape(p)
    u = np.ones((h, w), dtype=np.float64)
    v = np.zeros((h, w), dtype=np.float64)
    cy, cx = h // 2, w // 2
    r = int(np.clip(int(p["seed_size"]), 1, max(1, min(cy, cx))))
    u[cy - r : cy + r, cx - r : cx + r] = 0.50
    v[cy - r : cy + r, cx - r : cx + r] = 0.25
    # A symmetric seed on a symmetric grid stays symmetric forever, which makes a dull
    # picture; a little noise is what lets the pattern actually break and grow.
    rng = _rng(p["seed"])
    u += 0.01 * rng.standard_normal((h, w))
    v += 0.01 * rng.standard_normal((h, w))
    return {"u": np.clip(u, 0.0, 1.0), "v": np.clip(v, 0.0, 1.0)}


def _gray_scott_advance(state: State, p: Mapping[str, float], dt: float) -> Dict[str, Any]:
    """Explicit Euler on the Gray-Scott reaction-diffusion system.

    ``du/dt = Du lap(u) - u v^2 + F (1 - u)`` and ``dv/dt = Dv lap(v) + u v^2 - (F + k) v``.

    Explicit Euler on a five-point Laplacian at unit spacing has amplification factor
    ``1 + dt D k`` for a mode of Laplacian eigenvalue ``k``, and ``k`` reaches ``-8`` at
    the checkerboard mode -- so the scheme needs ``dt * max(Du, Dv) <= 0.25`` and diverges
    above it. The sliders stop at **0.24, not 0.25**, and the difference is not
    superstition: at exactly 0.25 the checkerboard's amplification is ``-1``, meaning that
    mode neither grows nor decays but flips sign forever. Measured over 40 steps of pure
    diffusion, a checkerboard of unit amplitude is still exactly 1.0 at 0.25, is 0.036 at
    0.24, and is 22 and climbing at 0.26. A permanently ringing pixel grid is not
    "stable"; 0.24 makes every mode strictly decay, and with the default ``dt = 1`` the
    whole slider range is then safe by construction.

    A caller that raises ``dt`` above 1 is outside that guarantee and the result is inf,
    deliberately not clipped -- silently clamping an unstable integration would hide the
    fact that the numbers stopped meaning anything.

    With ``feed = kill = 0`` the reaction only moves mass from ``u`` to ``v`` and the
    Laplacian sums to zero on a periodic grid, so ``sum(u + v)`` is then conserved exactly.
    That is the sharpest available check on this stepper and the test suite asserts it.
    """
    u, v = state["u"], state["v"]
    du, dv = float(p["Du"]), float(p["Dv"])
    feed, kill = float(p["feed"]), float(p["kill"])
    n = _substeps(p)
    h = dt / n
    with np.errstate(all="ignore"):
        for _ in range(n):
            uvv = u * v * v
            u, v = (
                u + h * (du * _laplacian(u) - uvv + feed * (1.0 - u)),
                v + h * (dv * _laplacian(v) + uvv - (feed + kill) * v),
            )
    return {"u": u, "v": v}


def _gray_scott_table(state: State, p: Mapping[str, float]) -> Dict[str, np.ndarray]:
    u = state["u"]
    gx, gy = _grid_coords(*u.shape)
    return {"x": gx, "y": gy, "u": u.ravel().copy(), "v": state["v"].ravel().copy()}


def _wave2d_setup(p: Mapping[str, float]) -> Dict[str, Any]:
    h, w = _grid_shape(p)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    sigma = max(float(p["width_sigma"]), 0.5)
    drops = max(1, int(p["drops"]))
    rng = _rng(p["seed"])
    u = np.zeros((h, w), dtype=np.float64)
    for i in range(drops):
        cy = h / 2.0 if drops == 1 else rng.uniform(0.2 * h, 0.8 * h)
        cx = w / 2.0 if drops == 1 else rng.uniform(0.2 * w, 0.8 * w)
        u += float(p["amplitude"]) * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)
        )
    return {"u": u, "v": np.zeros((h, w), dtype=np.float64)}


def _wave2d_advance(state: State, p: Mapping[str, float], dt: float) -> Dict[str, Any]:
    """Symplectic (semi-implicit) Euler on ``u_tt = c^2 lap(u) - damping * u_t``.

    Kicking the velocity with the current position and then drifting the position with the
    *new* velocity is the leapfrog/Verlet scheme in disguise, so with ``damping = 0`` the
    discrete energy stays in a bounded band instead of growing -- which explicit Euler,
    written the other way round, does not: it amplifies every mode and the surface
    explodes. The CFL condition here is ``c * dt <= 1/sqrt(2) ~ 0.707`` at unit spacing;
    the ``c`` slider stops at 2.5 and the default ``dt`` is 0.25, so the product is at most
    0.625 and the scheme is stable across the whole slider range by construction.
    """
    u, v = state["u"], state["v"]
    c2 = float(p["speed"]) ** 2
    damping = float(p["damping"])
    n = _substeps(p)
    h = dt / n
    for _ in range(n):
        v = v + h * c2 * _laplacian(u)
        if damping:
            v = v * math.exp(-damping * h)
        u = u + h * v
    return {"u": u, "v": v}


def _wave2d_energy(state: State, p: Mapping[str, float]) -> float:
    """``E = 1/2 sum(v^2) + 1/2 c^2 sum(|grad u|^2)`` on the periodic lattice."""
    u, v = state["u"], state["v"]
    c2 = float(p["speed"]) ** 2
    gx = np.roll(u, -1, axis=1) - u
    gy = np.roll(u, -1, axis=0) - u
    return 0.5 * float(np.sum(v * v)) + 0.5 * c2 * float(np.sum(gx * gx + gy * gy))


def _wave2d_table(state: State, p: Mapping[str, float]) -> Dict[str, np.ndarray]:
    u = state["u"]
    gx, gy = _grid_coords(*u.shape)
    return {"x": gx, "y": gy, "u": u.ravel().copy(), "v": state["v"].ravel().copy()}


# ----------------------------------------------------------------------------------
# Fractals
# ----------------------------------------------------------------------------------


def _mandelbrot_raster(
    center: Tuple[float, float], span: float, p: Mapping[str, float]
) -> Dict[str, Any]:
    side = int(np.clip(int(p["side"]), 16, MAX_GRID_SIDE))
    iters = escape_budget(side * side, int(p["max_iter"]))
    re, im = _raster_axes(center, span, side)
    c = re[None, :] + 1j * im[:, None]
    count, smooth = _escape_time(np.complex128(0.0), c, iters)
    return {
        "re": re,
        "im": im,
        "count": count,
        "smooth": smooth,
        "span": float(span),
        "side": side,
        "max_iter": iters,
    }


def _mandelbrot_center(p: Mapping[str, float]) -> Tuple[float, float]:
    return float(p["center_re"]), float(p["center_im"])


def _mandelbrot_span(p: Mapping[str, float], t: float) -> float:
    """``span(t) = span0 * exp(-rate * t)``, floored at :data:`MIN_SPAN`.

    Exponential rather than linear because a zoom is multiplicative: a linear span makes
    the first frames crawl and the last frames jump by factors of ten. Making the span a
    closed-form function of *time* rather than an accumulated per-step multiplication is
    what makes this process seekable -- "give me frame 400" costs one exponential, not 400
    steps -- and it is why the zoom is a continuous-time process even though the picture
    it draws is a discrete raster.
    """
    span = float(p["span0"]) * math.exp(-float(p["rate"]) * float(t))
    return max(span, MIN_SPAN)


def _mandelbrot_setup(p: Mapping[str, float]) -> Dict[str, Any]:
    return _mandelbrot_raster(_mandelbrot_center(p), _mandelbrot_span(p, 0.0), p)


def _mandelbrot_advance(state: State, p: Mapping[str, float], dt: float) -> Dict[str, Any]:
    return _mandelbrot_raster(_mandelbrot_center(p), _mandelbrot_span(p, state.time + dt), p)


def mandelbrot_frame(t: float, params: Optional[Mapping[str, float]] = None) -> State:
    """The zoom at time ``t`` directly, without stepping there.

    The parameterisation the animation actually needs: a timeline that is scrubbed, or a
    render job that farms frames out, asks for frame ``t`` and gets it in one raster's
    worth of work. Possible only because the zoom's span is a closed-form function of time
    -- see :func:`_mandelbrot_span` -- and it is why that was chosen over accumulating a
    per-frame multiplier.
    """
    p = MANDELBROT.resolve(params)
    dt = float(MANDELBROT.dt)
    return State(
        frame=int(round(float(t) / dt)) if dt else 0,
        time=float(t),
        data=_mandelbrot_raster(_mandelbrot_center(p), _mandelbrot_span(p, float(t)), p),
    )


#: Julia parameter paths, by the ``path`` parameter's index.
JULIA_PATHS: Tuple[str, ...] = ("circle", "cardioid", "rabbit_arc")


def julia_c(t: float, p: Mapping[str, float]) -> complex:
    """The Julia parameter ``c`` at time ``t``, for the selected path.

    * ``circle`` -- a circle of the given radius about ``(c_re, c_im)``. The general case.
    * ``cardioid`` -- the boundary of the Mandelbrot set's main cardioid,
      ``c = e^(i th) / 2 - e^(2 i th) / 4``. Every point of it is a parabolic parameter, so
      the Julia set stays connected and *just barely* so all the way round: this is the
      path along which the animation is most alive, because the set reorganises completely
      without ever shattering into dust.
    * ``rabbit_arc`` -- a short arc through Douady's rabbit at ``-0.123 + 0.745i``, for a
      slow morph around one famous set rather than a tour of all of them.
    """
    which = JULIA_PATHS[int(np.clip(int(p["path"]), 0, len(JULIA_PATHS) - 1))]
    theta = float(p["speed"]) * float(t)
    if which == "cardioid":
        return 0.5 * complex(math.cos(theta), math.sin(theta)) - 0.25 * complex(
            math.cos(2.0 * theta), math.sin(2.0 * theta)
        )
    if which == "rabbit_arc":
        base = complex(-0.123, 0.745)
        return base + 0.05 * complex(math.cos(theta), math.sin(theta))
    return complex(p["c_re"], p["c_im"]) + float(p["radius"]) * complex(
        math.cos(theta), math.sin(theta)
    )


def _julia_raster(t: float, p: Mapping[str, float]) -> Dict[str, Any]:
    side = int(np.clip(int(p["side"]), 16, MAX_GRID_SIDE))
    iters = escape_budget(side * side, int(p["max_iter"]))
    c = julia_c(t, p)
    re, im = _raster_axes((0.0, 0.0), 2.0 * float(p["extent"]), side)
    z0 = re[None, :] + 1j * im[:, None]
    count, smooth = _escape_time(z0, c, iters)
    return {
        "re": re,
        "im": im,
        "count": count,
        "smooth": smooth,
        "c": c,
        "side": side,
        "max_iter": iters,
    }


def _julia_setup(p: Mapping[str, float]) -> Dict[str, Any]:
    return _julia_raster(0.0, p)


def _julia_advance(state: State, p: Mapping[str, float], dt: float) -> Dict[str, Any]:
    return _julia_raster(state.time + dt, p)


def julia_frame(t: float, params: Optional[Mapping[str, float]] = None) -> State:
    """The Julia set at time ``t`` directly. Seekable, for the same reason as the zoom."""
    p = JULIA.resolve(params)
    dt = float(JULIA.dt)
    return State(
        frame=int(round(float(t) / dt)) if dt else 0,
        time=float(t),
        data=_julia_raster(float(t), p),
    )


#: The iterated function systems, as ``(matrices (k, 2, 2), offsets (k, 2), weights (k,))``.
#: Barnsley's fern is the one with the famous weights: the 1%-probability first map is the
#: whole stem, and giving it a fair share of the points would draw a stem and no fronds.
IFS_SYSTEMS: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {
    "fern": (
        np.array(
            [
                [[0.00, 0.00], [0.00, 0.16]],
                [[0.85, 0.04], [-0.04, 0.85]],
                [[0.20, -0.26], [0.23, 0.22]],
                [[-0.15, 0.28], [0.26, 0.24]],
            ]
        ),
        np.array([[0.0, 0.00], [0.0, 1.60], [0.0, 1.60], [0.0, 0.44]]),
        np.array([0.01, 0.85, 0.07, 0.07]),
    ),
    # Three half-scalings, one towards each vertex of the triangle (0, 0), (1, 0),
    # (0.5, 1) -- the offsets are half the vertex, which is what "move halfway to a
    # randomly chosen corner" means as an affine map.
    "sierpinski": (
        np.tile(0.5 * np.eye(2), (3, 1, 1)),
        np.array([[0.0, 0.0], [0.5, 0.0], [0.25, 0.5]]),
        np.full(3, 1.0 / 3.0),
    ),
    "dragon": (
        np.array([[[0.5, -0.5], [0.5, 0.5]], [[-0.5, 0.5], [-0.5, -0.5]]]),
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([0.5, 0.5]),
    ),
    "levy": (
        np.array([[[0.5, -0.5], [0.5, 0.5]], [[0.5, 0.5], [-0.5, 0.5]]]),
        np.array([[0.0, 0.0], [0.5, 0.5]]),
        np.array([0.5, 0.5]),
    ),
}

#: IFS names in menu order; the ``system`` parameter is an index into this tuple.
IFS_NAMES: Tuple[str, ...] = tuple(IFS_SYSTEMS)


def _chaos_chains(p: Mapping[str, float]) -> int:
    """Parallel chains per frame, clamped so ``chains * iters`` respects the frame ceiling."""
    chains = max(1, int(p["chains"]))
    iters = max(1, int(p["iters"]))
    return max(1, min(chains, MAX_POINTS_PER_FRAME // iters))


def _chaos_iterate(
    heads: np.ndarray,
    p: Mapping[str, float],
    rng: np.random.Generator,
    iters: int,
    record: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance every chain ``iters`` times, returning the heads and all points visited.

    ``record=False`` skips the point buffer entirely, for the burn-in, whose whole purpose
    is that its points are thrown away: at the ceiling that is 200 x 8192 positions never
    written rather than 26 MB allocated and dropped.

    **Why many chains and not one.** The chaos game is inherently sequential -- each point
    is an affine image of the one before -- so a single chain of 65536 points is 65536
    Python iterations, about 40 ms, and that is a frame gone. Running ``chains``
    independent chains in parallel makes each iteration one vectorised affine map over the
    whole ensemble, and the Python loop is over ``iters`` (single digits) instead. The
    attractor of an IFS is independent of the starting point, so the ensemble draws the
    same object; the only cost is that each chain needs its own burn-in, which
    :func:`_chaos_setup` pays once.
    """
    matrices, offsets, weights = IFS_SYSTEMS[
        IFS_NAMES[int(np.clip(int(p["system"]), 0, len(IFS_NAMES) - 1))]
    ]
    picks = rng.choice(len(weights), size=(iters, heads.shape[0]), p=weights)
    points = np.empty((iters, heads.shape[0], 2), dtype=np.float64) if record else None
    for i in range(iters):
        which = picks[i]
        heads = np.einsum("nij,nj->ni", matrices[which], heads) + offsets[which]
        if points is not None:
            points[i] = heads
    if points is None:
        return heads, np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64)
    return heads, points.reshape(-1, 2), picks.reshape(-1).astype(np.float64)


def _chaos_setup(p: Mapping[str, float]) -> Dict[str, Any]:
    chains = _chaos_chains(p)
    rng = _rng(p["seed"])
    heads = rng.uniform(-0.1, 0.1, size=(chains, 2))
    # Burn in before recording: the first few images of an arbitrary start are not on the
    # attractor and would show as a spray of stray points across the picture forever.
    burn = int(np.clip(int(p["burn"]), 0, 200))
    if burn:
        heads = _chaos_iterate(heads, p, rng, burn, record=False)[0]
    return {
        "heads": heads,
        "points": np.empty((0, 2), dtype=np.float64),
        "codes": np.empty(0, dtype=np.float64),
    }


def _chaos_advance(state: State, p: Mapping[str, float], _dt: float) -> Dict[str, Any]:
    iters = max(1, int(p["iters"]))
    rng = _rng(p["seed"], state.frame + 1)
    heads, points, codes = _chaos_iterate(state["heads"], p, rng, iters)
    stacked = np.concatenate([state["points"], points])
    all_codes = np.concatenate([state["codes"], codes])
    if stacked.shape[0] > MAX_ACCUM_POINTS:
        stacked = stacked[-MAX_ACCUM_POINTS:]
        all_codes = all_codes[-MAX_ACCUM_POINTS:]
    return {"heads": heads, "points": stacked, "codes": all_codes}


def _chaos_table(state: State, p: Mapping[str, float]) -> Dict[str, np.ndarray]:
    points = state["points"]
    return {"x": points[:, 0].copy(), "y": points[:, 1].copy(), "map": state["codes"].copy()}


# ----------------------------------------------------------------------------------
# The catalogue
# ----------------------------------------------------------------------------------


def _p(
    name: str, default: float, vmin: float, vmax: float, label: str = "", integer: bool = False
) -> ProcessParam:
    return ProcessParam(
        name=name, default=default, vmin=vmin, vmax=vmax, label=label or name, integer=integer
    )


NBODY = Process(
    key="nbody",
    label="N-body gravity",
    category="mechanics",
    time_model=TIME_CONTINUOUS,
    icon="orbit",
    description="Point masses under mutual gravity, integrated with velocity-Verlet so "
    "the orbits stay orbits instead of spiralling out.",
    kind="scatter",
    columns=("body", "x", "y", "vx", "vy", "speed", "mass"),
    plot_columns=("x", "y"),
    color_column="speed",
    setup=_nbody_setup,
    advance=_nbody_advance,
    tabulate=_nbody_table,
    energy=_nbody_energy,
    dt=0.02,
    substeps=4,
    params=(
        _p("bodies", 6.0, 2.0, float(MAX_BODIES), integer=True),
        _p("G", 1.0, 0.0, 5.0),
        _p("central_mass", 20.0, 0.1, 200.0),
        _p("body_mass", 1.0, 0.01, 20.0),
        _p("radius", 3.0, 0.2, 20.0),
        _p("spin", 1.0, 0.0, 2.0),
        # Softening sets the *stiffest timescale in the system* and so decides whether the
        # fixed step can resolve it at all. Measured over 5000 frames of the default disc,
        # whose closest approach is about 0.004: at softening 0.05 the energy band is 540%
        # of |E| -- the integration is meaningless -- at 0.1 it is 18%, and at 0.2 it is
        # 0.14% and stops growing. 0.2 is therefore the default, and the slider floor is
        # 0.01 rather than 0 so that "no softening" is at least a deliberate act. Lowering
        # it is legitimate and demands more sub-steps; ``total_energy`` is how you check.
        _p("softening", 0.2, 0.01, 2.0),
        _p("seed", 0.0, 0.0, 9999.0, integer=True),
    ),
)

TWO_BODY = Process(
    key="two_body",
    label="Two-body orbit",
    category="mechanics",
    time_model=TIME_CONTINUOUS,
    icon="orbit",
    description="A Kepler ellipse of settable eccentricity, both bodies about their "
    "barycentre. Period and shape are exact; see two_body_period().",
    kind="line",
    columns=("t", "x1", "y1", "x2", "y2", "r"),
    plot_columns=("x2", "y2"),
    color_column="t",
    setup=_two_body_setup,
    advance=_two_body_advance,
    tabulate=_history_table,
    energy=_two_body_energy,
    dt=0.05,
    substeps=16,
    table_layout=LAYOUT_HISTORY,
    params=(
        _p("eccentricity", 0.6, 0.0, 0.95),
        _p("semi_major", 1.0, 0.05, 20.0),
        _p("GM", 1.0, 0.01, 20.0),
        _p("mass_ratio", 0.1, 0.001, 1.0),
        _p("softening", 0.0, 0.0, 0.5),
        _p("trail", 2000.0, 2.0, float(MAX_TRAIL), integer=True),
    ),
)

PROJECTILE = Process(
    key="projectile",
    label="Projectile with drag",
    category="mechanics",
    time_model=TIME_CONTINUOUS,
    icon="chart_line",
    description="Ballistic flight with quadratic air drag, wind and a bouncing ground. "
    "Set drag to zero to recover the textbook parabola.",
    kind="line",
    columns=("t", "x", "y", "vx", "vy", "speed"),
    plot_columns=("x", "y"),
    color_column="speed",
    setup=_projectile_setup,
    advance=_projectile_advance,
    tabulate=_history_table,
    dt=0.02,
    substeps=4,
    table_layout=LAYOUT_HISTORY,
    params=(
        _p("speed", 30.0, 0.0, 200.0),
        _p("angle", 45.0, -90.0, 90.0),
        _p("height", 0.0, 0.0, 100.0),
        _p("gravity", 9.81, 0.0, 30.0),
        _p("drag", 0.02, 0.0, 1.0),
        _p("mass", 1.0, 0.01, 100.0),
        _p("wind", 0.0, -30.0, 30.0),
        _p("restitution", 0.5, 0.0, 1.0),
        _p("trail", 5000.0, 2.0, float(MAX_TRAIL), integer=True),
    ),
)

PENDULUM = Process(
    key="pendulum",
    label="Damped driven pendulum",
    category="mechanics",
    time_model=TIME_CONTINUOUS,
    icon="loop",
    description="A rigid pendulum with linear damping and a sinusoidal drive. Undamped "
    "and undriven it conserves energy; driven hard it goes chaotic.",
    kind="line",
    columns=("t", "theta", "omega", "x", "y"),
    plot_columns=("theta", "omega"),
    color_column="t",
    setup=_pendulum_setup,
    advance=_pendulum_advance,
    tabulate=_history_table,
    energy=_pendulum_energy,
    dt=0.02,
    substeps=4,
    table_layout=LAYOUT_HISTORY,
    params=(
        _p("theta0", 60.0, -180.0, 180.0),
        _p("omega0", 0.0, -20.0, 20.0),
        _p("length", 1.0, 0.05, 10.0),
        _p("mass", 1.0, 0.01, 100.0),
        _p("gravity", 9.81, 0.0, 30.0),
        _p("damping", 0.0, 0.0, 5.0),
        _p("drive", 0.0, 0.0, 20.0),
        _p("drive_freq", 2.0, 0.0, 20.0),
        _p("trail", 5000.0, 2.0, float(MAX_TRAIL), integer=True),
    ),
)

DOUBLE_PENDULUM = Process(
    key="double_pendulum",
    label="Double pendulum",
    category="mechanics",
    time_model=TIME_CONTINUOUS,
    icon="derivative",
    description="The canonical chaotic mechanism. Two identical runs a millidegree apart "
    "diverge completely within seconds.",
    kind="line",
    columns=("t", "theta1", "theta2", "x1", "y1", "x", "y"),
    plot_columns=("x", "y"),
    color_column="t",
    setup=_double_setup,
    advance=_double_advance,
    tabulate=_history_table,
    energy=_double_energy,
    dt=0.01,
    substeps=8,
    table_layout=LAYOUT_HISTORY,
    params=(
        _p("theta1", 120.0, -180.0, 180.0),
        _p("theta2", 100.0, -180.0, 180.0),
        _p("omega1", 0.0, -20.0, 20.0),
        _p("omega2", 0.0, -20.0, 20.0),
        _p("length1", 1.0, 0.05, 10.0),
        _p("length2", 1.0, 0.05, 10.0),
        _p("mass1", 1.0, 0.01, 100.0),
        _p("mass2", 1.0, 0.01, 100.0),
        _p("gravity", 9.81, 0.0, 30.0),
        _p("trail", 4000.0, 2.0, float(MAX_TRAIL), integer=True),
    ),
)

SPRING_CHAIN = Process(
    key="spring_chain",
    label="Spring chain (wave on a string)",
    category="mechanics",
    time_model=TIME_CONTINUOUS,
    icon="wave",
    description="Masses on springs with fixed ends -- the wave equation, one lattice "
    "spacing at a time. Pluck it or start it in a pure standing mode.",
    kind="line",
    columns=("i", "x", "y", "vy"),
    plot_columns=("x", "y"),
    color_column="vy",
    setup=_chain_setup,
    advance=_chain_advance,
    tabulate=_chain_table,
    energy=_chain_energy,
    dt=0.02,
    substeps=4,
    params=(
        _p("nodes", 128.0, 3.0, float(MAX_NODES), integer=True),
        _p("stiffness", 50.0, 0.1, 500.0),
        _p("mass", 1.0, 0.01, 10.0),
        _p("damping", 0.0, 0.0, 5.0),
        _p("mode", 0.0, 0.0, 8.0, integer=True),
        _p("pluck", 0.3, 0.02, 0.98),
        _p("amplitude", 0.1, -2.0, 2.0),
    ),
)

LIFE = Process(
    key="life",
    label="Conway's Game of Life",
    category="automaton",
    time_model=TIME_DISCRETE,
    icon="grid",
    description="B3/S23 on a grid, bounded or toroidal, from a classic seed or from "
    "random soup. One step is one generation; there is no timestep.",
    kind="scatter",
    columns=("x", "y", "age"),
    plot_columns=("x", "y"),
    color_column="age",
    setup=_life_setup,
    advance=_life_advance,
    tabulate=_life_table,
    params=(
        _p("width", 96.0, float(MIN_GRID_SIDE), float(MAX_GRID_SIDE), integer=True),
        _p("height", 96.0, float(MIN_GRID_SIDE), float(MAX_GRID_SIDE), integer=True),
        _p(
            "pattern",
            float(LIFE_PATTERN_NAMES.index("glider")),
            0.0,
            float(len(LIFE_PATTERN_NAMES) - 1),
            integer=True,
        ),
        _p("density", 0.3, 0.0, 1.0),
        _p("toroidal", 1.0, 0.0, 1.0, integer=True),
        _p("seed", 0.0, 0.0, 9999.0, integer=True),
    ),
)

ELEMENTARY_CA = Process(
    key="elementary_ca",
    label="Elementary cellular automaton",
    category="automaton",
    time_model=TIME_DISCRETE,
    icon="table",
    description="Wolfram's one-dimensional rules drawn as a space-time diagram. Rule 30 "
    "from a single cell is a standard randomness source; rule 110 is Turing-complete.",
    kind="scatter",
    columns=("x", "y", "gen"),
    plot_columns=("x", "y"),
    color_column="gen",
    setup=_ca_setup,
    advance=_ca_advance,
    tabulate=_ca_table,
    params=(
        _p("rule", 30.0, 0.0, 255.0, integer=True),
        _p("width", 256.0, float(MIN_GRID_SIDE), float(MAX_GRID_SIDE), integer=True),
        _p("rows", 256.0, 2.0, float(MAX_CA_ROWS), integer=True),
        _p("init", 0.0, 0.0, 1.0, integer=True),
        _p("density", 0.5, 0.0, 1.0),
        _p("toroidal", 1.0, 0.0, 1.0, integer=True),
        _p("seed", 0.0, 0.0, 9999.0, integer=True),
    ),
)

GRAY_SCOTT = Process(
    key="gray_scott",
    label="Gray-Scott reaction-diffusion",
    category="grid",
    time_model=TIME_CONTINUOUS,
    icon="noise",
    description="Two reacting, diffusing chemicals. Feed and kill rates select between "
    "spots, stripes, mitosis and coral; a PDE, so it has a real timestep.",
    kind="scatter",
    columns=("x", "y", "u", "v"),
    plot_columns=("x", "y"),
    color_column="v",
    setup=_gray_scott_setup,
    advance=_gray_scott_advance,
    tabulate=_gray_scott_table,
    dt=1.0,
    substeps=1,
    params=(
        _p("width", 128.0, float(MIN_GRID_SIDE), float(MAX_GRID_SIDE), integer=True),
        _p("height", 128.0, float(MIN_GRID_SIDE), float(MAX_GRID_SIDE), integer=True),
        _p("feed", 0.037, 0.0, 0.11),
        _p("kill", 0.06, 0.0, 0.075),
        # 0.24 rather than 0.25: see _gray_scott_advance -- at exactly 0.25 the
        # checkerboard mode is neutrally stable and rings forever instead of decaying.
        _p("Du", 0.16, 0.0, 0.24),
        _p("Dv", 0.08, 0.0, 0.24),
        _p("seed_size", 8.0, 1.0, 64.0, integer=True),
        _p("seed", 0.0, 0.0, 9999.0, integer=True),
    ),
)

WAVE2D = Process(
    key="wave2d",
    label="Wave equation on a grid",
    category="grid",
    time_model=TIME_CONTINUOUS,
    icon="wave",
    description="Ripples on a periodic sheet, stepped with symplectic Euler so the energy "
    "stays bounded. Stable across the whole speed slider by construction.",
    kind="scatter",
    columns=("x", "y", "u", "v"),
    plot_columns=("x", "y"),
    color_column="u",
    setup=_wave2d_setup,
    advance=_wave2d_advance,
    tabulate=_wave2d_table,
    energy=_wave2d_energy,
    dt=0.25,
    substeps=1,
    params=(
        _p("width", 128.0, float(MIN_GRID_SIDE), float(MAX_GRID_SIDE), integer=True),
        _p("height", 128.0, float(MIN_GRID_SIDE), float(MAX_GRID_SIDE), integer=True),
        _p("speed", 1.0, 0.0, 2.5),
        _p("damping", 0.0, 0.0, 2.0),
        _p("amplitude", 1.0, -5.0, 5.0),
        _p("width_sigma", 3.0, 0.5, 40.0),
        _p("drops", 1.0, 1.0, 12.0, integer=True),
        _p("seed", 0.0, 0.0, 9999.0, integer=True),
    ),
)

MANDELBROT = Process(
    key="mandelbrot_zoom",
    label="Mandelbrot zoom",
    category="fractal",
    time_model=TIME_CONTINUOUS,
    icon="zoom_in",
    description="A continuous exponential zoom towards a chosen point. Seekable: any "
    "frame costs one raster, not the frames before it. Floors at a span of 1e-12.",
    kind="scatter",
    columns=("x", "y", "iter", "smooth"),
    plot_columns=("x", "y"),
    color_column="smooth",
    setup=_mandelbrot_setup,
    advance=_mandelbrot_advance,
    tabulate=_raster_table,
    dt=1.0,
    substeps=1,
    params=(
        # The default centre is the seahorse-valley point every zoom video uses, because
        # it stays interesting for the whole 12 decades this can reach.
        _p("center_re", -0.743643887037151, -2.5, 1.5),
        _p("center_im", 0.13182590420533, -1.5, 1.5),
        _p("span0", 3.0, 1e-9, 4.0),
        _p("rate", 0.15, 0.0, 1.0),
        _p("side", 192.0, 16.0, float(MAX_GRID_SIDE), integer=True),
        _p("max_iter", 256.0, 16.0, float(MAX_ITERATIONS), integer=True),
    ),
)

JULIA = Process(
    key="julia_path",
    label="Julia set on a path",
    category="fractal",
    time_model=TIME_CONTINUOUS,
    icon="function",
    description="The filled Julia set of z^2 + c, with c tracing a path through parameter "
    "space. The cardioid path keeps the set connected all the way round.",
    kind="scatter",
    columns=("x", "y", "iter", "smooth"),
    plot_columns=("x", "y"),
    color_column="smooth",
    setup=_julia_setup,
    advance=_julia_advance,
    tabulate=_raster_table,
    dt=1.0,
    substeps=1,
    params=(
        _p("path", 1.0, 0.0, float(len(JULIA_PATHS) - 1), integer=True),
        _p("speed", 0.02, -1.0, 1.0),
        _p("radius", 0.1, 0.0, 1.5),
        _p("c_re", -0.4, -2.0, 2.0),
        _p("c_im", 0.6, -2.0, 2.0),
        _p("extent", 1.6, 0.1, 4.0),
        _p("side", 192.0, 16.0, float(MAX_GRID_SIDE), integer=True),
        _p("max_iter", 256.0, 16.0, float(MAX_ITERATIONS), integer=True),
    ),
)

CHAOS_GAME = Process(
    key="chaos_game",
    label="Chaos game (IFS attractor)",
    category="fractal",
    time_model=TIME_DISCRETE,
    icon="chart_scatter",
    description="An iterated function system drawn by accumulating random orbits -- the "
    "Barnsley fern and friends, filling in a little more every frame.",
    kind="scatter",
    columns=("x", "y", "map"),
    plot_columns=("x", "y"),
    color_column="map",
    setup=_chaos_setup,
    advance=_chaos_advance,
    tabulate=_chaos_table,
    params=(
        _p("system", 0.0, 0.0, float(len(IFS_NAMES) - 1), integer=True),
        _p("chains", 512.0, 1.0, float(MAX_POINTS_PER_FRAME), integer=True),
        _p("iters", 8.0, 1.0, 64.0, integer=True),
        _p("burn", 20.0, 0.0, 200.0, integer=True),
        _p("seed", 0.0, 0.0, 9999.0, integer=True),
    ),
)

#: The catalogue, keyed by :attr:`Process.key`, in display order.
PROCESSES: Dict[str, Process] = {
    process.key: process
    for process in (
        NBODY,
        TWO_BODY,
        PROJECTILE,
        PENDULUM,
        DOUBLE_PENDULUM,
        SPRING_CHAIN,
        LIFE,
        ELEMENTARY_CA,
        GRAY_SCOTT,
        WAVE2D,
        MANDELBROT,
        JULIA,
        CHAOS_GAME,
    )
}

#: Every process key, in catalogue order.
PROCESS_KEYS: Tuple[str, ...] = tuple(PROCESSES)


def process(key: str) -> Process:
    """The :class:`Process` named ``key``. Raises :class:`ProcessError` if unknown."""
    spec = PROCESSES.get(str(key).strip().lower())
    if spec is None:
        raise ProcessError(f"unknown process {key!r}; expected one of {', '.join(PROCESS_KEYS)}")
    return spec


def by_category(category: str) -> Tuple[Process, ...]:
    """Every process in ``category``, in catalogue order."""
    return tuple(pr for pr in PROCESSES.values() if pr.category == category)


def by_time_model(time_model: str) -> Tuple[Process, ...]:
    """Every process with the given time model, in catalogue order."""
    return tuple(pr for pr in PROCESSES.values() if pr.time_model == time_model)


def category_label(category: str) -> str:
    """The human label for a category key, or the key itself if it is unknown."""
    for key, label in CATEGORIES:
        if key == category:
            return label
    return category


def process_labels() -> List[Tuple[str, str]]:
    """``(key, label)`` for every process, in display order."""
    return [(pr.key, pr.label) for pr in PROCESSES.values()]
