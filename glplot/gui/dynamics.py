"""Dynamical-system data generation: integrate ODE trajectories into :class:`DataSet`.

Pure logic -- numpy, stdlib and the AST-hardened :mod:`glplot.gui.expressions` evaluator
only. No imgui, no engine imports (CONTRACT 5.1 rule 7). The Dynamics panel is a thin
shell over this module the same way the Functions panel is a shell over ``expressions``.

Where the Functions panel evaluates ``y = f(x)`` once, a dynamical system is a set of
coupled first-order ODEs ``dvar/dt = f(vars..., t, params...)`` that must be *integrated
forward in time* to produce a trajectory. This module owns three things:

* :class:`DynamicalSystem` -- a system as data: its state variables, its per-variable
  derivative expressions (strings, validated through the same allowlist the Functions
  panel uses), its parameters with slider bounds, and sensible defaults for the initial
  state and the integration window.
* :data:`SYSTEMS` -- ten classic preloaded systems (Lorenz, Goodwin, ...), each one just
  a :class:`DynamicalSystem` literal, so a custom system and a built-in are the same kind
  of object and run down the same integrator.
* :func:`integrate` -- a fixed-step RK4 integrator that returns a :class:`DataSet` whose
  first column is ``t`` and whose remaining columns are the state variables, ready to plot
  or hand to the DataStore.

**Why expression strings, not Python callables.** Storing each derivative as a string that
round-trips through :func:`glplot.gui.expressions.validate` means a *custom* system typed
into the panel is exactly as safe and exactly as fast as a built-in: neither can reach the
interpreter's builtins, both compile once and eval per step. It also makes the whole
library declarative -- a system is inspectable, copyable data with no code attached.

**Why a fixed-step RK4 and not scipy.** GLPlot's only hard dependency for the GUI is numpy;
pulling in scipy for one integrator is a poor trade. RK4 at a small step is more than
accurate enough to *draw* an attractor, which is what this is for -- not to certify a
Lyapunov exponent. Divergent or stiff parameter choices simply produce inf/nan tails,
which every downstream widget (``mini_plot``, the density renderer) already treats as gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import expressions
from .datasets import Column, DataSet
from .expressions import SAFE_NAMES, ExpressionError

#: Hard ceiling on the number of integration steps. A step is a handful of small ``eval``
#: calls; two million of them is a few seconds, which is the most a one-shot "Plot" action
#: should ever cost. The panel clamps to this and the preview uses far fewer.
MAX_STEPS = 2_000_000

#: Smallest usable step count -- below this there is no trajectory to speak of.
MIN_STEPS = 2

#: The reserved names an equation may reference on top of its own state variables and
#: parameters: ``t`` is the current time (for forced systems like Duffing).
_TIME_NAME = "t"


class DynamicsError(ValueError):
    """Raised for a malformed system or an out-of-range integration request.

    A subclass of :class:`ValueError` so callers that already catch ``(ExpressionError,
    ValueError)`` -- as every panel action does -- catch this too without a new arm.
    """


@dataclass(frozen=True)
class DynamicalSystem:
    """A system of first-order ODEs, stored as data.

    Attributes
    ----------
    key
        Stable machine identifier (``"lorenz"``); the :data:`SYSTEMS` dict key.
    label
        Human name for the layer/dataset it produces (``"Lorenz"``).
    icon
        An icon name from :data:`glplot.gui.icons.ICON_SHAPES` for the picker. Advisory:
        an unknown name draws a placeholder rather than raising.
    description
        One-line summary shown as a tooltip.
    variables
        The state variable names, in order. Each must be a Python identifier, distinct,
        not ``t``, and not a name in :data:`~glplot.gui.expressions.SAFE_NAMES` (a variable
        called ``e`` would silently shadow Euler's number in its own equations).
    equations
        One derivative expression per variable, in the same order: ``equations[i]`` is
        ``d(variables[i])/dt``. Each is a string in the Functions-panel language, free to
        reference any variable, any parameter, ``t``, and any ``SAFE_NAMES`` callable.
    params
        Parameters as ``(name, default, vmin, vmax)`` tuples -- the value and its slider
        bounds. A name here must not collide with a variable or with ``t``.
    initial
        Default initial state, one value per variable, in variable order.
    dt
        Default integration step.
    steps
        Default number of recorded steps (the trajectory has ``steps + 1`` rows).
    transient
        Default number of steps to integrate and *discard* before recording, to let the
        trajectory settle onto its attractor/limit cycle. Zero for conservative systems
        whose transient is the interesting part.
    plot
        Default ``(x, y)`` column pair for the phase portrait the panel draws. Either name
        may be ``t``. ``None`` falls back to the first two variables.
    """

    key: str
    label: str
    icon: str
    description: str
    variables: Tuple[str, ...]
    equations: Tuple[str, ...]
    params: Tuple[Tuple[str, float, float, float], ...] = ()
    initial: Tuple[float, ...] = ()
    dt: float = 0.01
    steps: int = 5000
    transient: int = 0
    plot: Optional[Tuple[str, str]] = None

    def __post_init__(self) -> None:
        self.validate()

    # -- introspection ---------------------------------------------------------

    def param_names(self) -> List[str]:
        """The parameter names, in declared order."""
        return [spec[0] for spec in self.params]

    def param_defaults(self) -> Dict[str, float]:
        """``name -> default`` for every parameter."""
        return {spec[0]: float(spec[1]) for spec in self.params}

    def allowed_names(self) -> set:
        """Every bare name an equation is permitted to reference besides ``SAFE_NAMES``."""
        return set(self.variables) | set(self.param_names()) | {_TIME_NAME}

    def plot_axes(self) -> Tuple[str, str]:
        """The default phase-portrait axes, always resolvable to real columns."""
        if self.plot is not None:
            return self.plot
        if len(self.variables) >= 2:
            return self.variables[0], self.variables[1]
        return _TIME_NAME, self.variables[0]

    def columns(self) -> List[str]:
        """The column names a trajectory carries: ``t`` then every variable."""
        return [_TIME_NAME, *self.variables]

    # -- validation ------------------------------------------------------------

    def validate(self) -> List[object]:
        """Prove the system is well-formed and return the compiled equation code objects.

        Raises :class:`DynamicsError` for a structural problem (mismatched lengths, a
        variable named ``t`` or shadowing a builtin, a bad initial-state length) and
        :class:`~glplot.gui.expressions.ExpressionError` for an equation that does not
        parse or references an unknown name. Compiling here -- once -- is what lets
        :func:`integrate` eval per step without re-validating.
        """
        if not self.variables:
            raise DynamicsError("a system needs at least one state variable.")
        if len(self.equations) != len(self.variables):
            raise DynamicsError(
                f"{len(self.variables)} variables but {len(self.equations)} equations; "
                "there must be exactly one derivative per variable."
            )
        seen: set = set()
        for name in self.variables:
            if not isinstance(name, str) or not name.isidentifier():
                raise DynamicsError(f"variable name {name!r} is not a valid identifier.")
            if name == _TIME_NAME:
                raise DynamicsError("'t' is reserved for time and cannot be a variable name.")
            if name in SAFE_NAMES:
                raise DynamicsError(
                    f"variable {name!r} shadows the built-in constant/function of the same "
                    "name; rename it (e.g. add a subscript)."
                )
            if name in seen:
                raise DynamicsError(f"variable {name!r} is declared twice.")
            seen.add(name)

        for spec in self.params:
            pname = spec[0]
            if pname in self.variables or pname == _TIME_NAME:
                raise DynamicsError(f"parameter {pname!r} collides with a variable or with 't'.")

        if self.initial and len(self.initial) != len(self.variables):
            raise DynamicsError(
                f"initial state has {len(self.initial)} values but the system has "
                f"{len(self.variables)} variables."
            )

        allowed = self.allowed_names()
        codes: List[object] = []
        for var, eq in zip(self.variables, self.equations):
            try:
                tree = expressions.validate(eq, names=allowed)
                codes.append(compile(tree, "<dynamics>", "eval"))
            except ExpressionError as exc:
                raise ExpressionError(f"d{var}/dt: {exc}") from exc
        return codes


def _rk4_step(codes, variables, namespace, t, y, dt, out, k):
    """Advance the state ``y`` by one RK4 step of size ``dt``. Returns the new state.

    ``out`` and ``k`` are scratch buffers reused across steps to avoid per-step
    allocation. ``namespace`` is the eval globals (``SAFE_NAMES`` + parameters); this
    writes the current variable values and ``t`` into it.
    """
    n = len(variables)

    def deriv(time, state):
        for i in range(n):
            namespace[variables[i]] = state[i]
        namespace[_TIME_NAME] = time
        for i in range(n):
            k[i] = eval(codes[i], _EVAL_GLOBALS, namespace)  # nosec B307
        return k

    k1 = deriv(t, y).copy()
    k2 = deriv(t + 0.5 * dt, y + 0.5 * dt * k1).copy()
    k3 = deriv(t + 0.5 * dt, y + 0.5 * dt * k2).copy()
    k4 = deriv(t + dt, y + dt * k3)
    np.add(y, (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), out=out)
    return out


#: Empty builtins for every equation eval -- the same lockdown ``expressions.evaluate``
#: uses. The equations were already validated against the allowlist; this is defence in
#: depth so a bug in validation still cannot reach ``__import__`` and friends.
_EVAL_GLOBALS: Dict[str, object] = {"__builtins__": {}}


def integrate(
    system: DynamicalSystem,
    *,
    params: Optional[Mapping[str, float]] = None,
    initial: Optional[Sequence[float]] = None,
    dt: Optional[float] = None,
    steps: Optional[int] = None,
    transient: Optional[int] = None,
    name: Optional[str] = None,
) -> DataSet:
    """Integrate ``system`` with RK4 and return the trajectory as a :class:`DataSet`.

    The result has ``steps + 1`` rows and columns ``[t, *system.variables]``, with
    ``source = "dynamics"``. Every argument overrides the system's stored default:

    * ``params`` -- partial; unspecified parameters keep their defaults.
    * ``initial`` -- the starting state, one value per variable.
    * ``dt`` / ``steps`` / ``transient`` -- the integration window. ``transient`` steps
      are integrated first and thrown away so the recorded trajectory starts on the
      attractor.

    Raises :class:`DynamicsError` for an out-of-range ``steps`` or a wrong-length
    ``initial``/``params``, and :class:`~glplot.gui.expressions.ExpressionError` if an
    equation fails to compile (custom systems are validated here, not only at construction).

    Non-finite values are *not* an error: a divergent parameter choice yields inf/nan in
    the tail, which is a faithful readout of "this configuration blows up", and every
    consumer of the columns is nan-safe.
    """
    codes = system.validate()

    values = system.param_defaults()
    if params:
        for pname in params:
            if pname not in values:
                raise DynamicsError(
                    f"unknown parameter {pname!r}; this system has: "
                    f"{', '.join(system.param_names()) or '(none)'}."
                )
        values.update({k: float(v) for k, v in params.items()})

    if initial is None:
        state0 = np.array(system.initial or (0.0,) * len(system.variables), dtype=np.float64)
    else:
        state0 = np.asarray(initial, dtype=np.float64).ravel()
    if state0.shape[0] != len(system.variables):
        raise DynamicsError(
            f"initial state has {state0.shape[0]} values but the system has "
            f"{len(system.variables)} variables."
        )

    dt_val = float(system.dt if dt is None else dt)
    if not np.isfinite(dt_val) or dt_val == 0.0:
        raise DynamicsError("dt must be a finite, non-zero number.")

    n_steps = int(system.steps if steps is None else steps)
    if not MIN_STEPS <= n_steps <= MAX_STEPS:
        raise DynamicsError(f"steps must be between {MIN_STEPS} and {MAX_STEPS:,} (got {n_steps}).")

    n_transient = int(system.transient if transient is None else transient)
    n_transient = max(0, min(n_transient, MAX_STEPS))

    n_vars = len(system.variables)
    variables = system.variables
    namespace: Dict[str, object] = dict(SAFE_NAMES)
    namespace.update(values)

    out = np.empty(n_vars, dtype=np.float64)
    k = np.empty(n_vars, dtype=np.float64)

    y = state0.copy()
    t = 0.0
    with np.errstate(all="ignore"):
        for _ in range(n_transient):
            y = _rk4_step(codes, variables, namespace, t, y, dt_val, out, k).copy()
            t += dt_val

        ts = np.empty(n_steps + 1, dtype=np.float64)
        traj = np.empty((n_steps + 1, n_vars), dtype=np.float64)
        ts[0] = t
        traj[0] = y
        for step in range(1, n_steps + 1):
            y = _rk4_step(codes, variables, namespace, t, y, dt_val, out, k).copy()
            t += dt_val
            ts[step] = t
            traj[step] = y

    columns = [Column(_TIME_NAME, ts)]
    columns.extend(Column(variables[i], traj[:, i]) for i in range(n_vars))
    dataset = DataSet(name or system.label, columns)
    dataset.source = "dynamics"
    return dataset


# ---------------------------------------------------------------------------
# The preloaded library -- ten classic systems.
#
# Each is a plain DynamicalSystem literal, so it is validated at import time (a typo in an
# equation fails the test suite, not a user's first click) and a custom system is the same
# object built at runtime. Parameters and windows are tuned to draw the recognisable
# attractor/limit cycle, not for any particular quantitative accuracy.
# ---------------------------------------------------------------------------

LORENZ = DynamicalSystem(
    key="lorenz",
    label="Lorenz",
    icon="wave",
    description="Lorenz attractor - the canonical butterfly of deterministic chaos.",
    variables=("x", "y", "z"),
    equations=("sigma * (y - x)", "x * (rho - z) - y", "x * y - beta * z"),
    params=(
        ("sigma", 10.0, 0.0, 30.0),
        ("rho", 28.0, 0.0, 100.0),
        ("beta", 8.0 / 3.0, 0.0, 10.0),
    ),
    initial=(1.0, 1.0, 1.0),
    dt=0.01,
    steps=8000,
    transient=500,
    plot=("x", "z"),
)

ROSSLER = DynamicalSystem(
    key="rossler",
    label="Rossler",
    icon="refresh",
    description="Rossler attractor - a single-scroll chaotic spiral from three simple terms.",
    variables=("x", "y", "z"),
    equations=("-y - z", "x + a * y", "b + z * (x - c)"),
    params=(
        ("a", 0.2, 0.0, 1.0),
        ("b", 0.2, 0.0, 2.0),
        ("c", 5.7, 0.0, 20.0),
    ),
    initial=(0.0, 1.0, 0.0),
    dt=0.03,
    steps=12000,
    transient=1000,
    plot=("x", "y"),
)

VAN_DER_POL = DynamicalSystem(
    key="van_der_pol",
    label="Van der Pol",
    icon="sliders",
    description="Van der Pol oscillator - a self-sustaining relaxation limit cycle.",
    variables=("x", "y"),
    equations=("y", "mu * (1 - x**2) * y - x"),
    params=(("mu", 2.0, 0.0, 10.0),),
    initial=(2.0, 0.0),
    dt=0.02,
    steps=3000,
    transient=200,
    plot=("x", "y"),
)

LOTKA_VOLTERRA = DynamicalSystem(
    key="lotka_volterra",
    label="Lotka-Volterra",
    icon="chart_line",
    description="Predator-prey model - populations orbit a shared equilibrium forever.",
    variables=("prey", "pred"),
    equations=(
        "alpha * prey - beta * prey * pred",
        "delta * prey * pred - gamma * pred",
    ),
    params=(
        ("alpha", 1.1, 0.0, 4.0),
        ("beta", 0.4, 0.0, 4.0),
        ("gamma", 0.4, 0.0, 4.0),
        ("delta", 0.1, 0.0, 4.0),
    ),
    initial=(10.0, 5.0),
    dt=0.02,
    steps=4000,
    transient=0,
    plot=("prey", "pred"),
)

GOODWIN = DynamicalSystem(
    key="goodwin",
    label="Goodwin",
    icon="chart_bar",
    description="Goodwin growth cycle - employment and wage share circle like predator-prey.",
    variables=("emp", "wage"),
    # Goodwin's class-struggle model reduces exactly to Lotka-Volterra: employment rate
    # `emp` and wage share `wage` orbit a fixed point set by the parameters.
    equations=(
        "emp * ((1 - wage) / sigma - (alpha + beta))",
        "wage * (rho * emp - (gamma + alpha))",
    ),
    params=(
        ("sigma", 2.0, 0.1, 5.0),
        ("alpha", 0.02, 0.0, 0.5),
        ("beta", 0.02, 0.0, 0.5),
        ("rho", 0.8, 0.0, 3.0),
        ("gamma", 0.5, 0.0, 3.0),
    ),
    initial=(0.9, 0.9),
    dt=0.05,
    steps=6000,
    transient=0,
    plot=("emp", "wage"),
)

DUFFING = DynamicalSystem(
    key="duffing",
    label="Duffing",
    icon="function",
    description="Forced Duffing oscillator - a double-well spring driven into chaos.",
    variables=("x", "y"),
    equations=("y", "-delta * y - alpha * x - beta * x**3 + gamma * cos(omega * t)"),
    params=(
        ("delta", 0.3, 0.0, 2.0),
        ("alpha", -1.0, -5.0, 5.0),
        ("beta", 1.0, -5.0, 5.0),
        ("gamma", 0.37, 0.0, 2.0),
        ("omega", 1.2, 0.0, 5.0),
    ),
    initial=(0.1, 0.0),
    dt=0.02,
    steps=15000,
    transient=2000,
    plot=("x", "y"),
)

FITZHUGH_NAGUMO = DynamicalSystem(
    key="fitzhugh_nagumo",
    label="FitzHugh-Nagumo",
    icon="derivative",
    description="FitzHugh-Nagumo neuron - excitable spiking from a cubic + slow recovery.",
    variables=("v", "w"),
    equations=("v - v**3 / 3 - w + I", "eps * (v + a - b * w)"),
    params=(
        ("I", 0.5, 0.0, 2.0),
        ("a", 0.7, 0.0, 2.0),
        ("b", 0.8, 0.0, 2.0),
        ("eps", 0.08, 0.0, 1.0),
    ),
    initial=(-1.0, 1.0),
    dt=0.05,
    steps=6000,
    transient=0,
    plot=("v", "w"),
)

CHUA = DynamicalSystem(
    key="chua",
    label="Chua",
    icon="target",
    description="Chua's circuit - a double-scroll attractor from a piecewise-linear diode.",
    variables=("x", "y", "z"),
    # The Chua diode's piecewise-linear g(x) is inlined with abs():
    #   g(x) = m1*x + 0.5*(m0 - m1)*(|x+1| - |x-1|)
    equations=(
        "alpha * (y - x - (m1 * x + 0.5 * (m0 - m1) * (abs(x + 1) - abs(x - 1))))",
        "x - y + z",
        "-beta * y",
    ),
    params=(
        ("alpha", 15.6, 0.0, 30.0),
        ("beta", 28.0, 0.0, 60.0),
        ("m0", -1.143, -3.0, 0.0),
        ("m1", -0.714, -3.0, 0.0),
    ),
    initial=(0.7, 0.0, 0.0),
    dt=0.01,
    steps=12000,
    transient=1000,
    plot=("x", "y"),
)

HENON_HEILES = DynamicalSystem(
    key="henon_heiles",
    label="Henon-Heiles",
    icon="layers",
    description="Henon-Heiles - a 2-DOF Hamiltonian galaxy orbit; energy stays bounded.",
    variables=("x", "y", "px", "py"),
    equations=(
        "px",
        "py",
        "-x - 2 * lam * x * y",
        "-y - lam * (x**2 - y**2)",
    ),
    params=(("lam", 1.0, 0.0, 2.0),),
    # Energy at this state is ~0.09, below the 1/6 escape threshold, so the orbit is bounded.
    initial=(0.0, 0.0, 0.35, 0.25),
    dt=0.01,
    steps=20000,
    transient=0,
    plot=("x", "y"),
)

BRUSSELATOR = DynamicalSystem(
    key="brusselator",
    label="Brusselator",
    icon="chart_bar",
    description="Brusselator - an autocatalytic chemical reaction settling to a limit cycle.",
    variables=("x", "y"),
    equations=("a - (b + 1) * x + x**2 * y", "b * x - x**2 * y"),
    params=(
        ("a", 1.0, 0.0, 5.0),
        ("b", 3.0, 0.0, 5.0),
    ),
    initial=(1.0, 1.0),
    dt=0.02,
    steps=6000,
    transient=500,
    plot=("x", "y"),
)

#: The preloaded library, keyed by :attr:`DynamicalSystem.key`, in display order.
SYSTEMS: Dict[str, DynamicalSystem] = {
    system.key: system
    for system in (
        LORENZ,
        ROSSLER,
        VAN_DER_POL,
        LOTKA_VOLTERRA,
        GOODWIN,
        DUFFING,
        FITZHUGH_NAGUMO,
        CHUA,
        HENON_HEILES,
        BRUSSELATOR,
    )
}


def system_labels() -> List[Tuple[str, str]]:
    """``(key, label)`` for every preloaded system, in display order."""
    return [(system.key, system.label) for system in SYSTEMS.values()]


# ---------------------------------------------------------------------------
# Trajectory analysis -- the inverse problem.
#
# Everything above *generates* a trajectory from known equations. These functions go the
# other way: given a single measured/observed signal (one column of a trajectory, or any
# time series), reconstruct and characterise the dynamics behind it. They are what the Math
# Lab "Dynamics" tab runs. Pure numpy; each raises ValueError with an actionable message on
# a signal too short or too flat to analyse, matching the mathops entry-point convention the
# panel already catches.
# ---------------------------------------------------------------------------

#: Above this many samples, the Lyapunov estimator strides the signal down to keep its
#: O(M^2) nearest-neighbour search interactive; the effective time step is scaled to match.
MAX_LYAPUNOV_POINTS = 2500


def _finite(y: np.ndarray) -> np.ndarray:
    """Return ``y`` as a 1-D float64 array with non-finite samples dropped."""
    arr = np.asarray(y, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def delay_embedding(y: np.ndarray, delay: int, dim: int = 2) -> np.ndarray:
    """Takens time-delay embedding of a scalar signal into ``dim`` dimensions.

    Returns an ``(M, dim)`` array whose row ``i`` is ``[y[i], y[i+delay], ...,
    y[i+(dim-1)*delay]]``. This reconstructs the system's phase space from a single
    observable: a 2-D embedding of one Lorenz coordinate already redraws a butterfly-shaped
    attractor, which is the whole point of the phase-portrait analysis.

    Raises ValueError if the signal is too short to form at least two embedded points.
    """
    signal = _finite(y)
    delay = max(1, int(delay))
    dim = max(2, int(dim))
    span = (dim - 1) * delay
    rows = signal.size - span
    if rows < 2:
        raise ValueError(
            "signal is too short for this delay/dimension "
            f"(needs > {span + 1} finite samples, has {signal.size}); "
            "lower the delay or the embedding dimension."
        )
    return np.column_stack([signal[i * delay : i * delay + rows] for i in range(dim)])


def local_maxima(y: np.ndarray) -> np.ndarray:
    """Indices of the strict interior local maxima of ``y`` (a sample above both neighbours)."""
    signal = np.asarray(y, dtype=np.float64).ravel()
    if signal.size < 3:
        return np.empty(0, dtype=np.intp)
    higher = (signal[1:-1] > signal[:-2]) & (signal[1:-1] >= signal[2:])
    return np.flatnonzero(higher) + 1


def return_map(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Successive-maxima return map: ``(m_n, m_{n+1})`` over the signal's local maxima.

    The Lorenz z-coordinate maxima famously fall on a sharp tent-shaped curve -- a
    one-dimensional map hiding inside a three-dimensional flow. Plotting each maximum
    against the next exposes that deterministic structure where the raw time series looks
    random. Returns two equal-length arrays; both are empty when the signal has fewer than
    two maxima.
    """
    signal = _finite(y)
    peaks = signal[local_maxima(signal)]
    if peaks.size < 2:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    return peaks[:-1], peaks[1:]


def lyapunov_rosenstein(
    y: np.ndarray,
    *,
    dt: float = 1.0,
    delay: int = 1,
    dim: int = 3,
    min_separation: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, object]:
    """Estimate the largest Lyapunov exponent by Rosenstein's method.

    The largest Lyapunov exponent measures how fast nearby trajectories diverge: positive
    means chaos (sensitive dependence on initial conditions), zero a limit cycle, negative a
    fixed point. Rosenstein's estimator embeds the signal, tracks the mean log distance
    between each point and its nearest neighbour as both evolve, and reads the exponent off
    the initial slope of that divergence curve.

    Returns a dict with ``times`` and ``divergence`` (the curve to plot), ``fit`` (the fitted
    line over the linear region, same length as ``times``, nan outside it), ``lyapunov`` (the
    slope, in units of 1/time), ``n_points`` (embedded points used) and ``stride`` (signal
    decimation applied for speed; the effective step is ``dt * stride``).

    Raises ValueError for a signal too short to embed and search.
    """
    signal = _finite(y)
    if signal.size < 20:
        raise ValueError(
            f"need at least 20 finite samples to estimate a Lyapunov exponent (has {signal.size})."
        )

    # Stride the signal down so the O(M^2) neighbour search stays interactive; scale dt.
    stride = max(1, int(np.ceil(signal.size / MAX_LYAPUNOV_POINTS)))
    if stride > 1:
        signal = signal[::stride]
    step_dt = float(dt) * stride

    emb = delay_embedding(signal, delay, dim)
    n = emb.shape[0]
    if n < 10:
        raise ValueError("not enough embedded points; lower the delay or embedding dimension.")

    if min_separation is None:
        # A Theiler window: exclude temporally close points so a neighbour is a genuine
        # recurrence, not the same orbit segment one step over.
        min_separation = max(1, (dim - 1) * delay)
    min_separation = int(min_separation)

    if max_steps is None:
        max_steps = max(2, min(n // 2, 200))
    max_steps = int(max_steps)

    # Nearest neighbour of each point, excluding its Theiler window. Looped over points so no
    # (M, M) distance matrix is ever allocated.
    neighbours = np.empty(n, dtype=np.intp)
    for i in range(n):
        d2 = np.sum((emb - emb[i]) ** 2, axis=1)
        lo, hi = max(0, i - min_separation), min(n, i + min_separation + 1)
        d2[lo:hi] = np.inf
        neighbours[i] = int(np.argmin(d2))

    idx = np.arange(n)
    ks: List[int] = []
    log_div: List[float] = []
    with np.errstate(all="ignore"):
        for k in range(max_steps + 1):
            valid = (idx + k < n) & (neighbours + k < n)
            if not np.any(valid):
                break
            a = emb[idx[valid] + k]
            b = emb[neighbours[valid] + k]
            dist = np.sqrt(np.sum((a - b) ** 2, axis=1))
            dist = dist[dist > 0.0]
            ks.append(k)
            log_div.append(float(np.mean(np.log(dist))) if dist.size else float("nan"))

    times = np.asarray(ks, dtype=np.float64) * step_dt
    divergence = np.asarray(log_div, dtype=np.float64)

    # Fit the slope over the initial rise (up to the divergence peak): after neighbours have
    # separated to the attractor's diameter the curve saturates and no longer measures the
    # exponent. Guard against a degenerate one-or-two-point region.
    finite = np.isfinite(divergence)
    fit = np.full_like(times, np.nan)
    lyap = float("nan")
    if int(np.count_nonzero(finite)) >= 2:
        peak = int(np.nanargmax(np.where(finite, divergence, -np.inf)))
        end = max(peak + 1, 2)
        region = finite.copy()
        region[end:] = False
        if int(np.count_nonzero(region)) >= 2:
            slope, intercept = np.polyfit(times[region], divergence[region], 1)
            lyap = float(slope)
            fit[region] = slope * times[region] + intercept

    return {
        "times": times,
        "divergence": divergence,
        "fit": fit,
        "lyapunov": lyap,
        "n_points": n,
        "stride": stride,
    }
