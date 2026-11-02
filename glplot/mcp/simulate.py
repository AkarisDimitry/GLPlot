"""``simulate``: integrate a system of ordinary differential equations into a dataset.

Animations of real dynamics -- a pendulum, a predator-prey cycle, the Lorenz attractor, a planet's
orbit -- need trajectories, and a trajectory is not a formula of ``t``. This module solves
``d(state)/dt = f(state, t, parameters)`` where each right-hand side is an *expression* run by the
same validated evaluator as everywhere else (no attribute access, no builtins, no imports), and
returns a table of ``t`` and every state (plus derived columns such as energy) that ``add_plot``
can draw -- and that ``reveal`` / ``head`` can draw progressively.

Uses :func:`scipy.integrate.solve_ivp` (a GLPlot dependency) for the adaptive and stiff solvers and
a fixed-step RK4 of its own for predictable cost. Pure: no GL, no MCP SDK.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..gui.datasets import Column, DataSet
from ..gui.expressions import SAFE_NAMES, ExpressionError, validate
from .state import (
    DEFAULT_LIMITS,
    Limits,
    McpError,
    as_float_array,
    evaluate_expression,
    is_identifier,
    suggest,
)

__all__ = ["METHODS", "simulate"]

#: Solver names accepted, by the scipy method they use ("rk4" is the fixed-step one built here).
METHODS: Dict[str, str] = {
    "rk45": "RK45",
    "rk23": "RK23",
    "dop853": "DOP853",
    "radau": "Radau",
    "bdf": "BDF",
    "lsoda": "LSODA",
    "rk4": "rk4",
}

_MAX_STATES = 12


class _Budget:
    """Counts right-hand-side evaluations so a stiff or runaway system cannot spin forever."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def tick(self) -> None:
        self.used += 1
        if self.used > self.limit:
            raise McpError(
                f"The system needed more than {self.limit:,} right-hand-side evaluations. Use a "
                "stiff solver (method='radau' or 'lsoda'), fewer steps or a shorter t_end."
            )


def _compile(expression: str, allowed: Sequence[str], what: str) -> Any:
    if not isinstance(expression, str) or not expression.strip():
        raise McpError(f"{what} must be a non-empty expression string.")
    try:
        tree = validate(expression, names=allowed)
        return compile(tree, "<glplot>", "eval")
    except ExpressionError as exc:
        raise McpError(
            f"{what} {expression!r}: {exc}. Names available: {', '.join(allowed)}."
        ) from exc


def _right_hand_side(
    codes: List[Any],
    states: Sequence[str],
    parameters: Mapping[str, float],
    budget: _Budget,
    names: Sequence[str],
) -> Callable[[float, np.ndarray], List[float]]:
    scope: Dict[str, Any] = dict(SAFE_NAMES)
    scope.update(parameters)

    def rhs(t: float, y: np.ndarray) -> List[float]:
        budget.tick()
        scope["t"] = t
        for name, value in zip(states, y):
            scope[name] = value
        out = []
        for code, label in zip(codes, names):
            try:
                # numpy's overflow warnings stay off: a warning raised inside an eval with no
                # builtins needs ``__import__`` on Python 3.13 and surfaced as a KeyError; a
                # blow-up is caught below from the non-finite values instead
                with np.errstate(all="ignore"):
                    value = eval(code, {"__builtins__": {}}, scope)  # nosec B307 - validated AST
            except (ArithmeticError, ValueError, TypeError) as exc:
                raise McpError(
                    f"d{label}/dt failed at t={t:.6g}: {type(exc).__name__}: {exc}"
                ) from exc
            out.append(float(value))
        return out

    return rhs


def _rk4(
    rhs: Callable[[float, np.ndarray], List[float]],
    y0: np.ndarray,
    grid: np.ndarray,
    substeps: int,
) -> np.ndarray:
    """Classic fixed-step Runge-Kutta, ``substeps`` equal steps between consecutive grid times."""
    out = np.empty((grid.size, y0.size))
    out[0] = y0
    y = y0.copy()
    for k in range(grid.size - 1):
        t0, t1 = grid[k], grid[k + 1]
        h = (t1 - t0) / substeps
        t = t0
        for _ in range(substeps):
            k1 = np.asarray(rhs(t, y))
            k2 = np.asarray(rhs(t + h / 2, y + h / 2 * k1))
            k3 = np.asarray(rhs(t + h / 2, y + h / 2 * k2))
            k4 = np.asarray(rhs(t + h, y + h * k3))
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        if not np.all(np.isfinite(y)):
            raise McpError(
                f"The solution blew up near t={t1:.6g}. Use more steps, substeps, or an adaptive "
                "method (method='rk45' or 'lsoda')."
            )
        out[k + 1] = y
    return out


def simulate(
    name: str,
    equations: Mapping[str, str],
    initial: Mapping[str, float],
    *,
    t_end: float,
    t_start: float = 0.0,
    steps: int = 1000,
    parameters: Optional[Mapping[str, float]] = None,
    derived: Optional[Mapping[str, str]] = None,
    method: str = "rk45",
    substeps: int = 4,
    limits: Limits = DEFAULT_LIMITS,
) -> Tuple[DataSet, Dict[str, Any]]:
    """Solve the system and return ``(table, report)``.

    The table has ``t``, one column per state and one per ``derived`` expression (evaluated over
    the finished trajectory, so ``energy = 0.5*v**2 + ...`` can use arrays and ``cumsum``).
    """
    if not isinstance(equations, Mapping) or not equations:
        raise McpError(
            "equations maps each state to its derivative, e.g. "
            "{'theta': 'omega', 'omega': '-g/L*sin(theta) - c*omega'}."
        )
    states = list(equations)
    if len(states) > _MAX_STATES:
        raise McpError(f"{len(states)} states; the limit is {_MAX_STATES}.")
    for state in states:
        if not is_identifier(state) or state == "t":
            raise McpError(f"State name {state!r} must be an identifier and not 't' (the time).")
    if not isinstance(initial, Mapping) or set(initial) != set(states):
        missing = sorted(set(states) - set(initial or {}))
        extra = sorted(set(initial or {}) - set(states))
        raise McpError(
            "initial must give a starting value for every state"
            + (f" (missing: {', '.join(missing)})" if missing else "")
            + (f" (unknown: {', '.join(extra)})" if extra else "")
            + "."
        )
    start_values = []
    for state in states:
        value = initial[state]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise McpError(f"initial[{state!r}] must be a finite number, got {value!r}.")
        start_values.append(float(value))
    params: Dict[str, float] = {}
    for key, value in (parameters or {}).items():
        if not is_identifier(key) or key == "t" or key in states:
            raise McpError(
                f"Parameter name {key!r} must be an identifier that is not a state or 't'."
            )
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise McpError(f"Parameter {key!r} must be a finite number, got {value!r}.")
        params[key] = float(value)
    for key, value in (("t_start", t_start), ("t_end", t_end)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise McpError(f"{key} must be a finite number.")
    if t_end <= t_start:
        raise McpError(f"t_end ({t_end:g}) must be greater than t_start ({t_start:g}).")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 2 <= steps <= 200_000:
        raise McpError("steps must be a whole number between 2 and 200000 (the rows returned).")
    if isinstance(substeps, bool) or not isinstance(substeps, int) or not 1 <= substeps <= 200:
        raise McpError("substeps must be a whole number between 1 and 200.")
    key = str(method).lower()
    if key not in METHODS:
        raise McpError(
            f"Unknown method {method!r}.{suggest(key, METHODS)} Use one of: {', '.join(METHODS)}."
        )

    allowed = ["t", *states, *params]
    codes = [_compile(equations[s], allowed, f"d{s}/dt") for s in states]
    budget = _Budget(limits.animation.max_simulation_evals)
    rhs = _right_hand_side(codes, states, params, budget, states)
    grid = np.linspace(float(t_start), float(t_end), steps)
    y0 = np.asarray(start_values, dtype=np.float64)
    rhs(grid[0], y0)  # surface a typo at the first call, with a clear name
    budget.used = 0

    if METHODS[key] == "rk4":
        trajectory = _rk4(rhs, y0, grid, substeps)
    else:
        from scipy.integrate import solve_ivp

        solution = solve_ivp(
            rhs,
            (float(t_start), float(t_end)),
            y0,
            t_eval=grid,
            method=METHODS[key],
            rtol=1e-8,
            atol=1e-10,
        )
        if not solution.success:
            raise McpError(
                f"The solver stopped at t={solution.t[-1] if solution.t.size else t_start:.6g}: "
                f"{solution.message} Try method='lsoda' or 'radau' for stiff systems, or check "
                "the equations for a blow-up."
            )
        trajectory = solution.y.T
        if not np.all(np.isfinite(trajectory)):
            raise McpError("The solution contains non-finite values; check the equations.")

    columns = [Column("t", grid)] + [
        Column(state, np.ascontiguousarray(trajectory[:, i])) for i, state in enumerate(states)
    ]
    table = DataSet(name, columns)
    scope: Dict[str, Any] = {"t": grid, **params}
    scope.update({state: columns[i + 1].values for i, state in enumerate(states)})
    for column_name, expression in (derived or {}).items():
        if not is_identifier(column_name) or column_name in scope:
            raise McpError(
                f"Derived column name {column_name!r} must be an identifier not already used."
            )
        result = evaluate_expression(
            expression, scope, limits=limits, what=f"derived column {column_name!r}"
        )
        values = as_float_array(
            result, f"derived column {column_name!r}", length=steps, limits=limits
        )
        table.columns.append(Column(column_name, values))
        scope[column_name] = values
    report: Dict[str, Any] = {
        "method": key,
        "rows": steps,
        "evaluations": budget.used,
        "states": states,
        "final": {state: float(f"{trajectory[-1, i]:.6g}") for i, state in enumerate(states)},
        "range": {
            state: [float(f"{trajectory[:, i].min():.6g}"), float(f"{trajectory[:, i].max():.6g}")]
            for i, state in enumerate(states)
        },
    }
    return table, report
