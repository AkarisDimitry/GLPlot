"""``simulate``: ODE solutions checked against closed forms, and every way it refuses.

The harmonic oscillator, exponential decay and a driven stiff problem all have known answers, so
the numbers are compared with those -- not with the solver's own output.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.mcp import simulate
from glplot.mcp.state import AnimationLimits, Limits, McpError
from glplot.mcp.toolbox import Toolbox

OSCILLATOR = dict(equations={"x": "v", "v": "-w**2 * x"}, initial={"x": 1.0, "v": 0.0})


@pytest.fixture
def box():
    toolbox = Toolbox()
    yield toolbox
    toolbox.close_all()


def column(box, dataset, name):
    return np.asarray(box.datasets.column(dataset, name))


@pytest.mark.parametrize("method", ["rk45", "dop853", "lsoda", "rk4", "rk23", "radau", "bdf"])
def test_harmonic_oscillator_matches_cosine(box, method):
    box.simulate(
        "osc", t_end=6.0, steps=300, parameters={"w": 2.0}, method=method, substeps=20, **OSCILLATOR
    )
    t, x = column(box, "osc", "t"), column(box, "osc", "x")
    tolerance = 5e-3 if method in ("rk23", "radau", "bdf") else 1e-5
    assert np.max(np.abs(x - np.cos(2.0 * t))) < tolerance


def test_exponential_decay_and_the_returned_table_layout(box):
    reply = box.simulate(
        "decay", {"y": "-k*y"}, {"y": 3.0}, t_end=2.0, steps=101, parameters={"k": 1.5}
    )
    t, y = column(box, "decay", "t"), column(box, "decay", "y")
    assert t[0] == 0.0 and t[-1] == 2.0 and len(t) == 101
    assert np.max(np.abs(y - 3.0 * np.exp(-1.5 * t))) < 1e-6
    assert reply.data["rows"] == 101 and reply.data["replaced"] is False
    assert reply.data["report"]["final"]["y"] == pytest.approx(3.0 * np.exp(-3.0), rel=1e-5)
    assert "add_plot" in reply.data["plot_hint"]
    assert box.datasets.get("decay").source == "mcp"


def test_a_nonzero_start_time(box):
    box.simulate("d", {"y": "1"}, {"y": 0.0}, t_start=5.0, t_end=7.0, steps=11)
    assert column(box, "d", "t")[0] == 5.0
    assert column(box, "d", "y")[-1] == pytest.approx(2.0)  # dy/dt = 1 over 2 time units


def test_explicit_time_dependence(box):
    box.simulate("d", {"y": "cos(t)"}, {"y": 0.0}, t_end=3.0, steps=50)
    t, y = column(box, "d", "t"), column(box, "d", "y")
    assert np.max(np.abs(y - np.sin(t))) < 1e-6


def test_derived_columns_see_the_whole_trajectory(box):
    box.simulate(
        "osc",
        t_end=10.0,
        steps=400,
        parameters={"w": 1.0},
        derived={"energy": "0.5*v**2 + 0.5*w**2*x**2", "peak": "cumsum(abs(x))"},
        method="dop853",
        **OSCILLATOR,
    )
    energy = column(box, "osc", "energy")
    assert np.ptp(energy) < 1e-6 and energy[0] == pytest.approx(0.5)  # conserved
    assert column(box, "osc", "peak")[-1] > column(box, "osc", "peak")[1]


def test_lorenz_stays_bounded_and_chaotic(box):
    box.simulate(
        "lz",
        {"x": "s*(y - x)", "y": "x*(r - z) - y", "z": "x*y - b*z"},
        {"x": 1, "y": 1, "z": 1},
        t_end=30,
        steps=1500,
        parameters={"s": 10, "r": 28, "b": 8 / 3},
        method="lsoda",
    )
    x, z = column(box, "lz", "x"), column(box, "lz", "z")
    assert np.all(np.abs(x) < 25) and 0 < z.max() < 55
    assert np.sum(np.diff(np.sign(x)) != 0) > 5, "the attractor switches lobes"


def test_a_stiff_problem_needs_and_gets_a_stiff_solver(box):
    box.simulate(
        "stiff", {"y": "-1000*(y - cos(t))"}, {"y": 2.0}, t_end=1.0, steps=200, method="lsoda"
    )
    y = column(box, "stiff", "y")
    assert abs(y[-1] - np.cos(1.0)) < 2e-3  # settles onto the slow solution


def test_simulating_again_replaces_the_dataset(box):
    box.simulate("d", {"y": "-y"}, {"y": 1.0}, t_end=1)
    again = box.simulate("d", {"y": "-y"}, {"y": 1.0}, t_end=2, steps=50)
    assert again.data["replaced"] is True and box.datasets.get("d").n_rows() == 50


def test_the_solution_is_recorded_for_export(box):
    box.simulate("d", {"y": "-y"}, {"y": 1.0}, t_end=1, method="LSODA")
    origin = box.datasets.get("d").mcp_origin
    assert origin["kind"] == "simulate" and origin["spec"]["method"] == "lsoda"


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        (dict(equations={}, initial={}, t_end=1), "maps each state"),
        (
            dict(equations={"x": "y"}, initial={"x": 1}, t_end=1),
            "unknown name 'y'.*Names available: t, x",
        ),
        (dict(equations={"x": "zz"}, initial={"x": 1}, t_end=1), "unknown name 'zz'"),
        (dict(equations={"x": "x"}, initial={}, t_end=1), "starting value.*missing: x"),
        (dict(equations={"x": "x"}, initial={"x": 1, "q": 2}, t_end=1), "unknown: q"),
        (dict(equations={"x": "x"}, initial={"x": "one"}, t_end=1), "finite number"),
        (dict(equations={"t": "1"}, initial={"t": 1}, t_end=1), "not 't'"),
        (dict(equations={"1x": "1"}, initial={"1x": 1}, t_end=1), "identifier"),
        (dict(equations={"x": "a"}, initial={"x": 1}, parameters={"x": 1}, t_end=1), "not a state"),
        (
            dict(equations={"x": "1"}, initial={"x": 1}, parameters={"p": float("nan")}, t_end=1),
            "finite",
        ),
        (dict(equations={"x": "1"}, initial={"x": 1}, t_end=0), "greater than t_start"),
        (dict(equations={"x": "1"}, initial={"x": 1}, t_end=1, steps=1), "between 2 and"),
        (dict(equations={"x": "1"}, initial={"x": 1}, t_end=1, substeps=0), "between 1 and"),
        (
            dict(equations={"x": "1"}, initial={"x": 1}, t_end=1, method="euler"),
            "Unknown method.*rk45",
        ),
        (dict(equations={"x": "x"}, initial={"x": 1}, t_end=1, derived={"x": "1"}), "already used"),
        (dict(equations={"x": "x"}, initial={"x": 1}, t_end=1, derived={"d": "ghost"}), "ghost"),
        (
            dict(
                equations={f"s{i}": "1" for i in range(13)},
                initial={f"s{i}": 0 for i in range(13)},
                t_end=1,
            ),
            "limit is 12",
        ),
    ],
)
def test_refusals_say_what_to_fix(box, kwargs, fragment):
    with pytest.raises(McpError, match=fragment):
        box.simulate("bad", **kwargs)
    assert "bad" not in box.datasets


def test_blow_up_is_reported_not_returned(box):
    with pytest.raises(McpError, match="solver stopped|blew up|non-finite"):
        box.simulate("bad", {"x": "x**2"}, {"x": 1.0}, t_end=100, steps=50)
    with pytest.raises(McpError, match="blew up"):
        box.simulate(
            "bad", {"x": "x**3"}, {"x": 5.0}, t_end=100, steps=50, method="rk4", substeps=1
        )


def test_a_singularity_in_the_equations_is_reported(box):
    with pytest.raises(McpError, match="dx/dt failed at t=|blew up"):
        box.simulate(
            "bad", {"x": "1/(t - 1)"}, {"x": 0.0}, t_end=2, steps=3, method="rk4", substeps=1
        )


def test_the_evaluation_budget_stops_runaway_work():
    small = Toolbox(limits=Limits(animation=AnimationLimits(max_simulation_evals=500)))
    try:
        with pytest.raises(McpError, match="right-hand-side evaluations"):
            small.simulate("d", {"y": "-y"}, {"y": 1.0}, t_end=10, steps=2000, method="rk4")
    finally:
        small.close_all()


def test_dataset_limits_still_apply():
    tiny = Toolbox(limits=Limits(max_datasets=1))
    try:
        tiny.simulate("a", {"y": "-y"}, {"y": 1.0}, t_end=1)
        with pytest.raises(McpError, match="Too many datasets"):
            tiny.simulate("b", {"y": "-y"}, {"y": 1.0}, t_end=1)
    finally:
        tiny.close_all()


def test_methods_table_matches_scipy_names():
    assert set(simulate.METHODS) >= {"rk45", "dop853", "lsoda", "radau", "bdf", "rk4"}
