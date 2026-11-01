"""Worked animations as data: sequences of tool calls a model can copy and adapt.

Every entry is a list of ``(tool, arguments)`` pairs. The guide (``get_guide('mcp-animation')``)
prints them as calls, and ``tests/test_mcp_animation.py`` *runs* them against a real toolbox, so a
recipe that stops working fails a test instead of misleading a model.

The recipes are chosen to cover the ways something can move: a parameter swept in stages, a curve
drawing itself, a simulated trajectory, a camera orbit, data recomputed every frame, colour and
size changing, a 3-D surface, a loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

__all__ = ["COOKBOOK", "Recipe", "format_call", "render_cookbook"]

Call = Tuple[str, Dict[str, Any]]


@dataclass(frozen=True)
class Recipe:
    """One worked animation."""

    key: str
    title: str
    summary: str
    calls: Tuple[Call, ...]


def _animate(**kwargs: Any) -> Call:
    return ("animate", {"figure_id": "fig1", **kwargs})


COOKBOOK: Tuple[Recipe, ...] = (
    Recipe(
        "wave_stages",
        "A wave that travels, grows, then squeezes (three stages)",
        "A parameter swept in stages. `phase` is derived from t so the wave travels the whole "
        "time; `amp` and `k` are parameters that a stage moves and then holds.",
        (
            ("create_figure", {"width": 900, "height": 520, "title": "Traveling wave"}),
            (
                "add_function",
                {
                    "figure_id": "fig1",
                    "expression": "amp*sin(k*x - phase)",
                    "x_min": 0,
                    "x_max": 10,
                    "parameters": {"amp": 1, "k": 2, "phase": 0},
                    "color": "tab:blue",
                    "linewidth": 2,
                },
            ),
            (
                "set_axes",
                {
                    "figure_id": "fig1",
                    "xlim": [0, 10],
                    "ylim": [-3.2, 3.2],
                    "xlabel": "x",
                    "ylabel": "y",
                },
            ),
            _animate(
                output="wave.gif",
                fps=25,
                variables={"phase": "2*pi*time", "amp": 1, "k": 2},
                stages=[
                    {"name": "travel", "duration": 1.5},
                    {"name": "grow", "duration": 1.5, "ease": "in_out", "animate": {"amp": [1, 3]}},
                    {"name": "squeeze", "duration": 1.5, "animate": {"k": [2, 5]}},
                ],
                bindings=[
                    {"target": "a1", "property": "parameters.phase", "value": "phase"},
                    {"target": "a1", "property": "parameters.amp", "value": "amp"},
                    {"target": "a1", "property": "parameters.k", "value": "k"},
                    {
                        "target": "axes",
                        "property": "title",
                        "value": {"format": "amp {amp:.1f}   k {k:.1f}"},
                    },
                ],
            ),
        ),
    ),
    Recipe(
        "curve_draws_itself",
        "A parametric curve that draws itself, with a moving head and a fading tail",
        "`reveal` shows the first fraction of the points, `trail` keeps only the last N, and "
        "`head` puts a dot on the newest point. The stage `frac` goes 0 -> 1.",
        (
            ("create_figure", {"width": 760, "height": 760, "title": "Lissajous 3:2"}),
            (
                "create_dataset",
                {
                    "name": "liss",
                    "columns": {
                        "t": "linspace(0, 2*pi, 600)",
                        "x": "sin(3*t + pi/2)",
                        "y": "sin(2*t)",
                    },
                },
            ),
            (
                "add_plot",
                {
                    "figure_id": "fig1",
                    "kind": "line",
                    "dataset": "liss",
                    "x": "x",
                    "y": "y",
                    "color": "tab:purple",
                    "linewidth": 2,
                    "head": True,
                    "reveal": 0.02,
                },
            ),
            (
                "set_axes",
                {
                    "figure_id": "fig1",
                    "xlim": [-1.2, 1.2],
                    "ylim": [-1.2, 1.2],
                    "equal_aspect": True,
                    "hide_axes": True,
                },
            ),
            _animate(
                output="lissajous.gif",
                fps=30,
                variables={"frac": 0.02},
                stages=[
                    {"name": "draw", "duration": 3, "animate": {"frac": [0.02, 1]}},
                    {"name": "hold", "duration": 0.5},
                ],
                bindings=[
                    {"target": "a1", "property": "reveal", "value": "frac"},
                    {"target": "a1", "property": "trail", "value": "150"},
                ],
            ),
        ),
    ),
    Recipe(
        "pendulum_simulated",
        "A damped pendulum from its equations, drawn as it swings",
        "`simulate` solves the ODE into a dataset; the animation then reveals it. Plot the phase "
        "portrait (angle against angular velocity) for a picture of the whole dynamics.",
        (
            ("create_figure", {"width": 860, "height": 520, "title": "Damped pendulum"}),
            (
                "simulate",
                {
                    "name": "pend",
                    "equations": {"theta": "omega", "omega": "-g/L*sin(theta) - c*omega"},
                    "initial": {"theta": 2.6, "omega": 0},
                    "t_end": 14,
                    "steps": 420,
                    "parameters": {"g": 9.81, "L": 1.0, "c": 0.18},
                    "derived": {"energy": "0.5*omega**2 + g/L*(1 - cos(theta))"},
                },
            ),
            (
                "add_plot",
                {
                    "figure_id": "fig1",
                    "kind": "line",
                    "dataset": "pend",
                    "x": "theta",
                    "y": "omega",
                    "color": "tab:red",
                    "head": True,
                    "reveal": 0.02,
                },
            ),
            (
                "set_axes",
                {
                    "figure_id": "fig1",
                    "xlim": [-3.2, 3.2],
                    "ylim": [-6.5, 6.5],
                    "xlabel": "angle (rad)",
                    "ylabel": "angular velocity (rad/s)",
                },
            ),
            _animate(
                output="pendulum.gif",
                fps=30,
                variables={"frac": 0.02},
                stages=[{"name": "swing", "duration": 4, "animate": {"frac": [0.02, 1]}}],
                bindings=[
                    {"target": "a1", "property": "reveal", "value": "frac"},
                    {
                        "target": "axes",
                        "property": "title",
                        "value": {"format": "damped pendulum   t = {frac:.2f} of the run"},
                    },
                ],
            ),
        ),
    ),
    Recipe(
        "lorenz_3d_orbit",
        "The Lorenz attractor drawn in 3-D, then the camera orbits it (two stages)",
        "A 3-D animation in two stages: the trajectory draws itself, then the camera turns a full "
        "circle (`azim` and `elev` are axes properties). Pin the limits so the box does not move.",
        (
            ("create_figure", {"width": 900, "height": 700, "projection": "3d"}),
            (
                "simulate",
                {
                    "name": "lorenz",
                    "equations": {
                        "x": "sigma*(y - x)",
                        "y": "x*(rho - z) - y",
                        "z": "x*y - beta*z",
                    },
                    "initial": {"x": 1, "y": 1, "z": 1},
                    "t_end": 40,
                    "steps": 1600,
                    "parameters": {"sigma": 10, "rho": 28, "beta": 2.6667},
                    "method": "lsoda",
                },
            ),
            (
                "add_plot",
                {
                    "figure_id": "fig1",
                    "kind": "line",
                    "dataset": "lorenz",
                    "x": "x",
                    "y": "y",
                    "z": "z",
                    "color": "tab:purple",
                    "linewidth": 1.2,
                    "head": True,
                    "reveal": 0.01,
                },
            ),
            (
                "set_axes",
                {
                    "figure_id": "fig1",
                    "xlim": [-22, 22],
                    "ylim": [-28, 28],
                    "zlim": [0, 50],
                    "xlabel": "x",
                    "ylabel": "y",
                    "zlabel": "z",
                },
            ),
            _animate(
                output="lorenz.mp4",
                fps=30,
                variables={"frac": 0.01, "azim": -60},
                stages=[
                    {"name": "draw", "duration": 3, "animate": {"frac": [0.01, 1]}},
                    {
                        "name": "orbit",
                        "duration": 3,
                        "ease": "in_out",
                        "animate": {"azim": [-60, 300]},
                    },
                ],
                bindings=[
                    {"target": "a1", "property": "reveal", "value": "frac"},
                    {"target": "axes", "property": "azim", "value": "azim"},
                    {"target": "axes", "property": "elev", "value": "20 + 10*sin(2*pi*t)"},
                ],
            ),
        ),
    ),
    Recipe(
        "bars_grow_and_recolour",
        "Bars that grow to their values and shift colour",
        "A `data:` binding recomputes a dataset column every frame from the other columns and "
        "the variables; a colormap binding moves the colour along a ramp. Here the bar heights "
        "are `final * grow`, with `grow` eased 0 -> 1.",
        (
            ("create_figure", {"width": 820, "height": 520, "title": "Quarterly sales"}),
            (
                "create_dataset",
                {
                    "name": "sales",
                    "columns": {
                        "q": [1, 2, 3, 4],
                        "final": [3.2, 5.1, 2.4, 6.3],
                        "h": [3.2, 5.1, 2.4, 6.3],
                    },
                },
            ),
            (
                "add_plot",
                {
                    "figure_id": "fig1",
                    "kind": "bar",
                    "dataset": "sales",
                    "x": "q",
                    "y": "h",
                    "tick_labels": ["Q1", "Q2", "Q3", "Q4"],
                    "color": "tab:blue",
                },
            ),
            (
                "set_axes",
                {
                    "figure_id": "fig1",
                    "ylim": [0, 7],
                    "xlabel": "quarter",
                    "ylabel": "sales (M EUR)",
                },
            ),
            _animate(
                output="bars.gif",
                fps=30,
                variables={"grow": 0},
                stages=[
                    {"name": "grow", "duration": 1.8, "ease": "out", "animate": {"grow": [0, 1]}},
                    {"name": "hold", "duration": 0.6},
                ],
                bindings=[
                    {"target": "data:sales", "property": "h", "value": "final*grow"},
                    {
                        "target": "a1",
                        "property": "color",
                        "value": {"colormap": ["#4c78a8", "#f58518"], "at": "grow"},
                    },
                ],
            ),
        ),
    ),
    Recipe(
        "surface_ripple",
        "A rippling 3-D surface that turns slowly, as a seamless loop",
        "A field expression with a parameter `phase`, bound to `4*pi*t` (two full ripples) while "
        "the camera makes one turn: both are whole periods, so `loop=true` closes seamlessly.",
        (
            ("create_figure", {"width": 900, "height": 680, "projection": "3d"}),
            (
                "add_field",
                {
                    "figure_id": "fig1",
                    "kind": "surface",
                    "expression": "sin(sqrt(x**2 + y**2) - phase) * exp(-0.12*(x**2 + y**2))",
                    "x_range": [-6, 6],
                    "y_range": [-6, 6],
                    "resolution": 70,
                    "parameters": {"phase": 0},
                    "cmap": "viridis",
                },
            ),
            ("set_axes", {"figure_id": "fig1", "zlim": [-1, 1], "elev": 35}),
            _animate(
                output="ripple.gif",
                fps=24,
                duration=3,
                loop=True,
                bindings=[
                    {"target": "a1", "property": "parameters.phase", "value": "4*pi*t"},
                    {"target": "axes", "property": "azim", "value": "-60 + 360*t"},
                ],
            ),
        ),
    ),
    Recipe(
        "sweep_a_slider",
        "Sweep a control: damping from underdamped to overdamped",
        "Controls and animation share variables. Define a slider, bind the function to it, and "
        "let a stage move the slider's value; `export_script(kind='interactive')` would give the "
        "same figure as a live slider.",
        (
            ("create_figure", {"width": 860, "height": 500, "title": "Damped oscillator"}),
            (
                "add_function",
                {
                    "figure_id": "fig1",
                    "expression": "exp(-zeta*w*x)*cos(w*sqrt(abs(1 - zeta**2))*x)",
                    "x_min": 0,
                    "x_max": 12,
                    "samples": 600,
                    "parameters": {"zeta": 0.1, "w": 3.0},
                    "color": "tab:green",
                    "linewidth": 2,
                },
            ),
            (
                "set_axes",
                {"figure_id": "fig1", "ylim": [-1.1, 1.1], "xlabel": "t (s)", "ylabel": "x"},
            ),
            (
                "add_control",
                {"figure_id": "fig1", "name": "damping", "min": 0.05, "max": 1.6, "value": 0.1},
            ),
            (
                "bind_property",
                {
                    "figure_id": "fig1",
                    "target": "a1",
                    "property": "parameters.zeta",
                    "value": "damping",
                },
            ),
            (
                "bind_property",
                {
                    "figure_id": "fig1",
                    "target": "axes",
                    "property": "title",
                    "value": {"format": "damping ratio {damping:.2f}"},
                },
            ),
            _animate(
                output="damping.gif",
                fps=25,
                stages=[
                    {
                        "name": "sweep",
                        "duration": 3,
                        "ease": "smooth",
                        "animate": {"damping": [0.05, 1.6]},
                    }
                ],
            ),
        ),
    ),
    Recipe(
        "points_morph",
        "Points that move, grow and change colour from one arrangement to another",
        "A data binding computes each point's position as a blend between two layouts (`lerp`), "
        "the size follows a pulse, and the colour moves along a ramp.",
        (
            ("create_figure", {"width": 760, "height": 760, "title": "Circle to line"}),
            (
                "create_dataset",
                {
                    "name": "pts",
                    "columns": {
                        "i": "arange(0, 60)",
                        "cx": "cos(2*pi*i/60)",
                        "cy": "sin(2*pi*i/60)",
                        "lx": "-1 + 2*i/59",
                        "ly": "0.15*sin(8*pi*i/59)",
                        "x": "cx",
                        "y": "cy",
                    },
                },
            ),
            (
                "add_plot",
                {
                    "figure_id": "fig1",
                    "kind": "scatter",
                    "dataset": "pts",
                    "x": "x",
                    "y": "y",
                    "color": "tab:blue",
                    "size": 8,
                },
            ),
            (
                "set_axes",
                {
                    "figure_id": "fig1",
                    "xlim": [-1.3, 1.3],
                    "ylim": [-1.3, 1.3],
                    "equal_aspect": True,
                    "xlabel": "x",
                    "ylabel": "y",
                },
            ),
            _animate(
                output="morph.gif",
                fps=30,
                variables={"m": 0},
                stages=[
                    {"name": "morph", "duration": 2, "ease": "in_out", "animate": {"m": [0, 1]}},
                    {"name": "back", "duration": 2, "ease": "in_out", "animate": {"m": [1, 0]}},
                ],
                bindings=[
                    {"target": "data:pts", "property": "x", "value": "lerp(cx, lx, m)"},
                    {"target": "data:pts", "property": "y", "value": "lerp(cy, ly, m)"},
                    {"target": "a1", "property": "size", "value": "6 + 8*pingpong(m)"},
                    {
                        "target": "a1",
                        "property": "color",
                        "value": {"colormap": "plasma", "at": "m"},
                    },
                ],
            ),
        ),
    ),
)


def format_call(tool: str, args: Dict[str, Any]) -> str:
    """One call as ``tool(name=value, ...)`` with JSON values: what a model would write."""
    parts = [f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in args.items()]
    return f"{tool}({', '.join(parts)})"


def render_cookbook() -> str:
    """The recipes as Markdown for the animation guide."""
    out: List[str] = []
    for recipe in COOKBOOK:
        out.append(f"### {recipe.title}")
        out.append("")
        out.append(recipe.summary)
        out.append("")
        out.append("```text")
        for tool, args in recipe.calls:
            out.append(format_call(tool, args))
        out.append("```")
        out.append("")
    return "\n".join(out).rstrip() + "\n"
