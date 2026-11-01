"""Controls, bindings and timelines (``glplot.mcp.dynamics``): pure logic, no figure, no MCP SDK.

The timeline is checked against values known by construction: where a tween starts and ends, what
holds afterwards, what a derived variable is at a given frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.mcp import dynamics
from glplot.mcp.state import AnimationLimits, Limits, McpError

# -- easing ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(dynamics.EASINGS))
def test_every_easing_runs_from_zero_to_one(name):
    ease = dynamics.EASINGS[name]
    assert float(ease(0.0)) == pytest.approx(0.0)
    assert float(ease(1.0)) == pytest.approx(1.0)
    values = np.asarray(ease(np.linspace(0, 1, 50)))
    assert np.all(np.diff(values) >= -1e-12), "an easing never goes backwards"


def test_easings_clamp_outside_the_unit_interval():
    assert float(dynamics.EASINGS["in_out"](-3.0)) == 0.0
    assert float(dynamics.EASINGS["in_out"](7.0)) == 1.0


def test_helper_functions_have_the_documented_meaning():
    h = dynamics.HELPERS
    assert float(h["lerp"](2.0, 6.0, 0.25)) == pytest.approx(3.0)
    assert float(h["pingpong"](0.5)) == pytest.approx(1.0)
    assert float(h["pingpong"](1.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(h["ramp"](0.5, 0.25, 0.75)) == pytest.approx(0.5)
    assert float(h["ramp"](0.0, 0.25, 0.75)) == 0.0
    assert h["pulse"](np.array([0.1, 0.5, 0.9]), 0.4, 0.6).tolist() == [0.0, 1.0, 0.0]
    assert float(h["ramp"](3.0, 1.0, 1.0)) in (0.0, 1.0)  # a zero-width ramp must not divide by 0


# -- controls ----------------------------------------------------------------------------


def test_slider_defaults_to_the_middle_and_snaps_to_its_step():
    control = dynamics.make_control("amp", "slider", minimum=0, maximum=10, step=2)
    assert control.value in (4.0, 6.0)  # the middle, 5, snapped to a multiple of the step
    control.value = dynamics.set_control_value(control, 7.2)
    assert control.value == 8.0 and dynamics.control_variable(control) == 8.0


def test_log_slider_starts_at_the_geometric_middle():
    control = dynamics.make_control("f", "slider", minimum=1, maximum=100, log=True)
    assert control.value == pytest.approx(10.0)


def test_integer_slider_rounds():
    control = dynamics.make_control("count", "slider", minimum=1, maximum=9, integer=True)
    assert dynamics.set_control_value(control, 3.6) == 4.0


def test_slider_refuses_out_of_range():
    control = dynamics.make_control("a", "slider", minimum=0, maximum=1)
    with pytest.raises(McpError, match="outside slider 'a'"):
        dynamics.set_control_value(control, 2)


def test_toggle_choice_and_button_behave():
    toggle = dynamics.make_control("on", "toggle", value=True)
    assert dynamics.control_variable(toggle) == 1.0
    assert dynamics.set_control_value(toggle, 0) is False
    choice = dynamics.make_control("mode", "choice", options=["sin", "cos"])
    assert dynamics.control_variable(choice) == 0.0
    assert dynamics.set_control_value(choice, "cos") == "cos"
    assert dynamics.set_control_value(choice, 0) == "sin"  # an index works too
    assert dynamics.control_variable(choice) == 0.0
    button = dynamics.make_control("reset", "button", sets={"a": 1})
    assert dynamics.control_variable(button) is None
    with pytest.raises(McpError, match="button"):
        dynamics.set_control_value(button, 1)


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        (dict(name="t", kind="slider", minimum=0, maximum=1), "reserved"),
        (dict(name="not valid", kind="slider", minimum=0, maximum=1), "identifier"),
        (dict(name="a", kind="knob"), "Unknown control kind"),
        (dict(name="a", kind="slider"), "needs min and max"),
        (dict(name="a", kind="slider", minimum=2, maximum=1), "below max"),
        (dict(name="a", kind="slider", minimum=0, maximum=1, log=True), "min > 0"),
        (dict(name="a", kind="slider", minimum=0, maximum=1, step=-1), "positive"),
        (dict(name="a", kind="choice", options=["one"]), "two or more"),
        (dict(name="a", kind="choice", options=["x", "y"], value="z"), "not one of"),
        (dict(name="a", kind="toggle", value="yes"), "true or false"),
    ],
)
def test_control_errors_say_what_to_do(kwargs, fragment):
    name = kwargs.pop("name")
    kind = kwargs.pop("kind")
    with pytest.raises(McpError, match=fragment):
        dynamics.make_control(name, kind, **kwargs)


def test_control_description_is_json_friendly():
    import json

    for control in (
        dynamics.make_control("a", "slider", minimum=0, maximum=1),
        dynamics.make_control("b", "toggle"),
        dynamics.make_control("c", "choice", options=["x", "y"]),
        dynamics.make_control("d", "button"),
    ):
        json.dumps(control.describe())


# -- targets and paths -------------------------------------------------------------------


@pytest.mark.parametrize(
    "target, expected",
    [
        ("a12", ("artist", "a12")),
        ("axes", ("axes", 1)),
        ("axes:3", ("axes", 3)),
        ("data:wave", ("data", "wave")),
    ],
)
def test_parse_target(target, expected):
    assert dynamics.parse_target(target) == expected


@pytest.mark.parametrize("target", ["figure", "a", "axes:x", "data:", "A1", 5])
def test_parse_target_rejects_nonsense(target):
    with pytest.raises(McpError):
        dynamics.parse_target(target)


def test_set_path_nests_and_validates():
    args = {"color": "red"}
    dynamics.set_path(args, "parameters.a", 2)
    dynamics.set_path(args, "color", "blue")
    assert args == {"color": "blue", "parameters": {"a": 2}}
    with pytest.raises(McpError, match="not an object"):
        dynamics.set_path(args, "color.x", 1)
    with pytest.raises(McpError, match="not a property name"):
        dynamics.set_path(args, "a b", 1)


def test_coerce_property_types():
    assert dynamics.coerce_property("trail", 150.0) == 150 and isinstance(
        dynamics.coerce_property("trail", 150.0), int
    )
    assert dynamics.coerce_property("head", 1.0) is True
    assert dynamics.coerce_property("grid", 0.0) is False
    assert dynamics.coerce_property("parameters.bins", 3.7) == 4  # the leaf name decides
    assert dynamics.coerce_property("linewidth", 2.5) == 2.5
    assert dynamics.coerce_property("color", "red") == "red"


# -- value specifications ----------------------------------------------------------------

CTX = {
    "t": np.linspace(0, 1, 5),
    "time": np.linspace(0, 2, 5),
    "frame": np.arange(5.0),
    "amp": np.full(5, 3.0),
}


def values(spec, **kw):
    return dynamics.evaluate_value(spec, CTX, 5, **kw)


def test_number_and_expression_values():
    assert values(2) == [2] * 5
    assert values(True) == [True] * 5
    assert values("1 + 2*t") == pytest.approx([1.0, 1.5, 2.0, 2.5, 3.0])
    assert values("amp * 2") == [6.0] * 5
    assert values("smoothstep(t)")[2] == pytest.approx(0.5)  # helpers are in the vocabulary


def test_list_value_is_per_frame_lists():
    rows = values(["0", "10*t"])
    assert rows[0] == [0.0, 0.0] and rows[4] == [0.0, 10.0]


def test_colormap_value_moves_along_the_ramp():
    colours = values({"colormap": ["#000000", "#ffffff"], "at": "t"})
    assert colours[0] == "#000000" and colours[-1] == "#ffffff"
    assert colours[2] in ("#808080", "#7f7f7f")
    named = values({"colormap": "viridis", "at": "t"})
    assert named[0] != named[-1]


def test_choose_value_picks_by_index():
    picked = values({"choose": ["a", "b", "c"], "index": "floor(t*4)"})
    assert picked == ["a", "b", "c", "c", "c"]  # index 3 and 4 are clamped to the last item


def test_format_value_fills_variables_and_formats_numbers():
    assert values({"format": "t={t:.2f} amp={amp:g}"})[2] == "t=0.50 amp=3"
    assert values({"format": "{{literal}} {time:.0f}s"})[4] == "{literal} 2s"
    assert values({"format": "frame {frame:d}"})[3] == "frame 3"  # the d format takes an integer


@pytest.mark.parametrize(
    "spec, fragment",
    [
        ({"format": "{missing}"}, "not a variable"),
        ({"format": "{t.real}"}, "brace"),
        ({"format": "{t:%s}"}, "number format"),
        ({"format": "{t"}, "brace"),
        ({"colormap": "viridis"}, "needs 'at'"),
        ({"colormap": "no_such_map", "at": "t"}, "colormap"),
        ({"choose": [], "index": "t"}, "non-empty"),
        ({"nonsense": 1}, "needs one of"),
        ("t +", "Could not evaluate"),
        ("1/(t - t)", "not finite"),
        ("linspace(0, 1, 3)", "shape"),
        (None, "cannot use"),
    ],
)
def test_value_errors(spec, fragment):
    with pytest.raises(McpError, match=fragment):
        values(spec)


# -- the timeline ------------------------------------------------------------------------


def test_simple_timeline_from_duration_and_fps():
    tl = dynamics.build_timeline(duration=2, fps=10, variables={"phase": "2*pi*t"})
    assert tl.n == 20 and tl.fps == 10 and tl.duration == pytest.approx(2.0)
    assert tl.t[0] == 0 and tl.t[-1] == pytest.approx(1.0)  # non-loop: both ends included
    assert tl.time[-1] == pytest.approx(1.9)
    assert tl.variables["phase"][-1] == pytest.approx(2 * np.pi)


def test_loop_stops_one_step_short_of_the_start():
    tl = dynamics.build_timeline(frames=8, loop=True, variables={"phase": "2*pi*t"})
    assert tl.t[-1] == pytest.approx(7 / 8)
    # one more step would land exactly on the first frame's value, so the loop closes
    assert tl.variables["phase"][-1] + 2 * np.pi / 8 == pytest.approx(2 * np.pi)


def test_stage_tween_starts_and_ends_exactly_then_holds():
    tl = dynamics.build_timeline(
        fps=10,
        variables={"amp": 1},
        stages=[
            {"name": "a", "frames": 5},
            {"name": "grow", "frames": 6, "animate": {"amp": [1, 3]}},
            {"name": "hold", "frames": 4},
        ],
    )
    amp = tl.variables["amp"]
    assert tl.n == 15
    assert amp[:5].tolist() == [1.0] * 5  # before the stage it holds its starting value
    assert amp[5] == 1.0 and amp[10] == 3.0  # begins and ends exactly
    assert np.all(np.diff(amp[5:11]) > 0)
    assert amp[11:].tolist() == [3.0] * 4  # and holds afterwards
    assert [s["name"] for s in tl.stages] == ["a", "grow", "hold"]
    assert tl.stages[1]["start"] == 5 and tl.stages[1]["end"] == 11


def test_a_stage_starts_where_the_last_ended():
    tl = dynamics.build_timeline(
        fps=10,
        variables={"v": 0},
        stages=[
            {"frames": 5, "animate": {"v": 4}},  # a bare number: from wherever it is, to 4
            {"frames": 5, "animate": {"v": 10}},
        ],
    )
    v = tl.variables["v"]
    assert v[0] == 0 and v[4] == 4 and v[5] == 4 and v[9] == 10


def test_explicit_from_overrides_where_it_was_and_ease_changes_the_shape():
    linear = dynamics.build_timeline(
        fps=10, variables={"v": 0}, stages=[{"frames": 11, "animate": {"v": [2, 4]}}]
    ).variables["v"]
    eased = dynamics.build_timeline(
        fps=10,
        variables={"v": 0},
        stages=[{"frames": 11, "animate": {"v": {"from": 2, "to": 4, "ease": "in"}}}],
    ).variables["v"]
    assert linear[0] == 2 and linear[5] == pytest.approx(3.0) and linear[10] == 4
    assert eased[0] == 2 and eased[10] == 4 and eased[5] < linear[5]  # 'in' starts slowly


def test_stage_expression_uses_stage_progress():
    tl = dynamics.build_timeline(
        fps=10,
        variables={"k": 2},
        stages=[{"frames": 5}, {"frames": 6, "animate": {"k": "2 + 3*s"}}],
    )
    k = tl.variables["k"]
    assert k[4] == 2 and k[5] == 2 and k[10] == pytest.approx(5.0)
    assert k[10] == 5.0 and tl.s[5] == 0 and tl.s[10] == 1


def test_derived_variable_follows_the_parameters_it_uses():
    tl = dynamics.build_timeline(
        fps=10,
        variables={"f": 1, "w": "2*pi*f"},
        stages=[{"frames": 6, "animate": {"f": [1, 3]}}],
    )
    assert tl.variables["w"][0] == pytest.approx(2 * np.pi)
    assert tl.variables["w"][-1] == pytest.approx(6 * np.pi)
    assert "w" not in tl.parameters and "f" in tl.parameters


def test_controls_are_parameters_a_stage_can_sweep():
    tl = dynamics.build_timeline(
        fps=10,
        controls={"damping": 0.5},
        stages=[{"frames": 5, "animate": {"damping": [0.1, 1.5]}}],
    )
    assert tl.variables["damping"][0] == pytest.approx(0.1)
    assert tl.variables["damping"][-1] == pytest.approx(1.5)
    flat = dynamics.build_timeline(frames=4, controls={"damping": 0.5})
    assert flat.variables["damping"].tolist() == [0.5] * 4


def test_a_variable_the_stage_introduces_needs_a_start():
    tl = dynamics.build_timeline(fps=10, stages=[{"frames": 5, "animate": {"x1": [0, 2]}}])
    assert tl.variables["x1"][-1] == 2.0
    with pytest.raises(McpError, match="no starting value"):
        dynamics.build_timeline(fps=10, stages=[{"frames": 5, "animate": {"zz": 2}}])


def test_open_stages_share_the_global_duration():
    tl = dynamics.build_timeline(
        fps=10, duration=3, stages=[{"name": "a"}, {"name": "b", "duration": 1}, {"name": "c"}]
    )
    assert [s["frames"] for s in tl.stages] == [
        10,
        10,
        10,
    ]  # 30 frames: b took 10, a and c split 20


def test_summary_reports_ranges_and_stages():
    tl = dynamics.build_timeline(
        fps=10,
        variables={"amp": 1},
        stages=[{"name": "g", "frames": 5, "animate": {"amp": [1, 2]}}],
    )
    summary = tl.summary()
    assert summary["frames"] == 5 and summary["variables"]["amp"]["max"] == 2.0
    assert summary["stages"][0]["name"] == "g"
    import json

    json.dumps(summary)


def test_at_returns_scalars_for_one_frame():
    tl = dynamics.build_timeline(frames=5, variables={"a": "t*10"})
    at = tl.at(2)
    assert at["frame"] == 2.0 and at["a"] == pytest.approx(5.0) and at["n"] == 5.0


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({}, "how long"),
        (dict(frames=1), "at least 2"),
        (dict(frames=1000), "limit"),
        (dict(frames=10, fps=0), "fps"),
        (dict(frames=10, fps=500), "fps"),
        (dict(frames=10, variables={"t": 1}), "reserved"),
        (dict(frames=10, variables={"a": [1]}), "number .* or an expression"),
        (dict(frames=10, variables={"a": "ghost + 1"}), "unknown name|Could not"),
        (dict(frames=10, stages=[{"frames": 0}]), "at least 1"),
        (dict(frames=10, stages=[{"duration": -1}]), "positive"),
        (dict(frames=10, stages=[{"frames": 4, "bogus": 1}]), "unknown key"),
        (dict(frames=10, stages=[{"frames": 4, "ease": "wobble"}]), "unknown ease"),
        (dict(frames=10, stages=["nope"]), "object"),
        (dict(stages=[{"name": "x"}]), "no length"),
        (
            dict(variables={"a": "2*t"}, stages=[{"frames": 4, "animate": {"a": [0, 1]}}]),
            "expression in variables",
        ),
        (dict(variables={"a": 1}, stages=[{"frames": 4, "animate": {"a": [1, 2, 3]}}]), "from, to"),
        (
            dict(variables={"a": 1}, stages=[{"frames": 4, "animate": {"a": {"nope": 1}}}]),
            "unknown key",
        ),
        (
            dict(variables={"a": 1}, stages=[{"frames": 4, "animate": {"a": {"from": 1}}}]),
            "needs 'to'",
        ),
    ],
)
def test_timeline_errors_name_the_problem(kwargs, fragment):
    with pytest.raises(McpError, match=fragment):
        dynamics.build_timeline(**{"fps": 10, **kwargs})


def test_stage_count_and_frame_limits_come_from_the_limits_object():
    tight = Limits(animation=AnimationLimits(max_frames=10, max_stages=1))
    with pytest.raises(McpError, match="limit"):
        dynamics.build_timeline(frames=11, limits=tight)
    with pytest.raises(McpError, match="stages"):
        dynamics.build_timeline(fps=10, stages=[{"frames": 2}, {"frames": 2}], limits=tight)


def test_timeline_guide_lists_the_vocabulary():
    text = dynamics.timeline_guide()
    for name in dynamics.EASINGS:
        assert f"`{name}`" in text
    assert "`smoothstep`" in text and "`time`" in text
