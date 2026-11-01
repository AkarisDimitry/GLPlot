"""``animate`` and the tools around it: update_artist, controls, bindings, reveal/trail/head.

What an animation draws is checked on the pixels (a dot bound to time moves across the image at a
steady rate), what it writes is checked by reading the file back, and what it refuses is checked
for the message. No MCP SDK is needed.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from glplot.mcp import cookbook, motion
from glplot.mcp.state import AnimationLimits, FileSandbox, Limits, McpError
from glplot.mcp.toolbox import Toolbox

needs_ffmpeg = pytest.mark.skipif(
    motion.available_formats()["mp4"] is not None, reason="needs ffmpeg with an H.264 encoder"
)


@pytest.fixture
def box(tmp_path):
    toolbox = Toolbox(FileSandbox([str(tmp_path)]))
    yield toolbox
    toolbox.close_all()


def image_of(result):
    return Image.open(io.BytesIO(result.image)).convert("RGB")


def red_centroid(image):
    """(x, y) of the red pixels' centre, or None."""
    array = np.asarray(image, dtype=int)
    mask = (array[..., 0] > 180) & (array[..., 1] < 90) & (array[..., 2] < 90)
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def dot_figure(box, **axes):
    """A red dot at x=0 in a pinned frame: the simplest thing whose motion can be measured."""
    box.create_figure(width=600, height=360)
    box.add_plot("fig1", "scatter", x=[0], y=[0], color="red", size=24)
    box.set_axes("fig1", xlim=[0, 10], ylim=[-1, 1], xlabel="x", ylabel="y", **axes)
    return "fig1"


# ======================================================================================
# update_artist
# ======================================================================================


def panel(box, figure="fig1", index=0):
    return box.describe_figure(figure).data["panels"][index]


def test_update_artist_changes_what_is_drawn(box):
    box.create_figure(width=600, height=360)
    box.add_plot("fig1", "line", x=[0, 1, 2], y=[0, 1, 0], color="blue")
    before = box.render_figure("fig1").image
    reply = box.update_artist("fig1", "a1", {"color": "red", "linewidth": 4})
    after = box.render_figure("fig1")
    assert reply.data["recipe"]["color"] == "red" and reply.data["recipe"]["linewidth"] == 4
    assert red_centroid(image_of(after)) is not None
    assert before != after.image


def test_update_artist_reaches_inside_an_object_argument(box):
    box.create_figure(width=600, height=360)
    box.add_function("fig1", "a*sin(x)", 0, 6, parameters={"a": 1})
    assert panel(box)["ylim"][1] < 1.5
    box.update_artist("fig1", "a1", {"parameters.a": 4})
    assert panel(box)["ylim"][1] > 3.5


def test_update_artist_none_restores_the_default(box):
    box.create_figure()
    box.add_plot("fig1", "line", x=[0, 1], y=[0, 1], color="red", label="mine")
    reply = box.update_artist("fig1", "a1", {"color": None, "label": None})
    assert "color" not in reply.data["recipe"] and "label" not in reply.data["recipe"]
    assert "label" not in reply.data["artist"]


def test_update_artist_keeps_ids_and_order(box):
    box.create_figure()
    for k in range(3):
        box.add_plot("fig1", "line", x=[0, 1], y=[k, k + 1])
    box.update_artist("fig1", "a2", {"color": "red"})
    ids = [a["artist_id"] for a in box.describe_figure("fig1").data["artists"]]
    assert ids == ["a1", "a2", "a3"]
    box.add_plot("fig1", "line", x=[0, 1], y=[0, 1])
    assert box.describe_figure("fig1").data["artists"][-1]["artist_id"] == "a4"  # ids keep counting


def test_update_artist_empty_changes_redraws_from_the_data(box):
    box.create_dataset("d", {"x": [0, 1, 2], "y": [0, 1, 2]})
    box.create_figure()
    box.add_plot("fig1", "line", dataset="d", x="x", y="y")
    assert panel(box)["ylim"][1] < 3
    box.create_dataset("d", {"x": [0, 1, 2], "y": [0, 10, 20]})  # replaces the data
    assert panel(box)["ylim"][1] < 3  # the picture keeps what it drew...
    reply = box.update_artist("fig1", "a1", {})
    assert panel(box)["ylim"][1] > 15 and "redrawn" in reply.message  # ...until refreshed


def test_a_failed_update_leaves_the_figure_untouched(box):
    box.create_dataset("d", {"x": [0, 1, 2], "y": [0, 1, 2]})
    box.create_figure(width=500, height=300)
    box.add_plot("fig1", "line", dataset="d", x="x", y="y", color="blue")
    before = box.render_figure("fig1").image
    with pytest.raises(McpError, match="Could not rebuild"):
        box.update_artist("fig1", "a1", {"y": "nope"})
    assert box.render_figure("fig1").image == before
    assert box.describe_figure("fig1").data["artists"][0]["artist_id"] == "a1"
    box.update_artist("fig1", "a1", {"color": "red"})  # and it is still usable
    assert box._figures["fig1"].log[0].args["color"] == "red"
    assert not [k for k in box._figures if "~" in k], "no scratch figure left behind"


def test_update_artist_errors_name_the_choices(box):
    box.create_figure()
    box.add_plot("fig1", "line", x=[0, 1], y=[0, 1])
    with pytest.raises(McpError, match="Did you mean 'color'"):
        box.update_artist("fig1", "a1", {"colour": "red"})
    with pytest.raises(McpError, match="no artist 'a9'.*a1"):
        box.update_artist("fig1", "a9", {})
    with pytest.raises(McpError, match="object like"):
        box.update_artist("fig1", "a1", ["color"])
    with pytest.raises(McpError, match="not an object"):
        box.update_artist("fig1", "a1", {"color.x": 1})


def test_update_artist_can_move_it_to_another_panel(box):
    box.create_figure(rows=1, cols=2)
    box.add_plot("fig1", "line", x=[0, 1], y=[0, 1])
    box.update_artist("fig1", "a1", {"panel": 2})
    assert box.describe_figure("fig1").data["artists"][0]["panel"] == 2


def test_removed_artists_leave_the_record(box):
    box.create_figure()
    box.add_plot("fig1", "line", x=[0, 1], y=[0, 1])
    box.add_plot("fig1", "line", x=[0, 1], y=[1, 0])
    box.remove_artist("fig1", "a1")
    assert [op.artist_id for op in box._figures["fig1"].log] == ["a2"]
    with pytest.raises(McpError, match="no artist 'a1'"):
        box.update_artist("fig1", "a1", {})
    box.update_artist("fig1", "a2", {"color": "red"})  # replay does not resurrect a1
    assert [a["artist_id"] for a in box.describe_figure("fig1").data["artists"]] == ["a2"]


def test_set_axes_calls_merge_in_the_record(box):
    box.create_figure()
    box.add_plot("fig1", "line", x=[0, 1], y=[0, 1])
    box.set_axes("fig1", xlabel="a")
    box.set_axes("fig1", ylabel="b", xlabel="c")
    ops = [op for op in box._figures["fig1"].log if op.tool == "set_axes"]
    assert len(ops) == 1 and ops[0].args == {"xlabel": "c", "ylabel": "b"}


def test_operation_cap_is_enforced(box):
    capped = Toolbox(limits=Limits(animation=AnimationLimits(max_operations=3)))
    try:
        capped.create_figure()
        for _ in range(3):
            capped.add_plot("fig1", "line", x=[0, 1], y=[0, 1])
        with pytest.raises(McpError, match="holds 3 operations"):
            capped.add_plot("fig1", "line", x=[0, 1], y=[0, 1])
    finally:
        capped.close_all()


# ======================================================================================
# controls and bindings
# ======================================================================================


def wave(box):
    box.create_figure(width=600, height=360)
    box.add_function("fig1", "a*sin(x)", 0, 6, parameters={"a": 1}, label="wave")
    box.add_control("fig1", "amp", min=0.5, max=4, value=1)
    box.bind_property("fig1", "a1", "parameters.a", "amp")
    return "fig1"


def test_a_bound_control_redraws_the_figure_when_it_moves(box):
    wave(box)
    assert panel(box)["ylim"][1] < 1.5
    reply = box.set_control("fig1", "amp", 3)
    assert reply.data["changed"] == {"amp": 3.0} and panel(box)["ylim"][1] > 2.5


def test_render_params_try_values_without_keeping_them(box):
    wave(box)
    shown = box.render_figure("fig1", params={"amp": 4})
    assert shown.data["panels"][0]["ylim"][1] > 3.5
    assert panel(box)["ylim"][1] < 1.5, "the figure itself did not change"
    assert not [k for k in box._figures if "~" in k]
    kept = box.save_figure("fig1", "p.png", params={"amp": 2})
    assert kept.data["bytes"] > 0


def test_unknown_param_and_control_are_named(box):
    wave(box)
    with pytest.raises(McpError, match="no control 'amq'.*Did you mean 'amp'"):
        box.render_figure("fig1", params={"amq": 1})
    with pytest.raises(McpError, match="outside slider 'amp'"):
        box.set_control("fig1", "amp", 99)
    with pytest.raises(McpError, match="no control"):
        box.set_control("fig1", "zzz", 1)
    with pytest.raises(McpError, match="Give the new value"):
        box.set_control("fig1", "amp")


def test_toggle_choice_and_button_controls(box):
    box.create_figure(width=600, height=360)
    box.add_function("fig1", "a*sin(k*x + m)", 0, 6, parameters={"a": 1, "k": 1, "m": 0})
    box.add_control("fig1", "amp", min=1, max=5, value=1)
    box.add_control("fig1", "double", kind="toggle")
    box.add_control("fig1", "mode", kind="choice", options=["sin", "cos"])
    box.bind_property("fig1", "a1", "parameters.a", "amp * (1 + double)")
    box.bind_property("fig1", "a1", "parameters.m", "mode * 1.5708")
    box.add_control("fig1", "preset", kind="button", sets={"amp": 3, "double": True})
    box.add_control("fig1", "reset", kind="button")
    assert panel(box)["ylim"][1] < 1.5
    box.set_control("fig1", "preset")
    assert box._figures["fig1"].controls["amp"].value == 3.0
    assert box._figures["fig1"].controls["double"].value is True
    assert panel(box)["ylim"][1] > 5.5  # amp 3 * (1 + 1)
    box.set_control("fig1", "reset")  # no `sets`: everything back to its default
    assert box._figures["fig1"].controls["amp"].value == 1.0 and panel(box)["ylim"][1] < 1.5
    box.set_control("fig1", "mode", "cos")
    assert box._figures["fig1"].controls["mode"].value == "cos"


def test_a_button_must_set_controls_that_exist(box):
    wave(box)
    with pytest.raises(McpError, match="not a control"):
        box.add_control("fig1", "go", kind="button", sets={"ghost": 1})


def test_controls_have_a_cap_and_can_be_replaced(box):
    capped = Toolbox(limits=Limits(animation=AnimationLimits(max_controls=2)))
    try:
        capped.create_figure()
        capped.add_control("fig1", "a", min=0, max=1)
        capped.add_control("fig1", "b", min=0, max=1)
        replaced = capped.add_control("fig1", "a", min=0, max=5, value=2)
        assert "Replaced" in replaced.message
        with pytest.raises(McpError, match="already has 2 controls"):
            capped.add_control("fig1", "c", min=0, max=1)
    finally:
        capped.close_all()


def test_bind_and_unbind(box):
    wave(box)
    reply = box.bind_property("fig1", "a1", "parameters.a", None)
    assert reply.data["bindings"] == [] and "Removed" in reply.message
    box.set_control("fig1", "amp", 3)
    assert panel(box)["ylim"][1] < 1.5, "an unbound control no longer drives the curve"
    with pytest.raises(McpError, match="no binding to remove"):
        box.bind_property("fig1", "a1", "parameters.a", None)


def test_a_binding_replaces_the_one_on_the_same_property(box):
    wave(box)
    box.bind_property("fig1", "a1", "parameters.a", "amp * 2")
    assert len(box._figures["fig1"].bindings) == 1
    assert panel(box)["ylim"][1] > 1.8


def test_a_bad_binding_changes_nothing(box):
    wave(box)
    before = box.render_figure("fig1").image
    for target, prop, value, fragment in [
        ("a9", "linewidth", 2, "No artist 'a9'"),
        ("a1", "colour", "'x'", "colour"),
        ("a1", "linewidth", "ghost + 1", "Could not evaluate"),
        ("axes:4", "title", "'x'", "has 1 panel"),
        ("data:none", "y", "x", "No dataset"),
        ("figure", "x", 1, "Unknown binding target"),
        ("a1", "", 1, "property is the name"),
    ]:
        with pytest.raises(McpError, match=fragment):
            box.bind_property("fig1", target, prop, value)
    assert box.render_figure("fig1").image == before
    assert len(box._figures["fig1"].bindings) == 1


def test_removing_an_artist_removes_its_bindings(box):
    wave(box)
    box.remove_artist("fig1", "a1")
    assert box._figures["fig1"].bindings == []
    box.set_control("fig1", "amp", 2)  # redraws an empty figure without complaint


def test_axes_bindings_and_text_formats(box):
    wave(box)
    box.bind_property("fig1", "axes", "xlabel", {"format": "amp = {amp:.1f}"})
    box.bind_property("fig1", "axes", "xlim", ["0", "2 + amp"])
    box.set_control("fig1", "amp", 2.5)
    shown = panel(box)
    assert shown["xlabel"] == "amp = 2.5" and shown["xlim"] == [0, 4.5]


def test_a_data_binding_recomputes_a_column(box):
    box.create_dataset("d", {"x": [0, 1, 2, 3], "y": [0, 1, 2, 3]})
    box.create_figure()
    box.add_plot("fig1", "line", dataset="d", x="x", y="y")
    box.add_control("fig1", "gain", min=1, max=10, value=1)
    box.bind_property("fig1", "data:d", "y", "x * gain")
    assert panel(box)["ylim"][1] < 4
    box.set_control("fig1", "gain", 10)
    assert panel(box)["ylim"][1] > 25
    assert box.datasets.get("d").get("y").tolist() == [
        0,
        1,
        2,
        3,
    ], "the dataset itself is untouched"


def test_data_bindings_take_expressions_only(box):
    box.create_dataset("d", {"x": [0, 1, 2], "y": [0, 1, 2]})
    box.create_figure()
    box.add_plot("fig1", "line", dataset="d", x="x", y="y")
    with pytest.raises(McpError, match="expression string"):
        box.bind_property("fig1", "data:d", "y", 3)


# ======================================================================================
# second y axis, hatching, reveal / trail / head
# ======================================================================================


def captured_figure(box, figure_id="fig1"):
    """The matplotlib figure render_figure draws (it does not report ticks, patches or twins)."""
    from glplot.mcp import toolbox as toolbox_module

    seen = []
    real = toolbox_module._summarise_axes

    def spy(fig):
        seen.append(fig)
        return real(fig)

    toolbox_module._summarise_axes = spy
    try:
        box.render_figure(figure_id)
        return seen[-1]
    finally:
        toolbox_module._summarise_axes = real


def test_right_hand_axis_is_a_real_second_axis(box):
    box.create_figure(width=700, height=400)
    box.add_plot("fig1", "line", x=[0, 1, 2], y=[1, 2, 3], color="blue")
    box.add_plot("fig1", "line", x=[0, 1, 2], y=[100, 400, 200], axis="right", color="red")
    box.set_axes("fig1", ylabel="left", y2label="right (kPa)", y2lim=[0, 500])
    fig = captured_figure(box)
    labels = {ax.get_ylabel(): ax for ax in fig.axes}
    assert set(labels) == {"left", "right (kPa)"}
    assert labels["right (kPa)"].get_ylim() == (0, 500)
    assert labels["left"].get_ylim()[1] < 4, "the left axis is fitted to its own data only"


def test_right_hand_axis_errors(box):
    box.create_figure()
    box.add_plot("fig1", "line", x=[0, 1], y=[0, 1])
    with pytest.raises(McpError, match="no right-hand y axis"):
        box.set_axes("fig1", y2label="nope")
    with pytest.raises(McpError, match="axis must be"):
        box.add_plot("fig1", "line", x=[0, 1], y=[0, 1], axis="up")
    with pytest.raises(McpError, match="second y axis"):
        box.add_plot("fig1", "hist", y=[1, 2, 3], axis="right")
    box.create_figure(projection="3d")
    with pytest.raises(McpError, match="second y axis"):
        box.add_plot("fig2", "line", x=[0, 1], y=[0, 1], z=[0, 1], axis="right")
    box.add_plot("fig1", "line", x=[0, 1], y=[0, 1], axis="right")
    with pytest.raises(McpError, match="y2scale"):
        box.set_axes("fig1", y2scale="cubic")


def test_hatch_reaches_the_figure_without_a_misleading_warning(box):
    box.create_figure()
    reply = box.add_plot("fig1", "bar", x=[1, 2, 3], y=[3, 1, 2], hatch="//")
    assert "warnings" not in reply.data, "the export draws hatching, so no 'exported only' note"
    fig = captured_figure(box)
    assert {p.get_hatch() for p in fig.axes[0].patches} == {"//"}


@pytest.mark.parametrize("kind", ["bar", "barh", "area", "hist"])
def test_hatch_on_every_kind_that_takes_it(box, kind):
    box.create_figure()
    kwargs = dict(y=[1, 2, 3, 2]) if kind == "hist" else dict(x=[1, 2, 3], y=[3, 1, 2])
    box.add_plot("fig1", kind, hatch="x", **kwargs)
    box.render_figure("fig1")


def count_points(box, artist="a1"):
    return next(
        a["points"] for a in box.describe_figure("fig1").data["artists"] if a["artist_id"] == artist
    )


@pytest.mark.parametrize(
    "reveal, trail, shown",
    [(None, None, 10), (0.5, None, 5), (0.55, None, 6), (1.0, None, 10), (0.0, None, 1),
     (None, 3, 3), (0.5, 3, 3), (0.2, 5, 2), (1.0, 4, 4)],
)  # fmt: skip
def test_reveal_and_trail_show_the_right_points(box, reveal, trail, shown):
    box.create_figure()
    box.add_plot("fig1", "line", x=list(range(10)), y=list(range(10)), reveal=reveal, trail=trail)
    assert count_points(box) == shown


def test_reveal_cuts_every_series_the_same_way(box):
    box.create_dataset("d", {"x": list(range(10)), "y": list(range(10)), "c": list(range(10))})
    box.create_figure()
    box.add_plot("fig1", "scatter", dataset="d", x="x", y="y", color_by="c", size=8, reveal=0.4)
    assert count_points(box) == 4  # colours and sizes are cut with x and y, or this would fail


def test_head_adds_a_dot_on_the_last_shown_point(box):
    box.create_figure(width=600, height=360)
    box.add_plot("fig1", "line", x=[0, 5, 10], y=[0, 0, 0], reveal=0.7, head=True, color="red")
    box.set_axes("fig1", xlim=[0, 10], ylim=[-1, 1])
    layers = box.describe_figure("fig1").data["artists"][0]["layers"]
    plain = Toolbox()
    try:
        plain.create_figure()
        plain.add_plot("fig1", "line", x=[0, 5, 10], y=[0, 0, 0], reveal=0.7)
        assert layers == plain.describe_figure("fig1").data["artists"][0]["layers"] + 1
    finally:
        plain.close_all()


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        (dict(reveal=1.5), "between 0 and 1"),
        (dict(reveal=-0.1), "between 0 and 1"),
        (dict(reveal="half"), "fraction"),
        (dict(trail=0), "at least 1"),
        (dict(trail=2.5), "whole number"),
        (dict(head=True, kind="bar"), "line, scatter or step"),
        (dict(reveal=0.5, kind="hist"), "x-vs-y"),
    ],
)
def test_reveal_trail_head_errors(box, kwargs, fragment):
    box.create_figure()
    kind = kwargs.pop("kind", "line")
    with pytest.raises(McpError, match=fragment):
        box.add_plot("fig1", kind, x=[0, 1, 2], y=[0, 1, 2], **kwargs)


def test_reveal_works_in_3d(box):
    box.create_figure(projection="3d")
    box.add_plot(
        "fig1", "line", x=list(range(8)), y=list(range(8)), z=list(range(8)), reveal=0.5, head=True
    )
    assert count_points(box) == 4
    box.render_figure("fig1")


# ======================================================================================
# animate: what it draws
# ======================================================================================


def test_a_bound_dot_moves_across_the_image_at_a_steady_rate(box):
    dot_figure(box)
    positions = []
    for frame in (0, 5, 10, 15, 20):
        reply = box.animate(
            "fig1",
            frames=21,
            fps=10,
            bindings=[{"target": "a1", "property": "x", "value": ["10*t"]}],
            frame=frame,
        )
        positions.append(red_centroid(image_of(reply)))
    xs = np.array([p[0] for p in positions])
    assert np.all(np.diff(xs) > 20), "it moves to the right every time"
    steps = np.diff(xs)
    assert steps.max() / steps.min() < 1.15, "at a steady rate (t is linear)"
    assert max(abs(p[1] - positions[0][1]) for p in positions) < 1.5, "and stays on one line"


def test_stages_move_the_dot_in_turn_and_hold_it(box):
    dot_figure(box)
    spec = dict(
        fps=10,
        variables={"px": 0, "py": 0},
        stages=[
            {"name": "right", "frames": 11, "animate": {"px": [0, 8]}},
            {"name": "up", "frames": 11, "animate": {"py": [0, 0.8]}},
            {"name": "hold", "frames": 5},
        ],
        bindings=[
            {"target": "a1", "property": "x", "value": ["px"]},
            {"target": "a1", "property": "y", "value": ["py"]},
        ],
    )
    at = {
        frame: red_centroid(image_of(box.animate("fig1", frame=frame, **spec)))
        for frame in (0, 10, 11, 21, 26)
    }
    assert at[10][0] > at[0][0] + 100 and abs(at[10][1] - at[0][1]) < 2  # stage 1: only x moves
    assert abs(at[11][0] - at[10][0]) < 2  # stage 2 starts exactly where stage 1 ended...
    assert at[21][1] < at[10][1] - 40 and abs(at[21][0] - at[10][0]) < 2  # ...then only y moves
    assert at[26] == pytest.approx(at[21], abs=1.0)  # and the last stage holds


def test_the_animate_reply_reports_stages_and_variables(box):
    dot_figure(box)
    reply = box.animate(
        "fig1",
        fps=10,
        variables={"p": 0},
        stages=[
            {"name": "go", "frames": 6, "animate": {"p": [0, 5]}},
            {"name": "stay", "frames": 4},
        ],
        bindings=[{"target": "a1", "property": "x", "value": ["p"]}],
    )
    data = reply.data
    assert data["frames"] == 10 and data["fps"] == 10 and data["duration_s"] == 1.0
    assert [s["name"] for s in data["stages"]] == ["go", "stay"]
    assert data["variables"]["p"] == {"first": 0.0, "last": 5.0, "min": 0.0, "max": 5.0}
    assert data["bindings"][0]["target"] == "a1"
    assert data["sheet"][0] == 0 and data["sheet"][-1] == 9
    json.dumps(data)  # everything in it is JSON
    sheet = image_of(reply)
    assert sheet.width > 400 and "No file written" in " ".join(data["notes"])
    assert not box.sandbox.roots[0].joinpath("x").exists()


def test_animate_does_not_change_the_figure(box):
    dot_figure(box)
    before = box.render_figure("fig1").image
    log = [(op.tool, json.dumps(op.args, sort_keys=True)) for op in box._figures["fig1"].log]
    box.animate("fig1", frames=6, bindings=[{"target": "a1", "property": "x", "value": ["9*t"]}])
    assert box.render_figure("fig1").image == before
    assert [
        (op.tool, json.dumps(op.args, sort_keys=True)) for op in box._figures["fig1"].log
    ] == log
    assert list(box._figures) == ["fig1"], "no scratch figures survive"


def test_animate_uses_the_figures_own_bindings_and_sweeps_its_controls(box):
    box.create_figure(width=600, height=360)
    box.add_function("fig1", "a*sin(x)", 0, 6, parameters={"a": 1})
    box.set_axes("fig1", ylim=[-4, 4], xlabel="x", ylabel="y")
    box.add_control("fig1", "amp", min=0.5, max=3, value=1)
    box.bind_property("fig1", "a1", "parameters.a", "amp")
    reply = box.animate(
        "fig1",
        fps=10,
        stages=[{"name": "swell", "frames": 8, "animate": {"amp": [0.5, 3]}}],
    )
    assert (
        reply.data["variables"]["amp"]["min"] == 0.5
        and reply.data["variables"]["amp"]["max"] == 3.0
    )
    assert not [i for i in reply.data["issues"] if i["severity"] == "warning"]
    assert box._figures["fig1"].controls["amp"].value == 1.0, "the control itself is left alone"


def test_3d_camera_orbit_changes_the_picture(box):
    box.create_figure(projection="3d", width=500, height=400)
    box.add_plot("fig1", "line", x=[0, 1, 2], y=[0, 1, 0], z=[0, 1, 2])
    box.set_axes("fig1", xlim=[0, 2], ylim=[0, 1], zlim=[0, 2], xlabel="x", ylabel="y", zlabel="z")
    reply = box.animate(
        "fig1", frames=6, bindings=[{"target": "axes", "property": "azim", "value": "30 + 300*t"}]
    )
    assert "static" not in [i["code"] for i in reply.data["issues"]]


def test_a_dataset_column_can_be_recomputed_every_frame(box):
    box.create_dataset("d", {"x": "linspace(0, 6, 50)", "y": "sin(x)"})
    box.create_figure()
    box.add_plot("fig1", "line", dataset="d", x="x", y="y")
    box.set_axes("fig1", ylim=[-3, 3], xlim=[0, 6], xlabel="x", ylabel="y")
    reply = box.animate(
        "fig1",
        frames=6,
        variables={"k": "1 + 4*t"},
        bindings=[{"target": "data:d", "property": "y", "value": "sin(k*x)"}],
    )
    assert "static" not in [i["code"] for i in reply.data["issues"]]
    assert box.datasets.get("d").get("y")[1] == pytest.approx(
        np.sin(6 / 49)
    )  # the dataset is intact


def test_a_column_named_like_the_clock_is_called_out(box):
    box.create_dataset("d", {"t": "linspace(0, 6, 30)", "y": "sin(t)"})
    box.create_figure()
    box.add_plot("fig1", "line", dataset="d", x="t", y="y")
    reply = box.animate(
        "fig1", frames=4, bindings=[{"target": "data:d", "property": "y", "value": "sin(t) * time"}]
    )
    assert any("share a name" in note for note in reply.data["notes"])


def test_right_hand_axis_and_hatch_animate(box):
    box.create_figure(width=600, height=360)
    box.add_plot("fig1", "bar", x=[1, 2, 3], y=[3, 1, 2], hatch="//")
    box.add_plot("fig1", "line", x=[1, 2, 3], y=[10, 40, 20], axis="right", color="red")
    box.set_axes("fig1", ylim=[0, 4], xlabel="x", ylabel="l", y2label="r", y2lim=[0, 50])
    reply = box.animate(
        "fig1", frames=4, bindings=[{"target": "a2", "property": "alpha", "value": "0.3 + 0.7*t"}]
    )
    assert reply.data["frames"] == 4


# ======================================================================================
# animate: what it writes
# ======================================================================================


def moving_dot(box, **kwargs):
    dot_figure(box)
    spec = dict(
        frames=9,
        fps=10,
        bindings=[{"target": "a1", "property": "x", "value": ["10*t"]}],
    )
    spec.update(kwargs)
    return box.animate("fig1", **spec)


def test_gif_has_every_frame_and_they_differ(box, tmp_path):
    reply = moving_dot(box, output="dot.gif")
    gif = Image.open(tmp_path / "dot.gif")
    assert gif.n_frames == 9 and list(gif.size) == reply.data["size_px"]
    assert reply.data["file"]["bytes"] == (tmp_path / "dot.gif").stat().st_size
    assert reply.data["file"]["format"] == "gif" and reply.data["file"]["fps_effective"] == 10.0
    centres = []
    for index in range(gif.n_frames):
        gif.seek(index)
        centres.append(red_centroid(gif.convert("RGB"))[0])
    assert np.all(np.diff(centres) > 20)


def test_holds_add_frames_to_the_file_not_to_the_animation(box, tmp_path):
    reply = moving_dot(box, output="hold.gif", hold_start=0.2, hold_end=0.3)
    gif = Image.open(tmp_path / "hold.gif")
    total = 0
    for index in range(gif.n_frames):
        gif.seek(index)
        total += gif.info["duration"]
    # The GIF writer merges identical consecutive frames and adds up their delays, so a hold
    # is kept as time: 9 frames + 2 + 3 held, each 100 ms.
    assert total == (9 + 2 + 3) * 100
    assert reply.data["frames"] == 9 and reply.data["held_frames"] == [2, 3]


def test_apng_and_numbered_png_sequence(box, tmp_path):
    moving_dot(box, output="dot.apng")
    assert Image.open(tmp_path / "dot.apng").n_frames == 9
    reply = moving_dot(box, output="frames/f_%04d.png")
    names = sorted(p.name for p in (tmp_path / "frames").iterdir())
    assert names == [f"f_{i:04d}.png" for i in range(1, 10)]
    assert reply.data["file"]["files"] == 9
    sizes = {Image.open(tmp_path / "frames" / n).size for n in names}
    assert len(sizes) == 1, "every frame has the same size"


@needs_ffmpeg
def test_mp4_is_written_with_the_right_frame_count(box, tmp_path):
    import subprocess

    reply = moving_dot(box, output="dot.mp4", fps=10)
    assert reply.data["file"]["codec"] in motion._H264
    probe = subprocess.run(
        [motion.find_ffmpeg(), "-hide_banner", "-i", str(tmp_path / "dot.mp4")],
        capture_output=True,
        text=True,
    ).stderr
    assert "10 fps" in probe and "Video: h264" in probe or "Video: mpeg4" in probe


def test_mp4_without_ffmpeg_fails_before_rendering(box, monkeypatch):
    dot_figure(box)
    monkeypatch.setattr(motion, "find_ffmpeg", lambda: None)
    with pytest.raises(McpError, match="Cannot write mp4.*ffmpeg"):
        box.animate(
            "fig1",
            output="x.mp4",
            frames=3,
            bindings=[{"target": "a1", "property": "x", "value": ["t"]}],
        )


def test_single_frame_mode_writes_nothing(box, tmp_path):
    dot_figure(box)
    reply = box.animate(
        "fig1",
        output="never.gif",
        frames=5,
        frame=2,
        bindings=[{"target": "a1", "property": "x", "value": ["10*t"]}],
    )
    assert reply.data["frame"] == 2 and reply.data["variables"]["t"] == 0.5
    assert not (tmp_path / "never.gif").exists()
    assert image_of(reply).width > 200


def test_scale_and_quality_choose_the_resolution(box):
    dot_figure(box)
    small = box.animate(
        "fig1",
        frames=3,
        scale=0.5,
        bindings=[{"target": "a1", "property": "x", "value": ["t"]}],
    )
    big = box.animate(
        "fig1",
        frames=3,
        scale=1.0,
        bindings=[{"target": "a1", "property": "x", "value": ["t"]}],
    )
    assert small.data["size_px"][0] * 2 == pytest.approx(big.data["size_px"][0], abs=3)


# ======================================================================================
# animate: the review, and what it refuses
# ======================================================================================


def issue_codes(reply):
    return [item["code"] for item in reply.data["issues"]]


def test_a_static_animation_is_flagged(box):
    dot_figure(box)
    reply = box.animate(
        "fig1", frames=5, bindings=[{"target": "a1", "property": "alpha", "value": "1"}]
    )
    assert "static" in issue_codes(reply)


def test_a_loop_that_does_not_close_is_flagged_and_one_that_does_is_not(box):
    # A wide bar, so each step changes part of the picture (a tiny dot that already jumps by more
    # than its own size at every step cannot show a seam in a pixel comparison).
    box.create_figure(width=600, height=360)
    box.add_plot("fig1", "bar", x=[0], y=[1], width=3, color="red")
    box.set_axes("fig1", xlim=[-3, 13], ylim=[0, 1.2], xlabel="x", ylabel="y")
    open_ = box.animate(
        "fig1", frames=20, loop=True, bindings=[{"target": "a1", "property": "x", "value": ["9*t"]}]
    )
    assert "loop_seam" in issue_codes(open_)
    closed = box.animate(
        "fig1",
        frames=20,
        loop=True,
        bindings=[{"target": "a1", "property": "x", "value": ["5 + 4*sin(2*pi*t)"]}],
    )
    assert "loop_seam" not in issue_codes(closed)


def test_an_unpinned_axis_that_rescales_is_flagged(box):
    box.create_figure(width=600, height=360)
    box.add_plot("fig1", "line", x=list(range(30)), y=list(range(30)), reveal=0.2)
    box.set_axes("fig1", xlabel="x", ylabel="y")  # nothing pinned
    spec = dict(frames=8, bindings=[{"target": "a1", "property": "reveal", "value": "0.2 + 0.8*t"}])
    assert "xlim_moves" in issue_codes(box.animate("fig1", **spec))
    box.set_axes("fig1", xlim=[0, 30], ylim=[0, 30])
    assert "xlim_moves" not in issue_codes(box.animate("fig1", **spec))


def test_figure_warnings_are_included_with_a_scope(box):
    dot_figure(box)
    box.create_figure()
    box.add_plot("fig2", "line", x=[0, 1, 2], y=[0, 1, 2])  # no labels, no title
    reply = box.animate(
        "fig2", frames=4, bindings=[{"target": "a1", "property": "linewidth", "value": "1 + 3*t"}]
    )
    figure_issues = [i for i in reply.data["issues"] if i["scope"] == "figure"]
    assert {i["code"] for i in figure_issues} >= {"no_xlabel", "no_ylabel"}


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        (dict(), "Say how long|how long"),
        (dict(frames=5), "Nothing varies"),
        (dict(frames=5, variables={"v": 1}), "No binding drives anything"),
        (dict(frames=5, bindings=[{"target": "a1"}]), "needs exactly"),
        (
            dict(frames=5, bindings=[{"target": "a1", "property": "x", "value": 1, "extra": 1}]),
            "unknown",
        ),
        (dict(frames=5, bindings="x"), "list of"),
        (dict(frames=5, bindings=["x"]), "must be an object"),
        (
            dict(frames=5, bindings=[{"target": "a7", "property": "x", "value": "t"}]),
            "No artist 'a7'",
        ),
        (
            dict(
                frames=5,
                quality="best",
                bindings=[{"target": "a1", "property": "x", "value": ["t"]}],
            ),
            "quality",
        ),
        (
            dict(frames=5, repeat=-1, bindings=[{"target": "a1", "property": "x", "value": ["t"]}]),
            "repeat",
        ),
        (
            dict(
                frames=5,
                sheet_frames=0,
                bindings=[{"target": "a1", "property": "x", "value": ["t"]}],
            ),
            "sheet_frames",
        ),
        (
            dict(frames=5, scale=9, bindings=[{"target": "a1", "property": "x", "value": ["t"]}]),
            "scale",
        ),
        (
            dict(frames=5, frame=5, bindings=[{"target": "a1", "property": "x", "value": ["t"]}]),
            "from 0 to 4",
        ),
        (
            dict(
                frames=5, hold_end=-1, bindings=[{"target": "a1", "property": "x", "value": ["t"]}]
            ),
            "seconds",
        ),
        (
            dict(
                frames=5,
                hold_start=99,
                bindings=[{"target": "a1", "property": "x", "value": ["t"]}],
            ),
            "30 s",
        ),
    ],
)
def test_animate_refusals_say_what_to_fix(box, kwargs, fragment):
    dot_figure(box)
    with pytest.raises(McpError, match=fragment):
        box.animate("fig1", **kwargs)
    assert list(box._figures) == ["fig1"], "a refused call leaves nothing behind"


@pytest.mark.parametrize(
    "output, fragment",
    [
        ("x.png", "one image"),
        ("x.txt", "Cannot write '.txt'"),
        ("x", "no extension"),
        ("../escape.gif", "outside"),
    ],
)
def test_animate_output_is_validated_before_anything_renders(box, output, fragment):
    dot_figure(box)
    with pytest.raises(McpError, match=fragment):
        box.animate(
            "fig1",
            output=output,
            frames=4,
            bindings=[{"target": "a1", "property": "x", "value": ["t"]}],
        )


def test_animate_will_not_overwrite_unless_asked(box, tmp_path):
    (tmp_path / "dot.gif").write_bytes(b"keep me")
    with pytest.raises(McpError, match="already exists"):
        moving_dot(box, output="dot.gif")
    assert (tmp_path / "dot.gif").read_bytes() == b"keep me"
    moving_dot(box, output="dot.gif", overwrite=True)
    assert Image.open(tmp_path / "dot.gif").n_frames == 9


def test_a_png_sequence_refuses_to_clobber_later_files(box, tmp_path):
    (tmp_path / "f").mkdir()
    (tmp_path / "f" / "f_0003.png").write_bytes(b"keep")
    with pytest.raises(McpError, match="f_0003.png already exists"):
        moving_dot(box, output="f/f_%04d.png")


def test_without_a_root_files_are_refused_but_the_preview_works():
    bare = Toolbox()
    try:
        dot_figure(bare)
        spec = dict(frames=4, bindings=[{"target": "a1", "property": "x", "value": ["t"]}])
        with pytest.raises(McpError, match="disabled"):
            bare.animate("fig1", output="x.gif", **spec)
        reply = bare.animate("fig1", **spec)
        assert "cannot write files" in " ".join(reply.data["notes"])
    finally:
        bare.close_all()


def test_too_many_pixels_and_too_slow_are_refused_up_front(box):
    capped = Toolbox(limits=Limits(animation=AnimationLimits(max_pixels=1000)))
    try:
        dot_figure(capped)
        with pytest.raises(McpError, match="Mpx"):
            capped.animate(
                "fig1", frames=5, bindings=[{"target": "a1", "property": "x", "value": ["t"]}]
            )
    finally:
        capped.close_all()
    slow = Toolbox(limits=Limits(animation=AnimationLimits(max_seconds=0.0001, max_frames=600)))
    try:
        dot_figure(slow)
        with pytest.raises(McpError, match="would take|limit"):
            slow.animate(
                "fig1", frames=500, bindings=[{"target": "a1", "property": "x", "value": ["t"]}]
            )
    finally:
        slow.close_all()


def test_too_many_frames_is_refused(box):
    dot_figure(box)
    with pytest.raises(McpError, match="limit of 600"):
        box.animate(
            "fig1", frames=601, bindings=[{"target": "a1", "property": "x", "value": ["t"]}]
        )


def test_a_binding_that_fails_on_frame_zero_names_the_artist_and_leaves_the_figure(box):
    dot_figure(box)
    with pytest.raises(McpError, match="Could not rebuild.*a1"):
        box.animate(
            "fig1", frames=4, bindings=[{"target": "a1", "property": "colour", "value": "t"}]
        )
    assert list(box._figures) == ["fig1"]


# ======================================================================================
# the cookbook is real
# ======================================================================================


@pytest.mark.parametrize("recipe", cookbook.COOKBOOK, ids=lambda r: r.key)
def test_every_cookbook_recipe_runs(tmp_path, recipe):
    import copy

    toolbox = Toolbox(FileSandbox([str(tmp_path)]))
    try:
        reply = None
        for tool, args in recipe.calls:
            args = copy.deepcopy(args)
            if tool == "animate":
                args["fps"] = 6  # same animation, fewer frames, so the suite stays quick
                extension = str(args.get("output", "")).rsplit(".", 1)[-1]
                if motion.available_formats().get(extension):
                    args.pop("output")  # a video recipe on a machine without ffmpeg: preview only
                reply = getattr(toolbox, tool)(**args)
            else:
                getattr(toolbox, tool)(**args)
        assert reply is not None and reply.image
        warnings = [i for i in reply.data["issues"] if i["severity"] == "warning"]
        assert not warnings, warnings
        if "file" in reply.data:
            assert (tmp_path / reply.data["file"]["path"].rsplit("/", 1)[-1]).stat().st_size > 0
    finally:
        toolbox.close_all()


def test_cookbook_renders_as_markdown_calls():
    text = cookbook.render_cookbook()
    for recipe in cookbook.COOKBOOK:
        assert recipe.title in text
    assert "animate(figure_id=" in text and "```text" in text
    json.loads(cookbook.format_call("x", {"a": [1, 2]}).split("a=")[1].rstrip(")"))


# ======================================================================================
# what the model is told
# ======================================================================================


def test_animation_guide_is_generated_from_the_code(monkeypatch):
    from glplot.mcp import dynamics
    from glplot.mcp.guide import build_instructions, mcp_topics

    topics = {t.id: t.body for t in mcp_topics()}
    text = topics["mcp-animation"]
    for name in list(dynamics.EASINGS) + list(dynamics.HELPERS):
        assert f"`{name}`" in text, name
    for recipe in cookbook.COOKBOOK:
        assert recipe.title in text
    assert "600 frames" in text and "xlim_moves" in text and "loop_seam" in text
    assert "mcp-animation" in build_instructions(None) and "export_script" in build_instructions(
        None
    )
    assert ".gif" in text and "png-sequence" not in text.split("## Output")[1].split("##")[0]
    monkeypatch.setattr(motion, "find_ffmpeg", lambda: None)
    assert "ffmpeg" in {t.id: t.body for t in mcp_topics()}["mcp-animation"].split("## Output")[1]
    interactive = topics["mcp-interactive"]
    for word in ("update_artist", "add_control", "bind_property", "export_script", 'axis="right"'):
        assert word in interactive


def test_the_new_prompts_render():
    from glplot.mcp import catalog

    animate = catalog.find_prompt("animate_this").build(description="a pendulum", output="p.gif")
    assert "a pendulum" in animate and "PINNED" in animate and "'p.gif'" in animate
    assert "Preview first" in catalog.find_prompt("animate_this").build(description="x")
    interactive = catalog.find_prompt("interactive_figure").build(
        figure_id="fig1", parameters="amplitude"
    )
    assert "fig1" in interactive and "amplitude" in interactive and "export_script" in interactive
    with pytest.raises(McpError, match="needs: description"):
        catalog.find_prompt("animate_this").build()


def test_server_info_reports_what_this_machine_can_write(box, monkeypatch):
    info = box.server_info().data["server"]["animation"]
    assert info["formats"]["gif"] == "available" and info["max_frames"] == 600
    assert "in_out" in info["easings"] and "smoothstep" in info["functions"]
    assert {"rk45", "lsoda"} <= set(info["ode_methods"])
    assert info["script_kinds"] == ["figure", "interactive", "animation"]
    monkeypatch.setattr(motion, "find_ffmpeg", lambda: None)
    assert "ffmpeg" in box.server_info().data["server"]["animation"]["formats"]["mp4"]


def test_describe_figure_shows_recipes_controls_and_bindings(box):
    box.create_figure()
    box.add_plot("fig1", "line", x=list(range(50)), y=list(range(50)), color="red")
    box.add_control("fig1", "w", min=1, max=3)
    box.bind_property("fig1", "a1", "linewidth", "w")
    box.animate(
        "fig1", frames=3, bindings=[{"target": "a1", "property": "alpha", "value": "0.5 + 0.5*t"}]
    )
    data = box.describe_figure("fig1").data
    recipe = data["artists"][0]["recipe"]
    assert recipe["tool"] == "add_plot" and recipe["color"] == "red"
    assert recipe["x"] == "[50 values]", "long lists are summarised, not dumped"
    assert data["controls"][0]["name"] == "w" and data["bindings"][0]["property"] == "linewidth"
    assert data["animation"]["frames"] == 3
    json.dumps(data)
