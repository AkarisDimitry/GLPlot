"""``export_script``: the generated Python is run and compared with the figure it came from.

The strongest check available is to *execute* the script and render its figure and the toolbox's
with the same export settings: the pictures must match pixel for pixel. That is done for a wide
spread of figures (every plot kind, fields, 3-D, panels, twin axes, simulated and computed data),
for the interactive form (moving a control in the script must equal ``set_control``) and for the
animation form (frame ``i`` of the script must equal the toolbox figure at that frame's values;
the timeline arrays must equal :func:`glplot.mcp.dynamics.build_timeline`).
"""

from __future__ import annotations

import io
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from PIL import Image

import glplot.pyplot as gplt
from glplot.mcp import dynamics, script
from glplot.mcp.state import FileSandbox, McpError
from glplot.mcp.toolbox import Toolbox
from glplot.utils.preview import render_preview_bytes


@pytest.fixture
def box(tmp_path):
    toolbox = Toolbox(FileSandbox([str(tmp_path)]))
    yield toolbox
    toolbox.close_all()


def source(reply):
    """The code block of an export_script reply."""
    return reply.message.split("```python\n", 1)[1].rsplit("```", 1)[0]


TIMELINE_KEYS = ("frames", "duration", "fps", "loop", "stages", "variables")


def timeline_of(spec, **extra):
    """The timeline for an ``animate`` specification (its bindings are not part of it)."""
    return dynamics.build_timeline(**{k: v for k, v in spec.items() if k in TIMELINE_KEYS}, **extra)


def run(text, monkeypatch, name="exported"):
    """Execute a script without opening windows or writing files; return its namespace."""
    monkeypatch.setattr(gplt, "savefig", lambda *a, **k: None)
    monkeypatch.setattr(gplt, "show", lambda *a, **k: None)
    namespace = {"__name__": name}
    exec(compile(text, "<exported>", "exec"), namespace)
    return namespace


def activate(namespace):
    """Make the script's figure pyplot's current one. A script runs on its own, so it always is;
    here the toolbox has been used in between and made its own figure current."""
    gplt._CURRENT_PLOT = namespace["fig"]


def move(namespace, name, value):
    """Move a widget of the script (the script's figure must be current: it redraws it)."""
    activate(namespace)
    namespace[f"_{name}"].value = value


def picture(plot):
    data = render_preview_bytes(plot, 1.0, format="png", bbox_inches="tight", pad_inches=0.15)
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=int)


def assert_same_picture(a, b):
    assert a.shape == b.shape, f"sizes differ: {a.shape} vs {b.shape}"
    assert np.abs(a - b).max() == 0, f"{np.count_nonzero(np.abs(a - b).sum(axis=-1))} pixels differ"


def check_figure(box, monkeypatch, kind="figure"):
    reply = box.export_script("fig1", kind)
    namespace = run(source(reply), monkeypatch)
    try:
        assert_same_picture(picture(namespace["fig"]), picture(box._figures["fig1"].plot))
    finally:
        gplt.close(namespace["fig"])
    return reply, namespace


# ======================================================================================
# figures, one per feature
# ======================================================================================


def lines_and_annotations(box):
    box.create_dataset(
        "wave", {"x": "linspace(0, 10, 120)", "y": "sin(x)*exp(-x/6)", "g": "floor(x/2.5)"}
    )
    box.create_figure(width=700, height=420, title="Lines")
    box.add_plot("fig1", "line", dataset="wave", x="x", y="y", label="damped", linewidth=2)
    box.add_plot(
        "fig1", "scatter", dataset="wave", x="x", y="y", color_by="g", cmap="viridis", size=5
    )
    box.add_function("fig1", "0.5*cos(k*x)", 0, 10, parameters={"k": 3}, color="tab:red")
    box.add_annotation("fig1", "hline", y=0, color="gray", linestyle="--")
    box.add_annotation("fig1", "text", x=5, y=0.6, text="peak", fontsize=14)
    box.set_axes("fig1", xlabel="x", ylabel="y", legend=True, grid=True)


def every_xy_kind(box):
    box.create_figure(width=700, height=420, rows=2, cols=3)
    x, y = [1, 2, 3, 4, 5], [2.0, 3.5, 1.5, 4.0, 2.5]
    box.add_plot("fig1", "step", x=x, y=y, panel=1, color="tab:green")
    box.add_plot("fig1", "stem", x=x, y=y, panel=2)
    box.add_plot(
        "fig1", "bar", x=x, y=y, panel=3, error=[0.2] * 5, tick_labels=list("abcde"), hatch="//"
    )
    box.add_plot("fig1", "area", x=x, y=y, y2=[1.0] * 5, panel=4, alpha=0.5)
    box.add_plot("fig1", "errorbar", x=x, y=y, error=[0.3, 0.2, 0.4, 0.1, 0.5], panel=5)
    box.add_plot("fig1", "barh", x=x, y=y, panel=6, tick_labels=list("vwxyz"))


def distributions(box):
    box.create_figure(width=700, height=420, rows=2, cols=3)
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 1, 60).round(3).tolist(), rng.normal(1, 2, 60).round(3).tolist()
    box.create_dataset("g", {"a": a, "b": b})
    box.add_plot("fig1", "hist", dataset="g", y="a", bins=12, hatch="x", panel=1)
    box.add_plot("fig1", "box", dataset="g", columns=["a", "b"], panel=2)
    box.add_plot("fig1", "violin", dataset="g", columns=["a", "b"], panel=3)
    box.add_plot("fig1", "pie", y=[3, 1, 2], tick_labels=["x", "y", "z"], panel=4)
    box.add_plot("fig1", "hist2d", dataset="g", x="a", y="b", bins=8, panel=5, cmap="viridis")
    box.add_plot("fig1", "hexbin", dataset="g", x="a", y="b", panel=6, cmap="plasma")


def twin_and_hatch(box):
    box.create_figure(width=700, height=420, title="Twin")
    box.add_plot("fig1", "bar", x=[1, 2, 3], y=[3, 1, 2], hatch="//", tick_labels=["a", "b", "c"])
    box.add_plot("fig1", "line", x=[1, 2, 3], y=[10, 40, 25], axis="right", color="red")
    box.set_axes("fig1", ylabel="count", y2label="rate", y2lim=[0, 50])


def motion_options(box):
    box.create_figure(width=600, height=360)
    box.add_plot(
        "fig1",
        "line",
        x=list(range(12)),
        y=[i * i % 7 for i in range(12)],
        reveal=0.6,
        trail=5,
        head=True,
    )
    box.add_plot(
        "fig1", "scatter", x=list(range(12)), y=list(range(12)), reveal=0.5, head=True, size=7
    )
    box.set_axes("fig1", xlim=[0, 11], ylim=[-1, 12], xlabel="i", ylabel="v")


def fields_2d(box):
    box.create_figure(width=700, height=420, rows=2, cols=2)
    box.add_field(
        "fig1",
        "image",
        expression="sin(x)*cos(y)",
        x_range=[-3, 3],
        y_range=[-3, 3],
        resolution=40,
        cmap="viridis",
        panel=1,
    )
    box.add_field(
        "fig1",
        "contour",
        expression="x*y + a",
        parameters={"a": 1},
        x_range=[-2, 2],
        y_range=[-2, 2],
        resolution=30,
        levels=6,
        panel=2,
    )
    box.add_field(
        "fig1",
        "pcolormesh",
        matrix=[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
        x_range=[0, 3],
        y_range=[0, 3],
        panel=3,
        vmin=0,
        vmax=4,
    )
    box.add_field(
        "fig1",
        "contourf",
        expression="x**2 + y**2",
        x_range=[-1, 1],
        y_range=[-1, 1],
        resolution=25,
        levels=5,
        panel=4,
    )


def three_d(box):
    box.create_figure(width=700, height=500, projection="3d")
    box.add_field(
        "fig1",
        "surface",
        expression="sin(sqrt(x**2 + y**2))",
        x_range=[-4, 4],
        y_range=[-4, 4],
        resolution=25,
        cmap="viridis",
    )
    box.add_plot("fig1", "line", x=[0, 1, 2], y=[0, 1, 0], z=[1, 2, 1], color="red")
    box.set_axes("fig1", elev=30, azim=45, zlim=[-1, 3], xlabel="x", ylabel="y", zlabel="z")


def simulated(box):
    box.simulate(
        "pend",
        {"th": "om", "om": "-g/L*sin(th) - c*om"},
        {"th": 2.0, "om": 0.0},
        t_end=8,
        steps=120,
        parameters={"g": 9.81, "L": 1.0, "c": 0.2},
        derived={"energy": "0.5*om**2 + g/L*(1 - cos(th))"},
        method="dop853",
    )
    box.create_figure(width=700, height=420)
    box.add_plot("fig1", "line", dataset="pend", x="t", y="th", label="angle")
    box.add_plot("fig1", "line", dataset="pend", x="t", y="energy", axis="right", color="tab:red")
    box.set_axes("fig1", xlabel="t", ylabel="angle", y2label="energy", legend=True)


def computed(box):
    box.create_dataset("raw", {"x": "linspace(0, 6, 80)", "y": "sin(x) + noise(x, 0.1, 4)"})
    box.analyze_dataset("raw", "smooth", y="y", options={"window": 9})
    box.create_dataset("derived", {"z": "2*y"}, base="raw")
    box.create_figure(width=700, height=420)
    box.add_plot("fig1", "line", dataset="derived", x="x", y="z")
    box.add_plot("fig1", "line", dataset="raw", x="x", y="y_smooth", color="red")
    box.set_axes("fig1", xlabel="x", ylabel="y")


def recipe(box):
    box.create_figure(width=700, height=420)
    box.add_recipe("fig1", "signal_trace", sample=True)


def bound_at_current_values(box):
    box.create_dataset("d", {"x": [0, 1, 2, 3], "y": [0, 1, 2, 3]})
    box.create_figure(width=600, height=360)
    box.add_plot("fig1", "line", dataset="d", x="x", y="y")
    box.add_function("fig1", "a*sin(x)", 0, 3, parameters={"a": 1}, color="red")
    box.add_control("fig1", "gain", min=1, max=5, value=3)
    box.add_control("fig1", "amp", min=0.5, max=2, value=2)
    box.bind_property("fig1", "data:d", "y", "x * gain")
    box.bind_property("fig1", "a2", "parameters.a", "amp")
    box.bind_property("fig1", "a2", "linewidth", "amp * 2")
    box.bind_property("fig1", "axes", "xlabel", {"format": "gain {gain:.1f}"})
    box.set_axes("fig1", ylabel="y")


FIGURES = [
    lines_and_annotations,
    every_xy_kind,
    distributions,
    twin_and_hatch,
    motion_options,
    fields_2d,
    three_d,
    simulated,
    computed,
    recipe,
    bound_at_current_values,
]


@pytest.mark.parametrize("build", FIGURES, ids=lambda f: f.__name__)
def test_the_script_draws_the_same_picture(box, monkeypatch, build):
    build(box)
    reply, namespace = check_figure(box, monkeypatch)
    assert reply.data["kind"] == "figure" and reply.data["lines"] > 5
    text = source(reply)
    assert "import glplot.pyplot as plt" in text and "plt.savefig" in text
    compile(text, "<exported>", "exec")


def test_categories_from_a_file_become_tick_labels(box, monkeypatch, tmp_path):
    (tmp_path / "sales.csv").write_text("region,total\nNorth,3\nSouth,5\nEast,2\n")
    box.load_dataset("sales.csv", "sales")
    reply = box.analyze_dataset("sales", "aggregate", y="total", options={"by": "region"})
    name = reply.data["created"]["dataset"]
    box.create_figure(width=600, height=360)
    box.add_plot("fig1", "bar", dataset=name, x="region", y="total_mean")
    reply = box.export_script("fig1")
    assert "'North', 'South', 'East'" in source(reply)  # the labels are written into the call
    run(source(reply), monkeypatch)


def test_datasets_are_rebuilt_from_how_they_were_made(box):
    simulated(box)
    text = source(box.export_script("fig1"))
    assert "solve_ivp" in text and "np.sin(th)" in text and "_rhs_" in text
    assert "dop853".upper() in text.upper()
    box.create_dataset("a", {"x": "linspace(0, 1, 5)"})
    box.create_dataset("b", {"y": "x * 2"}, base="a")
    box.create_figure()
    box.add_plot("fig2", "line", dataset="b", x="x", y="y")
    text = source(box.export_script("fig2"))
    assert text.index("_make_a") < text.index("_make_b"), "a base dataset is built first"
    assert "np.linspace(0, 1, 5)" in text


def test_a_dataset_too_big_to_embed_is_called_out(box):
    box.create_dataset("big", {"x": "linspace(0, 1, 6000)", "y": "x"})
    box.analyze_dataset("big", "smooth", y="y", options={"window": 5})
    box.create_figure()
    box.add_plot("fig1", "line", dataset="big", x="x", y="y_smooth")
    reply = box.export_script("fig1")
    assert any("too big to embed" in note for note in reply.data["notes"])
    assert "SystemExit" in source(reply)


def test_a_loaded_file_is_loaded_again(box, tmp_path):
    (tmp_path / "t.csv").write_text("x,y\n0,0\n1,2\n2,4\n")
    box.load_dataset("t.csv", "t")
    box.create_figure()
    box.add_plot("fig1", "line", dataset="t", x="x", y="y")
    text = source(box.export_script("fig1"))
    assert "genfromtxt" in text and "t.csv" in text


def test_the_script_is_plain_numpy_and_glplot(box, monkeypatch):
    lines_and_annotations(box)
    text = source(box.export_script("fig1"))
    forbidden = ["glplot.mcp", "from glplot.gui.expressions", "Toolbox", "evaluate("]
    assert not [word for word in forbidden if word in text], "no dependence on the MCP machinery"


def test_scripts_are_short_enough_to_read(box):
    lines_and_annotations(box)
    reply = box.export_script("fig1")
    assert reply.data["lines"] < 80


# ======================================================================================
# interactive scripts
# ======================================================================================


def controlled(box):
    box.create_figure(width=600, height=360, title="Wave")
    box.add_function(
        "fig1", "a*sin(k*x - m)", 0, 8, parameters={"a": 1, "k": 2, "m": 0}, color="tab:blue"
    )
    box.set_axes("fig1", ylim=[-4, 4], xlabel="x", ylabel="y")
    box.add_control("fig1", "amp", min=0.5, max=3, value=1)
    box.add_control("fig1", "k", min=1, max=6, step=1, value=2)
    box.add_control("fig1", "fast", kind="toggle")
    box.add_control("fig1", "mode", kind="choice", options=["sin", "cos"])
    box.add_control("fig1", "reset", kind="button", sets={"amp": 1, "k": 2})
    box.bind_property("fig1", "a1", "parameters.a", "amp")
    box.bind_property("fig1", "a1", "parameters.k", "k + 2*fast")
    box.bind_property("fig1", "a1", "parameters.m", "mode * 1.5708")
    box.bind_property("fig1", "axes", "title", {"format": "amp {amp:.1f}, k {k:.0f}"})


def test_interactive_script_matches_the_figure_at_every_control_value(box, monkeypatch):
    controlled(box)
    namespace = run(source(box.export_script("fig1", "interactive")), monkeypatch)
    try:
        assert_same_picture(picture(namespace["fig"]), picture(box._figures["fig1"].plot))
        for name, value in [("amp", 2.5), ("k", 4), ("fast", True), ("mode", "cos")]:
            move(namespace, name, value)  # moving the widget redraws the figure
            box.set_control("fig1", name, value)
            assert_same_picture(picture(namespace["fig"]), picture(box._figures["fig1"].plot))
    finally:
        gplt.close(namespace["fig"])


def test_a_button_in_the_script_does_what_set_control_does(box, monkeypatch):
    controlled(box)
    namespace = run(source(box.export_script("fig1", "interactive")), monkeypatch)
    try:
        for name, value in [("amp", 3.0), ("k", 5)]:
            move(namespace, name, value)
            box.set_control("fig1", name, value)
        activate(namespace)
        namespace["_press_reset"]()
        box.set_control("fig1", "reset")
        assert namespace["_amp"].value == 1.0 and namespace["_k"].value == 2
        assert_same_picture(picture(namespace["fig"]), picture(box._figures["fig1"].plot))
    finally:
        gplt.close(namespace["fig"])


def test_interactive_script_declares_real_widgets(box):
    controlled(box)
    text = source(box.export_script("fig1", "interactive"))
    for fragment in (
        "plt.slider(",
        "plt.checkbox(",
        "plt.dropdown(",
        "plt.button(",
        "plt.show()",
        "def build():",
    ):
        assert fragment in text
    assert "step=1.0" in text  # the k slider's step is kept
    assert (
        "p_k = k + 2 * fast" in text
    ), "a function's own parameter does not clobber the control's name"


def test_interactive_with_data_bindings(box, monkeypatch):
    box.create_dataset("d", {"x": [0, 1, 2, 3], "y": [0, 1, 2, 3]})
    box.create_figure(width=600, height=360)
    box.add_plot("fig1", "line", dataset="d", x="x", y="y")
    box.add_control("fig1", "gain", min=1, max=5, value=1)
    box.bind_property("fig1", "data:d", "y", "x * gain")
    namespace = run(source(box.export_script("fig1", "interactive")), monkeypatch)
    try:
        move(namespace, "gain", 4.0)
        box.set_control("fig1", "gain", 4.0)
        assert_same_picture(picture(namespace["fig"]), picture(box._figures["fig1"].plot))
    finally:
        gplt.close(namespace["fig"])


# ======================================================================================
# animation scripts
# ======================================================================================

SPEC = dict(
    fps=10,
    loop=True,
    variables={"ph": "2*pi*t", "amp": 1, "k": 2, "w": "2*pi*k"},
    stages=[
        {"name": "travel", "frames": 6},
        {"name": "grow", "frames": 8, "ease": "in_out", "animate": {"amp": [1, 2.5]}},
        {
            "name": "squeeze",
            "frames": 7,
            "ease": "smooth",
            "animate": {"k": {"from": 2, "to": 5, "ease": "in"}},
        },
        {"name": "wobble", "frames": 5, "animate": {"amp": "2.5 - s"}},
        {"name": "hold", "frames": 3},
    ],
    bindings=[
        {"target": "a1", "property": "parameters.phase", "value": "ph"},
        {"target": "a1", "property": "parameters.amp", "value": "amp"},
        {"target": "a1", "property": "parameters.k", "value": "k"},
        {
            "target": "axes",
            "property": "title",
            "value": {"format": "stage {stage:.0f}  t {t:.2f}"},
        },
    ],
)


def animated(box):
    box.create_figure(width=600, height=360, title="Wave")
    box.add_function(
        "fig1", "amp*sin(k*x - phase)", 0, 8, parameters={"amp": 1, "k": 2, "phase": 0}
    )
    box.set_axes("fig1", ylim=[-4, 4], xlabel="x", ylabel="y")
    box.animate("fig1", **SPEC)


def test_the_timeline_in_the_script_equals_the_one_the_server_built(box, monkeypatch):
    animated(box)
    namespace = run(source(box.export_script("fig1", "animation")), monkeypatch)
    try:
        expected = timeline_of(SPEC)
        assert namespace["N"] == expected.n and namespace["FPS"] == expected.fps
        for name, values in expected.variables.items():
            assert np.allclose(namespace[f"_tl_{name}"], values, atol=1e-9), name
        for key in ("t", "time", "s", "stage", "frame"):
            assert np.allclose(namespace[f"_{key}"], getattr(expected, key)), key
    finally:
        gplt.close(namespace["fig"])


def test_a_frame_of_the_script_equals_the_same_frame_of_the_server(box, monkeypatch):
    # Sweep a control, so the server's figure at a frame is the figure at that control value.
    box.create_figure(width=600, height=360)
    box.add_function("fig1", "a*sin(x)", 0, 6, parameters={"a": 1}, color="tab:blue")
    box.set_axes("fig1", ylim=[-4, 4], xlabel="x", ylabel="y")
    box.add_control("fig1", "amp", min=0.5, max=3, value=1)
    box.bind_property("fig1", "a1", "parameters.a", "amp")
    box.animate(
        "fig1", fps=10, stages=[{"name": "swell", "frames": 5, "animate": {"amp": [0.5, 3.0]}}]
    )
    namespace = run(source(box.export_script("fig1", "animation")), monkeypatch)
    try:
        timeline = dynamics.build_timeline(
            fps=10, controls={"amp": 1.0}, stages=[{"frames": 5, "animate": {"amp": [0.5, 3.0]}}]
        )
        for index in (0, 2, 4):
            activate(namespace)
            namespace["update"](index)
            shown = box._build_for(
                box._figures["fig1"], params={"amp": float(timeline.variables["amp"][index])}
            )
            try:
                assert_same_picture(picture(namespace["fig"]), picture(shown.plot))
            finally:
                box._release(shown)
    finally:
        gplt.close(namespace["fig"])


def test_every_frame_of_the_script_is_different_where_it_should_be(box, monkeypatch):
    animated(box)
    namespace = run(source(box.export_script("fig1", "animation")), monkeypatch)
    try:
        frames = []
        for index in range(namespace["N"]):
            activate(namespace)
            namespace["update"](index)
            frames.append(picture(namespace["fig"]))
        moved = [not np.array_equal(a, b) for a, b in zip(frames, frames[1:])]
        assert sum(moved) >= namespace["N"] - 6, "all but the held frames change"
    finally:
        gplt.close(namespace["fig"])


def test_the_animation_script_runs_end_to_end_and_writes_a_gif(box, tmp_path):
    animated(box)
    program = tmp_path / "make.py"
    program.write_text(source(box.export_script("fig1", "animation")))
    done = subprocess.run(
        [sys.executable, "-W", "ignore", str(program)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr[-800:]
    gif = Image.open(tmp_path / "fig1.gif")
    assert gif.n_frames == timeline_of(SPEC).n


def test_animation_script_with_a_data_binding_and_helpers(box, monkeypatch):
    box.create_dataset(
        "pts",
        {"i": "arange(0, 20)", "cx": "cos(2*pi*i/20)", "lx": "-1 + i/9.5", "x": "cx", "y": "0*i"},
    )
    box.create_figure(width=500, height=400)
    box.add_plot("fig1", "scatter", dataset="pts", x="x", y="y", color="tab:blue", size=6)
    box.set_axes("fig1", xlim=[-1.3, 1.3], ylim=[-1, 1], xlabel="x", ylabel="y")
    box.animate(
        "fig1",
        fps=8,
        variables={"m": 0},
        stages=[{"frames": 4, "ease": "in_out", "animate": {"m": [0, 1]}}],
        bindings=[
            {"target": "data:pts", "property": "x", "value": "lerp(cx, lx, m)"},
            {"target": "a1", "property": "size", "value": "6 + 8*pingpong(m)"},
            {"target": "a1", "property": "color", "value": {"colormap": "plasma", "at": "m"}},
        ],
    )
    text = source(box.export_script("fig1", "animation"))
    assert (
        "def lerp(" in text
        and "def pingpong(" in text
        and "def _unit(" in text
        and "def _ramp(" in text
    )
    namespace = run(text, monkeypatch)
    try:
        namespace["update"](0)
        first = picture(namespace["fig"])
        namespace["update"](3)
        assert not np.array_equal(first, picture(namespace["fig"]))
    finally:
        gplt.close(namespace["fig"])


def test_a_script_agrees_with_the_server_on_integers_and_plain_text(box, monkeypatch):
    # bins is a whole number and color a plain string: the server turns the float an expression
    # gives into an int, and takes "red" as the colour. The script has to do both, or numpy
    # refuses bins=17.3 and Python looks for a variable called red.
    rng = np.random.default_rng(3)
    box.create_dataset("g", {"v": [float(v) for v in rng.normal(size=300)]})
    box.create_figure(width=600, height=360, title="Bins")
    box.add_plot("fig1", "hist", dataset="g", y="v", bins=5)
    box.add_annotation("fig1", "text", x=1, y=10, text="here", fontsize=14)
    box.set_axes("fig1", xlim=[-4, 4], ylim=[0, 120], xlabel="v", ylabel="n")
    box.add_control("fig1", "b", min=5, max=40, value=5)
    bindings = [
        ("a1", "bins", "b"),
        ("a1", "color", "red"),
        ("a2", "x", "-4 + b/5"),
        ("axes", "title", "Fixed"),
    ]
    for target, prop, value in bindings:
        box.bind_property("fig1", target, prop, value)
    box.animate("fig1", fps=10, stages=[{"name": "fine", "frames": 5, "animate": {"b": [5, 40]}}])
    text = source(box.export_script("fig1", "animation"))
    assert "int(round(" in text and "color='red'" in text and "color=red" not in text
    namespace = run(text, monkeypatch)
    try:
        timeline = dynamics.build_timeline(
            fps=10, controls={"b": 5.0}, stages=[{"frames": 5, "animate": {"b": [5, 40]}}]
        )
        for index in (0, 1, 3, 4):
            activate(namespace)
            namespace["update"](index)
            # the server draws the same frame by holding the control at that frame's value
            shown = box._build_for(
                box._figures["fig1"], params={"b": float(timeline.variables["b"][index])}
            )
            try:
                assert_same_picture(picture(namespace["fig"]), picture(shown.plot))
            finally:
                box._release(shown)
    finally:
        gplt.close(namespace["fig"])


def frames_match(box, monkeypatch, frames):
    """Run the animation script; check frame ``i`` against the server holding ``params``."""
    namespace = run(source(box.export_script("fig1", "animation")), monkeypatch)
    try:
        for index, params in frames:
            activate(namespace)
            namespace["update"](index)
            shown = box._build_for(box._figures["fig1"], params=params)
            try:
                assert_same_picture(picture(namespace["fig"]), picture(shown.plot))
            finally:
                box._release(shown)
    finally:
        gplt.close(namespace["fig"])


def test_a_3d_curve_that_draws_itself_from_nothing_matches_the_server(box, monkeypatch):
    # reveal=0 shows one point, and a 3-D line needs two: both sides show the first segment.
    th = np.linspace(0, 6 * np.pi, 120)
    box.create_dataset("h", {"x": list(np.cos(th)), "y": list(np.sin(th)), "z": list(th / 10)})
    box.create_figure(width=500, height=400, projection="3d")
    box.add_plot("fig1", "line", dataset="h", x="x", y="y", z="z", reveal=0.5, trail=30)
    box.set_axes("fig1", xlim=[-1.2, 1.2], ylim=[-1.2, 1.2], zlim=[0, 2], xlabel="x", ylabel="y")
    box.add_control("fig1", "r", min=0, max=1, value=0.5)
    box.add_control("fig1", "tail", min=1, max=60, value=30, integer=True)
    box.bind_property("fig1", "a1", "reveal", "r")
    box.bind_property("fig1", "a1", "trail", "tail")
    box.animate("fig1", frames=4, stages=[{"frames": 4, "animate": {"r": [0, 1], "tail": [1, 50]}}])
    timeline = dynamics.build_timeline(
        frames=4,
        controls={"r": 0.5, "tail": 30.0},
        stages=[{"frames": 4, "animate": {"r": [0, 1], "tail": [1, 50]}}],
    )
    frames_match(
        box,
        monkeypatch,
        [
            (
                i,
                {
                    "r": float(timeline.variables["r"][i]),
                    "tail": float(timeline.variables["tail"][i]),
                },
            )
            for i in range(4)
        ],
    )


def test_a_contour_with_a_bound_number_of_levels_matches_the_server(box, monkeypatch):
    box.create_figure(width=500, height=400)
    box.add_field(
        "fig1",
        "contour",
        expression="sin(x)*cos(y*k)",
        x_range=[-3, 3],
        y_range=[-3, 3],
        resolution=40,
        parameters={"k": 1},
        levels=5,
    )
    box.add_control("fig1", "L", min=3, max=20, value=5)
    box.bind_property("fig1", "a1", "levels", "L")
    box.animate("fig1", frames=3, stages=[{"frames": 3, "animate": {"L": [3, 20]}}])
    frames_match(box, monkeypatch, [(i, {"L": v}) for i, v in enumerate((3.0, 11.5, 20.0))])


def server_frame(box, index):
    """The server's picture of frame ``index`` of the figure's last animation."""
    from glplot.mcp import animator

    entry = box._figures["fig1"]
    spec = entry.animation
    timeline = timeline_of(
        {k: v for k, v in spec.items() if v is not None}, controls=box._context_values(entry)
    )
    combined = animator._merge(entry.bindings, animator.parse_bindings(spec["bindings"]))
    per_artist, per_axes, _ = box._bound_arguments(entry, combined, timeline.arrays(), timeline.n)
    built = box._build(entry, artist_args=per_artist[index], axes_args=per_axes[index])
    try:
        return picture(built.plot)
    finally:
        box._release(built)


def test_a_step_by_step_explainer_script_matches_the_server_frame_by_frame(box, monkeypatch):
    # the cookbook's 3Blue1Brown-style recipe: effects hide, draw, write, pulse and fade, and the
    # 3b1b preset changes the export (fonts, axes through the origin, tips, grid)
    from glplot.mcp import cookbook

    recipe = next(r for r in cookbook.COOKBOOK if r.key == "explainer_3b1b")
    for tool, args in recipe.calls:
        args = dict(args)
        if tool == "animate":
            args.pop("output", None)
            args["fps"] = 6
        getattr(box, tool)(**args)
    text = source(box.export_script("fig1", "animation"))
    assert "fx_a1_draw" in text and "plt.plot_style('3b1b')" in text
    namespace = run(text, monkeypatch)
    try:
        n = namespace["N"]
        shots = []
        for index in (0, 2, 5, 9, n // 2, n - 6, n - 1):
            activate(namespace)
            namespace["update"](index)
            shots.append(picture(namespace["fig"]))
            assert_same_picture(shots[-1], server_frame(box, index))
        changed = [not np.array_equal(a, b) for a, b in zip(shots, shots[1:])]
        assert all(changed), "every compared frame shows a different step of the scene"
    finally:
        gplt.close(namespace["fig"])


def test_a_pulse_returns_in_the_script_as_it_does_on_the_server(box, monkeypatch):
    box.create_figure(width=500, height=300)
    box.add_function("fig1", "k*x", 0, 1, parameters={"k": 1})
    box.animate(
        "fig1",
        variables={"k": 1},
        stages=[
            {"frames": 5, "animate": {"k": {"from": 1, "to": 3, "ease": "there_and_back"}}},
            {"frames": 3, "ease": "rush_into", "animate": {"k": 2}},
        ],
        bindings=[{"target": "a1", "property": "parameters.k", "value": "k"}],
    )
    namespace = run(source(box.export_script("fig1", "animation")), monkeypatch)
    try:
        expected = timeline_of(box._figures["fig1"].animation).variables["k"]
        assert np.allclose(namespace["_tl_k"], expected) and expected[4] == pytest.approx(1)
    finally:
        gplt.close(namespace["fig"])


def test_annotations_of_every_new_kind_in_a_script(box, monkeypatch):
    box.create_figure(width=500, height=400, style="3b1b")
    box.set_axes("fig1", xlim=[-1, 3], ylim=[-1, 3], legend=False)
    box.add_annotation("fig1", "arrow", x=0, y=0, dx=2, dy=1, reveal=0.5)
    box.add_annotation(
        "fig1", "text", x=0.5, y=0.9, coords="axes", align="center", text="$v$ grows", alpha=0.6
    )
    box.add_annotation("fig1", "text", x=0, y=2, text="half written", reveal=0.5)
    box.add_annotation(
        "fig1", "text", x=1, y=0.5, align="center", text="centred, half", reveal=0.5,
        background=True,
    )  # fmt: skip
    box.add_plot("fig1", "scatter", x=[1], y=[1], size=20)
    reply, _ = check_figure(box, monkeypatch)
    assert "layout_text='centred, half'" in source(reply) and "bbox=" in source(reply)


def test_choose_binding_in_a_script(box, monkeypatch):
    box.create_figure(width=500, height=300)
    box.add_plot("fig1", "line", x=[0, 1, 2], y=[0, 1, 0])
    box.animate(
        "fig1",
        frames=4,
        bindings=[
            {
                "target": "a1",
                "property": "color",
                "value": {"choose": ["red", "green", "blue"], "index": "floor(t*3)"},
            }
        ],
    )
    run(source(box.export_script("fig1", "animation")), monkeypatch)["update"](2)


# ======================================================================================
# refusals and unit pieces
# ======================================================================================


def test_export_errors(box):
    box.create_figure()
    with pytest.raises(McpError, match="nothing drawn"):
        box.export_script("fig1")
    box.add_plot("fig1", "line", x=[0, 1], y=[0, 1])
    with pytest.raises(McpError, match="kind must be one of"):
        box.export_script("fig1", "video")
    with pytest.raises(McpError, match="no controls"):
        box.export_script("fig1", "interactive")
    with pytest.raises(McpError, match="no animation yet"):
        box.export_script("fig1", "animation")
    with pytest.raises(McpError, match="No figure"):
        box.export_script("fig9")


def test_oversized_scripts_are_refused_with_the_reason(box):
    box.create_figure()
    box.add_plot("fig1", "scatter", x=list(range(2000)), y=list(range(2000)))
    for _ in range(40):
        box.add_plot("fig1", "line", x=list(range(2000)), y=list(range(2000)))
    with pytest.raises(McpError, match="characters, over the"):
        box.export_script("fig1")


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("sin(x) + cos(2*x)", "np.sin(x) + np.cos(2 * x)"),
        ("pi * x**2", "np.pi * x ** 2"),
        ("tau", "2 * np.pi"),
        ("max(x) - min(x)", "np.max(x) - np.min(x)"),
        ("where(x > 0, x, 0)", "np.where(x > 0, x, 0)"),
        ("a*x if b else 0", "a * x if b else 0"),
    ],
)
def test_expressions_become_numpy(expression, expected):
    assert script.translate(expression, {}) == expected


def test_translation_renames_variables_and_collects_helpers():
    helpers = {}
    out = script.translate(
        "noise(x, 0.1) + smoothstep(t) * amp", helpers, {"t": "_t", "amp": "_tl_amp"}
    )
    assert out == "noise(x, 0.1) + smoothstep(_t) * _tl_amp"
    assert set(helpers) == {"noise", "smoothstep"}


def test_helper_sources_are_valid_python_that_runs(tmp_path):
    names = sorted(dynamics.HELPERS) + [
        "noise",
        "gauss",
        "sigmoid",
        "heaviside",
        "sawtooth",
        "square",
        "rect",
        "step",
        "interp",
    ]
    import inspect

    from glplot.gui import expressions

    parts = [
        "from __future__ import annotations",
        "",
        "import numpy as np",
        "",
        textwrap.dedent(inspect.getsource(expressions._unit)),
    ]
    for name in names:
        parts.append(script._helper_source(name))
    parts.append("print(float(smoothstep(0.5)), float(ease_in_out(0.5)), float(lerp(1, 3, 0.5)))")
    program = tmp_path / "h.py"
    program.write_text("\n\n".join(parts) + "\n")
    done = subprocess.run(
        [sys.executable, str(program)], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["0.5", "0.5", "2.0"]


def test_literals():
    assert script.lit([1, 2.5, None, "a", True]) == "[1, 2.5, None, 'a', True]"
    assert script.lit({"a": [1]}) == "{'a': [1]}"
    assert script.lit(float("nan")) == "np.nan" and script.lit(float("-inf")) == "-np.inf"
    assert script.lit(np.array([1.0, 2.0])) == "[1.0, 2.0]"
    assert script.lit(script.Raw("x_[-1:]")) == "x_[-1:]"
    assert script.lit(np.float64(3)) == "3.0" and script.lit(np.int64(3)) == "3"
