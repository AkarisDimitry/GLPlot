"""Stage effects (create, write, fade_in, indicate...) and the annotations an explainer is made of.

Effects are compiled into stage variables and bindings (:mod:`glplot.mcp.choreography`), so most
checks read the compiled timeline: an artist is hidden before the stage that brings it in, drawn
during it, held after. The pictures are checked too: a hidden artist really leaves no pixels.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from glplot.mcp import animator, choreography, dynamics
from glplot.mcp.state import FileSandbox, McpError
from glplot.mcp.toolbox import Toolbox


@pytest.fixture
def box(tmp_path):
    toolbox = Toolbox(FileSandbox([str(tmp_path)]))
    yield toolbox
    toolbox.close_all()


def scene(box, style="manim"):
    """Curve a1, title a2 (axes coords), dot a3, formula a4: the parts of an explainer."""
    box.create_figure(width=640, height=360, style=style)
    box.set_axes("fig1", xlim=[-1, 4], ylim=[-1, 5], xlabel="x", ylabel="y", legend=False)
    box.add_function("fig1", "0.5*x**2", x_min=-1, x_max=3)
    box.add_annotation(
        "fig1", "text", x=0.5, y=1.04, coords="axes", align="center", text="Title", fontsize=24
    )
    box.add_plot("fig1", "scatter", x=[1.0], y=[0.5], size=14, color="#FFFFFF")
    box.add_annotation("fig1", "text", x=0.1, y=0.8, coords="axes", text="$f'(x) = x$")


STAGES = [
    {"name": "draw", "frames": 4, "effects": {"create": "a1"}},
    {"name": "title", "frames": 4, "effects": {"write": "a2", "fade_in": "a3"}},
    {"name": "point", "frames": 4, "effects": {"indicate": "a4"}},
    {"name": "clear", "frames": 4, "effects": {"fade_out": ["a3", "a4"], "uncreate": "a1"}},
]


def timeline(stages=STAGES, variables=None, box=None):
    """The compiled stages' variables, frame by frame."""
    stages, variables, _bindings, _plan = choreography.compile_effects(
        box, box._figures["fig1"], stages, variables, None
    )
    return dynamics.build_timeline(stages=stages, variables=variables).variables


# ======================================================================================
# the compiled timeline
# ======================================================================================


def test_without_effects_nothing_is_touched(box):
    scene(box)
    stages, variables, bindings = [{"frames": 3}], {"k": 1}, [{"target": "a1"}]
    out = choreography.compile_effects(box, box._figures["fig1"], stages, variables, bindings)
    assert out[0] is stages and out[1] is variables and out[2] is bindings and out[3] is None


def test_an_artist_an_effect_brings_in_is_hidden_until_its_stage(box):
    scene(box)
    v = timeline(box=box)
    # a1 is created in stage 1 (frames 0-3): visible from its first frame, drawn 0 -> 1
    assert list(v["fx_a1_show"][:4]) == [1, 1, 1, 1]
    assert v["fx_a1_draw"][0] == 0 and v["fx_a1_draw"][3] == 1
    # a2 and a3 come in during stage 2 (frames 4-7): nothing of them before
    assert np.all(v["fx_a2_show"][:4] == 0) and np.all(v["fx_a2_draw"][:4] == 0)
    assert np.all(v["fx_a3_show"][:4] == 0) and v["fx_a3_show"][7] == 1
    assert v["fx_a2_draw"][7] == 1


def test_indicate_pulses_and_comes_back(box):
    scene(box)
    v = timeline(box=box)
    pulse = v["fx_a4_pulse"]
    assert np.all(pulse[:8] == 0) and pulse[8:12].max() > 0.5 and pulse[11] == pytest.approx(0)
    assert np.all(pulse[12:] == pytest.approx(0))


def test_taken_out_stays_out(box):
    scene(box)
    v = timeline(box=box)
    assert v["fx_a3_show"][-1] == 0 and v["fx_a4_show"][-1] == 0
    # uncreate draws the curve back and hides what reveal=0 would leave (its first point)
    assert v["fx_a1_draw"][-1] == 0 and v["fx_a1_show"][-1] == 0 and v["fx_a1_show"][12] == 1


def test_the_bindings_use_the_artists_own_look(box):
    scene(box)
    _, _, bindings, plan = choreography.compile_effects(
        box, box._figures["fig1"], STAGES, None, None
    )
    by = {(b["target"], b["property"]): b["value"] for b in bindings}
    assert by[("a1", "reveal")] == "fx_a1_draw" and by[("a1", "alpha")] == "fx_a1_show"
    assert by[("a4", "color")] == {"colormap": ["#ffffff", "#F7D96F"], "at": "fx_a4_pulse"}
    assert by[("a4", "fontsize")].startswith("12.0 * (1 + 0.2")
    assert plan["stages"][1] == {"stage": "title", "effects": {"write": ["a2"], "fade_in": ["a3"]}}


def test_manim_spellings_are_understood(box):
    scene(box)
    stages = [{"frames": 3, "effects": {"FadeIn": "a3", "ShowCreation": "a1", "Write": "a2"}}]
    _, _, _, plan = choreography.compile_effects(box, box._figures["fig1"], stages, None, None)
    assert set(plan["stages"][0]["effects"]) == {"fade_in", "create", "write"}


@pytest.mark.parametrize(
    "stages, fragment",
    [
        ([{"frames": 3, "effects": {"fade": "a1"}}], "unknown effect 'fade'.*fade_in"),
        ([{"frames": 3, "effects": {"create": "a9"}}], "no artist 'a9'"),
        ([{"frames": 3, "effects": {"create": 3}}], "artist id"),
        ([{"frames": 3, "effects": "create a1"}], "effects is an object"),
    ],
)
def test_effect_mistakes_are_named(box, stages, fragment):
    scene(box)
    with pytest.raises(McpError, match=fragment):
        box.animate("fig1", stages=stages)
    assert list(box._figures) == ["fig1"]


def test_an_effect_and_a_binding_on_the_same_property_is_refused(box):
    scene(box)
    with pytest.raises(McpError, match="a3.alpha is driven by an effect and by a binding"):
        box.animate(
            "fig1",
            stages=[{"frames": 3, "effects": {"fade_in": "a3"}}],
            bindings=[{"target": "a3", "property": "alpha", "value": "t"}],
        )


def test_a_variable_named_like_an_effect_channel_is_refused(box):
    scene(box)
    with pytest.raises(McpError, match="fx_a3_show"):
        box.animate(
            "fig1",
            variables={"fx_a3_show": 1},
            stages=[{"frames": 3, "effects": {"fade_in": "a3"}}],
        )


def test_what_cannot_be_drawn_in_part_fades_and_the_reply_says_so(box):
    box.create_figure(width=500, height=300, style="manim")
    box.add_field("fig1", "image", expression="sin(x)*cos(y)", x_range=[0, 3], y_range=[0, 3])
    result = box.animate("fig1", stages=[{"frames": 4, "effects": {"create": "a1"}}])
    assert result.data["effects"] == [{"stage": "stage 1", "effects": {"create": ["a1"]}}]
    assert any("fades it" in note for note in result.data["notes"])


def test_a_recipe_cannot_fade(box):
    box.create_figure(width=500, height=300)
    box.add_recipe("fig1", "signal_trace", sample=True)
    with pytest.raises(McpError, match="add_recipe"):
        box.animate("fig1", stages=[{"frames": 3, "effects": {"fade_in": "a1"}}])


# ======================================================================================
# the pictures
# ======================================================================================


def pixels(png):
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=int)


def test_frames_show_the_scene_being_built(box):
    scene(box)
    result = box.animate("fig1", stages=STAGES, sheet_frames=4)
    assert result.data["frames"] == 16 and not [
        i for i in result.data["issues"] if i["code"] == "static"
    ]
    assert result.data["effects"][0]["effects"] == {"create": ["a1"]}
    first = pixels(box.animate("fig1", frame=0, stages=STAGES).image)
    built = pixels(box.animate("fig1", frame=11, stages=STAGES).image)
    blue = lambda img: np.count_nonzero(  # noqa: E731 - manim's BLUE_C, the curve
        (np.abs(img - [88, 196, 221]).sum(axis=-1) < 60)
    )
    white = lambda img: np.count_nonzero(img.min(axis=-1) > 235)  # noqa: E731
    assert blue(first) < 0.2 * blue(built), "the curve is drawn in, not there from frame 0"
    assert white(built) > white(first), "the title, the dot and the formula have appeared"


def test_the_animation_records_the_compiled_spec_and_the_plan(box):
    scene(box)
    box.animate("fig1", stages=STAGES)
    recorded = box._figures["fig1"].animation
    assert all("effects" not in stage for stage in recorded["stages"])
    assert recorded["variables"]["fx_a1_draw"] == 0.0
    assert recorded["effects"][3]["effects"] == {"fade_out": ["a3", "a4"], "uncreate": ["a1"]}
    # what animate stored is what it ran: the timeline rebuilds from it on its own
    rebuilt = dynamics.build_timeline(
        stages=recorded["stages"], variables=recorded["variables"]
    ).variables
    assert set(rebuilt) >= {"fx_a1_draw", "fx_a4_pulse"}
    animator.parse_bindings(recorded["bindings"])


# ======================================================================================
# annotations an explainer is made of
# ======================================================================================


def layers_of(box, artist_id):
    entry = box._figures["fig1"]
    artist = next(a for a in entry.artists if a.artist_id == artist_id)
    return [layer for layer in entry.plot.scene.layers if layer.layer_id in artist.layer_ids]


def test_text_in_axes_coords_aligned_and_faded(box):
    box.create_figure(width=500, height=300)
    box.add_annotation(
        "fig1", "text", x=0.5, y=0.9, coords="axes", align="center", text="Hi", alpha=0.5
    )
    (layer,) = layers_of(box, "a1")
    assert layer.metadata["axes_fraction"] == (0.5, 0.9) and layer.metadata["ha"] == "center"
    assert layer.style.color[3] == pytest.approx(0.5)


def test_text_is_written_letter_by_letter_and_maths_appears_whole():
    text = "Area: $\\int_0^1 x\\,dx$ ok"
    assert dynamics.reveal_text(text, 0) == ""
    assert dynamics.reveal_text(text, 1) == text
    partial = [dynamics.reveal_text(text, f) for f in np.linspace(0, 1, 40)]
    assert all(p.count("$") % 2 == 0 for p in partial), "a formula is never cut in half"
    assert all(len(a) <= len(b) for a, b in zip(partial, partial[1:])), "it only grows"
    assert "Area: $\\int_0^1 x\\,dx$" in partial, "the formula arrives in one piece"


def test_partial_text_is_drawn(box):
    box.create_figure(width=500, height=300)
    box.add_annotation("fig1", "text", x=1, y=1, text="abcdefgh", reveal=0.5)
    (layer,) = layers_of(box, "a1")
    assert layer.text == "abcd"


def test_unset_annotation_colour_is_the_page_ink(box):
    box.create_figure(width=500, height=300, style="manim")
    box.add_annotation("fig1", "text", x=1, y=1, text="white on black")
    box.add_annotation("fig1", "hline", y=0)
    for artist in ("a1", "a2"):
        assert tuple(layers_of(box, artist)[0].style.color[:3]) == (1.0, 1.0, 1.0)
    box.create_figure(width=500, height=300)
    box.add_annotation("fig2", "text", x=1, y=1, text="black on white")
    entry = box._figures["fig2"]
    assert tuple(entry.plot.scene.layers[-1].style.color[:3]) == (0.0, 0.0, 0.0)


def test_an_arrow_is_a_vector_that_can_grow(box):
    box.create_figure(width=500, height=500, style="manim")
    box.set_axes("fig1", xlim=[-1, 3], ylim=[-1, 3], equal_aspect=True)
    box.add_annotation("fig1", "arrow", x=0, y=0, dx=2, dy=1)
    shaft = layers_of(box, "a1")
    assert len(shaft) >= 2, "a shaft and a head"
    # its colour came from the cycle, and is recorded for the exported script
    assert box._figures["fig1"].log[-1].resolved["color"] == "#58c4dd"
    box.add_annotation("fig1", "arrow", x=0, y=0, dx=2, dy=1, reveal=0)  # nothing to draw
    result = box.animate(
        "fig1",
        stages=[{"frames": 4, "effects": {"grow": "a1"}}],
        bindings=[{"target": "a2", "property": "dx", "value": "2*cos(pi*t)"}],
    )
    assert result.data["frames"] == 4


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        (dict(kind="arrow", x=0, y=0, dx=1), "needs x, y .* dx, dy"),
        (dict(kind="hline", y=0, coords="axes"), "coords='axes' places text"),
        (dict(kind="hline", y=0, reveal=0.5), "reveal applies to text"),
        (dict(kind="text", x=0, y=0, text="a", align="middle"), "align"),
        (dict(kind="text", x=0, y=0, text="a", alpha=2), "alpha is a number between 0 and 1"),
        (dict(kind="text", x=0, y=0, text="a", coords="pixels"), "coords is 'data'"),
    ],
)
def test_annotation_refusals(box, kwargs, fragment):
    box.create_figure(width=500, height=300)
    with pytest.raises(McpError, match=fragment):
        box.add_annotation("fig1", **kwargs)


def test_a_legend_switched_off_stays_off_and_is_not_asked_for(box):
    box.create_figure(width=500, height=300)
    box.add_function("fig1", "x", x_min=0, x_max=1, label="one")
    box.add_function("fig1", "2*x", x_min=0, x_max=1, label="two")
    box.set_axes("fig1", xlabel="x", ylabel="y", title="T")
    shown = box.render_figure("fig1").data
    assert shown["panels"][0].get("legend") == ["one", "two"]
    box.set_axes("fig1", legend=False)
    hidden = box.render_figure("fig1").data
    assert "legend" not in hidden["panels"][0]
    assert not [i for i in hidden["issues"] if i["code"] == "legend_missing"]


def test_a_2d_marker_size_is_a_diameter_in_pixels_like_a_3d_one(box):
    box.create_figure(width=500, height=300)
    box.add_plot("fig1", "scatter", x=[0], y=[0], size=10)
    box.add_plot("fig1", "scatter", x=[1], y=[0], size=20)
    small, big = (layers_of(box, a)[0].style.point_size for a in ("a1", "a2"))
    assert big == pytest.approx(4 * small), "twice the diameter is four times the area"
    assert small == pytest.approx((10 * 72 / 100) ** 2)
