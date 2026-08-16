"""Outline / silhouette rendering: the ``LayerStyle`` contract and the 3D draw path.

Two kinds of test live here, because the feature has two halves that fail in different
ways.

The first half is the **state machine**: how many draws the renderer issues, which
uniforms it pushes, and -- above all -- that a layer which never asked for an outline
draws exactly as it did before the feature existed. That is checked without a GL context
at all, by swapping the ``OpenGL.GL`` entry points the renderer module star-imported for
recorders. It is the half that a headless CI can run, and the half a regression would
actually break.

The second half is the **shader**: outline uniforms are useless if the program stops
linking or the driver optimises a uniform away. That needs a real context, and a real
context is exactly what this process must not have -- several suites here replace GL entry
points with mocks, and a mock reaching a live driver segfaults the interpreter instead of
failing a test. So the link runs in a throwaway subprocess. A skip is the honest result on
a headless box; a link error on a machine that does have GL is a real failure and is
reported as one, with the driver's info log.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from glplot.core.context import RenderContext
from glplot.core.layers import Layer3D, LayerStyle
from glplot.options import EngineOptions, RenderMode
from glplot.renderers import geometry3d as g3d
from glplot.utils.shaders import GEOMETRY3D_FS, GEOMETRY3D_VS

# ----------------------------------------------------------------------------------
# LayerStyle
# ----------------------------------------------------------------------------------


class TestLayerStyleOutlineFields:
    """The new fields exist, default to "off", and leave the old scatter ones alone."""

    def test_outline_defaults_are_off(self):
        style = LayerStyle()
        assert style.outline_enabled is False
        assert style.outline_color == (0.0, 0.0, 0.0, 1.0)
        assert style.outline_width == 1.5
        assert style.outline_alpha == 1.0

    def test_point_outline_fields_are_untouched(self):
        """The 2D scatter's own ring keeps its name, its default and its meaning."""
        style = LayerStyle()
        assert style.point_outline_enabled is False
        assert style.point_outline_color == (0.0, 0.0, 0.0, 1.0)
        assert style.point_outline_width == 1.0

    def test_general_and_point_outline_are_independent_switches(self):
        style = LayerStyle()
        style.outline_enabled = True
        assert style.point_outline_enabled is False
        style.point_outline_enabled = True
        style.outline_enabled = False
        assert style.point_outline_enabled is True

    def test_every_layer_carries_the_fields(self):
        """``LayerStyle`` is shared by every layer type, so 2D layers get them too."""
        layer = Layer3D(vertices=np.zeros((3, 3), dtype=np.float32), primitive="triangles")
        assert layer.style.outline_enabled is False
        assert layer.style.outline_width == 1.5


# ----------------------------------------------------------------------------------
# The dilation sampling
# ----------------------------------------------------------------------------------


class TestOutlineOffsets:
    def test_zero_or_negative_width_yields_no_copies(self):
        assert len(g3d._outline_offsets(0.0, solid=False)) == 0
        assert len(g3d._outline_offsets(-2.0, solid=True)) == 0
        assert len(g3d._outline_offsets(float("nan"), solid=False)) == 0

    def test_solid_primitives_use_the_fixed_cheap_ring(self):
        for width in (1.0, 4.0, 20.0):
            assert len(g3d._outline_offsets(width, solid=True)) == g3d.OUTLINE_SOLID_STEPS

    def test_thin_primitives_sample_more_densely_as_the_width_grows(self):
        thin_1px = len(g3d._outline_offsets(1.0, solid=False))
        thin_3px = len(g3d._outline_offsets(3.0, solid=False))
        assert thin_1px == g3d.OUTLINE_MIN_STEPS
        assert thin_3px > thin_1px
        assert len(g3d._outline_offsets(100.0, solid=False)) == g3d.OUTLINE_MAX_STEPS

    def test_offsets_lie_on_the_circle_of_the_requested_radius(self):
        offsets = g3d._outline_offsets(2.5, solid=False)
        radii = np.hypot(offsets[:, 0], offsets[:, 1])
        assert np.allclose(radii, 2.5, atol=1e-5)
        # A full ring: the displacements cancel, so the outline is centred on the geometry
        # rather than pulling it in one direction.
        assert np.allclose(offsets.sum(axis=0), 0.0, atol=1e-4)


# ----------------------------------------------------------------------------------
# The draw path, without a GL context
# ----------------------------------------------------------------------------------

#: Every uniform ``Geometry3DRenderer`` resolves. The stubbed draw path assigns each one a
#: distinct fake location so a recorded ``glUniform*`` call can be named again.
UNIFORM_NAMES = (
    "u_mvp",
    "u_alpha",
    "u_point_size",
    "u_z_range",
    "u_ssao_enabled",
    "u_ssao_strength",
    "u_is_points",
    "u_ref_w",
    "u_clip_lo",
    "u_clip_hi",
    "u_outline_enabled",
    "u_outline_pass",
    "u_outline_color",
    "u_outline_width",
    "u_outline_alpha",
    "u_outline_offset",
    "u_outline_depth_bias",
    "u_viewport",
)


class GLRecorder:
    """A stand-in for the GL entry points ``geometry3d`` star-imported.

    Records everything the renderer does in order, which is what lets a test say "an
    outline costs exactly K extra draws and touches no other state".
    """

    def __init__(self) -> None:
        self.location_names = {i: name for i, name in enumerate(UNIFORM_NAMES)}
        self.uniforms: list[tuple[str, tuple]] = []
        self.draws: list[tuple] = []
        self.state: list[tuple[str, int]] = []

    # -- uniforms ------------------------------------------------------------
    def _record(self, loc, *values):
        self.uniforms.append((self.location_names.get(int(loc), f"?{loc}"), values))

    def uniform_matrix4fv(self, loc, count, transpose, value):
        self._record(loc, "mat4")

    # -- draws ---------------------------------------------------------------
    def draw_arrays(self, mode, first, count):
        self.draws.append(("arrays", int(mode), int(count), self.last_outline_pass()))

    def draw_elements(self, mode, count, gl_type, indices):
        self.draws.append(("elements", int(mode), int(count), self.last_outline_pass()))

    # -- helpers -------------------------------------------------------------
    def last_outline_pass(self) -> int:
        """The value of ``u_outline_pass`` at the moment of the call being recorded."""
        for name, values in reversed(self.uniforms):
            if name == "u_outline_pass":
                return int(values[0])
        return 0

    def last(self, name: str):
        for uname, values in reversed(self.uniforms):
            if uname == name:
                return values
        return None

    def values(self, name: str) -> list:
        return [values for uname, values in self.uniforms if uname == name]


@pytest.fixture
def recorder(monkeypatch) -> GLRecorder:
    """Swap the module's GL calls for recorders, so ``draw()`` runs headless."""
    rec = GLRecorder()
    monkeypatch.setattr(g3d, "glEnable", lambda cap: rec.state.append(("enable", int(cap))))
    monkeypatch.setattr(g3d, "glDisable", lambda cap: rec.state.append(("disable", int(cap))))
    monkeypatch.setattr(g3d, "glUseProgram", lambda prog: None)
    monkeypatch.setattr(g3d, "glBindVertexArray", lambda vao: None)
    monkeypatch.setattr(g3d, "glUniformMatrix4fv", rec.uniform_matrix4fv)
    for n in (1, 2, 3, 4):
        for suffix in ("f", "i"):
            name = f"glUniform{n}{suffix}"
            if hasattr(g3d, name):
                monkeypatch.setattr(g3d, name, rec._record)
    monkeypatch.setattr(g3d, "glDrawArrays", rec.draw_arrays)
    monkeypatch.setattr(g3d, "glDrawElements", rec.draw_elements)
    return rec


def make_renderer() -> g3d.Geometry3DRenderer:
    """A renderer with fake -- but distinct and non-negative -- uniform locations."""
    renderer = g3d.Geometry3DRenderer(EngineOptions())
    renderer.prog = 1
    for i, name in enumerate(UNIFORM_NAMES):
        setattr(renderer, name, i)
    return renderer


def make_layer(primitive: str, *, indexed: bool = False) -> Layer3D:
    """A tiny 3D layer with its GPU buffers already "uploaded" (nothing to re-upload)."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.5], [1.0, 1.0, 1.0], [0.0, 1.0, 0.25]],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32) if indexed else None
    layer = Layer3D(vertices=vertices, indices=indices, primitive=primitive)
    layer._gl = g3d.GLGeometry3DBuffers(
        vao=7,
        vbo_pos=1,
        vbo_col=2,
        ebo=3 if indexed else 0,
        count=len(vertices) if indices is None else len(indices),
    )
    layer.dirty.gpu_dirty = False
    return layer


def make_ctx(**kwargs) -> RenderContext:
    params = dict(
        mvp=np.eye(4, dtype=np.float32),
        window_world=(-1.0, 1.0, -1.0, 1.0),
        width_px=400,
        height_px=300,
        fb_width=800,
        fb_height=600,
        mode=RenderMode.EXACT,
        dpr=2.0,
    )
    params.update(kwargs)
    return RenderContext(**params)


class TestOutlineOffIsUnchanged:
    """Default styles must draw exactly what they drew before outlines existed."""

    @pytest.mark.parametrize("primitive", ["points", "lines", "triangles"])
    def test_single_draw_call(self, recorder, primitive):
        layer = make_layer(primitive)
        make_renderer().draw(layer, make_ctx())
        assert len(recorder.draws) == 1
        assert recorder.draws[0][3] == 0  # not an outline pass

    def test_indexed_geometry_still_draws_through_the_element_buffer(self, recorder):
        layer = make_layer("triangles", indexed=True)
        make_renderer().draw(layer, make_ctx())
        assert [d[0] for d in recorder.draws] == ["elements"]
        assert recorder.draws[0][2] == 6

    def test_outline_uniforms_are_pushed_disabled(self, recorder):
        """Pushed unconditionally, so no layer inherits the previous layer's outline."""
        make_renderer().draw(make_layer("triangles"), make_ctx())
        assert recorder.last("u_outline_enabled") == (0,)
        assert recorder.last("u_outline_pass") == (0,)
        assert recorder.last("u_outline_width") == (0.0,)
        assert recorder.last("u_outline_offset") == (0.0, 0.0)

    @pytest.mark.parametrize("outline", [False, True])
    def test_gl_state_is_the_historical_sequence(self, recorder, outline):
        """No culling, no polygon offset, no depth-mask games -- with or without outlines.

        Run for both, because a technique that needed extra GL state would be a technique
        that can leak it into the next layer.
        """
        layer = make_layer("triangles")
        layer.style.outline_enabled = outline
        make_renderer().draw(layer, make_ctx())
        assert recorder.state == [
            ("enable", int(g3d.GL_DEPTH_TEST)),
            ("enable", int(g3d.GL_PROGRAM_POINT_SIZE)),
            ("disable", int(g3d.GL_PROGRAM_POINT_SIZE)),
            ("disable", int(g3d.GL_DEPTH_TEST)),
        ]

    def test_geometry_is_not_touched(self, recorder):
        layer = make_layer("triangles", indexed=True)
        vertices, indices = layer.vertices.copy(), layer.indices.copy()
        buffers = g3d.GLGeometry3DBuffers(**vars(layer._gl))
        make_renderer().draw(layer, make_ctx())
        assert np.array_equal(layer.vertices, vertices)
        assert np.array_equal(layer.indices, indices)
        assert vars(layer._gl) == vars(buffers)

    @pytest.mark.parametrize("width", [0.0, -1.0])
    def test_zero_width_is_as_good_as_disabled(self, recorder, width):
        layer = make_layer("lines")
        layer.style.outline_enabled = True
        layer.style.outline_width = width
        make_renderer().draw(layer, make_ctx())
        assert len(recorder.draws) == 1
        assert recorder.last("u_outline_enabled") == (0,)


class TestOutlineOnDrawPath:
    def test_lines_get_one_dilation_copy_per_direction(self, recorder):
        layer = make_layer("lines")
        layer.style.outline_enabled = True
        layer.style.outline_width = 1.5
        ctx = make_ctx()
        expected = len(g3d._outline_offsets(1.5 * ctx.dpr, solid=False))
        make_renderer().draw(layer, ctx)
        assert len(recorder.draws) == expected + 1
        # Depth-tested layer: the geometry goes down first so it owns the depth buffer, and
        # the depth-biased copies that follow survive only outside it.
        assert [d[3] for d in recorder.draws] == [0] + [1] * expected

    def test_axis_overlay_draws_its_outline_first_instead(self, recorder):
        """The axis wireframe skips the depth test, so only draw order can order it."""
        layer = make_layer("lines")
        layer.metadata["artist"] = "axis3d"
        layer.style.outline_enabled = True
        make_renderer().draw(layer, make_ctx())
        assert recorder.draws[-1][3] == 0  # the real geometry is last
        assert all(d[3] == 1 for d in recorder.draws[:-1])
        assert ("enable", int(g3d.GL_DEPTH_TEST)) not in recorder.state

    def test_triangles_use_the_cheap_solid_ring(self, recorder):
        layer = make_layer("triangles", indexed=True)
        layer.style.outline_enabled = True
        layer.style.outline_width = 4.0
        make_renderer().draw(layer, make_ctx())
        assert len(recorder.draws) == g3d.OUTLINE_SOLID_STEPS + 1
        assert all(d[0] == "elements" for d in recorder.draws)

    def test_points_get_no_extra_draw_at_all(self, recorder):
        """The point ring is analytic: one pass, with the ring inside the sprite."""
        layer = make_layer("points")
        layer.style.outline_enabled = True
        make_renderer().draw(layer, make_ctx())
        assert len(recorder.draws) == 1
        assert recorder.last("u_outline_enabled") == (1,)

    def test_offsets_are_the_dilation_ring_in_framebuffer_pixels(self, recorder):
        layer = make_layer("lines")
        layer.style.outline_enabled = True
        layer.style.outline_width = 2.0
        ctx = make_ctx(dpr=2.0)
        make_renderer().draw(layer, ctx)
        # (0, 0) is pushed twice around the ring: once before the pass, once after.
        ring = [v for v in recorder.values("u_outline_offset") if v != (0.0, 0.0)]
        radii = [float(np.hypot(*v)) for v in ring]
        assert radii  # the pass really ran
        assert np.allclose(radii, 4.0, atol=1e-4)  # width * dpr

    def test_pass_flag_and_offset_are_restored_before_the_real_draw(self, recorder):
        layer = make_layer("lines")
        layer.style.outline_enabled = True
        make_renderer().draw(layer, make_ctx())
        assert recorder.last("u_outline_pass") == (0,)
        assert recorder.last("u_outline_offset") == (0.0, 0.0)

    def test_width_is_scaled_by_the_device_pixel_ratio(self, recorder):
        layer = make_layer("points")
        layer.style.outline_enabled = True
        layer.style.outline_width = 1.5
        make_renderer().draw(layer, make_ctx(dpr=3.0))
        assert recorder.last("u_outline_width") == pytest.approx((4.5,))

    def test_colour_and_alpha_reach_the_shader(self, recorder):
        layer = make_layer("triangles")
        layer.style.outline_enabled = True
        layer.style.outline_color = (1.0, 0.5, 0.25, 0.75)
        layer.style.outline_alpha = 0.5
        layer.style.alpha = 0.2  # deliberately NOT folded in
        make_renderer().draw(layer, make_ctx(global_alpha=0.8))
        assert recorder.last("u_outline_color") == pytest.approx((1.0, 0.5, 0.25, 0.75))
        assert recorder.last("u_outline_alpha") == pytest.approx((0.4,))  # 0.5 * global 0.8

    def test_viewport_uniform_is_the_panel_rect_not_the_window(self, recorder):
        layer = make_layer("lines")
        layer.style.outline_enabled = True
        make_renderer().draw(layer, make_ctx(fb_width=321, fb_height=123))
        assert recorder.last("u_viewport") == pytest.approx((321.0, 123.0))

    def test_depth_bias_is_pushed_so_the_copies_lose_the_depth_test(self, recorder):
        layer = make_layer("triangles")
        layer.style.outline_enabled = True
        make_renderer().draw(layer, make_ctx())
        assert recorder.last("u_outline_depth_bias") == pytest.approx((g3d.OUTLINE_DEPTH_BIAS,))
        assert g3d.OUTLINE_DEPTH_BIAS > 0.0  # positive = pushed towards the far plane


# ----------------------------------------------------------------------------------
# The shader itself
# ----------------------------------------------------------------------------------


def _declares_uniform(source: str, name: str) -> bool:
    """True when ``source`` has a real ``uniform <type> <name>;`` declaration.

    A substring search would pass on a mention in a comment, and the comments in these
    shaders name every uniform they explain.
    """
    return (
        re.search(rf"^\s*uniform\s+\w+\s+{re.escape(name)}\s*;", source, re.MULTILINE) is not None
    )


class TestShaderSource:
    """Source-level checks that need no driver, so they run everywhere."""

    @pytest.mark.parametrize(
        "uniform",
        [
            "u_outline_enabled",
            "u_outline_pass",
            "u_outline_width",
            "u_outline_offset",
            "u_viewport",
            "u_outline_depth_bias",
        ],
    )
    def test_vertex_shader_declares_the_outline_uniforms(self, uniform):
        assert _declares_uniform(GEOMETRY3D_VS, uniform)

    @pytest.mark.parametrize(
        "uniform",
        ["u_outline_enabled", "u_outline_pass", "u_outline_color", "u_outline_alpha"],
    )
    def test_fragment_shader_declares_the_outline_uniforms(self, uniform):
        assert _declares_uniform(GEOMETRY3D_FS, uniform)

    def test_ring_and_dilation_are_both_gated(self):
        """Nothing outline-related may run when the feature is off."""
        assert "u_outline_enabled == 1" in GEOMETRY3D_FS
        assert "u_outline_pass == 1" in GEOMETRY3D_FS
        assert "u_outline_pass == 1" in GEOMETRY3D_VS

    def test_fill_fraction_is_carried_from_vertex_to_fragment(self):
        assert "out float v_fill_frac;" in GEOMETRY3D_VS
        assert "in float v_fill_frac;" in GEOMETRY3D_FS


#: Linking the real program needs a real GL context, and a real GL context must not exist
#: in the pytest process: several suites here replace ``OpenGL.GL`` entry points with
#: mocks, and those mocks reach the driver for real once a context is current -- which
#: segfaults the interpreter rather than failing a test. So the link happens in a
#: throwaway subprocess that reports back as one line of JSON.
_LINK_PROBE = r"""
import json, sys

def emit(payload):
    sys.stdout.write("GLPLOT_PROBE " + json.dumps(payload) + "\n")

try:
    import glfw
except Exception as exc:
    emit({"context": False, "reason": "glfw unavailable: %s" % exc})
    raise SystemExit

if not glfw.init():
    emit({"context": False, "reason": "glfw.init() failed"})
    raise SystemExit

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
try:
    window = glfw.create_window(64, 64, "glplot-outline-probe", None, None)
except Exception as exc:
    window = None
if not window:
    emit({"context": False, "reason": "no GL context on this machine"})
    raise SystemExit
glfw.make_context_current(window)

from glplot.options import EngineOptions
from glplot.renderers.geometry3d import Geometry3DRenderer

names = json.loads(sys.argv[1])
try:
    renderer = Geometry3DRenderer(EngineOptions())
    renderer.initialize()
except Exception as exc:
    emit({"context": True, "linked": False, "error": str(exc)})
else:
    emit({
        "context": True,
        "linked": True,
        "prog": int(renderer.prog),
        "uniforms": {n: int(getattr(renderer, n)) for n in names},
    })
glfw.destroy_window(window)
glfw.terminate()
"""


@pytest.fixture(scope="module")
def link_probe():
    """Link ``GEOMETRY3D_VS``/``GEOMETRY3D_FS`` for real, in a subprocess.

    Skips when the machine cannot give the subprocess a context -- a skip means "not
    checked", never "passed". A context that *does* exist and a program that then fails to
    compile or link is a hard failure, carrying the driver's info log.
    """
    repo_root = Path(__file__).resolve().parents[1]
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _LINK_PROBE, json.dumps(list(UNIFORM_NAMES))],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover - environmental
        pytest.skip(f"could not run the GL probe: {exc}")

    line = next(
        (ln for ln in proc.stdout.splitlines() if ln.startswith("GLPLOT_PROBE ")),
        None,
    )
    if line is None:  # pragma: no cover - environmental
        pytest.skip(
            "GL probe produced no result (it most likely crashed before reaching a "
            f"context)\nstdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
        )
    result = json.loads(line[len("GLPLOT_PROBE ") :])
    if not result.get("context"):
        pytest.skip(f"no GL context available: {result.get('reason')}")
    return result


class TestShaderLinksAndResolvesUniforms:
    """Needs a driver. A skip here means "not checked", never "passed"."""

    def test_program_links(self, link_probe):
        assert link_probe.get("linked"), f"shader failed to build: {link_probe.get('error')}"
        assert link_probe["prog"] > 0

    def test_every_uniform_resolves_after_initialize(self, link_probe):
        assert link_probe.get("linked"), f"shader failed to build: {link_probe.get('error')}"
        # A uniform the driver optimised away resolves to -1, which would make the feature
        # silently dead -- exactly the failure this test exists to catch.
        unresolved = [name for name, loc in link_probe["uniforms"].items() if loc < 0]
        assert unresolved == []
