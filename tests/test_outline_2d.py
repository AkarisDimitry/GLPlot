"""Outline / silhouette on the 2D primitives: polyline, patch, scatter, line family.

The 3D half of the feature is tested in ``tests/test_outline.py``; this is its counterpart,
and it is deliberately built the same way, because the same two things can break:

* **The draw path** -- how many draws a layer costs, which uniforms it pushes, and above
  all that a layer which never asked for an outline draws exactly what it drew before the
  feature existed. That runs with no GL context at all, by swapping the ``OpenGL.GL`` entry
  points each renderer star-imported for recorders.
* **The shaders** -- an outline uniform is useless if the program stops linking or the
  driver optimises the uniform away. That needs a real driver, and a real driver is exactly
  what this process must not have: other suites here replace GL entry points with mocks,
  and a mock reaching a live driver segfaults the interpreter instead of failing a test. So
  the link runs in a throwaway subprocess. A skip is the honest result on a headless box; a
  link failure on a machine that does have GL is reported as a failure, with the info log.

The preview bridge (``glplot.utils.preview``) needs neither: matplotlib's Agg backend runs
anywhere, so the export half is tested for real.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from glplot.core.context import RenderContext
from glplot.core.layers import (
    LayerStyle,
    LineFamilyLayer,
    PatchLayer,
    PolylineLayer,
    ScatterLayer,
    TextLayer,
)
from glplot.options import EngineOptions, RenderMode
from glplot.renderers import line_family as lf
from glplot.renderers import patch as pt
from glplot.renderers import polyline as pl
from glplot.renderers import scatter as sc
from glplot.renderers.geometry3d import OUTLINE_SOLID_STEPS
from glplot.utils.shaders import (
    EXACT_LINES_FS,
    PATCH_FS,
    PATCH_VS,
    SCATTER_FS,
    SCATTER_VS,
    WIDE_LINES_INSTANCED_VS,
    WIDE_SEGMENT_INSTANCED_FS,
    WIDE_SEGMENT_INSTANCED_VS,
)

# ----------------------------------------------------------------------------------
# Shared harness
# ----------------------------------------------------------------------------------

#: Every uniform the four 2D renderers resolve on their *exact* program. The stubbed draw
#: path gives each one a distinct fake location, so a recorded ``glUniform*`` call can be
#: named again. Ordered per renderer; the fake locations are per-renderer too.
UNIFORMS = {
    "polyline": (
        "u_mvp",
        "u_viewport",
        "u_width",
        "u_color",
        "u_alpha",
        "u_offset",
        "u_use_colormap",
        "u_scheme",
        "u_id_norm",
        "u_window",
        "u_vertex_color",
        "u_outline_pass",
        "u_outline_width",
        "u_outline_color",
        "u_outline_alpha",
    ),
    "patch": (
        "u_mvp",
        "u_color",
        "u_alpha",
        "u_offset",
        "u_window",
        "u_vertex_color",
        "u_outline_pass",
        "u_outline_offset",
        "u_outline_color",
        "u_outline_alpha",
        "u_viewport",
    ),
    "scatter": (
        "u_mvp",
        "u_size",
        "u_alpha",
        "u_offset",
        "u_point_size_px",
        "u_marker_shape",
        "u_point_outline_enabled",
        "u_point_outline_color",
        "u_point_outline_width_px",
        "u_outline_enabled",
        "u_outline_color",
        "u_outline_width",
        "u_outline_alpha",
    ),
    "line_family": (
        "u_ndc_scale",
        "u_ndc_offset",
        "u_xrange",
        "u_window",
        "u_use_color",
        "u_alpha",
        "u_width",
        "u_viewport",
        "u_keep_prob",
        "u_total_count",
        "u_offset",
        "u_use_colormap",
        "u_scheme",
        "u_antialiasing",
        "u_outline_pass",
        "u_outline_width",
        "u_outline_color",
        "u_outline_alpha",
    ),
}


class GLRecorder:
    """A stand-in for the GL entry points a renderer module star-imported.

    Records uniforms and draws in order, which is what lets a test say "an outline costs
    exactly K extra draws and touches nothing else".
    """

    def __init__(self, names: tuple[str, ...]) -> None:
        self.location_names = {i: name for i, name in enumerate(names)}
        self.uniforms: list[tuple[str, tuple]] = []
        self.draws: list[tuple] = []
        self.state: list[tuple[str, int]] = []

    # -- uniforms ------------------------------------------------------------
    def _record(self, loc, *values):
        self.uniforms.append((self.location_names.get(int(loc), f"?{loc}"), values))

    def uniform_matrix4fv(self, loc, count, transpose, value):
        self._record(loc, "mat4")

    # -- draws ---------------------------------------------------------------
    def _draw(self, kind, count):
        self.draws.append((kind, int(count), self.pass_flag()))

    def draw_arrays(self, mode, first, count):
        self._draw("arrays", count)

    def draw_elements(self, mode, count, gl_type, indices):
        self._draw("elements", count)

    def draw_arrays_instanced(self, mode, first, count, primcount):
        self._draw("arrays_instanced", primcount)

    def draw_elements_instanced(self, mode, count, gl_type, indices, primcount):
        self._draw("elements_instanced", primcount)

    # -- helpers -------------------------------------------------------------
    def pass_flag(self) -> int:
        """``u_outline_pass`` as it stood at the moment of the call being recorded."""
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


def install_recorder(monkeypatch, module, names: tuple[str, ...]) -> GLRecorder:
    """Swap ``module``'s GL calls for recorders so its ``draw()`` runs headless."""
    rec = GLRecorder(names)
    noop = lambda *a, **k: None  # noqa: E731 - a one-expression stub reads better inline
    monkeypatch.setattr(module, "glUseProgram", noop)
    monkeypatch.setattr(module, "glBindVertexArray", noop)
    monkeypatch.setattr(module, "glBindBuffer", noop)
    monkeypatch.setattr(module, "glBufferData", noop, raising=False)
    monkeypatch.setattr(module, "glBufferSubData", noop, raising=False)
    monkeypatch.setattr(module, "glEnable", lambda cap: rec.state.append(("enable", int(cap))))
    monkeypatch.setattr(module, "glDisable", lambda cap: rec.state.append(("disable", int(cap))))
    monkeypatch.setattr(module, "glUniformMatrix4fv", rec.uniform_matrix4fv)
    for n in (1, 2, 3, 4):
        for suffix in ("f", "i"):
            name = f"glUniform{n}{suffix}"
            if hasattr(module, name):
                monkeypatch.setattr(module, name, rec._record)
    monkeypatch.setattr(module, "glDrawArrays", rec.draw_arrays, raising=False)
    monkeypatch.setattr(module, "glDrawElements", rec.draw_elements, raising=False)
    monkeypatch.setattr(module, "glDrawArraysInstanced", rec.draw_arrays_instanced, raising=False)
    monkeypatch.setattr(
        module, "glDrawElementsInstanced", rec.draw_elements_instanced, raising=False
    )
    return rec


def make_renderer(cls, kind: str):
    """A renderer with fake -- but distinct and non-negative -- uniform locations."""
    renderer = cls(EngineOptions())
    renderer.prog = 1
    for i, name in enumerate(UNIFORMS[kind]):
        setattr(renderer, name, i)
    return renderer


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


def make_polyline() -> PolylineLayer:
    """A 4-point line with its GPU buffers already "uploaded" (nothing to re-upload).

    Four points is under the LOD floor, so ``_lod_stride`` returns 1 and the draw path does
    not try to re-upload anything.
    """
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]], dtype=np.float32)
    layer = PolylineLayer(pts=pts, color=(0.2, 0.4, 0.9, 1.0), width=2.0)
    layer._gl = pl.GLWideSegmentBuffers(vao=7, vbo_quad=1, vbo_inst=2, ebo=3, instance_count=3)
    layer.dirty.gpu_dirty = False
    return layer


def make_patch(*, indexed: bool = False) -> PatchLayer:
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32) if indexed else None
    layer = PatchLayer(vertices=vertices, indices=indices, mode="triangles" if indexed else "strip")
    layer.style.face_color = (0.2, 0.4, 0.8, 1.0)
    layer._gl = pt.GLPatchBuffers(vao=7, vbo=1, ebo=3 if indexed else None)
    layer._gl.count = 6 if indexed else 4
    layer.dirty.gpu_dirty = False
    return layer


def make_scatter() -> ScatterLayer:
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]], dtype=np.float32)
    layer = ScatterLayer(pts=pts, colors=np.ones((3, 4), dtype=np.float32), size=8.0)
    layer._gl = sc.GLScatterBuffers(vao=7, vbo_pts=1, vbo_col=2, vbo_size=3)
    layer._gl.count = 3
    layer.dirty.gpu_dirty = False
    return layer


def make_line_family(count: int = 64) -> LineFamilyLayer:
    ab = np.stack([np.linspace(-1.0, 1.0, count), np.linspace(0.0, 1.0, count)], axis=1).astype(
        np.float32
    )
    layer = LineFamilyLayer(ab=ab, x_range=(-1.0, 1.0))
    layer._gl = lf.GLLineBuffers()
    layer._gl.vao = 7
    layer._gl.has_color = False
    layer.dirty.gpu_dirty = False
    return layer


# ----------------------------------------------------------------------------------
# Off by default: the picture must not move
# ----------------------------------------------------------------------------------


class TestOutlineOffIsUnchanged:
    """A layer that never asked for an outline draws exactly what it always drew."""

    def test_every_2d_layer_carries_the_fields(self):
        for layer in (make_polyline(), make_patch(), make_scatter(), make_line_family()):
            assert layer.style.outline_enabled is False
            assert layer.style.outline_color == (0.0, 0.0, 0.0, 1.0)
            assert layer.style.outline_width == 1.5
            assert layer.style.outline_alpha == 1.0

    def test_polyline_draws_once(self, monkeypatch):
        rec = install_recorder(monkeypatch, pl, UNIFORMS["polyline"])
        make_renderer(pl.PolylineRenderer, "polyline").draw(make_polyline(), make_ctx())
        assert len(rec.draws) == 1
        assert rec.draws[0][2] == 0  # not an outline pass
        assert rec.last("u_outline_pass") == (0,)
        assert rec.last("u_outline_width") == (0.0,)

    def test_patch_draws_once(self, monkeypatch):
        rec = install_recorder(monkeypatch, pt, UNIFORMS["patch"])
        make_renderer(pt.PatchRenderer, "patch").draw(make_patch(), make_ctx())
        assert len(rec.draws) == 1
        assert rec.draws[0][2] == 0
        assert rec.last("u_outline_pass") == (0,)
        assert rec.last("u_outline_offset") == (0.0, 0.0)

    def test_scatter_draws_once(self, monkeypatch):
        rec = install_recorder(monkeypatch, sc, UNIFORMS["scatter"])
        make_renderer(sc.ScatterRenderer, "scatter").draw(make_scatter(), make_ctx())
        assert len(rec.draws) == 1
        assert rec.last("u_outline_enabled") == (0,)
        assert rec.last("u_outline_width") == (0.0,)

    def test_line_family_draws_once(self, monkeypatch):
        rec = install_recorder(monkeypatch, lf, UNIFORMS["line_family"])
        make_renderer(lf.LineFamilyRenderer, "line_family").draw(make_line_family(), make_ctx())
        assert len(rec.draws) == 1
        assert rec.last("u_outline_pass") == (0,)
        assert rec.last("u_outline_width") == (0.0,)

    def test_scatter_point_ring_still_works_and_is_a_separate_switch(self, monkeypatch):
        """``point_outline_*`` keeps its own uniforms and is not touched by the new flag."""
        rec = install_recorder(monkeypatch, sc, UNIFORMS["scatter"])
        layer = make_scatter()
        layer.style.point_outline_enabled = True
        layer.style.point_outline_color = (1.0, 0.0, 0.0, 1.0)
        layer.style.point_outline_width = 2.0
        make_renderer(sc.ScatterRenderer, "scatter").draw(layer, make_ctx(dpr=2.0))
        assert rec.last("u_point_outline_enabled") == (1,)
        assert rec.last("u_point_outline_color") == pytest.approx((1.0, 0.0, 0.0, 1.0))
        assert rec.last("u_point_outline_width_px") == pytest.approx((4.0,))
        # ...and the general outline stayed off, with no extra draw.
        assert rec.last("u_outline_enabled") == (0,)
        assert len(rec.draws) == 1

    @pytest.mark.parametrize("width", [0.0, -1.0, float("nan")])
    def test_zero_or_broken_width_is_as_good_as_disabled(self, monkeypatch, width):
        for module, cls, kind, make in (
            (pl, pl.PolylineRenderer, "polyline", make_polyline),
            (pt, pt.PatchRenderer, "patch", make_patch),
            (sc, sc.ScatterRenderer, "scatter", make_scatter),
            (lf, lf.LineFamilyRenderer, "line_family", make_line_family),
        ):
            with pytest.MonkeyPatch.context() as mp:
                rec = install_recorder(mp, module, UNIFORMS[kind])
                layer = make()
                layer.style.outline_enabled = True
                layer.style.outline_width = width
                make_renderer(cls, kind).draw(layer, make_ctx())
                assert len(rec.draws) == 1, kind

    def test_no_extra_gl_state_is_touched(self, monkeypatch):
        """Whatever the outline does, it must not leak enable/disable into the next layer."""
        for module, cls, kind, make in (
            (pl, pl.PolylineRenderer, "polyline", make_polyline),
            (pt, pt.PatchRenderer, "patch", make_patch),
            (lf, lf.LineFamilyRenderer, "line_family", make_line_family),
        ):
            for outline in (False, True):
                with pytest.MonkeyPatch.context() as mp:
                    rec = install_recorder(mp, module, UNIFORMS[kind])
                    layer = make()
                    layer.style.outline_enabled = outline
                    make_renderer(cls, kind).draw(layer, make_ctx())
                    assert rec.state == [], (kind, outline)

    def test_geometry_is_not_touched(self, monkeypatch):
        """An outline is a draw-time decision: it must not rewrite the layer's arrays."""
        install_recorder(monkeypatch, pt, UNIFORMS["patch"])
        layer = make_patch(indexed=True)
        layer.style.outline_enabled = True
        vertices, indices = layer.vertices.copy(), layer.indices.copy()
        make_renderer(pt.PatchRenderer, "patch").draw(layer, make_ctx())
        assert np.array_equal(layer.vertices, vertices)
        assert np.array_equal(layer.indices, indices)


# ----------------------------------------------------------------------------------
# On: the draw path of each technique
# ----------------------------------------------------------------------------------


class TestPolylineCasing:
    def test_costs_exactly_one_extra_draw(self, monkeypatch):
        rec = install_recorder(monkeypatch, pl, UNIFORMS["polyline"])
        layer = make_polyline()
        layer.style.outline_enabled = True
        make_renderer(pl.PolylineRenderer, "polyline").draw(layer, make_ctx())
        assert len(rec.draws) == 2
        # Casing under the stroke: 2D has no depth test, so order is the only arbiter.
        assert [d[2] for d in rec.draws] == [1, 0]

    def test_both_passes_draw_the_same_instances(self, monkeypatch):
        rec = install_recorder(monkeypatch, pl, UNIFORMS["polyline"])
        layer = make_polyline()
        layer.style.outline_enabled = True
        make_renderer(pl.PolylineRenderer, "polyline").draw(layer, make_ctx())
        assert {d[1] for d in rec.draws} == {layer._gl.instance_count}

    def test_width_is_scaled_by_the_device_pixel_ratio(self, monkeypatch):
        rec = install_recorder(monkeypatch, pl, UNIFORMS["polyline"])
        layer = make_polyline()
        layer.style.outline_enabled = True
        layer.style.outline_width = 1.5
        make_renderer(pl.PolylineRenderer, "polyline").draw(layer, make_ctx(dpr=3.0))
        assert rec.last("u_outline_width") == pytest.approx((4.5,))

    def test_pass_flag_is_restored_before_the_stroke(self, monkeypatch):
        rec = install_recorder(monkeypatch, pl, UNIFORMS["polyline"])
        layer = make_polyline()
        layer.style.outline_enabled = True
        make_renderer(pl.PolylineRenderer, "polyline").draw(layer, make_ctx())
        assert rec.last("u_outline_pass") == (0,)

    def test_colour_and_alpha_reach_the_shader(self, monkeypatch):
        rec = install_recorder(monkeypatch, pl, UNIFORMS["polyline"])
        layer = make_polyline()
        layer.style.outline_enabled = True
        layer.style.outline_color = (1.0, 0.5, 0.25, 0.75)
        layer.style.outline_alpha = 0.5
        layer.style.alpha = 0.2  # deliberately NOT folded in
        make_renderer(pl.PolylineRenderer, "polyline").draw(layer, make_ctx(global_alpha=0.8))
        assert rec.last("u_outline_color") == pytest.approx((1.0, 0.5, 0.25, 0.75))
        assert rec.last("u_outline_alpha") == pytest.approx((0.4,))  # 0.5 * global 0.8

    def test_density_pass_never_draws_a_casing(self, monkeypatch):
        """The accumulation program shares the vertex shader; it must stay a plain count."""
        rec = install_recorder(monkeypatch, pl, UNIFORMS["polyline"])
        layer = make_polyline()
        layer.style.outline_enabled = True
        renderer = make_renderer(pl.PolylineRenderer, "polyline")
        renderer.accum_prog = 2
        renderer.draw_density(layer, make_ctx())
        assert len(rec.draws) == 1


class TestPatchDilation:
    def test_costs_one_extra_draw_per_direction(self, monkeypatch):
        rec = install_recorder(monkeypatch, pt, UNIFORMS["patch"])
        layer = make_patch()
        layer.style.outline_enabled = True
        make_renderer(pt.PatchRenderer, "patch").draw(layer, make_ctx())
        assert len(rec.draws) == OUTLINE_SOLID_STEPS + 1
        # Copies first, fill last -- the only order painter's algorithm offers in 2D.
        assert [d[2] for d in rec.draws] == [1] * OUTLINE_SOLID_STEPS + [0]

    def test_indexed_geometry_still_draws_through_the_element_buffer(self, monkeypatch):
        rec = install_recorder(monkeypatch, pt, UNIFORMS["patch"])
        layer = make_patch(indexed=True)
        layer.style.outline_enabled = True
        make_renderer(pt.PatchRenderer, "patch").draw(layer, make_ctx())
        assert {d[0] for d in rec.draws} == {"elements"}
        assert {d[1] for d in rec.draws} == {6}

    def test_offsets_are_the_dilation_ring_in_framebuffer_pixels(self, monkeypatch):
        rec = install_recorder(monkeypatch, pt, UNIFORMS["patch"])
        layer = make_patch()
        layer.style.outline_enabled = True
        layer.style.outline_width = 2.0
        make_renderer(pt.PatchRenderer, "patch").draw(layer, make_ctx(dpr=2.0))
        ring = [v for v in rec.values("u_outline_offset") if v != (0.0, 0.0)]
        assert len(ring) == OUTLINE_SOLID_STEPS
        radii = [float(np.hypot(*v)) for v in ring]
        assert np.allclose(radii, 4.0, atol=1e-4)  # width * dpr
        # A full ring: the displacements cancel, so the silhouette is centred on the shape.
        assert np.allclose(np.asarray(ring).sum(axis=0), 0.0, atol=1e-4)

    def test_offset_and_pass_are_restored_before_the_fill(self, monkeypatch):
        rec = install_recorder(monkeypatch, pt, UNIFORMS["patch"])
        layer = make_patch()
        layer.style.outline_enabled = True
        make_renderer(pt.PatchRenderer, "patch").draw(layer, make_ctx())
        assert rec.last("u_outline_pass") == (0,)
        assert rec.last("u_outline_offset") == (0.0, 0.0)

    def test_viewport_uniform_is_the_panel_rect_not_the_window(self, monkeypatch):
        rec = install_recorder(monkeypatch, pt, UNIFORMS["patch"])
        layer = make_patch()
        layer.style.outline_enabled = True
        make_renderer(pt.PatchRenderer, "patch").draw(layer, make_ctx(fb_width=321, fb_height=123))
        assert rec.last("u_viewport") == pytest.approx((321.0, 123.0))

    def test_a_patch_with_no_face_gets_no_outline(self, monkeypatch):
        """An outline is the edge of a fill, not a substitute for one."""
        rec = install_recorder(monkeypatch, pt, UNIFORMS["patch"])
        layer = make_patch()
        layer.style.face_color = None
        layer.style.outline_enabled = True
        make_renderer(pt.PatchRenderer, "patch").draw(layer, make_ctx())
        assert rec.draws == []

    def test_colour_and_alpha_reach_the_shader(self, monkeypatch):
        rec = install_recorder(monkeypatch, pt, UNIFORMS["patch"])
        layer = make_patch()
        layer.style.outline_enabled = True
        layer.style.outline_color = (0.0, 1.0, 0.0, 1.0)
        layer.style.outline_alpha = 0.25
        layer.style.alpha = 0.1  # deliberately NOT folded in
        make_renderer(pt.PatchRenderer, "patch").draw(layer, make_ctx(global_alpha=0.5))
        assert rec.last("u_outline_color") == pytest.approx((0.0, 1.0, 0.0, 1.0))
        assert rec.last("u_outline_alpha") == pytest.approx((0.125,))

    def test_density_pass_never_dilates(self, monkeypatch):
        rec = install_recorder(monkeypatch, pt, UNIFORMS["patch"])
        layer = make_patch()
        layer.style.outline_enabled = True
        renderer = make_renderer(pt.PatchRenderer, "patch")
        renderer.accum_prog = 2
        renderer.u_accum_mvp = renderer.u_accum_alpha = 0
        renderer.u_accum_weighted = renderer.u_accum_window = 0
        renderer.draw_density(layer, make_ctx())
        assert len(rec.draws) == 1


class TestScatterRing:
    def test_costs_no_extra_draw_at_all(self, monkeypatch):
        """The ring is analytic: one pass, with the ring inside the (grown) sprite."""
        rec = install_recorder(monkeypatch, sc, UNIFORMS["scatter"])
        layer = make_scatter()
        layer.style.outline_enabled = True
        make_renderer(sc.ScatterRenderer, "scatter").draw(layer, make_ctx())
        assert len(rec.draws) == 1
        assert rec.last("u_outline_enabled") == (1,)

    def test_width_is_scaled_by_the_device_pixel_ratio(self, monkeypatch):
        rec = install_recorder(monkeypatch, sc, UNIFORMS["scatter"])
        layer = make_scatter()
        layer.style.outline_enabled = True
        layer.style.outline_width = 1.5
        make_renderer(sc.ScatterRenderer, "scatter").draw(layer, make_ctx(dpr=3.0))
        assert rec.last("u_outline_width") == pytest.approx((4.5,))

    def test_colour_and_alpha_reach_the_shader(self, monkeypatch):
        rec = install_recorder(monkeypatch, sc, UNIFORMS["scatter"])
        layer = make_scatter()
        layer.style.outline_enabled = True
        layer.style.outline_color = (0.1, 0.2, 0.3, 0.9)
        layer.style.outline_alpha = 0.5
        layer.style.alpha = 0.2  # deliberately NOT folded in
        make_renderer(sc.ScatterRenderer, "scatter").draw(layer, make_ctx(global_alpha=0.8))
        assert rec.last("u_outline_color") == pytest.approx((0.1, 0.2, 0.3, 0.9))
        assert rec.last("u_outline_alpha") == pytest.approx((0.4,))

    def test_both_rings_can_be_on_at_once(self, monkeypatch):
        rec = install_recorder(monkeypatch, sc, UNIFORMS["scatter"])
        layer = make_scatter()
        layer.style.outline_enabled = True
        layer.style.point_outline_enabled = True
        make_renderer(sc.ScatterRenderer, "scatter").draw(layer, make_ctx())
        assert rec.last("u_outline_enabled") == (1,)
        assert rec.last("u_point_outline_enabled") == (1,)
        assert len(rec.draws) == 1

    def test_density_pass_is_untouched(self, monkeypatch):
        rec = install_recorder(monkeypatch, sc, UNIFORMS["scatter"])
        layer = make_scatter()
        layer.style.outline_enabled = True
        renderer = make_renderer(sc.ScatterRenderer, "scatter")
        renderer.accum_prog = 2
        renderer.u_accum_mvp = renderer.u_accum_size = 0
        renderer.u_accum_alpha = renderer.u_accum_offset = 0
        renderer.draw_density(layer, make_ctx())
        assert len(rec.draws) == 1


class TestLineFamilyCasing:
    def test_a_small_family_gets_one_extra_draw(self, monkeypatch):
        rec = install_recorder(monkeypatch, lf, UNIFORMS["line_family"])
        layer = make_line_family(64)
        layer.style.outline_enabled = True
        make_renderer(lf.LineFamilyRenderer, "line_family").draw(layer, make_ctx())
        assert len(rec.draws) == 2
        assert [d[2] for d in rec.draws] == [1, 0]
        assert {d[1] for d in rec.draws} == {64}

    def test_a_family_over_the_cap_is_refused_with_a_warning(self, monkeypatch):
        rec = install_recorder(monkeypatch, lf, UNIFORMS["line_family"])
        layer = make_line_family(lf.OUTLINE_MAX_INSTANCES + 1)
        layer.style.outline_enabled = True
        renderer = make_renderer(lf.LineFamilyRenderer, "line_family")
        with pytest.warns(RuntimeWarning, match="outline_enabled ignored"):
            renderer.draw(layer, make_ctx())
        assert len(rec.draws) == 1  # the lines themselves, and nothing else
        assert rec.last("u_outline_pass") == (0,)

    def test_the_refusal_is_warned_about_once_not_once_per_frame(self, monkeypatch):
        install_recorder(monkeypatch, lf, UNIFORMS["line_family"])
        layer = make_line_family(lf.OUTLINE_MAX_INSTANCES + 1)
        layer.style.outline_enabled = True
        renderer = make_renderer(lf.LineFamilyRenderer, "line_family")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(5):
                renderer.draw(layer, make_ctx())
        assert len([w for w in caught if issubclass(w.category, RuntimeWarning)]) == 1

    def test_the_cap_is_exactly_inclusive(self, monkeypatch):
        """A family *at* the bound still gets its casing; one line more does not."""
        rec = install_recorder(monkeypatch, lf, UNIFORMS["line_family"])
        layer = make_line_family(lf.OUTLINE_MAX_INSTANCES)
        layer.style.outline_enabled = True
        make_renderer(lf.LineFamilyRenderer, "line_family").draw(layer, make_ctx())
        assert len(rec.draws) == 2

    def test_density_pass_never_draws_a_casing(self, monkeypatch):
        rec = install_recorder(monkeypatch, lf, UNIFORMS["line_family"])
        layer = make_line_family()
        layer.style.outline_enabled = True
        renderer = make_renderer(lf.LineFamilyRenderer, "line_family")
        renderer.accum_prog = 2
        for name in (
            "u_accum_ndc_scale",
            "u_accum_ndc_offset",
            "u_accum_xrange",
            "u_accum_window",
            "u_accum_use_color",
            "u_accum_alpha",
            "u_accum_width",
            "u_accum_viewport",
            "u_accum_keep_prob",
            "u_accum_offset",
        ):
            setattr(renderer, name, 0)
        renderer.draw_density(layer, make_ctx())
        assert len(rec.draws) == 1


# ----------------------------------------------------------------------------------
# The shader sources
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
        "source,uniform",
        [
            (WIDE_SEGMENT_INSTANCED_VS, "u_outline_pass"),
            (WIDE_SEGMENT_INSTANCED_VS, "u_outline_width"),
            (WIDE_SEGMENT_INSTANCED_FS, "u_outline_pass"),
            (WIDE_SEGMENT_INSTANCED_FS, "u_outline_color"),
            (WIDE_SEGMENT_INSTANCED_FS, "u_outline_alpha"),
            (PATCH_VS, "u_outline_pass"),
            (PATCH_VS, "u_outline_offset"),
            (PATCH_VS, "u_viewport"),
            (PATCH_FS, "u_outline_pass"),
            (PATCH_FS, "u_outline_color"),
            (PATCH_FS, "u_outline_alpha"),
            (SCATTER_VS, "u_outline_enabled"),
            (SCATTER_VS, "u_outline_width"),
            (SCATTER_FS, "u_outline_enabled"),
            (SCATTER_FS, "u_outline_color"),
            (SCATTER_FS, "u_outline_alpha"),
            (SCATTER_FS, "u_point_outline_enabled"),
            (SCATTER_FS, "u_point_outline_color"),
            (SCATTER_FS, "u_point_outline_width_px"),
            (WIDE_LINES_INSTANCED_VS, "u_outline_pass"),
            (WIDE_LINES_INSTANCED_VS, "u_outline_width"),
            (EXACT_LINES_FS, "u_outline_pass"),
            (EXACT_LINES_FS, "u_outline_color"),
            (EXACT_LINES_FS, "u_outline_alpha"),
        ],
    )
    def test_outline_uniforms_are_declared(self, source, uniform):
        assert _declares_uniform(source, uniform)

    @pytest.mark.parametrize(
        "source",
        [
            WIDE_SEGMENT_INSTANCED_VS,
            WIDE_SEGMENT_INSTANCED_FS,
            PATCH_VS,
            PATCH_FS,
            WIDE_LINES_INSTANCED_VS,
            EXACT_LINES_FS,
        ],
    )
    def test_every_outline_branch_is_gated(self, source):
        """Nothing outline-related may run when the pass flag is off."""
        assert "u_outline_pass == 1" in source

    def test_the_scatter_ring_is_gated_on_its_own_flag(self):
        assert "u_outline_enabled == 1" in SCATTER_VS
        assert "u_outline_enabled == 1" in SCATTER_FS
        assert "u_point_outline_enabled == 1" in SCATTER_FS

    def test_fill_fraction_is_carried_from_vertex_to_fragment(self):
        assert "out float v_fill_frac;" in SCATTER_VS
        assert "in float v_fill_frac;" in SCATTER_FS

    def test_the_stroke_width_is_the_uniform_plus_nothing_when_the_pass_is_off(self):
        """The casing widens a local copy, so the stroke's own float is untouched."""
        assert "float width = u_width;" in WIDE_SEGMENT_INSTANCED_VS
        assert "float width = u_width;" in WIDE_LINES_INSTANCED_VS

    def test_the_wide_segment_shader_still_writes_every_clip_distance(self):
        """Regression guard shared with tests/test_robustness.py: undefined clip distances
        make the driver cull vertices at random, which is how a solid line becomes dashes."""
        for i in range(4):
            assert f"gl_ClipDistance[{i}]" in WIDE_SEGMENT_INSTANCED_VS
            assert f"gl_ClipDistance[{i}]" in PATCH_VS


# ----------------------------------------------------------------------------------
# The programs, linked for real
# ----------------------------------------------------------------------------------

#: Linking needs a real GL context, and a real GL context must not exist in the pytest
#: process: several suites here replace ``OpenGL.GL`` entry points with mocks, and those
#: mocks reach the driver for real once a context is current -- which segfaults the
#: interpreter rather than failing a test. So the link happens in a throwaway subprocess
#: that reports back as one line of JSON. Same pattern, same reasons, as tests/test_outline.py.
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
    window = glfw.create_window(64, 64, "glplot-outline2d-probe", None, None)
except Exception:
    window = None
if not window:
    emit({"context": False, "reason": "no GL context on this machine"})
    raise SystemExit
glfw.make_context_current(window)

from glplot.options import EngineOptions
from glplot.renderers.line_family import LineFamilyRenderer
from glplot.renderers.patch import PatchRenderer
from glplot.renderers.polyline import PolylineRenderer
from glplot.renderers.scatter import ScatterRenderer

CLASSES = {
    "polyline": PolylineRenderer,
    "patch": PatchRenderer,
    "scatter": ScatterRenderer,
    "line_family": LineFamilyRenderer,
}

wanted = json.loads(sys.argv[1])
result = {"context": True, "renderers": {}}
for kind, names in wanted.items():
    try:
        renderer = CLASSES[kind](EngineOptions())
        renderer.initialize()
    except Exception as exc:
        result["renderers"][kind] = {"linked": False, "error": str(exc)}
    else:
        result["renderers"][kind] = {
            "linked": True,
            "prog": int(renderer.prog),
            "uniforms": {n: int(getattr(renderer, n)) for n in names},
        }
emit(result)
glfw.destroy_window(window)
glfw.terminate()
"""


@pytest.fixture(scope="module")
def link_probe():
    """Link all four 2D programs for real, in a subprocess.

    Skips when the machine cannot give the subprocess a context -- a skip means "not
    checked", never "passed". A context that *does* exist and a program that then fails to
    compile or link is a hard failure, carrying the driver's info log.
    """
    repo_root = Path(__file__).resolve().parents[1]
    wanted = {kind: list(names) for kind, names in UNIFORMS.items()}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _LINK_PROBE, json.dumps(wanted)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover - environmental
        pytest.skip(f"could not run the GL probe: {exc}")

    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("GLPLOT_PROBE ")), None)
    if line is None:  # pragma: no cover - environmental
        pytest.skip(
            "GL probe produced no result (it most likely crashed before reaching a "
            f"context)\nstdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
        )
    result = json.loads(line[len("GLPLOT_PROBE ") :])
    if not result.get("context"):
        pytest.skip(f"no GL context available: {result.get('reason')}")
    return result


#: Uniforms the driver is *expected* to optimise away, so the check below does not report
#: them. ``WIDE_LINES_INSTANCED_VS`` declares ``u_ndc_offset`` and then says in a comment of
#: its own that it does not need it -- the projection there is centre-relative. That is a
#: dead uniform that predates outlines by a long way; it is listed rather than fixed because
#: fixing it means touching a shader this change has no business touching.
DEAD_UNIFORMS = {"line_family": {"u_ndc_offset"}}


class TestProgramsLinkAndResolveUniforms:
    """Needs a driver. A skip here means "not checked", never "passed"."""

    @pytest.mark.parametrize("kind", sorted(UNIFORMS))
    def test_program_links(self, link_probe, kind):
        info = link_probe["renderers"][kind]
        assert info.get("linked"), f"{kind} shaders failed to build: {info.get('error')}"
        assert info["prog"] > 0

    @pytest.mark.parametrize("kind", sorted(UNIFORMS))
    def test_every_uniform_resolves_after_initialize(self, link_probe, kind):
        info = link_probe["renderers"][kind]
        assert info.get("linked"), f"{kind} shaders failed to build: {info.get('error')}"
        # A uniform the driver optimised away resolves to -1, which would make the feature
        # silently dead -- exactly the failure this test exists to catch.
        dead = DEAD_UNIFORMS.get(kind, set())
        unresolved = [n for n, loc in info["uniforms"].items() if loc < 0 and n not in dead]
        assert unresolved == []

    @pytest.mark.parametrize("kind", sorted(UNIFORMS))
    def test_no_outline_uniform_was_optimised_away(self, link_probe, kind):
        """Stated separately from the sweep above so a dead *outline* uniform can never be
        excused by the DEAD_UNIFORMS list, which is about pre-existing ones."""
        info = link_probe["renderers"][kind]
        assert info.get("linked"), f"{kind} shaders failed to build: {info.get('error')}"
        outline = {n: loc for n, loc in info["uniforms"].items() if "outline" in n}
        assert outline, f"{kind} resolves no outline uniform at all"
        assert [n for n, loc in outline.items() if loc < 0] == []


# ----------------------------------------------------------------------------------
# The matplotlib export bridge
# ----------------------------------------------------------------------------------


class TestPreviewHelpers:
    def test_an_untouched_style_asks_for_nothing(self):
        """The empty answers are what keep an untouched export byte-identical."""
        from glplot.utils import preview

        style = LayerStyle()
        assert preview._outline_rgba(style) is None
        assert preview._outline_path_effects(style) is None
        assert preview._outline_edge_kwargs(style) == {}

    @pytest.mark.parametrize("width", [0.0, -1.0])
    def test_a_zero_width_outline_asks_for_nothing_either(self, width):
        from glplot.utils import preview

        style = LayerStyle(outline_enabled=True, outline_width=width)
        assert preview._outline_rgba(style) is None
        assert preview._outline_edge_kwargs(style) == {}

    def test_outline_alpha_multiplies_the_colour_but_layer_alpha_does_not(self):
        from glplot.utils import preview

        style = LayerStyle(
            outline_enabled=True,
            outline_color=(1.0, 0.0, 0.0, 0.8),
            outline_alpha=0.5,
            alpha=0.1,
        )
        assert preview._outline_rgba(style) == pytest.approx((1.0, 0.0, 0.0, 0.4))

    def test_the_stroke_is_the_artists_width_plus_the_outline_on_both_sides(self):
        from glplot.utils import preview

        style = LayerStyle(outline_enabled=True, outline_width=2.0)
        effects = preview._outline_path_effects(style, base_linewidth=3.0)
        assert effects is not None and len(effects) == 1
        assert effects[0]._gc["linewidth"] == pytest.approx(7.0)

    def test_marker_edges_leave_the_outline_width_showing_outside(self):
        from glplot.utils import preview

        style = LayerStyle(outline_enabled=True, outline_width=1.5)
        kwargs = preview._outline_edge_kwargs(style)
        assert kwargs["linewidths"] == pytest.approx(3.0)
        assert kwargs["edgecolors"] == [(0.0, 0.0, 0.0, 1.0)]


class _FakeEngine:
    """The three attributes ``render_preview`` reads off an engine, and nothing else."""

    def __init__(self, layers):
        self.width, self.height = 640, 480
        self.scene = type("Scene", (), {"layers": layers})()
        self.grid_visible = False
        self.title = ""


def _preview_layers():
    line = PolylineLayer(
        pts=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.4]], dtype=np.float32),
        color=(0.1, 0.3, 0.9, 1.0),
        width=2.0,
    )
    patch = PatchLayer(
        vertices=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    )
    patch.style.face_color = (0.9, 0.5, 0.1, 1.0)
    scatter = ScatterLayer(
        pts=np.array([[0.2, 0.8], [1.4, 0.2]], dtype=np.float32),
        colors=np.array([[0.2, 0.7, 0.3, 1.0]] * 2, dtype=np.float32),
        size=90.0,
    )
    text = TextLayer(x=0.5, y=0.5, text="label")
    return [line, patch, scatter, text]


class TestPreviewExport:
    """An exported PNG that silently drops the outline is the bug this section exists for."""

    def _render(self, tmp_path, name, outline: bool):
        from glplot.utils.preview import render_preview

        layers = _preview_layers()
        if outline:
            for layer in layers:
                layer.style.outline_enabled = True
                layer.style.outline_color = (1.0, 0.0, 0.0, 1.0)
                layer.style.outline_width = 3.0
        target = tmp_path / name
        render_preview(_FakeEngine(layers), str(target))
        return target.read_bytes()

    def test_the_outline_reaches_the_png(self, tmp_path):
        off = self._render(tmp_path, "off.png", outline=False)
        on = self._render(tmp_path, "on.png", outline=True)
        assert len(off) > 0 and len(on) > 0
        assert off != on, "the exported PNG is identical with and without an outline"

    def test_two_exports_of_the_same_scene_agree(self, tmp_path):
        """Guards the comparison above: the export is deterministic, so a difference means
        the outline and not the renderer's mood."""
        assert self._render(tmp_path, "a.png", outline=False) == self._render(
            tmp_path, "b.png", outline=False
        )

    def test_every_2d_artist_carries_the_outline(self, tmp_path):
        """Not just "the picture changed": each artist type must actually get the effect."""
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as mpl

        from glplot.utils.preview import render_preview

        layers = _preview_layers()
        for layer in layers:
            layer.style.outline_enabled = True

        created = []
        real_subplots = mpl.subplots

        def spy(*args, **kwargs):
            fig, ax = real_subplots(*args, **kwargs)
            created.append(ax)
            return fig, ax

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(mpl, "subplots", spy)
            render_preview(_FakeEngine(layers), str(tmp_path / "artists.png"))

        assert created, "render_preview did not build a 2D axes"
        ax = created[0]
        stroked = [a for a in ax.lines if a.get_path_effects()]
        assert stroked, "the polyline lost its casing"
        assert [p for p in ax.patches if p.get_path_effects()], "the patch lost its silhouette"
        assert [t for t in ax.texts if t.get_path_effects()], "the text lost its halo"
        widths = [c.get_linewidths() for c in ax.collections]
        assert any(np.any(np.asarray(w) > 0) for w in widths), "the scatter lost its ring"
