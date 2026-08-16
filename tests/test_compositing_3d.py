"""Per-layer 3D compositing: the blend mode, occlusion and automatic-alpha levers.

Figure-wide blending was never enough in 3D -- one scene legitimately wants a volumetric
cloud in ADDITIVE (density reads as brightness, tails survive at alpha 0.02) beside surfaces
in ordinary ALPHA, and one heuristic cannot choose the occlusion for both a see-through
surface and a solid-reading cloud. These three ``LayerStyle`` fields give the choice back:

* ``blend_mode`` overrides ``options.blend_mode`` for one layer's draw and restores it;
* ``depth_write`` overrides the translucent-points-and-lines heuristic either way;
* ``auto_alpha`` solves the per-point alpha for a target covered-pixel opacity under the
  current view, so a cloud stays readable across zoom instead of saturating and vanishing.

Checked without a GL context, by swapping the entry points ``geometry3d`` and its
``blending`` helper star-import for recorders. ``apply_blend_mode`` lives in
``utils.blending`` and is imported *into* ``geometry3d``'s namespace, so it is patched
there. The numeric ``auto_alpha_for`` is pure numpy and is tested directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.core.context import RenderContext
from glplot.core.layers import Layer3D
from glplot.options import BlendMode, EngineOptions, RenderMode
from glplot.renderers import geometry3d as g3d
from glplot.utils import blending as bl

UNIFORM_NAMES = (
    "u_mvp",
    "u_alpha",
    "u_auto_alpha",
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


class Recorder:
    def __init__(self) -> None:
        self.state: list = []
        self.uniforms: dict = {}
        self.blend_modes: list = []  # each apply_blend_mode(mode) call, in order

    def enable(self, cap):
        self.state.append(("enable", int(cap)))

    def disable(self, cap):
        self.state.append(("disable", int(cap)))

    def depth_mask(self, flag):
        self.state.append(("depth_mask", bool(flag)))

    def draw_arrays(self, mode, first, count):
        self.state.append(("draw", int(mode)))

    def draw_elements(self, mode, count, gl_type, indices):
        self.state.append(("draw", int(mode)))

    def uniform1f(self, loc, value):
        self.uniforms[int(loc)] = float(value)

    def apply_blend_mode(self, mode, *, premultiplied_target=False):
        self.state.append(("blend", mode.name))
        self.blend_modes.append(mode.name)


@pytest.fixture
def rec(monkeypatch) -> Recorder:
    r = Recorder()
    monkeypatch.setattr(g3d, "glEnable", r.enable)
    monkeypatch.setattr(g3d, "glDisable", r.disable)
    monkeypatch.setattr(g3d, "glDepthMask", r.depth_mask)
    monkeypatch.setattr(g3d, "glUseProgram", lambda prog: None)
    monkeypatch.setattr(g3d, "glBindVertexArray", lambda vao: None)
    monkeypatch.setattr(g3d, "glUniformMatrix4fv", lambda *a: None)
    monkeypatch.setattr(g3d, "glUniform1f", r.uniform1f)
    for n in (1, 2, 3, 4):
        for suffix in ("f", "i"):
            name = f"glUniform{n}{suffix}"
            if hasattr(g3d, name) and name != "glUniform1f":
                monkeypatch.setattr(g3d, name, lambda *a: None)
    monkeypatch.setattr(g3d, "glDrawArrays", r.draw_arrays)
    monkeypatch.setattr(g3d, "glDrawElements", r.draw_elements)
    monkeypatch.setattr(g3d, "glGenBuffers", lambda n: 42)
    monkeypatch.setattr(g3d, "glBindBuffer", lambda target, buf: None)
    monkeypatch.setattr(g3d, "glBufferData", lambda *a: None)
    # apply_blend_mode is imported into geometry3d's namespace.
    monkeypatch.setattr(g3d, "apply_blend_mode", r.apply_blend_mode)
    return r


def make_renderer(blend_mode=BlendMode.ALPHA) -> g3d.Geometry3DRenderer:
    opts = EngineOptions()
    opts.blend_mode = blend_mode
    renderer = g3d.Geometry3DRenderer(opts)
    renderer.prog = 1
    for i, name in enumerate(UNIFORM_NAMES):
        setattr(renderer, name, i)
    return renderer


def make_layer(primitive: str = "points", *, alpha: float | None = None, n: int = 4) -> Layer3D:
    verts = np.random.default_rng(0).normal(0, 1, (n, 3)).astype(np.float32)
    colors = np.tile(np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1))
    if alpha is not None:
        colors[:, 3] = alpha
    layer = Layer3D(vertices=verts, colors=colors, primitive=primitive)
    layer.style.point_size = 3.0
    layer._gl = g3d.GLGeometry3DBuffers(vao=7, vbo_pos=1, vbo_col=2, count=n)
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


# ----------------------------------------------------------------------------------
# Per-layer blend mode
# ----------------------------------------------------------------------------------


class TestPerLayerBlendMode:
    def test_a_layer_without_a_mode_never_touches_blend_state(self, rec):
        make_renderer().draw(make_layer(alpha=0.3), make_ctx())
        assert not rec.blend_modes

    def test_a_layers_mode_is_applied_then_the_figures_is_restored(self, rec):
        layer = make_layer(alpha=0.3)
        layer.style.blend_mode = BlendMode.ADDITIVE
        make_renderer(blend_mode=BlendMode.ALPHA).draw(layer, make_ctx())
        # Its own mode before the draw, the figure's back after it.
        assert rec.blend_modes == ["ADDITIVE", "ALPHA"]
        draws = [i for i, e in enumerate(rec.state) if e[0] == "draw"]
        first_blend = rec.state.index(("blend", "ADDITIVE"))
        last_blend = len(rec.state) - 1 - rec.state[::-1].index(("blend", "ALPHA"))
        assert first_blend < draws[0] < last_blend

    def test_the_figure_mode_restored_is_the_options_value(self, rec):
        layer = make_layer(alpha=0.3)
        layer.style.blend_mode = BlendMode.SCREEN
        make_renderer(blend_mode=BlendMode.SUBTRACTIVE).draw(layer, make_ctx())
        assert rec.blend_modes == ["SCREEN", "SUBTRACTIVE"]


class TestOrderIndependentSkipsTheSort:
    def test_an_additive_cloud_is_not_depth_sorted(self, rec):
        """Addition gives the same image in any order, so sorting it is wasted work -- and
        the draw must not go through an order index buffer."""
        layer = make_layer("points", alpha=0.3, n=64)
        layer.style.blend_mode = BlendMode.ADDITIVE
        make_renderer().draw(layer, make_ctx())
        assert getattr(layer, "_depth_order", None) is None

    def test_an_alpha_cloud_is_still_sorted(self, rec):
        layer = make_layer("points", alpha=0.3, n=64)
        layer.style.blend_mode = BlendMode.ALPHA
        make_renderer().draw(layer, make_ctx())
        assert getattr(layer, "_depth_order", None) is not None

    def test_an_additive_layer_reads_as_translucent(self):
        """It never replaces the destination, so it is see-through even at alpha 1.0 and
        must not write depth and punch a hole in what is behind it."""
        layer = make_layer("points", alpha=1.0)
        assert g3d.is_translucent(layer) is False
        layer.style.blend_mode = BlendMode.ADDITIVE
        assert g3d.is_translucent(layer) is True


# ----------------------------------------------------------------------------------
# Forced occlusion
# ----------------------------------------------------------------------------------


class TestDepthWriteOverride:
    def test_forcing_writes_off_on_an_opaque_mesh(self, rec):
        """The heuristic keeps a mesh's writes; forcing off makes it see-through."""
        layer = make_layer("triangles")
        layer.indices = np.array([0, 1, 2], dtype=np.uint32)
        layer.style.depth_write = False
        make_renderer().draw(layer, make_ctx())
        assert ("depth_mask", False) in rec.state

    def test_forcing_writes_on_for_a_translucent_cloud(self, rec):
        """A dense cloud the caller wants to read as solid -- no mask, so it occludes."""
        layer = make_layer("points", alpha=0.3)
        layer.style.depth_write = True
        make_renderer().draw(layer, make_ctx())
        assert not any(e[0] == "depth_mask" for e in rec.state)

    def test_the_axis_overlay_ignores_the_override(self, rec):
        """It draws with the depth test off and must keep doing so, whatever a stray
        depth_write on it says."""
        layer = make_layer("lines", alpha=0.3)
        layer.metadata["artist"] = "axis3d"
        layer.style.depth_write = True
        make_renderer().draw(layer, make_ctx())
        assert not any(e[0] == "depth_mask" for e in rec.state)

    def test_none_leaves_the_heuristic_in_charge(self, rec):
        """A translucent cloud with no override still drops its writes."""
        layer = make_layer("points", alpha=0.3)
        assert layer.style.depth_write is None
        make_renderer().draw(layer, make_ctx())
        assert ("depth_mask", False) in rec.state


# ----------------------------------------------------------------------------------
# Automatic alpha
# ----------------------------------------------------------------------------------


class TestAutoAlphaUniform:
    def test_off_pushes_the_negative_sentinel(self, rec):
        r = make_renderer()
        r.draw(make_layer(alpha=0.3), make_ctx())
        assert rec.uniforms[r.u_auto_alpha] < 0.0

    def test_on_pushes_a_value_in_range(self, rec):
        layer = make_layer("points", alpha=0.3, n=5000)
        layer.style.auto_alpha = 0.85
        r = make_renderer()
        r.draw(layer, make_ctx())
        pushed = rec.uniforms[r.u_auto_alpha]
        assert 0.0 < pushed <= 1.0

    def test_a_layer_asking_for_auto_alpha_reads_as_translucent(self):
        layer = make_layer("points", alpha=1.0)
        assert g3d.is_translucent(layer) is False
        layer.style.auto_alpha = 0.9
        assert g3d.is_translucent(layer) is True


class TestAutoAlphaMath:
    """``auto_alpha_for`` is deterministic numpy; test the numbers directly."""

    def _cloud(self, n=200_000):
        v = np.random.default_rng(0).normal(0, 1, (n, 3)).astype(np.float32)
        layer = Layer3D(vertices=v, primitive="points")
        layer.style.point_size = 3.0
        layer.style.auto_alpha = 0.85
        return layer

    def _scaled_mvp(self, s):
        m = np.eye(4, dtype=np.float32)
        m[0, 0] = m[1, 1] = m[2, 2] = s
        return m

    def test_off_returns_the_negative_sentinel(self):
        layer = self._cloud()
        layer.style.auto_alpha = None
        assert g3d.auto_alpha_for(layer, np.eye(4), make_ctx()) == -1.0

    def test_zooming_out_lowers_the_per_point_alpha(self):
        """More points stack on a pixel when the cloud is small on screen, so each one has
        to be fainter to reach the same covered-pixel opacity."""
        layer = self._cloud()
        wide = g3d.auto_alpha_for(layer, self._scaled_mvp(0.3), make_ctx())
        tight = g3d.auto_alpha_for(layer, self._scaled_mvp(2.0), make_ctx())
        assert 0.0 < wide < tight <= 1.0

    def test_it_never_exceeds_the_target(self):
        layer = self._cloud()
        for s in (0.2, 0.5, 1.0, 5.0):
            a = g3d.auto_alpha_for(layer, self._scaled_mvp(s), make_ctx())
            assert a <= 0.85 + 1e-6

    def test_an_empty_cloud_is_the_sentinel(self):
        layer = Layer3D(vertices=np.zeros((0, 3), dtype=np.float32), primitive="points")
        layer.style.auto_alpha = 0.9
        assert g3d.auto_alpha_for(layer, np.eye(4), make_ctx()) == -1.0


# ----------------------------------------------------------------------------------
# The blend helper itself
# ----------------------------------------------------------------------------------


class TestBlendingHelper:
    @pytest.mark.parametrize(
        "mode,oi",
        [
            (BlendMode.ADDITIVE, True),
            (BlendMode.SUBTRACTIVE, True),
            (BlendMode.SCREEN, True),
            (BlendMode.ALPHA, False),
            (BlendMode.AUTO, False),
            (BlendMode.OFF, False),
        ],
    )
    def test_order_independence(self, mode, oi):
        assert bl.is_order_independent(mode) is oi

    def test_none_is_neither(self):
        assert bl.is_order_independent(None) is False
        assert bl.hides_nothing(None) is False

    def test_accumulating_modes_hide_nothing(self):
        assert bl.hides_nothing(BlendMode.ADDITIVE) is True
        assert bl.hides_nothing(BlendMode.ALPHA) is False
