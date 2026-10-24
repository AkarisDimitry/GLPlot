"""Rendering that must not depend on which OS or GPU driver runs the script.

A 3D scatter on a surface rendered correctly on macOS and lost its particles on a Linux
PC. Two independent causes are pinned down here, both without a GL context:

* **Undefined clip distances.** The engine enables GL_CLIP_DISTANCE0-3 for the scene
  pass, and a vertex shader that does not write ``gl_ClipDistance`` then has undefined
  clip values. macOS keeps the primitive; another driver may cull it. ``link_program``
  now makes every shader write them, so the behaviour is macOS's everywhere.
* **The matplotlib fallback.** With no display, the OpenGL export context fails and
  animations fall back to matplotlib, whose surface used a fixed 0.92 collection alpha
  that overrode the caller's ``alpha=``: a translucent PES came out near-opaque, burying
  every particle drawn under it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as mpl_plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import glplot.pyplot as gplt  # noqa: E402
from glplot import animation  # noqa: E402
from glplot.utils import shaders  # noqa: E402
from glplot.utils.gl_utils import with_defined_clip_distances  # noqa: E402


def _vertex_shaders():
    return {
        name: src
        for name, src in vars(shaders).items()
        if isinstance(src, str) and "gl_Position" in src
    }


class TestClipDistancesAreAlwaysDefined:
    def test_every_vertex_shader_writes_gl_clip_distance_once_linked(self):
        missing = [
            name
            for name, src in _vertex_shaders().items()
            if "gl_ClipDistance" not in with_defined_clip_distances(src)
        ]
        assert missing == []

    def test_the_3d_shader_gets_all_four_planes_at_the_top_of_main(self):
        out = with_defined_clip_distances(shaders.GEOMETRY3D_VS)
        body = out[out.index("void main()") :]
        for i in range(4):
            assert f"gl_ClipDistance[{i}] = 1.0;" in body
        # Written before anything can return early.
        assert body.index("gl_ClipDistance[3]") < body.index("gl_Position")

    def test_shaders_that_really_clip_are_left_alone(self):
        assert with_defined_clip_distances(shaders.SCATTER_VS) == shaders.SCATTER_VS
        assert with_defined_clip_distances(shaders.PATCH_VS) == shaders.PATCH_VS

    def test_source_without_main_is_returned_unchanged(self):
        src = "#version 330 core\nuniform float u;\n"
        assert with_defined_clip_distances(src) == src


class TestPreviewHonoursSurfaceAlpha:
    def _surface_face_alphas(self, alpha):
        from glplot.utils.preview import _draw_layers

        gplt.figure()
        x, y = np.meshgrid(np.linspace(-1, 1, 6), np.linspace(-1, 1, 6))
        gplt.plot_surface(x, y, x * x + y * y, cmap="viridis", alpha=alpha)
        layers = gplt.gcf().panels[0].scene.layers
        fig = mpl_plt.figure()
        try:
            ax = fig.add_subplot(111, projection="3d")
            _draw_layers(ax, layers, has_3d=True, engine=gplt.gcf())
            (surf,) = [c for c in ax.collections if c.get_facecolors().shape[0] > 1]
            return surf.get_facecolors()[:, 3]
        finally:
            mpl_plt.close(fig)
            gplt.close()

    def test_a_translucent_surface_keeps_its_alpha(self):
        assert np.allclose(self._surface_face_alphas(0.4), 0.4)

    def test_an_opaque_surface_previews_as_before(self):
        assert np.allclose(self._surface_face_alphas(1.0), 0.92)


class TestHeadlessLinuxHint:
    def test_linux_without_a_display_suggests_xvfb(self, monkeypatch):
        monkeypatch.setattr(animation.sys, "platform", "linux")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert "xvfb-run" in animation._headless_linux_hint()

    @pytest.mark.parametrize("var", ["DISPLAY", "WAYLAND_DISPLAY"])
    def test_linux_with_a_display_says_nothing(self, monkeypatch, var):
        monkeypatch.setattr(animation.sys, "platform", "linux")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setenv(var, ":0")
        assert animation._headless_linux_hint() == ""

    def test_other_platforms_say_nothing(self, monkeypatch):
        monkeypatch.setattr(animation.sys, "platform", "darwin")
        monkeypatch.delenv("DISPLAY", raising=False)
        assert animation._headless_linux_hint() == ""
