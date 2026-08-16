"""Test per-layer colormaps and the layer-metadata helpers in glplot.gui.layerops.

Focus on the ``set_layer_style(cmap=...)`` trap, the three CPU colormap tiers
(imshow / 2D scatter / scatter3d), the shader-colormap override and per-layer SSAO,
without requiring OpenGL or GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.engine import GPULinePlot
from glplot.gui import layerops


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


def _imshow_layer():
    """An imshow layer and the plot holding it."""
    layer = gplt.imshow(np.linspace(0.0, 1.0, 64).reshape(8, 8), cmap="viridis")
    return gplt._get_or_create_plot(), layer


class TestSetLayerStyleTrap:
    """The reason set_layer_metadata exists at all."""

    def test_set_layer_style_cmap_writes_style_which_no_renderer_reads(self):
        """REGRESSION: set_layer_style(cmap=) is accepted and changes nothing on screen.

        LayerStyle declares cmap/vmin/vmax, so the call cannot raise; the renderer reads
        layer.metadata (scatter.py:185-189) and never layer.style, so the picture never
        moves. This documents the trap rather than fixing it: the fix is to route through
        set_layer_colormap, and this test fails the day style.cmap becomes live.
        """
        plot, layer = _imshow_layer()
        layerops.set_layer_style(plot, layer, cmap="hot")
        assert layer.style.cmap == "hot"
        assert layer.metadata["cmap"] == "viridis"

    def test_set_layer_metadata_writes_what_the_renderer_reads(self):
        """set_layer_metadata reaches metadata and flags the GPU re-upload."""
        plot, layer = _imshow_layer()
        layer.dirty.gpu_dirty = False
        changed = layerops.set_layer_metadata(plot, layer, cmap="hot")
        assert changed == ["cmap"]
        assert layer.metadata["cmap"] == "hot"
        assert layer.dirty.gpu_dirty is True

    def test_set_layer_metadata_applies_the_dirty_incantation(self):
        """A metadata write must mark the scene dirty and kill the stale impostor."""
        plot, layer = _imshow_layer()
        plot.frame.dirty_scene = False
        plot.cache.refresh_requested = False
        plot.cache.capture_window = (0.0, 1.0, 0.0, 1.0)
        layerops.set_layer_metadata(plot, layer, cmap="magma")
        assert plot.frame.dirty_scene is True
        assert plot.cache.refresh_requested is True
        assert plot.cache.capture_window is None

    def test_unchanged_value_reports_no_change(self):
        """Writing the value that is already there is a no-op, not a redraw."""
        plot, layer = _imshow_layer()
        assert layerops.set_layer_metadata(plot, layer, cmap="viridis") == []

    def test_gpu_dirty_can_be_declined(self):
        """Bookkeeping keys must not force a re-upload."""
        plot, layer = _imshow_layer()
        layer.dirty.gpu_dirty = False
        layerops.set_layer_metadata(plot, layer, gpu_dirty=False, note="hi")
        assert layer.dirty.gpu_dirty is False


class TestColormapKind:
    """Which colormap mechanism each layer type supports."""

    def test_imshow_is_an_image(self):
        """An imshow layer rebuilds its texture from metadata."""
        _, layer = _imshow_layer()
        assert layerops.layer_colormap_kind(layer) == "image"

    def test_scatter_with_scalars_is_values2d(self):
        """A c=-coloured scatter retains its scalars and can be re-mapped."""
        x = np.linspace(0.0, 1.0, 20)
        layer = gplt.scatter(x, x, c=x, cmap="viridis")
        assert layerops.layer_colormap_kind(layer) == "values2d"

    def test_scatter_with_literal_colour_has_no_colormap(self):
        """Literal RGBA has no source scalars, so no picker may be offered."""
        x = np.linspace(0.0, 1.0, 20)
        layer = gplt.scatter(x, x, color="red")
        assert layerops.layer_colormap_kind(layer) is None

    def test_scatter3d_is_values3d(self):
        """scatter3d retains zdata (pyplot.py:1121)."""
        z = np.linspace(0.0, 5.0, 20)
        layer = gplt.scatter3d(z, z, z, cmap="viridis")
        assert layerops.layer_colormap_kind(layer) == "values3d"

    def test_polyline_is_a_shader_colormap(self):
        """polyline colours by vertex id in the shader, not by data values."""
        x = np.linspace(0.0, 1.0, 20)
        layer = gplt.plot(x, x)[0]
        assert layerops.layer_colormap_kind(layer) == "gl_line"


class TestRetainedCvalues:
    """pyplot.scatter used to bake colours into a VBO and discard the scalars."""

    def test_scatter_retains_cvalues(self):
        """REGRESSION: without metadata['cvalues'] a 2D scatter cannot be re-mapped."""
        x = np.linspace(0.0, 1.0, 32)
        layer = gplt.scatter(x, x**2, c=x, cmap="viridis")
        assert layer.metadata["cvalues"] is not None
        assert np.allclose(layer.metadata["cvalues"], x)

    def test_cvalues_are_float32(self):
        """Retention is 4 bytes/point, not 8: the array is cast, not merely referenced."""
        x = np.linspace(0.0, 1.0, 32, dtype=np.float64)
        layer = gplt.scatter(x, x, c=x)
        assert layer.metadata["cvalues"].dtype == np.float32

    def test_cvalues_are_a_copy_not_a_view(self):
        """A retained view would let a later edit rewrite the source behind the colours."""
        values = np.linspace(0.0, 1.0, 32)
        layer = gplt.scatter(values, values, c=values)
        values[0] = 999.0
        assert layer.metadata["cvalues"][0] != 999.0

    def test_literal_colour_retains_nothing(self):
        """No scalars, nothing to retain."""
        x = np.linspace(0.0, 1.0, 8)
        layer = gplt.scatter(x, x, color=(1.0, 0.0, 0.0, 1.0))
        assert layer.metadata["cvalues"] is None

    def test_retention_cap_is_enforced(self):
        """Past the cap the scalars are dropped rather than doubling a huge scatter."""
        oversized = np.zeros(gplt.CVALUES_RETAIN_MAX_POINTS + 1, dtype=np.float32)
        assert gplt._retained_cvalues(oversized) is None

    def test_under_the_cap_retention_happens(self):
        """The cap must not bite at ordinary sizes."""
        values = np.zeros(1000, dtype=np.float32)
        assert gplt._retained_cvalues(values) is not None


class TestSetLayerColormap:
    """The CPU colormap paths."""

    def test_imshow_cmap_reaches_metadata(self):
        """imshow needs only the metadata write; the texture rebuilds on gpu_dirty."""
        plot, layer = _imshow_layer()
        layer.dirty.gpu_dirty = False
        assert layerops.set_layer_colormap(plot, layer, cmap="hot") is True
        assert layer.metadata["cmap"] == "hot"
        assert layer.dirty.gpu_dirty is True

    def test_scatter2d_colors_are_recomputed(self):
        """A 2D scatter's colours live in a VBO, so they are re-mapped on the CPU."""
        x = np.linspace(0.0, 1.0, 32)
        layer = gplt.scatter(x, x, c=x, cmap="viridis")
        plot = gplt._get_or_create_plot()
        before = layer.colors.copy()
        layerops.set_layer_colormap(plot, layer, cmap="plasma")
        assert not np.allclose(before, layer.colors)
        assert len(layer.colors) == len(layer.pts)
        assert layer.dirty.gpu_dirty is True

    def test_scatter2d_recolour_preserves_alpha(self):
        """A recolour must not quietly make a translucent scatter opaque."""
        x = np.linspace(0.0, 1.0, 16)
        layer = gplt.scatter(x, x, c=x, cmap="viridis", alpha=0.25)
        plot = gplt._get_or_create_plot()
        before = layer.colors[:, 3].copy()
        layerops.set_layer_colormap(plot, layer, cmap="plasma")
        assert np.allclose(layer.colors[:, 3], before)

    def test_scatter3d_colors_are_recomputed_from_zdata(self):
        """scatter3d re-maps from the retained z, and needs gpu_dirty to re-upload."""
        z = np.linspace(0.0, 5.0, 24)
        layer = gplt.scatter3d(z, z, z, cmap="viridis")
        plot = gplt._get_or_create_plot()
        before = layer.colors.copy()
        layer.dirty.gpu_dirty = False
        layerops.set_layer_colormap(plot, layer, cmap="inferno")
        assert not np.allclose(before, layer.colors)
        assert layer.dirty.gpu_dirty is True

    def test_vmin_changes_the_mapping(self):
        """Pinning the low end re-maps the colours."""
        x = np.linspace(0.0, 1.0, 32)
        layer = gplt.scatter(x, x, c=x, cmap="viridis")
        plot = gplt._get_or_create_plot()
        before = layer.colors.copy()
        layerops.set_layer_colormap(plot, layer, vmin=0.5)
        assert not np.allclose(before, layer.colors)

    def test_explicit_none_restores_autoscaling(self):
        """vmin=None is a real request, distinct from omitting vmin."""
        x = np.linspace(0.0, 1.0, 16)
        layer = gplt.scatter(x, x, c=x, vmin=0.2)
        plot = gplt._get_or_create_plot()
        layerops.set_layer_colormap(plot, layer, vmin=None)
        assert layer.metadata["vmin"] is None

    def test_omitting_an_argument_leaves_it_alone(self):
        """The sentinel must not be confused with None."""
        x = np.linspace(0.0, 1.0, 16)
        layer = gplt.scatter(x, x, c=x, vmin=0.2)
        plot = gplt._get_or_create_plot()
        layerops.set_layer_colormap(plot, layer, cmap="hot")
        assert layer.metadata["vmin"] == 0.2

    def test_refuses_a_shader_colormap_layer(self):
        """Silently ignoring the call is how set_layer_style became a trap."""
        x = np.linspace(0.0, 1.0, 16)
        layer = gplt.plot(x, x)[0]
        plot = gplt._get_or_create_plot()
        with pytest.raises(ValueError):
            layerops.set_layer_colormap(plot, layer, cmap="hot")

    def test_refuses_a_layer_with_no_colormap(self):
        """A patch has no colormap; asking for one is an error, not a no-op."""
        plot = GPULinePlot()
        plot.add_patch(
            np.ascontiguousarray(np.random.rand(6, 2), dtype=np.float32),
            indices=np.arange(6, dtype=np.uint32),
            mode="triangles",
            label="p",
        )
        with pytest.raises(ValueError):
            layerops.set_layer_colormap(plot, plot.scene.layers[0], cmap="hot")

    def test_data_range_seeds_the_range_fields(self):
        """Turning autoscale off must start from the data, not from 0..1."""
        values = np.linspace(3.0, 9.0, 16)
        layer = gplt.scatter(values, values, c=values)
        assert layerops.layer_colormap_data_range(layer) == pytest.approx((3.0, 9.0))


class TestShaderColormapOverride:
    """The polyline / line_family per-layer uniform override."""

    def test_default_follows_the_scene(self):
        """No override is the default, so the scene-global toggle keeps working."""
        x = np.linspace(0.0, 1.0, 16)
        layer = gplt.plot(x, x)[0]
        assert layerops.layer_gl_colormap(layer) == (None, None)
        assert "use_colormap" not in layer.metadata

    def test_override_is_recorded_in_metadata(self):
        """The renderers read metadata, not style (style.use_colormap is dead)."""
        x = np.linspace(0.0, 1.0, 16)
        layer = gplt.plot(x, x)[0]
        plot = gplt._get_or_create_plot()
        assert layerops.set_layer_gl_colormap(plot, layer, enabled=True, scheme_index=4)
        assert layer.metadata["use_colormap"] is True
        assert layer.metadata["cmap_index"] == 4

    def test_none_restores_following_the_scene(self):
        """The third state is what makes the global toggle authoritative again."""
        x = np.linspace(0.0, 1.0, 16)
        layer = gplt.plot(x, x)[0]
        plot = gplt._get_or_create_plot()
        layerops.set_layer_gl_colormap(plot, layer, enabled=True)
        layerops.set_layer_gl_colormap(plot, layer, enabled=None)
        assert layerops.layer_gl_colormap(layer) == (None, None)

    def test_scheme_index_is_clamped(self):
        """The GLSL colormap() falls through to Classic, so an out-of-range index lies."""
        x = np.linspace(0.0, 1.0, 16)
        layer = gplt.plot(x, x)[0]
        plot = gplt._get_or_create_plot()
        layerops.set_layer_gl_colormap(plot, layer, scheme_index=999)
        assert layer.metadata["cmap_index"] == len(layerops.GL_COLORMAP_NAMES) - 1

    def test_gl_names_come_from_density_schemes(self):
        """The GUI must never hardcode the scheme list (CONTRACT §4.5)."""
        from glplot.utils.shaders import DENSITY_SCHEMES

        assert list(layerops.GL_COLORMAP_NAMES) == list(DENSITY_SCHEMES)

    def test_refuses_a_data_colormap_layer(self):
        """Two namespaces; mixing them would silently render the wrong map."""
        x = np.linspace(0.0, 1.0, 16)
        layer = gplt.scatter(x, x, c=x)
        plot = gplt._get_or_create_plot()
        with pytest.raises(ValueError):
            layerops.set_layer_gl_colormap(plot, layer, enabled=True)


class TestRendererReadsTheOverride:
    """The uniform plumbing itself, replayed without a GL context."""

    @staticmethod
    def _resolve(metadata, scene_enabled, scene_scheme):
        """The exact expression polyline.py / line_family.py now evaluate."""
        use_cmap = metadata.get("use_colormap")
        scheme = metadata.get("cmap_index")
        if use_cmap is None:
            use_cmap = scene_enabled
        return (1 if use_cmap else 0, scene_scheme if scheme is None else int(scheme))

    def test_no_override_preserves_default_behaviour(self):
        """The published default must not shift: absent keys mean scene-global."""
        assert self._resolve({}, False, 9) == (0, 9)
        assert self._resolve({}, True, 3) == (1, 3)

    def test_override_wins_over_the_scene(self):
        """A layer that opted in is coloured even with the scene toggle off."""
        assert self._resolve({"use_colormap": True, "cmap_index": 4}, False, 9) == (1, 4)

    def test_explicit_off_beats_a_scene_that_is_on(self):
        """False is an override, not an absence."""
        assert self._resolve({"use_colormap": False}, True, 9) == (0, 9)

    def test_renderers_actually_contain_the_lookup(self):
        """Guards against the replay above drifting from the shipped source."""
        import inspect

        from glplot.renderers import line_family, polyline

        for module in (polyline, line_family):
            source = inspect.getsource(module)
            assert 'metadata.get("use_colormap")' in source
            assert 'metadata.get("cmap_index")' in source


class TestPerLayerSsao:
    """metadata['ssao'] / ['ssao_strength'] are live at geometry3d.py:197-198."""

    def _layer(self):
        z = np.linspace(0.0, 5.0, 16)
        layer = gplt.scatter3d(z, z, z)
        return gplt._get_or_create_plot(), layer

    def test_enable_reaches_metadata(self):
        """geometry3d ORs metadata['ssao'] with the scene switch."""
        plot, layer = self._layer()
        assert layerops.set_layer_ssao(plot, layer, enabled=True) is True
        assert layer.metadata["ssao"] is True

    def test_strength_is_clamped(self):
        """The uniform is a 0..1 strength; anything else is a shader artefact."""
        plot, layer = self._layer()
        layerops.set_layer_ssao(plot, layer, strength=5.0)
        assert layer.metadata["ssao_strength"] == 1.0

    def test_strength_none_restores_the_scene_default(self):
        """Clearing the override must be expressible."""
        plot, layer = self._layer()
        layerops.set_layer_ssao(plot, layer, strength=0.9)
        layerops.set_layer_ssao(plot, layer, strength=None)
        assert layer.metadata["ssao_strength"] is None

    def test_no_gpu_dirty_needed(self):
        """SSAO is read as a uniform at draw time, so it needs no re-upload."""
        plot, layer = self._layer()
        layer.dirty.gpu_dirty = False
        layerops.set_layer_ssao(plot, layer, enabled=True)
        assert layer.dirty.gpu_dirty is False

    def test_refuses_a_2d_layer(self):
        """SSAO is 3D-only; a 2D checkbox would be a dead widget."""
        x = np.linspace(0.0, 1.0, 16)
        layer = gplt.plot(x, x)[0]
        plot = gplt._get_or_create_plot()
        with pytest.raises(ValueError):
            layerops.set_layer_ssao(plot, layer, enabled=True)
