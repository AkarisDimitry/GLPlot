"""Test the post-processing chain in glplot.managers.effects and its shaders.

Regression coverage for the four proven bloom defects. GL cannot be exercised
headless, so the shader math is CPU-replayed and the GLSL source is asserted to
still match that replay; the CPU-side policy (tone-map selection, knee, mip
level/offset choice, target formats) is tested directly on EffectManager, which
constructs without a window or a GL context.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.engine import GPULinePlot
from glplot.managers.effects import (
    BLOOM_MAX_LEVELS,
    DEFAULT_GLOW_KNEE,
    TONEMAP_ACES,
    TONEMAP_NONE,
    TONEMAP_REINHARD,
    EffectManager,
)
from glplot.utils import shaders


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


def _manager() -> EffectManager:
    """An EffectManager over a real engine. Creates no window and no GL context."""
    return EffectManager(GPULinePlot())


def _composite(scene, bloom, intensity, bloom_enabled, tonemap):
    """CPU mirror of POST_COMPOSITE_FS. Kept honest by TestCompositeShaderSource."""
    color = np.array(scene, dtype=float)
    if bloom_enabled:
        color = color + np.array(bloom, dtype=float) * intensity
    if tonemap == TONEMAP_REINHARD:
        color = color / (color + 1.0)
    elif tonemap == TONEMAP_ACES:
        a, b, c2, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        color = np.clip((color * (a * color + b)) / (color * (c2 * color + d) + e), 0.0, 1.0)
    return color


def _extract(color, threshold, knee_rel):
    """CPU mirror of BLOOM_EXTRACT_FS."""
    color = np.maximum(np.array(color, dtype=float), 0.0)
    b = float(color.max())
    knee = max(threshold * knee_rel, 1e-4)
    soft = float(np.clip(b - threshold + knee, 0.0, 2.0 * knee))
    soft = soft * soft / (4.0 * knee)
    return color * (max(soft, b - threshold) / max(b, 1e-4))


class TestGlowDoesNotDarken:
    """The headline regression: tone mapping must not be welded to bloom."""

    def test_enabling_glow_does_not_darken_a_white_pixel(self):
        """A white pixel with no bloom contribution must survive glow being enabled."""
        off = _composite((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 0.8, False, TONEMAP_NONE)
        on = _composite((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 0.8, True, TONEMAP_NONE)
        assert np.allclose(off, 1.0)
        assert np.allclose(on, 1.0)  # was 0.5 -- Reinhard fired on the bloom toggle
        assert np.allclose(off, on)

    def test_enabling_glow_is_identity_for_every_colour_without_bloom(self):
        """With zero bloom energy the composite must be untouched by the glow flag."""
        for scene in [(1, 1, 1), (0.5, 0.5, 0.5), (1, 0, 0), (0.13, 0.7, 0.42), (0, 0, 0)]:
            off = _composite(scene, (0, 0, 0), 0.8, False, TONEMAP_NONE)
            on = _composite(scene, (0, 0, 0), 0.8, True, TONEMAP_NONE)
            assert np.allclose(off, on), scene

    def test_glow_only_ever_adds_light(self):
        """With the default tone map, bloom is purely additive."""
        base = _composite((0.5, 0.5, 0.5), (0.0, 0.0, 0.0), 0.8, True, TONEMAP_NONE)
        glowing = _composite((0.5, 0.5, 0.5), (0.2, 0.2, 0.2), 0.8, True, TONEMAP_NONE)
        assert np.all(glowing >= base)
        assert np.allclose(glowing, 0.5 + 0.2 * 0.8)

    def test_default_tonemap_is_none(self):
        """Restrained by default: no tone map unless the scene explicitly asks."""
        assert _manager().tonemap_index() == TONEMAP_NONE

    def test_tonemap_index_reads_visual_option_when_present(self):
        """An opt-in tonemap_index on VisualOptions is honored."""
        mgr = _manager()
        mgr.options.visual.tonemap_index = TONEMAP_ACES
        assert mgr.tonemap_index() == TONEMAP_ACES

    def test_tonemap_index_rejects_garbage(self):
        """Out-of-range or non-numeric values fall back to no tone mapping."""
        mgr = _manager()
        for bad in (99, -1, "reinhard", None):
            mgr.options.visual.tonemap_index = bad
            assert mgr.tonemap_index() == TONEMAP_NONE


class TestCompositeShaderSource:
    """Guard that the GLSL still matches the CPU mirror above."""

    def test_tone_mapping_is_not_gated_on_bloom_enabled(self):
        """The Reinhard call must live under u_tonemap, never under u_bloom_enabled."""
        src = shaders.POST_COMPOSITE_FS
        bloom_branch = src[src.index("if (u_bloom_enabled == 1)") :]
        bloom_branch = bloom_branch[: bloom_branch.index("}")]
        assert "tonemap" not in bloom_branch
        assert "u_tonemap == 1" in src
        assert "u_tonemap == 2" in src

    def test_composite_declares_the_tonemap_uniform(self):
        """u_tonemap must exist for EffectManager.tonemap_index() to drive anything."""
        assert re.search(r"^uniform\s+int\s+u_tonemap\s*;", shaders.POST_COMPOSITE_FS, re.M)

    def test_both_tonemap_operators_are_implemented(self):
        """Selector values 1 and 2 must map to real operators, not stubs."""
        assert "vec3 tonemap_reinhard" in shaders.POST_COMPOSITE_FS
        assert "vec3 tonemap_aces" in shaders.POST_COMPOSITE_FS


class TestBloomExtractKnee:
    """The bright pass must ramp, not pop, and must not eat light backgrounds."""

    def test_default_knee(self):
        """glow_knee() falls back to the module default when the option is absent."""
        assert _manager().glow_knee() == DEFAULT_GLOW_KNEE

    def test_knee_reads_glow_option_when_present(self):
        """An opt-in GlowOptions.knee is honored and clamped to [0, 1]."""
        mgr = _manager()
        mgr.options.visual.glow.knee = 0.25
        assert mgr.glow_knee() == 0.25
        mgr.options.visual.glow.knee = 5.0
        assert mgr.glow_knee() == 1.0
        mgr.options.visual.glow.knee = -3.0
        assert mgr.glow_knee() == 0.0
        mgr.options.visual.glow.knee = "wide"
        assert mgr.glow_knee() == DEFAULT_GLOW_KNEE

    def test_contribution_is_continuous_across_the_threshold(self):
        """No binary cut: adjacent brightnesses must not jump (that is the popping)."""
        xs = np.linspace(0.0, 1.2, 200)
        vals = np.array([float(_extract((x, x, x), 0.7, DEFAULT_GLOW_KNEE).max()) for x in xs])
        assert np.abs(np.diff(vals)).max() < 0.02

    def test_contribution_is_monotonic(self):
        """A brighter pixel must never contribute less bloom than a darker one."""
        xs = np.linspace(0.0, 2.0, 200)
        vals = np.array([float(_extract((x, x, x), 0.7, DEFAULT_GLOW_KNEE).max()) for x in xs])
        assert np.all(np.diff(vals) >= -1e-9)

    def test_dark_pixels_contribute_nothing(self):
        """Well below the threshold the bright pass must be exactly black."""
        assert np.allclose(_extract((0.0, 0.0, 0.0), 0.7, DEFAULT_GLOW_KNEE), 0.0)
        assert np.allclose(_extract((0.1, 0.1, 0.1), 0.7, DEFAULT_GLOW_KNEE), 0.0)

    def test_light_background_does_not_detonate(self):
        """A 0.95 background used to pass through at full brightness."""
        out = _extract((0.95, 0.95, 0.95), 0.7, DEFAULT_GLOW_KNEE)
        assert out.max() < 0.4 * 0.95

    def test_saturated_colour_can_bloom(self):
        """Max-channel brightness: a pure blue line must be able to glow."""
        assert _extract((0.0, 0.0, 1.0), 0.7, DEFAULT_GLOW_KNEE).max() > 0.0

    def test_extract_shader_declares_knee_and_has_no_binary_cut(self):
        """Source guard for the knee curve."""
        src = shaders.BLOOM_EXTRACT_FS
        assert re.search(r"^uniform\s+float\s+u_knee\s*;", src, re.M)
        assert "if (brightness > u_threshold)" not in src


class TestBloomBlur:
    """The blur must be a real widening kernel, not radius-scaled tap offsets."""

    def test_ghost_copy_blur_is_gone(self):
        """GAUSSIAN_BLUR_FS multiplied the tap offset by u_radius; it must not return."""
        assert not hasattr(shaders, "GAUSSIAN_BLUR_FS")
        assert "u_radius" not in shaders.BLOOM_DOWNSAMPLE_FS
        assert "u_radius" not in shaders.BLOOM_UPSAMPLE_FS

    def test_dual_filter_programs_exist(self):
        """Both halves of the dual filter must be present and declare u_halfpixel."""
        for src in (shaders.BLOOM_DOWNSAMPLE_FS, shaders.BLOOM_UPSAMPLE_FS):
            assert re.search(r"^uniform\s+vec2\s+u_halfpixel\s*;", src, re.M)
            assert src.lstrip().startswith("#version 330 core")

    def test_blur_width_is_monotonic_in_radius(self):
        """Raising the radius slider must widen the blur, with no octave steps."""
        mgr = _manager()
        mgr.bloom_chain = [None] * (BLOOM_MAX_LEVELS + 1)
        last = -1.0
        for radius in [1.0, 2.0, 3.9, 4.0, 6.0, 12.0, 20.0, 32.0]:
            levels = mgr._bloom_levels(radius)
            offset = mgr._bloom_offset(radius, levels)
            assert 0.5 <= offset <= 2.0
            width = (2**levels) * offset
            assert width > last
            last = width

    def test_levels_are_capped(self):
        """Level count never exceeds the allocated chain or BLOOM_MAX_LEVELS."""
        mgr = _manager()
        mgr.bloom_chain = [None] * 3
        assert mgr._bloom_levels(1024.0) == 2
        mgr.bloom_chain = [None] * (BLOOM_MAX_LEVELS + 4)
        assert mgr._bloom_levels(1024.0) == BLOOM_MAX_LEVELS

    def test_degenerate_chain_reports_no_levels(self):
        """A window too small to hold a mip chain must disable bloom, not crash."""
        mgr = _manager()
        mgr.bloom_chain = []
        assert mgr._bloom_levels(6.0) == 0
        mgr.bloom_chain = [None]
        assert mgr._bloom_levels(6.0) == 0


class TestBloomChainFormat:
    """The bloom chain must carry HDR, or bright pixels clamp at the bright pass."""

    def test_chain_is_float(self):
        """_rebuild_targets must allocate 16F for both scene and bloom targets."""
        import inspect

        from glplot.managers import effects

        src = inspect.getsource(effects.EffectManager._rebuild_targets)
        formats = re.findall(r"_create_target\([^)]*?(GL_RGBA\w+)", src, re.S)
        assert "GL_RGBA8" not in formats
        assert formats.count("GL_RGBA16F") == 2
