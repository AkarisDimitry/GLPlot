from __future__ import annotations

from typing import Tuple

# --- SHARED UTILS ---

HEATMAP_FUNCS = r"""
vec3 heatmap_classic(float x) {
    x = clamp(x, 0.0, 1.0);
    return vec3(
        smoothstep(0.0, 0.3, x),
        smoothstep(0.3, 0.6, x),
        smoothstep(0.6, 1.0, x)
    );
}

vec3 heatmap_viridis_like(float x) {
    x = clamp(x, 0.0, 1.0);
    vec3 c0 = vec3(0.267, 0.005, 0.329);
    vec3 c1 = vec3(0.283, 0.141, 0.458);
    vec3 c2 = vec3(0.254, 0.265, 0.530);
    vec3 c3 = vec3(0.207, 0.372, 0.553);
    vec3 c4 = vec3(0.164, 0.471, 0.558);
    vec3 c5 = vec3(0.128, 0.567, 0.551);
    vec3 c6 = vec3(0.135, 0.659, 0.518);
    vec3 c7 = vec3(0.267, 0.749, 0.441);
    vec3 c8 = vec3(0.478, 0.821, 0.318);
    vec3 c9 = vec3(0.741, 0.873, 0.150);

    if (x < 0.11) return mix(c0, c1, x / 0.11);
    if (x < 0.22) return mix(c1, c2, (x - 0.11) / 0.11);
    if (x < 0.33) return mix(c2, c3, (x - 0.22) / 0.11);
    if (x < 0.44) return mix(c3, c4, (x - 0.33) / 0.11);
    if (x < 0.55) return mix(c4, c5, (x - 0.44) / 0.11);
    if (x < 0.66) return mix(c5, c6, (x - 0.55) / 0.11);
    if (x < 0.77) return mix(c6, c7, (x - 0.66) / 0.11);
    if (x < 0.88) return mix(c7, c8, (x - 0.77) / 0.11);
    return mix(c8, c9, (x - 0.88) / 0.12);
}

vec3 heatmap_plasma_like(float x) {
    x = clamp(x, 0.0, 1.0);
    vec3 c0 = vec3(0.050, 0.030, 0.528);
    vec3 c1 = vec3(0.291, 0.071, 0.718);
    vec3 c2 = vec3(0.507, 0.104, 0.749);
    vec3 c3 = vec3(0.692, 0.165, 0.564);
    vec3 c4 = vec3(0.845, 0.277, 0.388);
    vec3 c5 = vec3(0.954, 0.468, 0.199);
    vec3 c6 = vec3(0.940, 0.975, 0.131);

    if (x < 0.16) return mix(c0, c1, x / 0.16);
    if (x < 0.32) return mix(c1, c2, (x - 0.16) / 0.16);
    if (x < 0.48) return mix(c2, c3, (x - 0.32) / 0.16);
    if (x < 0.64) return mix(c3, c4, (x - 0.48) / 0.16);
    if (x < 0.80) return mix(c4, c5, (x - 0.64) / 0.16);
    return mix(c5, c6, (x - 0.80) / 0.20);
}

vec3 heatmap_inferno(float x) {
    x = clamp(x, 0.0, 1.0);
    vec3 c0 = vec3(0.000, 0.000, 0.016);
    vec3 c1 = vec3(0.073, 0.038, 0.201);
    vec3 c2 = vec3(0.243, 0.053, 0.404);
    vec3 c3 = vec3(0.449, 0.111, 0.437);
    vec3 c4 = vec3(0.665, 0.177, 0.366);
    vec3 c5 = vec3(0.866, 0.301, 0.228);
    vec3 c6 = vec3(0.976, 0.505, 0.096);
    vec3 c7 = vec3(0.985, 0.768, 0.263);
    vec3 c8 = vec3(0.988, 0.941, 0.729);

    if (x < 0.125) return mix(c0, c1, x / 0.125);
    if (x < 0.250) return mix(c1, c2, (x - 0.125) / 0.125);
    if (x < 0.375) return mix(c2, c3, (x - 0.250) / 0.125);
    if (x < 0.500) return mix(c3, c4, (x - 0.375) / 0.125);
    if (x < 0.625) return mix(c4, c5, (x - 0.500) / 0.125);
    if (x < 0.750) return mix(c5, c6, (x - 0.625) / 0.125);
    if (x < 0.875) return mix(c6, c7, (x - 0.750) / 0.125);
    return mix(c7, c8, (x - 0.875) / 0.125);
}

vec3 heatmap_turbo(float x) {
    x = clamp(x, 0.0, 1.0);
    const vec4 kL = vec4(0.23, 0.11, 0.32, 1.0);
    const vec4 kH = vec4(0.92, 0.95, 0.64, 1.0);
    const vec3 g0 = vec3(0.12, 0.01, 0.22);
    const vec3 g1 = vec3(0.13, 0.15, 0.48);
    const vec3 g2 = vec3(0.15, 0.65, 0.51);
    const vec3 g3 = vec3(0.85, 0.60, 0.12);
    const vec3 g4 = vec3(0.92, 0.11, 0.43);

    if (x < 0.25) return mix(g0, g1, x / 0.25);
    if (x < 0.50) return mix(g1, g2, (x - 0.25) / 0.25);
    if (x < 0.75) return mix(g2, g3, (x - 0.50) / 0.25);
    return mix(g3, g4, (x - 0.75) / 0.25);
}

vec3 heatmap_ink_fire(float x) {
    x = clamp(x, 0.0, 1.0);
    vec3 c0 = vec3(1.0, 1.0, 1.0); // White
    vec3 c1 = vec3(1.0, 0.9, 0.2); // Yellow
    vec3 c2 = vec3(1.0, 0.2, 0.1); // Red
    vec3 c3 = vec3(0.4, 0.0, 0.0); // Dark Red
    vec3 c4 = vec3(0.0, 0.0, 0.0); // Black

    if (x < 0.25) return mix(c0, c1, x / 0.25);
    if (x < 0.50) return mix(c1, c2, (x - 0.25) / 0.25);
    if (x < 0.75) return mix(c2, c3, (x - 0.50) / 0.25);
    return mix(c3, c4, (x - 0.75) / 0.25);
}

vec3 heatmap_magma(float x) {
    x = clamp(x, 0.0, 1.0);
    vec3 c0 = vec3(0.001, 0.000, 0.031);
    vec3 c1 = vec3(0.170, 0.047, 0.360);
    vec3 c2 = vec3(0.447, 0.051, 0.439);
    vec3 c3 = vec3(0.729, 0.160, 0.345);
    vec3 c4 = vec3(0.960, 0.419, 0.231);
    vec3 c5 = vec3(0.988, 0.768, 0.470);
    vec3 c6 = vec3(0.988, 0.988, 0.823);

    if (x < 0.16) return mix(c0, c1, x / 0.16);
    if (x < 0.32) return mix(c1, c2, (x - 0.16) / 0.16);
    if (x < 0.48) return mix(c2, c3, (x - 0.32) / 0.16);
    if (x < 0.64) return mix(c3, c4, (x - 0.48) / 0.16);
    if (x < 0.80) return mix(c4, c5, (x - 0.64) / 0.16);
    return mix(c5, c6, (x - 0.80) / 0.20);
}

vec3 heatmap_grayscale(float x) {
    return vec3(clamp(x, 0.0, 1.0));
}

vec3 heatmap_ocean(float x) {
    x = clamp(x, 0.0, 1.0);
    vec3 c0 = vec3(0.0, 0.0, 0.1);
    vec3 c1 = vec3(0.0, 0.2, 0.6);
    vec3 c2 = vec3(0.0, 0.8, 1.0);
    vec3 c3 = vec3(1.0, 1.0, 1.0);
    if (x < 0.33) return mix(c0, c1, x / 0.33);
    if (x < 0.66) return mix(c1, c2, (x - 0.33) / 0.33);
    return mix(c2, c3, (x - 0.66) / 0.34);
}

vec3 heatmap_hot(float x) {
    x = clamp(x, 0.0, 1.0);
    vec3 c0 = vec3(0.0, 0.0, 0.0);
    vec3 c1 = vec3(0.8, 0.0, 0.0);
    vec3 c2 = vec3(1.0, 0.5, 0.0);
    vec3 c3 = vec3(1.0, 1.0, 0.0);
    vec3 c4 = vec3(1.0, 1.0, 1.0);
    if (x < 0.25) return mix(c0, c1, x / 0.25);
    if (x < 0.50) return mix(c1, c2, (x - 0.25) / 0.25);
    if (x < 0.75) return mix(c2, c3, (x - 0.50) / 0.25);
    return mix(c3, c4, (x - 0.75) / 0.25);
}

vec3 heatmap_cool(float x) {
    x = clamp(x, 0.0, 1.0);
    return mix(vec3(0.0, 1.0, 1.0), vec3(1.0, 0.0, 1.0), x);
}

vec3 apply_heatmap(int scheme, float x) {
    if (scheme == 1) return heatmap_viridis_like(x);
    if (scheme == 2) return heatmap_plasma_like(x);
    if (scheme == 3) return heatmap_inferno(x);
    if (scheme == 4) return heatmap_turbo(x);
    if (scheme == 5) return heatmap_ink_fire(x);
    if (scheme == 6) return heatmap_magma(x);
    if (scheme == 7) return heatmap_grayscale(x);
    if (scheme == 8) return heatmap_ocean(x);
    if (scheme == 9) return heatmap_hot(x);
    if (scheme == 10) return heatmap_cool(x);
    return heatmap_classic(x);
}
"""

DENSITY_SCHEMES = [
    "Classic (B-W-C)",
    "Viridis",
    "Plasma",
    "Inferno",
    "Turbo (Rainbow)",
    "Ink Fire (White BG)",
    "Magma",
    "Grayscale",
    "Ocean",
    "Hot",
    "Cool",
]

# ======================================================================================
# Fractal: a per-pixel escape-time field computed live on the GPU.
#
# The whole point of doing this on the GPU rather than baking an image (as
# ``examples/ex_mandelbrot.py`` does on the CPU) is that it is **resolution-independent by
# construction**. A quad is drawn at the fractal's world extent; the vertex shader passes
# each corner's world coordinate down; the rasteriser interpolates it, so every *screen*
# fragment receives its own world coordinate and runs its own escape loop. Zoom the camera
# and each pixel now covers a smaller world region and recomputes at that finer scale — the
# detail refines automatically, every frame, for free. There is no grid to re-sample and no
# CPU recompute; "recompute on zoom for more precision" falls out of the architecture.
#
# The one honest limit is float32. World coordinates are interpolated in single precision,
# so once the visible span drops below ~1e-4 neighbouring pixels can no longer be told
# apart and the image pixelates. Going deeper needs double-precision or perturbation-theory
# emulation in the shader, which is a feature of its own; the renderer clamps the effective
# zoom and says so rather than showing mush.
FRACTAL_VS = r"""
#version 330 core
layout(location = 0) in vec2 a_pos;   // world coordinates of the quad corners
uniform mat4 u_mvp;
out vec2 v_world;
void main() {
    gl_Position = u_mvp * vec4(a_pos, 0.0, 1.0);
    v_world = a_pos;
}
"""

# The escape loop. ``u_type`` selects the family: 0 = Mandelbrot (c varies per pixel, z0=0),
# 1 = Julia (c is a fixed parameter, z0 varies per pixel). Both are z -> z^2 + c; the only
# difference is which of z0/c the pixel's world coordinate feeds. The loop bound is a
# compile-time constant with a runtime break, because a uniform loop bound is rejected by
# some strict (Metal-backed) GL drivers.
FRACTAL_FS = "#version 330 core\n" + HEATMAP_FUNCS + r"""
in vec2 v_world;
uniform int   u_max_iter;
uniform int   u_type;       // 0 = mandelbrot, 1 = julia
uniform vec2  u_julia_c;
uniform int   u_scheme;     // apply_heatmap index
uniform float u_gain;       // colour spread
uniform float u_alpha;
uniform vec3  u_inset_color;
out vec4 fragColor;

const int ITER_CAP = 2000;

void main() {
    vec2 c = (u_type == 0) ? v_world : u_julia_c;
    vec2 z = (u_type == 0) ? vec2(0.0) : v_world;
    int  n = 0;
    for (int i = 0; i < ITER_CAP; i++) {
        if (i >= u_max_iter) break;
        // z = z^2 + c
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        if (dot(z, z) > 256.0) break;   // generous bailout smooths the escape estimate
        n = i;
    }
    if (n >= u_max_iter - 1) {
        // Never escaped within the budget: treat as inside the set.
        fragColor = vec4(u_inset_color, u_alpha);
        return;
    }
    // Continuous (fractional) escape count — the standard renormalisation that removes the
    // integer banding so the colormap reads smooth.
    float sm = float(n) + 1.0 - log2(log(length(z)) + 1e-8);
    float t = clamp(sqrt(sm / float(u_max_iter)) * u_gain, 0.0, 1.0);
    fragColor = vec4(apply_heatmap(u_scheme, t), u_alpha);
}
"""

GEOMETRY3D_VS = r"""
#version 330 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec4 a_color;
layout(location = 2) in float a_size;   // per-point size multiplier (point clouds only)

uniform mat4 u_mvp;
uniform float u_point_size;
uniform float u_ref_w;   // clip-space w at scene centre (≈ camera distance);
                          // 0 = disable perspective sizing
uniform vec2 u_z_range;
uniform int u_is_points;

// ---- outline / silhouette ------------------------------------------------------
// Two unrelated techniques share these uniforms, because they are two answers to the
// same request on different primitives. See Geometry3DRenderer's class docstring for
// which one runs where and what each one cannot do.
//
// Master switch. 0 reproduces the pre-outline shader instruction for instruction:
// every branch below is guarded by it, and v_fill_frac is pinned to 1.0.
uniform int u_outline_enabled;
// 1 while the renderer is drawing one of the offset copies whose union dilates the
// layer's screen footprint (lines and triangles). 0 for the real geometry, and for the
// point ring, which is a single pass and needs no copies.
uniform int u_outline_pass;
// Ring thickness in framebuffer pixels, DPR already applied on the CPU. Only the point
// path reads it; the dilation copies get their displacement pre-computed instead.
uniform float u_outline_width;
// Screen-space displacement of THIS copy, in framebuffer pixels.
uniform vec2 u_outline_offset;
// Framebuffer size of the current viewport, in pixels; converts u_outline_offset to NDC.
uniform vec2 u_viewport;
// NDC depth added to every dilation copy. The layer is drawn first and owns the depth
// buffer over its own body; pushing the copies further away makes them fail GL_LESS
// there, so only the dilated rim survives. Without the bias a copy would land at the
// body's own depth, and whether it painted over the layer would come down to floating
// point -- the outline colour flooding the shape instead of edging it.
uniform float u_outline_depth_bias;

out vec4 v_color;
out float v_z_norm;
// World position, carried through so the fragment shader can honour explicit axis
// limits. Without it a set_zlim would only move the box, leaving the data spilling
// outside it -- which reads as a rendering bug rather than as a limit.
out vec3 v_world;
// Fraction of the point sprite's radius that is fill rather than ring, so the fragment
// shader can place the ring without knowing the per-point size. 1.0 (no ring) for every
// non-point primitive and whenever the outline is off.
out float v_fill_frac;

void main() {
    gl_Position = u_mvp * vec4(a_pos, 1.0);
    v_world = a_pos;
    // Perspective-correct point size: points closer than the scene centre appear
    // proportionally larger, farther points proportionally smaller.
    float sz = (u_ref_w > 0.0)
        ? u_point_size * u_ref_w / max(gl_Position.w, 1e-4)
        : u_point_size;
    // a_size makes marker size a data-driven dimension for point clouds. It is only read
    // when the layer is points (its VAO binds attribute 2); line/triangle layers leave the
    // attribute disabled, so the guard keeps their gl_PointSize (unused anyway) well-defined.
    float mult = (u_is_points == 1) ? a_size : 1.0;
    float fill = sz * mult;
    // Grow the sprite by the ring on both sides rather than eating into the marker, so a
    // point keeps the size the user asked for and the ring is drawn *outside* it -- the
    // same convention as the 2D scatter's point_outline_*.
    float grow = (u_is_points == 1 && u_outline_enabled == 1) ? 2.0 * u_outline_width : 0.0;
    float total = max(fill + grow, 0.5);
    gl_PointSize = clamp(total, 0.5, 512.0);
    v_fill_frac = (grow > 0.0) ? clamp(fill / max(total, 1e-4), 0.0, 1.0) : 1.0;
    if (u_outline_pass == 1) {
        // A shift of one framebuffer pixel is 2/viewport in NDC, and NDC is clip space
        // divided by w -- hence the multiply, which keeps the displacement a constant
        // number of pixels at every depth.
        gl_Position.xy += (u_outline_offset / max(u_viewport, vec2(1.0))) * 2.0 * gl_Position.w;
        gl_Position.z += u_outline_depth_bias * gl_Position.w;
    }
    v_color = a_color;
    v_z_norm = clamp((a_pos.z - u_z_range.x) / max(u_z_range.y - u_z_range.x, 1e-6), 0.0, 1.0);
}
"""

GEOMETRY3D_FS = r"""
#version 330 core
in vec4 v_color;
in float v_z_norm;
in vec3 v_world;
in float v_fill_frac;
uniform float u_alpha;
// Solved-for per-fragment alpha when ``style.auto_alpha`` is on, negative when it is not.
// See the bottom of main() for why it replaces rather than scales.
uniform float u_auto_alpha;
uniform int u_ssao_enabled;
uniform float u_ssao_strength;
uniform int u_is_points;
// Explicit axis limits in world units. Disabled when any lo >= hi, which is the state
// the engine sets when no limit has been asked for -- so the default costs three
// comparisons and never discards.
uniform vec3 u_clip_lo;
uniform vec3 u_clip_hi;
// Outline: see GEOMETRY3D_VS. u_outline_alpha is a separate factor rather than being
// folded into u_outline_color so the Style panel can fade a silhouette without editing
// the colour, and so the outline does not inherit the layer's own alpha -- an outline on
// a 20%-opaque mesh is there precisely to stay readable.
uniform int u_outline_enabled;
uniform int u_outline_pass;
uniform vec4 u_outline_color;
uniform float u_outline_alpha;
out vec4 fragColor;

void main() {
    if (u_clip_lo.x < u_clip_hi.x && (v_world.x < u_clip_lo.x || v_world.x > u_clip_hi.x)) discard;
    if (u_clip_lo.y < u_clip_hi.y && (v_world.y < u_clip_lo.y || v_world.y > u_clip_hi.y)) discard;
    if (u_clip_lo.z < u_clip_hi.z && (v_world.z < u_clip_lo.z || v_world.z > u_clip_hi.z)) discard;
    if (u_is_points == 1) {
        vec2 d = gl_PointCoord - vec2(0.5);
        if (dot(d, d) > 0.25) discard;
        // The ring, compared in squared radius so the common (outline off) path pays a
        // single multiply and no sqrt. 0.25 is the sprite's own radius squared.
        if (u_outline_enabled == 1 && dot(d, d) > 0.25 * v_fill_frac * v_fill_frac) {
            fragColor = vec4(u_outline_color.rgb, u_outline_color.a * u_outline_alpha);
            return;
        }
    }
    if (u_outline_pass == 1) {
        // A dilation copy is flat colour: no SSAO, no per-vertex colour. It exists only to
        // widen the silhouette, and shading it would make the rim read as more geometry.
        fragColor = vec4(u_outline_color.rgb, u_outline_color.a * u_outline_alpha);
        return;
    }
    float ao = 1.0;
    if (u_ssao_enabled == 1) {
        float cavity = 1.0 - smoothstep(0.10, 0.95, v_z_norm);
        float rim = smoothstep(0.0, 0.18, v_z_norm);
        ao = clamp(1.0 - u_ssao_strength * cavity * rim, 0.18, 1.0);
    }
    // Automatic alpha *replaces* the layer's own rather than scaling it: the alpha it
    // solves for is an absolute answer to "what makes a covered pixel land at the target
    // opacity", and the number it would otherwise be multiplied by is already baked into
    // the colour buffer for a volume3d (pyplot.volume3d does colors[:, 3] *= alpha), so
    // scaling would apply the caller's alpha twice. Negative is the "off" sentinel and is
    // what every layer that has not asked for it pushes, so the common path is unchanged.
    float alpha = (u_auto_alpha >= 0.0) ? u_auto_alpha : v_color.a * u_alpha;
    fragColor = vec4(v_color.rgb * ao, alpha);
}
"""


def mix(
    c1: Tuple[float, float, float], c2: Tuple[float, float, float], x: float
) -> Tuple[float, float, float]:
    return (
        c1[0] * (1.0 - x) + c2[0] * x,
        c1[1] * (1.0 - x) + c2[1] * x,
        c1[2] * (1.0 - x) + c2[2] * x,
    )


def eval_colormap(scheme_index: int, x: float) -> tuple[float, float, float]:
    """Evaluate colormap RGB color for a given normalized value [0.0, 1.0]."""
    x = max(0.0, min(1.0, x))
    if scheme_index == 1:  # Viridis
        c0 = (0.267, 0.005, 0.329)
        c1 = (0.283, 0.141, 0.458)
        c2 = (0.254, 0.265, 0.530)
        c3 = (0.207, 0.372, 0.553)
        c4 = (0.164, 0.471, 0.558)
        c5 = (0.128, 0.567, 0.551)
        c6 = (0.135, 0.659, 0.518)
        c7 = (0.267, 0.749, 0.441)
        c8 = (0.478, 0.821, 0.318)
        c9 = (0.741, 0.873, 0.150)
        if x < 0.11:
            return mix(c0, c1, x / 0.11)
        if x < 0.22:
            return mix(c1, c2, (x - 0.11) / 0.11)
        if x < 0.33:
            return mix(c2, c3, (x - 0.22) / 0.11)
        if x < 0.44:
            return mix(c3, c4, (x - 0.33) / 0.11)
        if x < 0.55:
            return mix(c4, c5, (x - 0.44) / 0.11)
        if x < 0.66:
            return mix(c5, c6, (x - 0.55) / 0.11)
        if x < 0.77:
            return mix(c6, c7, (x - 0.66) / 0.11)
        if x < 0.88:
            return mix(c7, c8, (x - 0.77) / 0.11)
        return mix(c8, c9, (x - 0.88) / 0.12)
    elif scheme_index == 2:  # Plasma
        c0 = (0.050, 0.030, 0.528)
        c1 = (0.291, 0.071, 0.718)
        c2 = (0.507, 0.104, 0.749)
        c3 = (0.692, 0.165, 0.564)
        c4 = (0.845, 0.277, 0.388)
        c5 = (0.954, 0.468, 0.199)
        c6 = (0.940, 0.975, 0.131)
        if x < 0.16:
            return mix(c0, c1, x / 0.16)
        if x < 0.32:
            return mix(c1, c2, (x - 0.16) / 0.16)
        if x < 0.48:
            return mix(c2, c3, (x - 0.32) / 0.16)
        if x < 0.64:
            return mix(c3, c4, (x - 0.48) / 0.16)
        if x < 0.80:
            return mix(c4, c5, (x - 0.64) / 0.16)
        return mix(c5, c6, (x - 0.80) / 0.20)
    elif scheme_index == 3:  # Inferno
        c0 = (0.000, 0.000, 0.016)
        c1 = (0.073, 0.038, 0.201)
        c2 = (0.243, 0.053, 0.404)
        c3 = (0.449, 0.111, 0.437)
        c4 = (0.665, 0.177, 0.366)
        c5 = (0.866, 0.301, 0.228)
        c6 = (0.976, 0.505, 0.096)
        c7 = (0.985, 0.768, 0.263)
        c8 = (0.988, 0.941, 0.729)
        if x < 0.125:
            return mix(c0, c1, x / 0.125)
        if x < 0.250:
            return mix(c1, c2, (x - 0.125) / 0.125)
        if x < 0.375:
            return mix(c2, c3, (x - 0.250) / 0.125)
        if x < 0.500:
            return mix(c3, c4, (x - 0.375) / 0.125)
        if x < 0.625:
            return mix(c4, c5, (x - 0.500) / 0.125)
        if x < 0.750:
            return mix(c5, c6, (x - 0.625) / 0.125)
        if x < 0.875:
            return mix(c6, c7, (x - 0.750) / 0.125)
        return mix(c7, c8, (x - 0.875) / 0.125)
    elif scheme_index == 4:  # Turbo
        g0 = (0.12, 0.01, 0.22)
        g1 = (0.13, 0.15, 0.48)
        g2 = (0.15, 0.65, 0.51)
        g3 = (0.85, 0.60, 0.12)
        g4 = (0.92, 0.11, 0.43)
        if x < 0.25:
            return mix(g0, g1, x / 0.25)
        if x < 0.50:
            return mix(g1, g2, (x - 0.25) / 0.25)
        if x < 0.75:
            return mix(g2, g3, (x - 0.50) / 0.25)
        return mix(g3, g4, (x - 0.75) / 0.25)
    elif scheme_index == 5:  # Ink Fire
        c0 = (1.0, 1.0, 1.0)
        c1 = (1.0, 0.9, 0.2)
        c2 = (1.0, 0.2, 0.1)
        c3 = (0.4, 0.0, 0.0)
        c4 = (0.0, 0.0, 0.0)
        if x < 0.25:
            return mix(c0, c1, x / 0.25)
        if x < 0.50:
            return mix(c1, c2, (x - 0.25) / 0.25)
        if x < 0.75:
            return mix(c2, c3, (x - 0.50) / 0.25)
        return mix(c3, c4, (x - 0.75) / 0.25)
    elif scheme_index == 6:  # Magma
        c0 = (0.001, 0.000, 0.031)
        c1 = (0.170, 0.047, 0.360)
        c2 = (0.447, 0.051, 0.439)
        c3 = (0.729, 0.160, 0.345)
        c4 = (0.960, 0.419, 0.231)
        c5 = (0.988, 0.768, 0.470)
        c6 = (0.988, 0.988, 0.823)
        if x < 0.16:
            return mix(c0, c1, x / 0.16)
        if x < 0.32:
            return mix(c1, c2, (x - 0.16) / 0.16)
        if x < 0.48:
            return mix(c2, c3, (x - 0.32) / 0.16)
        if x < 0.64:
            return mix(c3, c4, (x - 0.48) / 0.16)
        if x < 0.80:
            return mix(c4, c5, (x - 0.64) / 0.16)
        return mix(c5, c6, (x - 0.80) / 0.20)
    elif scheme_index == 7:  # Grayscale
        return (x, x, x)
    elif scheme_index == 8:  # Ocean
        c0 = (0.0, 0.0, 0.1)
        c1 = (0.0, 0.2, 0.6)
        c2 = (0.0, 0.8, 1.0)
        c3 = (1.0, 1.0, 1.0)
        if x < 0.33:
            return mix(c0, c1, x / 0.33)
        if x < 0.66:
            return mix(c1, c2, (x - 0.33) / 0.33)
        return mix(c2, c3, (x - 0.66) / 0.34)
    elif scheme_index == 9:  # Hot
        c0 = (0.0, 0.0, 0.0)
        c1 = (0.8, 0.0, 0.0)
        c2 = (1.0, 0.5, 0.0)
        c3 = (1.0, 1.0, 0.0)
        c4 = (1.0, 1.0, 1.0)
        if x < 0.25:
            return mix(c0, c1, x / 0.25)
        if x < 0.50:
            return mix(c1, c2, (x - 0.25) / 0.25)
        if x < 0.75:
            return mix(c2, c3, (x - 0.50) / 0.25)
        return mix(c3, c4, (x - 0.75) / 0.25)
    elif scheme_index == 10:  # Cool
        return mix((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), x)
    else:  # Classic / Default (0)

        def smoothstep(e0, e1, v):
            t = max(0.0, min(1.0, (v - e0) / (e1 - e0)))
            return t * t * (3.0 - 2.0 * t)

        return (smoothstep(0.0, 0.3, x), smoothstep(0.3, 0.6, x), smoothstep(0.6, 1.0, x))


def get_colormap_min_color(
    scheme_index: int, invert: bool, light_to_color: bool = True
) -> tuple[float, float, float]:
    """Evaluate the minimum density (x=0.0) color for the specified scheme."""
    # If inverted, the minimum density maps to the high end (1.0) of the colormap.
    x = 1.0 if invert else 0.0
    return eval_colormap(scheme_index, x)


# ==============================================================================
# PASS 1: EXACT RENDERING (Primal Geometry)
# ==============================================================================

# --- LINES ---

EXACT_LINES_VS = r"""
#version 330 core
layout(location=0) in float a_t;
layout(location=1) in vec2  a_ab;
layout(location=2) in vec4  a_col;

uniform mat4  u_mvp;
uniform vec2  u_xrange;
uniform vec4  u_window;
uniform int   u_use_color;
uniform float u_alpha;
uniform int   u_enable_subsample;
uniform float u_keep_prob;
uniform int   u_total_count;
uniform vec2  u_layer_offset;

out vec4 v_col;
flat out float v_id_norm;
out float v_side;

void main() {
    float x = mix(u_xrange.x, u_xrange.y, a_t);
    float y = a_ab.x * x + a_ab.y;
    vec2  w = vec2(x, y) + u_layer_offset;

    gl_Position = u_mvp * vec4(w, 0.0, 1.0);

    gl_ClipDistance[0] =  w.x - u_window.x;
    gl_ClipDistance[1] =  u_window.y - w.x;
    gl_ClipDistance[2] =  w.y - u_window.z;
    gl_ClipDistance[3] =  u_window.w - w.y;

    float l = u_window.x, r = u_window.y;
    float xmin = u_xrange.x, xmax = u_xrange.y;
    float xA = max(l, xmin);
    float xB = min(r, xmax);
    bool noOverlapX = (xA > xB);
    float yA = a_ab.x * xA + a_ab.y;
    float yB = a_ab.x * xB + a_ab.y;
    float bottom = u_window.z, top = u_window.w;
    bool outsideY = (yA > top && yB > top) || (yA < bottom && yB < bottom);

    uint id = uint(gl_InstanceID);
    v_id_norm = (u_total_count > 1) ? float(id) / float(u_total_count - 1) : 0.0;

    id ^= id >> 17; id *= 0xed5ad4bbu; id ^= id >> 11;
    id *= 0xac4c1b51u; id ^= id >> 15; id *= 0x31848babu;
    float rnd = float(id & 0x00FFFFFFu) * (1.0/16777215.0);
    bool drop = (u_enable_subsample == 1) && (rnd > u_keep_prob);

    if (noOverlapX || outsideY || drop) {
        gl_ClipDistance[0] = -1.0;
        gl_ClipDistance[1] = -1.0;
        gl_ClipDistance[2] = -1.0;
        gl_ClipDistance[3] = -1.0;
    }

    v_col = (u_use_color == 1) ? a_col : vec4(0.0, 0.0, 0.0, 1.0);
    v_col.a *= u_alpha;
    v_side = 0.0;
}
"""

WIDE_LINES_INSTANCED_VS = r"""
#version 330 core

layout(location=0) in vec2  a_corner;   // (t, side): t in {0,1}, side in {-0.5,+0.5}
layout(location=1) in vec2  a_ab;       // slope, intercept
layout(location=2) in vec4  a_col;      // instanced color

uniform vec2  u_xrange;
uniform vec4  u_window;         // l, r, b, t
uniform vec2  u_ndc_scale;
uniform vec2  u_ndc_offset;
uniform vec2  u_viewport_size;
uniform float u_width;
uniform float u_alpha;
uniform int   u_use_color;
uniform float u_keep_prob;
uniform int   u_total_count;
uniform vec2  u_layer_offset;

// ---- outline / silhouette (LayerStyle.outline_*) --------------------------------
// Same casing as WIDE_SEGMENT_INSTANCED_VS, for the same reason: this shader builds the
// line's width itself, so a casing is one wider re-draw rather than a K-copy dilation.
// One extra instanced draw doubles the cost of the whole layer, which is why
// LineFamilyRenderer refuses the outline above a few hundred thousand lines instead of
// silently halving the frame rate. 0 leaves the pre-outline geometry untouched.
uniform int   u_outline_pass;
uniform float u_outline_width;   // framebuffer px per side, DPR already applied

out vec4 v_col;
flat out float v_id_norm;
out float v_side;

void main() {
    uint id = uint(gl_InstanceID);
    v_id_norm = (u_total_count > 1) ? float(id) / float(u_total_count - 1) : 0.0;

    // Early probabilistic LOD
    uint h = id;
    h ^= h >> 17; h *= 0xed5ad4bbu; h ^= h >> 11;
    h *= 0xac4c1b51u; h ^= h >> 15; h *= 0x31848babu;
    float rnd = float(h & 0x00FFFFFFu) * (1.0 / 16777215.0);
    bool drop = rnd > u_keep_prob;

    float l = u_window.x;
    float r = u_window.y;
    float b = u_window.z;
    float t = u_window.w;

    float xmin = u_xrange.x;
    float xmax = u_xrange.y;

    float slope = a_ab.x;
    float intercept = a_ab.y;

    float xA = max(l, xmin);
    float xB = min(r, xmax);
    bool noOverlapX = (xA >= xB);

    float yA = slope * xA + intercept;
    float yB = slope * xB + intercept;

    // Use a small adaptive epsilon to prevent precision-induced popping at edges
    float eps = (t - b) * 0.1;
    float ext_t = t + eps;
    float ext_b = b - eps;

    bool outsideY = (yA > ext_t && yB > ext_t) || (yA < ext_b && yB < ext_b);

    if (drop || noOverlapX || outsideY) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        v_col = vec4(0.0);
        return;
    }

    // Y clipping to prevent precision loss on very steep lines
    if (abs(slope) > 1e-9) {
        if (yA > ext_t) { xA = xA + (ext_t - yA) / slope; yA = ext_t; }
        else if (yA < ext_b) { xA = xA + (ext_b - yA) / slope; yA = ext_b; }

        if (yB > ext_t) { xB = xB + (ext_t - yB) / slope; yB = ext_t; }
        else if (yB < ext_b) { xB = xB + (ext_b - yB) / slope; yB = ext_b; }
    }

    vec2 center = vec2(u_window.x + u_window.y, u_window.z + u_window.w) * 0.5;
    vec2 p0_rel = vec2(xA, yA) - center + u_layer_offset;
    vec2 p1_rel = vec2(xB, yB) - center + u_layer_offset;

    // Use relative NDC projection (no u_ndc_offset needed if using center-relative)
    vec2 ndc0 = p0_rel * u_ndc_scale;
    vec2 ndc1 = p1_rel * u_ndc_scale;

    vec2 dir_px = (ndc1 - ndc0) * (0.5 * u_viewport_size);
    float len2 = dot(dir_px, dir_px);

    vec2 n_px = (len2 > 1e-12)
        ? normalize(vec2(-dir_px.y, dir_px.x))
        : vec2(0.0, 1.0);

    vec2 p_ndc = mix(ndc0, ndc1, a_corner.x);
    // a_corner.y is +-0.5, and 1px = 2/viewport in NDC, so the two factors of 2
    // cancel only if the side term is doubled first. Without it the quad is half
    // the requested width and thin lines land under one pixel.
    float width = u_width;
    // The casing is only widened, never lengthened: every line in a family already spans
    // the whole x range and is clipped to the window above, so extending its ends would
    // push the casing into the axis gutter and nothing else.
    if (u_outline_pass == 1) width += 2.0 * u_outline_width;
    p_ndc += n_px * ((width / u_viewport_size) * (a_corner.y * 2.0));

    gl_Position = vec4(p_ndc, 0.0, 1.0);
    v_side = a_corner.y * 2.0; // Map [-0.5, 0.5] to [-1, 1]

    v_col = (u_use_color == 1) ? a_col : vec4(0.0, 0.0, 0.0, 1.0);
    v_col.a *= u_alpha;
}
"""

EXACT_LINES_FS = (
    r"""
#version 330 core
in vec4 v_col;
flat in float v_id_norm;
in float v_side;
out vec4 FragColor;

uniform int u_use_colormap;
uniform int u_scheme;
uniform int u_antialiasing;

// Outline / silhouette; see WIDE_LINES_INSTANCED_VS. Declared here rather than in a
// program of its own because this fragment shader is linked against two vertex shaders
// (the exact line path and the instanced line-family path) and only the second one can
// ever turn the pass on -- an unset uniform is 0, which is "off".
uniform int u_outline_pass;
uniform vec4 u_outline_color;
uniform float u_outline_alpha;

"""
    + HEATMAP_FUNCS
    + r"""

void main() {
    float aa = 1.0;
    if (u_antialiasing == 1) {
        float d = abs(v_side);
        float w = fwidth(d);
        aa = 1.0 - smoothstep(1.0 - w, 1.0, d);
    }
    float alpha = v_col.a * aa;

    if (u_outline_pass == 1) {
        // Flat colour, and deliberately not v_col.a: the casing carries its own opacity so
        // it stays readable under a line family drawn at a very low per-line alpha.
        FragColor = vec4(u_outline_color.rgb, u_outline_color.a * u_outline_alpha * aa);
        return;
    }
    if (alpha <= 0.0) discard;

    vec4 color = v_col;
    if (u_use_colormap == 1) {
        color.rgb = apply_heatmap(u_scheme, v_id_norm);
    }
    FragColor = vec4(color.rgb, alpha);
}
"""
)

# --- IMAGE (textured quad for imshow) ---

IMAGE_VS = r"""
#version 330 core
layout(location=0) in vec2 a_pos;
layout(location=1) in vec2 a_uv;
uniform mat4 u_mvp;
out vec2 v_uv;
void main() {
    gl_Position = u_mvp * vec4(a_pos, 0.0, 1.0);
    v_uv = a_uv;
}
"""

IMAGE_FS = r"""
#version 330 core
in vec2 v_uv;
uniform sampler2D u_tex;
uniform float u_alpha;
out vec4 f_color;
void main() {
    vec4 col = texture(u_tex, v_uv);
    if (col.a < 0.004) discard;
    col.a *= u_alpha;
    f_color = col;
}
"""

# --- SCATTERS ---

SCATTER_VS = r"""
#version 330 core
layout(location=0) in vec2 a_pos;
layout(location=1) in vec4 a_color;
layout(location=2) in float a_size;   // per-point size multiplier; 1.0 = the layer's u_size
uniform mat4  u_mvp;
uniform float u_size;
uniform float u_alpha;
uniform vec2  u_layer_offset;
uniform vec4  u_window;   // (xmin, xmax, ymin, ymax) world space

// ---- outline / silhouette (LayerStyle.outline_*) --------------------------------
// The general outline: the same analytic ring GEOMETRY3D_VS/FS draw around a 3D point.
// The sprite is grown by twice the ring width here and the fragment shader paints
// everything past the marker's own radius in the outline colour -- one pass, no extra
// draw call. Unrelated to u_point_outline_* in SCATTER_FS, which is this renderer's
// older ring drawn *inside* the marker; the two are independent switches (see
// LayerStyle) and may both be on, in which case the general ring sits outside the
// per-point one.
//
// 0 reproduces the pre-outline shader exactly: `grow` is then the literal 0.0, so
// gl_PointSize is the same float it has always been and v_fill_frac is pinned to 1.0.
uniform int   u_outline_enabled;
// Ring thickness in framebuffer pixels, DPR already applied on the CPU.
uniform float u_outline_width;

out vec4 v_col;
// Fraction of the sprite's radius that is marker rather than ring, so the fragment shader
// can place the ring without knowing this point's size. 1.0 whenever the outline is off,
// which is what keeps the fragment arithmetic identical to the pre-outline shader.
out float v_fill_frac;
void main() {
    vec2 world_pos = a_pos + u_layer_offset;
    gl_Position = u_mvp * vec4(world_pos, 0.0, 1.0);
    // Must always write gl_ClipDistance when GL_CLIP_DISTANCE0-3 are enabled (see PATCH_VS);
    // leaving it undefined is what let markers spill past the axis margins into the tick
    // label gutters -- and beyond, to the window edge -- whenever the camera view is a crop
    // of the data range (any zoom/pan) instead of an exact fit around it.
    gl_ClipDistance[0] = world_pos.x - u_window.x;
    gl_ClipDistance[1] = u_window.y - world_pos.x;
    gl_ClipDistance[2] = world_pos.y - u_window.z;
    gl_ClipDistance[3] = u_window.w - world_pos.y;
    // a_size lets marker size be a data-driven dimension. The buffer holds 1.0 for every
    // point when no per-point sizes were given, so gl_PointSize == u_size as before.
    float fill = u_size * a_size;
    // Grow the sprite by the ring on both sides rather than eating into the marker, so a
    // point keeps the size the user asked for and the ring is drawn *outside* it -- the
    // same convention as the 3D point ring.
    float grow = (u_outline_enabled == 1) ? 2.0 * u_outline_width : 0.0;
    gl_PointSize = fill + grow;
    v_fill_frac = (grow > 0.0) ? clamp(fill / max(fill + grow, 1e-4), 0.0, 1.0) : 1.0;
    v_col = a_color;
    v_col.a *= u_alpha;
}
"""

SCATTER_FS = r"""
#version 330 core
in vec4 v_col;
in float v_fill_frac;
out vec4 FragColor;

// The scatter's own per-marker ring (LayerStyle.point_outline_*): drawn *inside* the
// marker, so the sprite keeps its size and the fill shrinks. Historical behaviour, and
// what `edgecolors=` on an existing script still means.
uniform int u_point_outline_enabled;
uniform vec4 u_point_outline_color;
uniform float u_point_outline_width_px;
uniform float u_point_size_px;

// The general outline (LayerStyle.outline_*): drawn *outside* the marker, in the band
// SCATTER_VS grew the sprite by. See that shader for why the two coexist.
uniform int u_outline_enabled;
uniform vec4 u_outline_color;
uniform float u_outline_alpha;

// Which shape the sprite is masked to: 0 circle (the default and the historical
// behaviour), 1 square, 2 triangle-up, 3 diamond, 4 x-cross, 5 plus. `marker=` used to be
// stored on the layer and honoured only by the headless PNG export, so the live window
// drew every marker as a circle and silently disagreed with the exported figure.
uniform int u_marker_shape;

// Half-width of a cross's stroke, as a fraction of the sprite half-extent. Thin enough to
// read as a drawn cross rather than a blob, thick enough to survive a 4 px sprite.
const float CROSS_HALF_WIDTH = 0.30;

// A normalised radius for `shape`: <1 inside, ==1 on the boundary, >1 outside -- the same
// contract `length(q)` has for a circle, so every band/feather/outline test below keeps
// working unchanged whichever shape is selected.
float marker_radius(vec2 q, int shape) {
    if (shape == 1) return max(abs(q.x), abs(q.y));            // square (Chebyshev)
    // gl_PointCoord runs y-DOWN, so the apex of an "up" triangle is at q.y == -1.
    if (shape == 2) return max(q.y, 2.0 * abs(q.x) - q.y);     // triangle, apex up
    if (shape == 6) return max(-q.y, 2.0 * abs(q.x) + q.y);    // triangle, apex down
    if (shape == 3) return abs(q.x) + abs(q.y);                // diamond (Manhattan)
    if (shape == 4) {                                          // x: plus, rotated 45 deg
        vec2 d = vec2(q.x + q.y, q.y - q.x) * 0.70710678;
        return min(max(abs(d.x) / CROSS_HALF_WIDTH, abs(d.y)),
                   max(abs(d.y) / CROSS_HALF_WIDTH, abs(d.x)));
    }
    if (shape == 5) {                                          // plus: two crossed bars
        return min(max(abs(q.x) / CROSS_HALF_WIDTH, abs(q.y)),
                   max(abs(q.y) / CROSS_HALF_WIDTH, abs(q.x)));
    }
    return length(q);                                          // 0: circle
}

void main() {
    vec2 p = gl_PointCoord - vec2(0.5);
    // q spans [-1, 1] across the sprite, so `length(q)` is bit-for-bit the `length(p)*2.0`
    // this shader used before shapes existed.
    vec2 q = p * 2.0;
    float r = marker_radius(q, u_marker_shape);  // 0 center, ~1 edge

    if (r > 1.0) discard;

    // Soft edge antialiasing, computed once: the sprite's rim is the outline's outer edge
    // when there is an outline and the marker's edge when there is not.
    float feather = fwidth(r) * 1.5;
    float edge = 1.0 - smoothstep(1.0 - feather, 1.0, r);

    // Radius rescaled to the marker itself. v_fill_frac is exactly 1.0 with the outline
    // off, so `rf` is bit-for-bit `r` and every line below behaves as it always has.
    float rf = r;
    if (u_outline_enabled == 1) {
        rf = r / max(v_fill_frac, 1e-4);
        if (rf > 1.0) {
            // The ring. Its alpha deliberately ignores v_col.a (the layer's own alpha):
            // an outline on a translucent cloud exists to keep it readable.
            FragColor = vec4(u_outline_color.rgb, u_outline_color.a * u_outline_alpha * edge);
            return;
        }
    }

    float outline_frac = (u_point_size_px > 0.0)
        ? clamp(u_point_outline_width_px / u_point_size_px, 0.0, 0.49)
        : 0.0;

    float fill_radius = 1.0 - 2.0 * outline_frac;

    vec4 col = v_col;

    if (u_point_outline_enabled == 1 && rf > fill_radius) {
        col.rgb = u_point_outline_color.rgb;
        col.a *= u_point_outline_color.a;
    }

    col.a *= edge;

    FragColor = col;
}
"""

#: matplotlib marker string -> the ``u_marker_shape`` value ``SCATTER_FS`` switches on.
#:
#: Lives beside the shader rather than in the renderer so the two cannot drift: the
#: integers here are only meaningful as the cases that shader implements.
#:
#: Deliberately partial. matplotlib knows dozens of markers (and accepts paths and
#: verts); these are the six the fragment shader draws. Anything else falls back to the
#: circle -- the shape GLPlot has always drawn for every marker -- so an unknown marker
#: degrades to today's behaviour rather than to nothing.
MARKER_SHAPE_INDEX = {
    "o": 0,
    ".": 0,
    "s": 1,
    "^": 2,
    "v": 6,
    "D": 3,
    "d": 3,
    "x": 4,
    "X": 4,
    "+": 5,
    "P": 5,
}


def marker_shape_index(marker: object) -> int:
    """``u_marker_shape`` for a layer's ``metadata["marker"]``; 0 (circle) for anything else."""
    if not isinstance(marker, str):
        return 0
    return MARKER_SHAPE_INDEX.get(marker, 0)


# --- STRIPS ---

STRIP_VS = r"""
#version 330 core
layout(location=0) in vec2 a_pos;
uniform mat4  u_mvp;
uniform vec4  u_color;
uniform float u_alpha;
uniform vec2  u_layer_offset;
out vec4 v_col;
void main() {
    gl_Position = u_mvp * vec4(a_pos + u_layer_offset, 0.0, 1.0);
    v_col = u_color;
    v_col.a *= u_alpha;
}
"""

STRIP_FS = r"""
#version 330 core
in vec4 v_col;
out vec4 FragColor;
void main() {
    FragColor = v_col;
}
"""

# --- WIDE LINES (Quad Expansion) ---

WIDE_LINE_VS = r"""
#version 330 core
layout(location=0) in vec2 a_pos;
layout(location=1) in vec2 a_next;
layout(location=2) in float a_side; // -1.0 or 1.0

uniform mat4  u_mvp;
uniform vec2  u_viewport_size;
uniform float u_width;
uniform vec4  u_color;
uniform float u_alpha;

out vec4 v_col;

void main() {
    float width = max(1.0, u_width);

    vec4 p1 = u_mvp * vec4(a_pos, 0.0, 1.0);
    vec4 p2 = u_mvp * vec4(a_next, 0.0, 1.0);

    vec2 ndc1 = p1.xy / p1.w;
    vec2 ndc2 = p2.xy / p2.w;

    // Direction and normal in screen space
    vec2 dir = normalize((ndc2 - ndc1) * u_viewport_size);
    vec2 norm = vec2(-dir.y, dir.x);

    // Offset in NDC space
    vec2 offset = norm * (width / u_viewport_size) * a_side;

    // Determine if this vertex belongs to the start or end of the segment
    // We assume 4 vertices per segment (0,1 at start; 2,3 at end)
    float is_end = float(gl_VertexID % 4 >= 2);
    vec4 p = mix(p1, p2, is_end);

    p.xy += offset * p.w;

    gl_Position = p;
    v_col = u_color;
    v_col.a *= u_alpha;
}
"""

WIDE_LINE_FS = r"""
#version 330 core
in vec4 v_col;
out vec4 FragColor;
void main() {
    FragColor = v_col;
}
"""

# --- INSTANCED SEGMENTS (GPU Expansion) ---

WIDE_SEGMENT_INSTANCED_VS = r"""
#version 330 core

layout(location=0) in vec2 a_corner;   // (t, side): t in {0,1}, side in {-0.5,+0.5}
layout(location=1) in vec2 i_p0;       // segment start
layout(location=2) in vec2 i_p1;       // segment end
layout(location=3) in vec4 i_col0;     // per-segment colour at its start vertex
layout(location=4) in vec4 i_col1;     // per-segment colour at its end vertex

uniform mat4  u_mvp;                   // margin-adjusted ortho projection (matches patch/scatter)
uniform vec2  u_viewport_size;         // framebuffer size in pixels
uniform vec4  u_color;
uniform float u_alpha;
uniform float u_width;
uniform float u_id_norm;
uniform vec2  u_layer_offset;
uniform vec4  u_window;                // (xmin, xmax, ymin, ymax) world space
uniform bool  u_use_vertex_color;      // colour a line by data: gradient per segment

// ---- outline / silhouette (LayerStyle.outline_*) --------------------------------
// A polyline's outline is a *casing*: the same segment quads, drawn once more underneath
// the stroke and 2*u_outline_width px wider. That is one extra draw call, not the K-copy
// screen-space dilation the 3D renderer needs, because this shader already builds the
// stroke's width itself -- widening it is a change to one float, and the result is an
// exact uniform-width casing rather than a ring sampled at K directions.
// 0 reproduces the pre-outline shader exactly: the whole block below is skipped, so not
// one float of the stroke's own geometry is touched.
uniform int   u_outline_pass;
// Casing thickness on each side, in framebuffer pixels, DPR already applied on the CPU.
uniform float u_outline_width;

out vec4 v_col;
flat out float v_id_norm;

void main() {
    // Use the same MVP as the patch renderer so shaft endpoints and arrowhead
    // vertices are in the same coordinate space (margin-adjusted plot area).
    vec2 ndc0 = (u_mvp * vec4(i_p0 + u_layer_offset, 0.0, 1.0)).xy;
    vec2 ndc1 = (u_mvp * vec4(i_p1 + u_layer_offset, 0.0, 1.0)).xy;

    // Convert NDC delta to pixels
    vec2 dir_px = (ndc1 - ndc0) * (0.5 * u_viewport_size);
    float len2 = dot(dir_px, dir_px);

    vec2 n_px = (len2 > 1e-12)
        ? normalize(vec2(-dir_px.y, dir_px.x))
        : vec2(0.0, 1.0);

    // Extend sub-pixel segments to minimum 1px so every pixel along the
    // path is covered.  Without this, near-horizontal segments shorter than
    // 1px leave gaps between rasterised quads, making smooth curves appear
    // as dotted lines at wide zoom-out.
    float len_px = sqrt(max(len2, 0.0));
    float draw_scale = (len_px > 1e-6) ? max(len_px, 1.0) / len_px : 1.0;
    vec2 ndc1_draw = ndc0 + (ndc1 - ndc0) * draw_scale;

    vec2 p_ndc = mix(ndc0, ndc1_draw, a_corner.x);

    // u_width is full width; offset from centerline is ±0.5*u_width.
    // a_corner.y is ±0.5 and 1px = 2/viewport in NDC, so the side term must be
    // doubled; without it the quad is half the requested width.
    float width = u_width;
    if (u_outline_pass == 1) {
        // The casing: wider by u_outline_width on each side...
        width += 2.0 * u_outline_width;
        // ...and longer by the same at both ends. Without the lengthwise extension the
        // stroke pokes out of its own silhouette at the two ends of the path, and the
        // notch on the outer side of every join (these are independent quads, with no
        // join geometry) stays uncased -- which is exactly where a casing is read.
        vec2 t_px = (len2 > 1e-12) ? normalize(dir_px) : vec2(1.0, 0.0);
        p_ndc += t_px * (2.0 * u_outline_width / u_viewport_size) * (a_corner.x * 2.0 - 1.0);
    }
    vec2 offset_ndc = n_px * (width / u_viewport_size) * (a_corner.y * 2.0);
    p_ndc += offset_ndc;

    gl_Position = vec4(p_ndc, 0.0, 1.0);

    // World-space clip distances — must always be written when the engine
    // enables GL_CLIP_DISTANCE0-3.  Omitting them leaves the values undefined,
    // which causes the driver to randomly cull vertices, making lines appear
    // as dotted dashes.  We clip against the visible world window so that
    // off-screen segments are skipped cheaply, matching the line-family optimisation.
    vec2 world_pos = mix(i_p0, i_p1, a_corner.x) + u_layer_offset;
    gl_ClipDistance[0] = world_pos.x - u_window.x;
    gl_ClipDistance[1] = u_window.y - world_pos.x;
    gl_ClipDistance[2] = world_pos.y - u_window.z;
    gl_ClipDistance[3] = u_window.w - world_pos.y;

    // A data-driven line colour interpolates each segment between its two endpoints'
    // colours (a_corner.x is the along-segment t), so the whole path is a smooth gradient.
    // Gated on the uniform: a flat line binds no colour buffer and keeps u_color.
    v_col = u_use_vertex_color ? mix(i_col0, i_col1, a_corner.x) : u_color;
    v_col.a *= u_alpha;
    v_id_norm = u_id_norm;
}
"""

WIDE_SEGMENT_INSTANCED_FS = (
    r"""
#version 330 core
in vec4 v_col;
flat in float v_id_norm;
out vec4 FragColor;

uniform int u_use_colormap;
uniform int u_scheme;

// Outline / silhouette; see WIDE_SEGMENT_INSTANCED_VS. u_outline_alpha is a factor of its
// own rather than being folded into u_outline_color so the casing can stay opaque under a
// translucent stroke -- which is the whole point of a casing -- while a scene-wide fade
// (ctx.global_alpha, folded in on the CPU) still fades it.
uniform int u_outline_pass;
uniform vec4 u_outline_color;
uniform float u_outline_alpha;

"""
    + HEATMAP_FUNCS
    + r"""

void main() {
    if (u_outline_pass == 1) {
        // The casing is flat colour: no colormap, no per-vertex gradient. It exists to
        // separate the stroke from the background, and shading it would read as a second
        // line rather than as the first one's edge.
        FragColor = vec4(u_outline_color.rgb, u_outline_color.a * u_outline_alpha);
        return;
    }
    vec4 color = v_col;
    if (u_use_colormap == 1) {
        color.rgb = apply_heatmap(u_scheme, v_id_norm);
    }
    FragColor = color;
}
"""
)

#: What every density accumulation shader writes into the shared RGBA32F target.
#:
#: ``r`` is the weight -- the scalar the buffer used to hold on its own, so every number the
#: normalisation and the max readback work with is unchanged, and both readbacks still ask
#: the driver for one channel (``GL_RED``) rather than four. ``gba`` is that weight *times*
#: the fragment's colour, which is what makes the buffer carry a colour at all: summed under
#: ``GL_ONE, GL_ONE`` the channels come out as ``sum(w_i)`` and ``sum(w_i * c_i)``, so
#: ``gba / r`` at resolve time is the weight-averaged colour of everything that landed on
#: that pixel. Premultiplying is the whole trick -- an average cannot be accumulated, but a
#: weighted sum and its weight can, and their ratio is the average.
DENSITY_WEIGHT_TO_RGBA = r"""
vec4 density_sample(vec3 rgb, float w) {
    return vec4(w, rgb * w);
}
"""


WIDE_SEGMENT_DENSITY_FS = (
    r"""
#version 330 core
in vec4 v_col;
layout(location=0) out vec4 FragValue;

uniform int u_density_weighted;
"""
    + DENSITY_WEIGHT_TO_RGBA
    + r"""
void main() {
    float w = (u_density_weighted == 1) ? v_col.a : 1.0;
    FragValue = density_sample(v_col.rgb, w);
}
"""
)

# --- PATCHES ---

PATCH_VS = r"""
#version 330 core
layout(location=0) in vec2 a_pos;
// Per-vertex colour, for a patch whose pieces are not all one colour -- a hexbin's
// hexagons, a tripcolor's triangles. Gated on u_use_vertex_color rather than always read,
// because a patch with one face_color (every bar, every fill_between, every pie wedge)
// binds no colour buffer at all and this attribute would then be undefined.
layout(location=1) in vec4 a_col;
uniform mat4  u_mvp;
uniform vec4  u_color;
uniform float u_alpha;
uniform bool  u_use_vertex_color;
uniform vec2  u_layer_offset;
uniform vec4  u_window;   // (xmin, xmax, ymin, ymax) world space

// ---- outline / silhouette (LayerStyle.outline_*) --------------------------------
// A filled shape has no width uniform to widen, so its outline is the same screen-space
// dilation the 3D renderer uses on triangles: the layer is redrawn once per direction on
// a circle of radius u_outline_width, and the union of the copies is the shape's footprint
// grown by a disc. The copies go down *first* here, unlike in 3D -- see PatchRenderer.
// 0 reproduces the pre-outline shader exactly: gl_Position is never touched.
uniform int   u_outline_pass;
// Screen-space displacement of THIS copy, in framebuffer pixels.
uniform vec2  u_outline_offset;
// Framebuffer size of the current viewport, in px; converts u_outline_offset to NDC.
uniform vec2  u_viewport;
out vec4 v_col;
void main() {
    vec2 world_pos = a_pos + u_layer_offset;
    gl_Position = u_mvp * vec4(world_pos, 0.0, 1.0);
    if (u_outline_pass == 1) {
        // One framebuffer pixel is 2/viewport in NDC, and NDC is clip space divided by w.
        gl_Position.xy += (u_outline_offset / max(u_viewport, vec2(1.0))) * 2.0 * gl_Position.w;
    }
    // Must always write gl_ClipDistance when GL_CLIP_DISTANCE0-3 are enabled.
    // Undefined values cause random arrowhead/patch culling on macOS Metal.
    gl_ClipDistance[0] = world_pos.x - u_window.x;
    gl_ClipDistance[1] = u_window.y - world_pos.x;
    gl_ClipDistance[2] = world_pos.y - u_window.z;
    gl_ClipDistance[3] = u_window.w - world_pos.y;
    v_col = u_use_vertex_color ? a_col : u_color;
    v_col.a *= u_alpha;
}
"""

PATCH_FS = r"""
#version 330 core
in vec4 v_col;
out vec4 FragColor;
// Outline / silhouette; see PATCH_VS. u_outline_alpha is separate from the colour's own
// alpha so a silhouette can stay opaque under a translucent fill; ctx.global_alpha is
// folded into it on the CPU, so a scene-wide fade still fades the silhouette.
uniform int u_outline_pass;
uniform vec4 u_outline_color;
uniform float u_outline_alpha;
void main() {
    if (u_outline_pass == 1) {
        FragColor = vec4(u_outline_color.rgb, u_outline_color.a * u_outline_alpha);
        return;
    }
    FragColor = v_col;
}
"""

# ==============================================================================
# PASS 2: DENSITY ACCUMULATION
# ==============================================================================


DENSITY_ACCUM_FS = (
    r"""
#version 330 core
in vec4 v_col;
layout(location=0) out vec4 FragValue;

uniform int u_density_weighted;
"""
    + DENSITY_WEIGHT_TO_RGBA
    + r"""
void main() {
    float w = (u_density_weighted == 1) ? v_col.a : 1.0;
    FragValue = density_sample(v_col.rgb, w);
}
"""
)

DENSITY_POINTS_VS = r"""
#version 330 core
layout(location=0) in vec2 a_pos;
layout(location=1) in vec4 a_col;
uniform mat4  u_mvp;
uniform float u_size;
uniform float u_alpha;
uniform vec2  u_layer_offset;
uniform vec4  u_window;   // (xmin, xmax, ymin, ymax) world space
out float v_alpha;
// The marker's own colour, carried so the accumulation can average it (see
// DENSITY_WEIGHT_TO_RGBA). Only the alpha used to make it this far, back when the density
// buffer was a single channel and a scatter's colour could not survive the pass.
out vec3 v_rgb;
void main() {
    vec2 world_pos = a_pos + u_layer_offset;
    gl_Position = u_mvp * vec4(world_pos, 0.0, 1.0);
    gl_PointSize = u_size;
    // See SCATTER_VS: must always write gl_ClipDistance when GL_CLIP_DISTANCE0-3 are
    // enabled (DensityRenderer.begin_accum turns them on for this pass too).
    gl_ClipDistance[0] = world_pos.x - u_window.x;
    gl_ClipDistance[1] = u_window.y - world_pos.x;
    gl_ClipDistance[2] = world_pos.y - u_window.z;
    gl_ClipDistance[3] = u_window.w - world_pos.y;
    v_alpha = a_col.a * u_alpha;
    v_rgb = a_col.rgb;
}
"""

DENSITY_POINTS_FS = (
    r"""
#version 330 core
in float v_alpha;
in vec3 v_rgb;
layout(location=0) out vec4 FragValue;
"""
    + DENSITY_WEIGHT_TO_RGBA
    + r"""
void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (dot(coord, coord) > 0.25) discard;
    FragValue = density_sample(v_rgb, v_alpha);
}
"""
)

DENSITY_RESOLVE_FS = (
    r"""
#version 330 core
#define log10(x) (log(x) / log(10.0))

in vec2 v_uv;
out vec4 FragColor;

uniform sampler2D u_tex;
uniform float u_gain;
uniform float u_max_val;
uniform int u_is_log;
uniform int u_scheme;
uniform int u_invert;
uniform int u_light_to_color;
uniform vec2 u_uv_min;
uniform vec2 u_uv_max;
// 1 when the layers in this pass carry colours of their own and the image should be made
// of *those* rather than of the density colormap. See DensityRenderer.resolve.
uniform int u_tint;

"""
    + HEATMAP_FUNCS
    + r"""

void main() {
    // Discard fragments outside the plotting viewport bounds to keep margins clean
    if (v_uv.x < u_uv_min.x || v_uv.x > u_uv_max.x || v_uv.y < u_uv_min.y || v_uv.y > u_uv_max.y) {
        discard;
    }

    // .r is the accumulated weight -- the scalar this buffer used to hold on its own, so
    // every line of normalisation below is untouched. .gba is the weight-premultiplied
    // colour sum that rides alongside it (see DENSITY_WEIGHT_TO_RGBA).
    vec4 acc = texture(u_tex, v_uv);
    float val = acc.r;
    if (val <= 0.0) {
        discard;
    }

    float norm = 0.0;
    if (u_is_log == 1) {
        norm = log10(1.0 + val * u_gain) / max(log10(1.0 + u_max_val * u_gain), 1e-6);
    } else {
        // Linear normalization scaled by gain so gain still acts as sensitivity threshold
        norm = (val * u_gain) / max(u_max_val, 1e-6);
    }
    norm = clamp(norm, 0.0, 1.0);

    if (u_tint == 1) {
        // Colour comes from the data, intensity from the density: the pixel is painted in
        // the weight-averaged colour of whatever landed on it, and how strongly it is
        // painted is the same normalised density the colormap would have read. Output is
        // premultiplied so the resolve can be composited over the background instead of
        // replacing it, which is what makes a sparse region fade out rather than sit on a
        // flat slab of the ramp's dark end.
        //
        // The ramp switches (u_scheme, u_invert, u_light_to_color) are deliberately not
        // consulted here. They shape a colormap, and in this mode there is no colormap to
        // shape -- honouring "invert" by making dense regions *fainter* would be a
        // different picture, not an inverted one.
        vec3 hue = acc.gba / max(acc.r, 1e-6);
        FragColor = vec4(hue * norm, norm);
        return;
    }

    if (u_invert == 1) {
        if (u_light_to_color == 1) {
            // Prevent mapping all the way to 0.0 (which is black).
            // Zero density (norm = 0.0) maps to 1.0 (white/bright end of colormap).
            // Max density (norm = 1.0) maps to 0.25 (vibrant color, no black).
            norm = 1.0 - 0.75 * norm;
        } else {
            norm = 1.0 - norm;
        }
    } else {
        if (u_light_to_color == 1) {
            // Prevent mapping all the way to 1.0 (which is white/light/yellow).
            // Zero density (norm = 0.0) maps to 0.0 (black/dark end of colormap).
            // Max density (norm = 1.0) maps to 0.75 (vibrant color, no white/bright clipping).
            norm = 0.75 * norm;
        }
    }
    FragColor = vec4(apply_heatmap(u_scheme, norm), 1.0);
}
"""
)

# ==============================================================================
# PASS 3: PICKING & INTERACTION
# ==============================================================================

PICKING_LINES_VS = r"""
#version 330 core
layout(location=0) in float a_t;
layout(location=1) in vec2  a_ab;

uniform mat4  u_mvp;
uniform vec2  u_xrange;
uniform vec4  u_window;
uniform vec2  u_layer_offset;
uniform int   u_id_offset;

flat out int v_id;

void main() {
    float x = mix(u_xrange.x, u_xrange.y, a_t);
    float y = a_ab.x * x + a_ab.y;
    vec2  w = vec2(x, y) + u_layer_offset;

    gl_Position = u_mvp * vec4(w, 0.0, 1.0);

    gl_ClipDistance[0] =  w.x - u_window.x;
    gl_ClipDistance[1] =  u_window.y - w.x;
    gl_ClipDistance[2] =  w.y - u_window.z;
    gl_ClipDistance[3] =  u_window.w - w.y;

    v_id = u_id_offset + gl_InstanceID + 1;
}
"""

PICKING_LINES_FS = r"""
#version 330 core
flat in int v_id;
layout(location=0) out int FragID;
void main() {
    FragID = v_id;
}
"""

PICKING_SCATTER_VS = r"""
#version 330 core
layout(location=0) in vec2 a_pos;

uniform mat4  u_mvp;
uniform float u_size;
uniform int   u_id_offset;
uniform vec2  u_layer_offset;

flat out int v_id;

void main() {
    gl_Position = u_mvp * vec4(a_pos + u_layer_offset, 0.0, 1.0);
    gl_PointSize = u_size;
    v_id = u_id_offset + gl_VertexID + 1;
}
"""

PICKING_SCATTER_FS = r"""
#version 330 core
flat in int v_id;
layout(location=0) out int FragID;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (dot(coord, coord) > 0.25) discard;
    FragID = v_id;
}
"""

PICKING_STRIP_VS = r"""
#version 330 core
layout(location=0) in vec2 a_pos;

uniform mat4 u_mvp;
uniform int  u_id;
uniform vec2 u_layer_offset;

flat out int v_id;

void main() {
    gl_Position = u_mvp * vec4(a_pos + u_layer_offset, 0.0, 1.0);
    v_id = u_id;
}
"""

PICKING_STRIP_FS = r"""
#version 330 core
flat in int v_id;
layout(location=0) out int FragID;
void main() {
    FragID = v_id;
}
"""

PICKING_PATCH_VS = r"""
#version 330 core
layout(location=0) in vec2 a_pos;
uniform mat4 u_mvp;
uniform int  u_id;
uniform vec2 u_layer_offset;
flat out int v_id;
void main() {
    gl_Position = u_mvp * vec4(a_pos + u_layer_offset, 0.0, 1.0);
    v_id = u_id;
}
"""

PICKING_PATCH_FS = r"""
#version 330 core
flat in int v_id;
layout(location=0) out int FragID;
void main() {
    FragID = v_id;
}
"""

INTERACTION_FULLSCREEN_VS = r"""
#version 330 core
out vec2 v_uv;
const vec2 verts[4] = vec2[4](
    vec2(-1.0, -1.0), vec2( 1.0, -1.0),
    vec2(-1.0,  1.0), vec2( 1.0,  1.0)
);
void main() {
    gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
    v_uv = verts[gl_VertexID] * 0.5 + 0.5;
}
"""

CACHE_IMPOSTOR_FS = r"""
#version 330 core
in vec2 v_uv;
out vec4 FragColor;

uniform sampler2D u_tex;
uniform vec4 u_cache_window;
uniform vec4 u_cur_window;
// Axis margins as a fraction of the viewport: (left, right, bottom, top).
// Normalised rather than in pixels so this stays correct under any device pixel
// ratio -- the fraction is identical in logical and framebuffer space.
uniform vec4 u_margins;

void main() {
    // A world window does NOT span the viewport: `mvp()` insets it by the axis
    // margins, so world `u_cur_window.x` lands at screen x = margin_l, not 0. Both
    // this pass and the cached texture were drawn through that inset, so the
    // remap has to go screen -> inset -> world -> inset -> texture. Mapping v_uv
    // straight to world (as this shader used to) skips both insets; they cancel
    // only while cur == cache, and the cache is captured 3x padded so they never
    // are. The result was the plot visibly jumping on grab and sitting offset for
    // the whole drag.
    vec2 inset = vec2(1.0 - u_margins.x - u_margins.y, 1.0 - u_margins.z - u_margins.w);
    vec2 t = (v_uv - vec2(u_margins.x, u_margins.z)) / max(inset, vec2(1e-6));

    // Outside the plot area lies the axis gutter, which this pass does not own.
    if (any(lessThan(t, vec2(0.0))) || any(greaterThan(t, vec2(1.0)))) {
        FragColor = vec4(0.0);
        return;
    }

    float wx = mix(u_cur_window.x, u_cur_window.y, t.x);
    float wy = mix(u_cur_window.z, u_cur_window.w, t.y);

    float cx = (wx - u_cache_window.x) / (u_cache_window.y - u_cache_window.x);
    float cy = (wy - u_cache_window.z) / (u_cache_window.w - u_cache_window.z);

    // Nothing was captured beyond the cache window; smearing its edge pixels
    // across the gap would look like real data.
    if (cx < 0.0 || cx > 1.0 || cy < 0.0 || cy > 1.0) {
        FragColor = vec4(0.0);
        return;
    }

    vec2 uv = vec2(u_margins.x + cx * inset.x, u_margins.z + cy * inset.y);
    FragColor = texture(u_tex, uv);
}
"""

# ==============================================================================
# PASS 4: POST-PROCESSING
# ==============================================================================

POST_FX_VS = r"""
#version 330 core
out vec2 v_uv;

const vec2 verts[4] = vec2[4](
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0,  1.0),
    vec2( 1.0,  1.0)
);

void main() {
    vec2 p = verts[gl_VertexID];
    v_uv = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

GRADIENT_BG_FS = r"""
#version 330 core
in vec2 v_uv;
uniform vec3 u_top_color;
uniform vec3 u_bottom_color;
layout(location=0) out vec4 FragColor;

void main() {
    FragColor = vec4(mix(u_bottom_color, u_top_color, v_uv.y), 1.0);
}
"""

# Bright-pass with a soft knee.
#
# The previous version used a binary ``if (brightness > u_threshold)`` cut that
# passed the *whole* colour through unattenuated. Two consequences, both real:
#   * pixels popped in and out of the bloom source under motion (aliased edges);
#   * on a light background every pixel passed the test, so the entire canvas
#     became a bloom source.
# The knee curve below is the standard quadratic soft-knee: contribution ramps
# smoothly across ``[threshold - knee, threshold + knee]`` and, above the knee,
# degrades to *subtractive* thresholding (``b - threshold``) rather than passing
# the full colour. A 0.95 background at threshold 0.7 now contributes ~0.26 of
# its brightness instead of 100% of it.
#
# ``u_knee`` is the knee half-width relative to the threshold (0 = hard cut).
# Brightness is the max channel, not luminance: with a luminance metric a
# saturated blue line (luma 0.07) could never bloom at any sane threshold, which
# for a plotting library is a bug, not a feature.
BLOOM_EXTRACT_FS = r"""
#version 330 core
in vec2 v_uv;
uniform sampler2D u_tex;
uniform float     u_threshold;
uniform float     u_knee;
layout(location=0) out vec4 FragColor;

void main() {
    vec3 color = max(texture(u_tex, v_uv).rgb, vec3(0.0));
    float b = max(max(color.r, color.g), color.b);

    float knee = max(u_threshold * u_knee, 1e-4);
    float soft = clamp(b - u_threshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee);
    float contrib = max(soft, b - u_threshold) / max(b, 1e-4);

    FragColor = vec4(color * contrib, 1.0);
}
"""

# Dual-filter (Kawase) downsample — 5 bilinear taps, destination half-res.
#
# This replaces GAUSSIAN_BLUR_FS, whose ``u_radius`` multiplied the tap *offset*
# instead of widening the kernel: at radius 6 its 5 fixed weights sat at +/-6,
# 12, 18 and 24 half-res texels, i.e. four discrete ghost copies with visible
# ringing that got worse the further the radius slider was pushed. A dual filter
# widens by *iterating* down/up the mip chain, so every tap stays adjacent to its
# neighbours and the kernel is a true wide near-gaussian at any radius.
#
# ``u_halfpixel`` is half a texel of the DESTINATION target, pre-scaled by the
# fractional radius offset, and is supplied by the CPU (which knows both sizes).
BLOOM_DOWNSAMPLE_FS = r"""
#version 330 core
in vec2 v_uv;
uniform sampler2D u_tex;
uniform vec2      u_halfpixel;
layout(location=0) out vec4 FragColor;

void main() {
    vec2 hp = u_halfpixel;
    vec3 s = texture(u_tex, v_uv).rgb * 4.0;
    s += texture(u_tex, v_uv - hp).rgb;
    s += texture(u_tex, v_uv + hp).rgb;
    s += texture(u_tex, v_uv + vec2(hp.x, -hp.y)).rgb;
    s += texture(u_tex, v_uv - vec2(hp.x, -hp.y)).rgb;
    FragColor = vec4(s / 8.0, 1.0);
}
"""

# Dual-filter (Kawase) upsample — 8 bilinear taps in a tent, destination 2x res.
BLOOM_UPSAMPLE_FS = r"""
#version 330 core
in vec2 v_uv;
uniform sampler2D u_tex;
uniform vec2      u_halfpixel;
layout(location=0) out vec4 FragColor;

void main() {
    vec2 hp = u_halfpixel;
    vec3 s = texture(u_tex, v_uv + vec2(-hp.x * 2.0, 0.0)).rgb;
    s += texture(u_tex, v_uv + vec2(-hp.x, hp.y)).rgb * 2.0;
    s += texture(u_tex, v_uv + vec2(0.0, hp.y * 2.0)).rgb;
    s += texture(u_tex, v_uv + vec2(hp.x, hp.y)).rgb * 2.0;
    s += texture(u_tex, v_uv + vec2(hp.x * 2.0, 0.0)).rgb;
    s += texture(u_tex, v_uv + vec2(hp.x, -hp.y)).rgb * 2.0;
    s += texture(u_tex, v_uv + vec2(0.0, -hp.y * 2.0)).rgb;
    s += texture(u_tex, v_uv + vec2(-hp.x, -hp.y)).rgb * 2.0;
    FragColor = vec4(s / 12.0, 1.0);
}
"""

# Final composite.
#
# Tone mapping used to be welded to ``u_bloom_enabled``: Reinhard was applied if
# and only if bloom was on, so ticking "Enable Glow" mapped a white 1.0 pixel to
# 0.5 — the whole image darkened 2x before any glow appeared. Tone mapping is now
# an independent, explicit choice (u_tonemap: 0 none / 1 Reinhard / 2 ACES) that
# does not change when glow is toggled. The default is 0, which leaves the
# glow-off look bit-identical to today's and makes glow-on additive-only.
POST_COMPOSITE_FS = r"""
#version 330 core
in vec2 v_uv;
uniform sampler2D u_scene_tex;
uniform sampler2D u_bloom_tex;

uniform int   u_bloom_enabled;
uniform float u_bloom_intensity;
uniform int   u_tonemap;
uniform float u_grain;

layout(location=0) out vec4 FragColor;

// Value noise for the paper/chalk grain. Hashed off gl_FragCoord and NOT off a time
// uniform: this is the texture of the page, so it must stay nailed to the pixel grid.
// An animated grain would crawl while the plot sits still, and under reactive_rendering
// it would freeze the moment the loop went idle anyway.
float grain_hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec3 tonemap_reinhard(vec3 c) {
    return c / (c + vec3(1.0));
}

vec3 tonemap_aces(vec3 c) {
    const float a = 2.51;
    const float b = 0.03;
    const float c2 = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((c * (a * c + b)) / (c * (c2 * c + d) + e), 0.0, 1.0);
}

void main() {
    vec3 color = texture(u_scene_tex, v_uv).rgb;

    if (u_bloom_enabled == 1) {
        color += texture(u_bloom_tex, v_uv).rgb * u_bloom_intensity;
    }

    if (u_tonemap == 1) {
        color = tonemap_reinhard(color);
    } else if (u_tonemap == 2) {
        color = tonemap_aces(color);
    }

    // Grain last, after tone mapping: it is a property of the paper, not of the light,
    // so it must not be re-mapped. Applied on 2x2 pixel cells because a 1px hash reads
    // as sensor noise, while a 2px cell reads as tooth in the board. Zero by default.
    if (u_grain > 0.0) {
        float n = grain_hash(floor(gl_FragCoord.xy * 0.5)) - 0.5;
        color += vec3(n * u_grain);
    }

    FragColor = vec4(color, 1.0);
}
"""
