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
    "Cool"
]

def mix(c1, c2, x):
    return (
        c1[0] * (1.0 - x) + c2[0] * x,
        c1[1] * (1.0 - x) + c2[1] * x,
        c1[2] * (1.0 - x) + c2[2] * x,
    )

def eval_colormap(scheme_index: int, x: float) -> tuple[float, float, float]:
    """Evaluate colormap RGB color for a given normalized value [0.0, 1.0]."""
    x = max(0.0, min(1.0, x))
    if scheme_index == 1:    # Viridis
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
        if x < 0.11: return mix(c0, c1, x / 0.11)
        if x < 0.22: return mix(c1, c2, (x - 0.11) / 0.11)
        if x < 0.33: return mix(c2, c3, (x - 0.22) / 0.11)
        if x < 0.44: return mix(c3, c4, (x - 0.33) / 0.11)
        if x < 0.55: return mix(c4, c5, (x - 0.44) / 0.11)
        if x < 0.66: return mix(c5, c6, (x - 0.55) / 0.11)
        if x < 0.77: return mix(c6, c7, (x - 0.66) / 0.11)
        if x < 0.88: return mix(c7, c8, (x - 0.77) / 0.11)
        return mix(c8, c9, (x - 0.88) / 0.12)
    elif scheme_index == 2:  # Plasma
        c0 = (0.050, 0.030, 0.528)
        c1 = (0.291, 0.071, 0.718)
        c2 = (0.507, 0.104, 0.749)
        c3 = (0.692, 0.165, 0.564)
        c4 = (0.845, 0.277, 0.388)
        c5 = (0.954, 0.468, 0.199)
        c6 = (0.940, 0.975, 0.131)
        if x < 0.16: return mix(c0, c1, x / 0.16)
        if x < 0.32: return mix(c1, c2, (x - 0.16) / 0.16)
        if x < 0.48: return mix(c2, c3, (x - 0.32) / 0.16)
        if x < 0.64: return mix(c3, c4, (x - 0.48) / 0.16)
        if x < 0.80: return mix(c4, c5, (x - 0.64) / 0.16)
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
        if x < 0.125: return mix(c0, c1, x / 0.125)
        if x < 0.250: return mix(c1, c2, (x - 0.125) / 0.125)
        if x < 0.375: return mix(c2, c3, (x - 0.250) / 0.125)
        if x < 0.500: return mix(c3, c4, (x - 0.375) / 0.125)
        if x < 0.625: return mix(c4, c5, (x - 0.500) / 0.125)
        if x < 0.750: return mix(c5, c6, (x - 0.625) / 0.125)
        if x < 0.875: return mix(c6, c7, (x - 0.750) / 0.125)
        return mix(c7, c8, (x - 0.875) / 0.125)
    elif scheme_index == 4:  # Turbo
        g0 = (0.12, 0.01, 0.22)
        g1 = (0.13, 0.15, 0.48)
        g2 = (0.15, 0.65, 0.51)
        g3 = (0.85, 0.60, 0.12)
        g4 = (0.92, 0.11, 0.43)
        if x < 0.25: return mix(g0, g1, x / 0.25)
        if x < 0.50: return mix(g1, g2, (x - 0.25) / 0.25)
        if x < 0.75: return mix(g2, g3, (x - 0.50) / 0.25)
        return mix(g3, g4, (x - 0.75) / 0.25)
    elif scheme_index == 5:  # Ink Fire
        c0 = (1.0, 1.0, 1.0)
        c1 = (1.0, 0.9, 0.2)
        c2 = (1.0, 0.2, 0.1)
        c3 = (0.4, 0.0, 0.0)
        c4 = (0.0, 0.0, 0.0)
        if x < 0.25: return mix(c0, c1, x / 0.25)
        if x < 0.50: return mix(c1, c2, (x - 0.25) / 0.25)
        if x < 0.75: return mix(c2, c3, (x - 0.50) / 0.25)
        return mix(c3, c4, (x - 0.75) / 0.25)
    elif scheme_index == 6:  # Magma
        c0 = (0.001, 0.000, 0.031)
        c1 = (0.170, 0.047, 0.360)
        c2 = (0.447, 0.051, 0.439)
        c3 = (0.729, 0.160, 0.345)
        c4 = (0.960, 0.419, 0.231)
        c5 = (0.988, 0.768, 0.470)
        c6 = (0.988, 0.988, 0.823)
        if x < 0.16: return mix(c0, c1, x / 0.16)
        if x < 0.32: return mix(c1, c2, (x - 0.16) / 0.16)
        if x < 0.48: return mix(c2, c3, (x - 0.32) / 0.16)
        if x < 0.64: return mix(c3, c4, (x - 0.48) / 0.16)
        if x < 0.80: return mix(c4, c5, (x - 0.64) / 0.16)
        return mix(c5, c6, (x - 0.80) / 0.20)
    elif scheme_index == 7:  # Grayscale
        return (x, x, x)
    elif scheme_index == 8:  # Ocean
        c0 = (0.0, 0.0, 0.1)
        c1 = (0.0, 0.2, 0.6)
        c2 = (0.0, 0.8, 1.0)
        c3 = (1.0, 1.0, 1.0)
        if x < 0.33: return mix(c0, c1, x / 0.33)
        if x < 0.66: return mix(c1, c2, (x - 0.33) / 0.33)
        return mix(c2, c3, (x - 0.66) / 0.34)
    elif scheme_index == 9:  # Hot
        c0 = (0.0, 0.0, 0.0)
        c1 = (0.8, 0.0, 0.0)
        c2 = (1.0, 0.5, 0.0)
        c3 = (1.0, 1.0, 0.0)
        c4 = (1.0, 1.0, 1.0)
        if x < 0.25: return mix(c0, c1, x / 0.25)
        if x < 0.50: return mix(c1, c2, (x - 0.25) / 0.25)
        if x < 0.75: return mix(c2, c3, (x - 0.50) / 0.25)
        return mix(c3, c4, (x - 0.75) / 0.25)
    elif scheme_index == 10: # Cool
        return mix((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), x)
    else:                    # Classic / Default (0)
        def smoothstep(e0, e1, v):
            t = max(0.0, min(1.0, (v - e0) / (e1 - e0)))
            return t * t * (3.0 - 2.0 * t)
        return (
            smoothstep(0.0, 0.3, x),
            smoothstep(0.3, 0.6, x),
            smoothstep(0.6, 1.0, x)
        )

def get_colormap_min_color(scheme_index: int, invert: bool, light_to_color: bool = True) -> tuple[float, float, float]:
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
    p_ndc += n_px * ((u_width / u_viewport_size) * a_corner.y);

    gl_Position = vec4(p_ndc, 0.0, 1.0);
    v_side = a_corner.y * 2.0; // Map [-0.5, 0.5] to [-1, 1]

    v_col = (u_use_color == 1) ? a_col : vec4(0.0, 0.0, 0.0, 1.0);
    v_col.a *= u_alpha;
}
"""

EXACT_LINES_FS = r"""
#version 330 core
in vec4 v_col;
flat in float v_id_norm;
in float v_side;
out vec4 FragColor;

uniform int u_use_colormap;
uniform int u_scheme;
uniform int u_antialiasing;

""" + HEATMAP_FUNCS + r"""

void main() {
    float alpha = v_col.a;
    if (u_antialiasing == 1) {
        float d = abs(v_side);
        float w = fwidth(d);
        alpha *= 1.0 - smoothstep(1.0 - w, 1.0, d);
    }
    if (alpha <= 0.0) discard;

    vec4 color = v_col;
    if (u_use_colormap == 1) {
        color.rgb = apply_heatmap(u_scheme, v_id_norm);
    }
    FragColor = vec4(color.rgb, alpha);
}
"""

# --- SCATTERS ---

SCATTER_VS = r"""
#version 330 core
layout(location=0) in vec2 a_pos;
layout(location=1) in vec4 a_color;
uniform mat4  u_mvp;
uniform float u_size;
uniform float u_alpha;
uniform vec2  u_layer_offset;
out vec4 v_col;
void main() {
    gl_Position = u_mvp * vec4(a_pos + u_layer_offset, 0.0, 1.0);
    gl_PointSize = u_size;
    v_col = a_color;
    v_col.a *= u_alpha;
}
"""

SCATTER_FS = r"""
#version 330 core
in vec4 v_col;
out vec4 FragColor;

uniform int u_outline_enabled;
uniform vec4 u_outline_color;
uniform float u_outline_width_px;
uniform float u_point_size_px;

void main() {
    vec2 p = gl_PointCoord - vec2(0.5);
    float r = length(p) * 2.0;  // 0 center, ~1 edge

    if (r > 1.0) discard;

    float outline_frac = (u_point_size_px > 0.0)
        ? clamp(u_outline_width_px / u_point_size_px, 0.0, 0.49)
        : 0.0;

    float fill_radius = 1.0 - 2.0 * outline_frac;

    vec4 col = v_col;

    if (u_outline_enabled == 1 && r > fill_radius) {
        col.rgb = u_outline_color.rgb;
        col.a *= u_outline_color.a;
    }

    // Soft edge antialiasing
    float feather = fwidth(r) * 1.5;
    col.a *= 1.0 - smoothstep(1.0 - feather, 1.0, r);

    FragColor = col;
}
"""

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

uniform vec2  u_ndc_scale;             // ( 2/(r-l),  2/(t-b) )
uniform vec2  u_ndc_offset;            // (-(r+l)/(r-l), -(t+b)/(t-b))
uniform vec2  u_viewport_size;         // framebuffer size in pixels
uniform vec4  u_color;
uniform float u_alpha;
uniform float u_width;
uniform float u_id_norm;
uniform vec2  u_layer_offset;

out vec4 v_col;
flat out float v_id_norm;

void main() {
    vec2 ndc0 = (i_p0 + u_layer_offset) * u_ndc_scale + u_ndc_offset;
    vec2 ndc1 = (i_p1 + u_layer_offset) * u_ndc_scale + u_ndc_offset;

    // Convert NDC delta to pixels
    vec2 dir_px = (ndc1 - ndc0) * (0.5 * u_viewport_size);
    float len2 = dot(dir_px, dir_px);

    vec2 n_px = (len2 > 1e-12)
        ? normalize(vec2(-dir_px.y, dir_px.x))
        : vec2(0.0, 1.0);

    vec2 p_ndc = mix(ndc0, ndc1, a_corner.x);

    // u_width is full width; offset from centerline is ±0.5*u_width
    vec2 offset_ndc = n_px * (u_width / u_viewport_size) * a_corner.y;
    p_ndc += offset_ndc;

    gl_Position = vec4(p_ndc, 0.0, 1.0);

    v_col = u_color;
    v_col.a *= u_alpha;
    v_id_norm = u_id_norm;
}
"""

WIDE_SEGMENT_INSTANCED_FS = r"""
#version 330 core
in vec4 v_col;
flat in float v_id_norm;
out vec4 FragColor;

uniform int u_use_colormap;
uniform int u_scheme;

""" + HEATMAP_FUNCS + r"""

void main() {
    vec4 color = v_col;
    if (u_use_colormap == 1) {
        color.rgb = apply_heatmap(u_scheme, v_id_norm);
    }
    FragColor = color;
}
"""

WIDE_SEGMENT_DENSITY_FS = r"""
#version 330 core
in vec4 v_col;
layout(location=0) out float FragValue;

uniform int u_density_weighted;

void main() {
    FragValue = (u_density_weighted == 1) ? v_col.a : 1.0;
}
"""

# --- PATCHES ---

PATCH_VS = r"""
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

PATCH_FS = r"""
#version 330 core
in vec4 v_col;
out vec4 FragColor;
void main() {
    FragColor = v_col;
}
"""

# ==============================================================================
# PASS 2: DENSITY ACCUMULATION
# ==============================================================================

DENSITY_ACCUM_FS = r"""
#version 330 core
in vec4 v_col;
layout(location=0) out float FragValue;

uniform int u_density_weighted;

void main() {
    FragValue = (u_density_weighted == 1) ? v_col.a : 1.0;
}
"""

DENSITY_POINTS_VS = r"""
#version 330 core
layout(location=0) in vec2 a_pos;
layout(location=1) in vec4 a_col;
uniform mat4  u_mvp;
uniform float u_size;
uniform float u_alpha;
uniform vec2  u_layer_offset;
out float v_alpha;
void main() {
    gl_Position = u_mvp * vec4(a_pos + u_layer_offset, 0.0, 1.0);
    gl_PointSize = u_size;
    v_alpha = a_col.a * u_alpha;
}
"""

DENSITY_POINTS_FS = r"""
#version 330 core
in float v_alpha;
layout(location=0) out float FragValue;
void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (dot(coord, coord) > 0.25) discard;
    FragValue = v_alpha;
}
"""

DENSITY_RESOLVE_FS = r"""
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

""" + HEATMAP_FUNCS + r"""

void main() {
    // Discard fragments outside the plotting viewport bounds to keep margins clean
    if (v_uv.x < u_uv_min.x || v_uv.x > u_uv_max.x || v_uv.y < u_uv_min.y || v_uv.y > u_uv_max.y) {
        discard;
    }

    float val = texture(u_tex, v_uv).r;
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

void main() {
    float wx = mix(u_cur_window.x, u_cur_window.y, v_uv.x);
    float wy = mix(u_cur_window.z, u_cur_window.w, v_uv.y);

    float cache_u = (wx - u_cache_window.x) / (u_cache_window.y - u_cache_window.x);
    float cache_v = (wy - u_cache_window.z) / (u_cache_window.w - u_cache_window.z);

    cache_u = clamp(cache_u, 0.0, 1.0);
    cache_v = clamp(cache_v, 0.0, 1.0);
    FragColor = texture(u_tex, vec2(cache_u, cache_v));
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

BLOOM_EXTRACT_FS = r"""
#version 330 core
in vec2 v_uv;
uniform sampler2D u_tex;
uniform float     u_threshold;
layout(location=0) out vec4 FragColor;

void main() {
    vec4 color = texture(u_tex, v_uv);
    float brightness = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    if (brightness > u_threshold) {
        FragColor = color;
    } else {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}
"""

GAUSSIAN_BLUR_FS = r"""
#version 330 core
in vec2 v_uv;
uniform sampler2D u_tex;
uniform int       u_horizontal;
uniform float     u_radius;
layout(location=0) out vec4 FragColor;

void main() {
    float weight[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    vec2 tex_offset = 1.0 / textureSize(u_tex, 0);
    vec3 result = texture(u_tex, v_uv).rgb * weight[0];
    
    if(u_horizontal == 1) {
        for(int i = 1; i < 5; ++i) {
            result += texture(u_tex, v_uv + vec2(tex_offset.x * float(i) * u_radius, 0.0)).rgb * weight[i];
            result += texture(u_tex, v_uv - vec2(tex_offset.x * float(i) * u_radius, 0.0)).rgb * weight[i];
        }
    } else {
        for(int i = 1; i < 5; ++i) {
            result += texture(u_tex, v_uv + vec2(0.0, tex_offset.y * float(i) * u_radius)).rgb * weight[i];
            result += texture(u_tex, v_uv - vec2(0.0, tex_offset.y * float(i) * u_radius)).rgb * weight[i];
        }
    }
    FragColor = vec4(result, 1.0);
}
"""

POST_COMPOSITE_FS = r"""
#version 330 core
in vec2 v_uv;
uniform sampler2D u_scene_tex;
uniform sampler2D u_bloom_tex;

uniform int   u_bloom_enabled;
uniform float u_bloom_intensity;

layout(location=0) out vec4 FragColor;

void main() {
    vec3 hdr_scene = texture(u_scene_tex, v_uv).rgb;
    vec3 bloom = vec3(0.0);

    if (u_bloom_enabled == 1) {
        bloom = texture(u_bloom_tex, v_uv).rgb * u_bloom_intensity;
    }

    vec3 color = hdr_scene + bloom;

    // HDR Tone mapping if bloom is present
    if (u_bloom_enabled == 1) {
        color = color / (color + vec3(1.0));
    }

    FragColor = vec4(color, 1.0);
}
"""
