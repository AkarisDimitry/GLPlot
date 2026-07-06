"""Generate the GLPlot OpenGL CPU/GPU rendering pipeline diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent / "results" / "glplot_pipeline_diagram.png"

BG = "#111126"
FG = "#f4f3fb"
MUTED = "#b8b4c8"
LINE = "#b8b5c8"

COLORS = {
    "input": ("#456a3d", "#c7d8be"),
    "cpu": ("#98672e", "#e1bd83"),
    "shader": ("#405aa0", "#c7d3ff"),
    "fixed": ("#2f8063", "#b9e7d6"),
    "density": ("#7e4d92", "#dfc1ee"),
    "resolve": ("#9b4b48", "#efc3be"),
    "export": ("#745fa6", "#d6c8ff"),
}

BOX_TEXT_SCALE = 1.58


def add_box(ax, x, y, w, h, title, subtitle, kind, title_size=8.5, subtitle_size=6.5):
    face, edge = COLORS[kind]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.006",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.62,
        title,
        ha="center",
        va="center",
        color=FG,
        fontsize=title_size * BOX_TEXT_SCALE,
        fontweight="bold",
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.240,
            subtitle,
            ha="center",
            va="center",
            color=edge,
            fontsize=subtitle_size * BOX_TEXT_SCALE,
            linespacing=1.15,
        )
    return patch


def arrow(ax, x1, y1, x2, y2, color=LINE, lw=1.2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            color=color,
            linewidth=lw,
            shrinkA=4,
            shrinkB=4,
        )
    )


def label(ax, x, y, text, color=MUTED, size=5.8, rotation=0, weight="normal"):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color=color,
        fontsize=size,
        rotation=rotation,
        fontstyle="italic" if size < 6.2 else "normal",
        fontweight=weight,
    )


def rail(ax, x, y0, y1, text, color, width=0.022):
    ax.add_patch(Rectangle((x, y0), width, y1 - y0, facecolor=color, alpha=0.55, linewidth=0))
    ax.text(
        x + width / 2,
        (y0 + y1) / 2,
        text,
        color=FG,
        fontsize=5.5,
        fontweight="bold",
        rotation=90,
        ha="center",
        va="center",
    )


def legend(ax):
    x, y = 0.465, 0.045
    ax.text(x, y + 0.115, "Legend", color=FG, fontsize=6.7, fontweight="bold")
    items = [
        ("input", "Input / output data"),
        ("cpu", "CPU setup / transfer"),
        ("shader", "GPU programmable shader"),
        ("fixed", "GPU fixed-function stage"),
        ("density", "GPU density accumulation"),
        ("resolve", "GPU resolve / SSAO pass"),
        ("export", "Export / screen"),
    ]
    for i, (kind, text) in enumerate(items):
        yy = y + 0.09 - i * 0.018
        face, edge = COLORS[kind]
        ax.add_patch(
            Rectangle((x, yy), 0.028, 0.010, facecolor=face, edgecolor=edge, linewidth=0.7)
        )
        ax.text(x + 0.037, yy + 0.005, text, color=MUTED, fontsize=5.0, va="center")


def pipeline_2d(ax, x0):
    w = 0.34
    cx = x0 + w / 2
    ys = [0.845, 0.775, 0.705, 0.635, 0.565, 0.495]
    h = 0.038

    ax.text(
        cx,
        0.905,
        "a)  GLPlot 2D Rendering Pipeline",
        ha="center",
        color=FG,
        fontsize=13,
        fontweight="bold",
    )
    rail(ax, x0 - 0.05, 0.66, 0.77, "CPU", COLORS["cpu"][0])
    rail(ax, x0 - 0.05, 0.425, 0.615, "GPU", COLORS["fixed"][0])
    rail(ax, x0 - 0.05, 0.235, 0.27, "CPU", COLORS["cpu"][0])

    add_box(
        ax,
        x0,
        ys[0],
        w,
        h,
        "INPUT: line data",
        "(x, y), or analytical y = a*x + b",
        "input",
        10.3,
        7.6,
    )
    add_box(
        ax,
        x0,
        ys[1],
        w,
        h,
        "CPU: VBO staging",
        "np.column_stack / astype(float32)\ncontiguous RAM for OpenGL",
        "cpu",
        10.3,
        7.1,
    )
    add_box(ax, x0, ys[2], w, h, "glBufferData", "CPU -> GPU one-time transfer", "cpu", 10.3, 7.4)
    add_box(
        ax,
        x0,
        ys[3],
        w,
        h,
        "Vertex Shader  [GPU]",
        "WIDE_SEGMENT_VS / LINE_VS",
        "shader",
        10.3,
        7.4,
    )
    add_box(
        ax,
        x0,
        ys[4],
        w,
        h,
        "Viewport Transform  [GPU]",
        "clip-space -> window pixels",
        "fixed",
        10.0,
        7.4,
    )
    add_box(
        ax, x0, ys[5], w, h, "Rasterizer  [GPU]", "fixed-function AA quad fill", "fixed", 10.3, 7.4
    )

    for y1, y2 in zip(ys, ys[1:]):
        arrow(ax, cx, y1, cx, y2 + h)

    left_x = x0 - 0.055
    exact_x = x0 + 0.062
    dens_x = x0 + 0.278
    branch_w = 0.155
    y_frag = 0.425
    y_blend = 0.345
    y_fb = 0.275
    y_colormap = 0.200

    ax.plot([cx, cx], [ys[5] - 0.006, y_frag + h + 0.012], color=LINE, lw=1.0, alpha=0.7)
    arrow(ax, cx - 0.085, ys[5], exact_x, y_frag + h)
    arrow(ax, cx + 0.085, ys[5], dens_x, y_frag + h)
    label(
        ax, exact_x, ys[5] + 0.010, "EXACT COLOR PATH", COLORS["shader"][1], size=7.4, weight="bold"
    )
    label(ax, dens_x, ys[5] + 0.010, "DENSITY PATH", COLORS["density"][1], size=7.4, weight="bold")

    add_box(
        ax,
        exact_x - branch_w / 2,
        y_frag,
        branch_w,
        h,
        "Fragment Shader",
        "WIDE_SEGMENT_FS",
        "shader",
        8.8,
        7.0,
    )
    add_box(
        ax,
        exact_x - branch_w / 2,
        y_blend,
        branch_w,
        h,
        "Alpha Blend",
        "Cout = a*Csrc +\n(1-a)*Cdst",
        "fixed",
        8.8,
        6.6,
    )
    add_box(
        ax,
        exact_x - branch_w / 2,
        y_fb,
        branch_w,
        h,
        "Framebuffer",
        "RGBA8 draw target",
        "fixed",
        8.8,
        7.0,
    )
    arrow(ax, exact_x, y_frag, exact_x, y_blend + h)
    arrow(ax, exact_x, y_blend, exact_x, y_fb + h)
    label(
        ax, exact_x, y_blend + h + 0.014, "alpha = 1 - smoothstep(r, r+w, d)", COLORS["cpu"][1], 6.3
    )
    label(ax, exact_x, y_fb + h + 0.014, "d = dist(pixel, line)", COLORS["cpu"][1], 6.0)

    rail(ax, left_x, y_fb, y_frag + h, "GPU", COLORS["shader"][0], 0.022)

    add_box(
        ax,
        dens_x - branch_w / 2,
        y_frag,
        branch_w,
        h,
        "Fragment Shader",
        "DENSITY_ACCUM_FS",
        "shader",
        8.8,
        7.0,
    )
    add_box(
        ax,
        dens_x - branch_w / 2,
        y_blend,
        branch_w,
        h,
        "Additive Blend",
        "D(p) += coverage\nGL_ONE -> R32F FBO",
        "density",
        8.8,
        6.6,
    )
    add_box(
        ax,
        dens_x - branch_w / 2,
        y_fb,
        branch_w,
        h,
        "Resolve Pass",
        "DENSITY_RESOLVE_FS",
        "resolve",
        8.8,
        7.0,
    )
    add_box(
        ax,
        dens_x - branch_w / 2,
        y_colormap,
        branch_w,
        h,
        "Colormap",
        "RGB = cmap(t)\nt in [0,1]",
        "density",
        8.8,
        6.6,
    )
    arrow(ax, dens_x, y_frag, dens_x, y_blend + h)
    arrow(ax, dens_x, y_blend, dens_x, y_fb + h)
    arrow(ax, dens_x, y_fb, dens_x, y_colormap + h)
    label(ax, dens_x, y_fb + h + 0.014, "Dmax = max_p D(p)", COLORS["cpu"][1], 6.0)
    label(
        ax, dens_x, y_colormap + h + 0.014, "t = log(1 + D) / log(1 + Dmax)", COLORS["cpu"][1], 6.0
    )
    rail(
        ax,
        dens_x + branch_w / 2 + 0.012,
        y_colormap,
        y_frag + h,
        "GPU",
        COLORS["density"][0],
        0.022,
    )

    hud_y = 0.145
    add_box(
        ax,
        x0,
        hud_y,
        w,
        h,
        "HUD / Axis Overlay  [GPU]",
        "ImGui + axis.py\nguides composited after data pass",
        "fixed",
        9.8,
        6.8,
    )
    arrow(ax, exact_x, y_fb, cx, hud_y + h)
    arrow(ax, dens_x, y_colormap, cx, hud_y + h)

    read_y = 0.075
    add_box(ax, x0, read_y, w, h, "glReadPixels -> CPU", "only for savefig/readback", "cpu")
    add_box(ax, x0, 0.015, w, h, "savefig() / screen", "PNG 800x600 + 2x supersampling", "export")
    arrow(ax, cx, hud_y, cx, read_y + h)
    arrow(ax, cx, read_y, cx, 0.015 + h)


def pipeline_3d(ax, x0):
    w = 0.33
    cx = x0 + w / 2
    h = 0.038
    ys = [0.845, 0.775, 0.705, 0.635, 0.565, 0.495, 0.425, 0.355, 0.285]

    ax.text(
        cx,
        0.905,
        "b)  GLPlot 3D Rendering Pipeline",
        ha="center",
        color=FG,
        fontsize=11,
        fontweight="bold",
    )
    rail(ax, x0 - 0.05, 0.66, 0.77, "CPU", COLORS["cpu"][0])
    rail(ax, x0 - 0.05, 0.18, 0.615, "GPU", COLORS["fixed"][0])
    rail(ax, x0 - 0.05, 0.055, 0.095, "CPU", COLORS["cpu"][0])

    add_box(
        ax, x0, ys[0], w, h, "INPUT: 3D geometry", "bars / surface / wireframe / scatter3d", "input"
    )
    add_box(
        ax, x0, ys[1], w, h, "CPU: geometry build", "np.stack((x,y,z), RGB/RGBA) -> float32", "cpu"
    )
    add_box(ax, x0, ys[2], w, h, "CPU: MVP matrix", "P * V * M via NumPy, 16 floats", "cpu")
    add_box(
        ax,
        x0,
        ys[3],
        w,
        h,
        "glBufferData + glUniform",
        "one-time VBO upload + 64-byte MVP/frame",
        "cpu",
    )
    add_box(ax, x0, ys[4], w, h, "Vertex Shader  [GPU]", "GEOMETRY3D_VS", "shader")
    add_box(
        ax,
        x0,
        ys[5],
        w,
        h,
        "Perspective Point Size  [GPU]",
        "gl_PointSize = size * w_ref / w_clip",
        "fixed",
    )
    add_box(
        ax,
        x0,
        ys[6],
        w,
        h,
        "Z-depth Normalize  [GPU]",
        "v_z_norm = (z - z_min) / (z_max - z_min)",
        "fixed",
    )
    add_box(
        ax,
        x0,
        ys[7],
        w,
        h,
        "Rasterizer + Depth Test  [GPU]",
        "GL_DEPTH_TEST fixed-function",
        "fixed",
    )
    add_box(ax, x0, ys[8], w, h, "Fragment Shader  [GPU]", "GEOMETRY3D_FS", "shader")

    for y1, y2 in zip(ys, ys[1:]):
        arrow(ax, cx, y1, cx, y2 + h)

    label(
        ax,
        cx,
        ys[1] + h + 0.012,
        "MVP = P(fov,aspect) · lookAt(eye,ctr,up) · M",
        COLORS["cpu"][1],
        5.2,
    )
    label(
        ax, cx, ys[4] + h + 0.012, "gl_Position = u_mvp · vec4(a_pos, 1.0)", COLORS["cpu"][1], 5.2
    )
    label(ax, cx, ys[5] + h + 0.012, "clamped point-size for zoom stability", COLORS["cpu"][1], 5.2)
    label(ax, cx, ys[7] + h + 0.012, "keep fragment if z_new < z_buf", COLORS["cpu"][1], 5.2)

    ssao_y = 0.215
    blend_y = 0.145
    hud_y = 0.095
    read_y = 0.035
    save_y = -0.025
    add_box(ax, x0, ssao_y, w, h, "SSAO / rim shading  [GPU]", "ao = (1 - cavity) * rim", "resolve")
    add_box(ax, x0, blend_y, w, h, "Alpha Blending  [GPU]", "src_alpha / 1-src_alpha", "fixed")
    add_box(ax, x0, hud_y, w, h, "HUD / Axis Guides  [GPU]", "ImGui + guide lines", "fixed")
    arrow(ax, cx, ys[8], cx, ssao_y + h)
    arrow(ax, cx, ssao_y, cx, blend_y + h)
    arrow(ax, cx, blend_y, cx, hud_y + h)
    label(
        ax, cx, ssao_y + h + 0.012, "ao = clamp(1 - cavity + rim, 0.18, 1.0)", COLORS["cpu"][1], 5.2
    )

    add_box(ax, x0, read_y, w, h, "glReadPixels -> CPU", "only for savefig/readback", "cpu")
    add_box(ax, x0, save_y, w, h, "savefig() / screen", "PNG 800x600 + 2x supersampling", "export")
    arrow(ax, cx, hud_y, cx, read_y + h)
    arrow(ax, cx, read_y, cx, save_y + h)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 21), dpi=180)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.04, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.955,
        "GLPlot — OpenGL Rendering Pipeline (CPU vs GPU correctly labelled)",
        ha="center",
        va="center",
        color=FG,
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.932,
        "CPU work is limited to necessary data staging, contiguous buffers, MVP math, and CPU<->GPU transfer. Heavy per-frame rendering remains on the GPU.",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=7.7,
    )

    pipeline_2d(ax, 0.09)
    pipeline_3d(ax, 0.64)
    legend(ax)

    fig.savefig(OUT, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
