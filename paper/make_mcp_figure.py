"""Draw paper/mcp.png, the paper's figure of GLPlot's MCP server, in the style of Figure 1.

Left: the loop an AI agent runs for one figure -- a request, a typed tool call over MCP, the
server's checks, GLPlot's headless render, and the image returned with a review of provable
problems, which the agent acts on. Right: what the same session can deliver next, and the
knowledge tools the agent reads instead of guessing the API. The review line in step 7 is the
server's real reply to the figure built in ``_review()``; the tool count is read from
``glplot/mcp/tools.json``.

Run from the repository root::

    python paper/make_mcp_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "mcp.png"

# Figure 1's palette: a pale fill, a stronger border, the step number in the border colour
KINDS = {
    "io": ("#eef6e8", "#6aa84f", "Input / output"),
    "agent": ("#f1e7f5", "#8e5ea2", "AI agent (LLM)"),
    "mcp": ("#fdf3e1", "#e69138", "MCP protocol"),
    "check": ("#e3f4f4", "#3aa6a6", "Safety checks"),
    "glplot": ("#e3eefa", "#5b8fd6", "GLPlot (headless)"),
    "review": ("#fbe3e3", "#d9534f", "Review / feedback"),
}
INK = "#222222"
FACES = ("Quicksand", "Poppins", "Avenir", "Futura", "DejaVu Sans")


def _weight(weight) -> int:
    return int(font_manager.weight_dict.get(weight, weight))


def _font() -> str:
    """The first face of FACES installed as an upright regular (Figure 1's look), not only as a
    thin or italic cut: matplotlib registers one face per font collection."""
    upright = {
        f.name
        for f in font_manager.fontManager.ttflist
        if f.style == "normal" and 350 <= _weight(f.weight) <= 500
    }
    return next(face for face in FACES if face in upright)


def _review() -> dict:
    """The warning the MCP server returns for a plot without a y label."""
    from glplot.mcp.toolbox import Toolbox

    box = Toolbox()
    try:
        fig = box.create_figure(width=900, height=560).data["figure_id"]
        box.add_function(fig, expression="exp(-x/3)*sin(3*x)", x_min=0, x_max=10)
        box.set_axes(fig, xlabel="t (s)")
        issues = box.render_figure(fig).data["issues"]
        return next(i for i in issues if i["severity"] == "warning")
    finally:
        box.close_all()


def _tool_count() -> int:
    data = json.loads((ROOT / "glplot" / "mcp" / "tools.json").read_text())
    return len(data["tools"] if isinstance(data, dict) else data)


def _step(ax, x, y, w, h, number, title, subtitle, kind):
    fill, edge, _ = KINDS[kind]
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
            facecolor=fill, edgecolor=edge, linewidth=1.4,
        )
    )  # fmt: skip
    ax.text(x + 0.012, y + h / 2, number, ha="left", va="center", fontsize=21, color=edge)
    cx = x + w / 2 + 0.018
    ax.text(cx, y + h * 0.68, title, ha="center", va="center", fontsize=12.5, color=INK)
    ax.text(cx, y + h * 0.3, subtitle, ha="center", va="center", fontsize=9.3, color=INK)


def _down(ax, x, y0, y1):
    ax.add_patch(
        FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=11, linewidth=1.3,
                        color=INK, shrinkA=0, shrinkB=0)
    )  # fmt: skip


def _bracket(ax, x, y0, y1, label, color):
    ax.plot([x + 0.012, x, x, x + 0.012], [y1, y1, y0, y0], color=color, linewidth=1.3)
    ax.text(x - 0.008, (y0 + y1) / 2, label, ha="right", va="center", fontsize=11.5,
            color=color, rotation=90)  # fmt: skip


def main() -> None:
    plt.rcParams["font.family"] = _font()
    warning = _review()
    tools = _tool_count()

    fig = plt.figure(figsize=(9.2, 10.4 * 0.9), dpi=250)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 1)
    ax.axis("off")

    ax.text(0.31, 0.975, "One figure: the agent loop", ha="center", fontsize=14, color=INK)
    ax.text(0.79, 0.975, "Then: what it can deliver", ha="center", fontsize=14, color=INK)

    # left column -- the loop
    lx, lw, h, gap = 0.085, 0.45, 0.068, 0.022
    tops = [0.94 - k * (h + gap) for k in range(8)]
    fix = f"{warning['code']}: {warning['message']}"
    steps = [
        ("1", "User request", '"plot the decay and fit it"', "io"),
        ("2", "LLM agent plans", "reads get_guide, search_api", "agent"),
        ("3", "Typed tool call", "JSON-RPC over stdio / HTTP", "mcp"),
        ("4", "Server checks", "schema, limits, file roots; no code run", "check"),
        ("5", "Scene operations", "create_figure, add_plot, set_style ...", "glplot"),
        ("6", "Headless render", "same scene model as the GPU window", "glplot"),
        ("7", "Review of the render", fix, "review"),
        ("8", "Image + review returned", "the agent sees what it drew", "io"),
    ]
    for top, (number, title, subtitle, kind) in zip(tops, steps):
        _step(ax, lx, top - h, lw, h, number, title, subtitle, kind)
    for upper, lower in zip(tops, tops[1:]):
        _down(ax, lx + lw / 2, upper - h, lower)
    # 8 -> 2: the agent acts on the review and calls again
    loop_x = lx + lw + 0.022
    y_from, y_to = tops[7] - h / 2, tops[1] - h / 2
    ax.plot([lx + lw, loop_x, loop_x], [y_from, y_from, y_to], color=KINDS["agent"][1],
            linewidth=1.4)  # fmt: skip
    ax.add_patch(
        FancyArrowPatch((loop_x, y_to), (lx + lw, y_to), arrowstyle="-|>", mutation_scale=11,
                        linewidth=1.4, color=KINDS["agent"][1], shrinkA=0, shrinkB=0)
    )  # fmt: skip
    ax.text(
        loop_x + 0.008,
        (y_from + y_to) / 2,
        "acts on the review, calls again",
        ha="left",
        va="center",
        fontsize=10,
        color=KINDS["agent"][1],
        rotation=90,
    )

    _bracket(ax, 0.045, tops[1] - h, tops[1], "AGENT", KINDS["agent"][1])
    _bracket(ax, 0.045, tops[3] - h, tops[2], "MCP", KINDS["mcp"][1])
    _bracket(ax, 0.045, tops[6] - h, tops[4], "GLPlot", KINDS["glplot"][1])

    # right column -- deliverables and knowledge
    rx, rw = 0.6, 0.38
    right = [
        ("9", "animate", "stages, effects; contact sheet", "glplot"),
        ("10", "compose_video", "scenes cut into one MP4 / GIF", "glplot"),
        ("11", "export_script", "the Python that rebuilds it", "io"),
        ("12", "save_figure", "PNG, SVG, PDF in allowed folders", "io"),
    ]
    rtops = [0.94 - k * (h + gap) for k in range(len(right))]
    for top, (number, title, subtitle, kind) in zip(rtops, right):
        _step(ax, rx, top - h, rw, h, number, title, subtitle, kind)
    for upper, lower in zip(rtops, rtops[1:]):
        _down(ax, rx + rw / 2, upper - h, lower)

    kt = rtops[-1] - h - 0.06
    kh = 0.15
    fill, edge, _ = KINDS["agent"]
    ax.add_patch(
        FancyBboxPatch(
            (rx, kt - kh), rw, kh, boxstyle="round,pad=0,rounding_size=0.008",
            facecolor="white", edgecolor=edge, linewidth=1.3, linestyle=(0, (4, 3)),
        )
    )  # fmt: skip
    ax.text(rx + rw / 2, kt - 0.028, "The agent learns the library", ha="center", va="center",
            fontsize=12, color=edge)  # fmt: skip
    ax.text(
        rx + rw / 2,
        kt - 0.09,
        "get_guide   search_api\nget_api_doc   get_example\n"
        "the guides of python -m glplot.llm",
        ha="center", va="center", fontsize=9.6, color=INK, linespacing=1.45,
    )  # fmt: skip

    ax.text(
        rx + rw / 2,
        kt - kh - 0.055,
        f"{tools} tools  ·  works after pip install\n"
        "Claude Code, Claude Desktop, Cursor,\nVS Code, Codex (glplot-mcp config)",
        ha="center", va="center", fontsize=9.6, color=INK, linespacing=1.45,
    )  # fmt: skip

    # legend, as in Figure 1
    kinds = list(KINDS.values())
    for k, (fill, edge, label) in enumerate(kinds):
        col, row = k % 3, k // 3
        x, y = 0.04 + col * 0.33, 0.17 - row * 0.045
        ax.add_patch(
            FancyBboxPatch(
                (x, y - 0.014), 0.04, 0.028, boxstyle="round,pad=0,rounding_size=0.005",
                facecolor=fill, edgecolor=edge, linewidth=1.2,
            )
        )  # fmt: skip
        ax.text(x + 0.055, y, label, ha="left", va="center", fontsize=11, color=INK)

    fig.savefig(OUT, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
