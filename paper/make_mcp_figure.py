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
import tempfile
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
FALLBACK_FACES = ("Avenir Next", "Avenir", "Futura", "DejaVu Sans")


def _quicksand() -> list[str]:
    """Static cuts of Quicksand, Figure 1's typeface, at medium (text) and bold (titles).

    Quicksand usually ships as one variable font, and matplotlib only draws a variable font's
    default instance (Light), so the two weights are instantiated into a temporary folder.
    Returns an empty list when Quicksand is not installed."""
    sources = [f for f in font_manager.findSystemFonts() if "quicksand" in Path(f).name.lower()]
    if not sources:
        return []
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer

    folder = Path(tempfile.mkdtemp(prefix="glplot-fig-fonts-"))
    paths = []
    for weight in (500, 700):
        font = TTFont(sources[0])
        if "fvar" in font:
            font = instancer.instantiateVariableFont(font, {"wght": weight})
        font["OS/2"].usWeightClass = weight
        path = folder / f"Quicksand-{weight}.ttf"
        font.save(path)
        paths.append(str(path))
    return paths


def _use_font() -> None:
    """Quicksand when it is installed, else the nearest geometric sans that is."""
    paths = _quicksand()
    for path in paths:
        font_manager.fontManager.addfont(path)
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = "Quicksand" if paths else next(f for f in FALLBACK_FACES if f in installed)
    plt.rcParams["font.family"] = family
    plt.rcParams["font.weight"] = 500


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
            facecolor=fill, edgecolor=edge, linewidth=1.6,
        )
    )  # fmt: skip
    ax.text(x + 0.014, y + h / 2, number, ha="left", va="center", fontsize=27, color=edge,
            fontweight="bold")  # fmt: skip
    pad = 0.05 if len(number) < 2 else 0.068
    cx = x + pad + (w - pad) / 2
    ax.text(cx, y + h * 0.66, title, ha="center", va="center", fontsize=16, color=INK,
            fontweight="bold")  # fmt: skip
    ax.text(cx, y + h * 0.29, subtitle, ha="center", va="center", fontsize=11.8, color=INK)


def _down(ax, x, y0, y1):
    ax.add_patch(
        FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=13, linewidth=1.5,
                        color=INK, shrinkA=0, shrinkB=0)
    )  # fmt: skip


def _bracket(ax, x, y0, y1, label, color):
    ax.plot([x + 0.012, x, x, x + 0.012], [y1, y1, y0, y0], color=color, linewidth=1.6)
    ax.text(x - 0.008, (y0 + y1) / 2, label, ha="right", va="center", fontsize=14,
            color=color, rotation=90, fontweight="bold")  # fmt: skip


def main() -> None:
    _use_font()
    warning = _review()
    tools = _tool_count()

    fig = plt.figure(figsize=(9.2, 10.4 * 0.9), dpi=250)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 1)
    ax.axis("off")

    ax.text(0.31, 0.968, "One figure: the agent loop", ha="center", fontsize=17, color=INK,
            fontweight="bold")  # fmt: skip
    ax.text(0.785, 0.968, "Then: what it can deliver", ha="center", fontsize=17, color=INK,
            fontweight="bold")  # fmt: skip

    # left column -- the loop
    lx, lw, h, gap = 0.075, 0.45, 0.074, 0.017
    tops = [0.94 - k * (h + gap) for k in range(8)]
    fix = f"{warning['code']}: {warning['message']}"
    steps = [
        ("1", "User request", '"plot the decay and fit it"', "io"),
        ("2", "LLM agent plans", "reads get_guide, search_api", "agent"),
        ("3", "Typed tool call", "JSON-RPC over stdio / HTTP", "mcp"),
        ("4", "Server checks", "schema, limits, file roots; no code run", "check"),
        ("5", "Scene operations", "create_figure, add_plot, set_style", "glplot"),
        ("6", "Headless render", "same scene model as the GPU window", "glplot"),
        ("7", "Review of the render", fix, "review"),
        ("8", "Image + review returned", "the agent sees what it drew", "io"),
    ]
    for top, (number, title, subtitle, kind) in zip(tops, steps):
        _step(ax, lx, top - h, lw, h, number, title, subtitle, kind)
    for upper, lower in zip(tops, tops[1:]):
        _down(ax, lx + lw / 2, upper - h, lower)
    # 8 -> 2: the agent acts on the review and calls again
    loop_x = lx + lw + 0.016
    y_from, y_to = tops[7] - h / 2, tops[1] - h / 2
    ax.plot([lx + lw, loop_x, loop_x], [y_from, y_from, y_to], color=KINDS["agent"][1],
            linewidth=1.6)  # fmt: skip
    ax.add_patch(
        FancyArrowPatch((loop_x, y_to), (lx + lw, y_to), arrowstyle="-|>", mutation_scale=13,
                        linewidth=1.6, color=KINDS["agent"][1], shrinkA=0, shrinkB=0)
    )  # fmt: skip
    ax.text(
        loop_x + 0.006,
        (y_from + y_to) / 2,
        "acts on the review, calls again",
        ha="left",
        va="center",
        fontsize=12.5,
        color=KINDS["agent"][1],
        rotation=90,
        fontweight="bold",
    )

    _bracket(ax, 0.038, tops[1] - h, tops[1], "AGENT", KINDS["agent"][1])
    _bracket(ax, 0.038, tops[3] - h, tops[2], "MCP", KINDS["mcp"][1])
    _bracket(ax, 0.038, tops[6] - h, tops[4], "GLPlot", KINDS["glplot"][1])

    # right column -- deliverables and knowledge
    rx, rw = 0.6, 0.39
    right = [
        ("9", "animate", "stages, effects, contact sheet", "glplot"),
        ("10", "compose_video", "scenes cut into one MP4 / GIF", "glplot"),
        ("11", "export_script", "the Python that rebuilds it", "io"),
        ("12", "save_figure", "PNG, SVG, PDF to allowed folders", "io"),
    ]
    rtops = [0.94 - k * (h + gap) for k in range(len(right))]
    for top, (number, title, subtitle, kind) in zip(rtops, right):
        _step(ax, rx, top - h, rw, h, number, title, subtitle, kind)
    for upper, lower in zip(rtops, rtops[1:]):
        _down(ax, rx + rw / 2, upper - h, lower)

    kt = rtops[-1] - h - 0.045
    kh = 0.142
    fill, edge, _ = KINDS["agent"]
    ax.add_patch(
        FancyBboxPatch(
            (rx, kt - kh), rw, kh, boxstyle="round,pad=0,rounding_size=0.008",
            facecolor="white", edgecolor=edge, linewidth=1.3, linestyle=(0, (4, 3)),
        )
    )  # fmt: skip
    ax.text(rx + rw / 2, kt - 0.03, "The agent learns the library", ha="center", va="center",
            fontsize=15, color=edge, fontweight="bold")  # fmt: skip
    ax.text(
        rx + rw / 2,
        kt - 0.097,
        "get_guide   search_api\nget_api_doc   get_example\n"
        "the guides of python -m glplot.llm",
        ha="center", va="center", fontsize=12, color=INK, linespacing=1.5,
    )  # fmt: skip

    ax.text(
        rx + rw / 2,
        kt - kh - 0.065,
        f"{tools} tools  ·  works after pip install\n"
        "Claude Code, Claude Desktop, Cursor,\nVS Code, Codex (glplot-mcp config)",
        ha="center", va="center", fontsize=12, color=INK, linespacing=1.5,
    )  # fmt: skip

    # legend, as in Figure 1
    kinds = list(KINDS.values())
    for k, (fill, edge, label) in enumerate(kinds):
        col, row = k % 3, k // 3
        x, y = 0.03 + col * 0.33, 0.18 - row * 0.048
        ax.add_patch(
            FancyBboxPatch(
                (x, y - 0.014), 0.04, 0.028, boxstyle="round,pad=0,rounding_size=0.005",
                facecolor=fill, edgecolor=edge, linewidth=1.2,
            )
        )  # fmt: skip
        ax.text(x + 0.055, y, label, ha="left", va="center", fontsize=14, color=INK)

    fig.savefig(OUT, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
