"""Draw paper/mcp.png, the paper's figure of GLPlot's MCP server.

(a) The loop a language-model assistant runs: typed tool calls to ``glplot-mcp``, a headless
render, and an image returned with a review of provable problems. The review text is the
server's real reply to the figure built in ``_review()`` below, not a mock-up.
(b) Four frames of ``examples/mcp/results/derivative_explainer.mp4``, a four-scene explainer
video made entirely through MCP tool calls by ``examples/mcp/derivative_explainer_video.py``.

Run from the repository root (needs ffmpeg for the frames)::

    python paper/make_mcp_figure.py
"""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "examples" / "mcp" / "results" / "derivative_explainer.mp4"
OUT = ROOT / "paper" / "mcp.png"
FRAMES_AT = (3.5, 12.5, 26.0, 33.0)  # seconds: title, the square, the slope, the recap

INK = "#1f2328"
BLUE = "#0969da"
ORANGE = "#bc4c00"
GREY = "#57606a"


def _review() -> list:
    """The review the MCP server returns for a plot with no y label and no title."""
    from glplot.mcp.toolbox import Toolbox

    box = Toolbox()
    try:
        fig = box.create_figure(width=900, height=560).data["figure_id"]
        box.add_function(fig, expression="exp(-x/3)*sin(3*x)", x_min=0, x_max=10)
        box.set_axes(fig, xlabel="t (s)")
        return box.render_figure(fig).data["issues"]
    finally:
        box.close_all()


def _tool_count() -> int:
    data = json.loads((ROOT / "glplot" / "mcp" / "tools.json").read_text())
    return len(data["tools"] if isinstance(data, dict) else data)


def _frame(seconds: float) -> Image.Image:
    png = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(seconds), "-i", str(VIDEO), "-frames:v", "1",
         "-f", "image2pipe", "-vcodec", "png", "-"],
        check=True, capture_output=True,
    ).stdout  # fmt: skip
    return Image.open(io.BytesIO(png)).convert("RGB")


def _box(ax, x, y, w, h, title, body, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.8, edgecolor=color, facecolor="white",
        )
    )  # fmt: skip
    ax.text(x + w / 2, y + h - 0.06, title, ha="center", va="top", fontsize=15,
            fontweight="bold", color=color)  # fmt: skip
    ax.text(x + w / 2, y + h - 0.2, body, ha="center", va="top", fontsize=11.5, color=INK,
            linespacing=1.4)  # fmt: skip


def _arrow(ax, start, end, label, color, above=True):
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=1.9,
                        color=color, shrinkA=2, shrinkB=2)
    )  # fmt: skip
    mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + (0.035 if above else -0.035))
    ax.text(*mid, label, ha="center", va="bottom" if above else "top", fontsize=11, color=color)


def main() -> None:
    issues = _review()
    warning = next(i for i in issues if i["severity"] == "warning")
    tools = _tool_count()

    fig = plt.figure(figsize=(9, 5.2), dpi=250)
    top = fig.add_axes([0.0, 0.32, 1.0, 0.68])
    top.set_xlim(0, 1)
    top.set_ylim(0, 1)
    top.axis("off")
    top.text(0.005, 0.99, "(a)", fontsize=15, fontweight="bold", va="top", color=INK)

    client = "any MCP client\n(Claude, Codex,\nCursor, ...)"
    _box(top, 0.025, 0.38, 0.185, 0.48, "AI agent", client, INK)
    _box(top, 0.38, 0.25, 0.24, 0.61, "glplot-mcp",
         f"{tools} typed tools:\ncreate_figure, add_plot,\nrender_figure, animate, ...\n\n"
         "no code execution;\nfiles only in chosen folders", BLUE)  # fmt: skip
    renderer = "headless renderer,\nsame scene model\nas the window"
    _box(top, 0.79, 0.38, 0.185, 0.48, "GLPlot", renderer, INK)

    _arrow(top, (0.215, 0.74), (0.375, 0.74), "tool call", INK)
    _arrow(top, (0.625, 0.74), (0.785, 0.74), "scene ops", INK)
    _arrow(top, (0.785, 0.48), (0.625, 0.48), "render", ORANGE, above=False)
    _arrow(top, (0.375, 0.48), (0.215, 0.48), "image + review", ORANGE, above=False)

    review = (
        f"{warning['severity']}  {warning['code']}: {warning['message']}\nfix: {warning['fix']}"
    )
    top.text(0.5, 0.16, review, ha="center", va="top", fontsize=11.5, family="monospace",
             color=ORANGE, bbox=dict(boxstyle="round,pad=0.45", facecolor="#fff8f2",
                                     edgecolor=ORANGE, linewidth=1.1))  # fmt: skip

    fig.text(0.005, 0.305, "(b)", fontsize=15, fontweight="bold", va="top", color=INK)
    width, gap, left = 0.238, 0.008, 0.02
    for k, seconds in enumerate(FRAMES_AT):
        ax = fig.add_axes([left + k * (width + gap), 0.03, width, width * 9 / 16 * 9 / 5.2])
        ax.imshow(_frame(seconds))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GREY)
    fig.savefig(OUT, facecolor="white")
    print("wrote", OUT, Image.open(OUT).size)


if __name__ == "__main__":
    main()
