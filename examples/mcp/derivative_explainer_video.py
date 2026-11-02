"""A complete 3Blue1Brown-style explainer video, made only through GLPlot's MCP server.

This is the sequence of tool calls an agent makes for "explain why the derivative of x^2 is 2x":
it talks to the server exactly as an MCP client does (the official ``mcp`` SDK over stdio), never
to GLPlot's Python API. Four scenes, one figure each, every one built step by step with stage
``effects`` and previewed before the video is cut together with ``compose_video``.

1. Title card: the question is written, a subtitle fades in.
2. The square: a square of side x grows by dx; the strips 2x*dx and the corner (dx)^2 appear,
   the equation is written, the corner is pointed at, dx shrinks towards 0, and the equation
   becomes the derivative.
3. The graph: the tangent to y = x^2 slides along the curve while its slope is read out live.
4. Recap.

Run it from the repository root (it needs ``pip install "glplot[mcp]"`` and ffmpeg for MP4)::

    python examples/mcp/derivative_explainer_video.py

It writes ``examples/mcp/results/derivative_explainer.mp4``, one contact sheet per scene, and
``derivative_explainer.gif``, the small preview the repository README shows (GitHub does not play
an MP4 inside a README).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

OUT = Path(__file__).resolve().parent / "results"
W, H = 1280, 720
BLUE, BLUE_E, YELLOW, TEAL, RED = "#58C4DD", "#236B8E", "#F7D96F", "#5CD0B3", "#FC6255"


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    server = StdioServerParameters(
        command=sys.executable,
        args=["-W", "ignore", "-m", "glplot.mcp", "--root", str(OUT)],
        env={**os.environ, "PYTHONWARNINGS": "ignore"},
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            async def call(tool: str, **arguments):
                result = await session.call_tool(tool, arguments)
                text = "".join(c.text for c in result.content if c.type == "text")
                if getattr(result, "isError", False) or getattr(result, "is_error", False):
                    raise RuntimeError(f"{tool} failed: {text}")
                images = [c for c in result.content if c.type == "image"]
                print(f"{tool:>15}: {text.splitlines()[0][:110]}")
                payload = text.split("\n", 1)[1] if "\n" in text else "{}"
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    data = {}
                return data, images

            async def preview(figure: str, name: str, **spec):
                """animate without output: the contact sheet is what a model looks at."""
                data, images = await call("animate", figure_id=figure, sheet_frames=12, **spec)
                (OUT / f"{name}_sheet.png").write_bytes(base64.b64decode(images[0].data))
                problems = [i for i in data.get("issues", []) if i["severity"] == "warning"]
                if problems:
                    print("   issues:", [p["code"] for p in problems])
                return data

            async def card(**axes):
                data, _ = await call("create_figure", width=W, height=H, style="3b1b")
                figure = data["figure_id"]
                await call("set_axes", figure_id=figure, hide_axes=True, legend=False, **axes)
                return figure

            async def text(figure: str, **arguments):
                data, _ = await call("add_annotation", figure_id=figure, kind="text", **arguments)
                return data["artist_id"]

            # ---- scene 1: the question ---------------------------------------------------
            title = await card(xlim=[0, 1], ylim=[0, 1])
            q = await text(title, x=0.5, y=0.55, coords="axes", align="center", fontsize=46,
                           text="Why is the derivative of $x^2$ equal to $2x$?")  # fmt: skip
            sub = await text(title, x=0.5, y=0.4, coords="axes", align="center", fontsize=30,
                             color=BLUE, text="a geometric explanation")  # fmt: skip
            await preview(title, "scene1", fps=24, stages=[
                {"name": "question", "duration": 2.0, "effects": {"write": q}},
                {"name": "subtitle", "duration": 1.0, "effects": {"fade_in": sub}},
                {"name": "pause", "duration": 1.2},
            ])  # fmt: skip

            # ---- scene 2: the square that grows ---------------------------------------------
            sq = await card(xlim=[-0.9, 7.4], ylim=[-0.7, 3.95], equal_aspect=True)

            async def area(**arguments):
                data, _ = await call("add_plot", figure_id=sq, kind="area", **arguments)
                return data["artist_id"]

            # opaque fills: a translucent area shows its triangulation seams in the export
            body = await area(x=[0, 2], y=[2, 2], color=BLUE_E)
            right = await area(x=[2, 2.6], y=[2, 2], color=YELLOW)
            top = await area(x=[0, 2], y=[2.6, 2.6], y2=[2, 2], color=YELLOW)
            corner = await area(x=[2, 2.6], y=[2.6, 2.6], y2=[2, 2], color=TEAL)
            side_b = await text(sq, x=1.0, y=-0.45, align="center", fontsize=30, text="$x$")
            side_l = await text(sq, x=-0.45, y=0.9, align="center", fontsize=30, text="$x$")
            dx_b = await text(sq, x=2.3, y=-0.45, align="center", fontsize=26, color=YELLOW,
                              text="$dx$")  # fmt: skip
            dx_l = await text(sq, x=-0.5, y=2.2, align="center", fontsize=26, color=YELLOW,
                              text="$dx$")  # fmt: skip
            grow = await text(sq, x=3.3, y=3.0, fontsize=30,
                              text=r"$(x+dx)^2 = x^2 + 2x\,dx + (dx)^2$")  # fmt: skip
            diff = await text(sq, x=3.3, y=2.0, fontsize=30, color=YELLOW,
                              text=r"$d(x^2) = 2x\,dx + (dx)^2$")  # fmt: skip
            result = await text(sq, x=3.3, y=2.0, fontsize=34, color=YELLOW,
                                text=r"$\frac{d(x^2)}{dx} = 2x + dx \;\to\; 2x$")  # fmt: skip
            note = await text(sq, x=3.3, y=1.1, fontsize=24, color=TEAL,
                              text="$(dx)^2$ vanishes as $dx \\to 0$")  # fmt: skip
            await preview(sq, "scene2", fps=24, variables={"dx": 0.6}, stages=[
                {"name": "square", "duration": 1.2,
                 "effects": {"fade_in": body, "write": [side_b, side_l]}},
                {"name": "grow", "duration": 1.4,
                 "effects": {"fade_in": [right, top, corner], "write": [dx_b, dx_l]}},
                {"name": "area", "duration": 1.6, "effects": {"write": grow}},
                {"name": "pause", "duration": 0.6},
                {"name": "difference", "duration": 1.4, "effects": {"write": diff}},
                {"name": "corner", "duration": 1.0, "effects": {"indicate": corner}},
                {"name": "note", "duration": 1.0, "effects": {"write": note}},
                {"name": "dx to 0", "duration": 2.4, "ease": "smoother",
                 "animate": {"dx": [0.6, 0.12]}},
                {"name": "divide", "duration": 1.2, "effects": {"transform": {diff: result}}},
                {"name": "result", "duration": 0.9, "effects": {"indicate": result}},
                {"name": "final pause", "duration": 1.0},
            ], bindings=[
                {"target": right, "property": "x", "value": ["2", "2 + dx"]},
                {"target": top, "property": "y", "value": ["2 + dx", "2 + dx"]},
                {"target": corner, "property": "x", "value": ["2", "2 + dx"]},
                {"target": corner, "property": "y", "value": ["2 + dx", "2 + dx"]},
                {"target": dx_b, "property": "x", "value": "2 + dx/2"},
                {"target": dx_l, "property": "y", "value": "2 + dx/2 - 0.1"},
            ])  # fmt: skip

            # ---- scene 3: the graph and its slope ----------------------------------------------
            data, _ = await call("create_figure", width=W, height=H, style="3b1b")
            gr = data["figure_id"]
            await call("set_axes", figure_id=gr, xlim=[-0.5, 3.3], ylim=[-1.2, 10.5],
                       xlabel="x", ylabel="y", grid=True, legend=False)  # fmt: skip
            data, _ = await call("add_function", figure_id=gr, expression="x**2", x_min=-0.6,
                                 x_max=3.2, linewidth=4)  # fmt: skip
            curve = data["artist_id"]
            head = await text(gr, x=0.5, y=1.05, coords="axes", align="center", fontsize=34,
                              text="The slope of $x^2$ at every point")  # fmt: skip
            data, _ = await call("add_plot", figure_id=gr, kind="scatter", x=[0.5], y=[0.25],
                                 size=16, color="#FFFFFF")  # fmt: skip
            dot = data["artist_id"]
            data, _ = await call("add_function", figure_id=gr,
                                 expression="2*x0*(x - x0) + x0**2", x_min=-0.5, x_max=3.3,
                                 parameters={"x0": 0.5}, color=YELLOW, linewidth=3)  # fmt: skip
            tangent = data["artist_id"]
            formula = await text(
                gr,
                x=0.17,
                y=0.8,
                coords="axes",
                fontsize=30,
                text=r"$f(x) = x^2 \quad\Rightarrow\quad f'(x_0) = 2x_0$",
            )
            readout = await text(gr, x=0.17, y=0.69, coords="axes", fontsize=28, color=YELLOW,
                                 text="slope = 1.00")  # fmt: skip
            await preview(gr, "scene3", fps=24, variables={"x0": 0.5, "m": "2*x0"}, stages=[
                {"name": "curve", "duration": 1.4, "effects": {"create": curve}},
                {"name": "title", "duration": 1.2, "effects": {"write": head}},
                {"name": "dot", "duration": 0.6, "effects": {"fade_in": dot}},
                {"name": "tangent", "duration": 1.2,
                 "effects": {"create": tangent, "write": [formula, readout]}},
                {"name": "slide", "duration": 3.0, "ease": "smoother",
                 "animate": {"x0": [0.5, 2.6]}},
                {"name": "point at it", "duration": 0.9, "effects": {"indicate": formula}},
                {"name": "back", "duration": 1.6, "ease": "smoother", "animate": {"x0": 1.2}},
                {"name": "pause", "duration": 0.8},
            ], bindings=[
                {"target": dot, "property": "x", "value": ["x0"]},
                {"target": dot, "property": "y", "value": ["x0**2"]},
                {"target": tangent, "property": "parameters.x0", "value": "x0"},
                {"target": readout, "property": "text",
                 "value": {"format": "slope = 2 * {x0:.2f} = {m:.2f}"}},
            ])  # fmt: skip

            # ---- scene 4: recap ------------------------------------------------------------
            recap = await card(xlim=[0, 1], ylim=[0, 1])
            law = await text(recap, x=0.5, y=0.55, coords="axes", align="center", fontsize=64,
                             text=r"$\frac{d}{dx}\,x^2 = 2x$")  # fmt: skip
            why = await text(recap, x=0.5, y=0.36, coords="axes", align="center", fontsize=28,
                             color=BLUE,
                             text="the area grows by two strips of width dx: 2x dx")  # fmt: skip
            await preview(recap, "scene4", fps=24, stages=[
                {"name": "rule", "duration": 1.2, "effects": {"write": law}},
                {"name": "why", "duration": 1.4, "effects": {"write": why}},
                {"name": "point at it", "duration": 0.9, "effects": {"indicate": law}},
                {"name": "pause", "duration": 1.0},
            ])  # fmt: skip

            # ---- the video ------------------------------------------------------------------
            data, images = await call(
                "compose_video",
                segments=[title, sq, gr, {"figure_id": recap, "hold_end": 1.5}],
                output="derivative_explainer.mp4",
                transition="fade",
                overwrite=True,
            )
            (OUT / "derivative_explainer_sheet.png").write_bytes(base64.b64decode(images[0].data))
            for scene in data["scenes"]:
                print(
                    f"   scene {scene['scene']} ({scene['figure_id']}): "
                    f"{scene['start_s']:.1f}-{scene['end_s']:.1f} s, {scene['source']}"
                )
            print("   file:", data["file"]["path"], f"{data['file']['bytes'] / 1e6:.1f} MB")
    write_preview(OUT / "derivative_explainer.mp4", OUT / "derivative_explainer.gif")


def write_preview(video: Path, gif: Path) -> None:
    """A 640 px, 12 fps GIF of the video for the README, with a palette made for it."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or not video.exists():
        return
    palette = (
        "fps=12,scale=640:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=64:stats_mode=diff[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    )
    subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-i", str(video), "-vf", palette, str(gif)], check=True
    )
    print("   preview:", gif, f"{gif.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    asyncio.run(main())
