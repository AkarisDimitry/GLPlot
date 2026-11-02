# Driving GLPlot through MCP

These scripts talk to GLPlot's MCP server exactly as an MCP client (Claude, an agent framework, an
IDE) does: through the official `mcp` SDK over stdio, never through GLPlot's Python API. They are the
sequence of tool calls a model makes.

| Script | What it makes |
|---|---|
| [derivative_explainer_video.py](derivative_explainer_video.py) | A complete step-by-step explainer video in the manim style, "why is the derivative of x^2 equal to 2x?": four scenes (title, the growing square, the tangent's slope, recap), each built step by step with stage `effects`, previewed, then cut together with `compose_video`. Writes [results/derivative_explainer.mp4](results/derivative_explainer.mp4), contact sheets and [results/derivative_explainer.gif](results/derivative_explainer.gif), the preview the main README shows. |

```bash
pip install "glplot[mcp]"      # the SDK client; MP4 also needs ffmpeg
python examples/mcp/derivative_explainer_video.py
```

From a model's side the same workflow is `get_guide('mcp-video')` or the prompt `explainer_video`.
