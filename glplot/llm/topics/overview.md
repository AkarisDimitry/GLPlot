# GLPlot in one page

summary: What GLPlot is, how a script uses it, and where to look next. Read this first.

GLPlot is a Python plotting library with a **matplotlib-like API that draws on the GPU** (OpenGL).
Millions of points, dense line families and large 3-D scenes stay interactive. If you know
`plt.plot()` you know most of it.

```python
import numpy as np
import glplot.pyplot as plt          # the same role as matplotlib.pyplot

x = np.linspace(0, 10, 200)
plt.figure("Demo", figsize=(8, 5))
plt.plot(x, np.sin(x), label="sin")
plt.xlabel("x"); plt.ylabel("y"); plt.legend()
plt.savefig("demo.png")              # headless export, no window needed
# plt.show()                         # or: open the interactive GPU window (blocks until closed)
```

## The one decision that matters: window or file

| You want | Call | Notes |
|---|---|---|
| An interactive window (pan, zoom, rotate, sliders) | `plt.show()` | Needs a display and an OpenGL 3.3+ context. **Blocks** until the window closes. |
| A file, in a script / CI / agent / server | `plt.savefig("a.png")` | Works with **no display**: it falls back to a matplotlib re-drawing of the scene. |

If you are an agent running a script, **use `savefig` and never `show()`** -- `show()` waits for a
person to close a window. Save, then look at the image, then fix what is wrong.

## What is different from matplotlib

* `import glplot.pyplot as plt` is the whole migration for most scripts; `fig, ax = plt.subplots()`
  and `ax.plot(...)` work too. See the `matplotlib-compat` topic for the exceptions.
* Animation is `glplot.animation` (a drop-in for `matplotlib.animation`), **not** `plt.FuncAnimation`.
* Output size is `figsize x 240` pixels by default (`savefig(scale=2.0)` at 120 dpi), with a floor of
  4 x 3 inches.
* Things that exist only because it is a GPU library: `plt.slider`/`plt.interact` controls, a
  workstation GUI (`python -m glplot`), density rendering for huge point clouds, 3-D scatter of
  millions of points, style presets (`plt.plot_style`).
* An option GLPlot cannot honour **warns once** (`MatplotlibCompatWarning`) instead of failing --
  read the warnings, they are the list of things that did not take effect.

## What ships in the package

* `glplot.pyplot` -- the plotting API (about 250 functions; search it with the `search_api` tool or
  `python -m glplot.llm search <words>`).
* `glplot.animation` -- `FuncAnimation`, `ArtistAnimation`, GIF/MP4/HTML writers.
* Native controls -- `plt.slider`, `plt.interact`, `plt.button` ... (topic `controls`).
* `python -m glplot` -- the workstation GUI (topic `workstation`).
* `python -m glplot.mcp` -- an MCP server so an LLM can build plots and *see* them (topic `mcp`).
* `python -m glplot.llm` -- this documentation, from the command line.

## Where to read next

{{topics}}
