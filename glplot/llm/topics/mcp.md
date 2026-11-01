# GLPlot as an MCP server

summary: Run and register the MCP server (works after a plain `pip install glplot`), what its tools do -- including animations (GIF/MP4), simulation, sliders and script export -- and its safety model.

GLPlot ships a [Model Context Protocol](https://modelcontextprotocol.io) server: a model creates figures, builds data,
draws, restyles and **sees every render** (an image plus a text summary of the axes), so it can check and fix its own
plot. It is headless -- no window, no GPU needed.

## Run it

```bash
pip install glplot                 # this alone is enough
glplot-mcp --root ~/data           # or: python -m glplot.mcp --root ~/data
```

* Without the optional `mcp` package GLPlot serves MCP with its **built-in stdio server** (standard library only), so an
  MCP client can launch it directly. `pip install "glplot[mcp]"` additionally installs the official SDK, which is then
  preferred, and enables `--transport streamable-http` (one long-running server, several clients).
* `--root DIR` (repeatable) lets the model read data from and save images into that directory. **Without `--root`
  the model has no file access at all**; it builds data with `create_dataset` and expressions instead.
* `glplot-mcp doctor` checks the install and runs a protocol self-test; `glplot-mcp tools` lists the tools.

## Register it in a client

`glplot-mcp config` prints the exact settings, with this interpreter's absolute path, for each client:

```bash
glplot-mcp config --client claude-code --root ~/data      # prints: claude mcp add glplot -- <python> -m glplot.mcp --root ...
glplot-mcp config --client claude-desktop --root ~/data   # JSON for claude_desktop_config.json
glplot-mcp config --client cursor | vscode | codex | generic
glplot-mcp config --client claude-desktop --write         # merge into the file, keeping a .bak copy
```

## What the model can do

| Group | Tools |
|---|---|
| Figures | `create_figure` (grid of panels, 2-D or 3-D, style), `describe_figure`, `close_figure` |
| Data | `create_dataset` (lists or **expressions** such as `sin(t)*exp(-t/5)`), `load_dataset` (numbers, plus text columns as categories and dates), `describe_dataset`, `select_rows` (filter/sort/thin), `delete_dataset` |
| Analysis | `analyze_dataset`: fit (linear, polynomial, gaussian, lorentzian, exp_decay, power_law, sigmoid, multipeak, custom), smooth, rolling, derivative, integrate, fft, peaks, filter, detrend, normalize, resample, histogram, correlate, autocorrelation, aggregate (mean/sum/... per category), describe |
| Drawing | `add_plot` (line, scatter, step, stem, bar, barh, area, errorbar, hist, box, violin, pie, hist2d, hexbin; `axis='right'` second y axis, `hatch`, `reveal`/`trail`/`head`), `add_function`, `add_field` (image, contour, surface...), `add_recipe`, `add_annotation`, `update_artist`, `remove_artist` |
| Axes and style | `set_axes` (title, labels, limits, scales, grid, legend, aspect, 3-D camera, right-hand axis), `set_style` |
| Dynamics | `simulate` (ODEs: pendulum, orbits, Lorenz...), `add_control` / `set_control` / `bind_property` (sliders, toggles, choices, buttons that drive properties), `animate` (time and stages drive properties, stage effects such as create/write/fade_in/indicate; GIF, APNG, MP4, WebM, PNG frames; contact sheet + measured issues), `compose_video` (scenes cut into one video) |
| Output | `render_figure` (image + resolved axes), `save_figure` (PNG, SVG, PDF, JPEG), `export_script` (the Python for the figure, an interactive version, or an animation) |
| Knowledge (read-only) | `get_guide`, `search_api`, `get_api_doc`, `get_example`, `server_info`, `list_recipes` |

The knowledge tools are how a model learns GLPlot instead of guessing: `get_guide()` lists these guides,
`search_api("heatmap")` finds Python functions, `get_example()` returns runnable scripts. Resources
(`glplot://guide`, `glplot://api/<name>`, `glplot://llms-full.txt`, ...) and prompts (`quick_look`, `fix_my_figure`,
`fit_and_plot`, `publication_figure`, `python_script_for`, ...) are also served.

## A typical session

```text
create_figure(title="Damped oscillator")                         -> fig1
add_function(fig1, "sin(a*x) * exp(-x/6)", 0, 20, parameters={"a": 2})
set_axes(fig1, xlabel="t (s)", ylabel="amplitude", legend=True)
render_figure(fig1)                                              -> image + {"xlim": [-1, 21], ...}
```

## Animation

`animate` rebuilds the figure for every frame with properties bound to time, in stages -- a curve drawing
itself, a parameter swept, a camera orbit, data recomputed -- and returns a contact sheet plus measured
problems (nothing moves, axes rescale, the loop does not close). Preview with no `output`, fix, then write
`.gif` / `.mp4`. `simulate` supplies trajectories; `export_script(kind='animation')` returns the Python.
Read `get_guide('mcp-animation')`.

Step by step like a 3Blue1Brown video: a stage's `effects` (`create`, `write`, `fade_in`, `indicate`,
`transform`, `fade_out`...) bring artists in and out, and `create_figure(style='3b1b')` gives manim's look
(black page, serif maths in Computer Modern, axes through the origin). A **complete video** is one figure per
scene, each animated and previewed, then `compose_video([...scenes...], output='explainer.mp4')` cuts them
together with fades (and an optional soundtrack). Read `get_guide('mcp-video')`, or use the prompt
`explainer_video`.

## Safety model

No tool runs arbitrary code; formulas go through an AST-validated evaluator (no attribute access, no builtins, no
imports) with size-capped array constructors. File access is opt-in, confined to `--root` after resolving symlinks,
limited to data and image extensions, and never overwrites unless asked. Inline data, dataset rows, figure counts and
image sizes are bounded; errors name the argument and what would be accepted. State is per session and in memory.
Over HTTP (needs the official SDK) it binds to localhost by default; on any other address it refuses to start
without a bearer token (`GLPLOT_MCP_TOKEN` or `--token`), and every request must send `Authorization: Bearer <token>`.

## If you are the model

Call `server_info()` first (versions, file roots, limits), then `get_guide()`. Prefer `add_recipe` for a figure that
already looks good, `add_function` for formulas, expressions over typed numbers, and always `render_figure` and look
before saying it is done. To hand the user a script rather than a figure, read `get_guide("quickstart")` and
`get_example(...)` first.
