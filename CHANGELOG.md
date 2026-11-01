# Changelog

All notable changes to GLPlot are documented in this file.

The format follows Keep a Changelog, and this project uses semantic versioning
for public releases.

## [Unreleased]

### Added

- **Animation through MCP (`animate`).** A model can now make GIFs, APNGs, MP4s, WebMs and numbered PNGs of
  any figure, 2-D or 3-D. Properties follow expressions of time (`t`, `time`, `frame`) and of **variables
  that move in stages**: each stage starts where the last ended and ends exactly on its target, with easings
  (`linear`, `in`, `out`, `in_out`, `smooth`, `smoother`, `sine`). Bindings drive any argument of the call that
  drew an artist (`color`, `linewidth`, `reveal`, a function's `parameters.phase`...), axes properties (title,
  limits, the 3-D camera `azim`/`elev`) or a dataset column recomputed every frame. The figure is rebuilt per
  frame from a record of its construction, so every kind of artist animates the same way and the figure is
  never touched. The reply is a **contact sheet** plus measured facts and `issues` -- nothing moves, the axes
  rescale, the loop does not close, an abrupt jump, a huge GIF -- each with the fix. Every frame has the
  same size (text is measured over sample frames and one crop is used); video needs `ffmpeg`, GIF/APNG need
  nothing more than Pillow; frame count, pixels and render time are bounded and refused up front.
- **`simulate`**: integrate ordinary differential equations written as expressions (pendulum, orbits,
  predator-prey, Lorenz...) into a dataset with derived columns such as energy; RK45, DOP853, LSODA/Radau/BDF
  or fixed-step RK4. Checked against closed-form solutions.
- **`add_plot(reveal=, trail=, head=)`**: show the first fraction of a series, only its last N points, and a
  dot on the newest -- a curve that draws itself, a moving tail -- in 2-D and 3-D.
- **`update_artist`**: change what is already drawn (colour, width, which column, `parameters.a`...) without
  removing it; the artist keeps its id and stacking order, and a change that cannot be drawn leaves the figure
  exactly as it was. `describe_figure` now shows each artist's recipe.
- **Controls: `add_control` / `set_control` / `bind_property`.** Sliders, toggles, choices and buttons that
  drive properties; `render_figure(params=...)` / `save_figure(params=...)` try values without changing the
  figure; a stage of `animate` can sweep a slider.
- **`export_script`**: the Python that rebuilds a figure, from the record of its construction -- as a plain
  figure, as an **interactive** script with real `plt.slider` / `checkbox` / `dropdown` / `button` widgets, or
  as an **animation** (`FuncAnimation`, the timeline as numpy arrays). Datasets are rebuilt in the script
  (expressions become numpy, `simulate` becomes `solve_ivp`). The tests execute every script and compare its
  picture with the figure's pixel for pixel. The script is returned as text; the server writes no `.py` files.
- **A second y axis and hatching in MCP**: `add_plot(axis="right")` with `set_axes(y2label=, y2lim=, y2scale=)`,
  and `hatch=` on bar, barh, area and hist.
- Guides `mcp-animation` (with eight worked recipes the test suite runs) and `mcp-interactive`; prompts
  `animate_this` and `interactive_figure`; `server_info` reports the formats this machine can write.
- The easing and interpolation functions `lerp`, `smoothstep`, `smootherstep`, `ease_in`, `ease_out`,
  `ease_in_out`, `ease_sine`, `pingpong`, `ramp` and `pulse` joined the expression vocabulary (workstation too).
- **`pip install glplot` is enough for MCP.** A dependency-free stdio MCP server (`glplot.mcp.builtin`,
  standard library only) runs when the optional `mcp` package is absent, serving exactly the same
  tools, resources and prompts as the SDK server; `glplot[mcp]` still adds the official SDK
  (preferred when installed) and the HTTP transport. Tool schemas come from a shipped snapshot,
  `glplot/mcp/tools.json`, kept in step with the SDK server by a test. Verified against the official
  1.x and 2.x clients.
- **`glplot-mcp` commands.** `config` prints (or `--write`s, keeping a `.bak`) the registration for Claude
  Code, Claude Desktop, Cursor, VS Code and Codex with this interpreter's absolute path; `doctor`
  checks the install and runs a protocol self-test; `tools` lists the tools; `--builtin` / `--sdk`
  choose the server.
- **Documentation a model can read from the install** (`glplot.llm`, `python -m glplot.llm`): 13
  guide topics (overview, quickstart, choosing a plot, axes and styling, 3-D, animation, controls,
  performance, export, matplotlib compatibility, workstation, MCP, troubleshooting), a searchable index
  of every public function built by introspection (`search_api("second y axis")`), signatures and
  docstrings, 15 runnable examples, `llms.txt` and `llms-full.txt`. The guides are tested against the
  code: examples are executed, every function they name must exist, and tables are generated from the
  live registries.
- **`analyze_dataset` MCP tool: a model can analyse data, not just draw it.** Fifteen operations from the
  Math Lab -- `fit` (linear, polynomial, gaussian, lorentzian, exp_decay, power_law, sigmoid, multipeak or a
  custom expression such as `a*exp(-x/b)+c`, optionally outlier-robust, with parameter standard errors and
  R^2), `smooth`, `rolling`, `derivative`, `integrate`, `fft`, `peaks`, `filter`, `detrend`, `normalize`,
  `resample`, `histogram`, `correlate`, `autocorrelation`, `describe`. The result is kept: a same-length
  result becomes a column of the dataset (`v_fit`), a different-length one a new dataset (`run_fft`), and
  the reply carries a one-line result, the JSON report, notes and an `add_plot` hint. Options come from a
  declared table that validates them, generates the `mcp-analysis` guide and the tool description.
  New prompt `fit_and_plot`; `server_info` lists the operations; `Limits.max_analysis_rows` (1M) caps input.
- **HTTP transport hardening.** `--transport streamable-http` now takes a bearer token (`--token`, or the
  `GLPLOT_MCP_TOKEN` environment variable, constant-time compared and never logged) and **refuses to listen on
  a non-localhost address without one** (`--insecure` overrides, with a warning). `--allow-host NAME` lists the
  `Host` names clients may use and turns the SDK's DNS-rebinding check on for exactly those. Before, mcp 1.x
  answered `421 Invalid Host header` to anything but localhost, so a server bound to a network interface was
  unreachable, while mcp 2.x served it with no authentication at all. Tested with real servers and the
  official clients on both SDK generations.
- **`render_figure` reviews the figure.** The reply now carries `issues`: what a deterministic check of
  the rendered figure can *prove* is wrong -- an axis without a label, no title, an empty panel,
  overlapping tick labels (long category names), points a log axis cannot draw, data cut off by the
  limits, a crowded legend, constant or gappy series -- each with a severity (`warning`,
  `suggestion`, `info`), a code and the tool call that fixes it. An empty list means nothing was found,
  so a model can render, fix and render again until it is clean. The check costs nothing measurable
  and skips the one step that needs a second draw on figures of more than 50,000 points.
- **Five read-only MCP knowledge tools** -- `get_guide`, `search_api`, `get_api_doc`, `get_example`,
  `server_info` -- so clients that never read resources can still learn the library; 11 new resources
  (`glplot://guide/{topic}`, `glplot://api/{name}`, `glplot://examples/{name}`, `glplot://llms.txt`,
  `glplot://llms-full.txt`, `glplot://capabilities`, ...) and five new prompts (`quick_look`,
  `fix_my_figure`, `publication_figure`, `compare_series`, `python_script_for`). The connect-time
  instructions now point the model at them.
- `python -m glplot.llm agent-instructions` prints a note to paste into `CLAUDE.md` / `AGENTS.md`, and
  `skill --write DIR` writes a Claude Code skill, so a coding agent in a project that merely has GLPlot
  installed is pointed at the packaged documentation. Tool descriptions of `add_plot`, `add_function`,
  `set_axes` and `create_dataset` now include an example call.
- **Text and dates in data files, for the MCP tools.** A CSV column of names used to load as all-NaN with
  no hint that anything was lost. `load_dataset` now reads ISO dates as days since 1970-01-01 and other text
  columns as categories (labels kept, shown by `describe_dataset`). New `select_rows` tool filters
  (`where` expression, `equals` labels or numbers), sorts, thins and truncates a dataset into a new one;
  `analyze_dataset` gains `aggregate` (mean, sum, median, min, max, std, count per category or value), and
  `add_plot` takes bar/barh tick labels and pie slice names from the category column automatically. The
  workstation's paste/import path shares the same table parser.
- `logspace` and `geomspace` in the expression language (workstation and MCP, with the same size cap
  as `linspace`), so a log-spaced x axis does not need `10**linspace(...)`.
- `plt.stats()` is implemented (it always raised before): count and min/max/mean/std of the plotted
  points, in the view or in total.

### Changed

- MCP tool replies keep non-ASCII text as written (`T (°C)`, `µm`, accents) instead of `\uXXXX` escapes.
- `plt.step`, `mesh3d`, `annotate`, `arrow`, `axhline`, `axvline`, `axline`, `hlines`, `vlines`,
  `alpha`, `set_global_alpha`, `export`, `blending`, `density`, `hud`, `lod`, `set_lod`, `reset_view` and
  `stats` gained docstrings; every public function now has one (enforced by a test).

### Fixed

- **Axis limits now behave as in matplotlib.** `ylim(...)` / `xlim(...)` pin only the axis you set: before,
  one limit on an axes with no data yet pinned *both* axes, so the other stayed at its empty default
  (-1, 1); now it keeps following the data, including data added later (`Panel.pinned_x/pinned_y`; an
  explicit `autoscale()` takes both back). On a log, symlog, asinh or logit axis, `xlim`, `ylim`, `axis(...)`
  and `ax.set_xlim` now read and write **data units** (`xlim(10, 1000)` used to frame 10**10..10**1000, and
  a limit a log axis cannot show is ignored with a warning, as in matplotlib). `axhline`, `axvline`,
  `axhspan`, `axvspan`, `axline`, `figtext` and `text(transform=transAxes)` placed their points in log10
  units on a log axis, which the axis then dropped, so the line or label was invisible; fractions of a log
  axis are now taken along the drawn axis. A bare `axis()` returns `(xmin, xmax, ymin, ymax)` instead of
  re-autoscaling.
- `plt.subplots(r, c)` followed by `plt.subplot(r, c, i)` rebuilt the grid and emptied every panel; it now
  selects the panel, as in matplotlib (a different shape still rebuilds).
- MCP `set_axes(xscale="log", xlim=[...])` in one call lost the limits: the scale change re-fitted the view
  after they were set. Scales are applied first.
- `grouped_bar` laid groups out so that neighbouring groups overlapped with the default spacing; the bar
  width is now computed as in matplotlib (`group_spacing` and `bar_spacing` in bar widths).
- `plot(); xscale("log")` -- the usual order -- exported an empty plot: the export applied the
  camera's log10 limits to a matplotlib axis as if they were data. It now converts them (all real
  scales, per panel); this also made log axes work together with `twinx`.
- A translucent `fill_between` / `fill_betweenx` exported with a field of vertical seams (per-triangle
  strokes overlapping at the face's alpha); the default edge no longer has opacity of its own.
- `plot(x, y, "o")` and `errorbar(fmt="o")` drew a connecting line; a marker-only format string now draws
  markers only, as in matplotlib. `errorbar` bars take the data's colour instead of always black.
- `fill_between(x, 0, y)` raised on a scalar boundary (also `fill_betweenx`); scalars now broadcast.
- Violin bodies exported as a comb, and `set_aspect("equal")` / `axis("equal")` was ignored by
  `savefig`; both are honoured in the export.

## [0.2.0] - 2026-10-24

This release adds the Gallery panel, a native MCP server, interactive controls and real
twin/secondary axes, and closes a long list of matplotlib-compatibility gaps. It is the first
version whose notes cover everything since 0.1.3 (see "0.1.4 - 0.1.13" below).

### Added

- **Gallery panel** (`Mod+Shift+L`): the example gallery as 22 *recipes* you can point at your own
  data. Each card is a real render of the OpenGL window; pick one, assign columns (guessed from
  their names), choose a look (the recipe's own, your current style, or any preset) and it is
  drawn as one undoable step -- 2-D and 3-D, from line/scatter/bar/histogram through density maps,
  scattered fields and matrix heat maps to surfaces, wireframes, 3-D bars, vector fields and
  trajectories. A live preview of *your* columns renders in the background before anything
  touches the scene, and "Try with sample data" adds the recipe's own example table and draws it
  in one step. Recipes are pure data (`glplot.gui.recipes`) built on the same `layerops` the Data
  panel's Plot button uses, so the layers stay editable in the Scene and Style panels.
- **Real thumbnails everywhere a look is chosen.** The Style panel's presets are now renders of
  each preset (with a "Preview on" switch: lines, scatter, bars, heat map, 3-D surface), not a
  three-curve sketch. They are captured from the real GL window by `tools/build_gallery_assets.py`
  (needs a display; output is committed under `glplot/gui/assets/gallery`, ~1.4 MB) and fall back
  to the old drawn cards where there is no window.
- **Six new looks**, all held to the existing contrast/distinguishability invariants: `aurora`,
  `synthwave` (luminous, with a soft bloom), `sunset`, `ocean` (gradient pages), `vintage` (aged
  paper with grain and mid-century pigments) and `candy` (thick strokes, sticker-outlined markers).
- **Every preset now owns a colormap** (`PlotStyle.cmap`) for anything coloured by value, so a chalk
  page no longer gets a stock viridis heat map. Six matplotlib-registered ramps ship with GLPlot
  (`aurora`, `synthwave`, `sunset`, `lagoon`, `parchment`, `candy`; lightness moves one way only).
  The Style panel's "Use the style's colormap" re-maps existing images, colour-by-column scatters
  and surfaces (their data is untouched; undo restores the exact ramp); `gplt.plot_style(...,
  colormap=True)` does the same from code and makes it the default for what the figure draws next,
  below an explicit `cmap=` and `set_cmap()`.
- **MCP: the recipes.** `list_recipes` and `add_recipe` expose the gallery to a model: it names a
  recipe and a dataset (or `sample=true` for the recipe's own example table), the columns are
  guessed from their names and reported back, `columns={...}` overrides only what it names, and
  `look` picks the recipe's own look, the figure's current one (`keep`) or any style key. A recipe
  is one removable artist, combines with every other `add_*` tool, and stays out of the legend
  unless `label=` asks. Refusals say what to change (the slots, the available columns, a
  suggestion). The usage guide gains a generated table of every recipe.
- `tools/gl_capture.py`: read back what the real GL window draws (the way to see glow, tone-map and
  3-D shading, which the matplotlib export does not have).
- Native MCP (Model Context Protocol) server, `python -m glplot.mcp` / `glplot-mcp`, so an LLM
  can make plots and see them. 18 tools (`create_figure`, `create_dataset`, `load_dataset`,
  `add_plot` with 14 kinds, `add_function`, `add_field`, `add_annotation`, `set_axes`,
  `set_style`, `render_figure`, `save_figure`, ...), a generated `glplot://guide` resource,
  figure/dataset resources and two prompts. Renders return an image plus the resolved axis
  limits, labels and legend text. Runs headless over stdio or streamable HTTP, on `mcp` 1.x
  and 2.x. Optional extra: `pip install "glplot[mcp]"`; importing `glplot` never imports it.
  Safety: no code-execution tool, formulas go through the workstation's AST-validated
  evaluator with size-capped array constructors, and file access is off unless `--root` is given,
  then confined to it after symlink resolution, extension-limited, and never overwrites unless
  asked.
- `glplot.utils.preview.render_preview_bytes`: the headless export as in-memory bytes, with a
  hook to read the resolved axes from the same render.

- Native interactive controls (`glplot.controls`): `gplt.slider`, `range_slider`,
  `checkbox`, `dropdown`, `radio`, `textbox`, `number`, `color_picker`, `readout` and
  `button`, plus `@gplt.interact` to bind a function to controls (returned `(x, y)` is
  drawn and kept updated; a returned `dict` is shown as a results table). Callbacks are
  coalesced to one run per frame on the GL thread; errors are shown in the panel instead
  of stopping the render loop. Groups, presets, JSON save/load, reset to defaults.
- Controls panel in the workstation (`Mod+Shift+C`). A script with controls opens it
  automatically beside the plot; every button is a command-palette action and may have a
  keyboard shortcut.
- `glplot.widgets`: matplotlib-compatible `Slider`, `RangeSlider`, `Button`,
  `CheckButtons`, `RadioButtons` and `TextBox` on the native controls; `gplt.Slider`,
  `gplt.Button` and `gplt.Widget` now resolve to them.
- Artist data setters: `set_data`/`set_xdata`/`set_ydata`/`get_data` on line layers and
  `set_offsets`/`get_offsets`/`set_sizes` on scatter layers; a minimal `fig.canvas`
  (`draw_idle`, `draw`, `flush_events`).
- matplotlib `Figure` methods on the figure object: `tight_layout`, `subplots_adjust`,
  `suptitle`, `colorbar`, `legend`, `text`, `clf`, `gca`, `sca`, `show`, `subplots`, `axes`,
  `get_axes`, `number`, `get_size_inches`/`set_size_inches` (and the `figwidth`/`figheight`
  forms), `set_facecolor`/`get_facecolor`, `patch`, plus layout-only methods
  (`align_labels`, `autofmt_xdate`, `set_layout_engine`, ...) that run and warn once.
- matplotlib `Axes` methods: `set(...)`/`update`, `get_xlabel`/`get_ylabel`/`get_title`,
  `get_xscale`/`get_yscale`, `get_xticks`/`get_yticks` and the tick-label getters,
  `get_xbound`/`set_xbound` (and y), `get_legend_handles_labels`, `get_lines`, `has_data`,
  `get_position`/`set_position`, `set_xmargin`/`set_ymargin`, `autoscale_view`, `relim`,
  `set_facecolor`/`patch`, and cosmetic setters (`label_outer`, `set_axisbelow`, ...) that
  warn once. Two proxies for one panel now compare equal, so `ax in fig.axes` holds.

### Changed

- Release hygiene: `glplot.__version__`, `pyproject.toml`, `CITATION.cff` and the docs now state
  the same version, and `tests/test_version_consistency.py` fails when they drift (they had:
  0.1.13 on PyPI reported itself as 0.1.10). Contact addresses use `fhi.mpg.de`. `CONTRIBUTING`
  names the real minimum Python (3.11), and the README explains the `matplotlib<3.10` pin.
- Linting is one configuration for flake8 (`.flake8`) and ruff (`[tool.ruff]` in
  `pyproject.toml`): the CI lint job runs black, isort, flake8 and ruff, and `.pre-commit-config.yaml`
  runs the same four locally. `E741` (`l, r, b, t` extents) is off by design.
- Added `SECURITY.md` (how to report a vulnerability, and what the MCP server's file sandbox does
  and does not cover) and issue / pull-request templates.
- "Only Neon glows" is now "only the luminous looks glow": `neon`, `aurora` and `synthwave`. The
  tests that pinned the old rule name that family explicitly, cap the bloom, and require a new
  glowing preset to be added to it on purpose.
- `gplt.plot_style(name)` also applies the preset's colormap by default (`colormap=False` restores
  the old behaviour); `styles.apply_style` and `style_command` take `colormaps=` and default to off.

### Fixed

- The workstation theme crashed with `imgui-bundle` 1.93 (`Style.curve_tessellation_tol` was
  renamed `curve_tessellation_max_error`); both spellings work now.
- The headless export's 3-D line branch put a layer's raw `label` in the legend, ignoring the
  bookkeeping-label rule every other branch honours, so a 3-D wireframe/quiver layer named after
  its recipe (or an engine-assigned name) was listed as a legend entry nobody asked for.
- A colour-by-column scatter built through the GUI's layer functions (the Data panel's "Color by",
  gallery recipes) was drawn with **square** markers while `pyplot.scatter(c=...)` drew circles:
  `layerops._tag_colormapped` wrote `marker="s"` with a comment that nothing read it, and the
  scatter renderer reads it now. It no longer writes it.
- The `neon` preset's glow was invisible on thin lines (threshold above what a 1-2 px line reaches
  once the bloom is averaged down); it now blooms, with thicker default lines.
- `pcolormesh()` followed by `savefig()` died with `KeyError: 'X'`: the export had a branch for
  an older `pcolormesh` implementation that stored `X`/`Y`/`C` arrays, and the current one tags
  its patch with the same name but not those arrays. A layer without them now takes the generic
  per-vertex-coloured patch path, which draws it correctly.
- `fig.savefig(...)` crashed the interpreter (segfault) when no window was open, and took
  only `(filename, scale)`. It now goes through `gplt.savefig`: every matplotlib keyword,
  headless export, and it no longer changes the current figure. `gplt.export()` likewise.
- `plt.sca(axs[i])` was a silent no-op, so the next `plt.plot(...)` drew into the wrong panel.
- `ax.set_xlim(a, b)` alone froze the y axis at its `(-1, 1)` default instead of fitting
  it to the data (a `100*sin(x)` curve exported as an empty frame); `get_xlim`/`get_ylim`/
  `get_xticks` now report the fit that is owed before the first render, as matplotlib does.
- Exporting a scatter with one colour was ~20x slower than matplotlib (1.8 s vs 0.08 s at
  500k points) because an all-equal `(N, 4)` colour array sent Agg down its per-point path.
  Uniform colours are now collapsed to one. Marker positions follow matplotlib's own
  single-colour rasterization, so such exports differ from before by sub-pixel placement.
- Axis state was shared by every panel of a figure: `axs[0].set_xticks(...)`, a formatter,
  `set_yscale("log")` or `legend()` on one axes changed (or appeared on) all of them. Scale,
  tick positions/labels/formatter/locator/rotation and the legend are now per axes, live and
  in the export, and every panel of an export gets its own scale and ticks (non-primary panels
  used to ignore them).
- `plt.close("all")` closed nothing (only a figure object was understood), so every figure
  stayed alive: a closed 1000x1000 `imshow` kept ~150 MB and later exports slowed 2-3x. `close`
  now takes `"all"`, a number, a name or a figure, as in matplotlib.
- One panel's `set_xlim`/`set_ylim`/`set_yscale` switched off the initial fit of every other
  panel, so they opened empty at `(-1, 1)` in the window. The fit is per panel now.
- Reversed limits (`set_ylim(10, 0)`, `invert_yaxis()`, `imshow`'s default extent) produced a
  +-1e12 view. Axes can be inverted: ticks, labels, density, fractals, images and the mouse all
  follow, and `get_xlim()` reports `(hi, lo)`. `imshow(img)` now has matplotlib's default extent
  and inverted y, so overlays in pixel coordinates land on the right pixels.
- `set_xlim(left=0)`, `set_ylim(top=2)`, `xlim(None, 5)` raised `TypeError`; the missing end now
  stays where it is.
- `fig.suptitle(...)` on a multi-panel figure replaced the active panel's title; it is its own
  text above the panels in exports.
- Exported grid panels were flush against each other, so one panel's tick labels printed over its
  neighbour's; they are now spaced as with matplotlib's `wspace`/`hspace`. `inset_axes` and
  `set_position` rects are left where they were put.
- A bar with an `edgecolor` exported with a diagonal stroke across it (it was two triangles).
  A bar is one polygon in exports now, and its edge is drawn in the window too.
- `set_xticks([...])` followed by `set_xticklabels([...])` raised when the pinned ticks were not
  all in view; `-0` tick labels now print as `0`.
- `plot(x, Y)` with a 2-D `Y` (or `plot(Y)`, or matching 2-D `X`/`Y`) raised; it draws one line
  per column, and `label=[...]` labels them. `savefig` to a `BytesIO` raised.
- `setp(...)` silently ignored anything but a layer's own style fields (including `linewidth=`);
  it now calls the artist's `set_<prop>` and warns about what it cannot set.

### Added (compatibility, continued)

- Axis objects: `ax.xaxis`/`ax.yaxis` with `set_major_formatter` (a `Formatter`, a callable or a
  format string), `set_major_locator` (`MaxNLocator`, `MultipleLocator`, `FixedLocator`,
  `NullLocator`, `LinearLocator` map onto the live ticker; others apply to exports),
  `set_ticks`/`set_ticklabels`, `set_inverted`, `label`.
- Tick-label rotation (`xticks(rotation=...)`, `tick_params(rotation=...)`,
  `plt.setp(ax.get_xticklabels(), rotation=45)`), applied in exports.
- `title`/`xlabel`/`ylabel`/`xticks` accept matplotlib's text style keywords (`fontweight`,
  `rotation`, `va`, ...) and say once that they are ignored, instead of raising `TypeError`.
- `bar`/`hist` take `edgecolor`/`ec`, `linewidth`/`lw`, `facecolor`, `hatch` (warned) and
  `tick_label`; `text` honours `transform=ax.transAxes`, `ha`, `va` and `rotation` in exports.
- Artist handles: `line.set_color`/`set_linewidth`/`set_alpha`/`set_label`/`set_visible`,
  `ax.lines`/`collections`/`patches`/`texts`, `ax.add_line`, `ax.get_legend()`,
  `ax.set_prop_cycle`, `ax.set_visible`, `fig.delaxes` (leaves a blank cell), `fig.dpi`,
  `fig.supxlabel`/`supylabel`, `fig.canvas.get_width_height`.

### Added (twin and secondary axes, hatching)

- `twinx()` / `twiny()` are real: a second y (or x) scale in the same panel, with its own label,
  limits (`set_ylim`, reversed to invert), scale, ticks, formatter, locator, direction and tick
  colour, while `set_xlim`/`set_xlabel` stay the shared axis'. The first axes' range is fitted to
  its own data only. Saved figures use a real matplotlib twin axes drawn from the values you
  plotted, with one legend over both axes and room left for its labels. In the window the twin's
  values are mapped onto the first axes by a fixed relation, so zoom and pan move the two
  together, and its ticks (in its own units, in its colour) sit just inside the frame. Lines,
  markers, bars, fills and text can be drawn on a twin; images, contours and 3D layers cannot and
  stay on the first axes with a warning. `line.set_data`/`get_ydata` on a twin line speak in the
  twin's own units.
- `secondary_xaxis` / `secondary_yaxis` with `functions=(forward, inverse)`: ticks in converted
  units (degrees <-> radians, Hz <-> period, Celsius <-> Fahrenheit), placed where the inverse
  says they fall, so the relation need not be linear. Real matplotlib secondary axes in saved
  figures; inside-the-frame ticks in the window. `set_xlabel`, `set_xticks`, `set_color`,
  `tick_params`, `set_functions`.
- `hatch=` is drawn in saved figures for `bar`, `barh`, `hist`, `fill`, `fill_between`,
  `fill_betweenx`, `stackplot` and `pie` (a sequence is cycled over bars/bands/wedges). The window
  fills solid and says so once.
- `barh` takes `edgecolor`/`ec`, `linewidth`/`lw`, `tick_label` and `log`; `fill`, `fill_between`
  and `fill_betweenx` take `facecolor`, `edgecolor`, `linewidth`.

### Fixed (continued)

- `pie()` came out an ellipse: matplotlib sets an equal aspect and frames the view to
  `center +- 1.25` (the docstring claimed the opposite). It does now, and a saved pie has no
  frame or ticks.
- `axis("off")` / `set_axis_off()` were ignored by every 2D saved figure (frame, ticks and axis
  titles stayed); they are applied per axes, and the title stays, as in matplotlib.
- An explicit edge on `fill`/`fill_between` stroked every triangle of its triangulation, drawing
  the seams as stripes; it is now the outline of the shape.
- The documentation could not be built: `docs/requirements.txt` named a package that does not
  exist on PyPI (`sphinxcontrib-autodoc-pydantic`; it is `autodoc_pydantic`), and an autosummary
  entry resolved against the wrong module. `sphinx-build -b html docs/source` now succeeds.

## 0.1.4 - 0.1.13 (2026-07-19 to 2026-09-25)

These ten versions were published to PyPI without individual notes; this is a summary of what
they brought, reconstructed from the project history rather than release by release.

- **GUI workstation** (`python -m glplot`): live editing of data and layers with undo, a command
  palette, Data / Style / Scene / Selection / Timeline panels, and a restricted AST evaluator for
  column formulas. **Math Lab**: fits, smoothing, filters, PCA/UMAP, confidence intervals,
  trained models with a train/validation/test split, and background (async) computation with
  cooperative cancel.
- **Multi-panel figures** (`subplots`, `add_gridspec`, per-panel interaction, split/merge in the
  GUI), **3-D plotting** (surfaces, wireframes, 3-D bars, vector fields, camera, per-layer
  blending and depth control) and **animation** (`glplot.animation.FuncAnimation`, GIF/MP4
  export).
- **Real axis scales**: `log`, `symlog`, `asinh` and `logit`, for lines, scatters, bars and images;
  `hexbin` and `hist2d` as real colour-mapped patches; colour and size as data-driven encodings.
- **Style presets** (`plot_style`, `chalk`, `journal`, `neon`, ...) with contrast invariants, plus
  matplotlib-compatibility fixes (`savefig` keywords, `set_aspect`, spines, error bars,
  datetime and categorical axes, `GridSpec`).

## [0.1.3] - 2026-07-07

### Added

- Versioned changelog for release history.
- Minimum supported versions for runtime, development, and test dependencies.

### Changed

- Synchronized package version metadata across `pyproject.toml`, `glplot.__version__`,
  citation metadata, and publication documentation.

## [0.1.2] - 2026-07-06

### Added

- Publication readiness checklist and citation metadata.
- Gallery and benchmark documentation for scientific software review.

### Changed

- CI and lint configuration for multi-platform publication checks.

[Unreleased]: https://github.com/AkarisDimitry/GLPlot/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/AkarisDimitry/GLPlot/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/AkarisDimitry/GLPlot/releases/tag/v0.1.3
[0.1.2]: https://github.com/AkarisDimitry/GLPlot/releases/tag/v0.1.2
