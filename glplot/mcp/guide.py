"""The usage guide a model reads before it plots. Pure text, no MCP SDK.

Served as the ``glplot://guide`` resource and trimmed into the server's ``instructions``.
The expression vocabulary and the style list are *generated* from the code that enforces
them (:data:`glplot.gui.expressions.SAFE_NAMES`, :func:`glplot.pyplot.plot_styles`), so the
guide cannot promise a function the evaluator would reject.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .. import pyplot as gplt
from ..gui import recipes
from ..llm.guides import Topic
from . import analysis, cookbook, dynamics, motion
from .guide_text import ANIMATION_BODY, ANIMATION_INTRO, INTERACTIVE_BODY, VIDEO_BODY
from .state import DEFAULT_LIMITS, FileSandbox, Limits, expression_vocabulary
from .toolbox import ANNOTATION_KINDS, FIELD_KINDS

__all__ = ["build_guide", "build_instructions", "mcp_topics"]

#: A short, curated colormap list. matplotlib has ~170; a model needs the dozen that are good.
COLORMAPS = (
    "viridis, plasma, inferno, magma, cividis (perceptually uniform, colour-blind safe); "
    "turbo (rainbow-like but uniform); coolwarm, RdBu, seismic (diverging, for data centred "
    "on zero); gray, bone; Blues, Greens, Reds, YlOrRd (sequential); tab10, Set2 (categorical)"
)


def _slot_text(slot: recipes.Slot) -> str:
    """``x*`` -- plus the slot's label when the key alone would mislead (``x* (Slope)``)."""
    if slot.multi:
        return "every column"
    text = slot.key + ("*" if slot.required else "")
    first = slot.label.split()[0].lower().strip("()") if slot.label else ""
    return text if first == slot.key else f"{text} ({slot.label})"


def _recipe_table() -> str:
    """A markdown table of every recipe, generated from the registry so it cannot go stale."""
    rows = ["| recipe | draws | slots (* = required) | look |", "|---|---|---|---|"]
    for spec in recipes.RECIPES:
        slots = ", ".join(_slot_text(slot) for slot in spec.slots)
        rows.append(
            f"| `{spec.key}` | {spec.description} ({spec.dimension}) | {slots} | {spec.style} |"
        )
    return "\n".join(rows)


def _preset_count() -> int:
    from ..gui.styles import STYLE_KEYS

    return len(STYLE_KEYS)


def build_instructions(sandbox: Optional[FileSandbox] = None) -> str:
    """The text sent to the client at connect time: the loop, and where to learn the rest."""
    files = (
        "Files may be read from and saved to: " + ", ".join(str(r) for r in sandbox.roots) + "."
        if sandbox is not None and sandbox.enabled
        else "File access is disabled; build data with create_dataset instead."
    )
    return (
        "GLPlot draws plots headlessly and returns them as images, so you can look at your own "
        "output and iterate.\n"
        "Workflow: create_figure -> (create_dataset | load_dataset) -> add_plot / add_function "
        "/ add_field / add_recipe -> set_axes (title, labels, limits, legend) -> render_figure "
        "-> adjust -> save_figure. add_recipe draws a ready-made plot kind in a designed look "
        "(list_recipes shows them) and is the fastest way to a figure that already looks good.\n"
        "Analyse before you plot when the question is quantitative: analyze_dataset fits curves "
        "(with parameter errors and R^2), smooths, takes spectra, finds peaks, filters and more, "
        "and keeps the result as a new column or dataset you can add_plot.\n"
        "To make something MOVE, build the still figure first, pin its axes, then animate: "
        "properties follow time, in stages (get_guide('mcp-animation')); preview without an "
        "output file to get a contact sheet and a list of measured problems, then write a "
        "GIF/MP4. simulate solves differential equations (pendulum, orbits, Lorenz) into a "
        "dataset to animate; update_artist edits what is drawn; add_control + bind_property "
        "make a figure depend on sliders; export_script returns the Python behind a figure, an "
        "interactive version or an animation for the user to keep.\n"
        f"A look is one argument: create_figure(style=...) or set_style takes {_preset_count()} presets by use "
        "-- paper (journal, tufte), talks, dashboards (ggplot, material), explainer videos "
        "(manim, kurzgesagt), newsroom charts (economist, ft, fivethirtyeight), editor themes "
        "(nord, catppuccin, tokyonight...) -- by key or by name ('The Economist'); it sets the "
        "page, palette, faces and axes of the saved image, so leave color= unset and let it "
        "choose. When the user names a person's, a publication's or a theme's look, use it.\n"
        "A complete explainer video (manim style: create_figure(style='manim'), stage "
        "effects create/write/fade_in/indicate/transform/fade_out) is one figure per scene, each "
        "animated and previewed, then compose_video cuts them into one MP4 "
        "(get_guide('mcp-video')).\n"
        "Keep large data out of the conversation: load it from a file or generate it with an "
        "expression (e.g. 'sin(2*pi*t)*exp(-t)') instead of typing numbers. Always set a title "
        "and axis labels, and call render_figure to check the result before answering.\n"
        "Learn the library on demand instead of guessing: get_guide() lists the guide topics "
        "(mcp-* topics explain these tools; the others cover the Python API -- plotting, 3-D, "
        "animation, controls, styles, differences from matplotlib); search_api('heatmap') finds "
        "Python functions by what they do; get_api_doc(name) gives a signature and docstring; "
        "get_example() returns runnable scripts; server_info() reports versions, limits and file "
        "access. If the user wants a Python script rather than a figure, read those first: GLPlot "
        "is matplotlib-like but not identical (import glplot.pyplot as plt; unattended scripts "
        "call plt.savefig, never plt.show). " + files
    )


#: (topic id, heading, one-line summary) for each section of the MCP guide, in reading order.
_SECTION_META = (
    (
        "mcp-workflow",
        "Workflow",
        "The tool loop: create a figure, add data and plots, render, fix, save.",
    ),
    ("mcp-data", "Data", "Datasets, the expression language, the function list and file access."),
    (
        "mcp-analysis",
        "Analysis",
        "analyze_dataset: fit, smooth, FFT, peaks, filters, histogram and more, with every option.",
    ),
    (
        "mcp-plot-kinds",
        "add_plot kinds",
        "Every add_plot kind with its arguments, plus add_field and add_annotation.",
    ),
    (
        "mcp-recipes",
        "Recipes",
        "Ready-made plot kinds in a designed look: the table of recipes and their slots.",
    ),
    (
        "mcp-animation",
        "Animation",
        "animate: time, stages and variables drive properties; GIF/MP4/WebM output; "
        "simulate for dynamics; worked recipes.",
    ),
    (
        "mcp-video",
        "A whole video",
        "compose_video: several animated scenes cut into one explainer video, step by step "
        "in the manim style; the workflow for an agent.",
    ),
    (
        "mcp-interactive",
        "Controls, edits and scripts",
        "Sliders and buttons, bindings, update_artist, a second y axis, and export_script.",
    ),
    ("mcp-styling", "Styling", "Style presets, colours and colormaps."),
    (
        "mcp-notes",
        "Things to know",
        "Multi-panel figures, defaults, limits and other behaviour worth knowing.",
    ),
)

_INTRO = """GLPlot renders figures headlessly (a matplotlib-based export) and returns PNGs. Everything is
addressed by ids: `figure_id` (`fig1`...), `artist_id` (`a1`... one per add_* call) and dataset
names. Ids are per server session.
"""

_MORE = """## Learning more

Read-only tools that explain the library itself -- use them rather than guessing from memory:

* `get_guide()` -- list every guide topic; `get_guide("quickstart")` returns one. The `mcp-*` topics are
  this guide split up; the rest cover the **Python API** (plot types, axes and styling, 3-D, animation,
  controls, performance, matplotlib differences, troubleshooting).
* `search_api("heatmap")` -- find Python functions by what they do; `get_api_doc("hist2d")` -- signature and docstring.
* `get_example()` / `get_example("surface_3d")` -- complete runnable scripts.
* `server_info()` -- versions, limits, which file roots are open, what is currently loaded.
"""


def _animation_guide(limits: Limits) -> str:
    """The animation topic: prose, with the vocabulary, formats and limits read from the code."""
    caps = limits.animation
    formats = motion.available_formats()
    files = {
        name: why for name, why in formats.items() if f".{name}" in motion.ANIMATION_EXTENSIONS
    }
    available = ", ".join(f".{name}" for name, why in files.items() if why is None)
    missing = "; ".join(f"{name}: {why}" for name, why in files.items() if why)
    body = ANIMATION_BODY.format(
        vocabulary=dynamics.timeline_guide(),
        formats=available + (f" (not available here -- {missing})" if missing else ""),
        max_frames=caps.max_frames,
        max_seconds=int(caps.max_seconds),
        max_stages=caps.max_stages,
    )
    recipes_text = (
        "\n## Worked recipes\n\nEach one runs as written (the test suite executes them). Change "
        "the numbers, not the shape.\n\n" + cookbook.render_cookbook()
    )
    return ANIMATION_INTRO + "\n" + body + recipes_text


def _sections(sandbox: Optional[FileSandbox], limits: Limits) -> List[Tuple[str, str]]:
    """The guide's sections as ``(topic id, markdown body)``; the heading is added by the caller."""
    vocabulary = expression_vocabulary()
    styles = ", ".join(row[0] for row in gplt.plot_styles())
    if sandbox is not None and sandbox.enabled:
        file_text = (
            "Readable/writable roots: "
            + ", ".join(f"`{r}`" for r in sandbox.roots)
            + ". Relative paths start at the first root. "
            + f"Read: {', '.join(sandbox.READ_EXTENSIONS)}. "
            + f"Write: {', '.join(sandbox.WRITE_EXTENSIONS)} (never overwrites unless `overwrite=true`)."
        )
    else:
        file_text = (
            "File access is **disabled** on this server (it was started without `--root`). "
            "`load_dataset`, `save_figure` and `add_field(file=...)` will refuse."
        )
    recipe_table = _recipe_table()
    workflow = """1. `create_figure(title?, width, height, rows, cols, projection='2d'|'3d', style?)`
2. Get data: `create_dataset(name, columns)`, `load_dataset(path)`, or skip it with
   `add_function` for a formula. To *analyse* it (fit, smooth, FFT, peaks...) use
   `analyze_dataset`; the result lands in the dataset ready to plot.
3. Draw: `add_plot`, `add_function`, `add_field`, `add_recipe`, `add_annotation`.
4. Polish: `set_axes` (title, labels, limits, scales, grid, legend), `set_style`.
5. **Look**: `render_figure` returns the image plus the resolved axis limits, labels and legend
   text, and an `issues` list: what it could prove is wrong (a missing axis label or title, an
   empty panel, overlapping tick labels, points a log axis cannot draw, data cut off by the limits)
   with a severity and the call that fixes it. Check both, then fix what is wrong (`remove_artist`,
   `set_axes`) and render again until `issues` is empty or only holds things you chose.
6. `save_figure(path)` when it is right.

To animate: finish the still first (data, plots, `set_axes` with pinned limits), then
`animate(figure_id, variables=..., stages=..., bindings=...)` *without* `output` to preview (a contact
sheet and `issues` come back), fix, and repeat with `output='name.gif'`. Dynamics that are not a formula
of time come from `simulate`. To change something already drawn use `update_artist`; to make the figure
depend on sliders, `add_control` + `bind_property`; to give the user the Python, `export_script`. See
`get_guide('mcp-animation')` and `get_guide('mcp-interactive')`.
"""
    data = f"""`create_dataset(name, columns)`: each column is a list of numbers or an **expression string**.
Expressions run in order and see earlier columns by name, so
`{{"t": "linspace(0, 10, 500)", "y": "sin(t) * exp(-t / 5)", "noisy": "y + noise(t, 0.05, 1)"}}`
works. `base="other"` starts from another dataset's columns. A scalar result repeats to the
column length. Column names must be identifiers.

Functions: {", ".join(vocabulary["functions"])}.
Constants: {", ".join(vocabulary["constants"])}.
No attribute access (`np.sin` is rejected), no builtins, no imports. `min`/`max`/`sum`/`mean`
are reductions; use `minimum`/`maximum` for elementwise. Constructors are capped at
{limits.max_array_elements:,} elements. Inline lists are capped at {limits.max_inline_values:,}
values: for more, write a file and `load_dataset` it.

**Text in files.** `load_dataset` reads a column of ISO dates (`2020-01-05`, `2020-01-05 10:30`) as
*days since 1970-01-01* (a number you can plot, fit and difference), and any other text column
(city names, groups) as a **category**: the column holds each row's label index and
`describe_dataset` shows the labels (`kind: "category"`). A cell that is empty becomes NaN. Use
`select_rows(dataset, equals={{"city": "Lima"}})` to pick rows by label (or `where="temp > 10"`,
`sort_by`, `limit`, `every`), and `analyze_dataset(dataset, "aggregate", y="temp",
options={{"by": "city"}})` for a statistic per category -- the result plots straight to a bar
chart, whose tick labels are the category names. A category column plotted as x/y/color is just
its indices; the reply says which label each index stands for.

{file_text}
"""
    kinds = f"""| kind | needs | optional |
|---|---|---|
| line | y (x defaults to 0..n-1) | z (3-D), color, linewidth, linestyle, marker, size, label |
| scatter | x, y | z (3-D), color or color_by (+cmap, colorbar), size or size_by, marker, label |
| step | x, y | color, linewidth, linestyle |
| stem | x, y | label |
| bar | x (positions), y (heights) | error, width, color, tick_labels (category names) |
| barh | x (positions), y (lengths) | color, tick_labels |
| area | x, y | y2 (lower edge, default 0), color, alpha |
| errorbar | x, y, error | color, label |
| hist | y (values) | bins, color, alpha |
| box, violin | y (values), or dataset + columns=[...] for one box per column | tick names come from columns |
| pie | y (non-negative sizes) | tick_labels (slice names) |
| hist2d, hexbin | x, y | cmap, bins |

`x`, `y`, `z`, `error`, `color_by`, ... take either a **column name** (with `dataset=`) or a short
list of numbers. An option a kind does not use is ignored and reported back, not an error.
Series without a colour take the next colour of the figure's cycle.

`add_field(kind, expression | matrix | file, x_range, y_range, resolution)` draws a 2-D scalar field
z(x, y): kinds {", ".join(FIELD_KINDS)}. `surface`/`wireframe` need `create_figure(projection='3d')`;
in a 3-D figure `add_plot` accepts only `line` and `scatter`, both with `z`. The expression uses
`x` and `y`, e.g. `sin(sqrt(x**2 + y**2))`. `file` takes a 2-D `.npy` or an image.

`add_annotation(kind, ...)`: {", ".join(ANNOTATION_KINDS)} (`hline` at `y`, `vline` at `x`).
"""
    recipes_text = f"""`add_recipe(figure_id, recipe, dataset=..., columns={{slot: column}}, look=...)` draws a plot kind
*in the look it was designed in* -- page, palette, colormap and (3-D) camera -- so one call gives a
finished-looking figure. `sample=true` uses the recipe's own example table (added as a dataset)
instead of `dataset`. Columns are guessed from their names; `columns` overrides only what you name
(`'(index)'` is the row number). `look` is `'recipe'` (default), `'keep'` (the figure's current look)
or any style key. A 3-D recipe needs `create_figure(projection='3d')`. Axis names come from the
column names; add a title, limits or a legend with `set_axes`. `list_recipes` returns this table
as JSON.

{recipe_table}
"""
    styling = f"""Presets for `set_style`/`create_figure(style=)`: {styles}.
Colours: names (`red`), hex (`#1f77b4`), `tab:blue`. Colormaps: {COLORMAPS}.
"""
    notes = f"""* Multi-panel figures (`rows`/`cols`): pass `panel=N` (1-based, row-major) to every add_*/set_axes
  call. `xscale`/`yscale` are figure-wide in GLPlot, so they are refused on multi-panel figures.
* `area` is opaque by default; a translucent fill shows seams in the export.
* `contourf` is drawn as a continuous ramp rather than discrete bands (reported as a warning).
* Titles are set with `set_axes(title=...)`; a figure title on a grid goes in `create_figure(title=)`.
* Renders are capped near {limits.max_image_edge}px on the long edge. Use `save_figure(scale=)` for
  a larger file.
* Close figures you are done with (`close_figure`); at most {limits.max_figures} stay open.
"""
    bodies = {
        "mcp-animation": _animation_guide(limits),
        "mcp-video": VIDEO_BODY.format(
            max_gif=limits.animation.max_video_gif_frames,
            max_frames=limits.animation.max_video_frames,
        ),
        "mcp-interactive": INTERACTIVE_BODY,
        "mcp-workflow": workflow,
        "mcp-data": data,
        "mcp-analysis": analysis.operations_guide(),
        "mcp-plot-kinds": kinds,
        "mcp-recipes": recipes_text,
        "mcp-styling": styling,
        "mcp-notes": notes,
    }
    return [(topic_id, bodies[topic_id]) for topic_id, _title, _summary in _SECTION_META]


def mcp_topics(
    sandbox: Optional[FileSandbox] = None, limits: Limits = DEFAULT_LIMITS
) -> List[Topic]:
    """The MCP guide as topics, for :func:`glplot.llm.guide` to serve beside the packaged ones."""
    meta = {topic_id: (title, summary) for topic_id, title, summary in _SECTION_META}
    topics = []
    for topic_id, body in _sections(sandbox, limits):
        title, summary = meta[topic_id]
        topics.append(Topic(topic_id, f"MCP: {title}", summary, body.rstrip() + "\n"))
    return topics


def build_guide(sandbox: Optional[FileSandbox] = None, limits: Limits = DEFAULT_LIMITS) -> str:
    """The full markdown guide (the ``glplot://guide`` resource)."""
    meta = {topic_id: title for topic_id, title, _summary in _SECTION_META}
    parts = ["# GLPlot MCP guide", "", _INTRO]
    for topic_id, body in _sections(sandbox, limits):
        parts.append(f"## {meta[topic_id]}\n\n{body}")
    parts.append(_MORE)
    return "\n".join(parts)
