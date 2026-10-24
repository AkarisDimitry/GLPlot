"""The usage guide a model reads before it plots. Pure text, no MCP SDK.

Served as the ``glplot://guide`` resource and trimmed into the server's ``instructions``.
The expression vocabulary and the style list are *generated* from the code that enforces
them (:data:`glplot.gui.expressions.SAFE_NAMES`, :func:`glplot.pyplot.plot_styles`), so the
guide cannot promise a function the evaluator would reject.
"""

from __future__ import annotations

from typing import Optional

from .. import pyplot as gplt
from ..gui import recipes
from .state import DEFAULT_LIMITS, FileSandbox, Limits, expression_vocabulary
from .toolbox import ANNOTATION_KINDS, FIELD_KINDS

__all__ = ["build_guide", "build_instructions"]

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


def build_instructions(sandbox: Optional[FileSandbox] = None) -> str:
    """The short text sent to the client at connect time."""
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
        "Keep large data out of the conversation: load it from a file or generate it with an "
        "expression (e.g. 'sin(2*pi*t)*exp(-t)') instead of typing numbers. Always set a title "
        "and axis labels, and call render_figure to check the result before answering. "
        "Read the glplot://guide resource for the per-kind argument table and the expression "
        "vocabulary. " + files
    )


def build_guide(sandbox: Optional[FileSandbox] = None, limits: Limits = DEFAULT_LIMITS) -> str:
    """The full markdown guide."""
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
    return f"""# GLPlot MCP guide

GLPlot renders figures headlessly (a matplotlib-based export) and returns PNGs. Everything is
addressed by ids: `figure_id` (`fig1`...), `artist_id` (`a1`... one per add_* call) and dataset
names. Ids are per server session.

## Workflow

1. `create_figure(title?, width, height, rows, cols, projection='2d'|'3d', style?)`
2. Get data: `create_dataset(name, columns)`, `load_dataset(path)`, or skip it with
   `add_function` for a formula.
3. Draw: `add_plot`, `add_function`, `add_field`, `add_recipe`, `add_annotation`.
4. Polish: `set_axes` (title, labels, limits, scales, grid, legend), `set_style`.
5. **Look**: `render_figure` returns the image plus the resolved axis limits, labels and legend
   text. Check it, then fix what is wrong (`remove_artist`, `set_axes`).
6. `save_figure(path)` when it is right.

## Data

`create_dataset(name, columns)`: each column is a list of numbers or an **expression string**.
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

{file_text}

## add_plot kinds

| kind | needs | optional |
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

## Recipes

`add_recipe(figure_id, recipe, dataset=..., columns={{slot: column}}, look=...)` draws a plot kind
*in the look it was designed in* -- page, palette, colormap and (3-D) camera -- so one call gives a
finished-looking figure. `sample=true` uses the recipe's own example table (added as a dataset)
instead of `dataset`. Columns are guessed from their names; `columns` overrides only what you name
(`'(index)'` is the row number). `look` is `'recipe'` (default), `'keep'` (the figure's current look)
or any style key. A 3-D recipe needs `create_figure(projection='3d')`. Axis names come from the
column names; add a title, limits or a legend with `set_axes`. `list_recipes` returns this table
as JSON.

{recipe_table}

## Styling

Presets for `set_style`/`create_figure(style=)`: {styles}.
Colours: names (`red`), hex (`#1f77b4`), `tab:blue`. Colormaps: {COLORMAPS}.

## Things to know

* Multi-panel figures (`rows`/`cols`): pass `panel=N` (1-based, row-major) to every add_*/set_axes
  call. `xscale`/`yscale` are figure-wide in GLPlot, so they are refused on multi-panel figures.
* `area` is opaque by default; a translucent fill shows seams in the export.
* `contourf` is drawn as a continuous ramp rather than discrete bands (reported as a warning).
* Titles are set with `set_axes(title=...)`; a figure title on a grid goes in `create_figure(title=)`.
* Renders are capped near {limits.max_image_edge}px on the long edge. Use `save_figure(scale=)` for
  a larger file.
* Close figures you are done with (`close_figure`); at most {limits.max_figures} stay open.
"""
