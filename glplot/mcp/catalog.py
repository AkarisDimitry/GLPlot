"""The MCP *resources* and *prompts* this server offers, described once, with no SDK.

:mod:`glplot.mcp.server` (the official ``mcp`` SDK) and :mod:`glplot.mcp.builtin` (the
dependency-free stdio server) both serve exactly what is listed here, so the two can never
disagree about which resources exist or what a prompt says. Tools are the exception: their JSON
schemas come from the typed signatures in ``server.py`` and are exported to ``tools.json`` (see
:mod:`glplot.mcp.schema`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Pattern, Tuple, Union

from .state import McpError
from .toolbox import Toolbox

__all__ = [
    "PROMPTS",
    "RESOURCES",
    "PromptArg",
    "PromptDef",
    "ResourceDef",
    "find_prompt",
    "find_resource",
]


# ----------------------------------------------------------------------------------
# Resources
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceDef:
    """A readable thing. ``uri`` may be a template such as ``glplot://api/{name}``."""

    uri: str
    name: str
    description: str
    mime_type: str
    read: Callable[[Toolbox, Dict[str, str]], Union[str, bytes]]

    @property
    def is_template(self) -> bool:
        return "{" in self.uri

    @property
    def pattern(self) -> Pattern[str]:
        escaped = re.escape(self.uri)
        return re.compile("^" + re.sub(r"\\\{(\w+)\\\}", r"(?P<\1>[^/]+)", escaped) + "$")


def _guide(box: Toolbox, params: Dict[str, str]) -> str:
    from .guide import build_guide

    return build_guide(box.sandbox, box.limits)


def _topic(box: Toolbox, params: Dict[str, str]) -> str:
    return box.get_guide(params["topic"]).message


def _api_index(box: Toolbox, params: Dict[str, str]) -> str:
    from ..llm.render import api_reference

    return api_reference()


def _api_doc(box: Toolbox, params: Dict[str, str]) -> str:
    return box.get_api_doc(params["name"]).message


def _examples_index(box: Toolbox, params: Dict[str, str]) -> str:
    return box.get_example(None).message


def _example(box: Toolbox, params: Dict[str, str]) -> str:
    from .. import llm

    try:
        return llm.example(params["name"])["code"]
    except KeyError as exc:
        raise McpError(str(exc)) from exc


def _llms(box: Toolbox, params: Dict[str, str]) -> str:
    from .. import llm

    return llm.llms_txt()


def _llms_full(box: Toolbox, params: Dict[str, str]) -> str:
    from ..llm.render import llms_full_txt
    from .guide import mcp_topics

    return llms_full_txt(mcp_topics(box.sandbox, box.limits))


def _capabilities(box: Toolbox, params: Dict[str, str]) -> str:
    return json.dumps(box.server_info().data, default=str)


def _recipes(box: Toolbox, params: Dict[str, str]) -> str:
    return json.dumps(box.list_recipes(None).data, default=str)


def _styles(box: Toolbox, params: Dict[str, str]) -> str:
    from .. import pyplot as gplt

    return json.dumps([{"key": k, "title": t, "description": d} for k, t, d in gplt.plot_styles()])


def _figure_image(box: Toolbox, params: Dict[str, str]) -> bytes:
    return box.render_png(params["figure_id"])


def _dataset(box: Toolbox, params: Dict[str, str]) -> str:
    return json.dumps(box.describe_dataset(params["name"], 10).data, default=str)


RESOURCES: Tuple[ResourceDef, ...] = (
    ResourceDef(
        "glplot://guide",
        "guide",
        "How to plot with GLPlot over MCP: workflow, per-kind arguments, expression functions.",
        "text/markdown",
        _guide,
    ),
    ResourceDef(
        "glplot://guide/{topic}",
        "guide-topic",
        "One guide topic (see get_guide() for the ids): the MCP workflow and the Python API.",
        "text/markdown",
        _topic,
    ),
    ResourceDef(
        "glplot://api",
        "api-reference",
        "Every GLPlot Python function with its call shape, grouped by category.",
        "text/markdown",
        _api_index,
    ),
    ResourceDef(
        "glplot://api/{name}",
        "api-doc",
        "Signature and full docstring of one GLPlot function, e.g. glplot://api/hist2d.",
        "text/markdown",
        _api_doc,
    ),
    ResourceDef(
        "glplot://examples",
        "examples",
        "The runnable example scripts, with titles.",
        "text/plain",
        _examples_index,
    ),
    ResourceDef(
        "glplot://examples/{name}",
        "example",
        "The code of one example script, e.g. glplot://examples/line_plot.",
        "text/x-python",
        _example,
    ),
    ResourceDef(
        "glplot://llms.txt",
        "llms-txt",
        "A short llms.txt: what GLPlot is and where to read more.",
        "text/plain",
        _llms,
    ),
    ResourceDef(
        "glplot://llms-full.txt",
        "llms-full-txt",
        "All the documentation in one document: every guide topic, then the API reference.",
        "text/plain",
        _llms_full,
    ),
    ResourceDef(
        "glplot://capabilities",
        "capabilities",
        "Versions, limits, file access and the size of the API, as JSON.",
        "application/json",
        _capabilities,
    ),
    ResourceDef(
        "glplot://recipes",
        "recipes",
        "The ready-made recipes (plot kind + look), as JSON.",
        "application/json",
        _recipes,
    ),
    ResourceDef(
        "glplot://styles",
        "styles",
        "The style presets, as JSON.",
        "application/json",
        _styles,
    ),
    ResourceDef(
        "glplot://figures/{figure_id}/image",
        "figure-image",
        "The current rendering of an open figure, as a PNG.",
        "image/png",
        _figure_image,
    ),
    ResourceDef(
        "glplot://datasets/{name}",
        "dataset",
        "A dataset's columns, statistics and first rows, as JSON.",
        "application/json",
        _dataset,
    ),
)


def find_resource(uri: str) -> Optional[Tuple[ResourceDef, Dict[str, str]]]:
    """The resource that serves ``uri`` and the template parameters in it, or None."""
    for definition in RESOURCES:
        match = definition.pattern.match(uri)
        if match:
            return definition, match.groupdict()
    return None


# ----------------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptArg:
    name: str
    description: str
    required: bool = False


@dataclass(frozen=True)
class PromptDef:
    name: str
    description: str
    arguments: Tuple[PromptArg, ...]
    render: Callable[..., str]

    def build(self, **given: Any) -> str:
        """The prompt text. Raises :class:`McpError` for a missing required or unknown argument."""
        known = {a.name for a in self.arguments}
        extra = sorted(set(given) - known)
        if extra:
            raise McpError(f"Prompt {self.name!r} has no argument(s) {', '.join(extra)}.")
        missing = [a.name for a in self.arguments if a.required and not given.get(a.name)]
        if missing:
            raise McpError(f"Prompt {self.name!r} needs: {', '.join(missing)}.")
        return self.render(**{k: v for k, v in given.items() if v is not None})


def _plot_data_file(path: str, goal: str = "") -> str:
    extra = f" The goal: {goal}" if goal else ""
    return (
        f"Load {path!r} with load_dataset, look at its columns with describe_dataset, then "
        "choose the plot that best shows the data. Build the figure with create_figure and "
        "add_plot, add a title, axis labels with units and a legend where useful, and call "
        "render_figure to check it. Fix anything that looks wrong (overlapping labels, "
        "unreadable scale, missing legend) before you answer." + extra
    )


def _explore_function(expression: str, x_min: Any = -10.0, x_max: Any = 10.0) -> str:
    return (
        f"Plot y = {expression} for x from {x_min} to {x_max} using create_figure and "
        "add_function, label the axes, render_figure to look at it, then describe the "
        "shape: zeros, extrema, asymptotes and symmetry. If the default range hides "
        "something interesting, re-plot with a better one."
    )


def _quick_look(dataset: str, question: str = "") -> str:
    extra = f" The question to answer: {question}" if question else ""
    return (
        f"Use describe_dataset on {dataset!r} and decide what its columns are (a time or position "
        "axis, measurements, categories). Make the three most informative views -- for example a "
        "line or scatter of the main relationship, a histogram of the key variable and a "
        "multi-panel overview -- each with create_figure, add_plot and set_axes (title, "
        "labelled axes with units). render_figure every one and say what you see: trends, "
        "outliers, gaps, suspicious values. Close the figures you do not keep." + extra
    )


def _fix_my_figure(figure_id: str, problem: str = "") -> str:
    extra = f" The user says: {problem}" if problem else ""
    return (
        f"Call render_figure on {figure_id!r} and inspect the picture and the axis summary like a "
        "reviewer: are the axes labelled with units, is the title informative, do tick labels "
        "overlap, is anything clipped, is the scale appropriate (log for data over decades), "
        "are the colours distinguishable (consider set_style 'colorsafe'), is there a legend "
        "when there are several series? Fix each issue with set_axes / set_style / "
        "remove_artist + add_*, render again, and repeat until it is clean. Then summarise "
        "what you changed." + extra
    )


def _publication_figure(figure_id: str, venue: str = "a scientific journal") -> str:
    return (
        f"Prepare {figure_id!r} for {venue}. First render_figure to see it. Then: use "
        "set_style 'journal' (or 'colorsafe' / 'print' if it must survive greyscale), give every "
        "axis a "
        "name and unit, shorten the title to what the figure shows, keep the legend only if "
        "several series need telling apart, remove chart junk, and make sure nothing is "
        "clipped or overlapping. Render again and save_figure as PNG at scale 2 (and "
        "an SVG or PDF if a root directory is available)."
    )


def _compare_series(dataset: str, columns: str = "") -> str:
    which = f" the columns {columns}" if columns else " the numeric columns"
    return (
        f"Compare{which} of {dataset!r}. Start with describe_dataset. If the series share a unit "
        "and range, overlay them in one figure (add_plot line/scatter with a label each, "
        "legend on); if their ranges differ a lot, use a rows x cols grid with one panel per "
        "series (panel=N) or a log scale. Add a normalised view if shapes matter more than "
        "levels. render_figure, then state which series differ and how."
    )


def _fit_and_plot(dataset: str, x: str, y: str, model: str = "") -> str:
    which = (
        f"the {model} model"
        if model
        else "a sensible model (look at the data first; try linear, then exp_decay, gaussian, "
        "power_law, sigmoid or a custom expression, and keep the best R^2)"
    )
    return (
        f"Fit {y!r} against {x!r} in dataset {dataset!r} with {which} using analyze_dataset "
        "(operation 'fit'). Report the parameters with their standard errors and the R^2, and say "
        "whether the fit is believable. Then make a figure: add_plot the data as scatter and the "
        "fitted column as a line on the same figure, label the axes with units, turn the legend "
        "on, render_figure and check that the curve follows the points. If the fit is poor, try "
        "another model or pass options.p0, and compare."
    )


def _python_script_for(task: str) -> str:
    return (
        f"Write a complete Python script with GLPlot for: {task}\n\n"
        "Before writing code, learn the library instead of assuming it behaves like matplotlib: "
        "call get_guide('overview') and get_guide('quickstart'), search_api for the plot types "
        "you need and get_api_doc for each, and get_example for the closest runnable script. "
        "Rules: `import glplot.pyplot as plt`; end the script with plt.savefig('name.png') and "
        "never call plt.show() in an unattended run; label axes with units; animation is "
        "`import glplot.animation as animation`, not plt. Give the script, the command to run "
        "it and what the output file should show. If you can run it, run it, open the image and "
        "fix what is wrong."
    )


def _animate_this(description: str, figure_id: str = "", output: str = "") -> str:
    target = f"Use figure {figure_id}." if figure_id else "Build the still figure first."
    where = f" Write it to {output!r}." if output else " Preview first, then write a .gif or .mp4."
    return (
        f"Animate: {description}\n\n{target} Read get_guide('mcp-animation') before you start. "
        "Steps: (1) the still figure with everything in its final position and the axes PINNED "
        "(set_axes xlim/ylim/zlim); data that is a trajectory comes from `simulate`. "
        "(2) Decide what varies and in which stages (variables + stages), and which property "
        "follows which variable (bindings). (3) animate WITHOUT output: look at the contact sheet "
        "and read every issue (static, xlim_moves, loop_seam, abrupt_change). "
        f"(4) Fix and repeat.{where} (5) Say what the animation shows stage by stage, and offer "
        "export_script(kind='animation') for the Python."
    )


def _interactive_figure(figure_id: str, parameters: str = "") -> str:
    which = f" Parameters to expose: {parameters}." if parameters else ""
    return (
        f"Make figure {figure_id} depend on parameters.{which} Read get_guide('mcp-interactive'). "
        "For each parameter add_control (slider with a sensible range, or toggle/choice), then "
        "bind_property to the artists or axes it should drive, then try values with "
        "render_figure(params=...) and look at them. Finally give the user "
        "export_script(kind='interactive') so they get real sliders in a window."
    )


PROMPTS: Tuple[PromptDef, ...] = (
    PromptDef(
        "plot_data_file",
        "Plot a data file and iterate until the figure is publication-ready.",
        (
            PromptArg("path", "File inside the server's allowed directory.", True),
            PromptArg("goal", "What the figure should show."),
        ),
        _plot_data_file,
    ),
    PromptDef(
        "explore_function",
        "Plot a formula over a range and describe its behaviour.",
        (
            PromptArg("expression", "y as a function of x, e.g. 'sin(x)/x'.", True),
            PromptArg("x_min", "Left end of the range (default -10)."),
            PromptArg("x_max", "Right end of the range (default 10)."),
        ),
        _explore_function,
    ),
    PromptDef(
        "quick_look",
        "Explore a dataset: make the most informative plots and report what they show.",
        (
            PromptArg("dataset", "Dataset name (from create_dataset / load_dataset).", True),
            PromptArg("question", "A question the plots should help answer."),
        ),
        _quick_look,
    ),
    PromptDef(
        "fix_my_figure",
        "Review an existing figure like a reviewer and fix labels, scale, colours and clutter.",
        (
            PromptArg("figure_id", "Figure id from create_figure.", True),
            PromptArg("problem", "What looks wrong, if you know."),
        ),
        _fix_my_figure,
    ),
    PromptDef(
        "publication_figure",
        "Prepare a figure for a journal or slide: style, labels with units, clean export.",
        (
            PromptArg("figure_id", "Figure id from create_figure.", True),
            PromptArg("venue", "Where it will appear (default: a scientific journal)."),
        ),
        _publication_figure,
    ),
    PromptDef(
        "compare_series",
        "Compare several columns of a dataset: overlay, small multiples or log scale as needed.",
        (
            PromptArg("dataset", "Dataset name.", True),
            PromptArg("columns", "Comma-separated columns to compare (default: all numeric)."),
        ),
        _compare_series,
    ),
    PromptDef(
        "fit_and_plot",
        "Fit a model to two columns, report parameters and R^2, and plot data and fit together.",
        (
            PromptArg("dataset", "Dataset name.", True),
            PromptArg("x", "x column.", True),
            PromptArg("y", "y column.", True),
            PromptArg("model", "Model to try (default: pick one)."),
        ),
        _fit_and_plot,
    ),
    PromptDef(
        "animate_this",
        "Plan, preview, check and write an animation (GIF/MP4) of something described in words.",
        (
            PromptArg("description", "What should move or change, and how.", True),
            PromptArg("figure_id", "An existing figure to animate (default: build one)."),
            PromptArg("output", "File to write, e.g. 'wave.gif' or 'orbit.mp4'."),
        ),
        _animate_this,
    ),
    PromptDef(
        "interactive_figure",
        "Turn a figure into one with sliders/toggles bound to its properties, plus a script.",
        (
            PromptArg("figure_id", "Figure id from create_figure.", True),
            PromptArg("parameters", "Which parameters to expose, e.g. 'amplitude, frequency'."),
        ),
        _interactive_figure,
    ),
    PromptDef(
        "python_script_for",
        "Write (and verify) a Python script that uses GLPlot, after reading its guides and "
        "examples.",
        (PromptArg("task", "What the script should plot or animate.", True),),
        _python_script_for,
    ),
)


def find_prompt(name: str) -> PromptDef:
    for definition in PROMPTS:
        if definition.name == name:
            return definition
    raise McpError(f"Unknown prompt {name!r}. Available: {', '.join(p.name for p in PROMPTS)}.")
