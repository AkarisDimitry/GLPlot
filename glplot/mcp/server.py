"""Describe the :class:`~glplot.mcp.toolbox.Toolbox` to the MCP protocol.

This is the only module that imports the ``mcp`` SDK, and it adds no behaviour: every tool
here is a typed signature, a description the model reads, and a call into the toolbox. The
signatures are written out rather than generated so the JSON schema a client shows its model
carries a hand-written description per argument -- that text is the model's only manual.

Two SDK generations are supported: ``mcp`` 2.x (``MCPServer``) and 1.x (``FastMCP``). They
differ in a class name and where ``ToolError`` lives; everything else used here is common.
"""

import argparse
import json
import logging
import sys
from typing import Annotated, Any, Dict, List, Optional, Sequence, Union

from pydantic import Field

from .guide import build_guide, build_instructions
from .state import DEFAULT_LIMITS, FileSandbox, Limits, McpError
from .toolbox import ANNOTATION_KINDS, FIELD_KINDS, PLOT_KINDS, Toolbox, ToolResult

__all__ = ["SdkUnavailable", "build_server", "main"]

logger = logging.getLogger(__name__)


class SdkUnavailable(ImportError):
    """The optional ``mcp`` package is not installed."""


def _load_sdk() -> Dict[str, Any]:
    """Import whichever MCP SDK is installed and return the names this module needs."""
    try:  # mcp >= 2
        from mcp.server.mcpserver import Image, MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
    except ImportError:
        try:  # mcp 1.x
            from mcp.server.fastmcp import FastMCP as MCPServer
            from mcp.server.fastmcp import Image
            from mcp.server.fastmcp.exceptions import ToolError
        except ImportError as exc:
            raise SdkUnavailable(
                "The MCP server needs the optional 'mcp' package. Install it with "
                '`pip install "glplot[mcp]"` (or `pip install mcp`).'
            ) from exc
    try:
        from mcp.types import ToolAnnotations
    except ImportError:  # pragma: no cover - mcp 2.x re-exports from mcp_types
        from mcp_types import ToolAnnotations
    return {
        "MCPServer": MCPServer,
        "Image": Image,
        "ToolError": ToolError,
        "ToolAnnotations": ToolAnnotations,
    }


# ----------------------------------------------------------------------------------
# Argument types, described once
# ----------------------------------------------------------------------------------

FigureId = Annotated[str, Field(description="Figure id from create_figure, e.g. 'fig1'.")]
Panel = Annotated[
    int,
    Field(
        ge=1,
        le=36,
        description="Panel number in a multi-panel figure, 1-based, row-major. Leave at 1 for "
        "a single-panel figure.",
    ),
]
Series = Annotated[
    Union[str, List[Optional[float]], None],
    Field(
        description="A column name (requires `dataset`) or a short list of numbers; null in a "
        "list is a gap. Prefer column names for anything over ~100 values."
    ),
]
Color = Annotated[
    Optional[str],
    Field(
        description="Colour name ('red'), hex ('#1f77b4') or 'tab:blue'. Default: next in cycle."
    ),
]


def _content(sdk: Dict[str, Any], result: ToolResult) -> Any:
    """What the SDK sends back: the text, and the picture first when there is one."""
    text = result.text()
    if result.image is None:
        return text
    return [sdk["Image"](data=result.image, format=result.image_format), text]


def build_server(
    toolbox: Optional[Toolbox] = None,
    *,
    sandbox: Optional[FileSandbox] = None,
    limits: Limits = DEFAULT_LIMITS,
    name: str = "glplot",
) -> Any:
    """Build the MCP server for ``toolbox`` (a fresh one if not given).

    Raises :class:`SdkUnavailable` if the ``mcp`` package is missing.
    """
    sdk = _load_sdk()
    ToolError, Annotations = sdk["ToolError"], sdk["ToolAnnotations"]
    box = toolbox if toolbox is not None else Toolbox(sandbox, limits)
    server = sdk["MCPServer"](name, instructions=build_instructions(box.sandbox))

    def run(method: Any, *args: Any, **kwargs: Any) -> Any:
        """Call a toolbox method; make its ``McpError`` the one thing the model is shown."""
        try:
            return _content(sdk, method(*args, **kwargs))
        except McpError as exc:
            raise ToolError(str(exc)) from exc

    def tool(*, read_only: bool = False, destructive: bool = False) -> Any:
        """Register the decorated function as a tool, with the hints a client uses to decide
        whether it needs the user's approval (read-only tools usually do not)."""

        def decorator(fn: Any) -> Any:
            server.add_tool(
                fn,
                name=fn.__name__,
                annotations=Annotations(
                    readOnlyHint=read_only,
                    destructiveHint=destructive,
                    idempotentHint=read_only,
                    openWorldHint=False,
                ),
            )
            return fn

        return decorator

    # -- figures ------------------------------------------------------------------

    @tool()
    def create_figure(
        title: Annotated[Optional[str], Field(description="Overall title.")] = None,
        width: Annotated[int, Field(ge=200, le=3000, description="Width in pixels.")] = 1000,
        height: Annotated[int, Field(ge=200, le=3000, description="Height in pixels.")] = 640,
        rows: Annotated[int, Field(ge=1, le=6, description="Panel rows.")] = 1,
        cols: Annotated[int, Field(ge=1, le=6, description="Panel columns.")] = 1,
        projection: Annotated[
            str, Field(description="'2d' or '3d' (applies to every panel).")
        ] = "2d",
        style: Annotated[
            Optional[str],
            Field(description="Style preset, e.g. 'journal', 'dark', 'chalk'. See glplot://guide."),
        ] = None,
    ) -> Any:
        """Create an empty figure and return its figure_id. Start here.

        Use rows/cols for a grid of panels; address a panel with `panel=N` in later calls.
        """
        return run(box.create_figure, title, width, height, rows, cols, projection, style)

    @tool(read_only=True)
    def describe_figure(
        figure_id: Annotated[
            Optional[str], Field(description="Figure to describe; omit to list all open figures.")
        ] = None,
    ) -> Any:
        """List open figures, or describe one: its artists and what each panel currently shows
        (axis limits, labels, legend entries). Cheaper than render_figure and returns no image."""
        return run(box.describe_figure, figure_id)

    @tool(destructive=True)
    def close_figure(figure_id: FigureId) -> Any:
        """Close a figure and free its data. Datasets are kept."""
        return run(box.close_figure, figure_id)

    # -- datasets -----------------------------------------------------------------

    @tool()
    def create_dataset(
        name: Annotated[
            str, Field(description="Dataset name: letters, digits, _ . - (no spaces).")
        ],
        columns: Annotated[
            Dict[str, Union[List[Optional[float]], str]],
            Field(
                description="Ordered map of column name -> list of numbers OR an expression "
                "string over earlier columns, e.g. {'t': 'linspace(0, 10, 500)', "
                "'y': 'sin(t)*exp(-t/5)'}. Column names must be identifiers."
            ),
        ],
        base: Annotated[
            Optional[str],
            Field(description="Start from this dataset's columns; new columns may use them."),
        ] = None,
    ) -> Any:
        """Create (or replace) a named table of equal-length numeric columns.

        Expressions are evaluated safely (no attribute access, no builtins) with numpy
        functions such as sin, exp, sqrt, linspace, arange, noise(n, scale, seed); see
        glplot://guide for the full list. Use this instead of typing out data points.
        """
        return run(box.create_dataset, name, columns, base)

    @tool()
    def load_dataset(
        path: Annotated[
            str,
            Field(description="File inside the server's allowed directory (relative or absolute)."),
        ],
        name: Annotated[
            Optional[str], Field(description="Name to give it; default: file name.")
        ] = None,
    ) -> Any:
        """Load a CSV/TSV/TXT, NPY/NPZ, JSON or image file as a dataset.

        The first row is used as column names when it is a header. A 2-D .npy becomes one
        column per array column. Returns the columns with min/max/mean/std and the first rows.
        """
        return run(box.load_dataset, path, name)

    @tool(read_only=True)
    def describe_dataset(
        name: Annotated[
            Optional[str], Field(description="Dataset to describe; omit to list all datasets.")
        ] = None,
        rows: Annotated[int, Field(ge=0, le=50, description="Rows of data to show.")] = 5,
    ) -> Any:
        """Show a dataset's columns, per-column statistics and first rows, or list all datasets."""
        return run(box.describe_dataset, name, rows)

    @tool(destructive=True)
    def delete_dataset(name: Annotated[str, Field(description="Dataset to delete.")]) -> Any:
        """Delete a dataset. Figures already drawn from it keep their picture."""
        return run(box.delete_dataset, name)

    # -- drawing ------------------------------------------------------------------

    @tool()
    def add_plot(
        figure_id: FigureId,
        kind: Annotated[
            str,
            Field(
                description="One of: "
                + ", ".join(PLOT_KINDS)
                + ". See the table in glplot://guide."
            ),
        ],
        dataset: Annotated[
            Optional[str], Field(description="Dataset that column-name arguments refer to.")
        ] = None,
        x: Series = None,
        y: Series = None,
        z: Annotated[
            Series,
            Field(description="Third coordinate. Required on a 3-D figure (line/scatter only)."),
        ] = None,
        error: Annotated[
            Series,
            Field(description="+/- size for errorbar (per point or a single number) and bar."),
        ] = None,
        y2: Annotated[Series, Field(description="area: the lower edge (default 0).")] = None,
        columns: Annotated[
            Optional[List[str]],
            Field(description="box/violin: dataset columns, one box per column."),
        ] = None,
        color: Color = None,
        color_by: Annotated[
            Series, Field(description="scatter: values mapped through `cmap` to colour each point.")
        ] = None,
        size_by: Annotated[
            Series, Field(description="scatter: values mapped to marker size.")
        ] = None,
        cmap: Annotated[
            Optional[str],
            Field(description="Colormap for color_by / hist2d / hexbin, e.g. 'viridis'."),
        ] = None,
        label: Annotated[Optional[str], Field(description="Legend text for this series.")] = None,
        alpha: Annotated[Optional[float], Field(ge=0, le=1, description="Opacity 0-1.")] = None,
        linewidth: Annotated[
            Optional[float], Field(gt=0, le=50, description="Line width in px.")
        ] = None,
        linestyle: Annotated[Optional[str], Field(description="'-', '--', '-.' or ':'.")] = None,
        marker: Annotated[
            Optional[str], Field(description="Marker: 'o', 's', '^', '+', 'x', ...")
        ] = None,
        size: Annotated[
            Optional[float],
            Field(gt=0, le=200, description="Marker size in px (line markers, scatter)."),
        ] = None,
        bins: Annotated[
            Optional[int], Field(ge=1, le=1000, description="hist/hist2d: number of bins.")
        ] = None,
        width: Annotated[Optional[float], Field(gt=0, le=10, description="bar: bar width.")] = None,
        tick_labels: Annotated[
            Optional[List[str]],
            Field(description="bar/barh: category names under the bars; pie: slice names."),
        ] = None,
        colorbar: Annotated[
            bool, Field(description="Add a colourbar (scatter with color_by).")
        ] = False,
        panel: Panel = 1,
    ) -> Any:
        """Draw data as a line, scatter, bar, histogram, box, pie, ... and return an artist_id.

        Each kind needs different series; the per-kind table is in glplot://guide. Options a
        kind does not use are ignored and listed in the reply. Returns the artist_id used by
        remove_artist.
        """
        return run(
            box.add_plot,
            figure_id,
            kind,
            dataset,
            x,
            y,
            z,
            error,
            y2,
            columns,
            color,
            color_by,
            size_by,
            cmap,
            label,
            alpha,
            linewidth,
            linestyle,
            marker,
            size,
            bins,
            width,
            tick_labels,
            colorbar,
            panel,
        )

    @tool()
    def add_function(
        figure_id: FigureId,
        expression: Annotated[
            str,
            Field(
                description="y as a function of x, e.g. 'sin(a*x)/x'. "
                "Parameters go in `parameters`."
            ),
        ],
        x_min: Annotated[float, Field(description="Left end of the x range.")],
        x_max: Annotated[float, Field(description="Right end of the x range (> x_min).")],
        samples: Annotated[int, Field(ge=2, le=100000, description="Sample count.")] = 500,
        parameters: Annotated[
            Optional[Dict[str, float]],
            Field(description="Named constants usable in the expression, e.g. {'a': 2}."),
        ] = None,
        kind: Annotated[str, Field(description="'line', 'area', 'scatter' or 'step'.")] = "line",
        label: Annotated[
            Optional[str], Field(description="Legend text. Default: the expression itself.")
        ] = None,
        color: Color = None,
        alpha: Annotated[Optional[float], Field(ge=0, le=1, description="Opacity 0-1.")] = None,
        linewidth: Annotated[
            Optional[float], Field(gt=0, le=50, description="Line width in px.")
        ] = None,
        linestyle: Annotated[Optional[str], Field(description="'-', '--', '-.' or ':'.")] = None,
        panel: Panel = 1,
    ) -> Any:
        """Plot y = expression(x) over [x_min, x_max] in one call, without making a dataset.

        The quickest way to draw a formula or to overlay several (call it once per curve).
        """
        return run(
            box.add_function,
            figure_id,
            expression,
            x_min,
            x_max,
            samples,
            parameters,
            kind,
            label,
            color,
            alpha,
            linewidth,
            linestyle,
            panel,
        )

    @tool()
    def add_field(
        figure_id: FigureId,
        kind: Annotated[
            str,
            Field(
                description="One of: "
                + ", ".join(FIELD_KINDS)
                + " (surface/wireframe need a 3-D figure)."
            ),
        ] = "image",
        expression: Annotated[
            Optional[str], Field(description="z in terms of x and y, e.g. 'sin(x)*cos(y)'.")
        ] = None,
        matrix: Annotated[
            Optional[List[List[float]]],
            Field(description="Small inline grid, rows of numbers (row 0 = lowest y)."),
        ] = None,
        file: Annotated[
            Optional[str],
            Field(description="A 2-D .npy file or an image file in the allowed directory."),
        ] = None,
        x_range: Annotated[
            Optional[List[float]],
            Field(min_length=2, max_length=2, description="[x_min, x_max]. Default [0, 1]."),
        ] = None,
        y_range: Annotated[
            Optional[List[float]],
            Field(min_length=2, max_length=2, description="[y_min, y_max]. Default [0, 1]."),
        ] = None,
        resolution: Annotated[
            Optional[int],
            Field(
                ge=2,
                le=2000,
                description="Cells per side for `expression` (default 200; 80 for surfaces).",
            ),
        ] = None,
        parameters: Annotated[
            Optional[Dict[str, float]], Field(description="Named constants for the expression.")
        ] = None,
        cmap: Annotated[Optional[str], Field(description="Colormap, e.g. 'viridis'.")] = None,
        levels: Annotated[
            Optional[int], Field(ge=2, le=100, description="contour/contourf: number of levels.")
        ] = None,
        vmin: Annotated[
            Optional[float], Field(description="Value mapped to the low end of the colormap.")
        ] = None,
        vmax: Annotated[
            Optional[float], Field(description="Value mapped to the high end of the colormap.")
        ] = None,
        alpha: Annotated[Optional[float], Field(ge=0, le=1, description="Opacity 0-1.")] = None,
        label: Annotated[Optional[str], Field(description="Legend text.")] = None,
        colorbar: Annotated[bool, Field(description="Add a colourbar.")] = False,
        panel: Panel = 1,
    ) -> Any:
        """Draw a 2-D scalar field z(x, y) as an image, contours, or a 3-D surface.

        Give exactly one source: `expression`, `matrix` or `file`.
        """
        return run(
            box.add_field,
            figure_id,
            kind,
            expression,
            matrix,
            file,
            x_range,
            y_range,
            resolution,
            parameters,
            cmap,
            levels,
            vmin,
            vmax,
            alpha,
            label,
            colorbar,
            panel,
        )

    @tool(read_only=True)
    def list_recipes(
        category: Annotated[
            Optional[str],
            Field(description="Only this category: Lines, Scatter, Density, Bars, Fields or 3D."),
        ] = None,
    ) -> Any:
        """List the ready-made recipes: what each draws, the columns (slots) it reads, its look.

        A recipe is a plot kind *and* the look it was designed in (palette, colormap, page,
        3-D camera), so it is the quickest route to a figure that already looks good.
        """
        return run(box.list_recipes, category)

    @tool()
    def add_recipe(
        figure_id: FigureId,
        recipe: Annotated[
            str,
            Field(
                description="Recipe key from list_recipes, "
                "e.g. 'signal_trace', 'starfield', 'surface3d'."
            ),
        ],
        dataset: Annotated[
            Optional[str],
            Field(
                description="Dataset to draw from (create_dataset / load_dataset). "
                "Omit with sample=true."
            ),
        ] = None,
        columns: Annotated[
            Optional[Dict[str, Optional[str]]],
            Field(
                description="Slot -> column overrides, e.g. {'y': 'signal', 'c': 'age'}. "
                "Columns are guessed from their names; use '(index)' for the row number "
                "and null to empty a slot. Only what you want to change."
            ),
        ] = None,
        sample: Annotated[
            bool,
            Field(
                description="Use the recipe's own example table instead of a dataset "
                "(it is added as a dataset)."
            ),
        ] = False,
        look: Annotated[
            str,
            Field(
                description="'recipe' (its own page and palette, default), 'keep' (the figure's "
                "current look) or a style key such as 'journal' or 'dark'."
            ),
        ] = "recipe",
        title: Annotated[Optional[str], Field(description="Panel title. Default: none.")] = None,
        label: Annotated[
            Optional[str],
            Field(description="Legend text for this series. Omit and it stays out of the legend."),
        ] = None,
        panel: Panel = 1,
    ) -> Any:
        """Draw a ready-made recipe (plot kind + look) from a dataset, or from its sample data.

        Axis names come from the column names, the page and colours from the look. Returns an
        artist_id (for remove_artist) and the columns it used. A 3-D recipe needs a figure
        created with projection='3d'. Add titles, limits and a legend with set_axes afterwards.
        """
        return run(
            box.add_recipe, figure_id, recipe, dataset, columns, sample, look, title, label, panel
        )

    @tool()
    def add_annotation(
        figure_id: FigureId,
        kind: Annotated[str, Field(description="One of: " + ", ".join(ANNOTATION_KINDS) + ".")],
        x: Annotated[Optional[float], Field(description="text position / vline position.")] = None,
        y: Annotated[Optional[float], Field(description="text position / hline position.")] = None,
        text: Annotated[Optional[str], Field(description="Text for kind='text'.")] = None,
        color: Color = None,
        linestyle: Annotated[
            Optional[str], Field(description="Line kinds: '-', '--', '-.' or ':'.")
        ] = None,
        linewidth: Annotated[
            Optional[float], Field(gt=0, le=50, description="Line width in px.")
        ] = None,
        fontsize: Annotated[
            Optional[float], Field(gt=0, le=200, description="Text size in px.")
        ] = None,
        label: Annotated[Optional[str], Field(description="Legend text for a line.")] = None,
        panel: Panel = 1,
    ) -> Any:
        """Add a text note, a horizontal reference line (at y) or a vertical one (at x). 2-D only."""  # noqa: E501
        return run(
            box.add_annotation,
            figure_id,
            kind,
            x,
            y,
            text,
            color,
            linestyle,
            linewidth,
            fontsize,
            label,
            panel,
        )

    @tool(destructive=True)
    def remove_artist(
        figure_id: FigureId,
        artist_id: Annotated[
            Optional[str],
            Field(description="Artist to remove, e.g. 'a2'. Omit to remove everything."),
        ] = None,
    ) -> Any:
        """Remove what one add_* call drew (or everything). Use after render_figure shows a mistake."""  # noqa: E501
        return run(box.remove_artist, figure_id, artist_id)

    # -- axes and style -------------------------------------------------------------

    @tool()
    def set_axes(
        figure_id: FigureId,
        panel: Panel = 1,
        title: Annotated[Optional[str], Field(description="Panel title.")] = None,
        xlabel: Annotated[Optional[str], Field(description="x-axis label; include units.")] = None,
        ylabel: Annotated[Optional[str], Field(description="y-axis label; include units.")] = None,
        zlabel: Annotated[Optional[str], Field(description="z-axis label (3-D only).")] = None,
        xlim: Annotated[
            Optional[List[float]], Field(min_length=2, max_length=2, description="[low, high].")
        ] = None,
        ylim: Annotated[
            Optional[List[float]], Field(min_length=2, max_length=2, description="[low, high].")
        ] = None,
        zlim: Annotated[
            Optional[List[float]],
            Field(min_length=2, max_length=2, description="[low, high], 3-D only."),
        ] = None,
        xscale: Annotated[
            Optional[str],
            Field(description="'linear', 'log', 'symlog', 'logit'. Single-panel figures only."),
        ] = None,
        yscale: Annotated[
            Optional[str],
            Field(description="'linear', 'log', 'symlog', 'logit'. Single-panel figures only."),
        ] = None,
        grid: Annotated[Optional[bool], Field(description="Show or hide the grid.")] = None,
        legend: Annotated[
            Optional[bool], Field(description="True to show a legend of labelled series.")
        ] = None,
        equal_aspect: Annotated[
            Optional[bool], Field(description="Same units per pixel on x and y.")
        ] = None,
        hide_axes: Annotated[
            Optional[bool], Field(description="True to hide the axes (images, pies).")
        ] = None,
        elev: Annotated[
            Optional[float], Field(ge=-90, le=90, description="3-D camera elevation, degrees.")
        ] = None,
        azim: Annotated[
            Optional[float], Field(ge=-360, le=360, description="3-D camera azimuth, degrees.")
        ] = None,
    ) -> Any:
        """Set title, axis labels, limits, scales, grid, legend and (3-D) camera on one panel.

        Pass only what you want to change. Always label axes with units.
        """
        return run(
            box.set_axes,
            figure_id,
            panel,
            title,
            xlabel,
            ylabel,
            zlabel,
            xlim,
            ylim,
            zlim,
            xscale,
            yscale,
            grid,
            legend,
            equal_aspect,
            hide_axes,
            elev,
            azim,
        )

    @tool()
    def set_style(
        figure_id: FigureId,
        preset: Annotated[str, Field(description="Preset key, e.g. 'journal', 'dark'.")],
    ) -> Any:
        """Apply a style preset (background, palette, line width, grid) to the whole figure."""
        return run(box.set_style, figure_id, preset)

    # -- output ---------------------------------------------------------------------

    @tool(read_only=True)
    def render_figure(
        figure_id: FigureId,
        scale: Annotated[float, Field(ge=0.25, le=3.0, description="Size multiplier.")] = 1.0,
    ) -> Any:
        """Render the figure and return it as an image plus a text summary of each panel's axes.

        Look at the image and the summary (axis limits, labels, legend) before telling the
        user it is done; fix problems with set_axes / remove_artist / add_*.
        """
        return run(box.render_figure, figure_id, scale)

    @tool()
    def save_figure(
        figure_id: FigureId,
        path: Annotated[
            str,
            Field(
                description="Output file in the allowed directory; "
                "format from extension (.png .svg .pdf .jpg)."
            ),
        ],
        scale: Annotated[
            float, Field(ge=0.25, le=4.0, description="Resolution multiplier (PNG/JPG).")
        ] = 1.0,
        transparent: Annotated[bool, Field(description="Transparent background.")] = False,
        overwrite: Annotated[bool, Field(description="Replace an existing file.")] = False,
    ) -> Any:
        """Save the figure to a file. Refuses to overwrite unless overwrite=true."""
        return run(box.save_figure, figure_id, path, scale, transparent, overwrite)

    # -- resources & prompts ------------------------------------------------------

    @server.resource(
        "glplot://guide",
        name="guide",
        description="How to plot with GLPlot: workflow, per-kind arguments, expression functions.",
        mime_type="text/markdown",
    )
    def guide() -> str:
        return build_guide(box.sandbox, box.limits)

    @server.resource(
        "glplot://figures/{figure_id}/image",
        name="figure-image",
        description="The current rendering of an open figure, as a PNG.",
        mime_type="image/png",
    )
    def figure_image(figure_id: str) -> bytes:
        try:
            return box.render_png(figure_id)
        except McpError as exc:
            raise ToolError(str(exc)) from exc

    @server.resource(
        "glplot://datasets/{name}",
        name="dataset",
        description="A dataset's columns, statistics and first rows, as JSON.",
        mime_type="application/json",
    )
    def dataset_resource(name: str) -> str:
        try:
            return json.dumps(box.describe_dataset(name, 10).data)
        except McpError as exc:
            raise ToolError(str(exc)) from exc

    @server.prompt(
        name="plot_data_file",
        description="Plot a data file and iterate until the figure is publication-ready.",
    )
    def plot_data_file(path: str, goal: str = "") -> str:
        extra = f" The goal: {goal}" if goal else ""
        return (
            f"Load {path!r} with load_dataset, look at its columns with describe_dataset, then "
            "choose the plot that best shows the data. Build the figure with create_figure and "
            "add_plot, add a title, axis labels with units and a legend where useful, and call "
            "render_figure to check it. Fix anything that looks wrong (overlapping labels, "
            "unreadable scale, missing legend) before you answer." + extra
        )

    @server.prompt(
        name="explore_function",
        description="Plot a formula over a range and describe its behaviour.",
    )
    def explore_function(expression: str, x_min: float = -10.0, x_max: float = 10.0) -> str:
        return (
            f"Plot y = {expression} for x from {x_min} to {x_max} using create_figure and "
            "add_function, label the axes, render_figure to look at it, then describe the "
            "shape: zeros, extrema, asymptotes and symmetry. If the default range hides "
            "something interesting, re-plot with a better one."
        )

    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glplot-mcp",
        description="Run GLPlot as a Model Context Protocol server so an LLM can make plots.",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory the model may read data from and save images into (repeatable). "
        "Default: none, which disables file access.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="stdio (default) for a client that launches this process; streamable-http to serve.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="HTTP bind address (default: localhost)."
    )
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765).")
    parser.add_argument("--max-figures", type=int, default=Limits.max_figures)
    parser.add_argument("--version", action="store_true", help="Print the SDK and GLPlot versions.")
    return parser


def _versions() -> str:
    """``mcp X, glplot Y``; GLPlot may be run from a source tree with no installed metadata."""
    import importlib.metadata as metadata

    def version(distribution: str, fallback: str = "unknown") -> str:
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return fallback

    import glplot

    return f"mcp {version('mcp')}, glplot {version('glplot', glplot.__version__)}"


def _warm_up() -> None:
    """Render one throwaway figure before serving.

    The first render in a fresh environment builds matplotlib's font cache, which can take
    many seconds -- long enough that a client sees its first real tool call hang. Paying for
    it here, before the client has been told the server is ready, moves that wait to startup,
    and a broken matplotlib shows up in the log now rather than as a failed first plot. A
    failure is logged, not fatal: the tools report their own errors.
    """
    toolbox = Toolbox()
    try:
        figure = toolbox.create_figure(width=300, height=200).data["figure_id"]
        toolbox.add_plot(figure, "line", y=[0.0, 1.0])
        toolbox.set_axes(figure, title="warm-up")
        toolbox.render_figure(figure)
    except Exception as exc:  # noqa: BLE001 - startup must survive any renderer problem
        logger.warning("render warm-up failed (%s); the first plot may be slow or fail", exc)
    finally:
        toolbox.close_all()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``python -m glplot.mcp`` and the ``glplot-mcp`` script."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="glplot-mcp: %(message)s")
    try:
        _load_sdk()
    except SdkUnavailable as exc:
        print(f"glplot-mcp: {exc}", file=sys.stderr)
        return 2
    if args.version:
        print(_versions(), file=sys.stderr)
        return 0
    try:
        sandbox = FileSandbox(list(args.root))
    except ValueError as exc:
        print(f"glplot-mcp: {exc}", file=sys.stderr)
        return 2
    if args.max_figures < 1:
        print("glplot-mcp: --max-figures must be at least 1", file=sys.stderr)
        return 2
    limits = Limits(max_figures=args.max_figures)
    _warm_up()
    server = build_server(sandbox=sandbox, limits=limits)
    roots = ", ".join(str(r) for r in sandbox.roots) or "none (file access disabled)"
    logger.info("serving over %s; file roots: %s", args.transport, roots)
    if args.transport == "stdio":
        server.run()
    else:
        if args.host not in ("127.0.0.1", "localhost", "::1"):
            logger.warning(
                "binding to %s exposes figure creation and file access to the network "
                "with no authentication",
                args.host,
            )
        try:  # mcp 2.x takes the bind address here
            server.run(transport="streamable-http", host=args.host, port=args.port)
        except TypeError:  # mcp 1.x keeps it in the settings
            server.settings.host = args.host
            server.settings.port = args.port
            server.run(transport="streamable-http")
    return 0
