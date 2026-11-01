"""Describe the :class:`~glplot.mcp.toolbox.Toolbox` to the MCP protocol.

This is the only module that imports the ``mcp`` SDK, and it adds no behaviour: every tool
here is a typed signature, a description the model reads, and a call into the toolbox. The
signatures are written out rather than generated so the JSON schema a client shows its model
carries a hand-written description per argument -- that text is the model's only manual.

Two SDK generations are supported: ``mcp`` 2.x (``MCPServer``) and 1.x (``FastMCP``). They
differ in a class name and where ``ToolError`` lives; everything else used here is common.
"""

import logging
from typing import Annotated, Any, Dict, List, Optional, Sequence, Union

from pydantic import Field

from . import analysis, catalog
from .cli import build_parser
from .guide import build_instructions
from .state import DEFAULT_LIMITS, FileSandbox, Limits, McpError
from .toolbox import ANNOTATION_KINDS, FIELD_KINDS, PLOT_KINDS, Toolbox, ToolResult

__all__ = ["SdkUnavailable", "build_parser", "build_server", "main"]

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
    box.server_flavor = "mcp-sdk"

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

        Example: create_dataset("run", {"t": "linspace(0, 10, 500)", "v": "sin(t)*exp(-t/5)"}).
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

        The first row is used as column names when it is a header. Text columns become categories
        (labels kept; see select_rows and analyze_dataset 'aggregate') and ISO date columns become
        days since 1970-01-01. A 2-D .npy becomes one column per array column. Returns the
        columns with min/max/mean/std and the first rows.
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

    def analyze_dataset(
        dataset: Annotated[
            str, Field(description="Dataset to analyse (from create_dataset / load_dataset).")
        ],
        operation: Annotated[
            str,
            Field(
                description="One of: "
                + ", ".join(op.name for op in analysis.OPERATIONS)
                + ". get_guide('mcp-analysis') lists each one's options."
            ),
        ],
        y: Annotated[
            Optional[str],
            Field(description="Column to analyse. Default: the only column that is not x."),
        ] = None,
        x: Annotated[
            Optional[str],
            Field(
                description="x column (the abscissa). Default: the row index, i.e. unit spacing. "
                "Required by correlate."
            ),
        ] = None,
        options: Annotated[
            Optional[Dict[str, Any]],
            Field(
                description="Operation-specific options, e.g. {'model': 'exp_decay'} for fit, "
                "{'method': 'savgol', 'window': 31} for smooth, {'cutoff': 5} for filter. Unknown "
                "names are refused with the valid ones listed."
            ),
        ] = None,
        output: Annotated[
            Optional[str],
            Field(
                description="Name for the column (same-length results) or the new dataset (tables) "
                "to create. Default: <y>_<suffix> or <dataset>_<operation>."
            ),
        ] = None,
    ) -> Any:
        return run(box.analyze_dataset, dataset, operation, y, x, options, output)

    # The operation list is generated from the registry, so the description cannot drift from it.
    analyze_dataset.__doc__ = (
        "Analyse a dataset numerically and keep the result so it can be plotted. Operations: "
        + analysis.operations_summary()
        + ".\n\n"
        "A 'column' result is added to the dataset (e.g. v_fit), a 'table' result becomes a new "
        "dataset (e.g. run_fft), a 'report' only returns numbers. The reply includes a plot_hint "
        "with the add_plot call that draws it. fit supports linear, polynomial, gaussian, "
        "lorentzian, exp_decay, power_law, sigmoid, multipeak and custom expressions, and reports "
        "parameters with standard errors and R^2. Options per operation: "
        "get_guide('mcp-analysis').\n\n"
        'Examples: analyze_dataset("run", "fit", x="t", y="v", options={"model": "exp_decay"}); '
        'analyze_dataset("signal", "fft", x="t", y="v"); '
        'analyze_dataset("run", "smooth", y="v", options={"window": 31}).'
    )
    tool()(analyze_dataset)

    @tool()
    def select_rows(
        dataset: Annotated[str, Field(description="Dataset to take rows from.")],
        where: Annotated[
            Optional[str],
            Field(
                description="Keep rows where this expression over the columns is true, e.g. "
                "'temp > 10' or '(temp > 10) & (year == 2021)' (use & | ~ with parentheses, "
                "not and/or/not)."
            ),
        ] = None,
        equals: Annotated[
            Optional[Dict[str, Any]],
            Field(
                description="Keep rows whose column equals a value or any of a list: "
                "{'city': 'Lima'} or {'city': ['Lima', 'Oslo']} for a text (category) column, "
                "{'year': 2021} for a numeric one."
            ),
        ] = None,
        sort_by: Annotated[
            Optional[str], Field(description="Column to sort the kept rows by (NaN last).")
        ] = None,
        descending: Annotated[bool, Field(description="Sort from largest to smallest.")] = False,
        every: Annotated[
            Optional[int], Field(ge=1, description="Keep every n-th of the remaining rows.")
        ] = None,
        limit: Annotated[
            Optional[int], Field(ge=1, description="Keep only the first n rows (after sorting).")
        ] = None,
        output: Annotated[
            Optional[str], Field(description="Name of the new dataset. Default: <dataset>_rows.")
        ] = None,
    ) -> Any:
        """Filter, sort or thin out a dataset's rows into a new dataset (the original is kept).

        Conditions combine: where and equals first, then sort_by, every and limit. Text columns
        keep their labels. Fails, saying so, when no row matches. Examples:
        select_rows("weather", equals={"city": "Lima"});
        select_rows("run", where="v > 0", sort_by="t");
        select_rows("scores", sort_by="score", descending=True, limit=10).
        """
        return run(
            box.select_rows, dataset, where, equals, sort_by, descending, every, limit, output
        )

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
        hatch: Annotated[
            Optional[str],
            Field(
                description="bar/barh/area/hist fill pattern: '/', 'x', '+', '.', 'o', '//' "
                "(a backslash pattern works too)."
            ),
        ] = None,
        axis: Annotated[
            str,
            Field(
                description="'left' (default) or 'right': draw against a second y axis on the "
                "right (line, scatter, step, bar, area, errorbar). Label it with "
                "set_axes(y2label=..., y2lim=...)."
            ),
        ] = "left",
        reveal: Annotated[
            Optional[float],
            Field(
                ge=0,
                le=1,
                description="Show only the first fraction (0-1) of the points: a curve drawing "
                "itself in. Bind it to time in animate. line/scatter/step/bar/area/errorbar.",
            ),
        ] = None,
        trail: Annotated[
            Optional[int],
            Field(ge=1, description="Show only the last N of the revealed points: a moving tail."),
        ] = None,
        head: Annotated[
            bool,
            Field(description="Put a dot on the last shown point (line/scatter/step): the 'head'."),
        ] = False,
    ) -> Any:
        """Draw data as a line, scatter, bar, histogram, box, pie, ... and return an artist_id.

        Each kind needs different series; the per-kind table is in glplot://guide. Options a
        kind does not use are ignored and listed in the reply. Returns the artist_id used by
        remove_artist.

        Examples: add_plot("fig1", "scatter", x=[1, 2, 3], y=[2, 4, 9], color="red");
        add_plot("fig1", "line", dataset="run", x="t", y="v", label="run");
        add_plot("fig1", "hist", dataset="run", y="v", bins=30).
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
            hatch,
            axis,
            reveal,
            trail,
            head,
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
        reveal: Annotated[
            Optional[float],
            Field(
                ge=0,
                le=1,
                description="Draw only the first share of the curve (0..1); bind it, or use the "
                "create effect, and the graph draws itself.",
            ),
        ] = None,
        head: Annotated[bool, Field(description="A dot on the last point drawn.")] = False,
    ) -> Any:
        """Plot y = expression(x) over [x_min, x_max] in one call, without making a dataset.

        The quickest way to draw a formula or to overlay several (call it once per curve).

        Example: add_function("fig1", "sin(a*x)/x", x_min=-10, x_max=10, parameters={"a": 2}).
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
            reveal,
            head,
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
        dx: Annotated[
            Optional[float], Field(description="kind='arrow': tip minus tail along x (data units).")
        ] = None,
        dy: Annotated[
            Optional[float], Field(description="kind='arrow': tip minus tail along y (data units).")
        ] = None,
        alpha: Annotated[
            Optional[float],
            Field(ge=0, le=1, description="Opacity 0..1 (fade it by binding this)."),
        ] = None,
        reveal: Annotated[
            Optional[float],
            Field(
                ge=0,
                le=1,
                description="Text: the share of it written so far (a $...$ formula appears whole); "
                "arrow: the share of its length grown. Bind it, or use the write/grow effects.",
            ),
        ] = None,
        coords: Annotated[
            str,
            Field(
                description="Text position in 'data' units or in 'axes' fractions (0..1 of the "
                "box: x=0.5, y=0.92 is a centred caption that stays put).",
            ),
        ] = "data",
        align: Annotated[
            str, Field(description="Text alignment at x: 'left', 'center' or 'right'.")
        ] = "left",
        background: Annotated[
            bool,
            Field(
                description="A box in the page colour behind the text, hiding grid and axis "
                "lines under it (manim's BackgroundRectangle)."
            ),
        ] = False,
    ) -> Any:
        """Add text (with $...$ maths), a horizontal/vertical line, or an arrow (a vector from x, y to x+dx, y+dy). 2-D only. Example: add_annotation('fig1', 'text', x=0.5, y=0.9, coords='axes', align='center', text='$f(x) = x^2$', fontsize=30)."""  # noqa: E501
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
            dx,
            dy,
            alpha,
            reveal,
            coords,
            align,
            background,
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
        y2label: Annotated[
            Optional[str],
            Field(description="Label of the right-hand y axis (after add_plot(axis='right'))."),
        ] = None,
        y2lim: Annotated[
            Optional[List[float]],
            Field(min_length=2, max_length=2, description="[low, high] of the right-hand y axis."),
        ] = None,
        y2scale: Annotated[
            Optional[str], Field(description="Scale of the right-hand y axis: 'linear', 'log'...")
        ] = None,
    ) -> Any:
        """Set title, axis labels, limits, scales, grid, legend and (3-D) camera on one panel.

        Pass only what you want to change. Always label axes with units.

        Example: set_axes("fig1", title="Decay", xlabel="t (s)", ylabel="signal (V)", yscale="log",
        legend=True).
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
            y2label,
            y2lim,
            y2scale,
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
        params: Annotated[
            Optional[Dict[str, Any]],
            Field(
                description="Control values to render at, e.g. {'amp': 2, 'mode': 'cos'} "
                "(see add_control). The figure itself keeps its own values."
            ),
        ] = None,
    ) -> Any:
        """Render the figure and return it as an image plus a text summary of each panel's axes.

        Look at the image and the summary (axis limits, labels, legend) before telling the
        user it is done; fix problems with set_axes / remove_artist / add_*.

        The reply also reviews the figure and lists `issues` it can prove (missing axis labels or
        title, an empty panel, overlapping tick labels, points a log axis cannot draw, data cut off
        by the limits...), each with a severity and the call that fixes it. An empty list means
        nothing was found.
        """
        return run(box.render_figure, figure_id, scale, params)

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
        params: Annotated[
            Optional[Dict[str, Any]],
            Field(
                description="Control values to render at, e.g. {'amp': 2, 'mode': 'cos'} "
                "(see add_control). The figure itself keeps its own values."
            ),
        ] = None,
    ) -> Any:
        """Save the figure to a file. Refuses to overwrite unless overwrite=true."""
        return run(box.save_figure, figure_id, path, scale, transparent, overwrite, params)

    # -- change what is drawn, controls, animation, scripts ---------------------------

    @tool()
    def update_artist(
        figure_id: FigureId,
        artist_id: Annotated[str, Field(description="Artist to change, e.g. 'a2'.")],
        changes: Annotated[
            Dict[str, Any],
            Field(
                description="Arguments of the call that drew it, with their new values: "
                "{'color': 'tab:red', 'linewidth': 3}. null puts one back to its default; "
                "'parameters.a' reaches inside an object argument; {} redraws from the data."
            ),
        ],
    ) -> Any:
        """Change something already drawn without removing it: colour, width, columns, label...

        The artist keeps its id and its place in the stacking order. `describe_figure` lists the
        artists; the reply shows the recipe (the arguments now in force).

        Examples: update_artist("fig1", "a1", {"color": "tab:green", "linewidth": 3});
        update_artist("fig1", "a2", {"y": "v_fit"});
        update_artist("fig1", "a1", {"parameters.a": 4}).
        """
        return run(box.update_artist, figure_id, artist_id, changes)

    @tool()
    def add_control(
        figure_id: FigureId,
        name: Annotated[
            str,
            Field(description="Identifier used in expressions (e.g. 'amp'). Not t, s, time, x, y."),
        ],
        kind: Annotated[
            str,
            Field(
                description="'slider' (needs min and max), 'toggle' (true/false -> 1/0), "
                "'choice' (needs options; its value in expressions is the option's index) or "
                "'button' (sets other controls, or resets them all)."
            ),
        ] = "slider",
        min: Annotated[Optional[float], Field(description="Slider minimum.")] = None,  # noqa: A002
        max: Annotated[Optional[float], Field(description="Slider maximum.")] = None,  # noqa: A002
        value: Annotated[
            Any, Field(description="Starting value (slider/toggle/choice). Default: the middle.")
        ] = None,
        step: Annotated[Optional[float], Field(gt=0, description="Slider snap step.")] = None,
        options: Annotated[
            Optional[List[str]], Field(description="choice: two or more option names.")
        ] = None,
        label: Annotated[Optional[str], Field(description="Text shown beside it.")] = None,
        integer: Annotated[bool, Field(description="Slider takes whole numbers only.")] = False,
        log: Annotated[bool, Field(description="Slider spans decades (min > 0).")] = False,
        sets: Annotated[
            Optional[Dict[str, Any]],
            Field(description="button: {control: value} pairs it applies; omit to reset all."),
        ] = None,
    ) -> Any:
        """Define a slider, toggle, choice or button: a named value the figure can depend on.

        A control does nothing until a property follows it: bind_property(figure, 'a1',
        'linewidth', 'width'). Then set_control moves it and the figure is redrawn, render_figure
        (params={...}) tries values without keeping them, animate can sweep it, and
        export_script(kind='interactive') writes the real widgets as Python.

        Example: add_control("fig1", "amp", min=0, max=5, value=1).
        """
        return run(
            box.add_control, figure_id, name, kind, min, max, value, step, options, label,
            integer, log, sets,
        )  # fmt: skip

    @tool()
    def set_control(
        figure_id: FigureId,
        name: Annotated[str, Field(description="Control name.")],
        value: Annotated[
            Any,
            Field(
                description="New value (slider number, toggle true/false, choice option name). "
                "Omit for a button."
            ),
        ] = None,
    ) -> Any:
        """Move a control (the figure is redrawn at the new value) or press a button."""
        return run(box.set_control, figure_id, name, value)

    @tool()
    def bind_property(
        figure_id: FigureId,
        target: Annotated[
            str,
            Field(
                description="What to drive: an artist id ('a1'); 'axes' or 'axes:2' (panel "
                "title, xlabel, xlim, ylim, zlim, elev, azim, legend...); or 'data:<dataset>' "
                "(then `property` is a column recomputed from the others)."
            ),
        ],
        property: Annotated[  # noqa: A002
            str,
            Field(
                description="The argument to drive: 'color', 'linewidth', 'alpha', 'size', "
                "'reveal', 'trail', 'parameters.a' (a function's parameter), 'x_max', 'azim'..."
            ),
        ],
        value: Annotated[
            Any,
            Field(
                description="A number, an expression of controls and time ('1 + 2*amp', "
                "'sin(2*pi*t)'), a list of those ('xlim': ['0', '10*t']), "
                "{'colormap': 'viridis'|[colours], 'at': expr}, {'choose': [...], 'index': expr} "
                "or {'format': 'phase {time:.1f} s'}. null removes the binding."
            ),
        ] = None,
    ) -> Any:
        """Make one property follow an expression of the figure's controls (and of time).

        Bindings apply on every redraw and in `animate`. Names in the expression are control
        names; inside animate also t (0..1), time (s), frame, s (stage progress) and variables.

        Examples: bind_property("fig1", "a1", "parameters.amp", "amp");
        bind_property("fig1", "axes", "title", {"format": "amp = {amp:.1f}"}).
        """
        return run(box.bind_property, figure_id, target, property, value)

    @tool()
    def animate(
        figure_id: FigureId,
        output: Annotated[
            Optional[str],
            Field(
                description="File to write in the allowed directory: .gif, .apng, .mp4 or "
                ".webm (those two need ffmpeg), or 'frames/f_%04d.png' for numbered images. "
                "Omit to only preview: a contact sheet of frames is always returned."
            ),
        ] = None,
        frames: Annotated[
            Optional[int], Field(ge=2, le=600, description="Number of frames (or give duration).")
        ] = None,
        duration: Annotated[
            Optional[float], Field(gt=0, description="Length in seconds (frames = duration*fps).")
        ] = None,
        fps: Annotated[float, Field(ge=1, le=120, description="Frames per second.")] = 30.0,
        loop: Annotated[
            bool,
            Field(
                description="True for a seamless loop: t runs 0..1 excluding the end, so the "
                "last frame leads into the first. Use whole periods (2*pi*t)."
            ),
        ] = False,
        stages: Annotated[
            Optional[List[Dict[str, Any]]],
            Field(
                description="Consecutive stages, each {'name', 'duration' (s) or 'frames', "
                "'ease', 'animate': {...}}. `animate` moves variables: {'amp': [1, 3]} (from, to), "
                "{'amp': 3} (to, from wherever it is), {'amp': {'from': 1, 'to': 3, 'ease': "
                "'in_out'}} or {'k': '2 + 3*s'} (expression of s, 0..1 across the stage). A stage "
                "starts where the last ended and ends exactly on its target; afterwards the value "
                "holds. Easings: linear, in, out, in_out, smooth, smoother (manim's smooth), "
                "sine, rush_into, rush_from, there_and_back. Step by step like a manim scene: "
                "'effects': {'create': 'a1', 'write': ['a2'], 'fade_in': 'a3', 'indicate': 'a2', "
                "'fade_out': 'a4'} (also uncreate, unwrite, grow) -- an artist brought in by an "
                "effect is hidden until its stage; see get_guide('mcp-animation')."
            ),
        ] = None,
        variables: Annotated[
            Optional[Dict[str, Any]],
            Field(
                description="Named values: a NUMBER is a parameter (its starting value; stages "
                "may move it), an expression STRING is derived each frame from t, time, s, stage "
                "and the parameters: {'amp': 1, 'phase': '2*pi*t', 'w': '2*pi*f'}."
            ),
        ] = None,
        bindings: Annotated[
            Optional[List[Dict[str, Any]]],
            Field(
                description="What the variables drive, [{'target', 'property', 'value'}] -- the "
                "same as bind_property (added to the figure's own bindings). value is an "
                "expression of the variables: {'target': 'a1', 'property': 'parameters.phase', "
                "'value': 'phase'}; {'target': 'axes', 'property': 'azim', 'value': '30+360*t'}."
            ),
        ] = None,
        scale: Annotated[
            Optional[float],
            Field(ge=0.25, le=3.0, description="Resolution multiplier. Default: fit ~960 px."),
        ] = None,
        hold_start: Annotated[
            float, Field(ge=0, le=30, description="Seconds to hold the first frame.")
        ] = 0.0,
        hold_end: Annotated[
            float, Field(ge=0, le=30, description="Seconds to hold the last frame.")
        ] = 0.0,
        sheet_frames: Annotated[
            int, Field(ge=1, le=16, description="Frames in the contact sheet returned.")
        ] = 8,
        frame: Annotated[
            Optional[int],
            Field(ge=0, description="Render only this frame, full size (nothing is written)."),
        ] = None,
        overwrite: Annotated[bool, Field(description="Replace an existing file.")] = False,
        repeat: Annotated[
            int,
            Field(ge=0, le=1000, description="GIF/APNG: extra plays after the first (0=forever)."),
        ] = 0,
        quality: Annotated[str, Field(description="mp4/webm: 'draft', 'good' or 'high'.")] = "good",
    ) -> Any:
        """Animate the figure: properties follow time, in stages, and save a GIF/MP4/WebM/PNGs.

        You cannot watch an animation, so the reply is a contact sheet (frames side by side) plus
        measured facts: variable ranges per stage and `issues` (nothing moves, the axes rescale,
        the loop does not close, an abrupt jump). Build the figure first, pin the axes
        (set_axes xlim/ylim), then say what varies: variables (+ stages) and bindings.
        Preview with output omitted, fix, then write the file. Read get_guide('mcp-animation').

        Example, a wave that travels, grows, then squeezes:
        animate("fig1", output="wave.gif", fps=25, variables={"phase": "2*pi*t", "amp": 1, "k": 2},
        stages=[{"name": "travel", "duration": 2}, {"name": "grow", "duration": 2,
        "animate": {"amp": [1, 3]}}, {"name": "squeeze", "duration": 2, "animate": {"k": [2, 5]}}],
        bindings=[{"target": "a1", "property": "parameters.phase", "value": "phase"},
        {"target": "a1", "property": "parameters.amp", "value": "amp"},
        {"target": "a1", "property": "parameters.k", "value": "k"}]).
        3-D orbit: bindings=[{"target": "axes", "property": "azim", "value": "30 + 360*t"}].
        """
        return run(
            box.animate, figure_id, output, frames, duration, fps, loop, stages, variables,
            bindings, scale, hold_start, hold_end, sheet_frames, frame, overwrite, repeat, quality,
        )  # fmt: skip

    @tool()
    def compose_video(
        segments: Annotated[
            List[Any],
            Field(
                description="The scenes in order. Each is a figure id ('fig1': its last "
                "animate) or an object {'figure_id': 'fig1', 'hold_end': 1} (extra seconds on "
                "its last frame). A figure that was never animated is a still: "
                "{'figure_id': 'fig4', 'duration': 3}."
            ),
        ],
        output: Annotated[
            Optional[str],
            Field(
                description="The video file: .mp4 (best for anything long; needs ffmpeg), .webm, "
                ".gif or .apng. Omit to preview: a contact sheet of every scene is returned."
            ),
        ] = None,
        fps: Annotated[
            Optional[float],
            Field(ge=1, le=60, description="Frame rate; default the fastest scene's."),
        ] = None,
        transition: Annotated[
            str,
            Field(
                description="Between scenes: 'fade' (out to the page colour and back in), "
                "'crossfade' (the scenes overlap and blend) or 'cut'."
            ),
        ] = "fade",
        transition_s: Annotated[
            float, Field(ge=0, le=5, description="Length of each transition in seconds.")
        ] = 0.6,
        size: Annotated[
            Optional[List[int]],
            Field(
                min_length=2,
                max_length=2,
                description="[width, height] of the video; default the first scene's. Scenes "
                "of another shape are letterboxed (give all the same create_figure size).",
            ),
        ] = None,
        audio: Annotated[
            Optional[str],
            Field(
                description="A soundtrack (.mp3, .wav, .m4a, .ogg...) inside the allowed "
                "directory, muxed into an .mp4/.webm and cut to the video's length."
            ),
        ] = None,
        quality: Annotated[str, Field(description="'draft', 'good' or 'high'.")] = "good",
        repeat: Annotated[
            int, Field(ge=0, le=1000, description="GIF/APNG: extra plays (0: forever).")
        ] = 0,
        overwrite: Annotated[bool, Field(description="Replace an existing file.")] = False,
        sheet_frames: Annotated[
            int, Field(ge=1, le=16, description="Frames in the contact sheet.")
        ] = 12,
    ) -> Any:
        """Cut several scenes (animated figures) together into ONE video: a complete explainer.

        Build each scene as its own figure (same width x height for all), animate and preview it
        until it is right, then compose. Frames animate rendered are reused, so composing is
        mostly encoding; a scene changed since is re-rendered. The reply lists each scene's start
        and end time. Workflow: get_guide('mcp-video').

        Example: compose_video(["fig1", {"figure_id": "fig2", "hold_end": 1}, "fig3"],
        output="explainer.mp4", transition="fade").
        """
        return run(
            box.compose_video, segments, output, fps, transition, transition_s, size, audio,
            quality, repeat, overwrite, sheet_frames,
        )  # fmt: skip

    @tool()
    def simulate(
        name: Annotated[str, Field(description="Name of the dataset to create.")],
        equations: Annotated[
            Dict[str, str],
            Field(
                description="Each state's derivative as an expression of the states, t and the "
                "parameters: {'theta': 'omega', 'omega': '-g/L*sin(theta) - c*omega'}."
            ),
        ],
        initial: Annotated[Dict[str, float], Field(description="Starting value of every state.")],
        t_end: Annotated[float, Field(description="Final time.")],
        t_start: Annotated[float, Field(description="Initial time.")] = 0.0,
        steps: Annotated[
            int, Field(ge=2, le=200000, description="Rows returned (evenly spaced in t).")
        ] = 1000,
        parameters: Annotated[
            Optional[Dict[str, float]], Field(description="Constants: {'g': 9.81, 'L': 1}.")
        ] = None,
        derived: Annotated[
            Optional[Dict[str, str]],
            Field(
                description="Extra columns computed from the solution, e.g. "
                "{'energy': '0.5*omega**2 + g/L*(1-cos(theta))'}."
            ),
        ] = None,
        method: Annotated[
            str,
            Field(
                description="'rk45' (default), 'dop853' (accurate), 'lsoda'/'radau'/'bdf' "
                "(stiff), 'rk23', or 'rk4' (fixed step)."
            ),
        ] = "rk45",
        substeps: Annotated[
            int, Field(ge=1, le=200, description="rk4: internal steps between rows.")
        ] = 4,
    ) -> Any:
        """Integrate ordinary differential equations into a dataset (t, each state, derived).

        For dynamics that are not a formula of t: a pendulum, predator-prey, Lorenz, an orbit, a
        driven oscillator. Plot the result, and animate it with add_plot(reveal=..., head=true)
        bound to time. Equations use the same safe expression language as everywhere.

        Example: simulate("lorenz", {"x": "s*(y-x)", "y": "x*(r-z)-y", "z": "x*y-b*z"},
        {"x": 1, "y": 1, "z": 1}, t_end=40, parameters={"s": 10, "r": 28, "b": 2.667}).
        """
        return run(
            box.simulate, name, equations, initial, t_end, t_start, steps, parameters, derived,
            method, substeps,
        )  # fmt: skip

    @tool(read_only=True)
    def export_script(
        figure_id: FigureId,
        kind: Annotated[
            str,
            Field(
                description="'figure' (the figure as it is now, saved to PNG), 'interactive' "
                "(controls become real sliders/toggles/menus/buttons that redraw it) or "
                "'animation' (the last animate call as a FuncAnimation saved to GIF)."
            ),
        ] = "figure",
    ) -> Any:
        """Return the Python script that rebuilds this figure, to give the user to keep and edit.

        Generated from the calls that built the figure; datasets are rebuilt from how they were
        made (expressions -> numpy, simulate -> solve_ivp). The script is returned as text in the
        reply (nothing is written to disk). Run it with python; it needs only glplot.
        """
        return run(box.export_script, figure_id, kind)

    # -- knowledge: learn the library on demand ------------------------------------

    @tool(read_only=True)
    def get_guide(
        topic: Annotated[
            Optional[str],
            Field(
                description="Topic id, e.g. 'quickstart', 'three-d', 'animation', "
                "'mcp-plot-kinds'. "
                "Omit to list every topic."
            ),
        ] = None,
    ) -> Any:
        """Read GLPlot's documentation: a guide topic as Markdown, or the list of topics.

        Topics starting `mcp-` explain these tools; the others cover the Python API (plot types,
        axes and styling, 3-D, animation, controls, performance, differences from matplotlib,
        troubleshooting). Read the relevant one before writing a script or choosing a plot kind.
        """
        return run(box.get_guide, topic)

    @tool(read_only=True)
    def search_api(
        query: Annotated[
            str,
            Field(
                description="What you want to do or a name: 'heatmap', 'second y axis', 'error "
                "bars', 'save gif'. Empty lists every category."
            ),
        ] = "",
        limit: Annotated[int, Field(ge=1, le=40, description="Most matches to return.")] = 8,
        category: Annotated[
            Optional[str],
            Field(
                description="Only this category id (figure, plot2d, fields, plot3d, axes, "
                "annotate, "
                "color, controls, output, signal, perf, interaction, compat)."
            ),
        ] = None,
    ) -> Any:
        """Find GLPlot Python functions by name or by what they do (about 250 functions).

        Returns each match's call shape and one-line summary. Follow up with get_api_doc for the
        full docstring. Useful before writing a script, or to answer "can GLPlot do X?".
        """
        return run(box.search_api, query, limit, category)

    @tool(read_only=True)
    def get_api_doc(
        name: Annotated[
            str,
            Field(
                description="Function or class name, e.g. 'hist2d', 'plt.twinx', 'FuncAnimation'."
            ),
        ],
    ) -> Any:
        """The exact signature and full docstring of one GLPlot function (read from the code).

        An unknown name returns near-miss suggestions.
        """
        return run(box.get_api_doc, name)

    @tool(read_only=True)
    def get_example(
        name: Annotated[
            Optional[str],
            Field(description="Example id, e.g. 'line_plot', 'surface_3d'. Omit to list them."),
        ] = None,
    ) -> Any:
        """A complete runnable Python script using GLPlot, or the list of examples.

        Every example ends in plt.savefig, so it runs with no display. Use one as the starting
        point for a script rather than writing it from memory.
        """
        return run(box.get_example, name)

    @tool(read_only=True)
    def server_info() -> Any:
        """Versions, what this server may do (file roots, limits), and what is open now."""
        return run(box.server_info)

    # -- resources & prompts ------------------------------------------------------
    def serve(template: str, **params: str) -> Any:
        """Read the catalog resource ``template``; its ``McpError`` is what the client sees."""
        definition = next(d for d in catalog.RESOURCES if d.uri == template)
        try:
            return definition.read(box, params)
        except McpError as exc:
            raise ToolError(str(exc)) from exc

    def resource(template: str) -> Any:
        definition = next(d for d in catalog.RESOURCES if d.uri == template)
        return server.resource(
            definition.uri,
            name=definition.name,
            description=definition.description,
            mime_type=definition.mime_type,
        )

    @resource("glplot://guide")
    def guide() -> str:
        return serve("glplot://guide")

    @resource("glplot://guide/{topic}")
    def guide_topic(topic: str) -> str:
        return serve("glplot://guide/{topic}", topic=topic)

    @resource("glplot://api")
    def api_reference() -> str:
        return serve("glplot://api")

    @resource("glplot://api/{name}")
    def api_doc(name: str) -> str:
        return serve("glplot://api/{name}", name=name)

    @resource("glplot://examples")
    def examples_index() -> str:
        return serve("glplot://examples")

    @resource("glplot://examples/{name}")
    def example_code(name: str) -> str:
        return serve("glplot://examples/{name}", name=name)

    @resource("glplot://llms.txt")
    def llms_txt() -> str:
        return serve("glplot://llms.txt")

    @resource("glplot://llms-full.txt")
    def llms_full_txt() -> str:
        return serve("glplot://llms-full.txt")

    @resource("glplot://capabilities")
    def capabilities() -> str:
        return serve("glplot://capabilities")

    @resource("glplot://recipes")
    def recipes_resource() -> str:
        return serve("glplot://recipes")

    @resource("glplot://styles")
    def styles_resource() -> str:
        return serve("glplot://styles")

    @resource("glplot://figures/{figure_id}/image")
    def figure_image(figure_id: str) -> bytes:
        return serve("glplot://figures/{figure_id}/image", figure_id=figure_id)

    @resource("glplot://datasets/{name}")
    def dataset_resource(name: str) -> str:
        return serve("glplot://datasets/{name}", name=name)

    def prompt(name: str) -> Any:
        definition = catalog.find_prompt(name)
        return server.prompt(name=definition.name, description=definition.description)

    @prompt("plot_data_file")
    def plot_data_file(path: str, goal: str = "") -> str:
        return catalog.find_prompt("plot_data_file").build(path=path, goal=goal)

    @prompt("explore_function")
    def explore_function(expression: str, x_min: float = -10.0, x_max: float = 10.0) -> str:
        return catalog.find_prompt("explore_function").build(
            expression=expression, x_min=x_min, x_max=x_max
        )

    @prompt("quick_look")
    def quick_look(dataset: str, question: str = "") -> str:
        return catalog.find_prompt("quick_look").build(dataset=dataset, question=question)

    @prompt("fix_my_figure")
    def fix_my_figure(figure_id: str, problem: str = "") -> str:
        return catalog.find_prompt("fix_my_figure").build(figure_id=figure_id, problem=problem)

    @prompt("publication_figure")
    def publication_figure(figure_id: str, venue: str = "a scientific journal") -> str:
        return catalog.find_prompt("publication_figure").build(figure_id=figure_id, venue=venue)

    @prompt("compare_series")
    def compare_series(dataset: str, columns: str = "") -> str:
        return catalog.find_prompt("compare_series").build(dataset=dataset, columns=columns)

    @prompt("fit_and_plot")
    def fit_and_plot(dataset: str, x: str, y: str, model: str = "") -> str:
        return catalog.find_prompt("fit_and_plot").build(dataset=dataset, x=x, y=y, model=model)

    @prompt("animate_this")
    def animate_this(description: str, figure_id: str = "", output: str = "") -> str:
        return catalog.find_prompt("animate_this").build(
            description=description, figure_id=figure_id, output=output
        )

    @prompt("interactive_figure")
    def interactive_figure(figure_id: str, parameters: str = "") -> str:
        return catalog.find_prompt("interactive_figure").build(
            figure_id=figure_id, parameters=parameters
        )

    @prompt("python_script_for")
    def python_script_for(task: str) -> str:
        return catalog.find_prompt("python_script_for").build(task=task)

    @prompt("explainer_video")
    def explainer_video(topic: str, length: str = "", output: str = "", language: str = "") -> str:
        return catalog.find_prompt("explainer_video").build(
            topic=topic, length=length, output=output, language=language
        )

    return server


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point kept for ``from glplot.mcp.server import main``; the CLI lives in :mod:`.cli`."""
    from .cli import main as cli_main

    return cli_main(argv)
