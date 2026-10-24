Using GLPlot from an LLM (MCP server)
======================================

GLPlot ships a `Model Context Protocol <https://modelcontextprotocol.io>`_ server, so any MCP
client (Claude Desktop, Claude Code, an IDE agent, your own) can have a language model build
plots. The model creates figures, builds datasets, draws, restyles, and **sees every result**:
each render comes back as an image plus a text summary of the axes, so it can check its own work
and iterate the way a person would.

.. code-block:: bash

   pip install "glplot[mcp]"
   python -m glplot.mcp --root ~/data

No window is opened. Figures are rendered by the same headless export ``savefig`` falls back to
when there is no GL context, so the server runs on a laptop, a CI runner or a container.

Connecting a client
-------------------

The client launches the server as a subprocess and talks to it over stdin/stdout.

**Claude Code**

.. code-block:: bash

   claude mcp add glplot -- python -m glplot.mcp --root ~/data

**Claude Desktop** (``claude_desktop_config.json``)

.. code-block:: json

   {
     "mcpServers": {
       "glplot": {
         "command": "python",
         "args": ["-m", "glplot.mcp", "--root", "/Users/me/data"]
       }
     }
   }

Use the full path of the interpreter that has ``glplot[mcp]`` installed if it is not the one on
the client's ``PATH``. The ``glplot-mcp`` console script is the same program.

To serve over HTTP instead (one long-running server, several clients):

.. code-block:: bash

   python -m glplot.mcp --transport streamable-http --port 8765 --root ~/data

It binds to ``127.0.0.1`` by default and has **no authentication**; the server warns if you bind
anywhere else.

What the model can do
---------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Tool
     - Purpose
   * - ``create_figure``
     - A figure, optionally a ``rows`` x ``cols`` grid of panels, 2-D or 3-D, with a style preset.
   * - ``create_dataset``
     - A table from number lists and/or **expressions** over earlier columns
       (``"sin(t) * exp(-t / 5)"``). Large data never has to travel through the conversation.
   * - ``load_dataset``
     - CSV/TSV/TXT, NPY/NPZ, JSON or an image, from inside an allowed directory.
   * - ``describe_dataset`` / ``delete_dataset``
     - Columns, statistics and first rows; free a dataset.
   * - ``add_plot``
     - ``line``, ``scatter`` (colour/size encodings), ``step``, ``stem``, ``bar``, ``barh``,
       ``area``, ``errorbar``, ``hist``, ``box``, ``violin``, ``pie``, ``hist2d``, ``hexbin``;
       ``line`` and ``scatter`` take ``z`` on a 3-D figure.
   * - ``add_function``
     - ``y = f(x)`` over a range in one call, with named parameters.
   * - ``add_field``
     - A 2-D field ``z(x, y)`` as ``image``, ``contour``, ``contourf``, ``pcolormesh``, or on a
       3-D figure ``surface`` / ``wireframe``; from an expression, a small matrix, a ``.npy`` or
       an image file.
   * - ``list_recipes`` / ``add_recipe``
     - The gallery's 22 *recipes*: a plot kind **and** the look it was designed in (page, palette,
       colormap, 3-D camera). ``add_recipe`` draws one from a dataset -- columns are guessed from
       their names and ``columns={'y': 'signal'}`` overrides only what you name -- or from the
       recipe's own example table with ``sample=true``. ``look`` is the recipe's own, ``keep``
       (the figure's current look) or any style key. The fastest route to a figure that already
       looks good.
   * - ``add_annotation``
     - Text, horizontal and vertical reference lines.
   * - ``set_axes`` / ``set_style``
     - Titles, labels, limits, log scales, grid, legend, equal aspect, 3-D camera; style presets.
   * - ``render_figure``
     - A PNG and a summary (resolved limits, labels, legend entries) per panel.
   * - ``describe_figure`` / ``remove_artist`` / ``close_figure``
     - Inspect the figure; undo what one ``add_*`` call drew; free it.
   * - ``save_figure``
     - PNG, SVG, PDF or JPEG inside an allowed directory.

The server also publishes ``glplot://guide`` (a usage guide generated from the code, including
the exact expression vocabulary and a table of every recipe with its slots and look), ``glplot://figures/{id}/image``, ``glplot://datasets/{name}``,
and two prompts, ``plot_data_file`` and ``explore_function``.

A typical exchange, as the model sees it::

   create_figure(title="Damped oscillator")                    -> fig1
   add_function(fig1, "sin(a*x) * exp(-x/6)", 0, 20, parameters={"a": 2})
   set_axes(fig1, xlabel="t (s)", ylabel="amplitude", legend=True)
   render_figure(fig1)                                         -> image + {"xlim": [-1, 21], ...}

Or, in one call, with a designed look::

   create_figure(title="Spiral")                                -> fig1
   add_recipe(fig1, "starfield", sample=true)                   -> columns {x: x_pc, y: y_pc, c: age}
   set_axes(fig1, title="Colour-mapped scatter")
   render_figure(fig1)

A recipe is also an artist, so ``remove_artist`` takes it back out, and it combines with every
other ``add_*`` tool on the same figure (``look='keep'`` leaves the figure's current look, which is
what you want when overlaying a second one).

Every add call returns an *artist id* (``a1``, ``a2``...). One call can create many scene layers
(a bar chart is one patch per bar), so ``remove_artist`` takes the whole group back out.

Safety model
------------

A language model is on the other end, and a document it is reading can try to steer it, so the
server is conservative by default:

* **No code execution.** There is no ``run_python`` tool. Formulas go through the same
  AST-validated evaluator as the workstation's expression box: no attribute access, no builtins,
  no imports. Array constructors (``zeros``, ``arange``, ``linspace``, ``noise``...) are
  size-capped, so ``zeros(10**12)`` is refused before it allocates.
* **File access is confined to a root.** ``--root`` (repeatable) chooses it; without ``--root``
  the root is the directory the server was started in, so pass ``--root`` explicitly rather than
  rely on where a client happens to launch it. Paths are resolved (symlinks included) and must stay
  inside a root; only data and image extensions are accepted; ``save_figure`` only writes images
  and never overwrites unless asked. A client that launches the server from ``/`` gets *no* root
  rather than the whole disk.
* **Bounded.** Inline lists, dataset rows, figure and dataset counts, and image size all have
  limits (see :class:`glplot.mcp.state.Limits`). Errors name the argument and what would have
  been accepted, so the model can correct itself.
* **State is per session and in memory.** Nothing persists between server runs.

Known limitations
-----------------

* Renders use the matplotlib-based export, not the OpenGL pipeline, so effects that exist only
  in the live window (SSAO shading, GPU density accumulation) are not in the images.
* Axis scales (``xscale``/``yscale``) are figure-wide in GLPlot, so they are refused on
  multi-panel figures rather than drawn on the wrong panel.
* A translucent ``area`` shows seams in the export, so ``area`` is opaque unless ``alpha`` is
  given; ``contourf`` is drawn as a continuous ramp rather than discrete bands (the reply says so).

Compatibility
-------------

The server works with both the 1.x (``FastMCP``) and 2.x (``MCPServer``) generations of the
``mcp`` package; ``pip install "glplot[mcp]"`` installs a compatible one. Importing ``glplot`` or
``glplot.mcp`` never imports the SDK, so GLPlot itself does not depend on it.
