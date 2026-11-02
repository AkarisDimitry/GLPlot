Using GLPlot from an LLM (MCP server)
======================================

GLPlot ships a `Model Context Protocol <https://modelcontextprotocol.io>`_ server, so any MCP
client (Claude Desktop, Claude Code, an IDE agent, your own) can have a language model build
plots. The model creates figures, builds datasets, draws, restyles, and **sees every result**:
each render comes back as an image plus a text summary of the axes, so it can check its own work
and iterate the way a person would.

.. code-block:: bash

   pip install glplot
   glplot-mcp --root ~/data           # or: python -m glplot.mcp --root ~/data

That is all it takes: without the optional ``mcp`` package GLPlot serves the protocol with a small
**built-in stdio server** (standard library only), so an MCP client can launch it right after a
plain ``pip install glplot``. ``pip install "glplot[mcp]"`` additionally installs the official SDK,
which is then preferred and adds the HTTP transport. ``glplot-mcp doctor`` checks the install and
runs a protocol self-test.

No window is opened. Figures are rendered by the same headless export ``savefig`` falls back to
when there is no GL context, so the server runs on a laptop, a CI runner or a container.

Connecting a client
-------------------

The client launches the server as a subprocess and talks to it over stdin/stdout. The quickest way
to register it is to let GLPlot write the settings, with this interpreter's absolute path:

.. code-block:: bash

   glplot-mcp config --client claude-desktop --root ~/data           # print it
   glplot-mcp config --client claude-desktop --root ~/data --write   # merge into the file, keep a .bak
   glplot-mcp config                                                 # every client: claude-code,
                                                                     # claude-desktop, cursor, vscode, codex

``--write`` handles the JSON-file clients (Claude Desktop, Cursor, VS Code); it keeps everything else in
the file, leaves a ``.bak`` copy and refuses a file it cannot parse. By hand:

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

Use the full path of the interpreter that has GLPlot installed if it is not the one on the client's
``PATH`` (``glplot-mcp config`` does this for you). The ``glplot-mcp`` console script is the same program.

To serve over HTTP instead (one long-running server, several clients):

.. code-block:: bash

   python -m glplot.mcp --transport streamable-http --port 8765 --root ~/data

It needs the official SDK (``pip install "glplot[mcp]"``) and binds to ``127.0.0.1`` by default. To serve
other machines, set a bearer token; GLPlot **refuses to listen on a non-localhost address without one**:

.. code-block:: bash

   export GLPLOT_MCP_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(24))')"
   python -m glplot.mcp --transport streamable-http --host 0.0.0.0 --port 8765 --root ~/data
   # clients send:  Authorization: Bearer <token>

* Every request must carry ``Authorization: Bearer <token>`` or it gets a 401; the comparison is
  constant-time and the token is never logged. Prefer the environment variable to ``--token``, which shows
  up in the process list. Tokens shorter than 8 characters are refused.
* The SDK's DNS-rebinding check (mcp 1.x rejects any ``Host`` other than ``localhost`` with a 421) is switched
  off behind a token on a network bind, so clients can use the machine's name or address. To keep it on, list
  the names clients may use: ``--allow-host plots.example.org`` (repeatable; localhost is always allowed).
* ``--insecure`` allows a network bind with no token; use it only on a network you trust, since anyone who can
  reach the port can then create figures and use the allowed file roots.
* TLS is not terminated here: put a reverse proxy in front for HTTPS.

Register an HTTP server in a client with its URL and the header, for example for Claude Code
``claude mcp add --transport http glplot http://host:8765/mcp --header "Authorization: Bearer <token>"``.

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
   * - ``select_rows``
     - Filter (``where='temp > 10'``, ``equals={'city': 'Lima'}``), sort, thin out or truncate a dataset
       into a new one; text columns keep their labels.
   * - ``analyze_dataset``
     - **Numerical analysis that keeps its result**: ``fit`` (linear, polynomial, gaussian, lorentzian,
       exp_decay, power_law, sigmoid, multipeak, or any ``custom`` expression such as ``a*exp(-x/b)+c``,
       with parameter standard errors and R\ :sup:`2`), ``smooth``, ``rolling``, ``derivative``,
       ``integrate``, ``fft``, ``peaks``, ``filter`` (Butterworth and other IIR), ``detrend``,
       ``normalize``, ``resample``, ``histogram``, ``correlate``, ``autocorrelation``, ``aggregate`` (statistic per
       category or value) and ``describe``.
       See :ref:`mcp-analysis`.
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
     - A PNG, a summary (resolved limits, labels, legend entries) per panel, and a **review**: the
       ``issues`` it can prove (missing axis label or title, empty panel, overlapping tick labels,
       points a log axis cannot draw, data cut off by the limits), each with a severity and the call
       that fixes it.
   * - ``update_artist``
     - Change something already drawn (colour, width, which column, a function's parameter...) without
       removing it; it keeps its id and stacking order, and a change that cannot be drawn leaves the figure
       as it was. See :ref:`mcp-dynamics`.
   * - ``simulate``
     - Integrate **ordinary differential equations** (pendulum, orbits, predator-prey, Lorenz...) from
       expressions into a dataset: ``t``, each state and derived columns such as energy. RK45, DOP853,
       LSODA/Radau/BDF for stiff systems, or fixed-step RK4.
   * - ``add_control`` / ``set_control`` / ``bind_property``
     - A figure that depends on **parameters**: sliders, toggles, choices and buttons; properties follow
       expressions of them; ``render_figure(params=...)`` tries values without keeping them.
   * - ``animate``
     - **Animation**: properties follow time, in stages, and the result is a GIF, APNG, MP4, WebM or
       numbered PNGs. Returns a contact sheet and measured problems. See :ref:`mcp-animation`.
   * - ``compose_video``
     - A **whole video**: several animated figures (the scenes) cut together into one MP4/WebM/GIF, with
       fades or crossfades and an optional soundtrack. Reuses the frames ``animate`` rendered. See
       :ref:`mcp-video`.
   * - ``export_script``
     - The **Python script** behind a figure -- also an interactive version (real sliders) or an animation
       (``FuncAnimation``) -- for the user to keep and edit.
   * - ``describe_figure`` / ``remove_artist`` / ``close_figure``
     - Inspect the figure (artists with their recipe, controls, bindings); undo what one ``add_*`` call
       drew; free it.
   * - ``save_figure``
     - PNG, SVG, PDF or JPEG inside an allowed directory.
   * - ``get_guide`` / ``search_api`` / ``get_api_doc`` / ``get_example`` / ``server_info``
     - **Read-only knowledge tools**: the guide topics (this server's workflow *and* the Python API),
       a search over ~250 functions by name or description, a function's real signature and docstring,
       runnable example scripts, and the server's versions, limits and file access. See
       :ref:`mcp-knowledge`.

.. _mcp-analysis:

Analysing data
--------------

A model asked a *question* about data ("what is the decay time?", "which frequency dominates?") needs
numbers, not just a picture. ``analyze_dataset(dataset, operation, y=..., x=..., options={...})`` runs one
of 15 routines from the workstation's Math Lab and **keeps the result in the session** so it can be plotted:

* a *column* result (``fit``, ``smooth``, ``filter``, ``derivative``...) is added to the dataset, named
  ``<y>_fit``, ``<y>_smooth``... unless ``output=`` says otherwise; running it again replaces the column;
* a *table* result of a different length (``fft``, ``peaks``, ``histogram``, ``resample``,
  ``autocorrelation``) becomes a new dataset such as ``run_fft``;
* a *report* (``describe``, ``correlate``) only returns numbers.

Every reply carries a one-line result (``Exponential decay: R^2 0.9987 (a=3.00, tau=4.00, c=0.50)``), the
full report as JSON, notes where something needs attention (an uneven x for an FFT, a poor fit) and a
``plot_hint`` with the ``add_plot`` call that draws the result. Options are validated against a declared table:
an unknown name is refused with the valid ones listed, values are range-checked, and the whole reference is
generated from that table (``get_guide("mcp-analysis")``), so it cannot drift from the code. A custom fit
expression goes through the same restricted evaluator as dataset columns (no attribute access, no imports).

.. code-block:: text

   create_dataset("run", {"t": "linspace(0, 10, 400)", "v": "3*exp(-t/4) + 0.5 + noise(t, 0.05, 1)"})
   analyze_dataset("run", "fit", x="t", y="v", options={"model": "exp_decay"})   -> column v_fit, tau = 3.96
   create_figure(title="Decay"); add_plot(fig1, "scatter", dataset="run", x="t", y="v", label="data")
   add_plot(fig1, "line", dataset="run", x="t", y="v_fit", label="fit"); render_figure(fig1)

Input is capped (``Limits.max_analysis_rows``, 1,000,000 rows); a fit or filter cannot be interrupted, so a
bigger table is resampled first. The prompt ``fit_and_plot`` walks a model through the whole sequence.

.. _mcp-animation:

Animations
----------

A model cannot watch an animation, so ``animate`` gives it what it *can* check. The workflow is: build the
**still** figure, pin its axes, say what varies, **preview** (no ``output``: a contact sheet of frames comes
back), read the ``issues``, fix, then write the file.

How it works: the figure keeps a record of the calls that built it; for every frame it is **rebuilt** from
that record with each bound property replaced by its value at that frame. Every kind of artist -- lines,
scatter, bars, fields, 3-D surfaces, a right-hand axis -- therefore animates the same way, through the same
checks as a still, and the figure itself is never touched.

* **Variables and stages.** A number in ``variables`` is a parameter that a *stage* can move; an expression
  (``"phase": "2*pi*time"``) is derived each frame. A stage ``{"name", "duration", "ease", "animate":
  {"amp": [1, 3]}}`` starts where the last one ended and ends exactly on its target; afterwards the value
  holds. Easings: ``linear``, ``in``, ``out``, ``in_out``, ``smooth``, ``smoother``, ``sine``. Inside every
  expression: ``t`` (0..1), ``time`` (s), ``frame``, ``s`` (progress in the stage), ``stage``, and helpers
  ``lerp``, ``smoothstep``, ``ease_in_out``, ``pingpong``, ``ramp``, ``pulse``.
* **Bindings** say what follows what: any argument of the call that drew an artist (``color``,
  ``linewidth``, ``reveal``, ``parameters.phase``, ``x``...), an axes property (``title``, ``xlim``, 3-D ``azim``
  / ``elev``) or a **dataset column** recomputed each frame. Values can be numbers, expressions, lists, a
  colour ramp, a ``choose``-from-list, or a text ``format``.
* **Drawing in and moving.** ``add_plot(reveal=..., trail=..., head=true)`` shows part of the data: bind
  ``reveal`` to a variable and the curve draws itself, with a dot on its head and a fading tail.
* **Dynamics.** ``simulate`` produces the trajectory of a pendulum or the Lorenz system, ready to reveal.
* **Output.** ``.gif``, ``.apng``, ``.mp4`` and ``.webm`` (the video formats need ``ffmpeg`` on the PATH;
  ``server_info`` says), or ``frames/f_%04d.png``. Files stay inside ``--root`` and are never overwritten
  unless asked. Every frame has the same size.
* **Review.** ``issues`` are measured on the frames: ``static`` (nothing moves), ``xlim_moves`` (the view
  rescales), ``loop_seam`` (the loop does not close), ``abrupt_change``, ``mostly_still``, ``large_file``.

.. code-block:: text

   add_function(fig1, "amp*sin(k*x - phase)", 0, 10, parameters={"amp": 1, "k": 2, "phase": 0})
   set_axes(fig1, xlim=[0, 10], ylim=[-3.2, 3.2], xlabel="x", ylabel="y")
   animate(fig1, output="wave.gif", fps=25, variables={"phase": "2*pi*time", "amp": 1, "k": 2},
           stages=[{"name": "travel", "duration": 1.5},
                   {"name": "grow", "duration": 1.5, "ease": "in_out", "animate": {"amp": [1, 3]}},
                   {"name": "squeeze", "duration": 1.5, "animate": {"k": [2, 5]}}],
           bindings=[{"target": "a1", "property": "parameters.phase", "value": "phase"},
                     {"target": "a1", "property": "parameters.amp", "value": "amp"},
                     {"target": "a1", "property": "parameters.k", "value": "k"}])

The guide ``get_guide("mcp-animation")`` carries nine worked recipes (a travelling wave, a self-drawing
Lissajous curve, a damped pendulum, the Lorenz attractor with a camera orbit, growing bars, a rippling
surface as a seamless loop, a swept slider, morphing points, a 3Blue1Brown-style explainer); the test suite *runs* each of them, so they
cannot rot. The prompt ``animate_this`` walks a model through preview, review and output.

.. _mcp-video:

Step by step, and a whole video (the 3Blue1Brown style)
-------------------------------------------------------

An explainer is built one *play* at a time, as in a manim scene. A stage of ``animate`` can carry
``effects``, all of which happen together in that stage::

   animate(fig1, stages=[
       {"name": "curve",   "duration": 1.2, "effects": {"create": "a1"}},
       {"name": "title",   "duration": 1.0, "effects": {"write": "a2"}},
       {"name": "slide",   "duration": 2.0, "animate": {"x0": [0.5, 3.5]}},
       {"name": "point",   "duration": 0.8, "effects": {"indicate": "a5"}},
       {"name": "next",    "duration": 1.0, "effects": {"transform": {"a5": "a6"}}},
       {"name": "clear",   "duration": 0.8, "effects": {"fade_out": ["a3", "a4"]}}])

``create`` / ``uncreate`` draw a curve, an arrow or a text in and out; ``write`` / ``unwrite`` write text
letter by letter (a ``$...$`` formula appears whole); ``grow`` grows an arrow; ``fade_in`` / ``fade_out``;
``indicate`` flashes yellow and grows, then settles; ``transform`` fades one artist out as another comes in.
An artist an effect brings in is hidden until its stage. Effects are compiled into stage variables and
bindings, so the contact sheet, the review, the limits and ``export_script`` work on them unchanged.

``create_figure(style="3b1b")`` gives the look of 3Blue1Brown's manim videos: a black page, manim's
BLUE_C / YELLOW_C / TEAL palette, white serif text with ``$...$`` maths in Computer Modern, and axes that
cross at the origin with arrow tips; ``set_axes(grid=True)`` is a NumberPlane grid. Text annotations take
``coords="axes"`` (captions that stay put), ``align``, ``alpha`` and ``reveal``; ``kind="arrow"`` draws a
vector (``x``, ``y``, ``dx``, ``dy``).

A complete video is **one figure per scene**, each animated and previewed on its own, then::

   compose_video(["fig1", "fig2", {"figure_id": "fig3", "hold_end": 1}],
                 output="explainer.mp4", transition="fade")

Frames ``animate`` already rendered are reused (a scene changed since is rendered again), every scene is
fitted to one size and frame rate, ``transition`` is ``fade``, ``crossfade`` or ``cut``, and ``audio=``
adds a soundtrack. The guide topic ``mcp-video`` and the prompt ``explainer_video`` give a model the whole
workflow; ``examples/mcp/derivative_explainer_video.py`` makes a four-scene video through the official MCP
client, start to finish.

.. _mcp-dynamics:

Controls, edits, a second axis and scripts
------------------------------------------

* ``update_artist(fig, "a1", {"color": "tab:red", "parameters.a": 4})`` edits what is drawn.
* ``add_control`` defines a ``slider``, ``toggle``, ``choice`` or ``button``; ``bind_property`` makes a property
  follow an expression of them; ``set_control`` moves one and redraws; ``render_figure(params={...})`` tries
  values without changing the figure.
* ``add_plot(axis="right")`` draws against a second y axis (``set_axes(y2label=..., y2lim=...)``);
  ``hatch="//"`` patterns bars, areas and histograms.
* ``export_script(fig, kind="figure" | "interactive" | "animation")`` returns the Python for the figure. The
  datasets are rebuilt in it (expressions become numpy, ``simulate`` becomes ``solve_ivp``); the interactive
  form declares real ``plt.slider`` / ``plt.checkbox`` / ``plt.dropdown`` / ``plt.button`` widgets; the animation
  form is a ``FuncAnimation``. The test suite executes the scripts and compares their pictures with the
  figure's, pixel for pixel. The script is returned as text; the server never writes ``.py`` files.

.. _mcp-knowledge:

Learning the library from the server
------------------------------------

A model should not have to guess what GLPlot can do. Most clients show a model the *tools* but never
read *resources* on their own, so the documentation is reachable both ways:

* **Tools** (always visible): ``get_guide()`` lists the topics and ``get_guide("quickstart")`` returns
  one; ``search_api("second y axis")`` finds functions by what they do; ``get_api_doc("twinx")`` gives the
  exact signature and docstring (read from the code, so it cannot be stale); ``get_example("surface_3d")``
  returns a complete script that runs with no display; ``server_info()`` reports versions, limits and
  which directories the model may use. The connect-time instructions tell the model they exist.
* **Resources**: ``glplot://guide`` (the MCP guide, generated from the code, with the exact expression
  vocabulary and a table of every recipe), ``glplot://guide/{topic}``, ``glplot://api`` and
  ``glplot://api/{name}``, ``glplot://examples`` and ``glplot://examples/{name}``, ``glplot://llms.txt``,
  ``glplot://llms-full.txt`` (everything in one document), ``glplot://capabilities``,
  ``glplot://recipes``, ``glplot://styles``, ``glplot://figures/{id}/image`` and ``glplot://datasets/{name}``.
* **Prompts**: ``plot_data_file``, ``explore_function``, ``quick_look`` (explore a dataset),
  ``fix_my_figure`` (review and repair), ``publication_figure`` (the look chosen by venue),
  ``restyle_like`` ("make it look like The Economist"), ``compare_series``, ``animate_this``,
  ``explainer_video``, ``interactive_figure`` and
  ``fit_and_plot`` (fit a model, report parameters and R\ :sup:`2`, plot data and fit together) and
  ``python_script_for`` (write and verify a GLPlot script after reading the guides).

The topics cover the Python API as well as the tools -- plot types, axes and styling, 3-D, animation,
controls, performance, differences from matplotlib, troubleshooting -- because a model asked for a *script*
rather than a figure needs the same knowledge. The same material is available without MCP; see
:doc:`llm-docs`.

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
* **File access is opt-in and confined to a root.** Without ``--root`` the model cannot read or
  write any file; it builds data with ``create_dataset`` instead. ``--root`` (repeatable) grants
  one directory. The directory the server happens to be started in is never used, because a client
  decides that, not you. Paths are resolved (symlinks included) and must stay inside a root; only
  data and image extensions are accepted; ``save_figure`` only writes images and never overwrites
  unless asked. ``/`` is refused as a root.
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

Two implementations serve exactly the same tools, resources and prompts:

* the **official-SDK server** (``glplot.mcp.server``), used when the ``mcp`` package is installed. It
  supports both the 1.x (``FastMCP``) and 2.x (``MCPServer``) generations and adds the streamable-HTTP
  transport;
* the **built-in server** (``glplot.mcp.builtin``), used otherwise or with ``--builtin``. It speaks
  newline-delimited JSON-RPC over stdio using only the standard library (``initialize``, ``tools/*``,
  ``resources/*``, ``prompts/*``, ``ping``). It does not implement HTTP, sampling or subscriptions.

The built-in server takes its tool schemas from ``glplot/mcp/tools.json``, a snapshot of what the SDK server
publishes; a test fails when the snapshot is stale (regenerate it with
``python -m glplot.mcp.schema --write``). Importing ``glplot`` or ``glplot.mcp`` never imports the SDK, so
GLPlot itself does not depend on it.

Other commands: ``glplot-mcp tools`` lists the tools with their safety class, ``glplot-mcp doctor``
checks the install, ``glplot-mcp --version`` prints the versions, and ``--builtin`` / ``--sdk`` force one
implementation.
