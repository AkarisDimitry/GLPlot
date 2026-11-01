# The workstation GUI

summary: `python -m glplot`, the interactive environment for editing scenes, data, styles and analysis, in case a user asks.

`python -m glplot` opens a window with a plot and dockable panels. It exists for people exploring and styling data
interactively; a script or an LLM normally uses the Python API or the MCP server instead. Useful to know when
advising a user:

* **Data editor** -- a spreadsheet of datasets; plot columns as X/Y/Z, colour, size; derived columns from formulas
  (`a*x+b`) with live parameter sliders; undo history.
* **Style panel** -- 25 style presets rendered as cards, per-layer plot type and options, colormaps.
* **Gallery** -- 22 recipes (plot kind + look): pick a card, map your columns, one undoable step.
* **Math Lab** -- fitting, smoothing, filtering, FFT, histograms, PCA/UMAP and other transforms with confidence
  intervals, run in the background; trained models can be reused.
* **Controls panel** -- the sliders and buttons from `plt.slider` / `plt.interact` (Mod+Shift+C).
* **Timeline** -- animation scrubbing and export.
* **Command palette** (Mod+K) -- every action by name; Mod+Shift+L opens the Gallery.

Scripts that call `plt.show()` open the same window with their figure loaded.
