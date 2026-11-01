# matplotlib compatibility: same, different, missing

summary: What carries over from matplotlib unchanged, what differs, and the names that deliberately do not exist.

**Same:** `import glplot.pyplot as plt`; pyplot and object styles (`fig, ax = plt.subplots()`, `ax.plot`, `ax.set_xlabel`);
format strings; `rcParams`/`rc_context`; colormaps and norms; `Normalize`, `LogNorm`, tickers and formatters; patches
(`Rectangle`, `Circle`, ...) via `add_patch`; `GridSpec`, `subplot_mosaic`; `Slider`/`Button` widgets via `glplot.widgets`;
`matplotlib.animation` via `glplot.animation`.

## Behaves differently

| Topic | GLPlot |
|---|---|
| Display | `show()` opens a GPU window and blocks; use `savefig` unattended. |
| Unsupported option | Accepted, then a one-time `MatplotlibCompatWarning` says it had no effect. Read the warnings. |
| `errorbar(capsize=)` | Data units, not points. |
| Hatches, `suptitle`, tick-label rotation | Drawn in exported files; the live window does not show them. |
| Small figures | Exports have a 4 x 3 inch floor. |
| Grids | `plt.subplot(r, c, i)` with the grid's own shape selects a panel; a different shape rebuilds the grid (empty). Prefer the axes objects. |
| `projection="polar"` | Not a projection. `plt.polar(theta, r)` draws a polar plot into 2-D axes. |
| `plt.twinx()` | Real second y axis, but module-level `plt.plot` after it still targets the first axes: draw into the returned object. |
| Animation | `glplot.animation`, not `plt`. |
| Interactive widgets | `plt.slider`, `plt.interact` are GLPlot's own. |
| Python-side matplotlib objects | `Figure`, `Axes` classes are re-exported for isinstance checks; GLPlot's figure is a `GPULinePlot`, not a matplotlib `Figure`. |
| matplotlib pins | GLPlot needs `matplotlib>=3.6,<3.10`. |

## Names that do not exist

These raise `AttributeError` with the reason:

{{unsupported}}

## Finding a replacement

Use `search_api` / `python -m glplot.llm search "<matplotlib name or idea>"`; the index knows common matplotlib wording
(`heatmap`, `confidence band`, `secondary axis`) and maps it to GLPlot functions.
