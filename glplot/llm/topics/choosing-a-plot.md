# Which function draws what

summary: A cheat sheet from what you want to show to the function that does it, with the full category list.

| I want to show | Use |
|---|---|
| A curve / time series | `plot(x, y)` |
| Points, optionally coloured or sized by data | `scatter(x, y, c=v, s=size, cmap=...)` |
| Millions of points | `scatter` (live window), `hist2d` / `hexbin` (density), see topic `performance` |
| One distribution | `hist(data, bins=...)`, `boxplot`, `violinplot`, `ecdf` |
| Two variables' density | `hist2d(x, y, bins=...)`, `hexbin(x, y)` |
| Categories | `bar(names, values)`, `barh`, `grouped_bar({"a": [...], "b": [...]})`, `pie` |
| Measurements with uncertainty | `errorbar(x, y, yerr=...)`, `fill_between(x, lo, hi, alpha=0.25)` |
| A step function | `step`, `stairs`; impulses: `stem` |
| Stacked areas | `stackplot(x, y1, y2, ...)` |
| A matrix / image / heatmap | `imshow(Z)`, `matshow`, `pcolormesh(X, Y, Z)` |
| Level curves of z(x, y) | `contour`, `contourf`, `tricontour` (scattered points) |
| A vector field | `quiver(x, y, u, v)`, `streamplot` |
| A 3-D surface / cloud / curve | `plot_surface`, `scatter3d`, `plot3d`, `bar3d`, `volume3d` (topic `three-d`) |
| Spectra | `psd`, `specgram`, `magnitude_spectrum`, `xcorr` |
| A reference line / shaded region | `axhline`, `axvline`, `axhspan`, `axvspan` |
| Text / arrows | `text`, `annotate`, `arrow` |
| Log or other scales | `xscale("log")`, `yscale("symlog")`, `loglog`, `semilogx`, `semilogy` |
| Two y axes | `ax2 = plt.twinx()`, then draw into `ax2` |
| Controls that drive the plot | `slider`, `interact` (topic `controls`) |
| A movie / GIF | `glplot.animation.FuncAnimation` (topic `animation`) |

## Every function, by category

{{categories}}

Use the `search_api` tool (or `python -m glplot.llm search "<what you want>"`) to find a function by
description, and `get_api_doc` (or `python -m glplot.llm describe <name>`) for its signature and docstring.
