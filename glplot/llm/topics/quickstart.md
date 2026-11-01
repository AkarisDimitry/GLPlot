# Quickstart: scripts that work first time

summary: The smallest correct scripts for the common cases, and the habits that avoid the usual failures.

```python
import numpy as np
import glplot.pyplot as plt
```

**Line plot.** `plt.plot(x, y)`, `plt.plot(x, y, "r--", lw=2, label="a")`; several calls draw on the same
axes. Format strings (`"r-o"`, `"b--"`, `"k."`) follow matplotlib; a marker with no line style
(`"o"`) draws markers only.

**Scatter.** `plt.scatter(x, y, c=values, cmap="viridis", s=10, alpha=0.7)`; add `plt.colorbar(label="...")`.

**Several panels.**

```python
fig, axs = plt.subplots(2, 2, figsize=(9, 6))
axs[0, 0].plot(x, y)
axs[0, 1].hist(data, bins=40)
axs[1, 0].scatter(x, y)
axs[1, 1].bar(["a", "b"], [3, 5])
plt.suptitle("Four panels")
plt.savefig("grid.png")
```

Inside a grid, call methods **on the axes objects** (`axs[0, 1].scatter(...)`). `plt.subplot(r, c, i)` with the
grid's own shape just selects that panel; a *different* shape rebuilds the grid and empties it.

**Always label.** `plt.title(...)`, `plt.xlabel("name (unit)")`, `plt.ylabel(...)`, `plt.legend()` when
more than one series has a `label=`.

**Save and look.** `plt.savefig("out.png")` (also `.pdf`, `.svg` on the headless path). Open the file
and check it: axis ranges, overlapping labels, an empty panel, a missing legend. Fix and save again.

## Habits that avoid failures

1. **Do not call `plt.show()` in an unattended script** (it blocks); call `plt.savefig(...)`.
2. **Do not type large data into code.** Generate it with numpy or load it from a file.
3. **Axis limits work per axis, as in matplotlib**: `plt.ylim(...)` pins y and x keeps following the data (before
   or after you plot). On a log axis limits are in data units (`plt.xlim(10, 1000)`); a non-positive end is ignored
   with a warning. `plt.axis()` with no arguments returns `(xmin, xmax, ymin, ymax)`.
4. **`errorbar(capsize=...)` is in data units**, not points as in matplotlib -- pick a value relative to your x range.
5. One figure per `plt.figure(...)`; `plt.close("all")` frees them in a long-running script.
6. Anything that looks wrong in the file but fine in matplotlib: check the `matplotlib-compat` topic.

## Runnable examples

{{examples}}
