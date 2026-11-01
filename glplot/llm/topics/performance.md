# Performance and large data

summary: What stays interactive at millions of points, density and LOD options, and why exports are slower than the window.

The live window draws on the GPU: scatter and line plots with millions of points pan and zoom smoothly, and the
renderers decimate by zoom level (level of detail), so zooming out does not make frames slower.

| Situation | What to do |
|---|---|
| 1-50 million scatter points | `plt.scatter(x, y, s=1.5, alpha=0.7)` and `plt.show()`; it stays interactive. |
| A cloud so dense it is a blob | `plt.hist2d(x, y, bins=170, cmap="plasma")` or `plt.hexbin(...)`; or `plt.show(density=True)`. |
| 100,000s of lines | `plt.plot_lines(a, b, x_range=(x0, x1))` (each line is a slope/intercept pair, drawn on the GPU). |
| Very large 3-D clouds | `plt.scatter3d(...)`, `plt.volume3d(...)`; rotate live. |
| Huge arrays in a saved file | Aggregate first (`hist2d`, binned means) or subsample: the headless export is matplotlib. |

## Switches

`plt.figure(density=True)` / `plt.show(density=True)` density rendering; `plt.lod(...)`/`plt.set_lod(...)` level of
detail; `plt.figure(budget=8)` the level-of-detail budget (lines per pixel); `plt.hud(True)` frame-time overlay; `plt.profile(...)`;
`plt.density_gain(...)`. Defaults are chosen so most scripts need none of them.

## Exports are not the GPU

`plt.savefig` without a window re-draws the scene with matplotlib (Agg). That is portable and works everywhere, but a
5-million-point scatter takes seconds there and looks slightly different from the window (no glow, no GPU density
tinting). An animation saved headless pays that cost per frame.

## Habits

* Keep arrays as `float32`/`float64` numpy arrays; avoid Python lists of millions of numbers.
* One `scatter` call with a colour array beats thousands of small calls.
* Uniform colour is fastest; per-point colour/size cost a little more memory.
