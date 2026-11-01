# Saving figures

summary: savefig options, output size, formats, and the two export paths (window versus headless).

```python
plt.savefig("fig.png")                    # default scale=2.0 -> figsize x 240 px
plt.savefig("fig.png", scale=1.0)         # figsize x 120 px
plt.savefig("fig.pdf")                    # vector, headless path
plt.savefig("fig.svg")
plt.savefig("fig.png", dpi=300, transparent=True, bbox_inches="tight")   # matplotlib keywords (headless path)
fig.savefig("fig.png")                    # the figure/axes objects work too
```

* **Size:** `figsize=(w, h)` inches; the PNG is `w * 120 * scale` by `h * 120 * scale` pixels. There is a floor of
  4 x 3 inches: smaller figures are exported at that size.
* **Two paths.** With a live window the scene is read back from the GPU. Without one (the usual case for scripts) the
  scene is re-drawn with matplotlib; `dpi`, `bbox_inches`, `transparent`, `facecolor`, `pad_inches`, `format` and any other
  `Figure.savefig` keyword apply there, and are accepted-but-warned on the window path.
* `plt.savefig` creates no directories: the folder must exist.
* The export draws everything that is data (lines, markers, patches, images, colorbars, legends, twin axes, hatches,
  log scales). Live-only effects (glow, SSAO, GPU density tinting) are not in it.
* Hatches (`hatch=`) and figure-level `suptitle` are drawn **only in exports**; the live window fills solid.
* Animations: `ani.save("x.gif" | "x.mp4")`, see topic `animation`.
* From code that needs bytes or pixels rather than a file: `glplot.utils.preview.render_preview_array(fig)`.
