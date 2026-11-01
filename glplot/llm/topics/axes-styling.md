# Axes, labels, scales and styling

summary: Limits, scales, ticks, legends, twin axes, colorbars, colormaps and the 25 style presets.

## Labels, limits, ticks

```python
plt.title("Title"); plt.xlabel("time (s)"); plt.ylabel("signal (V)")
plt.xlim(0, 10); plt.ylim(-1, 1)          # per axis: set one and the other keeps fitting the data
plt.xticks([0, 5, 10], ["start", "mid", "end"])
plt.grid(True); plt.legend(loc="upper right")
plt.set_aspect("equal")                    # same units per pixel on x and y; also plt.axis("equal")
```

Object style works everywhere a function does: `ax.set_xlabel(...)`, `ax.set_xlim(...)`,
`ax.set_title(...)`, `ax.legend()`.

## Scales

`plt.xscale(name)` / `plt.yscale(name)` take `"linear"`, `"log"`, `"symlog"`, `"asinh"`, `"logit"`
(all real, applied to the data on the GPU). Values must be positive on a log axis; `symlog` and
`asinh` accept any sign. `loglog`, `semilogx`, `semilogy` set the scale and plot in one call. Scales
are per axes in a subplot grid (`axs[1].set_yscale("log")` does not change the others). Limits stay in data
units on a scaled axis: after `xscale("log")`, `plt.xlim(10, 1000)` frames 10 to 1000.

## Two y axes

```python
ax1 = plt.gca(); ax1.plot(t, a); ax1.set_ylabel("a")
ax2 = plt.twinx(); ax2.plot(t, b, color="tab:red"); ax2.set_ylabel("b")
```

Draw into the object `twinx()` returns. `twiny()` is the x version; `secondary_xaxis` /
`secondary_yaxis` add a derived scale via `functions=(forward, inverse)`.

## Colour

* Colours: names (`"red"`), hex (`"#1f77b4"`), cycle (`"C0"`), `"tab:blue"`, RGB/RGBA tuples.
* Colormaps: any matplotlib name plus GLPlot's own ({{colormaps}}). `plt.colorbar(label=...)` adds the bar;
  `plt.clim(lo, hi)` sets its range; `plt.set_cmap(name)` changes the default.
* Perceptually uniform and colour-blind safe: `viridis`, `plasma`, `cividis`. Diverging data centred on
  zero: `RdBu`, `coolwarm`, `seismic`. Avoid `jet` for quantitative data.

## Style presets

`plt.plot_style("name")` restyles the **whole current figure** (page, palette, line width, grid and
the colormap that goes with the look); call it before or after plotting. `plt.plot_styles()` returns
`(key, title, description)` for every preset. The presets:

{{styles}}

Effects such as glow, SSAO depth shading and density tinting exist only in the live window; the
saved file shows the colours, page and lines of the preset.

## Text size, fonts, rcParams

`plt.rcParams` / `plt.rc(...)` / `plt.rc_context(...)` are matplotlib's, applied to the headless export.
Pass `fontsize=` to `text`, `title`, `xlabel`.
