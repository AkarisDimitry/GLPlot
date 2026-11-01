# Animation

summary: FuncAnimation and ArtistAnimation, saving GIF/MP4 headless, live playback, the rules that keep both consistent, and how an MCP client animates (the animate tool).

`glplot.animation` is a drop-in replacement for `matplotlib.animation`. It is **not** on `plt`:

```python
import numpy as np
import glplot.animation as animation
import glplot.pyplot as plt

x = np.linspace(0, 2 * np.pi, 200)
FRAMES = 40
fig = plt.figure("Wave", figsize=(7, 4))
plt.xlim(0, 2 * np.pi); plt.ylim(-1.2, 1.2)
(line,) = plt.plot(x, np.sin(x))

def update(frame):
    phase = 2 * np.pi * frame / FRAMES
    line.set_ydata(np.sin(x - phase))     # artist setters: set_data / set_xdata / set_ydata
    plt.title(f"phase {phase:.2f}")
    return [line]

ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=40)
ani.save("wave.gif", fps=25)       # no window needed; plt.show() plays it live instead
```

## Saving

* `ani.save("x.gif", fps=...)` uses Pillow (always available). `ani.save("x.mp4", fps=...)` needs the
  `ffmpeg` executable on PATH. `ani.to_jshtml()` returns an HTML player. `animation.writers.list()` shows the writers.
* Without a window, frames are drawn by the matplotlib-based export (portable, not pixel-identical to the
  GPU). Pass `savefig_kwargs={"renderer": "opengl"}` to render through OpenGL when a context exists.
* `blit=` is accepted and ignored; every frame is a full re-render.

## Rules that avoid broken animations

1. **Make `update(frame)` a pure function of `frame`.** Compute positions from the frame number rather than
   mutating state left by the previous call -- live playback and saving then agree, and re-running a frame is safe.
2. **Pin the view** (`plt.xlim/ylim` with both axes, or via `plt.axis`) so the axes do not rescale every frame.
3. Updating data in place is cheap: `line.set_ydata`, `scatter_layer.set_offsets(xy)`. `plt.cla()` then redrawing
   also works and is the simplest form for a bar chart.
4. Return a list of the artists you changed (an empty list is accepted).
5. Frame cost matters: a 200-frame GIF at figsize (8, 5) is 200 full exports. Reduce `frames`, `figsize`, or `scale`.
6. `ArtistAnimation(fig, [[artist, ...], ...])` plays a precomputed list of artist groups.

## Live playback

With `plt.show()` a `FuncAnimation` plays in the window on a wall-clock timer, with the controls panel
available. `ani.pause()` / `ani.resume()` stop and restart it.

## From an MCP client

A model driving GLPlot through MCP does not write `FuncAnimation` code: the `animate` tool binds figure
properties to time and stages and writes the GIF/MP4, and `export_script(kind="animation")` returns the
equivalent `FuncAnimation` script. See `get_guide("mcp-animation")`.
