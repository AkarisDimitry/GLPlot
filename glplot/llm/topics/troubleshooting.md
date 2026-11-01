# Troubleshooting

summary: Symptom-to-cause list for the failures people (and agents) actually hit.

| Symptom | Cause and fix |
|---|---|
| Script hangs forever | `plt.show()` waits for a window to close. Use `plt.savefig(...)` in unattended runs. |
| `show()` fails to open a window / GLFW or OpenGL error | No display or no OpenGL 3.3+ context (SSH, container, CI). Use `savefig`; on Linux CI use xvfb + Mesa. |
| Saved plot is empty or one panel is blank | `plt.subplot(r, c, i)` with a *different* grid shape than the figure has (it rebuilds the grid); draw into `axs[i]` instead. |
| Curve missing after `xscale("log")` | Non-positive values on a log axis are dropped; check the data range. |
| Legend is missing | Give series a `label=` and call `plt.legend()`. |
| `AttributeError: ... does not provide ...` | A matplotlib name GLPlot does not implement; the message gives the reason. Search for a replacement. |
| `MatplotlibCompatWarning: ... has no effect` | The option is accepted for compatibility but not honoured. Expected; adjust or ignore. |
| `ValueError: projection='polar' is not supported` | Use `plt.polar(theta, r)`. |
| MP4 save fails | `ffmpeg` is not on PATH. Save a GIF (Pillow), or install ffmpeg. |
| First export is slow (many seconds) | matplotlib is building its font cache; only the first run in a fresh environment. |
| Saved animation jumps or rescales | `update(frame)` depends on previous state, or the view is not pinned (topic `animation`). |
| `ImportError: mcp` from `glplot-mcp` | Not an error: without the optional `mcp` package GLPlot serves MCP with its built-in stdio server. `pip install "glplot[mcp]"` adds HTTP transport and the official SDK. |
| Huge export takes minutes | The headless export is matplotlib; aggregate (`hist2d`) or subsample before saving. |
| Unsure what is installed | `python -m glplot.llm doctor` prints versions and runs a render self-test. |
