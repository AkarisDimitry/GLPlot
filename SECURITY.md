# Security policy

## Supported versions

Security fixes are made to the latest release on PyPI. GLPlot is pre-1.0; older 0.x versions
are not patched.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Report it privately, either by

- opening a private advisory on GitHub:
  <https://github.com/AkarisDimitry/GLPlot/security/advisories/new>, or
- emailing the maintainer, Juan Manuel Lombardi, at <lombardi@fhi.mpg.de>.

Include the GLPlot version, how to reproduce the problem, and what you think the impact is.
You can expect an acknowledgement within a week. Fixes are released as a new version and
described in the [CHANGELOG](CHANGELOG.md); you will be credited unless you prefer not to be.

## What is in scope

GLPlot is a plotting library: it draws what the calling code gives it. Most of it runs with the
privileges of your own Python process, and that is by design. The parts that take input from
somewhere else, and so deserve scrutiny, are:

### The MCP server (`python -m glplot.mcp`, `glplot-mcp`)

The server lets a language model make plots. Treat the model as an untrusted caller. The server
is built so that it cannot be asked to run arbitrary code or reach arbitrary files:

- **No code execution.** There is no tool that evaluates Python. Formulas (`add_function`,
  `add_field`, dataset columns) go through the same AST-validated evaluator the GUI uses, which
  allows a fixed set of numeric operations, and the array constructors in it are size-capped.
- **No file access by default.** Reading and writing files is off until you pass `--root DIR`.
  With a root, paths are confined to it after symlink resolution, limited to known extensions
  (data: tables, `.npy`/`.npz`, images; output: `.png`, `.svg`, `.pdf`, `.jpg`), files over
  256 MB are refused, and an existing file is never overwritten unless the call says so.
- **Local transport by default.** The default transport is stdio. `--transport streamable-http`
  binds to `127.0.0.1:8765`. The HTTP transport has **no authentication**; if you bind it to
  another address (`--host`), anything that can reach the port can draw and, with `--root`,
  read and write inside that root. Put it behind your own access control.
- **Resource limits.** Figure count, image size and array sizes are capped; the limits are in
  `glplot.mcp.state.Limits`.

A way to read or write outside `--root`, to make the server execute code, or to exhaust memory
past the documented limits is a vulnerability: please report it.

### Files you load

`load_dataset`, `add_field(file=...)` and the GUI's file loaders parse data files with NumPy,
Matplotlib and Pillow. A crash on a malformed file is a bug (open an ordinary issue); memory
corruption or code execution through one is a vulnerability. GLPlot does not call
`numpy.load(..., allow_pickle=True)`.

## What is not in scope

- Code you pass to GLPlot yourself (a script, a callback given to `gplt.interact`, a
  `FuncAnimation` update function) runs as you, as in any Python program.
- Vulnerabilities in dependencies (NumPy, Matplotlib, PyOpenGL, GLFW, `imgui-bundle`, `mcp`):
  report those upstream. If GLPlot pins a version that is affected, tell us.
