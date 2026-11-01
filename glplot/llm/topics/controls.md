# Interactive controls

summary: Sliders, buttons and interact() that put widgets in a Controls panel beside the plot.

Native controls live in a Controls panel next to the plot. They are created with ordinary function calls:

```python
amp  = plt.slider("Amplitude", 0, 5, 1.0, step=0.1)
freq = plt.slider("Frequency", 0.1, 10, 2.0, log=True, unit="Hz")
show = plt.checkbox("Show grid", True)
kind = plt.dropdown("Wave", ["sin", "cos"])
plt.button("Reset")(lambda: ...)            # decorator form: @plt.button("Reset")
```

* `plt.slider`, `plt.range_slider`, `plt.checkbox`, `plt.dropdown`, `plt.radio`, `plt.textbox`, `plt.number`,
  `plt.color_picker`, `plt.readout`, `plt.button`. Group them with `with plt.control_group("Filter"):`.
* Read a control with `.value`; react with `@control.on_change`; list them with `plt.controls()`.
* `@plt.interact(amp=(0, 5), freq=(0.1, 10), wave=["sin", "cos"])` wraps a function: each keyword becomes a control,
  the function runs once immediately and on every change (at most once per frame). If it returns `(x, y)` the curve is
  drawn and updated in place; a `dict` is shown as a results table.
* Useful slider options: `step=`, `log=True`, `integer=True`, `unit=`, `fmt="%.2f"`, `live=False` (update on release,
  for expensive callbacks), `help="tooltip"`.

## Scripts versus the window

The function runs once at creation, so a **headless script still gets a figure** for the initial values and
`plt.savefig(...)` works. Interactivity needs `plt.show()`. A script with controls opens the Controls panel
automatically. Callbacks are coalesced to one per frame on the GL thread; an exception is shown in the panel instead of
stopping the render loop.

Matplotlib's own widgets (`Slider`, `Button`, `CheckButtons`, `RadioButtons`, `TextBox`) run unchanged via
`glplot.widgets`; `plt.Slider` / `plt.Button` resolve to them.

An agent without a display cannot use the window; give the user the script and tell them to run it, or render
the initial state with `savefig`.
