"""The function generator panel — write ``a*sin(b*x)``, get sliders and a live curve.

The panel is a thin, stateful shell around :mod:`glplot.gui.expressions`: it owns the
text, the domain and the parameter values, and it re-evaluates only when one of those
actually changed. Two things are worth knowing before editing this file.

**Auto-detected parameters.** Every free variable in the expression — any bare name that
is neither ``x`` nor part of ``expressions.SAFE_NAMES`` — becomes a slider, via
:func:`glplot.gui.expressions.free_variables` (which walks the validated AST, so a
hostile expression raises instead of reporting variables). Typing ``a*sin(b*x)`` yields
``a`` and ``b`` sliders that drive the preview live. Slider values persist in
:attr:`FunctionsPanel.params` even when a name leaves the expression, so deleting and
retyping ``b`` restores the value the user had dialled in rather than resetting to 1.

**Change-gated evaluation, not per-frame.** ``draw`` runs 60x a second; parsing and
evaluating there would be wasteful and would make a 1e6-sample domain unusable. Both the
AST parse (:meth:`_sync_params`, gated on the text) and the numpy evaluation
(:meth:`_refresh_preview`, gated on a signature of every input) run only on change. The
preview is additionally capped at :data:`PREVIEW_MAX_SAMPLES` points regardless of the
requested sample count — the actions re-evaluate at full resolution, so dragging a
slider stays interactive even with a million samples queued up behind it.

Errors are inline and non-fatal: an invalid expression shows the ``ExpressionError``
message in a red box and the *last good curve* stays on screen, because a preview that
blanks on every half-typed keystroke is unreadable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import expressions, layerops, layerops3d, widgets
from ..datasets import Column, DataSet
from ..expressions import ExpressionError
from ..history import Command, snapshot
from ..icons import icon_button
from ..theme import COLORS
from .base import Panel

try:
    import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - exercised only on GL-less systems
    IMGUI_AVAILABLE = False
    imgui = None


#: Hard ceiling on the generated sample count. A million float64 x/y pairs is 16 MB and
#: about 10 ms to evaluate — enough rope, not enough to hang the loop.
MAX_SAMPLES = 1_000_000

#: The preview never evaluates more than this many points no matter what the user asked
#: for. ``mini_plot`` decimates to 2000 anyway, so a denser preview buys nothing but
#: latency on every slider frame. "Plot"/"Create dataset" always use the full count.
PREVIEW_MAX_SAMPLES = 20_000

#: The same idea for a field, counted in *cells* rather than samples, because a grid costs
#: ``nx * ny``: the 20k that makes a cheap curve would be a 400M-cell grid. Both axes are
#: scaled by one factor to reach it, so the preview keeps the aspect the full field will
#: have. "Plot" always uses the full grid.
PREVIEW_MAX_CELLS = 65_536

#: Parameter values are remembered after a name leaves the expression (see the module
#: docstring). Prune beyond this many so a long editing session cannot grow unbounded.
MAX_REMEMBERED_PARAMS = 64

#: Default slider bounds for a newly discovered parameter.
DEFAULT_PARAM_MIN = -10.0
DEFAULT_PARAM_MAX = 10.0
DEFAULT_PARAM_VALUE = 1.0

#: What a field can be drawn as, as ``(key, label)`` in picker order. The same
#: ``z = f(x, y)`` is a heat map in 2D and a height field in 3D; nothing about the
#: expression changes, only what is built from the grid it produces.
_FIELD_OUTPUTS: Tuple[Tuple[str, str], ...] = (
    ("image", "Heat map (2D image)"),
    ("surface", "Surface (3D)"),
    ("wireframe", "Wireframe (3D)"),
)

_HELP_FIELD_OUTPUT = (
    "The same expression, three pictures. A heat map is a 2D image layer — flat, exact, "
    "and the cheapest to read a value off. A surface lifts z out of the plane and gives "
    "the eye a shape, which is what makes a saddle or a ridge obvious. A wireframe shows "
    "the same surface as a mesh, so you can see through it to what lies behind.\n\n"
    "Choosing a 3D output switches the figure to a 3D view."
)


@dataclass
class Param:
    """One auto-detected free variable: its live value and its slider bounds."""

    value: float = DEFAULT_PARAM_VALUE
    vmin: float = DEFAULT_PARAM_MIN
    vmax: float = DEFAULT_PARAM_MAX


@dataclass(frozen=True)
class Preset:
    """A one-click example. ``params`` is ``(name, value, vmin, vmax)`` per parameter."""

    label: str
    icon: str
    expr: str
    x_min: float
    x_max: float
    samples: int
    params: Tuple[Tuple[str, float, float, float], ...] = ()


#: The preset palette. Every expression uses only ``SAFE_NAMES`` callables and free
#: variables, so each one round-trips through ``free_variables`` into its own sliders.
PRESETS: Tuple[Preset, ...] = (
    Preset(
        "Sine",
        "wave",
        "a * sin(b * x + c)",
        -10.0,
        10.0,
        500,
        (("a", 1.0, -5.0, 5.0), ("b", 1.0, -10.0, 10.0), ("c", 0.0, -3.2, 3.2)),
    ),
    Preset(
        "Damped oscillation",
        "smooth",
        "a * exp(-d * x) * sin(w * x)",
        0.0,
        20.0,
        800,
        (("a", 1.0, -5.0, 5.0), ("d", 0.2, 0.0, 2.0), ("w", 3.0, 0.0, 20.0)),
    ),
    Preset(
        "Gaussian",
        "chart_bar",
        "a * gauss(x, mu, sigma)",
        -6.0,
        6.0,
        500,
        (("a", 1.0, -5.0, 5.0), ("mu", 0.0, -5.0, 5.0), ("sigma", 1.0, 0.05, 5.0)),
    ),
    Preset(
        "Sigmoid",
        "chart_line",
        "a * sigmoid(k * (x - x0))",
        -10.0,
        10.0,
        500,
        (("a", 1.0, -5.0, 5.0), ("k", 1.0, -5.0, 5.0), ("x0", 0.0, -10.0, 10.0)),
    ),
    Preset(
        "Sawtooth",
        "filter",
        "a * sawtooth(x, period)",
        -5.0,
        5.0,
        1000,
        (("a", 1.0, -5.0, 5.0), ("period", 2.0, 0.1, 10.0)),
    ),
    Preset(
        "Chirp",
        "sliders",
        "sin(2 * pi * (f0 + k * x) * x)",
        0.0,
        10.0,
        2000,
        (("f0", 0.5, 0.0, 5.0), ("k", 0.3, 0.0, 3.0)),
    ),
    Preset(
        "Lorentzian",
        "target",
        "a * g**2 / ((x - x0)**2 + g**2)",
        -10.0,
        10.0,
        600,
        (("a", 1.0, -5.0, 5.0), ("g", 1.0, 0.05, 5.0), ("x0", 0.0, -10.0, 10.0)),
    ),
    Preset(
        "Polynomial",
        "function",
        "c3 * x**3 + c2 * x**2 + c1 * x + c0",
        -5.0,
        5.0,
        500,
        (
            ("c3", 0.1, -2.0, 2.0),
            ("c2", -0.5, -5.0, 5.0),
            ("c1", 1.0, -10.0, 10.0),
            ("c0", 0.0, -10.0, 10.0),
        ),
    ),
    Preset(
        "Step",
        "chevron_up",
        "a * step(x - x0)",
        -5.0,
        5.0,
        500,
        (("a", 1.0, -5.0, 5.0), ("x0", 0.0, -5.0, 5.0)),
    ),
    Preset(
        "Sinc",
        "derivative",
        "a * sinc(b * x)",
        -10.0,
        10.0,
        1000,
        (("a", 1.0, -5.0, 5.0), ("b", 1.0, -5.0, 5.0)),
    ),
    Preset(
        "Exponential decay",
        "chevron_down",
        # The time constant is t0, not tau: `tau` is 2*pi in SAFE_NAMES, so it would
        # silently bind to the constant and never surface as a slider.
        "a * exp(-x / t0)",
        0.0,
        10.0,
        500,
        (("a", 1.0, -5.0, 5.0), ("t0", 2.0, 0.05, 10.0)),
    ),
    Preset(
        "Logistic curve",
        "refresh",
        "r * x * (1 - x)",
        0.0,
        1.0,
        500,
        (("r", 3.7, 0.0, 4.0),),
    ),
    Preset(
        "Noise + signal",
        "noise",
        "a * sin(x) + noise(x, s, 0)",
        0.0,
        20.0,
        1000,
        (("a", 1.0, -5.0, 5.0), ("s", 0.2, 0.0, 2.0)),
    ),
    Preset(
        "Sum of harmonics",
        "layers",
        "a1 * sin(x) + a2 * sin(2 * x) / 2 + a3 * sin(3 * x) / 3",
        -10.0,
        10.0,
        1000,
        (("a1", 1.0, -2.0, 2.0), ("a2", 1.0, -2.0, 2.0), ("a3", 1.0, -2.0, 2.0)),
    ),
)

#: Layer types whose data ``layerops.update_layer_xy`` can swap, i.e. the ones
#: "Replace selected layer" can target.
_REPLACEABLE = ("scatter", "polyline", "line_family")

_HELP_EXPRESSION = (
    "A numpy expression in x. Available: sin cos tan exp log sqrt abs floor ceil clip "
    "where maximum minimum linspace gauss sigmoid rect step sawtooth square sinc "
    "heaviside noise, and the constants pi e tau inf nan.\n\n"
    "min/max are reductions (they collapse the curve to one value); use maximum/minimum "
    "for the elementwise versions: maximum(x, 0) clips at zero, max(x) does not.\n\n"
    "Any other bare name becomes a parameter slider automatically."
)

_HELP_LOG = "Sample x geometrically instead of uniformly. Requires x min and x max > 0."

_HELP_PLOT_LIVE = (
    "Plot the function itself instead of a table of samples.\n\n"
    "'Plot' evaluates the expression once at the sample count above and stores the "
    "points; zooming in magnifies those points. 'Plot live' stores the expression and "
    "re-evaluates it every time the view moves, at about one sample per pixel column of "
    "the plot. So the resolution is always what the screen can show, the cost is the same "
    "at any zoom, and zooming in reveals detail that was never computed before.\n\n"
    "Right for anything you can evaluate — analytic curves, recursions, series. The "
    "parameter sliders are baked in at the moment you press it: move one afterwards and "
    "the plotted curve keeps the values it was made with, so press it again for the new "
    "ones."
)


class FunctionsPanel(Panel):
    """Generate a curve from an expression, with auto-detected parameter sliders."""

    title = "Functions"
    icon = "function"
    default_open = True

    def __init__(self, ws: Any) -> None:
        super().__init__(ws)
        self.expr: str = "a * sin(b * x)"
        self.x_min: float = -10.0
        self.x_max: float = 10.0
        self.samples: int = 500
        self.log_spacing: bool = False
        #: Field (2-D) domain. Only read when the expression names ``y`` — see
        #: :meth:`is_field`. Kept beside the x domain rather than in a separate mode
        #: object so that switching an expression back and forth does not lose it.
        self.y_min: float = -10.0
        self.y_max: float = 10.0
        self.y_samples: int = 200
        self.cmap: str = "viridis"
        #: How a field is drawn: "image" (2D heat map), "surface" or "wireframe" (3D).
        #: The same z = f(x, y) expression, three pictures — see ``_draw_field_actions``.
        self.field_output: str = "image"
        #: Grid-line stride for the wireframe output.
        self.wire_stride: int = 1
        self.kind: str = "line"
        self.color: List[float] = [0.26, 0.59, 0.98]
        self.line_width: float = 1.5
        self.point_size: float = 6.0
        #: The selected kind's parameters (bar width, bins, ...), defaulted from its own
        #: registry entry and re-defaulted whenever the kind changes -- carrying a bar's
        #: width onto a histogram would apply a number that kind never declared.
        self.kind_opts: Dict[str, Any] = layerops.default_kind_options(self.kind)

        #: name -> Param, including names no longer referenced (see the module docstring).
        self.params: Dict[str, Param] = {
            "a": Param(1.0, -5.0, 5.0),
            "b": Param(1.0, -10.0, 10.0),
        }
        #: The free variables of the current expression, in slider order.
        self._param_order: List[str] = ["a", "b"]

        # Change gates. `_synced_expr` is the text `_param_order` was parsed from;
        # `_signature` is every input the preview curve depends on.
        self._synced_expr: Optional[str] = None
        self._signature: Optional[Tuple[Any, ...]] = None

        self._curve: Optional[Tuple[np.ndarray, np.ndarray]] = None
        #: Set by `_sync_params` from the text, read by `is_field`.
        self._is_field: bool = False
        #: The last good field matrix, mirroring `_curve`'s role for the 2-D mode.
        self._field: Optional[np.ndarray] = None
        self._preview_truncated: bool = False
        self._error: Optional[str] = None
        self._status: str = ""
        self._status_ok: bool = True
        self._edit_ranges: bool = False

    # -- evaluation ------------------------------------------------------------

    def _bindings(self) -> Dict[str, float]:
        """The variable bindings for the current expression's free variables only."""
        return {name: self.params[name].value for name in self._param_order if name in self.params}

    def _build_x(self, n: int) -> np.ndarray:
        """Build the domain. Raises ValueError with a message fit for the error box."""
        if not np.isfinite(self.x_min) or not np.isfinite(self.x_max):
            raise ValueError("x min and x max must be finite numbers.")
        if self.x_max == self.x_min:
            raise ValueError("x max must differ from x min.")
        if self.log_spacing:
            if self.x_min <= 0.0 or self.x_max <= 0.0:
                raise ValueError("Log spacing needs x min and x max greater than zero.")
            return np.logspace(np.log10(self.x_min), np.log10(self.x_max), n)
        return np.linspace(self.x_min, self.x_max, n)

    def _evaluate(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the expression over ``n`` samples.

        Raises:
            ExpressionError: for anything wrong with the expression itself.
            ValueError: for a domain that cannot be built.
        """
        xs = self._build_x(n)
        ys = expressions.evaluate_1d(self.expr, xs, variables=self._bindings())
        return xs, ys

    @property
    def is_field(self) -> bool:
        """True when the expression names ``y``, making it a field over x *and* y.

        The mode is read off the text rather than toggled with a switch, which is the same
        bargain the parameter sliders already make: type a new free name and you get a
        slider, type ``y`` and you get a second axis. The consequence is worth stating —
        ``y`` can no longer be a slider parameter, because a name cannot be an axis and a
        knob at once. Any other name still can.
        """
        return self._is_field

    def _sync_params(self) -> None:
        """Re-derive the slider list from the expression. Gated on the text changing.

        A rejected expression (mid-keystroke, or hostile) leaves the previous slider set
        in place: the error box already says what is wrong, and yanking the sliders away
        on every incomplete edit would make the panel jump around as the user types.
        """
        if self.expr == self._synced_expr:
            return
        self._synced_expr = self.expr
        try:
            names = expressions.free_variables(self.expr)
        except ExpressionError:
            return

        # `y` names the second axis, so it is excluded from the sliders rather than
        # becoming one. Detected here, once per edit, rather than re-parsed per frame.
        self._is_field = "y" in names
        names = [n for n in names if n != "y"]

        for name in names:
            if name not in self.params:
                self.params[name] = Param()
        self._param_order = names

        if len(self.params) > MAX_REMEMBERED_PARAMS:
            keep = set(names)
            self.params = {k: v for k, v in self.params.items() if k in keep}

    def _build_y(self, n: int) -> np.ndarray:
        """The field's y axis. Linear only — log spacing is an x-axis control here."""
        if self.y_max == self.y_min:
            raise ValueError("y max must differ from y min.")
        return np.linspace(self.y_min, self.y_max, n)

    def _clamped_y_samples(self) -> int:
        return int(np.clip(self.y_samples, 2, MAX_SAMPLES))

    def _field_shape(self, cap: Optional[int] = None) -> Tuple[int, int]:
        """``(nx, ny)`` for the field, optionally capped to ``cap`` *cells* in total.

        A grid costs ``nx * ny``, so the 1-D sample cap does not transfer: 20k samples is
        a cheap curve and a 20k x 20k grid is 400M cells. The cap scales both axes by the
        same factor, which keeps the preview's aspect and so keeps it honest about shape.
        """
        nx, ny = self._clamped_samples(), self._clamped_y_samples()
        if cap is None or nx * ny <= cap:
            return nx, ny
        scale = (cap / float(nx * ny)) ** 0.5
        return max(2, int(nx * scale)), max(2, int(ny * scale))

    def _evaluate_field(self, nx: int, ny: int) -> np.ndarray:
        """Evaluate the expression over an ``nx`` by ``ny`` grid.

        Raises:
            ExpressionError: for anything wrong with the expression itself.
            ValueError: for a domain that cannot be built.
        """
        return expressions.evaluate_2d(
            self.expr, self._build_x(nx), self._build_y(ny), variables=self._bindings()
        )

    def _preview_signature(self, n: int) -> Tuple[Any, ...]:
        """Everything the preview depends on, in a comparable form."""
        return (
            self.expr,
            self.x_min,
            self.x_max,
            n,
            self.log_spacing,
            self.is_field,
            self.y_min if self.is_field else None,
            self.y_max if self.is_field else None,
            self._field_shape(PREVIEW_MAX_CELLS) if self.is_field else None,
            tuple((name, self.params[name].value) for name in self._param_order),
        )

    def _refresh_preview(self) -> None:
        """Recompute the preview if and only if an input changed."""
        n = int(min(self._clamped_samples(), PREVIEW_MAX_SAMPLES))
        signature = self._preview_signature(n)
        if signature == self._signature:
            return
        self._signature = signature

        if self.is_field:
            nx, ny = self._field_shape(PREVIEW_MAX_CELLS)
            self._preview_truncated = (nx, ny) != self._field_shape()
            try:
                self._field = self._evaluate_field(nx, ny)
            except (ExpressionError, ValueError) as exc:
                # Keep the last good field, exactly as the curve path does.
                self._error = str(exc)
                return
            self._error = None
            return

        self._preview_truncated = n < self._clamped_samples()
        try:
            self._curve = self._evaluate(n)
        except (ExpressionError, ValueError) as exc:
            # Keep self._curve: the last good curve stays on screen behind the error.
            self._error = str(exc)
            return
        self._error = None

    def _clamped_samples(self) -> int:
        """The sample count actually used, whatever the input widget currently holds."""
        return int(np.clip(self.samples, 2, MAX_SAMPLES))

    # -- drawing ---------------------------------------------------------------

    def draw(self) -> None:
        """Render the panel body. The workspace owns the surrounding begin/end."""
        if not IMGUI_AVAILABLE:
            return

        self._draw_presets()
        imgui.separator()
        self._draw_expression()
        # Order matters: the sliders below must reflect the text typed a line above,
        # and the preview must reflect the sliders, all within this one frame.
        self._sync_params()
        self._draw_domain()
        self._draw_params()
        self._refresh_preview()
        self._draw_preview()
        self._draw_actions()

    def _draw_presets(self) -> None:
        """A wrapped row of one-click preset icons."""
        imgui.text_disabled("Presets")
        imgui.same_line()
        widgets.help_marker(
            "Load an example expression, its domain and its parameter sliders. "
            "Hover an icon for the formula."
        )

        size = 24.0
        spacing = imgui.get_style().item_spacing.x
        avail = max(size, imgui.get_content_region_available().x)
        per_row = max(1, int((avail + spacing) // (size + spacing)))

        for index, preset in enumerate(PRESETS):
            if index % per_row != 0:
                imgui.same_line()
            if icon_button(
                f"##preset_{index}",
                preset.icon,
                size=size,
                tooltip=f"{preset.label}\n{preset.expr}",
            ):
                self._apply_preset(preset)

    def _apply_preset(self, preset: Preset) -> None:
        """Load a preset's expression, domain and parameters."""
        self.expr = preset.expr
        self.x_min = preset.x_min
        self.x_max = preset.x_max
        self.samples = preset.samples
        self.log_spacing = False
        for name, value, vmin, vmax in preset.params:
            self.params[name] = Param(value, vmin, vmax)
        # Force both gates: the preset may reuse the current text (re-clicking the same
        # preset is a "reset to defaults" gesture) and must still re-derive and re-plot.
        self._synced_expr = None
        self._signature = None
        self._error = None
        self._status = f"Loaded preset: {preset.label}"
        self._status_ok = True

    def _draw_expression(self) -> None:
        """The multiline expression editor."""
        imgui.text("f(x) =")
        imgui.same_line()
        widgets.help_marker(_HELP_EXPRESSION)

        # buffer_length is omitted deliberately (CONTRACT §2.4): an explicit length
        # silently truncates, and the default grows the buffer.
        changed, value = imgui.input_text_multiline(
            "##expression",
            self.expr,
            width=-1.0,
            height=imgui.get_text_line_height() * 3.0,
        )
        if changed:
            self.expr = value

    def _draw_domain(self) -> None:
        """x min / x max / sample count / log spacing."""
        if not widgets.section("Domain", default_open=True):
            return

        field_width = _half_field_width("x min", "x max")

        imgui.set_next_item_width(field_width)
        changed, value = imgui.input_float("x min", self.x_min, 0.0, 0.0, "%.4g")
        if changed:
            self.x_min = float(value)
        imgui.same_line()
        imgui.set_next_item_width(field_width)
        changed, value = imgui.input_float("x max", self.x_max, 0.0, 0.0, "%.4g")
        if changed:
            self.x_max = float(value)

        # input_int draws its -/+ step buttons *inside* the item width, so it needs the
        # two extra frame-height squares on top of the text box.
        step_room = 2.0 * (imgui.get_frame_height() + imgui.get_style().item_inner_spacing.x)
        imgui.set_next_item_width(field_width + step_room)
        changed, value = imgui.input_int("Samples", self.samples)
        if changed:
            self.samples = int(np.clip(value, 2, MAX_SAMPLES))
        imgui.same_line()
        changed, value = imgui.checkbox("Log spacing", self.log_spacing)
        if changed:
            self.log_spacing = bool(value)
        imgui.same_line()
        widgets.help_marker(_HELP_LOG)

        if self.is_field:
            self._draw_y_domain(field_width)

    def _draw_y_domain(self, field_width: float) -> None:
        """The second axis, shown only once the expression asks for one."""
        imgui.spacing()
        imgui.text_disabled("Field (y is an axis)")
        imgui.same_line()
        widgets.help_marker(
            "The expression names y, so it is plotted as a field over x and y rather "
            "than a curve: y is a second axis, not a parameter slider. The result is an "
            "image layer. Rename it (to 'k', say) to get the slider back."
        )

        imgui.set_next_item_width(field_width)
        changed, value = imgui.input_float("y min", self.y_min, 0.0, 0.0, "%.4g")
        if changed:
            self.y_min = float(value)
        imgui.same_line()
        imgui.set_next_item_width(field_width)
        changed, value = imgui.input_float("y max", self.y_max, 0.0, 0.0, "%.4g")
        if changed:
            self.y_max = float(value)

        step_room = 2.0 * (imgui.get_frame_height() + imgui.get_style().item_inner_spacing.x)
        imgui.set_next_item_width(field_width + step_room)
        changed, value = imgui.input_int("y samples", self.y_samples)
        if changed:
            self.y_samples = int(np.clip(value, 2, MAX_SAMPLES))

    def _draw_params(self) -> None:
        """One live slider per auto-detected free variable."""
        if not widgets.section("Parameters", default_open=True):
            return

        if not self._param_order:
            imgui.text_disabled("No parameters.")
            imgui.same_line()
            widgets.help_marker(
                "Any name in the expression other than x and the built-in functions "
                "becomes a slider here. Try a * sin(b * x)."
            )
            return

        changed, value = imgui.checkbox("Edit ranges", self._edit_ranges)
        if changed:
            self._edit_ranges = bool(value)
        imgui.same_line()
        widgets.help_marker("Show min/max inputs for each slider's range.")

        for name in self._param_order:
            param = self.params.get(name)
            if param is None:  # pragma: no cover - _sync_params always fills these in
                param = self.params.setdefault(name, Param())
            imgui.push_id(name)
            self._draw_param_row(name, param)
            imgui.pop_id()

    def _draw_param_row(self, name: str, param: Param) -> None:
        """One parameter: reset button, slider, and optionally its range inputs."""
        if icon_button("##reset", "refresh", size=20.0, tooltip=f"Reset {name} to 1"):
            param.value = DEFAULT_PARAM_VALUE
        imgui.same_line()

        # A slider whose bounds are inverted or collapsed silently refuses to move;
        # order them here rather than fighting the user's half-typed range.
        lo, hi = sorted((param.vmin, param.vmax))
        if lo == hi:
            hi = lo + 1.0
        imgui.set_next_item_width(-1.0 if not self._edit_ranges else -160.0)
        changed, value = imgui.slider_float(name, param.value, lo, hi, format="%.4g")
        if changed:
            param.value = float(value)

        if self._edit_ranges:
            imgui.same_line()
            imgui.set_next_item_width(70.0)
            changed, value = imgui.input_float("##min", param.vmin, 0.0, 0.0, "%.3g")
            if changed:
                param.vmin = float(value)
            imgui.same_line()
            imgui.set_next_item_width(70.0)
            changed, value = imgui.input_float("##max", param.vmax, 0.0, 0.0, "%.3g")
            if changed:
                param.vmax = float(value)

    def _draw_preview(self) -> None:
        """The live curve, plus the inline error if the current text is invalid."""
        if self.is_field:
            self._draw_field_preview()
            if self._error:
                widgets.error_box(self._error)
            return

        if self._curve is not None:
            xs, ys = self._curve
            label = self.expr.strip().replace("\n", " ")
            if len(label) > 48:
                label = label[:45] + "..."
            widgets.mini_plot("##fn_preview", ys, x=xs, height=110.0, label=label)
            if self._preview_truncated:
                imgui.text_disabled(
                    f"Preview at {len(xs)} of {self._clamped_samples()} samples "
                    "(actions use the full count)."
                )
        else:
            imgui.text_disabled("No curve yet.")

        if self._error:
            widgets.error_box(self._error)

    def _draw_field_actions(self) -> None:
        """Output controls for the field mode: how to draw it, a colormap, and Plot.

        No "Create dataset" or "Replace layer": both are about a two-column table and a
        field is a matrix, so there is no honest column pair to hand them.

        The **output** picker is what makes this panel dimensionality-aware. The same
        ``z = f(x, y)`` is a heat map in 2D and a surface in 3D — the expression does not
        change, only what is built from it — so the choice belongs here rather than in two
        near-identical panels.
        """
        imgui.set_next_item_width(_half_field_width("Draw as"))
        labels = [label for _, label in _FIELD_OUTPUTS]
        current = dict(_FIELD_OUTPUTS).get(self.field_output, labels[0])
        changed, picked = widgets.enum_combo("Draw as", current, labels)
        if changed:
            for key, label in _FIELD_OUTPUTS:
                if label == picked:
                    self.field_output = key
                    break
        imgui.same_line()
        widgets.help_marker(_HELP_FIELD_OUTPUT)

        imgui.set_next_item_width(_half_field_width("Colormap"))
        _changed, self.cmap = widgets.enum_combo(
            "Colormap", self.cmap, list(layerops.CPU_COLORMAP_NAMES)
        )

        if self.field_output == "wireframe":
            imgui.set_next_item_width(_half_field_width("Stride"))
            changed, value = imgui.slider_int("Stride", int(self.wire_stride), 1, 32)
            if changed:
                self.wire_stride = int(value)
            imgui.same_line()
            widgets.help_marker(
                "Draw every Nth grid line. On a fine grid every line is a sheet of solid "
                "ink; every 4th reads as a surface."
            )

        imgui.spacing()
        ready = self._error is None and self._field is not None
        if not ready:
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
        if imgui.button("Plot", 100.0, 0.0) and ready:
            self._action_plot_field()
        if not ready:
            imgui.pop_style_var(1)
        imgui.same_line()
        widgets.help_marker(
            "Evaluates the expression at the full grid size and adds it to the scene. An "
            "image cannot be converted to another plot type, so the Scene panel's type "
            "menu will be greyed out for it — its colormap is editable there instead. A "
            "surface or wireframe switches the figure to a 3D view."
        )

        if self._status:
            imgui.spacing()
            token = "ok" if self._status_ok else "warn"
            imgui.text_colored(self._status, *COLORS[token])

    def _action_plot_field(self) -> None:
        """Add the field to the scene, undoably, in whichever form ``field_output`` asks.

        Evaluated here at the full grid, not reused from the preview: the preview is
        capped to `PREVIEW_MAX_CELLS` and plotting what it happened to sample would
        silently hand over a coarser field than the domain the user set.
        """
        try:
            matrix = self._evaluate_field(*self._field_shape())
        except (ExpressionError, ValueError) as exc:
            self._error = str(exc)
            self._status, self._status_ok = "Nothing plotted.", False
            return

        if self.field_output in ("surface", "wireframe"):
            self._action_plot_field_3d(matrix)
            return

        extent = (self.x_min, self.x_max, self.y_min, self.y_max)
        label = self._label()
        plot, hud = self.plot, self.hud
        cmap = self.cmap
        holder: Dict[str, Any] = {}

        def do() -> None:
            holder["layer"] = layerops.add_image_layer(plot, matrix, extent, label=label, cmap=cmap)

        def undo() -> None:
            layer = holder.pop("layer", None)
            if layer is not None:
                layerops.remove_layer(plot, hud, layer)

        self.push_command(Command(label=f"Plot field {label}", do=do, undo=undo))
        nx, ny = self._field_shape()
        self._status = f"Plotted {label} as a {nx} x {ny} image."
        self._status_ok = True

    def _action_plot_field_3d(self, matrix: np.ndarray) -> None:
        """Add the field as a 3D surface or wireframe, undoably.

        ``evaluate_2d`` returns the grid as ``(len(y), len(x))`` — row ``i`` is ``y[i]``,
        column ``j`` is ``x[j]`` — which is exactly the row-major ``(nv, nu)`` order
        :func:`glplot.gui.layerops3d.resolve_grid` reshapes with. Passing the shape
        explicitly rather than letting it detect an ``(x, y)`` lattice costs nothing and
        removes the one way this could go wrong: a domain with a repeated sample.
        """
        nx, ny = self._field_shape()
        xs = self._build_x(nx)
        ys = self._build_y(ny)
        grid_x, grid_y = np.meshgrid(xs, ys)

        kind = "surface3d" if self.field_output == "surface" else "wireframe3d"
        options: Dict[str, Any] = {"nu": nx, "nv": ny}
        if kind == "wireframe3d":
            options["stride"] = int(self.wire_stride)

        flat_x = grid_x.ravel()
        flat_y = grid_y.ravel()
        flat_z = np.asarray(matrix, dtype=np.float64).ravel()
        label = self._label()
        plot, hud = self.plot, self.hud
        cmap = self.cmap
        width = self.line_width
        holder: Dict[str, Any] = {}

        def do() -> None:
            if not plot.active_panel.is_3d():
                plot.set_ndim(3)
            holder["layer"] = layerops3d.add_xyz_layer(
                plot,
                flat_x,
                flat_y,
                flat_z,
                kind=kind,
                label=label,
                options=options,
                cmap=cmap,
                width=width,
            )
            # The axes name themselves: the expression's own variables are x and y, and
            # the value it computes is z.
            plot.axes3d.xlabel = plot.axes3d.xlabel or "x"
            plot.axes3d.ylabel = plot.axes3d.ylabel or "y"
            plot.axes3d.zlabel = plot.axes3d.zlabel or self.expr
            plot.frame_3d_view()

        def undo() -> None:
            layer = holder.pop("layer", None)
            if layer is not None:
                layerops.remove_layer(plot, hud, layer)

        self.push_command(Command(label=f"Plot {kind} {label}", do=do, undo=undo))
        self._status = f"Plotted {label} as a {nx} x {ny} {self.field_output}."
        self._status_ok = True

    def _draw_field_preview(self) -> None:
        """What can honestly be shown of a field without a second plot renderer.

        ``mini_plot`` draws a curve; there is no widget here that draws a grid, and
        picking one row of the field to show as a curve would be a preview of something
        the user is not plotting. So this reports the grid instead — its size, and the
        range the colormap will be stretched across, which is the number that decides
        whether the image comes out flat.
        """
        if self._field is None:
            imgui.text_disabled("No field yet.")
            return

        nx, ny = self._field_shape()
        widgets.stat_row("Grid", f"{nx} x {ny}  ({nx * ny:,} cells)")
        finite = self._field[np.isfinite(self._field)]
        if len(finite):
            widgets.stat_row("Range", f"{float(finite.min()):.4g} .. {float(finite.max()):.4g}")
        else:
            imgui.text_disabled("Every cell is NaN or infinite — nothing to colour.")
        if self._preview_truncated:
            imgui.text_disabled(
                f"Sampled at {self._field.shape[1]} x {self._field.shape[0]} for this "
                "readout (Plot uses the full grid)."
            )

    def _draw_actions(self) -> None:
        """Output style controls and the three action buttons."""
        if not widgets.section("Output", default_open=True):
            return

        if self.is_field:
            self._draw_field_actions()
            return

        imgui.set_next_item_width(_half_field_width("Kind"))
        # Every kind `add_xy_layer` builds, read from the registry rather than copied:
        # a hardcoded pair here is a curve the user can compute but cannot plot.
        kind_changed, self.kind = widgets.enum_combo("Kind", self.kind, layerops.KIND_KEYS)
        if kind_changed:
            self.kind_opts = layerops.default_kind_options(self.kind)

        imgui.same_line()
        # NO_INPUTS: a swatch that opens the full picker on click. The three R/G/B
        # spinboxes would eat the row and this is not a colour-grading panel.
        changed, value = imgui.color_edit3("Color", *self.color, flags=imgui.COLOR_EDIT_NO_INPUTS)
        if changed:
            self.color = list(value)

        # Which control to show follows the kind's `layer_type`, not the kind's name:
        # `step`/`stem` are polylines too, and `area`/`bar`/`hist` are patches, which
        # `add_xy_layer` builds from `face_color` alone -- it passes them neither width
        # nor size. Each of these would otherwise be a slider that moves nothing.
        layer_type = layerops.kind_spec(self.kind).layer_type
        if layer_type == "polyline":
            imgui.set_next_item_width(_label_room("Line width"))
            changed, value = widgets.labeled_drag_float(
                "Line width", self.line_width, speed=0.05, vmin=0.1, vmax=20.0, fmt="%.2f"
            )
            if changed:
                self.line_width = value
        elif layer_type == "scatter":
            imgui.set_next_item_width(_label_room("Point size"))
            changed, value = widgets.labeled_drag_float(
                "Point size", self.point_size, speed=0.1, vmin=0.5, vmax=64.0, fmt="%.1f"
            )
            if changed:
                self.point_size = value

        widgets.kind_options_editor(self.kind_opts)

        imgui.spacing()
        ready = self._error is None and self._curve is not None

        if not ready:
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
        if imgui.button("Plot", 100.0, 0.0) and ready:
            self._action_plot()
        imgui.same_line()
        if imgui.button("Create dataset", 120.0, 0.0) and ready:
            self._action_create_dataset()
        if not ready:
            imgui.pop_style_var(1)

        # "Plot live" — the screen-sampled variant. Kept next to "Plot" rather than hidden
        # behind a mode switch because the two produce genuinely different objects and the
        # choice belongs to the user: a table of samples, or the function itself.
        live_ready = ready and not self.is_field
        if not live_ready:
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
        if imgui.button("Plot live", 100.0, 0.0) and live_ready:
            self._action_plot_live()
        hovered = imgui.is_item_hovered()
        if not live_ready:
            imgui.pop_style_var(1)
        if hovered:
            imgui.set_tooltip(
                "A field over x and y is a surface, not a curve, so there is nothing to "
                "resample per pixel column."
                if self.is_field
                else _HELP_PLOT_LIVE
            )

        target = self._replace_target()
        imgui.same_line()
        if icon_button(
            "##replace",
            "upload",
            size=imgui.get_frame_height(),
            tooltip=(
                f"Replace the data of '{getattr(target, 'label', '?')}'"
                if target is not None
                else "Replace selected layer\n(select a line, scatter or line-family layer first)"
            ),
            enabled=ready and target is not None,
        ):
            self._action_replace(target)

        if self._status:
            imgui.spacing()
            token = "ok" if self._status_ok else "warn"
            color = COLORS.get(token, (0.6, 0.6, 0.6, 1.0))
            imgui.text_colored(self._status, color[0], color[1], color[2], 1.0)

    # -- actions ---------------------------------------------------------------

    def _label(self) -> str:
        """A stable, explicit layer/dataset label (§1.5 forbids the engine default).

        ``f(x)`` or ``f(x, y)`` to match what was actually plotted: a field named ``f(x)``
        would misreport its own second axis in the layer list.
        """
        head = "f(x, y)" if self.is_field else "f(x)"
        text = " ".join(self.expr.split())
        if len(text) > 40:
            text = text[:37] + "..."
        return f"{head} = {text}" if text else head

    def _full_curve(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Re-evaluate at the requested sample count, surfacing failure in the panel."""
        try:
            return self._evaluate(self._clamped_samples())
        except (ExpressionError, ValueError) as exc:
            self._error = str(exc)
            self._status = "Nothing generated."
            self._status_ok = False
            return None

    def _replace_target(self) -> Optional[Any]:
        """The selected layer, if it is one whose data can be swapped."""
        interaction = getattr(self.plot, "interaction", None)
        layer_id = getattr(interaction, "selected_layer_id", None)
        if layer_id is None:
            return None
        layer = layerops.find_layer(self.plot, layer_id)
        if layer is None:
            return None
        if getattr(layer, "layer_type", "") not in _REPLACEABLE:
            return None
        return layer

    def _action_plot(self) -> None:
        """Add the curve as a new layer, undoably (CONTRACT §1.1: all of it deferred)."""
        curve = self._full_curve()
        if curve is None:
            return
        xs, ys = curve
        label = self._label()
        plot = self.plot
        hud = self.hud
        kind = self.kind
        color = tuple(self.color) + (1.0,)
        width = self.line_width
        size = self.point_size
        # Copied, not aliased: the command outlives this call, and a later slider drag
        # would otherwise rewrite what an already-queued (or undone) plot was built with.
        options = dict(self.kind_opts)

        # The layer created by `do` must be reachable from `undo`, and redo builds a
        # fresh one — hence a holder rather than a captured value.
        holder: Dict[str, Any] = {}

        def do() -> None:
            holder["layer"] = layerops.add_xy_layer(
                plot,
                xs,
                ys,
                kind=kind,
                label=label,
                color=color,
                width=width,
                size=size,
                options=options,
            )

        def undo() -> None:
            layer = holder.pop("layer", None)
            if layer is not None:
                layerops.remove_layer(plot, hud, layer)

        self.push_command(Command(label=f"Plot {label}", do=do, undo=undo))
        self._status = f"Plotted {len(xs)} points as '{label}'."
        self._status_ok = True

    def _action_plot_live(self) -> None:
        """Add the expression as a screen-sampled layer, undoably (CONTRACT §1.1).

        The mirror of :meth:`_action_plot`, differing in what is captured: that one
        captures the *points*, this one captures a compiled callable. Compiled here rather
        than inside ``do`` so an expression the validator rejects is reported in the panel's
        own error box, where the user is looking, instead of from inside the render loop.
        """
        try:
            func = expressions.compile_1d(self.expr, variables=self._bindings())
        except ExpressionError as exc:
            self._error = str(exc)
            self._status = "Nothing generated."
            self._status_ok = False
            return

        try:
            xlim = (float(self.x_min), float(self.x_max))
            if not all(np.isfinite(v) for v in xlim) or xlim[0] == xlim[1]:
                raise ValueError("x min and x max must be finite and different.")
        except (TypeError, ValueError) as exc:
            self._error = str(exc)
            self._status = "Nothing generated."
            self._status_ok = False
            return

        label = f"live {self._label()}"
        plot, hud = self.plot, self.hud
        color = tuple(self.color) + (1.0,)
        width = self.line_width
        holder: Dict[str, Any] = {}

        def do() -> None:
            holder["layer"] = layerops.add_function_layer(
                plot, func, xlim, label=label, color=color, width=width
            )

        def undo() -> None:
            layer = holder.pop("layer", None)
            if layer is not None:
                layerops.remove_layer(plot, hud, layer)

        self.push_command(Command(label=f"Plot live {label}", do=do, undo=undo))
        self._status = f"Plotted '{label}' live — it resamples as you zoom."
        self._status_ok = True

    def _action_create_dataset(self) -> None:
        """Add the curve to the DataStore as a two-column dataset."""
        curve = self._full_curve()
        if curve is None:
            return
        xs, ys = curve
        store = self.store
        dataset = DataSet(self._label(), [Column("x", xs), Column("y", ys)])
        dataset.source = "function"

        def do() -> None:
            store.add(dataset)
            # Reported from inside `do`, not at queue time: `add` unique-ifies the name
            # on collision, so the final name is only known once it has actually landed.
            self._status = f"Created dataset '{dataset.name}' ({len(xs)} rows)."
            self._status_ok = True

        def undo() -> None:
            store.remove(dataset)

        # Queued like every other action, so undo history stays in one strict order
        # even though the store itself is not part of the frame-unsafe scene state.
        self.push_command(Command(label=f"Create dataset {dataset.name}", do=do, undo=undo))

    def _action_replace(self, layer: Any) -> None:
        """Swap the selected layer's data for the generated curve."""
        if layer is None:
            return
        curve = self._full_curve()
        if curve is None:
            return
        xs, ys = curve
        plot = self.plot
        label = str(getattr(layer, "label", "layer"))

        old = snapshot(_layer_points(layer))

        def do() -> None:
            layerops.update_layer_xy(plot, layer, xs, ys)

        if old is None or old.ndim != 2 or old.shape[1] != 2:
            # Either the array is too large to snapshot (history.snapshot returns None)
            # or it is not the (N, 2) table update_layer_xy round-trips through. Run it
            # anyway, but be honest that it cannot be taken back.
            command = Command.not_undoable(f"Replace {label}", do)
            # UndoStack.push clears the whole history for a not-undoable command, which
            # the user is entitled to know about before it happens silently.
            self._status = f"Replaced '{label}': original too large to undo, history cleared."
            self._status_ok = False
        else:

            def undo() -> None:
                layerops.update_layer_xy(plot, layer, old[:, 0], old[:, 1])

            command = Command(label=f"Replace {label}", do=do, undo=undo)
            self._status = f"Replaced '{label}' with {len(xs)} points."
            self._status_ok = True

        self.push_command(command)


def _half_field_width(*labels: str) -> float:
    """Width for each of two side-by-side widgets whose labels must stay on screen.

    imgui draws a widget's label *after* the widget and does not count it in the item
    width, so sizing two fields at ``avail / 2`` pushes the second label off the right
    edge. Subtract the labels (widest wins, so the row stays symmetric) and the spacing
    imgui will insert between and after them.
    """
    style = imgui.get_style()
    label_width = max((imgui.calc_text_size(label).x for label in labels), default=0.0)
    overhead = 2.0 * (label_width + style.item_inner_spacing.x) + style.item_spacing.x
    avail = imgui.get_content_region_available().x
    return max(48.0, (avail - overhead) * 0.5)


def _label_room(label: str) -> float:
    """A negative item width that reserves room for ``label`` (imgui's -N convention)."""
    return -(imgui.calc_text_size(label).x + imgui.get_style().item_inner_spacing.x + 4.0)


def _layer_points(layer: Any) -> Optional[np.ndarray]:
    """Return a replaceable layer's (N, 2) data array, matching ``update_layer_xy``.

    ``line_family`` keeps its table in ``ab`` (whose columns are a/b, not x/y); the
    others keep it in ``pts``. Returns None for any other layer type.
    """
    layer_type = getattr(layer, "layer_type", "")
    if layer_type in ("scatter", "polyline"):
        return getattr(layer, "pts", None)
    if layer_type == "line_family":
        return getattr(layer, "ab", None)
    return None
