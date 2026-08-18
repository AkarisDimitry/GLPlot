"""The dynamical-systems panel — pick Lorenz, tweak sliders, watch the attractor draw.

A thin, stateful shell over :mod:`glplot.gui.dynamics` exactly as the Functions panel is a
shell over :mod:`glplot.gui.expressions`. It owns the editable description of a system --
the state-variable names, one derivative expression per variable, the parameter values and
their slider bounds, the initial state and the integration window -- and rebuilds a
:class:`~glplot.gui.dynamics.DynamicalSystem` from those fields to integrate a trajectory.

**One editable path, presets just fill it.** There is no read-only "built-in" mode. The ten
preloaded systems are buttons that *load their definition into the editable fields*; from
there Lorenz and a system you typed yourself are the same object. This mirrors the Functions
panel, where a preset loads an expression you are then free to edit.

**Auto-detected parameters.** Every free name in the equations that is neither a state
variable nor ``t`` becomes a slider, via :func:`glplot.gui.expressions.free_variables`.
Loading a preset seeds the slider bounds; editing an equation to introduce a new name grows
a slider for it with default bounds. Slider values persist by name even after a name leaves
the equations, so deleting and retyping ``rho`` restores the value you had dialled in.

**Change-gated integration.** ``draw`` runs every frame; integrating a 10k-step trajectory
there would make every slider drag unusable. Integration (:meth:`_refresh_preview`) runs only
when a signature of every input changes, and the preview is capped to
:data:`PREVIEW_MAX_STEPS` steps regardless of the requested count -- the Plot / Create-dataset
actions re-integrate at full resolution, so dragging a slider stays interactive with a
million steps queued behind it. An invalid system shows its error inline and keeps the last
good trajectory on screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import dynamics, expressions, layerops, layerops3d, widgets
from ..datasets import DataSet
from ..dynamics import DynamicalSystem, DynamicsError
from ..expressions import ExpressionError
from ..history import Command
from ..icons import icon_button
from ..theme import COLORS
from .base import Panel

try:
    from imgui_bundle import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - exercised only on GL-less systems
    IMGUI_AVAILABLE = False
    imgui = None


#: The preview never integrates more than this many steps no matter what the user asked
#: for. The full step count is used by Plot / Create dataset.
PREVIEW_MAX_STEPS = 2500

#: Prune remembered parameter values beyond this many so a long session cannot grow
#: unbounded (see the module docstring on persistence-by-name).
MAX_REMEMBERED_PARAMS = 128

#: Default slider bounds for a parameter discovered in an edited equation.
DEFAULT_PARAM_MIN = -10.0
DEFAULT_PARAM_MAX = 10.0
DEFAULT_PARAM_VALUE = 1.0

_TIME_NAME = "t"

#: The "no third axis" entry of the Z picker: a 2D projection, the historical behaviour.
_NO_Z_AXIS = "(2D)"

_HELP_EQUATIONS = (
    "Each row is the time derivative of one state variable: for variables x, y the rows "
    "are dx/dt and dy/dt. Write a numpy expression in the variables, the parameters, and "
    "t (time, for forced systems). The same functions as the Functions panel are "
    "available: sin cos exp sqrt abs ... Any bare name that is not a variable or t becomes "
    "a parameter slider."
)

_HELP_VARIABLES = (
    "Comma-separated state-variable names (e.g. x, y, z). One equation row appears per "
    "variable. Names must be valid identifiers and cannot be 't' or a built-in like 'e'."
)

_HELP_TRANSIENT = (
    "Steps integrated and discarded before recording, to let the trajectory settle onto "
    "its attractor or limit cycle. Set 0 to keep the initial transient."
)


@dataclass
class Param:
    """One auto-detected parameter: its live value and its slider bounds."""

    value: float = DEFAULT_PARAM_VALUE
    vmin: float = DEFAULT_PARAM_MIN
    vmax: float = DEFAULT_PARAM_MAX


class DynamicsPanel(Panel):
    """Generate an ODE trajectory from a system of derivatives, with live sliders."""

    title = "Dynamics"
    icon = "wave"
    default_open = False

    def __init__(self, ws: Any) -> None:
        super().__init__(ws)

        #: The current system's editable definition. Seeded from Lorenz.
        self.label: str = "Lorenz"
        self.var_text: str = "x, y, z"
        #: var name -> derivative expression. Kept for names not currently present too,
        #: so toggling a variable back restores its equation.
        self.equations: Dict[str, str] = {}
        #: parameter name -> Param, including names no longer referenced.
        self.params: Dict[str, Param] = {}
        #: variable name -> initial value, persisted by name like the equations.
        self.initial: Dict[str, float] = {}

        self.dt: float = 0.01
        self.steps: int = 8000
        self.transient: int = 500

        #: Phase-portrait axes; either may be ``t``.
        self.x_axis: str = "x"
        self.y_axis: str = "z"
        #: Third phase-space axis. "" plots a 2D projection, which is the historical
        #: behaviour and stays the default; a name plots the trajectory as a 3D curve.
        self.z_axis: str = ""
        #: The 3D plot kind used when a Z axis is chosen.
        self.kind3d: str = "line3d"

        #: Output style for the plotted layer.
        self.kind: str = "line"
        self.color: List[float] = [0.26, 0.59, 0.98]
        self.line_width: float = 1.2
        self.point_size: float = 3.0
        self.kind_opts: Dict[str, Any] = layerops.default_kind_options(self.kind)

        #: Derived, in slider order: the free names of the current equations.
        self._param_order: List[str] = []

        # Change gates. `_model_sig` is what `_param_order`/`initial` were synced from;
        # `_preview_sig` is every input the preview trajectory depends on.
        self._model_sig: Optional[Tuple[Any, ...]] = None
        self._preview_sig: Optional[Tuple[Any, ...]] = None

        self._traj: Optional[DataSet] = None
        self._preview_truncated: bool = False
        self._error: Optional[str] = None
        self._status: str = ""
        self._status_ok: bool = True
        self._edit_ranges: bool = False

        self._apply_system(dynamics.LORENZ)

    # -- model plumbing --------------------------------------------------------

    def _current_variables(self) -> List[str]:
        """Parse the variable-name text into a de-duplicated identifier list."""
        raw = self.var_text.replace(";", ",").replace(" ", ",")
        names: List[str] = []
        for token in raw.split(","):
            token = token.strip()
            if token and token not in names:
                names.append(token)
        return names

    def _apply_system(self, system: DynamicalSystem) -> None:
        """Load a preloaded system's definition into the editable fields."""
        self.label = system.label
        self.var_text = ", ".join(system.variables)
        for var, eq in zip(system.variables, system.equations):
            self.equations[var] = eq
        for name, value, vmin, vmax in system.params:
            self.params[name] = Param(float(value), float(vmin), float(vmax))
        initial = system.initial or (0.0,) * len(system.variables)
        for var, value in zip(system.variables, initial):
            self.initial[var] = float(value)
        self.dt = float(system.dt)
        self.steps = int(system.steps)
        self.transient = int(system.transient)
        self.x_axis, self.y_axis = system.plot_axes()
        # Force both gates: re-loading the same preset is a reset gesture and must re-sync.
        self._model_sig = None
        self._preview_sig = None
        self._error = None
        self._status = f"Loaded system: {system.label}"
        self._status_ok = True

    def _sync_model(self) -> None:
        """Re-derive the parameter/initial lists from the equations. Gated on the text.

        A mid-edit equation that will not parse leaves the previous slider set in place --
        the error box says what is wrong, and yanking sliders on every keystroke would make
        the panel jump around.
        """
        variables = self._current_variables()
        signature = (self.var_text, tuple(self.equations.get(v, "") for v in variables))
        if signature == self._model_sig:
            return
        self._model_sig = signature

        # Ensure an equation and an initial value exist for every variable.
        for var in variables:
            self.equations.setdefault(var, "0")
            self.initial.setdefault(var, 0.0)

        # Auto-detect parameters: any free name in any equation that is not a variable or t.
        exclude = set(variables) | {_TIME_NAME}
        order: List[str] = []
        for var in variables:
            try:
                names = expressions.free_variables(self.equations.get(var, "0"), exclude=exclude)
            except ExpressionError:
                # Keep the last good parameter set for a half-typed equation.
                return
            for name in names:
                if name not in order:
                    order.append(name)
        self._param_order = order
        for name in order:
            self.params.setdefault(name, Param())
        self._prune_params()

    def _prune_params(self) -> None:
        """Drop remembered parameter values beyond the cap, keeping referenced ones."""
        if len(self.params) <= MAX_REMEMBERED_PARAMS:
            return
        keep = set(self._param_order)
        for name in list(self.params):
            if len(self.params) <= MAX_REMEMBERED_PARAMS:
                break
            if name not in keep:
                del self.params[name]

    def _build_system(self) -> DynamicalSystem:
        """Assemble a DynamicalSystem from the editable fields. May raise on a bad system."""
        variables = self._current_variables()
        equations = tuple(self.equations.get(var, "0") for var in variables)
        params = tuple(
            (name, self.params[name].value, self.params[name].vmin, self.params[name].vmax)
            for name in self._param_order
            if name in self.params
        )
        initial = tuple(float(self.initial.get(var, 0.0)) for var in variables)
        return DynamicalSystem(
            key="custom",
            label=self.label.strip() or "system",
            icon="wave",
            description="",
            variables=tuple(variables),
            equations=equations,
            params=params,
            initial=initial,
            dt=self.dt,
            steps=max(dynamics.MIN_STEPS, min(int(self.steps), dynamics.MAX_STEPS)),
            transient=max(0, int(self.transient)),
            plot=(self.x_axis, self.y_axis),
        )

    def _columns(self) -> List[str]:
        """The trajectory's column names for the axis pickers: t then every variable."""
        return [_TIME_NAME, *self._current_variables()]

    # -- evaluation ------------------------------------------------------------

    def _preview_signature(self) -> Tuple[Any, ...]:
        variables = tuple(self._current_variables())
        return (
            variables,
            tuple(self.equations.get(v, "") for v in variables),
            tuple((n, self.params[n].value) for n in self._param_order if n in self.params),
            tuple(self.initial.get(v, 0.0) for v in variables),
            self.dt,
            int(self.steps),
            int(self.transient),
        )

    def _refresh_preview(self) -> None:
        """Re-integrate the preview trajectory when any input changed. Gated per frame."""
        signature = self._preview_signature()
        if signature == self._preview_sig:
            return
        self._preview_sig = signature

        try:
            system = self._build_system()
            steps = min(int(self.steps), PREVIEW_MAX_STEPS)
            self._preview_truncated = steps < int(self.steps)
            transient = min(int(self.transient), PREVIEW_MAX_STEPS)
            self._traj = dynamics.integrate(system, steps=steps, transient=transient)
            self._error = None
        except (DynamicsError, ExpressionError, ValueError) as exc:
            self._error = str(exc)

    def _full_trajectory(self) -> Optional[DataSet]:
        """Integrate at the full requested step count, surfacing failure in the panel."""
        try:
            system = self._build_system()
            return dynamics.integrate(system)
        except (DynamicsError, ExpressionError, ValueError) as exc:
            self._error = str(exc)
            self._status = "Nothing generated."
            self._status_ok = False
            return None

    # -- draw ------------------------------------------------------------------

    def draw(self) -> None:
        """Render the panel body. The workspace owns the surrounding begin/end."""
        if not IMGUI_AVAILABLE:
            return

        self._draw_presets()
        imgui.separator()
        self._draw_definition()
        # Order matters: sliders reflect the equations above them, the preview reflects the
        # sliders, all within one frame.
        self._sync_model()
        self._draw_params()
        self._draw_initial()
        self._draw_integration()
        self._refresh_preview()
        self._draw_preview()
        self._draw_actions()

    def _draw_presets(self) -> None:
        """A wrapped row of one-click preset icons for the ten preloaded systems."""
        imgui.text_disabled("Systems")
        imgui.same_line()
        widgets.help_marker(
            "Load a classic system's equations, parameters and initial state. Hover an "
            "icon for its name. Everything stays editable afterwards."
        )

        size = 24.0
        spacing = imgui.get_style().item_spacing.x
        avail = max(size, imgui.get_content_region_avail().x)
        per_row = max(1, int((avail + spacing) // (size + spacing)))

        for index, system in enumerate(dynamics.SYSTEMS.values()):
            if index % per_row != 0:
                imgui.same_line()
            if icon_button(
                f"##dynsys_{system.key}",
                system.icon,
                size=size,
                tooltip=f"{system.label}\n{system.description}",
            ):
                self._apply_system(system)

    def _draw_definition(self) -> None:
        """The variable-name field and one derivative editor per variable."""
        imgui.set_next_item_width(-1.0)
        changed, value = imgui.input_text("Variables", self.var_text)
        if changed:
            self.var_text = value
        imgui.same_line(0.0, 4.0)
        widgets.help_marker(_HELP_VARIABLES)

        variables = self._current_variables()
        if not variables:
            widgets.error_box("Name at least one state variable, e.g. x, y, z.")
            return

        imgui.text("Derivatives")
        imgui.same_line()
        widgets.help_marker(_HELP_EQUATIONS)
        for var in variables:
            imgui.push_id(f"eq_{var}")
            label = f"d{var}/dt"
            imgui.set_next_item_width(-(imgui.calc_text_size(label).x + 16.0))
            changed, value = imgui.input_text(label, self.equations.get(var, "0"))
            if changed:
                self.equations[var] = value
            imgui.pop_id()

    def _draw_params(self) -> None:
        """One live slider per auto-detected parameter."""
        if not self._param_order:
            return
        if not widgets.section("Parameters", default_open=True):
            return

        changed, value = imgui.checkbox("Edit ranges", self._edit_ranges)
        if changed:
            self._edit_ranges = bool(value)
        imgui.same_line()
        widgets.help_marker("Show min/max inputs for each slider's range.")

        for name in self._param_order:
            param = self.params.setdefault(name, Param())
            imgui.push_id(f"param_{name}")
            self._draw_param_row(name, param)
            imgui.pop_id()

    def _draw_param_row(self, name: str, param: Param) -> None:
        """One parameter: slider, and optionally its range inputs."""
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

    def _draw_initial(self) -> None:
        """One input per variable for the initial state."""
        variables = self._current_variables()
        if not variables:
            return
        if not widgets.section("Initial state", default_open=False):
            return
        for var in variables:
            imgui.push_id(f"init_{var}")
            imgui.set_next_item_width(140.0)
            label = f"{var}0"
            changed, value = imgui.input_float(
                label, float(self.initial.get(var, 0.0)), 0.0, 0.0, "%.4g"
            )
            if changed:
                self.initial[var] = float(value)
            imgui.pop_id()

    def _draw_integration(self) -> None:
        """Step size, recorded steps, and discarded transient."""
        if not widgets.section("Integration", default_open=True):
            return

        imgui.set_next_item_width(140.0)
        changed, value = imgui.input_float("dt", self.dt, 0.0, 0.0, "%.4g")
        if changed and np.isfinite(value) and value != 0.0:
            self.dt = float(value)

        imgui.set_next_item_width(140.0)
        changed, value = imgui.input_int("Steps", int(self.steps))
        if changed:
            self.steps = int(np.clip(value, dynamics.MIN_STEPS, dynamics.MAX_STEPS))

        imgui.set_next_item_width(140.0)
        changed, value = imgui.input_int("Transient", int(self.transient))
        if changed:
            self.transient = int(np.clip(value, 0, dynamics.MAX_STEPS))
        imgui.same_line()
        widgets.help_marker(_HELP_TRANSIENT)

    def _draw_preview(self) -> None:
        """The live phase portrait (x-axis vs y-axis columns), plus any inline error."""
        columns = self._columns()
        # Keep the axis pickers pointed at columns that still exist.
        if self.x_axis not in columns:
            self.x_axis = columns[0]
        if self.y_axis not in columns:
            self.y_axis = columns[1] if len(columns) > 1 else columns[0]

        imgui.set_next_item_width(120.0)
        _changed, self.x_axis = widgets.enum_combo("X axis", self.x_axis, columns)
        imgui.same_line()
        imgui.set_next_item_width(120.0)
        _changed, self.y_axis = widgets.enum_combo("Y axis", self.y_axis, columns)

        # A phase portrait of a three-variable system projected onto two axes throws the
        # attractor's structure away — the Lorenz butterfly only reads as a butterfly in
        # three. The Z picker turns the same trajectory into a space curve.
        if len(columns) > 2:
            imgui.same_line()
            imgui.set_next_item_width(120.0)
            options = [_NO_Z_AXIS] + list(columns)
            current = self.z_axis if self.z_axis in columns else _NO_Z_AXIS
            _changed, picked = widgets.enum_combo("Z axis", current, options)
            self.z_axis = "" if picked == _NO_Z_AXIS else picked
            imgui.same_line()
            widgets.help_marker(
                "Pick a third variable to plot the trajectory as a 3D space curve instead "
                "of a 2D projection. The figure switches to a 3D view; the preview above "
                "stays 2D."
            )
        elif self.z_axis:
            # The system lost a variable while a Z axis was selected.
            self.z_axis = ""

        if self._traj is not None:
            xs = self._traj.get(self.x_axis)
            ys = self._traj.get(self.y_axis)
            if xs is not None and ys is not None:
                label = f"{self.y_axis} vs {self.x_axis}"
                widgets.mini_plot("##dyn_preview", ys, x=xs, height=140.0, label=label)
            if self._preview_truncated:
                imgui.text_disabled(
                    f"Preview at {PREVIEW_MAX_STEPS} of {int(self.steps)} steps "
                    "(actions use the full count)."
                )
        else:
            imgui.text_disabled("No trajectory yet.")

        if self._error:
            widgets.error_box(self._error)

    def _draw_actions(self) -> None:
        """Output style controls and the two action buttons."""
        if not widgets.section("Output", default_open=True):
            return

        imgui.set_next_item_width(140.0)
        kind_changed, self.kind = widgets.enum_combo("Kind", self.kind, layerops.KIND_KEYS)
        if kind_changed:
            self.kind_opts = layerops.default_kind_options(self.kind)
        imgui.same_line()
        changed, value = imgui.color_edit3(
            "Color", self.color, flags=imgui.ColorEditFlags_.no_inputs
        )
        if changed:
            self.color = list(value)

        layer_type = layerops.kind_spec(self.kind).layer_type
        if layer_type == "polyline":
            imgui.set_next_item_width(160.0)
            changed, value = widgets.labeled_drag_float(
                "Line width", self.line_width, speed=0.05, vmin=0.1, vmax=20.0, fmt="%.2f"
            )
            if changed:
                self.line_width = value
        elif layer_type == "scatter":
            imgui.set_next_item_width(160.0)
            changed, value = widgets.labeled_drag_float(
                "Point size", self.point_size, speed=0.1, vmin=0.5, vmax=64.0, fmt="%.1f"
            )
            if changed:
                self.point_size = value

        widgets.kind_options_editor(self.kind_opts)

        imgui.spacing()
        ready = self._error is None and self._traj is not None
        if not ready:
            imgui.push_style_var(imgui.StyleVar_.alpha, imgui.get_style().alpha * 0.5)
        if imgui.button("Plot", (100.0, 0.0)) and ready:
            self._action_plot()
        imgui.same_line()
        if imgui.button("Create dataset", (130.0, 0.0)) and ready:
            self._action_create_dataset()
        if not ready:
            imgui.pop_style_var(1)

        if self._status:
            imgui.spacing()
            token = "ok" if self._status_ok else "warn"
            color = COLORS.get(token, (0.6, 0.6, 0.6, 1.0))
            imgui.text_colored((color[0], color[1], color[2], 1.0), self._status)

    # -- actions ---------------------------------------------------------------

    def _action_plot(self) -> None:
        """Add the chosen phase portrait as a new layer, undoably (§1.1).

        Two shapes, chosen by whether a Z axis is selected: a 2D projection through
        ``layerops.add_xy_layer``, or a 3D space curve through
        ``layerops3d.add_xyz_layer``. The 3D branch is what makes an attractor readable —
        a Lorenz projected onto two axes is a scribble.
        """
        traj = self._full_trajectory()
        if traj is None:
            return
        if self.z_axis:
            self._action_plot_3d(traj)
            return
        xs = traj.get(self.x_axis)
        ys = traj.get(self.y_axis)
        if xs is None or ys is None:  # pragma: no cover - pickers are kept valid
            self._error = f"Axis columns {self.x_axis!r}/{self.y_axis!r} are not in the trajectory."
            return
        xs = np.array(xs, copy=True)
        ys = np.array(ys, copy=True)
        label = f"{self.label}: {self.y_axis} vs {self.x_axis}"
        plot, hud = self.plot, self.hud
        kind = self.kind
        color = tuple(self.color) + (1.0,)
        width, size = self.line_width, self.point_size
        options = dict(self.kind_opts)
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

    def _action_plot_3d(self, traj: DataSet) -> None:
        """Add the trajectory as a 3D space curve, colour-mapped along time.

        Coloured by ``t`` rather than flat: a closed orbit drawn in one colour hides the
        direction of travel and the number of windings, which are usually the two things
        the portrait was drawn to show.
        """
        columns = [self.x_axis, self.y_axis, self.z_axis]
        arrays = [traj.get(name) for name in columns]
        if any(a is None for a in arrays):  # pragma: no cover - pickers are kept valid
            self._error = f"Axis columns {columns} are not all in the trajectory."
            return
        xs, ys, zs = (np.array(a, copy=True) for a in arrays)

        time = traj.get("t")
        cvalues = None if time is None else np.array(time, copy=True)
        label = f"{self.label}: {self.z_axis} vs {self.y_axis} vs {self.x_axis}"
        plot, hud = self.plot, self.hud
        kind = self.kind3d
        width, size = self.line_width, self.point_size
        holder: Dict[str, Any] = {}

        def do() -> None:
            if not plot.active_panel.is_3d():
                plot.set_ndim(3)
            holder["layer"] = layerops3d.add_xyz_layer(
                plot,
                xs,
                ys,
                zs,
                kind=kind,
                label=label,
                color=tuple(self.color) + (1.0,),
                width=width,
                size=size,
                c=cvalues,
                cmap="viridis",
            )
            # Name the box's axes after the phase-space variables the user picked; an
            # unlabelled attractor is a pretty shape rather than a plot.
            plot.axes3d.xlabel = self.x_axis
            plot.axes3d.ylabel = self.y_axis
            plot.axes3d.zlabel = self.z_axis
            plot.frame_3d_view()

        def undo() -> None:
            layer = holder.pop("layer", None)
            if layer is not None:
                layerops.remove_layer(plot, hud, layer)

        self.push_command(Command(label=f"Plot {label}", do=do, undo=undo))
        self._status = f"Plotted {len(xs)} points as a 3D '{label}'."
        self._status_ok = True

    def _action_create_dataset(self) -> None:
        """Add the full trajectory (t + every variable) to the DataStore, undoably."""
        traj = self._full_trajectory()
        if traj is None:
            return
        store = self.store

        def do() -> None:
            store.add(traj)
            self._status = (
                f"Created dataset '{traj.name}' ({traj.n_rows()} rows, {traj.n_cols()} cols)."
            )
            self._status_ok = True

        def undo() -> None:
            store.remove(traj)

        self.push_command(Command(label=f"Create dataset {traj.name}", do=do, undo=undo))
