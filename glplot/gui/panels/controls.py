"""The Controls panel -- the figure's sliders, buttons, toggles and menus, drawn live.

The panel is a view of ``plot.controls`` (:class:`glplot.controls.ControlSet`); it owns no
controls of its own. User code defines them (``gplt.slider``, ``gplt.interact`` ...), the
panel draws them as an inspector -- label column on the left, widget on the right -- and
writes what the user does back through :meth:`Control.set` / :meth:`Button.click`.

**Why writing from draw is allowed here.** CONTRACT §1.1 forbids mutating the *scene* from
a draw callback. Setting a control does not: while the window runs, the set only records
the change and wakes the loop, and the callbacks run at the top of the next frame from
``GPULinePlot._flush_controls`` -- the same slot the command queue drains in. That is
also what coalesces a drag into one callback run per frame.

Two widget behaviours worth knowing:

* ``live=False`` sliders and text boxes keep a *draft* here while the user drags or types
  and commit it once, on release (``is_item_deactivated_after_edit``). An expensive
  callback then runs once per gesture instead of once per frame.
* A button with ``confirm=`` opens a modal first. Palette and keyboard invocations go
  through :meth:`ControlsPanel.request_click`, so a shortcut cannot bypass the question.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from ... import controls as ctl
from .. import notifications, widgets
from ..icons import icon_button
from ..theme import COLORS
from .base import Panel

try:
    from imgui_bundle import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - exercised only on GL-less systems
    IMGUI_AVAILABLE = False
    imgui = None

logger = logging.getLogger(__name__)

#: Label column bounds, in points. The column fits the longest label in the group, but
#: never squeezes the widgets below 55% of the panel or collapses under a short label.
_LABEL_MIN_W = 64.0
_LABEL_MAX_FRAC = 0.45

_CONFIRM_POPUP = "Confirm##controls_confirm"
_PRESET_POPUP = "##controls_presets"

_EMPTY_HELP = (
    "No controls yet. Define them from Python and they appear here:\n\n"
    '  amp = gplt.slider("Amplitude", 0, 5)\n'
    '  mode = gplt.dropdown("Mode", ["fast", "exact"])\n\n'
    "  @gplt.interact(freq=(0.1, 10))\n"
    "  def curve(freq=1.0):\n"
    "      return x, np.sin(freq * x)\n\n"
    '  @gplt.button("Randomize", shortcut="Mod+R")\n'
    "  def randomize(): ..."
)


class ControlsPanel(Panel):
    """Draws the figure's interactive controls and the results of its bound functions."""

    title = "Controls"
    icon = "sliders"
    default_open = False

    def __init__(self, ws: Any) -> None:
        super().__init__(ws)
        #: In-progress values of ``live=False`` controls, keyed by ``id(control)``.
        self._drafts: Dict[int, Any] = {}
        #: Edit buffers of text boxes, keyed by ``id(control)``.
        self._text: Dict[int, str] = {}
        #: Button awaiting its ``confirm`` answer.
        self._confirm_target: Optional[ctl.Button] = None
        self._open_confirm = False
        self._preset_name = ""

    # -- model access ----------------------------------------------------------------

    @property
    def controls(self) -> ctl.ControlSet:
        return self.plot.controls

    def request_click(self, button: ctl.Button) -> None:
        """Press ``button`` the way its definition asks: via the confirm modal if it has one.

        The palette and keyboard shortcuts come through here too. The modal is drawn by
        this panel, so the workspace must keep the panel open while one is pending.
        """
        if not button.enabled:
            return
        if button.confirm:
            self._confirm_target = button
            self._open_confirm = True
            return
        self._click(button)

    def _click(self, button: ctl.Button) -> None:
        try:
            button.click()
        except Exception as exc:  # script-mode dispatch (tests): surface, do not crash
            logger.exception("Button %r raised", button.label)
            notifications.push(f"{button.label}: {type(exc).__name__}: {exc}", kind="error")

    def _commit(self, control: ctl.Control, value: Any) -> None:
        try:
            control.set(value)
        except ctl.ControlError as exc:
            notifications.push(str(exc), kind="error")
        except Exception as exc:  # script-mode dispatch (tests): surface, do not crash
            logger.exception("Control %r callback raised", control.label)
            notifications.push(f"{control.label}: {type(exc).__name__}: {exc}", kind="error")

    # -- frame -----------------------------------------------------------------------

    def draw(self) -> None:
        if not IMGUI_AVAILABLE:
            return
        cs = self.controls
        self._draw_toolbar(cs)
        self._draw_problems(cs)

        groups = [(name, [c for c in members if c.visible]) for name, members in cs.groups()]
        groups = [(name, members) for name, members in groups if members]
        if not groups and not cs.bindings:
            imgui.spacing()
            imgui.push_text_wrap_pos(0.0)
            imgui.text_disabled(_EMPTY_HELP)
            imgui.pop_text_wrap_pos()
        # A lone default group needs no header: the window is already titled "Controls".
        headed = not (len(groups) == 1 and groups[0][0] == ctl.DEFAULT_GROUP)
        # One label column for the whole panel, so widgets line up across groups.
        label_w = self._label_width([c for _, members in groups for c in members])
        for name, members in groups:
            if headed and not widgets.section(f"{name}##ctlgroup_{name}"):
                continue
            imgui.push_id(f"ctlgroup_{name}")
            self._draw_group(members, label_w)
            imgui.pop_id()
            imgui.spacing()

        self._draw_results(cs)
        self._draw_functions(cs)
        self._draw_confirm()

    # -- toolbar ---------------------------------------------------------------------

    def _draw_toolbar(self, cs: ctl.ControlSet) -> None:
        inputs = [c for c in cs if c.is_input]
        modified = any(not c._same(c.value, c.default) for c in inputs)
        if icon_button(
            "##ctl_reset",
            "refresh",
            tooltip="Reset every control to its default",
            enabled=modified,
        ):
            self._reset_all(cs)
        imgui.same_line()
        if icon_button(
            "##ctl_presets",
            "save",
            tooltip="Presets: save these values under a name, or restore saved ones",
            enabled=bool(inputs),
        ):
            imgui.open_popup(_PRESET_POPUP)
        imgui.same_line()
        if icon_button(
            "##ctl_copy",
            "copy",
            tooltip="Copy the current values as a Python dict",
            enabled=bool(inputs),
        ):
            imgui.set_clipboard_text(repr(cs.snapshot()))
            notifications.push("Control values copied to the clipboard.", kind="success")
        self._draw_presets_popup(cs)

        n_fn = len(cs.bindings)
        parts = [f"{len(inputs)} control{'s' if len(inputs) != 1 else ''}"]
        if n_fn:
            parts.append(f"{n_fn} function{'s' if n_fn != 1 else ''}")
            if cs.last_flush_ms > 0:
                parts.append(f"{cs.last_flush_ms:.1f} ms")
        imgui.same_line()
        imgui.align_text_to_frame_padding()
        imgui.text_disabled("  " + " · ".join(parts))
        imgui.separator()

    def _reset_all(self, cs: ctl.ControlSet) -> None:
        self._drafts.clear()
        self._text.clear()
        try:
            cs.reset()
        except Exception as exc:  # script-mode dispatch (tests)
            notifications.push(f"Reset: {type(exc).__name__}: {exc}", kind="error")

    def _draw_presets_popup(self, cs: ctl.ControlSet) -> None:
        if not imgui.begin_popup(_PRESET_POPUP):
            return
        try:
            imgui.text("Presets")
            imgui.separator()
            if not cs.presets:
                imgui.text_disabled("None saved yet.")
            for name in list(cs.presets):
                imgui.push_id(f"preset_{name}")
                if imgui.small_button("x"):
                    cs.delete_preset(name)
                    imgui.pop_id()
                    continue
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Delete this preset")
                imgui.same_line()
                clicked, _ = imgui.selectable(name, False)
                if clicked:
                    self._drafts.clear()
                    self._text.clear()
                    try:
                        cs.load_preset(name)
                    except Exception as exc:
                        notifications.push(f"Preset {name!r}: {exc}", kind="error")
                    imgui.close_current_popup()
                imgui.pop_id()
            imgui.separator()
            imgui.set_next_item_width(160.0)
            _, self._preset_name = imgui.input_text_with_hint(
                "##preset_name", "name", self._preset_name
            )
            imgui.same_line()
            name = self._preset_name.strip()
            imgui.begin_disabled(not name)
            if imgui.button("Save current"):
                cs.save_preset(name)
                notifications.push(f"Saved preset {name!r}.", kind="success")
                self._preset_name = ""
            imgui.end_disabled()
        finally:
            imgui.end_popup()

    def _draw_problems(self, cs: ctl.ControlSet) -> None:
        """Failing functions, at the top: an error below the fold is an error unseen."""
        for binding in cs.bindings:
            if binding.error:
                widgets.error_box(f"{binding.label}: {binding.error}")

    # -- controls --------------------------------------------------------------------

    def _label_width(self, members: Sequence[ctl.Control]) -> float:
        widest = max(
            (imgui.calc_text_size(c.label).x for c in members if c.kind != "button"),
            default=0.0,
        )
        avail = imgui.get_content_region_avail().x
        return max(_LABEL_MIN_W, min(widest + 14.0, avail * _LABEL_MAX_FRAC))

    def _draw_group(self, members: List[ctl.Control], label_w: float) -> None:
        pending_buttons: List[ctl.Button] = []
        for control in members:
            if isinstance(control, ctl.Button):
                pending_buttons.append(control)
                continue
            if pending_buttons:
                self._draw_buttons(pending_buttons)
                pending_buttons = []
            imgui.push_id(f"ctl_{control.key}")
            try:
                self._draw_row(control, label_w)
            finally:
                imgui.pop_id()
        if pending_buttons:
            self._draw_buttons(pending_buttons)

    def _draw_row(self, control: ctl.Control, label_w: float) -> None:
        modified = control.is_input and not control._same(control.value, control.default)
        imgui.align_text_to_frame_padding()
        if modified:
            imgui.text_colored(COLORS["accent"], control.label)
        elif not control.enabled:
            imgui.text_disabled(control.label)
        else:
            imgui.text(control.label)
        if imgui.is_item_hovered():
            tip = control.help
            if modified:
                tip = (tip + "\n\n" if tip else "") + "Double-click to reset to the default."
            if tip:
                imgui.set_tooltip(tip)
            if modified and imgui.is_mouse_double_clicked(0):
                self._drafts.pop(id(control), None)
                self._text.pop(id(control), None)
                self._commit(control, control.default)

        imgui.same_line(label_w)
        imgui.set_next_item_width(-1.0)
        imgui.begin_disabled(not control.enabled)
        try:
            drawer = _DRAWERS.get(control.kind)
            if drawer is None:
                imgui.text_disabled(repr(control.value))
            else:
                drawer(self, control)
        finally:
            imgui.end_disabled()
        self._draw_context_menu(control)
        if control.error:
            widgets.error_box(control.error)

    def _draw_context_menu(self, control: ctl.Control) -> None:
        if not imgui.begin_popup_context_item("##ctx"):
            return
        try:
            if control.is_input:
                default = ctl.format_value(getattr(control, "default", None))
                if imgui.menu_item(f"Reset to default ({default})", "", False)[0]:
                    self._drafts.pop(id(control), None)
                    self._text.pop(id(control), None)
                    self._commit(control, control.default)
            if imgui.menu_item("Copy value", "", False)[0]:
                imgui.set_clipboard_text(str(control.to_json()))
            if imgui.menu_item("Copy key", "", False)[0]:
                imgui.set_clipboard_text(control.key)
        finally:
            imgui.end_popup()

    # -- one drawer per kind ---------------------------------------------------------

    def _live_or_draft(self, control: Any, changed: bool, value: Any) -> None:
        """Commit now (live) or keep a draft until the widget is released."""
        if getattr(control, "live", True):
            if changed:
                self._commit(control, value)
            return
        if changed:
            self._drafts[id(control)] = value
        if imgui.is_item_deactivated_after_edit() and id(control) in self._drafts:
            self._commit(control, self._drafts.pop(id(control)))

    def _draw_slider(self, control: ctl.Slider) -> None:
        current = self._drafts.get(id(control), control.value)
        if control.integer:
            changed, value = imgui.slider_int(
                "##v", int(current), int(control.min), int(control.max), control.fmt
            )
        else:
            flags = imgui.SliderFlags_.logarithmic if control.log else 0
            changed, value = imgui.slider_float(
                "##v", float(current), float(control.min), float(control.max), control.fmt, flags
            )
        self._live_or_draft(control, changed, value)

    def _draw_range(self, control: ctl.RangeSlider) -> None:
        lo, hi = self._drafts.get(id(control), control.value)
        speed = max(control.step or 0.0, (control.max - control.min) / 400.0)
        if control.integer:
            changed, lo, hi = imgui.drag_int_range2(
                "##v", int(lo), int(hi), float(speed), int(control.min), int(control.max)
            )
        else:
            changed, lo, hi = imgui.drag_float_range2(
                "##v",
                float(lo),
                float(hi),
                float(speed),
                float(control.min),
                float(control.max),
                control.fmt,
            )
        self._live_or_draft(control, changed, (lo, hi))

    def _draw_checkbox(self, control: ctl.Checkbox) -> None:
        changed, value = imgui.checkbox("##v", bool(control.value))
        if changed:
            self._commit(control, value)

    def _draw_dropdown(self, control: ctl.Dropdown) -> None:
        current = control.index
        if control.style == "radio":
            for i, label in enumerate(control.labels):
                if i:
                    imgui.same_line()
                if imgui.radio_button(f"{label}##opt{i}", i == current) and i != current:
                    self._commit(control, control.values[i])
            return
        if imgui.begin_combo("##v", control.labels[current]):
            try:
                for i, label in enumerate(control.labels):
                    clicked, _ = imgui.selectable(f"{label}##opt{i}", i == current)
                    if clicked and i != current:
                        self._commit(control, control.values[i])
                    if i == current:
                        imgui.set_item_default_focus()
            finally:
                imgui.end_combo()

    def _draw_text(self, control: ctl.TextBox) -> None:
        key = id(control)
        buf = self._text.get(key, control.value)
        if control.multiline:
            height = imgui.get_text_line_height() * 4 + 8.0
            changed, buf = imgui.input_text_multiline("##v", buf, (-1.0, height))
        elif control.placeholder:
            changed, buf = imgui.input_text_with_hint("##v", control.placeholder, buf)
        else:
            changed, buf = imgui.input_text("##v", buf)
        active = imgui.is_item_active()
        if control.live and changed:
            self._commit(control, buf)
        if imgui.is_item_deactivated_after_edit():
            self._commit(control, buf)
            active = False
        if active:
            self._text[key] = buf
        else:
            # Not being edited: follow the model, so a value set from code shows up.
            self._text.pop(key, None)

    def _draw_number(self, control: ctl.Number) -> None:
        value = control.value
        speed = control.step or max(abs(float(value)) * 0.005, 0.01)
        bounded = control.min is not None and control.max is not None
        lo = control.min if bounded else 0.0
        hi = control.max if bounded else 0.0
        if control.integer:
            changed, new = imgui.drag_int(
                "##v", int(value), max(float(speed), 0.2), int(lo), int(hi), control.fmt
            )
        else:
            changed, new = imgui.drag_float(
                "##v", float(value), float(speed), float(lo), float(hi), control.fmt
            )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                (control.help + "\n\n" if control.help else "")
                + "Drag to change, double-click to type."
            )
        if changed:
            self._commit(control, new)

    def _draw_color(self, control: ctl.ColorPicker) -> None:
        rgba = list(control.value)
        # Hex, not three R/G/B fields: one compact, copyable value; the swatch opens the
        # full picker for anyone who wants sliders.
        flags = imgui.ColorEditFlags_.display_hex
        if control.alpha:
            changed, new = imgui.color_edit4("##v", rgba, flags | imgui.ColorEditFlags_.alpha_bar)
        else:
            changed, new = imgui.color_edit3("##v", rgba[:3], flags)
            new = list(new) + [rgba[3]]
        if changed:
            self._commit(control, new)

    def _draw_readout(self, control: ctl.Readout) -> None:
        imgui.align_text_to_frame_padding()
        imgui.text(control.text)

    def _draw_buttons(self, buttons: List[ctl.Button]) -> None:
        """A flowing row of buttons, wrapping onto a new line when the panel is narrow."""
        style = imgui.get_style()
        right_edge = imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x
        for i, button in enumerate(buttons):
            if not button.visible:
                continue
            width = imgui.calc_text_size(button.label).x + 2.0 * style.frame_padding.x
            if i:
                imgui.same_line()
                if imgui.get_cursor_screen_pos().x + width > right_edge:
                    imgui.new_line()
            imgui.push_id(f"ctlbtn_{button.key}")
            imgui.begin_disabled(not button.enabled)
            if imgui.button(button.label):
                self.request_click(button)
            imgui.end_disabled()
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                tip = button.help or button.label
                action = getattr(self.ws, "registry", None)
                action = action.get(f"controls.{button.key}") if action is not None else None
                if action is not None and action.chord is not None:
                    from .. import keys

                    tip += f"  ({keys.format(action.chord)})"
                imgui.set_tooltip(tip)
            imgui.pop_id()
            if button.error:
                widgets.error_box(button.error)
        imgui.spacing()

    # -- functions -------------------------------------------------------------------

    def _draw_results(self, cs: ctl.ControlSet) -> None:
        with_outputs = [b for b in cs.bindings if b.outputs]
        if not with_outputs or not widgets.section("Results##ctl_results"):
            return
        for binding in with_outputs:
            rows = [(name, ctl.format_value(value)) for name, value in binding.outputs.items()]
            if len(with_outputs) > 1:
                imgui.text_colored(COLORS["accent"], binding.label)
            widgets.stats_table(f"##ctl_out_{id(binding)}", ("Quantity", "Value"), rows)
            imgui.spacing()

    def _draw_functions(self, cs: ctl.ControlSet) -> None:
        bindings = cs.bindings
        if not bindings or not widgets.section("Functions##ctl_functions", default_open=False):
            return
        for binding in bindings:
            imgui.push_id(f"ctlfn_{id(binding)}")
            try:
                changed, enabled = imgui.checkbox("##auto", binding.enabled)
                if changed:
                    binding.enabled = enabled
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Re-run automatically when its controls change")
                imgui.same_line()
                if imgui.small_button("Run"):
                    try:
                        binding.run()
                    except Exception as exc:  # script-mode dispatch (tests)
                        notifications.push(f"{binding.label}: {exc}", kind="error")
                imgui.same_line()
                imgui.text(binding.label)
                imgui.same_line()
                deps = ", ".join(binding.params) or "no controls"
                timing = f"{binding.runs} runs, {binding.last_ms:.1f} ms" if binding.runs else ""
                imgui.text_disabled(f"({deps}) {timing}")
            finally:
                imgui.pop_id()

    def _draw_confirm(self) -> None:
        if self._open_confirm:
            imgui.open_popup(_CONFIRM_POPUP)
            self._open_confirm = False
        target = self._confirm_target
        if target is None:
            return
        answer = widgets.confirm_popup(_CONFIRM_POPUP, str(target.confirm))
        if answer is None:
            return
        self._confirm_target = None
        if answer:
            self._click(target)


_DRAWERS = {
    "slider": ControlsPanel._draw_slider,
    "range": ControlsPanel._draw_range,
    "checkbox": ControlsPanel._draw_checkbox,
    "dropdown": ControlsPanel._draw_dropdown,
    "text": ControlsPanel._draw_text,
    "number": ControlsPanel._draw_number,
    "color": ControlsPanel._draw_color,
    "readout": ControlsPanel._draw_readout,
}
