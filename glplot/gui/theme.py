"""Visual theme for the mathematical workstation GUI.

Defines the single source of truth for colour in the GUI: a small set of semantic
tokens in :data:`COLORS` plus a full Dear ImGui colour ramp derived from them.
Panels and widgets must read :data:`COLORS` rather than hardcoding literals, so a
theme switch is one call and one place.

Design rules this file follows (and callers should preserve):

* **One accent hue.** A single azure drives every interactive affordance --
  focus, selection, checkmarks, slider grabs, primary buttons. Semantic
  green/amber/red appear only for ok/warn/error, never for decoration.
* **Restrained neutrals.** Backgrounds step through four closely spaced greys;
  separation comes from 1px borders, not from high-contrast fills.
* **Every interactive colour has three states** (rest / hovered / active) and
  they are ordered consistently: hovered is brighter, active is deeper.

``apply_theme`` must be called after ``imgui.create_context()``. It is safe to
call with imgui missing -- the token table still updates so that non-imgui code
(previews, exporters) can share the palette.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

try:
    from imgui_bundle import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - imgui is a hard dependency in CI
    IMGUI_AVAILABLE = False
    imgui = None

RGBA = Tuple[float, float, float, float]

__all__ = [
    "COLORS",
    "IMGUI_AVAILABLE",
    "apply_theme",
    "color_u32",
    "get_color",
    "is_dark",
    "pop_accent",
    "push_accent",
]


def _mix(a: RGBA, b: RGBA, t: float) -> RGBA:
    """Linearly interpolate two RGBA tuples (``t=0`` -> ``a``, ``t=1`` -> ``b``)."""
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    )


def _alpha(c: RGBA, a: float) -> RGBA:
    """Return ``c`` with its alpha replaced by ``a``."""
    return (c[0], c[1], c[2], a)


# --- semantic palettes -------------------------------------------------------
# Tokens are the public contract. Panels read these; the imgui ramp below is
# derived from them so the two can never drift apart.

_DARK: Dict[str, RGBA] = {
    # surfaces, darkest to lightest
    "bg": (0.086, 0.090, 0.106, 1.00),
    "panel_bg": (0.118, 0.125, 0.145, 1.00),
    "frame_bg": (0.157, 0.165, 0.192, 1.00),
    "raised_bg": (0.196, 0.208, 0.239, 1.00),
    "border": (0.223, 0.235, 0.271, 1.00),
    # text
    "text": (0.878, 0.890, 0.918, 1.00),
    "text_dim": (0.482, 0.506, 0.561, 1.00),
    "muted": (0.482, 0.506, 0.561, 1.00),
    # the one accent hue
    "accent": (0.259, 0.588, 0.925, 1.00),
    "accent_hover": (0.361, 0.671, 0.976, 1.00),
    "accent_active": (0.180, 0.478, 0.804, 1.00),
    "on_accent": (1.000, 1.000, 1.000, 1.00),
    # status
    "ok": (0.298, 0.757, 0.475, 1.00),
    "warn": (0.937, 0.702, 0.259, 1.00),
    "err": (0.906, 0.353, 0.353, 1.00),
    # plotting surfaces (mini_plot and friends)
    "grid": (1.000, 1.000, 1.000, 0.07),
    "axis": (0.420, 0.443, 0.494, 1.00),
    "plot_line": (0.361, 0.671, 0.976, 1.00),
    "plot_overlay": (0.949, 0.596, 0.278, 1.00),
    "selection": (0.259, 0.588, 0.925, 0.30),
}

_LIGHT: Dict[str, RGBA] = {
    "bg": (0.965, 0.969, 0.976, 1.00),
    "panel_bg": (1.000, 1.000, 1.000, 1.00),
    "frame_bg": (0.925, 0.933, 0.945, 1.00),
    "raised_bg": (0.878, 0.890, 0.910, 1.00),
    "border": (0.808, 0.824, 0.851, 1.00),
    "text": (0.106, 0.122, 0.153, 1.00),
    "text_dim": (0.420, 0.451, 0.502, 1.00),
    "muted": (0.420, 0.451, 0.502, 1.00),
    "accent": (0.145, 0.451, 0.827, 1.00),
    "accent_hover": (0.216, 0.541, 0.898, 1.00),
    "accent_active": (0.102, 0.353, 0.678, 1.00),
    "on_accent": (1.000, 1.000, 1.000, 1.00),
    "ok": (0.086, 0.596, 0.325, 1.00),
    "warn": (0.706, 0.475, 0.055, 1.00),
    "err": (0.788, 0.196, 0.196, 1.00),
    "grid": (0.000, 0.000, 0.000, 0.08),
    "axis": (0.549, 0.573, 0.604, 1.00),
    "plot_line": (0.145, 0.451, 0.827, 1.00),
    "plot_overlay": (0.851, 0.408, 0.098, 1.00),
    "selection": (0.145, 0.451, 0.827, 0.28),
}

#: Semantic colour tokens for the active theme. Mutated in place by
#: :func:`apply_theme` -- ``from glplot.gui.theme import COLORS`` stays valid
#: across a theme switch, so never rebind this name.
COLORS: Dict[str, RGBA] = dict(_DARK)

_STATE = {"dark": True}


def is_dark() -> bool:
    """True when the dark palette is active. Useful for luminance-dependent choices."""
    return bool(_STATE["dark"])


def get_color(name: str, alpha: Optional[float] = None) -> RGBA:
    """Return a token as an RGBA tuple, optionally overriding its alpha.

    Unknown names fall back to ``text`` rather than raising: a missing token must
    never take down a draw callback mid-frame.
    """
    c = COLORS.get(name, COLORS["text"])
    return c if alpha is None else _alpha(c, alpha)


def color_u32(name: str, alpha: Optional[float] = None) -> int:
    """Return a token packed as an ``ImU32`` for ``draw_list`` calls.

    Returns ``0`` when imgui is unavailable; callers are already guarded and a
    packed colour has no meaning without a context.
    """
    if not IMGUI_AVAILABLE:
        return 0
    c = get_color(name, alpha)
    return imgui.get_color_u32((c[0], c[1], c[2], c[3]))


def _apply_style_vars() -> None:
    """Geometry half of the theme: rounding, spacing, borders."""
    style = imgui.get_style()

    # Rounding: soft but not bubbly. Frames stay tighter than windows so inputs
    # read as inset rather than floating.
    style.window_rounding = 6.0
    style.child_rounding = 6.0
    style.frame_rounding = 4.0
    style.popup_rounding = 6.0
    style.scrollbar_rounding = 8.0
    style.grab_rounding = 3.0
    style.tab_rounding = 4.0

    # Borders carry the separation, so every container gets exactly 1px.
    style.window_border_size = 1.0
    style.child_border_size = 1.0
    style.popup_border_size = 1.0
    style.frame_border_size = 1.0
    style.tab_border_size = 0.0

    # Spacing: generous vertical rhythm, compact cells for the data table.
    style.window_padding = (10.0, 10.0)
    style.frame_padding = (8.0, 4.0)
    style.item_spacing = (8.0, 6.0)
    style.item_inner_spacing = (6.0, 4.0)
    style.cell_padding = (6.0, 3.0)
    style.touch_extra_padding = (0.0, 0.0)
    style.indent_spacing = 18.0
    style.scrollbar_size = 12.0
    style.grab_min_size = 10.0

    # Alignment: left-aligned titles read as a document header, not a dialog.
    style.window_title_align = (0.0, 0.5)
    style.button_text_align = (0.5, 0.5)
    style.selectable_text_align = (0.0, 0.5)
    style.window_min_size = (200.0, 100.0)

    style.anti_aliased_lines = True
    style.anti_aliased_fill = True
    style.curve_tessellation_tol = 1.25
    style.alpha = 1.0


def _apply_style_colors() -> None:
    """Colour half of the theme: the full imgui ramp, derived from COLORS."""
    style = imgui.get_style()

    bg = COLORS["bg"]
    panel = COLORS["panel_bg"]
    frame = COLORS["frame_bg"]
    raised = COLORS["raised_bg"]
    border = COLORS["border"]
    text = COLORS["text"]
    dim = COLORS["text_dim"]
    accent = COLORS["accent"]
    accent_h = COLORS["accent_hover"]
    accent_a = COLORS["accent_active"]

    style.set_color_(imgui.Col_.text, text)
    style.set_color_(imgui.Col_.text_disabled, dim)

    style.set_color_(imgui.Col_.window_bg, panel)
    style.set_color_(imgui.Col_.child_bg, _alpha(bg, 0.40))
    style.set_color_(imgui.Col_.popup_bg, _mix(panel, bg, 0.35))
    style.set_color_(imgui.Col_.border, border)
    style.set_color_(imgui.Col_.border_shadow, (0.0, 0.0, 0.0, 0.0))

    style.set_color_(imgui.Col_.frame_bg, frame)
    style.set_color_(imgui.Col_.frame_bg_hovered, raised)
    style.set_color_(imgui.Col_.frame_bg_active, _mix(raised, accent, 0.22))

    style.set_color_(imgui.Col_.title_bg, bg)
    style.set_color_(imgui.Col_.title_bg_active, _mix(bg, accent, 0.14))
    style.set_color_(imgui.Col_.title_bg_collapsed, _alpha(bg, 0.75))
    style.set_color_(imgui.Col_.menu_bar_bg, bg)

    style.set_color_(imgui.Col_.scrollbar_bg, _alpha(bg, 0.60))
    style.set_color_(imgui.Col_.scrollbar_grab, raised)
    style.set_color_(imgui.Col_.scrollbar_grab_hovered, _mix(raised, text, 0.20))
    style.set_color_(imgui.Col_.scrollbar_grab_active, accent)

    style.set_color_(imgui.Col_.check_mark, accent_h)
    style.set_color_(imgui.Col_.slider_grab, accent)
    style.set_color_(imgui.Col_.slider_grab_active, accent_h)

    # Rest-state buttons are quiet: a frame-coloured fill with a border. The
    # accent is reserved for push_accent() primaries and for hover feedback.
    style.set_color_(imgui.Col_.button, frame)
    style.set_color_(imgui.Col_.button_hovered, _mix(frame, accent, 0.45))
    style.set_color_(imgui.Col_.button_active, accent_a)

    style.set_color_(imgui.Col_.header, _alpha(accent, 0.22))
    style.set_color_(imgui.Col_.header_hovered, _alpha(accent, 0.38))
    style.set_color_(imgui.Col_.header_active, _alpha(accent, 0.55))

    style.set_color_(imgui.Col_.separator, border)
    style.set_color_(imgui.Col_.separator_hovered, _alpha(accent, 0.70))
    style.set_color_(imgui.Col_.separator_active, accent)

    style.set_color_(imgui.Col_.resize_grip, _alpha(dim, 0.30))
    style.set_color_(imgui.Col_.resize_grip_hovered, _alpha(accent, 0.55))
    style.set_color_(imgui.Col_.resize_grip_active, _alpha(accent, 0.85))

    style.set_color_(imgui.Col_.tab, _mix(bg, frame, 0.50))
    style.set_color_(imgui.Col_.tab_hovered, _alpha(accent, 0.45))
    style.set_color_(imgui.Col_.tab_selected, _mix(frame, accent, 0.30))
    style.set_color_(imgui.Col_.tab_dimmed, bg)
    style.set_color_(imgui.Col_.tab_dimmed_selected, _mix(bg, frame, 0.70))

    style.set_color_(imgui.Col_.plot_lines, COLORS["plot_line"])
    style.set_color_(imgui.Col_.plot_lines_hovered, COLORS["plot_overlay"])
    style.set_color_(imgui.Col_.plot_histogram, COLORS["plot_line"])
    style.set_color_(imgui.Col_.plot_histogram_hovered, COLORS["plot_overlay"])

    # Table colours matter as much as anything here: the data editor is a grid
    # the user stares at for minutes at a time. Alternating rows stay under 3%
    # contrast -- enough to track a row, not enough to strobe.
    style.set_color_(imgui.Col_.table_header_bg, _mix(frame, bg, 0.30))
    style.set_color_(imgui.Col_.table_border_strong, border)
    style.set_color_(imgui.Col_.table_border_light, _mix(bg, border, 0.55))
    style.set_color_(imgui.Col_.table_row_bg, (0.0, 0.0, 0.0, 0.0))
    style.set_color_(imgui.Col_.table_row_bg_alt, _alpha(text, 0.025))

    style.set_color_(imgui.Col_.text_selected_bg, COLORS["selection"])
    style.set_color_(imgui.Col_.drag_drop_target, _alpha(COLORS["warn"], 0.90))
    style.set_color_(imgui.Col_.nav_cursor, accent)
    style.set_color_(imgui.Col_.nav_windowing_highlight, _alpha(text, 0.70))
    style.set_color_(imgui.Col_.nav_windowing_dim_bg, _alpha(dim, 0.20))
    style.set_color_(imgui.Col_.modal_window_dim_bg, (0.0, 0.0, 0.0, 0.55))


def apply_theme(dark: bool = True) -> None:
    """Install the GLPlot theme into the current imgui context.

    Updates :data:`COLORS` in place first (so the tokens are correct even when
    imgui is unavailable), then applies the style vars and the full colour ramp.
    Must be called after ``imgui.create_context()``.
    """
    _STATE["dark"] = bool(dark)
    COLORS.clear()
    COLORS.update(_DARK if dark else _LIGHT)

    if not IMGUI_AVAILABLE:
        return

    _apply_style_vars()
    _apply_style_colors()


def push_accent() -> None:
    """Style the next button(s) as the primary action. Pair with :func:`pop_accent`.

    Pushes exactly four colours; ``pop_accent`` pops exactly four. Every return
    path between the two -- including early-outs and exceptions handled by the
    caller -- must reach the pop.
    """
    if not IMGUI_AVAILABLE:
        return
    a = COLORS["accent"]
    ah = COLORS["accent_hover"]
    aa = COLORS["accent_active"]
    on = COLORS["on_accent"]
    imgui.push_style_color(imgui.Col_.button, a)
    imgui.push_style_color(imgui.Col_.button_hovered, ah)
    imgui.push_style_color(imgui.Col_.button_active, aa)
    imgui.push_style_color(imgui.Col_.text, on)


def pop_accent() -> None:
    """Undo :func:`push_accent`. Pops the four colours it pushed."""
    if not IMGUI_AVAILABLE:
        return
    imgui.pop_style_color(4)
