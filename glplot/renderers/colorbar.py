"""The live colorbar overlay: a gradient strip with its own tick axis, drawn beside a panel.

A colorbar is deliberately **not** a :class:`~glplot.core.panel.Panel`: anything registered
on ``fig.panels`` is click-to-activate and pan/zoom-able (``GPULinePlot._panel_index_at``),
so a colorbar built that way would silently steal "current axes" the moment a script did
``im = plt.imshow(...); plt.colorbar(); plt.title(...)`` -- the title would land on the
colorbar. Instead, ``colorbar()`` (in :mod:`glplot.pyplot`) shrinks the host panel's own
``rect_frac`` and records a :class:`~glplot.core.panel.ColorbarSpec` describing the vacated
strip; :func:`draw_colorbars` paints those specs once per frame, entirely with
``imgui.get_background_draw_list()`` -- no GL state, no camera, no MVP -- in the same
whole-window pixel space :meth:`GPULinePlot._draw_panel_borders` already draws the panel
frames in.

The gradient's *color content* is always ``cmap`` swept linearly across the bar, regardless
of ``norm``: after normalisation every colormap operates on a plain 0..1 domain by
definition, so ``norm`` cannot change what color a given *fraction* of the bar is -- only
where a given *data value* lands along it. That is exactly what the tick locator uses it
for: real ``matplotlib.ticker`` locators (chosen by the norm's type) pick "nice" data
values, and each one is placed at ``norm(value)`` along the bar, the same callable
``pyplot._normalize_cvalues`` already uses to color scatter/imshow/contour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

import numpy as np

from .legend import _background_is_light

if TYPE_CHECKING:
    from ..core.panel import ColorbarSpec

#: Gradient stops chained along the bar. A plain ``cmap(fraction)`` sweep -- see the module
#: docstring for why this needs no `norm` -- so more stops only buys smoothness, not
#: correctness; 128 is comfortably seamless at any bar length this renders.
_GRADIENT_STOPS = 128

#: Outward offsets, in logical px, for the tick stroke and its label.
_TICK_LEN_PX = 5.0
_TICK_LABEL_GAP_PX = 6.0
_AXIS_LABEL_GAP_PX = 6.0

#: Live text sizes, in px, for a bar's ticks and its axis label.
#:
#: The headless export draws the same two runs at 18/22 pt (``preview.py``'s
#: ``COLORBAR_TICK_FONTSIZE``/``COLORBAR_LABEL_FONTSIZE``). These are px rather than pt --
#: imgui sizes glyphs in px -- and are the closest match that stays legible next to the
#: engine's own ~13 px default chrome, which every other live label still uses.
_TICK_TEXT_PX = 15.0
_LABEL_TEXT_PX = 17.0

#: Gap, in logical px, left between an external bar and its host panel's edge on screen.
#: See :func:`_close_pad_gap` for why the live view cannot just use ``pad`` directly.
_EXTERNAL_BAR_GAP_PX = 10.0

#: For an `inset=True` bar, which edge its ticks/label point toward -- inward, since
#: there is no room outside the host panel for them (that is the whole reason to inset
#: the bar in the first place).
_OPPOSITE_SIDE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}

#: How far an inset bar's text is haloed, in logical px. An inset bar draws *over* the
#: plotted content, so its numbers need to stay legible against whatever is behind them
#: -- a one-pixel outline in the opposing ink does that without an opaque backing panel
#: hiding a rectangle of the very data the bar is describing.
_INSET_TEXT_HALO_PX = 1.0


def draw_colorbars(plot: Any) -> None:
    """Draw every panel's colorbars, once per frame, in whole-window pixel space.

    Called from the engine right after the per-panel overlay pass (axis labels + legend)
    and before the HUD update -- one call earlier than :meth:`GPULinePlot._draw_panel_borders`
    already runs at, for the same reason: both need the *whole* window's pixel dimensions,
    not a single panel's swapped-in ones (``_draw_panels`` only swaps those for the scene
    and per-panel overlay passes).
    """
    panels = list(getattr(plot, "panels", None) or [])
    if not any(getattr(p, "colorbars", None) for p in panels):
        return
    try:
        from imgui_bundle import imgui
    except (ImportError, Exception):  # noqa: B014 - imgui raises non-ImportError GL-less
        return

    # Deferred: pyplot imports the renderer manager (transitively, this module), so an
    # module-scope import here would be circular. Same idiom as
    # renderers/axis.py's `from .legend import _add_scaled_text`.
    from ..pyplot import _colormap_values

    width = float(getattr(plot, "width", 0) or 0)
    height = float(getattr(plot, "height", 0) or 0)
    if width <= 0 or height <= 0:
        return

    is_light = _background_is_light(getattr(plot, "options", None))
    ink = imgui.get_color_u32((0.15, 0.15, 0.15, 1.0) if is_light else (0.85, 0.85, 0.85, 1.0))
    draw_list = imgui.get_background_draw_list()

    for panel in panels:
        for cb in getattr(panel, "colorbars", None) or []:
            _draw_one(imgui, draw_list, width, height, panel, cb, ink, is_light, _colormap_values)


def _draw_one(imgui, draw_list, width, height, panel, cb, ink, is_light, colormap_values) -> None:
    x0f, y0f, wf, hf = cb.rect_frac
    left = x0f * width
    right = (x0f + wf) * width
    top = (1.0 - (y0f + hf)) * height
    bottom = (1.0 - y0f) * height
    if right <= left or bottom <= top:
        return  # Degenerate rect (e.g. a figure resized to near-zero) -- nothing to draw.

    # An inset bar has no room outside the panel for its ticks -- point them inward
    # instead. A tick label at the very end of the bar (t=0 or t=1) is *centred* on that
    # endpoint, so half its width overhangs past the bar's own edge -- clip everything to
    # the host panel's own rect so that overhang can never bleed into a neighbouring
    # panel or off the window, regardless of how the margins happen to size out.
    tick_side = _OPPOSITE_SIDE[cb.location] if cb.inset else cb.location
    # The halo must contrast with the INK, not with the theme: `ink` is already dark on a
    # light theme, so keying the halo off `is_light` the same way painted dark-on-dark and
    # left an inset bar's numbers muddier than no halo at all. The headless export strokes
    # white under dark text for exactly this reason (`preview.py::_finalize_colorbar`).
    halo = imgui.get_color_u32((1.0, 1.0, 1.0, 0.9) if is_light else (0.08, 0.08, 0.1, 0.9))
    text_halo = halo if cb.inset else None
    if cb.inset:
        px0f, py0f, pwf, phf = panel.rect_frac
        draw_list.push_clip_rect(
            (px0f * width, (1.0 - (py0f + phf)) * height),
            ((px0f + pwf) * width, (1.0 - py0f) * height),
            True,
        )
    else:
        left, top, right, bottom = _close_pad_gap(panel, width, height, cb, left, top, right, bottom)

    stops = np.asarray(
        colormap_values(
            np.linspace(0.0, 1.0, _GRADIENT_STOPS + 1, dtype=np.float32),
            cmap=cb.cmap,
            vmin=0.0,
            vmax=1.0,
        )
    )
    _draw_gradient(imgui, draw_list, left, top, right, bottom, stops, cb.orientation)
    draw_list.add_rect((left, top), (right, bottom), ink, rounding=0.0, thickness=1.0)

    widest_tick_w = 0.0
    for value in _resolve_ticks(cb):
        frac = min(max(float(cb.norm(value)), 0.0), 1.0)
        label = _format_tick(value, cb.format)
        w = _draw_tick(
            imgui, draw_list, left, top, right, bottom, frac, cb.orientation, tick_side,
            label, ink, text_halo,
        )
        widest_tick_w = max(widest_tick_w, w)

    if cb.label:
        _draw_axis_label(
            imgui, draw_list, left, top, right, bottom, cb.orientation, tick_side,
            cb.label, ink, widest_tick_w, text_halo,
        )

    if cb.inset:
        draw_list.pop_clip_rect()


def _close_pad_gap(panel, width, height, cb, left, top, right, bottom):
    """Slide an external bar back toward its host panel, and return the new rect.

    ``colorbar()`` reserves ``fraction + pad`` of the panel but places the bar flush
    against the panel's *original* outer edge, so the whole ``pad`` opens up as a gap
    between the shrunk panel and the bar. That is right for the headless export, whose
    matplotlib axes carries its own outer margins -- but a live panel reserves no outer
    figure margin at all and draws its ticks *inside* its own rect, so on screen the same
    geometry reads as a bar floating detached from its plot.

    Drawing-time only: ``cb.rect_frac`` is left untouched, because the export reads it
    verbatim to place the same bar and must keep agreeing with it.

    The shift is clamped at zero (never pushes a bar *outward*) and skipped when the panel
    carries more than one bar on the same side, where sliding them all to the same edge
    would stack them on top of each other.
    """
    same_side = [c for c in (getattr(panel, "colorbars", None) or []) if c.location == cb.location]
    if len(same_side) > 1:
        return left, top, right, bottom

    px0f, py0f, pwf, phf = panel.rect_frac
    p_left, p_right = px0f * width, (px0f + pwf) * width
    p_top, p_bottom = (1.0 - (py0f + phf)) * height, (1.0 - py0f) * height

    if cb.location == "right":
        shift = max((left - p_right) - _EXTERNAL_BAR_GAP_PX, 0.0)
        return left - shift, top, right - shift, bottom
    if cb.location == "left":
        shift = max((p_left - right) - _EXTERNAL_BAR_GAP_PX, 0.0)
        return left + shift, top, right + shift, bottom
    if cb.location == "top":
        shift = max((p_top - bottom) - _EXTERNAL_BAR_GAP_PX, 0.0)
        return left, top + shift, right, bottom + shift
    shift = max((top - p_bottom) - _EXTERNAL_BAR_GAP_PX, 0.0)  # "bottom"
    return left, top - shift, right, bottom - shift


def _sized_text(imgui, draw_list, x: float, y: float, ink: int, text: str, px: float, halo=None):
    """Draw ``text`` at ``px`` pixels, with an optional outline in the opposing ink.

    imgui-bundle's ``ImDrawList.add_text`` exposes the ``(font, font_size, ...)`` overload
    that pyimgui did not, and the backend re-rasterises the atlas at the requested size --
    so this is genuinely sized text, not the glyph-quad rescaling
    ``renderers/axis.py::draw_text_scaled`` had to resort to (which blurs, because it
    stretches 13 px bitmaps). The helpers over there still carry docstrings saying no such
    overload exists; those predate this repo's move off pyimgui.

    The halo is four offset copies underneath -- imgui has no text-outline primitive --
    and is the live counterpart of the ``withStroke`` path effect the headless export
    uses. Only an inset bar asks for one: a bar outside the panel has empty gutter behind
    its numbers and needs no help.
    """
    font = imgui.get_font()
    if halo is not None:
        for dx, dy in ((-_INSET_TEXT_HALO_PX, 0.0), (_INSET_TEXT_HALO_PX, 0.0),
                       (0.0, -_INSET_TEXT_HALO_PX), (0.0, _INSET_TEXT_HALO_PX)):
            draw_list.add_text(font, px, (x + dx, y + dy), halo, text)
    draw_list.add_text(font, px, (x, y), ink, text)


def _measure(imgui, text: str, px: float):
    """``(width, height)`` of ``text`` at ``px``, for the same layout maths as the drawing.

    ``imgui.calc_text_size`` measures at the *current* font size, so placing sized text by
    it would centre and right-align every label against the wrong extent.
    """
    w, h = imgui.calc_text_size(text)
    base = imgui.get_font_size() or 1.0
    scale = px / base
    return w * scale, h * scale


def _draw_gradient(imgui, draw_list, left, top, right, bottom, stops, orientation: str) -> None:
    """Chain ``add_rect_filled_multi_color`` slices -- the same 2-stop primitive
    ``renderers/legend.py:_draw_swatch`` uses for its own gradient, just chained across
    more stops for a smooth sweep."""
    n = len(stops) - 1
    if n <= 0:
        return
    if orientation == "vertical":
        # fraction 0 (vmin) at the bottom, fraction 1 (vmax) at the top -- matplotlib's
        # own convention for a vertical colorbar.
        for i in range(n):
            f0, f1 = i / n, (i + 1) / n
            y_lo = bottom - f0 * (bottom - top)  # screen y of the lower-value edge
            y_hi = bottom - f1 * (bottom - top)  # screen y of the higher-value edge (smaller, higher up)
            c_lo = imgui.get_color_u32(tuple(float(c) for c in stops[i]))
            c_hi = imgui.get_color_u32(tuple(float(c) for c in stops[i + 1]))
            draw_list.add_rect_filled_multi_color((left, y_hi), (right, y_lo), c_hi, c_hi, c_lo, c_lo)
    else:
        # fraction 0 (vmin) at the left, fraction 1 (vmax) at the right.
        for i in range(n):
            f0, f1 = i / n, (i + 1) / n
            x_lo = left + f0 * (right - left)
            x_hi = left + f1 * (right - left)
            c_lo = imgui.get_color_u32(tuple(float(c) for c in stops[i]))
            c_hi = imgui.get_color_u32(tuple(float(c) for c in stops[i + 1]))
            draw_list.add_rect_filled_multi_color((x_lo, top), (x_hi, bottom), c_lo, c_hi, c_hi, c_lo)


def _resolve_ticks(cb: "ColorbarSpec") -> List[float]:
    """Tick data-values, honouring an explicit ``cb.ticks`` or auto-locating from ``norm``.

    Real ``matplotlib.ticker`` locators, not a reimplementation -- consistent with how
    ``norm`` itself is real matplotlib rather than a hand-rolled scale, and how
    ``managers/axis.py`` already borrows verbatim scale math elsewhere.
    """
    if cb.ticks is not None:
        return [float(v) for v in cb.ticks]

    from matplotlib import ticker

    vmin, vmax = float(cb.norm.vmin), float(cb.norm.vmax)
    cls_name = type(cb.norm).__name__
    if cls_name == "LogNorm":
        locator = ticker.LogLocator()
    elif cls_name == "SymLogNorm":
        # Base is not read off `norm` (matplotlib does not expose it uniformly across
        # versions) -- 10 is SymLogNorm's own default and the only base this codebase's
        # own symlog axis scale (`utils/scale.py`) uses, so it is not a guess.
        locator = ticker.SymmetricalLogLocator(linthresh=getattr(cb.norm, "linthresh", 1.0), base=10)
    else:
        locator = ticker.MaxNLocator(nbins=6)
    try:
        raw = locator.tick_values(vmin, vmax)
    except Exception:
        raw = np.linspace(vmin, vmax, 5)

    eps = 1e-9 * max(abs(vmax - vmin), 1.0)
    return [float(v) for v in raw if vmin - eps <= v <= vmax + eps]


def _format_tick(value: float, fmt: Any) -> str:
    """A tick's label text. Plain digits only -- the draw list has no mathtext/LaTeX, so
    a log tick reads as ``1e+03``, not a superscript ``10³``."""
    if fmt is not None:
        try:
            if callable(fmt):
                return str(fmt(value))
            fmt_str = str(fmt)
            return fmt_str % value if "%" in fmt_str else fmt_str.format(value)
        except Exception:
            pass  # Fall through to the auto-format below rather than drawing nothing.
    if value == 0:
        return "0"
    av = abs(value)
    if av >= 1e4 or av < 1e-3:
        return f"{value:.2e}"
    if float(value).is_integer():
        return f"{int(round(value))}"
    return f"{value:.3g}"


def _draw_tick(
    imgui,
    draw_list,
    left: float,
    top: float,
    right: float,
    bottom: float,
    frac: float,
    orientation: str,
    location: str,
    label: str,
    ink: int,
    halo=None,
) -> float:
    """Draw one tick's stroke + label on the bar's outward edge; returns the label's width."""
    text_w, text_h = _measure(imgui, label, _TICK_TEXT_PX)
    if orientation == "vertical":
        y = bottom - frac * (bottom - top)
        if location == "left":
            x_edge = left
            stroke_end = x_edge - _TICK_LEN_PX
            draw_list.add_line((x_edge, y), (stroke_end, y), ink, 1.0)
            _sized_text(
                imgui, draw_list, stroke_end - _TICK_LABEL_GAP_PX - text_w, y - text_h * 0.5,
                ink, label, _TICK_TEXT_PX, halo,
            )
        else:  # "right" (default)
            x_edge = right
            stroke_end = x_edge + _TICK_LEN_PX
            draw_list.add_line((x_edge, y), (stroke_end, y), ink, 1.0)
            _sized_text(
                imgui, draw_list, stroke_end + _TICK_LABEL_GAP_PX, y - text_h * 0.5,
                ink, label, _TICK_TEXT_PX, halo,
            )
    else:
        x = left + frac * (right - left)
        if location == "top":
            y_edge = top
            stroke_end = y_edge - _TICK_LEN_PX
            draw_list.add_line((x, y_edge), (x, stroke_end), ink, 1.0)
            _sized_text(
                imgui, draw_list, x - text_w * 0.5, stroke_end - _TICK_LABEL_GAP_PX - text_h,
                ink, label, _TICK_TEXT_PX, halo,
            )
        else:  # "bottom" (default)
            y_edge = bottom
            stroke_end = y_edge + _TICK_LEN_PX
            draw_list.add_line((x, y_edge), (x, stroke_end), ink, 1.0)
            _sized_text(
                imgui, draw_list, x - text_w * 0.5, stroke_end + _TICK_LABEL_GAP_PX,
                ink, label, _TICK_TEXT_PX, halo,
            )
    return text_w


def _draw_axis_label(
    imgui,
    draw_list,
    left: float,
    top: float,
    right: float,
    bottom: float,
    orientation: str,
    location: str,
    text: str,
    ink: int,
    widest_tick_w: float,
    halo=None,
) -> None:
    """The colorbar's own axis label, beyond the tick labels on the outward edge.

    A vertical bar's label is rotated to read bottom-to-top, exactly as the y-axis label
    does, through the same ``renderers/axis.py`` helper. That helper rewrites the glyph
    quads it just emitted, which is incompatible with both sizing them (it would fight
    :func:`_sized_text`'s own rasterisation) and stacking halo copies -- so a rotated
    label stays at the default size in plain ink, while a horizontal one gets both.
    """
    from .axis import _draw_text_rotated

    if orientation == "vertical":
        _, h = imgui.calc_text_size(text)
        mid_y = 0.5 * (top + bottom)
        if location == "left":
            center_x = max(left - _TICK_LABEL_GAP_PX - widest_tick_w - _AXIS_LABEL_GAP_PX - h * 0.5, h * 0.5 + 1.0)
        else:
            center_x = right + _TICK_LABEL_GAP_PX + widest_tick_w + _AXIS_LABEL_GAP_PX + h * 0.5
        anchor = (left, max(top - h - 2.0, 1.0))
        _draw_text_rotated(imgui, draw_list, text, ink, anchor, (center_x, mid_y))
    else:
        w, h = _measure(imgui, text, _LABEL_TEXT_PX)
        _, tick_h = _measure(imgui, "0", _TICK_TEXT_PX)
        mid_x = 0.5 * (left + right)
        tick_extent = _TICK_LABEL_GAP_PX + tick_h + _AXIS_LABEL_GAP_PX
        y = (bottom + tick_extent) if location == "bottom" else (top - tick_extent - h)
        _sized_text(imgui, draw_list, mid_x - w * 0.5, y, ink, text, _LABEL_TEXT_PX, halo)
