from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
from OpenGL.GL import *

from ..options import STOCK_WINDOW_TITLES, resolve_axis_margins
from ..utils.gl_utils import link_program
from ..utils.shaders import STRIP_FS, STRIP_VS

if TYPE_CHECKING:
    from ..core.context import RenderContext
    from ..managers.axis import AxisManager
    from ..options import EngineOptions


#: The ``EngineOptions.axis_grid_color`` default. While the option still holds this exact
#: value the grid keeps its historical automatic behaviour: dark ink on a light background,
#: light ink on a dark one. Any other value is treated as a deliberate user override and is
#: used verbatim. See :meth:`AxisRenderer._grid_color`.
AUTO_GRID_COLOR = (0.2, 0.2, 0.2)

#: Minor grid lines are drawn at this fraction of ``axis_grid_alpha``. Minor ticks exist to
#: subdivide, not to compete with the majors; at parity the grid reads as a flat mesh and
#: the major spacing -- the thing the tick labels are annotating -- becomes unreadable.
MINOR_GRID_ALPHA_SCALE = 0.45

#: Minor tick marks as a fraction of ``axis_tick_len_px``.
MINOR_TICK_LEN_SCALE = 0.55

#: Gap in pixels between a tick label and the spine it is aligned against.
_TICK_LABEL_GAP = 8.0

#: The point sizes ``fontsize=`` is measured against, so a caller's number carries the meaning it
#: has in matplotlib rather than a pixel count that happens to match this font atlas. matplotlib
#: resolves an unset title to ``axes.titlesize='large'`` == 1.2 x ``font.size``(10.0) and an unset
#: axis label to ``axes.labelsize='medium'`` == ``font.size``. Passing 14 must come out ~17%
#: larger on a title, and 40% larger on a label, for the same reason it does there.
_MPL_DEFAULT_TITLE_PT = 12.0
_MPL_DEFAULT_LABEL_PT = 10.0


def _draw_text_rotated(
    imgui,
    draw_list,
    text: str,
    color: int,
    anchor: Tuple[float, float],
    center: Tuple[float, float],
    scale: float = 1.0,
) -> bool:
    """Draw `text` rotated 90 degrees counter-clockwise at `scale`, centred on `center`.

    Returns True if the text was rotated. False means it was left horizontal at `anchor`,
    which the caller must therefore choose to be a position it is happy to live with.

    **Why this way.** imgui has no rotated-text call, and rotated glyph quads cannot be
    built by hand either: the per-glyph atlas UVs are not reachable from Python. What is
    reachable is the vertex buffer ``add_text`` just wrote (``draw_list.vtx_buffer``, a
    live sequence of ``ImDrawVert``), so the glyphs are emitted horizontally and then
    transformed in place -- the standard Dear ImGui rotation idiom.

    The alternatives were both rejected against the "do not ship something ugly" bar:
    per-glyph vertical stacking (letters stacked like a totem, with no kerning and no
    descender handling) and a horizontal label parked above the axis. The latter survives
    as the fallback below, but a rotated y-label is what every plotting tool a scientist
    has used puts there, so it is what we draw when we can.

    **Why the layout is checked, not trusted.** The vertex range just emitted is
    bounds-checked against what ``calc_text_size`` promised before a single vertex is
    rewritten. If the check fails -- glyphs CPU-culled against the clip rect because the
    text is wider than the viewport, say -- nothing is touched and the text simply stays
    where it was drawn. The fallback is the unmodified draw, which is why this cannot
    corrupt the frame it is unsure about.
    """
    ax, ay = anchor
    width, height = imgui.calc_text_size(text)
    if width <= 0.0 or height <= 0.0:
        return True  # Nothing to place (empty string, or all-whitespace).

    first = len(draw_list.vtx_buffer)
    draw_list.add_text((ax, ay), color, text)
    count = len(draw_list.vtx_buffer) - first

    # Glyph quads only. A non-multiple of 4 means something other than text landed in the
    # range and the arithmetic below would not be addressing glyph corners.
    if count <= 0 or count % 4 != 0:
        return count <= 0  # No verts at all == nothing drawn == nothing to fall back to.

    try:
        xs = [draw_list.vtx_buffer[i].pos.x for i in range(first, first + count)]
        ys = [draw_list.vtx_buffer[i].pos.y for i in range(first, first + count)]
    except (ValueError, TypeError, IndexError):  # pragma: no cover - defensive
        return False

    # The proof: every glyph corner must sit inside the box `add_text` was asked to fill
    # (plus a pixel of slack for glyph bearing), and the run must actually span most of it.
    # Misread memory fails this by many orders of magnitude, not by a rounding error.
    slack = 2.0
    inside = (
        min(xs) >= ax - slack
        and max(xs) <= ax + width + slack
        and min(ys) >= ay - slack
        and max(ys) <= ay + height + slack
    )
    spans = (max(xs) - min(xs)) >= 0.3 * width and (max(ys) - min(ys)) >= 0.3 * height
    if not (inside and spans):
        return False

    # Rotate about the emitted text's own centre, then land that centre on `center`.
    # Screen y grows downward, so (dx, dy) -> (dy, -dx) turns the reading direction from
    # left-to-right into bottom-to-top: a y-label reads upward, as it does in matplotlib.
    # `scale` rides along in the same transform rather than in a pass of its own: both are
    # about the emitted text's centre, so scaling the offset before rotating it *is* the
    # composition, and the glyphs are only walked once.
    ox, oy = ax + width * 0.5, ay + height * 0.5
    cx, cy = center
    for i in range(first, first + count):
        v = draw_list.vtx_buffer[i]
        dx, dy = (v.pos.x - ox) * scale, (v.pos.y - oy) * scale
        v.pos.x = cx + dy
        v.pos.y = cy - dx
    return True


def draw_text_scaled(
    imgui,
    draw_list,
    text: str,
    color: int,
    anchor: Tuple[float, float],
    scale: float,
) -> bool:
    """Draw `text` at `scale` about its own centre, anchored at `anchor`. True if scaled.

    pyimgui 2.0 exposes no way to draw text at a size other than the current font's:
    ``ImDrawList::AddText``'s font/size overload is unwrapped, and there is no
    ``imgui.get_font()`` to feed it even if it were. What *is* reachable is the vertex
    buffer ``add_text`` just wrote, so this uses the same idiom (and the same struct, and
    the same "verify before you poke" discipline) as :func:`_draw_text_rotated` above --
    emit horizontally, then transform the glyph quads in place.

    Returns False when the transform was not applied, in which case the text is still
    drawn, just at the default size. That is the whole failure mode: a slightly small
    label, never a corrupted frame and never an exception. ``scale`` at or below 1 is
    treated as "nothing to do" and returns True without touching memory.
    """
    if scale <= 1.0 + 1e-6:
        draw_list.add_text((anchor[0], anchor[1]), color, text)
        return True

    ax, ay = anchor
    width, height = imgui.calc_text_size(text)
    if width <= 0.0 or height <= 0.0:
        return True  # Nothing to place (empty string, or all-whitespace).

    first = len(draw_list.vtx_buffer)
    draw_list.add_text((ax, ay), color, text)
    count = len(draw_list.vtx_buffer) - first

    # Glyph quads only. A non-multiple of 4 means something other than text landed in the
    # range and the arithmetic below would not be addressing glyph corners.
    if count <= 0 or count % 4 != 0:
        return count <= 0

    try:
        xs = [draw_list.vtx_buffer[i].pos.x for i in range(first, first + count)]
        ys = [draw_list.vtx_buffer[i].pos.y for i in range(first, first + count)]
    except (ValueError, TypeError, IndexError):  # pragma: no cover - defensive
        return False

    # The same proof as the rotated path: every glyph corner must sit inside the box
    # `add_text` was asked to fill. Misread memory fails this by orders of magnitude.
    slack = 2.0
    inside = (
        min(xs) >= ax - slack
        and max(xs) <= ax + width + slack
        and min(ys) >= ay - slack
        and max(ys) <= ay + height + slack
    )
    if not inside:
        return False

    ox, oy = ax + width * 0.5, ay + height * 0.5
    for i in range(first, first + count):
        v = draw_list.vtx_buffer[i]
        v.pos.x = ox + (v.pos.x - ox) * scale
        v.pos.y = oy + (v.pos.y - oy) * scale
    return True


class AxisRenderer:
    """
    Specialized renderer for the plot framework: grid, spines, and ticks.
    """

    def __init__(self, options: EngineOptions) -> None:
        self.options = options
        self.prog = 0
        self.u_mvp = -1
        self.u_color = -1
        self.u_alpha = -1

        # Temp buffer for line drawing
        self.vbo = 0
        self.vao = 0

    def initialize(self) -> None:
        self.prog = link_program(STRIP_VS, STRIP_FS)
        self.u_mvp = glGetUniformLocation(self.prog, "u_mvp")
        self.u_color = glGetUniformLocation(self.prog, "u_color")
        self.u_alpha = glGetUniformLocation(self.prog, "u_alpha")

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

    def _grid_color(self, is_light: bool) -> Tuple[float, float, float]:
        """Resolve the grid line colour, honouring ``options.axis_grid_color``.

        The option defaults to :data:`AUTO_GRID_COLOR`; while it is untouched the grid
        picks its colour from the background luminance exactly as it always has, so the
        default look is unchanged. Setting the option to anything else overrides that.
        """
        color = getattr(self.options, "axis_grid_color", None)
        if color is not None:
            try:
                rgb = (float(color[0]), float(color[1]), float(color[2]))
            except (TypeError, ValueError, IndexError):
                rgb = None
            if rgb is not None and rgb != AUTO_GRID_COLOR:
                return rgb
        return (0.2, 0.2, 0.2) if is_light else (0.8, 0.8, 0.8)

    def _font_scale(self, option: str, default_pt: float) -> float:
        """``fontsize=`` on an annotation, as a multiplier of the atlas font. 1.0 when unset.

        ``default_pt`` is the matplotlib point size that means "unscaled" for this
        annotation, so a caller's number keeps the meaning it has there rather than being
        read as a pixel count that happens to match this font atlas.

        Clamped for the same reason ``legend_font_scale`` is: ``_add_scaled_text`` moves
        glyph quads that are already in the draw list, so an absurd scale does not fail --
        it silently paints over the whole frame.
        """
        pt = getattr(self.options, option, None)
        if pt is None:
            return 1.0
        try:
            scale = float(pt) / default_pt
        except (TypeError, ValueError):
            return 1.0
        return min(max(scale, 0.5), 4.0)

    def _ink(self, option: str, fallback: int) -> int:
        """``color=`` on an annotation as an ImGui u32, or ``fallback`` when unset/bad.

        ``fallback`` is the luminance-derived ink the annotations use by default, so an
        unset colour keeps the automatic light-on-dark behaviour untouched -- and so does
        an unparseable one, which must not take the frame down mid-draw.
        """
        color = getattr(self.options, option, None)
        if color is None:
            return fallback
        try:
            from imgui_bundle import imgui

            from ..pyplot import _normalize_rgba

            r, g, b, a = _normalize_rgba(color)
            return imgui.get_color_u32((float(r), float(g), float(b), float(a)))
        except (ImportError, ValueError, TypeError):
            return fallback

    def draw(self, axis: AxisManager, ctx: RenderContext) -> None:
        # Check overall visibility
        if not any(
            [
                self.options.axis_show_grid,
                self.options.axis_show_frame,
                self.options.axis_show_labels,
            ]
        ):
            return

        glUseProgram(self.prog)
        glUniformMatrix4fv(self.u_mvp, 1, GL_TRUE, ctx.mvp)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        win = ctx.window_world

        # Get background color to calculate contrast
        if ctx.is_density:
            from ..utils.shaders import get_colormap_min_color

            scheme_idx = getattr(self.options, "density_scheme_index", 0)
            invert = getattr(self.options, "density_invert", False)
            ltc = getattr(self.options, "density_light_to_color", True)
            bg_color = get_colormap_min_color(scheme_idx, invert, ltc)
        else:
            v = self.options.visual.gradient_background
            if v.enabled:
                bg_color = (
                    0.5 * (v.top_color[0] + v.bottom_color[0]),
                    0.5 * (v.top_color[1] + v.bottom_color[1]),
                    0.5 * (v.top_color[2] + v.bottom_color[2]),
                )
            else:
                bg_color = self.options.visual.background_color

        lum = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
        is_light = lum > 0.5

        # World units per screen pixel, for anything sized in pixels (tick marks). The
        # frame occupies the viewport minus the gutters, so that -- not the full width --
        # is what the world window is stretched across.
        margin_l, margin_r, margin_b, margin_t = resolve_axis_margins(self.options)
        plot_w_px = max(ctx.width_px - margin_l - margin_r, 1.0)
        plot_h_px = max(ctx.height_px - margin_t - margin_b, 1.0)
        world_per_px_x = (win[1] - win[0]) / plot_w_px
        world_per_px_y = (win[3] - win[2]) / plot_h_px

        # Opacity for the whole axis pass, set BEFORE the first conditional draw.
        #
        # This used to live inside the ``axis_show_grid`` branch below, which made the grid
        # switch silently govern the spines and the tick marks as well: a GL uniform starts
        # at 0, ``STRIP_VS`` does ``v_col.a *= u_alpha``, so with the grid off the first
        # axis frame of the process drew the frame and every tick at alpha 0 — invisible.
        # It stayed invisible until something else happened to set the uniform.
        #
        # It was masked in normal use because the workstation opens on a preset whose grid
        # is on, so the uniform was always already 1.0 by the time a grid-off style was
        # applied. It bit exactly the figures that turn the grid off from the very first
        # frame, which is what a journal or high-density style does.
        glUniform1f(self.u_alpha, 1.0)

        # 1. Draw Grid
        if self.options.axis_show_grid:
            c = self._grid_color(is_light)

            # Minor grid first, so the majors overdraw it rather than the reverse.
            if getattr(self.options, "axis_minor_ticks", False):
                minor_lines = []
                for x in axis.ticks_x.minor:
                    minor_lines.extend([(x, win[2]), (x, win[3])])
                for y in axis.ticks_y.minor:
                    minor_lines.extend([(win[0], y), (win[1], y)])
                if minor_lines:
                    glUniform4f(
                        self.u_color,
                        c[0],
                        c[1],
                        c[2],
                        self.options.axis_grid_alpha * MINOR_GRID_ALPHA_SCALE,
                    )
                    data = np.array(minor_lines, dtype=np.float32)
                    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STREAM_DRAW)
                    glDrawArrays(GL_LINES, 0, len(minor_lines))

            glUniform4f(self.u_color, c[0], c[1], c[2], self.options.axis_grid_alpha)

            grid_lines = []
            # Vertical lines (X-ticks)
            for x in axis.ticks_x.major:
                grid_lines.extend([(x, win[2]), (x, win[3])])
            # Horizontal lines (Y-ticks)
            for y in axis.ticks_y.major:
                grid_lines.extend([(win[0], y), (win[1], y)])

            if grid_lines:
                data = np.array(grid_lines, dtype=np.float32)
                glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STREAM_DRAW)
                glDrawArrays(GL_LINES, 0, len(grid_lines))

        # 1b. Tick marks -- short strokes inward from the bottom and left spines. Sized in
        # pixels, so they stay the same length at every zoom.
        if getattr(self.options, "axis_show_ticks", False):
            self._draw_tick_marks(axis, win, world_per_px_x, world_per_px_y, is_light)

        # 2. Draw Spines (Frame)
        if self.options.axis_show_frame:
            c_spine = (0.15, 0.15, 0.15, 1.0) if is_light else (0.85, 0.85, 0.85, 1.0)
            glUniform4f(self.u_color, c_spine[0], c_spine[1], c_spine[2], c_spine[3])
            frame = [
                (win[0], win[2]),
                (win[1], win[2]),
                (win[1], win[2]),
                (win[1], win[3]),
                (win[1], win[3]),
                (win[0], win[3]),
                (win[0], win[3]),
                (win[0], win[2]),
            ]
            data_frame = np.array(frame, dtype=np.float32)
            glBufferData(GL_ARRAY_BUFFER, data_frame.nbytes, data_frame, GL_STREAM_DRAW)
            glDrawArrays(GL_LINES, 0, 8)

        glBindVertexArray(0)
        glUseProgram(0)

    def _draw_tick_marks(
        self,
        axis: AxisManager,
        win: Tuple[float, float, float, float],
        world_per_px_x: float,
        world_per_px_y: float,
        is_light: bool,
    ) -> None:
        """Stroke major and minor tick marks inward from the bottom and left spines.

        Assumes the caller has the axis program bound and the VAO/VBO active.
        """
        length_px = max(float(getattr(self.options, "axis_tick_len_px", 5.0)), 0.0)
        if length_px <= 0.0:
            return

        c = (0.15, 0.15, 0.15, 1.0) if is_light else (0.85, 0.85, 0.85, 1.0)
        glUniform4f(self.u_color, c[0], c[1], c[2], c[3])

        for scale, ticks_x, ticks_y in (
            (1.0, axis.ticks_x.major, axis.ticks_y.major),
            (MINOR_TICK_LEN_SCALE, axis.ticks_x.minor, axis.ticks_y.minor),
        ):
            len_x = length_px * scale * world_per_px_x
            len_y = length_px * scale * world_per_px_y
            marks = []
            for x in ticks_x:
                marks.extend([(x, win[2]), (x, win[2] + len_y)])
            for y in ticks_y:
                marks.extend([(win[0], y), (win[0] + len_x, y)])
            if marks:
                data = np.array(marks, dtype=np.float32)
                glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STREAM_DRAW)
                glDrawArrays(GL_LINES, 0, len(marks))

    def _resolve_annotation(self, axis: AxisManager, option_name: str, attr_name: str) -> str:
        """Resolve an axis label from the two channels that can set it.

        The GUI writes `options.<option_name>`; the public pyplot API writes
        `plot.<attr_name>` (`gplt.xlabel()` -> `plot.xlabel`). The attribute channel was,
        until this renderer read it, consumed by nothing but the matplotlib savefig
        fallback -- so `gplt.xlabel("Time (s)")` set a string that never reached a pixel.
        The option wins when both are set: it is the one the user can see themselves typing.
        """
        value = str(getattr(self.options, option_name, "") or "")
        if value:
            return value
        plot = getattr(axis, "plot", None)
        return str(getattr(plot, attr_name, "") or "")

    def _resolve_title(self, axis: AxisManager) -> str:
        """Resolve the on-plot title, per panel, refusing to promote a stock caption.

        `axis_title` (the GUI channel) is unambiguous and wins. Otherwise the *active*
        panel's own title is used -- this runs inside the per-panel overlay pass, which
        swaps `active_panel_index` around it, so each panel resolves its own.

        The window caption is the last resort, and only on a single-panel figure. It is
        what makes `gplt.title()` draw something there, but `set_title` writes it *and*
        the active panel's title, so on a split figure the caption holds whichever panel
        was titled last -- falling back to it per panel stamped that one title across all
        of them. And since `GPULinePlot` defaults its caption to "GLPlot", an
        unconditional fallback would stamp that literal string on every untitled plot;
        only a caption the caller actually chose is ever promoted.
        """
        value = str(getattr(self.options, "axis_title", "") or "")
        if value:
            return value
        plot = getattr(axis, "plot", None)
        panel = getattr(plot, "active_panel", None)
        candidate = str(getattr(panel, "title", "") or "")
        if not candidate:
            if len(getattr(plot, "panels", None) or []) > 1:
                return ""
            candidate = str(getattr(plot, "title", "") or "")
        # The stock-caption test applies to whichever source won, not just the caption:
        # `set_title` copies its argument onto the active panel *as well as* the window
        # caption, so checking only the latter let `set_title("GLPlot")` reach the plot
        # through the panel and caption it with the product name anyway.
        return "" if candidate in STOCK_WINDOW_TITLES else candidate

    def _draw_labels(self, axis: AxisManager, ctx: RenderContext) -> None:
        """Draw numeric labels along the axes."""
        try:
            from imgui_bundle import imgui
        except ImportError:
            return

        draw_list = imgui.get_background_draw_list()

        # Get background color to calculate contrast
        if ctx.is_density:
            from ..utils.shaders import get_colormap_min_color

            scheme_idx = getattr(self.options, "density_scheme_index", 0)
            invert = getattr(self.options, "density_invert", False)
            ltc = getattr(self.options, "density_light_to_color", True)
            bg_color = get_colormap_min_color(scheme_idx, invert, ltc)
        else:
            v = self.options.visual.gradient_background
            if v.enabled:
                bg_color = (
                    0.5 * (v.top_color[0] + v.bottom_color[0]),
                    0.5 * (v.top_color[1] + v.bottom_color[1]),
                    0.5 * (v.top_color[2] + v.bottom_color[2]),
                )
            else:
                bg_color = self.options.visual.background_color

        lum = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
        is_light = lum > 0.5

        if is_light:
            color = imgui.get_color_u32((0.15, 0.15, 0.15, 1.0))
        else:
            color = imgui.get_color_u32((0.85, 0.85, 0.85, 1.0))

        tick_color = self._ink("axis_tick_color", color)
        tick_scale = self._font_scale("axis_tick_fontsize", _MPL_DEFAULT_LABEL_PT)

        win = ctx.window_world

        # Helper to project world to screen. px_offset shifts panel-local pixels into
        # window space, so a split panel's tick labels land over the panel, not the corner.
        off_x, off_y = ctx.px_offset

        def project(wx, wy):
            pos_world = np.array([wx, wy, 0.0, 1.0], dtype=np.float32)
            pos_ndc = ctx.mvp @ pos_world
            if pos_ndc[3] != 0:
                pos_ndc /= pos_ndc[3]
            screen_x = (pos_ndc[0] + 1.0) * 0.5 * ctx.width_px + off_x
            screen_y = (1.0 - pos_ndc[1]) * 0.5 * ctx.height_px + off_y
            return screen_x, screen_y

        # X-Axis Labels (along bottom).
        #
        # Centred on the tick with the measured width. The old fixed `sx - 15` offset
        # assumed every label was ~30px wide: "0" sat 11px left of its own tick and
        # "-100000" hung 20px right of it. It is worth fixing now rather than later
        # because `axis_tick_format` can put much wider strings here than the old
        # fixed-point-only formatter ever could -- "5.0e-10" is 49px.
        for val, label in zip(axis.ticks_x.major, axis.ticks_x.labels):
            sx, sy = project(val, win[2])
            # Offset labels slightly below the spine
            w = imgui.calc_text_size(label)[0] * tick_scale
            draw_text_scaled(
                imgui, draw_list, label, tick_color, (sx - w * 0.5, sy + 5), tick_scale
            )

        # Y-Axis Labels (along left).
        #
        # Right-aligned against the spine using the measured text width, NOT a fixed origin.
        # A fixed origin put every label at the same x no matter how wide it was, which meant
        # long values ("-100000") grew rightward into the frame while short ones ("0") left a
        # ragged gap -- and it pinned the whole column to `spine - 45` regardless of how much
        # gutter `axis_margin_l` had actually reserved, so a GUI that widened the margin to
        # clear its own chrome got no benefit. Measuring makes the labels track the gutter.
        widest_tick = 0.0
        for val, label in zip(axis.ticks_y.major, axis.ticks_y.labels):
            sx, sy = project(win[0], val)
            text_w = imgui.calc_text_size(label)[0] * tick_scale
            widest_tick = max(widest_tick, text_w)
            draw_text_scaled(
                imgui,
                draw_list,
                label,
                tick_color,
                (sx - _TICK_LABEL_GAP - text_w, sy - 7),
                tick_scale,
            )

        self._draw_annotations(imgui, draw_list, axis, ctx, project, color, widest_tick)

    def _draw_annotations(
        self,
        imgui,
        draw_list,
        axis: AxisManager,
        ctx: RenderContext,
        project,
        color: int,
        widest_tick: float,
    ) -> None:
        """Draw the x-label, y-label and plot title into the axis gutters.

        Placement is derived from the *projected* spine positions rather than from the
        margin options directly, so it cannot disagree with the matrix that actually put
        the frame there -- and it follows `axis_margin_*` for free when a GUI widens a
        gutter to clear its own chrome.
        """
        xlabel = self._resolve_annotation(axis, "axis_xlabel", "xlabel")
        ylabel = self._resolve_annotation(axis, "axis_ylabel", "ylabel")
        title = self._resolve_title(axis)
        if not (xlabel or ylabel or title):
            return

        win = ctx.window_world
        left_px, bottom_px = project(win[0], win[2])
        right_px, top_px = project(win[1], win[3])
        mid_x = 0.5 * (left_px + right_px)
        mid_y = 0.5 * (top_px + bottom_px)
        # The panel's window-space top and bottom edges. `project` is offset-aware (spine
        # positions already include px_offset), but the gutter references below -- the panel
        # top and its bottom edge -- are not, so add the offset to them explicitly. (0 for a
        # full-window panel, so single-viewport placement is unchanged.)
        off_y = float(ctx.px_offset[1])
        panel_bottom_px = off_y + ctx.height_px

        # Every sized run below measures its extent *scaled*: `_add_scaled_text` grows the
        # glyphs about the anchor it is handed, so centring on the unscaled width would let
        # the text drift out of its gutter as it grew.
        from .legend import _add_scaled_text

        if xlabel:
            # Bottom gutter, under the tick labels, pinned to the viewport's bottom edge so
            # a wider `axis_margin_b` gives the label room instead of just moving it.
            scale = self._font_scale("axis_xlabel_fontsize", _MPL_DEFAULT_LABEL_PT)
            w, h = imgui.calc_text_size(xlabel)
            w, h = w * scale, h * scale
            y = max(panel_bottom_px - h - 2.0, bottom_px + 2.0)
            _add_scaled_text(
                imgui,
                draw_list,
                mid_x - w * 0.5,
                y,
                self._ink("axis_xlabel_color", color),
                xlabel,
                scale,
            )

        if title:
            # Top gutter, centred between the panel's top edge and the top spine.
            scale = self._font_scale("axis_title_fontsize", _MPL_DEFAULT_TITLE_PT)
            w, h = imgui.calc_text_size(title)
            w, h = w * scale, h * scale
            y = max(0.5 * (off_y + top_px - h), off_y + 1.0)
            _add_scaled_text(
                imgui,
                draw_list,
                mid_x - w * 0.5,
                y,
                self._ink("axis_title_color", color),
                title,
                scale,
            )

        if ylabel:
            w, h = imgui.calc_text_size(ylabel)
            # Left of the widest tick label, centred on the frame. Clamped off the viewport
            # edge: if the gutter is too narrow the label overlaps the ticks rather than
            # vanishing off-screen, which is a visible cue to widen `axis_margin_l`.
            center_x = max(left_px - _TICK_LABEL_GAP - widest_tick - 2.0 - h * 0.5, h * 0.5 + 1.0)
            # The anchor doubles as the no-rotation fallback position, so it has to be a
            # placement worth keeping: horizontal, above the frame at the left spine --
            # where a number of plotting tools put the y-label anyway.
            anchor = (left_px, max(top_px - h - 2.0, 1.0))
            _draw_text_rotated(
                imgui,
                draw_list,
                ylabel,
                self._ink("axis_ylabel_color", color),
                anchor,
                (center_x, mid_y),
                self._font_scale("axis_ylabel_fontsize", _MPL_DEFAULT_LABEL_PT),
            )
