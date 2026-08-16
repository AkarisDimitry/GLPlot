from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from ..utils.scale import forward as _scale_forward
from ..utils.scale import inverse as _scale_inverse

if TYPE_CHECKING:
    from ..core.context import RenderContext
    from ..engine import GPULinePlot

#: Outside this decade window the automatic formatter switches to scientific notation.
#: Inside it, it reproduces the historical `%.Nf` behaviour exactly, so ordinary plots are
#: byte-for-byte unchanged and only the ranges that used to render as a column of `0`s
#: (or as unreadable 12-digit strings) move.
_SCI_LOW_EXP = -4
_SCI_HIGH_EXP = 6


@dataclass
class AxisTicks:
    major: np.ndarray = field(default_factory=lambda: np.array([]))
    labels: List[str] = field(default_factory=list)
    #: Positions between the majors. Empty unless `options.axis_minor_ticks`. Never
    #: labelled -- minor ticks carry spacing information, not values.
    minor: np.ndarray = field(default_factory=lambda: np.array([]))
    #: The spacing the majors were generated on. 0.0 for an empty axis. The renderer has no
    #: other way to recover it, and `major` may be too short to difference.
    step: float = 0.0


def _nice_step(raw_step: float) -> float:
    """Round `raw_step` up the 1-2-5 ladder to the next 'nice' number."""
    p = 10 ** np.floor(np.log10(raw_step))
    m = raw_step / p
    if m < 1.5:
        return float(1.0 * p)
    if m < 3.5:
        return float(2.0 * p)
    if m < 7.5:
        return float(5.0 * p)
    return float(10.0 * p)


def _decimals_for_step(step: float) -> int:
    """Decimal places needed to write `step` (and therefore every tick) exactly.

    The historical rule was `-floor(log10(step))` clamped to 0 for `step >= 1`, which is
    correct for every step off the 1-2-5 ladder -- those are integers once they reach 1 --
    and silently wrong for any other spacing. `axis_tick_step_*` lets the user pin an
    arbitrary one, and a hand-typed 2.5 printed its ticks as `0, 2, 5, 8`: three of the
    four labels wrong, on axes the user had just taken manual control of.
    """
    for places in range(13):
        scaled = step * (10**places)
        # Tolerance relative to `scaled`, never absolute: an absolute epsilon calls 1e-9 an
        # integer (it rounds to 0 and misses by only 1e-9) and prints a nanometre axis as a
        # column of `0`s -- the exact failure this formatter exists to remove.
        if abs(scaled - round(scaled)) <= 1e-9 * abs(scaled):
            return places
    return 12


def _auto_format(major: np.ndarray, step: float) -> List[str]:
    """Format ticks with a precision derived from the step.

    Fixed-point inside the decade window, scientific outside it. The old formatter was
    fixed-point unconditionally, which meant a 1e-9 span printed a column of identical
    `0`s -- every label carrying zero information -- and a 1e+9 span printed ten-digit
    integers that overran the gutter.
    """
    if step <= 0 or not np.isfinite(step):
        return [str(v) for v in major]

    finite = major[np.isfinite(major)]
    magnitude = float(np.max(np.abs(finite))) if finite.size else 0.0
    scale = max(magnitude, step)
    exponent = int(np.floor(np.log10(scale))) if scale > 0 else 0

    if exponent < _SCI_LOW_EXP or exponent >= _SCI_HIGH_EXP:
        # Significant digits needed to separate neighbouring ticks at this magnitude.
        digits = max(0, exponent - int(np.floor(np.log10(step))))
        fmt = f"{{:.{min(digits, 12)}e}}"
    else:
        fmt = f"{{:.{_decimals_for_step(step)}f}}"
    return [fmt.format(v) for v in major]


def _format_decade(decade: int) -> str:
    """Label for a log-axis major tick at ``10**decade``: a plain number inside the same
    decade window ``_auto_format`` uses, scientific outside it."""
    value = 10.0**decade
    if _SCI_LOW_EXP <= decade < _SCI_HIGH_EXP:
        return f"{value:g}"
    return f"1e{decade}"


def _apply_format(major: np.ndarray, step: float, spec: str) -> List[str]:
    """Format ticks with a user-supplied `spec`, falling back to automatic on error.

    Three forms are accepted because all three are things a scientist reasonably types into
    a text box: printf (`%.2f`), str.format (`{:.3g}`) and a bare format spec (`.3g`). A
    spec that raises must not break the frame -- the GUI edits this string live, so it is
    invalid on almost every keystroke of typing a valid one.
    """
    spec = spec.strip()
    if not spec:
        return _auto_format(major, step)
    try:
        if "%" in spec:
            return [spec % v for v in major]
        if "{" in spec:
            return [spec.format(v) for v in major]
        return [format(v, spec) for v in major]
    except (ValueError, TypeError, KeyError, IndexError):
        return _auto_format(major, step)


class AxisManager:
    """
    Logical manager for coordinate axes.
    Generates 'nice' ticks and labels based on the visible data range.
    """

    def __init__(self, plot: GPULinePlot) -> None:
        self.plot = plot
        self.options = plot.options
        self.ticks_x = AxisTicks()
        self.ticks_y = AxisTicks()

    def update(self, ctx: RenderContext) -> None:
        """Recalculate ticks for the current view."""
        options = self.options

        # Auto targets: ~one tick per 160px across, 120px down. `axis_tick_count_*`
        # overrides the density; `axis_tick_step_*` overrides the spacing outright.
        count_x = int(getattr(options, "axis_tick_count_x", 0) or 0)
        count_y = int(getattr(options, "axis_tick_count_y", 0) or 0)
        target_x = count_x if count_x > 0 else max(4, int(ctx.width_px / 160))
        target_y = count_y if count_y > 0 else max(4, int(ctx.height_px / 120))

        step_x = float(getattr(options, "axis_tick_step_x", 0.0) or 0.0)
        step_y = float(getattr(options, "axis_tick_step_y", 0.0) or 0.0)

        win = ctx.window_world
        self.ticks_x = self._resolve_ticks("x", win[0], win[1], target_x, step_x)
        self.ticks_y = self._resolve_ticks("y", win[2], win[3], target_y, step_y)

    def _resolve_ticks(
        self, axis: str, vmin: float, vmax: float, target_count: int, forced_step: float
    ) -> AxisTicks:
        """Explicit ticks if `gplt.xticks()`/`yticks()` set any, else generated ones.

        Explicit positions outrank the count/step knobs: a caller who named the positions
        does not want them re-derived from the viewport. `None` means "not set"; an empty
        tuple is the real instruction `xticks([])` sends, and must clear the axis rather
        than fall through to the generator -- hence the `is None` test.
        """
        values = getattr(self.options, f"axis_tick_values_{axis}", None)
        if values is None:
            scale_mode = getattr(self.options, f"axis_scale_{axis}", "linear")
            scale_params = getattr(self.options, f"axis_scale_params_{axis}", None)
            return self._generate_ticks(
                vmin, vmax, target_count, forced_step, scale_mode, scale_params
            )

        major = np.asarray(values, dtype=float)
        labels = getattr(self.options, f"axis_tick_labels_{axis}", None)
        # Only what is on-screen is drawn, but the *stored* positions stay whole (see
        # `options.axis_tick_values_x`), so the filter happens here and the mask has to
        # carry the labels with it or a pan would slide them onto the wrong ticks.
        visible = (major >= vmin) & (major <= vmax)
        shown = major[visible]
        if labels is None:
            # The formatter derives its precision from the spacing. Unsorted or duplicated
            # positions are legal input, so this takes the smallest *magnitude* gap and
            # tolerates it being zero rather than assuming a monotonic ladder.
            gaps = np.abs(np.diff(np.sort(major))) if major.size > 1 else np.array([])
            gaps = gaps[gaps > 0]
            step = float(gaps.min()) if gaps.size else 0.0
            text = _apply_format(
                shown, step, str(getattr(self.options, "axis_tick_format", "") or "")
            )
        else:
            text = [str(t) for t, keep in zip(labels, visible) if keep]
        # No minors: they subdivide a regular step, and explicit ticks need not have one.
        return AxisTicks(major=shown, labels=text, minor=np.array([]), step=0.0)

    def _generate_ticks(
        self,
        vmin: float,
        vmax: float,
        target_count: int,
        forced_step: float = 0.0,
        scale_mode: str = "linear",
        scale_params: Optional[dict] = None,
    ) -> AxisTicks:
        if vmin >= vmax:
            return AxisTicks()

        has_forced_step = forced_step > 0 and np.isfinite(forced_step)
        if scale_mode == "log" and not has_forced_step:
            # `vmin`/`vmax` are already log10-space (the camera's "world" on a log axis --
            # see `glplot.utils.scale`); a forced step, if the caller set one, still wins
            # and falls through to the linear ladder below, applied directly in log space.
            return self._generate_log_ticks(vmin, vmax, target_count)
        if scale_mode in ("symlog", "asinh") and not has_forced_step:
            return self._generate_symmetric_ticks(
                vmin, vmax, target_count, scale_mode, scale_params
            )
        if scale_mode == "logit" and not has_forced_step:
            return self._generate_logit_ticks(vmin, vmax, target_count)

        span = vmax - vmin
        if forced_step > 0 and np.isfinite(forced_step):
            step = float(forced_step)
            # A hand-typed step can be absurdly small for the current zoom. Ticks are one
            # draw call but the label loop is per-tick imgui work, so clamp rather than
            # stall the frame; the view simply keeps the finest spacing that stays sane.
            if span / step > 1000:
                step = span / 1000.0
        else:
            step = _nice_step(span / max(1, target_count))

        # Calculate start/end
        start = np.ceil(vmin / step) * step
        end = np.floor(vmax / step) * step

        if start > end:
            return AxisTicks()

        major = np.arange(start, end + step / 2, step)

        labels = _apply_format(
            major, step, str(getattr(self.options, "axis_tick_format", "") or "")
        )

        minor = self._generate_minor(vmin, vmax, start, step)

        return AxisTicks(major=major, labels=labels, minor=minor, step=step)

    def _generate_log_ticks(self, vmin: float, vmax: float, target_count: int) -> AxisTicks:
        """Ticks for a log10-scaled axis. ``vmin``/``vmax`` arrive already in log10 space.

        At least a decade visible: one major tick per decade (``1, 10, 100, ...``), plus
        the classic ``2..9 x 10^decade`` minors matplotlib's own ``LogLocator`` draws. Less
        than a decade visible (zoomed in far enough that decade ticks would be sparse or
        absent): fall back to the same 1-2-5 nice-number ladder a linear axis uses, computed
        in real (not log) space so the labels are round numbers, then placed at their log10
        position on screen.
        """
        decade_span = vmax - vmin
        if decade_span >= 1.0:
            start = np.ceil(vmin)
            end = np.floor(vmax)
            if start > end:
                # The window sits inside one decade despite spanning >= 1 log-unit at its
                # very edges -- one tick, at the nearest decade, beats none.
                major = np.array([round((vmin + vmax) / 2.0)])
            else:
                major = np.arange(start, end + 0.5, 1.0)
            labels = [_format_decade(int(round(d))) for d in major]
            minor = np.array([])
            if getattr(self.options, "axis_minor_ticks", False):
                decades = np.arange(np.floor(vmin), np.ceil(vmax) + 1.0)
                subs = np.arange(2, 10)
                candidates = np.log10(np.outer(10.0**decades, subs)).ravel()
                minor = np.sort(candidates[(candidates >= vmin) & (candidates <= vmax)])
            return AxisTicks(major=major, labels=labels, minor=minor, step=1.0)

        vmin_real, vmax_real = 10.0**vmin, 10.0**vmax
        step_real = _nice_step((vmax_real - vmin_real) / max(1, target_count))
        start_real = np.ceil(vmin_real / step_real) * step_real
        end_real = np.floor(vmax_real / step_real) * step_real
        if start_real > end_real:
            return AxisTicks()
        major_real = np.arange(start_real, end_real + step_real / 2, step_real)
        major_real = major_real[major_real > 0]
        if major_real.size == 0:
            return AxisTicks()
        labels = _apply_format(
            major_real, step_real, str(getattr(self.options, "axis_tick_format", "") or "")
        )
        return AxisTicks(major=np.log10(major_real), labels=labels, minor=np.array([]), step=0.0)

    def _generate_symmetric_ticks(
        self, vmin: float, vmax: float, target_count: int, mode: str, params: Optional[dict]
    ) -> AxisTicks:
        """Ticks for 'symlog'/'asinh': linear-ish near zero, log-ish outside, both signs.

        ``vmin``/``vmax`` arrive already transformed. Unlike `_generate_log_ticks`, this is
        not a reproduction of matplotlib's own locator (`SymmetricalLogLocator`/
        `AsinhLocator` are considerably more intricate than `LogLocator`) -- it is a
        simpler, still-correct one: decade candidates mirrored around zero (`..., -100,
        -10, -1, 0, 1, 10, 100, ...`), each kept only if it actually falls in the visible
        window, falling back to the usual real-space nice-ladder when that yields too few
        ticks to be useful (the window sits inside the linear region around zero).
        """
        vmin_real = float(_scale_inverse(np.array(vmin), mode, params))
        vmax_real = float(_scale_inverse(np.array(vmax), mode, params))
        lo_real, hi_real = min(vmin_real, vmax_real), max(vmin_real, vmax_real)

        candidates = set()
        if lo_real <= 0.0 <= hi_real:
            candidates.add(0.0)
        max_abs = max(abs(lo_real), abs(hi_real), 1e-300)
        top_decade = int(np.ceil(np.log10(max_abs))) + 1
        # A candidate exactly at the window edge (e.g. `yscale('symlog')` on data that
        # tops out at exactly 1000.0) can miss a plain `<=` by a couple ULPs once it has
        # round-tripped through forward()/inverse() -- relative tolerance, not exactness.
        tol = 1e-9 * max_abs
        for decade in range(-1, top_decade + 1):
            v = 10.0**decade
            if lo_real - tol <= v <= hi_real + tol:
                candidates.add(v)
            if lo_real - tol <= -v <= hi_real + tol:
                candidates.add(-v)
        major_real = np.array(sorted(candidates))

        tick_format = str(getattr(self.options, "axis_tick_format", "") or "")
        if major_real.size < 3:
            # The window sits inside (or barely past) the linear region: nice round
            # numbers in real space, the same ladder a linear axis uses.
            step_real = _nice_step((hi_real - lo_real) / max(1, target_count))
            start_real = np.ceil(lo_real / step_real) * step_real
            end_real = np.floor(hi_real / step_real) * step_real
            if start_real > end_real:
                return AxisTicks()
            major_real = np.arange(start_real, end_real + step_real / 2, step_real)
            labels = _apply_format(major_real, step_real, tick_format)
        else:
            gaps = np.abs(np.diff(major_real))
            gaps = gaps[gaps > 0]
            step_real = float(gaps.min()) if gaps.size else 1.0
            labels = _apply_format(major_real, step_real, tick_format)

        major = _scale_forward(major_real, mode, params)
        return AxisTicks(major=major, labels=labels, minor=np.array([]), step=0.0)

    def _generate_logit_ticks(self, vmin: float, vmax: float, target_count: int) -> AxisTicks:
        """Ticks for 'logit': ``vmin``/``vmax`` arrive already transformed.

        Candidates follow matplotlib's own `LogitLocator` scheme (`matplotlib/scale.py`'s
        ``ideal_ticks``): ``0.5`` at the centre, then ``p``/``1-p`` pairs at
        ``0.1, 0.01, 0.001, ...`` mirrored around it -- logit's domain is `(0, 1)`, not
        symmetric around zero, so this is its own generator rather than reusing
        `_generate_symmetric_ticks`. Not a reproduction of `LogitLocator`'s `nbins`-aware
        windowing (same documented simplification as the log/symlog/asinh generators
        above): every candidate in range is kept, then the real-space nice-ladder covers
        a span too narrow for any candidate to land in.
        """
        vmin_real = float(_scale_inverse(np.array(vmin), "logit"))
        vmax_real = float(_scale_inverse(np.array(vmax), "logit"))
        lo_real, hi_real = min(vmin_real, vmax_real), max(vmin_real, vmax_real)

        candidates = set()
        if lo_real <= 0.5 <= hi_real:
            candidates.add(0.5)
        # Relative tolerance, not exactness -- a candidate at the window edge (e.g.
        # `yscale('logit')` on data that tops out at exactly 0.999) can miss a plain
        # `<=` by a couple ULPs once it has round-tripped through forward()/inverse(),
        # same trap `_generate_symmetric_ticks` hit.
        tol = 1e-9
        for k in range(1, 12):
            p = 10.0**-k
            if lo_real - tol <= p <= hi_real + tol:
                candidates.add(p)
            if lo_real - tol <= 1.0 - p <= hi_real + tol:
                candidates.add(1.0 - p)
        major_real = np.array(sorted(candidates))

        tick_format = str(getattr(self.options, "axis_tick_format", "") or "")
        if major_real.size < 3:
            step_real = _nice_step((hi_real - lo_real) / max(1, target_count))
            start_real = np.ceil(lo_real / step_real) * step_real
            end_real = np.floor(hi_real / step_real) * step_real
            if start_real > end_real:
                return AxisTicks()
            major_real = np.arange(start_real, end_real + step_real / 2, step_real)
            major_real = major_real[(major_real > 0) & (major_real < 1)]
            if major_real.size == 0:
                return AxisTicks()
            labels = _apply_format(major_real, step_real, tick_format)
        else:
            gaps = np.abs(np.diff(major_real))
            gaps = gaps[gaps > 0]
            step_real = float(gaps.min()) if gaps.size else 1.0
            labels = _apply_format(major_real, step_real, tick_format)

        major = _scale_forward(major_real, "logit")
        return AxisTicks(major=major, labels=labels, minor=np.array([]), step=0.0)

    def _generate_minor(self, vmin: float, vmax: float, start: float, step: float) -> np.ndarray:
        """Subdivide the major spacing, dropping any position that coincides with a major.

        Generated across the whole visible span rather than only between the first and last
        major, so the partial intervals at both ends are subdivided too -- otherwise the
        minor ticks stop short of the frame and look like a rendering bug.
        """
        if not getattr(self.options, "axis_minor_ticks", False):
            return np.array([])

        subdivisions = int(getattr(self.options, "axis_minor_subdivisions", 5) or 0)
        if subdivisions < 2:
            return np.array([])

        minor_step = step / subdivisions
        first = np.ceil(vmin / minor_step) * minor_step
        if first > vmax:
            return np.array([])

        minor = np.arange(first, vmax + minor_step / 2, minor_step)
        minor = minor[(minor >= vmin) & (minor <= vmax)]
        if minor.size == 0:
            return minor

        # Drop the ones sitting on a major. Comparing the index against the subdivision
        # count is exact-ish where a float compare against `major` would miss on rounding.
        index = np.rint((minor - start) / minor_step)
        on_major = np.isclose(minor, start + index * minor_step) & (
            np.mod(index, subdivisions) == 0
        )
        return minor[~on_major]
