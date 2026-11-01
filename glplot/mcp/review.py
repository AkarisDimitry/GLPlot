"""A deterministic review of a rendered figure, for a model that cannot trust its eyes on pixels.

A model looks at the picture, but small text, a legend that is simply absent, a series that left
the visible range or a log axis that quietly dropped its zeros are easy to miss in an image and
trivial to check in the figure the picture came from. :func:`review_figure` reads that matplotlib
figure (the one :func:`glplot.utils.preview.render_preview_bytes` hands to its ``inspect`` hook) and
reports what it can *prove*, each with the tool call that fixes it.

It makes no aesthetic judgements. Every finding is a fact about the figure, with a code a test can
pin and a severity saying how much it matters:

* ``warning``    -- something is probably wrong (an unlabelled axis, no legend for several
  series, an empty panel, points a log axis cannot show);
* ``suggestion`` -- the figure works but would read better (a title, a crowded legend);
* ``info``       -- a fact worth knowing, often intentional (a zoomed view hides data).

Pure functions of a matplotlib ``Figure``: no MCP SDK, no GL.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

__all__ = ["MAX_ISSUES", "review_figure"]

#: Findings returned at most; a model needs the worst few, not a lint report.
MAX_ISSUES = 8

_SEVERITY_ORDER = {"warning": 0, "suggestion": 1, "info": 2}
#: Above this many plotted points the tick-label measurement (which needs a full draw) is skipped.
_DRAW_BUDGET_POINTS = 50_000
#: Fraction of the data span outside the view before it is worth mentioning.
_CLIP_TOLERANCE = 0.02


def _issue(panel: int, severity: str, code: str, message: str, fix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {"panel": panel, "severity": severity, "code": code, "message": message}
    if fix:
        out["fix"] = fix
    return out


def _data_axes(fig: Any) -> List[Any]:
    return [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]


def _is_3d(ax: Any) -> bool:
    return hasattr(ax, "get_zlim")


def _series(ax: Any) -> List[Dict[str, Any]]:
    """The plotted point series of ``ax`` (lines and scatters), with their label and values."""
    found: List[Dict[str, Any]] = []
    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        if x.size:
            found.append({"label": line.get_label(), "x": x, "y": y, "kind": "line"})
    for collection in ax.collections:
        offsets = getattr(collection, "get_offsets", None)
        if offsets is None:
            continue
        try:
            points = np.asarray(offsets(), dtype=float)
        except (TypeError, ValueError):
            continue
        if points.ndim == 2 and points.shape[1] == 2 and points.shape[0]:
            found.append(
                {
                    "label": collection.get_label(),
                    "x": points[:, 0],
                    "y": points[:, 1],
                    "kind": "scatter",
                }
            )
    return found


def _is_public(label: str) -> bool:
    return bool(label) and not str(label).startswith("_")


def _has_content(ax: Any) -> bool:
    return bool(ax.lines or ax.collections or ax.patches or ax.images or ax.texts or ax.containers)


def _total_points(axes: Sequence[Any]) -> int:
    total = 0
    for ax in axes:
        for series in _series(ax):
            total += series["x"].size
        for image in ax.images:
            array = image.get_array()
            total += int(np.size(array)) if array is not None else 0
    return total


def _span_text(low: float, high: float) -> str:
    return f"{low:.4g}..{high:.4g}"


def _check_axes(
    ax: Any,
    panel: int,
    fig_title: bool,
    known_labels: Sequence[str] = (),
    legend_wanted: bool = True,
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    where = f"panel {panel}"
    if not _has_content(ax):
        return [
            _issue(
                panel,
                "warning",
                "empty_panel",
                f"{where.capitalize()} has nothing drawn in it.",
                f"add_plot(..., panel={panel}), or build the figure with fewer rows/cols.",
            )
        ]

    # The export only attaches a series' label once a legend is asked for, so the labels the caller
    # gave are the ones to count; the figure's own handles cover whatever else it knows.
    _, labels = ax.get_legend_handles_labels()
    named = list(dict.fromkeys([*known_labels, *[l for l in labels if _is_public(l)]]))
    legend = ax.get_legend()
    if legend_wanted and len(named) >= 2 and legend is None:
        issues.append(
            _issue(
                panel,
                "warning",
                "legend_missing",
                f"{where.capitalize()} has {len(named)} labelled series ({', '.join(named[:4])}"
                f"{'...' if len(named) > 4 else ''}) but no legend.",
                f"set_axes(figure_id, panel={panel}, legend=true).",
            )
        )
    if legend is not None and not [t for t in legend.get_texts() if _is_public(t.get_text())]:
        issues.append(
            _issue(
                panel,
                "suggestion",
                "legend_empty",
                f"{where.capitalize()} shows a legend with no entries.",
                "give each series a label= in add_plot, or turn the legend off.",
            )
        )
    if len(named) >= 9:
        issues.append(
            _issue(
                panel,
                "suggestion",
                "crowded_legend",
                f"{where.capitalize()} has {len(named)} labelled series; the legend will be "
                "crowded and the colours hard to tell apart.",
                "split them over several panels (create_figure rows/cols) or plot fewer.",
            )
        )

    drawn_axes = ax.axison and not ax.images
    if drawn_axes and not _is_3d(ax):
        for axis, text in (("x", ax.get_xlabel()), ("y", ax.get_ylabel())):
            if not text.strip():
                issues.append(
                    _issue(
                        panel,
                        "warning",
                        f"no_{axis}label",
                        f"{where.capitalize()} has no {axis}-axis label.",
                        f"set_axes(figure_id, panel={panel}, {axis}label='quantity (unit)').",
                    )
                )
    if not fig_title and not ax.get_title():
        issues.append(
            _issue(
                panel,
                "suggestion",
                "no_title",
                f"{where.capitalize()} has no title.",
                f"set_axes(figure_id, panel={panel}, title='What this shows').",
            )
        )
    if _is_3d(ax):
        return issues

    series = _series(ax)
    for axis in ("x", "y"):
        if getattr(ax, f"get_{axis}scale")() != "log":
            continue
        dropped = sum(int(np.sum(s[axis] <= 0)) for s in series)
        if dropped:
            issues.append(
                _issue(
                    panel,
                    "warning",
                    f"log_{axis}_nonpositive",
                    f"{dropped} point(s) have {axis} <= 0 and are not drawn on the log {axis} "
                    "axis.",
                    f"set_axes(figure_id, {axis}scale='symlog') or 'linear', or remove those "
                    "values.",
                )
            )
    for s in series:
        name = s["label"] if _is_public(s["label"]) else "a series"
        y = s["y"][np.isfinite(s["y"])]
        if s["kind"] == "line" and y.size >= 3 and float(np.ptp(y)) == 0.0:
            issues.append(
                _issue(
                    panel,
                    "info",
                    "constant_series",
                    f"{name.capitalize() if name == 'a series' else repr(name)} in {where} is "
                    f"constant (y = {y[0]:.4g}).",
                    "check that the right column was plotted.",
                )
            )
        gaps = int(np.sum(~np.isfinite(s["y"])))
        if gaps and s["y"].size:
            issues.append(
                _issue(
                    panel,
                    "info",
                    "missing_values",
                    f"{name.capitalize() if name == 'a series' else repr(name)} in {where} has "
                    f"{gaps} missing (NaN/inf) value(s); they appear as gaps.",
                )
            )

    limits = ax.dataLim
    for axis, view in (("x", ax.get_xlim()), ("y", ax.get_ylim())):
        low, high = sorted(float(v) for v in view)
        data = limits.intervalx if axis == "x" else limits.intervaly
        data_low, data_high = float(data[0]), float(data[1])
        if getattr(ax, f"get_{axis}scale")() == "log" and data_low <= 0:
            continue  # the values a log axis cannot show were already reported
        span = data_high - data_low
        if not np.isfinite(span) or span <= 0:
            continue
        hidden = max(0.0, low - data_low) + max(0.0, data_high - high)
        if hidden / span > _CLIP_TOLERANCE:
            issues.append(
                _issue(
                    panel,
                    "info",
                    f"data_outside_{axis}lim",
                    f"{where.capitalize()}: the data spans {axis} = "
                    f"{_span_text(data_low, data_high)} but the view shows "
                    f"{_span_text(low, high)}; about {100 * min(1.0, hidden / span):.0f}% "
                    "of the range is cut off (fine if you meant to zoom).",
                    f"set_axes(figure_id, panel={panel}, "
                    f"{axis}lim=[{data_low:.4g}, {data_high:.4g}]) "
                    "to show everything.",
                )
            )
    return issues


def _tick_overlap(fig: Any, axes: Sequence[Any]) -> List[Dict[str, Any]]:
    """Adjacent tick labels whose boxes overlap -- the usual fate of long category names."""
    issues: List[Dict[str, Any]] = []
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:  # noqa: BLE001 - a backend without a renderer just skips the check
        return issues
    for panel, ax in enumerate(axes, start=1):
        if _is_3d(ax) or not ax.axison:
            continue
        for axis, labels in (("x", ax.get_xticklabels()), ("y", ax.get_yticklabels())):
            boxes = [
                t.get_window_extent(renderer)
                for t in labels
                if t.get_visible() and t.get_text().strip()
            ]
            boxes.sort(key=lambda b: b.x0 if axis == "x" else -b.y1)
            touching = 0
            for first, second in zip(boxes, boxes[1:]):
                if axis == "x":
                    touching += int(second.x0 < first.x1 - 1.0 and second.y0 < first.y1)
                else:
                    touching += int(second.y1 > first.y0 + 1.0 and second.x0 < first.x1)
            if touching >= 1 and len(boxes) >= 3:
                issues.append(
                    _issue(
                        panel,
                        "warning",
                        f"{axis}_tick_labels_overlap",
                        f"Panel {panel}: {touching} pair(s) of {axis}-axis tick labels overlap.",
                        "use a wider figure (create_figure width), shorter names, or barh for long "
                        "category names.",
                    )
                )
    return issues


def review_figure(
    fig: Any,
    *,
    labelled: Optional[Mapping[int, Sequence[str]]] = None,
    measure_text: Optional[bool] = None,
    no_legend: Sequence[int] = (),
) -> List[Dict[str, Any]]:
    """The provable problems of ``fig``, worst first, at most :data:`MAX_ISSUES`.

    ``labelled`` maps a 1-based panel number to the series labels the caller attached there (the
    figure itself only carries a label once a legend is requested). ``measure_text`` forces (or
    forbids) the tick-label overlap check, which needs a full draw; by default it runs unless the
    figure holds so many points that a second draw would be slow. ``no_legend`` lists the panels
    whose legend was switched off on purpose, which are not told to add one.
    """
    axes = _data_axes(fig)
    has_suptitle = getattr(fig, "_suptitle", None) is not None and bool(fig._suptitle.get_text())
    issues: List[Dict[str, Any]] = []
    for panel, ax in enumerate(axes, start=1):
        issues.extend(
            _check_axes(
                ax,
                panel,
                has_suptitle,
                (labelled or {}).get(panel, ()),
                legend_wanted=panel not in set(no_legend),
            )
        )
    if measure_text is None:
        measure_text = _total_points(axes) <= _DRAW_BUDGET_POINTS
    if measure_text:
        issues.extend(_tick_overlap(fig, axes))
    # A title missing on every panel is one finding, not N: keep the first.
    seen_title = False
    unique: List[Dict[str, Any]] = []
    for item in issues:
        if item["code"] == "no_title":
            if seen_title:
                continue
            seen_title = True
            if len(axes) > 1:
                item = dict(item, message="The figure has no title.", panel=0)
                item["fix"] = "create_figure(title=...) or set_axes(title=...) per panel."
        unique.append(item)
    unique.sort(key=lambda item: (_SEVERITY_ORDER[item["severity"]], item["panel"]))
    return unique[:MAX_ISSUES]
