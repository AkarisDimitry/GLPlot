"""``animate``: build a timeline, replay the figure once per frame, encode, and report.

The figure is never mutated while animating. For each frame the toolbox rebuilds a scratch copy of
the figure from its recorded operations with that frame's property values substituted
(:meth:`glplot.mcp.toolbox.Toolbox._build`), renders it, and throws it away. Every kind of artist,
2-D or 3-D, therefore animates the same way and through the same validated code path as a static
figure, and a broken binding fails on frame 0 with the figure intact.

Everything about *what* varies lives in :mod:`glplot.mcp.dynamics`; everything about *files* in
:mod:`glplot.mcp.motion`. This module is the loop between them.
"""

from __future__ import annotations

import io
import re
import time as clock
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import dynamics, motion
from .state import McpError

__all__ = ["run_animation", "parse_bindings"]

_SEQUENCE = re.compile(r"%0(\d)d")
#: Longest edge, in pixels, an animation frame is rendered at unless ``scale`` says otherwise.
_AUTO_EDGE = 960
#: The export's native resolution; ``scale`` multiplies it.
_BASE_DPI = 120.0


def _dpi(crop: Any, scale: Optional[float], *, fit: bool) -> float:
    """Resolution for the frames: ``scale`` x native, or (``fit``) whatever keeps ~960 px."""
    if scale is not None:
        return _BASE_DPI * float(scale)
    if not fit:
        return _BASE_DPI
    return min(_BASE_DPI, _AUTO_EDGE / max(crop.width, crop.height))


def parse_bindings(items: Optional[Sequence[Mapping[str, Any]]]) -> List[dynamics.Binding]:
    """A list of ``{target, property, value}`` objects as :class:`Binding` objects."""
    out: List[dynamics.Binding] = []
    if items is None:
        return out
    if not isinstance(items, (list, tuple)):
        raise McpError("bindings is a list of {target, property, value} objects.")
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise McpError(f"bindings[{index}] must be an object {{target, property, value}}.")
        unknown = set(item) - {"target", "property", "value"}
        missing = {"target", "property", "value"} - set(item)
        if unknown or missing:
            raise McpError(
                f"bindings[{index}] needs exactly target, property and value"
                + (f" (unknown: {sorted(unknown)})" if unknown else "")
                + (f" (missing: {sorted(missing)})" if missing else "")
                + "."
            )
        out.append(dynamics.Binding(str(item["target"]), str(item["property"]), item["value"]))
    return out


def _merge(base: Sequence[dynamics.Binding], extra: Sequence[dynamics.Binding]) -> List[Any]:
    """The figure's bindings with ``extra`` on top (the same target and property is replaced)."""
    merged = {(b.target, b.prop): b for b in base}
    merged.update({(b.target, b.prop): b for b in extra})
    return list(merged.values())


def _resolve_output(
    box: Any, output: Optional[str], overwrite: bool, frames: int
) -> Tuple[Optional[Path], Optional[str], Optional[str]]:
    """``(path, format, pattern)`` for ``output``; checked before any frame is rendered."""
    if output is None:
        return None, None, None
    text = str(output)
    suffix = Path(text).suffix.lower()
    sequence = _SEQUENCE.search(Path(text).name)
    if suffix == ".png" and sequence:
        pattern = Path(text).name
        first = box.sandbox.write_path(str(Path(text).with_name(pattern % 1)), overwrite=overwrite)
        for number in range(2, frames + 1):
            later = first.with_name(pattern % number)
            if later.exists() and not overwrite:
                raise McpError(
                    f"{later.name} already exists. Pass overwrite=true or choose another folder."
                )
        return first, "png-sequence", pattern
    if suffix == ".png":
        raise McpError(
            "A .png output is one image. For an animation use .gif, .apng, .mp4 or .webm, or a "
            "numbered pattern such as 'frames/f_%04d.png' to write one image per frame."
        )
    if suffix not in motion.ANIMATION_EXTENSIONS:
        raise McpError(
            f"Cannot write {suffix or 'a file with no extension'!r}. Use one of: "
            f"{', '.join(motion.ANIMATION_EXTENSIONS)}, or 'frames/f_%04d.png'."
        )
    fmt = suffix[1:]
    problem = motion.available_formats().get(fmt)
    if problem:
        raise McpError(f"Cannot write {fmt}: {problem}. Use .gif or .apng instead.")
    target = box.sandbox.write_path(
        text, overwrite=overwrite, extensions=motion.ANIMATION_EXTENSIONS
    )
    return target, fmt, None


def _hold_frames(seconds: Any, fps: float, what: str) -> int:
    if seconds in (None, 0, 0.0):
        return 0
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds < 0:
        raise McpError(f"{what} is a number of seconds, 0 or more.")
    if seconds > 30:
        raise McpError(f"{what} of {seconds:g} s is more than the 30 s allowed.")
    return int(round(seconds * fps))


def run_animation(
    box: Any,
    entry: Any,
    *,
    output: Optional[str],
    frames: Optional[int],
    duration: Optional[float],
    fps: float,
    loop: bool,
    stages: Optional[Sequence[Mapping[str, Any]]],
    variables: Optional[Mapping[str, Any]],
    bindings: Optional[Sequence[Mapping[str, Any]]],
    scale: Optional[float],
    hold_start: Any,
    hold_end: Any,
    sheet_frames: int,
    frame: Optional[int],
    overwrite: bool,
    repeat: int,
    quality: str,
) -> Tuple[Dict[str, Any], bytes, str]:
    """Run an animation for ``entry`` (a toolbox figure). Returns ``(data, image, headline)``."""
    caps = box.limits.animation
    started = clock.monotonic()
    notes: List[str] = []

    if quality not in ("draft", "good", "high"):
        raise McpError("quality must be 'draft', 'good' or 'high'.")
    if isinstance(repeat, bool) or not isinstance(repeat, int) or not 0 <= repeat <= 1000:
        raise McpError("repeat is the number of extra plays of a GIF/APNG (0: forever).")
    if (
        isinstance(sheet_frames, bool)
        or not isinstance(sheet_frames, int)
        or not 1 <= sheet_frames <= 16
    ):
        raise McpError("sheet_frames must be a whole number between 1 and 16.")
    if scale is not None and (
        isinstance(scale, bool) or not isinstance(scale, (int, float)) or not 0.25 <= scale <= 3.0
    ):
        raise McpError("scale must be between 0.25 and 3.0 (or omitted to fit about 960 px).")

    controls = box._context_values(entry)
    timeline = dynamics.build_timeline(
        frames=frames,
        duration=duration,
        fps=fps,
        loop=loop,
        stages=stages,
        variables=variables,
        controls=controls,
        limits=box.limits,
    )
    n = timeline.n
    extra = parse_bindings(bindings)
    if len(entry.bindings) + len(extra) > caps.max_bindings:
        raise McpError(f"More than {caps.max_bindings} bindings.")
    combined = _merge(entry.bindings, extra)
    if not combined and not timeline.parameters and not variables:
        raise McpError(
            "Nothing varies: add bindings (what changes) and/or stages and variables (how). "
            "Example: bindings=[{'target': 'a1', 'property': 'parameters.phase', "
            "'value': '2*pi*t'}]. See get_guide('mcp-animation')."
        )
    context = timeline.arrays()
    per_artist, per_axes, data_bindings = box._bound_arguments(entry, combined, context, n)
    if not any(per_artist) and not any(per_axes) and not data_bindings:
        raise McpError(
            "No binding drives anything: bindings target artists ('a1'), 'axes' or 'data:name'. "
            "Variables alone do not draw -- bind a property to them."
        )
    for binding in data_bindings:
        _, name = dynamics.parse_target(binding.target)
        clash = sorted(set(box.datasets.get(name).column_names()) & set(context))
        if clash:
            notes.append(
                f"Column(s) {', '.join(clash)} of {name!r} share a name with animation variables "
                "and win inside data expressions; use time or frame for the clock there."
            )

    n_hold_start = _hold_frames(hold_start, timeline.fps, "hold_start")
    n_hold_end = _hold_frames(hold_end, timeline.fps, "hold_end")
    target, fmt, pattern = _resolve_output(
        box, output, bool(overwrite), n + n_hold_start + n_hold_end
    )
    if frame is not None and (
        isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < n
    ):
        raise McpError(f"frame must be a whole number from 0 to {n - 1}.")

    def render(
        i: int,
        dpi: Optional[float] = None,
        review: bool = False,
        crop: Any = None,
        bounds: Optional[List[Any]] = None,
    ) -> Tuple[bytes, List[Dict[str, Any]], Any]:
        scalars = timeline.at(i)
        datasets = box._frame_datasets(data_bindings, scalars)
        built = box._build(
            entry, artist_args=per_artist[i], axes_args=per_axes[i], datasets=datasets
        )
        try:
            # Layout is always scale 1: the export's ``scale`` resizes the figure in inches while
            # fonts stay in points, so it would re-flow the text. Resolution is the dpi.
            png, view = box._render(
                built, 1.0, "png", 6000, review=review, crop=crop, bounds=bounds, dpi=dpi
            )
            return png, view, list(built.issues)
        finally:
            box._release(built)

    def fixed_crop(samples: Sequence[int]) -> Any:
        """One crop (inches) that holds every sampled frame's text, with a small margin."""
        from matplotlib.transforms import Bbox

        boxes: List[Any] = []
        for i in samples:
            render(i, bounds=boxes)
        merged = Bbox.union(boxes)
        margin = 0.12 + 0.02 * max(merged.width, merged.height)
        return Bbox.from_extents(
            merged.x0 - margin, merged.y0 - margin, merged.x1 + margin, merged.y1 + margin
        )

    from PIL import Image

    # -- one frame at full size, nothing written ---------------------------------------
    if frame is not None:
        crop = fixed_crop(motion.sample_indices(n, 5))
        png, view, issues = render(frame, _dpi(crop, scale, fit=False), review=True, crop=crop)
        data = {
            "frame": frame,
            "time_s": round(float(timeline.time[frame]), 4),
            "variables": {k: _round(v) for k, v in timeline.at(frame).items()},
            "panels": view,
            "issues": issues,
        }
        return data, png, f"Frame {frame} of {n} (t={timeline.time[frame]:.2f} s)."

    # -- frame 0 sets the size ---------------------------------------------------------
    crop = fixed_crop(motion.sample_indices(n, 5))
    dpi = _dpi(crop, scale, fit=True)
    first_png, first_view, issues = render(0, dpi, review=True, crop=crop)
    width, height = Image.open(io.BytesIO(first_png)).size
    total_pixels = (n + n_hold_start + n_hold_end) * width * height
    if total_pixels > caps.max_pixels:
        raise McpError(
            f"{n} frames of {width}x{height} px is {total_pixels / 1e6:.0f} Mpx; the limit is "
            f"{caps.max_pixels / 1e6:.0f}. Lower scale, fps or the duration."
        )

    pngs: List[bytes] = [first_png]
    views: List[List[Dict[str, Any]]] = [first_view]
    timing_checked = False
    for i in range(1, n):
        png, view, _ = render(i, dpi, crop=crop)
        pngs.append(png)
        views.append(view)
        elapsed = clock.monotonic() - started
        if not timing_checked and i == 3:
            timing_checked = True
            projected = elapsed / 4.0 * n
            if projected > caps.max_seconds * 1.5:
                raise McpError(
                    f"At about {elapsed / 4.0:.2f} s per frame, {n} frames would take "
                    f"{projected:.0f} s (the limit is {caps.max_seconds:.0f} s). Use fewer frames "
                    "(lower fps or duration), a smaller figure, or scale=0.5."
                )
        if elapsed > caps.max_seconds:
            raise McpError(
                f"Stopped after {i + 1} of {n} frames: {elapsed:.0f} s is over the "
                f"{caps.max_seconds:.0f} s limit. Use fewer frames or a smaller scale."
            )

    # -- verdicts, sheet, file ---------------------------------------------------------
    steps = motion.frame_steps(pngs, loop=loop)
    bound_limits: Dict[Tuple[int, str], bool] = {}
    for binding in combined:
        kind, ident = dynamics.parse_target(binding.target)
        if kind == "axes" and binding.prop in ("xlim", "ylim"):
            bound_limits[(ident, binding.prop[0])] = True
    for operation in entry.log:
        if operation.tool == "set_axes":
            for key in ("xlim", "ylim"):
                if operation.args.get(key) is not None:
                    bound_limits[(int(operation.args.get("panel", 1)), key[0])] = True
    stage_edges = [s["start"] for s in timeline.stages[1:]]

    sequence = (
        [pngs[0]] * n_hold_start + pngs + [pngs[-1]] * n_hold_end
        if (n_hold_start or n_hold_end)
        else pngs
    )
    written: Dict[str, Any] = {}
    if target is not None and fmt is not None:
        started_write = clock.monotonic()
        if fmt == "png-sequence":
            paths = motion.write_sequence(sequence, target.parent, pattern or "frame_%04d.png")
            written = {
                "path": str(target.parent / (pattern or "frame_%04d.png")),
                "files": len(paths),
                "bytes": sum(p.stat().st_size for p in paths),
                "format": "png-sequence",
            }
        else:
            background = (255, 255, 255)
            info = motion.encode(
                sequence,
                timeline.fps,
                target,
                fmt,
                repeat=repeat,
                background=background,
                quality=quality,
            )
            written = {"path": str(target), "bytes": target.stat().st_size, **info}
            if (
                info.get("fps_effective")
                and abs(info["fps_effective"] - timeline.fps) > 0.03 * timeline.fps
            ):
                notes.append(
                    f"{fmt.upper()} timing is quantised: it plays at "
                    f"{info['fps_effective']:g} fps, "
                    f"not {timeline.fps:g}."
                )
        written["encode_s"] = round(clock.monotonic() - started_write, 2)
    else:
        notes.append(
            "No file written (no output given): the contact sheet is the preview. Pass "
            "output='name.gif' (or .mp4 / .webm / .apng / 'frames/f_%04d.png') to save it."
            + (
                ""
                if box.sandbox.enabled
                else " This server has no --root, so it cannot write files."
            )
        )

    indices = motion.sample_indices(n, sheet_frames)
    names = [timeline.stages[int(timeline.stage[i])]["name"] for i in indices] if stages else None
    labels = [
        f"#{i}  {timeline.time[i]:.2f}s" + (f"  [{names[k]}]" if names else "")
        for k, i in enumerate(indices)
    ]
    columns = 4 if len(indices) > 6 else min(3, len(indices)) if len(indices) > 2 else len(indices)
    sheet = motion.contact_sheet(pngs, indices, labels, columns=columns)

    found = motion.review_motion(
        steps,
        loop=loop,
        views=views,
        bound_limits=bound_limits,
        file_bytes=written.get("bytes"),
        fmt=fmt,
        stage_edges=stage_edges,
    )
    found = [dict(item, scope="motion") for item in found] + [
        dict(item, scope="figure") for item in issues if item.get("severity") == "warning"
    ]
    summary = timeline.summary()
    data: Dict[str, Any] = {
        "frames": n,
        "fps": timeline.fps,
        "duration_s": summary["duration_s"],
        "size_px": [width, height],
        "dpi": round(dpi, 1),
        "stages": summary["stages"] if stages else [],
        "variables": summary.get("variables", {}),
        "bindings": [b.describe() for b in combined],
        "sheet": indices,
        "issues": found,
        "elapsed_s": round(clock.monotonic() - started, 2),
    }
    if loop:
        data["loop"] = True
    if n_hold_start or n_hold_end:
        data["held_frames"] = [n_hold_start, n_hold_end]
    if written:
        data["file"] = written
    if notes:
        data["notes"] = notes
    worst = sum(1 for item in found if item["severity"] == "warning")
    headline = (
        f"Rendered {n} frames ({timeline.duration:.1f} s at {timeline.fps:g} fps, "
        f"{width}x{height} px)"
        + (f" -> {written['path']} ({written['bytes'] / 1024:.0f} KiB)" if written else "")
        + (f"; {worst} warning{'s' if worst != 1 else ''} in issues." if worst else ".")
    )
    return data, sheet, headline


def _round(value: Any) -> Any:
    try:
        return float(f"{float(value):.6g}")
    except (TypeError, ValueError):
        return value
