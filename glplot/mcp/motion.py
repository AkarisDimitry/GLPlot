"""Turning rendered frames into an animation file, a contact sheet, and a verdict on the motion.

A model cannot watch an animation, so ``animate`` hands back what it *can* check:

* a **contact sheet** -- a handful of frames side by side, labelled with their frame number and
  time, which shows what changed from the first frame to the last;
* a **motion review** -- facts measured on the frames themselves: nothing moves, the axes limits
  jump between frames (so the picture zooms or jitters), the loop does not close, one step is far
  bigger than the rest, the file is huge.

Frames arrive as PNG bytes (small, and cheap to keep for hundreds of frames). Encoding needs
only Pillow, which GLPlot already depends on, for GIF and APNG; MP4 and WebM use an ``ffmpeg``
executable when one is on the PATH. Nothing here touches a figure or the MCP SDK.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .state import McpError

__all__ = [
    "ANIMATION_EXTENSIONS",
    "available_formats",
    "contact_sheet",
    "encode",
    "find_ffmpeg",
    "frame_steps",
    "review_motion",
    "sample_indices",
    "write_sequence",
]

#: What ``animate`` may write (one file), by extension. A ``%04d`` pattern in a ``.png`` name
#: writes one image per frame instead.
ANIMATION_EXTENSIONS: Tuple[str, ...] = (".gif", ".apng", ".mp4", ".webm")

#: Mean absolute change (0..1 of full scale) below which two frames count as identical.
_STILL = 5e-5


# ----------------------------------------------------------------------------------
# ffmpeg and formats
# ----------------------------------------------------------------------------------


def find_ffmpeg() -> Optional[str]:
    """The ``ffmpeg`` executable: ``$GLPLOT_FFMPEG``, ``$IMAGEIO_FFMPEG_EXE`` or the PATH."""
    for name in ("GLPLOT_FFMPEG", "IMAGEIO_FFMPEG_EXE"):
        candidate = os.environ.get(name)
        if candidate and Path(candidate).is_file():
            return candidate
    return shutil.which("ffmpeg")


@lru_cache(maxsize=4)
def _encoders(ffmpeg: str) -> frozenset:
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            names.add(parts[1])
    return frozenset(names)


_H264 = ("libx264", "libopenh264", "h264_videotoolbox", "h264_nvenc", "h264_qsv", "mpeg4")
_VPX = ("libvpx-vp9", "libvpx")


def _pick(ffmpeg: str, preferred: Sequence[str]) -> Optional[str]:
    have = _encoders(ffmpeg)
    return next((codec for codec in preferred if codec in have), None)


def available_formats() -> Dict[str, Optional[str]]:
    """Every output format with ``None`` when it works, else why it does not."""
    ffmpeg = find_ffmpeg()
    why_not = "needs the ffmpeg executable on the PATH (e.g. `brew install ffmpeg`)"
    out: Dict[str, Optional[str]] = {"gif": None, "apng": None, "png-sequence": None}
    out["mp4"] = None if ffmpeg and _pick(ffmpeg, _H264) else why_not
    out["webm"] = (
        None
        if ffmpeg and _pick(ffmpeg, _VPX)
        else why_not + " built with libvpx" if ffmpeg else why_not
    )
    return out


# ----------------------------------------------------------------------------------
# Decoding and fitting frames
# ----------------------------------------------------------------------------------


def _image(png: bytes) -> Any:
    from PIL import Image

    image = Image.open(io.BytesIO(png))
    image.load()
    return image.convert("RGB")


def _fitted(png: bytes, size: Tuple[int, int], background: Tuple[int, int, int]) -> Any:
    """The frame as RGB exactly ``size`` (a frame a pixel off is padded or cropped, not refused)."""
    from PIL import Image

    image = _image(png)
    if image.size == size:
        return image
    canvas = Image.new("RGB", size, background)
    canvas.paste(image, (0, 0))
    return canvas


def _mp4_dimensions(size: Tuple[int, int]) -> Tuple[int, int]:
    return size[0] + size[0] % 2, size[1] + size[1] % 2


# ----------------------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------------------


def write_sequence(frames: Sequence[bytes], directory: Path, pattern: str) -> List[Path]:
    """Write each frame as ``pattern % number`` (1-based) in ``directory``; return the paths."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for number, png in enumerate(frames, start=1):
        path = directory / (pattern % number)
        path.write_bytes(png)
        paths.append(path)
    return paths


def encode(
    frames: Sequence[bytes],
    fps: float,
    path: Path,
    fmt: str,
    *,
    repeat: int = 0,
    background: Tuple[int, int, int] = (255, 255, 255),
    quality: str = "good",
) -> Dict[str, Any]:
    """Encode ``frames`` (PNG bytes) to ``path`` as ``fmt`` and describe what was written.

    ``repeat`` is how many extra times a GIF/APNG plays after the first (0: forever).
    ``quality`` ("draft", "good", "high") trades size for fidelity in MP4/WebM.
    """
    if not frames:
        raise McpError("There are no frames to encode.")
    first = _image(frames[0])
    size = first.size
    info: Dict[str, Any] = {"format": fmt, "size_px": list(size)}

    if fmt in ("gif", "apng"):
        from PIL import Image

        images = [first] + [_fitted(png, size, background) for png in frames[1:]]
        if fmt == "gif":
            # A GIF's frame delay is a whole number of centiseconds, and most players treat
            # anything under 2 as 10: the nearest delay is what the file really plays at.
            delay = max(20, int(round(1000.0 / fps / 10.0)) * 10)
            palette = [
                im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=0) for im in images
            ]
            palette[0].save(
                path,
                save_all=True,
                append_images=palette[1:],
                duration=delay,
                loop=repeat,
                optimize=False,
                disposal=1,
            )
            info["fps_effective"] = round(1000.0 / delay, 3)
            info["codec"] = "gif (256 colours per frame)"
        else:
            delay = max(1, int(round(1000.0 / fps)))
            images[0].save(
                path,
                format="PNG",
                save_all=True,
                append_images=images[1:],
                duration=delay,
                loop=repeat,
            )
            info["fps_effective"] = round(1000.0 / delay, 3)
            info["codec"] = "apng (lossless)"
        return info

    if fmt in ("mp4", "webm"):
        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            raise McpError(
                f"{fmt} output needs the ffmpeg executable, which is not on the PATH. Install it "
                "(`brew install ffmpeg`, `apt install ffmpeg`) or use a .gif or .apng file."
            )
        codec = _pick(ffmpeg, _H264 if fmt == "mp4" else _VPX)
        if codec is None:
            raise McpError(
                f"This ffmpeg has no {'H.264' if fmt == 'mp4' else 'VP8/VP9'} encoder. "
                "Use a .gif file, or install a full ffmpeg build."
            )
        width, height = _mp4_dimensions(size)
        crf = {"draft": "30", "good": "23", "high": "17"}.get(quality, "23")
        args = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
            "-r", f"{fps:g}", "-i", "-",
            "-vf", f"pad={width}:{height}:0:0:color=white",
            "-an", "-c:v", codec, "-pix_fmt", "yuv420p",
        ]  # fmt: skip
        if codec == "libx264":
            args += ["-crf", crf, "-preset", "medium"]
        elif codec in ("libvpx-vp9", "libvpx"):
            args += ["-crf", crf, "-b:v", "0"]
        elif codec == "mpeg4":
            args += ["-q:v", {"draft": "8", "good": "4", "high": "2"}.get(quality, "4")]
        else:
            args += ["-b:v", {"draft": "2M", "good": "6M", "high": "14M"}.get(quality, "6M")]
        if fmt == "mp4":
            args += ["-movflags", "+faststart"]
        args.append(str(path))
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=errors)
            try:
                assert process.stdin is not None
                for png in frames:
                    process.stdin.write(np.asarray(_fitted(png, size, background)).tobytes())
                process.stdin.close()
                code = process.wait(timeout=300)
            except (BrokenPipeError, subprocess.TimeoutExpired, OSError) as exc:
                process.kill()
                errors.seek(0)
                detail = errors.read().decode("utf-8", "replace").strip()[-400:]
                raise McpError(f"ffmpeg failed while encoding {fmt}: {detail or exc}") from exc
            if code != 0:
                errors.seek(0)
                detail = errors.read().decode("utf-8", "replace").strip()[-400:]
                path.unlink(missing_ok=True)
                raise McpError(f"ffmpeg failed ({code}) while encoding {fmt}: {detail}")
        info["fps_effective"] = float(fps)
        info["codec"] = codec
        return info

    raise McpError(f"Cannot encode {fmt!r}.")  # pragma: no cover - the caller validates it


# ----------------------------------------------------------------------------------
# The contact sheet
# ----------------------------------------------------------------------------------


def sample_indices(n: int, count: int) -> List[int]:
    """``count`` frame numbers spread evenly over ``n`` frames, first and last included."""
    count = max(1, min(int(count), n))
    if count == 1:
        return [0]
    return sorted({int(round(i * (n - 1) / (count - 1))) for i in range(count)})


def _font(size: int) -> Any:
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow before 10.1 has only the bitmap font
        return ImageFont.load_default()


def contact_sheet(
    frames: Sequence[bytes],
    indices: Sequence[int],
    labels: Sequence[str],
    *,
    columns: int = 4,
    max_width: int = 1600,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> bytes:
    """The chosen frames in a grid, each captioned, as one PNG no wider than ``max_width``."""
    from PIL import Image, ImageDraw

    chosen = [_image(frames[i]) for i in indices]
    columns = max(1, min(columns, len(chosen)))
    rows = -(-len(chosen) // columns)
    gap = 6
    cell_w = (max_width - gap * (columns + 1)) // columns
    base_w, base_h = chosen[0].size
    cell_w = min(cell_w, base_w)
    cell_h = int(round(base_h * cell_w / base_w))
    caption = max(14, cell_w // 22)
    sheet = Image.new(
        "RGB",
        (columns * cell_w + gap * (columns + 1), rows * (cell_h + caption + gap) + gap),
        (235, 235, 235),
    )
    font = _font(max(10, caption - 3))
    draw = ImageDraw.Draw(sheet)
    for slot, (image, label) in enumerate(zip(chosen, labels)):
        row, col = divmod(slot, columns)
        x = gap + col * (cell_w + gap)
        y = gap + row * (cell_h + caption + gap)
        tile = image.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        sheet.paste(tile, (x, y + caption))
        draw.text((x + 2, y + 1), label, fill=(40, 40, 40), font=font)
    out = io.BytesIO()
    sheet.save(out, format="PNG", optimize=False)
    return out.getvalue()


# ----------------------------------------------------------------------------------
# Measuring motion
# ----------------------------------------------------------------------------------


def _gray(png: bytes, edge: int = 160) -> np.ndarray:
    from PIL import Image

    image = _image(png).convert("L")
    scale = edge / max(image.size)
    if scale < 1.0:
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.BILINEAR,
        )
    return np.asarray(image, dtype=np.float32) / 255.0


def frame_steps(frames: Sequence[bytes], *, loop: bool = False) -> np.ndarray:
    """Mean absolute change between consecutive frames (0..1); a loop adds last -> first."""
    small = [_gray(png) for png in frames]
    shape = small[0].shape
    small = [g if g.shape == shape else np.resize(g, shape) for g in small]
    steps = [float(np.mean(np.abs(b - a))) for a, b in zip(small, small[1:])]
    if loop and len(small) > 2:
        steps.append(float(np.mean(np.abs(small[0] - small[-1]))))
    return np.asarray(steps, dtype=np.float64)


def _issue(severity: str, code: str, message: str, fix: str = "") -> Dict[str, Any]:
    out = {"severity": severity, "code": code, "message": message}
    if fix:
        out["fix"] = fix
    return out


def review_motion(
    steps: np.ndarray,
    *,
    loop: bool,
    views: Sequence[Sequence[Mapping[str, Any]]],
    bound_limits: Mapping[Tuple[int, str], bool],
    file_bytes: Optional[int] = None,
    fmt: Optional[str] = None,
    stage_edges: Sequence[int] = (),
) -> List[Dict[str, Any]]:
    """Provable problems with the motion: facts about the frames, each with a fix to try."""
    issues: List[Dict[str, Any]] = []
    transitions = steps[: len(steps) - 1] if loop and len(steps) > 1 else steps
    moving = transitions[transitions > _STILL]
    if transitions.size and moving.size == 0:
        issues.append(
            _issue(
                "warning",
                "static",
                "No frame differs from the next: the animation does not move.",
                "bind a property to t, time or a variable (bindings / stages), and check the "
                "target ids with describe_figure.",
            )
        )
        return issues
    if transitions.size >= 4:
        still = float(np.mean(transitions <= _STILL))
        if still > 0.5:
            issues.append(
                _issue(
                    "info",
                    "mostly_still",
                    f"{100 * still:.0f}% of the frame-to-frame steps show no change; only part of "
                    "the animation moves.",
                    "shorten the stage that does nothing, or make it do something.",
                )
            )
    if transitions.size >= 6 and moving.size >= 3:
        # A real jump is a spike: its neighbours are small. A fast stretch of an eased move is
        # not -- its neighbours are nearly as large -- so compare each step with its neighbours.
        padded = np.concatenate([[transitions[0]], transitions, [transitions[-1]]])
        around = (padded[:-2] + padded[2:]) / 2.0
        spike = transitions / np.maximum(around, 1e-9)
        biggest = int(np.argmax(np.where(transitions > 0.01, spike, 0.0)))
        if spike[biggest] > 6.0 and transitions[biggest] > 0.01:
            where = f"between frames {biggest} and {biggest + 1}"
            edge = " (a stage boundary)" if (biggest + 1) in set(stage_edges) else ""
            issues.append(
                _issue(
                    "info",
                    "abrupt_change",
                    f"The picture jumps {spike[biggest]:.0f}x more than its neighbouring steps "
                    f"{where}{edge}.",
                    "ease the change (ease='in_out'), or start the next stage from where the "
                    "previous one ended.",
                )
            )
    if loop and len(steps) > 2 and moving.size:
        seam = float(steps[-1])
        typical = float(np.median(moving))
        if seam > max(3.0 * typical, 0.002):
            issues.append(
                _issue(
                    "warning",
                    "loop_seam",
                    f"The last frame does not lead into the first (the jump is "
                    f"{seam / max(typical, 1e-9):.0f}x a normal step), so the loop will visibly "
                    "snap.",
                    "make every moving quantity periodic over the animation: whole turns "
                    "(2*pi*t), and use loop=true so the last frame stops short of the first.",
                )
            )
    # Does the view move between frames?
    panels = max((len(v) for v in views), default=0)
    for panel in range(panels):
        for axis in ("x", "y"):
            lows, highs = [], []
            for view in views:
                if panel < len(view) and view[panel].get(f"{axis}lim"):
                    low, high = view[panel][f"{axis}lim"]
                    lows.append(low)
                    highs.append(high)
            if len(lows) < 2:
                continue
            span = max(highs) - min(lows)
            if span <= 0:
                continue
            drift = max(max(lows) - min(lows), max(highs) - min(highs)) / span
            if drift > 0.02 and not bound_limits.get((panel + 1, axis)):
                issues.append(
                    _issue(
                        "warning",
                        f"{axis}lim_moves",
                        f"Panel {panel + 1}: the {axis} range changes between frames "
                        f"({min(lows):.4g}..{max(highs):.4g}), so the picture rescales or jitters.",
                        f"pin it first: set_axes(figure_id, panel={panel + 1}, "
                        f"{axis}lim=[lo, hi]).",
                    )
                )
    if file_bytes is not None:
        if fmt in ("gif", "apng") and file_bytes > 25 * 1024 * 1024:
            issues.append(
                _issue(
                    "suggestion",
                    "large_file",
                    f"The {fmt} is {file_bytes / 1048576:.0f} MB.",
                    "lower scale or fps, or write an .mp4 (ffmpeg), which is far smaller.",
                )
            )
    return issues
