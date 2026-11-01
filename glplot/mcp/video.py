"""``compose_video``: several animated figures (the scenes) cut together into one video.

A complete explainer is a title card, a few scenes and a recap: one figure (and one ``animate``)
each, so every scene is built, previewed and fixed on its own. This module joins them.

* Frames come from the figure's **last animation**. ``animate`` keeps what it rendered (see
  ``Toolbox._remember_frames``), so a scene that has not changed since is not rendered again;
  one that has is re-rendered from its recorded specification.
* A figure with no animation is a **still**: ``{"figure_id": "fig3", "duration": 3}``.
* Every frame is fitted to one size (letterboxed in the page colour, never stretched) and one
  frame rate (each scene keeps its duration: frames are repeated or skipped).
* ``transition``: ``cut``; ``fade`` (out to the page colour and back in); ``crossfade`` (the two
  scenes overlap and blend, which shortens the video by the transition at each join).

The output frames are a *plan* (which source frame, which blend) materialised one at a time
while encoding, so a long MP4 never holds every decoded image at once.
"""

from __future__ import annotations

import time as clock
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import motion
from .state import McpError

__all__ = ["AUDIO_EXTENSIONS", "TRANSITIONS", "compose"]

TRANSITIONS = ("cut", "fade", "crossfade")
AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac")


@dataclass
class _Scene:
    figure_id: str
    frames: List[Any]  # PNG bytes, one per source frame
    fps: float
    source: str  # "cached", "rendered" or "still"
    page: Tuple[int, int, int] = (0, 0, 0)
    first: int = 0  # index of the scene's first output frame
    count: int = 0  # output frames before transitions

    def source_index(self, j: int, fps: float) -> int:
        """The source frame shown at output frame ``j`` of this scene."""
        return min(len(self.frames) - 1, int(j * self.fps / fps + 1e-9))


class _Frames(Sequence):
    """The composed video, frame by frame, built on demand (a ``Sequence`` of RGB images)."""

    def __init__(self, plan: List[Tuple[Any, ...]], scenes: List[_Scene], size, fps) -> None:
        self.plan, self.scenes, self.size, self.fps = plan, scenes, size, fps
        self._cache: Dict[Tuple[int, int], Any] = {}

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, index):  # type: ignore[override]
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        step = self.plan[index]
        if step[0] == "frame":
            return self._fitted(step[1], step[2])
        from PIL import Image

        if step[0] == "fade":  # ("fade", scene, j, level): towards the page colour
            scene, j, level = step[1], step[2], step[3]
            image = self._fitted(scene, j)
            page = Image.new("RGB", self.size, self.scenes[scene].page)
            return Image.blend(page, image, max(0.0, min(1.0, level)))
        # ("blend", scene_a, j_a, scene_b, j_b, weight of b)
        a = self._fitted(step[1], step[2])
        b = self._fitted(step[3], step[4])
        return Image.blend(a, b, max(0.0, min(1.0, step[5])))

    def _fitted(self, scene_index: int, j: int) -> Any:
        scene = self.scenes[scene_index]
        k = scene.source_index(j, self.fps)
        key = (scene_index, k)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        image = _letterbox(motion._image(scene.frames[k]), self.size, scene.page)
        if len(self._cache) > 8:
            self._cache.clear()
        self._cache[key] = image
        return image


def _letterbox(image: Any, size: Tuple[int, int], page: Tuple[int, int, int]) -> Any:
    """``image`` scaled to fit inside ``size`` (aspect kept), centred on the page colour."""
    from PIL import Image

    if image.size == tuple(size):
        return image
    scale = min(size[0] / image.width, size[1] / image.height)
    fitted = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, page)
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def _page(png: bytes) -> Tuple[int, int, int]:
    """The page colour of a frame: its top-left pixel (the figure background)."""
    image = motion._image(png)
    return tuple(int(c) for c in image.getpixel((1, 1)))[:3]  # type: ignore[return-value]


def _parse_segments(segments: Any) -> List[Dict[str, Any]]:
    if not isinstance(segments, (list, tuple)) or not segments:
        raise McpError(
            "segments is a list of scenes in order: figure ids ('fig1') or objects "
            "{'figure_id': 'fig1', 'hold_end': 1} / {'figure_id': 'fig3', 'duration': 3} (a still)."
        )
    out: List[Dict[str, Any]] = []
    for index, item in enumerate(segments):
        if isinstance(item, str):
            item = {"figure_id": item}
        if not isinstance(item, Mapping) or not isinstance(item.get("figure_id"), str):
            raise McpError(f"segments[{index}] must be a figure id or {{'figure_id': ...}}.")
        unknown = set(item) - {"figure_id", "duration", "hold_end"}
        if unknown:
            raise McpError(
                f"segments[{index}] has unknown key(s) {sorted(unknown)}; use figure_id, "
                "duration (a still's length) and hold_end (seconds on the last frame)."
            )
        for key in ("duration", "hold_end"):
            value = item.get(key)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= 60
            ):
                raise McpError(f"segments[{index}].{key} is seconds between 0 and 60.")
        out.append(dict(item))
    return out


def compose(
    box: Any,
    segments: Any,
    *,
    output: Optional[str],
    fps: Optional[float],
    transition: str,
    transition_s: float,
    size: Optional[Sequence[int]],
    audio: Optional[str],
    quality: str,
    repeat: int,
    overwrite: bool,
    sheet_frames: int,
) -> Tuple[Dict[str, Any], bytes, str]:
    """Join the scenes; returns ``(data, contact sheet PNG, headline)``."""
    caps = box.limits.animation
    started = clock.monotonic()
    notes: List[str] = []
    if transition not in TRANSITIONS:
        raise McpError(f"transition must be one of: {', '.join(TRANSITIONS)}.")
    if isinstance(transition_s, bool) or not isinstance(transition_s, (int, float)):
        raise McpError("transition_s is a number of seconds.")
    if not 0 <= transition_s <= 5:
        raise McpError("transition_s must be between 0 and 5 seconds.")
    if fps is not None and (
        isinstance(fps, bool) or not isinstance(fps, (int, float)) or not 1 <= fps <= 60
    ):
        raise McpError("fps must be between 1 and 60.")
    if size is not None:
        if (
            not isinstance(size, (list, tuple))
            or len(size) != 2
            or not all(isinstance(v, int) and 64 <= v <= 3840 for v in size)
        ):
            raise McpError("size is [width, height] in pixels, each between 64 and 3840.")
    if quality not in ("draft", "good", "high"):
        raise McpError("quality must be 'draft', 'good' or 'high'.")
    items = _parse_segments(segments)

    # -- the output is checked before anything is rendered ------------------------------------
    target = fmt = None
    if output is not None:
        suffix = Path(str(output)).suffix.lower()
        if suffix not in (".mp4", ".webm", ".gif", ".apng"):
            raise McpError("A composed video is written as .mp4, .webm, .gif or .apng.")
        fmt = suffix[1:]
        problem = motion.available_formats().get(fmt)
        if problem:
            raise McpError(f"Cannot write {fmt}: {problem}. Use .gif instead.")
        target = box.sandbox.write_path(
            str(output), overwrite=bool(overwrite), extensions=motion.ANIMATION_EXTENSIONS
        )
    sound = None
    if audio is not None:
        if fmt not in ("mp4", "webm"):
            raise McpError("A soundtrack needs output='name.mp4' (or .webm).")
        sound = box.sandbox.read_path(str(audio), extensions=AUDIO_EXTENSIONS)

    # -- every scene's frames --------------------------------------------------------------
    # Kept frames are taken first: rendering a changed scene may push another scene's frames
    # out of the cache, and the references held here keep them for this video.
    entries = [box._figure(item["figure_id"]) for item in items]
    kept = {e.figure_id: box._cached_frames(e) for e in entries if e.animation}
    scenes: List[_Scene] = []
    for item, entry in zip(items, entries):
        if entry.animation:
            if item.get("duration") is not None:
                raise McpError(
                    f"{entry.figure_id} is animated: its length is its animation's. Use hold_end "
                    "to stay on its last frame longer (duration is for a still)."
                )
            if kept.get(entry.figure_id) is not None:
                frames, rate = kept[entry.figure_id]  # type: ignore[misc]
                source = "cached"
            else:
                frames, rate, source = box._frames_for(entry, started, caps.max_video_seconds)
                kept[entry.figure_id] = (frames, rate)
        else:
            if item.get("duration") is None:
                raise McpError(
                    f"{entry.figure_id} has no animation: animate it first, or give it a duration "
                    f"to show it as a still: {{'figure_id': '{entry.figure_id}', 'duration': 3}}."
                )
            frames, rate, source = [box._still_frame(entry)], 1.0, "still"
        scenes.append(_Scene(entry.figure_id, frames, rate, source, page=_page(frames[0])))

    moving = [s.fps for s in scenes if s.source != "still"]
    rate_out = float(fps) if fps is not None else (max(moving) if moving else 30.0)
    if size is None:
        # the first scene's width, in the shape its figure was created with (1280x720 -> 16:9):
        # frames are cropped to what they show, which would make an odd-shaped video
        first = motion._image(scenes[0].frames[0])
        width = first.width + first.width % 2
        height = round(width * entries[0].height / entries[0].width)
        out_size = (width, height + height % 2)
    else:
        out_size = (int(size[0]), int(size[1]))
    for scene in scenes:
        shape = motion._image(scene.frames[0]).size
        # a frame is cropped to what it shows, so shapes differ a little; past ~6% the bands
        # of page colour around a letterboxed scene start to show
        if abs(shape[0] / shape[1] - out_size[0] / out_size[1]) > 0.06 * out_size[0] / out_size[1]:
            notes.append(
                f"{scene.figure_id} is {shape[0]}x{shape[1]} px, another shape than the video's "
                f"{out_size[0]}x{out_size[1]}: it is letterboxed in its page colour. Give every "
                "scene the same create_figure(width, height), or set size=[w, h]."
            )

    # -- the plan: which source frame each output frame shows -------------------------------
    for item, scene in zip(items, scenes):
        if scene.source == "still":
            scene.count = max(1, round(float(item["duration"]) * rate_out))
        else:
            scene.count = max(1, round(len(scene.frames) * rate_out / scene.fps))
        scene.count += round(float(item.get("hold_end") or 0) * rate_out)
    k = round(float(transition_s) * rate_out) if transition != "cut" else 0
    plan: List[Tuple[Any, ...]] = []
    for index, scene in enumerate(scenes):
        steps: List[Tuple[Any, ...]] = [("frame", index, j) for j in range(scene.count)]
        if transition == "fade" and k:
            half = max(1, min(k // 2, scene.count // 3))
            if index > 0:
                for j in range(half):
                    steps[j] = ("fade", index, j, (j + 1) / (half + 1))
            if index < len(scenes) - 1:
                for j in range(half):
                    at = scene.count - half + j
                    steps[at] = ("fade", index, at, 1 - (j + 1) / (half + 1))
        if transition == "crossfade" and k and index > 0:
            prev = scenes[index - 1]
            overlap = max(1, min(k, scene.count // 2, prev.count // 2))
            tail = plan[-overlap:]
            del plan[-overlap:]
            for j in range(overlap):
                last = tail[j]
                weight = (j + 1) / (overlap + 1)
                steps[j] = ("blend", last[1], last[2], index, j, weight)
        scene.first = len(plan)
        plan.extend(steps)

    total = len(plan)
    if total > caps.max_video_frames:
        raise McpError(
            f"The video would be {total} frames ({total / rate_out:.0f} s at {rate_out:g} fps); "
            f"the limit is {caps.max_video_frames}. Lower fps or shorten scenes."
        )
    if fmt == "gif" and total > caps.max_video_gif_frames:
        raise McpError(
            f"A {total}-frame GIF is too big ({caps.max_video_gif_frames} at most): write .mp4, "
            "or lower fps."
        )
    video = _Frames(plan, scenes, out_size, rate_out)

    written: Dict[str, Any] = {}
    if target is not None and fmt is not None:
        began = clock.monotonic()
        page = scenes[0].page
        info = motion.encode(
            video, rate_out, target, fmt, repeat=repeat, background=page, quality=quality,
            audio=sound,
        )  # fmt: skip
        written = {"path": str(target), "bytes": target.stat().st_size, **info}
        written["encode_s"] = round(clock.monotonic() - began, 2)
    else:
        notes.append(
            "No file written (no output given): the contact sheet is the preview. Pass "
            "output='video.mp4' to write it."
        )

    # -- the sheet: the start, middle and end of each scene ------------------------------------
    per_scene = max(1, min(3, sheet_frames // max(1, len(scenes))))
    indices: List[int] = []
    labels: List[str] = []
    for number, scene in enumerate(scenes, start=1):
        span = [scene.first + round(f * (scene.count - 1)) for f in _fractions(per_scene)]
        for at in span:
            at = min(total - 1, at)
            if at not in indices:
                indices.append(at)
                labels.append(f"scene {number} ({scene.figure_id})  {at / rate_out:.1f}s")
    columns = min(4, max(1, len(indices)))
    sheet = motion.contact_sheet(video, indices, labels, columns=columns, background=(40, 40, 40))

    described = [
        {
            "scene": number,
            "figure_id": scene.figure_id,
            "start_s": round(scene.first / rate_out, 3),
            "end_s": round((scene.first + scene.count) / rate_out, 3),
            "frames": scene.count,
            "source": scene.source,
        }
        for number, scene in enumerate(scenes, start=1)
    ]
    data: Dict[str, Any] = {
        "frames": total,
        "fps": rate_out,
        "duration_s": round(total / rate_out, 3),
        "size_px": list(out_size),
        "transition": transition,
        "scenes": described,
        "elapsed_s": round(clock.monotonic() - started, 2),
    }
    if written:
        data["file"] = written
    if notes:
        data["notes"] = notes
    headline = (
        f"Composed {len(scenes)} scene{'s' if len(scenes) != 1 else ''} into {total} frames "
        f"({total / rate_out:.1f} s at {rate_out:g} fps, {out_size[0]}x{out_size[1]} px)"
        + (f" -> {written['path']} ({written['bytes'] / 1024:.0f} KiB)." if written else ".")
    )
    return data, sheet, headline


def _fractions(count: int) -> List[float]:
    if count == 1:
        return [0.5]
    return [i / (count - 1) for i in range(count)] if count > 2 else [0.15, 0.85]
