"""Turning a sequence of RGB frames into a file on disk.

This module is the *output* half of GLPlot's animation support and it deliberately knows
nothing about GLPlot. It takes pixels — an iterable of ``(H, W, 3)`` uint8 arrays, or a
callable that renders frame *i* on demand — plus a frame rate and a path, and writes a
PNG sequence, an animated GIF or an MP4. :mod:`glplot.animation` owns the matplotlib-shaped
API and calls in here; the GUI, the gallery runner or a user script can call in here just
as well with frames from any source at all.

Why the split? Because the two halves fail for entirely different reasons. A matplotlib
``FuncAnimation`` fails because a user callback raised; an encoder fails because ffmpeg is
not installed. Keeping them apart means the "install imageio-ffmpeg" message lives in one
place instead of being reinvented inside every writer class.

The dependency ladder
---------------------
Three formats, three very different dependency stories, and the ordering below is the
whole design:

1. **PNG sequence** — :func:`write_png_sequence`. numpy and the standard library, nothing
   else. This is the floor: it *cannot* be unavailable, which is what makes it a safe
   fallback target for everything above it.
2. **Animated GIF** — :func:`write_gif`. Needs Pillow. matplotlib depends on Pillow, and
   GLPlot depends on matplotlib, so in practice this is always available too — but it is
   still probed at call time rather than assumed.
3. **MP4** — :func:`write_mp4`. Needs a real video encoder: ``imageio`` + ``imageio-ffmpeg``
   (preferred, because it is a pip install and needs no system package), else a plain
   ``ffmpeg`` binary on ``PATH``, else nothing and we raise.

**No optional dependency is imported at module import time.** Importing
``glplot.utils.anim_export`` on a machine with no imageio and no ffmpeg must succeed and
must keep succeeding right up until someone actually asks for an MP4. Every probe below is
a function, not a module-level ``try: import``, and none of them memoise: they re-check on
every call. That is a deliberate trade of a few microseconds (an already-imported module is
a dict lookup, and :func:`shutil.which` is a handful of ``stat`` calls) for the ability to
monkeypatch them in tests and for never baking a stale "unavailable" into the module the
first time something is asked.

Rejected: raising ``ImportError`` from the top of the module behind a feature flag. It
turns "you cannot make MP4s" into "you cannot use GLPlot's animation module at all", which
is a much worse failure for the 90% of users who only ever wanted a GIF.

The even-dimensions trap
------------------------
H.264 cannot encode a frame whose width or height is odd — libx264 refuses with
``width not divisible by 2``. Worse, ``imageio`` does not propagate that failure: it
returns normally and leaves a **zero-byte** ``.mp4`` behind. So :func:`write_mp4` crops
each frame to even dimensions before encoding (losing at most one pixel row and one pixel
column) *and* verifies afterwards that the file it wrote is non-empty. Both halves are
load-bearing; neither alone is enough.

Rejected: padding to even instead of cropping. Padding invents a black line along one edge
of every frame, which is visible; cropping one row off a 961-pixel-tall figure is not.
Rejected: ``macro_block_size=16`` (imageio's default), which silently *rescales* the whole
movie to a multiple of 16 and makes the output a different size than the frames the caller
handed us.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Union

import numpy as np

#: Frames per second used when a caller does not say. 30 rather than matplotlib's 5,
#: because GLPlot animations are GPU-rendered scenes people expect to look smooth, not
#: flip-books. The matplotlib-facing writers in :mod:`glplot.animation` keep mpl's 5 for
#: signature parity; this default only applies to direct calls into this module.
DEFAULT_FPS: float = 30.0

#: ``str.format`` template for the individual files of a PNG sequence. ``index`` is
#: zero-based and zero-padded to four digits so a lexicographic sort is a chronological
#: sort — that is what ``ffmpeg -i frame_%04d.png`` and every image viewer's "next file"
#: button assume.
PNG_SEQUENCE_TEMPLATE: str = "{stem}_{index:04d}.png"

#: Extensions routed to :func:`write_gif`.
GIF_EXTENSIONS: frozenset = frozenset({".gif"})

#: Extensions routed to :func:`write_mp4`. ``.mp4`` and ``.m4v`` are h264-in-MP4;  the rest
#: are containers ffmpeg picks a codec for from the extension alone. GLPlot does not
#: promise every one of these works on every ffmpeg build — it promises to hand the request
#: to the encoder rather than reject it here.
VIDEO_EXTENSIONS: frozenset = frozenset({".mp4", ".m4v", ".mov", ".avi", ".webm", ".mkv"})

#: Extensions routed to :func:`write_png_sequence`.
PNG_EXTENSIONS: frozenset = frozenset({".png"})

#: Default video codec. h264 is the only codec that plays in every browser, every phone and
#: QuickTime without the user installing anything, which matters because the usual next step
#: after saving an MP4 is emailing it to someone.
DEFAULT_CODEC: str = "libx264"

#: Pixel format forced for h264. libx264's own default is ``yuv444p``, which Safari,
#: QuickTime and most hardware decoders refuse to play — the file opens, the picture is
#: black. This is the same fix matplotlib applies in ``FFMpegBase.output_args``.
H264_PIX_FMT: str = "yuv420p"

_MISSING_VIDEO_HINT = (
    "install a video encoder with 'pip install imageio imageio-ffmpeg' (no system "
    "package needed), or put an 'ffmpeg' binary on PATH "
    "('brew install ffmpeg' on macOS, 'apt install ffmpeg' on Debian/Ubuntu)"
)

FrameSource = Union[Iterable[np.ndarray], Callable[[int], np.ndarray]]


# ----------------------------------------------------------------------------------
# Capability probes -- all of them lazy, none of them raising
# ----------------------------------------------------------------------------------


def pillow_available() -> bool:
    """Is Pillow importable? Required for GIF output and for PNG writing quality."""
    try:
        import PIL.Image  # noqa: F401
    except Exception:
        return False
    return True


def imageio_available() -> bool:
    """Is the ``imageio`` + ``imageio-ffmpeg`` pair importable?

    Both halves are checked. ``imageio`` alone installs happily and then fails at
    ``get_writer(format="FFMPEG")`` time with a plugin error that mentions neither package
    by name, which is exactly the kind of message this module exists to prevent.
    """
    try:
        import imageio  # noqa: F401
        import imageio_ffmpeg  # noqa: F401
    except Exception:
        return False
    return True


def ffmpeg_binary() -> Optional[str]:
    """Absolute path to an ``ffmpeg`` executable on ``PATH``, or ``None``.

    Checked with :func:`shutil.which` rather than by running ``ffmpeg -version``: spawning
    a process to answer "is this feature available" costs ~50 ms and is asked repeatedly
    (once per :meth:`~glplot.animation.FFMpegWriter.isAvailable` call, and the writer
    registry calls that for every writer every time you list it).
    """
    return shutil.which("ffmpeg")


def video_backend() -> Optional[str]:
    """Which MP4 backend this machine has: ``"imageio"``, ``"ffmpeg"``, or ``None``.

    The preference order is not arbitrary. ``imageio-ffmpeg`` ships its own statically
    linked binary, so it works identically on a colleague's laptop and in CI; a ``PATH``
    ffmpeg is whatever the machine happens to have, which may be a decode-only build
    without libx264.
    """
    if imageio_available():
        return "imageio"
    if ffmpeg_binary() is not None:
        return "ffmpeg"
    return None


def available_formats() -> List[str]:
    """Extensions this machine can actually write, best first.

    Useful for a GUI that wants to populate a "save as" dropdown without offering the user
    a format that will fail three seconds later.
    """
    formats = []
    if video_backend() is not None:
        formats.extend(sorted(VIDEO_EXTENSIONS))
    if pillow_available():
        formats.extend(sorted(GIF_EXTENSIONS))
    formats.extend(sorted(PNG_EXTENSIONS))
    return formats


# ----------------------------------------------------------------------------------
# Frame normalisation
# ----------------------------------------------------------------------------------


def normalize_frame(frame: object, index: int = 0) -> np.ndarray:
    """Coerce one frame to a C-contiguous ``(H, W, 3)`` uint8 RGB array.

    Accepts what real render pipelines actually produce:

    * ``(H, W, 3)`` uint8 — returned as-is (made contiguous if it is a view or a flip,
      which matters because ``ndarray.tobytes()`` on a ``np.flipud`` result is not the
      byte order an encoder's raw-video pipe expects).
    * ``(H, W, 4)`` RGBA — alpha dropped. Video has no alpha channel and GIF's is
      one-bit; compositing against an assumed background would guess at a colour the
      caller never chose.
    * ``(H, W)`` greyscale — broadcast to three channels.
    * floating point — assumed to be in ``[0, 1]`` (matplotlib's convention, and what
      ``plt.imsave`` consumes), clipped and scaled to ``[0, 255]``.

    ``index`` only ever appears in error messages; it is what turns "bad frame shape" into
    a message the caller can act on.
    """
    array = np.asarray(frame)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(
            f"Frame {index} has shape {array.shape}: expected a 2D greyscale image or a "
            f"3D (height, width, channels) image. Pass an array of pixels, not a figure."
        )

    channels = array.shape[2]
    if channels == 1:
        array = np.repeat(array, 3, axis=2)
    elif channels == 4:
        array = array[:, :, :3]
    elif channels != 3:
        raise ValueError(
            f"Frame {index} has {channels} channels: expected 1 (grey), 3 (RGB) or " f"4 (RGBA)."
        )

    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array, 0.0, 1.0) * 255.0
        array = array.astype(np.uint8)

    return np.ascontiguousarray(array)


def iter_frames(frames: FrameSource, count: Optional[int] = None) -> Iterator[np.ndarray]:
    """Normalise any accepted frame source into a stream of uint8 RGB arrays.

    ``frames`` is either an iterable of images or a callable ``render(i) -> image``. The
    callable form exists so a caller can stream a million-line scene without holding every
    rendered frame in memory at once — but it has no natural end, so ``count`` is required
    with it and rejected as ambiguous without it.
    """
    if callable(frames):
        if count is None:
            raise ValueError(
                "A callable frame source needs 'count': a render function has no length, "
                "so there is no way to know when to stop. Pass count=<number of frames>, "
                "or pass a list/generator of frames instead."
            )
        for index in range(int(count)):
            yield normalize_frame(frames(index), index)
        return

    for index, frame in enumerate(frames):
        if count is not None and index >= int(count):
            return
        yield normalize_frame(frame, index)


def _require_frames(stream: Iterator[np.ndarray], path: object) -> np.ndarray:
    """Pull the first frame, or explain that there were none.

    An empty animation is always a caller bug — a ``frames=0``, a generator that returned
    without yielding, an update function that raised on frame 0 and was swallowed. Left
    unchecked it surfaces as ``IndexError: list index out of range`` from inside Pillow.
    """
    try:
        return next(stream)
    except StopIteration:
        raise ValueError(
            f"Cannot write {path}: the frame source produced no frames. Check that the "
            f"animation has a non-zero frame count and that the update function returns "
            f"without raising on the first frame."
        ) from None


def _prepare_output(path: Union[str, os.PathLike]) -> Path:
    """Resolve the output path and make sure its directory exists.

    Creating the parent is a deliberate convenience: an animation save is a long operation
    and failing it at the very end because ``out/`` did not exist wastes every frame that
    was already rendered.
    """
    resolved = Path(path).expanduser()
    parent = resolved.parent
    if str(parent) and not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot write {resolved}: the directory {parent} does not exist and "
                f"could not be created ({exc}). Create it first, or save somewhere "
                f"writable."
            ) from exc
    return resolved


def _verify_nonempty(path: Path, what: str) -> None:
    """Fail loudly if an encoder claimed success but left nothing behind.

    This is not paranoia. ``imageio`` returns normally from ``writer.close()`` after
    libx264 has refused every single frame, leaving a zero-byte MP4 on disk; the caller's
    next clue is a video player saying "cannot open file". Checking the size turns a silent
    corruption into a diagnosable error at the point it happened.
    """
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(
            f"Writing {what} to {path} produced an empty file. The encoder accepted the "
            f"frames but wrote nothing, which usually means it rejected the frame size or "
            f"the codec. Try a PNG sequence or a GIF to confirm the frames themselves are "
            f"good."
        )


# ----------------------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------------------


def write_png_sequence(
    frames: FrameSource,
    path: Union[str, os.PathLike],
    *,
    fps: float = DEFAULT_FPS,
    count: Optional[int] = None,
) -> List[str]:
    """Write each frame as its own PNG next to *path*. Always available.

    ``path`` names the *sequence*, not a file: ``frames.png`` produces ``frames_0000.png``,
    ``frames_0001.png`` and so on in the same directory (see :data:`PNG_SEQUENCE_TEMPLATE`).
    Any extension on ``path`` is ignored; ``.png`` is always used.

    ``fps`` is accepted and **not** written anywhere — a PNG sequence has no timing
    metadata. It is in the signature so this function is a drop-in for the other two
    writers in a dispatch table, and so a caller degrading from MP4 does not have to
    special-case the argument list. The frame rate the caller intended is preserved in the
    returned filenames only in the sense that they are in order.

    Returns the list of paths written, in frame order.
    """
    resolved = _prepare_output(path)
    stem = resolved.stem
    directory = resolved.parent if str(resolved.parent) else Path(".")

    stream = iter_frames(frames, count)
    first = _require_frames(stream, resolved)

    written: List[str] = []
    save = _png_saver()
    for index, frame in enumerate(_chain_first(first, stream)):
        target = directory / PNG_SEQUENCE_TEMPLATE.format(stem=stem, index=index)
        save(frame, target)
        written.append(str(target))

    _verify_nonempty(Path(written[0]), "a PNG frame sequence")
    return written


def _png_saver() -> Callable[[np.ndarray, Path], None]:
    """Pick the PNG encoder once, rather than re-deciding inside the frame loop.

    Pillow if it is there, matplotlib's ``imsave`` if it somehow is not. Both are pure-Python
    entry points to the same libpng; the only reason to prefer Pillow is that it does not
    drag in pyplot's global state for what should be a stateless byte-writing operation.
    """
    if pillow_available():
        from PIL import Image

        def save_pillow(frame: np.ndarray, target: Path) -> None:
            Image.fromarray(frame, mode="RGB").save(target)

        return save_pillow

    def save_mpl(frame: np.ndarray, target: Path) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        plt.imsave(str(target), frame)

    return save_mpl


def _chain_first(first: np.ndarray, rest: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
    """Put a peeked-at frame back on the front of its own stream."""
    yield first
    yield from rest


def write_gif(
    frames: FrameSource,
    path: Union[str, os.PathLike],
    *,
    fps: float = DEFAULT_FPS,
    loop: int = 0,
    count: Optional[int] = None,
) -> str:
    """Write an animated GIF via Pillow. Requires Pillow.

    ``loop=0`` means "loop forever", which is GIF's own convention and not a typo; any
    other value is a repeat count.

    **Frame rate is quantised.** GIF stores per-frame delay as an integer number of
    hundredths of a second, so 30 fps (33.33 ms) is stored as 33 ms and plays at 30.3 fps,
    and any fps above 100 collapses to the same 10 ms floor. Nothing can be done about that
    at this layer — it is the file format. Ask for MP4 if the timing has to be exact.

    Unlike the MP4 path this holds every frame in memory: Pillow's ``save_all`` needs the
    whole list to build the shared palette. A thousand 1536x960 frames is roughly 4 GB of
    ``Image`` objects, so long animations should go to MP4 or a PNG sequence.
    """
    if not pillow_available():
        raise RuntimeError(
            f"Cannot write a GIF to {path}: Pillow is not installed. Install it with "
            f"'pip install Pillow'. A PNG frame sequence needs no extra dependency — save "
            f"to a '.png' path instead to get one file per frame."
        )

    from PIL import Image

    resolved = _prepare_output(path)
    stream = iter_frames(frames, count)
    first = _require_frames(stream, resolved)

    images = [Image.fromarray(frame, mode="RGB") for frame in _chain_first(first, stream)]
    duration_ms = max(int(round(1000.0 / max(float(fps), 1e-6))), 1)
    images[0].save(
        resolved,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=int(loop),
        # Pillow defaults `optimize` to True whenever no explicit palette is given (see
        # GifImagePlugin._save), which makes _write_multiple_frames diff every frame
        # against the last with a full-resolution ImageMath mask to find a transparent
        # fill for unchanged pixels -- a pure-Python-driven pixel op per frame pair, on
        # top of the RGB->P quantization every frame already needs. For a real animation
        # (dozens of full-figure frames), that is easily a bottleneck rather than a
        # rounding error, and was slow enough on a cold Windows CI runner to blow the
        # suite's per-test timeout with nothing actually stuck. It only ever changes file
        # size, never the decoded pixels, so turning it off trades a somewhat larger GIF
        # for a bounded, predictable encode.
        optimize=False,
    )
    _verify_nonempty(resolved, "a GIF")
    return str(resolved)


class VideoStream:
    """A push-based video encoder: construct, :meth:`append` frames, :meth:`close`.

    :func:`write_mp4` is the pull-based convenience wrapper around this, and is what most
    callers want. This class exists for the other shape of caller — matplotlib's
    ``MovieWriter`` protocol, where frames arrive one ``grab_frame()`` at a time from code
    that is driving the loop itself and cannot be turned inside out into an iterator.

    Nothing is opened until the first :meth:`append`. That is not laziness for its own
    sake: an encoder must be told the frame size up front, and the frame size is not known
    until a frame exists. GLPlot's own figures make this concrete — the headless preview
    renderer sizes its output from the figure's inches-and-dpi, not from
    ``fig.fb_width``, so guessing at ``setup()`` time would guess wrong.

    Memory is flat in the number of frames: each one is encoded and dropped. That is the
    entire reason this is not simply "collect a list and call :func:`write_mp4`" — a
    thousand frames of a 1536x960 figure is 4.2 GB of uint8 held at once, and animations
    that long are exactly the ones people want as MP4 rather than GIF.
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        fps: float = DEFAULT_FPS,
        *,
        codec: Optional[str] = None,
        bitrate: Optional[int] = None,
        extra_args: Optional[Sequence[str]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self.backend = video_backend()
        if self.backend is None:
            raise RuntimeError(
                f"Cannot write a video to {path}: no video encoder is available. To fix, "
                f"{_MISSING_VIDEO_HINT}. Neither is needed for the other formats — save to "
                f"a '.gif' path for an animated GIF, or a '.png' path for a PNG frame "
                f"sequence."
            )
        self.path = _prepare_output(path)
        self.fps = float(fps)
        self.codec = codec or DEFAULT_CODEC
        self.bitrate = bitrate
        self.extra_args = list(extra_args or ())
        self.metadata = dict(metadata or {})
        self.frame_count = 0
        self._shape: Optional[tuple] = None
        self._sink: object = None
        self._process: Optional[subprocess.Popen] = None

    # -- context manager ------------------------------------------------------------

    def __enter__(self) -> "VideoStream":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- frames ---------------------------------------------------------------------

    def append(self, frame: object) -> None:
        """Normalise, crop to even dimensions, and encode one frame."""
        image = _crop_even(normalize_frame(frame, self.frame_count), self.frame_count)
        if self._shape is None:
            self._shape = image.shape
            self._open(image.shape)
        elif image.shape != self._shape:
            raise ValueError(
                f"Frame {self.frame_count} is {image.shape[1]}x{image.shape[0]} pixels but "
                f"frame 0 was {self._shape[1]}x{self._shape[0]}. Every frame of a video "
                f"must be the same size — resize the figure before starting the animation, "
                f"not during it."
            )
        self._write(image)
        self.frame_count += 1

    def close(self) -> None:
        """Flush and finalise. Safe to call twice; a stream with no frames writes nothing."""
        if self._sink is not None:
            self._sink.close()  # type: ignore[attr-defined]
            self._sink = None
        if self._process is not None:
            self._close_process()
        if self.frame_count:
            _verify_nonempty(self.path, "a video")

    # -- backends -------------------------------------------------------------------

    def _open(self, shape: tuple) -> None:
        """Open the encoder, falling back from imageio to a raw binary if it refuses.

        imageio's FFMPEG plugin dispatches on the *filename extension*, not on the format
        argument, and it only recognises a handful: a perfectly ordinary ``.m4v`` or
        ``.mkv`` target fails with "`FFMPEG` can not handle the given uri" even though the
        ffmpeg it wraps would encode it happily. A bare ``ffmpeg`` binary has no such list,
        so when one is on PATH it is a strictly better encoder for those containers.

        The fallback is here rather than in :meth:`__init__` because the failure is not
        predictable from the extension alone — imageio's supported set varies by version,
        and hardcoding a copy of it here would go stale.
        """
        if self.backend == "imageio":
            try:
                self._open_imageio()
                return
            except Exception as exc:
                if ffmpeg_binary() is None:
                    raise RuntimeError(
                        f"Cannot write a video to {self.path}: imageio refused this file "
                        f"type ({exc}). Try a '.mp4' path, which imageio always accepts, or "
                        f"install a system ffmpeg ('brew install ffmpeg' / 'apt install "
                        f"ffmpeg'), which handles every container."
                    ) from exc
                self.backend = "ffmpeg"
        self._open_ffmpeg(shape)

    def _open_imageio(self) -> None:
        """Open imageio's FFMPEG plugin.

        ``macro_block_size=1`` disables imageio's own resize-to-a-multiple-of-16 behaviour.
        We have already guaranteed even dimensions, which is the real encoder constraint;
        imageio's default would additionally rescale a 1536x962 figure to 1536x960 *by
        interpolation*, quietly resampling every frame of the movie.
        """
        try:
            import imageio.v2 as imageio
        except Exception:  # pragma: no cover - imageio >= 2.9 always has .v2
            import imageio  # type: ignore[no-redef]

        kwargs: dict = {
            "format": "FFMPEG",
            "mode": "I",
            "fps": self.fps,
            "codec": self.codec,
            "macro_block_size": 1,
            "pixelformat": H264_PIX_FMT,
        }
        if self.bitrate is not None and int(self.bitrate) > 0:
            # imageio wants bits per second; matplotlib and ffmpeg both speak kilobits.
            kwargs["bitrate"] = int(self.bitrate) * 1000
        output_params = list(self.extra_args)
        for key, value in self.metadata.items():
            output_params.extend(["-metadata", f"{key}={value}"])
        if output_params:
            kwargs["output_params"] = output_params
        self._sink = imageio.get_writer(str(self.path), **kwargs)

    def _open_ffmpeg(self, shape: tuple) -> None:
        """Spawn ``ffmpeg`` reading raw RGB from a pipe.

        ``-loglevel error`` is not cosmetic. ffmpeg's default per-frame progress goes to
        ``stderr``; with ``stderr=PIPE`` and nobody draining it, a long animation deadlocks
        the moment the OS pipe buffer fills, and the symptom is a hung save with no output.
        """
        binary = ffmpeg_binary()
        if binary is None:  # pragma: no cover - guarded in __init__
            raise RuntimeError(f"Cannot write a video to {self.path}: {_MISSING_VIDEO_HINT}.")
        height, width = shape[0], shape[1]
        command = [
            binary,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-framerate",
            str(self.fps),
            "-i",
            "pipe:",
            "-vcodec",
            self.codec,
            "-pix_fmt",
            H264_PIX_FMT,
        ]
        if self.bitrate is not None and int(self.bitrate) > 0:
            command.extend(["-b:v", f"{int(self.bitrate)}k"])
        for key, value in self.metadata.items():
            command.extend(["-metadata", f"{key}={value}"])
        command.extend(self.extra_args)
        command.append(str(self.path))
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def _write(self, image: np.ndarray) -> None:
        if self._sink is not None:
            self._sink.append_data(image)  # type: ignore[attr-defined]
            return
        assert self._process is not None and self._process.stdin is not None
        try:
            self._process.stdin.write(image.tobytes())
        except BrokenPipeError:
            # ffmpeg died early. Its own stderr, surfaced by _close_process, is the useful
            # message; swallowing here avoids burying it under a pipe error.
            pass

    def _close_process(self) -> None:
        process = self._process
        self._process = None
        assert process is not None
        if process.stdin is not None:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        stderr = process.stderr.read() if process.stderr else b""
        returncode = process.wait()
        if returncode != 0:
            detail = stderr.decode("utf-8", "replace").strip() or f"exit code {returncode}"
            raise RuntimeError(
                f"Cannot write a video to {self.path}: ffmpeg failed. ffmpeg said: "
                f"{detail}. If the codec is unsupported by this ffmpeg build, try "
                f"'pip install imageio imageio-ffmpeg', which ships its own encoder."
            )


def _crop_even(frame: np.ndarray, index: int) -> np.ndarray:
    """Round a frame's width and height down to even. See the module docstring."""
    height, width = frame.shape[0] & ~1, frame.shape[1] & ~1
    if height == 0 or width == 0:
        raise ValueError(
            f"Frame {index} is {frame.shape[1]}x{frame.shape[0]} pixels, which is too small "
            f"to encode as video: h264 needs at least 2x2 after rounding down to even "
            f"dimensions."
        )
    if (height, width) == frame.shape[:2]:
        return frame
    return np.ascontiguousarray(frame[:height, :width])


def write_mp4(
    frames: FrameSource,
    path: Union[str, os.PathLike],
    *,
    fps: float = DEFAULT_FPS,
    codec: Optional[str] = None,
    bitrate: Optional[int] = None,
    count: Optional[int] = None,
    extra_args: Optional[Sequence[str]] = None,
    metadata: Optional[dict] = None,
) -> str:
    """Write an MP4 (or any container in :data:`VIDEO_EXTENSIONS`). Requires an encoder.

    Backend selection is :func:`video_backend`'s: ``imageio``/``imageio-ffmpeg`` if
    importable, else an ``ffmpeg`` binary on ``PATH``, else :class:`RuntimeError` naming
    both installation routes.

    ``bitrate`` is in **kilobits per second**, matching matplotlib's
    ``rcParams['animation.bitrate']`` and ffmpeg's ``-b:v 2000k``. ``-1`` or ``None`` lets
    the encoder choose, which is almost always the right answer for h264 — its rate control
    is better at picking a bitrate than a human guessing one.

    Every frame is cropped to even width and height first; see the module docstring for why
    that is not optional.
    """
    resolved = _prepare_output(path)
    # Probed before a single frame is pulled. A frame source may be a generator or an
    # expensive GPU render, and finding out afterwards that there is no encoder would both
    # waste that work and leave a one-shot iterator half-consumed for any fallback path.
    if video_backend() is None:
        raise RuntimeError(
            f"Cannot write a video to {resolved}: no video encoder is available. To fix, "
            f"{_MISSING_VIDEO_HINT}. Neither is needed for the other formats — save to a "
            f"'.gif' path for an animated GIF, or a '.png' path for a PNG frame sequence."
        )

    stream = iter_frames(frames, count)
    first = _require_frames(stream, resolved)

    with VideoStream(
        resolved,
        fps,
        codec=codec,
        bitrate=bitrate,
        extra_args=extra_args,
        metadata=metadata,
    ) as video:
        for frame in _chain_first(first, stream):
            video.append(frame)

    _verify_nonempty(resolved, "a video")
    return str(resolved)


def write_animation(
    frames: FrameSource,
    path: Union[str, os.PathLike],
    *,
    fps: float = DEFAULT_FPS,
    count: Optional[int] = None,
    fallback: bool = True,
    **kwargs: object,
) -> str:
    """Write frames to *path*, choosing the format from its extension.

    This is the one function most callers want. ``.mp4`` (and the rest of
    :data:`VIDEO_EXTENSIONS`) goes to :func:`write_mp4`, ``.gif`` to :func:`write_gif`,
    anything else to :func:`write_png_sequence`.

    ``fallback=True`` (the default) degrades down the ladder in the module docstring when a
    dependency is missing — MP4 to GIF to PNG sequence — and **warns** each time it does,
    naming the file it actually wrote. That warning is the whole point: a save that quietly
    produces a different format than the one requested is worse than a save that fails,
    because the user finds out when they try to embed the missing MP4 in a talk. Pass
    ``fallback=False`` to get the :class:`RuntimeError` instead.

    Returns the path actually written — which is *not* necessarily ``path``, and which is
    the first PNG of the sequence when it degrades that far. Callers that care where the
    output landed must use the return value rather than assuming.

    Degradation is decided by *probing* the ladder before anything is rendered, never by
    catching a failure and retrying. ``frames`` may legitimately be a one-shot generator or
    a callable doing an expensive GPU render per frame; a retry would replay a source that
    cannot be replayed and silently produce a truncated movie.
    """
    resolved = _prepare_output(path)
    suffix = resolved.suffix.lower()

    if suffix in VIDEO_EXTENSIONS:
        if video_backend() is not None:
            return write_mp4(  # type: ignore[arg-type]
                frames, resolved, fps=fps, count=count, **kwargs
            )
        if not fallback:
            # Raises the RuntimeError naming both installation routes.
            return write_mp4(  # type: ignore[arg-type]
                frames, resolved, fps=fps, count=count, **kwargs
            )
        target = resolved.with_suffix(".gif" if pillow_available() else ".png")
        warnings.warn(
            f"No video encoder available, so {resolved.name} could not be written; wrote "
            f"{target.name} instead. To get the video next time, {_MISSING_VIDEO_HINT}.",
            stacklevel=2,
        )
        suffix = target.suffix
        resolved = target

    if suffix in GIF_EXTENSIONS:
        if pillow_available() or not fallback:
            # Without Pillow this raises the RuntimeError naming 'pip install Pillow'.
            return write_gif(frames, resolved, fps=fps, count=count)
        png_path = resolved.with_suffix(".png")
        warnings.warn(
            f"Pillow is not installed, so {resolved.name} could not be written; wrote a "
            f"PNG frame sequence based on {png_path.name} instead. To get the GIF next "
            f"time, 'pip install Pillow'.",
            stacklevel=2,
        )
        return write_png_sequence(frames, png_path, fps=fps, count=count)[0]

    return write_png_sequence(frames, resolved, fps=fps, count=count)[0]
