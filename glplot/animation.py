"""A drop-in replacement for :mod:`matplotlib.animation`, backed by GLPlot figures.

The promise of this module is a single line::

    -import matplotlib.animation as animation
    +import glplot.animation as animation

and nothing else in the script changes. ``FuncAnimation(fig, update, frames=200,
interval=20, blit=True)`` constructs, ``ani.save("out.mp4", fps=30)`` writes an MP4,
``writers.list()`` lists writers, ``ani.to_jshtml()`` returns a player. Every public class
here carries matplotlib's exact signature, verified against the installed matplotlib by
``tests/test_animation_api.py`` with :func:`inspect.signature` rather than by hand — if
matplotlib changes a default, that test fails and this module is wrong until it is fixed.

What ``fig`` is
---------------
In matplotlib ``fig`` is a ``Figure`` with a ``canvas``. Here it is normally a
:class:`glplot.engine.GPULinePlot`, which has neither a canvas nor an event loop of its
own that we can hook. Everything that matplotlib routes through ``fig.canvas`` — blitting,
``mpl_connect('draw_event')``, ``new_timer()`` — is therefore adapted rather than used:

* If the figure *does* have a working canvas (i.e. somebody passed a real matplotlib
  Figure), it is used, and this module behaves as a slightly slower matplotlib.
* Otherwise a null canvas stands in: connections return an id and are never fired, and the
  timer never ticks. Nothing raises ``AttributeError``, which is the point.

That dual path is deliberate. A user porting a script wants their matplotlib figures to
keep working while they move plots over one at a time, and a test suite wants to exercise
this module without a GPU.

How a frame is produced
-----------------------
:func:`figure_to_rgb` is the whole bridge, and it has two modes:

* **Live GL** — when ``fig.window`` exists, ``fig.export.savefig()`` renders the scene
  offscreen through the real pipeline at full quality.
* **Headless** — when it does not, :func:`glplot.utils.preview.render_preview` draws the
  same scene through matplotlib's Agg backend. This is the path that makes
  ``ani.save(...)`` work in CI, in a notebook, and in any script that never called
  ``show()``.

Both render *to a PNG file* and read it back, because that is the only frame-producing
interface the engine exposes; there is no "give me the pixels" entry point to call
instead. The cost is one temp-file round trip per frame, which is real but is dwarfed by
the render itself. Rejected: reimplementing ``ExportManager.savefig``'s ``glReadPixels``
here to skip the file. It would duplicate the panel/scissor/projection logic and would go
stale the first time the render pipeline changed.

**The headless path is not pixel-identical to the GL path.** ``render_preview`` is a
matplotlib re-drawing of the scene, not the GPU renderer; it approximates. An animation
saved without a window looks like a matplotlib plot of the same data, which is usually
what a user wants from CI and never quite what they want from a demo reel. Call
``fig.run()`` first if the GPU look matters.

Limitations, stated plainly
---------------------------
* **``blit`` is accepted and ignored.** Every GLPlot frame is a full scene re-render; there
  is no background to restore and no artist-level region to blit. Passing ``blit=True``
  changes nothing except that it does not raise. Output is identical either way — blitting
  is an optimisation in matplotlib, not a correctness flag — so ignoring it cannot make a
  saved animation wrong. It is recorded in ``ani._blit`` if you want to assert on it.
* **There is no live on-screen playback.** ``GPULinePlot.run()`` owns its main loop and
  exposes no per-frame user hook, so an ``Animation`` cannot drive an open window the way
  matplotlib's timer drives a GUI canvas. Constructing an animation and calling
  ``plt.show()`` shows the *scene*, not the animation. :meth:`Animation.save`,
  :meth:`Animation.to_html5_video` and :meth:`Animation.to_jshtml` all work fully; only
  interactive playback does not. :meth:`Animation.pause` / :meth:`Animation.resume` set the
  documented flag and are otherwise inert.
* **In-place array mutation is assumed.** A user callback that writes into
  ``layer.pts[:, 1]`` leaves no trace numpy can report, so after every callback this module
  marks every layer dirty rather than trying to detect what changed. Cheap, and the only
  option that is actually correct.

Everything under "Limitations" is a *documented* no-op. A silently ignored flag would be a
bug; a flag whose exact behaviour is written down is an API decision.
"""

from __future__ import annotations

import abc
import base64
import contextlib
import io
import itertools
import json
import os
import shutil
import tempfile
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .utils import anim_export

__all__ = [
    "Animation",
    "TimedAnimation",
    "FuncAnimation",
    "ArtistAnimation",
    "AbstractMovieWriter",
    "MovieWriter",
    "FileMovieWriter",
    "MovieWriterRegistry",
    "PillowWriter",
    "FFMpegBase",
    "FFMpegWriter",
    "FFMpegFileWriter",
    "ImageMagickBase",
    "ImageMagickWriter",
    "ImageMagickFileWriter",
    "PNGFileWriter",
    "HTMLWriter",
    "writers",
    "adjusted_figsize",
    "figure_to_rgb",
]

#: Reference dpi for the ``dpi`` -> GLPlot ``scale`` conversion. matplotlib sizes a figure
#: as inches x dpi and defaults ``figure.dpi`` to 100; GLPlot's ``savefig`` takes a
#: dimensionless ``scale`` multiplier instead. Dividing by this constant makes
#: ``save(..., dpi=200)`` mean "twice the default resolution" in both libraries, which is
#: the only interpretation under which a ported script produces the image it used to.
BASE_DPI: float = 100.0

#: What ``matplotlib.rcParams['animation.embed_limit']`` defaults to, in megabytes. Read
#: from live rcParams when matplotlib is importable; this is the fallback so
#: :meth:`Animation.to_html5_video` still has a limit if it is not.
DEFAULT_EMBED_LIMIT_MB: float = 20.0

#: ``interval`` is milliseconds per frame, so fps is 1000/interval. matplotlib's default
#: interval of 200 ms therefore means 5 fps, which is also every writer's default ``fps``.
#: Kept as a named constant because the 1000 appears in four places and a units mistake in
#: any of them produces an animation that is 1000x too fast rather than an exception.
MS_PER_SECOND: float = 1000.0

#: Options handed to the HTML5 ``<video>`` tag built by :meth:`Animation.to_html5_video`.
#: ``loop`` is appended separately when the animation repeats, matching matplotlib.
HTML5_VIDEO_OPTIONS: Tuple[str, ...] = ("controls", "autoplay")

VIDEO_TAG = """<video {size} {options}>
  <source type="video/mp4" src="data:video/mp4;base64,{video}">
  Your browser does not support the video tag.
</video>"""

#: Template for the non-embedded frame list of :class:`HTMLWriter`. Present for matplotlib
#: name parity; GLPlot's HTMLWriter always embeds, so this is never formatted.
INCLUDED_FRAMES = """
  for (var i=0; i<{Nframes}; i++){{
    frames[i] = "{frame_dir}/frame" + ("0000000" + i).slice(-7) +
                ".{frame_format}";
  }}
"""

STYLE_INCLUDE = """
<style>
.glplot-anim { font-family: system-ui, -apple-system, sans-serif; display: inline-block; }
.glplot-anim img { display: block; max-width: 100%; }
.glplot-anim .controls { display: flex; gap: .4em; align-items: center; padding: .4em 0; }
.glplot-anim button { cursor: pointer; padding: .15em .6em; }
.glplot-anim input[type=range] { flex: 1 1 auto; min-width: 8em; }
</style>
"""

JS_INCLUDE = """
<script>
(function() {
  var el = document.getElementById("%(id)s");
  var frames = %(frames)s;
  var interval = %(interval)f;
  var mode = "%(mode)s";
  var img = el.querySelector("img");
  var slider = el.querySelector("input[type=range]");
  var toggle = el.querySelector(".toggle");
  var i = 0, dir = 1, timer = null;
  function show(k) { i = k; img.src = frames[k]; slider.value = k; }
  function step() {
    var next = i + dir;
    if (next >= frames.length) {
      if (mode === "once") { stop(); return; }
      if (mode === "reflect") { dir = -1; next = frames.length - 2; }
      else { next = 0; }
    } else if (next < 0) {
      if (mode === "reflect") { dir = 1; next = 1; } else { next = frames.length - 1; }
    }
    show(Math.max(0, Math.min(frames.length - 1, next)));
  }
  function play() { if (!timer) { timer = setInterval(step, interval); toggle.textContent = "Pause"; } }
  function stop() { if (timer) { clearInterval(timer); timer = null; } toggle.textContent = "Play"; }
  toggle.addEventListener("click", function() { timer ? stop() : play(); });
  slider.addEventListener("input", function() { stop(); show(parseInt(slider.value, 10)); });
  slider.max = frames.length - 1;
  show(0);
  if (mode !== "once") { play(); }
})();
</script>
"""

DISPLAY_TEMPLATE = """
<div class="glplot-anim" id="%(id)s">
  <img>
  <div class="controls">
    <button class="toggle">Play</button>
    <input type="range" min="0" value="0" step="1">
  </div>
</div>
"""


# ----------------------------------------------------------------------------------
# Figure adaptation: matplotlib Figure or GPULinePlot, one interface
# ----------------------------------------------------------------------------------


def _is_mpl_figure(fig: Any) -> bool:
    """Is this an actual matplotlib ``Figure`` rather than a GLPlot engine?

    Duck-typed on ``get_size_inches`` instead of ``isinstance(fig, Figure)`` so this module
    never has to import pyplot just to answer the question — importing pyplot has global
    side effects (backend selection) that a library must not trigger on import.
    """
    return hasattr(fig, "get_size_inches") and hasattr(fig, "savefig") and hasattr(fig, "canvas")


def _dpi_to_scale(dpi: Optional[float]) -> float:
    """Convert matplotlib's dpi to GLPlot's ``savefig`` scale multiplier."""
    if dpi is None:
        return 1.0
    try:
        value = float(dpi)
    except (TypeError, ValueError):
        return 1.0
    if value <= 0:
        return 1.0
    return value / BASE_DPI


def figure_pixel_size(fig: Any, dpi: Optional[float] = None) -> Tuple[int, int]:
    """Predict the ``(width, height)`` in pixels of a frame grabbed from *fig*.

    A prediction, not a measurement — :attr:`AbstractMovieWriter.frame_size` prefers the
    real size of the first grabbed frame and only falls back here before one exists. The
    headless path in particular does not render at ``fb_width`` x ``fb_height``: it builds a
    matplotlib figure of ``max(width/100, 4)`` by ``max(height/100, 3)`` inches at 120 dpi,
    so a 1280x800 GLPlot figure comes out 1536x960. Reproducing that arithmetic here keeps
    ``frame_size`` from lying to a caller who asks before saving.
    """
    scale = _dpi_to_scale(dpi)
    if _is_mpl_figure(fig):
        width_in, height_in = fig.get_size_inches()
        effective = float(dpi) if dpi else float(getattr(fig, "dpi", BASE_DPI))
        return int(width_in * effective), int(height_in * effective)

    if getattr(fig, "window", None) is not None:
        width = int(getattr(fig, "fb_width", 0) * scale)
        height = int(getattr(fig, "fb_height", 0) * scale)
        if width > 0 and height > 0:
            return width, height

    # Mirrors glplot.utils.preview.render_preview's own figsize/dpi choice.
    width_in = max(float(getattr(fig, "width", 1280)) / 100.0, 4.0) * scale
    height_in = max(float(getattr(fig, "height", 800)) / 100.0, 3.0) * scale
    return int(width_in * 120), int(height_in * 120)


def figure_to_rgb(fig: Any, dpi: Optional[float] = None) -> np.ndarray:
    """Render *fig* once and return its pixels as an ``(H, W, 3)`` uint8 array.

    The single bridge between "a figure" and "a frame", and the only function in this
    module that knows a GLPlot engine from a matplotlib one. See the module docstring for
    the two GLPlot render modes and why both go through a temporary PNG.

    Raises :class:`RuntimeError` naming the figure type if neither renderer applies —
    passing something that is not a figure at all is otherwise diagnosed several frames
    later as a confusing Pillow error.
    """
    with tempfile.TemporaryDirectory(prefix="glplot-anim-") as tmpdir:
        target = os.path.join(tmpdir, "frame.png")

        if _is_mpl_figure(fig):
            fig.savefig(target, dpi=dpi, format="png")
        elif hasattr(fig, "scene"):
            _render_glplot_figure(fig, target, _dpi_to_scale(dpi))
        else:
            raise RuntimeError(
                f"Cannot grab a frame from {type(fig).__name__}: expected a "
                f"glplot.engine.GPULinePlot (what glplot.pyplot.figure() returns) or a "
                f"matplotlib Figure. Pass the figure object itself, not an axes or a "
                f"layer."
            )

        return anim_export.normalize_frame(_read_png(target))


def _render_glplot_figure(fig: Any, target: str, scale: float) -> None:
    """Render a GLPlot engine to *target*, through GL if there is a context and Agg if not.

    ``ExportManager.savefig`` prints a "Exported high-res image to ..." line on every call.
    That is fine for one interactive export and intolerable at 300 frames, so stdout is
    swallowed for the duration — this is the only place that print reaches, and suppressing
    it here beats every alternative that would require editing the engine.
    """
    if getattr(fig, "window", None) is not None and hasattr(fig, "export"):
        with contextlib.redirect_stdout(io.StringIO()):
            fig.export.savefig(target, scale=scale)
        return

    from .utils.preview import render_preview

    render_preview(fig, target, scale)


def _read_png(path: str) -> np.ndarray:
    """Read a PNG back as an array, via Pillow if present and matplotlib otherwise."""
    if anim_export.pillow_available():
        from PIL import Image

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.image as mpimg

    return mpimg.imread(path)


def _mark_scene_dirty(fig: Any) -> None:
    """Tell the engine that the user's update callback changed something.

    Every layer is marked dirty, not just the ones the callback returned. GLPlot's dirty
    flags are set by its own mutating API, but the documented ``FuncAnimation`` idiom is to
    reach into an artist and assign — ``line.set_ydata(...)``, or in GLPlot's case writing
    straight into ``layer.pts`` — and an in-place ndarray write is invisible to everything
    except the array itself. Marking everything is O(layers), runs once per frame, and is
    the only approach that cannot miss a change.
    """
    frame_state = getattr(fig, "frame", None)
    if frame_state is not None and hasattr(frame_state, "dirty_scene"):
        frame_state.dirty_scene = True

    scene = getattr(fig, "scene", None)
    for layer in getattr(scene, "layers", ()) or ():
        dirty = getattr(layer, "dirty", None)
        if dirty is None:
            continue
        dirty.data_dirty = True
        dirty.gpu_dirty = True
        dirty.bounds_dirty = True


def _set_artist_visible(artist: Any, visible: bool) -> None:
    """Show or hide an "artist", whether it is a matplotlib Artist or a GLPlot layer.

    matplotlib artists have ``set_visible``; GLPlot layers carry ``style.visible``.
    :class:`ArtistAnimation` is defined entirely in terms of this operation, so supporting
    both here is what lets one ``ArtistAnimation`` script work against either library.
    Anything with neither interface is skipped rather than raising — an ``ArtistAnimation``
    frame is often a mixed list, and a stray item should not abort a 500-frame save.
    """
    if hasattr(artist, "set_visible"):
        artist.set_visible(visible)
        return
    style = getattr(artist, "style", None)
    if style is not None and hasattr(style, "visible"):
        style.visible = visible


class _NullCanvas:
    """The minimum surface :class:`Animation` needs when a figure has no real canvas.

    matplotlib's ``Animation.__init__`` unconditionally touches ``fig.canvas.supports_blit``
    and ``fig.canvas.mpl_connect``, and ``TimedAnimation.__init__`` touches
    ``fig.canvas.new_timer``. A ``GPULinePlot`` has none of these. Rather than fork the
    constructors into "with canvas" and "without canvas" branches — which would have to be
    kept in sync with matplotlib's forever — the missing object is supplied.

    Every method is inert and says so. ``supports_blit`` is ``False`` because it genuinely
    is not supported, so even a caller that inspects it rather than trusting the docstring
    gets the truth.
    """

    supports_blit: bool = False

    def __init__(self, figure: Any) -> None:
        self.figure = figure
        self._next_cid = itertools.count(1)

    def mpl_connect(self, event: str, handler: Callable) -> int:
        """Return a plausible connection id. The handler is never called."""
        return next(self._next_cid)

    def mpl_disconnect(self, cid: int) -> None:
        """No-op counterpart to :meth:`mpl_connect`."""

    def new_timer(self, interval: Optional[float] = None, callbacks: Any = None) -> "_NullTimer":
        return _NullTimer(interval=interval, callbacks=callbacks)

    def draw(self) -> None:
        """No-op: GLPlot redraws from its own loop, not on demand from here."""

    def draw_idle(self) -> None:
        """No-op, see :meth:`draw`."""

    def flush_events(self) -> None:
        """No-op: there is no matplotlib event queue behind a GLPlot figure."""


class _NullTimer:
    """A timer that never fires, for figures with no event loop to fire it.

    :class:`TimedAnimation` needs *an* event source; it does not need one that works in
    order for :meth:`Animation.save` to render every frame, because saving drives the frame
    sequence directly and never consults the timer. Live playback is what a real timer
    would buy, and live playback is unavailable for the separate reason given in the module
    docstring.
    """

    def __init__(self, interval: Optional[float] = None, callbacks: Any = None) -> None:
        self.interval = interval if interval is not None else 200
        self.callbacks: List[Callable] = list(callbacks or [])
        self.running = False

    def add_callback(self, func: Callable, *args: Any, **kwargs: Any) -> Callable:
        self.callbacks.append(func)
        return func

    def remove_callback(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        with contextlib.suppress(ValueError):
            self.callbacks.remove(func)

    def start(self, interval: Optional[float] = None) -> None:
        if interval is not None:
            self.interval = interval
        self.running = True

    def stop(self) -> None:
        self.running = False


def _canvas_for(fig: Any) -> Any:
    """The figure's canvas, or a :class:`_NullCanvas` standing in for it."""
    canvas = getattr(fig, "canvas", None)
    if canvas is not None and hasattr(canvas, "mpl_connect"):
        return canvas
    return _NullCanvas(fig)


def _rc(key: str, default: Any) -> Any:
    """Read a matplotlib rcParam, falling back if matplotlib is absent or the key is not.

    matplotlib is a hard dependency of GLPlot today (``utils/export.py`` imports pyplot to
    write a PNG), so the fallback is defensive rather than expected. It costs three lines
    and means this module keeps working if that ever stops being true.
    """
    try:
        import matplotlib

        return matplotlib.rcParams[key]
    except Exception:
        return default


def adjusted_figsize(w: float, h: float, dpi: float, n: int) -> Tuple[float, float]:
    """Round a figure size so its pixel dimensions are a multiple of *n*.

    Present with matplotlib's exact signature because scripts and downstream libraries call
    it directly. The reason it exists is the same in both libraries: video encoders reject
    frame sizes they cannot divide into macroblocks, and it is far better to shrink the
    figure by a fraction of an inch up front than to have libx264 refuse the movie.

    GLPlot's own writers do not rely on this — :mod:`glplot.utils.anim_export` crops frames
    to even dimensions at encode time instead, which works no matter where the frames came
    from and does not require the caller to have resized anything.
    """
    wnew = int(w * dpi / n) * n / dpi
    hnew = int(h * dpi / n) * n / dpi
    return wnew, hnew


# ----------------------------------------------------------------------------------
# Writer registry
# ----------------------------------------------------------------------------------


class MovieWriterRegistry:
    """Registry of writer classes by human-readable name.

    A faithful copy of matplotlib's, including the detail that iterating it yields only
    *available* writers while ``_registered`` holds them all — so ``'ffmpeg' in
    writers.list()`` answers "can I use ffmpeg", not "does the name exist". That
    distinction is the whole reason the class is not just a dict.
    """

    def __init__(self) -> None:
        self._registered: Dict[str, type] = {}

    def register(self, name: str) -> Callable[[type], type]:
        """Class decorator that files a writer under *name*."""

        def wrapper(writer_cls: type) -> type:
            self._registered[name] = writer_cls
            return writer_cls

        return wrapper

    def is_available(self, name: str) -> bool:
        """Is the writer registered under *name* usable on this machine?"""
        try:
            writer_cls = self._registered[name]
        except KeyError:
            return False
        checker = getattr(writer_cls, "isAvailable", None)
        return True if checker is None else bool(checker())

    def __iter__(self) -> Iterator[str]:
        for name in self._registered:
            if self.is_available(name):
                yield name

    def __contains__(self, name: object) -> bool:
        return self.is_available(str(name))

    def list(self) -> List[str]:
        """The names of every writer available right now."""
        return [*self]

    @property
    def avail(self) -> Dict[str, type]:
        """``{name: class}`` for the available writers.

        matplotlib removed this property in 3.6 in favour of :meth:`list`. It is kept here
        because dropping a name can only break callers, never fix them, and code written
        against older matplotlib still says ``writers.avail``. Prefer :meth:`list`.
        """
        return {name: self._registered[name] for name in self}

    def __getitem__(self, name: str) -> type:
        if self.is_available(name):
            return self._registered[name]
        raise RuntimeError(f"Requested MovieWriter ({name}) not available")


#: The module-level registry, populated by the ``@writers.register`` decorators below.
#: ``glplot.animation.writers`` and ``matplotlib.animation.writers`` are separate objects;
#: registering a custom writer with one does not register it with the other.
writers = MovieWriterRegistry()


# ----------------------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------------------


class AbstractMovieWriter(abc.ABC):
    """Base class for everything that turns grabbed frames into a file.

    The protocol is matplotlib's, unchanged: :meth:`setup`, then :meth:`grab_frame` per
    frame, then :meth:`finish`, with :meth:`saving` as the context manager that guarantees
    the last one runs. Subclasses override :meth:`grab_frame` only when they need something
    other than "append the RGB array to a list"; most only override :meth:`finish`.

    The one substantive difference from matplotlib is :attr:`frame_size`, which reports the
    measured size of the first grabbed frame rather than a figure-inches computation — see
    that property.
    """

    def __init__(
        self,
        fps: int = 5,
        metadata: Optional[dict] = None,
        codec: Optional[str] = None,
        bitrate: Optional[int] = None,
    ) -> None:
        self.fps = fps
        self.metadata = dict(metadata) if metadata is not None else {}
        self.codec = codec if codec is not None else _rc("animation.codec", "h264")
        self.bitrate = bitrate if bitrate is not None else _rc("animation.bitrate", -1)

        self.fig: Any = None
        self.outfile: Optional[str] = None
        self.dpi: Optional[float] = None
        self._frames: List[np.ndarray] = []
        self._measured_size: Optional[Tuple[int, int]] = None

    def setup(self, fig: Any, outfile: Any, dpi: Optional[float] = None) -> None:
        """Bind the writer to a figure and an output path. Called once, before any frame."""
        self.fig = fig
        self.outfile = str(outfile)
        self.dpi = dpi if dpi is not None else _default_dpi(fig)
        self._frames = []
        self._measured_size = None

    def grab_frame(self, **savefig_kwargs: Any) -> None:
        """Render the current state of the figure and keep it as the next frame.

        ``savefig_kwargs`` is accepted for signature parity and ignored apart from ``dpi``.
        matplotlib forwards these to ``Figure.savefig``, where they set the face colour,
        transparency and so on. GLPlot renders through its own pipeline, whose appearance is
        configured on the figure rather than per-save, so there is nothing to forward them
        to. They are ignored rather than rejected because :meth:`Animation.save` *always*
        passes ``facecolor`` and ``transparent``, exactly as matplotlib does, and refusing
        them would break the ported script this module exists to support.
        """
        dpi = savefig_kwargs.get("dpi", self.dpi)
        frame = figure_to_rgb(self.fig, dpi)
        if self._measured_size is None:
            self._measured_size = (frame.shape[1], frame.shape[0])
        self._frames.append(frame)

    @abc.abstractmethod
    def finish(self) -> None:
        """Write the accumulated frames out. Subclasses must override.

        Abstract, as in matplotlib, so ``AbstractMovieWriter()`` raises ``TypeError``
        rather than constructing a writer that silently writes nothing.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement finish(); it cannot write a file. "
            f"Use PillowWriter, FFMpegWriter or PNGFileWriter, or override finish()."
        )

    @contextlib.contextmanager
    def saving(self, fig: Any, outfile: Any, dpi: Optional[float], *args: Any, **kwargs: Any):
        """``with writer.saving(fig, path, dpi): ...`` — setup on entry, finish on exit."""
        self.setup(fig, outfile, dpi, *args, **kwargs)
        try:
            yield self
        finally:
            self.finish()

    @property
    def frame_size(self) -> Tuple[int, int]:
        """``(width, height)`` in pixels of a movie frame.

        matplotlib computes this from ``fig.get_size_inches() * dpi``, which is exact for a
        matplotlib figure and wrong for a GLPlot one: the headless renderer picks its own
        figsize and dpi (see :func:`figure_pixel_size`), and the GL renderer works from the
        framebuffer size, which may differ from the requested window size on a HiDPI
        display. So the *measured* size of the first real frame wins whenever one exists,
        and the prediction is only used before saving has started.
        """
        if self._measured_size is not None:
            return self._measured_size
        return figure_pixel_size(self.fig, self.dpi)

    def _frame_source(self) -> List[np.ndarray]:
        """The collected frames, with the empty case turned into a useful message."""
        if not self._frames:
            raise RuntimeError(
                f"Cannot write {self.outfile}: no frames were grabbed. The animation "
                f"produced nothing to save — check that 'frames' is non-empty and that the "
                f"update function is being called."
            )
        return self._frames


def _default_dpi(fig: Any) -> float:
    """The dpi to use when a caller passes none: the figure's own, else matplotlib's."""
    dpi = getattr(fig, "dpi", None)
    if isinstance(dpi, (int, float)) and dpi > 0:
        return float(dpi)
    value = _rc("savefig.dpi", BASE_DPI)
    if value == "figure":
        return BASE_DPI
    try:
        return float(value)
    except (TypeError, ValueError):
        return BASE_DPI


class MovieWriter(AbstractMovieWriter):
    """Base for writers that pipe frames into an external encoder.

    Matches matplotlib's constructor exactly, including its refusal to be instantiated
    directly — ``MovieWriter`` there is an abstract class that needs a format mixin, and a
    script that relies on that ``TypeError`` (a surprising number of test suites do) gets
    the same one here.
    """

    #: matplotlib pipes raw ``rgba`` into ffmpeg. GLPlot's writers hand normalised RGB
    #: arrays to :mod:`glplot.utils.anim_export`, which owns the pixel format from there
    #: on, so this is reported for parity rather than consulted.
    supported_formats: List[str] = ["rgba"]

    _exec_key = "animation.ffmpeg_path"
    _args_key = "animation.ffmpeg_args"

    def __init__(
        self,
        fps: int = 5,
        codec: Optional[str] = None,
        bitrate: Optional[int] = None,
        extra_args: Optional[Sequence[str]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        if type(self) is MovieWriter:
            raise TypeError(
                "MovieWriter cannot be instantiated directly. Please use one of its " "subclasses."
            )
        super().__init__(fps=fps, metadata=metadata, codec=codec, bitrate=bitrate)
        self._frame_format = self.supported_formats[0]
        self.extra_args = extra_args

    @property
    def frame_format(self) -> str:
        """Image format used for the frames (``png``, ``rgba``, ...).

        matplotlib exposes this as a plain attribute on ``MovieWriter`` and as a read-only
        property on ``FileMovieWriter``. A settable property here satisfies both: assignment
        works where matplotlib allows it, and reading works everywhere matplotlib's does,
        without two classes disagreeing about what kind of thing the name is.
        """
        return self._frame_format

    @frame_format.setter
    def frame_format(self, value: str) -> None:
        self._frame_format = value

    @classmethod
    def isAvailable(cls) -> bool:
        """Is the external tool this writer needs present?"""
        return shutil.which(cls.bin_path()) is not None

    @classmethod
    def bin_path(cls) -> str:
        """Name or path of the command-line tool, from rcParams as in matplotlib."""
        return str(_rc(cls._exec_key, "ffmpeg"))

    def _args(self) -> List[str]:
        """The encoder command line. Informational here; see :class:`FFMpegWriter`."""
        return [self.bin_path()]

    def finish(self) -> None:
        """Write the frames out, choosing the format from the output extension.

        Concrete rather than abstract so that ``FileMovieWriter`` stays instantiable, which
        it is in matplotlib. Subclasses that care about the encoder override this;
        :func:`glplot.utils.anim_export.write_animation` is a sensible default for the ones
        that do not.
        """
        assert self.outfile is not None
        anim_export.write_animation(self._frame_source(), self.outfile, fps=self.fps)


class FileMovieWriter(MovieWriter):
    """Base for writers that put each frame on disk before encoding.

    matplotlib's version writes numbered temp files and hands the pattern to the encoder.
    GLPlot's file-based writers keep frames in memory and let
    :mod:`glplot.utils.anim_export` stream them, so the temp files never exist — but the
    class, its ``setup`` signature with ``frame_prefix``, and its place in the hierarchy do,
    because ``isinstance(writer, FileMovieWriter)`` is a real check in real code.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.frame_prefix: Optional[str] = None

    def setup(
        self,
        fig: Any,
        outfile: Any,
        dpi: Optional[float] = None,
        frame_prefix: Optional[str] = None,
    ) -> None:
        super().setup(fig, outfile, dpi=dpi)
        self.frame_prefix = frame_prefix


@writers.register("pillow")
class PillowWriter(AbstractMovieWriter):
    """Animated GIF via Pillow. Always available in practice, and the universal fallback.

    matplotlib's ``PillowWriter.isAvailable()`` returns ``True`` unconditionally on the
    grounds that Pillow is a hard matplotlib dependency. This one actually checks, because
    GLPlot is used in stripped-down environments where that assumption has been known to be
    false, and a writer that claims to be available and then raises ``ImportError`` is worse
    than one that is honestly missing from ``writers.list()``.
    """

    @classmethod
    def isAvailable(cls) -> bool:
        return anim_export.pillow_available()

    def finish(self) -> None:
        assert self.outfile is not None
        suffix = Path(self.outfile).suffix.lower()
        frames = self._frame_source()
        if suffix in anim_export.PNG_EXTENSIONS:
            # A '.png' target through the GIF writer means a frame sequence; writing a
            # single-frame GIF called 'out.png' would be a surprise, not a service.
            anim_export.write_png_sequence(frames, self.outfile, fps=self.fps)
            return
        anim_export.write_gif(frames, self.outfile, fps=self.fps)


@writers.register("png")
class PNGFileWriter(FileMovieWriter):
    """One PNG per frame. The floor of the dependency ladder, and a GLPlot addition.

    matplotlib has no equivalent — the closest is ``FFMpegFileWriter``, which writes frames
    only as a step towards a video and deletes them. This writer treats the sequence as the
    deliverable, which is what you want when you are about to hand the frames to a video
    editor, or when you are on a machine with neither ffmpeg nor Pillow and need *some*
    output.

    ``fps`` is stored and not written: PNG has nowhere to record it. See
    :func:`glplot.utils.anim_export.write_png_sequence`.
    """

    supported_formats: List[str] = ["rgb"]

    @classmethod
    def isAvailable(cls) -> bool:
        return True

    def finish(self) -> None:
        assert self.outfile is not None
        anim_export.write_png_sequence(self._frame_source(), self.outfile, fps=self.fps)


class FFMpegBase:
    """The ffmpeg-flavoured half of an ffmpeg writer, kept separate as matplotlib does.

    matplotlib composes its writers from a format mixin and a base class, and downstream
    code subclasses the mixin to customise ``output_args``. Preserving the split preserves
    that extension point.
    """

    _exec_key = "animation.ffmpeg_path"
    _args_key = "animation.ffmpeg_args"

    @property
    def output_args(self) -> List[str]:
        """Encoder flags, mirroring matplotlib's so an inspecting caller sees the same list.

        These are *reported*, not executed: :mod:`glplot.utils.anim_export` builds its own
        command line so that the imageio backend and the raw-binary backend produce the same
        video from the same arguments. Keeping this property honest about codec, pixel
        format and bitrate still matters, because ``-pix_fmt yuv420p`` for h264 is the
        difference between a file that plays in Safari and one that does not, and a reader
        checking this property deserves to see that it is handled.
        """
        args: List[str] = []
        outfile = getattr(self, "outfile", "") or ""
        codec = getattr(self, "codec", None) or anim_export.DEFAULT_CODEC
        if Path(outfile).suffix == ".gif":
            codec = "gif"
        else:
            args.extend(["-vcodec", codec])
        if codec in ("h264", "libx264"):
            args.extend(["-pix_fmt", anim_export.H264_PIX_FMT])
        bitrate = getattr(self, "bitrate", -1)
        if bitrate and int(bitrate) > 0:
            args.extend(["-b", f"{int(bitrate)}k"])
        for key, value in (getattr(self, "metadata", None) or {}).items():
            args.extend(["-metadata", f"{key}={value}"])
        args.extend(list(getattr(self, "extra_args", None) or ()))
        return args + ["-y", outfile]


@writers.register("ffmpeg")
class FFMpegWriter(FFMpegBase, MovieWriter):
    """MP4 (and other video containers) via ffmpeg.

    Available when *either* ``imageio`` + ``imageio-ffmpeg`` is importable or an ``ffmpeg``
    binary is on ``PATH`` — a deliberate widening of matplotlib's check, which only looks
    for the binary and therefore reports "unavailable" on the very common setup of a pip
    install with no system ffmpeg. See :func:`glplot.utils.anim_export.video_backend`.

    Unlike the other writers here this one **streams**: frames are encoded and released as
    they arrive, so memory stays flat over a long animation instead of holding every frame
    until :meth:`finish`. That is the difference between saving a 2000-frame movie and
    running out of RAM at frame 900.
    """

    def __init__(
        self,
        fps: int = 5,
        codec: Optional[str] = None,
        bitrate: Optional[int] = None,
        extra_args: Optional[Sequence[str]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        super().__init__(
            fps=fps, codec=codec, bitrate=bitrate, extra_args=extra_args, metadata=metadata
        )
        self._stream: Optional[anim_export.VideoStream] = None

    @classmethod
    def isAvailable(cls) -> bool:
        return anim_export.video_backend() is not None

    def setup(self, fig: Any, outfile: Any, dpi: Optional[float] = None) -> None:
        super().setup(fig, outfile, dpi=dpi)
        self._stream = None

    def grab_frame(self, **savefig_kwargs: Any) -> None:
        dpi = savefig_kwargs.get("dpi", self.dpi)
        frame = figure_to_rgb(self.fig, dpi)
        if self._measured_size is None:
            self._measured_size = (frame.shape[1], frame.shape[0])
        if self._stream is None:
            assert self.outfile is not None
            self._stream = anim_export.VideoStream(
                self.outfile,
                fps=self.fps,
                codec=self._encoder_codec(),
                bitrate=self.bitrate,
                extra_args=self.extra_args,
                metadata=self.metadata,
            )
        self._stream.append(frame)

    def finish(self) -> None:
        if self._stream is None:
            # Nothing was ever grabbed; _frame_source raises the explanatory error.
            self._frame_source()
            return
        stream, self._stream = self._stream, None
        stream.close()

    def _encoder_codec(self) -> str:
        """Translate matplotlib's codec name into one ffmpeg recognises.

        ``rcParams['animation.codec']`` is ``'h264'``, which names the *format*; ffmpeg's
        encoder for it is ``libx264``. Passing 'h264' through unchanged selects ffmpeg's
        decoder-only h264 entry on some builds and fails with a message about an unknown
        encoder, which sends people looking for a broken ffmpeg install that is fine.
        """
        codec = self.codec or anim_export.DEFAULT_CODEC
        return {"h264": "libx264", "hevc": "libx265", "vp9": "libvpx-vp9"}.get(codec, codec)


@writers.register("ffmpeg_file")
class FFMpegFileWriter(FFMpegBase, FileMovieWriter):
    """``ffmpeg`` fed from files rather than a pipe, in matplotlib. Here, an alias.

    matplotlib distinguishes the two because piping raw video through a subprocess used to
    be unreliable on Windows. GLPlot's encoder path handles both backends identically and
    has no such failure mode, so this class exists for the name and for
    ``writers['ffmpeg_file']`` to resolve; it produces byte-identical output to
    :class:`FFMpegWriter`. Documented rather than hidden, because a user who deliberately
    chose the file writer to work around something should know they did not get it.
    """

    @classmethod
    def isAvailable(cls) -> bool:
        return anim_export.video_backend() is not None

    def setup(
        self,
        fig: Any,
        outfile: Any,
        dpi: Optional[float] = None,
        frame_prefix: Optional[str] = None,
    ) -> None:
        super().setup(fig, outfile, dpi=dpi, frame_prefix=frame_prefix)

    def finish(self) -> None:
        assert self.outfile is not None
        anim_export.write_mp4(
            self._frame_source(),
            self.outfile,
            fps=self.fps,
            codec={"h264": "libx264"}.get(self.codec or "", self.codec),
            bitrate=self.bitrate,
            extra_args=self.extra_args,
            metadata=self.metadata,
        )


class ImageMagickBase:
    """rcParam keys for the ImageMagick writers, split out exactly as matplotlib does."""

    _exec_key = "animation.convert_path"
    _args_key = "animation.convert_args"

    @classmethod
    def bin_path(cls) -> str:
        return str(_rc(cls._exec_key, "magick"))

    @classmethod
    def isAvailable(cls) -> bool:
        """Is an ImageMagick binary installed?

        Both names are checked. ImageMagick 7 ships ``magick`` and drops the unprefixed
        ``convert``; ImageMagick 6 ships ``convert``. matplotlib's rcParam still defaults to
        one of them depending on version, so relying on the rcParam alone misreports the
        other installation as missing.
        """
        return any(shutil.which(name) is not None for name in (cls.bin_path(), "magick", "convert"))

    def _args(self) -> List[str]:
        """The ImageMagick command line, mirroring matplotlib's shape.

        Reported for parity and for debugging; :meth:`ImageMagickWriter.finish` builds the
        real invocation from the PNG files it just wrote, because ImageMagick's raw-input
        syntax differs between versions 6 and 7 in ways that fail silently rather than
        loudly.
        """
        extra_args = list(getattr(self, "extra_args", None) or ())
        return [
            self.bin_path(),
            "-size",
            "%ix%i" % self.frame_size,  # type: ignore[attr-defined]
            "-depth",
            "8",
            "-delay",
            str(100 / self.fps),  # type: ignore[attr-defined]
            "-loop",
            "0",
            f"{self.frame_format}:{self.input_names}",  # type: ignore[attr-defined]
            *extra_args,
            self.outfile,  # type: ignore[attr-defined]
        ]


@writers.register("imagemagick")
class ImageMagickWriter(ImageMagickBase, MovieWriter):
    """Animated GIF via ImageMagick.

    Registered for name parity with matplotlib, and genuinely usable when ImageMagick is
    installed. The frames are written as a PNG sequence into a temporary directory and
    handed to ``magick`` — GLPlot never pipes raw frames to it, because ImageMagick's raw
    input syntax differs between versions 6 and 7 in ways that fail silently.

    On a machine without ImageMagick this simply does not appear in ``writers.list()``, and
    :class:`PillowWriter` produces a better GIF anyway.
    """

    supported_formats: List[str] = ["png"]

    @property
    def input_names(self) -> str:
        """What :meth:`ImageMagickBase._args` names as ImageMagick's input.

        matplotlib's pipe-fed writer returns ``'<fmt>:-'``. GLPlot writes real PNG files
        into a temporary directory instead (see the class docstring), so this reports the
        glob those files match rather than a pipe that is never used.
        """
        return f"frame_*.{self.frame_format}"

    def finish(self) -> None:
        assert self.outfile is not None
        frames = self._frame_source()
        binary = shutil.which(self.bin_path()) or shutil.which("magick") or shutil.which("convert")
        if binary is None:
            raise RuntimeError(
                f"Cannot write {self.outfile} with ImageMagick: no 'magick' or 'convert' "
                f"binary on PATH. Install ImageMagick, or use writer='pillow', which needs "
                f"no external tool."
            )
        with tempfile.TemporaryDirectory(prefix="glplot-magick-") as tmpdir:
            paths = anim_export.write_png_sequence(
                frames, os.path.join(tmpdir, "frame.png"), fps=self.fps
            )
            delay = max(int(round(100.0 / max(float(self.fps), 1e-6))), 1)
            command = [binary, "-delay", str(delay), "-loop", "0", *paths, self.outfile]
            _run_tool(command, self.outfile, "ImageMagick")


@writers.register("imagemagick_file")
class ImageMagickFileWriter(ImageMagickBase, FileMovieWriter):
    """File-fed ImageMagick. An alias for :class:`ImageMagickWriter`, which is already
    file-fed; see :class:`FFMpegFileWriter` for the same reasoning."""

    supported_formats: List[str] = ["png"]

    input_names = ImageMagickWriter.input_names
    finish = ImageMagickWriter.finish


def _run_tool(command: List[str], outfile: str, tool: str) -> None:
    """Run an external encoder and turn a non-zero exit into a message that names it."""
    import subprocess

    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip() or f"exit {result.returncode}"
        raise RuntimeError(
            f"Cannot write {outfile}: {tool} failed. {tool} said: {detail}. Try "
            f"writer='pillow' for a GIF or writer='ffmpeg' for a video, neither of which "
            f"needs {tool}."
        )


@writers.register("html")
class HTMLWriter(FileMovieWriter):
    """A self-contained HTML page with a JavaScript frame player.

    Backs :meth:`Animation.to_jshtml`. The output is one file with the frames base64-encoded
    into it and no external references at all — no CDN, no sidecar directory — so it can be
    emailed, committed, or opened from a filesystem with no server.

    ``embed_frames=False`` (matplotlib writes the frames beside the HTML instead) is
    **rejected rather than silently ignored**: a caller who asks for separate frames and
    receives an embedded file gets a working page, so the failure would go unnoticed until
    they tried to swap a frame out. See :meth:`setup`.
    """

    supported_formats: List[str] = ["png"]

    def __init__(
        self,
        fps: int = 30,
        codec: Optional[str] = None,
        bitrate: Optional[int] = None,
        extra_args: Optional[Sequence[str]] = None,
        metadata: Optional[dict] = None,
        embed_frames: bool = False,
        default_mode: str = "loop",
        embed_limit: Optional[float] = None,
    ) -> None:
        super().__init__(
            fps=fps, codec=codec, bitrate=bitrate, extra_args=extra_args, metadata=metadata
        )
        self.embed_frames = embed_frames
        self.default_mode = default_mode
        self.embed_limit = embed_limit
        if default_mode not in ("loop", "once", "reflect"):
            raise ValueError(
                f"default_mode is {default_mode!r}: expected 'loop', 'once' or 'reflect'."
            )

    @classmethod
    def isAvailable(cls) -> bool:
        return anim_export.pillow_available()

    def setup(
        self,
        fig: Any,
        outfile: Any,
        dpi: Optional[float] = None,
        frame_dir: Optional[str] = None,
    ) -> None:
        if frame_dir is not None:
            raise NotImplementedError(
                "HTMLWriter(frame_dir=...) is not supported: GLPlot's HTML output always "
                "embeds its frames so the page is a single self-contained file. Omit "
                "frame_dir, or save a PNG sequence with writer='png' if you need the "
                "frames as separate files."
            )
        super().setup(fig, outfile, dpi=dpi)

    def finish(self) -> None:
        assert self.outfile is not None
        Path(self.outfile).write_text(self._build_html(), encoding="utf-8")

    def _build_html(self) -> str:
        """Assemble the page: CSS, the player markup, then the frames and the script."""
        frames = self._frame_source()
        from PIL import Image

        encoded = []
        for frame in frames:
            buffer = io.BytesIO()
            Image.fromarray(frame, mode="RGB").save(buffer, format="png")
            encoded.append(
                "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
            )

        element_id = "glplot-anim-" + uuid.uuid4().hex[:12]
        interval = MS_PER_SECOND / max(float(self.fps), 1e-6)
        return "".join(
            [
                STYLE_INCLUDE,
                DISPLAY_TEMPLATE % {"id": element_id},
                JS_INCLUDE
                % {
                    "id": element_id,
                    "frames": json.dumps(encoded),
                    "interval": interval,
                    "mode": self.default_mode,
                },
            ]
        )


# ----------------------------------------------------------------------------------
# Animations
# ----------------------------------------------------------------------------------


class Animation:
    """Base class for animations, with matplotlib's constructor and public methods.

    Subclasses supply the frame data (:meth:`new_frame_seq`) and what to do with one
    (:meth:`_draw_frame`); this class supplies :meth:`save` and the HTML exports, which is
    where all the real work happens given that live playback is unavailable (module
    docstring).
    """

    def __init__(self, fig: Any, event_source: Any = None, blit: bool = False) -> None:
        self._draw_was_started = False
        self._fig = fig
        #: Recorded exactly as requested and then never consulted. GLPlot re-renders the
        #: whole scene per frame, so there is no partial redraw to optimise. Kept so that
        #: ``ani._blit`` tells the truth about what the caller asked for.
        self._blit = bool(blit)
        self._paused = False
        self._canvas = _canvas_for(fig)

        self.frame_seq = self.new_frame_seq()
        self.event_source = event_source
        self._first_draw_id = self._canvas.mpl_connect("draw_event", self._start)
        self._close_id = self._canvas.mpl_connect("close_event", self._stop)

    # -- frame sequences ------------------------------------------------------------

    def new_frame_seq(self) -> Iterator[Any]:
        """A fresh iterator over the animation's frame data."""
        return iter(getattr(self, "_framedata", ()))

    def new_saved_frame_seq(self) -> Iterator[Any]:
        """The frame sequence used by :meth:`save`, which may differ from the live one."""
        return self.new_frame_seq()

    # -- drawing --------------------------------------------------------------------

    def _init_draw(self) -> None:
        """Put the figure into its frame-zero state. Overridden by subclasses."""
        self._draw_was_started = True

    def _draw_frame(self, framedata: Any) -> None:
        raise NotImplementedError

    def _draw_next_frame(self, framedata: Any, blit: bool = False) -> None:
        self._pre_draw(framedata, blit)
        self._draw_frame(framedata)
        self._post_draw(framedata, blit)

    def _pre_draw(self, framedata: Any, blit: bool) -> None:
        """Hook before the user's callback. Nothing to clear without blitting."""

    def _post_draw(self, framedata: Any, blit: bool) -> None:
        """Hook after the user's callback: tell the engine the scene changed."""
        _mark_scene_dirty(self._fig)

    # -- playback (documented no-ops on a GLPlot figure) ----------------------------

    def pause(self) -> None:
        """Pause the animation. Inert without a live event loop; see the module docstring."""
        self._paused = True
        if self.event_source is not None:
            with contextlib.suppress(Exception):
                self.event_source.stop()

    def resume(self) -> None:
        """Resume the animation. Inert without a live event loop."""
        self._paused = False
        if self.event_source is not None:
            with contextlib.suppress(Exception):
                self.event_source.start()

    def _start(self, *args: Any) -> None:
        """matplotlib's draw_event hook. Never fired by a GLPlot figure."""
        self._draw_was_started = True

    def _stop(self, *args: Any) -> None:
        """matplotlib's close_event hook. Never fired by a GLPlot figure."""
        if self.event_source is not None:
            with contextlib.suppress(Exception):
                self.event_source.stop()

    def _step(self, *args: Any) -> bool:
        """Advance one frame. Returns False at the end of a non-repeating sequence."""
        try:
            self._draw_next_frame(next(self.frame_seq), self._blit)
            return True
        except StopIteration:
            return False

    # -- saving ---------------------------------------------------------------------

    def save(
        self,
        filename: Any,
        writer: Any = None,
        fps: Optional[float] = None,
        dpi: Optional[float] = None,
        codec: Optional[str] = None,
        bitrate: Optional[int] = None,
        extra_args: Optional[Sequence[str]] = None,
        metadata: Optional[dict] = None,
        extra_anim: Optional[Sequence["Animation"]] = None,
        savefig_kwargs: Optional[dict] = None,
        *,
        progress_callback: Optional[Callable[[int, Optional[int]], Any]] = None,
    ) -> None:
        """Render every frame and write them to *filename*.

        Signature-identical to ``matplotlib.animation.Animation.save``, and behaviourally
        identical apart from two things worth knowing:

        **Writer selection.** matplotlib always uses ``rcParams['animation.writer']`` when
        ``writer`` is ``None``, whatever the filename says, and falls back to Pillow if that
        writer is unavailable. This version looks at the extension first — ``.gif`` picks
        Pillow, ``.mp4`` picks ffmpeg, ``.html`` picks the HTML writer, ``.png`` picks the
        frame-sequence writer — and only consults the rcParam for an extension it does not
        recognise. The files produced are the same in every case where matplotlib succeeds;
        the difference is that ``save("out.gif")`` on a machine with no ffmpeg works here
        and warns there. The fallback chain, if the chosen writer is unavailable, is Pillow
        and then the PNG sequence, which cannot be unavailable.

        **``dpi`` is a scale factor.** GLPlot's renderer has no dpi; ``dpi`` is divided by
        :data:`BASE_DPI` and passed as ``savefig``'s ``scale``, so ``dpi=200`` renders at
        twice the default size. See :func:`figure_pixel_size` for what that works out to.

        ``extra_anim`` is honoured: the animations are stepped together, one saved frame per
        tick, as in matplotlib. ``savefig_kwargs`` is accepted and ignored except for
        ``dpi``; see :meth:`AbstractMovieWriter.grab_frame`.
        """
        all_anim: List[Animation] = [self]
        if extra_anim is not None:
            all_anim.extend(anim for anim in extra_anim if anim._fig is self._fig)

        for anim in all_anim:
            anim._draw_was_started = True

        # This check must precede the fps-from-interval derivation below. Once fps has been
        # filled in from ``self._interval`` it is no longer evidence of what the caller
        # asked for, and every TimedAnimation would trip the "don't pass fps with a writer
        # instance" error. matplotlib orders it the same way, for the same reason.
        if writer is None:
            writer = _default_writer_name(filename)
        elif not isinstance(writer, str) and any(
            arg is not None for arg in (fps, codec, bitrate, extra_args, metadata)
        ):
            raise RuntimeError(
                "Passing in values for arguments fps, codec, bitrate, extra_args, or "
                "metadata is not supported when writer is an existing MovieWriter "
                "instance. These should instead be passed as arguments when creating "
                "the MovieWriter instance."
            )

        if savefig_kwargs is None:
            savefig_kwargs = {}
        else:
            savefig_kwargs = dict(savefig_kwargs)
        savefig_kwargs.pop("bbox_inches", None)

        if fps is None and hasattr(self, "_interval"):
            fps = MS_PER_SECOND / float(self._interval)

        writer_kwargs: Dict[str, Any] = {}
        if codec is not None:
            writer_kwargs["codec"] = codec
        if bitrate is not None:
            writer_kwargs["bitrate"] = bitrate
        if extra_args is not None:
            writer_kwargs["extra_args"] = extra_args
        if metadata is not None:
            writer_kwargs["metadata"] = metadata

        if isinstance(writer, str):
            writer_cls = _resolve_writer_class(writer, filename)
            writer = writer_cls(fps if fps is not None else 5, **writer_kwargs)

        total_frames = _total_frames(all_anim)
        with writer.saving(self._fig, str(filename), dpi):
            for anim in all_anim:
                anim._init_draw()
            frame_number = 0
            for data in zip(*[anim.new_saved_frame_seq() for anim in all_anim]):
                for anim, datum in zip(all_anim, data):
                    anim._draw_next_frame(datum, blit=False)
                    if progress_callback is not None:
                        progress_callback(frame_number, total_frames)
                        frame_number += 1
                writer.grab_frame(**savefig_kwargs)

    # -- HTML ------------------------------------------------------------------------

    def to_html5_video(self, embed_limit: Optional[float] = None) -> str:
        """An HTML5 ``<video>`` tag with the animation embedded as base64 h264.

        Requires a video encoder (:func:`glplot.utils.anim_export.video_backend`); raises
        :class:`RuntimeError` naming what to install if there is none, rather than returning
        a broken tag. ``embed_limit`` is in megabytes and defaults to
        ``rcParams['animation.embed_limit']`` (20 MB); exceeding it returns the string
        ``"Video too large to embed."``, exactly as matplotlib does, because a notebook that
        silently gains a 400 MB cell is a worse outcome than a message.
        """
        if not hasattr(self, "_base64_video"):
            limit_mb = embed_limit
            if limit_mb is None:
                limit_mb = _rc("animation.embed_limit", DEFAULT_EMBED_LIMIT_MB)
            limit_bytes = float(limit_mb) * 1024 * 1024

            if anim_export.video_backend() is None:
                raise RuntimeError(
                    "Cannot build an HTML5 video: no video encoder is available. To fix, "
                    "install one with 'pip install imageio imageio-ffmpeg', or put an "
                    "'ffmpeg' binary on PATH. Use to_jshtml() instead — it needs only "
                    "Pillow and produces a player that works in the same places."
                )

            with tempfile.TemporaryDirectory(prefix="glplot-anim-") as tmpdir:
                path = Path(tmpdir, "temp.m4v")
                writer = FFMpegWriter(
                    fps=MS_PER_SECOND / float(getattr(self, "_interval", 200)),
                    codec="h264",
                    bitrate=_rc("animation.bitrate", -1),
                )
                self.save(str(path), writer=writer)
                encoded = base64.encodebytes(path.read_bytes())
                size = writer.frame_size

            if len(encoded) >= limit_bytes:
                warnings.warn(
                    f"Animation movie is {len(encoded)} bytes, exceeding the limit of "
                    f"{int(limit_bytes)}. If you're sure you want a large animation "
                    f"embedded, raise the animation.embed_limit rc parameter (in MB) or "
                    f"pass embed_limit=.",
                    stacklevel=2,
                )
            else:
                self._base64_video = encoded.decode("ascii")
                self._video_size = f'width="{size[0]}" height="{size[1]}"'

        if hasattr(self, "_base64_video"):
            options = list(HTML5_VIDEO_OPTIONS)
            if getattr(self, "_repeat", False):
                options.append("loop")
            return VIDEO_TAG.format(
                video=self._base64_video, size=self._video_size, options=" ".join(options)
            )
        return "Video too large to embed."

    def to_jshtml(
        self,
        fps: Optional[float] = None,
        embed_frames: bool = True,
        default_mode: Optional[str] = None,
    ) -> str:
        """A self-contained HTML player with the frames embedded as base64 PNGs.

        Needs only Pillow, which makes it the export that works everywhere. The markup is
        GLPlot's own (:data:`DISPLAY_TEMPLATE` / :data:`JS_INCLUDE`), not matplotlib's, so
        the controls look different — play/pause and a scrubber, rather than matplotlib's
        five-button transport. The semantics of ``default_mode`` are the same:  ``'loop'``,
        ``'once'`` or ``'reflect'``, defaulting to ``'loop'`` when the animation repeats.

        ``embed_frames=False`` raises :class:`NotImplementedError` rather than quietly
        embedding anyway; see :class:`HTMLWriter`.
        """
        if fps is None and hasattr(self, "_interval"):
            fps = MS_PER_SECOND / float(self._interval)
        if default_mode is None:
            default_mode = "loop" if getattr(self, "_repeat", False) else "once"
        if not embed_frames:
            raise NotImplementedError(
                "to_jshtml(embed_frames=False) is not supported: GLPlot's HTML output "
                "always embeds its frames so the result is a single self-contained file. "
                "Use embed_frames=True, or save a PNG sequence with "
                "ani.save('frames.png', writer='png') if you need the frames separately."
            )

        if not hasattr(self, "_html_representation"):
            with tempfile.TemporaryDirectory(prefix="glplot-anim-") as tmpdir:
                path = Path(tmpdir, "temp.html")
                writer = HTMLWriter(
                    fps=fps if fps is not None else 30,
                    embed_frames=True,
                    default_mode=default_mode,
                )
                self.save(str(path), writer=writer)
                self._html_representation = path.read_text(encoding="utf-8")
        return self._html_representation

    def _repr_html_(self) -> Optional[str]:
        """IPython display hook, honouring ``rcParams['animation.html']`` like matplotlib."""
        fmt = _rc("animation.html", "none")
        if fmt == "html5":
            return self.to_html5_video()
        if fmt == "jshtml":
            return self.to_jshtml()
        return None

    def __del__(self) -> None:
        """Warn about an animation that was built and then never rendered.

        matplotlib does this because the classic mistake — ``FuncAnimation(...)`` without
        binding the result — produces an animation that is garbage collected before it ever
        draws, and an empty plot with no error at all. The same mistake is available here.
        """
        if not getattr(self, "_draw_was_started", True):
            warnings.warn(
                "Animation was deleted without rendering anything. This is most often the "
                "result of not assigning the Animation to a variable that persists — for "
                "example, 'ani = FuncAnimation(...)' rather than a bare 'FuncAnimation(...)'."
            )


def _default_writer_name(filename: Any) -> str:
    """Pick a writer name from the output extension. See :meth:`Animation.save`."""
    suffix = Path(str(filename)).suffix.lower()
    if suffix in anim_export.VIDEO_EXTENSIONS:
        return "ffmpeg"
    if suffix in anim_export.GIF_EXTENSIONS:
        return "pillow"
    if suffix in anim_export.PNG_EXTENSIONS:
        return "png"
    if suffix in (".html", ".htm"):
        return "html"
    return str(_rc("animation.writer", "pillow"))


def _resolve_writer_class(name: str, filename: Any) -> type:
    """Look *name* up in the registry, degrading down the ladder if it is unavailable.

    Never raises for an unavailable writer. The PNG sequence writer is always available by
    construction, so this always terminates with something that can write a file — and it
    warns on the way down so the user learns why they got a GIF instead of the MP4 they
    asked for. Silence here would be the worst of both worlds: the save appears to succeed
    and the wrong file is on disk.
    """
    if writers.is_available(name):
        return writers[name]

    for fallback, reason in (
        ("pillow", "an animated GIF"),
        ("png", "a PNG frame sequence"),
    ):
        if writers.is_available(fallback):
            warnings.warn(
                f"MovieWriter '{name}' is unavailable, so {filename} was written as "
                f"{reason} using '{fallback}' instead. Install the missing encoder to get "
                f"the requested format: see glplot.utils.anim_export.video_backend().",
                stacklevel=3,
            )
            return writers[fallback]

    raise RuntimeError(f"Requested MovieWriter ({name}) not available")


def _total_frames(all_anim: Sequence["Animation"]) -> Optional[int]:
    """Total frames across animations for ``progress_callback``, or None if unknowable."""
    counts = [getattr(anim, "_save_count", None) for anim in all_anim]
    if any(count is None for count in counts):
        return None
    return sum(counts)  # type: ignore[arg-type]


class TimedAnimation(Animation):
    """An :class:`Animation` whose frames are separated by a fixed wall-clock interval.

    ``interval`` (milliseconds per frame) is what :meth:`Animation.save` turns into ``fps``
    when the caller does not pass one, which is why ``FuncAnimation(..., interval=20)``
    followed by a bare ``save()`` produces a 50 fps file. That conversion is the only place
    ``interval`` has any effect here, since there is no live loop for it to pace.
    """

    def __init__(
        self,
        fig: Any,
        interval: float = 200,
        repeat_delay: float = 0,
        repeat: bool = True,
        event_source: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._interval = interval
        # matplotlib's undocumented back-compat: repeat_delay=None means 0.
        self._repeat_delay = repeat_delay if repeat_delay is not None else 0
        self._repeat = repeat
        self._paused = False
        if event_source is None:
            canvas = _canvas_for(fig)
            event_source = canvas.new_timer(interval=self._interval)
        super().__init__(fig, event_source=event_source, *args, **kwargs)

    @property
    def repeat(self) -> bool:
        """Whether the animation loops. Read-only, as in matplotlib."""
        return self._repeat


class FuncAnimation(TimedAnimation):
    """Animation built by repeatedly calling *func*. The class almost everyone uses.

    ``func(frame, *fargs)`` is called once per frame and is expected to mutate the figure.
    In matplotlib it returns the artists it touched, for blitting; here that return value is
    stored on ``ani._drawn_artists`` and otherwise unused, since blitting does not apply
    (module docstring). A callback written for matplotlib therefore needs no change:
    returning artists is harmless, and so is returning nothing.

    ``frames`` is interpreted exactly as matplotlib does, and the distinctions matter:

    * ``None`` — count from zero forever. Only usable with an explicit ``save_count``.
    * an integer *n* — ``range(n)``, and ``save_count`` becomes *n*.
    * an iterable — used as the frame data itself, and ``len()`` (when it has one) becomes
      ``save_count``. With ``repeat=True`` the iterable is ``itertools.tee``-d so a
      generator can be replayed.
    * a callable — treated as a generator factory, called once per pass.

    The GLPlot-specific part is what "mutate the figure" means. Both idioms work: calling
    ``glplot.pyplot`` functions from inside the callback, and writing into a layer's arrays
    (``fig.scene.layers[0].pts[:, 1] = ...``). After every call, every layer is marked dirty
    so a live GL context re-uploads; see :func:`_mark_scene_dirty`.
    """

    def __init__(
        self,
        fig: Any,
        func: Callable,
        frames: Any = None,
        init_func: Optional[Callable] = None,
        fargs: Optional[tuple] = None,
        save_count: Optional[int] = None,
        *,
        cache_frame_data: bool = True,
        **kwargs: Any,
    ) -> None:
        self._args = fargs if fargs else ()
        self._func = func
        self._init_func = init_func
        self._save_count = save_count
        self._drawn_artists: Any = []

        if frames is None:
            self._iter_gen: Callable[[], Iterator[Any]] = itertools.count
        elif callable(frames):
            self._iter_gen = frames
        elif np.iterable(frames):
            if kwargs.get("repeat", True):
                self._tee_from = frames

                def iter_frames(frames: Any = frames) -> Iterator[Any]:
                    this, self._tee_from = itertools.tee(self._tee_from, 2)
                    yield from this

                self._iter_gen = iter_frames
            else:
                self._iter_gen = lambda: iter(frames)
            if hasattr(frames, "__len__"):
                self._save_count = len(frames)
                if save_count is not None:
                    warnings.warn(
                        f"You passed in an explicit save_count={save_count} which is being "
                        f"ignored in favor of len(frames)={len(frames)}.",
                        stacklevel=2,
                    )
        else:
            self._iter_gen = lambda: iter(range(frames))
            self._save_count = frames
            if save_count is not None:
                warnings.warn(
                    f"You passed in an explicit save_count={save_count} which is being "
                    f"ignored in favor of frames={frames}.",
                    stacklevel=2,
                )

        if self._save_count is None and cache_frame_data:
            warnings.warn(
                f"frames={frames!r} which we can infer the length of, did not pass an "
                f"explicit save_count and passed cache_frame_data=True. To avoid a "
                f"possibly unbounded cache, frame data caching has been disabled. To "
                f"suppress this warning either pass cache_frame_data=False or "
                f"save_count=MAX_FRAMES.",
                stacklevel=2,
            )
            cache_frame_data = False

        self._cache_frame_data = cache_frame_data
        self._save_seq: List[Any] = []

        super().__init__(fig, **kwargs)

        # __init__ above draws frame zero into _save_seq; that is init state, not a frame.
        self._save_seq = []

    @property
    def save_count(self) -> Optional[int]:
        """How many frames :meth:`Animation.save` will write, when that is knowable."""
        return self._save_count

    def new_frame_seq(self) -> Iterator[Any]:
        return self._iter_gen()

    def new_saved_frame_seq(self) -> Iterator[Any]:
        if self._save_seq:
            self._old_saved_seq = list(self._save_seq)
            return iter(self._old_saved_seq)
        if self._save_count is None:
            frame_seq = self.new_frame_seq()

            def gen() -> Iterator[Any]:
                with contextlib.suppress(StopIteration):
                    while True:
                        yield next(frame_seq)

            return gen()
        return itertools.islice(self.new_frame_seq(), self._save_count)

    def _init_draw(self) -> None:
        super()._init_draw()
        if self._init_func is None:
            # No init function: draw the first frame so the figure is in a valid state,
            # then rewind, exactly as matplotlib does.
            try:
                frame_data = next(self.new_frame_seq())
            except StopIteration:
                warnings.warn(
                    "Can not start iterating the frames for the initial draw. This can be "
                    "caused by passing in a 0 length sequence for *frames*.",
                    stacklevel=2,
                )
                return
            self._draw_frame(frame_data)
        else:
            self._drawn_artists = self._init_func()
        _mark_scene_dirty(self._fig)
        # Load-bearing, and matplotlib does exactly the same at the end of its _init_draw.
        # The initial draw above went through _draw_frame, which appends to _save_seq --
        # so without this reset, _save_seq holds one entry, new_saved_frame_seq() decides
        # there is a cached sequence to replay, and save() writes a one-frame movie. That
        # failure is silent: the file is valid, it just has 1 frame instead of 200.
        self._save_seq = []

    def _draw_frame(self, framedata: Any) -> None:
        if self._cache_frame_data:
            self._save_seq.append(framedata)
            if self._save_count is not None:
                self._save_seq = self._save_seq[-self._save_count :]
        self._drawn_artists = self._func(framedata, *self._args)


class ArtistAnimation(TimedAnimation):
    """Animation over a pre-built list of per-frame artist collections.

    ``artists`` is a list of lists: ``artists[i]`` is everything visible in frame *i*. Each
    frame hides every artist from every other frame and shows its own, which is why the
    same object may appear in several frames without them fighting.

    GLPlot layers work here as well as matplotlib artists do — :func:`_set_artist_visible`
    speaks both ``set_visible()`` and ``layer.style.visible`` — so the natural GLPlot idiom
    is to plot every frame's data once up front and let this class flip the visibility.
    That is genuinely faster than :class:`FuncAnimation` for a fixed set of frames, because
    no data is re-uploaded.
    """

    def __init__(self, fig: Any, artists: Sequence[Any], *args: Any, **kwargs: Any) -> None:
        self._drawn_artists: Any = []
        self._framedata = artists
        super().__init__(fig, *args, **kwargs)

    @property
    def _save_count(self) -> int:
        """Frame total for ``progress_callback``.

        ``FuncAnimation`` stores this as an attribute because it often cannot know the
        count; ``ArtistAnimation`` always can, since the frame list is given up front, so it
        is derived rather than stored and can never fall out of sync with the data.
        """
        return len(self._framedata)

    def new_frame_seq(self) -> Iterator[Any]:
        return iter(self._framedata)

    def new_saved_frame_seq(self) -> Iterator[Any]:
        return self.new_frame_seq()

    def _init_draw(self) -> None:
        super()._init_draw()
        for frame in self._framedata:
            for artist in frame:
                _set_artist_visible(artist, False)
        _mark_scene_dirty(self._fig)

    def _draw_frame(self, artists: Any) -> None:
        # Hide the previous frame's artists before showing this one's, or a frame that
        # shares an artist with its predecessor would leave stale geometry on screen.
        for previous in self._drawn_artists:
            _set_artist_visible(previous, False)
        self._drawn_artists = artists
        for artist in artists:
            _set_artist_visible(artist, True)
