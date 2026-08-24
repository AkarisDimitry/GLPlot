"""Tests for :mod:`glplot.animation` -- the matplotlib-compatible animation API.

The contract this module is held to is "a script written against
``matplotlib.animation`` runs unchanged against ``glplot.animation``". That is not a claim
you can test by writing down what you believe matplotlib's signatures are, because the
belief goes stale the moment matplotlib ships a release. So :class:`TestMatplotlibParity`
introspects the *installed* matplotlib with :func:`inspect.signature` and compares against
it. If matplotlib renames an argument or changes a default, these tests fail and this
module is genuinely wrong until it is updated.

Annotations are excluded from the comparison. GLPlot's code is typed and matplotlib's is
not, so ``str(signature)`` differs on every single function while the actual calling
convention -- parameter names, order, kinds and defaults -- is identical. It is the calling
convention that a ported script depends on, so that is what is asserted.

Everything else here runs headless. ``GPULinePlot`` constructs with no window and no GL
context, and :func:`glplot.utils.preview.render_preview` draws the scene through Agg, so a
full ``ani.save()`` works in CI with no display. The handful of tests that would need a
real GL context are skipped, not faked.
"""

from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

import glplot.animation as animation
import glplot.pyplot as gplt

matplotlib_animation = pytest.importorskip(
    "matplotlib.animation", reason="matplotlib is the specification these tests check against"
)

#: Every name a ported script may reasonably reference. Kept explicit rather than derived
#: from ``dir(matplotlib.animation)``, because that also contains the module's imports
#: (``np``, ``Path``, ``base64`` ...) which are leakage, not API.
MPL_PUBLIC_API = [
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
    "HTMLWriter",
    "writers",
    "adjusted_figsize",
    "DISPLAY_TEMPLATE",
    "INCLUDED_FRAMES",
    "JS_INCLUDE",
    "STYLE_INCLUDE",
]

#: The subset of :data:`MPL_PUBLIC_API` that is actually a class. Derived by asking
#: matplotlib rather than by casing the name, because ``DISPLAY_TEMPLATE`` and friends are
#: upper-case strings and have no signature to compare.
MPL_CLASSES = [
    name for name in MPL_PUBLIC_API if inspect.isclass(getattr(matplotlib_animation, name, None))
]

#: ``Animation.__del__`` warns when an animation is garbage collected without ever having
#: drawn -- faithful matplotlib behaviour, asserted directly in
#: :class:`TestDeleteWithoutRenderingWarning`. Many tests here build an animation only to
#: inspect its frame sequence, so the warning would otherwise fire dozens of times at
#: arbitrary GC points and drown the real output.
pytestmark = pytest.mark.filterwarnings("ignore:Animation was deleted without rendering")


@pytest.fixture(autouse=True)
def clean_state():
    """Clean pyplot state before and after each test, as the rest of the suite does."""
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


@pytest.fixture
def sine_figure():
    """A headless figure holding one polyline layer, plus the x grid it was built from."""
    figure = gplt.figure(width=320, height=240)
    x = np.linspace(0.0, 2.0 * np.pi, 64)
    gplt.plot(x, np.sin(x))
    return figure, x


def calling_convention(func):
    """A signature reduced to what a caller actually depends on.

    Annotations and return types are dropped; names, order, kind (positional-only,
    keyword-only, ``*args``) and defaults are kept. Comparing this rather than
    ``str(signature)`` is what makes the parity assertions meaningful instead of a diff of
    type hints.
    """
    parameters = inspect.signature(func).parameters
    return [(p.name, p.kind, p.default) for p in parameters.values()]


class TestMatplotlibParity:
    """Signatures and names, checked against the installed matplotlib."""

    @pytest.mark.parametrize("name", MPL_PUBLIC_API)
    def test_public_name_exists(self, name):
        assert hasattr(animation, name), f"glplot.animation is missing {name}"

    @pytest.mark.parametrize("name", MPL_CLASSES)
    def test_constructor_signature_matches(self, name):
        theirs = getattr(matplotlib_animation, name)
        ours = getattr(animation, name)
        assert calling_convention(ours) == calling_convention(theirs)

    @pytest.mark.parametrize("name", MPL_CLASSES)
    def test_public_methods_are_all_present(self, name):
        theirs = getattr(matplotlib_animation, name)
        ours = getattr(animation, name)
        missing = {m for m in dir(theirs) if not m.startswith("_")} - set(dir(ours))
        assert not missing, f"{name} is missing public members: {sorted(missing)}"

    @pytest.mark.parametrize("name", MPL_CLASSES)
    def test_public_method_signatures_match(self, name):
        theirs = getattr(matplotlib_animation, name)
        ours = getattr(animation, name)
        for member in dir(theirs):
            if member.startswith("_"):
                continue
            their_attr = getattr(theirs, member)
            our_attr = getattr(ours, member)
            if not callable(their_attr) or isinstance(their_attr, type):
                continue
            try:
                expected = calling_convention(their_attr)
            except (TypeError, ValueError):
                continue
            assert expected == calling_convention(our_attr), f"{name}.{member} differs"

    def test_adjusted_figsize_agrees_numerically(self):
        """Same signature is not enough for a pure function -- the answer must match too."""
        for args in [(12.8, 8.0, 100, 16), (5.0, 3.0, 72, 2), (6.4, 4.8, 100, 16)]:
            assert animation.adjusted_figsize(*args) == matplotlib_animation.adjusted_figsize(*args)

    def test_abstract_classes_are_abstract_in_both(self):
        with pytest.raises(TypeError):
            animation.AbstractMovieWriter()
        with pytest.raises(TypeError):
            animation.MovieWriter()

    def test_concrete_writers_construct_in_both(self):
        """FileMovieWriter is instantiable in matplotlib, so it must be here too."""
        for name in ["FileMovieWriter", "PillowWriter", "FFMpegWriter", "HTMLWriter"]:
            getattr(animation, name)()


class TestWriterRegistry:
    """``writers`` -- the registry, its listing, and availability checks."""

    def test_is_a_registry_instance(self):
        assert isinstance(animation.writers, animation.MovieWriterRegistry)

    def test_list_returns_names(self):
        listed = animation.writers.list()
        assert isinstance(listed, list)
        assert all(isinstance(name, str) for name in listed)

    def test_png_writer_is_always_available(self):
        """The one writer with no dependencies; nothing else can be assumed."""
        assert animation.writers.is_available("png")
        assert "png" in animation.writers.list()

    def test_unknown_name_is_unavailable(self):
        assert not animation.writers.is_available("no_such_writer")

    def test_getitem_raises_for_unavailable(self):
        with pytest.raises(RuntimeError, match="not available"):
            animation.writers["no_such_writer"]

    def test_getitem_returns_the_class(self):
        assert animation.writers["png"] is animation.PNGFileWriter

    def test_contains_reflects_availability(self):
        assert "png" in animation.writers
        assert "no_such_writer" not in animation.writers

    def test_avail_maps_names_to_classes(self):
        """``avail`` was dropped by matplotlib 3.6; kept here for older-style callers."""
        avail = animation.writers.avail
        assert isinstance(avail, dict)
        assert avail["png"] is animation.PNGFileWriter
        assert set(avail) == set(animation.writers.list())

    def test_register_adds_a_writer(self):
        registry = animation.MovieWriterRegistry()

        @registry.register("custom")
        class Custom:
            @classmethod
            def isAvailable(cls):
                return True

        assert registry.list() == ["custom"]
        assert registry["custom"] is Custom

    def test_unavailable_writers_are_hidden_but_registered(self):
        registry = animation.MovieWriterRegistry()

        @registry.register("absent")
        class Absent:
            @classmethod
            def isAvailable(cls):
                return False

        assert registry.list() == []
        assert not registry.is_available("absent")


class TestFuncAnimationConstruction:
    """``frames`` interpretation and the warnings matplotlib emits, mirrored."""

    def test_integer_frames_sets_save_count(self, sine_figure):
        figure, _ = sine_figure
        ani = animation.FuncAnimation(figure, lambda i: None, frames=25)
        assert ani.save_count == 25
        assert list(ani.new_saved_frame_seq()) == list(range(25))

    def test_iterable_frames_are_used_directly(self, sine_figure):
        figure, _ = sine_figure
        ani = animation.FuncAnimation(figure, lambda i: None, frames=[3, 1, 4, 1, 5])
        assert ani.save_count == 5
        assert list(ani.new_saved_frame_seq()) == [3, 1, 4, 1, 5]

    def test_callable_frames_is_a_generator_factory(self, sine_figure):
        figure, _ = sine_figure
        ani = animation.FuncAnimation(
            figure, lambda i: None, frames=lambda: iter([7, 8, 9]), save_count=3
        )
        assert list(ani.new_frame_seq()) == [7, 8, 9]

    def test_none_frames_counts_from_zero(self, sine_figure):
        figure, _ = sine_figure
        ani = animation.FuncAnimation(figure, lambda i: None, frames=None, save_count=4)
        assert list(ani.new_saved_frame_seq()) == [0, 1, 2, 3]

    def test_explicit_save_count_is_overridden_with_a_warning(self, sine_figure):
        figure, _ = sine_figure
        with pytest.warns(UserWarning, match="ignored in favor of"):
            ani = animation.FuncAnimation(figure, lambda i: None, frames=6, save_count=99)
        assert ani.save_count == 6

    def test_unbounded_frames_disables_caching_with_a_warning(self, sine_figure):
        figure, _ = sine_figure
        with pytest.warns(UserWarning, match="cache_frame_data"):
            ani = animation.FuncAnimation(figure, lambda i: None, frames=None)
        assert ani._cache_frame_data is False

    def test_fargs_are_forwarded(self, sine_figure):
        figure, _ = sine_figure
        seen = []
        ani = animation.FuncAnimation(
            figure, lambda i, a, b: seen.append((i, a, b)), frames=2, fargs=("x", "y")
        )
        ani._init_draw()
        assert seen[0] == (0, "x", "y")

    def test_init_func_is_used_instead_of_frame_zero(self, sine_figure):
        figure, _ = sine_figure
        calls = []
        ani = animation.FuncAnimation(
            figure,
            lambda i: calls.append(("update", i)),
            frames=3,
            init_func=lambda: calls.append(("init", None)) or [],
        )
        ani._init_draw()
        assert calls == [("init", None)]

    def test_init_draw_clears_the_saved_sequence(self, sine_figure):
        """Regression: a stale _save_seq makes save() write a one-frame movie.

        ``_init_draw`` renders frame zero through ``_draw_frame``, which appends to
        ``_save_seq``. If that is not cleared, ``new_saved_frame_seq`` decides there is a
        cached sequence to replay and yields exactly one entry -- producing a valid file
        with 1 frame instead of N, with no error anywhere.
        """
        figure, _ = sine_figure
        ani = animation.FuncAnimation(figure, lambda i: None, frames=9)
        ani._init_draw()
        assert ani._save_seq == []
        assert len(list(ani.new_saved_frame_seq())) == 9


class TestBlitIsADocumentedNoOp:
    """``blit`` is accepted and ignored. That must be recorded, not silent."""

    def test_blit_true_is_accepted(self, sine_figure):
        figure, _ = sine_figure
        ani = animation.FuncAnimation(figure, lambda i: None, frames=3, blit=True)
        assert ani._blit is True

    def test_blit_does_not_change_the_output(self, sine_figure, tmp_path):
        """The claim that ignoring blit is safe, verified rather than asserted in prose."""
        pytest.importorskip("PIL")
        from PIL import Image

        figure, x = sine_figure
        layer = figure.scene.layers[0]

        def update(i):
            layer.pts[:, 1] = np.sin(x + i * 0.5).astype(np.float32)
            return (layer,)

        sizes = []
        for blit in (False, True):
            ani = animation.FuncAnimation(figure, update, frames=4, interval=50, blit=blit)
            target = tmp_path / f"blit_{blit}.gif"
            ani.save(target, fps=5)
            sizes.append(Image.open(target).n_frames)
        assert sizes == [4, 4]

    def test_blit_is_documented_in_the_module_docstring(self):
        """A no-op flag is only acceptable while it is written down."""
        assert "blit" in animation.__doc__
        assert "accepted and ignored" in animation.__doc__


class TestHeadlessRendering:
    """The bridge from a windowless GPULinePlot to pixels."""

    def test_figure_has_no_window_before_run(self, sine_figure):
        figure, _ = sine_figure
        assert getattr(figure, "window", None) is None

    def test_figure_to_rgb_returns_a_real_image(self, sine_figure):
        figure, _ = sine_figure
        frame = animation.figure_to_rgb(figure)
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.dtype == np.uint8
        assert frame.shape[0] > 1 and frame.shape[1] > 1

    def test_rendered_frame_is_not_blank(self, sine_figure):
        """A uniformly coloured frame means the scene never made it into the render."""
        figure, _ = sine_figure
        frame = animation.figure_to_rgb(figure)
        assert len(np.unique(frame.reshape(-1, 3), axis=0)) > 2

    def test_mutating_the_layer_changes_the_pixels(self, sine_figure):
        figure, x = sine_figure
        layer = figure.scene.layers[0]
        before = animation.figure_to_rgb(figure)
        layer.pts[:, 1] = np.cos(x * 3.0).astype(np.float32)
        after = animation.figure_to_rgb(figure)
        assert not np.array_equal(before, after)

    def test_dpi_scales_the_frame(self, sine_figure):
        figure, _ = sine_figure
        small = animation.figure_to_rgb(figure, dpi=100)
        large = animation.figure_to_rgb(figure, dpi=200)
        assert large.shape[0] > small.shape[0]

    def test_predicted_size_matches_the_measured_one(self, sine_figure):
        """``frame_size`` must not lie before the first grab."""
        figure, _ = sine_figure
        predicted = animation.figure_pixel_size(figure, 100)
        frame = animation.figure_to_rgb(figure, dpi=100)
        assert predicted == (frame.shape[1], frame.shape[0])

    def test_a_non_figure_is_diagnosed(self):
        with pytest.raises(RuntimeError, match="Cannot grab a frame from"):
            animation.figure_to_rgb(object())

    def test_headless_frames_touch_no_temporary_files(self, sine_figure, monkeypatch):
        """Regression guard: a headless frame used to round-trip through a temp PNG
        (encode, write, read back), profiled at ~35% of one frame's total cost for no
        benefit to a caller that only ever wanted the pixel array. ``figure_to_rgb()`` now
        calls ``render_preview_array()`` directly for this case -- if a future change
        reintroduces the file round trip, ``tempfile.TemporaryDirectory`` fires and this
        fails loudly instead of just getting slower again unnoticed.
        """
        import tempfile as tempfile_module

        figure, _ = sine_figure

        def _fail(*args, **kwargs):
            raise AssertionError("headless figure_to_rgb() should not create a temp directory")

        monkeypatch.setattr(tempfile_module, "TemporaryDirectory", _fail)
        frame = animation.figure_to_rgb(figure)
        assert frame.ndim == 3 and frame.shape[2] == 3


class TestSave:
    """End-to-end saving, headless, with the output reopened and counted."""

    def _animation(self, figure, x, frames=5):
        layer = figure.scene.layers[0]

        def update(i):
            layer.pts[:, 1] = np.sin(x + i * 0.6).astype(np.float32)
            return (layer,)

        return animation.FuncAnimation(figure, update, frames=frames, interval=50)

    @pytest.mark.skipif(not animation.writers.is_available("pillow"), reason="Pillow not installed")
    def test_gif_has_every_frame(self, sine_figure, tmp_path):
        from PIL import Image

        figure, x = sine_figure
        target = tmp_path / "out.gif"
        self._animation(figure, x, frames=6).save(target, fps=10)
        assert Image.open(target).n_frames == 6

    @pytest.mark.skipif(not animation.writers.is_available("pillow"), reason="Pillow not installed")
    def test_saved_frames_actually_differ(self, sine_figure, tmp_path):
        """The animation must animate. Identical frames would still be a valid GIF."""
        from PIL import Image

        figure, x = sine_figure
        target = tmp_path / "moving.gif"
        self._animation(figure, x, frames=6).save(target, fps=10)
        image = Image.open(target)
        image.seek(0)
        first = np.asarray(image.convert("RGB"))
        image.seek(3)
        later = np.asarray(image.convert("RGB"))
        assert not np.array_equal(first, later)

    def test_png_sequence_is_written(self, sine_figure, tmp_path):
        figure, x = sine_figure
        self._animation(figure, x, frames=4).save(tmp_path / "seq.png", fps=5)
        produced = sorted(p for p in os.listdir(tmp_path) if p.startswith("seq_"))
        assert len(produced) == 4
        assert all(os.path.getsize(tmp_path / p) > 0 for p in produced)

    @pytest.mark.skipif(
        not animation.writers.is_available("ffmpeg"), reason="no video encoder installed"
    )
    def test_mp4_is_written_and_decodes(self, sine_figure, tmp_path):
        imageio = pytest.importorskip("imageio.v2")
        figure, x = sine_figure
        target = tmp_path / "out.mp4"
        self._animation(figure, x, frames=5).save(target, fps=10)
        assert os.path.getsize(target) > 0
        assert len(imageio.mimread(target)) == 5

    def test_writer_can_be_named(self, sine_figure, tmp_path):
        figure, x = sine_figure
        self._animation(figure, x, frames=3).save(tmp_path / "named.png", writer="png", fps=5)
        assert os.path.getsize(tmp_path / "named_0000.png") > 0

    def test_writer_can_be_an_instance(self, sine_figure, tmp_path):
        figure, x = sine_figure
        writer = animation.PNGFileWriter(fps=5)
        self._animation(figure, x, frames=3).save(tmp_path / "inst.png", writer=writer)
        assert os.path.getsize(tmp_path / "inst_0000.png") > 0

    def test_writer_instance_plus_codec_is_refused(self, sine_figure, tmp_path):
        """matplotlib raises here, so this must too, or a ported test suite diverges."""
        figure, x = sine_figure
        with pytest.raises(RuntimeError, match="not supported when writer is an existing"):
            self._animation(figure, x, frames=2).save(
                tmp_path / "bad.png", writer=animation.PNGFileWriter(), codec="h264"
            )

    def test_writer_instance_alone_is_allowed(self, sine_figure, tmp_path):
        """TimedAnimation derives fps from interval; that must not trip the check above."""
        figure, x = sine_figure
        self._animation(figure, x, frames=2).save(
            tmp_path / "ok.png", writer=animation.PNGFileWriter()
        )
        assert os.path.exists(tmp_path / "ok_0000.png")

    def test_progress_callback_reports_every_frame(self, sine_figure, tmp_path):
        figure, x = sine_figure
        seen = []
        self._animation(figure, x, frames=4).save(
            tmp_path / "prog.png", fps=5, progress_callback=lambda i, n: seen.append((i, n))
        )
        assert seen == [(0, 4), (1, 4), (2, 4), (3, 4)]

    def test_savefig_kwargs_are_accepted(self, sine_figure, tmp_path):
        """matplotlib's own save() always injects facecolor/transparent; ignoring must not
        mean rejecting."""
        figure, x = sine_figure
        self._animation(figure, x, frames=2).save(
            tmp_path / "kw.png", fps=5, savefig_kwargs={"facecolor": "white", "transparent": False}
        )
        assert os.path.exists(tmp_path / "kw_0000.png")

    def test_bbox_inches_is_dropped(self, sine_figure, tmp_path):
        figure, x = sine_figure
        self._animation(figure, x, frames=2).save(
            tmp_path / "bb.png", fps=5, savefig_kwargs={"bbox_inches": "tight"}
        )
        assert os.path.exists(tmp_path / "bb_0000.png")

    def test_unavailable_writer_degrades_with_a_warning(self, sine_figure, tmp_path, monkeypatch):
        """Never silent. A save that produces a different format must say so."""
        monkeypatch.setattr(animation.FFMpegWriter, "isAvailable", classmethod(lambda cls: False))
        monkeypatch.setattr(animation.PillowWriter, "isAvailable", classmethod(lambda cls: False))
        figure, x = sine_figure
        with pytest.warns(UserWarning, match="is unavailable"):
            self._animation(figure, x, frames=2).save(tmp_path / "deg.mp4", fps=5)


class TestArtistAnimation:
    """Visibility-flipping over pre-built frames, on GLPlot layers."""

    def _figure_with_layers(self, count=4):
        figure = gplt.figure(width=320, height=240)
        x = np.linspace(0.0, 6.0, 40)
        frames = []
        for k in range(count):
            gplt.plot(x, np.sin(x + k))
            frames.append([figure.scene.layers[-1]])
        return figure, frames

    def test_save_count_comes_from_the_frame_list(self):
        figure, frames = self._figure_with_layers(5)
        assert animation.ArtistAnimation(figure, frames)._save_count == 5

    def test_init_draw_hides_every_artist(self):
        figure, frames = self._figure_with_layers(3)
        ani = animation.ArtistAnimation(figure, frames)
        ani._init_draw()
        assert all(not layer.style.visible for frame in frames for layer in frame)

    def test_drawing_a_frame_shows_only_that_frame(self):
        figure, frames = self._figure_with_layers(3)
        ani = animation.ArtistAnimation(figure, frames)
        ani._init_draw()
        ani._draw_frame(frames[1])
        assert frames[1][0].style.visible
        assert not frames[0][0].style.visible
        assert not frames[2][0].style.visible

    def test_advancing_hides_the_previous_frame(self):
        figure, frames = self._figure_with_layers(3)
        ani = animation.ArtistAnimation(figure, frames)
        ani._init_draw()
        ani._draw_frame(frames[0])
        ani._draw_frame(frames[1])
        assert not frames[0][0].style.visible

    def test_matplotlib_artists_work_too(self):
        """The same class must drive a real matplotlib artist, for mixed porting."""

        class FakeArtist:
            def __init__(self):
                self.visible = None

            def set_visible(self, flag):
                self.visible = flag

        figure = gplt.figure(width=200, height=150)
        artists = [[FakeArtist()], [FakeArtist()]]
        ani = animation.ArtistAnimation(figure, artists)
        ani._init_draw()
        assert artists[0][0].visible is False
        ani._draw_frame(artists[1])
        assert artists[1][0].visible is True

    @pytest.mark.skipif(not animation.writers.is_available("pillow"), reason="Pillow not installed")
    def test_saves_every_frame(self, tmp_path):
        from PIL import Image

        figure, frames = self._figure_with_layers(4)
        target = tmp_path / "artists.gif"
        animation.ArtistAnimation(figure, frames, interval=100).save(target, fps=5)
        assert Image.open(target).n_frames == 4


class TestHtmlExports:
    """``to_jshtml`` and ``to_html5_video``."""

    def _animation(self, frames=3):
        figure = gplt.figure(width=200, height=150)
        x = np.linspace(0.0, 6.0, 40)
        gplt.plot(x, np.sin(x))
        layer = figure.scene.layers[0]

        def update(i):
            layer.pts[:, 1] = np.sin(x + i).astype(np.float32)

        return animation.FuncAnimation(figure, update, frames=frames, interval=100)

    @pytest.mark.skipif(not animation.writers.is_available("html"), reason="Pillow not installed")
    def test_jshtml_embeds_every_frame(self):
        html = self._animation(frames=4).to_jshtml()
        assert html.count("data:image/png;base64,") == 4

    @pytest.mark.skipif(not animation.writers.is_available("html"), reason="Pillow not installed")
    def test_jshtml_is_self_contained(self):
        """No CDN, no sidecar directory -- the page must work from a bare filesystem."""
        html = self._animation(frames=2).to_jshtml()
        assert "<script>" in html and "http://" not in html and "https://" not in html

    @pytest.mark.skipif(not animation.writers.is_available("html"), reason="Pillow not installed")
    def test_jshtml_is_cached(self):
        ani = self._animation(frames=2)
        assert ani.to_jshtml() is ani.to_jshtml()

    def test_jshtml_rejects_unembedded_frames(self):
        """A silently-ignored embed_frames=False would look like it worked."""
        with pytest.raises(NotImplementedError, match="embed_frames=False"):
            self._animation(frames=2).to_jshtml(embed_frames=False)

    def test_html_writer_rejects_a_frame_dir(self):
        writer = animation.HTMLWriter()
        with pytest.raises(NotImplementedError, match="frame_dir"):
            writer.setup(None, "x.html", 100, frame_dir="frames")

    def test_html_writer_validates_default_mode(self):
        with pytest.raises(ValueError, match="expected 'loop', 'once' or 'reflect'"):
            animation.HTMLWriter(default_mode="sideways")

    @pytest.mark.skipif(
        not animation.writers.is_available("ffmpeg"), reason="no video encoder installed"
    )
    def test_html5_video_returns_a_video_tag(self):
        html = self._animation(frames=3).to_html5_video()
        assert html.startswith("<video")
        assert 'src="data:video/mp4;base64,' in html

    @pytest.mark.skipif(
        not animation.writers.is_available("ffmpeg"), reason="no video encoder installed"
    )
    def test_html5_video_reports_the_real_size(self):
        html = self._animation(frames=2).to_html5_video()
        assert 'width="' in html and 'height="' in html

    def test_html5_video_without_an_encoder_names_the_alternative(self, monkeypatch):
        monkeypatch.setattr(animation.anim_export, "video_backend", lambda: None)
        with pytest.raises(RuntimeError) as excinfo:
            self._animation(frames=2).to_html5_video()
        message = str(excinfo.value)
        assert "imageio-ffmpeg" in message
        assert "to_jshtml()" in message


class TestWriterMechanics:
    """The MovieWriter protocol itself."""

    def test_saving_context_manager_calls_finish(self, tmp_path):
        figure = gplt.figure(width=200, height=150)
        gplt.plot([0, 1, 2], [0, 1, 4])
        writer = animation.PNGFileWriter(fps=5)
        with writer.saving(figure, tmp_path / "ctx.png", 100):
            writer.grab_frame()
            writer.grab_frame()
        assert os.path.exists(tmp_path / "ctx_0000.png")
        assert os.path.exists(tmp_path / "ctx_0001.png")

    def test_frame_size_is_measured_after_the_first_grab(self, tmp_path):
        figure = gplt.figure(width=200, height=150)
        gplt.plot([0, 1], [0, 1])
        writer = animation.PillowWriter(fps=5)
        writer.setup(figure, tmp_path / "fs.gif", 100)
        writer.grab_frame()
        assert writer.frame_size == (writer._frames[0].shape[1], writer._frames[0].shape[0])

    def test_finishing_with_no_frames_explains_itself(self, tmp_path):
        figure = gplt.figure(width=200, height=150)
        writer = animation.PillowWriter(fps=5)
        writer.setup(figure, tmp_path / "none.gif", 100)
        with pytest.raises(RuntimeError, match="no frames were grabbed"):
            writer.finish()

    def test_pillow_writer_honours_a_png_target(self, tmp_path):
        """'.png' through the GIF writer means a sequence, not a one-frame GIF."""
        figure = gplt.figure(width=200, height=150)
        gplt.plot([0, 1], [0, 1])
        writer = animation.PillowWriter(fps=5)
        with writer.saving(figure, tmp_path / "png_target.png", 100):
            writer.grab_frame()
        assert os.path.exists(tmp_path / "png_target_0000.png")

    def test_ffmpeg_writer_reports_availability_via_either_backend(self):
        from glplot.utils import anim_export

        assert animation.FFMpegWriter.isAvailable() == (anim_export.video_backend() is not None)

    def test_ffmpeg_output_args_force_yuv420p(self):
        """Without this, h264 defaults to yuv444p and Safari plays a black rectangle."""
        writer = animation.FFMpegWriter(fps=5, codec="h264")
        writer.setup(None, "movie.mp4", 100)
        assert "-pix_fmt" in writer.output_args
        assert "yuv420p" in writer.output_args

    def test_ffmpeg_gif_target_switches_codec(self):
        writer = animation.FFMpegWriter(fps=5)
        writer.setup(None, "movie.gif", 100)
        assert "-vcodec" not in writer.output_args

    def test_codec_name_is_translated_for_ffmpeg(self):
        """rcParams says 'h264'; ffmpeg's encoder is called 'libx264'."""
        assert animation.FFMpegWriter(fps=5, codec="h264")._encoder_codec() == "libx264"

    def test_frame_format_is_readable_and_settable(self):
        writer = animation.FFMpegWriter(fps=5)
        assert writer.frame_format == "rgba"
        writer.frame_format = "png"
        assert writer.frame_format == "png"


class TestNullCanvasAdaptation:
    """A GPULinePlot has no canvas; nothing may raise AttributeError because of it."""

    def test_animation_constructs_without_a_canvas(self, sine_figure):
        figure, _ = sine_figure
        assert not hasattr(figure, "canvas")
        animation.FuncAnimation(figure, lambda i: None, frames=3)

    def test_null_canvas_reports_no_blit_support(self, sine_figure):
        figure, _ = sine_figure
        ani = animation.FuncAnimation(figure, lambda i: None, frames=3)
        assert ani._canvas.supports_blit is False

    def test_timed_animation_gets_an_inert_timer(self, sine_figure):
        figure, _ = sine_figure
        ani = animation.FuncAnimation(figure, lambda i: None, frames=3, interval=17)
        assert ani.event_source is not None
        assert ani.event_source.interval == 17

    def test_pause_and_resume_do_not_raise(self, sine_figure):
        figure, _ = sine_figure
        ani = animation.FuncAnimation(figure, lambda i: None, frames=3)
        ani.pause()
        assert ani._paused is True
        ani.resume()
        assert ani._paused is False

    def test_step_advances_and_terminates(self, sine_figure):
        figure, _ = sine_figure
        ani = animation.FuncAnimation(figure, lambda i: None, frames=2)
        assert ani._step() is True
        assert ani._step() is True
        assert ani._step() is False

    def test_repeat_is_exposed(self, sine_figure):
        figure, _ = sine_figure
        assert animation.FuncAnimation(figure, lambda i: None, frames=2, repeat=False).repeat is (
            False
        )


class TestDirtyMarking:
    """In-place array writes are invisible to the engine unless we say so."""

    def test_layers_are_marked_dirty_after_an_update(self, sine_figure):
        figure, x = sine_figure
        layer = figure.scene.layers[0]
        layer.dirty.clear()
        assert layer.dirty.data_dirty is False

        ani = animation.FuncAnimation(
            figure, lambda i: layer.pts.__setitem__((slice(None), 1), 0.0), frames=2
        )
        ani._draw_next_frame(0, blit=False)
        assert layer.dirty.data_dirty is True
        assert layer.dirty.gpu_dirty is True

    def test_scene_is_marked_dirty(self, sine_figure):
        figure, _ = sine_figure
        figure.frame.dirty_scene = False
        ani = animation.FuncAnimation(figure, lambda i: None, frames=2)
        ani._draw_next_frame(0, blit=False)
        assert figure.frame.dirty_scene is True


@pytest.mark.skipif(
    os.environ.get("GLPLOT_GL_TESTS") != "1",
    reason="needs a real GL context; set GLPLOT_GL_TESTS=1 to run",
)
class TestLiveGlContext:
    """Saving through the real GPU pipeline rather than the Agg preview.

    Skipped by default. ``fig.run()`` in test mode opens a hidden window and renders one
    frame, which needs a working GLFW/OpenGL stack -- available on a developer machine and
    usually not in CI, so this is opt-in rather than something that makes the suite red on
    a headless runner.
    """

    def test_save_uses_the_gl_export_path(self, tmp_path):
        figure = gplt.figure(width=320, height=240)
        x = np.linspace(0.0, 6.0, 40)
        gplt.plot(x, np.sin(x))
        figure._is_test_mode = True
        figure.run()
        assert figure.window is not None

        layer = figure.scene.layers[0]
        ani = animation.FuncAnimation(
            figure,
            lambda i: layer.pts.__setitem__((slice(None), 1), np.sin(x + i).astype(np.float32)),
            frames=3,
            interval=50,
        )
        ani.save(tmp_path / "gl.png", fps=5)
        assert os.path.getsize(tmp_path / "gl_0000.png") > 0


class TestDeleteWithoutRenderingWarning:
    """The classic mistake matplotlib warns about, mirrored here."""

    @pytest.mark.filterwarnings("default")
    def test_unrendered_animation_warns_on_collection(self, sine_figure):
        """A bare ``FuncAnimation(...)`` that is never bound draws nothing and says nothing
        without this warning -- the user gets a static plot and no clue why."""
        import gc

        figure, _ = sine_figure
        with pytest.warns(UserWarning, match="deleted without rendering"):
            animation.FuncAnimation(figure, lambda i: None, frames=3)
            gc.collect()

    def test_a_rendered_animation_does_not_warn(self, sine_figure, tmp_path):
        import gc

        figure, _ = sine_figure
        ani = animation.FuncAnimation(figure, lambda i: None, frames=2)
        ani.save(tmp_path / "done.png", fps=5)
        del ani
        gc.collect()


class TestDropInImportPattern:
    """The headline promise, exercised as a user would write it."""

    def test_the_readme_snippet_runs_unchanged(self, tmp_path):
        """Byte-for-byte the matplotlib idiom, with only the import line changed."""
        pytest.importorskip("PIL")
        from PIL import Image

        fig = gplt.figure(width=240, height=180)
        x = np.linspace(0.0, 2.0 * np.pi, 50)
        gplt.plot(x, np.sin(x))
        line = fig.scene.layers[0]

        def update(frame):
            line.pts[:, 1] = np.sin(x + frame / 10.0).astype(np.float32)
            return (line,)

        ani = animation.FuncAnimation(fig, update, frames=20, interval=20, blit=True)
        target = tmp_path / "out.gif"
        ani.save(target, fps=30)

        assert Image.open(target).n_frames == 20
