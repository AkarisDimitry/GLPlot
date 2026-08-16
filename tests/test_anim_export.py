"""Tests for :mod:`glplot.utils.anim_export` -- the frame-sequence file writers.

These tests write real files and read them back. A writer test that only checks "no
exception was raised" is close to worthless: the failure mode that actually happens is an
encoder that returns success and leaves a zero-byte or single-frame file behind (imageio
does exactly this when libx264 rejects the frame size), and only reopening the output
catches it. So every format test asserts on the decoded frame count and, where the format
carries it, on the frame content.

Nothing here needs a GPU or a window: the frames are synthetic numpy arrays. Tests that
need an optional encoder are skipped rather than failed when it is absent, and the
*absence* path is tested too by monkeypatching the probes to report nothing installed.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from glplot.utils import anim_export

#: Deliberately odd in both dimensions. h264 cannot encode odd frame sizes, so any test
#: using this exercises the crop-to-even path rather than the happy path -- which is the
#: whole reason the crop exists.
ODD_SIZE = (97, 131)


def make_frame(index: int, size=ODD_SIZE) -> np.ndarray:
    """A frame whose content depends on *index*, so decoded output can be told apart."""
    frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    frame[:, :, 2] = 40
    row = (index * 7) % (size[0] - 10)
    frame[row : row + 10, :, 0] = 255
    return frame


def make_frames(count: int, size=ODD_SIZE):
    return [make_frame(i, size) for i in range(count)]


class TestCapabilityProbes:
    """The probes must answer without importing anything at module scope."""

    def test_png_is_always_available(self):
        assert ".png" in anim_export.available_formats()

    def test_probes_return_bools(self):
        assert isinstance(anim_export.pillow_available(), bool)
        assert isinstance(anim_export.imageio_available(), bool)

    def test_video_backend_is_a_known_value(self):
        assert anim_export.video_backend() in ("imageio", "ffmpeg", None)

    def test_ffmpeg_binary_is_path_or_none(self):
        binary = anim_export.ffmpeg_binary()
        assert binary is None or os.path.exists(binary)

    def test_module_imports_without_optional_deps(self):
        """Importing the module must not require imageio, ffmpeg or even Pillow."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.modules['imageio']=None; sys.modules['PIL']=None;"
                " import glplot.utils.anim_export as m; print(m.DEFAULT_FPS)",
            ],
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr.decode()


class TestNormalizeFrame:
    """Frame coercion: the shapes real render pipelines actually hand over."""

    def test_uint8_rgb_passthrough(self):
        frame = make_frame(0)
        assert anim_export.normalize_frame(frame).shape == (*ODD_SIZE, 3)

    def test_rgba_drops_alpha(self):
        rgba = np.zeros((4, 5, 4), dtype=np.uint8)
        rgba[..., 3] = 128
        assert anim_export.normalize_frame(rgba).shape == (4, 5, 3)

    def test_greyscale_broadcasts_to_three_channels(self):
        out = anim_export.normalize_frame(np.full((4, 5), 200, dtype=np.uint8))
        assert out.shape == (4, 5, 3)
        assert np.all(out == 200)

    def test_float_is_scaled_from_unit_range(self):
        out = anim_export.normalize_frame(np.ones((3, 3, 3), dtype=float))
        assert out.dtype == np.uint8
        assert out.max() == 255

    def test_float_is_clipped_not_wrapped(self):
        """Out-of-range floats must clamp; wrapping would turn white into black."""
        out = anim_export.normalize_frame(np.full((2, 2, 3), 5.0))
        assert out.max() == 255

    def test_result_is_contiguous(self):
        """A flipped view must be made contiguous, or .tobytes() feeds ffmpeg garbage."""
        flipped = np.flipud(make_frame(0))
        assert anim_export.normalize_frame(flipped).flags["C_CONTIGUOUS"]

    def test_bad_channel_count_names_the_frame(self):
        with pytest.raises(ValueError, match="Frame 3 has 7 channels"):
            anim_export.normalize_frame(np.zeros((4, 4, 7)), 3)

    def test_bad_rank_is_rejected(self):
        with pytest.raises(ValueError, match="expected a 2D greyscale"):
            anim_export.normalize_frame(np.zeros((2, 2, 2, 2)), 0)


class TestIterFrames:
    """Both frame-source shapes, and the ambiguity that is refused."""

    def test_iterable_source(self):
        assert len(list(anim_export.iter_frames(make_frames(4)))) == 4

    def test_callable_source_with_count(self):
        assert len(list(anim_export.iter_frames(make_frame, count=3))) == 3

    def test_callable_without_count_is_refused(self):
        with pytest.raises(ValueError, match="callable frame source needs 'count'"):
            next(anim_export.iter_frames(make_frame))

    def test_count_truncates_an_iterable(self):
        assert len(list(anim_export.iter_frames(make_frames(10), count=4))) == 4


class TestPngSequence:
    """The floor of the ladder: always available, one file per frame."""

    def test_writes_one_file_per_frame(self, tmp_path):
        paths = anim_export.write_png_sequence(make_frames(5), tmp_path / "anim.png", fps=10)
        assert len(paths) == 5
        assert all(os.path.getsize(p) > 0 for p in paths)

    def test_filenames_sort_chronologically(self, tmp_path):
        paths = anim_export.write_png_sequence(make_frames(12), tmp_path / "anim.png")
        assert paths == sorted(paths), "zero-padding must make lexical order == frame order"
        assert paths[0].endswith("anim_0000.png")
        assert paths[11].endswith("anim_0011.png")

    def test_content_round_trips(self, tmp_path):
        pytest.importorskip("PIL")
        from PIL import Image

        paths = anim_export.write_png_sequence(make_frames(3), tmp_path / "rt.png")
        decoded = np.asarray(Image.open(paths[1]).convert("RGB"))
        assert np.array_equal(decoded, make_frame(1))

    def test_missing_parent_directory_is_created(self, tmp_path):
        target = tmp_path / "deep" / "deeper" / "anim.png"
        paths = anim_export.write_png_sequence(make_frames(2), target)
        assert os.path.exists(paths[0])

    def test_empty_source_explains_itself(self, tmp_path):
        with pytest.raises(ValueError, match="produced no frames"):
            anim_export.write_png_sequence([], tmp_path / "empty.png")


@pytest.mark.skipif(not anim_export.pillow_available(), reason="Pillow not installed")
class TestGif:
    def test_frame_count_survives_the_round_trip(self, tmp_path):
        from PIL import Image

        path = anim_export.write_gif(make_frames(7), tmp_path / "a.gif", fps=10)
        assert Image.open(path).n_frames == 7

    def test_fps_becomes_a_frame_duration(self, tmp_path):
        from PIL import Image

        path = anim_export.write_gif(make_frames(3), tmp_path / "b.gif", fps=20)
        assert Image.open(path).info["duration"] == 50

    def test_loop_forever_by_default(self, tmp_path):
        from PIL import Image

        path = anim_export.write_gif(make_frames(3), tmp_path / "c.gif")
        assert Image.open(path).info.get("loop") == 0

    def test_frames_are_not_all_identical(self, tmp_path):
        """Guards the failure where every frame is the first one."""
        from PIL import Image

        path = anim_export.write_gif(make_frames(6), tmp_path / "d.gif", fps=10)
        image = Image.open(path)
        image.seek(0)
        first = np.asarray(image.convert("RGB"))
        image.seek(4)
        later = np.asarray(image.convert("RGB"))
        assert not np.array_equal(first, later)

    def test_callable_source(self, tmp_path):
        from PIL import Image

        path = anim_export.write_gif(make_frame, tmp_path / "e.gif", fps=5, count=4)
        assert Image.open(path).n_frames == 4


@pytest.mark.skipif(anim_export.video_backend() is None, reason="no video encoder installed")
class TestMp4:
    def test_odd_sized_frames_are_cropped_and_encoded(self, tmp_path):
        """The regression this whole code path exists for.

        With odd dimensions and no crop, imageio returns cleanly and leaves a zero-byte
        file. Both the crop and the emptiness check are being tested here.
        """
        path = anim_export.write_mp4(make_frames(6), tmp_path / "odd.mp4", fps=10)
        assert os.path.getsize(path) > 0

    def test_decoded_frame_count_matches(self, tmp_path):
        imageio = pytest.importorskip("imageio.v2")
        path = anim_export.write_mp4(make_frames(8), tmp_path / "count.mp4", fps=10)
        assert len(imageio.mimread(path)) == 8

    def test_decoded_size_is_the_even_crop(self, tmp_path):
        imageio = pytest.importorskip("imageio.v2")
        path = anim_export.write_mp4(make_frames(4), tmp_path / "size.mp4", fps=10)
        frames = imageio.mimread(path)
        assert frames[0].shape[:2] == (ODD_SIZE[0] - 1, ODD_SIZE[1] - 1)

    def test_even_frames_are_not_cropped(self, tmp_path):
        imageio = pytest.importorskip("imageio.v2")
        path = anim_export.write_mp4(make_frames(4, (64, 64)), tmp_path / "even.mp4", fps=10)
        assert imageio.mimread(path)[0].shape[:2] == (64, 64)

    def test_bitrate_is_accepted(self, tmp_path):
        path = anim_export.write_mp4(make_frames(4), tmp_path / "br.mp4", fps=10, bitrate=500)
        assert os.path.getsize(path) > 0

    def test_containers_imageio_rejects_still_work(self, tmp_path):
        """.m4v is refused by imageio's URI check; the raw-binary fallback must cover it."""
        if anim_export.ffmpeg_binary() is None:
            pytest.skip("fallback needs an ffmpeg binary on PATH")
        path = anim_export.write_mp4(make_frames(4), tmp_path / "x.m4v", fps=10)
        assert os.path.getsize(path) > 0

    def test_mid_stream_size_change_is_refused(self, tmp_path):
        frames = make_frames(3) + [make_frame(0, (50, 50))]
        with pytest.raises(ValueError, match="must be the same size"):
            anim_export.write_mp4(frames, tmp_path / "ragged.mp4", fps=10)


@pytest.mark.skipif(
    anim_export.ffmpeg_binary() is None, reason="no ffmpeg binary for the raw-pipe backend"
)
def test_raw_ffmpeg_backend_produces_a_playable_file(tmp_path, monkeypatch):
    """Force the non-preferred backend so the subprocess pipe path is covered too."""
    monkeypatch.setattr(anim_export, "imageio_available", lambda: False)
    assert anim_export.video_backend() == "ffmpeg"
    path = anim_export.write_mp4(make_frames(6), tmp_path / "raw.mp4", fps=10)
    assert os.path.getsize(path) > 0


class TestVideoStream:
    """The push-based encoder that backs the matplotlib-style FFMpegWriter."""

    @pytest.mark.skipif(anim_export.video_backend() is None, reason="no video encoder")
    def test_append_then_close(self, tmp_path):
        target = tmp_path / "stream.mp4"
        with anim_export.VideoStream(target, fps=10) as stream:
            for i in range(5):
                stream.append(make_frame(i))
        assert stream.frame_count == 5
        assert os.path.getsize(target) > 0

    @pytest.mark.skipif(anim_export.video_backend() is None, reason="no video encoder")
    def test_close_is_idempotent(self, tmp_path):
        stream = anim_export.VideoStream(tmp_path / "idem.mp4", fps=10)
        stream.append(make_frame(0))
        stream.append(make_frame(1))
        stream.close()
        stream.close()

    def test_missing_encoder_names_both_install_routes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(anim_export, "imageio_available", lambda: False)
        monkeypatch.setattr(anim_export, "ffmpeg_binary", lambda: None)
        with pytest.raises(RuntimeError) as excinfo:
            anim_export.VideoStream(tmp_path / "nope.mp4", fps=10)
        message = str(excinfo.value)
        assert "imageio-ffmpeg" in message
        assert "ffmpeg" in message
        assert ".gif" in message, "must point at the format that needs no dependency"


class TestWriteAnimationDispatch:
    """Extension-driven dispatch and the graceful-degradation ladder."""

    def test_png_extension_writes_a_sequence(self, tmp_path):
        out = anim_export.write_animation(make_frames(3), tmp_path / "d.png", fps=5)
        assert out.endswith("_0000.png")

    @pytest.mark.skipif(not anim_export.pillow_available(), reason="Pillow not installed")
    def test_gif_extension_writes_a_gif(self, tmp_path):
        from PIL import Image

        out = anim_export.write_animation(make_frames(4), tmp_path / "d.gif", fps=5)
        assert Image.open(out).n_frames == 4

    @pytest.mark.skipif(anim_export.video_backend() is None, reason="no video encoder")
    def test_mp4_extension_writes_a_video(self, tmp_path):
        out = anim_export.write_animation(make_frames(4), tmp_path / "d.mp4", fps=5)
        assert out.endswith(".mp4") and os.path.getsize(out) > 0

    def test_unknown_extension_falls_back_to_png(self, tmp_path):
        out = anim_export.write_animation(make_frames(2), tmp_path / "d.wat", fps=5)
        assert os.path.getsize(out) > 0

    @pytest.mark.skipif(not anim_export.pillow_available(), reason="Pillow not installed")
    def test_video_degrades_to_gif_with_a_warning(self, tmp_path, monkeypatch):
        """Degradation must be loud. A silent format swap is the bug being prevented."""
        monkeypatch.setattr(anim_export, "imageio_available", lambda: False)
        monkeypatch.setattr(anim_export, "ffmpeg_binary", lambda: None)
        with pytest.warns(UserWarning, match="No video encoder available"):
            out = anim_export.write_animation(make_frames(4), tmp_path / "fall.mp4", fps=5)
        assert out.endswith(".gif")
        assert os.path.getsize(out) > 0

    def test_fallback_false_raises_instead(self, tmp_path, monkeypatch):
        monkeypatch.setattr(anim_export, "imageio_available", lambda: False)
        monkeypatch.setattr(anim_export, "ffmpeg_binary", lambda: None)
        with pytest.raises(RuntimeError, match="imageio-ffmpeg"):
            anim_export.write_animation(
                make_frames(4), tmp_path / "hard.mp4", fps=5, fallback=False
            )

    def test_degradation_does_not_reconsume_a_generator(self, tmp_path, monkeypatch):
        """A one-shot source must survive the fallback decision intact.

        Deciding by probing rather than by catching a failure is what makes this true; a
        try/except retry would replay a half-consumed generator and write a short movie.
        """
        pytest.importorskip("PIL")
        from PIL import Image

        monkeypatch.setattr(anim_export, "imageio_available", lambda: False)
        monkeypatch.setattr(anim_export, "ffmpeg_binary", lambda: None)
        source = (make_frame(i) for i in range(6))
        with pytest.warns(UserWarning):
            out = anim_export.write_animation(source, tmp_path / "gen.mp4", fps=5)
        assert Image.open(out).n_frames == 6


class TestErrorMessages:
    """Every failure must name the thing, the problem, and the fix."""

    def test_gif_without_pillow_names_the_install(self, tmp_path, monkeypatch):
        monkeypatch.setattr(anim_export, "pillow_available", lambda: False)
        with pytest.raises(RuntimeError) as excinfo:
            anim_export.write_gif(make_frames(2), tmp_path / "x.gif")
        message = str(excinfo.value)
        assert "pip install Pillow" in message
        assert "PNG frame sequence" in message

    def test_video_error_suggests_the_dependency_free_formats(self, tmp_path, monkeypatch):
        monkeypatch.setattr(anim_export, "imageio_available", lambda: False)
        monkeypatch.setattr(anim_export, "ffmpeg_binary", lambda: None)
        with pytest.raises(RuntimeError) as excinfo:
            anim_export.write_mp4(make_frames(2), tmp_path / "x.mp4")
        message = str(excinfo.value)
        assert "pip install imageio imageio-ffmpeg" in message
        assert ".gif" in message and ".png" in message

    def test_too_small_for_video_is_explained(self, tmp_path):
        if anim_export.video_backend() is None:
            pytest.skip("no video encoder")
        with pytest.raises(ValueError, match="too small to encode"):
            anim_export.write_mp4([np.zeros((1, 1, 3), np.uint8)], tmp_path / "tiny.mp4")
