"""Frames to files, the contact sheet, and the motion review (``glplot.mcp.motion``).

Frames are synthetic PNGs whose content is known (a square at a known place), so what the encoders
write, what the contact sheet shows and what the review concludes can each be checked exactly.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from glplot.mcp import motion
from glplot.mcp.state import McpError

needs_ffmpeg = pytest.mark.skipif(
    motion.find_ffmpeg() is None or motion.available_formats()["mp4"] is not None,
    reason="needs an ffmpeg with an H.264/mpeg4 encoder on the PATH",
)


def frame_png(x: int, size=(120, 80), colour=(220, 30, 30)) -> bytes:
    """A white frame with a 20 px red square whose left edge is at ``x``."""
    image = Image.new("RGB", size, (255, 255, 255))
    image.paste(Image.new("RGB", (20, 20), colour), (x, 30))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def moving(n=8, step=10):
    return [frame_png(5 + step * i) for i in range(n)]


# -- sampling and measuring ---------------------------------------------------------------


@pytest.mark.parametrize("n, count", [(100, 8), (10, 4), (5, 8), (2, 2), (1, 1), (50, 1)])
def test_sample_indices_cover_first_and_last(n, count):
    picked = motion.sample_indices(n, count)
    assert picked == sorted(set(picked))
    assert picked[0] == 0
    if min(n, count) > 1:
        assert picked[-1] == n - 1
    assert len(picked) <= min(n, count)


def test_frame_steps_are_zero_for_identical_frames_and_positive_for_motion():
    still = motion.frame_steps([frame_png(10)] * 5)
    assert still.tolist() == [0.0] * 4
    walk = motion.frame_steps(moving(6))
    assert walk.shape == (5,) and np.all(walk > 0)


def test_loop_adds_the_closing_step():
    frames = moving(6)
    open_ = motion.frame_steps(frames)
    closed = motion.frame_steps(frames, loop=True)
    assert closed.shape == (6,) and closed[:5].tolist() == open_.tolist()
    assert closed[-1] > closed[:5].max() * 0.5  # last -> first is a big jump for a runner


# -- the review ---------------------------------------------------------------------------


def codes(issues):
    return [item["code"] for item in issues]


def review(steps, **kw):
    defaults = dict(loop=False, views=[], bound_limits={})
    return motion.review_motion(np.asarray(steps, dtype=float), **{**defaults, **kw})


def test_a_still_animation_is_a_warning():
    found = review([0.0] * 10)
    assert codes(found) == ["static"] and found[0]["severity"] == "warning"
    assert "bind" in found[0]["fix"]


def test_mostly_still_is_only_information():
    found = review([0.05, 0, 0, 0, 0, 0, 0.05, 0.05])
    assert codes(found) == ["mostly_still"] and found[0]["severity"] == "info"


def test_smooth_motion_has_no_findings():
    ease = np.sin(np.linspace(0, np.pi, 30)) * 0.05 + 0.01
    assert review(ease) == []


def test_a_single_jump_is_reported_with_its_frames_and_stage_boundary():
    steps = [0.02] * 20
    steps[9] = 0.4
    found = review(steps, stage_edges=[10])
    assert codes(found) == ["abrupt_change"]
    assert "frames 9 and 10" in found[0]["message"] and "stage boundary" in found[0]["message"]


def test_a_loop_that_does_not_close_is_a_warning():
    found = review([0.02] * 9 + [0.3], loop=True)
    assert "loop_seam" in codes(found)
    assert not review([0.02] * 9 + [0.02], loop=True)


def view(low, high):
    return [{"xlim": [low, high], "ylim": [0, 1]}]


def test_a_moving_axis_range_is_a_warning_unless_it_was_pinned_or_bound():
    views = [view(0, 1), view(0, 2), view(0, 3)]
    found = review([0.02] * 2, views=views)
    assert codes(found) == ["xlim_moves"] and "set_axes" in found[0]["fix"]
    assert review([0.02] * 2, views=views, bound_limits={(1, "x"): True}) == []
    assert review([0.02] * 2, views=[view(0, 1)] * 3) == []


def test_a_huge_gif_is_a_suggestion():
    found = review([0.02] * 5, file_bytes=40 * 1024 * 1024, fmt="gif")
    assert codes(found) == ["large_file"] and found[0]["severity"] == "suggestion"
    assert review([0.02] * 5, file_bytes=40 * 1024 * 1024, fmt="mp4") == []


# -- encoders ---------------------------------------------------------------------------


def test_gif_has_every_frame_and_the_requested_delay(tmp_path):
    path = tmp_path / "a.gif"
    info = motion.encode(moving(7), 25, path, "gif")
    image = Image.open(path)
    assert image.n_frames == 7 and image.size == (120, 80)
    assert image.info["duration"] == 40 and info["fps_effective"] == 25.0
    assert image.info.get("loop") == 0  # forever


def test_gif_quantises_the_frame_rate_and_says_so(tmp_path):
    info = motion.encode(moving(3), 30, tmp_path / "a.gif", "gif")
    assert info["fps_effective"] == pytest.approx(33.333, rel=1e-3)  # 3 centiseconds
    fast = motion.encode(moving(3), 100, tmp_path / "b.gif", "gif")
    assert fast["fps_effective"] == 50.0  # nothing below 2 cs is dependable


def test_gif_frames_really_differ(tmp_path):
    path = tmp_path / "a.gif"
    motion.encode(moving(5), 10, path, "gif")
    image = Image.open(path)
    seen = []
    for index in range(image.n_frames):
        image.seek(index)
        seen.append(np.asarray(image.convert("RGB")))
    assert all(not np.array_equal(a, b) for a, b in zip(seen, seen[1:]))
    red = np.argwhere((seen[0][..., 0] > 200) & (seen[0][..., 1] < 80))
    assert red[:, 1].min() == 5  # the square starts at x=5 in frame 0
    red = np.argwhere((seen[4][..., 0] > 200) & (seen[4][..., 1] < 80))
    assert red[:, 1].min() == 45  # and at x=45 in frame 4


def test_gif_repeat_count(tmp_path):
    path = tmp_path / "a.gif"
    motion.encode(moving(3), 10, path, "gif", repeat=2)
    assert Image.open(path).info.get("loop") == 2


def test_apng_is_lossless_and_animated(tmp_path):
    path = tmp_path / "a.apng"
    info = motion.encode(moving(4), 20, path, "apng")
    image = Image.open(path)
    assert image.n_frames == 4 and info["codec"].startswith("apng")
    assert np.array_equal(
        np.asarray(image.convert("RGB")),
        np.asarray(Image.open(io.BytesIO(moving(4)[0])).convert("RGB")),
    )


def test_a_frame_of_another_size_is_padded_not_refused(tmp_path):
    frames = moving(3) + [frame_png(5, size=(100, 70))]
    motion.encode(frames, 10, tmp_path / "a.gif", "gif")
    assert Image.open(tmp_path / "a.gif").n_frames == 4


@needs_ffmpeg
def test_mp4_has_every_frame_and_even_dimensions(tmp_path):
    frames = [frame_png(5 + 8 * i, size=(121, 81)) for i in range(12)]  # odd size on purpose
    path = tmp_path / "a.mp4"
    info = motion.encode(frames, 24, path, "mp4")
    assert path.stat().st_size > 0 and info["codec"] in motion._H264
    import subprocess

    probe = subprocess.run(
        [motion.find_ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True
    ).stderr
    assert "122x82" in probe and "24 fps" in probe  # padded to even, frame rate kept


@needs_ffmpeg
def test_webm_when_the_encoder_exists(tmp_path):
    if motion.available_formats()["webm"] is not None:
        pytest.skip("this ffmpeg has no libvpx")
    path = tmp_path / "a.webm"
    motion.encode(moving(6), 12, path, "webm")
    assert path.stat().st_size > 0


def test_mp4_without_ffmpeg_says_how_to_get_it(tmp_path, monkeypatch):
    monkeypatch.setattr(motion, "find_ffmpeg", lambda: None)
    with pytest.raises(McpError, match="ffmpeg"):
        motion.encode(moving(3), 10, tmp_path / "a.mp4", "mp4")
    assert "ffmpeg" in motion.available_formats()["mp4"]
    assert motion.available_formats()["gif"] is None


def test_nothing_to_encode_is_an_error(tmp_path):
    with pytest.raises(McpError, match="no frames"):
        motion.encode([], 10, tmp_path / "a.gif", "gif")


def test_ffmpeg_path_can_be_forced_by_the_environment(tmp_path, monkeypatch):
    fake = tmp_path / "ffmpeg-fake"
    fake.write_text("")
    monkeypatch.setenv("GLPLOT_FFMPEG", str(fake))
    assert motion.find_ffmpeg() == str(fake)


def test_numbered_png_sequence(tmp_path):
    paths = motion.write_sequence(moving(3), tmp_path / "out", "f_%04d.png")
    assert [p.name for p in paths] == ["f_0001.png", "f_0002.png", "f_0003.png"]
    assert Image.open(paths[2]).size == (120, 80)


# -- the contact sheet ------------------------------------------------------------------


def test_contact_sheet_is_one_png_no_wider_than_asked():
    frames = [frame_png(5 + 5 * i, size=(900, 600)) for i in range(8)]
    labels = [f"#{i}" for i in range(8)]
    sheet = motion.contact_sheet(frames, list(range(8)), labels, columns=4, max_width=1200)
    image = Image.open(io.BytesIO(sheet))
    assert image.format == "PNG" and image.width <= 1200 and image.height > 0


def test_contact_sheet_shows_the_chosen_frames_in_order():
    frames = [frame_png(5 + 20 * i, size=(300, 200)) for i in range(5)]
    sheet = Image.open(
        io.BytesIO(motion.contact_sheet(frames, [0, 4], ["a", "b"], columns=2, max_width=700))
    ).convert("RGB")
    array = np.asarray(sheet)
    red = np.argwhere((array[..., 0] > 200) & (array[..., 1] < 80) & (array[..., 2] < 80))
    left = red[red[:, 1] < sheet.width // 2][:, 1].min()
    right = red[red[:, 1] >= sheet.width // 2][:, 1].min() - sheet.width // 2
    assert right > left  # frame 4's square sits further along its tile than frame 0's


def test_contact_sheet_single_frame():
    sheet = motion.contact_sheet([frame_png(5)], [0], ["only"], columns=4)
    assert Image.open(io.BytesIO(sheet)).width > 0
