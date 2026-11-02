"""``compose_video``: scenes (animated figures) cut together into one video.

The scenes are small and short so the whole file runs in seconds; what is checked is the
arithmetic of the cut (frame counts, where each scene starts, what a transition does to the
pixels), the reuse of what ``animate`` rendered, and every refusal.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from glplot.mcp import motion
from glplot.mcp.state import AnimationLimits, FileSandbox, Limits, McpError
from glplot.mcp.toolbox import Toolbox

HAS_FFMPEG = motion.find_ffmpeg() is not None


@pytest.fixture
def box(tmp_path):
    toolbox = Toolbox(FileSandbox([str(tmp_path)]))
    yield toolbox
    toolbox.close_all()


def card(box, text, *, frames=6, fps=12, still=False):
    """A text-only manim scene: the text is written in, then held."""
    figure = box.create_figure(width=480, height=270, style="manim").data["figure_id"]
    box.set_axes(figure, hide_axes=True, xlim=[0, 1], ylim=[0, 1], legend=False)
    box.add_annotation(
        figure, "text", x=0.5, y=0.5, coords="axes", align="center", text=text, fontsize=30
    )
    if not still:
        box.animate(
            figure,
            fps=fps,
            stages=[
                {"frames": frames // 2, "effects": {"write": "a1"}},
                {"frames": frames - frames // 2},
            ],
        )
    return figure


def brightness(video_frames):
    return [float(np.asarray(motion._image(f)).mean()) for f in video_frames]


# ======================================================================================
# the cut
# ======================================================================================


def test_scenes_are_joined_in_order_from_what_animate_rendered(box):
    one, two = card(box, "One"), card(box, "Two", frames=8)
    result = box.compose_video([one, two], transition="cut")
    scenes = result.data["scenes"]
    assert [s["figure_id"] for s in scenes] == [one, two]
    assert [s["source"] for s in scenes] == ["cached", "cached"]
    assert [s["frames"] for s in scenes] == [6, 8] and result.data["frames"] == 14
    assert scenes[1]["start_s"] == pytest.approx(6 / 12)
    assert result.data["duration_s"] == pytest.approx(14 / 12, abs=1e-3)
    assert result.image.startswith(b"\x89PNG"), "a contact sheet of the scenes"
    assert any("No file written" in note for note in result.data["notes"])


def test_a_still_and_a_hold(box):
    one = card(box, "One")
    end = card(box, "The end", still=True)
    result = box.compose_video(
        [{"figure_id": one, "hold_end": 0.5}, {"figure_id": end, "duration": 1}]
    )
    scenes = result.data["scenes"]
    assert scenes[0]["frames"] == 6 + 6 and scenes[1]["frames"] == 12
    assert scenes[1]["source"] == "still"


def test_fps_resamples_each_scene_and_keeps_its_duration(box):
    one = card(box, "One", frames=12, fps=12)  # one second
    result = box.compose_video([one], fps=24, transition="cut")
    assert result.data["frames"] == 24 and result.data["duration_s"] == pytest.approx(1.0)


def test_a_fade_dips_to_the_page_colour_at_the_join(box, monkeypatch):
    # two stills, so every frame of a scene is the same and only the fade can change it
    one, two = card(box, "One", still=True), card(box, "Two", still=True)
    scenes = [{"figure_id": one, "duration": 1}, {"figure_id": two, "duration": 1}]
    video_cut = _frames(box, monkeypatch, scenes, "cut", fps=12)
    video_fade = _frames(box, monkeypatch, scenes, "fade", transition_s=0.5, fps=12)
    assert len(video_cut) == len(video_fade) == 24, "a fade does not change the length"
    level_cut, level_fade = brightness(video_cut), brightness(video_fade)
    assert level_fade[11] < level_cut[11] and level_fade[12] < level_cut[12]
    assert level_fade[5] == pytest.approx(level_cut[5]), "away from the join, untouched"


def test_a_crossfade_overlaps_the_scenes(box):
    one, two = card(box, "One", frames=12), card(box, "Two", frames=12)
    result = box.compose_video([one, two], transition="crossfade", transition_s=0.25)
    overlap = round(0.25 * 12)
    assert result.data["frames"] == 24 - overlap
    assert result.data["scenes"][1]["start_s"] == pytest.approx((12 - overlap) / 12)


def test_a_scene_changed_since_it_was_animated_is_rendered_again(box):
    one, two = card(box, "One"), card(box, "Two")
    box.update_artist(two, "a1", {"color": "#FC6255"})
    first = box.compose_video([one, two])
    assert [s["source"] for s in first.data["scenes"]] == ["cached", "rendered"]
    again = box.compose_video([one, two])
    assert [s["source"] for s in again.data["scenes"]] == ["cached", "cached"]
    box.create_dataset("anything", {"x": [1.0, 2.0]})  # any dataset change counts
    assert [s["source"] for s in box.compose_video([one]).data["scenes"]] == ["rendered"]


def test_the_frame_cache_has_a_budget(box, tmp_path):
    # room for one scene's frames, not two
    one_scene = box._figures[card(box, "One")].rendered["bytes"]
    limits = dataclasses.replace(
        Limits(), animation=AnimationLimits(max_frame_cache_bytes=int(one_scene * 1.5))
    )
    box.close_all()
    small = Toolbox(FileSandbox([str(tmp_path)]), limits)
    try:
        one, two = card(small, "One"), card(small, "Two")
        kept = [small._figures[f].rendered is not None for f in (one, two)]
        assert kept == [False, True], "the older scene's frames made room for the newer"
        sources = [s["source"] for s in small.compose_video([one, two]).data["scenes"]]
        assert sources == ["rendered", "cached"]
    finally:
        small.close_all()


def test_letterboxing_to_a_requested_size(box, monkeypatch):
    one = card(box, "One")
    frames = _frames(box, monkeypatch, [one], "cut", size=[400, 400])
    assert motion._image(frames[0]).size == (400, 400)


# ======================================================================================
# files
# ======================================================================================


def test_a_gif_is_written(box, tmp_path):
    one, two = card(box, "One"), card(box, "Two")
    result = box.compose_video([one, two], output="video.gif")
    path = tmp_path / "video.gif"
    assert path.exists() and result.data["file"]["path"] == str(path.resolve())
    with Image.open(path) as gif:
        assert gif.size == tuple(result.data["size_px"])


@pytest.mark.skipif(not HAS_FFMPEG, reason="needs ffmpeg")
def test_an_mp4_with_a_soundtrack_cut_to_the_video(box, tmp_path):
    import wave

    one, two = card(box, "One", frames=12), card(box, "Two", frames=12)  # two seconds
    samples = (0.1 * np.sin(np.linspace(0, 2 * np.pi * 440 * 5, 5 * 8000)) * 32767).astype("<i2")
    with wave.open(str(tmp_path / "voice.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(samples.tobytes())  # five seconds of sound
    result = box.compose_video([one, two], output="talk.mp4", audio="voice.wav")
    assert result.data["file"]["audio"] == "voice.wav"
    probe = shutil.which("ffprobe")
    if probe:
        out = subprocess.run(
            [probe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
             str(tmp_path / "talk.mp4")],
            capture_output=True, text=True, check=True,
        )  # fmt: skip
        assert float(out.stdout) == pytest.approx(2.0, abs=0.15), "the video sets the length"


# ======================================================================================
# refusals
# ======================================================================================


@pytest.mark.parametrize(
    "segments, kwargs, fragment",
    [
        ([], {}, "segments is a list"),
        (["fig9"], {}, "fig9"),
        ([{"figure": "fig1"}], {}, "figure_id"),
        ([{"figure_id": "fig1", "speed": 2}], {}, "unknown key"),
        ([{"figure_id": "fig1", "duration": 2}], {}, "is animated"),
        (["still"], {}, "has no animation"),
        (["fig1"], {"transition": "wipe"}, "transition must be one of"),
        (["fig1"], {"transition_s": 9}, "between 0 and 5"),
        (["fig1"], {"size": [10, 10]}, "between 64 and 3840"),
        (["fig1"], {"output": "video.png"}, r"\.mp4, \.webm, \.gif"),
        (["fig1"], {"output": "video.gif", "audio": "x.mp3"}, "needs output='name.mp4'"),
        (["fig1"], {"fps": 0}, "fps must be between"),
    ],
)
def test_refusals_say_what_to_do(box, segments, kwargs, fragment):
    card(box, "One")
    still = card(box, "Two", still=True)
    segments = [still if s == "still" else s for s in segments]
    with pytest.raises(McpError, match=fragment):
        box.compose_video(segments, **kwargs)


def test_too_many_frames_for_a_gif(tmp_path):
    limits = dataclasses.replace(Limits(), animation=AnimationLimits(max_video_gif_frames=10))
    small = Toolbox(FileSandbox([str(tmp_path)]), limits)
    try:
        one, two = card(small, "One"), card(small, "Two")
        with pytest.raises(McpError, match="write .mp4"):
            small.compose_video([one, two], output="long.gif")
    finally:
        small.close_all()


def test_without_a_root_a_file_cannot_be_written():
    closed = Toolbox()
    try:
        one = card(closed, "One")
        assert closed.compose_video([one]).data["frames"] == 6, "a preview still works"
        with pytest.raises(McpError, match="File access is disabled"):
            closed.compose_video([one], output="video.gif")
    finally:
        closed.close_all()


# ======================================================================================
# the transform effect (one equation becoming the next)
# ======================================================================================


def test_transform_fades_one_artist_out_as_the_next_comes_in(box):
    figure = card(box, "x^2", still=True)
    box.add_annotation(figure, "text", x=0.5, y=0.5, coords="axes", align="center", text="2x")
    result = box.animate(
        figure,
        stages=[{"frames": 4}, {"frames": 6, "effects": {"TransformMatchingTex": {"a1": "a2"}}}],
    )
    assert result.data["effects"][1]["effects"] == {"fade_out": ["a1"], "fade_in": ["a2"]}
    variables = box._figures[figure].animation["variables"]
    assert variables["fx_a1_show"] == 1.0 and variables["fx_a2_show"] == 0.0
    with pytest.raises(McpError, match="transform is"):
        box.animate(figure, stages=[{"frames": 4, "effects": {"transform": "a1"}}])


def _frames(box, monkeypatch, segments, transition, **kwargs):
    """The composed frames themselves, caught on their way to the encoder (a GIF would merge
    identical consecutive frames)."""
    caught = []

    def encode(frames, fps, path, fmt, **_):
        caught[:] = [frames[i] for i in range(len(frames))]
        path.write_bytes(b"x")
        return {"format": fmt}

    monkeypatch.setattr(motion, "encode", encode)
    result = box.compose_video(segments, output=f"v{len(caught)}.gif", transition=transition,
                               overwrite=True, **kwargs)  # fmt: skip
    assert len(caught) == result.data["frames"]
    return caught
