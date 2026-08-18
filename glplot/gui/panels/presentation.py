"""The Presentation panel: scenes as chapters, and a presenter's way through them.

A timeline is one undifferentiated ribbon. A *presentation* is that ribbon with chapters
you can name, jump between, play one at a time, and advance with a clicker — which is the
whole difference between an animation you export and a talk you give. The chapters already
exist as :class:`glplot.core.timeline.Scene`; this panel is what makes them usable.

Two halves, and they are deliberately different in kind:

* **Authoring** — the scene list. Select, rename, retime, reorder, split at the playhead,
  delete, and ``auto_scenes`` for the case where the animation was authored as one
  continuous run and the chapters are wanted afterwards.
* **Presenting** — a mode. Once you press Present the panel stops being a list editor and
  becomes a clicker: the current scene's name, how far through it you are, and one button
  each way. :attr:`~glplot.core.timeline.Scene.hold` is what makes that a *presentation*
  rather than a video — playback stops at the end of each scene and waits for the click,
  the behaviour every slide tool has and no video player does.

Scene reordering moves the scene's **span**, not the keyframes inside it. That is stated
on the button's tooltip rather than hidden, because the alternative — silently retiming
every key that happens to fall inside the span — would quietly rewrite an animation the
user did not ask to change.

CONTRACT §1.1 holds here exactly as in every other panel: nothing below mutates the
timeline from ``draw``. Playback is advanced by :func:`glplot.gui.panels.timeline.
queue_playback`, which is safe to call from both panels in the same frame.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ...core.timeline import Scene, Timeline
from .. import widgets
from ..icons import icon_button
from .base import Panel
from .timeline import apply_timeline, queue_playback

try:
    from imgui_bundle import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - exercised only on GL-less systems
    IMGUI_AVAILABLE = False
    imgui = None

__all__ = ["PresentationPanel"]

#: Length of a scene created from the playhead when there is no later scene to butt against.
_DEFAULT_SCENE_LEN = 2.0

#: How close to a scene's end counts as "at the end" for the hold. One frame at 240fps: any
#: looser and a fast scene holds early, any tighter and a slow frame steps straight past it.
_HOLD_EPS = 1.0 / 240.0

#: Smallest scene a drag may produce. `Scene` refuses end <= start outright.
_MIN_SCENE = 1e-3

_HELP_HOLD = (
    "Stop at the end of this scene and wait, instead of running on into the next one. This "
    "is what makes the timeline a presentation rather than a video: the pace is yours, and "
    "a click advances. Turn it off for a scene that should flow straight into the one "
    "after it."
)

_HELP_REORDER = (
    "Swaps this scene's span with its neighbour's, keeping both durations. It moves the "
    "chapter, NOT the keyframes inside it - the animation itself is untouched, so a "
    "reorder changes the running order of the talk and nothing about the data."
)

_HELP_AUTO = (
    "Replace every scene with N equal chapters spanning the whole timeline. The quickest "
    "way to get a presentation out of an animation that was authored as one run."
)


class PresentationPanel(Panel):
    """Scenes as chapters: author them, then present them."""

    title = "Presentation"
    icon = "play"
    default_open = False

    def __init__(self, ws: Any) -> None:
        super().__init__(ws)
        #: Index of the scene the list has selected (authoring).
        self._selected: int = 0
        #: Index of the scene being presented. Tracked explicitly rather than derived from
        #: ``scene_at()``: scenes are half-open, so the instant the playhead reaches a
        #: scene's end ``scene_at`` already answers "the next one" and a hold could never
        #: be detected.
        self._present_index: int = 0
        #: Whether presentation playback is running. Panel state, not timeline state: the
        #: timeline knows about playing, not about presenting.
        self._presenting: bool = False
        self._rename_index: Optional[int] = None
        self._rename_buf: str = ""
        self._focus_rename: bool = False
        self._auto_count: int = 4
        self._error: str = ""

    # -- state -----------------------------------------------------------------

    @property
    def timeline(self) -> Timeline:
        """The active panel's timeline."""
        return self.plot.active_panel.timeline

    def _clamp(self) -> None:
        """Keep both indices inside the scene list as scenes come and go."""
        count = len(self.timeline.scenes)
        if count == 0:
            self._selected = 0
            self._present_index = 0
            self._presenting = False
            return
        self._selected = int(np.clip(self._selected, 0, count - 1))
        self._present_index = int(np.clip(self._present_index, 0, count - 1))
        if self._rename_index is not None and not 0 <= self._rename_index < count:
            self._cancel_rename()

    def _current(self) -> Optional[Scene]:
        """The scene the presenter is on, or None when there are none."""
        scenes = self.timeline.scenes
        if not scenes:
            return None
        return scenes[int(np.clip(self._present_index, 0, len(scenes) - 1))]

    # -- public verbs (the workspace's Animation actions call these) -----------

    def enter_presentation(self) -> None:
        """Start presenting from the current scene, or from the first one."""
        timeline = self.timeline
        if not timeline.scenes:
            self._error = "There are no scenes yet. Create one, or use 'Auto scenes'."
            return
        self._error = ""
        self._presenting = True
        index = timeline.scene_index()
        self._present_index = index if index >= 0 else 0
        self._go_to(self._present_index, play=True)

    def exit_presentation(self) -> None:
        """Leave presentation playback and pause where we are."""
        self._presenting = False
        timeline = self.timeline
        self.submit(timeline.pause)

    def toggle_presentation(self) -> None:
        """Enter presentation playback, or leave it."""
        if self._presenting:
            self.exit_presentation()
        else:
            self.enter_presentation()

    def next_scene(self) -> None:
        """Advance a chapter — the clicker's forward button."""
        self._go_to(self._present_index + 1, play=self._presenting)

    def previous_scene(self) -> None:
        """Go back a chapter."""
        self._go_to(self._present_index - 1, play=self._presenting)

    # -- queued mutations ------------------------------------------------------

    def _go_to(self, index: int, *, play: bool) -> None:
        """Put the playhead at the start of scene ``index``, optionally playing it."""
        timeline, plot = self.timeline, self.plot
        scenes = timeline.scenes
        if not scenes:
            return
        target = int(np.clip(int(index), 0, len(scenes) - 1))
        self._present_index = target
        self._selected = target

        def apply() -> None:
            timeline.go_to_scene(target)
            if play:
                timeline.play()
            apply_timeline(plot, timeline)

        self.submit(apply)

    def _hold_at(self, scene: Scene) -> None:
        """Pin the playhead on a held scene's end and stop. Idempotent by construction."""
        timeline, plot = self.timeline, self.plot
        end = float(scene.end)

        def apply() -> None:
            timeline.pause()
            timeline.seek(end)
            apply_timeline(plot, timeline)

        self.submit(apply)

    def _toggle_play(self) -> None:
        timeline = self.timeline
        self.submit(timeline.toggle)

    def _set_scene_field(self, scene: Scene, name: str, value: Any) -> None:
        self.submit(lambda: setattr(scene, name, value))

    def _set_scene_span(self, scene: Scene, start: float, end: float) -> None:
        """Retime a scene, refusing to invert it. Clamped in the drain, not in ``draw``."""
        timeline = self.timeline

        def apply() -> None:
            lo = max(0.0, float(start))
            hi = max(lo + _MIN_SCENE, float(end))
            scene.start, scene.end = lo, hi
            timeline.scenes.sort(key=lambda s: s.start)
            timeline.fit_duration()

        self.submit(apply)

    def _delete(self, scene: Scene) -> None:
        """Remove a scene by identity, not by name: two scenes may share a name."""
        timeline = self.timeline

        def apply() -> None:
            index = next((i for i, s in enumerate(timeline.scenes) if s is scene), None)
            if index is not None:
                del timeline.scenes[index]

        self.submit(apply)

    def _swap(self, index: int) -> None:
        """Swap scene ``index`` with the one before it, keeping both durations and any gap."""
        timeline = self.timeline

        def apply() -> None:
            scenes = timeline.scenes
            if not 1 <= index < len(scenes):
                return
            first, second = scenes[index - 1], scenes[index]
            gap = max(0.0, second.start - first.end)
            start = first.start
            d_first, d_second = first.duration, second.duration
            second.start, second.end = start, start + d_second
            first.start = second.end + gap
            first.end = first.start + d_first
            scenes.sort(key=lambda s: s.start)
            timeline.fit_duration()

        self.submit(apply)

    def _new_scene(self) -> None:
        """Create a scene starting at the playhead and ending at the next one (or later)."""
        timeline = self.timeline
        start = float(timeline.time)
        later = [s.start for s in timeline.scenes if s.start > start + 1e-9]
        end = min(later) if later else max(timeline.duration, start + _DEFAULT_SCENE_LEN)
        if end <= start + _MIN_SCENE:
            end = start + _DEFAULT_SCENE_LEN
        name = self._unique_name()
        self._error = ""

        def apply() -> None:
            timeline.add_scene(name, start, end)

        self.submit(apply)

    def _split(self) -> None:
        """Cut the scene under the playhead in two."""
        timeline = self.timeline
        cut = float(timeline.time)
        scene = timeline.scene_at()
        if scene is None or not (scene.start + _MIN_SCENE < cut < scene.end - _MIN_SCENE):
            self._error = "Put the playhead inside a scene, away from its edges, to split it."
            return
        self._error = ""
        name = self._unique_name()

        def apply() -> None:
            index = next((i for i, s in enumerate(timeline.scenes) if s is scene), None)
            if index is None:
                return
            tail = Scene(
                name=name,
                start=cut,
                end=scene.end,
                description=scene.description,
                hold=scene.hold,
            )
            scene.end = cut
            timeline.scenes.insert(index + 1, tail)
            timeline.scenes.sort(key=lambda s: s.start)

        self.submit(apply)

    def _auto_scenes(self, count: int) -> None:
        timeline = self.timeline
        n = max(1, int(count))
        self._error = ""
        self.submit(lambda: timeline.auto_scenes(n))

    def _unique_name(self) -> str:
        """``Scene N`` for the smallest N not already taken."""
        taken = {s.name for s in self.timeline.scenes}
        index = len(taken) + 1
        while f"Scene {index}" in taken:
            index += 1
        return f"Scene {index}"

    # -- rename ----------------------------------------------------------------

    def _begin_rename(self, index: int, scene: Scene) -> None:
        self._rename_index = index
        self._rename_buf = str(scene.name)
        self._focus_rename = True

    def _cancel_rename(self) -> None:
        self._rename_index = None
        self._rename_buf = ""
        self._focus_rename = False

    def _commit_rename(self, scene: Scene, text: str) -> None:
        new = str(text).strip()
        self._cancel_rename()
        if new and new != scene.name:
            self._set_scene_field(scene, "name", new)

    # -- frame -----------------------------------------------------------------

    def draw(self) -> None:
        if not IMGUI_AVAILABLE:
            return
        self._clamp()
        timeline = self.timeline

        # Presentation playback must run whether or not the Timeline panel is open, and
        # `queue_playback` is written so both panels calling it in one frame advance the
        # timeline exactly once between them.
        queue_playback(self)
        self._enforce_hold(timeline)

        self._draw_presenter(timeline)
        imgui.separator()
        self._draw_scene_list(timeline)
        imgui.spacing()
        self._draw_authoring(timeline)
        if self._error:
            widgets.error_box(self._error)

    def _enforce_hold(self, timeline: Timeline) -> None:
        """Stop at the end of a held scene. The whole of "advance on click"."""
        if not (self._presenting and timeline.playing):
            return
        scene = self._current()
        if scene is None or not scene.hold:
            return
        if timeline.time >= scene.end - _HOLD_EPS:
            self._hold_at(scene)

    # -- presenter -------------------------------------------------------------

    def _draw_presenter(self, timeline: Timeline) -> None:
        """The mode switch and, while presenting, the clicker."""
        scene = self._current()
        count = len(timeline.scenes)

        if not self._presenting:
            width = max(140.0, imgui.get_content_region_avail().x * 0.42)
            if imgui.button("Present", (width, 34.0)):
                self.enter_presentation()
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Play the presentation from the current scene. Each held scene stops "
                    "at its end and waits for you to advance."
                )
            imgui.same_line()
            imgui.text_disabled(
                f"{count} scene{'s' if count != 1 else ''}"
                if count
                else "No scenes yet - create one below."
            )
            return

        imgui.text("PRESENTING")
        imgui.same_line()
        if imgui.button("Exit"):
            self.exit_presentation()
        imgui.same_line()
        imgui.text_disabled(f"scene {self._present_index + 1} of {count}")

        if scene is not None:
            imgui.text(str(scene.name))
            imgui.progress_bar(
                float(scene.progress(timeline.time)),
                (-1.0, 14.0),
                f"{timeline.time:.2f}s / {scene.end:.2f}s",
            )
            if scene.description:
                imgui.text_wrapped(str(scene.description))

        playing = bool(timeline.playing)
        if icon_button("##pr_prev", "skip_start", size=28.0, tooltip="Previous scene"):
            self.previous_scene()
        imgui.same_line()
        if icon_button(
            "##pr_play",
            "pause" if playing else "play",
            size=28.0,
            tooltip="Pause" if playing else "Resume",
            active=playing,
        ):
            self._toggle_play()
        imgui.same_line()
        if icon_button("##pr_next", "skip_end", size=28.0, tooltip="Next scene"):
            self.next_scene()
        imgui.same_line()
        if icon_button("##pr_replay", "refresh", size=28.0, tooltip="Replay this scene"):
            self._go_to(self._present_index, play=True)
        imgui.same_line()
        held = scene is not None and bool(scene.hold)
        if icon_button(
            "##pr_hold",
            "stop",
            size=28.0,
            tooltip="Hold at the end of this scene" if not held else "Run straight on",
            active=held,
            enabled=scene is not None,
        ):
            if scene is not None:
                self._set_scene_field(scene, "hold", not held)

    # -- scene list ------------------------------------------------------------

    def _draw_scene_list(self, timeline: Timeline) -> None:
        """One row per chapter: name, span, hold, reorder, delete."""
        scenes = timeline.scenes
        imgui.text(f"Scenes ({len(scenes)})")
        imgui.same_line()
        widgets.help_marker(
            "Chapters of the timeline. A scene is a span, not a container: a track may "
            "cross a boundary, which is what lets a camera move continue across a chapter "
            "change.\n\n" + _HELP_REORDER
        )
        if not scenes:
            imgui.spacing()
            imgui.text_disabled("No scenes yet.")
            imgui.text_wrapped(
                "Create one from the playhead below, or split the whole timeline into "
                "equal chapters with 'Auto scenes'."
            )
            return

        current = timeline.scene_index()
        for index, scene in enumerate(list(scenes)):
            imgui.push_id(f"pr_scene_{index}")
            try:
                self._draw_scene_row(timeline, scene, index, current)
            finally:
                # pop_id on every path: an unbalanced id stack corrupts the whole frame.
                imgui.pop_id()

    def _draw_scene_row(self, timeline: Timeline, scene: Scene, index: int, current: int) -> None:
        """A single scene row, or its inline rename field."""
        if self._rename_index == index:
            self._draw_rename(scene)
            return

        if icon_button(
            "##goto",
            "play",
            size=18.0,
            tooltip="Go to this scene and play it",
        ):
            self._go_to(index, play=True)
        imgui.same_line()

        label = str(scene.name).replace("##", "# #")
        marker = " *" if index == current else ""
        clicked, _ = imgui.selectable(
            f"{index + 1}. {label}{marker}##pick", index == self._selected, 0, (190.0, 0.0)
        )
        if clicked:
            self._selected = index
            if imgui.is_mouse_double_clicked(0):
                self._begin_rename(index, scene)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Click to select, double-click to rename.")

        imgui.same_line()
        imgui.text_disabled(f"{scene.start:.2f} - {scene.end:.2f}  ({scene.duration:.2f}s)")

        imgui.same_line()
        if icon_button(
            "##hold",
            "stop",
            size=18.0,
            tooltip="Holds at the end" if scene.hold else "Runs on into the next scene",
            active=bool(scene.hold),
        ):
            self._set_scene_field(scene, "hold", not scene.hold)
        imgui.same_line()
        if icon_button(
            "##up",
            "chevron_up",
            size=18.0,
            tooltip="Move earlier - swaps this chapter's span with the one before it",
            enabled=index > 0,
        ):
            self._swap(index)
        imgui.same_line()
        if icon_button(
            "##down",
            "chevron_down",
            size=18.0,
            tooltip="Move later - swaps this chapter's span with the one after it",
            enabled=index + 1 < len(timeline.scenes),
        ):
            self._swap(index + 1)
        imgui.same_line()
        if icon_button("##del", "trash", size=18.0, tooltip="Delete this scene"):
            self._delete(scene)

    def _draw_rename(self, scene: Scene) -> None:
        """Inline rename. Enter commits, Escape cancels, click-away commits."""
        if self._focus_rename:
            imgui.set_keyboard_focus_here()
            self._focus_rename = False
        imgui.push_item_width(max(120.0, imgui.get_content_region_avail().x - 8.0))
        # buffer_length stays at its -1 default (CONTRACT §2.4): an explicit length
        # shorter than the text silently truncates it.
        entered, text = imgui.input_text(
            "##rename",
            self._rename_buf,
            flags=imgui.InputTextFlags_.enter_returns_true | imgui.InputTextFlags_.auto_select_all,
        )
        imgui.pop_item_width()
        self._rename_buf = text
        if entered:
            self._commit_rename(scene, text)
        elif imgui.is_item_deactivated():
            # Escape has already reverted imgui's buffer, so check the key rather than
            # trusting the returned text.
            if imgui.is_key_pressed(imgui.Key.escape):
                self._cancel_rename()
            else:
                self._commit_rename(scene, text)

    # -- authoring -------------------------------------------------------------

    def _draw_authoring(self, timeline: Timeline) -> None:
        """Create, split, auto-chapter, and retime the selected scene."""
        if widgets.section("Create", default_open=True):
            imgui.text_disabled(f"Playhead at {timeline.time:.3f}s")
            if imgui.button("New scene from playhead"):
                self._new_scene()
            imgui.same_line()
            if imgui.button("Split at playhead"):
                self._split()

            imgui.push_item_width(110.0)
            changed, count = imgui.input_int("chapters", int(self._auto_count))
            imgui.pop_item_width()
            if changed:
                self._auto_count = int(np.clip(int(count), 1, 200))
            imgui.same_line()
            if imgui.button("Auto scenes"):
                self._auto_scenes(self._auto_count)
            imgui.same_line()
            widgets.help_marker(_HELP_AUTO)

        scenes = timeline.scenes
        if not scenes or not widgets.section("Selected scene", default_open=True):
            return
        scene = scenes[int(np.clip(self._selected, 0, len(scenes) - 1))]
        imgui.text(str(scene.name))
        imgui.same_line()
        if imgui.small_button("Rename"):
            self._begin_rename(self._selected, scene)

        imgui.push_item_width(120.0)
        changed_start, start = imgui.drag_float(
            "Start", float(scene.start), 0.01, 0.0, float(timeline.duration), "%.3f s"
        )
        changed_end, end = imgui.drag_float(
            "End", float(scene.end), 0.01, 0.0, float(timeline.duration), "%.3f s"
        )
        imgui.pop_item_width()
        if changed_start or changed_end:
            self._set_scene_span(scene, float(start), float(end))

        changed, hold = imgui.checkbox("Hold at the end", bool(scene.hold))
        imgui.same_line()
        widgets.help_marker(_HELP_HOLD)
        if changed:
            self._set_scene_field(scene, "hold", bool(hold))

        _, description = imgui.input_text("Note", str(scene.description))
        if description != scene.description:
            self._set_scene_field(scene, "description", str(description))

        imgui.spacing()
        if imgui.button("Go to scene"):
            self._go_to(self._selected, play=False)
        imgui.same_line()
        if imgui.button("Play this scene"):
            self._go_to(self._selected, play=True)
