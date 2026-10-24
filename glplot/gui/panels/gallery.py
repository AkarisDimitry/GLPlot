"""The Gallery panel: the gallery examples, pointed at your own data.

Every card is a *recipe* (:mod:`glplot.gui.recipes`) -- a kind of plot, the columns it reads
and the look it was designed in -- shown by a real render of the OpenGL window. Pick one, say
which column goes where, and the plot is drawn on the scene as one undoable step; or press
**Try with sample data** to see it work before there is any data at all.

The panel is a thin shell over pure logic: which recipes exist, how columns are guessed,
whether the data can be drawn and how it is drawn all live in ``recipes``. What is here is the
state of the form (dataset, column assignment, look) and the drawing of it.

Two renders, kept deliberately distinct:

* the **card** thumbnails are the stored GL captures (``gallery_assets``) -- what the recipe looks
  like on its own sample data, in the real engine;
* the **preview** in the detail pane is rendered from *your* columns, on a background thread, by
  the headless export. It answers "does this column mapping make sense?" before the scene is
  touched; it is a preview of the data's shape and the look's palette, not a stand-in for the
  render the Apply button draws.

CONTRACT §1.1: ``draw`` never mutates the scene. Apply and "Try with sample data" go through
:meth:`Panel.push_command`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import fuzzy, gallery_assets, recipes, styles, widgets
from ..background import BackgroundJob
from ..datasets import Column, DataSet
from ..history import Command
from .base import Panel
from .thumbs import Thumbnails

try:
    from imgui_bundle import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - imgui is a hard dependency in CI
    IMGUI_AVAILABLE = False
    imgui = None

logger = logging.getLogger(__name__)

__all__ = ["GalleryPanel"]

#: Narrowest a card is allowed to get before the grid drops a column.
CARD_MIN_WIDTH = 158.0
CARD_GAP = 8.0
PAD = 6.0
#: Thumbnails are 8:5; the card's picture box has the same shape so nothing is cropped.
THUMB_ASPECT = gallery_assets.THUMB_SIZE[0] / gallery_assets.THUMB_SIZE[1]

#: Pixel size the live preview is rendered at (8:5, like every thumbnail).
PREVIEW_RENDER = (1000, 625)
#: Past this many rows a recipe's preview is a thinned copy of the data (every k-th row). The
#: headless export draws through matplotlib, and 10 million points is minutes, not a preview.
PREVIEW_ROW_LIMIT = 150_000
#: Kinds whose rows form a lattice or an ordered structure that thinning would tear apart; for
#: these a dataset past the limit simply gets no preview (Apply still draws it).
_NO_THINNING = {"surface3d", "wireframe3d", "bar3d", "quiver3d", "image"}

_NONE_CHOICE = "(choose a column)"


def search_score(recipe: recipes.Recipe, query: str) -> int:
    """How well ``query`` matches ``recipe``: 0 = not at all, higher = better.

    A literal substring is what a person typing a word means, so those rank first and by where
    they hit: the name, then the category/kind/tags, then the description. Only the *short*
    fields (the name and the tags) fall back to the palette's subsequence matcher, for the
    abbreviation case ("hxbn"). Running that matcher over the whole description, as a palette
    does over a short title, is how a search for "heat" ends up matching nearly every recipe:
    h, e, a and t all occur, in order, somewhere in two lines of prose.
    """
    needle = query.strip().lower()
    if not needle:
        return 1
    name = recipe.name.lower()
    if needle in name:
        return 400 + (50 if name.startswith(needle) else 0)
    labels = " ".join((recipe.category, recipe.kind) + recipe.tags).lower()
    if needle in labels:
        return 300
    if needle in recipe.description.lower():
        return 200
    if len(needle) >= 2:
        best = 0
        for text in (recipe.name,) + recipe.tags:
            hit = fuzzy.score(needle, text)
            if hit is not None:
                best = max(best, hit[0])
        return min(best, 150)
    return 0


@dataclass
class _Preview:
    """The state of the live preview: which request it answers, and its job/result."""

    key: Optional[tuple] = None
    job: Optional[BackgroundJob] = None
    started: float = 0.0
    error: Optional[str] = None
    ready: bool = False
    note: str = ""


class GalleryPanel(Panel):
    """The recipe gallery. See the module docstring."""

    title = "Gallery"
    icon = "grid"
    default_open = False

    #: Render previews on a worker thread. A test (or a host that cannot thread) sets it False
    #: and gets the render inline.
    async_preview = True

    def __init__(self, ws: Any) -> None:
        super().__init__(ws)
        self._thumbs = Thumbnails(lambda: getattr(self.plot, "window", None))
        self._category: Optional[str] = None
        self._query = ""
        self._selected: Optional[str] = None
        self._dataset: Optional[str] = None
        self._assignment: Dict[str, Optional[str]] = {}
        self._look: str = recipes.RECIPE_LOOK
        self._status = ""
        self._scroll_top = False
        self._problem_key: Optional[tuple] = None
        self._problem: Optional[str] = None
        self._preview = _Preview()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def selected_recipe(self) -> Optional[recipes.Recipe]:
        """The recipe whose detail is open, or ``None``."""
        if self._selected is None:
            return None
        try:
            return recipes.get_recipe(self._selected)
        except KeyError:
            return None

    def select(self, key: Optional[str]) -> None:
        """Open ``key``'s detail (or close it with ``None``) and guess its column assignment."""
        if key is not None:
            recipes.get_recipe(key)  # an unknown key fails here, not on the next frame
        self._selected = key
        self._status = ""
        self._scroll_top = key is not None
        self._reassign()
        self._invalidate()

    def choose_dataset(self, name: Optional[str]) -> None:
        """Make ``name`` the dataset the recipe reads, and re-guess its columns."""
        self._dataset = name
        self._reassign()
        self._invalidate()

    def assign(self, slot_key: str, column: Optional[str]) -> None:
        """Put ``column`` in ``slot_key`` (``None`` empties the slot)."""
        self._assignment[slot_key] = column
        self._invalidate()

    def set_look(self, look: str) -> None:
        self._look = look
        self._invalidate()

    def _ensure_dataset(self) -> None:
        """Keep ``_dataset`` pointing at a dataset that exists (the first, if it was lost)."""
        names = self.store.names()
        if self._dataset not in names:
            self._dataset = names[0] if names else None

    def _current_dataset(self) -> Optional[DataSet]:
        """The chosen dataset; falls back to the first one so a fresh panel is usable at once."""
        previous = self._dataset
        self._ensure_dataset()
        if self._dataset != previous:
            self._reassign()
        return self.store.get(self._dataset) if self._dataset else None

    def _reassign(self) -> None:
        self._ensure_dataset()
        recipe = self.selected_recipe
        ds = self.store.get(self._dataset) if self._dataset else None
        if recipe is None or ds is None:
            self._assignment = {}
            return
        self._assignment = recipes.assign_columns(recipe, ds.column_names())

    def _invalidate(self) -> None:
        self._problem_key = None
        self._preview.key = None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _build_data(self, recipe: recipes.Recipe, ds: DataSet) -> Dict[str, np.ndarray]:
        """The arrays a recipe draws, resolved from the current column assignment."""
        table = {column.name: column.values for column in ds.columns}
        return recipes.slot_arrays(recipe, table, self._assignment, ds.n_rows())

    def _labels(self, recipe: recipes.Recipe, ds: DataSet) -> Dict[str, str]:
        """Slot key -> the column name it came from, for the axis titles."""
        return recipes.slot_labels(recipe, self._assignment)

    def _problem_for(self, recipe: recipes.Recipe, ds: Optional[DataSet]) -> Optional[str]:
        """Why the recipe cannot be drawn from the current choices, or ``None``. Memoised.

        Validation walks the data (finite checks, grid detection), so it runs when a choice
        changes, never once per frame.
        """
        if ds is None:
            return "Choose a dataset, or press Try with sample data."
        key = (
            recipe.key,
            ds.name,
            ds.n_rows(),
            ds.n_cols(),
            tuple(sorted((k, v) for k, v in self._assignment.items() if v)),
        )
        if key != self._problem_key:
            self._problem = recipes.validate_data(recipe, self._build_data(recipe, ds))
            self._problem_key = key
        return self._problem

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def apply_selected(self) -> bool:
        """Queue the selected recipe, drawn from the chosen dataset. False if it cannot be."""
        recipe = self.selected_recipe
        ds = self._current_dataset()
        if recipe is None or ds is None or self._problem_for(recipe, ds) is not None:
            return False
        data = {k: np.array(v, copy=True) for k, v in self._build_data(recipe, ds).items()}
        self.push_command(
            recipes.recipe_command(
                self.plot,
                self.hud,
                recipe,
                data,
                name=ds.name,
                look=self._look,
                labels=self._labels(recipe, ds),
            )
        )
        self._status = f"Drew {recipe.name} from {ds.name!r}"
        return True

    def try_sample(self) -> bool:
        """Create the selected recipe's sample dataset and draw the recipe from it, as one step."""
        recipe = self.selected_recipe
        if recipe is None or recipe.sample is None:
            return False
        table = recipes.sample_table(recipe)
        dataset = DataSet(recipe.name, [Column(name, values) for name, values in table.items()])
        dataset.source = "sample"
        store = self.store
        inner: Dict[str, Command] = {}
        plot, hud, look = self.plot, self.hud, self._look

        def do() -> None:
            store.add(dataset)
            data = recipes.sample_data(recipe)
            if "command" not in inner:
                inner["command"] = recipes.recipe_command(
                    plot,
                    hud,
                    recipe,
                    data,
                    name=dataset.name,
                    look=look,
                    labels=recipes.sample_labels(recipe),
                    title=recipe.title,
                )
            inner["command"].do()

        def undo() -> None:
            if "command" in inner:
                inner["command"].undo()
            store.remove(dataset)

        self.push_command(Command(label=f"Try {recipe.name} with sample data", do=do, undo=undo))
        # The panel's own form follows the new dataset once the queue has run; naming it now
        # means the form already points at it on the next frame.
        self._dataset = dataset.name
        self._assignment = {
            slot: column for slot, column in recipes.sample_assignment(recipe).items()
        }
        self._invalidate()
        self._status = f"Added sample data {dataset.name!r} and drew {recipe.name}"
        return True

    # ------------------------------------------------------------------
    # Live preview
    # ------------------------------------------------------------------

    def _thin(
        self, recipe: recipes.Recipe, data: Dict[str, np.ndarray]
    ) -> Optional[Dict[str, np.ndarray]]:
        """``data`` thinned for a preview, or ``None`` if this recipe cannot be thinned."""
        rows = max((np.shape(v)[0] for v in data.values()), default=0)
        if rows <= PREVIEW_ROW_LIMIT:
            return {k: np.array(v, copy=True) for k, v in data.items()}
        if recipe.kind in _NO_THINNING:
            return None
        stride = int(np.ceil(rows / PREVIEW_ROW_LIMIT))
        return {k: np.array(v[::stride], copy=True) for k, v in data.items()}

    def _request_preview(self, recipe: recipes.Recipe, ds: Optional[DataSet]) -> None:
        """Start (or keep) the background render for the current choices."""
        state = self._preview
        if ds is None or not self._thumbs.available or self._problem_for(recipe, ds) is not None:
            return
        key = (
            recipe.key,
            ds.name,
            ds.n_rows(),
            self._look,
            tuple(sorted((k, v) for k, v in self._assignment.items() if v)),
        )
        if key == state.key:
            return
        if state.job is not None:
            state.job.cancel()
        state.key, state.job, state.error, state.ready, state.note = key, None, None, False, ""
        data = self._thin(recipe, self._build_data(recipe, ds))
        if data is None:
            state.note = "Too many rows to preview this recipe; Apply still draws all of them."
            return
        labels, look = self._labels(recipe, ds), self._look

        def render() -> np.ndarray:
            # Rendered at figure scale and shown reduced: the export sizes its fonts for a
            # full-size figure, so a 480-pixel render would put a 12-point title on a
            # postage stamp. Drawn large and scaled down by the panel, the text keeps the
            # proportions it has in an exported figure.
            return recipes.render_preview_rgb(
                recipe,
                data,
                look=look,
                labels=labels,
                width=PREVIEW_RENDER[0],
                height=PREVIEW_RENDER[1],
            )

        state.started = time.monotonic()
        if self.async_preview:
            state.job = BackgroundJob(render)
        else:
            try:
                self._thumbs.put(("preview",), render())
                state.ready = True
            except Exception as exc:  # noqa: BLE001 - a preview must never break the panel
                state.error = str(exc)

    def _poll_preview(self) -> None:
        state = self._preview
        if state.job is None:
            return
        status = state.job.poll()
        if status.state == "running":
            return
        state.job = None
        if status.state == "done" and status.result is not None:
            self._thumbs.put(("preview",), status.result)
            state.ready = True
        elif status.state == "error":
            state.error = str(status.error)
            logger.debug("gallery preview failed: %s", status.error)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self) -> None:
        if not IMGUI_AVAILABLE:
            return
        self._thumbs.begin_frame()
        self._poll_preview()
        if self._scroll_top:
            imgui.set_scroll_y(0.0)
            self._scroll_top = False

        imgui.text_wrapped(
            "Pick a recipe, say which column goes where, and the plot is drawn on your scene "
            "in the look it was designed in. One undo step takes it back."
        )
        imgui.spacing()
        self._draw_dataset_row()
        self._draw_filters()
        recipe = self.selected_recipe
        if recipe is not None:
            imgui.separator()
            self._draw_detail(recipe)
        imgui.separator()
        self._draw_cards()

    # -- dataset ----------------------------------------------------------------

    def _draw_dataset_row(self) -> None:
        names = self.store.names()
        if not names:
            imgui.push_text_wrap_pos(0.0)
            imgui.text_disabled(
                "No datasets yet. Paste a table into the Data panel (Ctrl+V), or open a "
                "recipe and press Try with sample data."
            )
            imgui.pop_text_wrap_pos()
            return
        self._current_dataset()
        items = list(names)
        index = items.index(self._dataset) if self._dataset in items else 0
        imgui.set_next_item_width(max(120.0, imgui.get_content_region_avail().x - 70.0))
        changed, index = imgui.combo("Data", index, items)
        if changed:
            self.choose_dataset(items[index])

    # -- filters ----------------------------------------------------------------

    def _draw_filters(self) -> None:
        imgui.set_next_item_width(-1.0)
        changed, text = imgui.input_text_with_hint(
            "##gallery_search", "Search recipes...", self._query
        )
        if changed:
            self._query = text
        chips: List[Tuple[Optional[str], str]] = [(None, "All")] + [
            (c, c) for c in recipes.CATEGORIES
        ]
        style = imgui.get_style()
        right_edge = imgui.get_cursor_screen_pos()[0] + float(imgui.get_content_region_avail()[0])
        for position, (category, label) in enumerate(chips):
            chip_w = float(imgui.calc_text_size(label)[0]) + 2.0 * float(style.frame_padding[0])
            if position:
                # Stay on the line only if the chip fits; otherwise it starts the next row,
                # rather than running off the panel's right edge where it cannot be clicked.
                if (
                    imgui.get_item_rect_max()[0] + float(style.item_spacing[0]) + chip_w
                    <= right_edge
                ):
                    imgui.same_line()
            active = self._category == category
            if active:
                imgui.push_style_color(
                    imgui.Col_.button, imgui.get_style_color_vec4(imgui.Col_.button_active)
                )
            if imgui.small_button(f"{label}##cat_{label}"):
                self._category = category
            if active:
                imgui.pop_style_color(1)
        imgui.spacing()

    def _visible_recipes(self) -> List[recipes.Recipe]:
        """Recipes passing the category chip and the search box, best match first."""
        pool = [r for r in recipes.RECIPES if self._category in (None, r.category)]
        query = self._query.strip()
        if not query:
            return pool
        scored = [(search_score(recipe, query), recipe) for recipe in pool]
        return [recipe for score, recipe in sorted(scored, key=lambda p: -p[0]) if score > 0]

    # -- detail -----------------------------------------------------------------

    def _draw_detail(self, recipe: recipes.Recipe) -> None:
        ds = self._current_dataset()
        draw_list = imgui.get_window_draw_list()

        avail = float(imgui.get_content_region_avail().x)
        width = max(160.0, avail)
        height = width / THUMB_ASPECT
        x, y = imgui.get_cursor_screen_pos()
        imgui.dummy((width, height))
        texture, caption = self._detail_texture(recipe, ds)
        Thumbnails.draw(
            draw_list,
            texture,
            (x, y, x + width, y + height),
            fit=False,
            busy=self._preview.job is not None,
        )
        draw_list.add_rect(
            (x, y), (x + width, y + height), imgui.get_color_u32(imgui.Col_.border), rounding=4.0
        )
        if self._preview.job is not None:
            imgui.set_cursor_screen_pos((x + 8.0, y + 8.0))
            widgets.spinner("##gallery_preview_spin")
            imgui.set_cursor_screen_pos((x, y + height))
            imgui.dummy((0.0, 0.0))
        imgui.push_text_wrap_pos(0.0)
        imgui.text_disabled(caption)
        imgui.pop_text_wrap_pos()

        imgui.text(recipe.name)
        imgui.same_line()
        imgui.text_disabled(f"{recipe.dimension} / {recipe.kind}")
        imgui.text_wrapped(recipe.description)
        if recipe.example:
            imgui.text_disabled(f"From examples/gallery/{recipe.example}")
        imgui.spacing()

        self._draw_slots(recipe, ds)
        self._draw_look(recipe)
        imgui.spacing()

        problem = self._problem_for(recipe, ds)
        if ds is not None:
            self._request_preview(recipe, ds)
        if problem is not None and ds is not None:
            widgets.error_box(problem)
        elif self._preview.error:
            imgui.text_disabled("Preview unavailable: " + self._preview.error)
        elif self._preview.note:
            imgui.text_disabled(self._preview.note)

        imgui.begin_disabled(problem is not None)
        if imgui.button("Apply to scene", (max(120.0, avail * 0.5 - 4.0), 0.0)):
            self.apply_selected()
        imgui.end_disabled()
        if problem is not None and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(problem)
        if recipe.sample is not None:
            imgui.same_line()
            if imgui.button("Try with sample data", (-1.0, 0.0)):
                self.try_sample()
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Adds this recipe's example table to the Data panel and draws the recipe "
                    "from it, so you can see what it does before using your own."
                )
        if self._status:
            imgui.text_disabled(self._status)

    def _detail_texture(self, recipe: recipes.Recipe, ds: Optional[DataSet]) -> Tuple[Any, str]:
        """The big image: the live preview of the user's data if it is ready, else the card's."""
        if (
            self._preview.ready
            and self._preview.key is not None
            and self._preview.key[0] == recipe.key
        ):
            texture = self._thumbs.peek(("preview",))
            if texture is not None:
                return texture, "Preview of your data. Apply draws the real render."
        stock = self._card_texture(recipe)
        return stock, "Example render on the recipe's own sample data"

    def _draw_slots(self, recipe: recipes.Recipe, ds: Optional[DataSet]) -> None:
        imgui.text("Columns")
        if ds is None:
            imgui.text_disabled("Choose a dataset above to map its columns.")
            return
        names = ds.column_names()
        items = [_NONE_CHOICE, recipes.INDEX] + names
        for slot in recipe.slots:
            if slot.multi:
                imgui.text(
                    f"{slot.label}: every column ({ds.n_cols()} columns x {ds.n_rows()} rows)"
                )
                continue
            current = self._assignment.get(slot.key)
            index = items.index(current) if current in items else 0
            label = slot.label + ("" if slot.required else " (optional)")
            imgui.set_next_item_width(max(100.0, imgui.get_content_region_avail().x * 0.55))
            changed, index = imgui.combo(f"{label}##slot_{slot.key}", index, items)
            if slot.hint:
                imgui.same_line()
                widgets.help_marker(slot.hint)
            if changed:
                self.assign(slot.key, None if index == 0 else items[index])

    def _draw_look(self, recipe: recipes.Recipe) -> None:
        own = styles.get_style(recipe.style).name
        choices: List[Tuple[str, str]] = [
            (recipes.RECIPE_LOOK, f"Recipe's look ({own})"),
            (recipes.KEEP_LOOK, "Keep my current look"),
        ] + [(s.key, s.name) for s in styles.STYLES]
        labels = [label for _, label in choices]
        keys = [key for key, _ in choices]
        index = keys.index(self._look) if self._look in keys else 0
        imgui.set_next_item_width(max(100.0, imgui.get_content_region_avail().x * 0.55))
        changed, index = imgui.combo("Look##gallery_look", index, labels)
        imgui.same_line()
        widgets.help_marker(
            "The page (background, grid, effects) and the colours of the plot it draws. "
            "'Keep my current look' draws the recipe in the style the figure already has."
        )
        if changed:
            self.set_look(keys[index])

    # -- cards ------------------------------------------------------------------

    def _card_texture(self, recipe: recipes.Recipe) -> Any:
        return self._thumbs.get(
            ("recipe", recipe.key),
            lambda: gallery_assets.load_rgba(gallery_assets.recipe_asset(recipe.key)),
        )

    def _draw_cards(self) -> None:
        visible = self._visible_recipes()
        if not visible:
            imgui.text_disabled("No recipe matches that search.")
            return
        avail = float(imgui.get_content_region_avail().x)
        columns = max(1, int((avail + CARD_GAP) // (CARD_MIN_WIDTH + CARD_GAP)))
        card_w = (avail - CARD_GAP * (columns - 1)) / columns
        for position, recipe in enumerate(visible):
            if position % columns:
                imgui.same_line(0.0, CARD_GAP)
            if self._draw_card(recipe, card_w):
                self.select(recipe.key)

    def _draw_card(self, recipe: recipes.Recipe, width: float) -> bool:
        line_h = float(imgui.get_text_line_height())
        image_h = (width - 2 * PAD) / THUMB_ASPECT
        height = PAD + image_h + 4.0 + 2 * line_h + PAD
        selected = recipe.key == self._selected

        draw_list = imgui.get_window_draw_list()
        x, y = imgui.get_cursor_screen_pos()
        imgui.push_id(recipe.key)
        clicked = imgui.invisible_button("##card", (width, height))
        hovered = imgui.is_item_hovered()
        imgui.pop_id()

        texture = self._card_texture(recipe)
        Thumbnails.draw(
            draw_list,
            texture,
            (x + PAD, y + PAD, x + width - PAD, y + PAD + image_h),
            fit=False,
            busy=self._thumbs.available and texture is None,
        )
        text_y = y + PAD + image_h + 4.0
        draw_list.push_clip_rect((x, y), (x + width, y + height), True)
        draw_list.add_text((x + PAD, text_y), imgui.get_color_u32(imgui.Col_.text), recipe.name)
        draw_list.add_text(
            (x + PAD, text_y + line_h),
            imgui.get_color_u32(imgui.Col_.text_disabled),
            f"{recipe.dimension} / {recipe.category.lower()}",
        )
        draw_list.pop_clip_rect()

        if selected:
            border, thickness = imgui.get_color_u32(imgui.Col_.slider_grab), 2.0
        elif hovered:
            border, thickness = imgui.get_color_u32(imgui.Col_.text, 0.45), 1.0
        else:
            border, thickness = imgui.get_color_u32(imgui.Col_.border), 1.0
        draw_list.add_rect(
            (x, y), (x + width, y + height), border, rounding=5.0, thickness=thickness
        )
        if hovered:
            imgui.set_tooltip(recipe.description)
        return bool(clicked)
