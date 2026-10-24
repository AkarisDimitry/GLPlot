"""Thumbnails for the panels that show them: a texture cache tied to the figure's window.

The Style panel (a render of every preset) and the Gallery panel (a render of every recipe)
both need the same three things -- a texture per image, uploaded only when a GL context exists,
and drawn into a rectangle the card has already laid out -- so they share this rather than each
growing a copy:

* :class:`Thumbnails` owns a :class:`~glplot.gui.textures.TextureCache` and keeps it honest about
  *which window it belongs to*. A figure with no window gets a cache that cannot upload (every
  lookup answers ``None``, the card draws its placeholder); the first frame a window exists the
  cache is rebuilt with the real GL functions; and if the window is replaced -- closed and
  reopened -- the stale texture ids are forgotten instead of reused.
* :meth:`Thumbnails.draw` paints a texture, or a quiet placeholder when there is none, into the
  rectangle it is given, through the window's draw list so it adds nothing to the layout.

A placeholder is a feature, not a failure mode: it is what a card shows for the few frames it
takes to decode and upload its image, and what it shows for good where there is no GL.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Tuple

import numpy as np

from ..textures import Texture, TextureCache, gl_delete, gl_upload

try:
    from imgui_bundle import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - imgui is a hard dependency in CI
    IMGUI_AVAILABLE = False
    imgui = None

logger = logging.getLogger(__name__)

__all__ = ["Thumbnails", "fit_rect"]


def fit_rect(
    box: Tuple[float, float, float, float], aspect: float
) -> Tuple[float, float, float, float]:
    """The largest ``aspect`` (width / height) rectangle centred in ``box = (x0, y0, x1, y1)``."""
    x0, y0, x1, y1 = box
    width, height = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    if width / height > aspect:
        inner_w = height * aspect
        pad = (width - inner_w) / 2.0
        return (x0 + pad, y0, x1 - pad, y1)
    inner_h = width / aspect
    pad = (height - inner_h) / 2.0
    return (x0, y0 + pad, x1, y1 - pad)


class Thumbnails:
    """A window-aware texture cache plus the one way to draw its images."""

    def __init__(self, window_of: Callable[[], Any], *, budget: int = 3) -> None:
        self._window_of = window_of
        self._budget = budget
        self._window: Any = None
        self._cache = TextureCache(budget=budget)

    # -- lifecycle -------------------------------------------------------------

    def begin_frame(self) -> None:
        """Re-tie the cache to the figure's window, and reset the per-frame upload budget.

        Called at the top of the panel's ``draw``. The only place the cache is rebuilt, so the
        decision "can this panel upload textures at all" is made once per frame in one spot.
        """
        window = self._window_of()
        if window is not self._window:
            # The old textures belong to a context that is gone (or never existed): drop the
            # entries without deleting them, then start a cache that matches what exists now.
            self._cache.forget()
            self._window = window
            if window is None:
                self._cache = TextureCache(budget=self._budget)
            else:
                self._cache = TextureCache(gl_upload, gl_delete, budget=self._budget)
        self._cache.begin_frame()

    @property
    def available(self) -> bool:
        """Whether this panel can show images right now (a window exists)."""
        return self._cache.enabled

    # -- access ----------------------------------------------------------------

    def get(self, key: Any, loader: Callable[[], Optional[np.ndarray]]) -> Optional[Texture]:
        """The texture for ``key`` (built from ``loader`` on first use), or ``None``."""
        return self._cache.get(key, loader)

    def put(self, key: Any, rgba: np.ndarray) -> Optional[Texture]:
        """Replace ``key``'s texture with ``rgba``."""
        return self._cache.put(key, rgba)

    def peek(self, key: Any) -> Optional[Texture]:
        return self._cache.peek(key)

    def drop(self, key: Any) -> None:
        self._cache.drop(key)

    def release(self) -> None:
        self._cache.release()

    # -- drawing ---------------------------------------------------------------

    @staticmethod
    def draw(
        draw_list: Any,
        texture: Optional[Texture],
        box: Tuple[float, float, float, float],
        *,
        rounding: float = 4.0,
        fit: bool = True,
        busy: bool = False,
    ) -> None:
        """Paint ``texture`` into ``box = (x0, y0, x1, y1)``, or a placeholder if it is ``None``.

        With ``fit`` the image keeps its aspect ratio, centred in the box; without it the box is
        filled exactly (the right call when the box already has the thumbnails' 8:5 shape).
        ``busy`` marks a placeholder as "coming" rather than "absent" with a slightly brighter
        wash, so a card waiting on its render is not mistaken for one that has none.
        """
        if not IMGUI_AVAILABLE or draw_list is None:
            return
        x0, y0, x1, y1 = box
        if texture is None:
            wash = 0.16 if busy else 0.09
            draw_list.add_rect_filled(
                (x0, y0),
                (x1, y1),
                imgui.get_color_u32((0.5, 0.5, 0.55, wash)),
                rounding=rounding,
            )
            return
        if fit:
            x0, y0, x1, y1 = fit_rect(box, texture.width / max(texture.height, 1))
        draw_list.add_image_rounded(
            imgui.ImTextureRef(int(texture.id)),
            (x0, y0),
            (x1, y1),
            (0.0, 0.0),
            (1.0, 1.0),
            imgui.get_color_u32((1.0, 1.0, 1.0, 1.0)),
            float(rounding),
        )
