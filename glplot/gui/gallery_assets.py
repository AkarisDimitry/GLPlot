"""Where the gallery's thumbnails live, and how to read one. Pure: no imgui, no GL.

The Style panel and the Gallery panel show *real renders*, captured from the OpenGL window by
``tools/build_gallery_assets.py`` and committed under ``glplot/gui/assets/gallery``:

* ``styles/<style key>__<scene>.webp`` -- every preset drawn on each of five sample scenes
  (:data:`STYLE_SCENES`), so a card can answer "what does *my kind of plot* look like in
  this?" rather than only "what is its palette?";
* ``recipes/<recipe key>.webp`` -- every recipe drawn from its own sample table.

Everything here degrades to "no image": a missing file, an unreadable one, or a Pillow without
WebP all answer ``None``, and the panels fall back to the drawn thumbnails they had before
images existed. A GUI that refuses to open because a thumbnail is missing would be a bad
trade for a thumbnail.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ASSET_ROOT",
    "SCENE_LABELS",
    "STYLE_SCENES",
    "THUMB_SIZE",
    "load_rgba",
    "recipe_asset",
    "style_asset",
]

#: Folder the thumbnails live in, next to this module so a wheel carries them.
ASSET_ROOT: Path = Path(__file__).resolve().parent / "assets" / "gallery"

#: Pixel size of every stored thumbnail (8:5, the window the captures are taken from).
THUMB_SIZE: Tuple[int, int] = (400, 250)

#: The sample scenes a style is rendered on, in the order the panel offers them. Kept in step
#: with ``tools/build_gallery_assets.SCENES`` by a test, so neither can gain a scene alone.
STYLE_SCENES: Tuple[str, ...] = ("lines", "scatter", "bars", "field", "surface")

#: What the panel calls each scene.
SCENE_LABELS: Dict[str, str] = {
    "lines": "Lines",
    "scatter": "Scatter",
    "bars": "Bars",
    "field": "Heat map",
    "surface": "3D surface",
}


def style_asset(key: str, scene: str) -> Optional[Path]:
    """The thumbnail of style ``key`` on ``scene``, or ``None`` if it was not generated."""
    if scene not in STYLE_SCENES:
        return None
    path = ASSET_ROOT / "styles" / f"{key}__{scene}.webp"
    return path if path.is_file() else None


def recipe_asset(key: str) -> Optional[Path]:
    """The thumbnail of recipe ``key``, or ``None`` if it was not generated."""
    path = ASSET_ROOT / "recipes" / f"{key}.webp"
    return path if path.is_file() else None


def load_rgba(path: Optional[Path]) -> Optional[np.ndarray]:
    """Decode ``path`` to an ``(H, W, 4)`` uint8 array, or ``None`` if it cannot be read.

    Never raises: a corrupt or unsupported file is logged once at debug level and treated as
    absent, so one bad thumbnail costs one card its picture, not the panel its draw call.
    """
    if path is None:
        return None
    try:
        from PIL import Image

        with Image.open(path) as handle:
            return np.asarray(handle.convert("RGBA"), dtype=np.uint8).copy()
    except Exception as exc:  # noqa: BLE001 - any decode failure means "no image"
        logger.debug("could not read gallery thumbnail %s: %s", path, exc)
        return None
