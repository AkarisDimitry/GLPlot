"""Images in imgui: a small texture cache that never touches GL unless it is safe to.

imgui draws an image from an OpenGL texture id, so showing a thumbnail means creating a
texture on the GL thread, once, and remembering it. :class:`TextureCache` does exactly that and
nothing more: it maps a key to a texture, builds a missing one from a loader function the first
time it is asked, and spreads a burst of new textures over several frames so opening a panel of
two dozen cards does not hitch.

**GL is injected, and off by default.** The upload and delete functions are constructor
arguments. The panels hand in the real ones only when their figure owns a live window; a panel
under test (or one whose figure has no window yet) gets a cache whose uploader answers ``None``,
which every caller already treats as "draw the placeholder". That keeps a code path that makes
GL calls from running anywhere a GL context is not guaranteed -- the test-suite above all, where
live GL calls are deliberately never made (see the notes on why: they pass in isolation and fail
non-deterministically in the full run).

``Texture`` ids are only valid for the GL context that created them. A window that is closed and
reopened has a new context and the old ids are garbage, so :meth:`TextureCache.forget` drops the
bookkeeping *without* deleting (deleting an id from a dead context is itself an error).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, NamedTuple, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["Texture", "TextureCache", "gl_delete", "gl_upload"]


class Texture(NamedTuple):
    """A GL texture and its pixel size."""

    id: int
    width: int
    height: int


Uploader = Callable[[np.ndarray], Optional[int]]
Deleter = Callable[[int], None]


def gl_upload(rgba: np.ndarray) -> Optional[int]:
    """Create an RGBA8 texture from an ``(H, W, 4)`` uint8 array; return its id, or ``None``.

    **GL thread only**, with a context current. The previous texture binding is restored, so
    a draw callback that uploads in the middle of a frame leaves the engine's state alone.
    """
    try:
        from OpenGL import GL
    except Exception:  # noqa: BLE001 - no PyOpenGL: no textures
        return None
    try:
        height, width = int(rgba.shape[0]), int(rgba.shape[1])
        data = np.ascontiguousarray(rgba, dtype=np.uint8)
        previous = int(GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_2D))
        previous_align = int(GL.glGetIntegerv(GL.GL_UNPACK_ALIGNMENT))
        texture = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA8,
            width,
            height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            data,
        )
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, previous_align)
        GL.glBindTexture(GL.GL_TEXTURE_2D, previous)
        return texture
    except Exception as exc:  # noqa: BLE001 - a GL error must not take the frame down
        logger.warning("gallery texture upload failed: %s", exc)
        return None


def gl_delete(texture_id: int) -> None:
    """Delete a texture created by :func:`gl_upload`. **GL thread only.**"""
    try:
        from OpenGL import GL

        GL.glDeleteTextures([int(texture_id)])
    except Exception as exc:  # noqa: BLE001
        logger.debug("gallery texture delete failed: %s", exc)


def _no_upload(rgba: np.ndarray) -> Optional[int]:
    return None


def _no_delete(texture_id: int) -> None:
    return None


class TextureCache:
    """``key -> Texture``, built lazily and at most ``budget`` new textures per frame."""

    def __init__(
        self,
        uploader: Optional[Uploader] = None,
        deleter: Optional[Deleter] = None,
        *,
        budget: int = 3,
    ) -> None:
        self._upload: Uploader = uploader or _no_upload
        self._delete: Deleter = deleter or _no_delete
        self._budget_per_frame = max(1, int(budget))
        self._budget = self._budget_per_frame
        self._textures: Dict[Any, Texture] = {}
        #: Keys whose loader returned nothing or whose upload failed: not retried every frame.
        self._failed: Set[Any] = set()

    @property
    def enabled(self) -> bool:
        """Whether this cache can ever produce a texture (it was given a real uploader)."""
        return self._upload is not _no_upload

    def begin_frame(self) -> None:
        """Reset the per-frame budget. Call once at the top of a panel's ``draw``."""
        self._budget = self._budget_per_frame

    def peek(self, key: Any) -> Optional[Texture]:
        """The cached texture for ``key``, without building one."""
        return self._textures.get(key)

    def get(self, key: Any, loader: Callable[[], Optional[np.ndarray]]) -> Optional[Texture]:
        """The texture for ``key``, building it from ``loader()`` if there is budget left.

        ``None`` means "not available *now*" -- over this frame's budget, a failed load, or a
        cache with no uploader -- and the caller draws a placeholder. A failure is remembered
        so a missing thumbnail is not re-read from disk sixty times a second.
        """
        texture = self._textures.get(key)
        if texture is not None:
            return texture
        if key in self._failed or not self.enabled or self._budget <= 0:
            return None
        self._budget -= 1
        rgba = loader()
        if rgba is None:
            self._failed.add(key)
            return None
        return self.put(key, rgba)

    def put(self, key: Any, rgba: np.ndarray) -> Optional[Texture]:
        """Upload ``rgba`` as ``key``'s texture, replacing (and deleting) any previous one."""
        if not self.enabled:
            return None
        array = np.asarray(rgba)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            self._failed.add(key)
            return None
        if array.shape[2] == 3:
            alpha = np.full(array.shape[:2] + (1,), 255, dtype=np.uint8)
            array = np.concatenate([array.astype(np.uint8), alpha], axis=2)
        texture_id = self._upload(array)
        if texture_id is None:
            self._failed.add(key)
            return None
        self.drop(key)
        texture = Texture(int(texture_id), int(array.shape[1]), int(array.shape[0]))
        self._textures[key] = texture
        self._failed.discard(key)
        return texture

    def drop(self, key: Any) -> None:
        """Delete ``key``'s texture, if it has one."""
        texture = self._textures.pop(key, None)
        if texture is not None:
            self._delete(texture.id)

    def release(self) -> None:
        """Delete every texture. **GL thread, context still current.**"""
        for key in list(self._textures):
            self.drop(key)
        self._failed.clear()

    def forget(self) -> None:
        """Drop every entry *without* deleting: the GL context that owned them is gone."""
        self._textures.clear()
        self._failed.clear()

    def __len__(self) -> int:
        return len(self._textures)
