from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np
from OpenGL.GL import *

from ..renderers.base import GLOffscreenTarget
from ..utils.gl_utils import link_program
from ..utils.shaders import (
    PICKING_LINES_FS,
    PICKING_LINES_VS,
    PICKING_PATCH_FS,
    PICKING_PATCH_VS,
    PICKING_SCATTER_FS,
    PICKING_SCATTER_VS,
    PICKING_STRIP_FS,
    PICKING_STRIP_VS,
)

if TYPE_CHECKING:
    from ..core import SceneData
    from ..options import EngineOptions


def _as_int32(raw: Any) -> np.ndarray:
    """Flatten a glReadPixels GL_RED_INTEGER/GL_INT result to a 1-D int32 array.

    PyOpenGL hands back a shaped ndarray for multi-pixel reads and a bytes-like
    buffer for some single-pixel ones; normalise both.
    """
    if isinstance(raw, np.ndarray):
        return np.ascontiguousarray(raw).ravel().astype(np.int32, copy=False)
    return np.frombuffer(raw, dtype=np.int32)


class PickingManager:
    def __init__(self, options: EngineOptions) -> None:
        self.options = options
        self.target = GLOffscreenTarget()
        self.pid_lines = -1
        self.pid_scatter = -1
        self.pid_strip = -1
        self.pid_patch = -1

    def initialize(self, fb_width: int, fb_height: int) -> None:
        self.pid_lines = link_program(PICKING_LINES_VS, PICKING_LINES_FS)
        self.pid_scatter = link_program(PICKING_SCATTER_VS, PICKING_SCATTER_FS)
        self.pid_strip = link_program(PICKING_STRIP_VS, PICKING_STRIP_FS)
        self.pid_patch = link_program(PICKING_PATCH_VS, PICKING_PATCH_FS)
        self.rebuild_target(fb_width, fb_height)

    def rebuild_target(self, fb_width: int, fb_height: int) -> None:
        if self.target.fbo:
            glDeleteFramebuffers(1, [self.target.fbo])
            glDeleteTextures(1, [self.target.tex])

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        # We store 32-bit Integer IDs
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_R32I, fb_width, fb_height, 0, GL_RED_INTEGER, GL_INT, None
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print(f"Picking Framebuffer incomplete: {status}")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self.target = GLOffscreenTarget(fbo=fbo, tex=tex, width=fb_width, height=fb_height)

    def draw_pick_scene(
        self,
        scene: SceneData,
        buffers: Any,
        mvp: np.ndarray,
        window: Tuple[float, float, float, float],
    ) -> None:
        glBindFramebuffer(GL_FRAMEBUFFER, self.target.fbo)
        glViewport(0, 0, self.target.width, self.target.height)

        # 1. State for picking: no blending, no multisampling
        glDisable(GL_BLEND)
        glDisable(GL_MULTISAMPLE)

        # Clear with 0 (no object) using integer clear
        glClearBufferiv(GL_COLOR, 0, np.array([0], dtype=np.int32))

        current_offset = 0
        for i, layer in enumerate(scene.layers):
            if not layer.style.visible:
                continue

            # IDs are 1-based, 0 is "nothing"
            current_offset + 1

            if layer.layer_type == "line_family":
                if layer.ab is not None and hasattr(layer, "_gl"):
                    glUseProgram(self.pid_lines)
                    glUniformMatrix4fv(
                        glGetUniformLocation(self.pid_lines, "u_mvp"), 1, GL_FALSE, mvp
                    )
                    glUniform2f(glGetUniformLocation(self.pid_lines, "u_xrange"), *layer.x_range)
                    glUniform4f(glGetUniformLocation(self.pid_lines, "u_window"), *window)
                    glUniform2f(
                        glGetUniformLocation(self.pid_lines, "u_layer_offset"), *layer.translation
                    )
                    glUniform1i(glGetUniformLocation(self.pid_lines, "u_id_offset"), current_offset)
                    layer._gl.render(len(layer.ab))
                    current_offset += len(layer.ab)

            elif layer.layer_type == "scatter":
                if hasattr(layer, "_gl"):
                    glUseProgram(self.pid_scatter)
                    glEnable(GL_PROGRAM_POINT_SIZE)
                    glUniformMatrix4fv(
                        glGetUniformLocation(self.pid_scatter, "u_mvp"), 1, GL_FALSE, mvp
                    )
                    glUniform1f(
                        glGetUniformLocation(self.pid_scatter, "u_size"),
                        float(layer.style.point_size),
                    )
                    glUniform1i(
                        glGetUniformLocation(self.pid_scatter, "u_id_offset"), current_offset
                    )
                    glUniform2f(
                        glGetUniformLocation(self.pid_scatter, "u_layer_offset"), *layer.translation
                    )
                    glBindVertexArray(layer._gl.vao)
                    glDrawArrays(GL_POINTS, 0, layer._gl.count)
                    current_offset += layer._gl.count

            elif layer.layer_type == "polyline":
                if hasattr(layer, "_gl"):
                    # Polyline is treated as 1 object for now
                    glUseProgram(self.pid_strip)
                    glUniformMatrix4fv(
                        glGetUniformLocation(self.pid_strip, "u_mvp"), 1, GL_FALSE, mvp
                    )
                    glUniform1i(glGetUniformLocation(self.pid_strip, "u_id"), current_offset + 1)
                    glUniform2f(
                        glGetUniformLocation(self.pid_strip, "u_layer_offset"), *layer.translation
                    )
                    glBindVertexArray(layer._gl.vao)
                    # Polyline uses instanced quads (6 verts per segment)
                    glDrawElements(
                        GL_TRIANGLES, layer._gl.instance_count * 6, GL_UNSIGNED_SHORT, None
                    )
                    current_offset += 1

            elif layer.layer_type == "patch":
                if hasattr(layer, "_gl"):
                    glUseProgram(self.pid_patch)
                    glUniformMatrix4fv(
                        glGetUniformLocation(self.pid_patch, "u_mvp"), 1, GL_FALSE, mvp
                    )
                    glUniform1i(glGetUniformLocation(self.pid_patch, "u_id"), current_offset + 1)
                    glUniform2f(
                        glGetUniformLocation(self.pid_patch, "u_layer_offset"), *layer.translation
                    )
                    glBindVertexArray(layer._gl.vao)
                    mode = GL_TRIANGLE_STRIP
                    if getattr(layer, "mode", "") == "triangles":
                        mode = GL_TRIANGLES
                    if layer._gl.ebo:
                        glDrawElements(mode, layer._gl.count, GL_UNSIGNED_INT, None)
                    else:
                        glDrawArrays(mode, 0, layer._gl.count)
                    current_offset += 1

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)

    @staticmethod
    def _offset_table(scene: SceneData) -> Tuple[List[Any], np.ndarray, np.ndarray]:
        """Per-layer id ranges, mirroring ``draw_pick_scene``'s id assignment exactly.

        Returns ``(layers, starts, counts)``. Id ``v`` belongs to entry ``j`` when
        ``starts[j] < v <= starts[j] + counts[j]``, and its element index is
        ``v - starts[j] - 1`` (ids are 1-based; 0 means "nothing").

        Layers that contribute no ids are omitted -- invisible ones, ones with no
        ``_gl`` (never uploaded, hence never drawn into the pick target), and empty
        ones. That mirrors ``draw_pick_scene``, which only advances its offset for
        layers it actually draws, and it leaves ``starts`` strictly increasing so
        ``searchsorted`` can decode without ambiguity.
        """
        layers: List[Any] = []
        starts: List[int] = []
        counts: List[int] = []
        offset = 0
        for layer in scene.layers:
            if not layer.style.visible:
                continue
            gl = getattr(layer, "_gl", None)
            if gl is None:
                continue

            count = 0
            if layer.layer_type == "line_family":
                count = len(layer.ab) if layer.ab is not None else 0
            elif layer.layer_type == "scatter":
                count = int(gl.count)
            elif layer.layer_type in ("polyline", "patch"):
                count = 1

            if count <= 0:
                continue

            layers.append(layer)
            starts.append(offset)
            counts.append(count)
            offset += count

        return (
            layers,
            np.asarray(starts, dtype=np.int64),
            np.asarray(counts, dtype=np.int64),
        )

    @staticmethod
    def _decode_ids(
        values: np.ndarray, starts: np.ndarray, counts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised id decode: ids -> (owner table index, element index).

        ``owner`` is -1 for ids owned by nobody (0/negative, or past the last
        layer's range). O(len(values) * log(len(starts))) -- never per pixel.
        """
        values = np.asarray(values, dtype=np.int64)
        owner = np.full(values.shape, -1, dtype=np.int64)
        element = np.zeros(values.shape, dtype=np.int64)
        if starts.size == 0:
            return owner, element

        valid = values > 0
        if not np.any(valid):
            return owner, element

        v = values[valid]
        # starts is strictly increasing and starts[0] == 0, so for v >= 1 this
        # lands on the last layer whose start is <= v - 1, i.e. the owner.
        j = np.searchsorted(starts, v - 1, side="right") - 1
        j = np.clip(j, 0, starts.size - 1)
        el = v - starts[j] - 1
        ok = el < counts[j]

        owner[valid] = np.where(ok, j, -1)
        element[valid] = np.where(ok, el, 0)
        return owner, element

    def pick_readback(self, sx: float, sy: float, scene: SceneData) -> Optional[dict]:
        """sx, sy must be in framebuffer/pixel coordinates (already DPR-scaled)."""
        gx = int(sx)
        gy = int(self.target.height - sy)

        if gx < 0 or gx >= self.target.width or gy < 0 or gy >= self.target.height:
            return None

        glBindFramebuffer(GL_FRAMEBUFFER, self.target.fbo)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        pixels = glReadPixels(gx, gy, 1, 1, GL_RED_INTEGER, GL_INT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        val = int(_as_int32(pixels)[0])
        if val <= 0:
            return None

        # Decode through the same global offset table draw_pick_scene encoded with.
        layers, starts, counts = self._offset_table(scene)
        owner, element = self._decode_ids(np.array([val], dtype=np.int64), starts, counts)
        j = int(owner[0])
        if j < 0:
            return None

        layer = layers[j]
        return {
            "type": layer.layer_type,
            "layer_id": layer.layer_id,  # Use stable ID
            "element_idx": int(element[0]),
            "layer": layer,
        }

    def pick_rect_readback(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        scene: SceneData,
        max_pixels: int = 64_000_000,
    ) -> Dict[int, Set[int]]:
        """Rect-pick every element whose id appears inside the screen rect.

        All four coordinates are **framebuffer/pixel** coordinates (already
        DPR-scaled) with y measured DOWN from the top of the window -- the same
        convention as ``pick_readback``'s ``(sx, sy)``. The GL y-flip
        (``gy = target.height - sy``) happens here, which also swaps which corner
        is the bottom one; callers must not pre-flip.

        Corners may be given in any order. The rect is clamped to the target and
        both endpoint pixels are inclusive, so a zero-area rect degenerates to the
        exact single pixel ``pick_readback`` would have sampled.

        Returns ``{layer_id: {element_idx, ...}}``, empty when nothing was hit.
        The caller must have drawn the CURRENT view with ``draw_pick_scene``
        first; this only reads the target back.

        ``max_pixels`` is a backstop against a pathological target size. A
        full-screen Retina readback is a few million ints (~14MB) -- fine on a
        release, but this must never run per frame.
        """
        w_t, h_t = int(self.target.width), int(self.target.height)
        if w_t <= 0 or h_t <= 0 or not self.target.fbo:
            return {}

        # Reject rects entirely outside the target before clamping, otherwise the
        # clamp would drag them onto a real edge pixel and pick it.
        if max(x0, x1) < 0.0 or min(x0, x1) >= w_t or max(y0, y1) < 0.0 or min(y0, y1) >= h_t:
            return {}

        gx0 = int(np.floor(min(x0, x1)))
        gx1 = int(np.floor(max(x0, x1)))
        # Window y is top-down, GL y is bottom-up. Flipping inverts the ordering,
        # so the TOP window edge (min y) becomes the HIGH GL row.
        gy0 = int(np.floor(h_t - max(y0, y1)))
        gy1 = int(np.floor(h_t - min(y0, y1)))

        gx0 = max(0, min(gx0, w_t - 1))
        gx1 = max(0, min(gx1, w_t - 1))
        gy0 = max(0, min(gy0, h_t - 1))
        gy1 = max(0, min(gy1, h_t - 1))

        w = gx1 - gx0 + 1
        h = gy1 - gy0 + 1
        if w <= 0 or h <= 0 or w * h > max_pixels:
            return {}

        glBindFramebuffer(GL_FRAMEBUFFER, self.target.fbo)
        # engine.py's export path leaves GL_PACK_ALIGNMENT at 1; do not inherit a
        # stale value, an odd row width under alignment 8 would pad and misparse.
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        pixels = glReadPixels(gx0, gy0, w, h, GL_RED_INTEGER, GL_INT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        buf = _as_int32(pixels)
        ids = np.unique(buf[buf > 0])
        if ids.size == 0:
            return {}

        layers, starts, counts = self._offset_table(scene)
        owner, element = self._decode_ids(ids, starts, counts)

        result: Dict[int, Set[int]] = {}
        for j in np.unique(owner[owner >= 0]):
            layer = layers[int(j)]
            hits = element[owner == j]
            result.setdefault(int(layer.layer_id), set()).update(int(e) for e in hits)
        return result
