from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from OpenGL.GL import *

from ..utils.gl_utils import link_program
from ..utils.shaders import GEOMETRY3D_FS, GEOMETRY3D_VS

if TYPE_CHECKING:
    from ..core.context import RenderContext
    from ..core.layers import Layer3D
    from ..options import EngineOptions


@dataclass
class GLGeometry3DBuffers:
    vao: int = 0
    vbo_pos: int = 0
    vbo_col: int = 0
    ebo: int = 0
    count: int = 0


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def _perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.deg2rad(fovy_deg) * 0.5)
    out = np.zeros((4, 4), dtype=np.float32)
    out[0, 0] = f / max(aspect, 1e-6)
    out[1, 1] = f
    out[2, 2] = (far + near) / (near - far)
    out[2, 3] = (2.0 * far * near) / (near - far)
    out[3, 2] = -1.0
    return out


def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = _normalize(center - eye)
    s = _normalize(np.cross(f, up))
    u = np.cross(s, f)
    out = np.eye(4, dtype=np.float32)
    out[0, :3] = s
    out[1, :3] = u
    out[2, :3] = -f
    out[0, 3] = -float(np.dot(s, eye))
    out[1, 3] = -float(np.dot(u, eye))
    out[2, 3] = float(np.dot(f, eye))
    return out


def _bounds_3d(layer: Layer3D) -> tuple[np.ndarray, float]:
    b = layer.metadata.get("scene_bounds") or layer.get_bounds_3d()
    if b is None:
        center = np.zeros(3, dtype=np.float32)
        radius = 1.0
    else:
        mins = np.array([b[0], b[2], b[4]], dtype=np.float32)
        maxs = np.array([b[1], b[3], b[5]], dtype=np.float32)
        center = 0.5 * (mins + maxs)
        radius = max(float(np.linalg.norm(maxs - mins)) * 0.5, 1e-3)
    return center, radius


class Geometry3DRenderer:
    """GPU renderer for large 3D point clouds, meshes, wireframes, volumes, and bars."""

    def __init__(self, options: EngineOptions):
        self.options = options
        self.prog = 0
        self.u_mvp = -1
        self.u_alpha = -1
        self.u_point_size = -1
        self.u_z_range = -1
        self.u_ssao_enabled = -1
        self.u_ssao_strength = -1
        self.u_is_points = -1
        self.u_ref_w = -1

    def initialize(self) -> None:
        self.prog = link_program(GEOMETRY3D_VS, GEOMETRY3D_FS)
        self.u_mvp = glGetUniformLocation(self.prog, "u_mvp")
        self.u_alpha = glGetUniformLocation(self.prog, "u_alpha")
        self.u_point_size = glGetUniformLocation(self.prog, "u_point_size")
        self.u_z_range = glGetUniformLocation(self.prog, "u_z_range")
        self.u_ssao_enabled = glGetUniformLocation(self.prog, "u_ssao_enabled")
        self.u_ssao_strength = glGetUniformLocation(self.prog, "u_ssao_strength")
        self.u_is_points = glGetUniformLocation(self.prog, "u_is_points")
        self.u_ref_w = glGetUniformLocation(self.prog, "u_ref_w")

    def _create_buffers(self, layer: Layer3D) -> GLGeometry3DBuffers:
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo_pos = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))

        vbo_col = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_col)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))

        ebo = 0
        if layer.indices is not None:
            ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)

        glBindVertexArray(0)
        return GLGeometry3DBuffers(vao=vao, vbo_pos=vbo_pos, vbo_col=vbo_col, ebo=ebo)

    def update_gpu_data(self, layer: Layer3D, bufs: GLGeometry3DBuffers) -> None:
        if layer.vertices is None or len(layer.vertices) == 0:
            bufs.count = 0
            layer.dirty.gpu_dirty = False
            return

        vertices = np.ascontiguousarray(layer.vertices, dtype=np.float32)
        colors = layer.colors
        if colors is None:
            base = layer.style.color or (0.1, 0.45, 1.0, 1.0)
            colors = np.tile(np.asarray(base, dtype=np.float32), (len(vertices), 1))
        colors = np.ascontiguousarray(colors, dtype=np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, bufs.vbo_col)
        glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)

        if layer.indices is not None and bufs.ebo:
            indices = np.ascontiguousarray(layer.indices, dtype=np.uint32)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufs.ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
            bufs.count = len(indices)
        else:
            bufs.count = len(vertices)
        layer.dirty.gpu_dirty = False

    def _mvp(self, layer: Layer3D, ctx: RenderContext) -> np.ndarray:
        camera = layer.metadata.get("camera", {})
        elev = float(camera.get("elev", layer.metadata.get("elev", 28.0)))
        azim = float(camera.get("azim", layer.metadata.get("azim", -45.0)))
        fov = float(camera.get("fov", 42.0))
        center, radius = _bounds_3d(layer)
        az = np.deg2rad(azim)
        el = np.deg2rad(elev)
        distance = float(camera.get("distance", radius * 2.9))
        eye = center + distance * np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
            dtype=np.float32,
        )
        near = max(radius * 0.02, 1e-3)
        far = max(radius * 12.0, near + 1.0)
        return _perspective(fov, ctx.aspect, near, far) @ _look_at(eye, center, np.array([0, 0, 1], dtype=np.float32))

    def draw(self, layer: Layer3D, ctx: RenderContext) -> None:
        if layer.vertices is None or len(layer.vertices) == 0:
            return
        if not hasattr(layer, "_gl") or layer._gl is None:
            layer._gl = self._create_buffers(layer)
            layer.dirty.gpu_dirty = True
        if layer.dirty.gpu_dirty:
            self.update_gpu_data(layer, layer._gl)

        mode_map = {
            "points": GL_POINTS,
            "lines": GL_LINES,
            "triangles": GL_TRIANGLES,
        }
        mode = mode_map.get(layer.primitive, GL_POINTS)

        # axis3d wireframe is drawn last and must always be visible regardless of
        # what volumetric data lies in front of it — skip depth testing for it.
        is_axis_overlay = layer.metadata.get("artist") == "axis3d"
        if not is_axis_overlay:
            glEnable(GL_DEPTH_TEST)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glUseProgram(self.prog)
        glUniformMatrix4fv(self.u_mvp, 1, GL_TRUE, self._mvp(layer, ctx))
        glUniform1f(self.u_alpha, float(ctx.global_alpha * layer.style.alpha))
        glUniform1f(self.u_point_size, float(layer.style.point_size * ctx.dpr))
        b = layer.get_bounds_3d()
        zmin, zmax = (0.0, 1.0) if b is None else (float(b[4]), float(b[5]))
        glUniform2f(self.u_z_range, zmin, zmax)
        ssao = getattr(self.options.visual, "ssao", None)
        enabled = bool(getattr(ssao, "enabled", False) or layer.metadata.get("ssao", False))
        strength = float(layer.metadata.get("ssao_strength", getattr(ssao, "strength", 0.45)))
        glUniform1i(self.u_ssao_enabled, 1 if enabled else 0)
        glUniform1f(self.u_ssao_strength, strength)
        glUniform1i(self.u_is_points, 1 if mode == GL_POINTS else 0)

        # Perspective-correct point sizing: pass camera distance as the reference
        # clip-space w so that points closer than the scene centre look proportionally
        # larger and farther points look smaller — matching natural perspective depth cues.
        # Only for point primitives; lines/triangles keep a fixed screen-space width.
        if mode == GL_POINTS:
            camera = layer.metadata.get("camera", {})
            _, radius = _bounds_3d(layer)
            ref_w = float(camera.get("distance", radius * 3.2))
        else:
            ref_w = 0.0
        glUniform1f(self.u_ref_w, ref_w)

        glBindVertexArray(layer._gl.vao)
        if layer.indices is not None and layer._gl.ebo:
            glDrawElements(mode, layer._gl.count, GL_UNSIGNED_INT, None)
        else:
            glDrawArrays(mode, 0, layer._gl.count)
        glBindVertexArray(0)
        glUseProgram(0)
        glDisable(GL_PROGRAM_POINT_SIZE)
        glDisable(GL_DEPTH_TEST)
