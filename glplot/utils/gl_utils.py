import re

import numpy as np
from OpenGL.GL import *

#: Number of world-window clip planes the engine enables (GL_CLIP_DISTANCE0..3).
_ENGINE_CLIP_PLANES = 4
_MAIN_OPEN = re.compile(r"void\s+main\s*\(\s*(?:void)?\s*\)\s*\{")


def ortho(left: float, r: float, b: float, t: float, n: float = -1.0, f: float = 1.0) -> np.ndarray:
    rl, tb, fn = (r - left), (t - b), (f - n)
    return np.array(
        [
            [2.0 / rl, 0.0, 0.0, -(r + left) / rl],
            [0.0, 2.0 / tb, 0.0, -(t + b) / tb],
            [0.0, 0.0, -2.0 / fn, -(f + n) / fn],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def compile_shader(src: str, stype: int) -> int:
    sid = glCreateShader(stype)
    glShaderSource(sid, src)
    glCompileShader(sid)
    if not glGetShaderiv(sid, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(sid).decode(errors="ignore"))
    return sid


def with_defined_clip_distances(vs_src: str) -> str:
    """Make a vertex shader that ignores clipping say so, instead of leaving it undefined.

    The engine keeps GL_CLIP_DISTANCE0-3 enabled for the whole scene pass, so the 2D
    primitives are clipped to the visible world window. A vertex shader that never writes
    ``gl_ClipDistance`` while those planes are enabled produces *undefined* clip values
    (GLSL spec). macOS happens to treat them as "keep everything", which is why the 3D
    renderer, the axis strips, images and picking all look right there. Other drivers are
    free to read garbage and cull the primitive: a point is all-or-nothing, so a whole 3D
    scatter can vanish on a Linux/Windows GPU while the same script works on a Mac.

    Writing 1.0 (inside every plane) at the top of ``main`` pins down exactly what macOS
    already did, on every driver. Shaders that clip for real already write the array and
    are returned untouched.
    """
    if "gl_ClipDistance" in vs_src:
        return vs_src
    match = _MAIN_OPEN.search(vs_src)
    if match is None:
        return vs_src
    writes = "".join(f"\n    gl_ClipDistance[{i}] = 1.0;" for i in range(_ENGINE_CLIP_PLANES))
    return vs_src[: match.end()] + writes + vs_src[match.end() :]


def link_program(vs_src: str, fs_src: str) -> int:
    vs = compile_shader(with_defined_clip_distances(vs_src), GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL_FRAGMENT_SHADER)
    pid = glCreateProgram()
    glAttachShader(pid, vs)
    glAttachShader(pid, fs)
    glLinkProgram(pid)
    glDeleteShader(vs)
    glDeleteShader(fs)
    if not glGetProgramiv(pid, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(pid).decode(errors="ignore"))
    return pid
