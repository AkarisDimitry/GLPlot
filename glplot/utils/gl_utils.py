import numpy as np
from OpenGL.GL import *


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


def link_program(vs_src: str, fs_src: str) -> int:
    vs = compile_shader(vs_src, GL_VERTEX_SHADER)
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
