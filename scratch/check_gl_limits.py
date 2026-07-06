import glfw
from OpenGL.GL import *
import numpy as np


def check_gl_limits():
    if not glfw.init():
        return

    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(100, 100, "Limits Check", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    # Check Point Size Range
    pt_range = glGetFloatv(GL_POINT_SIZE_RANGE)
    pt_granularity = glGetFloatv(GL_POINT_SIZE_GRANULARITY)

    print(f"GL_POINT_SIZE_RANGE: {pt_range}")
    print(f"GL_POINT_SIZE_GRANULARITY: {pt_granularity}")

    # Check if GL_PROGRAM_POINT_SIZE works
    glEnable(GL_PROGRAM_POINT_SIZE)
    if glIsEnabled(GL_PROGRAM_POINT_SIZE):
        print("GL_PROGRAM_POINT_SIZE is ENABLED")
    else:
        print("GL_PROGRAM_POINT_SIZE is DISABLED (Problematic for modern GL)")

    glfw.terminate()


if __name__ == "__main__":
    check_gl_limits()
