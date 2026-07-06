#!/usr/bin/env python3
"""Check OpenGL capabilities and limits on current system.

This diagnostic tool verifies that your system supports all required
OpenGL features for GLPlot to function properly.

Usage:
    python tools/check_gl_capabilities.py
"""

import sys

import glfw
from OpenGL.GL import *


def check_gl_capabilities():
    """Check and report OpenGL capabilities."""
    print("=" * 70)
    print("GLPlot OpenGL Capabilities Check")
    print("=" * 70)

    # Initialize GLFW
    if not glfw.init():
        print("❌ ERROR: Failed to initialize GLFW")
        return False

    # Create hidden window
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(100, 100, "Capabilities Check", None, None)

    if not window:
        print("❌ ERROR: Failed to create GLFW window")
        glfw.terminate()
        return False

    glfw.make_context_current(window)

    try:
        # Check GL version
        gl_version = glGetString(GL_VERSION).decode()
        gl_vendor = glGetString(GL_VENDOR).decode()
        gl_renderer = glGetString(GL_RENDERER).decode()

        print(f"\n✓ OpenGL Version: {gl_version}")
        print(f"✓ Vendor: {gl_vendor}")
        print(f"✓ Renderer: {gl_renderer}")

        # Check Point Size Range
        pt_range = glGetFloatv(GL_POINT_SIZE_RANGE)
        pt_granularity = glGetFloatv(GL_POINT_SIZE_GRANULARITY)

        print(f"\n✓ GL_POINT_SIZE_RANGE: {pt_range[0]:.2f} - {pt_range[1]:.2f}")
        print(f"✓ GL_POINT_SIZE_GRANULARITY: {pt_granularity}")

        # Check for GL_PROGRAM_POINT_SIZE
        glEnable(GL_PROGRAM_POINT_SIZE)
        if glIsEnabled(GL_PROGRAM_POINT_SIZE):
            print("✓ GL_PROGRAM_POINT_SIZE: ENABLED (Good)")
        else:
            print("⚠ GL_PROGRAM_POINT_SIZE: DISABLED (May cause issues)")

        # Check for texture units
        max_texture_units = glGetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS)
        print(f"\n✓ Max Vertex Texture Units: {max_texture_units}")

        # Check for framebuffer support
        max_renderbuffer_size = glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE)
        print(f"✓ Max Renderbuffer Size: {max_renderbuffer_size}")

        # Check for texture size
        max_texture_size = glGetIntegerv(GL_MAX_TEXTURE_SIZE)
        print(f"✓ Max Texture Size: {max_texture_size} x {max_texture_size}")

        # Check extensions
        extensions = glGetString(GL_EXTENSIONS).decode().split()
        print(f"\n✓ Total Extensions: {len(extensions)}")

        critical_extensions = [
            "GL_EXT_framebuffer_object",
            "GL_ARB_vertex_buffer_object",
            "GL_ARB_texture_float",
        ]

        for ext in critical_extensions:
            if ext in extensions:
                print(f"  ✓ {ext}")
            else:
                print(f"  ⚠ {ext} (Missing)")

        print("\n" + "=" * 70)
        print("✓ All checks completed successfully!")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ ERROR during capability check: {e}")
        return False

    finally:
        glfw.terminate()


if __name__ == "__main__":
    success = check_gl_capabilities()
    sys.exit(0 if success else 1)
