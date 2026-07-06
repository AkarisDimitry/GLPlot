import numpy as np
import glfw
from vispy import app, scene

# Force GLFW backend to match GLPlot and our known working setup
app.use_app("glfw")


def main():
    # Initialize glfw and reset window hints to defaults (essential for macOS)
    glfw.init()
    glfw.default_window_hints()

    # 100,000 lines
    N = 100000
    a = np.random.randn(N).astype(np.float32) * 2.0
    b = np.random.randn(N).astype(np.float32) * 250.0

    x_range = (-5.0, 5.0)

    # 2 vertices per line segment
    pos = np.empty((2 * N, 3), dtype=np.float32)
    pos[0::2, 0] = x_range[0]
    pos[0::2, 1] = a * x_range[0] + b
    pos[0::2, 2] = 0.0
    pos[1::2, 0] = x_range[1]
    pos[1::2, 1] = a * x_range[1] + b
    pos[1::2, 2] = 0.0

    # Uniform bright cyan color with transparency (alpha = 0.05)
    colors = np.zeros((N, 4), dtype=np.float32)
    colors[:, 0] = 0.0  # Red
    colors[:, 1] = 0.8  # Green
    colors[:, 2] = 1.0  # Blue
    colors[:, 3] = 0.05  # Alpha
    color_arr = np.repeat(colors, 2, axis=0)

    print("Initializing interactive VisPy canvas using GLFW...")
    canvas = scene.SceneCanvas(
        keys="interactive", show=True, size=(1024, 768), title="VisPy Interactive - 100,000 Lines"
    )
    grid = canvas.central_widget.add_grid()
    view = grid.add_view()

    # Add lines and enable alpha blending (disabling depth test)
    line = scene.visuals.Line(pos, color=color_arr, connect="segments", parent=view.scene)
    line.set_gl_state(depth_test=False, blend=True, blend_func=("src_alpha", "one_minus_src_alpha"))

    # Camera setup (pan/zoom)
    view.camera = "panzoom"
    view.camera.set_range(x=(x_range[0], x_range[1]), y=(-1000, 1000))

    print("Opening interactive GUI window. Close the window to exit.")
    app.run()


if __name__ == "__main__":
    main()
