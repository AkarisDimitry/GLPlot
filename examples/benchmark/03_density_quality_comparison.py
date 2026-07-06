import time
import os
import sys
import numpy as np
import warnings
import glfw

# Force local glplot import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib

matplotlib.use("Agg")  # Force non-GUI backend
import matplotlib.pyplot as plt
import glplot.pyplot as gplt

# Try importing VisPy
try:
    from vispy import app, scene

    app.use_app("glfw")
    vispy_available = True
except Exception as e:
    print(f"Warning: VisPy failed to import: {e}")
    vispy_available = False

# Try importing fastplotlib
try:
    import fastplotlib as fpl

    fastplotlib_available = True
except Exception as e:
    print(f"Warning: fastplotlib failed to import: {e}")
    fastplotlib_available = False

# Suppress warnings
warnings.filterwarnings("ignore")


def run_vispy_density(N, a, b, x_range, width, height, output_path):
    """Run VisPy to render transparent segments representing the caustic."""
    glfw.init()
    glfw.default_window_hints()

    canvas = scene.SceneCanvas(show=False, size=(width, height), keys="interactive")
    grid = canvas.central_widget.add_grid()
    view = grid.add_view()

    # Construct 2 vertices per segment
    pos = np.empty((2 * N, 3), dtype=np.float32)
    pos[0::2, 0] = x_range[0]
    pos[0::2, 1] = a * x_range[0] + b
    pos[0::2, 2] = 0.0
    pos[1::2, 0] = x_range[1]
    pos[1::2, 1] = a * x_range[1] + b
    pos[1::2, 2] = 0.0

    # Set a bright color and alpha
    colors = np.zeros((N, 4), dtype=np.float32)
    colors[:, 0] = 0.0  # Red
    colors[:, 1] = 0.8  # Green
    colors[:, 2] = 1.0  # Blue
    colors[:, 3] = 0.05  # Alpha
    color_arr = np.repeat(colors, 2, axis=0)

    line = scene.visuals.Line(pos, color=color_arr, connect="segments", parent=view.scene)
    line.set_gl_state(depth_test=False, blend=True, blend_func=("src_alpha", "one_minus_src_alpha"))

    view.camera = "panzoom"
    view.camera.set_range(x=(x_range[0], x_range[1]), y=(-10, 20))

    canvas.app.process_events()
    img = canvas.render()
    plt.imsave(output_path, np.ascontiguousarray(img))
    canvas.close()


def run_fastplotlib_density(N, a, b, x_range, width, height, output_path):
    """Run fastplotlib to render transparent segments representing the caustic."""
    fig = fpl.Figure(canvas="offscreen", size=(width, height))
    subplot = fig[0, 0]

    # Construct data for LineCollection: shape (N, 2, 3)
    data = np.empty((N, 2, 3), dtype=np.float32)
    data[:, 0, 0] = x_range[0]
    data[:, 0, 1] = a * x_range[0] + b
    data[:, 0, 2] = 0.0
    data[:, 1, 0] = x_range[1]
    data[:, 1, 1] = a * x_range[1] + b
    data[:, 1, 2] = 0.0

    # Set bright cyan color and alpha
    colors = np.zeros((N, 4), dtype=np.float32)
    colors[:, 0] = 0.0  # Red
    colors[:, 1] = 0.8  # Green
    colors[:, 2] = 1.0  # Blue
    colors[:, 3] = 0.05  # Alpha

    # Add line collection
    subplot.add_line_collection(data, colors=colors)

    # Position camera to view our bounding box
    subplot.camera.width = x_range[1] - x_range[0]
    subplot.camera.height = 30
    subplot.camera.position = (0, 5, 0)

    # Force render and export
    fig._render()
    fig.export(output_path)
    if fig._output is not None:
        fig.close()


def main():
    print("=" * 60)
    print("   GLPLOT VS VISPY VS FASTPLOTLIB: DENSITY QUALITY COMPARISON")
    print("=" * 60)

    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)

    # 20,000 lines
    N = 20000
    x_range = (-5.0, 5.0)
    width, height = 800, 600

    # Generate tangent lines to the parabola y = x^2
    a = np.random.uniform(-8.0, 8.0, N).astype(np.float32)
    b = -(a**2) / 4.0

    # 1. Render using GLPlot GPU Density
    print("Rendering GLPlot GPU Density...")
    gl_out = os.path.join(output_dir, "density_quality_glplot.png")
    gplt._cleanup_pyplot_state()
    gplt.figure("GLPlot Density Caustic", width=width, height=height, density=True)
    gplt.options(density_resolution_scale=1.0, density_gain=3.0)
    colors = np.ones((N, 4), dtype=np.float32)
    gplt.plot_lines(a, b, x_range=x_range, colors=colors)
    gplt.xlim(x_range[0], x_range[1])
    gplt.ylim(-10, 20)
    gplt.show(test_mode=True)
    gplt.savefig(gl_out, density=True)
    print(f"  GLPlot image saved: {gl_out}")

    # 2. Render using VisPy Transparent Lines
    if vispy_available:
        print("Rendering VisPy Transparent Lines...")
        vp_out = os.path.join(output_dir, "density_quality_vispy.png")
        run_vispy_density(N, a, b, x_range, width, height, vp_out)
        print(f"  VisPy image saved: {vp_out}")
    else:
        vp_out = None

    # 3. Render using fastplotlib Transparent Lines
    if fastplotlib_available:
        print("Rendering fastplotlib Transparent Lines...")
        fpl_out = os.path.join(output_dir, "density_quality_fastplotlib.png")
        try:
            run_fastplotlib_density(N, a, b, x_range, width, height, fpl_out)
            print(f"  fastplotlib image saved: {fpl_out}")
        except Exception as e:
            print(f"  fastplotlib failed: {e}")
            fpl_out = None
    else:
        fpl_out = None

    # 4. Create Side-by-Side Comparison Plot
    print("Generating comparison plot...")
    try:
        active_plots = 1 + (1 if vp_out else 0) + (1 if fpl_out else 0)
        fig, axes = plt.subplots(1, active_plots, figsize=(8 * active_plots, 7), dpi=120)

        if active_plots == 1:
            axes = [axes]

        curr_ax = 0

        # Plot GLPlot
        img_gl = plt.imread(gl_out)
        axes[curr_ax].imshow(img_gl)
        axes[curr_ax].axis("off")
        axes[curr_ax].set_title(
            "GLPlot GPU Analytical Density (Heatmap)", fontsize=13, fontweight="bold", pad=10
        )
        curr_ax += 1

        # Plot VisPy
        if vp_out:
            img_vp = plt.imread(vp_out)
            axes[curr_ax].imshow(img_vp)
            axes[curr_ax].axis("off")
            axes[curr_ax].set_title(
                "VisPy GPU Segment Blending (OpenGL)", fontsize=13, fontweight="bold", pad=10
            )
            curr_ax += 1

        # Plot fastplotlib
        if fpl_out:
            img_fpl = plt.imread(fpl_out)
            axes[curr_ax].imshow(img_fpl)
            axes[curr_ax].axis("off")
            axes[curr_ax].set_title(
                "fastplotlib GPU Segment Blending (WGPU)", fontsize=13, fontweight="bold", pad=10
            )
            curr_ax += 1

        comparison_img = os.path.join(output_dir, "density_quality_comparison.png")
        plt.tight_layout()
        plt.savefig(comparison_img)
        plt.close()
        print(f"Density quality comparison plot saved: {comparison_img}")
    except Exception as e:
        print(f"Failed to compile side-by-side plot: {e}")

    print("=" * 60)


if __name__ == "__main__":
    main()
