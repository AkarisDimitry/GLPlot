import numpy as np

from glplot.engine import GPULinePlot
from glplot.options import BlendMode


def test_gpu_line_plot_init_and_legacy_aliases():
    plot = GPULinePlot(title="Demo")
    assert plot.width == 1280
    assert plot.height == 800
    assert plot.title == "Demo"
    assert plot.show_density is False
    assert plot.N == 0


def test_set_lines_ab_registers_legacy_and_layer_data():
    plot = GPULinePlot()
    ab = np.array([[1.0, 0.0], [2.0, 1.0]], dtype=np.float32)
    plot.set_lines_ab(ab, x_range=(-2, 2), label="family")
    assert plot.scene.lines.count == 2
    assert plot.N == 2
    assert plot._xrange == (-2.0, 2.0)
    assert plot.scene.layers[0].layer_type == "line_family"
    assert plot.scene.layers[0].label == "family"


def test_layer_helpers_keep_legacy_scene_lists_in_sync():
    plot = GPULinePlot()
    x = np.array([0, 1], dtype=np.float32)
    y = np.array([0, 1], dtype=np.float32)
    colors = np.tile([1, 0, 0, 1], (2, 1)).astype(np.float32)

    plot.add_line_strip(x, y, color=(0, 0, 0, 1), width=2)
    plot.add_scatter(x, y, colors=colors, size=5)
    plot.add_text(0, 0, "Test", fontsize=12)

    assert len(plot._line_strips) == 1
    assert len(plot._scatters) == 1
    assert len(plot._text_annotations) == 1
    assert len(plot.scene.strips) == 1
    assert len(plot.scene.scatters) == 1
    assert plot.scene.texts[0]["str"] == "Test"


def test_public_state_setters():
    plot = GPULinePlot()
    plot.set_density_enabled(True)
    plot.set_hud_enabled(False)
    plot.set_blending_mode("off")
    plot.set_global_alpha(0.4)
    plot.set_lod(True, 123)
    plot.set_title("Updated")

    assert plot.show_density is True
    assert plot.options.enable_hud is False
    assert plot.options.blend_mode == BlendMode.OFF
    assert plot.global_alpha == 0.4
    assert plot.options.default_global_alpha == 0.4
    assert plot.max_lines_per_px == 123
    assert plot.title == "Updated"


def test_view_and_autoscale():
    plot = GPULinePlot()
    plot.set_view(xlim=(-10, 10), ylim=(-5, 5))
    assert plot.camera.cx == 0.0
    assert plot.camera.cy == 0.0
    assert plot.camera.zoom_x == 0.1
    assert plot.camera.zoom_y == 0.2

    plot.set_lines_ab(np.array([[0.0, 5.0]], dtype=np.float32), x_range=(-10, 10))
    plot.autoscale()
    left, right = plot.get_xlim()
    bottom, top = plot.get_ylim()
    assert left < -10
    assert right > 10
    assert bottom < 5
    assert top > 5
