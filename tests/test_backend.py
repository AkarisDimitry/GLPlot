import pytest
import numpy as np
from unittest.mock import MagicMock
import time

# We mock all PyOpenGL modules before importing backend
import sys
from unittest.mock import MagicMock

# Mocking OpenGL and GLFW
mock_gl = MagicMock()
sys.modules['OpenGL'] = MagicMock()
sys.modules['OpenGL'].GL = mock_gl
sys.modules['OpenGL.GL'] = mock_gl
sys.modules['glfw'] = MagicMock()

import glplot.backend as backend

@pytest.fixture(autouse=True)
def clean_gl_mock():
    mock_gl.reset_mock()

def test_gpu_line_plot_init():
    plot = backend.GPULinePlot()
    assert plot.width == 1280
    assert plot.height == 800
    assert not plot.show_density

def test_set_lines_ab():
    plot = backend.GPULinePlot()
    ab = np.array([[1.0, 0.0], [2.0, 1.0]], dtype=np.float32)
    plot.set_lines_ab(ab)
    assert plot.scene.lines.count == 2
    
def test_callbacks():
    plot = backend.GPULinePlot()
    plot.width = 100
    plot.height = 100
    
    # Test resize
    plot._on_resize(None, 200, 300)
    assert plot.width == 200
    assert plot.height == 300
    assert plot.frame.dirty_scene == True
    
    # Test FB resize
    plot._on_fb_resize(None, 400, 600)
    assert plot.fb_width == 400
    assert plot.fb_height == 600

def test_world_window():
    plot = backend.GPULinePlot()
    plot.width = 800
    plot.height = 600
    plot.camera.cx = 0.0
    plot.camera.cy = 0.0
    plot.camera.zoom_x = 1.0
    plot.camera.zoom_y = 1.0
    l, r, b, t = plot.camera_controller.world_window(800, 600)
    # With anisotropic camera, 1.0 scaling with padding 1.0 yields bounds exactly -1.0 to 1.0.
    assert b == -1.0
    assert t == 1.0
    assert l == -1.0
    assert r == 1.0

def test_lod_logic():
    plot = backend.GPULinePlot()
    plot.set_lines_ab(np.zeros((1000, 2), dtype=np.float32))
    plot.width = 100
    plot.options.default_line_budget_per_px = 1
    # target_lines = 1 * 100 = 100
    # prob = 100 / 1000 = 0.1
    # Note: _compute_lod_keep_prob is internal
    assert plot._compute_lod_keep_prob() == 0.1
    
    plot.set_lines_ab(np.zeros((50, 2), dtype=np.float32))
    assert plot._compute_lod_keep_prob() == 1.0 

def test_add_text():
    plot = backend.GPULinePlot()
    plot.add_text(0, 0, "Test", fontsize=12)
    assert len(plot.scene.texts) == 1
    assert plot.scene.texts[0]['str'] == "Test"

def test_add_scatter():
    plot = backend.GPULinePlot()
    x = np.array([0, 1])
    y = np.array([0, 1])
    cols = np.tile([1, 0, 0, 1], (2, 1)).astype(np.float32)
    plot.add_scatter(x, y, colors=cols, size=5)
    assert len(plot.scene.scatters) == 1
    assert plot.scene.scatters[0].pts.shape == (2, 2)

def test_public_api():
    plot = backend.GPULinePlot()
    plot.set_density_enabled(True)
    assert plot.show_density == True
    
    plot.set_hud_enabled(False)
    assert plot.options.enable_hud == False
    
    plot.set_blending_mode("off")
    assert plot.policy.runtime.blending_enabled == False
    
    plot.set_view(xlim=(-10, 10), ylim=(-5, 5))
    assert plot.camera.cx == 0.0
    assert plot.camera.cy == 0.0
    assert plot.camera.zoom_x == 1.0 / 10.0 # based on xlim half-width
    assert plot.camera.zoom_y == 1.0 / 5.0 # based on ylim half-height

def test_autoscale():
    plot = backend.GPULinePlot()
    ab = np.array([[0.0, 5.0]], dtype=np.float32) # y = 5
    plot.set_lines_ab(ab, x_range=(-10, 10))
    plot.autoscale()
    # xlim should be around (-10, 10) with padding
    l, r = plot.get_xlim()
    assert l < -10
    assert r > 10
    # ylim should be around 5
    b, t = plot.get_ylim()
    assert b < 5
    assert t > 5

def test_headless_mode():
    plot = backend.GPULinePlot()
    plot._is_test_mode = True
    plot.window = MagicMock()
    # Mock glReadPixels
    mock_gl.glReadPixels.return_value = np.zeros(1280*800*3, dtype=np.uint8).tobytes()
    
    # This should run without error in mock env
    plot.run()
    assert backend.glfw.window_hint.called
    # Check if visibility hint was set to false
    # window_hint(glfw.VISIBLE, glfw.FALSE)

def test_get_density_array():
    plot = backend.GPULinePlot()
    plot.density_renderer.accum_target = MagicMock()
    plot.density_renderer.accum_target.fbo = 123
    plot.density_renderer.accum_target.width = 100
    plot.density_renderer.accum_target.height = 50
    
    import sys
    current_mock_gl = sys.modules['OpenGL.GL']
    dummy_data = np.arange(5000, dtype=np.float32)
    current_mock_gl.glReadPixels.return_value = dummy_data.tobytes()
    current_mock_gl.glGetIntegerv.return_value = 0
    
    res = plot.get_density_array()
    assert res.shape == (50, 100)
    assert np.allclose(res.flatten(), dummy_data)
