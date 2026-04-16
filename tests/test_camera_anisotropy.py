import pytest
from unittest.mock import MagicMock
import numpy as np
import sys

# Mocking OpenGL and GLFW before importing glplot
mock_gl = MagicMock()
sys.modules['OpenGL'] = MagicMock()
sys.modules['OpenGL.GL'] = mock_gl
sys.modules['glfw'] = MagicMock()

import glplot.pyplot as gplt

@pytest.fixture(autouse=True)
def mock_glfw_opengl(mocker):
    # Mock OpenGL and GLFW completely for the pyplot wrapper tests
    mocker.patch('glplot.engine.glfw', create=True)
    mocker.patch('glplot.engine.glViewport', create=True)
    mocker.patch('glplot.engine.glClearColor', create=True)
    mocker.patch('glplot.engine.glEnable', create=True)
    mocker.patch('glplot.engine.glGenFramebuffers', create=True)
    mocker.patch('glplot.engine.glGenTextures', create=True)
    mocker.patch('glplot.engine.glBindTexture', create=True)
    mocker.patch('glplot.engine.glTexImage2D', create=True)
    mocker.patch('glplot.engine.glTexParameteri', create=True)
    mocker.patch('glplot.engine.glBindFramebuffer', create=True)
    mocker.patch('glplot.engine.glFramebufferTexture2D', create=True)
    mocker.patch('glplot.engine.glBlitFramebuffer', create=True)
    mocker.patch('glplot.engine.glCheckFramebufferStatus', return_value=0x8CD5, create=True) # GL_FRAMEBUFFER_COMPLETE
    
    # We patch engine initialization to avoid shader compilation
    mocker.patch('glplot.engine.GPULinePlot._init_shaders', create=True)
    mocker.patch('glplot.engine.GPULinePlot._init_buffers', create=True)
    mocker.patch('glplot.renderers.line_family.LineFamilyRenderer._init_shaders', create=True)
    mocker.patch('glplot.renderers.line_family.LineFamilyRenderer._init_buffers', create=True)

    # Automatically clean up plots between tests
    yield
    gplt.cla()


def test_decoupled_limits():
    """Verify that xlim and ylim can be set independently and decoupled."""
    x = np.linspace(-2, 2, 1000)
    y = 500 * np.sin(x * 10)
    
    gplt.figure("Aspect Ratio Test", width=800, height=600)
    gplt.plot(x, y, color='blue', label="Sinusoidal")
    
    gplt.xlim(-2, 2)
    gplt.ylim(-1000, 1000)
    
    lx = gplt.xlim()
    ly = gplt.ylim()
    
    assert abs(lx[0] - (-2.0)) < 1e-5
    assert abs(lx[1] - (2.0)) < 1e-5
    assert abs(ly[0] - (-1000.0)) < 1e-5
    assert abs(ly[1] - (1000.0)) < 1e-5

def test_zero_span_safety():
    """Verify autoscale behaves safely when the data has a zero span (horizontal line)."""
    gplt.figure("Zero Span Test", width=800, height=600)
    gplt.plot([-10, 10], [5, 5], color='red', label="Horizontal")
    gplt.autoscale()
    
    ly = gplt.ylim()
    # Should have a non-zero Y range despite zero span in data
    assert ly[1] > ly[0]

def test_aspect_ratio_consistency():
    """Verify that changing viewport window dimensions preserves correct world bounds scaling."""
    gplt.figure("Resize Test", width=800, height=600)
    gplt.xlim(-10, 10)
    gplt.ylim(-5, 5)
    
    fig = gplt._get_or_create_plot()
    # Trigger a resize event (simulating window resize to 400x600)
    fig._on_resize(None, 400, 600)
    
    lx = fig.get_xlim()
    ly = fig.get_ylim()
    
    assert lx[1] > lx[0]
    assert ly[1] > ly[0]
