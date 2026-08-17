import numpy as np

import glplot.pyplot as gplt
from glplot.core.layers import Layer3D
from glplot.managers.hud import HudManager
from glplot.renderers.geometry3d import _bounds_3d, _look_at, _perspective


def test_3d_layers_auto_create_axes_and_sync_global_view():
    gplt._cleanup_pyplot_state()


def test_ssao_option_and_bar3d_edges():
    gplt._cleanup_pyplot_state()
    fig = gplt.figure(ssao=True)
    assert fig.options.visual.ssao.enabled is True
    gplt.ssao(False, strength=0.2, radius=2.0)
    assert fig.options.visual.ssao.enabled is False
    assert fig.options.visual.ssao.strength == 0.2
    assert fig.options.visual.ssao.radius == 2.0

    bars = gplt.bar3d([0], [0], [0], 1, 1, [2], edge_color="white", ssao=True, ssao_strength=0.7)
    assert bars[0].metadata["artist"] == "bar3d"
    assert bars[0].metadata["ssao"] is True
    assert bars[0].metadata["ssao_strength"] == 0.7
    assert bars[1].metadata["artist"] == "bar3d_edges"
    assert bars[1].layer_type == "wireframe3d"
    assert bars[1].vertices.shape == (24, 3)
    box_edges = bars[1].vertices.reshape(-1, 2, 3)
    assert len(box_edges) == 12
    starts_at_corner = np.isclose(box_edges[:, 0], [0.0, 0.0, 0.0], atol=1e-6).all(axis=1)
    ends_at_face_diagonal = np.isclose(box_edges[:, 1], [1.0, 1.0, 0.0], atol=1e-6).all(axis=1)
    assert not np.any(starts_at_corner & ends_at_face_diagonal)

    gplt._cleanup_pyplot_state()
    hex_bars = gplt.bar3d([0], [0], [0], 1, 1, [2], shape="hex", edge_color="white")
    assert hex_bars[0].indices.shape == (60,)
    assert hex_bars[1].vertices.shape == (36, 3)
    assert len(hex_bars[1].vertices.reshape(-1, 2, 3)) == 18

    gplt._cleanup_pyplot_state()
    separated = gplt.bar3d([0, 1], [0, 0], [0, 0], 1, 1, [2, 4], c=[2, 4], cmap="viridis", gap=0.2)
    assert separated[0].metadata["gap"] == 0.2
    assert separated[0].metadata["cmap"] == "viridis"
    assert separated[0].colors.shape == (16, 4)
    first_bar = separated[0].vertices[:8]
    assert np.isclose(first_bar[:, 0].min(), 0.1)
    assert np.isclose(first_bar[:, 0].max(), 0.9)
    assert not np.allclose(separated[0].colors[0], separated[0].colors[8])

    gplt._cleanup_pyplot_state()
    layer = gplt.scatter3d([0, 1], [0, 1], [0, 2], s=4)
    fig = gplt.gcf()
    axis_layers = [ly for ly in fig.scene.layers if ly.metadata.get("artist") == "axis3d"]
    assert len(axis_layers) == 1
    assert axis_layers[0].vertices.shape[1] == 3
    assert axis_layers[0].style.line_width >= 2.0
    scene_bounds = fig.get_3d_bounds()
    axis_bounds = axis_layers[0].metadata["scene_bounds"]
    assert axis_bounds[0] <= scene_bounds[0]
    assert axis_bounds[1] >= scene_bounds[1]
    assert axis_bounds[2] <= scene_bounds[2]
    assert axis_bounds[3] >= scene_bounds[3]
    assert axis_bounds[4] <= scene_bounds[4]
    assert axis_bounds[5] >= scene_bounds[5]
    assert np.allclose(
        axis_layers[0].colors[0], axis_layers[0].colors[2]
    )  # uniform color: no axis coloring

    fig.set_3d_view(elev=41, azim=-22, fov=55, show_axes=True)
    assert layer.metadata["camera"]["elev"] == 41
    assert layer.metadata["camera"]["azim"] == -22
    assert layer.metadata["camera"]["fov"] == 55
    assert axis_layers[0].metadata["camera"]["elev"] == 41
    assert np.allclose(layer.metadata["scene_bounds"], axis_bounds)

    fig.set_3d_view(show_axes=False)
    assert not any(ly.metadata.get("artist") == "axis3d" for ly in fig.scene.layers)
    gplt._cleanup_pyplot_state()


def test_hud_layer_summary_helpers_support_3d_layers():
    layer = Layer3D(
        vertices=np.zeros((1234, 3), dtype=np.float32), primitive="points", layer_type="scatter3d"
    )
    assert HudManager._is_3d_layer(layer) is True
    assert HudManager._layer_size_text(layer) == "1,234 verts"


def test_layer3d_bounds_and_camera_math_are_finite():
    layer = Layer3D(
        vertices=np.array(
            [
                [-1.0, -2.0, -3.0],
                [2.0, 3.0, 4.0],
            ],
            dtype=np.float32,
        ),
        primitive="points",
        layer_type="scatter3d",
    )

    assert layer.get_bounds_3d() == (-1.0, 2.0, -2.0, 3.0, -3.0, 4.0)
    center, radius = _bounds_3d(layer)
    assert np.allclose(center, [0.5, 0.5, 0.5])
    assert radius > 0

    projection = _perspective(45.0, 16 / 9, 0.1, 100.0)
    view = _look_at(
        np.array([3, 4, 5], dtype=np.float32), center, np.array([0, 0, 1], dtype=np.float32)
    )
    mvp = projection @ view
    assert mvp.shape == (4, 4)
    assert np.all(np.isfinite(mvp))
