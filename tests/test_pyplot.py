import numpy as np
import pytest

import glplot.pyplot as gplt


@pytest.fixture(autouse=True)
def clean_pyplot_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


def layer_types():
    return [layer.layer_type for layer in gplt.gcf().scene.layers]


def test_figure_accepts_matplotlib_figsize():
    fig = gplt.figure(title="Test", figsize=(8, 6), dpi=100)
    assert fig.title == "Test"
    assert fig.width == 800
    assert fig.height == 600


def test_plot_accepts_y_only_and_returns_artists():
    artists = gplt.plot([0, 1, 4], "r-o", label="quadratic")
    assert len(artists) == 2
    assert layer_types() == ["polyline", "scatter"]
    assert gplt.gcf().scene.layers[0].label == "quadratic"


def test_plot_accepts_repeated_matplotlib_groups():
    x = np.arange(3)
    artists = gplt.plot(x, x, "r-", x, x**2, "bo")
    assert len(artists) == 3
    assert layer_types() == ["polyline", "polyline", "scatter"]


def test_plot_sequence_mismatch():
    with pytest.raises(ValueError):
        gplt.plot([1, 2], [1])


def test_plot_lines_sets_line_family():
    gplt.plot_lines([1, 2, 3], [0, 1, 2], x_range=(0, 10))
    fig = gplt.gcf()
    assert fig.N == 3
    assert fig._xrange == (0.0, 10.0)
    assert layer_types() == ["line_family"]


def test_plot_lines_mismatch():
    with pytest.raises(ValueError):
        gplt.plot_lines([1, 2], [1], x_range=(0, 1))


def test_scatter_accepts_matplotlib_aliases():
    layer = gplt.scatter([0, 1], [0, 1], c="tab:orange", s=10, alpha=0.5, marker="x")
    assert layer.layer_type == "scatter"
    assert layer.colors.shape == (2, 4)
    assert layer.style.point_size == 10
    assert layer.metadata["marker"] == "x"


def test_scatter_accepts_numeric_cmap_values():
    layer = gplt.scatter([0, 1, 2], [0, 1, 0], c=[0.0, 0.5, 1.0], cmap="plasma", s=8)
    assert layer.colors.shape == (3, 4)
    assert layer.metadata["cmap"] == "plasma"
    assert not np.allclose(layer.colors[0], layer.colors[-1])


def test_fill_between_creates_patch():
    # fill_between returns the patch Layer, per its docstring -- not the figure. It used
    # to return the figure, and this test read `.scene` off it; the contract is now the
    # one the docstring always promised and that stackplot/fill_betweenx rely on.
    layer = gplt.fill_between([0, 1, 2], [1, 2, 1], 0, color="#336699", alpha=0.25)
    assert layer.layer_type == "patch"
    assert layer.vertices.shape == (6, 2)


def test_bar_and_hist_create_patch_layers():
    bars = gplt.bar([0, 1], [2, 3], color="seagreen")
    assert len(bars) == 2
    assert layer_types() == ["patch", "patch"]

    gplt.clf()
    counts, edges, artists = gplt.hist([0, 0, 1, 2], bins=2)
    assert counts.sum() == 4
    assert len(edges) == 3
    assert len(artists) == 2


def test_reference_lines_and_axline():
    gplt.plot([0, 1], [0, 1])
    h = gplt.axhline(0.5, color="r", linestyle="--")
    v = gplt.axvline(0.25, color="b", linestyle=":")
    a = gplt.axline((0, 0), slope=1.0, color="k")
    assert h.metadata["artist"] == "axhline"
    assert v.metadata["artist"] == "axvline"
    assert a.metadata["artist"] == "axline"
    assert len(gplt.gcf()._line_strips) == 4


def test_hlines_vlines_step_errorbar_and_stem():
    gplt.hlines([1, 2], 0, 3)
    gplt.vlines([1, 2], 0, 3)
    step_artists = gplt.step([0, 1, 2], [0, 1, 0], where="mid")
    err_artists = gplt.errorbar([0, 1], [1, 2], yerr=[0.1, 0.2], fmt="ro", capsize=0.05)
    stem_artists = gplt.stem([0, 1, 2], [1, 3, 2])
    assert step_artists[0].metadata["artist"] == "step"
    assert all(a.metadata.get("artist_group") == "errorbar" for a in err_artists)
    assert all(a.metadata.get("artist_group") == "stem" for a in stem_artists)


def test_plot3d_and_scatter3d_project_and_store_zdata():
    t = np.linspace(0, 1, 5)
    line = gplt.plot3d(t, t * 2, t * 3, "C1-", elev=20, azim=-35)
    cloud = gplt.scatter3d(t, t * 2, t * 3, cmap="viridis", s=5)
    assert line[0].metadata["artist"] == "plot3d"
    assert line[0].layer_type == "wireframe3d"
    assert line[0].vertices.shape == (8, 3)
    assert line[0].metadata["zdata"].shape == (5,)
    assert cloud.metadata["artist"] == "scatter3d"
    assert cloud.layer_type == "scatter3d"
    assert cloud.colors.shape == (5, 4)


def test_arrow_quiver_and_annotate_create_line_patch_text_layers():
    arrow_artists = gplt.arrow(0, 0, 1, 1, color="tab:red")
    quiver_artists = gplt.quiver([0, 1], [0, 1], [1, 0], [0, 1], scale=0.2)
    anno_artists = gplt.annotate("peak", xy=(1, 1), xytext=(0.5, 0.8), arrowprops={"color": "k"})
    assert {a.metadata.get("artist") for a in arrow_artists} == {"arrow"}
    assert all(a.metadata.get("artist_group") == "quiver" for a in quiver_artists)
    assert anno_artists[0].metadata["artist"] == "annotate_text"
    assert "text" in layer_types()


def test_quiver3d_creates_single_3d_vector_layer():
    layer = gplt.quiver3d(
        [0, 1],
        [0, 1],
        [0, 0],
        [1, 0],
        [0, 1],
        [0.5, 0.25],
        scale=0.4,
        normalize=True,
        label="3d vectors",
    )[0]
    assert layer.layer_type == "wireframe3d"
    assert layer.metadata["artist"] == "quiver3d"
    assert layer.metadata["vector_count"] == 2
    assert layer.vertices.shape == (12, 3)
    assert layer.label == "3d vectors"


def test_imshow_matshow_and_hist2d():
    matrix = np.arange(16, dtype=np.float32).reshape(4, 4)
    img = gplt.imshow(matrix, cmap="magma", extent=(-1, 1, -2, 2))
    mat = gplt.matshow(matrix)
    counts, xedges, yedges, layer = gplt.hist2d([0, 0, 1, 1], [0, 1, 0, 1], bins=2)
    assert img.metadata["artist"] == "imshow"
    assert img.metadata["matrix"].shape == (4, 4)
    assert mat.metadata["artist"] == "imshow"
    assert counts.shape == (2, 2)
    assert layer.metadata["artist"] == "hist2d"


def test_imshow_accepts_rgb_and_rgba_images():
    """imshow() used to require a 2D scalar matrix; a loaded image (e.g. via imread())
    is (H, W, 3) or (H, W, 4) and must render with its own per-pixel colour, not be
    forced through the scalar colormap path."""
    rgb_u8 = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb_u8[0, 0] = [255, 0, 0]
    layer = gplt.imshow(rgb_u8)
    assert layer.metadata["matrix"].shape == (4, 4, 3)
    # uint8 0-255 is normalised to 0-1 before it reaches the renderer.
    assert layer.colors[0] == pytest.approx([1.0, 0.0, 0.0, 1.0])

    rgba_f32 = np.zeros((4, 4, 4), dtype=np.float32)
    rgba_f32[0, 0] = [0.0, 1.0, 0.0, 0.5]
    layer = gplt.imshow(rgba_f32)
    assert layer.metadata["matrix"].shape == (4, 4, 4)
    assert layer.colors[0] == pytest.approx([0.0, 1.0, 0.0, 0.5])

    # The original scalar/colormap path must still work unchanged.
    scalar = gplt.imshow(np.arange(16, dtype=np.float32).reshape(4, 4), cmap="viridis")
    assert scalar.metadata["matrix"].shape == (4, 4)
    assert scalar.metadata["cmap"] == "viridis"


def test_contour_pcolormesh_and_3d_surface_helpers():
    axis = np.linspace(-1, 1, 5, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    z = np.sin(x * y)
    pseudo_mesh = gplt.pcolormesh(x, y, z, cmap="viridis")
    c = gplt.contour(x, y, z, levels=4, colors="k")
    cf = gplt.contourf(x, y, z, levels=5, cmap="magma")
    surface = gplt.plot_surface(x, y, z, cmap="turbo", rstride=2, cstride=2)
    wire = gplt.plot_wireframe(x, y, z, rstride=2, cstride=2)
    bars = gplt.bar3d([0, 1], [0, 1], [0, 0], 0.2, 0.2, [1, 2])
    hex_bars = gplt.bar3d([0], [0], [0], 0.3, 0.3, [1], shape="hex")
    tri_mesh = gplt.mesh3d(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
    )
    volume = gplt.volume3d([0, 1], [0, 1], [0, 1], [0.2, 0.9], threshold=0.5)
    assert pseudo_mesh.metadata["artist"] == "pcolormesh"
    assert c.metadata["artist"] == "contour"
    assert cf.metadata["artist"] == "contourf"
    assert surface.metadata["artist"] == "plot_surface"
    assert surface.layer_type == "mesh3d"
    assert all(
        a.metadata["artist"] == "plot_wireframe" and a.layer_type == "wireframe3d" for a in wire
    )
    assert bars[0].metadata["artist"] == "bar3d"
    assert bars[0].layer_type == "bars3d"
    assert bars[1].metadata["artist"] == "bar3d_edges"
    assert bars[1].layer_type == "wireframe3d"
    assert hex_bars[0].metadata["shape"] == "hex"
    assert tri_mesh.layer_type == "mesh3d"
    assert volume.layer_type == "volume3d"
    assert len(volume.vertices) == 1


def test_labels_legend_and_view_helpers():
    gplt.plot([0, 1], [0, 1], label="line")
    gplt.xlabel("x")
    gplt.ylabel("y")
    gplt.zlabel("z")
    gplt.title("Demo")
    gplt.grid(True)
    labels = gplt.legend()
    fig = gplt.gcf()
    assert fig.xlabel == "x"
    assert fig.ylabel == "y"
    assert fig.zlabel == "z"
    assert fig.title == "Demo"
    assert fig.grid_visible is True
    assert labels == ["line"]


def test_legend_deduplicates_and_limits_items():
    for idx in range(4):
        gplt.plot([0, 1], [idx, idx + 1], label=f"series {idx}")
    gplt.plot([0, 1], [0, 1], label="series 1")
    labels = gplt.legend(max_items=2)
    assert labels == ["series 0", "series 1", "+2 more layers"]


def test_set_global_alpha_and_lod():
    gplt.set_global_alpha(0.5)
    gplt.set_lod(enabled=True, max_lines_per_px=500)
    fig = gplt.gcf()
    assert fig.global_alpha == 0.5
    assert fig.enable_subsample is True
    assert fig.max_lines_per_px == 500
