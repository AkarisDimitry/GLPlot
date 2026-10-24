"""The MCP tools, driven as plain methods against the real headless renderer.

No MCP SDK is needed here: :class:`~glplot.mcp.toolbox.Toolbox` is the whole behaviour and
:mod:`glplot.mcp.server` only describes it. Every test that renders goes through the same
matplotlib-preview path ``savefig`` uses when there is no GL window, so none of this opens
one (see the repo's rule against live GL calls in pytest).
"""

from __future__ import annotations

import io
import json
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.mcp import toolbox as toolbox_module
from glplot.mcp.state import FileSandbox, Limits, McpError
from glplot.mcp.toolbox import PLOT_KINDS, Toolbox
from glplot.pyplot import MatplotlibCompatWarning

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def root(tmp_path):
    base = tmp_path / "work"
    base.mkdir()
    return base


@pytest.fixture
def box(root):
    toolbox = Toolbox(FileSandbox([root]))
    yield toolbox
    toolbox.close_all()


def png_size(data: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(data)).size


def make_dataset(box, name="d", n=50):
    box.create_dataset(
        name,
        {"t": f"linspace(0, 10, {n})", "y": "sin(t)", "z": "cos(t)", "g": "floor(t)"},
    )
    return name


def scene_layers(box, figure_id):
    return box._figures[figure_id].plot.scene.layers


class TestCreateFigure:
    def test_returns_an_id_and_registers_the_figure(self, box):
        result = box.create_figure(title="T", width=800, height=500)
        assert result.data["figure_id"] == "fig1"
        assert box.figure_ids() == ["fig1"]

    def test_ids_are_unique_and_never_reused(self, box):
        first = box.create_figure().data["figure_id"]
        box.close_figure(first)
        assert box.create_figure().data["figure_id"] == "fig2"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"width": 10},
            {"height": 99999},
            {"rows": 0},
            {"cols": 7},
            {"projection": "4d"},
            {"width": 800.5},
        ],
    )
    def test_bad_geometry_is_refused(self, box, kwargs):
        with pytest.raises(McpError):
            box.create_figure(**kwargs)
        assert box.figure_ids() == []

    def test_unknown_style_lists_the_available_ones_and_leaves_no_figure(self, box):
        with pytest.raises(McpError, match="journal"):
            box.create_figure(style="nope")
        assert box.figure_ids() == []

    def test_figure_count_is_bounded(self, root):
        small = Toolbox(FileSandbox([root]), Limits(max_figures=2))
        try:
            small.create_figure()
            small.create_figure()
            with pytest.raises(McpError, match="Too many open figures"):
                small.create_figure()
        finally:
            small.close_all()

    def test_an_untitled_figure_renders_without_a_stray_title(self, box):
        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "line", y=[1, 2, 3])
        panel = box.render_figure(fig).data["panels"][0]
        assert "title" not in panel

    def test_title_is_applied(self, box):
        fig = box.create_figure(title="Hello").data["figure_id"]
        box.add_plot(fig, "line", y=[1, 2, 3])
        assert box.render_figure(fig).data["panels"][0]["title"] == "Hello"

    def test_style_preset_is_applied(self, box):
        fig = box.create_figure(style="dark").data["figure_id"]
        assert fig in box.figure_ids()

    def test_grid_creates_that_many_panels(self, box):
        result = box.create_figure(rows=2, cols=3)
        assert result.data["panels"] == 6
        assert len(box._figures["fig1"].plot.panels) == 6

    def test_close_releases_the_figure_from_pyplot(self, box):
        fig = box.create_figure().data["figure_id"]
        plot = box._figures[fig].plot
        box.close_figure(fig)
        assert plot not in gplt._ALL_PLOTS
        with pytest.raises(McpError, match="No figure"):
            box.close_figure(fig)

    def test_unknown_figure_suggests_the_nearest_id(self, box):
        box.create_figure()
        with pytest.raises(McpError, match="Did you mean 'fig1'"):
            box.add_plot("fig", "line", y=[1])


class TestDatasets:
    def test_expressions_see_earlier_columns(self, box):
        box.create_dataset("d", {"t": "linspace(0, 1, 5)", "u": "t * 2", "v": "u + 1"})
        assert box.datasets.column("d", "v").tolist() == [1.0, 1.5, 2.0, 2.5, 3.0]

    def test_lists_and_expressions_mix_and_scalars_repeat(self, box):
        box.create_dataset("d", {"a": [1, 2, 3], "b": "a ** 2", "c": "5"})
        assert box.datasets.column("d", "b").tolist() == [1.0, 4.0, 9.0]
        assert box.datasets.column("d", "c").tolist() == [5.0, 5.0, 5.0]

    def test_none_in_a_list_is_a_gap(self, box):
        box.create_dataset("d", {"a": [1, None, 3]})
        assert np.isnan(box.datasets.column("d", "a")[1])

    def test_base_dataset_columns_are_inherited_and_usable(self, box):
        box.create_dataset("src", {"t": "linspace(0, 1, 4)"})
        box.create_dataset("derived", {"u": "t * 10"}, base="src")
        assert box.datasets.get("derived").column_names() == ["t", "u"]
        assert box.datasets.column("derived", "u")[-1] == 10.0

    def test_replacing_says_so(self, box):
        box.create_dataset("d", {"a": [1]})
        assert "Replaced" in box.create_dataset("d", {"a": [2]}).message

    def test_a_leading_scalar_has_no_length_to_repeat_to(self, box):
        with pytest.raises(McpError, match="length"):
            box.create_dataset("d", {"a": "5"})

    def test_bad_expression_names_the_column(self, box):
        with pytest.raises(McpError, match="column 'y'"):
            box.create_dataset("d", {"t": [1, 2], "y": "boom(t)"})
        assert "d" not in box.datasets

    @pytest.mark.parametrize("name", ["my col", "1x", "a-b", "x__y"])
    def test_column_names_must_be_identifiers(self, box, name):
        with pytest.raises(McpError, match="identifier"):
            box.create_dataset("d", {name: [1, 2]})

    def test_empty_columns_are_refused(self, box):
        with pytest.raises(McpError):
            box.create_dataset("d", {})

    def test_inline_list_cap(self, root):
        small = Toolbox(FileSandbox([root]), Limits(max_inline_values=10))
        try:
            with pytest.raises(McpError, match="load_dataset"):
                small.create_dataset("d", {"a": list(range(11))})
        finally:
            small.close_all()

    def test_a_long_expression_result_is_not_subject_to_the_inline_cap(self, root):
        small = Toolbox(FileSandbox([root]), Limits(max_inline_values=10))
        try:
            small.create_dataset("d", {"a": "linspace(0, 1, 1000)"})
            assert small.datasets.get("d").n_rows() == 1000
        finally:
            small.close_all()

    def test_load_csv(self, box, root):
        (root / "run 1.csv").write_text("time,value\n0,1\n1,3\n2,5\n")
        result = box.load_dataset("run 1.csv")
        assert result.data["name"] == "run_1"
        assert [c["name"] for c in result.data["columns"]] == ["time", "value"]

    def test_load_npy_matrix_makes_one_column_per_array_column(self, box, root):
        np.save(root / "m.npy", np.arange(12.0).reshape(4, 3))
        box.load_dataset("m.npy", name="m")
        assert box.datasets.get("m").n_cols() == 3

    def test_load_refuses_files_outside_the_sandbox(self, box, tmp_path):
        (tmp_path / "secret.csv").write_text("a\n1\n")
        with pytest.raises(McpError, match="outside"):
            box.load_dataset(str(tmp_path / "secret.csv"))

    def test_load_reports_a_corrupt_file_without_a_traceback(self, box, root):
        (root / "bad.npy").write_bytes(b"not a numpy file")
        with pytest.raises(McpError):
            box.load_dataset("bad.npy")

    def test_pickled_arrays_are_refused(self, box, root):
        np.save(root / "obj.npy", np.array([{"a": 1}], dtype=object), allow_pickle=True)
        with pytest.raises(McpError):
            box.load_dataset("obj.npy")

    def test_describe_lists_all_or_one(self, box):
        make_dataset(box)
        listing = box.describe_dataset().data["datasets"]
        assert listing[0]["name"] == "d" and "t" in listing[0]["columns"]
        one = box.describe_dataset("d", rows=2).data
        assert one["rows"] == 50 and len(one["head"]) == 2

    def test_delete(self, box):
        make_dataset(box)
        box.delete_dataset("d")
        with pytest.raises(McpError):
            box.describe_dataset("d")


class TestAddPlotKinds:
    """Every kind must draw, show up as an artist, and survive a render."""

    @pytest.mark.parametrize("kind", PLOT_KINDS)
    def test_kind_draws_and_renders(self, box, kind):
        make_dataset(box)
        fig = box.create_figure().data["figure_id"]
        args = {"dataset": "d", "x": "t", "y": "y"}
        if kind in ("hist", "box", "violin"):
            args = {"dataset": "d", "y": "y"}
        elif kind == "pie":
            args = {"y": [3, 1, 2], "tick_labels": ["a", "b", "c"]}
        elif kind in ("bar", "barh"):
            args = {"x": [1, 2, 3], "y": [3, 1, 2], "tick_labels": ["a", "b", "c"]}
        elif kind == "errorbar":
            args["error"] = [0.1]
        result = box.add_plot(fig, kind, **args)
        assert result.data["artist_id"] == "a1"
        assert result.data["layers"] >= 1
        rendered = box.render_figure(fig)
        assert rendered.image.startswith(PNG)
        assert [a["kind"] for a in box.describe_figure(fig).data["artists"]] == [kind]

    def test_unknown_kind_suggests(self, box):
        fig = box.create_figure().data["figure_id"]
        with pytest.raises(McpError, match="Did you mean 'scatter'"):
            box.add_plot(fig, "scater", y=[1])

    def test_x_defaults_to_the_index(self, box):
        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "line", y=[5, 6, 7])
        layer = scene_layers(box, fig)[0]
        assert np.asarray(layer.pts)[:, 0].tolist() == [0.0, 1.0, 2.0]

    def test_inline_and_column_arguments_agree(self, box):
        make_dataset(box)
        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "line", dataset="d", x="t", y="y")
        by_column = np.asarray(scene_layers(box, fig)[0].pts).copy()
        box.remove_artist(fig)
        box.add_plot(
            fig,
            "line",
            x=box.datasets.column("d", "t").tolist(),
            y=box.datasets.column("d", "y").tolist(),
        )
        np.testing.assert_allclose(np.asarray(scene_layers(box, fig)[0].pts), by_column, atol=1e-5)

    def test_column_without_a_dataset_explains_how_to_fix_it(self, box):
        fig = box.create_figure().data["figure_id"]
        with pytest.raises(McpError, match="create_dataset"):
            box.add_plot(fig, "line", y="signal")

    def test_wrong_column_lists_the_columns(self, box):
        make_dataset(box)
        fig = box.create_figure().data["figure_id"]
        with pytest.raises(McpError, match="columns: t, y, z, g"):
            box.add_plot(fig, "line", dataset="d", x="t", y="q")

    def test_length_mismatch_is_reported(self, box):
        fig = box.create_figure().data["figure_id"]
        with pytest.raises(McpError, match="x has 3 values but y has 2"):
            box.add_plot(fig, "line", x=[1, 2, 3], y=[1, 2])

    def test_empty_series_is_refused(self, box):
        fig = box.create_figure().data["figure_id"]
        with pytest.raises(McpError, match="no data"):
            box.add_plot(fig, "line", y=[])

    def test_errorbar_requires_error(self, box):
        fig = box.create_figure().data["figure_id"]
        with pytest.raises(McpError, match="error"):
            box.add_plot(fig, "errorbar", x=[1, 2], y=[1, 2])

    def test_options_a_kind_does_not_use_are_reported_not_fatal(self, box):
        fig = box.create_figure().data["figure_id"]
        result = box.add_plot(fig, "bar", x=[1, 2], y=[1, 2], marker="o", linewidth=3)
        assert set(result.data["ignored"]) == {"marker", "linewidth"}
        assert "Ignored" in result.message

    def test_unused_scatter_colour_and_size_encodings(self, box):
        make_dataset(box)
        fig = box.create_figure().data["figure_id"]
        box.add_plot(
            fig,
            "scatter",
            dataset="d",
            x="t",
            y="y",
            color_by="z",
            size_by="t",
            cmap="plasma",
            colorbar=True,
        )
        assert box.render_figure(fig).image.startswith(PNG)

    def test_series_without_a_colour_take_the_cycle_not_black(self, box):
        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "scatter", x=[1, 2], y=[1, 2])
        box.add_plot(fig, "scatter", x=[1, 2], y=[2, 1])
        colours = [
            tuple(np.round(np.asarray(layer.colors)[0], 3)) for layer in scene_layers(box, fig)
        ]
        assert colours[0] != colours[1]
        assert all(c[:3] != (0.0, 0.0, 0.0) for c in colours)

    def test_an_explicit_colour_wins_over_the_cycle(self, box):
        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "scatter", x=[1, 2], y=[1, 2], color="red")
        assert tuple(np.round(np.asarray(scene_layers(box, fig)[0].colors)[0][:3], 2)) == (
            1.0,
            0.0,
            0.0,
        )

    def test_area_is_opaque_unless_asked_otherwise(self, box):
        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "area", x=[0, 1, 2], y=[1, 2, 1])
        assert float(scene_layers(box, fig)[0].style.face_color[3]) == 1.0
        box.add_plot(fig, "area", x=[0, 1, 2], y=[1, 2, 1], alpha=0.4)
        assert float(scene_layers(box, fig)[1].style.face_color[3]) == pytest.approx(0.4)

    def test_box_with_columns_makes_one_box_per_column(self, box):
        make_dataset(box)
        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "box", dataset="d", columns=["y", "z"])
        assert box.render_figure(fig).image.startswith(PNG)

    def test_negative_pie_values_are_refused(self, box):
        fig = box.create_figure().data["figure_id"]
        with pytest.raises(McpError, match="non-negative"):
            box.add_plot(fig, "pie", y=[1, -1])

    def test_a_failed_draw_leaves_the_figure_as_it_was(self, box):
        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "line", y=[1, 2, 3])
        before = len(scene_layers(box, fig))
        with pytest.raises(McpError):
            box.add_plot(fig, "line", y=[1, 2, 3], color="not-a-colour")
        assert len(scene_layers(box, fig)) == before
        assert [a["artist_id"] for a in box.describe_figure(fig).data["artists"]] == ["a1"]

    def test_unknown_colormap_is_a_readable_error(self, box):
        fig = box.create_figure().data["figure_id"]
        with pytest.raises(McpError):
            box.add_plot(fig, "hist2d", x=[1, 2, 3], y=[1, 2, 3], cmap="no-such-map")


class TestThreeD:
    def test_line_and_scatter_take_z(self, box):
        fig = box.create_figure(projection="3d").data["figure_id"]
        box.add_plot(fig, "line", x=[0, 1, 2], y=[0, 1, 2], z=[0, 1, 4])
        box.add_plot(fig, "scatter", x=[0, 1, 2], y=[0, 1, 2], z=[0, 1, 4], color="red")
        assert box.render_figure(fig).image.startswith(PNG)

    def test_z_is_required_in_3d_and_forbidden_in_2d(self, box):
        flat = box.create_figure().data["figure_id"]
        deep = box.create_figure(projection="3d").data["figure_id"]
        with pytest.raises(McpError, match="z is required"):
            box.add_plot(deep, "line", x=[0, 1], y=[0, 1])
        with pytest.raises(McpError, match="2-D"):
            box.add_plot(flat, "line", x=[0, 1], y=[0, 1], z=[0, 1])

    def test_other_kinds_are_refused_in_3d_with_the_alternative(self, box):
        fig = box.create_figure(projection="3d").data["figure_id"]
        with pytest.raises(McpError, match="surface"):
            box.add_plot(fig, "bar", x=[1], y=[1], z=[1])

    def test_surface_renders_and_removal_keeps_the_axes_chrome(self, box):
        fig = box.create_figure(projection="3d").data["figure_id"]
        box.add_field(fig, "surface", expression="sin(x)*cos(y)", x_range=[-3, 3], y_range=[-3, 3])
        box.add_plot(fig, "scatter", x=[0], y=[0], z=[0])
        chrome_before = [layer for layer in scene_layers(box, fig) if layer.label.startswith("3D")]
        box.remove_artist(fig, "a1")
        assert [a["artist_id"] for a in box.describe_figure(fig).data["artists"]] == ["a2"]
        chrome_after = [layer for layer in scene_layers(box, fig) if layer.label.startswith("3D")]
        assert len(chrome_after) == len(chrome_before) > 0
        assert box.render_figure(fig).image.startswith(PNG)

    def test_surface_needs_a_3d_figure_and_vice_versa(self, box):
        flat = box.create_figure().data["figure_id"]
        deep = box.create_figure(projection="3d").data["figure_id"]
        with pytest.raises(McpError, match="3-D figure"):
            box.add_field(flat, "surface", expression="x")
        with pytest.raises(McpError, match="surface"):
            box.add_field(deep, "image", expression="x")

    def test_camera_and_zlabel(self, box):
        fig = box.create_figure(projection="3d").data["figure_id"]
        box.add_plot(fig, "line", x=[0, 1], y=[0, 1], z=[0, 1])
        box.set_axes(fig, zlabel="height", zlim=[0, 2], elev=10, azim=20)
        panel = box.render_figure(fig).data["panels"][0]
        assert panel["zlabel"] == "height" and panel["zlim"] == [0.0, 2.0]

    def test_3d_only_options_are_refused_on_2d(self, box):
        fig = box.create_figure().data["figure_id"]
        for kwargs in ({"zlabel": "z"}, {"zlim": [0, 1]}, {"elev": 10}):
            with pytest.raises(McpError, match="3-D|2-D"):
                box.set_axes(fig, **kwargs)


class TestAddFunction:
    def test_parameters_and_default_label(self, box):
        fig = box.create_figure().data["figure_id"]
        result = box.add_function(fig, "a * x ** 2", 0, 2, samples=5, parameters={"a": 3})
        layer = scene_layers(box, fig)[0]
        assert np.asarray(layer.pts)[:, 1].tolist() == pytest.approx([0, 0.75, 3, 6.75, 12])
        assert layer.label == "a * x ** 2"
        assert "a1" in result.message

    def test_large_sample_counts_bypass_the_inline_cap(self, root):
        small = Toolbox(FileSandbox([root]), Limits(max_inline_values=10))
        try:
            fig = small.create_figure().data["figure_id"]
            small.add_function(fig, "x", 0, 1, samples=500)
        finally:
            small.close_all()

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"expression": "nope(x)"}, "Could not evaluate"),
            ({"expression": "x", "x_min": 3, "x_max": 1}, "x_min"),
            ({"expression": "x", "samples": 1}, "samples"),
            ({"expression": "x", "parameters": {"x": 1}}, "identifier"),
            ({"expression": "x", "parameters": {"a": float("nan")}}, "finite"),
            ({"expression": "x", "kind": "bar"}, "kind"),
            ({"expression": "zeros(10**9)"}, "limit"),
            ({"expression": "x", "samples": 10**9}, "samples"),
        ],
    )
    def test_bad_input_is_refused(self, box, kwargs, match):
        fig = box.create_figure().data["figure_id"]
        args = {"expression": "x", "x_min": 0, "x_max": 1, **kwargs}
        with pytest.raises(McpError, match=match):
            box.add_function(fig, **args)

    def test_a_constant_expression_repeats(self, box):
        fig = box.create_figure().data["figure_id"]
        box.add_function(fig, "2", 0, 1, samples=4)
        assert np.asarray(scene_layers(box, fig)[0].pts)[:, 1].tolist() == [2.0] * 4


class TestAddField:
    def test_expression_matrix_and_file_sources(self, box, root):
        np.save(root / "f.npy", np.add.outer(np.linspace(-1, 1, 20), np.linspace(-1, 1, 30)) ** 2)
        fig = box.create_figure().data["figure_id"]
        box.add_field(fig, "image", expression="x*y", x_range=[0, 1], y_range=[0, 1], resolution=20)
        box.add_field(fig, "pcolormesh", matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        box.add_field(fig, "contour", file="f.npy", x_range=[-1, 1], y_range=[-1, 1], levels=5)
        assert len(box.describe_figure(fig).data["artists"]) == 3
        assert box.render_figure(fig).image.startswith(PNG)

    def test_exactly_one_source(self, box):
        fig = box.create_figure().data["figure_id"]
        with pytest.raises(McpError, match="exactly one"):
            box.add_field(fig, "image")
        with pytest.raises(McpError, match="exactly one"):
            box.add_field(fig, "image", expression="x", matrix=[[1, 2], [3, 4]])

    def test_options_a_kind_does_not_take_are_dropped_and_reported(self, box):
        fig = box.create_figure(projection="3d").data["figure_id"]
        result = box.add_field(fig, "wireframe", expression="x*y", cmap="viridis", resolution=10)
        assert result.data["ignored"] == ["cmap"]

    def test_resolution_is_bounded(self, box):
        fig = box.create_figure().data["figure_id"]
        with pytest.raises(McpError, match="resolution"):
            box.add_field(fig, "image", expression="x", resolution=100000)

    def test_ragged_and_tiny_matrices_are_refused(self, box):
        fig = box.create_figure().data["figure_id"]
        with pytest.raises(McpError, match="same length|2-D"):
            box.add_field(fig, "image", matrix=[[1, 2], [3]])
        with pytest.raises(McpError, match="2x2"):
            box.add_field(fig, "image", matrix=[[1, 2]])

    def test_file_must_be_inside_the_sandbox_and_2d(self, box, root, tmp_path):
        np.save(tmp_path / "out.npy", np.zeros((3, 3)))
        np.save(root / "cube.npy", np.zeros((2, 2, 2)))
        fig = box.create_figure().data["figure_id"]
        with pytest.raises(McpError, match="outside"):
            box.add_field(fig, "image", file=str(tmp_path / "out.npy"))
        with pytest.raises(McpError, match="2-D"):
            box.add_field(fig, "image", file="cube.npy")

    def test_a_colour_image_file(self, box, root):
        import matplotlib.image as mimage

        mimage.imsave(str(root / "pic.png"), np.random.default_rng(0).random((12, 16, 3)))
        fig = box.create_figure().data["figure_id"]
        box.add_field(fig, "image", file="pic.png")
        with pytest.raises(McpError, match="kind='image'"):
            box.add_field(fig, "contour", file="pic.png")
        assert box.render_figure(fig).image.startswith(PNG)

    def test_colorbar_failure_does_not_lose_the_plot(self, box, monkeypatch):
        def refuse(*args, **kwargs):
            raise RuntimeError("no colormapped layer")

        monkeypatch.setattr(gplt, "colorbar", refuse)
        fig = box.create_figure().data["figure_id"]
        result = box.add_field(fig, "image", expression="x", colorbar=True, resolution=5)
        assert "Colorbar not added" in result.message
        assert len(box.describe_figure(fig).data["artists"]) == 1

    def test_expression_cannot_allocate_a_huge_array(self, box):
        fig = box.create_figure().data["figure_id"]
        with pytest.raises(McpError, match="exceeds the limit"):
            box.add_field(fig, "image", expression="x + zeros(10**9)", resolution=4)


class TestAnnotations:
    def test_text_hline_vline(self, box):
        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "line", x=[0, 10], y=[0, 1])
        box.add_annotation(fig, "hline", y=0.5, label="limit", color="gray", linestyle="--")
        box.add_annotation(fig, "vline", x=5)
        box.add_annotation(fig, "text", x=2, y=0.2, text="note")
        assert [a["kind"] for a in box.describe_figure(fig).data["artists"]] == [
            "line",
            "hline",
            "vline",
            "text",
        ]
        assert box.render_figure(fig).image.startswith(PNG)

    @pytest.mark.parametrize(
        "kwargs",
        [{"kind": "text", "x": 1}, {"kind": "hline"}, {"kind": "vline"}, {"kind": "arrow"}],
    )
    def test_missing_arguments_are_refused(self, box, kwargs):
        fig = box.create_figure().data["figure_id"]
        with pytest.raises(McpError):
            box.add_annotation(fig, **kwargs)

    def test_refused_on_3d(self, box):
        fig = box.create_figure(projection="3d").data["figure_id"]
        with pytest.raises(McpError, match="2-D"):
            box.add_annotation(fig, "hline", y=1)


class TestRemoveArtist:
    def test_one_call_is_removed_as_a_unit(self, box):
        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "bar", x=[1, 2, 3, 4], y=[1, 2, 3, 4])
        box.add_plot(fig, "line", y=[1, 2, 3])
        bar_layers = box.describe_figure(fig).data["artists"][0]["layers"]
        assert bar_layers > 1  # one patch per bar: the reason artists exist
        total = len(scene_layers(box, fig))
        box.remove_artist(fig, "a1")
        assert len(scene_layers(box, fig)) == total - bar_layers
        assert [a["artist_id"] for a in box.describe_figure(fig).data["artists"]] == ["a2"]
        assert box.render_figure(fig).image.startswith(PNG)

    def test_no_id_removes_everything(self, box):
        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "line", y=[1, 2])
        box.add_plot(fig, "scatter", x=[1], y=[1])
        box.remove_artist(fig)
        assert box.describe_figure(fig).data["artists"] == []
        assert scene_layers(box, fig) == []

    def test_unknown_id_lists_the_real_ones(self, box):
        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "line", y=[1, 2])
        with pytest.raises(McpError, match="a1"):
            box.remove_artist(fig, "a9")

    def test_artist_ids_are_not_reused(self, box):
        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "line", y=[1, 2])
        box.remove_artist(fig, "a1")
        assert box.add_plot(fig, "line", y=[1, 2]).data["artist_id"] == "a2"

    def test_removal_targets_the_right_panel(self, box):
        fig = box.create_figure(rows=1, cols=2).data["figure_id"]
        box.add_plot(fig, "line", y=[1, 2], panel=1)
        box.add_plot(fig, "line", y=[3, 4], panel=2)
        box.remove_artist(fig, "a2")
        plot = box._figures[fig].plot
        assert len(plot.panels[0].scene.layers) == 1
        assert len(plot.panels[1].scene.layers) == 0


class TestMultiPanel:
    def test_each_panel_gets_its_own_data_and_labels(self, box):
        fig = box.create_figure(rows=1, cols=2, title="Both").data["figure_id"]
        box.add_plot(fig, "line", y=[1, 2, 3], panel=1)
        box.add_plot(fig, "hist", y=[1, 1, 2, 3, 3, 3], panel=2)
        box.set_axes(fig, panel=1, title="Left", xlabel="a")
        box.set_axes(fig, panel=2, title="Right", xlabel="b")
        panels = box.render_figure(fig).data["panels"]
        assert [p["title"] for p in panels] == ["Left", "Right"]
        assert [p["xlabel"] for p in panels] == ["a", "b"]

    def test_panel_must_exist(self, box):
        single = box.create_figure().data["figure_id"]
        grid = box.create_figure(rows=2, cols=2).data["figure_id"]
        with pytest.raises(McpError, match="single panel"):
            box.add_plot(single, "line", y=[1], panel=2)
        with pytest.raises(McpError, match="between 1 and 4"):
            box.add_plot(grid, "line", y=[1], panel=5)

    def test_spacing_survives_selecting_a_panel(self, box):
        fig = box.create_figure(rows=1, cols=2).data["figure_id"]
        before = [p.rect_frac for p in box._figures[fig].plot.panels]
        box.add_plot(fig, "line", y=[1, 2], panel=2)
        box.set_axes(fig, panel=1, title="x")
        assert [p.rect_frac for p in box._figures[fig].plot.panels] == before
        assert before[0][2] < 0.5  # a gap was left between the panels

    def test_scales_are_refused_rather_than_drawn_on_the_wrong_panel(self, box):
        fig = box.create_figure(rows=1, cols=2).data["figure_id"]
        with pytest.raises(McpError, match="figure-wide"):
            box.set_axes(fig, panel=2, yscale="log")

    def test_scales_work_on_a_single_panel(self, box):
        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "line", x=[1, 10, 100], y=[1, 10, 100])
        box.set_axes(fig, xscale="log", yscale="log")
        panel = box.render_figure(fig).data["panels"][0]
        assert panel["xscale"] == "log" and panel["yscale"] == "log"

    def test_grid_figure_title_is_a_suptitle(self, box):
        fig = box.create_figure(rows=1, cols=2, title="Overall").data["figure_id"]
        box.add_plot(fig, "line", y=[1, 2], panel=1)
        box.add_plot(fig, "line", y=[1, 2], panel=2)
        assert box.render_figure(fig).image.startswith(PNG)


class TestSetAxes:
    def test_reports_what_it_applied(self, box):
        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "line", y=[1, 2, 3], label="s")
        result = box.set_axes(
            fig, title="T", xlabel="x", ylabel="y", xlim=[0, 5], ylim=[0, 4], grid=True, legend=True
        )
        assert result.data["applied"] == [
            "title",
            "xlabel",
            "ylabel",
            "xlim",
            "ylim",
            "grid",
            "legend",
        ]
        panel = box.render_figure(fig).data["panels"][0]
        assert panel["xlim"] == [0.0, 5.0] and panel["ylim"] == [0.0, 4.0]
        assert panel["legend"] == ["s"]

    def test_nothing_to_set_is_an_error(self, box):
        fig = box.create_figure().data["figure_id"]
        with pytest.raises(McpError, match="Nothing to set"):
            box.set_axes(fig)

    @pytest.mark.parametrize(
        "kwargs",
        [{"xlim": [1]}, {"xlim": [1, 1]}, {"ylim": [0, float("inf")]}, {"xscale": "weird"}],
    )
    def test_bad_values_are_refused(self, box, kwargs):
        fig = box.create_figure().data["figure_id"]
        with pytest.raises(McpError):
            box.set_axes(fig, **kwargs)

    def test_equal_aspect_is_applied_to_the_export(self, box, monkeypatch):
        seen = []
        original = toolbox_module._summarise_axes

        def spy(fig):
            seen.append([ax.get_aspect() for ax in fig.axes])
            return original(fig)

        monkeypatch.setattr(toolbox_module, "_summarise_axes", spy)
        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "line", x=[0, 1], y=[0, 5])
        box.render_figure(fig)
        box.set_axes(fig, equal_aspect=True)
        box.render_figure(fig)
        box.set_axes(fig, equal_aspect=False)
        box.render_figure(fig)
        assert seen[0] == ["auto"] and seen[1] == [1.0] and seen[2] == ["auto"]

    def test_pie_is_drawn_round_without_being_asked(self, box, monkeypatch):
        seen = []
        original = toolbox_module._summarise_axes
        monkeypatch.setattr(
            toolbox_module,
            "_summarise_axes",
            lambda fig: (seen.append(fig.axes[0].get_aspect()), original(fig))[1],
        )
        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "pie", y=[1, 2, 3])
        box.render_figure(fig)
        assert seen == [1.0]

    def test_style_preset(self, box):
        fig = box.create_figure().data["figure_id"]
        assert box.set_style(fig, "chalk").data["style"] == "chalk"
        with pytest.raises(McpError, match="Available"):
            box.set_style(fig, "nope")


class TestRenderAndSave:
    def test_render_returns_png_and_a_summary(self, box):
        fig = box.create_figure(width=900, height=600).data["figure_id"]
        box.add_plot(fig, "line", y=[1, 2, 3], label="s")
        box.set_axes(fig, title="T", xlabel="x", ylabel="y", legend=True)
        result = box.render_figure(fig)
        assert result.image.startswith(PNG) and result.image_format == "png"
        panel = result.data["panels"][0]
        assert panel["title"] == "T" and panel["legend"] == ["s"]
        json.dumps(result.data)

    def test_long_edge_is_capped(self, root):
        small = Toolbox(FileSandbox([root]), Limits(max_image_edge=500))
        try:
            fig = small.create_figure(width=2000, height=1200).data["figure_id"]
            small.add_plot(fig, "line", y=[1, 2, 3])
            width, height = png_size(small.render_figure(fig).image)
            assert max(width, height) <= 520  # tight bbox pads a little past the cap
        finally:
            small.close_all()

    def test_scale_changes_the_size(self, box):
        fig = box.create_figure(width=600, height=400).data["figure_id"]
        box.add_plot(fig, "line", y=[1, 2, 3])
        small = png_size(box.render_figure(fig, scale=0.5).image)
        large = png_size(box.render_figure(fig, scale=1.5).image)
        assert large[0] > small[0] * 2

    def test_scale_is_bounded(self, box):
        fig = box.create_figure().data["figure_id"]
        for scale in (0.01, 99):
            with pytest.raises(McpError, match="scale"):
                box.render_figure(fig, scale=scale)

    def test_render_does_not_change_the_figure(self, box):
        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "line", y=[1, 2, 3])
        first = box.describe_figure(fig).data
        box.render_figure(fig)
        box.render_figure(fig)
        assert box.describe_figure(fig).data == first

    def test_empty_figure_renders(self, box):
        fig = box.create_figure().data["figure_id"]
        assert box.render_figure(fig).image.startswith(PNG)

    @pytest.mark.parametrize(
        "name, magic",
        [("a.png", PNG), ("a.jpg", b"\xff\xd8"), ("a.pdf", b"%PDF"), ("a.svg", b"<?xml")],
    )
    def test_save_formats(self, box, root, name, magic):
        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "line", y=[1, 2, 3])
        result = box.save_figure(fig, name)
        written = (root / name).read_bytes()
        assert written.startswith(magic) and result.data["bytes"] == len(written)

    def test_save_never_clobbers_unless_told(self, box, root):
        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "line", y=[1, 2, 3])
        box.save_figure(fig, "f.png")
        first = (root / "f.png").read_bytes()
        with pytest.raises(McpError, match="overwrite"):
            box.save_figure(fig, "f.png", scale=2.0)
        assert (root / "f.png").read_bytes() == first
        box.save_figure(fig, "f.png", scale=2.0, overwrite=True)
        assert (root / "f.png").read_bytes() != first

    def test_save_is_confined(self, box, root):
        fig = box.create_figure().data["figure_id"]
        with pytest.raises(McpError, match="outside"):
            box.save_figure(fig, "../escape.png")
        with pytest.raises(McpError, match="Cannot save"):
            box.save_figure(fig, "payload.py")
        assert not list(root.parent.glob("escape*"))

    def test_a_refused_save_writes_nothing(self, box, root):
        fig = box.create_figure().data["figure_id"]
        with pytest.raises(McpError):
            box.save_figure(fig, "sub/dir/x.exe")
        assert not (root / "sub").exists() or not any((root / "sub").rglob("*"))

    def test_transparent_png(self, box, root):
        from PIL import Image

        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "line", y=[1, 2, 3])
        box.save_figure(fig, "t.png", transparent=True)
        assert Image.open(root / "t.png").mode == "RGBA"

    def test_resource_png(self, box):
        fig = box.create_figure().data["figure_id"]
        assert box.render_png(fig).startswith(PNG)
        with pytest.raises(McpError):
            box.render_png("fig99")


class TestProcessHygiene:
    def test_stray_prints_go_to_stderr_not_the_protocol_channel(self, box, monkeypatch, capsys):
        real = gplt.title

        def noisy(*args, **kwargs):
            print("library chatter")
            return real(*args, **kwargs)

        monkeypatch.setattr(gplt, "title", noisy)
        fig = box.create_figure().data["figure_id"]
        box.set_axes(fig, title="T")
        captured = capsys.readouterr()
        assert "library chatter" not in captured.out
        assert "library chatter" in captured.err

    def test_compat_warnings_reach_the_model(self, box, monkeypatch):
        real = gplt.title

        def warns(*args, **kwargs):
            warnings.warn(
                "title(fontsize=...) has no effect.", MatplotlibCompatWarning, stacklevel=2
            )
            return real(*args, **kwargs)

        monkeypatch.setattr(gplt, "title", warns)
        fig = box.create_figure().data["figure_id"]
        result = box.set_axes(fig, title="T")
        assert result.data["warnings"] == ["title(fontsize=...) has no effect."]
        assert "warnings" in result.text()

    def test_unrelated_warnings_are_not_forwarded(self, box, monkeypatch):
        real = gplt.title

        def warns(*args, **kwargs):
            warnings.warn("numpy internals", RuntimeWarning, stacklevel=2)
            return real(*args, **kwargs)

        monkeypatch.setattr(gplt, "title", warns)
        fig = box.create_figure().data["figure_id"]
        assert "warnings" not in box.set_axes(fig, title="T").data

    def test_unexpected_errors_are_not_swallowed(self, box, monkeypatch):
        def broken(*args, **kwargs):
            raise ZeroDivisionError("a real bug")

        monkeypatch.setattr(gplt, "title", broken)
        fig = box.create_figure().data["figure_id"]
        with pytest.raises(ZeroDivisionError):
            box.set_axes(fig, title="T")

    def test_closing_everything_leaves_pyplot_clean(self, box):
        plots = []
        for _ in range(3):
            fig = box.create_figure().data["figure_id"]
            plots.append(box._figures[fig].plot)
        box.close_all()
        assert box.figure_ids() == []
        assert not any(plot in gplt._ALL_PLOTS for plot in plots)

    def test_concurrent_calls_do_not_cross_figures(self, box):
        """The SDK runs sync tools on worker threads; pyplot's current-figure is global."""
        ids = [box.create_figure(title=f"F{i}").data["figure_id"] for i in range(4)]
        errors = []
        barrier = threading.Barrier(len(ids))

        def work(index, figure_id):
            try:
                barrier.wait(timeout=10)
                for step in range(6):
                    box.add_plot(figure_id, "line", y=[index, index + 1, step])
                    box.set_axes(figure_id, xlabel=f"x{index}")
                panel = box.render_figure(figure_id).data["panels"][0]
                assert panel["title"] == f"F{index}"
                assert panel["xlabel"] == f"x{index}"
            except Exception as exc:  # pragma: no cover - failure path
                errors.append((index, repr(exc)))

        with ThreadPoolExecutor(max_workers=len(ids)) as pool:
            list(pool.map(work, range(len(ids)), ids))
        assert not errors
        for figure_id in ids:
            assert len(box.describe_figure(figure_id).data["artists"]) == 6
            assert len(scene_layers(box, figure_id)) == 6


class TestDescribeFigure:
    def test_listing_and_detail(self, box):
        assert box.describe_figure().data["figures"] == []
        fig = box.create_figure(rows=1, cols=2).data["figure_id"]
        box.add_plot(fig, "line", y=[1, 2], label="s", panel=2)
        listing = box.describe_figure().data["figures"][0]
        assert listing["figure_id"] == fig and listing["artists"] == 1
        detail = box.describe_figure(fig).data
        assert detail["grid"] == [1, 2] and detail["projection"] == "2d"
        assert detail["artists"][0]["panel"] == 2 and detail["artists"][0]["label"] == "s"
        assert len(detail["panels"]) == 2

    def test_dataset_name_is_recorded_on_the_artist(self, box):
        make_dataset(box)
        fig = box.create_figure().data["figure_id"]
        box.add_plot(fig, "line", dataset="d", x="t", y="y")
        assert box.describe_figure(fig).data["artists"][0]["dataset"] == "d"


class TestResultText:
    def test_text_is_the_message_then_one_json_line(self, box):
        result = box.create_figure()
        message, _, payload = result.text().partition("\n")
        assert message.startswith("Created fig1")
        assert json.loads(payload) == {"figure_id": "fig1", "panels": 1}

    def test_numpy_scalars_and_nan_serialise(self):
        from glplot.mcp.toolbox import ToolResult

        text = ToolResult(
            "m", {"a": np.float32(1.5), "b": np.int64(3), "c": np.float64("nan")}
        ).text()
        assert json.loads(text.split("\n")[1]) == {"a": 1.5, "b": 3, "c": None}
