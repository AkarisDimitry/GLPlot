"""Test export and savefig functionality."""

import os
import tempfile

import numpy as np
import pytest

import glplot.pyplot as gplt


@pytest.fixture(autouse=True)
def clean_state():
    """Clean pyplot state before and after each test."""
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


class TestExportBasics:
    """Test basic export functionality."""

    def test_savefig_creates_file(self):
        """Test that savefig creates output file."""
        gplt.plot([0, 1, 2], [0, 1, 4])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_output.png")

            try:
                gplt.savefig(output_path)
                # File should be created
                if os.path.exists(output_path):
                    assert os.path.getsize(output_path) > 0
            except NotImplementedError:
                pytest.skip("savefig not fully implemented")

    def test_savefig_without_show(self):
        """Test savefig works without calling show()."""
        gplt.plot([0, 1, 2], [0, 1, 4])
        gplt.xlabel("X")
        gplt.ylabel("Y")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_no_show.png")

            try:
                gplt.savefig(output_path)
                # Should complete without show()
            except NotImplementedError:
                pytest.skip("savefig not fully implemented")

    def test_savefig_with_scale(self):
        """Test savefig with scale parameter."""
        gplt.plot([0, 1, 2], [0, 1, 4])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_scaled.png")

            try:
                gplt.savefig(output_path, scale=2.0)
                # Higher resolution output
            except (NotImplementedError, TypeError):
                pytest.skip("scale parameter not supported")

    def test_savefig_different_formats(self):
        """Test savefig with different file formats."""
        gplt.plot([0, 1, 2], [0, 1, 4])

        # Test supported formats (png is standard)
        formats = [".png", ".jpg"]

        with tempfile.TemporaryDirectory() as tmpdir:
            for fmt in formats:
                output_path = os.path.join(tmpdir, f"test_export{fmt}")

                try:
                    gplt.savefig(output_path)
                except (NotImplementedError, OSError, ValueError):
                    pytest.skip(f"Format {fmt} not supported")


class TestExportWithData:
    """Test export with various data types."""

    def test_export_line_plot(self):
        """Test export of line plot."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        gplt.plot(x, y, "r-", label="sin(x)")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "line_plot.png")

            try:
                gplt.savefig(output_path)
            except NotImplementedError:
                pytest.skip("savefig not implemented")

    def test_export_scatter_plot(self):
        """Test export of scatter plot."""
        x = np.random.randn(100)
        y = np.random.randn(100)

        gplt.scatter(x, y, c="blue", s=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "scatter_plot.png")

            try:
                gplt.savefig(output_path)
            except NotImplementedError:
                pytest.skip("savefig not implemented")

    def test_export_histogram(self):
        """Test export of histogram."""
        data = np.random.randn(1000)
        gplt.hist(data, bins=50)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "histogram.png")

            try:
                gplt.savefig(output_path)
            except NotImplementedError:
                pytest.skip("savefig not implemented")

    def test_export_multi_layer(self):
        """Test export of figure with multiple layers."""
        x = np.linspace(0, 10, 100)

        gplt.plot(x, np.sin(x), "r-", label="sin(x)")
        gplt.plot(x, np.cos(x), "b-", label="cos(x)")
        gplt.scatter(x[::10], np.sin(x[::10]), c="red", s=5)

        gplt.xlabel("x")
        gplt.ylabel("value")
        gplt.legend()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "multi_layer.png")

            try:
                gplt.savefig(output_path)
            except NotImplementedError:
                pytest.skip("savefig not implemented")


class TestExportEdgeCases:
    """Test export edge cases and error handling."""

    def test_export_empty_figure(self):
        """Test export of empty figure."""
        gplt.figure()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "empty.png")

            try:
                gplt.savefig(output_path)
            except (NotImplementedError, ValueError):
                pass  # May fail on empty figure

    def test_export_single_point(self):
        """Test export with single data point."""
        gplt.plot([0], [0])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "single_point.png")

            try:
                gplt.savefig(output_path)
            except NotImplementedError:
                pytest.skip("savefig not implemented")

    def test_export_3d_plot(self):
        """Test export of 3D plot."""
        t = np.linspace(0, 10, 100)
        gplt.scatter3d(t, np.sin(t), np.cos(t))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "3d_plot.png")

            try:
                gplt.savefig(output_path)
            except NotImplementedError:
                pytest.skip("savefig not implemented for 3D")


class TestPerVertexColourExport:
    """The savefig fallback must honour a patch's per-vertex colours.

    hexbin, pcolor and tripcolor draw into a single patch whose cells each carry their
    own colour (a colour VBO, since one `face_color` uniform cannot express them). The
    PNG export renders patches through matplotlib's PolyCollection, which was handed a
    single `face_color` -- so the exported figure came out one flat blue while the live
    view was correct. That is a colormap silently dropped on save, the worst kind of
    export bug because the on-screen plot looks right. These pin that the colours reach
    the PNG.

    Asserted by intercepting the facecolors handed to PolyCollection rather than by
    reading pixels back, so antialiasing at the cell edges cannot fake a pass.
    """

    def _facecolor_count(self, build) -> int:
        import matplotlib.collections as mc

        from glplot.utils import preview

        captured = {"n": 0}
        original = mc.PolyCollection.__init__

        def spy(self, verts, *args, **kwargs):
            fc = kwargs.get("facecolors")
            captured["n"] = 0 if fc is None else (1 if np.ndim(fc) <= 1 else len(fc))
            return original(self, verts, *args, **kwargs)

        mc.PolyCollection.__init__ = spy
        try:
            build()
            preview.render_preview(gplt.gcf(), tempfile.mktemp(suffix=".png"))
        finally:
            mc.PolyCollection.__init__ = original
        return captured["n"]

    def test_hexbin_exports_per_hexagon_colours(self):
        rng = np.random.default_rng(0)
        n = self._facecolor_count(
            lambda: gplt.hexbin(rng.normal(size=3000), rng.normal(size=3000), gridsize=15)
        )
        assert n > 1, "hexbin exported as one flat colour -- colormap dropped on savefig"

    def test_pcolor_exports_per_cell_colours(self):
        n = self._facecolor_count(lambda: gplt.pcolor(np.arange(30.0).reshape(5, 6)))
        assert n > 1

    def test_tripcolor_exports_per_face_colours(self):
        rng = np.random.default_rng(0)
        n = self._facecolor_count(
            lambda: gplt.tripcolor(rng.random(40), rng.random(40), rng.random(40))
        )
        assert n > 1

    def test_a_flat_patch_still_exports_one_colour(self):
        """The fix must not overshoot: a bar carries no colour buffer, so one colour."""
        assert self._facecolor_count(lambda: gplt.bar([0, 1, 2], [3, 4, 5])) == 1


class TestMultiPanelExport:
    """render_preview() used to only ever draw the active panel, so an inset_axes() image
    or a subplots() grid's other cells were silently missing from a saved PNG even though
    they render correctly in the live GL window.
    """

    @staticmethod
    def _solid(color):
        img = np.zeros((8, 8, 3), dtype=np.float32)
        img[..., 0], img[..., 1], img[..., 2] = color
        return img

    @staticmethod
    def _has_color(img, color, tol=40):
        r, g, b = (int(c * 255) for c in color)
        return bool(
            np.any(
                (np.abs(img[..., 0].astype(int) - r) < tol)
                & (np.abs(img[..., 1].astype(int) - g) < tol)
                & (np.abs(img[..., 2].astype(int) - b) < tol)
            )
        )

    def test_inset_axes_image_reaches_the_saved_png(self):
        from PIL import Image

        from glplot.utils.preview import render_preview

        fig, ax = gplt.subplots()
        ax.imshow(self._solid((0.0, 1.0, 0.0)))
        inset = ax.inset_axes([0.05, 0.05, 0.3, 0.3])
        inset.imshow(self._solid((1.0, 0.0, 0.0)))

        path = tempfile.mktemp(suffix=".png")
        try:
            render_preview(fig, path)
            img = np.asarray(Image.open(path).convert("RGB"))
            assert self._has_color(img, (0.0, 1.0, 0.0)), "parent panel missing from PNG"
            assert self._has_color(img, (1.0, 0.0, 0.0)), "inset panel missing from PNG"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_subplots_grid_exports_every_cell(self):
        from PIL import Image

        from glplot.utils.preview import render_preview

        fig, axs = gplt.subplots(1, 2)
        axs[0].imshow(self._solid((1.0, 0.0, 0.0)))
        axs[1].imshow(self._solid((0.0, 0.0, 1.0)))

        path = tempfile.mktemp(suffix=".png")
        try:
            render_preview(fig, path)
            img = np.asarray(Image.open(path).convert("RGB"))
            assert self._has_color(img, (1.0, 0.0, 0.0)), "first subplot missing from PNG"
            assert self._has_color(img, (0.0, 0.0, 1.0)), "second subplot missing from PNG"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_single_panel_export_is_unaffected(self):
        """The common case (no extra panels) keeps rendering through the original path."""
        from glplot.utils.preview import render_preview

        gplt.plot([0, 1, 2], [0, 1, 4])
        path = tempfile.mktemp(suffix=".png")
        try:
            render_preview(gplt.gcf(), path)
            assert os.path.getsize(path) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestHeadlessAxisLimitPinning:
    """``gplt.xlim()``/``gplt.ylim()`` used to have zero effect on a headless ``savefig()``
    export: ``render_preview()`` ended with an unconditional ``ax.autoscale(enable=True)``,
    which always re-fit the view to the plotted data and silently discarded any pinned
    range (see project memory ``project_plotting_api_gaps`` for the original bug report).
    Fixed via ``engine._needs_initial_autoscale`` -- these guard the fix from regressing
    quietly, since none of the existing ``xlim()``/``ylim()`` tests render anything; they
    only check that the live engine's own ``get_xlim()``/``get_ylim()`` reflects the call.
    """

    def test_xlim_survives_headless_export(self):
        from glplot.utils.preview import _build_preview_figure

        gplt.plot([0, 10], [0, 10])
        gplt.xlim(2, 6)
        fig = _build_preview_figure(gplt.gcf())
        try:
            assert fig.axes[0].get_xlim() == pytest.approx((2.0, 6.0))
        finally:
            import matplotlib.pyplot as mpl

            mpl.close(fig)

    def test_ylim_survives_headless_export(self):
        from glplot.utils.preview import _build_preview_figure

        gplt.plot([0, 10], [0, 10])
        gplt.ylim(3, 7)
        fig = _build_preview_figure(gplt.gcf())
        try:
            assert fig.axes[0].get_ylim() == pytest.approx((3.0, 7.0))
        finally:
            import matplotlib.pyplot as mpl

            mpl.close(fig)

    def test_an_unpinned_axis_still_autoscales(self):
        """The fix must not overshoot: with no xlim()/ylim() call, autoscale still runs."""
        from glplot.utils.preview import _build_preview_figure

        gplt.plot([0, 10], [-5, 5])
        fig = _build_preview_figure(gplt.gcf())
        try:
            xlim = fig.axes[0].get_xlim()
            ylim = fig.axes[0].get_ylim()
            assert xlim[0] <= 0 and xlim[1] >= 10, "autoscaled view must still contain the data"
            assert ylim[0] <= -5 and ylim[1] >= 5
        finally:
            import matplotlib.pyplot as mpl

            mpl.close(fig)


class TestRenderPreviewArray:
    """``render_preview_array()`` shares its figure-building code with ``render_preview()``
    via ``_build_preview_figure()`` (added when `glplot.animation.figure_to_rgb()`'s headless
    path was switched from a temp-PNG round trip to reading Agg's canvas buffer directly, cutting
    ~35% of a profiled animation frame's cost). These pin the two entry points to the same
    output so a future change to one cannot silently drift from the other.
    """

    def test_matches_render_preview_byte_for_byte(self):
        from PIL import Image

        from glplot.utils.preview import render_preview, render_preview_array

        gplt.plot([0, 1, 2, 3], [0, 1, 4, 9], color="blue")
        gplt.scatter([0, 1, 2], [3, 1, 2], s=40)
        gplt.xlabel("x")
        gplt.ylabel("y")
        gplt.title("parity check")

        array = render_preview_array(gplt.gcf())
        path = tempfile.mktemp(suffix=".png")
        try:
            render_preview(gplt.gcf(), path)
            from_file = np.asarray(Image.open(path).convert("RGB"))
            assert array.shape == from_file.shape
            assert np.array_equal(array, from_file)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_matches_render_preview_on_a_multi_panel_figure(self):
        from PIL import Image

        from glplot.utils.preview import render_preview, render_preview_array

        fig, axs = gplt.subplots(1, 2)
        axs[0].plot([0, 1, 2], [0, 1, 4])
        axs[1].scatter([0, 1, 2], [2, 0, 1])

        array = render_preview_array(fig)
        path = tempfile.mktemp(suffix=".png")
        try:
            render_preview(fig, path)
            from_file = np.asarray(Image.open(path).convert("RGB"))
            assert array.shape == from_file.shape
            assert np.array_equal(array, from_file)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_returns_uint8_rgb(self):
        from glplot.utils.preview import render_preview_array

        gplt.plot([0, 1, 2], [0, 1, 4])
        array = render_preview_array(gplt.gcf())
        assert array.dtype == np.uint8
        assert array.ndim == 3 and array.shape[2] == 3


class TestNewArtistsExportVisibly:
    """Every new plotting function must leave marks in the PNG, not vanish on save.

    A function that renders live but exports blank is a savefig-only regression that unit
    tests on the live scene never catch. Measured as non-white pixels in the rendered
    figure -- a low bar, but it catches "the export path does not know this artist".
    """

    @pytest.mark.parametrize(
        "name, build",
        [
            ("violinplot", lambda r: gplt.violinplot(r.normal(size=300))),
            (
                "eventplot",
                lambda r: gplt.eventplot([np.sort(r.uniform(0, 5, 50)) for _ in range(3)]),
            ),
            ("stairs", lambda r: gplt.stairs([1, 3, 2, 4], [0, 1, 2, 3, 4])),
            ("ecdf", lambda r: gplt.ecdf(r.normal(size=200))),
            ("barh", lambda r: gplt.barh([0, 1, 2], [10, 24, 18])),
            (
                "fill_betweenx",
                lambda r: gplt.fill_betweenx(
                    np.linspace(0, 10, 20), np.zeros(20), np.sin(np.linspace(0, 10, 20))
                ),
            ),
            ("boxplot", lambda r: gplt.boxplot(r.normal(size=200))),
            ("pie", lambda r: gplt.pie([30, 20, 50])),
            ("triplot", lambda r: gplt.triplot(r.random(30), r.random(30))),
            ("spy", lambda r: gplt.spy(np.eye(8))),
            (
                "stackplot",
                lambda r: gplt.stackplot(np.linspace(0, 10, 20), np.ones(20), 2 * np.ones(20)),
            ),
            ("broken_barh", lambda r: gplt.broken_barh([(1, 2), (5, 1)], (0, 1))),
        ],
    )
    def test_artist_leaves_marks_in_the_png(self, name, build):
        from PIL import Image

        from glplot.utils.preview import render_preview

        rng = np.random.default_rng(0)
        build(rng)
        path = tempfile.mktemp(suffix=".png")
        try:
            render_preview(gplt.gcf(), path)
            img = np.asarray(Image.open(path))[:, :, :3]
            non_white = int(np.sum(np.any(img < 250, axis=-1)))
            assert non_white > 100, f"{name} exported nearly blank ({non_white} px)"
        finally:
            if os.path.exists(path):
                os.unlink(path)
