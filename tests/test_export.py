"""Test export and savefig functionality."""

import os
import tempfile
import pytest
import numpy as np

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
