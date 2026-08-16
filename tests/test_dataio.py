"""Test the file importers: numpy archives, JSON, delimited text and images.

Every fixture is written by the test itself into ``tmp_path`` -- including the images,
which are synthesised as small arrays and saved with ``matplotlib.image.imsave``. Nothing
here ships a binary fixture: a checked-in PNG would be untestable in review and would rot
silently against a Pillow upgrade, whereas a four-pixel array whose values are written out
in the test is readable and exact.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from glplot.gui import dataio
from glplot.gui import layerops3d as l3


def _write_image(path, array):
    """Save ``array`` (float 0..1 or uint8) as an image file and return the path."""
    from matplotlib import image as mpimg

    mpimg.imsave(str(path), np.asarray(array))
    return str(path)


class TestExtensionHelpers:
    """The little dispatch helpers a panel uses to build a file filter."""

    def test_stem_ignores_directory_and_extension(self):
        assert dataio.file_stem("/a/b/run 3.csv") == "run 3"

    def test_stem_of_a_nameless_path_falls_back(self):
        assert dataio.file_stem("/a/b/") == "data"

    def test_extension_is_lower_cased(self):
        assert dataio.extension_of("/a/B.CSV") == ".csv"

    def test_extension_of_a_bare_name_is_empty(self):
        assert dataio.extension_of("README") == ""

    @pytest.mark.parametrize("name", ["a.png", "a.JPG", "a.tiff"])
    def test_images_are_recognised(self, name):
        assert dataio.is_image(name)

    def test_a_table_is_not_an_image(self):
        assert not dataio.is_image("a.csv")

    def test_supported_covers_every_family(self):
        for extension in (".csv", ".npy", ".npz", ".json", ".png"):
            assert extension in dataio.SUPPORTED_EXTENSIONS
            assert dataio.is_supported("f" + extension)

    def test_an_unknown_extension_is_not_advertised_as_supported(self):
        assert not dataio.is_supported("f.qqq")

    def test_the_error_is_a_value_error(self):
        """Panels already catch ValueError around the text loader; this must land there."""
        assert issubclass(dataio.DataIOError, ValueError)


class TestText:
    """Delimited text, which delegates to clipboard.parse_table."""

    def test_csv_round_trips_with_its_header(self, tmp_path):
        path = tmp_path / "run.csv"
        path.write_text("t,y\n0,1\n1,4\n2,9\n")
        ds = dataio.load_dataset(str(path))
        assert ds.name == "run"
        assert ds.column_names() == ["t", "y"]
        assert np.allclose(ds.get("y"), [1.0, 4.0, 9.0])
        assert ds.source == "file"

    def test_tab_separated_and_crlf(self, tmp_path):
        path = tmp_path / "excel.tsv"
        path.write_bytes(b"a\tb\r\n1\t2\r\n3\t4\r\n")
        ds = dataio.load_dataset(str(path))
        assert ds.column_names() == ["a", "b"]
        assert np.allclose(ds.to_array(), [[1.0, 2.0], [3.0, 4.0]])

    def test_a_non_numeric_cell_becomes_nan(self, tmp_path):
        path = tmp_path / "holes.csv"
        path.write_text("x,y\n1,\n2,bad\n")
        ds = dataio.load_dataset(str(path))
        assert np.all(np.isnan(ds.get("y")))

    def test_a_commented_preamble_is_dropped(self, tmp_path):
        path = tmp_path / "instrument.dat"
        path.write_text("# recorded 2026-01-01\n# operator: nobody\n1 2\n3 4\n")
        ds = dataio.load_dataset(str(path))
        assert ds.n_rows() == 2 and ds.n_cols() == 2
        assert np.allclose(ds.to_array(), [[1.0, 2.0], [3.0, 4.0]])

    def test_the_last_comment_line_names_the_columns(self, tmp_path):
        """The numpy/gnuplot convention: '# t  y' above the data is the header."""
        path = tmp_path / "gnuplot.dat"
        path.write_text("# a run\n# t y\n0 1\n1 2\n")
        ds = dataio.load_dataset(str(path))
        assert ds.column_names() == ["t", "y"]

    def test_a_prose_comment_does_not_become_a_header(self, tmp_path):
        path = tmp_path / "prose.dat"
        path.write_text("# recorded on a wet Tuesday\n0 1\n1 2\n")
        ds = dataio.load_dataset(str(path))
        assert ds.column_names() == ["col1", "col2"]

    def test_a_real_header_row_beats_the_comment_hint(self, tmp_path):
        path = tmp_path / "both.csv"
        path.write_text("# x,y\ntime,value\n0,1\n")
        ds = dataio.load_dataset(str(path))
        assert ds.column_names() == ["time", "value"]

    def test_matlab_percent_comments_are_understood(self, tmp_path):
        path = tmp_path / "matlab.txt"
        path.write_text("% t y\n0 1\n2 3\n")
        ds = dataio.load_dataset(str(path))
        assert ds.column_names() == ["t", "y"]
        assert ds.n_rows() == 2

    def test_a_comment_after_the_data_is_still_dropped(self, tmp_path):
        path = tmp_path / "trailer.dat"
        path.write_text("1 2\n3 4\n# end of file\n")
        assert dataio.load_dataset(str(path)).n_rows() == 2

    def test_strip_comments_reports_the_hint_separately(self):
        body, hint = dataio.strip_comments("# one\n# t y\n1 2\n")
        assert body.strip() == "1 2"
        assert hint == "t y"

    def test_strip_comments_leaves_a_clean_file_alone(self):
        body, hint = dataio.strip_comments("1 2\n3 4\n")
        assert body == "1 2\n3 4"
        assert hint is None

    def test_an_unknown_extension_is_read_as_text(self, tmp_path):
        path = tmp_path / "measurements.out"
        path.write_text("1 2 3\n4 5 6\n")
        ds = dataio.load_dataset(str(path))
        assert ds.n_rows() == 2 and ds.n_cols() == 3

    def test_a_missing_file_names_itself(self, tmp_path):
        with pytest.raises(dataio.DataIOError, match="nope.csv"):
            dataio.load_dataset(str(tmp_path / "nope.csv"))

    def test_an_empty_file_names_itself(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("")
        with pytest.raises(dataio.DataIOError, match="empty.csv"):
            dataio.load_dataset(str(path))

    def test_a_file_of_nothing_but_comments_says_so(self, tmp_path):
        path = tmp_path / "notes.dat"
        path.write_text("# nothing here\n# really\n")
        with pytest.raises(dataio.DataIOError, match="only comments"):
            dataio.load_dataset(str(path))

    def test_an_explicit_name_overrides_the_file_name(self, tmp_path):
        path = tmp_path / "run.csv"
        path.write_text("t\n1\n")
        assert dataio.load_dataset(str(path), name="chosen").name == "chosen"


class TestNpy:
    """Single-array ``.npy`` files."""

    def test_a_2d_array_becomes_one_column_per_second_axis_entry(self, tmp_path):
        path = tmp_path / "grid.npy"
        data = np.arange(12, dtype=np.float64).reshape(4, 3)
        np.save(path, data)
        ds = dataio.load_dataset(str(path))
        assert ds.column_names() == ["col1", "col2", "col3"]
        assert ds.n_rows() == 4
        assert np.allclose(ds.to_array(), data)

    def test_a_1d_array_becomes_one_column_named_after_the_file(self, tmp_path):
        path = tmp_path / "signal.npy"
        np.save(path, np.linspace(0.0, 1.0, 5))
        ds = dataio.load_dataset(str(path))
        assert ds.column_names() == ["signal"]
        assert np.allclose(ds.get("signal"), np.linspace(0.0, 1.0, 5))

    def test_an_n_by_1_array_is_the_same_table_as_1d(self, tmp_path):
        path = tmp_path / "signal.npy"
        np.save(path, np.arange(5.0).reshape(5, 1))
        ds = dataio.load_dataset(str(path))
        assert ds.column_names() == ["signal"]

    def test_integers_widen_to_float(self, tmp_path):
        path = tmp_path / "counts.npy"
        np.save(path, np.array([1, 2, 3], dtype=np.int32))
        assert dataio.load_dataset(str(path)).get("counts").dtype == np.float64

    def test_a_3d_array_is_refused_with_its_shape(self, tmp_path):
        path = tmp_path / "cube.npy"
        np.save(path, np.zeros((2, 3, 4)))
        with pytest.raises(dataio.DataIOError, match=r"3-D array of shape \(2, 3, 4\)"):
            dataio.load_dataset(str(path))

    def test_a_scalar_is_refused(self, tmp_path):
        path = tmp_path / "scalar.npy"
        np.save(path, np.float64(3.0))
        with pytest.raises(dataio.DataIOError, match="0-D"):
            dataio.load_dataset(str(path))

    def test_complex_data_is_refused_rather_than_silently_halved(self, tmp_path):
        path = tmp_path / "spectrum.npy"
        np.save(path, np.array([1 + 2j, 3 + 4j]))
        with pytest.raises(dataio.DataIOError, match="complex"):
            dataio.load_dataset(str(path))

    def test_a_pickled_array_is_refused_with_advice(self, tmp_path):
        path = tmp_path / "objects.npy"
        np.save(path, np.array([{"a": 1}, {"b": 2}], dtype=object), allow_pickle=True)
        with pytest.raises(dataio.DataIOError, match="pickled"):
            dataio.load_dataset(str(path))


class TestNpz:
    """Multi-array ``.npz`` archives."""

    def test_each_member_becomes_a_column_under_its_own_name(self, tmp_path):
        path = tmp_path / "run.npz"
        np.savez(path, t=np.arange(4.0), y=np.arange(4.0) ** 2)
        ds = dataio.load_dataset(str(path))
        assert sorted(ds.column_names()) == ["t", "y"]
        assert np.allclose(ds.get("y"), [0.0, 1.0, 4.0, 9.0])

    def test_a_2d_member_expands_into_numbered_columns(self, tmp_path):
        path = tmp_path / "run.npz"
        np.savez(path, t=np.arange(3.0), xyz=np.zeros((3, 3)))
        ds = dataio.load_dataset(str(path))
        assert set(ds.column_names()) == {"t", "xyz1", "xyz2", "xyz3"}

    def test_a_single_column_member_keeps_its_bare_name(self, tmp_path):
        path = tmp_path / "run.npz"
        np.savez(path, y=np.zeros((3, 1)))
        assert dataio.load_dataset(str(path)).column_names() == ["y"]

    def test_a_scalar_member_becomes_a_constant_column(self, tmp_path):
        path = tmp_path / "run.npz"
        np.savez(path, t=np.arange(3.0), dt=np.float64(0.25))
        ds = dataio.load_dataset(str(path))
        assert np.allclose(ds.get("dt"), [0.25, 0.25, 0.25])

    def test_an_all_scalar_archive_is_a_single_row(self, tmp_path):
        path = tmp_path / "params.npz"
        np.savez(path, a=np.float64(1.0), b=np.float64(2.0))
        ds = dataio.load_dataset(str(path))
        assert ds.n_rows() == 1
        assert sorted(ds.column_names()) == ["a", "b"]

    def test_mismatched_lengths_name_the_members(self, tmp_path):
        path = tmp_path / "ragged.npz"
        np.savez(path, t=np.arange(4.0), y=np.arange(3.0))
        with pytest.raises(dataio.DataIOError, match=r"t \(4\)"):
            dataio.load_dataset(str(path))

    def test_a_3d_member_names_itself(self, tmp_path):
        path = tmp_path / "vol.npz"
        np.savez(path, v=np.zeros((2, 2, 2)))
        with pytest.raises(dataio.DataIOError, match="member 'v'"):
            dataio.load_dataset(str(path))


class TestJson:
    """The four JSON shapes."""

    def test_a_list_of_records(self, tmp_path):
        path = tmp_path / "records.json"
        path.write_text(json.dumps([{"t": 0, "y": 1.5}, {"t": 1, "y": 2.5}]))
        ds = dataio.load_dataset(str(path))
        assert ds.column_names() == ["t", "y"]
        assert np.allclose(ds.get("y"), [1.5, 2.5])

    def test_a_key_missing_from_a_record_becomes_nan(self, tmp_path):
        path = tmp_path / "sparse.json"
        path.write_text(json.dumps([{"t": 0}, {"t": 1, "y": 5}]))
        ds = dataio.load_dataset(str(path))
        assert ds.column_names() == ["t", "y"]
        assert np.isnan(ds.get("y")[0]) and ds.get("y")[1] == 5.0

    def test_a_key_appearing_only_later_is_still_a_column(self, tmp_path):
        """The schema is the union of the records, not whatever row 0 happened to have."""
        path = tmp_path / "late.json"
        path.write_text(json.dumps([{"a": 1}, {"a": 2, "b": 9}]))
        ds = dataio.load_dataset(str(path))
        assert np.isnan(ds.get("b")[0]) and ds.get("b")[1] == 9.0

    def test_text_and_null_values_become_nan(self, tmp_path):
        path = tmp_path / "mixed.json"
        path.write_text(json.dumps([{"y": "n/a"}, {"y": None}, {"y": "2.5"}, {"y": True}]))
        values = dataio.load_dataset(str(path)).get("y")
        assert np.isnan(values[0]) and np.isnan(values[1])
        assert values[2] == 2.5 and values[3] == 1.0

    def test_an_object_of_arrays(self, tmp_path):
        path = tmp_path / "arrays.json"
        path.write_text(json.dumps({"t": [0, 1, 2], "y": [3, 4, 5]}))
        ds = dataio.load_dataset(str(path))
        assert ds.column_names() == ["t", "y"]
        assert np.allclose(ds.get("t"), [0.0, 1.0, 2.0])

    def test_a_scalar_beside_arrays_is_broadcast(self, tmp_path):
        path = tmp_path / "meta.json"
        path.write_text(json.dumps({"t": [0, 1], "dt": 0.5}))
        assert np.allclose(dataio.load_dataset(str(path)).get("dt"), [0.5, 0.5])

    def test_unequal_arrays_are_refused(self, tmp_path):
        path = tmp_path / "ragged.json"
        path.write_text(json.dumps({"t": [0, 1, 2], "y": [3]}))
        with pytest.raises(dataio.DataIOError, match="different lengths"):
            dataio.load_dataset(str(path))

    def test_a_list_of_rows(self, tmp_path):
        path = tmp_path / "rows.json"
        path.write_text(json.dumps([[1, 2], [3, 4], [5, 6]]))
        ds = dataio.load_dataset(str(path))
        assert ds.column_names() == ["col1", "col2"]
        assert np.allclose(ds.to_array(), [[1, 2], [3, 4], [5, 6]])

    def test_ragged_rows_pad_with_nan(self, tmp_path):
        path = tmp_path / "ragged_rows.json"
        path.write_text(json.dumps([[1, 2, 3], [4]]))
        ds = dataio.load_dataset(str(path))
        assert ds.n_cols() == 3
        assert np.isnan(ds.get("col2")[1])

    def test_a_flat_list_becomes_one_column(self, tmp_path):
        path = tmp_path / "values.json"
        path.write_text(json.dumps([1, 2, 3]))
        ds = dataio.load_dataset(str(path))
        assert ds.column_names() == ["values"]
        assert np.allclose(ds.get("values"), [1.0, 2.0, 3.0])

    def test_a_nested_object_is_refused_by_key(self, tmp_path):
        path = tmp_path / "tree.json"
        path.write_text(json.dumps({"a": {"b": [1, 2]}}))
        with pytest.raises(dataio.DataIOError, match="nested object"):
            dataio.load_dataset(str(path))

    def test_a_bare_scalar_top_level_is_refused(self, tmp_path):
        path = tmp_path / "scalar.json"
        path.write_text("42")
        with pytest.raises(dataio.DataIOError, match="top level"):
            dataio.load_dataset(str(path))

    def test_invalid_json_names_the_file(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json")
        with pytest.raises(dataio.DataIOError, match="broken.json"):
            dataio.load_dataset(str(path))

    def test_an_empty_list_is_refused(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("[]")
        with pytest.raises(dataio.DataIOError, match="empty"):
            dataio.load_dataset(str(path))


class TestImageMatrix:
    """The 2-D luminance form."""

    def test_a_grayscale_png_round_trips_its_values(self, tmp_path):
        """imsave writes gray input through a colormap, so check a colour source instead."""
        source = np.zeros((2, 3, 3), dtype=np.float64)
        source[:, :, 0] = 1.0  # pure red everywhere
        image = dataio.load_image(_write_image(tmp_path / "red.png", source))
        assert image.rgb.shape == (2, 3, 3)
        assert np.allclose(image.rgb[:, :, 0], 1.0)
        assert np.allclose(image.rgb[:, :, 1], 0.0)
        assert np.allclose(image.luminance, dataio.LUMA_WEIGHTS[0], atol=1e-6)

    def test_luminance_uses_rec_709_weights(self, tmp_path):
        source = np.zeros((1, 3, 3), dtype=np.float64)
        source[0, 0] = (1.0, 0.0, 0.0)
        source[0, 1] = (0.0, 1.0, 0.0)
        source[0, 2] = (0.0, 0.0, 1.0)
        image = dataio.load_image(_write_image(tmp_path / "rgb.png", source))
        assert np.allclose(image.luminance[0], dataio.LUMA_WEIGHTS, atol=1e-6)

    def test_the_matrix_keeps_image_row_order(self, tmp_path):
        """Row 0 of ``luminance`` is the top row of the picture, as imshow expects."""
        source = np.zeros((2, 2, 3), dtype=np.float64)
        source[0] = 1.0  # white top row, black bottom row
        image = dataio.load_image(_write_image(tmp_path / "topbottom.png", source))
        assert image.luminance[0, 0] > 0.9
        assert image.luminance[1, 0] < 0.1

    def test_shape_helpers_agree_with_the_matrix(self, tmp_path):
        source = np.zeros((4, 7, 3), dtype=np.float64)
        image = dataio.load_image(_write_image(tmp_path / "s.png", source))
        assert (image.height, image.width) == (4, 7)
        assert image.source_shape == (4, 7)
        assert image.step == 1 and not image.downsampled

    def test_an_alpha_channel_is_kept_and_not_applied(self, tmp_path):
        source = np.zeros((1, 2, 4), dtype=np.float64)
        source[:, :, 0] = 1.0  # red
        source[0, 0, 3] = 1.0  # opaque
        source[0, 1, 3] = 0.0  # transparent, but still red
        image = dataio.load_image(_write_image(tmp_path / "alpha.png", source))
        assert image.alpha is not None
        assert np.allclose(image.alpha[0], [1.0, 0.0], atol=1e-6)
        assert np.allclose(image.rgb[0, :, 0], [1.0, 1.0], atol=1e-6)

    def test_a_file_without_alpha_reports_none(self, tmp_path):
        """A JPEG has no alpha channel at all — PNG always comes back RGBA from imsave."""
        source = np.zeros((2, 2, 3), dtype=np.float64)
        assert dataio.load_image(_write_image(tmp_path / "opaque.jpg", source)).alpha is None

    def test_a_uint8_jpeg_is_scaled_to_zero_one(self, tmp_path):
        source = np.zeros((2, 2, 3), dtype=np.float64)
        source[:, :, 1] = 1.0
        image = dataio.load_image(_write_image(tmp_path / "green.jpg", source))
        assert np.allclose(image.rgb[:, :, 1], 1.0, atol=0.02)
        assert float(image.rgb.max()) <= 1.0

    def test_a_missing_image_names_itself(self, tmp_path):
        with pytest.raises(dataio.DataIOError, match="ghost.png"):
            dataio.load_image(str(tmp_path / "ghost.png"))

    def test_a_file_that_is_not_an_image_names_itself(self, tmp_path):
        path = tmp_path / "fake.png"
        path.write_text("this is not a png")
        with pytest.raises(dataio.DataIOError, match="fake.png"):
            dataio.load_dataset(str(path))


class TestImageDownsampling:
    """The pixel cap, and the promise that it is never silent."""

    def test_a_small_image_is_untouched(self, tmp_path):
        source = np.zeros((8, 8, 3), dtype=np.float64)
        image = dataio.load_image(_write_image(tmp_path / "small.png", source), max_pixels=64)
        assert image.step == 1 and image.luminance.shape == (8, 8)

    def test_a_large_image_is_block_averaged(self, tmp_path):
        source = np.zeros((8, 8, 3), dtype=np.float64)
        source[:, :, 0] = np.tile([1.0, 0.0], (8, 4))  # vertical stripes, mean 0.5
        image = dataio.load_image(_write_image(tmp_path / "big.png", source), max_pixels=16)
        assert image.step == 2
        assert image.luminance.shape == (4, 4)
        # The mean of a red/black 2x2 block, not one of its two pixel values.
        assert np.allclose(image.luminance, 0.5 * dataio.LUMA_WEIGHTS[0], atol=1e-6)

    def test_the_name_states_the_reduction(self, tmp_path):
        source = np.zeros((8, 8, 3), dtype=np.float64)
        image = dataio.load_image(_write_image(tmp_path / "big.png", source), max_pixels=16)
        assert "downsampled 2x from 8x8" in image.name
        assert image.downsampled and image.source_shape == (8, 8)

    def test_the_name_states_the_size_when_it_was_not_reduced(self, tmp_path):
        source = np.zeros((3, 5, 3), dtype=np.float64)
        image = dataio.load_image(_write_image(tmp_path / "photo.png", source))
        assert image.name == "photo 5x3"

    def test_a_ragged_remainder_is_cropped_not_padded(self, tmp_path):
        source = np.zeros((9, 9, 3), dtype=np.float64)
        image = dataio.load_image(_write_image(tmp_path / "odd.png", source), max_pixels=16)
        assert image.step == 3
        assert image.luminance.shape == (3, 3)

    def test_the_default_cap_is_a_megapixel(self):
        assert dataio.MAX_IMAGE_PIXELS == 1_000_000


class TestImageTable:
    """The long ``(x, y, lum, r, g, b)`` form."""

    @pytest.mark.parametrize("suffix", [".png", ".jpg"])
    def test_the_first_six_columns_never_depend_on_the_channel_count(self, tmp_path, suffix):
        source = np.zeros((2, 3, 3), dtype=np.float64)
        ds = dataio.load_dataset(_write_image(tmp_path / ("flat" + suffix), source))
        assert ds.column_names()[:6] == ["x", "y", "lum", "r", "g", "b"]
        assert ds.n_rows() == 6
        assert ds.source == "image"

    def test_alpha_adds_a_column_only_when_present(self, tmp_path):
        source = np.zeros((2, 2, 3), dtype=np.float64)
        with_alpha = dataio.load_dataset(_write_image(tmp_path / "rgba.png", source))
        without = dataio.load_dataset(_write_image(tmp_path / "rgb.jpg", source))
        assert with_alpha.column_names()[-1] == "a"
        assert "a" not in without.column_names()

    def test_y_is_flipped_so_the_picture_stands_up(self, tmp_path):
        """Row 0 (the top of the image) must land at the largest y, not the smallest."""
        source = np.zeros((2, 1, 3), dtype=np.float64)
        source[0] = 1.0  # bright top row
        ds = dataio.load_dataset(_write_image(tmp_path / "flip.png", source))
        y, lum = ds.get("y"), ds.get("lum")
        assert lum[np.argmax(y)] > lum[np.argmin(y)]

    def test_coordinates_are_in_original_pixel_units(self, tmp_path):
        """A downsampled import keeps the extent of the file it came from."""
        source = np.zeros((8, 8, 3), dtype=np.float64)
        path = _write_image(tmp_path / "big.png", source)
        full = dataio.load_image(path).to_dataset()
        small = dataio.load_image(path, max_pixels=16).to_dataset()
        assert float(np.max(full.get("x"))) == pytest.approx(7.0)
        assert float(np.max(small.get("x"))) == pytest.approx(6.0)  # 3 blocks of 2 px

    def test_the_table_is_a_full_lattice_for_surface_plotting(self, tmp_path):
        source = np.zeros((3, 4, 3), dtype=np.float64)
        ds = dataio.load_dataset(_write_image(tmp_path / "grid.png", source))
        assert l3.grid_shape(ds.get("x"), ds.get("y")) == (4, 3)

    def test_it_builds_a_height_field_surface_end_to_end(self, tmp_path):
        """The point of the long form: lum as z, straight into a surface3d layer."""
        from glplot.engine import GPULinePlot

        rng = np.random.default_rng(0)
        source = rng.random((6, 5, 3))
        ds = dataio.load_dataset(_write_image(tmp_path / "field.png", source))
        plot = GPULinePlot()
        plot.set_ndim(3)
        layer = l3.add_xyz_layer(
            plot,
            ds.get("x"),
            ds.get("y"),
            ds.get("lum"),
            kind="surface3d",
            label="image",
        )
        assert len(layer.vertices) == 30
        assert np.all(np.isfinite(layer.vertices))

    def test_an_explicit_name_wins_over_the_generated_one(self, tmp_path):
        source = np.zeros((2, 2, 3), dtype=np.float64)
        image = dataio.load_image(_write_image(tmp_path / "x.png", source))
        assert image.to_dataset("chosen").name == "chosen"

    def test_a_uint8_source_is_scaled_to_zero_one(self, tmp_path):
        source = np.zeros((2, 2, 3), dtype=np.uint8)
        source[:, :, 1] = 255
        ds = dataio.load_dataset(_write_image(tmp_path / "u8.png", source))
        assert np.allclose(ds.get("g"), 1.0)
        assert np.allclose(ds.get("r"), 0.0)


class TestDispatch:
    """load_dataset is the one door in."""

    def test_every_format_comes_back_as_a_dataset(self, tmp_path):
        from glplot.gui.datasets import DataSet

        (tmp_path / "a.csv").write_text("x\n1\n")
        np.save(tmp_path / "b.npy", np.arange(3.0))
        np.savez(tmp_path / "c.npz", t=np.arange(3.0))
        (tmp_path / "d.json").write_text(json.dumps({"t": [1, 2]}))
        _write_image(tmp_path / "e.png", np.zeros((2, 2, 3)))
        for name in ("a.csv", "b.npy", "c.npz", "d.json", "e.png"):
            ds = dataio.load_dataset(str(tmp_path / name))
            assert isinstance(ds, DataSet)
            assert ds.n_rows() > 0 and ds.n_cols() > 0

    def test_a_dataset_is_not_registered_anywhere(self, tmp_path):
        """Importing must not touch a store; the caller adds it inside a command."""
        from glplot.gui.datasets import DataStore

        (tmp_path / "a.csv").write_text("x\n1\n")
        store = DataStore()
        dataio.load_dataset(str(tmp_path / "a.csv"))
        assert len(store) == 0

    def test_the_module_imports_no_imgui_and_no_engine(self):
        """CONTRACT 5.1 rule 7: this is pure logic, like datasets/clipboard/generators3d."""
        import ast
        import pathlib

        source = pathlib.Path(dataio.__file__).read_text()
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(name.startswith("imgui") for name in imported)
        assert not any("engine" in name for name in imported)

    def test_matplotlib_is_not_imported_at_module_level(self):
        """A CSV import must not pay for matplotlib's import."""
        import ast
        import pathlib

        tree = ast.parse(pathlib.Path(dataio.__file__).read_text())
        for node in tree.body:
            assert not isinstance(node, (ast.Import, ast.ImportFrom)) or "matplotlib" not in (
                getattr(node, "module", "") or ""
            )
