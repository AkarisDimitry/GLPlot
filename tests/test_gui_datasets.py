"""Test the GUI data model in glplot.gui.datasets.

Pure logic: no OpenGL context, no window and no imgui are created here. Layers are
built directly from glplot.core.layers, which allocates nothing on the GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.core.layers import (
    Layer3D,
    LineFamilyLayer,
    PatchLayer,
    PolylineLayer,
    ScatterLayer,
    TextLayer,
)
from glplot.gui.datasets import Column, DataSet, DataStore
from glplot.gui.expressions import ExpressionError


def _dataset(name: str = "ds") -> DataSet:
    """Build a small two-column dataset for reuse across tests."""
    return DataSet(
        name,
        [
            Column("x", np.array([0.0, 1.0, 2.0])),
            Column("y", np.array([10.0, 11.0, 12.0])),
        ],
    )


class TestColumn:
    """Test the Column value holder."""

    def test_values_coerced_to_float64(self):
        """Test that an int input is coerced to a 1-D float64 array."""
        column = Column("a", np.array([1, 2, 3], dtype=np.int32))
        assert column.values.dtype == np.float64
        assert column.values.ndim == 1

    def test_values_raveled(self):
        """Test that a 2-D input is flattened to 1-D."""
        column = Column("a", np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert column.values.shape == (4,)

    def test_stats_are_nan_safe(self):
        """Test that nan entries are ignored by the summary statistics."""
        column = Column("a", np.array([1.0, np.nan, 3.0]))
        assert np.allclose(column.min, 1.0)
        assert np.allclose(column.max, 3.0)
        assert np.allclose(column.mean, 2.0)

    def test_stats_all_nan_column(self):
        """Test that an all-nan column reports nan rather than raising."""
        column = Column("a", np.array([np.nan, np.nan]))
        assert np.isnan(column.min)
        assert np.isnan(column.std)

    def test_stats_empty_column(self):
        """Test that an empty column reports nan statistics."""
        column = Column("a", np.array([]))
        assert np.isnan(column.mean)

    def test_invalidate_refreshes_cached_stats(self):
        """Test that invalidate() forces the statistics to be recomputed."""
        column = Column("a", np.array([1.0, 2.0]))
        assert np.allclose(column.max, 2.0)
        column.values[1] = 9.0
        assert np.allclose(column.max, 2.0)
        column.invalidate()
        assert np.allclose(column.max, 9.0)


class TestDataSetShape:
    """Test the shape invariants of DataSet."""

    def test_empty_dataset_shape(self):
        """Test that a dataset with no columns is 0x0."""
        dataset = DataSet("empty")
        assert dataset.n_rows() == 0
        assert dataset.n_cols() == 0

    def test_n_rows_and_n_cols(self):
        """Test the row and column counts of a populated dataset."""
        dataset = _dataset()
        assert dataset.n_rows() == 3
        assert dataset.n_cols() == 2

    def test_unequal_column_lengths_rejected(self):
        """Test that constructing with ragged columns raises ValueError."""
        with pytest.raises(ValueError):
            DataSet("bad", [Column("x", np.zeros(3)), Column("y", np.zeros(2))])

    def test_column_names_in_order(self):
        """Test that column_names() preserves insertion order."""
        assert _dataset().column_names() == ["x", "y"]

    def test_index_of(self):
        """Test that index_of() finds columns and returns -1 when absent."""
        dataset = _dataset()
        assert dataset.index_of("y") == 1
        assert dataset.index_of("nope") == -1

    def test_repr_reports_shape(self):
        """Test that the repr mentions name, shape and source."""
        text = repr(_dataset("mine"))
        assert "mine" in text
        assert "3x2" in text


class TestDataSetColumns:
    """Test column add/remove/rename on DataSet."""

    def test_add_column_with_values(self):
        """Test that a column added with values is appended verbatim."""
        dataset = _dataset()
        column = dataset.add_column("z", np.array([1.0, 2.0, 3.0]))
        assert dataset.n_cols() == 3
        assert np.allclose(column.values, [1.0, 2.0, 3.0])

    def test_add_column_defaults_to_zeros(self):
        """Test that values=None fills the new column with zeros of n_rows."""
        dataset = _dataset()
        column = dataset.add_column("z")
        assert np.allclose(column.values, np.zeros(3))

    def test_add_column_to_empty_dataset_defines_n_rows(self):
        """Test that the first column of an empty dataset sets the row count."""
        dataset = DataSet("empty")
        dataset.add_column("x", np.arange(5.0))
        assert dataset.n_rows() == 5

    def test_add_column_wrong_length_rejected(self):
        """Test that adding a mismatched-length column raises ValueError."""
        dataset = _dataset()
        with pytest.raises(ValueError):
            dataset.add_column("z", np.zeros(2))

    def test_add_column_name_collision_suffixed(self):
        """Test that duplicate names are suffixed ' (2)', ' (3)', ..."""
        dataset = _dataset()
        assert dataset.add_column("x").name == "x (2)"
        assert dataset.add_column("x").name == "x (3)"
        assert dataset.column_names() == ["x", "y", "x (2)", "x (3)"]

    def test_all_columns_stay_equal_length_after_add(self):
        """Test that adding a column preserves the equal-length invariant."""
        dataset = _dataset()
        dataset.add_column("z")
        lengths = {len(column.values) for column in dataset.columns}
        assert lengths == {3}

    def test_remove_column(self):
        """Test that remove_column() drops the column and reports True."""
        dataset = _dataset()
        assert dataset.remove_column("x") is True
        assert dataset.column_names() == ["y"]

    def test_remove_missing_column_returns_false(self):
        """Test that removing an absent column is a no-op returning False."""
        dataset = _dataset()
        assert dataset.remove_column("nope") is False
        assert dataset.n_cols() == 2

    def test_rename_column(self):
        """Test that rename_column() renames in place and reports True."""
        dataset = _dataset()
        assert dataset.rename_column("x", "t") is True
        assert dataset.column_names() == ["t", "y"]

    def test_rename_missing_column_returns_false(self):
        """Test that renaming an absent column returns False."""
        assert _dataset().rename_column("nope", "t") is False

    def test_rename_to_same_name_is_noop(self):
        """Test that renaming a column to its own name does not add a suffix."""
        dataset = _dataset()
        assert dataset.rename_column("x", "x") is True
        assert dataset.column_names() == ["x", "y"]

    def test_rename_collision_suffixed(self):
        """Test that renaming onto an existing name is suffixed."""
        dataset = _dataset()
        dataset.rename_column("x", "y")
        assert dataset.column_names() == ["y (2)", "y"]

    def test_get_returns_live_array(self):
        """Test that get() exposes the live column array, not a copy."""
        dataset = _dataset()
        dataset.get("x")[0] = 99.0
        assert np.allclose(dataset.get_cell(0, 0), 99.0)

    def test_get_missing_column_returns_none(self):
        """Test that get() returns None for an unknown column."""
        assert _dataset().get("nope") is None


class TestDataSetRows:
    """Test row insertion and deletion on DataSet."""

    def test_insert_row(self):
        """Test that insert_row() adds a zero-filled row at the index."""
        dataset = _dataset()
        dataset.insert_row(1)
        assert dataset.n_rows() == 4
        assert np.allclose(dataset.get("x"), [0.0, 0.0, 1.0, 2.0])
        assert np.allclose(dataset.get("y"), [10.0, 0.0, 11.0, 12.0])

    def test_insert_row_clamped(self):
        """Test that an out-of-range insert index is clamped to [0, n_rows]."""
        dataset = _dataset()
        dataset.insert_row(99)
        assert np.allclose(dataset.get("x"), [0.0, 1.0, 2.0, 0.0])
        dataset.insert_row(-5)
        assert np.allclose(dataset.get("x"), [0.0, 0.0, 1.0, 2.0, 0.0])

    def test_insert_row_on_empty_dataset(self):
        """Test that inserting into a dataset with no columns is a no-op."""
        dataset = DataSet("empty")
        dataset.insert_row(0)
        assert dataset.n_rows() == 0

    def test_delete_row(self):
        """Test that delete_row() removes the row from every column."""
        dataset = _dataset()
        dataset.delete_row(1)
        assert np.allclose(dataset.get("x"), [0.0, 2.0])
        assert np.allclose(dataset.get("y"), [10.0, 12.0])

    def test_delete_row_negative_index(self):
        """Test that a negative delete index counts from the end."""
        dataset = _dataset()
        dataset.delete_row(-1)
        assert np.allclose(dataset.get("x"), [0.0, 1.0])

    def test_delete_row_out_of_range_raises(self):
        """Test that an out-of-range delete index raises IndexError."""
        with pytest.raises(IndexError):
            _dataset().delete_row(3)

    def test_delete_rows_multiple(self):
        """Test that delete_rows() removes several rows at once."""
        dataset = _dataset()
        dataset.delete_rows([0, 2])
        assert np.allclose(dataset.get("x"), [1.0])

    def test_delete_rows_ignores_invalid_and_duplicates(self):
        """Test that duplicate and out-of-range indices are ignored."""
        dataset = _dataset()
        dataset.delete_rows([1, 1, 99, -99])
        assert np.allclose(dataset.get("x"), [0.0, 2.0])

    def test_delete_rows_negative_indices(self):
        """Test that delete_rows() normalises negative indices."""
        dataset = _dataset()
        dataset.delete_rows([-1])
        assert np.allclose(dataset.get("x"), [0.0, 1.0])

    def test_delete_rows_empty_selection_is_noop(self):
        """Test that delete_rows([]) leaves the table untouched."""
        dataset = _dataset()
        dataset.delete_rows([])
        assert dataset.n_rows() == 3

    def test_rows_stay_equal_length_after_edits(self):
        """Test that row edits keep every column the same length."""
        dataset = _dataset()
        dataset.add_column("z")
        dataset.insert_row(0)
        dataset.delete_rows([1, 2])
        lengths = {len(column.values) for column in dataset.columns}
        assert lengths == {2}

    def test_row_edits_invalidate_stats(self):
        """Test that stats are recomputed after a row deletion."""
        dataset = _dataset()
        assert np.allclose(dataset.columns[0].max, 2.0)
        dataset.delete_row(2)
        assert np.allclose(dataset.columns[0].max, 1.0)


class TestDataSetCells:
    """Test single-cell access on DataSet."""

    def test_set_and_get_cell(self):
        """Test the round trip of set_cell() and get_cell()."""
        dataset = _dataset()
        dataset.set_cell(1, 0, 7.5)
        assert np.allclose(dataset.get_cell(1, 0), 7.5)

    def test_set_cell_accepts_nan(self):
        """Test that nan is a legal cell value."""
        dataset = _dataset()
        dataset.set_cell(0, 1, float("nan"))
        assert np.isnan(dataset.get_cell(0, 1))

    def test_cells_accept_negative_indices(self):
        """Test that negative row and column indices wrap from the end."""
        dataset = _dataset()
        dataset.set_cell(-1, -1, 5.0)
        assert np.allclose(dataset.get_cell(2, 1), 5.0)

    def test_set_cell_out_of_range_raises(self):
        """Test that out-of-range row or column indices raise IndexError."""
        dataset = _dataset()
        with pytest.raises(IndexError):
            dataset.set_cell(9, 0, 1.0)
        with pytest.raises(IndexError):
            dataset.set_cell(0, 9, 1.0)

    def test_set_cell_invalidates_stats(self):
        """Test that set_cell() drops the cached statistics."""
        dataset = _dataset()
        assert np.allclose(dataset.columns[0].max, 2.0)
        dataset.set_cell(0, 0, 100.0)
        assert np.allclose(dataset.columns[0].max, 100.0)


class TestDataSetConversion:
    """Test to_xy, to_array and copy."""

    def test_to_xy(self):
        """Test that to_xy() returns the two named column arrays."""
        dataset = _dataset()
        x, y = dataset.to_xy("x", "y")
        assert np.allclose(x, [0.0, 1.0, 2.0])
        assert np.allclose(y, [10.0, 11.0, 12.0])

    def test_to_xy_returns_live_arrays(self):
        """Test that to_xy() hands out the live arrays, not copies."""
        dataset = _dataset()
        x, _ = dataset.to_xy("x", "y")
        assert x is dataset.get("x")

    def test_to_xy_same_column_twice(self):
        """Test that to_xy() may name the same column for both axes."""
        dataset = _dataset()
        x, y = dataset.to_xy("x", "x")
        assert np.allclose(x, y)

    def test_to_xy_missing_column_raises(self):
        """Test that to_xy() raises ValueError for an unknown column."""
        dataset = _dataset()
        with pytest.raises(ValueError):
            dataset.to_xy("x", "nope")
        with pytest.raises(ValueError):
            dataset.to_xy("nope", "y")

    def test_to_array(self):
        """Test that to_array() stacks the columns into (n_rows, n_cols)."""
        array = _dataset().to_array()
        assert array.shape == (3, 2)
        assert array.dtype == np.float64
        assert np.allclose(array[:, 1], [10.0, 11.0, 12.0])

    def test_to_array_empty_dataset(self):
        """Test that an empty dataset converts to a (0, 0) array."""
        assert DataSet("empty").to_array().shape == (0, 0)

    def test_to_array_is_a_copy(self):
        """Test that mutating to_array()'s result does not touch the dataset."""
        dataset = _dataset()
        array = dataset.to_array()
        array[0, 0] = 42.0
        assert np.allclose(dataset.get_cell(0, 0), 0.0)

    def test_copy_default_name(self):
        """Test that copy() names the clone '<name> copy' by default."""
        assert _dataset("base").copy().name == "base copy"

    def test_copy_explicit_name(self):
        """Test that copy(new_name) uses the given name."""
        assert _dataset("base").copy("other").name == "other"

    def test_copy_is_independent(self):
        """Test that mutating the copy never touches the original."""
        original = _dataset()
        clone = original.copy()
        clone.set_cell(0, 0, 99.0)
        clone.add_column("z")
        clone.delete_row(1)
        assert np.allclose(original.get("x"), [0.0, 1.0, 2.0])
        assert original.n_cols() == 2
        assert original.n_rows() == 3

    def test_copy_arrays_are_not_shared(self):
        """Test that the copy's column arrays are distinct objects."""
        original = _dataset()
        clone = original.copy()
        assert clone.get("x") is not original.get("x")

    def test_original_edits_do_not_reach_copy(self):
        """Test that the independence holds in the other direction too."""
        original = _dataset()
        clone = original.copy()
        original.set_cell(0, 0, 55.0)
        assert np.allclose(clone.get_cell(0, 0), 0.0)

    def test_copy_preserves_manual_source(self):
        """Test that a manual dataset's copy stays 'manual' and unbound."""
        clone = _dataset().copy()
        assert clone.source == "manual"
        assert clone.layer_id is None
        assert clone.binding is None


class TestFromLayer:
    """Test DataSet.from_layer against real engine layer objects."""

    def test_from_scatter_layer(self):
        """Test that a scatter layer maps pts -> columns x, y."""
        pts = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        layer = ScatterLayer(pts=pts, label="pts layer")
        dataset = DataSet.from_layer(layer)
        assert dataset is not None
        assert dataset.column_names() == ["x", "y"]
        assert np.allclose(dataset.get("x"), [0.0, 2.0])
        assert np.allclose(dataset.get("y"), [1.0, 3.0])

    def test_from_polyline_layer(self):
        """Test that a polyline layer maps pts -> columns x, y."""
        layer = PolylineLayer(pts=np.zeros((4, 2), dtype=np.float32))
        dataset = DataSet.from_layer(layer)
        assert dataset is not None
        assert dataset.column_names() == ["x", "y"]
        assert dataset.n_rows() == 4

    def test_from_line_family_layer(self):
        """Test that a line_family layer maps ab -> columns a, b."""
        ab = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        layer = LineFamilyLayer(ab=ab)
        dataset = DataSet.from_layer(layer)
        assert dataset is not None
        assert dataset.column_names() == ["a", "b"]
        assert np.allclose(dataset.get("a"), [1.0, 3.0])
        assert dataset.binding.attr == "ab"

    def test_from_layer3d(self):
        """Test that a Layer3D maps vertices -> columns x, y, z."""
        vertices = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        layer = Layer3D(vertices=vertices, layer_type="scatter3d")
        dataset = DataSet.from_layer(layer)
        assert dataset is not None
        assert dataset.column_names() == ["x", "y", "z"]
        assert np.allclose(dataset.get("z"), [3.0, 6.0])
        assert dataset.binding.attr == "vertices"

    def test_from_layer_sets_layer_id_and_source(self):
        """Test that from_layer() records layer_id and source='layer'."""
        layer = ScatterLayer(pts=np.zeros((2, 2), dtype=np.float32))
        dataset = DataSet.from_layer(layer)
        assert dataset.layer_id == layer.layer_id
        assert dataset.source == "layer"
        assert dataset.is_bound() is True

    def test_from_layer_uses_label_as_name(self):
        """Test that the layer label becomes the dataset name when present."""
        layer = ScatterLayer(pts=np.zeros((1, 2), dtype=np.float32), label="signal")
        assert DataSet.from_layer(layer).name == "signal"

    def test_from_layer_name_override(self):
        """Test that an explicit name wins over the layer label."""
        layer = ScatterLayer(pts=np.zeros((1, 2), dtype=np.float32), label="signal")
        assert DataSet.from_layer(layer, name="custom").name == "custom"

    def test_from_layer_unlabelled_name_falls_back_to_layer_id(self):
        """Test that an unlabelled layer is named 'Layer <id>'."""
        layer = ScatterLayer(pts=np.zeros((1, 2), dtype=np.float32))
        assert DataSet.from_layer(layer).name == f"Layer {layer.layer_id}"

    def test_columns_are_float64_copies_not_views(self):
        """Test the documented contract: columns are float64 copies of float32 geometry."""
        pts = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        layer = ScatterLayer(pts=pts)
        dataset = DataSet.from_layer(layer)
        assert dataset.get("x").dtype == np.float64
        dataset.set_cell(0, 0, 99.0)
        assert np.allclose(layer.pts[0, 0], 0.0)

    def test_text_layer_has_no_tabular_form(self):
        """Test that from_layer() returns None for a text layer."""
        assert DataSet.from_layer(TextLayer(x=1.0, y=2.0, text="hi")) is None

    def test_patch_layer_has_no_tabular_form(self):
        """Test that a patch layer returns None despite carrying vertices."""
        layer = PatchLayer(vertices=np.zeros((3, 2), dtype=np.float32))
        assert DataSet.from_layer(layer) is None

    def test_layer_with_no_geometry_returns_none(self):
        """Test that a scatter layer with pts=None yields no dataset."""
        assert DataSet.from_layer(ScatterLayer(pts=None)) is None

    def test_wide_geometry_gets_generated_column_names(self):
        """Test that geometry wider than the default names is not truncated."""
        layer = Layer3D(vertices=np.zeros((2, 5), dtype=np.float32))
        dataset = DataSet.from_layer(layer)
        assert dataset.column_names() == ["x", "y", "z", "c4", "c5"]


class TestWriteBack:
    """Test the explicit write-back path onto a layer."""

    def test_cell_edit_reaches_layer_via_write_back(self):
        """Test that set_cell() plus write_back() lands in the layer's array."""
        layer = ScatterLayer(pts=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        dataset = DataSet.from_layer(layer)
        dataset.set_cell(0, 0, 99.0)
        assert dataset.write_back(layer) is True
        assert np.allclose(layer.pts[0, 0], 99.0)
        assert np.allclose(layer.pts[1], [2.0, 3.0])

    def test_write_back_preserves_layer_dtype(self):
        """Test that write-back rebuilds the geometry in the layer's float32 dtype."""
        layer = ScatterLayer(pts=np.zeros((2, 2), dtype=np.float32))
        dataset = DataSet.from_layer(layer)
        dataset.set_cell(0, 0, 1.0)
        dataset.write_back(layer)
        assert layer.pts.dtype == np.float32
        assert layer.pts.flags["C_CONTIGUOUS"]

    def test_write_back_sets_gpu_dirty(self):
        """Test that write-back flags the layer for re-upload."""
        layer = ScatterLayer(pts=np.zeros((2, 2), dtype=np.float32))
        dataset = DataSet.from_layer(layer)
        layer.dirty.clear()
        dataset.write_back(layer)
        assert layer.dirty.gpu_dirty is True

    def test_write_back_after_row_insert_changes_length(self):
        """Test that structural row edits survive write-back."""
        layer = ScatterLayer(pts=np.zeros((2, 2), dtype=np.float32))
        dataset = DataSet.from_layer(layer)
        dataset.insert_row(0)
        dataset.write_back(layer)
        assert layer.pts.shape == (3, 2)

    def test_write_back_line_family(self):
        """Test that a line_family dataset writes back to the ab attribute."""
        layer = LineFamilyLayer(ab=np.zeros((2, 2), dtype=np.float32))
        dataset = DataSet.from_layer(layer)
        dataset.set_cell(1, 1, 4.0)
        assert dataset.write_back(layer) is True
        assert np.allclose(layer.ab[1, 1], 4.0)

    def test_write_back_layer3d(self):
        """Test that a Layer3D dataset writes back to vertices."""
        layer = Layer3D(vertices=np.zeros((2, 3), dtype=np.float32))
        dataset = DataSet.from_layer(layer)
        dataset.set_cell(0, 2, 8.0)
        assert dataset.write_back(layer) is True
        assert np.allclose(layer.vertices[0, 2], 8.0)

    def test_write_back_unbound_returns_false(self):
        """Test that an unbound dataset refuses to write back."""
        layer = ScatterLayer(pts=np.zeros((2, 2), dtype=np.float32))
        assert _dataset().write_back(layer) is False

    def test_write_back_wrong_layer_returns_false(self):
        """Test that write-back to a different layer is refused."""
        layer = ScatterLayer(pts=np.zeros((2, 2), dtype=np.float32))
        other = ScatterLayer(pts=np.ones((2, 2), dtype=np.float32))
        dataset = DataSet.from_layer(layer)
        assert dataset.write_back(other) is False
        assert np.allclose(other.pts, 1.0)

    def test_write_back_after_bound_column_removed_returns_false(self):
        """Test that removing a bound column unbinds and blocks write-back."""
        layer = ScatterLayer(pts=np.zeros((2, 2), dtype=np.float32))
        dataset = DataSet.from_layer(layer)
        dataset.remove_column("y")
        assert dataset.is_bound() is False
        assert dataset.write_back(layer) is False

    def test_rename_bound_column_keeps_link(self):
        """Test that renaming a bound column updates the binding, not breaks it."""
        layer = ScatterLayer(pts=np.zeros((2, 2), dtype=np.float32))
        dataset = DataSet.from_layer(layer)
        dataset.rename_column("x", "time")
        assert dataset.binding.columns == ["time", "y"]
        dataset.set_cell(0, 0, 3.0)
        assert dataset.write_back(layer) is True
        assert np.allclose(layer.pts[0, 0], 3.0)

    def test_unbind_keeps_data(self):
        """Test that unbind() drops the link but leaves the table intact."""
        layer = ScatterLayer(pts=np.ones((2, 2), dtype=np.float32))
        dataset = DataSet.from_layer(layer)
        dataset.unbind()
        assert dataset.is_bound() is False
        assert dataset.layer_id is None
        assert dataset.source == "derived"
        assert np.allclose(dataset.get("x"), [1.0, 1.0])

    def test_copy_of_layer_dataset_drops_the_link(self):
        """Test that a copy of a layer-backed dataset is derived and unbound."""
        layer = ScatterLayer(pts=np.ones((2, 2), dtype=np.float32))
        clone = DataSet.from_layer(layer).copy()
        assert clone.source == "derived"
        assert clone.layer_id is None
        assert clone.is_bound() is False
        assert clone.write_back(layer) is False


class TestSyncFromLayer:
    """Test re-reading a layer into a bound dataset."""

    def test_sync_discards_local_edits(self):
        """Test that sync_from_layer() overwrites local edits with layer data."""
        layer = ScatterLayer(pts=np.zeros((2, 2), dtype=np.float32))
        dataset = DataSet.from_layer(layer)
        dataset.set_cell(0, 0, 5.0)
        layer.pts = np.full((2, 2), 7.0, dtype=np.float32)
        assert dataset.sync_from_layer(layer) is True
        assert np.allclose(dataset.get("x"), [7.0, 7.0])

    def test_sync_unbound_returns_false(self):
        """Test that syncing an unbound dataset returns False."""
        layer = ScatterLayer(pts=np.zeros((2, 2), dtype=np.float32))
        assert _dataset().sync_from_layer(layer) is False

    def test_sync_shape_mismatch_returns_false(self):
        """Test that a geometry with the wrong column count is refused."""
        layer = ScatterLayer(pts=np.zeros((2, 2), dtype=np.float32))
        dataset = DataSet.from_layer(layer)
        layer.pts = np.zeros((2, 3), dtype=np.float32)
        assert dataset.sync_from_layer(layer) is False

    def test_sync_wrong_layer_returns_false(self):
        """Test that syncing from a different layer is refused."""
        layer = ScatterLayer(pts=np.zeros((2, 2), dtype=np.float32))
        other = ScatterLayer(pts=np.ones((2, 2), dtype=np.float32))
        dataset = DataSet.from_layer(layer)
        assert dataset.sync_from_layer(other) is False


class TestDataStore:
    """Test the DataStore registry."""

    def test_add_and_get(self):
        """Test that a registered dataset is retrievable by name."""
        store = DataStore()
        dataset = store.add(_dataset("mine"))
        assert store.get("mine") is dataset
        assert len(store) == 1

    def test_get_missing_returns_none(self):
        """Test that get() returns None for an unknown name."""
        assert DataStore().get("nope") is None

    def test_add_renames_on_collision(self):
        """Test that a colliding dataset name is suffixed on add."""
        store = DataStore()
        store.add(_dataset("d"))
        second = store.add(_dataset("d"))
        assert second.name == "d (2)"
        assert store.names() == ["d", "d (2)"]

    def test_re_adding_same_object_is_noop(self):
        """Test that adding an already-registered dataset does not rename it."""
        store = DataStore()
        dataset = store.add(_dataset("d"))
        store.add(dataset)
        assert len(store) == 1
        assert dataset.name == "d"

    def test_unique_name(self):
        """Test that unique_name() returns the base when free, else a suffix."""
        store = DataStore()
        assert store.unique_name("d") == "d"
        store.add(_dataset("d"))
        assert store.unique_name("d") == "d (2)"
        store.add(_dataset("d"))
        assert store.unique_name("d") == "d (3)"

    def test_remove(self):
        """Test that remove() unregisters the dataset and reports True."""
        store = DataStore()
        dataset = store.add(_dataset("d"))
        assert store.remove(dataset) is True
        assert len(store) == 0
        assert store.get("d") is None

    def test_remove_unregistered_returns_false(self):
        """Test that removing a dataset that was never added returns False."""
        assert DataStore().remove(_dataset()) is False

    def test_iteration_order(self):
        """Test that iterating a store yields datasets in insertion order."""
        store = DataStore()
        store.add(_dataset("a"))
        store.add(_dataset("b"))
        assert [dataset.name for dataset in store] == ["a", "b"]

    def test_from_layer_registers(self):
        """Test that DataStore.from_layer() registers the new dataset."""
        store = DataStore()
        layer = ScatterLayer(pts=np.zeros((2, 2), dtype=np.float32), label="L")
        dataset = store.from_layer(layer)
        assert dataset is not None
        assert store.get("L") is dataset

    def test_from_layer_is_idempotent(self):
        """Test that a second from_layer() for the same layer reuses the dataset."""
        store = DataStore()
        layer = ScatterLayer(pts=np.zeros((2, 2), dtype=np.float32), label="L")
        first = store.from_layer(layer)
        assert store.from_layer(layer) is first
        assert len(store) == 1

    def test_from_layer_unsupported_registers_nothing(self):
        """Test that an untabular layer yields None and registers nothing."""
        store = DataStore()
        assert store.from_layer(TextLayer(text="hi")) is None
        assert len(store) == 0

    def test_get_by_layer_id(self):
        """Test that a bound dataset is retrievable by its layer id."""
        store = DataStore()
        layer = ScatterLayer(pts=np.zeros((2, 2), dtype=np.float32))
        dataset = store.from_layer(layer)
        assert store.get_by_layer_id(layer.layer_id) is dataset
        assert store.get_by_layer_id(-1) is None


class TestMoveColumn:
    """Test column reordering (SPEC_FIX P1-I: Data owns the table)."""

    def test_move_column_reorders(self):
        """Test that a column lands at the requested index."""
        ds = DataSet(
            "d",
            [
                Column("a", np.zeros(2)),
                Column("b", np.ones(2)),
                Column("c", np.full(2, 2.0)),
            ],
        )
        assert ds.move_column("c", 0) is True
        assert ds.column_names() == ["c", "a", "b"]

    def test_move_column_clamps_and_reports_missing(self):
        """Test that an out-of-range index clamps and an absent name returns False."""
        ds = _dataset()
        assert ds.move_column("x", 99) is True
        assert ds.column_names() == ["y", "x"]
        assert ds.move_column("nope", 0) is False

    def test_move_column_keeps_the_layer_binding(self):
        """Test that reordering does not unbind, unlike remove_column.

        The binding names its columns rather than indexing them, so column order cannot
        make the geometry unreassemblable.
        """
        layer = ScatterLayer(pts=np.zeros((3, 2), dtype=np.float32))
        ds = DataSet.from_layer(layer)
        ds.add_column("z", np.zeros(3))
        assert ds.move_column("z", 0) is True
        assert ds.is_bound()
        assert ds.write_back(layer) is True


class TestMoveRows:
    """Test row reordering."""

    def test_move_rows_permutes_every_column_identically(self):
        """Test that a row stays one row across all columns."""
        ds = DataSet("d", [Column("x", np.arange(5.0)), Column("y", np.arange(5.0) * 10)])
        assert ds.move_rows([0, 1], 4) is True
        assert np.allclose(ds.get("x"), [2.0, 3.0, 0.0, 1.0, 4.0])
        assert np.allclose(ds.get("y"), ds.get("x") * 10)

    def test_move_rows_to_the_end(self):
        """Test that dest == n_rows moves the rows past the last one."""
        ds = DataSet("d", [Column("x", np.arange(4.0))])
        assert ds.move_rows([0], 4) is True
        assert np.allclose(ds.get("x"), [1.0, 2.0, 3.0, 0.0])

    def test_move_rows_no_op_returns_false(self):
        """Test that moving every row, or none, changes nothing."""
        ds = DataSet("d", [Column("x", np.arange(3.0))])
        assert ds.move_rows([], 0) is False
        assert ds.move_rows([0, 1, 2], 0) is False
        assert ds.move_rows([99], 0) is False


class TestFilterRows:
    """Test the destructive half of the row filter."""

    def test_filter_rows_keeps_the_matches(self):
        """Test that only rows whose mask is True survive, in every column."""
        ds = DataSet("d", [Column("x", np.arange(5.0)), Column("y", np.arange(5.0) - 2)])
        dropped = ds.filter_rows(ds.get("y") > 0)
        assert dropped == 3
        assert np.allclose(ds.get("x"), [3.0, 4.0])
        assert np.allclose(ds.get("y"), [1.0, 2.0])

    def test_filter_rows_rejects_a_wrong_length_mask(self):
        """Test that a mask that is not n_rows long raises instead of truncating."""
        ds = _dataset()
        with pytest.raises(ValueError):
            ds.filter_rows(np.array([True]))

    def test_filter_rows_all_true_is_a_no_op(self):
        """Test that a matching-everything mask drops nothing."""
        ds = _dataset()
        assert ds.filter_rows(np.ones(ds.n_rows(), dtype=bool)) == 0
        assert ds.n_rows() == 3


class TestExpressions:
    """Test expressions evaluated with the columns bound as names."""

    def test_eval_expression_binds_columns(self):
        """Test that column names are usable directly in an expression."""
        ds = _dataset()
        assert np.allclose(ds.eval_expression("x + y"), [10.0, 12.0, 14.0])

    def test_eval_expression_broadcasts_a_scalar(self):
        """Test that a constant expression fills the column."""
        ds = _dataset()
        assert np.allclose(ds.eval_expression("0"), np.zeros(3))

    def test_eval_expression_rejects_an_unknown_name(self):
        """Test that a name no column carries is an error, not a zero."""
        ds = _dataset()
        with pytest.raises(ExpressionError):
            ds.eval_expression("q * 2")

    def test_bindable_columns_excludes_unspeakable_names(self):
        """Test that a column whose name is not an identifier is not bindable."""
        ds = DataSet("d", [Column("x", np.zeros(2)), Column("y (2)", np.zeros(2))])
        assert sorted(ds.bindable_columns()) == ["x"]

    def test_row_mask_is_boolean_and_nan_never_matches(self):
        """Test that a nan row fails the predicate rather than passing it."""
        ds = DataSet("d", [Column("y", np.array([1.0, np.nan, -1.0]))])
        mask = ds.row_mask("y > 0")
        assert mask.dtype == np.bool_
        assert mask.tolist() == [True, False, False]

    def test_row_mask_explains_the_precedence_trap(self):
        """Test that 'y > 0 & x < 10' reports the parenthesis fix.

        Python parses it as y > (0 & x) < 10, which is the spelling every user reaches
        for first; numpy's raw message names 'bitwise_and' and helps nobody.
        """
        ds = _dataset()
        with pytest.raises(ExpressionError) as excinfo:
            ds.row_mask("y > 0 & x < 10")
        assert "Parenthesise" in str(excinfo.value)

    def test_row_mask_explains_the_and_trap(self):
        """Test that 'and' between comparisons reports the &/| fix."""
        ds = _dataset()
        with pytest.raises(ExpressionError) as excinfo:
            ds.row_mask("(y > 0) and (x < 10)")
        assert "'&' and '|'" in str(excinfo.value)

    def test_eval_expression_binds_extra_variables(self):
        """Test that ``variables`` lets a name absent from the table resolve, for a
        Data Editor Transform-section slider parameter."""
        ds = _dataset()
        out = ds.eval_expression("a * x + b", variables={"a": 2.0, "b": 1.0})
        assert np.allclose(out, [1.0, 3.0, 5.0])

    def test_eval_expression_without_variables_is_unchanged(self):
        """Test that omitting ``variables`` behaves exactly as before it existed."""
        ds = _dataset()
        assert np.allclose(ds.eval_expression("x + y"), ds.eval_expression("x + y", variables=None))

    def test_eval_expression_a_column_wins_over_a_same_named_variable(self):
        """Test that real data always beats a stale/colliding parameter name."""
        ds = _dataset()
        out = ds.eval_expression("x", variables={"x": 999.0})
        assert np.allclose(out, [0.0, 1.0, 2.0])

    def test_eval_expression_variables_do_not_leak_into_a_call_without_them(self):
        """Test that one call's variables cannot bleed into a later, unrelated call."""
        ds = _dataset()
        ds.eval_expression("a", variables={"a": 5.0})
        with pytest.raises(ExpressionError):
            ds.eval_expression("a")

    def test_eval_expression_rejects_an_invalid_variable_name(self):
        """Test that a bad key in variables (not the caller's fault to guess) still
        raises ExpressionError, not some other exception, matching every other
        documented failure mode of this method."""
        ds = _dataset()
        with pytest.raises(ExpressionError):
            ds.eval_expression("x", variables={"not an identifier": 1.0})


class TestTransformColumn:
    """Test transforming a column with an expression."""

    def test_transform_in_place(self):
        """Test that target=None overwrites the source column."""
        ds = _dataset()
        column = ds.transform_column("y", "y * 2")
        assert column.name == "y"
        assert np.allclose(ds.get("y"), [20.0, 22.0, 24.0])

    def test_transform_into_a_new_column(self):
        """Test that a target name appends a column and leaves the source alone."""
        ds = _dataset()
        ds.transform_column("y", "y - x")
        ds = _dataset()
        ds.transform_column("y", "y - x", target="d")
        assert ds.column_names() == ["x", "y", "d"]
        assert np.allclose(ds.get("d"), [10.0, 10.0, 10.0])
        assert np.allclose(ds.get("y"), [10.0, 11.0, 12.0])

    def test_transform_reruns_into_the_same_column(self):
        """Test that re-running does not pile up 'd (2)', 'd (3)'..."""
        ds = _dataset()
        ds.transform_column("y", "y - x", target="d")
        ds.transform_column("y", "y + x", target="d")
        assert ds.column_names() == ["x", "y", "d"]
        assert np.allclose(ds.get("d"), [10.0, 12.0, 14.0])

    def test_transform_invalidates_the_stats_cache(self):
        """Test that a transformed column reports its new min/max, not the old one."""
        ds = _dataset()
        assert ds.get("y").max() == 12.0
        column = ds._find("y")
        assert column.max == 12.0
        ds.transform_column("y", "y * 2")
        assert column.max == 24.0

    def test_a_failing_transform_leaves_the_table_untouched(self):
        """Test that there is no half-applied transform."""
        ds = _dataset()
        with pytest.raises(ExpressionError):
            ds.transform_column("y", "y + nope")
        assert np.allclose(ds.get("y"), [10.0, 11.0, 12.0])

    def test_transform_rejects_a_missing_column(self):
        """Test that transforming a column that does not exist raises."""
        ds = _dataset()
        with pytest.raises(ValueError):
            ds.transform_column("nope", "1")

    def test_transform_bakes_in_the_given_variables(self):
        """Test that a parametrized transform (a*x+b) commits with the CURRENT slider
        values -- the "Apply" of Data Editor's Transform-section sliders."""
        ds = _dataset()
        column = ds.transform_column("y", "a * x + b", variables={"a": 2.0, "b": 1.0})
        assert np.allclose(column.values, [1.0, 3.0, 5.0])

    def test_transform_variables_do_not_persist_across_calls(self):
        """Test that re-running without variables is an unknown-name error again --
        a baked-in value is a one-time snapshot, not a remembered binding."""
        ds = _dataset()
        ds.transform_column("y", "a", target="d", variables={"a": 5.0})
        with pytest.raises(ExpressionError):
            ds.transform_column("y", "a", target="d2")


class TestDerivedBinding:
    """Test that a derived binding refuses write_back (SPEC_FIX P1-I plot types).

    A bar layer's geometry is 4N corners computed from N rows. Column-stacking the table
    onto it would not raise; it would quietly draw nonsense.
    """

    def test_write_back_refuses_a_derived_binding(self):
        """Test that write_back is a no-op when the geometry is computed."""
        layer = ScatterLayer(pts=np.zeros((3, 2), dtype=np.float32))
        ds = DataSet.from_layer(layer)
        ds.binding.derived = True
        ds.binding.kind = "bar"
        assert ds.write_back(layer) is False

    def test_sync_from_layer_refuses_a_derived_binding(self):
        """Test that reading back is refused too: the geometry is not the columns."""
        layer = ScatterLayer(pts=np.zeros((3, 2), dtype=np.float32))
        ds = DataSet.from_layer(layer)
        ds.binding.derived = True
        assert ds.sync_from_layer(layer) is False

    def test_a_plain_binding_still_writes_back(self):
        """Test that the default binding is not derived, so nothing regressed."""
        layer = ScatterLayer(pts=np.zeros((3, 2), dtype=np.float32))
        ds = DataSet.from_layer(layer)
        assert ds.binding.derived is False
        assert ds.write_back(layer) is True
