"""The MCP server's guard rails: file sandbox, bounded expressions, dataset store.

These are the rules that keep a language model from reading ``~/.ssh`` or asking for
``zeros(10**12)``, so each is tested from the attack side: what a model (or a prompt-injected
document it was reading) would actually send.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from glplot.gui.datasets import Column, DataSet
from glplot.mcp.state import (
    DatasetStore,
    FileSandbox,
    Limits,
    McpError,
    as_float_array,
    check_name,
    evaluate_expression,
    expression_vocabulary,
    slug,
    suggest,
)


@pytest.fixture
def root(tmp_path):
    base = tmp_path / "data"
    base.mkdir()
    (base / "table.csv").write_text("a,b\n1,2\n3,4\n")
    (base / "notes.py").write_text("print('hi')")
    (tmp_path / "secret.csv").write_text("a\n1\n")
    return base


class TestFileSandboxReads:
    def test_reads_a_file_inside_the_root(self, root):
        box = FileSandbox([root])
        assert box.read_path("table.csv") == (root / "table.csv").resolve()

    def test_relative_paths_start_at_the_first_root(self, root, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        assert FileSandbox([root, other]).read_path("table.csv").parent == root.resolve()

    def test_absolute_path_inside_the_root_is_fine(self, root):
        assert FileSandbox([root]).read_path(str(root / "table.csv"))

    def test_parent_traversal_is_refused(self, root):
        with pytest.raises(McpError, match="outside"):
            FileSandbox([root]).read_path("../secret.csv")

    def test_absolute_path_outside_the_root_is_refused(self, root, tmp_path):
        with pytest.raises(McpError, match="outside"):
            FileSandbox([root]).read_path(str(tmp_path / "secret.csv"))

    @pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
    def test_symlink_pointing_out_of_the_root_is_refused(self, root, tmp_path):
        (root / "link.csv").symlink_to(tmp_path / "secret.csv")
        with pytest.raises(McpError, match="outside"):
            FileSandbox([root]).read_path("link.csv")

    def test_unknown_extension_is_refused(self, root):
        with pytest.raises(McpError, match="Cannot read"):
            FileSandbox([root]).read_path("notes.py")

    def test_missing_file_says_where_it_looked(self, root):
        with pytest.raises(McpError, match="not found"):
            FileSandbox([root]).read_path("nope.csv")

    def test_directory_is_not_a_file(self, root):
        (root / "sub.csv").mkdir()
        with pytest.raises(McpError, match="regular file"):
            FileSandbox([root]).read_path("sub.csv")

    def test_oversized_file_is_refused(self, root):
        with pytest.raises(McpError, match="MB"):
            FileSandbox([root], max_read_bytes=4).read_path("table.csv")

    def test_no_roots_disables_everything_and_says_how_to_enable(self):
        box = FileSandbox(())
        assert not box.enabled
        with pytest.raises(McpError, match="--root"):
            box.read_path("table.csv")
        with pytest.raises(McpError, match="--root"):
            box.write_path("out.png")

    def test_the_filesystem_root_is_never_a_root(self):
        with pytest.raises(ValueError, match="whole filesystem"):
            FileSandbox([os.path.abspath(os.sep)])

    def test_a_file_is_not_a_root(self, root):
        with pytest.raises(ValueError, match="not a directory"):
            FileSandbox([root / "table.csv"])


class TestFileSandboxWrites:
    def test_writes_an_image_inside_the_root(self, root):
        target = FileSandbox([root]).write_path("out/fig.png")
        assert target.parent == (root / "out").resolve()
        assert target.parent.is_dir()

    @pytest.mark.parametrize("name", ["evil.py", "run.sh", "noext", "data.csv", "x.npy"])
    def test_only_image_formats_may_be_written(self, root, name):
        with pytest.raises(McpError, match="Cannot save"):
            FileSandbox([root]).write_path(name)

    def test_refuses_to_overwrite_by_default(self, root):
        (root / "fig.png").write_bytes(b"x")
        with pytest.raises(McpError, match="already exists"):
            FileSandbox([root]).write_path("fig.png")
        assert FileSandbox([root]).write_path("fig.png", overwrite=True)

    def test_cannot_write_outside_the_root(self, root):
        with pytest.raises(McpError, match="outside"):
            FileSandbox([root]).write_path("../escape.png")

    def test_a_directory_is_not_a_target(self, root):
        (root / "d.png").mkdir()
        with pytest.raises(McpError, match="directory"):
            FileSandbox([root]).write_path("d.png", overwrite=True)


class TestExpressions:
    def test_evaluates_with_variables(self):
        out = evaluate_expression("a * x + 1", {"a": 2.0, "x": np.arange(3.0)})
        assert out.tolist() == [1.0, 3.0, 5.0]

    @pytest.mark.parametrize(
        "expr",
        [
            "zeros(10**9)",
            "ones(5e9)",
            "full(10**10, 1.0)",
            "arange(1e12)",
            "arange(0, 1e12, 1)",
            "linspace(0, 1, 10**10)",
            "noise(10**10)",
            "zeros((100000, 100000))",
        ],
    )
    def test_memory_bombs_are_refused_before_allocating(self, expr):
        with pytest.raises(McpError, match="exceeds the limit|valid number"):
            evaluate_expression(expr)

    def test_reasonable_constructors_still_work(self):
        assert evaluate_expression("linspace(0, 1, 1000)").size == 1000
        assert evaluate_expression("arange(10)").size == 10
        assert evaluate_expression("zeros(100)").sum() == 0.0
        assert evaluate_expression("noise(50, 0.1, 3)").size == 50

    def test_limit_is_configurable(self):
        tiny = Limits(max_array_elements=10)
        with pytest.raises(McpError):
            evaluate_expression("linspace(0, 1, 11)", limits=tiny)
        assert evaluate_expression("linspace(0, 1, 10)", limits=tiny).size == 10

    @pytest.mark.parametrize(
        "expr",
        [
            "__import__('os').system('true')",
            "().__class__",
            "open('x')",
            "lambda: 1",
            "[x for x in y]",
            "np.sin(1)",
            "eval('1')",
        ],
    )
    def test_the_evaluators_allowlist_still_applies(self, expr):
        with pytest.raises(McpError, match="Could not evaluate"):
            evaluate_expression(expr, {"y": np.arange(3.0)})

    def test_errors_quote_the_expression(self):
        with pytest.raises(McpError, match=r"nonsense\(1\)"):
            evaluate_expression("nonsense(1)")

    def test_overlong_expression_is_refused(self):
        with pytest.raises(McpError, match="characters"):
            evaluate_expression("1+" * 2000 + "1")

    def test_blank_expression_is_refused(self):
        with pytest.raises(McpError):
            evaluate_expression("   ")

    def test_non_identifier_variable_names_are_not_bound(self):
        # "Temp (K)" cannot be typed in an expression, so it must not leak into the namespace
        # as something the evaluator would reject wholesale.
        out = evaluate_expression("t + 1", {"t": np.arange(2.0), "Temp (K)": np.arange(2.0)})
        assert out.tolist() == [1.0, 2.0]

    def test_vocabulary_is_generated_from_the_evaluator(self):
        vocab = expression_vocabulary()
        assert "sin" in vocab["functions"] and "linspace" in vocab["functions"]
        assert "pi" in vocab["constants"]


class TestAsFloatArray:
    def test_scalar_repeats_to_length(self):
        assert as_float_array(2.0, "c", length=3).tolist() == [2.0, 2.0, 2.0]

    def test_scalar_without_length_is_an_error(self):
        with pytest.raises(McpError, match="length"):
            as_float_array(2.0, "c")

    def test_length_mismatch_names_the_column(self):
        with pytest.raises(McpError, match="column 'b'.*3.*4"):
            as_float_array([1, 2, 3], "column 'b'", length=4)

    def test_strings_are_not_numbers(self):
        with pytest.raises(McpError, match="tick_labels"):
            as_float_array(["a", "b"], "y")

    def test_complex_is_refused_with_a_hint(self):
        with pytest.raises(McpError, match="abs"):
            as_float_array(np.array([1 + 2j]), "y")

    def test_nested_lists_are_refused(self):
        with pytest.raises(McpError, match="flat"):
            as_float_array([[1, 2], [3, 4]], "y")


class TestNames:
    @pytest.mark.parametrize("name", ["a", "data-1", "run_2.final", "X" * 64])
    def test_valid_names(self, name):
        assert check_name(name) == name

    @pytest.mark.parametrize("name", ["", "has space", "slash/name", "../x", "X" * 65, "ü", None])
    def test_invalid_names(self, name):
        with pytest.raises(McpError):
            check_name(name)

    def test_slug_makes_a_valid_name_from_a_filename(self):
        assert check_name(slug("my data (v2)"))
        assert slug("///") == "data"

    def test_suggest_offers_a_near_miss_only(self):
        assert "fig1" in suggest("fig", ["fig1", "other"])
        assert suggest("zzzz", ["fig1"]) == ""


class TestDatasetStore:
    def _table(self, name="d", rows=3):
        return DataSet(name, [Column("x", np.arange(rows, dtype=float))])

    def test_put_get_remove(self):
        store = DatasetStore()
        assert store.put(self._table()) is False
        assert store.get("d").n_rows() == 3
        assert store.put(self._table()) is True  # replaced
        store.remove("d")
        assert "d" not in store

    def test_missing_dataset_lists_what_exists_and_suggests(self):
        store = DatasetStore()
        store.put(self._table("signal"))
        with pytest.raises(McpError, match="signal") as info:
            store.get("signl")
        assert "Did you mean 'signal'" in str(info.value)

    def test_dataset_count_is_bounded(self):
        store = DatasetStore(Limits(max_datasets=2))
        store.put(self._table("a"))
        store.put(self._table("b"))
        with pytest.raises(McpError, match="Too many"):
            store.put(self._table("c"))
        store.put(self._table("a"))  # replacing is not growing

    def test_row_count_is_bounded(self):
        store = DatasetStore(Limits(max_dataset_rows=5))
        with pytest.raises(McpError, match="rows"):
            store.put(self._table(rows=6))

    def test_column_returns_a_copy(self):
        store = DatasetStore()
        store.put(self._table())
        values = store.column("d", "x")
        values[:] = 99
        assert store.column("d", "x").tolist() == [0.0, 1.0, 2.0]

    def test_wrong_column_lists_columns(self):
        store = DatasetStore()
        store.put(self._table())
        with pytest.raises(McpError, match="columns: x"):
            store.column("d", "y")

    def test_describe_is_json_safe_with_nan(self):
        import json

        store = DatasetStore()
        store.put(DataSet("n", [Column("v", np.array([1.0, np.nan, 3.0]))]))
        info = store.describe("n")
        assert json.dumps(info)  # no NaN/Infinity literals
        assert info["columns"][0]["nan_count"] == 1
        assert info["head"][1] == [None]
