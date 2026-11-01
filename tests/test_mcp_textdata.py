"""Text columns (names, dates) in loaded tables: categories, ``select_rows`` and ``aggregate``.

Checked with small files whose answers are known by construction. No MCP SDK is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.gui import dataio
from glplot.gui.clipboard import cells_to_array, parse_table, parse_table_cells
from glplot.gui.datasets import Column, DataSet
from glplot.mcp.state import FileSandbox, McpError
from glplot.mcp.textdata import (
    MAX_CATEGORIES,
    describe_encoding,
    interpret_text_columns,
    parse_date,
)
from glplot.mcp.toolbox import Toolbox

WEATHER = """city,year,temp,when
Madrid,2020,15.0,2020-01-05
Madrid,2021,16.0,2021-01-05
Lima,2020,19.0,2020-01-06
Lima,2021,19.5,2021-01-06
Oslo,2020,6.0,2020-01-07
Oslo,2021,6.6,2021-01-07
"""


@pytest.fixture
def box(tmp_path):
    (tmp_path / "weather.csv").write_text(WEATHER)
    toolbox = Toolbox(FileSandbox([str(tmp_path)]))
    toolbox.load_dataset("weather.csv", "w")
    yield toolbox
    toolbox.close_all()


def drawn_tick_labels(box, figure_id, monkeypatch):
    """The x tick labels of the rendered matplotlib figure (render_figure does not report them)."""
    from glplot.mcp import toolbox as toolbox_module

    seen = []
    real = toolbox_module._summarise_axes

    def spy(fig):
        seen.append([t.get_text() for t in fig.axes[0].get_xticklabels()])
        return real(fig)

    monkeypatch.setattr(toolbox_module, "_summarise_axes", spy)
    box.render_figure(figure_id)
    return seen[-1]


def column(box, dataset, name):
    return np.asarray(box.datasets.column(dataset, name))


# -- parsing ---------------------------------------------------------------------------------


def test_parse_date_iso_forms():
    assert parse_date("1970-01-02") == 1.0
    assert parse_date("1970-01-01 12:00") == 0.5
    assert parse_date("1970-01-01T12:00:00+00:00") == 0.5
    assert parse_date("1970-01-01T01:00:00+01:00") == 0.0  # same instant, other zone


@pytest.mark.parametrize("text", ["", "  ", "Lima", "5", "2020-13-45", "1/2/2020"])
def test_parse_date_rejects_non_dates(text):
    assert parse_date(text) is None


def test_text_columns_returns_only_columns_without_a_number(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("name,x,note\na,1,\nb,2,hi\nc,3,\n")
    found = dataio.text_columns(str(path))
    assert list(found) == ["name", "note"]
    assert found["name"] == ["a", "b", "c"]
    assert found["note"] == ["", "hi", ""]


def test_text_columns_of_a_numeric_file_is_empty(tmp_path):
    path = tmp_path / "n.csv"
    path.write_text("x,y\n1,2\n3,4\n")
    assert dataio.text_columns(str(path)) == {}


def test_cell_parser_agrees_with_parse_table():
    text = "a;b;c\n1;2;x\n3;4,5;y\n5;;z\n"
    headers, body = parse_table_cells(text)
    parsed_headers, parsed_data = parse_table(text)
    assert headers == list(parsed_headers)
    assert np.array_equal(cells_to_array(body, len(headers)), parsed_data, equal_nan=True)


# -- interpret_text_columns --------------------------------------------------------------------


def _make(n, names):
    return DataSet("t", [Column(name, np.full(n, np.nan)) for name in names])


def test_categories_are_numbered_in_order_of_first_appearance():
    table = _make(4, ["g"])
    report = interpret_text_columns(table, {"g": ["b", "a", "b", ""]})
    assert table.categories["g"] == ["b", "a"]
    values = np.asarray(table.get("g"))
    assert values[:3].tolist() == [0.0, 1.0, 0.0] and np.isnan(values[3])
    assert report["categories"] == {"g": ["b", "a"]}


def test_dates_become_days_and_are_flagged():
    table = _make(3, ["d"])
    report = interpret_text_columns(table, {"d": ["1970-01-01", "1970-01-03", ""]})
    values = np.asarray(table.get("d"))
    assert values[:2].tolist() == [0.0, 2.0] and np.isnan(values[2])
    assert "d" in table.date_columns and "d" not in table.categories
    assert "d" in report["dates"]


def test_a_mix_of_dates_and_words_is_a_category():
    table = _make(2, ["d"])
    interpret_text_columns(table, {"d": ["1970-01-01", "soon"]})
    assert table.categories["d"] == ["1970-01-01", "soon"]


def test_free_text_with_too_many_values_is_left_empty_and_reported():
    n = MAX_CATEGORIES + 1
    table = _make(n, ["id"])
    report = interpret_text_columns(table, {"id": [f"row{i}" for i in range(n)]})
    assert "id" not in table.categories
    assert np.isnan(np.asarray(table.get("id"))).all()
    assert "distinct" in report["ignored"]["id"]


def test_misaligned_column_is_reported_not_encoded():
    table = _make(3, ["g"])
    report = interpret_text_columns(table, {"g": ["a", "b"]})
    assert "g" not in table.categories
    assert "line up" in report["ignored"]["g"]


def test_describe_encoding_names_each_kind():
    text = describe_encoding(
        {"categories": {"c": ["x", "y"]}, "dates": {"d": "days"}, "ignored": {"i": "too many"}}
    )
    assert (
        "c: 2 categories (0=x, 1=y)" in text
        and "d: dates as days" in text
        and "i: too many" in text
    )
    assert describe_encoding({}) == ""


# -- load_dataset and describe_dataset ---------------------------------------------------------


def test_load_keeps_text_columns_instead_of_nan(box):
    table = box.datasets.get("w")
    assert table.categories["city"] == ["Madrid", "Lima", "Oslo"]
    assert column(box, "w", "city").tolist() == [0, 0, 1, 1, 2, 2]
    assert "when" in table.date_columns
    when = column(box, "w", "when")
    assert when[1] - when[0] == 366.0  # 2020-01-05 -> 2021-01-05 spans a leap day
    assert not np.isnan(column(box, "w", "temp")).any()


def test_load_reply_mentions_the_text_columns(tmp_path, box):
    reply = box.load_dataset("weather.csv", "again")
    assert "categories" in reply.message and "city" in reply.message


def test_describe_marks_category_and_date_columns(box):
    columns = {c["name"]: c for c in box.describe_dataset("w").data["columns"]}
    assert columns["city"]["kind"] == "category"
    assert columns["city"]["labels"] == ["Madrid", "Lima", "Oslo"]
    assert columns["when"]["kind"] == "date"
    assert "kind" not in columns["temp"]


# -- select_rows -----------------------------------------------------------------------------


def test_select_by_label(box):
    reply = box.select_rows("w", equals={"city": "Lima"})
    assert reply.data["rows_out"] == 2 and reply.data["dataset"] == "w_rows"
    assert column(box, "w_rows", "temp").tolist() == [19.0, 19.5]
    assert box.datasets.get("w_rows").categories["city"] == ["Madrid", "Lima", "Oslo"]
    assert box.datasets.get("w").n_rows() == 6  # the source is untouched


def test_select_by_several_labels_and_a_number(box):
    box.select_rows("w", equals={"city": ["Lima", "Oslo"], "year": 2021}, output="x")
    assert column(box, "x", "temp").tolist() == [19.5, 6.6]


def test_select_with_expression(box):
    box.select_rows("w", where="(temp > 10) & (year == 2021)", output="x")
    assert column(box, "x", "temp").tolist() == [16.0, 19.5]


def test_select_sort_every_limit(box):
    box.select_rows("w", sort_by="temp", descending=True, limit=3, output="top")
    assert column(box, "top", "temp").tolist() == [19.5, 19.0, 16.0]
    box.select_rows("w", sort_by="temp", output="asc")
    assert column(box, "asc", "temp").tolist() == sorted(column(box, "w", "temp").tolist())
    box.select_rows("w", every=2, output="thin")
    assert column(box, "thin", "year").tolist() == [2020.0, 2020.0, 2020.0]


def test_descending_sort_keeps_ties_in_original_order(box):
    box.create_dataset("ties", {"k": [1, 2, 1, 2], "i": [0, 1, 2, 3]})
    box.select_rows("ties", sort_by="k", descending=True, output="s")
    assert column(box, "s", "i").tolist() == [1.0, 3.0, 0.0, 2.0]


def test_sort_puts_nan_last(box):
    box.create_dataset("n", {"k": [3.0, float("nan"), 1.0]})
    box.select_rows("n", sort_by="k", output="s")
    got = column(box, "s", "k")
    assert got[:2].tolist() == [1.0, 3.0] and np.isnan(got[2])


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({}, "Nothing to do"),
        ({"equals": {"city": "Lma"}}, "Did you mean 'Lima'"),
        ({"equals": {"cty": "Lima"}}, "no column 'cty'"),
        ({"equals": {"temp": "hot"}}, "numeric"),
        ({"equals": {"city": 3}}, "not a label"),
        ({"equals": ["city"]}, "equals"),
        ({"where": "temp > 1000"}, "No rows match"),
        ({"where": "temp > 1 and year > 1"}, "Use '&' and '|'"),
        ({"where": "nope > 1"}, "where"),
        ({"sort_by": "tmp"}, "Did you mean 'temp'"),
        ({"limit": 0}, "at least 1"),
        ({"every": -2}, "at least 1"),
    ],
)
def test_select_errors_say_what_to_do(box, kwargs, fragment):
    with pytest.raises(McpError, match=fragment):
        box.select_rows("w", **kwargs)


def test_select_unknown_dataset(box):
    with pytest.raises(McpError):
        box.select_rows("missing", limit=1)


def test_select_output_replaces_only_when_asked_name_is_reused(box):
    box.select_rows("w", limit=1, output="x")
    again = box.select_rows("w", limit=2, output="x")
    assert again.data["replaced"] is True and box.datasets.get("x").n_rows() == 2


# -- aggregate -------------------------------------------------------------------------------


def test_aggregate_mean_per_category(box):
    reply = box.analyze_dataset("w", "aggregate", y="temp", options={"by": "city"})
    name = reply.data["created"]["dataset"]
    assert column(box, name, "temp_mean").tolist() == pytest.approx([15.5, 19.25, 6.3])
    assert column(box, name, "count").tolist() == [2, 2, 2]
    assert box.datasets.get(name).categories["city"] == ["Madrid", "Lima", "Oslo"]


@pytest.mark.parametrize(
    "stat, expected",
    [
        ("sum", [31.0, 38.5, 12.6]),
        ("min", [15.0, 19.0, 6.0]),
        ("max", [16.0, 19.5, 6.6]),
        ("median", [15.5, 19.25, 6.3]),
        ("count", [2, 2, 2]),
    ],
)
def test_aggregate_statistics(box, stat, expected):
    reply = box.analyze_dataset("w", "aggregate", y="temp", options={"by": "city", "stat": stat})
    assert column(box, reply.data["created"]["dataset"], f"temp_{stat}").tolist() == pytest.approx(
        expected
    )


def test_aggregate_by_a_numeric_key(box):
    reply = box.analyze_dataset("w", "aggregate", y="temp", options={"by": "year"})
    name = reply.data["created"]["dataset"]
    assert column(box, name, "year").tolist() == [2020.0, 2021.0]
    assert column(box, name, "temp_mean").tolist() == pytest.approx(
        [(15 + 19 + 6) / 3, (16 + 19.5 + 6.6) / 3]
    )


def test_aggregate_errors(box):
    with pytest.raises(McpError, match="by"):
        box.analyze_dataset("w", "aggregate", y="temp")
    with pytest.raises(McpError, match="stat"):
        box.analyze_dataset("w", "aggregate", y="temp", options={"by": "city", "stat": "mode"})
    with pytest.raises(McpError, match="city"):
        box.analyze_dataset("w", "aggregate", y="temp", options={"by": "cty"})


# -- plotting a category -----------------------------------------------------------------------


def test_bar_takes_tick_labels_from_the_category_column(box, monkeypatch):
    name = box.analyze_dataset("w", "aggregate", y="temp", options={"by": "city"}).data["created"][
        "dataset"
    ]
    figure = box.create_figure().data["figure_id"]
    reply = box.add_plot(figure, "bar", dataset=name, x="city", y="temp_mean")
    assert any("Tick labels taken from" in note for note in reply.data["notes"])
    assert drawn_tick_labels(box, figure, monkeypatch) == ["Madrid", "Lima", "Oslo"]


def test_explicit_tick_labels_win_over_categories(box):
    name = box.analyze_dataset("w", "aggregate", y="temp", options={"by": "city"}).data["created"][
        "dataset"
    ]
    figure = box.create_figure().data["figure_id"]
    reply = box.add_plot(
        figure, "bar", dataset=name, x="city", y="temp_mean", tick_labels=["a", "b", "c"]
    )
    assert not any("Tick labels taken" in note for note in reply.data.get("notes", []))


def test_pie_names_slices_after_the_categories(box):
    name = box.analyze_dataset(
        "w", "aggregate", y="temp", options={"by": "city", "stat": "sum"}
    ).data["created"]["dataset"]
    figure = box.create_figure().data["figure_id"]
    reply = box.add_plot(figure, "pie", dataset=name, x="city", y="temp_sum")
    assert any("Slice names" in note for note in reply.data["notes"])


def test_scatter_on_a_category_explains_the_indices(box):
    figure = box.create_figure().data["figure_id"]
    reply = box.add_plot(figure, "scatter", dataset="w", x="city", y="temp")
    assert "0=Madrid" in reply.data["notes"][0]


def test_plain_numeric_plot_has_no_notes(box):
    figure = box.create_figure().data["figure_id"]
    reply = box.add_plot(figure, "scatter", dataset="w", x="year", y="temp")
    assert "notes" not in reply.data
