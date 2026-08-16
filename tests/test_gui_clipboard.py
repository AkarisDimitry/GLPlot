"""Test the clipboard text<->table codec in glplot.gui.clipboard.

Focus on delimiter detection, header detection, nan handling and the
format->parse round trip without requiring OpenGL or GPU. The module under
test is pure logic (numpy + stdlib only), so nothing here needs a GL context
or an imgui frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.gui.clipboard import detect_delimiter, format_table, parse_table


class TestDetectDelimiter:
    """Test delimiter auto-detection and its priority order."""

    def test_tab_detected(self):
        """Tab-separated text resolves to tab."""
        assert detect_delimiter("a\tb\n1\t2") == "\t"

    def test_comma_detected(self):
        """Comma-separated text resolves to comma."""
        assert detect_delimiter("a,b\n1,2") == ","

    def test_semicolon_detected(self):
        """Semicolon-separated text resolves to semicolon."""
        assert detect_delimiter("a;b\n1;2") == ";"

    def test_whitespace_returns_none(self):
        """Text with no candidate delimiter returns None, meaning 'split on whitespace'."""
        assert detect_delimiter("1 2\n3 4") is None

    def test_tab_wins_over_comma(self):
        """Tab outranks comma, so an Excel paste with quoted commas resolves to tab."""
        assert detect_delimiter('"a,b"\t1') == "\t"

    def test_comma_wins_over_semicolon(self):
        """Comma outranks semicolon, per the documented priority order."""
        assert detect_delimiter("a,b;c") == ","


class TestParseTableDelimiters:
    """Test parse_table across the delimiter variants it must accept."""

    def test_excel_tsv_with_crlf(self):
        """An Excel paste is tab-separated with CRLF line endings and a trailing newline."""
        headers, data = parse_table("x\ty\r\n1\t2\r\n3\t4\r\n")
        assert headers == ["x", "y"]
        assert data.shape == (2, 2)
        assert np.allclose(data, [[1.0, 2.0], [3.0, 4.0]])

    def test_bare_cr_line_endings(self):
        """Classic Mac exports use a bare CR as the line terminator."""
        headers, data = parse_table("a,b\r1,2\r3,4")
        assert headers == ["a", "b"]
        assert np.allclose(data, [[1.0, 2.0], [3.0, 4.0]])

    def test_semicolon_separated(self):
        """Semicolon-separated text parses like any other delimited text."""
        headers, data = parse_table("a;b\n1;2")
        assert headers == ["a", "b"]
        assert np.allclose(data, [[1.0, 2.0]])

    def test_whitespace_separated(self):
        """Runs of whitespace act as the delimiter when no candidate is present."""
        headers, data = parse_table("1   2\n3\t\t4".replace("\t\t", "   "))
        assert headers == ["col1", "col2"]
        assert np.allclose(data, [[1.0, 2.0], [3.0, 4.0]])

    def test_leading_bom_stripped(self):
        """A UTF-8 BOM from a CSV export must not corrupt the first header name."""
        headers, data = parse_table("﻿x,y\n1,2")
        assert headers == ["x", "y"]
        assert np.allclose(data, [[1.0, 2.0]])

    def test_single_column(self):
        """A single column of numbers has no delimiter and still parses."""
        headers, data = parse_table("1\n2\n3")
        assert headers == ["col1"]
        assert data.shape == (3, 1)
        assert np.allclose(data, [[1.0], [2.0], [3.0]])


class TestParseTableHeaders:
    """Test header-row detection."""

    def test_csv_with_header(self):
        """A first row holding non-numeric cells is a header."""
        headers, data = parse_table("time,value\n0,1.5\n1,2.5")
        assert headers == ["time", "value"]
        assert np.allclose(data, [[0.0, 1.5], [1.0, 2.5]])

    def test_csv_without_header(self):
        """An all-numeric first row is data; headers are generated."""
        headers, data = parse_table("1,2\n3,4")
        assert headers == ["col1", "col2"]
        assert data.shape == (2, 2)
        assert np.allclose(data, [[1.0, 2.0], [3.0, 4.0]])

    def test_partial_header_row_is_a_header(self):
        """One non-numeric cell is enough to make the whole first row a header."""
        headers, data = parse_table("x,2\n1,2")
        assert headers == ["x", "2"]
        assert np.allclose(data, [[1.0, 2.0]])

    def test_empty_leading_cell_does_not_make_a_header(self):
        """A body row with a hole is data with a missing value, not a header."""
        headers, data = parse_table(",5\n1,2")
        assert headers == ["col1", "col2"]
        assert data.shape == (2, 2)
        assert np.isnan(data[0, 0])
        assert np.allclose(data[0, 1], 5.0)

    def test_empty_header_cell_gets_generated_name(self):
        """A header row with a blank cell generates a name for that column."""
        headers, _ = parse_table("x,\n1,2")
        assert headers == ["x", "col2"]

    def test_header_only_input_yields_zero_rows(self):
        """A header with no body is a valid, empty table, not an error."""
        headers, data = parse_table("x,y")
        assert headers == ["x", "y"]
        assert data.shape == (0, 2)

    def test_short_header_row_is_padded_with_generated_names(self):
        """A header narrower than the body gets generated names for the extra columns."""
        headers, data = parse_table("x\n1\t2")
        assert headers == ["x", "col2"]
        assert np.allclose(data, [[1.0, 2.0]])


class TestParseTableQuoting:
    """Test that quoted fields are handled by the stdlib csv module."""

    def test_quoted_field_containing_the_delimiter(self):
        """A quoted comma inside a CSV field must not split the column."""
        headers, data = parse_table('name,x\n"a,b",1')
        assert headers == ["name", "x"]
        assert data.shape == (1, 2)
        assert np.isnan(data[0, 0])
        assert np.allclose(data[0, 1], 1.0)

    def test_quoted_header_containing_the_delimiter(self):
        """A quoted comma inside a header cell stays part of the header name."""
        headers, data = parse_table('"first,last",x\n1,2')
        assert headers == ["first,last", "x"]
        assert np.allclose(data, [[1.0, 2.0]])

    def test_quoted_number_still_parses(self):
        """Quoting a numeric cell does not stop it being a number."""
        _, data = parse_table('x,y\n"1.5",2')
        assert np.allclose(data, [[1.5, 2.0]])

    def test_quoted_tab_in_tsv(self):
        """A quoted tab inside a TSV field must not split the column."""
        headers, data = parse_table('name\tx\n"a\tb"\t1')
        assert headers == ["name", "x"]
        assert data.shape == (1, 2)
        assert np.allclose(data[0, 1], 1.0)


class TestParseTableNanHandling:
    """Test that empty and non-numeric body cells become nan."""

    def test_empty_cells_become_nan(self):
        """Holes in the body are nan, not zero."""
        _, data = parse_table("a,b\n1,\n,4")
        assert np.allclose(data[0, 0], 1.0)
        assert np.isnan(data[0, 1])
        assert np.isnan(data[1, 0])
        assert np.allclose(data[1, 1], 4.0)

    def test_non_numeric_body_cells_become_nan(self):
        """Text in the body is nan rather than a parse error."""
        _, data = parse_table("a,b\n1,oops\nn/a,4")
        assert np.isnan(data[0, 1])
        assert np.isnan(data[1, 0])

    def test_whitespace_padded_cells_parse(self):
        """Cells are stripped before being parsed, so ' 1 ' is 1.0."""
        _, data = parse_table("a,b\n 1 , 2 ")
        assert np.allclose(data, [[1.0, 2.0]])

    def test_ragged_rows_padded_with_nan(self):
        """Short rows are padded with nan out to the widest row."""
        headers, data = parse_table("a,b,c\n1,2\n3,4,5")
        assert headers == ["a", "b", "c"]
        assert data.shape == (2, 3)
        assert np.isnan(data[0, 2])
        assert np.allclose(data[1], [3.0, 4.0, 5.0])

    def test_ragged_rows_widest_row_defines_width(self):
        """The widest body row, not the first, defines the column count."""
        _, data = parse_table("1\n2,3,4")
        assert data.shape == (2, 3)
        assert np.allclose(data[0, 0], 1.0)
        assert np.isnan(data[0, 1])
        assert np.isnan(data[0, 2])

    def test_infinity_round_trips_as_a_float(self):
        """'inf' is a legitimate float, not a nan."""
        _, data = parse_table("a\ninf")
        assert np.isinf(data[0, 0])

    def test_trailing_newline_adds_no_row(self):
        """A trailing newline is not an extra all-nan row."""
        _, data = parse_table("x,y\n1,2\n")
        assert data.shape == (1, 2)

    def test_blank_lines_are_dropped(self):
        """Fully empty rows are dropped rather than becoming all-nan rows."""
        _, data = parse_table("x,y\n1,2\n\n3,4")
        assert data.shape == (2, 2)
        assert np.allclose(data, [[1.0, 2.0], [3.0, 4.0]])


class TestParseTableErrors:
    """Test that unparseable input raises ValueError with a human-readable message."""

    def test_empty_string_raises(self):
        """An empty paste is a ValueError, not an empty array."""
        with pytest.raises(ValueError, match="empty"):
            parse_table("")

    def test_whitespace_only_raises(self):
        """A whitespace-only paste is a ValueError."""
        with pytest.raises(ValueError, match="empty"):
            parse_table("   \n\n  \t \n")

    def test_error_message_is_human_readable(self):
        """The message must read as UI text, not as a traceback fragment."""
        with pytest.raises(ValueError) as excinfo:
            parse_table("")
        assert "Nothing to paste" in str(excinfo.value)


class TestFormatTable:
    """Test table -> delimited text formatting."""

    def test_default_is_tsv_with_header(self):
        """The default output is tab-separated with a header and LF terminators."""
        text = format_table(["x", "y"], np.array([[1.0, 2.0]]))
        assert text == "x\ty\n1\t2\n"

    def test_nan_renders_as_empty_cell(self):
        """nan must render as an empty cell so it reads back as nan."""
        text = format_table(["a", "b"], np.array([[1.0, np.nan]]))
        assert text == "a\tb\n1\t\n"

    def test_include_header_false_omits_the_header(self):
        """include_header=False emits the body only."""
        text = format_table(["a", "b"], np.array([[1.0, 2.0]]), include_header=False)
        assert text == "1\t2\n"

    def test_header_count_not_checked_without_header(self):
        """A header/column mismatch is irrelevant when the header is not emitted."""
        text = format_table([], np.array([[1.0, 2.0]]), include_header=False)
        assert text == "1\t2\n"

    def test_custom_delimiter(self):
        """A comma delimiter produces CSV."""
        text = format_table(["a", "b"], np.array([[1.0, 2.0]]), delimiter=",")
        assert text == "a,b\n1,2\n"

    def test_custom_format(self):
        """fmt controls the printf conversion of every non-nan value."""
        text = format_table(["a"], np.array([[1.0 / 3.0]]), fmt="%.3f")
        assert text == "a\n0.333\n"

    def test_one_dimensional_data_is_a_single_column(self):
        """A 1-D array is accepted and treated as one column."""
        text = format_table(["a"], np.array([1.0, 2.0]))
        assert text == "a\n1\n2\n"

    def test_delimiter_in_value_is_quoted(self):
        """csv QUOTE_MINIMAL quotes a header holding the delimiter, preserving the layout."""
        text = format_table(["a,b", "y"], np.array([[1.0, 2.0]]), delimiter=",")
        assert text == '"a,b",y\n1,2\n'

    def test_empty_table_emits_header_only(self):
        """A zero-row table still emits its header."""
        text = format_table(["x", "y"], np.zeros((0, 2)))
        assert text == "x\ty\n"

    def test_multi_character_delimiter_raises(self):
        """The delimiter must be exactly one character."""
        with pytest.raises(ValueError, match="one character"):
            format_table(["a"], np.zeros((1, 1)), delimiter="||")

    def test_header_column_mismatch_raises(self):
        """A header count that disagrees with the column count is an error."""
        with pytest.raises(ValueError, match="mismatch"):
            format_table(["a"], np.zeros((1, 2)))

    def test_three_dimensional_data_raises(self):
        """Only 1-D and 2-D data can be formatted."""
        with pytest.raises(ValueError, match="1-D or 2-D"):
            format_table(["a"], np.zeros((2, 2, 2)))


class TestRoundTrip:
    """Test that format_table -> parse_table preserves the table."""

    def test_tsv_round_trip(self):
        """The default TSV output parses back to the same headers and values."""
        headers = ["time", "value"]
        data = np.array([[0.0, 1.5], [1.0, 2.5], [2.0, -3.25]])
        out_headers, out_data = parse_table(format_table(headers, data))
        assert out_headers == headers
        assert np.allclose(out_data, data)

    def test_csv_round_trip(self):
        """A comma delimiter round-trips too."""
        headers = ["a", "b"]
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        out_headers, out_data = parse_table(format_table(headers, data, delimiter=","))
        assert out_headers == headers
        assert np.allclose(out_data, data)

    def test_nan_round_trips_as_nan(self):
        """An empty cell written for nan reads back as nan, not zero."""
        data = np.array([[1.0, np.nan], [np.nan, 4.0]])
        _, out_data = parse_table(format_table(["a", "b"], data))
        assert np.allclose(out_data, data, equal_nan=True)

    def test_quoted_header_round_trips(self):
        """A header containing the delimiter survives the quoting round trip."""
        headers = ["first,last", "x"]
        data = np.array([[1.0, 2.0]])
        out_headers, out_data = parse_table(format_table(headers, data, delimiter=","))
        assert out_headers == headers
        assert np.allclose(out_data, data)

    def test_round_trip_preserves_precision(self):
        """The default %.10g holds enough digits for the values to compare equal."""
        data = np.array([[1.0 / 3.0, 1e-12], [-2.5e8, 6.02214076e23]])
        _, out_data = parse_table(format_table(["a", "b"], data))
        assert np.allclose(out_data, data, rtol=1e-9)

    def test_headerless_round_trip_generates_names(self):
        """Without a header the values survive; the names are regenerated."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        out_headers, out_data = parse_table(format_table(["a", "b"], data, include_header=False))
        assert out_headers == ["col1", "col2"]
        assert np.allclose(out_data, data)

    def test_numeric_headers_do_not_round_trip(self):
        """Documented limitation: numeric-looking headers read back as a data row."""
        out_headers, out_data = parse_table(format_table(["1", "2"], np.array([[3.0, 4.0]])))
        assert out_headers == ["col1", "col2"]
        assert np.allclose(out_data, [[1.0, 2.0], [3.0, 4.0]])

    def test_parse_format_parse_is_stable(self):
        """Reformatting a parsed table reparses identically (idempotent after one pass)."""
        text = "x\ty\r\n1\t2\r\n3\t\r\n"
        headers, data = parse_table(text)
        headers2, data2 = parse_table(format_table(headers, data))
        assert headers2 == headers
        assert np.allclose(data2, data, equal_nan=True)
