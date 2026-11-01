"""The figure review that ``render_figure`` returns: provable problems, each with its fix.

Unit tests build small matplotlib figures by hand (so each check is pinned in isolation); the
integration tests go through the toolbox the way a model does, and check that following a suggested
fix really makes the finding go away. No MCP SDK is needed.
"""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as mpl  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from glplot.mcp.review import MAX_ISSUES, review_figure  # noqa: E402
from glplot.mcp.toolbox import Toolbox  # noqa: E402


def codes(issues):
    return [item["code"] for item in issues]


@pytest.fixture
def fig():
    figure = mpl.figure(figsize=(6, 4))
    yield figure
    mpl.close(figure)


def clean_axes(fig, *position, **kwargs):
    ax = fig.add_subplot(*position, **kwargs)
    ax.plot([0, 1, 2, 3], [0, 1, 4, 9], label="a")
    ax.set(xlabel="x (s)", ylabel="y (V)", title="t")
    return ax


class TestChecks:
    def test_a_complete_figure_has_nothing_to_report(self, fig):
        clean_axes(fig)
        assert review_figure(fig) == []

    def test_missing_labels_and_title(self, fig):
        ax = fig.add_subplot()
        ax.plot([0, 1, 2], [0, 1, 4])
        found = review_figure(fig)
        assert codes(found) == ["no_xlabel", "no_ylabel", "no_title"]
        assert [i["severity"] for i in found] == ["warning", "warning", "suggestion"]
        assert all("set_axes(" in i["fix"] for i in found)

    def test_an_image_or_hidden_axes_need_no_axis_labels(self, fig):
        ax = fig.add_subplot()
        ax.imshow(np.zeros((4, 4)))
        ax.set_title("t")
        assert "no_xlabel" not in codes(review_figure(fig))
        fig2 = mpl.figure()
        pie = fig2.add_subplot()
        pie.pie([1, 2, 3])
        pie.axis("off")
        pie.set_title("t")
        assert review_figure(fig2) == []
        mpl.close(fig2)

    def test_several_labelled_series_without_a_legend(self, fig):
        ax = clean_axes(fig)
        ax.plot([0, 1], [1, 2], label="b")
        assert "legend_missing" in codes(review_figure(fig))
        ax.legend()
        assert "legend_missing" not in codes(review_figure(fig))

    def test_labels_known_to_the_caller_count_even_if_the_figure_has_none_yet(self, fig):
        clean_axes(fig)
        assert "legend_missing" in codes(review_figure(fig, labelled={1: ["a", "b"]}))
        assert "legend_missing" not in codes(review_figure(fig, labelled={1: ["a"]}))

    def test_a_crowded_legend(self, fig):
        ax = clean_axes(fig)
        for k in range(9):
            ax.plot([0, 1], [k, k + 1], label=f"s{k}")
        ax.legend()
        assert "crowded_legend" in codes(review_figure(fig))

    def test_an_empty_panel(self, fig):
        clean_axes(fig, 1, 2, 1)
        fig.add_subplot(1, 2, 2)
        found = review_figure(fig)
        assert [(i["code"], i["panel"]) for i in found if i["code"] == "empty_panel"] == [
            ("empty_panel", 2)
        ]
        assert "add_plot(..., panel=2)" in found[0]["fix"]

    def test_a_title_missing_everywhere_is_one_finding_about_the_figure(self, fig):
        for index in (1, 2):
            ax = fig.add_subplot(1, 2, index)
            ax.plot([0, 1], [0, 1])
            ax.set(xlabel="x", ylabel="y")
        titles = [i for i in review_figure(fig) if i["code"] == "no_title"]
        assert len(titles) == 1 and titles[0]["panel"] == 0
        fig.suptitle("Overall")
        assert "no_title" not in codes(review_figure(fig))

    def test_points_a_log_axis_cannot_draw(self, fig):
        ax = clean_axes(fig)
        ax.cla()
        ax.set(xlabel="x", ylabel="y", title="t")
        ax.set_yscale("log")
        ax.plot([1, 2, 3, 4], [-1, 0, 5, 10])
        found = review_figure(fig)
        issue = next(i for i in found if i["code"] == "log_y_nonpositive")
        assert issue["severity"] == "warning" and "2 point(s)" in issue["message"]
        assert "data_outside_ylim" not in codes(found)  # said once, not twice

    def test_data_cut_off_by_the_limits_is_info_with_the_full_range_in_the_fix(self, fig):
        ax = clean_axes(fig)
        ax.set_xlim(0, 1)
        issue = next(i for i in review_figure(fig) if i["code"] == "data_outside_xlim")
        assert issue["severity"] == "info" and "0..3" in issue["message"]
        assert "xlim=[0, 3]" in issue["fix"]

    def test_a_view_that_shows_everything_is_not_flagged(self, fig):
        ax = clean_axes(fig)
        ax.set_xlim(-1, 4)
        assert "data_outside_xlim" not in codes(review_figure(fig))

    def test_constant_series_and_missing_values_are_info(self, fig):
        ax = clean_axes(fig)
        ax.plot([0, 1, 2, 3], [5, 5, 5, 5], label="flat")
        ax.plot([0, 1, 2, 3], [1, np.nan, 3, 4], label="holes")
        ax.legend()
        found = {i["code"]: i for i in review_figure(fig)}
        assert (
            found["constant_series"]["severity"] == "info"
            and "'flat'" in found["constant_series"]["message"]
        )
        assert "1 missing" in found["missing_values"]["message"]

    def test_a_two_point_line_is_not_called_constant(self, fig):
        ax = clean_axes(fig)
        ax.plot([0, 3], [2, 2])  # a reference line
        assert "constant_series" not in codes(review_figure(fig))

    def test_overlapping_tick_labels(self, fig):
        ax = fig.add_subplot()
        names = [f"a rather long category name number {k}" for k in range(6)]
        ax.bar(range(6), range(1, 7))
        ax.set_xticks(range(6), names)
        ax.set(xlabel="c", ylabel="n", title="t")
        issue = next(i for i in review_figure(fig) if i["code"] == "x_tick_labels_overlap")
        assert issue["severity"] == "warning" and "barh" in issue["fix"]

    def test_short_tick_labels_do_not_overlap(self, fig):
        ax = fig.add_subplot()
        ax.bar(range(4), range(1, 5))
        ax.set_xticks(range(4), list("abcd"))
        ax.set(xlabel="c", ylabel="n", title="t")
        assert review_figure(fig) == []

    def test_the_tick_check_can_be_switched_off_and_is_skipped_for_huge_figures(self, fig):
        ax = fig.add_subplot()
        ax.bar(range(6), range(1, 7))
        ax.set_xticks(range(6), [f"a rather long category name number {k}" for k in range(6)])
        ax.set(xlabel="c", ylabel="n", title="t")
        assert "x_tick_labels_overlap" not in codes(review_figure(fig, measure_text=False))
        big = mpl.figure()
        big_ax = big.add_subplot()
        big_ax.scatter(np.arange(60_000), np.arange(60_000))
        big_ax.set(xlabel="x", ylabel="y", title="t")
        assert review_figure(big) == []  # no draw was needed to say so
        mpl.close(big)

    def test_three_d_axes_get_only_the_basic_checks(self, fig):
        ax = fig.add_subplot(projection="3d")
        ax.plot([0, 1], [0, 1], [0, 1])
        found = codes(review_figure(fig))
        assert "no_title" in found and "log_y_nonpositive" not in found

    def test_findings_are_capped_and_worst_first(self, fig):
        for index in range(1, 10):
            fig.add_subplot(3, 3, index)  # nine empty panels, nine warnings
        found = review_figure(fig)
        assert len(found) == MAX_ISSUES
        severities = [i["severity"] for i in found]
        assert severities == sorted(severities, key={"warning": 0, "suggestion": 1, "info": 2}.get)

    def test_every_finding_has_the_same_shape(self, fig):
        fig.add_subplot().plot([0, 1], [0, 1])
        for item in review_figure(fig):
            assert set(item) >= {"panel", "severity", "code", "message"}
            assert item["severity"] in ("warning", "suggestion", "info")


class TestThroughTheToolbox:
    @pytest.fixture
    def box(self):
        toolbox = Toolbox()
        toolbox.create_dataset(
            "d", {"x": "linspace(0, 10, 100)", "a": "sin(x)", "b": "cos(x)", "z": "x - 5"}
        )
        yield toolbox
        toolbox.close_all()

    def test_a_finished_figure_reports_no_issues(self, box):
        box.create_figure(title="Waves")
        box.add_plot("fig1", "line", dataset="d", x="x", y="a", label="sin")
        box.set_axes("fig1", xlabel="t (s)", ylabel="amplitude (V)")
        result = box.render_figure("fig1")
        assert result.data["issues"] == [] and "No issues found" in result.message

    def test_a_bare_figure_lists_what_to_fix_and_the_message_says_how_many(self, box):
        box.create_figure()
        box.add_plot("fig1", "line", dataset="d", x="x", y="a")
        result = box.render_figure("fig1")
        assert set(codes(result.data["issues"])) == {"no_xlabel", "no_ylabel", "no_title"}
        assert "3 things to consider (2 warnings)" in result.message

    def test_following_each_pure_call_fix_clears_its_finding(self, box):
        box.create_figure()
        box.add_plot("fig1", "line", dataset="d", x="x", y="a")
        for issue in box.render_figure("fig1").data["issues"]:
            match = re.fullmatch(r"(set_axes\(.*\))\.", issue["fix"])
            assert match, issue
            eval(match.group(1), {"set_axes": box.set_axes, "figure_id": "fig1"})
        assert box.render_figure("fig1").data["issues"] == []

    def test_the_log_axis_warning_through_the_tools(self, box):
        box.create_figure()
        box.add_plot("fig1", "line", dataset="d", x="x", y="z")
        box.set_axes("fig1", title="t", xlabel="x", ylabel="y", yscale="log")
        found = box.render_figure("fig1").data["issues"]
        assert "log_y_nonpositive" in codes(found)

    def test_a_panel_left_empty_is_found(self, box):
        box.create_figure(rows=1, cols=2, title="Two")
        box.add_plot("fig1", "line", dataset="d", x="x", y="a", panel=1)
        box.set_axes("fig1", panel=1, xlabel="x", ylabel="y")
        found = box.render_figure("fig1").data["issues"]
        assert [(i["code"], i["panel"]) for i in found] == [("empty_panel", 2)]

    def test_saving_and_describing_do_not_pay_for_a_review(self, box, tmp_path, monkeypatch):
        from glplot.mcp import toolbox as toolbox_module

        calls = []
        monkeypatch.setattr(toolbox_module, "review_figure", lambda *a, **k: calls.append(1) or [])
        box.create_figure()
        box.add_plot("fig1", "line", dataset="d", x="x", y="a")
        box.describe_figure("fig1")
        box.set_axes("fig1", title="t")
        assert calls == []
        box.render_figure("fig1")
        assert calls == [1]

    def test_the_reply_json_carries_the_issues(self, box):
        import json

        box.create_figure()
        box.add_plot("fig1", "line", dataset="d", x="x", y="a")
        payload = json.loads(box.render_figure("fig1").text().split("\n", 1)[1])
        assert payload["issues"][0]["severity"] == "warning"

    def test_the_tool_description_and_guide_explain_the_issues(self, box):
        from glplot.mcp.schema import load_tools

        tool = next(t for t in load_tools() if t["name"] == "render_figure")
        assert "issues" in tool["description"]
        assert "`issues`" in box.get_guide("mcp-workflow").message
