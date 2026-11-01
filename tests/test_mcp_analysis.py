"""``analyze_dataset``: a model can analyse a table and plot what it finds.

The numbers are checked against values known by construction (an exponential with a known time
constant, a sine with a known frequency, an exact quadratic), not against the routines'
own output. No MCP SDK is needed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from glplot.mcp import analysis
from glplot.mcp.state import Limits, McpError
from glplot.mcp.toolbox import Toolbox


@pytest.fixture
def box():
    toolbox = Toolbox()
    toolbox.create_dataset(
        "run",
        {
            "t": "linspace(0, 10, 400)",
            "v": "3*exp(-t/4) + 0.5 + noise(t, 0.02, 1)",
            "s": "sin(2*pi*1.5*t) + 0.4*sin(2*pi*8*t)",
            "line": "2*t + 1",
            "quad": "t**2 - 3*t + 2",
        },
    )
    yield toolbox
    toolbox.close_all()


def values(box, dataset, column):
    return np.asarray(box.datasets.column(dataset, column))


def params(result):
    return {p["name"]: p["value"] for p in result.data["result"]["parameters"]}


class TestFit:
    def test_an_exponential_decay_recovers_its_parameters(self, box):
        result = box.analyze_dataset("run", "fit", x="t", y="v", options={"model": "exp_decay"})
        found = params(result)
        assert found["a"] == pytest.approx(3.0, abs=0.05)
        assert found["tau"] == pytest.approx(4.0, abs=0.1)
        assert found["c"] == pytest.approx(0.5, abs=0.05)
        stats = result.data["result"]["statistics"]
        assert stats["r_squared"] > 0.99 and stats["rmse"] < 0.05
        assert all(
            p["stderr"] is not None and p["stderr"] > 0 for p in result.data["result"]["parameters"]
        )

    def test_the_fitted_curve_becomes_a_column_that_follows_the_data(self, box):
        result = box.analyze_dataset("run", "fit", x="t", y="v", options={"model": "exp_decay"})
        assert result.data["created"] == {"column": "v_fit", "dataset": "run"}
        assert "v_fit" in box.datasets.get("run").column_names()
        residual = values(box, "run", "v") - values(box, "run", "v_fit")
        assert np.std(residual) < 0.05
        assert result.data["plot_hint"] == (
            "add_plot(figure_id, 'line', dataset='run', x='t', y='v_fit')"
        )

    def test_a_linear_fit_is_exact_for_a_line(self, box):
        found = params(box.analyze_dataset("run", "fit", x="t", y="line"))
        assert found["m"] == pytest.approx(2.0) and found["b"] == pytest.approx(1.0)

    def test_a_polynomial_fit_reports_coefficients_highest_power_first(self, box):
        result = box.analyze_dataset(
            "run", "fit", x="t", y="quad", options={"model": "polynomial", "degree": 2}
        )
        assert [p["name"] for p in result.data["result"]["parameters"]] == ["c2", "c1", "c0"]
        found = params(result)
        assert found["c2"] == pytest.approx(1.0) and found["c1"] == pytest.approx(-3.0)
        assert found["c0"] == pytest.approx(2.0)

    def test_a_custom_expression_names_its_own_parameters(self, box):
        result = box.analyze_dataset(
            "run", "fit", x="t", y="v", options={"model": "custom", "expression": "a*exp(-x/b)+c"}
        )
        found = params(result)
        assert set(found) == {"a", "b", "c"} and found["b"] == pytest.approx(4.0, abs=0.1)

    def test_a_custom_expression_cannot_run_code(self, box):
        for expression in ("__import__('os').getcwd()+a", "a*x.sum()", "open('f')+a"):
            with pytest.raises(McpError):
                box.analyze_dataset(
                    "run",
                    "fit",
                    x="t",
                    y="v",
                    options={"model": "custom", "expression": expression},
                )

    def test_custom_without_an_expression_says_what_to_pass(self, box):
        with pytest.raises(McpError, match="options.expression"):
            box.analyze_dataset("run", "fit", x="t", y="v", options={"model": "custom"})

    def test_multipeak_finds_two_peaks(self, box):
        box.create_dataset(
            "two",
            {
                "x": "linspace(-3, 3, 300)",
                "y": "2*exp(-(x-1)**2/0.3) + 1.2*exp(-(x+1.2)**2/0.2)",
            },
        )
        result = box.analyze_dataset(
            "two", "fit", x="x", y="y", options={"model": "multipeak", "peaks": 2}
        )
        centres = sorted([params(result)["mu1"], params(result)["mu2"]])
        assert centres == pytest.approx([-1.2, 1.0], abs=0.05)

    def test_a_robust_fit_resists_an_outlier(self, box):
        box.create_dataset("o", {"x": "linspace(0, 10, 100)", "y": "2*x + 1"})
        table = box.datasets.get("o")
        spiky = np.array(table.get("y"))
        spiky[50] += 200.0
        table.remove_column("y")
        table.add_column("y", spiky)
        plain = params(box.analyze_dataset("o", "fit", x="x", y="y", output="plain"))
        robust = params(
            box.analyze_dataset("o", "fit", x="x", y="y", options={"robust": True}, output="rob")
        )
        assert abs(robust["m"] - 2.0) < abs(plain["m"] - 2.0)
        assert robust["m"] == pytest.approx(2.0, abs=0.05)

    def test_refitting_replaces_the_column_instead_of_piling_up_copies(self, box):
        for _ in range(3):
            box.analyze_dataset("run", "fit", x="t", y="v", options={"model": "exp_decay"})
        names = box.datasets.get("run").column_names()
        assert names.count("v_fit") == 1 and not any("(" in n for n in names)

    def test_a_poor_fit_says_so(self, box):
        box.create_dataset("n", {"x": "linspace(0, 1, 200)", "y": "noise(x, 1.0, 5)"})
        result = box.analyze_dataset("n", "fit", x="x", y="y")
        assert any("Low R^2" in note for note in result.data["notes"])

    def test_too_few_rows_for_the_degree(self, box):
        box.create_dataset("tiny", {"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 4.0]})
        with pytest.raises(McpError, match="more than 3 rows"):
            box.analyze_dataset(
                "tiny", "fit", x="x", y="y", options={"model": "polynomial", "degree": 3}
            )


class TestSignalOperations:
    def test_every_smoothing_method_reduces_noise(self, box):
        box.create_dataset("n", {"x": "linspace(0, 10, 500)", "y": "sin(x) + noise(x, 0.3, 4)"})
        clean = np.sin(values(box, "n", "x"))
        raw = np.std(values(box, "n", "y") - clean)
        for method in ("moving_average", "gaussian", "savgol", "median", "ema"):
            box.analyze_dataset(
                "n", "smooth", y="y", options={"method": method, "window": 21}, output=f"s_{method}"
            )
            smoothed = np.std(values(box, "n", f"s_{method}") - clean)
            assert smoothed < raw, method

    def test_an_even_window_is_explained(self, box):
        result = box.analyze_dataset("run", "smooth", y="v", options={"window": 10})
        assert any("odd" in note for note in result.data["notes"])

    def test_the_derivative_of_a_parabola_is_a_line(self, box):
        box.analyze_dataset("run", "derivative", x="t", y="quad")
        t, d1 = values(box, "run", "t"), values(box, "run", "quad_d1")
        assert d1[10:-10] == pytest.approx(2 * t[10:-10] - 3, abs=0.02)

    def test_the_integral_of_a_line_is_a_parabola_and_the_total_is_reported(self, box):
        result = box.analyze_dataset("run", "integrate", x="t", y="line")
        assert result.data["result"]["total"] == pytest.approx(110.0, rel=1e-4)  # t^2 + t on 0..10
        assert values(box, "run", "line_integral")[-1] == pytest.approx(110.0, rel=1e-4)
        assert values(box, "run", "line_integral")[0] == 0.0

    def test_a_rolling_statistic(self, box):
        box.analyze_dataset("run", "rolling", y="line", options={"window": 5, "stat": "mean"})
        assert values(box, "run", "line_roll_mean")[200] == pytest.approx(
            values(box, "run", "line")[200], rel=1e-6
        )

    def test_the_spectrum_finds_the_strongest_frequency(self, box):
        result = box.analyze_dataset("run", "fft", x="t", y="s")
        assert result.data["result"]["dominant_frequency"] == pytest.approx(1.5, abs=0.15)
        assert result.data["result"]["dominant_amplitude"] == pytest.approx(1.0, abs=0.1)
        assert result.data["created"]["dataset"] == "run_fft"
        table = box.datasets.get("run_fft")
        assert table.column_names() == ["frequency", "amplitude"]

    def test_the_spectrum_warns_when_x_is_not_evenly_spaced(self, box):
        box.create_dataset("u", {"x": "linspace(0, 1, 100)**2", "y": "sin(x)"})
        result = box.analyze_dataset("u", "fft", x="x", y="y")
        assert any("not evenly spaced" in note for note in result.data["notes"])

    def test_peaks_are_found_and_measured(self, box):
        box.create_dataset("p", {"x": "linspace(0, 12.566, 500)", "y": "sin(x)"})
        result = box.analyze_dataset("p", "peaks", x="x", y="y", options={"prominence": 0.5})
        assert result.data["result"]["count"] == 2
        positions = values(box, "p_peaks", "position")
        assert positions == pytest.approx([np.pi / 2, 2.5 * np.pi], abs=0.05)
        assert values(box, "p_peaks", "height") == pytest.approx([1.0, 1.0], abs=0.01)

    def test_no_peaks_is_explained(self, box):
        result = box.analyze_dataset("run", "peaks", x="t", y="line")
        assert result.data["result"]["count"] == 0 and any(
            "No peaks" in n for n in result.data["notes"]
        )

    def test_a_lowpass_filter_removes_the_fast_component(self, box):
        box.analyze_dataset("run", "filter", x="t", y="s", options={"cutoff": 4})
        slow = np.sin(2 * np.pi * 1.5 * values(box, "run", "t"))
        residual = values(box, "run", "s_filtered") - slow
        assert np.std(residual[20:-20]) < 0.1  # the 8 Hz component (amplitude 0.4) is gone

    def test_a_bandpass_filter_keeps_only_its_band(self, box):
        box.analyze_dataset(
            "run", "filter", x="t", y="s", options={"cutoff": [6, 10], "btype": "bandpass"}
        )
        fast = 0.4 * np.sin(2 * np.pi * 8 * values(box, "run", "t"))
        assert np.std((values(box, "run", "s_filtered") - fast)[30:-30]) < 0.1

    def test_filter_arguments_are_checked(self, box):
        with pytest.raises(McpError, match="needs the option 'cutoff'"):
            box.analyze_dataset("run", "filter", x="t", y="s")
        with pytest.raises(McpError, match="needs 2 cutoff"):
            box.analyze_dataset(
                "run", "filter", x="t", y="s", options={"cutoff": 3, "btype": "bandpass"}
            )
        with pytest.raises(McpError, match="Nyquist"):
            box.analyze_dataset("run", "filter", x="t", y="s", options={"cutoff": 500})

    def test_detrend_removes_a_line(self, box):
        box.analyze_dataset("run", "detrend", x="t", y="line")
        assert np.abs(values(box, "run", "line_detrended")).max() < 1e-9

    def test_normalize_modes(self, box):
        box.analyze_dataset("run", "normalize", y="line")
        scaled = values(box, "run", "line_norm")
        assert scaled.min() == pytest.approx(0.0) and scaled.max() == pytest.approx(1.0)
        box.analyze_dataset("run", "normalize", y="v", options={"mode": "zscore"}, output="z")
        z = values(box, "run", "z")
        assert z.mean() == pytest.approx(0.0, abs=1e-9) and z.std() == pytest.approx(1.0, abs=1e-6)

    def test_resample_creates_an_even_grid(self, box):
        result = box.analyze_dataset("run", "resample", x="t", y="line", options={"n": 21})
        table = box.datasets.get(result.data["created"]["dataset"])
        assert table.n_rows() == 21 and table.column_names() == ["t", "line"]
        assert values(box, result.data["created"]["dataset"], "t")[[0, -1]] == pytest.approx(
            [0, 10]
        )
        assert values(box, result.data["created"]["dataset"], "line")[10] == pytest.approx(11.0)

    def test_histogram_counts_every_row(self, box):
        result = box.analyze_dataset("run", "histogram", y="v", options={"bins": 10})
        name = result.data["created"]["dataset"]
        assert values(box, name, "count").sum() == 400
        assert box.datasets.get(name).column_names() == [
            "bin_center",
            "bin_left",
            "bin_right",
            "count",
        ]
        box.analyze_dataset("run", "histogram", y="v", options={"density": True}, output="d")
        width = values(box, "d", "bin_right") - values(box, "d", "bin_left")
        assert (values(box, "d", "density") * width).sum() == pytest.approx(1.0)

    def test_correlation_of_a_perfect_line(self, box):
        result = box.analyze_dataset("run", "correlate", x="t", y="line")
        assert result.data["result"]["pearson_r"] == pytest.approx(1.0)
        assert result.data["result"]["slope"] == pytest.approx(2.0)
        assert "created" not in result.data

    def test_autocorrelation_starts_at_one_and_shows_the_period(self, box):
        box.create_dataset("w", {"x": "linspace(0, 20, 800)", "y": "sin(2*pi*x)"})
        result = box.analyze_dataset("w", "autocorrelation", x="x", y="y", options={"max_lag": 60})
        table = result.data["created"]["dataset"]
        acf = values(box, table, "autocorrelation")
        assert acf[0] == pytest.approx(1.0) and acf[40] > 0.9  # one period = 40 samples
        assert "lag_x" in box.datasets.get(table).column_names()

    def test_describe_reports_statistics(self, box):
        result = box.analyze_dataset("run", "describe", y="line")
        assert result.data["result"]["mean"] == pytest.approx(11.0)
        assert result.data["result"]["n"] == 400.0


class TestInputsAndOutputs:
    def test_y_defaults_to_the_only_other_column(self, box):
        box.create_dataset("two", {"x": "linspace(0, 1, 50)", "y": "x*2"})
        result = box.analyze_dataset("two", "fit", x="x")
        assert result.data["y"] == "y" and result.data["created"]["column"] == "y_fit"

    def test_an_ambiguous_y_asks_which(self, box):
        with pytest.raises(McpError, match="Which column"):
            box.analyze_dataset("run", "smooth")

    def test_the_row_index_is_the_default_abscissa_and_it_is_said(self, box):
        result = box.analyze_dataset("run", "derivative", y="line")
        assert result.data["x"] is None
        assert any("row index" in note for note in result.data["notes"])
        assert "x=" not in result.data["plot_hint"]

    def test_index_can_be_named_explicitly(self, box):
        assert box.analyze_dataset("run", "derivative", x="(index)", y="line").data["x"] is None

    def test_correlate_insists_on_two_real_columns(self, box):
        with pytest.raises(McpError, match="both x and y"):
            box.analyze_dataset("run", "correlate", y="v")

    def test_an_unused_x_is_reported_not_silently_dropped(self, box):
        result = box.analyze_dataset("run", "smooth", x="t", y="v")
        assert any("does not use x" in note for note in result.data["notes"])

    def test_default_names_and_a_chosen_name(self, box):
        assert box.analyze_dataset("run", "smooth", y="v").data["created"]["column"] == "v_smooth"
        chosen = box.analyze_dataset("run", "smooth", y="v", output="clean")
        assert chosen.data["created"]["column"] == "clean"
        assert (
            box.analyze_dataset("run", "fft", x="t", y="s").data["created"]["dataset"] == "run_fft"
        )
        assert (
            box.analyze_dataset("run", "fft", x="t", y="s").data["created"]["dataset"]
            == "run_fft_2"
        )
        again = box.analyze_dataset("run", "fft", x="t", y="s", output="spec")
        assert again.data["created"]["dataset"] == "spec"
        assert box.analyze_dataset("run", "fft", x="t", y="s", output="spec").data["created"][
            "replaced"
        ]

    def test_it_will_not_overwrite_its_own_input(self, box):
        with pytest.raises(McpError, match="overwrite the input column"):
            box.analyze_dataset("run", "smooth", y="v", output="v")
        with pytest.raises(McpError, match="overwrite the input column"):
            box.analyze_dataset("run", "detrend", x="t", y="v", output="t")

    def test_output_names_must_be_usable(self, box):
        with pytest.raises(McpError, match="column name"):
            box.analyze_dataset("run", "smooth", y="v", output="two words")
        with pytest.raises(McpError, match="dataset name"):
            box.analyze_dataset("run", "fft", x="t", y="s", output="bad name!")

    def test_a_column_with_an_awkward_name_gets_a_usable_result_name(self, box):
        box.create_dataset("f", {"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]})
        table = box.datasets.get("f")
        table.rename_column("y", "Temperature (K)")
        result = box.analyze_dataset("f", "smooth", y="Temperature (K)", options={"window": 3})
        assert result.data["created"]["column"] == "Temperature_K_smooth"

    def test_the_analysis_is_in_the_dataset_for_later_expressions(self, box):
        box.analyze_dataset("run", "fit", x="t", y="v", options={"model": "exp_decay"})
        box.create_dataset("resid", {"r": "v - v_fit"}, base="run")
        assert np.std(values(box, "resid", "r")) < 0.05

    def test_errors_name_the_problem_and_the_alternatives(self, box):
        with pytest.raises(McpError, match="Did you mean 'smooth'"):
            box.analyze_dataset("run", "smoth", y="v")
        with pytest.raises(McpError, match="Available:.*fit"):
            box.analyze_dataset("run", "nope", y="v")
        with pytest.raises(McpError, match="No dataset named 'rn'.*run"):
            box.analyze_dataset("rn", "fit")
        with pytest.raises(McpError, match="available|columns"):
            box.analyze_dataset("run", "fit", x="tt", y="v")

    def test_size_limits(self):
        small = Toolbox(limits=Limits(max_analysis_rows=50))
        try:
            small.create_dataset("big", {"x": "linspace(0, 1, 100)", "y": "x"})
            with pytest.raises(McpError, match="at most 50"):
                small.analyze_dataset("big", "fit", x="x", y="y")
            small.create_dataset("ok", {"x": "linspace(0, 1, 40)", "y": "x"})
            with pytest.raises(McpError, match="at most 50"):
                small.analyze_dataset("ok", "resample", x="x", y="y", options={"n": 51})
            small.create_dataset("one", {"x": [1.0], "y": [2.0]})
            with pytest.raises(McpError, match="at least 2"):
                small.analyze_dataset("one", "describe", y="y")
        finally:
            small.close_all()

    def test_a_result_can_be_plotted_from_its_hint(self, box):
        box.create_figure()
        box.add_plot("fig1", "scatter", dataset="run", x="t", y="v", label="data")
        hint = box.analyze_dataset("run", "fit", x="t", y="v", options={"model": "exp_decay"})
        box.add_plot("fig1", "line", dataset="run", x="t", y="v_fit", label="fit")
        box.set_axes("fig1", xlabel="t", ylabel="v", legend=True)
        assert hint.data["plot_hint"].startswith("add_plot(figure_id, 'line'")
        assert box.render_figure("fig1").image.startswith(b"\x89PNG")

    @pytest.mark.parametrize(
        ("operation", "kwargs"),
        [
            ("fft", {"x": "t", "y": "s"}),
            ("peaks", {"x": "t", "y": "s", "options": {"prominence": 0.5}}),
            ("histogram", {"y": "v"}),
            ("resample", {"x": "t", "y": "v", "options": {"n": 40}}),
            ("autocorrelation", {"y": "s", "options": {"max_lag": 50}}),
        ],
    )
    def test_every_table_result_draws_with_its_own_hint(self, box, operation, kwargs):
        result = box.analyze_dataset("run", operation, **kwargs)
        created = result.data["created"]["dataset"]
        px, py = (
            result.data["plot_hint"].split("x='")[1].split("'")[0],
            result.data["plot_hint"].split("y='")[1].split("'")[0],
        )
        kind = result.data["plot_hint"].split("figure_id, '")[1].split("'")[0]
        box.create_figure()
        box.add_plot("fig1", kind, dataset=created, x=px, y=py)
        assert box.render_figure("fig1").image.startswith(b"\x89PNG")

    def test_the_reply_is_valid_json_after_its_sentence(self, box):
        text = box.analyze_dataset(
            "run", "fit", x="t", y="v", options={"model": "exp_decay"}
        ).text()
        sentence, _, payload = text.partition("\n")
        assert sentence.startswith("fit of 'v':") and "R^2" in sentence
        assert json.loads(payload)["operation"] == "fit"


class TestOptionValidation:
    def test_unknown_options_are_refused_with_the_valid_ones(self, box):
        with pytest.raises(
            McpError, match="no option.*'windw'.*Did you mean 'window'.*Valid options"
        ):
            box.analyze_dataset("run", "smooth", y="v", options={"windw": 5})

    def test_values_are_type_and_range_checked(self, box):
        with pytest.raises(McpError, match="must be at least 1"):
            box.analyze_dataset("run", "smooth", y="v", options={"window": 0})
        with pytest.raises(McpError, match="whole number"):
            box.analyze_dataset("run", "smooth", y="v", options={"window": 5.5})
        with pytest.raises(McpError, match="must be a number"):
            box.analyze_dataset("run", "smooth", y="v", options={"window": "wide"})
        with pytest.raises(McpError, match="true or false"):
            box.analyze_dataset("run", "fit", x="t", y="v", options={"robust": "maybe"})

    def test_a_choice_names_the_valid_ones_and_the_near_miss(self, box):
        with pytest.raises(McpError, match="one of: moving_average.*Did you mean 'savgol'"):
            box.analyze_dataset("run", "smooth", y="v", options={"method": "savgl"})

    def test_loosely_typed_values_are_accepted(self, box):
        result = box.analyze_dataset(
            "run", "smooth", y="v", options={"window": "15", "method": "SAVGOL"}
        )
        assert result.data["created"]["column"] == "v_smooth"

    def test_a_number_alone_is_a_one_element_list(self, box):
        assert box.analyze_dataset("run", "filter", x="t", y="s", options={"cutoff": 3}).data

    def test_options_must_be_an_object(self, box):
        with pytest.raises(McpError):
            box.analyze_dataset("run", "smooth", y="v", options=[1, 2])  # type: ignore[arg-type]


class TestRegistryAndGuide:
    def test_every_operation_is_described_and_its_example_names_it(self):
        assert len(analysis.OPERATIONS) >= 15
        for op in analysis.OPERATIONS:
            assert len(op.summary) > 20 and op.example, op.name
            assert f'"{op.name}"' in op.example, op.name
            assert op.produces in ("column", "table", "report")
            for option in op.options:
                assert option.description, (op.name, option.name)
                if option.kind == "choice":
                    assert option.default in option.choices or option.default is None

    def test_the_guide_lists_every_operation_and_option(self, box):
        text = box.get_guide("mcp-analysis").message
        for op in analysis.OPERATIONS:
            assert f"### {op.name}" in text, op.name
            for option in op.options:
                assert f"`{option.name}`" in text, (op.name, option.name)

    def test_the_tool_description_lists_the_operations(self):
        from glplot.mcp.schema import load_tools

        tool = next(t for t in load_tools() if t["name"] == "analyze_dataset")
        for op in analysis.OPERATIONS:
            assert op.name in tool["description"], op.name
        assert tool["inputSchema"]["required"] == ["dataset", "operation"]
        assert tool["annotations"]["readOnlyHint"] is False

    def test_the_examples_in_the_guide_run(self, box):
        """Each ``example`` is a real call; run the ones that fit the sample table."""
        samples = {
            "describe": {"y": "v"},
            "fit": {"x": "t", "y": "v", "options": {"model": "exp_decay"}},
            "smooth": {"y": "v", "options": {"method": "savgol", "window": 31}},
            "rolling": {"y": "v", "options": {"window": 25, "stat": "std"}},
            "derivative": {"x": "t", "y": "v"},
            "integrate": {"x": "t", "y": "v"},
            "fft": {"x": "t", "y": "s"},
            "filter": {"x": "t", "y": "s", "options": {"cutoff": 5}},
            "detrend": {"x": "t", "y": "v"},
            "normalize": {"y": "v", "options": {"mode": "zscore"}},
            "resample": {"x": "t", "y": "v", "options": {"n": 1000}},
            "histogram": {"y": "v", "options": {"bins": 30}},
            "correlate": {"x": "t", "y": "v"},
            "autocorrelation": {"y": "s", "options": {"max_lag": 200}},
        }
        for name, kwargs in samples.items():
            assert box.analyze_dataset("run", name, **kwargs).data["operation"] == name

    def test_the_prompt_sends_the_model_to_fit_and_plot(self):
        from glplot.mcp import catalog

        text = catalog.find_prompt("fit_and_plot").build(dataset="run", x="t", y="v")
        assert "analyze_dataset" in text and "'fit'" in text and "R^2" in text
        assert "exp_decay" in catalog.find_prompt("fit_and_plot").build(
            dataset="d", x="a", y="b", model="exp_decay"
        )

    def test_the_connect_instructions_mention_analysis(self):
        from glplot.mcp.guide import build_instructions

        assert "analyze_dataset" in build_instructions()
