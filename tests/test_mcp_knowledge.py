"""The knowledge side of the MCP surface: tools, resources, prompts and the guide topics.

No MCP SDK is needed: these call the toolbox and the catalog directly.
"""

from __future__ import annotations

import json

import pytest

from glplot.mcp import catalog
from glplot.mcp.guide import build_guide, build_instructions, mcp_topics
from glplot.mcp.state import FileSandbox, McpError
from glplot.mcp.toolbox import Toolbox, ToolResult


@pytest.fixture
def box(tmp_path):
    toolbox = Toolbox(FileSandbox([tmp_path]))
    yield toolbox
    toolbox.close_all()


@pytest.fixture
def closed_box():
    toolbox = Toolbox()
    yield toolbox
    toolbox.close_all()


class TestGetGuide:
    def test_without_a_topic_it_lists_both_the_python_and_the_mcp_topics(self, box):
        result = box.get_guide()
        for topic in ("overview", "quickstart", "animation", "mcp", "mcp-workflow", "mcp-data"):
            assert f"`{topic}`" in result.message, topic
        assert "mcp-recipes" in result.data["topics"] and "quickstart" in result.data["topics"]

    def test_a_python_topic_is_markdown(self, box):
        text = box.get_guide("animation").message
        assert text.startswith("# ") and "glplot.animation" in text and "{{" not in text

    def test_an_mcp_topic_reflects_how_the_server_was_started(self, box, closed_box, tmp_path):
        assert str(tmp_path.resolve()) in box.get_guide("mcp-data").message
        assert "disabled" in closed_box.get_guide("mcp-data").message

    def test_the_mcp_topics_are_the_guide_split_up(self, box):
        whole = build_guide(box.sandbox, box.limits)
        for topic in mcp_topics(box.sandbox, box.limits):
            first_line = topic.body.strip().splitlines()[0]
            assert first_line in whole, topic.id
        assert [t.id for t in mcp_topics()] == [
            "mcp-workflow",
            "mcp-data",
            "mcp-analysis",
            "mcp-plot-kinds",
            "mcp-recipes",
            "mcp-animation",
            "mcp-interactive",
            "mcp-styling",
            "mcp-notes",
        ]

    def test_an_unknown_topic_is_a_readable_error_with_a_suggestion(self, box):
        with pytest.raises(McpError, match="quickstart"):
            box.get_guide("quikstart")

    def test_the_guide_points_at_the_knowledge_tools(self, box):
        guide = build_guide(box.sandbox, box.limits)
        for name in ("get_guide", "search_api", "get_api_doc", "get_example", "server_info"):
            assert name in guide, name

    def test_the_connect_instructions_do_too(self):
        text = build_instructions()
        for name in ("get_guide", "search_api", "get_api_doc", "get_example", "server_info"):
            assert name in text, name
        assert "plt.savefig" in text and "plt.show" in text


class TestSearchApi:
    def test_a_query_returns_call_shapes_and_structured_matches(self, box):
        result = box.search_api("heatmap", limit=3)
        assert "imshow(" in result.message and len(result.data["matches"]) == 3
        first = result.data["matches"][0]
        assert first["name"] == "imshow" and first["usage"].startswith("plt.imshow(")

    def test_no_query_lists_the_categories(self, box):
        result = box.search_api()
        assert "Line, scatter, bar and statistical plots" in result.message
        assert {c["id"] for c in result.data["categories"]} >= {"plot2d", "plot3d", "axes"}

    def test_category_filter(self, box):
        result = box.search_api("plot", limit=10, category="plot3d")
        assert result.data["matches"] and all(
            m["category"] == "plot3d" for m in result.data["matches"]
        )

    def test_bad_arguments_are_readable(self, box):
        with pytest.raises(McpError, match="limit must be between 1 and 40"):
            box.search_api("x", limit=0)
        with pytest.raises(McpError, match="plot3d"):
            box.search_api("x", category="plot3")

    def test_no_match_says_what_to_try(self, box):
        result = box.search_api("xyzzyq")
        assert result.data == {"matches": []} and "No function matches" in result.message


class TestGetApiDoc:
    def test_it_returns_the_real_signature_and_docstring(self, box):
        result = box.get_api_doc("plt.twinx")
        assert result.message.startswith("plt.twinx(") and "second y axis" in result.message.lower()
        assert result.data["category"] == "axes" and result.data["signature"].startswith("(")

    def test_unknown_names_suggest_the_right_one(self, box):
        with pytest.raises(McpError, match="scatter"):
            box.get_api_doc("scater")

    def test_a_class_outside_pyplot(self, box):
        assert (
            "from glplot.animation import FuncAnimation" in box.get_api_doc("FuncAnimation").message
        )


class TestGetExample:
    def test_listing_and_fetching(self, box):
        listing = box.get_example()
        assert "line_plot" in listing.message and "line_plot" in listing.data["examples"]
        one = box.get_example("surface_3d")
        assert "```python" in one.message and "plt.plot_surface" in one.message
        assert one.data["filename"] == "surface_3d.py"

    def test_unknown_example(self, box):
        with pytest.raises(McpError, match="line_plot"):
            box.get_example("line_plt")


class TestServerInfo:
    def test_it_reports_versions_limits_and_file_access(self, box, tmp_path):
        import glplot

        info = box.server_info().data
        assert info["glplot"] == glplot.__version__
        assert info["server"]["implementation"] == "unknown"  # set by the server that wraps it
        assert info["server"]["files"] == {"enabled": True, "roots": [str(tmp_path.resolve())]}
        assert "line" in info["server"]["plot_kinds"] and "surface" in info["server"]["field_kinds"]
        assert info["server"]["limits"]["max_figures"] == box.limits.max_figures
        assert info["api"]["functions"] > 200 and info["api"]["recipes"] == 22
        assert {"fit", "fft", "smooth"} <= set(info["server"]["analysis_operations"])
        assert info["server"]["limits"]["max_analysis_rows"] == box.limits.max_analysis_rows

    def test_without_a_root_it_says_files_are_off(self, closed_box):
        files = closed_box.server_info().data["server"]["files"]
        assert files["enabled"] is False and "no file" in files["note"]

    def test_it_lists_what_is_open_now(self, box):
        box.create_figure()
        box.create_dataset("d", {"x": [1.0, 2.0]})
        server = box.server_info().data["server"]
        assert server["open_figures"] == ["fig1"] and server["datasets"] == ["d"]


class TestReplyText:
    def test_non_ascii_text_is_kept_as_written(self):
        text = ToolResult(
            "ok", {"label": "T (°C)", "unit": "µm", "name": "Ångström", "kanji": "温度"}
        ).text()
        assert "T (°C)" in text and "µm" in text and "Ångström" in text and "温度" in text
        assert "\\u" not in text
        assert json.loads(text.split("\n", 1)[1])["unit"] == "µm"

    def test_a_title_with_symbols_round_trips_through_describe(self, box):
        box.create_figure(title="Rate (°C/min), µ=0.1")
        assert "µ=0.1" in box.describe_figure("fig1").text()


class TestCatalog:
    @pytest.mark.parametrize(
        ("uri", "needle"),
        [
            ("glplot://guide", "GLPlot MCP guide"),
            ("glplot://guide/three-d", "3-D"),
            ("glplot://api", "# API reference"),
            ("glplot://api/hist2d", "hist2d"),
            ("glplot://examples", "line_plot"),
            ("glplot://examples/line_plot", "plt.savefig"),
            ("glplot://llms.txt", "# GLPlot"),
            ("glplot://llms-full.txt", "mcp-data"),
            ("glplot://capabilities", "style_presets"),
            ("glplot://recipes", "signal_trace"),
            ("glplot://styles", "journal"),
        ],
    )
    def test_every_text_resource_reads(self, box, uri, needle):
        definition, params = catalog.find_resource(uri)
        body = definition.read(box, params)
        assert isinstance(body, str) and needle in body

    def test_templates_are_matched_with_their_parameters(self):
        definition, params = catalog.find_resource("glplot://figures/fig3/image")
        assert definition.name == "figure-image" and params == {"figure_id": "fig3"}
        assert catalog.find_resource("glplot://api/hist2d")[1] == {"name": "hist2d"}
        assert catalog.find_resource("glplot://api")[0].is_template is False

    @pytest.mark.parametrize(
        "uri", ["glplot://nothing", "glplot://api/a/b", "http://x", "glplot://"]
    )
    def test_unknown_uris_do_not_match(self, uri):
        assert catalog.find_resource(uri) is None

    def test_every_resource_has_a_description_and_a_mime_type(self):
        for definition in catalog.RESOURCES:
            assert len(definition.description) > 15 and "/" in definition.mime_type, definition.uri
        assert len({d.uri for d in catalog.RESOURCES}) == len(catalog.RESOURCES)

    def test_the_image_and_dataset_resources_serve_real_data(self, box):
        box.create_figure()
        box.add_plot("fig1", "line", y=[1.0, 3.0, 2.0])
        box.create_dataset("d", {"x": [1.0, 2.0, 3.0]})
        image = catalog.find_resource("glplot://figures/fig1/image")
        assert image[0].read(box, image[1]).startswith(b"\x89PNG")
        table = catalog.find_resource("glplot://datasets/d")
        assert json.loads(table[0].read(box, table[1]))["name"] == "d"
        with pytest.raises(McpError, match="No figure"):
            image[0].read(box, {"figure_id": "fig9"})

    def test_a_missing_resource_name_is_a_readable_error(self, box):
        definition, params = catalog.find_resource("glplot://api/scater")
        with pytest.raises(McpError, match="scatter"):
            definition.read(box, params)

    def test_prompts_validate_their_arguments(self):
        quick = catalog.find_prompt("quick_look")
        assert "'run'" in quick.build(dataset="run")
        assert "which is slow" in quick.build(dataset="run", question="which is slow")
        with pytest.raises(McpError, match="needs: dataset"):
            quick.build()
        with pytest.raises(McpError, match="no argument"):
            quick.build(dataset="d", colour="red")
        with pytest.raises(McpError, match="Unknown prompt"):
            catalog.find_prompt("nope")

    def test_every_prompt_builds_with_only_its_required_arguments(self):
        for prompt in catalog.PROMPTS:
            required = {a.name: "x" for a in prompt.arguments if a.required}
            text = prompt.build(**required)
            assert len(text) > 80 and prompt.description, prompt.name
            assert all(a.description for a in prompt.arguments), prompt.name

    def test_the_script_prompt_sends_the_model_to_the_docs_before_it_writes_code(self):
        text = catalog.find_prompt("python_script_for").build(task="an animated sine wave")
        assert text.index("get_guide") < text.index("Give the script")
        assert "an animated sine wave" in text


class TestToolDescriptionExamples:
    """The example calls written into the tool descriptions must actually work."""

    def test_the_examples_run_as_written(self, closed_box):
        box = closed_box
        box.create_dataset("run", {"t": "linspace(0, 10, 500)", "v": "sin(t)*exp(-t/5)"})
        box.create_figure()
        box.add_plot("fig1", "scatter", x=[1, 2, 3], y=[2, 4, 9], color="red")
        box.add_plot("fig1", "line", dataset="run", x="t", y="v", label="run")
        box.add_plot("fig1", "hist", dataset="run", y="v", bins=30)
        box.add_function("fig1", "sin(a*x)/x", x_min=-10, x_max=10, parameters={"a": 2})
        box.set_axes(
            "fig1", title="Decay", xlabel="t (s)", ylabel="signal (V)", yscale="log", legend=True
        )
        assert box.render_figure("fig1").image.startswith(b"\x89PNG")

    def test_the_descriptions_carry_them(self):
        from glplot.mcp.schema import load_tools

        tools = {t["name"]: t["description"] for t in load_tools()}
        for name in ("add_plot", "add_function", "set_axes", "create_dataset"):
            assert "Example" in tools[name], name
