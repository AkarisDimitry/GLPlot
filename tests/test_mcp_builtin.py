"""The built-in, dependency-free MCP server (``glplot.mcp.builtin``) and what it shares.

This is the server a client gets after a plain ``pip install glplot``, so nearly everything here
runs *without* the optional ``mcp`` package -- that is the point. The tests that need the SDK
(driving the built-in server with the official client, and checking the shipped tool schemas
against the SDK server's) skip cleanly without it.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from glplot.mcp import builtin, catalog
from glplot.mcp.builtin import BuiltinServer, serve_stdio, validate_arguments
from glplot.mcp.schema import load_tools
from glplot.mcp.state import FileSandbox, McpError
from glplot.mcp.toolbox import Toolbox, ToolResult

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS = {tool["name"]: tool for tool in load_tools()}


@pytest.fixture
def server(tmp_path):
    box = Toolbox(FileSandbox([tmp_path]))
    yield BuiltinServer(box)
    box.close_all()


def rpc(server, method, params=None, rid=1):
    reply = server.handle({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
    assert reply["id"] == rid
    return reply


def result(server, method, params=None):
    reply = rpc(server, method, params)
    assert "error" not in reply, reply
    return reply["result"]


def call(server, tool, **arguments):
    return result(server, "tools/call", {"name": tool, "arguments": arguments})


def text_of(reply):
    return "".join(block.get("text", "") for block in reply["content"])


# ----------------------------------------------------------------------------------------
# The shipped tool schemas
# ----------------------------------------------------------------------------------------


class TestToolSchemas:
    def test_the_snapshot_lists_every_tool_with_a_real_manual(self):
        assert len(TOOLS) >= 23
        for name in (
            "create_figure",
            "add_plot",
            "render_figure",
            "get_guide",
            "search_api",
            "get_api_doc",
            "get_example",
            "server_info",
            "add_recipe",
        ):
            assert name in TOOLS, name
        for name, tool in TOOLS.items():
            assert len(tool["description"]) > 20, name
            assert tool["inputSchema"]["type"] == "object", name
            for argument, spec in tool["inputSchema"]["properties"].items():
                assert "description" in spec, f"{name}.{argument} has no description"

    def test_an_argument_called_title_survives(self):
        """Regression: normalising schemas once deleted every property named 'title'."""
        for name in ("create_figure", "set_axes", "add_recipe"):
            assert "title" in TOOLS[name]["inputSchema"]["properties"], name

    def test_every_schema_argument_is_a_parameter_of_its_toolbox_method(self):
        """The built-in server calls ``method(**arguments)``, so the names must line up."""
        for name, tool in TOOLS.items():
            params = set(inspect.signature(getattr(Toolbox, name)).parameters) - {"self"}
            declared = set(tool["inputSchema"]["properties"])
            assert declared <= params, (name, declared - params)
            for required in tool["inputSchema"].get("required", []):
                assert required in params, (name, required)

    def test_safety_hints_match_the_toolbox_intent(self):
        read_only = {n for n, t in TOOLS.items() if t["annotations"]["readOnlyHint"]}
        assert {
            "render_figure",
            "describe_figure",
            "get_guide",
            "search_api",
            "get_api_doc",
            "get_example",
            "server_info",
            "list_recipes",
        } <= read_only
        destructive = {n for n, t in TOOLS.items() if t["annotations"]["destructiveHint"]}
        assert {"close_figure", "delete_dataset", "remove_artist"} <= destructive

    def test_the_snapshot_is_current(self):
        pytest.importorskip("mcp")
        from glplot.mcp.schema import export_tools

        assert (
            export_tools() == load_tools()
        ), "glplot/mcp/tools.json is stale: run `python -m glplot.mcp.schema --write`"


# ----------------------------------------------------------------------------------------
# Argument validation
# ----------------------------------------------------------------------------------------


class TestValidation:
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "minimum": 1, "maximum": 10},
            "x": {"type": "number", "exclusiveMinimum": 0},
            "flag": {"type": "boolean"},
            "name": {"type": "string"},
            "series": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"anyOf": [{"type": "number"}, {"type": "null"}]}},
                    {"type": "null"},
                ]
            },
            "pair": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            "table": {"type": "object", "additionalProperties": {"type": "number"}},
        },
        "required": ["n"],
    }

    def check(self, **arguments):
        return validate_arguments(self.schema, arguments)

    def test_valid_arguments_pass_through(self):
        assert self.check(n=3, x=0.5, flag=True, name="a", pair=[1, 2]) == {
            "n": 3,
            "x": 0.5,
            "flag": True,
            "name": "a",
            "pair": [1, 2],
        }

    def test_missing_unknown_and_out_of_range_name_the_argument(self):
        with pytest.raises(McpError, match="Missing required argument.*n"):
            validate_arguments(self.schema, {})
        with pytest.raises(McpError, match="Unexpected argument.*bogus.*Valid: n"):
            self.check(n=1, bogus=1)
        with pytest.raises(McpError, match="n: must be at most 10, got 11"):
            self.check(n=11)
        with pytest.raises(McpError, match="x: must be greater than 0"):
            self.check(n=1, x=0)
        with pytest.raises(McpError, match="pair: needs at least 2 items"):
            self.check(n=1, pair=[1])

    def test_wrong_types_are_refused_clearly(self):
        with pytest.raises(McpError, match="name: expected text"):
            self.check(n=1, name=5)
        with pytest.raises(McpError, match="n: expected an integer"):
            self.check(n=1.5)
        with pytest.raises(McpError, match="flag: expected true or false"):
            self.check(n=1, flag="maybe")

    def test_loosely_typed_but_clear_values_are_accepted_like_the_sdk_does(self):
        out = self.check(n="3", x="0.5", flag="true", pair="[1, 2]")
        assert out == {"n": 3, "x": 0.5, "flag": True, "pair": [1, 2]}
        assert self.check(n=4.0)["n"] == 4 and isinstance(self.check(n=4.0)["n"], int)

    def test_a_string_stays_a_string_when_a_string_is_allowed(self):
        """A column name must not be read as JSON just because it looks like a list."""
        assert self.check(n=1, series="[col]")["series"] == "[col]"
        assert self.check(n=1, series=[1, None, 2.5])["series"] == [1, None, 2.5]
        assert self.check(n=1, series=None)["series"] is None

    def test_booleans_are_not_numbers(self):
        with pytest.raises(McpError):
            self.check(n=True)

    def test_object_values_are_checked(self):
        assert self.check(n=1, table={"a": 1})["table"] == {"a": 1}
        with pytest.raises(McpError, match="table.a"):
            self.check(n=1, table={"a": "x"})

    def test_a_non_object_argument_set_is_refused(self):
        with pytest.raises(McpError, match="object"):
            validate_arguments(self.schema, [1, 2])  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------------------------


class TestLifecycle:
    def test_initialize_negotiates_a_supported_version_and_describes_itself(self, server):
        info = result(server, "initialize", {"protocolVersion": "2025-06-18"})
        assert info["protocolVersion"] == "2025-06-18"
        assert info["serverInfo"]["name"] == "glplot"
        assert set(info["capabilities"]) == {"tools", "resources", "prompts"}
        assert "get_guide" in info["instructions"] and "render_figure" in info["instructions"]

    @pytest.mark.parametrize("asked", ["1999-01-01", "2099-12-31", None])
    def test_an_unsupported_version_is_answered_with_the_newest_we_speak(self, server, asked):
        params = {} if asked is None else {"protocolVersion": asked}
        assert (
            result(server, "initialize", params)["protocolVersion"] == builtin.PROTOCOL_VERSIONS[-1]
        )

    def test_notifications_get_no_reply_and_initialized_is_recorded(self, server):
        assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
        assert server.initialized
        assert (
            server.handle({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {}})
            is None
        )

    def test_ping_and_unknown_methods(self, server):
        assert result(server, "ping") == {}
        reply = rpc(server, "no/such")
        assert reply["error"]["code"] == builtin.METHOD_NOT_FOUND

    def test_malformed_messages_are_errors_not_crashes(self, server):
        assert (
            server.handle({"id": 1, "method": "ping"})["error"]["code"] == builtin.INVALID_REQUEST
        )
        assert server.handle("nonsense")["error"]["code"] == builtin.INVALID_REQUEST
        reply = server.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/list", "params": [1]})
        assert reply["error"]["code"] == builtin.INVALID_PARAMS

    def test_a_response_from_the_client_is_ignored(self, server):
        assert server.handle({"jsonrpc": "2.0", "id": 9, "result": {}}) is None

    def test_a_batch_is_answered_in_one_array_without_the_notifications(self, server):
        replies = server.handle(
            [
                {"jsonrpc": "2.0", "id": 1, "method": "ping"},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "ping"},
            ]
        )
        assert [r["id"] for r in replies] == [1, 2]


class TestTools:
    def test_tools_list_is_the_snapshot(self, server):
        listed = result(server, "tools/list")["tools"]
        assert [t["name"] for t in listed] == list(TOOLS)
        sample = next(t for t in listed if t["name"] == "add_plot")
        assert sample["inputSchema"]["properties"]["kind"]["description"]
        assert set(sample["annotations"]) == {
            "readOnlyHint",
            "destructiveHint",
            "idempotentHint",
            "openWorldHint",
        }

    def test_a_full_session_ends_in_an_image(self, server):
        created = call(server, "create_figure", title="Built-in", width=400, height=300)
        figure = json.loads(text_of(created).split("\n", 1)[1])["figure_id"]
        call(server, "add_function", figure_id=figure, expression="sin(x)", x_min=0, x_max=6)
        call(server, "set_axes", figure_id=figure, xlabel="x", ylabel="y")
        rendered = call(server, "render_figure", figure_id=figure)
        assert rendered["isError"] is False
        image, text = rendered["content"]
        assert image["type"] == "image" and image["mimeType"] == "image/png"
        assert base64.b64decode(image["data"]).startswith(b"\x89PNG")
        assert "xlim" in text["text"]

    def test_files_follow_the_sandbox(self, server, tmp_path):
        (tmp_path / "run.csv").write_text("t,v\n0,1\n1,4\n2,9\n")
        loaded = call(server, "load_dataset", path="run.csv", name="run")
        assert not loaded["isError"]
        figure = json.loads(text_of(call(server, "create_figure")).split("\n", 1)[1])["figure_id"]
        call(server, "add_plot", figure_id=figure, kind="line", dataset="run", x="t", y="v")
        saved = call(server, "save_figure", figure_id=figure, path="out.png")
        assert not saved["isError"] and (tmp_path / "out.png").read_bytes().startswith(b"\x89PNG")

    def test_a_tool_failure_is_a_result_the_model_can_read(self, server):
        failed = call(server, "add_plot", figure_id="fig99", kind="line", y=[1, 2])
        assert failed["isError"] is True and "No figure 'fig99'" in text_of(failed)
        bad_kind = call(server, "create_figure")  # make a figure, then misuse it
        figure = json.loads(text_of(bad_kind).split("\n", 1)[1])["figure_id"]
        assert "Unknown kind" in text_of(
            call(server, "add_plot", figure_id=figure, kind="nope", y=[1])
        )

    def test_validation_failures_are_results_too(self, server):
        assert call(server, "create_figure", width=50)["isError"] is True
        assert "Unexpected argument" in text_of(call(server, "create_figure", bogus=1))
        assert "Missing required" in text_of(call(server, "close_figure"))

    def test_an_unknown_tool_is_a_protocol_error(self, server):
        reply = rpc(server, "tools/call", {"name": "launch_missiles", "arguments": {}})
        assert reply["error"]["code"] == builtin.INVALID_PARAMS

    def test_json_in_a_string_is_accepted_for_a_list_argument(self, server):
        figure = json.loads(text_of(call(server, "create_figure")).split("\n", 1)[1])["figure_id"]
        call(server, "add_plot", figure_id=figure, kind="line", y=[0, 1, 4])
        done = call(server, "set_axes", figure_id=figure, xlim="[0, 2]")
        assert done["isError"] is False

    def test_analysis_runs_and_options_may_arrive_as_a_json_string(self, server):
        created = call(
            server,
            "create_dataset",
            name="d",
            columns={"t": "linspace(0, 10, 200)", "v": "3*exp(-t/4)+0.5"},
        )
        assert not created["isError"]
        fitted = call(
            server,
            "analyze_dataset",
            dataset="d",
            operation="fit",
            x="t",
            y="v",
            options='{"model": "exp_decay"}',  # some clients send objects as text
        )
        assert fitted["isError"] is False
        reply = text_of(fitted)
        assert reply.startswith("fit of 'v':") and "R^2" in reply
        payload = json.loads(reply.split("\n", 1)[1])
        assert payload["created"]["column"] == "v_fit" and payload["plot_hint"]
        bad = call(
            server,
            "analyze_dataset",
            dataset="d",
            operation="fit",
            x="t",
            y="v",
            options={"model": "nope"},
        )
        assert bad["isError"] is True and "one of:" in text_of(bad)

    def test_the_knowledge_tools_answer(self, server):
        guide = call(server, "get_guide", topic="mcp-data")
        assert "linspace" in text_of(guide)
        assert "imshow" in text_of(call(server, "search_api", query="heatmap"))
        assert "plt.twinx" in text_of(call(server, "get_api_doc", name="twinx"))
        assert "plot_surface" in text_of(call(server, "get_example", name="surface_3d"))
        info = json.loads(text_of(call(server, "server_info")).split("\n", 1)[1])
        assert info["server"]["implementation"] == "builtin"
        assert info["server"]["files"]["enabled"] is True


class TestResourcesAndPrompts:
    def test_listing_matches_the_catalog(self, server):
        listed = result(server, "resources/list")["resources"]
        templates = result(server, "resources/templates/list")["resourceTemplates"]
        assert [r["uri"] for r in listed] == [d.uri for d in catalog.RESOURCES if not d.is_template]
        assert [t["uriTemplate"] for t in templates] == [
            d.uri for d in catalog.RESOURCES if d.is_template
        ]
        for entry in listed + templates:
            assert entry["description"] and entry["mimeType"]

    @pytest.mark.parametrize(
        ("uri", "needle"),
        [
            ("glplot://guide", "GLPlot MCP guide"),
            ("glplot://guide/animation", "FuncAnimation"),
            ("glplot://guide/mcp-recipes", "add_recipe"),
            ("glplot://api", "imshow"),
            ("glplot://api/hist2d", "hist2d"),
            ("glplot://examples", "line_plot"),
            ("glplot://examples/line_plot", "plt.savefig"),
            ("glplot://llms.txt", "# GLPlot"),
            ("glplot://llms-full.txt", "mcp-workflow"),
            ("glplot://capabilities", "style_presets"),
            ("glplot://recipes", "signal_trace"),
            ("glplot://styles", "journal"),
        ],
    )
    def test_text_resources_read(self, server, uri, needle):
        contents = result(server, "resources/read", {"uri": uri})["contents"][0]
        assert contents["uri"] == uri and needle in contents["text"]

    def test_the_image_resource_is_a_base64_blob(self, server):
        figure = json.loads(text_of(call(server, "create_figure")).split("\n", 1)[1])["figure_id"]
        call(server, "add_plot", figure_id=figure, kind="line", y=[1, 3, 2])
        contents = result(server, "resources/read", {"uri": f"glplot://figures/{figure}/image"})[
            "contents"
        ][0]
        assert contents["mimeType"] == "image/png" and "text" not in contents
        assert base64.b64decode(contents["blob"]).startswith(b"\x89PNG")

    def test_unknown_and_failing_resources_are_protocol_errors(self, server):
        assert (
            rpc(server, "resources/read", {"uri": "glplot://nothing"})["error"]["code"]
            == builtin.RESOURCE_NOT_FOUND
        )
        assert (
            rpc(server, "resources/read", {"uri": "glplot://api/scater"})["error"]["code"]
            == builtin.INVALID_PARAMS
        )
        assert rpc(server, "resources/read", {})["error"]["code"] == builtin.INVALID_PARAMS

    def test_prompts(self, server):
        listed = result(server, "prompts/list")["prompts"]
        assert [p["name"] for p in listed] == [p.name for p in catalog.PROMPTS]
        quick = next(p for p in listed if p["name"] == "quick_look")
        assert quick["arguments"][0] == {
            "name": "dataset",
            "description": quick["arguments"][0]["description"],
            "required": True,
        }
        got = result(
            server,
            "prompts/get",
            {
                "name": "fix_my_figure",
                "arguments": {"figure_id": "fig1", "problem": "labels overlap"},
            },
        )
        text = got["messages"][0]["content"]["text"]
        assert got["messages"][0]["role"] == "user" and "fig1" in text and "labels overlap" in text

    def test_prompt_errors_are_invalid_params(self, server):
        assert (
            rpc(server, "prompts/get", {"name": "nope"})["error"]["code"] == builtin.INVALID_PARAMS
        )
        assert (
            rpc(server, "prompts/get", {"name": "quick_look", "arguments": {}})["error"]["code"]
            == builtin.INVALID_PARAMS
        )
        assert (
            rpc(
                server, "prompts/get", {"name": "quick_look", "arguments": {"dataset": "d", "x": 1}}
            )["error"]["code"]
            == builtin.INVALID_PARAMS
        )

    def test_the_python_script_prompt_sends_the_model_to_the_docs_first(self, server):
        got = result(
            server,
            "prompts/get",
            {"name": "python_script_for", "arguments": {"task": "a damped wave"}},
        )
        text = got["messages"][0]["content"]["text"]
        assert (
            "a damped wave" in text
            and "get_guide" in text
            and "plt.savefig" in text
            and "plt.show()" in text
        )


# ----------------------------------------------------------------------------------------
# Transport
# ----------------------------------------------------------------------------------------


def _exchange(server, lines):
    stdin, stdout = io.BytesIO(("\n".join(lines) + "\n").encode()), io.BytesIO()
    code = serve_stdio(server, stdin, stdout)
    return code, [json.loads(line) for line in stdout.getvalue().decode().splitlines()]


class TestStdioLoop:
    def test_replies_are_one_json_object_per_line_and_notifications_are_silent(self, tmp_path):
        server = BuiltinServer(Toolbox(FileSandbox([tmp_path])))
        code, replies = _exchange(
            server,
            [
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {"protocolVersion": "2025-06-18"},
                    }
                ),
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                "",
                json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
            ],
        )
        assert code == 0 and [r["id"] for r in replies] == [1, 2]
        assert len(replies[1]["result"]["tools"]) == len(TOOLS)

    def test_invalid_json_gets_a_parse_error_and_the_loop_continues(self, tmp_path):
        server = BuiltinServer(Toolbox(FileSandbox([tmp_path])))
        _, replies = _exchange(
            server, ["{not json", json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping"})]
        )
        assert replies[0]["error"]["code"] == builtin.PARSE_ERROR and replies[0]["id"] is None
        assert replies[1] == {"jsonrpc": "2.0", "id": 3, "result": {}}

    def test_a_stray_print_inside_a_tool_cannot_reach_the_protocol_channel(
        self, tmp_path, monkeypatch
    ):
        def noisy(self, topic=None):
            print("STRAY LINE FROM A LIBRARY")
            return ToolResult("fine")

        monkeypatch.setattr(Toolbox, "get_guide", noisy)
        server = BuiltinServer(Toolbox(FileSandbox([tmp_path])))
        stdin, stdout = io.BytesIO(), io.BytesIO()
        stdin.write(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {"name": "get_guide", "arguments": {}},
                    }
                )
                + "\n"
            ).encode()
        )
        stdin.seek(0)
        serve_stdio(server, stdin, stdout)
        written = stdout.getvalue().decode()
        assert "STRAY" not in written
        assert json.loads(written)["result"]["content"][0]["text"] == "fine"

    def test_utf8_survives_the_round_trip(self, tmp_path):
        server = BuiltinServer(Toolbox(FileSandbox([tmp_path])))
        title = "Ångström ∆T — 温度"
        _, replies = _exchange(
            server,
            [
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {"name": "create_figure", "arguments": {"title": title}},
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {"name": "describe_figure", "arguments": {"figure_id": "fig1"}},
                    },
                    ensure_ascii=False,
                ),
            ],
        )
        assert replies[0]["result"]["isError"] is False
        assert title in json.dumps(replies[1], ensure_ascii=False)

    def test_figures_are_closed_when_the_client_goes_away(self, tmp_path):
        box = Toolbox(FileSandbox([tmp_path]))
        _exchange(
            BuiltinServer(box),
            [
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {"name": "create_figure", "arguments": {}},
                    }
                )
            ],
        )
        assert box.figure_ids() == []


def _spawn(*extra, root=None):
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
    args = [sys.executable, "-W", "ignore", "-m", "glplot.mcp", *extra]
    if root is not None:
        args += ["--root", str(root)]
    return subprocess.Popen(
        args,
        cwd=REPO_ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )


class TestRealProcess:
    """``python -m glplot.mcp --builtin`` in a subprocess, in raw JSON-RPC -- no SDK needed."""

    def test_a_session_over_real_pipes_keeps_stdout_clean(self, tmp_path):
        process = _spawn("--builtin", root=tmp_path)
        requests = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "t", "version": "0"},
                },
            },
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "create_figure", "arguments": {"width": 320, "height": 240}},
            },
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "add_plot",
                    "arguments": {"figure_id": "fig1", "kind": "line", "y": [0, 1, 4]},
                },
            },
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "render_figure", "arguments": {"figure_id": "fig1"}},
            },
        ]
        out, err = process.communicate(
            "\n".join(json.dumps(r) for r in requests) + "\n", timeout=120
        )
        assert process.returncode == 0, err
        lines = out.splitlines()
        replies = [json.loads(line) for line in lines]  # every stdout line must be JSON
        assert [r["id"] for r in replies] == [1, 2, 3, 4]
        assert replies[0]["result"]["serverInfo"]["name"] == "glplot"
        image = replies[3]["result"]["content"][0]
        assert image["type"] == "image" and base64.b64decode(image["data"]).startswith(b"\x89PNG")
        assert "built-in server" in err

    def test_it_runs_without_the_sdk_installed(self, tmp_path):
        """The promise: nothing but GLPlot's own dependencies is imported."""
        code = (
            "import sys;"
            "sys.modules['mcp'] = None;"  # makes `import mcp` raise ImportError
            "from glplot.mcp.cli import main;"
            "raise SystemExit(main(['--version']))"
        )
        done = subprocess.run(
            [sys.executable, "-W", "ignore", "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 0, done.stderr
        assert "not installed (built-in server)" in done.stderr


# ----------------------------------------------------------------------------------------
# Conformance with the official SDK
# ----------------------------------------------------------------------------------------


def _attr(obj, *names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(names)


class TestOfficialClientDrivesTheBuiltinServer:
    def test_a_real_mcp_client_can_use_every_part_of_it(self, tmp_path):
        pytest.importorskip("mcp")
        (tmp_path / "run.csv").write_text("t,v\n0,1\n1,4\n2,9\n")

        async def go():
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client

            params = StdioServerParameters(
                command=sys.executable,
                args=["-W", "ignore", "-m", "glplot.mcp", "--builtin", "--root", str(tmp_path)],
                env={
                    **os.environ,
                    "PYTHONPATH": str(REPO_ROOT),
                    "MPLBACKEND": "Agg",
                    "PYTHONUNBUFFERED": "1",
                },
                cwd=str(REPO_ROOT),
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    init = await session.initialize()
                    tools = (await session.list_tools()).tools
                    await session.call_tool("create_figure", {"title": "Official client"})
                    await session.call_tool("load_dataset", {"path": "run.csv", "name": "run"})
                    await session.call_tool(
                        "add_plot",
                        {"figure_id": "fig1", "kind": "line", "dataset": "run", "x": "t", "y": "v"},
                    )
                    rendered = await session.call_tool("render_figure", {"figure_id": "fig1"})
                    bad = await session.call_tool("create_figure", {"width": 50})
                    guide = await session.call_tool("get_guide", {"topic": "quickstart"})
                    resources = await session.list_resources()
                    templates = await session.list_resource_templates()
                    doc = await session.read_resource("glplot://api/hist2d")
                    prompts = await session.list_prompts()
                    prompt = await session.get_prompt("quick_look", {"dataset": "run"})
                    return (
                        init,
                        tools,
                        rendered,
                        bad,
                        guide,
                        resources,
                        templates,
                        doc,
                        prompts,
                        prompt,
                    )

        init, tools, rendered, bad, guide, resources, templates, doc, prompts, prompt = asyncio.run(
            asyncio.wait_for(go(), timeout=120)
        )
        assert _attr(init, "server_info", "serverInfo").name == "glplot"
        assert [t.name for t in tools] == list(TOOLS)
        assert rendered.content[0].type == "image"
        assert base64.b64decode(rendered.content[0].data).startswith(b"\x89PNG")
        assert _attr(bad, "is_error", "isError") is True
        assert "Quickstart" in guide.content[0].text
        assert len(resources.resources) == sum(1 for d in catalog.RESOURCES if not d.is_template)
        assert len(_attr(templates, "resource_templates", "resourceTemplates")) == sum(
            1 for d in catalog.RESOURCES if d.is_template
        )
        assert "hist2d" in doc.contents[0].text
        assert {p.name for p in prompts.prompts} == {p.name for p in catalog.PROMPTS}
        assert "describe_dataset" in prompt.messages[0].content.text

    def test_the_two_servers_expose_the_same_tools_resources_and_prompts(self, tmp_path):
        pytest.importorskip("mcp")
        from glplot.mcp.server import build_server

        sdk_toolbox = Toolbox(FileSandbox([tmp_path]))
        sdk = build_server(sdk_toolbox)

        async def go():
            import contextlib

            @contextlib.asynccontextmanager
            async def connect(server):
                try:
                    from mcp import Client

                    async with Client(server) as client:
                        yield client
                except ImportError:
                    from mcp.shared.memory import create_connected_server_and_client_session

                    async with create_connected_server_and_client_session(server._mcp_server) as s:
                        yield s

            async with connect(sdk) as client:
                tools = [t.name for t in (await client.list_tools()).tools]
                resources = sorted(str(r.uri) for r in (await client.list_resources()).resources)
                prompts = sorted(p.name for p in (await client.list_prompts()).prompts)
                return tools, resources, prompts

        try:
            tools, resources, prompts = asyncio.run(go())
        finally:
            sdk_toolbox.close_all()
        assert tools == list(TOOLS)
        assert resources == sorted(d.uri for d in catalog.RESOURCES if not d.is_template)
        assert prompts == sorted(p.name for p in catalog.PROMPTS)
