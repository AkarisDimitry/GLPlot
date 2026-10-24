"""The MCP protocol layer: tools, resources and prompts as a real client sees them.

The behaviour behind each tool is covered in ``test_mcp_toolbox.py``; this file checks the
*description* of it -- that the server registers what it promises, that a model's mistakes
come back as readable errors rather than crashes, and that the stdio transport (the one an
MCP client actually launches) keeps stdout clean for the protocol.

The SDK is an optional extra, so everything that needs it skips cleanly without it, while
the tests that must hold *without* it (importing the package, the friendly CLI error) run
regardless.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from glplot.mcp.guide import build_guide, build_instructions
from glplot.mcp.state import FileSandbox
from glplot.mcp.toolbox import Toolbox

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_TOOLS = {
    "create_figure",
    "describe_figure",
    "close_figure",
    "create_dataset",
    "load_dataset",
    "describe_dataset",
    "delete_dataset",
    "add_plot",
    "add_function",
    "add_field",
    "add_recipe",
    "list_recipes",
    "add_annotation",
    "remove_artist",
    "set_axes",
    "set_style",
    "render_figure",
    "save_figure",
}
READ_ONLY = {"describe_figure", "describe_dataset", "render_figure", "list_recipes"}
DESTRUCTIVE = {"close_figure", "delete_dataset", "remove_artist"}


def attr(obj, *names):
    """First attribute present: the SDK's 1.x and 2.x result types spell them differently."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"{type(obj).__name__} has none of {names}")


@contextlib.asynccontextmanager
async def connect(server):
    """An in-process client for ``server``, on whichever SDK generation is installed."""
    try:
        from mcp import Client  # mcp >= 2
    except ImportError:
        from mcp.shared.memory import create_connected_server_and_client_session

        async with create_connected_server_and_client_session(server._mcp_server) as session:
            yield session
    else:
        async with Client(server) as client:
            yield client


@pytest.fixture
def root(tmp_path):
    base = tmp_path / "work"
    base.mkdir()
    return base


@pytest.fixture
def server(root):
    pytest.importorskip("mcp")
    from glplot.mcp.server import build_server

    toolbox = Toolbox(FileSandbox([root]))
    yield build_server(toolbox)
    toolbox.close_all()


def run(coro):
    return asyncio.run(coro)


class TestRegistration:
    def test_every_tool_is_registered_with_a_description_and_schema(self, server):
        async def go():
            async with connect(server) as client:
                return (await client.list_tools()).tools

        tools = {tool.name: tool for tool in run(go())}
        assert set(tools) == EXPECTED_TOOLS
        for name, tool in tools.items():
            assert tool.description and len(tool.description) > 20, name
            schema = attr(tool, "input_schema", "inputSchema")
            assert schema["type"] == "object", name

    def test_every_argument_is_described(self, server):
        """The schema's per-argument text is the model's whole manual; a bare name is a guess."""

        async def go():
            async with connect(server) as client:
                return (await client.list_tools()).tools

        bare = []
        for tool in run(go()):
            schema = attr(tool, "input_schema", "inputSchema")
            for argument, spec in schema.get("properties", {}).items():
                if "description" not in spec:
                    bare.append(f"{tool.name}.{argument}")
        assert bare == []

    def test_safety_hints(self, server):
        async def go():
            async with connect(server) as client:
                return (await client.list_tools()).tools

        for tool in run(go()):
            hints = tool.annotations
            read_only = attr(hints, "read_only_hint", "readOnlyHint")
            destructive = attr(hints, "destructive_hint", "destructiveHint")
            assert read_only == (tool.name in READ_ONLY), tool.name
            if tool.name in DESTRUCTIVE:
                assert destructive is True, tool.name
            assert attr(hints, "open_world_hint", "openWorldHint") is False

    def test_kind_enumerations_are_in_the_schema_text(self, server):
        from glplot.mcp.toolbox import FIELD_KINDS, PLOT_KINDS

        async def go():
            async with connect(server) as client:
                return {t.name: t for t in (await client.list_tools()).tools}

        tools = run(go())
        plot_kind = attr(tools["add_plot"], "input_schema", "inputSchema")["properties"]["kind"]
        field_kind = attr(tools["add_field"], "input_schema", "inputSchema")["properties"]["kind"]
        assert all(kind in plot_kind["description"] for kind in PLOT_KINDS)
        assert all(kind in field_kind["description"] for kind in FIELD_KINDS)

    def test_resources_and_prompts(self, server):
        async def go():
            async with connect(server) as client:
                resources = await client.list_resources()
                templates = await client.list_resource_templates()
                prompts = await client.list_prompts()
                return resources, templates, prompts

        resources, templates, prompts = run(go())
        assert [str(r.uri) for r in resources.resources] == ["glplot://guide"]
        uris = [
            str(attr(t, "uri_template", "uriTemplate"))
            for t in attr(templates, "resource_templates", "resourceTemplates")
        ]
        assert sorted(uris) == ["glplot://datasets/{name}", "glplot://figures/{figure_id}/image"]
        assert {p.name for p in prompts.prompts} == {"plot_data_file", "explore_function"}


class TestCalls:
    def test_a_full_session_ends_in_an_image_the_model_can_see(self, server):
        async def go():
            async with connect(server) as client:
                created = await client.call_tool(
                    "create_figure", {"title": "Demo", "width": 900, "height": 560}
                )
                await client.call_tool(
                    "add_function",
                    {
                        "figure_id": "fig1",
                        "expression": "sin(a*x)*exp(-x/6)",
                        "x_min": 0,
                        "x_max": 20,
                        "parameters": {"a": 2},
                    },
                )
                await client.call_tool(
                    "set_axes",
                    {"figure_id": "fig1", "xlabel": "t (s)", "ylabel": "amp", "legend": True},
                )
                return created, await client.call_tool("render_figure", {"figure_id": "fig1"})

        created, rendered = run(go())
        assert not attr(created, "is_error", "isError")
        assert "fig1" in created.content[0].text
        kinds = [c.type for c in rendered.content]
        assert kinds == ["image", "text"]  # the picture first
        assert attr(rendered.content[0], "mime_type", "mimeType") == "image/png"
        assert base64.b64decode(rendered.content[0].data).startswith(b"\x89PNG")
        summary = rendered.content[1].text
        assert '"xlabel":"t (s)"' in summary and '"legend":["sin(a*x)*exp(-x/6)"]' in summary

    def test_a_recipe_session_over_the_protocol(self, server):
        """list_recipes -> add_recipe (sample) -> add_recipe (own data + override) -> render."""

        async def go():
            async with connect(server) as client:
                listing = await client.call_tool("list_recipes", {"category": "Lines"})
                await client.call_tool(
                    "create_figure", {"title": "Recipes", "width": 900, "height": 560}
                )
                sample = await client.call_tool(
                    "add_recipe", {"figure_id": "fig1", "recipe": "signal_trace", "sample": True}
                )
                await client.call_tool(
                    "create_dataset",
                    {
                        "name": "mine",
                        "columns": {"t": "linspace(0, 6, 80)", "volts": "cos(t)", "amps": "sin(t)"},
                    },
                )
                own = await client.call_tool(
                    "add_recipe",
                    {
                        "figure_id": "fig1",
                        "recipe": "step_response",
                        "dataset": "mine",
                        "columns": {"y": "amps"},
                        "look": "keep",
                    },
                )
                bad = await client.call_tool(
                    "add_recipe", {"figure_id": "fig1", "recipe": "surface3d", "sample": True}
                )
                rendered = await client.call_tool("render_figure", {"figure_id": "fig1"})
                return listing, sample, own, bad, rendered

        listing, sample, own, bad, rendered = run(go())
        assert '"recipe":"signal_trace"' in listing.content[0].text
        assert (
            not attr(sample, "is_error", "isError") and '"look":"chalk"' in sample.content[0].text
        )
        assert not attr(own, "is_error", "isError") and '"y":"amps"' in own.content[0].text
        assert attr(bad, "is_error", "isError") is True and "projection='3d'" in bad.content[0].text
        assert [c.type for c in rendered.content] == ["image", "text"]
        assert base64.b64decode(rendered.content[0].data).startswith(b"\x89PNG")

    def test_a_models_mistakes_come_back_as_actionable_errors(self, server):
        async def go():
            async with connect(server) as client:
                await client.call_tool("create_figure", {})
                return {
                    "figure": await client.call_tool(
                        "add_plot", {"figure_id": "fig9", "kind": "line", "y": [1, 2]}
                    ),
                    "column": await client.call_tool(
                        "add_plot", {"figure_id": "fig1", "kind": "line", "y": "col"}
                    ),
                    "kind": await client.call_tool(
                        "add_plot", {"figure_id": "fig1", "kind": "scater", "y": [1]}
                    ),
                    "expression": await client.call_tool(
                        "create_dataset", {"name": "d", "columns": {"a": "__import__('os')"}}
                    ),
                    "path": await client.call_tool("load_dataset", {"path": "../../etc/passwd"}),
                    "save": await client.call_tool(
                        "save_figure", {"figure_id": "fig1", "path": "x.py"}
                    ),
                }

        results = run(go())
        for name, result in results.items():
            assert attr(result, "is_error", "isError") is True, name
        text = {name: result.content[0].text for name, result in results.items()}
        assert "Did you mean 'fig1'" in text["figure"]
        assert "create_dataset" in text["column"]
        assert "Did you mean 'scatter'" in text["kind"]
        assert "Could not evaluate" in text["expression"]
        assert "outside" in text["path"]
        assert "Cannot save" in text["save"]

    def test_arguments_outside_the_schema_are_rejected_before_running(self, server):
        async def go():
            async with connect(server) as client:
                await client.call_tool("create_figure", {})
                return (
                    await client.call_tool(
                        "add_plot", {"figure_id": "fig1", "kind": "line", "y": [1], "alpha": 5}
                    ),
                    await client.call_tool("create_figure", {"width": 5}),
                    await client.call_tool("add_plot", {"figure_id": "fig1"}),
                )

        for result in run(go()):
            assert attr(result, "is_error", "isError") is True

    def test_an_unexpected_exception_does_not_leak_a_traceback(self, server, monkeypatch):
        import glplot.pyplot as gplt

        monkeypatch.setattr(gplt, "title", lambda *a, **k: 1 / 0)

        async def go():
            async with connect(server) as client:
                await client.call_tool("create_figure", {})
                return await client.call_tool("set_axes", {"figure_id": "fig1", "title": "x"})

        result = run(go())
        assert attr(result, "is_error", "isError") is True
        assert "ZeroDivisionError" not in result.content[0].text
        assert "Traceback" not in result.content[0].text

    def test_resources_serve_the_guide_the_figure_and_the_dataset(self, server):
        async def go():
            async with connect(server) as client:
                await client.call_tool("create_figure", {})
                await client.call_tool(
                    "add_plot", {"figure_id": "fig1", "kind": "line", "y": [1, 2, 3]}
                )
                await client.call_tool("create_dataset", {"name": "d", "columns": {"a": [1, 2, 3]}})
                guide = await client.read_resource("glplot://guide")
                image = await client.read_resource("glplot://figures/fig1/image")
                table = await client.read_resource("glplot://datasets/d")
                return guide, image, table

        guide, image, table = run(go())
        assert "# GLPlot MCP guide" in guide.contents[0].text
        assert attr(image.contents[0], "mime_type", "mimeType") == "image/png"
        assert base64.b64decode(image.contents[0].blob).startswith(b"\x89PNG")
        assert '"rows": 3' in table.contents[0].text

    def test_prompts_render(self, server):
        async def go():
            async with connect(server) as client:
                a = await client.get_prompt(
                    "plot_data_file", {"path": "run.csv", "goal": "compare runs"}
                )
                b = await client.get_prompt("explore_function", {"expression": "sin(x)/x"})
                return a, b

        a, b = run(go())
        assert (
            "run.csv" in a.messages[0].content.text and "compare runs" in a.messages[0].content.text
        )
        assert "sin(x)/x" in b.messages[0].content.text


class TestGuide:
    def test_guide_documents_every_tool_a_model_is_told_to_use(self, root):
        guide = build_guide(FileSandbox([root]))
        for name in (
            "create_figure",
            "create_dataset",
            "load_dataset",
            "add_plot",
            "add_function",
            "add_field",
            "add_recipe",
            "list_recipes",
            "add_annotation",
            "set_axes",
            "render_figure",
            "save_figure",
            "remove_artist",
            "close_figure",
        ):
            assert name in guide, name

    def test_guide_lists_every_recipe_with_its_slots_and_look(self, root):
        from glplot.gui import recipes

        guide = build_guide(FileSandbox([root]))
        for recipe in recipes.RECIPES:
            row = next(line for line in guide.splitlines() if line.startswith(f"| `{recipe.key}`"))
            assert recipe.style in row, recipe.key
            for slot in recipe.required_slots():
                if not slot.multi:
                    assert f"{slot.key}*" in row, (recipe.key, slot.key)

    def test_the_guide_labels_a_slot_whose_key_would_mislead(self, root):
        row = next(
            line
            for line in build_guide(FileSandbox([root])).splitlines()
            if line.startswith("| `line_family`")
        )
        assert "Slope" in row and "Intercept" in row  # its slots are keyed x and y

    def test_the_instructions_point_at_recipes(self):
        assert "add_recipe" in build_instructions() and "list_recipes" in build_instructions()

    def test_guide_lists_every_plot_kind(self, root):
        from glplot.mcp.toolbox import PLOT_KINDS

        guide = build_guide(FileSandbox([root]))
        assert all(
            f"| {kind}" in guide or f"| {kind}," in guide or kind in guide for kind in PLOT_KINDS
        )

    def test_the_expression_vocabulary_is_the_evaluators(self, root):
        from glplot.gui.expressions import SAFE_NAMES

        guide = build_guide(FileSandbox([root]))
        for name in ("sin", "linspace", "noise", "gauss"):
            assert name in SAFE_NAMES and name in guide

    def test_guide_says_where_files_may_go_or_that_they_may_not(self, root):
        assert str(root.resolve()) in build_guide(FileSandbox([root]))
        assert "disabled" in build_guide(FileSandbox(()))
        assert "disabled" in build_instructions(FileSandbox(()))
        assert str(root.resolve()) in build_instructions(FileSandbox([root]))

    def test_instructions_describe_the_loop(self):
        text = build_instructions()
        assert "render_figure" in text and "create_figure" in text


class TestEntryPoints:
    def test_importing_the_package_never_imports_the_sdk(self):
        code = (
            "import sys, glplot.mcp, glplot.mcp.state, glplot.mcp.toolbox, glplot.mcp.guide;"
            "bad = [m for m in sys.modules if m == 'mcp' or m.startswith('mcp.')];"
            "assert not bad, bad"
        )
        done = subprocess.run(
            [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
        )
        assert done.returncode == 0, done.stderr

    def test_lazy_exports(self):
        import glplot.mcp as package

        assert package.Toolbox is Toolbox
        assert package.Limits.__name__ == "Limits"
        with pytest.raises(AttributeError):
            package.nonexistent  # noqa: B018

    def test_a_missing_sdk_is_one_readable_line(self, monkeypatch, capsys):
        from glplot.mcp import __main__ as entry

        monkeypatch.setitem(sys.modules, "glplot.mcp.server", None)  # makes the import fail
        assert entry.main() == 2
        err = capsys.readouterr().err
        assert 'pip install "glplot[mcp]"' in err and "Traceback" not in err

    def test_default_roots(self, monkeypatch, tmp_path):
        pytest.importorskip("mcp")
        from glplot.mcp import server as server_module

        assert server_module.default_roots(["/a", "/b"]) == ["/a", "/b"]
        monkeypatch.chdir(tmp_path)
        assert server_module.default_roots([]) == [str(tmp_path.resolve())]
        monkeypatch.setattr(
            server_module.Path, "cwd", classmethod(lambda cls: Path(Path(tmp_path).anchor))
        )
        assert server_module.default_roots([]) == []  # never the whole disk

    def test_parser_defaults_are_safe(self):
        pytest.importorskip("mcp")
        from glplot.mcp.server import build_parser

        args = build_parser().parse_args([])
        assert args.transport == "stdio" and args.host == "127.0.0.1" and args.root == []

    def test_console_script_is_declared(self):
        import tomllib

        project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
        assert project["scripts"]["glplot-mcp"] == "glplot.mcp.__main__:main"
        assert any(dep.startswith("mcp") for dep in project["optional-dependencies"]["mcp"])


class TestStdioTransport:
    """The real thing: a client launches ``python -m glplot.mcp`` and talks over its pipes."""

    def test_a_client_can_drive_the_server_over_stdio(self, root):
        pytest.importorskip("mcp")
        (root / "run.csv").write_text("t,v\n0,1\n1,4\n2,9\n")

        async def go():
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client

            env = {
                **os.environ,
                "PYTHONPATH": str(REPO_ROOT),
                "PYTHONUNBUFFERED": "1",
                "MPLBACKEND": "Agg",
            }
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "glplot.mcp", "--root", str(root)],
                env=env,
                cwd=str(REPO_ROOT),
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    await session.call_tool("create_figure", {"title": "Over stdio"})
                    await session.call_tool("load_dataset", {"path": "run.csv", "name": "run"})
                    await session.call_tool(
                        "add_plot",
                        {"figure_id": "fig1", "kind": "line", "dataset": "run", "x": "t", "y": "v"},
                    )
                    await session.call_tool("create_figure", {"title": "Recipe over stdio"})
                    recipe = await session.call_tool(
                        "add_recipe", {"figure_id": "fig2", "recipe": "histogram", "sample": True}
                    )
                    rendered = await session.call_tool("render_figure", {"figure_id": "fig1"})
                    saved = await session.call_tool(
                        "save_figure", {"figure_id": "fig1", "path": "out.png"}
                    )
                    return tools, rendered, saved, recipe

        tools, rendered, saved, recipe = asyncio.run(asyncio.wait_for(go(), timeout=90))
        assert {t.name for t in tools.tools} == EXPECTED_TOOLS
        assert base64.b64decode(rendered.content[0].data).startswith(b"\x89PNG")
        assert not attr(saved, "is_error", "isError")
        assert (root / "out.png").read_bytes().startswith(b"\x89PNG")
        assert not attr(recipe, "is_error", "isError")
        assert '"recipe":"histogram"' in recipe.content[0].text

    def test_without_a_root_files_are_refused_not_the_cwd_served(self, tmp_path):
        pytest.importorskip("mcp")

        async def go():
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client

            env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "MPLBACKEND": "Agg"}
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "glplot.mcp", "--root", str(tmp_path)],
                env=env,
                cwd=str(REPO_ROOT),
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.call_tool(
                        "load_dataset", {"path": str(REPO_ROOT / "pyproject.toml")}
                    )

        result = asyncio.run(asyncio.wait_for(go(), timeout=90))
        assert attr(result, "is_error", "isError") is True
        assert "outside" in result.content[0].text
