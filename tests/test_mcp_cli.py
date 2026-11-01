"""``glplot-mcp``: choosing a server, registering it in clients, and the self-checks.

Everything here runs without the optional ``mcp`` package (the point of the built-in server);
the one test of the SDK path skips without it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from glplot.mcp import builtin, cli
from glplot.mcp.state import FileSandbox

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def no_warmup(monkeypatch):
    monkeypatch.setattr(cli, "_warm_up", lambda: None)


@pytest.fixture
def served(monkeypatch, no_warmup):
    """Replace the stdio loop with a recorder, so ``main`` can be run without a client."""
    seen = {}

    def fake(server, *args, **kwargs):
        seen["server"] = server
        return 0

    monkeypatch.setattr(builtin, "serve_stdio", fake)
    return seen


class TestParser:
    def test_defaults_are_safe(self):
        args = cli.build_parser().parse_args([])
        assert args.transport == "stdio" and args.host == "127.0.0.1" and args.root == []
        assert args.builtin is False and args.sdk is False

    def test_builtin_and_sdk_are_exclusive(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["--builtin", "--sdk"])

    def test_no_root_means_no_file_access_whatever_the_working_directory(
        self, monkeypatch, tmp_path
    ):
        from glplot.mcp.state import McpError

        (tmp_path / "secret.csv").write_text("x,y\n1,2\n")
        monkeypatch.chdir(tmp_path)
        sandbox = FileSandbox(list(cli.build_parser().parse_args([]).root))
        assert not sandbox.enabled
        with pytest.raises(McpError, match="File access is disabled"):
            sandbox.read_path("secret.csv")

    def test_the_console_script_and_the_sdk_extra_are_declared(self):
        import tomllib

        project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
        assert project["scripts"]["glplot-mcp"] == "glplot.mcp.__main__:main"
        assert any(dep.startswith("mcp") for dep in project["optional-dependencies"]["mcp"])
        # The SDK stays optional: a plain install must not require it.
        assert not any(dep.startswith("mcp") for dep in project["dependencies"])


class TestChoosingAServer:
    def test_without_the_sdk_the_builtin_server_runs(self, monkeypatch, served, tmp_path, caplog):
        monkeypatch.setattr(cli, "_sdk_version", lambda: None)
        with caplog.at_level("INFO", logger="glplot.mcp"):
            assert cli.main(["--root", str(tmp_path)]) == 0
        server = served["server"]
        assert isinstance(server, builtin.BuiltinServer)
        assert server.box.sandbox.enabled and server.box.sandbox.roots == [tmp_path.resolve()]
        assert "built-in server" in caplog.text

    def test_a_bare_invocation_works_the_same(self, monkeypatch, served):
        monkeypatch.setattr(cli, "_sdk_version", lambda: None)
        assert cli.main([]) == 0 and not served["server"].box.sandbox.enabled

    def test_builtin_can_be_forced_even_when_the_sdk_is_present(self, monkeypatch, served):
        monkeypatch.setattr(cli, "_sdk_version", lambda: "9.9.9")
        assert cli.main(["--builtin"]) == 0
        assert isinstance(served["server"], builtin.BuiltinServer)

    def test_http_needs_the_sdk_and_says_how_to_get_it(self, monkeypatch, no_warmup, capsys):
        monkeypatch.setattr(cli, "_sdk_version", lambda: None)
        assert cli.main(["--transport", "streamable-http"]) == 2
        err = capsys.readouterr().err
        assert 'pip install "glplot[mcp]"' in err and "Traceback" not in err

    def test_sdk_required_but_missing_is_one_readable_line(self, monkeypatch, no_warmup, capsys):
        monkeypatch.setattr(cli, "_sdk_version", lambda: None)
        assert cli.main(["--sdk"]) == 2
        assert 'pip install "glplot[mcp]"' in capsys.readouterr().err

    def test_builtin_with_http_is_refused(self, monkeypatch, no_warmup, capsys):
        monkeypatch.setattr(cli, "_sdk_version", lambda: "9.9.9")
        assert cli.main(["--builtin", "--transport", "streamable-http"]) == 2
        assert "--builtin" in capsys.readouterr().err

    def test_the_sdk_server_is_preferred_when_it_is_installed(self, monkeypatch, no_warmup):
        pytest.importorskip("mcp")
        from glplot.mcp import server as server_module

        ran = {}

        class Stub:
            def run(self, *args, **kwargs):
                ran["ran"] = True

        monkeypatch.setattr(server_module, "build_server", lambda **kwargs: Stub())
        assert cli.main([]) == 0 and ran == {"ran": True}

    def test_bad_arguments_exit_2_with_a_message(self, no_warmup, capsys):
        assert cli.main(["--max-figures", "0"]) == 2
        assert "--max-figures" in capsys.readouterr().err
        assert cli.main(["--root", "/"]) == 2
        assert capsys.readouterr().err.startswith("glplot-mcp:")

    def test_version_reports_both_sides(self, capsys):
        assert cli.main(["--version"]) == 0
        err = capsys.readouterr().err
        assert err.startswith("glplot ") and "mcp " in err and "python " in err

    def test_importing_the_cli_never_imports_the_sdk(self):
        import subprocess

        code = (
            "import sys, glplot.mcp.cli, glplot.mcp.builtin, glplot.mcp.catalog, glplot.mcp.schema;"
            "bad = [m for m in sys.modules if m == 'mcp' or m.startswith('mcp.')];"
            "assert not bad, bad"
        )
        done = subprocess.run(
            [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
        )
        assert done.returncode == 0, done.stderr


class TestClientConfig:
    command = "/opt/venv/bin/python"

    def config(self, client, roots=()):
        return cli.client_config(client, "glplot", self.command, ["-m", "glplot.mcp", *roots])

    def test_server_command_uses_the_absolute_interpreter_and_resolved_roots(self, tmp_path):
        command, args = cli._server_command([str(tmp_path / ".." / tmp_path.name)])
        assert Path(command).is_absolute() and command == sys.executable
        assert args == ["-m", "glplot.mcp", "--root", str(tmp_path.resolve())]
        assert cli._server_command([], python="/x/python")[0] == "/x/python"

    def test_claude_code_is_a_shell_line(self):
        text = self.config("claude-code", ["--root", "/data"])["text"]
        assert text.startswith(
            "claude mcp add glplot -- /opt/venv/bin/python -m glplot.mcp --root /data"
        )

    def test_claude_desktop_and_cursor_use_mcpservers(self):
        for client in ("claude-desktop", "cursor"):
            config = self.config(client)
            assert config["data"] == {
                "mcpServers": {"glplot": {"command": self.command, "args": ["-m", "glplot.mcp"]}}
            }
            assert config["path"].name in ("claude_desktop_config.json", "mcp.json")
            assert json.loads(config["text"].split("\n", 1)[1]) == config["data"]

    def test_vscode_declares_a_stdio_server(self):
        config = self.config("vscode")
        assert config["data"]["servers"]["glplot"]["type"] == "stdio"
        assert config["path"] == Path(".vscode") / "mcp.json"

    def test_codex_is_toml(self):
        text = self.config("codex")["text"]
        assert "[mcp_servers.glplot]" in text and 'command = "/opt/venv/bin/python"' in text
        assert 'args = ["-m", "glplot.mcp"]' in text

    def test_generic_is_plain_json(self):
        assert json.loads(self.config("generic")["text"])["glplot"]["command"] == self.command

    def test_claude_desktop_path_depends_on_the_platform(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr(sys, "platform", "darwin")
        assert "Application Support" in str(cli._claude_desktop_path())
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(cli.os, "name", "posix")
        assert str(cli._claude_desktop_path()).endswith(".config/Claude/claude_desktop_config.json")


class TestWritingAConfig:
    def entry(self, tmp_path, existing=None):
        path = tmp_path / "sub" / "config.json"
        if existing is not None:
            path.parent.mkdir(parents=True)
            path.write_text(existing)
        data = {"mcpServers": {"glplot": {"command": "py", "args": ["-m", "glplot.mcp"]}}}
        return path, {"path": path, "data": data, "key": "mcpServers"}

    def test_it_creates_the_file_and_parents(self, tmp_path):
        path, config = self.entry(tmp_path)
        assert cli.write_json_config(config, "glplot") == path
        assert json.loads(path.read_text()) == config["data"]
        assert not list(path.parent.glob("*.tmp"))

    def test_it_keeps_everything_else_and_leaves_a_backup(self, tmp_path):
        original = json.dumps({"theme": "dark", "mcpServers": {"other": {"command": "x"}}})
        path, config = self.entry(tmp_path, original)
        cli.write_json_config(config, "glplot")
        merged = json.loads(path.read_text())
        assert merged["theme"] == "dark" and set(merged["mcpServers"]) == {"other", "glplot"}
        assert path.with_name("config.json.bak").read_text() == original

    def test_it_replaces_an_older_entry_of_the_same_name(self, tmp_path):
        original = json.dumps({"mcpServers": {"glplot": {"command": "old"}}})
        path, config = self.entry(tmp_path, original)
        cli.write_json_config(config, "glplot")
        assert json.loads(path.read_text())["mcpServers"]["glplot"]["command"] == "py"

    @pytest.mark.parametrize("bad", ["{oops", "[1, 2]", '{"mcpServers": 5}'])
    def test_it_refuses_a_file_it_cannot_safely_merge_and_does_not_touch_it(self, tmp_path, bad):
        path, config = self.entry(tmp_path, bad)
        with pytest.raises(ValueError):
            cli.write_json_config(config, "glplot")
        assert path.read_text() == bad

    def test_the_command_writes_through_main(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(cli.os, "name", "posix")
        assert (
            cli.main(["config", "--client", "claude-desktop", "--write", "--root", str(tmp_path)])
            == 0
        )
        written = json.loads(
            (tmp_path / ".config" / "Claude" / "claude_desktop_config.json").read_text()
        )
        assert written["mcpServers"]["glplot"]["args"][-1] == str(tmp_path.resolve())
        assert "Restart" in capsys.readouterr().out


class TestSubcommands:
    def test_config_prints_every_client_and_warns_about_a_missing_root(self, capsys):
        assert cli.main(["config"]) == 0
        out = capsys.readouterr().out
        for client in cli.CLIENTS:
            assert f"## {client}" in out
        assert "No --root given" in out

    def test_write_needs_one_client(self, capsys):
        assert cli.main(["config", "--write"]) == 2
        assert "single --client" in capsys.readouterr().err

    def test_write_for_a_client_without_a_file_explains_the_alternative(self, capsys):
        assert cli.main(["config", "--client", "codex", "--write"]) == 2
        assert "no JSON file" in capsys.readouterr().err

    def test_tools_lists_every_tool_with_its_safety_class(self, capsys):
        assert cli.main(["tools"]) == 0
        out = capsys.readouterr().out
        assert "create_figure" in out and "[read-only" in out and "[destructive]" in out
        assert len(out.strip().splitlines()) >= 23

    def test_doctor_runs_the_protocol_self_test(self, capsys):
        assert cli.main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "headless render" in out and "mcp self-test" in out and "All checks passed" in out
        assert "render returned an image" in out
