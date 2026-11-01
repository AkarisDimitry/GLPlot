"""The streamable-HTTP transport: bearer-token auth, host checks, and not being left open.

The unit tests need nothing beyond GLPlot. The integration tests start a real server process and
talk HTTP to it (and, for two, connect the official MCP client), so they need the optional ``mcp``
package and skip without it.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from glplot.mcp import cli
from glplot.mcp.http import (
    TOKEN_ENV,
    BearerAuth,
    is_loopback,
    resolve_token,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKEN = "s3cret-token-123"
INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}
HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


class TestTokenAndLoopback:
    @pytest.mark.parametrize(
        ("host", "expected"),
        [
            ("127.0.0.1", True),
            ("localhost", True),
            ("LOCALHOST", True),
            ("::1", True),
            ("[::1]", True),
            ("127.1.2.3", True),
            ("0.0.0.0", False),
            ("192.168.0.5", False),
            ("plots.example.org", False),
            ("::", False),
        ],
    )
    def test_loopback_detection(self, host, expected):
        assert is_loopback(host) is expected

    def test_the_option_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv(TOKEN_ENV, "from-the-environment")
        assert resolve_token("from-the-option") == "from-the-option"
        assert resolve_token(None) == "from-the-environment"

    def test_no_token_means_no_auth(self, monkeypatch):
        monkeypatch.delenv(TOKEN_ENV, raising=False)
        assert resolve_token(None) is None
        monkeypatch.setenv(TOKEN_ENV, "   ")
        assert resolve_token(None) is None

    def test_a_weak_token_is_refused_with_advice(self):
        with pytest.raises(ValueError, match="at least 8.*secrets.token_urlsafe"):
            resolve_token("abc")


class _Recorder:
    """A tiny ASGI app that records the scope types it is called with."""

    def __init__(self):
        self.seen = []

    async def __call__(self, scope, receive, send):
        self.seen.append(scope["type"])
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})


def _call(app, scope):
    sent = []

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    return sent


def _http(headers):
    return {"type": "http", "headers": [(k.encode(), v.encode()) for k, v in headers.items()]}


class TestBearerAuth:
    def test_the_right_token_reaches_the_app(self):
        inner = _Recorder()
        sent = _call(BearerAuth(inner, TOKEN), _http({"authorization": f"Bearer {TOKEN}"}))
        assert inner.seen == ["http"] and sent[0]["status"] == 200

    def test_the_scheme_name_is_case_insensitive(self):
        inner = _Recorder()
        _call(BearerAuth(inner, TOKEN), _http({"authorization": f"bearer {TOKEN}"}))
        assert inner.seen == ["http"]

    @pytest.mark.parametrize(
        "headers",
        [
            {},
            {"authorization": "Bearer wrong-token-123"},
            {"authorization": f"Basic {TOKEN}"},
            {"authorization": TOKEN},
            {"authorization": "Bearer "},
            {"authorization": f"Bearer {TOKEN}x"},
            {"authorization": f"Bearer {TOKEN[:-1]}"},
        ],
    )
    def test_anything_else_is_a_401_and_never_reaches_the_app(self, headers):
        inner = _Recorder()
        sent = _call(BearerAuth(inner, TOKEN), _http(headers))
        assert inner.seen == []
        assert sent[0]["status"] == 401
        assert (b"www-authenticate", b'Bearer realm="glplot-mcp"') in sent[0]["headers"]
        assert b"unauthorized" in sent[1]["body"] and TOKEN.encode() not in sent[1]["body"]

    def test_lifespan_passes_through_so_the_sdk_can_start_its_session_manager(self):
        inner = _Recorder()
        _call(BearerAuth(inner, TOKEN), {"type": "lifespan"})
        assert inner.seen == ["lifespan"]

    def test_a_websocket_without_the_token_is_closed(self):
        inner = _Recorder()
        sent = _call(BearerAuth(inner, TOKEN), {"type": "websocket", "headers": []})
        assert sent == [{"type": "websocket.close", "code": 1008}] and inner.seen == []


class TestCommandLine:
    def test_a_network_bind_without_a_token_is_refused_before_anything_starts(
        self, monkeypatch, capsys
    ):
        monkeypatch.delenv(TOKEN_ENV, raising=False)
        started = []
        monkeypatch.setattr(cli, "_warm_up", lambda: started.append("warm-up"))
        assert cli.main(["--transport", "streamable-http", "--host", "0.0.0.0"]) == 2
        err = capsys.readouterr().err
        assert "refusing to listen on 0.0.0.0" in err and TOKEN_ENV in err and "--insecure" in err
        assert started == []

    def test_a_weak_token_is_a_readable_error(self, capsys):
        code = cli.main(["--transport", "streamable-http", "--host", "0.0.0.0", "--token", "abc"])
        assert code == 2 and "at least 8" in capsys.readouterr().err

    def test_auth_options_make_no_sense_over_stdio(self, capsys):
        for option in (["--token", TOKEN], ["--allow-host", "x.example"], ["--insecure"]):
            assert cli.main(option) == 2
            assert "streamable-http only" in capsys.readouterr().err

    def test_loopback_needs_no_token(self, monkeypatch, capsys):
        monkeypatch.delenv(TOKEN_ENV, raising=False)
        monkeypatch.setattr(cli, "_sdk_version", lambda: None)  # stops after the checks
        assert cli.main(["--transport", "streamable-http", "--host", "127.0.0.1"]) == 2
        assert "needs the official 'mcp' package" in capsys.readouterr().err  # not "refusing"

    def test_the_options_are_documented_in_help(self, capsys):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["--help"])
        text = capsys.readouterr().out
        for word in ("--token", "GLPLOT_MCP_TOKEN", "--allow-host", "--insecure"):
            assert word in text


class TestSecuritySettings:
    @pytest.fixture(autouse=True)
    def _needs_the_sdk(self):
        pytest.importorskip("mcp")

    def test_loopback_without_a_list_keeps_the_sdk_default(self):
        from glplot.mcp.http import security_settings

        assert security_settings("127.0.0.1", [], False) is None

    def test_a_network_bind_behind_a_token_turns_the_host_check_off(self):
        from glplot.mcp.http import security_settings

        assert security_settings("0.0.0.0", [], True).enable_dns_rebinding_protection is False

    def test_an_allow_list_turns_it_on_for_exactly_those_names_and_localhost(self):
        from glplot.mcp.http import security_settings

        settings = security_settings("0.0.0.0", ["plots.lan"], True)
        assert settings.enable_dns_rebinding_protection is True
        assert {"plots.lan", "plots.lan:*", "localhost", "localhost:*"} <= set(
            settings.allowed_hosts
        )
        assert "http://plots.lan:*" in settings.allowed_origins
        assert "evil.example" not in settings.allowed_hosts


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class _Server:
    def __init__(self, *arguments, env=None):
        pytest.importorskip("mcp")
        self.port = _free_port()
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-W",
                "ignore",
                "-m",
                "glplot.mcp",
                "--transport",
                "streamable-http",
                "--port",
                str(self.port),
                *arguments,
            ],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "MPLBACKEND": "Agg", **(env or {})},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.url = f"http://127.0.0.1:{self.port}/mcp"
        self._wait()

    def _wait(self):
        import httpx

        deadline = time.time() + 90
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise AssertionError(f"server exited early: {self.process.stderr.read()}")
            try:
                httpx.get(self.url, timeout=1.0)
                return
            except httpx.TransportError:
                time.sleep(0.25)
        raise AssertionError("server did not start")

    def post(self, headers=None, body=INIT):
        import httpx

        return httpx.post(self.url, json=body, headers={**HEADERS, **(headers or {})}, timeout=30)

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.log = self.process.stderr.read()


@pytest.fixture
def open_server():
    server = _Server()
    yield server
    server.close()


@pytest.fixture
def secured_server():
    # Bound to every interface (still reachable on loopback), which is what needs the token.
    server = _Server("--host", "0.0.0.0", env={TOKEN_ENV: TOKEN})
    yield server
    server.close()


@pytest.fixture
def allow_list_server():
    server = _Server("--allow-host", "plots.lan")
    yield server
    server.close()


class TestRealServer:
    def test_loopback_works_without_a_token_and_the_official_client_can_use_it(self, open_server):
        async def go():
            try:
                from mcp.client.streamable_http import streamablehttp_client as connect
            except ImportError:
                from mcp.client.streamable_http import streamable_http_client as connect
            from mcp import ClientSession

            async with connect(open_server.url) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    info = await session.call_tool("server_info", {})
                    return listed, info

        listed, info = asyncio.run(asyncio.wait_for(go(), timeout=90))
        assert len(listed.tools) >= 24
        assert "GLPlot server information" in info.content[0].text

    def test_the_token_is_required_and_the_right_one_is_accepted(self, secured_server):
        assert secured_server.post().status_code == 401
        assert secured_server.post({"Authorization": "Bearer nope-nope-nope"}).status_code == 401
        assert secured_server.post({"Authorization": f"Bearer {TOKEN}"}).status_code == 200

    def test_a_valid_token_works_with_the_official_client(self, secured_server):
        async def go():
            try:
                from mcp.client.streamable_http import streamablehttp_client as connect
            except ImportError:
                from mcp.client.streamable_http import streamable_http_client as connect
            import httpx
            from mcp import ClientSession

            headers = {"Authorization": f"Bearer {TOKEN}"}
            try:
                context = connect(secured_server.url, headers=headers)
            except TypeError:  # mcp 2.x takes a configured httpx client instead
                context = connect(
                    secured_server.url, http_client=httpx.AsyncClient(headers=headers)
                )
            async with context as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    return await session.list_tools()

        assert len(asyncio.run(asyncio.wait_for(go(), timeout=90)).tools) >= 24

    def test_a_network_host_name_is_accepted_behind_a_token(self, secured_server):
        """Used to be a 421 on mcp 1.x: any Host other than localhost was refused."""
        reply = secured_server.post(
            {"Authorization": f"Bearer {TOKEN}", "Host": "plots.somewhere.lan:8765"}
        )
        assert reply.status_code == 200

    def test_the_host_still_needs_the_token_whatever_it_is_called(self, secured_server):
        assert secured_server.post({"Host": "plots.somewhere.lan"}).status_code == 401

    def test_the_allow_list_accepts_its_names_and_refuses_the_rest(self, allow_list_server):
        assert allow_list_server.post({"Host": "plots.lan:8765"}).status_code == 200
        assert allow_list_server.post({"Host": "localhost"}).status_code == 200
        assert allow_list_server.post({"Host": "evil.example"}).status_code == 421

    def test_the_token_is_never_written_to_the_log(self, secured_server):
        secured_server.post({"Authorization": f"Bearer {TOKEN}"})
        secured_server.close()
        assert TOKEN not in secured_server.log
