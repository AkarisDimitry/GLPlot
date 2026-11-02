"""The ``glplot-mcp`` / ``python -m glplot.mcp`` command line. No SDK needed to import it.

* ``glplot-mcp [serve] [--root DIR] ...`` -- run the server. With the optional ``mcp`` package
  installed the official-SDK server is used (stdio or streamable HTTP); without it the built-in
  stdio server in :mod:`glplot.mcp.builtin` runs instead, so ``pip install glplot`` alone is enough
  for an MCP client to connect. ``--builtin`` / ``--sdk`` force one.
* ``glplot-mcp config`` -- print (or ``--write``) the settings that register the server in Claude
  Code, Claude Desktop, Cursor, VS Code or Codex, with this interpreter's absolute path.
* ``glplot-mcp doctor`` -- check the install and run a protocol self-test.
* ``glplot-mcp tools`` -- list the tools.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .state import FileSandbox, Limits

__all__ = ["build_parser", "main"]

logger = logging.getLogger("glplot.mcp")

SUBCOMMANDS = ("serve", "config", "doctor", "tools")
_TOKEN_ENV = "GLPLOT_MCP_TOKEN"
CLIENTS = ("claude-code", "claude-desktop", "cursor", "vscode", "codex", "generic")


# ----------------------------------------------------------------------------------
# serve
# ----------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Arguments of ``serve`` (also what a bare ``glplot-mcp --root DIR`` means)."""
    parser = argparse.ArgumentParser(
        prog="glplot-mcp",
        description="Run GLPlot as a Model Context Protocol server so an LLM can make plots. "
        "Other commands: config, doctor, tools (see `glplot-mcp config --help`).",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory the model may read data from and save images into (repeatable). "
        "Default: none, which disables file access.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="stdio (default) for a client that launches this process; streamable-http to serve "
        "(needs the official SDK: pip install 'glplot[mcp]').",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="HTTP bind address (default: localhost)."
    )
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765).")
    parser.add_argument(
        "--token",
        metavar="SECRET",
        default=None,
        help="streamable-http: require 'Authorization: Bearer SECRET' on every request. Prefer the "
        "GLPLOT_MCP_TOKEN environment variable, which does not show up in the process list. "
        "Required when --host is not localhost.",
    )
    parser.add_argument(
        "--allow-host",
        action="append",
        default=[],
        metavar="NAME",
        help="streamable-http: a Host name clients may use (repeatable), e.g. plots.example.org. "
        "Turns on the SDK's DNS-rebinding check for exactly these names plus localhost.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="streamable-http: allow a non-localhost --host with no token (anyone who can reach "
        "the port can use the server).",
    )
    parser.add_argument("--max-figures", type=int, default=Limits.max_figures)
    which = parser.add_mutually_exclusive_group()
    which.add_argument(
        "--builtin",
        action="store_true",
        help="Use the built-in dependency-free stdio server even if the mcp SDK is installed.",
    )
    which.add_argument(
        "--sdk",
        action="store_true",
        help="Require the official mcp SDK (error if it is not installed).",
    )
    parser.add_argument("--version", action="store_true", help="Print the versions and exit.")
    return parser


def _sdk_version() -> Optional[str]:
    """The installed official SDK's version, or None when it cannot be imported."""
    import importlib.metadata as metadata

    try:
        version = metadata.version("mcp")
        import mcp  # noqa: F401
        import pydantic  # noqa: F401
    except (metadata.PackageNotFoundError, ImportError):
        return None
    return version


def _versions() -> str:
    import glplot

    sdk = _sdk_version() or "not installed (built-in server)"
    return f"glplot {glplot.__version__}, mcp {sdk}, python {platform.python_version()}"


def _warm_up() -> None:
    """Render one throwaway figure before serving.

    The first render in a fresh environment builds matplotlib's font cache, which can take many
    seconds -- long enough that a client sees its first real tool call hang. Paying for it here,
    before the client has been told the server is ready, moves that wait to startup, and a broken
    matplotlib shows up in the log now rather than as a failed first plot. A failure is logged,
    not fatal: the tools report their own errors.
    """
    from .toolbox import Toolbox

    toolbox = Toolbox()
    try:
        figure = toolbox.create_figure(width=300, height=200).data["figure_id"]
        toolbox.add_plot(figure, "line", y=[0.0, 1.0])
        toolbox.set_axes(figure, title="warm-up")
        toolbox.render_figure(figure)
    except Exception as exc:  # noqa: BLE001 - startup must survive any renderer problem
        logger.warning("render warm-up failed (%s); the first plot may be slow or fail", exc)
    finally:
        toolbox.close_all()


def _serve(args: argparse.Namespace) -> int:
    if args.version:
        print(_versions(), file=sys.stderr)
        return 0
    if args.max_figures < 1:
        print("glplot-mcp: --max-figures must be at least 1", file=sys.stderr)
        return 2
    try:
        sandbox = FileSandbox(list(args.root))
    except ValueError as exc:
        print(f"glplot-mcp: {exc}", file=sys.stderr)
        return 2
    limits = Limits(max_figures=args.max_figures)

    token: Optional[str] = None
    if args.transport == "streamable-http":
        from .http import is_loopback, resolve_token

        try:
            token = resolve_token(args.token)
        except ValueError as exc:
            print(f"glplot-mcp: {exc}", file=sys.stderr)
            return 2
        if not is_loopback(args.host) and token is None and not args.insecure:
            print(
                f"glplot-mcp: refusing to listen on {args.host} with no authentication: anyone who "
                "can reach the port could create figures and use the allowed file roots. Set a "
                f"token (--token, or the {_TOKEN_ENV} environment variable), or pass --insecure to "
                "accept that.",
                file=sys.stderr,
            )
            return 2
    elif args.token or args.allow_host or args.insecure:
        print(
            "glplot-mcp: --token, --allow-host and --insecure apply to --transport "
            "streamable-http only; over stdio the client launches the server itself.",
            file=sys.stderr,
        )
        return 2

    use_sdk = not args.builtin and _sdk_version() is not None
    if args.sdk and not use_sdk:
        print(
            "glplot-mcp: --sdk needs the official 'mcp' package. "
            'Install it with: pip install "glplot[mcp]"',
            file=sys.stderr,
        )
        return 2
    if args.transport != "stdio" and not use_sdk:
        print(
            "glplot-mcp: --transport streamable-http needs the official 'mcp' package"
            + (" (and cannot be combined with --builtin)." if args.builtin else ".")
            + ' Install it with: pip install "glplot[mcp]"',
            file=sys.stderr,
        )
        return 2

    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="glplot-mcp: %(message)s")
    roots = ", ".join(str(r) for r in sandbox.roots) or "none (file access disabled)"
    _warm_up()
    if not use_sdk:
        from .builtin import BuiltinServer, serve_stdio

        logger.info(
            "serving over stdio with the built-in server (pip install 'glplot[mcp]' adds HTTP and "
            "the official SDK); file roots: %s",
            roots,
        )
        return serve_stdio(BuiltinServer(sandbox=sandbox, limits=limits))

    from .server import build_server

    server = build_server(sandbox=sandbox, limits=limits)
    logger.info("serving over %s with the official mcp SDK; file roots: %s", args.transport, roots)
    if args.transport == "stdio":
        server.run()
        return 0
    from .http import is_loopback, serve_http

    if not is_loopback(args.host):
        logger.warning(
            "listening on %s: %s",
            args.host,
            (
                "bearer-token auth is on"
                if token
                else "NO AUTHENTICATION (--insecure): anyone who can reach this port can use it"
            ),
        )
    logger.info(
        "streamable-http on %s:%s%s", args.host, args.port, " (auth: bearer token)" if token else ""
    )
    serve_http(server, args.host, args.port, token=token, allow_hosts=args.allow_host)
    return 0


# ----------------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------------


def _server_command(roots: Sequence[str], python: Optional[str] = None) -> Tuple[str, List[str]]:
    """``(command, args)`` that start this server with the interpreter that has GLPlot installed."""
    args = ["-m", "glplot.mcp"]
    for root in roots:
        args += ["--root", str(Path(root).expanduser().resolve())]
    return python or sys.executable, args


def _on_windows() -> bool:
    """Whether this is Windows. A function, so a test can pretend otherwise: patching
    ``os.name`` itself changes it for the whole process, and on Windows ``pathlib`` then builds
    every ``Path()`` as a ``PosixPath`` and fails -- pytest's own reporting included."""
    return os.name == "nt"


def _shell_join(parts: Sequence[str]) -> str:
    if _on_windows():
        return subprocess.list2cmdline(list(parts))
    return " ".join(shlex.quote(p) for p in parts)


def _claude_desktop_path() -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if _on_windows():
        return (
            Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
            / "Claude"
            / ("claude_desktop_config.json")
        )
    return home / ".config" / "Claude" / "claude_desktop_config.json"


def client_config(client: str, name: str, command: str, args: List[str]) -> Dict[str, Any]:
    """What to install for ``client``: a ``text`` to show, and (for JSON files) the ``path`` and
    ``data`` to merge. Pure, so it is easy to test."""
    entry = {"command": command, "args": args}
    if client == "claude-code":
        line = _shell_join(["claude", "mcp", "add", name, "--", command, *args])
        return {
            "client": client,
            "text": f"{line}\n\n# add --scope user to make it available in every project",
        }
    if client in ("claude-desktop", "cursor"):
        path = (
            _claude_desktop_path()
            if client == "claude-desktop"
            else Path.home() / ".cursor" / "mcp.json"
        )
        data = {"mcpServers": {name: entry}}
        return {
            "client": client,
            "path": path,
            "data": data,
            "key": "mcpServers",
            "text": f"# {path}\n" + json.dumps(data, indent=2),
        }
    if client == "vscode":
        path = Path(".vscode") / "mcp.json"
        data = {"servers": {name: {"type": "stdio", **entry}}}
        return {
            "client": client,
            "path": path,
            "data": data,
            "key": "servers",
            "text": f"# {path} (in the workspace)\n" + json.dumps(data, indent=2),
        }
    if client == "codex":
        array = ", ".join(json.dumps(a) for a in args)
        return {
            "client": client,
            "text": f"# ~/.codex/config.toml\n[mcp_servers.{name}]\n"
            f"command = {json.dumps(command)}\n"
            f"args = [{array}]",
        }
    return {"client": "generic", "text": json.dumps({name: entry}, indent=2)}


def write_json_config(config: Dict[str, Any], name: str) -> Path:
    """Merge ``config`` into its JSON file, keeping a ``.bak``; refuses a file it cannot parse."""
    path: Path = config["path"]
    key: str = config["key"]
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8") or "{}")
        except ValueError as exc:
            raise ValueError(f"{path} is not valid JSON ({exc}); fix or remove it first") from exc
        if not isinstance(existing, dict):
            raise ValueError(f"{path} does not contain a JSON object")
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    servers = existing.setdefault(key, {})
    if not isinstance(servers, dict):
        raise ValueError(f"{path}: '{key}' is not an object")
    servers[name] = config["data"][key][name]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def _config(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="glplot-mcp config",
        description="Show how to register the GLPlot MCP server in an MCP client.",
    )
    parser.add_argument("--client", choices=CLIENTS + ("all",), default="all")
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory the model may read/write (repeatable). Default: none.",
    )
    parser.add_argument(
        "--name", default="glplot", help="Server name in the client (default: glplot)."
    )
    parser.add_argument("--python", default=None, help="Interpreter to run (default: this one).")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Merge the entry into the client's JSON config file (claude-desktop, cursor, vscode) "
        "and keep a .bak copy.",
    )
    args = parser.parse_args(list(argv))
    command, command_args = _server_command(args.root, args.python)
    clients = CLIENTS if args.client == "all" else (args.client,)
    if args.write and args.client == "all":
        print(
            "glplot-mcp config: --write needs a single --client "
            "(claude-desktop, cursor or vscode).",
            file=sys.stderr,
        )
        return 2
    for client in clients:
        config = client_config(client, args.name, command, command_args)
        if args.write:
            if "path" not in config:
                print(
                    f"glplot-mcp config: {client} has no JSON file to write; "
                    "run the command below instead.\n" + config["text"],
                    file=sys.stderr,
                )
                return 2
            try:
                written = write_json_config(config, args.name)
            except (OSError, ValueError) as exc:
                print(f"glplot-mcp config: {exc}", file=sys.stderr)
                return 1
            print(f"Wrote {args.name!r} to {written}. Restart {client} to pick it up.")
            continue
        print(f"## {client}\n{config['text']}\n")
    if not args.write and not args.root:
        print(
            "# No --root given: the model cannot read or write files; it builds data with "
            "create_dataset. Add --root DIR to allow a directory."
        )
    return 0


# ----------------------------------------------------------------------------------
# doctor / tools
# ----------------------------------------------------------------------------------


def _protocol_selftest() -> Dict[str, Any]:
    """Drive the built-in server through initialize -> tools/list -> a real plot, in memory."""
    from .builtin import BuiltinServer

    server = BuiltinServer()
    try:

        def call(
            method: str, params: Optional[Dict[str, Any]] = None, rid: int = 1
        ) -> Dict[str, Any]:
            reply = server.handle(
                {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
            )
            if reply is None or "error" in reply:
                raise RuntimeError(f"{method}: {reply}")
            return reply["result"]

        info = call(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "doctor", "version": "0"},
            },
        )
        tools = call("tools/list")["tools"]
        figure = call(
            "tools/call", {"name": "create_figure", "arguments": {"width": 300, "height": 200}}
        )
        text = figure["content"][-1]["text"]
        figure_id = json.loads(text.split("\n", 1)[1])["figure_id"]
        call(
            "tools/call",
            {
                "name": "add_plot",
                "arguments": {"figure_id": figure_id, "kind": "line", "y": [0, 1, 4]},
            },
        )
        rendered = call(
            "tools/call", {"name": "render_figure", "arguments": {"figure_id": figure_id}}
        )
        has_image = rendered["content"][0]["type"] == "image"
        return {
            "protocol": info["protocolVersion"],
            "tools": len(tools),
            "render_returns_image": has_image,
        }
    finally:
        server.box.close_all()


def _doctor(argv: Sequence[str]) -> int:
    from .. import llm
    from ..llm.environment import format_doctor

    results = llm.doctor()

    def selftest() -> str:
        outcome = _protocol_selftest()
        if not outcome["render_returns_image"]:
            raise RuntimeError("render_figure did not return an image")
        return (
            f"built-in server answered: {outcome['tools']} tools, protocol "
            f"{outcome['protocol']}, render returned an image"
        )

    try:
        detail = selftest()
        results.append({"check": "mcp self-test", "ok": True, "detail": detail})
    except Exception as exc:  # noqa: BLE001 - report, do not raise
        results.append(
            {"check": "mcp self-test", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}
        )
    print(format_doctor(results))
    command, command_args = _server_command([])
    print(f"\nTo register with a client: {command} -m glplot.mcp config   (or: glplot-mcp config)")
    return 0 if all(item["ok"] for item in results) else 1


def _tools(argv: Sequence[str]) -> int:
    from .schema import load_tools

    for tool in load_tools():
        first = tool["description"].split("\n\n", 1)[0].replace("\n", " ")
        flags = (
            "read-only"
            if tool["annotations"]["readOnlyHint"]
            else ("destructive" if tool["annotations"]["destructiveHint"] else "writes")
        )
        print(f"{tool['name']:18} [{flags:11}] {first}")
    return 0


# ----------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``python -m glplot.mcp`` and the ``glplot-mcp`` script."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = arguments[0] if arguments and arguments[0] in SUBCOMMANDS else "serve"
    rest = arguments[1:] if arguments and arguments[0] in SUBCOMMANDS else arguments
    if command == "config":
        return _config(rest)
    if command == "doctor":
        return _doctor(rest)
    if command == "tools":
        return _tools(rest)
    return _serve(build_parser().parse_args(rest))
