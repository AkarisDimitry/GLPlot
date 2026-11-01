"""The tool schemas, exported from the typed server into ``tools.json``.

The official-SDK server (``server.py``) describes every tool with a typed signature and a
per-argument description; that text is the model's manual. The built-in server cannot import the
SDK (that is its whole point), so it serves the *same* descriptions from ``tools.json``, a snapshot
of what an MCP client sees from the SDK server. A test fails when the snapshot is stale, so the two
servers cannot drift apart.

Regenerate after changing a tool's signature or description::

    python -m glplot.mcp.schema --write        # needs the optional ``mcp`` package
    python -m glplot.mcp.schema --check        # exit 1 when tools.json is stale

Loading the snapshot (:func:`load_tools`) needs only the standard library.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import json
import sys
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["export_tools", "load_tools", "main", "normalise"]

#: ``tools.json`` lives beside this module and is shipped in the wheel.
SCHEMA_FILE = "tools.json"

_HINTS = ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")


def _strip_titles(value: Any, names: bool = False) -> Any:
    """Drop the ``title`` keys pydantic adds to every schema node (they only repeat the name).

    Also puts ``type`` first in each node: the two SDK generations emit the same schema with
    ``type`` in different places, and the snapshot must not depend on which one wrote it.

    ``names`` marks a ``properties`` mapping, whose keys are *argument names*: an argument called
    ``title`` (``create_figure`` has one) is data, not an annotation, and must survive.
    """
    if isinstance(value, dict):
        if names:
            return {key: _strip_titles(item) for key, item in value.items()}
        items = {
            key: _strip_titles(item, names=(key == "properties"))
            for key, item in value.items()
            if key != "title"
        }
        if "type" in items:
            items = {"type": items.pop("type"), **items}
        return items
    if isinstance(value, list):
        return [_strip_titles(item) for item in value]
    return value


def _get(obj: Any, *names: str) -> Any:
    """First attribute present: the SDK's 1.x and 2.x result types spell them differently."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def normalise(tool: Any) -> Dict[str, Any]:
    """One SDK ``Tool`` as the plain dict ``tools.json`` stores."""
    hints = _get(tool, "annotations")
    annotations = {
        hint: bool(_get(hints, hint, _snake(hint))) if hints is not None else False
        for hint in _HINTS
    }
    return {
        "name": tool.name,
        "description": inspect.cleandoc(tool.description or ""),
        "inputSchema": _strip_titles(_get(tool, "input_schema", "inputSchema")),
        "annotations": annotations,
    }


def _snake(camel: str) -> str:
    return "".join(f"_{c.lower()}" if c.isupper() else c for c in camel)


@contextlib.asynccontextmanager
async def _connect(server: Any):
    """An in-process client of ``server``, on whichever SDK generation is installed."""
    try:
        from mcp import Client  # mcp >= 2
    except ImportError:
        from mcp.shared.memory import create_connected_server_and_client_session

        async with create_connected_server_and_client_session(server._mcp_server) as session:
            yield session
    else:
        async with Client(server) as client:
            yield client


def export_tools() -> List[Dict[str, Any]]:
    """Ask the real SDK server for its tools, exactly as a client would. Needs the ``mcp`` SDK."""
    from .server import build_server
    from .toolbox import Toolbox

    box = Toolbox()
    server = build_server(box)

    async def go() -> List[Any]:
        async with _connect(server) as client:
            return list((await client.list_tools()).tools)

    try:
        return [normalise(tool) for tool in asyncio.run(go())]
    finally:
        box.close_all()


@lru_cache(maxsize=1)
def load_tools() -> List[Dict[str, Any]]:
    """The shipped snapshot. Standard library only."""
    text = (resources.files("glplot.mcp") / SCHEMA_FILE).read_text(encoding="utf-8")
    return json.loads(text)


def _render(tools: List[Dict[str, Any]]) -> str:
    return json.dumps(tools, indent=1, ensure_ascii=False) + "\n"


def _path() -> Path:
    return Path(__file__).with_name(SCHEMA_FILE)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m glplot.mcp.schema", description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="regenerate tools.json")
    group.add_argument("--check", action="store_true", help="exit 1 if tools.json is stale")
    args = parser.parse_args(argv)
    try:
        fresh = _render(export_tools())
    except ImportError as exc:
        print(f"glplot.mcp.schema: needs the optional 'mcp' package ({exc})", file=sys.stderr)
        return 2
    path = _path()
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if args.write:
        path.write_text(fresh, encoding="utf-8")
        print(f"wrote {path} ({len(json.loads(fresh))} tools)")
        return 0
    if fresh != current:
        print(
            "tools.json is stale: run `python -m glplot.mcp.schema --write` and commit it.",
            file=sys.stderr,
        )
        return 1
    print("tools.json is up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
