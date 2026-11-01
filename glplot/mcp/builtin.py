"""A dependency-free MCP server over stdio.

``pip install glplot`` must be enough for a language model to use GLPlot, and the official ``mcp``
SDK is an optional extra (it pulls in a web stack a plotting library should not impose). This
module is what runs when the SDK is absent: a small, standard-library implementation of the part of
the Model Context Protocol an agent client uses -- ``initialize``, ``tools/*``, ``resources/*``,
``prompts/*`` and ``ping`` -- over newline-delimited JSON-RPC on stdin/stdout.

It serves *exactly* what :mod:`glplot.mcp.server` serves: the tool schemas come from the
``tools.json`` snapshot of that server, the resources and prompts from :mod:`glplot.mcp.catalog`,
and every call goes to the same :class:`~glplot.mcp.toolbox.Toolbox`. Only the transport is
different. Streamable HTTP and the other optional protocol corners (sampling, subscriptions,
completion) need the SDK; ``glplot[mcp]`` installs it and ``python -m glplot.mcp`` then prefers it.

stdout *is* the protocol channel, so the process-wide ``sys.stdout`` is pointed at stderr for the
whole session; only this module writes to the real stdout.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import sys
from typing import IO, Any, Dict, List, Optional, Tuple

from . import catalog
from .guide import build_instructions
from .schema import load_tools
from .state import DEFAULT_LIMITS, FileSandbox, Limits, McpError
from .toolbox import Toolbox, ToolResult

__all__ = ["BuiltinServer", "PROTOCOL_VERSIONS", "serve_stdio", "validate_arguments"]

logger = logging.getLogger(__name__)

#: Protocol revisions this server speaks, oldest first. A client asking for one of them gets it;
#: anything else is answered with the newest, as the specification says, and the client decides.
PROTOCOL_VERSIONS: Tuple[str, ...] = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")

# JSON-RPC 2.0 error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
RESOURCE_NOT_FOUND = -32002


# ----------------------------------------------------------------------------------
# Argument validation: the slice of JSON Schema the tool schemas use
# ----------------------------------------------------------------------------------

_TYPE_NAMES = {
    "string": "text",
    "number": "a number",
    "integer": "an integer",
    "boolean": "true or false",
    "array": "a list",
    "object": "an object",
    "null": "null",
}


class _Invalid(Exception):
    """A value does not satisfy its schema; the message says where and why."""


def _describe(value: Any) -> str:
    text = json.dumps(value, default=str)
    return text if len(text) <= 40 else text[:37] + "..."


def _type_ok(kind: str, value: Any) -> bool:
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and value.is_integer()
        )
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    return True


def _coerce(kind: str, value: Any) -> Any:
    """The lenient reading a model's sloppy-but-clear argument deserves (``"5"`` for 5).

    The official SDK validates in pydantic's lax mode and pre-parses JSON that arrived as a
    string; clients (and models) rely on that, so the built-in server accepts the same.
    """
    if isinstance(value, str):
        text = value.strip()
        if kind in ("integer", "number"):
            try:
                number = float(text)
            except ValueError:
                return value
            return int(number) if kind == "integer" and number.is_integer() else number
        if kind == "boolean" and text.lower() in ("true", "false"):
            return text.lower() == "true"
        if kind in ("array", "object") and text[:1] in "[{":
            try:
                return json.loads(text)
            except ValueError:
                return value
        if kind == "null" and text.lower() in ("null", "none"):
            return None
    if kind == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _validate(schema: Dict[str, Any], value: Any, path: str, strict: bool) -> Any:
    """``value`` checked against ``schema`` (and coerced unless ``strict``); raises ``_Invalid``."""
    if "anyOf" in schema:
        errors: List[str] = []
        for option in schema["anyOf"]:
            try:
                return _validate(option, value, path, strict)
            except _Invalid as exc:
                errors.append(str(exc))
        raise _Invalid(
            f"{path}: {_describe(value)} matches none of the allowed forms ({'; '.join(errors)})"
        )
    kind = schema.get("type")
    if kind is not None:
        kinds = [kind] if isinstance(kind, str) else list(kind)
        if not any(_type_ok(k, value) for k in kinds):
            if strict:
                raise _Invalid(
                    f"{path}: expected {' or '.join(_TYPE_NAMES.get(k, k) for k in kinds)}"
                )
            for k in kinds:
                converted = _coerce(k, value)
                if _type_ok(k, converted):
                    value = converted
                    break
            else:
                raise _Invalid(
                    f"{path}: expected {' or '.join(_TYPE_NAMES.get(k, k) for k in kinds)}, "
                    f"got {_describe(value)}"
                )
        elif "integer" in kinds and isinstance(value, float):
            value = int(value)
    if "enum" in schema and value not in schema["enum"]:
        raise _Invalid(f"{path}: must be one of {schema['enum']}, got {_describe(value)}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        for key, test, word in (
            ("minimum", lambda v, b: v >= b, "at least"),
            ("maximum", lambda v, b: v <= b, "at most"),
            ("exclusiveMinimum", lambda v, b: v > b, "greater than"),
            ("exclusiveMaximum", lambda v, b: v < b, "less than"),
        ):
            if key in schema and not test(value, schema[key]):
                raise _Invalid(f"{path}: must be {word} {schema[key]}, got {value}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise _Invalid(f"{path}: must have at least {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise _Invalid(f"{path}: must have at most {schema['maxLength']} characters")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise _Invalid(f"{path}: needs at least {schema['minItems']} items, got {len(value)}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise _Invalid(f"{path}: allows at most {schema['maxItems']} items, got {len(value)}")
        if "items" in schema:
            value = [
                _validate(schema["items"], item, f"{path}[{index}]", strict)
                for index, item in enumerate(value)
            ]
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        extra = schema.get("additionalProperties")
        out: Dict[str, Any] = {}
        for key, item in value.items():
            child = f"{path}.{key}" if path else key
            if key in properties:
                out[key] = _validate(properties[key], item, child, strict)
            elif isinstance(extra, dict):
                out[key] = _validate(extra, item, child, strict)
            else:
                out[key] = item
        value = out
    return value


def validate_arguments(
    schema: Dict[str, Any], arguments: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Check a tool call's ``arguments`` against its ``inputSchema`` and return the cleaned dict.

    Missing required arguments, unknown names and out-of-range values raise ``McpError`` with the
    argument named; values a model typed loosely (``"5"``, ``"true"``, a JSON list in a string)
    are accepted, as they are by the official SDK.
    """
    arguments = {} if arguments is None else arguments
    if not isinstance(arguments, dict):
        raise McpError("arguments must be an object of name -> value.")
    properties = schema.get("properties") or {}
    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        raise McpError(
            f"Unexpected argument(s): {', '.join(unknown)}. "
            f"Valid: {', '.join(properties) or 'none'}."
        )
    missing = [name for name in schema.get("required") or [] if name not in arguments]
    if missing:
        raise McpError(f"Missing required argument(s): {', '.join(missing)}.")
    cleaned: Dict[str, Any] = {}
    for name, value in arguments.items():
        spec = properties[name]
        try:
            # Exact types first: a string must stay a column name, not be read as a JSON list.
            cleaned[name] = _validate(spec, value, name, strict=True)
        except _Invalid:
            try:
                cleaned[name] = _validate(spec, value, name, strict=False)
            except _Invalid as exc:
                raise McpError(str(exc)) from None
    return cleaned


# ----------------------------------------------------------------------------------
# The server
# ----------------------------------------------------------------------------------


def _content(result: ToolResult) -> List[Dict[str, Any]]:
    """The reply to a tool call: the picture first when there is one, then the text."""
    blocks: List[Dict[str, Any]] = []
    if result.image is not None:
        blocks.append(
            {
                "type": "image",
                "data": base64.b64encode(result.image).decode("ascii"),
                "mimeType": f"image/{result.image_format}",
            }
        )
    blocks.append({"type": "text", "text": result.text()})
    return blocks


class _RpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code, self.message, self.data = code, message, data


class BuiltinServer:
    """Protocol logic for one session; no I/O of its own (see :func:`serve_stdio`)."""

    def __init__(
        self,
        toolbox: Optional[Toolbox] = None,
        *,
        sandbox: Optional[FileSandbox] = None,
        limits: Limits = DEFAULT_LIMITS,
        name: str = "glplot",
    ) -> None:
        self.box = toolbox if toolbox is not None else Toolbox(sandbox, limits)
        self.box.server_flavor = "builtin"
        self.name = name
        self.tools = {tool["name"]: tool for tool in load_tools()}
        self.initialized = False

    # -- dispatch -------------------------------------------------------------------

    def handle(self, message: Any) -> Optional[Any]:
        """Answer one decoded message (or batch). ``None`` means no reply is owed."""
        if isinstance(message, list):
            replies = [r for r in (self.handle(item) for item in message) if r is not None]
            return replies or None
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return self._error(
                message.get("id") if isinstance(message, dict) else None,
                INVALID_REQUEST,
                "Not a JSON-RPC 2.0 message.",
            )
        if "method" not in message:
            return None  # a response to a request we never sent
        method, params = message["method"], message.get("params") or {}
        if "id" not in message:
            self._notification(method, params)
            return None
        request_id = message["id"]
        try:
            handler = self._METHODS.get(method)
            if handler is None:
                raise _RpcError(METHOD_NOT_FOUND, f"Method not found: {method}")
            if not isinstance(params, dict):
                raise _RpcError(INVALID_PARAMS, "params must be an object")
            return {"jsonrpc": "2.0", "id": request_id, "result": handler(self, params)}
        except _RpcError as exc:
            return self._error(request_id, exc.code, exc.message, exc.data)
        except Exception as exc:  # noqa: BLE001 - one bad request must not end the session
            logger.exception("request %r failed", method)
            return self._error(request_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _error(request_id: Any, code: int, message: str, data: Any = None) -> Dict[str, Any]:
        error: Dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return {"jsonrpc": "2.0", "id": request_id, "error": error}

    def _notification(self, method: str, params: Dict[str, Any]) -> None:
        if method == "notifications/initialized":
            self.initialized = True
        # notifications/cancelled and everything else: nothing to do, nothing to reply.

    # -- lifecycle ------------------------------------------------------------------

    def _initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        wanted = params.get("protocolVersion")
        version = wanted if wanted in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[-1]
        import glplot

        return {
            "protocolVersion": version,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": {"name": self.name, "version": glplot.__version__},
            "instructions": build_instructions(self.box.sandbox),
        }

    def _ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    # -- tools ----------------------------------------------------------------------

    def _tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "tools": [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "inputSchema": tool["inputSchema"],
                    "annotations": tool["annotations"],
                }
                for tool in self.tools.values()
            ]
        }

    def _tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        tool = self.tools.get(name) if isinstance(name, str) else None
        if tool is None:
            raise _RpcError(INVALID_PARAMS, f"Unknown tool: {name!r}")
        try:
            arguments = validate_arguments(tool["inputSchema"], params.get("arguments"))
            result = getattr(self.box, name)(**arguments)
            return {"content": _content(result), "isError": False}
        except McpError as exc:
            # A tool's own failure is a *result* the model reads and acts on, not a protocol error.
            return {
                "content": [{"type": "text", "text": f"Error executing tool {name}: {exc}"}],
                "isError": True,
            }

    # -- resources ------------------------------------------------------------------

    def _resources_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "resources": [
                {
                    "uri": d.uri,
                    "name": d.name,
                    "description": d.description,
                    "mimeType": d.mime_type,
                }
                for d in catalog.RESOURCES
                if not d.is_template
            ]
        }

    def _resource_templates_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "resourceTemplates": [
                {
                    "uriTemplate": d.uri,
                    "name": d.name,
                    "description": d.description,
                    "mimeType": d.mime_type,
                }
                for d in catalog.RESOURCES
                if d.is_template
            ]
        }

    def _resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        uri = params.get("uri")
        if not isinstance(uri, str):
            raise _RpcError(INVALID_PARAMS, "resources/read needs a 'uri'")
        found = catalog.find_resource(uri)
        if found is None:
            raise _RpcError(RESOURCE_NOT_FOUND, f"Unknown resource: {uri}", {"uri": uri})
        definition, values = found
        try:
            body = definition.read(self.box, values)
        except McpError as exc:
            raise _RpcError(INVALID_PARAMS, str(exc), {"uri": uri}) from None
        entry: Dict[str, Any] = {"uri": uri, "mimeType": definition.mime_type}
        if isinstance(body, bytes):
            entry["blob"] = base64.b64encode(body).decode("ascii")
        else:
            entry["text"] = body
        return {"contents": [entry]}

    # -- prompts --------------------------------------------------------------------

    def _prompts_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "prompts": [
                {
                    "name": p.name,
                    "description": p.description,
                    "arguments": [
                        {"name": a.name, "description": a.description, "required": a.required}
                        for a in p.arguments
                    ],
                }
                for p in catalog.PROMPTS
            ]
        }

    def _prompts_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        try:
            definition = catalog.find_prompt(str(name))
            text = definition.build(**(params.get("arguments") or {}))
        except McpError as exc:
            raise _RpcError(INVALID_PARAMS, str(exc)) from None
        return {
            "description": definition.description,
            "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
        }

    _METHODS = {
        "initialize": _initialize,
        "ping": _ping,
        "tools/list": _tools_list,
        "tools/call": _tools_call,
        "resources/list": _resources_list,
        "resources/templates/list": _resource_templates_list,
        "resources/read": _resources_read,
        "prompts/list": _prompts_list,
        "prompts/get": _prompts_get,
    }


def serve_stdio(
    server: BuiltinServer,
    stdin: Optional[IO[bytes]] = None,
    stdout: Optional[IO[bytes]] = None,
) -> int:
    """Run ``server`` over newline-delimited JSON-RPC until stdin closes. Returns an exit code."""
    raw_in = stdin if stdin is not None else sys.stdin.buffer
    raw_out = stdout if stdout is not None else sys.stdout.buffer
    reader = io.TextIOWrapper(raw_in, encoding="utf-8", newline="\n")
    writer = io.TextIOWrapper(raw_out, encoding="utf-8", newline="\n", write_through=True)
    # Whatever a library below prints must not reach the protocol channel.
    saved_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError as exc:
                reply: Any = BuiltinServer._error(None, PARSE_ERROR, f"Invalid JSON: {exc}")
            else:
                reply = server.handle(message)
            if reply is not None:
                writer.write(json.dumps(reply, separators=(",", ":"), ensure_ascii=False) + "\n")
                writer.flush()
    finally:
        sys.stdout = saved_stdout
        server.box.close_all()
        # The wrappers must not close the process's real streams when they are collected.
        reader.detach()
        writer.detach()
    return 0
