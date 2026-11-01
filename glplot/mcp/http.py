"""The streamable-HTTP transport: bearer-token auth and host checks around the SDK's ASGI app.

The official SDK serves MCP over HTTP with no authentication, and (in its 1.x generation) rejects
any ``Host`` header that is not ``localhost`` -- so a server bound to a network interface was either
unreachable or open to everyone on the network. This module builds the SDK's ASGI app itself so
it can

* require ``Authorization: Bearer <token>`` on every request (``--token`` /
  ``GLPLOT_MCP_TOKEN``), compared in constant time and never logged;
* accept the ``Host`` names the operator lists (``--allow-host``) -- or, behind a token, any;
* refuse to bind a non-loopback address with neither a token nor an explicit ``--insecure``.

It needs the optional ``mcp`` SDK (and so ``uvicorn``, which the SDK depends on). Importing this
module does not: the SDK and uvicorn are imported when a server is actually started.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import os
from typing import Any, Awaitable, Callable, Iterable, List, Optional, Sequence

__all__ = [
    "BearerAuth",
    "TOKEN_ENV",
    "is_loopback",
    "resolve_token",
    "security_settings",
    "serve_http",
]

logger = logging.getLogger("glplot.mcp")

#: The environment variable the token may come from. Preferred over ``--token``, which is visible in
#: the process list.
TOKEN_ENV = "GLPLOT_MCP_TOKEN"

_LOOPBACK_NAMES = {"localhost", "ip6-localhost"}

Scope = dict
Receive = Callable[[], Awaitable[dict]]
Send = Callable[[dict], Awaitable[None]]


def is_loopback(host: str) -> bool:
    """True for an address only this machine can reach (``127.0.0.1``, ``::1``, ``localhost``)."""
    name = str(host).strip().strip("[]").lower()
    if name in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def resolve_token(option: Optional[str]) -> Optional[str]:
    """The bearer token from ``--token``, else the environment; None when auth is off."""
    token = option if option else os.environ.get(TOKEN_ENV)
    if token is None:
        return None
    token = token.strip()
    if not token:
        return None
    if len(token) < 8:
        raise ValueError(
            f"The token is only {len(token)} characters; use at least 8 (e.g. "
            '`python -c "import secrets; print(secrets.token_urlsafe(24))"`).'
        )
    return token


class BearerAuth:
    """ASGI middleware: 401 unless the request carries the right ``Authorization: Bearer`` token.

    Lifespan events pass straight through (the SDK's session manager starts there); only HTTP and
    WebSocket requests are checked. The comparison is constant-time.
    """

    def __init__(self, app: Callable[..., Awaitable[None]], token: str) -> None:
        self.app = app
        self._token = token.encode("utf-8")

    def _authorised(self, scope: Scope) -> bool:
        for name, value in scope.get("headers", ()):
            if name == b"authorization":
                kind, _, presented = value.partition(b" ")
                return kind.lower() == b"bearer" and hmac.compare_digest(
                    presented.strip(), self._token
                )
        return False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket") and not self._authorised(scope):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            body = json.dumps(
                {"error": "unauthorized", "hint": "send Authorization: Bearer <token>"}
            )
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"www-authenticate", b'Bearer realm="glplot-mcp"'),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body.encode("utf-8")})
            return
        await self.app(scope, receive, send)


def security_settings(host: str, allow_hosts: Sequence[str], has_token: bool) -> Optional[Any]:
    """The SDK's DNS-rebinding settings for this bind, or None to leave its default alone.

    * Loopback bind, no ``--allow-host``: None (the SDK's own localhost protection stays).
    * ``--allow-host`` given: protection on, accepting localhost and the names listed (each also
      with any port).
    * Network bind behind a token and no list: protection off. A rebinding attack needs a browser
      that holds no secret; every request here must present the token.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    if allow_hosts:
        names: List[str] = ["localhost", "127.0.0.1", "[::1]"]
        for entry in allow_hosts:
            entry = entry.strip()
            if entry and entry not in names:
                names.append(entry)
        hosts = list(dict.fromkeys(n for name in names for n in (name, f"{name}:*")))
        origins = [f"http://{h}" for h in hosts]
        return TransportSecuritySettings(
            enable_dns_rebinding_protection=True, allowed_hosts=hosts, allowed_origins=origins
        )
    if is_loopback(host):
        return None
    return TransportSecuritySettings(enable_dns_rebinding_protection=False)


def build_app(
    server: Any,
    host: str,
    port: int,
    *,
    token: Optional[str] = None,
    allow_hosts: Iterable[str] = (),
) -> Any:
    """The SDK's streamable-HTTP ASGI app for ``server``, behind bearer auth when a token is set."""
    security = security_settings(host, list(allow_hosts), token is not None)
    try:  # mcp 2.x takes the bind address and the security settings here
        app = server.streamable_http_app(host=host, transport_security=security)
    except TypeError:  # mcp 1.x keeps them in the settings object
        server.settings.host = host
        server.settings.port = port
        if security is not None:
            server.settings.transport_security = security
        app = server.streamable_http_app()
    return BearerAuth(app, token) if token else app


def serve_http(
    server: Any,
    host: str,
    port: int,
    *,
    token: Optional[str] = None,
    allow_hosts: Iterable[str] = (),
) -> None:
    """Serve ``server`` over streamable HTTP until interrupted."""
    import uvicorn

    app = build_app(server, host, port, token=token, allow_hosts=allow_hosts)
    uvicorn.run(app, host=host, port=port, log_level="info")
