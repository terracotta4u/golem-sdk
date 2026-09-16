from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator


class GolemError(Exception):
    """A Golem HTTP or protocol error.

    Attributes:
        status: HTTP status if the request reached Golem, otherwise ``None``.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class TurnEvent:
    """One SSE event from ``GET /v1/turns/{id}``.

    Attributes:
        name: Event type: ``log``, ``done``, or ``error``.
        data: JSON payload for that event.
    """

    name: str
    data: dict[str, Any]


class Client:
    """HTTP client for Golem's ``/v1`` API.

    Golem injects ``GOLEM_URL`` and ``GOLEM_TOKEN`` when it launches an
    extension. Prefer ``from_env`` over constructing this by hand.

    Attributes:
        url: Golem base URL with no trailing slash.
        token: Bearer token, or empty if Golem is running without one.
    """

    def __init__(self, url: str, token: str = "") -> None:
        """Connect to a Golem server.

        Args:
            url: Base URL, for example ``http://127.0.0.1:8743``.
            token: Bearer token. Empty skips the ``Authorization`` header.
        """
        self.url = url.rstrip("/")
        self.token = token

    @classmethod
    def from_env(cls) -> Client:
        """Build a client from ``GOLEM_URL`` and ``GOLEM_TOKEN``.

        ``GOLEM_URL`` defaults to ``http://127.0.0.1:8743`` if unset. Token
        may be empty.
        """
        url = os.environ.get("GOLEM_URL", "").strip() or "http://127.0.0.1:8743"
        token = os.environ.get("GOLEM_TOKEN", "").strip()
        return cls(url, token)

    def wait_ready(self, timeout: float = 30.0) -> None:
        """Block until ``GET /v1/health`` returns ``{"ok": true}``.

        Args:
            timeout: Seconds to wait before raising ``GolemError``.

        Raises:
            GolemError: Golem did not become ready in time.
        """
        deadline = time.monotonic() + timeout
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                body = self._request("GET", "/v1/health")
                if body.get("ok"):
                    return
                last = GolemError("golem health not ok")
            except Exception as exc:
                last = exc
            time.sleep(0.4)
        if last is None:
            last = GolemError("golem not ready")
        raise GolemError(f"golem not ready: {last}") from last

    def post_turn(self, conversation_id: str, channel: str, text: str) -> str:
        """Start a turn. Does not wait for the assistant reply.

        Args:
            conversation_id: Stable id for this chat (channel thread, CLI
                session, and so on).
            channel: Channel name, for example ``telegram`` or ``cli``.
            text: User message.

        Returns:
            Turn id. Pass it to ``stream_turn`` to read SSE events.

        Raises:
            GolemError: Golem rejected the turn or returned no id.
        """
        accepted = self._request(
            "POST",
            f"/v1/conversations/{conversation_id}/turns",
            {"channel": channel, "text": text},
        )
        if accepted.get("error"):
            raise GolemError(accepted["error"])
        turn_id = accepted.get("id")
        if not turn_id:
            raise GolemError("missing turn id")
        return str(turn_id)

    def stream_turn(self, turn_id: str) -> Iterator[TurnEvent]:
        """Yield SSE events until ``done`` or ``error``.

        Args:
            turn_id: Id returned by ``post_turn``.

        Yields:
            ``TurnEvent`` values named ``log``, ``done``, or ``error``.

        Raises:
            GolemError: The stream failed or an event was not JSON.
        """
        req = urllib.request.Request(
            self.url + f"/v1/turns/{turn_id}",
            headers=self._headers(),
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=None) as resp:
                for name, raw in _read_sse(resp):
                    data = json.loads(raw) if raw else {}
                    if not isinstance(data, dict):
                        raise GolemError(f"invalid event data: {raw}")
                    yield TurnEvent(name=name, data=data)
                    if name in ("done", "error"):
                        return
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            raise GolemError(
                f"unexpected status {exc.code}: {raw.decode(errors='replace')}",
                status=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise GolemError(f"request failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise GolemError(f"invalid json: {exc}") from exc

    def send(self, conversation_id: str, channel: str, text: str) -> str:
        """Post a turn and block until the assistant reply.

        Args:
            conversation_id: Stable id for this chat.
            channel: Channel name, for example ``telegram`` or ``cli``.
            text: User message.

        Returns:
            Assistant text from the ``done`` event.

        Raises:
            GolemError: The turn failed or ended without ``done``.
        """
        turn_id = self.post_turn(conversation_id, channel, text)
        for ev in self.stream_turn(turn_id):
            if ev.name == "done":
                return str(ev.data.get("text") or "")
            if ev.name == "error":
                raise GolemError(str(ev.data.get("error") or "turn failed"))
        raise GolemError("turn ended without done")

    def register(self, name: str, callback_url: str, capabilities: list[dict[str, Any]]) -> None:
        """Register this process with Golem.

        ``Extension.run`` calls this. Use it directly only if you bind your
        own callback server.

        Args:
            name: Extension name, matching the installed package.
            callback_url: Loopback URL Golem should POST to
                (``http://127.0.0.1:<port>``).
            capabilities: Provider and channel descriptors.

        Raises:
            GolemError: Register returned a non-ok body.
        """
        body = self._request(
            "POST",
            "/v1/extensions/register",
            {"name": name, "callback_url": callback_url, "capabilities": capabilities},
        )
        if not isinstance(body, dict) or not body.get("ok"):
            raise GolemError("register failed")

    def heartbeat(self, name: str) -> None:
        """Refresh this extension's registration so it does not expire.

        Args:
            name: Same name passed to ``register``.

        Raises:
            GolemError: Heartbeat returned a non-ok body. HTTP 404 means
                Golem no longer has this name; register again.
        """
        body = self._request("POST", "/v1/extensions/heartbeat", {"name": name})
        if not isinstance(body, dict) or not body.get("ok"):
            raise GolemError("heartbeat failed")

    def _headers(self, body: dict[str, Any] | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        if body is not None:
            headers["Content-Type"] = "application/json"
        return headers

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        data = None
        headers = self._headers(body)
        if body is not None:
            data = json.dumps(body).encode()
        req = urllib.request.Request(
            self.url + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            raise GolemError(
                f"unexpected status {exc.code}: {raw.decode(errors='replace')}",
                status=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise GolemError(f"request failed: {exc.reason}") from exc
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GolemError(f"invalid json: {raw.decode(errors='replace')}") from exc


def _read_sse(resp: Any) -> Iterator[tuple[str, str]]:
    event = "message"
    data: list[str] = []
    while True:
        line = resp.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip("\r\n")
        if text == "":
            if data:
                yield event, "\n".join(data)
            event = "message"
            data = []
            continue
        if text.startswith(":"):
            continue
        if text.startswith("event:"):
            event = text[6:].lstrip()
        elif text.startswith("data:"):
            value = text[5:]
            if value.startswith(" "):
                value = value[1:]
            data.append(value)
    if data:
        yield event, "\n".join(data)
