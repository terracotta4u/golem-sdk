from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator


class GolemError(Exception):
    pass


@dataclass(frozen=True)
class TurnEvent:
    name: str
    data: dict[str, Any]


class Client:
    def __init__(self, url: str, token: str = "") -> None:
        self.url = url.rstrip("/")
        self.token = token

    @classmethod
    def from_env(cls) -> Client:
        url = os.environ.get("GOLEM_URL", "").strip() or "http://127.0.0.1:8743"
        token = os.environ.get("GOLEM_TOKEN", "").strip()
        return cls(url, token)

    def wait_ready(self, timeout: float = 30.0) -> None:
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
            raise GolemError(f"unexpected status {exc.code}: {raw.decode(errors='replace')}") from exc
        except urllib.error.URLError as exc:
            raise GolemError(f"request failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise GolemError(f"invalid json: {exc}") from exc

    def send(self, conversation_id: str, channel: str, text: str) -> str:
        turn_id = self.post_turn(conversation_id, channel, text)
        for ev in self.stream_turn(turn_id):
            if ev.name == "done":
                return str(ev.data.get("text") or "")
            if ev.name == "error":
                raise GolemError(str(ev.data.get("error") or "turn failed"))
        raise GolemError("turn ended without done")

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
            raise GolemError(f"unexpected status {exc.code}: {raw.decode(errors='replace')}") from exc
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
