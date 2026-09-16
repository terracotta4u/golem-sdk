import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from golem import Extension, GolemError, Message, Provider, ToolCall, UnsupportedFormat
from golem.provider import FunctionCall, JSONSchema, ToolDef


class _Golem:
    def __init__(self) -> None:
        self.registers: list[dict[str, Any]] = []
        self.heartbeats: list[str] = []
        self.registered = threading.Event()
        self.heartbeat_status = 200
        self.token = "secret"

    def handle(self, handler: BaseHTTPRequestHandler) -> None:
        auth = handler.headers.get("Authorization")
        if auth != "Bearer " + self.token:
            _write_json(handler, 401, {"error": "unauthorized"})
            return
        if handler.command == "GET" and handler.path == "/v1/health":
            _write_json(handler, 200, {"ok": True})
            return
        if handler.command == "POST" and handler.path == "/v1/extensions/register":
            body = _read_json(handler)
            self.registers.append(body)
            self.registered.set()
            _write_json(handler, 200, {"ok": True})
            return
        if handler.command == "POST" and handler.path == "/v1/extensions/heartbeat":
            body = _read_json(handler)
            self.heartbeats.append(str(body.get("name") or ""))
            if self.heartbeat_status != 200:
                _write_json(
                    handler, self.heartbeat_status, {"error": "extension not found"}
                )
                return
            _write_json(handler, 200, {"ok": True})
            return
        _write_json(handler, 404, {"error": "not found"})


@pytest.fixture
def golem() -> Any:
    state = _Golem()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            state.handle(self)

        def do_POST(self) -> None:
            state.handle(self)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    state.url = f"http://{host}:{port}"
    try:
        yield state
    finally:
        server.shutdown()
        thread.join(timeout=2)


class Fake(Provider):
    def __init__(self, *, slow: threading.Event | None = None) -> None:
        self.slow = slow
        self.models: list[str] = []

    def chat(
        self, model: str, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        self.models.append(model)
        if self.slow is not None:
            self.slow.set()
            time.sleep(0.25)
        if tools:
            return Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(
                            name=tools[0].name, arguments='{"path":"foo.go"}'
                        ),
                    )
                ],
            )
        text = messages[0].content if messages else ""
        return Message(role="assistant", content="echo:" + text)

    def chat_structured(
        self, model: str, messages: list[Message], schema: JSONSchema
    ) -> Any:
        if schema.name == "nope":
            raise UnsupportedFormat("no json schema")
        return {"memories": ["User prefers uv."]}

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.5] for t in texts]


class ChatOnly(Provider):
    def chat(
        self, model: str, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        return Message(role="assistant", content=model)


class EmbedOnly(Provider):
    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] for t in texts]


class EmptyProvider(Provider):
    pass


def _start(ext: Extension, stop: threading.Event) -> threading.Thread:
    thread = threading.Thread(target=ext.run, kwargs={"stop": stop}, daemon=True)
    thread.start()
    return thread


def _stop(thread: threading.Thread, stop: threading.Event) -> None:
    stop.set()
    thread.join(timeout=2)


def _wait_registered(golem: _Golem) -> dict[str, Any]:
    if not golem.registered.wait(timeout=3):
        raise AssertionError("extension did not register")
    assert golem.registers
    return golem.registers[0]


def _callback_post(
    url: str, path: str, token: str, body: dict[str, Any]
) -> tuple[int, Any]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=data,
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        parsed = json.loads(raw) if raw else {}
        return exc.code, parsed


def test_run_registers_and_dispatches(golem: _Golem) -> None:
    stop = threading.Event()
    fake = Fake()
    ext = Extension(
        "golem-openrouter", golem.url, golem.token, heartbeat_interval=0.05
    ).provider("openrouter", fake)
    thread = _start(ext, stop)
    try:
        reg = _wait_registered(golem)
        assert reg["name"] == "golem-openrouter"
        assert reg["callback_url"].startswith("http://127.0.0.1:")
        caps = reg["capabilities"]
        assert caps[0]["kind"] == "provider"
        assert caps[0]["id"] == "openrouter"
        assert caps[0]["chat"] is True
        assert caps[0]["structured"] is True
        assert caps[0]["embed"] is True

        status, body = _callback_post(
            reg["callback_url"],
            "/v1/chat",
            golem.token,
            {
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "read foo.go"}],
                "tools": [
                    {
                        "name": "read",
                        "description": "read a file",
                        "parameters": {"type": "object"},
                    }
                ],
            },
        )
        assert status == 200
        assert body["role"] == "assistant"
        assert body["tool_calls"][0]["function"]["name"] == "read"
        assert fake.models == ["openai/gpt-4o-mini"]
    finally:
        _stop(thread, stop)


def test_structured_embed_and_unsupported(golem: _Golem) -> None:
    stop = threading.Event()
    ext = Extension(
        "golem-openrouter", golem.url, golem.token, heartbeat_interval=0.05
    ).provider("openrouter", Fake())
    thread = _start(ext, stop)
    try:
        url = _wait_registered(golem)["callback_url"]
        status, body = _callback_post(
            url,
            "/v1/chat/structured",
            golem.token,
            {
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "extract"}],
                "schema": {
                    "name": "memories",
                    "strict": True,
                    "schema": {"type": "object"},
                },
            },
        )
        assert status == 200
        assert body == {"data": {"memories": ["User prefers uv."]}}

        status, body = _callback_post(
            url,
            "/v1/chat/structured",
            golem.token,
            {
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "extract"}],
                "schema": {
                    "name": "nope",
                    "strict": True,
                    "schema": {"type": "object"},
                },
            },
        )
        assert status != 200
        assert body["error"]["code"] == "unsupported_format"

        status, body = _callback_post(
            url,
            "/v1/embed",
            golem.token,
            {"model": "openai/text-embedding-3-small", "texts": ["hello", "world"]},
        )
        assert status == 200
        assert body == {"vectors": [[5.0, 0.5], [5.0, 0.5]]}
    finally:
        _stop(thread, stop)


def test_callback_requires_token(golem: _Golem) -> None:
    stop = threading.Event()
    ext = Extension(
        "golem-openrouter", golem.url, golem.token, heartbeat_interval=0.05
    ).provider("openrouter", Fake())
    thread = _start(ext, stop)
    try:
        url = _wait_registered(golem)["callback_url"]
        status, body = _callback_post(
            url, "/v1/chat", "wrong", {"model": "m", "messages": []}
        )
        assert status == 401
        assert "error" in body
    finally:
        _stop(thread, stop)


def test_heartbeat_during_slow_chat(golem: _Golem) -> None:
    stop = threading.Event()
    entered = threading.Event()
    ext = Extension(
        "golem-openrouter", golem.url, golem.token, heartbeat_interval=0.05
    ).provider("openrouter", Fake(slow=entered))
    thread = _start(ext, stop)
    try:
        url = _wait_registered(golem)["callback_url"]
        before = len(golem.heartbeats)
        status, body = _callback_post(
            url,
            "/v1/chat",
            golem.token,
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert status == 200
        assert body["content"] == "echo:hi"
        assert entered.is_set()
        assert len(golem.heartbeats) > before
    finally:
        _stop(thread, stop)


def test_task_and_extra_capability(golem: _Golem) -> None:
    stop = threading.Event()
    ran = threading.Event()

    def loop(client: Any, done: threading.Event) -> None:
        assert client.url == golem.url
        ran.set()
        done.wait()

    ext = (
        Extension("custom", golem.url, golem.token, heartbeat_interval=0.05)
        .capability({"kind": "widget", "id": "w1"})
        .task(loop)
    )
    thread = _start(ext, stop)
    try:
        reg = _wait_registered(golem)
        assert reg["capabilities"] == [{"kind": "widget", "id": "w1"}]
        assert ran.wait(timeout=2)
    finally:
        _stop(thread, stop)


def test_provider_requires_a_route() -> None:
    with pytest.raises(ValueError, match="chat, chat_structured, or embed"):
        Extension("golem-embed", "http://127.0.0.1:9").provider("embed", EmptyProvider())


def test_embed_only_omits_chat(golem: _Golem) -> None:
    stop = threading.Event()
    ext = Extension(
        "golem-embed", golem.url, golem.token, heartbeat_interval=0.05
    ).provider("local-embed", EmbedOnly())
    thread = _start(ext, stop)
    try:
        reg = _wait_registered(golem)
        cap = reg["capabilities"][0]
        assert cap["kind"] == "provider"
        assert cap["id"] == "local-embed"
        assert cap["embed"] is True
        assert "chat" not in cap
        assert "structured" not in cap

        status, body = _callback_post(
            reg["callback_url"],
            "/v1/embed",
            golem.token,
            {"model": "nomic-embed-text", "texts": ["hi", "yo"]},
        )
        assert status == 200
        assert body == {"vectors": [[2.0], [2.0]]}

        status, body = _callback_post(
            reg["callback_url"],
            "/v1/chat",
            golem.token,
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert status == 404
        assert "error" in body
    finally:
        _stop(thread, stop)


def test_chat_only_omits_structured_and_embed(golem: _Golem) -> None:
    stop = threading.Event()
    ext = Extension(
        "golem-openrouter", golem.url, golem.token, heartbeat_interval=0.05
    ).provider("openrouter", ChatOnly())
    thread = _start(ext, stop)
    try:
        cap = _wait_registered(golem)["capabilities"][0]
        assert cap["chat"] is True
        assert "structured" not in cap
        assert "embed" not in cap
    finally:
        _stop(thread, stop)


def test_reregister_after_heartbeat_404(golem: _Golem) -> None:
    stop = threading.Event()
    ext = Extension(
        "golem-openrouter", golem.url, golem.token, heartbeat_interval=0.05
    ).provider("openrouter", ChatOnly())
    thread = _start(ext, stop)
    try:
        _wait_registered(golem)
        golem.heartbeat_status = 404
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if len(golem.registers) >= 2:
                break
            time.sleep(0.05)
        golem.heartbeat_status = 200
        assert len(golem.registers) >= 2
        assert golem.registers[1]["name"] == "golem-openrouter"
    finally:
        _stop(thread, stop)


def test_from_env(monkeypatch: pytest.MonkeyPatch, golem: _Golem) -> None:
    monkeypatch.setenv("GOLEM_URL", golem.url)
    monkeypatch.setenv("GOLEM_TOKEN", golem.token)
    stop = threading.Event()
    ext = Extension.from_env("golem-openrouter", heartbeat_interval=0.05).provider(
        "openrouter", ChatOnly()
    )
    thread = _start(ext, stop)
    try:
        assert _wait_registered(golem)["name"] == "golem-openrouter"
    finally:
        _stop(thread, stop)


def test_client_register_and_heartbeat(golem: _Golem) -> None:
    from golem import Client

    client = Client(golem.url, golem.token)
    client.register(
        "golem-openrouter",
        "http://127.0.0.1:9",
        [{"kind": "provider", "id": "openrouter", "chat": True}],
    )
    client.heartbeat("golem-openrouter")
    assert golem.registers[0]["callback_url"] == "http://127.0.0.1:9"
    assert golem.heartbeats == ["golem-openrouter"]


def test_client_heartbeat_404(golem: _Golem) -> None:
    from golem import Client

    golem.heartbeat_status = 404
    with pytest.raises(GolemError) as exc:
        Client(golem.url, golem.token).heartbeat("missing")
    assert exc.value.status == 404


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if length == 0:
        return {}
    return json.loads(handler.rfile.read(length))


def _write_json(
    handler: BaseHTTPRequestHandler, status: int, body: dict[str, Any]
) -> None:
    raw = json.dumps(body).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)
