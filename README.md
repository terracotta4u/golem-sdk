# golem-sdk

Python SDK for [Golem](https://github.com/terracotta4u/golem) extensions (channels, providers, and other long-running processes).

Install from this directory:

```sh
uv sync --dev
```

Import as `golem`. The wire protocol lives in Golem: [docs/extensions.md](https://github.com/terracotta4u/golem/blob/main/docs/extensions.md).

```python
from golem import Client, Extension, Provider

client = Client.from_env()
client.wait_ready()
text = client.send(conversation_id, "telegram", "hello")
```

`post_turn` and `stream_turn` expose the same flow as SSE events (`log`, `done`, `error`).

A provider extension binds a loopback callback, registers, and heartbeats:

```python
from golem import Extension, Message, Provider

class Echo(Provider):
    def chat(self, model, messages, tools=None) -> Message:
        return Message(role="assistant", content=messages[-1].content)

Extension.from_env("golem-echo").provider("echo", Echo()).run()
```

`task()` runs your own loop (a channel). `capability()` registers kinds Golem does not call.

## Develop

```sh
uv sync --dev
uv run pytest
```
