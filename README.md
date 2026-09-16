# Golem SDK

Python SDK for creating [Golem](https://github.com/terracotta4u/golem) extensions. This includes adding new channels, providers, and other long-running processes.

## Installation

This package is available on PyPI:

```bash
pip install golem-agent-sdk
```

## Usage

Import as `golem`. Golem injects `GOLEM_URL` and `GOLEM_TOKEN` into the process. The wire protocol lives in Golem: [docs/extensions.md](https://github.com/terracotta4u/golem/blob/main/docs/extensions.md).

The smallest extension registers, prints a line, and stays alive until it is stopped:

```python
from golem import Extension


def hello(client, stop):
    print("hello world")
    stop.wait()


Extension.from_env("echo").task(hello).run()
```



### Provider Extensions

A provider is a model backend. Golem calls your process for chat (and optionally structured output or embeddings). `run()` binds a loopback callback, registers, and heartbeats.

```python
from golem import Extension, Message, Provider


class Echo(Provider):
    def chat(self, model, messages, tools=None) -> Message:
        return Message(role="assistant", content=messages[-1].content)


Extension.from_env("golem-echo").provider("echo", Echo()).run()
```

The `id` passed to `provider()` is the name used in Golem's `default_model.provider`. Override `chat_structured` or `embed` to advertise those routes.

### Channel Extensions

A channel feeds messages into Golem (Telegram, CLI, and so on). `task()` runs your loop; `client.send()` posts a turn and returns the assistant reply. `capability()` advertises the channel.

```python
from golem import Extension


def cli(client, stop):
    while not stop.is_set():
        line = input("you: ").strip()
        if not line:
            continue
        print(client.send("local", "cli", line))


(
    Extension.from_env("golem-cli")
    .capability({"kind": "channel", "id": "cli"})
    .task(cli)
    .run()
)
```

`post_turn` and `stream_turn` expose the same flow as SSE events (`log`, `done`, `error`).

## Packaging Extensions

Ship the extension as a Python package. Golem takes the name and description from `pyproject.toml`, then runs the first `[project.scripts]` entry whose name matches the package:

```toml
[project]
name = "golem-openrouter"
description = "OpenRouter provider for Golem"

[project.scripts]
golem-openrouter = "golem_openrouter:main"
```

## Development

Using uv is strongly recommended:

```bash
uv sync --dev
uv run pytest
```

