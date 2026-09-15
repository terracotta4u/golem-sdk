# golem-sdk

Python SDK for [Golem](https://github.com/terracotta4u/golem) extensions (channels, providers, and other long-running processes).

Install from this directory:

```sh
uv sync --dev
```

Import as `golem`. The wire protocol lives in Golem: [docs/extensions.md](https://github.com/terracotta4u/golem/blob/main/docs/extensions.md).

## Develop

```sh
uv sync --dev
uv run pytest
```
