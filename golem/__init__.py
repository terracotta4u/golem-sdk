"""Python SDK for Golem extensions (channels, providers, and other processes)."""

from golem.client import Client, GolemError, TurnEvent
from golem.extension import Extension
from golem.provider import (
    FunctionCall,
    JSONSchema,
    Message,
    Provider,
    ToolCall,
    ToolDef,
    UnsupportedFormat,
)

__all__ = [
    "Client",
    "Extension",
    "FunctionCall",
    "GolemError",
    "JSONSchema",
    "Message",
    "Provider",
    "ToolCall",
    "ToolDef",
    "TurnEvent",
    "UnsupportedFormat",
    "__version__",
]

__version__ = "0.1.0"
