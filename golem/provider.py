from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class UnsupportedFormat(Exception):
    """The model cannot satisfy the requested structured-output schema."""


@dataclass
class FunctionCall:
    """A function the model wants to invoke.

    Attributes:
        name: Tool name, matching a ``ToolDef.name``.
        arguments: JSON object encoded as a string.
    """

    name: str
    arguments: str = ""


@dataclass
class ToolCall:
    """One tool invocation on an assistant message.

    Attributes:
        id: Call id. A later tool-result ``Message`` must echo this in
            ``tool_call_id``.
        type: Always ``function`` on the current protocol.
        function: Name and arguments to run.
    """

    id: str
    type: str = "function"
    function: FunctionCall = field(default_factory=lambda: FunctionCall(""))


@dataclass
class Message:
    """One chat message on the provider wire.

    Attributes:
        role: ``user``, ``assistant``, ``system``, or ``tool``.
        content: Text body. Empty when the assistant only issued tool calls.
        tool_calls: Assistant-requested tool calls.
        tool_call_id: Set on tool-result messages to the matching ``ToolCall.id``.
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Parse a wire JSON object into a ``Message``."""
        calls: list[ToolCall] = []
        for raw in data.get("tool_calls") or []:
            if not isinstance(raw, dict):
                continue
            fn = raw.get("function") if isinstance(raw.get("function"), dict) else {}
            calls.append(
                ToolCall(
                    id=str(raw.get("id") or ""),
                    type=str(raw.get("type") or "function"),
                    function=FunctionCall(
                        name=str(fn.get("name") or ""),
                        arguments=str(fn.get("arguments") or ""),
                    ),
                )
            )
        return cls(
            role=str(data.get("role") or ""),
            content=str(data.get("content") or ""),
            tool_calls=calls,
            tool_call_id=str(data.get("tool_call_id") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON object Golem expects on ``/v1/chat``."""
        out: dict[str, Any] = {"role": self.role}
        if self.content:
            out["content"] = self.content
        if self.tool_calls:
            out["tool_calls"] = [
                {
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        return out


@dataclass
class ToolDef:
    """A tool the model may call during chat.

    Attributes:
        name: Identifier the model uses in a ``FunctionCall``.
        description: Shown to the model so it knows when to call the tool.
        parameters: JSON Schema for the tool arguments.
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolDef:
        """Parse a wire JSON object into a ``ToolDef``."""
        params = data.get("parameters")
        return cls(
            name=str(data.get("name") or ""),
            description=str(data.get("description") or ""),
            parameters=params if isinstance(params, dict) else {},
        )


@dataclass
class JSONSchema:
    """Structured-output schema for ``Provider.chat_structured``.

    Attributes:
        name: Schema name Golem sends on the wire.
        strict: Whether the model must match the schema exactly.
        schema: JSON Schema object (the inner ``schema`` field on the wire).
    """

    name: str
    strict: bool = False
    schema: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JSONSchema:
        """Parse a wire JSON object into a ``JSONSchema``."""
        inner = data.get("schema")
        return cls(
            name=str(data.get("name") or ""),
            strict=bool(data.get("strict")),
            schema=inner if isinstance(inner, dict) else {},
        )


class Provider(ABC):
    """Model backend Golem calls for chat, structured output, or embeddings.

    Implement ``chat``. Override ``chat_structured`` and ``embed`` to advertise
    those routes when this instance is passed to ``Extension.provider``.
    """

    @abstractmethod
    def chat(
        self, model: str, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Message:
        """Run one chat completion.

        Args:
            model: Model id from Golem conf (for example ``openai/gpt-4o-mini``).
            messages: Conversation so far, including tool results.
            tools: Tools the model may call, or ``None`` if none are offered.

        Returns:
            Assistant message. May include ``tool_calls`` instead of text.
        """
        raise NotImplementedError

    def chat_structured(
        self, model: str, messages: list[Message], schema: JSONSchema
    ) -> Any:
        """Return a JSON value that matches ``schema``.

        Args:
            model: Model id from Golem conf.
            messages: Conversation so far.
            schema: Requested output shape.

        Returns:
            Parsed JSON (typically a dict) matching ``schema``.

        Raises:
            UnsupportedFormat: Default if this provider does not support
                structured output. Override to advertise the route.
        """
        raise UnsupportedFormat("no json schema")

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """Embed each string in ``texts``.

        Args:
            model: Embedding model id from Golem conf.
            texts: Inputs to embed, in order.

        Returns:
            One vector per input, in the same order.

        Raises:
            NotImplementedError: Default if this provider does not support
                embeddings. Override to advertise the route.
        """
        raise NotImplementedError
