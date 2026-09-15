from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class UnsupportedFormat(Exception):
    pass


@dataclass
class FunctionCall:
    name: str
    arguments: str = ""


@dataclass
class ToolCall:
    id: str
    type: str = "function"
    function: FunctionCall = field(default_factory=lambda: FunctionCall(""))


@dataclass
class Message:
    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
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
        out: dict[str, Any] = {"role": self.role}
        if self.content:
            out["content"] = self.content
        if self.tool_calls:
            out["tool_calls"] = [
                {
                    "id": call.id,
                    "type": call.type,
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        return out


@dataclass
class ToolDef:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolDef:
        params = data.get("parameters")
        return cls(
            name=str(data.get("name") or ""),
            description=str(data.get("description") or ""),
            parameters=params if isinstance(params, dict) else {},
        )


@dataclass
class JSONSchema:
    name: str
    strict: bool = False
    schema: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JSONSchema:
        inner = data.get("schema")
        return cls(
            name=str(data.get("name") or ""),
            strict=bool(data.get("strict")),
            schema=inner if isinstance(inner, dict) else {},
        )


class Provider(ABC):
    @abstractmethod
    def chat(self, model: str, messages: list[Message], tools: list[ToolDef] | None = None) -> Message:
        raise NotImplementedError

    def chat_structured(self, model: str, messages: list[Message], schema: JSONSchema) -> Any:
        raise UnsupportedFormat("no json schema")

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
