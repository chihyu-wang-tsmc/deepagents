"""Small helpers for reading and writing agent messages."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def text(message: BaseMessage | None) -> str:
    """The text content of a message, or an empty string."""
    return message.text if message is not None else ""


def is_user_message(message: BaseMessage) -> bool:
    """A human message the user wrote, not one a middleware injected.

    Jev feedback, pinned skills, and rubric revisions are human messages too; they
    carry a `jev` or `lc_source` marker in `additional_kwargs`.
    """
    if not isinstance(message, HumanMessage):
        return False
    return not ({"jev", "lc_source"} & set(message.additional_kwargs))


def latest_human(messages: Sequence[AnyMessage]) -> HumanMessage | None:
    """The most recent message the user wrote."""
    return next((m for m in reversed(messages) if is_user_message(m)), None)  # ty: ignore[invalid-explicit-return]  # narrowed by is_user_message


def latest_request(messages: Sequence[AnyMessage]) -> str:
    """The text of the most recent user message."""
    return text(latest_human(messages))


def final_answer(messages: Sequence[AnyMessage]) -> AIMessage | None:
    """The last message, when it is a text reply with no tool calls."""
    if not messages:
        return None
    last = messages[-1]
    if isinstance(last, AIMessage) and not last.tool_calls and last.text.strip():
        return last
    return None


def new_tool_messages(messages: Sequence[AnyMessage]) -> list[ToolMessage]:
    """Tool results that arrived after the model's last message."""
    fresh: list[ToolMessage] = []
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            break
        if isinstance(message, ToolMessage):
            fresh.append(message)
    return list(reversed(fresh))


def current_turn(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """Messages since the user's most recent message, that message included."""
    for index in range(len(messages) - 1, -1, -1):
        if is_user_message(messages[index]):
            return list(messages[index:])
    return list(messages)


def append_system_text(system: SystemMessage | None, addition: str) -> SystemMessage:
    """Append text to a system message, keeping content blocks intact."""
    if system is None:
        return SystemMessage(content=addition)
    if isinstance(system.content, str):
        return SystemMessage(content=f"{system.content}\n\n{addition}")
    return SystemMessage(content=[*system.content, {"type": "text", "text": addition}])


def clip(value: str, limit: int) -> str:
    """Cut text to `limit` characters, marking the cut."""
    if len(value) <= limit:
        return value
    return f"{value[:limit]}\n...(truncated)"
