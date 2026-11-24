"""Check each gated tool call before it runs: intent, leaks, irreversibility, injection.

Builds on LangChain's `AutoModeMiddleware` (one "is this risky" Noul) with the
per-action safety questions of https://evals.typesafe.ai/agent_trace_observability,
asked in one request. Risky calls pause for a person (human-in-the-loop) or are
refused; calls driven by instructions inside tool results are always refused.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langchain_core.messages import ToolMessage
from langgraph.types import interrupt

from langchain_jev._types import TOOL_GATE
from langchain_jev.middleware._base import Ask, Flow, Handle, JevMiddleware
from langchain_jev.middleware.guardrails import JEV_MARK
from langchain_jev.questions import TOOL_GATE_QUESTIONS

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from langchain.agents.middleware.types import ToolCallRequest
    from langchain_core.decisions import ClassifierResponse, Question
    from langgraph.types import Command

_RECENT = 30
"""Messages of context sent with each call, as in `AutoModeMiddleware`."""


def gate(nouls: Mapping[str, float], thresholds: Mapping[str, float]) -> tuple[str, list[str]]:
    """Decide `allow`, `ask`, or `block` for one tool call, with the reasons."""
    blocks = [
        name
        for name, failed in (
            (
                "instructed_by_tool_output",
                nouls["instructed_by_tool_output"] >= thresholds["instructed_by_tool_output"],
            ),
            (
                "within_grant",
                nouls.get("within_grant", 1.0) < thresholds["within_grant"],
            ),
        )
        if failed
    ]
    if blocks:
        return "block", blocks
    asks = [
        name
        for name, failed in (
            ("risky", nouls["risky"] >= thresholds["risky"]),
            ("leaks_data", nouls["leaks_data"] >= thresholds["leaks_data"]),
            (
                "irreversible_without_consent",
                nouls["irreversible"] >= thresholds["irreversible"]
                and nouls["user_asked_or_agreed"] < thresholds["user_asked_or_agreed"],
            ),
        )
        if failed
    ]
    return ("ask", asks) if asks else ("allow", [])


class ToolGateMiddleware(JevMiddleware):
    """Allow, refuse, or pause each gated tool call based on Jev's checks."""

    blueprint = TOOL_GATE

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Gate synchronously."""
        return self.drive(self._flow(request), cast("Callable[[object], object]", handler))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Gate asynchronously."""
        return await self.adrive(
            self._flow(request), cast("Callable[[object], Awaitable[object]]", handler)
        )

    def _flow(self, request: ToolCallRequest) -> Flow[ToolMessage | Command]:
        name = request.tool_call["name"]
        mode = self.mode(request.state, request.runtime, name)
        if not self.jev.config.tool_gate.checks(name) or mode == "off":
            return cast("ToolMessage | Command", (yield Handle(request)))
        response = cast(
            "ClassifierResponse | None",
            (yield Ask(self._state(request), self._questions())),
        )
        action, reasons, nouls = self._decide(response)
        self.record(mode, action, tool=name, reasons=reasons, **nouls)
        if mode == "on" and action == "ask" and self.jev.config.tool_gate.ask_human:
            refusal = self._ask_human(request, reasons)
            if refusal is not None:
                return refusal
        elif mode == "on" and action in {"ask", "block"}:
            return _refusal(request, reasons)
        return cast("ToolMessage | Command", (yield Handle(request)))

    def _questions(self) -> dict[str, Question]:
        with_grant = bool(self.jev.config.tool_gate.policy)
        return {
            name: self.question(name)
            for name in TOOL_GATE_QUESTIONS
            if with_grant or name != "within_grant"
        }

    def _state(self, request: ToolCallRequest) -> dict[str, object]:
        state = call_state(request)
        if self.jev.config.tool_gate.policy:
            state["agent_instructions"] = self.jev.config.tool_gate.policy
        return state

    def _decide(
        self, response: ClassifierResponse | None
    ) -> tuple[str, list[str], dict[str, float]]:
        if response is None:
            return ("block" if self.fails_closed else "allow"), ["jev_unavailable"], {}
        nouls = {key: answer.noul for key, answer in response.nouls.items()}
        thresholds = {
            key: self.threshold(f"tool_gate.{key}")
            for key in (
                "risky",
                "instructed_by_tool_output",
                "user_asked_or_agreed",
                "within_grant",
                "leaks_data",
                "irreversible",
            )
        }
        action, reasons = gate(nouls, thresholds)
        return action, reasons, nouls

    def _ask_human(self, request: ToolCallRequest, reasons: list[str]) -> ToolMessage | None:
        description = f"Jev flagged this call ({', '.join(reasons)}); approve to run it."
        rejection = ask_approval(request, description)
        return None if rejection is None else _refusal(request, reasons, rejection)


def call_state(request: ToolCallRequest) -> dict[str, object]:
    """What Jev reads about one tool call: the call, its tool, and the recent messages."""
    call = request.tool_call
    state: dict[str, object] = {
        "messages": list(request.state.get("messages", []))[-_RECENT:],
        "tool_call": {"id": call["id"], "name": call["name"], "args": call["args"]},
    }
    if request.tool is not None and request.tool.description:
        state["tool_description"] = request.tool.description
    return state


def ask_approval(request: ToolCallRequest, description: str) -> str | None:
    """Pause for approval in the format of LangChain's `HumanInTheLoopMiddleware`.

    Returns:
        `None` when the reviewer approves, otherwise the text the model reads instead.
    """
    call = request.tool_call
    decision = interrupt(
        {
            "action_requests": [
                {"name": call["name"], "args": call["args"], "description": description}
            ],
            "review_configs": [
                {"action_name": call["name"], "allowed_decisions": ["approve", "reject"]}
            ],
        }
    )["decisions"][0]
    if decision.get("type") == "approve":
        return None
    return (
        decision.get("message")
        or f"A reviewer rejected the call to `{call['name']}`; it was not run."
    )


def _refusal(
    request: ToolCallRequest, reasons: list[str], message: str | None = None
) -> ToolMessage:
    content = message or (
        f"The call to `{request.tool_call['name']}` was not run: Jev flagged it "
        f"({', '.join(reasons)}). Ask the user before trying anything similar."
    )
    return refused_call(request, content, TOOL_GATE)


def refused_call(request: ToolCallRequest, content: str, blueprint: str) -> ToolMessage:
    """The result the model reads in place of a tool call a Jev blueprint did not run."""
    call = request.tool_call
    return ToolMessage(
        content=content,
        tool_call_id=call["id"],
        name=call["name"],
        status="error",
        additional_kwargs={JEV_MARK: blueprint},
    )
