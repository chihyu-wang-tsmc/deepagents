"""Handle simple requests with Jev alone, without calling the agent's model."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from langchain.agents.middleware.types import hook_config
from langchain_core.messages import AIMessage, ToolMessage

from langchain_jev._messages import latest_request
from langchain_jev._types import FAST_PATH
from langchain_jev.fast_routes import CallFunction, FastAction, HandOff, Reply
from langchain_jev.middleware._base import JevMiddleware
from langchain_jev.middleware.guardrails import JEV_MARK
from langchain_jev.middleware.run import entry_answers

if TYPE_CHECKING:
    from collections.abc import Mapping

    from langchain_core.decisions import Question
    from langchain_core.decisions.types import Answer
    from langgraph.runtime import Runtime

    from langchain_jev._runtime import JevRuntime
    from langchain_jev._types import Mode
    from langchain_jev.fast_routes import FastRoute
    from langchain_jev.state import JevState


class FastPathMiddleware(JevMiddleware):
    """Answer directly when a route is confident; otherwise leave it to the model."""

    blueprint = FAST_PATH

    def __init__(self, jev: JevRuntime) -> None:
        """Use the configured routes."""
        super().__init__(jev)
        settings = jev.config.fast_path
        if settings is None:
            msg = "FastPathMiddleware needs `JevConfig.fast_path`."
            raise ValueError(msg)
        self.routes: list[FastRoute] = list(settings.routes)

    def entry_questions(self, snapshot: Mapping[str, Mode]) -> dict[str, Question]:  # noqa: ARG002 - routes have no parts
        """Every route's questions, prefixed by the route name."""
        return {
            f"{route.name}::{qid}": q
            for route in self.routes
            for qid, q in route.questions().items()
        }

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """Take the first confident route's action and end the run."""
        mode = self.mode(state, runtime)
        answers = entry_answers(state, self.blueprint)
        if mode == "off" or not answers or not latest_request(state["messages"]):
            return None
        for route in self.routes:
            action = route.decide(_scoped(answers, route.name), self.jev.thresholds)
            if action is not None:
                return self._take(mode, route.name, action)
        self.record(mode, "pass")
        return None

    def _take(self, mode: Mode, route: str, action: FastAction) -> dict[str, object] | None:
        self.record(mode, _label(action), route=route, **_details(action))
        if mode != "on":
            return None
        if isinstance(action, CallFunction):
            messages = self._call(action)
            if messages is None:
                return None
        else:
            messages = [_ai(action.message if isinstance(action, HandOff) else action.text)]
        return {"messages": messages, "jump_to": "end"}

    def _call(self, action: CallFunction) -> list[AIMessage | ToolMessage] | None:
        call_id = f"jev_{uuid.uuid4().hex[:12]}"
        try:
            result = action.function(**action.kwargs)
        except Exception as error:  # noqa: BLE001 - a failing function hands the request back to the model
            self.record("on", "error", function=action.name, error=type(error).__name__)
            return None
        reply = action.render(result) if action.render else str(result)
        call = {
            "name": action.name,
            "args": dict(action.kwargs),
            "id": call_id,
            "type": "tool_call",
        }
        return [
            AIMessage(content="", tool_calls=[call], additional_kwargs={JEV_MARK: FAST_PATH}),
            ToolMessage(
                content=str(result),
                tool_call_id=call_id,
                name=action.name,
                additional_kwargs={JEV_MARK: FAST_PATH},
            ),
            _ai(reply),
        ]


def _scoped(answers: Mapping[str, Answer], route: str) -> dict[str, Answer]:
    prefix = f"{route}::"
    return {
        key.removeprefix(prefix): value for key, value in answers.items() if key.startswith(prefix)
    }


def _label(action: FastAction) -> str:
    if isinstance(action, CallFunction):
        return f"call:{action.name}"
    return "handoff" if isinstance(action, HandOff) else "reply"


def _details(action: FastAction) -> dict[str, object]:
    if isinstance(action, CallFunction):
        return {"kwargs": action.kwargs, "confidence": action.confidence}
    if isinstance(action, HandOff):
        return {"reason": action.reason}
    return {"text": action.text if isinstance(action, Reply) else ""}


def _ai(content: str) -> AIMessage:
    return AIMessage(content=content, additional_kwargs={JEV_MARK: FAST_PATH})
