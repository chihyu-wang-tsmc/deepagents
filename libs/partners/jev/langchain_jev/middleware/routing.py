"""Pick the model for each call, and drop tools for requests a reply can answer.

The routing and coding questions ride in the entry request. A request whose answer
needs code goes to the coding model, which beats the routed model. A retry after a
failed verification escalates to `RouterSettings.escalate_to` (the "cheap first,
escalate on failure" cascade of https://docs.typesafe.ai/cookbooks/sde_cascade), and
escalation beats both.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING

from langchain.chat_models import init_chat_model
from langchain_core.decisions import Choice
from langchain_core.decisions.types import ChoiceAnswer, NoulAnswer

from langchain_jev._types import CHAT_MODE, CODING_ROUTER, ESCALATION, MODEL_ROUTER
from langchain_jev.middleware._base import JevMiddleware
from langchain_jev.middleware.run import entry_answers
from langchain_jev.switches import mode_for

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain_core.decisions import Question
    from langchain_core.language_models import BaseChatModel

    from langchain_jev._runtime import JevRuntime
    from langchain_jev._types import Mode


class ModelRouterMiddleware(JevMiddleware):
    """Send each model call to the model Jev picked for the user's request."""

    blueprint = MODEL_ROUTER

    def __init__(self, jev: JevRuntime) -> None:
        """Build the candidate models once."""
        super().__init__(jev)
        settings = jev.config.router
        if settings is None:
            msg = "ModelRouterMiddleware needs `JevConfig.router`."
            raise ValueError(msg)
        self.settings = settings
        self.models: dict[str, BaseChatModel] = {
            name: init_chat_model(choice.model) if isinstance(choice.model, str) else choice.model
            for name, choice in settings.choices.items()
        }

    def entry_questions(self, snapshot: Mapping[str, Mode]) -> dict[str, Question]:  # noqa: ARG002 - one question, no parts
        """The routing Choice."""
        criteria = self.settings.criteria()
        return {"model_route": Choice(instructions=self.instructions("route"), criteria=criteria)}

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Call the routed model."""
        return handler(self._routed(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Call the routed model."""
        return await handler(self._routed(request))

    def _routed(self, request: ModelRequest) -> ModelRequest:
        mode = self.mode(request.state, request.runtime)
        answers = entry_answers(request.state, self.blueprint) or {}
        answer = answers.get("model_route")
        if mode == "off" or not isinstance(answer, ChoiceAnswer):
            return request
        choice, reason = self._choose(request.state, answer)
        self.record(mode, f"route:{choice}", reason=reason, confidence=answer.confidence)
        if mode != "on":
            return request
        return request.override(model=self.models[choice])

    def _choose(self, state: Mapping[str, object], answer: ChoiceAnswer) -> tuple[str, str]:
        target = self.settings.escalate_to
        if target and _escalating(state):
            return target, "escalated after failed verification"
        low = answer.confidence < self.threshold("model_router.min_confidence")
        if low and self.settings.fallback:
            return self.settings.fallback, "low routing confidence"
        return answer.choice, "routed"


class CodingRouterMiddleware(JevMiddleware):
    """Send a request whose answer needs code to the coding model."""

    blueprint = CODING_ROUTER

    def __init__(self, jev: JevRuntime) -> None:
        """Build the coding model once."""
        super().__init__(jev)
        settings = jev.config.coding
        if settings is None:
            msg = "CodingRouterMiddleware needs `JevConfig.coding`."
            raise ValueError(msg)
        model = settings.model
        self.model: BaseChatModel = init_chat_model(model) if isinstance(model, str) else model

    def entry_questions(self, snapshot: Mapping[str, Mode]) -> dict[str, Question]:  # noqa: ARG002 - one question, no parts
        """The coding Noul."""
        return {"coding": self.question("coding")}

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Call the coding model for a request that needs code."""
        return handler(self._routed(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Call the coding model for a request that needs code."""
        return await handler(self._routed(request))

    def _routed(self, request: ModelRequest) -> ModelRequest:
        mode = self.mode(request.state, request.runtime)
        answer = (entry_answers(request.state, self.blueprint) or {}).get("coding")
        if mode == "off" or not isinstance(answer, NoulAnswer):
            return request
        coding = answer.noul >= self.threshold("coding_router.min_coding")
        router = self.jev.config.router
        escalated = coding and bool(router and router.escalate_to) and _escalating(request.state)
        action = "route:escalated" if escalated else ("route:coding" if coding else "pass")
        self.record(mode, action, coding=answer.noul)
        if mode != "on" or not coding or escalated:
            return request
        return request.override(model=self.model)


def _escalating(state: Mapping[str, object]) -> bool:
    """Whether this call retries a failed answer on the escalation model."""
    snapshot = state.get("jev_modes")
    on = isinstance(snapshot, Mapping) and mode_for(snapshot, ESCALATION) == "on"
    return on and bool(state.get("jev_escalate"))


class ChatModeMiddleware(JevMiddleware):
    """Offer no tools when one text reply fully answers the request."""

    blueprint = CHAT_MODE

    def entry_questions(self, snapshot: Mapping[str, Mode]) -> dict[str, Question]:  # noqa: ARG002 - one question, no parts
        """The chat-or-agent Choice."""
        return {"mode": self.question("mode")}

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Call the model without tools for a chat request."""
        return handler(self._chat(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Call the model without tools for a chat request."""
        return await handler(self._chat(request))

    def _chat(self, request: ModelRequest) -> ModelRequest:
        mode = self.mode(request.state, request.runtime)
        answer = (entry_answers(request.state, self.blueprint) or {}).get("mode")
        if mode == "off" or not isinstance(answer, ChoiceAnswer) or not request.tools:
            return request
        probability = answer.probabilities.get("chat", 0.0)
        chat = probability >= self.threshold("chat_mode.min_probability")
        self.record(mode, "chat" if chat else "agent", chat=probability)
        if mode != "on" or not chat:
            return request
        return request.override(tools=[])
