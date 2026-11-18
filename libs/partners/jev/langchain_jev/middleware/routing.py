"""Pick the model for each call, and drop tools for requests a reply can answer.

The routing question rides in the entry request. The routed model also serves the
retry after a failed verification, which then escalates to `RouterSettings.escalate_to`
(the "cheap first, escalate on failure" cascade of
https://docs.typesafe.ai/cookbooks/sde_cascade).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING

from langchain.chat_models import init_chat_model
from langchain_core.decisions import Choice
from langchain_core.decisions.types import ChoiceAnswer

from langchain_jev._types import CHAT_MODE, ESCALATION, MODEL_ROUTER
from langchain_jev.middleware._base import JevMiddleware
from langchain_jev.middleware.run import entry_answers
from langchain_jev.questions import MODE
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
        criteria = {name: choice.criteria for name, choice in self.settings.choices.items()}
        return {"model_route": Choice(instructions=self.settings.instructions, criteria=criteria)}

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
        snapshot = state.get("jev_modes")
        escalating = isinstance(snapshot, Mapping) and mode_for(snapshot, ESCALATION) == "on"
        if escalating and state.get("jev_escalate") and self.settings.escalate_to:
            return self.settings.escalate_to, "escalated after failed verification"
        low = answer.confidence < self.threshold("model_router.min_confidence")
        if low and self.settings.fallback:
            return self.settings.fallback, "low routing confidence"
        return answer.choice, "routed"


class ChatModeMiddleware(JevMiddleware):
    """Offer no tools when one text reply fully answers the request."""

    blueprint = CHAT_MODE

    def entry_questions(self, snapshot: Mapping[str, Mode]) -> dict[str, Question]:  # noqa: ARG002 - one question, no parts
        """The chat-or-agent Choice."""
        return {"mode": MODE}

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
