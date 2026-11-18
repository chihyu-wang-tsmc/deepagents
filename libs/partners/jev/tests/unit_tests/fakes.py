"""Test doubles: a scripted decision model and a scripted chat model."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from langchain_core.decisions import BaseDecisionModel, ClassifierResponse
from langchain_core.language_models import BaseChatModel, LangSmithParams
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field, PrivateAttr

from langchain_jev.classifiers import choice_confidence

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.runnables import Runnable

Rule = float | int | str | Callable[[object, str], float | int | str | None] | None


class ScriptedJev(BaseDecisionModel):
    """Answers each question from `rules`; unknown Nouls answer 0, Choices the first option."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "scripted-jev"
    rules: dict[str, Any] = Field(default_factory=dict)
    calls: list[tuple[object, dict[str, Any]]] = Field(default_factory=list)
    fail: bool = False

    @property
    def _provider(self) -> str:
        return "scripted"

    def _classify(self, state, questions) -> ClassifierResponse:
        self.calls.append((state, dict(questions)))
        if self.fail:
            msg = "Jev is down"
            raise RuntimeError(msg)
        answers = {qid: self._answer(qid, q, state) for qid, q in questions.items()}
        return ClassifierResponse.model_validate({"model": self.model, "answers": answers})

    def _lookup(self, qid: str) -> Rule:
        parts = qid.split("::")
        candidates = ["::".join(parts[i:]) for i in range(len(parts))]
        candidates.append(parts[-1].split(".")[-1])
        return next((self.rules[c] for c in candidates if c in self.rules), None)

    def _answer(self, qid: str, question, state) -> dict[str, Any]:
        rule = self._lookup(qid)
        value = rule(state, qid) if callable(rule) else rule
        if question.type == "noul":
            return {"type": "noul", "noul": 0.0 if value is None else float(value)}
        if question.type == "choice":
            return _choice(list(question.criteria), value)
        levels = len(question.criteria)
        level = levels - 1 if value is None else int(value)
        probabilities = {i: 1.0 if i == level else 0.0 for i in range(levels)}
        return {
            "type": "score",
            "score": float(level),
            "legend": dict(enumerate(question.criteria)),
            "probabilities": probabilities,
            "confidence": 1.0,
        }

    def asked(self, qid_suffix: str) -> list[object]:
        """States of every request that asked a question ending in `qid_suffix`."""
        return [state for state, qs in self.calls if any(q.endswith(qid_suffix) for q in qs)]


def _choice(options: list[str], value: object) -> dict[str, Any]:
    chosen = options[0] if value is None else str(value)
    if len(options) == 1:
        probabilities = {chosen: 1.0}
    else:
        rest = 0.1 / (len(options) - 1)
        probabilities = {o: 0.9 if o == chosen else rest for o in options}
    return {
        "type": "choice",
        "choice": chosen,
        "probabilities": probabilities,
        "confidence": choice_confidence(list(probabilities.values())),
    }


class FakeChat(BaseChatModel):
    """Returns scripted replies in order and records what it was sent."""

    responses: list[Any] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    tool_names: list[list[str]] = Field(default_factory=list)
    label: str = "fake"
    _bound: list[str] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> LangSmithParams:
        return LangSmithParams(
            ls_provider="jevtest", ls_model_name=self.label, ls_model_type="chat"
        )

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable:
        self._bound = [getattr(t, "name", None) or t.get("name", "") for t in tools]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        self.tool_names.append(self._bound)
        self._bound = []
        reply = self.responses.pop(0) if self.responses else "(no scripted reply left)"
        message = AIMessage(content=reply) if isinstance(reply, str) else reply
        return ChatResult(generations=[ChatGeneration(message=message)])

    def sent_text(self, call: int = -1) -> str:
        """Everything one call was sent, as text."""
        return "\n".join(m.text for m in self.calls[call])
