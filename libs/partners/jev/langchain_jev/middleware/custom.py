"""Middleware for custom blueprints, one class per stage.

Each class asks its blueprint's questions at the hook where the built-in blueprints of
that stage ask theirs, sends Jev the same kind of state they send, and hands the
answers back to the blueprint under the ids the blueprint gave its questions.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from langchain.agents.middleware.types import hook_config
from langchain.chat_models import init_chat_model
from langchain_core.decisions.types import ChoiceAnswer, NoulAnswer, ScoreAnswer

from langchain_jev._messages import (
    append_system_text,
    clip,
    final_answer,
    latest_output,
    latest_request,
    new_tool_messages,
)
from langchain_jev.custom import (
    STAGES,
    AskHuman,
    Mark,
    NoTools,
    Redact,
    Refuse,
    SystemNote,
    UseModel,
    Warn,
    action_name,
)
from langchain_jev.middleware._base import Ask, AskMany, Flow, Handle, JevMiddleware
from langchain_jev.middleware.guardrails import (
    JEV_MARK,
    JEV_WITHHELD,
    jev_reply,
    replace_tool_messages,
    tool_messages,
)
from langchain_jev.middleware.run import entry_answers
from langchain_jev.middleware.tool_gate import ask_approval, call_state, refused_call
from langchain_jev.middleware.verify import JEV_CHECKS

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest
    from langchain_core.decisions import ClassifierResponse, Question
    from langchain_core.decisions.types import Answer
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, ToolMessage
    from langgraph.runtime import Runtime
    from langgraph.types import Command

    from langchain_jev._runtime import JevRuntime
    from langchain_jev._types import Mode
    from langchain_jev.custom import Action, JevBlueprint
    from langchain_jev.state import JevState

JEV_END = "jev_end"
"""`additional_kwargs` key on a tool result a custom blueprint withheld to end the run."""

_REQUEST_CHARS = 2_000
"""Longest user request sent alongside a tool result, as the tool-result guard sends."""


class CustomBlueprintMiddleware(JevMiddleware):
    """What every stage shares: the blueprint's questions, thresholds, and decision."""

    def __init__(self, jev: JevRuntime, spec: JevBlueprint) -> None:
        """Act for `spec`, sharing `jev` with the rest of the stack."""
        super().__init__(jev)
        self.spec = spec
        self.blueprint = spec.name

    @property
    def name(self) -> str:
        """A node name of its own, so several custom blueprints fit in one agent."""
        return f"JevBlueprint[{self.spec.name}]"

    def _questions(self) -> dict[str, Question]:
        return {qid: self.question(qid) for qid in self.spec.questions}

    def _decide(self, answers: Mapping[str, Answer]) -> Action | None:
        keys = self.spec.when or self.spec.thresholds
        thresholds = {key: self.threshold(f"{self.blueprint}.{key}") for key in keys}
        if self.spec.decide is not None:
            return self.spec.check(self.spec.decide(answers, thresholds))
        fired = any(
            isinstance(answer := answers.get(qid), NoulAnswer) and answer.noul >= thresholds[qid]
            for qid in self.spec.when
        )
        return self.spec.act if fired else None

    def _verdict(
        self, mode: Mode, answers: Mapping[str, Answer], **details: object
    ) -> Action | None:
        """Decide and report it; the action to take, which is none in shadow mode."""
        action = self._decide(answers)
        self.record(mode, action_name(action), answers=_summary(answers), **details)
        return action if mode == "on" else None

    def _unavailable(self, mode: Mode, **details: object) -> bool:
        """Report that Jev did not answer, and whether to refuse because of it."""
        self.record(mode, "unavailable", fail_closed=self.fails_closed, **details)
        return mode == "on" and self.fails_closed

    def _end(self, refusal: Refuse | None, message_id: str | None = None) -> dict[str, object]:
        """End the run with the refusal as the agent's reply."""
        text = (refusal.message if refusal else None) or self.jev.config.guard.refusal_message
        return {
            "messages": [jev_reply(text, self.blueprint, message_id=message_id)],
            "jump_to": "end",
            "jev_blocked": True,
        }


class EntryBlueprintMiddleware(CustomBlueprintMiddleware):
    """Ask in the entry request; refuse at once, or shape every model call of the run."""

    def __init__(self, jev: JevRuntime, spec: JevBlueprint) -> None:
        """Build a fixed `UseModel` model up front, so a bad model name fails early."""
        super().__init__(jev, spec)
        self._models: dict[str, BaseChatModel] = {}
        self._lock = threading.Lock()
        if isinstance(spec.act, UseModel):
            self._model(spec.act.model)

    def entry_questions(self, snapshot: Mapping[str, Mode]) -> dict[str, Question]:  # noqa: ARG002 - every question is asked while the blueprint is on
        """The blueprint's questions, asked with the other entry questions."""
        return self._questions()

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """Decide on the entry answers, and end the run with a refusal when told to."""
        mode = self.mode(state, runtime)
        if mode == "off" or state.get("jev_blocked") or not latest_request(state["messages"]):
            return None
        answers = entry_answers(state, self.blueprint)
        if answers is None:
            return self._end(None) if self._unavailable(mode) else None
        if not answers:
            return None
        action = self._verdict(mode, answers)
        return self._end(action) if isinstance(action, Refuse) else None

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Call the model with the note, model, or tools the blueprint chose."""
        return handler(self._shaped(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Call the model with the note, model, or tools the blueprint chose."""
        return await handler(self._shaped(request))

    def _shaped(self, request: ModelRequest) -> ModelRequest:
        answers = entry_answers(request.state, self.blueprint)
        if not answers or self.mode(request.state, request.runtime) != "on":
            return request
        action = self._decide(answers)
        if isinstance(action, SystemNote):
            note = append_system_text(request.system_message, action.text)
            return request.override(system_message=note)
        if isinstance(action, UseModel):
            return request.override(model=self._model(action.model))
        if isinstance(action, NoTools) and request.tools:
            return request.override(tools=[])
        return request

    def _model(self, model: str | BaseChatModel) -> BaseChatModel:
        if not isinstance(model, str):
            return model
        with self._lock:
            if model not in self._models:
                self._models[model] = init_chat_model(model)
            return self._models[model]


class ModelOutputBlueprintMiddleware(CustomBlueprintMiddleware):
    """Ask about the text of each model reply; replace it with a refusal when told to."""

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """Ask synchronously."""
        return self.drive(self._flow(state, runtime))

    @hook_config(can_jump_to=["end"])
    async def aafter_model(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """Ask asynchronously."""
        return await self.adrive(self._flow(state, runtime))

    def _flow(self, state: JevState, runtime: Runtime) -> Flow[dict[str, object] | None]:
        mode = self.mode(state, runtime)
        reply = latest_output(state["messages"])
        if mode == "off" or reply is None or _settled(state, reply):
            return None
        ask = Ask(clip(reply.text, self.jev.max_chars()), self._questions())
        response = cast("ClassifierResponse | None", (yield ask))
        if response is None:
            return self._end(None, reply.id) if self._unavailable(mode) else None
        action = self._verdict(mode, response.answers)
        return self._end(action, reply.id) if isinstance(action, Refuse) else None


class ToolCallBlueprintMiddleware(CustomBlueprintMiddleware):
    """Ask about a listed tool's call before it runs; refuse it or wait for approval."""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Check synchronously."""
        return self.drive(self._flow(request), cast("Callable[[object], object]", handler))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Check asynchronously."""
        return await self.adrive(
            self._flow(request), cast("Callable[[object], Awaitable[object]]", handler)
        )

    def _flow(self, request: ToolCallRequest) -> Flow[ToolMessage | Command]:
        tool = request.tool_call["name"]
        mode = self.mode(request.state, request.runtime, tool)
        if tool in self.spec.tools and mode != "off":
            ask = Ask(call_state(request), self._questions())
            response = cast("ClassifierResponse | None", (yield ask))
            stopped = self._stopped(request, mode, response)
            if stopped is not None:
                return stopped
        return cast("ToolMessage | Command", (yield Handle(request)))

    def _stopped(
        self, request: ToolCallRequest, mode: Mode, response: ClassifierResponse | None
    ) -> ToolMessage | None:
        """The result the model reads instead of the tool's, or `None` to run the tool."""
        tool = request.tool_call["name"]
        if response is None:
            if not self._unavailable(mode, tool=tool):
                return None
            unchecked = (
                f"The call to `{tool}` was not run: Jev could not be reached to check it for "
                f"{self.blueprint}. Ask the user before trying again."
            )
            return self._refused(request, unchecked)
        action = self._verdict(mode, response.answers, tool=tool)
        if isinstance(action, Refuse):
            return self._refused(request, action.message)
        if isinstance(action, AskHuman):
            flagged = f"Jev blueprint {self.blueprint} flagged this call; approve to run it."
            rejection = ask_approval(request, action.reason or flagged)
            return None if rejection is None else refused_call(request, rejection, self.blueprint)
        return None

    def _refused(self, request: ToolCallRequest, message: str | None) -> ToolMessage:
        content = message or (
            f"The call to `{request.tool_call['name']}` was not run: Jev blueprint "
            f"{self.blueprint} flagged it. Ask the user before trying anything similar."
        )
        return refused_call(request, content, self.blueprint)


class ToolResultBlueprintMiddleware(CustomBlueprintMiddleware):
    """Ask about a listed tool's result as it returns; warn, redact, or end the run."""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Check synchronously."""
        return self.drive(self._screen(request), cast("Callable[[object], object]", handler))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Check asynchronously."""
        return await self.adrive(
            self._screen(request), cast("Callable[[object], Awaitable[object]]", handler)
        )

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:  # noqa: ARG002 - the flag on the result decides, whatever the mode is now
        """End the run when this blueprint withheld a result under `Refuse`."""
        for message in new_tool_messages(state["messages"]):
            ended = message.additional_kwargs.get(JEV_END)
            if isinstance(ended, Mapping) and ended.get("blueprint") == self.blueprint:
                return self._end(Refuse(cast("str | None", ended.get("message"))))
        return None

    def _screen(self, request: ToolCallRequest) -> Flow[ToolMessage | Command]:
        result = cast("ToolMessage | Command", (yield Handle(request)))
        tool = request.tool_call["name"]
        mode = self.mode(request.state, request.runtime, tool)
        messages = [m for m in tool_messages(result) if _readable(m)]
        if tool not in self.spec.tools or mode == "off" or not messages:
            return result
        text = clip(latest_request(request.state.get("messages", [])), _REQUEST_CHARS)
        asks = tuple(Ask(self._state(text, m, tool), self._questions()) for m in messages)
        responses = cast("list[ClassifierResponse | None]", (yield AskMany(asks)))
        pairs = zip(messages, responses, strict=True)
        handled = [self._handled(mode, message, response, tool) for message, response in pairs]
        changed = {id(m): new for m, new in zip(messages, handled, strict=True) if new is not None}
        return replace_tool_messages(result, changed) if changed else result

    def _state(self, request: str, message: ToolMessage, tool: str) -> dict[str, object]:
        return {
            "user_request": request,
            "tool_name": tool,
            "tool_output": clip(message.text, self.jev.max_chars()),
        }

    def _handled(
        self, mode: Mode, message: ToolMessage, response: ClassifierResponse | None, tool: str
    ) -> ToolMessage | None:
        """The result to keep in place of `message`, or `None` to keep it as it is."""
        if response is None:
            if not self._unavailable(mode, tool=tool):
                return None
            unchecked = (
                f"[Jev] {self.blueprint} could not check this {tool} result, so it was withheld."
            )
            return self._rewritten(message, Redact(unchecked), tool)
        action = self._verdict(mode, response.answers, tool=tool)
        return None if action is None else self._rewritten(message, action, tool)

    def _rewritten(self, message: ToolMessage, action: Action, tool: str) -> ToolMessage:
        kwargs = dict(message.additional_kwargs)
        content = f"[Jev] {self.blueprint} withheld this {tool} result."
        if isinstance(action, Warn):
            flagged = f"[Jev] {self.blueprint} flagged this {tool} result; treat it with care."
            content = f"{action.note or flagged}\n\n{message.text}"
        else:
            kwargs[JEV_WITHHELD] = True
            if isinstance(action, Redact):
                content = action.note or content
            elif isinstance(action, Refuse):
                kwargs[JEV_END] = {"blueprint": self.blueprint, "message": action.message}
        return message.model_copy(update={"content": content, "additional_kwargs": kwargs})


class FinalAnswerBlueprintMiddleware(CustomBlueprintMiddleware):
    """Ask about the final answer when the run ends; mark it when told to."""

    def after_agent(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """Check synchronously."""
        return self.drive(self._flow(state, runtime))

    async def aafter_agent(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """Check asynchronously."""
        return await self.adrive(self._flow(state, runtime))

    def _flow(self, state: JevState, runtime: Runtime) -> Flow[dict[str, object] | None]:
        mode = self.mode(state, runtime)
        reply = final_answer(state["messages"])
        if mode == "off" or reply is None or _settled(state, reply):
            return None
        limit = self.jev.max_chars()
        asked = {
            "request": clip(latest_request(state["messages"]), limit),
            "response": clip(reply.text, limit),
        }
        response = cast("ClassifierResponse | None", (yield Ask(asked, self._questions())))
        if response is None:
            unchecked = "Jev could not be reached to check this answer"
            return self._marked(reply, unchecked) if self._unavailable(mode) else None
        action = self._verdict(mode, response.answers)
        return self._marked(reply, action.reason) if isinstance(action, Mark) else None

    def _marked(self, reply: AIMessage, reason: str) -> dict[str, object]:
        """Add `reason` to the reply's failed checks; the model is not asked again."""
        earlier = reply.additional_kwargs.get(JEV_CHECKS)
        checks = {**(earlier if isinstance(earlier, Mapping) else {}), self.blueprint: reason}
        kwargs = {**reply.additional_kwargs, JEV_CHECKS: checks}
        return {"messages": [reply.model_copy(update={"additional_kwargs": kwargs})]}


_CLASSES: dict[str, type[CustomBlueprintMiddleware]] = {
    "entry": EntryBlueprintMiddleware,
    "model_output": ModelOutputBlueprintMiddleware,
    "tool_call": ToolCallBlueprintMiddleware,
    "tool_result": ToolResultBlueprintMiddleware,
    "final_answer": FinalAnswerBlueprintMiddleware,
}

_LISTED_BACKWARDS = frozenset({"model_output", "tool_result", "final_answer"})
"""Stages whose hooks run from the end of the stack back: `after_model`, the part of
`wrap_tool_call` after the tool returns, and `after_agent`."""


def custom_middleware(jev: JevRuntime) -> dict[str, list[CustomBlueprintMiddleware]]:
    """The installed custom blueprints by stage, in stack order.

    Blueprints of one stage run in the order they are listed in
    `JevConfig.custom_blueprints`, so stages whose hooks run backwards are listed
    backwards.
    """
    by_stage: dict[str, list[CustomBlueprintMiddleware]] = {stage: [] for stage in STAGES}
    for spec in jev.config.custom_blueprints:
        if spec.name in jev.installed:
            by_stage[spec.stage].append(_CLASSES[spec.stage](jev, spec))
    for stage in _LISTED_BACKWARDS:
        by_stage[stage].reverse()
    return by_stage


def _settled(state: JevState, reply: AIMessage) -> bool:
    """Whether the run was already ended, or the reply was written by a Jev blueprint."""
    return bool(state.get("jev_blocked")) or JEV_MARK in reply.additional_kwargs


def _readable(message: ToolMessage) -> bool:
    """A result with text that no Jev blueprint has withheld or replaced."""
    return bool(message.text.strip()) and not message.additional_kwargs.get(JEV_WITHHELD)


def _summary(answers: Mapping[str, Answer]) -> dict[str, object]:
    """Each answer's headline value, for the decision event."""
    values: dict[str, object] = {}
    for qid, answer in answers.items():
        if isinstance(answer, NoulAnswer):
            values[qid] = answer.noul
        elif isinstance(answer, ChoiceAnswer):
            values[qid] = answer.choice
        elif isinstance(answer, ScoreAnswer):
            values[qid] = answer.score
    return values
