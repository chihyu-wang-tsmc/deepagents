"""Middleware for custom blueprints: one front per blueprint, and one back per position.

The front asks the blueprint's questions at the hook where the built-in blueprints of
its stage ask theirs, sends Jev the same kind of state they send, and carries out the
actions at the stage's own position. An action at a later position is carried out by
a back, placed beside the built-in blueprints of that position.

A back decides again from the answers that triggered the action: the entry request's,
or the `jev_answers` record the front left on the message it asked about. Front and
back use the same answers and the same decision, so they always agree, and an action
at a later position keeps acting until the run ends.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, TypeVar, cast

from langchain.agents.middleware.types import ModelRequest, hook_config
from langchain.chat_models import init_chat_model
from langchain_core.decisions.types import Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer
from langchain_core.messages import BaseMessage, ToolMessage
from pydantic import TypeAdapter

from langchain_jev._messages import (
    append_system_text,
    clip,
    current_turn,
    final_answer,
    latest_output,
    latest_request,
    new_tool_messages,
)
from langchain_jev.custom import (
    OWN,
    POSITIONS,
    AskHuman,
    DropTools,
    Mark,
    Notify,
    NoTools,
    Redact,
    Refuse,
    RewriteReply,
    RewriteToolResult,
    ShapeModelCall,
    ShapeToolCall,
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
    from collections.abc import Awaitable, Callable, Iterator

    from langchain.agents.middleware.types import ModelResponse, ToolCallRequest
    from langchain_core.decisions import ClassifierResponse, Question
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, AnyMessage
    from langgraph.runtime import Runtime
    from langgraph.types import Command

    from langchain_jev._runtime import JevRuntime
    from langchain_jev._types import Mode
    from langchain_jev.custom import Action, At, JevBlueprint, Placed
    from langchain_jev.state import JevState

JEV_END = "jev_end"
"""`additional_kwargs` key on a tool result a custom blueprint withheld to end the run."""

JEV_ANSWERS = "jev_answers"
"""`additional_kwargs` key on a message a front asked about: by blueprint, the answers
that triggered actions at later positions."""

_REQUEST_CHARS = 2_000
"""Longest user request sent alongside a tool result, as the tool-result guard sends."""

_ANSWER: TypeAdapter[Answer] = TypeAdapter(Answer)
MessageT = TypeVar("MessageT", bound=BaseMessage)
ExpectedT = TypeVar("ExpectedT")
Triggered = list[tuple["Action", Mapping[str, Answer]]]
"""Actions with the answers that first triggered each one."""


class CustomBlueprintMiddleware(JevMiddleware):
    """What fronts and backs share: the blueprint, its decision, and how actions apply."""

    position: ClassVar[At | None] = None
    """The position a back acts at; `None` for the front."""

    def __init__(self, jev: JevRuntime, spec: JevBlueprint) -> None:
        """Act for `spec`, sharing `jev` with the rest of the stack."""
        super().__init__(jev)
        self.spec = spec
        self.blueprint = spec.name
        self._acts: tuple[Placed, ...] = () if spec.decide else spec.check(spec.actions())
        self._models: dict[str, BaseChatModel] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """A node name of its own, so several custom blueprints fit in one agent."""
        where = f"@{self.position}" if self.position else ""
        return f"JevBlueprint[{self.spec.name}{where}]"

    def _questions(self) -> dict[str, Question]:
        return {qid: self.question(qid) for qid in self.spec.questions}

    def _decide(self, answers: Mapping[str, Answer]) -> tuple[Placed, ...]:
        keys = self.spec.when or self.spec.thresholds
        thresholds = {key: self.threshold(f"{self.blueprint}.{key}") for key in keys}
        if self.spec.decide is not None:
            return self.spec.check(self.spec.decide(answers, thresholds))
        fired = any(
            isinstance(answer := answers.get(qid), NoulAnswer) and answer.noul >= thresholds[qid]
            for qid in self.spec.when
        )
        return self._acts if fired else ()

    def _verdict(
        self, mode: Mode, answers: Mapping[str, Answer], **details: object
    ) -> tuple[Placed, ...]:
        """Decide and report it; the actions to take, which are none in shadow mode."""
        placed = self._decide(answers)
        action = "+".join(action_name(a) for a, _ in placed) or "pass"
        where: dict[str, object] = {"at": [at for _, at in placed]} if placed else {}
        self.record(mode, action, answers=_summary(answers), **where, **details)
        return placed if mode == "on" else ()

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

    def _recorded(
        self, message: MessageT, answers: Mapping[str, Answer], placed: tuple[Placed, ...]
    ) -> MessageT:
        """`message` carrying `answers`, when they trigger actions at a later position."""
        if not any(at != OWN[self.spec.stage] for _, at in placed):
            return message
        records = dict(_mapping(message.additional_kwargs.get(JEV_ANSWERS)))
        records[self.blueprint] = {qid: a.model_dump(mode="json") for qid, a in answers.items()}
        kwargs = {**message.additional_kwargs, JEV_ANSWERS: records}
        return message.model_copy(update={"additional_kwargs": kwargs})

    def _shaped(self, request: ModelRequest, actions: Sequence[Action]) -> ModelRequest:
        """`request` with every `model_call` action applied, in order."""
        for action in actions:
            if isinstance(action, SystemNote):
                note = append_system_text(request.system_message, action.text)
                request = request.override(system_message=note)
            elif isinstance(action, UseModel):
                request = request.override(model=self._model(action.model))
            elif isinstance(action, NoTools) and request.tools:
                request = request.override(tools=[])
            elif isinstance(action, DropTools):
                kept = [t for t in request.tools if _tool_name(t) not in action.names]
                request = request.override(tools=kept)
            elif isinstance(action, ShapeModelCall):
                request = _expect(action.shape(request), ModelRequest, self.blueprint)
        return request

    def _model(self, model: str | BaseChatModel) -> BaseChatModel:
        if not isinstance(model, str):
            return model
        with self._lock:
            if model not in self._models:
                self._models[model] = init_chat_model(model)
            return self._models[model]

    def _settled_reply(self, reply: AIMessage, actions: Sequence[Action]) -> tuple[AIMessage, bool]:
        """The reply after the `model_output` actions, and whether it ends the run."""
        refusal = next((a for a in actions if isinstance(a, Refuse)), None)
        if refusal is not None:
            text = refusal.message or self.jev.config.guard.refusal_message
            return jev_reply(text, self.blueprint, message_id=reply.id), True
        text = reply.text
        for action in actions:
            if isinstance(action, RewriteReply):
                text = _expect(action.rewrite(text), str, self.blueprint)
        if text == reply.text:
            return reply, False
        return reply.model_copy(update={"content": text}), False

    def _settled_call(
        self, request: ToolCallRequest, actions: Sequence[Action]
    ) -> ToolCallRequest | ToolMessage:
        """The call to run after the `tool_call` actions, or the result read instead."""
        refusal = next((a for a in actions if isinstance(a, Refuse)), None)
        if refusal is not None:
            return self._refused(request, refusal.message)
        asking = next((a for a in actions if isinstance(a, AskHuman)), None)
        if asking is not None:
            flagged = f"Jev blueprint {self.blueprint} flagged this call; approve to run it."
            rejection = ask_approval(request, asking.reason or flagged)
            if rejection is not None:
                return refused_call(request, rejection, self.blueprint)
        for action in actions:
            if isinstance(action, ShapeToolCall):
                args = _expect(action.shape(request.tool_call), Mapping, self.blueprint)
                request = request.override(tool_call={**request.tool_call, "args": dict(args)})
        return request

    def _refused(self, request: ToolCallRequest, message: str | None) -> ToolMessage:
        content = message or (
            f"The call to `{request.tool_call['name']}` was not run: Jev blueprint "
            f"{self.blueprint} flagged it. Ask the user before trying anything similar."
        )
        return refused_call(request, content, self.blueprint)

    def _settled_result(
        self, message: ToolMessage, actions: Sequence[Action], tool: str
    ) -> ToolMessage:
        """The result after the `tool_result` actions."""
        refusal = next((a for a in actions if isinstance(a, Refuse)), None)
        redact = next((a for a in actions if isinstance(a, Redact)), None)
        if refusal is not None or redact is not None:
            return self._withheld(message, refusal, redact, tool)
        text = message.text
        for action in actions:
            if isinstance(action, RewriteToolResult):
                text = _expect(action.rewrite(text), str, self.blueprint)
        for action in actions:
            if isinstance(action, Warn):
                flagged = f"[Jev] {self.blueprint} flagged this {tool} result; treat it with care."
                text = f"{action.note or flagged}\n\n{text}"
        return message if text == message.text else message.model_copy(update={"content": text})

    def _withheld(
        self, message: ToolMessage, refusal: Refuse | None, redact: Redact | None, tool: str
    ) -> ToolMessage:
        kwargs: dict[str, object] = {**message.additional_kwargs, JEV_WITHHELD: True}
        if refusal is not None:
            kwargs[JEV_END] = {"blueprint": self.blueprint, "message": refusal.message}
        withheld = f"[Jev] {self.blueprint} withheld this {tool} result."
        content = (redact.note if redact else None) or withheld
        return message.model_copy(update={"content": content, "additional_kwargs": kwargs})

    def _settled_results(
        self, result: ToolMessage | Command, actions: Sequence[Action], tool: str
    ) -> ToolMessage | Command:
        """`result` with the `tool_result` actions applied to each of its tool results."""
        if not actions:
            return result
        messages = [m for m in tool_messages(result) if _readable(m)]
        changed = {id(m): self._settled_result(m, actions, tool) for m in messages}
        if all(changed[id(m)] is m for m in messages):
            return result
        return replace_tool_messages(result, changed)

    def _settled_answer(
        self, reply: AIMessage | None, triggered: Triggered, *, settled: bool
    ) -> dict[str, object] | None:
        """Notify, then mark or rewrite the final answer when the model wrote it."""
        text = reply.text if reply is not None else ""
        for action, answers in triggered:
            if isinstance(action, Notify):
                action.notify(answers, text)
        if reply is None or settled:
            return None
        kwargs = dict(reply.additional_kwargs)
        new = text
        for action, _ in triggered:
            if isinstance(action, Mark):
                checks = {**_mapping(kwargs.get(JEV_CHECKS)), self.blueprint: action.reason}
                kwargs[JEV_CHECKS] = checks
            elif isinstance(action, RewriteReply):
                new = _expect(action.rewrite(new), str, self.blueprint)
        if new == text and kwargs == reply.additional_kwargs:
            return None
        update = {"content": new, "additional_kwargs": kwargs}
        return {"messages": [reply.model_copy(update=update)]}


# -- fronts: ask Jev, and act at the stage's own position --------------------------------


class EntryFront(CustomBlueprintMiddleware):
    """Ask in the entry request; refuse at once when told to."""

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
        refusal = _first_refusal(self._verdict(mode, answers), "before_agent")
        return self._end(refusal) if refusal is not None else None


class ModelOutputFront(CustomBlueprintMiddleware):
    """Ask about the text of each model reply; refuse or rewrite it when told to."""

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
        placed = self._verdict(mode, response.answers)
        message, ended = self._settled_reply(reply, _at(placed, "model_output"))
        message = self._recorded(message, response.answers, placed)
        if message is reply:
            return None
        update: dict[str, object] = {"messages": [message]}
        return {**update, "jump_to": "end", "jev_blocked": True} if ended else update


class ToolCallFront(CustomBlueprintMiddleware):
    """Ask about a listed tool's call before it runs; act on the call and on its result."""

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
        if tool not in self.spec.tools or mode == "off":
            return cast("ToolMessage | Command", (yield Handle(request)))
        ask = Ask(call_state(request), self._questions())
        response = cast("ClassifierResponse | None", (yield ask))
        if response is None:
            if not self._unavailable(mode, tool=tool):
                return cast("ToolMessage | Command", (yield Handle(request)))
            unchecked = (
                f"The call to `{tool}` was not run: Jev could not be reached to check it "
                f"for {self.blueprint}. Ask the user before trying again."
            )
            return self._refused(request, unchecked)
        placed = self._verdict(mode, response.answers, tool=tool)
        return (yield from self._run(request, response.answers, placed))

    def _run(
        self, request: ToolCallRequest, answers: Mapping[str, Answer], placed: tuple[Placed, ...]
    ) -> Flow[ToolMessage | Command]:
        """Act on the call, run it unless refused, act on its result, and record the answers."""
        tool = request.tool_call["name"]
        outcome = self._settled_call(request, self._for_tool(placed, "tool_call", tool))
        if isinstance(outcome, ToolMessage):
            return self._recorded(outcome, answers, placed)
        result = cast("ToolMessage | Command", (yield Handle(outcome)))
        result = self._settled_results(result, self._for_tool(placed, "tool_result", tool), tool)
        recorded = {id(m): self._recorded(m, answers, placed) for m in tool_messages(result)}
        return replace_tool_messages(result, recorded)

    def _for_tool(self, placed: tuple[Placed, ...], at: At, tool: str) -> list[Action]:
        return [a for a, where in placed if where == at and tool in self.spec.tools_for(a, at)]


class ToolResultFront(CustomBlueprintMiddleware):
    """Ask about a listed tool's result as it returns; act on it when told to."""

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
        changed = {id(m): self._handled(mode, m, r, tool) for m, r in pairs}
        return replace_tool_messages(result, changed)

    def _state(self, request: str, message: ToolMessage, tool: str) -> dict[str, object]:
        return {
            "user_request": request,
            "tool_name": tool,
            "tool_output": clip(message.text, self.jev.max_chars()),
        }

    def _handled(
        self, mode: Mode, message: ToolMessage, response: ClassifierResponse | None, tool: str
    ) -> ToolMessage:
        """The result to keep in place of `message`."""
        if response is None:
            if not self._unavailable(mode, tool=tool):
                return message
            unchecked = (
                f"[Jev] {self.blueprint} could not check this {tool} result, so it was withheld."
            )
            return self._settled_result(message, [Redact(unchecked)], tool)
        placed = self._verdict(mode, response.answers, tool=tool)
        settled = self._settled_result(message, _at(placed, "tool_result"), tool)
        return self._recorded(settled, response.answers, placed)


class FinalAnswerFront(CustomBlueprintMiddleware):
    """Ask about the final answer when the run ends; mark, rewrite, or notify."""

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
            if not self._unavailable(mode):
                return None
            unchecked: Triggered = [(Mark("Jev could not be reached to check this answer"), {})]
            return self._settled_answer(reply, unchecked, settled=False)
        placed = self._verdict(mode, response.answers)
        triggered: Triggered = [(a, response.answers) for a in _at(placed, "final_answer")]
        return self._settled_answer(reply, triggered, settled=False)


# -- backs: act at a later position, from the answers that triggered the action ---------


class _Back(CustomBlueprintMiddleware):
    """Find the actions this run's answers trigger at this back's position."""

    def _triggered(self, state: Mapping[str, object], tool: str | None = None) -> Triggered:
        """Each distinct action for this position, with the answers that first triggered it."""
        found: Triggered = []
        for answers in self._sources(state):
            for action, at in self._decide(answers):
                if at != self.position or any(action == seen for seen, _ in found):
                    continue
                if tool is None or tool in self.spec.tools_for(action, at):
                    found.append((action, answers))
        return found

    def _sources(self, state: Mapping[str, object]) -> Iterator[Mapping[str, Answer]]:
        """The entry answers, or the answers recorded on this run's messages."""
        if self.spec.stage == "entry":
            answers = entry_answers(state, self.blueprint)
            if answers:
                yield answers
            return
        messages = cast("list[AnyMessage]", state.get("messages", []))
        for message in current_turn(messages):
            records = _mapping(message.additional_kwargs.get(JEV_ANSWERS))
            record = _mapping(records.get(self.blueprint))
            if record:
                yield {qid: _ANSWER.validate_python(value) for qid, value in record.items()}

    def _on(self, state: Mapping[str, object], runtime: object, part: str | None = None) -> bool:
        return self.mode(state, runtime, part) == "on"

    def _applied(self, actions: Sequence[Action], **details: object) -> None:
        """Report what this back did, such as `refuse@tool_call`."""
        for action in actions:
            self.record("on", f"{action_name(action)}@{self.position}", **details)


class ModelCallBack(_Back):
    """Shape every model call the run's answers call for: note, model, or tools."""

    position = "model_call"

    def __init__(self, jev: JevRuntime, spec: JevBlueprint) -> None:
        """Build a fixed `UseModel` model up front, so a bad model name fails early."""
        super().__init__(jev, spec)
        for action, at in self._acts:
            if at == "model_call" and isinstance(action, UseModel):
                self._model(action.model)

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Call the model as the blueprint's actions say."""
        return handler(self._call(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Call the model as the blueprint's actions say."""
        return await handler(self._call(request))

    def _call(self, request: ModelRequest) -> ModelRequest:
        if not self._on(request.state, request.runtime):
            return request
        return self._shaped(request, [a for a, _ in self._triggered(request.state)])


class BeforeModelBack(_Back):
    """End the run as a turn starts, after a `Refuse` the run's answers call for."""

    position = "before_model"

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """End the run for a withheld tool result, or for a `Refuse` at `before_model`."""
        for message in new_tool_messages(state["messages"]):
            ended = _mapping(message.additional_kwargs.get(JEV_END))
            if ended.get("blueprint") == self.blueprint:
                return self._end(Refuse(cast("str | None", ended.get("message"))))
        if not self._on(state, runtime):
            return None
        refusal = next((a for a, _ in self._triggered(state) if isinstance(a, Refuse)), None)
        if refusal is None:
            return None
        self._applied([refusal])
        return self._end(refusal)


class ModelOutputBack(_Back):
    """Refuse or rewrite each model reply after the run's answers call for it."""

    position = "model_output"

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """Act on the newest reply."""
        reply = latest_output(state["messages"])
        if reply is None or _settled(state, reply) or not self._on(state, runtime):
            return None
        actions = [a for a, _ in self._triggered(state)]
        message, ended = self._settled_reply(reply, actions)
        if message is reply:
            return None
        self._applied([a for a in actions if isinstance(a, Refuse)][:1] if ended else actions)
        update: dict[str, object] = {"messages": [message]}
        return {**update, "jump_to": "end", "jev_blocked": True} if ended else update


class ToolCallBack(_Back):
    """Refuse, pause, or change listed tools' calls after the run's answers call for it."""

    position = "tool_call"

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Act before the tool runs."""
        outcome = self._before(request)
        return outcome if isinstance(outcome, ToolMessage) else handler(outcome)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Act before the tool runs."""
        outcome = self._before(request)
        return outcome if isinstance(outcome, ToolMessage) else await handler(outcome)

    def _before(self, request: ToolCallRequest) -> ToolCallRequest | ToolMessage:
        tool = request.tool_call["name"]
        if not self._on(request.state, request.runtime, tool):
            return request
        actions = [a for a, _ in self._triggered(request.state, tool)]
        if not actions:
            return request
        outcome = self._settled_call(request, actions)
        stopped = [a for a in actions if isinstance(a, (Refuse, AskHuman))][:1]
        self._applied(stopped if isinstance(outcome, ToolMessage) else actions, tool=tool)
        return outcome


class ToolResultBack(_Back):
    """Act on listed tools' results after the run's answers call for it."""

    position = "tool_result"

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Act after the tool returns."""
        return self._after(request, handler(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Act after the tool returns."""
        return self._after(request, await handler(request))

    def _after(
        self, request: ToolCallRequest, result: ToolMessage | Command
    ) -> ToolMessage | Command:
        tool = request.tool_call["name"]
        if not self._on(request.state, request.runtime, tool):
            return result
        actions = [a for a, _ in self._triggered(request.state, tool)]
        settled = self._settled_results(result, actions, tool)
        if settled is not result:
            self._applied(actions, tool=tool)
        return settled


class FinalAnswerBack(_Back):
    """Mark, rewrite, or notify at the end of a run whose answers call for it."""

    position = "final_answer"

    def after_agent(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """Act on the final answer."""
        if not self._on(state, runtime):
            return None
        triggered = self._triggered(state)
        if not triggered:
            return None
        reply = final_answer(state["messages"])
        self._applied([a for a, _ in triggered])
        settled = reply is None or _settled(state, reply)
        return self._settled_answer(reply, triggered, settled=settled)


_FRONTS: dict[str, type[CustomBlueprintMiddleware]] = {
    "entry": EntryFront,
    "model_output": ModelOutputFront,
    "tool_call": ToolCallFront,
    "tool_result": ToolResultFront,
    "final_answer": FinalAnswerFront,
}
_BACKS: dict[str, type[_Back]] = {
    "model_call": ModelCallBack,
    "before_model": BeforeModelBack,
    "model_output": ModelOutputBack,
    "tool_call": ToolCallBack,
    "tool_result": ToolResultBack,
    "final_answer": FinalAnswerBack,
}
GROUPS: tuple[str, ...] = (
    "entry",
    "model_call",
    "model_output",
    "tool_call",
    "tool_result",
    "before_model",
    "final_answer",
)
"""Where custom middleware sits in the stack: the entry fronts, then each position."""

_LISTED_BACKWARDS = frozenset({"model_output", "tool_result", "final_answer"})
"""Groups whose hooks run from the end of the stack back: `after_model`, the part of
`wrap_tool_call` after the tool returns, and `after_agent`."""


def custom_middleware(jev: JevRuntime) -> dict[str, list[CustomBlueprintMiddleware]]:
    """The installed custom blueprints' fronts and backs, by group, in stack order.

    Within a group, blueprints act in the order `JevConfig.custom_blueprints` lists
    them, so groups whose hooks run backwards are listed backwards.
    """
    groups: dict[str, list[CustomBlueprintMiddleware]] = {group: [] for group in GROUPS}
    for spec in jev.config.custom_blueprints:
        if spec.name not in jev.installed:
            continue
        groups[spec.stage].append(_FRONTS[spec.stage](jev, spec))
        for at in back_positions(spec):
            groups[at].append(_BACKS[at](jev, spec))
    for group in _LISTED_BACKWARDS:
        groups[group].reverse()
    return groups


def back_positions(spec: JevBlueprint) -> tuple[At, ...]:
    """The positions where `spec` needs a back: every later position its actions use.

    A `Refuse` at `tool_result` ends the run when the next turn starts, so a blueprint
    that may refuse there also gets a `before_model` back.
    """
    backs = set(spec.positions()) - {OWN[spec.stage]}
    if spec.decide is not None:
        ends = "tool_result" in spec.positions()
    else:
        ends = any(isinstance(a, Refuse) and spec.place(a) == "tool_result" for a in spec.actions())
    if ends:
        backs.add("before_model")
    return tuple(at for at in POSITIONS if at in backs)


def _at(placed: tuple[Placed, ...], at: At) -> list[Action]:
    return [action for action, where in placed if where == at]


def _first_refusal(placed: tuple[Placed, ...], at: At) -> Refuse | None:
    return next((a for a in _at(placed, at) if isinstance(a, Refuse)), None)


def _settled(state: Mapping[str, object], reply: AIMessage) -> bool:
    """Whether the run was already ended, or the reply was written by a Jev blueprint."""
    return bool(state.get("jev_blocked")) or JEV_MARK in reply.additional_kwargs


def _readable(message: ToolMessage) -> bool:
    """A result with text that no Jev blueprint has withheld or replaced."""
    return bool(message.text.strip()) and not message.additional_kwargs.get(JEV_WITHHELD)


def _mapping(value: object) -> Mapping[str, object]:
    return cast("Mapping[str, object]", value) if isinstance(value, Mapping) else {}


def _tool_name(tool: object) -> str | None:
    if isinstance(tool, Mapping):
        function = _mapping(tool.get("function"))
        return cast("str | None", tool.get("name") or function.get("name"))
    return getattr(tool, "name", None)


def _expect(value: object, kind: type[ExpectedT], blueprint: str) -> ExpectedT:
    """`value`, after checking that a function in an action returned the right kind."""
    if not isinstance(value, kind):
        msg = (
            f"A function in an action of Jev blueprint {blueprint!r} returned "
            f"{type(value).__name__}; it must return {kind.__name__}."
        )
        raise TypeError(msg)
    return value


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
