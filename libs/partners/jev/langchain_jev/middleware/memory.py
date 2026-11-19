"""Remember lasting facts about a user, and bring them back in later runs.

Follows https://docs.typesafe.ai/cookbooks/basic_memory: after each run one Noul asks
whether the user stated a lasting fact; saved facts then join the state of later Jev
requests (so routing and guards can use them) and the agent's prompt.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from langchain_core.messages import HumanMessage
from langgraph.config import get_config

from langchain_jev._messages import clip, final_answer, latest_request
from langchain_jev._types import MEMORY_READ, MEMORY_WRITE
from langchain_jev.memory_store import GraphStoreMemory, JevMemoryStore
from langchain_jev.middleware._base import Ask, AskMany, Flow, JevMiddleware
from langchain_jev.middleware.guardrails import JEV_MARK

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain_core.decisions import ClassifierResponse
    from langgraph.runtime import Runtime

    from langchain_jev.state import JevState

_NAMESPACE = "jev_memory"
_MEMORY_INTRO = "Things this user said earlier that still hold:"


class _Memory(JevMiddleware):
    def _store(self, runtime: object) -> JevMemoryStore:
        configured = self.jev.config.memory.store
        if configured is not None:
            return configured
        store = getattr(runtime, "store", None)
        return GraphStoreMemory(store) if store is not None else self.jev.memory

    def _namespace(self, runtime: object) -> tuple[str, ...]:
        key = self.jev.config.memory.user_key
        context = getattr(runtime, "context", None)
        user = context.get(key) if isinstance(context, Mapping) else getattr(context, key, None)
        if user is None:
            try:
                user = get_config().get("configurable", {}).get("thread_id")
            except RuntimeError:  # outside a graph run
                user = None
        return (_NAMESPACE, str(user or "default"))


class MemoryReadMiddleware(_Memory):
    """Load saved facts when a run starts and show them to Jev and the agent."""

    blueprint = MEMORY_READ

    def before_agent(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """Load synchronously."""
        return self.drive(self._flow(state, runtime))

    async def abefore_agent(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """Load asynchronously."""
        return await self.adrive(self._flow(state, runtime))

    def _flow(self, state: JevState, runtime: Runtime) -> Flow[dict[str, object] | None]:
        mode = self.mode(state, runtime)
        if mode == "off":
            return {"jev_memories": []}
        records = self._store(runtime).load(
            self._namespace(runtime), self.jev.config.memory.max_items
        )
        request = latest_request(state["messages"])
        if records and request and self.jev.config.memory.filter_relevant:
            records = cast("list[dict]", (yield from self._relevant(request, records)))
        self.record(mode, "load", count=len(records))
        return {"jev_memories": records if mode == "on" else []}

    def _relevant(self, request: str, records: list[dict]) -> Flow[list[dict]]:
        asks = tuple(
            Ask(
                {"request": request, "memory": r.get("message", "")},
                {"relevant": self.question("relevant")},
            )
            for r in records
        )
        responses = cast("list[ClassifierResponse | None]", (yield AskMany(asks)))
        limit = self.threshold("memory_read.relevant_min")
        return [
            r
            for r, resp in zip(records, responses, strict=True)
            if resp is None or resp.nouls["relevant"].noul >= limit
        ]

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Show saved facts to the model."""
        return handler(self._remembered(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Show saved facts to the model."""
        return await handler(self._remembered(request))

    def _remembered(self, request: ModelRequest) -> ModelRequest:
        records = request.state.get("jev_memories") or []
        if not records or self.mode(request.state, request.runtime) != "on":
            return request
        facts = "\n".join(f"- {r.get('message', '')}" for r in records)
        content = f"<user_memory>\n{_MEMORY_INTRO}\n{facts}\n</user_memory>"
        note = HumanMessage(content=content, additional_kwargs={JEV_MARK: MEMORY_READ})
        return request.override(messages=_before_latest_human(request.messages, note))


class MemoryWriteMiddleware(_Memory):
    """After each run, save the user's message when it states a lasting fact."""

    blueprint = MEMORY_WRITE

    def after_agent(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """Decide synchronously."""
        return self.drive(self._flow(state, runtime))

    async def aafter_agent(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """Decide asynchronously."""
        return await self.adrive(self._flow(state, runtime))

    def _flow(self, state: JevState, runtime: Runtime) -> Flow[None]:
        mode = self.mode(state, runtime)
        request = latest_request(state["messages"])
        reply = final_answer(state["messages"])
        if mode == "off" or state.get("jev_blocked") or not request:
            return None
        record = {
            "request": {"message": clip(request, self.jev.max_chars())},
            "assistant_reply": clip(reply.text if reply else "", 2_000),
        }
        asked = {"remember": self.question("remember")}
        response = cast("ClassifierResponse | None", (yield Ask(record, asked)))
        if response is None:
            return None
        probability = response.nouls["remember"].noul
        save = probability >= self.threshold("memory_write.remember_at")
        self.record(mode, "save" if save else "skip", remember=probability)
        if save and mode == "on":
            saved = {
                "message": request,
                "remember": probability,
                "saved_at": datetime.now(UTC).isoformat(),
            }
            self._store(runtime).save(self._namespace(runtime), saved)
        return None


def _before_latest_human(messages: list, note: HumanMessage) -> list:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return [*messages[:index], note, *messages[index:]]
    return [note, *messages]
