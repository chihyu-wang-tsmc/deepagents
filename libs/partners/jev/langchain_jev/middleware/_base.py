"""The base every Jev blueprint middleware builds on.

Each blueprint writes its logic once, as a generator that yields what it needs from
the outside world: an `Ask` (one Jev request), an `AskMany` (several, in parallel),
or a `Handle` (run the next handler, such as the tool or the model). `drive` and
`adrive` answer those steps synchronously or asynchronously, so every middleware
works with both `invoke` and `ainvoke` without two copies of its rules.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Generator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeVar, cast

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.runnables.config import ContextThreadPoolExecutor

from langchain_jev._types import JevEvent, Mode
from langchain_jev.state import JevState
from langchain_jev.switches import mode_for

if TYPE_CHECKING:
    from langchain_core.decisions import ClassifierResponse, Question

    from langchain_jev._runtime import JevRuntime

ResultT = TypeVar("ResultT")
_PARALLEL = 4
"""Most Jev requests one hook sends at once; public endpoints rate-limit."""


@dataclass(frozen=True)
class Ask:
    """One Jev request."""

    state: object
    """JSON-compatible state; LangChain messages are converted when sent."""
    questions: Mapping[str, Question]


@dataclass(frozen=True)
class AskMany:
    """Several Jev requests sent in parallel; answers come back in order."""

    asks: tuple[Ask, ...]


@dataclass(frozen=True)
class Handle:
    """Run the next handler (the tool, or the model) with `request`."""

    request: object


Step = Ask | AskMany | Handle
Flow = Generator[Step, object, ResultT]


class JevMiddleware(AgentMiddleware[JevState]):
    """Shared plumbing: modes, thresholds, events, and the step drivers."""

    blueprint: ClassVar[str] = ""
    state_schema = JevState

    def __init__(self, jev: JevRuntime) -> None:
        """Share `jev` with the rest of the stack."""
        super().__init__()
        self.jev = jev

    def mode(self, state: Mapping[str, object], runtime: object, part: str | None = None) -> Mode:
        """This blueprint's mode in the current run (or one part of it)."""
        snapshot = state.get("jev_modes")
        if not isinstance(snapshot, Mapping):
            snapshot = self.jev.snapshot(getattr(runtime, "context", None))
        return mode_for(cast("Mapping[str, Mode]", snapshot), self.blueprint, part)

    def threshold(self, name: str) -> float:
        """A threshold by full name, for example `rag_passages.relevant_min`."""
        return self.jev.thresholds[name]

    def question(self, name: str, /, **values: str) -> Question:
        """This blueprint's question `name`, with any wording from `JevConfig.questions`."""
        return self.jev.questions.question(f"{self.blueprint}.{name}", **values)

    def instructions(self, name: str) -> str:
        """Instructions for this blueprint's choice `name`, whose options come at run time."""
        return self.jev.questions.instructions(f"{self.blueprint}.{name}")

    def record(self, mode: Mode, action: str, **details: object) -> None:
        """Report a decision (or, in shadow mode, the decision it would have made)."""
        self.jev.emit(JevEvent(self.blueprint, mode, action, dict(details)))

    @property
    def fails_closed(self) -> bool:
        """Whether this blueprint refuses when Jev cannot answer."""
        return self.blueprint in self.jev.config.fail_closed

    def drive(
        self, flow: Flow[ResultT], handler: Callable[[object], object] | None = None
    ) -> ResultT:
        """Run `flow`, answering its steps synchronously."""
        try:
            step = next(flow)
            while True:
                step = flow.send(self._sync_step(step, handler))
        except StopIteration as done:
            return cast("ResultT", done.value)

    async def adrive(
        self,
        flow: Flow[ResultT],
        handler: Callable[[object], Awaitable[object]] | None = None,
    ) -> ResultT:
        """Run `flow`, answering its steps asynchronously."""
        try:
            step = next(flow)
            while True:
                step = flow.send(await self._async_step(step, handler))
        except StopIteration as done:
            return cast("ResultT", done.value)

    def _sync_step(self, step: Step, handler: Callable[[object], object] | None) -> object:
        if isinstance(step, Ask):
            return self._ask(step)
        if isinstance(step, AskMany):
            # Copy the run's context into each worker so tracing callbacks see the requests.
            with ContextThreadPoolExecutor(max_workers=_PARALLEL) as pool:
                return list(pool.map(self._ask, step.asks))
        if handler is None:
            msg = "This hook has no handler to run."
            raise RuntimeError(msg)
        return handler(step.request)

    async def _async_step(
        self, step: Step, handler: Callable[[object], Awaitable[object]] | None
    ) -> object:
        if isinstance(step, Ask):
            return await self._aask(step)
        if isinstance(step, AskMany):
            limit = asyncio.Semaphore(_PARALLEL)

            async def bounded(ask: Ask) -> ClassifierResponse | None:
                async with limit:
                    return await self._aask(ask)

            return list(await asyncio.gather(*(bounded(ask) for ask in step.asks)))
        if handler is None:
            msg = "This hook has no handler to run."
            raise RuntimeError(msg)
        return await handler(step.request)

    def _ask(self, step: Ask) -> ClassifierResponse | None:
        try:
            return self.jev.ask(self.blueprint, step.state, step.questions)
        except Exception as error:  # noqa: BLE001 - the blueprint's fail policy decides what a failed request means
            self.record("on", "error", error=type(error).__name__, message=str(error)[:300])
            return None

    async def _aask(self, step: Ask) -> ClassifierResponse | None:
        try:
            return await self.jev.aask(self.blueprint, step.state, step.questions)
        except Exception as error:  # noqa: BLE001 - the blueprint's fail policy decides what a failed request means
            self.record("on", "error", error=type(error).__name__, message=str(error)[:300])
            return None
