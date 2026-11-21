"""Run bookkeeping and the single Jev request asked about each user message."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, cast

from langchain_core.decisions.types import Answer
from langgraph.types import Overwrite
from pydantic import TypeAdapter

from langchain_jev._messages import clip, latest_request
from langchain_jev.middleware._base import Ask, Flow, JevMiddleware
from langchain_jev.switches import mode_for

if TYPE_CHECKING:
    from langchain_core.decisions import ClassifierResponse, Question
    from langgraph.runtime import Runtime

    from langchain_jev._runtime import JevRuntime
    from langchain_jev._types import Mode
    from langchain_jev.state import JevState

_ANSWER = TypeAdapter(Answer)
_FAILED = "__failed__"


class EntryParticipant(Protocol):
    """A blueprint whose questions ride along in the entry request."""

    blueprint: str

    def entry_questions(self, snapshot: Mapping[str, Mode]) -> dict[str, Question]:
        """Questions about the user's message, keyed by local id."""
        ...


class JevRunMiddleware(JevMiddleware):
    """Read this run's modes once and clear what the previous run left behind."""

    def before_agent(self, state: JevState, runtime: Runtime) -> dict[str, object]:  # noqa: ARG002 - state is reset, not read
        """Snapshot the modes for the whole run."""
        return {
            "jev_modes": self.jev.snapshot(runtime.context),
            "jev_entry": {},
            "jev_blocked": False,
            "jev_review": None,
            "jev_skill_hint": None,
            "jev_skill_done": False,
            "jev_evidence": Overwrite([]),
        }


class JevEntryMiddleware(JevMiddleware):
    """Ask every entry blueprint's questions about the user's message in one request.

    Batching questions that share a state is what
    https://docs.typesafe.ai/patterns/fan-out recommends: one request costs about the
    same as one question.
    """

    def __init__(self, jev: JevRuntime, participants: Sequence[EntryParticipant]) -> None:
        """Collect questions from `participants` at the start of each run."""
        super().__init__(jev)
        self.participants = list(participants)

    def before_agent(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """Ask the entry questions synchronously."""
        return self.drive(self._flow(state, runtime))

    async def abefore_agent(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """Ask the entry questions asynchronously."""
        return await self.adrive(self._flow(state, runtime))

    def _flow(self, state: JevState, runtime: Runtime) -> Flow[dict[str, object] | None]:
        snapshot = cast(
            "Mapping[str, Mode]",
            state.get("jev_modes") or self.jev.snapshot(runtime.context),
        )
        questions = self._questions(snapshot)
        request = latest_request(state["messages"])
        if not questions or not request:
            return None
        response = cast(
            "ClassifierResponse | None",
            (yield Ask(self._state(request), questions)),
        )
        if response is None:
            return {"jev_entry": {_FAILED: {}}}
        return {"jev_entry": {k: a.model_dump(mode="json") for k, a in response.answers.items()}}

    def _questions(self, snapshot: Mapping[str, Mode]) -> dict[str, Question]:
        questions: dict[str, Question] = {}
        for participant in self.participants:
            if mode_for(snapshot, participant.blueprint) == "off":
                continue
            for qid, question in participant.entry_questions(snapshot).items():
                questions[f"{participant.blueprint}::{qid}"] = question
        return questions

    def _state(self, request: str) -> dict[str, object]:
        return {"message": clip(request, self.jev.max_chars())}


def entry_answers(state: Mapping[str, object], blueprint: str) -> dict[str, Answer] | None:
    """This blueprint's answers from the entry request, or `None` if it failed."""
    entry = state.get("jev_entry")
    if not isinstance(entry, Mapping) or _FAILED in entry:
        return None
    prefix = f"{blueprint}::"
    return {
        key.removeprefix(prefix): _ANSWER.validate_python(value)
        for key, value in entry.items()
        if key.startswith(prefix)
    }
