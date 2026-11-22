"""Run bookkeeping and the single Jev request asked about each user message."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, cast

from langchain_core.decisions.types import Answer
from langgraph.types import Overwrite
from pydantic import TypeAdapter

from langchain_jev._messages import clip, earlier_turns, latest_request
from langchain_jev.middleware._base import Ask, AskMany, Flow, JevMiddleware
from langchain_jev.switches import mode_for

if TYPE_CHECKING:
    from langchain_core.decisions import BaseDecisionModel, ClassifierResponse, Question
    from langchain_core.messages import AnyMessage
    from langgraph.runtime import Runtime

    from langchain_jev._runtime import JevRuntime
    from langchain_jev._types import Mode
    from langchain_jev.state import JevState

_ANSWER = TypeAdapter(Answer)
_FAILED = "__failed__"
_TURN_CHARS = 1_000
"""Longest text kept from each earlier message in the entry request."""
_Group = tuple[tuple[str, ...], dict[str, "Question"]]
"""The blueprints sharing one entry request, and its questions keyed `blueprint::qid`."""


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
        groups = self._groups(snapshot)
        request = latest_request(state["messages"])
        if not groups or not request:
            return None
        message = self._state(state["messages"], request)
        asks = tuple(Ask(message, questions, owners) for owners, questions in groups)
        if len(asks) == 1:
            responses = [cast("ClassifierResponse | None", (yield asks[0]))]
        else:
            responses = cast("list[ClassifierResponse | None]", (yield AskMany(asks)))
        return {"jev_entry": _entry_record(groups, responses)}

    def _groups(self, snapshot: Mapping[str, Mode]) -> list[_Group]:
        """The entry questions, one request per decision model the blueprints use."""
        groups: list[tuple[BaseDecisionModel, list[str], dict[str, Question]]] = []
        for participant in self.participants:
            blueprint = participant.blueprint
            if mode_for(snapshot, blueprint) == "off":
                continue
            asked = participant.entry_questions(snapshot)
            if not asked:
                continue
            classifier = self.jev.classifier(blueprint)
            group = next((g for g in groups if g[0] is classifier), None)
            if group is None:
                group = (classifier, [], {})
                groups.append(group)
            group[1].append(blueprint)
            group[2].update({f"{blueprint}::{qid}": q for qid, q in asked.items()})
        return [(tuple(owners), questions) for _, owners, questions in groups]

    def _state(self, messages: Sequence[AnyMessage], request: str) -> dict[str, object]:
        """The latest user message, and the turns before it when there are any."""
        state: dict[str, object] = {"message": clip(request, self.jev.max_chars())}
        limit = min(_TURN_CHARS, self.jev.max_chars())
        earlier = earlier_turns(messages, self.jev.config.guard.input_history, limit)
        if earlier:
            state["earlier_messages"] = earlier
        return state


def _entry_record(
    groups: Sequence[_Group], responses: Sequence[ClassifierResponse | None]
) -> dict[str, dict]:
    """Every answer by `blueprint::qid`, plus the blueprints whose request failed."""
    record: dict[str, dict] = {}
    failed: dict[str, bool] = {}
    for (owners, _), response in zip(groups, responses, strict=True):
        if response is None:
            failed.update(dict.fromkeys(owners, True))
        else:
            record.update({k: a.model_dump(mode="json") for k, a in response.answers.items()})
    if failed:
        record[_FAILED] = failed
    return record


def entry_answers(state: Mapping[str, object], blueprint: str) -> dict[str, Answer] | None:
    """This blueprint's answers from the entry request, or `None` if it failed."""
    entry = state.get("jev_entry")
    if not isinstance(entry, Mapping):
        return None
    failed = entry.get(_FAILED)
    if isinstance(failed, Mapping) and (not failed or blueprint in failed):
        return None
    prefix = f"{blueprint}::"
    return {
        key.removeprefix(prefix): _ANSWER.validate_python(value)
        for key, value in entry.items()
        if key.startswith(prefix)
    }
