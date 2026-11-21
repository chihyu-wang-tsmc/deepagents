"""Agent state the Jev middleware share within one agent.

Every key is private: it stays out of the agent's input and output, is not copied
into subagents, and is not written back to the parent, so a subagent's decisions
never overwrite its parent's.
"""

from __future__ import annotations

from typing import Annotated, NotRequired

from langchain.agents.middleware.types import AgentState, PrivateStateAttr

from langchain_jev._types import Mode  # noqa: TC001 - LangGraph reads state annotations at runtime


def _passage_key(passage: dict) -> tuple[str, str]:
    return str(passage.get("id", "")), str(passage.get("text", "")).strip()


def _extend(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    """Combine evidence from every retrieval in the run, parallel calls included.

    A passage found again (same id and same text) is kept once, in its latest place,
    so the newest evidence stays last. Passages that only share an id are both kept:
    a source without ids is numbered from 1 on every call.
    """
    merged = [*(left or []), *(right or [])]
    last = {_passage_key(p): i for i, p in enumerate(merged)}
    return [p for i, p in enumerate(merged) if last[_passage_key(p)] == i]


class JevState(AgentState):
    """Private keys the Jev blueprints read and write."""

    jev_modes: NotRequired[Annotated[dict[str, Mode], PrivateStateAttr]]
    """Blueprint modes for this run, read once when the run starts."""

    jev_entry: NotRequired[Annotated[dict[str, dict], PrivateStateAttr]]
    """Answers to the questions asked about the user's message, by question id."""

    jev_blocked: NotRequired[Annotated[bool, PrivateStateAttr]]
    """Set when a guard ended the run, so later blueprints skip it."""

    jev_review: NotRequired[Annotated[str | None, PrivateStateAttr]]
    """The hazard an input guard sent to review, if any."""

    jev_skill_hint: NotRequired[Annotated[str | None, PrivateStateAttr]]
    """The skill suggestion shown to the model in this run."""

    jev_skill_done: NotRequired[Annotated[bool, PrivateStateAttr]]
    """Whether skill suggestion already ran in this run."""

    jev_evidence: NotRequired[Annotated[list[dict], PrivateStateAttr, _extend]]
    """Passages the RAG blueprint accepted in this run."""
