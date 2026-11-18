"""Agent state the Jev middleware share within one agent.

Every key is private: it stays out of the agent's input and output, is not copied
into subagents, and is not written back to the parent, so a subagent's decisions
never overwrite its parent's.
"""

from __future__ import annotations

from typing import Annotated, NotRequired

from langchain.agents.middleware.types import AgentState, PrivateStateAttr

from langchain_jev._types import Mode  # noqa: TC001 - LangGraph reads state annotations at runtime


def _extend(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    """Combine evidence written by parallel tool calls in the same step."""
    return [*(left or []), *(right or [])]


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

    jev_memories: NotRequired[Annotated[list[dict], PrivateStateAttr]]
    """Facts saved for this user, loaded when the run starts."""

    jev_skill_hint: NotRequired[Annotated[str | None, PrivateStateAttr]]
    """The skill suggestion shown to the model in this run."""

    jev_skill_done: NotRequired[Annotated[bool, PrivateStateAttr]]
    """Whether skill suggestion already ran in this run."""

    jev_evidence: NotRequired[Annotated[list[dict], PrivateStateAttr, _extend]]
    """Passages the RAG blueprint accepted in this run."""

    jev_escalate: NotRequired[Annotated[bool, PrivateStateAttr]]
    """Set after a failed verification so the router picks the stronger model."""

    jev_attempts: NotRequired[Annotated[int, PrivateStateAttr]]
    """How many times verification sent the answer back in this run."""
