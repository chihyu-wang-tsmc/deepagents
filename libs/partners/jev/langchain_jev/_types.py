"""Blueprint ids, modes, and the decision events every blueprint emits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable

Mode = Literal["off", "shadow", "on"]
"""How a blueprint behaves in a run.

- `off`: the blueprint does nothing and asks Jev nothing.
- `shadow`: the blueprint asks Jev and records what it would do, but changes nothing.
- `on`: the blueprint asks Jev and acts on the answer.
"""

MODES: frozenset[str] = frozenset({"off", "shadow", "on"})

INPUT_GUARD = "input_guard"
TOOL_RESULT_GUARD = "tool_result_guard"
OUTPUT_GUARD = "output_guard"
MODEL_ROUTER = "model_router"
CODING_ROUTER = "coding_router"
CHAT_MODE = "chat_mode"
SKILL_SUGGESTION = "skill_suggestion"
SKILL_PINNING = "skill_pinning"
SKILL_LOAD_CHECK = "skill_load_check"
RAG_PASSAGES = "rag_passages"
TOOL_GATE = "tool_gate"
VERIFY_ANSWER = "verify_answer"
CITATION_CHECK = "citation_check"
ANSWER_SCORING = "answer_scoring"
RUN_REVIEW = "run_review"

BLUEPRINTS: tuple[str, ...] = (
    INPUT_GUARD,
    TOOL_RESULT_GUARD,
    OUTPUT_GUARD,
    MODEL_ROUTER,
    CODING_ROUTER,
    CHAT_MODE,
    SKILL_SUGGESTION,
    SKILL_PINNING,
    SKILL_LOAD_CHECK,
    RAG_PASSAGES,
    TOOL_GATE,
    VERIFY_ANSWER,
    CITATION_CHECK,
    ANSWER_SCORING,
    RUN_REVIEW,
)
"""Every blueprint id, in the order the diagram reads them."""

DEPENDENCIES: dict[str, tuple[str, ...]] = {
    SKILL_PINNING: (SKILL_SUGGESTION,),
}
"""Blueprints that only make sense when another blueprint is installed."""

REMOVED_BLUEPRINTS: frozenset[str] = frozenset(
    {"escalation", "fast_path", "memory_read", "memory_write"}
)
"""Blueprints this package no longer has, so errors can say so instead of "unknown"."""


def removed_note(names: Iterable[str]) -> str:
    """Say which of `names` (blueprints, or keys starting with one) were removed."""
    gone = sorted({name.split(".", 1)[0] for name in names} & REMOVED_BLUEPRINTS)
    if not gone:
        return ""
    verb = "was" if len(gone) == 1 else "were"
    return f" ({', '.join(gone)} {verb} removed from langchain-jev.)"


@dataclass(frozen=True)
class JevEvent:
    """One decision a blueprint made, or would have made in `shadow` mode."""

    blueprint: str
    """The blueprint id, for example `input_guard`."""

    mode: Mode
    """The blueprint's mode in this run."""

    action: str
    """What the blueprint did, for example `block`, `pass`, `route:fast`, or `error`."""

    details: dict[str, object] = field(default_factory=dict)
    """Probabilities and other values the decision was based on."""

    @property
    def acted(self) -> bool:
        """Whether the decision changed the run (it never does in `shadow` mode)."""
        return self.mode == "on"
