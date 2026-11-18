"""Blueprint ids, modes, and the decision events every blueprint emits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

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
CHAT_MODE = "chat_mode"
ESCALATION = "escalation"
FAST_PATH = "fast_path"
SKILL_SUGGESTION = "skill_suggestion"
SKILL_PINNING = "skill_pinning"
SKILL_LOAD_CHECK = "skill_load_check"
RAG_PASSAGES = "rag_passages"
MEMORY_READ = "memory_read"
MEMORY_WRITE = "memory_write"
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
    CHAT_MODE,
    ESCALATION,
    FAST_PATH,
    SKILL_SUGGESTION,
    SKILL_PINNING,
    SKILL_LOAD_CHECK,
    RAG_PASSAGES,
    MEMORY_READ,
    MEMORY_WRITE,
    TOOL_GATE,
    VERIFY_ANSWER,
    CITATION_CHECK,
    ANSWER_SCORING,
    RUN_REVIEW,
)
"""Every blueprint id, in the order the diagram reads them."""

DEPENDENCIES: dict[str, tuple[str, ...]] = {
    SKILL_PINNING: (SKILL_SUGGESTION,),
    ESCALATION: (MODEL_ROUTER,),
}
"""Blueprints that only make sense when another blueprint is installed."""


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
