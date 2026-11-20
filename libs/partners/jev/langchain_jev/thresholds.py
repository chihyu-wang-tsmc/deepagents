"""Every threshold the blueprints compare Jev's answers against.

Values come from the Typesafe page each blueprint follows. They were measured for
the model each page names; after switching to another Jev model, re-tune them with
`JevModel.thresholds` (per model) or `JevConfig.thresholds` (per agent).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

GUARD_POLICIES: dict[str, dict[str, float]] = {
    "strict": {"review": 0.35, "action": 0.70, "severity_block": 2.0},
    "permissive": {"review": 0.70, "action": 0.85, "severity_block": 2.0},
}
"""Named guardrail policies from https://docs.typesafe.ai/cookbooks/llm_guardrails."""

DEFAULT_THRESHOLDS: dict[str, float] = {
    # https://docs.typesafe.ai/cookbooks/llm_guardrails (strict policy)
    "input_guard.review": 0.35,
    "input_guard.action": 0.70,
    "input_guard.severity_block": 2.0,
    "output_guard.review": 0.35,
    "output_guard.action": 0.70,
    "output_guard.severity_block": 2.0,
    # https://docs.typesafe.ai/cookbooks/classifying_rag_passages (injection_max)
    "tool_result_guard.injection": 0.70,
    # https://docs.typesafe.ai/patterns/confidence-routing: 0 keeps the top choice
    "model_router.min_confidence": 0.0,
    # open-source-jev `coding_report.py`: P(coding) >= 0.5 counts as coding in production
    "coding_router.min_coding": 0.50,
    # https://docs.typesafe.ai/patterns/probability-routing: a costly action needs more
    "chat_mode.min_probability": 0.80,
    # https://docs.typesafe.ai/patterns/companion-noul
    "fast_path.min_fits": 0.70,
    "fast_path.min_confidence": 0.70,
    # https://docs.typesafe.ai/cookbooks/skill_suggestion
    "skill_suggestion.action": 0.30,
    "skill_suggestion.named": 0.50,
    "skill_suggestion.fits": 0.30,
    "skill_pinning.min_fits": 0.80,
    "skill_load_check.min_fits": 0.30,
    # https://docs.typesafe.ai/cookbooks/classifying_rag_passages
    "rag_passages.injection_max": 0.70,
    "rag_passages.contradicts_min": 0.70,
    "rag_passages.relevant_min": 0.45,
    "rag_passages.evidence_min": 0.55,
    # https://docs.typesafe.ai/cookbooks/basic_memory
    "memory_write.remember_at": 0.50,
    "memory_read.relevant_min": 0.50,
    # LangChain `AutoModeMiddleware` and the agent-trace safety checks use 0.5
    "tool_gate.risky": 0.50,
    "tool_gate.instructed_by_tool_output": 0.50,
    "tool_gate.user_asked_or_agreed": 0.50,
    "tool_gate.within_grant": 0.50,
    "tool_gate.leaks_data": 0.50,
    "tool_gate.irreversible": 0.50,
    # https://docs.typesafe.ai/cookbooks/verify_llm_outputs
    "verify_answer.review_at": 0.20,
    # https://docs.typesafe.ai/cookbooks/citation_check
    "citation_check.auto_accept": 0.80,
    # https://docs.typesafe.ai/patterns/composite-scoring (normalized 0-1 composite)
    "answer_scoring.min_score": 0.50,
}

NOUL_THRESHOLDS: frozenset[str] = frozenset(
    {
        "input_guard.review",
        "input_guard.action",
        "output_guard.review",
        "output_guard.action",
        "tool_result_guard.injection",
        "coding_router.min_coding",
        "fast_path.min_fits",
        "skill_suggestion.action",
        "skill_suggestion.named",
        "skill_suggestion.fits",
        "skill_pinning.min_fits",
        "skill_load_check.min_fits",
        "rag_passages.injection_max",
        "rag_passages.contradicts_min",
        "rag_passages.relevant_min",
        "rag_passages.evidence_min",
        "memory_write.remember_at",
        "memory_read.relevant_min",
        "tool_gate.risky",
        "tool_gate.instructed_by_tool_output",
        "tool_gate.user_asked_or_agreed",
        "tool_gate.within_grant",
        "tool_gate.leaks_data",
        "tool_gate.irreversible",
        "verify_answer.review_at",
    }
)
"""Thresholds compared with Noul probabilities, which a clamping server distorts."""


def merge_thresholds(*layers: Mapping[str, float]) -> dict[str, float]:
    """Merge threshold layers, later layers winning.

    Raises:
        ValueError: If a layer names a threshold that does not exist.
    """
    merged = dict(DEFAULT_THRESHOLDS)
    for layer in layers:
        unknown = sorted(set(layer) - set(DEFAULT_THRESHOLDS))
        if unknown:
            msg = f"Unknown Jev thresholds: {', '.join(unknown)}"
            raise ValueError(msg)
        merged.update(layer)
    return merged


def clamped_thresholds(thresholds: Mapping[str, float], noul_floor: float) -> list[str]:
    """Names of Noul thresholds a server clamping at `noul_floor` turns into 0.5.

    A server that pushes every Noul probability below `noul_floor` (and above
    `1 - noul_floor`) to the boundary leaves only which side of 0.5 it fell on, so
    any threshold strictly between the two boundaries behaves like 0.5.
    """
    low, high = 1 - noul_floor, noul_floor
    return sorted(
        name
        for name in NOUL_THRESHOLDS
        if low < thresholds[name] < high and thresholds[name] != 0.5  # noqa: PLR2004
    )
