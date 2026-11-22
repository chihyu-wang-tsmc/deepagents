"""Which reworded questions a configured stack will never ask, and what would change that.

Only reasons known when the stack is built count: a blueprint that is off, a hazard
left out of `GuardSettings`, a check missing the text it compares against, or a
review that only hands runs on. Questions that wait for something during a run, such
as a citation to check, are not reported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_jev._types import (
    INPUT_GUARD,
    OUTPUT_GUARD,
    RUN_REVIEW,
    TOOL_GATE,
)
from langchain_jev.questions import INPUT_BATTERY, OUTPUT_BATTERY

if TYPE_CHECKING:
    from collections.abc import Mapping

    from langchain_jev._types import Mode
    from langchain_jev.config import JevConfig


def unasked_questions(config: JevConfig, installed: Mapping[str, Mode]) -> dict[str, str]:
    """Reworded questions the stack never asks, each with the setting that would.

    Keys outside every blueprint, such as `compare_answers.better`, are left out:
    the standalone functions read them only when they are passed in.
    """
    reasons = {key: _reason(key, config, installed) for key in sorted(config.questions)}
    return {key: reason for key, reason in reasons.items() if reason}


def asked_overrides(config: JevConfig, unasked: Mapping[str, str]) -> list[str]:
    """Reworded questions the stack does ask."""
    names = set(config.blueprint_names())
    return [
        key
        for key in sorted(config.questions)
        if key not in unasked and key.partition(".")[0] in names
    ]


def _reason(key: str, config: JevConfig, installed: Mapping[str, Mode]) -> str | None:
    blueprint, _, name = key.partition(".")
    if blueprint not in config.blueprint_names():
        return None
    if blueprint not in installed:
        return f"turn on {blueprint} in JevConfig.blueprints"
    if blueprint == RUN_REVIEW and not config.review.reviews_inline:
        return "set ReviewSettings.on_review so runs are reviewed here"
    return _hazard_reason(blueprint, name, config) or _missing_text(blueprint, name, config)


def _hazard_reason(blueprint: str, name: str, config: JevConfig) -> str | None:
    guards = {
        INPUT_GUARD: (INPUT_BATTERY, config.guard.input_hazards, "input_hazards"),
        OUTPUT_GUARD: (OUTPUT_BATTERY, config.guard.output_hazards, "output_hazards"),
    }
    if blueprint not in guards:
        return None
    battery, chosen, field = guards[blueprint]
    if name in battery and name not in chosen:
        return f"add {name!r} to GuardSettings.{field}"
    return None


def _missing_text(blueprint: str, name: str, config: JevConfig) -> str | None:
    needs = {
        (TOOL_GATE, "within_grant"): (config.tool_gate.policy, "set ToolGateSettings.policy"),
        (RUN_REVIEW, "within_grant"): (
            config.review.agent_instructions,
            "set ReviewSettings.agent_instructions",
        ),
    }
    present, fix = needs.get((blueprint, name), (True, ""))
    return None if present else fix
