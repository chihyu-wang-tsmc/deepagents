"""Every scenario's question, with the wording you configure in place of the default.

Each scenario has a key, `<blueprint>.<name>`, for example `input_guard.jailbreak` or
`tool_gate.risky`. Pass new wording through `JevConfig(questions={key: ...})`:

- A string replaces the question's instructions and keeps its yes/no or option
    descriptions.
- A `Noul`, `Choice`, or `Score` replaces the whole question. It must keep the
    default's type, a `Choice` must keep the same option names, and a `Score` must
    keep the same number of levels, because the blueprints read answers by them.

Three kinds of scenario take different overrides:

- Most are complete questions and take either form.
- Templates (`skill_suggestion.fits`, `skill_load_check.fits`,
    `fast_path.functions_fit`) are filled per call; their instructions may use the
    placeholders listed in `TEMPLATE_FIELDS`. Write a literal brace as `{{` or `}}`.
- Instructions-only scenarios (`model_router.route`, `fast_path.which`,
    `skill_suggestion.which`, `skill_suggestion.rerank`, `run_review.first_bad_step`)
    are choices whose options come from your configuration or the run, so they only
    take a string.

`default_questions()` lists every key with its default wording.
"""

from __future__ import annotations

import string
from functools import cache
from typing import TYPE_CHECKING, TypeAlias

from langchain_core.decisions import Choice, Noul, Score

from langchain_jev import questions as q

if TYPE_CHECKING:
    from collections.abc import Mapping

FullQuestion: TypeAlias = Noul | Choice | Score
QuestionOverride: TypeAlias = str | Noul | Choice | Score
"""New wording for one scenario: a string for the instructions, or a whole question."""


def _group(blueprint: str, questions: Mapping[str, FullQuestion]) -> dict[str, FullQuestion]:
    return {f"{blueprint}.{name}": question for name, question in questions.items()}


QUESTIONS: dict[str, FullQuestion] = {
    **_group("input_guard", q.INPUT_BATTERY),
    "input_guard.severity": q.SEVERITY,
    **_group("output_guard", q.OUTPUT_BATTERY),
    "output_guard.severity": q.SEVERITY,
    **_group("tool_result_guard", q.TOOL_RESULT_QUESTIONS),
    "chat_mode.mode": q.MODE,
    "fast_path.functions_fit": q.FUNCTIONS_FIT,
    **_group("skill_suggestion", q.SKILL_ACTION_QUESTIONS),
    "skill_suggestion.named": q.SKILL_NAMED_QUESTION,
    "skill_suggestion.fits": q.SKILL_FITS,
    "skill_load_check.fits": q.SKILL_FITS,
    **_group("rag_passages", q.PASSAGE_QUESTIONS),
    "memory_read.relevant": q.MEMORY_RELEVANT,
    "memory_write.remember": q.REMEMBER,
    **_group("tool_gate", q.TOOL_GATE_QUESTIONS),
    **_group("verify_answer", q.CHECK_QUESTIONS),
    "citation_check.relation": q.CITATION_RELATION,
    **_group("answer_scoring", q.DEFAULT_SCORES),
    **_group("run_review", q.REVIEW_SAFETY_QUESTIONS),
    **_group("run_review", q.OUTCOME_QUESTIONS),
    **_group("run_review", q.REACTION_QUESTIONS),
    **_group("run_review", q.CLARITY_QUESTIONS),
    "run_review.attribution": q.ATTRIBUTION,
    "compare_answers.better": q.ANSWER_COMPARISON,
}
"""Scenarios asked as complete questions."""

INSTRUCTIONS: dict[str, str] = {
    "model_router.route": q.MODEL_ROUTE_INSTRUCTIONS,
    "fast_path.which": q.FUNCTION_ROUTE_INSTRUCTIONS,
    "skill_suggestion.which": q.SKILL_CHOICE_INSTRUCTIONS,
    "skill_suggestion.rerank": q.SKILL_RERANK_INSTRUCTIONS,
    "run_review.first_bad_step": q.FIRST_BAD_STEP,
}
"""Scenarios whose options are only known at run time: only the instructions change."""

TEMPLATE_FIELDS: dict[str, tuple[str, ...]] = {
    "fast_path.functions_fit": ("functions",),
    "skill_suggestion.fits": ("name", "description"),
    "skill_load_check.fits": ("name", "description"),
}
"""Placeholders each template's instructions may use."""


def default_questions() -> dict[str, FullQuestion | str]:
    """Every scenario key with its default wording, to see what can be reworded.

    Returns:
        Questions for complete-question scenarios and plain instructions for
            instructions-only ones. The values are copies, so changing them does not
            change the defaults.
    """
    full: dict[str, FullQuestion | str] = {
        key: question.model_copy(deep=True) for key, question in QUESTIONS.items()
    }
    return {**full, **INSTRUCTIONS}


class QuestionBook:
    """The questions one Jev stack asks: the defaults with your overrides applied."""

    def __init__(
        self,
        overrides: Mapping[str, QuestionOverride] | None = None,
        *,
        route_instructions: str | None = None,
    ) -> None:
        """Check `overrides` and apply them.

        Args:
            overrides: New wording by scenario key.
            route_instructions: The routing question from `RouterSettings`, used
                unless `overrides` rewords `model_router.route`.

        Raises:
            ValueError: If a key is unknown, or an override changes a question's type,
                option names, or number of levels, or uses an unknown placeholder.
        """
        given = dict(overrides or {})
        _check_keys(given)
        self.overridden: tuple[str, ...] = tuple(sorted(given))
        self._questions = {key: _reword(key, d, given.get(key)) for key, d in QUESTIONS.items()}
        self._instructions = {key: _text(key, d, given.get(key)) for key, d in INSTRUCTIONS.items()}
        if route_instructions is not None and "model_router.route" not in given:
            self._instructions["model_router.route"] = route_instructions

    def question(self, key: str, /, **values: str) -> FullQuestion:
        """The question for `key`; a template's placeholders are filled from `values`."""
        question = self._questions[key]
        if key not in TEMPLATE_FIELDS:
            return question
        text = str(question.instructions).format(**values)
        return question.model_copy(update={"instructions": text})

    def instructions(self, key: str) -> str:
        """The instructions of a choice whose options are only known at run time."""
        return self._instructions[key]


@cache
def default_book() -> QuestionBook:
    """The defaults, for code that runs without a `JevConfig`."""
    return QuestionBook()


def _check_keys(given: Mapping[str, QuestionOverride]) -> None:
    unknown = sorted(set(given) - set(QUESTIONS) - set(INSTRUCTIONS))
    if unknown:
        msg = (
            f"Unknown Jev question keys: {', '.join(unknown)}. "
            "`default_questions()` lists every key."
        )
        raise ValueError(msg)


def _text(key: str, default: str, override: QuestionOverride | None) -> str:
    if override is None:
        return default
    if isinstance(override, str):
        return _nonempty(key, override)
    msg = f"Jev question {key!r} takes a string: its options are built at run time."
    raise ValueError(msg)


def _reword(key: str, default: FullQuestion, override: QuestionOverride | None) -> FullQuestion:
    if override is None:
        return default
    if isinstance(override, str):
        override = default.model_copy(update={"instructions": _nonempty(key, override)})
    _check_shape(key, default, override)
    if key in TEMPLATE_FIELDS:
        _check_fields(key, str(override.instructions), TEMPLATE_FIELDS[key])
    return override


def _nonempty(key: str, text: str) -> str:
    if not text.strip():
        msg = f"Jev question {key!r} cannot be empty."
        raise ValueError(msg)
    return text


def _check_shape(key: str, default: FullQuestion, override: object) -> None:
    if type(override) is not type(default):
        got = getattr(override, "type", type(override).__name__)
        msg = (
            f"Jev question {key!r} must stay a {default.type} question, got {got}. "
            "Pass a string to change only the instructions."
        )
        raise ValueError(msg)
    if (
        isinstance(default, Choice)
        and isinstance(override, Choice)
        and set(override.criteria) != set(default.criteria)
    ):
        msg = f"Jev question {key!r} must keep the options {sorted(default.criteria)}."
        raise ValueError(msg)
    if (
        isinstance(default, Score)
        and isinstance(override, Score)
        and len(override.criteria) != len(default.criteria)
    ):
        msg = f"Jev question {key!r} must keep its {len(default.criteria)} levels."
        raise ValueError(msg)


def _check_fields(key: str, text: str, fields: tuple[str, ...]) -> None:
    try:
        used = {name for _, name, _, _ in string.Formatter().parse(text) if name is not None}
    except ValueError as error:
        msg = f"Jev question {key!r} has an unmatched brace; write a literal brace as {{{{ or }}}}."
        raise ValueError(msg) from error
    unknown = sorted(used - set(fields))
    if unknown:
        allowed = ", ".join(f"{{{name}}}" for name in fields)
        msg = f"Jev question {key!r} may only use {allowed}; got {unknown}."
        raise ValueError(msg)
