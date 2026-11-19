"""Configuration for the whole Jev stack: which model, which blueprints, how strict."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from langchain_jev._types import (
    BLUEPRINTS,
    CHAT_MODE,
    CITATION_CHECK,
    DEPENDENCIES,
    ESCALATION,
    FAST_PATH,
    INPUT_GUARD,
    MODEL_ROUTER,
    MODES,
    OUTPUT_GUARD,
    RAG_PASSAGES,
    RUN_REVIEW,
    TOOL_GATE,
    VERIFY_ANSWER,
    JevEvent,
    Mode,
)
from langchain_jev.question_book import QuestionBook
from langchain_jev.questions import (
    INPUT_BATTERY,
    MODEL_ROUTE_INSTRUCTIONS,
    OUTPUT_BATTERY,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from deepagents.backends.protocol import BackendProtocol
    from langchain_core.decisions import BaseDecisionModel, Score
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AnyMessage, ToolMessage

    from langchain_jev.fast_routes import FastRoute
    from langchain_jev.memory_store import JevMemoryStore
    from langchain_jev.models import JevModel
    from langchain_jev.question_book import QuestionOverride
    from langchain_jev.review import RunReview
    from langchain_jev.switches import ModeSource

    ModelSpec = str | JevModel | BaseDecisionModel

DEFAULT_REFUSAL = "I can't help with that request."
DEFAULT_SUPPORT = (
    "It sounds like you may be going through something difficult. You don't have to "
    "face it alone: please reach out to someone you trust, or contact a local crisis "
    "line or emergency services right away if you are in danger."
)
DEFAULT_REVIEW_PROMPT = (
    "A safety check flagged this request for possible {hazard}. Answer conservatively: "
    "give general information only and do not make personal decisions for the user."
)

OPT_IN: frozenset[str] = frozenset({CHAT_MODE})
"""Blueprints that stay off unless `JevConfig.blueprints` turns them on."""


@dataclass(frozen=True)
class GuardSettings:
    """Input, tool-result, and output guardrails."""

    policy: Literal["strict", "permissive"] = "strict"
    """Named thresholds from the guardrails cookbook."""

    input_hazards: tuple[str, ...] = tuple(INPUT_BATTERY)
    """Hazards asked about each user message."""

    output_hazards: tuple[str, ...] = tuple(OUTPUT_BATTERY)
    """Hazards asked about each final reply."""

    refusal_message: str = DEFAULT_REFUSAL
    support_message: str = DEFAULT_SUPPORT

    review_prompt: str = DEFAULT_REVIEW_PROMPT
    """Added to the system prompt when an input is sent to review; `{hazard}` is filled."""

    tool_result_action: Literal["redact", "warn", "block"] = "redact"
    """What to do with a tool result that tries to instruct the agent."""


@dataclass(frozen=True)
class RouteChoice:
    """One model the router may pick, and when to pick it."""

    model: str | BaseChatModel
    criteria: str


@dataclass(frozen=True)
class RouterSettings:
    """Model routing; required for `model_router` and `escalation`."""

    choices: Mapping[str, RouteChoice]
    instructions: str = MODEL_ROUTE_INSTRUCTIONS
    fallback: str | None = None
    """Choice used when the routing confidence is below `model_router.min_confidence`."""

    escalate_to: str | None = None
    """Choice used after an answer fails verification (the `escalation` blueprint)."""


@dataclass(frozen=True)
class FastPathSettings:
    """Requests Jev handles without calling the agent's model."""

    routes: Sequence[FastRoute]


@dataclass(frozen=True)
class SkillSettings:
    """Skill suggestion, pinning, and the check on skill loads."""

    shortlist: int = 3
    excerpt_chars: int = 700
    placement: Literal["message", "system"] = "message"
    """Where the hint goes; a trailing message keeps the cached prompt prefix intact."""

    announce_none: bool = True
    """Tell the model when no skill fits, which cut needless loads in the cookbook."""

    load_check_action: Literal["warn", "block"] = "warn"
    backend: BackendProtocol | None = None
    """Backend to read `SKILL.md` excerpts from; bound automatically when available."""


@dataclass(frozen=True)
class RagSettings:
    """Passage classification for retrieval tools; required for `rag_passages`."""

    tools: Sequence[str]
    query_arg: str = "query"
    parse: Callable[[ToolMessage], list[dict[str, object]] | None] | None = None
    """Turn a tool result into passages; the default reads JSON lists of objects."""


@dataclass(frozen=True)
class MemorySettings:
    """The remember gate and saved-fact injection."""

    store: JevMemoryStore | None = None
    """Where facts live; defaults to the graph store, or memory in this process."""

    user_key: str = "user_id"
    """Runtime-context key naming the user; the thread id is used without it."""

    max_items: int = 20
    filter_relevant: bool = False
    """Ask Jev which saved facts matter for this request instead of sending all."""


@dataclass(frozen=True)
class ToolGateSettings:
    """Checks on tool calls before they run."""

    tools: Sequence[str] = ("execute", "write_file", "edit_file", "task")
    ask_human: bool = True
    """Pause for approval instead of refusing; needs a checkpointer."""

    policy: str | None = None
    """The agent's entitlements and rules; enables the `within_grant` check."""


@dataclass(frozen=True)
class ScoreSpec:
    """One scored dimension of an answer and its weight in the composite."""

    question: Score
    weight: float = 1.0


@dataclass(frozen=True)
class VerifySettings:
    """Answer verification, citation checks, scoring, and what to do on failure."""

    evidence_tools: Sequence[str] = ()
    """Tools whose raw results count as evidence when `rag_passages` is not used."""

    sources: Callable[[str], str | None] | None = None
    """Look up a cited source by id when it is not among this run's passages."""

    scores: Mapping[str, ScoreSpec] | None = None
    retry_on: frozenset[str] = frozenset({VERIFY_ANSWER, CITATION_CHECK})
    """Checks whose failure sends the answer back to the model."""

    max_retries: int = 1
    exhausted: Literal["accept", "annotate"] = "accept"
    add_citation_instructions: bool = False


@dataclass(frozen=True)
class ReviewSettings:
    """Hand finished runs to the agent-trace review."""

    on_review: Callable[[RunReview], None] | None = None
    """Called with the review when `inline` is true."""

    on_run_end: Callable[[list[AnyMessage]], None] | None = None
    """Called with the transcript, for reviewing later off the request path."""

    irreversible_tools: Sequence[str] = ("execute", "write_file", "edit_file")
    agent_instructions: str | None = None
    inline: bool = False


@dataclass(frozen=True)
class JevConfig:
    """Everything the Jev stack needs; every field has a working default.

    Example:
        ```python
        from langchain_jev import JevConfig, FileModeSource

        config = JevConfig(
            model="h2o-lightning-4b",  # or "typesafe", or JEV_MODEL=...
            blueprints={"answer_scoring": "shadow", "fast_path": "off"},
            mode_source=FileModeSource("jev_modes.json"),
        )
        ```
    """

    model: ModelSpec = "default"
    """Registry name, `JevModel`, or decision model. `default` reads `JEV_MODEL`."""

    models: Mapping[str, ModelSpec] = field(default_factory=dict)
    """Per-blueprint model overrides, for example a larger model for guards."""

    shadow_model: ModelSpec | None = None
    """A candidate model asked alongside the main one; disagreements are reported."""

    blueprints: Mapping[str, Mode] = field(default_factory=dict)
    """Build-time modes. `off` leaves a blueprint out of the stack entirely."""

    default_mode: Mode = "on"
    mode_source: ModeSource | None = None
    """Read at the start of each run, so modes change without a restart."""

    context_key: str = "jev_modes"
    thresholds: Mapping[str, float] = field(default_factory=dict)
    questions: Mapping[str, QuestionOverride] = field(default_factory=dict)
    """New wording for any scenario's question, by key such as `input_guard.jailbreak`.

    A string replaces the instructions and keeps the yes/no or option descriptions; a
    `Noul`, `Choice`, or `Score` replaces the whole question. `default_questions()`
    lists every key with its default wording. The thresholds were tuned on the default
    wording, so check them on your own data after rewording a question.
    """

    fail_closed: frozenset[str] = frozenset({INPUT_GUARD, OUTPUT_GUARD, TOOL_GATE})
    """Blueprints that refuse, rather than allow, when Jev cannot be reached."""

    on_event: Callable[[JevEvent], None] | None = None
    cache_size: int = 512
    guard: GuardSettings = field(default_factory=GuardSettings)
    router: RouterSettings | None = None
    fast_path: FastPathSettings | None = None
    skills: SkillSettings = field(default_factory=SkillSettings)
    rag: RagSettings | None = None
    memory: MemorySettings = field(default_factory=MemorySettings)
    tool_gate: ToolGateSettings = field(default_factory=ToolGateSettings)
    verify: VerifySettings = field(default_factory=VerifySettings)
    review: ReviewSettings = field(default_factory=ReviewSettings)

    def installed(self) -> dict[str, Mode]:
        """Blueprints that will be built, with their build-time modes.

        Raises:
            ValueError: If the configuration names unknown blueprints or modes, turns
                on a blueprint without the settings it needs, or breaks a dependency.
        """
        self._check_names()
        modes: dict[str, Mode] = {}
        for blueprint in BLUEPRINTS:
            mode = self._build_mode(blueprint)
            if mode != "off":
                modes[blueprint] = mode
        return self._drop_orphans(modes)

    def _drop_orphans(self, modes: dict[str, Mode]) -> dict[str, Mode]:
        """Drop defaulted blueprints whose dependency is off; reject explicit ones."""
        for blueprint, required in DEPENDENCIES.items():
            missing = [name for name in required if name not in modes]
            if blueprint not in modes or not missing:
                continue
            if blueprint in self.blueprints:
                msg = f"Jev blueprint {blueprint!r} needs {', '.join(missing)} installed."
                raise ValueError(msg)
            del modes[blueprint]
        return modes

    def _build_mode(self, blueprint: str) -> Mode:
        explicit = self.blueprints.get(blueprint)
        default: Mode = "off" if blueprint in OPT_IN else self.default_mode
        mode = explicit if explicit is not None else default
        missing = self._missing_setting(blueprint)
        if missing is None or mode == "off":
            return mode
        if explicit is not None:
            msg = f"Jev blueprint {blueprint!r} needs {missing}."
            raise ValueError(msg)
        return "off"

    def _missing_setting(self, blueprint: str) -> str | None:
        needs: dict[str, tuple[bool, str]] = {
            MODEL_ROUTER: (self.router is None, "`router` settings"),
            ESCALATION: (
                self.router is None or self.router.escalate_to is None,
                "`router.escalate_to`",
            ),
            FAST_PATH: (self.fast_path is None, "`fast_path` routes"),
            RAG_PASSAGES: (self.rag is None, "`rag` settings with retrieval tools"),
            RUN_REVIEW: (
                self.review.on_review is None and self.review.on_run_end is None,
                "`review.on_review` or `review.on_run_end`",
            ),
        }
        absent, what = needs.get(blueprint, (False, ""))
        return what if absent else None

    def _check_names(self) -> None:
        unknown = sorted(set(self.blueprints) - set(BLUEPRINTS))
        if unknown:
            msg = f"Unknown Jev blueprints: {', '.join(unknown)}"
            raise ValueError(msg)
        invalid = sorted(k for k, v in self.blueprints.items() if v not in MODES)
        if invalid or self.default_mode not in MODES:
            msg = f"Jev modes must be one of {sorted(MODES)}; check {invalid}."
            raise ValueError(msg)
        unknown_models = sorted(set(self.models) - set(BLUEPRINTS))
        if unknown_models:
            msg = f"`models` names unknown blueprints: {', '.join(unknown_models)}"
            raise ValueError(msg)
        self._check_router()
        self._check_questions()

    def _check_questions(self) -> None:
        QuestionBook(self.questions)
        custom_route = (
            self.router is not None and self.router.instructions != MODEL_ROUTE_INSTRUCTIONS
        )
        if custom_route and "model_router.route" in self.questions:
            msg = (
                "Set the routing question in one place: `questions['model_router.route']` "
                "or `RouterSettings.instructions`."
            )
            raise ValueError(msg)
        if self.verify.scores is not None and any(
            key.startswith("answer_scoring.") for key in self.questions
        ):
            msg = (
                "`answer_scoring.*` questions reword the default scores; with "
                "`VerifySettings.scores`, put the wording in each `ScoreSpec`."
            )
            raise ValueError(msg)

    def _check_router(self) -> None:
        if self.router is None:
            return
        names = set(self.router.choices)
        for label, choice in (
            ("fallback", self.router.fallback),
            ("escalate_to", self.router.escalate_to),
        ):
            if choice is not None and choice not in names:
                msg = f"`router.{label}` must be one of {sorted(names)}, got {choice!r}."
                raise ValueError(msg)
