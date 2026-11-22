"""Blueprints of your own: what to ask Jev, where in the agent loop, and what to do.

A `JevBlueprint` holds its questions and its action together. The stack asks the
questions at the blueprint's stage and hands the answers back under the same ids, so
what is asked and what is acted on always match. A custom blueprint is switched,
reworded, re-thresholded, and reported like a built-in one, under its own name.

Each stage asks Jev about the same text as the built-in blueprints placed there:

- `entry` (`before_agent`, inside the one entry request): the user's message and the
    turns before it, like `input_guard`, the routers, and `chat_mode`.
- `model_output` (`after_model`): the text of each model reply, like `output_guard`.
- `tool_call` (`wrap_tool_call`, before the tool runs): the call and the recent
    messages, like `tool_gate`.
- `tool_result` (`wrap_tool_call`, after the tool returns): the tool's result, like
    `tool_result_guard`.
- `final_answer` (`after_agent`): the request and the final answer, like the answer
    checks.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

from langchain_core.decisions import Choice, Noul, Score

if TYPE_CHECKING:
    from langchain_core.decisions.types import Answer
    from langchain_core.language_models import BaseChatModel

Stage = Literal["entry", "model_output", "tool_call", "tool_result", "final_answer"]
"""Where in the agent loop a custom blueprint asks Jev."""

STAGES: tuple[Stage, ...] = ("entry", "model_output", "tool_call", "tool_result", "final_answer")
TOOL_STAGES: frozenset[str] = frozenset({"tool_call", "tool_result"})

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
"""Blueprint names, question ids, and threshold names: they become `name.id` keys."""


@dataclass(frozen=True)
class Refuse:
    """Stop what Jev flagged and say why.

    - `entry`: reply with `message` instead of running the agent.
    - `model_output`: replace the model's reply with `message` and end the run.
    - `tool_call`: skip the tool; the model reads `message` as the call's result.
    - `tool_result`: withhold the result and end the run with `message`.
    """

    message: str | None = None
    """`None` uses `GuardSettings.refusal_message`, or at `tool_call` a note for the model."""


@dataclass(frozen=True)
class SystemNote:
    """Add `text` to the system prompt of every model call in the run (`entry`)."""

    text: str


@dataclass(frozen=True)
class UseModel:
    """Call `model` instead of the agent's model for the run (`entry`)."""

    model: str | BaseChatModel
    """A `provider:model` string for `init_chat_model`, or a chat model."""


@dataclass(frozen=True)
class NoTools:
    """Offer the model no tools for the run, so it answers in text (`entry`)."""


@dataclass(frozen=True)
class AskHuman:
    """Pause the tool call until a person approves it (`tool_call`; needs a checkpointer)."""

    reason: str | None = None
    """Shown to the reviewer; `None` says which blueprint flagged the call."""


@dataclass(frozen=True)
class Warn:
    """Keep the tool result and put `note` in front of it (`tool_result`)."""

    note: str | None = None


@dataclass(frozen=True)
class Redact:
    """Replace the tool result with `note` (`tool_result`)."""

    note: str | None = None


@dataclass(frozen=True)
class Mark:
    """Record `reason` in the final answer's `additional_kwargs["jev_checks"]` (`final_answer`).

    The answer keeps its text; `VerifySettings(annotate=True)` also adds a visible note.
    """

    reason: str


Action: TypeAlias = Refuse | SystemNote | UseModel | NoTools | AskHuman | Warn | Redact | Mark
"""What a custom blueprint does when its questions call for it."""

Decide: TypeAlias = Callable[["Mapping[str, Answer]", "Mapping[str, float]"], "Action | None"]
"""Answers by question id and thresholds by name in, an action or `None` (pass) out."""

ACTIONS: tuple[type, ...] = (Refuse, SystemNote, UseModel, NoTools, AskHuman, Warn, Redact, Mark)

ALLOWED: dict[str, tuple[type, ...]] = {
    "entry": (Refuse, SystemNote, UseModel, NoTools),
    "model_output": (Refuse,),
    "tool_call": (Refuse, AskHuman),
    "tool_result": (Warn, Redact, Refuse),
    "final_answer": (Mark,),
}
"""The actions each stage can take."""

_ACTION_NAMES: dict[type, str] = {
    Refuse: "refuse",
    SystemNote: "note",
    UseModel: "use_model",
    NoTools: "no_tools",
    AskHuman: "ask_human",
    Warn: "warn",
    Redact: "redact",
    Mark: "mark",
}


def action_name(action: Action | None) -> str:
    """The `JevEvent.action` reported for `action`: `pass` when there is none."""
    return "pass" if action is None else _ACTION_NAMES[type(action)]


@dataclass(frozen=True)
class JevBlueprint:
    """A blueprint of your own: its questions, where they are asked, and what to do.

    Give `when` and `act` for the common case, where any yes/no question at or above
    its threshold triggers one action. Give `decide` instead to read Choice or Score
    answers, or to combine several questions. Answers are `NoulAnswer` (`.noul`, the
    probability of yes), `ChoiceAnswer` (`.choice`, `.probabilities`), or `ScoreAnswer`
    (`.score`). `decide` may be called more than once a run, so keep it free of side
    effects.

    Every name a built-in blueprint takes works for this one too: turn it on in
    `JevConfig.blueprints`, reword `<name>.<question>` in `JevConfig.questions`, set
    `<name>.<threshold>` in `JevConfig.thresholds`, give it a model in `JevConfig.models`,
    and list it in `JevConfig.fail_closed`.

    Example:
        ```python
        from langchain_jev import JevBlueprint, JevConfig, Refuse, noul

        confidential = JevBlueprint(
            name="confidential_request",
            stage="entry",
            questions={
                "asks_confidential": noul(
                    "Does this message ask for confidential company data?",
                    yes="It asks for customer lists, internal prices, or similar.",
                    no="It asks for nothing confidential.",
                )
            },
            when={"asks_confidential": 0.7},
            act=Refuse("I can't share confidential company data."),
        )
        config = JevConfig(
            blueprints={"confidential_request": "on"},
            custom_blueprints=[confidential],
        )
        ```
    """

    name: str
    """Letters, digits, and underscores, starting with a letter; not a built-in's name."""

    stage: Stage
    """Where Jev is asked; see the module docstring for what it reads there."""

    questions: Mapping[str, Noul | Choice | Score]
    """The questions to ask, by id."""

    when: Mapping[str, float] = field(default_factory=dict)
    """Yes/no question ids and the probability at which each triggers `act`."""

    act: Action | None = None
    """The action `when` triggers."""

    decide: Decide | None = None
    """Choose the action yourself, instead of `when` and `act`."""

    thresholds: Mapping[str, float] = field(default_factory=dict)
    """Named values passed to `decide`, so `JevConfig.thresholds` can tune them."""

    tools: Sequence[str] = ()
    """The tools a `tool_call` or `tool_result` blueprint checks; required there."""

    def __post_init__(self) -> None:
        """Check the blueprint when it is defined.

        Raises:
            ValueError: If a name, stage, question, threshold, or tool list is missing,
                unknown, or out of range, or `when`/`act` and `decide` are mixed.
            TypeError: If a question, threshold, or action is of the wrong kind, or an
                action does not apply at the stage.
        """
        _check_name(self.name, "Jev blueprint names")
        if self.stage not in STAGES:
            msg = f"Jev blueprint {self.name!r} has unknown stage {self.stage!r}; use {STAGES}."
            raise ValueError(msg)
        self._check_questions()
        if self.decide is None:
            self._check_rule()
        else:
            self._check_decide()
        self._check_tools()

    def _check_questions(self) -> None:
        if not self.questions:
            msg = f"Jev blueprint {self.name!r} needs at least one question."
            raise ValueError(msg)
        for qid, question in self.questions.items():
            _check_name(qid, f"Question ids of Jev blueprint {self.name!r}")
            if not isinstance(question, (Noul, Choice, Score)):
                msg = (
                    f"Question {qid!r} of Jev blueprint {self.name!r} must be a Noul, "
                    f"Choice, or Score, got {type(question).__name__}."
                )
                raise TypeError(msg)

    def _check_decide(self) -> None:
        if self.when or self.act is not None:
            msg = f"Jev blueprint {self.name!r}: give either `when` and `act`, or `decide`."
            raise ValueError(msg)
        for key, value in self.thresholds.items():
            _check_name(key, f"Thresholds of Jev blueprint {self.name!r}")
            _check_number(self.name, key, value)

    def _check_rule(self) -> None:
        if self.act is None or not self.when:
            msg = (
                f"Jev blueprint {self.name!r} needs `when` and `act` (the questions that "
                "trigger one action), or `decide`."
            )
            raise ValueError(msg)
        if self.thresholds:
            msg = (
                f"Jev blueprint {self.name!r}: `thresholds` are for `decide`; with `when`, "
                "the values in `when` are the thresholds."
            )
            raise ValueError(msg)
        for qid, value in self.when.items():
            self._check_trigger(qid, value)
        self.check(self.act, source="`act`")

    def _check_trigger(self, qid: str, value: object) -> None:
        question = self.questions.get(qid)
        if question is None:
            msg = f"`when` of Jev blueprint {self.name!r} names {qid!r}, which is not a question."
            raise ValueError(msg)
        if not isinstance(question, Noul):
            msg = (
                f"`when` of Jev blueprint {self.name!r} names {qid!r}, a {question.type} "
                "question; `when` reads yes/no (Noul) answers, so use `decide` for it."
            )
            raise TypeError(msg)
        _check_number(self.name, qid, value)
        if not 0 <= cast("float", value) <= 1:
            msg = (
                f"`when` of Jev blueprint {self.name!r} gives {qid!r} {value!r}; a "
                "probability threshold is between 0 and 1."
            )
            raise ValueError(msg)

    def _check_tools(self) -> None:
        tools = (self.tools,) if isinstance(self.tools, str) else tuple(self.tools)
        object.__setattr__(self, "tools", tools)
        if self.stage in TOOL_STAGES and not tools:
            msg = f"Jev blueprint {self.name!r} at stage {self.stage!r} needs `tools` to check."
            raise ValueError(msg)
        if self.stage not in TOOL_STAGES and tools:
            msg = (
                f"Jev blueprint {self.name!r}: `tools` only apply at the tool_call and "
                "tool_result stages."
            )
            raise ValueError(msg)

    def check(self, action: object, *, source: str = "`decide`") -> Action | None:
        """`action` if it is `None` or an action this stage can take.

        Raises:
            TypeError: If `action` is not an action, or does not apply at this stage.
        """
        if action is None:
            return None
        allowed = ", ".join(t.__name__ for t in ALLOWED[self.stage])
        if not isinstance(action, ALLOWED[self.stage]):
            known = isinstance(action, ACTIONS)
            got = type(action).__name__ if known else repr(action)
            where = f", which does not apply at stage {self.stage!r}" if known else ""
            msg = (
                f"{source} of Jev blueprint {self.name!r} gave {got}{where}; return one of "
                f"{allowed}, or None to let it pass."
            )
            raise TypeError(msg)
        return cast("Action", action)

    def threshold_defaults(self) -> dict[str, float]:
        """This blueprint's thresholds by full name: `<name>.<key>`."""
        values = self.when or self.thresholds
        return {f"{self.name}.{key}": float(value) for key, value in values.items()}

    def noul_thresholds(self) -> frozenset[str]:
        """The full names of thresholds compared with yes/no probabilities."""
        return frozenset(f"{self.name}.{qid}" for qid in self.when)

    def default_questions(self) -> dict[str, Noul | Choice | Score]:
        """This blueprint's questions by full key: `<name>.<question id>`."""
        return {f"{self.name}.{qid}": q for qid, q in self.questions.items()}


def _check_name(value: object, what: str) -> None:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        msg = f"{what} use letters, digits, and underscores and start with a letter; got {value!r}."
        raise ValueError(msg)


def _check_number(blueprint: str, key: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"Threshold {key!r} of Jev blueprint {blueprint!r} must be a number, got {value!r}."
        raise TypeError(msg)
