"""Blueprints of your own: what to ask Jev, what to do, and where to do it.

A `JevBlueprint` holds its questions and its actions together. The stack asks the
questions at the blueprint's stage (the front) and carries out each action at its
position (the back), giving both the answers under the ids the blueprint gave its
questions, so what is asked and what is acted on always match. A custom blueprint is
switched, reworded, re-thresholded, and reported like a built-in one, under its name.

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

Each action acts at one position, its `at`:

- `before_agent`: before the model is first called (from `entry` only).
- `model_call` (`wrap_model_call`): every model call.
- `before_model`: the start of each turn.
- `model_output` (`after_model`): each model reply.
- `tool_call` (`wrap_tool_call`, before the tool runs): calls of the listed tools.
- `tool_result` (`wrap_tool_call`, after the tool returns): results of the listed tools.
- `final_answer` (`after_agent`): the final answer.

At its stage's own position an action acts on what Jev was just asked about. At a
later position it acts from then until the run ends. A position that comes before the
stage is refused when the blueprint is defined, since Jev has not been asked yet.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

from langchain_core.decisions import Choice, Noul, Score

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ModelRequest
    from langchain_core.decisions.types import Answer
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import ToolCall

Stage = Literal["entry", "model_output", "tool_call", "tool_result", "final_answer"]
"""Where in the agent loop a custom blueprint asks Jev."""

At = Literal[
    "before_agent",
    "model_call",
    "before_model",
    "model_output",
    "tool_call",
    "tool_result",
    "final_answer",
]
"""Where in the agent loop an action acts."""

STAGES: tuple[Stage, ...] = ("entry", "model_output", "tool_call", "tool_result", "final_answer")
POSITIONS: tuple[At, ...] = (
    "before_agent",
    "model_call",
    "before_model",
    "model_output",
    "tool_call",
    "tool_result",
    "final_answer",
)
TOOL_STAGES: frozenset[str] = frozenset({"tool_call", "tool_result"})
TOOL_POSITIONS: frozenset[str] = frozenset({"tool_call", "tool_result"})

OWN: dict[str, At] = {
    "entry": "before_agent",
    "model_output": "model_output",
    "tool_call": "tool_call",
    "tool_result": "tool_result",
    "final_answer": "final_answer",
}
"""The position each stage acts at in the hook where it asks Jev."""

REACHABLE: dict[str, frozenset[str]] = {
    "entry": frozenset(POSITIONS),
    "model_output": frozenset(POSITIONS) - {"before_agent"},
    "tool_call": frozenset(POSITIONS) - {"before_agent"},
    "tool_result": frozenset(POSITIONS) - {"before_agent"},
    "final_answer": frozenset({"final_answer"}),
}
"""The positions an action can reach from each stage: its own hook, or a later one."""

DECIDE_AT: dict[str, tuple[At, ...]] = {
    "entry": ("before_agent", "model_call"),
    "model_output": ("model_output",),
    "tool_call": ("tool_call",),
    "tool_result": ("tool_result",),
    "final_answer": ("final_answer",),
}
"""Where the actions `decide` returns may act when `acts_at` lists no positions."""

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
"""Blueprint names, question ids, and threshold names: they become `name.id` keys."""


@dataclass(frozen=True)
class _Action:
    at: At | None = field(default=None, kw_only=True)
    """Where the action acts; `None` picks its usual position for the blueprint's stage."""


@dataclass(frozen=True)
class _ToolAction(_Action):
    tools: Sequence[str] = field(default=(), kw_only=True)
    """The tools it acts on. Needed unless it acts on the blueprint's own tools."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools", _names(self.tools))


@dataclass(frozen=True)
class Refuse(_ToolAction):
    """Stop and say why.

    - `before_agent`, `before_model`: end the run with `message` as the reply.
    - `model_output`: replace the reply with `message` and end the run.
    - `tool_call`: skip the call; the model reads `message` as the call's result.
    - `tool_result`: withhold the result and end the run with `message`.
    """

    message: str | None = None
    """`None` uses `GuardSettings.refusal_message`, or at `tool_call` a note for the model."""


@dataclass(frozen=True)
class SystemNote(_Action):
    """Add `text` to the system prompt of every model call (`model_call`)."""

    text: str


@dataclass(frozen=True)
class UseModel(_Action):
    """Call `model` instead of the agent's model (`model_call`)."""

    model: str | BaseChatModel
    """A `provider:model` string for `init_chat_model`, or a chat model."""


@dataclass(frozen=True)
class NoTools(_Action):
    """Offer the model no tools, so it answers in text (`model_call`)."""


@dataclass(frozen=True)
class DropTools(_Action):
    """Offer the model every tool except `names` (`model_call`)."""

    names: Sequence[str]

    def __post_init__(self) -> None:
        """Keep `names` as a tuple, so equal actions compare equal."""
        object.__setattr__(self, "names", _names(self.names))


@dataclass(frozen=True)
class ShapeModelCall(_Action):
    """Change each model call yourself (`model_call`).

    `shape` gets the `ModelRequest` and returns it changed, usually through
    `request.override(...)`: its model, system message, messages, tools, or settings.
    """

    shape: Callable[[ModelRequest], ModelRequest]


@dataclass(frozen=True)
class AskHuman(_ToolAction):
    """Pause the call until a person approves it (`tool_call`; needs a checkpointer)."""

    reason: str | None = None
    """Shown to the reviewer; `None` says which blueprint flagged the call."""


@dataclass(frozen=True)
class ShapeToolCall(_ToolAction):
    """Change a call's arguments before the tool runs (`tool_call`).

    `shape` gets the call (`name`, `args`, `id`) and returns the arguments to run it with.
    """

    shape: Callable[[ToolCall], Mapping[str, object]]


@dataclass(frozen=True)
class Warn(_ToolAction):
    """Keep the tool result and put `note` in front of it (`tool_result`)."""

    note: str | None = None


@dataclass(frozen=True)
class Redact(_ToolAction):
    """Replace the tool result with `note` (`tool_result`)."""

    note: str | None = None


@dataclass(frozen=True)
class RewriteToolResult(_ToolAction):
    """Rewrite the text of the tool result yourself (`tool_result`)."""

    rewrite: Callable[[str], str]


@dataclass(frozen=True)
class RewriteReply(_Action):
    """Rewrite the reply's text yourself; its tool calls stay (`model_output`, `final_answer`)."""

    rewrite: Callable[[str], str]


@dataclass(frozen=True)
class Mark(_Action):
    """Record `reason` in the final answer's `additional_kwargs["jev_checks"]` (`final_answer`).

    The answer keeps its text; `VerifySettings(annotate=True)` also adds a visible note.
    """

    reason: str


@dataclass(frozen=True)
class Notify(_Action):
    """Call `notify(answers, final_answer_text)` once when the run ends (`final_answer`).

    It is called even when a refusal ended the run, so it can report what was stopped.
    """

    notify: Callable[[Mapping[str, Answer], str], None]


Action: TypeAlias = (
    Refuse
    | SystemNote
    | UseModel
    | NoTools
    | DropTools
    | ShapeModelCall
    | AskHuman
    | ShapeToolCall
    | Warn
    | Redact
    | RewriteToolResult
    | RewriteReply
    | Mark
    | Notify
)
"""What a custom blueprint does when its questions call for it."""

Decide: TypeAlias = Callable[
    ["Mapping[str, Answer]", "Mapping[str, float]"], "Action | Sequence[Action] | None"
]
"""Answers by question id and thresholds by name in; actions, or `None` (pass), out."""

Placed: TypeAlias = tuple[Action, At]
"""An action and the position it acts at."""

_MODEL_CALL = frozenset({"model_call"})
ACTION_AT: dict[type, frozenset[str]] = {
    Refuse: frozenset({"before_agent", "before_model", "model_output", "tool_call", "tool_result"}),
    SystemNote: _MODEL_CALL,
    UseModel: _MODEL_CALL,
    NoTools: _MODEL_CALL,
    DropTools: _MODEL_CALL,
    ShapeModelCall: _MODEL_CALL,
    AskHuman: frozenset({"tool_call"}),
    ShapeToolCall: frozenset({"tool_call"}),
    Warn: frozenset({"tool_result"}),
    Redact: frozenset({"tool_result"}),
    RewriteToolResult: frozenset({"tool_result"}),
    RewriteReply: frozenset({"model_output", "final_answer"}),
    Mark: frozenset({"final_answer"}),
    Notify: frozenset({"final_answer"}),
}
"""The positions each action can act at."""

ACTIONS: tuple[type, ...] = tuple(ACTION_AT)

_ACTION_NAMES: dict[type, str] = {
    Refuse: "refuse",
    SystemNote: "note",
    UseModel: "use_model",
    NoTools: "no_tools",
    DropTools: "drop_tools",
    ShapeModelCall: "shape_model_call",
    AskHuman: "ask_human",
    ShapeToolCall: "shape_tool_call",
    Warn: "warn",
    Redact: "redact",
    RewriteToolResult: "rewrite_tool_result",
    RewriteReply: "rewrite_reply",
    Mark: "mark",
    Notify: "notify",
}


def action_name(action: Action) -> str:
    """The name events use for `action`, such as `refuse` or `note`."""
    return _ACTION_NAMES[type(action)]


def tools_of(action: Action) -> tuple[str, ...]:
    """The tools `action` names itself; empty for actions that take none."""
    return cast("tuple[str, ...]", action.tools) if isinstance(action, _ToolAction) else ()


@dataclass(frozen=True)
class JevBlueprint:
    """A blueprint of your own: its questions, where they are asked, and what to do where.

    Give `when` and `act` for the common case: any yes/no question at or above its
    threshold triggers the actions in `act`. Give `decide` instead to read Choice or
    Score answers, to combine questions, or to pick actions from the answers. Answers
    are `NoulAnswer` (`.noul`, the probability of yes), `ChoiceAnswer` (`.choice`,
    `.probabilities`), or `ScoreAnswer` (`.score`). `decide` and the functions in
    `ShapeModelCall`, `ShapeToolCall`, `RewriteToolResult`, and `RewriteReply` may run
    more than once a run, so keep them free of side effects; `Notify` runs once.

    Each action acts at its `at`, by default its usual position for the stage. Actions
    `decide` returns may only act at the positions in `acts_at`, so the stack can
    build every position they need.

    Every name a built-in blueprint takes works for this one too: turn it on in
    `JevConfig.blueprints`, reword `<name>.<question>` in `JevConfig.questions`, set
    `<name>.<threshold>` in `JevConfig.thresholds`, give it a model in `JevConfig.models`,
    and list it in `JevConfig.fail_closed`.

    Example:
        ```python
        from langchain_jev import JevBlueprint, JevConfig, Refuse, SystemNote, noul

        personal_data = JevBlueprint(
            name="personal_data",
            stage="tool_result",
            tools=["read_ticket"],
            questions={
                "has_personal_data": noul(
                    "Does `tool_output` contain personal data such as phone numbers?",
                    yes="It contains personal data.",
                    no="It contains no personal data.",
                )
            },
            when={"has_personal_data": 0.7},
            act=[
                Refuse(
                    "No external email after reading personal data.",
                    at="tool_call",
                    tools=["send_email"],
                ),
                SystemNote("Do not repeat personal data to the user.", at="model_call"),
            ],
        )
        config = JevConfig(
            blueprints={"personal_data": "on"},
            custom_blueprints=[personal_data],
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

    act: Action | Sequence[Action] | None = None
    """The action, or actions, `when` triggers."""

    decide: Decide | None = None
    """Choose the actions yourself, instead of `when` and `act`."""

    thresholds: Mapping[str, float] = field(default_factory=dict)
    """Named values passed to `decide`, so `JevConfig.thresholds` can tune them."""

    tools: Sequence[str] = ()
    """The tools a `tool_call` or `tool_result` blueprint asks about; required there."""

    acts_at: Sequence[At] = ()
    """The positions the actions `decide` returns may act at; see `DECIDE_AT`."""

    def __post_init__(self) -> None:
        """Check the blueprint when it is defined.

        Raises:
            ValueError: If a name, stage, question, threshold, position, or tool list is
                missing, unknown, or out of range, or `when`/`act` and `decide` are mixed.
            TypeError: If a question, threshold, or action is of the wrong kind, or an
                action cannot act at its position.
        """
        _check_name(self.name, "Jev blueprint names")
        if self.stage not in STAGES:
            msg = f"Jev blueprint {self.name!r} has unknown stage {self.stage!r}; use {STAGES}."
            raise ValueError(msg)
        self._check_questions()
        self._check_tools()
        if self.decide is None:
            self._check_rule()
        else:
            self._check_decide()

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

    def _check_tools(self) -> None:
        object.__setattr__(self, "tools", _names(self.tools))
        if self.stage in TOOL_STAGES and not self.tools:
            msg = f"Jev blueprint {self.name!r} at stage {self.stage!r} needs `tools` to check."
            raise ValueError(msg)
        if self.stage not in TOOL_STAGES and self.tools:
            msg = (
                f"Jev blueprint {self.name!r}: `tools` only apply at the tool_call and "
                "tool_result stages; give an action its own `tools` instead."
            )
            raise ValueError(msg)

    def _check_decide(self) -> None:
        if self.when or self.act is not None:
            msg = f"Jev blueprint {self.name!r}: give either `when` and `act`, or `decide`."
            raise ValueError(msg)
        for key, value in self.thresholds.items():
            _check_name(key, f"Thresholds of Jev blueprint {self.name!r}")
            _check_number(self.name, key, value)
        object.__setattr__(self, "acts_at", tuple(self.acts_at))
        for at in self.acts_at:
            self._reachable(at, "`acts_at`")

    def _check_rule(self) -> None:
        if not self.actions() or not self.when:
            msg = (
                f"Jev blueprint {self.name!r} needs `when` and `act` (the questions that "
                "trigger the actions), or `decide`."
            )
            raise ValueError(msg)
        if self.thresholds or self.acts_at:
            msg = (
                f"Jev blueprint {self.name!r}: `thresholds` and `acts_at` are for `decide`; "
                "with `when`, its values are the thresholds and each action's `at` says "
                "where it acts."
            )
            raise ValueError(msg)
        for qid, value in self.when.items():
            self._check_trigger(qid, value)
        self.check(self.actions(), source="`act`")

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

    def actions(self) -> tuple[Action, ...]:
        """The actions in `act`, as a tuple."""
        if self.act is None:
            return ()
        if isinstance(self.act, (list, tuple)):
            return tuple(self.act)
        return (cast("Action", self.act),)

    def positions(self) -> tuple[At, ...]:
        """Every position this blueprint's actions may act at, in loop order."""
        if self.decide is not None:
            chosen = set(self.acts_at or DECIDE_AT[self.stage])
        else:
            chosen = {self.place(action) for action in self.actions()}
        return tuple(at for at in POSITIONS if at in chosen)

    def place(self, action: Action) -> At:
        """The position `action` acts at.

        Raises:
            TypeError: If the action cannot act at its `at`, or anywhere after this stage.
            ValueError: If its `at` comes before this stage, or it needs an `at`.
        """
        kind = type(action).__name__
        allowed = ACTION_AT[type(action)]
        if action.at is not None:
            if action.at in POSITIONS and action.at not in allowed:
                msg = f"{kind} cannot act at {action.at!r}; it acts at {_listed(allowed)}."
                raise TypeError(msg)
            return self._reachable(action.at, kind)
        reachable = allowed & REACHABLE[self.stage]
        if not reachable:
            msg = (
                f"{kind} acts at {_listed(allowed)}, which Jev blueprint {self.name!r} "
                f"cannot reach from stage {self.stage!r}."
            )
            raise TypeError(msg)
        if OWN[self.stage] in reachable:
            return OWN[self.stage]
        if len(reachable) == 1:
            return cast("At", next(iter(reachable)))
        msg = f"{kind} in Jev blueprint {self.name!r} can act at {_listed(reachable)}; give `at`."
        raise ValueError(msg)

    def _reachable(self, at: object, what: str) -> At:
        if at not in POSITIONS:
            msg = (
                f"{what} of Jev blueprint {self.name!r} names unknown position {at!r}; "
                f"use {POSITIONS}."
            )
            raise ValueError(msg)
        if at not in REACHABLE[self.stage]:
            msg = (
                f"{what} of Jev blueprint {self.name!r} acts at {at!r}, before Jev is asked "
                f"at stage {self.stage!r}; choose from {_listed(REACHABLE[self.stage])}."
            )
            raise ValueError(msg)
        return cast("At", at)

    def tools_for(self, action: Action, at: At) -> tuple[str, ...]:
        """The tools `action` acts on at `at`: its own, or this blueprint's for the same tool."""
        own = tools_of(action)
        if own or at not in TOOL_POSITIONS:
            return own
        same_tool = self.stage == at or (self.stage == "tool_call" and at == "tool_result")
        return cast("tuple[str, ...]", self.tools) if same_tool else ()

    def check(self, result: object, *, source: str = "`decide`") -> tuple[Placed, ...]:
        """`result` as actions with their positions, after checking each one.

        Args:
            result: `None`, an action, or a list or tuple of actions.
            source: Where `result` came from, for error messages.

        Raises:
            TypeError: If an item is not an action, or cannot act where it would.
            ValueError: If an action's position or tools do not fit this blueprint.
        """
        if result is None:
            return ()
        items = tuple(result) if isinstance(result, (list, tuple)) else (result,)
        return tuple(self._placed(item, source) for item in items)

    def _placed(self, item: object, source: str) -> Placed:
        if not isinstance(item, ACTIONS):
            names = ", ".join(t.__name__ for t in ACTIONS)
            msg = f"{source} of Jev blueprint {self.name!r} gave {item!r}; use one of {names}."
            raise TypeError(msg)
        action = cast("Action", item)
        at = self.place(action)
        if self.decide is not None and at not in self.positions():
            msg = (
                f"{source} of Jev blueprint {self.name!r} gave {type(action).__name__} at "
                f"{at!r}; list {at!r} in `acts_at` so the stack builds that position."
            )
            raise TypeError(msg)
        self._check_action_tools(action, at, source)
        return action, at

    def _check_action_tools(self, action: Action, at: At, source: str) -> None:
        kind = type(action).__name__
        if at not in TOOL_POSITIONS and tools_of(action):
            msg = f"{source} of Jev blueprint {self.name!r}: {kind} at {at!r} takes no `tools`."
            raise ValueError(msg)
        if at in TOOL_POSITIONS and not self.tools_for(action, at):
            msg = (
                f"{source} of Jev blueprint {self.name!r}: {kind} at {at!r} needs `tools`, "
                "the tools whose calls or results it acts on."
            )
            raise ValueError(msg)

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


def _names(value: str | Sequence[str]) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else tuple(value)


def _listed(positions: frozenset[str]) -> str:
    return ", ".join(at for at in POSITIONS if at in positions)


def _check_name(value: object, what: str) -> None:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        msg = f"{what} use letters, digits, and underscores and start with a letter; got {value!r}."
        raise ValueError(msg)


def _check_number(blueprint: str, key: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"Threshold {key!r} of Jev blueprint {blueprint!r} must be a number, got {value!r}."
        raise TypeError(msg)
