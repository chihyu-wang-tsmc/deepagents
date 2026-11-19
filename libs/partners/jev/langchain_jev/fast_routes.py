"""Requests Jev can handle on its own, without calling the agent's model.

Two kinds of route feed the `fast_path` blueprint:

- `FunctionRoute` follows https://docs.typesafe.ai/cookbooks/function_calling: one
    Choice picks the function, one question per closed-set argument fills it, and
    a companion Noul (https://docs.typesafe.ai/patterns/companion-noul) says
    whether any function fits at all.
- `DecisionRoute` follows the customer-service workflow on
    https://evals.typesafe.ai/customer_service: Jev answers your questions and
    your own rules turn the answers into the next action.
"""

from __future__ import annotations

import inspect
import math
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from langchain_core.decisions import Choice, Noul, Question
from langchain_core.decisions.types import Answer, ChoiceAnswer, NoulAnswer

from langchain_jev.question_book import default_book

if TYPE_CHECKING:
    from langchain_jev.question_book import QuestionBook

Shape = Literal["choice", "set", "flag"]
JsonArg = str | bool | list[str] | None


@dataclass(frozen=True)
class Reply:
    """Answer the user with fixed text."""

    text: str


@dataclass(frozen=True)
class CallFunction:
    """Run a function and answer with its result."""

    name: str
    function: Callable[..., object]
    kwargs: dict[str, JsonArg]
    confidence: float = 1.0
    render: Callable[[object], str] | None = None
    """Turns the function's return value into the reply; `str` by default."""


@dataclass(frozen=True)
class HandOff:
    """Answer with `message` and report a handoff event for a person to pick up."""

    reason: str
    message: str


FastAction = Reply | CallFunction | HandOff


class FastRoute(Protocol):
    """A set of questions about the user's message and the rule that reads them."""

    name: str

    def questions(self) -> dict[str, Question]:
        """Questions asked about the user's message, keyed by local id."""
        ...

    def decide(
        self, answers: Mapping[str, Answer], thresholds: Mapping[str, float]
    ) -> FastAction | None:
        """The action to take, or `None` to let the agent's model handle it."""
        ...


@dataclass(frozen=True)
class DecisionRoute:
    """Your questions, and your rule that turns their answers into an action."""

    name: str
    asked: Mapping[str, Question]
    rule: Callable[[Mapping[str, Answer]], FastAction | None]

    def questions(self) -> dict[str, Question]:
        """The questions, as given."""
        return dict(self.asked)

    def decide(
        self,
        answers: Mapping[str, Answer],
        thresholds: Mapping[str, float],  # noqa: ARG002 - your rule holds its own thresholds
    ) -> FastAction | None:
        """Apply the rule."""
        return self.rule(answers)


@dataclass(frozen=True)
class FunctionRoute:
    """Call one of `functions` when a request clearly asks for it.

    `spec` uses the cookbook's format: `{"functions": {name: {"description": ...,
    "arguments": {arg: {"question": ..., "options": {value: description},
    "stated": ...}}}}}`. Only closed-set arguments (`Literal`, `list[Literal]`, and
    `bool`) are asked about; every other argument keeps its default.
    """

    name: str
    spec: Mapping[str, object]
    functions: Mapping[str, Callable[..., object]]
    render: Callable[[object], str] | None = None
    shapes: dict[str, dict[str, tuple[Shape, tuple]]] = field(init=False)

    def __post_init__(self) -> None:
        """Read the closed sets once."""
        shapes = {name: closed_sets(fn) for name, fn in self.functions.items()}
        object.__setattr__(self, "shapes", shapes)

    def questions(self, book: QuestionBook | None = None) -> dict[str, Question]:
        """The function choice, the companion Noul, and one question per argument.

        Args:
            book: Wording from `JevConfig.questions`; a `question` in `spec` still
                wins for this route's function choice.
        """
        book = book or default_book()
        described = _spec_functions(self.spec)
        which = self.spec.get("question")
        instructions = str(which) if which is not None else book.instructions("fast_path.which")
        questions: dict[str, Question] = {
            "__tool__": Choice(
                instructions=instructions,
                criteria={name: described[name]["description"] for name in self.functions},
            ),
            "__fits__": book.question(
                "fast_path.functions_fit", functions=", ".join(self.functions)
            ),
        }
        for name, shapes in self.shapes.items():
            arguments = described[name].get("arguments", {})
            for argument, (shape, values) in shapes.items():
                questions.update(
                    _argument_questions(f"{name}.{argument}", shape, values, arguments[argument])
                )
        return questions

    def decide(
        self, answers: Mapping[str, Answer], thresholds: Mapping[str, float]
    ) -> FastAction | None:
        """Call the chosen function when it fits and every judgment is confident."""
        fits = answers.get("__fits__")
        route = answers.get("__tool__")
        if not isinstance(fits, NoulAnswer) or not isinstance(route, ChoiceAnswer):
            return None
        if fits.noul < thresholds["fast_path.min_fits"]:
            return None
        call = self._call(route, answers)
        if call.confidence < thresholds["fast_path.min_confidence"]:
            return None
        return call

    def _call(self, route: ChoiceAnswer, answers: Mapping[str, Answer]) -> CallFunction:
        name = route.choice
        function = self.functions[name]
        defaults = {p.name: p.default for p in inspect.signature(function).parameters.values()}
        kwargs: dict[str, JsonArg] = {}
        judgements = [route.probabilities.get(name, route.confidence)]
        for argument, (shape, values) in self.shapes[name].items():
            value, probabilities = _argument_value(f"{name}.{argument}", shape, values, answers)
            judgements.extend(probabilities)
            if not isinstance(value, _Omitted):
                kwargs[argument] = value
            elif defaults[argument] is inspect.Parameter.empty:
                judgements.append(0.0)  # a required argument nobody stated
        return CallFunction(
            name=name,
            function=function,
            kwargs=kwargs,
            confidence=min(judgements),
            render=self.render,
        )


class _Omitted:
    pass


_OMITTED = _Omitted()


def closed_sets(fn: Callable[..., object]) -> dict[str, tuple[Shape, tuple]]:
    """Map each closed-set argument of `fn` to its shape and allowed values."""
    hints = typing.get_type_hints(fn)
    shapes: dict[str, tuple[Shape, tuple]] = {}
    for name in inspect.signature(fn).parameters:
        shape = _shape(hints.get(name))
        if shape is not None:
            shapes[name] = shape
    return shapes


def _shape(hint: object) -> tuple[Shape, tuple] | None:
    if hint is bool:
        return ("flag", (True, False))
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        members = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(members) != 1:
            return None
        return _shape(members[0])
    if origin is Literal:
        return ("choice", typing.get_args(hint))
    if origin is list:
        (item,) = typing.get_args(hint)
        if typing.get_origin(item) is Literal:
            return ("set", typing.get_args(item))
    return None


def _spec_functions(spec: Mapping[str, object]) -> dict[str, dict]:
    functions = spec.get("functions")
    if not isinstance(functions, Mapping):
        msg = "A function route spec needs a `functions` mapping."
        raise TypeError(msg)
    return {str(name): dict(entry) for name, entry in functions.items()}


def _argument_questions(
    qid: str, shape: Shape, values: tuple, entry: Mapping[str, object]
) -> dict[str, Question]:
    question = str(entry["question"])
    asked: dict[str, Question] = {}
    if shape == "choice":
        options = entry["options"]
        if not isinstance(options, Mapping):
            msg = f"Argument {qid} needs `options` describing each value."
            raise TypeError(msg)
        asked[qid] = Choice(instructions=question, criteria={str(k): v for k, v in options.items()})
    elif shape == "flag":
        asked[qid] = Noul(instructions=question)
    else:
        for member in values:
            asked[f"{qid}.{member}"] = Noul(instructions=question.format(member))
    if "stated" in entry:
        asked[f"{qid}?"] = Noul(instructions=str(entry["stated"]))
    return asked


def _argument_value(
    qid: str, shape: Shape, values: tuple, answers: Mapping[str, Answer]
) -> tuple[JsonArg | _Omitted, list[float]]:
    """The argument's value and the probabilities behind it."""
    stated = answers.get(f"{qid}?")
    judgements: list[float] = []
    if isinstance(stated, NoulAnswer):
        if stated.noul < 0.5:  # noqa: PLR2004 - the cookbook omits an unstated argument
            return _OMITTED, [1 - stated.noul]
        judgements.append(stated.noul)
    if shape == "choice":
        answer = answers[qid]
        if not isinstance(answer, ChoiceAnswer):
            return _OMITTED, [0.0]
        by_text = {str(v): v for v in values}
        return by_text[answer.choice], [
            *judgements,
            answer.probabilities[answer.choice],
        ]
    if shape == "flag":
        answer = answers[qid]
        p = answer.noul if isinstance(answer, NoulAnswer) else 0.0
        return p >= 0.5, [*judgements, max(p, 1 - p)]  # noqa: PLR2004
    members = {m: answers.get(f"{qid}.{m}") for m in values}
    probabilities = {m: a.noul if isinstance(a, NoulAnswer) else 0.0 for m, a in members.items()}
    chosen = [str(m) for m, p in probabilities.items() if p >= 0.5]  # noqa: PLR2004
    certainty = [max(p, 1 - p) for p in probabilities.values()]
    return chosen, [*judgements, math.prod(certainty)]
