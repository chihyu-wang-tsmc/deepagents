"""See what each Jev request read, asked, and got back, to tune the questions.

A `JevTrace` set on `JevConfig.trace` is handed every request the stack sends to Jev
(answered, failed, or served from the cache) and every decision a blueprint makes.
It prints a short block per request and can append each request and decision to a
JSONL file, one JSON object per line, to work through a batch of your own inputs:

- what Jev read: the state the blueprint sends, such as the user's message;
- each question and its answer: the probability of yes for a Noul, each option's
  probability for a Choice, and the level for a Score;
- the thresholds of the blueprints asking, after every override;
- the decisions, such as `pass` or `route:fast`.

`JevTrace.from_env()` turns it on without code changes: `JEV_TRACE=1` prints the
short form, `JEV_TRACE=full` adds each question's wording and the whole state, and
`JEV_TRACE_FILE=jev_trace.jsonl` writes the file.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.decisions.types import ChoiceAnswer, NoulAnswer, ScoreAnswer

from langchain_jev._types import INPUT_GUARD, MODEL_ROUTER, OUTPUT_GUARD
from langchain_jev.classifiers import serialize_state

if TYPE_CHECKING:
    from langchain_core.decisions import Question
    from langchain_core.decisions.types import Answer

    from langchain_jev._types import JevEvent

_CLIP = 160
"""Longest text the short form prints for one value of the state."""

_OFF = frozenset({"", "0", "false", "no", "off"})

_GUARDS = frozenset({INPUT_GUARD, OUTPUT_GUARD})
"""Guards ask about hazard `x` as `hazard.x`; `JevConfig.questions` rewords it as `x`."""

_REWORD: dict[tuple[str, str], str] = {(MODEL_ROUTER, "model_route"): "route"}
"""Questions sent under another id than the one `JevConfig.questions` rewords them by."""


@dataclass(frozen=True)
class JevRequest:
    """One request a Jev stack sent, and what came back."""

    blueprints: tuple[str, ...]
    """The blueprints whose questions these are."""

    model: str
    """The name of the Jev model asked."""

    state: object
    """What Jev read."""

    questions: Mapping[str, Question]
    """The questions by the id they were sent under."""

    answers: Mapping[str, Answer] | None
    """The answers by the same ids, or `None` when the request failed."""

    thresholds: Mapping[str, float]
    """The thresholds of `blueprints` by full name, after every override."""

    cached: bool = False
    """Whether the answers came from the cache instead of a new request."""

    error: str | None = None
    """Why the request failed."""

    entry: bool = False
    """Whether this is the entry request, whose ids are `blueprint::id`."""


class JevTrace:
    """Print each Jev request and decision, and append them to a JSONL file.

    Example:
        ```python
        from langchain_jev import JevConfig, JevTrace

        config = JevConfig(
            blueprints={"input_guard": "on"},
            trace=JevTrace(path="jev_trace.jsonl"),  # or JevTrace.from_env()
        )
        ```
    """

    def __init__(
        self,
        *,
        full: bool = False,
        path: str | os.PathLike[str] | None = None,
        echo: Callable[[str], object] | None = print,
        decisions: bool = True,
    ) -> None:
        """Choose what to print and where to write.

        Args:
            full: Also print each question's wording and the whole state.
            path: Append each request and decision to this file as one JSON line.
            echo: Where printed lines go; `None` prints nothing.
            decisions: Print a line per decision too. Turn it off when
                `JevConfig.on_event` already prints them; the file keeps them anyway.
        """
        self.full = full
        self.path = Path(path) if path is not None else None
        self.echo = echo
        self.decisions = decisions
        self._lock = threading.Lock()

    @classmethod
    def from_env(
        cls, *, decisions: bool = True, echo: Callable[[str], object] = print
    ) -> JevTrace | None:
        """A trace set by `JEV_TRACE` and `JEV_TRACE_FILE`, or `None` when neither is set.

        `JEV_TRACE=1` prints the short form and `JEV_TRACE=full` the full one.
        `JEV_TRACE_FILE` names the JSONL file; set alone, it writes without printing.

        Args:
            decisions: Passed to the trace; see `JevTrace`.
            echo: Where printed lines go.
        """
        level = os.environ.get("JEV_TRACE", "").strip().lower()
        path = os.environ.get("JEV_TRACE_FILE", "").strip() or None
        shown = level not in _OFF
        if not shown and path is None:
            return None
        return cls(
            full=level == "full", path=path, echo=echo if shown else None, decisions=decisions
        )

    def request(self, request: JevRequest) -> None:
        """Print and record one request."""
        lines = _request_lines(request, full=self.full) if self.echo is not None else []
        self._write(lines, _request_record(request))

    def decision(self, event: JevEvent) -> None:
        """Print and record one decision."""
        shown = self.echo is not None and self.decisions
        lines = [f"[jev decision] {event.blueprint} {event.mode} {event.action}"] if shown else []
        self._write(lines, _decision_record(event))

    def _write(self, lines: list[str], record: dict[str, object]) -> None:
        with self._lock:
            if self.echo is not None:
                for line in lines:
                    self.echo(line)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _request_lines(request: JevRequest, *, full: bool) -> list[str]:
    lines = [_header(request), *_state_lines(request.state, full=full)]
    if request.answers is None:
        lines.append(f"  failed: {request.error}")
    for key, question in request.questions.items():
        answer = request.answers.get(key) if request.answers is not None else None
        lines.append(f"  {_label(request, key):<36} {_answer_text(question, answer)}")
        if full:
            lines.append(f"      {question.instructions}")
    if request.thresholds:
        values = ", ".join(f"{name} {value:g}" for name, value in request.thresholds.items())
        lines.append(f"  thresholds: {values}")
    return lines


def _header(request: JevRequest) -> str:
    where = "entry: " if request.entry else ""
    notes = [request.model, *(["cached"] if request.cached else [])]
    return f"[jev request] {where}{', '.join(request.blueprints)} · {' · '.join(notes)}"


def _label(request: JevRequest, key: str) -> str:
    """`blueprint.question id`: for most questions, the key `JevConfig.questions` takes."""
    head, sep, rest = key.partition("::")
    own = request.entry and sep and head in request.blueprints
    blueprint, qid = (head, rest) if own else (request.blueprints[0], key)
    if blueprint in _GUARDS:
        qid = qid.removeprefix("hazard.")
    return f"{blueprint}.{_REWORD.get((blueprint, qid), qid)}"


def _state_lines(state: object, *, full: bool) -> list[str]:
    if full:
        text = json.dumps(_json(state), ensure_ascii=False, indent=2)
        return ["  read:", *(f"    {line}" for line in text.splitlines())]
    if not isinstance(state, Mapping):
        return [f"  read: {_summary(state)}"]
    lines = [f"{key} = {_summary(value)}" for key, value in state.items()]
    return [f"  {'read: ' if i == 0 else '      '}{line}" for i, line in enumerate(lines)]


def _summary(value: object) -> str:
    if isinstance(value, str):
        flat = " ".join(value.split())
        return json.dumps(flat if len(flat) <= _CLIP else f"{flat[:_CLIP]}…", ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return f"{len(value)} items"
    text = json.dumps(_json(value), ensure_ascii=False, default=str)
    return text if len(text) <= _CLIP else f"{text[:_CLIP]}…"


def _answer_text(question: Question, answer: Answer | None) -> str:
    if answer is None:
        return "—"
    if isinstance(answer, NoulAnswer):
        return f"yes {answer.noul:.2f}"
    if isinstance(answer, ChoiceAnswer):
        ranked = sorted(answer.probabilities.items(), key=lambda item: -item[1])[:5]
        return " | ".join(f"{name} {p:.2f}" for name, p in ranked) or str(answer.choice)
    if isinstance(answer, ScoreAnswer):
        criteria = getattr(question, "criteria", None)
        levels = len(criteria) if isinstance(criteria, list) else 0
        top = f" of {levels - 1}" if levels else ""
        return f"score {answer.score:g}{top}"
    return str(answer)


def _json(value: object) -> object:
    try:
        return serialize_state(value)
    except TypeError:
        return str(value)


def _request_record(request: JevRequest) -> dict[str, object]:
    questions = request.questions.items()
    answers = None if request.answers is None else request.answers.items()
    return {
        "type": "request",
        "time": _now(),
        "model": request.model,
        "blueprints": list(request.blueprints),
        "cached": request.cached,
        "state": _json(request.state),
        "questions": {k: q.model_dump(mode="json", exclude_none=True) for k, q in questions},
        "answers": None if answers is None else {k: a.model_dump(mode="json") for k, a in answers},
        "error": request.error,
        "thresholds": dict(request.thresholds),
    }


def _decision_record(event: JevEvent) -> dict[str, object]:
    return {
        "type": "decision",
        "time": _now(),
        "blueprint": event.blueprint,
        "mode": event.mode,
        "action": event.action,
        "details": event.details,
    }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")
