"""Check the final answer, then retry, escalate, or accept it.

Three checks share one hook and run in parallel:

- `verify_answer` (https://docs.typesafe.ai/cookbooks/verify_llm_outputs): does the
    answer follow from the evidence the agent retrieved?
- `citation_check` (https://docs.typesafe.ai/cookbooks/citation_check): is each quote
    in its source, and does the source support the claim?
- `answer_scoring` (https://docs.typesafe.ai/patterns/composite-scoring): weighted
    Score questions about the answer's quality.

A failed check sends feedback to the model and loops back, as `RubricMiddleware`
does. With `escalation` on, the retry runs on the router's stronger model: the
"cheap first, escalate on failure" cascade of https://docs.typesafe.ai/cookbooks/sde_cascade.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from langchain.agents.middleware.types import hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_jev._messages import (
    append_system_text,
    clip,
    current_turn,
    final_answer,
    latest_request,
)
from langchain_jev._types import (
    ANSWER_SCORING,
    CITATION_CHECK,
    ESCALATION,
    VERIFY_ANSWER,
    JevEvent,
)
from langchain_jev.config import ScoreSpec
from langchain_jev.middleware._base import Ask, AskMany, Flow, JevMiddleware
from langchain_jev.middleware.guardrails import JEV_MARK
from langchain_jev.questions import (
    CHECK_QUESTIONS,
    CITATION_INSTRUCTIONS,
    CITATION_RELATION,
    DEFAULT_SCORES,
    RELATION_TO_VERDICT,
)
from langchain_jev.switches import mode_for

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain_core.decisions import ClassifierResponse
    from langgraph.runtime import Runtime

    from langchain_jev._types import Mode
    from langchain_jev.state import JevState

_CITATION = re.compile(r'\[cite:\s*(?P<source>[^|\]]+?)\s*\|\s*"(?P<quote>[^"]+)"\s*\]')
_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})
"""Fold curly quotes so a quote still matches its source (from the citation cookbook)."""
CHECKS = (VERIFY_ANSWER, CITATION_CHECK, ANSWER_SCORING)


@dataclass
class CheckResult:
    """The outcome of one check on one answer."""

    blueprint: str
    failed: bool
    summary: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass
class _Check:
    blueprint: str
    asks: tuple[Ask, ...]
    evaluate: Callable[[list[ClassifierResponse | None]], CheckResult | None]


class AnswerCheckMiddleware(JevMiddleware):
    """Verify, citation-check, and score the final answer; retry or escalate on failure."""

    blueprint = VERIFY_ANSWER

    @hook_config(can_jump_to=["model"])
    def after_agent(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """Check synchronously."""
        return self.drive(self._flow(state, runtime))

    @hook_config(can_jump_to=["model"])
    async def aafter_agent(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """Check asynchronously."""
        return await self.adrive(self._flow(state, runtime))

    def _flow(self, state: JevState, runtime: Runtime) -> Flow[dict[str, object] | None]:
        reply = final_answer(state["messages"])
        if reply is None or state.get("jev_blocked") or JEV_MARK in reply.additional_kwargs:
            return None
        snapshot = cast(
            "Mapping[str, Mode]",
            state.get("jev_modes") or self.jev.snapshot(runtime.context),
        )
        modes = {bp: mode_for(snapshot, bp) for bp in CHECKS}
        checks = self._plan(state, reply.text, modes)
        if not checks:
            return None
        asks = tuple(ask for check in checks for ask in check.asks)
        responses = cast("list[ClassifierResponse | None]", (yield AskMany(asks)) if asks else [])
        results = self._evaluate(checks, responses, modes)
        return self._after(state, snapshot, reply, results, modes)

    def _plan(self, state: JevState, answer: str, modes: Mapping[str, Mode]) -> list[_Check]:
        request = latest_request(state["messages"])
        planned: list[_Check | None] = []
        if modes[VERIFY_ANSWER] != "off":
            planned.append(
                self._verify(
                    request,
                    answer,
                    _evidence(state, self.jev.config.verify.evidence_tools),
                )
            )
        if modes[CITATION_CHECK] != "off":
            planned.append(self._citations(state, answer))
        if modes[ANSWER_SCORING] != "off":
            planned.append(self._scoring(request, answer))
        return [check for check in planned if check is not None]

    def _evaluate(
        self,
        checks: list[_Check],
        responses: list[ClassifierResponse | None],
        modes: Mapping[str, Mode],
    ) -> list[CheckResult]:
        results: list[CheckResult] = []
        start = 0
        for check in checks:
            chunk = responses[start : start + len(check.asks)]
            start += len(check.asks)
            result = check.evaluate(chunk)
            if result is not None:
                self.jev.emit(
                    JevEvent(
                        result.blueprint,
                        modes[result.blueprint],
                        "fail" if result.failed else "pass",
                        result.details,
                    )
                )
                results.append(result)
        return results

    def _verify(self, request: str, answer: str, evidence: list[dict]) -> _Check | None:
        if not evidence:
            return None
        paragraphs = clip(
            "\n\n".join(f"[{p['id']}] {p.get('title', '')}\n{p['text']}" for p in evidence),
            self.jev.max_chars(),
        )
        ask = Ask(
            {
                "request": request,
                "paragraphs": paragraphs,
                "candidate_response": answer,
            },
            CHECK_QUESTIONS,
        )
        limit = self.threshold("verify_answer.review_at")

        def evaluate(responses: list[ClassifierResponse | None]) -> CheckResult | None:
            response = responses[0]
            if response is None:
                return None
            nouls = {key: response.nouls[key].noul for key in CHECK_QUESTIONS}
            risk = nouls["answerable"] if nouls["declines"] >= 0.5 else 1 - nouls["supported"]  # noqa: PLR2004 - the cookbook's split
            summary = f"the answer may not follow from the retrieved sources (risk {risk:.2f})"
            return CheckResult(VERIFY_ANSWER, risk >= limit, summary, {**nouls, "risk": risk})

        return _Check(VERIFY_ANSWER, (ask,), evaluate)

    def _citations(self, state: JevState, answer: str) -> _Check | None:
        citations = parse_citations(answer)
        if not citations:
            return None
        sources = _source_lookup(state, self.jev.config.verify.sources)
        located = [(claim, source, quote, sources(source)) for claim, source, quote in citations]
        asks = tuple(
            Ask(
                {"claim": claim, "section": clip(text, self.jev.max_chars())},
                {"relation": CITATION_RELATION},
            )
            for claim, _, quote, text in located
            if text is not None and _normalize(quote) in _normalize(text)
        )
        auto = self.threshold("citation_check.auto_accept")

        def evaluate(responses: list[ClassifierResponse | None]) -> CheckResult | None:
            answers = iter(responses)
            verdicts = [_verdict(quote, text, answers, auto) for _, _, quote, text in located]
            failed = [v for v in verdicts if v["verdict"] != "verified" and v["auto"]]
            summary = (
                "citations that are missing from, or not supported by, their sources: "
                + ", ".join(
                    f"{c[1]} ({v['verdict']})"
                    for c, v in zip(located, verdicts, strict=True)
                    if v in failed
                )
            )
            return CheckResult(CITATION_CHECK, bool(failed), summary, {"verdicts": verdicts})

        return _Check(CITATION_CHECK, asks, evaluate)

    def _scoring(self, request: str, answer: str) -> _Check:
        specs = self.jev.config.verify.scores or {
            k: ScoreSpec(q) for k, q in DEFAULT_SCORES.items()
        }
        ask = Ask(
            {"request": request, "response": answer},
            {k: spec.question for k, spec in specs.items()},
        )
        limit = self.threshold("answer_scoring.min_score")

        def evaluate(responses: list[ClassifierResponse | None]) -> CheckResult | None:
            response = responses[0]
            if response is None:
                return None
            scores = {
                k: response.scores[k].score / (len(spec.question.criteria) - 1)
                for k, spec in specs.items()
            }
            weight = sum(spec.weight for spec in specs.values()) or 1.0
            composite = sum(scores[k] * spec.weight for k, spec in specs.items()) / weight
            summary = f"the answer scored {composite:.2f} of 1 on quality"
            return CheckResult(
                ANSWER_SCORING,
                composite < limit,
                summary,
                {**scores, "composite": composite},
            )

        return _Check(ANSWER_SCORING, (ask,), evaluate)

    def _after(
        self,
        state: JevState,
        snapshot: Mapping[str, Mode],
        reply: AIMessage,
        results: list[CheckResult],
        modes: Mapping[str, Mode],
    ) -> dict[str, object] | None:
        settings = self.jev.config.verify
        failures = [
            r
            for r in results
            if r.failed and modes[r.blueprint] == "on" and r.blueprint in settings.retry_on
        ]
        attempts = int(state.get("jev_attempts") or 0)
        if not failures:
            return None
        if attempts < settings.max_retries:
            return self._retry(snapshot, failures, attempts)
        if settings.exhausted == "annotate":
            note = (
                "\n\n(Note: an automatic check could not confirm this answer: "
                + "; ".join(r.summary for r in failures)
                + ".)"
            )
            return {"messages": [reply.model_copy(update={"content": reply.text + note})]}
        return None

    def _retry(
        self, snapshot: Mapping[str, Mode], failures: list[CheckResult], attempts: int
    ) -> dict[str, object]:
        lines = "\n".join(f"- {r.summary}" for r in failures)
        feedback = HumanMessage(
            content=(
                "[Jev check] Your previous answer did not pass these checks:\n"
                f"{lines}\nRevise it using only what the sources support, and say so if they "
                "do not answer the question."
            ),
            additional_kwargs={JEV_MARK: VERIFY_ANSWER},
        )
        update: dict[str, object] = {
            "messages": [feedback],
            "jump_to": "model",
            "jev_attempts": attempts + 1,
        }
        escalation = mode_for(snapshot, ESCALATION)
        if escalation != "off":
            self.jev.emit(JevEvent(ESCALATION, escalation, "escalate", {"attempt": attempts + 1}))
        if escalation == "on":
            update["jev_escalate"] = True
        self.record("on", "retry", attempt=attempts + 1, failed=[r.blueprint for r in failures])
        return update

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Ask for checkable citations when configured."""
        return handler(self._instructed(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Ask for checkable citations when configured."""
        return await handler(self._instructed(request))

    def _instructed(self, request: ModelRequest) -> ModelRequest:
        if not self.jev.config.verify.add_citation_instructions:
            return request
        snapshot = request.state.get("jev_modes")
        if not isinstance(snapshot, Mapping) or mode_for(snapshot, CITATION_CHECK) == "off":
            return request
        return request.override(
            system_message=append_system_text(request.system_message, CITATION_INSTRUCTIONS)
        )


def parse_citations(answer: str) -> list[tuple[str, str, str]]:
    """Find `[cite: <source> | "<quote>"]` markers as (claim, source, quote)."""
    found: list[tuple[str, str, str]] = []
    position = 0
    for match in _CITATION.finditer(answer):
        before = answer[position : match.start()]
        claim = re.split(r"(?<=[.!?])\s+", before.strip())[-1] if before.strip() else ""
        found.append((claim, match.group("source").strip(), match.group("quote")))
        position = match.end()
    return found


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.translate(_QUOTES)).strip()


def _verdict(
    quote: str,
    text: str | None,
    answers: Iterator[ClassifierResponse | None],
    auto: float,
) -> dict[str, object]:
    """The cookbook's verdict: a missing quote is fabricated without asking Jev."""
    if text is None:
        return {
            "verdict": "unsupported",
            "confidence": None,
            "auto": True,
            "reason": "source not found",
        }
    if _normalize(quote) not in _normalize(text):
        return {"verdict": "fabricated", "confidence": None, "auto": True}
    response = next(answers, None)
    if response is None:
        return {"verdict": "unchecked", "confidence": None, "auto": False}
    relation = response.choices["relation"]
    return {
        "verdict": RELATION_TO_VERDICT[relation.choice],
        "confidence": relation.confidence,
        "auto": relation.confidence >= auto,
    }


def _evidence(state: JevState, tools: Sequence[str]) -> list[dict]:
    accepted = list(state.get("jev_evidence") or [])
    if accepted or not tools:
        return accepted
    return [
        {"id": m.tool_call_id, "title": m.name or "", "text": m.text}
        for m in current_turn(state["messages"])
        if isinstance(m, ToolMessage) and m.name in tools
    ]


def _source_lookup(
    state: JevState, fallback: Callable[[str], str | None] | None
) -> Callable[[str], str | None]:
    by_id = {str(p.get("id")): str(p.get("text", "")) for p in state.get("jev_evidence") or []}

    def lookup(source: str) -> str | None:
        if source in by_id:
            return by_id[source]
        return fallback(source) if fallback else None

    return lookup
