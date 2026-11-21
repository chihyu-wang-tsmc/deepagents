"""Check the final answer, and mark it when a check fails.

Three checks share one hook and run in parallel:

- `verify_answer` (https://docs.typesafe.ai/cookbooks/verify_llm_outputs): does the
    answer follow from the evidence the agent retrieved?
- `citation_check` (https://docs.typesafe.ai/cookbooks/citation_check): is each quote
    in its source, and does the source support the claim?
- `answer_scoring` (https://docs.typesafe.ai/patterns/composite-scoring): weighted
    Score questions about the answer's quality.

A failed check does not run the model again: retrying on the same model mostly
costs time. The reply is marked instead: `additional_kwargs["jev_checks"]` maps each
failed check to why it failed, and `VerifySettings.annotate` also adds a visible note.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, cast

from langchain_core.messages import AIMessage, ToolMessage

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
    VERIFY_ANSWER,
    JevEvent,
)
from langchain_jev.config import ScoreSpec
from langchain_jev.middleware._base import Ask, AskMany, Flow, JevMiddleware
from langchain_jev.middleware.guardrails import JEV_MARK
from langchain_jev.questions import (
    CHECK_QUESTIONS,
    CITATION_INSTRUCTIONS,
    DEFAULT_SCORES,
    RELATION_TO_VERDICT,
)
from langchain_jev.switches import mode_for

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain_core.decisions import ClassifierResponse, Score
    from langgraph.runtime import Runtime

    from langchain_jev._types import Mode
    from langchain_jev.state import JevState

_CITATION = re.compile(r'\[cite:\s*(?P<source>[^|\]]+?)\s*\|\s*"(?P<quote>[^"]+)"\s*\]')
_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})
"""Fold curly quotes so a quote still matches its source (from the citation cookbook)."""
CHECKS = (VERIFY_ANSWER, CITATION_CHECK, ANSWER_SCORING)
JEV_CHECKS = "jev_checks"
"""Key in the reply's `additional_kwargs`: each failed check, mapped to why it failed."""


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
    """Verify, citation-check, and score the final answer; mark the checks it fails."""

    blueprint = VERIFY_ANSWER

    def after_agent(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """Check synchronously."""
        return self.drive(self._flow(state, runtime))

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
        asks = tuple(
            replace(ask, blueprints=(check.blueprint,)) for check in checks for ask in check.asks
        )
        responses = cast("list[ClassifierResponse | None]", (yield AskMany(asks)) if asks else [])
        results = self._evaluate(checks, responses, modes)
        return self._mark(reply, results, modes)

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
        paragraphs, left_out = _fit_evidence(evidence, answer, self.jev.max_chars())
        ask = Ask(
            {
                "request": request,
                "paragraphs": paragraphs,
                "candidate_response": answer,
            },
            {key: self.question(key) for key in CHECK_QUESTIONS},
        )
        limit = self.threshold("verify_answer.review_at")

        def evaluate(responses: list[ClassifierResponse | None]) -> CheckResult | None:
            response = responses[0]
            if response is None:
                return None
            nouls = {key: response.nouls[key].noul for key in CHECK_QUESTIONS}
            risk = nouls["answerable"] if nouls["declines"] >= 0.5 else 1 - nouls["supported"]  # noqa: PLR2004 - the cookbook's split
            summary = f"the answer may not follow from the retrieved sources (risk {risk:.2f})"
            if left_out:
                total = len(evidence)
                summary += f"; {left_out} of {total} passages were left out to fit the length limit"
            details = {**nouls, "risk": risk, "passages": len(evidence), "left_out": left_out}
            return CheckResult(VERIFY_ANSWER, risk >= limit, summary, details)

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
                {"relation": self.jev.questions.question("citation_check.relation")},
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
        book = self.jev.questions
        specs = self.jev.config.verify.scores or {
            key: ScoreSpec(cast("Score", book.question(f"answer_scoring.{key}")))
            for key in DEFAULT_SCORES
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

    def _mark(
        self, reply: AIMessage, results: list[CheckResult], modes: Mapping[str, Mode]
    ) -> dict[str, object] | None:
        """Record the failed checks on the reply itself; the model is not asked again."""
        failed = {
            r.blueprint: r.summary for r in results if r.failed and modes[r.blueprint] == "on"
        }
        if not failed:
            return None
        for blueprint, summary in failed.items():
            self.jev.emit(JevEvent(blueprint, "on", "mark", {"summary": summary}))
        update: dict[str, object] = {
            "additional_kwargs": {**reply.additional_kwargs, JEV_CHECKS: failed}
        }
        if self.jev.config.verify.annotate:
            note = "; ".join(failed.values())
            update["content"] = (
                f"{reply.text}\n\n(Note: an automatic check could not confirm this answer: {note}.)"
            )
        return {"messages": [reply.model_copy(update=update)]}

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


def _block(passage: dict) -> str:
    return f"[{passage['id']}] {passage.get('title', '')}\n{passage['text']}"


def _fit_evidence(evidence: list[dict], answer: str, limit: int) -> tuple[str, int]:
    """Fit the evidence into `limit` characters, and count the passages left out.

    The evidence of a run with several retrievals can be longer than one Jev state
    field. Cutting the joined text would drop the newest passages, which the answer
    most likely uses. Passages the answer cites go in first, then the newest ones;
    whatever fits is shown in retrieval order. A passage longer than the whole limit
    is clipped.
    """
    cited = {source for _, source, _ in parse_citations(answer)}
    cited |= {str(p["id"]) for p in evidence if f"[{p['id']}]" in answer}
    first = [i for i, p in enumerate(evidence) if str(p["id"]) in cited]
    chosen = set(first)
    order = first + [i for i in reversed(range(len(evidence))) if i not in chosen]
    kept: list[int] = []
    used = 0
    for index in order:
        cost = len(_block(evidence[index])) + (2 if kept else 0)  # the blank line between
        if used + cost <= limit:
            kept.append(index)
            used += cost
    if not kept:
        return clip(_block(evidence[order[0]]), limit), len(evidence) - 1
    return "\n\n".join(_block(evidence[i]) for i in sorted(kept)), len(evidence) - len(kept)


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
