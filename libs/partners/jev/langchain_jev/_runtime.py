"""What every Jev middleware in one stack shares: models, thresholds, cache, events."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, cast

from langchain_jev._question_use import asked_overrides, unasked_questions
from langchain_jev._types import (
    BLUEPRINTS,
    CITATION_CHECK,
    RAG_PASSAGES,
    VERIFY_ANSWER,
    JevEvent,
    Mode,
)
from langchain_jev.classifiers import serialize_state
from langchain_jev.models import JevModel, as_jev_model
from langchain_jev.question_book import QuestionBook
from langchain_jev.switches import context_modes, resolve_modes
from langchain_jev.thresholds import (
    GUARD_POLICIES,
    clamped_thresholds,
    merge_thresholds,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from langchain_core.decisions import BaseDecisionModel, ClassifierResponse, Question
    from langchain_core.decisions.types import ClassifierRequest, State

    from langchain_jev.config import JevConfig

logger = logging.getLogger("langchain_jev")

_DISAGREEMENT = 0.2
"""How far a shadow model's Noul may drift from the main model's before it is reported."""


class JevRuntime:
    """Built once per `JevConfig` and shared by every middleware it produces."""

    def __init__(self, config: JevConfig) -> None:
        """Validate `config` and build its decision models."""
        self.config = config
        self.installed: dict[str, Mode] = config.installed()
        self.model: JevModel = as_jev_model(config.model)
        self._default = self.model.factory()
        self._overrides = {bp: as_jev_model(spec) for bp, spec in config.models.items()}
        self._built: dict[str, BaseDecisionModel] = {}
        self._shadow = as_jev_model(config.shadow_model).factory() if config.shadow_model else None
        self.thresholds = self._thresholds()
        router = config.router
        self.questions = QuestionBook(
            config.questions, route_instructions=router.instructions if router else None
        )
        self._cache: OrderedDict[str, ClassifierResponse] = OrderedDict()
        self._lock = threading.Lock()
        self._report_setup()

    def _report_setup(self) -> None:
        if self.installed:
            self._warn_clamped()
            self._warn_missing_sources()
        else:
            logger.warning(
                "No Jev blueprint is on, so Jev is never asked. Every blueprint is off "
                "until JevConfig(blueprints={...}) turns it on, for example "
                "blueprints={'input_guard': 'on', 'tool_gate': 'on'}."
            )
        self._report_questions()

    def _warn_missing_sources(self) -> None:
        """Answer checks that would check nothing, or fail every citation, without a source."""
        verify = self.config.verify
        rag = RAG_PASSAGES in self.installed
        if VERIFY_ANSWER in self.installed and not rag and not verify.evidence_tools:
            logger.warning(
                "verify_answer has no evidence to check answers against, so it never checks "
                "one. Turn on rag_passages or set VerifySettings(evidence_tools=[...])."
            )
        if CITATION_CHECK in self.installed and not rag and verify.sources is None:
            logger.warning(
                "citation_check cannot look up cited sources, so it marks every citation "
                "unsupported. Turn on rag_passages or set VerifySettings(sources=...)."
            )

    def _report_questions(self) -> None:
        unasked = unasked_questions(self.config, self.installed)
        if unasked:
            logger.warning(
                "Reworded Jev questions that are never asked: %s.",
                "; ".join(f"{key} ({fix})" for key, fix in unasked.items()),
            )
        asked = asked_overrides(self.config, unasked)
        if asked:
            logger.info(
                "Jev questions reworded: %s. Thresholds were tuned on the default "
                "wording; check them on your own data.",
                ", ".join(asked),
            )

    def _thresholds(self) -> dict[str, float]:
        policy = GUARD_POLICIES[self.config.guard.policy]
        guards = {
            f"{side}.{name}": value
            for side in ("input_guard", "output_guard")
            for name, value in policy.items()
        }
        # Later layers win: a model's re-tuned values beat the policy's starting points.
        return merge_thresholds(guards, self.model.thresholds, self.config.thresholds)

    def _warn_clamped(self) -> None:
        if not self.model.noul_floor:
            return
        affected = [
            name
            for name in clamped_thresholds(self.thresholds, self.model.noul_floor)
            if name.split(".", 1)[0] in self.installed
        ]
        if affected:
            logger.warning(
                "Jev model %r clamps Noul probabilities at %.3f, so these thresholds "
                "act like 0.5: %s. Set the server's noul_floor to 0 to keep them.",
                self.model.name,
                self.model.noul_floor,
                ", ".join(affected),
            )

    def classifier(self, blueprint: str) -> BaseDecisionModel:
        """The decision model a blueprint asks."""
        if blueprint not in self._overrides:
            return self._default
        with self._lock:
            if blueprint not in self._built:
                self._built[blueprint] = self._overrides[blueprint].factory()
            return self._built[blueprint]

    def max_chars(self) -> int:
        """Longest text to put in one state field."""
        return self.model.max_state_chars

    def snapshot(self, context: object) -> dict[str, Mode]:
        """Blueprint modes for a run starting now."""
        source = self.config.mode_source
        hot = source.modes() if source is not None else {}
        return resolve_modes(self.installed, hot, context_modes(context, self.config.context_key))

    def ask(
        self, blueprint: str, state: object, questions: Mapping[str, Question]
    ) -> ClassifierResponse:
        """Ask synchronously, from the cache when the same request was asked before."""
        classifier = self.classifier(blueprint)
        key = _cache_key(classifier, state, questions)
        cached = self._cached(key)
        if cached is not None:
            return cached
        response = classifier.invoke(_request(state, questions))
        self._store(key, response)
        if self._shadow is not None:
            self._compare(blueprint, response, self._shadow_sync(state, questions))
        return response

    async def aask(
        self, blueprint: str, state: object, questions: Mapping[str, Question]
    ) -> ClassifierResponse:
        """Ask asynchronously, from the cache when the same request was asked before."""
        classifier = self.classifier(blueprint)
        key = _cache_key(classifier, state, questions)
        cached = self._cached(key)
        if cached is not None:
            return cached
        request = _request(state, questions)
        if self._shadow is None:
            response = await classifier.ainvoke(request)
        else:
            response, shadow = await asyncio.gather(
                classifier.ainvoke(request), self._shadow_async(state, questions)
            )
            self._compare(blueprint, response, shadow)
        self._store(key, response)
        return response

    def _cached(self, key: str) -> ClassifierResponse | None:
        with self._lock:
            response = self._cache.get(key)
            if response is not None:
                self._cache.move_to_end(key)
            return response

    def _store(self, key: str, response: ClassifierResponse) -> None:
        if self.config.cache_size <= 0:
            return
        with self._lock:
            self._cache[key] = response
            while len(self._cache) > self.config.cache_size:
                self._cache.popitem(last=False)

    def _shadow_sync(
        self, state: object, questions: Mapping[str, Question]
    ) -> ClassifierResponse | None:
        try:
            return self._shadow.invoke(_request(state, questions)) if self._shadow else None
        except Exception:  # noqa: BLE001 - a failing candidate model must not affect the run
            logger.debug("Shadow Jev model failed.", exc_info=True)
            return None

    async def _shadow_async(
        self, state: object, questions: Mapping[str, Question]
    ) -> ClassifierResponse | None:
        if self._shadow is None:
            return None
        try:
            return await self._shadow.ainvoke(_request(state, questions))
        except Exception:  # noqa: BLE001 - a failing candidate model must not affect the run
            logger.debug("Shadow Jev model failed.", exc_info=True)
            return None

    def _compare(
        self,
        blueprint: str,
        main: ClassifierResponse,
        shadow: ClassifierResponse | None,
    ) -> None:
        if shadow is None:
            return
        by_owner: dict[str, list[str]] = {}
        for name in _differences(main, shadow):
            by_owner.setdefault(_owner(name, blueprint), []).append(name)
        for owner, names in by_owner.items():
            self.emit(JevEvent(owner, "shadow", "model_disagreement", {"questions": names}))

    def emit(self, event: JevEvent) -> None:
        """Log a decision and pass it to `JevConfig.on_event`."""
        level = logging.DEBUG if event.action in {"pass", "allow", "skip"} else logging.INFO
        logger.log(
            level,
            "jev %s [%s] %s %s",
            event.blueprint,
            event.mode,
            event.action,
            event.details,
        )
        callback = self.config.on_event
        if callback is None:
            return
        try:
            callback(event)
        except Exception:
            logger.exception("Jev on_event callback failed.")


def _request(state: object, questions: Mapping[str, Question]) -> ClassifierRequest:
    """A decision-model request; `serialize_state` rejects state with no JSON form."""
    return {"state": cast("State", state), "questions": dict(questions)}


def _cache_key(
    classifier: BaseDecisionModel, state: object, questions: Mapping[str, Question]
) -> str:
    payload = {
        "model": classifier.model,
        "class": type(classifier).__qualname__,
        "state": serialize_state(state),
        "questions": {
            k: q.model_dump(mode="json", exclude_none=True) for k, q in questions.items()
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _owner(question: str, asker: str) -> str:
    """The blueprint a question belongs to: entry questions are keyed `blueprint::qid`."""
    head, sep, _ = question.partition("::")
    return head if sep and head in BLUEPRINTS else asker


def _differences(main: ClassifierResponse, shadow: ClassifierResponse) -> list[str]:
    differing: list[str] = []
    for name, answer in main.answers.items():
        other = shadow.answers.get(name)
        if (
            other is None
            or other.type != answer.type
            or (answer.type == "choice" and answer.choice != getattr(other, "choice", None))
            or (
                answer.type == "noul"
                and abs(answer.noul - getattr(other, "noul", answer.noul)) > _DISAGREEMENT
            )
        ):
            differing.append(name)
    return differing
