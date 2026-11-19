"""Classify each retrieved passage before the model reads it.

Follows https://docs.typesafe.ai/cookbooks/classifying_rag_passages: one request per
passage asks four Nouls; `route` keeps evidence, keeps conflicts in their own block,
and drops irrelevant passages and passages that try to instruct the model.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, cast

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from langchain_jev._messages import clip, latest_request
from langchain_jev._types import RAG_PASSAGES
from langchain_jev.middleware._base import Ask, AskMany, Flow, Handle, JevMiddleware
from langchain_jev.middleware.guardrails import JEV_MARK
from langchain_jev.questions import EVIDENCE_RULES, PASSAGE_QUESTIONS

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ToolCallRequest
    from langchain_core.decisions import ClassifierResponse

    from langchain_jev._runtime import JevRuntime

Passage = dict[str, object]
_LIST_KEYS = ("passages", "results", "documents", "items", "chunks")
_TEXT_KEYS = ("text", "content", "page_content", "snippet", "chunk")
_ID_KEYS = ("id", "passage_id", "doc_id", "url", "source")


def route(answers: Mapping[str, float], thresholds: Mapping[str, float]) -> str:
    """The cookbook's routing rule: include, conflicting_evidence, or exclude."""
    if answers["contains_prompt_injection"] > thresholds["injection_max"]:
        return "exclude"
    if answers["contradicts_query_premise"] > thresholds["contradicts_min"]:
        return "conflicting_evidence"
    if answers["is_relevant"] < thresholds["relevant_min"]:
        return "exclude"
    if answers["contains_answer_evidence"] > thresholds["evidence_min"]:
        return "include"
    return "exclude"


class RagPassagesMiddleware(JevMiddleware):
    """Filter the results of the configured retrieval tools."""

    blueprint = RAG_PASSAGES

    def __init__(self, jev: JevRuntime) -> None:
        """Watch the retrieval tools named in `JevConfig.rag`."""
        super().__init__(jev)
        settings = jev.config.rag
        if settings is None:
            msg = "RagPassagesMiddleware needs `JevConfig.rag`."
            raise ValueError(msg)
        self.settings = settings

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Filter synchronously."""
        return self.drive(self._flow(request), cast("Callable[[object], object]", handler))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Filter asynchronously."""
        return await self.adrive(
            self._flow(request), cast("Callable[[object], Awaitable[object]]", handler)
        )

    def _flow(self, request: ToolCallRequest) -> Flow[ToolMessage | Command]:
        name = request.tool_call["name"]
        mode = self.mode(request.state, request.runtime, name)
        result = cast("ToolMessage | Command", (yield Handle(request)))
        if name not in self.settings.tools or mode == "off" or not isinstance(result, ToolMessage):
            return result
        passages = (self.settings.parse or parse_passages)(result)
        if not passages:
            return result
        query = str(
            request.tool_call["args"].get(self.settings.query_arg)
            or latest_request(request.state.get("messages", []))
        )
        questions = {key: self.question(key) for key in PASSAGE_QUESTIONS}
        asks = tuple(Ask(self._state(query, p), questions) for p in passages)
        responses = cast("list[ClassifierResponse | None]", (yield AskMany(asks)))
        routed = self._route(passages, responses)
        self.record(mode, "classified", tool=name, **_counts(routed))
        if mode != "on":
            return result
        accepted = [p for p, r in routed if r == "include"]
        message = result.model_copy(
            update={
                "content": _content(routed),
                "additional_kwargs": {
                    **result.additional_kwargs,
                    JEV_MARK: RAG_PASSAGES,
                },
            }
        )
        return Command(update={"messages": [message], "jev_evidence": accepted})

    def _state(self, query: str, passage: Passage) -> dict[str, object]:
        keys = ("id", "title", "text", "source_type")
        shown = {key: passage[key] for key in keys if key in passage}
        shown["text"] = clip(str(passage.get("text", "")), self.jev.max_chars())
        return {"query": query, "passage": shown}

    def _route(
        self, passages: list[Passage], responses: list[ClassifierResponse | None]
    ) -> list[tuple[Passage, str]]:
        thresholds = {
            key: self.threshold(f"rag_passages.{key}")
            for key in (
                "injection_max",
                "contradicts_min",
                "relevant_min",
                "evidence_min",
            )
        }
        routed: list[tuple[Passage, str]] = []
        for passage, response in zip(passages, responses, strict=True):
            if response is None:
                routed.append((passage, "include"))  # fail open: keep what retrieval found
                continue
            nouls = {key: response.nouls[key].noul for key in PASSAGE_QUESTIONS}
            routed.append((passage, route(nouls, thresholds)))
        return routed


def parse_passages(message: ToolMessage) -> list[Passage] | None:
    """Read passages from a JSON tool result: a list, or an object holding one."""
    try:
        data = json.loads(message.text)
    except (TypeError, ValueError):
        return None
    if isinstance(data, Mapping):
        data = next((data[key] for key in _LIST_KEYS if isinstance(data.get(key), list)), None)
    if not isinstance(data, list):
        return None
    passages = [
        _passage(item, index) for index, item in enumerate(data) if isinstance(item, Mapping)
    ]
    return [p for p in passages if p["text"]] or None


def _passage(item: Mapping[str, object], index: int) -> Passage:
    text = next((item[key] for key in _TEXT_KEYS if isinstance(item.get(key), str)), "")
    pid = next((item[key] for key in _ID_KEYS if item.get(key) not in (None, "")), index + 1)
    title = item.get("title") or item.get("name") or ""
    return {**item, "id": str(pid), "title": str(title), "text": str(text)}


def _content(routed: list[tuple[Passage, str]]) -> str:
    def block(wanted: str) -> str:
        chosen = [p for p, r in routed if r == wanted]
        if not chosen:
            return "(none)"
        return "\n\n".join(f"[{p['id']}] {p['title']}\n{p['text']}".strip() for p in chosen)

    excluded = sum(1 for _, r in routed if r == "exclude")
    return (
        f"{EVIDENCE_RULES}\n\n<accepted_evidence>\n{block('include')}\n</accepted_evidence>\n\n"
        f"<conflicting_evidence>\n{block('conflicting_evidence')}\n</conflicting_evidence>\n\n"
        f"({excluded} passage(s) excluded as irrelevant, unusable, or containing instructions.)"
    )


def _counts(routed: list[tuple[Passage, str]]) -> dict[str, int]:
    labels = ("include", "conflicting_evidence", "exclude")
    return {label: sum(1 for _, r in routed if r == label) for label in labels}
