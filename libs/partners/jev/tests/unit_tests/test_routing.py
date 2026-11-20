"""Coding requests go to the coding model; fast and powerful get measured descriptions."""

from __future__ import annotations

import json

import pytest
from deepagents import create_deep_agent
from langchain_core.decisions import Choice
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain_jev import (
    CodingSettings,
    JevConfig,
    JevEvent,
    RouteChoice,
    RouterSettings,
    VerifySettings,
    jev_middleware,
)
from langchain_jev.questions import DEFAULT_ROUTE_CRITERIA
from tests.unit_tests.fakes import FakeChat, ScriptedJev


def _run(chat: FakeChat, config: JevConfig, text: str, **kwargs: object) -> dict:
    agent = create_deep_agent(model=chat, middleware=jev_middleware(config), **kwargs)
    return agent.invoke({"messages": [HumanMessage(content=text)]})


def _coding_config(coder: FakeChat, coding: float, mode: str = "on", **kwargs: object) -> JevConfig:
    return JevConfig(
        model=ScriptedJev(rules={"coding": coding}),
        coding=CodingSettings(model=coder),
        blueprints={"coding_router": mode},
        **kwargs,
    )


def test_a_request_that_needs_code_goes_to_the_coding_model() -> None:
    main, coder = FakeChat(label="main"), FakeChat(label="coder", responses=["def add(a, b): ..."])

    result = _run(main, _coding_config(coder, 0.9), "Write a function that adds two numbers.")

    assert result["messages"][-1].content == "def add(a, b): ..."
    assert main.calls == []


def test_other_requests_keep_the_agent_model() -> None:
    main, coder = FakeChat(label="main", responses=["Paris."]), FakeChat(label="coder")

    result = _run(main, _coding_config(coder, 0.1), "What is the capital of France?")

    assert result["messages"][-1].content == "Paris."
    assert coder.calls == []


def test_shadow_mode_records_the_switch_but_keeps_the_model() -> None:
    events: list[JevEvent] = []
    main, coder = FakeChat(label="main", responses=["def add(a, b): ..."]), FakeChat(label="coder")
    config = _coding_config(coder, 0.9, mode="shadow", on_event=events.append)

    _run(main, config, "Write a function that adds two numbers.")

    assert len(main.calls) == 1
    assert coder.calls == []
    assert ("shadow", "route:coding") in [(e.mode, e.action) for e in events]


def test_router_choices_named_fast_and_powerful_use_the_measured_descriptions() -> None:
    fast, powerful = FakeChat(label="fast", responses=["Hi."]), FakeChat(label="powerful")
    jev = ScriptedJev(rules={"model_route": "fast"})
    router = RouterSettings(choices={"fast": RouteChoice(fast), "powerful": RouteChoice(powerful)})
    config = JevConfig(model=jev, router=router, blueprints={"model_router": "on"})

    _run(fast, config, "Say hi.")

    qid = "model_router::model_route"
    sent = next(questions[qid] for _, questions in jev.calls if qid in questions)
    assert isinstance(sent, Choice)
    assert sent.criteria == DEFAULT_ROUTE_CRITERIA


def test_the_coding_model_beats_the_routed_model_but_not_escalation() -> None:
    @tool
    def search_docs(query: str) -> str:
        """Search the API docs."""
        return json.dumps([{"id": "p1", "text": "`add` returns the sum of its two arguments."}])

    calls = [{"name": "search_docs", "args": {"query": "add"}, "id": "s1"}]
    coder = FakeChat(
        label="coder",
        responses=[AIMessage(content="", tool_calls=calls), "`add` returns the difference."],
    )
    fast = FakeChat(label="fast")
    powerful = FakeChat(label="powerful", responses=["`add` returns the sum."])

    def supported(state: object, qid: str) -> float:
        answer = state.get("candidate_response", "") if isinstance(state, dict) else ""
        return 0.95 if "sum" in answer else 0.05

    config = JevConfig(
        model=ScriptedJev(rules={"coding": 0.9, "model_route": "fast", "supported": supported}),
        router=RouterSettings(
            choices={"fast": RouteChoice(fast), "powerful": RouteChoice(powerful)},
            escalate_to="powerful",
        ),
        coding=CodingSettings(model=coder),
        verify=VerifySettings(evidence_tools=["search_docs"]),
        blueprints={
            "model_router": "on",
            "coding_router": "on",
            "escalation": "on",
            "verify_answer": "on",
        },
        cache_size=0,
    )

    result = _run(fast, config, "What does the `add` function return?", tools=[search_docs])

    assert fast.calls == []
    assert len(coder.calls) == 2
    assert len(powerful.calls) == 1
    assert result["messages"][-1].content == "`add` returns the sum."


def test_missing_routing_settings_fail_when_the_config_is_checked() -> None:
    with pytest.raises(ValueError, match="coding"):
        JevConfig(blueprints={"coding_router": "on"}).installed()
    with pytest.raises(ValueError, match="needs `criteria` for cheap"):
        JevConfig(router=RouterSettings(choices={"cheap": RouteChoice("openai:x")})).installed()
