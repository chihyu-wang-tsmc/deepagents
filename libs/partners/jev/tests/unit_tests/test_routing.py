"""Coding requests go to the coding model; fast and powerful get measured descriptions."""

from __future__ import annotations

import pytest
from deepagents import create_deep_agent
from langchain_core.decisions import Choice
from langchain_core.messages import HumanMessage

from langchain_jev import (
    CodingSettings,
    JevConfig,
    JevEvent,
    RouteChoice,
    RouterSettings,
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


def test_the_coding_model_beats_the_routed_model() -> None:
    coder = FakeChat(label="coder", responses=["def add(a, b): return a + b"])
    fast = FakeChat(label="fast")
    config = JevConfig(
        model=ScriptedJev(rules={"coding": 0.9, "model_route": "fast"}),
        router=RouterSettings(
            choices={"fast": RouteChoice(fast), "powerful": RouteChoice(FakeChat())}
        ),
        coding=CodingSettings(model=coder),
        blueprints={"model_router": "on", "coding_router": "on"},
    )

    result = _run(fast, config, "Write a Python function that adds two numbers.")

    assert fast.calls == []
    assert len(coder.calls) == 1
    assert result["messages"][-1].content == "def add(a, b): return a + b"


def test_missing_routing_settings_fail_when_the_config_is_checked() -> None:
    with pytest.raises(ValueError, match="coding"):
        JevConfig(blueprints={"coding_router": "on"}).installed()
    with pytest.raises(ValueError, match="needs `criteria` for cheap"):
        JevConfig(router=RouterSettings(choices={"cheap": RouteChoice("openai:x")})).installed()
