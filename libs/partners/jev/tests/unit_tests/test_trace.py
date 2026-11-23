"""`JevTrace`: what each Jev request read, asked, and got back, and what was decided."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage

from langchain_jev import (
    JevBlueprint,
    JevConfig,
    JevRequest,
    JevTrace,
    Refuse,
    jev_middleware,
    noul,
)
from langchain_jev.questions import SKILL_FITS
from tests.unit_tests.fakes import FakeChat, ScriptedJev

if TYPE_CHECKING:
    from pathlib import Path

ENTRY = {"input_guard": "on", "chat_mode": "on"}


def _run(config: JevConfig, text: str = "Delete /tmp/a.txt", *, times: int = 1) -> None:
    agent = create_deep_agent(
        model=FakeChat(responses=["done"] * times), middleware=jev_middleware(config)
    )
    for _ in range(times):
        agent.invoke({"messages": [HumanMessage(content=text)]})


def _traced(lines: list[str], **settings: object) -> JevConfig:
    return JevConfig(
        model=ScriptedJev(rules={"jailbreak": 0.12, "mode": "agent"}),
        blueprints=ENTRY,
        trace=JevTrace(echo=lines.append),
        **settings,  # ty: ignore[invalid-argument-type]
    )


def test_each_request_is_printed_with_its_answers_and_thresholds() -> None:
    lines: list[str] = []

    _run(_traced(lines))

    assert lines[0] == "[jev request] entry: input_guard, chat_mode · scripted-jev"
    assert lines[1] == '  read: message = "Delete /tmp/a.txt"'
    shown = {line.split()[0]: line.split(maxsplit=1)[1] for line in lines[2:7]}
    assert shown["input_guard.jailbreak"] == "yes 0.12"
    assert shown["input_guard.severity"] == "score 3 of 3"
    assert shown["chat_mode.mode"] == "agent 0.90 | chat 0.10"
    [thresholds] = [line for line in lines if line.startswith("  thresholds:")]
    assert "input_guard.action 0.7" in thresholds
    assert "chat_mode.min_probability 0.8" in thresholds
    assert "[jev decision] input_guard on pass" in lines
    assert "[jev decision] chat_mode on agent" in lines


def test_full_form_adds_each_question_and_the_whole_state() -> None:
    lines: list[str] = []
    config = JevConfig(
        model=ScriptedJev(),
        blueprints={"chat_mode": "on"},
        trace=JevTrace(full=True, echo=lines.append),
    )

    _run(config)

    assert lines[1:4] == ["  read:", "    {", '      "message": "Delete /tmp/a.txt"']
    wording = lines[lines.index(next(line for line in lines if "chat_mode.mode" in line)) + 1]
    assert wording.strip().startswith("Should this request be handled in chat mode or agent mode?")


def test_the_thresholds_shown_are_the_ones_in_use() -> None:
    lines: list[str] = []

    _run(_traced(lines, thresholds={"input_guard.action": 0.9}))

    [thresholds] = [line for line in lines if line.startswith("  thresholds:")]
    assert "input_guard.action 0.9" in thresholds


def test_decisions_can_be_left_to_on_event() -> None:
    lines: list[str] = []
    config = JevConfig(
        model=ScriptedJev(),
        blueprints=ENTRY,
        trace=JevTrace(echo=lines.append, decisions=False),
    )

    _run(config)

    assert lines
    assert not [line for line in lines if line.startswith("[jev decision]")]


def test_requests_and_decisions_are_written_to_a_jsonl_file(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    config = JevConfig(
        model=ScriptedJev(rules={"jailbreak": 0.12}),
        blueprints=ENTRY,
        trace=JevTrace(path=path, echo=None),
    )

    _run(config)

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    request = records[0]
    assert request["type"] == "request"
    assert request["blueprints"] == ["input_guard", "chat_mode"]
    assert request["state"] == {"message": "Delete /tmp/a.txt"}
    assert request["answers"]["input_guard::hazard.jailbreak"]["noul"] == 0.12
    assert (
        "Does this message" in request["questions"]["input_guard::hazard.jailbreak"]["instructions"]
    )
    assert request["thresholds"]["input_guard.review"] == 0.35
    decisions = [(r["blueprint"], r["action"]) for r in records if r["type"] == "decision"]
    assert ("input_guard", "pass") in decisions


def test_failed_requests_are_traced() -> None:
    lines: list[str] = []
    config = JevConfig(
        model=ScriptedJev(fail=True),
        blueprints={"chat_mode": "on"},
        trace=JevTrace(echo=lines.append),
    )

    _run(config)

    assert "  failed: RuntimeError: Jev is down" in lines
    assert "[jev decision] chat_mode on error" in lines


def test_answers_from_the_cache_are_traced_as_cached() -> None:
    lines: list[str] = []
    config = JevConfig(
        model=ScriptedJev(), blueprints={"chat_mode": "on"}, trace=JevTrace(echo=lines.append)
    )

    _run(config, times=2)

    headers = [line for line in lines if line.startswith("[jev request]")]
    assert headers == [
        "[jev request] entry: chat_mode · scripted-jev",
        "[jev request] entry: chat_mode · scripted-jev · cached",
    ]


async def test_requests_are_traced_with_ainvoke() -> None:
    lines: list[str] = []
    agent = create_deep_agent(
        model=FakeChat(responses=["done"]), middleware=jev_middleware(_traced(lines))
    )

    await agent.ainvoke({"messages": [HumanMessage(content="Delete /tmp/a.txt")]})

    assert lines[0] == "[jev request] entry: input_guard, chat_mode · scripted-jev"


def test_a_custom_blueprint_is_traced_with_its_thresholds() -> None:
    lines: list[str] = []
    confidential = JevBlueprint(
        name="confidential_request",
        stage="entry",
        questions={"asks": noul("Is it confidential?", yes="It is.", no="It is not.")},
        when={"asks": 0.7},
        act=Refuse("No."),
    )
    config = JevConfig(
        model=ScriptedJev(rules={"asks": 0.4}),
        blueprints={"confidential_request": "on"},
        custom_blueprints=[confidential],
        trace=JevTrace(echo=lines.append),
    )

    _run(config)

    assert lines[0] == "[jev request] entry: confidential_request · scripted-jev"
    assert "confidential_request.asks" in lines[2]
    assert lines[3] == "  thresholds: confidential_request.asks 0.7"


def test_ids_with_double_colons_outside_the_entry_request_keep_their_blueprint() -> None:
    lines: list[str] = []
    request = JevRequest(
        blueprints=("skill_suggestion",),
        model="jev",
        state={"message": "Make a spreadsheet."},
        questions={"fits::xlsx": SKILL_FITS},
        answers=None,
        thresholds={},
        error="RuntimeError: down",
    )

    JevTrace(echo=lines.append).request(request)

    assert lines[0] == "[jev request] skill_suggestion · jev"
    assert lines[3].split() == ["skill_suggestion.fits::xlsx", "—"]


def test_a_failing_trace_never_breaks_the_run(caplog: pytest.LogCaptureFixture) -> None:
    def broken(line: str) -> None:
        msg = "the terminal went away"
        raise OSError(msg)

    chat = FakeChat(responses=["done"])
    config = JevConfig(model=ScriptedJev(), blueprints=ENTRY, trace=JevTrace(echo=broken))
    agent = create_deep_agent(model=chat, middleware=jev_middleware(config))

    with caplog.at_level(logging.ERROR, logger="langchain_jev"):
        result = agent.invoke({"messages": [HumanMessage(content="hi")]})

    assert result["messages"][-1].content == "done"
    assert "Jev trace failed." in caplog.text


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        ({}, None),
        ({"JEV_TRACE": "0"}, None),
        ({"JEV_TRACE": "1"}, (False, None, True)),
        ({"JEV_TRACE": "full"}, (True, None, True)),
        ({"JEV_TRACE_FILE": "trace.jsonl"}, (False, "trace.jsonl", False)),
    ],
)
def test_from_env_reads_jev_trace_and_jev_trace_file(
    monkeypatch: pytest.MonkeyPatch, variables: dict[str, str], expected: tuple | None
) -> None:
    for name in ("JEV_TRACE", "JEV_TRACE_FILE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in variables.items():
        monkeypatch.setenv(name, value)

    trace = JevTrace.from_env()

    if expected is None:
        assert trace is None
        return
    full, path, prints = expected
    assert trace is not None
    assert trace.full is full
    assert (str(trace.path) if trace.path else None) == path
    assert (trace.echo is not None) is prints
