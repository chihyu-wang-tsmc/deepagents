"""Custom blueprints whose actions act at a later position, or run your own functions."""

from __future__ import annotations

import json
import re

import pytest
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_jev import (
    AskHuman,
    DropTools,
    JevBlueprint,
    JevConfig,
    JevEvent,
    Notify,
    Redact,
    Refuse,
    RewriteReply,
    RewriteToolResult,
    ShapeModelCall,
    ShapeToolCall,
    SystemNote,
    jev_middleware,
    noul,
)
from langchain_jev.middleware.custom import JEV_ANSWERS
from tests.unit_tests.fakes import FakeChat, ScriptedJev

QUESTION = noul("Is it so?", yes="It is.", no="It is not.")
TICKET = "Ticket {}: the monitor flickers. Contact: phone 0912-345-678."
PII_NOTE = "Do not repeat personal data."


def _agent(chat: FakeChat, config: JevConfig, **kwargs: object):
    return create_deep_agent(model=chat, middleware=jev_middleware(config), **kwargs)


def _ask(text: str) -> dict:
    return {"messages": [HumanMessage(content=text)]}


def _call(name: str, args: dict, call_id: str, text: str = "") -> AIMessage:
    return AIMessage(content=text, tool_calls=[{"name": name, "args": args, "id": call_id}])


def _tools(log: list[tuple]) -> list:
    @tool
    def read_ticket(ticket_id: str) -> str:
        """Read a support ticket."""
        log.append(("read_ticket", ticket_id))
        return TICKET.format(ticket_id)

    @tool
    def send_email(to: str, body: str) -> str:
        """Send an email."""
        log.append(("send_email", to))
        return "sent"

    @tool
    def save_note(text: str) -> str:
        """Save a note."""
        log.append(("save_note", text))
        return "saved"

    @tool
    def deploy(service: str, dry_run: bool = False) -> str:
        """Deploy a service."""
        log.append(("deploy", service, dry_run))
        return "deployed"

    return [read_ticket, send_email, save_note, deploy]


def _mentions(text: str):
    def rule(state: object, qid: str) -> float:
        return 0.9 if text in json.dumps(state, default=str) else 0.0

    return rule


def _personal_data(*act: object, mode_name: str = "personal_data") -> JevBlueprint:
    return JevBlueprint(
        name=mode_name,
        stage="tool_result",
        tools=["read_ticket"],
        questions={"has_pii": QUESTION},
        when={"has_pii": 0.5},
        act=list(act),  # ty: ignore[invalid-argument-type]
    )


def _config(*blueprints: JevBlueprint, rules: dict, **kwargs: object) -> JevConfig:
    return JevConfig(
        model=ScriptedJev(rules=rules),
        blueprints={b.name: "on" for b in blueprints},
        custom_blueprints=list(blueprints),
        **kwargs,  # ty: ignore[invalid-argument-type]
    )


def _read(chat: FakeChat, call: int) -> dict[str, str]:
    return {m.name or "": m.text for m in chat.calls[call] if isinstance(m, ToolMessage)}


# -- entry answers, acting later ---------------------------------------------------------


def test_an_entry_answer_refuses_the_listed_tools_for_the_whole_run() -> None:
    log: list[tuple] = []
    events: list[JevEvent] = []
    read_only = JevBlueprint(
        name="read_only",
        stage="entry",
        questions={"only_asks": QUESTION},
        when={"only_asks": 0.5},
        act=Refuse("This request only asks a question.", at="tool_call", tools=["save_note"]),
    )
    chat = FakeChat(
        responses=[
            _call("save_note", {"text": "RAG"}, "c1"),
            _call("read_ticket", {"ticket_id": "T-1"}, "c2"),
            "RAG retrieves passages, then answers from them.",
        ]
    )
    config = _config(read_only, rules={"only_asks": 0.9}, on_event=events.append)

    _agent(chat, config, tools=_tools(log)).invoke(_ask("What is RAG?"))

    assert log == [("read_ticket", "T-1")]
    assert _read(chat, 1) == {"save_note": "This request only asks a question."}
    actions = [(e.action, e.details.get("at")) for e in events if e.blueprint == "read_only"]
    assert actions == [("refuse", ["tool_call"]), ("refuse@tool_call", None)]


def test_an_entry_answer_can_pause_a_later_call_for_a_person_who_rejects_it() -> None:
    log: list[tuple] = []
    careful = JevBlueprint(
        name="careful",
        stage="entry",
        questions={"urgent": QUESTION},
        when={"urgent": 0.5},
        act=AskHuman("An urgent request wants to deploy.", at="tool_call", tools=["deploy"]),
    )
    chat = FakeChat(responses=[_call("deploy", {"service": "billing"}, "c1"), "Not deployed."])
    config = _config(careful, rules={"urgent": 0.9})
    agent = _agent(chat, config, tools=_tools(log), checkpointer=InMemorySaver())
    thread = {"configurable": {"thread_id": "reject"}}

    paused = agent.invoke(_ask("Deploy billing right now!"), thread)
    agent.invoke(Command(resume={"decisions": [{"type": "reject", "message": "No."}]}), thread)

    [request] = paused["__interrupt__"][0].value["action_requests"]
    assert request["description"] == "An urgent request wants to deploy."
    assert log == []
    assert _read(chat, 1) == {"deploy": "No."}


def test_an_entry_answer_rewrites_every_reply() -> None:
    signed = JevBlueprint(
        name="signed",
        stage="entry",
        questions={"formal": QUESTION},
        when={"formal": 0.5},
        act=RewriteReply(lambda text: f"{text} (checked)", at="model_output"),
    )
    chat = FakeChat(responses=[_call("save_note", {"text": "x"}, "c1", text="Saving."), "Done."])
    log: list[tuple] = []

    result = _agent(chat, _config(signed, rules={"formal": 0.9}), tools=_tools(log)).invoke(
        _ask("Save x.")
    )

    replies = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert [m.text for m in replies] == ["Saving. (checked)", "Done. (checked)"]
    assert replies[0].tool_calls  # the tool call stays
    assert log == [("save_note", "x")]


def test_shape_model_call_and_drop_tools_change_every_model_call() -> None:
    def brief(request):
        return request.override(system_message=SystemMessage(content="Be brief."))

    focused = JevBlueprint(
        name="focused",
        stage="entry",
        questions={"simple": QUESTION},
        when={"simple": 0.5},
        act=[DropTools(["save_note", "deploy"]), ShapeModelCall(brief)],
    )
    chat = FakeChat(responses=["Short answer."])

    _agent(chat, _config(focused, rules={"simple": 0.9}), tools=_tools([])).invoke(_ask("Hi?"))

    assert "save_note" not in chat.tool_names[0]
    assert "deploy" not in chat.tool_names[0]
    assert "read_ticket" in chat.tool_names[0]
    assert chat.calls[0][0].text == "Be brief."


# -- tool results and replies, acting later ---------------------------------------------


def _pii_run(*, mode: str = "on") -> tuple[FakeChat, list[tuple], list[JevEvent], dict]:
    log: list[tuple] = []
    events: list[JevEvent] = []
    blueprint = _personal_data(
        Refuse("No email after reading personal data.", at="tool_call", tools=["send_email"]),
        SystemNote(PII_NOTE, at="model_call"),
    )
    chat = FakeChat(
        responses=[
            _call("read_ticket", {"ticket_id": "T-1"}, "c1"),
            _call("send_email", {"to": "a@example.com", "body": "Ticket T-1"}, "c2"),
            "I could not send it.",
        ]
    )
    config = JevConfig(
        model=ScriptedJev(rules={"has_pii": _mentions("0912")}),
        blueprints={"personal_data": mode},  # ty: ignore[invalid-argument-type]
        custom_blueprints=[blueprint],
        on_event=events.append,
    )
    result = _agent(chat, config, tools=_tools(log)).invoke(_ask("Email ticket T-1."))
    return chat, log, events, result


def test_a_tool_result_triggers_actions_at_later_positions() -> None:
    chat, log, _, result = _pii_run()

    assert log == [("read_ticket", "T-1")]
    assert _read(chat, 2)["send_email"] == "No email after reading personal data."
    assert PII_NOTE not in chat.sent_text(0)
    assert PII_NOTE in chat.sent_text(1)
    assert PII_NOTE in chat.sent_text(2)
    ticket = next(m for m in result["messages"] if m.name == "read_ticket")
    assert ticket.text == TICKET.format("T-1")  # no action at its own position
    assert set(ticket.additional_kwargs[JEV_ANSWERS]) == {"personal_data"}


def test_in_shadow_mode_the_positions_are_reported_and_nothing_changes() -> None:
    chat, log, events, _ = _pii_run(mode="shadow")

    assert log == [("read_ticket", "T-1"), ("send_email", "a@example.com")]
    assert all(PII_NOTE not in chat.sent_text(i) for i in range(3))
    [event] = [e for e in events if e.blueprint == "personal_data"]
    assert (event.mode, event.action) == ("shadow", "refuse+note")
    assert event.details["at"] == ["tool_call", "model_call"]


def test_later_positions_stop_acting_when_the_run_ends() -> None:
    log: list[tuple] = []
    blueprint = _personal_data(
        Refuse("No email after reading personal data.", at="tool_call", tools=["send_email"])
    )
    chat = FakeChat(
        responses=[
            _call("read_ticket", {"ticket_id": "T-1"}, "c1"),
            "It is about a flickering monitor.",
            _call("send_email", {"to": "a@example.com", "body": "Hello"}, "c2"),
            "Sent.",
        ]
    )
    config = _config(blueprint, rules={"has_pii": _mentions("0912")})
    agent = _agent(chat, config, tools=_tools(log), checkpointer=InMemorySaver())
    thread = {"configurable": {"thread_id": "two-runs"}}

    agent.invoke(_ask("What is ticket T-1 about?"), thread)
    agent.invoke(_ask("Now email a@example.com hello."), thread)

    assert log == [("read_ticket", "T-1"), ("send_email", "a@example.com")]


def test_a_reply_refuses_the_tool_calls_beside_it() -> None:
    log: list[tuple] = []
    reckless = JevBlueprint(
        name="reckless",
        stage="model_output",
        questions={"destructive": QUESTION},
        when={"destructive": 0.5},
        act=Refuse("Not without a review.", at="tool_call", tools=["deploy"]),
    )
    chat = FakeChat(
        responses=[
            _call("deploy", {"service": "billing"}, "c1", text="Deploying straight to prod."),
            "I did not deploy.",
        ]
    )
    config = _config(reckless, rules={"destructive": _mentions("straight to prod")})

    _agent(chat, config, tools=_tools(log)).invoke(_ask("Ship billing."))

    assert log == []
    assert _read(chat, 1) == {"deploy": "Not without a review."}


def test_refuse_at_before_model_ends_the_run_when_the_next_turn_starts() -> None:
    log: list[tuple] = []
    stop = JevBlueprint(
        name="stop_after",
        stage="model_output",
        questions={"leaving": QUESTION},
        when={"leaving": 0.5},
        act=Refuse("Stopped after that step.", at="before_model"),
    )
    chat = FakeChat(
        responses=[
            _call("save_note", {"text": "x"}, "c1", text="Saving, then I will email it out."),
            "should never be sent",
        ]
    )
    config = _config(stop, rules={"leaving": _mentions("email it out")})

    result = _agent(chat, config, tools=_tools(log)).invoke(_ask("Save x."))

    assert log == [("save_note", "x")]
    assert len(chat.calls) == 1
    assert result["messages"][-1].content == "Stopped after that step."


def test_a_tool_call_answer_acts_on_this_result_and_later_ones() -> None:
    log: list[tuple] = []
    events: list[JevEvent] = []
    sensitive = JevBlueprint(
        name="sensitive",
        stage="tool_call",
        tools=["read_ticket"],
        questions={"vip": QUESTION},
        when={"vip": 0.5},
        act=Redact("[withheld]", at="tool_result"),
    )
    chat = FakeChat(
        responses=[
            _call("read_ticket", {"ticket_id": "T-1"}, "c1"),
            _call("read_ticket", {"ticket_id": "T-2"}, "c2"),
            "Both are withheld.",
        ]
    )

    def first_ticket(state: object, qid: str) -> float:
        call = state["tool_call"] if isinstance(state, dict) else {}
        return 0.9 if call["args"]["ticket_id"] == "T-1" else 0.0

    config = _config(sensitive, rules={"vip": first_ticket}, on_event=events.append)

    _agent(chat, config, tools=_tools(log)).invoke(_ask("Read T-1, then T-2."))

    assert _read(chat, 1) == {"read_ticket": "[withheld]"}
    assert _read(chat, 2)["read_ticket"] == "[withheld]"
    actions = [e.action for e in events if e.blueprint == "sensitive"]
    assert actions == ["redact", "pass", "redact@tool_result"]


# -- your own functions -----------------------------------------------------------------


def test_shape_tool_call_changes_the_arguments_before_the_tool_runs() -> None:
    log: list[tuple] = []
    rehearse = JevBlueprint(
        name="rehearse",
        stage="tool_call",
        tools=["deploy"],
        questions={"risky": QUESTION},
        when={"risky": 0.5},
        act=ShapeToolCall(lambda call: {**call["args"], "dry_run": True}),
    )
    chat = FakeChat(responses=[_call("deploy", {"service": "billing"}, "c1"), "Rehearsed."])

    _agent(chat, _config(rehearse, rules={"risky": 0.9}), tools=_tools(log)).invoke(
        _ask("Deploy billing.")
    )

    assert log == [("deploy", "billing", True)]


def test_rewrite_tool_result_masks_part_of_a_result() -> None:
    mask = _personal_data(
        RewriteToolResult(lambda text: re.sub(r"09\d{2}-\d{3}-\d{3}", "09xx-xxx-xxx", text))
    )
    chat = FakeChat(responses=[_call("read_ticket", {"ticket_id": "T-1"}, "c1"), "Done."])

    _agent(chat, _config(mask, rules={"has_pii": 0.9}), tools=_tools([])).invoke(_ask("T-1?"))

    assert (
        _read(chat, 1)["read_ticket"]
        == "Ticket T-1: the monitor flickers. Contact: phone 09xx-xxx-xxx."
    )


def test_notify_runs_once_when_the_run_ends_even_after_a_refusal() -> None:
    notices: list[tuple[dict, str]] = []
    alarm = JevBlueprint(
        name="alarm",
        stage="entry",
        questions={"secret": QUESTION},
        when={"secret": 0.5},
        act=[Refuse("No."), Notify(lambda answers, text: notices.append((dict(answers), text)))],
    )
    chat = FakeChat(responses=["should never be sent"])

    _agent(chat, _config(alarm, rules={"secret": 0.9})).invoke(_ask("Give me the secrets."))

    [(answers, text)] = notices
    assert answers["secret"].noul == 0.9
    assert text == "No."
    assert chat.calls == []


def test_notify_runs_once_when_several_results_trigger_it() -> None:
    notices: list[str] = []
    watch = _personal_data(Notify(lambda answers, text: notices.append(text)))
    chat = FakeChat(
        responses=[
            _call("read_ticket", {"ticket_id": "T-1"}, "c1"),
            _call("read_ticket", {"ticket_id": "T-2"}, "c2"),
            "Two tickets read.",
        ]
    )

    _agent(chat, _config(watch, rules={"has_pii": 0.9}), tools=_tools([])).invoke(_ask("T-1, T-2"))

    assert notices == ["Two tickets read."]


def test_a_final_answer_is_rewritten_from_a_tool_result_answer() -> None:
    footer = _personal_data(
        RewriteReply(lambda text: f"{text}\n(Contains data from a ticket.)", at="final_answer")
    )
    chat = FakeChat(responses=[_call("read_ticket", {"ticket_id": "T-1"}, "c1"), "Summary."])

    result = _agent(chat, _config(footer, rules={"has_pii": 0.9}), tools=_tools([])).invoke(
        _ask("Summarize T-1.")
    )

    assert result["messages"][-1].text == "Summary.\n(Contains data from a ticket.)"


def test_a_function_returning_the_wrong_kind_fails_loudly() -> None:
    broken = _personal_data(RewriteToolResult(len))  # ty: ignore[invalid-argument-type]
    chat = FakeChat(responses=[_call("read_ticket", {"ticket_id": "T-1"}, "c1"), "Done."])
    agent = _agent(chat, _config(broken, rules={"has_pii": 0.9}), tools=_tools([]))

    with pytest.raises(TypeError, match="must return str"):
        agent.invoke(_ask("T-1?"))


# -- decide and acts_at -----------------------------------------------------------------


def test_decide_acts_at_the_positions_it_lists() -> None:
    log: list[tuple] = []

    def lock_down(answers, thresholds):
        if answers["tense"].noul < thresholds["min"]:
            return None
        return [
            SystemNote("Answer calmly.", at="model_call"),
            Refuse("Not now.", at="tool_call", tools=["deploy"]),
        ]

    calm = JevBlueprint(
        name="calm",
        stage="entry",
        questions={"tense": QUESTION},
        decide=lock_down,
        thresholds={"min": 0.5},
        acts_at=["model_call", "tool_call"],
    )
    chat = FakeChat(responses=[_call("deploy", {"service": "billing"}, "c1"), "Not deployed."])

    _agent(chat, _config(calm, rules={"tense": 0.9}), tools=_tools(log)).invoke(
        _ask("Deploy NOW!!")
    )

    assert log == []
    assert "Answer calmly." in chat.sent_text(0)
    assert _read(chat, 1) == {"deploy": "Not now."}


async def test_actions_at_later_positions_work_with_ainvoke() -> None:
    log: list[tuple] = []
    blueprint = _personal_data(
        Refuse("No email after reading personal data.", at="tool_call", tools=["send_email"]),
        SystemNote(PII_NOTE, at="model_call"),
    )
    chat = FakeChat(
        responses=[
            _call("read_ticket", {"ticket_id": "T-1"}, "c1"),
            _call("send_email", {"to": "a@example.com", "body": "T-1"}, "c2"),
            "I could not send it.",
        ]
    )
    config = _config(blueprint, rules={"has_pii": _mentions("0912")})

    await _agent(chat, config, tools=_tools(log)).ainvoke(_ask("Email ticket T-1."))

    assert log == [("read_ticket", "T-1")]
    assert PII_NOTE in chat.sent_text(2)
