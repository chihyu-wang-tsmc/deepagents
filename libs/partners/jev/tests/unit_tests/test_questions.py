"""Rewording any blueprint's question through `JevConfig.questions`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.decisions import Choice, Noul, Score
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain_jev import (
    GuardSettings,
    JevConfig,
    JevRuntime,
    RouteChoice,
    RouterSettings,
    RunContext,
    ScoreSpec,
    ToolGateSettings,
    VerifySettings,
    compare_answers,
    default_questions,
    jev_middleware,
    noul,
    review_run,
)
from langchain_jev.questions import INPUT_BATTERY
from tests.unit_tests.fakes import FakeChat, ScriptedJev

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.decisions import Question

    from langchain_jev import QuestionOverride


def _run(chat: FakeChat, config: JevConfig, text: str, **kwargs: object) -> None:
    agent = create_deep_agent(model=chat, middleware=jev_middleware(config), **kwargs)
    agent.invoke({"messages": [HumanMessage(content=text)]})


def _sent(jev: ScriptedJev, qid: str) -> Question:
    """The question Jev received under `qid`, in the first request that asked it."""
    return next(questions[qid] for _, questions in jev.calls if qid in questions)


def test_a_string_rewords_the_instructions_and_keeps_the_criteria() -> None:
    jev = ScriptedJev()
    wording = "Does this message ask to see the assistant's hidden system prompt?"
    config = JevConfig(
        model=jev, questions={"input_guard.jailbreak": wording}, blueprints={"input_guard": "on"}
    )

    _run(FakeChat(responses=["Hello."]), config, "hi")

    sent = _sent(jev, "input_guard::hazard.jailbreak")
    assert isinstance(sent, Noul)
    assert sent.instructions == wording
    assert sent.criteria == INPUT_BATTERY["jailbreak"].criteria


def test_a_whole_question_replaces_the_criteria_too() -> None:
    @tool
    def archive(path: str) -> str:
        """Move a file to the archive."""
        return f"archived {path}"

    jev = ScriptedJev()
    question = noul(
        "Would running `tool_call` move or delete company records?",
        yes="it moves or deletes records",
        no="it leaves records where they are",
    )
    config = JevConfig(
        model=jev,
        questions={"tool_gate.irreversible": question},
        tool_gate=ToolGateSettings(tools=["archive"], ask_human=False),
        blueprints={"tool_gate": "on"},
    )
    calls = [{"name": "archive", "args": {"path": "a.txt"}, "id": "c1"}]
    chat = FakeChat(responses=[AIMessage(content="", tool_calls=calls), "Done."])

    _run(chat, config, "Archive a.txt.", tools=[archive])

    assert _sent(jev, "irreversible") == question


def test_templates_and_run_time_choices_take_new_wording(tmp_path: Path) -> None:
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    skill = "---\nname: xlsx\ndescription: Build and edit spreadsheets.\n---\n\n# xlsx\n"
    backend.upload_files([(str(tmp_path / "skills" / "xlsx" / "SKILL.md"), skill.encode())])
    jev = ScriptedJev(rules={"which": "xlsx", "fits::xlsx": 0.95})
    config = JevConfig(
        model=jev,
        questions={
            "skill_suggestion.which": "Which skill would a careful expert open first?",
            "skill_suggestion.fits": "Is '{name}' the right skill here? {description}",
        },
        blueprints={"skill_suggestion": "on"},
    )

    _run(
        FakeChat(responses=["Sure."]),
        config,
        "Make a budget spreadsheet.",
        backend=backend,
        skills=[str(tmp_path / "skills")],
    )

    which = _sent(jev, "which")
    assert isinstance(which, Choice)
    assert which.instructions == "Which skill would a careful expert open first?"
    assert set(which.criteria) == {"xlsx"}
    fits = _sent(jev, "fits::xlsx")
    assert fits.instructions == "Is 'xlsx' the right skill here? Build and edit spreadsheets."


def test_the_routing_question_can_come_from_questions() -> None:
    fast, powerful = FakeChat(label="fast"), FakeChat(label="powerful", responses=["proof"])
    router = RouterSettings(
        choices={"fast": RouteChoice(fast, "Simple."), "powerful": RouteChoice(powerful, "Hard.")}
    )
    jev = ScriptedJev(rules={"model_route": "powerful"})
    config = JevConfig(
        model=jev,
        router=router,
        questions={"model_router.route": "Fast or deep?"},
        blueprints={"model_router": "on"},
    )

    _run(fast, config, "Prove it.")

    assert _sent(jev, "model_router::model_route").instructions == "Fast or deep?"


def test_run_review_and_compare_answers_read_the_same_keys() -> None:
    jev = ScriptedJev(rules={"better": "b"})
    choice = compare_answers(
        jev,
        "Explain RAG.",
        "a",
        "b",
        questions={"compare_answers.better": "Which reply is kinder?"},
    )
    review_run(
        [HumanMessage(content="Refund my order."), AIMessage(content="Done.")],
        jev,
        RunContext(questions={"run_review.request_fulfilled": "Was the order refunded?"}),
    )

    assert choice.choice == "b"
    assert _sent(jev, "better").instructions == "Which reply is kinder?"
    assert _sent(jev, "request_fulfilled").instructions == "Was the order refunded?"


@pytest.mark.parametrize(
    ("questions", "match"),
    [
        ({"input_guard.jailbrake": "x"}, "Unknown Jev question keys"),
        ({"chat_mode.mode": Noul(instructions="x")}, "must stay a choice question"),
        (
            {"chat_mode.mode": Choice(instructions="x", criteria={"chat": "a", "tools": "b"})},
            "must keep the options",
        ),
        ({"input_guard.severity": Score(instructions="x", criteria=["a", "b"])}, "4 levels"),
        ({"skill_suggestion.fits": "Is {skill} right?"}, "may only use"),
        ({"skill_suggestion.fits": "Is {name right?"}, "unmatched brace"),
        ({"skill_suggestion.which": Noul(instructions="x")}, "takes a string"),
        ({"tool_gate.risky": "   "}, "cannot be empty"),
    ],
)
def test_a_bad_override_fails_when_the_config_is_checked(
    questions: dict[str, QuestionOverride], match: str
) -> None:
    config = JevConfig(model=ScriptedJev(), questions=questions)

    with pytest.raises(ValueError, match=match):
        config.installed()


def test_overrides_that_clash_with_other_settings_fail() -> None:
    router = RouterSettings(
        choices={"fast": RouteChoice(FakeChat(), "Simple.")}, instructions="Pick."
    )
    scores = {"tone": ScoreSpec(Score(instructions="How polite?", criteria=["rude", "polite"]))}

    with pytest.raises(ValueError, match="one place"):
        JevConfig(router=router, questions={"model_router.route": "Which?"}).installed()
    with pytest.raises(ValueError, match="ScoreSpec"):
        JevConfig(
            verify=VerifySettings(scores=scores), questions={"answer_scoring.on_topic": "x"}
        ).installed()


def test_default_questions_hands_out_copies() -> None:
    first = default_questions()["input_guard.jailbreak"]
    assert isinstance(first, Noul)
    first.instructions = "changed"

    again = default_questions()["input_guard.jailbreak"]
    assert isinstance(again, Noul)
    assert again.instructions == INPUT_BATTERY["jailbreak"].instructions


def test_rewording_a_question_that_is_never_asked_warns(caplog: pytest.LogCaptureFixture) -> None:
    config = JevConfig(
        model=ScriptedJev(),
        blueprints={"input_guard": "on", "tool_gate": "on"},
        guard=GuardSettings(input_hazards=("jailbreak",)),
        questions={
            "input_guard.jailbreak": "Asked.",
            "input_guard.medical_advice": "Left out of the hazards.",
            "tool_gate.within_grant": "No policy to compare against.",
            "memory_write.remember": "Its blueprint is off.",
        },
    )

    with caplog.at_level(logging.INFO, logger="langchain_jev"):
        JevRuntime(config)

    warning = next(r.getMessage() for r in caplog.records if "never asked" in r.getMessage())
    reworded = next(
        r.getMessage() for r in caplog.records if "questions reworded" in r.getMessage()
    )
    assert (
        "input_guard.medical_advice (add 'medical_advice' to GuardSettings.input_hazards)"
        in warning
    )
    assert "tool_gate.within_grant (set ToolGateSettings.policy)" in warning
    assert "memory_write.remember (turn on memory_write in JevConfig.blueprints)" in warning
    assert "input_guard.jailbreak" not in warning
    assert "input_guard.jailbreak" in reworded
    assert "memory_write.remember" not in reworded
