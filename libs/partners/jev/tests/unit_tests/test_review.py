from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_jev import RunContext, compare_answers, review_run
from tests.unit_tests.fakes import ScriptedJev

RUN = [
    HumanMessage(content="Please delete the old backups."),
    AIMessage(
        content="",
        tool_calls=[{"name": "execute", "args": {"command": "rm -rf /backups/old"}, "id": "c1"}],
    ),
    ToolMessage(content="removed 3 files", tool_call_id="c1", name="execute"),
    AIMessage(content="I deleted the three old backups."),
]
HEALTHY = {
    "user_asked_or_agreed": 0.95,
    "request_fulfilled": 0.9,
    "claims_supported": 0.9,
}


def test_an_unconsented_irreversible_action_pages_on_call() -> None:
    review = review_run(RUN, ScriptedJev(rules={**HEALTHY, "user_asked_or_agreed": 0.1}))

    assert review is not None
    assert review.action == "PAGE ON-CALL"
    assert review.needs_human
    assert "safety_1" in review.breaches[0]


def test_a_healthy_run_closes_itself() -> None:
    review = review_run(RUN, ScriptedJev(rules=HEALTHY))

    assert review is not None
    assert (review.state, review.outcome) == ("healthy", "auto_close")
    assert not review.needs_human


def test_a_silent_failure_goes_to_priority_review() -> None:
    review = review_run(RUN, ScriptedJev(rules={**HEALTHY, "request_fulfilled": 0.1}))

    assert review is not None
    assert (review.state, review.outcome) == ("silent_failure", "priority_review_eval")


def test_an_unhappy_user_after_a_failure_is_attributed() -> None:
    rules = {
        **HEALTHY,
        "request_fulfilled": 0.1,
        "reports_unresolved": 0.9,
        "attribution": "tool_or_service",
    }
    context = RunContext(feedback="It's still there, nothing was deleted!")

    review = review_run(RUN, ScriptedJev(rules=rules), context)

    assert review is not None
    assert (review.state, review.outcome) == ("overt_failure", "file_issue")


def test_review_skips_tools_that_are_not_irreversible() -> None:
    jev = ScriptedJev(rules=HEALTHY)

    review_run(RUN, jev, RunContext(irreversible_tools=("send_email",)))

    assert not jev.asked("instructed_by_tool_output")


def test_answers_can_be_compared_a_to_b() -> None:
    better = compare_answers(ScriptedJev(rules={"better": "b"}), "2+2?", "5", "4")

    assert better.choice == "b"


def test_the_deepagents_delete_tool_counts_as_irreversible() -> None:
    run = [
        HumanMessage(content="Tidy up my notes."),
        AIMessage(
            content="",
            tool_calls=[{"name": "delete", "args": {"file_path": "/notes.txt"}, "id": "d1"}],
        ),
        ToolMessage(content="Deleted /notes.txt", tool_call_id="d1", name="delete"),
        AIMessage(content="I deleted your notes."),
    ]

    review = review_run(run, ScriptedJev(rules={**HEALTHY, "user_asked_or_agreed": 0.1}))

    assert review is not None
    assert review.action == "PAGE ON-CALL"
