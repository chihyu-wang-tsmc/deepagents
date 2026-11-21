"""Model swapping, blueprint switches, and configuration checks."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

import pytest

from langchain_jev import (
    EnvModeSource,
    FileModeSource,
    GuardSettings,
    JevConfig,
    JevModel,
    JevRuntime,
    RagSettings,
    RouteChoice,
    RouterSettings,
    StaticModeSource,
    VerifySettings,
    get_jev_model,
    jev_middleware,
    register_jev_model,
)
from langchain_jev.switches import mode_for, resolve_modes
from tests.unit_tests.fakes import FakeChat, ScriptedJev

if TYPE_CHECKING:
    from pathlib import Path


def test_every_blueprint_is_off_until_turned_on(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="langchain_jev"):
        stack = jev_middleware(JevConfig(model=ScriptedJev()))
    listed = JevConfig(blueprints={"input_guard": "on", "tool_gate": "shadow"}).installed()

    assert stack == []
    assert "No Jev blueprint is on" in caplog.text
    assert listed == {"input_guard": "on", "tool_gate": "shadow"}


def test_default_mode_on_turns_on_every_blueprint_that_needs_no_settings() -> None:
    installed = JevConfig(model=ScriptedJev(), default_mode="on").installed()

    assert "input_guard" in installed
    assert "tool_gate" in installed
    assert "chat_mode" not in installed  # opt-in
    assert "model_router" not in installed  # needs router settings
    assert "rag_passages" not in installed  # needs retrieval tools


def test_turning_on_a_blueprint_without_its_settings_is_an_error() -> None:
    with pytest.raises(ValueError, match="model_router"):
        JevConfig(blueprints={"model_router": "on"}).installed()


def test_unknown_blueprints_and_thresholds_are_errors() -> None:
    with pytest.raises(ValueError, match="Unknown Jev blueprints"):
        JevConfig(blueprints={"input_gaurd": "off"}).installed()
    with pytest.raises(ValueError, match="Unknown Jev thresholds"):
        JevRuntime(JevConfig(model=ScriptedJev(), thresholds={"input_guard.actoin": 0.9}))


def test_removed_blueprints_are_named_in_the_error() -> None:
    with pytest.raises(ValueError, match=r"memory_read was removed"):
        JevConfig(blueprints={"memory_read": "on"}).installed()
    with pytest.raises(ValueError, match=r"escalation was removed"):
        JevConfig(blueprints={"escalation": "on"}).installed()


def test_removed_hazards_are_named_in_the_error() -> None:
    hazards = GuardSettings(input_hazards=("jailbreak", "self_harm"))
    with pytest.raises(ValueError, match=r"self_harm was removed"):
        JevConfig(guard=hazards).installed()
    with pytest.raises(ValueError, match=r"output_guard\.medical_advice was removed"):
        JevRuntime(
            JevConfig(model=ScriptedJev(), questions={"output_guard.medical_advice": "Reworded."})
        )
    with pytest.raises(ValueError, match=r"fast_path was removed"):
        JevRuntime(JevConfig(model=ScriptedJev(), thresholds={"fast_path.min_fits": 0.7}))
    with pytest.raises(ValueError, match=r"memory_write was removed"):
        JevRuntime(JevConfig(model=ScriptedJev(), questions={"memory_write.remember": "Reworded."}))


def test_answer_checks_without_a_source_warn(caplog: pytest.LogCaptureFixture) -> None:
    config = JevConfig(
        model=ScriptedJev(), blueprints={"verify_answer": "on", "citation_check": "on"}
    )

    with caplog.at_level(logging.WARNING, logger="langchain_jev"):
        JevRuntime(config)

    messages = [r.getMessage() for r in caplog.records]
    assert any("verify_answer has no evidence" in m for m in messages)
    assert any("citation_check cannot look up cited sources" in m for m in messages)


def test_answer_checks_with_a_source_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    checks = {"verify_answer": "on", "citation_check": "on"}
    with_rag = JevConfig(
        model=ScriptedJev(),
        rag=RagSettings(tools=["search_docs"]),
        blueprints={**checks, "rag_passages": "on"},
    )
    with_settings = JevConfig(
        model=ScriptedJev(),
        verify=VerifySettings(evidence_tools=["search_docs"], sources=lambda source_id: None),
        blueprints=checks,
    )

    with caplog.at_level(logging.WARNING, logger="langchain_jev"):
        JevRuntime(with_rag)
        JevRuntime(with_settings)

    messages = [r.getMessage() for r in caplog.records]
    assert not any("has no evidence" in m or "cannot look up" in m for m in messages)


def test_a_dependent_follows_its_dependency_unless_asked_for() -> None:
    off = JevConfig(default_mode="on", blueprints={"skill_suggestion": "off"}).installed()
    assert "skill_pinning" not in off
    with pytest.raises(ValueError, match="skill_pinning"):
        JevConfig(blueprints={"skill_suggestion": "off", "skill_pinning": "on"}).installed()


def test_router_choices_must_exist() -> None:
    router = RouterSettings(
        choices={"fast": RouteChoice(FakeChat(), "Simple.")}, fallback="powerful"
    )
    with pytest.raises(ValueError, match="fallback"):
        JevConfig(router=router).installed()


def test_models_are_swapped_by_name_with_their_own_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = ScriptedJev(model="better-jev")
    register_jev_model(
        JevModel(
            name="better-jev", factory=lambda: candidate, thresholds={"input_guard.action": 0.9}
        ),
        replace=True,
    )
    monkeypatch.setenv("JEV_MODEL", "better-jev")

    jev = JevRuntime(JevConfig())

    assert get_jev_model().name == "better-jev"
    assert jev.classifier("input_guard") is candidate
    assert jev.thresholds["input_guard.action"] == 0.9


def test_one_blueprint_can_use_its_own_model() -> None:
    guard, default = ScriptedJev(model="guard"), ScriptedJev(model="default")
    jev = JevRuntime(JevConfig(model=default, models={"input_guard": guard}))

    assert jev.classifier("input_guard") is guard
    assert jev.classifier("tool_gate") is default


def test_a_clamping_server_warns_about_affected_thresholds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clamped = JevModel(name="clamped", factory=ScriptedJev, noul_floor=0.801)

    with caplog.at_level(logging.WARNING, logger="langchain_jev"):
        JevRuntime(JevConfig(model=clamped, blueprints={"input_guard": "on"}))

    assert "input_guard.action" in caplog.text
    assert "rag_passages.relevant_min" not in caplog.text  # rag_passages is off


def test_more_specific_mode_layers_win_and_cannot_add_blueprints() -> None:
    installed = {"input_guard": "on", "tool_gate": "on"}

    snapshot = resolve_modes(
        installed,
        {"tool_gate": "shadow", "rag_passages": "on"},
        {"tool_gate": "off"},
    )

    assert snapshot == {"input_guard": "on", "tool_gate": "off"}


def test_a_part_can_narrow_but_not_widen_its_blueprint() -> None:
    snapshot = {
        "input_guard": "shadow",
        "input_guard.harmful_request": "on",
        "input_guard.drop_safeguards": "off",
    }

    assert mode_for(snapshot, "input_guard", "harmful_request") == "shadow"
    assert mode_for(snapshot, "input_guard", "drop_safeguards") == "off"
    assert mode_for(snapshot, "input_guard", "jailbreak") == "shadow"


def test_env_source_ignores_bad_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "JEV_MODES", "tool_gate=off, rag_passages=shadow, nonsense=off, input_guard=maybe"
    )

    assert EnvModeSource().modes() == {"tool_gate": "off", "rag_passages": "shadow"}


def test_file_source_reloads_and_keeps_the_last_good_modes(tmp_path: Path) -> None:
    path = tmp_path / "modes.json"
    source = FileModeSource(path, ttl=0)
    path.write_text(json.dumps({"tool_gate": "off"}), encoding="utf-8")
    first = source.modes()
    path.write_text("{not json", encoding="utf-8")
    os.utime(path, (1, 1))

    assert first == {"tool_gate": "off"}
    assert source.modes() == {"tool_gate": "off"}


def test_static_source_flips_blueprints_from_code() -> None:
    source = StaticModeSource()
    source.set("tool_gate", "shadow")
    jev = JevRuntime(
        JevConfig(model=ScriptedJev(), mode_source=source, blueprints={"tool_gate": "on"})
    )

    assert jev.snapshot(None)["tool_gate"] == "shadow"
    assert jev.snapshot({"jev_modes": {"tool_gate": "on"}})["tool_gate"] == "on"
