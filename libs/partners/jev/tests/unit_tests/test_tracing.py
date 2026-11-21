"""Jev requests reach the run's tracing callbacks, including ones sent in parallel."""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from langchain_jev import InMemoryJevMemory, JevConfig, MemorySettings, jev_middleware
from tests.unit_tests.fakes import FakeChat, ScriptedJev


class _RunNames(BaseCallbackHandler):
    def __init__(self) -> None:
        self.names: list[str] = []

    def on_chain_start(self, serialized: Any, inputs: Any, **kwargs: Any) -> None:
        self.names.append(kwargs.get("name") or "")


def test_parallel_jev_requests_are_traced() -> None:
    store = InMemoryJevMemory()
    for fact in ("We self-host on Windows Server.", "Our plan is Enterprise."):
        store.save(("jev_memory", "u-1"), {"message": fact})
    config = JevConfig(
        model=ScriptedJev(rules={"relevant": 0.9}),
        memory=MemorySettings(store=store),
        blueprints={"memory_read": "on"},
    )
    agent = create_deep_agent(model=FakeChat(responses=["ok"]), middleware=jev_middleware(config))
    tracer = _RunNames()

    agent.invoke(
        {"messages": [HumanMessage(content="Dashboards are slow.")]},
        config={"callbacks": [tracer]},
        context={"user_id": "u-1"},
    )

    assert tracer.names.count("ScriptedJev") == 2  # one relevance request per saved fact
