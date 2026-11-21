"""Jev requests reach the run's tracing callbacks, including ones sent in parallel."""

from __future__ import annotations

import json
from typing import Any

from deepagents import create_deep_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain_jev import JevConfig, RagSettings, jev_middleware
from tests.unit_tests.fakes import FakeChat, ScriptedJev


class _RunNames(BaseCallbackHandler):
    def __init__(self) -> None:
        self.names: list[str] = []

    def on_chain_start(self, serialized: Any, inputs: Any, **kwargs: Any) -> None:
        self.names.append(kwargs.get("name") or "")


def test_parallel_jev_requests_are_traced() -> None:
    @tool
    def search_docs(query: str) -> str:
        """Search the auth docs."""
        return json.dumps(
            [
                {"id": "p1", "title": "Rotation", "text": "Refresh tokens rotate on every use."},
                {"id": "p2", "title": "Expiry", "text": "Access tokens expire after an hour."},
            ]
        )

    chat = FakeChat(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "search_docs", "args": {"query": "tokens"}, "id": "s1"}],
            ),
            "Refresh tokens rotate on every use.",
        ]
    )
    config = JevConfig(
        model=ScriptedJev(rules={"is_relevant": 0.9, "contains_answer_evidence": 0.9}),
        rag=RagSettings(tools=["search_docs"]),
        blueprints={"rag_passages": "on"},
    )
    agent = create_deep_agent(model=chat, tools=[search_docs], middleware=jev_middleware(config))
    tracer = _RunNames()

    agent.invoke(
        {"messages": [HumanMessage(content="How do tokens work?")]},
        config={"callbacks": [tracer]},
    )

    assert tracer.names.count("ScriptedJev") == 2  # one classification request per passage
