from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.decisions import Choice, Noul, Score
from pydantic import SecretStr

from langchain_jev import JevClassifierError, SystemOneClassifier

QUESTIONS = {
    "urgent": Noul(instructions="Is this urgent?"),
    "team": Choice(instructions="Which team?", criteria={"infra": "Deploys.", "billing": "Bills."}),
    "severity": Score(instructions="How bad?", criteria=["Cosmetic.", "Degraded.", "Outage."]),
}
SHIM_ANSWERS = {
    "urgent": {"type": "noul", "noul": 0.9},
    "team": {"type": "choice", "choice": "infra", "probabilities": {"infra": 0.8, "billing": 0.2}},
    "severity": {"type": "score", "score": 1.8, "probabilities": {"0": 0.1, "1": 0.0, "2": 0.9}},
}


def _server(seen: list[httpx.Request], statuses: list[int] | None = None) -> httpx.MockTransport:
    queue = list(statuses or [])

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        status = queue.pop(0) if queue else 200
        if status != 200:
            return httpx.Response(status, json={"error": "busy"})
        return httpx.Response(
            200,
            json={
                "answers": SHIM_ANSWERS,
                "usage": {"input_tokens": 12, "output_tokens": 3, "latency_ms": 9},
            },
        )

    return httpx.MockTransport(handle)


def test_shim_answers_are_completed_to_the_typesafe_shape() -> None:
    seen: list[httpx.Request] = []
    classifier = SystemOneClassifier(
        base_url="http://jev.local",
        model="h2oai/h2o-lightning-4b",
        send_model=False,
        transport=_server(seen),
    )

    response = classifier.invoke({"state": "Checkout is down.", "questions": QUESTIONS})

    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/systemone"
    assert "model" not in body
    assert "authorization" not in seen[0].headers
    assert response.model == "h2oai/h2o-lightning-4b"
    assert response.choices["team"].confidence == pytest.approx(0.6)
    assert response.scores["severity"].legend == {0: "Cosmetic.", 1: "Degraded.", 2: "Outage."}
    assert 0 < response.scores["severity"].confidence <= 1
    assert response.usage is not None
    assert response.usage["total_tokens"] == 15


def test_hosted_requests_send_the_model_and_key() -> None:
    seen: list[httpx.Request] = []
    classifier = SystemOneClassifier(
        model="jev-latest", api_key=SecretStr("sk-test"), transport=_server(seen)
    )

    classifier.invoke({"state": "hi", "questions": {"urgent": QUESTIONS["urgent"]}})

    assert json.loads(seen[0].content)["model"] == "jev-latest"
    assert seen[0].headers["authorization"] == "Bearer sk-test"


def test_transient_failures_are_retried() -> None:
    seen: list[httpx.Request] = []
    classifier = SystemOneClassifier(transport=_server(seen, [503, 429]), retry_backoff=0)

    response = classifier.invoke({"state": "hi", "questions": {"urgent": QUESTIONS["urgent"]}})

    assert len(seen) == 3
    assert response.nouls["urgent"].noul == 0.9


def test_a_rejected_request_is_not_retried() -> None:
    seen: list[httpx.Request] = []
    classifier = SystemOneClassifier(transport=_server(seen, [422]), retry_backoff=0)

    with pytest.raises(JevClassifierError, match="rejected"):
        classifier.invoke({"state": "hi", "questions": {"urgent": QUESTIONS["urgent"]}})
    assert len(seen) == 1


async def test_async_requests_use_the_async_transport() -> None:
    seen: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"answers": SHIM_ANSWERS})

    classifier = SystemOneClassifier(async_transport=httpx.MockTransport(handle))

    response = await classifier.ainvoke({"state": "hi", "questions": QUESTIONS})

    assert len(seen) == 1
    assert response.nouls["urgent"].noul == 0.9
