"""A `BaseDecisionModel` for any server that speaks the System One API.

TypeSafe's hosted API (`https://api.typesafe.ai`), the h2o-lightning shim, and other
self-hosted Jev-compatible models all accept `POST /v1/systemone` with a state and
typed questions. One class talks to all of them; swapping models means changing its
`base_url` and `model`, not the middleware that uses it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import httpx
from langchain_core.decisions import BaseDecisionModel, ClassifierResponse, Question
from langchain_core.messages import BaseMessage, convert_to_openai_messages
from pydantic import ConfigDict, Field, JsonValue, PrivateAttr, SecretStr

if TYPE_CHECKING:
    from langchain_core.decisions.types import State

RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
"""HTTP statuses worth retrying: rate limits and a server that is still warming up."""


class JevClassifierError(RuntimeError):
    """The decision server could not answer the request."""


class SystemOneClassifier(BaseDecisionModel):
    """Ask a System One server typed questions.

    Example:
        ```python
        from langchain_jev import SystemOneClassifier

        # A self-hosted h2o-lightning shim: no key, and the server picks the model.
        local = SystemOneClassifier(
            base_url="http://localhost:8741",
            model="h2oai/h2o-lightning-4b",
            send_model=False,
        )
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "jev-latest"
    """Model name sent to the server, and reported to LangSmith."""

    base_url: str = "https://api.typesafe.ai"
    """Server root; requests go to `/v1/systemone` beneath it."""

    api_key: SecretStr | None = None
    """Bearer token, when the server requires one."""

    send_model: bool = True
    """Whether to send `model`; self-hosted shims that serve one model ignore it."""

    provider: str = "systemone"
    """Provider name reported to LangSmith as `ls_provider`."""

    timeout: float = Field(default=60.0, gt=0)
    """Seconds to wait for one HTTP response."""

    max_retries: int = Field(default=2, ge=0)
    """Extra attempts after a transient failure (see `RETRY_STATUSES`)."""

    retry_backoff: float = Field(default=0.5, ge=0)
    """Seconds before the first retry; each later retry waits twice as long."""

    transport: httpx.BaseTransport | None = Field(default=None, exclude=True)
    """Optional transport for the synchronous client, for example in tests."""

    async_transport: httpx.AsyncBaseTransport | None = Field(default=None, exclude=True)
    """Optional transport for the asynchronous client, for example in tests."""

    _client: httpx.Client | None = PrivateAttr(default=None)

    @property
    def _provider(self) -> str:
        return self.provider

    @property
    def endpoint(self) -> str:
        """The full URL requests are posted to."""
        return f"{self.base_url.rstrip('/')}/v1/systemone"

    def _classify(self, state: State, questions: dict[str, Question]) -> ClassifierResponse:
        payload = self._payload(state, questions)
        for attempt in range(self.max_retries + 1):
            response = self._post(payload)
            if response is not None and response.status_code not in RETRY_STATUSES:
                return _parse(response, payload, self.model)
            if attempt < self.max_retries:
                time.sleep(self.retry_backoff * 2**attempt)
        raise _unavailable(self.endpoint, response)

    async def _aclassify(self, state: State, questions: dict[str, Question]) -> ClassifierResponse:
        payload = self._payload(state, questions)
        async with httpx.AsyncClient(
            transport=self.async_transport, timeout=self.timeout
        ) as client:
            for attempt in range(self.max_retries + 1):
                response = await _apost(client, self.endpoint, payload, self._headers())
                if response is not None and response.status_code not in RETRY_STATUSES:
                    return _parse(response, payload, self.model)
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_backoff * 2**attempt)
        raise _unavailable(self.endpoint, response)

    def close(self) -> None:
        """Close the pooled synchronous HTTP client, if one was opened."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def _payload(self, state: State, questions: Mapping[str, Question]) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "state": serialize_state(state),
            "questions": {
                name: question.model_dump(mode="json", exclude_none=True)
                for name, question in questions.items()
            },
        }
        if self.send_model:
            payload["model"] = self.model
        return payload

    def _headers(self) -> dict[str, str]:
        if self.api_key is None or not self.api_key.get_secret_value():
            return {}
        return {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}

    def _post(self, payload: dict[str, JsonValue]) -> httpx.Response | None:
        if self._client is None:
            self._client = httpx.Client(transport=self.transport, timeout=self.timeout)
        try:
            return self._client.post(self.endpoint, json=payload, headers=self._headers())
        except httpx.TransportError:
            return None


async def _apost(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: dict[str, JsonValue],
    headers: dict[str, str],
) -> httpx.Response | None:
    try:
        return await client.post(endpoint, json=payload, headers=headers)
    except httpx.TransportError:
        return None


def _unavailable(endpoint: str, response: httpx.Response | None) -> JevClassifierError:
    reason = "no response" if response is None else f"HTTP {response.status_code}"
    msg = f"Jev server at {endpoint} is unavailable ({reason})."
    return JevClassifierError(msg)


def _parse(
    response: httpx.Response, payload: dict[str, JsonValue], model: str
) -> ClassifierResponse:
    if not response.is_success:
        msg = f"Jev server rejected the request (HTTP {response.status_code})."
        raise JevClassifierError(msg)
    data = response.json()
    questions = cast("dict[str, dict[str, Any]]", payload["questions"])
    answers = {
        name: normalize_answer(answer, questions.get(name, {}))
        for name, answer in (data.get("answers") or {}).items()
    }
    return ClassifierResponse.model_validate(
        {
            "model": data.get("model") or model,
            "answers": answers,
            "usage": _usage(data.get("usage")),
            "request_id": data.get("request_id") or response.headers.get("TypeSafe-Request-Id"),
        }
    )


def _usage(usage: object) -> dict[str, int] | None:
    if not isinstance(usage, Mapping):
        return None
    return {
        key: int(usage[key])
        for key in ("input_tokens", "output_tokens", "total_tokens")
        if isinstance(usage.get(key), int | float)
    }


def normalize_answer(answer: Mapping[str, Any], question: Mapping[str, Any]) -> dict[str, Any]:
    """Fill fields some servers leave out, using the formulas in Typesafe's docs.

    The h2o-lightning shim, for example, may omit `legend` on Score answers or report
    a different confidence. Confidence follows https://docs.typesafe.ai/confidence,
    so different servers' answers compare on the same scale.
    """
    kind = answer.get("type")
    if kind == "choice":
        return _normalize_choice(answer)
    if kind == "score":
        return _normalize_score(answer, question)
    return dict(answer)


def _normalize_choice(answer: Mapping[str, Any]) -> dict[str, Any]:
    probabilities = {str(k): float(v) for k, v in answer["probabilities"].items()}
    choice = answer.get("choice") or max(probabilities, key=probabilities.__getitem__)
    confidence = answer.get("confidence")
    if confidence is None:
        confidence = choice_confidence(list(probabilities.values()))
    return {
        **answer,
        "choice": choice,
        "probabilities": probabilities,
        "confidence": confidence,
    }


def _normalize_score(answer: Mapping[str, Any], question: Mapping[str, Any]) -> dict[str, Any]:
    probabilities = {int(k): float(v) for k, v in answer["probabilities"].items()}
    ordered = [probabilities.get(level, 0.0) for level in range(len(probabilities))]
    legend = answer.get("legend") or dict(enumerate(question.get("criteria") or []))
    score = answer.get("score")
    if score is None:
        score = sum(level * p for level, p in probabilities.items())
    confidence = answer.get("confidence")
    if confidence is None:
        confidence = score_confidence(ordered)
    return {
        **answer,
        "score": score,
        "legend": legend,
        "probabilities": probabilities,
        "confidence": confidence,
    }


def choice_confidence(probabilities: Sequence[float]) -> float:
    """`(n * p_max - 1) / (n - 1)`: 0 when spread evenly, 1 when all on one option."""
    n = len(probabilities)
    if n < 2:  # noqa: PLR2004 - a single option is certain by construction
        return 1.0
    return min(1.0, max(0.0, (n * max(probabilities) - 1) / (n - 1)))


def score_confidence(probabilities: Sequence[float]) -> float:
    """1 minus the mean distance from the peak level, relative to an even spread."""
    n = len(probabilities)
    if n < 2:  # noqa: PLR2004 - a single level is certain by construction
        return 1.0
    peak = max(range(n), key=probabilities.__getitem__)
    spread = sum(p * abs(level - peak) for level, p in enumerate(probabilities))
    even = sum(abs(level - (n - 1) / 2) for level in range(n)) / n
    return max(0.0, 1 - spread / even)


def serialize_state(state: object) -> JsonValue:
    """Convert state into JSON, turning LangChain messages into role/content objects.

    Raises:
        TypeError: If the state holds a value that has no JSON form.
    """
    if isinstance(state, BaseMessage):
        return serialize_state(convert_to_openai_messages(state))
    if isinstance(state, Mapping):
        return {str(key): serialize_state(value) for key, value in state.items()}
    if isinstance(state, list | tuple):
        return [serialize_state(item) for item in state]
    if state is None or isinstance(state, str | int | float | bool):
        return state
    msg = f"Cannot send a {type(state).__name__} to Jev; use JSON-compatible state."
    raise TypeError(msg)
