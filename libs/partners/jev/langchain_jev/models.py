"""Named Jev models, so swapping one for another is a configuration change.

A `JevModel` bundles how to build the decision model with what is specific to it:
the thresholds re-tuned for its probabilities and the limits of what it accepts.
Register a new model once, then select it by name with `JevConfig(model=...)` or
the `JEV_MODEL` environment variable; no middleware changes.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.decisions import BaseDecisionModel
from pydantic import SecretStr

from langchain_jev.classifiers import SystemOneClassifier

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

DEFAULT_MODEL_ENV = "JEV_MODEL"
"""Environment variable naming the model used when a config asks for `default`."""

FALLBACK_MODEL = "h2o-lightning-4b"
"""The model used when neither the config nor `JEV_MODEL` names one."""


@dataclass(frozen=True)
class JevModel:
    """A decision model plus the settings that belong to it.

    Example:
        ```python
        from langchain_jev import JevModel, SystemOneClassifier, register_jev_model

        register_jev_model(
            JevModel(
                name="better-jev",
                factory=lambda: SystemOneClassifier(
                    base_url="http://gpu-box:8741", model="better-jev", send_model=False
                ),
                thresholds={"input_guard.action": 0.8},
            )
        )
        ```
    """

    name: str
    """Registry name, for example `h2o-lightning-4b`."""

    factory: Callable[[], BaseDecisionModel]
    """Builds the decision model. Called once per agent build."""

    thresholds: Mapping[str, float] = field(default_factory=dict)
    """Thresholds re-tuned for this model; they override the package defaults."""

    noul_floor: float | None = None
    """Set when the server clamps Noul probabilities to `[1 - floor, floor]` edges.

    Thresholds strictly between the edges then behave like 0.5; the package logs a
    warning listing them. Set the server's floor to 0 to keep raw probabilities.
    """

    max_choice_options: int | None = None
    """Most options one Choice may carry, or `None` for no known limit."""

    max_state_chars: int = 32_000
    """Longest text sent in one state field; longer text is truncated."""

    description: str = ""


_REGISTRY: dict[str, JevModel] = {}
_LOCK = threading.Lock()


def register_jev_model(model: JevModel, *, replace: bool = False) -> None:
    """Make `model` selectable by name.

    Raises:
        ValueError: If the name is taken and `replace` is false.
    """
    with _LOCK:
        if model.name in _REGISTRY and not replace:
            msg = f"A Jev model named {model.name!r} is already registered."
            raise ValueError(msg)
        _REGISTRY[model.name] = model


def get_jev_model(name: str = "default") -> JevModel:
    """Look up a registered model; `default` reads `JEV_MODEL`.

    Raises:
        KeyError: If no model has that name.
    """
    if name == "default":
        name = os.environ.get(DEFAULT_MODEL_ENV) or FALLBACK_MODEL
    with _LOCK:
        model = _REGISTRY.get(name)
    if model is None:
        msg = f"No Jev model named {name!r}; registered: {', '.join(list_jev_models())}"
        raise KeyError(msg)
    return model


def list_jev_models() -> list[str]:
    """Names of every registered model."""
    with _LOCK:
        return sorted(_REGISTRY)


def as_jev_model(spec: str | JevModel | BaseDecisionModel) -> JevModel:
    """Accept a registry name, a `JevModel`, or a ready decision model."""
    if isinstance(spec, JevModel):
        return spec
    if isinstance(spec, BaseDecisionModel):
        return JevModel(name=spec.model, factory=lambda: spec)
    return get_jev_model(spec)


def _h2o_lightning(model: str, url_env: str) -> SystemOneClassifier:
    base_url = os.environ.get(url_env) or os.environ.get("H2O_BASE_URL")
    return SystemOneClassifier(
        base_url=base_url or "http://localhost:8741",
        model=model,
        send_model=False,
        provider="h2o-lightning",
    )


def _typesafe() -> SystemOneClassifier:
    key = os.environ.get("TYPESAFE_API_KEY")
    return SystemOneClassifier(
        base_url=os.environ.get("TYPESAFE_BASE_URL") or "https://api.typesafe.ai",
        model=os.environ.get("TYPESAFE_MODEL") or "jev-latest",
        api_key=SecretStr(key) if key else None,
        provider="typesafe",
    )


def _register_builtins() -> None:
    register_jev_model(
        JevModel(
            name="h2o-lightning-4b",
            factory=lambda: _h2o_lightning("h2oai/h2o-lightning-4b", "H2O_4B_BASE_URL"),
            noul_floor=0.801,
            description="Open-source H2O-Lightning-4B behind its System One shim.",
        )
    )
    register_jev_model(
        JevModel(
            name="h2o-lightning-31b",
            factory=lambda: _h2o_lightning("h2oai/h2o-lightning-31b", "H2O_31B_BASE_URL"),
            noul_floor=0.801,
            description="Open-source H2O-Lightning-31B behind its System One shim.",
        )
    )
    register_jev_model(
        JevModel(
            name="typesafe",
            factory=_typesafe,
            description="TypeSafe's hosted Jev; reads TYPESAFE_API_KEY and TYPESAFE_MODEL.",
        )
    )


_register_builtins()
