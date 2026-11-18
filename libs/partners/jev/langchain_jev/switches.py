"""Turn any blueprint off, on, or into shadow mode, at three levels.

1. Build time: `JevConfig.blueprints` decides which blueprints are installed. An
    `off` blueprint is not in the middleware stack at all.
2. Without a restart: a `ModeSource` (an environment variable, a JSON file, or your
    own flag service) is read at the start of every run.
3. Per run: `agent.invoke(..., context={"jev_modes": {...}})` overrides both for one
    run, for one user, thread, or experiment arm.

The more specific level wins. Keys are blueprint ids (`tool_gate`) or a blueprint id
and one of its parts (`input_guard.medical_advice`, `tool_gate.execute`). A part can
narrow its blueprint (`on` to `shadow` or `off`) but never widen it.

The modes are read once when a run starts and kept for the whole run, including
after a human-in-the-loop interrupt resumes, so a decision never flips mid-run.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from langchain_jev._types import BLUEPRINTS, MODES, Mode

logger = logging.getLogger(__name__)

_RANK: dict[str, int] = {"off": 0, "shadow": 1, "on": 2}


@runtime_checkable
class ModeSource(Protocol):
    """Anything that can report blueprint modes without restarting the agent."""

    def modes(self) -> Mapping[str, Mode]:
        """Return the current overrides, keyed by blueprint id or `id.part`."""
        ...


class StaticModeSource:
    """An in-process switchboard; flip blueprints from your own code."""

    def __init__(self, modes: Mapping[str, Mode] | None = None) -> None:
        """Start with optional overrides."""
        self._lock = threading.Lock()
        self._modes: dict[str, Mode] = parse_modes(modes or {}, source="StaticModeSource")

    def set(self, key: str, mode: Mode) -> None:
        """Override one blueprint or part from the next run on."""
        parsed = parse_modes({key: mode}, source="StaticModeSource")
        with self._lock:
            self._modes.update(parsed)

    def clear(self, key: str | None = None) -> None:
        """Drop one override, or all of them."""
        with self._lock:
            if key is None:
                self._modes.clear()
            else:
                self._modes.pop(key, None)

    def modes(self) -> dict[str, Mode]:
        """Return a copy of the current overrides."""
        with self._lock:
            return dict(self._modes)


class EnvModeSource:
    """Read overrides such as `tool_gate=off,rag_passages=shadow` from a variable."""

    def __init__(self, variable: str = "JEV_MODES") -> None:
        """Watch `variable`."""
        self.variable = variable

    def modes(self) -> dict[str, Mode]:
        """Parse the variable as it is now."""
        raw = os.environ.get(self.variable, "")
        pairs = (item.split("=", 1) for item in raw.split(",") if "=" in item)
        return parse_modes(
            {key.strip(): value.strip() for key, value in pairs},
            source=self.variable,
        )


class FileModeSource:
    """Read overrides from a JSON object such as `{"tool_gate": "off"}`.

    The file is re-read when it changes, at most every `ttl` seconds. A file that
    fails to parse keeps the last good overrides, so a bad edit never flips a guard.
    """

    def __init__(self, path: str | os.PathLike[str], *, ttl: float = 2.0) -> None:
        """Watch the JSON file at `path`."""
        self.path = Path(path)
        self.ttl = ttl
        self._lock = threading.Lock()
        self._checked = float("-inf")
        self._mtime: float | None = None
        self._modes: dict[str, Mode] = {}

    def modes(self) -> dict[str, Mode]:
        """Return the overrides, reloading the file if it changed."""
        with self._lock:
            now = time.monotonic()
            if now - self._checked >= self.ttl:
                self._checked = now
                self._reload()
            return dict(self._modes)

    def _reload(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            self._mtime, self._modes = None, {}
            return
        if mtime == self._mtime:
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Keeping previous Jev modes: %s is not valid JSON.", self.path)
            return
        if not isinstance(raw, Mapping):
            logger.warning("Keeping previous Jev modes: %s is not an object.", self.path)
            return
        self._mtime, self._modes = mtime, parse_modes(raw, source=str(self.path))


def parse_modes(raw: Mapping[str, object], *, source: str) -> dict[str, Mode]:
    """Keep the entries with a known blueprint and a valid mode; log the rest."""
    parsed: dict[str, Mode] = {}
    for key, value in raw.items():
        blueprint = key.split(".", 1)[0]
        if blueprint not in BLUEPRINTS or value not in MODES:
            logger.warning("Ignoring Jev mode %r=%r from %s.", key, value, source)
            continue
        parsed[key] = cast("Mode", value)
    return parsed


def context_modes(context: object, key: str) -> dict[str, Mode]:
    """Read per-run overrides from the runtime context, a mapping or an object."""
    if context is None:
        return {}
    raw = context.get(key) if isinstance(context, Mapping) else getattr(context, key, None)
    if not isinstance(raw, Mapping):
        return {}
    return parse_modes(raw, source="runtime context")


def resolve_modes(installed: Mapping[str, Mode], *layers: Mapping[str, Mode]) -> dict[str, Mode]:
    """Merge override layers onto the installed modes, later layers winning.

    A blueprint that is not installed stays absent: overrides cannot add middleware
    that was never built.
    """
    snapshot: dict[str, Mode] = dict(installed)
    for layer in layers:
        snapshot.update({k: m for k, m in layer.items() if k.split(".", 1)[0] in installed})
    return snapshot


def mode_for(snapshot: Mapping[str, Mode], blueprint: str, part: str | None = None) -> Mode:
    """The effective mode of a blueprint, or of one part of it."""
    mode = snapshot.get(blueprint, "off")
    if part is None:
        return mode
    narrowed = snapshot.get(f"{blueprint}.{part}", mode)
    return mode if _RANK[narrowed] > _RANK[mode] else narrowed
