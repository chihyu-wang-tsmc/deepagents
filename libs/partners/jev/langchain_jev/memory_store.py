"""Where the memory blueprints keep facts worth remembering about a user."""

from __future__ import annotations

import threading
import uuid
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore


@runtime_checkable
class JevMemoryStore(Protocol):
    """Saves and loads memory records under a namespace such as `("jev", user)`."""

    def load(self, namespace: tuple[str, ...], limit: int) -> list[dict]:
        """Return up to `limit` records, oldest first."""
        ...

    def save(self, namespace: tuple[str, ...], record: dict) -> None:
        """Add one record."""
        ...


class InMemoryJevMemory:
    """Records kept in this process; they are lost on restart."""

    def __init__(self) -> None:
        """Start empty."""
        self._lock = threading.Lock()
        self._records: dict[tuple[str, ...], list[dict]] = {}

    def load(self, namespace: tuple[str, ...], limit: int) -> list[dict]:
        """Return the newest `limit` records, oldest first."""
        with self._lock:
            return list(self._records.get(namespace, [])[-limit:])

    def save(self, namespace: tuple[str, ...], record: dict) -> None:
        """Append one record."""
        with self._lock:
            self._records.setdefault(namespace, []).append(dict(record))


class GraphStoreMemory:
    """Records kept in the agent's LangGraph store, so they survive restarts."""

    def __init__(self, store: BaseStore) -> None:
        """Wrap `store`."""
        self.store = store

    def load(self, namespace: tuple[str, ...], limit: int) -> list[dict]:
        """Return up to `limit` records."""
        items = self.store.search(namespace, limit=limit)
        return [dict(item.value) for item in sorted(items, key=lambda item: item.created_at)]

    def save(self, namespace: tuple[str, ...], record: dict) -> None:
        """Store one record under a fresh key."""
        self.store.put(namespace, uuid.uuid4().hex, dict(record))
