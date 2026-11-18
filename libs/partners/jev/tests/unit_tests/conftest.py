from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def harness_registry() -> Iterator[None]:
    """Restore deepagents' global harness-profile registry after a test."""
    from deepagents.profiles.harness.harness_profiles import (
        _HARNESS_PROFILES,
        _ensure_harness_profiles_loaded,
    )

    _ensure_harness_profiles_loaded()
    original = dict(_HARNESS_PROFILES)
    try:
        yield
    finally:
        _HARNESS_PROFILES.clear()
        _HARNESS_PROFILES.update(original)
