"""Assemble the Jev middleware in the right order, and register them for subagents.

Order matters for three kinds of hook:

- `before_agent` runs in list order: the run snapshot first, then memories (so the
    entry request can see them), then the entry request, then the blueprints that
    act on its answers.
- `wrap_tool_call` nests in list order: the tool gate wraps the skill-load check,
    which wraps passage classification, so a refused call never reaches retrieval.
- `after_agent` runs in reverse: answer checks run first and may loop back to the
    model; memory and review only run once the answer is final.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepagents import HarnessProfile, register_harness_profile

from langchain_jev._runtime import JevRuntime
from langchain_jev._types import (
    ANSWER_SCORING,
    CHAT_MODE,
    CITATION_CHECK,
    CODING_ROUTER,
    FAST_PATH,
    INPUT_GUARD,
    MEMORY_READ,
    MEMORY_WRITE,
    MODEL_ROUTER,
    OUTPUT_GUARD,
    RAG_PASSAGES,
    RUN_REVIEW,
    SKILL_LOAD_CHECK,
    SKILL_SUGGESTION,
    TOOL_GATE,
    TOOL_RESULT_GUARD,
    VERIFY_ANSWER,
)
from langchain_jev.config import JevConfig
from langchain_jev.middleware.fast_path import FastPathMiddleware
from langchain_jev.middleware.guardrails import (
    InputGuardMiddleware,
    OutputGuardMiddleware,
    ToolResultGuardMiddleware,
)
from langchain_jev.middleware.memory import MemoryReadMiddleware, MemoryWriteMiddleware
from langchain_jev.middleware.rag import RagPassagesMiddleware
from langchain_jev.middleware.review import RunReviewMiddleware
from langchain_jev.middleware.routing import (
    ChatModeMiddleware,
    CodingRouterMiddleware,
    ModelRouterMiddleware,
)
from langchain_jev.middleware.run import (
    EntryParticipant,
    JevEntryMiddleware,
    JevRunMiddleware,
)
from langchain_jev.middleware.skills import (
    SkillLoadCheckMiddleware,
    SkillSuggestionMiddleware,
)
from langchain_jev.middleware.tool_gate import ToolGateMiddleware
from langchain_jev.middleware.verify import AnswerCheckMiddleware

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain.agents.middleware.types import AgentMiddleware

    from langchain_jev.middleware._base import JevMiddleware

_ENTRY: tuple[tuple[str, Callable[[JevRuntime], JevMiddleware]], ...] = (
    (INPUT_GUARD, InputGuardMiddleware),
    (FAST_PATH, FastPathMiddleware),
    (MODEL_ROUTER, ModelRouterMiddleware),
    (CODING_ROUTER, CodingRouterMiddleware),
    (CHAT_MODE, ChatModeMiddleware),
)
_REST: tuple[tuple[str, Callable[[JevRuntime], JevMiddleware]], ...] = (
    (SKILL_SUGGESTION, SkillSuggestionMiddleware),
    (TOOL_RESULT_GUARD, ToolResultGuardMiddleware),
    (OUTPUT_GUARD, OutputGuardMiddleware),
    (TOOL_GATE, ToolGateMiddleware),
    (SKILL_LOAD_CHECK, SkillLoadCheckMiddleware),
    (RAG_PASSAGES, RagPassagesMiddleware),
    (RUN_REVIEW, RunReviewMiddleware),
    (MEMORY_WRITE, MemoryWriteMiddleware),
)
_CHECKS = (VERIFY_ANSWER, CITATION_CHECK, ANSWER_SCORING)


def jev_middleware(
    config: JevConfig | None = None, *, runtime: JevRuntime | None = None
) -> list[AgentMiddleware[Any, Any, Any]]:
    """Build the ordered middleware for the blueprints `config` turns on.

    Pass the result to `create_deep_agent(middleware=...)` for the main agent only,
    or use `register_jev_profile` so subagents get the same checks. With no blueprint
    on, the list is empty.

    Args:
        config: What to install; `JevConfig()` turns nothing on.
        runtime: Share one runtime (its models, cache, and in-process memory)
            between several stacks.
    """
    jev = runtime or JevRuntime(config or JevConfig())
    installed = jev.installed
    if not installed:
        return []
    stack: list[AgentMiddleware[Any, Any, Any]] = [JevRunMiddleware(jev)]
    if MEMORY_READ in installed:
        stack.append(MemoryReadMiddleware(jev))
    entry = [build(jev) for blueprint, build in _ENTRY if blueprint in installed]
    if entry:
        stack.append(JevEntryMiddleware(jev, [_participant(m) for m in entry]))
        stack.extend(entry)
    stack.extend(build(jev) for blueprint, build in _REST if blueprint in installed)
    if any(check in installed for check in _CHECKS):
        stack.append(AnswerCheckMiddleware(jev))
    return stack


def register_jev_profile(providers: Sequence[str], config: JevConfig | None = None) -> JevRuntime:
    """Install the Jev stack on every deep agent whose model comes from `providers`.

    Uses `HarnessProfile.extra_middleware`, which `create_deep_agent` applies to the
    main agent, the general-purpose subagent, and declarative subagents, so a
    subagent's tool calls pass the same tool gate as the main agent's. Each agent
    stack gets fresh middleware instances that share one runtime.

    Args:
        providers: Profile keys such as `openai`, `anthropic`, or `openai:gpt-5.4`.
        config: What to install; `JevConfig()` turns nothing on.

    Returns:
        The shared runtime, for inspection or to build more stacks with it.
    """
    jev = JevRuntime(config or JevConfig())
    for key in providers:
        register_harness_profile(
            key, HarnessProfile(extra_middleware=lambda: jev_middleware(runtime=jev))
        )
    return jev


def _participant(middleware: JevMiddleware) -> EntryParticipant:
    if not hasattr(middleware, "entry_questions"):
        msg = f"{type(middleware).__name__} does not take part in the entry request."
        raise TypeError(msg)
    return middleware  # ty: ignore[invalid-explicit-return]  # checked structurally above
