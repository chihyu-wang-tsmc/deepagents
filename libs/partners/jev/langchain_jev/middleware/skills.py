"""Suggest one skill per turn, optionally pin it, and check skill loads.

Follows https://docs.typesafe.ai/cookbooks/skill_suggestion: request 1 ranks every
skill and asks whether the request needs one at all; request 2 re-reads the top
three with their full descriptions and may reject them all. The winner's name goes
to the model as a short hint; the agent keeps its own skill list and judgment.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, cast

from langchain_core.decisions import Choice
from langchain_core.messages import HumanMessage, ToolMessage

from langchain_jev._messages import append_system_text, latest_request
from langchain_jev._types import (
    SKILL_LOAD_CHECK,
    SKILL_PINNING,
    SKILL_SUGGESTION,
    JevEvent,
)
from langchain_jev.middleware._base import Ask, AskMany, Flow, Handle, JevMiddleware
from langchain_jev.middleware.guardrails import JEV_MARK
from langchain_jev.questions import (
    SKILL_ACTION_QUESTIONS,
    SKILL_INVERTED,
    skill_suggestion_block,
)
from langchain_jev.switches import mode_for

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
    from langchain.agents.middleware.types import (
        ModelRequest,
        ModelResponse,
        ToolCallRequest,
    )
    from langchain_core.decisions import ClassifierResponse, Question
    from langgraph.runtime import Runtime
    from langgraph.types import Command

    from langchain_jev._runtime import JevRuntime
    from langchain_jev._types import Mode
    from langchain_jev.state import JevState

_FRONTMATTER = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_HINT_SOURCE = "jev_skill_suggestion"


class SkillSuggestionMiddleware(JevMiddleware):
    """Rank the agent's skills for the user's request and hint at the best one."""

    blueprint = SKILL_SUGGESTION

    def __init__(self, jev: JevRuntime) -> None:
        """Read `SKILL.md` excerpts from the configured backend, if any."""
        super().__init__(jev)
        self.backend: BackendProtocol | None = jev.config.skills.backend

    def bind_backend(self, backend: BackendProtocol) -> None:
        """Receive the agent's backend from `create_deep_agent`."""
        if self.backend is None:
            self.backend = backend

    def before_model(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """Suggest synchronously, once per run."""
        return self.drive(self._flow(state, runtime))

    async def abefore_model(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """Suggest asynchronously, once per run."""
        return await self.adrive(self._flow(state, runtime))

    def _flow(self, state: JevState, runtime: Runtime) -> Flow[dict[str, object] | None]:
        if state.get("jev_skill_done"):
            return None
        mode = self.mode(state, runtime)
        skills = list(state.get("skills_metadata") or [])
        request = latest_request(state["messages"])
        if mode == "off":
            return {"jev_skill_done": True}
        if not skills or not request:
            self.record(mode, "skip", reason="no_request" if skills else "no_skills")
            return {"jev_skill_done": True}
        shortlist = yield from self._rank(request, skills)
        winner, fits = (None, 0.0)
        if shortlist:
            winner, fits = yield from self._rerank(request, shortlist, skills)
        return self._update(state, mode, winner, fits)

    def _rank(self, request: str, skills: list[dict]) -> Flow[list[str] | None]:
        groups = _groups(skills, self.jev.model.max_choice_options)
        asks = tuple(
            Ask(_skill_state(request), self._wide(group, first=i == 0))
            for i, group in enumerate(groups)
        )
        responses = cast("list[ClassifierResponse | None]", (yield AskMany(asks)))
        first = responses[0]
        if first is None or not self._needs_skill(first):
            return None
        ranked: list[tuple[str, float]] = []
        for response in responses:
            if response is not None:
                ranked.extend(response.choices["which"].probabilities.items())
        ranked.sort(key=lambda item: -item[1])
        return [name for name, _ in ranked[: self.jev.config.skills.shortlist]]

    def _wide(self, group: list[dict], *, first: bool) -> dict[str, Question]:
        criteria = {skill["name"]: skill["description"] for skill in group}
        questions: dict[str, Question] = {
            "which": Choice(instructions=self.instructions("which"), criteria=criteria)
        }
        if first:
            questions.update({f"gate::{k}": self.question(k) for k in SKILL_ACTION_QUESTIONS})
            questions["named"] = self.question("named")
        return questions

    def _needs_skill(self, response: ClassifierResponse) -> bool:
        oriented = [
            1 - answer.noul if key.removeprefix("gate::") in SKILL_INVERTED else answer.noul
            for key, answer in response.nouls.items()
            if key.startswith("gate::")
        ]
        action = sum(oriented) / len(oriented)
        named = response.nouls["named"].noul
        return action >= self.threshold("skill_suggestion.action") or named >= self.threshold(
            "skill_suggestion.named"
        )

    def _rerank(
        self, request: str, shortlist: list[str], skills: list[dict]
    ) -> Flow[tuple[str | None, float]]:
        by_name = {skill["name"]: skill for skill in skills}
        excerpts = self._excerpts([by_name[name] for name in shortlist])
        criteria = {
            name: f"{by_name[name]['description']} — {excerpts.get(name, '')}".rstrip(" —")
            for name in shortlist
        }
        questions: dict[str, Question] = {
            "which": Choice(instructions=self.instructions("rerank"), criteria=criteria)
        }
        for name in shortlist:
            description = by_name[name]["description"]
            questions[f"fits::{name}"] = self.question("fits", name=name, description=description)
        response = cast("ClassifierResponse | None", (yield Ask(_skill_state(request), questions)))
        if response is None:
            return None, 0.0
        winner = response.choices["which"].choice
        fits = response.nouls[f"fits::{winner}"].noul
        best = max(response.nouls[f"fits::{name}"].noul for name in shortlist)
        return (winner, fits) if best >= self.threshold("skill_suggestion.fits") else (None, best)

    def _excerpts(self, skills: list[dict]) -> dict[str, str]:
        limit = self.jev.config.skills.excerpt_chars
        if self.backend is None or limit <= 0:
            return {}
        try:
            responses = self.backend.download_files([skill["path"] for skill in skills])
        except Exception:  # noqa: BLE001 - an excerpt only sharpens the hint; go on without it
            return {}
        excerpts: dict[str, str] = {}
        for skill, response in zip(skills, responses, strict=True):
            if response.content and not response.error:
                body = _FRONTMATTER.sub(
                    "", response.content.decode("utf-8", errors="replace"), count=1
                )
                excerpts[skill["name"]] = body.strip()[:limit]
        return excerpts

    def _update(
        self, state: JevState, mode: Mode, winner: str | None, fits: float
    ) -> dict[str, object]:
        self.record(mode, f"suggest:{winner}" if winner else "suggest:none", fits=fits)
        update: dict[str, object] = {"jev_skill_done": True}
        if mode != "on":
            return update
        if winner or self.jev.config.skills.announce_none:
            update["jev_skill_hint"] = skill_suggestion_block((winner,) if winner else ())
        snapshot = state.get("jev_modes")
        pin_on = isinstance(snapshot, Mapping) and mode_for(snapshot, SKILL_PINNING) == "on"
        if winner and pin_on and fits >= self.threshold("skill_pinning.min_fits"):
            update["pinned_skills"] = [winner]
            self.jev.emit(JevEvent(SKILL_PINNING, "on", f"pin:{winner}", {"fits": fits}))
        return update

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Show the hint to the model."""
        return handler(self._hinted(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Show the hint to the model."""
        return await handler(self._hinted(request))

    def _hinted(self, request: ModelRequest) -> ModelRequest:
        hint = request.state.get("jev_skill_hint")
        if not hint or self.mode(request.state, request.runtime) != "on":
            return request
        if self.jev.config.skills.placement == "system":
            return request.override(
                system_message=append_system_text(request.system_message, str(hint))
            )
        return request.override(messages=_after_latest_human(request.messages, str(hint)))


class SkillLoadCheckMiddleware(JevMiddleware):
    """When the agent reads a skill's `SKILL.md`, check that the skill fits the request."""

    blueprint = SKILL_LOAD_CHECK

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Check synchronously."""
        return self.drive(self._flow(request), cast("Callable[[object], object]", handler))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Check asynchronously."""
        return await self.adrive(
            self._flow(request), cast("Callable[[object], Awaitable[object]]", handler)
        )

    def _flow(self, request: ToolCallRequest) -> Flow[ToolMessage | Command]:
        skill = _skill_read(request)
        mode = self.mode(request.state, request.runtime, skill["name"] if skill else None)
        if skill is None or mode == "off":
            return cast("ToolMessage | Command", (yield Handle(request)))
        text = latest_request(request.state.get("messages", []))
        question = {
            "fits": self.question("fits", name=skill["name"], description=skill["description"])
        }
        response = cast("ClassifierResponse | None", (yield Ask(_skill_state(text), question)))
        fits = response.nouls["fits"].noul if response is not None else 1.0
        poor = fits < self.threshold("skill_load_check.min_fits")
        action = self.jev.config.skills.load_check_action if poor else "pass"
        self.record(mode, action, skill=skill["name"], fits=fits)
        if mode == "on" and action == "block":
            return _note(
                request,
                f"[Jev] Not loading skill '{skill['name']}': it does not appear to fit "
                f"this request (p={fits:.2f}).",
                error=True,
            )
        result = cast("ToolMessage | Command", (yield Handle(request)))
        if mode == "on" and action == "warn" and isinstance(result, ToolMessage):
            warning = (
                f"[Jev] Skill '{skill['name']}' may not fit this request (p={fits:.2f}); "
                "use it only if it clearly applies.\n\n"
            )
            return result.model_copy(update={"content": warning + result.text})
        return result


def _groups(skills: list[dict], size: int | None) -> list[list[dict]]:
    if not size or len(skills) <= size:
        return [skills]
    return [skills[i : i + size] for i in range(0, len(skills), size)]


def _skill_state(request: str) -> dict[str, str]:
    return {"request": request, "recent_context": ""}


def _after_latest_human(messages: Sequence[object], hint: str) -> list:
    note = HumanMessage(content=hint, additional_kwargs={JEV_MARK: _HINT_SOURCE})
    items = list(messages)
    for index in range(len(items) - 1, -1, -1):
        if isinstance(items[index], HumanMessage):
            return [*items[: index + 1], note, *items[index + 1 :]]
    return [*items, note]


def _skill_read(request: ToolCallRequest) -> dict | None:
    if request.tool_call["name"] != "read_file":
        return None
    path = str(request.tool_call["args"].get("file_path", ""))
    skills = request.state.get("skills_metadata") or []
    return next((skill for skill in skills if skill.get("path") == path), None)


def _note(request: ToolCallRequest, content: str, *, error: bool) -> ToolMessage:
    return ToolMessage(
        content=content,
        tool_call_id=request.tool_call["id"],
        name=request.tool_call["name"],
        status="error" if error else "success",
        additional_kwargs={JEV_MARK: SKILL_LOAD_CHECK},
    )
