"""Hand each finished run to the agent-trace review."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_jev._types import RUN_REVIEW
from langchain_jev.middleware._base import Flow, JevMiddleware
from langchain_jev.review import RunContext, review_flow

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

    from langchain_jev.state import JevState


class RunReviewMiddleware(JevMiddleware):
    """Pass the transcript on for review, or review it inline when configured.

    Reviewing inline adds a few Jev requests after the final answer. Leave `inline`
    off to hand the transcript to `on_run_end` and review it off the request path
    with `review_run`.
    """

    blueprint = RUN_REVIEW

    def after_agent(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """Review synchronously."""
        return self.drive(self._flow(state, runtime))

    async def aafter_agent(self, state: JevState, runtime: Runtime) -> dict[str, object] | None:
        """Review asynchronously."""
        return await self.adrive(self._flow(state, runtime))

    def _flow(self, state: JevState, runtime: Runtime) -> Flow[None]:
        mode = self.mode(state, runtime)
        settings = self.jev.config.review
        messages = list(state["messages"])
        if mode == "off":
            return None
        if not settings.inline:
            if mode == "on" and settings.on_run_end is not None:
                settings.on_run_end(messages)
            self.record(mode, "handed_off", messages=len(messages))
            return None
        context = RunContext(
            irreversible_tools=settings.irreversible_tools,
            instructions=settings.agent_instructions,
            questions=self.jev.config.questions,
        )
        review = yield from review_flow(messages, context)
        if review is None:
            return None
        self.record(mode, review.outcome, state=review.state, breaches=review.breaches)
        if mode == "on" and settings.on_review is not None:
            settings.on_review(review)
        return None
