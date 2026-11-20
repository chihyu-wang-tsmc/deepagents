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
    """Review each finished run, and hand its transcript on.

    With `on_review` set, the run is reviewed right after it ends, which adds a few
    Jev requests after the final answer. With `on_run_end` set, the transcript is
    handed over to review later, off the request path, with `review_run`.
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
        if mode == "on" and settings.on_run_end is not None:
            settings.on_run_end(messages)
        inline = settings.on_review is not None if settings.inline is None else settings.inline
        if not inline:
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
