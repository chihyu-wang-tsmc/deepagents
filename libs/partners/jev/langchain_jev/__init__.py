"""Jev (System One) decision middleware for deepagents.

Typesafe's agent use cases as deepagents middleware: guardrails, model routing,
skill suggestion, RAG passage classification, tool gating, answer verification, and
run review. Each blueprint can be switched off, on, or into shadow mode at build time,
without a restart, or per run; the Jev model is swapped by name. `JevBlueprint` adds
blueprints of your own: asked at the five stages the built-in ones use, and acting at
any later position in the loop.
"""

from langchain_jev._runtime import JevRuntime
from langchain_jev._types import BLUEPRINTS, JevEvent, Mode
from langchain_jev.classifiers import JevClassifierError, SystemOneClassifier
from langchain_jev.config import (
    CodingSettings,
    GuardSettings,
    JevConfig,
    RagSettings,
    ReviewSettings,
    RouteChoice,
    RouterSettings,
    ScoreSpec,
    SkillSettings,
    ToolGateSettings,
    VerifySettings,
)
from langchain_jev.custom import (
    POSITIONS,
    STAGES,
    Action,
    AskHuman,
    At,
    DropTools,
    JevBlueprint,
    Mark,
    Notify,
    NoTools,
    Redact,
    Refuse,
    RewriteReply,
    RewriteToolResult,
    ShapeModelCall,
    ShapeToolCall,
    Stage,
    SystemNote,
    UseModel,
    Warn,
)
from langchain_jev.models import (
    JevModel,
    get_jev_model,
    list_jev_models,
    register_jev_model,
)
from langchain_jev.question_book import QuestionOverride, default_questions
from langchain_jev.questions import noul
from langchain_jev.review import (
    RunContext,
    RunReview,
    areview_run,
    compare_answers,
    review_run,
)
from langchain_jev.stack import jev_middleware, register_jev_profile
from langchain_jev.switches import (
    EnvModeSource,
    FileModeSource,
    ModeSource,
    StaticModeSource,
)
from langchain_jev.trace import JevRequest, JevTrace

__all__ = [
    "BLUEPRINTS",
    "POSITIONS",
    "STAGES",
    "Action",
    "AskHuman",
    "At",
    "CodingSettings",
    "DropTools",
    "EnvModeSource",
    "FileModeSource",
    "GuardSettings",
    "JevBlueprint",
    "JevClassifierError",
    "JevConfig",
    "JevEvent",
    "JevModel",
    "JevRequest",
    "JevRuntime",
    "JevTrace",
    "Mark",
    "Mode",
    "ModeSource",
    "NoTools",
    "Notify",
    "QuestionOverride",
    "RagSettings",
    "Redact",
    "Refuse",
    "ReviewSettings",
    "RewriteReply",
    "RewriteToolResult",
    "RouteChoice",
    "RouterSettings",
    "RunContext",
    "RunReview",
    "ScoreSpec",
    "ShapeModelCall",
    "ShapeToolCall",
    "SkillSettings",
    "Stage",
    "StaticModeSource",
    "SystemNote",
    "SystemOneClassifier",
    "ToolGateSettings",
    "UseModel",
    "VerifySettings",
    "Warn",
    "areview_run",
    "compare_answers",
    "default_questions",
    "get_jev_model",
    "jev_middleware",
    "list_jev_models",
    "noul",
    "register_jev_model",
    "register_jev_profile",
    "review_run",
]
