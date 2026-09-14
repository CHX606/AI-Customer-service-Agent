"""客服请求分析职责模块：request_models。"""
from typing import Literal
from pydantic import BaseModel, Field
from back.domain.conversation import IntentAction, IntentType, RelationType

RouteSource = Literal[
    "fast_rule",
    "light_model",
    "legacy_context",
    "legacy_full",
    "fallback_context",
    "fallback_full",
]


class RequestAnalysis(BaseModel):
    """一次模型调用返回的完整路由结果。"""

    relation: RelationType
    resolved_query: str = Field(min_length=1)
    context_reason: str
    is_self_contained: bool
    references_active_issue: bool
    answers_last_question: bool
    explicit_new_issue: bool

    scope: Literal["in_scope", "chitchat", "out_of_scope", "uncertain"]
    scope_reason: str

    intent: IntentType = "unknown"
    action: IntentAction = "clarify"
    known_information: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    clarifying_question: str | None = None
    intent_reason: str = ""

    rewritten_queries: list[str] = Field(default_factory=list, max_length=2)
    rewrite_reason: str | None = None
    route_source: RouteSource = "legacy_full"


class RouteDecision(BaseModel):
    """Router V2 的最小单轮输出，避免生成完整分析对象。"""

    scope: Literal["in_scope", "chitchat", "out_of_scope", "uncertain"]
    intent: IntentType = "unknown"
    action: IntentAction = "clarify"
    clarifying_question: str | None = Field(default=None, max_length=200)
    rewritten_query: str | None = Field(default=None, max_length=200)
