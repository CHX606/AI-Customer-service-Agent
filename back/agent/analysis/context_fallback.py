"""客服请求分析职责模块：context_fallback。"""
from back.domain.conversation import ActiveIssue, IntentAction
from back.agent.analysis.request_models import RequestAnalysis
from back.agent.analysis.request_rules import analyze_request_fast_path, _direct_analysis

def _fallback_context_analysis(
    *,
    current_query: str,
    recent_messages: list[dict[str, str]],
    active_issue: ActiveIssue | None,
    business_scope: list[str],
) -> RequestAnalysis:
    """上下文模型不可用时保留已有信息，避免把连接异常变成空白回复。"""

    current_direct = analyze_request_fast_path(
        current_query=current_query,
        recent_messages=[],
        active_issue=None,
        business_scope=business_scope,
    )
    active_intent = active_issue.get("intent") if active_issue else None
    if current_direct is not None and (
        current_direct.scope in {"chitchat", "out_of_scope"}
        or (
            current_direct.scope == "in_scope"
            and active_intent not in {None, "unknown", "other_business"}
            and current_direct.intent not in {
                "unknown",
                "other_business",
                active_intent,
            }
        )
    ):
        return current_direct.model_copy(
            update={"route_source": "fallback_context"}
        )

    if active_issue is None:
        if current_direct is not None:
            return current_direct.model_copy(
                update={"route_source": "fallback_context"}
            )
        return _direct_analysis(
            current_query=current_query,
            scope="uncertain",
            scope_reason="上下文分析服务暂时不可用，需要用户补充完整问题。",
            clarifying_question="请完整描述一下业务对象和当前故障现象。",
            missing_information=["完整业务对象和故障现象"],
            self_contained=False,
        ).model_copy(update={"route_source": "fallback_context"})

    context_parts: list[str] = []
    seen: set[str] = set()

    def add_context(value: str | None) -> None:
        cleaned = " ".join((value or "").split())
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            context_parts.append(cleaned)

    add_context(active_issue.get("summary"))
    for message in recent_messages[-4:]:
        if message.get("role") == "user":
            add_context(message.get("content"))
    add_context(current_query)
    resolved_query = "；".join(context_parts) or current_query.strip()

    resolved_direct = analyze_request_fast_path(
        current_query=resolved_query,
        recent_messages=[],
        active_issue=None,
        business_scope=business_scope,
    )
    intent = active_intent or "other_business"
    action: IntentAction = "retrieve"
    clarifying_question = None
    missing_information: list[str] = []
    if resolved_direct is not None and resolved_direct.scope == "in_scope":
        if resolved_direct.intent != "unknown":
            intent = resolved_direct.intent
        action = resolved_direct.action
        clarifying_question = resolved_direct.clarifying_question
        missing_information = resolved_direct.missing_information

    return RequestAnalysis(
        relation="continue",
        resolved_query=resolved_query,
        context_reason="上下文模型暂时不可用，系统使用会话中已确认的信息继续处理。",
        is_self_contained=False,
        references_active_issue=True,
        answers_last_question=active_issue.get("status") == "awaiting_user",
        explicit_new_issue=False,
        scope="in_scope",
        scope_reason="当前消息继续已有业务问题。",
        intent=intent,
        action=action,
        known_information=context_parts,
        missing_information=missing_information,
        clarifying_question=clarifying_question,
        intent_reason="根据当前活动问题和已确认的会话信息执行降级路由。",
        rewritten_queries=[],
        rewrite_reason=None,
        route_source="fallback_context",
    )
