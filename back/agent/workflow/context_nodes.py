"""客服工作流职责模块：context_nodes"""
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from back.agent.workflow.lifecycle import set_issue_intent, start_or_continue_issue, transition_issue, user_confirms_resolution
from back.agent.analysis.request import build_search_queries, route_request
from back.agent.workflow.state import CustomerServiceState
from back.tenant.service import get_tenant_profile
from back.core import llm as models

def _context_llm():
    """测试默认模型可被 patch；独立配置时使用专用上下文模型。"""
    return models.model if models.CONTEXT_MODEL_NAME == models.MODEL_NAME else models.get_context_model()

def build_recent_messages(
    messages: list[BaseMessage],
    limit: int = 6,
) -> list[dict[str, str]]:
    """将 LangChain 消息转换成上下文分析需要的格式。"""
    recent_messages = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue

        if not message.content:
            continue

        recent_messages.append(
            {
                "role": role,
                "content": str(message.content),
            }
        )
    return recent_messages[-limit:]

def analyze_request_node(state: CustomerServiceState):
    """分层完成上下文、范围、意图和检索查询分析。"""
    messages = state["messages"]
    current_user_index = None
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            current_user_index = index
            break

    if current_user_index is None:
        raise ValueError("消息列表中没有找到用户消息")

    current_query = str(messages[current_user_index].content)
    active_issue = state.get("active_issue")

    # 用户明确确认问题已解决时不再调用分析模型。
    if active_issue and user_confirms_resolution(current_query):
        return {
            "relation": "continue",
            "route_source": "fast_rule",
            "resolved_query": current_query,
            "context_reason": "用户明确确认当前问题已经解决。",
            "scope": "chitchat",
            "scope_reason": "用户正在结束当前业务问题。",
            "intent": "unknown",
            "action": "clarify",
            "known_information": [],
            "missing_information": [],
            "clarifying_question": None,
            "intent_reason": "无需继续分析业务意图。",
            "rewritten_queries": [],
            "rewrite_reason": None,
            "search_queries": [],
            "active_issue": transition_issue(active_issue, "resolved"),
        }

    profile = get_tenant_profile(state.get("tenant_id", "default"))
    recent_messages = build_recent_messages(messages[:current_user_index])
    router_llm = models.model if models.ROUTER_MODEL_NAME == models.MODEL_NAME else models.get_router_model()
    fallback_llm = (
        _context_llm()
        if active_issue is not None or recent_messages
        else models.model
    )
    analysis = route_request(
        router_llm=router_llm,
        fallback_llm=fallback_llm,
        current_query=current_query,
        recent_messages=recent_messages,
        active_issue=active_issue,
        company_name=profile.company_name,
        business_scope=profile.business_scope,
    )

    if analysis.scope == "in_scope":
        active_issue = start_or_continue_issue(
            active_issue=active_issue,
            relation=analysis.relation,
            summary=analysis.resolved_query,
        )
        active_issue = set_issue_intent(active_issue, analysis.intent)

    return {
        "relation": analysis.relation,
        "route_source": analysis.route_source,
        "resolved_query": analysis.resolved_query,
        "context_reason": analysis.context_reason,
        "scope": analysis.scope,
        "scope_reason": analysis.scope_reason,
        "intent": analysis.intent,
        "action": analysis.action,
        "known_information": analysis.known_information,
        "missing_information": analysis.missing_information,
        "clarifying_question": analysis.clarifying_question,
        "intent_reason": analysis.intent_reason,
        "rewritten_queries": analysis.rewritten_queries,
        "rewrite_reason": analysis.rewrite_reason,
        "search_queries": build_search_queries(analysis),
        "active_issue": active_issue,
    }
