"""客服请求分析职责模块：request。"""
import json
import logging
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from back.agent.analysis.scope import _normalize_query
from back.domain.conversation import ActiveIssue
from back.agent.analysis.request_models import RequestAnalysis, RouteDecision
from back.agent.analysis.request_rules import _env_flag, analyze_request_fast_path, normalize_request_analysis, build_search_queries
from back.agent.analysis.context_fallback import _fallback_context_analysis
from back.agent.analysis.structured_output import _invoke_structured_output

logger = logging.getLogger(__name__)

def analyze_request(
    *,
    llm: BaseChatModel,
    current_query: str,
    recent_messages: list[dict[str, str]],
    active_issue: ActiveIssue | None,
    company_name: str,
    business_scope: list[str],
) -> RequestAnalysis:
    """用一次结构化模型调用完成请求分析。"""
    scope_text = "、".join(business_scope) if business_scope else "企业相关业务"

    system_prompt = f"""
你是{company_name}客服系统的请求分析器。请一次完成以下工作，不要回答用户问题：

1. 上下文：判断 relation（continue/new_issue/correction/uncertain），并生成可独立理解的 resolved_query。
2. 范围：判断 scope（in_scope/chitchat/out_of_scope/uncertain）。企业业务范围包括：{scope_text}。
3. 意图与动作：仅当 scope=in_scope 时判断 intent，并选择 action：
   - profile：企业介绍、营业时间、联系方式等固定资料；
   - retrieve：问题具体，进入知识库检索；
   - clarify：缺少决定检索方向的关键信息，需要追问。
4. 查询改写：仅当 action=retrieve 时生成1到2条简短检索问题；其他分支返回空列表。

必须遵守：
- 没有正在处理的问题时 relation=new_issue。
- “另外/换个问题”等明确切换为 new_issue；回答上轮追问通常为 continue；明确推翻旧信息为 correction。
- new_issue 的 resolved_query 不得混入旧问题；continue/correction 可合并用户已确认的上下文。
- 问候、感谢、告别为 chitchat；天气、新闻、通用编程等企业无关问题为 out_of_scope。
- 完全没有业务对象或故障现象的短句为 uncertain。
- 不得编造用户未提供的事实、原因或解决方案。
- retrieve/profile 时 missing_information 必须为空且 clarifying_question=null。
- 改写必须保留否定关系、事件顺序、平台和软件名，不能替换流量/套餐/订阅/节点等不同概念。
- 所有 reason 只需一句简短说明。
"""

    input_data = {
        "active_issue": active_issue,
        "recent_messages": recent_messages,
        "current_query": current_query,
    }
    result = _invoke_structured_output(
        llm,
        RequestAnalysis,
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(input_data, ensure_ascii=False)),
        ],
    )
    return normalize_request_analysis(
        result,
        current_query=current_query,
        active_issue=active_issue,
        business_scope=business_scope,
    )


def _analyze_single_turn_with_router(
    *,
    llm: BaseChatModel,
    current_query: str,
    company_name: str,
    business_scope: list[str],
) -> RequestAnalysis:
    """用最小结构化输出处理无法由确定性规则覆盖的单轮请求。"""
    scope_text = "、".join(business_scope) if business_scope else "企业相关业务"
    result = _invoke_structured_output(
        llm,
        RouteDecision,
        [
            SystemMessage(
                content=f"""
你是{company_name}客服系统的轻量路由器，只分类，不回答用户问题。
企业业务范围：{scope_text}。

输出规则：
- scope 只能是 in_scope/chitchat/out_of_scope/uncertain。
- in_scope 时选择 intent，并选择 profile/retrieve/clarify。
- profile 只用于企业介绍、营业时间、联系方式等固定资料。
- 问题具体时 retrieve；缺少决定检索方向的信息时 clarify。
- retrieve 可给出一条简短 rewritten_query；必须保留否定、顺序、平台和软件名。
- 非 in_scope 时 intent=unknown、action=clarify，且不要生成 rewritten_query。
- 不得编造事实、原因或解决方案。
"""
            ),
            HumanMessage(content=current_query),
        ],
    )

    rewritten_query = (result.rewritten_query or "").strip()
    rewritten_queries = []
    if (
        result.scope == "in_scope"
        and result.action == "retrieve"
        and rewritten_query
        and _normalize_query(rewritten_query).casefold()
        != _normalize_query(current_query).casefold()
    ):
        rewritten_queries = [rewritten_query]

    analysis = RequestAnalysis(
        relation="new_issue",
        resolved_query=current_query.strip(),
        context_reason="当前消息作为独立单轮请求处理。",
        is_self_contained=result.scope != "uncertain",
        references_active_issue=False,
        answers_last_question=False,
        explicit_new_issue=False,
        scope=result.scope,
        scope_reason="Router V2 完成单轮业务范围判断。",
        intent=result.intent,
        action=result.action,
        known_information=[],
        missing_information=(
            ["决定检索方向的具体业务对象或故障现象"]
            if result.scope == "in_scope" and result.action == "clarify"
            else []
        ),
        clarifying_question=result.clarifying_question,
        intent_reason="Router V2 完成单轮意图与动作判断。",
        rewritten_queries=rewritten_queries,
        rewrite_reason=("Router V2 生成单条检索改写。" if rewritten_queries else None),
        route_source="light_model",
    )
    return normalize_request_analysis(
        analysis,
        current_query=current_query,
        active_issue=None,
        business_scope=business_scope,
    )


def route_request(
    *,
    router_llm: BaseChatModel,
    fallback_llm: BaseChatModel,
    current_query: str,
    recent_messages: list[dict[str, str]],
    active_issue: ActiveIssue | None,
    company_name: str,
    business_scope: list[str],
    router_v2_enabled: bool | None = None,
) -> RequestAnalysis:
    """分层路由请求，并在轻量路由失败时回退完整分析器。"""
    enabled = (
        _env_flag("ROUTER_V2_ENABLED", True)
        if router_v2_enabled is None
        else router_v2_enabled
    )
    fallback_kwargs = {
        "llm": fallback_llm,
        "current_query": current_query,
        "recent_messages": recent_messages,
        "active_issue": active_issue,
        "company_name": company_name,
        "business_scope": business_scope,
    }
    if not enabled:
        return analyze_request(**fallback_kwargs).model_copy(
            update={"route_source": "legacy_full"}
        )

    fast_result = analyze_request_fast_path(
        current_query=current_query,
        recent_messages=recent_messages,
        active_issue=active_issue,
        business_scope=business_scope,
    )
    if fast_result is not None:
        return normalize_request_analysis(
            fast_result,
            current_query=current_query,
            active_issue=active_issue,
            business_scope=business_scope,
        )

    # 多轮关系、指代消解、追问回答和纠正语义继续使用原完整分析器。
    if active_issue is not None or recent_messages:
        try:
            return analyze_request(**fallback_kwargs).model_copy(
                update={"route_source": "legacy_context"}
            )
        except Exception as exc:
            logger.warning("上下文分析调用失败，使用本地会话信息降级：%s", exc)
            return _fallback_context_analysis(
                current_query=current_query,
                recent_messages=recent_messages,
                active_issue=active_issue,
                business_scope=business_scope,
            )

    try:
        return _analyze_single_turn_with_router(
            llm=router_llm,
            current_query=current_query,
            company_name=company_name,
            business_scope=business_scope,
        )
    except Exception as exc:
        logger.warning("Router V2 调用失败，回退完整请求分析器：%s", exc)
        return analyze_request(**fallback_kwargs).model_copy(
            update={"route_source": "fallback_full"}
        )
