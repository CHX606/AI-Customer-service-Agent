"""客服请求分析职责模块：request_rules。"""
import os
import re
from typing import Literal
from back.agent.workflow.lifecycle import decide_relation_with_lifecycle
from back.agent.analysis.scope import _is_underspecified_query, _normalize_query
from back.domain.conversation import ActiveIssue, IntentAction, IntentType
from back.agent.analysis.request_models import RequestAnalysis

_PROFILE_QUERY = re.compile(
    r"(?:你好|您好)?(?:请问|请|麻烦|能否|可以)?(?:"
    r"这是干啥的|这是做什么的|你们是谁|你们是做什么的|"
    r"介绍一下(?:你们(?:的)?(?:公司|企业)?|公司|企业)|"
    r"(?:你们的?|公司的?|企业的?|客服的?)?"
    r"(?:营业时间|联系方式|客服电话)(?:是什么|是多少|是啥)?|"
    r"怎么联系你们|(?:联系|转|找)?人工客服"
    r")(?:吗|呢|呀|啊)?"
)

_VAGUE_BUSINESS_FAILURES = {
    "节点不能用",
    "节点用不了",
    "节点有问题",
    "套餐不能用",
    "套餐用不了",
}

_EXTERNAL_SERVICE_TERMS = (
    "x.com",
    "twitter",
    "推特",
    "google",
    "谷歌",
    "gmail",
    "youtube",
    "telegram",
    "instagram",
    "facebook",
    "discord",
    "reddit",
    "github",
    "chatgpt",
)
_UNSAFE_EXTERNAL_TERMS = (
    "绕过",
    "规避风控",
    "避开风控",
    "批量注册",
    "伪造ip",
    "伪造位置",
    "盗号",
    "盗回",
)
_EXTERNAL_SCOPE_TERMS = (
    "外网应用",
    "外网网站",
    "twitter",
    "google",
    "youtube",
    "telegram",
)

_CHITCHAT_QUERIES = {
    "你好",
    "您好",
    "嗨",
    "hello",
    "hi",
    "谢谢",
    "谢谢你",
    "谢谢你的帮助",
    "谢谢帮助",
    "好的谢谢",
    "感谢",
    "非常感谢",
    "多谢",
    "辛苦了",
    "再见",
    "再见了",
    "拜拜",
    "晚安",
}

_OUT_OF_SCOPE_TERMS = (
    "天气",
    "下雨",
    "气温",
    "晴天",
    "新闻",
    "股票",
    "菜谱",
    "做饭",
    "写代码",
    "编程题",
    "数学题",
    "高考",
    "世界杯",
)

_GENERIC_FAILURE_TERMS = (
    "不能用",
    "用不了",
    "有问题",
    "不可以用",
)

_FAILURE_DETAIL_TERMS = (
    "错误",
    "报错",
    "状态",
    "超时",
    "延迟",
    "连接",
    "网页",
    "到账",
    "流量",
    "订阅",
    "支付",
    "成功",
    "失败",
    "显示",
)

# 顺序同时代表多意图问题的主意图优先级。例如“续费后流量未重置”
# 应按流量问题检索，而不是只按续费规则检索。
_INTENT_RULES: tuple[
    tuple[IntentType, tuple[str, ...], tuple[str, ...]], ...
] = (
    ("traffic", ("流量", "g数", "gb"), ("流量",)),
    ("refund", ("退款", "退钱", "原路退"), ("退款", "售后")),
    (
        "subscription_import",
        ("订阅", "导入", "一键导入", "订阅链接"),
        ("订阅",),
    ),
    (
        "node_connection",
        ("节点", "线路", "无法上网", "网页打不开", "连接超时", "代理"),
        ("节点", "网络", "连接"),
    ),
    (
        "software_usage",
        (
            "clash",
            "shadowrocket",
            "小火箭",
            "客户端",
            "软件下载",
            "安装",
            "配置",
        ),
        ("客户端", "软件", "安装", "配置"),
    ),
    (
        "account",
        ("账号", "密码", "验证码", "注册", "登录"),
        ("账号", "登录"),
    ),
    (
        "order_payment",
        ("订单", "支付", "扣款", "付款", "付完钱", "付了钱", "付钱", "下单"),
        ("订单", "支付", "购买"),
    ),
    ("renewal", ("续费", "续订", "有效期延长"), ("续费",)),
    ("package", ("套餐", "换套餐"), ("套餐",)),
)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}


def _is_profile_query(query: str, *, allow_generic_intro: bool = True) -> bool:
    """仅对完整的企业资料提问走捷径，避免覆盖具体业务诉求。"""
    normalized = _normalize_query(query)
    if allow_generic_intro and normalized == "介绍一下":
        return True
    return _PROFILE_QUERY.fullmatch(normalized) is not None


def _scope_supports(
    business_scope: list[str],
    scope_terms: tuple[str, ...],
) -> bool:
    normalized_scope = _normalize_query("、".join(business_scope)).casefold()
    return any(term.casefold() in normalized_scope for term in scope_terms)


def _classify_supported_intent(
    query: str,
    business_scope: list[str],
) -> IntentType | None:
    normalized_query = _normalize_query(query).casefold()
    for intent, query_terms, scope_terms in _INTENT_RULES:
        if not any(term.casefold() in normalized_query for term in query_terms):
            continue
        if _scope_supports(business_scope, scope_terms):
            return intent
    return None


def _is_vague_business_failure(query: str) -> bool:
    normalized = _normalize_query(query).casefold()
    if normalized in {
        _normalize_query(item).casefold() for item in _VAGUE_BUSINESS_FAILURES
    }:
        return True
    if not any(target in normalized for target in ("节点", "套餐")):
        return False
    if not any(term in normalized for term in _GENERIC_FAILURE_TERMS):
        return False
    return not any(term in normalized for term in _FAILURE_DETAIL_TERMS)


def _direct_analysis(
    *,
    current_query: str,
    scope: Literal["in_scope", "chitchat", "out_of_scope", "uncertain"],
    scope_reason: str,
    intent: IntentType = "unknown",
    action: IntentAction = "clarify",
    clarifying_question: str | None = None,
    missing_information: list[str] | None = None,
    self_contained: bool = True,
) -> RequestAnalysis:
    """构造无需模型调用的单轮分析结果。"""
    return RequestAnalysis(
        relation="new_issue",
        resolved_query=current_query.strip(),
        context_reason="当前消息可作为独立的新问题处理。",
        is_self_contained=self_contained,
        references_active_issue=False,
        answers_last_question=False,
        explicit_new_issue=False,
        scope=scope,
        scope_reason=scope_reason,
        intent=intent,
        action=action,
        known_information=[],
        missing_information=missing_information or [],
        clarifying_question=clarifying_question,
        intent_reason="Router V2 根据确定性规则完成单轮路由。",
        rewritten_queries=[],
        rewrite_reason=None,
        route_source="fast_rule",
    )


def analyze_request_fast_path(
    *,
    current_query: str,
    recent_messages: list[dict[str, str]],
    active_issue: ActiveIssue | None,
    business_scope: list[str],
) -> RequestAnalysis | None:
    """仅处理高置信度单轮请求；涉及上下文时必须返回 None。"""
    if active_issue is not None or recent_messages:
        return None

    normalized = _normalize_query(current_query).casefold()
    normalized_scope = _normalize_query("、".join(business_scope)).casefold()
    supports_external_apps = any(
        term in normalized_scope for term in _EXTERNAL_SCOPE_TERMS
    )
    is_external_query = any(
        term in normalized for term in _EXTERNAL_SERVICE_TERMS
    )

    if _is_profile_query(current_query):
        return _direct_analysis(
            current_query=current_query,
            scope="in_scope",
            scope_reason="用户正在咨询企业固定资料。",
            intent="company_info",
            action="profile",
        )

    if _is_vague_business_failure(current_query):
        is_node = "节点" in normalized
        return _direct_analysis(
            current_query=current_query,
            scope="in_scope",
            scope_reason="属于业务问题，但故障现象不足以确定检索方向。",
            intent="node_connection" if is_node else "package",
            action="clarify",
            clarifying_question=(
                "请问是所有节点都不能用，还是某个节点不能用？当前显示什么错误或状态？"
                if is_node
                else "请问套餐是否已经到账，具体在哪一步不能使用？"
            ),
            missing_information=["具体故障范围和页面状态"],
        )

    if _is_underspecified_query(current_query):
        return _direct_analysis(
            current_query=current_query,
            scope="uncertain",
            scope_reason="问题未包含具体业务对象或可判断的故障现象，需要补充信息。",
            clarifying_question="可以具体说说遇到的是哪项业务、出现了什么现象吗？",
            missing_information=["具体业务对象或故障现象"],
            self_contained=False,
        )

    if normalized in _CHITCHAT_QUERIES:
        return _direct_analysis(
            current_query=current_query,
            scope="chitchat",
            scope_reason="属于简单问候、感谢或告别。",
        )

    if any(term in normalized for term in _UNSAFE_EXTERNAL_TERMS):
        return _direct_analysis(
            current_query=current_query,
            scope="out_of_scope",
            scope_reason="请求涉及规避平台安全机制，不属于可支持的客服范围。",
        )

    if is_external_query and supports_external_apps:
        return _direct_analysis(
            current_query=current_query,
            scope="in_scope",
            scope_reason="属于企业明确支持的外网应用使用与排障范围。",
            intent="other_business",
            action="retrieve",
        )

    intent = _classify_supported_intent(current_query, business_scope)
    has_out_of_scope_term = any(term in normalized for term in _OUT_OF_SCOPE_TERMS)
    if intent is not None and has_out_of_scope_term:
        # 混合问题交给模型判断，避免关键词规则粗暴覆盖用户真实意图。
        return None
    if intent is not None:
        return _direct_analysis(
            current_query=current_query,
            scope="in_scope",
            scope_reason="问题包含租户已配置业务范围内的明确对象。",
            intent=intent,
            action="retrieve",
        )
    if has_out_of_scope_term:
        return _direct_analysis(
            current_query=current_query,
            scope="out_of_scope",
            scope_reason="问题与企业已配置业务范围无关。",
        )
    return None


def build_search_queries(analysis: RequestAnalysis) -> list[str]:
    """仅为需要检索的请求生成去重查询列表。"""
    if analysis.scope != "in_scope" or analysis.action != "retrieve":
        return []

    queries: list[str] = []
    seen: set[str] = set()
    for query in [analysis.resolved_query, *analysis.rewritten_queries]:
        cleaned = " ".join(query.split())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        queries.append(cleaned)
    return queries


def normalize_request_analysis(
    analysis: RequestAnalysis,
    *,
    current_query: str,
    active_issue: ActiveIssue | None,
    business_scope: list[str] | None = None,
) -> RequestAnalysis:
    """用确定性规则修正模型输出中的生命周期和字段一致性。"""
    final_relation = decide_relation_with_lifecycle(
        model_relation=analysis.relation,
        active_issue=active_issue,
        is_self_contained=analysis.is_self_contained,
        references_active_issue=analysis.references_active_issue,
        answers_last_question=analysis.answers_last_question,
        explicit_new_issue=analysis.explicit_new_issue,
    )

    resolved_query = analysis.resolved_query.strip() or current_query.strip()
    context_reason = analysis.context_reason
    if final_relation != analysis.relation:
        if final_relation == "new_issue":
            resolved_query = current_query.strip()
        context_reason = f"{context_reason} 系统结合问题生命周期修正了上下文关系。".strip()

    updates: dict[str, object] = {
        "relation": final_relation,
        "resolved_query": resolved_query,
        "context_reason": context_reason,
    }

    normalized_current = _normalize_query(current_query)
    normalized_external_query = normalized_current.casefold()
    normalized_scope = "、".join(business_scope or []).casefold()
    is_external_service_query = any(
        term in normalized_external_query for term in _EXTERNAL_SERVICE_TERMS
    )
    supports_external_apps = any(
        term in normalized_scope for term in _EXTERNAL_SCOPE_TERMS
    )
    is_unsafe_external_query = any(
        term in normalized_external_query for term in _UNSAFE_EXTERNAL_TERMS
    )
    if is_unsafe_external_query:
        updates.update(
            {
                "resolved_query": current_query.strip(),
                "scope": "out_of_scope",
                "scope_reason": "请求涉及规避或破坏平台安全机制，不属于可支持的客服范围。",
                "intent": "unknown",
                "action": "clarify",
                "known_information": [],
                "missing_information": [],
                "clarifying_question": None,
                "intent_reason": "安全边界请求不进入业务检索。",
                "rewritten_queries": [],
                "rewrite_reason": None,
            }
        )
    elif _is_profile_query(current_query, allow_generic_intro=active_issue is None):
        updates.update(
            {
                "resolved_query": current_query.strip(),
                "scope": "in_scope",
                "scope_reason": "用户正在咨询企业固定资料。",
                "intent": "company_info",
                "action": "profile",
                "known_information": [],
                "missing_information": [],
                "clarifying_question": None,
                "intent_reason": "企业资料问题直接读取租户配置。",
                "rewritten_queries": [],
                "rewrite_reason": None,
            }
        )
    elif normalized_current in _VAGUE_BUSINESS_FAILURES:
        question = (
            "请问是所有节点都不能用，还是某个节点不能用？当前显示什么错误或状态？"
            if "节点" in normalized_current
            else "请问套餐是否已经到账，具体在哪一步不能使用？"
        )
        updates.update(
            {
                "scope": "in_scope",
                "scope_reason": "属于业务问题，但故障现象不足以确定检索方向。",
                "action": "clarify",
                "missing_information": ["具体故障范围和页面状态"],
                "clarifying_question": question,
                "intent_reason": "需要先确认具体故障现象。",
                "rewritten_queries": [],
                "rewrite_reason": None,
            }
        )
    elif (
        is_external_service_query
        and supports_external_apps
        and not is_unsafe_external_query
    ):
        # 平台名称只修正业务范围，不替换模型已整合的设备、现象和追问信息。
        needs_clarification = analysis.scope == "in_scope" and analysis.action == "clarify"
        reset_context = final_relation == "new_issue" and final_relation != analysis.relation
        updates.update(
            {
                "resolved_query": resolved_query,
                "scope": "in_scope",
                "scope_reason": "属于企业明确支持的外网应用使用与排障范围。",
                "intent": "other_business",
                "action": "clarify" if needs_clarification else "retrieve",
                "known_information": [] if reset_context else analysis.known_information,
                "missing_information": analysis.missing_information if needs_clarification else [],
                "clarifying_question": (
                    analysis.clarifying_question or "可以具体说明遇到的故障现象吗？"
                ) if needs_clarification else None,
                "intent_reason": (
                    analysis.intent_reason if needs_clarification
                    else "需要检索外网应用使用与排障知识库。"
                ),
                "rewritten_queries": [] if needs_clarification or reset_context else analysis.rewritten_queries,
                "rewrite_reason": None if needs_clarification or reset_context else analysis.rewrite_reason,
            }
        )
    elif _is_underspecified_query(resolved_query):
        updates.update(
            {
                "scope": "uncertain",
                "scope_reason": "问题未包含具体业务对象或可判断的故障现象，需要补充信息。",
                "intent": "unknown",
                "action": "clarify",
                "known_information": [],
                "missing_information": ["具体业务对象或故障现象"],
                "clarifying_question": "可以具体说说遇到的是哪项业务、出现了什么现象吗？",
                "intent_reason": "当前问题信息不足。",
                "rewritten_queries": [],
                "rewrite_reason": None,
            }
        )
    elif analysis.scope != "in_scope":
        updates.update(
            {
                "intent": "unknown",
                "action": "clarify",
                "known_information": [],
                "missing_information": [],
                "clarifying_question": None,
                "intent_reason": "该分支不需要执行业务意图分析。",
                "rewritten_queries": [],
                "rewrite_reason": None,
            }
        )
    elif analysis.action == "profile":
        updates.update(
            {
                "intent": "company_info",
                "missing_information": [],
                "clarifying_question": None,
                "rewritten_queries": [],
                "rewrite_reason": None,
            }
        )
    elif analysis.action == "clarify":
        updates.update(
            {
                "rewritten_queries": [],
                "rewrite_reason": None,
                "clarifying_question": (
                    analysis.clarifying_question
                    or "可以具体说说您遇到的是哪一种情况吗？"
                ),
            }
        )
    else:
        updates.update(
            {
                "missing_information": [],
                "clarifying_question": None,
            }
        )

    return analysis.model_copy(update=updates)
