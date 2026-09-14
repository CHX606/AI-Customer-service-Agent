"""合并请求分析器的离线单元测试。"""

import pytest
from pydantic import ValidationError

from back.agent.analysis.request import (
    RequestAnalysis,
    RouteDecision,
    analyze_request,
    analyze_request_fast_path,
    build_search_queries,
    normalize_request_analysis,
    route_request,
)


def make_analysis(**updates) -> RequestAnalysis:
    data = {
        "relation": "new_issue",
        "resolved_query": "续费后流量没有重置",
        "context_reason": "当前消息是完整的新问题。",
        "is_self_contained": True,
        "references_active_issue": False,
        "answers_last_question": False,
        "explicit_new_issue": False,
        "scope": "in_scope",
        "scope_reason": "属于流量业务问题。",
        "intent": "traffic",
        "action": "retrieve",
        "known_information": ["已经续费", "流量没有重置"],
        "missing_information": [],
        "clarifying_question": None,
        "intent_reason": "问题足以检索。",
        "rewritten_queries": ["续费成功后流量未重置怎么办"],
        "rewrite_reason": "标准化口语表达。",
    }
    data.update(updates)
    return RequestAnalysis(**data)


class StructuredRunner:
    def __init__(self, parent, result):
        self.parent = parent
        self.result = result

    def invoke(self, messages):
        self.parent.calls += 1
        self.parent.messages = messages
        return self.result


class FakeStructuredModel:
    def __init__(self, result):
        self.result = result
        self.calls = 0
        self.schema = None
        self.messages = []

    def with_structured_output(self, schema):
        self.schema = schema
        return StructuredRunner(self, self.result)


class FailingStructuredRunner:
    def invoke(self, _messages):
        raise TimeoutError("router timeout")


class FailingStructuredModel:
    def with_structured_output(self, _schema):
        return FailingStructuredRunner()


class RawTextRunner:
    def __init__(self, result):
        self.result = result

    def invoke(self, _messages):
        return self.result


class RawTextStructuredModel:
    def __init__(self, text):
        self.text = text

    def with_structured_output(self, _schema, *, include_raw=False):
        assert include_raw is True
        raw = type("RawMessage", (), {"content": [{"type": "text", "text": self.text}]})()
        return RawTextRunner(
            {
                "raw": raw,
                "parsed": None,
                "parsing_error": ValueError("provider returned text JSON"),
            }
        )


def test_analyze_request_uses_exactly_one_model_call():
    fake = FakeStructuredModel(make_analysis())
    result = analyze_request(
        llm=fake,
        current_query="续费后流量怎么没恢复？",
        recent_messages=[],
        active_issue=None,
        company_name="测试企业",
        business_scope=["套餐", "流量"],
    )

    assert fake.calls == 1
    assert fake.schema is RequestAnalysis
    assert result.relation == "new_issue"
    assert "测试企业" in fake.messages[0].content
    assert "套餐、流量" in fake.messages[0].content
    assert "续费后流量怎么没恢复" in fake.messages[1].content


def test_analyze_request_accepts_text_json_from_responses_api():
    raw = make_analysis(
        relation="continue",
        route_source="fast_rule",
    ).model_dump_json()
    result = analyze_request(
        llm=RawTextStructuredModel(raw),
        current_query="Windows，用的是 Clash Verge",
        recent_messages=[{"role": "assistant", "content": "请问使用什么设备？"}],
        active_issue={
            "summary": "客户端不能用",
            "status": "awaiting_user",
            "last_clarifying_question": "请问使用什么设备？",
        },
        company_name="测试企业",
        business_scope=["客户端", "节点"],
    )

    assert result.relation == "continue"
    assert result.route_source == "fast_rule"


@pytest.mark.parametrize(
    ("query", "scope", "action", "intent"),
    [
        ("营业时间是什么？", "in_scope", "profile", "company_info"),
        ("续费后流量怎么没恢复？", "in_scope", "retrieve", "traffic"),
        ("节点不能用", "in_scope", "clarify", "node_connection"),
        ("北京明天天气怎么样？", "out_of_scope", "clarify", "unknown"),
        ("你好", "chitchat", "clarify", "unknown"),
    ],
)
def test_fast_path_handles_only_deterministic_single_turns(
    query,
    scope,
    action,
    intent,
):
    result = analyze_request_fast_path(
        current_query=query,
        recent_messages=[],
        active_issue=None,
        business_scope=["账号", "套餐", "续费", "流量", "节点"],
    )

    assert result is not None
    assert result.scope == scope
    assert result.action == action
    assert result.intent == intent


def test_fast_path_refuses_contextual_requests():
    result = analyze_request_fast_path(
        current_query="还是不行",
        recent_messages=[{"role": "assistant", "content": "请问显示什么错误？"}],
        active_issue={"summary": "节点连接失败", "status": "awaiting_user"},
        business_scope=["节点连接"],
    )
    assert result is None


def test_route_request_fast_path_makes_no_model_call():
    router = FakeStructuredModel(
        RouteDecision(scope="in_scope", intent="traffic", action="retrieve")
    )
    fallback = FakeStructuredModel(make_analysis())

    result = route_request(
        router_llm=router,
        fallback_llm=fallback,
        current_query="续费后流量怎么没恢复？",
        recent_messages=[],
        active_issue=None,
        company_name="测试企业",
        business_scope=["套餐续费与流量重置"],
        router_v2_enabled=True,
    )

    assert result.action == "retrieve"
    assert result.intent == "traffic"
    assert result.route_source == "fast_rule"
    assert router.calls == 0
    assert fallback.calls == 0


def test_fast_path_routes_completed_payment_without_effect_to_retrieval():
    result = analyze_request_fast_path(
        current_query="我刚付完钱，怎么还是什么变化都没有？",
        recent_messages=[],
        active_issue=None,
        business_scope=["套餐购买与订单支付"],
    )

    assert result is not None
    assert result.scope == "in_scope"
    assert result.action == "retrieve"
    assert result.intent == "order_payment"


def test_route_request_uses_minimal_schema_for_unmatched_single_turn():
    router = FakeStructuredModel(
        RouteDecision(
            scope="in_scope",
            intent="other_business",
            action="retrieve",
            rewritten_query="产品家庭共享功能",
        )
    )
    fallback = FakeStructuredModel(make_analysis())

    result = route_request(
        router_llm=router,
        fallback_llm=fallback,
        current_query="你们的产品支持家庭共享吗？",
        recent_messages=[],
        active_issue=None,
        company_name="测试企业",
        business_scope=["产品功能咨询"],
        router_v2_enabled=True,
    )

    assert router.calls == 1
    assert router.schema is RouteDecision
    assert fallback.calls == 0
    assert result.rewritten_queries == ["产品家庭共享功能"]
    assert result.route_source == "light_model"


def test_route_request_keeps_full_analysis_for_contextual_turns():
    router = FakeStructuredModel(
        RouteDecision(scope="in_scope", intent="package", action="retrieve")
    )
    fallback = FakeStructuredModel(
        make_analysis(
            relation="continue",
            answers_last_question=True,
            references_active_issue=True,
            resolved_query="套餐已到账但仍不能使用",
        )
    )

    result = route_request(
        router_llm=router,
        fallback_llm=fallback,
        current_query="已经到账了，但还是不能用",
        recent_messages=[{"role": "assistant", "content": "套餐到账了吗？"}],
        active_issue={"summary": "套餐不能使用", "status": "awaiting_user"},
        company_name="测试企业",
        business_scope=["套餐"],
        router_v2_enabled=True,
    )

    assert router.calls == 0
    assert fallback.calls == 1
    assert fallback.schema is RequestAnalysis
    assert result.relation == "continue"
    assert result.route_source == "legacy_context"


def test_route_request_uses_local_context_when_context_model_fails():
    result = route_request(
        router_llm=FailingStructuredModel(),
        fallback_llm=FailingStructuredModel(),
        current_query=(
            "只有 ChatGPT 登录页一直空白，其他外网网站正常，"
            "Windows 11 的 Chrome，节点延迟正常。"
        ),
        recent_messages=[
            {"role": "user", "content": "网站打不开。"},
            {"role": "assistant", "content": "请问是哪个网站打不开？"},
        ],
        active_issue={
            "summary": "网站打不开",
            "status": "awaiting_user",
            "intent": "other_business",
            "last_clarifying_question": "请问是哪个网站打不开？",
        },
        company_name="测试企业",
        business_scope=["外网应用使用与排障"],
        router_v2_enabled=True,
    )

    assert result.route_source == "fallback_context"
    assert result.relation == "continue"
    assert result.scope == "in_scope"
    assert result.action == "retrieve"
    assert result.intent == "other_business"
    assert "ChatGPT" in result.resolved_query


def test_context_fallback_can_start_an_obvious_different_issue():
    result = route_request(
        router_llm=FailingStructuredModel(),
        fallback_llm=FailingStructuredModel(),
        current_query="我想申请退款。",
        recent_messages=[
            {"role": "user", "content": "节点连接不上。"},
        ],
        active_issue={
            "summary": "节点连接失败",
            "status": "answered",
            "intent": "node_connection",
        },
        company_name="测试企业",
        business_scope=["节点连接", "退款售后"],
        router_v2_enabled=True,
    )

    assert result.route_source == "fallback_context"
    assert result.relation == "new_issue"
    assert result.intent == "refund"
    assert result.resolved_query == "我想申请退款。"


def test_route_request_falls_back_when_minimal_router_fails():
    fallback = FakeStructuredModel(make_analysis())
    result = route_request(
        router_llm=FailingStructuredModel(),
        fallback_llm=fallback,
        current_query="你们的产品支持家庭共享吗？",
        recent_messages=[],
        active_issue=None,
        company_name="测试企业",
        business_scope=["产品功能咨询"],
        router_v2_enabled=True,
    )

    assert fallback.calls == 1
    assert fallback.schema is RequestAnalysis
    assert result.scope == "in_scope"
    assert result.route_source == "fallback_full"


def test_route_request_can_disable_v2_for_one_click_rollback():
    router = FakeStructuredModel(
        RouteDecision(scope="in_scope", intent="traffic", action="retrieve")
    )
    fallback = FakeStructuredModel(make_analysis())
    result = route_request(
        router_llm=router,
        fallback_llm=fallback,
        current_query="续费后流量怎么没恢复？",
        recent_messages=[],
        active_issue=None,
        company_name="测试企业",
        business_scope=["套餐续费与流量重置"],
        router_v2_enabled=False,
    )

    assert router.calls == 0
    assert fallback.calls == 1
    assert fallback.schema is RequestAnalysis
    assert result.route_source == "legacy_full"


def test_no_active_issue_forces_new_issue_and_current_query():
    analysis = make_analysis(
        relation="continue",
        resolved_query="混入了旧问题",
        is_self_contained=True,
    )
    result = normalize_request_analysis(
        analysis,
        current_query="新的退款问题",
        active_issue=None,
    )
    assert result.relation == "new_issue"
    assert result.resolved_query == "新的退款问题"
    assert "生命周期" in result.context_reason


def test_explicit_switch_forces_new_issue():
    result = normalize_request_analysis(
        make_analysis(
            relation="continue",
            explicit_new_issue=True,
            resolved_query="旧问题和退款问题",
        ),
        current_query="另外，支持退款吗？",
        active_issue={"summary": "流量异常", "status": "open"},
    )
    assert result.relation == "new_issue"
    assert result.resolved_query == "另外，支持退款吗？"


def test_reference_to_active_issue_forces_continue():
    result = normalize_request_analysis(
        make_analysis(
            relation="new_issue",
            references_active_issue=True,
            is_self_contained=False,
        ),
        current_query="这个还是不行",
        active_issue={"summary": "节点无法连接", "status": "answered"},
    )
    assert result.relation == "continue"


def test_correction_is_preserved_when_referencing_active_issue():
    result = normalize_request_analysis(
        make_analysis(relation="correction", references_active_issue=True),
        current_query="不是续费，是第一次购买",
        active_issue={"summary": "续费未到账", "status": "open"},
    )
    assert result.relation == "correction"


def test_answer_to_awaiting_question_forces_continue():
    result = normalize_request_analysis(
        make_analysis(
            relation="new_issue",
            answers_last_question=True,
            is_self_contained=False,
        ),
        current_query="已经到账了",
        active_issue={"summary": "套餐未到账", "status": "awaiting_user"},
    )
    assert result.relation == "continue"


@pytest.mark.parametrize("status", ["resolved", "handed_off"])
def test_closed_issue_with_complete_query_starts_new_issue(status):
    result = normalize_request_analysis(
        make_analysis(relation="continue", is_self_contained=True),
        current_query="如何修改密码？",
        active_issue={"summary": "旧问题", "status": status},
    )
    assert result.relation == "new_issue"
    assert result.resolved_query == "如何修改密码？"


@pytest.mark.parametrize(
    "query",
    ["不行", "还是不行", "怎么还是不行？", "有问题", "怎么办"],
)
def test_underspecified_query_is_forced_to_uncertain(query):
    result = normalize_request_analysis(
        make_analysis(resolved_query=query),
        current_query=query,
        active_issue=None,
    )
    assert result.scope == "uncertain"
    assert result.action == "clarify"
    assert result.intent == "unknown"
    assert result.clarifying_question
    assert result.rewritten_queries == []


@pytest.mark.parametrize("scope", ["chitchat", "out_of_scope", "uncertain"])
def test_non_business_scope_clears_business_fields(scope):
    result = normalize_request_analysis(
        make_analysis(scope=scope),
        current_query="你好" if scope == "chitchat" else "北京天气",
        active_issue=None,
    )
    assert result.intent == "unknown"
    assert result.action == "clarify"
    assert result.known_information == []
    assert result.rewritten_queries == []
    assert build_search_queries(result) == []


def test_profile_action_is_coerced_to_company_info():
    result = normalize_request_analysis(
        make_analysis(
            intent="other_business",
            action="profile",
            rewritten_queries=["不应检索"],
            clarifying_question="不应追问",
        ),
        current_query="营业时间是什么？",
        active_issue=None,
    )
    assert result.intent == "company_info"
    assert result.clarifying_question is None
    assert result.rewritten_queries == []
    assert build_search_queries(result) == []


def test_clarify_action_supplies_safe_fallback_question():
    result = normalize_request_analysis(
        make_analysis(action="clarify", clarifying_question=None),
        current_query="支付出现异常",
        active_issue=None,
    )
    assert result.clarifying_question == "可以具体说说您遇到的是哪一种情况吗？"
    assert result.rewritten_queries == []


def test_retrieve_action_clears_inconsistent_missing_fields():
    result = normalize_request_analysis(
        make_analysis(
            action="retrieve",
            missing_information=["错误的缺失项"],
            clarifying_question="错误的追问",
        ),
        current_query="续费后流量没有重置",
        active_issue=None,
    )
    assert result.missing_information == []
    assert result.clarifying_question is None


def test_search_queries_keep_original_first_and_remove_duplicates():
    analysis = make_analysis(
        resolved_query=" 续费后流量没有重置 ",
        rewritten_queries=[
            "续费后流量没有重置",
            "续费成功后流量未刷新",
        ],
    )
    assert build_search_queries(analysis) == [
        "续费后流量没有重置",
        "续费成功后流量未刷新",
    ]


def test_rewritten_query_schema_rejects_more_than_two_queries():
    with pytest.raises(ValidationError):
        make_analysis(rewritten_queries=["一", "二", "三"])


@pytest.mark.parametrize(
    "query",
    [
        "这是干啥的？",
        "你们是谁？",
        "介绍一下你们",
        "营业时间是什么？",
        "怎么联系你们？",
        "你们的联系方式是什么？",
    ],
)
def test_company_profile_phrases_override_unstable_model_scope(query):
    result = normalize_request_analysis(
        make_analysis(
            scope="uncertain",
            intent="unknown",
            action="clarify",
            resolved_query=query,
            rewritten_queries=[],
        ),
        current_query=query,
        active_issue=None,
    )
    assert result.scope == "in_scope"
    assert result.intent == "company_info"
    assert result.action == "profile"
    assert result.clarifying_question is None


@pytest.mark.parametrize(
    "query",
    ["节点不能用", "节点用不了", "节点有问题", "套餐不能用"],
)
def test_vague_business_failure_overrides_overeager_retrieval(query):
    result = normalize_request_analysis(
        make_analysis(
            scope="in_scope",
            action="retrieve",
            resolved_query=f"{query}，无法正常使用",
        ),
        current_query=query,
        active_issue=None,
    )
    assert result.scope == "in_scope"
    assert result.action == "clarify"
    assert result.rewritten_queries == []
    assert result.clarifying_question


@pytest.mark.parametrize(
    "query",
    [
        "怎么登录X，也就是Twitter？",
        "Google登录时一直做人机验证",
        "YouTube视频一直缓冲",
        "Telegram一直显示Connecting",
        "GitHub的git clone超时",
    ],
)
def test_declared_external_app_scope_forces_retrieval(query):
    result = normalize_request_analysis(
        make_analysis(
            scope="out_of_scope",
            action="clarify",
            intent="unknown",
            resolved_query=query,
            rewritten_queries=[],
        ),
        current_query=query,
        active_issue=None,
        business_scope=["X/Twitter、Google、YouTube、Telegram等外网应用的连接与排障"],
    )

    assert result.scope == "in_scope"
    assert result.action == "retrieve"
    assert result.intent == "other_business"
    assert build_search_queries(result) == [query]


def test_unsafe_external_app_request_is_not_forced_into_retrieval():
    query = "教我批量注册Twitter小号并避开风控"
    result = normalize_request_analysis(
        make_analysis(
            scope="out_of_scope",
            action="clarify",
            intent="unknown",
            resolved_query=query,
            rewritten_queries=[],
        ),
        current_query=query,
        active_issue=None,
        business_scope=["X/Twitter等外网应用的连接与排障"],
    )

    assert result.scope == "out_of_scope"
    assert result.action == "clarify"
    assert build_search_queries(result) == []


@pytest.mark.parametrize(
    "query",
    [
        "怎么绕过X的手机验证码和身份验证？",
        "教我批量注册Twitter小号并避开风控。",
        "怎么伪造IP位置骗过平台账号安全检测？",
        "给我一个能盗回别人Telegram账号的方法。",
    ],
)
def test_fast_path_blocks_platform_security_bypass_requests(query):
    result = analyze_request_fast_path(
        current_query=query,
        recent_messages=[],
        active_issue=None,
        business_scope=["账号登录", "外网应用连接与排障"],
    )

    assert result is not None
    assert result.scope == "out_of_scope"
    assert result.action == "clarify"
    assert result.intent == "unknown"
    assert build_search_queries(result) == []
