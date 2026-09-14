import pytest
pytestmark = pytest.mark.live_model

"""测试企业资料咨询、意图识别与回答分支。"""

from langchain_core.messages import HumanMessage

from back.agent.workflow.response_nodes import answer_from_profile
from back.agent.workflow.graph import customer_service_graph
from back.agent.analysis.intent import analyze_intent
from back.agent.analysis.scope import analyze_scope
from back.core.llm import model
from back.domain.tenant import TenantProfile
from back.tenant.service import init_tenant_system, save_tenant_profile


COMPANY_QUESTIONS = [
    "这是干啥的",
    "你们是谁",
    "介绍一下你们",
    "营业时间是什么",
    "怎么联系你们",
]


def test_company_info_scope_and_intent():
    for query in COMPANY_QUESTIONS:
        scope_res = analyze_scope(llm=model, resolved_query=query)
        assert scope_res.scope == "in_scope", (
            f"问题「{query}」应被判定为 in_scope，实际判定为 {scope_res.scope}"
        )

        intent_res = analyze_intent(llm=model, resolved_query=query)
        assert intent_res.intent == "company_info", (
            f"问题「{query}」意图应为 company_info，实际为 {intent_res.intent}"
        )
        assert intent_res.action == "profile", (
            f"问题「{query}」action 应为 profile，实际为 {intent_res.action}"
        )


def test_answer_from_profile_node():
    init_tenant_system()

    res = answer_from_profile(
        {
            "tenant_id": "default",
            "resolved_query": "你们是做什么的？",
            "messages": [HumanMessage(content="你们是做什么的？")],
        }
    )

    answer = res["messages"][0].content
    assert "网络" in answer or "可乐云" in answer or "加速" in answer
    assert "知识库暂未查到" not in answer


def test_answer_from_profile_missing_config():
    # 创建空资料租户
    empty_profile = TenantProfile(
        tenant_id="empty_tenant",
        company_name="空壳科技",
        brand_name_en=None,
        assistant_name="空壳客服",
        short_description="",
        business_scope=[],
        business_hours=None,
        public_contact=None,
        welcome_title="你好",
        welcome_description="欢迎",
        tone="客观",
        handoff_message="转人工",
        suggested_questions=[],
    )
    save_tenant_profile(empty_profile)

    res = answer_from_profile(
        {
            "tenant_id": "empty_tenant",
            "resolved_query": "你们的营业时间是什么？",
            "messages": [HumanMessage(content="你们的营业时间是什么？")],
        }
    )

    answer = res["messages"][0].content
    assert "未配置" in answer or "暂无" in answer or "尚未" in answer
