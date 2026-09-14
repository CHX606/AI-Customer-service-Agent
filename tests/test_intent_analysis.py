import pytest
pytestmark = pytest.mark.live_model

import json

from back.agent.analysis.intent import analyze_intent
from back.core.llm import model


TEST_CASES = [
    {
        "name": "企业介绍资料",
        "query": "这是干啥的？介绍一下你们平台。",
        "expected_action": "profile",
    },
    {
        "name": "明确流量问题",
        "query": "续费成功后套餐有效期延长了，但流量没有重置。",
        "expected_action": "retrieve",
    },
    {
        "name": "模糊套餐问题",
        "query": "我购买了套餐，但是不能用。",
        "expected_action": "clarify",
    },
    {
        "name": "明确到账问题",
        "query": "订单已经支付成功，但是套餐一直没有到账。",
        "expected_action": "retrieve",
    },
    {
        "name": "模糊连接问题",
        "query": "节点不能用。",
        "expected_action": "clarify",
    },
    {
        "name": "明确连接问题",
        "query": "Windows上的Clash Verge显示连接成功，但无法打开网页。",
        "expected_action": "retrieve",
    },
    {
        "name": "明确退款问题",
        "query": "购买套餐后可以申请退款吗？",
        "expected_action": "retrieve",
    },
]


def test_intent_analysis():
    for test_case in TEST_CASES:
        result = analyze_intent(
            llm=model,
            resolved_query=test_case["query"],
        )
        assert result.action == test_case["expected_action"], (
            f"预期action={test_case['expected_action']}，"
            f"实际action={result.action}"
        )


if __name__ == "__main__":
    test_intent_analysis()
    print("意图分析测试全部通过")
