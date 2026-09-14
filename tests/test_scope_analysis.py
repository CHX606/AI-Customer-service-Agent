import pytest
pytestmark = pytest.mark.live_model

import json

from back.agent.analysis.scope import analyze_scope
from back.core.llm import model


TEST_CASES = [
    {
        "name": "企业资料咨询",
        "query": "你们是谁？这是做什么的？",
        "expected_scope": "in_scope",
    },
    {
        "name": "营业时间咨询",
        "query": "营业时间是什么？怎么联系你们？",
        "expected_scope": "in_scope",
    },
    {
        "name": "明确业务问题",
        "query": "续费成功后流量没有重置怎么办？",
        "expected_scope": "in_scope",
    },
    {
        "name": "模糊业务问题",
        "query": "我购买了套餐，但是不能用。",
        "expected_scope": "in_scope",
    },
    {
        "name": "知识库可能没有答案的业务问题",
        "query": "你们支持开具增值税专用发票吗？",
        "expected_scope": "in_scope",
    },
    {
        "name": "简单寒暄",
        "query": "你好",
        "expected_scope": "chitchat",
    },
    {
        "name": "明确业务外问题",
        "query": "北京明天会下雨吗？",
        "expected_scope": "out_of_scope",
    },
    {
        "name": "信息不足",
        "query": "怎么还是不行？",
        "expected_scope": "uncertain",
    },
]


def test_scope_analysis():
    for test_case in TEST_CASES:
        result = analyze_scope(
            llm=model,
            resolved_query=test_case["query"],
        )
        assert result.scope == test_case["expected_scope"], (
            f"待测语句「{test_case['query']}」预期 scope={test_case['expected_scope']}，实际为 {result.scope}"
        )


if __name__ == "__main__":
    test_scope_analysis()
    print("业务范围判断测试全部通过")