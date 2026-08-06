import json

from back.agent.scope_analyzer import analyze_scope
from back.llm import model


TEST_CASES = [
    {
        "name": "明确业务问题",
        "query": "续费成功后流量没有重置怎么办？",
    },
    {
        "name": "模糊业务问题",
        "query": "我购买了套餐，但是不能用。",
    },
    {
        "name": "知识库可能没有答案的业务问题",
        "query": "你们支持开具增值税专用发票吗？",
    },
    {
        "name": "简单寒暄",
        "query": "你好",
    },
    {
        "name": "明确业务外问题",
        "query": "北京明天会下雨吗？",
    },
    {
        "name": "信息不足",
        "query": "怎么还是不行？",
    },
]


def main():
    for test_case in TEST_CASES:
        print(
            f"\n===== {test_case['name']} ====="
        )

        result = analyze_scope(
            llm=model,
            resolved_query=test_case["query"],
        )

        print(
            json.dumps(
                result.model_dump(),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()