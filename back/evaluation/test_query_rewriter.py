import json

from back.agent.query_rewriter import rewrite_query
from back.llm import model


TEST_CASES = [
    {
        "name": "口语化续费流量问题",
        "resolved_query": (
            "我又买了一个月，日期变了，"
            "但是剩下的流量还是没回来。"
        ),
        "intent": "traffic",
        "known_information": [
            "用户进行了续费",
            "套餐有效期已经延长",
            "流量没有恢复",
        ],
    },
    {
        "name": "客户端连接问题",
        "resolved_query": (
            "Windows上的Clash Verge显示连接成功，"
            "但是网页打不开。"
        ),
        "intent": "node_connection",
        "known_information": [
            "用户使用Windows",
            "用户使用Clash Verge",
            "客户端显示连接成功",
            "无法打开网页",
        ],
    },
    {
        "name": "首次购买未到账",
        "resolved_query": (
            "我是第一次购买，订单已经付款，"
            "但是套餐没有到账。"
        ),
        "intent": "package",
        "known_information": [
            "用户是第一次购买",
            "订单已经支付",
            "套餐没有到账",
        ],
    },
]


def main():
    for test_case in TEST_CASES:
        print(
            f"\n===== {test_case['name']} ====="
        )

        result = rewrite_query(
            llm=model,
            resolved_query=test_case["resolved_query"],
            intent=test_case["intent"],
            known_information=(
                test_case["known_information"]
            ),
        )

        assert 1 <= len(
            result.rewritten_queries
        ) <= 3

        assert all(
            query.strip()
            for query in result.rewritten_queries
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