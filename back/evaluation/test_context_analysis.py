import json

from back.agent.query_analyzer import analyze_context
from back.llm import model


test_cases = [
    {
        "name": "首次提出问题",
        "active_issue": None,
        "recent_messages": [],
        "current_query": "续费以后流量为什么没有恢复？",
    },
    {
        "name": "继续当前问题",
        "active_issue": {
            "summary": "用户购买套餐后无法使用",
        },
        "recent_messages": [
            {
                "role": "user",
                "content": "我购买了套餐，但是不能用",
            },
            {
                "role": "assistant",
                "content": "请问套餐已经到账了吗？",
            },
        ],
        "current_query": "已经到账了，但还是不能用",
    },
    {
        "name": "切换到新问题",
        "active_issue": {
            "summary": "用户续费后流量没有重置",
        },
        "recent_messages": [
            {
                "role": "user",
                "content": "续费以后流量没有恢复",
            },
            {
                "role": "assistant",
                "content": "请问套餐有效期已经延长了吗？",
            },
        ],
        "current_query": "另外，你们支持退款吗？",
    },
    {
        "name": "修正之前的信息",
        "active_issue": {
            "summary": "用户续费后套餐没有到账",
        },
        "recent_messages": [
            {
                "role": "user",
                "content": "套餐一直没有到账",
            },
            {
                "role": "assistant",
                "content": "请问续费订单支付成功了吗？",
            },
        ],
        "current_query": "不是续费，是我第一次购买套餐",
    },
    {
        "name": "关系不确定",
        "active_issue": {
            "summary": "用户正在处理续费后的流量问题",
        },
        "recent_messages": [
            {
                "role": "user",
                "content": "续费以后流量没有恢复",
            },
        ],
        "current_query": "那这个可以退吗？",
    },
]


for test_case in test_cases:
    print(f"\n===== {test_case['name']} =====")

    result = analyze_context(
        llm=model,
        current_query=test_case["current_query"],
        recent_messages=test_case["recent_messages"],
        active_issue=test_case["active_issue"],
    )

    print(
        json.dumps(
            result.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )