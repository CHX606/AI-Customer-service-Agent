"""使用真实生产路由评估常见客服语义。"""

from collections import defaultdict
from time import perf_counter

from back.agent.analysis.request import build_search_queries, route_request
from back.core.llm import (
    CONTEXT_MODEL_NAME,
    MODEL_NAME,
    ROUTER_MODEL_NAME,
    get_context_model,
    get_router_model,
    model,
)
from evaluation.benchmarks.performance import latency_summary


CASES = [
    {"id": "company", "query": "这是干啥的？", "scope": "in_scope", "action": "profile"},
    {"id": "hours", "query": "你们营业时间是什么？", "scope": "in_scope", "action": "profile"},
    {
        "id": "renewal_traffic",
        "query": "续费成功后有效期延长了，但流量没有重置。",
        "scope": "in_scope",
        "action": "retrieve",
    },
    {
        "id": "paid_not_arrived",
        "query": "订单已经支付成功，但是套餐一直没有到账。",
        "scope": "in_scope",
        "action": "retrieve",
    },
    {
        "id": "clash_connected",
        "query": "Windows上的Clash Verge显示连接成功，但网页打不开。",
        "scope": "in_scope",
        "action": "retrieve",
    },
    {
        "id": "refund",
        "query": "购买套餐后可以申请退款吗？",
        "scope": "in_scope",
        "action": "retrieve",
    },
    {"id": "vague_node", "query": "节点不能用。", "scope": "in_scope", "action": "clarify"},
    {
        "id": "vague_package",
        "query": "我购买了套餐，但是不能用。",
        "scope": "in_scope",
        "action": "clarify",
    },
    {"id": "hello", "query": "你好", "scope": "chitchat"},
    {"id": "thanks", "query": "谢谢你的帮助", "scope": "chitchat"},
    {"id": "weather", "query": "北京明天会下雨吗？", "scope": "out_of_scope"},
    {"id": "underspecified", "query": "怎么还是不行？", "scope": "uncertain"},
    {
        "id": "awaiting_followup",
        "query": "已经到账了，但还是不能用",
        "scope": "in_scope",
        "relation": "continue",
        "active_issue": {
            "summary": "用户购买套餐后无法使用，正在确认套餐是否到账",
            "status": "awaiting_user",
            "intent": "package",
            "last_clarifying_question": "请问套餐已经到账了吗？",
        },
        "recent_messages": [
            {"role": "assistant", "content": "请问套餐已经到账了吗？"},
        ],
    },
    {
        "id": "explicit_switch",
        "query": "另外，购买的套餐支持退款吗？",
        "scope": "in_scope",
        "action": "retrieve",
        "relation": "new_issue",
        "active_issue": {"summary": "续费后流量没有重置", "status": "answered"},
    },
    {
        "id": "correction",
        "query": "不是续费，是我第一次购买套餐",
        "scope": "in_scope",
        "relation": "correction",
        "active_issue": {"summary": "用户续费后套餐没有到账", "status": "open"},
    },
    {
        "id": "context_pronoun",
        "query": "这个还是不行",
        "scope": "in_scope",
        "relation": "continue",
        "active_issue": {"summary": "节点无法连接", "status": "answered"},
        "recent_messages": [
            {"role": "assistant", "content": "请更换节点后重新连接。"},
        ],
    },
    {
        "id": "context_platform_detail",
        "query": "是Windows，用的Clash Verge",
        "scope": "in_scope",
        "relation": "continue",
        "active_issue": {
            "summary": "客户端无法连接，等待确认设备和软件",
            "status": "awaiting_user",
            "last_clarifying_question": "请问使用什么设备和客户端？",
        },
        "recent_messages": [
            {"role": "assistant", "content": "请问使用什么设备和客户端？"},
        ],
    },
    {
        "id": "context_symptom_detail",
        "query": "没有报错，就是显示连接成功但网页打不开",
        "scope": "in_scope",
        "relation": "continue",
        "active_issue": {
            "summary": "Clash Verge连接异常，等待确认报错",
            "status": "awaiting_user",
            "last_clarifying_question": "客户端显示什么错误？",
        },
        "recent_messages": [
            {"role": "assistant", "content": "客户端显示什么错误？"},
        ],
    },
    {
        "id": "context_new_account_issue",
        "query": "换个问题，登录密码忘了怎么办？",
        "scope": "in_scope",
        "action": "retrieve",
        "relation": "new_issue",
        "active_issue": {"summary": "节点连接失败", "status": "answered"},
    },
    {
        "id": "context_node_correction",
        "query": "不是全部节点，只有香港节点超时",
        "scope": "in_scope",
        "relation": "correction",
        "active_issue": {"summary": "所有节点都连接超时", "status": "open"},
    },
    {
        "id": "context_closed_new_issue",
        "query": "如何更新订阅链接？",
        "scope": "in_scope",
        "action": "retrieve",
        "relation": "new_issue",
        "active_issue": {"summary": "旧的支付问题", "status": "resolved"},
    },
    {
        "id": "context_followup_traffic",
        "query": "那流量什么时候重置？",
        "scope": "in_scope",
        "action": "retrieve",
        "relation": "continue",
        "active_issue": {"summary": "续费后流量未重置", "status": "answered"},
    },
    {
        "id": "context_answer_scope",
        "query": "所有节点都这样，显示连接超时",
        "scope": "in_scope",
        "action": "retrieve",
        "relation": "continue",
        "active_issue": {
            "summary": "节点不能使用，等待确认影响范围",
            "status": "awaiting_user",
            "last_clarifying_question": "是全部节点还是单个节点？",
        },
        "recent_messages": [
            {"role": "assistant", "content": "是全部节点还是单个节点？"},
        ],
    },
    {
        "id": "context_software_correction",
        "query": "我说错了，是iPhone上的小火箭",
        "scope": "in_scope",
        "relation": "correction",
        "active_issue": {"summary": "Windows上的Clash无法连接", "status": "open"},
    },
]


def run_evaluation() -> tuple[int, int]:
    passed = 0
    route_latencies: dict[str, list[float]] = defaultdict(list)
    router_llm = model if ROUTER_MODEL_NAME == MODEL_NAME else get_router_model()
    context_llm = model if CONTEXT_MODEL_NAME == MODEL_NAME else get_context_model()
    for case in CASES:
        started = perf_counter()
        result = route_request(
            router_llm=router_llm,
            fallback_llm=(
                context_llm
                if case.get("active_issue") or case.get("recent_messages")
                else model
            ),
            current_query=case["query"],
            recent_messages=case.get("recent_messages", []),
            active_issue=case.get("active_issue"),
            company_name="可乐云",
            business_scope=["账号", "套餐", "续费", "流量", "订阅", "节点", "客户端"],
        )
        latency_ms = (perf_counter() - started) * 1000
        route_latencies[result.route_source].append(latency_ms)
        checks = [result.scope == case["scope"]]
        if "action" in case:
            checks.append(result.action == case["action"])
        if "relation" in case:
            checks.append(result.relation == case["relation"])
        checks.append(len(build_search_queries(result)) <= 3)
        ok = all(checks)
        passed += int(ok)
        print(
            f"{case['id']} {'PASS' if ok else 'FAIL'} "
            f"scope={result.scope} action={result.action} relation={result.relation} "
            f"source={result.route_source} latency_ms={latency_ms:.2f} "
            f"resolved={result.resolved_query}"
        )
    print(f"request-analysis={passed}/{len(CASES)}")
    for source, values in sorted(route_latencies.items()):
        print(f"route-source={source} latency={latency_summary(values)}")
    return passed, len(CASES)


if __name__ == "__main__":
    success, total = run_evaluation()
    raise SystemExit(0 if success == total else 1)
