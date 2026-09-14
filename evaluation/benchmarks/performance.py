"""生产性能、检索质量和知识覆盖率基准测试。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from dotenv import load_dotenv


load_dotenv()


EVALUATION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = (
    EVALUATION_DIR / "datasets" / "customer_service_performance.json"
)
DEFAULT_RESULTS_DIR = EVALUATION_DIR / "results"

STATUS_ANALYZING = "正在理解您的问题…"
STATUS_RETRIEVING = "正在查询知识库…"
STATUS_RERANKING = "正在筛选相关资料…"
STATUS_GENERATING = "正在生成回答…"

GENERIC_REJECTION_MARKERS = (
    "超出业务范围",
    "不属于业务范围",
    "无法回答这个问题",
    "无法为您解答",
    "仅能回答",
    "只能回答",
    "只能处理",
)
CLARIFY_MARKERS = (
    "请问",
    "请提供",
    "能否提供",
    "具体",
    "哪一个",
    "什么设备",
    "什么软件",
    "截图",
)
REFUSAL_MARKERS = (
    "无法",
    "不能",
    "不可以",
    "不建议",
    "不支持",
    "不会协助",
    "合规",
    "违法",
)
OUT_OF_SCOPE_MARKERS = GENERIC_REJECTION_MARKERS + (
    "客服业务",
    "相关业务问题",
    "建议查询",
)


def percentile(values: list[float], quantile: float) -> float | None:
    """按线性插值计算百分位；空数据返回 None。"""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def latency_summary(values: list[float | None]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(clean),
        "mean": round(mean(clean), 2),
        "min": round(min(clean), 2),
        "p50": round(percentile(clean, 0.50) or 0.0, 2),
        "p95": round(percentile(clean, 0.95) or 0.0, 2),
        "p99": round(percentile(clean, 0.99) or 0.0, 2),
        "max": round(max(clean), 2),
    }


def load_dataset(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("测试集必须包含非空 cases 列表")

    required = {
        "id",
        "category",
        "query",
        "expected_sections",
        "expected_behavior",
        "expected_retrieval",
        "source_expectation",
    }
    seen: set[str] = set()
    for index, case in enumerate(data["cases"], start=1):
        missing = required - set(case)
        if missing:
            raise ValueError(f"第 {index} 条用例缺少字段: {sorted(missing)}")
        case_id = str(case["id"])
        if case_id in seen:
            raise ValueError(f"测试用例 ID 重复: {case_id}")
        seen.add(case_id)
        if not str(case["query"]).strip():
            raise ValueError(f"测试用例问题为空: {case_id}")
    return data


def select_cases(
    dataset: dict[str, Any],
    *,
    suite: str,
    categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected = []
    category_set = set(categories or [])
    for case in dataset["cases"]:
        if suite not in case.get("suites", ["full"]):
            continue
        if category_set and case["category"] not in category_set:
            continue
        selected.append(case)
    if not selected:
        raise ValueError("筛选后没有可运行的测试用例")
    return selected


def _contains_any(text: str, choices: list[str] | tuple[str, ...]) -> bool:
    normalized = text.casefold()
    return any(str(choice).casefold() in normalized for choice in choices)


def score_answer(case: dict[str, Any], answer: str) -> dict[str, Any]:
    """用可复核的关键词和行为规则做基础评分，不替代人工抽检。"""
    text = answer.strip()
    behavior = case["expected_behavior"]
    if behavior == "answer":
        behavior_pass = len(text) >= 8 and not _contains_any(
            text, GENERIC_REJECTION_MARKERS
        )
    elif behavior == "request_image":
        behavior_pass = _contains_any(text, ["截图", "图片", "界面"])
    elif behavior == "clarify":
        behavior_pass = (
            "?" in text
            or "？" in text
            or _contains_any(text, CLARIFY_MARKERS)
        )
    elif behavior == "refuse":
        behavior_pass = _contains_any(text, REFUSAL_MARKERS)
    elif behavior == "out_of_scope":
        behavior_pass = _contains_any(text, OUT_OF_SCOPE_MARKERS)
    else:
        raise ValueError(f"未知 expected_behavior: {behavior}")

    groups = case.get("required_keyword_groups", [])
    missing_groups = [group for group in groups if not _contains_any(text, group)]
    forbidden = case.get("forbidden_keywords", [])
    matched_forbidden = [word for word in forbidden if _contains_any(text, [word])]
    keywords_pass = not missing_groups and not matched_forbidden
    return {
        "behavior_pass": behavior_pass,
        "keywords_pass": keywords_pass,
        "missing_keyword_groups": missing_groups,
        "matched_forbidden_keywords": matched_forbidden,
        "quality_pass": behavior_pass and keywords_pass,
    }


def _milliseconds(started: float) -> float:
    return round((perf_counter() - started) * 1000, 2)


def _stage_duration(
    timestamps: dict[str, float], start_name: str, end_name: str
) -> float | None:
    start = timestamps.get(start_name)
    end = timestamps.get(end_name)
    if start is None or end is None:
        return None
    return round(max(0.0, end - start), 2)


def run_http_case(
    case: dict[str, Any],
    *,
    base_url: str,
    tenant_id: str,
    tenant_token: str | None,
    timeout: float,
    run_number: int,
    session_id: str | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    session_id = session_id or f"perf-{case['id']}-{run_number}-{uuid4().hex[:10]}"
    endpoint = base_url.rstrip("/") + "/chat/stream"
    payload = json.dumps(
        {
            "tenant_id": tenant_id,
            "session_id": session_id,
            "message": case["query"],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/x-ndjson"}
    if tenant_token:
        headers["X-Tenant-Token"] = tenant_token

    result: dict[str, Any] = {
        "case_id": case["id"],
        "category": case["category"],
        "source_expectation": case["source_expectation"],
        "query": case["query"],
        "run_number": run_number,
        "success": False,
        "answer": "",
        "events": [],
        "route_source": None,
        "reranker_queue_wait_ms": None,
        "reranker_inference_ms": None,
        "reranker_max_concurrency": None,
        "first_token_ms": None,
        "failure_reasons": [],
    }
    status_times: dict[str, float] = {}
    first_event_ms: float | None = None
    final_ms: float | None = None

    try:
        request = Request(endpoint, data=payload, headers=headers, method="POST")
        with urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                if not raw_line.strip():
                    continue
                elapsed_ms = _milliseconds(started)
                event = json.loads(raw_line.decode("utf-8"))
                if first_event_ms is None:
                    first_event_ms = elapsed_ms
                event_type = str(event.get("type", ""))
                route_source = event.get("route_source")
                if event_type != "token":
                    result["events"].append(
                        {
                            "type": event_type,
                            "message": event.get("message"),
                            "route_source": route_source,
                            "reranker_queue_wait_ms": event.get(
                                "reranker_queue_wait_ms"
                            ),
                            "reranker_inference_ms": event.get(
                                "reranker_inference_ms"
                            ),
                            "at_ms": elapsed_ms,
                        }
                    )
                if route_source:
                    result["route_source"] = str(route_source)
                for metric in (
                    "reranker_queue_wait_ms",
                    "reranker_inference_ms",
                    "reranker_max_concurrency",
                ):
                    if event.get(metric) is not None:
                        result[metric] = event[metric]
                if event_type == "status":
                    message = str(event.get("message", ""))
                    status_times.setdefault(message, elapsed_ms)
                elif event_type == "token" and result["first_token_ms"] is None:
                    result["first_token_ms"] = elapsed_ms
                elif event_type == "final":
                    result["answer"] = str(event.get("answer", ""))
                    final_ms = elapsed_ms
                elif event_type == "error":
                    result["error"] = str(event.get("message", "流式接口返回错误"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        result["error"] = f"HTTP {error.code}: {body[:500]}"
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    except Exception as error:  # noqa: BLE001 - 基准工具必须保留单例失败继续跑
        result["error"] = f"{type(error).__name__}: {error}"

    result["first_event_ms"] = first_event_ms
    result["time_to_answer_ms"] = final_ms
    result["total_ms"] = _milliseconds(started)
    result["routing_ms"] = _stage_duration(
        status_times, STATUS_ANALYZING, STATUS_RETRIEVING
    )
    result["retrieval_stage_ms"] = _stage_duration(
        status_times, STATUS_RETRIEVING, STATUS_RERANKING
    )
    result["rerank_stage_ms"] = _stage_duration(
        status_times, STATUS_RERANKING, STATUS_GENERATING
    )
    if final_ms is not None and STATUS_GENERATING in status_times:
        result["generation_stage_ms"] = round(
            max(0.0, final_ms - status_times[STATUS_GENERATING]), 2
        )
    else:
        result["generation_stage_ms"] = None

    actual_retrieval = STATUS_RETRIEVING in status_times
    result["actual_retrieval"] = actual_retrieval
    result["route_pass"] = actual_retrieval == bool(case["expected_retrieval"])
    result["success"] = final_ms is not None and bool(result["answer"])

    if not result["success"]:
        result["failure_reasons"].append("system_error")
        result["answer_score"] = {
            "behavior_pass": False,
            "keywords_pass": False,
            "quality_pass": False,
        }
    else:
        result["answer_score"] = score_answer(case, result["answer"])
        if not result["route_pass"]:
            reason = (
                "knowledge_gap_or_scope_routing"
                if case["source_expectation"] == "coverage_gap"
                else "routing_error"
            )
            result["failure_reasons"].append(reason)
        if not result["answer_score"]["behavior_pass"]:
            result["failure_reasons"].append("answer_behavior")
        if not result["answer_score"]["keywords_pass"]:
            result["failure_reasons"].append("answer_content")

    result["quality_pass"] = (
        result["success"]
        and result["route_pass"]
        and result["answer_score"]["quality_pass"]
    )
    return result


def run_retrieval_case(
    case: dict[str, Any], *, tenant_id: str, run_number: int
) -> dict[str, Any]:
    from back.knowledge.retrieval.hybrid import RRF_CANDIDATES, retrieve_hybrid_candidates
    from back.knowledge.retrieval.reranker import rerank_documents

    result: dict[str, Any] = {
        "case_id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "expected_sections": case["expected_sections"],
        "run_number": run_number,
        "success": False,
    }
    try:
        retrieval_started = perf_counter()
        candidates = retrieve_hybrid_candidates(
            case["query"], tenant_id=tenant_id, limit=RRF_CANDIDATES
        )
        result["retrieval_ms"] = _milliseconds(retrieval_started)

        rerank_started = perf_counter()
        documents = rerank_documents(case["query"], [item[0] for item in candidates])
        result["rerank_ms"] = _milliseconds(rerank_started)
        if documents:
            result["reranker_queue_wait_ms"] = documents[0].metadata.get(
                "reranker_queue_wait_ms"
            )
            result["reranker_inference_ms"] = documents[0].metadata.get(
                "reranker_inference_ms"
            )
        candidate_sections = [
            str(document.metadata.get("section_id", "")) for document, _ in candidates
        ]
        final_sections = [
            str(document.metadata.get("section_id", "")) for document in documents
        ]
        expected = set(str(item) for item in case["expected_sections"])
        candidate_rank = next(
            (index for index, section in enumerate(candidate_sections, 1) if section in expected),
            None,
        )
        final_rank = next(
            (index for index, section in enumerate(final_sections, 1) if section in expected),
            None,
        )
        result.update(
            {
                "success": True,
                "candidate_sections": candidate_sections,
                "final_sections": final_sections,
                "candidate_rank": candidate_rank,
                "final_rank": final_rank,
                "hit_at_1": final_rank == 1,
                "hit_at_3": final_rank is not None and final_rank <= 3,
                "hit_at_5": final_rank is not None and final_rank <= 5,
                "reciprocal_rank": 0.0 if final_rank is None else 1.0 / final_rank,
            }
        )
    except Exception as error:  # noqa: BLE001 - 单条失败不应终止整批评测
        result["error"] = f"{type(error).__name__}: {error}"
        result.setdefault("retrieval_ms", None)
        result.setdefault("rerank_ms", None)
        result.update(
            {
                "hit_at_1": False,
                "hit_at_3": False,
                "hit_at_5": False,
                "reciprocal_rank": 0.0,
            }
        )
    return result


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def aggregate_retrieval(samples: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [sample for sample in samples if sample["success"]]
    return {
        "runs": len(samples),
        "successful_runs": len(successful),
        "error_rate": _rate(len(samples) - len(successful), len(samples)),
        "hit_at_1": _rate(sum(bool(item["hit_at_1"]) for item in successful), len(successful)),
        "hit_at_3": _rate(sum(bool(item["hit_at_3"]) for item in successful), len(successful)),
        "hit_at_5": _rate(sum(bool(item["hit_at_5"]) for item in successful), len(successful)),
        "mrr": round(
            mean(item["reciprocal_rank"] for item in successful), 4
        ) if successful else 0.0,
        "retrieval_latency_ms": latency_summary(
            [item.get("retrieval_ms") for item in successful]
        ),
        "rerank_latency_ms": latency_summary(
            [item.get("rerank_ms") for item in successful]
        ),
        "reranker_queue_wait_ms": latency_summary(
            [item.get("reranker_queue_wait_ms") for item in successful]
        ),
        "reranker_inference_ms": latency_summary(
            [item.get("reranker_inference_ms") for item in successful]
        ),
    }


def aggregate_http(samples: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [sample for sample in samples if sample["success"]]
    quality_passes = [sample for sample in samples if sample.get("quality_pass")]
    sla_passes = [sample for sample in samples if sample.get("sla_pass", True)]
    overall_passes = [
        sample
        for sample in samples
        if sample.get("quality_pass") and sample.get("sla_pass", True)
    ]
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        categories[sample["category"]].append(sample)

    category_summary = {}
    for category, items in sorted(categories.items()):
        category_summary[category] = {
            "runs": len(items),
            "success_rate": _rate(sum(bool(item["success"]) for item in items), len(items)),
            "quality_pass_rate": _rate(
                sum(bool(item.get("quality_pass")) for item in items), len(items)
            ),
            "p95_time_to_answer_ms": latency_summary(
                [item.get("time_to_answer_ms") for item in items]
            )["p95"],
        }

    failure_counts = Counter(
        reason for sample in samples for reason in sample.get("failure_reasons", [])
    )
    route_source_counts = Counter(
        str(sample.get("route_source") or "unknown") for sample in samples
    )
    return {
        "runs": len(samples),
        "successful_runs": len(successful),
        "error_rate": _rate(len(samples) - len(successful), len(samples)),
        "quality_pass_rate": _rate(len(quality_passes), len(samples)),
        "sla_pass_rate": _rate(len(sla_passes), len(samples)),
        "overall_pass_rate": _rate(len(overall_passes), len(samples)),
        "first_event_latency_ms": latency_summary(
            [item.get("first_event_ms") for item in successful]
        ),
        "first_token_latency_ms": latency_summary(
            [item.get("first_token_ms") for item in successful]
        ),
        "time_to_answer_ms": latency_summary(
            [item.get("time_to_answer_ms") for item in successful]
        ),
        "routing_stage_ms": latency_summary(
            [item.get("routing_ms") for item in successful]
        ),
        "retrieval_stage_ms": latency_summary(
            [item.get("retrieval_stage_ms") for item in successful]
        ),
        "rerank_stage_ms": latency_summary(
            [item.get("rerank_stage_ms") for item in successful]
        ),
        "reranker_queue_wait_ms": latency_summary(
            [item.get("reranker_queue_wait_ms") for item in successful]
        ),
        "reranker_inference_ms": latency_summary(
            [item.get("reranker_inference_ms") for item in successful]
        ),
        "generation_stage_ms": latency_summary(
            [item.get("generation_stage_ms") for item in successful]
        ),
        "route_source_counts": dict(route_source_counts.most_common()),
        "route_source_rates": {
            source: _rate(count, len(samples))
            for source, count in route_source_counts.items()
        },
        "failure_reasons": dict(failure_counts.most_common()),
        "categories": category_summary,
    }


def annotate_http_sla(
    samples: list[dict[str, Any]], thresholds: dict[str, Any]
) -> None:
    """给每条请求标出慢在首事件、检索还是最终回答。"""
    for sample in samples:
        slow_reasons = []
        if (
            sample.get("first_event_ms") is not None
            and sample["first_event_ms"] > thresholds["first_event_p95_ms"]
        ):
            slow_reasons.append("first_event_slow")
        if (
            sample.get("retrieval_stage_ms") is not None
            and sample["retrieval_stage_ms"] > thresholds["retrieval_stage_p95_ms"]
        ):
            slow_reasons.append("retrieval_slow")
        if (
            sample.get("time_to_answer_ms") is not None
            and sample["time_to_answer_ms"] > thresholds["time_to_answer_p95_ms"]
        ):
            slow_reasons.append("answer_slow")
        sample.setdefault("failure_reasons", []).extend(
            reason
            for reason in slow_reasons
            if reason not in sample["failure_reasons"]
        )
        sample["sla_pass"] = not slow_reasons
        sample["case_pass"] = bool(sample.get("quality_pass")) and not slow_reasons


def evaluate_thresholds(
    report: dict[str, Any], thresholds: dict[str, Any]
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, actual: float | None, target: float, operator: str) -> None:
        if actual is None:
            passed = False
        elif operator == "<=":
            passed = actual <= target
        else:
            passed = actual >= target
        checks.append(
            {
                "name": name,
                "actual": actual,
                "target": target,
                "operator": operator,
                "passed": passed,
            }
        )

    retrieval = report.get("retrieval_summary")
    if retrieval:
        config = thresholds["retrieval"]
        add("检索 P95", retrieval["retrieval_latency_ms"]["p95"], config["p95_ms"], "<=")
        add("重排 P95", retrieval["rerank_latency_ms"]["p95"], config["rerank_p95_ms"], "<=")
        add("Hit@5", retrieval["hit_at_5"], config["hit_at_5"], ">=")
        add("MRR", retrieval["mrr"], config["mrr"], ">=")
        add("检索错误率", retrieval["error_rate"], config["error_rate"], "<=")

    http = report.get("http_summary")
    if http:
        config = thresholds["http"]
        add("首事件 P95", http["first_event_latency_ms"]["p95"], config["first_event_p95_ms"], "<=")
        if http["retrieval_stage_ms"]["count"]:
            add("端到端检索阶段 P95", http["retrieval_stage_ms"]["p95"], config["retrieval_stage_p95_ms"], "<=")
        add("首个答案 P95", http["time_to_answer_ms"]["p95"], config["time_to_answer_p95_ms"], "<=")
        add("回答质量通过率", http["quality_pass_rate"], config["quality_pass_rate"], ">=")
        add("接口错误率", http["error_rate"], config["error_rate"], "<=")
    return checks


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_markdown_report(report: dict[str, Any]) -> str:
    checks = report["threshold_checks"]
    overall = "PASS" if checks and all(item["passed"] for item in checks) else "FAIL"
    lines = [
        "# 客服系统性能测试报告",
        "",
        f"- 结果：**{overall}**",
        f"- 生成时间：{report['generated_at']}",
        f"- 测试集：{report['dataset']['name']}（{report['dataset']['selected_cases']} 条）",
        f"- 模式：{report['config']['mode']}，并发：{report['config']['concurrency']}，重复：{report['config']['repeats']}",
        f"- Reranker：{report['config']['reranker_model']}",
        f"- Reranker 后端：{report['config'].get('reranker_backend', 'torch')}，候选数：{report['config'].get('reranker_candidates', 12)}，最大长度：{report['config'].get('reranker_max_length', 512)}，批量：{report['config'].get('reranker_batch_size', 8)}",
        "",
        "## SLA 检查",
        "",
        "| 指标 | 实际 | 目标 | 结果 |",
        "|---|---:|---:|---|",
    ]
    for item in checks:
        lines.append(
            f"| {item['name']} | {_fmt(item['actual'])} | "
            f"{item['operator']} {_fmt(item['target'])} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |"
        )

    retrieval = report.get("retrieval_summary")
    if retrieval:
        lines.extend(
            [
                "",
                "## 检索",
                "",
                f"- Hit@1 / Hit@3 / Hit@5：{retrieval['hit_at_1']:.2%} / {retrieval['hit_at_3']:.2%} / {retrieval['hit_at_5']:.2%}",
                f"- MRR：{retrieval['mrr']:.4f}",
                f"- 检索耗时 P50 / P95：{_fmt(retrieval['retrieval_latency_ms']['p50'])} / {_fmt(retrieval['retrieval_latency_ms']['p95'])} ms",
                f"- 重排耗时 P50 / P95：{_fmt(retrieval['rerank_latency_ms']['p50'])} / {_fmt(retrieval['rerank_latency_ms']['p95'])} ms",
                f"- 重排排队 P50 / P95：{_fmt(retrieval['reranker_queue_wait_ms']['p50'])} / {_fmt(retrieval['reranker_queue_wait_ms']['p95'])} ms",
                f"- 重排推理 P50 / P95：{_fmt(retrieval['reranker_inference_ms']['p50'])} / {_fmt(retrieval['reranker_inference_ms']['p95'])} ms",
            ]
        )

    http = report.get("http_summary")
    if http:
        lines.extend(
            [
                "",
                "## 端到端",
                "",
                f"- 接口成功率：{1 - http['error_rate']:.2%}",
                f"- 自动质量通过率：{http['quality_pass_rate']:.2%}",
                f"- 单请求 SLA 通过率：{http['sla_pass_rate']:.2%}",
                f"- 质量与 SLA 同时通过率：{http['overall_pass_rate']:.2%}",
                f"- 首事件 P50 / P95：{_fmt(http['first_event_latency_ms']['p50'])} / {_fmt(http['first_event_latency_ms']['p95'])} ms",
                f"- 首 Token P50 / P95：{_fmt(http['first_token_latency_ms']['p50'])} / {_fmt(http['first_token_latency_ms']['p95'])} ms",
                f"- 完整答案 P50 / P95：{_fmt(http['time_to_answer_ms']['p50'])} / {_fmt(http['time_to_answer_ms']['p95'])} ms",
                f"- 请求路由 P50 / P95：{_fmt(http['routing_stage_ms']['p50'])} / {_fmt(http['routing_stage_ms']['p95'])} ms",
                f"- 检索阶段 P50 / P95：{_fmt(http['retrieval_stage_ms']['p50'])} / {_fmt(http['retrieval_stage_ms']['p95'])} ms",
                f"- 重排阶段 P50 / P95：{_fmt(http['rerank_stage_ms']['p50'])} / {_fmt(http['rerank_stage_ms']['p95'])} ms",
                f"- 重排排队 P50 / P95：{_fmt(http['reranker_queue_wait_ms']['p50'])} / {_fmt(http['reranker_queue_wait_ms']['p95'])} ms",
                f"- 重排推理 P50 / P95：{_fmt(http['reranker_inference_ms']['p50'])} / {_fmt(http['reranker_inference_ms']['p95'])} ms",
                f"- 生成阶段 P50 / P95：{_fmt(http['generation_stage_ms']['p50'])} / {_fmt(http['generation_stage_ms']['p95'])} ms",
                "- 路由来源："
                + "，".join(
                    f"{source} {count} 次（{http['route_source_rates'][source]:.2%}）"
                    for source, count in http["route_source_counts"].items()
                ),
                "",
                "### 分类结果",
                "",
                "| 分类 | 次数 | 成功率 | 质量通过率 | 回答 P95(ms) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for category, item in http["categories"].items():
            lines.append(
                f"| {category} | {item['runs']} | {item['success_rate']:.2%} | "
                f"{item['quality_pass_rate']:.2%} | {_fmt(item['p95_time_to_answer_ms'])} |"
            )

        lines.extend(["", "### 失败诊断", ""])
        if http["failure_reasons"]:
            for reason, count in http["failure_reasons"].items():
                lines.append(f"- {reason}: {count}")
        else:
            lines.append("- 无")

        failed = [item for item in report["http_samples"] if not item.get("case_pass")]
        if failed:
            lines.extend(
                [
                    "",
                    "### 需要处理的用例",
                    "",
                    "| 用例 | 分类 | 原因 | 回答耗时(ms) |",
                    "|---|---|---|---:|",
                ]
            )
            for item in failed[:30]:
                reasons = ", ".join(item.get("failure_reasons", [])) or "unknown"
                lines.append(
                    f"| {item['case_id']} | {item['category']} | {reasons} | "
                    f"{_fmt(item.get('time_to_answer_ms'))} |"
                )
    lines.append("")
    return "\n".join(lines)


def _inventory(cases: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(case["category"] for case in cases).most_common())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--mode", choices=["retrieval", "http", "all"], default="all")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--tenant-id", default="default")
    parser.add_argument(
        "--tenant-token",
        default=os.getenv("BENCHMARK_TENANT_TOKEN"),
        help="租户令牌；也可通过 BENCHMARK_TENANT_TOKEN 提供",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--reranker-model",
        default=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        help="retrieval 模式使用的精排模型，便于执行新旧模型 A/B 测试",
    )
    parser.add_argument(
        "--reranker-backend",
        choices=["torch", "onnx"],
        default=os.getenv("RERANKER_BACKEND", "torch"),
    )
    parser.add_argument(
        "--reranker-model-path",
        default=os.getenv("RERANKER_MODEL_PATH", ""),
    )
    parser.add_argument(
        "--reranker-onnx-file",
        default=os.getenv("RERANKER_ONNX_FILE", ""),
    )
    parser.add_argument(
        "--reranker-candidates",
        type=int,
        default=int(os.getenv("RERANKER_CANDIDATES", "12")),
    )
    parser.add_argument(
        "--reranker-max-length",
        type=int,
        default=int(os.getenv("RERANKER_MAX_LENGTH", "512")),
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=int(os.getenv("RERANKER_BATCH_SIZE", "8")),
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enforce", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repeats < 1 or args.concurrency < 1 or args.warmup < 0:
        raise ValueError("repeats/concurrency 必须至少为 1，warmup 不能小于 0")
    if min(
        args.reranker_candidates,
        args.reranker_max_length,
        args.reranker_batch_size,
    ) < 1:
        raise ValueError("Reranker 候选数、最大长度和批量必须至少为 1")

    dataset = load_dataset(args.dataset)
    cases = select_cases(dataset, suite=args.suite, categories=args.categories)
    print(f"已加载 {len(cases)} 条用例: {_inventory(cases)}")
    if args.dry_run:
        print("测试集校验通过；dry-run 未访问模型、OpenSearch 或接口。")
        return 0

    if not args.reranker_model.strip():
        raise ValueError("reranker-model 不能为空")
    if args.mode in {"retrieval", "all"}:
        os.environ["RERANKER_MODEL"] = args.reranker_model.strip()
        os.environ["RERANKER_BACKEND"] = args.reranker_backend
        os.environ["RERANKER_MODEL_PATH"] = args.reranker_model_path.strip()
        os.environ["RERANKER_ONNX_FILE"] = args.reranker_onnx_file.strip()
        os.environ["RERANKER_CANDIDATES"] = str(args.reranker_candidates)
        os.environ["RERANKER_MAX_LENGTH"] = str(args.reranker_max_length)
        os.environ["RERANKER_BATCH_SIZE"] = str(args.reranker_batch_size)

    retrieval_cases = [case for case in cases if case["expected_sections"]]
    retrieval_samples: list[dict[str, Any]] = []
    http_samples: list[dict[str, Any]] = []
    warmup: dict[str, Any] = {}

    if args.mode in {"retrieval", "all"} and retrieval_cases:
        for index in range(args.warmup):
            case = retrieval_cases[index % len(retrieval_cases)]
            sample = run_retrieval_case(case, tenant_id=args.tenant_id, run_number=0)
            warmup.setdefault("retrieval_ms", []).append(
                (sample.get("retrieval_ms") or 0) + (sample.get("rerank_ms") or 0)
            )
            print(
                f"检索预热进度: {index + 1}/{args.warmup}",
                flush=True,
            )
        total_retrieval_runs = len(retrieval_cases) * args.repeats
        completed_retrieval_runs = 0
        for run_number in range(1, args.repeats + 1):
            for case in retrieval_cases:
                retrieval_samples.append(
                    run_retrieval_case(
                        case, tenant_id=args.tenant_id, run_number=run_number
                    )
                )
                completed_retrieval_runs += 1
                print(
                    "检索评测进度: "
                    f"{completed_retrieval_runs}/{total_retrieval_runs} "
                    f"({case['id']})",
                    flush=True,
                )
        print(f"检索评测完成: {len(retrieval_samples)} 次")

    if args.mode in {"http", "all"}:
        for index in range(args.warmup):
            case = cases[index % len(cases)]
            sample = run_http_case(
                case,
                base_url=args.base_url,
                tenant_id=args.tenant_id,
                tenant_token=args.tenant_token,
                timeout=args.timeout,
                run_number=0,
            )
            warmup.setdefault("http_ms", []).append(sample["total_ms"])

        jobs = [
            (case, run_number)
            for run_number in range(1, args.repeats + 1)
            for case in cases
        ]
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(
                    run_http_case,
                    case,
                    base_url=args.base_url,
                    tenant_id=args.tenant_id,
                    tenant_token=args.tenant_token,
                    timeout=args.timeout,
                    run_number=run_number,
                )
                for case, run_number in jobs
            ]
            for future in as_completed(futures):
                http_samples.append(future.result())
        http_samples.sort(key=lambda item: (item["run_number"], item["case_id"]))
        annotate_http_sla(http_samples, dataset["thresholds"]["http"])
        print(f"端到端评测完成: {len(http_samples)} 次")

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "dataset": {
            "name": dataset.get("name", args.dataset.name),
            "version": dataset.get("version"),
            "path": str(args.dataset.resolve()),
            "selected_cases": len(cases),
            "inventory": _inventory(cases),
        },
        "config": {
            "suite": args.suite,
            "mode": args.mode,
            "base_url": args.base_url,
            "tenant_id": args.tenant_id,
            "repeats": args.repeats,
            "concurrency": args.concurrency,
            "warmup": args.warmup,
            "timeout": args.timeout,
            "reranker_model": args.reranker_model.strip(),
            "reranker_backend": args.reranker_backend,
            "reranker_model_path": args.reranker_model_path.strip(),
            "reranker_onnx_file": args.reranker_onnx_file.strip(),
            "reranker_candidates": args.reranker_candidates,
            "reranker_max_length": args.reranker_max_length,
            "reranker_batch_size": args.reranker_batch_size,
            "reranker_max_concurrency": int(
                os.getenv("RERANKER_MAX_CONCURRENCY", "2")
            ),
            "context_model": os.getenv("CONTEXT_MODEL") or os.getenv("MODEL"),
            "response_model": os.getenv("RESPONSE_MODEL") or os.getenv("MODEL"),
        },
        "warmup": warmup,
        "retrieval_samples": retrieval_samples,
        "http_samples": http_samples,
    }
    if retrieval_samples:
        report["retrieval_summary"] = aggregate_retrieval(retrieval_samples)
    if http_samples:
        report["http_summary"] = aggregate_http(http_samples)
    report["threshold_checks"] = evaluate_thresholds(report, dataset["thresholds"])
    report["passed"] = bool(report["threshold_checks"]) and all(
        item["passed"] for item in report["threshold_checks"]
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = args.output or DEFAULT_RESULTS_DIR / f"performance-{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path = output_path.with_suffix(".md")
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"JSON 报告: {output_path.resolve()}")
    print(f"Markdown 报告: {markdown_path.resolve()}")
    print(f"结果: {'PASS' if report['passed'] else 'FAIL'}")
    return 1 if args.enforce and not report["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
