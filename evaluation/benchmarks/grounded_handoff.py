"""有依据回答与转人工行为评测：统计该答未答、该转未转、内部用语泄露和两段式兜底。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

from evaluation.benchmarks.performance import latency_summary, run_http_case


load_dotenv()


EVALUATION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = EVALUATION_DIR / "datasets" / "grounded_handoff.json"
DEFAULT_RESULTS_DIR = EVALUATION_DIR / "results"

# 与 response_nodes 的固定文案对应；租户自定义的转人工提示语用 --handoff-marker 追加。
HANDOFF_MARKERS = ("暂未查到", "无法安全地给出确定结论")
# 独立固定申请引导才属于转人工；带有依据正文的 partial 末尾同样会附此入口。
# 与生产 SUPPORT_REQUEST_PROMPT 的一致性由离线回归验证，不为评测导入模型模块。
DIRECT_HANDOFF_REPLY = (
    "请点击“提交人工处理”，填写需要处理的事情。"
    "提交后会记录申请，由站长查看并处理。"
)
RETRY_MARKERS = ("换个说法",)
OUT_OF_SCOPE_MARKERS = ("我目前只能处理",)
CLARIFY_MARKERS = ("说的具体一点",)
CLARIFY_MAX_LENGTH = 80
OUTCOMES = {"answer", "clarify", "retry", "handoff", "out_of_scope"}


def classify_answer(
    answer: str,
    handoff_markers: tuple[str, ...] = HANDOFF_MARKERS,
    *,
    support_required: bool | None = None,
    support_reason: str | None = None,
) -> str:
    """优先使用已保存的人工元数据，兼容没有元数据的旧结果；partial 仍是回答。"""
    if support_required is True:
        if support_reason in {"explicit_request", "not_found", "conflict"}:
            return "handoff"
        if support_reason in {"partial", "human_action"}:
            return "answer"
    text = answer.strip()
    if text == DIRECT_HANDOFF_REPLY:
        return "handoff"
    if any(marker in text for marker in handoff_markers):
        return "handoff"
    if any(marker in text for marker in RETRY_MARKERS):
        return "retry"
    if any(marker in text for marker in OUT_OF_SCOPE_MARKERS):
        return "out_of_scope"
    if any(marker in text for marker in CLARIFY_MARKERS):
        return "clarify"
    # 追问通常是一句短问句；带问号的长回复仍按回答处理。
    if ("?" in text or "？" in text) and len(text) <= CLARIFY_MAX_LENGTH:
        return "clarify"
    return "answer"


def _contains_any(text: str, choices: list[str]) -> bool:
    normalized = text.casefold()
    return any(str(choice).casefold() in normalized for choice in choices)


def load_dataset(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("测试集必须包含非空 cases 列表")
    seen: set[str] = set()
    for case in cases:
        case_id = str(case.get("id", "")).strip()
        if not case_id or case_id in seen:
            raise ValueError(f"用例 ID 缺失或重复: {case_id!r}")
        seen.add(case_id)
        turns = case.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"用例至少需要一轮: {case_id}")
        for index, turn in enumerate(turns, start=1):
            if not str(turn.get("message", "")).strip():
                raise ValueError(f"用例 {case_id} 第 {index} 轮消息为空")
            expect = turn.get("expect")
            if not isinstance(expect, list) or not expect or not set(expect) <= OUTCOMES:
                raise ValueError(f"用例 {case_id} 第 {index} 轮 expect 无效: {expect!r}")
    return data


def evaluate_turn(
    turn: dict[str, Any],
    answer: str,
    *,
    handoff_markers: tuple[str, ...] = HANDOFF_MARKERS,
    global_forbidden: list[str] | None = None,
    support_required: bool | None = None,
    support_reason: str | None = None,
) -> dict[str, Any]:
    outcome = classify_answer(answer, handoff_markers,
                              support_required=support_required, support_reason=support_reason)
    missing = (
        [group for group in turn.get("required_keyword_groups", []) if not _contains_any(answer, group)]
        if outcome == "answer"
        else []
    )
    forbidden = [
        word
        for word in [*(global_forbidden or []), *turn.get("forbidden_keywords", [])]
        if _contains_any(answer, [word])
    ]
    return {
        "outcome": outcome,
        "expected": turn["expect"],
        "missing_keyword_groups": missing,
        "matched_forbidden_keywords": forbidden,
        "pass": outcome in turn["expect"] and not missing and not forbidden,
    }


def run_case(
    case: dict[str, Any],
    *,
    base_url: str,
    tenant_id: str,
    tenant_token: str | None,
    timeout: float,
    run_number: int,
    handoff_markers: tuple[str, ...],
    global_forbidden: list[str],
) -> dict[str, Any]:
    session_id = f"grounded-{case['id']}-{run_number}-{uuid4().hex[:10]}"
    turns = []
    for turn_number, turn in enumerate(case["turns"], start=1):
        # run_http_case 只用于发送流式请求；行为评分由本模块完成。
        sample = run_http_case(
            {
                "id": f"{case['id']}-turn-{turn_number}",
                "category": case["category"],
                "query": turn["message"],
                "expected_behavior": "answer",
                "expected_retrieval": True,
                "source_expectation": "knowledge_base",
            },
            base_url=base_url,
            tenant_id=tenant_id,
            tenant_token=tenant_token,
            timeout=timeout,
            run_number=run_number,
            session_id=session_id,
        )
        record = {
            "turn_number": turn_number,
            "message": turn["message"],
            "answer": sample.get("answer", ""),
            "route_source": sample.get("route_source"),
            "actual_retrieval": sample.get("actual_retrieval"),
            "time_to_answer_ms": sample.get("time_to_answer_ms"),
            "success": bool(sample.get("success")),
            "error": sample.get("error"),
        }
        # 当前发送器可能不提供元数据；仅在实际存在时保存，旧结果继续用文案兜底。
        for key in ("support_required", "support_reason"):
            if key in sample:
                record[key] = sample[key]
        turns.append(_score_record(
            turn, record, handoff_markers=handoff_markers, global_forbidden=global_forbidden
        ))
    return _build_sample(case, run_number=run_number, session_id=session_id, turns=turns)


def _score_record(
    turn: dict[str, Any],
    record: dict[str, Any],
    *,
    handoff_markers: tuple[str, ...],
    global_forbidden: list[str],
) -> dict[str, Any]:
    if record["success"]:
        return {**record, **evaluate_turn(
            turn,
            record["answer"],
            handoff_markers=handoff_markers,
            global_forbidden=global_forbidden,
            support_required=record.get("support_required"),
            support_reason=record.get("support_reason"),
        )}
    return {
        **record,
        "outcome": "error",
        "expected": turn["expect"],
        "missing_keyword_groups": [],
        "matched_forbidden_keywords": [],
        "pass": False,
    }


def _build_sample(
    case: dict[str, Any],
    *,
    run_number: int,
    session_id: str,
    turns: list[dict[str, Any]],
) -> dict[str, Any]:
    flow_pass = None
    if case.get("flow") == "retry_then_handoff":
        flow_pass = [turn["outcome"] for turn in turns[:2]] == ["retry", "handoff"]
    return {
        "case_id": case["id"],
        "category": case["category"],
        "note": case.get("note"),
        "run_number": run_number,
        "session_id": session_id,
        "flow": case.get("flow"),
        "flow_pass": flow_pass,
        "pass": all(turn["pass"] for turn in turns),
        "turns": turns,
    }


def rescore_samples(
    previous_samples: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    *,
    handoff_markers: tuple[str, ...],
    global_forbidden: list[str],
) -> list[dict[str, Any]]:
    """用当前测试集的标注给已保存的回答重新打分；已移出测试集的用例会被跳过。"""
    cases_by_id = {case["id"]: case for case in cases}
    samples = []
    for previous in previous_samples:
        case = cases_by_id.get(previous["case_id"])
        if case is None:
            continue
        if len(previous["turns"]) != len(case["turns"]):
            raise ValueError(f"用例 {case['id']} 的轮数已变化，不能重新打分，请重新运行")
        turns = [
            _score_record(
                turn, previous_turn, handoff_markers=handoff_markers, global_forbidden=global_forbidden
            )
            for turn, previous_turn in zip(case["turns"], previous["turns"], strict=True)
        ]
        samples.append(_build_sample(
            case, run_number=previous["run_number"], session_id=previous["session_id"], turns=turns
        ))
    return samples


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    turns = [turn for sample in samples for turn in sample["turns"]]
    should_answer = [turn for turn in turns if turn["expected"] == ["answer"]]
    should_not_answer = [turn for turn in turns if "answer" not in turn["expected"]]
    answered = [turn for turn in should_answer if turn["outcome"] == "answer"]
    flows = [sample["flow_pass"] for sample in samples if sample["flow_pass"] is not None]

    categories: dict[str, list[bool]] = defaultdict(list)
    for sample in samples:
        categories[sample["category"]].append(sample["pass"])

    return {
        "conversations": len(samples),
        "turns": len(turns),
        "error_rate": _rate(sum(turn["outcome"] == "error" for turn in turns), len(turns)),
        "should_answer_turns": len(should_answer),
        "unnecessary_handoff_rate": _rate(
            sum(turn["outcome"] == "handoff" for turn in should_answer), len(should_answer)
        ),
        "should_answer_answer_rate": _rate(len(answered), len(should_answer)),
        "answered_keyword_pass_rate": _rate(
            sum(not turn["missing_keyword_groups"] for turn in answered), len(answered)
        ),
        "should_not_answer_turns": len(should_not_answer),
        "wrongly_answered_rate": _rate(
            sum(turn["outcome"] == "answer" for turn in should_not_answer), len(should_not_answer)
        ),
        "forbidden_hit_turns": sum(bool(turn["matched_forbidden_keywords"]) for turn in turns),
        "retry_then_handoff_rate": _rate(sum(flows), len(flows)),
        "turn_pass_rate": _rate(sum(turn["pass"] for turn in turns), len(turns)),
        "conversation_pass_rate": _rate(sum(sample["pass"] for sample in samples), len(samples)),
        "semantic_cache_turns": sum(turn["route_source"] == "semantic_cache" for turn in turns),
        "category_pass_rate": {
            category: _rate(sum(values), len(values)) for category, values in sorted(categories.items())
        },
        "answer_latency_ms": latency_summary([turn["time_to_answer_ms"] for turn in turns]),
    }


def _fmt_rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def _cell(text: str, limit: int = 60) -> str:
    compact = " ".join(str(text).split()).replace("|", "\\|")
    return compact if len(compact) <= limit else compact[:limit] + "…"


def build_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# 有依据回答与转人工评测报告",
        "",
        f"- 会话数 / 轮次：{summary['conversations']} / {summary['turns']}",
        f"- 接口错误率：{_fmt_rate(summary['error_rate'])}",
        f"- **应答题误转人工率**：{_fmt_rate(summary['unnecessary_handoff_rate'])}"
        f"（{summary['should_answer_turns']} 轮应答题）",
        f"- 应答题回答率：{_fmt_rate(summary['should_answer_answer_rate'])}",
        f"- 已回答轮关键词命中率：{_fmt_rate(summary['answered_keyword_pass_rate'])}",
        f"- **不应答题被回答率**：{_fmt_rate(summary['wrongly_answered_rate'])}"
        f"（{summary['should_not_answer_turns']} 轮，需人工复核是否编造）",
        f"- 内部用语/禁用词命中轮数：{summary['forbidden_hit_turns']}",
        f"- 先追问再转人工流程符合率：{_fmt_rate(summary['retry_then_handoff_rate'])}",
        f"- 轮次通过率：{_fmt_rate(summary['turn_pass_rate'])}",
        f"- 语义缓存命中轮数：{summary['semantic_cache_turns']}（对比前后版本时应为 0）",
        "",
        "## 分类通过率",
        "",
        "| 分类 | 会话通过率 |",
        "|---|---|",
        *[
            f"| {category} | {_fmt_rate(rate)} |"
            for category, rate in summary["category_pass_rate"].items()
        ],
        "",
        "## 逐轮结果",
        "",
        "| 用例 | 轮 | 期望 | 实际 | 结果 | 问题说明 | 回答摘要 |",
        "|---|---|---|---|---|---|---|",
    ]
    for sample in report["samples"]:
        for turn in sample["turns"]:
            problems = []
            if turn["missing_keyword_groups"]:
                problems.append("缺关键词 " + "/".join("|".join(group) for group in turn["missing_keyword_groups"]))
            if turn["matched_forbidden_keywords"]:
                problems.append("命中禁用词 " + "、".join(turn["matched_forbidden_keywords"]))
            if turn.get("error"):
                problems.append(str(turn["error"]))
            lines.append(
                f"| {sample['case_id']} | {turn['turn_number']} | {'/'.join(turn['expected'])} | "
                f"{turn['outcome']} | {'PASS' if turn['pass'] else 'FAIL'} | "
                f"{_cell('；'.join(problems), 40)} | {_cell(turn['answer'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--tenant-id", default="default")
    parser.add_argument("--tenant-token", default=os.getenv("BENCHMARK_TENANT_TOKEN"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--handoff-marker",
        action="append",
        default=[],
        help="租户自定义转人工提示语中的固定片段，可重复传入",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rescore",
        type=Path,
        help="不访问接口，用当前测试集的标注给该 JSON 报告中已保存的回答重新打分",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repeats < 1 or args.concurrency < 1:
        raise ValueError("repeats/concurrency 必须至少为 1")

    dataset = load_dataset(args.dataset)
    selected_ids = set(args.case_ids or [])
    cases = [case for case in dataset["cases"] if not selected_ids or case["id"] in selected_ids]
    missing = selected_ids - {case["id"] for case in cases}
    if missing:
        raise ValueError(f"未找到用例: {sorted(missing)}")
    print(f"已加载 {len(cases)} 条用例")
    if args.dry_run:
        print("测试集校验通过；dry-run 未访问接口。")
        return 0

    handoff_markers = (*HANDOFF_MARKERS, *args.handoff_marker)
    global_forbidden = list(dataset.get("global_forbidden_keywords", []))
    jobs = [(case, run) for run in range(1, args.repeats + 1) for case in cases]
    samples: list[dict[str, Any]] = []
    if args.rescore:
        previous = json.loads(args.rescore.read_text(encoding="utf-8"))
        samples = rescore_samples(
            previous["samples"], cases, handoff_markers=handoff_markers, global_forbidden=global_forbidden
        )
        jobs = []
        print(f"已按当前标注重新打分 {len(samples)} 个会话（来源：{args.rescore}）")
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                run_case,
                case,
                base_url=args.base_url,
                tenant_id=args.tenant_id,
                tenant_token=args.tenant_token,
                timeout=args.timeout,
                run_number=run,
                handoff_markers=handoff_markers,
                global_forbidden=global_forbidden,
            )
            for case, run in jobs
        ]
        for future in as_completed(futures):
            sample = future.result()
            samples.append(sample)
            print(f"进度: {len(samples)}/{len(jobs)} ({sample['case_id']})", flush=True)
    samples.sort(key=lambda item: (item["run_number"], item["case_id"]))

    report = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "dataset": {
            "name": dataset["name"],
            "version": dataset["version"],
            "path": str(args.dataset.resolve()),
            "cases": len(cases),
        },
        "config": {
            "base_url": args.base_url,
            "tenant_id": args.tenant_id,
            "repeats": args.repeats,
            "concurrency": args.concurrency,
            "handoff_markers": list(handoff_markers),
            "rescored_from": str(args.rescore) if args.rescore else None,
        },
        "summary": aggregate(samples),
        "samples": samples,
    }
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = args.output or DEFAULT_RESULTS_DIR / f"grounded-handoff-{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")
    print(f"JSON 报告: {output.resolve()}")
    print(f"Markdown 报告: {output.with_suffix('.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
