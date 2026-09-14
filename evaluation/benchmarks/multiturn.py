"""客服追问闭环端到端基准测试。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

from evaluation.benchmarks.performance import latency_summary, run_http_case


load_dotenv()


EVALUATION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = EVALUATION_DIR / "datasets" / "customer_service_multiturn.json"
DEFAULT_RESULTS_DIR = EVALUATION_DIR / "results"


def _contains_any(text: str, choices: list[str]) -> bool:
    normalized = text.casefold()
    return any(choice.casefold() in normalized for choice in choices)


def load_multiturn_dataset(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("多轮测试集必须包含非空 cases 列表")

    seen: set[str] = set()
    required_turn = {
        "message",
        "expected_behavior",
        "expected_retrieval",
        "source_expectation",
        "required_keyword_groups",
    }
    for case in cases:
        case_id = str(case.get("id", "")).strip()
        if not case_id or case_id in seen:
            raise ValueError(f"多轮用例 ID 缺失或重复: {case_id!r}")
        seen.add(case_id)
        turns = case.get("turns")
        if not isinstance(turns, list) or len(turns) < 2:
            raise ValueError(f"多轮用例至少需要两轮: {case_id}")
        for index, turn in enumerate(turns, start=1):
            missing = required_turn - set(turn)
            if missing:
                raise ValueError(
                    f"多轮用例 {case_id} 第 {index} 轮缺少字段: {sorted(missing)}"
                )
            if not str(turn["message"]).strip():
                raise ValueError(f"多轮用例 {case_id} 第 {index} 轮消息为空")
    return data


def select_conversations(
    dataset: dict[str, Any], case_ids: list[str] | None = None
) -> list[dict[str, Any]]:
    selected_ids = set(case_ids or [])
    selected = [
        case
        for case in dataset["cases"]
        if not selected_ids or case["id"] in selected_ids
    ]
    missing = selected_ids - {case["id"] for case in selected}
    if missing:
        raise ValueError(f"未找到多轮用例: {sorted(missing)}")
    return selected


def _turn_case(
    conversation: dict[str, Any], turn: dict[str, Any], turn_number: int
) -> dict[str, Any]:
    return {
        "id": f"{conversation['id']}-turn-{turn_number}",
        "category": conversation["category"],
        "query": turn["message"],
        "expected_behavior": turn["expected_behavior"],
        "expected_retrieval": turn["expected_retrieval"],
        "source_expectation": turn["source_expectation"],
        "required_keyword_groups": turn.get("required_keyword_groups", []),
        "forbidden_keywords": turn.get("forbidden_keywords", []),
    }


def run_conversation(
    conversation: dict[str, Any],
    *,
    base_url: str,
    tenant_id: str,
    tenant_token: str | None,
    timeout: float,
    run_number: int,
) -> dict[str, Any]:
    session_id = (
        f"multiturn-{conversation['id']}-{run_number}-{uuid4().hex[:10]}"
    )
    samples: list[dict[str, Any]] = []
    for turn_number, turn in enumerate(conversation["turns"], start=1):
        sample = run_http_case(
            _turn_case(conversation, turn, turn_number),
            base_url=base_url,
            tenant_id=tenant_id,
            tenant_token=tenant_token,
            timeout=timeout,
            run_number=run_number,
            session_id=session_id,
        )
        repeated_groups = [
            group
            for group in turn.get("forbidden_repeat_groups", [])
            if _contains_any(sample.get("answer", ""), group)
        ]
        sample.update(
            {
                "turn_number": turn_number,
                "message": turn["message"],
                "repeated_information_groups": repeated_groups,
                "context_retention_pass": not repeated_groups,
            }
        )
        samples.append(sample)

    first = samples[0]
    final = samples[-1]
    first_followup_pass = bool(
        first.get("success")
        and first.get("answer_score", {}).get("behavior_pass")
    )
    final_resolution_pass = bool(final.get("quality_pass"))
    context_retention_pass = all(
        sample["context_retention_pass"] for sample in samples[1:]
    )
    success = all(sample.get("success") for sample in samples)
    return {
        "case_id": conversation["id"],
        "category": conversation["category"],
        "run_number": run_number,
        "session_id": session_id,
        "success": success,
        "first_followup_pass": first_followup_pass,
        "final_resolution_pass": final_resolution_pass,
        "context_retention_pass": context_retention_pass,
        "conversation_pass": bool(
            success
            and first_followup_pass
            and final_resolution_pass
            and context_retention_pass
        ),
        "final_answer_ms": final.get("time_to_answer_ms"),
        "turns": samples,
    }


def _rate(values: list[bool]) -> float:
    return round(mean(int(value) for value in values), 4) if values else 0.0


def aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runs": len(samples),
        "success_rate": _rate([sample["success"] for sample in samples]),
        "first_turn_followup_rate": _rate(
            [sample["first_followup_pass"] for sample in samples]
        ),
        "final_resolution_rate": _rate(
            [sample["final_resolution_pass"] for sample in samples]
        ),
        "context_retention_rate": _rate(
            [sample["context_retention_pass"] for sample in samples]
        ),
        "conversation_pass_rate": _rate(
            [sample["conversation_pass"] for sample in samples]
        ),
        "final_answer_latency_ms": latency_summary(
            [sample["final_answer_ms"] for sample in samples]
        ),
    }


def evaluate_thresholds(
    summary: dict[str, Any], thresholds: dict[str, Any]
) -> list[dict[str, Any]]:
    checks = [
        (
            "首轮合理追问率",
            summary["first_turn_followup_rate"],
            thresholds["first_turn_followup_rate"],
            ">=",
        ),
        (
            "补充信息后解决率",
            summary["final_resolution_rate"],
            thresholds["final_resolution_rate"],
            ">=",
        ),
        (
            "上下文保留率",
            summary["context_retention_rate"],
            thresholds["context_retention_rate"],
            ">=",
        ),
        (
            "接口错误率",
            1 - summary["success_rate"],
            thresholds["error_rate"],
            "<=",
        ),
        (
            "最终回答 P95(ms)",
            summary["final_answer_latency_ms"]["p95"],
            thresholds["final_answer_p95_ms"],
            "<=",
        ),
    ]
    return [
        {
            "name": name,
            "actual": actual,
            "target": target,
            "operator": operator,
            "passed": actual is not None
            and (actual >= target if operator == ">=" else actual <= target),
        }
        for name, actual, target, operator in checks
    ]


def build_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# 客服追问闭环测试报告",
        "",
        f"- 结果：**{'PASS' if report['passed'] else 'FAIL'}**",
        f"- 会话数：{summary['runs']}",
        f"- 首轮合理追问率：{summary['first_turn_followup_rate']:.2%}",
        f"- 补充信息后解决率：{summary['final_resolution_rate']:.2%}",
        f"- 上下文保留率：{summary['context_retention_rate']:.2%}",
        f"- 完整闭环通过率：{summary['conversation_pass_rate']:.2%}",
        f"- 最终回答 P95：{summary['final_answer_latency_ms']['p95']} ms",
        "",
        "| 用例 | 追问 | 最终解决 | 保留上下文 | 结果 |",
        "|---|---|---|---|---|",
    ]
    for sample in report["samples"]:
        lines.append(
            f"| {sample['case_id']} | "
            f"{'PASS' if sample['first_followup_pass'] else 'FAIL'} | "
            f"{'PASS' if sample['final_resolution_pass'] else 'FAIL'} | "
            f"{'PASS' if sample['context_retention_pass'] else 'FAIL'} | "
            f"{'PASS' if sample['conversation_pass'] else 'FAIL'} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--tenant-id", default="default")
    parser.add_argument(
        "--tenant-token", default=os.getenv("BENCHMARK_TENANT_TOKEN")
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enforce", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repeats < 1 or args.concurrency < 1:
        raise ValueError("repeats/concurrency 必须至少为 1")

    dataset = load_multiturn_dataset(args.dataset)
    cases = select_conversations(dataset, args.case_ids)
    print(f"已加载 {len(cases)} 条多轮用例")
    if args.dry_run:
        print("多轮测试集校验通过；dry-run 未访问接口。")
        return 0

    jobs = [
        (case, run_number)
        for run_number in range(1, args.repeats + 1)
        for case in cases
    ]
    samples: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                run_conversation,
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
            sample = future.result()
            samples.append(sample)
            print(
                f"多轮评测进度: {len(samples)}/{len(jobs)} "
                f"({sample['case_id']})",
                flush=True,
            )
    samples.sort(key=lambda item: (item["run_number"], item["case_id"]))
    summary = aggregate(samples)
    checks = evaluate_thresholds(summary, dataset["thresholds"])
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
            "timeout": args.timeout,
        },
        "summary": summary,
        "threshold_checks": checks,
        "samples": samples,
        "passed": all(check["passed"] for check in checks),
    }

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = args.output or DEFAULT_RESULTS_DIR / f"multiturn-{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(build_markdown(report), encoding="utf-8")
    print(f"JSON 报告: {output.resolve()}")
    print(f"Markdown 报告: {output.with_suffix('.md').resolve()}")
    print(f"结果: {'PASS' if report['passed'] else 'FAIL'}")
    return 1 if args.enforce and not report["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
