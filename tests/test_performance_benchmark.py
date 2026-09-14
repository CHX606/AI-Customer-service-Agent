"""性能基准工具的离线单元测试。"""

import json
from pathlib import Path
from unittest.mock import patch

from evaluation.benchmarks import performance as benchmark


def load_cases():
    return benchmark.load_dataset(benchmark.DEFAULT_DATASET_PATH)


def test_dataset_contains_production_and_external_app_coverage():
    dataset = load_cases()
    cases = dataset["cases"]

    assert len(cases) >= 55
    assert len({case["id"] for case in cases}) == len(cases)
    assert sum(case["category"] == "external_x_twitter" for case in cases) >= 6
    assert any("怎么登录X" in case["query"] for case in cases)
    assert all(
        case["source_expectation"] == "knowledge_base"
        for case in cases
        if case["id"].startswith("external_")
    )


def test_external_app_cases_have_retrieval_ground_truth_after_knowledge_expansion():
    cases = load_cases()["cases"]
    external = [case for case in cases if case["id"].startswith("external_")]

    assert external
    assert all(case["expected_retrieval"] for case in external)
    assert all(case["expected_sections"] for case in external)


def test_suite_and_category_selection():
    dataset = load_cases()
    smoke = benchmark.select_cases(dataset, suite="smoke")
    twitter = benchmark.select_cases(
        dataset, suite="full", categories=["external_x_twitter"]
    )

    assert 8 <= len(smoke) < len(dataset["cases"])
    assert twitter
    assert all(case["category"] == "external_x_twitter" for case in twitter)


def test_percentile_and_latency_summary():
    assert benchmark.percentile([], 0.95) is None
    assert benchmark.percentile([10], 0.95) == 10
    summary = benchmark.latency_summary([10, 20, 30, None])

    assert summary["count"] == 3
    assert summary["p50"] == 20
    assert summary["p95"] == 29


def test_score_answer_detects_expected_behavior_and_keyword_groups():
    case = {
        "expected_behavior": "answer",
        "required_keyword_groups": [["X", "Twitter"], ["密码", "登录"]],
    }
    good = benchmark.score_answer(case, "打开X（Twitter）后输入账号和密码登录。")
    rejected = benchmark.score_answer(case, "这个问题超出业务范围，无法回答这个问题。")

    assert good["quality_pass"] is True
    assert rejected["quality_pass"] is False


class FakeStreamingResponse:
    def __init__(self, events):
        self.lines = [
            (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
            for event in events
        ]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.lines)


def test_http_case_measures_retrieval_route_and_final_answer():
    case = {
        "id": "fake",
        "category": "external_x_twitter",
        "query": "怎么登录X？",
        "expected_behavior": "answer",
        "expected_retrieval": True,
        "source_expectation": "coverage_gap",
        "required_keyword_groups": [["X"], ["密码", "登录"]],
    }
    response = FakeStreamingResponse(
        [
            {"type": "status", "message": benchmark.STATUS_ANALYZING},
            {
                "type": "status",
                "message": benchmark.STATUS_RETRIEVING,
                "route_source": "fast_rule",
            },
            {"type": "status", "message": benchmark.STATUS_RERANKING},
            {
                "type": "status",
                "message": benchmark.STATUS_GENERATING,
                "reranker_queue_wait_ms": 12.5,
                "reranker_inference_ms": 980.0,
                "reranker_max_concurrency": 2,
            },
            {"type": "token", "delta": "打开"},
            {"type": "final", "answer": "打开X，输入账号和密码即可登录。"},
        ]
    )
    with patch.object(benchmark, "urlopen", return_value=response):
        result = benchmark.run_http_case(
            case,
            base_url="http://127.0.0.1:8000",
            tenant_id="default",
            tenant_token=None,
            timeout=1,
            run_number=1,
        )

    assert result["success"] is True
    assert result["actual_retrieval"] is True
    assert result["route_pass"] is True
    assert result["quality_pass"] is True
    assert result["routing_ms"] is not None
    assert result["route_source"] == "fast_rule"
    assert result["reranker_queue_wait_ms"] == 12.5
    assert result["reranker_inference_ms"] == 980.0
    assert result["retrieval_stage_ms"] is not None
    assert result["first_token_ms"] is not None
    assert result["time_to_answer_ms"] is not None


def test_aggregate_http_reports_knowledge_gap_diagnosis():
    samples = [
        {
            "case_id": "x1",
            "category": "external_x_twitter",
            "success": True,
            "quality_pass": False,
            "route_source": "fast_rule",
            "failure_reasons": ["knowledge_gap_or_scope_routing"],
            "first_event_ms": 10,
            "first_token_ms": 90,
            "time_to_answer_ms": 100,
            "routing_ms": 12,
            "retrieval_stage_ms": None,
            "rerank_stage_ms": None,
            "reranker_queue_wait_ms": 12,
            "reranker_inference_ms": 980,
            "generation_stage_ms": 80,
        }
    ]

    summary = benchmark.aggregate_http(samples)

    assert summary["error_rate"] == 0
    assert summary["quality_pass_rate"] == 0
    assert summary["routing_stage_ms"]["p95"] == 12
    assert summary["first_token_latency_ms"]["p95"] == 90
    assert summary["route_source_counts"] == {"fast_rule": 1}
    assert summary["reranker_queue_wait_ms"]["p95"] == 12
    assert summary["failure_reasons"]["knowledge_gap_or_scope_routing"] == 1


def test_dry_run_validates_dataset_without_external_services(tmp_path: Path):
    exit_code = benchmark.main(
        [
            "--dataset",
            str(benchmark.DEFAULT_DATASET_PATH),
            "--suite",
            "smoke",
            "--dry-run",
            "--output",
            str(tmp_path / "unused.json"),
        ]
    )

    assert exit_code == 0
    assert not (tmp_path / "unused.json").exists()
