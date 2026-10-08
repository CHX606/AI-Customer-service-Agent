"""有依据回答与转人工评测工具的离线测试。"""

import json
from pathlib import Path

import pytest

from back.agent.workflow.response_nodes import NOT_FOUND_RETRY, SUPPORT_REQUEST_PROMPT
from back.tenant.defaults import DEFAULT_KELECLOUD_PROFILE
from evaluation.benchmarks import grounded_handoff


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (DEFAULT_KELECLOUD_PROFILE.handoff_message, "handoff"),
        ("抱歉，目前检索到的资料存在不一致，我无法安全地给出确定结论，建议联系人工客服核实。", "handoff"),
        (NOT_FOUND_RETRY, "retry"),
        (SUPPORT_REQUEST_PROMPT, "handoff"),
        (f"  {SUPPORT_REQUEST_PROMPT}\n", "handoff"),
        (f"续费只延长有效期，具体赠送流量需要人工确认。\n{SUPPORT_REQUEST_PROMPT}", "answer"),
        ("需要办理的事项可以点击“提交人工处理”，普通续费规则按官网执行。", "answer"),
        ("抱歉，我目前只能处理可乐云相关的账号注册与登录管理等客服咨询。", "out_of_scope"),
        ("把问题说的具体一点，并且附上问题截图", "clarify"),
        ("请问您使用的是什么设备和客户端？", "clarify"),
        ("续费只延长有效期，不会立即刷新流量；流量用完需要购买流量重置。", "answer"),
        ("续费和重置的区别是……重置的价格资料中没有说明，建议联系人工客服确认。", "answer"),
    ],
)
def test_classify_answer_matches_production_messages(answer, expected):
    assert grounded_handoff.classify_answer(answer) == expected


def test_dataset_is_valid_and_covers_both_directions():
    dataset = grounded_handoff.load_dataset(grounded_handoff.DEFAULT_DATASET_PATH)
    turns = [turn for case in dataset["cases"] for turn in case["turns"]]

    assert sum(turn["expect"] == ["answer"] for turn in turns) >= 20
    assert sum("answer" not in turn["expect"] for turn in turns) >= 8


def test_evaluate_turn_checks_keywords_and_internal_leaks():
    turn = {"message": "能退钱吗", "expect": ["answer"], "required_keyword_groups": [["余额"]]}

    ok = grounded_handoff.evaluate_turn(turn, "不支持退款，最多退回网站余额。", global_forbidden=["快捷键"])
    missing = grounded_handoff.evaluate_turn(turn, "不支持退款。", global_forbidden=["快捷键"])
    leaked = grounded_handoff.evaluate_turn(turn, "请发送改邮箱快捷键，最多退余额。", global_forbidden=["快捷键"])

    assert ok["pass"] is True
    assert missing["pass"] is False and missing["missing_keyword_groups"] == [["余额"]]
    assert leaked["pass"] is False and leaked["matched_forbidden_keywords"] == ["快捷键"]


def test_run_case_reuses_session_and_scores_retry_then_handoff(monkeypatch):
    sessions = []
    answers = iter([NOT_FOUND_RETRY, DEFAULT_KELECLOUD_PROFILE.handoff_message])

    def fake_run_http_case(*_args, session_id=None, **_kwargs):
        sessions.append(session_id)
        return {"success": True, "answer": next(answers), "time_to_answer_ms": 100}

    monkeypatch.setattr(grounded_handoff, "run_http_case", fake_run_http_case)
    case = {
        "id": "ipv6",
        "category": "out_of_kb",
        "flow": "retry_then_handoff",
        "turns": [
            {"message": "支持 IPv6 吗", "expect": ["retry", "clarify", "handoff"]},
            {"message": "纯 IPv6 能连吗", "expect": ["retry", "clarify", "handoff"]},
        ],
    }

    result = grounded_handoff.run_case(
        case,
        base_url="http://127.0.0.1:8000",
        tenant_id="default",
        tenant_token=None,
        timeout=1,
        run_number=1,
        handoff_markers=grounded_handoff.HANDOFF_MARKERS,
        global_forbidden=[],
    )
    summary = grounded_handoff.aggregate([result])

    assert sessions[0] == sessions[1]
    assert result["pass"] is True and result["flow_pass"] is True
    assert summary["retry_then_handoff_rate"] == 1.0
    assert summary["wrongly_answered_rate"] == 0.0
    assert summary["unnecessary_handoff_rate"] is None


def test_aggregate_counts_unnecessary_handoff():
    turn = {
        "outcome": "handoff",
        "expected": ["answer"],
        "missing_keyword_groups": [],
        "matched_forbidden_keywords": [],
        "pass": False,
        "route_source": None,
        "time_to_answer_ms": 100,
    }
    sample = {"category": "colloquial_should_answer", "pass": False, "flow_pass": None, "turns": [turn]}

    summary = grounded_handoff.aggregate([sample])

    assert summary["unnecessary_handoff_rate"] == 1.0
    assert summary["should_answer_answer_rate"] == 0.0


def test_rescore_applies_new_labels_to_saved_answers_without_http(tmp_path: Path, monkeypatch):
    dataset = {
        "version": 1,
        "name": "rescore",
        "cases": [
            {"id": "keep", "category": "c", "turns": [{"message": "ios电脑怎么用", "expect": ["answer", "clarify"]}]},
        ],
    }
    report = {
        "samples": [
            {
                "case_id": case_id,
                "run_number": 1,
                "session_id": f"s-{case_id}",
                "turns": [{
                    "turn_number": 1,
                    "message": "ios电脑怎么用",
                    "answer": "请问你说的是 Mac 还是 iPhone？",
                    "route_source": None,
                    "time_to_answer_ms": 100,
                    "success": True,
                    "error": None,
                    "outcome": "clarify",
                    "expected": ["answer"],
                    "pass": False,
                }],
            }
            for case_id in ("keep", "removed")
        ]
    }
    dataset_path = tmp_path / "dataset.json"
    report_path = tmp_path / "old.json"
    output = tmp_path / "new.json"
    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(grounded_handoff, "run_http_case", lambda *a, **k: pytest.fail("不应访问接口"))

    grounded_handoff.main([
        "--dataset", str(dataset_path), "--rescore", str(report_path), "--output", str(output),
    ])

    result = json.loads(output.read_text(encoding="utf-8"))
    assert [sample["case_id"] for sample in result["samples"]] == ["keep"]
    assert result["samples"][0]["pass"] is True
    assert result["summary"]["turn_pass_rate"] == 1.0


def test_dry_run_validates_without_http(tmp_path: Path):
    exit_code = grounded_handoff.main(["--dry-run", "--output", str(tmp_path / "unused.json")])

    assert exit_code == 0
    assert not (tmp_path / "unused.json").exists()


@pytest.mark.parametrize("required,reason,expected", [
    (True, "explicit_request", "handoff"),
    (True, "not_found", "handoff"),
    (True, "conflict", "handoff"),
    (True, "partial", "answer"),
    (True, "human_action", "answer"),
    (False, "explicit_request", "answer"),
    (True, "unknown", "answer"),
])
def test_structured_support_metadata_distinguishes_handoff_from_supported_answer(required, reason, expected):
    assert grounded_handoff.classify_answer(
        "明确的业务答复", support_required=required, support_reason=reason,
    ) == expected


def test_partial_metadata_keeps_answer_scoring_and_required_keywords():
    turn = {"message": "续费重置价格是多少", "expect": ["answer"],
            "required_keyword_groups": [["续费"]]}
    result = grounded_handoff.evaluate_turn(
        turn, f"续费延长有效期，重置价格暂未查到，需要人工确认。\n{SUPPORT_REQUEST_PROMPT}",
        support_required=True, support_reason="partial",
    )
    assert result["outcome"] == "answer"
    assert result["pass"] is True


def test_run_case_retains_available_metadata_for_future_rescoring(monkeypatch):
    monkeypatch.setattr(grounded_handoff, "run_http_case", lambda *args, **kwargs: {
        "success": True, "answer": "申请入口提示文案已更新", "time_to_answer_ms": 100,
        "support_required": True, "support_reason": "explicit_request",
    })
    case = {"id": "direct-human", "category": "human_request",
            "turns": [{"message": "转人工客服", "expect": ["handoff"]}]}
    sample = grounded_handoff.run_case(
        case, base_url="http://127.0.0.1:8000", tenant_id="default", tenant_token=None,
        timeout=1, run_number=1, handoff_markers=grounded_handoff.HANDOFF_MARKERS, global_forbidden=[],
    )
    turn = sample["turns"][0]
    assert turn["support_required"] is True
    assert turn["support_reason"] == "explicit_request"
    assert turn["outcome"] == "handoff"
    assert sample["pass"] is True
    rescored = grounded_handoff.rescore_samples(
        [sample], [case], handoff_markers=grounded_handoff.HANDOFF_MARKERS, global_forbidden=[],
    )
    assert rescored[0]["turns"][0]["outcome"] == "handoff"
    assert rescored[0]["pass"] is True
