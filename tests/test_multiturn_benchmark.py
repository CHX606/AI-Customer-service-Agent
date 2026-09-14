"""追问闭环基准工具的离线测试。"""

from pathlib import Path

from evaluation.benchmarks import multiturn


def test_multiturn_dataset_contains_resolution_flows():
    dataset = multiturn.load_multiturn_dataset(multiturn.DEFAULT_DATASET_PATH)

    assert len(dataset["cases"]) >= 8
    assert all(len(case["turns"]) >= 2 for case in dataset["cases"])
    assert all(case["turns"][0]["expected_behavior"] == "clarify" for case in dataset["cases"])
    assert all(case["turns"][-1]["expected_behavior"] == "answer" for case in dataset["cases"])


def test_multiturn_case_selection():
    dataset = multiturn.load_multiturn_dataset(multiturn.DEFAULT_DATASET_PATH)
    selected = multiturn.select_conversations(
        dataset, ["clarify_site_to_chatgpt_blank"]
    )

    assert [case["id"] for case in selected] == ["clarify_site_to_chatgpt_blank"]
    assert len(selected[0]["turns"]) == 3


def test_run_conversation_reuses_session_and_scores_final_resolution(monkeypatch):
    sessions = []
    answers = iter(
        [
            {
                "success": True,
                "answer": "请问您使用什么设备和软件？",
                "answer_score": {"behavior_pass": True},
                "quality_pass": True,
                "time_to_answer_ms": 100,
            },
            {
                "success": True,
                "answer": "请在 Windows 的 Clash 中检查系统代理和节点。",
                "answer_score": {"behavior_pass": True},
                "quality_pass": True,
                "time_to_answer_ms": 200,
            },
        ]
    )

    def fake_run_http_case(*_args, session_id=None, **_kwargs):
        sessions.append(session_id)
        return next(answers)

    monkeypatch.setattr(multiturn, "run_http_case", fake_run_http_case)
    case = {
        "id": "same-session",
        "category": "clarification_resolution",
        "turns": [
            {
                "message": "不能用",
                "expected_behavior": "clarify",
                "expected_retrieval": False,
                "source_expectation": "no_retrieval",
                "required_keyword_groups": [],
            },
            {
                "message": "Windows，Clash",
                "expected_behavior": "answer",
                "expected_retrieval": True,
                "source_expectation": "knowledge_base",
                "required_keyword_groups": [["Windows"]],
                "forbidden_repeat_groups": [["什么设备"]],
            },
        ],
    }

    result = multiturn.run_conversation(
        case,
        base_url="http://127.0.0.1:8000",
        tenant_id="default",
        tenant_token=None,
        timeout=1,
        run_number=1,
    )

    assert sessions[0] == sessions[1]
    assert result["first_followup_pass"] is True
    assert result["final_resolution_pass"] is True
    assert result["context_retention_pass"] is True
    assert result["conversation_pass"] is True


def test_multiturn_dry_run_validates_without_http(tmp_path: Path):
    exit_code = multiturn.main(
        [
            "--dataset",
            str(multiturn.DEFAULT_DATASET_PATH),
            "--dry-run",
            "--output",
            str(tmp_path / "unused.json"),
        ]
    )

    assert exit_code == 0
    assert not (tmp_path / "unused.json").exists()
