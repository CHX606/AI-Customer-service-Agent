"""证据充分性判断及后续路由单元测试（Mock 模型调用）。"""

from unittest.mock import patch

from langchain_core.documents import Document

from back.agent.analysis.evidence import EvidenceAnalysis
from back.agent.workflow.legacy import evaluate_evidence_node, route_after_evidence


def _make_dummy_docs(n=3):
    return [
        Document(
            page_content=f"第{i}份资料正文内容",
            metadata={
                "section_id": f"1.{i}",
                "section_title": f"章节{i}",
            },
        )
        for i in range(1, n + 1)
    ]


def test_sufficient_evidence_with_valid_indexes():
    docs = _make_dummy_docs(3)
    mock_result = EvidenceAnalysis(
        evidence_status="sufficient",
        supporting_document_indexes=[1, 3],
        missing_information=[],
        clarifying_question=None,
        decision_reason="资料1和3明确支持回答",
    )

    with patch("back.agent.workflow.legacy.evaluate_evidence", return_value=mock_result):
        state = {
            "resolved_query": "续费后会有流量吗？",
            "intent": "traffic",
            "retrieved_documents": docs,
        }
        node_result = evaluate_evidence_node(state)
        merged_state = {**state, **node_result}

        assert node_result["evidence_status"] == "sufficient"
        assert node_result["supporting_document_indexes"] == [1, 3]
        assert len(node_result["supporting_documents"]) == 2
        assert node_result["supporting_documents"][0].page_content == "第1份资料正文内容"
        assert node_result["supporting_documents"][1].page_content == "第3份资料正文内容"

        next_route = route_after_evidence(merged_state)
        assert next_route == "answer"


def test_sufficient_evidence_with_invalid_indexes_downgrades():
    docs = _make_dummy_docs(2)
    mock_result = EvidenceAnalysis(
        evidence_status="sufficient",
        supporting_document_indexes=[99],  # 无效编号
        missing_information=[],
        clarifying_question=None,
        decision_reason="模型认为充分但编号越界",
    )

    with patch("back.agent.workflow.legacy.evaluate_evidence", return_value=mock_result):
        state = {
            "resolved_query": "某个问题",
            "intent": "traffic",
            "retrieved_documents": docs,
        }
        node_result = evaluate_evidence_node(state)
        merged_state = {**state, **node_result}

        assert node_result["evidence_status"] == "not_found"
        assert node_result["supporting_document_indexes"] == []
        assert node_result["supporting_documents"] == []

        next_route = route_after_evidence(merged_state)
        assert next_route == "handoff"


def test_insufficient_evidence_routes_to_clarify():
    docs = _make_dummy_docs(2)
    mock_result = EvidenceAnalysis(
        evidence_status="insufficient",
        supporting_document_indexes=[1],
        missing_information=["客户端具体报错信息"],
        clarifying_question="请问客户端提示的具体错误代码是什么？",
        decision_reason="缺少关键客户端状态",
    )

    with patch("back.agent.workflow.legacy.evaluate_evidence", return_value=mock_result):
        state = {
            "resolved_query": "节点连不上",
            "intent": "node_connection",
            "retrieved_documents": docs,
        }
        node_result = evaluate_evidence_node(state)
        merged_state = {**state, **node_result}

        assert node_result["evidence_status"] == "insufficient"
        assert node_result["evidence_clarifying_question"] == "请问客户端提示的具体错误代码是什么？"
        assert node_result["evidence_missing_information"] == ["客户端具体报错信息"]

        next_route = route_after_evidence(merged_state)
        assert next_route == "evidence_clarify"


def test_not_found_routes_to_handoff():
    docs = _make_dummy_docs(2)
    mock_result = EvidenceAnalysis(
        evidence_status="not_found",
        supporting_document_indexes=[],
        missing_information=[],
        clarifying_question=None,
        decision_reason="资料中无相关信息",
    )

    with patch("back.agent.workflow.legacy.evaluate_evidence", return_value=mock_result):
        state = {
            "resolved_query": "续费会赠送多少GB流量？",
            "intent": "traffic",
            "retrieved_documents": docs,
        }
        node_result = evaluate_evidence_node(state)
        merged_state = {**state, **node_result}

        assert node_result["evidence_status"] == "not_found"
        assert node_result["supporting_documents"] == []

        next_route = route_after_evidence(merged_state)
        assert next_route == "handoff"


def test_conflict_routes_to_handoff():
    docs = _make_dummy_docs(3)
    mock_result = EvidenceAnalysis(
        evidence_status="conflict",
        supporting_document_indexes=[1, 2],
        missing_information=[],
        clarifying_question=None,
        decision_reason="两份资料规则互相冲突",
    )

    with patch("back.agent.workflow.legacy.evaluate_evidence", return_value=mock_result):
        state = {
            "resolved_query": "冲突规则测试",
            "intent": "other_business",
            "retrieved_documents": docs,
        }
        node_result = evaluate_evidence_node(state)
        merged_state = {**state, **node_result}

        assert node_result["evidence_status"] == "conflict"

        next_route = route_after_evidence(merged_state)
        assert next_route == "handoff"
