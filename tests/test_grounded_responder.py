"""合并证据判断与回答生成的安全性测试。"""

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from back.agent.workflow.response_nodes import respond_with_grounded_knowledge
from back.agent.response.grounded import (
    GroundedResponse,
    generate_grounded_response,
    normalize_grounded_response,
    sanitize_grounded_answer,
)


def docs(count=3):
    return [
        Document(
            page_content=f"第{i}份知识库正文",
            metadata={"section_id": str(i), "section_title": f"章节{i}"},
        )
        for i in range(1, count + 1)
    ]


def response(**updates):
    data = {
        "evidence_status": "sufficient",
        "answer": "续费只延长有效期，流量按原套餐周期重置。",
        "supporting_document_indexes": [1],
        "missing_information": [],
        "clarifying_question": None,
        "decision_reason": "资料1直接说明续费规则。",
    }
    data.update(updates)
    return GroundedResponse(**data)


class StructuredRunner:
    def __init__(self, parent):
        self.parent = parent

    def invoke(self, messages):
        self.parent.calls += 1
        self.parent.messages = messages
        return self.parent.result


class FakeModel:
    def __init__(self, result):
        self.result = result
        self.calls = 0
        self.schema = None
        self.messages = []

    def with_structured_output(self, schema):
        self.schema = schema
        return StructuredRunner(self)


def test_generate_grounded_response_uses_one_structured_call():
    fake = FakeModel(response())
    result = generate_grounded_response(
        llm=fake,
        resolved_query="续费后为什么流量没重置？",
        intent="renewal",
        documents=docs(2),
        company_name="测试企业",
        assistant_name="测试客服",
        tone="简洁专业",
    )

    assert fake.calls == 1
    assert fake.schema is GroundedResponse
    assert result.evidence_status == "sufficient"
    assert "测试企业" in fake.messages[0].content
    assert "第1份知识库正文" in fake.messages[1].content
    assert "续费后为什么流量没重置" in fake.messages[1].content


def test_valid_support_indexes_are_deduplicated_in_order():
    result = normalize_grounded_response(
        response(supporting_document_indexes=[2, 2, 1]),
        document_count=3,
    )
    assert result.supporting_document_indexes == [2, 1]


@pytest.mark.parametrize("indexes", [[], [0], [4], [-1, 99]])
def test_sufficient_without_valid_support_is_downgraded(indexes):
    result = normalize_grounded_response(
        response(supporting_document_indexes=indexes),
        document_count=3,
    )
    assert result.evidence_status == "not_found"
    assert result.answer is None
    assert result.supporting_document_indexes == []


@pytest.mark.parametrize("answer", [None, "", "   "])
def test_sufficient_without_answer_is_downgraded(answer):
    result = normalize_grounded_response(
        response(answer=answer),
        document_count=3,
    )
    assert result.evidence_status == "not_found"
    assert result.answer is None


def test_invalid_support_indexes_are_removed_but_valid_answer_remains():
    result = normalize_grounded_response(
        response(supporting_document_indexes=[0, 2, 9]),
        document_count=3,
    )
    assert result.evidence_status == "sufficient"
    assert result.supporting_document_indexes == [2]


@pytest.mark.parametrize("indexes", [[], [1], [1, 1], [0, 2]])
def test_conflict_requires_two_distinct_valid_documents(indexes):
    result = normalize_grounded_response(
        response(
            evidence_status="conflict",
            answer=None,
            supporting_document_indexes=indexes,
        ),
        document_count=3,
    )
    assert result.evidence_status == "not_found"
    assert result.supporting_document_indexes == []


def test_valid_conflict_keeps_two_documents_and_clears_answer():
    result = normalize_grounded_response(
        response(
            evidence_status="conflict",
            answer="不应直接输出",
            supporting_document_indexes=[1, 2],
        ),
        document_count=3,
    )
    assert result.evidence_status == "conflict"
    assert result.answer is None
    assert result.supporting_document_indexes == [1, 2]


def test_insufficient_response_gets_fallback_question_and_missing_info():
    result = normalize_grounded_response(
        response(
            evidence_status="insufficient",
            answer="不应输出",
            supporting_document_indexes=[1],
            missing_information=[],
            clarifying_question=None,
        ),
        document_count=3,
    )
    assert result.answer is None
    assert result.supporting_document_indexes == []
    assert result.missing_information
    assert result.clarifying_question


@pytest.mark.parametrize("status", ["not_found", "conflict"])
def test_non_answer_status_clears_question_and_missing_fields(status):
    indexes = [1, 2] if status == "conflict" else [1]
    result = normalize_grounded_response(
        response(
            evidence_status=status,
            answer="不应输出",
            supporting_document_indexes=indexes,
            missing_information=["不应保留"],
            clarifying_question="不应保留",
        ),
        document_count=3,
    )
    assert result.answer is None
    assert result.missing_information == []
    assert result.clarifying_question is None


@pytest.mark.parametrize(
    ("grounded", "expected_status", "expected_issue_status", "expected_text"),
    [
        (response(), "sufficient", "answered", "续费只延长有效期"),
        (
            response(
                evidence_status="insufficient",
                answer=None,
                supporting_document_indexes=[],
                missing_information=["客户端报错"],
                clarifying_question="请问客户端显示什么错误？",
            ),
            "insufficient",
            "awaiting_user",
            "请问客户端显示什么错误",
        ),
        (
            response(
                evidence_status="not_found",
                answer=None,
                supporting_document_indexes=[],
            ),
            "not_found",
            "handed_off",
            "人工",
        ),
        (
            response(
                evidence_status="conflict",
                answer=None,
                supporting_document_indexes=[1, 2],
            ),
            "conflict",
            "handed_off",
            "不一致",
        ),
    ],
)
def test_graph_node_maps_grounded_outcome_to_safe_customer_message(
    grounded,
    expected_status,
    expected_issue_status,
    expected_text,
):
    state = {
        "tenant_id": "default",
        "resolved_query": "测试问题",
        "intent": "traffic",
        "retrieved_documents": docs(3),
        "active_issue": {"summary": "测试问题", "status": "open"},
    }
    with patch(
        "back.agent.workflow.response_nodes.generate_grounded_response",
        return_value=grounded,
    ):
        result = respond_with_grounded_knowledge(state)

    assert result["evidence_status"] == expected_status
    assert result["active_issue"]["status"] == expected_issue_status
    assert expected_text in result["messages"][0].content


def test_no_retrieved_documents_returns_handoff_without_model_call():
    state = {
        "tenant_id": "default",
        "resolved_query": "知识库没有的问题",
        "intent": "other_business",
        "retrieved_documents": [],
        "active_issue": {"summary": "问题", "status": "open"},
    }
    with patch("back.agent.workflow.response_nodes.generate_grounded_response") as mocked:
        result = respond_with_grounded_knowledge(state)

    mocked.assert_not_called()
    assert result["evidence_status"] == "not_found"
    assert result["active_issue"]["status"] == "handed_off"
    assert "人工" in result["messages"][0].content


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("这是正常回答。 资料1", "这是正常回答。"),
        ("这是正常回答。 参考资料2。", "这是正常回答。"),
        ("这是正常回答（资料1）", "这是正常回答"),
        ("这是正常回答[参考资料1、2]", "这是正常回答"),
        ("资料1：这是正常回答。", "这是正常回答。"),
        ("根据参考资料1，这是正常回答。", "这是正常回答。"),
        ("套餐1购买后立即生效。", "套餐1购买后立即生效。"),
        ("资料下载失败时请重试。", "资料下载失败时请重试。"),
    ],
)
def test_internal_document_citation_is_removed_without_changing_business_text(
    raw,
    expected,
):
    assert sanitize_grounded_answer(raw) == expected


def test_normalization_applies_answer_sanitization():
    result = normalize_grounded_response(
        response(answer="续费不会重置流量。 资料1"),
        document_count=3,
    )
    assert result.answer == "续费不会重置流量。"
