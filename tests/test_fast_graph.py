"""合并分析后的图路由与模型调用次数测试。"""

from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from back.agent.response.grounded import GroundedResponse
from back.agent.workflow.context_nodes import analyze_request_node
from back.agent.workflow.graph import customer_service_graph
from back.agent.workflow.routes import route_after_request_analysis
from back.agent.analysis.request import RequestAnalysis


def request_result(**updates) -> RequestAnalysis:
    data = {
        "relation": "new_issue",
        "resolved_query": "续费后流量没有重置",
        "context_reason": "完整的新问题",
        "is_self_contained": True,
        "references_active_issue": False,
        "answers_last_question": False,
        "explicit_new_issue": False,
        "scope": "in_scope",
        "scope_reason": "业务问题",
        "intent": "traffic",
        "action": "retrieve",
        "known_information": ["已经续费", "流量没有重置"],
        "missing_information": [],
        "clarifying_question": None,
        "intent_reason": "信息足够",
        "rewritten_queries": ["续费成功后流量未重置"],
        "rewrite_reason": "标准化表达",
    }
    data.update(updates)
    return RequestAnalysis(**data)


class StructuredRunner:
    def __init__(self, parent, schema):
        self.parent = parent
        self.schema = schema

    def invoke(self, _messages):
        self.parent.calls.append(("structured", self.schema.__name__))
        return self.parent.structured_results[self.schema]


class CountingModel:
    def __init__(self, structured_results):
        self.structured_results = structured_results
        self.calls = []

    def with_structured_output(self, schema):
        return StructuredRunner(self, schema)

    def invoke(self, _messages):
        self.calls.append(("chat", "answer"))
        return AIMessage(content="测试回答")


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"scope": "chitchat"}, "chitchat"),
        ({"scope": "out_of_scope"}, "out_of_scope"),
        ({"scope": "uncertain"}, "scope_uncertain"),
        ({"scope": "in_scope", "action": "profile"}, "answer_profile"),
        ({"scope": "in_scope", "action": "retrieve"}, "hybrid_search"),
        ({"scope": "in_scope", "action": "clarify"}, "clarify"),
        ({"scope": "in_scope", "action": None}, "clarify"),
    ],
)
def test_combined_route_covers_every_branch(state, expected):
    assert route_after_request_analysis(state) == expected


def test_retrieval_answer_path_skips_analysis_model_for_clear_single_turn():
    docs = [
        Document(
            page_content="续费只延长有效期，流量按套餐周期重置。",
            metadata={"section_id": "1.1", "section_title": "续费规则"},
        )
    ]
    grounded = GroundedResponse(
        evidence_status="sufficient",
        answer="测试回答",
        supporting_document_indexes=[1],
        missing_information=[],
        clarifying_question=None,
        decision_reason="资料直接支持回答",
    )
    fake_model = CountingModel(
        {
            RequestAnalysis: request_result(),
            GroundedResponse: grounded,
        }
    )

    with (
        patch("back.core.llm.model", fake_model),
        patch(
            "back.core.llm.get_response_model",
            return_value=fake_model,
        ),
        patch("back.agent.workflow.retrieval_nodes.retrieve_documents_multi_query", return_value=docs),
        patch("back.agent.workflow.retrieval_nodes.rerank_documents", return_value=docs),
    ):
        result = customer_service_graph.invoke(
            {
                "tenant_id": "default",
                "messages": [HumanMessage(content="续费后流量没有重置怎么办？")],
            }
        )

    assert fake_model.calls == [("structured", "GroundedResponse")]
    assert result["messages"][-1].content == "测试回答"
    assert result["search_queries"][0] == "续费后流量没有重置怎么办？"


@pytest.mark.parametrize(
    ("query", "expected_calls", "expected_answer"),
    [
        (
            "营业时间是什么？",
            [("chat", "answer")],
            "测试回答",
        ),
        (
            "节点不能用",
            [],
            "请问是所有节点都不能用，还是某个节点不能用？当前显示什么错误或状态？",
        ),
        (
            "北京明天天气怎么样？",
            [],
            None,
        ),
    ],
)
def test_non_retrieval_paths_do_not_make_unnecessary_calls(
    query,
    expected_calls,
    expected_answer,
):
    fake_model = CountingModel({})
    with (
        patch("back.core.llm.model", fake_model),
        patch(
            "back.core.llm.get_response_model",
            return_value=fake_model,
        ),
    ):
        result = customer_service_graph.invoke(
            {
                "tenant_id": "default",
                "messages": [HumanMessage(content=query)],
            }
        )

    assert fake_model.calls == expected_calls
    if expected_answer is not None:
        assert result["messages"][-1].content == expected_answer


def test_resolution_confirmation_skips_analysis_model_call():
    class FailIfUsed:
        def with_structured_output(self, _schema):
            raise AssertionError("确认解决时不应调用分析模型")

    with patch("back.core.llm.model", FailIfUsed()):
        result = analyze_request_node(
            {
                "tenant_id": "default",
                "messages": [HumanMessage(content="问题已经解决了")],
                "active_issue": {
                    "summary": "节点无法连接",
                    "status": "answered",
                    "intent": "node_connection",
                },
            }
        )

    assert result["scope"] == "chitchat"
    assert result["active_issue"]["status"] == "resolved"
    assert result["search_queries"] == []


@pytest.mark.parametrize("relation", ["continue", "correction", "new_issue"])
def test_not_found_followup_survives_session_reload_and_hands_off_on_second_miss(tmp_path, relation):
    from back.infrastructure.persistence.sessions import SQLiteSessionRepository

    path = tmp_path / "not-found-sessions.db"
    key = ("default", "not-found-conversation")
    with (
        patch("back.agent.workflow.context_nodes.route_request", return_value=request_result()),
        patch("back.agent.workflow.retrieval_nodes.retrieve_documents_multi_query", return_value=[]),
        patch("back.agent.workflow.retrieval_nodes.rerank_documents", return_value=[]),
    ):
        first = customer_service_graph.invoke({
            "tenant_id": "default",
            "messages": [HumanMessage(content="续费后流量没有重置怎么办？")],
        })
    assert first["active_issue"]["status"] == "awaiting_user"
    assert first["active_issue"]["not_found_count"] == 1
    SQLiteSessionRepository(path).save(key, first, 0)
    restored = SQLiteSessionRepository(path).load(key).state
    assert restored["active_issue"]["last_clarifying_question"] == first["messages"][-1].content

    with (
        patch(
            "back.agent.workflow.context_nodes.route_request",
            return_value=request_result(relation=relation, answers_last_question=relation != "new_issue"),
        ),
        patch("back.agent.workflow.retrieval_nodes.retrieve_documents_multi_query", return_value=[]),
        patch("back.agent.workflow.retrieval_nodes.rerank_documents", return_value=[]),
    ):
        second = customer_service_graph.invoke({
            **restored,
            "tenant_id": "default",
            "messages": [*restored["messages"], HumanMessage(content="页面仍然显示流量为零")],
        })
    assert second["active_issue"]["status"] == "handed_off"
    assert second["active_issue"]["not_found_count"] == 2
    assert "人工" in second["messages"][-1].content


def test_explicit_new_question_after_retry_starts_its_own_miss_count():
    from back.agent.workflow.response_nodes import NOT_FOUND_RETRY

    with (
        patch(
            "back.agent.workflow.context_nodes.route_request",
            return_value=request_result(
                relation="new_issue", explicit_new_issue=True,
                resolved_query="另外一个问题：退款怎么处理？", intent="refund",
            ),
        ),
        patch("back.agent.workflow.retrieval_nodes.retrieve_documents_multi_query", return_value=[]),
        patch("back.agent.workflow.retrieval_nodes.rerank_documents", return_value=[]),
    ):
        result = customer_service_graph.invoke({
            "tenant_id": "default",
            "active_issue": {
                "summary": "续费后流量未重置", "status": "awaiting_user",
                "intent": "traffic", "not_found_count": 1,
            },
            "messages": [
                HumanMessage(content="续费后流量未重置"),
                AIMessage(content=NOT_FOUND_RETRY),
                HumanMessage(content="另外一个问题：退款怎么处理？"),
            ],
        })
    assert result["active_issue"]["intent"] == "refund"
    assert result["active_issue"]["status"] == "awaiting_user"
    assert result["active_issue"]["not_found_count"] == 1


def test_explicit_human_request_bypasses_not_found_followup():
    fake_model = CountingModel({})
    with (
        patch("back.core.llm.model", fake_model),
        patch("back.core.llm.get_response_model", return_value=fake_model),
        patch("back.agent.workflow.retrieval_nodes.retrieve_documents_multi_query") as retrieve,
    ):
        result = customer_service_graph.invoke({
            "tenant_id": "default",
            "messages": [HumanMessage(content="转人工客服")],
            "active_issue": {
                "summary": "资料没找到的问题",
                "status": "awaiting_user",
                "not_found_count": 1,
                "last_clarifying_question": "请补充页面提示",
            },
        })
    retrieve.assert_not_called()
    assert result["action"] == "profile"
    assert "提交人工处理" in result["messages"][-1].content
    assert result["support_required"] is True
    assert result["active_issue"]["summary"] == "资料没找到的问题"
    assert fake_model.calls == []
