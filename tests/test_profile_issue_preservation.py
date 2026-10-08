"""企业资料插话保留当前问题，且下一轮正常恢复生命周期处理。"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from back.agent.analysis.request import RequestAnalysis
from back.agent.workflow.context_nodes import analyze_request_node
from back.agent.workflow.graph import customer_service_graph
from back.agent.workflow.response_nodes import SUPPORT_REQUEST_PROMPT


@pytest.fixture(autouse=True)
def profile_response(monkeypatch):
    monkeypatch.setenv("ROUTER_V2_ENABLED", "1")
    response_model = Mock()
    response_model.invoke.return_value = AIMessage(content="企业资料测试回答")
    with patch("back.agent.workflow.response_nodes._response_llm", return_value=response_model):
        yield


def original_issue(status="awaiting_user"):
    return {
        "summary": "续费后流量没有重置",
        "status": status,
        "intent": "traffic",
        "last_clarifying_question": "页面显示什么提示？",
        "not_found_count": 1,
    }


def business_analysis(**updates):
    data = {
        "relation": "continue",
        "resolved_query": "续费后流量仍没有重置",
        "context_reason": "用户继续反馈原问题",
        "is_self_contained": False,
        "references_active_issue": True,
        "answers_last_question": False,
        "explicit_new_issue": False,
        "scope": "in_scope",
        "scope_reason": "业务问题",
        "intent": "traffic",
        "action": "clarify",
        "known_information": [],
        "missing_information": ["页面提示"],
        "clarifying_question": "页面显示什么提示？",
        "intent_reason": "继续处理原业务问题",
        "rewritten_queries": [],
        "rewrite_reason": None,
    }
    data.update(updates)
    return RequestAnalysis(**data)


def ask_for_profile(issue):
    return customer_service_graph.invoke({
        "tenant_id": "default",
        "messages": [HumanMessage(content="转人工客服")],
        "active_issue": issue,
    })


@pytest.mark.parametrize("query", ["转人工客服", "营业时间是什么？", "怎么联系你们"])
@pytest.mark.parametrize("status", ["open", "awaiting_user", "answered", "resolved", "handed_off"])
def test_profile_side_question_preserves_complete_existing_issue(query, status):
    issue = original_issue(status)
    expected = dict(issue)
    with patch("back.agent.workflow.retrieval_nodes.retrieve_documents_multi_query") as retrieve:
        result = customer_service_graph.invoke({
            "tenant_id": "default",
            "messages": [HumanMessage(content=query)],
            "active_issue": issue,
        })

    retrieve.assert_not_called()
    assert result["action"] == "profile"
    assert result["messages"][-1].content == (
        SUPPORT_REQUEST_PROMPT if query in {"转人工客服", "怎么联系你们"} else "企业资料测试回答"
    )
    assert result["support_required"] is (query in {"转人工客服", "怎么联系你们"})
    assert result["active_issue"] == expected
    assert issue == expected


def test_followup_after_profile_routes_with_original_issue():
    issue = original_issue()
    first = ask_for_profile(issue)
    with patch(
        "back.agent.workflow.context_nodes.route_request",
        return_value=business_analysis(),
    ) as route:
        result = analyze_request_node({
            **first,
            "messages": [*first["messages"], HumanMessage(content="还是不行")],
        })

    assert route.call_args.kwargs["active_issue"] == issue
    assert route.call_args.kwargs["current_query"] == "还是不行"
    assert result["active_issue"]["intent"] == "traffic"
    assert result["active_issue"]["not_found_count"] == 1
    assert result["profile_preserves_issue"] is False


def test_first_profile_question_keeps_original_create_and_answer_behavior():
    result = customer_service_graph.invoke({
        "tenant_id": "default",
        "messages": [HumanMessage(content="营业时间是什么？")],
        "active_issue": None,
        "profile_preserves_issue": True,
    })

    assert result["active_issue"] == {
        "summary": "营业时间是什么？",
        "status": "answered",
        "intent": "company_info",
        "last_clarifying_question": None,
    }
    assert result["profile_preserves_issue"] is False


def test_genuine_new_business_question_replaces_issue_after_profile():
    first = ask_for_profile(original_issue())
    with patch(
        "back.agent.workflow.context_nodes.route_request",
        return_value=business_analysis(
            relation="new_issue",
            resolved_query="另外一个问题，订阅导入失败",
            is_self_contained=True,
            references_active_issue=False,
            explicit_new_issue=True,
            intent="subscription_import",
            clarifying_question="导入时显示什么错误？",
        ),
    ):
        result = customer_service_graph.invoke({
            **first,
            "messages": [*first["messages"], HumanMessage(content="另外一个问题，订阅导入失败")],
        })

    assert result["active_issue"] == {
        "summary": "另外一个问题，订阅导入失败",
        "status": "awaiting_user",
        "intent": "subscription_import",
        "last_clarifying_question": "导入时显示什么错误？",
    }
    assert result["profile_preserves_issue"] is False
    assert result["explicit_new_issue"] is True


def test_resolution_confirmation_resets_per_turn_flags():
    result = analyze_request_node({
        "tenant_id": "default",
        "messages": [HumanMessage(content="问题已经解决了")],
        "active_issue": original_issue(),
        "profile_preserves_issue": True,
        "explicit_new_issue": True,
    })

    assert result["active_issue"]["status"] == "resolved"
    assert result["profile_preserves_issue"] is False
    assert result["explicit_new_issue"] is False
