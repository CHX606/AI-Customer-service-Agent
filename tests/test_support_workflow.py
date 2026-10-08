"""人工事项入口与普通问答、会话、缓存及三种聊天协议的回归测试。"""

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
import pytest

from back.agent.analysis.request_rules import is_explicit_human_request, analyze_request_fast_path
from back.agent.response.grounded import GroundedResponse, normalize_grounded_response
from back.agent.workflow.graph import customer_service_graph
from back.agent.workflow.response_nodes import (
    NOT_FOUND_RETRY, SUPPORT_REQUEST_PROMPT, respond_with_grounded_knowledge,
    respond_clarify, respond_out_of_scope, respond_scope_uncertain,
)
from back.application.chat import ChatService
from back.application.images import ImageChatService
from back.domain.chat import CacheHit, ChatCommand, ChatResult
from back.infrastructure.persistence.sessions import SQLiteSessionRepository
from back.interfaces.http.app import app
import back.interfaces.http.chat as chat_http
from back.interfaces.http.schemas import ChatResponse


@pytest.mark.parametrize("query", [
    "人工客服", "人工服务", "转人工", "有人工客服吗？", "有人工客服不", "我要人工", "人工",
    "请帮我转人工客服", "麻烦找人工服务", "请为我转接人工",
    "怎么联系人工", "联系客服", "怎么联系你们", "如何联系客服", "人工客服的联系方式是什么？",
    "你们的联系方式是什么？", "客服电话是多少？",
])
def test_positive_human_requests_offer_form_without_llm_or_retrieval(query):
    issue = {"summary": "续费后流量未恢复", "intent": "traffic", "status": "awaiting_user",
             "not_found_count": 1, "last_clarifying_question": "页面提示是什么？"}
    with (
        patch("back.agent.workflow.response_nodes._response_llm") as llm,
        patch("back.agent.workflow.retrieval_nodes.retrieve_documents_multi_query") as retrieve,
    ):
        result = customer_service_graph.invoke({
            "tenant_id": "default", "messages": [HumanMessage(content=query)], "active_issue": issue,
        })
    llm.assert_not_called()
    retrieve.assert_not_called()
    assert result["active_issue"] == issue
    assert result["support_required"] is True
    assert result["support_reason"] == "explicit_request"
    assert result["messages"][-1].content == SUPPORT_REQUEST_PROMPT
    assert result["messages"][-1].additional_kwargs["support_required"] is True
    assert "待企业确认" not in result["messages"][-1].content


@pytest.mark.parametrize("query", [
    "不要转人工客服", "不用人工", "人工智能能用吗", "人工服务，怎么续订以后我之前的天数没了。",
    "能退款吗", "你们的退款条件是什么？",
])
def test_negation_and_business_policy_do_not_trigger_explicit_human_request(query):
    assert not is_explicit_human_request(query)
    result = analyze_request_fast_path(current_query=query, recent_messages=[], active_issue=None,
                                       business_scope=["账号", "退款", "续费"])
    assert result is None or result.action != "profile"


def grounded_result(status="sufficient", needs_human=False):
    return GroundedResponse(
        evidence_status=status, answer="可按官网说明办理。" if status in {"sufficient", "partial"} else None,
        needs_human=needs_human, supporting_document_indexes=[1, 2] if status == "conflict" else [1],
        decision_reason="资料说明办理方式", clarifying_question="页面显示什么状态？" if status == "insufficient" else None,
    )


@pytest.mark.parametrize("status,needs_human,required,reason,issue_status", [
    ("sufficient", False, False, None, "answered"),
    ("sufficient", True, True, "human_action", "answered"),
    ("partial", False, True, "partial", "answered"),
    ("insufficient", True, False, None, "awaiting_user"),
    ("conflict", False, True, "conflict", "handed_off"),
])
def test_grounded_response_preserves_answer_and_uses_structured_human_flag(
    status, needs_human, required, reason, issue_status,
):
    state = {
        "resolved_query": "能退款吗" if status == "sufficient" and not needs_human else "办理问题",
        "intent": "refund", "retrieved_documents": [Document(page_content="资料") for _ in range(2)],
        "active_issue": {"summary": "办理问题", "intent": "refund", "status": "open"},
    }
    with patch("back.agent.workflow.response_nodes.generate_grounded_response",
               return_value=grounded_result(status, needs_human)):
        result = respond_with_grounded_knowledge(state)
    assert result["support_required"] is required
    assert result["support_reason"] == reason
    assert result["active_issue"]["status"] == issue_status
    assert result["active_issue"]["intent"] == "refund"
    assert result["messages"][-1].additional_kwargs["support_required"] is required
    if status in {"sufficient", "partial"}:
        assert result["messages"][-1].content.startswith("可按官网说明办理。")
    if required:
        assert "提交人工处理" in result["messages"][-1].content
    if status == "conflict":
        assert "无法安全地给出确定结论" in result["messages"][-1].content


def test_not_found_retains_first_retry_and_offers_form_on_second_or_missing_issue():
    state = {"resolved_query": "未知业务", "intent": "other_business", "retrieved_documents": [],
             "active_issue": {"summary": "未知业务", "status": "open"}}
    first = respond_with_grounded_knowledge(state)
    assert first["messages"][-1].content == NOT_FOUND_RETRY
    assert first["support_required"] is False
    assert first["active_issue"]["status"] == "awaiting_user"
    second = respond_with_grounded_knowledge({**state, "active_issue": first["active_issue"]})
    assert second["support_required"] is True
    assert second["support_reason"] == "not_found"
    assert second["active_issue"]["not_found_count"] == 2
    assert "暂未查到" in second["messages"][-1].content
    missing_issue = respond_with_grounded_knowledge({**state, "active_issue": None})
    assert missing_issue["support_required"] is True


def test_grounded_normalization_clears_human_action_when_answer_is_rejected():
    result = normalize_grounded_response(grounded_result(needs_human=True), document_count=0)
    assert result.evidence_status == "not_found"
    assert result.needs_human is False


@pytest.mark.parametrize("respond,state", [
    (respond_clarify, {"clarifying_question": "页面提示是什么？"}),
    (respond_out_of_scope, {"messages": [HumanMessage(content="北京天气怎么样")]}),
    (respond_scope_uncertain, {}),
])
def test_non_human_response_resets_stale_per_turn_metadata(respond, state):
    result = respond({**state, "support_required": True, "support_reason": "explicit_request"})
    assert result["support_required"] is False
    assert result["support_reason"] is None


@pytest.fixture
def service(tmp_path):
    cache = Mock()
    cache.revision.return_value = None
    cache.find.return_value = None
    engine = Mock()
    def invoke(state):
        return {**state, "messages": [*state["messages"], AIMessage(content="普通有依据答复")],
                "scope": "in_scope", "action": "retrieve", "evidence_status": "sufficient"}
    engine.invoke.side_effect = invoke
    engine.stream.side_effect = lambda state: iter([{"type": "state", "state": engine.invoke(state)}])
    tenants = Mock()
    tenants.get.return_value = SimpleNamespace(tenant_id="default")
    return ChatService(engine, SQLiteSessionRepository(tmp_path / "support-sessions.db"), cache, tenants)


def test_explicit_human_request_bypasses_preexisting_profile_cache(service):
    service.cache.revision.return_value = 3
    service.cache.find.return_value = CacheHit("人工客服联系方式：待企业确认", 1.0, True)
    service.engine.invoke.side_effect = lambda state: customer_service_graph.invoke(state)
    result = service.execute(ChatCommand("转人工客服", "human-cache-bypass"))
    service.cache.find.assert_not_called()
    service.cache.store.assert_not_called()
    assert result.support_required is True
    assert result.answer == SUPPORT_REQUEST_PROMPT


def test_support_metadata_persists_in_ai_message_and_normal_answer_still_caches(service):
    service.cache.revision.return_value = 3
    service.engine.invoke.side_effect = lambda state: {
        **state, "messages": [*state["messages"], AIMessage(content="等待人工确认")],
        "scope": "in_scope", "action": "profile", "support_required": True, "support_reason": "human_action",
    }
    result = service.execute(ChatCommand("请人工查看办理进度", "pending-support"))
    service.cache.store.assert_not_called()
    restored = service.sessions.load(("default", "pending-support")).state
    assert restored["messages"][-1].additional_kwargs == {
        "support_required": True, "support_reason": "human_action",
    }
    assert result.support_reason == "human_action"
    service.engine.invoke.side_effect = lambda state: {
        **state, "messages": [*state["messages"], AIMessage(content="退款规则")],
        "scope": "in_scope", "action": "retrieve", "evidence_status": "sufficient",
    }
    normal = service.execute(ChatCommand("能退款吗", "normal-policy"))
    assert normal.support_required is False
    assert normal.support_reason is None
    service.cache.store.assert_called_once_with("default", "能退款吗", "退款规则", expected_revision=3)


@pytest.mark.parametrize("path", ["/chat", "/chat/stream", "/chat/image"])
@pytest.mark.parametrize("required", [False, True])
def test_text_image_and_stream_return_consistent_support_fields(monkeypatch, service, path, required):
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", "default")
    monkeypatch.setenv("IMAGE_FEATURES_ENABLED", "1")
    def invoke(state):
        return {**state, "messages": [*state["messages"], AIMessage(content="测试回答")],
                "scope": "in_scope", "action": "retrieve", "evidence_status": "sufficient",
                "support_required": required, "support_reason": "human_action" if required else None}
    service.engine.invoke.side_effect = invoke
    monkeypatch.setattr(chat_http, "get_chat_service", lambda: service)
    interpreter = Mock()
    interpreter.describe.return_value = "截图问题描述"
    monkeypatch.setattr(chat_http, "get_image_chat_service", lambda: ImageChatService(service, interpreter))
    client = TestClient(app)
    if path == "/chat/image":
        response = client.post(path, data={"message": "截图问题", "session_id": "metadata-image", "tenant_id": "default"},
                               files={"image": ("screenshot.png", b"fake-image", "image/png")})
    else:
        response = client.post(path, json={"message": "咨询", "session_id": "metadata-text", "tenant_id": "default"})
    assert response.status_code == 200
    if path == "/chat/stream":
        import json
        events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
        payload = events[-1]
        assert payload["type"] == "final"
    else:
        payload = response.json()
    assert payload["support_required"] is required
    assert payload["support_reason"] == ("human_action" if required else None)
    restored = service.sessions.load(("default", payload["session_id"])).state
    assert restored["messages"][-1].additional_kwargs["support_required"] is required


def test_response_defaults_keep_existing_constructor_compatible():
    result = ChatResult("普通答复", "session", "default")
    response = ChatResponse(**asdict(result))
    assert response.support_required is False
    assert response.support_reason is None


def test_website_profile_prompt_preserves_url_and_replaces_old_human_placeholder():
    from back.agent.workflow.response_nodes import answer_from_profile
    from back.tenant.defaults import DEFAULT_KELECLOUD_PROFILE
    profile = DEFAULT_KELECLOUD_PROFILE.model_copy(update={
        "public_contact": "官网地址：https://example.com\n人工客服联系方式：待企业确认",
    })
    fake = Mock()
    fake.invoke.return_value = AIMessage(content="官网地址：https://example.com")
    with (patch("back.agent.workflow.response_nodes.get_tenant_profile", return_value=profile),
          patch("back.agent.workflow.response_nodes._response_llm", return_value=fake)):
        result = answer_from_profile({"resolved_query": "官网地址是什么？",
                                      "messages": [HumanMessage(content="官网地址是什么？")]})
    prompt = fake.invoke.call_args.args[0][0].content
    assert "https://example.com" in prompt
    assert "待企业确认" not in prompt
    assert result["support_required"] is False
