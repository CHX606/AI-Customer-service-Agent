"""业务审查回归：具体诉求、多轮上下文、结束确认、创建和上传一致性。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
import pytest

from back.agent.analysis.request import route_request
from back.agent.analysis.request_models import RequestAnalysis
from back.agent.analysis.request_rules import build_search_queries, normalize_request_analysis
from back.agent.workflow.context_nodes import analyze_request_node
from back.agent.workflow.lifecycle import user_confirms_resolution
from back.bootstrap import get_chat_service, get_knowledge_service, get_profile_service
from back.domain.chat import ChatCommand
from back.domain.errors import Conflict
from back.domain.tenant import TenantProfileUpdate
from back.infrastructure.operation_locks import cache_mutation_lock
from back.interfaces.http.app import app
from back.knowledge.indexing import service as indexing
from back.knowledge.retrieval import cache
from back.tenant.defaults import DEFAULT_KELECLOUD_PROFILE
from back.tenant.service import create_tenant_profile


def analysis(**updates):
    values = {
        "relation": "continue",
        "resolved_query": "用户在 iPhone 的 Safari 中打开 Google，页面一直空白",
        "context_reason": "补充网站名称",
        "is_self_contained": False,
        "references_active_issue": True,
        "answers_last_question": True,
        "explicit_new_issue": False,
        "scope": "in_scope", "scope_reason": "网站访问故障",
        "intent": "other_business", "action": "retrieve",
        "known_information": ["iPhone", "Safari", "页面空白", "Google"],
        "intent_reason": "根据完整故障信息检索",
        "rewritten_queries": ["iPhone Safari Google 页面空白"],
    }
    values.update(updates)
    return RequestAnalysis(**values)


@pytest.mark.parametrize(("question", "intent"), [
    ("介绍一下退款规则", "refund"),
    ("介绍一下订阅导入步骤", "subscription_import"),
    ("介绍一下你们的退款规则", "refund"),
    ("修改账号联系方式的方法", "account"),
])
def test_specific_business_request_does_not_become_company_profile(question, intent):
    unused_model = Mock()
    unused_model.with_structured_output.side_effect = AssertionError("明确业务问题无需模型分类")
    result = route_request(
        router_llm=unused_model, fallback_llm=unused_model, current_query=question,
        recent_messages=[], active_issue=None, company_name="测试客服",
        business_scope=DEFAULT_KELECLOUD_PROFILE.business_scope,
    )
    assert (result.scope, result.intent, result.action) == ("in_scope", intent, "retrieve")
    assert build_search_queries(result) == [question]


def test_contextual_introduction_keeps_the_business_subject():
    request = analysis(
        resolved_query="用户希望介绍退款规则", intent="refund",
        known_information=["退款规则"], rewritten_queries=[],
    )
    result = normalize_request_analysis(
        request, current_query="介绍一下",
        active_issue={"summary": "退款规则", "status": "answered"},
    )
    assert result.action == "retrieve"
    assert result.resolved_query == request.resolved_query


@pytest.mark.parametrize(("question", "relation"), [
    ("Google", "continue"),
    ("不是 ChatGPT，是 Google", "correction"),
])
def test_platform_followup_preserves_complete_query_and_known_information(question, relation):
    request = analysis(relation=relation)
    result = normalize_request_analysis(
        request, current_query=question,
        active_issue={"summary": "iPhone 的 Safari 打开网页一直空白", "status": "awaiting_user"},
        business_scope=DEFAULT_KELECLOUD_PROFILE.business_scope,
    )
    assert result.relation == relation
    assert result.resolved_query == request.resolved_query
    assert result.known_information == request.known_information
    assert build_search_queries(result) == [request.resolved_query, *request.rewritten_queries]


def test_platform_name_does_not_discard_needed_clarification():
    request = analysis(
        resolved_query="用户使用 Google，但尚未说明故障现象", action="clarify",
        known_information=["Google"], missing_information=["故障现象"],
        clarifying_question="打开 Google 时具体显示什么？", rewritten_queries=[],
    )
    result = normalize_request_analysis(
        request, current_query="Google",
        active_issue={"summary": "网站有问题", "status": "awaiting_user"},
        business_scope=DEFAULT_KELECLOUD_PROFILE.business_scope,
    )
    assert result.action == "clarify"
    assert result.clarifying_question == request.clarifying_question
    assert result.missing_information == request.missing_information
    assert build_search_queries(result) == []


def test_explicit_new_platform_issue_does_not_keep_old_rewrites():
    result = normalize_request_analysis(
        analysis(relation="continue", explicit_new_issue=True, references_active_issue=False),
        current_query="换个问题，Telegram 无法登录",
        active_issue={"summary": "Google 页面空白", "status": "answered"},
        business_scope=DEFAULT_KELECLOUD_PROFILE.business_scope,
    )
    assert result.relation == "new_issue"
    assert build_search_queries(result) == ["换个问题，Telegram 无法登录"]
    assert result.known_information == []


@pytest.mark.parametrize("question", [
    "不是已经解决了，还是无法上网",
    "现在可以用了，但是退款还没到账",
    "你说已经解决了，实际上没有",
    "已经解决了吗？",
    "已经解决了？",
])
def test_negation_questions_and_remaining_requests_are_not_resolution(question):
    assert not user_confirms_resolution(question)


@pytest.mark.parametrize("question", [
    "已经解决了，谢谢", "问题已经解决了", "我这边已经恢复了，感谢", "现在可以用了！",
])
def test_clear_resolution_confirmation_remains_supported(question):
    assert user_confirms_resolution(question)


@pytest.mark.parametrize("question", [
    "不是已经解决了，还是无法上网", "现在可以用了，但是退款还没到账",
])
def test_unresolved_message_reaches_normal_analysis(monkeypatch, question):
    routed = Mock(return_value=analysis(resolved_query=question, intent="refund"))
    monkeypatch.setattr("back.agent.workflow.context_nodes.route_request", routed)
    result = analyze_request_node({
        "tenant_id": "default",
        "active_issue": {"summary": "无法上网", "status": "answered"},
        "messages": [HumanMessage(content=question)],
    })
    routed.assert_called_once()
    assert result["scope"] == "in_scope"
    assert result["active_issue"]["status"] != "resolved"
    assert result["search_queries"][0] == question


def test_duplicate_create_preserves_profile_cache_and_normal_update(monkeypatch):
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "1")
    monkeypatch.setattr(cache, "get_embedding_model", lambda: SimpleNamespace(embed_query=lambda _: [1.0, 0.0]))
    profiles = get_profile_service()
    old_hours, new_hours = "08:00-18:00", "10:00-20:00"
    profiles.update(TenantProfileUpdate(tenant_id="default", business_hours=old_hours))
    before = profiles.get("default")
    assert cache.store_semantic_answer("default", "营业时间是什么？", old_hours)
    revision = cache.get_cache_revision("default")
    proposed = before.model_copy(update={"business_hours": new_hours, "company_name": "重复创建"})
    client = TestClient(app)
    response = client.post(
        "/admin/tenants", headers={"X-Admin-Key": "master-admin-key"},
        json=proposed.model_dump(mode="json"),
    )
    assert response.status_code == 409
    assert profiles.get("default") == before
    assert cache.get_cache_revision("default") == revision
    assert cache.find_semantic_answer("default", "营业时间是什么？").answer == old_hours

    response = client.put(
        "/admin/profile", headers={"X-Admin-Key": "master-admin-key"},
        json={"tenant_id": "default", "business_hours": new_hours},
    )
    assert response.status_code == 200
    assert cache.find_semantic_answer("default", "营业时间是什么？") is None
    chat = get_chat_service()
    def fresh_answer(state):
        return {**state, "messages": [*state["messages"], AIMessage(content=profiles.get("default").business_hours)],
                "scope": "in_scope", "action": "profile"}
    monkeypatch.setattr(chat.engine, "invoke", fresh_answer)
    assert chat.execute(ChatCommand("营业时间是什么？", "after-profile-update")).answer == new_hours


def test_concurrent_creates_cannot_overwrite_the_winner():
    profiles = get_profile_service()
    template = profiles.get("default").model_copy(update={"tenant_id": "concurrent_create"})
    barrier = Barrier(2)
    def create(name):
        barrier.wait(timeout=5)
        try:
            return profiles.create(template.model_copy(update={"company_name": name})).company_name
        except Conflict:
            return "conflict"
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(create, name) for name in ("first", "second")]
        results = [job.result(timeout=10) for job in jobs]
    assert results.count("conflict") == 1
    winner = next(result for result in results if result != "conflict")
    assert profiles.get("concurrent_create").company_name == winner


def test_legacy_create_entry_rejects_duplicate_tenant():
    profiles = get_profile_service()
    before = profiles.get("default")
    with pytest.raises(Conflict):
        create_tenant_profile(before.model_copy(update={"company_name": "不应覆盖"}))
    assert profiles.get("default") == before


def test_busy_upload_has_no_side_effects_and_can_be_retried(monkeypatch):
    service = get_knowledge_service()
    before = service.list("default")
    write = Mock(wraps=service.files.write)
    monkeypatch.setattr(service.files, "write", write)
    # 保留真实文件、解析、状态和锁；仅替换外部搜索写入。
    add = Mock(return_value=1)
    monkeypatch.setattr(indexing, "add_source_documents", add)
    entered, release = Event(), Event()
    def busy_indexer():
        with cache_mutation_lock("default"):
            entered.set()
            assert release.wait(10)
    content = "测试知识：付款后请在订单页面检查状态。".encode("utf-8")
    client = TestClient(app)
    def upload():
        return client.post(
            "/admin/knowledge/files", headers={"X-Admin-Key": "master-admin-key"},
            data={"tenant_id": "default"}, files={"file": ("retry.txt", content, "text/plain")},
        )
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(busy_indexer)
        assert entered.wait(5)
        try:
            assert upload().status_code == 409
            assert service.list("default") == before
            write.assert_not_called()
            add.assert_not_called()
        finally:
            release.set()
        running.result(timeout=5)
    response = upload()
    assert response.status_code == 200, response.text
    source = service.get("default", response.json()["source_id"])
    assert source.status == "ready" and source.chunk_count > 0
    assert service.files.path(source).read_bytes() == content
    assert len(service.list("default")) == len(before) + 1
    assert upload().status_code == 409
    assert len(service.list("default")) == len(before) + 1
    write.assert_called_once()
    add.assert_called_once()


def test_upload_cache_failure_precedes_file_and_registry_writes(monkeypatch):
    service = get_knowledge_service()
    before = service.list("default")
    write = Mock(wraps=service.files.write)
    monkeypatch.setattr(service.files, "write", write)
    monkeypatch.setattr(cache, "invalidate_semantic_cache", Mock(side_effect=RuntimeError("private cache failure")))
    response = TestClient(app).post(
        "/admin/knowledge/files", headers={"X-Admin-Key": "master-admin-key"},
        data={"tenant_id": "default"}, files={"file": ("retry.txt", b"test content", "text/plain")},
    )
    assert response.status_code == 503
    assert "private cache failure" not in response.text
    assert service.list("default") == before
    write.assert_not_called()
