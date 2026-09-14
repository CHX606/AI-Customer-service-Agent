from back.bootstrap import get_chat_service, get_knowledge_service
"""语义问答缓存的匹配、安全、隔离、失效与聊天链路测试。"""

import json
from datetime import datetime
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
import pytest

import back.interfaces.http.app
import back.interfaces.http.chat
import back.knowledge.indexing.service
import back.knowledge.retrieval.cache as semantic_cache
from back.interfaces.http.app import app
from back.interfaces.http.chat import InternalChatRequest, PreparedChat
from back.knowledge.retrieval.cache import SemanticCacheHit
from back.interfaces.http.schemas import ChatRequest
from back.domain.tenant import KnowledgeSource
from back.tenant.service import get_tenant_profile, save_tenant_profile
from back.infrastructure.persistence.tenants import get_db_connection


client = TestClient(app)


class FakeEmbeddings:
    def __init__(self, vectors: dict[str, list[float]] | None = None):
        self.vectors = vectors or {}
        self.calls: list[str] = []

    def embed_query(self, question: str) -> list[float]:
        self.calls.append(question)
        return self.vectors.get(question, [1.0, 0.0, 0.0])


def enable_cache(monkeypatch: pytest.MonkeyPatch, threshold: str = "0.88") -> None:
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "1")
    monkeypatch.setenv("SEMANTIC_CACHE_THRESHOLD", threshold)


def add_tenant(tenant_id: str) -> None:
    profile = get_tenant_profile("default")
    assert profile is not None
    save_tenant_profile(
        profile.model_copy(
            update={
                "tenant_id": tenant_id,
                "company_name": f"{tenant_id}公司",
                "assistant_name": f"{tenant_id}客服",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }
        )
    )


def make_prepared(
    question: str = "退款政策是什么？",
    *,
    session_id: str = "cache-session",
    history: list | None = None,
) -> PreparedChat:
    messages = [*(history or []), HumanMessage(content=question)]
    return PreparedChat(
        tenant_id="default",
        session_id=session_id,
        session_key=("default", session_id),
        graph_input={
            "tenant_id": "default",
            "messages": messages,
            "active_issue": None,
        },
    )


def make_result(
    prepared: PreparedChat,
    *,
    answer: str = "支持在规定时间内申请退款。",
    action: str = "retrieve",
    evidence_status: str | None = "sufficient",
    scope: str = "in_scope",
):
    return {
        **prepared.graph_input,
        "messages": [
            *prepared.graph_input["messages"],
            AIMessage(content=answer),
        ],
        "scope": scope,
        "action": action,
        "evidence_status": evidence_status,
    }


def parse_events(lines):
    return [json.loads(line) for line in lines]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" 退款政策是什么？ ", "退款政策是什么"),
        ("ＡＢＣ 套餐！", "abc套餐"),
        ("如何\n续费？", "如何续费"),
    ],
)
def test_question_normalization(raw: str, expected: str):
    assert semantic_cache.normalize_question(raw) == expected


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_cache_enabled_truthy_values(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", value)
    assert semantic_cache.is_semantic_cache_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "off", "", "random"])
def test_cache_disabled_values(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", value)
    assert semantic_cache.is_semantic_cache_enabled() is False


def test_threshold_and_capacity_configuration_are_safely_bounded(monkeypatch):
    monkeypatch.setenv("SEMANTIC_CACHE_THRESHOLD", "0.1")
    assert semantic_cache.get_similarity_threshold() == 0.80
    monkeypatch.setenv("SEMANTIC_CACHE_THRESHOLD", "2")
    assert semantic_cache.get_similarity_threshold() == 0.999
    monkeypatch.setenv("SEMANTIC_CACHE_THRESHOLD", "broken")
    assert semantic_cache.get_similarity_threshold() == 0.88

    monkeypatch.setenv("SEMANTIC_CACHE_MAX_ENTRIES", "0")
    assert semantic_cache.get_max_entries_per_tenant() == 1
    monkeypatch.setenv("SEMANTIC_CACHE_MAX_ENTRIES", "99999")
    assert semantic_cache.get_max_entries_per_tenant() == 5000
    monkeypatch.setenv("SEMANTIC_CACHE_MAX_ENTRIES", "broken")
    assert semantic_cache.get_max_entries_per_tenant() == 200


@pytest.mark.parametrize(
    "question",
    [
        "订单号12345678怎么退款",
        "我的订单怎么还没到账",
        "请查一下手机号13800138000",
        "我的邮箱user@example.com无法登录",
        "请打开https://example.com处理",
        "验证码123456失效了",
        "你好",
    ],
)
def test_sensitive_or_too_short_questions_are_not_cacheable(question: str):
    assert semantic_cache.is_safe_cache_question(question) is False


@pytest.mark.parametrize(
    "question",
    ["退款政策是什么？", "账号应该如何注册？", "套餐有哪几种？", "三天内可以退款吗？"],
)
def test_generic_questions_are_cacheable(question: str):
    assert semantic_cache.is_safe_cache_question(question) is True


def test_disabled_cache_neither_stores_nor_reads(monkeypatch):
    fake = FakeEmbeddings()
    monkeypatch.setattr(semantic_cache, "get_embedding_model", lambda: fake)
    assert semantic_cache.store_semantic_answer("default", "退款政策是什么", "答案") is False
    assert semantic_cache.find_semantic_answer("default", "退款政策是什么") is None
    assert semantic_cache.count_semantic_cache_entries("default") == 0
    assert fake.calls == []


def test_cache_schema_is_created_lazily_for_existing_database(monkeypatch):
    enable_cache(monkeypatch)
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("DROP TABLE semantic_answer_cache")
    finally:
        conn.close()

    assert semantic_cache.invalidate_semantic_cache("default") == 0
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='semantic_answer_cache'"
        ).fetchone()
        assert row is not None
    finally:
        conn.close()


def test_exact_hit_avoids_second_embedding_call_and_updates_stats(monkeypatch):
    enable_cache(monkeypatch)
    fake = FakeEmbeddings()
    monkeypatch.setattr(semantic_cache, "get_embedding_model", lambda: fake)
    assert semantic_cache.store_semantic_answer("default", "退款政策是什么？", "退款答案")
    assert fake.calls == ["退款政策是什么？"]

    def fail_if_embedded():
        raise AssertionError("精确命中不应再次加载向量模型")

    monkeypatch.setattr(semantic_cache, "get_embedding_model", fail_if_embedded)
    hit = semantic_cache.find_semantic_answer("default", " 退款政策是什么 ")
    assert hit == SemanticCacheHit(answer="退款答案", similarity=1.0, exact=True)

    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT hit_count, last_hit_at FROM semantic_answer_cache"
        ).fetchone()
        assert row["hit_count"] == 1
        assert row["last_hit_at"] is not None
    finally:
        conn.close()


def test_semantically_similar_question_hits_above_threshold(monkeypatch):
    enable_cache(monkeypatch, threshold="0.95")
    fake = FakeEmbeddings(
        {
            "退款政策是什么？": [1.0, 0.0],
            "请问怎样申请退款": [0.999, 0.02],
        }
    )
    monkeypatch.setattr(semantic_cache, "get_embedding_model", lambda: fake)
    semantic_cache.store_semantic_answer("default", "退款政策是什么？", "退款答案")

    hit = semantic_cache.find_semantic_answer("default", "请问怎样申请退款")
    assert hit is not None
    assert hit.answer == "退款答案"
    assert hit.exact is False
    assert hit.similarity > 0.99


def test_semantically_different_question_misses_below_threshold(monkeypatch):
    enable_cache(monkeypatch, threshold="0.95")
    fake = FakeEmbeddings(
        {
            "退款政策是什么？": [1.0, 0.0],
            "如何安装客户端？": [0.0, 1.0],
        }
    )
    monkeypatch.setattr(semantic_cache, "get_embedding_model", lambda: fake)
    semantic_cache.store_semantic_answer("default", "退款政策是什么？", "退款答案")
    assert semantic_cache.find_semantic_answer("default", "如何安装客户端？") is None


def test_cache_is_strictly_isolated_between_tenants(monkeypatch):
    enable_cache(monkeypatch)
    add_tenant("tenant_b")
    fake = FakeEmbeddings()
    monkeypatch.setattr(semantic_cache, "get_embedding_model", lambda: fake)
    semantic_cache.store_semantic_answer("default", "退款政策是什么？", "A租户答案")

    assert semantic_cache.find_semantic_answer("tenant_b", "退款政策是什么？") is None
    semantic_cache.store_semantic_answer("tenant_b", "退款政策是什么？", "B租户答案")
    assert semantic_cache.find_semantic_answer("default", "退款政策是什么？").answer == "A租户答案"
    assert semantic_cache.find_semantic_answer("tenant_b", "退款政策是什么？").answer == "B租户答案"


def test_same_normalized_question_is_updated_not_duplicated(monkeypatch):
    enable_cache(monkeypatch)
    fake = FakeEmbeddings()
    monkeypatch.setattr(semantic_cache, "get_embedding_model", lambda: fake)
    semantic_cache.store_semantic_answer("default", "退款政策是什么？", "旧答案")
    semantic_cache.store_semantic_answer("default", " 退款政策是什么 ", "新答案")

    assert semantic_cache.count_semantic_cache_entries("default") == 1
    assert semantic_cache.find_semantic_answer("default", "退款政策是什么").answer == "新答案"


def test_capacity_evicts_oldest_unused_entries(monkeypatch):
    enable_cache(monkeypatch)
    monkeypatch.setenv("SEMANTIC_CACHE_MAX_ENTRIES", "2")
    fake = FakeEmbeddings(
        {
            "第一个退款问题": [1.0, 0.0, 0.0],
            "第二个续费问题": [0.0, 1.0, 0.0],
            "第三个安装问题": [0.0, 0.0, 1.0],
        }
    )
    monkeypatch.setattr(semantic_cache, "get_embedding_model", lambda: fake)
    semantic_cache.store_semantic_answer("default", "第一个退款问题", "答案1")
    semantic_cache.store_semantic_answer("default", "第二个续费问题", "答案2")
    semantic_cache.store_semantic_answer("default", "第三个安装问题", "答案3")

    assert semantic_cache.count_semantic_cache_entries("default") == 2
    assert semantic_cache.find_semantic_answer("default", "第一个退款问题") is None
    assert semantic_cache.find_semantic_answer("default", "第三个安装问题").answer == "答案3"


def test_corrupt_or_wrong_dimension_embeddings_are_ignored(monkeypatch):
    enable_cache(monkeypatch)
    fake = FakeEmbeddings({"退款政策是什么？": [1.0, 0.0], "换个问法测试": [1.0, 0.0]})
    monkeypatch.setattr(semantic_cache, "get_embedding_model", lambda: fake)
    semantic_cache.store_semantic_answer("default", "退款政策是什么？", "答案")

    conn = get_db_connection()
    try:
        with conn:
            conn.execute(
                "UPDATE semantic_answer_cache SET embedding_json = ?",
                ("not-json",),
            )
    finally:
        conn.close()
    assert semantic_cache.find_semantic_answer("default", "换个问法测试") is None

    conn = get_db_connection()
    try:
        with conn:
            conn.execute(
                "UPDATE semantic_answer_cache SET embedding_json = ?",
                (json.dumps([1.0, 0.0, 0.0]),),
            )
    finally:
        conn.close()
    assert semantic_cache.find_semantic_answer("default", "换个问法测试") is None


def test_invalidation_only_clears_target_tenant(monkeypatch):
    enable_cache(monkeypatch)
    add_tenant("tenant_b")
    fake = FakeEmbeddings()
    monkeypatch.setattr(semantic_cache, "get_embedding_model", lambda: fake)
    semantic_cache.store_semantic_answer("default", "退款政策是什么？", "A答案")
    semantic_cache.store_semantic_answer("tenant_b", "退款政策是什么？", "B答案")

    assert semantic_cache.invalidate_semantic_cache("default") == 1
    assert semantic_cache.count_semantic_cache_entries("default") == 0
    assert semantic_cache.count_semantic_cache_entries("tenant_b") == 1


def test_chat_cache_hit_skips_graph_and_preserves_followup_context(monkeypatch):
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", "default")
    hit = SemanticCacheHit(answer="缓存中的退款答案", similarity=0.98, exact=False)
    monkeypatch.setattr(semantic_cache, "find_semantic_answer", Mock(return_value=hit))
    graph = Mock()
    monkeypatch.setattr(get_chat_service().engine.graph, "invoke", graph)

    response = back.interfaces.http.chat._execute_chat(
        ChatRequest(
            tenant_id="default",
            session_id="cache-api-hit",
            message="请问怎样退款？",
        )
    )

    assert response.answer == "缓存中的退款答案"
    graph.assert_not_called()
    saved = get_chat_service().sessions.load(('default', 'cache-api-hit')).state
    assert [message.content for message in saved["messages"]] == [
        "请问怎样退款？",
        "缓存中的退款答案",
    ]


def test_first_request_populates_cache_and_second_exact_request_skips_graph(monkeypatch):
    enable_cache(monkeypatch)
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", "default")
    fake = FakeEmbeddings()
    monkeypatch.setattr(semantic_cache, "get_embedding_model", lambda: fake)

    graph = Mock(
        side_effect=lambda state: {
            **state,
            "messages": [*state["messages"], AIMessage(content="首次生成的可靠答案")],
            "scope": "in_scope",
            "action": "retrieve",
            "evidence_status": "sufficient",
        }
    )
    monkeypatch.setattr(get_chat_service().engine.graph, "invoke", graph)

    first = back.interfaces.http.chat._execute_chat(
        ChatRequest(
            tenant_id="default",
            session_id="exact-populate-1",
            message="退款政策是什么？",
        )
    )
    second = back.interfaces.http.chat._execute_chat(
        ChatRequest(
            tenant_id="default",
            session_id="exact-populate-2",
            message="退款政策是什么",
        )
    )

    assert first.answer == "首次生成的可靠答案"
    assert second.answer == "首次生成的可靠答案"
    assert graph.call_count == 1
    assert semantic_cache.count_semantic_cache_entries("default") == 1


def test_first_request_populates_cache_and_semantic_paraphrase_skips_graph(monkeypatch):
    enable_cache(monkeypatch, threshold="0.88")
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", "default")
    fake = FakeEmbeddings(
        {
            "套餐要怎么续费？": [1.0, 0.0],
            "如何给现有套餐续费？": [0.95, 0.1],
        }
    )
    monkeypatch.setattr(semantic_cache, "get_embedding_model", lambda: fake)
    graph = Mock(
        side_effect=lambda state: {
            **state,
            "messages": [*state["messages"], AIMessage(content="套餐续费答案")],
            "scope": "in_scope",
            "action": "retrieve",
            "evidence_status": "sufficient",
        }
    )
    monkeypatch.setattr(get_chat_service().engine.graph, "invoke", graph)

    first = back.interfaces.http.chat._execute_chat(
        ChatRequest(
            tenant_id="default",
            session_id="semantic-populate-1",
            message="套餐要怎么续费？",
        )
    )
    second = back.interfaces.http.chat._execute_chat(
        ChatRequest(
            tenant_id="default",
            session_id="semantic-populate-2",
            message="如何给现有套餐续费？",
        )
    )

    assert first.answer == "套餐续费答案"
    assert second.answer == "套餐续费答案"
    assert graph.call_count == 1


def test_low_similarity_second_request_runs_graph_again(monkeypatch):
    enable_cache(monkeypatch, threshold="0.88")
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", "default")
    fake = FakeEmbeddings(
        {
            "退款政策是什么？": [1.0, 0.0],
            "客户端在哪里下载？": [0.0, 1.0],
        }
    )
    monkeypatch.setattr(semantic_cache, "get_embedding_model", lambda: fake)
    answers = iter(["退款答案", "客户端下载答案"])
    graph = Mock(
        side_effect=lambda state: {
            **state,
            "messages": [*state["messages"], AIMessage(content=next(answers))],
            "scope": "in_scope",
            "action": "retrieve",
            "evidence_status": "sufficient",
        }
    )
    monkeypatch.setattr(get_chat_service().engine.graph, "invoke", graph)

    first = back.interfaces.http.chat._execute_chat(
        ChatRequest(
            tenant_id="default",
            session_id="semantic-miss-1",
            message="退款政策是什么？",
        )
    )
    second = back.interfaces.http.chat._execute_chat(
        ChatRequest(
            tenant_id="default",
            session_id="semantic-miss-2",
            message="客户端在哪里下载？",
        )
    )

    assert first.answer == "退款答案"
    assert second.answer == "客户端下载答案"
    assert graph.call_count == 2


def test_stream_cache_hit_returns_immediate_final_without_graph(monkeypatch):
    target = make_prepared(session_id="stream-cache-hit")
    monkeypatch.setattr(
        semantic_cache,
        "find_semantic_answer",
        Mock(return_value=SemanticCacheHit("缓存流式答案", 1.0, True)),
    )
    graph_stream = Mock()
    monkeypatch.setattr(get_chat_service().engine.graph, "stream", graph_stream)

    events = parse_events(back.interfaces.http.chat._stream_prepared_chat(target))
    assert [event["type"] for event in events] == ["status", "final"]
    assert events[-1]["answer"] == "缓存流式答案"
    graph_stream.assert_not_called()


def test_followup_question_bypasses_cache(monkeypatch):
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", "default")
    session_id = "followup-cache-bypass"
    get_chat_service().sessions.save(('default', session_id), {
        "messages": [HumanMessage(content="怎么退款？"), AIMessage(content="请提交申请。")],
        "active_issue": {"summary": "退款", "status": "answered"},
    }, expected_version=0)
    finder = Mock(return_value=SemanticCacheHit("不应使用", 1.0, True))
    monkeypatch.setattr(semantic_cache, "find_semantic_answer", finder)
    graph = Mock(
        side_effect=lambda state: {
            **state,
            "messages": [*state["messages"], AIMessage(content="正常上下文回答")],
            "scope": "in_scope",
            "action": "retrieve",
            "evidence_status": "not_found",
        }
    )
    monkeypatch.setattr(get_chat_service().engine.graph, "invoke", graph)

    result = back.interfaces.http.chat._execute_chat(
        ChatRequest(
            tenant_id="default",
            session_id=session_id,
            message="那多久到账？",
        )
    )
    assert result.answer == "正常上下文回答"
    finder.assert_not_called()
    assert len(graph.call_args.args[0]["messages"]) == 3


@pytest.mark.parametrize(
    ("action", "evidence_status", "scope", "expected_calls"),
    [
        ("retrieve", "sufficient", "in_scope", 1),
        ("retrieve", "not_found", "in_scope", 0),
        ("retrieve", "insufficient", "in_scope", 0),
        ("retrieve", "conflict", "in_scope", 0),
        ("profile", None, "in_scope", 1),
        ("profile", None, "out_of_scope", 0),
        ("clarify", None, "in_scope", 0),
    ],
)
def test_only_reliable_first_turn_answers_are_stored(
    monkeypatch,
    action,
    evidence_status,
    scope,
    expected_calls,
):
    target = make_prepared(session_id=f"store-policy-{action}-{evidence_status}-{scope}")
    store = Mock(return_value=True)
    monkeypatch.setattr(semantic_cache, "store_semantic_answer", store)
    get_chat_service().finalize(
        target,
        make_result(
            target,
            action=action,
            evidence_status=evidence_status,
            scope=scope,
        ),
    )
    assert store.call_count == expected_calls


def test_followup_answer_is_never_stored(monkeypatch):
    target = make_prepared(
        "那多久到账？",
        history=[HumanMessage(content="怎么退款？"), AIMessage(content="请提交申请。")],
    )
    store = Mock()
    monkeypatch.setattr(semantic_cache, "store_semantic_answer", store)
    get_chat_service().finalize(target, make_result(target))
    store.assert_not_called()


def test_cache_lookup_failure_falls_back_to_graph(monkeypatch):
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", "default")
    monkeypatch.setattr(
        semantic_cache,
        "find_semantic_answer",
        Mock(side_effect=RuntimeError("cache unavailable")),
    )
    target_result = {
        "tenant_id": "default",
        "messages": [
            HumanMessage(content="退款政策是什么？"),
            AIMessage(content="正常图回答"),
        ],
        "scope": "in_scope",
        "action": "retrieve",
        "evidence_status": "not_found",
    }
    graph = Mock(return_value=target_result)
    monkeypatch.setattr(get_chat_service().engine.graph, "invoke", graph)

    response = back.interfaces.http.chat._execute_chat(
        ChatRequest(
            tenant_id="default",
            session_id="cache-fallback",
            message="退款政策是什么？",
        )
    )
    assert response.answer == "正常图回答"
    graph.assert_called_once()


def test_cache_store_failure_does_not_lose_current_answer(monkeypatch):
    target = make_prepared(session_id="cache-store-failure")
    monkeypatch.setattr(
        semantic_cache,
        "store_semantic_answer",
        Mock(side_effect=RuntimeError("disk full")),
    )
    response = get_chat_service().finalize(target, make_result(target, answer="仍然返回答案"))
    assert response.answer == "仍然返回答案"


def test_image_or_explicitly_disabled_flow_never_looks_up_cache(monkeypatch):
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", "default")
    finder = Mock(return_value=SemanticCacheHit("不应使用", 1.0, True))
    monkeypatch.setattr(semantic_cache, "find_semantic_answer", finder)
    graph = Mock(
        return_value={
            "tenant_id": "default",
            "messages": [HumanMessage(content="图片内容"), AIMessage(content="图片分析答案")],
        }
    )
    monkeypatch.setattr(get_chat_service().engine.graph, "invoke", graph)
    response = back.interfaces.http.chat._execute_chat(
        InternalChatRequest("图片内容", "image-session", "default"),
        allow_semantic_cache=False,
    )
    assert response.answer == "图片分析答案"
    finder.assert_not_called()


def test_profile_update_endpoint_invalidates_tenant_cache(monkeypatch):
    invalidator = Mock(return_value=3)
    monkeypatch.setattr(semantic_cache, "store_semantic_answer", Mock())
    monkeypatch.setattr("back.knowledge.retrieval.cache.invalidate_semantic_cache", invalidator)
    response = client.put(
        "/admin/profile",
        headers={"X-Admin-Key": "default-admin-key"},
        json={"tenant_id": "default", "business_hours": "09:00-18:00"},
    )
    assert response.status_code == 200
    invalidator.assert_called_once_with("default")


def test_failed_profile_update_does_not_invalidate_cache(monkeypatch):
    invalidator = Mock()
    monkeypatch.setattr("back.knowledge.retrieval.cache.invalidate_semantic_cache", invalidator)
    response = client.put(
        "/admin/profile",
        headers={"X-Admin-Key": "master-admin-key"},
        json={"tenant_id": "missing_tenant", "business_hours": "全天"},
    )
    assert response.status_code == 404
    invalidator.assert_not_called()


def test_successful_knowledge_index_invalidates_cache(monkeypatch, tmp_path):
    source = KnowledgeSource(
        source_id="src_cache_index",
        tenant_id="default",
        original_filename="cache.txt",
        stored_filename="cache.txt",
        file_type="txt",
        content_hash="hash",
        status="processing",
        chunk_count=0,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    updated = source.model_copy(update={"status": "ready", "chunk_count": 1})
    monkeypatch.setattr(
        back.knowledge.indexing.service,
        "load_source_document",
        Mock(return_value=[Document(page_content="退款规则")]),
    )
    monkeypatch.setattr(
        back.knowledge.indexing.service,
        "split_documents",
        Mock(return_value=[Document(page_content="退款规则")]),
    )
    monkeypatch.setattr(back.knowledge.indexing.service, "add_source_documents", Mock(return_value=1))
    status = Mock(side_effect=[source, updated])
    monkeypatch.setattr(back.knowledge.indexing.service, "update_knowledge_source_status", status)
    invalidator = Mock(return_value=2)
    monkeypatch.setattr(back.knowledge.indexing.service, "invalidate_semantic_cache", invalidator)

    result = back.knowledge.indexing.service.index_knowledge_source(source, tmp_path / "cache.txt")
    assert result.status == "ready"
    invalidator.assert_called_once_with("default")


def test_knowledge_delete_endpoint_invalidates_cache(monkeypatch):
    source = KnowledgeSource(
        source_id="src_cache_delete",
        tenant_id="default",
        original_filename="cache.txt",
        stored_filename="cache.txt",
        file_type="txt",
        content_hash="hash",
        status="ready",
        chunk_count=1,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    monkeypatch.setattr(get_knowledge_service().repository, "get", Mock(return_value=source))
    monkeypatch.setattr("back.infrastructure.search.opensearch.delete_source_documents", Mock(return_value=1))
    monkeypatch.setattr(get_knowledge_service().repository, "delete", Mock(return_value=True))
    invalidator = Mock(return_value=1)
    monkeypatch.setattr("back.knowledge.retrieval.cache.invalidate_semantic_cache", invalidator)

    response = client.delete(
        "/admin/knowledge/files/src_cache_delete?tenant_id=default",
        headers={"X-Admin-Key": "default-admin-key"},
    )
    assert response.status_code == 200
    invalidator.assert_called_once_with("default")
