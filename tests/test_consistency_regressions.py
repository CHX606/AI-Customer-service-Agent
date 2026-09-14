"""知识删除、并发变更、缓存版本以及对话重生的真实仓储回归。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
import pytest

from back.bootstrap import get_chat_service, get_knowledge_service, get_profile_service
from back.domain.chat import ChatCommand
from back.domain.errors import Conflict, DependencyUnavailable
from back.domain.tenant import TenantProfileUpdate
from back.infrastructure.operation_locks import source_operation
from back.infrastructure.search.opensearch import OpenSearchKnowledgeStore
from back.interfaces.http.app import app
from back.knowledge.indexing import service as indexing
from back.knowledge.retrieval import cache
from back.tenant.service import get_all_knowledge_sources, update_knowledge_source_status


def fake_engine(monkeypatch):
    service = get_chat_service()
    def answer(state):
        return {**state, "messages": [*state["messages"], AIMessage(content="答案")],
                "scope": "in_scope", "action": "retrieve", "evidence_status": "sufficient",
                "active_issue": {"summary": "本轮问题", "status": "answered"}}
    invoke = Mock(side_effect=answer)
    monkeypatch.setattr(service.engine, "invoke", invoke)
    return service, invoke


@pytest.mark.parametrize("partial", [
    {"deleted": 0, "version_conflicts": 1},
    {"deleted": 1, "timed_out": True},
    {"deleted": 1, "failures": [{"reason": "search failure"}]},
])
def test_partial_search_delete_keeps_source_file_and_registry(monkeypatch, partial):
    service = get_knowledge_service()
    source = next(s for s in service.list("default") if s.file_type == "md")
    path = service.files.path(source)
    original = path.read_bytes()
    store = object.__new__(OpenSearchKnowledgeStore)
    store.tenant_id, store._ready = "default", True
    store.settings = SimpleNamespace(index_alias="test", request_timeout=1)
    store.client = Mock()
    store.client.delete_by_query.return_value = partial
    monkeypatch.setattr(service.indexer, "delete", lambda tenant, source_id: store.delete_source(source_id))
    with pytest.raises(DependencyUnavailable):
        service.delete("default", source.source_id)
    assert service.repository.get(source.source_id) is not None
    assert path.read_bytes() == original
    # 后续重试能够完成清理。
    store.client.delete_by_query.return_value = {"deleted": 1, "version_conflicts": 0}
    service.delete("default", source.source_id)
    assert service.repository.get(source.source_id) is None
    assert not path.exists()


def test_reindex_and_delete_cannot_publish_ghost_documents(monkeypatch):
    service = get_knowledge_service()
    source = next(s for s in service.list("default") if s.file_type == "md")
    loaded, resume = Event(), Event()
    documents = {source.source_id: ["old"]}
    def load(path, **metadata):
        loaded.set()
        assert resume.wait(5)
        return [Document(page_content="new", metadata=metadata)]
    def add(tenant_id, source_id, documents):
        indexed[source_id] = documents
    indexed = documents
    monkeypatch.setattr(indexing, "load_source_document", load)
    monkeypatch.setattr(indexing, "split_documents", lambda docs: docs)
    monkeypatch.setattr(indexing, "add_source_documents", add)
    monkeypatch.setattr(service.indexer, "delete", lambda tenant, source_id: len(indexed.pop(source_id, [])))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(service.reindex, "default", source.source_id)
        assert loaded.wait(5)
        try:
            with pytest.raises(Conflict):
                service.delete("default", source.source_id)
            assert service.repository.get(source.source_id) is not None
        finally:
            resume.set()
        assert future.result(timeout=5).status == "ready"
    service.delete("default", source.source_id)
    assert source.source_id not in indexed
    assert service.repository.get(source.source_id) is None
    # 已经拿到旧 source 对象的直接调用同样不能复活已删除资料。
    with pytest.raises(Exception, match="已删除"):
        indexing.index_knowledge_source(source, service.files.path(source))
    assert source.source_id not in indexed


def test_source_lock_excludes_another_process_and_releases(tmp_path):
    from back.infrastructure.persistence import tenants
    code = """
import sys
from pathlib import Path
from back.infrastructure.persistence import tenants
from back.infrastructure.operation_locks import source_operation
from back.domain.errors import Conflict
tenants.DEFAULT_DB_PATH = Path(sys.argv[1])
try:
    with source_operation('default', 'test-source'):
        print('acquired')
except Conflict:
    print('busy')
"""
    def probe():
        run = subprocess.run([sys.executable, "-B", "-c", code, str(tenants.DEFAULT_DB_PATH)],
                             capture_output=True, text=True, timeout=10)
        assert run.returncode == 0, run.stderr
        return run.stdout.strip()
    with source_operation("default", "test-source"):
        # 同线程允许索引入口的嵌套加锁。
        with source_operation("default", "test-source"):
            assert probe() == "busy"
    assert probe() == "acquired"


def enable_cache(monkeypatch):
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "1")
    monkeypatch.setattr(cache, "get_embedding_model", lambda: SimpleNamespace(embed_query=lambda _: [1.0, 0.0]))


def test_in_flight_answer_cannot_repopulate_cache_after_profile_change(monkeypatch):
    enable_cache(monkeypatch)
    service, invoke = fake_engine(monkeypatch)
    prepared = service.prepare(ChatCommand("营业时间是什么？", "inflight"))
    old_result = invoke(prepared.graph_input)
    get_profile_service().update(TenantProfileUpdate(tenant_id="default", business_hours="新时间"))
    assert service.finalize(prepared, old_result).answer == "答案"
    assert cache.find_semantic_answer("default", "营业时间是什么？") is None
    service.execute(ChatCommand("营业时间是什么？", "new-session"))
    assert cache.find_semantic_answer("default", "营业时间是什么？").answer == "答案"


def test_chat_during_mutation_does_not_cache_partial_knowledge(monkeypatch):
    enable_cache(monkeypatch)
    service, invoke = fake_engine(monkeypatch)
    entered, leave = Event(), Event()
    def mutate():
        with cache.semantic_cache_mutation("default"):
            entered.set()
            assert leave.wait(5)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(mutate)
        assert entered.wait(5)
        try:
            prepared = service.prepare(ChatCommand("退款政策是什么？", "during-update"))
            assert prepared.cache_revision is None
            assert service.find_cached(prepared) is None
        finally:
            leave.set()
        future.result(timeout=5)
    service.finalize(prepared, invoke(prepared.graph_input))
    assert cache.count_semantic_cache_entries("default") == 0


def test_cache_version_change_during_embedding_prevents_write(monkeypatch):
    enable_cache(monkeypatch)
    revision = cache.get_cache_revision("default")
    def embed(_):
        cache.invalidate_semantic_cache("default")
        return [1.0, 0.0]
    monkeypatch.setattr(cache, "get_embedding_model", lambda: SimpleNamespace(embed_query=embed))
    assert not cache.store_semantic_answer("default", "退款政策是什么？", "旧答案", expected_revision=revision)
    assert cache.count_semantic_cache_entries("default") == 0


def test_regenerate_replaces_turn_and_restores_context_before_it(monkeypatch):
    service, invoke = fake_engine(monkeypatch)
    service.execute(ChatCommand("第一问", "s", turn_id="u1"))
    service.execute(ChatCommand("第二问", "s", turn_id="u2"))
    service.execute(ChatCommand("第三问", "s", turn_id="u3"))
    service.execute(ChatCommand("第二问", "s", turn_id="u2", regenerate=True))
    saved = service.sessions.load(("default", "s"))
    assert [m.content for m in saved.state["messages"]] == ["第一问", "答案", "第二问", "答案"]
    assert invoke.call_args.args[0]["active_issue"] == {"summary": "本轮问题", "status": "answered"}
    service.execute(ChatCommand("第一问", "s", turn_id="u1", regenerate=True))
    assert invoke.call_args.args[0]["active_issue"] is None
    assert len(service.sessions.load(("default", "s")).state["messages"]) == 2


def test_legacy_append_and_repeated_text_remain_independent_turns(monkeypatch):
    service, _ = fake_engine(monkeypatch)
    for turn_id in (None, None, "one", "two"):
        service.execute(ChatCommand("同样的问题", "legacy", turn_id=turn_id))
    assert len(service.sessions.load(("default", "legacy")).state["messages"]) == 8


def test_same_turn_retry_after_lost_response_does_not_append(monkeypatch):
    service, _ = fake_engine(monkeypatch)
    for _ in range(2):
        service.execute(ChatCommand("问题", "retry", turn_id="same"))
    assert len(service.sessions.load(("default", "retry")).state["messages"]) == 2


def test_old_session_regeneration_uses_unique_answer_anchor(monkeypatch):
    service, _ = fake_engine(monkeypatch)
    service.execute(ChatCommand("旧问题", "old"))
    digest = hashlib.sha256("答案".encode()).hexdigest()
    service.execute(ChatCommand("旧问题", "old", turn_id="new-id", regenerate=True, previous_answer_hash=digest))
    history = service.sessions.load(("default", "old")).state["messages"]
    assert len(history) == 2 and history[0].id == "new-id"


def test_missing_or_ambiguous_regeneration_does_not_change_history(monkeypatch):
    service, _ = fake_engine(monkeypatch)
    service.execute(ChatCommand("第一问", "ambiguous"))
    service.execute(ChatCommand("第二问", "ambiguous"))
    before = service.sessions.load(("default", "ambiguous"))
    with pytest.raises(Conflict):
        service.execute(ChatCommand("问题", "ambiguous", turn_id="unknown", regenerate=True,
                                    previous_answer_hash=hashlib.sha256("答案".encode()).hexdigest()))
    assert service.sessions.load(("default", "ambiguous")).version == before.version


def test_image_regeneration_reuses_saved_description_without_image(monkeypatch):
    service, _ = fake_engine(monkeypatch)
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", "default")
    service.execute(ChatCommand("图片识别出的错误及上下文", "image", turn_id="image-turn"), allow_semantic_cache=False)
    result = TestClient(app).post("/chat", json={"message": "[已上传图片]", "session_id": "image",
                                 "turn_id": "image-turn", "regenerate": True})
    assert result.status_code == 200
    history = service.sessions.load(("default", "image")).state["messages"]
    assert len(history) == 2 and history[0].content == "图片识别出的错误及上下文"


def test_failed_regeneration_leaves_original_answer_intact(monkeypatch):
    service, _ = fake_engine(monkeypatch)
    service.execute(ChatCommand("问题", "failure", turn_id="turn"))
    before = service.sessions.load(("default", "failure"))
    monkeypatch.setattr(service.engine, "invoke", Mock(side_effect=RuntimeError("unavailable")))
    with pytest.raises(DependencyUnavailable):
        service.execute(ChatCommand("问题", "failure", turn_id="turn", regenerate=True))
    assert service.sessions.load(("default", "failure")).version == before.version


def test_stream_regeneration_carries_turn_identity_and_replaces_history(monkeypatch):
    service, invoke = fake_engine(monkeypatch)
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", "default")
    service.execute(ChatCommand("问题", "stream", turn_id="stream-turn"))
    def stream(state):
        yield {"type": "state", "state": invoke(state)}
    monkeypatch.setattr(service.engine, "stream", stream)
    response = TestClient(app).post("/chat/stream", json={"message": "问题", "session_id": "stream",
                                    "turn_id": "stream-turn", "regenerate": True})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]["type"] == "final"
    assert len(service.sessions.load(("default", "stream")).state["messages"]) == 2
